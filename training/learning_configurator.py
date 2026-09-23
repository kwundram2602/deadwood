import torch.nn as nn

from training.trainable_report import (
    effective_counts,
    frozen_columns,
    masked_convs,
    module_rows,
)


def _status(trainable: int, total: int) -> str:
    if trainable == 0:
        return "FROZEN"
    if trainable == total:
        return "TRAINABLE"
    return "PARTIAL"


class LearningConfigurator:
    """Freeze/unfreeze TorchGeo UNet layers for transfer learning phases."""

    def prepare_model_for_transfer_learning(self, model: nn.Module) -> nn.Module:
        print("Transfer learning: freezing encoder, training first conv + decoder + head")
        self._freeze_encoder(model)
        self._unfreeze_first_conv(model)
        self._set_trainable(model, "decoder", True)
        self._set_trainable(model, "segmentation_head", True)
        self._print_trainable_table(model)
        return model

    def prepare_model_for_fine_tuning(
        self, model: nn.Module, unfreeze_keys: list[str]
    ) -> nn.Module:
        print(f"Fine-tuning: unfreezing encoder submodules {list(unfreeze_keys)}")
        self._freeze_encoder(model)
        self._unfreeze_first_conv(model)
        self._unfreeze_keys(model, list(unfreeze_keys))
        self._set_trainable(model, "decoder", True)
        self._set_trainable(model, "segmentation_head", True)
        self._print_trainable_table(model)
        return model

    def prepare_model_for_head_only(self, model: nn.Module) -> nn.Module:
        print("Head only: freezing encoder and decoder, training segmentation head")
        self._freeze_encoder(model)
        self._set_trainable(model, "decoder", False)
        self._set_trainable(model, "segmentation_head", True)
        self._print_trainable_table(model)
        return model

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _freeze_encoder(self, model: nn.Module) -> None:
        m = model.module if hasattr(model, "module") else model
        if hasattr(m, "encoder"):
            for p in m.encoder.parameters():
                p.requires_grad = False
        else:
            print("WARNING: model has no 'encoder' attribute")

    def _unfreeze_first_conv(self, model: nn.Module) -> None:
        """First conv (channel-frozen via gradient mask) + its BN train in every phase.

        Only ResNet-style encoders name these conv1/bn1. Transformer encoders
        such as mit_b5 call theirs patch_embed1.proj, so nothing matches — say
        so rather than returning silently, which reads as "first conv trained"
        when it was not.
        """
        m = model.module if hasattr(model, "module") else model
        if not hasattr(m, "encoder"):
            return
        found = False
        for attr in ("conv1", "bn1"):
            if hasattr(m.encoder, attr):
                found = True
                for p in getattr(m.encoder, attr).parameters():
                    p.requires_grad = True
        if not found:
            print(
                "  NOTE: encoder has no conv1/bn1 — first conv stays frozen. "
                "Transformer encoders name it patch_embed1.proj; unfreeze it via "
                "unfreeze_keys if that is what you want."
            )

    def _set_trainable(self, model: nn.Module, attr: str, trainable: bool) -> None:
        m = model.module if hasattr(model, "module") else model
        if hasattr(m, attr):
            for p in getattr(m, attr).parameters():
                p.requires_grad = trainable
        else:
            print(f"WARNING: model has no '{attr}' attribute")

    def _unfreeze_keys(self, model: nn.Module, keys: list[str]) -> None:
        m = model.module if hasattr(model, "module") else model
        if not hasattr(m, "encoder"):
            print("WARNING: model has no 'encoder' attribute")
            return
        if not keys:
            raise ValueError(
                "unfreeze_keys is empty — fine-tuning would unfreeze no encoder "
                "submodules. Valid keys:\n" + self._format_valid_keys(m.encoder)
            )
        for key in keys:
            if not key:
                raise ValueError(
                    "unfreeze_keys entries must be non-empty encoder paths. "
                    "Valid keys:\n" + self._format_valid_keys(m.encoder)
                )
            try:
                submodule = m.encoder.get_submodule(key)
            except AttributeError:
                raise ValueError(
                    f"unfreeze_keys entry '{key}' not found in encoder. "
                    "Valid keys:\n" + self._format_valid_keys(m.encoder)
                ) from None
            for p in submodule.parameters():
                p.requires_grad = True
            print(f"  Unfroze encoder.{key}")

    def _format_valid_keys(self, encoder: nn.Module) -> str:
        """List encoder keys down to sub-block depth, with param counts."""
        lines: list[str] = []
        for name, child in encoder.named_children():
            total = sum(p.numel() for p in child.parameters())
            if total == 0:
                continue
            lines.append(f"  {name:<12} {total:>13,} params")
            for sub_name, sub in child.named_children():
                if not sub_name.isdigit():
                    continue
                sub_total = sum(p.numel() for p in sub.parameters())
                lines.append(f"  {name}.{sub_name:<10} {sub_total:>13,} params")
        return "\n".join(lines)

    def _print_trainable_table(self, model: nn.Module) -> None:
        # Rows come from trainable_report so this table and the per-phase
        # figure always report the same modules and the same counts.
        rows = [
            (("  " if depth else "") + name, _status(tr, tot), tr, tot)
            for name, depth, tr, tot in module_rows(model)
        ]

        name_w = max(len(r[0]) for r in rows) + 2 if rows else 20
        print(f"  {'Module':<{name_w}}  {'Status':<10}  {'Trainable':>13} / {'Total':>13}")
        print(f"  {'-' * name_w}  {'-' * 10}  {'-' * 13}   {'-' * 13}")
        for name, st, tr, tot in rows:
            print(f"  {name:<{name_w}}  {st:<10}  {tr:>13,} / {tot:>13,}")

        total_tr, total_all = effective_counts(model)
        pct = 100 * total_tr / total_all if total_all else 0.0
        print(f"\n  Trainable params: {total_tr:,} / {total_all:,} ({pct:.1f}%)")

        # Without this a PARTIAL first-conv row looks like a bug in the table.
        for name, mask in masked_convs(model):
            cols = frozen_columns(mask)
            per_col = mask[:, 0].numel() if mask.ndim > 1 else 0
            print(
                f"  NOTE: {name} is gradient-masked on input channels {cols} — "
                f"{len(cols) * per_col:,} of {mask.numel():,} weights cannot move "
                "(requires_grad stays True; the mask zeroes their grads)."
            )


def print_model_key_tree(model: nn.Module) -> None:
    """Print every named module with its param count (debug: true in config)."""
    m = model.module if hasattr(model, "module") else model
    print("Model key tree (module: param count):")
    for name, module in m.named_modules():
        if not name:
            continue
        n_params = sum(p.numel() for p in module.parameters())
        indent = "  " * name.count(".")
        print(f"  {indent}{name}: {n_params:,}")

import torch.nn as nn


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
        """First conv (channel-frozen via gradient mask) + its BN train in every phase."""
        m = model.module if hasattr(model, "module") else model
        if not hasattr(m, "encoder"):
            return
        for attr in ("conv1", "bn1"):
            if hasattr(m.encoder, attr):
                for p in getattr(m.encoder, attr).parameters():
                    p.requires_grad = True

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
        m = model.module if hasattr(model, "module") else model
        rows: list[tuple[str, int, int]] = []

        if hasattr(m, "encoder"):
            enc = m.encoder
            found_blocks = False
            for bname in ("conv1", "bn1", "layer0", "layer1", "layer2", "layer3", "layer4"):
                if hasattr(enc, bname):
                    block = getattr(enc, bname)
                    tr = sum(p.numel() for p in block.parameters() if p.requires_grad)
                    tot = sum(p.numel() for p in block.parameters())
                    if tot > 0:
                        rows.append((f"encoder.{bname}", tr, tot))
                        found_blocks = True
            if not found_blocks:
                tr = sum(p.numel() for p in enc.parameters() if p.requires_grad)
                tot = sum(p.numel() for p in enc.parameters())
                rows.append(("encoder", tr, tot))

        for attr in ("decoder", "segmentation_head"):
            if hasattr(m, attr):
                mod = getattr(m, attr)
                tr = sum(p.numel() for p in mod.parameters() if p.requires_grad)
                tot = sum(p.numel() for p in mod.parameters())
                rows.append((attr, tr, tot))

        name_w = max(len(r[0]) for r in rows) + 2 if rows else 20
        print(f"  {'Module':<{name_w}}  {'Status':<10}  {'Trainable':>13} / {'Total':>13}")
        print(f"  {'-' * name_w}  {'-' * 10}  {'-' * 13}   {'-' * 13}")
        for name, tr, tot in rows:
            status = "TRAINABLE" if tr > 0 else "FROZEN"
            print(f"  {name:<{name_w}}  {status:<10}  {tr:>13,} / {tot:>13,}")

        total_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_all = sum(p.numel() for p in model.parameters())
        pct = 100 * total_tr / total_all if total_all else 0.0
        print(f"\n  Trainable params: {total_tr:,} / {total_all:,} ({pct:.1f}%)")

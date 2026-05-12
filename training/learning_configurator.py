import torch.nn as nn


class LearningConfigurator:
    """Freeze/unfreeze TorchGeo UNet layers for transfer learning phases."""

    def prepare_model_for_transfer_learning(self, model: nn.Module) -> nn.Module:
        print("Transfer learning: freezing encoder, training decoder + head")
        self._freeze_encoder(model)
        self._set_trainable(model, "decoder", True)
        self._set_trainable(model, "segmentation_head", True)
        self._print_trainable_table(model)
        return model

    def prepare_model_for_fine_tuning(
        self, model: nn.Module, unfreeze_blocks: int = 1
    ) -> nn.Module:
        print(f"Fine-tuning: unfreezing last {unfreeze_blocks} encoder block(s)")
        self._freeze_encoder(model)
        self._unfreeze_last_n_blocks(model, unfreeze_blocks)
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

    def _set_trainable(self, model: nn.Module, attr: str, trainable: bool) -> None:
        m = model.module if hasattr(model, "module") else model
        if hasattr(m, attr):
            for p in getattr(m, attr).parameters():
                p.requires_grad = trainable
        else:
            print(f"WARNING: model has no '{attr}' attribute")

    def _unfreeze_last_n_blocks(self, model: nn.Module, n: int) -> None:
        m = model.module if hasattr(model, "module") else model
        if not hasattr(m, "encoder"):
            return
        # ResNet-style blocks inside TorchGeo encoder
        candidate_blocks = ["layer4", "layer3", "layer2", "layer1"]
        unfrozen = 0
        for block_name in candidate_blocks:
            if unfrozen >= n:
                break
            if hasattr(m.encoder, block_name):
                for p in getattr(m.encoder, block_name).parameters():
                    p.requires_grad = True
                unfrozen += 1
                print(f"  Unfroze encoder.{block_name}")
        if unfrozen == 0:
            print("WARNING: no encoder blocks were unfrozen")

    def _print_trainable_table(self, model: nn.Module) -> None:
        m = model.module if hasattr(model, "module") else model
        rows: list[tuple[str, int, int]] = []

        if hasattr(m, "encoder"):
            enc = m.encoder
            found_blocks = False
            for bname in ("layer0", "layer1", "layer2", "layer3", "layer4"):
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

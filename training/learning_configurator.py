import torch.nn as nn


class LearningConfigurator:
    """Freeze/unfreeze TorchGeo UNet layers for transfer learning phases."""

    def prepare_model_for_transfer_learning(self, model: nn.Module) -> nn.Module:
        print("Transfer learning: freezing encoder, training decoder + head")
        self._freeze_encoder(model)
        self._set_trainable(model, "decoder", True)
        self._set_trainable(model, "segmentation_head", True)
        self._print_param_counts(model)
        return model

    def prepare_model_for_fine_tuning(
        self, model: nn.Module, unfreeze_blocks: int = 1
    ) -> nn.Module:
        print(f"Fine-tuning: unfreezing last {unfreeze_blocks} encoder block(s)")
        self._freeze_encoder(model)
        self._unfreeze_last_n_blocks(model, unfreeze_blocks)
        self._set_trainable(model, "decoder", True)
        self._set_trainable(model, "segmentation_head", True)
        self._print_param_counts(model)
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

    def _print_param_counts(self, model: nn.Module) -> None:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(
            f"Trainable params: {trainable:,} / {total:,} "
            f"({100 * trainable / total:.1f}%)"
        )

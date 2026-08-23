import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchgeo.models import Unet_Weights, unet

from data.channels import ChannelSpec

_IDENTITY_RGB = {0: 0, 1: 1, 2: 2}


def build_model(cfg: DictConfig, device: torch.device, spec: ChannelSpec) -> nn.Module:
    """Instantiate a TorchGeo UNet, adapt channels per spec, load weights, and move to device."""
    weights = None
    if cfg.model.weights_name is not None:
        weights = getattr(Unet_Weights, cfg.model.weights_name)

    # Build with pretrained 3-channel encoder, then adapt to the channel spec
    model = unet(weights=weights, classes=cfg.model.num_classes)

    if spec.in_channels != 3 or spec.pretrained_assignment != _IDENTITY_RGB:
        _adapt_first_conv(model.encoder, spec.in_channels, spec.pretrained_assignment)

    frozen_names = list(cfg.model.get("frozen_pretrained_channels") or [])
    if frozen_names:
        try:
            frozen = spec.frozen_indices(frozen_names)
        except ValueError as e:
            raise ValueError(f"frozen_pretrained_channels: {e}") from e
        _freeze_input_channels(model.encoder, frozen, spec.in_channels)

    if cfg.model.weights_path is not None:
        print(f"Loading checkpoint: {cfg.model.weights_path}")
        state = torch.load(cfg.model.weights_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)

    # Freeze all params — LearningConfigurator will selectively unfreeze
    for p in model.parameters():
        p.requires_grad = False

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    return model.to(device)


def _find_first_conv(encoder: nn.Module) -> tuple[str, nn.Conv2d]:
    """Return (qualified name, module) of the first Conv2d in the encoder."""
    for name, module in encoder.named_modules():
        if isinstance(module, nn.Conv2d):
            return name, module
    raise ValueError("Encoder contains no Conv2d")


def _freeze_input_channels(
    encoder: nn.Module, channels: list[int], in_channels: int
) -> None:
    """Keep the given input-channel slices of the first conv frozen.

    Registers a gradient hook that zeroes the grads of those slices on every
    backward pass; the forward pass is untouched. The optimizer must put this
    weight in a weight_decay=0 group, otherwise AdamW still shrinks the
    frozen slices via decoupled weight decay.
    During backward pass the hook multiplies the gradient by a mask that is 0 for frozen channels and 1 for others. This means that the frozen channels will not be updated during training, effectively freezing them.
    """
    bad = [c for c in channels if not 0 <= c < in_channels]
    if bad:
        raise ValueError(
            f"frozen_pretrained_channels {bad} out of range for in_channels={in_channels}"
        )

    name, conv = _find_first_conv(encoder)
    # for every input channel set to 0 if it is supposed to be frozen
    mask = torch.ones_like(conv.weight)
    mask[:, channels] = 0.0
    # non-persistent buffer: moves with .to(device), stays out of the state_dict
    conv.register_buffer("frozen_channel_mask", mask, persistent=False)
    conv.weight.register_hook(lambda g: g * conv.frozen_channel_mask)
    print(f"Channel-freeze on encoder.{name}: input channels {channels} masked")


def _adapt_first_conv(
    encoder: nn.Module, to_ch: int, assignment: dict[int, int]
) -> None:
    """Replace the first Conv2d so it accepts to_ch input channels.

    assignment maps input position → pretrained slot column (0=R, 1=G, 2=B);
    those columns are copied from the pretrained conv, all other input
    positions are kaiming-initialised.
    Old encoder remains but with changed first conv.
    """
    name, old = _find_first_conv(encoder)
    new = nn.Conv2d(
        to_ch,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        padding_mode=old.padding_mode,
        dilation=old.dilation,
        groups=old.groups,
        bias=old.bias is not None,
    )
    with torch.no_grad():
        nn.init.kaiming_normal_(new.weight, mode="fan_out", nonlinearity="relu")
        for pos, slot in assignment.items():
            new.weight[:, pos] = old.weight[:, slot]
        if old.bias is not None:
            new.bias.copy_(old.bias)

    parent = encoder
    parts = name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new)
    print(
        f"Adapted encoder.{name}: 3 → {to_ch} input channels, "
        f"pretrained slot→position: { {v: k for k, v in assignment.items()} }"
    )

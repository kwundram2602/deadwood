"""What is *actually* trainable in a model, and a figure that shows it.

``requires_grad`` alone lies about the first conv. The channel freeze in
models.model registers a gradient hook that zeroes whole input-channel
slices of ``conv1.weight`` while leaving ``requires_grad`` True, so a naive
count reports the conv as fully trainable when a third of its weights can
never move. Everything here reads the ``frozen_channel_mask`` buffer that
freeze leaves behind, so the counts and the plot agree with what the
optimizer can reach.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from matplotlib.patches import Patch, Rectangle

# Buffer name models.model._freeze_input_channels registers on the first conv.
MASK_BUFFER = "frozen_channel_mask"


def _mask_overrides(mod: nn.Module) -> dict[int, int]:
    """{id(param): effectively trainable numel} for gradient-masked weights."""
    overrides: dict[int, int] = {}
    for sub in mod.modules():
        mask = sub._buffers.get(MASK_BUFFER)
        if mask is None:
            continue
        weight = getattr(sub, "weight", None)
        if weight is None:
            continue
        overrides[id(weight)] = int(mask.count_nonzero())
    return overrides


def effective_counts(mod: nn.Module) -> tuple[int, int]:
    """(trainable, total) parameter counts, honouring gradient masks."""
    overrides = _mask_overrides(mod)
    trainable = 0
    total = 0
    for p in mod.parameters():
        total += p.numel()
        if not p.requires_grad:
            continue
        trainable += overrides.get(id(p), p.numel())
    return trainable, total


def unwrap(model: nn.Module) -> nn.Module:
    """The model itself, or what DataParallel wrapped."""
    inner = getattr(model, "module", None)
    return inner if isinstance(inner, nn.Module) else model


def masked_convs(model: nn.Module) -> list[tuple[str, torch.Tensor]]:
    """(qualified name, mask) for every gradient-masked conv, outermost first."""
    found = []
    m = unwrap(model)
    for name, sub in m.named_modules():
        mask = sub._buffers.get(MASK_BUFFER)
        if mask is not None and getattr(sub, "weight", None) is not None:
            found.append((name, mask))
    return found


def frozen_columns(mask: torch.Tensor) -> list[int]:
    """Input-channel positions this mask zeroes out completely."""
    other_dims = [d for d in range(mask.ndim) if d != 1]
    per_column = mask.amax(dim=other_dims) if other_dims else mask
    return [i for i, keeps_grad in enumerate(per_column.tolist()) if keeps_grad == 0]


def _child(mod: nn.Module, name: str) -> nn.Module | None:
    """Named child submodule, or None — nn.Module.__getattr__ also yields tensors."""
    sub = getattr(mod, name, None)
    return sub if isinstance(sub, nn.Module) else None


def module_rows(model: nn.Module) -> list[tuple[str, int, int, int]]:
    """(label, depth, trainable, total) per reported module, in print order.

    One owner for the row set so the printed table and the figure cannot
    drift apart. Depth 1 rows are the numbered sub-blocks of a depth 0 block.
    """
    m = unwrap(model)
    rows: list[tuple[str, int, int, int]] = []

    encoder = _child(m, "encoder")
    if encoder is not None:
        found_blocks = False
        for bname, block in encoder.named_children():
            tr, tot = effective_counts(block)
            if tot == 0:
                continue
            found_blocks = True
            rows.append((f"encoder.{bname}", 0, tr, tot))
            for sub_name, sub in block.named_children():
                if not sub_name.isdigit():
                    continue
                sub_tr, sub_tot = effective_counts(sub)
                if sub_tot == 0:
                    continue
                rows.append((f"{bname}.{sub_name}", 1, sub_tr, sub_tot))
        if not found_blocks:
            rows.append(("encoder", 0, *effective_counts(encoder)))

    for attr in ("decoder", "segmentation_head"):
        sub = _child(m, attr)
        if sub is not None:
            rows.append((attr, 0, *effective_counts(sub)))

    return rows


# Trainable blue / frozen grey: distinguishable for colour-blind readers, and
# the hatch keeps them apart in greyscale print.
_TRAINABLE_C = "#0173B2"
_FROZEN_C = "#DCDCDC"
_FROZEN_EDGE = "#9A9A9A"

_PAPER_RC = {
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.frameon": False,
    # Keep SVG text as text, so the figure stays editable in Inkscape.
    "svg.fonttype": "none",
}


def _draw_module_panel(ax, rows: list[tuple[str, int, int, int]]) -> None:
    """Trainable fraction per module, absolute counts annotated at the bar end.

    Fractions rather than raw counts: the rows span 15k to 13M parameters, so
    a shared linear axis renders the first conv invisible and a log axis is
    hard to read off a printed page.
    """
    labels = [("   " if depth else "") + name for name, depth, _, _ in rows]
    fractions = [tr / tot if tot else 0.0 for _, _, tr, tot in rows]
    y = range(len(rows))

    ax.barh(y, fractions, color=_TRAINABLE_C, height=0.72, zorder=3)
    ax.barh(
        y,
        [1 - f for f in fractions],
        left=fractions,
        color=_FROZEN_C,
        edgecolor=_FROZEN_EDGE,
        hatch="///",
        linewidth=0.4,
        height=0.72,
        zorder=2,
    )
    for i, (_, _, tr, tot) in enumerate(rows):
        ax.text(1.02, i, f"{tr:,} / {tot:,}", va="center", ha="left", fontsize=6.5)

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25", "50", "75", "100"])
    ax.set_xlabel("trainable parameters (% of module)")
    ax.set_title("Per-module trainability", loc="left")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=0.25, zorder=0)


def _draw_conv_panel(ax, model: nn.Module, names: list[str] | None) -> None:
    """The first conv's input columns — where the gradient mask actually bites."""
    masked = masked_convs(model)
    ax.set_axis_off()
    if not masked:
        ax.text(
            0.0,
            0.5,
            "No gradient mask on any conv — the first conv is either trained "
            "in full or frozen outright (see the panel above).",
            va="center",
            ha="left",
            fontsize=7.5,
            style="italic",
            color="#555555",
        )
        return

    name, mask = masked[0]
    n_cols = mask.shape[1]
    frozen = set(frozen_columns(mask))
    per_col = mask[:, 0].numel()

    labels = [names[c] if names and c < len(names) else f"ch {c}" for c in range(n_cols)]
    rotation = 30 if max(len(lb) for lb in labels) > 8 else 0

    for col in range(n_cols):
        is_frozen = col in frozen
        ax.add_patch(
            Rectangle(
                (col + 0.05, 0.05),
                0.9,
                0.9,
                facecolor=_FROZEN_C if is_frozen else _TRAINABLE_C,
                edgecolor=_FROZEN_EDGE if is_frozen else _TRAINABLE_C,
                hatch="///" if is_frozen else None,
                linewidth=0.6,
            )
        )
        ax.text(
            col + 0.5,
            -0.16,
            labels[col],
            ha="center" if rotation == 0 else "right",
            va="top",
            fontsize=7,
            rotation=rotation,
            rotation_mode=None if rotation == 0 else "anchor",
        )
        ax.text(
            col + 0.5,
            0.5,
            "frozen" if is_frozen else "trained",
            ha="center",
            va="center",
            fontsize=6.5,
            color="#555555" if is_frozen else "white",
        )

    ax.set_xlim(-0.1, n_cols + 0.1)
    ax.set_ylim(-1.0, 1.25)
    ax.set_title(
        f"{name} input columns — {per_col:,} weights each, "
        f"{len(frozen) * per_col:,} of {mask.numel():,} masked",
        loc="left",
    )


def plot_trainable(
    model: nn.Module,
    out_dir: Path | str,
    phase: str,
    spec=None,
) -> list[Path]:
    """Write trainable_<phase>.png/.svg showing what this phase can actually train.

    spec: optional ChannelSpec, used only to name the first conv's input
    columns; without it they are labelled by index.
    """
    rows = module_rows(model)
    if not rows:
        raise ValueError("model exposes no encoder/decoder/segmentation_head to report")

    names = list(spec.input_channels) if spec is not None else None
    total_tr, total_all = effective_counts(model)
    pct = 100 * total_tr / total_all if total_all else 0.0

    with plt.rc_context(_PAPER_RC):
        height = 1.6 + 0.22 * len(rows)
        fig, (ax_mod, ax_conv) = plt.subplots(
            2,
            1,
            figsize=(7.0, height),
            height_ratios=[0.22 * len(rows), 1.3],
            constrained_layout=True,
        )
        _draw_module_panel(ax_mod, rows)
        _draw_conv_panel(ax_conv, model, names)

        handles = [
            Patch(facecolor=_TRAINABLE_C, label="trainable"),
            Patch(
                facecolor=_FROZEN_C,
                edgecolor=_FROZEN_EDGE,
                hatch="///",
                linewidth=0.4,
                label="frozen (requires_grad or gradient mask)",
            ),
        ]
        fig.legend(handles=handles, loc="lower right", ncols=2, bbox_to_anchor=(1.0, -0.02))
        fig.suptitle(
            f"Phase '{phase}' — {total_tr:,} of {total_all:,} parameters trainable ({pct:.1f}%)",
            ha="left",
            x=0.01,
            fontsize=10,
        )

        # dpi only bites on the raster output; the SVG ignores it.
        written = []
        for suffix in (".png", ".svg"):
            path = Path(out_dir) / f"trainable_{phase}{suffix}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            written.append(path)
        plt.close(fig)

    print(f"  Trainability figure -> {written[0]} (+ .svg)")
    return written

# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "matplotlib>=3.10",
#     "numpy>=2.4",
# ]
# ///
"""Interactive view of the non-ground mask ramp used in apply_dsm_mask.py.

Run with:  uvx marimo edit --sandbox assets/mask_function.py
"""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Mask function — smoothstep ramp

    `_smoothstep_confidence` in `explore_and_process/apply_dsm_mask.py`
    returns the *ground* confidence. The mask is its complement, and the
    double negation cancels:

    $$
    t = \mathrm{clip}\!\left(\frac{nDSM - h_t}{r},\, 0,\, 1\right)
    \qquad
    \text{mask} = 1 - \text{ground\_conf} = 3t^2 - 2t^3
    $$

    So the mask *is* the smoothstep curve: 0 below the height threshold
    $h_t$, 1 above $h_t + r$, and a C1-continuous S-curve in between.
    """)
    return


@app.cell
def _(mo):
    threshold = mo.ui.slider(0.1, 3.0, step=0.05, value=0.7, label="height_threshold $h_t$ (m)")
    ramp = mo.ui.slider(0.1, 3.0, step=0.05, value=0.7, label="height_ramp $r$ (m)")
    hard = mo.ui.checkbox(value=True, label="show hard threshold for comparison")
    mo.vstack([threshold, ramp, hard])
    return hard, ramp, threshold


@app.cell
def _(np):
    def ramp_t(ndsm, threshold, ramp):
        """Normalised ramp coordinate: 0 at the threshold, 1 at threshold + ramp."""
        return np.clip((ndsm - threshold) / ramp, 0.0, 1.0)

    def smoothstep_mask(ndsm, threshold, ramp):
        """Non-ground mask: 0 below threshold, 1 above threshold + ramp."""
        return 3 * ramp_t(ndsm, threshold, ramp) ** 2 - 2 * ramp_t(ndsm, threshold, ramp) ** 3

    return ramp_t, smoothstep_mask


@app.cell
def _():
    INK = "#2d3142"
    MUTED = "#4f5d75"
    ACCENT = "#eb6c36"
    SLATE = "#4f5d75"
    return ACCENT, INK, MUTED, SLATE


@app.cell
def _(ACCENT, INK, MUTED, hard, np, plt, ramp, smoothstep_mask, threshold):
    ht = threshold.value
    r = ramp.value

    x = np.linspace(0.0, max(4.0, ht + r + 1.0), 1000)
    y = smoothstep_mask(x, ht, r)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(x, y, lw=2.0, color=ACCENT, label="smoothstep mask  $3t^2-2t^3$", zorder=3)

    if hard.value:
        ax.plot(
            x,
            (x >= ht).astype(float),
            lw=1.5,
            ls="--",
            color=MUTED,
            label=f"hard threshold  $nDSM \\geq {ht:.2f}$",
            zorder=2,
        )

    # ramp interval: where the transition actually happens
    ax.axvspan(ht, ht + r, color=ACCENT, alpha=0.07, lw=0, zorder=0)
    for xv, lbl in ((ht, "$h_t$"), (ht + r, "$h_t + r$")):
        ax.axvline(xv, color=MUTED, lw=0.8, ls=":", zorder=1)
        ax.annotate(
            f"{lbl}\n{xv:.2f} m",
            (xv, 1.06),
            ha="center",
            va="bottom",
            fontsize=8,
            color=MUTED,
        )

    # midpoint: maximum uncertainty
    ax.plot([ht + r / 2], [0.5], "o", ms=6, color=ACCENT, zorder=4)
    ax.annotate(
        f"0.5 @ {ht + r / 2:.2f} m",
        (ht + r / 2, 0.5),
        xytext=(8, -12),
        textcoords="offset points",
        fontsize=8,
        color=INK,
    )

    ax.set_xlabel("nDSM — height above ground (m)")
    ax.set_ylabel("mask value  (1 = certainly crown)")
    ax.set_ylim(-0.05, 1.25)
    ax.set_xlim(x[0], x[-1])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.grid(axis="y", color=MUTED, alpha=0.15, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.legend(frameon=False, loc="center right", fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig
    return ht, r


@app.cell
def _(ACCENT, MUTED, SLATE, ht, np, plt, r, ramp_t, smoothstep_mask):
    def _():
        xt = np.linspace(0.0, max(4.0, ht + r + 1.0), 1000)

        fig_t, ax_t = plt.subplots(figsize=(7.5, 3.4))
        ax_t.plot(xt, ramp_t(xt, ht, r), lw=2.0, color=SLATE, label="$t$  (clipped linear ramp)")
        ax_t.plot(
            xt,
            smoothstep_mask(xt, ht, r),
            lw=1.5,
            ls="--",
            color=ACCENT,
            alpha=0.8,
            label="$3t^2-2t^3$  (mask)",
        )

        ax_t.axvspan(ht, ht + r, color=ACCENT, alpha=0.07, lw=0, zorder=0)
        for xv in (ht, ht + r):
            ax_t.axvline(xv, color=MUTED, lw=0.8, ls=":", zorder=1)

        # the two kinks clip() introduces — smoothstep rounds exactly these off
        ax_t.plot([ht, ht + r], [0.0, 1.0], "o", ms=6, mfc="none", mew=1.6, color=SLATE, zorder=4)
        ax_t.text(
            0.02,
            0.99,
            "circles: the kinks clip() leaves — $t$ is only C0 there,\nsmoothstep rounds exactly those off",
            transform=ax_t.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color=MUTED,
        )

        ax_t.set_xlabel("nDSM — height above ground (m)")
        ax_t.set_ylabel("$t$")
        ax_t.set_ylim(-0.08, 1.35)
        ax_t.set_xlim(xt[0], xt[-1])
        ax_t.set_yticks([0.0, 0.5, 1.0])
        ax_t.grid(axis="y", color=MUTED, alpha=0.15, lw=0.6)
        for side_t in ("top", "right"):
            ax_t.spines[side_t].set_visible(False)
        for side_t in ("left", "bottom"):
            ax_t.spines[side_t].set_color(MUTED)
        ax_t.tick_params(colors=MUTED, labelsize=9)
        ax_t.legend(frameon=False, loc="center right", fontsize=9)
        fig_t.tight_layout()
        return fig_t


    _()
    return


@app.cell
def _(ht, mo, np, r, smoothstep_mask):
    probes = np.array([0.0, ht, ht + r / 4, ht + r / 2, ht + 3 * r / 4, ht + r, ht + r + 2])
    rows = "\n".join(
        f"| {p:.2f} | {np.clip((p - ht) / r, 0, 1):.3f} | {m:.3f} | {1 - m:.3f} |"
        for p, m in zip(probes, smoothstep_mask(probes, ht, r))
    )
    mo.md(
        "| nDSM (m) | $t$ | mask | ground_conf |\n|---|---|---|---|\n"
        + rows
        + "\n\nNaN pixels are not covered by this curve: the pipeline forces "
        "`ground_conf = 0` there, i.e. mask = 1."
    )
    return


if __name__ == "__main__":
    app.run()

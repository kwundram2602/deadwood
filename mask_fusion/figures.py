"""Run figures: a legend table (PNG) and the rule chain as a TeX decision tree.

Both are built from the validated eco classes of the run, so they always show
the mapping and thresholds that produced the rasters next to them.
"""

import logging
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from mask_fusion.codes import COMBO_NAMES, EcoClass, build_lookup  # noqa: E402

logger = logging.getLogger(__name__)

INK = "#1f1f1f"
MUTED = "#6b6b6b"


def _tex_escape(text: str) -> str:
    for char in "\\&%$#_{}":
        text = text.replace(char, "\\" + char)
    return text


def _metres(value: float) -> str:
    """1.0 -> '1.0', 1.25 -> '1.25'."""
    return f"{value:.1f}" if round(value, 1) == value else f"{value:g}"


def _is_dark(rgb: tuple[int, int, int]) -> bool:
    r, g, b = rgb
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255 < 0.5


def legend_png(rows: Sequence[tuple[EcoClass, float, float]], path: str | Path) -> Path:
    """One row per eco class: swatch, ID, name, area m2, share %."""
    path = Path(path)
    n = len(rows)
    fig, ax = plt.subplots(figsize=(6.4, 0.34 * (n + 1) + 0.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(n + 0.6, -0.6)
    ax.axis("off")

    header: dict[str, Any] = dict(fontsize=9, fontweight="bold", color=INK, va="center")
    ax.text(0.13, 0, "ID", ha="right", **header)
    ax.text(0.17, 0, "Class", ha="left", **header)
    ax.text(0.80, 0, "Area m²", ha="right", **header)
    ax.text(0.97, 0, "Share %", ha="right", **header)
    ax.axhline(0.5, xmin=0.02, xmax=0.97, color=MUTED, linewidth=0.6)

    cell: dict[str, Any] = dict(fontsize=9, color=INK, va="center")
    for i, (e, area_m2, share_pct) in enumerate(rows, start=1):
        rgb = tuple(c / 255 for c in e.color)
        ax.add_patch(
            Rectangle((0.02, i - 0.32), 0.06, 0.64, facecolor=rgb, edgecolor=MUTED, linewidth=0.4)
        )
        ax.text(0.13, i, str(e.value), ha="right", **cell)
        ax.text(0.17, i, e.name, ha="left", **cell)
        ax.text(0.80, i, f"{area_m2:,.1f}", ha="right", **cell)
        ax.text(0.97, i, f"{share_pct:.2f}", ha="right", **cell)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def decision_tree_tex(eco: Sequence[EcoClass], h1: float, h2: float) -> str:
    """Standalone forest document: crown -> deadwood -> height class -> combo -> eco."""
    lut = build_lookup(eco)
    by_value = {e.value: e for e in eco}
    a, b = _metres(h1), _metres(h2)
    height_labels = {
        2: rf"$h \geq {b}$\,m",
        1: rf"${a} \leq h < {b}$\,m",
        0: rf"$h < {a}$\,m",
    }

    lines = [
        r"\documentclass[border=8pt]{standalone}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        r"\usepackage{forest}",
    ]
    lines += [
        rf"\definecolor{{eco{e.value}}}{{RGB}}{{{e.color[0]},{e.color[1]},{e.color[2]}}}"
        for e in eco
    ]
    lines += [
        r"\begin{document}",
        r"\begin{forest}",
        r"for tree={grow'=east, draw, rounded corners=2pt, align=left, font=\small,",
        r"  inner sep=3pt, l sep=7mm, s sep=1.2mm, parent anchor=east, child anchor=west,",
        r"  edge={-latex, draw=black!55}, edge path={\noexpand\path[\forestoption{edge}]",
        r"  (!u.parent anchor) -- +(3mm,0) |- (.child anchor)\forestoption{edge label};}}",
        # invisible root: the two crown branches are the first visible decision
        r"[, phantom",
    ]
    for crown in (1, 0):
        lines.append(f"    [{{crown = {crown}}}")
        for dead in (1, 0):
            lines.append(f"      [{{deadwood = {dead}}}")
            for hc in (2, 1, 0):
                code = crown + 2 * dead + 4 * hc
                e = by_value[int(lut[code])]
                text = r", text=white" if _is_dark(e.color) else ""
                lines.append(f"        [{height_labels[hc]}")
                lines.append(
                    f"          [{{code {code} \\texttt{{{_tex_escape(COMBO_NAMES[code])}}}\\\\"
                    f"$\\rightarrow$ {e.value} \\textbf{{{_tex_escape(e.name)}}}}}, "
                    f"fill=eco{e.value}{text}]"
                )
                lines.append("        ]")
            lines.append("      ]")
        lines.append("    ]")
    lines += ["]", r"\end{forest}", r"\end{document}", ""]
    return "\n".join(lines)


def compile_tex(tex: str, out_pdf: str | Path) -> Path | None:
    """pdflatex in a scratch dir; None (with a warning) if it is missing or fails."""
    out_pdf = Path(out_pdf)
    if shutil.which("pdflatex") is None:
        logger.warning("pdflatex not on PATH - decision tree kept as .tex only")
        return None
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "tree.tex").write_text(tex, encoding="utf-8")
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "tree.tex"],
            cwd=tmp,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            tail = "\n".join(proc.stdout.splitlines()[-15:])
            logger.warning("pdflatex failed - decision tree kept as .tex only:\n%s", tail)
            return None
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(tmp) / "tree.pdf", out_pdf)
    return out_pdf


def pdf_to_png(pdf: Path) -> Path | None:
    """300 dpi PNG next to the PDF when poppler's pdftoppm is available."""
    if shutil.which("pdftoppm") is None:
        return None
    subprocess.run(
        ["pdftoppm", "-png", "-r", "300", "-singlefile", pdf, pdf.with_suffix("")], check=True
    )
    return pdf.with_suffix(".png")

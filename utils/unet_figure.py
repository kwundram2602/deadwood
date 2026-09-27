"""Paper-style U-Net figure (PlotNeuralNet 3D blocks) from a torchview ``model_graph.gv``.

``save_model_graph`` writes a layer-by-layer torchview graph that is far too long for a
paper. This module reads that ``.gv`` back, collapses it to U-Net stages (encoder
features, decoder blocks, head, skip connections) and emits a TikZ figure in the U
layout using the PlotNeuralNet layer macros vendored in ``plotneuralnet_layers/``.
Compiling needs ``pdflatex`` on PATH; ``pdftoppm`` (poppler) adds an optional PNG.
"""

import math
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

LAYERS_DIR = Path(__file__).parent / "plotneuralnet_layers"

# PlotNeuralNet draws with its default pic scale; sizes below are in those units.
_SCALE = 0.2
# TikZ's default z unit vector is (-3.85mm, -3.85mm): a box of depth d reaches this
# far above its top edge on the canvas.
_Z_PROJ = 0.385
_X_GAP = 1.8  # cm between neighbouring blocks
_LABEL_H = 1.4  # cm reserved for the (up to three-line) label above each block
_UP_SLAB_W = 1.0  # units, width of the upsampling slab in front of a decoder block

_SUBGRAPH = re.compile(r"^\s*subgraph (cluster_\d+) \{")
_CLUSTER_LABEL = re.compile(r"\blabel=(\w+)")
_NODE_START = re.compile(r"^\s*(\d+) \[label=<")
_EDGE = re.compile(r"^\s*(\d+) -> (\d+)")
_OP = re.compile(r"<TD[^>]*>([\w.-]+)<BR/>")
_SHAPE = re.compile(r"\((\d+(?:, \d+)+)\s*\)")
_CLOSE = re.compile(r"^\s*\}\s*$")


@dataclass(frozen=True)
class GvNode:
    id: int
    op: str
    out_shape: tuple[int, ...]
    # (cluster id, cluster label) from outermost to innermost
    clusters: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class GvGraph:
    nodes: dict[int, GvNode]
    edges: list[tuple[int, int]]


@dataclass(frozen=True)
class Stage:
    name: str
    kind: Literal["input", "enc", "dec", "head"]
    channels: int
    height: int
    width: int
    skip_from: str | None = None
    detail: str | None = None  # third label line, e.g. "3x Bottleneck"


@dataclass(frozen=True)
class EncoderInfo:
    name: str  # e.g. "ResNet-50", or the torchview class name when not a ResNet
    block: str | None  # residual block class shared by all stages, if any
    blocks_per_stage: tuple[int, ...]


# Layers per residual block (conv layers on the main path) for the torchvision ResNets.
_RESNET_BLOCK_LAYERS = {"Bottleneck": 3, "BasicBlock": 2}
_OP_ABBREV = {"Conv2d": "Conv", "BatchNorm2d": "BN"}


def parse_gv(path: Path) -> GvGraph:
    """Parse the dot source torchview writes (HTML-table labels, nested clusters)."""
    nodes: dict[int, GvNode] = {}
    edges: list[tuple[int, int]] = []
    stack: list[tuple[str, str]] = []
    pending_cluster: str | None = None
    node_id: int | None = None
    buf: list[str] = []

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if node_id is not None:
            buf.append(line)
            if "</TABLE>>" in line:
                text = "\n".join(buf)
                op = _OP.search(text)
                shapes = _SHAPE.findall(text)
                if op is None or not shapes:
                    raise ValueError(f"node {node_id}: cannot read op/shape from label")
                out = tuple(int(v) for v in shapes[-1].split(", "))
                nodes[node_id] = GvNode(node_id, op.group(1), out, tuple(stack))
                node_id, buf = None, []
            continue
        if m := _SUBGRAPH.match(line):
            pending_cluster = m.group(1)
        elif pending_cluster is not None and (m := _CLUSTER_LABEL.search(line)):
            stack.append((pending_cluster, m.group(1)))
            pending_cluster = None
        elif m := _NODE_START.match(line):
            node_id, buf = int(m.group(1)), [line]
        elif m := _EDGE.match(line):
            edges.append((int(m.group(1)), int(m.group(2))))
        elif _CLOSE.match(line) and stack:
            stack.pop()
    return GvGraph(nodes, edges)


def _chw(node: GvNode) -> tuple[int, int, int]:
    if len(node.out_shape) != 4:
        raise ValueError(f"node {node.id} ({node.op}): expected NCHW, got {node.out_shape}")
    _, c, h, w = node.out_shape
    return c, h, w


def _encoder_nodes(g: GvGraph) -> tuple[tuple[str, str], list[GvNode]]:
    """The single top-level ``*Encoder`` cluster and its nodes, in graph order."""
    top_level = {n.clusters[0] for n in g.nodes.values() if n.clusters}
    encoders = [c for c in top_level if c[1].endswith("Encoder")]
    if len(encoders) != 1:
        raise ValueError(f"expected one *Encoder cluster, found {[c[1] for c in encoders]}")
    nodes = sorted(
        (n for n in g.nodes.values() if n.clusters and n.clusters[0] == encoders[0]),
        key=lambda n: n.id,
    )
    return encoders[0], nodes


def _encoder_stage_blocks(nodes: list[GvNode]) -> dict[str, list[GvNode]]:
    """Blocks directly inside each encoder sub-cluster (e.g. ResNet ``layer1..4``)."""
    stages: dict[str, list[GvNode]] = {}
    for n in nodes:
        if len(n.clusters) == 2:
            stages.setdefault(n.clusters[1][0], []).append(n)
    return stages


def describe_encoder(g: GvGraph) -> EncoderInfo:
    """Name the encoder; torchvision ResNets are recognised by their block class.

    A ResNet's depth is its conv layers on the main path: blocks x layers per block,
    plus the stem conv and the final fc (which smp drops, but the name keeps).
    """
    (_, label), nodes = _encoder_nodes(g)
    per_stage = list(_encoder_stage_blocks(nodes).values())
    ops = {n.op for blocks in per_stage for n in blocks}
    counts = tuple(len(blocks) for blocks in per_stage)
    if len(ops) != 1:
        return EncoderInfo(label, None, counts)
    block = ops.pop()
    if label == "ResNetEncoder" and block in _RESNET_BLOCK_LAYERS:
        depth = _RESNET_BLOCK_LAYERS[block] * sum(counts) + 2
        return EncoderInfo(f"ResNet-{depth}", block, counts)
    return EncoderInfo(label, block, counts)


def extract_stages(g: GvGraph) -> list[Stage]:
    """Collapse the layer graph to input, encoder features, decoder blocks and head.

    Encoder features are the encoder nodes that feed a decoder ``cat`` (the skips)
    plus the encoder's last node (the bottleneck). Each ``*DecoderBlock`` cluster is
    one decoder stage; its skip is whichever encoder feature enters its ``cat``.
    """
    by_op = defaultdict(list)
    for n in g.nodes.values():
        by_op[n.op].append(n)
    if not by_op["input-tensor"] or not by_op["output-tensor"]:
        raise ValueError("graph has no input-tensor/output-tensor node")

    preds: dict[int, list[int]] = defaultdict(list)
    for src, dst in g.edges:
        preds[dst].append(src)

    (_, enc_label), enc_nodes = _encoder_nodes(g)
    enc_ids = {n.id for n in enc_nodes}
    skip_src = {p for cat in by_op["cat"] for p in preds[cat.id] if p in enc_ids}
    if not skip_src:
        raise ValueError("no encoder -> cat edges; is this a U-Net?")
    feature_ids = sorted(skip_src | {max(enc_ids)})
    stage_blocks = _encoder_stage_blocks(enc_nodes)
    stage_index = {cid: k for k, cid in enumerate(stage_blocks, start=1)}
    resnet = enc_label == "ResNetEncoder"

    stages = [Stage("Input", "input", *_chw(by_op["input-tensor"][0]))]
    enc_name: dict[int, str] = {}
    for i, nid in enumerate(feature_ids, start=1):
        node = g.nodes[nid]
        if len(node.clusters) == 1:  # directly in the encoder: the stem
            stem_ops = [n.op for n in enc_nodes if len(n.clusters) == 1 and n.id <= nid]
            detail = "--".join(_OP_ABBREV.get(op, op) for op in stem_ops)
            name = "Stem" if resnet else f"Enc {i}"
        else:
            cid = node.clusters[1][0]
            detail = f"{len(stage_blocks[cid])}$\\times$ {node.op}"
            name = f"Layer {stage_index[cid]}" if resnet else f"Enc {i}"
        # The deepest feature feeds the decoder directly (smp's decoder centre is an
        # Identity), so it is the U-Net bottleneck rather than one more skip source.
        if not resnet and nid == feature_ids[-1]:
            name = "Bottleneck"
        enc_name[nid] = name
        stages.append(Stage(name, "enc", *_chw(node), detail=detail))
    heights = [s.height for s in stages[1:]]
    if heights != sorted(heights, reverse=True) or len(set(heights)) != len(heights):
        raise ValueError(f"encoder features do not halve in resolution: {heights}")

    blocks: dict[str, list[GvNode]] = {}
    for n in sorted(g.nodes.values(), key=lambda n: n.id):
        for cid, label in n.clusters:
            if label.endswith("DecoderBlock"):
                blocks.setdefault(cid, []).append(n)
    if not blocks:
        raise ValueError("no *DecoderBlock clusters found")
    for i, members in enumerate(blocks.values(), start=1):
        skip = next(
            (enc_name[p] for n in members if n.op == "cat" for p in preds[n.id] if p in enc_name),
            None,
        )
        stages.append(Stage(f"Dec {i}", "dec", *_chw(members[-1]), skip_from=skip))

    stages.append(Stage("Head", "head", *_chw(by_op["output-tensor"][0])))
    return stages


# ── TikZ ────────────────────────────────────────────────────────────────────────

_PREAMBLE = r"""\documentclass[border=8pt, tikz]{standalone}
\input{init}
\usetikzlibrary{3d, calc, fit, decorations.pathreplacing}
\def\ConvColor{rgb:yellow,5;red,2.5;white,5}
\def\PoolColor{rgb:red,1;black,0.3}
\def\UnpoolColor{rgb:blue,2;green,1;black,0.3}
\def\SoftmaxColor{rgb:magenta,5;black,7}
\def\InputColor{rgb:white,5;black,1}
\def\SkipColor{rgb:blue,4;red,1;green,1;black,3}
\begin{document}
\begin{tikzpicture}
\tikzstyle{flow}=[line width=0.6mm, draw=\edgecolor, opacity=0.8, -Stealth]
\tikzstyle{down}=[line width=0.6mm, draw=\PoolColor, opacity=0.8, -Stealth]
\tikzstyle{up}=[line width=0.6mm, draw=\UnpoolColor, opacity=0.8, -Stealth]
\tikzstyle{skip}=[line width=0.6mm, draw=\SkipColor, opacity=0.8, densely dashed, -Stealth]
"""

_END = r"""\end{tikzpicture}
\end{document}
"""


@dataclass(frozen=True)
class _Placed:
    stage: Stage
    x: float  # cm, west face of the (first) box
    y: float  # cm, vertical centre
    w: float  # units, total width incl. an upsampling slab
    h: float  # units, height == depth

    @property
    def half_h(self) -> float:
        return self.h * _SCALE / 2

    def front(self, fx: float, fy: float) -> str:
        """Point on the front face; fx/fy in [0, 1] from the south-west corner."""
        s = _SCALE
        return (
            f"({self.x + fx * self.w * s:.3f},"
            f"{self.y - self.half_h + fy * self.h * s:.3f},{self.h * s / 2:.3f})"
        )


def _level(stage: Stage, full: int) -> int:
    return round(math.log2(full / stage.height))


def _box_w(channels: int) -> float:
    return 1.0 + 0.8 * math.log2(max(channels, 1))


def _box_h(height: int) -> float:
    return 2.5 * math.log2(max(height, 2))


def _layout(stages: list[Stage]) -> list[_Placed]:
    full = stages[0].height
    levels = [_level(s, full) for s in stages]
    widths = [_box_w(s.channels) + (_UP_SLAB_W if s.kind == "dec" else 0) for s in stages]
    heights = [_box_h(s.height) for s in stages]

    # A skip at level L runs horizontally over every deeper block, on the front face
    # (projected _Z_PROJ * d/2 below the centre). Level L+1 must sit low enough for its
    # tallest block, its back-face projection and its label to clear that line.
    proj: dict[int, float] = defaultdict(float)
    reach: dict[int, float] = defaultdict(float)
    for lvl, h in zip(levels, heights, strict=True):
        hc = h * _SCALE
        proj[lvl] = max(proj[lvl], _Z_PROJ * hc / 2)
        reach[lvl] = max(reach[lvl], hc / 2 + _Z_PROJ * hc / 2 + _LABEL_H)
    y_of = {0: 0.0}
    for lvl in range(1, max(levels) + 1):
        y_of[lvl] = y_of[lvl - 1] - proj[lvl - 1] - reach[lvl] - 0.3

    placed, x = [], 0.0
    for s, lvl, w, h in zip(stages, levels, widths, heights, strict=True):
        placed.append(_Placed(s, x, y_of[lvl], w, h))
        x += w * _SCALE + _X_GAP
    return placed


def _label(p: _Placed) -> str:
    s = p.stage
    size = f"{s.height}$^2$" if s.height == s.width else f"{s.height}$\\times${s.width}"
    top = f"({p.x + p.w * _SCALE / 2:.3f},{p.y + p.half_h:.3f},{-p.h * _SCALE / 2:.3f})"
    detail = f"\\\\{{\\footnotesize\\itshape {s.detail}}}" if s.detail else ""
    return (
        f"\\node[anchor=south, align=center, font=\\small] at {top} "
        f"{{\\textbf{{{s.name}}}\\\\{s.channels} $\\times$ {size}{detail}}};"
    )


def _pic(p: _Placed, name: str) -> list[str]:
    s = p.stage
    at = f"({p.x:.3f},{p.y:.3f},0)"
    dims = f"height={p.h:.2f},depth={p.h:.2f}"
    if s.kind in ("input", "head"):
        fill = r"\InputColor" if s.kind == "input" else r"\SoftmaxColor"
        return [f"\\pic at {at} {{Box={{name={name},fill={fill},width={p.w:.2f},{dims}}}}};"]
    if s.kind == "enc":
        return [f"\\pic at {at} {{Box={{name={name},fill=\\ConvColor,width={p.w:.2f},{dims}}}}};"]
    conv_w = p.w - _UP_SLAB_W
    conv_at = f"({p.x + _UP_SLAB_W * _SCALE:.3f},{p.y:.3f},0)"
    return [
        f"\\pic at {at} {{Box={{name={name}-up,fill=\\UnpoolColor,opacity=0.5,"
        f"width={_UP_SLAB_W},{dims}}}}};",
        f"\\pic at {conv_at} {{Box={{name={name},fill=\\ConvColor,width={conv_w:.2f},{dims}}}}};",
    ]


def _legend(x: float, y: float) -> list[str]:
    rows = [
        (
            r"\fill[fill=\ConvColor, opacity=0.4] (0,0) rectangle ++(0.5,0.3);",
            "encoder / decoder conv stage",
        ),
        (
            r"\fill[fill=\UnpoolColor, opacity=0.6] (0,0) rectangle ++(0.5,0.3);",
            "upsample $\\times$2",
        ),
        (
            r"\fill[fill=\SoftmaxColor, opacity=0.6] (0,0) rectangle ++(0.5,0.3);",
            "segmentation head",
        ),
        (r"\draw[down] (0,0.15) -- ++(0.5,0);", "downsample $\\times$2"),
        (r"\draw[up] (0,0.15) -- ++(0.5,0);", "decoder path"),
        (r"\draw[skip] (0,0.15) -- ++(0.5,0);", "skip connection (concat)"),
    ]
    out = []
    for i, (glyph, text) in enumerate(rows):
        out.append(f"\\begin{{scope}}[shift={{({x:.3f},{y - 0.55 * i:.3f})}}]")
        out.append(glyph)
        out.append(f"\\node[anchor=west, font=\\small] at (0.65,0.15) {{{text}}};")
        out.append(r"\end{scope}")
    return out


def _bottleneck_box(encoder: EncoderInfo, x_mid: float, top: float) -> list[str]:
    """Inset explaining the ResNet bottleneck residual block, centred on ``x_mid``.

    The middle conv is pinned to the centre and the chain grows outwards from it;
    ``fit`` then frames whatever extent the text ends up with.
    """
    return [
        r"\begin{scope}[font=\small, >=Stealth]",
        r"\tikzstyle{op}=[draw, rounded corners=2pt, fill=\ConvColor, fill opacity=0.4,"
        r" text opacity=1, align=center, minimum height=1.1cm, inner sep=4pt]",
        f"\\node[anchor=north, font=\\bfseries] (bt) at ({x_mid:.3f},{top:.3f}) "
        f"{{{encoder.name} {encoder.block.lower() if encoder.block else ''} block}};",
        r"\node[op, below=0.5cm of bt] (c2) {3$\times$3 Conv, BN, ReLU};",
        r"\node[op, left=0.6cm of c2] (c1) {1$\times$1 Conv, BN, ReLU};",
        r"\node[left=0.9cm of c1] (bx) {$x$};",
        r"\node[op, right=0.6cm of c2] (c3) {1$\times$1 Conv, BN};",
        r"\node[draw, circle, inner sep=1pt, right=0.6cm of c3] (add) {$+$};",
        r"\node[op, fill=white, right=0.6cm of add] (relu) {ReLU};",
        r"\node[right=0.6cm of relu] (by) {$y$};",
        r"\foreach \a/\b in {bx/c1, c1/c2, c2/c3, c3/add, add/relu, relu/by}"
        r" \draw[->, thick] (\a) -- (\b);",
        r"\coordinate (split) at ($(bx.east)!0.45!(c1.west)$);",
        r"\draw[->, thick, densely dashed] (split) |- ($(c2.south)+(0,-0.5)$) -| (add);",
        r"\node[below=0.55cm of c2, font=\footnotesize] (sc)"
        r" {shortcut: identity, or 1$\times$1 Conv + BN in the first block of each layer};",
        r"\node[draw=black!40, rounded corners=4pt, inner sep=8pt, fit=(bt)(bx)(by)(sc)] {};",
        r"\end{scope}",
    ]


def _braces(placed: list[_Placed], encoder: EncoderInfo) -> list[str]:
    enc = [p for p in placed if p.stage.kind == "enc"]
    dec = [p for p in placed if p.stage.kind == "dec"]
    # lowest visible point: the front bottom edge sits _Z_PROJ * d/2 below the box
    y = min(p.y - p.half_h - _Z_PROJ * p.half_h for p in placed) - 0.4
    enc_text = f"{encoder.name} encoder"
    if encoder.block and encoder.blocks_per_stage:
        counts = "/".join(str(c) for c in encoder.blocks_per_stage)
        enc_text += f" ({counts} {encoder.block.lower()} blocks)"
    out = []
    for group, text in ((enc, enc_text), (dec, "U-Net decoder")):
        x0, x1 = group[0].x, group[-1].x + group[-1].w * _SCALE
        out.append(
            f"\\draw[decorate, decoration={{brace, mirror, amplitude=7pt}}, thick] "
            f"({x0:.3f},{y:.3f}) -- ({x1:.3f},{y:.3f}) "
            f"node[midway, below=9pt, font=\\small\\bfseries] {{{text}}};"
        )
    return out


def to_tikz(stages: list[Stage], encoder: EncoderInfo | None = None) -> str:
    """Emit a standalone LaTeX document drawing ``stages`` as a U.

    With ``encoder`` the figure also names the encoder/decoder halves and, for a
    ResNet bottleneck encoder, explains the residual block in an inset.
    """
    placed = _layout(stages)
    names = {p.stage.name: f"s{i}" for i, p in enumerate(placed)}
    by_name = {p.stage.name: p for p in placed}

    body = ["% blocks"]
    for p in placed:
        body += _pic(p, names[p.stage.name])
        body.append(_label(p))

    body.append("% connections")
    for a, b in zip(placed, placed[1:], strict=False):
        if b.y < a.y:  # descending: leave through the bottom, enter from the left
            body.append(f"\\draw[down] {a.front(0.5, 0)} |- {b.front(0, 0.5)};")
        elif b.y > a.y:  # ascending: leave to the right, enter from below
            body.append(f"\\draw[up] {a.front(1, 0.5)} -| {b.front(0.5, 0)};")
        else:
            body.append(f"\\draw[flow] {a.front(1, 0.5)} -- {b.front(0, 0.5)};")

    for p in placed:
        if p.stage.skip_from is not None:
            src = by_name[p.stage.skip_from]
            body.append(f"\\draw[skip] {src.front(1, 0.5)} -- {p.front(0, 0.5)};")

    bottom = min(p.y - p.half_h for p in placed)
    body += _legend(0.0, bottom + 2.9)

    if encoder is not None:
        body += _braces(placed, encoder)
        if encoder.block == "Bottleneck":
            # the empty middle of the U, level with the top row of labels
            top = max(p.y + p.half_h + _Z_PROJ * p.half_h for p in placed) + _LABEL_H
            x_mid = (placed[0].x + placed[-1].x + placed[-1].w * _SCALE) / 2
            body += _bottleneck_box(encoder, x_mid, top)
    return _PREAMBLE + "\n".join(body) + "\n" + _END


def render_tex(tex: str, out_pdf: Path, build_dir: Path) -> Path:
    """Compile ``tex`` in ``build_dir`` (layer files copied in) and copy the PDF out."""
    if shutil.which("pdflatex") is None:
        raise RuntimeError("pdflatex not on PATH")
    build_dir.mkdir(parents=True, exist_ok=True)
    for f in LAYERS_DIR.iterdir():
        if f.suffix in (".sty", ".tex"):
            shutil.copy(f, build_dir / f.name)
    (build_dir / "figure.tex").write_text(tex, encoding="utf-8")
    proc = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "figure.tex"],
        cwd=build_dir,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-25:])
        raise RuntimeError(f"pdflatex failed:\n{tail}")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(build_dir / "figure.pdf", out_pdf)
    return out_pdf


def plot_unet_from_gv(gv_path: Path, out_path: Path | None = None, keep_tex: bool = False) -> Path:
    """Write a PlotNeuralNet-style U-Net figure for ``gv_path``; returns the PDF path.

    Default output is ``model_unet.pdf`` next to the ``.gv``. A 300 dpi PNG is added
    when ``pdftoppm`` is available. With ``keep_tex`` the self-contained LaTeX
    sources stay in ``<stem>_tex/`` for hand-editing.
    """
    gv_path = Path(gv_path)
    out_pdf = Path(out_path) if out_path else gv_path.with_name("model_unet.pdf")
    g = parse_gv(gv_path)
    tex = to_tikz(extract_stages(g), describe_encoder(g))

    if keep_tex:
        render_tex(tex, out_pdf, out_pdf.with_name(f"{out_pdf.stem}_tex"))
    else:
        with tempfile.TemporaryDirectory() as tmp:
            render_tex(tex, out_pdf, Path(tmp))

    if shutil.which("pdftoppm"):
        subprocess.run(
            ["pdftoppm", "-png", "-r", "300", "-singlefile", out_pdf, out_pdf.with_suffix("")],
            check=True,
        )
    return out_pdf

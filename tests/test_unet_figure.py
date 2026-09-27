import os
import re
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.unet_figure import (
    EncoderInfo,
    describe_encoder,
    extract_stages,
    parse_gv,
    plot_unet_from_gv,
    to_tikz,
)

GV = Path(__file__).parent / "data" / "model_graph_resnet50_unet.gv"


@pytest.fixture(scope="module")
def stages():
    return extract_stages(parse_gv(GV))


def test_parse_gv_reads_every_node_and_edge():
    g = parse_gv(GV)
    assert len(g.nodes) == 54
    assert g.nodes[0].op == "input-tensor"
    assert g.nodes[0].out_shape == (1, 3, 512, 512)
    assert g.nodes[23].op == "cat"
    assert [label for _, label in g.nodes[23].clusters] == ["UnetDecoder", "UnetDecoderBlock"]
    assert (17, 23) in g.edges


def test_encoder_features_halve_resolution(stages):
    enc = [(s.name, s.channels, s.height) for s in stages if s.kind == "enc"]
    assert enc == [
        ("Stem", 64, 256),
        ("Layer 1", 256, 128),
        ("Layer 2", 512, 64),
        ("Layer 3", 1024, 32),
        ("Layer 4", 2048, 16),
    ]


def test_encoder_stage_details_count_blocks(stages):
    details = [s.detail for s in stages if s.kind == "enc"]
    assert details[0] == "Conv--BN--ReLU"  # stem ends before the max-pool
    assert details[1:] == [f"{n}$\\times$ Bottleneck" for n in (3, 4, 6, 3)]


def test_describe_encoder_recognises_resnet50():
    assert describe_encoder(parse_gv(GV)) == EncoderInfo("ResNet-50", "Bottleneck", (3, 4, 6, 3))


def test_decoder_stages_and_skips(stages):
    dec = [(s.channels, s.height, s.skip_from) for s in stages if s.kind == "dec"]
    assert dec == [
        (256, 32, "Layer 3"),
        (128, 64, "Layer 2"),
        (64, 128, "Layer 1"),
        (32, 256, "Stem"),
        (16, 512, None),
    ]


def test_input_and_head(stages):
    assert (stages[0].kind, stages[0].channels, stages[0].height) == ("input", 3, 512)
    assert (stages[-1].kind, stages[-1].channels, stages[-1].height) == ("head", 1, 512)


def test_to_tikz_draws_every_block_and_skip(stages):
    tex = to_tikz(stages)
    # input + 5 encoder + 5 decoder (slab + conv each) + head
    assert tex.count(r"\pic at") == 1 + 5 + 2 * 5 + 1
    # 3D endpoints only; the legend sample is a 2D line
    assert len(re.findall(r"\\draw\[skip\] \([^)]*,[^)]*,[^)]*\)", tex)) == 4
    assert "ResNet-50" not in tex  # no encoder info -> no braces or inset


def test_to_tikz_with_encoder_adds_braces_and_block_inset(stages):
    tex = to_tikz(stages, describe_encoder(parse_gv(GV)))
    assert tex.count("decoration={brace") == 2
    assert "ResNet-50 encoder (3/4/6/3 bottleneck blocks)" in tex
    assert "ResNet-50 bottleneck block" in tex


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not installed")
def test_plot_unet_from_gv_writes_pdf(tmp_path):
    pdf = plot_unet_from_gv(GV, tmp_path / "unet.pdf")
    assert pdf.read_bytes().startswith(b"%PDF")

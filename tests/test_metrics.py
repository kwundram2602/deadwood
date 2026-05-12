import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.metrics import NODATA, MetricAccumulator, pixel_metrics


def test_pixel_metrics_perfect():
    logits = torch.full((2, 1, 4, 4), 5.0)
    target = torch.ones(2, 1, 4, 4)
    m = pixel_metrics(logits, target)
    assert m["acc"] == pytest.approx(1.0, abs=0.01)
    assert m["f1"] == pytest.approx(1.0, abs=0.01)
    assert m["iou"] == pytest.approx(1.0, abs=0.01)


def test_pixel_metrics_iou_known_value():
    # Layout (1,1,2,2):
    # (0,0) logit=5  target=1 → TP
    # (0,1) logit=5  target=1 → TP
    # (1,0) logit=5  target=0 → FP
    # (1,1) logit=-5 target=1 → FN
    # IoU = TP/(TP+FP+FN) = 2/(2+1+1) = 0.5
    logits = torch.zeros(1, 1, 2, 2)
    logits[0, 0, 0, 0] = 5.0
    logits[0, 0, 0, 1] = 5.0
    logits[0, 0, 1, 0] = 5.0
    logits[0, 0, 1, 1] = -5.0
    target = torch.zeros(1, 1, 2, 2)
    target[0, 0, 0, 0] = 1.0
    target[0, 0, 0, 1] = 1.0
    target[0, 0, 1, 1] = 1.0
    m = pixel_metrics(logits, target)
    assert m["iou"] == pytest.approx(0.5, abs=0.01)


def test_pixel_metrics_nodata_ignored():
    logits = torch.full((1, 1, 4, 4), -5.0)
    target = torch.full((1, 1, 4, 4), float(NODATA))
    m = pixel_metrics(logits, target)
    assert m["acc"] == pytest.approx(0.0)
    assert m["f1"] == pytest.approx(0.0)


def test_pixel_metrics_custom_threshold():
    # sigmoid(1.0) ≈ 0.731; with threshold=0.8 that becomes 0 (FN)
    logits = torch.full((1, 1, 4, 4), 1.0)
    target = torch.ones(1, 1, 4, 4)
    m_low = pixel_metrics(logits, target, threshold=0.5)
    m_high = pixel_metrics(logits, target, threshold=0.8)
    assert m_low["f1"] > m_high["f1"]


def test_accumulator_compute_matches_pixel_metrics():
    torch.manual_seed(42)
    logits = torch.randn(4, 1, 8, 8)
    target = (torch.rand(4, 1, 8, 8) > 0.5).float()
    threshold = 0.5

    acc = MetricAccumulator()
    acc.update(logits, target, 1.0, 4)
    result = acc.compute(threshold=threshold)

    expected = pixel_metrics(logits, target, threshold=threshold)
    assert result["f1"] == pytest.approx(expected["f1"], abs=0.01)
    assert result["iou"] == pytest.approx(expected["iou"], abs=0.01)
    assert result["prec"] == pytest.approx(expected["prec"], abs=0.01)
    assert result["rec"] == pytest.approx(expected["rec"], abs=0.01)


def test_accumulator_auc_pr_perfect():
    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, :2, :] = 1.0
    logits = torch.where(target == 1, torch.tensor(5.0), torch.tensor(-5.0))
    acc = MetricAccumulator()
    acc.update(logits, target, 0.0, 1)
    result = acc.compute()
    assert result["auc_pr"] == pytest.approx(1.0, abs=0.01)


def test_accumulator_auc_pr_worst():
    # 1 positive, 15 negatives — worst predictor scores near baseline (1/16 ≈ 0.06)
    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, 0, 0] = 1.0
    # inverted: high score for negative, low score for the one positive
    logits = torch.where(target == 1, torch.tensor(-5.0), torch.tensor(5.0))
    acc = MetricAccumulator()
    acc.update(logits, target, 0.0, 1)
    result = acc.compute()
    assert result["auc_pr"] < 0.2


def test_accumulator_loss_averaged():
    logits = torch.randn(2, 1, 4, 4)
    target = torch.rand(2, 1, 4, 4)
    acc = MetricAccumulator()
    acc.update(logits, target, 2.0, 2)
    acc.update(logits, target, 4.0, 2)
    result = acc.compute()
    # (2.0*2 + 4.0*2) / (2+2) = 3.0
    assert result["loss"] == pytest.approx(3.0, abs=1e-5)


def test_accumulator_reset_clears_state():
    acc = MetricAccumulator()
    acc.update(torch.randn(2, 1, 4, 4), torch.rand(2, 1, 4, 4), 1.0, 2)
    acc.reset()
    assert acc._n == 0
    assert len(acc._probs) == 0
    assert len(acc._targets) == 0


def test_accumulator_all_nodata_does_not_crash():
    logits = torch.zeros(1, 1, 4, 4)
    target = torch.full((1, 1, 4, 4), float(NODATA))
    acc = MetricAccumulator()
    acc.update(logits, target, 0.0, 1)
    result = acc.compute()
    assert "auc_pr" in result
    assert "f1" in result

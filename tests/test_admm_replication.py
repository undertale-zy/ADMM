from __future__ import annotations

from pathlib import Path

import pytest
import torch

from admm_guide_support_fusion import GuideSupportFusionADMM
from admm_losses import noise_aware_loss, normalized_nmse, observed_echo_loss
from admm_rounds32_36 import Round33ConfidenceBand
from admm_support_fusion import support_gate
from admm_unrolled import complex_from_channels
from dense_aircraft_isar_dataset import DenseAircraftISARDataset
from evaluation import load_round_checkpoint
from reference_schedules import STAGE1_SCALAR_8, schedule_tensors
from round_registry import (
    ROUND_REGISTRY,
    build_dataset,
    build_model,
    compute_round_loss,
    get_round_config,
)
from structured_isar_dataset import StructuredISARDataset
from synthetic_isar_dataset import SyntheticISARDataset
from train_round import _save_checkpoint, _update_selector


FULL_PARAMETER_COUNTS = {
    1: 24,
    2: 19_707,
    3: 346_683,
    4: 19_711,
    6: 346_687,
    8: 19_712,
    10: 346_688,
    32: 19_718,
    33: 19_714,
    34: 30_149,
    35: 19_720,
    36: 19_714,
}


@pytest.mark.parametrize(("round_id", "expected"), FULL_PARAMETER_COUNTS.items())
def test_historical_parameter_counts(round_id: int, expected: int) -> None:
    model = build_model(round_id)
    assert sum(parameter.numel() for parameter in model.parameters()) == expected


def test_registry_contains_and_builds_all_rounds() -> None:
    assert tuple(ROUND_REGISTRY) == tuple(range(1, 37))
    with pytest.raises(TypeError):
        ROUND_REGISTRY[1] = get_round_config(1)  # type: ignore[index]
    for round_id in range(1, 37):
        model = build_model(round_id, (32, 16), (16, 8))
        assert model.image_shape == (32, 16)
        assert model.measurement_shape == (16, 8)


@pytest.mark.parametrize("round_id", (1, 2, 3, 4, 8, 12, 16, 20, 24, 28, 32, 33, 34, 35, 36))
def test_model_families_have_finite_forward_and_gradients(round_id: int) -> None:
    torch.manual_seed(100 + round_id)
    model = build_model(round_id, (32, 16), (16, 8))
    measurements, target = build_dataset(
        round_id,
        "train",
        image_shape=(32, 16),
        measurement_shape=(16, 8),
        samples=1,
    )[0]
    output = model(measurements[None])
    assert output.shape == (1, 2, 32, 16)
    assert torch.isfinite(output).all()
    total = compute_round_loss(
        round_id, output, target[None], measurements[None]
    )["total"]
    total.backward()  # type: ignore[union-attr]
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_support_and_confidence_boundaries() -> None:
    zero = torch.zeros(2, 3, dtype=torch.complex64)
    assert torch.equal(support_gate(zero, 0.01), torch.zeros_like(zero.real))

    guide = GuideSupportFusionADMM((8, 8), (4, 4), dc_steps=0)
    guide_image = torch.tensor([[[0.0, 1e-4, 1.0]]], dtype=torch.complex64)
    mask, clean = guide.firm_support(guide_image)
    assert mask[0, 0, 0] == 0
    assert mask[0, 0, 2] == 1
    assert clean[0, 0, 0] == 0

    amplitudes = torch.tensor([1e-4, 1e-3, 1e-2, 1.0], dtype=torch.complex64)[None, None]
    confidence = Round33ConfidenceBand.confidence(amplitudes)
    torch.testing.assert_close(
        confidence,
        torch.tensor([[[0.0, 0.0, 1.0, 1.0]]]),
        atol=1e-6,
        rtol=0.0,
    )


def test_embedded_stage1_teacher_matches_documented_schedule() -> None:
    model = GuideSupportFusionADMM((8, 8), (4, 4), dc_steps=0)
    expected = schedule_tensors(STAGE1_SCALAR_8)
    for actual, wanted in zip(model.guide_parameters, expected):
        torch.testing.assert_close(actual, wanted)
        assert not actual.requires_grad


def test_dataset_rng_protocols_are_deterministic_and_distinct() -> None:
    kwargs = {"image_shape": (32, 16), "measurement_shape": (16, 8), "seed": 77}
    point = SyntheticISARDataset(3, cache=False, **kwargs)
    cached = SyntheticISARDataset(3, cache=True, **kwargs)
    point_again = SyntheticISARDataset(3, cache=False, **kwargs)
    torch.testing.assert_close(point[2][0], point_again[2][0])
    assert not torch.equal(point[1][0], cached[1][0])

    structured_point = StructuredISARDataset(1, structured_probability=0.0, **kwargs)
    dense_point = DenseAircraftISARDataset(1, structured_probability=0.0, **kwargs)
    assert not torch.equal(point[0][0], structured_point[0][0])
    torch.testing.assert_close(structured_point[0][0], dense_point[0][0])

    structured = StructuredISARDataset(1, structured_probability=1.0, **kwargs)
    dense = DenseAircraftISARDataset(1, structured_probability=1.0, **kwargs)
    for dataset in (structured, dense):
        measurements, target = dataset[0]
        assert measurements.shape == (2, 16, 8)
        assert target.shape == (2, 32, 16)
        assert torch.isfinite(measurements).all()
        assert torch.isfinite(target).all()


def test_loss_formulas_and_historical_validation_weight() -> None:
    target = torch.zeros(1, 2, 4, 4)
    target[:, 0, 1, 1] = 1.0
    prediction = target.clone()
    measurements = torch.zeros(1, 2, 2, 2)

    predicted_complex = complex_from_channels(prediction)
    assert normalized_nmse(predicted_complex, predicted_complex) == 0
    base = observed_echo_loss(
        prediction, target, measurements, echo_weight=0.5, sparse_weight=1e-4
    )
    torch.testing.assert_close(
        base["total"], 0.5 * base["echo"] + 1e-4 * base["sparse"]
    )
    aware = noise_aware_loss(
        prediction,
        target,
        measurements,
        clean_weight=0.5,
        discrepancy_weight=0.1,
        background_weight=0.2,
        support_weight=0.1,
    )
    assert set(("clean_echo", "discrepancy", "background", "support")) <= set(aware)
    assert torch.isfinite(aware["total"])

    historical = compute_round_loss(
        2, prediction, target, measurements, validation=True, compatibility="historical"
    )
    corrected = compute_round_loss(
        2, prediction, target, measurements, validation=True, compatibility="corrected"
    )
    torch.testing.assert_close(
        corrected["total"] - historical["total"], 0.4 * historical["echo"]
    )


def test_selectors_preserve_historical_rules() -> None:
    trackers = {key: float("inf") for key in ("best", "score", "image", "echo", "constrained")}
    metrics = {"total": 1.0, "image": 0.02, "echo": 0.1, "background": 0.3}
    support_saves = _update_selector(metrics, get_round_config(20), trackers)
    assert set(support_saves) == {
        "checkpoint-best-score.pt",
        "checkpoint-best-image.pt",
        "checkpoint-best-echo.pt",
        "checkpoint-best-constrained.pt",
    }
    round_trackers = {key: float("inf") for key in trackers}
    assert _update_selector(metrics, get_round_config(32), round_trackers) == [
        "checkpoint-best.pt"
    ]
    assert round_trackers["best"] == pytest.approx(1.15)


@pytest.mark.parametrize("round_id", (1, 4, 32))
def test_checkpoint_schemas_strictly_reload(
    tmp_path: Path, round_id: int
) -> None:
    config = get_round_config(round_id)
    model = build_model(round_id, (32, 16), (16, 8))
    optimizer = torch.optim.AdamW(model.parameters())
    trackers = {key: float("inf") for key in ("best", "score", "image", "echo", "constrained")}
    checkpoint = tmp_path / f"round-{round_id}.pt"
    _save_checkpoint(
        checkpoint,
        model,
        optimizer,
        None,
        config,
        1,
        [],
        trackers,
        (32, 16),
        (16, 8),
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if config.checkpoint_schema == "base":
        assert "scaler_state" in payload and "best_validation" in payload
    elif config.checkpoint_schema == "support":
        assert "scaler_state" in payload and "trackers" in payload
    else:
        assert "scaler_state" not in payload and payload["model_family"] == "rounds32_36"
    loaded, _ = load_round_checkpoint(checkpoint, round_id=round_id)
    for expected, actual in zip(model.state_dict().values(), loaded.state_dict().values()):
        torch.testing.assert_close(actual, expected)

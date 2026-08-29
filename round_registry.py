"""Immutable historical configuration registry for experiment rounds 1--36."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping

from torch import nn
from torch.utils.data import Dataset

from admm_deep_scalar import DeepScalarADMM
from admm_guide_support_fusion import GuideSupportFusionADMM
from admm_losses import noise_aware_loss, observed_echo_loss
from admm_rounds32_36 import build_round_model
from admm_support_fusion import SupportFusionADMM
from admm_unrolled import PhysicsUnrolledADMM, Shape2D
from dense_aircraft_isar_dataset import DenseAircraftISARDataset
from structured_isar_dataset import StructuredISARDataset
from synthetic_isar_dataset import SyntheticISARDataset


@dataclass(frozen=True)
class RoundConfig:
    round_id: int
    name: str
    model_family: str
    proximal: str
    layers: int
    dataset: str
    probability: float
    train_samples: int
    validation_samples: int
    test_samples: int
    epochs: int
    batch_size: int
    gpus: int
    seed: int
    loss_family: str
    echo_weight: float
    background_weight: float
    support_weight: float
    selector: str
    checkpoint_schema: str
    visual_checkpoint: str
    run_directory: str


def _config(round_id: int, **kwargs: object) -> RoundConfig:
    defaults: dict[str, object] = {
        "name": f"round_{round_id:02d}",
        "model_family": "physics",
        "proximal": "none",
        "layers": 8,
        "dataset": "point",
        "probability": 0.0,
        "train_samples": 4000,
        "validation_samples": 500,
        "test_samples": 500,
        "epochs": 20,
        "batch_size": 2,
        "gpus": 4,
        "seed": 0,
        "loss_family": "base",
        "echo_weight": 0.5,
        "background_weight": 0.0,
        "support_weight": 0.0,
        "selector": "best",
        "checkpoint_schema": "base",
        "visual_checkpoint": "checkpoint-best.pt",
        "run_directory": f"round_{round_id:02d}",
    }
    defaults.update(kwargs)
    return RoundConfig(round_id=round_id, **defaults)  # type: ignore[arg-type]


_rounds: dict[int, RoundConfig] = {
    1: _config(1, name="Stage 1 Scalar-8", train_samples=20000, validation_samples=1000, test_samples=1000, epochs=100, gpus=8, echo_weight=0.1, visual_checkpoint="checkpoint-last.pt", run_directory="admm_stage1_full_8gpu"),
    2: _config(2, name="Stage 2 CNN", proximal="cnn", train_samples=20000, validation_samples=1000, test_samples=1000, epochs=100, gpus=8, run_directory="admm_stage2_cnnprox8"),
    3: _config(3, name="Stage 3 Transformer", proximal="transformer", train_samples=20000, validation_samples=1000, test_samples=1000, epochs=100, batch_size=1, gpus=8, run_directory="admm_stage3_transformer8"),
}

for round_id, proximal, background, run_name in (
    (4, "cnn", 0.05, "admm_stage2b_pilot_bg005"),
    (5, "cnn", 0.20, "admm_stage2b_pilot_bg020"),
    (6, "transformer", 0.05, "admm_stage3b_pilot_bg005"),
    (7, "transformer", 0.20, "admm_stage3b_pilot_bg020"),
):
    _rounds[round_id] = _config(round_id, model_family="support", proximal=proximal, batch_size=1 if proximal == "transformer" else 2, loss_family="support_l2", background_weight=background, selector="support", checkpoint_schema="support", visual_checkpoint="checkpoint-best-score.pt" if round_id == 7 else "checkpoint-best-constrained.pt", run_directory=run_name)

for round_id, proximal, background, run_name in (
    (8, "cnn", 0.05, "admm_stage2c_guide_pilot_bg005"),
    (9, "cnn", 0.20, "admm_stage2c_guide_pilot_bg020"),
    (10, "transformer", 0.05, "admm_stage3c_guide_pilot_bg005"),
    (11, "transformer", 0.20, "admm_stage3c_guide_pilot_bg020"),
):
    _rounds[round_id] = _config(round_id, model_family="guide", proximal=proximal, batch_size=1 if proximal == "transformer" else 2, loss_family="support_l1l2", background_weight=background, selector="score", checkpoint_schema="support", visual_checkpoint="checkpoint-best-score.pt", run_directory=run_name)

for round_id, layers in zip(range(12, 16), (12, 16, 24, 32)):
    _rounds[round_id] = _config(round_id, name=f"Deep Scalar-{layers}", layers=layers, run_directory=f"admm_stage1b_scalar{layers}_pilot")

for round_id, echo in zip(range(16, 20), (0.15, 0.25, 0.35, 0.45)):
    token = str(echo).replace(".", "p")
    _rounds[round_id] = _config(round_id, name=f"Balanced Scalar echo={echo}", model_family="deep_scalar", layers=12, loss_family="support_l1l2", echo_weight=echo, background_weight=0.1, selector="score", checkpoint_schema="support", visual_checkpoint="checkpoint-best-score.pt", run_directory=f"admm_stage1c_balanced_echo{token}")

for round_id, proximal, clean in (
    (20, "cnn", 0.1), (21, "cnn", 0.5), (22, "transformer", 0.1), (23, "transformer", 0.5)
):
    token = str(clean).replace(".", "p")
    _rounds[round_id] = _config(round_id, model_family="support", proximal=proximal, batch_size=1 if proximal == "transformer" else 2, loss_family="noise_aware", echo_weight=clean, background_weight=0.1, selector="support", checkpoint_schema="support", visual_checkpoint="checkpoint-best-constrained.pt", run_directory=f"admm_noiseaware_{proximal}_clean{token}")

for round_id, proximal, probability in (
    (24, "cnn", 0.50), (25, "transformer", 0.25), (26, "transformer", 0.50), (27, "transformer", 0.75)
):
    token = str(probability).replace("0.", "0p")
    _rounds[round_id] = _config(round_id, model_family="support", proximal=proximal, dataset="structured", probability=probability, batch_size=1 if proximal == "transformer" else 2, loss_family="noise_aware", echo_weight=0.1, background_weight=0.1, selector="score", checkpoint_schema="support", visual_checkpoint="checkpoint-best-score.pt", run_directory=f"admm_structured_{proximal}_p{token}")

for round_id, proximal, probability in (
    (28, "cnn", 0.25), (29, "cnn", 0.50), (30, "transformer", 0.25), (31, "transformer", 0.50)
):
    token = str(probability).replace("0.", "0p")
    _rounds[round_id] = _config(round_id, model_family="support", proximal=proximal, dataset="dense", probability=probability, batch_size=1 if proximal == "transformer" else 2, loss_family="noise_aware", echo_weight=0.1, background_weight=0.1, selector="score", checkpoint_schema="support", visual_checkpoint="checkpoint-best-score.pt", run_directory=f"admm_dense_aircraft_{proximal}_p{token}")

for round_id, variant, probability in (
    (32, "round32_guide_dc", 0.50),
    (33, "round33_confidence_band", 0.50),
    (34, "round34_stage1_residual", 0.50),
    (35, "round35_support_physics", 0.50),
    (36, "round36_support_distill", 0.75),
):
    _rounds[round_id] = _config(round_id, name=variant, model_family=variant, proximal="cnn", dataset="dense", probability=probability, seed=3200, loss_family="rounds", echo_weight=0.5, background_weight=0.2, support_weight=0.1 if round_id in (33, 36) else 0.0, selector="rounds", checkpoint_schema="rounds", visual_checkpoint="checkpoint-best.pt", run_directory=f"admm_{variant}")


ROUND_REGISTRY: Mapping[int, RoundConfig] = MappingProxyType(_rounds)


def get_round_config(round_id: int) -> RoundConfig:
    try:
        return ROUND_REGISTRY[int(round_id)]
    except (KeyError, ValueError) as error:
        raise ValueError("round_id must be between 1 and 36") from error


def build_model(
    round_id: int,
    image_shape: Shape2D = (512, 128),
    measurement_shape: Shape2D = (256, 64),
    *,
    guide_checkpoint: Path | str | None = None,
) -> nn.Module:
    config = get_round_config(round_id)
    if config.model_family.startswith("round3"):
        return build_round_model(config.model_family, image_shape, measurement_shape, guide_checkpoint=guide_checkpoint)
    if config.model_family == "support":
        return SupportFusionADMM(image_shape, measurement_shape, num_layers=config.layers, proximal=config.proximal)
    if config.model_family == "guide":
        return GuideSupportFusionADMM(image_shape, measurement_shape, num_layers=config.layers, proximal=config.proximal, guide_checkpoint=guide_checkpoint, dc_steps=2)
    if config.model_family == "deep_scalar":
        return DeepScalarADMM(image_shape, measurement_shape, num_layers=config.layers, proximal="cnn")
    return PhysicsUnrolledADMM(image_shape, measurement_shape, num_layers=config.layers, proximal=config.proximal)  # type: ignore[arg-type]


def build_dataset(
    round_id: int,
    split: Literal["train", "validation", "test"],
    *,
    image_shape: Shape2D = (512, 128),
    measurement_shape: Shape2D = (256, 64),
    samples: int | None = None,
) -> Dataset[tuple[object, object]]:
    config = get_round_config(round_id)
    offsets = {"train": 0, "validation": 1_000_000, "test": 2_000_000}
    counts = {"train": config.train_samples, "validation": config.validation_samples, "test": config.test_samples}
    common = dict(num_samples=samples or counts[split], image_shape=image_shape, measurement_shape=measurement_shape, seed=config.seed + offsets[split])
    if config.dataset == "structured":
        return StructuredISARDataset(**common, structured_probability=config.probability)
    if config.dataset == "dense":
        return DenseAircraftISARDataset(**common, structured_probability=config.probability)
    return SyntheticISARDataset(**common, cache=False)


def compute_round_loss(
    round_id: int,
    prediction: object,
    target: object,
    measurements: object,
    *,
    validation: bool = False,
    compatibility: Literal["historical", "corrected"] = "historical",
) -> dict[str, object]:
    config = get_round_config(round_id)
    echo_weight = config.echo_weight
    if validation and compatibility == "historical" and round_id in (*range(2, 4), *range(12, 16)):
        echo_weight = 0.1
    if config.loss_family == "base":
        return observed_echo_loss(prediction, target, measurements, echo_weight=echo_weight, sparse_weight=1e-4)  # type: ignore[arg-type,return-value]
    if config.loss_family.startswith("support"):
        return observed_echo_loss(prediction, target, measurements, echo_weight=echo_weight, background_weight=config.background_weight, background_l1=config.loss_family.endswith("l1l2"))  # type: ignore[arg-type,return-value]
    return noise_aware_loss(prediction, target, measurements, clean_weight=echo_weight, discrepancy_weight=0.1, background_weight=config.background_weight, support_weight=config.support_weight)  # type: ignore[arg-type,return-value]


__all__ = [
    "ROUND_REGISTRY",
    "RoundConfig",
    "build_dataset",
    "build_model",
    "compute_round_loss",
    "get_round_config",
]

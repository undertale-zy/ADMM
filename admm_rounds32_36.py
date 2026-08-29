"""The five post-history ADMM variants from experiment rounds 32--36."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from admm_guide_support_fusion import GuideSupportFusionADMM
from admm_unrolled import (
    Shape2D,
    complex_from_channels,
    complex_to_channels,
    fast_adjoint_operator,
    fast_forward_operator,
)
from reference_schedules import inverse_softplus


class _RoundMixin:
    round_variant: str

    @property
    def model_config(self) -> dict[str, object]:
        config = super().model_config  # type: ignore[misc]
        config["round_variant"] = self.round_variant
        return config


class Round32GuideDC(_RoundMixin, GuideSupportFusionADMM):
    round_variant = "round32_guide_dc"

    def __init__(
        self,
        image_shape: Shape2D,
        measurement_shape: Shape2D,
        *,
        guide_checkpoint: Path | str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(
            image_shape,
            measurement_shape,
            proximal="cnn",
            guide_checkpoint=guide_checkpoint,
            dc_steps=8,
            support_width_db=10.0,
            **kwargs,
        )


class Round33ConfidenceBand(_RoundMixin, GuideSupportFusionADMM):
    round_variant = "round33_confidence_band"

    def __init__(
        self,
        image_shape: Shape2D,
        measurement_shape: Shape2D,
        *,
        guide_checkpoint: Path | str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(
            image_shape,
            measurement_shape,
            proximal="cnn",
            guide_checkpoint=guide_checkpoint,
            dc_steps=4,
            **kwargs,
        )

    @staticmethod
    def confidence(guide_image: Tensor) -> Tensor:
        peak = torch.amax(
            torch.abs(guide_image), dim=(-2, -1), keepdim=True
        ).clamp_min(1e-12)
        db = 20.0 * torch.log10(torch.abs(guide_image) / peak + 1e-12)
        strong = (db >= -40.0).to(db.dtype)
        weak = torch.clamp((db + 60.0) / 20.0, 0.0, 1.0)
        return torch.maximum(strong, weak)

    def forward(self, measurements: Tensor) -> Tensor:
        observed = self._observed_from_channels(measurements)
        guide_image, _, learned_image, _ = self.compute_branches(observed)
        confidence = self.confidence(guide_image)
        output = guide_image + torch.sigmoid(self.raw_mix) * confidence * (
            learned_image - guide_image
        )
        for raw_step in self.raw_dc_steps:
            correction = fast_adjoint_operator(
                fast_forward_operator(output, self.measurement_shape) - observed,
                self.image_shape,
            )
            output = confidence * (
                output - torch.nn.functional.softplus(raw_step) * correction
            )
        return complex_to_channels(output)


class _ResidualCNN(nn.Module):
    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(2, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, 2, 3, padding=1),
        )
        self.scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, values: Tensor) -> Tensor:
        correction = complex_from_channels(
            self.network(complex_to_channels(values))
        )
        return torch.tanh(self.scale) * correction


class Round34Stage1Residual(_RoundMixin, GuideSupportFusionADMM):
    round_variant = "round34_stage1_residual"

    def __init__(
        self,
        image_shape: Shape2D,
        measurement_shape: Shape2D,
        *,
        guide_checkpoint: Path | str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(
            image_shape,
            measurement_shape,
            proximal="cnn",
            guide_checkpoint=guide_checkpoint,
            dc_steps=4,
            **kwargs,
        )
        self.refiner = _ResidualCNN()

    def forward(self, measurements: Tensor) -> Tensor:
        states = self.forward_states(measurements)
        output = states.guide_image + states.guide_mask * self.refiner(
            states.guide_image
        )
        return complex_to_channels(output)


class Round35SupportPhysics(_RoundMixin, GuideSupportFusionADMM):
    round_variant = "round35_support_physics"

    def __init__(
        self,
        image_shape: Shape2D,
        measurement_shape: Shape2D,
        *,
        guide_checkpoint: Path | str | None = None,
        physics_steps: int = 8,
        **kwargs: object,
    ) -> None:
        # Historical parent retains its default two registered DC parameters.
        super().__init__(
            image_shape,
            measurement_shape,
            proximal="cnn",
            guide_checkpoint=guide_checkpoint,
            dc_steps=2,
            **kwargs,
        )
        self.raw_physics_steps = nn.Parameter(torch.full((physics_steps,), -2.0))

    def forward(self, measurements: Tensor) -> Tensor:
        states = self.forward_states(measurements)
        output = states.guide_image * states.guide_mask
        for raw_step in self.raw_physics_steps:
            residual = fast_forward_operator(output, self.measurement_shape) - states.observed
            correction = fast_adjoint_operator(residual, self.image_shape)
            output = states.guide_mask * (
                output - torch.nn.functional.softplus(raw_step) * correction
            )
        return complex_to_channels(output)


class Round36SupportDistill(Round33ConfidenceBand):
    round_variant = "round36_support_distill"


ROUND_MODEL_TYPES = {
    "round32_guide_dc": Round32GuideDC,
    "round33_confidence_band": Round33ConfidenceBand,
    "round34_stage1_residual": Round34Stage1Residual,
    "round35_support_physics": Round35SupportPhysics,
    "round36_support_distill": Round36SupportDistill,
}


def build_round_model(
    variant: str,
    image_shape: Shape2D,
    measurement_shape: Shape2D,
    *,
    guide_checkpoint: Path | str | None = None,
) -> GuideSupportFusionADMM:
    try:
        model_type = ROUND_MODEL_TYPES[variant]
    except KeyError as error:
        raise ValueError(f"unknown round variant: {variant}") from error
    return model_type(
        image_shape,
        measurement_shape,
        guide_checkpoint=guide_checkpoint,
    )


__all__ = [
    "ROUND_MODEL_TYPES",
    "Round32GuideDC",
    "Round33ConfidenceBand",
    "Round34Stage1Residual",
    "Round35SupportPhysics",
    "Round36SupportDistill",
    "build_round_model",
]

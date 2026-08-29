"""Frozen Stage-1 support guide with a learned support-gated ADMM branch."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from admm_support_fusion import SupportGatedCNNProx, SupportGatedTransformerProx
from admm_unrolled import (
    PhysicsUnrolledADMM,
    Shape2D,
    complex_to_channels,
    fast_adjoint_operator,
    fast_forward_operator,
)
from reference_schedules import inverse_softplus, stage1_teacher_raw


@dataclass(frozen=True)
class GuideStates:
    observed: Tensor
    guide_image: Tensor
    guide_sparse: Tensor
    learned_image: Tensor
    learned_sparse: Tensor
    guide_mask: Tensor
    guide_clean: Tensor
    output: Tensor


def _checkpoint_raw_parameters(path: Path) -> tuple[Tensor, Tensor, Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state", payload)
    if not isinstance(state, dict):
        raise ValueError("guide checkpoint has no model_state mapping")
    values: list[Tensor] = []
    for key in ("raw_c", "raw_tau", "raw_beta"):
        value = state.get(key, state.get(f"module.{key}"))
        if not isinstance(value, Tensor):
            raise ValueError(f"guide checkpoint is missing {key}")
        values.append(value.detach().float().clone())
    return values[0], values[1], values[2]


def resolve_guide_raw(
    num_layers: int,
    checkpoint: Path | str | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Resolve guide scalars with an embedded documented schedule fallback."""

    candidate: Path | None = Path(checkpoint) if checkpoint is not None else None
    if candidate is None and os.environ.get("GUIDE_CHECKPOINT"):
        candidate = Path(os.environ["GUIDE_CHECKPOINT"])
    if candidate is None:
        historical = Path("runs/admm_stage1_full_8gpu/checkpoint-last.pt")
        if historical.is_file():
            candidate = historical
    values = _checkpoint_raw_parameters(candidate) if candidate is not None else stage1_teacher_raw()
    if any(tuple(value.shape) != (num_layers,) for value in values):
        raise ValueError("guide schedule layer count does not match model")
    return values


class GuideSupportFusionADMM(PhysicsUnrolledADMM):
    """Use frozen Stage-1 support to constrain a learned proximal branch."""

    def __init__(
        self,
        image_shape: Shape2D,
        measurement_shape: Shape2D,
        *,
        num_layers: int = 8,
        proximal: str = "cnn",
        guide_checkpoint: Path | str | None = None,
        cnn_channels: int = 32,
        cnn_depth: int = 3,
        transformer_patch: int = 8,
        transformer_dim: int = 96,
        transformer_depth: int = 2,
        transformer_heads: int = 4,
        prox_gate_threshold: float = 0.01,
        support_db_init: float = -55.0,
        support_width_db: float = 10.0,
        output_mix: float = 0.5,
        dc_steps: int = 2,
        dc_step_init: float = 0.5,
        **ignored: object,
    ) -> None:
        super().__init__(
            image_shape,
            measurement_shape,
            num_layers=num_layers,
            proximal="none",
        )
        if proximal == "cnn":
            self.proximal = SupportGatedCNNProx(
                channels=cnn_channels,
                depth=cnn_depth,
                gate_threshold=prox_gate_threshold,
            )
        elif proximal == "transformer":
            self.proximal = SupportGatedTransformerProx(
                self.image_shape,
                patch=transformer_patch,
                embed_dim=transformer_dim,
                depth=transformer_depth,
                heads=transformer_heads,
                gate_threshold=prox_gate_threshold,
            )
        else:
            raise ValueError("GuideSupport proximal must be cnn or transformer")
        self.proximal_kind = proximal  # type: ignore[assignment]
        guide_c, guide_tau, guide_beta = resolve_guide_raw(
            num_layers, guide_checkpoint
        )
        self.register_buffer("guide_raw_c", guide_c)
        self.register_buffer("guide_raw_tau", guide_tau)
        self.register_buffer("guide_raw_beta", guide_beta)
        if not -65.0 < support_db_init < -45.0:
            raise ValueError("support_db_init must be between -65 and -45 dB")
        support_fraction = torch.tensor((support_db_init + 65.0) / 20.0)
        self.raw_support_db = nn.Parameter(torch.logit(support_fraction))
        self.support_width_db = float(support_width_db)
        if not 0.0 < output_mix < 1.0:
            raise ValueError("output_mix must be between zero and one")
        self.raw_mix = nn.Parameter(torch.logit(torch.tensor(output_mix)))
        if dc_steps < 0:
            raise ValueError("dc_steps must be non-negative")
        self.dc_steps = int(dc_steps)
        self.raw_dc_steps = nn.Parameter(
            inverse_softplus(torch.full((dc_steps,), float(dc_step_init)))
        )

    @property
    def model_config(self) -> dict[str, object]:
        config = super().model_config
        config.update(
            {
                "model_family": "guide_support_fusion_admm_v1",
                "support_width_db": self.support_width_db,
                "dc_steps": self.dc_steps,
            }
        )
        return config

    @property
    def guide_parameters(self) -> tuple[Tensor, Tensor, Tensor]:
        return tuple(
            torch.nn.functional.softplus(value)
            for value in (self.guide_raw_c, self.guide_raw_tau, self.guide_raw_beta)
        )  # type: ignore[return-value]

    def firm_support(self, guide_image: Tensor) -> tuple[Tensor, Tensor]:
        low_db = -65.0 + 20.0 * torch.sigmoid(self.raw_support_db)
        high_db = low_db + self.support_width_db
        peak = torch.amax(torch.abs(guide_image), dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        low = peak * torch.pow(10.0, low_db / 20.0)
        high = peak * torch.pow(10.0, high_db / 20.0)
        mask = torch.clamp(
            (torch.abs(guide_image) - low) / (high - low + 1e-12), 0.0, 1.0
        )
        return mask, guide_image * mask

    def compute_branches(
        self, observed: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        guide_image, guide_sparse, _ = self.unroll_complex(
            observed, *self.guide_parameters, use_proximal=False
        )
        learned = self.parameters_per_layer
        learned_image, learned_sparse, _ = self.unroll_complex(
            observed, learned.c, learned.tau, learned.beta, use_proximal=True
        )
        return guide_image, guide_sparse, learned_image, learned_sparse

    def forward_states(self, measurements: Tensor) -> GuideStates:
        observed = self._observed_from_channels(measurements)
        guide_image, guide_sparse, learned_image, learned_sparse = self.compute_branches(
            observed
        )
        mask, guide_clean = self.firm_support(guide_image)
        output = guide_clean + torch.sigmoid(self.raw_mix) * mask * (
            learned_image - guide_clean
        )
        for raw_step in self.raw_dc_steps:
            correction = fast_adjoint_operator(
                fast_forward_operator(output, self.measurement_shape) - observed,
                self.image_shape,
            )
            output = mask * (
                output - torch.nn.functional.softplus(raw_step) * correction
            )
        return GuideStates(
            observed=observed,
            guide_image=guide_image,
            guide_sparse=guide_sparse,
            learned_image=learned_image,
            learned_sparse=learned_sparse,
            guide_mask=mask,
            guide_clean=guide_clean,
            output=output,
        )

    def forward(self, measurements: Tensor) -> Tensor:
        return complex_to_channels(self.forward_states(measurements).output)


__all__ = [
    "GuideStates",
    "GuideSupportFusionADMM",
    "resolve_guide_raw",
]

"""Zero-preserving support-gated proximal models for ADMM unfolding."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from admm_unrolled import (
    ComplexResidualProx,
    PhysicsUnrolledADMM,
    Shape2D,
    TransformerProx,
    complex_from_channels,
    complex_soft_threshold,
    complex_to_channels,
)
from reference_schedules import inverse_softplus


def support_gate(values: Tensor, threshold: Tensor | float) -> Tensor:
    """Return a smooth magnitude support gate with ``gate(0) == 0``."""

    threshold_tensor = torch.as_tensor(
        threshold, dtype=values.real.dtype, device=values.device
    )
    if torch.any(threshold_tensor <= 0):
        raise ValueError("support threshold must be positive")
    magnitude = torch.abs(values)
    return magnitude / (magnitude + threshold_tensor + 1e-12)


class SupportGatedCNNProx(ComplexResidualProx):
    def __init__(
        self,
        *,
        channels: int = 32,
        depth: int = 3,
        gate_threshold: float = 0.01,
    ) -> None:
        super().__init__(channels=channels, depth=depth)
        self.raw_gate_threshold = nn.Parameter(
            inverse_softplus(torch.tensor(gate_threshold))
        )

    @property
    def gate_threshold(self) -> Tensor:
        return torch.nn.functional.softplus(self.raw_gate_threshold)

    def forward(self, values: Tensor) -> Tensor:
        gate = support_gate(values, self.gate_threshold)
        return values + torch.tanh(self.scale) * gate * self.correction(values)


class SupportGatedTransformerProx(TransformerProx):
    def __init__(
        self,
        image_shape: Shape2D,
        *,
        patch: int = 8,
        embed_dim: int = 96,
        depth: int = 2,
        heads: int = 4,
        gate_threshold: float = 0.01,
    ) -> None:
        super().__init__(
            image_shape,
            patch=patch,
            embed_dim=embed_dim,
            depth=depth,
            heads=heads,
        )
        self.raw_gate_threshold = nn.Parameter(
            inverse_softplus(torch.tensor(gate_threshold))
        )

    @property
    def gate_threshold(self) -> Tensor:
        return torch.nn.functional.softplus(self.raw_gate_threshold)

    def forward(self, values: Tensor) -> Tensor:
        gate = support_gate(values, self.gate_threshold)
        return values + torch.tanh(self.scale) * gate * self.correction(values)


class SupportFusionADMM(PhysicsUnrolledADMM):
    """ADMM unfolding with zero-preserving proximal and final X/Z fusion."""

    def __init__(
        self,
        image_shape: Shape2D,
        measurement_shape: Shape2D,
        *,
        num_layers: int = 8,
        proximal: str = "cnn",
        cnn_channels: int = 32,
        cnn_depth: int = 3,
        transformer_patch: int = 8,
        transformer_dim: int = 96,
        transformer_depth: int = 2,
        transformer_heads: int = 4,
        prox_gate_threshold: float = 0.01,
        output_threshold: float = 0.001,
        output_gate_threshold: float = 0.01,
        output_mix: float = 0.5,
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
            raise ValueError("SupportFusion proximal must be cnn or transformer")
        self.proximal_kind = proximal  # type: ignore[assignment]
        self.raw_output_threshold = nn.Parameter(
            inverse_softplus(torch.tensor(output_threshold))
        )
        self.raw_output_gate_threshold = nn.Parameter(
            inverse_softplus(torch.tensor(output_gate_threshold))
        )
        if not 0.0 < output_mix < 1.0:
            raise ValueError("output_mix must be between zero and one")
        self.raw_output_mix = nn.Parameter(torch.logit(torch.tensor(output_mix)))

    @property
    def model_config(self) -> dict[str, object]:
        config = super().model_config
        config.update({"model_family": "support_fusion_admm_v1"})
        return config

    def forward(self, measurements: Tensor) -> Tensor:
        observed = self._observed_from_channels(measurements)
        params = self.parameters_per_layer
        image, sparse, _ = self.unroll_complex(
            observed, params.c, params.tau, params.beta
        )
        output_threshold = torch.nn.functional.softplus(self.raw_output_threshold)
        gate_threshold = torch.nn.functional.softplus(
            self.raw_output_gate_threshold
        )
        clean_sparse = complex_soft_threshold(sparse, output_threshold)
        mask = support_gate(clean_sparse, gate_threshold)
        mix = torch.sigmoid(self.raw_output_mix)
        output = clean_sparse + mix * mask * (image - clean_sparse)
        return complex_to_channels(output)


__all__ = [
    "SupportFusionADMM",
    "SupportGatedCNNProx",
    "SupportGatedTransformerProx",
    "support_gate",
]

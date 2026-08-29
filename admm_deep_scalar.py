"""Historical scalar-only depth and echo-weight sweep models."""

from __future__ import annotations

from admm_unrolled import PhysicsUnrolledADMM, Shape2D


class DeepScalarADMM(PhysicsUnrolledADMM):
    """Pure scalar unfolding; extra proximal arguments are intentionally ignored."""

    def __init__(
        self,
        image_shape: Shape2D,
        measurement_shape: Shape2D,
        *,
        num_layers: int = 12,
        **ignored: object,
    ) -> None:
        super().__init__(
            image_shape,
            measurement_shape,
            num_layers=num_layers,
            proximal="none",
        )


__all__ = ["DeepScalarADMM"]

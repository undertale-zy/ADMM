"""A lightweight, physics-guided ADMM unfolding network for 2-D ISAR.

The network follows the FFT form of the fixed-parameter Fast 2D-ADMM
iteration.  A complex image or echo is represented by two real channels in
the public PyTorch interface::

    measurements: [batch, 2, range_samples, azimuth_samples]
    image:        [batch, 2, range_pixels, azimuth_pixels]

Only three scalar parameters are learned at each layer: the data-consistency
step ``c``, the soft-threshold ``tau``, and the scaled-dual step ``beta``.
The FFT operators are fixed by the measurement and image sizes, so the model
keeps the reconstruction physics explicit and has very few trainable weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as torch_functional


Shape2D = tuple[int, int]


def _validate_shape(name: str, shape: Iterable[int]) -> Shape2D:
    values = tuple(shape)
    if len(values) != 2 or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in values
    ):
        raise ValueError(f"{name} must contain two positive integer dimensions")
    return values  # type: ignore[return-value]


def _validate_operator_shapes(
    image_shape: Shape2D, measurement_shape: Shape2D
) -> None:
    if any(image < measurement for image, measurement in zip(image_shape, measurement_shape)):
        raise ValueError("image_shape dimensions must not be smaller than measurements")


def complex_from_channels(values: Tensor) -> Tensor:
    """Convert ``[..., 2, H, W]`` real channels to a complex tensor."""

    if values.ndim < 3 or values.shape[-3] != 2:
        raise ValueError("expected a tensor with a two-channel dimension")
    if not torch.is_floating_point(values):
        values = values.float()
    return torch.complex(values.select(-3, 0), values.select(-3, 1))


def complex_to_channels(values: Tensor) -> Tensor:
    """Convert a complex ``[..., H, W]`` tensor to ``[..., 2, H, W]``."""

    if values.ndim < 2:
        raise ValueError("expected a tensor with at least two spatial dimensions")
    if not values.is_complex():
        values = values.to(torch.complex64)
    return torch.stack((values.real, values.imag), dim=-3)


def fast_forward_operator(
    image: Tensor,
    measurement_shape: Shape2D,
) -> Tensor:
    """Apply the normalized partial 2-D Fourier forward model.

    This is the torch equivalent of ``admm_2D_fast.m``.  It maps an image of
    shape ``[..., P, Q]`` to an echo of shape ``[..., M, N]`` where
    ``measurement_shape=(M, N)`` and ``image.shape[-2:]=(P, Q)``.
    """

    if image.ndim < 2 or not image.is_complex():
        raise ValueError("image must be a complex tensor with two spatial dimensions")
    measurement_shape = _validate_shape("measurement_shape", measurement_shape)
    image_shape = (int(image.shape[-2]), int(image.shape[-1]))
    _validate_operator_shapes(image_shape, measurement_shape)
    range_samples, azimuth_samples = measurement_shape
    range_pixels, azimuth_pixels = image_shape

    range_domain = torch.fft.ifft(image, n=range_pixels, dim=-2) * range_pixels**0.5
    range_domain = range_domain[..., :range_samples, :]
    echo = torch.fft.fft(range_domain, n=azimuth_pixels, dim=-1) / azimuth_pixels**0.5
    return echo[..., :, :azimuth_samples]


def fast_adjoint_operator(
    measurements: Tensor,
    image_shape: Shape2D,
) -> Tensor:
    """Apply the adjoint of :func:`fast_forward_operator`."""

    if measurements.ndim < 2 or not measurements.is_complex():
        raise ValueError(
            "measurements must be a complex tensor with two spatial dimensions"
        )
    image_shape = _validate_shape("image_shape", image_shape)
    measurement_shape = (int(measurements.shape[-2]), int(measurements.shape[-1]))
    _validate_operator_shapes(image_shape, measurement_shape)
    range_pixels, azimuth_pixels = image_shape

    image = torch.fft.fft(measurements, n=range_pixels, dim=-2) / range_pixels**0.5
    return torch.fft.ifft(image, n=azimuth_pixels, dim=-1) * azimuth_pixels**0.5


def complex_soft_threshold(values: Tensor, threshold: Tensor | float) -> Tensor:
    """Complex soft-thresholding with a differentiable zero-safe magnitude."""

    threshold_tensor = torch.as_tensor(threshold, dtype=values.real.dtype, device=values.device)
    if torch.any(threshold_tensor < 0):
        raise ValueError("threshold must be non-negative")
    magnitude = torch.abs(values)
    scale = torch.relu(magnitude - threshold_tensor) / (magnitude + 1e-12)
    return values * scale


def _inverse_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError("softplus initialization must be positive")
    # Stable for the small initial values used by this model.
    return float(value + torch.log(-torch.expm1(torch.tensor(-value))))


@dataclass(frozen=True)
class ADMMParameters:
    """Positive values used by one unfolded ADMM layer."""

    c: Tensor
    tau: Tensor
    beta: Tensor


class UnrolledADMM(nn.Module):
    """An ``L``-layer scalar-parameter ADMM unfolding network.

    The initial values ``c=0.5``, ``tau=0.0065`` and ``beta=1`` reproduce the
    scale of the fixed Fast 2D-ADMM implementation (with ``delta=1`` and
    ``alpha=0.0065``).  Parameters are softplus-transformed to keep them in
    their mathematically required ranges during training.
    """

    def __init__(
        self,
        image_shape: Shape2D,
        measurement_shape: Shape2D,
        *,
        num_layers: int = 8,
        init_c: float = 0.5,
        init_tau: float = 0.0065,
        init_beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.image_shape = _validate_shape("image_shape", image_shape)
        self.measurement_shape = _validate_shape("measurement_shape", measurement_shape)
        _validate_operator_shapes(self.image_shape, self.measurement_shape)
        if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers <= 0:
            raise ValueError("num_layers must be a positive integer")
        for name, value in (("init_c", init_c), ("init_tau", init_tau), ("init_beta", init_beta)):
            if not torch.isfinite(torch.tensor(value)) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        self.num_layers = num_layers
        self.raw_c = nn.Parameter(torch.full((num_layers,), _inverse_softplus(init_c)))
        self.raw_tau = nn.Parameter(torch.full((num_layers,), _inverse_softplus(init_tau)))
        self.raw_beta = nn.Parameter(torch.full((num_layers,), _inverse_softplus(init_beta)))

    @property
    def parameters_per_layer(self) -> ADMMParameters:
        """Return positive learned parameters, useful for logging and tests."""

        return ADMMParameters(
            c=torch_functional.softplus(self.raw_c),
            tau=torch_functional.softplus(self.raw_tau),
            beta=torch_functional.softplus(self.raw_beta),
        )

    def forward(self, measurements: Tensor) -> Tensor:
        """Reconstruct a batch of two-channel complex measurements."""

        if measurements.ndim != 4 or measurements.shape[1] != 2:
            raise ValueError("measurements must have shape [batch, 2, M, N]")
        if tuple(measurements.shape[-2:]) != self.measurement_shape:
            raise ValueError(
                f"measurements must have spatial shape {self.measurement_shape}, "
                f"received {tuple(measurements.shape[-2:])}"
            )
        if not torch.is_floating_point(measurements):
            measurements = measurements.float()
        observed = complex_from_channels(measurements)
        image = fast_adjoint_operator(observed, self.image_shape)
        sparse_image = torch.zeros_like(image)
        scaled_dual = torch.zeros_like(image)
        params = self.parameters_per_layer

        for layer in range(self.num_layers):
            difference = sparse_image - scaled_dual
            residual = fast_forward_operator(difference, self.measurement_shape) - observed
            next_image = difference - params.c[layer] * fast_adjoint_operator(
                residual, self.image_shape
            )
            next_sparse = complex_soft_threshold(
                next_image + scaled_dual, params.tau[layer]
            )
            next_dual = scaled_dual + params.beta[layer] * (next_image - next_sparse)
            image, sparse_image, scaled_dual = next_image, next_sparse, next_dual

        return complex_to_channels(image)


# A descriptive alias keeps imports readable for callers that prefer the name
# used in the plan document.
ADMMUnrolledNetwork = UnrolledADMM


__all__ = [
    "ADMMParameters",
    "ADMMUnrolledNetwork",
    "UnrolledADMM",
    "complex_from_channels",
    "complex_soft_threshold",
    "complex_to_channels",
    "fast_adjoint_operator",
    "fast_forward_operator",
]

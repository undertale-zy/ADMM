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
from typing import Iterable, Literal

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
    if values.dtype not in (torch.float32, torch.float64):
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


class ComplexResidualProx(nn.Module):
    """Shared residual CNN proximal used by the historical Stage 2 model."""

    def __init__(self, *, channels: int = 32, depth: int = 3) -> None:
        super().__init__()
        if channels <= 0 or depth <= 0:
            raise ValueError("channels and depth must be positive")
        layers: list[nn.Module] = [nn.Conv2d(2, channels, 3, padding=1), nn.GELU()]
        for _ in range(depth - 1):
            layers.extend((nn.Conv2d(channels, channels, 3, padding=1), nn.GELU()))
        layers.append(nn.Conv2d(channels, 2, 3, padding=1))
        self.network = nn.Sequential(*layers)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def correction(self, values: Tensor) -> Tensor:
        channels = complex_to_channels(values)
        return complex_from_channels(self.network(channels))

    def forward(self, values: Tensor) -> Tensor:
        return values + torch.tanh(self.scale) * self.correction(values)


class TransformerProx(nn.Module):
    """Patch-transformer residual proximal used by historical Stage 3."""

    def __init__(
        self,
        image_shape: Shape2D,
        *,
        patch: int = 8,
        embed_dim: int = 96,
        depth: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        image_shape = _validate_shape("image_shape", image_shape)
        if patch <= 0 or any(size % patch for size in image_shape):
            raise ValueError("image dimensions must be divisible by patch")
        if embed_dim <= 0 or depth <= 0 or heads <= 0 or embed_dim % heads:
            raise ValueError("invalid Transformer dimensions")
        self.image_shape = image_shape
        self.patch = patch
        self.embedding = nn.Conv2d(2, embed_dim, patch, stride=patch)
        token_count = (image_shape[0] // patch) * (image_shape[1] // patch)
        self.position = nn.Parameter(torch.zeros(1, token_count, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=4 * embed_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.projection = nn.ConvTranspose2d(embed_dim, 2, patch, stride=patch)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def correction(self, values: Tensor) -> Tensor:
        channels = complex_to_channels(values)
        embedded = self.embedding(channels)
        batch, width, rows, columns = embedded.shape
        tokens = embedded.flatten(2).transpose(1, 2) + self.position
        encoded = self.encoder(tokens)
        decoded = self.projection(encoded.transpose(1, 2).reshape(batch, width, rows, columns))
        return complex_from_channels(decoded)

    def forward(self, values: Tensor) -> Tensor:
        return values + torch.tanh(self.scale) * self.correction(values)


ProximalKind = Literal["none", "cnn", "transformer"]


class PhysicsUnrolledADMM(nn.Module):
    """An ``L``-layer physics-guided ADMM unfolding network.

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
        proximal: ProximalKind = "none",
        share_proximal: bool = True,
        cnn_channels: int = 32,
        cnn_depth: int = 3,
        transformer_patch: int = 8,
        transformer_dim: int = 96,
        transformer_depth: int = 2,
        transformer_heads: int = 4,
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
        if proximal not in ("none", "cnn", "transformer"):
            raise ValueError("proximal must be none, cnn, or transformer")
        self.proximal_kind = proximal
        self.share_proximal = bool(share_proximal)
        self._proximal_config = {
            "cnn_channels": cnn_channels,
            "cnn_depth": cnn_depth,
            "transformer_patch": transformer_patch,
            "transformer_dim": transformer_dim,
            "transformer_depth": transformer_depth,
            "transformer_heads": transformer_heads,
        }
        if proximal == "none":
            self.proximal: nn.Module | nn.ModuleList | None = None
        elif self.share_proximal:
            self.proximal = self._make_proximal()
        else:
            self.proximal = nn.ModuleList(
                self._make_proximal() for _ in range(self.num_layers)
            )

    @property
    def model_config(self) -> dict[str, object]:
        return {
            "image_shape": self.image_shape,
            "measurement_shape": self.measurement_shape,
            "num_layers": self.num_layers,
            "proximal": self.proximal_kind,
            "share_proximal": self.share_proximal,
            **self._proximal_config,
        }

    def _make_proximal(self) -> nn.Module:
        if self.proximal_kind == "cnn":
            return ComplexResidualProx(
                channels=int(self._proximal_config["cnn_channels"]),
                depth=int(self._proximal_config["cnn_depth"]),
            )
        return TransformerProx(
            self.image_shape,
            patch=int(self._proximal_config["transformer_patch"]),
            embed_dim=int(self._proximal_config["transformer_dim"]),
            depth=int(self._proximal_config["transformer_depth"]),
            heads=int(self._proximal_config["transformer_heads"]),
        )

    def _apply_proximal(self, values: Tensor, layer: int) -> Tensor:
        if self.proximal is None:
            return values
        if isinstance(self.proximal, nn.ModuleList):
            return self.proximal[layer](values)
        return self.proximal(values)

    @property
    def parameters_per_layer(self) -> ADMMParameters:
        """Return positive learned parameters, useful for logging and tests."""

        return ADMMParameters(
            c=torch_functional.softplus(self.raw_c),
            tau=torch_functional.softplus(self.raw_tau),
            beta=torch_functional.softplus(self.raw_beta),
        )

    def _observed_from_channels(self, measurements: Tensor) -> Tensor:
        if measurements.ndim != 4 or measurements.shape[1] != 2:
            raise ValueError("measurements must have shape [batch, 2, M, N]")
        if tuple(measurements.shape[-2:]) != self.measurement_shape:
            raise ValueError(
                f"measurements must have spatial shape {self.measurement_shape}, "
                f"received {tuple(measurements.shape[-2:])}"
            )
        if not torch.is_floating_point(measurements):
            measurements = measurements.float()
        return complex_from_channels(measurements)

    def unroll_complex(
        self,
        observed: Tensor,
        c: Tensor,
        tau: Tensor,
        beta: Tensor,
        *,
        use_proximal: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]:
        image = fast_adjoint_operator(observed, self.image_shape)
        sparse_image = torch.zeros_like(image)
        scaled_dual = torch.zeros_like(image)
        for layer in range(self.num_layers):
            difference = sparse_image - scaled_dual
            residual = fast_forward_operator(difference, self.measurement_shape) - observed
            next_image = difference - c[layer] * fast_adjoint_operator(
                residual, self.image_shape
            )
            next_sparse = complex_soft_threshold(
                next_image + scaled_dual, tau[layer]
            )
            if use_proximal:
                next_sparse = self._apply_proximal(next_sparse, layer)
            next_dual = scaled_dual + beta[layer] * (next_image - next_sparse)
            image, sparse_image, scaled_dual = next_image, next_sparse, next_dual
        return image, sparse_image, scaled_dual

    def forward(self, measurements: Tensor) -> Tensor:
        """Reconstruct a batch of two-channel complex measurements."""

        observed = self._observed_from_channels(measurements)
        params = self.parameters_per_layer
        image, _, _ = self.unroll_complex(
            observed, params.c, params.tau, params.beta
        )

        return complex_to_channels(image)


# Backward-compatible names used by the first local prototype.
UnrolledADMM = PhysicsUnrolledADMM
ADMMUnrolledNetwork = PhysicsUnrolledADMM


__all__ = [
    "ADMMParameters",
    "ADMMUnrolledNetwork",
    "ComplexResidualProx",
    "PhysicsUnrolledADMM",
    "TransformerProx",
    "UnrolledADMM",
    "complex_from_channels",
    "complex_soft_threshold",
    "complex_to_channels",
    "fast_adjoint_operator",
    "fast_forward_operator",
]

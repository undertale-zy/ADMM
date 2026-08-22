"""Two-dimensional ADMM reconstruction for sparse ISAR imaging.

This module ports the proposed algorithms from ``admm_2D.m`` and
``admm_2D_fast.m``.  The fast implementation is valid for the normalized
partial-DFT matrices used by the paper and the bundled Yak-42 example.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class ADMMResult:
    """Result and convergence diagnostics from a 2D-ADMM reconstruction."""

    image: ComplexArray
    iterations: int
    relative_change: float
    converged: bool


def soft_threshold(values: ArrayLike, threshold: float) -> ComplexArray:
    """Apply element-wise complex soft-thresholding.

    Each nonzero value keeps its phase while its magnitude is reduced by
    ``threshold``.  Values whose magnitude does not exceed the threshold are
    mapped to exactly zero.
    """

    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be a finite non-negative number")

    array = np.asarray(values, dtype=np.complex128)
    magnitude = np.abs(array)
    phase = np.zeros_like(array)
    np.divide(array, magnitude, out=phase, where=magnitude > 0)
    return phase * np.maximum(magnitude - threshold, 0.0)


def image_entropy(image: ArrayLike) -> float:
    """Return the energy-normalized image entropy used by the paper."""

    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError("image must be a two-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError("image contains NaN or infinite values")

    power = np.abs(array) ** 2
    total_energy = float(np.sum(power))
    if total_energy == 0.0:
        return 0.0

    probabilities = power[power > 0.0] / total_energy
    return float(-np.sum(probabilities * np.log(probabilities)))


def admm_2d(
    fr: ArrayLike,
    fa: ArrayLike,
    measurements: ArrayLike,
    *,
    tol: float = 1e-5,
    alpha: float = 0.0065,
    delta: float = 1.0,
    max_iterations: int = 40,
) -> ADMMResult:
    """Reconstruct a sparse image using explicit Fourier dictionaries.

    The observation model follows the MATLAB code's transposed storage
    convention::

        measurements = fr @ image @ fa.T + noise

    ``fr`` and ``fa`` must be normalized partial Fourier matrices whose rows
    are orthonormal, as required by the closed-form ADMM update in the paper.
    """

    fr_array = _as_finite_matrix("fr", fr)
    fa_array = _as_finite_matrix("fa", fa)
    observed = _as_finite_matrix("measurements", measurements)
    _validate_parameters(tol, alpha, delta, max_iterations)

    expected_shape = (fr_array.shape[0], fa_array.shape[0])
    if observed.shape != expected_shape:
        raise ValueError(
            "measurements must have shape "
            f"{expected_shape}, received {observed.shape}"
        )

    initial_image = fr_array.conj().T @ observed @ fa_array.conj()

    def residual_adjoint(image: ComplexArray) -> ComplexArray:
        residual = fr_array @ image @ fa_array.T - observed
        return fr_array.conj().T @ residual @ fa_array.conj()

    return _iterate_admm(
        initial_image,
        residual_adjoint,
        tol=tol,
        alpha=alpha,
        delta=delta,
        max_iterations=max_iterations,
    )


def admm_2d_fast(
    measurements: ArrayLike,
    image_shape: tuple[int, int],
    *,
    tol: float = 1e-5,
    alpha: float = 0.0065,
    delta: float = 1.0,
    max_iterations: int = 40,
) -> ADMMResult:
    """Reconstruct a sparse image using the paper's FFT-accelerated update.

    ``image_shape`` is ``(range_pixels, azimuth_pixels)`` and corresponds to
    ``(size(Fr, 2), size(Fa, 2))`` in ``admm_2D_fast.m``.  Unlike
    :func:`admm_2d`, this function does not accept arbitrary dictionaries: it
    implements the exact normalized DFT convention used in the Yak-42 demo.
    """

    observed = _as_finite_matrix("measurements", measurements)
    _validate_parameters(tol, alpha, delta, max_iterations)
    range_pixels, azimuth_pixels = _validate_image_shape(
        image_shape, observed.shape
    )
    range_samples, azimuth_samples = observed.shape

    temp = (
        np.fft.fft(observed, n=range_pixels, axis=0)
        / np.sqrt(range_pixels)
    )
    initial_image = (
        np.fft.ifft(temp, n=azimuth_pixels, axis=1)
        * np.sqrt(azimuth_pixels)
    )

    def residual_adjoint(image: ComplexArray) -> ComplexArray:
        projected_range = (
            np.fft.ifft(image, n=range_pixels, axis=0)
            * np.sqrt(range_pixels)
        )
        projected_range = projected_range[:range_samples, :]

        projected = (
            np.fft.fft(projected_range, axis=1)
            / np.sqrt(azimuth_pixels)
        )
        projected = projected[:, :azimuth_samples]
        residual = projected - observed

        adjoint_range = (
            np.fft.fft(residual, n=range_pixels, axis=0)
            / np.sqrt(range_pixels)
        )
        return (
            np.fft.ifft(adjoint_range, n=azimuth_pixels, axis=1)
            * np.sqrt(azimuth_pixels)
        )

    return _iterate_admm(
        initial_image,
        residual_adjoint,
        tol=tol,
        alpha=alpha,
        delta=delta,
        max_iterations=max_iterations,
    )


def _iterate_admm(
    initial_image: ComplexArray,
    residual_adjoint: Callable[[ComplexArray], ComplexArray],
    *,
    tol: float,
    alpha: float,
    delta: float,
    max_iterations: int,
) -> ADMMResult:
    image = np.asarray(initial_image, dtype=np.complex128)
    sparse_image = np.zeros_like(image)
    scaled_dual = np.zeros_like(image)
    relative_change = float("inf")
    converged = False

    for iteration in range(1, max_iterations + 1):
        split_difference = sparse_image - scaled_dual
        next_image = split_difference - (
            residual_adjoint(split_difference) / (delta + 1.0)
        )
        next_sparse = soft_threshold(
            next_image + scaled_dual,
            alpha / delta,
        )
        next_dual = scaled_dual + next_image - next_sparse

        change_norm = float(np.linalg.norm(next_image - image))
        image_norm = float(np.linalg.norm(image))
        if image_norm == 0.0:
            relative_change = 0.0 if change_norm == 0.0 else float("inf")
        else:
            relative_change = change_norm / image_norm

        # Keep the newest iterate before evaluating the paper's stop rule.
        image = next_image
        sparse_image = next_sparse
        scaled_dual = next_dual

        if relative_change <= tol:
            converged = True
            break

    return ADMMResult(
        image=image,
        iterations=iteration,
        relative_change=relative_change,
        converged=converged,
    )


def _as_finite_matrix(name: str, values: ArrayLike) -> ComplexArray:
    array = np.asarray(values, dtype=np.complex128)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def _validate_parameters(
    tol: float,
    alpha: float,
    delta: float,
    max_iterations: int,
) -> None:
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be a finite positive number")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be a finite non-negative number")
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be a finite positive number")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an integer")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")


def _validate_image_shape(
    image_shape: tuple[int, int],
    measurement_shape: tuple[int, int],
) -> tuple[int, int]:
    if len(image_shape) != 2:
        raise ValueError("image_shape must contain exactly two dimensions")
    if any(
        isinstance(size, bool) or not isinstance(size, int)
        for size in image_shape
    ):
        raise TypeError("image_shape dimensions must be integers")
    if any(size <= 0 for size in image_shape):
        raise ValueError("image_shape dimensions must be positive")
    if any(
        output < measured
        for output, measured in zip(image_shape, measurement_shape)
    ):
        raise ValueError(
            "image_shape dimensions must not be smaller than measurements"
        )
    return image_shape

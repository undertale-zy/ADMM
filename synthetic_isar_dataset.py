"""Synthetic sparse ISAR scenes for training the ADMM unfolding network.

The generator deliberately uses the same normalized partial-DFT convention as
``admm_2d_fast.py``.  Each sample is a sparse complex reflectivity image,
projected to a complex echo, and then corrupted by circular complex Gaussian
noise at a random SNR.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


Shape2D = tuple[int, int]


def _validate_shape(name: str, shape: Shape2D) -> Shape2D:
    values = tuple(shape)
    if len(values) != 2 or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in values
    ):
        raise ValueError(f"{name} must contain two positive integer dimensions")
    return values


def _validate_range(name: str, bounds: tuple[float, float]) -> tuple[float, float]:
    low, high = map(float, bounds)
    if not np.isfinite(low) or not np.isfinite(high) or low > high:
        raise ValueError(f"{name} must be two finite values in ascending order")
    return low, high


def _numpy_forward(image: np.ndarray, measurement_shape: Shape2D) -> np.ndarray:
    range_samples, azimuth_samples = measurement_shape
    range_pixels, azimuth_pixels = image.shape
    range_domain = np.fft.ifft(image, n=range_pixels, axis=0) * np.sqrt(range_pixels)
    range_domain = range_domain[:range_samples, :]
    echo = np.fft.fft(range_domain, n=azimuth_pixels, axis=1) / np.sqrt(azimuth_pixels)
    return np.asarray(echo[:, :azimuth_samples], dtype=np.complex64)


def _channels(values: np.ndarray) -> Tensor:
    return torch.from_numpy(
        np.stack((values.real.astype(np.float32), values.imag.astype(np.float32)), axis=0)
    )


def finalize_sparse_target(
    target: np.ndarray,
    measurement_shape: Shape2D,
    *,
    rng: np.random.Generator,
    snr_db_range: tuple[float, float] = (-10.0, 30.0),
) -> tuple[Tensor, Tensor, float]:
    """Peak-normalize a target, project it, and add complex Gaussian noise."""

    peak = float(np.max(np.abs(target)))
    if peak == 0.0:
        raise RuntimeError("scene generation unexpectedly produced an empty target")
    normalized = np.asarray(target / peak, dtype=np.complex64)
    clean_echo = _numpy_forward(normalized, measurement_shape)
    snr_db = float(rng.uniform(*snr_db_range))
    signal_power = float(np.mean(np.abs(clean_echo) ** 2))
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise_std = np.sqrt(noise_power / 2.0)
    noise = noise_std * (
        rng.standard_normal(measurement_shape)
        + 1j * rng.standard_normal(measurement_shape)
    )
    noisy_echo = np.asarray(clean_echo + noise, dtype=np.complex64)
    return _channels(noisy_echo), _channels(normalized), snr_db


def generate_sparse_isar_sample(
    image_shape: Shape2D,
    measurement_shape: Shape2D,
    *,
    rng: np.random.Generator,
    min_scatterers: int = 3,
    max_scatterers: int = 30,
    snr_db_range: tuple[float, float] = (-10.0, 30.0),
) -> tuple[Tensor, Tensor, float]:
    """Generate one ``(measurements, target, snr_db)`` tuple.

    Multiple scatterers may land on the same pixel; their complex amplitudes
    then add, which is a natural outcome for a discretized reflectivity map.
    The target is normalized by its peak magnitude so the network sees a
    consistent numerical scale over the SNR range.
    """

    image_shape = _validate_shape("image_shape", image_shape)
    measurement_shape = _validate_shape("measurement_shape", measurement_shape)
    if any(image < measurement for image, measurement in zip(image_shape, measurement_shape)):
        raise ValueError("image_shape dimensions must not be smaller than measurements")
    if (
        isinstance(min_scatterers, bool)
        or isinstance(max_scatterers, bool)
        or not isinstance(min_scatterers, int)
        or not isinstance(max_scatterers, int)
        or min_scatterers <= 0
        or min_scatterers > max_scatterers
    ):
        raise ValueError("scatterer limits must be positive integers with min <= max")
    snr_db_range = _validate_range("snr_db_range", snr_db_range)

    target = np.zeros(image_shape, dtype=np.complex64)
    count = int(rng.integers(min_scatterers, max_scatterers + 1))
    rows = rng.integers(0, image_shape[0], size=count)
    columns = rng.integers(0, image_shape[1], size=count)
    amplitudes = rng.uniform(0.5, 1.0, size=count)
    phases = rng.uniform(-np.pi, np.pi, size=count)
    values = amplitudes * np.exp(1j * phases)
    for row, column, value in zip(rows, columns, values):
        target[row, column] += value
    return finalize_sparse_target(
        target,
        measurement_shape,
        rng=rng,
        snr_db_range=snr_db_range,
    )


class SyntheticISARDataset(Dataset[tuple[Tensor, Tensor]]):
    """Deterministic random sparse scenes with optional in-memory caching."""

    def __init__(
        self,
        num_samples: int,
        *,
        image_shape: Shape2D = (128, 64),
        measurement_shape: Shape2D = (64, 32),
        min_scatterers: int = 3,
        max_scatterers: int = 30,
        snr_db_range: tuple[float, float] = (-10.0, 30.0),
        seed: int = 0,
        cache: bool = True,
    ) -> None:
        if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError("num_samples must be a positive integer")
        self.num_samples = num_samples
        self.image_shape = _validate_shape("image_shape", image_shape)
        self.measurement_shape = _validate_shape("measurement_shape", measurement_shape)
        if any(image < measurement for image, measurement in zip(self.image_shape, self.measurement_shape)):
            raise ValueError("image_shape dimensions must not be smaller than measurements")
        self.min_scatterers = min_scatterers
        self.max_scatterers = max_scatterers
        self.snr_db_range = _validate_range("snr_db_range", snr_db_range)
        self.seed = int(seed)
        self.cache = bool(cache)
        self._samples: list[tuple[Tensor, Tensor]] | None = None
        self._snr_db: list[float] | None = None
        if self.cache:
            self._materialize()

    def _materialize(self) -> None:
        rng = np.random.default_rng(self.seed)
        samples: list[tuple[Tensor, Tensor]] = []
        snrs: list[float] = []
        for _ in range(self.num_samples):
            measurements, target, snr_db = generate_sparse_isar_sample(
                self.image_shape,
                self.measurement_shape,
                rng=rng,
                min_scatterers=self.min_scatterers,
                max_scatterers=self.max_scatterers,
                snr_db_range=self.snr_db_range,
            )
            samples.append((measurements, target))
            snrs.append(snr_db)
        self._samples, self._snr_db = samples, snrs

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("dataset index must be an integer")
        if index < 0:
            index += self.num_samples
        if index < 0 or index >= self.num_samples:
            raise IndexError("dataset index out of range")
        if self._samples is not None:
            measurements, target = self._samples[index]
            return measurements.clone(), target.clone()

        rng = np.random.default_rng(self.seed + index)
        measurements, target, _ = generate_sparse_isar_sample(
            self.image_shape,
            self.measurement_shape,
            rng=rng,
            min_scatterers=self.min_scatterers,
            max_scatterers=self.max_scatterers,
            snr_db_range=self.snr_db_range,
        )
        return measurements, target

    def snr_db(self, index: int) -> float:
        """Return the deterministic SNR used for one sample."""

        if self._snr_db is not None:
            return self._snr_db[index]
        rng = np.random.default_rng(self.seed + index)
        _, _, snr_db = generate_sparse_isar_sample(
            self.image_shape,
            self.measurement_shape,
            rng=rng,
            min_scatterers=self.min_scatterers,
            max_scatterers=self.max_scatterers,
            snr_db_range=self.snr_db_range,
        )
        return snr_db


__all__ = [
    "SyntheticISARDataset",
    "finalize_sparse_target",
    "generate_sparse_isar_sample",
]

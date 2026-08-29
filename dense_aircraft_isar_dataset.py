"""Dense continuous-aircraft synthetic ISAR dataset (historical v2)."""

from __future__ import annotations

import numpy as np
from torch import Tensor
from torch.utils.data import Dataset

from synthetic_isar_dataset import (
    Shape2D,
    _validate_range,
    _validate_shape,
    finalize_sparse_target,
    generate_sparse_isar_sample,
)


def _gaussian_splat(
    target: np.ndarray,
    coordinate: np.ndarray,
    value: complex,
    sigma: tuple[float, float],
) -> None:
    rows, columns = target.shape
    row_radius = int(np.ceil(3.0 * sigma[0]))
    column_radius = int(np.ceil(3.0 * sigma[1]))
    center_row, center_column = coordinate
    row_low = max(0, int(np.floor(center_row)) - row_radius)
    row_high = min(rows, int(np.floor(center_row)) + row_radius + 1)
    column_low = max(0, int(np.floor(center_column)) - column_radius)
    column_high = min(columns, int(np.floor(center_column)) + column_radius + 1)
    if row_low >= row_high or column_low >= column_high:
        return
    row_grid = np.arange(row_low, row_high)[:, None]
    column_grid = np.arange(column_low, column_high)[None, :]
    exponent = -0.5 * (
        ((row_grid - center_row) / sigma[0]) ** 2
        + ((column_grid - center_column) / sigma[1]) ** 2
    )
    kernel = np.exp(exponent)
    kernel[kernel < np.exp(-4.5)] = 0.0
    target[row_low:row_high, column_low:column_high] += value * kernel


def _add_dense_line(
    target: np.ndarray,
    rng: np.random.Generator,
    start: np.ndarray,
    end: np.ndarray,
    *,
    count: int,
    amplitude_range: tuple[float, float],
    sigma: tuple[float, float],
    global_phase: float,
    phase_slope: float,
    phase_offset: float,
) -> None:
    base_amplitude = rng.uniform(*amplitude_range)
    for t in np.linspace(0.0, 1.0, count):
        if rng.random() < 0.08:
            continue
        coordinate = start + t * (end - start)
        coordinate += rng.normal(0.0, (0.45, 0.25), size=2)
        taper = 0.55 + 0.45 * np.sin(np.pi * t) ** 2
        texture = rng.lognormal(mean=0.0, sigma=0.22)
        amplitude = base_amplitude * taper * texture
        phase = (
            global_phase
            + phase_offset
            + phase_slope * (t - 0.5)
            + rng.normal(0.0, 0.10)
        )
        _gaussian_splat(target, coordinate, amplitude * np.exp(1j * phase), sigma)


def generate_dense_aircraft_sample(
    image_shape: Shape2D,
    measurement_shape: Shape2D,
    *,
    rng: np.random.Generator,
    snr_db_range: tuple[float, float] = (-10.0, 30.0),
) -> tuple[Tensor, Tensor, float]:
    rows, columns = image_shape
    center = np.array(
        [rng.uniform(0.42 * rows, 0.58 * rows), rng.uniform(0.42 * columns, 0.58 * columns)]
    )
    angle = rng.uniform(-0.32, 0.32)
    length = rng.uniform(0.16 * rows, 0.28 * rows)
    span = rng.uniform(0.22 * columns, 0.42 * columns)
    global_phase = rng.uniform(-np.pi, np.pi)
    phase_slope = rng.uniform(-1.2, 1.2)
    axis = np.array([np.cos(angle), np.sin(angle)])
    wing = np.array([-np.sin(angle), np.cos(angle)])
    target = np.zeros(image_shape, dtype=np.complex64)

    _add_dense_line(
        target,
        rng,
        center - 0.50 * length * axis,
        center + 0.50 * length * axis,
        count=100,
        amplitude_range=(0.10, 0.24),
        sigma=(2.5, 1.25),
        global_phase=global_phase,
        phase_slope=phase_slope,
        phase_offset=0.0,
    )
    wing_center = center - 0.07 * length * axis
    _add_dense_line(
        target,
        rng,
        wing_center - 0.50 * span * wing,
        wing_center + 0.50 * span * wing,
        count=70,
        amplitude_range=(0.06, 0.18),
        sigma=(2.0, 1.35),
        global_phase=global_phase,
        phase_slope=phase_slope,
        phase_offset=0.5,
    )
    tail_center = center + 0.38 * length * axis
    _add_dense_line(
        target,
        rng,
        tail_center - 0.24 * span * wing,
        tail_center + 0.24 * span * wing,
        count=36,
        amplitude_range=(0.05, 0.15),
        sigma=(1.8, 1.15),
        global_phase=global_phase,
        phase_slope=phase_slope,
        phase_offset=-0.4,
    )

    landmarks = (
        center - 0.49 * length * axis,
        center + 0.49 * length * axis,
        center - 0.07 * length * axis - 0.25 * span * wing,
        center - 0.07 * length * axis + 0.25 * span * wing,
        center - 0.07 * length * axis - 0.49 * span * wing,
        center - 0.07 * length * axis + 0.49 * span * wing,
    )
    for landmark in landmarks:
        coordinate = landmark + rng.normal(0.0, (0.5, 0.3), size=2)
        amplitude = rng.uniform(0.55, 1.0)
        phase = global_phase + rng.normal(0.0, 0.35)
        _gaussian_splat(
            target, coordinate, amplitude * np.exp(1j * phase), (1.15, 0.85)
        )
    return finalize_sparse_target(
        target, measurement_shape, rng=rng, snr_db_range=snr_db_range
    )


class DenseAircraftISARDataset(Dataset[tuple[Tensor, Tensor]]):
    """Mix point-only and Dense-v2 scenes using the historical RNG order."""

    def __init__(
        self,
        num_samples: int,
        *,
        image_shape: Shape2D = (512, 128),
        measurement_shape: Shape2D = (256, 64),
        structured_probability: float = 0.5,
        snr_db_range: tuple[float, float] = (-10.0, 30.0),
        seed: int = 0,
        **ignored: object,
    ) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if not 0.0 <= structured_probability <= 1.0:
            raise ValueError("structured_probability must be in [0, 1]")
        self.num_samples = int(num_samples)
        self.image_shape = _validate_shape("image_shape", image_shape)
        self.measurement_shape = _validate_shape("measurement_shape", measurement_shape)
        self.structured_probability = float(structured_probability)
        self.snr_db_range = _validate_range("snr_db_range", snr_db_range)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        if index < 0:
            index += self.num_samples
        if not 0 <= index < self.num_samples:
            raise IndexError("dataset index out of range")
        rng = np.random.default_rng(self.seed + index)
        dense = rng.random() < self.structured_probability
        generator = generate_dense_aircraft_sample if dense else generate_sparse_isar_sample
        measurements, target, _ = generator(
            self.image_shape,
            self.measurement_shape,
            rng=rng,
            snr_db_range=self.snr_db_range,
        )
        return measurements, target


__all__ = ["DenseAircraftISARDataset", "generate_dense_aircraft_sample"]

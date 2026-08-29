"""Point/structured-aircraft mixed synthetic ISAR dataset (historical v1)."""

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


def _line_points(
    rng: np.random.Generator,
    start: np.ndarray,
    end: np.ndarray,
    count: int,
    coords: list[np.ndarray],
) -> None:
    for t in np.linspace(0.0, 1.0, count):
        if rng.random() < 0.18:
            continue
        coordinate = start + t * (end - start)
        coordinate = coordinate + rng.normal(0.0, (1.5, 0.7), size=2)
        coords.append(coordinate)


def generate_structured_aircraft_sample(
    image_shape: Shape2D,
    measurement_shape: Shape2D,
    *,
    rng: np.random.Generator,
    snr_db_range: tuple[float, float] = (-10.0, 30.0),
) -> tuple[Tensor, Tensor, float]:
    """Generate the discrete-line aircraft target used in rounds 24--27."""

    rows, columns = image_shape
    center = np.array(
        [rng.uniform(0.4 * rows, 0.6 * rows), rng.uniform(0.4 * columns, 0.6 * columns)]
    )
    angle = rng.uniform(-0.35, 0.35)
    length = rng.uniform(0.14 * rows, 0.27 * rows)
    span = rng.uniform(0.18 * columns, 0.38 * columns)
    axis = np.array([np.cos(angle), np.sin(angle)])
    wing = np.array([-np.sin(angle), np.cos(angle)])
    coords: list[np.ndarray] = []

    _line_points(
        rng, center - 0.5 * length * axis, center + 0.5 * length * axis, 35, coords
    )
    wing_center = center - 0.05 * length * axis
    _line_points(
        rng, wing_center - 0.5 * span * wing, wing_center + 0.5 * span * wing, 25, coords
    )
    tail_center = center + 0.38 * length * axis
    _line_points(
        rng, tail_center - 0.22 * span * wing, tail_center + 0.22 * span * wing, 13, coords
    )
    for sign in (-1.0, 1.0):
        hub = center - 0.05 * length * axis + sign * 0.23 * span * wing
        for _ in range(5):
            coords.append(hub + rng.normal(0.0, (2.0, 1.0), size=2))

    target = np.zeros(image_shape, dtype=np.complex64)
    for coordinate in coords:
        amplitude = rng.uniform(0.35, 1.0)
        phase = rng.uniform(-np.pi, np.pi)
        row, column = np.rint(coordinate).astype(int)
        if 0 <= row < rows and 0 <= column < columns:
            target[row, column] += amplitude * np.exp(1j * phase)
    return finalize_sparse_target(
        target, measurement_shape, rng=rng, snr_db_range=snr_db_range
    )


class StructuredISARDataset(Dataset[tuple[Tensor, Tensor]]):
    """Mix point-only and Structured-v1 scenes using one per-index RNG."""

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
        structured = rng.random() < self.structured_probability
        generator = (
            generate_structured_aircraft_sample
            if structured
            else generate_sparse_isar_sample
        )
        measurements, target, _ = generator(
            self.image_shape,
            self.measurement_shape,
            rng=rng,
            snr_db_range=self.snr_db_range,
        )
        return measurements, target


StructuredAircraftISARDataset = StructuredISARDataset


__all__ = [
    "StructuredAircraftISARDataset",
    "StructuredISARDataset",
    "generate_structured_aircraft_sample",
]

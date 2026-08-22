from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from admm_2d import admm_2d, admm_2d_fast, image_entropy, soft_threshold
from yak42_admm_demo import load_yak42, prepare_measurements


MODULE_DIR = Path(__file__).resolve().parents[1]


def normalized_dft_rows(
    samples: int,
    pixels: int,
    *,
    sign: int,
) -> np.ndarray:
    sample_indices = np.arange(samples)
    pixel_indices = np.arange(pixels)
    return np.exp(
        sign
        * 1j
        * 2.0
        * np.pi
        * np.outer(sample_indices, pixel_indices)
        / pixels
    ) / np.sqrt(pixels)


def test_fast_matches_explicit_dft_implementation() -> None:
    range_samples, azimuth_samples = 6, 5
    range_pixels, azimuth_pixels = 10, 8
    fr = normalized_dft_rows(range_samples, range_pixels, sign=1)
    fa = normalized_dft_rows(azimuth_samples, azimuth_pixels, sign=-1)

    scene = np.zeros((range_pixels, azimuth_pixels), dtype=np.complex128)
    scene[2, 3] = 1.0 + 0.25j
    scene[7, 5] = 0.7 - 0.1j
    measurements = fr @ scene @ fa.T

    explicit = admm_2d(
        fr,
        fa,
        measurements,
        tol=1e-14,
        alpha=0.01,
        max_iterations=8,
    )
    fast = admm_2d_fast(
        measurements,
        (range_pixels, azimuth_pixels),
        tol=1e-14,
        alpha=0.01,
        max_iterations=8,
    )

    np.testing.assert_allclose(
        fast.image,
        explicit.image,
        rtol=1e-11,
        atol=1e-12,
    )
    assert fast.iterations == explicit.iterations
    assert fast.converged == explicit.converged


def test_soft_threshold_preserves_complex_phase_and_handles_zero() -> None:
    values = np.array([0.0, 3.0 + 4.0j, 1.0j])

    thresholded = soft_threshold(values, 2.0)

    np.testing.assert_allclose(thresholded, [0.0, 1.8 + 2.4j, 0.0])


def test_zero_measurements_converge_without_nan() -> None:
    result = admm_2d_fast(
        np.zeros((3, 2), dtype=np.complex128),
        (6, 4),
    )

    assert result.converged
    assert result.iterations == 1
    assert result.relative_change == 0.0
    assert np.all(result.image == 0.0)
    assert image_entropy(result.image) == 0.0


def test_converged_iteration_returns_the_newest_image() -> None:
    identity = np.eye(2, dtype=np.complex128)
    measurements = np.ones((2, 2), dtype=np.complex128)

    result = admm_2d(
        identity,
        identity,
        measurements,
        tol=1.0,
        alpha=0.0,
        delta=1.0,
        max_iterations=2,
    )

    assert result.converged
    assert result.iterations == 1
    np.testing.assert_allclose(result.image, 0.5 * measurements)


@pytest.mark.parametrize(
    ("measurements", "image_shape", "message"),
    [
        (np.zeros((3, 2)), (2, 4), "must not be smaller"),
        (np.zeros((3, 2)), (6, 0), "must be positive"),
    ],
)
def test_fast_rejects_invalid_image_shapes(
    measurements: np.ndarray,
    image_shape: tuple[int, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        admm_2d_fast(measurements, image_shape)


def test_explicit_version_rejects_mismatched_measurements() -> None:
    fr = np.zeros((3, 6), dtype=np.complex128)
    fa = np.zeros((2, 4), dtype=np.complex128)

    with pytest.raises(ValueError, match="measurements must have shape"):
        admm_2d(fr, fa, np.zeros((3, 3), dtype=np.complex128))


def test_yak42_preprocessing_shapes_are_preserved() -> None:
    raw = load_yak42(MODULE_DIR / "Yak42.mat")
    measurements, snr_db = prepare_measurements(raw, seed=0)

    assert raw.shape == (256, 256)
    assert measurements.shape == (256, 64)
    assert (2 * measurements.shape[0], 2 * measurements.shape[1]) == (512, 128)
    assert np.isfinite(snr_db)

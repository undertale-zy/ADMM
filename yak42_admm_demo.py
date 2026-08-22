"""Run the proposed Fast 2D-ADMM algorithm on the Yak-42 data set."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat

from admm_2d import ADMMResult, admm_2d_fast, image_entropy


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ComplexArray = NDArray[np.complex128]

DEFAULT_DATA_PATH = Path(__file__).with_name("Yak42.mat")
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("yak42_admm_fast.png")
DEFAULT_ALPHA = 0.0065
DEFAULT_TOL = 1e-5
DEFAULT_MAX_ITERATIONS = 40
DEFAULT_SEED = 0
DEFAULT_PRF = 100.0
RANGE_SPACING_METERS = 0.375
NOISE_STD = 0.505 * 0.03 * np.sqrt(2.0)


def load_yak42(path: Path) -> ComplexArray:
    """Load and validate the complex ``y`` matrix from ``Yak42.mat``."""

    if not path.is_file():
        raise FileNotFoundError(f"Yak-42 data file not found: {path}")

    contents = loadmat(path)
    if "y" not in contents:
        raise KeyError(f"MAT file does not contain a 'y' variable: {path}")

    measurements = np.asarray(contents["y"], dtype=np.complex128)
    if measurements.ndim != 2:
        raise ValueError("Yak-42 variable 'y' must be a two-dimensional matrix")
    if measurements.shape[1] < 192:
        raise ValueError(
            "Yak-42 variable 'y' must contain at least 192 azimuth columns"
        )
    if not np.all(np.isfinite(measurements)):
        raise ValueError("Yak-42 measurements contain NaN or infinite values")
    return measurements


def prepare_measurements(
    raw_measurements: ComplexArray,
    *,
    seed: int = DEFAULT_SEED,
) -> tuple[ComplexArray, float]:
    """Apply the Yak-42 slicing, normalization, noise, and IFFT steps."""

    # MATLAB columns 129:192 correspond to the half-open Python slice 128:192.
    selected = np.array(raw_measurements[:, 128:192], copy=True)
    peak_magnitude = float(np.max(np.abs(selected)))
    if peak_magnitude == 0.0:
        raise ValueError("selected Yak-42 measurements contain no signal energy")
    selected /= peak_magnitude

    rng = np.random.default_rng(seed)
    noise = NOISE_STD * (
        rng.standard_normal(selected.shape)
        + 1j * rng.standard_normal(selected.shape)
    )
    noise_variance = float(np.var(noise, ddof=1))
    signal_energy = float(np.linalg.norm(selected) ** 2)
    snr_db = 10.0 * np.log10(
        signal_energy / (noise_variance * selected.size)
    )

    noisy_measurements = selected + noise
    range_azimuth_signal = np.fft.ifft(noisy_measurements, axis=0)
    return np.asarray(range_azimuth_signal, dtype=np.complex128), float(snr_db)


def save_reconstruction(
    result: ADMMResult,
    output_path: Path,
    *,
    snr_db: float,
    prf: float = DEFAULT_PRF,
    range_spacing: float = RANGE_SPACING_METERS,
) -> None:
    """Save the paper-style contour plot for one ADMM reconstruction."""

    range_pixels, azimuth_pixels = result.image.shape
    doppler_axis = np.linspace(-prf / 2.0, prf / 2.0, azimuth_pixels)
    range_axis = np.linspace(
        -(range_pixels / 2.0) * range_spacing / 2.0,
        (range_pixels / 2.0) * range_spacing / 2.0,
        range_pixels,
    )
    shifted_magnitude = np.abs(np.fft.fftshift(result.image, axes=1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.4, 7.0), constrained_layout=True)
    axis.contour(
        doppler_axis,
        range_axis,
        shifted_magnitude,
        levels=40,
        cmap="viridis",
        linewidths=0.8,
    )
    axis.set_title(f"Yak-42 Fast 2D-ADMM | SNR {snr_db:.1f} dB")
    axis.set_xlabel("Doppler (Hz)")
    axis.set_ylabel("Range (m)")
    axis.set_xlim(-40.0, 40.0)
    axis.set_ylim(-35.0, 35.0)
    axis.grid(True, color="#d8dfe1", linewidth=0.6)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct the bundled Yak-42 data with Fast 2D-ADMM."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"input MAT file (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"output PNG file (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL)
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_measurements = load_yak42(args.data)
    measurements, snr_db = prepare_measurements(
        raw_measurements,
        seed=args.seed,
    )
    image_shape = (2 * measurements.shape[0], 2 * measurements.shape[1])

    start = time.perf_counter()
    result = admm_2d_fast(
        measurements,
        image_shape,
        tol=args.tol,
        alpha=args.alpha,
        delta=1.0,
        max_iterations=args.max_iterations,
    )
    elapsed_seconds = time.perf_counter() - start

    save_reconstruction(result, args.output, snr_db=snr_db)
    entropy = image_entropy(result.image)

    print(f"data: {args.data}")
    print(f"measurement_shape: {measurements.shape}")
    print(f"image_shape: {result.image.shape}")
    print(f"snr_db: {snr_db:.6f}")
    print(f"elapsed_seconds: {elapsed_seconds:.6f}")
    print(f"iterations: {result.iterations}")
    print(f"relative_change: {result.relative_change:.6e}")
    print(f"converged: {result.converged}")
    print(f"image_entropy: {entropy:.6f}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()

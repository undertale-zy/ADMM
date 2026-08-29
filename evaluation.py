"""Shared synthetic and Yak-42 evaluation helpers for all experiment rounds."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from admm_2d import admm_2d_fast, image_entropy
from admm_losses import background_penalty, normalized_nmse
from admm_unrolled import (
    complex_from_channels,
    complex_to_channels,
    fast_forward_operator,
)
from round_registry import build_model
from yak42_admm_demo import load_yak42, prepare_measurements


def load_round_checkpoint(
    checkpoint: Path,
    *,
    round_id: int | None = None,
    device: torch.device | str = "cpu",
    guide_checkpoint: Path | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = dict(payload.get("model_config", {}))
    resolved_round = round_id or config.get("round_id")
    if resolved_round is None:
        raise ValueError("round_id is required for a checkpoint without registry metadata")
    image_shape = tuple(config.get("image_shape", payload.get("image_shape", (512, 128))))
    measurement_shape = tuple(
        config.get("measurement_shape", payload.get("measurement_shape", (256, 64)))
    )
    model = build_model(
        int(resolved_round),
        image_shape,  # type: ignore[arg-type]
        measurement_shape,  # type: ignore[arg-type]
        guide_checkpoint=guide_checkpoint,
    )
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    return model, payload


def model_latency(
    model: nn.Module,
    measurement_shape: tuple[int, int],
    device: torch.device,
    *,
    warmup: int = 5,
    repeats: int = 30,
) -> float:
    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    sample = torch.zeros((1, 2, *measurement_shape), device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(sample)
        synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            model(sample)
        synchronize()
    return 1000.0 * (time.perf_counter() - start) / repeats


def evaluate_dataset(
    model: nn.Module,
    dataset: Dataset,
    *,
    device: torch.device,
    batch_size: int = 2,
) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    rows: list[dict[str, float]] = []
    with torch.no_grad():
        for measurements, target in loader:
            measurements = measurements.to(device)
            target = target.to(device)
            prediction = model(measurements)
            predicted = complex_from_channels(prediction)
            truth = complex_from_channels(target)
            observed = complex_from_channels(measurements)
            shape = (int(observed.shape[-2]), int(observed.shape[-1]))
            predicted_echo = fast_forward_operator(predicted, shape)
            clean_echo = fast_forward_operator(truth, shape)
            observed_energy = torch.sum(torch.abs(observed) ** 2, dim=(-2, -1)).clamp_min(1e-8)
            true_noise = torch.mean(
                torch.sum(torch.abs(observed - clean_echo) ** 2, dim=(-2, -1))
                / observed_energy
            )
            residual = torch.mean(
                torch.sum(torch.abs(predicted_echo - observed) ** 2, dim=(-2, -1))
                / observed_energy
            )
            rows.append(
                {
                    "image_nmse": float(normalized_nmse(predicted, truth).cpu()),
                    "clean_echo_nmse": float(normalized_nmse(predicted_echo, clean_echo).cpu()),
                    "observed_residual": float(residual.cpu()),
                    "true_noise_residual": float(true_noise.cpu()),
                    "discrepancy": float(((residual - true_noise) ** 2).cpu()),
                    "background": float(background_penalty(predicted, truth, include_l1=True).cpu()),
                }
            )
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def image_db_metrics(image: np.ndarray) -> dict[str, float]:
    magnitude = np.abs(image)
    peak = max(float(np.max(magnitude)), 1e-15)
    db = 20.0 * np.log10(magnitude / peak + 1e-15)
    band = (db >= -60.0) & (db < -40.0)
    energy = magnitude**2
    total_energy = max(float(np.sum(energy)), 1e-30)
    return {
        "entropy": image_entropy(image),
        "band_40_60_pixel_fraction": float(np.mean(band)),
        "band_40_60_energy_fraction": float(np.sum(energy[band]) / total_energy),
        "support_above_40_fraction": float(np.mean(db >= -40.0)),
        "support_above_60_fraction": float(np.mean(db >= -60.0)),
    }


def _observed_residual(image: np.ndarray, measurements: np.ndarray) -> float:
    image_tensor = torch.from_numpy(np.asarray(image, dtype=np.complex64))[None]
    predicted = fast_forward_operator(image_tensor, measurements.shape)[0].numpy()
    return float(
        np.sum(np.abs(predicted - measurements) ** 2)
        / max(float(np.sum(np.abs(measurements) ** 2)), 1e-8)
    )


def infer_yak42(
    model: nn.Module,
    data_path: Path,
    *,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    raw = load_yak42(data_path)
    measurements, snr_db = prepare_measurements(raw, seed=seed)
    channels = complex_to_channels(
        torch.from_numpy(measurements.astype(np.complex64))
    )[None].to(device)
    with torch.no_grad():
        prediction = model(channels)
    image = complex_from_channels(prediction[0]).cpu().numpy()
    metrics = image_db_metrics(image)
    metrics.update(
        {
            "seed": float(seed),
            "snr_db": float(snr_db),
            "observed_residual": _observed_residual(image, measurements),
        }
    )
    return image, measurements, metrics


def infer_fast_yak42(
    measurements: np.ndarray,
    *,
    image_shape: tuple[int, int] = (512, 128),
) -> tuple[np.ndarray, dict[str, float]]:
    start = time.perf_counter()
    result = admm_2d_fast(
        measurements,
        image_shape,
        tol=1e-5,
        alpha=0.0065,
        delta=1.0,
        max_iterations=40,
    )
    metrics = image_db_metrics(result.image)
    metrics.update(
        {
            "observed_residual": _observed_residual(result.image, measurements),
            "elapsed_seconds": time.perf_counter() - start,
            "iterations": float(result.iterations),
        }
    )
    return result.image, metrics


def save_db_panels(
    images: list[np.ndarray],
    titles: list[str],
    output_path: Path,
    *,
    dpi: int = 180,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(images), figsize=(6.3 * len(images), 5.2), squeeze=False, constrained_layout=True)
    for axis, image, title in zip(axes[0], images, titles):
        shifted = np.abs(np.fft.fftshift(image, axes=1))
        peak = max(float(np.max(shifted)), 1e-12)
        db = np.clip(20.0 * np.log10(shifted / peak + 1e-12), -60.0, 0.0)
        shown = axis.imshow(db, cmap="viridis", vmin=-60, vmax=0, origin="lower", aspect="auto", extent=(-50, 50, -48, 48))
        axis.set_xlim(-40, 40)
        axis.set_ylim(-35, 35)
        axis.set_title(title)
        axis.set_xlabel("Doppler (Hz)")
        axis.set_ylabel("Range (m)")
        figure.colorbar(shown, ax=axis, label="Normalized magnitude (dB)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


__all__ = [
    "evaluate_dataset",
    "image_db_metrics",
    "infer_fast_yak42",
    "infer_yak42",
    "load_round_checkpoint",
    "model_latency",
    "save_db_panels",
]

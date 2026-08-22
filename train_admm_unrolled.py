"""Train and evaluate the scalar-parameter ADMM unfolding network."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from admm_2d import admm_2d_fast, image_entropy
from admm_unrolled import (
    UnrolledADMM,
    complex_from_channels,
    complex_to_channels,
    fast_forward_operator,
)
from synthetic_isar_dataset import SyntheticISARDataset


def _sample_normalized_mse(prediction: Tensor, target: Tensor) -> Tensor:
    error = torch.sum(torch.abs(prediction - target) ** 2, dim=(-2, -1))
    energy = torch.sum(torch.abs(target) ** 2, dim=(-2, -1)).clamp_min(1e-8)
    return torch.mean(error / energy)


def compute_loss(
    prediction: Tensor,
    target: Tensor,
    measurements: Tensor,
    *,
    echo_weight: float = 0.1,
    sparse_weight: float = 1e-4,
) -> tuple[Tensor, dict[str, float]]:
    """Return the normalized image/echo/sparsity training objective."""

    predicted_image = complex_from_channels(prediction)
    target_image = complex_from_channels(target)
    observed = complex_from_channels(measurements)
    image_loss = _sample_normalized_mse(predicted_image, target_image)
    predicted_echo = fast_forward_operator(
        predicted_image, (int(observed.shape[-2]), int(observed.shape[-1]))
    )
    echo_error = torch.sum(torch.abs(predicted_echo - observed) ** 2, dim=(-2, -1))
    echo_energy = torch.sum(torch.abs(observed) ** 2, dim=(-2, -1)).clamp_min(1e-8)
    echo_loss = torch.mean(echo_error / echo_energy)
    sparse_loss = torch.mean(torch.abs(predicted_image))
    total = image_loss + echo_weight * echo_loss + sparse_weight * sparse_loss
    return total, {
        "image": float(image_loss.detach().cpu()),
        "echo": float(echo_loss.detach().cpu()),
        "sparse": float(sparse_loss.detach().cpu()),
        "total": float(total.detach().cpu()),
    }


def _select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return torch.device("mps")
    if requested != "auto":
        raise ValueError("device must be auto, cpu, cuda, or mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {"image": float("nan"), "echo": float("nan"), "sparse": float("nan"), "total": float("nan")}
    return {key: float(np.mean([item[key] for item in metrics])) for key in metrics[0]}


def run_epoch(
    model: UnrolledADMM,
    loader: DataLoader[tuple[Tensor, Tensor]],
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    metrics: list[dict[str, float]] = []
    for measurements, target in loader:
        measurements = measurements.to(device)
        target = target.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            prediction = model(measurements)
            loss, parts = compute_loss(prediction, target, measurements)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
        metrics.append(parts)
    return _mean_metrics(metrics)


def evaluate_fixed_fast_admm(
    dataset: SyntheticISARDataset,
    *,
    max_iterations: int = 40,
) -> dict[str, float]:
    """Evaluate the fixed Fast 2D-ADMM on the same synthetic test scenes.

    This is the internal physics baseline for the unfolding experiment.  No
    2D-FFT, SL0, or GP-SOONE comparison implementation is called here.
    """

    image_errors: list[float] = []
    entropies: list[float] = []
    for measurements, target in dataset:
        observed = measurements[0].numpy() + 1j * measurements[1].numpy()
        target_image = target[0].numpy() + 1j * target[1].numpy()
        result = admm_2d_fast(
            observed,
            dataset.image_shape,
            tol=1e-5,
            alpha=0.0065,
            delta=1.0,
            max_iterations=max_iterations,
        )
        target_energy = max(float(np.sum(np.abs(target_image) ** 2)), 1e-8)
        image_errors.append(float(np.sum(np.abs(result.image - target_image) ** 2) / target_energy))
        entropies.append(image_entropy(result.image))
    return {
        "image_nmse": float(np.mean(image_errors)),
        "image_entropy": float(np.mean(entropies)),
        "iterations": float(max_iterations),
    }


def save_checkpoint(
    path: Path,
    model: UnrolledADMM,
    optimizer: torch.optim.Optimizer,
    history: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "image_shape": model.image_shape,
        "measurement_shape": model.measurement_shape,
        "num_layers": model.num_layers,
        "history": history,
    }
    torch.save(payload, path)


def train_model(
    *,
    image_shape: tuple[int, int] = (128, 64),
    measurement_shape: tuple[int, int] = (64, 32),
    train_samples: int = 1024,
    validation_samples: int = 128,
    test_samples: int = 128,
    batch_size: int = 8,
    epochs: int = 20,
    num_layers: int = 8,
    learning_rate: float = 2e-3,
    seed: int = 0,
    device: str = "auto",
    output_dir: Path = Path("outputs/admm_unrolled"),
    cache: bool = True,
) -> tuple[UnrolledADMM, list[dict[str, Any]]]:
    """Train the first-stage network and persist its checkpoint and history."""

    if batch_size <= 0 or epochs <= 0 or learning_rate <= 0:
        raise ValueError("batch_size, epochs, and learning_rate must be positive")
    torch.manual_seed(seed)
    np.random.seed(seed)
    selected_device = _select_device(device)
    train_set = SyntheticISARDataset(
        train_samples,
        image_shape=image_shape,
        measurement_shape=measurement_shape,
        seed=seed,
        cache=cache,
    )
    validation_set = SyntheticISARDataset(
        validation_samples,
        image_shape=image_shape,
        measurement_shape=measurement_shape,
        seed=seed + 1_000_000,
        cache=cache,
    )
    test_set = SyntheticISARDataset(
        test_samples,
        image_shape=image_shape,
        measurement_shape=measurement_shape,
        seed=seed + 2_000_000,
        cache=cache,
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = UnrolledADMM(image_shape, measurement_shape, num_layers=num_layers).to(selected_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {selected_device}")
    print(f"image_shape: {image_shape}")
    print(f"measurement_shape: {measurement_shape}")
    print(f"train/validation/test: {train_samples}/{validation_samples}/{test_samples}")

    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, device=selected_device, optimizer=optimizer)
        with torch.no_grad():
            validation_metrics = run_epoch(
                model, validation_loader, device=selected_device, optimizer=None
            )
        row: dict[str, Any] = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
            "elapsed_seconds": time.perf_counter() - start,
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} | train {train_metrics['total']:.6f} | "
            f"validation {validation_metrics['total']:.6f} | "
            f"seconds {row['elapsed_seconds']:.2f}"
        )

    with torch.no_grad():
        test_metrics = run_epoch(model, test_loader, device=selected_device, optimizer=None)
    print(f"test: {json.dumps(test_metrics, sort_keys=True)}")
    fixed_metrics = evaluate_fixed_fast_admm(test_set)
    print(f"fixed_fast_admm: {json.dumps(fixed_metrics, sort_keys=True)}")
    history.append({"test": test_metrics, "fixed_fast_admm": fixed_metrics})
    save_checkpoint(output_dir / "admm_unrolled_checkpoint.pt", model, optimizer, history)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    learned = model.parameters_per_layer
    np.savetxt(
        output_dir / "learned_parameters.csv",
        torch.stack((learned.c, learned.tau, learned.beta), dim=1).detach().cpu().numpy(),
        delimiter=",",
        header="c,tau,beta",
        comments="",
    )
    return model, history


def _save_yak42_plot(image: np.ndarray, output_path: Path, snr_db: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    range_pixels, azimuth_pixels = image.shape
    shifted_magnitude = np.abs(np.fft.fftshift(image, axes=1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.4, 7.0), constrained_layout=True)
    axis.contour(
        np.linspace(-50.0, 50.0, azimuth_pixels),
        np.linspace(-35.0, 35.0, range_pixels),
        shifted_magnitude,
        levels=40,
        cmap="viridis",
        linewidths=0.8,
    )
    axis.set_title(f"Yak-42 ADMM unfolding | SNR {snr_db:.1f} dB")
    axis.set_xlabel("Doppler (Hz)")
    axis.set_ylabel("Range (m)")
    axis.grid(True, color="#d8dfe1", linewidth=0.6)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def evaluate_yak42(
    checkpoint_path: Path,
    *,
    data_path: Path,
    output_path: Path,
    device: str = "auto",
    seed: int = 0,
) -> dict[str, Any]:
    """Run a trained scalar-parameter network on the bundled Yak-42 echo."""

    from yak42_admm_demo import load_yak42, prepare_measurements

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    selected_device = _select_device(device)
    raw = load_yak42(data_path)
    measurements, snr_db = prepare_measurements(raw, seed=seed)
    measurement_shape = tuple(int(size) for size in measurements.shape)
    image_shape = (2 * measurement_shape[0], 2 * measurement_shape[1])
    model = UnrolledADMM(
        image_shape,
        measurement_shape,
        num_layers=int(payload["num_layers"]),
    )
    model.load_state_dict(payload["model_state"])
    model.to(selected_device).eval()
    channels = complex_to_channels(torch.from_numpy(measurements.astype(np.complex64)))[None].to(selected_device)
    start = time.perf_counter()
    with torch.no_grad():
        output = model(channels)
    elapsed = time.perf_counter() - start
    image = complex_from_channels(output[0]).cpu().numpy()
    _save_yak42_plot(image, output_path, snr_db)
    result = {
        "measurement_shape": measurement_shape,
        "image_shape": tuple(int(size) for size in image.shape),
        "snr_db": float(snr_db),
        "elapsed_seconds": elapsed,
        "image_entropy": image_entropy(image),
        "has_nan": bool(not np.all(np.isfinite(image))),
        "output": str(output_path),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--image-shape", nargs=2, type=int, default=(128, 64), metavar=("P", "Q"))
    parser.add_argument("--measurement-shape", nargs=2, type=int, default=(64, 32), metavar=("M", "N"))
    parser.add_argument("--train-samples", type=int, default=1024)
    parser.add_argument("--validation-samples", type=int, default=128)
    parser.add_argument("--test-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/admm_unrolled"))
    parser.add_argument("--no-cache", action="store_true", help="generate samples on demand")
    parser.add_argument("--evaluate-yak42", type=Path, help="checkpoint to evaluate instead of training")
    parser.add_argument("--yak42-data", type=Path, default=Path(__file__).with_name("Yak42.mat"))
    parser.add_argument("--yak42-output", type=Path, default=Path("outputs/admm_unrolled/yak42_unrolled.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.evaluate_yak42 is not None:
        evaluate_yak42(
            args.evaluate_yak42,
            data_path=args.yak42_data,
            output_path=args.yak42_output,
            device=args.device,
            seed=args.seed,
        )
        return
    train_model(
        image_shape=tuple(args.image_shape),
        measurement_shape=tuple(args.measurement_shape),
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        test_samples=args.test_samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_layers=args.layers,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        cache=not args.no_cache,
    )


if __name__ == "__main__":
    main()

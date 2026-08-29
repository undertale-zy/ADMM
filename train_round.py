"""Unified historical-compatible trainer for ADMM experiment rounds 1--36."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from admm_2d import admm_2d_fast, image_entropy
from admm_losses import detached_metrics
from admm_unrolled import complex_from_channels
from round_registry import (
    RoundConfig,
    build_dataset,
    build_model,
    compute_round_loss,
    get_round_config,
)


FAST_REFERENCE_ECHO = 0.11309962187112717


def _select_device(requested: str, local_rank: int) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda", local_rank)
    if requested == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable in this PyTorch runtime")
        return torch.device("mps")
    if requested != "auto":
        raise ValueError("device must be auto, cpu, mps, or cuda")
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _distributed_context(device: torch.device) -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    return rank, world_size, local_rank


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def _reduce_metrics(
    metrics: dict[str, float],
    batches: int,
    *,
    device: torch.device,
    world_size: int,
) -> dict[str, float]:
    if world_size == 1:
        return metrics
    keys = sorted(metrics)
    values = torch.tensor(
        [metrics[key] * batches for key in keys] + [float(batches)],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    count = max(float(values[-1]), 1.0)
    return {key: float(values[index] / count) for index, key in enumerate(keys)}


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    config: RoundConfig,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    compatibility: Literal["historical", "corrected"],
    world_size: int = 1,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    rows: list[dict[str, float]] = []
    for measurements, target in loader:
        measurements = measurements.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if training and device.type == "cuda"
            else contextlib.nullcontext()
        )
        with torch.set_grad_enabled(training), autocast:
            prediction = model(measurements)
            losses = compute_round_loss(
                config.round_id,
                prediction,
                target,
                measurements,
                validation=not training,
                compatibility=compatibility,
            )
            total: Tensor = losses["total"]  # type: ignore[assignment]
        if training:
            if scaler is not None:
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
        rows.append(detached_metrics(losses))  # type: ignore[arg-type]
    local = _mean_metrics(rows)
    return _reduce_metrics(local, len(rows), device=device, world_size=world_size)


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    config: RoundConfig,
    epoch: int,
    history: list[dict[str, Any]],
    trackers: dict[str, float],
    image_shape: tuple[int, int],
    measurement_shape: tuple[int, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_config = dict(getattr(model, "model_config", {}))
    model_config["round_id"] = config.round_id
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": model_config,
        "epoch": epoch,
        "history": history,
    }
    if config.checkpoint_schema == "base":
        payload.update(
            {
                "scaler_state": scaler.state_dict() if scaler else {},
                "image_shape": image_shape,
                "measurement_shape": measurement_shape,
                "num_layers": config.layers,
                "proximal": config.proximal,
                "share_proximal": True,
                "best_validation": trackers.get("best", float("inf")),
            }
        )
    elif config.checkpoint_schema == "support":
        payload.update(
            {
                "scaler_state": scaler.state_dict() if scaler else {},
                "trackers": trackers,
                "model_family": "support_fusion_admm_v1",
            }
        )
    else:
        payload["model_family"] = "rounds32_36"
    torch.save(payload, path)


def _update_selector(
    validation: dict[str, float],
    config: RoundConfig,
    trackers: dict[str, float],
) -> list[str]:
    saves: list[str] = []
    if config.checkpoint_schema == "base":
        if validation["total"] < trackers["best"]:
            trackers["best"] = validation["total"]
            saves.append("checkpoint-best.pt")
        return saves
    if config.checkpoint_schema == "rounds":
        score = validation["total"] + 0.5 * validation.get("background", 0.0)
        if score < trackers["best"]:
            trackers["best"] = score
            saves.append("checkpoint-best.pt")
        return saves

    score = validation["total"] + 5.0 * validation.get("background", 0.0)
    if score < trackers["score"]:
        trackers["score"] = score
        saves.append("checkpoint-best-score.pt")
    if validation["image"] < trackers["image"]:
        trackers["image"] = validation["image"]
        saves.append("checkpoint-best-image.pt")
    echo = validation.get("echo", validation.get("clean_echo", float("inf")))
    if echo < trackers["echo"]:
        trackers["echo"] = echo
        saves.append("checkpoint-best-echo.pt")
    if validation["image"] < 0.03 and echo <= FAST_REFERENCE_ECHO:
        background = validation.get("background", 0.0)
        if background < trackers["constrained"]:
            trackers["constrained"] = background
            saves.append("checkpoint-best-constrained.pt")
    return saves


def _selected_checkpoint(output_dir: Path, config: RoundConfig) -> Path | None:
    if config.checkpoint_schema == "base":
        return None
    if config.checkpoint_schema == "rounds":
        return output_dir / "checkpoint-best.pt"
    if config.selector == "score":
        return output_dir / "checkpoint-best-score.pt"
    constrained = output_dir / "checkpoint-best-constrained.pt"
    return constrained if constrained.is_file() else output_dir / "checkpoint-best-score.pt"


def _benchmark(
    model: nn.Module,
    measurement_shape: tuple[int, int],
    device: torch.device,
    schema: str,
) -> dict[str, float]:
    warmup = 3 if schema == "base" else 5
    repeats = 20
    sample = torch.zeros((1, 2, *measurement_shape), device=device)

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(sample)
        synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            model(sample)
        synchronize()
    return {
        "latency_ms": 1000.0 * (time.perf_counter() - start) / repeats,
        "warmup": float(warmup),
        "repeats": float(repeats),
    }


def _write_learned_parameters(path: Path, model: nn.Module) -> None:
    parameters = getattr(model, "parameters_per_layer", None)
    if parameters is None:
        return
    c = parameters.c.detach().cpu().tolist()
    tau = parameters.tau.detach().cpu().tolist()
    beta = parameters.beta.detach().cpu().tolist()
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("layer", "c", "tau", "beta"))
        writer.writerows(
            (layer, c_value, tau_value, beta_value)
            for layer, (c_value, tau_value, beta_value) in enumerate(
                zip(c, tau, beta), start=1
            )
        )


def _fixed_fast_reference(loader: DataLoader, image_shape: tuple[int, int]) -> dict[str, float]:
    image_errors: list[float] = []
    echo_errors: list[float] = []
    entropies: list[float] = []
    for measurements, target in loader:
        for sample in range(measurements.shape[0]):
            observed = measurements[sample, 0].numpy() + 1j * measurements[sample, 1].numpy()
            truth = target[sample, 0].numpy() + 1j * target[sample, 1].numpy()
            result = admm_2d_fast(observed, image_shape, max_iterations=40)
            image_errors.append(float(np.sum(np.abs(result.image - truth) ** 2) / max(np.sum(np.abs(truth) ** 2), 1e-8)))
            entropies.append(image_entropy(result.image))
    return {
        "image_nmse": float(np.mean(image_errors)),
        "image_entropy": float(np.mean(entropies)),
        "iterations": 40.0,
    }


def train_round(
    round_id: int,
    *,
    image_shape: tuple[int, int] = (512, 128),
    measurement_shape: tuple[int, int] = (256, 64),
    train_samples: int | None = None,
    validation_samples: int | None = None,
    test_samples: int | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    device: str = "auto",
    output_dir: Path | None = None,
    guide_checkpoint: Path | None = None,
    compatibility: Literal["historical", "corrected"] = "historical",
    num_workers: int = 0,
) -> dict[str, Any]:
    config = get_round_config(round_id)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    selected_device = _select_device(device, local_rank)
    rank, world_size, _ = _distributed_context(selected_device)
    if selected_device.type == "cuda":
        torch.cuda.set_device(selected_device)
        if config.checkpoint_schema != "rounds":
            torch.backends.cudnn.benchmark = True
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    output_dir = output_dir or Path(__file__).resolve().parent.parent / "runs" / config.run_directory
    raw_model = build_model(
        round_id,
        image_shape,
        measurement_shape,
        guide_checkpoint=guide_checkpoint,
    ).to(selected_device)
    find_unused = config.model_family == "guide" or config.model_family.startswith("round3")
    model: nn.Module = raw_model
    if world_size > 1:
        model = DistributedDataParallel(
            raw_model,
            device_ids=[selected_device.index] if selected_device.type == "cuda" else None,
            find_unused_parameters=find_unused,
        )

    train_set = build_dataset(round_id, "train", image_shape=image_shape, measurement_shape=measurement_shape, samples=train_samples)
    validation_set = build_dataset(round_id, "validation", image_shape=image_shape, measurement_shape=measurement_shape, samples=validation_samples)
    test_set = build_dataset(round_id, "test", image_shape=image_shape, measurement_shape=measurement_shape, samples=test_samples)
    train_sampler = DistributedSampler(train_set, shuffle=True, seed=config.seed) if world_size > 1 else None
    effective_batch = batch_size or config.batch_size
    train_loader = DataLoader(train_set, batch_size=effective_batch, shuffle=train_sampler is None, sampler=train_sampler, num_workers=num_workers, pin_memory=selected_device.type == "cuda")
    validation_loader = DataLoader(validation_set, batch_size=effective_batch, shuffle=False, num_workers=max(0, num_workers // 2))
    test_loader = DataLoader(test_set, batch_size=effective_batch, shuffle=False, num_workers=max(0, num_workers // 2))

    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=1e-3, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda") if selected_device.type == "cuda" else None
    history: list[dict[str, Any]] = []
    trackers = {key: float("inf") for key in ("best", "score", "image", "echo", "constrained")}
    total_epochs = epochs or config.epochs
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "train.log").write_text("", encoding="utf-8")

    for epoch in range(1, total_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        start = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, config, device=selected_device, optimizer=optimizer, scaler=scaler, compatibility=compatibility, world_size=world_size)
        if world_size > 1:
            dist.barrier()
        if rank == 0:
            with torch.no_grad():
                validation_metrics = run_epoch(raw_model, validation_loader, config, device=selected_device, optimizer=None, scaler=None, compatibility=compatibility)
            row = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics, "elapsed_seconds": time.perf_counter() - start}
            history.append(row)
            for filename in _update_selector(validation_metrics, config, trackers):
                _save_checkpoint(output_dir / filename, raw_model, optimizer, scaler, config, epoch, history, trackers, image_shape, measurement_shape)
            _save_checkpoint(output_dir / "checkpoint-last.pt", raw_model, optimizer, scaler, config, epoch, history, trackers, image_shape, measurement_shape)
            log_line = f"round {round_id:02d} epoch {epoch:03d} | train {train_metrics['total']:.6f} | validation {validation_metrics['total']:.6f}"
            print(log_line)
            with (output_dir / "train.log").open("a", encoding="utf-8") as log_file:
                log_file.write(log_line + "\n")
        if world_size > 1:
            dist.barrier()

    result: dict[str, Any] = {}
    if rank == 0:
        selected = _selected_checkpoint(output_dir, config)
        if selected is not None:
            payload = torch.load(selected, map_location=selected_device, weights_only=False)
            raw_model.load_state_dict(payload["model_state"], strict=True)
        with torch.no_grad():
            test_metrics = run_epoch(raw_model, test_loader, config, device=selected_device, optimizer=None, scaler=None, compatibility=compatibility)
        benchmark = _benchmark(raw_model, measurement_shape, selected_device, config.checkpoint_schema)
        parameter_count = sum(parameter.numel() for parameter in raw_model.parameters())
        if config.checkpoint_schema == "base":
            fixed_fast = _fixed_fast_reference(test_loader, image_shape)
            result = {
                "test": test_metrics,
                "fixed_fast_admm": fixed_fast,
                "benchmark": benchmark,
            }
            history.append({"test": test_metrics, "fixed_fast_admm": fixed_fast, "benchmark": benchmark})
        elif config.checkpoint_schema == "support":
            result = {
                "selected_checkpoint": str(selected),
                "test": test_metrics,
                "fixed_fast_admm_reference": {
                    "echo_nmse": FAST_REFERENCE_ECHO,
                    "image_nmse": 0.4275802185509148,
                    "image_entropy": 4.142121178538367,
                },
                "benchmark": benchmark,
                "parameter_count": parameter_count,
                "trackers": trackers,
                "config": config.__dict__,
            }
        else:
            result = {
                "selected_checkpoint": str(selected),
                "test": test_metrics,
                "parameter_count": parameter_count,
                "latency_ms": benchmark["latency_ms"],
                "config": config.__dict__,
            }
        if compatibility == "corrected":
            result["protocol"] = {
                "compatibility": compatibility,
                "image_shape": image_shape,
                "measurement_shape": measurement_shape,
                "train_samples": len(train_set),
                "validation_samples": len(validation_set),
                "test_samples": len(test_set),
                "world_size": world_size,
            }
        if round_id == 1:
            _write_learned_parameters(
                output_dir / "learned_parameters.csv", raw_model
            )
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, required=True, dest="round_id")
    parser.add_argument("--image-shape", nargs=2, type=int, default=(512, 128))
    parser.add_argument("--measurement-shape", nargs=2, type=int, default=(256, 64))
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--validation-samples", type=int)
    parser.add_argument("--test-samples", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--guide-checkpoint", type=Path)
    parser.add_argument("--compatibility", choices=("historical", "corrected"), default="historical")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.image_shape = (32, 16)
        args.measurement_shape = (16, 8)
        args.train_samples = 8
        args.validation_samples = 4
        args.test_samples = 4
        args.epochs = 1
        args.batch_size = 1
    train_round(
        args.round_id,
        image_shape=tuple(args.image_shape),
        measurement_shape=tuple(args.measurement_shape),
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        test_samples=args.test_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
        guide_checkpoint=args.guide_checkpoint,
        compatibility=args.compatibility,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()

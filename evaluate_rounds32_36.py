"""Fixed point/dense/Yak evaluation for one of experiment rounds 32--36."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dense_aircraft_isar_dataset import DenseAircraftISARDataset
from evaluation import (
    evaluate_dataset,
    infer_fast_yak42,
    infer_yak42,
    load_round_checkpoint,
    model_latency,
    save_db_panels,
)
from synthetic_isar_dataset import SyntheticISARDataset


FIXED_SYNTHETIC_SEED = 3_200_000


def evaluate_round(
    round_id: int,
    checkpoint: Path,
    *,
    data_path: Path,
    output_json: Path,
    output_figure: Path,
    samples: int = 500,
    yak_seeds: tuple[int, ...] = tuple(range(10)),
    device: str = "auto",
    compatibility: str = "historical",
    guide_checkpoint: Path | None = None,
) -> dict[str, object]:
    if round_id not in range(32, 37):
        raise ValueError("this evaluator accepts rounds 32 through 36")
    selected_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device if device != "auto" else "cpu")
    model, _ = load_round_checkpoint(checkpoint, round_id=round_id, device=selected_device, guide_checkpoint=guide_checkpoint)
    point = SyntheticISARDataset(samples, image_shape=(512, 128), measurement_shape=(256, 64), seed=FIXED_SYNTHETIC_SEED, cache=False)
    dense = DenseAircraftISARDataset(samples, image_shape=(512, 128), measurement_shape=(256, 64), seed=FIXED_SYNTHETIC_SEED, structured_probability=1.0)
    point_metrics = evaluate_dataset(model, point, device=selected_device)
    dense_metrics = evaluate_dataset(model, dense, device=selected_device)

    yak_rows: list[dict[str, float]] = []
    seed_zero_image = None
    last_measurements = None
    for seed in yak_seeds:
        image, measurements, metrics = infer_yak42(model, data_path, seed=seed, device=selected_device)
        yak_rows.append(metrics)
        last_measurements = measurements
        if seed == 0:
            seed_zero_image = image
    if seed_zero_image is None:
        raise ValueError("yak_seeds must include seed 0 for the standard figure")
    # Preserve the historical seed-9 baseline bug only in historical mode.
    if compatibility == "corrected":
        _, baseline_measurements, _ = infer_yak42(model, data_path, seed=0, device=selected_device)
    else:
        baseline_measurements = last_measurements
    assert baseline_measurements is not None
    baseline_image, baseline_metrics = infer_fast_yak42(baseline_measurements)
    keys = [key for key in yak_rows[0] if key not in ("seed", "snr_db")]
    yak_mean = {key: sum(row[key] for row in yak_rows) / len(yak_rows) for key in keys}
    result: dict[str, object] = {
        "round": round_id,
        "checkpoint": str(checkpoint),
        "point_only": point_metrics,
        "dense_only": dense_metrics,
        "yak42_by_seed": yak_rows,
        "yak42_mean": yak_mean,
        "fast_admm_figure_baseline": baseline_metrics,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "latency_ms": model_latency(model, (256, 64), selected_device),
    }
    if compatibility == "corrected":
        result["protocol"] = {
            "synthetic_seed": FIXED_SYNTHETIC_SEED,
            "synthetic_samples": samples,
            "yak_seeds": list(yak_seeds),
            "latency_warmup": 5,
            "latency_repeats": 30,
        }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    seed_zero_metrics = next(row for row in yak_rows if int(row["seed"]) == 0)
    save_db_panels(
        [baseline_image, seed_zero_image],
        [
            f"Fast-ADMM | H={baseline_metrics['entropy']:.3f}",
            f"Round {round_id} | H={seed_zero_metrics['entropy']:.3f}",
        ],
        output_figure,
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, required=True, dest="round_id")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("Yak42.mat"))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--yak-seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--compatibility", choices=("historical", "corrected"), default="historical")
    parser.add_argument("--guide-checkpoint", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_round(
        args.round_id,
        args.checkpoint,
        data_path=args.data,
        output_json=args.output_json,
        output_figure=args.output_figure,
        samples=args.samples,
        yak_seeds=tuple(args.yak_seeds),
        device=args.device,
        compatibility=args.compatibility,
        guide_checkpoint=args.guide_checkpoint,
    )


if __name__ == "__main__":
    main()

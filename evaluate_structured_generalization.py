"""Three-way Fast/Stage-1/current evaluation on point, structured and Yak data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaluation import evaluate_dataset, infer_fast_yak42, infer_yak42, load_round_checkpoint, save_db_panels
from reference_schedules import load_schedule_into_model
from round_registry import build_model
from structured_isar_dataset import StructuredISARDataset
from synthetic_isar_dataset import SyntheticISARDataset


FIXED_SEED = 3_000_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, required=True, dest="round_id")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("Yak42.mat"))
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    current, _ = load_round_checkpoint(args.checkpoint, round_id=args.round_id, device=device)
    stage1 = build_model(1).to(device)
    if args.stage1_checkpoint:
        payload = torch.load(args.stage1_checkpoint, map_location="cpu", weights_only=False)
        stage1.load_state_dict(payload["model_state"], strict=True)
    else:
        load_schedule_into_model(stage1, "stage1_scalar_8")
    stage1.eval()
    point = SyntheticISARDataset(args.samples, image_shape=(512, 128), measurement_shape=(256, 64), seed=FIXED_SEED, cache=False)
    structured = StructuredISARDataset(args.samples, image_shape=(512, 128), measurement_shape=(256, 64), seed=FIXED_SEED, structured_probability=1.0)
    result = {
        "seed": FIXED_SEED,
        "point_only": {
            "stage1": evaluate_dataset(stage1, point, device=device),
            "current": evaluate_dataset(current, point, device=device),
        },
        "structured_only": {
            "stage1": evaluate_dataset(stage1, structured, device=device),
            "current": evaluate_dataset(current, structured, device=device),
        },
    }
    stage1_image, measurements, stage1_metrics = infer_yak42(stage1, args.data, seed=0, device=device)
    current_image, _, current_metrics = infer_yak42(current, args.data, seed=0, device=device)
    fast_image, fast_metrics = infer_fast_yak42(measurements)
    result["yak42_seed0"] = {"fast_admm": fast_metrics, "stage1": stage1_metrics, "current": current_metrics}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    save_db_panels(
        [fast_image, stage1_image, current_image],
        ["Fast-ADMM", "Stage 1", f"Round {args.round_id}"],
        args.output_figure,
    )


if __name__ == "__main__":
    main()

"""General checkpoint evaluation on Yak-42 and the fixed Fast-ADMM baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaluation import infer_fast_yak42, infer_yak42, load_round_checkpoint, model_latency


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, required=True, dest="round_id")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("Yak42.mat"))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--guide-checkpoint", type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    model, _ = load_round_checkpoint(
        args.checkpoint,
        round_id=args.round_id,
        device=device,
        guide_checkpoint=args.guide_checkpoint,
    )
    rows = []
    baseline_rows = []
    for seed in args.seeds:
        _, measurements, metrics = infer_yak42(model, args.data, seed=seed, device=device)
        _, baseline = infer_fast_yak42(measurements)
        rows.append(metrics)
        baseline_rows.append({"seed": seed, **baseline})
    result = {
        "round": args.round_id,
        "checkpoint": str(args.checkpoint),
        "network": rows,
        "fast_admm": baseline_rows,
        "network_latency_ms": model_latency(model, (256, 64), device, warmup=3, repeats=20),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

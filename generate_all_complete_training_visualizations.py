"""Regenerate the eight R1--R31 group sheets and the 31-panel master sheet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluation import infer_yak42, load_round_checkpoint
from round_registry import ROUND_REGISTRY


GROUPS = (
    ("group_01_initial_full_models_seed0_60db.png", range(1, 4)),
    ("group_02_zero-preserving_support_fusion_seed0_60db.png", range(4, 8)),
    ("group_03_stage-1_guide_plus_firm_support_seed0_60db.png", range(8, 12)),
    ("group_04_deep_scalar_unfolding_seed0_60db.png", range(12, 16)),
    ("group_05_balanced_scalar_echo_sweep_seed0_60db.png", range(16, 20)),
    ("group_06_noise-aware_clean_echo_seed0_60db.png", range(20, 24)),
    ("group_07_structured_aircraft_v1_seed0_60db.png", range(24, 28)),
    ("group_08_dense-aircraft_v2_seed0_60db.png", range(28, 32)),
)


def _save_grid(
    rounds: list[int],
    images: dict[int, np.ndarray],
    metrics: dict[int, dict[str, float]],
    output: Path,
    *,
    rows: int,
    columns: int,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 3.3 * rows), squeeze=False, constrained_layout=True)
    for axis, round_id in zip(axes.flat, rounds):
        image = images[round_id]
        shifted = np.abs(np.fft.fftshift(image, axes=1))
        peak = max(float(np.max(shifted)), 1e-12)
        db = np.clip(20.0 * np.log10(shifted / peak + 1e-12), -60.0, 0.0)
        axis.imshow(db, cmap="viridis", vmin=-60, vmax=0, origin="lower", aspect="auto", extent=(-50, 50, -48, 48))
        axis.set_xlim(-40, 40)
        axis.set_ylim(-35, 35)
        row = metrics[round_id]
        axis.set_title(
            f"R{round_id} {ROUND_REGISTRY[round_id].name}\n"
            f"H={row['entropy']:.3f} band={100*row['band_40_60_pixel_fraction']:.3f}% "
            f"support={100*row['support_above_60_fraction']:.3f}%",
            fontsize=9,
        )
    for axis in list(axes.flat)[len(rounds):]:
        axis.axis("off")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    workspace_default = Path(__file__).resolve().parent.parent
    parser.add_argument("--run-root", type=Path, default=workspace_default / "runs")
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("Yak42.mat"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    images: dict[int, np.ndarray] = {}
    metrics: dict[int, dict[str, float]] = {}
    missing: list[int] = []
    for round_id in range(1, 32):
        config = ROUND_REGISTRY[round_id]
        checkpoint = args.run_root / config.run_directory / config.visual_checkpoint
        if not checkpoint.is_file():
            if args.allow_missing:
                missing.append(round_id)
                continue
            raise FileNotFoundError(f"missing R{round_id} checkpoint: {checkpoint}")
        model, _ = load_round_checkpoint(checkpoint, round_id=round_id, device=device)
        image, _, row = infer_yak42(model, args.data, seed=0, device=device)
        images[round_id], metrics[round_id] = image, row
        _save_grid([round_id], images, metrics, args.output_dir / "single" / f"round_{round_id:02d}.png", rows=1, columns=1, dpi=180)
    for filename, group_rounds in GROUPS:
        available = [round_id for round_id in group_rounds if round_id in images]
        if available:
            _save_grid(available, images, metrics, args.output_dir / filename, rows=1, columns=len(available), dpi=180)
    available_all = [round_id for round_id in range(1, 32) if round_id in images]
    if available_all:
        _save_grid(available_all, images, metrics, args.output_dir / "all_31_complete_trainings_yak42_seed0_60db.png", rows=4, columns=8, dpi=160)
    payload = {"rounds": metrics, "missing_rounds": missing}
    (args.output_dir / "all_31_training_visualization_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

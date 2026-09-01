from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .data import make_splits
from .metrics import cycle_scores, local_probe_scores, make_cycle_suite
from .model import ModelConfig
from .patching import (
    activation_patching_scores,
    make_patching_suite,
    summarize_patching,
)
from .plotting import generate_plot_suite
from .training import TrainConfig, evaluate, resolve_device, train_model


@dataclass(frozen=True)
class DepthSweepConfig:
    output_root: Path
    run_name: str | None
    layers: tuple[int, ...]
    seeds: tuple[int, ...]
    conditions: dict[str, int]
    width: int
    n_heads: int
    d_ff: int
    train_depth: int
    ood_depths: tuple[int, ...]
    validation_size: int
    test_size: int
    steps: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    dynamics_every: int
    dynamics_analysis_size: int
    cycle_count: int
    dynamics_cycle_count: int
    patch_count: int
    dynamics_patch_count: int
    device: str


def default_depth_sweep_config(
    output_root: Path,
    *,
    run_name: str | None = None,
    layers: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
    seeds: tuple[int, ...] = (0, 1, 2),
    steps: int = 3_000,
    dynamics_every: int = 500,
    device: str = "auto",
    validation_size: int = 1_000,
    test_size: int = 1_000,
    cycle_count: int = 512,
    dynamics_cycle_count: int = 256,
    patch_count: int = 256,
    dynamics_patch_count: int = 64,
    dynamics_analysis_size: int = 256,
) -> DepthSweepConfig:
    return DepthSweepConfig(
        output_root=output_root,
        run_name=run_name,
        layers=layers,
        seeds=seeds,
        conditions={"low_diversity": 256, "higher_diversity": 8_000},
        width=128,
        n_heads=4,
        d_ff=256,
        train_depth=3,
        ood_depths=(4, 5, 6),
        validation_size=validation_size,
        test_size=test_size,
        steps=steps,
        batch_size=256,
        learning_rate=3e-4,
        weight_decay=1e-2,
        dynamics_every=dynamics_every,
        dynamics_analysis_size=dynamics_analysis_size,
        cycle_count=cycle_count,
        dynamics_cycle_count=dynamics_cycle_count,
        patch_count=patch_count,
        dynamics_patch_count=dynamics_patch_count,
        device=device,
    )


def create_run_directory(output_root: Path, run_name: str | None) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    name = run_name or datetime.now(timezone.utc).strftime("depth_sweep_%Y%m%d_%H%M%S_%fZ")
    run_directory = output_root / name
    run_directory.mkdir(parents=False, exist_ok=False)
    return run_directory


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_artifacts(
    run_directory: Path,
    final_rows: list[dict[str, Any]],
    dynamics_rows: list[dict[str, Any]],
    final_patching_rows: list[dict[str, Any]],
    dynamics_patching_rows: list[dict[str, Any]],
) -> None:
    tables = run_directory / "tables"
    tables.mkdir(exist_ok=True)
    for name, rows in (
        ("results", final_rows),
        ("dynamics", dynamics_rows),
        ("patching_final", final_patching_rows),
        ("patching_dynamics", dynamics_patching_rows),
    ):
        _write_csv(tables / f"{name}.csv", rows)
        (tables / f"{name}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _metric_snapshot(
    model,
    *,
    id_expressions,
    ood_expressions,
    calibration_expressions,
    cycles,
    patching_suite,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    id_result = evaluate(model, id_expressions, batch_size=batch_size, device=device)
    ood_results = {
        depth: evaluate(model, expressions, batch_size=batch_size, device=device)
        for depth, expressions in ood_expressions.items()
    }
    deepest_ood = ood_expressions[max(ood_expressions)]
    probe_result = local_probe_scores(
        model,
        calibration_expressions,
        deepest_ood,
        device=device,
    )
    coherence_result = cycle_scores(model, cycles, device=device)
    patching_rows = activation_patching_scores(model, patching_suite, device=device)
    result: dict[str, float] = {
        "id_accuracy": id_result["accuracy"],
        "ood_accuracy_mean": sum(item["accuracy"] for item in ood_results.values())
        / len(ood_results),
        **{f"ood_depth_{depth}_accuracy": item["accuracy"] for depth, item in ood_results.items()},
        **probe_result,
        **coherence_result,
        **summarize_patching(patching_rows, n_layers=model.config.n_layers),
    }
    return result, patching_rows


def run_depth_sweep(config: DepthSweepConfig) -> Path:
    if config.width != 128:
        raise ValueError("This requested architecture sweep is fixed at width 128")
    if not config.layers or min(config.layers) < 1 or max(config.layers) > 6:
        raise ValueError("layers must be a non-empty subset of 1 through 6")
    if config.dynamics_every < 1 or config.dynamics_every > config.steps:
        raise ValueError("dynamics_every must be between 1 and steps")

    run_directory = create_run_directory(config.output_root, config.run_name)
    checkpoints = run_directory / "checkpoints"
    trajectories = run_directory / "trajectories"
    plots = run_directory / "plots"
    checkpoints.mkdir()
    trajectories.mkdir()
    plots.mkdir()

    serializable_config = asdict(config)
    serializable_config["output_root"] = str(config.output_root)
    (run_directory / "config.json").write_text(
        json.dumps(serializable_config, indent=2), encoding="utf-8"
    )
    (run_directory / "status.json").write_text(
        json.dumps({"status": "running", "run_directory": str(run_directory.resolve())}, indent=2),
        encoding="utf-8",
    )

    splits = make_splits(
        train_size=max(config.conditions.values()),
        validation_size=config.validation_size,
        test_size=config.test_size,
        train_depth=config.train_depth,
        ood_depths=config.ood_depths,
        seed=19_871,
    )
    cycles, _ = make_cycle_suite(config.cycle_count, operand_depth=2, seed=91_337)
    dynamics_cycles = cycles[: min(config.dynamics_cycle_count, len(cycles))]
    patching_suite = make_patching_suite(config.patch_count, seed=55_661)
    dynamics_patching_suite = patching_suite.subset(config.dynamics_patch_count)
    dynamics_size = config.dynamics_analysis_size
    dynamics_id = splits.test_id[:dynamics_size]
    dynamics_calibration = splits.validation[:dynamics_size]
    dynamics_ood = {
        depth: expressions[:dynamics_size] for depth, expressions in splits.test_ood.items()
    }
    device = resolve_device(config.device)

    final_rows: list[dict[str, Any]] = []
    dynamics_rows: list[dict[str, Any]] = []
    final_patching_rows: list[dict[str, Any]] = []
    dynamics_patching_rows: list[dict[str, Any]] = []
    total_models = len(config.layers) * len(config.conditions) * len(config.seeds)
    model_index = 0

    for architecture_layers in config.layers:
        model_config = ModelConfig(
            d_model=config.width,
            n_heads=config.n_heads,
            n_layers=architecture_layers,
            d_ff=config.d_ff,
            max_length=256,
        )
        for condition, train_size in config.conditions.items():
            for seed in config.seeds:
                model_index += 1
                run_name = f"layers_{architecture_layers}_{condition}_seed_{seed}"
                print(f"[{model_index}/{total_models}] training {run_name}", flush=True)
                model_trajectory: list[dict[str, Any]] = []

                def on_evaluation(
                    model,
                    step,
                    training_record,
                    architecture_layers=architecture_layers,
                    condition=condition,
                    seed=seed,
                    train_size=train_size,
                    model_trajectory=model_trajectory,
                    run_name=run_name,
                ) -> None:
                    snapshot, patch_rows = _metric_snapshot(
                        model,
                        id_expressions=dynamics_id,
                        ood_expressions=dynamics_ood,
                        calibration_expressions=dynamics_calibration,
                        cycles=dynamics_cycles,
                        patching_suite=dynamics_patching_suite,
                        batch_size=config.batch_size,
                        device=device,
                    )
                    metadata = {
                        "architecture_layers": architecture_layers,
                        "condition": condition,
                        "seed": seed,
                        "train_size": train_size,
                        "step": step,
                    }
                    row = {**metadata, **training_record, **snapshot}
                    dynamics_rows.append(row)
                    model_trajectory.append(row)
                    dynamics_patching_rows.extend(
                        {**metadata, **patch_row} for patch_row in patch_rows
                    )
                    (trajectories / f"{run_name}.json").write_text(
                        json.dumps(model_trajectory, indent=2), encoding="utf-8"
                    )

                train_config = TrainConfig(
                    steps=config.steps,
                    batch_size=config.batch_size,
                    learning_rate=config.learning_rate,
                    weight_decay=config.weight_decay,
                    eval_every=config.dynamics_every,
                    seed=seed,
                    device=config.device,
                )
                model, _ = train_model(
                    splits.train[:train_size],
                    splits.validation,
                    model_config=model_config,
                    train_config=train_config,
                    checkpoint_path=checkpoints / f"{run_name}.pt",
                    on_evaluation=on_evaluation,
                )

                train_result = evaluate(
                    model,
                    splits.train[:train_size],
                    batch_size=config.batch_size,
                    device=device,
                )
                final_snapshot, patch_rows = _metric_snapshot(
                    model,
                    id_expressions=splits.test_id,
                    ood_expressions=splits.test_ood,
                    calibration_expressions=splits.validation,
                    cycles=cycles,
                    patching_suite=patching_suite,
                    batch_size=config.batch_size,
                    device=device,
                )
                metadata = {
                    "architecture_layers": architecture_layers,
                    "condition": condition,
                    "seed": seed,
                    "train_size": train_size,
                }
                final_row = {
                    **metadata,
                    "train_accuracy": train_result["accuracy"],
                    **final_snapshot,
                }
                final_rows.append(final_row)
                final_patching_rows.extend({**metadata, **patch_row} for patch_row in patch_rows)
                print(json.dumps(final_row, sort_keys=True), flush=True)
                _write_artifacts(
                    run_directory,
                    final_rows,
                    dynamics_rows,
                    final_patching_rows,
                    dynamics_patching_rows,
                )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    generated_plots = generate_plot_suite(
        final_rows,
        dynamics_rows,
        final_patching_rows,
        output_dir=plots,
    )
    (run_directory / "status.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "models": len(final_rows),
                "plots": [str(path.relative_to(run_directory)) for path in generated_plots],
                "run_directory": str(run_directory.resolve()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Completed depth sweep: {run_directory.resolve()}", flush=True)
    return run_directory

from __future__ import annotations

import csv
import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .data import make_splits
from .metrics import cycle_scores, local_probe_scores, make_cycle_suite
from .model import ModelConfig
from .training import TrainConfig, evaluate, resolve_device, train_model


@dataclass(frozen=True)
class ExperimentConfig:
    output_dir: Path
    train_depth: int
    ood_depths: tuple[int, ...]
    conditions: dict[str, int]
    seeds: tuple[int, ...]
    validation_size: int
    test_size: int
    cycle_count: int
    cycle_operand_depth: int
    model: ModelConfig
    train: TrainConfig


def _plot_results(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    matplotlib_cache = output_path.parent / ".matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache.resolve()))
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for row in rows:
        label = f"{row['condition']}/s{row['seed']}"
        axes[0].scatter(row["cycle_holonomy_error"], row["ood_accuracy_mean"])
        axes[0].annotate(label, (row["cycle_holonomy_error"], row["ood_accuracy_mean"]), fontsize=7)
        axes[1].scatter(row["local_probe_mean"], row["ood_accuracy_mean"])
        axes[1].annotate(label, (row["local_probe_mean"], row["ood_accuracy_mean"]), fontsize=7)
    axes[0].set(xlabel="held-out cycle holonomy error (lower is coherent)", ylabel="OOD accuracy")
    axes[1].set(xlabel="OOD local-probe balanced accuracy", ylabel="OOD accuracy")
    figure.suptitle("Boolean compositional generalization: first falsification screen")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_experiment(config: ExperimentConfig) -> list[dict[str, float | int | str]]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    maximum_train_size = max(config.conditions.values())
    splits = make_splits(
        train_size=maximum_train_size,
        validation_size=config.validation_size,
        test_size=config.test_size,
        train_depth=config.train_depth,
        ood_depths=config.ood_depths,
        seed=19_871,
    )
    cycles, _ = make_cycle_suite(
        config.cycle_count,
        operand_depth=config.cycle_operand_depth,
        seed=91_337,
    )
    device = resolve_device(config.train.device)
    deepest_ood = splits.test_ood[max(config.ood_depths)]
    rows: list[dict[str, float | int | str]] = []

    serializable_config = asdict(config)
    serializable_config["output_dir"] = str(config.output_dir)
    (config.output_dir / "config.json").write_text(
        json.dumps(serializable_config, indent=2), encoding="utf-8"
    )

    for condition, train_size in config.conditions.items():
        for seed in config.seeds:
            run_name = f"{condition}_seed{seed}"
            run_train_config = TrainConfig(**{**asdict(config.train), "seed": seed})
            model, _ = train_model(
                splits.train[:train_size],
                splits.validation,
                model_config=config.model,
                train_config=run_train_config,
                checkpoint_path=config.output_dir / "checkpoints" / f"{run_name}.pt",
            )
            train_result = evaluate(
                model,
                splits.train[:train_size],
                batch_size=config.train.batch_size,
                device=device,
            )
            id_result = evaluate(
                model,
                splits.test_id,
                batch_size=config.train.batch_size,
                device=device,
            )
            ood_results = {
                depth: evaluate(
                    model,
                    expressions,
                    batch_size=config.train.batch_size,
                    device=device,
                )
                for depth, expressions in splits.test_ood.items()
            }
            probe_result = local_probe_scores(
                model,
                splits.validation,
                deepest_ood,
                device=device,
            )
            coherence_result = cycle_scores(model, cycles, device=device)
            row: dict[str, float | int | str] = {
                "condition": condition,
                "seed": seed,
                "train_size": train_size,
                "train_accuracy": train_result["accuracy"],
                "id_accuracy": id_result["accuracy"],
                "ood_accuracy_mean": sum(result["accuracy"] for result in ood_results.values())
                / len(ood_results),
            }
            row.update(
                {
                    f"ood_depth_{depth}_accuracy": result["accuracy"]
                    for depth, result in ood_results.items()
                }
            )
            row.update(probe_result)
            row.update(coherence_result)
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    fieldnames = list(rows[0])
    with (config.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (config.output_dir / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _plot_results(rows, config.output_dir / "summary.png")
    return rows


def smoke_config(
    output_dir: Path,
    *,
    device: str = "auto",
    steps: int | None = None,
    seeds: Sequence[int] = (0,),
) -> ExperimentConfig:
    return ExperimentConfig(
        output_dir=output_dir,
        train_depth=3,
        ood_depths=(4, 5),
        conditions={"low_diversity": 96, "higher_diversity": 768},
        seeds=tuple(seeds),
        validation_size=192,
        test_size=192,
        cycle_count=64,
        cycle_operand_depth=2,
        model=ModelConfig(d_model=32, n_heads=4, n_layers=1, d_ff=64, max_length=256),
        train=TrainConfig(
            steps=steps or 300,
            batch_size=64,
            learning_rate=8e-4,
            weight_decay=1e-2,
            eval_every=50,
            device=device,
        ),
    )


def pilot_config(
    output_dir: Path,
    *,
    device: str = "auto",
    steps: int | None = None,
    seeds: Sequence[int] = (0, 1, 2),
) -> ExperimentConfig:
    return ExperimentConfig(
        output_dir=output_dir,
        train_depth=3,
        ood_depths=(4, 5, 6),
        conditions={"low_diversity": 256, "higher_diversity": 8_000},
        seeds=tuple(seeds),
        validation_size=1_000,
        test_size=1_000,
        cycle_count=512,
        cycle_operand_depth=2,
        model=ModelConfig(d_model=128, n_heads=4, n_layers=3, d_ff=256, max_length=256),
        train=TrainConfig(
            steps=steps or 3_000,
            batch_size=256,
            learning_rate=3e-4,
            weight_decay=1e-2,
            eval_every=200,
            device=device,
        ),
    )

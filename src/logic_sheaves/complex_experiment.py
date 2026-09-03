from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .complex_plotting import generate_complex_plots
from .data import AssignedExpression, make_symbolic_splits
from .diagram_metrics import (
    fit_rewrite_transports,
    predictions,
    score_equivalence_diagrams,
    score_rewrite_pairs,
)
from .equivalence import DIAGRAM_BUILDERS, make_equivalence_suite, make_rewrite_calibration_pairs
from .model import ModelConfig
from .training import TrainConfig, evaluate, resolve_device, train_model


@dataclass(frozen=True)
class ComplexExperimentConfig:
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
    eval_every: int
    calibration_pairs_per_rewrite: int
    diagrams_per_family: int
    diagram_operand_depth: int
    device: str


def default_complex_config(
    output_root: Path,
    *,
    run_name: str | None = None,
    layers: tuple[int, ...] = (2, 4, 6),
    seeds: tuple[int, ...] = (0, 1, 2),
    steps: int = 2_000,
    device: str = "auto",
    validation_size: int = 1_000,
    test_size: int = 1_000,
    calibration_pairs_per_rewrite: int = 192,
    diagrams_per_family: int = 96,
) -> ComplexExperimentConfig:
    return ComplexExperimentConfig(
        output_root=output_root,
        run_name=run_name,
        layers=layers,
        seeds=seeds,
        conditions={"low_diversity": 512, "higher_diversity": 12_000},
        width=128,
        n_heads=4,
        d_ff=256,
        train_depth=3,
        ood_depths=(4, 5),
        validation_size=validation_size,
        test_size=test_size,
        steps=steps,
        batch_size=256,
        learning_rate=3e-4,
        weight_decay=1e-2,
        eval_every=max(1, min(250, steps)),
        calibration_pairs_per_rewrite=calibration_pairs_per_rewrite,
        diagrams_per_family=diagrams_per_family,
        diagram_operand_depth=2,
        device=device,
    )


def _create_run_directory(output_root: Path, run_name: str | None) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    stem = run_name or datetime.now(timezone.utc).strftime("equivalence_complex_%Y%m%d_%H%M%S_%fZ")
    candidate = output_root / stem
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"{stem}_{suffix}"
        suffix += 1
    candidate.mkdir()
    return candidate


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _root_operator_rows(
    model,
    examples: tuple[AssignedExpression, ...],
    *,
    split: str,
    depth: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    model_predictions = predictions(model, examples, device=device)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        grouped[example.expression.prefix_tokens()[0]].append(index)
    rows: list[dict[str, Any]] = []
    for operator, indices in sorted(grouped.items()):
        labels = np.asarray([examples[index].value for index in indices])
        selected_predictions = model_predictions[indices]
        recalls = [
            float((selected_predictions[labels == target] == target).mean())
            for target in (0, 1)
            if np.any(labels == target)
        ]
        rows.append(
            {
                "split": split,
                "depth": depth,
                "root_operator": operator,
                "count": len(indices),
                "accuracy": float((selected_predictions == labels).mean()),
                "balanced_accuracy": float(np.mean(recalls)),
            }
        )
    return rows


def _disjoint_diagram_suite(
    count_per_family: int,
    *,
    operand_depth: int,
    blocked: set[str],
) -> list:
    candidates = make_equivalence_suite(
        count_per_family * 2,
        operand_depth=operand_depth,
        seed=88_031,
    )
    grouped: dict[str, list] = defaultdict(list)
    for diagram in candidates:
        if any(str(vertex) in blocked for vertex in diagram.vertices):
            continue
        if len(grouped[diagram.family]) < count_per_family:
            grouped[diagram.family].append(diagram)
    if len(grouped) != len(DIAGRAM_BUILDERS) or any(
        len(items) != count_per_family for items in grouped.values()
    ):
        raise RuntimeError("Could not construct a data-disjoint equivalence suite")
    return [diagram for family in sorted(grouped) for diagram in grouped[family]]


def _disjoint_rewrite_pairs(
    count_per_label: int,
    *,
    operand_depth: int,
    seed: int,
    blocked: set[str],
) -> tuple[dict[str, list[tuple[AssignedExpression, AssignedExpression]]], set[str]]:
    candidates = make_rewrite_calibration_pairs(
        count_per_label * 3,
        operand_depth=operand_depth,
        seed=seed,
    )
    used = set(blocked)
    result: dict[str, list[tuple[AssignedExpression, AssignedExpression]]] = {}
    for label, pairs in candidates.items():
        selected: list[tuple[AssignedExpression, AssignedExpression]] = []
        for source, target in pairs:
            strings = {str(source), str(target)}
            if strings & used:
                continue
            selected.append((source, target))
            used.update(strings)
            if len(selected) == count_per_label:
                break
        if len(selected) != count_per_label:
            raise RuntimeError(f"Could not make {count_per_label} disjoint pairs for {label}")
        result[label] = selected
    return result, used


def _nanmean(rows: list[dict[str, Any]], key: str) -> float:
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    return float(np.nanmean(values))


def _summary_report(config: ComplexExperimentConfig, rows: list[dict[str, Any]]) -> str:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["architecture_layers"]), str(row["condition"]))].append(row)

    lines = [
        "# Logical equivalence complex experiment",
        "",
        "This run extends the original Boolean experiment with explicit variable assignments, ",
        "binary and n-ary operators, fixed-arity `MAJ3`, `ITE3`, `EXACT1_3`, and ",
        "`ATLEAST2_4`, and arbitrary equivalence diagrams rather than a single square.",
        "",
        "## Experimental design",
        "",
        f"- Architectures: {', '.join(f'L{layer}' for layer in config.layers)}, width {config.width}",
        f"- Seeds: {', '.join(map(str, config.seeds))}",
        f"- Training conditions: {config.conditions}",
        f"- Training depth: <= {config.train_depth}; OOD depths: {config.ood_depths}",
        f"- Optimization: {config.steps} steps, batch size {config.batch_size}",
        f"- Rewrite calibration: {config.calibration_pairs_per_rewrite} isolated pairs per label",
        f"- Held-out diagrams: {config.diagrams_per_family} per family",
        "",
        "The fitted transports never see a complete test loop or competing-path topology. All ",
        "reported diagram metrics use data-disjoint expressions and independently generated ",
        "assignments.",
        "",
        "## Model-level results",
        "",
        "| Layers | Condition | Train | ID | OOD | Diagram accuracy | Consistency | Transport | Holonomy | Path agreement |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (layers, condition), items in sorted(grouped.items()):
        means = {
            key: np.mean([float(item[key]) for item in items])
            for key in (
                "train_accuracy",
                "id_accuracy",
                "ood_accuracy_mean",
                "diagram_semantic_accuracy",
                "diagram_consistency_mean",
                "diagram_transport_mean",
                "diagram_holonomy_mean",
                "diagram_path_mean",
            )
        }
        lines.append(
            f"| {layers} | {condition} | {means['train_accuracy']:.3f} | "
            f"{means['id_accuracy']:.3f} | {means['ood_accuracy_mean']:.3f} | "
            f"{means['diagram_semantic_accuracy']:.3f} | "
            f"{means['diagram_consistency_mean']:.3f} | "
            f"{means['diagram_transport_mean']:.3f} | "
            f"{means['diagram_holonomy_mean']:.3f} | {means['diagram_path_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Metric interpretation",
            "",
            "- **Transport error** tests whether an isolated-rewrite map predicts the next hidden state.",
            "- **Holonomy error** composes those maps around every held-out loop and tests return to the start.",
            "- **Path-agreement error** compares two rewrite routes with the same endpoints.",
            "- **Prediction consistency** asks whether every equivalent vertex receives the same prediction.",
            "",
            "All representation errors are normalized by held-out activation variance. Low loop closure ",
            "without low edge transport is not sufficient evidence of coherent logical geometry.",
            "",
            "## Artifacts",
            "",
            "Raw model, diagram-family, rewrite, operator, and history tables are in `tables/`; ",
            "checkpoints are in `checkpoints/`; and figures A-K are in `plots/`.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_complex_experiment(config: ComplexExperimentConfig) -> Path:
    if not config.layers or min(config.layers) < 1:
        raise ValueError("layers must be non-empty positive integers")
    if config.calibration_pairs_per_rewrite <= config.width:
        raise ValueError("Use more calibration pairs per rewrite than representation dimensions")

    run_directory = _create_run_directory(config.output_root, config.run_name)
    checkpoints = run_directory / "checkpoints"
    tables = run_directory / "tables"
    plots = run_directory / "plots"
    checkpoints.mkdir()
    tables.mkdir()
    plots.mkdir()
    serializable = asdict(config)
    serializable["output_root"] = str(config.output_root)
    (run_directory / "config.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    (run_directory / "status.json").write_text(
        json.dumps({"status": "running", "run_directory": str(run_directory.resolve())}, indent=2),
        encoding="utf-8",
    )

    splits = make_symbolic_splits(
        train_size=max(config.conditions.values()),
        validation_size=config.validation_size,
        test_size=config.test_size,
        train_depth=config.train_depth,
        ood_depths=config.ood_depths,
        seed=4_281,
    )
    all_data = (
        *splits.train,
        *splits.validation,
        *splits.test_id,
        *(example for depth in config.ood_depths for example in splits.test_ood[depth]),
    )
    data_strings = {str(example) for example in all_data}
    calibration_pairs, used_strings = _disjoint_rewrite_pairs(
        config.calibration_pairs_per_rewrite,
        operand_depth=config.diagram_operand_depth,
        seed=73_013,
        blocked=data_strings,
    )
    rewrite_test_pairs, used_strings = _disjoint_rewrite_pairs(
        max(64, config.calibration_pairs_per_rewrite // 2),
        operand_depth=config.diagram_operand_depth,
        seed=73_014,
        blocked=used_strings,
    )
    diagrams = _disjoint_diagram_suite(
        config.diagrams_per_family,
        operand_depth=config.diagram_operand_depth,
        blocked=used_strings,
    )
    device = resolve_device(config.device)

    final_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    diagram_rows: list[dict[str, Any]] = []
    rewrite_rows: list[dict[str, Any]] = []
    operator_rows: list[dict[str, Any]] = []
    total = len(config.layers) * len(config.conditions) * len(config.seeds)
    model_index = 0

    for layers in config.layers:
        model_config = ModelConfig(
            d_model=config.width,
            n_heads=config.n_heads,
            n_layers=layers,
            d_ff=config.d_ff,
            max_length=256,
        )
        for condition, train_size in config.conditions.items():
            for seed in config.seeds:
                model_index += 1
                name = f"layers_{layers}_{condition}_seed_{seed}"
                print(f"[{model_index}/{total}] training {name}", flush=True)
                train_config = TrainConfig(
                    steps=config.steps,
                    batch_size=config.batch_size,
                    learning_rate=config.learning_rate,
                    weight_decay=config.weight_decay,
                    eval_every=config.eval_every,
                    seed=seed,
                    device=config.device,
                )
                model, history = train_model(
                    splits.train[:train_size],
                    splits.validation,
                    model_config=model_config,
                    train_config=train_config,
                    checkpoint_path=checkpoints / f"{name}.pt",
                )
                metadata = {
                    "architecture_layers": layers,
                    "condition": condition,
                    "seed": seed,
                    "train_size": train_size,
                }
                history_rows.extend({**metadata, **record} for record in history)

                train_result = evaluate(
                    model,
                    splits.train[:train_size],
                    batch_size=config.batch_size,
                    device=device,
                )
                id_result = evaluate(
                    model,
                    splits.test_id,
                    batch_size=config.batch_size,
                    device=device,
                )
                ood_results = {
                    depth: evaluate(
                        model,
                        splits.test_ood[depth],
                        batch_size=config.batch_size,
                        device=device,
                    )
                    for depth in config.ood_depths
                }
                transports = fit_rewrite_transports(model, calibration_pairs, device=device)
                model_rewrite_rows = score_rewrite_pairs(
                    model, rewrite_test_pairs, transports, device=device
                )
                model_diagram_rows = score_equivalence_diagrams(
                    model, diagrams, transports, device=device
                )
                ood_mean = float(np.mean([result["accuracy"] for result in ood_results.values()]))
                enriched_diagrams = [
                    {**metadata, "ood_accuracy_mean": ood_mean, **row}
                    for row in model_diagram_rows
                ]
                enriched_rewrites = [{**metadata, **row} for row in model_rewrite_rows]
                diagram_rows.extend(enriched_diagrams)
                rewrite_rows.extend(enriched_rewrites)

                model_operator_rows = _root_operator_rows(
                    model,
                    splits.test_id,
                    split="id",
                    depth=config.train_depth,
                    device=device,
                )
                deepest = max(config.ood_depths)
                model_operator_rows.extend(
                    _root_operator_rows(
                        model,
                        splits.test_ood[deepest],
                        split="ood",
                        depth=deepest,
                        device=device,
                    )
                )
                operator_rows.extend({**metadata, **row} for row in model_operator_rows)

                final_row = {
                    **metadata,
                    "train_accuracy": train_result["accuracy"],
                    "id_accuracy": id_result["accuracy"],
                    "ood_accuracy_mean": ood_mean,
                    **{
                        f"ood_depth_{depth}_accuracy": result["accuracy"]
                        for depth, result in ood_results.items()
                    },
                    "diagram_semantic_accuracy": _nanmean(
                        model_diagram_rows, "semantic_accuracy"
                    ),
                    "diagram_consistency_mean": _nanmean(
                        model_diagram_rows, "prediction_consistency"
                    ),
                    "diagram_transport_mean": _nanmean(model_diagram_rows, "transport_error"),
                    "diagram_holonomy_mean": _nanmean(model_diagram_rows, "holonomy_error"),
                    "diagram_path_mean": _nanmean(model_diagram_rows, "path_agreement_error"),
                    "rewrite_transport_mean": _nanmean(model_rewrite_rows, "transport_error"),
                }
                final_rows.append(final_row)
                print(
                    f"    train={train_result['accuracy']:.3f} ID={id_result['accuracy']:.3f} "
                    f"OOD={ood_mean:.3f} holonomy={final_row['diagram_holonomy_mean']:.3f} "
                    f"path={final_row['diagram_path_mean']:.3f}",
                    flush=True,
                )
                _write_rows(tables / "results.csv", final_rows)
                _write_rows(tables / "histories.csv", history_rows)
                _write_rows(tables / "diagrams.csv", diagram_rows)
                _write_rows(tables / "rewrites.csv", rewrite_rows)
                _write_rows(tables / "operators.csv", operator_rows)

    plot_paths = generate_complex_plots(
        final_rows=final_rows,
        history_rows=history_rows,
        diagram_rows=diagram_rows,
        rewrite_rows=rewrite_rows,
        operator_rows=operator_rows,
        output_dir=plots,
    )
    (run_directory / "report.md").write_text(
        _summary_report(config, final_rows), encoding="utf-8"
    )
    status = {
        "status": "complete",
        "models": len(final_rows),
        "diagram_families": len({row["family"] for row in diagram_rows}),
        "rewrite_labels": len({row["rewrite"] for row in rewrite_rows}),
        "plots": [str(path.relative_to(run_directory)) for path in plot_paths],
        "run_directory": str(run_directory.resolve()),
    }
    (run_directory / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return run_directory

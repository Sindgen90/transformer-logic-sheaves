from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

CONDITION_STYLES = {"low_diversity": "--", "higher_diversity": "-"}
CONDITION_MARKERS = {"low_diversity": "o", "higher_diversity": "^"}


def _setup_matplotlib(output_dir: Path):
    cache = output_dir.parent / ".matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache.resolve()))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _mean_std(
    rows: list[dict[str, Any]],
    metric: str,
) -> tuple[float, float]:
    values = np.asarray([float(row[metric]) for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def _group(rows: list[dict[str, Any]], *keys: str) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return dict(grouped)


def _layer_colors(plt, rows: list[dict[str, Any]]) -> dict[int, Any]:
    layers = sorted({int(row["architecture_layers"]) for row in rows})
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, max(len(layers), 2)))
    return {layer: colors[index] for index, layer in enumerate(layers)}


def _trajectory_panel(ax, rows, metric, ylabel, colors) -> None:
    grouped = _group(rows, "condition", "architecture_layers", "step")
    line_groups: dict[tuple[str, int], list[tuple[int, float, float]]] = defaultdict(list)
    for (condition, layers, step), group_rows in grouped.items():
        mean, std = _mean_std(group_rows, metric)
        line_groups[(str(condition), int(layers))].append((int(step), mean, std))
    for (condition, layers), points in sorted(line_groups.items()):
        points.sort()
        x = np.asarray([point[0] for point in points])
        mean = np.asarray([point[1] for point in points])
        std = np.asarray([point[2] for point in points])
        label = f"L{layers} {condition.replace('_diversity', '')}"
        ax.plot(
            x,
            mean,
            color=colors[layers],
            linestyle=CONDITION_STYLES.get(condition, "-"),
            label=label,
        )
        if np.any(std > 0):
            ax.fill_between(x, mean - std, mean + std, color=colors[layers], alpha=0.10)
    ax.set(xlabel="optimizer step", ylabel=ylabel)
    ax.grid(alpha=0.2)


def plot_a_training_curves(plt, rows, output_dir, colors) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, metric, ylabel in zip(
        axes,
        ("train_loss", "validation_loss", "validation_accuracy"),
        ("train loss", "validation loss", "validation accuracy"),
        strict=True,
    ):
        _trajectory_panel(axis, rows, metric, ylabel, colors)
    axes[-1].legend(fontsize=7, ncol=2, loc="best")
    figure.suptitle("A. Training curves by architecture depth")
    figure.tight_layout()
    path = output_dir / "A_training_curves.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_b_accuracy_by_expression_depth(plt, rows, output_dir, colors) -> Path:
    conditions = sorted({str(row["condition"]) for row in rows})
    figure, axes = plt.subplots(1, len(conditions), figsize=(7 * len(conditions), 5), squeeze=False)
    for axis, condition in zip(axes[0], conditions, strict=True):
        condition_rows = [row for row in rows if row["condition"] == condition]
        layers = sorted({int(row["architecture_layers"]) for row in condition_rows})
        for layer in layers:
            layer_rows = [row for row in condition_rows if int(row["architecture_layers"]) == layer]
            metrics = ["id_accuracy"] + sorted(
                [key for key in layer_rows[0] if key.startswith("ood_depth_")],
                key=lambda key: int(key.split("_")[2]),
            )
            x = [3] + [int(metric.split("_")[2]) for metric in metrics[1:]]
            means = [_mean_std(layer_rows, metric)[0] for metric in metrics]
            stds = [_mean_std(layer_rows, metric)[1] for metric in metrics]
            axis.errorbar(x, means, yerr=stds, marker="o", color=colors[layer], label=f"L{layer}")
        axis.axhline(0.5, color="black", linewidth=1, alpha=0.4)
        axis.set(
            title=condition.replace("_", " "),
            xlabel="expression depth (3 is ID ≤3)",
            ylabel="accuracy",
            xticks=x,
            ylim=(0.45, 1.02),
        )
        axis.grid(alpha=0.2)
    axes[0, -1].legend(title="Transformer layers", fontsize=8)
    figure.suptitle("B. Accuracy versus expression depth")
    figure.tight_layout()
    path = output_dir / "B_accuracy_by_expression_depth.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_c_accuracy_bars(plt, rows, output_dir) -> Path:
    conditions = sorted({str(row["condition"]) for row in rows})
    figure, axes = plt.subplots(1, len(conditions), figsize=(7 * len(conditions), 5), squeeze=False)
    metrics = ("train_accuracy", "id_accuracy", "ood_accuracy_mean")
    labels = ("train", "ID", "OOD mean")
    for axis, condition in zip(axes[0], conditions, strict=True):
        condition_rows = [row for row in rows if row["condition"] == condition]
        layers = sorted({int(row["architecture_layers"]) for row in condition_rows})
        x = np.arange(len(layers), dtype=float)
        width = 0.24
        for metric_index, (metric, label) in enumerate(zip(metrics, labels, strict=True)):
            means = []
            stds = []
            for layer in layers:
                selected = [
                    row for row in condition_rows if int(row["architecture_layers"]) == layer
                ]
                mean, std = _mean_std(selected, metric)
                means.append(mean)
                stds.append(std)
            axis.bar(
                x + (metric_index - 1) * width,
                means,
                width,
                yerr=stds,
                capsize=2,
                label=label,
            )
        axis.set(
            title=condition.replace("_", " "),
            xlabel="Transformer layers",
            ylabel="accuracy",
            xticks=x,
            xticklabels=layers,
            ylim=(0.45, 1.02),
        )
        axis.grid(axis="y", alpha=0.2)
    axes[0, -1].legend()
    figure.suptitle("C. Train, ID, and OOD accuracy")
    figure.tight_layout()
    path = output_dir / "C_train_id_ood_bars.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _metric_scatter(ax, rows, metric, xlabel, colors) -> None:
    for row in rows:
        layer = int(row["architecture_layers"])
        condition = str(row["condition"])
        ax.scatter(
            float(row[metric]),
            float(row["ood_accuracy_mean"]),
            color=colors[layer],
            marker=CONDITION_MARKERS.get(condition, "o"),
            alpha=0.85,
        )
    ax.set(xlabel=xlabel, ylabel="mean OOD accuracy")
    ax.grid(alpha=0.2)


def plot_d_coherence(plt, rows, output_dir, colors) -> Path:
    figure, axis = plt.subplots(figsize=(7, 5.5))
    _metric_scatter(
        axis,
        rows,
        "cycle_holonomy_error",
        "held-out cycle holonomy error (lower is coherent)",
        colors,
    )
    for layer, color in colors.items():
        axis.scatter([], [], color=color, label=f"L{layer}")
    for condition, marker in CONDITION_MARKERS.items():
        axis.scatter([], [], color="gray", marker=marker, label=condition.replace("_", " "))
    axis.legend(fontsize=8, ncol=2)
    figure.suptitle("D. Global coherence versus compositional generalization")
    figure.tight_layout()
    path = output_dir / "D_coherence_vs_generalization.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_e_baselines(plt, rows, output_dir, colors) -> Path:
    panels = (
        ("cycle_identity_energy", "identity energy ↓"),
        ("cycle_transport_error", "transport error ↓"),
        ("cycle_holonomy_error", "holonomy error ↓"),
        ("cycle_edge_cka", "edge CKA ↑"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(11, 9))
    for axis, (metric, label) in zip(axes.flat, panels, strict=True):
        _metric_scatter(axis, rows, metric, label, colors)
    figure.suptitle("E. Coherence metric and similarity baselines")
    figure.tight_layout()
    path = output_dir / "E_baseline_comparison.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_f_local_global(plt, rows, output_dir) -> Path:
    figure, axis = plt.subplots(figsize=(7, 5.5))
    points = axis.scatter(
        [float(row["local_probe_mean"]) for row in rows],
        [float(row["cycle_holonomy_error"]) for row in rows],
        c=[float(row["ood_accuracy_mean"]) for row in rows],
        s=[35 + 12 * int(row["architecture_layers"]) for row in rows],
        cmap="viridis",
        alpha=0.85,
    )
    figure.colorbar(points, ax=axis, label="mean OOD accuracy")
    axis.set(
        xlabel="OOD local-probe balanced accuracy",
        ylabel="cycle holonomy error (lower is coherent)",
    )
    axis.grid(alpha=0.2)
    figure.suptitle("F. Local decodability versus global coherence")
    figure.tight_layout()
    path = output_dir / "F_local_vs_global.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_g_operator_probes(plt, rows, output_dir, colors) -> Path:
    operators = ("and", "not", "or", "xor")
    figure, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
    for axis, operator in zip(axes.flat, operators, strict=True):
        metric = f"local_probe_{operator}"
        grouped = _group(rows, "condition", "architecture_layers")
        for (condition, layer), group_rows in grouped.items():
            mean, std = _mean_std(group_rows, metric)
            axis.errorbar(
                [int(layer)],
                [mean],
                yerr=[std],
                marker=CONDITION_MARKERS.get(str(condition), "o"),
                color=colors[int(layer)],
                capsize=2,
            )
        axis.axhline(0.5, color="black", linewidth=1, alpha=0.35)
        axis.set(title=operator.upper(), xlabel="Transformer layers", ylabel="balanced accuracy")
        axis.grid(alpha=0.2)
    for condition, marker in CONDITION_MARKERS.items():
        axes[0, 0].scatter([], [], color="gray", marker=marker, label=condition.replace("_", " "))
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("G. Operator-specific OOD linear probes")
    figure.tight_layout()
    path = output_dir / "G_operator_probes.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_h_correlations(plt, rows, output_dir) -> Path:
    candidates = [
        ("train_accuracy", "train acc"),
        ("id_accuracy", "ID acc"),
        ("ood_accuracy_mean", "OOD acc"),
        ("local_probe_mean", "probe"),
        ("cycle_identity_energy", "identity"),
        ("cycle_transport_error", "transport"),
        ("cycle_holonomy_error", "holonomy"),
        ("cycle_edge_cka", "CKA"),
        ("cycle_activation_variance", "act var"),
        ("patch_best_counterfactual_accuracy", "patch acc"),
        ("patch_best_effect_fraction", "patch effect"),
    ]
    candidates = [
        item
        for item in candidates
        if item[0] in rows[0] and np.isfinite([float(row[item[0]]) for row in rows]).sum() >= 3
    ]
    matrix = np.full((len(candidates), len(candidates)), np.nan)
    for row_index, (left, _) in enumerate(candidates):
        for column_index, (right, _) in enumerate(candidates):
            x = np.asarray([float(row[left]) for row in rows])
            y = np.asarray([float(row[right]) for row in rows])
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() >= 3 and x[valid].std() > 0 and y[valid].std() > 0:
                matrix[row_index, column_index] = np.corrcoef(x[valid], y[valid])[0, 1]
    labels = [label for _, label in candidates]
    size = max(7, 0.75 * len(labels))
    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(xticks=range(len(labels)), yticks=range(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_yticklabels(labels)
    for row_index in range(len(labels)):
        for column_index in range(len(labels)):
            value = matrix[row_index, column_index]
            if np.isfinite(value):
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if abs(value) > 0.55 else "black",
                )
    figure.suptitle("H. Model-level Pearson correlation matrix")
    figure.tight_layout()
    path = output_dir / "H_correlation_matrix.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_i_activation_patching(plt, rows, output_dir, colors) -> Path:
    rows = [row for row in rows if row["operator"] == "ALL"]
    panels = (
        ("counterfactual_target_accuracy", "counterfactual target accuracy"),
        ("counterfactual_effect_fraction", "counterfactual effect fraction"),
        ("equivalent_preservation", "same-truth preservation"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    grouped = _group(rows, "condition", "architecture_layers", "patch_stage")
    for axis, (metric, ylabel) in zip(axes, panels, strict=True):
        lines: dict[tuple[str, int], list[tuple[int, float, float]]] = defaultdict(list)
        for (condition, layer, stage), group_rows in grouped.items():
            mean, std = _mean_std(group_rows, metric)
            lines[(str(condition), int(layer))].append((int(stage), mean, std))
        for (condition, layer), points in sorted(lines.items()):
            points.sort()
            x = np.asarray([point[0] for point in points])
            means = np.asarray([point[1] for point in points])
            stds = np.asarray([point[2] for point in points])
            axis.plot(
                x,
                means,
                color=colors[layer],
                linestyle=CONDITION_STYLES.get(condition, "-"),
                marker=".",
                label=f"L{layer} {condition.replace('_diversity', '')}",
            )
            if np.any(stds > 0):
                axis.fill_between(x, means - stds, means + stds, color=colors[layer], alpha=0.10)
        axis.set(xlabel="patch stage (0 = embeddings)", ylabel=ylabel)
        axis.grid(alpha=0.2)
    axes[-1].legend(fontsize=7, ncol=2)
    figure.suptitle("I. Causal activation patching by network stage")
    figure.tight_layout()
    path = output_dir / "I_activation_patching.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_j_dynamics(plt, rows, output_dir, colors) -> Path:
    panels = (
        ("validation_accuracy", "validation accuracy"),
        ("ood_accuracy_mean", "mean OOD accuracy"),
        ("local_probe_mean", "local probe"),
        ("cycle_holonomy_error", "holonomy error"),
        ("patch_best_counterfactual_accuracy", "best patch accuracy"),
        ("patch_best_effect_fraction", "best patch effect"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    for axis, (metric, ylabel) in zip(axes.flat, panels, strict=True):
        _trajectory_panel(axis, rows, metric, ylabel, colors)
    axes[1, -1].legend(fontsize=7, ncol=2)
    figure.suptitle("J. Behavioral, representational, and causal training dynamics")
    figure.tight_layout()
    path = output_dir / "J_training_dynamics.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def generate_plot_suite(
    final_rows: list[dict[str, Any]],
    dynamics_rows: list[dict[str, Any]],
    final_patching_rows: list[dict[str, Any]],
    *,
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt = _setup_matplotlib(output_dir)
    colors = _layer_colors(plt, final_rows)
    paths = [
        plot_a_training_curves(plt, dynamics_rows, output_dir, colors),
        plot_b_accuracy_by_expression_depth(plt, final_rows, output_dir, colors),
        plot_c_accuracy_bars(plt, final_rows, output_dir),
        plot_d_coherence(plt, final_rows, output_dir, colors),
        plot_e_baselines(plt, final_rows, output_dir, colors),
        plot_f_local_global(plt, final_rows, output_dir),
        plot_g_operator_probes(plt, final_rows, output_dir, colors),
        plot_h_correlations(plt, final_rows, output_dir),
        plot_i_activation_patching(plt, final_patching_rows, output_dir, colors),
        plot_j_dynamics(plt, dynamics_rows, output_dir, colors),
    ]
    (output_dir / "plot_manifest.json").write_text(
        json.dumps([path.name for path in paths], indent=2), encoding="utf-8"
    )
    return paths

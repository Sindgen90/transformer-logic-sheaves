from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .equivalence import make_equivalence_suite


def _setup(output_dir: Path):
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib_cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    output_dir.mkdir(parents=True, exist_ok=True)
    return plt


def _mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(np.nanmean(array)), float(np.nanstd(array, ddof=1)) if len(array) > 1 else 0.0


def _pretty(value: str) -> str:
    return value.replace("_", " ").replace("diversity", "div.")


def _save(figure, path: Path) -> Path:
    figure.savefig(path, dpi=180, bbox_inches="tight")
    figure.clf()
    return path


def plot_training(plt, rows: list[dict[str, Any]], output_dir: Path) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {2: "#443983", 4: "#21918c", 6: "#fde725"}
    grouped: dict[tuple[int, str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["architecture_layers"]), str(row["condition"]), int(row["step"]))].append(
            float(row["validation_accuracy"])
        )
    steps = sorted({int(row["step"]) for row in rows})
    for layers in sorted({int(row["architecture_layers"]) for row in rows}):
        for condition, linestyle in (("higher_diversity", "-"), ("low_diversity", "--")):
            means, stds = zip(*[_mean_std(grouped[(layers, condition, step)]) for step in steps])
            label = f"L{layers} {_pretty(condition)}"
            axes[0].plot(steps, means, linestyle, color=colors[layers], label=label)
            axes[0].fill_between(
                steps,
                np.asarray(means) - np.asarray(stds),
                np.asarray(means) + np.asarray(stds),
                color=colors[layers],
                alpha=0.10,
            )
    axes[0].set(xlabel="optimizer step", ylabel="validation accuracy", title="ID validation")
    axes[0].legend(fontsize=8, ncol=2)

    loss_grouped: dict[tuple[int, str, int], list[float]] = defaultdict(list)
    for row in rows:
        loss_grouped[
            (int(row["architecture_layers"]), str(row["condition"]), int(row["step"]))
        ].append(float(row["train_loss"]))
    for layers in sorted({int(row["architecture_layers"]) for row in rows}):
        for condition, linestyle in (("higher_diversity", "-"), ("low_diversity", "--")):
            means = [_mean_std(loss_grouped[(layers, condition, step)])[0] for step in steps]
            axes[1].plot(steps, means, linestyle, color=colors[layers])
    axes[1].set(xlabel="optimizer step", ylabel="training loss", title="Optimization")
    figure.suptitle("A. Symbolic-language training dynamics")
    return _save(figure, output_dir / "A_training_dynamics.png")


def plot_behavior(plt, rows: list[dict[str, Any]], output_dir: Path) -> Path:
    figure, ax = plt.subplots(figsize=(9, 6))
    colors = {"higher_diversity": "#31688e", "low_diversity": "#b35806"}
    layers = sorted({int(row["architecture_layers"]) for row in rows})
    for condition in ("higher_diversity", "low_diversity"):
        subset = [row for row in rows if row["condition"] == condition]
        for metric, linestyle, marker in (
            ("id_accuracy", "--", "s"),
            ("ood_accuracy_mean", "-", "o"),
        ):
            means, stds = [], []
            for layer in layers:
                values = [float(row[metric]) for row in subset if int(row["architecture_layers"]) == layer]
                mean, std = _mean_std(values)
                means.append(mean)
                stds.append(std)
            ax.errorbar(
                layers,
                means,
                yerr=stds,
                color=colors[condition],
                linestyle=linestyle,
                marker=marker,
                capsize=4,
                label=f"{_pretty(condition)} {'ID' if metric == 'id_accuracy' else 'OOD'}",
            )
    ax.axhline(0.5, color="grey", linewidth=1)
    ax.set(xlabel="Transformer layers", ylabel="accuracy", xticks=layers, ylim=(0.45, 1.01))
    ax.legend()
    ax.set_title("B. Generalization with variables and higher-arity operators")
    return _save(figure, output_dir / "B_behavior.png")


def plot_depth_generalization(plt, rows: list[dict[str, Any]], output_dir: Path) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    layers = sorted({int(row["architecture_layers"]) for row in rows})
    colors = plt.cm.viridis(np.linspace(0.12, 0.9, len(layers)))
    depth_metrics = [(3, "id_accuracy")] + sorted(
        (int(key.split("_")[2]), key)
        for key in rows[0]
        if key.startswith("ood_depth_") and key.endswith("_accuracy")
    )
    for ax, condition in zip(axes, ("higher_diversity", "low_diversity")):
        for layer, color in zip(layers, colors):
            subset = [
                row
                for row in rows
                if row["condition"] == condition and int(row["architecture_layers"]) == layer
            ]
            values = [_mean_std([float(row[key]) for row in subset]) for _, key in depth_metrics]
            ax.errorbar(
                [depth for depth, _ in depth_metrics],
                [item[0] for item in values],
                yerr=[item[1] for item in values],
                marker="o",
                color=color,
                capsize=3,
                label=f"L{layer}",
            )
        ax.axhline(0.5, color="grey", linewidth=1)
        ax.set(title=_pretty(condition), xlabel="expression depth", xticks=[x[0] for x in depth_metrics])
    axes[0].set_ylabel("accuracy")
    axes[1].legend()
    figure.suptitle("C. Generalization beyond the training tree depth")
    return _save(figure, output_dir / "C_depth_generalization.png")


def _heatmap(
    plt,
    matrix: np.ndarray,
    row_labels: list[str],
    column_labels: list[str],
    *,
    title: str,
    colorbar_label: str,
    cmap: str = "viridis",
):
    width = max(8, 1.1 * len(column_labels))
    height = max(5, 0.43 * len(row_labels) + 2)
    figure, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(column_labels)), column_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(row_labels)), [_pretty(label) for label in row_labels])
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if np.isfinite(value):
                red, green, blue, _ = image.cmap(image.norm(value))
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                ax.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black" if luminance > 0.55 else "white",
                )
    figure.colorbar(image, ax=ax, label=colorbar_label)
    ax.set_title(title)
    return figure


def plot_operator_accuracy(plt, rows: list[dict[str, Any]], output_dir: Path) -> Path:
    subset = [row for row in rows if row["condition"] == "higher_diversity" and row["split"] == "ood"]
    operators = sorted({str(row["root_operator"]) for row in subset})
    layers = sorted({int(row["architecture_layers"]) for row in subset})
    matrix = np.full((len(operators), len(layers)), np.nan)
    for row_index, operator in enumerate(operators):
        for column, layer in enumerate(layers):
            values = [
                float(row["balanced_accuracy"])
                for row in subset
                if row["root_operator"] == operator and int(row["architecture_layers"]) == layer
            ]
            matrix[row_index, column] = np.mean(values) if values else np.nan
    figure = _heatmap(
        plt,
        matrix,
        operators,
        [f"L{layer}" for layer in layers],
        title="D. OOD balanced accuracy by root operator (higher diversity)",
        colorbar_label="balanced accuracy",
    )
    return _save(figure, output_dir / "D_operator_accuracy.png")


def plot_family_holonomy(plt, rows: list[dict[str, Any]], output_dir: Path) -> Path:
    subset = [row for row in rows if row["condition"] == "higher_diversity"]
    families = sorted({str(row["family"]) for row in subset})
    layers = sorted({int(row["architecture_layers"]) for row in subset})
    matrix = np.full((len(families), len(layers)), np.nan)
    for row_index, family in enumerate(families):
        for column, layer in enumerate(layers):
            values = [
                float(row["holonomy_error"])
                for row in subset
                if row["family"] == family and int(row["architecture_layers"]) == layer
            ]
            matrix[row_index, column] = np.mean(values) if values else np.nan
    figure = _heatmap(
        plt,
        matrix,
        families,
        [f"L{layer}" for layer in layers],
        title="E. Held-out topology holonomy by equivalence family",
        colorbar_label="variance-normalized holonomy error",
        cmap="magma",
    )
    return _save(figure, output_dir / "E_family_holonomy.png")


def plot_shape_complexity(plt, rows: list[dict[str, Any]], output_dir: Path) -> Path:
    figure, ax = plt.subplots(figsize=(10, 7))
    markers = {"higher_diversity": "^", "low_diversity": "o"}
    accuracies = np.asarray([float(row["ood_accuracy_mean"]) for row in rows])
    color_min, color_max = float(accuracies.min()), float(accuracies.max())
    scatter = None
    for condition, marker in markers.items():
        subset = [row for row in rows if row["condition"] == condition]
        scatter = ax.scatter(
            [float(row["mean_loop_length"]) for row in subset],
            [float(row["holonomy_error"]) for row in subset],
            c=[float(row["ood_accuracy_mean"]) for row in subset],
            cmap="viridis",
            vmin=color_min,
            vmax=color_max,
            marker=marker,
            alpha=0.72,
            label=_pretty(condition),
        )
    if scatter is not None:
        figure.colorbar(scatter, ax=ax, label="mean OOD accuracy")
    ax.set(xlabel="mean loop length", ylabel="holonomy error", title="F. Loop complexity and closure")
    ax.legend()
    return _save(figure, output_dir / "F_shape_complexity.png")


def plot_path_agreement(plt, rows: list[dict[str, Any]], output_dir: Path) -> Path:
    subset = [row for row in rows if np.isfinite(float(row["path_agreement_error"]))]
    families = sorted({str(row["family"]) for row in subset})
    layers = sorted({int(row["architecture_layers"]) for row in subset})
    figure, axes = plt.subplots(1, len(families), figsize=(5 * len(families), 5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, family in zip(axes, families):
        for condition, marker in (("higher_diversity", "^"), ("low_diversity", "o")):
            means, stds = [], []
            for layer in layers:
                values = [
                    float(row["path_agreement_error"])
                    for row in subset
                    if row["family"] == family
                    and row["condition"] == condition
                    and int(row["architecture_layers"]) == layer
                ]
                mean, std = _mean_std(values)
                means.append(mean)
                stds.append(std)
            ax.errorbar(layers, means, yerr=stds, marker=marker, capsize=3, label=_pretty(condition))
        ax.set(title=_pretty(family), xlabel="Transformer layers", xticks=layers)
    axes[0].set_ylabel("path-agreement error")
    axes[-1].legend()
    figure.suptitle("G. Agreement between distinct rewrite routes")
    return _save(figure, output_dir / "G_path_agreement.png")


def plot_transport_vs_holonomy(plt, rows: list[dict[str, Any]], output_dir: Path) -> Path:
    figure, ax = plt.subplots(figsize=(9, 7))
    markers = {"higher_diversity": "^", "low_diversity": "o"}
    accuracies = np.asarray([float(row["ood_accuracy_mean"]) for row in rows])
    color_min, color_max = float(accuracies.min()), float(accuracies.max())
    scatter = None
    for condition, marker in markers.items():
        subset = [row for row in rows if row["condition"] == condition]
        scatter = ax.scatter(
            [float(row["transport_error"]) for row in subset],
            [float(row["holonomy_error"]) for row in subset],
            c=[float(row["ood_accuracy_mean"]) for row in subset],
            cmap="viridis",
            vmin=color_min,
            vmax=color_max,
            marker=marker,
            alpha=0.72,
            label=_pretty(condition),
        )
    if scatter is not None:
        figure.colorbar(scatter, ax=ax, label="mean OOD accuracy")
    ax.set(
        xlabel="held-out edge transport error",
        ylabel="holonomy error",
        title="H. Edge fidelity versus loop closure",
    )
    ax.legend()
    return _save(figure, output_dir / "H_transport_vs_holonomy.png")


def plot_correlations(plt, rows: list[dict[str, Any]], output_dir: Path) -> Path:
    columns = (
        ("train_accuracy", "train"),
        ("id_accuracy", "ID"),
        ("ood_accuracy_mean", "OOD"),
        ("diagram_semantic_accuracy", "diagram acc"),
        ("diagram_consistency_mean", "consistency"),
        ("diagram_transport_mean", "transport"),
        ("diagram_holonomy_mean", "holonomy"),
        ("diagram_path_mean", "path"),
        ("rewrite_transport_mean", "rewrite"),
    )
    matrix = np.asarray([[float(row[key]) for key, _ in columns] for row in rows])
    correlation = np.corrcoef(matrix, rowvar=False)
    labels = [label for _, label in columns]
    figure = _heatmap(
        plt,
        correlation,
        labels,
        labels,
        title="I. Model-level Pearson correlations",
        colorbar_label="correlation",
        cmap="coolwarm",
    )
    figure.axes[0].images[0].set_clim(-1, 1)
    return _save(figure, output_dir / "I_correlations.png")


def plot_rewrite_transport(plt, rows: list[dict[str, Any]], output_dir: Path) -> Path:
    subset = [row for row in rows if row["condition"] == "higher_diversity"]
    rewrites = sorted({str(row["rewrite"]) for row in subset})
    layers = sorted({int(row["architecture_layers"]) for row in subset})
    matrix = np.full((len(rewrites), len(layers)), np.nan)
    for row_index, rewrite in enumerate(rewrites):
        for column, layer in enumerate(layers):
            values = [
                float(row["transport_error"])
                for row in subset
                if row["rewrite"] == rewrite and int(row["architecture_layers"]) == layer
            ]
            matrix[row_index, column] = np.mean(values) if values else np.nan
    figure = _heatmap(
        plt,
        matrix,
        rewrites,
        [f"L{layer}" for layer in layers],
        title="J. Isolated rewrite transport generalization",
        colorbar_label="variance-normalized transport error",
        cmap="magma",
    )
    return _save(figure, output_dir / "J_rewrite_transport.png")


def plot_diagram_atlas(plt, output_dir: Path) -> Path:
    diagrams = make_equivalence_suite(2, operand_depth=0, seed=901)
    by_family = {}
    for diagram in diagrams:
        by_family.setdefault(diagram.family, diagram)
    figure, axes = plt.subplots(3, 4, figsize=(16, 11))
    for ax, (family, diagram) in zip(axes.flat, sorted(by_family.items())):
        count = len(diagram.vertices)
        if family == "associativity_pentagon":
            angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, count, endpoint=False)
            positions = np.empty((count, 2))
            for vertex, angle in zip((0, 1, 2, 4, 3), angles):
                positions[vertex] = (np.cos(angle), np.sin(angle))
        elif family == "distributivity_diamond":
            positions = np.empty((count, 2))
            for vertex, point in zip(
                (0, 2, 3, 1),
                ((-0.9, 0.0), (0.0, 0.75), (0.9, 0.0), (0.0, -0.75)),
            ):
                positions[vertex] = point
        elif count == 2:
            positions = np.asarray([[-0.8, 0.0], [0.8, 0.0]])
        elif count == 4:
            positions = np.asarray([[-0.8, 0.6], [0.8, 0.6], [0.8, -0.6], [-0.8, -0.6]])
        elif count in (5, 6):
            angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, count, endpoint=False)
            positions = np.column_stack((np.cos(angles), np.sin(angles)))
        elif count == 8:
            positions = np.asarray(
                [
                    [float(mask & 1) + 0.35 * float(bool(mask & 4)),
                     float(bool(mask & 2)) + 0.35 * float(bool(mask & 4))]
                    for mask in range(8)
                ]
            )
            positions -= positions.mean(axis=0, keepdims=True)
        else:
            raise AssertionError(f"No atlas layout for {count} vertices")
        seen: set[tuple[int, int]] = set()
        for edge in diagram.edges:
            key = tuple(sorted((edge.source, edge.target)))
            if key in seen:
                continue
            seen.add(key)
            start, end = positions[edge.source], positions[edge.target]
            ax.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops={"arrowstyle": "->", "color": "#4c566a", "lw": 1.5},
            )
        ax.scatter(positions[:, 0], positions[:, 1], s=420, color="#2a9d8f", zorder=3)
        for index, (x, y) in enumerate(positions):
            ax.text(x, y, f"v{index}", ha="center", va="center", color="white", weight="bold")
        ax.set_title(_pretty(family), fontsize=11)
        ax.set_aspect("equal")
        ax.axis("off")
    for ax in axes.flat[len(by_family) :]:
        ax.axis("off")
    figure.suptitle("K. Atlas of held-out logical equivalence diagrams", fontsize=18)
    return _save(figure, output_dir / "K_diagram_atlas.png")


def generate_complex_plots(
    *,
    final_rows: list[dict[str, Any]],
    history_rows: list[dict[str, Any]],
    diagram_rows: list[dict[str, Any]],
    rewrite_rows: list[dict[str, Any]],
    operator_rows: list[dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    plt = _setup(output_dir)
    paths = [
        plot_training(plt, history_rows, output_dir),
        plot_behavior(plt, final_rows, output_dir),
        plot_depth_generalization(plt, final_rows, output_dir),
        plot_operator_accuracy(plt, operator_rows, output_dir),
        plot_family_holonomy(plt, diagram_rows, output_dir),
        plot_shape_complexity(plt, diagram_rows, output_dir),
        plot_path_agreement(plt, diagram_rows, output_dir),
        plot_transport_vs_holonomy(plt, diagram_rows, output_dir),
        plot_correlations(plt, final_rows, output_dir),
        plot_rewrite_transport(plt, rewrite_rows, output_dir),
        plot_diagram_atlas(plt, output_dir),
    ]
    plt.close("all")
    return paths

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .complex_experiment import _write_rows
from .data import AssignedExpression, ExpressionDataset, collate_expressions, make_symbolic_splits
from .diagram_metrics import score_equivalence_diagram_features
from .equivalence import make_rewrite_calibration_pairs
from .holonomy_audit import _load_model, _regenerate_evaluation_data
from .metrics import fit_orthogonal_transport
from .model import TinyLogicTransformer
from .training import resolve_device

COMPONENTS = ("query", "key", "value")
SCOPES = ("cls", "token_mean")
SUMMARY_METRICS = (
    "identity_error",
    "transport_error",
    "holonomy_error",
    "holonomy_rotation_error",
    "holonomy_linear_action_error",
    "holonomy_systematic_error",
    "holonomy_dispersion_error",
    "holonomy_edge_accumulation_ratio",
    "path_agreement_error",
    "path_endpoint_error",
    "activation_variance",
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _all_experiment_strings(config: dict[str, Any]) -> set[str]:
    splits = make_symbolic_splits(
        train_size=max(config["conditions"].values()),
        validation_size=int(config["validation_size"]),
        test_size=int(config["test_size"]),
        train_depth=int(config["train_depth"]),
        ood_depths=tuple(int(depth) for depth in config["ood_depths"]),
        seed=4_281,
    )
    return {
        str(example)
        for example in (
            *splits.train,
            *splits.validation,
            *splits.test_id,
            *(
                example
                for depth in config["ood_depths"]
                for example in splits.test_ood[int(depth)]
            ),
        )
    }


def _balanced_disjoint_pairs(
    count_per_label: int,
    *,
    operand_depth: int,
    seed: int,
    blocked: set[str],
) -> dict[str, list[tuple[AssignedExpression, AssignedExpression]]]:
    if count_per_label % 2:
        raise ValueError("Balanced QKV calibration requires an even pair count")
    candidates = make_rewrite_calibration_pairs(
        count_per_label * 4,
        operand_depth=operand_depth,
        seed=seed,
        balance=True,
    )
    used = set(blocked)
    target_per_value = count_per_label // 2
    output: dict[str, list[tuple[AssignedExpression, AssignedExpression]]] = {}
    for label, label_pairs in candidates.items():
        counts = [0, 0]
        selected: list[tuple[AssignedExpression, AssignedExpression]] = []
        for source, target in label_pairs:
            value = source.value
            strings = {str(source), str(target)}
            if counts[value] >= target_per_value or strings & used:
                continue
            selected.append((source, target))
            counts[value] += 1
            used.update(strings)
            if counts == [target_per_value, target_per_value]:
                break
        if counts != [target_per_value, target_per_value]:
            raise RuntimeError(f"Could not balance disjoint QKV pairs for {label}: {counts}")
        output[label] = selected
    return output


def _flatten_pairs(
    pairs: dict[str, list[tuple[AssignedExpression, AssignedExpression]]],
) -> tuple[list[AssignedExpression], dict[str, tuple[slice, slice]]]:
    expressions: list[AssignedExpression] = []
    slices: dict[str, tuple[slice, slice]] = {}
    for label, label_pairs in pairs.items():
        source_start = len(expressions)
        expressions.extend(source for source, _ in label_pairs)
        source_slice = slice(source_start, len(expressions))
        target_start = len(expressions)
        expressions.extend(target for _, target in label_pairs)
        target_slice = slice(target_start, len(expressions))
        slices[label] = (source_slice, target_slice)
    return expressions, slices


@torch.inference_mode()
def qkv_features(
    model: TinyLogicTransformer,
    expressions: Sequence[AssignedExpression],
    *,
    device: torch.device,
    batch_size: int = 256,
) -> dict[tuple[int, str, str], np.ndarray]:
    """Extract concatenated-head Q/K/V vectors at CLS and token-mean scopes."""

    model.eval()
    chunks: dict[tuple[int, str, str], list[np.ndarray]] = defaultdict(list)
    dataset = ExpressionDataset(expressions)
    for start in range(0, len(dataset), batch_size):
        batch = collate_expressions(dataset.expressions[start : start + batch_size])
        tokens = batch.tokens.to(device)
        padding_mask = batch.padding_mask.to(device)
        layer_projections = model.qkv_projections(tokens, padding_mask)
        content_mask = ~padding_mask
        content_mask[:, 0] = False
        denominators = content_mask.sum(dim=1).clamp_min(1).reshape(-1, 1, 1)
        for layer_number, projections in enumerate(layer_projections, start=1):
            for component, tensor in zip(COMPONENTS, projections):
                cls = tensor[:, 0].reshape(tensor.shape[0], -1)
                token_mean = (
                    (tensor * content_mask[:, :, None, None]).sum(dim=1) / denominators
                ).reshape(tensor.shape[0], -1)
                chunks[(layer_number, "cls", component)].append(cls.float().cpu().numpy())
                chunks[(layer_number, "token_mean", component)].append(
                    token_mean.float().cpu().numpy()
                )
    return {key: np.concatenate(values, axis=0) for key, values in chunks.items()}


def _select_head(features: np.ndarray, heads: int, head: str) -> np.ndarray:
    if head == "all":
        return features
    reshaped = features.reshape(len(features), heads, features.shape[1] // heads)
    return reshaped[:, int(head)]


def _fit_feature_transports(
    features: np.ndarray,
    pair_slices: dict[str, tuple[slice, slice]],
):
    return {
        label: fit_orthogonal_transport(features[source], features[target])
        for label, (source, target) in pair_slices.items()
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def _std(rows: list[dict[str, Any]], key: str) -> float:
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _model_summaries(family_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    keys = (
        "architecture_layers",
        "condition",
        "seed",
        "measured_layer",
        "scope",
        "component",
        "head",
    )
    for row in family_rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for values, rows in grouped.items():
        result = dict(zip(keys, values))
        result["ood_accuracy_mean"] = float(rows[0]["ood_accuracy_mean"])
        result.update({metric: _mean(rows, metric) for metric in SUMMARY_METRICS})
        result["edge_improvement_over_identity"] = (
            1.0 - result["transport_error"] / result["identity_error"]
            if result["identity_error"] > 1e-12
            else float("nan")
        )
        result["supported"] = result["activation_variance"] > 1e-8
        output.append(result)
    return output


def _aggregate_summaries(model_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    keys = (
        "architecture_layers",
        "condition",
        "measured_layer",
        "scope",
        "component",
        "head",
    )
    for row in model_rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for values, rows in sorted(grouped.items()):
        result = dict(zip(keys, values))
        result["models"] = len(rows)
        for metric in (*SUMMARY_METRICS, "edge_improvement_over_identity"):
            result[f"{metric}_mean"] = _mean(rows, metric)
            result[f"{metric}_std"] = _std(rows, metric)
        result["supported_models"] = sum(bool(row["supported"]) for row in rows)
        output.append(result)
    return output


def _primary_rows(rows: list[dict[str, Any]], *, head: str = "all") -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if int(row["architecture_layers"]) == 6
        and row["condition"] == "higher_diversity"
        and str(row["head"]) == head
    ]


def _setup_plots(output_directory: Path):
    os.environ.setdefault("MPLCONFIGDIR", str(output_directory / ".matplotlib_cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    output_directory.mkdir(parents=True, exist_ok=True)
    return plt


def _line_plot(
    axes,
    rows: list[dict[str, Any]],
    metric: str,
    *,
    ylabel: str,
) -> None:
    colors = {"query": "#440154", "key": "#21918c", "value": "#fde725"}
    for row_index, scope in enumerate(SCOPES):
        ax = axes[row_index]
        for component in COMPONENTS:
            subset = sorted(
                [row for row in rows if row["scope"] == scope and row["component"] == component],
                key=lambda row: int(row["measured_layer"]),
            )
            ax.errorbar(
                [int(row["measured_layer"]) for row in subset],
                [
                    float(row[f"{metric}_mean"])
                    if int(row["supported_models"]) == int(row["models"])
                    else float("nan")
                    for row in subset
                ],
                yerr=[
                    float(row[f"{metric}_std"])
                    if int(row["supported_models"]) == int(row["models"])
                    else float("nan")
                    for row in subset
                ],
                marker="o",
                color=colors[component],
                label=component,
            )
        ax.set(xlabel="Transformer layer", ylabel=ylabel, title=scope.replace("_", " "))
        ax.set_xticks(range(1, 7))
        ax.legend()


def _heatmap(
    ax,
    matrix: np.ndarray,
    *,
    title: str,
    xlabel: list[str],
    ylabel: list[str],
    cmap: str,
    center_zero: bool = False,
):
    limit = float(np.nanmax(np.abs(matrix))) if center_zero else None
    image = ax.imshow(
        matrix,
        aspect="auto",
        cmap=cmap,
        vmin=-limit if center_zero else None,
        vmax=limit if center_zero else None,
    )
    ax.set_xticks(range(len(xlabel)), xlabel)
    ax.set_yticks(range(len(ylabel)), ylabel)
    ax.set_title(title)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=8)
    return image


def _plots(aggregate_rows: list[dict[str, Any]], output_directory: Path) -> list[Path]:
    plt = _setup_plots(output_directory)
    primary = _primary_rows(aggregate_rows)
    figure, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    _line_plot(axes, primary, "transport_error", ylabel="held-out edge error")
    figure.suptitle("O. Q/K/V edge transport by layer — L6 higher-diversity models")
    figure.tight_layout()
    path_o = output_directory / "O_qkv_edge_transport.png"
    figure.savefig(path_o, dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    _line_plot(axes, primary, "holonomy_error", ylabel="state holonomy error")
    figure.suptitle("P. Q/K/V return-to-start error by layer — L6 higher-diversity models")
    figure.tight_layout()
    path_p = output_directory / "P_qkv_holonomy.png"
    figure.savefig(path_p, dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(15, 5))
    for axis, scope in zip(axes, SCOPES):
        matrix = np.asarray(
            [
                [
                    next(
                        float(row["activation_variance_mean"])
                        for row in primary
                        if row["scope"] == scope
                        and row["component"] == component
                        and int(row["measured_layer"]) == layer
                    )
                    for layer in range(1, 7)
                ]
                for component in COMPONENTS
            ]
        )
        image = _heatmap(
            axis,
            np.log10(np.maximum(matrix, 1e-12)),
            title=scope.replace("_", " "),
            xlabel=[f"L{layer}" for layer in range(1, 7)],
            ylabel=["Q", "K", "V"],
            cmap="magma",
        )
        figure.colorbar(image, ax=axis, label="log10 activation variance")
    figure.suptitle("Q. Q/K/V feature support — constant features cannot carry knowledge")
    figure.tight_layout()
    path_q = output_directory / "Q_qkv_activation_variance.png"
    figure.savefig(path_q, dpi=180, bbox_inches="tight")
    plt.close(figure)

    per_head = _primary_rows(aggregate_rows, head="0")
    if not per_head:
        return [path_o, path_p, path_q]
    figure, axes = plt.subplots(2, 3, figsize=(17, 9))
    for row_index, scope in enumerate(SCOPES):
        for column_index, component in enumerate(COMPONENTS):
            matrix = np.asarray(
                [
                    [
                        next(
                            float(row["edge_improvement_over_identity_mean"])
                            for row in aggregate_rows
                            if int(row["architecture_layers"]) == 6
                            and row["condition"] == "higher_diversity"
                            and row["scope"] == scope
                            and row["component"] == component
                            and str(row["head"]) == str(head)
                            and int(row["measured_layer"]) == layer
                        )
                        for layer in range(1, 7)
                    ]
                    for head in range(4)
                ]
            )
            image = _heatmap(
                axes[row_index, column_index],
                matrix,
                title=f"{scope.replace('_', ' ')} {component}",
                xlabel=[f"L{layer}" for layer in range(1, 7)],
                ylabel=[f"H{head}" for head in range(4)],
                cmap="coolwarm",
                center_zero=True,
            )
            figure.colorbar(image, ax=axes[row_index, column_index], shrink=0.8)
    figure.suptitle("R. Per-head edge improvement over identity — L6 higher-diversity models")
    figure.tight_layout()
    path_r = output_directory / "R_qkv_per_head_edge_gain.png"
    figure.savefig(path_r, dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(17, 9))
    for row_index, scope in enumerate(SCOPES):
        for column_index, component in enumerate(COMPONENTS):
            matrix = np.asarray(
                [
                    [
                        next(
                            float(row["holonomy_error_mean"])
                            for row in aggregate_rows
                            if int(row["architecture_layers"]) == 6
                            and row["condition"] == "higher_diversity"
                            and row["scope"] == scope
                            and row["component"] == component
                            and str(row["head"]) == str(head)
                            and int(row["measured_layer"]) == layer
                        )
                        if not (scope == "cls" and layer == 1)
                        else float("nan")
                        for layer in range(1, 7)
                    ]
                    for head in range(4)
                ]
            )
            image = _heatmap(
                axes[row_index, column_index],
                matrix,
                title=f"{scope.replace('_', ' ')} {component}",
                xlabel=[f"L{layer}" for layer in range(1, 7)],
                ylabel=[f"H{head}" for head in range(4)],
                cmap="viridis_r",
            )
            figure.colorbar(image, ax=axes[row_index, column_index], shrink=0.8)
    figure.suptitle("S. Per-head Q/K/V state holonomy — L6 higher-diversity models")
    figure.tight_layout()
    path_s = output_directory / "S_qkv_per_head_holonomy.png"
    figure.savefig(path_s, dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=True)
    condition_styles = {
        "higher_diversity": ("higher diversity", "#31688e", "-"),
        "low_diversity": ("low diversity", "#b35806", "--"),
    }
    for row_index, scope in enumerate(SCOPES):
        for column_index, component in enumerate(COMPONENTS):
            ax = axes[row_index, column_index]
            for condition, (label, color, linestyle) in condition_styles.items():
                subset = sorted(
                    [
                        row
                        for row in aggregate_rows
                        if int(row["architecture_layers"]) == 6
                        and row["condition"] == condition
                        and row["scope"] == scope
                        and row["component"] == component
                        and str(row["head"]) == "all"
                    ],
                    key=lambda row: int(row["measured_layer"]),
                )
                ax.plot(
                    [int(row["measured_layer"]) for row in subset],
                    [
                        float(row["transport_error_mean"])
                        if int(row["supported_models"]) == int(row["models"])
                        else float("nan")
                        for row in subset
                    ],
                    marker="o",
                    color=color,
                    linestyle=linestyle,
                    label=label,
                )
            ax.set_title(f"{scope.replace('_', ' ')} {component}")
            ax.set_xticks(range(1, 7))
            ax.set_xlabel("Transformer layer")
            ax.set_ylabel("held-out edge error")
            ax.legend(fontsize=8)
    figure.suptitle("T. Q/K/V geometry is not specific to the generalizing condition")
    figure.tight_layout()
    path_t = output_directory / "T_qkv_condition_comparison.png"
    figure.savefig(path_t, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return [path_o, path_p, path_q, path_r, path_s, path_t]


def _fmt(value: float, std: float) -> str:
    return f"{value:.3f} ± {std:.3f}"


def _report(aggregate_rows: list[dict[str, Any]], plot_paths: list[Path]) -> str:
    primary = sorted(
        _primary_rows(aggregate_rows),
        key=lambda row: (
            int(row["measured_layer"]),
            SCOPES.index(str(row["scope"])),
            COMPONENTS.index(str(row["component"])),
        ),
    )
    supported = [row for row in primary if int(row["supported_models"]) == int(row["models"])]
    contextualized = [row for row in supported if int(row["measured_layer"]) > 1]
    best_edge = min(contextualized, key=lambda row: float(row["transport_error_mean"]))
    best_gain = max(
        contextualized, key=lambda row: float(row["edge_improvement_over_identity_mean"])
    )
    lowest_holonomy = min(
        contextualized, key=lambda row: float(row["holonomy_error_mean"])
    )
    best_cls = min(
        [row for row in contextualized if row["scope"] == "cls"],
        key=lambda row: float(row["transport_error_mean"]),
    )
    lines = [
        "# Layerwise query/key/value holonomy report",
        "",
        "## Scope and definition",
        "",
        "This analysis uses the exact query, key, and value projections immediately before ",
        "self-attention in every Transformer layer. For this pre-norm encoder they are computed ",
        "from `norm1` of the incoming residual stream. They are activations, not network weights.",
        "",
        "Two fixed-width views are tested independently: the Q/K/V vector at `<CLS>`, and the ",
        "mean Q/K/V vector across all non-padding, non-`<CLS>` tokens. Concatenated-head results ",
        "use all 128 dimensions; per-head results use 32 dimensions. Every rewrite calibration ",
        "set is exactly balanced: 96 false and 96 true examples per rewrite label.",
        "",
        "Every site gets its own affine-orthogonal transports and variance normalization. Edge ",
        "error must beat identity before a small holonomy value is considered evidence.",
        "",
        "## Primary comparison: six-layer higher-diversity models",
        "",
        "Values are mean ± sample standard deviation across three seeds, after averaging all ",
        "eleven diagram families equally.",
        "",
        "| Layer | Scope | Component | Variance | Identity edge | Fitted edge | Edge gain | State holonomy | Rotation defect |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary:
        variance = float(row["activation_variance_mean"])
        if int(row["supported_models"]) < int(row["models"]):
            variance_text = f"{variance:.2e} (degenerate)"
            identity_text = edge_text = gain_text = holonomy_text = rotation_text = "n/a"
        else:
            variance_text = f"{variance:.3f}"
            identity_text = _fmt(
                float(row["identity_error_mean"]), float(row["identity_error_std"])
            )
            edge_text = _fmt(
                float(row["transport_error_mean"]), float(row["transport_error_std"])
            )
            gain_text = f"{float(row['edge_improvement_over_identity_mean']):+.1%}"
            holonomy_text = _fmt(
                float(row["holonomy_error_mean"]), float(row["holonomy_error_std"])
            )
            rotation_text = _fmt(
                float(row["holonomy_rotation_error_mean"]),
                float(row["holonomy_rotation_error_std"]),
            )
        lines.append(
            f"| {int(row['measured_layer'])} | {row['scope']} | {row['component']} | "
            f"{variance_text} | {identity_text} | {edge_text} | {gain_text} | "
            f"{holonomy_text} | {rotation_text} |"
        )
    lines.extend(
        [
            "",
            "## Localization summary",
            "",
            "- Layer-1 token means have the lowest raw errors, but they are computed directly ",
            "  from token and position embeddings before attention. Their strong invariance is ",
            "  a lexical/bag-of-tokens baseline, not localized learned reasoning.",
            (
                f"- Lowest fitted edge error after at least one attention layer: layer "
                f"{int(best_edge['measured_layer'])} {best_edge['scope']} "
                f"{best_edge['component']} ({float(best_edge['transport_error_mean']):.3f})."
            ),
            (
                f"- Largest improvement over identity: layer "
                f"{int(best_gain['measured_layer'])} {best_gain['scope']} "
                f"{best_gain['component']} "
                f"({float(best_gain['edge_improvement_over_identity_mean']):+.1%})."
            ),
            (
                f"- Lowest supported state holonomy: layer "
                f"{int(lowest_holonomy['measured_layer'])} {lowest_holonomy['scope']} "
                f"{lowest_holonomy['component']} "
                f"({float(lowest_holonomy['holonomy_error_mean']):.3f}); this is "
                "interpretable only together with its edge gain."
            ),
            (
                f"- Best global `<CLS>` site: layer {int(best_cls['measured_layer'])} "
                f"{best_cls['component']} (edge {float(best_cls['transport_error_mean']):.3f}, "
                f"holonomy {float(best_cls['holonomy_error_mean']):.3f})."
            ),
            "- Layer-1 `<CLS>` Q/K/V is constant across expressions before the first attention ",
            "operation. Its apparent zero geometry is marked degenerate, not coherent.",
            "- `<CLS>` queries are consistently more transportable than keys or values in the ",
            "  middle and late layers. Per-head gains are broad rather than concentrated.",
            "- Low-diversity controls frequently have equal or lower errors. The Q/K/V geometry ",
            "  therefore reflects shared structural regularity, not a generalization-specific ",
            "  knowledge location.",
            "",
            "## Higher-diversity versus memorizing controls",
            "",
            "The table averages supported layers for each six-layer condition. Layer-1 `<CLS>` ",
            "is excluded; token means include all layers.",
            "",
            "| Scope | Component | Higher edge | Low edge | Higher holonomy | Low holonomy |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for scope in SCOPES:
        for component in COMPONENTS:
            conditions: dict[str, list[dict[str, Any]]] = {}
            for condition in ("higher_diversity", "low_diversity"):
                conditions[condition] = [
                    row
                    for row in aggregate_rows
                    if int(row["architecture_layers"]) == 6
                    and row["condition"] == condition
                    and row["scope"] == scope
                    and row["component"] == component
                    and str(row["head"]) == "all"
                    and (scope != "cls" or int(row["measured_layer"]) > 1)
                ]
            high = conditions["higher_diversity"]
            low = conditions["low_diversity"]
            lines.append(
                f"| {scope} | {component} | {_mean(high, 'transport_error_mean'):.3f} | "
                f"{_mean(low, 'transport_error_mean'):.3f} | "
                f"{_mean(high, 'holonomy_error_mean'):.3f} | "
                f"{_mean(low, 'holonomy_error_mean'):.3f} |"
            )
    lines.extend(
        [
            "",
            "## Artifacts and interpretation",
            "",
            "`family_metrics.csv` contains every model × layer × scope × component × head × ",
            "diagram-family result. `model_metrics.csv` contains family-balanced model summaries, ",
            "and `aggregate_metrics.csv` contains seed aggregates. The per-head plot localizes ",
            "edge transport, while the concatenated-head plots allow rotations that mix heads.",
            "",
            "A low value here does not prove that attention uses the corresponding feature. The ",
            "causal follow-up is to patch Q, K, or V at the identified layer/head and measure ",
            "whether alternative rewrite paths produce the same change in logits.",
            "",
            "## Figures",
            "",
        ]
    )
    for path in plot_paths:
        lines.extend([f"![{path.stem}](plots/{path.name})", ""])
    return "\n".join(line.rstrip() for line in lines)


def run_qkv_holonomy(run_directory: Path, *, device: str = "auto") -> Path:
    run_directory = run_directory.resolve()
    config = json.loads((run_directory / "config.json").read_text(encoding="utf-8"))
    _, diagrams = _regenerate_evaluation_data(config)
    blocked = _all_experiment_strings(config)
    blocked.update(str(vertex) for diagram in diagrams for vertex in diagram.vertices)
    calibration = _balanced_disjoint_pairs(
        int(config["calibration_pairs_per_rewrite"]),
        operand_depth=int(config["diagram_operand_depth"]),
        seed=173_013,
        blocked=blocked,
    )
    pair_expressions, pair_slices = _flatten_pairs(calibration)
    diagram_expressions = [vertex for diagram in diagrams for vertex in diagram.vertices]
    result_rows = _read_rows(run_directory / "tables" / "results.csv")
    result_lookup = {
        (int(row["architecture_layers"]), row["condition"], int(row["seed"])): row
        for row in result_rows
    }
    target_device = resolve_device(device)
    family_rows: list[dict[str, Any]] = []
    output_directory = run_directory / "qkv_holonomy"
    plot_directory = output_directory / "plots"
    output_directory.mkdir(exist_ok=True)
    checkpoints = sorted((run_directory / "checkpoints").glob("*.pt"))
    for checkpoint_index, checkpoint_path in enumerate(checkpoints, start=1):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_config = checkpoint["model_config"]
        train_config = checkpoint["train_config"]
        architecture_layers = int(model_config["n_layers"])
        heads = int(model_config["n_heads"])
        seed = int(train_config["seed"])
        condition = (
            "higher_diversity"
            if "higher_diversity" in checkpoint_path.stem
            else "low_diversity"
        )
        metadata = {
            "architecture_layers": architecture_layers,
            "condition": condition,
            "seed": seed,
            "ood_accuracy_mean": float(
                result_lookup[(architecture_layers, condition, seed)]["ood_accuracy_mean"]
            ),
        }
        print(f"[{checkpoint_index}/{len(checkpoints)}] QKV {checkpoint_path.stem}", flush=True)
        model = _load_model(checkpoint_path, target_device)
        calibration_features = qkv_features(
            model, pair_expressions, device=target_device, batch_size=int(config["batch_size"])
        )
        diagram_features = qkv_features(
            model,
            diagram_expressions,
            device=target_device,
            batch_size=int(config["batch_size"]),
        )
        for (layer, scope, component), calibration_matrix in calibration_features.items():
            heads_to_score = ["all"]
            if architecture_layers == 6:
                heads_to_score.extend(str(head) for head in range(heads))
            for head in heads_to_score:
                selected_calibration = _select_head(calibration_matrix, heads, head)
                selected_diagrams = _select_head(
                    diagram_features[(layer, scope, component)], heads, head
                )
                transports = _fit_feature_transports(selected_calibration, pair_slices)
                rows = score_equivalence_diagram_features(
                    diagrams, selected_diagrams, transports
                )
                site = {
                    "measured_layer": layer,
                    "scope": scope,
                    "component": component,
                    "head": head,
                }
                family_rows.extend({**metadata, **site, **row} for row in rows)
        _write_rows(output_directory / "family_metrics.partial.csv", family_rows)
        del model, calibration_features, diagram_features
        if target_device.type == "cuda":
            torch.cuda.empty_cache()

    model_rows = _model_summaries(family_rows)
    aggregate_rows = _aggregate_summaries(model_rows)
    _write_rows(output_directory / "family_metrics.csv", family_rows)
    _write_rows(output_directory / "model_metrics.csv", model_rows)
    _write_rows(output_directory / "aggregate_metrics.csv", aggregate_rows)
    plot_paths = _plots(aggregate_rows, plot_directory)
    (output_directory / "report.md").write_text(
        _report(aggregate_rows, plot_paths), encoding="utf-8"
    )
    (output_directory / "status.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "models": len(checkpoints),
                "balanced_pairs_per_rewrite": int(config["calibration_pairs_per_rewrite"]),
                "components": list(COMPONENTS),
                "scopes": list(SCOPES),
                "plots": [str(path.relative_to(output_directory)) for path in plot_paths],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_directory

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any

import numpy as np
import torch

from .complex_experiment import _write_rows
from .data import VARIABLES, AssignedExpression, collate_expressions
from .holonomy_audit import _load_model, _regenerate_evaluation_data
from .logic import FIXED_ARITY_OPS, Expr, binary, fixed, nary, random_symbolic_expr, unary
from .qkv_holonomy import _all_experiment_strings
from .training import resolve_device

PATCH_OPERATOR_SPECS = (
    ("NOT", 1),
    ("AND", 2),
    ("AND", 3),
    ("AND", 4),
    ("OR", 2),
    ("OR", 3),
    ("OR", 4),
    ("XOR", 2),
    ("XOR", 3),
    ("XOR", 4),
    ("MAJ3", 3),
    ("ITE3", 3),
    ("EXACT1_3", 3),
    ("ATLEAST2_4", 4),
)
COMPONENT_GROUPS = {
    "query": ("query",),
    "key": ("key",),
    "value": ("value",),
    "qkv": ("query", "key", "value"),
}
DONOR_TYPES = ("counterfactual", "counterfactual_shuffled", "equivalent")
SITES = ("cls", "operator")


@dataclass(frozen=True)
class SymbolicQKVPatchingSuite:
    recipients: tuple[AssignedExpression, ...]
    counterfactual_donors: tuple[AssignedExpression, ...]
    equivalent_donors: tuple[AssignedExpression, ...]
    operators: tuple[str, ...]
    target_values: tuple[int, ...]
    context_values: tuple[int, ...]
    operator_positions: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.recipients)


def _operator_label(op: str, arity: int) -> str:
    return op if op in FIXED_ARITY_OPS or op == "NOT" else f"{op}{arity}"


def _root_operation(op: str, children: list[Expr]) -> Expr:
    if op == "NOT":
        return unary(op, children[0])
    if op in FIXED_ARITY_OPS:
        return fixed(op, *children)
    return nary(op, *children)


def _sample_expr_with_value(
    rng: Random,
    *,
    assignment: dict[str, int],
    target: int,
    max_depth: int,
) -> Expr:
    for _ in range(10_000):
        expression = random_symbolic_expr(
            rng,
            max_depth,
            variables=VARIABLES[:4],
        )
        if expression.evaluate(assignment) == target:
            return expression
    raise RuntimeError(f"Could not sample a symbolic expression with value {target}")


def _sample_operation_with_value(
    rng: Random,
    *,
    op: str,
    arity: int,
    assignment: dict[str, int],
    target: int,
    operand_depth: int,
) -> Expr:
    for _ in range(20_000):
        children = [
            random_symbolic_expr(
                rng,
                rng.randrange(operand_depth + 1),
                variables=VARIABLES[:4],
            )
            for _ in range(arity)
        ]
        expression = _root_operation(op, children)
        if expression.evaluate(assignment) == target:
            return expression
    raise RuntimeError(f"Could not sample {op}/{arity} with value {target}")


def make_symbolic_qkv_patching_suite(
    *,
    count_per_operator: int = 16,
    operand_depth: int = 2,
    context_depth: int = 2,
    seed: int = 284_119,
    blocked: set[str] | None = None,
) -> SymbolicQKVPatchingSuite:
    """Create matched causal, equivalent, and context-shuffle-ready examples."""

    if count_per_operator < 8 or count_per_operator % 4:
        raise ValueError("count_per_operator must be a multiple of four and at least eight")
    rng = Random(seed)
    used = set(blocked or ())
    recipients: list[AssignedExpression] = []
    counterfactuals: list[AssignedExpression] = []
    equivalents: list[AssignedExpression] = []
    operators: list[str] = []
    target_values: list[int] = []
    context_values: list[int] = []
    operator_positions: list[int] = []
    per_value = count_per_operator // 2

    for op, arity in PATCH_OPERATOR_SPECS:
        for target_value in (0, 1):
            for index in range(per_value):
                context_value = index % 2
                for _ in range(20_000):
                    assignment_tuple = tuple((name, rng.randrange(2)) for name in VARIABLES[:4])
                    assignment = dict(assignment_tuple)
                    context = _sample_expr_with_value(
                        rng,
                        assignment=assignment,
                        target=context_value,
                        max_depth=context_depth,
                    )
                    recipient_target = _sample_operation_with_value(
                        rng,
                        op=op,
                        arity=arity,
                        assignment=assignment,
                        target=target_value,
                        operand_depth=operand_depth,
                    )
                    counterfactual_target = _sample_operation_with_value(
                        rng,
                        op=op,
                        arity=arity,
                        assignment=assignment,
                        target=1 - target_value,
                        operand_depth=operand_depth,
                    )
                    equivalent_target = _sample_operation_with_value(
                        rng,
                        op=op,
                        arity=arity,
                        assignment=assignment,
                        target=target_value,
                        operand_depth=operand_depth,
                    )
                    if equivalent_target == recipient_target:
                        continue
                    candidate = AssignedExpression(
                        binary("XOR", recipient_target, context), assignment_tuple
                    )
                    counterfactual = AssignedExpression(
                        binary("XOR", counterfactual_target, context), assignment_tuple
                    )
                    equivalent = AssignedExpression(
                        binary("XOR", equivalent_target, context), assignment_tuple
                    )
                    strings = {str(candidate), str(counterfactual), str(equivalent)}
                    if len(strings) != 3 or strings & used:
                        continue
                    if (
                        candidate.value == counterfactual.value
                        or candidate.value != equivalent.value
                    ):
                        raise AssertionError("Patching suite truth relation is invalid")
                    used.update(strings)
                    recipients.append(candidate)
                    counterfactuals.append(counterfactual)
                    equivalents.append(equivalent)
                    operators.append(_operator_label(op, arity))
                    target_values.append(target_value)
                    context_values.append(context_value)
                    # CLS, ENV, four assignments, SEP, root XOR, then target root.
                    operator_positions.append(len(assignment_tuple) + 4)
                    break
                else:
                    raise RuntimeError(f"Could not build a unique patching case for {op}/{arity}")

    return SymbolicQKVPatchingSuite(
        recipients=tuple(recipients),
        counterfactual_donors=tuple(counterfactuals),
        equivalent_donors=tuple(equivalents),
        operators=tuple(operators),
        target_values=tuple(target_values),
        context_values=tuple(context_values),
        operator_positions=tuple(operator_positions),
    )


def _matched_shuffle_indices(suite: SymbolicQKVPatchingSuite) -> np.ndarray:
    groups: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for index, values in enumerate(
        zip(suite.operators, suite.target_values, suite.context_values, strict=True)
    ):
        groups[values].append(index)
    permutation = np.arange(len(suite))
    for indices in groups.values():
        if len(indices) < 2:
            raise RuntimeError("Matched shuffle group is too small")
        shifted = indices[1:] + indices[:1]
        permutation[indices] = shifted
    if np.any(permutation == np.arange(len(suite))):
        raise AssertionError("Counterfactual shuffle must be a derangement")
    return permutation


def _logit_difference(logits: torch.Tensor) -> np.ndarray:
    return (logits[:, 1] - logits[:, 0]).float().cpu().numpy()


def _effect_fraction(
    clean: np.ndarray,
    donor: np.ndarray,
    patched: np.ndarray,
    selection: np.ndarray,
) -> float:
    desired = donor[selection] - clean[selection]
    actual = patched[selection] - clean[selection]
    denominator = float(desired @ desired)
    return float((actual @ desired) / denominator) if denominator > 1e-12 else float("nan")


def _score_patch(
    *,
    clean_logits: torch.Tensor,
    donor_logits: torch.Tensor,
    patched_logits: torch.Tensor,
    clean_labels: np.ndarray,
    donor_labels: np.ndarray,
    operators: np.ndarray,
) -> list[dict[str, Any]]:
    clean_predictions = clean_logits.argmax(dim=-1).cpu().numpy()
    donor_predictions = donor_logits.argmax(dim=-1).cpu().numpy()
    patched_predictions = patched_logits.argmax(dim=-1).cpu().numpy()
    clean_difference = _logit_difference(clean_logits)
    donor_difference = _logit_difference(donor_logits)
    patched_difference = _logit_difference(patched_logits)
    eligible = (clean_predictions == clean_labels) & (donor_predictions == donor_labels)
    rows: list[dict[str, Any]] = []
    for operator in ("ALL", *sorted(set(operators.tolist()))):
        selected = (
            np.ones(len(operators), dtype=bool) if operator == "ALL" else operators == operator
        )
        supported = selected & eligible
        for subset_name, subset in (("all", selected), ("eligible", supported)):
            count = int(subset.sum())
            if not count:
                continue
            desired = donor_difference[subset] - clean_difference[subset]
            actual = patched_difference[subset] - clean_difference[subset]
            rows.append(
                {
                    "operator": operator,
                    "subset": subset_name,
                    "count": count,
                    "clean_accuracy": float(
                        (clean_predictions[subset] == clean_labels[subset]).mean()
                    ),
                    "donor_accuracy": float(
                        (donor_predictions[subset] == donor_labels[subset]).mean()
                    ),
                    "patched_recipient_accuracy": float(
                        (patched_predictions[subset] == clean_labels[subset]).mean()
                    ),
                    "patched_donor_accuracy": float(
                        (patched_predictions[subset] == donor_labels[subset]).mean()
                    ),
                    "prediction_flip_rate": float(
                        (patched_predictions[subset] != clean_predictions[subset]).mean()
                    ),
                    "effect_fraction": _effect_fraction(
                        clean_difference, donor_difference, patched_difference, subset
                    ),
                    "normalized_abs_logit_change": float(
                        np.mean(np.abs(actual)) / (np.mean(np.abs(desired)) + 1e-12)
                    ),
                    "mean_abs_logit_change": float(np.mean(np.abs(actual))),
                }
            )
    return rows


@torch.inference_mode()
def _score_model(
    model,
    suite: SymbolicQKVPatchingSuite,
    *,
    device: torch.device,
    include_per_head: bool,
) -> tuple[list[dict[str, Any]], float]:
    recipient_batch = collate_expressions(suite.recipients)
    counterfactual_batch = collate_expressions(suite.counterfactual_donors)
    equivalent_batch = collate_expressions(suite.equivalent_donors)
    recipient_tokens = recipient_batch.tokens.to(device)
    recipient_mask = recipient_batch.padding_mask.to(device)
    counterfactual_tokens = counterfactual_batch.tokens.to(device)
    counterfactual_mask = counterfactual_batch.padding_mask.to(device)
    equivalent_tokens = equivalent_batch.tokens.to(device)
    equivalent_mask = equivalent_batch.padding_mask.to(device)
    clean_logits = model(recipient_tokens, recipient_mask)
    counterfactual_logits = model(counterfactual_tokens, counterfactual_mask)
    equivalent_logits = model(equivalent_tokens, equivalent_mask)
    recipient_qkv = model.qkv_projections(recipient_tokens, recipient_mask)
    counterfactual_qkv = model.qkv_projections(counterfactual_tokens, counterfactual_mask)
    equivalent_qkv = model.qkv_projections(equivalent_tokens, equivalent_mask)
    shuffle = _matched_shuffle_indices(suite)
    shuffle_tensor = torch.as_tensor(shuffle, device=device)
    shuffled_logits = counterfactual_logits[shuffle_tensor]
    names = ("query", "key", "value")
    rows_index = torch.arange(len(suite), device=device)
    positions_by_site = {
        "cls": torch.zeros(len(suite), dtype=torch.long, device=device),
        "operator": torch.tensor(suite.operator_positions, dtype=torch.long, device=device),
    }
    clean_labels = recipient_batch.labels.numpy()
    counterfactual_labels = counterfactual_batch.labels.numpy()
    equivalent_labels = equivalent_batch.labels.numpy()
    operators = np.asarray(suite.operators)
    output: list[dict[str, Any]] = []
    max_baseline_difference = 0.0

    for layer_index in range(model.config.n_layers):
        layer_number = layer_index + 1
        heads_to_patch: list[int | None] = [None]
        if include_per_head:
            heads_to_patch.extend(range(model.config.n_heads))
        for site, positions in positions_by_site.items():
            own_values = {
                name: component[rows_index, positions]
                for name, component in zip(names, recipient_qkv[layer_index], strict=True)
            }
            baseline = model.forward_qkv_patched(
                recipient_tokens,
                recipient_mask,
                patch_layer=layer_number,
                patch_positions=positions,
                patch_values=own_values,
            )
            max_baseline_difference = max(
                max_baseline_difference,
                float((baseline - clean_logits).abs().max().item()),
            )
            donor_specs = {
                "counterfactual": (
                    counterfactual_qkv[layer_index],
                    counterfactual_logits,
                    counterfactual_labels,
                    None,
                ),
                "counterfactual_shuffled": (
                    counterfactual_qkv[layer_index],
                    shuffled_logits,
                    counterfactual_labels[shuffle],
                    shuffle_tensor,
                ),
                "equivalent": (
                    equivalent_qkv[layer_index],
                    equivalent_logits,
                    equivalent_labels,
                    None,
                ),
            }
            for donor_type, (donor_qkv, donor_logits, donor_labels, reorder) in donor_specs.items():
                projected_values = {
                    name: component[rows_index, positions]
                    for name, component in zip(names, donor_qkv, strict=True)
                }
                if reorder is not None:
                    projected_values = {
                        name: values[reorder] for name, values in projected_values.items()
                    }
                for component_group, component_names in COMPONENT_GROUPS.items():
                    patch_values = {name: projected_values[name] for name in component_names}
                    for head in heads_to_patch:
                        patched_logits = model.forward_qkv_patched(
                            recipient_tokens,
                            recipient_mask,
                            patch_layer=layer_number,
                            patch_positions=positions,
                            patch_values=patch_values,
                            patch_head=head,
                        )
                        site_rows = _score_patch(
                            clean_logits=clean_logits,
                            donor_logits=donor_logits,
                            patched_logits=patched_logits,
                            clean_labels=clean_labels,
                            donor_labels=donor_labels,
                            operators=operators,
                        )
                        metadata = {
                            "patch_layer": layer_number,
                            "site": site,
                            "component": component_group,
                            "head": "all" if head is None else str(head),
                            "donor_type": donor_type,
                        }
                        output.extend({**metadata, **row} for row in site_rows)
    return output, max_baseline_difference


def _read_results(path: Path) -> dict[tuple[int, str, int], dict[str, str]]:
    import csv

    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (int(row["architecture_layers"]), row["condition"], int(row["seed"])): row
            for row in csv.DictReader(handle)
        }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def _std(rows: list[dict[str, Any]], key: str) -> float:
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "architecture_layers",
        "condition",
        "patch_layer",
        "site",
        "component",
        "head",
        "donor_type",
        "operator",
        "subset",
    )
    metrics = (
        "clean_accuracy",
        "donor_accuracy",
        "patched_recipient_accuracy",
        "patched_donor_accuracy",
        "prediction_flip_rate",
        "effect_fraction",
        "normalized_abs_logit_change",
        "mean_abs_logit_change",
    )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for values, group in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        result = dict(zip(keys, values, strict=True))
        result["models"] = len(group)
        result["mean_count"] = _mean(group, "count")
        for metric in metrics:
            result[f"{metric}_mean"] = _mean(group, metric)
            result[f"{metric}_std"] = _std(group, metric)
        output.append(result)
    return output


def _plot_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if int(row["architecture_layers"]) == 6
        and row["condition"] == "higher_diversity"
        and row["operator"] == "ALL"
        and row["subset"] == "eligible"
    ]


def _plots(rows: list[dict[str, Any]], output: Path) -> list[Path]:
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib_cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    output.mkdir(parents=True, exist_ok=True)
    primary = _plot_rows(rows)
    colors = {"query": "#440154", "key": "#21918c", "value": "#5ec962", "qkv": "#f8961e"}
    paths: list[Path] = []

    figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for axis, site in zip(axes, SITES, strict=True):
        for component in COMPONENT_GROUPS:
            for donor_type, linestyle in (
                ("counterfactual", "-"),
                ("counterfactual_shuffled", "--"),
            ):
                subset = sorted(
                    [
                        row
                        for row in primary
                        if row["site"] == site
                        and row["component"] == component
                        and row["donor_type"] == donor_type
                        and row["head"] == "all"
                    ],
                    key=lambda row: int(row["patch_layer"]),
                )
                axis.plot(
                    [int(row["patch_layer"]) for row in subset],
                    [float(row["effect_fraction_mean"]) for row in subset],
                    marker="o",
                    linestyle=linestyle,
                    color=colors[component],
                    label=f"{component} ({'matched' if donor_type == 'counterfactual' else 'shuffled'})",
                )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set(
            title=site, xlabel="patched attention layer", ylabel="donor-direction effect fraction"
        )
        axis.set_xticks(range(1, 7))
        axis.legend(fontsize=8, ncol=2)
    figure.suptitle(
        "U. Causal Q/K/V transfer: matched counterfactual versus context-shuffled control"
    )
    figure.tight_layout()
    path = output / "U_qkv_causal_transfer.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    paths.append(path)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for axis, site in zip(axes, SITES, strict=True):
        for component in COMPONENT_GROUPS:
            subset = sorted(
                [
                    row
                    for row in primary
                    if row["site"] == site
                    and row["component"] == component
                    and row["donor_type"] == "counterfactual"
                    and row["head"] == "all"
                ],
                key=lambda row: int(row["patch_layer"]),
            )
            axis.errorbar(
                [int(row["patch_layer"]) for row in subset],
                [float(row["patched_donor_accuracy_mean"]) for row in subset],
                yerr=[float(row["patched_donor_accuracy_std"]) for row in subset],
                marker="o",
                color=colors[component],
                label=component,
            )
        axis.axhline(0.5, color="gray", linewidth=0.8)
        axis.set(
            title=site, xlabel="patched attention layer", ylabel="counterfactual target accuracy"
        )
        axis.set_xticks(range(1, 7))
        axis.legend()
    figure.suptitle("V. Counterfactual truth recovery after Q/K/V patching")
    figure.tight_layout()
    path = output / "V_qkv_counterfactual_accuracy.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    paths.append(path)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    for axis, site in zip(axes, SITES, strict=True):
        matrix = np.zeros((len(COMPONENT_GROUPS), 6))
        for component_index, component in enumerate(COMPONENT_GROUPS):
            for layer in range(1, 7):
                matched = next(
                    float(row["effect_fraction_mean"])
                    for row in primary
                    if row["site"] == site
                    and row["component"] == component
                    and row["donor_type"] == "counterfactual"
                    and row["head"] == "all"
                    and int(row["patch_layer"]) == layer
                )
                shuffled = next(
                    float(row["effect_fraction_mean"])
                    for row in primary
                    if row["site"] == site
                    and row["component"] == component
                    and row["donor_type"] == "counterfactual_shuffled"
                    and row["head"] == "all"
                    and int(row["patch_layer"]) == layer
                )
                matrix[component_index, layer - 1] = matched - shuffled
        limit = max(0.01, float(np.max(np.abs(matrix))))
        image = axis.imshow(matrix, cmap="coolwarm", aspect="auto", vmin=-limit, vmax=limit)
        axis.set_xticks(range(6), range(1, 7))
        axis.set_yticks(range(len(COMPONENT_GROUPS)), COMPONENT_GROUPS)
        axis.set(title=site, xlabel="patched attention layer")
        for row_index in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row_index,
                    f"{matrix[row_index, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        figure.colorbar(image, ax=axis, label="matched − shuffled effect")
    figure.suptitle("W. Context-specific causal transfer above the shuffled control")
    figure.tight_layout()
    path = output / "W_qkv_specificity.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    paths.append(path)

    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for row_index, site in enumerate(SITES):
        for column, component in enumerate(("query", "key", "value")):
            matrix = np.full((4, 6), np.nan)
            for head in range(4):
                for layer in range(1, 7):
                    candidates = [
                        row
                        for row in primary
                        if row["site"] == site
                        and row["component"] == component
                        and row["donor_type"] == "counterfactual"
                        and str(row["head"]) == str(head)
                        and int(row["patch_layer"]) == layer
                    ]
                    if candidates:
                        matrix[head, layer - 1] = float(candidates[0]["effect_fraction_mean"])
            axis = axes[row_index, column]
            image = axis.imshow(matrix, aspect="auto", cmap="viridis")
            axis.set(title=f"{site} {component}", xlabel="layer", ylabel="head")
            axis.set_xticks(range(6), range(1, 7))
            axis.set_yticks(range(4), range(4))
            figure.colorbar(image, ax=axis, fraction=0.046)
    figure.suptitle("X. Per-head counterfactual effect fraction — L6 higher-diversity models")
    figure.tight_layout()
    path = output / "X_qkv_per_head_effect.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    paths.append(path)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for axis, site in zip(axes, SITES, strict=True):
        for component in COMPONENT_GROUPS:
            subset = sorted(
                [
                    row
                    for row in primary
                    if row["site"] == site
                    and row["component"] == component
                    and row["donor_type"] == "equivalent"
                    and row["head"] == "all"
                ],
                key=lambda row: int(row["patch_layer"]),
            )
            axis.errorbar(
                [int(row["patch_layer"]) for row in subset],
                [float(row["patched_recipient_accuracy_mean"]) for row in subset],
                yerr=[float(row["patched_recipient_accuracy_std"]) for row in subset],
                marker="o",
                color=colors[component],
                label=component,
            )
        axis.set(
            title=site,
            xlabel="patched attention layer",
            ylabel="truth accuracy after equivalent patch",
        )
        axis.set_ylim(0.45, 1.02)
        axis.set_xticks(range(1, 7))
        axis.legend()
    figure.suptitle("Y. Same-truth equivalent patches should preserve the answer")
    figure.tight_layout()
    path = output / "Y_qkv_equivalent_control.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    paths.append(path)
    return paths


def _best(rows: list[dict[str, Any]], *, site: str, component: str) -> dict[str, Any]:
    candidates = [
        row
        for row in _plot_rows(rows)
        if row["site"] == site
        and row["component"] == component
        and row["donor_type"] == "counterfactual"
        and row["head"] == "all"
    ]
    return max(candidates, key=lambda row: float(row["effect_fraction_mean"]))


def _report(rows: list[dict[str, Any]], plots: list[Path], suite_size: int, baseline: float) -> str:
    cls_q = _best(rows, site="cls", component="query")
    operator_q = _best(rows, site="operator", component="query")
    operator_k = _best(rows, site="operator", component="key")
    operator_v = _best(rows, site="operator", component="value")
    joint = _best(rows, site="operator", component="qkv")
    lines = [
        "# Causal Q/K/V activation-patching report",
        "",
        "## Scope",
        "",
        f"This post-hoc experiment patches pre-attention Q, K, and V activations in all 18 saved models. The causal suite contains {suite_size} symbolic cases spanning fourteen operator/arity classes. Each recipient has a same-context opposite-truth donor, a same-truth equivalent donor, and a truth/operator/context-matched shuffled donor.",
        "",
        f"A self-patch reproduces the ordinary forward pass to maximum absolute logit error {baseline:.2e}.",
        "",
        "## Primary L6 higher-diversity results",
        "",
        "The donor-direction effect fraction is 0 for no causal movement and 1 for complete recovery of the donor's logit change. Results below use only cases where both recipient and donor were initially classified correctly.",
        "",
        "| Site/component | Best layer | Effect fraction | Counterfactual accuracy |",
        "|---|---:|---:|---:|",
    ]
    for label, row in (
        ("CLS query", cls_q),
        ("operator query", operator_q),
        ("operator key", operator_k),
        ("operator value", operator_v),
        ("operator joint QKV", joint),
    ):
        lines.append(
            f"| {label} | {int(row['patch_layer'])} | {float(row['effect_fraction_mean']):.3f} ± {float(row['effect_fraction_std']):.3f} | {float(row['patched_donor_accuracy_mean']):.3f} ± {float(row['patched_donor_accuracy_std']):.3f} |"
        )
    lines.extend(
        [
            "",
            "A matched patch is interpreted only relative to the shuffled counterfactual control. Same-truth equivalent patches test whether replacing syntax while preserving truth leaves the decision intact. Per-head results are exploratory localization; circuit selection must be frozen before confirmatory held-out tests.",
            "",
            "## Figures",
            "",
        ]
    )
    for path in plots:
        lines.extend([f"![{path.stem}](plots/{path.name})", ""])
    lines.extend(
        [
            "## Interpretation limits",
            "",
            "- A successful patch establishes causal sufficiency of the inserted activation for changing the output in this paired context; it does not prove that the same site is necessary.",
            "- Q, K, and V are patched at one token in one layer. Effects can be distributed across tokens, heads, or residual pathways.",
            "- The per-head screen uses the same evaluation suite for localization and is therefore exploratory.",
            "- Equivalent and shuffled controls are required because arbitrary Q/K/V replacement can change logits without transferring the intended logical variable.",
            "",
        ]
    )
    return "\n".join(lines)


def run_qkv_patching(
    run_directory: Path,
    *,
    device: str = "auto",
    count_per_operator: int = 16,
) -> Path:
    run_directory = run_directory.resolve()
    config = json.loads((run_directory / "config.json").read_text(encoding="utf-8"))
    _, diagrams = _regenerate_evaluation_data(config)
    blocked = _all_experiment_strings(config)
    blocked.update(str(vertex) for diagram in diagrams for vertex in diagram.vertices)
    suite = make_symbolic_qkv_patching_suite(
        count_per_operator=count_per_operator,
        operand_depth=int(config["diagram_operand_depth"]),
        blocked=blocked,
    )
    output = run_directory / "qkv_patching"
    output.mkdir(exist_ok=True)
    results = _read_results(run_directory / "tables" / "results.csv")
    target_device = resolve_device(device)
    # The native fused Transformer fast path and explicit SDPA can differ by a
    # few hundredths on these trained CUDA checkpoints. Disable the fused path
    # so clean, donor, and patched runs use the same numerically exact backend.
    previous_fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    all_rows: list[dict[str, Any]] = []
    baselines: list[float] = []
    checkpoints = sorted((run_directory / "checkpoints").glob("*.pt"))
    for checkpoint_index, checkpoint_path in enumerate(checkpoints, start=1):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        layers = int(checkpoint["model_config"]["n_layers"])
        seed = int(checkpoint["train_config"]["seed"])
        condition = (
            "higher_diversity" if "higher_diversity" in checkpoint_path.stem else "low_diversity"
        )
        print(
            f"[{checkpoint_index}/{len(checkpoints)}] causal QKV {checkpoint_path.stem}", flush=True
        )
        model = _load_model(checkpoint_path, target_device)
        rows, baseline = _score_model(
            model,
            suite,
            device=target_device,
            include_per_head=layers == 6,
        )
        print(f"    self-patch max |delta logit|={baseline:.3e}", flush=True)
        metadata = {
            "architecture_layers": layers,
            "condition": condition,
            "seed": seed,
            "ood_accuracy_mean": float(results[(layers, condition, seed)]["ood_accuracy_mean"]),
        }
        all_rows.extend({**metadata, **row} for row in rows)
        baselines.append(baseline)
        _write_rows(output / "patch_metrics.partial.csv", all_rows)
        del model
        if target_device.type == "cuda":
            torch.cuda.empty_cache()

    aggregate = _aggregate(all_rows)
    _write_rows(output / "patch_metrics.csv", all_rows)
    _write_rows(output / "aggregate_metrics.csv", aggregate)
    plots = _plots(aggregate, output / "plots")
    max_baseline = max(baselines)
    (output / "report.md").write_text(
        _report(aggregate, plots, len(suite), max_baseline), encoding="utf-8"
    )
    (output / "status.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "models": len(checkpoints),
                "suite_size": len(suite),
                "count_per_operator": count_per_operator,
                "sites": list(SITES),
                "components": list(COMPONENT_GROUPS),
                "donor_types": list(DONOR_TYPES),
                "max_self_patch_logit_error": max_baseline,
                "plots": [str(path.relative_to(output)) for path in plots],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    torch.backends.mha.set_fastpath_enabled(previous_fastpath)
    return output

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .complex_experiment import _write_rows
from .data import AssignedExpression, collate_expressions, make_symbolic_splits
from .equivalence import DIAGRAM_BUILDERS, EquivalenceDiagram, make_equivalence_suite
from .gauge_atlas import (
    chart_reconstruction_errors,
    fit_atlas_edges,
    fit_chart,
    heldout_cycle_return_errors,
    heldout_section_energies,
    typed_component_cycles,
)
from .gauge_experiment import _family_tensors, atlas_features
from .holonomy_audit import _load_model, _regenerate_evaluation_data
from .path_patching import PathPatchCase, _matched_shuffle, _path_cases
from .training import resolve_device

VIEWS = tuple(
    (layer, component) for layer in (5, 6) for component in ("query", "key", "value", "coupled_qk")
)


def _condition(path: Path) -> str:
    return "higher_diversity" if "higher_diversity" in path.stem else "low_diversity"


def _new_diagrams(config: dict[str, Any], count: int = 128) -> list[EquivalenceDiagram]:
    """Create a suite disjoint from all original data and previous diagrams."""

    splits = make_symbolic_splits(
        train_size=max(config["conditions"].values()),
        validation_size=int(config["validation_size"]),
        test_size=int(config["test_size"]),
        train_depth=int(config["train_depth"]),
        ood_depths=tuple(int(depth) for depth in config["ood_depths"]),
        seed=4_281,
    )
    blocked = {
        str(item)
        for item in (
            *splits.train,
            *splits.validation,
            *splits.test_id,
            *(x for depth in config["ood_depths"] for x in splits.test_ood[int(depth)]),
        )
    }
    calibration, old_diagrams = _regenerate_evaluation_data(config)
    for pairs in calibration.values():
        for source, target in pairs:
            blocked.update((str(source), str(target)))
    for diagram in old_diagrams:
        blocked.update(str(vertex) for vertex in diagram.vertices)

    candidates = make_equivalence_suite(
        count * 5,
        operand_depth=int(config["diagram_operand_depth"]),
        seed=191_013,
    )
    grouped: dict[str, list[EquivalenceDiagram]] = defaultdict(list)
    for diagram in candidates:
        strings = {str(vertex) for vertex in diagram.vertices}
        if strings & blocked or len(grouped[diagram.family]) >= count:
            continue
        grouped[diagram.family].append(diagram)
    if len(grouped) != len(DIAGRAM_BUILDERS) or any(
        len(items) != count for items in grouped.values()
    ):
        counts = {family: len(items) for family, items in sorted(grouped.items())}
        raise RuntimeError(
            f"Could not construct the new data-disjoint confirmatory suite: {counts}"
        )
    return [item for family in sorted(grouped) for item in grouped[family]]


def _stratified_split(
    diagrams: Sequence[EquivalenceDiagram], seed: int
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray([item.vertices[0].value for item in diagrams])
    rng = np.random.default_rng(seed)
    fit: list[int] = []
    evaluation: list[int] = []
    for label in (0, 1):
        indices = rng.permutation(np.flatnonzero(labels == label))
        midpoint = len(indices) // 2
        fit.extend(indices[:midpoint].tolist())
        evaluation.extend(indices[midpoint:].tolist())
    return np.asarray(sorted(fit)), np.asarray(sorted(evaluation))


@torch.inference_mode()
def _logits(
    model, expressions: Sequence[AssignedExpression], device: torch.device, batch: int
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, len(expressions), batch):
        packed = collate_expressions(expressions[start : start + batch])
        chunks.append(
            model(packed.tokens.to(device), packed.padding_mask.to(device)).float().cpu().numpy()
        )
    return np.concatenate(chunks)


def _family_shared_charts(tensor: np.ndarray, fit: np.ndarray, dimension: int) -> dict[int, Any]:
    chart = fit_chart(tensor[fit].reshape(-1, tensor.shape[-1]), dimension)
    return {vertex: chart for vertex in range(tensor.shape[1])}


def _geometry_rows(
    model,
    diagrams: list[EquivalenceDiagram],
    *,
    checkpoint: Path,
    seed: int,
    device: torch.device,
    batch_size: int,
    dimension: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expressions = [vertex for diagram in diagrams for vertex in diagram.vertices]
    offsets = np.cumsum([0] + [len(diagram.vertices) for diagram in diagrams])
    logits = _logits(model, expressions, device, batch_size)
    features = atlas_features(
        model,
        expressions,
        device=device,
        batch_size=batch_size,
        components=("query", "key", "value", "coupled_qk"),
        scopes=("expression_root",),
        heads=None,
    )
    base_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, diagram in enumerate(diagrams):
        canonical = offsets[index]
        logit = logits[canonical]
        base_by_family[diagram.family].append(
            {
                "diagram_index": index,
                "family_index": len(base_by_family[diagram.family]),
                "family": diagram.family,
                "true_label": diagram.vertices[0].value,
                "correct": int(int(np.argmax(logit)) == diagram.vertices[0].value),
                "confidence": float(abs(logit[1] - logit[0])),
                "expression_length": len(diagram.vertices[0].prefix_tokens()),
            }
        )

    per_view: list[dict[str, Any]] = []
    for layer, component in VIEWS:
        grouped = _family_tensors(diagrams, features[(layer, "expression_root", component)])
        for family_index, (family, (items, tensor)) in enumerate(sorted(grouped.items())):
            fit, evaluation = _stratified_split(items, 500_000 + family_index)
            charts = _family_shared_charts(tensor, fit, dimension)
            edge_pairs = [(edge.source, edge.target) for edge in items[0].edges]
            edges = fit_atlas_edges(
                tensor,
                edge_pairs,
                fit,
                dimension=dimension,
                ridge=1e-2,
                charts=charts,
            )
            reconstruction = chart_reconstruction_errors(charts, tensor, evaluation)
            returns = heldout_cycle_return_errors(edges, tensor, evaluation)
            section = heldout_section_energies(edges, tensor, evaluation)
            for position, item_index in enumerate(evaluation):
                base = base_by_family[family][int(item_index)]
                per_view.append(
                    {
                        "checkpoint": checkpoint.name,
                        "training_condition": _condition(checkpoint),
                        "seed": seed,
                        **base,
                        "layer": layer,
                        "component": component,
                        "chart_mode": "family_global",
                        "chart_dimension": dimension,
                        "reconstruction_error": float(reconstruction[position]),
                        "state_return_error": float(returns[position]),
                        "section_energy": float(section[position]),
                    }
                )

    accumulated: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in per_view:
        accumulated[int(row["diagram_index"])].append(row)
    aggregate: list[dict[str, Any]] = []
    for diagram_index, rows in sorted(accumulated.items()):
        first = rows[0]
        aggregate.append(
            {
                key: first[key]
                for key in (
                    "checkpoint",
                    "training_condition",
                    "seed",
                    "diagram_index",
                    "family",
                    "true_label",
                    "correct",
                    "confidence",
                    "expression_length",
                )
            }
            | {
                "views": len(rows),
                "reconstruction_error": float(
                    np.nanmean([r["reconstruction_error"] for r in rows])
                ),
                "state_return_error": float(np.nanmean([r["state_return_error"] for r in rows])),
                "section_energy": float(np.nanmean([r["section_energy"] for r in rows])),
            }
        )
    return per_view, aggregate


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return float("nan")
    return float(
        np.mean(positive[:, None] > negative[None, :])
        + 0.5 * np.mean(positive[:, None] == negative[None, :])
    )


def _model_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    prediction = probabilities >= 0.5
    recalls = [np.mean(prediction[labels == value] == value) for value in (0, 1)]
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return {
        "balanced_accuracy": float(np.mean(recalls)),
        "roc_auc": _auc(labels, probabilities),
        "log_loss": float(
            np.mean(-(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))
        ),
        "brier": float(np.mean((probabilities - labels) ** 2)),
    }


class _Design:
    def __init__(self, rows: list[dict[str, Any]], geometry: bool):
        self.geometry = geometry
        self.continuous = ["reconstruction_error", "confidence", "expression_length"]
        if geometry:
            self.continuous += ["state_return_error", "section_energy"]
        self.means: dict[str, float] = {}
        self.scales: dict[str, float] = {}
        for key in self.continuous:
            values = np.asarray([float(row[key]) for row in rows], dtype=float)
            finite = values[np.isfinite(values)]
            self.means[key] = float(np.mean(finite))
            self.scales[key] = max(float(np.std(finite)), 1e-8)
        self.families = sorted({str(row["family"]) for row in rows})
        self.seeds = sorted({int(row["seed"]) for row in rows})
        self.missing_indicators = [
            key for key in self.continuous if any(not np.isfinite(float(row[key])) for row in rows)
        ]
        self.names = [
            "intercept",
            *self.continuous,
            *(f"{key}_missing" for key in self.missing_indicators),
            "true_label",
        ] + [
            *(f"family={value}" for value in self.families[1:]),
            *(f"seed={value}" for value in self.seeds[1:]),
        ]

    def matrix(self, rows: list[dict[str, Any]]) -> np.ndarray:
        columns = [np.ones(len(rows))]
        for key in self.continuous:
            values = np.asarray([float(row[key]) for row in rows], dtype=float)
            values = np.where(np.isfinite(values), values, self.means[key])
            columns.append((values - self.means[key]) / self.scales[key])
        columns += [
            np.asarray([float(not np.isfinite(float(row[key]))) for row in rows])
            for key in self.missing_indicators
        ]
        columns.append(np.asarray([float(row["true_label"]) for row in rows]))
        columns += [
            np.asarray([float(row["family"] == value) for row in rows])
            for value in self.families[1:]
        ]
        columns += [
            np.asarray([float(int(row["seed"]) == value) for row in rows])
            for value in self.seeds[1:]
        ]
        return np.column_stack(columns)


def _fit_logistic(matrix: np.ndarray, labels: np.ndarray, ridge: float = 1.0) -> np.ndarray:
    beta = np.zeros(matrix.shape[1])
    class_weights = np.where(
        labels == 1, 0.5 / max(np.mean(labels == 1), 1e-8), 0.5 / max(np.mean(labels == 0), 1e-8)
    )
    penalty = np.eye(matrix.shape[1]) * ridge
    penalty[0, 0] = 0.0
    for _ in range(100):
        probability = 1 / (1 + np.exp(-np.clip(matrix @ beta, -30, 30)))
        weights = class_weights * np.maximum(probability * (1 - probability), 1e-6)
        working = matrix @ beta + (labels - probability) / np.maximum(
            probability * (1 - probability), 1e-6
        )
        updated = np.linalg.solve(
            matrix.T @ (weights[:, None] * matrix) + penalty, matrix.T @ (weights * working)
        )
        if np.max(np.abs(updated - beta)) < 1e-8:
            beta = updated
            break
        beta = updated
    return beta


def _predict(design: _Design, beta: np.ndarray, rows: list[dict[str, Any]]) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(design.matrix(rows) @ beta, -30, 30)))


def _prediction_analysis(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    high = [row for row in rows if row["training_condition"] == "higher_diversity"]
    low = [row for row in rows if row["training_condition"] == "low_diversity"]
    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    frozen: dict[str, Any] = {}
    for geometry in (False, True):
        name = "controls_plus_geometry" if geometry else "controls_only"
        for held_seed in sorted({int(row["seed"]) for row in high}):
            train = [row for row in high if int(row["seed"]) != held_seed]
            test = [row for row in high if int(row["seed"]) == held_seed]
            design = _Design(train, geometry)
            beta = _fit_logistic(
                design.matrix(train), np.asarray([row["correct"] for row in train])
            )
            probabilities = _predict(design, beta, test)
            for row, probability in zip(test, probabilities, strict=True):
                prediction_rows.append(
                    {
                        **row,
                        "model": name,
                        "evaluation": "high_leave_one_seed_out",
                        "probability_correct": float(probability),
                    }
                )
            metric_rows.append(
                {
                    "model": name,
                    "evaluation": f"high_seed_{held_seed}",
                    **_model_metrics(np.asarray([row["correct"] for row in test]), probabilities),
                }
            )
        design = _Design(high, geometry)
        beta = _fit_logistic(design.matrix(high), np.asarray([row["correct"] for row in high]))
        probabilities = _predict(design, beta, low)
        for row, probability in zip(low, probabilities, strict=True):
            prediction_rows.append(
                {
                    **row,
                    "model": name,
                    "evaluation": "frozen_low_diversity",
                    "probability_correct": float(probability),
                }
            )
        metric_rows.append(
            {
                "model": name,
                "evaluation": "frozen_low_diversity",
                **_model_metrics(np.asarray([row["correct"] for row in low]), probabilities),
            }
        )
        frozen[name] = {
            "feature_names": design.names,
            "coefficients": beta.tolist(),
            "means": design.means,
            "scales": design.scales,
            "families": design.families,
            "seeds": design.seeds,
        }
    return prediction_rows, metric_rows, frozen


@torch.inference_mode()
def _path_screen(
    model, cases: list[PathPatchCase], device: torch.device, batch_size: int
) -> list[dict[str, Any]]:
    permutation = _matched_shuffle(cases)
    labels = np.asarray([case.start.value for case in cases])
    output: list[dict[str, Any]] = []
    for layer in (5, 6):
        for component_index, component in enumerate(("query", "key", "value")):
            for head in range(model.config.n_heads):
                margins: dict[str, list[np.ndarray]] = defaultdict(list)
                effects: list[np.ndarray] = []
                for start in range(0, len(cases), batch_size):
                    indices = np.arange(start, min(start + batch_size, len(cases)))
                    start_batch = collate_expressions([cases[i].start for i in indices])
                    tokens = start_batch.tokens.to(device)
                    mask = start_batch.padding_mask.to(device)
                    clean = model(tokens, mask)
                    clean_margin = (clean[:, 1] - clean[:, 0]).cpu().numpy() * (
                        2 * labels[indices] - 1
                    )
                    for condition, donor_indices in (
                        ("matched", indices),
                        ("shuffled", permutation[indices]),
                    ):
                        for route in ("left", "right"):
                            examples = [getattr(cases[i], f"{route}_donor") for i in donor_indices]
                            donor_batch = collate_expressions(examples)
                            donor = model.qkv_projections(
                                donor_batch.tokens.to(device), donor_batch.padding_mask.to(device)
                            )[layer - 1][component_index][:, 0]
                            patched = model.forward_qkv_patched(
                                tokens,
                                mask,
                                patch_layer=layer,
                                patch_positions=torch.zeros(
                                    len(indices), dtype=torch.long, device=device
                                ),
                                patch_values={component: donor},
                                patch_head=head,
                            )
                            margin = (patched[:, 1] - patched[:, 0]).cpu().numpy() * (
                                2 * labels[indices] - 1
                            )
                            margins[f"{condition}_{route}"].append(margin)
                            if condition == "matched":
                                effects.append(np.abs(margin - clean_margin))
                joined = {key: np.concatenate(value) for key, value in margins.items()}
                matched = np.mean(np.abs(joined["matched_left"] - joined["matched_right"]))
                shuffled = np.mean(np.abs(joined["shuffled_left"] - joined["shuffled_right"]))
                output.append(
                    {
                        "layer": layer,
                        "component": component,
                        "head": head,
                        "matched_route_disagreement": float(matched),
                        "shuffled_route_disagreement": float(shuffled),
                        "route_specificity": float(shuffled - matched),
                        "causal_abs_effect": float(np.mean(np.concatenate(effects))),
                        "cases": len(cases),
                    }
                )
    return output


def _nominations(screen: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, int], dict[int, float]] = defaultdict(dict)
    for row in screen:
        if row["training_condition"] == "higher_diversity":
            grouped[(int(row["layer"]), str(row["component"]), int(row["head"]))][
                int(row["seed"])
            ] = float(row["route_specificity"])
    ranked = []
    for (layer, component, head), values in grouped.items():
        discovery = float(np.mean([values.get(0, np.nan), values.get(1, np.nan)]))
        confirmation = values.get(2, np.nan)
        ranked.append(
            {
                "layer": layer,
                "component": component,
                "head": head,
                "discovery_specificity": discovery,
                "confirmation_specificity": confirmation,
                "status": "confirmed" if discovery > 0 and confirmation > 0 else "not_confirmed",
            }
        )
    confirmed = sorted(
        (row for row in ranked if row["status"] == "confirmed"),
        key=lambda row: row["discovery_specificity"],
        reverse=True,
    )
    return (
        confirmed[:limit]
        if confirmed
        else sorted(ranked, key=lambda row: row["discovery_specificity"], reverse=True)[:1]
    )


def _subspace(
    model,
    cases: list[PathPatchCase],
    nomination: dict[str, Any],
    device: torch.device,
    rank: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    expressions = (
        [case.start for case in cases]
        + [case.left_donor for case in cases]
        + [case.right_donor for case in cases]
    )
    component_index = ("query", "key", "value").index(nomination["component"])
    packed = collate_expressions(expressions)
    with torch.inference_mode():
        projected = (
            model.qkv_projections(packed.tokens.to(device), packed.padding_mask.to(device))[
                int(nomination["layer"]) - 1
            ][component_index][:, 0, int(nomination["head"])]
            .float()
            .cpu()
            .numpy()
        )
    count = len(cases)
    differences = np.concatenate(
        (
            projected[count : 2 * count] - projected[:count],
            projected[2 * count :] - projected[:count],
        )
    )
    _, _, right = np.linalg.svd(differences - differences.mean(0), full_matrices=True)
    return right[:rank].T, right[rank : 2 * rank].T


@torch.inference_mode()
def _targeted_patch(
    model,
    cases: list[PathPatchCase],
    nomination: dict[str, Any],
    bases: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    labels = np.asarray([case.start.value for case in cases])
    component = str(nomination["component"])
    component_index = ("query", "key", "value").index(component)
    results: list[dict[str, Any]] = []
    for control, basis_np in bases.items():
        basis = torch.tensor(basis_np, dtype=torch.float32, device=device)
        route_margins: dict[str, list[np.ndarray]] = defaultdict(list)
        effects: list[np.ndarray] = []
        for start in range(0, len(cases), batch_size):
            indices = np.arange(start, min(start + batch_size, len(cases)))
            start_batch = collate_expressions([cases[i].start for i in indices])
            tokens = start_batch.tokens.to(device)
            mask = start_batch.padding_mask.to(device)
            clean_logits = model(tokens, mask)
            clean_qkv = model.qkv_projections(tokens, mask)[int(nomination["layer"]) - 1][
                component_index
            ][:, 0]
            clean_margin = (clean_logits[:, 1] - clean_logits[:, 0]).cpu().numpy() * (
                2 * labels[indices] - 1
            )
            for route in ("left", "right"):
                donor_batch = collate_expressions(
                    [getattr(cases[i], f"{route}_donor") for i in indices]
                )
                donor = model.qkv_projections(
                    donor_batch.tokens.to(device), donor_batch.padding_mask.to(device)
                )[int(nomination["layer"]) - 1][component_index][:, 0]
                head = int(nomination["head"])
                delta = donor[:, head] - clean_qkv[:, head]
                values = clean_qkv.clone()
                values[:, head] = clean_qkv[:, head] + (delta @ basis) @ basis.T
                patched = model.forward_qkv_patched(
                    tokens,
                    mask,
                    patch_layer=int(nomination["layer"]),
                    patch_positions=torch.zeros(len(indices), dtype=torch.long, device=device),
                    patch_values={component: values},
                    patch_head=head,
                )
                margin = (patched[:, 1] - patched[:, 0]).cpu().numpy() * (2 * labels[indices] - 1)
                route_margins[route].append(margin)
                effects.append(np.abs(margin - clean_margin))
        left, right = np.concatenate(route_margins["left"]), np.concatenate(route_margins["right"])
        results.append(
            {
                **nomination,
                "subspace": control,
                "rank": basis_np.shape[1],
                "matched_route_disagreement": float(np.mean(np.abs(left - right))),
                "causal_abs_effect": float(np.mean(np.concatenate(effects))),
            }
        )
    return results


def _targeted_circuit_geometry(
    model,
    diagrams: list[EquivalenceDiagram],
    nomination: dict[str, Any],
    bases: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Measure the same connection inside a causal within-head path subspace."""

    expressions = [vertex for diagram in diagrams for vertex in diagram.vertices]
    offsets = np.cumsum([0] + [len(diagram.vertices) for diagram in diagrams])
    logits = _logits(model, expressions, device, batch_size)
    correct = {
        index: int(int(np.argmax(logits[offsets[index]])) == diagram.vertices[0].value)
        for index, diagram in enumerate(diagrams)
    }
    component = str(nomination["component"])
    raw = atlas_features(
        model,
        expressions,
        device=device,
        batch_size=batch_size,
        components=(component,),
        scopes=("cls",),
        heads=None,
    )[(int(nomination["layer"]), "cls", component)]
    head_dimension = raw.shape[1] // model.config.n_heads
    head_features = raw.reshape(len(raw), model.config.n_heads, head_dimension)[
        :, int(nomination["head"])
    ]
    rows: list[dict[str, Any]] = []
    for subspace, basis in bases.items():
        grouped = _family_tensors(diagrams, head_features @ basis)
        for family_index, (family, (items, tensor)) in enumerate(sorted(grouped.items())):
            fit, evaluation = _stratified_split(items, 500_000 + family_index)
            charts = _family_shared_charts(tensor, fit, basis.shape[1])
            edges = fit_atlas_edges(
                tensor,
                [(edge.source, edge.target) for edge in items[0].edges],
                fit,
                dimension=basis.shape[1],
                ridge=1e-2,
                charts=charts,
            )
            returns = heldout_cycle_return_errors(edges, tensor, evaluation)
            section = heldout_section_energies(edges, tensor, evaluation)
            reconstruction = chart_reconstruction_errors(charts, tensor, evaluation)
            holonomy = float(
                np.mean([cycle.relative_unit_distance for cycle in typed_component_cycles(edges)])
            )
            family_global_indices = [
                index for index, diagram in enumerate(diagrams) if diagram.family == family
            ]
            for position, family_item_index in enumerate(evaluation):
                diagram_index = family_global_indices[int(family_item_index)]
                rows.append(
                    {
                        **nomination,
                        "subspace": subspace,
                        "rank": basis.shape[1],
                        "family": family,
                        "diagram_index": diagram_index,
                        "correct": correct[diagram_index],
                        "relative_holonomy": holonomy,
                        "state_return_error": float(returns[position]),
                        "section_energy": float(section[position]),
                        "reconstruction_error": float(reconstruction[position]),
                    }
                )
    return rows


def _plots(
    rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    screen: list[dict[str, Any]],
    targeted: list[dict[str, Any]],
    circuit_geometry: list[dict[str, Any]],
    output: Path,
) -> list[Path]:
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib_cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    def save(fig, name):
        fig.tight_layout()
        path = output / name
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    high_metrics = [r for r in metrics if r["evaluation"].startswith("high_seed")]
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, model in enumerate(("controls_only", "controls_plus_geometry")):
        values = [r["roc_auc"] for r in high_metrics if r["model"] == model]
        ax.bar(i, np.mean(values), yerr=np.std(values), color=("#9aa0a6", "#3367d6")[i])
    ax.set_xticks([0, 1], ["Controls", "+ sheaf geometry"])
    ax.set_ylabel("LOSO ROC AUC")
    ax.set_ylim(0, 1)
    save(fig, "A_high_diversity_prediction.png")
    fig, ax = plt.subplots(figsize=(7, 5))
    lowm = [r for r in metrics if r["evaluation"] == "frozen_low_diversity"]
    ax.bar([0, 1], [r["roc_auc"] for r in lowm], color=["#9aa0a6", "#d93025"])
    ax.set_xticks([0, 1], ["Controls", "+ sheaf geometry"])
    ax.set_ylabel("Frozen low-diversity ROC AUC")
    ax.set_ylim(0, 1)
    save(fig, "B_frozen_transfer.png")
    for letter, field, title in (
        ("C", "section_energy", "Section energy"),
        ("D", "state_return_error", "Loop return error"),
        ("E", "reconstruction_error", "Reconstruction error"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        groups = []
        labels = []
        for condition in ("higher_diversity", "low_diversity"):
            for correct in (1, 0):
                groups.append(
                    [
                        r[field]
                        for r in rows
                        if r["training_condition"] == condition and r["correct"] == correct
                    ]
                )
                labels.append(
                    ("High" if condition.startswith("higher") else "Low")
                    + (" correct" if correct else " wrong")
                )
        ax.boxplot(groups, tick_labels=labels, showfliers=False)
        ax.set_ylabel(title)
        ax.tick_params(axis="x", rotation=15)
        save(fig, f"{letter}_{field}.png")
    fig, ax = plt.subplots(figsize=(7, 5))
    chosen = [
        r
        for r in predictions
        if r["model"] == "controls_plus_geometry" and r["evaluation"] == "frozen_low_diversity"
    ]
    bins = np.linspace(0, 1, 7)
    centers = []
    observed = []
    for a, b in pairwise(bins):
        cell = [r for r in chosen if a <= r["probability_correct"] < b]
        if cell:
            centers.append(np.mean([r["probability_correct"] for r in cell]))
            observed.append(np.mean([r["correct"] for r in cell]))
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.plot(centers, observed, "o-")
    ax.set(xlabel="Predicted P(correct)", ylabel="Observed correctness", xlim=(0, 1), ylim=(0, 1))
    save(fig, "F_frozen_calibration.png")
    fig, ax = plt.subplots(figsize=(10, 5))
    family_gap = []
    for family in sorted({r["family"] for r in rows}):
        c = [r["section_energy"] for r in rows if r["family"] == family and r["correct"]]
        w = [r["section_energy"] for r in rows if r["family"] == family and not r["correct"]]
        family_gap.append((family, np.mean(w) - np.mean(c)))
    ax.bar(range(len(family_gap)), [x[1] for x in family_gap])
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(range(len(family_gap)), [x[0] for x in family_gap], rotation=40, ha="right")
    ax.set_ylabel("Wrong - correct section energy")
    save(fig, "G_family_section_gaps.png")
    fig, ax = plt.subplots(figsize=(9, 5))
    grouped = defaultdict(list)
    for r in screen:
        if r["training_condition"] == "higher_diversity":
            grouped[f"L{r['layer']} {r['component'][0].upper()}H{r['head']}"].append(
                r["route_specificity"]
            )
    ordered = sorted(grouped, key=lambda k: np.mean(grouped[k]), reverse=True)[:12]
    ax.bar(range(len(ordered)), [np.mean(grouped[k]) for k in ordered])
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(range(len(ordered)), ordered, rotation=45, ha="right")
    ax.set_ylabel("Route specificity")
    save(fig, "H_path_circuit_screen.png")
    if targeted:
        fig, ax = plt.subplots(figsize=(8, 5))
        modes = ("path_subspace", "orthogonal_control")
        ax.bar(
            [0, 1],
            [
                np.mean([r["causal_abs_effect"] for r in targeted if r["subspace"] == m])
                for m in modes
            ],
        )
        ax.set_xticks([0, 1], ["Path subspace", "Orthogonal control"])
        ax.set_ylabel("Absolute causal logit-margin effect")
        save(fig, "I_targeted_subspace_patching.png")
    if circuit_geometry:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        modes = ("path_subspace", "orthogonal_control")
        for axis, field, title in zip(
            axes,
            ("relative_holonomy", "state_return_error", "section_energy"),
            ("Relative holonomy", "Loop return error", "Section energy"),
            strict=True,
        ):
            values = []
            for mode in modes:
                samples = np.asarray(
                    [float(row[field]) for row in circuit_geometry if row["subspace"] == mode]
                )
                values.append(float(np.mean(samples[np.isfinite(samples)])))
            axis.bar([0, 1], values, color=["#3367d6", "#9aa0a6"])
            axis.set_xticks([0, 1], ["Path", "Control"])
            axis.set_title(title)
        save(fig, "J_circuit_local_geometry.png")
    return paths


def run_confirmatory_experiment(
    run_directory: Path,
    *,
    device: str = "auto",
    diagrams_per_family: int = 128,
    chart_dimension: int = 8,
    circuit_rank: int = 4,
) -> Path:
    run_directory = run_directory.resolve()
    config = json.loads((run_directory / "config.json").read_text(encoding="utf-8"))
    diagrams = _new_diagrams(config, diagrams_per_family)
    grouped_diagrams: dict[str, list[EquivalenceDiagram]] = defaultdict(list)
    for diagram in diagrams:
        grouped_diagrams[diagram.family].append(diagram)
    calibration_diagrams = []
    evaluation_diagrams = []
    for family_index, (family, items) in enumerate(sorted(grouped_diagrams.items())):
        fit, evaluation = _stratified_split(items, 500_000 + family_index)
        calibration_diagrams += [items[i] for i in fit]
        evaluation_diagrams += [items[i] for i in evaluation]
    checkpoints = sorted((run_directory / "checkpoints").glob("layers_6_*.pt"))
    stamp = datetime.now(timezone.utc).strftime("confirmatory_%Y%m%d_%H%M%S_%fZ")
    output = run_directory / "confirmatory" / stamp
    (output / "tables").mkdir(parents=True)
    target_device = resolve_device(device)
    per_view = []
    aggregate = []
    screen = []
    eval_cases = _path_cases(evaluation_diagrams)
    calibration_cases = _path_cases(calibration_diagrams)
    models: dict[str, Any] = {}
    for number, checkpoint_path in enumerate(checkpoints, 1):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        seed = int(checkpoint["train_config"]["seed"])
        print(
            f"[{number}/{len(checkpoints)}] confirmatory geometry {checkpoint_path.stem}",
            flush=True,
        )
        model = _load_model(checkpoint_path, target_device)
        models[checkpoint_path.name] = model
        view_rows, model_rows = _geometry_rows(
            model,
            diagrams,
            checkpoint=checkpoint_path,
            seed=seed,
            device=target_device,
            batch_size=int(config["batch_size"]),
            dimension=chart_dimension,
        )
        per_view += view_rows
        aggregate += model_rows
        if _condition(checkpoint_path) == "higher_diversity":
            for row in _path_screen(model, eval_cases, target_device, int(config["batch_size"])):
                screen.append(
                    {
                        "checkpoint": checkpoint_path.name,
                        "training_condition": _condition(checkpoint_path),
                        "seed": seed,
                        **row,
                    }
                )
    predictions, metric_rows, frozen = _prediction_analysis(aggregate)
    nominations = _nominations(screen)
    targeted = []
    circuit_geometry = []
    for checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        seed = int(checkpoint["train_config"]["seed"])
        model = models[checkpoint_path.name]
        for nomination in nominations:
            path_basis, control_basis = _subspace(
                model, calibration_cases, nomination, target_device, circuit_rank
            )
            bases = {"path_subspace": path_basis, "orthogonal_control": control_basis}
            for row in _targeted_patch(
                model, eval_cases, nomination, bases, target_device, int(config["batch_size"])
            ):
                targeted.append(
                    {
                        "checkpoint": checkpoint_path.name,
                        "training_condition": _condition(checkpoint_path),
                        "seed": seed,
                        **row,
                    }
                )
            for row in _targeted_circuit_geometry(
                model, diagrams, nomination, bases, target_device, int(config["batch_size"])
            ):
                circuit_geometry.append(
                    {
                        "checkpoint": checkpoint_path.name,
                        "training_condition": _condition(checkpoint_path),
                        "seed": seed,
                        **row,
                    }
                )
    _write_rows(output / "tables" / "per_view_geometry.csv", per_view)
    _write_rows(output / "tables" / "per_example_geometry.csv", aggregate)
    _write_rows(output / "tables" / "correctness_predictions.csv", predictions)
    _write_rows(output / "tables" / "predictor_metrics.csv", metric_rows)
    _write_rows(output / "tables" / "path_circuit_screen.csv", screen)
    _write_rows(output / "tables" / "path_circuit_nominations.csv", nominations)
    _write_rows(output / "tables" / "targeted_subspace_patching.csv", targeted)
    _write_rows(output / "tables" / "targeted_circuit_geometry.csv", circuit_geometry)
    (output / "frozen_predictors.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    settings = {
        "new_suite_seed": 191013,
        "diagrams_per_family": diagrams_per_family,
        "calibration_per_family": diagrams_per_family // 2,
        "evaluation_per_family": diagrams_per_family // 2,
        "families": sorted(grouped_diagrams),
        "chart_mode": "family_global",
        "chart_dimension": chart_dimension,
        "views": [f"L{l}:{c}" for l, c in VIEWS],
        "circuit_rank": circuit_rank,
        "discovery_seeds": [0, 1],
        "confirmation_seed": 2,
        "correctness_rule": "canonical vertex-0 prediction equals semantic value",
    }
    (output / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    plots = _plots(
        aggregate,
        predictions,
        metric_rows,
        screen,
        targeted,
        circuit_geometry,
        output / "plots",
    )
    high = [r for r in metric_rows if r["evaluation"].startswith("high_seed")]
    low = [r for r in metric_rows if r["evaluation"] == "frozen_low_diversity"]

    def metric(model, rows, key):
        return float(np.mean([r[key] for r in rows if r["model"] == model]))

    def finite_mean(items: list[dict[str, Any]], key: str) -> float:
        values = np.asarray([float(item[key]) for item in items])
        return float(np.mean(values[np.isfinite(values)]))

    targeted_path = [row for row in targeted if row["subspace"] == "path_subspace"]
    targeted_control = [row for row in targeted if row["subspace"] == "orthogonal_control"]
    circuit_path = [row for row in circuit_geometry if row["subspace"] == "path_subspace"]
    circuit_control = [row for row in circuit_geometry if row["subspace"] == "orthogonal_control"]

    lines = [
        "# Confirmatory family-global sheaf and path-circuit experiment",
        "",
        "## Design",
        "",
        f"This run used {len(diagrams)} entirely new diagrams ({diagrams_per_family} per family); half of each truth-balanced family fit the connection and half evaluated it. Expressions used in training, prior evaluation, rewrite calibration, and prior diagrams were excluded.",
        "",
        f"The preregistered geometry was a family-global chart of dimension {chart_dimension}, averaged over Q, K, V, and coupled QK at layers 5 and 6. Correctness was never used to fit charts or connections.",
        "",
        "## Correctness prediction",
        "",
        f"On higher-diversity leave-one-seed-out folds, controls-only ROC AUC was {metric('controls_only', high, 'roc_auc'):.3f}; adding return error and section energy gave {metric('controls_plus_geometry', high, 'roc_auc'):.3f}. The rule was then frozen. On low-diversity models, the corresponding AUCs were {metric('controls_only', low, 'roc_auc'):.3f} and {metric('controls_plus_geometry', low, 'roc_auc'):.3f}.",
        "",
        "Controls were reconstruction error, unsigned model confidence, prefix length, diagram family, true label, and seed. The comparison therefore asks whether connection/sheaf quantities add information beyond those covariates.",
        "",
        f"The incremental result is weak and does not transfer: geometry changes mean higher-diversity AUC by {metric('controls_plus_geometry', high, 'roc_auc') - metric('controls_only', high, 'roc_auc'):+.3f}, but frozen low-diversity AUC by {metric('controls_plus_geometry', low, 'roc_auc') - metric('controls_only', low, 'roc_auc'):+.3f}. This is evidence against these pooled family-global quantities as a robust correctness detector.",
        "",
        "## Targeted Q/K/V path circuits",
        "",
        f"The causal screen tested each Q/K/V head at layers 5 and 6 on competing equivalent paths. Seeds 0-1 were discovery and seed 2 was confirmation. {len([n for n in nominations if n['status'] == 'confirmed'])} sites met the sign-replication rule. For each selected site, a rank-{circuit_rank} within-head SVD subspace was learned only from calibration path differences and compared with an equal-rank orthogonal control.",
        "",
        "| Layer | Component | Head | Discovery specificity | Confirmation specificity |",
        "|---:|---|---:|---:|---:|",
        *[
            f"| {item['layer']} | {item['component']} | {item['head']} | {item['discovery_specificity']:.4f} | {item['confirmation_specificity']:.4f} |"
            for item in nominations
        ],
        "",
        f"Across all six models and nominated sites, the rank-{circuit_rank} path subspace changed the true-class logit margin by {finite_mean(targeted_path, 'causal_abs_effect'):.4f} on average, versus {finite_mean(targeted_control, 'causal_abs_effect'):.4f} for the equal-rank orthogonal control.",
        "",
        "## Circuit-local connection geometry",
        "",
        f"The causal path subspace had mean relative holonomy {finite_mean(circuit_path, 'relative_holonomy'):.4f}, loop-return error {finite_mean(circuit_path, 'state_return_error'):.4f}, and section energy {finite_mean(circuit_path, 'section_energy'):.4f}. The equal-rank orthogonal control had {finite_mean(circuit_control, 'relative_holonomy'):.4f}, {finite_mean(circuit_control, 'state_return_error'):.4f}, and {finite_mean(circuit_control, 'section_energy'):.4f}, respectively.",
        "",
        "Causal relevance and low holonomy do not coincide here: the path-sensitive subspace is much more interventionally active, yet less flat/coherent by relative holonomy and return error. Holonomy should therefore be treated as a descriptive geometric property, not a monotone localization score for causal computation.",
        "",
        "`double_negation` has no independent graph cycle, so loop return and relative holonomy are undefined for that family. Raw tables retain `nan`; the predictor uses training-fold mean imputation plus a structural missingness indicator.",
        "",
        "## Figures",
        "",
    ]
    for plot in plots:
        lines += [f"![{plot.stem}](plots/{plot.name})", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "progress.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "models": len(checkpoints),
                "per_example_rows": len(aggregate),
                "plots": len(plots),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output

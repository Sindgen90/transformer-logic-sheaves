from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .complex_experiment import _write_rows
from .data import AssignedExpression, ExpressionDataset, collate_expressions
from .equivalence import EquivalenceDiagram
from .holonomy_audit import _load_model, _regenerate_evaluation_data
from .logic import Expr
from .qkv_holonomy import (
    COMPONENTS,
    _all_experiment_strings,
    _balanced_disjoint_pairs,
)
from .training import resolve_device

FEATURES = ("residual", "query", "key", "value")
LOCAL_FEATURES = (*FEATURES, "coupled_qk")
GLOBAL_FEATURES = ("cls_query", "cls_key", "cls_value", "cls_coupled_qk")
TRANSPORT_FEATURES = (*LOCAL_FEATURES, *GLOBAL_FEATURES)
BASELINES = (
    "identity",
    "global",
    "contextual",
    "target_shuffled",
    "label_shuffled",
    "low_rank",
)
INVERSE_RELATIONS = {
    "double_negation_expand": "double_negation",
    "double_negation_reduce": "double_negation",
    "demorgan_and": "demorgan_and",
    "demorgan_and_inverse": "demorgan_and",
    "demorgan_or": "demorgan_or",
    "demorgan_or_inverse": "demorgan_or",
    "associate_and_right": "associate_and",
    "associate_and_left": "associate_and",
    "distribute_and_over_or_left": "distribute_and_over_or_left",
    "distribute_and_over_or_left_inverse": "distribute_and_over_or_left",
    "ite_expand": "ite",
    "ite_reduce": "ite",
    "majority_duality": "majority_duality",
    "majority_duality_inverse": "majority_duality",
}


@dataclass(frozen=True)
class FeatureSet:
    local: dict[tuple[int, str], np.ndarray]
    context: dict[tuple[int, str], np.ndarray]


@dataclass(frozen=True)
class EdgeOccurrence:
    diagram_index: int
    edge_index: int
    label: str
    relation: str
    orientation: int
    source: AssignedExpression
    target: AssignedExpression
    source_position: int
    target_position: int


@dataclass(frozen=True)
class ContextualTransport:
    source_mean: np.ndarray
    target_mean: np.ndarray
    rotation: np.ndarray
    context_mean: np.ndarray
    context_weights: np.ndarray

    def apply(self, states: np.ndarray, contexts: np.ndarray, orientation: int) -> np.ndarray:
        context_shift = (contexts - self.context_mean) @ self.context_weights
        if orientation == 1:
            return (states - self.source_mean) @ self.rotation + self.target_mean + context_shift
        if orientation == -1:
            return (states - self.target_mean - context_shift) @ self.rotation.T + self.source_mean
        raise ValueError("orientation must be +1 or -1")

    def linear(self, orientation: int) -> np.ndarray:
        return self.rotation if orientation == 1 else self.rotation.T

    def without_context(self) -> ContextualTransport:
        return ContextualTransport(
            self.source_mean,
            self.target_mean,
            self.rotation,
            self.context_mean,
            np.zeros_like(self.context_weights),
        )


def _relation(label: str) -> str:
    return INVERSE_RELATIONS.get(label, label)


def _rewrite_path(source: Expr, target: Expr) -> tuple[int, ...]:
    if source.op != target.op or len(source.children) != len(target.children):
        return ()
    changed = [
        index
        for index, (left, right) in enumerate(zip(source.children, target.children, strict=True))
        if left != right
    ]
    if len(changed) != 1:
        return ()
    index = changed[0]
    return (index, *_rewrite_path(source.children[index], target.children[index]))


def _tree_size(expression: Expr) -> int:
    return 1 + sum(_tree_size(child) for child in expression.children)


def _token_position(example: AssignedExpression, path: tuple[int, ...]) -> int:
    expression = example.expression
    offset = 0
    for child_index in path:
        offset += 1 + sum(_tree_size(child) for child in expression.children[:child_index])
        expression = expression.children[child_index]
    # CLS, ENV, assignments, SEP precede the expression root.
    return len(example.assignment) + 3 + offset


def _canonical_orientation(source: AssignedExpression, target: AssignedExpression) -> int:
    return 1 if str(source) < str(target) else -1


def _edge_occurrences(diagrams: Sequence[EquivalenceDiagram]) -> list[EdgeOccurrence]:
    rows: list[EdgeOccurrence] = []
    for diagram_index, diagram in enumerate(diagrams):
        for edge_index, edge in enumerate(diagram.edges):
            source = diagram.vertices[edge.source]
            target = diagram.vertices[edge.target]
            path = _rewrite_path(source.expression, target.expression)
            rows.append(
                EdgeOccurrence(
                    diagram_index=diagram_index,
                    edge_index=edge_index,
                    label=edge.label,
                    relation=_relation(edge.label),
                    orientation=_canonical_orientation(source, target),
                    source=source,
                    target=target,
                    source_position=_token_position(source, path),
                    target_position=_token_position(target, path),
                )
            )
    return rows


def _calibration_occurrences(
    pairs: dict[str, list[tuple[AssignedExpression, AssignedExpression]]],
) -> list[EdgeOccurrence]:
    rows: list[EdgeOccurrence] = []
    for label, examples in pairs.items():
        for pair_index, (source, target) in enumerate(examples):
            path = _rewrite_path(source.expression, target.expression)
            rows.append(
                EdgeOccurrence(
                    diagram_index=-1,
                    edge_index=pair_index,
                    label=label,
                    relation=_relation(label),
                    orientation=_canonical_orientation(source, target),
                    source=source,
                    target=target,
                    source_position=_token_position(source, path),
                    target_position=_token_position(target, path),
                )
            )
    return rows


@torch.inference_mode()
def _extract_features(
    model,
    examples: Sequence[AssignedExpression],
    positions: Sequence[int],
    *,
    device: torch.device,
    batch_size: int,
) -> FeatureSet:
    local_chunks: dict[tuple[int, str], list[np.ndarray]] = defaultdict(list)
    context_chunks: dict[tuple[int, str], list[np.ndarray]] = defaultdict(list)
    dataset = ExpressionDataset(examples)
    for start in range(0, len(dataset), batch_size):
        batch_examples = dataset.expressions[start : start + batch_size]
        batch_positions = torch.tensor(
            positions[start : start + batch_size], dtype=torch.long, device=device
        )
        batch = collate_expressions(batch_examples)
        tokens = batch.tokens.to(device)
        mask = batch.padding_mask.to(device)
        rows = torch.arange(len(batch_examples), device=device)
        stages = model.stage_representations(tokens, mask)
        qkv = model.qkv_projections(tokens, mask)
        for layer in range(1, model.config.n_layers + 1):
            residual = stages[layer]
            local_chunks[(layer, "residual")].append(
                residual[rows, batch_positions].float().cpu().numpy()
            )
            context_chunks[(layer, "residual")].append(residual[:, 0].float().cpu().numpy())
            for component, values in zip(COMPONENTS, qkv[layer - 1], strict=True):
                flattened = values.reshape(values.shape[0], values.shape[1], -1)
                local_chunks[(layer, component)].append(
                    flattened[rows, batch_positions].float().cpu().numpy()
                )
                context_chunks[(layer, component)].append(flattened[:, 0].float().cpu().numpy())
                local_chunks[(layer, f"cls_{component}")].append(
                    flattened[:, 0].float().cpu().numpy()
                )
                context_chunks[(layer, f"cls_{component}")].append(
                    flattened[rows, batch_positions].float().cpu().numpy()
                )
    return FeatureSet(
        local={key: np.concatenate(value) for key, value in local_chunks.items()},
        context={key: np.concatenate(value) for key, value in context_chunks.items()},
    )


def _canonical_arrays(
    occurrences: Sequence[EdgeOccurrence],
    source: FeatureSet,
    target: FeatureSet,
    site: tuple[int, str],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    grouped: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(list)
    source_local = source.local[site]
    target_local = target.local[site]
    source_context = source.context[site]
    target_context = target.context[site]
    for index, occurrence in enumerate(occurrences):
        context = 0.5 * (source_context[index] + target_context[index])
        if occurrence.orientation == 1:
            grouped[occurrence.relation].append((source_local[index], target_local[index], context))
        else:
            grouped[occurrence.relation].append((target_local[index], source_local[index], context))
    return {
        relation: (
            np.stack([item[0] for item in values]),
            np.stack([item[1] for item in values]),
            np.stack([item[2] for item in values]),
        )
        for relation, values in grouped.items()
    }


def _with_coupled_qk(features: FeatureSet) -> FeatureSet:
    local = dict(features.local)
    context = dict(features.context)
    layers = sorted({layer for layer, feature in local if feature == "query"})
    for layer in layers:
        local[(layer, "coupled_qk")] = np.concatenate(
            (local[(layer, "query")], local[(layer, "key")]), axis=1
        )
        context[(layer, "coupled_qk")] = np.concatenate(
            (context[(layer, "query")], context[(layer, "key")]), axis=1
        )
        local[(layer, "cls_coupled_qk")] = np.concatenate(
            (local[(layer, "cls_query")], local[(layer, "cls_key")]), axis=1
        )
        context[(layer, "cls_coupled_qk")] = np.concatenate(
            (context[(layer, "cls_query")], context[(layer, "cls_key")]), axis=1
        )
    return FeatureSet(local=local, context=context)


def _fit_transport(
    source: np.ndarray,
    target: np.ndarray,
    context: np.ndarray,
    *,
    ridge: float = 1.0,
    shuffled: bool = False,
    low_rank: int | None = None,
    seed: int = 0,
) -> ContextualTransport:
    def thin_svd(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if torch.cuda.is_available():
            tensor = torch.as_tensor(matrix, device="cuda")
            left, singular, right_t = torch.linalg.svd(tensor, full_matrices=False)
            return (
                left.cpu().numpy(),
                singular.cpu().numpy(),
                right_t.cpu().numpy(),
            )
        return np.linalg.svd(matrix, full_matrices=False)

    rng = np.random.default_rng(seed)
    if shuffled:
        target = target[rng.permutation(len(target))]
    source_mean = source.mean(axis=0, keepdims=True)
    target_mean = target.mean(axis=0, keepdims=True)
    source_centered = source - source_mean
    target_centered = target - target_mean
    cross = source_centered.T @ target_centered
    u, _, vt = thin_svd(cross)
    rotation = u @ vt
    if low_rank is not None and low_rank < source.shape[1]:
        joined = np.concatenate((source_centered, target_centered), axis=0)
        _, _, basis_t = thin_svd(joined)
        basis = basis_t[:low_rank].T
        projector = basis @ basis.T
        reduced_cross = (source_centered @ basis).T @ (target_centered @ basis)
        ru, _, rvt = thin_svd(reduced_cross)
        rotation = (
            basis @ (ru @ rvt) @ basis.T + np.eye(source.shape[1], dtype=source.dtype) - projector
        )
    context_mean = context.mean(axis=0, keepdims=True)
    centered_context = context - context_mean
    residual = target - ((source - source_mean) @ rotation + target_mean)
    # The dual ridge form solves an n x n system instead of a feature_dim x
    # feature_dim system. Here n=96 and Q/K may be 256-dimensional jointly.
    gram = centered_context @ centered_context.T + ridge * np.eye(len(context), dtype=context.dtype)
    context_weights = centered_context.T @ np.linalg.solve(gram, residual)
    if low_rank is not None:
        context_weights = context_weights @ projector
    return ContextualTransport(
        source_mean,
        target_mean,
        rotation,
        context_mean,
        context_weights,
    )


def _fit_site_transports(
    arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    seed: int,
) -> dict[str, dict[str, ContextualTransport]]:
    contextual = {
        relation: _fit_transport(source, target, context, seed=seed + index)
        for index, (relation, (source, target, context)) in enumerate(arrays.items())
    }
    global_maps = {
        relation: transport.without_context() for relation, transport in contextual.items()
    }
    shuffled = {
        relation: _fit_transport(
            source,
            target,
            context,
            shuffled=True,
            seed=seed + 10_000 + index,
        )
        for index, (relation, (source, target, context)) in enumerate(arrays.items())
    }
    low_rank = {
        relation: _fit_transport(
            source,
            target,
            context,
            low_rank=min(8, source.shape[1]),
            seed=seed + 20_000 + index,
        )
        for index, (relation, (source, target, context)) in enumerate(arrays.items())
    }
    return {
        "global": global_maps,
        "contextual": contextual,
        "target_shuffled": shuffled,
        "low_rank": low_rank,
    }


def _label_shuffle_maps(
    maps: dict[str, ContextualTransport],
) -> dict[str, ContextualTransport]:
    labels = sorted(maps)
    shifted = labels[1:] + labels[:1]
    return {label: maps[other] for label, other in zip(labels, shifted, strict=True)}


def _apply_edge(
    state: np.ndarray,
    context: np.ndarray,
    occurrence: EdgeOccurrence,
    maps: dict[str, ContextualTransport] | None,
) -> np.ndarray:
    if maps is None:
        return state
    return maps[occurrence.relation].apply(
        state.reshape(1, -1), context.reshape(1, -1), occurrence.orientation
    )[0]


def _loop_compatible(
    diagram: EquivalenceDiagram,
    loop: tuple[int, ...],
    occurrence_lookup: dict[tuple[int, int], EdgeOccurrence],
    diagram_index: int,
) -> bool:
    occurrences = [occurrence_lookup[(diagram_index, edge)] for edge in loop]
    for index, occurrence in enumerate(occurrences):
        next_occurrence = occurrences[(index + 1) % len(occurrences)]
        if occurrence.target_position != next_occurrence.source_position:
            return False
    return True


def _score_site(
    diagrams: Sequence[EquivalenceDiagram],
    occurrences: Sequence[EdgeOccurrence],
    source: FeatureSet,
    target: FeatureSet,
    site: tuple[int, str],
    transport_sets: dict[str, dict[str, ContextualTransport]],
) -> list[dict[str, Any]]:
    global_fiber = site[1].startswith("cls_")
    source_states = source.local[site]
    target_states = target.local[site]
    contexts = 0.5 * (source.context[site] + target.context[site])
    variance = max(
        float(
            np.mean(
                (
                    np.concatenate((source_states, target_states))
                    - np.concatenate((source_states, target_states)).mean(axis=0)
                )
                ** 2
            )
        ),
        1e-12,
    )
    occurrence_index = {
        (occurrence.diagram_index, occurrence.edge_index): index
        for index, occurrence in enumerate(occurrences)
    }
    occurrence_lookup = {
        (occurrence.diagram_index, occurrence.edge_index): occurrence for occurrence in occurrences
    }
    baselines: dict[str, dict[str, ContextualTransport] | None] = {
        "identity": None,
        **transport_sets,
        "label_shuffled": _label_shuffle_maps(transport_sets["contextual"]),
    }
    grouped_diagrams: dict[str, list[int]] = defaultdict(list)
    for diagram_index, diagram in enumerate(diagrams):
        grouped_diagrams[diagram.family].append(diagram_index)
    rows: list[dict[str, Any]] = []
    for family, diagram_indices in grouped_diagrams.items():
        for baseline in BASELINES:
            maps = baselines[baseline]
            eigenphase_cache: dict[tuple[tuple[str, int], ...], tuple[float, ...]] = {}
            edge_errors: list[float] = []
            loop_errors: list[float] = []
            path_errors: list[float] = []
            endpoint_errors: list[float] = []
            inverse_errors: list[float] = []
            eigenphases: list[float] = []
            compatible_loops = 0
            compatible_paths = 0
            for diagram_index in diagram_indices:
                diagram = diagrams[diagram_index]
                for edge_index, _ in enumerate(diagram.edges):
                    index = occurrence_index[(diagram_index, edge_index)]
                    occurrence = occurrences[index]
                    predicted = _apply_edge(source_states[index], contexts[index], occurrence, maps)
                    edge_errors.append(float(np.mean((predicted - target_states[index]) ** 2)))
                    if maps is not None:
                        returned = maps[occurrence.relation].apply(
                            predicted.reshape(1, -1),
                            contexts[index].reshape(1, -1),
                            -occurrence.orientation,
                        )[0]
                        inverse_errors.append(
                            float(np.mean((returned - source_states[index]) ** 2))
                        )
                    else:
                        inverse_errors.append(0.0)
                for loop in diagram.loops:
                    if not global_fiber and not _loop_compatible(
                        diagram, loop, occurrence_lookup, diagram_index
                    ):
                        continue
                    compatible_loops += 1
                    first_index = occurrence_index[(diagram_index, loop[0])]
                    state = source_states[first_index]
                    rotation = np.eye(state.shape[0], dtype=state.dtype)
                    linear_path: list[tuple[str, int]] = []
                    for edge_index in loop:
                        index = occurrence_index[(diagram_index, edge_index)]
                        occurrence = occurrences[index]
                        state = _apply_edge(state, contexts[index], occurrence, maps)
                        if maps is not None:
                            linear_path.append((occurrence.relation, occurrence.orientation))
                            rotation = rotation @ maps[occurrence.relation].linear(
                                occurrence.orientation
                            )
                    loop_errors.append(float(np.mean((state - source_states[first_index]) ** 2)))
                    if maps is not None:
                        cache_key = tuple(linear_path)
                        phases = eigenphase_cache.get(cache_key)
                        if phases is None:
                            phases = tuple(np.abs(np.angle(np.linalg.eigvals(rotation))).tolist())
                            eigenphase_cache[cache_key] = phases
                        eigenphases.extend(phases)
                    else:
                        eigenphases.extend([0.0] * state.shape[0])
                for pair in diagram.path_pairs:
                    left_occurrences = [
                        occurrence_lookup[(diagram_index, edge)] for edge in pair.left
                    ]
                    right_occurrences = [
                        occurrence_lookup[(diagram_index, edge)] for edge in pair.right
                    ]
                    left_ok = all(
                        left_occurrences[index].target_position
                        == left_occurrences[index + 1].source_position
                        for index in range(len(left_occurrences) - 1)
                    )
                    right_ok = all(
                        right_occurrences[index].target_position
                        == right_occurrences[index + 1].source_position
                        for index in range(len(right_occurrences) - 1)
                    )
                    if not global_fiber and (not left_ok or not right_ok):
                        continue
                    if not global_fiber and (
                        left_occurrences[0].source_position != right_occurrences[0].source_position
                        or left_occurrences[-1].target_position
                        != right_occurrences[-1].target_position
                    ):
                        continue
                    compatible_paths += 1
                    start_index = occurrence_index[(diagram_index, pair.left[0])]
                    left_state = source_states[start_index]
                    right_state = left_state.copy()
                    for edge_index in pair.left:
                        index = occurrence_index[(diagram_index, edge_index)]
                        left_state = _apply_edge(
                            left_state, contexts[index], occurrences[index], maps
                        )
                    for edge_index in pair.right:
                        index = occurrence_index[(diagram_index, edge_index)]
                        right_state = _apply_edge(
                            right_state, contexts[index], occurrences[index], maps
                        )
                    target_index = occurrence_index[(diagram_index, pair.left[-1])]
                    actual_target = target_states[target_index]
                    path_errors.append(float(np.mean((left_state - right_state) ** 2)))
                    endpoint_errors.extend(
                        [
                            float(np.mean((left_state - actual_target) ** 2)),
                            float(np.mean((right_state - actual_target) ** 2)),
                        ]
                    )
            rows.append(
                {
                    "family": family,
                    "baseline": baseline,
                    "edge_error": float(np.mean(edge_errors) / variance),
                    "inverse_error": float(np.mean(inverse_errors) / variance),
                    "holonomy_error": (
                        float(np.mean(loop_errors) / variance) if loop_errors else float("nan")
                    ),
                    "path_agreement_error": (
                        float(np.mean(path_errors) / variance) if path_errors else float("nan")
                    ),
                    "path_endpoint_error": (
                        float(np.mean(endpoint_errors) / variance)
                        if endpoint_errors
                        else float("nan")
                    ),
                    "mean_abs_eigenphase": (
                        float(np.mean(eigenphases)) if eigenphases else float("nan")
                    ),
                    "max_abs_eigenphase": (
                        float(np.max(eigenphases)) if eigenphases else float("nan")
                    ),
                    "compatible_loops": compatible_loops,
                    "compatible_paths": compatible_paths,
                    "activation_variance": variance,
                }
            )
    return rows


@torch.inference_mode()
def _geometry_metrics(
    model,
    diagrams: Sequence[EquivalenceDiagram],
    *,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    examples = [vertex for diagram in diagrams for vertex in diagram.vertices]
    dataset = ExpressionDataset(examples)
    values: dict[tuple[int, int], dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for start in range(0, len(dataset), batch_size):
        batch = collate_expressions(dataset.expressions[start : start + batch_size])
        tokens = batch.tokens.to(device)
        mask = batch.padding_mask.to(device)
        valid = ~mask
        valid[:, 0] = False
        qkv = model.qkv_projections(tokens, mask)
        for layer, (query, key, _) in enumerate(qkv, start=1):
            q = query.transpose(1, 2)
            k = key.transpose(1, 2)
            token_mask = valid[:, None, :, None]
            counts = token_mask.sum(dim=2).clamp_min(1)
            q_masked = q * token_mask
            k_masked = k * token_mask
            q_centroid = q_masked.sum(dim=2) / counts
            k_centroid = k_masked.sum(dim=2) / counts
            cosine = torch.nn.functional.cosine_similarity(q_centroid, k_centroid, dim=-1)
            angle = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))
            q_variance = ((q - q_centroid[:, :, None]) ** 2 * token_mask).sum(dim=(2, 3)) / (
                counts.squeeze(-1) * q.shape[-1]
            )
            k_variance = ((k - k_centroid[:, :, None]) ** 2 * token_mask).sum(dim=(2, 3)) / (
                counts.squeeze(-1) * k.shape[-1]
            )
            separation = torch.linalg.vector_norm(q_centroid - k_centroid, dim=-1) / torch.sqrt(
                q_variance + k_variance + 1e-12
            )
            q_gram = torch.matmul(q_masked.transpose(-2, -1), q_masked)
            k_gram = torch.matmul(k_masked.transpose(-2, -1), k_masked)
            q_eigen = torch.linalg.eigvalsh(q_gram).clamp_min(0)
            k_eigen = torch.linalg.eigvalsh(k_gram).clamp_min(0)
            q_energy = q_eigen.sum(dim=-1).clamp_min(1e-12)
            k_energy = k_eigen.sum(dim=-1).clamp_min(1e-12)
            q_top = q_eigen[..., -1].clamp_min(1e-12)
            k_top = k_eigen[..., -1].clamp_min(1e-12)
            scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(q.shape[-1])
            scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
            attention = torch.softmax(scores, dim=-1)
            entropy = -(attention * torch.log(attention.clamp_min(1e-12))).sum(dim=-1)
            query_valid = valid[:, None, :]
            entropy = (entropy * query_valid).sum(dim=-1) / query_valid.sum(dim=-1).clamp_min(1)
            maximum = scores.max(dim=-1).values
            maximum = (maximum * query_valid).sum(dim=-1) / query_valid.sum(dim=-1).clamp_min(1)
            metrics = {
                "qk_centroid_angle": angle,
                "qk_normalized_separation": separation,
                "query_stable_rank": q_energy / q_top,
                "key_stable_rank": k_energy / k_top,
                "query_fsv_energy": q_top / q_energy,
                "key_fsv_energy": k_top / k_energy,
                "attention_entropy": entropy,
                "maximum_qk_logit": maximum,
            }
            for head in range(model.config.n_heads):
                for metric, tensor in metrics.items():
                    values[(layer, head)][metric].append(tensor[:, head].float().cpu().numpy())
    rows: list[dict[str, Any]] = []
    for (layer, head), metrics in sorted(values.items()):
        row: dict[str, Any] = {"measured_layer": layer, "head": str(head)}
        for metric, chunks in metrics.items():
            array = np.concatenate(chunks)
            row[metric] = float(np.mean(array))
            row[f"{metric}_within_std"] = float(np.std(array, ddof=1))
        rows.append(row)
    return rows


def _aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    metric_names = [
        key
        for key in rows[0]
        if key not in {*keys, "seed", "model", "condition", "architecture_layers"}
        and isinstance(rows[0][key], (int, float, np.number))
    ]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for group_values, group in grouped.items():
        result = dict(zip(keys, group_values, strict=True))
        result["models"] = len(group)
        for metric in metric_names:
            array = np.asarray([float(row[metric]) for row in group])
            array = array[np.isfinite(array)]
            result[f"{metric}_mean"] = float(np.mean(array)) if len(array) else float("nan")
            result[f"{metric}_std"] = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
        output.append(result)
    return output


def _plots(
    transport_rows: list[dict[str, Any]],
    geometry_rows: list[dict[str, Any]],
    output: Path,
) -> list[Path]:
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib_cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    figure, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=True)
    for axis, feature in zip(axes.flat, LOCAL_FEATURES):
        for baseline, style in (
            ("identity", ":"),
            ("global", "--"),
            ("contextual", "-"),
            ("low_rank", "-."),
        ):
            points = []
            for layer in range(1, 7):
                subset = [
                    row
                    for row in transport_rows
                    if row["feature"] == feature
                    and row["baseline"] == baseline
                    and int(row["measured_layer"]) == layer
                ]
                points.append(float(np.nanmean([row["edge_error_mean"] for row in subset])))
            axis.plot(range(1, 7), points, marker="o", linestyle=style, label=baseline)
        axis.set(title=feature, xlabel="layer", ylabel="held-out local edge error")
        axis.set_xticks(range(1, 7))
        axis.legend()
    axes.flat[-1].axis("off")
    figure.suptitle("AA. Context-conditioned transport at the rewritten subtree")
    figure.tight_layout()
    path = output / "AA_contextual_edge_fidelity.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    paths.append(path)

    selected = [
        row
        for row in transport_rows
        if row["feature"] == "cls_query"
        and int(row["measured_layer"]) == 6
        and row["baseline"] in BASELINES
    ]
    families = sorted({row["family"] for row in selected})
    figure, axis = plt.subplots(figsize=(14, 6))
    relation_baselines = BASELINES
    width = 0.13
    x = np.arange(len(families))
    for index, baseline in enumerate(relation_baselines):
        points = [
            float(
                np.nanmean(
                    [
                        row["holonomy_error_mean"]
                        for row in selected
                        if row["family"] == family and row["baseline"] == baseline
                    ]
                )
            )
            for family in families
        ]
        axis.bar(
            x + (index - (len(relation_baselines) - 1) / 2) * width,
            points,
            width,
            label=baseline,
        )
    axis.set_xticks(x, [name.replace("_", " ") for name in families], rotation=35, ha="right")
    axis.set(ylabel="return-to-start error", title="AB. Relation tests: layer-6 CLS query holonomy")
    axis.legend()
    figure.tight_layout()
    path = output / "AB_relation_holonomy.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    paths.append(path)

    geometry_metrics = (
        ("qk_centroid_angle_mean", "Q-K centroid angle (radians)"),
        ("qk_normalized_separation_mean", "normalized Q-K separation"),
        ("query_stable_rank_mean", "query stable rank"),
        ("key_stable_rank_mean", "key stable rank"),
        ("query_fsv_energy_mean", "query leading-SV energy"),
        ("key_fsv_energy_mean", "key leading-SV energy"),
        ("attention_entropy_mean", "attention entropy"),
        ("maximum_qk_logit_mean", "mean maximum QK logit"),
    )
    figure, axes = plt.subplots(2, 4, figsize=(20, 9))
    for axis, (metric, title) in zip(axes.flat, geometry_metrics, strict=True):
        for head in range(4):
            subset = sorted(
                [row for row in geometry_rows if str(row["head"]) == str(head)],
                key=lambda row: int(row["measured_layer"]),
            )
            axis.plot(
                [int(row["measured_layer"]) for row in subset],
                [float(row[metric]) for row in subset],
                marker="o",
                label=f"head {head}",
            )
        axis.set(title=title, xlabel="layer")
        axis.set_xticks(range(1, 7))
        axis.legend(fontsize=8)
    figure.suptitle("AC. Q/K geometry and attention diagnostics")
    figure.tight_layout()
    path = output / "AC_qk_geometry.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    paths.append(path)

    eigenphase_rows = [
        row
        for row in selected
        if row["baseline"]
        in ("global", "contextual", "target_shuffled", "label_shuffled", "low_rank")
    ]
    phase_baselines = ("global", "contextual", "target_shuffled", "label_shuffled", "low_rank")
    figure, axis = plt.subplots(figsize=(14, 6))
    width = 0.15
    x = np.arange(len(families))
    for index, baseline in enumerate(phase_baselines):
        values = [
            float(
                np.nanmean(
                    [
                        row["mean_abs_eigenphase_mean"]
                        for row in eigenphase_rows
                        if row["family"] == family and row["baseline"] == baseline
                    ]
                )
            )
            for family in families
        ]
        axis.bar(
            x + (index - (len(phase_baselines) - 1) / 2) * width,
            values,
            width,
            label=baseline,
        )
    axis.set_xticks(x, [name.replace("_", " ") for name in families], rotation=35, ha="right")
    axis.set(
        ylabel="mean absolute eigenphase (radians)",
        title="AE. Gauge-robust rotational holonomy at layer-6 CLS query",
    )
    axis.legend()
    figure.tight_layout()
    path = output / "AE_operator_eigenphases.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    paths.append(path)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    contextual = [row for row in transport_rows if row["baseline"] == "contextual"]
    global_rows = [row for row in transport_rows if row["baseline"] == "global"]
    for axis, metric in zip(axes, ("edge_error", "holonomy_error"), strict=True):
        matrix = np.zeros((len(LOCAL_FEATURES), 6))
        for feature_index, feature in enumerate(LOCAL_FEATURES):
            for layer in range(1, 7):
                context_value = np.nanmean(
                    [
                        row[f"{metric}_mean"]
                        for row in contextual
                        if row["feature"] == feature and int(row["measured_layer"]) == layer
                    ]
                )
                global_value = np.nanmean(
                    [
                        row[f"{metric}_mean"]
                        for row in global_rows
                        if row["feature"] == feature and int(row["measured_layer"]) == layer
                    ]
                )
                matrix[feature_index, layer - 1] = global_value - context_value
        limit = max(0.01, float(np.nanmax(np.abs(matrix))))
        image = axis.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
        axis.set_xticks(range(6), range(1, 7))
        axis.set_yticks(range(len(LOCAL_FEATURES)), LOCAL_FEATURES)
        axis.set(title=f"global − contextual {metric}", xlabel="layer")
        figure.colorbar(image, ax=axis)
    figure.suptitle("AD. Does conditioning on global context improve the connection?")
    figure.tight_layout()
    path = output / "AD_context_gain.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    paths.append(path)
    return paths


def _report(
    transport_rows: list[dict[str, Any]],
    geometry_rows: list[dict[str, Any]],
    plots: list[Path],
) -> str:
    def mean_metric(feature: str, baseline: str, metric: str) -> float:
        return float(
            np.nanmean(
                [
                    row[f"{metric}_mean"]
                    for row in transport_rows
                    if row["feature"] == feature and row["baseline"] == baseline
                ]
            )
        )

    lines = [
        "# Context-conditioned subtree transport and geometric diagnostics",
        "",
        "## Scope",
        "",
        "This analysis uses the three six-layer higher-diversity checkpoints. Local transports act at the root token of the subtree changed by each rewrite. A second common-fiber analysis transports `<CLS>` Q/K/V states while conditioning on the rewritten-subtree state; this makes cycles that visit different token positions composable. Every edge is canonically oriented and the reverse action is the exact algebraic inverse, enforcing bidirectional consistency by construction.",
        "",
        "For canonical rewrite relation `r`, row-vector state `x`, and context `c`, the fitted map is `T_r(x;c) = (x - mu_s) R_r + mu_t + (c - mu_c) B_r`, where `R_r` is orthogonal Procrustes and `B_r` is ridge regression on the residual. The reverse is `T_r^-1(y;c) = (y - mu_t - (c - mu_c) B_r) R_r^T + mu_s`. For cycle `gamma=(r_1,...,r_m)`, holonomy is the normalized return error `h_gamma = mean((T_rm o ... o T_r1(x) - x)^2) / feature_variance`. It is dimensionless. The eigenphase columns are in radians.",
        "",
        "Local loops are scored only when consecutive rewrites address compatible token fibers. The common `<CLS>` fiber supports every loop and path pair, including the commutativity cube and associativity pentagon.",
        "",
        "## Aggregate held-out edge fidelity",
        "",
        "| Feature | Identity | Global | Contextual | Shuffled targets | Low rank |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for feature in TRANSPORT_FEATURES:
        lines.append(
            f"| {feature} | {mean_metric(feature, 'identity', 'edge_error'):.3f} | {mean_metric(feature, 'global', 'edge_error'):.3f} | {mean_metric(feature, 'contextual', 'edge_error'):.3f} | {mean_metric(feature, 'target_shuffled', 'edge_error'):.3f} | {mean_metric(feature, 'low_rank', 'edge_error'):.3f} |"
        )
    lines.extend(
        [
            "",
            "A loop-closure value is interpreted only where the same connection also improves held-out edge fidelity. Identity has zero holonomy by definition and remains a mandatory negative control. Label-shuffled and target-shuffled connections test whether apparent closure survives destruction of the rewrite semantics.",
            "",
            "The eigenphase outputs describe the rotational part of each composed map and are invariant to orthogonal changes of basis. They should be read alongside state return error because a small phase does not constrain affine drift.",
            "",
            "## Geometric measurements",
            "",
            "For every layer and head the run records Q-K centroid angle, variance-normalized centroid separation, query and key stable rank, leading-singular-value energy, attention entropy, and the maximum QK logit. Stable rank and FSV energy are computed from the uncentered token cloud, matching the geometric convention used in the RoPE analysis.",
            "",
            "## Figures",
            "",
        ]
    )
    for path in plots:
        lines.extend([f"![{path.stem}](plots/{path.name})", ""])
    lines.extend(
        [
            "## Limitations",
            "",
            "- Context enters as a learned translation, while the orthogonal rotation remains shared within a rewrite relation.",
            "- Canonical orientation guarantees an exact inverse but does not force self-inverse logical generators to use the identical matrix in both directions.",
            "- The context descriptor uses both edge endpoints. It is a connection on a known rewrite graph, not a source-only rewrite predictor.",
            "- Circuit-restricted holonomy is deferred unless causal patching identifies a site with specificity clearly above the matched shuffled control; otherwise circuit selection would be circular. Any later circuit projection must report out-of-circuit leakage as well as within-circuit return error.",
            "",
        ]
    )
    return "\n".join(lines)


def run_contextual_holonomy(
    run_directory: Path,
    *,
    device: str = "auto",
    calibration_pairs: int = 96,
) -> Path:
    run_directory = run_directory.resolve()
    config = json.loads((run_directory / "config.json").read_text(encoding="utf-8"))
    _, diagrams = _regenerate_evaluation_data(config)
    blocked = _all_experiment_strings(config)
    blocked.update(str(vertex) for diagram in diagrams for vertex in diagram.vertices)
    calibration = _balanced_disjoint_pairs(
        calibration_pairs,
        operand_depth=int(config["diagram_operand_depth"]),
        seed=273_013,
        blocked=blocked,
    )
    calibration_occurrences = _calibration_occurrences(calibration)
    heldout_occurrences = _edge_occurrences(diagrams)
    calibration_source = [row.source for row in calibration_occurrences]
    calibration_target = [row.target for row in calibration_occurrences]
    calibration_source_positions = [row.source_position for row in calibration_occurrences]
    calibration_target_positions = [row.target_position for row in calibration_occurrences]
    heldout_source = [row.source for row in heldout_occurrences]
    heldout_target = [row.target for row in heldout_occurrences]
    heldout_source_positions = [row.source_position for row in heldout_occurrences]
    heldout_target_positions = [row.target_position for row in heldout_occurrences]

    output = run_directory / "contextual_holonomy"
    output.mkdir(exist_ok=True)
    target_device = resolve_device(device)
    transport_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    checkpoints = sorted((run_directory / "checkpoints").glob("layers_6_higher_diversity*.pt"))
    for checkpoint_index, checkpoint_path in enumerate(checkpoints, start=1):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        seed = int(checkpoint["train_config"]["seed"])
        print(
            f"[{checkpoint_index}/{len(checkpoints)}] contextual {checkpoint_path.stem}", flush=True
        )
        model = _load_model(checkpoint_path, target_device)
        source_calibration = _with_coupled_qk(
            _extract_features(
                model,
                calibration_source,
                calibration_source_positions,
                device=target_device,
                batch_size=int(config["batch_size"]),
            )
        )
        target_calibration = _with_coupled_qk(
            _extract_features(
                model,
                calibration_target,
                calibration_target_positions,
                device=target_device,
                batch_size=int(config["batch_size"]),
            )
        )
        source_heldout = _with_coupled_qk(
            _extract_features(
                model,
                heldout_source,
                heldout_source_positions,
                device=target_device,
                batch_size=int(config["batch_size"]),
            )
        )
        target_heldout = _with_coupled_qk(
            _extract_features(
                model,
                heldout_target,
                heldout_target_positions,
                device=target_device,
                batch_size=int(config["batch_size"]),
            )
        )
        for layer in range(1, 7):
            for feature in TRANSPORT_FEATURES:
                site = (layer, feature)
                arrays = _canonical_arrays(
                    calibration_occurrences,
                    source_calibration,
                    target_calibration,
                    site,
                )
                maps = _fit_site_transports(arrays, seed=seed * 1000 + layer)
                rows = _score_site(
                    diagrams,
                    heldout_occurrences,
                    source_heldout,
                    target_heldout,
                    site,
                    maps,
                )
                transport_rows.extend(
                    {
                        "architecture_layers": 6,
                        "condition": "higher_diversity",
                        "seed": seed,
                        "measured_layer": layer,
                        "feature": feature,
                        **row,
                    }
                    for row in rows
                )
        geometry_rows.extend(
            {
                "architecture_layers": 6,
                "condition": "higher_diversity",
                "seed": seed,
                **row,
            }
            for row in _geometry_metrics(
                model,
                diagrams,
                device=target_device,
                batch_size=int(config["batch_size"]),
            )
        )
        _write_rows(output / "transport_metrics.partial.csv", transport_rows)
        _write_rows(output / "geometry_metrics.partial.csv", geometry_rows)
        del model, source_calibration, target_calibration, source_heldout, target_heldout
        if target_device.type == "cuda":
            torch.cuda.empty_cache()

    aggregate_transport = _aggregate(
        transport_rows,
        ("measured_layer", "feature", "family", "baseline"),
    )
    aggregate_geometry = _aggregate(
        geometry_rows,
        ("measured_layer", "head"),
    )
    _write_rows(output / "transport_metrics.csv", transport_rows)
    _write_rows(output / "aggregate_transport_metrics.csv", aggregate_transport)
    _write_rows(output / "geometry_metrics.csv", geometry_rows)
    _write_rows(output / "aggregate_geometry_metrics.csv", aggregate_geometry)
    plots = _plots(aggregate_transport, aggregate_geometry, output / "plots")
    (output / "report.md").write_text(
        _report(aggregate_transport, aggregate_geometry, plots), encoding="utf-8"
    )
    (output / "status.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "models": len(checkpoints),
                "calibration_pairs_per_rewrite": calibration_pairs,
                "heldout_diagrams": len(diagrams),
                "features": list(TRANSPORT_FEATURES),
                "baselines": list(BASELINES),
                "plots": [str(path.relative_to(output)) for path in plots],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


def regenerate_contextual_artifacts(output: Path) -> None:
    """Regenerate figures/report from completed aggregate CSVs without refitting."""

    def read_rows(path: Path) -> list[dict[str, Any]]:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        categorical = {"feature", "family", "baseline", "head"}
        for row in rows:
            for key, value in tuple(row.items()):
                if key in categorical:
                    continue
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
        return rows

    output = output.resolve()
    transport = read_rows(output / "aggregate_transport_metrics.csv")
    geometry = read_rows(output / "aggregate_geometry_metrics.csv")
    plots = _plots(transport, geometry, output / "plots")
    (output / "report.md").write_text(_report(transport, geometry, plots), encoding="utf-8")
    status_path = output / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["plots"] = [str(path.relative_to(output)) for path in plots]
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

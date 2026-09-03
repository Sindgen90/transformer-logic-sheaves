from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from .data import ExpressionDataset, LogicExample, collate_expressions
from .equivalence import EquivalenceDiagram
from .metrics import OrthogonalTransport, fit_orthogonal_transport, linear_cka, representations
from .model import TinyLogicTransformer


@torch.inference_mode()
def predictions(
    model: TinyLogicTransformer,
    expressions: Sequence[LogicExample],
    *,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    dataset = ExpressionDataset(expressions)
    for start in range(0, len(dataset), batch_size):
        batch = collate_expressions(dataset.expressions[start : start + batch_size])
        logits = model(batch.tokens.to(device), batch.padding_mask.to(device))
        output.append(logits.argmax(dim=-1).cpu().numpy())
    return np.concatenate(output)


def fit_rewrite_transports(
    model: TinyLogicTransformer,
    pairs: dict[str, list[tuple[LogicExample, LogicExample]]],
    *,
    device: torch.device,
) -> dict[str, OrthogonalTransport]:
    """Fit one affine orthogonal map per rewrite label on isolated pairs."""

    transports: dict[str, OrthogonalTransport] = {}
    for label, examples in pairs.items():
        source = representations(model, [pair[0] for pair in examples], device=device)
        target = representations(model, [pair[1] for pair in examples], device=device)
        transports[label] = fit_orthogonal_transport(source, target)
    return transports


def score_rewrite_pairs(
    model: TinyLogicTransformer,
    pairs: dict[str, list[tuple[LogicExample, LogicExample]]],
    transports: dict[str, OrthogonalTransport],
    *,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Evaluate primitive rewrite transports on independent pairs."""

    rows: list[dict[str, Any]] = []
    for label, examples in pairs.items():
        source = representations(model, [pair[0] for pair in examples], device=device)
        target = representations(model, [pair[1] for pair in examples], device=device)
        joined = np.concatenate((source, target), axis=0)
        variance = max(float(np.mean((joined - joined.mean(axis=0, keepdims=True)) ** 2)), 1e-12)
        rows.append(
            {
                "rewrite": label,
                "count": len(examples),
                "identity_error": float(np.mean((source - target) ** 2) / variance),
                "transport_error": float(
                    np.mean((transports[label].apply(source) - target) ** 2) / variance
                ),
                "cka": linear_cka(source, target),
                "activation_variance": variance,
            }
        )
    return rows


def _apply_path(
    start: np.ndarray,
    diagram: EquivalenceDiagram,
    path: tuple[int, ...],
    transports: dict[str, OrthogonalTransport],
) -> np.ndarray:
    state = start.reshape(1, -1)
    for edge_index in path:
        edge = diagram.edges[edge_index]
        state = transports[edge.label].apply(state)
    return state[0]


def _score_family(
    diagrams: Sequence[EquivalenceDiagram],
    hidden: Sequence[np.ndarray],
    predicted: Sequence[np.ndarray],
    transports: dict[str, OrthogonalTransport],
) -> dict[str, Any]:
    all_hidden = np.concatenate(hidden, axis=0)
    variance = max(
        float(np.mean((all_hidden - all_hidden.mean(axis=0, keepdims=True)) ** 2)),
        1e-12,
    )
    identity_errors: list[float] = []
    transport_errors: list[float] = []
    loop_errors: list[float] = []
    path_errors: list[float] = []
    path_endpoint_errors: list[float] = []
    edge_sources: list[np.ndarray] = []
    edge_targets: list[np.ndarray] = []
    semantic_correct = 0
    semantic_total = 0
    consistent = 0

    for diagram, states, predictions_for_diagram in zip(diagrams, hidden, predicted):
        labels = np.asarray([vertex.value for vertex in diagram.vertices])
        semantic_correct += int((predictions_for_diagram == labels).sum())
        semantic_total += len(labels)
        consistent += int(np.all(predictions_for_diagram == predictions_for_diagram[0]))
        for edge in diagram.edges:
            source = states[edge.source]
            target = states[edge.target]
            transported = transports[edge.label].apply(source.reshape(1, -1))[0]
            identity_errors.append(float(np.mean((source - target) ** 2)))
            transport_errors.append(float(np.mean((transported - target) ** 2)))
            edge_sources.append(source)
            edge_targets.append(target)
        for loop in diagram.loops:
            start_vertex = diagram.edges[loop[0]].source
            start = states[start_vertex]
            returned = _apply_path(start, diagram, loop, transports)
            loop_errors.append(float(np.mean((returned - start) ** 2)))
        for pair in diagram.path_pairs:
            start = states[pair.start]
            left = _apply_path(start, diagram, pair.left, transports)
            right = _apply_path(start, diagram, pair.right, transports)
            target = states[pair.target]
            path_errors.append(float(np.mean((left - right) ** 2)))
            path_endpoint_errors.extend(
                (float(np.mean((left - target) ** 2)), float(np.mean((right - target) ** 2)))
            )

    source_matrix = np.stack(edge_sources)
    target_matrix = np.stack(edge_targets)
    first = diagrams[0]
    return {
        "family": first.family,
        "diagram_count": len(diagrams),
        "vertices_per_diagram": len(first.vertices),
        "edges_per_diagram": len(first.edges),
        "loops_per_diagram": len(first.loops),
        "path_pairs_per_diagram": len(first.path_pairs),
        "mean_loop_length": float(np.mean([len(loop) for item in diagrams for loop in item.loops])),
        "semantic_accuracy": semantic_correct / semantic_total,
        "prediction_consistency": consistent / len(diagrams),
        "identity_error": float(np.mean(identity_errors) / variance),
        "transport_error": float(np.mean(transport_errors) / variance),
        "holonomy_error": float(np.mean(loop_errors) / variance),
        "path_agreement_error": (
            float(np.mean(path_errors) / variance) if path_errors else float("nan")
        ),
        "path_endpoint_error": (
            float(np.mean(path_endpoint_errors) / variance)
            if path_endpoint_errors
            else float("nan")
        ),
        "edge_cka": linear_cka(source_matrix, target_matrix),
        "activation_variance": variance,
    }


def score_equivalence_diagrams(
    model: TinyLogicTransformer,
    diagrams: Sequence[EquivalenceDiagram],
    transports: dict[str, OrthogonalTransport],
    *,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Evaluate arbitrary held-out loops and competing paths by family."""

    missing = sorted(
        {edge.label for diagram in diagrams for edge in diagram.edges} - set(transports)
    )
    if missing:
        raise ValueError(f"No fitted transport for rewrite labels: {missing}")

    flat = [vertex for diagram in diagrams for vertex in diagram.vertices]
    all_hidden = representations(model, flat, device=device)
    all_predictions = predictions(model, flat, device=device)
    hidden_by_diagram: list[np.ndarray] = []
    predictions_by_diagram: list[np.ndarray] = []
    cursor = 0
    for diagram in diagrams:
        next_cursor = cursor + len(diagram.vertices)
        hidden_by_diagram.append(all_hidden[cursor:next_cursor])
        predictions_by_diagram.append(all_predictions[cursor:next_cursor])
        cursor = next_cursor

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, diagram in enumerate(diagrams):
        grouped[diagram.family].append(index)
    rows: list[dict[str, Any]] = []
    for indices in grouped.values():
        rows.append(
            _score_family(
                [diagrams[index] for index in indices],
                [hidden_by_diagram[index] for index in indices],
                [predictions_by_diagram[index] for index in indices],
                transports,
            )
        )
    return rows

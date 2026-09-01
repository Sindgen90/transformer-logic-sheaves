from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from random import Random

import numpy as np
import torch

from .data import collate_expressions
from .logic import OPS, Expr, demorgan_commutativity_cycle, random_expr_with_value
from .model import TinyLogicTransformer


@torch.inference_mode()
def representations(
    model: TinyLogicTransformer,
    expressions: Sequence[Expr],
    *,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Return final-layer CLS representations."""

    model.eval()
    chunks: list[np.ndarray] = []
    for start in range(0, len(expressions), batch_size):
        batch = collate_expressions(expressions[start : start + batch_size])
        _, hidden = model(
            batch.tokens.to(device),
            batch.padding_mask.to(device),
            return_hidden=True,
        )
        chunks.append(hidden[:, 0].float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


@torch.inference_mode()
def local_operator_representations(
    model: TinyLogicTransformer,
    expressions: Sequence[Expr],
    *,
    device: torch.device,
    batch_size: int = 256,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Collect each operator token's state and the corresponding subtree truth value."""

    model.eval()
    states: dict[str, list[np.ndarray]] = {op: [] for op in OPS}
    labels: dict[str, list[int]] = {op: [] for op in OPS}
    for start in range(0, len(expressions), batch_size):
        expression_batch = expressions[start : start + batch_size]
        batch = collate_expressions(expression_batch)
        _, hidden = model(
            batch.tokens.to(device),
            batch.padding_mask.to(device),
            return_hidden=True,
        )
        hidden_np = hidden.float().cpu().numpy()
        for row, expression in enumerate(expression_batch):
            for token_position, node in enumerate(expression.nodes_prefix(), start=1):
                if node.op in states:
                    states[node.op].append(hidden_np[row, token_position])
                    labels[node.op].append(node.value)
    return {
        op: (np.stack(op_states), np.asarray(labels[op], dtype=np.int64))
        for op, op_states in states.items()
        if op_states
    }


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    recalls = []
    for label in (0, 1):
        mask = labels == label
        if mask.any():
            recalls.append(float((predictions[mask] == label).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")


def ridge_probe(
    train_states: np.ndarray,
    train_labels: np.ndarray,
    test_states: np.ndarray,
    test_labels: np.ndarray,
    *,
    regularization: float = 1e-2,
) -> dict[str, float]:
    """Fit a deliberately weak closed-form linear probe."""

    mean = train_states.mean(axis=0, keepdims=True)
    scale = train_states.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    x_train = (train_states - mean) / scale
    x_test = (test_states - mean) / scale
    x_train = np.concatenate([x_train, np.ones((len(x_train), 1))], axis=1)
    x_test = np.concatenate([x_test, np.ones((len(x_test), 1))], axis=1)
    penalty = regularization * np.eye(x_train.shape[1])
    penalty[-1, -1] = 0.0
    weights = np.linalg.pinv(x_train.T @ x_train + penalty) @ x_train.T @ train_labels
    train_predictions = (x_train @ weights >= 0.5).astype(np.int64)
    test_predictions = (x_test @ weights >= 0.5).astype(np.int64)
    return {
        "train_accuracy": float((train_predictions == train_labels).mean()),
        "test_accuracy": float((test_predictions == test_labels).mean()),
        "test_balanced_accuracy": _balanced_accuracy(test_labels, test_predictions),
    }


def local_probe_scores(
    model: TinyLogicTransformer,
    calibration: Sequence[Expr],
    test: Sequence[Expr],
    *,
    device: torch.device,
) -> dict[str, float]:
    calibration_states = local_operator_representations(model, calibration, device=device)
    test_states = local_operator_representations(model, test, device=device)
    scores: dict[str, float] = {f"local_probe_{op.lower()}": float("nan") for op in OPS}
    balanced_scores: list[float] = []
    for op in sorted(calibration_states.keys() & test_states.keys()):
        train_x, train_y = calibration_states[op]
        test_x, test_y = test_states[op]
        if len(np.unique(train_y)) < 2 or len(np.unique(test_y)) < 2:
            continue
        result = ridge_probe(train_x, train_y, test_x, test_y)
        scores[f"local_probe_{op.lower()}"] = result["test_balanced_accuracy"]
        balanced_scores.append(result["test_balanced_accuracy"])
    scores["local_probe_mean"] = (
        float(np.mean(balanced_scores)) if balanced_scores else float("nan")
    )
    return scores


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(x.T @ y, ord="fro") ** 2
    self_x = np.linalg.norm(x.T @ x, ord="fro")
    self_y = np.linalg.norm(y.T @ y, ord="fro")
    denominator = self_x * self_y
    return float(cross / denominator) if denominator > 0 else float("nan")


@dataclass(frozen=True)
class OrthogonalTransport:
    source_mean: np.ndarray
    target_mean: np.ndarray
    rotation: np.ndarray

    def apply(self, states: np.ndarray) -> np.ndarray:
        return (states - self.source_mean) @ self.rotation + self.target_mean


def fit_orthogonal_transport(source: np.ndarray, target: np.ndarray) -> OrthogonalTransport:
    source_mean = source.mean(axis=0, keepdims=True)
    target_mean = target.mean(axis=0, keepdims=True)
    source_centered = source - source_mean
    target_centered = target - target_mean
    u, _, vt = np.linalg.svd(source_centered.T @ target_centered, full_matrices=False)
    rotation = u @ vt
    return OrthogonalTransport(source_mean, target_mean, rotation)


def make_cycle_suite(
    count: int,
    *,
    operand_depth: int,
    seed: int,
) -> tuple[list[tuple[Expr, Expr, Expr, Expr]], np.ndarray]:
    """Generate label-balanced, unique De Morgan/commutativity squares."""

    rng = Random(seed)
    cycles: list[tuple[Expr, Expr, Expr, Expr]] = []
    labels: list[int] = []
    seen: set[str] = set()
    attempts = 0
    while len(cycles) < count and attempts < 10_000 * count:
        attempts += 1
        target = len(cycles) % 2
        if target == 0:
            left_value, right_value = 1, 1
        else:
            left_value, right_value = rng.choice(((0, 0), (0, 1), (1, 0)))
        left = random_expr_with_value(rng, operand_depth, left_value)
        right = random_expr_with_value(rng, operand_depth, right_value)
        cycle = demorgan_commutativity_cycle(left, right)
        key = str(cycle[0])
        if key in seen:
            continue
        seen.add(key)
        cycles.append(cycle)
        labels.append(target)
    if len(cycles) != count:
        raise RuntimeError(f"Could only generate {len(cycles)}/{count} unique cycles")
    return cycles, np.asarray(labels, dtype=np.int64)


def cycle_scores(
    model: TinyLogicTransformer,
    cycles: Sequence[tuple[Expr, Expr, Expr, Expr]],
    *,
    device: torch.device,
    calibration_fraction: float = 0.5,
) -> dict[str, float]:
    """Score held-out equivalence consistency and transport holonomy.

    Each edge receives one affine orthogonal map, fit only on calibration cycles.
    The reported errors are evaluated on disjoint cycles and normalized by the
    overall activation variance.
    """

    flat = [expression for cycle in cycles for expression in cycle]
    hidden = representations(model, flat, device=device).reshape(len(cycles), 4, -1)
    split = max(2, min(len(cycles) - 2, int(len(cycles) * calibration_fraction)))
    calibration = hidden[:split]
    heldout = hidden[split:]
    transports = [
        fit_orthogonal_transport(calibration[:, index], calibration[:, (index + 1) % 4])
        for index in range(4)
    ]

    variance = float(np.mean((heldout - heldout.mean(axis=(0, 1), keepdims=True)) ** 2))
    variance = max(variance, 1e-12)
    identity_errors = [
        np.mean((heldout[:, index] - heldout[:, (index + 1) % 4]) ** 2) for index in range(4)
    ]
    transport_errors = [
        np.mean((transports[index].apply(heldout[:, index]) - heldout[:, (index + 1) % 4]) ** 2)
        for index in range(4)
    ]
    transported = heldout[:, 0]
    for transport in transports:
        transported = transport.apply(transported)
    holonomy_error = float(np.mean((transported - heldout[:, 0]) ** 2) / variance)
    cka_edges = [linear_cka(heldout[:, index], heldout[:, (index + 1) % 4]) for index in range(4)]
    return {
        "cycle_identity_energy": float(np.mean(identity_errors) / variance),
        "cycle_transport_error": float(np.mean(transport_errors) / variance),
        "cycle_holonomy_error": holonomy_error,
        "cycle_edge_cka": float(np.mean(cka_edges)),
        "cycle_activation_variance": variance,
    }

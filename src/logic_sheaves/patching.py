from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any

import numpy as np
import torch

from .data import collate_expressions
from .logic import BINARY_OPS, OPS, Expr, binary, random_expr_with_value, unary
from .model import TinyLogicTransformer


@dataclass(frozen=True)
class PatchingSuite:
    recipients: tuple[Expr, ...]
    counterfactual_donors: tuple[Expr, ...]
    equivalent_donors: tuple[Expr, ...]
    operators: tuple[str, ...]
    target_positions: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.recipients)

    def subset(self, count: int) -> PatchingSuite:
        count = min(count, len(self))
        return PatchingSuite(
            recipients=self.recipients[:count],
            counterfactual_donors=self.counterfactual_donors[:count],
            equivalent_donors=self.equivalent_donors[:count],
            operators=self.operators[:count],
            target_positions=self.target_positions[:count],
        )


def _input_values(rng: Random, op: str, target: int) -> tuple[int, ...]:
    if op == "NOT":
        return (1 - target,)
    satisfying = {
        ("AND", 0): ((0, 0), (0, 1), (1, 0)),
        ("AND", 1): ((1, 1),),
        ("OR", 0): ((0, 0),),
        ("OR", 1): ((0, 1), (1, 0), (1, 1)),
        ("XOR", 0): ((0, 0), (1, 1)),
        ("XOR", 1): ((0, 1), (1, 0)),
    }
    return rng.choice(satisfying[(op, target)])


def _operation_with_value(rng: Random, op: str, target: int, operand_depth: int) -> Expr:
    inputs = _input_values(rng, op, target)
    operands = tuple(random_expr_with_value(rng, operand_depth, value) for value in inputs)
    if op == "NOT":
        return unary(op, operands[0])
    if op in BINARY_OPS:
        return binary(op, operands[0], operands[1])
    raise ValueError(f"Unsupported patch operator {op}")


def make_patching_suite(
    count: int,
    *,
    operand_depth: int = 2,
    context_depth: int = 1,
    seed: int = 55_661,
) -> PatchingSuite:
    """Create causal pairs with known counterfactual outputs.

    Every expression has the form XOR(target_subtree, context). Counterfactual
    donors use the same target operator but the opposite subtree truth value,
    so the correct root output necessarily flips. Equivalent donors use the
    same target truth value with independently generated syntax.
    """

    if count < len(OPS) * 2:
        raise ValueError(f"count must be at least {len(OPS) * 2} to cover operators and labels")
    rng = Random(seed)
    recipients: list[Expr] = []
    counterfactual_donors: list[Expr] = []
    equivalent_donors: list[Expr] = []
    operators: list[str] = []

    for index in range(count):
        op = OPS[index % len(OPS)]
        target_value = (index // len(OPS)) % 2
        context_value = rng.randrange(2)
        context = random_expr_with_value(rng, context_depth, context_value)
        recipient_target = _operation_with_value(rng, op, target_value, operand_depth)
        counterfactual_target = _operation_with_value(rng, op, 1 - target_value, operand_depth)
        equivalent_target = _operation_with_value(rng, op, target_value, operand_depth)
        recipient = binary("XOR", recipient_target, context)
        counterfactual = binary("XOR", counterfactual_target, context)
        equivalent = binary("XOR", equivalent_target, context)
        if counterfactual.value == recipient.value:
            raise AssertionError("Counterfactual donor must flip the root output")
        if equivalent.value != recipient.value:
            raise AssertionError("Equivalent donor must preserve the root output")
        recipients.append(recipient)
        counterfactual_donors.append(counterfactual)
        equivalent_donors.append(equivalent)
        operators.append(op)

    # [CLS] is position 0 and the root XOR is position 1, so the target
    # operator is always at position 2 regardless of subtree length.
    return PatchingSuite(
        recipients=tuple(recipients),
        counterfactual_donors=tuple(counterfactual_donors),
        equivalent_donors=tuple(equivalent_donors),
        operators=tuple(operators),
        target_positions=(2,) * count,
    )


def _effect_fraction(clean: np.ndarray, donor: np.ndarray, patched: np.ndarray) -> float:
    desired = donor - clean
    actual = patched - clean
    denominator = float(desired @ desired)
    if denominator < 1e-12:
        return float("nan")
    return float((actual @ desired) / denominator)


@torch.inference_mode()
def activation_patching_scores(
    model: TinyLogicTransformer,
    suite: PatchingSuite,
    *,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Patch operator states at every stage and measure causal transfer."""

    model.eval()
    recipient_batch = collate_expressions(suite.recipients)
    counterfactual_batch = collate_expressions(suite.counterfactual_donors)
    equivalent_batch = collate_expressions(suite.equivalent_donors)

    recipient_tokens = recipient_batch.tokens.to(device)
    recipient_mask = recipient_batch.padding_mask.to(device)
    counterfactual_tokens = counterfactual_batch.tokens.to(device)
    counterfactual_mask = counterfactual_batch.padding_mask.to(device)
    equivalent_tokens = equivalent_batch.tokens.to(device)
    equivalent_mask = equivalent_batch.padding_mask.to(device)
    positions = torch.tensor(suite.target_positions, device=device, dtype=torch.long)

    clean_logits = model(recipient_tokens, recipient_mask)
    counterfactual_logits = model(counterfactual_tokens, counterfactual_mask)
    counterfactual_stages = model.stage_representations(counterfactual_tokens, counterfactual_mask)
    equivalent_stages = model.stage_representations(equivalent_tokens, equivalent_mask)

    clean_labels = recipient_batch.labels.numpy()
    counterfactual_labels = counterfactual_batch.labels.numpy()
    clean_predictions = clean_logits.argmax(dim=-1).cpu().numpy()
    counterfactual_donor_predictions = counterfactual_logits.argmax(dim=-1).cpu().numpy()
    clean_logit_difference = (clean_logits[:, 1] - clean_logits[:, 0]).cpu().numpy()
    counterfactual_logit_difference = (
        (counterfactual_logits[:, 1] - counterfactual_logits[:, 0]).cpu().numpy()
    )
    rows: list[dict[str, Any]] = []

    for stage in range(model.config.n_layers + 1):
        batch_rows = torch.arange(len(suite), device=device)
        counterfactual_values = counterfactual_stages[stage][batch_rows, positions]
        equivalent_values = equivalent_stages[stage][batch_rows, positions]
        counterfactual_patched = model.forward_patched(
            recipient_tokens,
            recipient_mask,
            patch_stage=stage,
            patch_positions=positions,
            patch_values=counterfactual_values,
        )
        equivalent_patched = model.forward_patched(
            recipient_tokens,
            recipient_mask,
            patch_stage=stage,
            patch_positions=positions,
            patch_values=equivalent_values,
        )
        counterfactual_predictions = counterfactual_patched.argmax(dim=-1).cpu().numpy()
        equivalent_predictions = equivalent_patched.argmax(dim=-1).cpu().numpy()
        counterfactual_patched_difference = (
            (counterfactual_patched[:, 1] - counterfactual_patched[:, 0]).cpu().numpy()
        )

        groups: list[tuple[str, np.ndarray]] = [
            ("ALL", np.ones(len(suite), dtype=bool)),
            *((op, np.asarray(suite.operators) == op) for op in OPS),
        ]
        for operator, selection in groups:
            rows.append(
                {
                    "patch_stage": stage,
                    "patch_stage_name": "embedding" if stage == 0 else f"layer_{stage}",
                    "operator": operator,
                    "count": int(selection.sum()),
                    "clean_accuracy": float(
                        (clean_predictions[selection] == clean_labels[selection]).mean()
                    ),
                    "counterfactual_donor_accuracy": float(
                        (
                            counterfactual_donor_predictions[selection]
                            == counterfactual_labels[selection]
                        ).mean()
                    ),
                    "counterfactual_target_accuracy": float(
                        (
                            counterfactual_predictions[selection]
                            == counterfactual_labels[selection]
                        ).mean()
                    ),
                    "counterfactual_flip_rate": float(
                        (
                            counterfactual_predictions[selection] != clean_predictions[selection]
                        ).mean()
                    ),
                    "counterfactual_effect_fraction": _effect_fraction(
                        clean_logit_difference[selection],
                        counterfactual_logit_difference[selection],
                        counterfactual_patched_difference[selection],
                    ),
                    "equivalent_preservation": float(
                        (equivalent_predictions[selection] == clean_predictions[selection]).mean()
                    ),
                    "equivalent_truth_accuracy": float(
                        (equivalent_predictions[selection] == clean_labels[selection]).mean()
                    ),
                }
            )
    return rows


def summarize_patching(
    rows: list[dict[str, Any]],
    *,
    n_layers: int,
) -> dict[str, float]:
    overall = [
        row for row in rows if row["operator"] == "ALL" and 0 < int(row["patch_stage"]) < n_layers
    ]
    if not overall:
        return {
            "patch_best_counterfactual_accuracy": float("nan"),
            "patch_best_effect_fraction": float("nan"),
            "patch_mean_equivalent_preservation": float("nan"),
        }
    return {
        "patch_best_counterfactual_accuracy": float(
            max(row["counterfactual_target_accuracy"] for row in overall)
        ),
        "patch_best_effect_fraction": float(
            max(row["counterfactual_effect_fraction"] for row in overall)
        ),
        "patch_mean_equivalent_preservation": float(
            np.mean([row["equivalent_preservation"] for row in overall])
        ),
    }

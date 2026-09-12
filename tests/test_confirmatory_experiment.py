from __future__ import annotations

import numpy as np

from logic_sheaves.confirmatory_experiment import _Design, _fit_logistic, _model_metrics


def _rows() -> list[dict[str, object]]:
    rows = []
    for index in range(40):
        rows.append(
            {
                "reconstruction_error": index / 40,
                "confidence": 1 + index / 20,
                "expression_length": 8 + index % 4,
                "state_return_error": index / 10,
                "section_energy": index / 8,
                "true_label": index % 2,
                "family": ("square", "pentagon")[index % 2],
                "seed": index % 3,
                "correct": int(index >= 20),
            }
        )
    return rows


def test_frozen_design_has_stable_columns() -> None:
    rows = _rows()
    design = _Design(rows[:30], geometry=True)
    assert design.matrix(rows[:30]).shape[1] == len(design.names)
    assert design.matrix(rows[30:]).shape[1] == len(design.names)


def test_logistic_fit_recovers_ordered_signal() -> None:
    rows = _rows()
    design = _Design(rows, geometry=True)
    labels = np.asarray([row["correct"] for row in rows])
    beta = _fit_logistic(design.matrix(rows), labels)
    probabilities = 1 / (1 + np.exp(-design.matrix(rows) @ beta))
    metrics = _model_metrics(labels, probabilities)
    assert metrics["roc_auc"] > 0.95
    assert metrics["balanced_accuracy"] > 0.85

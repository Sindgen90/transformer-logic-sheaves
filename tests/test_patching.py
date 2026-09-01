import unittest

import numpy as np
import torch

from logic_sheaves.logic import OPS
from logic_sheaves.model import ModelConfig, TinyLogicTransformer
from logic_sheaves.patching import (
    activation_patching_scores,
    make_patching_suite,
    summarize_patching,
)


class PatchingTests(unittest.TestCase):
    def test_suite_has_known_counterfactuals_and_controls(self) -> None:
        suite = make_patching_suite(16, seed=3)
        self.assertEqual(set(suite.operators), set(OPS))
        self.assertTrue(all(position == 2 for position in suite.target_positions))
        for recipient, counterfactual, equivalent in zip(
            suite.recipients,
            suite.counterfactual_donors,
            suite.equivalent_donors,
            strict=True,
        ):
            self.assertNotEqual(recipient.value, counterfactual.value)
            self.assertEqual(recipient.value, equivalent.value)

    def test_patching_scores_cover_every_stage_and_operator(self) -> None:
        suite = make_patching_suite(16, seed=4)
        model = TinyLogicTransformer(ModelConfig(d_model=16, n_heads=4, n_layers=2, d_ff=32))
        rows = activation_patching_scores(model, suite, device=torch.device("cpu"))
        self.assertEqual(len(rows), 3 * (len(OPS) + 1))
        self.assertEqual({row["patch_stage"] for row in rows}, {0, 1, 2})
        summary = summarize_patching(rows, n_layers=2)
        self.assertTrue(np.isfinite(summary["patch_best_effect_fraction"]))


if __name__ == "__main__":
    unittest.main()

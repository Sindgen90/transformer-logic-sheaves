import unittest

import numpy as np

from logic_sheaves.qkv_patching import (
    PATCH_OPERATOR_SPECS,
    _matched_shuffle_indices,
    make_symbolic_qkv_patching_suite,
)


class SymbolicQKVPatchingTests(unittest.TestCase):
    def test_suite_is_balanced_and_has_expected_truth_relations(self) -> None:
        suite = make_symbolic_qkv_patching_suite(count_per_operator=8, seed=9)
        self.assertEqual(len(suite), len(PATCH_OPERATOR_SPECS) * 8)
        for recipient, counterfactual, equivalent, position in zip(
            suite.recipients,
            suite.counterfactual_donors,
            suite.equivalent_donors,
            suite.operator_positions,
            strict=True,
        ):
            self.assertNotEqual(recipient.value, counterfactual.value)
            self.assertEqual(recipient.value, equivalent.value)
            self.assertEqual(position, 8)

    def test_matched_shuffle_is_deranged_and_truth_matched(self) -> None:
        suite = make_symbolic_qkv_patching_suite(count_per_operator=8, seed=10)
        permutation = _matched_shuffle_indices(suite)
        self.assertTrue(np.all(permutation != np.arange(len(suite))))
        for source_index, donor_index in enumerate(permutation):
            self.assertEqual(suite.operators[source_index], suite.operators[donor_index])
            self.assertEqual(
                suite.counterfactual_donors[source_index].value,
                suite.counterfactual_donors[donor_index].value,
            )


if __name__ == "__main__":
    unittest.main()

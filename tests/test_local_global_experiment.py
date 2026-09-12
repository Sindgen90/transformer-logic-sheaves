import unittest

import numpy as np

from logic_sheaves.local_global_experiment import _balanced_splits


class LocalGlobalExperimentTests(unittest.TestCase):
    def test_balanced_splits_match_truth_and_outcome_counts(self) -> None:
        labels = np.asarray([0] * 20 + [1] * 20 + [0] * 12 + [1] * 12)
        correct = np.asarray([True] * 40 + [False] * 24)
        splits = _balanced_splits(labels, correct, seed=3)
        self.assertIsNotNone(splits)
        assert splits is not None
        for outcome, expected in (("correct", True), ("incorrect", False)):
            fit, evaluation = splits[outcome]
            selected = np.concatenate((fit, evaluation))
            self.assertTrue(np.all(correct[selected] == expected))
            self.assertEqual(int((labels[selected] == 0).sum()), 12)
            self.assertEqual(int((labels[selected] == 1).sum()), 12)
            self.assertFalse(set(fit) & set(evaluation))

    def test_balanced_splits_skip_sparse_cells(self) -> None:
        labels = np.asarray([0, 0, 1, 1] * 4)
        correct = np.asarray([True, False, True, False] * 4)
        self.assertIsNone(_balanced_splits(labels, correct, seed=0))


if __name__ == "__main__":
    unittest.main()

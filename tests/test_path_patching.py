import unittest

import numpy as np

from logic_sheaves.equivalence import make_equivalence_suite
from logic_sheaves.path_patching import _matched_shuffle, _path_cases


class PathPatchingTests(unittest.TestCase):
    def test_path_cases_and_shuffle(self) -> None:
        diagrams = make_equivalence_suite(8, operand_depth=1, seed=15)
        cases = _path_cases(diagrams)
        self.assertTrue(cases)
        self.assertTrue(all(case.start.value == case.left_donor.value for case in cases))
        self.assertTrue(all(case.start.value == case.right_donor.value for case in cases))
        permutation = _matched_shuffle(cases)
        self.assertEqual(len(permutation), len(cases))
        self.assertTrue(np.all(permutation != np.arange(len(cases))))


if __name__ == "__main__":
    unittest.main()

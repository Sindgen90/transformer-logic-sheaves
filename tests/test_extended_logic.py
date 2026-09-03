import random
import unittest

from logic_sheaves.data import VOCAB, AssignedExpression, encode, make_symbolic_splits
from logic_sheaves.logic import fixed, nary, random_symbolic_expr, unary, variable


class ExtendedLogicTests(unittest.TestCase):
    def test_variables_nary_and_fixed_arity_operators(self) -> None:
        assignment = (("x0", 1), ("x1", 0), ("x2", 1), ("x3", 0))
        conjunction = nary("AND", variable("x0"), variable("x1"), variable("x2"))
        majority = fixed("MAJ3", variable("x0"), variable("x1"), variable("x2"))
        conditional = fixed("ITE3", variable("x1"), variable("x0"), variable("x3"))
        exactly_one = fixed("EXACT1_3", variable("x0"), variable("x1"), variable("x3"))
        threshold = fixed(
            "ATLEAST2_4",
            variable("x0"),
            variable("x1"),
            variable("x2"),
            variable("x3"),
        )
        self.assertEqual(conjunction.evaluate(dict(assignment)), 0)
        self.assertEqual(majority.evaluate(dict(assignment)), 1)
        self.assertEqual(conditional.evaluate(dict(assignment)), 0)
        self.assertEqual(exactly_one.evaluate(dict(assignment)), 1)
        self.assertEqual(threshold.evaluate(dict(assignment)), 1)
        example = AssignedExpression(unary("NOT", conjunction), assignment)
        self.assertEqual(example.value, 1)
        self.assertIn("AND3", example.prefix_tokens())
        self.assertTrue(all(token in VOCAB for token in example.prefix_tokens()))
        self.assertEqual(len(encode(example)), len(example.prefix_tokens()) + 1)

    def test_symbolic_generation_and_disjoint_splits(self) -> None:
        rng = random.Random(11)
        for depth in range(5):
            for _ in range(20):
                expression = random_symbolic_expr(rng, depth, exact_depth=True)
                self.assertEqual(expression.depth, depth)
        splits = make_symbolic_splits(
            train_size=80,
            validation_size=30,
            test_size=30,
            train_depth=3,
            ood_depths=(4, 5),
            seed=12,
        )
        train = set(map(str, splits.train))
        validation = set(map(str, splits.validation))
        test = set(map(str, splits.test_id))
        self.assertFalse(train & validation)
        self.assertFalse(train & test)
        self.assertFalse(validation & test)
        for depth, examples in splits.test_ood.items():
            self.assertTrue(all(example.depth == depth for example in examples))


if __name__ == "__main__":
    unittest.main()

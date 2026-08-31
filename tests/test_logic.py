import random
import unittest

from logic_sheaves.logic import (
    binary,
    demorgan_commutativity_cycle,
    literal,
    random_expr,
    unary,
)


class LogicTests(unittest.TestCase):
    def test_evaluation_and_prefix_order(self) -> None:
        expression = binary("XOR", binary("AND", literal(1), literal(0)), unary("NOT", literal(0)))
        self.assertEqual(expression.value, 1)
        self.assertEqual(expression.depth, 2)
        self.assertEqual(expression.prefix_tokens(), ["XOR", "AND", "1", "0", "NOT", "0"])

    def test_exact_depth_generation(self) -> None:
        rng = random.Random(7)
        for depth in range(5):
            for _ in range(50):
                self.assertEqual(random_expr(rng, depth, exact_depth=True).depth, depth)

    def test_demorgan_square_is_equivalent(self) -> None:
        for left in (0, 1):
            for right in (0, 1):
                cycle = demorgan_commutativity_cycle(literal(left), literal(right))
                self.assertEqual(len({expression.value for expression in cycle}), 1)


if __name__ == "__main__":
    unittest.main()

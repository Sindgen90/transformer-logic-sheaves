import unittest

import numpy as np

from logic_sheaves.contextual_holonomy import (
    _fit_transport,
    _rewrite_path,
    _token_position,
)
from logic_sheaves.data import AssignedExpression
from logic_sheaves.logic import VARIABLES, binary, variable


class ContextualHolonomyTests(unittest.TestCase):
    def test_rewrite_path_descends_through_unchanged_context(self) -> None:
        left = binary(
            "XOR",
            binary("AND", variable("x0"), variable("x1")),
            variable("x2"),
        )
        right = binary(
            "XOR",
            binary("AND", variable("x1"), variable("x0")),
            variable("x2"),
        )
        path = _rewrite_path(left, right)
        self.assertEqual(path, (0,))
        example = AssignedExpression(left, tuple((name, 0) for name in VARIABLES[:4]))
        self.assertEqual(_token_position(example, path), 8)

    def test_bidirectional_transport_is_exactly_invertible(self) -> None:
        rng = np.random.default_rng(4)
        source = rng.normal(size=(64, 8))
        target = rng.normal(size=(64, 8))
        context = rng.normal(size=(64, 8))
        transport = _fit_transport(source, target, context, ridge=1.0)
        predicted = transport.apply(source, context, 1)
        returned = transport.apply(predicted, context, -1)
        np.testing.assert_allclose(returned, source, atol=1e-10)


if __name__ == "__main__":
    unittest.main()

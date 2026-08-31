import unittest

from logic_sheaves.data import collate_expressions
from logic_sheaves.logic import binary, literal, unary
from logic_sheaves.model import ModelConfig, TinyLogicTransformer


class ModelTests(unittest.TestCase):
    def test_forward_shapes(self) -> None:
        expressions = [
            binary("AND", literal(0), literal(1)),
            unary("NOT", binary("OR", literal(0), literal(1))),
        ]
        batch = collate_expressions(expressions)
        model = TinyLogicTransformer(ModelConfig(d_model=16, n_heads=4, n_layers=1, d_ff=32))
        logits, hidden = model(batch.tokens, batch.padding_mask, return_hidden=True)
        self.assertEqual(logits.shape, (2, 2))
        self.assertEqual(hidden.shape[:2], batch.tokens.shape)


if __name__ == "__main__":
    unittest.main()

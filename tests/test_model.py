import unittest

import torch

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

    def test_stage_capture_and_patching(self) -> None:
        recipients = [
            binary("XOR", binary("AND", literal(0), literal(1)), literal(0)),
            binary("XOR", unary("NOT", literal(0)), literal(1)),
        ]
        donors = [
            binary("XOR", binary("AND", literal(1), literal(1)), literal(0)),
            binary("XOR", unary("NOT", literal(1)), literal(1)),
        ]
        recipient_batch = collate_expressions(recipients)
        donor_batch = collate_expressions(donors)
        model = TinyLogicTransformer(ModelConfig(d_model=16, n_heads=4, n_layers=2, d_ff=32))
        recipient_logits = model(recipient_batch.tokens, recipient_batch.padding_mask)
        donor_logits = model(donor_batch.tokens, donor_batch.padding_mask)
        donor_stages = model.stage_representations(donor_batch.tokens, donor_batch.padding_mask)
        self.assertEqual(len(donor_stages), 3)

        rows = torch.arange(2)
        target_positions = torch.tensor([2, 2])
        operator_patched = model.forward_patched(
            recipient_batch.tokens,
            recipient_batch.padding_mask,
            patch_stage=2,
            patch_positions=target_positions,
            patch_values=donor_stages[2][rows, target_positions],
        )
        torch.testing.assert_close(operator_patched, recipient_logits)

        cls_positions = torch.tensor([0, 0])
        cls_patched = model.forward_patched(
            recipient_batch.tokens,
            recipient_batch.padding_mask,
            patch_stage=2,
            patch_positions=cls_positions,
            patch_values=donor_stages[2][rows, cls_positions],
        )
        torch.testing.assert_close(cls_patched, donor_logits)


if __name__ == "__main__":
    unittest.main()

import unittest

from logic_sheaves.data import CLS_ID, PAD_ID, collate_expressions, make_splits


class DataTests(unittest.TestCase):
    def test_splits_are_disjoint_and_ood_depth_is_exact(self) -> None:
        splits = make_splits(
            train_size=40,
            validation_size=20,
            test_size=20,
            train_depth=2,
            ood_depths=(3, 4),
            seed=3,
        )
        train = {str(expression) for expression in splits.train}
        validation = {str(expression) for expression in splits.validation}
        test = {str(expression) for expression in splits.test_id}
        self.assertFalse(train & validation)
        self.assertFalse(train & test)
        self.assertFalse(validation & test)
        for depth, expressions in splits.test_ood.items():
            self.assertTrue(all(expression.depth == depth for expression in expressions))

    def test_collation_adds_cls_and_padding(self) -> None:
        splits = make_splits(
            train_size=4,
            validation_size=2,
            test_size=2,
            train_depth=2,
            ood_depths=(3,),
            seed=9,
        )
        batch = collate_expressions(splits.train)
        self.assertTrue(batch.tokens[:, 0].eq(CLS_ID).all())
        self.assertTrue(batch.padding_mask.eq(batch.tokens.eq(PAD_ID)).all())
        self.assertEqual(batch.labels.shape, (4,))


if __name__ == "__main__":
    unittest.main()

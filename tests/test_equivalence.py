import unittest

import numpy as np
import torch

from logic_sheaves.diagram_metrics import fit_rewrite_transports, score_equivalence_diagrams
from logic_sheaves.equivalence import make_equivalence_suite, make_rewrite_calibration_pairs
from logic_sheaves.metrics import OrthogonalTransport
from logic_sheaves.model import ModelConfig, TinyLogicTransformer


class EquivalenceDiagramTests(unittest.TestCase):
    def test_all_diagram_families_are_valid_and_balanced(self) -> None:
        diagrams = make_equivalence_suite(4, operand_depth=1, seed=21)
        grouped: dict[str, list] = {}
        for diagram in diagrams:
            diagram.validate()
            grouped.setdefault(diagram.family, []).append(diagram)
            self.assertEqual(len({vertex.value for vertex in diagram.vertices}), 1)
        self.assertEqual(len(grouped), 11)
        self.assertTrue(all(len(items) == 4 for items in grouped.values()))
        self.assertTrue(
            all([item.vertices[0].value for item in items] == [0, 1, 0, 1] for items in grouped.values())
        )
        self.assertEqual(len(grouped["commutativity_cube"][0].loops), 6)
        self.assertEqual(len(grouped["associativity_pentagon"][0].vertices), 5)
        self.assertEqual(len(grouped["nary_and_hexagon"][0].vertices), 6)

    def test_every_diagram_rewrite_has_isolated_calibration_pairs(self) -> None:
        diagrams = make_equivalence_suite(2, operand_depth=1, seed=22)
        pairs = make_rewrite_calibration_pairs(4, operand_depth=1, seed=23)
        diagram_labels = {edge.label for diagram in diagrams for edge in diagram.edges}
        self.assertEqual(diagram_labels, set(pairs))
        for label_pairs in pairs.values():
            self.assertEqual(len(label_pairs), 4)
            self.assertTrue(all(source.value == target.value for source, target in label_pairs))

    def test_general_metrics_cover_loops_and_competing_paths(self) -> None:
        torch.manual_seed(24)
        model = TinyLogicTransformer(
            ModelConfig(d_model=16, n_heads=4, n_layers=1, d_ff=32, max_length=128)
        )
        calibration = make_rewrite_calibration_pairs(20, operand_depth=1, seed=25)
        transports = fit_rewrite_transports(model, calibration, device=torch.device("cpu"))
        diagrams = make_equivalence_suite(4, operand_depth=1, seed=26)
        rows = score_equivalence_diagrams(
            model, diagrams, transports, device=torch.device("cpu")
        )
        self.assertEqual(len(rows), 11)
        for row in rows:
            self.assertTrue(np.isfinite(row["holonomy_error"]))
            self.assertTrue(np.isfinite(row["transport_error"]))
            self.assertTrue(np.isfinite(row["holonomy_rotation_error"]))
            self.assertAlmostEqual(
                row["holonomy_error"],
                row["holonomy_systematic_error"] + row["holonomy_dispersion_error"],
                places=6,
            )
        path_rows = [row for row in rows if row["path_pairs_per_diagram"]]
        self.assertEqual({row["family"] for row in path_rows}, {
            "associativity_pentagon",
            "commutativity_cube",
            "distributivity_diamond",
        })
        self.assertTrue(all(np.isfinite(row["path_agreement_error"]) for row in path_rows))

    def test_identity_connection_has_zero_holonomy_but_nonzero_edge_error(self) -> None:
        torch.manual_seed(27)
        width = 16
        model = TinyLogicTransformer(
            ModelConfig(d_model=width, n_heads=4, n_layers=1, d_ff=32, max_length=128)
        )
        diagrams = make_equivalence_suite(4, operand_depth=1, seed=28)
        labels = {edge.label for diagram in diagrams for edge in diagram.edges}
        identity = OrthogonalTransport(
            source_mean=np.zeros((1, width)),
            target_mean=np.zeros((1, width)),
            rotation=np.eye(width),
        )
        rows = score_equivalence_diagrams(
            model,
            diagrams,
            {label: identity for label in labels},
            device=torch.device("cpu"),
        )
        self.assertTrue(all(abs(row["holonomy_error"]) < 1e-12 for row in rows))
        self.assertTrue(all(abs(row["holonomy_rotation_error"]) < 1e-12 for row in rows))
        self.assertTrue(all(row["transport_error"] > 0.0 for row in rows))


if __name__ == "__main__":
    unittest.main()

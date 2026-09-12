import unittest
from random import Random

import torch

from logic_sheaves.data import AssignedExpression
from logic_sheaves.equivalence import EquivalenceDiagram, commutativity_cube
from logic_sheaves.gauge_experiment import (
    _bitflip_deltas,
    _flip_diagrams,
    atlas_features,
    variable_influence_class,
)
from logic_sheaves.logic import VARIABLES, binary, unary, variable
from logic_sheaves.model import ModelConfig, TinyLogicTransformer


class GaugeExperimentTests(unittest.TestCase):
    def test_variable_influence_classes(self) -> None:
        assignment = (("x0", 0), ("x1", 0), ("x2", 1), ("x3", 1))

        def diagram(expression) -> EquivalenceDiagram:
            return EquivalenceDiagram(
                family="test",
                vertices=(AssignedExpression(expression, assignment),),
                edges=(),
                loops=(),
            )

        self.assertEqual(variable_influence_class(diagram(variable("x0")), "x1"), "absent")
        cancelled = binary("AND", variable("x0"), unary("NOT", variable("x0")))
        self.assertEqual(variable_influence_class(diagram(cancelled), "x0"), "globally_irrelevant")
        conjunction = binary("AND", variable("x0"), variable("x1"))
        self.assertEqual(variable_influence_class(diagram(conjunction), "x0"), "value_preserved")
        self.assertEqual(variable_influence_class(diagram(variable("x0")), "x0"), "value_changed")

    def test_bitflip_delta_matches_on_all_analysis_coordinates(self) -> None:
        common = {
            "checkpoint": "model.pt",
            "architecture_layers": 2,
            "training_condition": "higher_diversity",
            "seed": 0,
            "family": "square",
            "measured_layer": 1,
            "scope": "cls",
            "component": "query",
            "heads": "all",
            "flip_variable": "x0",
            "flip_effect": "value_changed",
            "control": "learned",
            "unit_holonomy_mean": 0.2,
            "edge_fidelity": 0.3,
            "state_return_error": 0.4,
            "shear_mean": 0.1,
            "sigma_min_mean": 0.8,
        }
        rows = [
            {**common, "assignment_condition": "original", "paper_holonomy_mean": 0.4},
            {**common, "assignment_condition": "bitflip_x0", "paper_holonomy_mean": 0.7},
        ]
        delta = _bitflip_deltas(rows)
        self.assertEqual(len(delta), 1)
        self.assertAlmostEqual(delta[0]["delta_paper_holonomy_mean"], 0.3)

    def test_assignment_flip_preserves_topology_and_flips_only_requested_bit(self) -> None:
        diagram = commutativity_cube(Random(4), 1, VARIABLES[:4])
        flipped = _flip_diagrams([diagram], "x2")[0]
        self.assertEqual(flipped.edges, diagram.edges)
        self.assertEqual(flipped.loops, diagram.loops)
        self.assertEqual(
            [vertex.expression for vertex in flipped.vertices],
            [vertex.expression for vertex in diagram.vertices],
        )
        for original, intervention in zip(diagram.vertices, flipped.vertices, strict=True):
            for (name, value), (flipped_name, flipped_value) in zip(
                original.assignment, intervention.assignment, strict=True
            ):
                self.assertEqual(name, flipped_name)
                self.assertEqual(flipped_value, 1 - value if name == "x2" else value)

    def test_feature_extraction_separates_components_and_head_restriction(self) -> None:
        diagram = commutativity_cube(Random(5), 1, VARIABLES[:4])
        examples = list(diagram.vertices[:3])
        model = TinyLogicTransformer(ModelConfig(d_model=16, n_heads=4, n_layers=2, d_ff=32)).eval()
        features = atlas_features(
            model,
            examples,
            device=torch.device("cpu"),
            batch_size=3,
            components=("residual", "query", "coupled_qk"),
            scopes=("cls", "expression_root", "expression_mean"),
            heads=(1, 3),
        )
        self.assertEqual(features[(2, "cls", "residual")].shape, (3, 16))
        self.assertEqual(features[(1, "expression_root", "query")].shape, (3, 8))
        self.assertEqual(features[(1, "expression_mean", "coupled_qk")].shape, (3, 16))


if __name__ == "__main__":
    unittest.main()

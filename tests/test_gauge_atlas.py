import unittest

import numpy as np

from logic_sheaves.gauge_atlas import (
    AtlasChart,
    AtlasEdge,
    bootstrap_transport_stability,
    fit_atlas,
    gauge_transform,
    holonomy_distances,
    proxy_connection_control,
    reparameterize_atlas,
    stable_subgraph,
    topology_shuffled_transport_null,
    typed_fundamental_cycles,
)


def _edge(source: int, target: int, defect: np.ndarray) -> AtlasEdge:
    width = defect.shape[0]
    chart = AtlasChart(np.zeros((1, width)), np.eye(width))
    return AtlasEdge(
        source=source,
        target=target,
        source_chart=chart,
        target_chart=chart,
        ridge_map=defect,
        transport=defect,
        proxy=np.eye(width),
        defect=defect,
        sigma_min=1.0,
        shear=0.0,
        transfer_mismatch=0.0,
        transfer_lower_bound=0.0,
    )


def _rotation(angle: float) -> np.ndarray:
    return np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


class GaugeAtlasTests(unittest.TestCase):
    def test_chart_reparameterization_gives_typed_loop_covariance(self) -> None:
        rng = np.random.default_rng(19)

        def random_orthogonal() -> np.ndarray:
            value, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            return value

        edges = {
            (0, 1): _edge(0, 1, random_orthogonal()),
            (1, 2): _edge(1, 2, random_orthogonal()),
            (0, 2): _edge(0, 2, random_orthogonal()),
        }
        # P=I is consistent with the shared identity chart bases; Q is independent.
        gauges = {vertex: random_orthogonal() for vertex in range(3)}
        transformed = reparameterize_atlas(edges, gauges)
        before = typed_fundamental_cycles(3, edges)[0]
        after = typed_fundamental_cycles(3, transformed)[0]
        base = before.traversals[0][0]
        np.testing.assert_allclose(
            after.transport_holonomy,
            gauges[base] @ before.transport_holonomy @ gauges[base].T,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            after.proxy_holonomy,
            gauges[base] @ before.proxy_holonomy @ gauges[base].T,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            after.relative_holonomy,
            gauges[base] @ before.relative_holonomy @ gauges[base].T,
            atol=1e-12,
        )
        self.assertAlmostEqual(
            before.relative_unit_distance, after.relative_unit_distance, places=12
        )
        # P^-1 Q is a source-fiber endomorphism, not a bifundamental edge map.
        for key, edge in edges.items():
            expected = gauges[edge.source] @ edge.defect @ gauges[edge.source].T
            np.testing.assert_allclose(transformed[key].defect, expected, atol=1e-12)

        via_public_api = gauge_transform(edges, gauges)
        for key in edges:
            np.testing.assert_allclose(
                via_public_api[key].transport, transformed[key].transport, atol=1e-12
            )

    def test_proxy_control_has_zero_relative_loop_holonomy(self) -> None:
        edges = {
            (0, 1): _edge(0, 1, _rotation(0.2)),
            (1, 2): _edge(1, 2, _rotation(-0.1)),
            (0, 2): _edge(0, 2, _rotation(0.4)),
        }
        cycle = typed_fundamental_cycles(3, proxy_connection_control(edges))[0]
        self.assertLess(cycle.relative_unit_distance, 1e-12)

    def test_topology_shuffle_remains_a_typed_connection(self) -> None:
        edges = {
            (0, 1): _edge(0, 1, _rotation(0.2)),
            (1, 2): _edge(1, 2, _rotation(-0.1)),
            (0, 2): _edge(0, 2, _rotation(0.4)),
        }
        shuffled = topology_shuffled_transport_null(3, edges)
        for edge in shuffled.values():
            np.testing.assert_allclose(edge.transport, edge.proxy @ edge.defect, atol=1e-12)

    def test_paper_normalization_can_exceed_one(self) -> None:
        paper, unit = holonomy_distances(-np.eye(2))
        self.assertAlmostEqual(paper, np.sqrt(2.0))
        self.assertAlmostEqual(unit, 1.0)

    def test_fit_and_evaluation_are_disjoint_and_exact_consistent_atlas_is_flat(self) -> None:
        rng = np.random.default_rng(12)
        latent = rng.normal(size=(80, 2))
        rotations = (_rotation(0.0), _rotation(0.25), _rotation(0.7))
        samples = np.stack([latent @ rotation.T for rotation in rotations], axis=1)
        result = fit_atlas(
            samples,
            ((0, 1), (1, 2), (0, 2)),
            np.arange(50),
            np.arange(50, 80),
            dimension=2,
            ridge=1e-8,
        )
        self.assertEqual(result.cycle_coverage, 1)
        self.assertLess(result.edge_fidelity, 1e-12)
        self.assertLess(result.state_return_error, 1e-12)
        self.assertLess(result.cycles[0].paper_distance, 1e-10)

    def test_fit_rejects_overlap_between_fit_and_evaluation(self) -> None:
        samples = np.zeros((6, 3, 2))
        with self.assertRaisesRegex(ValueError, "disjoint"):
            fit_atlas(
                samples,
                ((0, 1), (1, 2), (0, 2)),
                np.asarray([0, 1, 2]),
                np.asarray([2, 3, 4]),
                dimension=1,
                ridge=0.1,
            )

    def test_bootstrap_stability_and_multicriterion_filter(self) -> None:
        rng = np.random.default_rng(22)
        source = rng.normal(size=(96, 3))
        target = source @ _rotation(0.2).T if source.shape[1] == 2 else source.copy()
        chart = AtlasChart(np.zeros((1, 3)), np.eye(3))
        edge = _edge(0, 1, np.eye(3))
        edge = AtlasEdge(
            **{
                **edge.__dict__,
                "source_chart": chart,
                "target_chart": chart,
                "condition_ratio": 1.0,
            }
        )
        stability = bootstrap_transport_stability(
            edge, source, target, ridge=1e-8, samples=8, seed=5
        )
        self.assertLess(stability, 1e-10)
        edge = AtlasEdge(**{**edge.__dict__, "bootstrap_stability": stability})
        retained = stable_subgraph(
            {(0, 1): edge},
            min_sigma=0.5,
            min_condition_ratio=0.5,
            max_bootstrap_stability=0.1,
        )
        self.assertEqual(set(retained), {(0, 1)})


if __name__ == "__main__":
    unittest.main()

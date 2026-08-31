import unittest

import numpy as np

from logic_sheaves.metrics import fit_orthogonal_transport, linear_cka, ridge_probe


class MetricTests(unittest.TestCase):
    def test_procrustes_recovers_rigid_transport(self) -> None:
        rng = np.random.default_rng(4)
        source = rng.normal(size=(200, 8))
        q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
        offset = rng.normal(size=(1, 8))
        target = source @ q + offset
        transport = fit_orthogonal_transport(source, target)
        np.testing.assert_allclose(transport.apply(source), target, atol=1e-10)

    def test_linear_cka_is_one_for_orthogonal_change_of_basis(self) -> None:
        rng = np.random.default_rng(5)
        x = rng.normal(size=(100, 12))
        q, _ = np.linalg.qr(rng.normal(size=(12, 12)))
        self.assertAlmostEqual(linear_cka(x, x @ q), 1.0, places=10)

    def test_ridge_probe_finds_linear_signal(self) -> None:
        rng = np.random.default_rng(6)
        x = rng.normal(size=(300, 5))
        y = (x[:, 0] - 0.5 * x[:, 1] > 0).astype(np.int64)
        result = ridge_probe(x[:200], y[:200], x[200:], y[200:])
        self.assertGreater(result["test_balanced_accuracy"], 0.95)


if __name__ == "__main__":
    unittest.main()

import csv
import tempfile
import unittest
from pathlib import Path

from logic_sheaves.circuit_selection import nominate_circuits


class CircuitSelectionTests(unittest.TestCase):
    def test_nomination_requires_discovery_and_confirmation_specificity(self) -> None:
        fields = (
            "architecture_layers",
            "condition",
            "subset",
            "operator",
            "head",
            "patch_layer",
            "site",
            "component",
            "donor_type",
            "seed",
            "effect_fraction",
        )
        rows = []
        for head, confirmation in ((0, 0.03), (1, -0.02)):
            for seed in (0, 1, 2):
                specificity = 0.02 if seed < 2 else confirmation
                for donor, value in (
                    ("counterfactual", 0.1 + specificity),
                    ("counterfactual_shuffled", 0.1),
                ):
                    rows.append(
                        {
                            "architecture_layers": 6,
                            "condition": "higher_diversity",
                            "subset": "eligible",
                            "operator": "ALL",
                            "head": head,
                            "patch_layer": 6,
                            "site": "operator",
                            "component": "key",
                            "donor_type": donor,
                            "seed": seed,
                            "effect_fraction": value,
                        }
                    )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "patches.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            nominations = nominate_circuits(path)
        confirmed = [item for item in nominations if item.status == "confirmed"]
        rejected = [item for item in nominations if item.status == "rejected_reference"]
        self.assertEqual(confirmed[0].heads, (0,))
        self.assertEqual(rejected[0].heads, (1,))


if __name__ == "__main__":
    unittest.main()

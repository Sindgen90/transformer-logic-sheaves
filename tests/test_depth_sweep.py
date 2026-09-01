import tempfile
import unittest
from pathlib import Path

from logic_sheaves.depth_sweep import create_run_directory


class DepthSweepTests(unittest.TestCase):
    def test_named_runs_never_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            created = create_run_directory(root, "trial")
            self.assertTrue(created.is_dir())
            with self.assertRaises(FileExistsError):
                create_run_directory(root, "trial")


if __name__ == "__main__":
    unittest.main()

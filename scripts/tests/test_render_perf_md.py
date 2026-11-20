import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


RENDERER = Path(__file__).resolve().parents[1] / "render_perf_md.py"


class ReportVersionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def report(self, name, **fields):
        path = self.directory / name
        path.write_text(json.dumps({"measurements": {"memory_mb": 12}, **fields}))
        return path

    def render(self, *paths):
        return subprocess.run([sys.executable, RENDERER, *paths], capture_output=True, text=True)

    def test_known_and_unversioned_inputs(self):
        known = self.report("current.json", schema_version=1)
        legacy = self.report("legacy.json")
        for paths in [(known,), (legacy,), (known, legacy)]:
            with self.subTest(paths=paths):
                result = self.render(*paths)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("| Memory (MB) | 12.0 |", result.stdout)
                if len(paths) == 2:
                    self.assertIn("## Format Comparison", result.stdout)

    def test_unsupported_primary_version(self):
        for version in [2, True, "1", None]:
            with self.subTest(version=version):
                result = self.render(self.report("future.json", schema_version=version))
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
                self.assertIn("unsupported schema_version", result.stderr)

    def test_unsupported_comparison_keeps_primary_report(self):
        known = self.report("current.json", schema_version=1)
        future = self.report("future.json", schema_version=2)
        result = self.render(known, future)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("| Memory (MB) | 12.0 |", result.stdout)
        self.assertNotIn("## Format Comparison", result.stdout)
        self.assertIn("unsupported schema_version", result.stderr)

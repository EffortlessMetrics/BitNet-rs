"""Exercise the actual CI Core selector with broad and leaf-server plans."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


class ServerRoutingTests(unittest.TestCase):
    def run_plan(self, changed, selected, broad):
        root = Path(__file__).resolve().parents[1]
        workflow = (root / ".github/workflows/ci-core.yml").read_text(encoding="utf-8")
        marker = '          python - <<\'PY\' >> "$GITHUB_OUTPUT"\n'
        source = textwrap.dedent(workflow.split(marker, 1)[1].split("          PY\n", 1)[0])
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary) / "target/ci/ci-plan.json"
            fixture.parent.mkdir(parents=True)
            fixture.write_text(json.dumps({"packages": {"changed": changed, "selected": selected, "broad_sweep_required": broad}}), encoding="utf-8")
            result = subprocess.run([sys.executable, "-c", source], cwd=temporary, env={**os.environ, "EVENT_NAME": "pull_request"}, capture_output=True, text=True, check=True)
        return dict(line.split("=", 1) for line in result.stdout.splitlines())

    def test_broad_sweep_keeps_changed_server_even_when_selected_core_omits_it(self):
        result = self.run_plan(["bitnet-server", "bitnet-kernels"], ["bitnet-kernels"], True)
        self.assertEqual(result["run_server_on_broad_sweep"], "true")

    def test_leaf_server_uses_existing_selected_package_route_without_duplication(self):
        result = self.run_plan(["bitnet-server"], ["bitnet-server"], False)
        self.assertEqual(result["cpu_package_args"], "-p bitnet-server")
        self.assertEqual(result["broad_sweep_required"], "false")
        self.assertEqual(result["run_server_on_broad_sweep"], "false")

    def test_unrelated_broad_sweep_does_not_add_server_work(self):
        result = self.run_plan(["bitnet-kernels"], ["bitnet-kernels"], True)
        self.assertEqual(result["run_server_on_broad_sweep"], "false")

    def test_unsupported_package_fallback_does_not_drop_selected_server(self):
        result = self.run_plan(["bitnet-server"], ["bitnet-server", "unsupported-package"], False)
        self.assertEqual(result["broad_sweep_required"], "true")
        self.assertEqual(result["run_server_on_broad_sweep"], "true")


if __name__ == "__main__":
    unittest.main()

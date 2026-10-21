"""Exercise the actual PR Gate lane mapping and its fail-closed default."""

import os
from pathlib import Path
import re
import subprocess
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]


class GateRoutingTests(unittest.TestCase):
    def run_mapping(self, lanes):
        workflow = (ROOT / ".github/workflows/pr-gate.yml").read_text(encoding="utf-8")
        start = workflow.index("          required_jobs=()\n")
        end = workflow.index("          if [ ${#required_jobs[@]} -gt 0 ]; then\n", start)
        script = textwrap.dedent(workflow[start:end])
        plan_input = "jq -r '.selected_lanes[] | select(.blocking == true) | .id' ci-plan.json"
        self.assertEqual(script.count(plan_input), 1)
        script = script.replace(plan_input, "printf '%s\\n' \"$TEST_LANES\"")
        script += "printf '%s\\n' \"${required_jobs[@]}\"\n"
        return subprocess.run(
            ["bash", "--noprofile", "--norc", "-c", script],
            env={**os.environ, "TEST_LANES": "\n".join(lanes)},
            capture_output=True,
            text=True,
            timeout=10,
        )

    def test_rustdoc_maps_to_the_actual_core_documentation_check(self):
        workflow = (ROOT / ".github/workflows/ci-core.yml").read_text(encoding="utf-8")
        job_name = re.search(r"^  docs:\n    name: (.+)$", workflow, re.M)
        self.assertIsNotNone(job_name)
        result = self.run_mapping(["rustdoc-workspace"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), [job_name.group(1)])

    def test_existing_core_and_rustdoc_require_both_checks(self):
        result = self.run_mapping(["ci-core-build-test", "rustdoc-workspace"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["CI Core Success", "Documentation"])

    def test_unknown_blocking_lane_remains_a_hard_failure(self):
        result = self.run_mapping(["rustdoc-workspace", "unmapped-test-lane"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("PR Gate has no check mapping for selected blocking lane(s): unmapped-test-lane", result.stdout)


if __name__ == "__main__":
    unittest.main()

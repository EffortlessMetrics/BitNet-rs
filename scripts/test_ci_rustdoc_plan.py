import subprocess
import sys
import unittest
from pathlib import Path

from ci_rustdoc_plan import rustdoc_required


class RustdocRoutingTests(unittest.TestCase):
    def test_docs_and_public_api_require_workspace_proof(self):
        for path in ("docs/guide.md", "README.md", "crates/bitnet-models/src/lib.rs", "crates/bitnet-server/src/api/handlers.rs", "Cargo.lock", "crates/bitnet-common/Cargo.toml", "src/lib.rs", ".cargo/config.toml"):
            with self.subTest(path=path):
                self.assertTrue(rustdoc_required("pull_request", [path]))

    def test_workspace_manifests_outside_crates_require_proof(self):
        for member in ('crossval', 'tests', 'tests-new', 'xtask', 'tools/bitnet-task', 'xtask-build-helper', 'fuzz', 'tools/migrate-gen-config'):
            with self.subTest(member=member):
                self.assertTrue(rustdoc_required("pull_request", [member + "/Cargo.toml"]))

    def test_nonstandard_workspace_library_and_api_roots_require_proof(self):
        for path in ('crossval/src/lib.rs', 'xtask/src/lib.rs', 'xtask-build-helper/src/lib.rs', 'tests/lib.rs', 'tests-new/lib.rs', 'tools/custom/src/api/types.rs'):
            with self.subTest(path=path):
                self.assertTrue(rustdoc_required("pull_request", [path]))

    def test_leaf_rust_and_tracking_do_not_select_rustdoc(self):
        for paths in (["crates/bitnet-quantization/src/qk256.rs"], ["docs/tracking/progress.md"], [".codex/campaigns/state.md"], ["policy/clippy-debt.toml"]):
            with self.subTest(paths=paths):
                self.assertFalse(rustdoc_required("pull_request", paths))

    def test_mixed_docs_and_leaf_rust_still_selects(self):
        self.assertTrue(rustdoc_required("pull_request", ["crates/x/src/leaf.rs", "docs/api.md"]))

    def test_main_manual_unknown_and_empty_inputs_fail_closed(self):
        for event in ("push", "workflow_dispatch", "merge_group", "unknown"):
            self.assertTrue(rustdoc_required(event, ["docs/tracking/progress.md"]))
        self.assertTrue(rustdoc_required("pull_request", []))
        self.assertTrue(rustdoc_required("pull_request", ["crates/x/src/leaf.rs"], ["full-ci"]))

    def test_cli_emits_boolean_and_rejects_bad_labels(self):
        command = [sys.executable, str(Path(__file__).with_name("ci_rustdoc_plan.py")), "--event", "pull_request", "--changed-files", "crates/x/src/leaf.rs"]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout, "false\n")
        rejected = subprocess.run(command + ["--labels-json", "{}"], capture_output=True, text=True)
        self.assertNotEqual(rejected.returncode, 0)


if __name__ == "__main__":
    unittest.main()

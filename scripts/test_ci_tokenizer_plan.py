import json
from pathlib import Path
import subprocess
import sys
import tomllib
import unittest

from ci_tokenizer_plan import ROOT, SELF_PATHS, tokenizer_required


class TokenizerRoutingTests(unittest.TestCase):
    def test_every_policy_tokenizer_path_selects_real_lane(self):
        with (ROOT / 'policy/ci-risk-packs.toml').open('rb') as source:
            patterns = tomllib.load(source)['risk_pack']['tokenizer']['paths']
        for pattern in patterns:
            path = pattern.replace('**', 'nested/fixture.rs').replace('*', 'example')
            with self.subTest(path=path):
                self.assertTrue(tokenizer_required('pull_request', [path]))

    def test_changed_download_and_fixture_select_without_label(self):
        for path in ('crates/bitnet-tokenizers/src/download.rs', 'tests/fixtures/tokenizers/valid_tokenizer_a.json'):
            self.assertTrue(tokenizer_required('pull_request', [path]))

    def test_unrelated_paths_do_not_admit_extra_work(self):
        for path in ('crates/bitnet-server/src/lib.rs', 'docs/tracking/status.md', 'crates/bitnet-tokenizers-other/src/lib.rs'):
            self.assertFalse(tokenizer_required('pull_request', [path]))

    def test_self_changes_and_existing_opt_ins_select(self):
        for path in SELF_PATHS:
            self.assertTrue(tokenizer_required('pull_request', [path]))
        for label in ('tokenizer', 'full-ci'):
            self.assertTrue(tokenizer_required('pull_request', ['README.md'], [label]))

    def test_main_unknown_and_empty_diff_fail_closed(self):
        for event in ('push', 'workflow_dispatch', 'merge_group', 'unknown'):
            self.assertTrue(tokenizer_required(event, ['README.md']))
        self.assertTrue(tokenizer_required('pull_request', []))

    def test_cli_boolean_and_malformed_label_rejection(self):
        command = [sys.executable, str(Path(__file__).with_name('ci_tokenizer_plan.py')), '--event', 'pull_request', '--changed-files', 'crates/bitnet-tokenizers/src/download.rs']
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        self.assertEqual(result.stdout, 'true\n')
        for invalid in ({}, [1], 'tokenizer'):
            rejected = subprocess.run(command + ['--labels-json', json.dumps(invalid)], capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)

    def test_workflow_requires_successful_offline_tests(self):
        workflow = (ROOT / '.github/workflows/compatibility.yml').read_text(encoding='utf-8')
        job = workflow.split('  tokenizer-fixtures:\n', 1)[1].split('  gguf-compat:\n', 1)[0]
        self.assertIn("if: needs.route-msrv.outputs.tokenizer_selected == 'true'", job)
        self.assertIn('--lib --test json_loader --test rust_gguf_tokenizer_api', job)
        self.assertIn('cargo test --locked -p bitnet-tokenizers --no-default-features --features cpu', job)
        for invalid in ('continue-on-error', '|| true', '--features fixtures', 'tokenizer_compat_test', 'pip install', 'download-model', '--ignored', 'BITNET_SKIP_SLOW_TESTS'):
            self.assertNotIn(invalid, job)
        self.assertIn('tokenizer_selected: ${{ steps.tokenizer-route.outputs.selected }}', workflow)
        self.assertIn('changed="$(git diff --name-only "$BASE_SHA"...HEAD)"', workflow)
        self.assertIn('run: python3 -m unittest discover -s scripts -p \'test_ci_tokenizer_plan.py\'', workflow)

    def test_every_policy_parser_caller_selects_python_311_first(self):
        for name, test_step in (('compatibility.yml', 'Test tokenizer routing contract'), ('ci-core.yml', 'Test core proof routing')):
            workflow = (ROOT / '.github/workflows' / name).read_text(encoding='utf-8')
            setup, tests = workflow.index('      - name: Select Python for policy routing'), workflow.index('      - name: ' + test_step)
            self.assertLess(setup, tests)
            self.assertIn("python-version: '3.11'", workflow[setup:tests])


if __name__ == '__main__':
    unittest.main()

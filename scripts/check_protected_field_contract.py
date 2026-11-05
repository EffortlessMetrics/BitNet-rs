#!/usr/bin/env python3
"""Prove the two owner accesses and forbidden accesses using pinned Clippy."""

import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import tomllib


ROOT = Path(__file__).resolve().parents[1]
OWNER = "crates/bitnet-inference/src/generation_state.rs"
FIELD = "bitnet_inference::generation_state::GenerationTracker::state"
LINT = "clippy::disallowed_fields"
EXPECTED = {
    "clippy-0001": ("GenerationTracker::state", "self.state"),
    "clippy-0002": ("GenerationTracker::set_state", "self.state = state"),
}


def validate_source_contract(root=ROOT):
    source = (root / OWNER).read_text(encoding="utf-8")
    config = tomllib.loads((root / "clippy.toml").read_text(encoding="utf-8"))
    fields = [entry for entry in config["disallowed-fields"] if entry["path"] == FIELD]
    assert len(fields) == 1 and fields[0]["reason"], "missing/duplicate protected field"
    cargo = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))
    assert cargo["workspace"]["lints"]["clippy"]["disallowed_fields"] == "deny"
    receipts = tomllib.loads((root / "policy/clippy-exceptions.toml").read_text(encoding="utf-8"))["exception"]
    for identity, (container, expression) in EXPECTED.items():
        entries = [entry for entry in receipts if entry["id"] == identity]
        assert len(entries) == 1, f"missing/duplicate {identity}"
        entry = entries[0]
        assert entry["lint"] == LINT and entry["path"] == OWNER
        assert entry["classification"] == "encapsulation_owner"
        assert entry["owner"] and entry["reason"] and entry["expires"]
        assert entry["selector"] == {"kind": "fn", "container": container, "text_contains": expression}
    scoped = [entry for entry in receipts if entry["lint"] == LINT]
    assert len(scoped) == 2, "only two protected-field exceptions are authorized"
    getter = '''    #[expect(clippy::disallowed_fields, reason = "policy:clippy-0001")]
    pub fn state(&self) -> GenState {
        self.state
    }'''
    setter = '''    #[expect(clippy::disallowed_fields, reason = "policy:clippy-0002")]
    fn set_state(&mut self, state: GenState) {
        self.state = state;
    }'''
    assert source.count(getter) == 1 and source.count(setter) == 1, "owner sites changed"
    assert "#![deny(unfulfilled_lint_expectations)]" in source
    # The receipt checker checks IDs, not selectors. Independently constrain all
    # tracked Rust attributes for this lint to the two exact owner methods.
    tracked = subprocess.check_output(["git", "ls-files", "-z", "--", "*.rs"], cwd=root).decode().split("\0")
    sites = []
    attribute = re.compile(r"#!?\s*\[\s*(?:expect|allow)\s*\([^]]*clippy::disallowed_fields[^]]*\]", re.S)
    for path in filter(None, tracked):
        for match in attribute.finditer((root / path).read_text(encoding="utf-8")):
            sites.append((path, match.group()))
    assert len(sites) == 2 and all(path == OWNER for path, _ in sites), sites
    assert all(text.startswith("#[expect(") for _, text in sites), sites
    return source, getter, setter


def compile_probe(directory, source, *, test=False):
    directory.mkdir()
    (directory / "generation_state.rs").write_text(source, encoding="utf-8")
    crate = directory / "lib.rs"
    crate.write_text("pub mod generation_state;\n", encoding="utf-8")
    output = directory / ("lifecycle.exe" if os.name == "nt" else "lifecycle")
    command = ["clippy-driver", str(crate), "--crate-name", "bitnet_inference", "--edition=2024",
               "--error-format=json", "-D", LINT, "-D", "unfulfilled_lint_expectations"]
    command += ["--test", "-o", str(output)] if test else ["--crate-type=lib", "--emit=metadata", "-o", str(output.with_suffix(".rmeta"))]
    result = subprocess.run(command, env={**os.environ, "CLIPPY_CONF_DIR": str(ROOT)}, capture_output=True, text=True, timeout=60)
    diagnostics = []
    for line in result.stderr.splitlines():
        if line.startswith("{"):
            diagnostics.append(json.loads(line))
    return result, diagnostics, output


def expect_rejection(directory, source, code):
    result, diagnostics, _ = compile_probe(directory, source)
    errors = [item for item in diagnostics if item.get("level") == "error"]
    coded = [item["code"]["code"] for item in errors if item.get("code")]
    assert result.returncode != 0 and coded == [code], (directory.name, result.returncode, result.stderr)
    assert all(item.get("code") or item.get("message", "").startswith("aborting due to") for item in errors), result.stderr
    print(f"PASS {directory.name}: rejected by {code}", flush=True)


def main():
    source, getter, setter = validate_source_contract()
    pinned = tomllib.loads((ROOT / "rust-toolchain.toml").read_text(encoding="utf-8"))["toolchain"]["channel"]
    version = subprocess.run(["clippy-driver", "-Vv"], capture_output=True, text=True, check=True, timeout=30).stdout
    assert f"release: {pinned}\n" in version, version
    positive = '''
#[cfg(test)]
mod field_contract_positive {
    use super::{GenState, GenerationTracker, StopReason};
    #[test]
    fn methods_and_debug_use_the_real_owner() {
        let mut tracker = GenerationTracker::new(1);
        assert_eq!(tracker.state(), GenState::Idle);
        tracker.start_prefill(2);
        tracker.start_generation();
        assert_eq!(tracker.on_token(3), GenState::Stopped(StopReason::MaxTokens));
        assert!(format!("{tracker:?}").contains("Stopped"));
        tracker.cancel();
        assert_eq!(tracker.state(), GenState::Stopped(StopReason::Cancelled));
    }
}
'''
    probes = {
        "direct-read": "pub fn probe(value: &super::GenerationTracker) -> super::GenState { value.state }",
        "direct-write": "pub fn probe(value: &mut super::GenerationTracker) { value.state = super::GenState::Failed; }",
        "destructure": "pub fn probe(value: &super::GenerationTracker) -> super::GenState { let super::GenerationTracker { state, .. } = value; *state }",
    }
    with tempfile.TemporaryDirectory(prefix="bitnet-field-contract-") as temporary:
        root = Path(temporary)
        result, _, executable = compile_probe(root / "positive", source + positive, test=True)
        assert result.returncode == 0, result.stderr
        subprocess.run([str(executable), "--nocapture"], check=True, timeout=30)
        print("PASS positive: real owner, accessor-based Debug and lifecycle tests", flush=True)
        for name, probe in probes.items():
            expect_rejection(root / name, source + "\nmod field_contract_negative {\n" + probe + "\n}\n", LINT)
        stale_getter = source.replace(getter, getter.replace("        self.state\n", "        GenState::Idle\n"))
        stale_setter = source.replace(setter, setter.replace("        self.state = state;", "        let _ = state;"))
        expect_rejection(root / "stale-getter-expectation", stale_getter, "unfulfilled_lint_expectations")
        expect_rejection(root / "stale-setter-expectation", stale_setter, "unfulfilled_lint_expectations")
    print("PASS protected-field contract: two owner expectations, three forbidden accesses, two stale expectations", flush=True)


if __name__ == "__main__":
    main()

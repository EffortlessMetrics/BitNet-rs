//! Tests for the inspect --ln-stats command
//!
//! This test suite validates the LayerNorm gamma diagnostics functionality.

#[cfg(feature = "full-cli")]
use assert_cmd::Command;
#[cfg(feature = "full-cli")]
use predicates::prelude::*;

#[cfg(feature = "full-cli")]
#[test]
fn inspect_help_works() {
    Command::cargo_bin("bitnet")
        .unwrap()
        .args(["inspect", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--ln-stats"));
}

#[cfg(feature = "full-cli")]
#[test]
fn inspect_requires_model_path() {
    Command::cargo_bin("bitnet")
        .unwrap()
        .args(["inspect", "--ln-stats"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("required arguments were not provided"));
}

#[cfg(feature = "full-cli")]
#[test]
fn inspect_fails_on_missing_file() {
    Command::cargo_bin("bitnet")
        .unwrap()
        .args(["inspect", "--ln-stats", "/nonexistent/model.gguf"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Failed to open model"));
}

#[cfg(feature = "full-cli")]
#[test]
fn inspect_requires_inspection_mode() {
    // When no inspection mode is specified, should error
    Command::cargo_bin("bitnet")
        .unwrap()
        .args(["inspect", "/some/path.gguf"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("No inspection mode specified"));
}

// Note: Full integration test with a real GGUF model would require:
// 1. A test GGUF file with known LayerNorm gamma stats
// 2. Expected output validation (SHA256, RMS values, status)
// 3. Testing both strict mode and non-strict mode behavior
//
// Such tests should be added in the integration test suite once we have
// appropriate test fixtures.

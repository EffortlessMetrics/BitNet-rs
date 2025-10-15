//! Integration tests for `xtask verify-receipt` command
//!
//! Tests the CLI interface for receipt verification, ensuring proper
//! validation of compute_path, kernels, and optional GPU kernel requirements.

use assert_cmd::Command;
use predicates::prelude::*;
use std::{env, fs, path::PathBuf};

/// Helper to get workspace root
fn workspace_root() -> PathBuf {
    if let Ok(dir) = env::var("CARGO_WORKSPACE_DIR") {
        return PathBuf::from(dir);
    }
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        if path.join(".git").exists() {
            return path;
        }
        let cargo_toml = path.join("Cargo.toml");
        if cargo_toml.exists()
            && let Ok(contents) = fs::read_to_string(&cargo_toml)
            && contents.contains("[workspace]")
        {
            return path;
        }
        if !path.pop() {
            panic!("Could not find workspace root");
        }
    }
}

/// Helper to get fixture path
fn fixture_path(name: &str) -> PathBuf {
    workspace_root().join("xtask/tests/fixtures/receipts").join(format!("{}.json", name))
}

#[test]
fn test_verify_receipt_valid() {
    let mut cmd = Command::cargo_bin("xtask").unwrap();
    cmd.args(["verify-receipt", "--path", fixture_path("valid_receipt").to_str().unwrap()]);

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Receipt verification passed"))
        .stdout(predicate::str::contains("Schema: 1.0.0"))
        .stdout(predicate::str::contains("Compute path: real"));
}

#[test]
fn test_verify_receipt_invalid_compute_path() {
    let mut cmd = Command::cargo_bin("xtask").unwrap();
    cmd.args(["verify-receipt", "--path", fixture_path("invalid_compute_path").to_str().unwrap()]);

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("compute_path must be 'real'"))
        .stderr(predicate::str::contains("mock"));
}

#[test]
fn test_verify_receipt_missing_kernels() {
    let mut cmd = Command::cargo_bin("xtask").unwrap();
    cmd.args(["verify-receipt", "--path", fixture_path("missing_kernels").to_str().unwrap()]);

    cmd.assert().failure().stderr(predicate::str::contains("empty kernels[]"));
}

#[test]
fn test_verify_receipt_valid_gpu() {
    let mut cmd = Command::cargo_bin("xtask").unwrap();
    cmd.args([
        "verify-receipt",
        "--path",
        fixture_path("valid_gpu_receipt").to_str().unwrap(),
        "--require-gpu-kernels",
    ]);

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Receipt verification passed"))
        .stdout(predicate::str::contains("Backend: cuda"));
}

#[test]
fn test_verify_receipt_invalid_gpu() {
    let mut cmd = Command::cargo_bin("xtask").unwrap();
    cmd.args([
        "verify-receipt",
        "--path",
        fixture_path("invalid_gpu_receipt").to_str().unwrap(),
        "--require-gpu-kernels",
    ]);

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("no GPU kernels found"))
        .stderr(predicate::str::contains("silent CPU fallback"));
}

#[test]
fn test_verify_receipt_missing_file() {
    let mut cmd = Command::cargo_bin("xtask").unwrap();
    cmd.args(["verify-receipt", "--path", "nonexistent/receipt.json"]);

    cmd.assert().failure().stderr(predicate::str::contains("Failed to read receipt"));
}

#[test]
fn test_verify_receipt_default_path() {
    // Test that default path is ci/inference.json
    // If file exists with mock kernels, expect CPU validation failure
    let mut cmd = Command::cargo_bin("xtask").unwrap();
    cmd.arg("verify-receipt");

    // Should fail either because file doesn't exist OR because it contains mock kernels
    cmd.assert().failure().stderr(
        predicate::str::contains("ci/inference.json")
            .or(predicate::str::contains("no quantized kernels found")),
    );
}

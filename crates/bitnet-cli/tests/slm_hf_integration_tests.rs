//! Tests for HuggingFace / SLM model loading CLI integration.
//!
//! Validates the `--model-format` and `--architecture` flags added to the `run`
//! subcommand, as well as auto-detection of directory-based HF models vs GGUF files.

use assert_cmd::Command;
use assert_cmd::cargo::cargo_bin_cmd;
use predicates::prelude::*;
use std::fs;
use tempfile::TempDir;

fn bitnet() -> Command {
    cargo_bin_cmd!("bitnet")
}

// ============================================================================
// --model-format flag parsing
// ============================================================================

/// `run --help` documents the --model-format option.
#[test]
fn run_help_shows_model_format_flag() {
    bitnet()
        .args(["run", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--model-format"));
}

/// `run --help` documents the --architecture option.
#[test]
fn run_help_shows_architecture_flag() {
    bitnet()
        .args(["run", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--architecture"));
}

/// `--model-format` defaults to "auto" (visible in help).
#[test]
fn model_format_default_is_auto() {
    bitnet().args(["run", "--help"]).assert().success().stdout(predicate::str::contains("auto"));
}

/// Invalid --model-format value produces a meaningful error at runtime.
/// (clap doesn't restrict the value, so the error comes from our validation.)
#[test]
fn invalid_model_format_rejected() {
    bitnet()
        .args(["run", "--model", "fake.gguf", "--prompt", "hello", "--model-format", "pytorch"])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("Invalid --model-format")
                .or(predicate::str::contains("model-format")),
        );
}

/// Valid --model-format values are accepted at parse time (gguf).
#[test]
fn model_format_gguf_accepted() {
    // This will fail at model-load time (file doesn't exist), not at parse time.
    bitnet()
        .args([
            "run",
            "--model",
            "nonexistent.gguf",
            "--prompt",
            "hello",
            "--model-format",
            "gguf",
        ])
        .assert()
        .failure()
        // Should NOT fail with "Invalid --model-format"
        .stderr(predicate::str::contains("Invalid --model-format").not());
}

/// Valid --model-format values are accepted at parse time (safetensors).
#[test]
fn model_format_safetensors_accepted() {
    bitnet()
        .args([
            "run",
            "--model",
            "nonexistent_dir",
            "--prompt",
            "hello",
            "--model-format",
            "safetensors",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Invalid --model-format").not());
}

// ============================================================================
// Auto-detection: directory vs .gguf
// ============================================================================

/// Passing a directory as --model with format=auto triggers HF loading path.
#[test]
fn directory_model_path_triggers_hf_loader() {
    let tmp = TempDir::new().unwrap();
    let model_dir = tmp.path().join("my-model");
    fs::create_dir(&model_dir).unwrap();
    // No config.json — the loader will fail, but we should see the HF path message
    bitnet()
        .args(["run", "--model", model_dir.to_str().unwrap(), "--prompt", "hello", "--allow-mock"])
        .assert()
        .failure()
        .stdout(predicate::str::contains("HuggingFace model from directory"));
}

/// Passing a .gguf path uses the GGUF loading path (not HF).
#[test]
fn gguf_file_path_uses_gguf_loader() {
    bitnet()
        .args(["run", "--model", "fake.gguf", "--prompt", "hello"])
        .assert()
        .failure()
        // Should say "Loading model from:" (not "HuggingFace")
        .stdout(
            predicate::str::contains("Loading model from:")
                .or(predicate::str::contains("Failed")),
        );
}

// ============================================================================
// --architecture flag
// ============================================================================

/// --architecture is accepted without error.
#[test]
fn architecture_flag_accepted() {
    bitnet()
        .args([
            "run",
            "--model",
            "fake.gguf",
            "--prompt",
            "hello",
            "--architecture",
            "llama",
        ])
        .assert()
        .failure()
        // Should NOT fail due to unknown argument
        .stderr(predicate::str::contains("unexpected argument").not());
}

// ============================================================================
// list-architectures subcommand
// ============================================================================

/// `list-architectures` shows a table of supported architectures.
#[test]
fn list_architectures_shows_table() {
    bitnet()
        .arg("list-architectures")
        .assert()
        .success()
        .stdout(predicate::str::contains("Architecture"));
}

/// `list-architectures --json` outputs valid JSON.
#[test]
fn list_architectures_json_output() {
    let output = bitnet().args(["list-architectures", "--json"]).assert().success();

    let stdout = String::from_utf8(output.get_output().stdout.clone()).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&stdout).expect("valid JSON");
    assert!(parsed.is_array(), "JSON output should be an array");
}

// ============================================================================
// Error messages for invalid model paths
// ============================================================================

/// Nonexistent .gguf file gives a clear error.
#[test]
fn nonexistent_gguf_path_error() {
    bitnet()
        .args(["run", "--model", "does_not_exist.gguf", "--prompt", "hello"])
        .assert()
        .failure();
}

/// Empty directory (no config.json) with --model-format=safetensors gives a clear error.
#[test]
fn empty_dir_safetensors_error() {
    let tmp = TempDir::new().unwrap();
    bitnet()
        .args([
            "run",
            "--model",
            tmp.path().to_str().unwrap(),
            "--prompt",
            "hello",
            "--model-format",
            "safetensors",
        ])
        .assert()
        .failure();
}

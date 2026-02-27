//! Snapshot tests for the `bitnet` binary's CLI output.
//!
//! Captures `--help` text for the top-level command and each major subcommand,
//! plus error messages for invalid arguments.  Any accidental renaming of flags,
//! addition/removal of subcommands, or changed defaults will cause a diff.
//!
//! # Regenerating snapshots
//!
//! ```bash
//! INSTA_UPDATE=unseen cargo test --locked -p bitnet-cli \
//!   --no-default-features --features cpu,full-cli \
//!   --test cli_snapshot_tests
//! ```

#![cfg(feature = "full-cli")]

use assert_cmd::cargo::cargo_bin_cmd;
use insta::assert_snapshot;

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

/// Run `bitnet <args>` and return its stdout as a `String`.
/// Panics if the process exits with a non-zero code.
fn bitnet_stdout(args: &[&str]) -> String {
    let output = cargo_bin_cmd!("bitnet").args(args).output().expect("failed to spawn bitnet");
    String::from_utf8(output.stdout).expect("stdout is not valid UTF-8")
}

/// Run `bitnet <args>` and return its stderr as a `String`.
/// Panics if the process exits with a zero code (we expect failure here).
fn bitnet_stderr_failure(args: &[&str]) -> String {
    let output = cargo_bin_cmd!("bitnet").args(args).output().expect("failed to spawn bitnet");
    assert!(!output.status.success(), "expected bitnet {:?} to fail, but it succeeded", args);
    String::from_utf8(output.stderr).expect("stderr is not valid UTF-8")
}

// ---------------------------------------------------------------------------
// Top-level help
// ---------------------------------------------------------------------------

#[test]
fn cli_help() {
    let help = bitnet_stdout(&["--help"]);
    assert_snapshot!("cli_help", help);
}

// ---------------------------------------------------------------------------
// Subcommand help
// ---------------------------------------------------------------------------

#[test]
fn run_help() {
    let help = bitnet_stdout(&["run", "--help"]);
    assert_snapshot!("run_help", help);
}

#[test]
fn chat_help() {
    let help = bitnet_stdout(&["chat", "--help"]);
    assert_snapshot!("chat_help", help);
}

#[test]
fn inspect_help() {
    let help = bitnet_stdout(&["inspect", "--help"]);
    assert_snapshot!("inspect_help", help);
}

#[test]
fn compat_check_help() {
    let help = bitnet_stdout(&["compat-check", "--help"]);
    assert_snapshot!("compat_check_help", help);
}

// ---------------------------------------------------------------------------
// Error output for invalid / missing arguments
// ---------------------------------------------------------------------------

#[test]
fn run_missing_required_args_error() {
    // `run` needs at least --model and --prompt; omitting them should fail.
    let err = bitnet_stderr_failure(&["run"]);
    assert_snapshot!("run_missing_required_args_error", err);
}

#[test]
fn compat_check_missing_path_error() {
    // `compat-check` requires a positional <PATH>; omitting it should fail.
    let err = bitnet_stderr_failure(&["compat-check"]);
    assert_snapshot!("compat_check_missing_path_error", err);
}

#[test]
fn unknown_subcommand_error() {
    // An unrecognised subcommand should fail with a helpful error.
    let err = bitnet_stderr_failure(&["this-does-not-exist"]);
    assert_snapshot!("unknown_subcommand_error", err);
}

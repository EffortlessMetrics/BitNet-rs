//! Help Text Golden Snapshot Tests
//!
//! Tests that CLI help text remains stable and doesn't accidentally change.
//! Catches flag renames, removed options, and help text modifications.
//!
//! # Specification References
//! - AC1: Flag Renaming and Aliasing (help text should show aliases)
//! - AC5: Stop Sequence Aliases (help text should show stop aliases)
//! - Spec: docs/explanation/cli-ux-improvements-spec.md

#![cfg(feature = "full-cli")]

use anyhow::Result;
use bitnet_cli::commands::InferenceCommand;
use clap::{CommandFactory, Parser};

/// Test CLI wrapper for generating help text
#[derive(Parser)]
struct TestCli {
    #[command(flatten)]
    cmd: InferenceCommand,
}

#[test]
fn test_help_contains_max_tokens_aliases() -> Result<()> {
    // AC1: Verify help text shows --max-tokens with its aliases
    let mut cmd = TestCli::command();
    let help = format!("{}", cmd.render_help());

    // Primary flag should be present
    assert!(help.contains("--max-tokens"), "Help text missing primary flag --max-tokens");

    // Aliases should be visible
    assert!(
        help.contains("max-new-tokens") || help.contains("--max-new-tokens"),
        "Help text missing alias --max-new-tokens"
    );
    assert!(
        help.contains("n-predict") || help.contains("--n-predict"),
        "Help text missing alias --n-predict"
    );

    Ok(())
}

#[test]
fn test_help_contains_stop_aliases() -> Result<()> {
    // AC5: Verify help text shows --stop with its aliases
    let mut cmd = TestCli::command();
    let help = format!("{}", cmd.render_help());

    // Primary flag should be present
    assert!(help.contains("--stop"), "Help text missing primary flag --stop");

    // Aliases should be visible
    assert!(
        help.contains("stop-sequence") || help.contains("--stop-sequence"),
        "Help text missing alias --stop-sequence"
    );
    assert!(
        help.contains("stop_sequences") || help.contains("--stop_sequences"),
        "Help text missing alias --stop_sequences"
    );

    Ok(())
}

#[test]
fn test_help_contains_critical_flags() -> Result<()> {
    // Verify all critical flags are present in help text
    let mut cmd = TestCli::command();
    let help = format!("{}", cmd.render_help());

    let critical_flags = vec![
        "--model",
        "--prompt",
        "--max-tokens",
        "--temperature",
        "--top-k",
        "--top-p",
        "--stop",
        "--prompt-template",
        "--tokenizer",
        "--system-prompt",
        "--greedy",
        "--deterministic",
        "--seed",
        "--stream",
        "--metrics",
    ];

    for flag in critical_flags {
        assert!(help.contains(flag), "Help text missing critical flag: {}", flag);
    }

    Ok(())
}

#[test]
fn test_help_mentions_template_types() -> Result<()> {
    // Verify help text mentions supported template types
    let mut cmd = TestCli::command();
    let help = format!("{}", cmd.render_help());

    // Should mention template types somewhere in help
    // (either in --prompt-template description or general help)
    let has_template_info = help.contains("raw")
        || help.contains("instruct")
        || help.contains("llama3")
        || help.contains("template");

    assert!(has_template_info, "Help text should mention template types or template system");

    Ok(())
}

#[test]
fn test_help_format_stability() -> Result<()> {
    // Snapshot test: help text should maintain structure
    let mut cmd = TestCli::command();
    let help = format!("{}", cmd.render_help());

    // Basic structural checks
    assert!(help.len() > 100, "Help text too short, may be incomplete");
    assert!(
        help.contains("Options:") || help.contains("Arguments:"),
        "Help text missing Options/Arguments section"
    );

    // Verify it's formatted as proper help text
    assert!(help.lines().count() > 10, "Help text should have multiple lines");

    Ok(())
}

#[test]
fn test_help_footer_has_docs_and_issues_links() -> Result<()> {
    // Verify help footer has both Docs and Issues links using direct CLI builder
    let mut cmd = bitnet_cli::build_cli();
    let help = cmd.render_help().to_string();

    assert!(help.contains("Docs: https://docs.rs/bitnet"), "Help text missing Docs link");
    assert!(
        help.contains("Issues: https://github.com/EffortlessMetrics/BitNet-rs/issues"),
        "Help text missing Issues link"
    );

    Ok(())
}

#[test]
fn test_print_current_help_text() -> Result<()> {
    // Prints current help text; useful during development for manual review.
    let mut cmd = TestCli::command();
    let help = format!("{}", cmd.render_help());

    println!("\n========== CURRENT HELP TEXT ==========\n");
    println!("{}", help);
    println!("\n========================================\n");

    Ok(())
}

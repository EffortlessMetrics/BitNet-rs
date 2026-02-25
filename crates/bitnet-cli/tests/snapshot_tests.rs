//! Snapshot tests for the bitnet-cli crate.
//!
//! Pins key CLI structures (InferenceCommand defaults, help text excerpts)
//! to catch accidental flag renames, removed options, or default value changes.
//!
//! All tests use `full-cli` feature for maximum coverage.

#![cfg(feature = "full-cli")]

use bitnet_cli::commands::InferenceCommand;
use clap::{CommandFactory, Parser};
use insta::assert_snapshot;

/// Minimal wrapper to get clap to generate help for InferenceCommand.
#[derive(Parser)]
struct TestCli {
    #[command(flatten)]
    cmd: InferenceCommand,
}

// -- Default values ----------------------------------------------------------

#[test]
fn inference_command_defaults() {
    let cmd = InferenceCommand::default();
    // Snapshot the key defaults — catches accidental default value changes
    let defaults = format!(
        "max_tokens={} temperature={:.1} seed={:?} greedy={} prompt_template={:?}",
        cmd.max_tokens,
        cmd.temperature,
        cmd.seed,
        cmd.greedy,
        cmd.prompt_template
    );
    assert_snapshot!("inference_command_defaults", defaults);
}

// -- Help text excerpts ------------------------------------------------------

#[test]
fn help_contains_max_tokens_flag() {
    let mut cmd = TestCli::command();
    let help = cmd.render_help().to_string();
    // Snapshot the section containing --max-tokens and its aliases
    let relevant: String = help
        .lines()
        .filter(|l| {
            l.contains("max-tokens") || l.contains("max-new-tokens") || l.contains("n-predict")
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert_snapshot!("help_max_tokens_section", relevant);
}

#[test]
fn help_contains_stop_flags() {
    let mut cmd = TestCli::command();
    let help = cmd.render_help().to_string();
    let relevant: String = help
        .lines()
        .filter(|l| {
            l.contains("--stop") || l.contains("stop-sequence") || l.contains("stop_sequences")
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert_snapshot!("help_stop_section", relevant);
}

#[test]
fn help_contains_sampling_flags() {
    let mut cmd = TestCli::command();
    let help = cmd.render_help().to_string();
    let relevant: String = help
        .lines()
        .filter(|l| {
            l.contains("--temperature")
                || l.contains("--top-p")
                || l.contains("--top-k")
                || l.contains("--greedy")
                || l.contains("--seed")
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert_snapshot!("help_sampling_section", relevant);
}

#[test]
fn help_contains_prompt_template_flag() {
    let mut cmd = TestCli::command();
    let help = cmd.render_help().to_string();
    let relevant: String = help
        .lines()
        .filter(|l| l.contains("prompt-template") || l.contains("system-prompt"))
        .collect::<Vec<_>>()
        .join("\n");
    assert_snapshot!("help_prompt_template_section", relevant);
}

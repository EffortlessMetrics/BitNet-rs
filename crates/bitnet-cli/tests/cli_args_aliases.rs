//! CLI Argument Alias Tests
//!
//! Tests acceptance criteria AC1 and AC5 for the CLI UX Improvements specification.
//! Verifies that all flag aliases work identically to their primary flags.
//!
//! # Specification References
//! - AC1: Flag Renaming and Aliasing (--max-tokens aliases)
//! - AC5: Stop Sequence Aliases (--stop aliases)
//! - Spec: docs/explanation/cli-ux-improvements-spec.md

use anyhow::Result;
use bitnet_cli::commands::InferenceCommand;
use clap::Parser;

/// Test CLI wrapper for parsing InferenceCommand
#[derive(Parser)]
struct TestCli {
    #[command(flatten)]
    cmd: InferenceCommand,
}

/// Helper function to parse CLI arguments into InferenceCommand
fn parse_args(args: &[&str]) -> Result<InferenceCommand> {
    let cli = TestCli::try_parse_from(args)?;
    Ok(cli.cmd)
}

#[cfg(test)]
mod ac1_max_tokens_aliases {
    use super::*;

    // AC1:primary - Primary --max-tokens flag works
    #[test]
    fn test_max_tokens_primary_flag() -> Result<()> {
        // Tests feature spec: cli-ux-improvements-spec.md#AC1
        // Verify that --max-tokens is accepted as the primary flag

        let cmd = parse_args(&["test-cli", "--max-tokens", "16"])?;
        assert_eq!(cmd.max_tokens, 16);
        Ok(())
    }

    // AC1:alias1 - Backward-compatible --max-new-tokens alias works
    #[test]
    fn test_max_new_tokens_alias() -> Result<()> {
        // Tests feature spec: cli-ux-improvements-spec.md#AC1:alias1
        // Verify that --max-new-tokens works identically to --max-tokens

        let cmd = parse_args(&["test-cli", "--max-new-tokens", "16"])?;
        assert_eq!(cmd.max_tokens, 16);
        Ok(())
    }

    // AC1:alias2 - llama.cpp-style --n-predict alias works
    #[test]
    fn test_n_predict_alias() -> Result<()> {
        // Tests feature spec: cli-ux-improvements-spec.md#AC1:alias2
        // Verify that --n-predict works identically to --max-tokens

        let cmd = parse_args(&["test-cli", "--n-predict", "16"])?;
        assert_eq!(cmd.max_tokens, 16);
        Ok(())
    }

    // AC1:all_identical - All three aliases produce identical behavior
    #[test]
    fn test_all_max_tokens_aliases_identical() -> Result<()> {
        // Tests feature spec: cli-ux-improvements-spec.md#AC1
        // Verify that all three flag variants produce the same result

        let cmd1 = parse_args(&["test-cli", "--max-tokens", "32"])?;
        let cmd2 = parse_args(&["test-cli", "--max-new-tokens", "32"])?;
        let cmd3 = parse_args(&["test-cli", "--n-predict", "32"])?;

        // Expected: All three produce identical max_tokens value
        assert_eq!(cmd1.max_tokens, 32);
        assert_eq!(cmd2.max_tokens, 32);
        assert_eq!(cmd3.max_tokens, 32);
        assert_eq!(cmd1.max_tokens, cmd2.max_tokens);
        assert_eq!(cmd2.max_tokens, cmd3.max_tokens);

        Ok(())
    }

    // AC1:help_text - Help text shows primary flag with aliases documented
    #[test]
    #[ignore = "implementation pending: verify help text includes aliases"]
    fn test_help_text_shows_aliases() -> Result<()> {
        // Tests feature spec: cli-ux-improvements-spec.md#AC1
        // Verify that help text shows --max-tokens as primary with aliases

        // TODO: Capture help output and verify content
        // let help_output = get_cli_help_output("run");

        // Expected: Help text mentions all three variants
        // assert!(help_output.contains("--max-tokens"));
        // assert!(help_output.contains("--max-new-tokens"));
        // assert!(help_output.contains("--n-predict"));

        panic!("Test not implemented: needs help text verification");
    }
}

#[cfg(test)]
mod ac5_stop_sequence_aliases {
    use super::*;

    // AC5:primary - Primary --stop flag works
    #[test]
    fn test_stop_primary_flag() -> Result<()> {
        // Tests feature spec: cli-ux-improvements-spec.md#AC5:primary
        // Verify that --stop is accepted as the primary flag

        let cmd = parse_args(&["test-cli", "--stop", "</s>", "--stop", "\n\n"])?;

        // Expected: stop field contains both sequences
        assert_eq!(cmd.stop.len(), 2);
        assert!(cmd.stop.contains(&"</s>".to_string()));
        assert!(cmd.stop.contains(&"\n\n".to_string()));

        Ok(())
    }

    // AC5:alias1 - Singular --stop-sequence alias works
    #[test]
    fn test_stop_sequence_alias() -> Result<()> {
        // Tests feature spec: cli-ux-improvements-spec.md#AC5:alias1
        // Verify that --stop-sequence works identically to --stop

        let cmd = parse_args(&["test-cli", "--stop-sequence", "</s>"])?;

        // Expected: stop field contains the sequence
        assert_eq!(cmd.stop.len(), 1);
        assert_eq!(cmd.stop[0], "</s>");

        Ok(())
    }

    // AC5:alias2 - Plural --stop_sequences alias works
    #[test]
    fn test_stop_sequences_alias() -> Result<()> {
        // Tests feature spec: cli-ux-improvements-spec.md#AC5:alias2
        // Verify that --stop_sequences works identically to --stop

        let cmd = parse_args(&["test-cli", "--stop_sequences", "</s>"])?;

        // Expected: stop field contains the sequence
        assert_eq!(cmd.stop.len(), 1);
        assert_eq!(cmd.stop[0], "</s>");

        Ok(())
    }

    // AC5:all_identical - All three stop aliases produce identical behavior
    #[test]
    fn test_all_stop_aliases_identical() -> Result<()> {
        // Tests feature spec: cli-ux-improvements-spec.md#AC5
        // Verify that all three flag variants produce the same result

        let cmd1 = parse_args(&["test-cli", "--stop", "<|eot_id|>"])?;
        let cmd2 = parse_args(&["test-cli", "--stop-sequence", "<|eot_id|>"])?;
        let cmd3 = parse_args(&["test-cli", "--stop_sequences", "<|eot_id|>"])?;

        // Expected: All three produce identical stop sequences
        assert_eq!(cmd1.stop.len(), 1);
        assert_eq!(cmd2.stop.len(), 1);
        assert_eq!(cmd3.stop.len(), 1);
        assert_eq!(cmd1.stop[0], "<|eot_id|>");
        assert_eq!(cmd2.stop[0], "<|eot_id|>");
        assert_eq!(cmd3.stop[0], "<|eot_id|>");
        assert_eq!(cmd1.stop, cmd2.stop);
        assert_eq!(cmd2.stop, cmd3.stop);

        Ok(())
    }

    // AC5:multiple - Multiple stop sequences work with all aliases
    #[test]
    fn test_multiple_stop_sequences_with_aliases() -> Result<()> {
        // Tests feature spec: cli-ux-improvements-spec.md#AC5
        // Verify that multiple stop sequences work with all aliases

        let cmd = parse_args(&[
            "test-cli",
            "--stop",
            "</s>",
            "--stop-sequence",
            "<|eot_id|>",
            "--stop_sequences",
            "\n\n",
        ])?;

        // Expected: All sequences are captured (order may vary)
        assert_eq!(cmd.stop.len(), 3);
        assert!(cmd.stop.contains(&"</s>".to_string()));
        assert!(cmd.stop.contains(&"<|eot_id|>".to_string()));
        assert!(cmd.stop.contains(&"\n\n".to_string()));

        Ok(())
    }

    // AC5:help_text - Help text shows primary flag with aliases documented
    #[test]
    #[ignore = "implementation pending: verify help text includes stop aliases"]
    fn test_help_text_shows_stop_aliases() -> Result<()> {
        // Tests feature spec: cli-ux-improvements-spec.md#AC5
        // Verify that help text shows --stop as primary with aliases

        // TODO: Capture help output and verify content
        // let help_output = get_cli_help_output("run");

        // Expected: Help text mentions all three variants
        // assert!(help_output.contains("--stop"));
        // assert!(help_output.contains("--stop-sequence"));
        // assert!(help_output.contains("--stop_sequences"));

        panic!("Test not implemented: needs help text verification");
    }
}

#[cfg(test)]
mod backward_compatibility {
    use super::*;

    // AC10:backward_compat - Existing CLI workflows continue to work
    #[test]
    fn test_existing_workflows_unchanged() -> Result<()> {
        // Tests feature spec: cli-ux-improvements-spec.md#AC10
        // Verify that existing commands work without modification

        let cmd = parse_args(&[
            "test-cli",
            "--max-tokens",
            "16",
            "--stop",
            "</s>",
            "--temperature",
            "0.7",
        ])?;

        // Expected: All existing flags parse correctly
        assert_eq!(cmd.max_tokens, 16);
        assert_eq!(cmd.stop.len(), 1);
        assert_eq!(cmd.stop[0], "</s>");
        assert_eq!(cmd.temperature, 0.7);

        Ok(())
    }
}

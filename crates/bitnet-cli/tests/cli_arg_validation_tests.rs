//! CLI argument validation tests using clap's test utilities (no process spawning).
//!
//! Covers areas not addressed by existing tests:
//!   - `build_cli()` exported from the library returns a command named "bitnet"
//!   - `InferenceCommand` `--stop-id` collects multiple numeric token IDs
//!   - `InferenceCommand` `--greedy` default is false
//!   - `InferenceCommand` `--stream` default is false
//!   - `InferenceCommand` `--system-prompt` is optional (accepts None)
//!   - `InferenceCommand` `--prompt-template` defaults to "auto"
//!   - `InferenceCommand` `--stop-id` roundtrips correctly

// ── build_cli() library export ────────────────────────────────────────────────

/// The library's `build_cli()` must return a Command whose name is "bitnet".
#[test]
fn test_build_cli_command_name_is_bitnet() {
    let cmd = bitnet_cli::build_cli();
    assert_eq!(cmd.get_name(), "bitnet");
}

/// `build_cli()` must produce a valid Command (binary name can be used).
#[test]
fn test_build_cli_returns_valid_command() {
    let cmd = bitnet_cli::build_cli();
    // The binary name defaults to the command name when not set separately.
    // Just verifying it doesn't panic and has correct name is sufficient.
    assert!(!cmd.get_name().is_empty());
}

// ── InferenceCommand clap tests (require full-cli) ────────────────────────────

#[cfg(feature = "full-cli")]
mod inference_cmd {
    use bitnet_cli::commands::InferenceCommand;
    use clap::Parser;

    /// Thin wrapper used exclusively for testing.
    #[derive(Parser)]
    struct TestCli {
        #[command(flatten)]
        cmd: InferenceCommand,
    }

    fn parse(args: &[&str]) -> Result<InferenceCommand, clap::Error> {
        TestCli::try_parse_from(args).map(|c| c.cmd)
    }

    // ── greedy flag ───────────────────────────────────────────────────────────

    /// `--greedy` defaults to `false`.
    #[test]
    fn test_greedy_default_false() {
        let cmd = parse(&["test-cli"]).expect("empty args must parse");
        assert!(!cmd.greedy, "greedy must default to false");
    }

    /// `--greedy` can be set to `true`.
    #[test]
    fn test_greedy_flag_sets_true() {
        let cmd = parse(&["test-cli", "--greedy"]).expect("--greedy must be accepted");
        assert!(cmd.greedy, "--greedy flag must set greedy to true");
    }

    // ── stream flag ───────────────────────────────────────────────────────────

    /// `--stream` defaults to `false`.
    #[test]
    fn test_stream_default_false() {
        let cmd = parse(&["test-cli"]).expect("empty args must parse");
        assert!(!cmd.stream, "stream must default to false");
    }

    /// `--stream` flag can be set.
    #[test]
    fn test_stream_flag_sets_true() {
        let cmd = parse(&["test-cli", "--stream"]).expect("--stream must be accepted");
        assert!(cmd.stream, "--stream must set stream to true");
    }

    // ── prompt-template ───────────────────────────────────────────────────────

    /// `--prompt-template` defaults to `"auto"`.
    #[test]
    fn test_prompt_template_defaults_to_auto() {
        let cmd = parse(&["test-cli"]).expect("empty args must parse");
        assert_eq!(cmd.prompt_template, "auto", "prompt_template must default to \"auto\"");
    }

    /// `--prompt-template raw` is accepted.
    #[test]
    fn test_prompt_template_raw_accepted() {
        let cmd = parse(&["test-cli", "--prompt-template", "raw"])
            .expect("--prompt-template raw must be accepted");
        assert_eq!(cmd.prompt_template, "raw");
    }

    /// `--prompt-template instruct` is accepted.
    #[test]
    fn test_prompt_template_instruct_accepted() {
        let cmd = parse(&["test-cli", "--prompt-template", "instruct"])
            .expect("--prompt-template instruct must be accepted");
        assert_eq!(cmd.prompt_template, "instruct");
    }

    /// `--prompt-template llama3-chat` is accepted.
    #[test]
    fn test_prompt_template_llama3_chat_accepted() {
        let cmd = parse(&["test-cli", "--prompt-template", "llama3-chat"])
            .expect("--prompt-template llama3-chat must be accepted");
        assert_eq!(cmd.prompt_template, "llama3-chat");
    }

    // ── system-prompt ─────────────────────────────────────────────────────────

    /// `--system-prompt` is optional and defaults to `None`.
    #[test]
    fn test_system_prompt_default_none() {
        let cmd = parse(&["test-cli"]).expect("empty args must parse");
        assert!(cmd.system_prompt.is_none(), "system_prompt must default to None");
    }

    /// `--system-prompt` accepts a string value.
    #[test]
    fn test_system_prompt_accepted() {
        let cmd = parse(&["test-cli", "--system-prompt", "You are a helpful assistant"])
            .expect("--system-prompt must be accepted");
        assert_eq!(cmd.system_prompt.as_deref(), Some("You are a helpful assistant"));
    }

    // ── stop-id ───────────────────────────────────────────────────────────────

    /// A single `--stop-id` value is collected correctly.
    #[test]
    fn test_stop_id_single_value() {
        let cmd = parse(&["test-cli", "--stop-id", "128009"]).expect("--stop-id must be accepted");
        assert_eq!(cmd.stop_id, vec![128009u32]);
    }

    /// Multiple `--stop-id` values are all collected.
    #[test]
    fn test_stop_id_multiple_values_collected() {
        let cmd =
            parse(&["test-cli", "--stop-id", "128009", "--stop-id", "2", "--stop-id", "32000"])
                .expect("multiple --stop-id must be accepted");
        assert_eq!(cmd.stop_id, vec![128009u32, 2, 32000]);
    }

    /// Without `--stop-id`, the collection is empty.
    #[test]
    fn test_stop_id_defaults_to_empty_vec() {
        let cmd = parse(&["test-cli"]).expect("empty args must parse");
        assert!(cmd.stop_id.is_empty(), "stop_id must default to empty vec");
    }

    // ── combined sampling params ──────────────────────────────────────────────

    /// Sampling params parse correctly together.
    #[test]
    fn test_combined_sampling_params_parse() {
        let cmd = parse(&[
            "test-cli",
            "--temperature",
            "0.8",
            "--top-p",
            "0.95",
            "--max-tokens",
            "64",
            "--greedy",
            "--stop-id",
            "128009",
            "--stop",
            "</s>",
            "--system-prompt",
            "Be concise",
            "--prompt-template",
            "instruct",
        ])
        .expect("combined sampling params must parse");

        assert!((cmd.temperature - 0.8).abs() < 1e-6);
        assert!((cmd.top_p.unwrap_or(0.0) - 0.95).abs() < 1e-6);
        assert_eq!(cmd.max_tokens, 64);
        assert!(cmd.greedy);
        assert_eq!(cmd.stop_id, vec![128009u32]);
        assert_eq!(cmd.stop, vec!["</s>"]);
        assert_eq!(cmd.system_prompt.as_deref(), Some("Be concise"));
        assert_eq!(cmd.prompt_template, "instruct");
    }
}

//! Property-based tests for bitnet-cli using proptest.
//!
//! Tests cover CLI argument parsing invariants:
//! - max_tokens aliases (--max-tokens, --max-new-tokens, --n-predict)
//! - temperature stays within parsed range
//! - repetition_penalty stays within parsed range
//! - top_k and top_p optional fields parse correctly
//! - greedy mode invariants (overrides temperature)

#![cfg(feature = "full-cli")]

use bitnet_cli::commands::InferenceCommand;
use clap::Parser;
use proptest::prelude::*;

/// Test CLI wrapper for parsing InferenceCommand
#[derive(Parser)]
struct TestCli {
    #[command(flatten)]
    cmd: InferenceCommand,
}

fn parse_args(args: &[&str]) -> Result<InferenceCommand, clap::Error> {
    TestCli::try_parse_from(args).map(|c| c.cmd)
}

proptest! {
    /// max_tokens: valid positive values are always accepted
    #[test]
    fn prop_max_tokens_positive_valid(n in 1usize..=4096usize) {
        let n_str = n.to_string();
        let args = ["bitnet", "--max-tokens", &n_str];
        let cmd = parse_args(&args).expect("valid max_tokens should parse");
        prop_assert_eq!(cmd.max_tokens, n);
    }

    /// max-new-tokens alias: must behave identically to --max-tokens
    #[test]
    fn prop_max_new_tokens_alias_same_as_primary(n in 1usize..=4096usize) {
        let n_str = n.to_string();
        let primary = parse_args(&["bitnet", "--max-tokens", &n_str]).expect("should parse");
        let alias   = parse_args(&["bitnet", "--max-new-tokens", &n_str]).expect("should parse");
        prop_assert_eq!(primary.max_tokens, alias.max_tokens);
        prop_assert_eq!(primary.max_tokens, n);
    }

    /// n-predict alias: must behave identically to --max-tokens
    #[test]
    fn prop_n_predict_alias_same_as_primary(n in 1usize..=4096usize) {
        let n_str = n.to_string();
        let primary = parse_args(&["bitnet", "--max-tokens", &n_str]).expect("should parse");
        let alias   = parse_args(&["bitnet", "--n-predict", &n_str]).expect("should parse");
        prop_assert_eq!(primary.max_tokens, alias.max_tokens);
    }

    /// temperature: values ≥ 0.0 are accepted by clap (domain validation is in engine)
    #[test]
    fn prop_temperature_nonnegative_parses(
        temp_int in 0u32..=300u32  // 0.0 to 3.0 in 0.01 steps
    ) {
        let temp = format!("{:.2}", temp_int as f32 / 100.0);
        let cmd = parse_args(&["bitnet", "--temperature", &temp]).expect("should parse");
        prop_assert!(cmd.temperature >= 0.0, "temperature should be >= 0.0");
    }

    /// repetition_penalty: values ≥ 0.0 parse correctly
    #[test]
    fn prop_repetition_penalty_parses(
        penalty_int in 10u32..=500u32  // 0.1 to 5.0 in 0.01 steps
    ) {
        let penalty = format!("{:.2}", penalty_int as f32 / 100.0);
        let cmd = parse_args(&["bitnet", "--repetition-penalty", &penalty]).expect("should parse");
        prop_assert!(cmd.repetition_penalty >= 0.0);
    }

    /// top_k: positive integers parse as Some(k)
    #[test]
    fn prop_top_k_positive_parses(k in 1usize..=1000usize) {
        let k_str = k.to_string();
        let cmd = parse_args(&["bitnet", "--top-k", &k_str]).expect("should parse");
        prop_assert_eq!(cmd.top_k, Some(k));
    }

    /// top_p: when not provided, is None
    #[test]
    fn prop_top_p_absent_is_none(_seed in 0u32..100u32) {
        let cmd = parse_args(&["bitnet"]).expect("should parse");
        prop_assert!(cmd.top_p.is_none(), "top_p should be None when not specified");
    }

    /// greedy: when set, greedy flag is true
    #[test]
    fn prop_greedy_flag_sets_greedy(_seed in 0u32..100u32) {
        let cmd = parse_args(&["bitnet", "--greedy"]).expect("should parse");
        prop_assert!(cmd.greedy);
    }

    /// seed: provided values are stored exactly
    #[test]
    fn prop_seed_stored_exactly(seed in 0u64..=u64::MAX) {
        let seed_str = seed.to_string();
        let cmd = parse_args(&["bitnet", "--seed", &seed_str]).expect("should parse");
        prop_assert_eq!(cmd.seed, Some(seed));
    }

    /// --temperature 0.0 --greedy: both fields are stored correctly together
    #[test]
    fn prop_greedy_and_zero_temperature_both_stored(_seed in 0u32..100u32) {
        let cmd = parse_args(&["bitnet", "--temperature", "0.0", "--greedy"])
            .expect("should parse");
        prop_assert!(cmd.greedy, "greedy flag must be true");
        prop_assert_eq!(cmd.temperature, 0.0f32, "temperature must be 0.0");
    }

    /// stop sequences: all provided sequences are stored in order
    #[test]
    fn prop_stop_sequences_all_stored(
        n in 1usize..=5usize,
    ) {
        // Build n unique stop sequences
        let stops: Vec<String> = (0..n).map(|i| format!("<stop{i}>")).collect();
        let mut args = vec!["bitnet"];
        for s in &stops {
            args.push("--stop");
            args.push(s.as_str());
        }
        let cmd = parse_args(&args).expect("should parse");
        prop_assert_eq!(cmd.stop.len(), n, "all stop sequences must be stored");
        for s in &stops {
            prop_assert!(cmd.stop.contains(s), "missing stop sequence: {s}");
        }
    }

    /// stop-id: all provided token IDs are stored
    #[test]
    fn prop_stop_ids_all_stored(
        id in 1u32..=200000u32,
    ) {
        let id_str = id.to_string();
        let cmd = parse_args(&["bitnet", "--stop-id", &id_str]).expect("should parse");
        prop_assert!(cmd.stop_id.contains(&id), "stop-id {id} must be stored");
    }

    /// --prompt-template: any string is accepted by clap (validation is deferred)
    #[test]
    fn prop_prompt_template_string_stored_verbatim(
        template in "[a-z][a-z-]{0,19}",
    ) {
        let cmd = parse_args(&["bitnet", "--prompt-template", &template])
            .expect("clap should accept any string for --prompt-template");
        prop_assert_eq!(
            cmd.prompt_template, template,
            "--prompt-template value must be stored verbatim"
        );
    }
}

// ── Deterministic unit tests ─────────────────────────────────────────────────

/// --prompt-template raw/instruct/llama3-chat each parse to the correct TemplateType.
#[test]
fn test_explicit_prompt_templates_parse_to_correct_type() {
    use bitnet_inference::TemplateType;

    let cases = [
        ("raw", TemplateType::Raw),
        ("instruct", TemplateType::Instruct),
        ("llama3-chat", TemplateType::Llama3Chat),
    ];

    for (value, expected) in cases {
        let cmd = parse_args(&["bitnet", "--prompt-template", value]).expect("should parse");
        let resolved: TemplateType = cmd
            .prompt_template
            .parse()
            .unwrap_or_else(|e| panic!("--prompt-template {value:?} should resolve: {e}"));
        assert_eq!(resolved, expected, "wrong TemplateType for {value:?}");
    }
}

/// An unknown --prompt-template value is stored verbatim and fails str::parse.
#[test]
fn test_unknown_prompt_template_fails_at_parse_time() {
    use bitnet_inference::TemplateType;

    let cmd = parse_args(&["bitnet", "--prompt-template", "totally-invalid-template"])
        .expect("clap should accept any string");
    let result: Result<TemplateType, _> = cmd.prompt_template.parse();
    assert!(result.is_err(), "unknown template string should fail TemplateType::from_str");
}

/// --temperature 0.0 --greedy: the combination sets both fields correctly.
#[test]
fn test_greedy_zero_temperature_combination() {
    let cmd = parse_args(&["bitnet", "--temperature", "0.0", "--greedy"]).expect("should parse");
    assert!(cmd.greedy, "greedy must be true");
    assert_eq!(cmd.temperature, 0.0f32, "temperature must be 0.0");
}

/// Multiple --stop-id values are all stored.
#[test]
fn test_multiple_stop_ids_stored() {
    let cmd = parse_args(&["bitnet", "--stop-id", "128009", "--stop-id", "2", "--stop-id", "1"])
        .expect("should parse");
    assert_eq!(cmd.stop_id.len(), 3);
    assert!(cmd.stop_id.contains(&128009u32));
    assert!(cmd.stop_id.contains(&2u32));
    assert!(cmd.stop_id.contains(&1u32));
}

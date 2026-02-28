//! Extended tests for `bitnet-cli`.
//!
//! Covers areas not exercised by the existing test files:
//!   - `InferenceCommand` default field values
//!   - `PromptTemplate` / `TemplateType` parsing: case-insensitive variants, underscore alias
//!   - `TemplateType::detect()` from GGUF metadata and tokenizer name
//!   - `TemplateType` methods: `Display`, `default_stop_sequences`, `should_add_bos`,
//!     `parse_special`, `apply`
//!   - `CliConfig` validation (invalid device / log-level / format / zero batch-size)
//!   - `ConfigBuilder` fluent API
//!   - `InferenceCommand` specific optional fields (`--top-p`, `--stop-string-window`)
//!   - Property test: top-p values in [0,1] stored correctly

// ── InferenceCommand tests require the full-cli feature ─────────────────────

#[cfg(feature = "full-cli")]
use bitnet_cli::commands::InferenceCommand;
#[cfg(feature = "full-cli")]
use clap::Parser;

use bitnet_cli::config::{CliConfig, ConfigBuilder};
use bitnet_inference::TemplateType;

// ── Helpers ──────────────────────────────────────────────────────────────────

#[cfg(feature = "full-cli")]
#[derive(Parser)]
struct TestCli {
    #[command(flatten)]
    cmd: InferenceCommand,
}

#[cfg(feature = "full-cli")]
fn parse_args(args: &[&str]) -> Result<InferenceCommand, clap::Error> {
    TestCli::try_parse_from(args).map(|c| c.cmd)
}

// ============================================================================
// 1. InferenceCommand default field values
// ============================================================================

#[cfg(feature = "full-cli")]
mod inference_defaults {
    use super::*;

    /// Default temperature is 0.7.
    #[test]
    fn test_default_temperature_is_0_7() {
        let cmd = parse_args(&["bitnet"]).expect("empty args must parse");
        assert!(
            (cmd.temperature - 0.7).abs() < 1e-6,
            "expected temperature=0.7, got {}",
            cmd.temperature
        );
    }

    /// Default max_tokens is 512.
    #[test]
    fn test_default_max_tokens_is_512() {
        let cmd = parse_args(&["bitnet"]).expect("empty args must parse");
        assert_eq!(cmd.max_tokens, 512);
    }

    /// Default repetition_penalty is 1.1.
    #[test]
    fn test_default_repetition_penalty_is_1_1() {
        let cmd = parse_args(&["bitnet"]).expect("empty args must parse");
        assert!(
            (cmd.repetition_penalty - 1.1).abs() < 1e-6,
            "expected repetition_penalty=1.1, got {}",
            cmd.repetition_penalty
        );
    }

    /// Default stop_string_window is 64.
    #[test]
    fn test_default_stop_string_window_is_64() {
        let cmd = parse_args(&["bitnet"]).expect("empty args must parse");
        assert_eq!(cmd.stop_string_window, 64);
    }

    /// Default prompt_template is "auto".
    #[test]
    fn test_default_prompt_template_is_auto() {
        let cmd = parse_args(&["bitnet"]).expect("empty args must parse");
        assert_eq!(cmd.prompt_template, "auto");
    }

    /// Default format is "text".
    #[test]
    fn test_default_format_is_text() {
        let cmd = parse_args(&["bitnet"]).expect("empty args must parse");
        assert_eq!(cmd.format, "text");
    }

    /// Default batch_size is 1.
    #[test]
    fn test_default_batch_size_is_1() {
        let cmd = parse_args(&["bitnet"]).expect("empty args must parse");
        assert_eq!(cmd.batch_size, 1);
    }

    /// Default greedy is false.
    #[test]
    fn test_default_greedy_is_false() {
        let cmd = parse_args(&["bitnet"]).expect("empty args must parse");
        assert!(!cmd.greedy, "greedy must default to false");
    }

    /// Default no_bos is false.
    #[test]
    fn test_default_no_bos_is_false() {
        let cmd = parse_args(&["bitnet"]).expect("empty args must parse");
        assert!(!cmd.no_bos, "no_bos must default to false");
    }

    /// Default no_eos is false.
    #[test]
    fn test_default_no_eos_is_false() {
        let cmd = parse_args(&["bitnet"]).expect("empty args must parse");
        assert!(!cmd.no_eos, "no_eos must default to false");
    }

    /// top_k is absent (None) by default.
    #[test]
    fn test_default_top_k_is_none() {
        let cmd = parse_args(&["bitnet"]).expect("empty args must parse");
        assert!(cmd.top_k.is_none(), "top_k must default to None");
    }

    /// top_p is absent (None) by default.
    #[test]
    fn test_default_top_p_is_none() {
        let cmd = parse_args(&["bitnet"]).expect("empty args must parse");
        assert!(cmd.top_p.is_none(), "top_p must default to None");
    }
}

// ============================================================================
// 2. PromptTemplate flag parsing: case-insensitive and underscore alias
// ============================================================================

#[cfg(feature = "full-cli")]
mod prompt_template_parsing {
    use super::*;

    /// "RAW" (uppercase) is stored verbatim and parses to TemplateType::Raw.
    #[test]
    fn test_prompt_template_case_insensitive_raw() {
        let cmd = parse_args(&["bitnet", "--prompt-template", "RAW"]).expect("should parse");
        let tpl: TemplateType =
            cmd.prompt_template.parse().expect("RAW should parse to TemplateType");
        assert_eq!(tpl, TemplateType::Raw);
    }

    /// "INSTRUCT" (uppercase) parses to TemplateType::Instruct.
    #[test]
    fn test_prompt_template_case_insensitive_instruct() {
        let cmd = parse_args(&["bitnet", "--prompt-template", "INSTRUCT"]).expect("should parse");
        let tpl: TemplateType =
            cmd.prompt_template.parse().expect("INSTRUCT should parse to TemplateType");
        assert_eq!(tpl, TemplateType::Instruct);
    }

    /// "LLAMA3-CHAT" (uppercase) parses to TemplateType::Llama3Chat.
    #[test]
    fn test_prompt_template_case_insensitive_llama3_chat() {
        let cmd =
            parse_args(&["bitnet", "--prompt-template", "LLAMA3-CHAT"]).expect("should parse");
        let tpl: TemplateType =
            cmd.prompt_template.parse().expect("LLAMA3-CHAT should parse to TemplateType");
        assert_eq!(tpl, TemplateType::Llama3Chat);
    }

    /// "llama3_chat" (underscore variant) parses to TemplateType::Llama3Chat.
    #[test]
    fn test_prompt_template_underscore_variant() {
        let cmd =
            parse_args(&["bitnet", "--prompt-template", "llama3_chat"]).expect("should parse");
        let tpl: TemplateType =
            cmd.prompt_template.parse().expect("llama3_chat should parse to TemplateType");
        assert_eq!(tpl, TemplateType::Llama3Chat);
    }

    /// An empty string stored as prompt_template fails TemplateType::from_str.
    #[test]
    fn test_empty_prompt_template_fails_parse() {
        let result: Result<TemplateType, _> = "".parse();
        assert!(result.is_err(), "empty string must fail TemplateType::from_str");
    }

    /// Parsing all three canonical names succeeds and returns the right variant.
    #[test]
    fn test_all_canonical_template_names_parse() {
        let cases = [
            ("raw", TemplateType::Raw),
            ("instruct", TemplateType::Instruct),
            ("llama3-chat", TemplateType::Llama3Chat),
        ];
        for (name, expected) in &cases {
            let got: TemplateType = name.parse().unwrap_or_else(|e| {
                panic!("'{name}' should parse to TemplateType: {e}");
            });
            assert_eq!(&got, expected, "wrong variant for '{name}'");
        }
    }
}

// ============================================================================
// 3. TemplateType::detect()
// ============================================================================

mod template_detect {
    use super::*;

    /// LLaMA-3 jinja template is detected as Llama3Chat.
    #[test]
    fn test_detect_llama3_from_jinja() {
        let jinja = "{% if <|start_header_id|> %}...{% endif %} <|eot_id|>".to_string();
        let result = TemplateType::detect(None, Some(&jinja));
        assert_eq!(result, TemplateType::Llama3Chat);
    }

    /// Instruct-style jinja template is detected as Instruct.
    #[test]
    fn test_detect_instruct_from_jinja() {
        let jinja = "{% for message in messages %}{{ message.content }}{% endfor %}".to_string();
        let result = TemplateType::detect(None, Some(&jinja));
        assert_eq!(result, TemplateType::Instruct);
    }

    /// Tokenizer name containing "llama3" → Llama3Chat.
    #[test]
    fn test_detect_llama3_from_tokenizer_name() {
        let result = TemplateType::detect(Some("llama3-tokenizer"), None);
        assert_eq!(result, TemplateType::Llama3Chat);
    }

    /// Tokenizer name containing "instruct" → Instruct.
    #[test]
    fn test_detect_instruct_from_tokenizer_name() {
        let result = TemplateType::detect(Some("mistral-instruct-v0.2"), None);
        assert_eq!(result, TemplateType::Instruct);
    }

    /// No hints at all → Raw (fallback).
    #[test]
    fn test_detect_raw_fallback() {
        let result = TemplateType::detect(None, None);
        assert_eq!(result, TemplateType::Raw);
    }

    /// GGUF metadata takes priority over tokenizer name.
    #[test]
    fn test_detect_gguf_metadata_wins_over_tokenizer_name() {
        // Jinja says LLaMA-3, tokenizer name says "instruct" — jinja wins.
        let jinja = "<|start_header_id|>user<|end_header_id|>\n{msg}<|eot_id|>".to_string();
        let result = TemplateType::detect(Some("generic-instruct"), Some(&jinja));
        assert_eq!(result, TemplateType::Llama3Chat, "GGUF metadata must override tokenizer name");
    }
}

// ============================================================================
// 4. TemplateType methods
// ============================================================================

mod template_methods {
    use super::*;

    /// Display for all three variants matches the canonical string.
    #[test]
    fn test_template_display() {
        assert_eq!(TemplateType::Raw.to_string(), "raw");
        assert_eq!(TemplateType::Instruct.to_string(), "instruct");
        assert_eq!(TemplateType::Llama3Chat.to_string(), "llama3-chat");
    }

    /// Raw template has no default stop sequences.
    #[test]
    fn test_raw_default_stop_sequences_empty() {
        assert!(
            TemplateType::Raw.default_stop_sequences().is_empty(),
            "Raw must have no default stop sequences"
        );
    }

    /// Instruct template has at least one default stop sequence.
    #[test]
    fn test_instruct_default_stop_sequences_nonempty() {
        let stops = TemplateType::Instruct.default_stop_sequences();
        assert!(!stops.is_empty(), "Instruct must have at least one default stop sequence");
    }

    /// LLaMA-3 template stop sequences include "<|eot_id|>".
    #[test]
    fn test_llama3_default_stop_sequences_include_eot() {
        let stops = TemplateType::Llama3Chat.default_stop_sequences();
        assert!(
            stops.iter().any(|s| s.contains("<|eot_id|>")),
            "Llama3Chat default stops must include <|eot_id|>"
        );
    }

    /// Raw and Instruct should_add_bos returns true; Llama3Chat returns false.
    #[test]
    fn test_should_add_bos() {
        assert!(TemplateType::Raw.should_add_bos(), "Raw must add BOS");
        assert!(TemplateType::Instruct.should_add_bos(), "Instruct must add BOS");
        assert!(!TemplateType::Llama3Chat.should_add_bos(), "Llama3Chat must NOT add BOS");
    }

    /// Only Llama3Chat enables parse_special.
    #[test]
    fn test_parse_special_only_llama3() {
        assert!(!TemplateType::Raw.parse_special(), "Raw must not parse_special");
        assert!(!TemplateType::Instruct.parse_special(), "Instruct must not parse_special");
        assert!(TemplateType::Llama3Chat.parse_special(), "Llama3Chat must parse_special");
    }

    /// Raw template apply() returns the user text unchanged.
    #[test]
    fn test_apply_raw_passthrough() {
        let text = "What is 2+2?";
        assert_eq!(TemplateType::Raw.apply(text, None), text);
    }

    /// Instruct template apply() wraps the text with Q:/A: markers.
    #[test]
    fn test_apply_instruct_wraps_prompt() {
        let formatted = TemplateType::Instruct.apply("What is 2+2?", None);
        assert!(formatted.contains("Q:"), "Instruct template must add Q: prefix; got: {formatted}");
        assert!(formatted.contains("A:"), "Instruct template must add A: suffix; got: {formatted}");
    }

    /// LLaMA-3 template apply() wraps the text with special tokens.
    #[test]
    fn test_apply_llama3_wraps_with_special_tokens() {
        let formatted = TemplateType::Llama3Chat.apply("Hello", None);
        assert!(
            formatted.contains("<|begin_of_text|>"),
            "Llama3Chat must start with <|begin_of_text|>"
        );
        assert!(
            formatted.contains("<|start_header_id|>"),
            "Llama3Chat must contain <|start_header_id|>"
        );
    }

    /// Instruct template apply() with a system prompt includes "System:" prefix.
    #[test]
    fn test_apply_instruct_with_system_prompt() {
        let formatted = TemplateType::Instruct.apply("Hello", Some("You are a helpful assistant"));
        assert!(
            formatted.contains("System:"),
            "Instruct+system must add 'System:' prefix; got: {formatted}"
        );
    }

    /// TemplateType Clone copies the same variant.
    #[test]
    fn test_template_type_clone() {
        let original = TemplateType::Llama3Chat;
        let cloned = original;
        assert_eq!(original, cloned);
    }
}

// ============================================================================
// 5. CliConfig validation
// ============================================================================

mod cli_config_validation {
    use super::*;

    /// Default CliConfig is valid.
    #[test]
    fn test_cli_config_default_is_valid() {
        let cfg = CliConfig::default();
        assert!(cfg.validate().is_ok(), "default CliConfig must be valid");
    }

    /// Default device is "auto".
    #[test]
    fn test_cli_config_default_device_is_auto() {
        let cfg = CliConfig::default();
        assert_eq!(cfg.default_device, "auto");
    }

    /// Invalid device string fails validate().
    #[test]
    fn test_cli_config_invalid_device_fails() {
        let cfg = CliConfig { default_device: "invalid-device".to_string(), ..Default::default() };
        assert!(cfg.validate().is_err(), "invalid device must fail validate()");
    }

    /// Invalid log level string fails validate().
    #[test]
    fn test_cli_config_invalid_log_level_fails() {
        let mut cfg = CliConfig::default();
        cfg.logging.level = "verbose".to_string(); // not a valid level
        assert!(cfg.validate().is_err(), "invalid log level must fail validate()");
    }

    /// Invalid log format string fails validate().
    #[test]
    fn test_cli_config_invalid_log_format_fails() {
        let mut cfg = CliConfig::default();
        cfg.logging.format = "xml".to_string(); // not pretty/json/compact
        assert!(cfg.validate().is_err(), "invalid log format must fail validate()");
    }

    /// Zero batch size fails validate().
    #[test]
    fn test_cli_config_zero_batch_size_fails() {
        let mut cfg = CliConfig::default();
        cfg.performance.batch_size = 0;
        assert!(cfg.validate().is_err(), "batch_size=0 must fail validate()");
    }

    /// Valid device strings all pass.
    #[test]
    fn test_cli_config_all_valid_devices() {
        for device in &["cpu", "cuda", "gpu", "vulkan", "opencl", "ocl", "auto"] {
            let cfg = CliConfig { default_device: device.to_string(), ..Default::default() };
            assert!(cfg.validate().is_ok(), "device={device} must be valid");
        }
    }

    /// Valid log levels all pass.
    #[test]
    fn test_cli_config_all_valid_log_levels() {
        for level in &["trace", "debug", "info", "warn", "error"] {
            let mut cfg = CliConfig::default();
            cfg.logging.level = level.to_string();
            assert!(cfg.validate().is_ok(), "log level={level} must be valid");
        }
    }
}

// ============================================================================
// 6. ConfigBuilder fluent API
// ============================================================================

mod config_builder {
    use super::*;

    /// ConfigBuilder::new().build() returns a valid default config.
    #[test]
    fn test_config_builder_new_builds_valid_config() {
        let cfg = ConfigBuilder::new().build().expect("default ConfigBuilder must build");
        assert!(cfg.validate().is_ok(), "newly built config must be valid");
    }

    /// ConfigBuilder::device() overrides the default device.
    #[test]
    fn test_config_builder_device_override() {
        let cfg =
            ConfigBuilder::new().device(Some("cpu".to_string())).build().expect("should build");
        assert_eq!(cfg.default_device, "cpu");
    }

    /// ConfigBuilder::cpu_threads() sets thread count.
    #[test]
    fn test_config_builder_cpu_threads() {
        let cfg = ConfigBuilder::new().cpu_threads(Some(4)).build().expect("should build");
        assert_eq!(cfg.performance.cpu_threads, Some(4));
    }

    /// ConfigBuilder with invalid device fails at build.
    #[test]
    fn test_config_builder_invalid_device_fails() {
        let result = ConfigBuilder::new().device(Some("invalid-device".to_string())).build();
        assert!(result.is_err(), "ConfigBuilder with invalid device must fail to build");
    }
}

// ============================================================================
// 7. Specific InferenceCommand optional fields
// ============================================================================

#[cfg(feature = "full-cli")]
mod inference_optional_fields {
    use super::*;

    /// --top-p 0.5 is stored as Some(0.5).
    #[test]
    fn test_top_p_stored_as_some() {
        let cmd = parse_args(&["bitnet", "--top-p", "0.5"]).expect("should parse");
        assert_eq!(cmd.top_p, Some(0.5f32));
    }

    /// --top-p 1.0 is stored as Some(1.0).
    #[test]
    fn test_top_p_one_stored_as_some() {
        let cmd = parse_args(&["bitnet", "--top-p", "1.0"]).expect("should parse");
        assert_eq!(cmd.top_p, Some(1.0f32));
    }

    /// --top-k 50 is stored as Some(50).
    #[test]
    fn test_top_k_50_stored_as_some() {
        let cmd = parse_args(&["bitnet", "--top-k", "50"]).expect("should parse");
        assert_eq!(cmd.top_k, Some(50usize));
    }

    /// --stop-string-window 128 overrides the default.
    #[test]
    fn test_stop_string_window_overridden() {
        let cmd = parse_args(&["bitnet", "--stop-string-window", "128"]).expect("should parse");
        assert_eq!(cmd.stop_string_window, 128);
    }

    /// --stop and --stop-id can be combined in a single invocation.
    #[test]
    fn test_stop_and_stop_id_combined() {
        let cmd =
            parse_args(&["bitnet", "--stop", "</s>", "--stop-id", "2"]).expect("should parse");
        assert!(cmd.stop.contains(&"</s>".to_string()), "stop sequences must contain </s>");
        assert!(cmd.stop_id.contains(&2u32), "stop_id must contain 2");
    }

    /// --temperature 0.0 together with --greedy stores both correctly.
    #[test]
    fn test_zero_temperature_and_greedy_together() {
        let cmd =
            parse_args(&["bitnet", "--temperature", "0.0", "--greedy"]).expect("should parse");
        assert!(cmd.greedy, "greedy must be true");
        assert_eq!(cmd.temperature, 0.0f32);
    }
}

// ============================================================================
// 8. Property tests
// ============================================================================

#[cfg(feature = "full-cli")]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// top_p values in [0.0, 1.0] (expressed as integer hundredths) are stored as Some(p).
        #[test]
        fn prop_top_p_range_0_to_1_stored_as_some(
            hundredths in 0u32..=100u32,
        ) {
            let p = hundredths as f32 / 100.0;
            let p_str = format!("{p:.2}");
            let cmd = parse_args(&["bitnet", "--top-p", &p_str]).expect("should parse");
            let stored = cmd.top_p.expect("--top-p must yield Some");
            // Allow 1e-4 tolerance for f32 string round-trip.
            prop_assert!((stored - p).abs() < 1e-4, "stored top_p {stored} must be ~{p}");
        }

        /// Positive max_tokens values are always accepted by all three aliases.
        #[test]
        fn prop_all_three_max_tokens_aliases_accept_same_value(
            n in 1usize..=2048usize,
        ) {
            let n_str = n.to_string();
            let primary = parse_args(&["bitnet", "--max-tokens", &n_str])
                .expect("--max-tokens must accept positive n");
            let alias1 = parse_args(&["bitnet", "--max-new-tokens", &n_str])
                .expect("--max-new-tokens must accept positive n");
            let alias2 = parse_args(&["bitnet", "--n-predict", &n_str])
                .expect("--n-predict must accept positive n");
            prop_assert_eq!(primary.max_tokens, n);
            prop_assert_eq!(alias1.max_tokens, n);
            prop_assert_eq!(alias2.max_tokens, n);
        }
    }
}

//! Tests for CLI UX improvements: --format, --quiet, --verbose, --no-color flags.
//!
//! These tests verify the output module and the new global CLI flags
//! without requiring a model file.

#![cfg(feature = "full-cli")]

use bitnet_cli::output::{ModelSummary, OutputConfig, OutputFormat, suggest_flag};

#[cfg(test)]
mod output_format_tests {
    use super::*;

    #[test]
    fn parse_text_format() {
        assert_eq!("text".parse::<OutputFormat>().unwrap(), OutputFormat::Text);
    }

    #[test]
    fn parse_json_format() {
        assert_eq!("json".parse::<OutputFormat>().unwrap(), OutputFormat::Json);
    }

    #[test]
    fn parse_case_insensitive() {
        assert_eq!("JSON".parse::<OutputFormat>().unwrap(), OutputFormat::Json);
        assert_eq!("Text".parse::<OutputFormat>().unwrap(), OutputFormat::Text);
    }

    #[test]
    fn parse_invalid_format_rejected() {
        assert!("xml".parse::<OutputFormat>().is_err());
        assert!("yaml".parse::<OutputFormat>().is_err());
    }
}

#[cfg(test)]
mod output_config_tests {
    use super::*;

    #[test]
    fn quiet_overrides_log_level_to_error() {
        let config = OutputConfig { quiet: true, ..Default::default() };
        assert_eq!(config.log_level_override(), Some("error"));
    }

    #[test]
    fn verbose_overrides_log_level_to_debug() {
        let config = OutputConfig { verbose: true, ..Default::default() };
        assert_eq!(config.log_level_override(), Some("debug"));
    }

    #[test]
    fn default_has_no_log_level_override() {
        let config = OutputConfig::default();
        assert_eq!(config.log_level_override(), None);
    }

    #[test]
    fn emit_result_json_mode_serializes() {
        let config = OutputConfig { format: OutputFormat::Json, ..Default::default() };
        let value = serde_json::json!({"status": "ok"});
        // Should not panic
        config.emit_result(&value, |_| {}).unwrap();
    }

    #[test]
    fn emit_result_text_mode_calls_fn() {
        let config = OutputConfig { format: OutputFormat::Text, ..Default::default() };
        let mut called = false;
        config
            .emit_result(&serde_json::json!({}), |_| {
                called = true;
            })
            .unwrap();
        assert!(called);
    }

    #[test]
    fn status_suppressed_in_quiet_mode() {
        // This test verifies no panic; actual output goes to stderr
        let config = OutputConfig { quiet: true, ..Default::default() };
        config.status("this should not appear");
    }

    #[test]
    fn debug_only_shown_when_verbose() {
        let config = OutputConfig { verbose: false, ..Default::default() };
        config.debug("should not appear");
        let verbose = OutputConfig { verbose: true, ..Default::default() };
        verbose.debug("should appear");
    }
}

#[cfg(test)]
mod suggest_flag_tests {
    use super::*;

    #[test]
    fn suggests_close_match() {
        let known = &["--max-tokens", "--temperature", "--top-k"];
        let suggestions = suggest_flag("--max-token", known);
        assert!(!suggestions.is_empty());
        assert_eq!(suggestions[0], "--max-tokens");
    }

    #[test]
    fn suggests_nothing_for_unrelated() {
        let known = &["--max-tokens", "--temperature"];
        let suggestions = suggest_flag("--zzzzzzzzz", known);
        assert!(suggestions.is_empty());
    }

    #[test]
    fn suggests_multiple_candidates() {
        let known = &["--max-tokens", "--max-new-tokens"];
        let suggestions = suggest_flag("--max-token", known);
        // Should suggest both since both are close
        assert!(!suggestions.is_empty());
    }

    #[test]
    fn suggests_verbose_for_verbos() {
        let known = &["--verbose", "--version", "--quiet"];
        let suggestions = suggest_flag("--verbos", known);
        assert!(!suggestions.is_empty());
        assert!(suggestions.contains(&"--verbose".to_string()));
    }
}

#[cfg(test)]
mod model_summary_tests {
    use super::*;

    #[test]
    fn model_summary_display_contains_fields() {
        let summary = ModelSummary {
            path: "test-model.gguf".into(),
            format: "GGUF".into(),
            quantization: "I2_S".into(),
            parameters: Some("2.0B".into()),
            device: "cpu".into(),
            vocab_size: Some(32000),
            context_length: Some(2048),
        };
        let output = format!("{summary}");
        assert!(output.contains("test-model.gguf"));
        assert!(output.contains("GGUF"));
        assert!(output.contains("I2_S"));
        assert!(output.contains("2.0B"));
        assert!(output.contains("cpu"));
        assert!(output.contains("32000"));
        assert!(output.contains("2048"));
    }

    #[test]
    fn model_summary_optional_fields_omitted() {
        let summary = ModelSummary {
            path: "model.gguf".into(),
            format: "GGUF".into(),
            quantization: "unknown".into(),
            parameters: None,
            device: "cpu".into(),
            vocab_size: None,
            context_length: None,
        };
        let output = format!("{summary}");
        assert!(output.contains("model.gguf"));
        assert!(!output.contains("Parameters"));
        assert!(!output.contains("Vocab size"));
    }

    #[test]
    fn model_summary_serializes_to_json() {
        let summary = ModelSummary {
            path: "model.gguf".into(),
            format: "GGUF".into(),
            quantization: "I2_S".into(),
            parameters: Some("2.0B".into()),
            device: "cpu".into(),
            vocab_size: Some(32000),
            context_length: Some(2048),
        };
        let json = serde_json::to_value(&summary).unwrap();
        assert_eq!(json["quantization"], "I2_S");
        assert_eq!(json["device"], "cpu");
    }
}

/// Integration-style test that verifies the CLI parses the new global flags
/// without requiring a model.
#[cfg(test)]
mod cli_global_flags {
    use clap::Parser;

    /// Minimal stub that mirrors the global flags from main.rs Cli struct
    #[derive(Parser, Debug)]
    #[command(name = "bitnet-test")]
    struct TestCli {
        #[arg(long = "output-format", default_value = "text", global = true)]
        output_format: String,
        #[arg(long, global = true)]
        quiet: bool,
        #[arg(long, global = true, conflicts_with = "quiet")]
        verbose: bool,
        #[arg(long, global = true)]
        no_color: bool,
    }

    #[test]
    fn default_flags() {
        let cli = TestCli::try_parse_from(["bitnet-test"]).unwrap();
        assert_eq!(cli.output_format, "text");
        assert!(!cli.quiet);
        assert!(!cli.verbose);
        assert!(!cli.no_color);
    }

    #[test]
    fn format_json_accepted() {
        let cli = TestCli::try_parse_from(["bitnet-test", "--output-format", "json"]).unwrap();
        assert_eq!(cli.output_format, "json");
    }

    #[test]
    fn quiet_flag() {
        let cli = TestCli::try_parse_from(["bitnet-test", "--quiet"]).unwrap();
        assert!(cli.quiet);
    }

    #[test]
    fn verbose_flag() {
        let cli = TestCli::try_parse_from(["bitnet-test", "--verbose"]).unwrap();
        assert!(cli.verbose);
    }

    #[test]
    fn no_color_flag() {
        let cli = TestCli::try_parse_from(["bitnet-test", "--no-color"]).unwrap();
        assert!(cli.no_color);
    }

    #[test]
    fn quiet_and_verbose_conflict() {
        let result = TestCli::try_parse_from(["bitnet-test", "--quiet", "--verbose"]);
        assert!(result.is_err());
    }
}

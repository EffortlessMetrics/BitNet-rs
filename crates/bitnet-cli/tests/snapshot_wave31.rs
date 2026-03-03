//! Wave 31 snapshot tests for bitnet-cli.
//!
//! Covers: CLI help text stability, config output format, error message
//! formatting.

#![cfg(feature = "full-cli")]

use bitnet_cli::commands::InferenceCommand;
use bitnet_cli::config::{CliConfig, LoggingConfig, PerformanceConfig};
use bitnet_cli::exit;
use clap::{CommandFactory, Parser};

/// Minimal wrapper to get clap to generate help for InferenceCommand.
#[derive(Parser)]
struct TestCli {
    #[command(flatten)]
    cmd: InferenceCommand,
}

// ── CLI help text stability ─────────────────────────────────────────────────

#[test]
fn help_text_temperature_section() {
    let mut cmd = TestCli::command();
    let help = cmd.render_help().to_string();
    let relevant: String = help
        .lines()
        .filter(|l| l.contains("--temperature") || l.contains("TEMP"))
        .collect::<Vec<_>>()
        .join("\n");
    insta::assert_snapshot!(relevant);
}

#[test]
fn help_text_max_tokens_aliases() {
    let mut cmd = TestCli::command();
    let help = cmd.render_help().to_string();
    let relevant: String = help
        .lines()
        .filter(|l| {
            l.contains("--max-tokens")
                || l.contains("--max-new-tokens")
                || l.contains("--n-predict")
        })
        .collect::<Vec<_>>()
        .join("\n");
    insta::assert_snapshot!(relevant);
}

#[test]
fn help_text_stop_section() {
    let mut cmd = TestCli::command();
    let help = cmd.render_help().to_string();
    let relevant: String = help
        .lines()
        .filter(|l| l.contains("--stop") || l.contains("stop"))
        .collect::<Vec<_>>()
        .join("\n");
    insta::assert_snapshot!(relevant);
}

#[test]
fn help_text_greedy_deterministic() {
    let mut cmd = TestCli::command();
    let help = cmd.render_help().to_string();
    let relevant: String = help
        .lines()
        .filter(|l| l.contains("--greedy") || l.contains("--deterministic"))
        .collect::<Vec<_>>()
        .join("\n");
    insta::assert_snapshot!(relevant);
}

#[test]
fn help_text_device_section() {
    let mut cmd = TestCli::command();
    let help = cmd.render_help().to_string();
    let relevant: String = help
        .lines()
        .filter(|l| l.contains("--device") || l.contains("DEVICE"))
        .collect::<Vec<_>>()
        .join("\n");
    insta::assert_snapshot!(relevant);
}

// ── Config output format ────────────────────────────────────────────────────

#[test]
fn cli_config_default_json_wave31() {
    let cfg = CliConfig::default();
    insta::assert_json_snapshot!(cfg);
}

#[test]
fn logging_config_default_json() {
    let cfg = LoggingConfig::default();
    insta::assert_json_snapshot!(cfg);
}

#[test]
fn performance_config_default_json() {
    let cfg = PerformanceConfig::default();
    insta::assert_json_snapshot!(cfg);
}

#[test]
fn inference_command_default_debug() {
    let cmd = InferenceCommand::default();
    insta::assert_snapshot!(format!(
        "model_format={} max_tokens={} temperature={:.1} repetition_penalty={:.2}",
        cmd.model_format, cmd.max_tokens, cmd.temperature, cmd.repetition_penalty
    ));
}

#[test]
fn exit_codes_all_wave31() {
    let codes = [
        ("SUCCESS", exit::EXIT_SUCCESS),
        ("GENERIC_FAIL", exit::EXIT_GENERIC_FAIL),
        ("STRICT_TOKENIZER", exit::EXIT_STRICT_TOKENIZER),
        ("NLL_TOO_HIGH", exit::EXIT_NLL_TOO_HIGH),
        ("TAU_TOO_LOW", exit::EXIT_TAU_TOO_LOW),
        ("ARGMAX_MISMATCH", exit::EXIT_ARGMAX_MISMATCH),
        ("LN_SUSPICIOUS", exit::EXIT_LN_SUSPICIOUS),
        ("PERF_FAIL", exit::EXIT_PERF_FAIL),
        ("RSS_FAIL", exit::EXIT_RSS_FAIL),
    ];
    let formatted: Vec<String> = codes.iter().map(|(name, val)| format!("{name}={val}")).collect();
    insta::assert_debug_snapshot!(formatted);
}

// ── Error message formatting ────────────────────────────────────────────────

#[test]
fn config_validate_invalid_device() {
    let mut cfg = CliConfig::default();
    cfg.default_device = "tpu".to_string();
    let err = cfg.validate().unwrap_err();
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn config_validate_invalid_log_level() {
    let mut cfg = CliConfig::default();
    cfg.logging.level = "critical".to_string();
    let err = cfg.validate().unwrap_err();
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn config_validate_zero_batch_size() {
    let mut cfg = CliConfig::default();
    cfg.performance.batch_size = 0;
    let err = cfg.validate().unwrap_err();
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn config_validate_invalid_log_format() {
    let mut cfg = CliConfig::default();
    cfg.logging.format = "yaml".to_string();
    let err = cfg.validate().unwrap_err();
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn cli_config_default_toml_wave31() {
    let cfg = CliConfig::default();
    let toml = toml::to_string_pretty(&cfg).unwrap();
    insta::assert_snapshot!(toml);
}

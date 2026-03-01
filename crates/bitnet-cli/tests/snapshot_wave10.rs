//! Wave 10 snapshot tests for bitnet-cli.
//!
//! Covers: CliConfig defaults/serialization, exit codes, help text sections,
//! config validation errors, LoggingConfig, PerformanceConfig.

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

// -- CliConfig defaults ------------------------------------------------------

#[test]
fn cli_config_default_debug() {
    let cfg = CliConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn cli_config_default_toml() {
    let cfg = CliConfig::default();
    let toml = toml::to_string_pretty(&cfg).unwrap();
    insta::assert_snapshot!(toml);
}

#[test]
fn cli_config_default_json() {
    let cfg = CliConfig::default();
    insta::assert_json_snapshot!(cfg);
}

// -- LoggingConfig defaults --------------------------------------------------

#[test]
fn logging_config_default_debug() {
    let cfg = LoggingConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn logging_config_default_snapshot() {
    let cfg = LoggingConfig::default();
    insta::assert_snapshot!(format!(
        "level={} format={} timestamps={}",
        cfg.level, cfg.format, cfg.timestamps
    ));
}

// -- PerformanceConfig defaults ----------------------------------------------

#[test]
fn performance_config_default_debug() {
    let cfg = PerformanceConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn performance_config_default_snapshot() {
    let cfg = PerformanceConfig::default();
    insta::assert_snapshot!(format!(
        "cpu_threads={:?} batch_size={} memory_optimization={}",
        cfg.cpu_threads, cfg.batch_size, cfg.memory_optimization
    ));
}

// -- Exit codes snapshot -----------------------------------------------------

#[test]
fn exit_codes_snapshot() {
    let codes = format!(
        "SUCCESS={} GENERIC_FAIL={} STRICT_TOKENIZER={} NLL_TOO_HIGH={} TAU_TOO_LOW={} ARGMAX_MISMATCH={} LN_SUSPICIOUS={} PERF_FAIL={} RSS_FAIL={}",
        exit::EXIT_SUCCESS,
        exit::EXIT_GENERIC_FAIL,
        exit::EXIT_STRICT_TOKENIZER,
        exit::EXIT_NLL_TOO_HIGH,
        exit::EXIT_TAU_TOO_LOW,
        exit::EXIT_ARGMAX_MISMATCH,
        exit::EXIT_LN_SUSPICIOUS,
        exit::EXIT_PERF_FAIL,
        exit::EXIT_RSS_FAIL,
    );
    insta::assert_snapshot!(codes);
}

// -- Config validation errors ------------------------------------------------

#[test]
fn cli_config_validate_invalid_device_error() {
    let mut cfg = CliConfig::default();
    cfg.default_device = "quantum".to_string();
    let err = cfg.validate().unwrap_err();
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn cli_config_validate_invalid_log_level_error() {
    let mut cfg = CliConfig::default();
    cfg.logging.level = "verbose".to_string();
    let err = cfg.validate().unwrap_err();
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn cli_config_validate_invalid_log_format_error() {
    let mut cfg = CliConfig::default();
    cfg.logging.format = "xml".to_string();
    let err = cfg.validate().unwrap_err();
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn cli_config_validate_zero_batch_size_error() {
    let mut cfg = CliConfig::default();
    cfg.performance.batch_size = 0;
    let err = cfg.validate().unwrap_err();
    insta::assert_snapshot!(err.to_string());
}

// -- Help text: model/tokenizer flags ----------------------------------------

#[test]
fn help_contains_model_tokenizer_flags() {
    let mut cmd = TestCli::command();
    let help = cmd.render_help().to_string();
    let relevant: String = help
        .lines()
        .filter(|l| l.contains("--model") || l.contains("--tokenizer"))
        .collect::<Vec<_>>()
        .join("\n");
    insta::assert_snapshot!(relevant);
}

// -- Help text: output format flags ------------------------------------------

#[test]
fn help_contains_output_format_flags() {
    let mut cmd = TestCli::command();
    let help = cmd.render_help().to_string();
    let relevant: String = help
        .lines()
        .filter(|l| l.contains("--format") || l.contains("--metrics") || l.contains("--json"))
        .collect::<Vec<_>>()
        .join("\n");
    insta::assert_snapshot!(relevant);
}

// -- InferenceCommand defaults extended --------------------------------------

#[test]
fn inference_command_sampling_defaults() {
    let cmd = InferenceCommand::default();
    insta::assert_snapshot!(
        "inference_sampling_defaults",
        format!(
            "temperature={:.1} top_k={:?} top_p={:?} repetition_penalty={:.2}",
            cmd.temperature, cmd.top_k, cmd.top_p, cmd.repetition_penalty
        )
    );
}

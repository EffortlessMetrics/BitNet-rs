//! Wave 33 snapshot tests for bitnet-cli.
//!
//! Covers: CLI config defaults, logging config, performance config,
//! config validation errors, exit codes, help text sections, version format.

#![cfg(feature = "full-cli")]

use bitnet_cli::config::{CliConfig, LoggingConfig, PerformanceConfig};
use bitnet_cli::exit;

// ── CliConfig defaults ──────────────────────────────────────────────────────

#[test]
fn w33_cli_config_default_debug() {
    let cfg = CliConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w33_cli_config_default_json() {
    let cfg = CliConfig::default();
    insta::assert_json_snapshot!(cfg);
}

#[test]
fn w33_cli_config_default_device_display() {
    let cfg = CliConfig::default();
    insta::assert_snapshot!(format!("device={}", cfg.default_device));
}

#[test]
fn w33_cli_config_default_model_display() {
    let cfg = CliConfig::default();
    insta::assert_snapshot!(format!(
        "model={:?} quant={:?}",
        cfg.default_model, cfg.default_quantization
    ));
}

// ── LoggingConfig ───────────────────────────────────────────────────────────

#[test]
fn w33_logging_config_default_debug() {
    let cfg = LoggingConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w33_logging_config_default_summary() {
    let cfg = LoggingConfig::default();
    insta::assert_snapshot!(format!(
        "level={} format={} timestamps={}",
        cfg.level, cfg.format, cfg.timestamps
    ));
}

// ── PerformanceConfig ───────────────────────────────────────────────────────

#[test]
fn w33_performance_config_default_debug() {
    let cfg = PerformanceConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w33_performance_config_default_summary() {
    let cfg = PerformanceConfig::default();
    insta::assert_snapshot!(format!(
        "threads={:?} batch={} mem_opt={}",
        cfg.cpu_threads, cfg.batch_size, cfg.memory_optimization
    ));
}

// ── Config validation errors ────────────────────────────────────────────────

#[test]
fn w33_cli_config_validate_bad_device_error() {
    let mut cfg = CliConfig::default();
    cfg.default_device = "quantum".to_string();
    let err = cfg.validate().unwrap_err();
    insta::assert_snapshot!(format!("{err}"));
}

#[test]
fn w33_cli_config_validate_bad_log_level_error() {
    let mut cfg = CliConfig::default();
    cfg.logging.level = "verbose".to_string();
    let err = cfg.validate().unwrap_err();
    insta::assert_snapshot!(format!("{err}"));
}

#[test]
fn w33_cli_config_validate_bad_log_format_error() {
    let mut cfg = CliConfig::default();
    cfg.logging.format = "xml".to_string();
    let err = cfg.validate().unwrap_err();
    insta::assert_snapshot!(format!("{err}"));
}

#[test]
fn w33_cli_config_validate_zero_batch_error() {
    let mut cfg = CliConfig::default();
    cfg.performance.batch_size = 0;
    let err = cfg.validate().unwrap_err();
    insta::assert_snapshot!(format!("{err}"));
}

// ── Exit codes ──────────────────────────────────────────────────────────────

#[test]
fn w33_exit_codes_snapshot() {
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

// ── Version format ──────────────────────────────────────────────────────────

#[test]
fn w33_build_cli_version_contains_bitnet() {
    let cmd = bitnet_cli::build_cli();
    let name = cmd.get_name().to_string();
    insta::assert_snapshot!(name);
}

#[test]
fn w33_build_cli_about_text() {
    let cmd = bitnet_cli::build_cli();
    let about = cmd.get_about().map(|a| a.to_string()).unwrap_or_default();
    insta::assert_snapshot!(about);
}

// ── CLI help text ───────────────────────────────────────────────────────────

#[test]
fn w33_build_cli_subcommand_listing() {
    let cmd = bitnet_cli::build_cli();
    let subs: Vec<String> = cmd.get_subcommands().map(|s| s.get_name().to_string()).collect();
    insta::assert_debug_snapshot!(subs);
}

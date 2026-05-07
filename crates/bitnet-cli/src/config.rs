//! Configuration management for BitNet CLI.
//!
//! This module re-exports the SRP configuration core crate to keep
//! the existing `bitnet_cli::config::*` import path stable.

#[allow(unused_imports)]
pub use bitnet_cli_config_core::{
    APPLE_M4_DEVICE_LABELS_TEXT, CliConfig, ConfigBuilder, DEVICE_HELP, LEGACY_RUNTIME_DEVICE_HELP,
    LEGACY_RUNTIME_DEVICE_LABELS_TEXT, LoggingConfig, PerformanceConfig, SUPPORTED_DEVICE_LABELS,
    SUPPORTED_DEVICE_LABELS_TEXT, invalid_device_message, is_supported_device_label,
    unsupported_legacy_command_device_message,
};

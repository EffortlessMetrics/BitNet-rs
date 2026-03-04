//! Configuration management for BitNet CLI.
//!
//! This module re-exports the SRP configuration core crate to keep
//! the existing `bitnet_cli::config::*` import path stable.

#[allow(unused_imports)]
pub use bitnet_cli_config_core::{CliConfig, ConfigBuilder, LoggingConfig, PerformanceConfig};

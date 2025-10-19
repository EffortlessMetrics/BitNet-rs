//! Common types, traits, and utilities for BitNet inference
//!
//! This crate provides the foundational types and abstractions used across
//! the BitNet ecosystem, including configuration, error handling, and tensor
//! abstractions.

pub mod config;
pub mod error;
pub mod math;
pub mod strict_mode;
pub mod tensor;
pub mod types;
pub mod warn_once;

pub use config::*;
pub use error::*;
pub use math::ceil_div;
pub use strict_mode::{
    ComputationType, MissingKernelScenario, MockInferencePath, PerformanceMetrics,
    StrictModeConfig, StrictModeEnforcer,
};
pub use tensor::*;
pub use types::*;
pub use warn_once::warn_once_fn;

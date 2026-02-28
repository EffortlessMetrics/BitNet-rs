//! Common types, traits, and utilities for BitNet inference
//!
//! This crate provides the foundational types and abstractions used across
//! the BitNet ecosystem, including configuration, error handling, and tensor
//! abstractions.

pub mod backend_selection;
pub mod config;
pub mod error;
pub mod kernel_registry;
pub mod math;
pub mod memory_pool;
pub mod strict_mode;
pub mod tensor;
pub mod tensor_validation;
pub mod types;
pub mod warn_once;

pub use backend_selection::{
    BackendRequest, BackendSelectionError, BackendSelectionResult, BackendStartupSummary,
    select_backend,
};
pub use config::*;
pub use error::*;
pub use kernel_registry::{KernelBackend, KernelCapabilities, SimdLevel};
pub use math::ceil_div;
pub use strict_mode::{
    ComputationType, MissingKernelScenario, MockInferencePath, PerformanceMetrics,
    StrictModeConfig, StrictModeEnforcer,
};
pub use tensor::*;
pub use types::*;
pub use warn_once::warn_once_fn;

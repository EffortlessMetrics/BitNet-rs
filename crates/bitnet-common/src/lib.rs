//! Common types, traits, and utilities for BitNet inference
//!
//! This crate provides the foundational types and abstractions used across
//! the BitNet ecosystem, including configuration, error handling, and tensor
//! abstractions.

pub mod arch_registry;
pub mod backend_selection;
pub mod compute_pool;
pub mod config;
pub mod dtype_convert;
pub mod error_catalog;
pub mod error_context;

pub use arch_registry::{ArchDefaults, ArchitectureRegistry};
pub mod error;
pub mod kernel_registry;
pub mod memory_estimator;
pub mod memory_pool;
pub mod op_pool;
pub mod perf_profiler;
pub mod runtime_diag;
pub mod shape_inference;
pub mod shape_validator;
pub mod strict_mode;
pub mod tensor;
pub mod tensor_layout;
pub mod tensor_math;
pub mod tensor_pool;
pub mod tensor_serde;
pub mod tensor_validation;
pub mod thread_config;
pub mod token_ring;
pub mod types;

// Re-exports from microcrates for backward compatibility
pub mod math {
    pub use bitnet_math::*;
}
pub mod warn_once {
    pub use bitnet_warn_once::*;
}

pub use backend_selection::{
    BackendRequest, BackendSelectionError, BackendSelectionResult, BackendStartupSummary,
    select_backend,
};
pub use bitnet_math::ceil_div;
pub use bitnet_warn_once::warn_once;
pub use bitnet_warn_once::warn_once_fn;
pub use config::*;
pub use error::*;
pub use kernel_registry::{KernelBackend, KernelCapabilities, SimdLevel};
pub use strict_mode::{
    ComputationType, MissingKernelScenario, MockInferencePath, PerformanceMetrics,
    StrictModeConfig, StrictModeEnforcer,
};
pub use tensor::*;
pub use types::*;

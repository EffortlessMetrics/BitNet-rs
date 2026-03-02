//! # bitnet-test-support
//!
//! Shared test infrastructure for BitNet-rs: environment isolation,
//! model-path gating, and test helpers.
//!
//! This crate has **no runtime dependencies** and is designed to be safe to
//! use as a `[dev-dependencies]` entry in any crate without creating cycles.
//!
//! ## Modules
//!
//! - [`env_guard`] — `EnvGuard` (single var, RAII) and `EnvScope` (multi-var, one lock)

pub mod env_guard;

pub use bitnet_test_gating_core::{env_flag_enabled, model_path, run_e2e, run_slow_tests};
pub use env_guard::{EnvGuard, EnvScope};

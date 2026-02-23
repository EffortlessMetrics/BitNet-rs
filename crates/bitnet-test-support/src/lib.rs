//! # bitnet-test-support
//!
//! Shared test infrastructure for BitNet.rs: environment isolation,
//! model-path gating, and test helpers.
//!
//! This crate has **no runtime dependencies** and is designed to be safe to
//! use as a `[dev-dependencies]` entry in any crate without creating cycles.
//!
//! ## Modules
//!
//! - [`env_guard`] — `EnvGuard` (single var, RAII) and `EnvScope` (multi-var, one lock)

pub mod env_guard;

pub use env_guard::{EnvGuard, EnvScope};

/// Returns the model path from `BITNET_MODEL_PATH` env var, or `None` if not set.
/// Use this to gate tests that require a real GGUF model.
///
/// # Example
///
/// ```rust,ignore
/// #[test]
/// fn test_real_model() {
///     let Some(path) = bitnet_test_support::model_path() else {
///         return; // skip if no model
///     };
///     // ...use path...
/// }
/// ```
pub fn model_path() -> Option<std::path::PathBuf> {
    std::env::var("BITNET_MODEL_PATH").ok().map(Into::into)
}

/// Returns `true` if slow / integration tests should run.
/// Controlled by `BITNET_RUN_SLOW_TESTS=1`.
pub fn run_slow_tests() -> bool {
    std::env::var("BITNET_RUN_SLOW_TESTS").map(|v| v == "1").unwrap_or(false)
}

/// Returns `true` if end-to-end tests should run.
/// Controlled by `BITNET_RUN_E2E=1`.
pub fn run_e2e() -> bool {
    std::env::var("BITNET_RUN_E2E").map(|v| v == "1").unwrap_or(false)
}

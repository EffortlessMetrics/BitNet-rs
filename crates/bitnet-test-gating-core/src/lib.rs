//! # bitnet-test-gating-core
//!
//! Pure test-gating helpers for opt-in slow/E2E tests and model path discovery.

#![deny(unused_must_use)]

use std::path::PathBuf;

/// Returns the model path from `BITNET_MODEL_PATH` env var, or `None` if not set.
#[must_use]
pub fn model_path() -> Option<PathBuf> {
    std::env::var("BITNET_MODEL_PATH").ok().map(Into::into)
}

/// Returns true if an env var is exactly set to `"1"`.
#[must_use]
pub fn env_flag_enabled(key: &str) -> bool {
    std::env::var(key).map(|v| v == "1").unwrap_or(false)
}

/// Returns `true` if slow / integration tests should run.
/// Controlled by `BITNET_RUN_SLOW_TESTS=1`.
#[must_use]
pub fn run_slow_tests() -> bool {
    env_flag_enabled("BITNET_RUN_SLOW_TESTS")
}

/// Returns `true` if end-to-end tests should run.
/// Controlled by `BITNET_RUN_E2E=1`.
#[must_use]
pub fn run_e2e() -> bool {
    env_flag_enabled("BITNET_RUN_E2E")
}

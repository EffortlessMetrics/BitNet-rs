//! Safe environment variable management for tests
//!
//! Re-exports the shared EnvGuard from workspace test support.

#[allow(clippy::all)]
mod env_guard_impl {
    // Path resolution: CARGO_MANIFEST_DIR = /path/to/crates/bitnet-models
    // We need to go up 3 levels: bitnet-models -> crates -> repo_root, then into tests/support
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/support/env_guard.rs"));
}

#[allow(unused_imports)]
pub use env_guard_impl::{EnvGuard, EnvScope};

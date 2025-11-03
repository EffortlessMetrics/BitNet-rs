//! Safe environment variable management for tests
//!
//! Re-exports the shared EnvGuard from workspace test support.
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/support/env_guard.rs"));

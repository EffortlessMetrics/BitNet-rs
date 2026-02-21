//! Shared startup contracts for binaries that need a consistent feature + BDD profile view.
//!
//! The contract API is intentionally small and stable:
//! - It resolves scenario/environment using current process environment variables.
//! - It builds an `ActiveProfile` via `bitnet-runtime-profile`.
//! - It compares the active profile to a curated grid row and reports compatibility.

pub use bitnet_startup_contract::*;

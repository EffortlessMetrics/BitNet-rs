//! FFI Build Hygiene
//!
//! This module re-exports FFI build helpers from `xtask-build-helper`.
//! The actual implementation is in a separate crate to avoid cyclic dependencies
//! in the build graph.
//!
//! # Issue #469 AC6
//!
//! This module implements AC6 FFI build hygiene consolidation.

// Re-export all public items from xtask-build-helper
pub use xtask_build_helper::*;

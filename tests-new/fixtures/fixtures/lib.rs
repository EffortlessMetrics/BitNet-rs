//! Test fixtures library for BitNet.rs neural network components
//!
//! This module provides the fixture infrastructure for comprehensive testing
//! of BitNet.rs components supporting Issue #218.

// Include the main module
#[path = "mod.rs"]
mod fixtures_mod;

// Re-export from the main module
pub use fixtures_mod::*;
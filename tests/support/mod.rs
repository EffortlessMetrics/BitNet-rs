//! Shared test utilities and helpers
//!
//! This module provides common test infrastructure used across the workspace.

pub mod backend_helpers;
pub mod env_guard;
pub mod mock_fixtures;
pub mod platform;
pub mod platform_utils;

#[cfg(test)]
mod runtime_detection_warning_tests;

//! Test Fixtures Module for Issue #260 Mock Elimination
//!
//! Simple fixture loading and management for BitNet-rs neural network
//! quantization testing. Provides feature-gated access to comprehensive test
//! data including I2S/TL1/TL2 quantization, GGUF models, cross-validation,
//! and mock detection scenarios.

// Re-export all fixture modules
pub mod crossval;
pub mod models;
pub mod quantization;
pub mod strict_mode;

// Re-export fixture loader for convenient access
pub mod fixture_loader;
#[allow(unused_imports)]
pub use fixture_loader::*;

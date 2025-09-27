//! Cross-Validation Test Fixtures Module
//!
//! Provides test data for validating Rust implementations against
//! C++ reference implementations for Issue #260 mock elimination.

#![allow(unexpected_cfgs)]

#[cfg(feature = "crossval")]
pub mod cpp_reference_data;

// Re-export key types for convenience (feature-gated)
#[cfg(feature = "crossval")]
pub use cpp_reference_data::{
    CppReferenceFixture, CrossValInputData, CrossValOutput, DeviceContext, PerformanceMetrics,
    QuantizationMethod, QuantizationParameters, StatisticalTestType, ToleranceSpecification,
    ValidationMetadata, ValidationResult,
};

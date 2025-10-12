//! Model Test Fixtures Module
//!
//! Provides test data for QLinear layers, GGUF models, and layer
//! replacement scenarios for Issue #260 mock elimination validation.

#![allow(unused_imports)]

pub mod qlinear_layer_data;

// Re-export key types for convenience
pub use qlinear_layer_data::{
    FallbackDetectionData, GgufModelFixture, LayerReplacementScenario, LayerType, MockFingerprint,
    MockLinearLayer, PerformanceTarget, QLinearLayerFixture, QuantizationType, QuantizedWeightData,
};

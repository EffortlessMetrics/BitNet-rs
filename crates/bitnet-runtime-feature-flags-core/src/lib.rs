//! Core feature-to-`FeatureSet` conversion for BitNet runtime profile contracts.
//!
//! This crate intentionally owns only the normalized feature activation model.
//! The parent crate (`bitnet-runtime-feature-flags`) is responsible for
//! wiring Rust `cfg!(feature = "...")` values into a concrete activation set.

use bitnet_bdd_grid_core::{BitnetFeature, FeatureSet};

/// Structured feature activation snapshot used by runtime contracts.
#[derive(Debug, Default, Clone, Copy)]
pub struct FeatureActivation {
    /// CPU pipeline enablement requested.
    pub cpu: bool,
    /// GPU pipeline enablement requested.
    pub gpu: bool,
    /// CUDA backend enablement requested.
    pub cuda: bool,
    /// Inference feature enablement requested.
    pub inference: bool,
    /// Kernel feature enablement requested.
    pub kernels: bool,
    /// Tokenizer feature enablement requested.
    pub tokenizers: bool,
    /// Quantization feature enablement requested.
    pub quantization: bool,
    /// CLI feature enablement requested.
    pub cli: bool,
    /// Server feature enablement requested.
    pub server: bool,
    /// FFI feature enablement requested.
    pub ffi: bool,
    /// Python binding feature enablement requested.
    pub python: bool,
    /// WASM feature enablement requested.
    pub wasm: bool,
    /// Cross-validation feature enablement requested.
    pub crossval: bool,
    /// Trace feature enablement requested.
    pub trace: bool,
    /// IQ2-FFI feature enablement requested.
    pub iq2s_ffi: bool,
    /// C++ FFI feature enablement requested.
    pub cpp_ffi: bool,
    /// Fixtures feature enablement requested.
    pub fixtures: bool,
    /// Reporting feature enablement requested.
    pub reporting: bool,
    /// Trend feature enablement requested.
    pub trend: bool,
    /// Integration tests feature enablement requested.
    pub integration_tests: bool,
}

impl FeatureActivation {
    /// Convert the structured activations into canonical feature labels.
    pub fn to_labels(self) -> Vec<String> {
        feature_labels_from_activation(self)
    }
}

/// Build a canonical `FeatureSet` from explicit runtime-gated activation flags.
pub fn active_features_from_activation(activation: FeatureActivation) -> FeatureSet {
    let mut features = FeatureSet::new();

    if activation.cpu {
        features.insert(BitnetFeature::Cpu);
        features.insert(BitnetFeature::Inference);
        features.insert(BitnetFeature::Kernels);
        features.insert(BitnetFeature::Tokenizers);
    }

    if activation.gpu {
        features.insert(BitnetFeature::Gpu);
        features.insert(BitnetFeature::Inference);
        features.insert(BitnetFeature::Kernels);
    }

    if activation.cuda {
        features.insert(BitnetFeature::Cuda);
        features.insert(BitnetFeature::Gpu);
    }

    if activation.inference {
        features.insert(BitnetFeature::Inference);
    }

    if activation.kernels {
        features.insert(BitnetFeature::Kernels);
    }

    if activation.tokenizers {
        features.insert(BitnetFeature::Tokenizers);
    }

    if activation.quantization {
        features.insert(BitnetFeature::Quantization);
    }

    if activation.cli {
        features.insert(BitnetFeature::Cli);
    }

    if activation.server {
        features.insert(BitnetFeature::Server);
    }

    if activation.ffi {
        features.insert(BitnetFeature::Ffi);
    }

    if activation.python {
        features.insert(BitnetFeature::Python);
    }

    if activation.wasm {
        features.insert(BitnetFeature::Wasm);
    }

    if activation.crossval {
        features.insert(BitnetFeature::CrossValidation);
    }

    if activation.trace {
        features.insert(BitnetFeature::Trace);
    }

    if activation.iq2s_ffi {
        features.insert(BitnetFeature::Iq2sFfi);
    }

    if activation.cpp_ffi {
        features.insert(BitnetFeature::CppFfi);
    }

    if activation.fixtures {
        features.insert(BitnetFeature::Fixtures);
    }

    if activation.reporting {
        features.insert(BitnetFeature::Reporting);
    }

    if activation.trend {
        features.insert(BitnetFeature::Trend);
    }

    if activation.integration_tests {
        features.insert(BitnetFeature::IntegrationTests);
    }

    features
}

/// Stable helper for presenting active canonical feature labels for the provided
/// activation configuration.
pub fn feature_labels_from_activation(activation: FeatureActivation) -> Vec<String> {
    active_features_from_activation(activation).labels()
}

/// Stable helper for presenting active canonical feature labels in a compact
/// log-friendly format for the provided activation configuration.
pub fn feature_line_from_activation(activation: FeatureActivation) -> String {
    let labels = feature_labels_from_activation(activation);
    if labels.is_empty() {
        "features: none".to_string()
    } else {
        format!("features: {}", labels.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_feature_set_from_activation() {
        let activation = FeatureActivation {
            cpu: true,
            inference: true,
            kernels: true,
            tokenizers: true,
            ..Default::default()
        };

        let features = active_features_from_activation(activation);
        assert!(features.contains(BitnetFeature::Cpu));
        assert!(features.contains(BitnetFeature::Inference));
        assert!(features.contains(BitnetFeature::Kernels));
        assert!(features.contains(BitnetFeature::Tokenizers));
    }

    #[test]
    fn feature_line_is_empty_when_no_features() {
        let activation = FeatureActivation::default();
        assert_eq!(feature_line_from_activation(activation), "features: none");
    }
}

//! Compatibility façade over `bitnet-runtime-feature-flags-core`.
//!
//! This crate preserves the previous public surface and keeps all `cfg(feature =
//! "...")` wiring in one place.

use bitnet_bdd_grid_core::FeatureSet;
pub use bitnet_runtime_feature_flags_core::{
    FeatureActivation, active_features_from_activation, feature_labels_from_activation,
    feature_line_from_activation,
};

/// Detect active feature flags and project them into a canonical [`FeatureSet`].
pub fn active_features() -> FeatureSet {
    active_features_from_activation(activation_from_cfg())
}

/// Return active feature labels in canonical stable order.
pub fn feature_labels() -> Vec<String> {
    feature_labels_from_activation(activation_from_cfg())
}

/// Return a compact, human-readable feature line.
pub fn feature_line() -> String {
    feature_line_from_activation(activation_from_cfg())
}

fn activation_from_cfg() -> FeatureActivation {
    FeatureActivation {
        cpu: cfg!(feature = "cpu"),
        gpu: cfg!(feature = "gpu"),
        cuda: cfg!(feature = "cuda"),
        inference: cfg!(feature = "inference"),
        kernels: cfg!(feature = "kernels"),
        tokenizers: cfg!(feature = "tokenizers"),
        quantization: cfg!(feature = "quantization"),
        cli: cfg!(feature = "cli"),
        server: cfg!(feature = "server"),
        ffi: cfg!(feature = "ffi"),
        python: cfg!(feature = "python"),
        wasm: cfg!(feature = "wasm"),
        crossval: cfg!(feature = "crossval"),
        trace: cfg!(feature = "trace"),
        iq2s_ffi: cfg!(feature = "iq2s-ffi"),
        cpp_ffi: cfg!(feature = "cpp-ffi"),
        fixtures: cfg!(feature = "fixtures"),
        reporting: cfg!(feature = "reporting"),
        trend: cfg!(feature = "trend"),
        integration_tests: cfg!(feature = "integration-tests"),
    }
}

#[cfg(test)]
mod tests {
    use super::{active_features, feature_labels, feature_line};

    #[test]
    fn active_features_default_is_empty() {
        let features = active_features();
        let labels = feature_labels();

        assert_eq!(features.labels(), labels);
        assert!(feature_line().contains("features"));
    }
}

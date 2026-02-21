//! Compile-time feature discovery for BitNet profile contracts.

use bitnet_bdd_grid::{BitnetFeature, FeatureSet};

/// Detect active feature flags and project them into a canonical [`FeatureSet`].
pub fn active_features() -> FeatureSet {
    let mut features = FeatureSet::new();

    if cfg!(feature = "cpu") {
        features.insert(BitnetFeature::Cpu);
        features.insert(BitnetFeature::Inference);
        features.insert(BitnetFeature::Kernels);
        features.insert(BitnetFeature::Tokenizers);
    }

    if cfg!(feature = "gpu") {
        features.insert(BitnetFeature::Gpu);
        features.insert(BitnetFeature::Inference);
        features.insert(BitnetFeature::Kernels);
    }

    if cfg!(feature = "cuda") {
        features.insert(BitnetFeature::Cuda);
        features.insert(BitnetFeature::Gpu);
    }

    if cfg!(feature = "inference") {
        features.insert(BitnetFeature::Inference);
    }

    if cfg!(feature = "kernels") {
        features.insert(BitnetFeature::Kernels);
    }

    if cfg!(feature = "tokenizers") {
        features.insert(BitnetFeature::Tokenizers);
    }

    if cfg!(feature = "quantization") {
        features.insert(BitnetFeature::Quantization);
    }

    if cfg!(feature = "cli") {
        features.insert(BitnetFeature::Cli);
    }

    if cfg!(feature = "server") {
        features.insert(BitnetFeature::Server);
    }

    if cfg!(feature = "ffi") {
        features.insert(BitnetFeature::Ffi);
    }

    if cfg!(feature = "python") {
        features.insert(BitnetFeature::Python);
    }

    if cfg!(feature = "wasm") {
        features.insert(BitnetFeature::Wasm);
    }

    if cfg!(feature = "crossval") {
        features.insert(BitnetFeature::CrossValidation);
    }

    if cfg!(feature = "trace") {
        features.insert(BitnetFeature::Trace);
    }

    if cfg!(feature = "iq2s-ffi") {
        features.insert(BitnetFeature::Iq2sFfi);
    }

    if cfg!(feature = "cpp-ffi") {
        features.insert(BitnetFeature::CppFfi);
    }

    if cfg!(feature = "fixtures") {
        features.insert(BitnetFeature::Fixtures);
    }

    if cfg!(feature = "reporting") {
        features.insert(BitnetFeature::Reporting);
    }

    if cfg!(feature = "trend") {
        features.insert(BitnetFeature::Trend);
    }

    if cfg!(feature = "integration-tests") {
        features.insert(BitnetFeature::IntegrationTests);
    }

    features
}

/// Return active feature labels in canonical stable order.
pub fn feature_labels() -> Vec<String> {
    active_features().labels()
}

/// Return a compact, human-readable feature line.
pub fn feature_line() -> String {
    let labels = feature_labels();
    if labels.is_empty() {
        "features: none".to_string()
    } else {
        format!("features: {}", labels.join(", "))
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

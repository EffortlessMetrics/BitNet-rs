//! Property-based tests for `bitnet-runtime-feature-flags-core`.
//!
//! Tests the `FeatureActivation → FeatureSet` conversion invariants.

use bitnet_bdd_grid_core::BitnetFeature;
use bitnet_runtime_feature_flags_core::{
    FeatureActivation, active_features_from_activation, feature_line_from_activation,
};
use proptest::prelude::*;

proptest! {
    /// When `cuda` is true, `gpu` must always be present (cuda ⇒ gpu).
    #[test]
    fn cuda_implies_gpu(any_bool: bool) {
        let activation = FeatureActivation { cuda: true, gpu: any_bool, ..Default::default() };
        let features = active_features_from_activation(activation);
        prop_assert!(features.contains(BitnetFeature::Gpu));
        prop_assert!(features.contains(BitnetFeature::Cuda));
    }

    /// When `cpu` is true, inference, kernels, and tokenizers are always present.
    #[test]
    fn cpu_implies_inference_kernels_tokenizers(any_bool: bool) {
        let activation = FeatureActivation { cpu: true, inference: any_bool, ..Default::default() };
        let features = active_features_from_activation(activation);
        prop_assert!(features.contains(BitnetFeature::Cpu));
        prop_assert!(features.contains(BitnetFeature::Inference));
        prop_assert!(features.contains(BitnetFeature::Kernels));
        prop_assert!(features.contains(BitnetFeature::Tokenizers));
    }

    /// `feature_line_from_activation` always starts with "features: ".
    #[test]
    fn feature_line_always_starts_with_prefix(
        cpu in any::<bool>(),
        gpu in any::<bool>(),
        cuda in any::<bool>(),
    ) {
        let activation = FeatureActivation { cpu, gpu, cuda, ..Default::default() };
        let line = feature_line_from_activation(activation);
        prop_assert!(line.starts_with("features: "), "got: {}", line);
    }

    /// `to_labels()` on a default `FeatureActivation` always returns an empty vec.
    #[test]
    fn default_activation_has_no_labels(_unused in Just(())) {
        let activation = FeatureActivation::default();
        let labels = activation.to_labels();
        prop_assert!(labels.is_empty(), "expected empty, got: {:?}", labels);
    }
}

#[test]
fn feature_line_empty_activation_is_none() {
    let line = feature_line_from_activation(FeatureActivation::default());
    assert_eq!(line, "features: none");
}

#[test]
fn cuda_activation_includes_gpu_label() {
    let activation = FeatureActivation { cuda: true, ..Default::default() };
    let labels = activation.to_labels();
    assert!(labels.iter().any(|l| l == "gpu"), "labels: {:?}", labels);
    assert!(labels.iter().any(|l| l == "cuda"), "labels: {:?}", labels);
}

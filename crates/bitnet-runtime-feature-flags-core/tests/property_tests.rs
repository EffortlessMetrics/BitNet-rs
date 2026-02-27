//! Property-based tests for `bitnet-runtime-feature-flags-core`.
//!
//! Tests the `FeatureActivation → FeatureSet` conversion invariants.

use bitnet_bdd_grid_core::BitnetFeature;
use bitnet_runtime_feature_flags_core::{
    FeatureActivation, active_features_from_activation, feature_labels_from_activation,
    feature_line_from_activation,
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

proptest! {
    /// Enabling `gpu` (without cuda) places "gpu" in the label list.
    #[test]
    fn prop_gpu_activation_implies_gpu_label(any_bool: bool) {
        let activation = FeatureActivation { gpu: true, cpu: any_bool, ..Default::default() };
        let labels = feature_labels_from_activation(activation);
        prop_assert!(
            labels.contains(&"gpu".to_string()),
            "gpu=true but 'gpu' absent from labels: {:?}", labels
        );
    }

    /// All labels returned by `feature_labels_from_activation` are non-empty strings.
    #[test]
    fn prop_all_labels_nonempty(
        cpu in any::<bool>(),
        gpu in any::<bool>(),
        cuda in any::<bool>(),
        inference in any::<bool>(),
    ) {
        let activation = FeatureActivation { cpu, gpu, cuda, inference, ..Default::default() };
        for label in feature_labels_from_activation(activation) {
            prop_assert!(!label.is_empty(), "label must not be empty");
        }
    }

    /// CPU and GPU features can coexist; both labels appear when both are enabled.
    #[test]
    fn prop_cpu_and_gpu_can_coexist(_unused in Just(())) {
        let activation = FeatureActivation { cpu: true, gpu: true, ..Default::default() };
        let labels = feature_labels_from_activation(activation);
        prop_assert!(labels.contains(&"cpu".to_string()), "labels: {:?}", labels);
        prop_assert!(labels.contains(&"gpu".to_string()), "labels: {:?}", labels);
    }

    /// The label vector for any activation serializes to valid JSON.
    ///
    /// `Vec<String>` is always serializable; this guards against unexpected
    /// control characters or non-UTF-8 content slipping into label strings.
    #[test]
    fn prop_labels_serialize_to_valid_json(
        cpu in any::<bool>(),
        gpu in any::<bool>(),
        cuda in any::<bool>(),
    ) {
        let activation = FeatureActivation { cpu, gpu, cuda, ..Default::default() };
        let labels = feature_labels_from_activation(activation);
        let json = serde_json::to_string(&labels)
            .expect("Vec<String> labels must always serialize to JSON");
        prop_assert!(
            json.starts_with('['),
            "serialized labels must be a JSON array, got: {json:?}"
        );
    }

    /// `FeatureActivation` debug string is always non-empty.
    #[test]
    fn prop_feature_activation_debug_is_nonempty(
        cpu in any::<bool>(),
        gpu in any::<bool>(),
    ) {
        let activation = FeatureActivation { cpu, gpu, ..Default::default() };
        let debug = format!("{activation:?}");
        prop_assert!(!debug.is_empty(), "Debug output of FeatureActivation must not be empty");
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

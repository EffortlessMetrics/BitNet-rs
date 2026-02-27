use bitnet_runtime_feature_flags::{
    FeatureActivation, active_features_from_activation, feature_labels_from_activation,
    feature_line_from_activation,
};
use proptest::prelude::*;

prop_compose! {
    fn arb_activation()(
        cpu in any::<bool>(),
        gpu in any::<bool>(),
        cuda in any::<bool>(),
        inference in any::<bool>(),
        kernels in any::<bool>(),
        tokenizers in any::<bool>(),
        quantization in any::<bool>(),
        cli in any::<bool>(),
        server in any::<bool>(),
        ffi in any::<bool>(),
        python in any::<bool>(),
        wasm in any::<bool>(),
        crossval in any::<bool>(),
        trace in any::<bool>(),
        iq2s_ffi in any::<bool>(),
        cpp_ffi in any::<bool>(),
        fixtures in any::<bool>(),
        reporting in any::<bool>(),
        trend in any::<bool>(),
        integration_tests in any::<bool>(),
    ) -> FeatureActivation {
        FeatureActivation {
            cpu, gpu, cuda, inference, kernels, tokenizers, quantization, cli, server, ffi,
            python, wasm, crossval, trace, iq2s_ffi, cpp_ffi, fixtures, reporting, trend,
            integration_tests,
        }
    }
}

proptest! {
    /// `feature_line_from_activation` always produces a string starting with "features: ".
    #[test]
    fn feature_line_always_has_prefix(activation in arb_activation()) {
        let line = feature_line_from_activation(activation);
        prop_assert!(
            line.starts_with("features: "),
            "unexpected prefix in line: {:?}", line
        );
    }

    /// Labels from the `FeatureSet` and from the dedicated label helper are identical.
    #[test]
    fn labels_consistent_between_feature_set_and_helper(activation in arb_activation()) {
        let from_set = active_features_from_activation(activation).labels();
        let direct = feature_labels_from_activation(activation);
        prop_assert_eq!(from_set, direct);
    }

    /// Enabling `cpu` always places "cpu" in the label list.
    #[test]
    fn cpu_activation_implies_cpu_label(mut activation in arb_activation()) {
        activation.cpu = true;
        let labels = feature_labels_from_activation(activation);
        prop_assert!(
            labels.contains(&"cpu".to_string()),
            "cpu=true but 'cpu' absent from labels: {:?}", labels
        );
    }

    /// Enabling `cuda` always places "gpu" in the label list (cuda ⟹ gpu normalization).
    #[test]
    fn cuda_activation_implies_gpu_label(mut activation in arb_activation()) {
        activation.cuda = true;
        let labels = feature_labels_from_activation(activation);
        prop_assert!(
            labels.contains(&"gpu".to_string()),
            "cuda=true but 'gpu' absent from labels: {:?}", labels
        );
    }

    /// Every label reported by the helper appears verbatim inside `feature_line`.
    #[test]
    fn feature_line_contains_all_labels(activation in arb_activation()) {
        let labels = feature_labels_from_activation(activation);
        let line = feature_line_from_activation(activation);
        for label in &labels {
            prop_assert!(
                line.contains(label.as_str()),
                "label {:?} missing from line {:?}", label, line
            );
        }
    }
}

proptest! {
    /// Enabling `gpu` (without cuda) still places "gpu" in the label list.
    #[test]
    fn prop_gpu_activation_implies_gpu_label(mut activation in arb_activation()) {
        activation.gpu = true;
        let labels = feature_labels_from_activation(activation);
        prop_assert!(
            labels.contains(&"gpu".to_string()),
            "gpu=true but 'gpu' absent from labels: {:?}", labels
        );
    }

    /// All labels in the returned vector are non-empty strings.
    #[test]
    fn prop_all_labels_nonempty(activation in arb_activation()) {
        for label in feature_labels_from_activation(activation) {
            prop_assert!(!label.is_empty(), "label must not be empty");
        }
    }

    /// The default (all-false) activation always yields the literal "features: none".
    #[test]
    fn prop_default_activation_yields_none_line(_seed in 0u32..1u32) {
        let line = feature_line_from_activation(FeatureActivation::default());
        prop_assert_eq!(line, "features: none");
    }

    /// The label vector for any activation serializes to valid JSON.
    ///
    /// `Vec<String>` is always serializable; this property guards against
    /// unexpected control characters or non-UTF-8 content in label strings.
    #[test]
    fn prop_labels_serialize_to_valid_json(activation in arb_activation()) {
        let labels = feature_labels_from_activation(activation);
        let json = serde_json::to_string(&labels)
            .expect("Vec<String> labels must always serialize to JSON");
        prop_assert!(
            json.starts_with('['),
            "serialized labels must be a JSON array, got: {json:?}"
        );
    }

    /// Feature activation debug string is always non-empty.
    #[test]
    fn prop_feature_activation_debug_is_nonempty(activation in arb_activation()) {
        let debug = format!("{activation:?}");
        prop_assert!(!debug.is_empty(), "Debug output of FeatureActivation must not be empty");
    }
}

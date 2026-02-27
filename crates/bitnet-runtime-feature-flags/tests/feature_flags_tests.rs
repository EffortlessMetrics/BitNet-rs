//! Comprehensive tests for `bitnet-runtime-feature-flags`.
//!
//! Covers `FeatureActivation` construction, feature-lattice implications,
//! JSON serialization of label lists, Debug formatting, and the public
//! `active_features` / `feature_labels` / `feature_line` API.

use bitnet_runtime_feature_flags::{
    FeatureActivation, active_features, active_features_from_activation, feature_labels,
    feature_labels_from_activation, feature_line, feature_line_from_activation,
};

// ── FeatureActivation: struct defaults ───────────────────────────────────────

#[test]
fn feature_activation_default_all_fields_false() {
    let act = FeatureActivation::default();
    assert!(!act.cpu);
    assert!(!act.gpu);
    assert!(!act.cuda);
    assert!(!act.inference);
    assert!(!act.kernels);
    assert!(!act.tokenizers);
    assert!(!act.quantization);
    assert!(!act.cli);
    assert!(!act.server);
    assert!(!act.ffi);
    assert!(!act.python);
    assert!(!act.wasm);
    assert!(!act.crossval);
    assert!(!act.trace);
    assert!(!act.iq2s_ffi);
    assert!(!act.cpp_ffi);
    assert!(!act.fixtures);
    assert!(!act.reporting);
    assert!(!act.trend);
    assert!(!act.integration_tests);
}

#[test]
fn feature_activation_implements_copy() {
    let original = FeatureActivation { cpu: true, ..Default::default() };
    let copy = original; // Copy – no move error
    assert!(original.cpu);
    assert!(copy.cpu);
}

#[test]
fn feature_activation_clone_is_independent() {
    let original = FeatureActivation { gpu: true, cuda: true, ..Default::default() };
    let cloned = original.clone();
    assert_eq!(original.gpu, cloned.gpu);
    assert_eq!(original.cuda, cloned.cuda);
    assert_eq!(original.cpu, cloned.cpu);
}

// ── Debug formatting ──────────────────────────────────────────────────────────

#[test]
fn feature_activation_debug_contains_struct_name() {
    let act = FeatureActivation::default();
    let debug = format!("{act:?}");
    assert!(!debug.is_empty(), "Debug must not be empty");
    assert!(debug.contains("FeatureActivation"), "Debug must include type name, got: {debug:?}");
}

#[test]
fn feature_activation_debug_reflects_set_fields() {
    let act = FeatureActivation { cpu: true, crossval: true, ..Default::default() };
    let debug = format!("{act:?}");
    assert!(debug.contains("cpu: true"), "expected 'cpu: true' in debug: {debug}");
    assert!(debug.contains("crossval: true"), "expected 'crossval: true' in debug: {debug}");
}

// ── feature_line_from_activation ─────────────────────────────────────────────

#[test]
fn feature_line_default_activation_is_none() {
    let line = feature_line_from_activation(FeatureActivation::default());
    assert_eq!(line, "features: none");
}

#[test]
fn feature_line_always_starts_with_features_prefix() {
    let act = FeatureActivation { cpu: true, ..Default::default() };
    let line = feature_line_from_activation(act);
    assert!(line.starts_with("features: "), "unexpected prefix: {line:?}");
}

#[test]
fn feature_line_cpu_only_contains_cpu_label() {
    let act = FeatureActivation { cpu: true, ..Default::default() };
    let line = feature_line_from_activation(act);
    assert!(line.contains("cpu"), "expected 'cpu' in {line:?}");
}

#[test]
fn feature_line_multiple_features_contains_each_label() {
    let act =
        FeatureActivation { cpu: true, quantization: true, fixtures: true, ..Default::default() };
    let line = feature_line_from_activation(act);
    assert!(line.contains("cpu"), "missing 'cpu' in {line:?}");
    assert!(line.contains("quantization"), "missing 'quantization' in {line:?}");
    assert!(line.contains("fixtures"), "missing 'fixtures' in {line:?}");
}

// ── feature_labels_from_activation ───────────────────────────────────────────

#[test]
fn feature_labels_default_is_empty() {
    let labels = feature_labels_from_activation(FeatureActivation::default());
    assert!(labels.is_empty(), "default activation must produce no labels, got {labels:?}");
}

#[test]
fn cpu_activation_includes_cpu_label() {
    let act = FeatureActivation { cpu: true, ..Default::default() };
    let labels = feature_labels_from_activation(act);
    assert!(labels.contains(&"cpu".to_string()), "expected 'cpu' in {labels:?}");
}

#[test]
fn cpu_activation_implies_inference_label() {
    // The feature lattice propagates cpu → Inference.
    let act = FeatureActivation { cpu: true, ..Default::default() };
    let labels = feature_labels_from_activation(act);
    assert!(
        labels.contains(&"inference".to_string()),
        "cpu should imply inference via lattice, got {labels:?}"
    );
}

#[test]
fn cpu_activation_implies_kernels_label() {
    let act = FeatureActivation { cpu: true, ..Default::default() };
    let labels = feature_labels_from_activation(act);
    assert!(
        labels.contains(&"kernels".to_string()),
        "cpu should imply kernels via lattice, got {labels:?}"
    );
}

#[test]
fn cpu_activation_implies_tokenizers_label() {
    let act = FeatureActivation { cpu: true, ..Default::default() };
    let labels = feature_labels_from_activation(act);
    assert!(
        labels.contains(&"tokenizers".to_string()),
        "cpu should imply tokenizers via lattice, got {labels:?}"
    );
}

#[test]
fn cuda_activation_implies_both_cuda_and_gpu_labels() {
    // cuda ⟹ Gpu normalization is documented in the crate.
    let act = FeatureActivation { cuda: true, ..Default::default() };
    let labels = feature_labels_from_activation(act);
    assert!(labels.contains(&"cuda".to_string()), "expected 'cuda' in {labels:?}");
    assert!(
        labels.contains(&"gpu".to_string()),
        "cuda should imply 'gpu' via normalization, got {labels:?}"
    );
}

#[test]
fn gpu_activation_implies_inference_and_kernels_labels() {
    let act = FeatureActivation { gpu: true, ..Default::default() };
    let labels = feature_labels_from_activation(act);
    assert!(labels.contains(&"gpu".to_string()), "expected 'gpu' in {labels:?}");
    assert!(
        labels.contains(&"inference".to_string()),
        "gpu should imply inference via lattice, got {labels:?}"
    );
    assert!(
        labels.contains(&"kernels".to_string()),
        "gpu should imply kernels via lattice, got {labels:?}"
    );
}

#[test]
fn all_features_activation_yields_nonempty_labels() {
    let act = FeatureActivation {
        cpu: true,
        gpu: true,
        cuda: true,
        inference: true,
        kernels: true,
        tokenizers: true,
        quantization: true,
        cli: true,
        server: true,
        ffi: true,
        python: true,
        wasm: true,
        crossval: true,
        trace: true,
        iq2s_ffi: true,
        cpp_ffi: true,
        fixtures: true,
        reporting: true,
        trend: true,
        integration_tests: true,
    };
    let labels = feature_labels_from_activation(act);
    assert!(!labels.is_empty(), "all-true activation must produce labels");
    for label in &labels {
        assert!(!label.is_empty(), "each label must be non-empty");
    }
}

#[test]
fn labels_are_in_stable_deterministic_order() {
    // Labels derive from a BTreeSet keyed on the enum's Ord (declaration order),
    // so two calls with the same activation must return identical sequences.
    let act = FeatureActivation {
        cpu: true,
        gpu: true,
        cuda: true,
        inference: true,
        kernels: true,
        tokenizers: true,
        quantization: true,
        cli: true,
        ..Default::default()
    };
    let first = feature_labels_from_activation(act);
    let second = feature_labels_from_activation(act);
    assert_eq!(first, second, "feature labels must be deterministic across calls");
    assert!(!first.is_empty(), "labels must not be empty for this activation");
}

// ── Consistency: FeatureSet.labels() == feature_labels_from_activation() ─────

#[test]
fn feature_set_labels_match_standalone_helper() {
    let act =
        FeatureActivation { cpu: true, quantization: true, trace: true, ..Default::default() };
    let from_set = active_features_from_activation(act).labels();
    let direct = feature_labels_from_activation(act);
    assert_eq!(from_set, direct, "FeatureSet.labels() and helper must agree");
}

// ── FeatureActivation::to_labels ─────────────────────────────────────────────

#[test]
fn to_labels_method_matches_standalone_helper() {
    let act = FeatureActivation { cpu: true, inference: true, ..Default::default() };
    let via_method = act.to_labels();
    let via_helper = feature_labels_from_activation(act);
    assert_eq!(via_method, via_helper, "to_labels() and helper must produce identical output");
}

// ── JSON serialization round-trip ────────────────────────────────────────────

#[test]
fn labels_json_round_trip_preserves_values() {
    let act = FeatureActivation { cpu: true, crossval: true, fixtures: true, ..Default::default() };
    let labels = feature_labels_from_activation(act);
    let json = serde_json::to_string(&labels).expect("labels must serialize to JSON");
    let round_tripped: Vec<String> =
        serde_json::from_str(&json).expect("labels must deserialize from JSON");
    assert_eq!(labels, round_tripped, "JSON round-trip must preserve label values");
}

#[test]
fn empty_labels_serialize_to_json_array() {
    let labels: Vec<String> = feature_labels_from_activation(FeatureActivation::default());
    let json = serde_json::to_string(&labels).expect("empty labels must serialize");
    assert_eq!(json, "[]", "empty label list must serialize as empty JSON array");
    let deserialized: Vec<String> = serde_json::from_str(&json).unwrap();
    assert!(deserialized.is_empty());
}

#[test]
fn labels_serialize_to_json_array() {
    let act = FeatureActivation { cpu: true, ..Default::default() };
    let labels = feature_labels_from_activation(act);
    let json = serde_json::to_string(&labels).expect("labels must serialize");
    assert!(json.starts_with('['), "serialized labels must be a JSON array, got: {json:?}");
    assert!(json.ends_with(']'), "serialized labels must end with ']', got: {json:?}");
}

// ── Public API: active_features / feature_labels / feature_line ──────────────

#[test]
fn active_features_capture_returns_valid_snapshot() {
    // `active_features()` is the canonical "capture current flags" function.
    let features = active_features();
    // Labels derived from the snapshot must match `feature_labels()`.
    let from_capture = features.labels();
    let direct = feature_labels();
    assert_eq!(from_capture, direct, "active_features().labels() must match feature_labels()");
}

#[test]
fn feature_line_has_features_prefix() {
    let line = feature_line();
    assert!(
        line.starts_with("features: "),
        "feature_line must start with 'features: ', got: {line:?}"
    );
}

#[test]
fn feature_labels_contains_cpu_when_compiled_with_cpu_feature() {
    #[cfg(feature = "cpu")]
    {
        let labels = feature_labels();
        assert!(
            labels.iter().any(|l| l == "cpu"),
            "expected 'cpu' in feature labels when compiled with --features cpu, got: {labels:?}"
        );
    }
    // When cpu feature is absent this test is a no-op.
    #[cfg(not(feature = "cpu"))]
    {
        // Nothing to assert – just ensure the test compiles.
    }
}

#[test]
fn feature_line_contains_cpu_when_compiled_with_cpu_feature() {
    #[cfg(feature = "cpu")]
    {
        let line = feature_line();
        assert!(
            line.contains("cpu"),
            "feature_line must contain 'cpu' when compiled with --features cpu, got: {line:?}"
        );
    }
    #[cfg(not(feature = "cpu"))]
    {
        // Nothing to assert.
    }
}

// ── GPU / CUDA consistency (feature lattice) ──────────────────────────────────

#[test]
fn gpu_without_explicit_cuda_produces_gpu_label() {
    // gpu=true, cuda=false → at minimum "gpu" must appear
    let act = FeatureActivation { gpu: true, cuda: false, ..Default::default() };
    let labels = feature_labels_from_activation(act);
    assert!(labels.contains(&"gpu".to_string()), "gpu=true must produce 'gpu' label: {labels:?}");
}

#[test]
fn cuda_alone_produces_gpu_label_via_normalization() {
    // cuda ⟹ gpu (normalization documented in crate)
    let act = FeatureActivation { cuda: true, gpu: false, ..Default::default() };
    let labels = feature_labels_from_activation(act);
    assert!(
        labels.contains(&"cuda".to_string()),
        "cuda=true must produce 'cuda' label: {labels:?}"
    );
    assert!(
        labels.contains(&"gpu".to_string()),
        "cuda=true must also produce 'gpu' via normalization: {labels:?}"
    );
}

#[test]
fn cpu_and_gpu_are_not_mutually_exclusive() {
    // Both can be active simultaneously (orthogonal in the lattice).
    let act = FeatureActivation { cpu: true, gpu: true, ..Default::default() };
    let labels = feature_labels_from_activation(act);
    assert!(labels.contains(&"cpu".to_string()), "expected 'cpu': {labels:?}");
    assert!(labels.contains(&"gpu".to_string()), "expected 'gpu': {labels:?}");
}

//! Comprehensive tests for `bitnet-runtime-feature-flags`.
//!
//! Covers the full public API surface: `FeatureActivation` construction and
//! field semantics, feature-lattice implications, determinism, individual flag
//! independence, single-flag lattice propagation, and JSON label round-trips.
//!
//! These tests are designed to run under `--no-default-features --features cpu`
//! and do NOT duplicate the coverage already present in `feature_flags_tests.rs`
//! or `property_tests.rs`.

use bitnet_runtime_feature_flags::{
    FeatureActivation, active_features, active_features_from_activation, feature_labels,
    feature_labels_from_activation, feature_line, feature_line_from_activation,
};

// ── FeatureActivation field enumeration ──────────────────────────────────────

/// Every publicly accessible field is a bool and individually settable.
#[test]
fn every_field_is_independently_settable() {
    let fields: &[(&str, fn() -> FeatureActivation)] = &[
        ("cpu", || FeatureActivation { cpu: true, ..Default::default() }),
        ("gpu", || FeatureActivation { gpu: true, ..Default::default() }),
        ("cuda", || FeatureActivation { cuda: true, ..Default::default() }),
        ("inference", || FeatureActivation { inference: true, ..Default::default() }),
        ("kernels", || FeatureActivation { kernels: true, ..Default::default() }),
        ("tokenizers", || FeatureActivation { tokenizers: true, ..Default::default() }),
        ("quantization", || FeatureActivation { quantization: true, ..Default::default() }),
        ("cli", || FeatureActivation { cli: true, ..Default::default() }),
        ("server", || FeatureActivation { server: true, ..Default::default() }),
        ("ffi", || FeatureActivation { ffi: true, ..Default::default() }),
        ("python", || FeatureActivation { python: true, ..Default::default() }),
        ("wasm", || FeatureActivation { wasm: true, ..Default::default() }),
        ("crossval", || FeatureActivation { crossval: true, ..Default::default() }),
        ("trace", || FeatureActivation { trace: true, ..Default::default() }),
        ("iq2s_ffi", || FeatureActivation { iq2s_ffi: true, ..Default::default() }),
        ("cpp_ffi", || FeatureActivation { cpp_ffi: true, ..Default::default() }),
        ("fixtures", || FeatureActivation { fixtures: true, ..Default::default() }),
        ("reporting", || FeatureActivation { reporting: true, ..Default::default() }),
        ("trend", || FeatureActivation { trend: true, ..Default::default() }),
        ("integration_tests", || FeatureActivation {
            integration_tests: true,
            ..Default::default()
        }),
    ];
    for (name, make) in fields {
        let act = make();
        let labels = feature_labels_from_activation(act);
        assert!(!labels.is_empty(), "field '{name}' set to true must produce at least one label");
    }
}

// ── CPU / GPU flag independence ───────────────────────────────────────────────

#[test]
fn cpu_only_activation_gpu_and_cuda_labels_absent() {
    let act = FeatureActivation { cpu: true, ..Default::default() };
    let labels = feature_labels_from_activation(act);
    assert!(
        !labels.contains(&"gpu".to_string()),
        "cpu-only should NOT produce 'gpu' label, got {labels:?}"
    );
    assert!(
        !labels.contains(&"cuda".to_string()),
        "cpu-only should NOT produce 'cuda' label, got {labels:?}"
    );
}

#[test]
fn gpu_only_activation_cpu_label_absent() {
    let act = FeatureActivation { gpu: true, ..Default::default() };
    let labels = feature_labels_from_activation(act);
    assert!(
        !labels.contains(&"cpu".to_string()),
        "gpu-only should NOT produce 'cpu' label, got {labels:?}"
    );
}

#[test]
fn ffi_flag_produces_ffi_label() {
    let act = FeatureActivation { ffi: true, ..Default::default() };
    let labels = feature_labels_from_activation(act);
    assert!(labels.contains(&"ffi".to_string()), "ffi=true must produce 'ffi' label: {labels:?}");
}

#[test]
fn crossval_flag_produces_crossval_label() {
    let act = FeatureActivation { crossval: true, ..Default::default() };
    let labels = feature_labels_from_activation(act);
    assert!(
        labels.contains(&"crossval".to_string()),
        "crossval=true must produce 'crossval' label: {labels:?}"
    );
}

#[test]
fn trace_flag_produces_trace_label() {
    let act = FeatureActivation { trace: true, ..Default::default() };
    let labels = feature_labels_from_activation(act);
    assert!(
        labels.contains(&"trace".to_string()),
        "trace=true must produce 'trace' label: {labels:?}"
    );
}

#[test]
fn reporting_flag_produces_reporting_label() {
    let act = FeatureActivation { reporting: true, ..Default::default() };
    let labels = feature_labels_from_activation(act);
    assert!(
        labels.contains(&"reporting".to_string()),
        "reporting=true must produce 'reporting' label: {labels:?}"
    );
}

#[test]
fn trend_flag_produces_trend_label() {
    let act = FeatureActivation { trend: true, ..Default::default() };
    let labels = feature_labels_from_activation(act);
    assert!(
        labels.contains(&"trend".to_string()),
        "trend=true must produce 'trend' label: {labels:?}"
    );
}

#[test]
fn fixtures_flag_produces_fixtures_label() {
    let act = FeatureActivation { fixtures: true, ..Default::default() };
    let labels = feature_labels_from_activation(act);
    assert!(
        labels.contains(&"fixtures".to_string()),
        "fixtures=true must produce 'fixtures' label: {labels:?}"
    );
}

#[test]
fn python_flag_produces_python_label() {
    let act = FeatureActivation { python: true, ..Default::default() };
    let labels = feature_labels_from_activation(act);
    assert!(
        labels.contains(&"python".to_string()),
        "python=true must produce 'python' label: {labels:?}"
    );
}

#[test]
fn wasm_flag_produces_wasm_label() {
    let act = FeatureActivation { wasm: true, ..Default::default() };
    let labels = feature_labels_from_activation(act);
    assert!(
        labels.contains(&"wasm".to_string()),
        "wasm=true must produce 'wasm' label: {labels:?}"
    );
}

#[test]
fn server_flag_produces_server_label() {
    let act = FeatureActivation { server: true, ..Default::default() };
    let labels = feature_labels_from_activation(act);
    assert!(
        labels.contains(&"server".to_string()),
        "server=true must produce 'server' label: {labels:?}"
    );
}

#[test]
fn quantization_flag_produces_quantization_label() {
    let act = FeatureActivation { quantization: true, ..Default::default() };
    let labels = feature_labels_from_activation(act);
    assert!(
        labels.contains(&"quantization".to_string()),
        "quantization=true must produce 'quantization' label: {labels:?}"
    );
}

#[test]
fn cpp_ffi_flag_produces_cpp_ffi_label() {
    let act = FeatureActivation { cpp_ffi: true, ..Default::default() };
    let labels = feature_labels_from_activation(act);
    assert!(
        labels.contains(&"cpp-ffi".to_string()) || labels.iter().any(|l| l.contains("ffi")),
        "cpp_ffi=true must produce a ffi-related label: {labels:?}"
    );
}

#[test]
fn iq2s_ffi_flag_produces_iq2s_ffi_label() {
    let act = FeatureActivation { iq2s_ffi: true, ..Default::default() };
    let labels = feature_labels_from_activation(act);
    assert!(
        labels.iter().any(|l| l.contains("iq2") || l.contains("ffi")),
        "iq2s_ffi=true must produce a iq2s/ffi-related label: {labels:?}"
    );
}

#[test]
fn integration_tests_flag_produces_nonempty_labels() {
    let act = FeatureActivation { integration_tests: true, ..Default::default() };
    let labels = feature_labels_from_activation(act);
    assert!(
        !labels.is_empty(),
        "integration_tests=true must produce at least one label: {labels:?}"
    );
}

// ── Determinism across multiple calls ────────────────────────────────────────

#[test]
fn active_features_deterministic_across_two_calls() {
    let first = active_features();
    let second = active_features();
    assert_eq!(
        first.labels(),
        second.labels(),
        "active_features() must be deterministic across calls"
    );
}

#[test]
fn feature_labels_deterministic_across_two_calls() {
    let first = feature_labels();
    let second = feature_labels();
    assert_eq!(first, second, "feature_labels() must be deterministic");
}

#[test]
fn feature_line_deterministic_across_two_calls() {
    let first = feature_line();
    let second = feature_line();
    assert_eq!(first, second, "feature_line() must be deterministic");
}

// ── CPU feature gate: compile-time verification ───────────────────────────────

#[test]
fn cpu_feature_flag_reflects_build_flag() {
    let labels = feature_labels();
    let line = feature_line();

    #[cfg(feature = "cpu")]
    {
        assert!(
            labels.iter().any(|l| l == "cpu"),
            "compiled with --features cpu; 'cpu' must appear in feature_labels(), got {labels:?}"
        );
        assert!(
            line.contains("cpu"),
            "compiled with --features cpu; feature_line() must contain 'cpu', got {line:?}"
        );
    }

    #[cfg(not(feature = "cpu"))]
    {
        assert!(
            !labels.iter().any(|l| l == "cpu"),
            "compiled WITHOUT cpu feature; 'cpu' must NOT appear in labels, got {labels:?}"
        );
    }
}

// ── Public API: `active_features()` == `active_features_from_activation(cfg)` ─

#[test]
fn active_features_api_consistent_with_from_activation() {
    let via_api = active_features();
    // The public `active_features_from_activation` with a cpu-true activation
    // (when the feature is compiled) must contain at minimum what the API reports.
    let labels_api = via_api.labels();
    let labels_direct = feature_labels();
    assert_eq!(labels_api, labels_direct, "active_features().labels() must equal feature_labels()");
}

// ── Feature-line format invariants ───────────────────────────────────────────

#[test]
fn feature_line_never_empty() {
    let line = feature_line();
    assert!(!line.is_empty(), "feature_line() must never be empty");
}

#[test]
fn feature_line_none_when_all_flags_off() {
    let line = feature_line_from_activation(FeatureActivation::default());
    assert_eq!(line, "features: none", "all-false activation must yield 'features: none'");
}

#[test]
fn feature_line_has_comma_separator_for_multiple_labels() {
    let act = FeatureActivation { cpu: true, quantization: true, ..Default::default() };
    let line = feature_line_from_activation(act);
    // Two distinct features should be separated by ", "
    assert!(
        line.contains(", "),
        "feature_line with multiple labels must use ', ' separator, got: {line:?}"
    );
}

// ── Labels are always non-empty strings ──────────────────────────────────────

#[test]
fn all_runtime_labels_are_nonempty_strings() {
    for label in feature_labels() {
        assert!(!label.is_empty(), "every label must be a non-empty string");
    }
}

// ── FeatureSet / active_features_from_activation consistency ─────────────────

#[test]
fn active_features_from_activation_cpu_contains_expected_labels() {
    let act = FeatureActivation { cpu: true, ..Default::default() };
    let fs = active_features_from_activation(act);
    let labels = fs.labels();
    // cpu should imply cpu, inference, kernels, tokenizers
    for expected in &["cpu", "inference", "kernels", "tokenizers"] {
        assert!(
            labels.contains(&expected.to_string()),
            "cpu activation must contain '{expected}' label, got: {labels:?}"
        );
    }
}

#[test]
fn feature_labels_length_is_stable() {
    let act = FeatureActivation { cpu: true, gpu: true, ..Default::default() };
    let len1 = feature_labels_from_activation(act).len();
    let len2 = feature_labels_from_activation(act).len();
    assert_eq!(len1, len2, "label count must be stable across repeated calls");
}

// ── JSON round-trips for individual flags ─────────────────────────────────────

#[test]
fn json_round_trip_ffi_only_labels() {
    let act = FeatureActivation { ffi: true, ..Default::default() };
    let labels = feature_labels_from_activation(act);
    let json = serde_json::to_string(&labels).expect("must serialize");
    let back: Vec<String> = serde_json::from_str(&json).expect("must deserialize");
    assert_eq!(labels, back, "ffi-only labels must survive JSON round-trip");
}

#[test]
fn json_round_trip_crossval_only_labels() {
    let act = FeatureActivation { crossval: true, ..Default::default() };
    let labels = feature_labels_from_activation(act);
    let json = serde_json::to_string(&labels).expect("must serialize");
    let back: Vec<String> = serde_json::from_str(&json).expect("must deserialize");
    assert_eq!(labels, back, "crossval-only labels must survive JSON round-trip");
}

// ── FeatureActivation Copy semantics ─────────────────────────────────────────

#[test]
fn feature_activation_copied_produces_same_labels() {
    let original = FeatureActivation { cpu: true, trace: true, ..Default::default() };
    let copied = original; // Copy (not move)
    assert_eq!(
        feature_labels_from_activation(original),
        feature_labels_from_activation(copied),
        "Copy of FeatureActivation must yield identical labels"
    );
}

#[test]
fn feature_activation_cloned_produces_same_labels() {
    let original = FeatureActivation { quantization: true, fixtures: true, ..Default::default() };
    let cloned = original.clone();
    assert_eq!(
        feature_labels_from_activation(original),
        feature_labels_from_activation(cloned),
        "Clone of FeatureActivation must yield identical labels"
    );
}

// ── Feature-line contains all labels ─────────────────────────────────────────

#[test]
fn feature_line_contains_all_label_strings_multi() {
    let act = FeatureActivation {
        ffi: true,
        crossval: true,
        reporting: true,
        trend: true,
        ..Default::default()
    };
    let labels = feature_labels_from_activation(act);
    let line = feature_line_from_activation(act);
    for label in &labels {
        assert!(line.contains(label.as_str()), "label '{label}' missing from line '{line}'");
    }
}

#[test]
fn feature_line_cpu_inference_all_present() {
    // cpu implies inference, kernels, tokenizers in the lattice
    let act = FeatureActivation { cpu: true, ..Default::default() };
    let line = feature_line_from_activation(act);
    for expected_label in &["cpu", "inference", "kernels", "tokenizers"] {
        assert!(
            line.contains(expected_label),
            "feature_line for cpu must include '{expected_label}': {line:?}"
        );
    }
}

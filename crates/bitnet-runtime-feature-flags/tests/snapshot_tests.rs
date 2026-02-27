use bitnet_runtime_feature_flags::{
    FeatureActivation, active_features_from_activation, feature_labels, feature_line,
    feature_line_from_activation,
};

#[test]
fn feature_line_starts_with_features_prefix() {
    // Regression: the canonical feature line prefix must never change
    let line = feature_line();
    insta::assert_snapshot!(&line[..9]);
}

#[test]
fn feature_labels_count_with_cpu_feature() {
    // When compiled with --features cpu, labels must be non-empty and include "cpu".
    // The exact count varies by context (workspace vs isolated build) due to Cargo
    // feature unification, so we assert presence of "cpu" rather than an exact count.
    let labels = feature_labels();
    assert!(!labels.is_empty(), "feature labels must not be empty");
    assert!(
        labels.iter().any(|l| *l == "cpu"),
        "expected 'cpu' in feature labels, got: {labels:?}"
    );
}

#[test]
fn feature_line_contains_cpu() {
    // The feature line must always contain "cpu" when compiled with --features cpu.
    // The exact list varies by context due to workspace feature unification, so we
    // only assert the stable parts: prefix and the presence of "cpu".
    let line = feature_line();
    assert!(
        line.starts_with("features: "),
        "feature_line must start with 'features: ', got: {line:?}"
    );
    assert!(line.contains("cpu"), "feature_line must contain 'cpu', got: {line:?}");
}

#[test]
fn active_features_cpu_activation_labels() {
    // Uses an explicit activation struct (not cfg!()) so the snapshot is
    // stable across workspace vs isolated build contexts.
    let act = FeatureActivation { cpu: true, ..Default::default() };
    let features = active_features_from_activation(act);
    let labels = features.labels();
    insta::assert_snapshot!("active_features_cpu_activation", labels.join("\n"));
}

#[test]
fn feature_line_cpu_activation() {
    // Pins the canonical feature-line format for a cpu-only activation.
    // Uses explicit activation to avoid workspace feature-unification variance.
    let act = FeatureActivation { cpu: true, ..Default::default() };
    let line = feature_line_from_activation(act);
    insta::assert_snapshot!("feature_line_cpu_activation", line);
}

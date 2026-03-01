//! Edge-case tests for bitnet-runtime-feature-flags façade.

use bitnet_runtime_feature_flags::{active_features, feature_labels, feature_line};

// ---------------------------------------------------------------------------
// active_features
// ---------------------------------------------------------------------------

#[test]
fn active_features_returns_feature_set() {
    let features = active_features();
    let labels = features.labels();
    // In test builds, at least some features should be active
    assert!(!labels.is_empty());
}

#[test]
fn active_features_labels_are_non_empty_strings() {
    let features = active_features();
    for label in features.labels() {
        assert!(!label.is_empty());
        assert!(!label.contains(' '));
    }
}

// ---------------------------------------------------------------------------
// feature_labels
// ---------------------------------------------------------------------------

#[test]
fn feature_labels_matches_active_features() {
    let from_features = active_features().labels();
    let from_labels = feature_labels();
    // Both should produce the same set of labels
    let mut a: Vec<String> = from_features;
    let mut b: Vec<String> = from_labels;
    a.sort();
    b.sort();
    assert_eq!(a, b);
}

#[test]
fn feature_labels_not_empty() {
    let labels = feature_labels();
    assert!(!labels.is_empty());
}

// ---------------------------------------------------------------------------
// feature_line
// ---------------------------------------------------------------------------

#[test]
fn feature_line_contains_features_keyword() {
    let line = feature_line();
    assert!(line.contains("features"), "feature_line: {line}");
}

#[test]
fn feature_line_not_empty() {
    let line = feature_line();
    assert!(!line.is_empty());
}

#[test]
fn feature_line_contains_some_label() {
    let line = feature_line();
    let labels = feature_labels();
    // At least one label should appear in the line
    let found = labels.iter().any(|l| line.contains(l.as_str()));
    assert!(found, "feature_line '{line}' should contain at least one label from {labels:?}");
}

// ---------------------------------------------------------------------------
// Consistency across calls
// ---------------------------------------------------------------------------

#[test]
fn feature_labels_stable_across_calls() {
    let a = feature_labels();
    let b = feature_labels();
    assert_eq!(a, b);
}

#[test]
fn feature_line_stable_across_calls() {
    let a = feature_line();
    let b = feature_line();
    assert_eq!(a, b);
}

#[test]
fn active_features_stable_across_calls() {
    let a = active_features().labels();
    let b = active_features().labels();
    assert_eq!(a, b);
}

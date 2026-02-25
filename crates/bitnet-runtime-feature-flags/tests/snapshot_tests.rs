use bitnet_runtime_feature_flags::{feature_labels, feature_line};

#[test]
fn feature_line_starts_with_features_prefix() {
    // Regression: the canonical feature line prefix must never change
    let line = feature_line();
    insta::assert_snapshot!(&line[..9]);
}

#[test]
fn feature_labels_count_with_cpu_feature() {
    // When compiled with --features cpu, labels must be non-empty
    let labels = feature_labels();
    insta::assert_snapshot!(&labels.len().to_string());
}

#[test]
fn feature_line_format_stable() {
    // The full feature_line output must match a stable canonical format
    insta::assert_snapshot!(feature_line());
}

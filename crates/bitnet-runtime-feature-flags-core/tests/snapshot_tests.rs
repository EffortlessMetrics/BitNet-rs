use bitnet_runtime_feature_flags_core::{FeatureActivation, active_features_from_activation};

#[test]
fn cpu_only_activation_labels() {
    let act = FeatureActivation { cpu: true, ..Default::default() };
    let mut labels = act.to_labels();
    labels.sort();
    insta::assert_snapshot!(labels.join("\n"));
}

#[test]
fn gpu_and_cuda_activation_labels() {
    let act = FeatureActivation { gpu: true, cuda: true, ..Default::default() };
    let mut labels = act.to_labels();
    labels.sort();
    insta::assert_snapshot!(labels.join("\n"));
}

#[test]
fn empty_activation_labels_is_empty() {
    let act = FeatureActivation::default();
    let labels = act.to_labels();
    insta::assert_snapshot!(format!("count={}", labels.len()));
}

#[test]
fn cpu_activation_feature_set_contains_inference() {
    let act = FeatureActivation { cpu: true, ..Default::default() };
    let features = active_features_from_activation(act);
    let mut labels: Vec<_> = features.labels().into_iter().collect();
    labels.sort();
    insta::assert_snapshot!(labels.join("\n"));
}

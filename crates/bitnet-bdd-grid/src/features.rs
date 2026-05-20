use crate::{FeatureSet, feature_set_from_names, try_feature_set_from_names};

pub(crate) fn curated_features(features: &[&str]) -> FeatureSet {
    try_feature_set_from_names(features).unwrap_or_else(|unknown| {
        eprintln!("curated BDD grid contains unknown feature names: {}", unknown.join(", "));
        feature_set_from_names(features)
    })
}

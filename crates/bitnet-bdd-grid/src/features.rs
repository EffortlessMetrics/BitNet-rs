use crate::{FeatureSet, try_feature_set_from_names};

pub(crate) fn curated_features(features: &[&str]) -> FeatureSet {
    match try_feature_set_from_names(features) {
        Ok(feature_set) => feature_set,
        Err(unknown) => {
            assert!(
                unknown.is_empty(),
                "curated BDD grid contains unknown feature names: {}",
                unknown.join(", ")
            );
            FeatureSet::default()
        }
    }
}

//! Core BDD scenario + feature-grid primitives shared across BitNet crates.
//!
//! This crate intentionally stays free from curated policy content and instead
//! provides stable, low-level types plus reusable grid helpers.

mod environment;
mod features;
mod grid;
mod labels;
mod scenario;

pub use environment::ExecutionEnvironment;
pub use features::{BitnetFeature, FeatureSet, feature_set_from_names, try_feature_set_from_names};
pub use grid::{BddCell, BddGrid};
pub use scenario::TestingScenario;

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use super::*;

    #[test]
    fn test_scenario_parsing() {
        assert_eq!(TestingScenario::from_str("unit"), Ok(TestingScenario::Unit));
        assert_eq!(
            TestingScenario::from_str("perf").map_err(|e| e.to_string()),
            Ok(TestingScenario::Performance)
        );
        assert!(TestingScenario::from_str("unknown").is_err());
    }

    #[test]
    fn test_grid_lookup_and_validation() {
        let cell = BddCell {
            scenario: TestingScenario::Unit,
            environment: ExecutionEnvironment::Local,
            required_features: feature_set_from_names(&["inference", "kernels", "tokenizers"]),
            optional_features: feature_set_from_names(&["reporting"]),
            forbidden_features: FeatureSet::new(),
            intent: "Unit test row",
        };

        let active = feature_set_from_names(&["inference", "kernels", "tokenizers"]);
        assert!(cell.supports(&active));
        assert!(cell.violations(&active).0.is_empty());
        assert!(cell.violations(&active).1.is_empty());

        // Verify grid lookup with a leaked static slice (test-only).
        let rows: &'static [BddCell] = Box::leak(Box::new([cell]));
        let grid = BddGrid::from_rows(rows);
        let found = grid.find(TestingScenario::Unit, ExecutionEnvironment::Local);
        assert!(found.is_some());
    }

    #[test]
    fn test_try_feature_set_from_names_rejects_unknown_features() {
        let result = try_feature_set_from_names(&["inference", "unknown-feature"]);
        assert_eq!(result, Err(vec!["unknown-feature".to_string()]));
    }
}

//! Core BDD scenario + feature-grid primitives shared across BitNet crates.
//!
//! This crate intentionally stays free from curated policy content and instead
//! provides stable, low-level types plus reusable grid helpers.

use std::fmt;
use std::str::FromStr;

pub use bitnet_feature_set_core::{BitnetFeature, FeatureSet, feature_set_from_names};

/// Logical test scenario axis for BDD planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TestingScenario {
    Unit,
    Integration,
    EndToEnd,
    Performance,
    CrossValidation,
    Smoke,
    Development,
    Debug,
    Minimal,
}

impl fmt::Display for TestingScenario {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unit => write!(f, "unit"),
            Self::Integration => write!(f, "integration"),
            Self::EndToEnd => write!(f, "e2e"),
            Self::Performance => write!(f, "performance"),
            Self::CrossValidation => write!(f, "crossval"),
            Self::Smoke => write!(f, "smoke"),
            Self::Development => write!(f, "development"),
            Self::Debug => write!(f, "debug"),
            Self::Minimal => write!(f, "minimal"),
        }
    }
}

impl FromStr for TestingScenario {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "unit" => Ok(Self::Unit),
            "integration" => Ok(Self::Integration),
            "e2e" | "end-to-end" | "endtoend" => Ok(Self::EndToEnd),
            "performance" | "perf" => Ok(Self::Performance),
            "crossval" | "cross-validation" => Ok(Self::CrossValidation),
            "smoke" => Ok(Self::Smoke),
            "development" | "dev" => Ok(Self::Development),
            "debug" => Ok(Self::Debug),
            "minimal" | "min" => Ok(Self::Minimal),
            _ => Err("unknown testing scenario"),
        }
    }
}

/// Execution environment axis for BDD planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExecutionEnvironment {
    Local,
    Ci,
    PreProduction,
    Production,
}

impl fmt::Display for ExecutionEnvironment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Local => write!(f, "local"),
            Self::Ci => write!(f, "ci"),
            Self::PreProduction => write!(f, "pre-prod"),
            Self::Production => write!(f, "production"),
        }
    }
}

impl FromStr for ExecutionEnvironment {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "local" | "dev" | "development" => Ok(Self::Local),
            "ci" | "ci/cd" | "cicd" => Ok(Self::Ci),
            "pre-prod" | "preprod" | "pre-production" | "preproduction" | "staging" => {
                Ok(Self::PreProduction)
            }
            "prod" | "production" => Ok(Self::Production),
            _ => Err("unknown execution environment"),
        }
    }
}

/// Cell in the BDD grid.
#[derive(Debug, Clone)]
pub struct BddCell {
    /// Scenario this row applies to.
    pub scenario: TestingScenario,
    /// Environment this row applies to.
    pub environment: ExecutionEnvironment,
    /// Required features for the scenario.
    pub required_features: FeatureSet,
    /// Optional features for the scenario.
    pub optional_features: FeatureSet,
    /// Forbidden features for the scenario.
    pub forbidden_features: FeatureSet,
    /// Human-readable intent for this row.
    pub intent: &'static str,
}

impl BddCell {
    /// Returns true when a feature set is valid for this row.
    pub fn supports(&self, features: &FeatureSet) -> bool {
        features.satisfies(&self.required_features, &self.forbidden_features)
    }

    /// Missing and forbidden diagnostics.
    pub fn violations(&self, features: &FeatureSet) -> (FeatureSet, FeatureSet) {
        (
            features.missing_required(&self.required_features),
            features.forbidden_overlap(&self.forbidden_features),
        )
    }
}

/// Immutable, small in-memory grid for scenario/environment contracts.
#[derive(Debug, Clone, Copy)]
pub struct BddGrid {
    rows: &'static [BddCell],
}

impl BddGrid {
    /// Construct a grid from static rows.
    pub const fn from_rows(rows: &'static [BddCell]) -> Self {
        Self { rows }
    }

    /// Iterate rows in deterministic order.
    pub const fn rows(&self) -> &'static [BddCell] {
        self.rows
    }

    /// Find a single row by scenario/environment pair.
    pub fn find(
        &self,
        scenario: TestingScenario,
        environment: ExecutionEnvironment,
    ) -> Option<&'static BddCell> {
        self.rows.iter().find(|cell| cell.scenario == scenario && cell.environment == environment)
    }

    /// Find all rows for a scenario.
    pub fn rows_for_scenario(&self, scenario: TestingScenario) -> Vec<&'static BddCell> {
        self.rows.iter().filter(|cell| cell.scenario == scenario).collect()
    }

    /// Validate a feature set against a scenario/environment cell.
    pub fn validate(
        &self,
        scenario: TestingScenario,
        environment: ExecutionEnvironment,
        features: &FeatureSet,
    ) -> Option<(FeatureSet, FeatureSet)> {
        self.find(scenario, environment).map(|cell| cell.violations(features))
    }
}

#[cfg(test)]
mod tests {
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
}

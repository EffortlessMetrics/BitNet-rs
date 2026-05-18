use crate::{ExecutionEnvironment, FeatureSet, TestingScenario};

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

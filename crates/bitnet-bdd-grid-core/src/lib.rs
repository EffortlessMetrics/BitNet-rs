//! Core BDD scenario + feature-grid primitives shared across BitNet crates.
//!
//! This crate intentionally stays free from curated policy content and instead
//! provides stable, low-level types plus reusable grid helpers.

use std::collections::BTreeSet;
use std::fmt;
use std::str::FromStr;

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

/// Canonical feature axes for feature-flag contracts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BitnetFeature {
    Cpu,
    Gpu,
    Cuda,
    Metal,
    Vulkan,
    Oneapi,
    Inference,
    Kernels,
    Tokenizers,
    Quantization,
    Cli,
    Server,
    Ffi,
    Python,
    Wasm,
    CrossValidation,
    Trace,
    Iq2sFfi,
    CppFfi,
    Fixtures,
    Reporting,
    Trend,
    IntegrationTests,
}

impl fmt::Display for BitnetFeature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::Gpu => write!(f, "gpu"),
            Self::Cuda => write!(f, "cuda"),
            Self::Metal => write!(f, "metal"),
            Self::Vulkan => write!(f, "vulkan"),
            Self::Oneapi => write!(f, "oneapi"),
            Self::Inference => write!(f, "inference"),
            Self::Kernels => write!(f, "kernels"),
            Self::Tokenizers => write!(f, "tokenizers"),
            Self::Quantization => write!(f, "quantization"),
            Self::Cli => write!(f, "cli"),
            Self::Server => write!(f, "server"),
            Self::Ffi => write!(f, "ffi"),
            Self::Python => write!(f, "python"),
            Self::Wasm => write!(f, "wasm"),
            Self::CrossValidation => write!(f, "crossval"),
            Self::Trace => write!(f, "trace"),
            Self::Iq2sFfi => write!(f, "iq2s-ffi"),
            Self::CppFfi => write!(f, "cpp-ffi"),
            Self::Fixtures => write!(f, "fixtures"),
            Self::Reporting => write!(f, "reporting"),
            Self::Trend => write!(f, "trend"),
            Self::IntegrationTests => write!(f, "integration-tests"),
        }
    }
}

impl FromStr for BitnetFeature {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "gpu" => Ok(Self::Gpu),
            "cuda" => Ok(Self::Cuda),
            "metal" => Ok(Self::Metal),
            "vulkan" => Ok(Self::Vulkan),
            "oneapi" => Ok(Self::Oneapi),
            "inference" => Ok(Self::Inference),
            "kernels" => Ok(Self::Kernels),
            "tokenizers" => Ok(Self::Tokenizers),
            "quantization" => Ok(Self::Quantization),
            "cli" => Ok(Self::Cli),
            "server" => Ok(Self::Server),
            "ffi" => Ok(Self::Ffi),
            "python" => Ok(Self::Python),
            "wasm" => Ok(Self::Wasm),
            "crossval" | "cross-validation" => Ok(Self::CrossValidation),
            "trace" => Ok(Self::Trace),
            "iq2s-ffi" => Ok(Self::Iq2sFfi),
            "cpp-ffi" => Ok(Self::CppFfi),
            "fixtures" => Ok(Self::Fixtures),
            "reporting" => Ok(Self::Reporting),
            "trend" => Ok(Self::Trend),
            "integration-tests" => Ok(Self::IntegrationTests),
            _ => Err("unknown feature"),
        }
    }
}

/// Ordered set of supported features.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FeatureSet(BTreeSet<BitnetFeature>);

impl FeatureSet {
    /// Construct an empty set.
    pub fn new() -> Self {
        Self(BTreeSet::new())
    }

    /// Insert a feature.
    pub fn insert(&mut self, feature: BitnetFeature) -> bool {
        self.0.insert(feature)
    }

    /// Test whether a feature is enabled.
    pub fn contains(&self, feature: BitnetFeature) -> bool {
        self.0.contains(&feature)
    }

    /// Add all features from the provided slice.
    pub fn extend<I>(&mut self, features: I)
    where
        I: IntoIterator<Item = BitnetFeature>,
    {
        self.0.extend(features);
    }

    /// Create a set from a string list (e.g. CLI/env var values).
    pub fn from_names<I, S>(features: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut set = Self::new();
        for feature in features {
            if let Ok(feature) = feature.as_ref().parse() {
                set.insert(feature);
            }
        }
        set
    }

    /// Human-readable representation for logs and diagnostics.
    pub fn labels(&self) -> Vec<String> {
        self.0.iter().map(ToString::to_string).collect()
    }

    /// Compute feature mismatches against requirements.
    pub fn missing_required(&self, required: &Self) -> Self {
        Self(required.0.difference(&self.0).copied().collect())
    }

    /// Compute forbidden feature overlap (features that should not be active).
    pub fn forbidden_overlap(&self, forbidden: &Self) -> Self {
        Self(self.0.intersection(&forbidden.0).copied().collect())
    }

    /// Check if this set satisfies required / forbidden constraints.
    pub fn satisfies(&self, required: &Self, forbidden: &Self) -> bool {
        self.missing_required(required).is_empty() && self.forbidden_overlap(forbidden).is_empty()
    }

    /// Expose iteration for callers that need compatibility logic.
    pub fn iter(&self) -> impl Iterator<Item = &BitnetFeature> {
        self.0.iter()
    }

    /// Whether set is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl From<&[BitnetFeature]> for FeatureSet {
    fn from(value: &[BitnetFeature]) -> Self {
        Self(value.iter().copied().collect())
    }
}

impl From<&[&str]> for FeatureSet {
    fn from(value: &[&str]) -> Self {
        Self::from_names(value.iter().copied())
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

/// Canonical, reusable helper for mapping runtime feature selections to `FeatureSet`.
pub fn feature_set_from_names(features: &[&str]) -> FeatureSet {
    let mut set = FeatureSet::new();
    for feature in features {
        if let Ok(feature) = feature.parse() {
            set.insert(feature);
        }
    }
    set
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

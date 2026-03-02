//! Core testing axes shared across BitNet policy/runtime crates.

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

#[cfg(test)]
mod tests {
    use super::{ExecutionEnvironment, TestingScenario};
    use std::str::FromStr;

    #[test]
    fn scenario_round_trip() {
        for scenario in [
            TestingScenario::Unit,
            TestingScenario::Integration,
            TestingScenario::EndToEnd,
            TestingScenario::Performance,
            TestingScenario::CrossValidation,
            TestingScenario::Smoke,
            TestingScenario::Development,
            TestingScenario::Debug,
            TestingScenario::Minimal,
        ] {
            let display = scenario.to_string();
            assert_eq!(TestingScenario::from_str(&display), Ok(scenario));
        }
    }

    #[test]
    fn environment_round_trip() {
        for environment in [
            ExecutionEnvironment::Local,
            ExecutionEnvironment::Ci,
            ExecutionEnvironment::PreProduction,
            ExecutionEnvironment::Production,
        ] {
            let display = environment.to_string();
            assert_eq!(ExecutionEnvironment::from_str(&display), Ok(environment));
        }
    }
}

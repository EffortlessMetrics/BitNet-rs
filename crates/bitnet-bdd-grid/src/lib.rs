//! Canonical curated BDD grid for BitNet.
//!
//! This crate intentionally keeps curated policy data here, while low-level
//! primitives (scenarios, features, grid and helper types) live in
//! `bitnet-bdd-grid-core` so they can be reused independently.

use std::sync::LazyLock;

pub use bitnet_bdd_grid_core::{
    BddCell, BddGrid, BitnetFeature, ExecutionEnvironment, FeatureSet, TestingScenario,
    feature_set_from_names,
};

static CURATED_ROWS: LazyLock<Box<[BddCell]>> = LazyLock::new(build_curated_rows);

fn build_curated_rows() -> Box<[BddCell]> {
    vec![
        BddCell {
            scenario: TestingScenario::Unit,
            environment: ExecutionEnvironment::Local,
            required_features: feature_set_from_names(&["inference", "kernels", "tokenizers"]),
            optional_features: feature_set_from_names(&["reporting", "fixtures"]),
            forbidden_features: feature_set_from_names(&["cpp-ffi"]),
            intent: "Fast, isolated unit execution with core inference path",
        },
        BddCell {
            scenario: TestingScenario::Integration,
            environment: ExecutionEnvironment::Local,
            required_features: feature_set_from_names(&["inference", "kernels", "tokenizers"]),
            optional_features: feature_set_from_names(&["crossval", "reporting", "fixtures"]),
            forbidden_features: FeatureSet::new(),
            intent: "Component and module interaction in local build",
        },
        BddCell {
            scenario: TestingScenario::CrossValidation,
            environment: ExecutionEnvironment::Ci,
            required_features: feature_set_from_names(&[
                "inference",
                "kernels",
                "tokenizers",
                "crossval",
            ]),
            optional_features: feature_set_from_names(&["fixtures", "reporting", "trend"]),
            forbidden_features: FeatureSet::new(),
            intent: "Reference parity / regression surface with controlled fixtures",
        },
        BddCell {
            scenario: TestingScenario::Performance,
            environment: ExecutionEnvironment::Ci,
            required_features: feature_set_from_names(&["inference", "kernels"]),
            optional_features: feature_set_from_names(&["gpu", "cuda", "reporting", "trend"]),
            forbidden_features: FeatureSet::new(),
            intent: "Throughput and latency-sensitive checks",
        },
        BddCell {
            scenario: TestingScenario::Smoke,
            environment: ExecutionEnvironment::PreProduction,
            required_features: feature_set_from_names(&["inference"]),
            optional_features: feature_set_from_names(&["tokenizers", "kernels", "crossval"]),
            forbidden_features: FeatureSet::new(),
            intent: "Minimum viable run for deployment safety",
        },
        BddCell {
            scenario: TestingScenario::Debug,
            environment: ExecutionEnvironment::Local,
            required_features: feature_set_from_names(&["inference"]),
            optional_features: feature_set_from_names(&["trace", "reporting"]),
            forbidden_features: FeatureSet::new(),
            intent: "Detailed behavior introspection and diagnostics",
        },
        BddCell {
            scenario: TestingScenario::Minimal,
            environment: ExecutionEnvironment::Local,
            required_features: feature_set_from_names(&["inference"]),
            optional_features: FeatureSet::new(),
            forbidden_features: FeatureSet::new(),
            intent: "Fastest-path execution with hard constraints",
        },
        BddCell {
            scenario: TestingScenario::EndToEnd,
            environment: ExecutionEnvironment::Ci,
            required_features: feature_set_from_names(&[
                "inference",
                "kernels",
                "tokenizers",
                "reporting",
                "crossval",
            ]),
            optional_features: feature_set_from_names(&["fixtures", "trend", "server"]),
            forbidden_features: FeatureSet::new(),
            intent: "Full stack workflow checks spanning startup through response path",
        },
        // Reason: CPU-only unit path validates the explicit `cpu` kernel feature gate and
        // ensures deterministic scalar execution without GPU dependency in CI.
        BddCell {
            scenario: TestingScenario::Unit,
            environment: ExecutionEnvironment::Ci,
            required_features: feature_set_from_names(&["cpu", "inference", "kernels"]),
            optional_features: feature_set_from_names(&["tokenizers", "reporting"]),
            forbidden_features: FeatureSet::new(),
            intent: "Deterministic CPU-only unit path with explicit kernel feature",
        },
        // Reason: GGUF loading and multi-format quantization (I2_S BitNet32, QK256, TL1/TL2)
        // integration tests require the `quantization` feature gate to be exercised in CI.
        BddCell {
            scenario: TestingScenario::Integration,
            environment: ExecutionEnvironment::Ci,
            required_features: feature_set_from_names(&[
                "inference",
                "kernels",
                "tokenizers",
                "quantization",
            ]),
            optional_features: feature_set_from_names(&["fixtures", "reporting"]),
            forbidden_features: FeatureSet::new(),
            intent: "GGUF model loading and quantization format integration (I2_S, QK256, TL1/TL2)",
        },
        // Reason: Local backend selection benchmarks exercise CPU-auto and GPU-explicit
        // dispatch paths; GPU path is compile-only until CUDA runtime is present.
        BddCell {
            scenario: TestingScenario::Performance,
            environment: ExecutionEnvironment::Local,
            required_features: feature_set_from_names(&["inference", "kernels"]),
            optional_features: feature_set_from_names(&["cpu", "gpu", "cuda", "reporting"]),
            forbidden_features: FeatureSet::new(),
            intent: "Local backend selection and kernel dispatch benchmarks",
        },
        // Reason: Sampling strategy development cells (greedy, top-p, top-k) exercise the
        // `SamplingStrategy` variants; not yet covered by a dedicated CI scenario.
        BddCell {
            scenario: TestingScenario::Development,
            environment: ExecutionEnvironment::Local,
            required_features: feature_set_from_names(&["inference", "kernels", "tokenizers"]),
            optional_features: feature_set_from_names(&["reporting", "trace"]),
            forbidden_features: FeatureSet::new(),
            intent: "Sampling strategy development and greedy/top-p/top-k path exercising",
        },
        // Reason: Receipt generation and schema v1.0.0 validation smoke path; the
        // `reporting` feature gate must be present to write and verify inference receipts.
        BddCell {
            scenario: TestingScenario::Smoke,
            environment: ExecutionEnvironment::Ci,
            required_features: feature_set_from_names(&["inference", "reporting"]),
            optional_features: feature_set_from_names(&["kernels", "tokenizers"]),
            forbidden_features: FeatureSet::new(),
            intent: "Smoke path for receipt generation and schema v1.0.0 validation",
        },
    ]
    .into_boxed_slice()
}

/// Canonical curated profile rows used by runtime profile resolution and tooling.
pub fn curated() -> BddGrid {
    BddGrid::from_rows(LazyLock::force(&CURATED_ROWS).as_ref())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_lookup_and_validation() {
        let grid = curated();
        let cell = grid.find(TestingScenario::Unit, ExecutionEnvironment::Local);
        assert!(cell.is_some());

        let active = feature_set_from_names(&["inference", "kernels", "tokenizers"]);
        let cell = cell.unwrap_or_else(|| panic!("unit/local row exists in curated grid"));
        assert!(cell.supports(&active));
        assert!(cell.violations(&active).0.is_empty());
        assert!(cell.violations(&active).1.is_empty());
    }
}

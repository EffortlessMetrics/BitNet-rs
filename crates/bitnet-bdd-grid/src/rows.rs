use crate::{BddCell, ExecutionEnvironment, FeatureSet, TestingScenario};

use super::features::curated_features;

pub(crate) fn build_curated_rows() -> Box<[BddCell]> {
    vec![
        BddCell {
            scenario: TestingScenario::Unit,
            environment: ExecutionEnvironment::Local,
            required_features: curated_features(&["inference", "kernels", "tokenizers"]),
            optional_features: curated_features(&["reporting", "fixtures"]),
            forbidden_features: curated_features(&["cpp-ffi"]),
            intent: "Fast, isolated unit execution with core inference path",
        },
        BddCell {
            scenario: TestingScenario::Integration,
            environment: ExecutionEnvironment::Local,
            required_features: curated_features(&["inference", "kernels", "tokenizers"]),
            optional_features: curated_features(&["crossval", "reporting", "fixtures"]),
            forbidden_features: FeatureSet::new(),
            intent: "Component and module interaction in local build",
        },
        BddCell {
            scenario: TestingScenario::CrossValidation,
            environment: ExecutionEnvironment::Ci,
            required_features: curated_features(&[
                "inference",
                "kernels",
                "tokenizers",
                "crossval",
            ]),
            optional_features: curated_features(&["fixtures", "reporting", "trend"]),
            forbidden_features: FeatureSet::new(),
            intent: "Reference parity / regression surface with controlled fixtures",
        },
        BddCell {
            scenario: TestingScenario::Performance,
            environment: ExecutionEnvironment::Ci,
            required_features: curated_features(&["inference", "kernels"]),
            optional_features: curated_features(&["gpu", "cuda", "reporting", "trend"]),
            forbidden_features: FeatureSet::new(),
            intent: "Throughput and latency-sensitive checks",
        },
        BddCell {
            scenario: TestingScenario::Smoke,
            environment: ExecutionEnvironment::PreProduction,
            required_features: curated_features(&["inference"]),
            optional_features: curated_features(&["tokenizers", "kernels", "crossval"]),
            forbidden_features: FeatureSet::new(),
            intent: "Minimum viable run for deployment safety",
        },
        BddCell {
            scenario: TestingScenario::Debug,
            environment: ExecutionEnvironment::Local,
            required_features: curated_features(&["inference"]),
            optional_features: curated_features(&["trace", "reporting"]),
            forbidden_features: FeatureSet::new(),
            intent: "Detailed behavior introspection and diagnostics",
        },
        BddCell {
            scenario: TestingScenario::Minimal,
            environment: ExecutionEnvironment::Local,
            required_features: curated_features(&["inference"]),
            optional_features: FeatureSet::new(),
            forbidden_features: FeatureSet::new(),
            intent: "Fastest-path execution with hard constraints",
        },
        BddCell {
            scenario: TestingScenario::EndToEnd,
            environment: ExecutionEnvironment::Ci,
            required_features: curated_features(&[
                "inference",
                "kernels",
                "tokenizers",
                "reporting",
                "crossval",
            ]),
            optional_features: curated_features(&["fixtures", "trend", "server"]),
            forbidden_features: FeatureSet::new(),
            intent: "Full stack workflow checks spanning startup through response path",
        },
        // Reason: CPU-only unit path validates the explicit `cpu` kernel feature gate and
        // ensures deterministic scalar execution without GPU dependency in CI.
        BddCell {
            scenario: TestingScenario::Unit,
            environment: ExecutionEnvironment::Ci,
            required_features: curated_features(&["cpu", "inference", "kernels"]),
            optional_features: curated_features(&["tokenizers", "reporting"]),
            forbidden_features: FeatureSet::new(),
            intent: "Deterministic CPU-only unit path with explicit kernel feature",
        },
        // Reason: GGUF loading and multi-format quantization (I2_S BitNet32, QK256, TL1/TL2)
        // integration tests require the `quantization` feature gate to be exercised in CI.
        BddCell {
            scenario: TestingScenario::Integration,
            environment: ExecutionEnvironment::Ci,
            required_features: curated_features(&[
                "inference",
                "kernels",
                "tokenizers",
                "quantization",
            ]),
            optional_features: curated_features(&["fixtures", "reporting"]),
            forbidden_features: FeatureSet::new(),
            intent: "GGUF model loading and quantization format integration (I2_S, QK256, TL1/TL2)",
        },
        // Reason: Local backend selection benchmarks exercise CPU-auto and GPU-explicit
        // dispatch paths; GPU path is compile-only until CUDA runtime is present.
        BddCell {
            scenario: TestingScenario::Performance,
            environment: ExecutionEnvironment::Local,
            required_features: curated_features(&["inference", "kernels"]),
            optional_features: curated_features(&["cpu", "gpu", "cuda", "reporting"]),
            forbidden_features: FeatureSet::new(),
            intent: "Local backend selection and kernel dispatch benchmarks",
        },
        // Reason: Sampling strategy development cells (greedy, top-p, top-k) exercise the
        // `SamplingStrategy` variants; not yet covered by a dedicated CI scenario.
        BddCell {
            scenario: TestingScenario::Development,
            environment: ExecutionEnvironment::Local,
            required_features: curated_features(&["inference", "kernels", "tokenizers"]),
            optional_features: curated_features(&["reporting", "trace"]),
            forbidden_features: FeatureSet::new(),
            intent: "Sampling strategy development and greedy/top-p/top-k path exercising",
        },
        // Reason: Receipt generation and schema v1.0.0 validation smoke path; the
        // `reporting` feature gate must be present to write and verify inference receipts.
        BddCell {
            scenario: TestingScenario::Smoke,
            environment: ExecutionEnvironment::Ci,
            required_features: curated_features(&["inference", "reporting"]),
            optional_features: curated_features(&["kernels", "tokenizers"]),
            forbidden_features: FeatureSet::new(),
            intent: "Smoke path for receipt generation and schema v1.0.0 validation",
        },
        // Feature: health — Scenario: "server returns 200 on /health"
        // Reason: Pre-production health-check validates that the server crate starts and
        // responds correctly before promotion; `server` and `reporting` must both be
        // present so the health endpoint and its receipt are exercised together.
        BddCell {
            scenario: TestingScenario::EndToEnd,
            environment: ExecutionEnvironment::PreProduction,
            required_features: curated_features(&["inference", "server", "reporting"]),
            optional_features: curated_features(&["kernels", "tokenizers"]),
            forbidden_features: FeatureSet::new(),
            intent: "Server health-check path (/health endpoint returns 200)",
        },
        // Feature: prompt-templates — Scenario: "auto template picks instruct for base models"
        // Reason: Template auto-detection logic in the CLI must be exercised in CI so that
        // regressions to the `cli`+`tokenizers` dispatch path are caught early.
        BddCell {
            scenario: TestingScenario::Development,
            environment: ExecutionEnvironment::Ci,
            required_features: curated_features(&["inference", "tokenizers", "cli"]),
            optional_features: curated_features(&["kernels", "reporting"]),
            forbidden_features: FeatureSet::new(),
            intent: "Prompt-template auto-detection selects instruct for base models",
        },
        // Feature: tokenizer — Scenario: "encode then decode returns original tokens"
        // Reason: Local cross-validation round-trips catch tokenizer parity regressions
        // (Issue #469) without requiring the full CI cross-validation fixture set.
        BddCell {
            scenario: TestingScenario::CrossValidation,
            environment: ExecutionEnvironment::Local,
            required_features: curated_features(&["inference", "tokenizers"]),
            optional_features: curated_features(&["fixtures", "reporting"]),
            forbidden_features: FeatureSet::new(),
            intent: "Tokenizer encode/decode round-trip preserves original token sequence",
        },
        // Feature: receipts — Scenario: "receipt JSON validates against schema v1.0.0"
        // Reason: A dedicated Debug/CI cell for receipt schema validation lets developers
        // iterate on the schema in CI debug mode without the full smoke-path overhead.
        BddCell {
            scenario: TestingScenario::Debug,
            environment: ExecutionEnvironment::Ci,
            required_features: curated_features(&["inference", "reporting"]),
            optional_features: curated_features(&["kernels", "tokenizers"]),
            forbidden_features: FeatureSet::new(),
            intent: "Receipt JSON validates against schema v1.0.0 with all required gates",
        },
        // Feature: sampling — Scenario: "same seed produces identical output"
        // Reason: A Minimal/CI cell verifies that the deterministic seed path in
        // `SamplingStrategy` produces bit-identical output; minimal feature set keeps
        // the check fast and independent of tokenizer or reporting state.
        BddCell {
            scenario: TestingScenario::Minimal,
            environment: ExecutionEnvironment::Ci,
            required_features: curated_features(&["inference", "kernels"]),
            optional_features: FeatureSet::new(),
            forbidden_features: FeatureSet::new(),
            intent: "Deterministic seed produces identical output across repeated runs",
        },
        // Feature: device-probe — Scenario: "CPU feature detection reports SIMD capabilities"
        // Reason: A Smoke/Local cell verifies that `bitnet-device-probe` correctly detects
        // the active CPU feature flags (AVX2, AVX-512, NEON) without requiring GPU runtime.
        // This is the fastest sanity check for the device-detection path.
        BddCell {
            scenario: TestingScenario::Smoke,
            environment: ExecutionEnvironment::Local,
            required_features: curated_features(&["cpu", "inference"]),
            optional_features: curated_features(&["reporting"]),
            forbidden_features: curated_features(&["gpu", "cuda"]),
            intent: "Device probe detects CPU features and reports available SIMD capabilities",
        },
        // Feature: logits — Scenario: "pure logits transforms pass numerical precision checks"
        // Reason: An Integration/PreProduction cell exercises `bitnet-logits` temperature
        // scaling, top-k, and top-p transforms at staging quality gates before promotion.
        // The `cpu` + `kernels` pair ensures SIMD paths are exercised end-to-end.
        BddCell {
            scenario: TestingScenario::Integration,
            environment: ExecutionEnvironment::PreProduction,
            required_features: curated_features(&["cpu", "inference", "kernels"]),
            optional_features: curated_features(&["reporting"]),
            forbidden_features: FeatureSet::new(),
            intent: "Pure logits transforms (temperature, top-k, top-p) pass numerical precision checks",
        },
        // Feature: generation — Scenario: "stopping criteria fire deterministically"
        // Reason: A Debug/PreProduction cell provides detailed introspection for
        // `bitnet-generation` stopping-criteria logic (max-tokens, stop-id, stop-string)
        // in a pre-production staging environment without the overhead of full fixtures.
        BddCell {
            scenario: TestingScenario::Debug,
            environment: ExecutionEnvironment::PreProduction,
            required_features: curated_features(&["inference"]),
            optional_features: curated_features(&["trace", "reporting"]),
            forbidden_features: FeatureSet::new(),
            intent: "Generation stopping criteria (max-tokens, stop-id, stop-string) fire deterministically",
        },
        // Feature: engine-core — Scenario: "session contracts match C++ reference"
        // Reason: A CrossValidation/PreProduction cell validates that `bitnet-engine-core`
        // session invariants align with the C++ reference implementation at each decode
        // step before production promotion.  `crossval` + `fixtures` are optional so the
        // cell also serves as a lightweight staging gate when the C++ reference is absent.
        BddCell {
            scenario: TestingScenario::CrossValidation,
            environment: ExecutionEnvironment::PreProduction,
            required_features: curated_features(&["inference", "kernels"]),
            optional_features: curated_features(&["crossval", "fixtures", "reporting"]),
            forbidden_features: FeatureSet::new(),
            intent: "Engine-core session contracts match C++ reference for all decode-step invariants",
        },
        // Feature: metal — Scenario: "Metal backend compiles and exposes GPU capability"
        // Reason: An EndToEnd/Local cell verifies the `metal` feature gate compiles without errors
        // on all platforms. Metal is macOS/iOS only but compile-time checks run everywhere
        // to catch feature-flag regressions before macOS runners are needed.
        BddCell {
            scenario: TestingScenario::EndToEnd,
            environment: ExecutionEnvironment::Local,
            required_features: curated_features(&["cpu", "metal"]),
            optional_features: curated_features(&["reporting"]),
            forbidden_features: curated_features(&["cuda"]),
            intent: "Metal backend feature gate compiles without errors (compile-only check)",
        },
        // Feature: vulkan — Scenario: "Vulkan backend compiles and probes availability"
        // Reason: A Minimal/PreProduction cell verifies that `--features vulkan` compiles.
        // Runtime probe returns `false` without a Vulkan GPU — only compile-time coverage needed.
        BddCell {
            scenario: TestingScenario::Minimal,
            environment: ExecutionEnvironment::PreProduction,
            required_features: curated_features(&["cpu", "vulkan"]),
            optional_features: curated_features(&["reporting"]),
            forbidden_features: curated_features(&["cuda"]),
            intent: "Vulkan backend feature gate compiles and probe reports compiled=true",
        },
        // Feature: oneapi — Scenario: "Intel oneAPI backend compiles without errors"
        // Reason: A Development/PreProduction cell verifies the `oneapi` feature gate compiles.
        // oneAPI runtime requires Intel hardware; only compile coverage is checked in CI.
        BddCell {
            scenario: TestingScenario::Development,
            environment: ExecutionEnvironment::PreProduction,
            required_features: curated_features(&["cpu", "oneapi"]),
            optional_features: curated_features(&["reporting"]),
            forbidden_features: curated_features(&["cuda"]),
            intent: "Intel oneAPI backend feature gate compiles without errors (compile-only check)",
        },
    ]
    .into_boxed_slice()
}

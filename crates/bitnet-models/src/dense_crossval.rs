//! Dense model cross-validation framework.
//!
//! Provides infrastructure for comparing dense SLM outputs against reference
//! implementations (PyTorch, ONNX Runtime, NumPy) with configurable tolerances.
//!
//! # Components
//!
//! - [`DenseCrossvalConfig`]: tolerance settings and test prompts
//! - [`CrossvalResult`] / [`CrossvalReport`]: structured comparison results
//! - [`CrossvalComparison`]: token and logit comparison utilities
//! - [`GoldenFixture`]: regression testing with saved reference outputs
//! - [`ReferenceOutput`]: captures expected output from reference backends

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// DenseCrossvalConfig
// ---------------------------------------------------------------------------

/// Configuration for dense model cross-validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseCrossvalConfig {
    /// Reference backend name (e.g., "pytorch", "onnxruntime", "numpy").
    pub reference_backend: String,
    /// Absolute tolerance for logit comparison.
    pub tolerance_atol: f32,
    /// Relative tolerance for logit comparison.
    pub tolerance_rtol: f32,
    /// Prompts to use for cross-validation.
    pub test_prompts: Vec<String>,
    /// Maximum tokens to generate per prompt.
    pub max_tokens: usize,
    /// Whether to compare logits.
    pub check_logits: bool,
    /// Whether to compare generated token IDs.
    pub check_tokens: bool,
}

impl Default for DenseCrossvalConfig {
    fn default() -> Self {
        Self {
            reference_backend: "pytorch".to_string(),
            tolerance_atol: 1e-5,
            tolerance_rtol: 1e-4,
            test_prompts: Vec::new(),
            max_tokens: 8,
            check_logits: true,
            check_tokens: true,
        }
    }
}

// ---------------------------------------------------------------------------
// CrossvalResult
// ---------------------------------------------------------------------------

/// Result of cross-validating a single prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossvalResult {
    /// The prompt that was tested.
    pub prompt: String,
    /// Token IDs from the reference implementation.
    pub expected_tokens: Vec<u32>,
    /// Token IDs from the implementation under test.
    pub actual_tokens: Vec<u32>,
    /// Whether token sequences matched exactly.
    pub token_match: bool,
    /// Maximum absolute logit difference across all steps.
    pub max_logit_diff: f32,
    /// Mean absolute logit difference across all steps.
    pub mean_logit_diff: f32,
    /// Whether this prompt passed all configured checks.
    pub passed: bool,
}

// ---------------------------------------------------------------------------
// CrossvalReport
// ---------------------------------------------------------------------------

/// Aggregated report for a cross-validation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossvalReport {
    /// Configuration used for this run.
    pub config: DenseCrossvalConfig,
    /// Per-prompt results.
    pub results: Vec<CrossvalResult>,
    /// Total number of prompts tested.
    pub total_prompts: usize,
    /// Number of prompts that passed.
    pub passed_prompts: usize,
    /// Whether all prompts passed.
    pub overall_pass: bool,
    /// Maximum logit difference across all prompts.
    pub max_logit_diff_across_all: f32,
    /// Wall-clock time for the full run in milliseconds.
    pub run_time_ms: u64,
}

impl CrossvalReport {
    /// Build a report from a config and a set of per-prompt results.
    pub fn from_results(
        config: DenseCrossvalConfig,
        results: Vec<CrossvalResult>,
        run_time_ms: u64,
    ) -> Self {
        let total_prompts = results.len();
        let passed_prompts = results.iter().filter(|r| r.passed).count();
        let overall_pass = passed_prompts == total_prompts;
        let max_logit_diff_across_all =
            results.iter().map(|r| r.max_logit_diff).fold(0.0_f32, f32::max);

        Self {
            config,
            results,
            total_prompts,
            passed_prompts,
            overall_pass,
            max_logit_diff_across_all,
            run_time_ms,
        }
    }
}

// ---------------------------------------------------------------------------
// ReferenceOutput
// ---------------------------------------------------------------------------

/// Captured output from a reference implementation for a single prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceOutput {
    /// The input prompt.
    pub prompt: String,
    /// Generated token IDs.
    pub token_ids: Vec<u32>,
    /// Per-step logit vectors (one inner vec per generation step).
    pub logits: Vec<Vec<f32>>,
    /// Backend that produced this output (e.g., "pytorch").
    pub backend: String,
    /// Model name / identifier.
    pub model_name: String,
}

// ---------------------------------------------------------------------------
// CrossvalComparison
// ---------------------------------------------------------------------------

/// Stateless comparison utilities for cross-validation.
pub struct CrossvalComparison;

impl CrossvalComparison {
    /// Compare two token sequences.
    ///
    /// Returns `(match_count, total, exact_match)` where `total` is the
    /// length of the longer sequence and `match_count` is the number of
    /// positions that agree (shorter sequence is implicitly padded with
    /// mismatches).
    pub fn compare_tokens(expected: &[u32], actual: &[u32]) -> (usize, usize, bool) {
        let total = expected.len().max(actual.len());
        if total == 0 {
            return (0, 0, true);
        }
        let match_count = expected.iter().zip(actual.iter()).filter(|(e, a)| e == a).count();
        let exact_match = match_count == total && expected.len() == actual.len();
        (match_count, total, exact_match)
    }

    /// Compare two logit vectors element-wise.
    ///
    /// Uses the standard `|a - b| <= atol + rtol * |b|` tolerance check
    /// (matching NumPy `allclose` semantics).
    ///
    /// Returns `(max_diff, mean_diff, within_tolerance)`.
    pub fn compare_logits(
        expected: &[f32],
        actual: &[f32],
        atol: f32,
        rtol: f32,
    ) -> (f32, f32, bool) {
        if expected.is_empty() && actual.is_empty() {
            return (0.0, 0.0, true);
        }
        let len = expected.len().min(actual.len());
        if len == 0 {
            // One is empty, the other is not — cannot be within tolerance.
            return (f32::INFINITY, f32::INFINITY, false);
        }

        let mut max_diff: f32 = 0.0;
        let mut sum_diff: f32 = 0.0;
        let mut all_within = true;

        for i in 0..len {
            let diff = (expected[i] - actual[i]).abs();
            max_diff = max_diff.max(diff);
            sum_diff += diff;
            if diff > atol + rtol * expected[i].abs() {
                all_within = false;
            }
        }

        // Length mismatch means extra elements are unmatched.
        if expected.len() != actual.len() {
            all_within = false;
        }

        let mean_diff = sum_diff / len as f32;
        (max_diff, mean_diff, all_within)
    }

    /// Full comparison of a reference output against actual inference output.
    pub fn compare_outputs(
        reference: &ReferenceOutput,
        actual_tokens: &[u32],
        actual_logits: &[Vec<f32>],
        config: &DenseCrossvalConfig,
    ) -> CrossvalResult {
        let (_, _, token_match) = Self::compare_tokens(&reference.token_ids, actual_tokens);

        let mut max_logit_diff: f32 = 0.0;
        let mut sum_logit_diff: f32 = 0.0;
        let mut logit_steps: usize = 0;
        let mut logits_within = true;

        for (ref_step, act_step) in reference.logits.iter().zip(actual_logits.iter()) {
            let (step_max, step_mean, step_ok) = Self::compare_logits(
                ref_step,
                act_step,
                config.tolerance_atol,
                config.tolerance_rtol,
            );
            max_logit_diff = max_logit_diff.max(step_max);
            sum_logit_diff += step_mean;
            logit_steps += 1;
            if !step_ok {
                logits_within = false;
            }
        }
        // If step counts differ, logits cannot fully match.
        if reference.logits.len() != actual_logits.len() {
            logits_within = false;
        }

        let mean_logit_diff =
            if logit_steps > 0 { sum_logit_diff / logit_steps as f32 } else { 0.0 };

        let passed =
            (!config.check_tokens || token_match) && (!config.check_logits || logits_within);

        CrossvalResult {
            prompt: reference.prompt.clone(),
            expected_tokens: reference.token_ids.clone(),
            actual_tokens: actual_tokens.to_vec(),
            token_match,
            max_logit_diff,
            mean_logit_diff,
            passed,
        }
    }
}

// ---------------------------------------------------------------------------
// GoldenFixture
// ---------------------------------------------------------------------------

/// A saved reference output for regression testing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenFixture {
    /// Model name (e.g., "microsoft/phi-4-mini").
    pub model_name: String,
    /// Architecture family (e.g., "Phi4", "LLaMA3").
    pub architecture: String,
    /// The prompt used to generate the fixture.
    pub prompt: String,
    /// Expected token IDs for the first `N` generation steps.
    pub expected_tokens: Vec<u32>,
    /// Top-k logits from the first generation step (for sanity checks).
    pub expected_first_logits: Vec<f32>,
    /// Absolute tolerance for logit comparison against this fixture.
    pub tolerance: f32,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- DenseCrossvalConfig defaults ----------------------------------------

    #[test]
    fn test_config_defaults() {
        let cfg = DenseCrossvalConfig::default();
        assert_eq!(cfg.reference_backend, "pytorch");
        assert!((cfg.tolerance_atol - 1e-5).abs() < f32::EPSILON);
        assert!((cfg.tolerance_rtol - 1e-4).abs() < f32::EPSILON);
        assert!(cfg.test_prompts.is_empty());
        assert_eq!(cfg.max_tokens, 8);
        assert!(cfg.check_logits);
        assert!(cfg.check_tokens);
    }

    #[test]
    fn test_config_custom() {
        let cfg = DenseCrossvalConfig {
            reference_backend: "onnxruntime".to_string(),
            tolerance_atol: 1e-3,
            tolerance_rtol: 1e-2,
            test_prompts: vec!["hello".into()],
            max_tokens: 16,
            check_logits: false,
            check_tokens: true,
        };
        assert_eq!(cfg.reference_backend, "onnxruntime");
        assert_eq!(cfg.max_tokens, 16);
        assert!(!cfg.check_logits);
    }

    // -- CrossvalResult construction -----------------------------------------

    #[test]
    fn test_result_pass() {
        let r = CrossvalResult {
            prompt: "hi".into(),
            expected_tokens: vec![1, 2, 3],
            actual_tokens: vec![1, 2, 3],
            token_match: true,
            max_logit_diff: 1e-6,
            mean_logit_diff: 5e-7,
            passed: true,
        };
        assert!(r.passed);
        assert!(r.token_match);
    }

    #[test]
    fn test_result_fail() {
        let r = CrossvalResult {
            prompt: "hi".into(),
            expected_tokens: vec![1, 2, 3],
            actual_tokens: vec![1, 2, 4],
            token_match: false,
            max_logit_diff: 0.5,
            mean_logit_diff: 0.2,
            passed: false,
        };
        assert!(!r.passed);
        assert!(!r.token_match);
    }

    // -- compare_tokens ------------------------------------------------------

    #[test]
    fn test_compare_tokens_exact_match() {
        let (m, t, exact) = CrossvalComparison::compare_tokens(&[1, 2, 3], &[1, 2, 3]);
        assert_eq!(m, 3);
        assert_eq!(t, 3);
        assert!(exact);
    }

    #[test]
    fn test_compare_tokens_partial_match() {
        let (m, t, exact) = CrossvalComparison::compare_tokens(&[1, 2, 3], &[1, 2, 4]);
        assert_eq!(m, 2);
        assert_eq!(t, 3);
        assert!(!exact);
    }

    #[test]
    fn test_compare_tokens_no_match() {
        let (m, t, exact) = CrossvalComparison::compare_tokens(&[1, 2, 3], &[4, 5, 6]);
        assert_eq!(m, 0);
        assert_eq!(t, 3);
        assert!(!exact);
    }

    #[test]
    fn test_compare_tokens_empty() {
        let (m, t, exact) = CrossvalComparison::compare_tokens(&[], &[]);
        assert_eq!(m, 0);
        assert_eq!(t, 0);
        assert!(exact);
    }

    #[test]
    fn test_compare_tokens_length_mismatch() {
        let (m, t, exact) = CrossvalComparison::compare_tokens(&[1, 2], &[1, 2, 3]);
        assert_eq!(m, 2);
        assert_eq!(t, 3);
        assert!(!exact);
    }

    // -- compare_logits ------------------------------------------------------

    #[test]
    fn test_compare_logits_within_tolerance() {
        let expected = vec![1.0, 2.0, 3.0];
        let actual = vec![1.0 + 1e-6, 2.0 - 1e-6, 3.0 + 1e-6];
        let (max_d, _mean_d, ok) =
            CrossvalComparison::compare_logits(&expected, &actual, 1e-5, 1e-4);
        assert!(ok);
        assert!(max_d < 1e-5);
    }

    #[test]
    fn test_compare_logits_outside_tolerance() {
        let expected = vec![1.0, 2.0, 3.0];
        let actual = vec![1.0, 2.0, 4.0];
        let (max_d, _mean_d, ok) =
            CrossvalComparison::compare_logits(&expected, &actual, 1e-5, 1e-4);
        assert!(!ok);
        assert!((max_d - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_compare_logits_mixed() {
        let expected = vec![1.0, 2.0, 3.0];
        // First two within tolerance, third outside.
        let actual = vec![1.0 + 1e-7, 2.0 - 1e-7, 3.5];
        let (_max_d, _mean_d, ok) =
            CrossvalComparison::compare_logits(&expected, &actual, 1e-5, 1e-4);
        assert!(!ok);
    }

    #[test]
    fn test_compare_logits_relative_vs_absolute() {
        // diff = 0.01; atol = 0.001, rtol = 0.1
        // threshold = 0.001 + 0.1 * |10.0| = 1.001 → 0.01 < 1.001 → passes
        let expected = vec![10.0];
        let actual = vec![10.01];
        let (_, _, ok) = CrossvalComparison::compare_logits(&expected, &actual, 0.001, 0.1);
        assert!(ok);

        // Same diff but small expected value: threshold = 0.001 + 0.1 * 0.01 = 0.002
        // 0.01 > 0.002 → fails
        let expected2 = vec![0.01];
        let actual2 = vec![0.02];
        let (_, _, ok2) = CrossvalComparison::compare_logits(&expected2, &actual2, 0.001, 0.1);
        assert!(!ok2);
    }

    #[test]
    fn test_compare_logits_empty() {
        let (max_d, mean_d, ok) = CrossvalComparison::compare_logits(&[], &[], 1e-5, 1e-4);
        assert!(ok);
        assert_eq!(max_d, 0.0);
        assert_eq!(mean_d, 0.0);
    }

    #[test]
    fn test_compare_logits_length_mismatch() {
        let expected = vec![1.0, 2.0];
        let actual = vec![1.0];
        let (_, _, ok) = CrossvalComparison::compare_logits(&expected, &actual, 1e-5, 1e-4);
        assert!(!ok);
    }

    #[test]
    fn test_compare_logits_exact_zero_diff() {
        let v = vec![1.0, 2.0, 3.0];
        let (max_d, mean_d, ok) = CrossvalComparison::compare_logits(&v, &v, 1e-5, 1e-4);
        assert!(ok);
        assert_eq!(max_d, 0.0);
        assert_eq!(mean_d, 0.0);
    }

    #[test]
    fn test_compare_logits_at_boundary() {
        // diff exactly at atol boundary (no rtol contribution for expected=0)
        let expected = vec![0.0];
        let actual = vec![1e-5];
        let (_, _, ok) = CrossvalComparison::compare_logits(&expected, &actual, 1e-5, 0.0);
        assert!(ok);
    }

    #[test]
    fn test_compare_logits_just_over_boundary() {
        let expected = vec![0.0];
        let actual = vec![1e-5 + 1e-10];
        let (_, _, ok) = CrossvalComparison::compare_logits(&expected, &actual, 1e-5, 0.0);
        // Just over boundary — should fail.
        assert!(!ok);
    }

    // -- compare_outputs (full pipeline) -------------------------------------

    #[test]
    fn test_compare_outputs_pass() {
        let reference = ReferenceOutput {
            prompt: "test".into(),
            token_ids: vec![10, 20, 30],
            logits: vec![vec![0.1, 0.2], vec![0.3, 0.4], vec![0.5, 0.6]],
            backend: "pytorch".into(),
            model_name: "test-model".into(),
        };
        let actual_tokens = vec![10, 20, 30];
        let actual_logits = vec![vec![0.1, 0.2], vec![0.3, 0.4], vec![0.5, 0.6]];
        let cfg = DenseCrossvalConfig::default();

        let result =
            CrossvalComparison::compare_outputs(&reference, &actual_tokens, &actual_logits, &cfg);
        assert!(result.passed);
        assert!(result.token_match);
        assert_eq!(result.max_logit_diff, 0.0);
    }

    #[test]
    fn test_compare_outputs_token_mismatch() {
        let reference = ReferenceOutput {
            prompt: "test".into(),
            token_ids: vec![10, 20, 30],
            logits: vec![vec![0.1], vec![0.2], vec![0.3]],
            backend: "pytorch".into(),
            model_name: "m".into(),
        };
        let actual_tokens = vec![10, 20, 99];
        let actual_logits = vec![vec![0.1], vec![0.2], vec![0.3]];
        let cfg = DenseCrossvalConfig::default();

        let result =
            CrossvalComparison::compare_outputs(&reference, &actual_tokens, &actual_logits, &cfg);
        assert!(!result.passed);
        assert!(!result.token_match);
    }

    // -- CrossvalReport ------------------------------------------------------

    #[test]
    fn test_report_all_pass() {
        let results = vec![
            CrossvalResult {
                prompt: "a".into(),
                expected_tokens: vec![1],
                actual_tokens: vec![1],
                token_match: true,
                max_logit_diff: 1e-6,
                mean_logit_diff: 1e-6,
                passed: true,
            },
            CrossvalResult {
                prompt: "b".into(),
                expected_tokens: vec![2],
                actual_tokens: vec![2],
                token_match: true,
                max_logit_diff: 2e-6,
                mean_logit_diff: 2e-6,
                passed: true,
            },
        ];
        let report = CrossvalReport::from_results(DenseCrossvalConfig::default(), results, 100);
        assert!(report.overall_pass);
        assert_eq!(report.total_prompts, 2);
        assert_eq!(report.passed_prompts, 2);
    }

    #[test]
    fn test_report_some_fail() {
        let results = vec![
            CrossvalResult {
                prompt: "a".into(),
                expected_tokens: vec![1],
                actual_tokens: vec![1],
                token_match: true,
                max_logit_diff: 1e-6,
                mean_logit_diff: 1e-6,
                passed: true,
            },
            CrossvalResult {
                prompt: "b".into(),
                expected_tokens: vec![2],
                actual_tokens: vec![3],
                token_match: false,
                max_logit_diff: 0.5,
                mean_logit_diff: 0.3,
                passed: false,
            },
        ];
        let report = CrossvalReport::from_results(DenseCrossvalConfig::default(), results, 200);
        assert!(!report.overall_pass);
        assert_eq!(report.passed_prompts, 1);
        assert!((report.max_logit_diff_across_all - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_report_none_pass() {
        let results = vec![CrossvalResult {
            prompt: "x".into(),
            expected_tokens: vec![1],
            actual_tokens: vec![9],
            token_match: false,
            max_logit_diff: 1.0,
            mean_logit_diff: 1.0,
            passed: false,
        }];
        let report = CrossvalReport::from_results(DenseCrossvalConfig::default(), results, 50);
        assert!(!report.overall_pass);
        assert_eq!(report.passed_prompts, 0);
    }

    // -- GoldenFixture per architecture --------------------------------------

    #[test]
    fn test_golden_fixture_phi4() {
        let f = GoldenFixture {
            model_name: "microsoft/phi-4-mini".into(),
            architecture: "Phi4".into(),
            prompt: "Hello".into(),
            expected_tokens: vec![15496, 11, 314],
            expected_first_logits: vec![0.1, -0.2, 0.3],
            tolerance: 1e-4,
        };
        assert_eq!(f.architecture, "Phi4");
    }

    #[test]
    fn test_golden_fixture_llama3() {
        let f = GoldenFixture {
            model_name: "meta-llama/Llama-3-8B".into(),
            architecture: "LLaMA3".into(),
            prompt: "The".into(),
            expected_tokens: vec![450, 1234],
            expected_first_logits: vec![0.5, -0.1],
            tolerance: 1e-4,
        };
        assert_eq!(f.architecture, "LLaMA3");
    }

    #[test]
    fn test_golden_fixture_qwen2() {
        let f = GoldenFixture {
            model_name: "Qwen/Qwen2-7B".into(),
            architecture: "Qwen2".into(),
            prompt: "What".into(),
            expected_tokens: vec![100, 200],
            expected_first_logits: vec![0.2],
            tolerance: 1e-3,
        };
        assert_eq!(f.architecture, "Qwen2");
    }

    #[test]
    fn test_golden_fixture_gemma() {
        let f = GoldenFixture {
            model_name: "google/gemma-2b".into(),
            architecture: "Gemma".into(),
            prompt: "Explain".into(),
            expected_tokens: vec![55, 66, 77],
            expected_first_logits: vec![-0.1, 0.4],
            tolerance: 1e-4,
        };
        assert_eq!(f.architecture, "Gemma");
    }

    #[test]
    fn test_golden_fixture_mistral() {
        let f = GoldenFixture {
            model_name: "mistralai/Mistral-7B-v0.1".into(),
            architecture: "Mistral".into(),
            prompt: "Once".into(),
            expected_tokens: vec![9038, 2501],
            expected_first_logits: vec![0.0, 0.1, -0.3],
            tolerance: 1e-5,
        };
        assert_eq!(f.architecture, "Mistral");
    }

    // -- ReferenceOutput serialization round-trip -----------------------------

    #[test]
    fn test_reference_output_serde_roundtrip() {
        let original = ReferenceOutput {
            prompt: "round-trip test".into(),
            token_ids: vec![42, 99, 7],
            logits: vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]],
            backend: "numpy".into(),
            model_name: "test-model".into(),
        };
        let json = serde_json::to_string(&original).expect("serialize");
        let decoded: ReferenceOutput = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(decoded.prompt, original.prompt);
        assert_eq!(decoded.token_ids, original.token_ids);
        assert_eq!(decoded.logits, original.logits);
        assert_eq!(decoded.backend, original.backend);
        assert_eq!(decoded.model_name, original.model_name);
    }

    // -- Edge cases ----------------------------------------------------------

    #[test]
    fn test_compare_tokens_single_token() {
        let (m, t, exact) = CrossvalComparison::compare_tokens(&[42], &[42]);
        assert_eq!(m, 1);
        assert_eq!(t, 1);
        assert!(exact);
    }

    #[test]
    fn test_compare_logits_single_element() {
        let (max_d, mean_d, ok) = CrossvalComparison::compare_logits(&[1.0], &[1.0], 1e-5, 1e-4);
        assert!(ok);
        assert_eq!(max_d, 0.0);
        assert_eq!(mean_d, 0.0);
    }

    #[test]
    fn test_compare_outputs_empty_logits() {
        let reference = ReferenceOutput {
            prompt: "empty".into(),
            token_ids: vec![1],
            logits: vec![],
            backend: "pytorch".into(),
            model_name: "m".into(),
        };
        let cfg = DenseCrossvalConfig { check_logits: false, ..DenseCrossvalConfig::default() };
        let result = CrossvalComparison::compare_outputs(&reference, &[1], &[], &cfg);
        assert!(result.passed);
    }

    #[test]
    fn test_compare_outputs_mismatched_logit_steps() {
        let reference = ReferenceOutput {
            prompt: "mismatch".into(),
            token_ids: vec![1, 2],
            logits: vec![vec![0.1], vec![0.2]],
            backend: "pytorch".into(),
            model_name: "m".into(),
        };
        let cfg = DenseCrossvalConfig::default();
        // Only one step of actual logits vs two reference steps.
        let result = CrossvalComparison::compare_outputs(&reference, &[1, 2], &[vec![0.1]], &cfg);
        assert!(!result.passed);
    }
}

//! Test scaffolding for lazy_static → std::sync::OnceLock migration
//!
//! This test suite validates the migration from `lazy_static` to `std::sync::OnceLock`
//! for LayerNorm regex pattern initialization in `bitnet_st_tools::common::is_ln_gamma()`.
//!
//! **Test Organization**:
//! - **Baseline Tests** (active): Validate current lazy_static behavior
//! - **Post-Migration Tests** (#[ignore]): Activate after OnceLock migration
//!
//! **Related Specification**: `/tmp/phase2_lazy_static_upgrade_spec.md`
//! **Target Function**: `bitnet_st_tools::common::is_ln_gamma()`

use bitnet_st_tools::common::is_ln_gamma;
use std::thread;

// ============================================================================
// BASELINE TESTS (Active - Run with Current lazy_static Implementation)
// ============================================================================

/// Tests feature spec: phase2_lazy_static_upgrade_spec.md#AC3-functionality-remains-identical
///
/// Validates LayerNorm regex matching behavior with positive cases.
/// This test captures the baseline behavior before migration.
#[test]
fn test_layernorm_regex_baseline_positive_matches() {
    // Test cases that SHOULD match LayerNorm patterns
    let positive_cases = vec![
        // Standard transformer LayerNorm patterns
        "model.layers.0.input_layernorm.weight",
        "model.layers.1.post_attention_layernorm.weight",
        "model.layers.31.input_layernorm.weight",
        // Attention/FFN norm patterns
        "transformer.blocks.0.attn_norm.weight",
        "transformer.blocks.1.ffn_norm.weight",
        "decoder.layer.5.ffn_layernorm.weight",
        // RMS norm pattern
        "model.layers.0.rms_norm.weight",
        // Final norm patterns
        "decoder.final_layernorm.weight",
        "encoder.final_norm.weight",
        "model.norm.weight",
        // Path separator variations (slash vs dot)
        "layers/0/ffn_layernorm.weight",
        "layers.0.ffn_layernorm.weight",
        // Edge case: standalone norm
        "norm.weight",
        "rms_norm.weight",
    ];

    for case in positive_cases {
        assert!(is_ln_gamma(case), "Expected '{}' to match LayerNorm pattern (baseline)", case);
    }
}

/// Tests feature spec: phase2_lazy_static_upgrade_spec.md#AC3-functionality-remains-identical
///
/// Validates LayerNorm regex matching behavior with negative cases.
/// This test captures the baseline behavior before migration.
#[test]
fn test_layernorm_regex_baseline_negative_matches() {
    // Test cases that SHOULD NOT match LayerNorm patterns
    let negative_cases = vec![
        // Wrong suffix (.bias instead of .weight)
        "model.layers.0.input_layernorm.bias",
        "model.layers.1.post_attention_layernorm.bias",
        // No norm keyword (embedding, head, projection layers)
        "model.embed_tokens.weight",
        "model.lm_head.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        // Missing .weight suffix
        "model.layers.0.input_layernorm",
        "model.layers.0.input_layernorm_gamma",
        // No .weight at all
        "model.layers.0.input_layernorm.scale",
        // Path-only (no tensor name)
        "model.layers.0",
        // Empty string
        "",
    ];

    for case in negative_cases {
        assert!(
            !is_ln_gamma(case),
            "Expected '{}' to NOT match LayerNorm pattern (baseline)",
            case
        );
    }
}

/// Tests feature spec: phase2_lazy_static_upgrade_spec.md#test-3-thread-safety-validation
///
/// Validates that lazy_static regex is thread-safe under concurrent access.
/// This establishes the thread-safety baseline that OnceLock must maintain.
#[test]
fn test_layernorm_regex_thread_safety_baseline() {
    const NUM_THREADS: usize = 10;
    const ITERATIONS_PER_THREAD: usize = 100;

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|thread_id| {
            thread::spawn(move || {
                for iteration in 0..ITERATIONS_PER_THREAD {
                    // Test different LayerNorm patterns concurrently
                    let ln_name = format!("model.layers.{}.input_layernorm.weight", iteration);
                    let non_ln_name = format!("model.layers.{}.self_attn.q_proj.weight", iteration);

                    // Verify correct matches
                    assert!(
                        is_ln_gamma(&ln_name),
                        "Thread {} iteration {}: Expected '{}' to match",
                        thread_id,
                        iteration,
                        ln_name
                    );

                    assert!(
                        !is_ln_gamma(&non_ln_name),
                        "Thread {} iteration {}: Expected '{}' to NOT match",
                        thread_id,
                        iteration,
                        non_ln_name
                    );
                }
            })
        })
        .collect();

    // Wait for all threads to complete
    for handle in handles {
        handle.join().expect("Thread should complete without panic");
    }
}

/// Tests feature spec: phase2_lazy_static_upgrade_spec.md#AC3-functionality-remains-identical
///
/// Validates regex pattern with real-world transformer architecture tensor names.
/// This test uses realistic patterns from popular models (LLaMA, GPT, etc.).
#[test]
fn test_layernorm_regex_real_world_patterns() {
    // Real tensor names from LLaMA-style models
    let llama_patterns = vec![
        ("model.layers.0.input_layernorm.weight", true),
        ("model.layers.0.post_attention_layernorm.weight", true),
        ("model.norm.weight", true),
        ("model.layers.0.self_attn.q_proj.weight", false),
        ("model.embed_tokens.weight", false),
    ];

    // Real tensor names from GPT-style models
    let gpt_patterns = vec![
        ("transformer.h.0.ln_1.weight", false), // GPT uses ln_1/ln_2, not matching our pattern
        ("transformer.h.0.attn.c_attn.weight", false),
        ("transformer.ln_f.weight", false), // ln_f not in our pattern list
    ];

    // Real tensor names from BERT-style models
    let bert_patterns = vec![
        ("encoder.layer.0.attention.output.LayerNorm.weight", false), // Capitalized LayerNorm
        ("encoder.layer.0.output.LayerNorm.weight", false),
    ];

    // Combine and test all patterns
    let all_patterns = llama_patterns.into_iter().chain(gpt_patterns).chain(bert_patterns);

    for (tensor_name, expected_match) in all_patterns {
        let actual_match = is_ln_gamma(tensor_name);
        assert_eq!(
            actual_match, expected_match,
            "Pattern mismatch for '{}': expected {}, got {}",
            tensor_name, expected_match, actual_match
        );
    }
}

/// Tests feature spec: phase2_lazy_static_upgrade_spec.md#AC3-functionality-remains-identical
///
/// Validates edge cases and boundary conditions for regex matching.
#[test]
fn test_layernorm_regex_edge_cases() {
    // Edge case 1: Very long paths
    let long_path =
        format!("model.{}.input_layernorm.weight", "layers.0.sublayers.1.blocks.2.attention");
    assert!(is_ln_gamma(&long_path), "Should match long path with LayerNorm pattern");

    // Edge case 2: Multiple separators
    let multi_sep = "model/layers/0.input_layernorm.weight";
    assert!(is_ln_gamma(multi_sep), "Should match with mixed separators");

    // Edge case 3: Short names
    assert!(is_ln_gamma("norm.weight"), "Should match minimal pattern");
    assert!(is_ln_gamma("final_norm.weight"), "Should match final_norm");

    // Edge case 4: Case sensitivity
    assert!(
        !is_ln_gamma("model.layers.0.INPUT_LAYERNORM.weight"),
        "Should NOT match uppercase (regex is case-sensitive)"
    );

    // Edge case 5: Partial suffix matches
    assert!(
        !is_ln_gamma("model.layers.0.input_layernorm.weight.extra"),
        "Should NOT match if extra suffix after .weight"
    );

    // Edge case 6: Unicode in path (should work with regex)
    let unicode_path = "model.層.0.input_layernorm.weight";
    // This should match since the pattern only cares about the final norm.weight part
    assert!(is_ln_gamma(unicode_path), "Should handle unicode in path components");
}

// ============================================================================
// POST-MIGRATION TESTS (Ignored - Activate After OnceLock Migration)
// ============================================================================

/// Tests feature spec: phase2_lazy_static_upgrade_spec.md#test-4-oncelock-initialization-semantics
///
/// Validates that OnceLock provides single initialization guarantee.
/// This test should be activated AFTER the migration to OnceLock.
///
/// **Status**: Active - Validates OnceLock behavior
#[test]
fn test_oncelock_single_initialization() {
    // This test validates that the regex is compiled exactly once,
    // even when is_ln_gamma() is called multiple times.
    //
    // With OnceLock, we expect:
    // 1. First call triggers regex compilation
    // 2. Subsequent calls reuse the cached Regex
    // 3. No additional compilation overhead

    // Call is_ln_gamma multiple times
    for i in 0..1000 {
        let name = format!("model.layers.{}.input_layernorm.weight", i);
        assert!(is_ln_gamma(&name), "Iteration {}: Should match", i);
    }

    // Note: This test validates behavior, not implementation details.
    // To verify single initialization at the implementation level,
    // use `AtomicUsize` counter in the initialization closure during development.
}

/// Tests feature spec: phase2_lazy_static_upgrade_spec.md#test-5-oncelock-thread-safety
///
/// Validates OnceLock maintains thread-safety under concurrent access.
/// This test should be activated AFTER the migration to OnceLock.
///
/// **Status**: Active - Validates OnceLock thread-safety
#[test]
fn test_oncelock_concurrent_access() {
    const NUM_THREADS: usize = 100;
    const ITERATIONS_PER_THREAD: usize = 50;

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|thread_id| {
            thread::spawn(move || {
                // All threads try to access is_ln_gamma concurrently
                // This stresses OnceLock's initialization synchronization
                for iteration in 0..ITERATIONS_PER_THREAD {
                    let name = format!(
                        "model.layers.{}.input_layernorm.weight",
                        thread_id * 100 + iteration
                    );
                    let result = is_ln_gamma(&name);
                    assert!(result, "Thread {} failed at iteration {}", thread_id, iteration);
                }
            })
        })
        .collect();

    // Wait for all threads to complete
    for (idx, handle) in handles.into_iter().enumerate() {
        handle.join().unwrap_or_else(|_| panic!("Thread {} panicked", idx));
    }
}

/// Tests feature spec: phase2_lazy_static_upgrade_spec.md#AC2-replace-lazy-static-macro
///
/// Validates that OnceLock access pattern (`get_or_init`) works correctly.
/// This test should be activated AFTER the migration to OnceLock.
///
/// **Status**: Active - Validates OnceLock access pattern
#[test]
fn test_oncelock_access_pattern() {
    // Test that OnceLock's get_or_init pattern works correctly
    // by calling is_ln_gamma in various scenarios

    // Scenario 1: First call (triggers initialization)
    assert!(is_ln_gamma("model.layers.0.input_layernorm.weight"));

    // Scenario 2: Immediate subsequent call (uses cached value)
    assert!(is_ln_gamma("model.layers.1.post_attention_layernorm.weight"));

    // Scenario 3: Negative match (still uses cached Regex)
    assert!(!is_ln_gamma("model.embed_tokens.weight"));

    // Scenario 4: Many rapid calls (stress test cached access)
    for i in 0..100 {
        let name = if i % 2 == 0 {
            format!("model.layers.{}.input_layernorm.weight", i)
        } else {
            format!("model.layers.{}.mlp.gate_proj.weight", i)
        };
        let _ = is_ln_gamma(&name);
    }
}

/// Tests feature spec: phase2_lazy_static_upgrade_spec.md#test-6-regex-matching-parity
///
/// Validates post-migration regex matching produces identical results.
/// This test should be activated AFTER the migration to OnceLock.
///
/// **Status**: Active - Validates regex matching parity
#[test]
fn test_regex_pattern_parity_post_migration() {
    // This test ensures the OnceLock migration didn't change regex behavior.
    // It runs the same test cases as baseline tests.

    // Positive cases (should match)
    let positive_cases = vec![
        "model.layers.0.input_layernorm.weight",
        "model.layers.1.post_attention_layernorm.weight",
        "transformer.blocks.0.attn_norm.weight",
        "transformer.blocks.1.ffn_norm.weight",
        "decoder.final_layernorm.weight",
        "encoder.final_norm.weight",
        "model.norm.weight",
    ];

    for case in positive_cases {
        assert!(is_ln_gamma(case), "Post-migration: Expected '{}' to match", case);
    }

    // Negative cases (should NOT match)
    let negative_cases = vec![
        "model.layers.0.input_layernorm.bias",
        "model.embed_tokens.weight",
        "model.lm_head.weight",
        "model.layers.0.self_attn.q_proj.weight",
    ];

    for case in negative_cases {
        assert!(!is_ln_gamma(case), "Post-migration: Expected '{}' to NOT match", case);
    }
}

/// Tests feature spec: phase2_lazy_static_upgrade_spec.md#test-8-performance-parity
///
/// Validates no performance regression from OnceLock migration.
/// This test should be activated AFTER the migration to OnceLock.
///
/// **Status**: Active - Validates performance parity
#[test]
fn test_oncelock_performance_parity() {
    use std::time::Instant;

    // Warm-up: ensure regex is initialized
    let _ = is_ln_gamma("model.norm.weight");

    // Benchmark: 10k iterations should complete quickly
    let test_cases = vec![
        "model.layers.0.input_layernorm.weight",
        "model.layers.1.post_attention_layernorm.weight",
        "model.layers.2.self_attn.q_proj.weight", // Non-matching (exercises fast path)
    ];

    let start = Instant::now();
    for _ in 0..10_000 {
        for case in &test_cases {
            let _ = is_ln_gamma(case);
        }
    }
    let duration = start.elapsed();

    // Expect < 100ms for 30k calls (10k iterations × 3 cases)
    // This is conservative; actual performance should be much faster
    assert!(
        duration.as_millis() < 100,
        "Performance regression detected: {}ms (expected < 100ms)",
        duration.as_millis()
    );

    println!("✓ Performance: 30k calls completed in {:?}", duration);
}

// ============================================================================
// INTEGRATION TESTS (Active - Verify Current Implementation)
// ============================================================================

/// Tests feature spec: phase2_lazy_static_upgrade_spec.md#AC5-all-tests-pass
///
/// Integration test: Validates is_ln_gamma works correctly in realistic usage scenarios.
/// This test simulates how the function is used in st-ln-inspect and st-merge-ln-f16 binaries.
#[test]
fn test_layernorm_detection_integration() {
    // Simulate a SafeTensors model with mixed tensor types
    let mock_model_tensors = vec![
        // LayerNorm tensors (should be detected)
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.1.input_layernorm.weight",
        "model.layers.1.post_attention_layernorm.weight",
        "model.norm.weight",
        // Non-LayerNorm tensors (should be filtered out)
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.1.self_attn.q_proj.weight",
        "model.layers.1.mlp.gate_proj.weight",
        "model.lm_head.weight",
    ];

    // Filter LayerNorm tensors using is_ln_gamma
    let detected_ln_tensors: Vec<&str> =
        mock_model_tensors.iter().filter(|&&name| is_ln_gamma(name)).copied().collect();

    // Verify correct detection count
    assert_eq!(
        detected_ln_tensors.len(),
        5,
        "Should detect exactly 5 LayerNorm tensors, found: {:?}",
        detected_ln_tensors
    );

    // Verify all detected tensors are LayerNorm patterns
    let expected_ln_tensors = vec![
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.1.input_layernorm.weight",
        "model.layers.1.post_attention_layernorm.weight",
        "model.norm.weight",
    ];

    for expected in &expected_ln_tensors {
        assert!(
            detected_ln_tensors.contains(expected),
            "Expected LayerNorm tensor '{}' not detected",
            expected
        );
    }
}

/// Tests feature spec: phase2_lazy_static_upgrade_spec.md#AC3-functionality-remains-identical
///
/// Regression test: Validates that fast-path optimization (ends_with(".weight")) works.
#[test]
fn test_layernorm_fast_path_optimization() {
    // Test cases that should be rejected by fast-path (before regex evaluation)
    let fast_path_rejects = vec![
        "model.layers.0.input_layernorm.bias", // ends with .bias
        "model.layers.0.input_layernorm",      // no suffix
        "model.norm.scale",                    // ends with .scale
        "model.norm",                          // no suffix
        "",                                    // empty string
    ];

    for case in fast_path_rejects {
        // These should return false before regex is evaluated
        assert!(!is_ln_gamma(case), "Fast-path should reject '{}'", case);
    }

    // Test cases that pass fast-path but fail regex
    let regex_rejects = vec![
        "model.embed_tokens.weight",              // .weight but not LayerNorm
        "model.layers.0.self_attn.q_proj.weight", // .weight but not LayerNorm
    ];

    for case in regex_rejects {
        assert!(!is_ln_gamma(case), "Regex should reject '{}' (passed fast-path)", case);
    }
}

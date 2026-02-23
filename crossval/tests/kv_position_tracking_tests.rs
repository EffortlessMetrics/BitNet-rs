//! Comprehensive TDD test scaffolding for manual KV position tracking
//!
//! **Specification**: `docs/specs/cpp-wrapper-kv-position-tracking.md`
//!
//! This test suite validates the manual KV cache position tracking implementation
//! that replaces the removed `llama_get_kv_cache_token_count()` API. It ensures:
//!
//! - Socket 1 persistent context with `n_past` field
//! - Position initialization, increment, and validation
//! - Multi-turn conversation support with KV cache reuse
//! - Rust vs C++ parity with position tracking
//! - Performance benefits from KV cache reuse (10-100× speedup)
//!
//! **Test Organization**:
//! - Category 1: Unit Tests (8 tests) - FR1-FR4, AR1-AR3
//! - Category 2: Integration Tests (4 tests) - TR2
//! - Category 3: Parity Tests (3 tests) - PR1-PR3
//! - Category 4: Performance Tests (2 tests) - TR3
//!
//! **Feature Gates**: `#[cfg(feature = "crossval")]` for FFI-dependent tests

use anyhow::{Context, Result};
use serial_test::serial;

// ============================================================================
// Test Helpers and Utilities
// ============================================================================

/// Helper: Create a test context for Socket 1 persistent evaluation
///
/// This helper wraps `bitnet_cpp_init_context()` FFI call with proper
/// error handling and resource cleanup.
///
/// **Spec Reference**: FR1 - Position Initialization
#[cfg(feature = "crossval")]
fn create_test_context(
    model_path: &std::path::Path,
    n_ctx: i32,
    n_gpu_layers: i32,
) -> Result<*mut BitnetContext> {
    // TODO: Implement FFI wrapper for bitnet_cpp_init_context()
    // Expected behavior:
    // 1. Validate model_path exists
    // 2. Call bitnet_cpp_init_context() with error buffer
    // 3. Verify ctx->n_past = 0 after initialization
    // 4. Return persistent context handle
    unimplemented!("create_test_context: wrap bitnet_cpp_init_context() with error handling")
}

/// Helper: Evaluate tokens with position tracking
///
/// Wraps `bitnet_cpp_eval_with_context()` FFI call, updating position state.
///
/// **Spec Reference**: FR2 - Position Increment, AR2 - Non-const context
#[cfg(feature = "crossval")]
fn eval_tokens(ctx: *mut BitnetContext, tokens: &[i32], seq_id: i32) -> Result<Vec<Vec<f32>>> {
    // TODO: Implement FFI wrapper for bitnet_cpp_eval_with_context()
    // Expected behavior:
    // 1. Validate ctx->n_past + tokens.len() <= ctx->n_ctx (FR3)
    // 2. Call bitnet_cpp_eval_with_context() with mutable ctx
    // 3. Extract logits from output buffer (rows × cols)
    // 4. Verify ctx->n_past incremented by tokens.len() (FR2)
    // 5. Return logits matrix (n_tokens × n_vocab)
    unimplemented!("eval_tokens: wrap bitnet_cpp_eval_with_context() with position validation")
}

/// Helper: Reset KV cache and position tracking
///
/// Wraps `bitnet_cpp_reset_context()` FFI call for new conversations.
///
/// **Spec Reference**: FR4 - Context Reset, AR3 - Reset API
#[cfg(feature = "crossval")]
fn reset_context(ctx: *mut BitnetContext) -> Result<()> {
    // TODO: Implement FFI wrapper for bitnet_cpp_reset_context()
    // Expected behavior:
    // 1. Call bitnet_cpp_reset_context() with error buffer
    // 2. Verify ctx->n_past = 0 after reset
    // 3. Verify KV cache cleared (if API available)
    // 4. Return Ok(()) on success
    unimplemented!("reset_context: wrap bitnet_cpp_reset_context() for KV cache clearing")
}

/// Helper: Query current position from context
///
/// Wraps `bitnet_cpp_get_position()` FFI call (optional API).
///
/// **Spec Reference**: Section 3.4 - Position Query (Optional)
#[cfg(feature = "crossval")]
fn get_position(ctx: *const BitnetContext) -> Result<i32> {
    // TODO: Implement FFI wrapper for bitnet_cpp_get_position()
    // Expected behavior:
    // 1. Call bitnet_cpp_get_position() with output pointer
    // 2. Validate 0 <= n_past <= n_ctx
    // 3. Return current position
    unimplemented!("get_position: wrap bitnet_cpp_get_position() for position query")
}

/// Helper: Assert position equals expected value
///
/// Validates position state matches expected value with clear error messages.
///
/// **Spec Reference**: Appendix B - Position Tracking Invariants
#[cfg(feature = "crossval")]
fn assert_position_equals(ctx: *const BitnetContext, expected: i32) -> Result<()> {
    // TODO: Implement position validation with get_position()
    // Expected behavior:
    // 1. Query current position via get_position()
    // 2. Compare with expected value
    // 3. Return detailed error if mismatch: "Expected n_past={}, got {}"
    unimplemented!("assert_position_equals: validate position state with clear diagnostics")
}

/// Helper: Compare Rust vs C++ position tracking
///
/// Validates parity between Rust and C++ implementations for multi-turn scenarios.
///
/// **Spec Reference**: PR2 - Multi-Turn Parity
#[cfg(feature = "crossval")]
fn compare_rust_cpp_positions(
    rust_logits: &[Vec<f32>],
    cpp_logits: &[Vec<f32>],
    tolerance: f64,
) -> Result<()> {
    // TODO: Implement logits comparison with cosine similarity
    // Expected behavior:
    // 1. Validate rust_logits.len() == cpp_logits.len()
    // 2. Compare each position using cosine similarity
    // 3. Assert similarity >= tolerance (e.g., 0.9999)
    // 4. Return detailed error with divergence position if mismatch
    unimplemented!("compare_rust_cpp_positions: validate parity with cosine similarity threshold")
}

/// Mock BitnetContext for compilation (removed when FFI implemented)
#[cfg(feature = "crossval")]
#[repr(C)]
struct BitnetContext {
    _private: [u8; 0],
}

// ============================================================================
// Category 1: Unit Tests - Position Tracking Correctness
// ============================================================================

/// Test: FR1 - Position initialization to zero
///
/// **Acceptance Criteria**: `n_past = 0` after `bitnet_cpp_init_context()`
///
/// **Spec Reference**: Section 6.1 - Unit Tests: Position Tracking
#[test]
#[ignore = "TODO: Enable when bitnet_cpp_init_context() FFI implemented"]
#[cfg(feature = "crossval")]
#[serial(bitnet_env)]
fn test_position_init_zero() -> Result<()> {
    // Setup: Create persistent context
    let model_path = std::path::Path::new("models/test.gguf");
    let ctx = create_test_context(model_path, 512, 0).context("Failed to create test context")?;

    // Verify: Position initialized to 0 (empty KV cache)
    assert_position_equals(ctx, 0).context("Position should be 0 after context initialization")?;

    // Cleanup
    unsafe {
        // TODO: Call bitnet_cpp_free_context(ctx)
    }

    Ok(())
}

/// Test: FR2 - Position increment after successful evaluation
///
/// **Acceptance Criteria**: `n_past += n_tokens` after `bitnet_cpp_eval_with_context()`
///
/// **Spec Reference**: Section 5.3 - Position Initialization (Prompt)
#[test]
#[ignore = "TODO: Enable when bitnet_cpp_eval_with_context() FFI implemented"]
#[cfg(feature = "crossval")]
#[serial(bitnet_env)]
fn test_position_increment_after_eval() -> Result<()> {
    // Setup: Create context and evaluate tokens
    let model_path = std::path::Path::new("models/test.gguf");
    let ctx = create_test_context(model_path, 512, 0)?;

    // Initial state: n_past = 0
    assert_position_equals(ctx, 0)?;

    // Evaluate 4 tokens
    let tokens1 = vec![1, 2, 3, 4];
    eval_tokens(ctx, &tokens1, 0).context("Failed to evaluate tokens1")?;

    // Verify: Position incremented by 4
    assert_position_equals(ctx, 4).context("Position should be 4 after evaluating 4 tokens")?;

    // Evaluate 2 more tokens
    let tokens2 = vec![5, 6];
    eval_tokens(ctx, &tokens2, 0).context("Failed to evaluate tokens2")?;

    // Verify: Position incremented by 2 (total 6)
    assert_position_equals(ctx, 6)
        .context("Position should be 6 after evaluating 6 tokens total")?;

    // Cleanup
    unsafe {
        // TODO: Call bitnet_cpp_free_context(ctx)
    }

    Ok(())
}

/// Test: FR3 - Position validation (overflow detection)
///
/// **Acceptance Criteria**: Error when `n_past + n_tokens > n_ctx`
///
/// **Spec Reference**: Section 5.6 - Position Overflow Handling
#[test]
#[ignore = "TODO: Enable when position validation implemented"]
#[cfg(feature = "crossval")]
#[serial(bitnet_env)]
fn test_position_validation_bounds() -> Result<()> {
    // Setup: Create small context (n_ctx = 10)
    let model_path = std::path::Path::new("models/test.gguf");
    let ctx = create_test_context(model_path, 10, 0)?;

    // Fill context to capacity (10 tokens)
    let tokens1 = vec![1; 10];
    eval_tokens(ctx, &tokens1, 0).context("Failed to fill context")?;

    // Verify: Position at capacity
    assert_position_equals(ctx, 10)?;

    // Try to overflow (should fail)
    let tokens2 = vec![2];
    let result = eval_tokens(ctx, &tokens2, 0);

    // Assert: Evaluation fails with overflow error
    assert!(result.is_err(), "Expected overflow error when n_past + n_tokens > n_ctx");

    // Verify error message contains "KV cache overflow"
    let error_msg = result.unwrap_err().to_string();
    assert!(
        error_msg.contains("KV cache overflow") || error_msg.contains("overflow"),
        "Error message should mention overflow, got: {}",
        error_msg
    );

    // Verify: Position unchanged after failed evaluation
    assert_position_equals(ctx, 10)
        .context("Position should remain unchanged after failed evaluation")?;

    // Cleanup
    unsafe {
        // TODO: Call bitnet_cpp_free_context(ctx)
    }

    Ok(())
}

/// Test: FR4 - Context reset clears KV cache
///
/// **Acceptance Criteria**: `n_past = 0` after `bitnet_cpp_reset_context()`
///
/// **Spec Reference**: Section 5.5 - Position Reset (New Conversation)
#[test]
#[ignore = "TODO: Enable when bitnet_cpp_reset_context() FFI implemented"]
#[cfg(feature = "crossval")]
#[serial(bitnet_env)]
fn test_position_reset_clears_cache() -> Result<()> {
    // Setup: Create context and fill with tokens
    let model_path = std::path::Path::new("models/test.gguf");
    let ctx = create_test_context(model_path, 512, 0)?;

    // Evaluate 100 tokens
    let tokens = vec![1; 100];
    eval_tokens(ctx, &tokens, 0).context("Failed to evaluate tokens")?;

    // Verify: Position at 100
    assert_position_equals(ctx, 100)?;

    // Reset context
    reset_context(ctx).context("Failed to reset context")?;

    // Verify: Position cleared to 0
    assert_position_equals(ctx, 0).context("Position should be 0 after reset")?;

    // Verify: Can evaluate new tokens without overflow
    let new_tokens = vec![2, 3, 4];
    eval_tokens(ctx, &new_tokens, 0)
        .context("Should be able to evaluate new tokens after reset")?;

    assert_position_equals(ctx, 3).context("Position should be 3 after evaluating 3 new tokens")?;

    // Cleanup
    unsafe {
        // TODO: Call bitnet_cpp_free_context(ctx)
    }

    Ok(())
}

/// Test: AR1 - Context initialization sets position to zero
///
/// **Acceptance Criteria**: Socket 1 API correctly initializes `n_past = 0`
///
/// **Spec Reference**: Section 3.1 - Context Initialization (Socket 1)
#[test]
#[ignore = "TODO: Enable when Socket 1 FFI implemented"]
#[cfg(feature = "crossval")]
#[serial(bitnet_env)]
fn test_context_init_sets_position() -> Result<()> {
    // Setup: Initialize multiple contexts with different parameters
    let model_path = std::path::Path::new("models/test.gguf");

    let ctx1 = create_test_context(model_path, 512, 0)?;
    assert_position_equals(ctx1, 0).context("Context 1 position should be 0")?;

    let ctx2 = create_test_context(model_path, 2048, 0)?;
    assert_position_equals(ctx2, 0).context("Context 2 position should be 0")?;

    // Verify: Position is 0 regardless of n_ctx parameter
    // This validates consistent initialization behavior

    // Cleanup
    unsafe {
        // TODO: Call bitnet_cpp_free_context(ctx1)
        // TODO: Call bitnet_cpp_free_context(ctx2)
    }

    Ok(())
}

/// Test: AR2 - Evaluation updates position state (non-const context)
///
/// **Acceptance Criteria**: `bitnet_cpp_eval_with_context()` takes mutable context
///
/// **Spec Reference**: Section 3.2 - Evaluation with Position Tracking (Socket 3)
#[test]
#[ignore = "TODO: Enable when Socket 3 signature updated to non-const"]
#[cfg(feature = "crossval")]
#[serial(bitnet_env)]
fn test_eval_updates_position_state() -> Result<()> {
    // Setup: Create context
    let model_path = std::path::Path::new("models/test.gguf");
    let ctx = create_test_context(model_path, 512, 0)?;

    // Verify: eval_tokens accepts mutable context (not const)
    // This test validates API contract change from spec Section 3.2

    let tokens = vec![1, 2, 3];
    let _logits = eval_tokens(ctx, &tokens, 0)?;

    // Position should be updated to 3
    assert_position_equals(ctx, 3)?;

    // Cleanup
    unsafe {
        // TODO: Call bitnet_cpp_free_context(ctx)
    }

    Ok(())
}

/// Test: AR3 - Reset API signature and behavior
///
/// **Acceptance Criteria**: `bitnet_cpp_reset_context()` implemented and tested
///
/// **Spec Reference**: Section 3.3 - Context Reset (Socket 1 Extension)
#[test]
#[ignore = "TODO: Enable when reset API implemented"]
#[cfg(feature = "crossval")]
#[serial(bitnet_env)]
fn test_reset_api_signature() -> Result<()> {
    // Setup: Create and populate context
    let model_path = std::path::Path::new("models/test.gguf");
    let ctx = create_test_context(model_path, 512, 0)?;

    let tokens = vec![1, 2, 3, 4];
    eval_tokens(ctx, &tokens, 0)?;

    // Verify: reset_context() API exists and works
    reset_context(ctx).context("Reset API should succeed")?;

    // Position should be 0 after reset
    assert_position_equals(ctx, 0)?;

    // Cleanup
    unsafe {
        // TODO: Call bitnet_cpp_free_context(ctx)
    }

    Ok(())
}

/// Test: FR3 - Position overflow error handling (edge case: n_past > n_ctx)
///
/// **Acceptance Criteria**: Error includes diagnostic information
///
/// **Spec Reference**: Section 2.5 - Position Validation and Overflow Handling
#[test]
#[ignore = "TODO: Enable when overflow validation implemented"]
#[cfg(feature = "crossval")]
#[serial(bitnet_env)]
fn test_position_overflow_error() -> Result<()> {
    // Setup: Create context with n_ctx = 8
    let model_path = std::path::Path::new("models/test.gguf");
    let ctx = create_test_context(model_path, 8, 0)?;

    // Fill to 7 tokens
    let tokens1 = vec![1; 7];
    eval_tokens(ctx, &tokens1, 0)?;

    // Try to overflow with 2 tokens (7 + 2 > 8)
    let tokens2 = vec![2, 3];
    let result = eval_tokens(ctx, &tokens2, 0);

    // Assert: Error message includes diagnostic info
    assert!(result.is_err(), "Should fail with overflow");

    let error_msg = result.unwrap_err().to_string();

    // Verify error message contains: n_past, n_tokens, n_ctx
    assert!(
        error_msg.contains("7") && error_msg.contains("2") && error_msg.contains("8"),
        "Error message should include n_past=7, n_tokens=2, n_ctx=8, got: {}",
        error_msg
    );

    // Cleanup
    unsafe {
        // TODO: Call bitnet_cpp_free_context(ctx)
    }

    Ok(())
}

// ============================================================================
// Category 2: Integration Tests - Multi-Turn Scenarios
// ============================================================================

/// Test: TR2 - Multi-turn conversation with KV cache reuse
///
/// **Acceptance Criteria**: Prefill prompt, generate tokens incrementally
///
/// **Spec Reference**: Section 6.2 - Integration Tests: Multi-Turn Evaluation
#[test]
#[ignore = "TODO: Enable when multi-turn support implemented"]
#[cfg(feature = "crossval")]
#[serial(bitnet_env)]
fn test_multi_turn_conversation() -> Result<()> {
    // Setup: Create persistent context
    let model_path = std::path::Path::new("models/test.gguf");
    let ctx = create_test_context(model_path, 512, 0)?;

    // Turn 1: Prefill prompt (4 tokens)
    let prompt = vec![1, 2, 3, 4];
    let _logits1 = eval_tokens(ctx, &prompt, 0).context("Failed to evaluate prompt")?;

    assert_position_equals(ctx, 4).context("Position should be 4 after prompt")?;

    // Turn 2: Generate 5 tokens (autoregressive)
    for i in 0..5 {
        let next_token = vec![10 + i];
        let _logits =
            eval_tokens(ctx, &next_token, 0).context(format!("Failed to evaluate token {}", i))?;

        // Position increments by 1 each step
        assert_position_equals(ctx, 4 + i + 1).context(format!(
            "Position should be {} after token {}",
            4 + i + 1,
            i
        ))?;
    }

    assert_position_equals(ctx, 9)
        .context("Position should be 9 after 4 prompt + 5 generated tokens")?;

    // Turn 3: Continue conversation (2 new tokens)
    let followup = vec![20, 21];
    eval_tokens(ctx, &followup, 0).context("Failed to evaluate followup")?;

    assert_position_equals(ctx, 11).context("Position should be 11 after followup")?;

    // Cleanup
    unsafe {
        // TODO: Call bitnet_cpp_free_context(ctx)
    }

    Ok(())
}

/// Test: TR2 - Autoregressive generation (token-by-token)
///
/// **Acceptance Criteria**: No redundant prompt recomputation
///
/// **Spec Reference**: Section 5.4 - Position Increment (Autoregressive)
#[test]
#[ignore = "TODO: Enable when autoregressive generation implemented"]
#[cfg(feature = "crossval")]
#[serial(bitnet_env)]
fn test_autoregressive_generation() -> Result<()> {
    // Setup: Create context
    let model_path = std::path::Path::new("models/test.gguf");
    let ctx = create_test_context(model_path, 512, 0)?;

    // Prefill prompt
    let prompt = vec![1, 2, 3, 4];
    eval_tokens(ctx, &prompt, 0)?;

    // Generate 10 tokens one-by-one (efficient KV cache reuse)
    for step in 0..10 {
        let next_token = vec![42 + step];
        eval_tokens(ctx, &next_token, 0).context(format!("Failed at generation step {}", step))?;

        // Verify: Position increments correctly
        let expected_pos = 4 + step + 1;
        assert_position_equals(ctx, expected_pos)
            .context(format!("Position mismatch at step {}", step))?;
    }

    // Final position: 4 prompt + 10 generated = 14
    assert_position_equals(ctx, 14)?;

    // Cleanup
    unsafe {
        // TODO: Call bitnet_cpp_free_context(ctx)
    }

    Ok(())
}

/// Test: TR2 - Prefill and decode phases
///
/// **Acceptance Criteria**: Batch prefill followed by incremental decode
///
/// **Spec Reference**: Section 2.4 - Position Tracking Across Evaluation Phases
#[test]
#[ignore = "TODO: Enable when prefill/decode phases implemented"]
#[cfg(feature = "crossval")]
#[serial(bitnet_env)]
fn test_prefill_and_decode() -> Result<()> {
    // Setup: Create context
    let model_path = std::path::Path::new("models/test.gguf");
    let ctx = create_test_context(model_path, 512, 0)?;

    // Phase 1: Prefill (large batch, one-time cost)
    let prefill_tokens = vec![1; 50]; // 50 tokens
    eval_tokens(ctx, &prefill_tokens, 0).context("Prefill phase failed")?;

    assert_position_equals(ctx, 50).context("Position should be 50 after prefill")?;

    // Phase 2: Decode (single token, repeated)
    for step in 0..20 {
        let decode_token = vec![100 + step];
        eval_tokens(ctx, &decode_token, 0).context(format!("Decode step {} failed", step))?;
    }

    // Final position: 50 prefill + 20 decode = 70
    assert_position_equals(ctx, 70).context("Position should be 70 after prefill + decode")?;

    // Cleanup
    unsafe {
        // TODO: Call bitnet_cpp_free_context(ctx)
    }

    Ok(())
}

/// Test: TR1 - End-to-end position tracking workflow
///
/// **Acceptance Criteria**: Full multi-turn workflow with reset
///
/// **Spec Reference**: Section 1.3 - Need for Position Tracking
#[test]
#[ignore = "TODO: Enable when end-to-end workflow implemented"]
#[cfg(feature = "crossval")]
#[serial(bitnet_env)]
fn test_position_tracking_e2e() -> Result<()> {
    // Setup: Create context
    let model_path = std::path::Path::new("models/test.gguf");
    let ctx = create_test_context(model_path, 512, 0)?;

    // Conversation 1: Prefill + generate
    let conv1_prompt = vec![1, 2, 3, 4, 5];
    eval_tokens(ctx, &conv1_prompt, 0)?;
    assert_position_equals(ctx, 5)?;

    // Generate 10 tokens
    for _ in 0..10 {
        eval_tokens(ctx, &[42], 0)?;
    }
    assert_position_equals(ctx, 15)?;

    // Reset for new conversation
    reset_context(ctx)?;
    assert_position_equals(ctx, 0)?;

    // Conversation 2: New prompt
    let conv2_prompt = vec![6, 7, 8];
    eval_tokens(ctx, &conv2_prompt, 0)?;
    assert_position_equals(ctx, 3)?;

    // Cleanup
    unsafe {
        // TODO: Call bitnet_cpp_free_context(ctx)
    }

    Ok(())
}

// ============================================================================
// Category 3: Parity Tests - Rust vs C++ Position Tracking
// ============================================================================

/// Test: PR1 - Single-turn parity (backward compatibility)
///
/// **Acceptance Criteria**: Full-batch evaluation produces identical logits
///
/// **Spec Reference**: Section 6.3 - Parity Tests: Rust vs C++
#[test]
#[ignore = "TODO: Enable when Rust implementation available for comparison"]
#[cfg(feature = "crossval")]
#[serial(bitnet_env)]
fn test_rust_cpp_position_parity_single_turn() -> Result<()> {
    // Setup: Create C++ context
    let model_path = std::path::Path::new("models/test.gguf");
    let cpp_ctx = create_test_context(model_path, 512, 0)?;

    // Evaluate entire sequence at once (like Socket 0)
    let tokens = vec![1, 2, 3, 4];
    let cpp_logits = eval_tokens(cpp_ctx, &tokens, 0).context("C++ evaluation failed")?;

    // TODO: Implement Rust inference for comparison
    // let rust_logits = eval_rust_inference(&tokens)?;

    // Verify: Position tracking works transparently
    assert_position_equals(cpp_ctx, 4)
        .context("C++ position should be 4 after single-turn evaluation")?;

    // TODO: Compare logits with Rust implementation
    // compare_rust_cpp_positions(&rust_logits, &cpp_logits, 0.9999)?;

    // Cleanup
    unsafe {
        // TODO: Call bitnet_cpp_free_context(cpp_ctx)
    }

    Ok(())
}

/// Test: PR2 - Multi-turn parity (incremental vs full evaluation)
///
/// **Acceptance Criteria**: Incremental matches full evaluation logits
///
/// **Spec Reference**: Section 6.3 - Parity Tests: Multi-Turn
#[test]
#[ignore = "TODO: Enable when multi-turn parity validated"]
#[cfg(feature = "crossval")]
#[serial(bitnet_env)]
fn test_rust_cpp_position_parity_multi_turn() -> Result<()> {
    // Setup: Create C++ context
    let model_path = std::path::Path::new("models/test.gguf");
    let cpp_ctx = create_test_context(model_path, 512, 0)?;

    // Turn 1: Prefill prompt
    let prompt = vec![1, 2, 3, 4];
    let _cpp_logits1 = eval_tokens(cpp_ctx, &prompt, 0).context("C++ prefill failed")?;

    // Turn 2: Generate token (uses KV cache)
    let next_token = vec![5];
    let cpp_logits2 = eval_tokens(cpp_ctx, &next_token, 0).context("C++ decode failed")?;

    // TODO: Compare with Rust full evaluation (no KV cache)
    // let rust_logits_full = eval_rust_inference(&vec![1, 2, 3, 4, 5])?;

    // Verify: Last position should match between incremental and full
    // compare_rust_cpp_positions(
    //     &vec![rust_logits_full[4].clone()],
    //     &cpp_logits2,
    //     0.9999,
    // )?;

    // Cleanup
    unsafe {
        // TODO: Call bitnet_cpp_free_context(cpp_ctx)
    }

    Ok(())
}

/// Test: PR3 - Logits parity with position tracking
///
/// **Acceptance Criteria**: Cosine similarity > 0.9999 for all positions
///
/// **Spec Reference**: Section 8.3 - Parity Requirements
#[test]
#[ignore = "TODO: Enable when cosine similarity validation implemented"]
#[cfg(feature = "crossval")]
#[serial(bitnet_env)]
fn test_rust_cpp_logits_parity_with_position() -> Result<()> {
    // Setup: Create C++ context
    let model_path = std::path::Path::new("models/test.gguf");
    let cpp_ctx = create_test_context(model_path, 512, 0)?;

    // Evaluate sequence with position tracking
    let tokens = vec![1, 2, 3, 4, 5];
    let cpp_logits = eval_tokens(cpp_ctx, &tokens, 0)?;

    // TODO: Evaluate with Rust implementation
    // let rust_logits = eval_rust_inference(&tokens)?;

    // Verify: High cosine similarity (> 0.9999) for all positions
    // for (pos, (rust_logit, cpp_logit)) in rust_logits.iter().zip(&cpp_logits).enumerate() {
    //     let similarity = cosine_similarity(rust_logit, cpp_logit);
    //     assert!(
    //         similarity > 0.9999,
    //         "Position {} parity failed: cosine similarity = {}",
    //         pos,
    //         similarity
    //     );
    // }

    // Cleanup
    unsafe {
        // TODO: Call bitnet_cpp_free_context(cpp_ctx)
    }

    Ok(())
}

// ============================================================================
// Category 4: Performance Tests - KV Cache Reuse Benefits
// ============================================================================

/// Benchmark: TR3 - Socket 0 vs Socket 1 overhead comparison
///
/// **Acceptance Criteria**: Socket 1 shows 10-100× speedup for multi-turn
///
/// **Spec Reference**: Section 7.1 - Socket 0 vs Socket 1 Overhead Analysis
#[test]
#[ignore = "TODO: Enable when Socket 0 and Socket 1 both implemented"]
#[cfg(feature = "crossval")]
#[serial(bitnet_env)]
fn bench_socket0_vs_socket1_overhead() -> Result<()> {
    use std::time::Instant;

    let model_path = std::path::Path::new("models/test.gguf");

    // Measure: Socket 0 (stateless, reloads model each call)
    let socket0_start = Instant::now();
    for _ in 0..10 {
        // TODO: Call crossval_bitnet_eval_with_tokens (Socket 0)
        // Simulates 10 separate calls with model reload overhead
    }
    let socket0_time = socket0_start.elapsed();

    // Measure: Socket 1 (persistent context, KV cache reuse)
    let socket1_start = Instant::now();
    let ctx = create_test_context(model_path, 512, 0)?;

    let prompt = vec![1; 50];
    eval_tokens(ctx, &prompt, 0)?; // Prefill

    for _ in 0..10 {
        let token = vec![42];
        eval_tokens(ctx, &token, 0)?; // Incremental decode
    }

    unsafe {
        // TODO: Call bitnet_cpp_free_context(ctx)
    }
    let socket1_time = socket1_start.elapsed();

    // Verify: Socket 1 is faster (expected 2-10× speedup)
    println!("Socket 0 time (10 calls): {:?}", socket0_time);
    println!("Socket 1 time (1 session): {:?}", socket1_time);

    let speedup = socket0_time.as_secs_f64() / socket1_time.as_secs_f64();
    println!("Speedup: {:.2}×", speedup);

    assert!(speedup > 2.0, "Expected at least 2× speedup, got {:.2}×", speedup);

    Ok(())
}

/// Benchmark: TR3 - Multi-turn KV cache benefit validation
///
/// **Acceptance Criteria**: 10-100× validation for autoregressive generation
///
/// **Spec Reference**: Section 7.2 - Context Persistence Benefits
#[test]
#[ignore = "TODO: Enable when KV cache performance benchmarking implemented"]
#[cfg(feature = "crossval")]
#[serial(bitnet_env)]
fn bench_multi_turn_kv_cache_benefit() -> Result<()> {
    use std::time::Instant;

    let model_path = std::path::Path::new("models/test.gguf");
    let ctx = create_test_context(model_path, 512, 0)?;

    // Measure: Full evaluation (no KV cache reuse)
    // Simulate by resetting context before each token
    let prompt = vec![1; 100];
    let full_eval_start = Instant::now();

    for i in 0..100 {
        reset_context(ctx)?;
        let tokens = &prompt[0..=i];
        eval_tokens(ctx, tokens, 0)?;
    }

    let full_eval_time = full_eval_start.elapsed();

    // Reset for incremental test
    reset_context(ctx)?;

    // Measure: Incremental evaluation (with KV cache reuse)
    let incremental_start = Instant::now();

    // Prefill prompt
    eval_tokens(ctx, &prompt, 0)?;

    // Generate 100 tokens incrementally
    for _ in 0..100 {
        let next = vec![42];
        eval_tokens(ctx, &next, 0)?;
    }

    let incremental_time = incremental_start.elapsed();

    // Verify: Incremental is much faster (expected 10-100× speedup)
    println!("Full eval (100 tokens): {:?}", full_eval_time);
    println!("Incremental (100 tokens): {:?}", incremental_time);

    let speedup = full_eval_time.as_secs_f64() / incremental_time.as_secs_f64();
    println!("KV cache speedup: {:.2}×", speedup);

    assert!(
        speedup > 10.0,
        "Expected at least 10× speedup from KV cache reuse, got {:.2}×",
        speedup
    );

    // Cleanup
    unsafe {
        // TODO: Call bitnet_cpp_free_context(ctx)
    }

    Ok(())
}

// ============================================================================
// Additional Edge Case Tests
// ============================================================================

/// Test: Edge case - Position exactly at boundary (n_past + n_tokens == n_ctx)
///
/// **Spec Reference**: Appendix B - Position Tracking Invariants
#[test]
#[ignore = "TODO: Enable when boundary validation implemented"]
#[cfg(feature = "crossval")]
#[serial(bitnet_env)]
fn test_position_boundary_exact() -> Result<()> {
    // Setup: Create context with n_ctx = 10
    let model_path = std::path::Path::new("models/test.gguf");
    let ctx = create_test_context(model_path, 10, 0)?;

    // Fill exactly to boundary (no overflow)
    let tokens = vec![1; 10];
    eval_tokens(ctx, &tokens, 0).context("Should succeed when n_past + n_tokens == n_ctx")?;

    assert_position_equals(ctx, 10)?;

    // Cleanup
    unsafe {
        // TODO: Call bitnet_cpp_free_context(ctx)
    }

    Ok(())
}

/// Test: Edge case - Reset idempotency (reset twice should work)
///
/// **Spec Reference**: Section 10.2 - Position Tracking Bugs
#[test]
#[ignore = "TODO: Enable when reset idempotency validated"]
#[cfg(feature = "crossval")]
#[serial(bitnet_env)]
fn test_reset_idempotency() -> Result<()> {
    // Setup: Create and populate context
    let model_path = std::path::Path::new("models/test.gguf");
    let ctx = create_test_context(model_path, 512, 0)?;

    let tokens = vec![1, 2, 3];
    eval_tokens(ctx, &tokens, 0)?;

    // Reset once
    reset_context(ctx)?;
    assert_position_equals(ctx, 0)?;

    // Reset again (should be idempotent)
    reset_context(ctx)?;
    assert_position_equals(ctx, 0)?;

    // Verify: Can still evaluate after double reset
    eval_tokens(ctx, &vec![4, 5], 0)?;
    assert_position_equals(ctx, 2)?;

    // Cleanup
    unsafe {
        // TODO: Call bitnet_cpp_free_context(ctx)
    }

    Ok(())
}

/// Test: Edge case - Failed evaluation does not increment position
///
/// **Spec Reference**: Appendix B - Invariant 2
#[test]
#[ignore = "TODO: Enable when error handling validated"]
#[cfg(feature = "crossval")]
#[serial(bitnet_env)]
fn test_position_unchanged_on_failure() -> Result<()> {
    // Setup: Create context
    let model_path = std::path::Path::new("models/test.gguf");
    let ctx = create_test_context(model_path, 10, 0)?;

    // Evaluate 5 tokens successfully
    let tokens1 = vec![1; 5];
    eval_tokens(ctx, &tokens1, 0)?;
    assert_position_equals(ctx, 5)?;

    // Try to overflow (should fail)
    let tokens2 = vec![2; 10]; // 5 + 10 > 10
    let result = eval_tokens(ctx, &tokens2, 0);
    assert!(result.is_err(), "Expected overflow error");

    // Verify: Position unchanged (still 5)
    assert_position_equals(ctx, 5)
        .context("Position should remain unchanged after failed evaluation")?;

    // Cleanup
    unsafe {
        // TODO: Call bitnet_cpp_free_context(ctx)
    }

    Ok(())
}

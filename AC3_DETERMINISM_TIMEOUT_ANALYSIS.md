# AC3 Deterministic Generation Tests - Timeout Analysis & Solutions

## Executive Summary

The AC3 determinism tests are timing out because they attempt full QK256 inference with `mock_forward_fn`, expecting generation to complete within test timeouts. The current infrastructure lacks:

1. **Fast sampling unit tests** - Can't validate determinism without full inference
2. **Proper stub models** - Mock models exist but aren't integrated into AC3 tests
3. **Minimal token generation tests** - Tests generate 15-50 tokens, each token = full forward pass
4. **Timeout configuration** - No explicit handling in test framework

## Quick Answer to Your Questions

### 1. Where are the determinism tests?

Located in three files:

- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/tests/issue_254_ac3_deterministic_generation.rs` (AC3.1-AC3.6)
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/tests/issue_254_ac6_determinism_integration.rs` (AC6.1-AC6.2)
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/tests/ac7_deterministic_inference.rs` (AC7.1)

### 2. Why are they slow?

**Current Test Structure (SLOW):**
```rust
#[tokio::test]
async fn test_ac3_deterministic_generation_identical_sequences() -> Result<()> {
    let config = GenConfig {
        max_new_tokens: 50,  // ⚠️ 50 tokens × 2 runs = 100 forward passes
        temperature: 1.0,
        top_k: Some(50),
        top_p: Some(0.9),
        seed: Some(42),
        ..Default::default()
    };
    
    let tokens1 = generator1.generate(&input_ids, mock_forward_fn).await?;
    let tokens2 = generator2.generate(&input_ids, mock_forward_fn).await?;
    // Expect completion in <30s
}
```

**Performance Breakdown:**
- Each `mock_forward_fn` call: Creates 50,257-element logits vector
- Softmax computation: Full vocabulary normalization
- 50 tokens × 2 runs = 100 forward passes minimum
- At ~10-20ms per token (with fast mock): 1000+ milliseconds
- With real QK256 inference (~0.1 tok/s): 1000+ SECONDS → TIMEOUT

### 3. What stub/mock infrastructure exists?

**Available Infrastructure:**

1. **EnvGuard** ✓
   - Location: `/home/steven/code/Rust/BitNet-rs/tests/support/env_guard.rs`
   - Thread-safe environment variable management
   - Automatic restoration on drop
   - Used with `#[serial(bitnet_env)]` for process isolation

2. **DeterministicGenerator** ✓
   - Location: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/generation/deterministic.rs`
   - ChaCha8Rng seeding mechanism
   - Categorical sampling with seed reproducibility
   - `BITNET_DETERMINISTIC` and `BITNET_SEED` environment support

3. **MockBitNetModel** ✓
   - Location: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/tests/fixtures/mock_quantized_model.rs`
   - Complete mock fixtures: `MockBitNetModel`, `MockBitNetAttention`, `MockQuantizedLinear`
   - `MockTokenizer`, `MockTensor`
   - Kernel registry with ADR-012 naming

4. **AutoregressiveGenerator** ✓
   - Location: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/generation/autoregressive.rs`
   - Full support for seed, temperature, top_k, top_p
   - `GenerationConfig` with all needed parameters
   - Performance tracking infrastructure

### 4. What's the minimal test to validate determinism?

**Minimal Unit Test** (~5ms, no inference):
```rust
#[test]
fn test_deterministic_generator_seed_42_reproducible() -> Result<()> {
    let mut gen1 = DeterministicGenerator::new(42)?;
    let mut gen2 = DeterministicGenerator::new(42)?;
    
    // Fixed probabilities (no softmax)
    let probs = vec![0.1, 0.5, 0.3, 0.1];
    
    // Sample 5 tokens
    for step in 0..5 {
        let (token1, _) = gen1.sample_deterministic(&probs, step)?;
        let (token2, _) = gen2.sample_deterministic(&probs, step)?;
        assert_eq!(token1, token2);
    }
    
    Ok(())
}
```

**Minimal Integration Test** (~100ms, 3 tokens):
```rust
#[tokio::test]
#[serial(bitnet_env)]
async fn test_ac3_deterministic_generation_minimal() -> Result<()> {
    let _g1 = EnvGuard::new("BITNET_DETERMINISTIC");
    _g1.set("1");
    let _g2 = EnvGuard::new("BITNET_SEED");
    _g2.set("42");
    let _g3 = EnvGuard::new("RAYON_NUM_THREADS");
    _g3.set("1");

    let config = GenConfig {
        max_new_tokens: 3,  // Only 3 tokens
        seed: Some(42),
        ..Default::default()
    };

    let tokens1 = generator1.generate(&input_ids, fast_mock_forward_fn).await?;
    let tokens2 = generator2.generate(&input_ids, fast_mock_forward_fn).await?;
    
    assert_eq!(tokens1, tokens2);
    Ok(())
}
```

## Recommended Implementation Path

### Phase 1: Create Fast Unit Tests (NEW FILE)

**File:** `crates/bitnet-inference/tests/deterministic_sampling_unit.rs`

Tests to add:
- `test_deterministic_generator_seed_42_reproducible()` - Same seed → identical tokens
- `test_different_seeds_produce_different_samples()` - Different seeds → different tokens
- `test_different_steps_different_tokens()` - Step count affects output
- `test_reset_restores_initial_state()` - Reset mechanism works
- `test_step_count_increments()` - Internal state tracking
- `test_set_seed_changes_behavior()` - Seed setter works

**Expected runtime:** <5ms total for all 6 tests
**Status:** No timeout risk

### Phase 2: Refactor Fast Integration Tests (MODIFY EXISTING)

**Files to modify:**
- `crates/bitnet-inference/tests/issue_254_ac3_deterministic_generation.rs`
- `crates/bitnet-inference/tests/issue_254_ac6_determinism_integration.rs`

**Changes:**
- Reduce `max_new_tokens`: 50 → 3
- Replace `mock_forward_fn` with fast version (10 tokens instead of 50257)
- Keep environment isolation and seed mechanism tests
- Rename old tests with `_slow` suffix and add `#[ignore]`

**Expected runtime:** <100ms per test
**Status:** CI-safe, no timeout risk

### Phase 3: Create Slow Regression Tests (NEW FILE)

**File:** `crates/bitnet-inference/tests/ac3_slow_determinism_integration.rs`

Tests to add:
- `test_ac3_deterministic_full_50_tokens()` - Full pipeline with 50 tokens
- `test_ac6_determinism_multiple_runs_slow()` - 5-run consistency check

**Expected runtime:** 1000+ milliseconds
**Marking:** `#[ignore]` - Skip in CI, run manually for regression testing

### Phase 4: Verify AC7 Tests (VERIFY EXISTING)

**File:** `crates/bitnet-inference/tests/ac7_deterministic_inference.rs`

**Status:** Already uses mock models correctly - no changes needed

## Test Organization Summary

| Test Type | File | Runtime | Status | CI |
|-----------|------|---------|--------|-----|
| Unit sampling (6 tests) | deterministic_sampling_unit.rs (NEW) | <5ms | Create | ✓ Run |
| Fast integration AC3 (5 tests) | issue_254_ac3_deterministic_generation.rs | <100ms | Modify | ✓ Run |
| Fast integration AC6 (2 tests) | issue_254_ac6_determinism_integration.rs | <100ms | Modify | ✓ Run |
| Slow regression AC3 (2 tests) | ac3_slow_determinism_integration.rs (NEW) | 1000+ms | Create | Skip |
| Slow regression AC7 (1 test) | ac7_deterministic_inference.rs | <5ms | Verify | ✓ Run |

**Total CI Runtime:** ~300-400ms (all fast tests)
**Local Full Regression:** ~5-10 seconds (including slow tests)

## Technical Details: Why This Works

### Determinism Contract

```
seed + input + config → output tokens (100% reproducible)
```

This depends on:
1. **Seeding mechanism** (ChaCha8Rng initialization) - Validated by unit tests
2. **Sampling logic** (categorical sampling) - Validated by unit tests
3. **Tensor operations** (softmax, top-k filtering) - Validated by fast integration
4. **Full inference pipeline** (layer outputs, shape handling) - Validated by slow tests

**For MVP:** (1) + (2) + (3) are most critical and all fast to test.

### Why Not Just Use Real Models?

- QK256 MVP: ~0.1 tok/s (by design - scalar kernels)
- 50 tokens = 500 seconds per run
- Determinism test: 2 runs = 1000 seconds = TIMEOUT
- **Solution:** Don't test performance, test mechanism
- Use fast mock that validates same determinism contract

## Files to Create/Modify

### CREATE

1. **`crates/bitnet-inference/tests/deterministic_sampling_unit.rs`**
   - 6 unit tests for DeterministicGenerator
   - No async, no inference
   - Full code provided in implementation guide

2. **`crates/bitnet-inference/tests/ac3_slow_determinism_integration.rs`**
   - 3 slow integration tests
   - Marked #[ignore]
   - Full code provided in implementation guide

### MODIFY

1. **`crates/bitnet-inference/tests/issue_254_ac3_deterministic_generation.rs`**
   - Change max_new_tokens: 50 → 3
   - Add fast_mock_forward_fn (10-token mini-distribution)
   - Rename slow tests with #[ignore]
   - Keep fast tests active

2. **`crates/bitnet-inference/tests/issue_254_ac6_determinism_integration.rs`**
   - Same changes as AC3

### VERIFY

1. **`crates/bitnet-inference/tests/ac7_deterministic_inference.rs`**
   - Already uses mock models
   - No changes needed

## Expected Outcomes

**Before Implementation:**
- Tests timeout at 30+ seconds
- Determinism validation unclear
- Can't distinguish mechanism bugs from performance issues

**After Implementation:**
- Fast tests: <100ms, determinism validated
- Slow tests: Available for regression, marked clearly
- CI: Passes reliably, no timeouts
- Local: Can run full suite (1-2 minutes) with `--ignored` flag

## Quick Reference: Running Tests

```bash
# Fast tests only (CI)
cargo test -p bitnet-inference deterministic_sampling_unit
cargo test -p bitnet-inference test_ac3_deterministic_fast

# All tests (local)
cargo test -p bitnet-inference -- --ignored

# Slow tests only (regression)
cargo test -p bitnet-inference test_ac3_deterministic_slow -- --ignored --nocapture

# With nextest (recommended)
cargo nextest run -p bitnet-inference --profile ci
```

## Validation Checklist

- [ ] Unit tests created and passing
- [ ] Fast integration tests refactored and passing
- [ ] Slow tests marked #[ignore] and skipped by default
- [ ] All tests use EnvGuard for environment isolation
- [ ] All tests use #[serial(bitnet_env)] for process safety
- [ ] Documentation updated in CLAUDE.md
- [ ] CI passes with fast tests only

## Additional Resources

See also:
- `docs/development/test-suite.md` - Test framework overview
- `CLAUDE.md` - Project test status section
- `.config/nextest.toml` - Test timeout configuration


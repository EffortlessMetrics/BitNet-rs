# CPU Inference Test Plan

**Issue:** #462 - CPU Forward Pass with Real Inference
**Status:** Specification
**Date:** 2025-10-14

## Context

This test plan defines the comprehensive validation strategy for CPU forward pass implementation with real inference. The plan covers 13 test cases spanning unit, integration, and end-to-end validation as identified in issue-finalizer analysis.

**Coverage Goals:**
- **AC1:** CPU forward pass real inference (4 tests)
- **AC2:** CLI priming and decode loops (3 tests)
- **AC3:** Receipt CPU validation (3 tests)
- **AC4:** TL LUT helper (3 tests)
- **AC5:** Baseline and README verification (manual validation)

**Test Infrastructure:**
- TDD approach with `// AC:ID` tags for traceability
- Feature-gated tests: `--no-default-features --features cpu`
- Deterministic fixtures with `BITNET_DETERMINISTIC=1`
- Cross-validation against C++ reference when applicable

## Test Cases

### AC1: CPU Forward Pass Real Inference

#### Test 1.1: BOS Token Returns Non-Zero Finite Logits

**Test ID:** `test_ac1_cpu_forward_bos_nonzero_logits`
**Priority:** P0 (Critical)
**Type:** Unit Test
**AC Mapping:** AC1

**Description:**
Validate that forward pass on BOS (Beginning of Sequence) token returns non-zero finite logits tensor.

**Preconditions:**
- CPU inference engine initialized with test model
- Model loaded with quantized weights (I2S/TL1/TL2)
- KV cache allocated for test sequence length

**Test Steps:**
1. Create BOS token tensor: `[1u32]` with shape `[1]`
2. Call `engine.forward_parallel(&bos_token, step=0)`
3. Validate output shape: `[1, vocab_size]`
4. Check all logits are finite (no NaN/Inf)
5. Check at least one logit is non-zero

**Expected Results:**
- Output shape matches `[1, vocab_size]` (e.g., `[1, 32000]`)
- All logits are finite: `logits.iter().all(|x| x.is_finite())`
- At least one non-zero logit: `logits.iter().any(|x| x != 0.0)`
- KV cache updated at position 0 for all layers

**Validation Command:**
```bash
cargo test -p bitnet-inference test_ac1_cpu_forward_bos_nonzero_logits \
  --no-default-features --features cpu -- --nocapture
```

**Code Location:**
- File: `crates/bitnet-inference/tests/issue_462_cpu_forward_tests.rs`
- Function: `test_ac1_cpu_forward_bos_nonzero_logits()`

**AC Tag:**
```rust
// AC:AC1 - CPU forward pass returns non-zero finite logits for BOS token
#[test]
fn test_ac1_cpu_forward_bos_nonzero_logits() { /* ... */ }
```

---

#### Test 1.2: 16-Token Greedy Decode Without Panic

**Test ID:** `test_ac1_greedy_decode_16_tokens`
**Priority:** P0 (Critical)
**Type:** Integration Test
**AC Mapping:** AC1

**Description:**
Validate that autoregressive generation of 16 tokens completes without panic using greedy decoding.

**Preconditions:**
- CPU inference engine with primed KV cache (BOS token)
- Greedy sampler (temperature = 0.0)
- Deterministic mode: `BITNET_DETERMINISTIC=1`, `BITNET_SEED=42`

**Test Steps:**
1. Initialize engine with BOS token (step 0)
2. For steps 1..16:
   - Call `forward_parallel(prev_token, step)`
   - Sample next token: `argmax(logits)`
   - Validate token ID within vocab range
   - Check KV cache updated at position `step`
3. Verify all 16 tokens generated without panic

**Expected Results:**
- 16 tokens generated successfully
- All token IDs: `0 <= token_id < vocab_size`
- No panics or runtime errors
- KV cache length = 17 (BOS + 16 generated)
- Deterministic: Same seed → same output sequence

**Validation Command:**
```bash
BITNET_DETERMINISTIC=1 BITNET_SEED=42 RAYON_NUM_THREADS=1 \
  cargo test -p bitnet-inference test_ac1_greedy_decode_16_tokens \
  --no-default-features --features cpu -- --nocapture
```

**Code Location:**
- File: `crates/bitnet-inference/tests/issue_462_cpu_forward_tests.rs`
- Function: `test_ac1_greedy_decode_16_tokens()`

**AC Tag:**
```rust
// AC:AC1 - 16-token greedy decode without panic
#[test]
fn test_ac1_greedy_decode_16_tokens() { /* ... */ }
```

---

#### Test 1.3: Quantized Linear Path Enforcement

**Test ID:** `test_ac1_quantized_linear_strict_mode`
**Priority:** P1 (Quality)
**Type:** Unit Test
**AC Mapping:** AC1

**Description:**
Validate that forward pass uses QuantizedLinear I2S/TL1/TL2 paths with strict mode blocking FP32 staging.

**Preconditions:**
- Strict mode enabled: `BITNET_STRICT_MODE=1`
- Model with quantized weights (no FP32 fallback available)

**Test Steps:**
1. Enable strict mode in engine config
2. Call `forward_parallel(bos_token, 0)`
3. Verify receipt contains quantized kernel IDs:
   - `i2s_*`, `tl1_*`, `tl2_*` present
   - No `fp32_*`, `fallback_*`, `dequant*` kernels
4. Validate output logits (non-zero, finite)

**Expected Results:**
- Forward pass completes successfully
- Receipt `kernels` array contains ≥1 quantized kernel
- No FP32/fallback kernels in receipt
- Output logits valid

**Validation Command:**
```bash
BITNET_STRICT_MODE=1 \
  cargo test -p bitnet-inference test_ac1_quantized_linear_strict_mode \
  --no-default-features --features cpu -- --nocapture
```

**Code Location:**
- File: `crates/bitnet-inference/tests/issue_462_cpu_forward_tests.rs`
- Function: `test_ac1_quantized_linear_strict_mode()`

**AC Tag:**
```rust
// AC:AC1 - Strict mode enforces quantized paths (no FP32 staging)
#[test]
fn test_ac1_quantized_linear_strict_mode() { /* ... */ }
```

---

#### Test 1.4: KV Cache Population and Retrieval

**Test ID:** `test_ac1_kv_cache_update_retrieval`
**Priority:** P1 (Quality)
**Type:** Unit Test
**AC Mapping:** AC1

**Description:**
Validate that KV cache is correctly populated during forward pass and retrieval returns expected shapes.

**Preconditions:**
- CPU inference engine initialized
- Known input sequence: `[BOS, token1, token2]`

**Test Steps:**
1. Forward BOS token (step 0)
   - Check cache length = 1
   - Validate K,V shape for layer 0: `[1, num_heads, head_dim]`
2. Forward token1 (step 1)
   - Check cache length = 2
   - Validate K,V shape for layer 0: `[2, num_heads, head_dim]`
3. Forward token2 (step 2)
   - Check cache length = 3
   - Validate K,V shape for layer 0: `[3, num_heads, head_dim]`
4. Retrieve cache for layer 0:
   - Check returned K,V shapes match current length
   - Validate non-zero values (not all zeros)

**Expected Results:**
- Cache length increments correctly: 1, 2, 3
- K,V shapes match expected dimensions per step
- Retrieved cache contains non-zero values
- Cache accessible for all layers (0..num_layers)

**Validation Command:**
```bash
cargo test -p bitnet-inference test_ac1_kv_cache_update_retrieval \
  --no-default-features --features cpu -- --nocapture
```

**Code Location:**
- File: `crates/bitnet-inference/tests/issue_462_cpu_forward_tests.rs`
- Function: `test_ac1_kv_cache_update_retrieval()`

**AC Tag:**
```rust
// AC:AC1 - KV cache update and retrieval correctness
#[test]
fn test_ac1_kv_cache_update_retrieval() { /* ... */ }
```

---

### AC2: CLI Priming and Decode Loop

#### Test 2.1: CLI Question Answering E2E

**Test ID:** `test_ac2_cli_inference_question_answering`
**Priority:** P0 (Critical)
**Type:** End-to-End Test
**AC Mapping:** AC2

**Description:**
Validate complete CLI workflow: "Q: What is 2+2? A:" → "4" within 16 tokens.

**Preconditions:**
- GGUF model and tokenizer available
- CLI binary built with CPU feature
- Deterministic mode for reproducible output

**Test Steps:**
1. Run CLI command:
   ```bash
   cargo run -p bitnet-cli --no-default-features --features cpu -- \
     run --model <test_model.gguf> \
     --prompt "Q: What is 2+2? A:" \
     --max-new-tokens 16 \
     --temperature 0.0
   ```
2. Capture stdout output
3. Validate output contains "4" within first 16 tokens
4. Check no errors in stderr

**Expected Results:**
- Command exits with code 0
- Output contains expected answer substring
- Generation completes within 16 tokens
- Receipt generated with valid kernels

**Validation Command:**
```bash
BITNET_DETERMINISTIC=1 BITNET_SEED=42 \
  cargo test -p bitnet-cli test_ac2_cli_inference_question_answering \
  --no-default-features --features cpu -- --nocapture
```

**Code Location:**
- File: `crates/bitnet-cli/tests/issue_462_cli_inference_tests.rs`
- Function: `test_ac2_cli_inference_question_answering()`

**AC Tag:**
```rust
// AC:AC2 - CLI question answering E2E workflow
#[test]
fn test_ac2_cli_inference_question_answering() { /* ... */ }
```

---

#### Test 2.2: Priming Loop KV Cache Population

**Test ID:** `test_ac2_cli_priming_loop`
**Priority:** P1 (Quality)
**Type:** Integration Test
**AC Mapping:** AC2

**Description:**
Validate that priming loop correctly populates KV cache for all prompt tokens before decode starts.

**Preconditions:**
- Test prompt: "Hello world test"
- Tokenizer produces known token count (e.g., 5 tokens)

**Test Steps:**
1. Tokenize prompt → `[token0, token1, token2, token3, token4]`
2. Run priming loop:
   - For each token at position `i`:
     - Call `forward(token_i, step=i)`
     - Discard logits
     - Verify KV cache length = i+1
3. After priming:
   - Check KV cache length = 5
   - Verify cache contains non-zero values
   - Ready for decode loop at step 5

**Expected Results:**
- All prompt tokens processed
- KV cache populated for positions 0..4
- No decode started during priming
- Cache ready for autoregressive generation

**Validation Command:**
```bash
cargo test -p bitnet-cli test_ac2_cli_priming_loop \
  --no-default-features --features cpu -- --nocapture
```

**Code Location:**
- File: `crates/bitnet-cli/tests/issue_462_cli_inference_tests.rs`
- Function: `test_ac2_cli_priming_loop()`

**AC Tag:**
```rust
// AC:AC2 - Priming loop populates KV cache correctly
#[test]
fn test_ac2_cli_priming_loop() { /* ... */ }
```

---

#### Test 2.3: Decode Loop Token Sampling

**Test ID:** `test_ac2_cli_decode_loop_sampling`
**Priority:** P1 (Quality)
**Type:** Integration Test
**AC Mapping:** AC2

**Description:**
Validate decode loop with different sampling strategies (greedy, top-k, top-p).

**Preconditions:**
- Engine with primed KV cache
- Test fixtures for sampling validation

**Test Steps:**
1. **Greedy sampling (temperature = 0.0):**
   - Generate 10 tokens
   - Verify deterministic output (same seed → same tokens)
2. **Top-k sampling (k = 50, temperature = 0.7):**
   - Generate 10 tokens
   - Verify tokens within top-k candidates
3. **Top-p sampling (p = 0.95, temperature = 0.9):**
   - Generate 10 tokens
   - Verify nucleus sampling constraint

**Expected Results:**
- Greedy: Deterministic token sequence
- Top-k: All tokens from top-k candidates
- Top-p: Cumulative probability ≥ p
- All strategies: Valid token IDs, no panics

**Validation Command:**
```bash
cargo test -p bitnet-cli test_ac2_cli_decode_loop_sampling \
  --no-default-features --features cpu -- --nocapture
```

**Code Location:**
- File: `crates/bitnet-cli/tests/issue_462_cli_inference_tests.rs`
- Function: `test_ac2_cli_decode_loop_sampling()`

**AC Tag:**
```rust
// AC:AC2 - Decode loop sampling strategies
#[test]
fn test_ac2_cli_decode_loop_sampling() { /* ... */ }
```

---

### AC3: Receipt CPU Validation

#### Test 3.1: CPU Receipt Honesty (Positive)

**Test ID:** `test_ac3_receipt_cpu_kernel_honesty_positive`
**Priority:** P0 (Critical)
**Type:** Integration Test
**AC Mapping:** AC3

**Description:**
Validate that receipt with CPU quantized kernels passes verification.

**Preconditions:**
- Receipt with `backend="cpu"` and quantized kernel IDs

**Test Steps:**
1. Create receipt JSON:
   ```json
   {
     "schema_version": "1.0.0",
     "compute_path": "real",
     "backend": "cpu",
     "kernels": ["i2s_gemv", "tl1_matmul", "tl2_matmul"]
   }
   ```
2. Write to temp file
3. Run: `cargo run -p xtask -- verify-receipt <path>`
4. Check exit code 0

**Expected Results:**
- Verification passes
- Output: "✅ Receipt verification passed"
- No errors in stderr

**Validation Command:**
```bash
cargo test -p xtask test_ac3_receipt_cpu_kernel_honesty_positive \
  --no-default-features --features cpu
```

**Code Location:**
- File: `xtask/tests/issue_462_receipt_validation_tests.rs`
- Function: `test_ac3_receipt_cpu_kernel_honesty_positive()`

**AC Tag:**
```rust
// AC:AC3 - CPU receipt with quantized kernels passes validation
#[test]
fn test_ac3_receipt_cpu_kernel_honesty_positive() { /* ... */ }
```

---

#### Test 3.2: CPU Receipt Honesty (Negative - Mock Kernels)

**Test ID:** `test_ac3_receipt_cpu_kernel_honesty_negative`
**Priority:** P0 (Critical)
**Type:** Integration Test
**AC Mapping:** AC3

**Description:**
Validate that receipt with mock/non-quantized kernels fails verification.

**Preconditions:**
- Receipt with `backend="cpu"` but no quantized kernels

**Test Steps:**
1. Create receipt JSON:
   ```json
   {
     "schema_version": "1.0.0",
     "compute_path": "real",
     "backend": "cpu",
     "kernels": ["rope_apply", "softmax_cpu", "mock_kernel"]
   }
   ```
2. Write to temp file
3. Run: `cargo run -p xtask -- verify-receipt <path>`
4. Check exit code 1

**Expected Results:**
- Verification fails
- Error message: "no quantized kernels found"
- Suggests expected kernel prefixes: `i2s_`, `tl1_`, `tl2_`

**Validation Command:**
```bash
cargo test -p xtask test_ac3_receipt_cpu_kernel_honesty_negative \
  --no-default-features --features cpu
```

**Code Location:**
- File: `xtask/tests/issue_462_receipt_validation_tests.rs`
- Function: `test_ac3_receipt_cpu_kernel_honesty_negative()`

**AC Tag:**
```rust
// AC:AC3 - CPU receipt without quantized kernels fails validation
#[test]
fn test_ac3_receipt_cpu_kernel_honesty_negative() { /* ... */ }
```

---

#### Test 3.3: GPU Backend with CPU Kernels Fails

**Test ID:** `test_ac3_receipt_gpu_cpu_kernel_mismatch`
**Priority:** P1 (Quality)
**Type:** Integration Test
**AC Mapping:** AC3

**Description:**
Validate that GPU backend receipt with CPU kernels fails (silent CPU fallback detection).

**Preconditions:**
- Receipt with `backend="cuda"` but CPU kernel IDs

**Test Steps:**
1. Create receipt JSON:
   ```json
   {
     "schema_version": "1.0.0",
     "compute_path": "real",
     "backend": "cuda",
     "kernels": ["i2s_gemv", "tl1_matmul"]
   }
   ```
2. Write to temp file
3. Run: `cargo run -p xtask -- verify-receipt <path>`
4. Check exit code 1

**Expected Results:**
- Verification fails
- Error message: "no GPU kernels found"
- Suggests GPU kernel prefixes: `gemm_`, `cuda_`, `i2s_gpu_`
- Detects silent CPU fallback

**Validation Command:**
```bash
cargo test -p xtask test_ac3_receipt_gpu_cpu_kernel_mismatch \
  --no-default-features --features cpu
```

**Code Location:**
- File: `xtask/tests/issue_462_receipt_validation_tests.rs`
- Function: `test_ac3_receipt_gpu_cpu_kernel_mismatch()`

**AC Tag:**
```rust
// AC:AC3 - GPU backend with CPU kernels fails (silent fallback)
#[test]
fn test_ac3_receipt_gpu_cpu_kernel_mismatch() { /* ... */ }
```

---

### AC4: TL LUT Helper

#### Test 4.1: Valid LUT Index Calculation

**Test ID:** `test_ac4_tl_lut_index_bounds_valid`
**Priority:** P0 (Critical)
**Type:** Unit Test
**AC Mapping:** AC4

**Description:**
Validate correct LUT index calculation for valid inputs (TL1, TL2 configurations).

**Preconditions:**
- TL LUT helper module available
- Known configurations: (block_bytes, elems_per_block)

**Test Steps:**
1. TL1 config: `block_bytes=16, elems_per_block=128`
   - `lut_index(0, 0, 16, 128)` → 0
   - `lut_index(0, 8, 16, 128)` → 1 (8/8 = 1 byte offset)
   - `lut_index(1, 0, 16, 128)` → 16 (block 1 offset)
   - `lut_index(5, 64, 16, 128)` → 88 (5*16 + 64/8 = 88)
2. TL2 config: `block_bytes=32, elems_per_block=256`
   - `lut_index(0, 0, 32, 256)` → 0
   - `lut_index(2, 16, 32, 256)` → 66 (2*32 + 16/8 = 66)

**Expected Results:**
- All indices calculated correctly
- No panics or errors
- Results match manual calculation

**Validation Command:**
```bash
cargo test -p bitnet-kernels test_ac4_tl_lut_index_bounds_valid \
  --no-default-features --features cpu
```

**Code Location:**
- File: `crates/bitnet-kernels/src/tl_lut/tests.rs`
- Function: `test_ac4_tl_lut_index_bounds_valid()`

**AC Tag:**
```rust
// AC:AC4 - Valid LUT index calculation (TL1, TL2)
#[test]
fn test_ac4_tl_lut_index_bounds_valid() { /* ... */ }
```

---

#### Test 4.2: Invalid LUT Index (Out of Bounds)

**Test ID:** `test_ac4_tl_lut_index_bounds_invalid`
**Priority:** P0 (Critical)
**Type:** Unit Test
**AC Mapping:** AC4

**Description:**
Validate error handling for out-of-bounds element indices.

**Preconditions:**
- TL LUT helper with bounds checking

**Test Steps:**
1. Test `elem_in_block >= elems_per_block`:
   - `lut_index(0, 128, 16, 128)` → Error
   - `lut_index(1, 200, 16, 128)` → Error
2. Verify error type: `LutIndexError::OutOfBounds(elem, max)`
3. Check error message descriptive

**Expected Results:**
- Returns `Err(LutIndexError::OutOfBounds(...))`
- Error message: "Element index X exceeds elements per block Y"
- No panic (graceful error handling)

**Validation Command:**
```bash
cargo test -p bitnet-kernels test_ac4_tl_lut_index_bounds_invalid \
  --no-default-features --features cpu
```

**Code Location:**
- File: `crates/bitnet-kernels/src/tl_lut/tests.rs`
- Function: `test_ac4_tl_lut_index_bounds_invalid()`

**AC Tag:**
```rust
// AC:AC4 - Out-of-bounds element index error handling
#[test]
fn test_ac4_tl_lut_index_bounds_invalid() { /* ... */ }
```

---

#### Test 4.3: TL1/TL2 Matmul Integration

**Test ID:** `test_ac4_tl_matmul_with_safe_lut`
**Priority:** P1 (Quality)
**Type:** Integration Test
**AC Mapping:** AC4

**Description:**
Validate TL1/TL2 quantized matmul integration with safe LUT helper (re-enable TL tests).

**Preconditions:**
- QuantizedLinear layer with TL1/TL2 quantization
- Input tensor: `[1, in_features]`

**Test Steps:**
1. **TL1 matmul:**
   - Create QuantizedLinear with TL1
   - Forward pass with test input
   - Verify output shape: `[1, out_features]`
   - Check no panics from LUT indexing
2. **TL2 matmul:**
   - Same steps with TL2 configuration
3. Remove `#[ignore]` from previously disabled TL tests

**Expected Results:**
- TL1/TL2 matmul completes successfully
- Output shape correct
- No LUT indexing errors
- All TL tests pass without `#[ignore]`

**Validation Command:**
```bash
cargo test -p bitnet-inference test_ac4_tl_matmul_with_safe_lut \
  --no-default-features --features cpu -- --include-ignored
```

**Code Location:**
- File: `crates/bitnet-inference/tests/issue_462_tl_matmul_tests.rs`
- Function: `test_ac4_tl_matmul_with_safe_lut()`

**AC Tag:**
```rust
// AC:AC4 - TL1/TL2 matmul integration with safe LUT helper
#[test] // Previously #[ignore], now enabled
fn test_ac4_tl_matmul_with_safe_lut() { /* ... */ }
```

---

### AC5: Baseline and README

#### Test 5.1: Baseline Receipt Verification (Manual)

**Test ID:** `manual_ac5_baseline_verification`
**Priority:** P1 (Quality)
**Type:** Manual Validation
**AC Mapping:** AC5

**Description:**
Manually verify that baseline CPU receipt validates and README quickstart is functional.

**Preconditions:**
- Successful CPU inference run generating `ci/inference.json`
- README.md updated with quickstart section

**Test Steps:**
1. Run CPU benchmark:
   ```bash
   cargo run -p xtask -- benchmark --model <model.gguf> --tokens 128
   ```
2. Verify receipt generated: `ci/inference.json`
3. Validate receipt:
   ```bash
   cargo run -p xtask -- verify-receipt
   ```
4. Pin baseline:
   ```bash
   cp ci/inference.json docs/baselines/$(date +%Y%m%d)-cpu.json
   ```
5. Test README quickstart:
   - Copy-paste commands from README
   - Verify output matches documented example

**Expected Results:**
- Receipt validates successfully
- Baseline pinned to `docs/baselines/`
- README quickstart produces expected output
- Deterministic flags documented

**Validation Checklist:**
- [ ] `ci/inference.json` validates with `verify-receipt`
- [ ] Baseline copied to `docs/baselines/<timestamp>-cpu.json`
- [ ] README quickstart tested (copy-paste works)
- [ ] Deterministic flags documented: `BITNET_DETERMINISTIC=1 BITNET_SEED=42`

---

## Test Data Requirements

### Models

**Test Model (Small):**
- Size: ≤500MB for fast CI execution
- Format: GGUF with I2S/TL1/TL2 quantization
- Vocab: Standard tokenizer (e.g., LLaMA, GPT)
- Location: `tests/fixtures/models/test-model.gguf`

**Provision Command:**
```bash
cargo run -p xtask -- download-model --id test-model-small
```

### Tokenizers

**Test Tokenizer:**
- Format: HuggingFace `tokenizer.json`
- Compatibility: Matches test model vocab
- Location: `tests/fixtures/tokenizers/test-tokenizer.json`

**Auto-Discovery:**
- Models in `models/` directory auto-discovered
- Tokenizer co-located: `models/tokenizer.json`

### Fixtures

**Test Prompts:**
```rust
const TEST_PROMPTS: &[&str] = &[
    "Q: What is 2+2? A:",           // Simple arithmetic
    "Hello world",                  // Basic generation
    "The quick brown fox",          // Common phrase
    "<s>",                          // BOS token only
];
```

**Expected Outputs (Deterministic):**
```rust
const EXPECTED_OUTPUTS: &[&str] = &[
    "4",                            // Arithmetic answer
    " test",                        // Continuation
    " jumps over",                  // Common phrase completion
];
```

### Environment Setup

**Deterministic Inference:**
```bash
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export RAYON_NUM_THREADS=1
```

**Strict Mode:**
```bash
export BITNET_STRICT_MODE=1
```

**CI Environment:**
```bash
export CI=1
export BITNET_CI_ENHANCED_STRICT=1
```

## Test Execution

### Local Development

**Run all AC1 tests:**
```bash
cargo test -p bitnet-inference test_ac1 \
  --no-default-features --features cpu -- --nocapture
```

**Run all AC2 tests:**
```bash
cargo test -p bitnet-cli test_ac2 \
  --no-default-features --features cpu -- --nocapture
```

**Run all AC3 tests:**
```bash
cargo test -p xtask test_ac3 \
  --no-default-features --features cpu
```

**Run all AC4 tests:**
```bash
cargo test -p bitnet-kernels test_ac4 \
  --no-default-features --features cpu
```

**Run full test suite:**
```bash
cargo test --workspace \
  --no-default-features --features cpu -- test_ac
```

### CI Pipeline

**GitHub Actions Workflow:**
```yaml
name: Issue 462 CPU Forward Pass Tests

on:
  pull_request:
    paths:
      - 'crates/bitnet-inference/**'
      - 'crates/bitnet-cli/**'
      - 'crates/bitnet-kernels/**'
      - 'xtask/**'

jobs:
  cpu-forward-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Run AC1 tests
        run: |
          cargo test -p bitnet-inference test_ac1 \
            --no-default-features --features cpu

      - name: Run AC2 tests
        run: |
          cargo test -p bitnet-cli test_ac2 \
            --no-default-features --features cpu

      - name: Run AC3 tests
        run: |
          cargo test -p xtask test_ac3 \
            --no-default-features --features cpu

      - name: Run AC4 tests
        run: |
          cargo test -p bitnet-kernels test_ac4 \
            --no-default-features --features cpu

      - name: Verify baseline receipt
        run: |
          cargo run -p xtask -- verify-receipt ci/inference.json
```

### Cross-Validation

**Against C++ Reference:**
```bash
# Run cross-validation suite
BITNET_DETERMINISTIC=1 BITNET_SEED=42 \
  cargo run -p xtask -- crossval \
  --model models/test-model.gguf \
  --prompt "Test input"

# Expected: ≥99% cosine similarity
```

## Performance Baselines

### CPU Throughput Targets

**Tiny Model (500M):**
- Throughput: ≥10 tok/s
- First token latency: ≤1s
- Memory: KV cache ≤512 MB

**2B Model:**
- Throughput: ≥5 tok/s
- First token latency: ≤2s
- Memory: KV cache ≤1 GB

**Measurement Command:**
```bash
cargo run -p xtask -- benchmark \
  --model models/test-model.gguf \
  --tokens 128 \
  --warmup 3
```

### Deterministic Baseline

**Seed 42 Output (Reference):**
```
Prompt: "Q: What is 2+2? A:"
Generated (greedy): "4"
Tokens: [50, 45, 32, ...]  # Reproducible token IDs
```

**Validation:**
```bash
# Run twice with same seed, verify identical output
BITNET_DETERMINISTIC=1 BITNET_SEED=42 \
  cargo run -p bitnet-cli --features cpu -- \
  run --model <model> --prompt "Test" --temperature 0.0 > output1.txt

BITNET_DETERMINISTIC=1 BITNET_SEED=42 \
  cargo run -p bitnet-cli --features cpu -- \
  run --model <model> --prompt "Test" --temperature 0.0 > output2.txt

diff output1.txt output2.txt  # Should be identical
```

## Debugging and Troubleshooting

### Common Issues

**Issue: Test model not found**
```
Error: Model not found at tests/fixtures/models/test-model.gguf

Solution:
cargo run -p xtask -- download-model --id test-model-small
```

**Issue: Non-deterministic output**
```
Error: Token sequence differs across runs

Solution:
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export RAYON_NUM_THREADS=1
```

**Issue: Receipt validation fails**
```
Error: CPU backend verification failed: no quantized kernels found

Solution:
1. Check strict mode: BITNET_STRICT_MODE=1
2. Verify quantization support: cargo test -p bitnet-kernels --features cpu
3. Inspect receipt kernels: jq '.kernels' ci/inference.json
```

**Issue: TL LUT index panic**
```
Error: index out of bounds: the len is 1024 but the index is 2048

Solution:
1. Verify TL helper integrated: grep -r "use bitnet_kernels::tl_lut" crates/
2. Check bounds checking enabled (not using unsafe unchecked)
3. Validate block configuration: block_bytes, elems_per_block
```

### Test Debugging

**Enable verbose logging:**
```bash
RUST_LOG=debug cargo test test_ac1_cpu_forward_bos_nonzero_logits \
  --features cpu -- --nocapture
```

**Inspect KV cache:**
```rust
#[test]
fn debug_kv_cache() {
    let engine = create_test_engine().unwrap();
    let cache = engine.kv_cache.read().unwrap();

    println!("Cache length: {}", cache.len());
    for layer in 0..cache.num_layers() {
        let (k, v) = cache.get(layer).unwrap();
        println!("Layer {}: K shape {:?}, V shape {:?}", layer, k.shape(), v.shape());
    }
}
```

**Validate logits distribution:**
```rust
fn print_logits_stats(logits: &[f32]) {
    let mean = logits.iter().sum::<f32>() / logits.len() as f32;
    let variance = logits.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / logits.len() as f32;
    let std_dev = variance.sqrt();

    println!("Logits stats: mean={:.4}, std={:.4}, min={:.4}, max={:.4}",
             mean, std_dev,
             logits.iter().cloned().fold(f32::INFINITY, f32::min),
             logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
}
```

## References

### Related Documentation

- `docs/explanation/cpu-inference-architecture.md` - Architecture design
- `docs/explanation/cpu-inference-api-contracts.md` - API specifications
- `docs/explanation/tl-lut-helper-spec.md` - TL LUT helper design
- `docs/explanation/receipt-cpu-validation-spec.md` - Receipt validation logic
- `docs/development/test-suite.md` - Testing framework overview
- `docs/development/validation-framework.md` - Quality assurance system

### Issue References

- **Issue #462:** CPU Forward Pass with Real Inference (this test plan)
- **Issue #254:** Real inference specification (predecessor)
- **PR #461:** Strict quantized hot-path enforcement
- **PR #452:** Receipt verification gate

### Existing Test Patterns

- `crates/bitnet-inference/tests/issue_254_*.rs` - Real inference tests
- `crates/bitnet-cli/tests/*.rs` - CLI integration tests
- `xtask/tests/*.rs` - Tooling tests
- `crates/bitnet-kernels/src/*/tests.rs` - Kernel unit tests

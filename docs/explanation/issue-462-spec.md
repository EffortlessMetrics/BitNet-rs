# Issue #462: Implement CPU Forward Pass with Real Inference (Cross MVP)

## Context

BitNet.rs currently returns placeholder logits (zeros [1, 32000]) from the CPU forward path in `CpuInferenceEngine::forward_parallel()`. This blocks actual token generation and question-answering workflows, preventing the CPU MVP from performing real neural network inference.

**Current State:**
- `CpuInferenceEngine::forward_parallel()` returns zero-filled tensors instead of computed logits
- QuantizedLinear layer selection works (I2S/TL1/TL2 native quantized paths or FP32 fallback)
- CLI can load GGUF models, tokenizers, and create inference engines
- Strict mode (`bitnet-inference/src/strict_mode.rs`) enforces quantization-only hot paths
- KV cache structures exist but are not populated during forward pass

**Problem Impact:**
- CPU inference pipeline is non-functional for production use cases
- Receipt validation cannot verify honest compute (no real kernel IDs)
- Cross-validation against C++ reference implementation is blocked
- User-facing CLI cannot perform question-answering workflows

**Affected BitNet.rs Components:**
- `bitnet-inference` (core inference engine with CPU forward pass)
- `bitnet-cli` (user-facing inference commands)
- `bitnet-kernels` (quantized linear algebra operations)
- `xtask` (receipt verification and validation gates)

**Inference Pipeline Impact:**
- **Model Loading** ✓ (functional)
- **Quantization** ✓ (functional with I2S/TL1/TL2 support)
- **Kernels** ⚠ (quantized kernels exist but not integrated into forward pass)
- **Inference** ✗ (placeholder implementation returns zeros)
- **Output** ✗ (no token generation or sampling)

## User Story

As a BitNet.rs developer or end-user, I want the CPU inference engine to perform real neural network forward passes with quantized weights so that I can generate coherent text from language models on CPU-only systems without GPU dependencies.

## Acceptance Criteria

### P0 - Critical Path to MVP

**AC1: CPU Forward Pass Real Inference**
- Replace `CpuInferenceEngine::forward_parallel()` placeholder with real one-step autoregressive decode
- Implement full transformer layer processing: embedding → LayerNorm → Q/K/V projection → attention → output projection → FFN → logits
- Use existing `QuantizedLinear` paths (I2S/TL1/TL2) with strict guards blocking FP32 staging in hot path
- KV cache management: append computed K,V tensors to `KVCache[layer]: [H, T, Dh]` structure
- Single-step attention: `softmax((Q @ K^T) / sqrt(Dh)) @ V` with causal masking for position `T`
- Return non-zero finite logits tensor `[1, vocab_size]` for BOS token input
- **File:** `crates/bitnet-inference/src/cpu.rs`
- **Tests:** `cargo test -p bitnet-inference test_cpu_forward_nonzero --no-default-features --features cpu`

**AC2: CLI Priming and Decode Loop**
- Implement priming loop: tokenize prompt → feed each token through `forward()` → update KV cache → discard logits
- Implement decode loop: `logits → sample(greedy|top-k|top-p) → next_token → forward() → print → repeat`
- Enable `bitnet run --model <path> --prompt "Q: What is 2+2? A:" --max-new-tokens 16 --temperature 0.0` workflow
- Print generated tokens to stdout with streaming output
- **File:** `crates/bitnet-cli/src/commands/inference.rs`
- **Tests:** `cargo test -p bitnet-cli test_inference_command_priming --no-default-features --features cpu`

### P1 - Quality & Validation

**AC3: Receipt Honesty - CPU Kernel Validation**
- Add CPU symmetry rule to `xtask/src/verify_receipt.rs`: when `backend="cpu"`, require ≥1 CPU quantized kernel ID
- CPU quantized kernel prefixes: `i2s_`, `tl1_`, `tl2_` (use `starts_with()` not `contains()`)
- Excluded prefixes: `dequant*`, `fp32_*`, `fallback_*` (classified as fallback, not quantized)
- Keep explicit fallback classification: `kernel_id == "matmul_f32"`
- Ensure GPU negative test (CUDA backend with CPU kernels) continues to fail
- **File:** `xtask/src/verify_receipt.rs`
- **Tests:** `cargo test -p xtask test_receipt_cpu_kernel_requirement --no-default-features --features cpu`

**AC4: TL1/TL2 LUT Index Helper**
- Create new module `crates/bitnet-kernels/src/tl_lut.rs` with safe LUT index helper
- Function signature: `pub fn lut_index(block_idx: usize, elem_in_block: usize, block_bytes: usize, elems_per_block: usize) -> Result<usize>`
- Include bounds checking: `elem_in_block < elems_per_block`, computed index within LUT bounds
- Update call sites in `crates/bitnet-inference/src/layers/quantized_linear.rs` TL1/TL2 paths to use helper
- Re-enable TL tests in `crates/bitnet-inference/tests` (remove `#[ignore]` attributes)
- **Files:** `crates/bitnet-kernels/src/tl_lut.rs` (new), `crates/bitnet-inference/src/layers/quantized_linear.rs`
- **Tests:** `cargo test -p bitnet-kernels test_tl_lut_index_bounds --no-default-features --features cpu`

**AC5: Baseline Receipt and README Quickstart**
- Pin baseline CPU receipt to `docs/baselines/YYYYMMDD-cpu.json` (copy from `ci/inference.json` after successful run)
- Add 10-line quickstart section to `README.md` showing question → answer workflow example
- Document deterministic inference: `BITNET_DETERMINISTIC=1 BITNET_SEED=42 cargo run -p bitnet-cli ...`
- Include expected output format and feature flag requirements (`--no-default-features --features cpu`)
- **Files:** `docs/baselines/<timestamp>-cpu.json`, `README.md`
- **Validation:** Baseline receipt must validate with `cargo run -p xtask -- verify-receipt`

### P2 - Optional GPU Validation

**AC6: GPU Baseline Receipt Validation**
- Create GPU baseline test in `crates/bitnet-inference/tests/issue_260_mock_elimination_inference_tests.rs`
- Gate with `#[cfg(feature="gpu")]` and environment variable `BITNET_ENABLE_GPU_TESTS=1`
- Skip cleanly with `#[ignore]` if `Device::new_cuda(0)` fails (no GPU available)
- Run 128-token GPU benchmark with in-memory receipt capture
- Assert: receipt contains ≥1 GPU kernel ID prefix (`gemm_`, `wmma_`, `cuda_`, `i2s_gpu_`, `tl*_gpu_`)
- Throughput envelope: 50-100 tok/s baseline (allow per-fingerprint widening via `.ci/fingerprints.yml`)
- **File:** `crates/bitnet-inference/tests/issue_260_mock_elimination_inference_tests.rs`
- **Tests:** `BITNET_ENABLE_GPU_TESTS=1 cargo test -p bitnet-inference test_gpu_baseline_receipt --no-default-features --features gpu`

**AC7: Enable Required CPU Gate in CI**
- Update repository branch protection settings to require "Model Gates (CPU)" job
- Block PRs with invalid receipts (e.g., `compute_path:"mock"`, missing CPU kernels)
- Document gate requirements in `docs/reference/validation-gates.md`
- **Validation:** Manual verification of GitHub repository settings post-merge

## Technical Implementation Notes

**Affected Crates:**
- `bitnet-inference` (CPU forward pass engine, layer processing)
- `bitnet-cli` (inference command wiring, priming/decode loops)
- `bitnet-kernels` (TL LUT helper, quantized operations)
- `xtask` (receipt verification, kernel ID validation)

**Pipeline Stages:**
- Model Loading ✓ (functional)
- Quantization ✓ (I2S/TL1/TL2 support)
- Kernels ⚠ (integration required)
- Inference ✗ (placeholder → real implementation)
- Output ✗ (sampling and token generation)

**Performance Considerations:**
- CPU-only inference with SIMD optimization (AVX2/AVX-512/NEON)
- Memory efficiency for KV cache management (in-place updates)
- Deterministic inference via `BITNET_DETERMINISTIC=1 BITNET_SEED=42`
- Throughput baseline: TBD via CPU benchmark receipt (target ≥5 tok/s for 2B model)

**Quantization Requirements:**
- I2S: Primary 2-bit signed quantization (99%+ accuracy vs FP32)
- TL1/TL2: Table lookup quantization with safe LUT indexing
- Strict mode enforcement: block FP32 staging in hot path via `crates/bitnet-inference/src/strict_mode.rs`
- Cross-validation: systematic comparison with C++ reference via `cargo run -p xtask -- crossval`

**Feature Flags:**
- CPU builds: `cargo build --no-default-features --features cpu`
- GPU builds: `cargo build --no-default-features --features gpu`
- Testing: `cargo test --workspace --no-default-features --features cpu`
- Benchmarking: `cargo run -p xtask -- benchmark --model <path> --tokens 128`

**GGUF Compatibility:**
- Model loading via `bitnet-models` with GGUF format support
- Tensor alignment and metadata validation
- LayerNorm FP16/FP32 preservation (no quantization)
- Verification: `cargo run -p xtask -- verify --model <path>`

**Deterministic Inference:**
- Environment: `BITNET_DETERMINISTIC=1 BITNET_SEED=42`
- Single-threaded: `RAYON_NUM_THREADS=1`
- Reproducible sampling with fixed seed
- GPU determinism: `BITNET_GPU_FAKE=none` for testing

**Testing Strategy:**
- **Unit Tests:** BOS token → non-zero finite logits (AC1)
- **Integration Tests:** 16-token greedy decode without panic (AC2)
- **Receipt Tests:** CPU backend requires CPU quantized kernels (AC3)
- **LUT Tests:** Bounds checking and index safety (AC4)
- **E2E Tests:** CLI question-answering workflow (AC5)
- **GPU Tests:** Optional GPU baseline validation (AC6)
- **TDD Approach:** All tests use `// AC:ID` comment tags for traceability

**Cross-Validation:**
- C++ reference implementation compatibility via `cargo run -p xtask -- crossval`
- Validate logits output matches C++ within tolerance (cosine similarity ≥0.99)
- Use `BITNET_GGUF` environment variable for model path override
- Auto-discovery of models in `models/` directory if not set

**Error Handling:**
- Use `anyhow::Result<T>` patterns throughout
- Preserve error context with `.context()` annotations
- Graceful fallback for GPU unavailability
- Clear error messages for model loading failures

## Definition of Done

**AC1 (CPU Forward Pass):**
- [ ] `CpuInferenceEngine::forward_parallel()` implements real transformer forward pass
- [ ] BOS token input returns non-zero finite logits `[1, vocab_size]`
- [ ] KV cache populated with K,V tensors per layer
- [ ] Uses QuantizedLinear I2S/TL1/TL2 paths (no FP32 staging)
- [ ] Unit test: `cargo test -p bitnet-inference test_cpu_forward_nonzero --no-default-features --features cpu` passes

**AC2 (CLI Wiring):**
- [ ] Priming loop tokenizes and feeds prompt through forward pass
- [ ] Decode loop samples next token and generates text
- [ ] CLI command: `cargo run -p bitnet-cli --no-default-features --features cpu -- run --model <path> --prompt "Q: What is 2+2? A:" --max-new-tokens 16 --temperature 0.0` produces coherent output
- [ ] Integration test: 16-token greedy decode completes without panic

**AC3 (Receipt CPU Validation):**
- [ ] `xtask/src/verify_receipt.rs` enforces CPU kernel requirement for `backend="cpu"`
- [ ] CPU quantized prefixes validated: `i2s_`, `tl1_`, `tl2_`
- [ ] Excluded prefixes: `dequant*`, `fp32_*`, `fallback_*`
- [ ] Test: `cargo test -p xtask test_receipt_cpu_kernel_requirement --no-default-features --features cpu` passes
- [ ] Negative test: GPU backend with CPU kernels fails validation

**AC4 (TL LUT Helper):**
- [ ] New module `crates/bitnet-kernels/src/tl_lut.rs` with `lut_index()` function
- [ ] Bounds checking for `elem_in_block` and computed index
- [ ] Call sites in `quantized_linear.rs` updated to use helper
- [ ] TL tests re-enabled (remove `#[ignore]`)
- [ ] Test: `cargo test -p bitnet-kernels test_tl_lut_index_bounds --no-default-features --features cpu` passes

**AC5 (Baseline & Quickstart):**
- [ ] Baseline receipt saved to `docs/baselines/<timestamp>-cpu.json`
- [ ] Receipt validates: `cargo run -p xtask -- verify-receipt` passes
- [ ] README.md updated with 10-line quickstart example
- [ ] Deterministic inference documented with environment variables

**AC6 (GPU Baseline - Optional):**
- [ ] GPU test in `issue_260_mock_elimination_inference_tests.rs` with proper feature gating
- [ ] Skip cleanly when GPU unavailable
- [ ] 128-token benchmark captures receipt with GPU kernel IDs
- [ ] Throughput baseline established (50-100 tok/s)
- [ ] Test: `BITNET_ENABLE_GPU_TESTS=1 cargo test -p bitnet-inference test_gpu_baseline_receipt --no-default-features --features gpu` passes

**AC7 (CI Gate - Post-Merge):**
- [ ] GitHub branch protection requires "Model Gates (CPU)" job
- [ ] Invalid receipts block PR merges
- [ ] Documentation updated in `docs/reference/validation-gates.md`

## Priority Breakdown

**P0 (Critical Path - 1-2 days):**
- AC1: CPU Forward Pass Real Inference
- AC2: CLI Priming and Decode Loop

**P1 (Quality & Validation - 1 day):**
- AC3: Receipt Honesty - CPU Kernel Validation
- AC4: TL1/TL2 LUT Index Helper
- AC5: Baseline Receipt and README Quickstart

**P2 (Optional GPU - 0.5-1 day):**
- AC6: GPU Baseline Receipt Validation
- AC7: Enable Required CPU Gate in CI

**Total Estimated Effort:** 2.5-4 days (P0+P1+P2 complete)

## Success Metrics

- CPU inference produces non-zero logits for all supported quantization types (I2S/TL1/TL2)
- Question-answering workflow completes end-to-end: `cargo run -p bitnet-cli --no-default-features --features cpu -- run --model <path> --prompt "Q: What is 2+2? A:" --max-new-tokens 16`
- Receipt validation enforces honest compute (no mock kernels, no FP32 staging)
- Cross-validation with C++ reference achieves ≥99% cosine similarity
- Deterministic inference reproducible across runs with same seed
- Baseline receipt establishes CPU throughput benchmark (target ≥5 tok/s for 2B model)

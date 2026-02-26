# Issue #469: MVP Sprint

## Context

Following the successful merge of QK256 (GGML I2_S) pure-Rust quantization support in PR #468, this polish sprint addresses UX improvements, logging consistency, and runtime guardrails to ensure production readiness. The QK256 implementation introduced dual-flavor I2_S quantization (BitNet32-F16 and QK256) with automatic format detection, but several developer experience and operational concerns require refinement before v0.1.0-mvp release.

**Affected BitNet-rs Components:**
- `bitnet-models`: GGUF loader strict mode, K/V cache guardrails, QK256 tolerance handling
- `bitnet-cli`: CLI flag for strict loader mode, diagnostic logging
- `bitnet-quantization`: QK256 size tolerance centralization, K/V cache post-slice assertions
- `bitnet-tokenizers`: Vocab size parity exposure for cross-validation
- `crossval`: Parity harness receipt generation, timeout consistency with main inference
- `xtask`: FFI build hygiene (compile_cpp_shim consolidation), CI parity smoke test improvements
- `docs/`: README quick-start updates, QK256 usage documentation

**Neural Network Inference Pipeline Impact:**
- **Model Loading**: Strict mode validation with graceful degradation
- **Quantization**: QK256 tolerance enforcement and diagnostic logging
- **Inference**: K/V cache guardrails with once-per-layer warning system
- **Cross-Validation**: Receipt generation consistency and timeout alignment
- **FFI Bridge**: Build hygiene and -isystem usage for cleaner compiler output

## User Story

As a **BitNet-rs developer or production deployment engineer**, I want **polished UX, consistent logging, and runtime guardrails for the QK256 MVP** so that **I can confidently deploy QK256-quantized models with clear diagnostics, predictable error handling, and cross-validation parity with the C++ reference implementation**.

## Acceptance Criteria

**AC1: Loader Strict Mode UX**
- Given a user wants explicit control over GGUF loading tolerance, when they pass `--strict-loader` to `bitnet-cli run/chat/inspect`, then the loader MUST enforce strict QK256 size validation (reject tensors with >0.1% size deviation) and emit clear error messages indicating the exact tensor name, expected size, and actual size deviation percentage.
- The strict mode MUST be opt-in (default: permissive with warnings) to maintain backward compatibility.
- Diagnostic output MUST include actionable guidance: "Tensor 'blk.0.attn_q.weight' size mismatch: expected 98304 bytes (256-elem blocks), got 98560 bytes (+0.26% deviation). Use --strict-loader to enforce exact alignment or regenerate GGUF with clean export."

**AC2: QK256 Tolerance & Logs Centralization**
- Given QK256 tolerance logic is currently scattered, when the loader encounters QK256 tensors, then it MUST use a single centralized constant `QK256_SIZE_TOLERANCE` (default: 0.001 = 0.1%) defined in `bitnet-quantization` and re-exported in `bitnet-models`.
- All tolerance-related logging MUST reference this constant and emit at `warn!` level in permissive mode, `error!` level in strict mode.
- The tolerance constant MUST be documented in `docs/reference/quantization-support.md` with rationale for the 0.1% threshold (accounts for metadata padding while rejecting grossly misaligned tensors).

**AC3: K/V Cache Guardrails**
- Given K/V cache slicing is safety-critical for inference correctness, when the inference engine performs cache operations, then it MUST assert post-slice dimensions match expected `[batch, n_heads, seq_len, head_dim]` shapes with `debug_assert!` in hot path and explicit `anyhow::ensure!` in cache initialization.
- Dimension mismatches MUST emit once-per-layer warnings (using `std::sync::Once` guards) to avoid log spam: "Layer 3 K-cache shape mismatch: expected [1, 16, 128, 64], got [1, 16, 127, 64]. This indicates a cache management bug."
- K/V cache assertions MUST validate:
  - Batch dimension == 1 (no batching support yet)
  - Number of heads matches model config (`n_heads` or `n_heads_kv` for GQA)
  - Sequence length ≤ max context length
  - Head dimension matches `d_head = d_model / n_heads`

**AC4: Parity Harness Receipts & Timeout Consistency**
- Given cross-validation requires reproducible metrics, when `cargo run -p xtask -- crossval` executes parity tests, then it MUST generate inference receipts (`ci/inference.json`) with identical schema to main inference receipts (v1.0.0).
- Receipt fields MUST include:
  - `compute_path: "real"` (no mock inference)
  - `kernel_ids: Vec<String>` with actual kernel invocations
  - `parity: { cpp_available: bool, cosine_similarity: f64, exact_match_rate: f64, status: "ok|warn|error" }`
  - `backend: "cpu"|"cuda"` matching runtime detection
- Timeout handling MUST use the same `tokio::time::timeout` duration (default: 60s) as main inference to ensure consistent behavior in CI environments.

**AC5: Tokenizer Parity**
- Given cross-validation compares tokenization output, when the tokenizer is initialized, then it MUST expose `real_vocab_size()` method returning the actual vocabulary size from the tokenizer model (not the padded size from GGUF metadata).
- The `real_vocab_size()` MUST be used in parity harness assertions: `assert_eq!(rust_tokenizer.real_vocab_size(), cpp_tokenizer.vocab_size(), "Vocab size mismatch breaks tokenization parity")`.
- Tokenizer debug output MUST log both sizes: `Tokenizer initialized: real_vocab_size=32000, gguf_padded_size=32064 (padding for alignment)`.

**AC6: FFI Build Hygiene**
- Given FFI compilation generates verbose warnings, when `build.rs` compiles C++ shims, then it MUST use a single consolidated `compile_cpp_shim()` function (defined in `xtask/src/ffi.rs` and re-used in all FFI build scripts).
- The `compile_cpp_shim()` MUST use `-isystem` for third-party includes (CUDA, BitNet C++) instead of `-I` to suppress warnings from external headers.
- FFI build output MUST emit only BitNet-rs-specific warnings (suppressing CUDA SDK and C++ reference implementation warnings).
- All `build.rs` files in `bitnet-kernels`, `bitnet-quantization`, and `crossval` MUST migrate to the unified `compile_cpp_shim()` helper.

**AC7: CI/Parity Smoke Test**
- Given CI must validate QK256 parity, when `.github/workflows/parity.yml` runs, then it MUST set `BITNET_DISABLE_MINIMAL_LOADER=1` to force full GGUF loader path (bypassing minimal loader optimizations that might mask QK256 issues).
- The parity smoke test (`scripts/parity_smoke.sh`) MUST execute with both:
  - BitNet32-F16 models (existing baseline)
  - QK256 models (new format validation)
- CI MUST fail if cosine similarity < 0.99 or exact match rate < 0.95 for either format.
- Smoke test output MUST include both formats in summary: "Parity validated: BitNet32-F16 (cosine=0.9987), QK256 (cosine=0.9923)".

**AC8: Docs & README Quick-Start**
- Given new users need QK256 guidance, when they read `README.md` and `docs/quickstart.md`, then both documents MUST include QK256-specific quick-start sections.
- `README.md` MUST add:
  - QK256 format explanation (256-elem blocks, GGML compatibility)
  - Command: `cargo run -p bitnet-cli --no-default-features --features cpu -- run --model model-qk256.gguf --prompt "Test" --max-tokens 16`
  - Link to `docs/howto/use-qk256-models.md`
- `docs/quickstart.md` MUST add:
  - Section: "Using QK256 Models (GGML I2_S)"
  - Automatic flavor detection explanation
  - Strict loader mode usage: `--strict-loader`
  - Cross-validation command: `cargo run -p xtask -- crossval`
- Both documents MUST reference dual-flavor architecture doc: `docs/explanation/i2s-dual-flavor.md`

## Technical Implementation Notes

### Affected Crates
- **bitnet-cli**: Add `--strict-loader` flag, wire to loader config
- **bitnet-models**: Implement strict mode toggle, centralize QK256 tolerance, K/V guardrails
- **bitnet-quantization**: Define `QK256_SIZE_TOLERANCE` constant, export for models crate
- **bitnet-tokenizers**: Add `real_vocab_size()` method, update debug logging
- **crossval**: Generate v1.0.0 receipts, align timeout with main inference
- **xtask**: Consolidate `compile_cpp_shim()` helper, update CI env vars

### Pipeline Stages Affected
- **Model Loading**: Strict mode validation, tolerance enforcement (Entry point)
- **Quantization**: QK256 format detection, size validation (Dequantization stage)
- **Inference**: K/V cache guardrails, dimension assertions (Forward pass)
- **Cross-Validation**: Receipt generation, parity metrics (Validation stage)

### Performance Considerations
- K/V cache assertions use `debug_assert!` in release builds (zero overhead)
- Once-per-layer warnings use `std::sync::Once` guards (amortized cost)
- Strict loader mode adds <1% overhead (one-time validation at load)
- FFI build hygiene reduces compilation noise but does not affect runtime performance

### Quantization Requirements
- QK256 tolerance: 0.1% maximum deviation (rejects misaligned tensors)
- BitNet32-F16 unaffected (inline scales, no block alignment issues)
- Automatic flavor detection based on tensor size (existing logic preserved)
- Cross-validation must pass for both I2_S flavors

### Cross-Validation
- Receipt schema: v1.0.0 (matches main inference)
- Parity metrics: cosine similarity ≥ 0.99, exact match rate ≥ 0.95
- Timeout: 60s (aligned with main inference default)
- Both Rust and C++ paths must generate identical token sequences for deterministic inputs

### Feature Flags
- All changes CPU/GPU agnostic (no new feature gates required)
- Existing `--no-default-features --features cpu|gpu` patterns unchanged
- FFI build hygiene applies to `crossval` feature only

### GGUF Compatibility
- Strict loader mode enforces exact QK256 block alignment (256-elem)
- Permissive mode (default) allows up to 0.1% deviation with warnings
- Clean GGUF export (`scripts/export_clean_gguf.sh`) ensures strict compliance

### Testing Strategy
- **AC1**: Unit test `--strict-loader` flag parsing, integration test loader rejection with misaligned tensor
- **AC2**: Unit test tolerance constant usage, integration test warning/error logging
- **AC3**: Unit test K/V cache dimension assertions, integration test once-per-layer warning guards
- **AC4**: Integration test receipt generation schema, timeout behavior in crossval
- **AC5**: Unit test `real_vocab_size()` method, integration test parity assertions
- **AC6**: Build system test for `-isystem` usage, verify warning suppression
- **AC7**: CI test with `BITNET_DISABLE_MINIMAL_LOADER=1`, validate both I2_S flavors
- **AC8**: Documentation smoke test (links, code examples), quick-start reproducibility

### TDD Implementation Guidance
All tests MUST use `// AC:<ID>` comment tags for traceability:

```rust
// AC1: Strict loader mode rejects misaligned QK256 tensors
#[test]
fn test_strict_loader_rejects_misaligned_qk256() {
    let loader = GGUFLoader::new().strict_mode(true);
    let result = loader.load("tests/fixtures/misaligned-qk256.gguf");
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("size mismatch"));
}

// AC3: K/V cache guardrails validate dimensions
#[test]
fn test_kv_cache_dimension_assertions() {
    let cache = KVCache::new(1, 16, 2048, 64);
    // Correct slice succeeds
    let k_slice = cache.slice_k(0, 0..128);
    assert_eq!(k_slice.shape(), &[1, 16, 128, 64]);

    // Incorrect dimensions panic in debug mode
    #[cfg(debug_assertions)]
    {
        let result = std::panic::catch_unwind(|| {
            cache.slice_k_invalid(0, 0..128)
        });
        assert!(result.is_err());
    }
}

// AC5: Tokenizer exposes real vocab size
#[test]
fn test_tokenizer_real_vocab_size() {
    let tokenizer = Tokenizer::from_file("tests/fixtures/tokenizer.json").unwrap();
    assert_eq!(tokenizer.real_vocab_size(), 32000);
    assert_eq!(tokenizer.padded_vocab_size(), 32064); // GGUF alignment
}
```

### Dependencies and Constraints
- **Ordering**: AC6 (FFI hygiene) should merge first (reduces build noise for remaining work)
- **Blocking**: AC7 (CI smoke test) depends on AC1/AC2 (strict loader, tolerance) for meaningful validation
- **Documentation**: AC8 (docs) should merge last (ensures examples reflect final UX)

### Success Criteria
- All 8 acceptance criteria implemented with TDD test coverage
- CI passes with QK256 parity validation enabled (`BITNET_DISABLE_MINIMAL_LOADER=1`)
- Documentation quick-starts validated with real QK256 models
- No performance regression in inference benchmarks (K/V assertions use debug_assert)
- FFI build warnings reduced by >80% (external headers via -isystem)

### Routing
**FINALIZE → spec-analyzer** for requirements validation and technical feasibility assessment.

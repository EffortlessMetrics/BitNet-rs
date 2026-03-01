# Implementation Targets

_Last updated: 2026-02-28 | Status: v0.2.1-dev (pre-alpha)_

This document defines the four highest-priority implementation targets for BitNet-rs.
Each target includes current state, concrete steps to "fully working," acceptance criteria,
and verification commands.

---

## Target 1: QK256 AVX2 >=3x Uplift

### Current State

- **Scalar baseline**: ~0.1 tok/s for 2B models (MVP, ships in v0.2.0)
- **AVX2 foundation (merged)**: ~0.12 tok/s (~1.2x uplift) — AVX2 dequantization path wired in
- **Target**: >=3x over scalar baseline (>=0.3 tok/s for 2B models)
- **Location**: `crates/bitnet-kernels/src/cpu/x86.rs` and `crates/bitnet-quantization/src/i2s/qk256_avx2.rs`

### Steps to Fully Working

1. **Nibble-LUT unpack via `pshufb`** — Map 2-bit to signed i8 using VPSHUFB shuffle table
   - 32 elements per AVX2 register; process 4 codes per shuffle
   - Expected: ~1.8x over scalar on this step alone
2. **FMA tiling (8-16 row unroll)** — Unroll dot-products across 8-16 output rows with VFMADD
   - Eliminate loop overhead and improve ILP
   - Expected: ~2.5x cumulative
3. **Load combine** — Batch AVX2 loads to reduce crossing penalty; use `_mm256_loadu_si256` once per block
4. **Prefetch** — Software prefetch next code block and input row (use `_mm_prefetch` with T0 hint)
   - Expected: ~3x cumulative on bandwidth-limited workloads

### Acceptance Criteria

- `cargo bench --bench kernel_benchmarks --features cpu,avx2` shows >=3x vs scalar baseline
- Property tests pass: `cargo test -p bitnet-kernels --features cpu,avx2 -- avx2_correctness`
- Max absolute difference vs scalar: <=1e-5 on randomized shapes (already validated for step 1)
- No regression in CPU golden path E2E tests: `cargo nextest run -p bitnet-inference --features cpu`

### Verification Commands

```bash
# Benchmark (shows current vs target)
cargo bench --bench kernel_benchmarks --features cpu,avx2

# Correctness check
cargo test -p bitnet-kernels --no-default-features --features cpu -- qk256_avx2

# E2E regression guard
cargo nextest run -p bitnet-inference --no-default-features --features cpu
```

---

## Target 2: CUDA Kernel Implementation

### Current State

- **7 kernel stubs defined**: RMSNorm, Attention, Softmax, QK256 GEMV, RoPE, BatchNorm, Activations
- **GPU-HAL scaffolded**: 16-module abstraction layer in `bitnet-gpu-hal`
- **Receipt system ready**: Schema v1.0.0 with `backend="cuda"` validation gate
- **Status**: No kernel produces real PTX output; all CUDA paths fall through to CPU

### Steps to Fully Working

1. **RMSNorm** (`bitnet_kernels.cu`) — Implement warp-reduce RMS + element-wise scale; test against CPU scalar
2. **Softmax** — Numerically stable online softmax (two-pass or Welford); test with temperature scaling
3. **QK256 GEMV kernel** — 2-bit dequant + FMA dot product; must match scalar within 1e-4
4. **Attention** — Scaled dot-product with causal mask; fused QKV for memory efficiency
5. **RoPE** — Rotary position embedding table lookup + in-place application
6. **Wire into inference** — Register kernels in `KernelRegistry`; update `BackendCapabilities`
7. **Receipt validation** — Run `cargo run -p xtask -- benchmark --model <model>` and verify `kernels` list shows CUDA IDs

### Acceptance Criteria

- `cargo run -p xtask -- verify-receipt --require-gpu-kernels` passes
- Cosine similarity vs CPU: >=0.999 per token position (via `crossval-per-token`)
- `cargo nextest run -p bitnet-inference --features gpu` — GPU tests not ignored
- CUDA smoke lane (`gpu-smoke.yml`) uploads receipt with `compute_path="real"`

### Verification Commands

```bash
# Run CUDA smoke (requires CUDA runtime)
cargo test -p bitnet-kernels --no-default-features --features gpu -- cuda_kernel

# Per-token cross-validation against CPU reference
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 4 \
  --cos-tol 0.999

# Receipt verification
cargo run -p xtask -- benchmark --model models/model.gguf --tokens 128
cargo run -p xtask -- verify-receipt --require-gpu-kernels
```

---

## Target 3: bitnet-server Completion

### Current State

- HTTP server scaffold exists in `bitnet-server/`
- OpenAPI spec defined
- Health endpoints implemented
- **Known TODOs** (8 items):
  1. `/v1/completions` — Text completion endpoint (stream + non-stream)
  2. `/v1/chat/completions` — Chat completion with tool use
  3. `/v1/models` — Model listing endpoint
  4. Request batching — Queue multiple inference requests
  5. Streaming via SSE — Server-sent events for token streaming
  6. Auth middleware — API key validation
  7. Rate limiting — Per-key request throttling
  8. Prometheus metrics — `/metrics` endpoint with token throughput

### Steps to Fully Working

1. Implement `/v1/completions` (non-streaming) — Wire to `bitnet-inference` engine; return JSON
2. Add SSE streaming — Use `axum::response::sse::Sse` with token-by-token events
3. Implement `/v1/chat/completions` — Apply prompt template from request `messages`
4. Add `/v1/models` — List loaded models from config
5. Wire auth middleware — Header extraction + key validation
6. Add Prometheus export — `metrics` crate + `axum-prometheus`
7. Integration tests — Test all endpoints with mock engine

### Acceptance Criteria

- `curl http://localhost:8080/v1/completions -d '{"model":"...","prompt":"Hello"}'` returns valid JSON
- Streaming works: `curl -N http://localhost:8080/v1/completions -d '{"stream":true,...}'` emits SSE tokens
- All 8 TODOs resolved with tests
- `cargo nextest run -p bitnet-server --features cpu` passes without ignoring server tests

### Verification Commands

```bash
# Start server
cargo run -p bitnet-server --features cpu -- --model models/model.gguf

# Test completion endpoint
curl -s http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"bitnet","prompt":"Hello","max_tokens":8}'

# Test health
curl http://localhost:8080/health

# Run server tests
cargo nextest run -p bitnet-server --no-default-features --features cpu
```

---

## Target 4: Fuzz Wave-10 and Beyond

### Current State

- **49 targets** across 9 waves (Wave 9: softmax_stability, embedding_lookup, memory_layout)
- All targets run nightly with 60-second budget
- Corpus caching per-target in GitHub Actions
- Crash artifacts uploaded automatically

### Planned Wave-10 Targets (5 targets)

1. **conv1d_quantized** — Fuzz 1D convolution with quantized weights; check for panics and numeric stability
2. **pooling_ops** — Average/max pooling boundary conditions with arbitrary shapes
3. **embedding_table_lookup** — Random token IDs with out-of-bounds detection
4. **softmax_extreme_values** — Fuzz softmax with inf/nan/denormal inputs
5. **rope_extreme_positions** — RoPE with large position indices (overflow detection)

### Steps to Fully Working

1. Create `fuzz/fuzz_targets/conv1d_quantized.rs` — Use `arbitrary::Arbitrary` for input shape + weights
2. Create remaining 4 target files following Wave 9 patterns
3. Add to `fuzz/Cargo.toml` under `[[bin]]` entries
4. Update `nightly-fuzz.yml` to include new targets
5. Run locally: `cargo +nightly fuzz run conv1d_quantized -- -max_total_time=60`
6. Build initial corpus from edge cases (zero-length, max-size, boundary values)

### Acceptance Criteria

- `cargo +nightly fuzz build` compiles all 54 targets (49 + 5 new)
- Nightly workflow runs all targets without timeout
- No new crashes in corpus from initial 60-second runs
- Wave-10 targets listed in `docs/development/test-suite.md` fuzz table

### Verification Commands

```bash
# Build all fuzz targets
cargo +nightly fuzz build

# Run a single wave-10 target
cargo +nightly fuzz run conv1d_quantized -- -max_total_time=60

# List all targets
cargo +nightly fuzz list

# CI: nightly workflow
gh workflow run nightly-fuzz.yml
```

---

## Summary Table

| Target | Current State | Key Blocker | Estimated Scope |
|--------|--------------|-------------|-----------------|
| QK256 AVX2 >=3x | 1.2x (foundation merged) | Nibble-LUT + FMA tiling | ~200 LOC SIMD |
| CUDA Kernels | 7 stubs, no PTX | RMSNorm to chain | ~2000 LOC CUDA C |
| bitnet-server | Scaffold, 8 TODOs | `/v1/completions` endpoint | ~1000 LOC Rust |
| Fuzz Wave-10 | 49 targets done | 5 new target files | ~500 LOC Rust |

All targets are tracked in the project's GitHub issues. Contributions welcome — see `docs/development/` for patterns and `CLAUDE.md` for build commands.

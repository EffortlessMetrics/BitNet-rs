# PR Plan - BitNet-rs MVP Finalization

This document outlines the PR slicing strategy for the final MVP features. Each PR is focused on a single concern with clear acceptance criteria.

## PR-A: Unify Stop Logic + UTF-8 Tail Window

**Branch**: `feat/unify-stop-logic`

**Title**: `inference(stop): unify stops IDs→EOS→strings; add UTF-8 tail window`

**Description**:
Unifies stop sequence evaluation across all generation paths (run, chat, streaming) with priority-based checking:
1. Token IDs (O(1) lookup)
2. EOS token (from tokenizer or explicit)
3. String sequences (rolling UTF-8-safe tail buffer)

Adds configurable `stop_string_window` parameter to control tail buffer size (default 64 bytes).

**Files Changed**:
- `crates/bitnet-inference/src/generation/autoregressive.rs`
- `crates/bitnet-inference/src/generation/streaming.rs`
- `crates/bitnet-cli/src/commands/run.rs`
- `crates/bitnet-cli/src/commands/chat.rs`

**Acceptance Criteria**:
- All generation paths call unified `should_stop()` method
- Tests pass: `cargo test -p bitnet-inference stop_ --features cpu`
- CLI supports `--stop-string-window <N>` flag
- UTF-8 safety validated with property tests

**Test Command**:
```bash
cargo test -p bitnet-inference --features cpu --no-default-features -- stop_
```

**Commit Message Template**:
```
inference(stop): unify stop evaluation across all generation paths

- Add priority-based stop checking: token IDs → EOS → strings
- Implement UTF-8-safe rolling tail buffer for string matching
- Add configurable stop_string_window parameter (default: 64 bytes)
- Unify behavior across run, chat, and streaming modes

Tests: cargo test -p bitnet-inference stop_
Fixes: #<issue-number>
```

---

## PR-B: QK256 AVX2 Foundation

**Branch**: `feat/qk256-avx2-foundation`

**Title**: `simd(qk256,avx2): runtime dispatch + parity tests + benches`

**Description**:
Implements AVX2-accelerated QK256 dequantization with runtime CPU feature detection and scalar fallback.
Initial implementation achieves ~1.2× speedup with correctness parity ≤1e-5 max absolute difference.

Future optimizations planned for ≥3× target: nibble-LUT via pshufb, FMA tiling, load combining, prefetch.

**Files Changed**:
- `crates/bitnet-models/src/gguf/quantization/qk256.rs` (AVX2 impl)
- `crates/bitnet-models/src/gguf/quantization/qk256_dispatch.rs` (runtime dispatch)
- `crates/bitnet-models/tests/qk256_avx2_correctness.rs` (parity tests)
- `crates/bitnet-kernels/benches/kernel_benchmarks.rs` (AVX2 benchmarks)
- `crates/bitnet-kernels/Cargo.toml` (AVX2 feature)

**Acceptance Criteria**:
- Runtime dispatch works (AVX2 if available, scalar fallback otherwise)
- Parity tests pass: ≤1e-5 max abs diff vs scalar
- No UB (miri clean on safe wrappers)
- Benchmarks run: `cargo bench --bench kernel_benchmarks --features cpu,avx2`

**Test Commands**:
```bash
# Parity tests
cargo test -p bitnet-models qk256_avx2 --features cpu,avx2

# Benchmarks
cargo bench --bench kernel_benchmarks --features cpu,avx2 -- qk256_dequant
```

**Commit Message Template**:
```
simd(qk256,avx2): add AVX2 fast path with runtime dispatch

- Implement AVX2-accelerated QK256 dequantization
- Add runtime CPU feature detection with scalar fallback
- Achieve ~1.2× initial speedup (target ≥3× with future opts)
- Correctness parity: ≤1e-5 max absolute difference vs scalar
- Add property-based tests with randomized inputs

Benchmarks: cargo bench --bench kernel_benchmarks --features cpu,avx2
Tests: cargo test -p bitnet-models qk256_avx2
Planned: nibble-LUT pshufb, FMA tiling, prefetch for ≥3× uplift
```

---

## PR-C: Receipts & CI Gates

**Branch**: `feat/receipt-ci-gates`

**Title**: `receipts(parity): schema v1.0.0, workspace path, jq CI validation`

**Description**:
Implements production receipt schema v1.0.0 with comprehensive jq-based validation gates in CI.
Receipts are written to workspace-relative paths and validated for schema compliance, compute honesty,
kernel presence, and parity metrics.

**Files Changed**:
- `crates/bitnet-inference/src/receipts.rs` (schema v1.0.0)
- `crossval/src/lib.rs` (timeout + workspace path)
- `.github/workflows/parity-proof.yml` (jq gates)

**Acceptance Criteria**:
- CI fails if receipt missing or invalid
- jq gates validate: schema_version, compute_path, kernels, timeout, flavor
- Receipt written to `docs/baselines/<YYYY-MM-DD>/parity-bitnetcpp.json`
- Local smoke test passes: `./scripts/parity_smoke.sh`

**Test Commands**:
```bash
# Local receipt generation
export BITNET_DISABLE_MINIMAL_LOADER=1 BITNET_DETERMINISTIC=1 BITNET_SEED=42 RAYON_NUM_THREADS=1 PARITY_TEST_TIMEOUT_SECS=60
./scripts/parity_smoke.sh models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json

# Validate receipt
jq '.schema_version,.compute_path,.validation,.quant,.timeout_seconds' docs/baselines/*/parity-bitnetcpp.json
```

**Commit Message Template**:
```
receipts(parity): add schema v1.0.0 + jq CI validation gates

- Implement receipt schema v1.0.0 with kernel hygiene validation
- Add jq-based CI gates for schema, compute_path, kernels, timeout
- Write receipts to workspace-relative paths (docs/baselines/)
- Validate quant.flavor and parity.status in CI

CI gates: schema_version==1.0.0, compute_path==real, kernels.length>0
Tests: ./scripts/parity_smoke.sh + jq validation
```

---

## PR-D: Health Endpoints Polish

**Branch**: `feat/health-endpoints-polish`

**Title**: `server(health): /live,/ready,/health JSON; sysinfo metrics`

**Description**:
Implements production-ready health check endpoints with comprehensive metrics collection.
Endpoints provide liveness probes (<100ms), readiness toggles, and detailed system health.

**Files Changed**:
- `crates/bitnet-server/src/health/*.rs` (endpoints)
- `crates/bitnet-server/examples/health_endpoints.rs`
- `crates/bitnet-server/tests/health_tests.rs`

**Acceptance Criteria**:
- `/health/live` responds <100ms
- `/health/ready` toggles correctly
- `/health` returns comprehensive JSON
- Integration tests pass

**Test Commands**:
```bash
# Run example server
cargo run -p bitnet-server --no-default-features --features cpu --example health_endpoints

# Test endpoints
curl http://127.0.0.1:8080/health | jq
curl http://127.0.0.1:8080/health/live
curl http://127.0.0.1:8080/health/ready

# Run integration tests
cargo test -p bitnet-server health --features cpu
```

**Commit Message Template**:
```
server(health): add production-ready health check endpoints

- Implement /health/live (liveness probe, <100ms)
- Implement /health/ready (readiness toggle)
- Implement /health (comprehensive JSON metrics)
- Add sysinfo integration for system metrics

Tests: cargo test -p bitnet-server health
Example: cargo run -p bitnet-server --example health_endpoints
```

---

## PR-E: Documentation Sweep

**Branch**: `docs/mvp-finalization`

**Title**: `docs(cli,simd,stops,receipts): QK256 quick start; stop window; auto templates; AVX2 notes`

**Description**:
Updates README.md and CLAUDE.md with new MVP features including strict receipts, stop configuration,
QK256 AVX2 foundation, and unified stop semantics.

**Files Changed**:
- `README.md` (strict receipts, stop config)
- `CLAUDE.md` (stop semantics, QK256 AVX2)

**Acceptance Criteria**:
- Examples are copy-pasteable
- Links valid (no broken references)
- Markdown lint passes

**Test Commands**:
```bash
# Test example commands work
RUST_LOG=warn cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 4

# Run markdown lint
npm run lint:md  # or markdownlint tool
```

**Commit Message Template**:
```
docs(mvp): document strict receipts, stop config, and AVX2 foundation

README.md:
- Add strict receipts section with CI/local workflow
- Add stop configuration section (IDs→EOS→strings priority)
- Document stop_string_window parameter

CLAUDE.md:
- Document unified stop semantics across all gen paths
- Add QK256 AVX2 foundation section with performance notes
- Update planned optimizations for ≥3× target

All examples tested and copy-pasteable.
```

---

## Pre-PR Checklist

Before opening any PR, run:

```bash
# Format
cargo fmt --all -- --check

# Clippy
RUSTFLAGS="-Dwarnings" cargo clippy --workspace --all-features --all-targets -- -D warnings

# Key test shards
cargo test -p bitnet-inference --features cpu --no-default-features
cargo test -p bitnet-models --features cpu --no-default-features
cargo test -p bitnet-server --features cpu --no-default-features
cargo test -p bitnet-kernels --features cpu --no-default-features

# Parity smoke
./scripts/parity_smoke.sh models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json
```

---

## Branch Strategy

All feature branches should:
1. Branch from `main`
2. Follow naming convention: `feat/`, `docs/`, `fix/`
3. Keep commits atomic and well-described
4. Include tests for new functionality
5. Pass all quality gates before PR

## Review Process

Each PR should be:
1. Small and focused (single concern)
2. Self-contained (can merge independently)
3. Well-tested (acceptance criteria met)
4. Documented (README/CLAUDE updates if user-facing)
5. Lint-clean (fmt + clippy passing)

---

## Post-MVP Roadmap

After these PRs land, prioritize:

### AVX2 Optimization Sprint (Target: ≥3× uplift)
- Nibble LUT unpack via `pshufb` (2-bit → signed i8)
- FMA tiling (8-16 rows, unroll dot-products)
- Load combining (reduce AVX lane crossings)
- Prefetch (next code block & input ahead)
- Benchmark gates: fail PR if <2.5× at 4k cols

### Infrastructure
- Dead code cleanup sweep
- Unsafe hygiene audit (static mut → const/sync primitives)
- Issue resolution: #254, #260, #439, #469

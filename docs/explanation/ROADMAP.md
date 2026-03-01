# bitnet-rs Roadmap

_Last updated: 2026-02-28_

## Status: v0.2.1-dev — Pre-Alpha

bitnet-rs is pre-alpha software. The scaffold is mature and growing. CPU inference
works end-to-end. GPU backends are scaffolded but not yet validated. Correctness,
performance, and validation work is ongoing. Do not use in production.

### What is working

- CPU inference with SIMD optimization (AVX2/AVX-512/NEON)
- GGUF model loading and quantization (I2_S BitNet32-F16, QK256, TL1/TL2)
- Receipt verification (schema v1.0.0, 8 gates; 25/25 tests passing)
- Strict mode runtime guards (12/12 tests passing)
- Cross-validation framework (EnvGuard, serial patterns, fixture infrastructure)
- 49 fuzz targets across 9 waves (nightly CI)
- CPU golden path E2E tests (deterministic, seed=42; pinned output [140,459,459,459])
- SRP microcrate ecosystem (bitnet-logits, bitnet-gguf, bitnet-generation, bitnet-device-probe, bitnet-engine-core)
- KV cache optimization module (paged cache, LRU/SlidingWindow/AttentionBased eviction)
- GPU-HAL scaffolded (16-module abstraction; CUDA furthest along; no backend validated end-to-end)
- 38 CI workflows; 92-crate workspace; Criterion benchmarks

### Current limitations

- **QK256 performance**: Scalar kernels only (~0.1 tok/s for 2B models). Use
  `--max-tokens 4-16` for validation. AVX2 foundation merged at 1.2x; targeting >=3x.
- **GPU backends**: All scaffolded but not validated end-to-end. Metal, Vulkan,
  oneAPI, ROCm, and OpenCL have kernel stubs. CUDA is furthest along.
- **Model quality**: microsoft-bitnet-b1.58-2B-4T-gguf produces non-sensical output
  in some configurations. Known model quality limitation, not an inference engine bug.
- **~462 tests skipped** in workspace runs (all with justification strings).

---

## Dates

### Current State (v0.2.1-dev — Feb 2026)

**Phase 0 — Foundation (SHIPPED):**
- CPU inference with SIMD (AVX2/AVX-512/NEON)
- GGUF model loading and quantization (I2_S BitNet32-F16, QK256, TL1/TL2)
- Receipt verification (schema v1.0.0, 8 gates)
- Strict mode runtime guards
- Cross-validation framework (receipts, EnvGuard, serial patterns)
- 49 fuzz targets (9 waves)
- CPU golden path E2E tests (deterministic, seed=42)
- SRP microcrate ecosystem (bitnet-logits, bitnet-gguf, bitnet-generation, etc.)
- KV cache optimization module (paged cache, LRU/SlidingWindow/AttentionBased eviction)
- GPU-HAL scaffolded (16-module abstraction; backends not yet validated end-to-end)

**Phase 1 — Performance & Correctness (IN PROGRESS, Q1-Q2 2026):**
- QK256 AVX2 >=3x uplift (foundation merged at 1.2x; nibble-LUT + FMA tiling needed)
- CUDA kernel stubs to real PTX implementation (7 kernels: RMSNorm, Attention,
  Softmax, QK256 GEMV, RoPE, BatchNorm, Activations)
- Receipt validation end-to-end with real GPU runs
- Cross-validation accuracy improvement (cosine similarity targets)

**Phase 2 — Server & Multi-GPU (PLANNED, Q2-Q3 2026):**
- bitnet-server completion (8 known TODOs)
- Metal/Vulkan/oneAPI/ROCm/OpenCL backend validation
- Python bindings stability
- Performance parity with C++ reference

**Phase 3 — Stable API (FUTURE, Q3 2026+):**
- Semantic versioning policy enforcement
- Production deployment guides
- Community ecosystem

---

### v0.2.0 — Quality & Measurement: partially shipped, remainder in Phase 1

The quality infrastructure items from the original v0.2.0 plan shipped across 2025:
receipt verification, strict mode, EnvGuard, SRP microcrates, fuzz targets,
Criterion benchmarks, KV cache module. The original milestone date of Q4 2025 was
optimistic. Remaining work (QK256 SIMD uplift, CUDA kernel validation, real GPU
receipt runs) carries forward into Phase 1.

**Shipped from v0.2.0 plan:**
- Receipt verification with schema v1.0.0 and 8 quality gates
- Strict mode runtime enforcement (exit code 8 on policy violations)
- Cross-validation framework with token parity and per-position logits
- 49 fuzz targets with nightly CI (crash artifact upload)
- Criterion benchmarks (6 functions: logits pipeline, top-k, repetition penalty, etc.)
- KV cache optimization module (paged cache, eviction policies)
- GPU-HAL 16-module abstraction layer

**Carried forward:**
- QK256 SIMD >=3x uplift (AVX2 foundation at 1.2x; nibble-LUT + FMA tiling outstanding)
- CUDA kernel validation (stubs exist; PTX implementation needed)
- Real GPU receipt runs

---

### v0.3.0 — Ecosystem Integration: Q2-Q3 2026

**Practical Focus:**
- Python bindings stability and performance testing
- bitnet-server completion (currently has 8 known TODOs)
- C API completion for actual llama.cpp compatibility
- Docker deployment examples and guides

**Measured Improvements:**
- Document actual vs claimed performance with test methodology
- Cross-validation accuracy measurement against C++ reference
- Memory usage profiling with optimization recommendations
- Platform compatibility verification (Linux, macOS, Windows)
- Metal/Vulkan/oneAPI/ROCm/OpenCL backend validation (one backend at a time)

---

### v1.0.0 — Stable API: Q3 2026+

Moved from Q2 2026 to Q3 2026+ to allow adequate time for GPU backend validation
and performance parity work. Stable API cannot be declared while core compute paths
remain unvalidated.

**API Maturity:**
- Semantic versioning policy documented and enforced
- Public API surface audit and cleanup
- Breaking change policy and deprecation process
- Cross-platform compatibility verified (Linux, macOS, Windows)

**Quality Assurance:**
- Comprehensive test coverage with measured results
- Security audit and vulnerability management
- Error handling and logging standards
- Performance regression CI enforcement

---

## Success metrics

### Phase 1 Completion Criteria (Q2 2026)

**Performance (Evidence-Based):**

- QK256 AVX2 throughput >=3x scalar baseline (measured with `cargo bench --bench kernel_benchmarks`)
- CUDA kernel integration test passing with real PTX (not stubs)
- Receipt captures real GPU kernel IDs in CUDA path
- CPU golden path E2E tokens unchanged (pinned: [140,459,459,459])

**Quality (Zero-Tolerance):**

- `clippy -D warnings` = 0 in production crates
- All `unsafe` sites have safety documentation
- No bare `#[ignore]` without justification string (enforced by pre-commit hook)

**Validation:**

- CUDA receipt end-to-end: `compute_path == "real"` with GPU kernel IDs
- Cosine similarity vs C++ reference >= 0.999 on benchmark prompts

### v1.0.0 Stability Metrics (Q3 2026+)

**API Maturity:**

- Semantic versioning policy documented and enforced
- Breaking change policy with migration guides
- Public API surface area audit and cleanup
- Cross-platform compatibility verified (Linux, macOS, Windows)

**Performance Validation:**

- Benchmark suite comparing against C++ reference implementation
- Memory usage profiling with optimization recommendations
- GPU acceleration validated on at least one non-CUDA backend
- Performance regression CI gates enforced in all merge requests

**Production Readiness:**

- Docker images and deployment examples
- Security audit and vulnerability management
- Error handling and logging standards
- Monitoring and observability integration

### Long-term Success Indicators

**Technical Excellence:**

- Documented security audit results and vulnerability management
- Memory safety validation with comprehensive testing
- Platform-specific optimizations with measured performance improvements

**Ecosystem Integration:**

- Python bindings stable and tested
- Language bindings available (Python confirmed; JavaScript, Go planned)
- Edge deployment examples (WebAssembly target)

---

## Technical Architecture

### Current Architecture (v0.2.1-dev)

**Core Design Principles:**
- **Feature-gated compilation**: Default features empty, explicit feature selection required
- **Zero-copy operations**: Memory-mapped models with careful lifetime management
- **SIMD abstraction**: Unified interface over platform-specific instructions (AVX2/AVX-512/NEON)
- **Cross-validation framework**: Systematic comparison with C++ reference (bitnet.cpp + llama.cpp)
- **Universal tokenizer**: Auto-detecting backend with graceful fallback
- **Receipt verification**: Honest compute gates (schema v1.0.0, 8 gates)
- **Kernel registry**: Centralized KernelBackend/KernelCapabilities/SimdLevel in bitnet-common

**Workspace Structure (92 crates total; key crates listed):**
```
bitnet/                      # Main library with unified public API
bitnet-common/               # Shared types, traits, kernel registry, tensor validation
bitnet-models/               # Model loading (GGUF, SafeTensors)
bitnet-quantization/         # 1-bit quantization (I2_S, TL1, TL2, IQ2_S)
bitnet-kernels/              # SIMD/CUDA kernels with mixed precision
bitnet-inference/            # Autoregressive generation engine
bitnet-tokenizers/           # Universal tokenizer with auto-discovery
bitnet-server/               # HTTP server (8 known TODOs; not production-ready)
bitnet-compat/               # GGUF compatibility and diagnostics
bitnet-ffi/                  # C API for llama.cpp compatibility
bitnet-py/                   # Python bindings (PyO3 ABI3-py312)
bitnet-wasm/                 # WebAssembly bindings
bitnet-st2gguf/              # SafeTensors to GGUF converter
bitnet-cli/                  # Command-line interface and utilities
bitnet-gpu-hal/              # Unified GPU hardware abstraction (scaffold; 16 modules)
bitnet-opencl/               # Intel Arc OpenCL backend (experimental)

# SRP Microcrates (wired into CI):
bitnet-logits/               # Logits pipeline
bitnet-gguf/                 # GGUF parsing
bitnet-generation/           # Token generation
bitnet-device-probe/         # Hardware detection
bitnet-engine-core/          # Core engine
bitnet-validation/           # Validation infrastructure
bitnet-prompt-templates/     # Prompt template system

# Infrastructure:
crossval/                    # C++ reference validation framework
tests/                       # Shared test infrastructure (EnvGuard, fixtures)
xtask/                       # Developer tooling
```

### Architecture Evolution Roadmap

**Phase 1 — Performance (Q1-Q2 2026):**
- QK256 nibble-LUT + FMA tiling for >=3x AVX2 uplift
- CUDA PTX kernel implementation (replacing stubs)
- Runtime dispatch hardening

**Phase 2 — Multi-Backend (Q2-Q3 2026):**
- Metal/Vulkan/oneAPI/ROCm/OpenCL validation (one backend at a time)
- GPU-HAL backend selector tested end-to-end
- Python bindings stability

**Phase 3 — Stable API (Q3 2026+):**
- API surface area audit and cleanup
- Memory management optimization
- Error handling standardization
- Plugin architecture for custom kernels (research item)

---

## Development Workflow & Contribution Guide

### Getting Started (New Contributors)

**Quick Setup:**
```bash
# Clone and build CPU version
git clone https://github.com/EffortlessMetrics/BitNet-rs.git
cd BitNet-rs
cargo build --no-default-features --features cpu

# Run verification
cargo run -p xtask -- verify --help

# Run tests
cargo nextest run --workspace --no-default-features --features cpu
```

**Development Commands:**
```bash
# Format and lint
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings

# Cross-validation (requires model)
export BITNET_GGUF="path/to/model.gguf"
cargo run -p xtask -- crossval

# Performance testing
cargo bench --bench kernel_benchmarks --features cpu
```

### Contribution Areas

**Good First Issues:**
- Documentation improvements
- Test coverage expansion
- Example applications

**Intermediate Contributions:**
- Performance optimizations (QK256 nibble-LUT, FMA tiling)
- Platform-specific SIMD implementations
- GPU kernel implementations (PTX for CUDA path)
- Integration examples

**Advanced Contributions:**
- Novel quantization techniques
- GPU backend validation (Metal, Vulkan, oneAPI, ROCm)
- Research validation and benchmarking
- Distributed inference architecture

### Quality Standards

**Code Requirements:**
- Zero clippy warnings in production code
- All `#[ignore]` attributes must have justification strings (pre-commit enforced)
- Safety documentation for all `unsafe` code
- Performance regression protection (Criterion benchmarks)

**Documentation Standards:**
- Copy-pasteable examples for all features
- Architecture decision records for major changes
- Performance benchmarks with baseline comparisons
- Cross-platform compatibility notes

---

## Risk Management & Mitigation

### Technical Risks

**QK256 Performance Gap (High Impact, High Probability near-term):**
- **Current state**: Scalar kernels only (~0.1 tok/s for 2B models). NOT production-ready.
- **Mitigation**: AVX2 foundation merged at 1.2x; nibble-LUT + FMA tiling planned
- **Detection**: `cargo bench --bench kernel_benchmarks` tracks regression
- **Timeline**: Phase 1 (Q1-Q2 2026)

**GPU Backend Validation (High Impact, Medium Probability):**
- **Current state**: All GPU backends scaffolded; none validated end-to-end
- **Mitigation**: Focus on CUDA first (furthest along); validate one backend at a time
- **Detection**: Receipt verification with real kernel IDs (not stubs)
- **Timeline**: CUDA in Phase 1; others in Phase 2

**Performance Regression (High Impact, Medium Probability):**
- **Mitigation**: Criterion benchmarks in CI; `xtask bench-compare` available
- **Detection**: Baseline JSON comparison with configurable thresholds
- **Response**: Immediate revert policy for regressions >5%

**Memory Safety Issues (High Impact, Low Probability):**
- **Mitigation**: Comprehensive unsafe code documentation; 49 fuzz targets nightly
- **Detection**: Address Sanitizer; nightly fuzzing with crash artifact upload
- **Response**: Security advisory process and immediate patch releases

### Ecosystem Risks

**Dependency Vulnerabilities (Medium Impact, Medium Probability):**
- **Mitigation**: `cargo audit` in CI
- **Detection**: Weekly dependency vulnerability reports
- **Response**: Immediate updates for critical vulnerabilities

**Platform Compatibility (Medium Impact, Low Probability):**
- **Mitigation**: Multi-platform CI testing (Linux, macOS; Windows planned)
- **Detection**: Community issue reporting and automated testing
- **Response**: Platform-specific fixes with regression test coverage

---

## Verification Appendix (MVP)

### Code quality

- `cargo clippy --all-targets --all-features -- -D warnings` -> exit 0
- `rg -n "TODO|unwrap\(" -- !**/tests/**` -> no matches (or documented exceptions)
- `rg -n "unsafe fn|unsafe\s*\{"` -> each site links to safety doc section

**Status: Partially done.** Clippy is green in core crates. Unsafe documentation
is ongoing. Some TODOs remain in xtask and scaffolded GPU backends.

### Tokenizer + fixture

- SPM fixture tests: skip without model file; pass with fixture.
- Real tokenizer contract tests wired to `BITNET_GGUF` env var.

**Status: Done.** EnvGuard and serial patterns ensure isolation.

### Real inference smoke

- `cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run --model models/model.gguf --prompt "Hello" --max-tokens 8` -> non-mock path executes and returns tokens

**Status: Done.** CPU golden path E2E tests pass deterministically (seed=42,
pinned tokens [140,459,459,459]).

### Benchmarks + gating

- `cargo bench --bench kernel_benchmarks --features cpu` -> Criterion results logged
- `cargo run -p xtask -- bench-compare --baseline benchmarks/baseline/inference.json --current ci/inference.json` -> exit 0

**Status: Partially done.** Criterion benchmarks in place. `bench-compare` xtask
command exists. Automated CI gating on benchmark regression is planned.

### Receipt verification

- `cargo run -p xtask -- verify-receipt` -> exit 0
- Receipt schema v1.0.0 with 8 gates: 25/25 tests passing

**Status: Done.**

### Release sanity

- `cargo run -p xtask -- verify` -> runs model inspect, inference smoke, license check

**Status: Partial.** xtask verify runs; full hermetic fresh-machine test not yet
systematically validated. Nix flake (`nix flake check`) provides hermetic validation.

---

## Monitoring & Success Tracking

### Key Performance Indicators (KPIs)

**Technical Health:**

- Build success rate across platforms (target: >95%; CI covers Linux CPU always)
- Test pass rate (non-ignored): tracked per PR in CI
- QK256 throughput: measured by Criterion; current baseline ~0.1 tok/s (scalar)
- Fuzz target coverage: 49 targets, 60s nightly; crash artifacts uploaded
- Receipt gate pass rate: 25/25 (schema v1.0.0)

**Community Health:**

- Issue response time: aim for reasonable cadence; no formal SLA in pre-alpha
- Documentation completeness: ongoing; see `docs/` directory

### Reporting & Review Process

**Per-PR:**
- CI must pass (38 workflows); receipt gates enforced
- No bare `#[ignore]` without justification (pre-commit hook)
- Clippy clean in changed crates

**Milestone Reviews:**
- Phase 1 checkpoint: QK256 >=3x uplift + CUDA receipt end-to-end
- Phase 2 checkpoint: At least one non-CPU GPU backend validated
- Phase 3 checkpoint: Semantic versioning policy enforced; API surface audited

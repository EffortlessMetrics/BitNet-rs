# CI Labels Reference

BitNet-rs uses GitHub labels to selectively trigger heavy CI workflows. This keeps fast feedback for most PRs while ensuring comprehensive validation when needed.

## Label-Gated Workflows

These workflows only run when their corresponding label is applied to a PR:

### `coverage`

**Triggers**: Code coverage analysis
**When to use**: For changes to core inference, quantization, or kernel code
**CI Impact**: ~5-10 minutes; generates coverage reports with lcov/grcov

**Examples**:
- Changes to `bitnet-inference` or `bitnet-quantization`
- New kernel implementations in `bitnet-kernels`
- Major refactoring of core logic

**Note**: Coverage on `main` runs on every push with strict thresholds (≥70% required).

---

### `receipts`

**Triggers**: CPU receipt verification gates (Model Gates workflow)
**When to use**: For changes affecting inference compute paths or quantization algorithms
**CI Impact**: ~3-5 minutes; validates receipt schema and honest compute requirements

**Examples**:
- Changes to inference engine (`bitnet-inference`)
- Quantization algorithm modifications
- Receipt generation or validation logic

**Validation Gates**:
- `compute_path` must be `"real"` (not `"mocked"`)
- Non-empty `kernels` array with valid kernel IDs
- Schema v1.0.0 compliance
- Kernel ID hygiene (length ≤128, count ≤10K)

**Note**: On `main` and on **scheduled** nightly runs, receipt gates run strictly (blocking merge).

---

### `framework`

**Triggers**: Full integration test suite
**When to use**: For major architectural changes or cross-cutting concerns
**CI Impact**: ~10-15 minutes; runs complete test matrix across all crates

**Examples**:
- Changes to workspace structure
- FFI bridge modifications
- End-to-end integration scenarios
- Major refactoring affecting multiple crates

---

### `gpu`

**Triggers**: GPU-specific tests with CUDA runtime
**When to use**: For changes to GPU kernels, CUDA code, or device selection logic
**CI Impact**: ~8-12 minutes; requires self-hosted GPU runner (if available)

**Examples**:
- CUDA kernel implementations
- GPU quantization/dequantization
- Device detection or fallback logic
- Mixed precision (FP16/BF16) operations

**Note**: Requires CUDA 12.0+ toolkit and compatible GPU hardware.

---

### `quant`

**Triggers**: Quantization matrix testing (multiple quantization formats)
**When to use**: For changes to quantization algorithms or multi-format support
**CI Impact**: ~6-10 minutes; validates I2_S, TL1, TL2, QK256 formats

**Examples**:
- New quantization format implementations
- Changes to quantization kernels
- Flavor detection logic
- Cross-validation with C++ reference

---

### `crossval`

**Triggers**: Cross-validation determinism checks against C++ reference
**When to use**: For changes affecting inference results or numerical parity
**CI Impact**: ~8-12 minutes; requires BitNet.cpp setup and per-token validation

**Examples**:
- Inference engine modifications
- Quantization algorithm changes
- Numerical accuracy concerns
- Token generation logic

**Requirements**:
- `BITNET_CPP_DIR` configured
- C++ reference libraries available
- Deterministic inference enabled

**Validation**:
- Cosine similarity ≥0.999 threshold
- Per-token logits comparison
- Rust vs C++ parity

---

### `perf`

**Triggers**: Performance regression gates
**When to use**: For changes that could impact throughput or latency
**CI Impact**: ~10-15 minutes; runs benchmarks and compares against baselines

**Examples**:
- Kernel optimizations
- SIMD intrinsics
- Memory layout changes
- Critical path modifications

**Baselines**: Stored in `docs/baselines/` with datestamped filenames.

---

### `lut`

**Triggers**: TL-LUT (table lookup) stress testing
**When to use**: For changes to TL1/TL2 quantization or lookup table logic
**CI Impact**: ~5-8 minutes; validates lookup table implementations

**Examples**:
- TL1/TL2 quantization changes
- Lookup table generation
- Device-aware LUT selection (ARM NEON / x86 AVX)

---

## Always-Run Workflows

These workflows run on every PR and on `main` (Linux runners only):

### Core CI (`ci-core.yml`)

**Required checks**:
1. **Build & Test (ubuntu-latest)** - Workspace compilation and test suite
2. **Clippy** - Lint checks with `RUSTFLAGS=-Dwarnings`
3. **Documentation** - Documentation build (`cargo doc --no-deps`)
4. **CI Core Success** - Status rollup for branch protection

**Runs on**: All PRs and on `main`
**Timeout**: 30 minutes global
**Features**: Uses `--locked` for determinism, runs with `cpu` features by default
**Notes**: All jobs run on Linux (ubuntu-22.04). Clippy job aligns environment with Build & Test job for consistent warning detection.

---

### Link Checking (`link-check.yml`)

**Triggers**: Weekly schedule, manually dispatchable
**Tool**: `lychee` for broken link detection
**Scope**: Markdown files in docs/, README.md, CLAUDE.md

---

## Informational Workflows

These workflows provide feedback but don't block merges:

### Security checks

**Security & license audit**: PR-time audit runs as **Security and License Audit** job in `.github/workflows/compatibility.yml` (informational on PRs). Nightly/`main` security scan runs as **Security check** in `.github/workflows/security.yml` (required on `main`).

**Security Guard**: `Security Guard` job in `.github/workflows/validation.yml` flags banned correction flags (currently informational; will be made required in v0.2).

---

### Typos (`ci-typos.yml`)

**Triggers**: All PRs
**Purpose**: Spell checking with `typos` tool
**Current**: Informational only

---

## Label Usage Guidelines

### For Contributors

**Default behavior**: Don't add any labels unless you're changing core paths.

**Add labels proactively if**:
- You're modifying inference, quantization, or kernels → `receipts`, `quant`
- You're touching GPU code → `gpu`
- You want coverage feedback → `coverage`
- You're concerned about performance → `perf`

**Ask maintainers to add labels if**:
- You're unsure which labels apply
- You want comprehensive validation before merge
- You're making architectural changes → `framework`

### For Maintainers

**Label PRs based on changed paths**:

```bash
# Inference/quantization changes
gh pr edit <PR> --add-label receipts,quant

# GPU kernel work
gh pr edit <PR> --add-label gpu

# Cross-cutting architectural changes
gh pr edit <PR> --add-label framework,coverage

# Performance optimization
gh pr edit <PR> --add-label perf

# TL-LUT quantization
gh pr edit <PR> --add-label lut
```

**Require heavy checks before merge**:
- Apply relevant labels
- Wait for all label-gated checks to complete
- Review generated artifacts (receipts, coverage reports, benchmarks)

---

## Label Matrix

| Label | Workflow | Duration | Blocking on `main` | Blocking on PRs |
|-------|----------|----------|-------------------|-----------------|
| (none) | core-ci | ~5min | ✅ Yes | ✅ Yes |
| `coverage` | code-coverage | ~8min | ✅ Yes (≥70%) | ⚠️ Informational |
| `receipts` | model-gates | ~5min | ✅ Yes (strict) | ⚠️ Label-gated |
| `framework` | integration | ~12min | ❌ No | ⚠️ Label-gated |
| `gpu` | gpu-tests | ~10min | ❌ No | ⚠️ Label-gated |
| `quant` | quant-matrix | ~8min | ❌ No | ⚠️ Label-gated |
| `crossval` | crossval | ~10min | ❌ No | ⚠️ Label-gated |
| `perf` | perf-gates | ~12min | ❌ No | ⚠️ Label-gated |
| `lut` | lut-stress | ~6min | ❌ No | ⚠️ Label-gated |

---

## Quick Reference

**Fast PR (no labels needed)**:
- Docs changes
- CLI improvements
- Test additions
- Refactoring without numerical changes

**Heavy PR (add labels)**:
- Inference engine → `receipts`, `crossval`
- Quantization → `quant`, `receipts`
- GPU kernels → `gpu`
- Performance → `perf`
- Architecture → `framework`, `coverage`

**When in doubt**: Ask maintainers or add conservative labels (`coverage`, `framework`).

---

## See also

- [PR template](../../.github/pull_request_template.md) - Quick label checklist
- [CI workflows](../../.github/workflows/) – Workflow files
- [Branch protection](https://github.com/EffortlessMetrics/BitNet-rs/settings/branches) *(maintainers only)*

<\!-- ci: trigger core checks for PR template change -->

<\!-- ci: trigger core checks for dependabot.yml change -->

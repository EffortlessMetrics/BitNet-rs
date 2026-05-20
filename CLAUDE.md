# CLAUDE.md

Essential guidance for working with the bitnet-rs codebase.

## Repo Source-Of-Truth Stack

BitNet-rs uses this linked source-of-truth stack:

```text
Roadmap → Proposal → Spec → ADR → Plan → Active goal → PR → Proof
```

Before making changes, read:

1. `docs/reference/SPEC_SYSTEM.md`;
2. `.bitnet-rs/goals/active.toml` when present, otherwise the campaign
   `active.toml` explicitly named by the task;
3. the linked implementation plan;
4. the linked spec for the selected work item;
5. any linked ADRs.

Work on exactly one ready work item at a time. Do not create a new lane, mix
proposal/spec/ADR/plan/runtime changes, broaden support claims, or hand-edit
generated status unless the selected work item explicitly requires it. A change
is ready only when the intended artifact exists, linked docs are updated, proof
commands have run or are honestly marked unavailable, claim boundaries are
respected, and `git diff --check` passes.

Stop and report instead of guessing when the active goal is missing or stale,
linked specs are missing, proof commands cannot run, generated status differs
from committed status, requested work conflicts with an ADR, or unrelated staged
changes exist.

## Project Identity

- **Name:** bitnet-rs — 1-bit LLM inference engine in Rust
- **Version:** v0.2.1-dev (pre-alpha)
- **MSRV:** 1.93.0 (Rust 2024 edition, pinned in `rust-toolchain.toml`)
- **Status:** CPU inference works with SIMD optimization. GPU backends are scaffolded but not validated. Do not use in production.

## Rust 1.95 Rollout Rails

The Rust 1.95 / next minor rollout is a continuation of the Rust 1.93 CI
economics control plane, not a new rollout from scratch. Start each rollout PR
from clean `origin/main`, keep one PR per objective, open draft PRs first, and
do not combine MSRV bump, lint activation, no-panic baseline, release bump, or
API cleanup.

Critical doctrine:

```text
ripr is static mutation-exposure analysis.

It catches much of the same signal mutation testing catches -- weak test/oracle
exposure -- but earlier and cheaper, because it runs statically and can run
per PR.

Mutation testing remains the runtime empirical backstop, especially for
nightly and release readiness. The CI design should use ripr to shift
mutation signal left, not to pretend mutation is unnecessary.
```

First PR is documentation-only:

```text
docs/rust-1.95-rollout-refresh
docs(policy): refresh Rust 1.95 and next-minor rollout map
```

Roadmap and acceptance gates live in
`docs/development/RUST_1_95_ROLLOUT.md`.

## Build and Test

Default features are **empty** — always specify `--no-default-features --features cpu` or `gpu`.

```bash
# Build
cargo build --locked --no-default-features --features cpu
cargo build --locked --no-default-features --features gpu

# Optimised release
RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=thin" \
  cargo build --locked --release --no-default-features --features cpu,full-cli

# Test (nextest recommended — 5-min timeout prevents hangs)
cargo nextest run --locked --workspace --no-default-features --features cpu
cargo nextest run --locked --profile ci   # 4 threads, no retries

# Quality
cargo fmt --all && cargo clippy --locked --all-targets --no-default-features --features cpu -- -D warnings

# Quick inference check
RUST_LOG=warn cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "What is 2+2?" --max-tokens 8

# Nix (reproducible builds)
nix develop && nix build .#bitnet-cli && nix flake check
```

See: [Build reference](docs/development/build-commands.md) | [Nix guide](docs/kv-pool/NIX_FLAKE_USAGE.md)

## Architecture

```
bitnet-tokenizers ──────────────────────────────────────┐
                                                         │
bitnet-models  (GGUF loader, dual I2_S flavor detection) │
  └── bitnet-quantization  (I2_S / TL1 / TL2 / IQ2_S)  │
        └── bitnet-kernels (AVX2 / AVX-512 / NEON / CUDA)│
                                                         ▼
                        bitnet-inference  (autoregressive engine)
                          ├── bitnet-logits       (temperature / top-k / top-p)
                          ├── bitnet-sampling     (greedy, nucleus, repetition penalty)
                          ├── bitnet-generation   (decode loop, stop criteria)
                          ├── bitnet-prompt-templates  (59+ template variants)
                          └── bitnet-receipts     (honest-compute receipt schema)
                                                         │
                                          ┌──────────────┴──────────────┐
                                     bitnet-cli                  bitnet-server
```

**Scale:** ~200 workspace crates, 2,500+ .rs files, 129 crate dirs under `crates/`.

**Key crates:** `bitnet` (root), `bitnet-inference`, `bitnet-quantization`, `bitnet-kernels`, `bitnet-models`, `bitnet-tokenizers`, `bitnet-st2gguf`, `bitnet-cli`, `crossval`. Plus 48+ SRP microcrates (`bitnet-logits`, `bitnet-gguf`, `bitnet-generation`, `bitnet-device-probe`, etc.).

**GPU scaffold:** `bitnet-gpu-hal`, `bitnet-opencl`, `bitnet-vulkan`, `bitnet-wgpu`, `bitnet-rocm`, `bitnet-metal` — all feature-gated, not validated end-to-end.

**Quantization formats:** I2_S BitNet32-F16 (primary path), I2_S QK256/GGML (MVP scalar, ~0.1 tok/s), TL1, TL2, IQ2_S via FFI. QK256 priority in flavor detection.

## Feature Flags

| Flag | Purpose |
|------|---------|
| `cpu` | SIMD-optimised CPU inference (AVX2/AVX-512/NEON) |
| `gpu` | GPU umbrella — CUDA backend (requires CUDA 12.x) |
| `cuda` | Backward-compat alias for `gpu` |
| `full-cli` | Enable all CLI subcommands |
| `ffi` | C++ FFI bridge for cross-validation |
| `fixtures` | GGUF fixture-based integration tests (test-only) |
| `crossval-all` | All cross-validation features (`inference` + `crossval` + `ffi`) |

Always use the unified GPU predicate:
```rust
#[cfg(any(feature = "gpu", feature = "cuda"))]
fn gpu_only_function() { /* ... */ }
```

## Patterns and Conventions

### Test patterns

- `#[ignore = "reason"]` — all ignored tests have justification (enforced by pre-commit hook)
- `#[serial(bitnet_env)]` — required for tests mutating environment variables
- `EnvGuard` — RAII guard for env var isolation (`tests::helpers::env_guard::EnvGuard`)
- TDD scaffolds use `panic!("not yet implemented")` inside `#[ignore]` — this is intentional
- ~58,700 test annotations; ~2,800 intentionally ignored (TDD scaffolds, resource-gated, slow, CUDA, crossval)

### Optional-dependency test targets

Any `[[test]]`, `[[bench]]`, or `[[example]]` that imports an optional crate (e.g. `wgpu`,
`opencl3`, `pollster`, `cuda-*`) **must** declare `required-features` in `Cargo.toml`:

```toml
[[test]]
name = "metal_device_integration_tests"
path = "tests/metal_device_integration_tests.rs"
required-features = ["metal-runtime"]

[[test]]
name = "metal_compute_pipeline_tests"
path = "tests/metal_compute_pipeline_tests.rs"
required-features = ["metal-runtime", "cpu"]   # match the file-level #![cfg(...)]
```

Without this gate, CPU-only and no-features builds attempt to compile the test and fail with
unresolved-crate errors. Check the file-level `#![cfg(...)]` to confirm which features are needed.

### Platform-specific dead code

Use `cfg_attr` rather than a blanket `#[allow(dead_code)]` when a struct or function is only used
on a specific target:

```rust
// Only suppress the lint on non-aarch64 targets where NEON methods are absent
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
pub struct PagedCacheBlock { ... }
```

Do not use `#[allow(dead_code)]` without a `cfg_attr` scope — prefer the cfg guard or remove the
dead code instead.

### Feature gates

```rust
// GPU code uses the unified predicate
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn gpu_function() { /* ... */ }

// Runtime checks
if bitnet_kernels::device_features::gpu_compiled() {
    // GPU support was compiled in
}
if bitnet_kernels::device_features::gpu_available_runtime() {
    // GPU hardware is available
}
```

### Key environment variables

| Variable | Purpose |
|----------|---------|
| `BITNET_GGUF` | Model path override (auto-discovers `models/` if unset) |
| `BITNET_DETERMINISTIC=1` | Enable deterministic mode |
| `BITNET_STRICT_MODE=1` | Fail on LayerNorm/projection warnings (exit code 8) |
| `BITNET_SKIP_SLOW_TESTS=1` | Skip slow QK256 tests |
| `BITNET_CPP_DIR` | Path to bitnet.cpp for cross-validation |

Full list: [docs/environment-variables.md](docs/environment-variables.md)

### Commit and PR conventions

- Conventional commits: `feat:`, `fix:`, `perf:`, `docs:`, `refactor:`, `test:`
- All cargo/cross commands in CI use `--locked` (enforced by Guards gate)
- GitHub Actions must be SHA-pinned (no floating tags)
- Run `make guards` before push to catch CI blockers locally

### Campaign Agent Authority

Campaign work items are authoritative for review and merge flow. For items with
`review_mode = "codex_premerge"`,
`merge_policy = "automerge_when_green"`, and
`human_gate = "on_blocker_only"`, Codex agents are expected to edit, validate,
commit, push, open or update the PR, refresh the agent-owned PR branch when
needed through merge-from-main, rebase, `gh pr update-branch`, or
`--force-with-lease` after branch/status/diff inspection, address
CI/bot/reviewer feedback, merge when GitHub reports the PR green and mergeable,
and close out tracker PRs when required. Commit, push, PR creation, agent-owned
PR branch refresh, CI/bot/reviewer repair, merge, and tracker closeout are not
human approval gates.

Human involvement is required only for true blockers: permissions or branch
protection prevent the merge, direct mutation of `origin/main`, destructive
cleanup, or secret/model-binary exposure is possible, kernel/math/tokenizer/
loader semantics are in unresolved conflict, acceptance criteria conflict with
repository policy, or a cost/exposure/release decision is outside the ticket
scope.

This policy supersedes older agent runbook wording that treats ordinary commit,
push, PR creation, PR branch refresh, CI repair, merge, or tracker closeout as a
human approval boundary for `codex_premerge` plus `automerge_when_green` plus
`on_blocker_only` work items.

## Critical Gotchas

1. **Empty default features** — `cargo build` alone fails. Always pass `--no-default-features --features cpu|gpu`.

2. **TDD scaffolds aren't bugs** — `panic!()` inside `#[ignore = "TDD scaffold: ..."]` tests is intentional. Check the justification string.

3. **Model quality != inference bugs** — microsoft-bitnet-b1.58-2B-4T produces garbled output in some configs. This is a known model limitation.

4. **QK256 is slow** — Scalar kernels only (~0.1 tok/s for 2B). Use `--max-tokens 4-16` for validation. SIMD optimization is planned.

5. **FFI linker errors** — Use `--no-default-features --features cpu` to avoid FFI. For cross-validation: `cargo run --no-default-features --locked -p xtask -- fetch-cpp`.

## Repository Contracts

- Always specify features: `--no-default-features --features cpu|gpu`
- Use xtask for operations: `cargo run --no-default-features --locked -p xtask --`
- Never modify GGUF in-place: use `bitnet-compat export-fixed`
- Use `#[serial(bitnet_env)]` for env-mutating tests
- Check `#[ignore = "..."]` justification before investigating test failures

## Key Documentation

| Topic | Location |
|-------|----------|
| Quick start | [docs/quickstart.md](docs/quickstart.md) |
| Build reference | [docs/development/build-commands.md](docs/development/build-commands.md) |
| Test suite | [docs/development/test-suite.md](docs/development/test-suite.md) |
| Feature flags | [docs/explanation/FEATURES.md](docs/explanation/FEATURES.md) |
| Environment variables | [docs/environment-variables.md](docs/environment-variables.md) |
| Architecture | [docs/architecture-overview.md](docs/architecture-overview.md) |
| Inference CLI | [docs/reference/inference-cli-reference.md](docs/reference/inference-cli-reference.md) |
| Cross-validation CLI | [docs/reference/crossval-cli-reference.md](docs/reference/crossval-cli-reference.md) |
| Quantization | [docs/reference/quantization-support.md](docs/reference/quantization-support.md) |
| Validation gates | [docs/reference/validation-gates.md](docs/reference/validation-gates.md) |
| GPU setup | [docs/GPU_SETUP.md](docs/GPU_SETUP.md) |
| C++ cross-validation | [docs/howto/cpp-setup.md](docs/howto/cpp-setup.md) |
| Model validation | [docs/howto/validate-models.md](docs/howto/validate-models.md) |
| QK256 usage | [docs/howto/use-qk256-models.md](docs/howto/use-qk256-models.md) |
| Roadmap | [ROADMAP.md](ROADMAP.md) |
| Nix flake | [docs/kv-pool/NIX_FLAKE_USAGE.md](docs/kv-pool/NIX_FLAKE_USAGE.md) |

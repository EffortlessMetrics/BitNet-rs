# CLAUDE.md

Essential guidance for working with the bitnet-rs codebase.

## Project Identity

- **Name:** bitnet-rs — 1-bit LLM inference engine in Rust
- **Version:** v0.2.1-dev (pre-alpha)
- **MSRV:** 1.92.0 (Rust 2024 edition, pinned in `rust-toolchain.toml`)
- **Status:** CPU inference works with SIMD optimization. GPU backends are scaffolded but not validated. Do not use in production.

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

## Critical Gotchas

1. **Empty default features** — `cargo build` alone fails. Always pass `--no-default-features --features cpu|gpu`.

2. **TDD scaffolds aren't bugs** — `panic!()` inside `#[ignore = "TDD scaffold: ..."]` tests is intentional. Check the justification string.

3. **Model quality != inference bugs** — microsoft-bitnet-b1.58-2B-4T produces garbled output in some configs. This is a known model limitation.

4. **QK256 is slow** — Scalar kernels only (~0.1 tok/s for 2B). Use `--max-tokens 4-16` for validation. SIMD optimization is planned.

5. **FFI linker errors** — Use `--no-default-features --features cpu` to avoid FFI. For cross-validation: `cargo run --locked --no-default-features -p xtask -- fetch-cpp`.

## Repository Contracts

- Always specify features: `--no-default-features --features cpu|gpu`
- Use xtask for operations: `cargo run --locked --no-default-features -p xtask --` (xtask's default features pull in `gpu`; pass `--features gpu` explicitly when you want it)
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

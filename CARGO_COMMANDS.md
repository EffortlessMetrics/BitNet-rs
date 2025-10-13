# Cargo Commands Reference

BitNet-rs uses **cargo as the source of truth**. All operations are pure cargo commands.

## Essential Commands

### CPU Development (Default)

```bash
# Build with CPU features
cargo build --locked --workspace --no-default-features --features cpu

# Run tests
cargo test --locked --workspace --no-default-features --features cpu

# Run benchmarks
cargo bench --workspace --no-default-features --features cpu

# Check code quality
cargo clippy --workspace --all-targets --no-default-features --features cpu -- -D warnings
cargo fmt --all -- --check
```

### GPU Development

```bash
# Detect GPU hardware
cargo xtask gpu-preflight

# Quick smoke test (CPU↔GPU parity)
cargo xtask gpu-smoke

# Build with GPU support
cargo build --locked --workspace --no-default-features --features gpu

# Run GPU tests
cargo test --locked --workspace --no-default-features --features gpu
```

### Utilities (via cargo xtask)

```bash
# Download models
cargo xtask download-model

# Run demos
cargo xtask demo --which all

# Cross-validation
cargo xtask full-crossval

# Generate test fixtures
cargo xtask gen-fixtures

# Clean caches
cargo xtask clean-cache
```

## Cargo Aliases

Pre-configured in `.cargo/config.toml`:

```bash
cargo xtask           # Alias for: cargo run -p xtask --
cargo tw              # Test workspace with CPU
cargo gpu-tests       # Test workspace with GPU
cargo gpu-smoke       # Quick GPU smoke test
cargo gpu-build       # Build with GPU features
```

## Environment Variables

```bash
# Control parallelism
RAYON_NUM_THREADS=1 cargo test    # Single-threaded for determinism

# GPU selection
CUDA_VISIBLE_DEVICES=0 cargo test --features gpu

# Logging
RUST_LOG=debug cargo run

# Backtrace
RUST_BACKTRACE=1 cargo test
```

## Docker (still uses cargo internally)

```bash
# Build images
export DOCKER_BUILDKIT=1
docker compose build bitnet-cpu
docker compose build bitnet-gpu

# Run services
docker compose up bitnet-cpu
docker compose --profile gpu up bitnet-gpu
```

## CI/CD

GitHub Actions uses these exact commands:

```yaml
# CPU path
- run: cargo build --locked --workspace --no-default-features --features cpu
- run: cargo test --locked --workspace --no-default-features --features cpu

# GPU path (on self-hosted runner)
- run: cargo xtask gpu-preflight
- run: cargo build --locked --workspace --no-default-features --features gpu
- run: cargo xtask gpu-smoke
```

## Why Cargo-First?

1. **Single source of truth** - No Makefile/script divergence
2. **Reproducible** - `--locked` ensures exact dependencies
3. **Discoverable** - `cargo --list` shows all commands
4. **Portable** - Works on any OS with Rust installed
5. **Native** - Leverages Rust ecosystem tooling

## Optional Makefile

If you prefer Make shortcuts, use `Makefile.minimal`:

```bash
make build    # → cargo build --locked --workspace --no-default-features --features cpu
make test     # → cargo test --locked --workspace --no-default-features --features cpu
make gpu      # → cargo xtask gpu-preflight
```

But remember: **cargo is the source of truth**.

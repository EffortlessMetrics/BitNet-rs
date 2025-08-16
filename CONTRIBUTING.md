# Contributing to BitNet.rs

Thank you for your interest in contributing to BitNet.rs! This document provides guidelines and best practices for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/BitNet-rs.git`
3. Create a feature branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests: `cargo test --workspace`
6. Submit a pull request

## Development Setup

### Prerequisites

- Rust 1.70.0 or later
- Git
- Optional: CUDA toolkit for GPU features
- Optional: Python 3.8+ for Python bindings

### Building

```bash
# Build with default features (CPU only)
cargo build --release

# Build with GPU support
cargo build --release --features gpu

# Build with all features
cargo build --release --features full
```

## Code Style

- Use `cargo fmt` to format your code
- Run `cargo clippy` and address all warnings
- Follow Rust naming conventions
- Add documentation comments for public APIs
- Write unit tests for new functionality

## Testing

```bash
# Run all tests
cargo test --workspace

# Run tests with specific features
cargo test --workspace --features cpu

# Run benchmarks
cargo bench --workspace

# Run cross-validation tests (if C++ deps available)
cargo test --workspace --features crossval
```

## Configuration & testing conventions

- **Do not** apply fast‑feedback/resource/quality clamps inside `ScenarioConfigManager`.
  The test wrapper owns this logic to prevent double application.
- Use the shared `ENV_LOCK` + `env_guard()` in any test touching `std::env`.
- Use `BYTES_PER_KB`, `BYTES_PER_MB`, or `BYTES_PER_GB` for all unit→bytes conversions.
  (These are **binary** units: 1 KB = 1024 B, 1 MB = 1024 KB, 1 GB = 1024 MB.)
- Import units via `bitnet_tests::units::{BYTES_PER_KB, BYTES_PER_MB, BYTES_PER_GB}`; 
  avoid `bitnet_tests::common::units::*`.
- Prefer typed env helpers (`env_u64`, `env_usize`, `env_duration_secs`) and a
  case‑insensitive `env_bool`.
- When adding new cross‑validation thresholds, clamp to **non‑negative** values.
- If you add a clamp, add a **regression test** that fails if it's applied twice.

## Pull Request Process

1. Update documentation if you change APIs
2. Add tests for new functionality
3. Ensure all tests pass
4. Update the changelog if applicable
5. Request review from maintainers

## Reporting Issues

When reporting issues, please include:
- BitNet.rs version
- Rust version (`rustc --version`)
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages or logs

## License

By contributing, you agree that your contributions will be licensed under the same terms as the project (MIT OR Apache-2.0).
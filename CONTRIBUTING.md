# Contributing to BitNet.rs

Thank you for your interest in contributing to BitNet.rs! This document provides guidelines and best practices for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/BitNet-rs.git`
3. Create a feature branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests: `cargo test --workspace`
6. Submit a pull request

### Getting Started for New Contributors

We warmly welcome new contributors! If you're looking for a way to get involved, here are some great places to start:

**1. "Good First Issues"**

We use the `good first issue` label on GitHub to mark issues that are well-suited for newcomers. These issues are typically:
*   Well-defined with a clear scope.
*   Have a low to moderate difficulty level.
*   Provide a great introduction to the codebase.

You can find them here: [https://github.com/microsoft/BitNet/labels/good%20first%20issue](https://github.com/microsoft/BitNet/labels/good%20first%20issue)

**2. Documentation**

Improving documentation is one of the most valuable contributions you can make. If you find a section in our `README.md`, `docs/`, or code comments that is unclear, confusing, or missing information, please open an issue or submit a pull request! This includes:
*   Fixing typos and grammatical errors.
*   Adding more detailed explanations.
*   Creating new examples to clarify usage.

**3. Adding Tests**

We strive for high test coverage. A great way to contribute is to add unit or integration tests for parts of the code that are not well-covered. You can run `cargo cov-html` (see README) to generate a coverage report and find areas for improvement.

**A Typical First Pull Request Workflow**

1.  Find an issue you'd like to work on (or create one if you've found a bug or have an idea for an improvement).
2.  Leave a comment on the issue to let others know you're working on it.
3.  Fork the repository and create a new branch for your changes.
4.  Make your code changes. Remember to add tests and documentation where appropriate!
5.  Run `cargo fmt` to format your code and `cargo clippy` to check for lints.
6.  Run the test suite with `cargo test --workspace` to ensure your changes haven't broken anything.
7.  Commit your changes with a descriptive message.
8.  Push your branch to your fork and open a pull request against the `main` branch of the `microsoft/BitNet` repository.

We're excited to see your contributions!

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
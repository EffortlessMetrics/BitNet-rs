# Contributing to BitNet.rs

Welcome to BitNet.rs! We appreciate your interest in contributing to our high-performance 1-bit neural network quantization and inference library for Rust.

## Quick Start for Contributors

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/BitNet-rs.git
   cd BitNet-rs
   ```

2. **Setup Development Environment**
   ```bash
   # Install Rust 1.90.0 or later
   rustup update stable

   # Install development tools
   cargo install cargo-nextest cargo-mutants
   ```

3. **Run Tests**
   ```bash
   # Quick test with CPU features
   cargo test --workspace --no-default-features --features cpu

   # Full test suite
   ./scripts/test-all.sh
   ```

## Development Workflow

### Feature Development

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Follow TDD Approach**
   - Write tests first
   - Implement minimal code to pass tests
   - Refactor with safety

3. **Use xtask Commands**
   ```bash
   # Download test models
   cargo run -p xtask -- download-model

   # Verify implementation
   cargo run -p xtask -- verify --model models/test.gguf

   # Cross-validate against C++ reference
   cargo run -p xtask -- crossval
   ```

### Code Quality Standards

- **MSRV**: Minimum Rust 1.90.0 (2024 edition)
- **Features**: Always specify `--no-default-features --features cpu|gpu`
- **Safety**: Minimize `unsafe` code; document all usage
- **Performance**: Target >99% quantization accuracy
- **Testing**: Maintain 100% test coverage for critical paths

### Neural Network Specific Guidelines

- **Quantization**: Support I2S, TL1, TL2, and IQ2_S formats
- **GPU Support**: CUDA kernels with CPU fallback
- **GGUF Compatibility**: Maintain compatibility with upstream formats
- **Cross-validation**: All changes must pass C++ reference comparison

## Documentation Requirements

- **API Documentation**: All public APIs must have comprehensive rustdoc
- **Examples**: Include working examples for new features
- **Performance**: Document performance characteristics and benchmarks
- **Migration**: Update migration guides for breaking changes

## Testing Requirements

### Required Test Types

1. **Unit Tests**
   ```bash
   cargo test --workspace --no-default-features --features cpu
   ```

2. **Integration Tests**
   ```bash
   cargo test --test integration --no-default-features --features cpu
   ```

3. **Cross-validation Tests**
   ```bash
   export BITNET_GGUF="models/test.gguf"
   cargo test --package crossval --no-default-features --features cpu
   ```

4. **Property Tests**
   ```bash
   cargo test property_ --no-default-features --features cpu
   ```

5. **Mutation Tests** (CI only)
   ```bash
   cargo mutants --package bitnet-quantization
   ```

### GPU Testing

```bash
# Requires CUDA toolkit
cargo test --workspace --no-default-features --features gpu
```

## Pull Request Process

### Before Submitting

1. **Format and Lint**
   ```bash
   cargo fmt --all
   cargo clippy --all-targets --all-features -- -D warnings
   ```

2. **Run Full Test Suite**
   ```bash
   ./scripts/test-all.sh
   ```

3. **Update Documentation**
   ```bash
   cargo doc --workspace --no-default-features --features cpu --no-deps
   ```

4. **Cross-validate Changes**
   ```bash
   cargo run -p xtask -- full-crossval
   ```

### PR Requirements

- **Title**: Use conventional commits format (`feat:`, `fix:`, `docs:`)
- **Description**: Include what, why, and testing performed
- **Tests**: All new code must include tests
- **Documentation**: Update relevant documentation
- **Backwards Compatibility**: Document any breaking changes

### Review Process

1. **Automated Checks**: All CI checks must pass
2. **Code Review**: At least one maintainer approval required
3. **Cross-validation**: Must pass accuracy validation
4. **Performance**: No significant performance regressions

## Architecture Guidelines

### Crate Organization

- **`bitnet`**: Main library with unified public API
- **`bitnet-quantization`**: Quantization algorithms (I2S, TL1, TL2)
- **`bitnet-kernels`**: High-performance SIMD/CUDA kernels
- **`bitnet-inference`**: Inference engine with streaming
- **`bitnet-models`**: Model loading (GGUF, SafeTensors)
- **`bitnet-tokenizers`**: Universal tokenizer with GGUF integration

### Design Principles

1. **Zero-Copy**: Minimize allocations and copies
2. **Device-Aware**: Automatic GPU/CPU selection
3. **Type Safety**: Leverage Rust's type system for correctness
4. **Performance**: Target high-performance computing workloads
5. **Compatibility**: Maintain API stability and GGUF compatibility

## Getting Help

- **Documentation**: [docs/](docs/) directory with comprehensive guides
- **Examples**: [examples/](examples/) directory with working code
- **Issues**: GitHub Issues for bug reports and feature requests
- **Discussions**: GitHub Discussions for questions and ideas

## License

By contributing to BitNet.rs, you agree that your contributions will be licensed under the same terms as the project (MIT OR Apache-2.0).

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

---

Thank you for contributing to BitNet.rs! Your contributions help advance high-performance neural network inference in Rust.

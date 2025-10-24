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

### Working with Test Fixtures

BitNet.rs uses a **3-layer fixture architecture** for testing neural network operations, quantization algorithms, and model loading:

#### Fixture Patterns

1. **Inline Fixtures** — Hardcoded test data in the test file
   - **Use for**: Trivial constants, magic numbers, simple shapes
   - **Example**: `let vocab_size = 32000;`

2. **Generated Fixtures** — Programmatic test data from seed functions
   - **Use for**: Randomized tensors, quantization roundtrip tests, property-based testing
   - **Example**: `helpers::lcg_random(seed)` for deterministic random data
   - **Available generators**:
     - `lcg_random(seed)` — Deterministic LCG PRNG for numeric data
     - `generate_gguf_fixture(config)` — Minimal GGUF files for tokenizer/model tests
     - `create_test_gguf_with_i2s(name, shape, data, type)` — I2S quantized tensor fixtures

3. **File-Based Fixtures** — External test data loaded from disk
   - **Use for**: Full GGUF models, real tokenizer files, reference data
   - **Example**: `tests/fixtures/gguf/llama3-128k.gguf`

#### Using Generated Fixtures

**Deterministic Random Data (LCG PRNG):**
```rust
use helpers::lcg_random;

#[test]
fn test_quantization_roundtrip() {
    let seed = 42;
    let input: Vec<f32> = (0..1024).map(|_| lcg_random(seed) as f32).collect();

    // Quantize → Dequantize → Validate
    let quantized = quantize_i2s(&input);
    let dequantized = dequantize_i2s(&quantized);

    assert!(calculate_correlation(&input, &dequantized) > 0.99);
}
```

**GGUF Model Fixtures:**
```rust
use fixtures::gguf_fixtures::{generate_gguf_fixture, GgufFixtureConfig};

#[test]
fn test_tokenizer_auto_discovery() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("test.gguf");

    generate_gguf_fixture(&model_path, &GgufFixtureConfig {
        model_type: "llama".to_string(),
        vocab_size: 128256,
        has_embedded_tokenizer: true,
        tokenizer_type: Some("hf".to_string()),
        corrupted: false,
    }).unwrap();

    // Test tokenizer discovery logic
    let tokenizer = load_tokenizer(&model_path).unwrap();
    assert_eq!(tokenizer.vocab_size(), 128256);
}
```

**I2S Quantized Tensor Fixtures:**
```rust
#[test]
fn test_qk256_flavor_detection() {
    // Generate QK256 quantized tensor (256-elem blocks, separate scales)
    let rows = 4;
    let cols = 256;
    let data = vec![0u8; rows * cols / 4 + rows * 2]; // 2-bit packed + scales

    let file = create_test_gguf_with_i2s("test.weight", &[rows, cols], data, 26);

    // Validate flavor detection
    let result = load_gguf_full(file.path(), Device::Cpu).unwrap();
    assert!(result.i2s_qk256.contains_key("test.weight"));
}
```

#### Running Fixture Tests

```bash
# Run all tests (includes fixture-based tests)
cargo test --workspace --no-default-features --features cpu

# Run specific fixture tests
cargo test -p bitnet-models test_qk256_flavor_detection --no-default-features --features cpu
cargo test -p bitnet-tokenizers test_gguf_fixture --no-default-features --features cpu

# Skip slow fixture-heavy tests (e.g., full model loading)
BITNET_SKIP_SLOW_TESTS=1 cargo test --workspace --no-default-features --features cpu
```

#### Adding New Fixtures

**Step-by-step guide:**

1. **Identify fixture requirements** — What test data do you need?
   - Simple constants? → Use inline fixtures
   - Randomized tensors? → Use generated fixtures with seeded PRNG
   - Full models? → Use file-based fixtures or `generate_gguf_fixture`

2. **Choose appropriate layer**
   ```rust
   // Inline: trivial constants
   let embedding_dim = 512;

   // Generated: seeded random data
   let weights = helpers::generate_random_tensor(42, shape);

   // File-based: real model artifacts
   let model = load_gguf("tests/fixtures/gguf/llama3-128k.gguf");
   ```

3. **Add generator function if needed** (for new fixture types)
   ```rust
   // In crates/bitnet-models/tests/helpers/qk256_fixtures.rs
   pub fn generate_qk256_4x256(seed: u64) -> Vec<u8> {
       // Generate deterministic QK256 quantized tensor
       let mut rng = lcg_random(seed);
       let rows = 4;
       let cols = 256;
       let data_size = rows * cols / 4 + rows * 2; // 2-bit packed + scales
       (0..data_size).map(|_| (rng() % 256) as u8).collect()
   }
   ```

4. **Write test using fixture**
   ```rust
   #[test]
   fn test_new_quantization_format() {
       let data = helpers::generate_qk256_4x256(42);
       let quantized = QuantizedTensor::from_bytes(&data);
       assert_eq!(quantized.shape(), &[4, 256]);
   }
   ```

5. **Document fixture purpose** — Add rustdoc comments
   ```rust
   /// Generate QK256 quantized tensor with deterministic seed
   ///
   /// # Format
   /// - 4 rows × 256 columns (1024 elements)
   /// - 2-bit packed representation (256 bytes)
   /// - Separate FP16 scales (8 bytes)
   ///
   /// # Seed
   /// Use consistent seeds for reproducibility:
   /// - `42` — Default test seed
   /// - `1337` — Edge case seed (extreme values)
   pub fn generate_qk256_4x256(seed: u64) -> Vec<u8> { /* ... */ }
   ```

**Best practices:**
- Use **seeded generators** for reproducible randomness (never `rand::thread_rng()`)
- Keep inline fixtures **minimal** and well-commented
- Prefer **generated fixtures** over hardcoded binary blobs
- Use **file-based fixtures** sparingly (increases repo size)
- Document fixture **format** and **seed meanings** in rustdoc

**See also:**
- `tests/helpers/issue_261_test_helpers.rs` — Quantization accuracy helpers
- `crates/bitnet-tokenizers/tests/fixtures/gguf_fixtures.rs` — GGUF fixture generator
- `crates/bitnet-models/tests/qk256_dual_flavor_tests.rs` — I2S fixture examples

### GPU Testing

```bash
# Requires CUDA toolkit
cargo test --workspace --no-default-features --features gpu
```

### Environment Variable Testing

BitNet.rs uses environment variables for runtime configuration (e.g., `BITNET_STRICT_MODE`, `BITNET_DETERMINISTIC`, `BITNET_GGUF`). Tests that mutate environment variables must use **EnvGuard** and **serial execution** to prevent flaky tests and race conditions.

**Why This Matters**: Environment variables are process-global state. Without isolation, parallel tests can interfere with each other:
- Test A sets `BITNET_STRICT_MODE=1` → Test B unexpectedly sees strict mode enabled
- Test C clears `BITNET_GGUF` → Test D's model path is lost
- Non-deterministic failures occur depending on test execution order

**Using EnvGuard for Safe Environment Mutations:**

```rust
use tests::support::env_guard::EnvGuard;
use serial_test::serial;

#[test]
#[serial(bitnet_env)]  // Prevents parallel execution with other env-mutating tests
fn test_strict_mode_validation() {
    // EnvGuard automatically restores original env state on drop
    let _guard = EnvGuard::new("BITNET_STRICT_MODE", Some("1"));

    // Test code that depends on BITNET_STRICT_MODE=1
    let result = validate_model("models/test.gguf");
    assert!(result.is_err(), "Expected strict mode to fail on warnings");

    // _guard drops here → BITNET_STRICT_MODE restored to original value
}

#[test]
#[serial(bitnet_env)]  // Same serial group ensures sequential execution
fn test_model_path_override() {
    let _guard = EnvGuard::new("BITNET_GGUF", Some("/tmp/test.gguf"));

    // Test code that uses BITNET_GGUF env var
    assert_eq!(get_model_path(), Some("/tmp/test.gguf".into()));
}
```

**Best Practices:**
- **Always use `#[serial(bitnet_env)]`** on tests that call `EnvGuard::new()` or directly mutate environment variables
- **Group related env tests** under the same serial tag (e.g., `bitnet_env`) to prevent cross-contamination
- **Document env dependencies** in test rustdoc comments (e.g., `/// Requires BITNET_STRICT_MODE to be unset`)
- **Avoid direct `std::env::set_var`** — use `EnvGuard` instead for automatic cleanup
- **Run env tests sequentially** in CI to ensure reproducibility (nextest already handles this via `#[serial]`)

**See also:**
- `tests/support/env_guard.rs` — EnvGuard implementation
- `crates/bitnet-common/tests/issue_260_strict_mode_tests.rs` — EnvGuard usage examples
- `crates/bitnet-models/tests/loader_strict_mode.rs` — Serial env testing patterns

## Local Development Setup

### Enable Pre-commit Hooks (Recommended)

BitNet.rs provides Git hooks that enforce quality standards locally before commits reach CI:

```bash
# Enable pre-commit hooks
git config core.hooksPath .githooks
```

**What the hooks check:**
- ✅ **#[ignore] Annotation Hygiene**: Ensures all `#[ignore]` attributes include a reason
- ⚠️ **Environment Mutation Safety**: Warns about raw `std::env::set_var()` calls (should use EnvGuard)

**See**: `.githooks/README.md` for full documentation

### Why Use Pre-commit Hooks?

Pre-commit hooks catch issues early in your local workflow:
- **Faster feedback** than waiting for CI (seconds vs minutes)
- **Prevents accidental commits** of bare `#[ignore]` markers
- **Enforces EnvGuard pattern** for environment mutations
- **Saves CI resources** by catching issues before push

**Note**: You can temporarily bypass hooks with `git commit --no-verify`, but CI will still enforce these checks.

## Pull Request Process

### Before Submitting

1. **Enable Pre-commit Hooks** (first time only)
   ```bash
   git config core.hooksPath .githooks
   ```

2. **Run Local Quality Gates** (Recommended)
   ```bash
   # Comprehensive quality gates: fmt → clippy → tests → (bench) → verify-receipt
   ./scripts/local_gates.sh
   ```

   Or run individual checks:

3. **Format and Lint**
   ```bash
   cargo fmt --all
   cargo clippy --all-targets --all-features -- -D warnings
   ```

4. **Run Full Test Suite**
   ```bash
   ./scripts/test-all.sh
   ```

5. **Verify Inference Receipt** (if you have ci/inference.json)
   ```bash
   # Verify CPU receipt
   cargo run -p xtask -- verify-receipt --path ci/inference.json

   # Verify GPU receipt (requires GPU kernels)
   cargo run -p xtask -- verify-receipt --path ci/inference.json --require-gpu-kernels
   ```

6. **Update Documentation**
   ```bash
   cargo doc --workspace --no-default-features --features cpu --no-deps
   ```

7. **Cross-validate Changes** (optional, for inference changes)
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

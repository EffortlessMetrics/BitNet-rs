# Cross-Validation Setup Guide

This guide explains how to set up and use the cross-validation features in BitNet.rs to compare against the legacy C++ implementation.

## Overview

Cross-validation is **optional** and **feature-gated** to ensure zero overhead when not needed. It's primarily used for:

- **Development testing** - Ensuring BitNet.rs maintains numerical accuracy
- **Performance benchmarking** - Comparing speed and memory usage
- **Migration validation** - Verifying compatibility during migration
- **Research purposes** - Academic comparison studies

> **Note**: Cross-validation is **not required** for production use of BitNet.rs. It's a development and testing tool only.

## Feature Gate System

### Zero Overhead Design

Cross-validation is behind the `crossval` feature flag:

```toml
# Cargo.toml - crossval feature is NOT enabled by default
[features]
default = ["cpu"]  # No crossval by default
crossval = []      # Enable cross-validation features
```

**When crossval is disabled (default):**
- ✅ Zero compilation overhead
- ✅ No C++ dependencies required
- ✅ No additional binary size
- ✅ Fast builds and tests

**When crossval is enabled:**
- 🔧 C++ implementation downloaded and built
- 🔧 FFI bindings generated
- 🔧 Cross-validation tests available
- 🔧 Performance benchmarks enabled

### Enabling Cross-Validation

#### For Testing
```bash
# Run cross-validation tests
cargo test --features crossval

# Run performance benchmarks
cargo bench --features crossval

# Build with cross-validation support
cargo build --features crossval
```

#### For Development
```bash
# Add to your shell profile for persistent enabling
export CARGO_FEATURES="crossval"

# Or use cargo config
echo 'default-features = ["crossval"]' >> .cargo/config.toml
```

### Enabling the C++ FFI in Tests

The cross-validation harness uses stubbed bindings when the C++ library is
absent. To test against the real BitNet.cpp implementation, build the library
and enable the `cpp-ffi` feature:

```bash
# Build the C++ library
./ci/fetch_bitnet_cpp.sh

# Run tests with real FFI bindings
cargo test -p bitnet-tests --features crossval,cpp-ffi
```

Ensure the resulting shared library is on your linker path (e.g., `LD_LIBRARY_PATH`).

## Prerequisites

### System Dependencies

Cross-validation requires additional system dependencies:

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install clang libclang-dev cmake build-essential git
```

#### macOS
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install CMake (if not already installed)
brew install cmake
```

#### Windows
1. **Visual Studio**: Install Visual Studio 2019+ with C++ tools
2. **CMake**: Download from https://cmake.org/download/
3. **Git**: Download from https://git-scm.com/download/win
4. **LLVM/Clang**: Download from https://llvm.org/

### Rust Dependencies

Ensure you have a recent Rust version:

```bash
# Update Rust
rustup update stable

# Verify version (1.89+ required)
rustc --version
```

## Setup Process

### Automatic Setup (Recommended)

The easiest way to set up cross-validation:

```bash
# 1. Download and build C++ implementation
./ci/fetch_bitnet_cpp.sh

# 2. Set up environment
source ~/.cache/bitnet_cpp/setup_env.sh

# 3. Run cross-validation tests
cargo test --features crossval

# 4. Run performance benchmarks
cargo bench --features crossval
```

### Manual Setup (Advanced)

If you need custom configuration:

```bash
# 1. Set custom C++ path
export BITNET_CPP_PATH=/path/to/your/bitnet.cpp

# 2. Build the C++ implementation
cd $BITNET_CPP_PATH
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 3. Run cross-validation
cargo test --features crossval
```

### Verification

Verify your setup is working:

```bash
# Check C++ implementation availability
cargo test --features crossval cpp_availability

# Run a simple cross-validation test
cargo test --features crossval test_basic_generation

# Run performance comparison
cargo bench --features crossval comparison
```

## Usage Examples

### Basic Cross-Validation

```rust
#[cfg(feature = "crossval")]
use bitnet_crossval::{
    comparison::CrossValidator,
    CrossvalConfig,
};

#[cfg(feature = "crossval")]
fn run_cross_validation() -> Result<(), Box<dyn std::error::Error>> {
    let config = CrossvalConfig {
        tolerance: 1e-6,
        max_tokens: 100,
        benchmark: true,
    };
    
    let validator = CrossValidator::new(config);
    
    // Load a test fixture
    let fixture = bitnet_crossval::fixtures::TestFixture::load("minimal_test")?;
    
    // Run validation
    let results = validator.validate_fixture(&fixture)?;
    
    for result in results {
        if result.tokens_match {
            println!("✅ Tokens match for prompt: '{}'", result.prompt);
        } else {
            println!("❌ Token mismatch for prompt: '{}'", result.prompt);
        }
    }
    
    Ok(())
}

#[cfg(not(feature = "crossval"))]
fn run_cross_validation() -> Result<(), Box<dyn std::error::Error>> {
    println!("Cross-validation not available (compile with --features crossval)");
    Ok(())
}
```

### Performance Benchmarking

```rust
#[cfg(feature = "crossval")]
use criterion::{criterion_group, criterion_main, Criterion};

#[cfg(feature = "crossval")]
fn benchmark_inference(c: &mut Criterion) {
    use bitnet_crossval::cpp_bindings::CppModel;
    
    let model = CppModel::load("fixtures/test_model.gguf")
        .expect("Failed to load model");
    
    c.bench_function("cpp_inference", |b| {
        b.iter(|| {
            model.generate("Hello, world!", 50)
                .expect("Generation should succeed")
        });
    });
}

#[cfg(feature = "crossval")]
criterion_group!(benches, benchmark_inference);

#[cfg(feature = "crossval")]
criterion_main!(benches);

#[cfg(not(feature = "crossval"))]
fn main() {
    println!("Benchmarks require crossval feature");
}
```

### Conditional Compilation

```rust
// This code only compiles when crossval feature is enabled
#[cfg(feature = "crossval")]
mod cross_validation {
    use bitnet_crossval::*;
    
    pub fn validate_model(model_path: &str) -> Result<bool> {
        // Cross-validation logic here
        Ok(true)
    }
}

// This code compiles when crossval feature is disabled
#[cfg(not(feature = "crossval"))]
mod cross_validation {
    pub fn validate_model(_model_path: &str) -> Result<bool, &'static str> {
        Err("Cross-validation not available")
    }
}

// Usage that works regardless of feature
fn main() {
    match cross_validation::validate_model("test.gguf") {
        Ok(valid) => println!("Model validation: {}", valid),
        Err(e) => println!("Validation error: {}", e),
    }
}
```

## Development Workflow

### Daily Development

For regular BitNet.rs development, cross-validation is **not needed**:

```bash
# Normal development workflow (fast)
cargo build
cargo test
cargo clippy
```

### Before Releases

Enable cross-validation for release validation:

```bash
# Pre-release validation
./ci/fetch_bitnet_cpp.sh
cargo test --features crossval --release
cargo bench --features crossval --release
```

### CI/CD Integration

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  # Fast primary CI (no cross-validation)
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: cargo test --workspace
  
  # Optional cross-validation (slower)
  crossval:
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      - name: Install dependencies
        run: sudo apt install clang cmake build-essential
      - name: Setup C++ implementation
        run: ./ci/fetch_bitnet_cpp.sh
      - name: Run cross-validation
        run: cargo test --features crossval
```

## Troubleshooting

### Common Issues

#### 1. "C++ implementation not found"

**Error:**
```
ERROR: BitNet C++ implementation not found at ~/.cache/bitnet_cpp
```

**Solution:**
```bash
# Run the setup script
./ci/fetch_bitnet_cpp.sh

# Or set custom path
export BITNET_CPP_PATH=/path/to/bitnet.cpp
```

#### 2. "clang not found"

**Error:**
```
ERROR: clang not found - cannot generate C++ bindings
```

**Solution:**
```bash
# Ubuntu/Debian
sudo apt install clang libclang-dev

# macOS
xcode-select --install

# Windows
# Install LLVM from https://llvm.org/
```

#### 3. "Feature crossval not enabled"

**Error:**
```
Cross-validation not available (compile with --features crossval)
```

**Solution:**
```bash
# Enable the feature
cargo test --features crossval
cargo bench --features crossval
```

#### 4. Build failures

**Error:**
```
Failed to compile bitnet-sys
```

**Solution:**
```bash
# Clean and rebuild
cargo clean
./ci/fetch_bitnet_cpp.sh --clean --force
cargo build --features crossval
```

### Debug Mode

Enable detailed logging:

```bash
# Enable debug logging
RUST_LOG=debug cargo test --features crossval

# Enable trace logging for specific modules
RUST_LOG=bitnet_crossval=trace cargo test --features crossval
```

### Performance Issues

If cross-validation is slow:

```bash
# Use release mode for better performance
cargo test --features crossval --release
cargo bench --features crossval --release

# Reduce test scope
cargo test --features crossval -- --test-threads=1
```

## Best Practices

### 1. Feature Usage

- **Default OFF**: Never enable crossval by default
- **Explicit enabling**: Always require explicit `--features crossval`
- **Documentation**: Clearly document when crossval is needed
- **Graceful degradation**: Handle missing crossval gracefully

### 2. Development Workflow

- **Fast iteration**: Use normal builds for daily development
- **Pre-commit**: Run crossval tests before important commits
- **Release validation**: Always run crossval before releases
- **CI separation**: Keep fast CI separate from crossval CI

### 3. Testing Strategy

- **Unit tests**: Don't require crossval for unit tests
- **Integration tests**: Use crossval for integration validation
- **Performance tests**: Use crossval for performance comparison
- **Regression tests**: Use crossval to detect regressions

### 4. Documentation

- **Clear prerequisites**: Document all system dependencies
- **Setup instructions**: Provide step-by-step setup guides
- **Troubleshooting**: Include common issues and solutions
- **Examples**: Show both enabled and disabled usage

## Advanced Configuration

### Custom C++ Build

```bash
# Use custom C++ build configuration
export BITNET_CPP_CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=Debug -DENABLE_PROFILING=ON"
./ci/fetch_bitnet_cpp.sh
```

### Multiple C++ Versions

```bash
# Test against multiple C++ versions
for version in v1.0.0 v1.1.0 v1.2.0; do
    ./ci/bump_bitnet_tag.sh update $version
    cargo test --features crossval
done
```

### Custom Test Fixtures

```bash
# Generate custom test fixtures
cargo xtask gen-fixtures --prompts 10 --deterministic

# Validate fixtures
cargo xtask validate-fixtures

# Clean generated fixtures
cargo xtask clean-fixtures
```

## Performance Optimization

### Build Optimization

```bash
# Optimize C++ build for performance
export BITNET_CPP_CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=Release -DOPTIMIZE_FOR_NATIVE=ON"
./ci/fetch_bitnet_cpp.sh --clean --force
```

### Parallel Testing

```bash
# Run tests in parallel
cargo test --features crossval --jobs 4

# Run benchmarks with more iterations
cargo bench --features crossval -- --sample-size 100
```

### Memory Optimization

```bash
# Reduce memory usage during testing
export BITNET_CPP_MEMORY_LIMIT=2GB
cargo test --features crossval
```

---

**Need help?** Join our [Discord community](https://discord.gg/bitnet-rust) or check the [troubleshooting guide](troubleshooting.md) for more assistance with cross-validation setup.
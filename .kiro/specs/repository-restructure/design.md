# Repository Restructure Design

## Overview

This design document outlines the comprehensive restructuring of the BitNet repository to establish BitNet.rs as the primary implementation while preserving the original C++ implementation as a legacy benchmark target.

## Architecture

### Target Repository Structure

```
/
├── README.md                    # 🦀 "BitNet Rust – Production Implementation"
├── Cargo.toml                   # Root workspace configuration
├── build.rs                     # Rust build script
├── LICENSE                      # Project license
├── CHANGELOG.md                 # Rust implementation changelog
├── FEATURES.md                  # Rust feature documentation
├── SECURITY.md                  # Security policy
├── CONTRIBUTING.md              # Rust-focused contribution guide
├── .github/                     # GitHub workflows (Rust-focused)
│   └── workflows/
│       ├── rust-ci.yml          # Primary Rust CI/CD (fast)
│       ├── nightly-crossval.yml # Optional external legacy comparison
│       └── release.yml          # Rust package releases
├── crates/                      # Rust implementation (unchanged)
│   ├── bitnet-common/
│   ├── bitnet-models/
│   ├── bitnet-quantization/
│   ├── bitnet-kernels/
│   ├── bitnet-inference/
│   ├── bitnet-tokenizers/
│   ├── bitnet-cli/
│   ├── bitnet-server/
│   ├── bitnet-ffi/
│   ├── bitnet-py/
│   ├── bitnet-wasm/
│   └── bitnet-sys/              # FFI bindings (feature = "crossval")
├── examples/                    # Rust examples
├── benches/                     # Rust benchmarks
├── tests/                       # Rust integration tests
├── docs/                        # Rust documentation
├── ci/                          # CI helper scripts
│   ├── fetch_bitnet_cpp.sh      # Downloads & builds Microsoft's BitNet.cpp
│   ├── apply_patches.sh         # Applies minimal patches if needed
│   └── bump_bitnet_tag.sh       # Updates pinned version
├── patches/                     # Minimal patches (ideally empty)
│   └── (only if absolutely necessary)
├── crossval/                    # Cross-validation harness
│   ├── Cargo.toml               # Separate crate with crossval feature
│   ├── benches/                 # Criterion benchmarks vs C++
│   ├── tests/                   # Token-level equivalence tests
│   └── fixtures/                # Small test models (~20KB)
├── deployment/                  # Rust deployment configs
├── docker/                      # Rust Docker configurations
├── k8s/                         # Kubernetes configs for Rust
├── helm/                        # Helm charts for Rust
├── monitoring/                  # Monitoring for Rust services
└── .vscode/                     # Rust-focused workspace
```

**Key Changes:**
- ❌ **No C++ source code in repository**
- ✅ **External fetch**: `ci/fetch_bitnet_cpp.sh` downloads Microsoft's official release
- ✅ **Minimal patches**: Only if absolutely necessary for FFI compatibility
- ✅ **Feature-gated**: Cross-validation behind `--features crossval`
- ✅ **Zero maintenance**: We never fork or maintain C++ code

## Components and Interfaces

### 1. Primary Rust Implementation

**Location:** Root level and `/crates/`

**Responsibilities:**
- Main library implementation
- CLI tools and server
- Language bindings (Python, WASM, C API)
- Documentation and examples
- Primary CI/CD and testing

**Interfaces:**
- Standard Rust crate APIs
- C FFI for language bindings
- REST API for server
- CLI interface

### 2. External Legacy Reference

**Location:** Downloaded on-demand to `$HOME/.cache/bitnet_cpp/`

**Responsibilities:**
- Benchmark and comparison target
- Cross-validation reference
- Migration compatibility testing
- Performance baseline

**Interfaces:**
- FFI bindings via `bitnet-sys` crate
- Direct binary execution for CLI comparison
- Shared library linking for performance tests

**Key Design Decisions:**
- **External dependency**: Never checked into our repository
- **Pinned version**: Fixed tag/commit for deterministic testing
- **Cached builds**: Avoid recompilation via CI cache
- **Minimal patches**: Only applied if absolutely necessary for FFI compatibility
- **Feature-gated**: Only built when `--features crossval` is enabled

### 3. Cross-Validation Framework

**Location:** `/crossval/` (separate crate)

**Responsibilities:**
- Automated comparison testing
- Performance benchmarking
- Numerical accuracy validation
- Compatibility verification

**Components:**

#### 3.1 Cross-Validation Tests
```rust
// crossval/tests/token_equivalence.rs
#[cfg(feature = "crossval")]
#[test]
fn token_equivalence_small_prompt() {
    let model = fixtures::mini_model(); // 20 kB GGUF stub
    let prompt = "Rust and C++ walk into a bar…";

    let rust_out = bitnet_rs::generate(&model, prompt);
    let cpp_out = bitnet_cpp::generate(&model, prompt); // via FFI

    assert_eq!(rust_out.tokens, cpp_out.tokens); // exact match
}

// crossval/benches/performance.rs
#[cfg(feature = "crossval")]
fn criterion_benchmark(c: &mut Criterion) {
    let model = fixtures::standard_model();

    c.bench_function("rust_inference", |b| {
        b.iter(|| bitnet_rs::generate(&model, "test prompt"))
    });

    c.bench_function("cpp_inference", |b| {
        b.iter(|| bitnet_cpp::generate(&model, "test prompt"))
    });
}
```

#### 3.2 Test Fixtures
- Standard model files (GGUF format)
- Test prompt datasets
- Expected output baselines
- Performance test configurations

#### 3.3 Automated Reporting
- Numerical accuracy reports
- Performance comparison charts
- Regression detection
- Compatibility matrices

## Data Models

### Repository Configuration

```toml
# Cargo.toml (Root workspace)
[workspace]
resolver = "2"
members = [
    "crates/*"
]

[workspace.metadata.bitnet]
primary_implementation = "rust"
legacy_path = "legacy/bitnet.cpp"
cross_validation_path = "cross-validation"

[workspace.metadata.cross-validation]
enabled = true
tolerance = 1e-6
benchmark_models = [
    "bitnet_b1_58-3B",
    "bitnet_b1_58-large"
]
```

### Legacy Configuration

```cmake
# legacy/bitnet.cpp/CMakeLists.txt
cmake_minimum_required(VERSION 3.14)
project("bitnet.cpp-legacy" C CXX)

# Mark as legacy build
add_compile_definitions(BITNET_LEGACY_BUILD)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Isolated build directory
set(CMAKE_BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build)
```

## Error Handling

### Migration Errors
- **File conflicts:** Automated detection and resolution of conflicting files
- **Build failures:** Clear error messages directing to appropriate build system
- **Path dependencies:** Automated path updates in configuration files

### Cross-Validation Errors
- **Numerical mismatches:** Detailed reporting with tolerance analysis
- **Performance regressions:** Automated alerts with threshold configuration
- **API incompatibilities:** Clear documentation of breaking changes

### Legacy Access Errors
- **Missing dependencies:** Clear setup instructions for legacy environment
- **Build system conflicts:** Isolated build processes prevent interference
- **Version mismatches:** Compatibility matrices and version pinning

## Testing Strategy

### 1. Rust Implementation Testing
- **Unit tests:** Comprehensive coverage of all crates
- **Integration tests:** End-to-end workflow validation
- **Performance tests:** Benchmarking and regression detection
- **Property-based tests:** Algorithmic correctness validation

### 2. Legacy Implementation Testing
- **Preservation tests:** Ensure legacy functionality remains intact
- **Build tests:** Validate C++ build process in isolated environment
- **Compatibility tests:** Verify API compatibility for migration

### 3. Cross-Validation Testing
- **Numerical accuracy:** Token-level output comparison
- **Performance parity:** Throughput and latency comparison
- **API compatibility:** Function signature and behavior validation
- **Model compatibility:** Support for same model formats

### 4. Migration Testing
- **Documentation accuracy:** Validate migration guides with real scenarios
- **Tool compatibility:** Ensure migration tools work correctly
- **Performance validation:** Confirm performance improvements in migration

## Implementation Phases

### Phase 1: Structure Preparation (2 days)
1. Create new directory structure
2. Move C++ files to `/legacy/bitnet.cpp/`
3. Update build configurations
4. Create cross-validation framework skeleton

### Phase 2: Build System Isolation (1 day)
1. Isolate C++ build system in legacy directory
2. Update Rust build to be root-focused
3. Ensure no build conflicts
4. Update CI/CD configurations

### Phase 3: Documentation Update (1 day)
1. Rewrite README to focus on Rust
2. Create legacy documentation
3. Write migration guides
4. Update all references and links

### Phase 4: Cross-Validation Implementation (2 days)
1. Implement comparison scripts
2. Create test fixtures
3. Set up automated benchmarking
4. Integrate with CI/CD

### Phase 5: Validation and Testing (1 day)
1. Test all build processes
2. Validate cross-validation framework
3. Ensure documentation accuracy
4. Performance validation

## Monitoring and Observability

### Build Monitoring
- **Rust build success rates:** Track primary implementation build health
- **Legacy build status:** Monitor legacy build for cross-validation needs
- **Cross-validation results:** Track comparison test outcomes

### Performance Monitoring
- **Benchmark trends:** Monitor performance improvements over time
- **Regression detection:** Automated alerts for performance degradation
- **Comparison metrics:** Track Rust vs C++ performance ratios

### Usage Analytics
- **Primary vs Legacy usage:** Track which implementation users choose
- **Migration patterns:** Monitor adoption of Rust implementation
- **Documentation effectiveness:** Track user success with migration guides

## Security Considerations

### Access Control
- **Legacy isolation:** Ensure legacy code cannot affect primary implementation
- **Build isolation:** Prevent cross-contamination between build systems
- **Dependency isolation:** Separate dependency trees for each implementation

### Supply Chain Security
- **Rust dependencies:** Standard Rust security practices and auditing
- **Legacy dependencies:** Isolated C++ dependency management
- **Cross-validation security:** Secure handling of test data and models

### Vulnerability Management
- **Primary focus:** Security updates prioritized for Rust implementation
- **Legacy maintenance:** Minimal security updates for legacy code
- **Disclosure process:** Clear process for reporting security issues

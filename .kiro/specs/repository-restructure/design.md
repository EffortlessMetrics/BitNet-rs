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
├── .gitattributes               # Mark legacy/* as linguist-vendored
├── .github/                     # GitHub workflows (Rust-focused)
│   └── workflows/
│       ├── rust-ci.yml          # Primary Rust CI/CD (fast)
│       ├── nightly-crossval.yml # Optional legacy comparison (nightly)
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
│   └── bitnet-wasm/
├── examples/                    # Rust examples
├── benches/                     # Rust benchmarks
├── tests/                       # Rust integration tests
├── docs/                        # Rust documentation
├── scripts/                     # Rust-focused scripts
│   └── dev_setup.sh             # --with-legacy flag for GPU toolchain
├── deployment/                  # Rust deployment configs
├── docker/                      # Rust Docker configurations
├── k8s/                         # Kubernetes configs for Rust
├── helm/                        # Helm charts for Rust
├── monitoring/                  # Monitoring for Rust services
├── .vscode/                     # Workspace: Rust primary, legacy secondary
├── .idea/                       # IntelliJ workspace configuration
├── legacy/                      # 🏛️ Legacy C++ (sandboxed & slim)
│   ├── README.md                # "Legacy C++ reference – not for production"
│   ├── cpp/                     # Core C++ implementation
│   │   ├── CMakeLists.txt       # C++ build system
│   │   ├── src/                 # C++ source files
│   │   ├── include/             # C++ headers
│   │   ├── 3rdparty/            # C++ dependencies
│   │   ├── gpu/                 # GPU implementation
│   │   ├── utils/               # C++ utilities
│   │   ├── preset_kernels/      # Precomputed kernels
│   │   ├── setup_env.py         # C++ environment setup
│   │   ├── run_inference.py     # C++ inference runner
│   │   ├── run_inference_server.py # C++ server
│   │   └── requirements.txt     # Python dependencies
│   └── docker/                  # Isolated C++ build containers
│       ├── ubuntu-cuda.Dockerfile
│       └── ubuntu-cpu.Dockerfile
└── tools/                       # Cross-implementation tooling
    ├── crossval/                # Rust ↔ C++ comparison harness
    │   ├── pytest + rust scripts
    │   └── fixtures/            # Test data and models
    └── bench/                   # Criterion.rs + gbench wrappers
```

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

### 2. Legacy C++ Implementation

**Location:** `/legacy/cpp/` (sandboxed & slim)

**Responsibilities:**
- Benchmark and comparison target
- Cross-validation reference
- Migration compatibility testing
- Performance baseline

**Interfaces:**
- C++ API (preserved for compatibility)
- Python bindings (legacy)
- CLI interface (legacy)

**Key Design Decisions:**
- **Sandboxed**: Complete build isolation via Docker containers
- **Slim checkout**: `.gitattributes` marks as `linguist-vendored`, optional in shallow clones
- **Patch-friendly**: Kept in-repo (not submodule) for cross-validation tweaks
- **Clear deprecation**: Prominent "not for production" warnings
- **CI efficiency**: Only runs on `/legacy` changes or nightly schedule

### 3. Cross-Validation Framework

**Location:** `/tools/crossval/` (integrated tooling)

**Responsibilities:**
- Automated comparison testing
- Performance benchmarking  
- Numerical accuracy validation
- Compatibility verification

**Components:**

#### 3.1 Comparison Scripts
```python
# tools/crossval/compare_implementations.py
def compare_inference(model_path, prompts, tolerance=1e-6):
    """Compare inference outputs between Rust and C++ implementations"""
    rust_results = run_rust_inference(model_path, prompts)
    # C++ runs in isolated Docker container
    cpp_results = run_cpp_inference_docker(model_path, prompts)
    return validate_numerical_accuracy(rust_results, cpp_results, tolerance)

def benchmark_performance(model_path, test_cases):
    """Benchmark performance with criterion.rs + gbench wrappers"""
    rust_metrics = run_criterion_benchmarks(model_path, test_cases)
    cpp_metrics = run_gbench_in_docker(model_path, test_cases)
    return generate_performance_report(rust_metrics, cpp_metrics)
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
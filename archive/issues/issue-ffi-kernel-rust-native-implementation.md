# [Architecture] FFI Kernel Migration to Pure Rust Implementation

## Problem Description

BitNet-rs FFI kernel system currently relies on external C++ implementations, creating deployment complexity and runtime dependencies. Two critical FFI components require native Rust implementation:

1. **C++ Bridge Dependency**: `FfiKernel::new()` requires runtime availability checks for C++ libraries
2. **External Implementation**: `matmul_i2s()` and `quantize()` functions directly call C++ bridge implementations
3. **Deployment Complexity**: FFI bridge requires separate C++ compilation and library distribution
4. **Runtime Detection**: Availability checking through function calls instead of compile-time detection

This architecture prevents self-contained deployment and adds complexity to the build and distribution process.

## Environment

- **Affected Crates**: `bitnet-kernels`, `bitnet-common`
- **Primary Files**:
  - `crates/bitnet-kernels/src/ffi.rs`
  - `crates/bitnet-kernels/src/ffi/bridge/cpp.rs` (C++ bridge)
- **Build Configuration**: `--no-default-features --features ffi`
- **Target Platforms**: Linux x86_64, ARM64, Windows, macOS
- **Kernel Operations**: I2S matrix multiplication, quantization (I2S, TL1, TL2)

## Impact Assessment

- **Severity**: Medium-High - Affects deployment simplicity and platform compatibility
- **Deployment Impact**: Requires separate C++ library distribution
- **Build Complexity**: Increases build time and toolchain requirements
- **Platform Support**: Limited by C++ bridge availability
- **Performance**: FFI overhead affects kernel call latency
- **Maintenance**: Dual-language codebase increases complexity

## Proposed Solution

### Pure Rust Implementation Strategy

Replace FFI bridge with native Rust kernels featuring SIMD optimizations and cross-platform compatibility.

## Implementation Plan

### Phase 1: Native Matmul Kernel Development (Week 1-3)
- [ ] Implement scalar I2S matrix multiplication kernel
- [ ] Add AVX2 SIMD optimizations for x86_64
- [ ] Implement AVX-512 optimizations for high-end CPUs
- [ ] Add NEON optimizations for ARM64 platforms
- [ ] Create comprehensive unit tests and benchmarks

### Phase 2: Native Quantization Implementation (Week 3-5)
- [ ] Implement I2S quantization with block processing
- [ ] Add TL1 and TL2 quantization support
- [ ] Optimize quantization kernels with SIMD instructions
- [ ] Add error handling and input validation
- [ ] Create quantization accuracy tests

### Phase 3: Kernel Integration and Interface (Week 5-6)
- [ ] Replace `FfiKernel` with `NativeKernel` in provider system
- [ ] Update `KernelManager` to use native implementation
- [ ] Remove C++ bridge dependencies and FFI code
- [ ] Add capability detection and runtime selection
- [ ] Update build system to remove C++ requirements

## Acceptance Criteria

### Functional Requirements
- [ ] Complete I2S matrix multiplication in pure Rust
- [ ] Full quantization support for I2S, TL1, TL2 formats
- [ ] SIMD optimizations for x86_64 (AVX2, AVX-512) and ARM64 (NEON)
- [ ] Runtime capability detection and optimal kernel selection
- [ ] Zero C++ dependencies in final implementation

### Performance Requirements
- [ ] Native implementation within 90% of FFI bridge performance
- [ ] SIMD implementations show >2x speedup over scalar code
- [ ] Memory usage comparable to or better than FFI implementation
- [ ] Startup time reduced by removing C++ library loading

### Quality Requirements
- [ ] Numerical accuracy within 1e-5 of reference implementation
- [ ] Comprehensive test coverage >95% for all kernel paths
- [ ] Cross-platform validation on all target architectures
- [ ] No memory leaks or undefined behavior (verified with Miri)

## Related Issues

- BitNet-rs #251: Production-ready inference server (benefits from simplified deployment)
- BitNet-rs #218: Device-aware quantization system (uses native quantization kernels)
- BitNet-rs #260: Mock elimination project (eliminates FFI bridge complexity)

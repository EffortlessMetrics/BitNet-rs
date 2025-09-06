# Changelog

All notable changes to BitNet.rs will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Production-Ready Streaming Inference** ([#182](https://github.com/EffortlessSteven/BitNet-rs/pull/182)):
  - Real async streaming implementation using `GenerationStream` with futures and `StreamExt::next()`
  - Enhanced NaN-safe sampling operations with hardened floating-point comparisons in `top_k_filter` and `top_p_filter`
  - Accurate performance metrics collection during streaming with proper prefill execution via `engine.eval_ids()`
  - Integration tests enabled by default (removed feature gates) for comprehensive test coverage
  - Futures crate dependency reintroduced for async stream processing with `StreamExt::next()` support
  - Robust error handling with `.unwrap_or(std::cmp::Ordering::Equal)` for NaN-resilient sorting operations
- **Device-Aware GPU Quantization Support**:
  - Enhanced I2S, TL1, and TL2 quantizers with automatic GPU acceleration
  - Device-aware dequantization with intelligent CPU fallback  
  - GPU memory optimization and mixed precision support
  - Comprehensive GPU vs CPU validation tests with configurable tolerance
  - Proper feature gating with `#[cfg(feature = "gpu")]` for CPU-only builds
- **Device Memory Tracking Infrastructure** ([#185](https://github.com/EffortlessSteven/BitNet-rs/pull/185)):
  - Real-time host memory monitoring via `memory-stats` crate for process-specific usage
  - System memory tracking via `sysinfo` crate with optimized refresh calls
  - Thread-safe memory statistics with Arc<Mutex<DeviceStatsInternal>> protection
  - Memory efficiency metrics and usage percentage calculations in DeviceStats
  - Comprehensive test coverage for memory tracking functionality
  - Integration with device-aware quantization for automatic memory monitoring
- **Enhanced CUDA Kernel Infrastructure**:
  - Improved CUDA kernel provider with better error handling
  - Memory management optimization with automatic leak detection
  - Performance monitoring with built-in kernel execution timing
  - Device information extraction and capability detection
  - Advanced validation framework with numerical accuracy testing
- **Documentation Quality Improvements**:
  - Fixed HTML tag warnings in API documentation
  - Enhanced quantization documentation with device-aware capabilities
  - Updated CLAUDE.md with comprehensive GPU validation commands
  - Improved inline documentation with proper backtick formatting
  - Fixed broken intra-doc links and reference formatting
  - Added comprehensive project analysis in `GOALS_VS_REALITY_ANALYSIS.md` with goal-by-goal assessment ([#152](https://github.com/EffortlessSteven/BitNet-rs/pull/152))
- **Teacher-Forcing Scoring and Perplexity Calculation** ([#134](https://github.com/EffortlessSteven/BitNet-rs/pull/134)):
  - New `score` CLI command with real teacher-forcing evaluation using inference engine
  - Device selection support (`--device cpu|cuda|metal|auto`) with automatic fallback
  - Batch processing (`--batch-size`) for improved throughput on large datasets
  - JSON output with detailed metrics (tokens, mean_nll, perplexity, latency)
  - External tokenizer support (`--tokenizer`) and token limit controls (`--max-tokens`)
  - Comprehensive CLI documentation with usage examples and troubleshooting guides
- **Python Binding Environment Documentation** ([#134](https://github.com/EffortlessSteven/BitNet-rs/pull/134)):
  - New `PYTHON_BINDING_ENVIRONMENT.md` documenting PyO3 linking requirements
  - Workspace configuration explanation for optional Python binding tests
  - System package installation instructions for Ubuntu/Debian, CentOS/RHEL/Fedora, and macOS
  - CI/CD recommendations for environment-dependent testing
- **Streaming Token ID Support** ([#107](https://github.com/EffortlessSteven/BitNet-rs/pull/107)):
  - Enhanced `StreamResponse` struct with `token_ids: Vec<u32>` field for real-time token ID access
  - Server-Sent Events (SSE) streaming endpoint with JSON token metadata at `/v1/stream`
  - Token-by-token streaming with configurable buffering and error handling
  - Comprehensive test coverage for streaming functionality and token ID accuracy
  - Updated examples and documentation to demonstrate token ID streaming usage
- **2D Convolution Operations Support** ([#165](https://github.com/EffortlessSteven/BitNet-rs/pull/165)):
  - Complete 2D convolution implementation with `conv2d` and `conv2d_quantized` functions
  - Support for NCHW input format and OIHW weight format with configurable parameters
  - Stride, padding, and dilation operations for flexible convolution configurations
  - Quantized convolution with I2S, TL1, and TL2 quantization types
  - On-the-fly dequantization with per-channel scaling factors
  - PyTorch reference testing framework for numerical correctness validation
  - Comprehensive unit tests covering basic functionality, edge cases, and error handling
  - Integration with existing BitNet.rs kernel architecture and error handling patterns
- **GGUF Validation API**:
  - Fast 24-byte header-only validation without loading full model
  - Production-ready parser with typed errors and non-exhaustive enums
  - `compat-check` CLI command with stable exit codes for CI automation
  - `--strict` flag for enforcing version and sanity checks
  - Early validation in engine before heavy memory allocations
  - JSON output for programmatic validation scripts
- **GGUF KV Metadata Reader**:
  - Read and inspect GGUF key-value metadata without full model loading
  - Support for all GGUF value types (except arrays, deferred)
  - `--show-kv` flag in CLI to display model metadata
  - `--kv-limit` flag to control number of displayed KV pairs
  - JSON output includes metadata when `--show-kv` is used
  - Safety limits to prevent excessive memory usage

### Changed
- **Cargo Configuration Cleanup** ([#113](https://github.com/EffortlessSteven/BitNet-rs/pull/113)):
  - Remove tool-generated metadata files (`.crates.toml`, `.crates2.json`) from version control
  - Commit `Cargo.lock` files for reproducible builds across environments
  - Standardize GPU feature aliases in cargo config to use `gpu` instead of `cuda`

### Fixed
- **Code Quality and Security Improvements**:
  - Fixed critical PyO3 security vulnerability (RUSTSEC-2025-0020)
  - Resolved 45+ clippy warnings across workspace for better code quality
  - Updated dependencies (atty→is-terminal, removed wee_alloc)
  - Enhanced type safety and error handling documentation
  - Resolved all clippy warnings across the codebase with proper type improvements
  - Enhanced kernel validation system with improved error handling and performance metrics
  - Fixed FFI bridge test tolerance calculations for accurate migration recommendations
  - Improved universal tokenizer documentation and error handling
  - Enhanced model loading with better GGUF handling and error propagation
  - Standardized code formatting and documentation strings throughout
- **bitnet-server Build Issues**:
  - Restored Git metadata support using vergen-gix v1.x
  - Moved runtime dependencies from build-dependencies to correct section
  - Made health endpoint robust with option_env! for graceful fallbacks
- **Memory Safety and Environment Variable Handling** ([#181](https://github.com/EffortlessSteven/BitNet-rs/pull/181)):
  - Enhanced BitNetTensor with proper device tracking and memory leak prevention
  - Replaced unsafe `Box::leak()` with safe `OnceLock<Vec<f32>>` caching for host data
  - Safe type conversion using `bytemuck::cast_slice` instead of manual transmutation  
  - Removed redundant `DeviceType` enum in favor of unified `Device` enum
  - Rust 2024 compliance: marked environment variable manipulations as `unsafe`
  - Improved Clone trait implementation for BitNetTensor with proper data handling
- **IQ2_S FFI Layout Enhancement and Parity Testing** ([#142](https://github.com/EffortlessSteven/BitNet-rs/pull/142)):
  - Enhanced `BlockIq2S` struct with perfect GGML `block_iq2_s` layout compatibility (82 bytes)
  - Added compile-time size and alignment assertions for layout parity verification
  - Enabled previously ignored `iq2s_rust_matches_ffi` parity test with precise element-by-element comparison
  - Replaced hardcoded block size with compile-time `size_of::<BlockIq2S>()` for consistency
  - Zero API breaking changes, maintaining full backward compatibility
- **IQ2_S Quantization Layout Alignment** ([#132](https://github.com/EffortlessSteven/BitNet-rs/pull/132)):
  - Updated IQ2_S block layout from 66B to 82B to match GGML specification exactly
  - Corrected QMAP values from `[-2,-1,0,1]` to `[-2,-1,1,2]` eliminating zero mapping
  - Updated test expectations to match new quantization mapping pattern
  - Ensures bit-exact compatibility between Rust and GGML backends
  - Fixed unsafe code warnings with proper unsafe blocks for Rust 2024 compliance
- **Security Vulnerability Resolution** ([#107](https://github.com/EffortlessSteven/BitNet-rs/pull/107)):
  - Updated PyO3 from v0.21.2 to v0.25.1 to resolve CVE-2024-9979 buffer overflow vulnerability
  - Updated related Python binding dependencies (numpy, pyo3-async-runtimes) for compatibility
  - Enhanced security posture of Python bindings and server components
- **FFI Safety and Validation Improvements**:
  - Enhanced FFI functions with `unsafe fn` signatures for Rust 2024 safety compliance
  - Fixed clippy warnings in test infrastructure and removed unneeded unit expressions
  - Added proper unsafe blocks for raw pointer operations in C API layer
  - Maintained full API compatibility with existing C clients while improving memory safety
- **CUDA Device Information Querying** (PR #102):
  - Real device property querying using cudarc's CUdevice_attribute API
  - Comprehensive device information extraction (compute capability, memory, multiprocessor count)
  - Enhanced error handling for device query failures
  - Enhanced test coverage for device information validation

### Enhanced
- **Weight Mapper Model Compatibility Validation** ([#144](https://github.com/EffortlessSteven/BitNet-rs/pull/144)):
  - Enhanced `validate_model_compatibility()` to use weight mapper for GGUF tensor validation
  - Replaced TODO placeholder with actual GGUF parsing and tensor name mapping
  - Added detection of unmapped tensors with detailed error reporting and debugging metrics
  - Comprehensive test coverage with fixture support for both success and corruption scenarios
  - Improved fallback handling in universal tokenizer with SpmTokenizer typo fixes
- **GPU Kernel Refactoring** (PR #108):
  - **CUDA Implementation**: Enhanced cudarc 0.17 API compatibility with performance tracking and error handling
  - **Memory Management**: Implemented OptimizedMemoryPool with device-specific allocation, caching, leak detection, and device_id() access method  
  - **Mixed Precision Infrastructure**: Added PrecisionMode support for FP16/BF16 operations on modern GPUs
  - **Comprehensive Validation**: GpuValidator with numerical accuracy, performance benchmarking, and memory health checks
  - **FFI Bridge Improvements**: Enhanced C++ kernel integration with feature gating and performance comparison tools
  - **Device Information**: Detailed CudaDeviceInfo with compute capability, memory, and precision support detection
  - **Launch Parameter Optimization**: Dynamic kernel configuration based on device capabilities and workload characteristics
- **Documentation Synchronization** (Post-PR #113):
  - Updated all documentation to use standardized `gpu` feature flag instead of `cuda`
  - Maintained backward compatibility by documenting `cuda` as an alias for `gpu`
  - Synchronized build commands across CLAUDE.md, README.md, FEATURES.md, and GPU setup guides
  - Updated cargo aliases in `.cargo/config.toml` to use `gpu` feature consistently
  - Enhanced lockfile tracking for reproducible builds (Cargo.lock now versioned)
  - Enhanced Diátaxis framework compliance with clearer tutorial/reference categorization
- **CI/Docker Git Metadata Support**:
  - Added Git metadata injection in GitHub Actions CI
  - Updated Dockerfile with VCS build args for metadata without .git
  - Added docker-build.sh script for easy builds with Git metadata
  - Added OCI standard labels for container registries
  - Environment variable overrides for deterministic builds
- **Enhanced GGUF Metadata Inspection with Categorization** (PR #97):
  - **Comprehensive ModelInfo API**: Added `kv_specs()`, `quantization_hints()`, and `tensor_summaries()` methods
  - **Advanced Categorization**: Added `get_categorized_metadata()` organizing KV pairs by purpose (model params, architecture, tokenizer, training, quantization)
  - **Tensor Statistics**: Added `get_tensor_statistics()` with parameter counts, memory estimates, and data type distribution
  - **JSON Serialization**: Added `to_json()` and `to_json_compact()` methods for automation and scripting
  - **Enhanced TensorSummary**: Includes tensor categories, parameter counts, and dtype names for comprehensive analysis
  - **Memory-Efficient Parsing**: Lightweight header-only inspection for CI/CD pipelines
  - **Error-Resilient Handling**: Robust parsing of malformed GGUF files with detailed error messages
  - **New Example**: `inspect_gguf_metadata.rs` demonstrating comprehensive metadata extraction with categorized output
  - **Enhanced CLI Integration**: Updated `bitnet inspect` command documentation with JSON output capabilities
  - **Validation Framework Integration**: Added GGUF inspection capabilities to validation workflows
- **Advanced Device-Aware Quantization with GPU Fallback** ([#106](https://github.com/EffortlessSteven/BitNet-rs/pull/106)):
  - **New `gpu` feature flag** with `cuda` backward-compatible alias for clearer GPU functionality
  - **DeviceAwareQuantizer** with intelligent automatic GPU detection and CPU fallback
  - **DeviceAwareQuantizerFactory** for streamlined device selection and auto-detection
  - **Comprehensive GPU quantization kernels** for I2S, TL1, and TL2 algorithms with optimized CUDA implementations
  - **Automatic CPU fallback** with graceful degradation and detailed error reporting
  - **Thread-safe concurrent GPU operations** with proper memory management and cleanup
  - **Performance monitoring** with device statistics and operation tracking
  - **Enhanced error handling** with new QuantizationFailed and MatmulFailed error types
  - **Comprehensive test suite** with 12 specialized test cases for GPU/CPU accuracy comparison, fallback validation, memory management, and concurrent operations
  - **FFI safety enhancements** with documentation for all 30+ unsafe functions for Rust 2024 compliance

### Changed
- **Cargo Configuration Cleanup** ([#113](https://github.com/EffortlessSteven/BitNet-rs/pull/113)):
  - Remove tool-generated metadata files (`.crates.toml`, `.crates2.json`) from version control
  - Commit `Cargo.lock` files for reproducible builds across environments
  - Standardize GPU feature aliases in cargo config to use `gpu` instead of `cuda`

### Fixed
- **Code Quality and Security Improvements**:
  - Fixed critical PyO3 security vulnerability (RUSTSEC-2025-0020)
  - Resolved 45+ clippy warnings across workspace for better code quality
  - Updated dependencies (atty→is-terminal, removed wee_alloc)
  - Enhanced type safety and error handling documentation
  - Resolved all clippy warnings across the codebase with proper type improvements
  - Enhanced kernel validation system with improved error handling and performance metrics
  - Fixed FFI bridge test tolerance calculations for accurate migration recommendations
  - Improved universal tokenizer documentation and error handling
  - Enhanced model loading with better GGUF handling and error propagation
  - Standardized code formatting and documentation strings throughout
- **bitnet-server Build Issues**:
  - Restored Git metadata support using vergen-gix v1.x
  - Moved runtime dependencies from build-dependencies to correct section
  - Made health endpoint robust with option_env! for graceful fallbacks
- **Memory Safety and Environment Variable Handling** ([#181](https://github.com/EffortlessSteven/BitNet-rs/pull/181)):
  - Enhanced BitNetTensor with proper device tracking and memory leak prevention
  - Replaced unsafe `Box::leak()` with safe `OnceLock<Vec<f32>>` caching for host data
  - Safe type conversion using `bytemuck::cast_slice` instead of manual transmutation  
  - Removed redundant `DeviceType` enum in favor of unified `Device` enum
  - Rust 2024 compliance: marked environment variable manipulations as `unsafe`
  - Improved Clone trait implementation for BitNetTensor with proper data handling
- **IQ2_S FFI Layout Enhancement and Parity Testing** ([#142](https://github.com/EffortlessSteven/BitNet-rs/pull/142)):
  - Enhanced `BlockIq2S` struct with perfect GGML `block_iq2_s` layout compatibility (82 bytes)
  - Added compile-time size and alignment assertions for layout parity verification
  - Enabled previously ignored `iq2s_rust_matches_ffi` parity test with precise element-by-element comparison
  - Replaced hardcoded block size with compile-time `size_of::<BlockIq2S>()` for consistency
  - Zero API breaking changes, maintaining full backward compatibility
- **IQ2_S Quantization Layout Alignment** ([#132](https://github.com/EffortlessSteven/BitNet-rs/pull/132)):
  - Updated IQ2_S block layout from 66B to 82B to match GGML specification exactly
  - Corrected QMAP values from `[-2,-1,0,1]` to `[-2,-1,1,2]` eliminating zero mapping
  - Updated test expectations to match new quantization mapping pattern
  - Ensures bit-exact compatibility between Rust and GGML backends
  - Fixed unsafe code warnings with proper unsafe blocks for Rust 2024 compliance
- **Security Vulnerability Resolution** ([#107](https://github.com/EffortlessSteven/BitNet-rs/pull/107)):
  - Updated PyO3 from v0.21.2 to v0.25.1 to resolve CVE-2024-9979 buffer overflow vulnerability
  - Updated related Python binding dependencies (numpy, pyo3-async-runtimes) for compatibility
  - Enhanced security posture of Python bindings and server components
- **FFI Safety and Validation Improvements**:
  - Enhanced FFI functions with `unsafe fn` signatures for Rust 2024 safety compliance
  - Fixed clippy warnings in test infrastructure and removed unneeded unit expressions
  - Added proper unsafe blocks for raw pointer operations in C API layer
  - Maintained full API compatibility with existing C clients while improving memory safety
- **CUDA Device Information Querying** (PR #102):
  - Real device property querying using cudarc's CUdevice_attribute API
  - Comprehensive device information extraction (compute capability, memory, multiprocessor count)
  - Enhanced error handling for device query failures
  - Enhanced test coverage for device information validation

### Enhanced
- **Weight Mapper Model Compatibility Validation** ([#144](https://github.com/EffortlessSteven/BitNet-rs/pull/144)):
  - Enhanced `validate_model_compatibility()` to use weight mapper for GGUF tensor validation
  - Replaced TODO placeholder with actual GGUF parsing and tensor name mapping
  - Added detection of unmapped tensors with detailed error reporting and debugging metrics
  - Comprehensive test coverage with fixture support for both success and corruption scenarios
  - Improved fallback handling in universal tokenizer with SpmTokenizer typo fixes
- **GPU Kernel Refactoring** (PR #108):
  - **CUDA Implementation**: Enhanced cudarc 0.17 API compatibility with performance tracking and error handling
  - **Memory Management**: Implemented OptimizedMemoryPool with device-specific allocation, caching, leak detection, and device_id() access method  
  - **Mixed Precision Infrastructure**: Added PrecisionMode support for FP16/BF16 operations on modern GPUs
  - **Comprehensive Validation**: GpuValidator with numerical accuracy, performance benchmarking, and memory health checks
  - **FFI Bridge Improvements**: Enhanced C++ kernel integration with feature gating and performance comparison tools
  - **Device Information**: Detailed CudaDeviceInfo with compute capability, memory, and precision support detection
  - **Launch Parameter Optimization**: Dynamic kernel configuration based on device capabilities and workload characteristics
- **Documentation Synchronization** (Post-PR #113):
  - Updated all documentation to use standardized `gpu` feature flag instead of `cuda`
  - Maintained backward compatibility by documenting `cuda` as an alias for `gpu`
  - Synchronized build commands across CLAUDE.md, README.md, FEATURES.md, and GPU setup guides
  - Updated cargo aliases in `.cargo/config.toml` to use `gpu` feature consistently
  - Enhanced lockfile tracking for reproducible builds (Cargo.lock now versioned)
  - Enhanced Diátaxis framework compliance with clearer tutorial/reference categorization
- **CI/Docker Git Metadata Support**:
  - Added Git metadata injection in GitHub Actions CI
  - Updated Dockerfile with VCS build args for metadata without .git
  - Added docker-build.sh script for easy builds with Git metadata
  - Added OCI standard labels for container registries
  - Environment variable overrides for deterministic builds

## [0.3.0] - 2025-01-03

### Added
- **IQ2_S Quantization Support**:
  - Native Rust implementation with optimized dequantization
  - FFI backend via GGML for compatibility
  - Comprehensive unit tests and validation scripts
  - Backend parity testing between FFI and native implementations
- **Enhanced Test Suite**:
  - Feature-gated test configuration system
  - Improved fixture management with conditional compilation
  - Comprehensive integration test coverage
  - CI-friendly reporting with multiple output formats
- **Comprehensive CI Validation Framework**:
  - 8-gate acceptance system with JSON-driven detection
  - Distinct exit codes (0-10) for precise CI triage
  - Performance ratio gates with baseline comparisons
  - Deterministic execution environment (SEED=42, THREADS=1)
  - Portable memory profiling with GNU time/gtime
- **Score/Perplexity Subcommand**:
  - Teacher-forcing perplexity calculation skeleton
  - JSON output with tokenizer origin tracking
  - Support for external SentencePiece models
  - Ready for logits API integration
- **Strict Mode Enforcement**:
  - Zero unmapped tensors requirement
  - SentencePiece tokenizer validation
  - BOS token policy enforcement
  - Deterministic tie-breaking (lowest ID)
- Cross-validation framework for numerical accuracy testing
- Performance benchmarking suite with automated regression detection

### Fixed
- Model loading edge cases and error handling improvements
- Memory management optimizations for large models
- Cross-platform compatibility improvements

### Changed
- Improved API ergonomics and error messages
- Enhanced documentation with more examples
- Streamlined build process and dependency management

## [0.2.0] - 2024-12-15

### Added
- Basic quantization support (I2_S, TL1, TL2)
- GGUF format compatibility
- Python bindings with PyO3
- C API for llama.cpp compatibility
- Streaming inference capabilities
- Initial CUDA support

### Fixed
- Memory safety improvements
- Performance optimizations
- Cross-validation accuracy

## [0.1.0] - 2024-11-01

### Added
- Initial release
- Basic BitNet model loading and inference
- CPU-only quantization support
- Core API design and architecture
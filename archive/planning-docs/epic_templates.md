# BitNet-rs Tech Debt Epics: Issue Templates

**Purpose**: Consolidate 23 related issues from TDD scaffolding phase into 4 tracking epics

---

## Epic 1: TL1/TL2 Production Quantization

**Title**: `[EPIC] TL1/TL2 Production Quantization Implementation`

**Labels**: `epic`, `area/quantization`, `priority/medium`, `milestone/v0.2.0`

**Description**:

### Summary

Complete production-grade table lookup (TL1/TL2) quantization with device-aware selection, replacing simplified CPU implementations with optimized algorithms.

### Motivation

Current TL1/TL2 quantization implementations use simplified CPU fallbacks and stubs. This epic tracks the work to deliver production-ready table lookup quantization with:
- Device-aware selection (ARM NEON / x86 AVX optimization)
- Cross-validation parity with C++ reference implementation
- Performance targets for 1-bit neural network inference

### Scope

**In Scope**:
- TL1 quantization with production lookup tables
- TL2 quantization CPU implementation with offset support
- Device-aware quantization selection (ARM NEON / x86 AVX)
- Dequantization fast paths with SIMD optimization
- Cross-validation parity validation

**Out of Scope**:
- I2_S quantization optimization (tracked separately in #417)
- GPU CUDA quantization kernels (post-MVP)
- IQ2_S quantization (handled via FFI bridge)

### Consolidated Issues

This epic consolidates the following issues:
- #346: Replace simplified TL1 quantization with production table lookup implementation
- #399: Replace Simplified CPU Quantization Implementations with Production-Ready Algorithms
- #401: TL2 Quantizer CPU Implementation Contains Multiple Simulation Stubs Blocking Production Deployment
- #403: Missing CPUQuantizer::quantize_tl2 implementation in device_aware_quantizer.rs (duplicate)
- #416: Replace simplified TL1 dequantization with proper lookup table implementation
- #419: Missing CPUQuantizer::dequantize_tl2 Implementation Causes TL2 Functionality Gap

**Total**: 6 issues

### Acceptance Criteria

- [ ] **TL1 Quantization**: Production table lookup implementation with 2-bit signed quantization
- [ ] **TL2 Quantization**: CPU implementation with offset support for improved accuracy
- [ ] **Device Selection**: Automatic ARM NEON / x86 AVX selection based on runtime capability detection
- [ ] **Dequantization**: Fast paths with SIMD optimization (AVX2/AVX-512/NEON)
- [ ] **Cross-Validation**: Parity with C++ reference implementation (cosine similarity ≥0.999)
- [ ] **Performance**: Meet throughput targets for 1-bit model inference (≥3× scalar baseline)
- [ ] **Testing**: Comprehensive unit tests, property-based tests, and integration tests
- [ ] **Documentation**: Quantization algorithm documentation in `docs/reference/quantization-support.md`

### Performance Targets

- **TL1/TL2 throughput**: ≥3× improvement over scalar baseline
- **Cross-validation parity**: Cosine similarity ≥0.999 vs C++ reference
- **Device detection overhead**: <10ms for capability detection

### Implementation Tasks

**Phase 1: TL1 Production Implementation**
- [ ] Implement production TL1 quantization with 2-bit lookup tables
- [ ] Add device-aware selection for ARM NEON / x86 AVX
- [ ] Implement TL1 dequantization fast paths with SIMD
- [ ] Add cross-validation tests for TL1 parity

**Phase 2: TL2 Production Implementation**
- [ ] Implement TL2 quantization with offset support
- [ ] Add TL2 dequantization implementation
- [ ] Integrate TL2 with device-aware quantization system
- [ ] Add cross-validation tests for TL2 parity

**Phase 3: Optimization & Validation**
- [ ] Profile and optimize hot paths (quantization/dequantization loops)
- [ ] Add benchmarks for TL1/TL2 throughput measurement
- [ ] Validate performance targets (≥3× scalar baseline)
- [ ] Update documentation with algorithm details

### Dependencies

- Device capability detection system (ARM NEON / x86 AVX runtime checks)
- Cross-validation framework (C++ reference comparison)
- SIMD optimization infrastructure (AVX2/AVX-512/NEON intrinsics)

### Related Work

- **I2_S Optimization**: #417 (QK256 dequantization optimization, separate track)
- **QK256 AVX2**: PR #475 (AVX2 foundation for SIMD patterns)
- **Device-Aware Quantization**: Current architecture supports device selection pattern

### References

- `crates/bitnet-quantization/src/tl1.rs` - TL1 quantization implementation
- `crates/bitnet-quantization/src/tl2.rs` - TL2 quantization implementation
- `crates/bitnet-quantization/src/device_aware_quantizer.rs` - Device selection logic
- `docs/reference/quantization-support.md` - Quantization algorithm documentation

---

## Epic 2: Tokenizer Production Hardening

**Title**: `[EPIC] Tokenizer Production Hardening - GGUF Tokenizer & Mock Cleanup`

**Labels**: `epic`, `area/tokenization`, `priority/medium`, `milestone/v0.2.0`

**Description**:

### Summary

Production-grade GGUF tokenizer implementation with BPE/SentencePiece support, eliminating test mock tokenizers from production code paths.

### Motivation

Current tokenizer implementation has functional auto-discovery (PR #430) but contains:
- Simplified byte-level tokenization in `GgufTokenizer::encode`
- Oversimplified decoding without proper BPE/SentencePiece detokenization
- Mock tokenizers (`BasicTokenizer`, `MockTokenizer`) in production code paths
- Incomplete embedded GGUF tokenizer extraction

This epic tracks production hardening to deliver:
- Full BPE/SentencePiece tokenization support
- Proper embedded GGUF tokenizer extraction
- Removal of test mocks from production paths
- Parity validation with HuggingFace tokenizers

### Scope

**In Scope**:
- BPE tokenization/detokenization in `GgufTokenizer`
- SentencePiece tokenization/detokenization support
- Embedded GGUF tokenizer extraction from metadata
- Removal of `BasicTokenizer` and `MockTokenizer` from non-test code
- Parity validation with HuggingFace tokenizers

**Out of Scope**:
- Auto-discovery improvements (PR #430 complete, functional)
- Fallback chain modifications (existing implementation sufficient)
- Custom tokenizer training (use HuggingFace ecosystem)

### Consolidated Issues

This epic consolidates the following issues:
- #381: Implement Real Embedded Tokenizer Extraction from GGUF Metadata (Replace BasicTokenizer Stubs)
- #395: BasicTokenizer::encode Uses Placeholder Implementation That Breaks Real Tokenization
- #397: Replace BasicTokenizer::token_to_piece simulation with production vocabulary mapping
- #398: BasicTokenizer::decode returns placeholder text instead of actual decoding
- #400: MockTokenizer::encode uses naive character-based simulation - implement realistic tokenization
- #402: Implement Real Tokenization in GgufTokenizer::encode - Replace Byte-Level Simulation with BPE/SentencePiece Support
- #404: GGUF Tokenizer Decode Implementation is Oversimplified - Missing Proper BPE/SentencePiece Detokenization
- #409: BasicTokenizer decode() returns placeholder text instead of actual token decoding (duplicate)

**Total**: 8 issues

### Acceptance Criteria

- [ ] **BPE Tokenization**: Production-grade BPE encoding/decoding in `GgufTokenizer`
- [ ] **SentencePiece Support**: SentencePiece tokenization for GGUF models with SPM vocab
- [ ] **Embedded Extraction**: Extract GGUF tokenizer metadata (`tokenizer.ggml.*` keys) and reconstruct vocabulary
- [ ] **Mock Removal**: `BasicTokenizer` and `MockTokenizer` restricted to test-only code (`#[cfg(test)]`)
- [ ] **Parity Validation**: Token sequences match HuggingFace tokenizers (≥99.9% agreement)
- [ ] **Vocab Mapping**: Correct `token_to_piece` and `piece_to_token` mappings from GGUF metadata
- [ ] **Testing**: Comprehensive tokenization/detokenization tests with real model vocabularies
- [ ] **Documentation**: Tokenizer architecture documentation in `docs/tokenizer-architecture.md`

### Implementation Tasks

**Phase 1: GGUF Embedded Tokenizer Extraction**
- [ ] Implement GGUF metadata extraction (`tokenizer.ggml.*` keys)
- [ ] Reconstruct vocabulary from GGUF tokenizer arrays
- [ ] Add support for BPE merge tables from GGUF
- [ ] Add support for SentencePiece models from GGUF

**Phase 2: Production BPE/SentencePiece Implementation**
- [ ] Implement BPE tokenization algorithm in `GgufTokenizer::encode`
- [ ] Implement BPE detokenization in `GgufTokenizer::decode`
- [ ] Add SentencePiece tokenization support (detect from GGUF metadata)
- [ ] Add proper token-to-piece mapping with special token handling

**Phase 3: Mock Cleanup & Validation**
- [ ] Move `BasicTokenizer` to `#[cfg(test)]` test-only code
- [ ] Move `MockTokenizer` to `#[cfg(test)]` test-only code
- [ ] Add parity validation tests vs HuggingFace tokenizers
- [ ] Validate token sequences on representative prompts (≥99.9% agreement)

**Phase 4: Documentation & Integration**
- [ ] Update tokenizer architecture documentation
- [ ] Add GGUF tokenizer usage examples
- [ ] Update CLI documentation for tokenizer discovery
- [ ] Add troubleshooting guide for tokenizer mismatches

### Performance Targets

- **Tokenization throughput**: ≥10K tokens/sec for typical prompts
- **Parity validation**: ≥99.9% token sequence agreement with HuggingFace
- **Vocab extraction overhead**: <100ms for GGUF metadata parsing

### Dependencies

- GGUF metadata reader (existing in `bitnet-models`)
- Tokenizers crate (for HuggingFace parity validation)
- BPE algorithm reference implementation (study tokenizers.rs)

### Related Work

- **PR #430**: Universal Tokenizer Discovery System (auto-discovery complete)
- **Tokenizer Auto-Detection**: CLI auto-detects templates from GGUF metadata
- **Fallback Chain**: HuggingFace download fallback functional

### References

- `crates/bitnet-tokenizers/src/gguf_tokenizer.rs` - GGUF tokenizer implementation
- `crates/bitnet-tokenizers/src/discovery.rs` - Auto-discovery system
- `crates/bitnet-tokenizers/src/mock.rs` - Mock tokenizers (to be test-only)
- `docs/tokenizer-architecture.md` - Tokenizer architecture documentation
- External: [tokenizers.rs](https://github.com/huggingface/tokenizers) - Reference BPE implementation

---

## Epic 3: GPU Device Discovery & Memory Management

**Title**: `[EPIC] GPU Device Discovery & Memory Management - Real CUDA Capability Detection`

**Labels**: `epic`, `area/gpu`, `priority/medium`, `milestone/v0.2.0`

**Description**:

### Summary

Real CUDA device capability detection and GPU memory lifecycle management, replacing environment variable simulation and hardcoded placeholders with production-grade CUDA queries.

### Motivation

Current GPU infrastructure uses simplified stubs and environment variable simulation:
- GPU discovery simulates availability with `BITNET_GPU_FAKE` env var
- Tensor core support hardcoded, not queried from CUDA
- Mixed precision detection (FP16/BF16) simulated
- GPU memory uses 8GB placeholder instead of `cudaGetDeviceProperties`
- Memory deallocation not implemented

This epic tracks production GPU infrastructure to deliver:
- Real CUDA device queries (`cudaGetDeviceProperties`, `cudaMemGetInfo`)
- Tensor core and mixed precision capability detection
- GPU memory allocation/deallocation lifecycle
- Device compatibility validation for inference workloads

### Scope

**In Scope**:
- Real CUDA device discovery (no env var simulation)
- Tensor core capability detection from compute capability
- Mixed precision support detection (FP16/BF16)
- GPU memory queries and lifecycle management
- Device selection logic for optimal performance
- Device compatibility validation (compute capability, memory)

**Out of Scope**:
- Multi-GPU support (single device for MVP+1)
- GPU utilization monitoring (deferred to server observability epic)
- CUDA kernel optimization (separate performance work)

### Consolidated Issues

This epic consolidates the following issues:
- #355: AC2 Device Discovery Failure in Multi-Head Attention Tests
- #361: Replace stub supports_tensor_cores with real hardware capability detection
- #362: Implement Real Device Selection Logic in get_optimal_device Method
- #363: Implement Real GPU Discovery and Memory Detection - Replace Environment Variable Simulation
- #364: Replace stub mixed precision detection with real CUDA capability checking (duplicate)
- #365: Implement Device Compatibility Validation for Production Inference
- #366: Replace hardcoded 8GB placeholder with actual CUDA memory query in GpuMemoryManager::new
- #367: Implement Real GPU Memory Deallocation in GpuMemoryManager
- #406: Implement intelligent device configuration in ProductionModelLoader::get_optimal_device_config

**Total**: 9 issues

### Acceptance Criteria

- [ ] **Real GPU Discovery**: Use `cudaGetDeviceCount` and `cudaGetDeviceProperties` (no env var simulation)
- [ ] **Tensor Core Detection**: Query compute capability and determine tensor core support
- [ ] **Mixed Precision Detection**: Detect FP16/BF16 support from compute capability (≥5.3 for FP16, ≥8.0 for BF16)
- [ ] **Memory Queries**: Use `cudaMemGetInfo` for available/total GPU memory
- [ ] **Memory Lifecycle**: Implement `cudaMalloc` and `cudaFree` in `GpuMemoryManager`
- [ ] **Device Selection**: Intelligent device selection based on capability, memory, and workload
- [ ] **Compatibility Validation**: Validate device meets minimum requirements (compute capability ≥6.0)
- [ ] **Testing**: GPU preflight tests validate detection and selection logic
- [ ] **Documentation**: GPU architecture documentation in `docs/gpu-kernel-architecture.md`

### Implementation Tasks

**Phase 1: Real CUDA Device Discovery**
- [ ] Replace `BITNET_GPU_FAKE` simulation with `cudaGetDeviceCount`
- [ ] Query device properties with `cudaGetDeviceProperties`
- [ ] Parse compute capability (major, minor) for feature detection
- [ ] Add device enumeration and selection logic

**Phase 2: Capability Detection**
- [ ] Implement tensor core detection (compute capability ≥7.0)
- [ ] Implement FP16 support detection (compute capability ≥5.3)
- [ ] Implement BF16 support detection (compute capability ≥8.0)
- [ ] Add mixed precision capability struct

**Phase 3: Memory Management**
- [ ] Replace 8GB placeholder with `cudaMemGetInfo` query
- [ ] Implement `cudaMalloc` wrapper in `GpuMemoryManager`
- [ ] Implement `cudaFree` and deallocation lifecycle
- [ ] Add memory usage tracking and leak detection

**Phase 4: Device Selection & Validation**
- [ ] Implement `get_optimal_device` with capability-based scoring
- [ ] Add device compatibility validation (minimum compute 6.0)
- [ ] Add memory requirement validation for model sizes
- [ ] Implement graceful CPU fallback when GPU insufficient

**Phase 5: Testing & Integration**
- [ ] Add GPU preflight tests (existing command: `cargo run -p xtask -- gpu-preflight`)
- [ ] Test device discovery across GPU generations (Pascal, Volta, Turing, Ampere, Ada)
- [ ] Test memory lifecycle (allocate, use, deallocate, no leaks)
- [ ] Update GPU development documentation

### Performance Targets

- **Device discovery overhead**: <50ms for full capability detection
- **Memory query overhead**: <10ms for `cudaMemGetInfo`
- **Allocation/deallocation latency**: <100ms for typical model sizes

### Dependencies

- CUDA Toolkit 11.0+ (for modern CUDA APIs)
- cuBLAS/cuDNN (optional, for mixed precision optimizations)
- GPU with compute capability ≥6.0 (minimum supported)

### Related Work

- **GPU Preflight Command**: `cargo run -p xtask -- gpu-preflight` validates GPU compilation and runtime
- **Device-Aware Quantization**: Existing pattern for CPU/GPU selection (extend to real detection)
- **Feature Gate Unification**: PR #440 unified `gpu` and `cuda` feature predicates

### References

- `crates/bitnet-kernels/src/device_manager.rs` - Device manager implementation
- `crates/bitnet-kernels/src/gpu_memory_manager.rs` - GPU memory management
- `crates/bitnet-kernels/src/device_features.rs` - Device capability detection
- `docs/gpu-kernel-architecture.md` - GPU architecture documentation
- `docs/GPU_SETUP.md` - GPU configuration guide
- External: [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html)

---

## Epic 4: Server Production Observability

**Title**: `[EPIC] Server Production Observability - Health Endpoints & Metrics`

**Labels**: `epic`, `area/server`, `priority/low`, `milestone/v0.3.0`

**Description**:

### Summary

Production-grade health endpoints, metrics export, and resource management for `bitnet-server` inference server deployment.

### Motivation

Current `bitnet-server` has scaffolded health endpoints and metrics but not production-ready:
- Health endpoints compile but lack real liveness/readiness checks
- Metrics system (OpenTelemetry OTLP) integrated but not comprehensive
- Model loading/unloading lifecycle incomplete
- System resource monitoring (CPU, memory, GPU) not implemented

This epic tracks server production hardening for v0.3.0+ deployment to deliver:
- Liveness and readiness health endpoints with real checks
- Prometheus/OTLP metrics export with inference telemetry
- Model hot-swapping and unload functionality
- System resource monitoring and alerting

### Scope

**In Scope**:
- Liveness and readiness health endpoints (AC5)
- Prometheus/OTLP metrics export (OpenTelemetry)
- Model loading/unloading lifecycle and hot-swapping
- System resource monitoring (CPU, memory, GPU utilization)
- Production deployment documentation

**Out of Scope**:
- Horizontal scaling and load balancing (future work)
- Distributed inference (single-server focus)
- Advanced autoscaling policies (v0.4.0+)

### Consolidated Issues

This epic consolidates the following issues:
- #353: AC5 Health Endpoint Tests Failing Due to Compilation Errors in bitnet-server
- #370: Replace Stub Health Metrics System with Production Implementation
- #371: Missing Model Unload Functionality for Resource Management

**Total**: 3 issues

### Acceptance Criteria

- [ ] **Liveness Endpoint**: `/health/live` returns 200 when server process running
- [ ] **Readiness Endpoint**: `/health/ready` returns 200 when model loaded and inference ready
- [ ] **Metrics Export**: Prometheus/OTLP metrics for inference requests, latency, throughput
- [ ] **Model Lifecycle**: Load, hot-swap, and unload models without server restart
- [ ] **Resource Monitoring**: Track CPU, memory, GPU utilization with configurable thresholds
- [ ] **Testing**: AC5 health endpoint tests pass, integration tests for model lifecycle
- [ ] **Documentation**: Deployment guide in `docs/operations/` with health endpoint usage

### Implementation Tasks

**Phase 1: Health Endpoints (AC5)**
- [ ] Implement liveness endpoint `/health/live` with process status check
- [ ] Implement readiness endpoint `/health/ready` with model load status
- [ ] Add health check dependencies (model state, inference engine status)
- [ ] Fix AC5 health endpoint test compilation errors

**Phase 2: Metrics System**
- [ ] Add inference request counter (total requests, successes, failures)
- [ ] Add inference latency histogram (p50, p95, p99)
- [ ] Add throughput gauge (tokens/sec, requests/sec)
- [ ] Add resource utilization gauges (CPU, memory, GPU)
- [ ] Configure OTLP exporter for Prometheus scraping

**Phase 3: Model Lifecycle**
- [ ] Implement model unload functionality (`/api/unload`)
- [ ] Add model hot-swap without server restart (`/api/load`)
- [ ] Add model state tracking (loaded, unloaded, loading, error)
- [ ] Test model lifecycle transitions (load → unload → reload)

**Phase 4: Resource Monitoring**
- [ ] Replace placeholder memory monitoring with `sysinfo` crate
- [ ] Add CPU utilization tracking (per-core, aggregate)
- [ ] Add GPU utilization tracking (if GPU available)
- [ ] Add configurable alerting thresholds (high memory, low throughput)

**Phase 5: Documentation & Deployment**
- [ ] Write deployment guide (`docs/operations/deployment.md`)
- [ ] Document health endpoints and metrics (Prometheus scraping)
- [ ] Add Docker deployment examples
- [ ] Add Kubernetes deployment manifests (optional)

### Performance Targets

- **Health endpoint latency**: <10ms response time
- **Metrics overhead**: <1% impact on inference throughput
- **Model hot-swap downtime**: <500ms transition time

### Dependencies

- OpenTelemetry OTLP exporter (PR #448 complete)
- `sysinfo` crate for system resource monitoring
- Model state management in `bitnet-server`

### Related Work

- **PR #448**: OpenTelemetry OTLP migration (metrics infrastructure ready)
- **Health Endpoints**: AC5 tests scaffolded in `tests/ac05_health_checks.rs`
- **Server Architecture**: Core inference server in `crates/bitnet-server/src/`

### References

- `crates/bitnet-server/src/health.rs` - Health endpoint implementation
- `crates/bitnet-server/src/metrics.rs` - Metrics collection
- `crates/bitnet-server/src/model_manager.rs` - Model lifecycle management
- `crates/bitnet-server/tests/ac05_health_checks.rs` - AC5 health tests
- `docs/health-endpoints.md` - Health endpoint documentation
- External: [OpenTelemetry Rust](https://github.com/open-telemetry/opentelemetry-rust)

---

## Epic Creation Commands

```bash
# Create Epic 1: TL1/TL2 Production Quantization
gh issue create \
  --title "[EPIC] TL1/TL2 Production Quantization Implementation" \
  --body-file epic_templates.md \
  --label "epic,area/quantization,priority/medium,milestone/v0.2.0"

# Create Epic 2: Tokenizer Production Hardening
gh issue create \
  --title "[EPIC] Tokenizer Production Hardening - GGUF Tokenizer & Mock Cleanup" \
  --body-file epic_templates.md \
  --label "epic,area/tokenization,priority/medium,milestone/v0.2.0"

# Create Epic 3: GPU Device Discovery & Memory Management
gh issue create \
  --title "[EPIC] GPU Device Discovery & Memory Management - Real CUDA Capability Detection" \
  --body-file epic_templates.md \
  --label "epic,area/gpu,priority/medium,milestone/v0.2.0"

# Create Epic 4: Server Production Observability
gh issue create \
  --title "[EPIC] Server Production Observability - Health Endpoints & Metrics" \
  --body-file epic_templates.md \
  --label "epic,area/server,priority/low,milestone/v0.3.0"
```

**Note**: Manually extract each epic's description from this file and create separate issues, or use a script to parse and create them individually.

---

## Post-Epic Consolidation Commands

After creating epics, close consolidated issues and link to epics:

```bash
# Close issues consolidated into Epic 1 (TL1/TL2 Quantization)
gh issue close 346 399 401 416 419 \
  -c "Closed as consolidated into **Epic 1: TL1/TL2 Production Quantization** (#<EPIC1_NUMBER>). Track production table lookup quantization implementation in the epic."

# Close issues consolidated into Epic 2 (Tokenizer Hardening)
gh issue close 381 395 397 398 400 402 404 409 \
  -c "Closed as consolidated into **Epic 2: Tokenizer Production Hardening** (#<EPIC2_NUMBER>). Track GGUF tokenizer implementation and mock cleanup in the epic."

# Close issues consolidated into Epic 3 (GPU Discovery)
gh issue close 355 361 362 363 365 366 367 406 \
  -c "Closed as consolidated into **Epic 3: GPU Device Discovery & Memory Management** (#<EPIC3_NUMBER>). Track real CUDA capability detection and memory lifecycle in the epic."

# Note: Epic 4 issues (#353, #370, #371) kept open as discrete issues for now,
# will be consolidated after epic creation to maintain tracking continuity
```

---

**Total Epics**: 4
**Total Consolidated Issues**: 23
**Remaining Discrete Issues**: 13
**Final Tracking Items**: 17 (4 epics + 13 discrete)

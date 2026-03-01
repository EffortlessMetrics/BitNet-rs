# Changelog

All notable changes to bitnet-rs will be documented in this file.

## [Unreleased]

### Wave 96: Inference Pipeline & Scheduling

- **Inference scheduler** (#1631): 112 tests for task scheduling with priority queues, work stealing, and deadline-aware dispatch
- **Attention mechanism** (#1632): 112 tests for multi-head/grouped-query attention with causal masking and flash-attention support
- **KV cache manager** (#1634): 106 tests for key-value cache lifecycle with paged allocation and eviction policies
- **Activation catalog** (#1640): 133 tests for GELU/SiLU/ReLU/Swish activation functions with fused variants
- **Tensor math primitives** (#1641): 112 tests for element-wise, reduction, and broadcast operations on typed tensors
- **Model architecture** (#1642): 121 tests for transformer block composition with layer normalization and residual connections
- **Runtime config** (#1643): 117 tests for hierarchical configuration with env/file/CLI override resolution
- **DType system** (#1644): 118 tests for numeric type metadata, promotion rules, and cast validation
- **Sampling strategies** (#1646): 112 tests for top-k, top-p, temperature, repetition penalty, and greedy decoding
- **Embedding operations** (#1630): 111 tests for token/positional embedding lookup with RoPE and learned encodings

### Wave 95: GPU HAL Foundation Modules

- **Kernel autotuner** (#1239): 96 tests for runtime kernel parameter search with performance profiling and cache
- **Execution planner** (#1629): 107 tests for compute graph scheduling with operator fusion and memory planning
- **Quantization toolkit** (#1621): 96 tests for weight quantization schemes with calibration and accuracy validation
- **Parallel communication** (#1617): 122 tests for collective operations (all-reduce, scatter, gather) across devices
- **Model serialization** (#1619): 122 tests for GGUF/SafeTensors model I/O with lazy loading and integrity checks
- **Error taxonomy** (#1618): 99 tests for structured error hierarchy with context propagation and recovery hints
- **Device abstraction layer** (#1371): 105 tests for unified device API across CPU/CUDA/Level-Zero/Metal backends
- **Tokenizer pipeline** (#1622): 124 tests for BPE/WordPiece/Unigram tokenization with pre/post-processing steps

### Wave 94: Workspace Integration

- **Workspace integration v17** (#1625): 91 module declarations consolidating new GPU HAL crates into workspace

### Wave 93: Documentation

- **CHANGELOG waves 87-92** (#1616): Documentation update for waves 87-92

### Wave 86: Training Support & Production Infrastructure

### Wave 54: Inference Tracing & Weight Compression

- **Attention mechanism variants** (#1387): 7 attention types (MHA/MQA/GQA/Flash/SlidingWindow/Linear/Sparse), RoPE, ALiBi — 82 tests
- **Model architecture registry** (#1386): `ArchitectureRegistry` with 7 types, 7 known models, memory/compute estimation — 75 tests
- **Inference tracing** (#1388): `InferenceTracer` with Chrome trace export, span hierarchy, trace analysis — 62 tests
- **Weight compression** (#1389): `WeightCompressor` with GPTQ/AWQ/ternary, n-bit pack/unpack, compression analysis — 61 tests
- **CHANGELOG waves 52-53** (#1383): 10 entries documenting PRs #1367–#1382

- **Gradient checkpoint** (#1586): 89 tests for memory-efficient training with activation checkpointing
- **Continuous profiling** (#1362): 112 tests for CPU/GPU/latency tracking and production observability
- **Server protocol** (#1585): 112 tests for OpenAI-compatible API with streaming and tool-use support
- **Workspace v15** (#1580): 127 module stubs consolidating new crates into workspace

### Wave 85: Async Runtime & Distributed Inference

- **Async runtime** (#1575): 96 tests for concurrent task management with structured cancellation
- **Quantization algorithms** (#1581): 106 tests for AWQ/GPTQ/SmoothQuant weight compression
- **Multi-device** (#1582): 106 tests for heterogeneous compute across CPU/GPU devices
- **Docker GPU CI** (#1572): Intel/NVIDIA/CPU Docker images and CI workflow for GPU testing
- **Performance comparison** (#1583): 94 tests for cross-backend benchmarking and regression detection

### Wave 84: HAL Unification & Pipeline Validation

- **HAL traits** (#1571): 113 tests for unified GPU hardware abstraction layer traits
- **Backend selector** (#1573): 90 tests for automatic backend detection and fallback chains
- **SPIR-V compiler** (#1576): 105 tests for shader compilation, optimization, and caching
- **Benchmark harness** (#1577): 97 tests for performance measurement and statistical analysis
- **E2E integration** (#1579): 122 tests for full pipeline validation across all backends

### Wave 83: GPU Backend HAL Suite

- **Level Zero** (#1566): 114 tests for Intel low-level GPU access via oneAPI Level-Zero
- **Metal** (#1564): 112 tests for Apple GPU compute with MSL shader pipeline
- **ROCm** (#1569): 108 tests for AMD HIP backend and device management
- **WebGPU** (#1567): 121 tests for cross-platform wgpu compute on Vulkan/Metal/DX12
- **CUDA HAL** (#1568): 115 tests for NVIDIA CUDA backend with unified HAL interface

### Wave 82: Compute Infrastructure & Model Support

- **Mixed precision** (#1339): 92 tests for FP32/FP16/BF16/INT8 type casting and promotion
- **Thread pool** (#1563): 85 tests for work-stealing executor with configurable parallelism
- **Architecture registry** (#1560): 113 tests for LLaMA/BitNet/GPT2/Mistral model detection
- **Compute graph** (#1562): 94 tests for DAG optimization with dead-code elimination and common subexpression elimination
- **Workspace v14** (#1558): 109 module stubs integrating new infrastructure crates

### Wave 81: Storage, Profiling & Kernel Optimization

- **Memory-mapped I/O** (#1556): 165 tests for zero-copy model loading with LRU page pool
- **Tensor serialization** (#1559): 93 tests for Binary/SafeTensors/NumPy format support
- **GPU memory profiler** (#1554): 87 tests for allocation tracking and leak detection
- **Kernel fusion** (#1555): 86 tests for pattern matching and optimization passes
- **CHANGELOG waves 75-80** (#1549): Documentation update for waves 75-80

### Wave 66-69: Advanced Inference Pipeline & Production Features

- **Inference profiler** (#1455): GPU-HAL inference profiling with layer-level timing, bottleneck detection, and optimization suggestions
- **Shard planner** (#1456): Model sharding planner with balanced partitioning, communication cost estimation, and pipeline schedules
- **CHANGELOG waves 63-65** (#1457): Documentation update for waves 63-65
- **KV compression** (#1458): KV cache compression with quantization, eviction policies, and memory savings tracking
- **Mixture of depths** (#1459): Dynamic depth routing with token-level skip decisions and auxiliary loss
- **Token budget** (#1460): Token budget manager with per-request allocation, borrowing, and throttling
- **Conversation manager** (#1461): Multi-turn conversation management with history, context windows, and summarization
- **Model warmup** (#1462): Model warmup engine with progressive layer warming and JIT compilation cache
- **Safety guardrails** (#1463): Content safety guardrails with toxicity, PII, and prompt injection detection
- **Embedding retrieval** (#1464): Embedding retrieval engine with similarity search, HNSW index, and batch retrieval
- **Workspace integration v11** (#1465): Consolidated 44 module declarations across 8 organized sections
- **RAG engine** (#1466): Retrieval-augmented generation with document chunking, BM25, hybrid retrieval, and reranking
- **Tool use framework** (#1467): Tool use framework with registry, parser, validator, executor, and prompt formatter
- **Priority queue** (#1469): Multi-level priority queue with fair scheduling, deadline-aware ordering, and backpressure
- **Model card generator** (#1470): Model card generator with GGUF metadata extraction, markdown/JSON/HuggingFace renderers
- **Adaptive compute** (#1472): Adaptive computation engine with dynamic precision, early exit, layer skipping, and compute budgets
- **A/B testing** (#1475): A/B testing framework with deterministic traffic splitting and statistical significance testing
- **Checkpoint manager** (#1476): Inference checkpoint manager with incremental diffs, compression, and auto-scheduling
- **Batched tokenization** (#1477): Batched tokenization with parallel encoding, configurable padding/truncation strategies
- **Multi-modal fusion** (#1478): Multi-modal fusion system for text/image/audio with 5 fusion strategies
- **Feature store** (#1479): Feature store with LRU/LFU/FIFO/TTL eviction, memory limits, and versioning
- **Streaming aggregator** (#1480): Streaming output aggregator with chunk ordering, backpressure, and fan-out/fan-in
### Wave 70-74: Production Infrastructure & Core Operations

- **Streaming aggregator** (#1480): Streaming output aggregation with chunk ordering and backpressure handling
- **Inference cache** (#1481): Deterministic inference output caching with prefix matching and LRU eviction
- **Custom ops** (#1482): Custom operator registry with type-safe signatures, fusion patterns, and computation graphs
- **Model compiler** (#1483): Model compiler with IR graph and 5 optimization passes
- **CHANGELOG waves 66-69** (#1484): Documentation update for waves 66-69
- **Deployment manager** (#1485): Model deployment lifecycle with blue/green, canary, and rolling strategies
- **Attention analyzer** (#1489): Attention pattern analysis with sparsity detection and head pruning
- **Hardware tests** (#1492): Cross-backend hardware abstraction test framework with CPU reference implementations
- **Tensor parallelism v2** (#1493): Advanced tensor parallelism with all-reduce, scatter-gather, and collective ops
- **Prompt template** (#1494): Prompt template engine supporting ChatML, Llama2/3, Mistral, Phi3, Zephyr formats
- **Config validator** (#1495): Configuration validation framework for inference, model, device, and server settings
- **Memory allocator v2** (#1496): Advanced GPU memory allocator with buddy, slab, and arena strategies
- **Workspace integration v12** (#1499): Consolidated 66 module declarations across 13 categories
- **Load balancer** (#1500): Request load balancer with circuit breaker and consistent hashing
- **Inference benchmark** (#1501): Inference benchmarking framework with percentile stats and regression detection
- **Token sampler v2** (#1502): Advanced token sampling with logit processor pipeline and min-p/typical-p
- **Semantic search** (#1505): Vector search engine with HNSW and IVF indexing
- **Observability** (#1507): Comprehensive observability with metrics, tracing, structured logging, Prometheus export
- **Rate limiter** (#1508): Rate limiting with token bucket, sliding window, leaky bucket, and per-user quotas
- **Layer norm** (#1512): Layer normalization with RMSNorm, GroupNorm, and fused norm+activation
- **API gateway** (#1514): API gateway with OpenAI-compatible endpoints, auth, and CORS

### Added
- **CHANGELOG wave 59** (#1421): Documents PRs #1409–#1413
- **Sequence parallelism** (#1422): `SequenceDistributor` with boundary handling, all-gather/reduce-scatter, transformer block simulation — 71 tests
- **Communication overlap scheduler** (#1423): `OverlapPlanner` with pipeline bubble analysis, bandwidth estimation (PCIe/NVLink/IB/Ethernet), stream management — 80 tests
- **Watermarking** (#1424): Kirchenbauer-style green/red partitioning, logit biasing, z-score detection, distortion-free variant — 62 tests
- **Gradient accumulation** (#1428): `GradientBuffer` with mixed-precision accumulation, norm/value clipping, distributed all-reduce, variable scheduling — 66 tests
- **Pipeline parallelism** (#1429): GPipe/1F1B/Interleaved schedules, bubble fraction analysis, balanced stage partitioning — 93 tests
- **NUMA allocator** (#1430): `NumaTopology` with GPU affinity, pinned memory pools, locality-aware allocation, transfer optimization — 73 tests
- **Model migration** (#1431): `MigrationPlanner` with BFS path finding, weight transforms (rename/reshape/cast), rollback support — 76 tests
- **Activation recomputation** (#1432): `CheckpointPlanner` with DP knapsack, selective storage, dependency-aware recomputation — 70 tests
- **CHANGELOG wave 60** (#1433): Documents PRs #1416–#1425
- **Expert parallelism** (#1434): All-to-all dispatch/combine, token permutation, capacity management, load balance loss — 73 tests
- **Model validation expansion** (#1138): Shape and distribution checks added to model validation pipeline for stronger load-time correctness guarantees
- **Multi-model GPU serving** (#1127): LRU eviction policy for multi-model GPU serving enables concurrent model hosting with automatic memory management
- **GPU error code mapping** (#1132): Comprehensive GPU error code mapping in `bitnet-common` provides unified error types across CUDA, OpenCL, Vulkan, ROCm, and Metal backends
- **Tokenizer error handling** (#1131): Improved error handling and edge case coverage in `bitnet-tokenizers` for more robust tokenization
- **GPU telemetry and monitoring** (#1124): GPU telemetry system tracks utilization, memory, temperature, and kernel timing for observability
- **Tensor parallelism** (#1125): Multi-device tensor parallelism splits layers across GPU devices for large-model inference
- **GPU-to-GPU peer transfer** (#1126): OpenCL peer-to-peer memory transfer between GPU devices eliminates host-side round-trips
- **ONNX export with GPU metadata** (#1122): ONNX export path now embeds GPU acceleration metadata for interop with external inference runtimes
- **GPU power management** (#1121): GPU power management integration in `bitnet-device-probe` for thermal throttling awareness and power-state queries
- **GpuBatchScheduler** (#1119): Multi-request GPU batching scheduler coalesces inference requests for higher throughput
- **OpenCL USM support** (#1117): OpenCL 3.0 Unified Shared Memory (USM) support eliminates explicit buffer copies on supported devices
- **GPU health endpoint** (#1116): GPU health check REST endpoint in `bitnet-server` reports device status, memory, and thermal state
- **gpu-info subcommand** (#1115): New `gpu-info` CLI subcommand for device discovery and capability reporting
- **kernel-bench subcommand** (#1118): New `kernel-bench` CLI subcommand for micro-benchmarking individual kernel implementations
- **FP16/INT8 mixed-precision kernels** (#1114): OpenCL FP16/INT8 mixed-precision kernel variants for reduced memory bandwidth on supported hardware
- **Persistent kernel cache** (#1113): OpenCL persistent kernel compilation cache avoids recompilation across runs
- **bitnet-metal microcrate** (#1112): Metal compute backend microcrate with MSL shaders for macOS/Apple Silicon GPU inference
- **bitnet-rocm microcrate** (#1111): AMD ROCm GPU backend microcrate with HIP runtime integration
- **bitnet-webgpu microcrate** (#1108): Cross-platform GPU backend via `wgpu` for WebGPU and native Vulkan/Metal/DX12 targets
- **bitnet-level-zero microcrate** (#1085): Intel Level-Zero backend microcrate for low-level Intel GPU access
- **bitnet-vulkan microcrate** (#1071): Vulkan compute backend microcrate with SPIR-V shader pipeline
- **GPU server integration** (#1128): GPU backends wired into the axum HTTP server for accelerated inference serving
- **Intel subgroup kernels** (#1087): Intel subgroup-optimized OpenCL kernels tuned for Arc GPU EU architecture
- **WASM kernel validation shim** (#1084): OpenCL WASM-compatible kernel validation shim enables browser-side correctness checks
- **Multi-device work scheduler** (#1083): Inference work scheduler distributes computation across heterogeneous GPU devices
- **Vulkan tiled matmul** (#1082): Tiled matrix multiplication compute shader for Vulkan backend with shared-memory optimization
- **GPU transformer orchestration** (#1081): Complete GPU transformer layer orchestration in OpenCL — attention, FFN, and LayerNorm fused pipeline
- **GPU activation kernels** (#1080): OpenCL SiLU and GELU activation function kernels for GPU-resident forward passes
- **GPU linear projection** (#1079): OpenCL GPU linear projection kernels for attention Q/K/V and FFN layers
- **GPU embedding lookup** (#1078): OpenCL GPU embedding lookup kernels for token-to-vector conversion on device
- **Work-group auto-tuning** (#1077): OpenCL work-group size auto-tuning profiles device limits for optimal occupancy
- **GPU memory enforcement** (#1076): OpenCL GPU memory limit enforcement and telemetry prevents OOM and tracks allocation watermarks
- **GPU KV cache kernels** (#1075): OpenCL GPU KV cache management kernels for on-device cache append and retrieval
- **OpenCL kernel profiling** (#1058): Event-based kernel profiling in OpenCL captures per-kernel timing for performance analysis
- **Receipt schema extension** (#1090): Receipt schema extended with OpenCL device info, kernel timing, and backend metadata
- **CUDA kernel scaffolding** (#1092): CUDA kernel scaffolding for QK256 GEMV, attention, and RMSNorm operations
- **HIP kernel stubs** (#1102): ROCm HIP kernel stubs for QK256 GEMV, attention, and RMSNorm mirroring CUDA scaffolding
- **CPU kernel gap fill** (#1095): CPU kernel gaps filled and dead orphan source files removed for cleaner kernel registry
- **GGUF-to-GPU tensor loader** (#1096): GGUF-to-GPU tensor loader for Intel Arc enables direct model loading to GPU memory
- **Production-ready general LLMs** (#1049): Extended bitnet-rs infrastructure for production-ready general LLM support beyond BitNet models
- `feat(gpu): add initial ROCm feature wiring and build/link support` — ROCm (AMD GPU) backend skeleton with feature gate, build-system integration, and link support for HIP runtime (#1014)
- `feat(gpu): add initial support for Metal backend and detection` — Metal GPU backend with runtime detection for macOS/Apple Silicon; capability wired into DeviceProbe (#1021)
- `feat(kernels): wire QK256 dispatch to canonical AVX2/scalar path` — QK256 GEMV dispatch now routes through the canonical AVX2/scalar runtime path; AVX2 CI coverage added (#1018)
- `feat(transformer): add strict LayerNorm bias enforcement toggle` — New toggle to enforce or reject LayerNorm bias tensors during model load for stricter validation (#1013)
- `feat(npu): add initial NPU backend scaffolding for kernels, inference, and CLI` — NPU backend skeleton wired across kernels, inference engine, and CLI device selection (#1008)
- `feat(bdd-grid): add Metal, Vulkan, oneAPI backend cells to BDD grid` — Three new BDD grid cells covering Metal (EndToEnd/Local), Vulkan (Minimal/PreProduction), and Intel oneAPI (Development/PreProduction) backends (#1010)

### Changed
- **Nix flake OpenCL deps** (#1066): OpenCL development dependencies added to Nix flake for reproducible GPU development environments
- `chore: prepare the workspace for crates.io release` — Workspace metadata, manifests, and dependency versions updated for crates.io publishing readiness (#1028)
- `chore: prepare v0.2.0 release` — Version bumped to 0.2.0; release notes consolidated (#1006)
- `ci: expand nightly fuzz schedule to all 34 fuzz targets` — Nightly CI fuzz schedule now covers all available fuzz targets (up from 7); timeboxed 5-minute runs per target (#1004)
- `docs: update backend roadmap and architecture docs for v0.2` — Updated dual-backend roadmap and architecture documentation for post-v0.2 state (#1001)

### Fixed
- **Snapshot kernel provider count** (#1089): Updated kernel provider count 2→3 in snapshots after Metal backend addition
- `fix: handle .contiguous() panics on block-quantized tensors during transpose` — Prevents panics when calling `.contiguous()` on block-quantized tensors that require transpose (#1048)
- `fix: input validation falsely rejecting CRLF sequences` — Input validation no longer rejects valid payloads containing CRLF (`\r\n`) line endings (#1030)
- `fix: improve Model Download System` — Model download reliability improvements including better error handling, retry logic, and progress reporting (#1027)
- `fix(config): accept 'npu' as valid device identifier in CLI and server` — NPU device identifier (`npu`) now accepted as a valid device string in CLI and server config (#1002)

### Performance
- **AVX2 QK256 GEMV optimization** (#1105): SIMD unpack and 4-wide FMA tiling on QK256 GEMV hot paths for improved throughput on x86-64
- **NEON kernel enhancements** (#1103): Enhanced ARM64 NEON kernel implementations for improved performance on Apple Silicon and ARM servers
- **OpenCL tiled matmul** (#1060): Tiled matrix multiplication kernel with local memory optimization for OpenCL GPU backends
- `perf: optimize softmax_in_place for sparse distributions` — Faster softmax computation for sparse logit distributions by skipping negligible values (#1031)

### Testing
- **GPU vs CPU cross-validation** (#1120): GPU vs CPU cross-validation test suite verifies numerical equivalence across backends
- **Numerical stability tests** (#1123): 40-test numerical stability suite covering edge cases in kernels (NaN, Inf, denormals, large tensors)
- **GPU stress tests** (#1107): GPU stress tests for reliability validation under sustained load and memory pressure
- **GPU snapshot regression** (#1106): Comprehensive GPU kernel output regression snapshots via insta for deterministic output tracking
- **GPU feature flag matrix** (#1104): GPU feature flag compatibility matrix tests verify correct compilation across feature combinations
- **TDD scaffold enablement** (#1097): TDD scaffolds enabled and ignored test count further reduced
- **Snapshot tests expansion** (#1094): Insta snapshot tests added for `bitnet-feature-matrix` and `bitnet-test-support` crates
- **Property tests expansion** (#1091): Property tests added for additional crates strengthening invariant coverage
- **BDD multi-backend scenarios** (#1086): Comprehensive BDD scenarios for multi-backend device selection and fallback behavior
- **Fuzz targets: kernel/KV/config/sampling/tensors** (#1088): Five new fuzz targets covering kernel dispatch, KV cache ops, config parsing, sampling strategies, and tensor operations
- **Fuzz targets: attention/softmax/rmsnorm** (#1070): Three new fuzz targets for attention, softmax, and RMSNorm numerical stability
- **OpenCL property tests** (#1069): Extended property tests for OpenCL kernel CPU reference implementations
- **OpenCL e2e pipeline tests** (#1055): End-to-end OpenCL inference pipeline tests validate full forward pass on GPU
- **Criterion benchmarks: OpenCL** (#1073): Criterion benchmarks for OpenCL kernel CPU reference implementations
- **Criterion benchmarks: quantization/kernels** (#1101): Expanded Criterion benchmarks covering quantization algorithms and kernel operations
- `test: reduce ignored test count (wave 7) — enable unblocked tests` — Wave 7 reduction of `#[ignore]` tests; newly unblocked tests converted to active (#1025)
- `test(bitnet-device-probe): add property tests for new backend capabilities` — Property tests covering ROCm, Metal, Vulkan, NPU, and oneAPI backend capability invariants (#1003)

### CI
- **ROCm/HIP smoke workflow** (#1130): CI smoke workflow for ROCm/HIP compilation validates AMD GPU backend builds
- **Fuzz target matrix completion** (#1099): Fuzz-nightly and fuzz-ci workflow matrices updated to include all fuzz targets
- **Mutation testing: OpenCL** (#1056): Mutation testing configuration added for OpenCL modules via `cargo-mutants`
- **Intel GPU Docker image** (#1065): Docker image with Intel compute runtime for GPU CI jobs

### Documentation
- **GPU ADRs** (#1109): Architecture Decision Records documenting GPU backend selection rationale and trade-offs
- **GPU contributor guide** (#1110): GPU development contributor guide covering kernel authoring, testing, and review workflow
- **Intel GPU tuning guide** (#1068): Intel GPU performance tuning guide with Arc-specific optimization recommendations
- **Multi-GPU architecture docs** (#1053): Multi-GPU backend architecture documentation covering device management and scheduling
- **Dual-backend roadmap update** (#1054): Dual-backend roadmap updated with Intel GPU work and OpenCL integration milestones
- **CHANGELOG update wave** (#1098): CHANGELOG entries added for PRs #1003–#1048
- `docs: clarify BitNet.cpp vs llama.cpp cross-validation relationship` — Clarified the distinct roles of BitNet.cpp and llama.cpp in the dual-backend cross-validation architecture (#1026)
- `docs: add general LLM production readiness plan for BitNet-rs` — Production readiness roadmap covering reliability, scalability, and deployment considerations (#1012)
- `docs: add macOS 26 Apple Silicon integration roadmap` — Roadmap for macOS 26 integration with Apple Silicon GPU/NPU acceleration (#1007)
- `docs: add Apple Silicon GPU/NPU backend roadmap` — Detailed backend roadmap for Metal GPU and Core ML NPU acceleration on Apple Silicon (#1005)
- `docs: add canonical CUDA GPU setup guide` — Comprehensive CUDA GPU setup guide with corrected build command examples and environment configuration (#998)

## [0.2.0] - 2026-02-27

### Highlights
- 1200+ comprehensive tests across all microcrates (BDD, unit, property, integration, fuzz)
- Complete microcrate SRP extraction (bitnet-sampling, bitnet-transformer, bitnet-receipts, bitnet-prompt-templates, bitnet-device-probe, bitnet-logits, bitnet-generation, bitnet-engine-core)
- Feature lattice normalized: gpu/cuda correctly orthogonal
- Kernel registry with capability detection
- Modern Diataxis documentation structure

### Added
- `feat(kernels): implement AVX-512 kernels for TL2 quantization` — New AVX-512 optimised matrix-kernel for TL2 2-bit quantization; runtime-dispatched alongside existing AVX2 and scalar paths (#997)
- `feat(vulkan): add initial Vulkan runtime probing and feature wiring` — Initial Vulkan backend probe and feature gate wiring; detects Vulkan ICD at runtime and exposes capability via `DeviceProbe` (#993)
- `feat(metal): enable and wire Metal backend support across workspace` — Metal GPU backend enabled and wired across workspace feature flags; capability detection, kernel dispatch, and device-probe integration (#992)
- `feat(kernels): enable AVX-512 kernel selection and harden TL2 quantization buffers` — AVX-512 kernel selected at runtime when `avx512f` is available; TL2 buffer bounds hardened to prevent overflows (#989)
- `feat(kernels): improve NEON kernel implementation and configuration safety` — NEON kernel path improved with safer configuration; alignment guarantees and bounds validated on AArch64 (#988)
- `feat(oneapi): add initial Intel oneAPI backend support and runtime detection` — New Intel oneAPI backend skeleton with runtime detection for Intel GPGPU; `oneapi` feature gate wired in workspace (#986)
- `feat(cli): accept OpenCL/Vulkan device aliases and map to GPU backend` — CLI now accepts `opencl` and `vulkan` device strings, mapping them to the GPU backend for device selection (#985)
- `feat(device-probe): add OpenGL probing and capability wiring for GPU backend detection` — OpenGL capability probing added to `DeviceProbe`; detected OpenGL level exposed as GPU backend capability (#984)
- `test: add CPU golden path E2E test infrastructure` — New E2E golden path test infrastructure for CPU inference running deterministic scenarios end-to-end without a real model download (#949)
- `test: reduce ignored test count (wave 4) — enable unblocked tests` — Wave 4 reduction of `#[ignore]` tests; newly unblocked tests converted to active following resolution of blockers #254 and #260; remaining `#[ignore]` markers carry justification strings (#947)
- `feat: improve C# FFI bindings with modern P/Invoke` — Modern P/Invoke bindings and safe wrappers for C# consumers; replaces legacy `DllImport` patterns with `LibraryImport`-based source-generated P/Invoke for reduced marshalling overhead (#942)
- test(bitnet-bdd-grid,sampling,receipts,generation): expand BDD scenario coverage with 80+ scenario tests across four microcrates (#945)
- test(bitnet-sys,bitnet-compat): add targeted unit tests for capability invariants and export path validation (#946)
- `test: add insta snapshot tests for key serializable types` — 2 snapshot tests using `insta` crate pinning `RuntimeFeatureFlags` and related serializable types; added `INSTA_UPDATE: unseen` to CI workflow to prevent unreviewed snapshot acceptance (#938)
- `feat(fuzz): add wave 4 fuzz targets` — 4 new fuzz targets: `engine_core_no_panic` (engine-core API panic safety), `honest_compute_no_panic` (honest-compute pipeline panic safety), `runtime_context_parse_no_panic` (runtime context parsing panic safety), `feature_activation_no_panic` (feature activation logic panic safety) (#937)
- `test(bitnet-tokenizers,bitnet-validation): add comprehensive unit tests` — 45 tests for `bitnet-tokenizers` (TokenizerConfig defaults, BasicTokenizer construction, BOS/EOS/PAD semantics, family detection, builder profiles, property tests) and 37 tests for `bitnet-validation` (Ruleset defaults, projection RMS bounds, architecture detection, policy loading edge cases, property tests) (#934)
- `test(bitnet-runtime-feature-flags,bitnet-kernels): add comprehensive unit tests` — 33 tests for `bitnet-runtime-feature-flags` (feature snapshot determinism, CPU/GPU independence, JSON roundtrips, lattice implications) and 47 tests for `bitnet-kernels` (KernelManager construction, provider listing, FallbackKernel numeric correctness, dimension validation, device features API) (#932)
- `test(integration): add multi-crate integration tests` — 69 integration tests in `tests/integration/multi_crate_tests.rs` spanning 8 microcrates (sampling, logits, device-probe, prompt-templates, generation, engine-core, GGUF, honest-compute); added 8 new path dependencies to `tests/Cargo.toml` (#931)
- `test(bitnet-device-probe,bitnet-logits): add comprehensive unit tests` — 15 tests for `bitnet-device-probe` (SIMD level ordering, probe consistency, cross-probe invariants) and 26 tests for `bitnet-logits` (top-p, repetition penalty, argmax, temperature, softmax, top-k, property tests) (#929)
- `test(bitnet-generation,bitnet-engine-core): add comprehensive unit tests` — 24 tests for `bitnet-generation` (stop criteria, streaming events, serde roundtrips, property tests) and 36 tests for `bitnet-engine-core` (session lifecycle, backend info, concurrency config, engine state, error variants, session ID uniqueness) (#928)
- `test(bitnet-cli,bitnet-gguf): add comprehensive unit tests` — 55 tests for `bitnet-cli` (InferenceCommand defaults, PromptTemplate parsing, TemplateType detection, CliConfig validation, ConfigBuilder, property tests) and 40 tests for `bitnet-gguf` (GGUF value types, metadata, header parsing, constants, property tests) (#926)
- `test(bitnet-inference,bitnet-models): add extended engine and model tests` — 43 tests for `bitnet-inference` (SamplingConfig, SamplingStrategy, GenerationConfig, InferenceReceipt, ModelInfo, ProductionInferenceEngine, streaming, error handling) and 43 tests for `bitnet-models` (ModelMetadata, QuantizationType, BitNetConfig, LoadConfig, ProductionModelLoader, tensor operations) (#924)
- `test: reduce ignored test count (wave 3)` — reduced ignored test count; added justification strings to all remaining `#[ignore]` markers; added env-var guards for environment-sensitive tests (#925)
- `test(bitnet-st2gguf,bitnet-server): add extended tests` — 26 extended tests for bitnet-st2gguf (conversion pipeline, quantization type handling, error variants) and 20 tests for bitnet-server (batch engine config, CORS, security middleware) (#922)
- `test(bitnet-transformer,bitnet-honest-compute): add comprehensive tests` — 29 tests for bitnet-transformer (RoPE invariants, KVCache shape/seq-len properties) and 37 tests for bitnet-honest-compute (ComputeMode gating, receipt validation, honest-compute enforcement) (#921)
- `test(bitnet-bdd-grid,bitnet-testing-policy): add comprehensive grid and policy tests` — 36 tests for bitnet-bdd-grid (grid rows/columns, cell lookup, scenario validation) and 28 tests for bitnet-testing-policy (PolicyDiagnostics coherence, GridCompatibility invariants) (#917)
- `feat(fuzz): add wave 3 fuzz targets` — 3 new fuzz targets: `gguf_writer_roundtrip` (GGUF writer round-trip safety), `tokenizer_encode_no_panic` (tokenizer encode panic safety), `validation_no_panic` (validation pipeline panic safety) (#919)
- `test(bitnet-trace): add 20+ comprehensive tests` — 20+ tests covering TraceSink and TraceComparison APIs, JSON round-trips, hash invariants, rms calculations, and decode-step tracing (#916)
- `test(bitnet-validation,bitnet-kernel-registry): add comprehensive tests` — 65 tests for bitnet-validation (LayerNorm bounds, error messages, policy keys, gate modes, RMS validation) and 56 tests for bitnet-kernel-registry (kernel dispatch, capability registry, SIMD selection behaviour) (#915)
- `test(bitnet-quantization): add 25+ extended integration and property tests` — 25+ extended integration and property tests covering I2_S, TL1, TL2, and QK256 roundtrip accuracy, block-size invariants, and edge-case inputs (#914)
- `test(bitnet-device-probe): add 25+ comprehensive tests` — SIMD level detection ordering, GPU probe validation, DeviceCapabilities construction/equality, and property tests for probe determinism and capability invariants (#912)
- `test(bitnet-runtime-feature-flags,bitnet-feature-contract): add comprehensive tests` — RuntimeFeatureFlags snapshot tests, CPU-always-true invariant, CUDA/GPU orthogonality, and JSON round-trips for feature-flag and feature-contract crates (#911)
- `test: reduce ignored test count by converting to env-var guards (wave 2)` — 13 fewer ignored tests (1043→1030); `tokenization_smoke` converted to `CROSSVAL_GGUF` opt-in; AC3/AC6 acceptance tests gated on `BITNET_RUN_SLOW_TESTS` env-var (#910)
- `test(bitnet-compat,bitnet-sys): add 72 comprehensive tests` — bitnet-compat: `diagnose`, `export_fixed`, `verify_idempotent`, `stamp` paths; bitnet-sys: `CompileTimeLibCapabilities` and disabled-feature stubs (#909)
- `test(bitnet-server): add 43 server security and middleware tests` — 7 CORS, 5 security header, 9 request validation, 4 model path, 3 config/auth, 4 IP extraction, 8 property tests, 1 health endpoint tests; uses `tower::ServiceExt::oneshot` (no real server needed) (#907)
- `test(bitnet-tokenizers): add 45 comprehensive property-based tests` — 40 proptest + 5 integration tests covering encode/decode round-trips, vocab consistency, special tokens bounds, Unicode safety, config JSON round-trip, auto-discovery, batch encoding (#906)
- `feat(xtask): add grid-check command for BDD grid compile verification` — `xtask grid-check [--dry-run]` checks all supported feature cells compile; 18 cells including cpu, cpu+full-cli, cpu+crossval, cpu+gpu (#905)
- `test: add E2E golden path test with synthetic GGUF fixture` — 5 deterministic tests using in-memory `GgufWriter` fixture (< 2 KB, no model download); tests header parsing, metadata round-trip, full load result, receipt invariants (#904)
- `feat(fuzz): add sampling, receipt, and prompt-template fuzz targets` — 3 new fuzz targets: `sampling_no_panic`, `receipt_json_roundtrip`, `prompt_template_no_panic` (#903)
- `test(bitnet-logits): add 24 comprehensive property-based tests` — softmax, log-softmax, repetition penalty, temperature, top-k, top-p, numerical stability tests (#902)
- `test: remove #[ignore] from 10 self-skipping tests` — 10 tests with env-var guards converted from `#[ignore]` to active tests (#901)
- `test(bitnet-engine-core): add 20 tests for state transitions, config validation, and session IDs` — 20 new tests covering `EngineState` transitions, `EngineConfig` validation invariants, and session ID uniqueness/format in `crates/bitnet-engine-core/tests/`; raises engine-core test coverage (#899)
- `test(bitnet-prompt-templates): add 9 property-based tests` — 9 new proptests covering template rendering determinism, format invariants, and stop-sequence emission in `crates/bitnet-templates/tests/prompt_template_proptests.rs` (#898)
- `test(bitnet-generation): add 10 property tests for Unicode safety and long context` — 10 new property tests validating UTF-8 safety of generated output and correct behaviour under long-context (>2 048 token) inputs in `crates/bitnet-generation/tests/unicode_and_long_context_proptests.rs` (#897)
- `chore: prepare v0.1.3 release notes and changelog` — consolidated release notes for v0.1.3 into `CHANGELOG.md`; updated `Cargo.toml` workspace version to `0.1.3` and pinned dependency versions for the release (#896)
- `test(bitnet-models): add 15 property-based tests for model loading` — 15 new proptests in `crates/bitnet-models/tests/model_load_proptests.rs` covering `ModelConfig` round-trips, tensor-shape invariants, quantization-type parsing edge cases, and QK256 vs BitNet32-F16 flavor disambiguation (#895)
- `test(bitnet-inference): add 77 integration tests that run without a real model` — 77 new no-model integration tests in `crates/bitnet-inference/tests/` exercising `ProductionInferenceEngine` construction, `SamplingStrategy` configuration paths, `GenerationStream` lifecycle, and error-handling branches without requiring a GGUF file (#894)
- `test(bitnet-cli): add snapshot tests for CLI help output` — 8 new insta snapshot tests in `crates/bitnet-cli/tests/snapshot_tests.rs` pinning `--help` and `--version` output for all top-level subcommands; uses `assert_cmd` + `insta` for regression detection (#893)

### Documentation
- `docs: update changelog for PRs #931-#932` — Updated CHANGELOG.md and roadmap with 1400+ tests milestone (#935)
- `docs: comprehensive README rewrite and test-suite docs update` — README rewrite highlighting 1000+ test milestone, updated feature flag table, xtask grid-check documentation, and revised `docs/development/test-suite.md` with current test counts (#920)
- `docs: update changelog for PRs #905-#912` — Changelog entries for server security tests, tokenizer property tests, xtask grid-check feature, E2E golden path tests, fuzz targets (sampling/receipt/template), logits property tests, and device-probe tests (#913)

### Changed
- scripts: improve crates.io readiness validation with comprehensive checks (#940)
- `chore: bump version to 0.2.0` — All crates and workspace version bumped from `0.1.x` to `0.2.0`; `CHANGELOG.md` `[Unreleased]` section promoted to `[0.2.0]` with release date (#933)

### Fixed
- `ffi: preserve exact C last-error messages` — FFI layer now captures and preserves exact error strings returned by C `last_error` callbacks rather than truncating or reformatting them; improves debuggability for C/C++ consumer integrations (#941)

## [v0.1.3] - 2026-02-27

Test coverage expansion wave 2: property tests across 10+ crates, new fuzz targets, BDD-style kernel tests, CLI snapshot tests, CI hardening, and integration tests for quantization, device-probe, logits, transformer, and tokenizer.

### Added
- `test(bitnet-receipts): add property-based tests for receipt validation` — 5 new property tests (20 total) covering validation gate invariants in `crates/bitnet-receipts/tests/`; raises receipt proptest coverage (#891)
- `test(bitnet-sampling): add comprehensive property-based tests` — 14 new property tests for sampling invariants in `crates/bitnet-sampling/tests/`; covers temperature, top-k/p, repetition penalty, and greedy determinism (#890)
- `feat(fuzz): add gguf_header_parse fuzz target` — new `fuzz/fuzz_targets/gguf_header_parse.rs` fuzz target exercising GGUF header parsing paths for robustness (#889)
- `test(bitnet-kernels): add BDD-style kernel tests` — 30 new tests across 7 sections in `crates/bitnet-kernels/tests/`; validates kernel dispatch, capability registry, and SIMD selection behaviour (#888)
- `test(bitnet-common): add property-based tests` — 39 property tests across Device, Config, errors, and math invariants in `crates/bitnet-common/tests/`; broadens foundational crate proptest coverage (#887)
- `test(bitnet-gguf): add property-based tests` — 20 new property tests in `crates/bitnet-gguf/tests/extended_proptests.rs`; brings total gguf proptest count to 103 (#886)
- `test(server,ffi): add property-based tests to bitnet-server and bitnet-ffi` — additional property tests for `bitnet-server` (batch/concurrency/security config invariants) and `bitnet-ffi` (C API round-trips); further raises proptest coverage across both crates (#880)
- `test(tokenizers): expand tokenizer_proptests.rs to 18 property tests` — expanded from 7 to 18 property tests covering edge cases in encode/decode round-trips, special-token handling, vocab-size bounds, and determinism invariants in `crates/bitnet-tokenizers/tests/tokenizer_proptests.rs` (#874)
- `test(device-probe,logits): add integration tests` — integration tests for `bitnet-device-probe` (SimdLevel ordering, `probe_device` smoke test) and `bitnet-logits` (softmax, top-k, temperature, argmax invariants) (#876)
- `test(snapshot): add snapshot tests wave 2 for generation and engine-core` — insta snapshot tests pinning `StopReason` display variants, `GenerationStats` fields, and receipt schema version string for regression detection (#877)
- `feat(fuzz): add generation_stop_checker fuzz target` — new fuzz target exercising `check_stop` from `bitnet-generation`; covers stop-token, stop-sequence, max-tokens, and combined stopping criteria (#870)
- `test(bitnet-transformer): add 8 property-based tests` — 8 property tests for RoPE and KV cache invariants in `crates/bitnet-transformer/tests/transformer_proptests.rs` (#871)
- `test(quantization): add integration tests for quantization roundtrip` — 10 integration tests for I2S, TL1, and TL2 quantization roundtrip accuracy in `crates/bitnet-quantization/tests/quant_integration_tests.rs` (#873)
- `test(inference): add property-based tests for inference invariants` — 20 property tests in `crates/bitnet-inference/tests/inference_proptests.rs` covering SamplingConfig, GenerationConfig, stop tokens, and receipt schema invariants (#861)
- `feat(device-probe): extract bitnet-device-probe SRP microcrate` — new `crates/bitnet-device-probe/` SRP microcrate exporting SimdLevel, CpuProbe, DeviceProbe, probe_device(); 5 property tests (#862)
- `feat(logits): extract bitnet-logits SRP microcrate` — named unit tests in `crates/bitnet-logits/tests/logits_tests.rs` for softmax, top-k, temperature, and argmax invariants (#863)
- `test(e2e): add CPU golden path integration test` — extended CPU E2E golden path tests with 2 new always-on tests (#864)
- `test: add property tests for bitnet-models` — 10 property tests in `crates/bitnet-models/tests/model_load_proptests.rs` covering config roundtrip, shape invariants, quantization type parsing, and flavor detection (#859)
- `fix(ci): increase grid-check timeout to 20min; deduplicate cargo check calls` — BDD Grid Check timeout raised to 20 minutes; duplicate `cargo check` calls removed to speed up CI (#859)
- `test: add property tests for runtime microcrates` — 51+ property tests across 6 thin façade crates: bitnet-feature-matrix, bitnet-runtime-bootstrap, bitnet-runtime-context, bitnet-startup-contract, bitnet-testing-policy, bitnet-testing-scenarios; plus sampling and tokenizer proptests (#857)
- `test: add property tests for bitnet-compat and bitnet-kernels` — 14 new proptests: 8 in `crates/bitnet-compat/tests/compat_proptests.rs` and 6 in `crates/bitnet-kernels/tests/kernel_proptests.rs` (#856)
- `test: add insta snapshot tests for receipts and prompt templates` — added `crates/bitnet-common/tests/snapshot_tests.rs` with insta snapshot test pinning `BitNetConfig::default()` for regression detection (#855)
- `test(receipts): add kernel ID hygiene property tests` — 7 property tests in `crates/bitnet-receipts/tests/receipt_validation_tests.rs` covering kernel ID hygiene invariants (#852)
- `feat(fuzz): add gguf_header_roundtrip structured fuzz target` — new `fuzz/fuzz_targets/gguf_header_roundtrip.rs` structured round-trip fuzz target (#853)
- `feat(xtask): add grid-check command` — new `cargo xtask grid-check` command providing a BDD compile-coverage gate with 18 cells and `cpu_only` mode; verifies grid contract at compile time (#850)
- `test(bitnet-engine-core): add 6 property tests for clone, field, and seed invariants` — 6 new proptests covering clone/field/seed invariants in `crates/bitnet-engine-core/tests/property_tests.rs` (#849)
- `test: audit and reduce ignored tests` — reduced ignored test count from 433 → 423 (−10) by enabling tests no longer blocked (#847)
- `ci: add nightly fuzz run workflow` — scheduled fuzz job running at 2am UTC, 60s per target across all 10 fuzz targets, with crash artifact upload on failure (#844)
- `test(bitnet-generation): add 6 new property tests for stopping invariants` — 6 new proptests covering stop-token, stop-sequence, max-tokens, and combined stopping criteria in `crates/bitnet-generation/tests/property_tests.rs` (#846)
- `feat: add bitnet-logits property tests` — 4 new proptests for logits invariants in `crates/bitnet-logits/tests/logits_tests.rs` (#843)
- `feat(wasm): add copy-to-clipboard button for generated text in browser example` — clipboard UX improvement in WASM browser example (#809)
- `test: expand proptest coverage for compat, templates, and feature-flag crates` — 18 new proptests across bitnet-compat, bitnet-templates, and bitnet-runtime-feature-flags (#841)

### Fixed
- `fix(ci): pass PR branch refs through env to prevent script injection` — security hardening in `.github/workflows/pr-size-guard.yml`: PR branch refs are now passed via environment variables instead of being interpolated directly into shell commands, eliminating a potential script-injection vector (#885)
- `fix(ci): fix property-tests and performance-tracking workflows` — added push/PR triggers and a `cargo-tests` smoke job to the property-tests and performance-tracking CI workflows so they gate correctly on PRs (#879)
- `fix(ci): add --force to cargo install cargo-fuzz` — prevents fuzz CI failures when the `cargo-fuzz` binary is already present in the runner cache (#872)
- `fix(server): implement rate limiter cleanup to prevent memory leak` — `ConcurrencyManager::cleanup_rate_limiters` now properly cleans up idle entries to prevent unbounded memory growth (#810)

### Performance
- `perf(logits): optimize apply_top_p with sparse filtering` — skip zero-probability tokens before sorting, reducing unnecessary work in nucleus sampling (#811)
- `ci: increase fuzz build timeout to 60 minutes` — fuzz CI job timeout raised to 60 min, improved caching, added `RUSTFLAGS=-C debuginfo=0` to speed up fuzz builds (#840)
- `test: add numerical accuracy integration tests for bitnet-quantization` — 6 new integration tests (I2S dequantize, TL1 LUT, TL2 symmetry, round-trip accuracy, QK256 block size, zero vector) (#838)
- `test: add CPU golden path E2E validation tests` — 4 new E2E tests (stop token, receipt kernel IDs, schema version, max_tokens boundary) (#837)

## [v0.1.2] - 2026-02-27

Test coverage expansion wave (proptests, fuzz targets, BDD grid), reduced ignored tests to 77.

### Fixed
- `chore: fix stale MSRV cache key in compatibility workflow (1.89 → 1.92)` — prevents incorrect CI cache hits from cached 1.89 toolchain artifacts (#805)

### Added
- `test: expand proptest coverage for thin-coverage crates` — 15 new property tests across bitnet-bdd-grid, bitnet-honest-compute, bitnet-rope, and bitnet-trace (#834)
- `test: add snapshot tests for sampling, gguf, and receipts` — 5 new insta snapshots pinning GgufFileInfo display output and receipt schema version string for regression detection (#833)
- `test: audit and reduce ignored tests` — reduced ignored test count from 91 → 77 by enabling tests that are no longer blocked (#831)
- `test: expand server and CLI integration tests` — 18 new integration tests covering health endpoint, CORS, security validation, and CLI template parsing (#830)
- `feat(fuzz): add transformer_config and gguf_kv_read fuzz targets` — 2 new fuzz targets covering TransformerConfig deserialization and GGUF KV key/value read paths (#829)
- `test(bdd): expand BDD grid with new scenario cells` — 5 new BDD cells (18 total, was 13) in `crates/bitnet-bdd-grid/src/lib.rs` (#828)
- `test: proptest coverage for bitnet-common` — 9 new property tests in `crates/bitnet-common/tests/property_tests.rs` (MockTensor shapes, QuantizationType round-trips, error types, device consistency, KernelCapabilities ordering, warn_once deduplication) (#826)
- `test: expand proptest coverage for bitnet-sampling` — 7 new property tests in `crates/bitnet-sampling/tests/property_tests.rs` (temperature scaling, top-k/p filtering, repetition penalty, seed reproducibility, empty context, greedy determinism) (#825)
- `test: proptest coverage for bitnet-receipts` — 14 new proptests in `crates/bitnet-receipts/tests/property_tests.rs` (schema version, builder, JSON round-trips, kernel ID validation, honest compute gates, token counts) (#823)
- `test: proptest coverage for bitnet-validation` — 8 new proptests in `crates/bitnet-validation/tests/property_tests.rs` (LayerNorm bounds, error messages, policy keys, gate modes) (#822)
- `test: proptest coverage for bitnet-models` — 13 new proptests in `crates/bitnet-models/tests/property_tests.rs` (ModelConfig validation, GgufTensorType element sizes, GGUF magic bytes, path safety) (#821)
- `feat(fuzz): add quantization_input fuzz target` — new `fuzz/fuzz_targets/quantization_input.rs` covering I2S/TL1/TL2 dequantize, QK256 parsing, gemv_qk256_row, unpack/code_to_f32 (#820)
- `test: expand property test coverage for bitnet-tokenizers` — 8 new proptests in `crates/bitnet-tokenizers/tests/property_tests.rs` (token encoding bounds, special tokens, vocab size, encode-decode consistency, config validation, round-trip determinism) (#819)
- `feat(fuzz): add safetensors parser fuzz targets` — `fuzz/fuzz_targets/safetensors_metadata.rs` targets `SafeTensors::read_metadata()` header path; `fuzz/fuzz_targets/safetensors_parser.rs` replaced original stub; `fuzz/Cargo.toml` updated (#813)
- `test(bdd): expand BDD grid coverage` — 5 new BDD grid cells added in `crates/bitnet-bdd-grid/src/lib.rs` (Unit/Ci, Integration/Ci, Performance/Local, Development/Local, Smoke/Ci); grid snapshot count updated from 8 to 13 (#814)
- `test: add proptest coverage for bitnet-ffi` — `crates/bitnet-ffi/tests/property_tests.rs` with 31 tests (25 proptest + 6 unit): BitNetCConfig/BitNetCInferenceConfig round-trips, BitNetCError display invariants, MemoryStats arithmetic, thread-local error state (#815)
- `test: add proptest coverage for bitnet-logits and bitnet-generation` — temperature scaling, softmax invariants, top-k filtering, repetition penalty (bitnet-logits); stop criteria, token accumulation, streaming order (bitnet-generation) (#806)
- `feat(fuzz): add tokenizer_encode_decode fuzz target` — covers BasicTokenizer, UniversalTokenizer, wrapper tokenizers, and HfTokenizer BPE round-trips (#807)
- `test: add 22 proptest cases for bitnet-quantization` — TL1/TL2/I2_S round-trip bounded error, scale positivity, block alignment, and edge cases (all-zeros, alternating signs) (#808)

## [v0.1.1] - 2026-02-26

### Security
- Restrict model loading to configured directories via `BITNET_ALLOWED_MODEL_DIRECTORIES` (#753)
- Harden model path validation: prevent symlink traversal and empty-string allowlist bypass (#756)

### Changed
- Project renamed from BitNet.rs to BitNet-rs throughout (1,531 files, 6,281 occurrences) (#755)
- `refactor(quantization): dead code cleanup` — Removed unused `KernelProvider` imports and unused fields from `bitnet-quantization` (#779)

### Added
- `ci: add fuzz/** to ci-core.yml path triggers` — Added `fuzz/**` glob to the `paths` filter in `ci-core.yml` so fuzz PRs get required CI checks (#803)
- `test(integration): expand SRP cross-crate integration tests` — Expanded `srp_integration_test.rs` to 22 tests (10 new), covering bitnet-logits→bitnet-sampling pipeline, bitnet-generation stop criteria, RoPE tables, bitnet-device-probe determinism, bitnet-engine-core session config (#802)
- `test: add InferenceReceipt::to_json_string() convenience method + snapshot test` — Added `to_json_string()` on `InferenceReceipt`; snapshot test pins receipt JSON output for regression detection (#801)
- `test: add 21 proptest cases for bitnet-gguf and 7 for bitnet-sys` — `bitnet-gguf`: 21 properties covering `GgufValue`, `TensorInfo`, `GgufMetadataKv` round-trips and invariants; `bitnet-sys`: 7 properties for `CompileTimeLibCapabilities` summary logic (#800)
- `feat(fuzz): add gguf_metadata_values fuzz target` — New `fuzz/fuzz_targets/gguf_metadata_values.rs` for GGUF parser panic safety; exercises arbitrary metadata value sequences (#799)
- `chore: release v0.1.1` — Version bump 0.1.0 → 0.1.1 across workspace; `Cargo.lock` regenerated (#798)
- `chore: GitHub repo settings update` — Updated `.github/settings.yml` description and topics; added `.github/settings.yml` to `ci-core.yml` path triggers (#794)
- `chore: docs update batch #790-#791` — Updated `CHANGELOG.md` and `CLAUDE.md` for PRs #790 (E2E golden-path tests) and #791 (README modernization) (#793)
- `feat(fuzz): BPE tokenizer encode fuzz target (re-create)` — Recreated `fuzz/fuzz_targets/tokenizer_encode.rs` for BPE encode/decode paths with 4 exercise paths; fuzz total remains 15 (#792)
- `feat(fuzz): RoPE table generation fuzz target` — `fuzz/fuzz_targets/rope_table_gen.rs` using `arbitrary::Arbitrary`; verifies sin²+cos²≈1 (Pythagorean) invariant across arbitrary dimensions and base values (#783)
- `test(transformer): 5 new KVCache/config property tests` — shape invariants after N appends, layer independence, layer count validation, head divisibility check, seq_len monotonicity (#784)
- `test(tokenizers): 5 new property tests for encode/decode` — BOS/EOS prepend behaviour, decode never panics, tokenize preserves words, config serde round-trip, EOS ID bounds (#785)
- `docs(api): Examples/Errors/Panics sections for SRP crate APIs` — documentation improvements across `bitnet-logits`, `bitnet-generation`, `bitnet-engine-core`, `bitnet-sampling`, `bitnet-device-probe` (#786)
- `bench: criterion benchmarks for SRP ops (srp_ops.rs)` — 6 benchmark functions: logits pipeline, top-k at k=5/k=50, repetition penalty, argmax, RoPE build_tables, KV cache append (#787)
- `feat(fuzz): BPE tokenizer encode fuzz target` — `fuzz/fuzz_targets/tokenizer_encode.rs` with 4 exercise paths (empty, ASCII, Unicode, max-length boundary); fuzz total: 13→15 (#788)
- `test(e2e): reproducibility and pinned-output golden-path tests` — Added `crates/bitnet-inference/tests/e2e_cpu_golden_path.rs` with 2 deterministic E2E tests: `test_e2e_golden_path_reproducible` (seed=42, same seed gives identical tokens) and `test_e2e_golden_path_pinned_output` (pins greedy-argmax tokens [140,459,459,459] as regression guard); no model download required (#790)
- `docs: modernize README to well-designed FOSS format` — Rewrote README.md: added Rust 2024 edition badge, Features bullet list, new architecture diagram, Feature flags table; removed verbose receipt verification section; trimmed to ~90 lines net (#791)
- `ci: add BDD grid-check job to CI Core workflow` — Standalone `grid-check` job in `ci-core.yml` runs `xtask grid-check --cpu-only` in parallel with the build matrix (#772)
- `chore: docs update for PRs #765–#771` — CHANGELOG and CLAUDE.md updated to reflect merged PRs (#773)
- `test(sampling): expand proptest coverage for bitnet-sampling` — 7 new proptests covering top_k, repetition_penalty, temperature entropy, multi-step, and reset invariants (#774)
- `feat(ci): nightly scheduled fuzz workflow with corpus caching` — New `.github/workflows/nightly-fuzz.yml`; runs 7 fuzz targets for 60 s nightly, caches corpus per-target, uploads crash artifacts (#775)
- `feat(inference): wire bitnet-logits into bitnet-inference (SRP integration)` — Replaced duplicate logits math in `generation/sampling.rs` with `bitnet-logits` crate (#776)
- `feat(ci): CUDA smoke lane weekly schedule with receipt upload` — `gpu-smoke.yml` updated with weekly schedule and receipt artifact upload (#777)
- `feat(inference)`: BackendStartupSummary — startup logs `requested=X detected=[…] selected=Y` (#771)
- `test(srp-crates): expanded proptest coverage for bitnet-logits, bitnet-generation, bitnet-engine-core (#768)`
- `test(gguf): expanded property tests, snapshot tests, and unit tests for bitnet-gguf — 33 → 49 tests (#767)`
- **`bitnet-device-probe` microcrate — `CpuCapabilities`/`GpuCapabilities` with proptest coverage** (#765)
- **CPU golden-path E2E test with receipt invariants** (#766)
- **Property tests for 6 infrastructure crates** (PR #762): Extends proptest coverage to 6 additional crates. Proptest workspace total: **50 crates**.
- Keyboard navigation (ArrowLeft/ArrowRight/Home/End) for WASM browser example tab list (#754)
- **Property tests for `bitnet-test-support`** (PR #749): 10 new property + unit tests for `EnvGuard`/`EnvScope` API semantics — set/restore/remove round-trips, nested scope isolation, `model_path()`/`run_slow_tests()`/`run_e2e()` env helpers. Workspace total: **3,520 tests, all passing**. Proptest coverage spans **38 crates** (+2: `bitnet-test-support`, `bitnet-testing-scenarios-profile-core`).
- **Property tests for `bitnet-testing-scenarios-profile-core`** (PR #750): 17 new property + unit tests for Default value invariants across 5 structs — `FixtureProfile`, `CrossValidationProfile`, `ComparisonToleranceProfile`, `ReportingProfile`, `ResourceConstraints`. Includes fuzz-grade shape coverage for numeric fields, URL/path fields, and nested struct coherence.
- **CI env isolation fixes for `bitnet-runtime-context-core` and `bitnet-runtime-profile-contract-core`** (in PR #746): Tests that check `from_env_with_defaults()` Local default now use `temp_env::with_vars` to clear the `CI` env var so they pass in GitHub Actions. Snapshot tests for `active_context_default_fields` also isolated. `test_config_profile_defaults` snapshot uses `insta::with_settings!` filter to normalize the CPU-dependent `max_parallel_tests` field. Added `filters` feature to workspace insta definition.
- **CI coverage for BDD/policy/testing-infra crates** (PR #746): Added 19 BDD/policy/testing-infrastructure crates to the Build & Test matrix in `ci-core.yml`. These crates (including `bitnet-bdd-grid`, `bitnet-feature-contract`, `bitnet-testing-policy-core`, `bitnet-runtime-profile-contract-core`, etc.) had full test suites but were excluded from CI. All tests run with `--no-default-features --features cpu` to satisfy profile snapshot requirements.
- **Property tests for `bitnet-testing-policy-tests`** (PR #745): 8 new property/unit tests for `PolicyDiagnostics` invariants — `from_context_is_deterministic`, `is_grid_compatible_coherent_with_violations` (key: `is_grid_compatible() ↔ violations().is_some_and(|m,f| m.is_empty() && f.is_empty())`), `summary_never_panics_and_contains_scenario`, `feature_contract_consistent_does_not_panic`, `diagnostics_for_context_matches_from_context`, `profile_config_does_not_panic`, plus 2 unit tests (unit/local and e2e/ci). Workspace total: **3,493 tests, all passing**. Proptest coverage spans **36 crates**.

### Fixed
- **TL2 kernel fix** (PR #761): Replaced `matmul_i2s` with a `dequantize+matmul` pipeline (same pattern as TL1 fix in #760), eliminating 3 compounding bugs that caused incorrect TL2 quantization results.
- **TL1 kernel fix** (PR #760): Replaced `matmul_i2s` with a `dequantize+matmul` pipeline, fixing 3 compounding bugs that caused incorrect TL1 quantization results.
- **xtask `find_bitnet_lib_dirs` race condition** (PR #748): `test_find_bitnet_lib_dirs_both_tiers` was failing intermittently because `test_find_bitnet_lib_dirs_env_override` used a nested-function anti-pattern for `#[serial(bitnet_env)]` — the inner function's attribute doesn't actually serialize the outer `#[test]`. Fixed by adding `#[serial(bitnet_env)]` directly to all 5 related tests and flattening the nested-function pattern.


- **Docs update for PRs #739-#741** (PR #742): Updated test counts to 3,472, fuzz targets to 13, proptest crates to 33 across `CLAUDE.md`, `docs/development/test-suite.md`, `CHANGELOG.md`, `docs/reference/dual-backend-roadmap.md`.
- **Fuzz targets for `bitnet-logits` and `bitnet-generation`; proptest for `bitnet-runtime-context-core`** (PR #740): 2 new fuzz targets (bringing total to **13**) + 8 new property tests. `logits_transforms` fuzzes all 6 public functions in `bitnet-logits` with arbitrary f32 data including NaN/infinity — invariants: no panic, `argmax` returns valid index, `apply_top_k` count ≤ len, `softmax_in_place` outputs are non-negative and finite. `generation_stop_check` fuzzes `check_stop()` in `bitnet-generation` with arbitrary `StopCriteria`, token IDs, and decoded tail text — invariant: never panics. `bitnet-runtime-context-core` (+8 proptest/unit): `TestingScenario`/`ExecutionEnvironment` Display→FromStr round-trips (all variants), `from_env_with_defaults` default precedence (serial+temp-env isolation), parse-error-not-panic on garbage strings. Workspace total: **3,472 tests, all passing**. Proptest coverage spans **33 crates**.
- **Property tests for `bitnet-sys` and `bitnet-st-tools`** (PR #738): 18 new proptest properties across 2 previously uncovered crates. `bitnet-sys` (+10): `CompileTimeLibCapabilities` implication invariants (`has_cuda ⇒ available`, `has_bitnet_shim ⇒ available`), `summary()` canonical key presence/determinism, clone equality, and `cpp=available`/`cpp=unavailable` token correctness. `bitnet-st-tools` (+8): `is_ln_gamma()` fast-path (`non-.weight` suffix always returns `false`), known LN name matching, projection name rejection, determinism, and no-panic on arbitrary Unicode input. Workspace total: **3,464 tests, all passing**. Proptest coverage spans **32 crates**.
- **First-inference tutorial** (PR #735): Adds `docs/tutorials/first-inference.md` — a step-by-step guide covering model loading, greedy vs. temperature sampling, interactive chat, and performance notes. Fixes the broken link in `docs/README.md` and expands the Diataxis navigation index with 11 previously-unlisted guide links.
- **Property tests for `bitnet-common`, `bitnet-tokenizers`, `bitnet-inference`** (PR #732): 42 new proptest properties. `bitnet-common` (+17): Device ordering, JSON round-trips, `BackendSelection::select_backend(Cuda, cpu_only_caps)` always returns Err. `bitnet-tokenizers` (+9): `TokenizerConfig` field bounds, `BasicTokenizer` round-trip. `bitnet-inference` (+16): `GenerationConfig` builder invariants, stop-token semantics, `validate()` correctness. Workspace total: **3,426 tests, all passing**. Proptest coverage spans **29 crates**.
- **Property tests for `bitnet-quantization` and `bitnet-models`** (PR #733): 20 new proptest properties. `bitnet-quantization` (+12): `qk256_tolerance_bytes` monotonicity/floor/proportionality, `calculate_scale` positive-finite, `pack_2bit_values`/`unpack_2bit_values` round-trip, `quantize_value`/`dequantize_value` order-preservation. `bitnet-models` (+8): `is_layernorm_weight`/`is_projection_weight` mutual exclusivity, re-export identity. Workspace total: **3,446 tests, all passing**. Proptest coverage spans **30 crates**.
- **Property tests for `bitnet-trace`, `bitnet-kernels`, `bitnet-server`, and `bitnet-cli`** (PRs #724–#726): Extends proptest coverage to the four remaining high-value crates — `bitnet-trace` (+6 tests: JSON round-trip, name non-empty, `num_elements` = shape product, blake3 = 64 hex chars, rms ≥ 0.0, optional fields omitted when None); `bitnet-kernels` (+8 tests: always-has-providers, non-empty provider names, stable selection, CPU kernel available, TL1/I2S quantize output sizes, `gpu_compiled()` constancy); `bitnet-server` (+11 tests: `BatchEngineConfig` max_batch_size/concurrent batches positive, `RequestPriority` transitive ordering, `BatchRequest` builder invariants, `SecurityConfig` field bounds, `SecurityValidator` prompt length enforcement, temperature/top_p range validation); `bitnet-cli` (+9 tests: max_tokens positive, all three aliases equivalent, temperature/repetition_penalty parse range, top_k/top_p option handling, greedy flag, u64 seed precision). Workspace total: **3,384 tests, all passing**. Proptest coverage now spans **26 crates**.

### Fixed
- **Config doctest calling renamed method** (PR #736): `bitnet-inference/src/config.rs` module-level doctest called `with_max_new_tokens()` which was renamed to `with_max_tokens()`. All 16 doctests now pass.
- **CLI tests asserting stderr vs stdout** (PR #730): Error messages logged via `tracing::error!` go to stderr (when using `tracing-subscriber`), not stdout. `inspect_ln_stats.rs` and `validation_workflow.rs` now use `.stderr(...)` assertions.
- **Workspace snapshot tests (4 tests across 3 crates)** (PR #720): Replaced exact-count/exact-value snapshot assertions with presence checks in `bitnet-runtime-feature-flags` (`feature_labels_count_with_cpu_feature`, `feature_line_format_stable`), `bitnet-startup-contract-core` (`cli_component_observe_is_compatible_or_has_state`), and `bitnet-testing-policy-kit` (`active_feature_labels_returns_list`). Cargo feature unification in workspace builds activates extra features (`fixtures`, `reporting`, `trend`, `quantization`) via other crates, making context-dependent exact-match snapshots fail. Full workspace run now: **3,359 passed, 0 failed, 462 skipped**.
- **Duplicate workspace members in `Cargo.toml`** (PR #741): `bitnet-device-probe`, `bitnet-logits`, `bitnet-generation`, `bitnet-engine-core`, and `bitnet-gguf` were each listed twice in `[workspace.members]`. Cargo deduplicates silently, so there was no functional impact, but the duplicates were confusing. Removed the redundant entries from the phase-6 block at the bottom of the list.

### Documentation
- **Dual-Backend Roadmap update** (PR #719): Marks PRs #711–#717 as implemented in the roadmap tracking table; adds retrospective rows for previously-implemented but un-tracked items (Phase 6 SRP microcrates, BDD grid runner, CPU golden path, GPU smoke lane).

### Added
- **CLAUDE.md Test Count and Category Update** (PR #717): Updates test count from 970+ to 2,082+, skipped count from ~466 to ~462, and adds snapshot tests (37 files, 200+ assertions) and property tests (20 files, 100+ properties) to the Working Test Categories section.
- **`docs/development/test-suite.md` Snapshot/Property/Fuzz Update** (PR #716): Adds dedicated sections for Snapshot Tests (insta, 37 files, 200+ assertions, run/review/update commands), Property Tests (proptest, 20 files, PROPTEST_CASES env, key invariants), and Fuzz Testing (11 targets table, corpus location, nightly CI schedule); updates test counts and category table.
- **Snapshot Tests for 5 Thin Wrapper/Policy-Facade Crates** (PR #712): Adds `insta` to dev-deps and creates `snapshot_tests.rs` for — `bitnet-runtime-profile-contract-core` (4 snapshots: canonical grid row count, active profile summary format, Unit/Local has cell, Unit/Local no GPU required), `bitnet-testing-policy-contract` (3 snapshots: `PolicyContract::detect()`, feature consistency, drift_check semantics), `bitnet-testing-policy-interop` (3 snapshots: `Environment` alias, grid row count, validate helper), `bitnet-testing-policy-tests` (3 snapshots: `GridScenario`/`GridEnvironment` aliases, `PolicyDiagnostics` cell presence), `bitnet-testing-profile` (2 snapshots: identity conversion helpers, grid row count). Adds 15 snapshot tests total.
- **`bitnet-st2gguf` Snapshot Tests** (PR #711): Adds `insta` to dev-deps and creates `snapshot_tests.rs` for `bitnet-st2gguf` — 6 snapshots: `ConversionOptions` defaults, `ConversionOptions` strict mode, `QuantizationType` I2_S format string, `QuantizationType` TL2 format string, `ConversionError` missing file message, `ConversionError` unsupported format message. Catches silent default drift in the SafeTensors→GGUF conversion pipeline.
- **Markdownlint Workflow npm Resilience** (PR #714): Adds `|| echo ...` fallback on npm install step and `command -v markdownlint-cli2 &&` guard on the lint invocation so transient npm 403 errors no longer fail documentation PRs.
- **Snapshot Tests for 5 Runtime/Policy Microcrates** (PR #710): Adds `insta` to dev-deps and creates `snapshot_tests.rs` for — `bitnet-runtime-context-core` (3 snapshots: `ActiveContext` scenario/environment Display strings for Unit/Local and Integration/Ci defaults, all TestingScenario variants), `bitnet-startup-contract-diagnostics` (3 snapshots: `StartupContractReport::profile_summary()` format, `info` line count, `warnings` count for compatible contract), `bitnet-startup-contract-guard` (3 snapshots: `RuntimeComponent` label strings, `is_compatible()` with Observe policy, `feature_line` prefix invariant), `bitnet-runtime-feature-flags` (3 snapshots: `feature_line()` prefix + full output, `feature_labels()` count without features), `bitnet-testing-scenarios-profile-core` (7 snapshots: `ConfigurationContext::default()`, `ReportingProfile` default formats, `FixtureProfile`/`ComparisonToleranceProfile`/`CrossValidationProfile`/`TestConfigProfile` defaults, `ReportFormat` variants Debug strings). Adds 19 snapshot tests total.
- **Snapshot Tests for `bitnet-inference`, `bitnet-kernels`, `bitnet-models`, `bitnet-server`** (PR #709): Adds `insta` to dev-deps and creates `snapshot_tests.rs` for four high-value large crates — `bitnet-inference` (7 snapshots: `GenerationConfig` default fields, greedy/creative constructors, `InferenceConfig` defaults, validation error message), `bitnet-kernels` (4 snapshots: `KernelManager` provider count, fallback presence, selection returns Ok, name non-empty, behind `cpu` feature), `bitnet-models` (5 snapshots: `LoadConfig` defaults, `ProductionLoadConfig` defaults + max size, `DeviceStrategy` Debug formats), `bitnet-server` (6 snapshots: `ServerSettings` host/port/timeouts/device, `BatchEngineConfig` batch size, `ConcurrencyConfig` max requests, `DeviceConfig` variants). Adds 22 snapshot tests total.
- **Snapshot Tests for 5 BDD Infrastructure Microcrates** (PR #708): Adds `insta` to dev-deps and creates `snapshot_tests.rs` for — `bitnet-runtime-feature-flags-core` (4 snapshots: `to_labels()` for CPU/GPU+CUDA/empty, active feature set labels), `bitnet-testing-scenarios-core` (4 snapshots: all scenario descriptions, Unit timeout, CI log level, scenario count), `bitnet-startup-contract-core` (3 snapshots: `RuntimeComponent.label()` for all variants, summary structure tokens, `is_compatible()` value), `bitnet-feature-contract` (4 snapshots: consistent/inconsistent contract snapshots, empty input), `bitnet-testing-policy-core` (3 snapshots: unit/local summary format, `active_profile_summary()`, integration/CI snapshot). Adds 18 snapshot tests total.
- **Snapshot Tests for `bitnet-quantization` and `bitnet-cli`** (PR #707): Adds `insta` to dev-deps and creates `snapshot_tests.rs` for two high-value crates — `bitnet-quantization` (5 snapshots: `QuantizationType` Display/Debug format strings, `best_for_arch()` platform-specific selection for x86_64/aarch64, and `validate_round_trip()` boolean results for I2S and TL2 with known inputs), `bitnet-cli` (5 snapshots gated behind `full-cli`: `InferenceCommand` field defaults, help text sections for `--max-tokens`+aliases, `--stop`+aliases, sampling flags, and `--prompt-template`). Catches silent default drift and alias removal regressions.
- **Snapshot Tests for `bitnet-trace`, `bitnet-compat`, `bitnet-bdd-grid-core`** (PR #706): Adds `insta` to dev-deps and creates `snapshot_tests.rs` for three more microcrates — `bitnet-trace` (4 JSON snapshots pinning `TraceRecord` serialization format: minimal, with optional fields, decode-step, logits), `bitnet-compat` (2 snapshots pinning `GgufCompatibilityFixer::diagnose()` diagnostic message strings and issue count for a minimal GGUF), `bitnet-bdd-grid-core` (6 snapshots pinning `TestingScenario`/`ExecutionEnvironment`/`BitnetFeature` Display strings, `FeatureSet::labels()`, `BddCell` debug format, `BddGrid::validate()` None return). Adds 12 snapshot tests total across the three crates.
- **`bitnet-bdd-grid` Proptest and Snapshot Tests** (PR #705): Adds `proptest` and `insta` dev-deps to `bitnet-bdd-grid` and creates `property_tests.rs` (8 tests: 5 proptest properties for LazyLock stability, required∩forbidden always disjoint, `find()` consistent with `rows()`, `supports()`/`violations()` semantics, all intents non-empty; 3 unit tests: cell count == 8, Unit/Local present, Smoke/PreProduction present) and `snapshot_tests.rs` (4 snapshots pinning the full grid summary, Unit/Local cell features, EndToEnd/CI cell features, and total cell count).
- **Snapshot Tests for `bitnet-receipts`, `bitnet-sampling`, `bitnet-prompt-templates`** (PR #703): Adds dedicated `snapshot_tests.rs` files for three key SRP microcrates — `bitnet-receipts` (6 insta JSON snapshots: cpu_real_receipt, mock_receipt, receipt_with_backend_summary, receipt_with_model_info, receipt_with_performance, receipt_with_test_results), `bitnet-sampling` (6 insta debug snapshots: default/greedy/creative config, greedy_sample output, seeded sample seed=42, repetition penalty before/after context), `bitnet-prompt-templates` (7 insta snapshots: raw_simple, instruct_no_system, instruct_with_system_prompt, llama3_single_turn, llama3_with_system, instruct_multi_turn, llama3_multi_turn). Adds 19 snapshot tests total; previously only 1 snapshot existed in receipts and 2 in prompt-templates.
- **`bitnet-device-probe` Proptest Integration Tests** (PR #701): `crates/bitnet-device-probe/tests/property_tests.rs` (7 proptest tests) — covers `simd_level_display_never_empty` / `simd_level_consistent_across_calls` (determinism invariants), `gpu_compiled_is_stable` (compile-time constant must not vary), `device_capabilities_cpu_rust_always_true`, `device_capabilities_cuda_compiled_matches_gpu_compiled`, `gpu_fake_cuda_returns_true` / `gpu_fake_none_returns_false` (BITNET_GPU_FAKE env-var override semantics), and BITNET_STRICT_MODE interaction tests. Completes proptest coverage for all SRP microcrates in the workspace.
- **`bitnet-transformer` KVCache Snapshot Tests** (PR #702): `crates/bitnet-transformer/tests/snapshot_tests.rs` (5 insta snapshots) — pins `KVCache` and `LayerKVCache` structural invariants: initial state (num_layers, seq_len=0, max_seq_len, n_kv_heads), post-append (only target layer seq_len increments), post-clear (all layers reset), layer-initial-state, and 3-append seq_len accumulation. First snapshot coverage for this crate.
- **GGUF `open()` File Integration Tests** (PR #699): `crates/bitnet-gguf/tests/file_open_tests.rs` (4 tests) — exercises the real mmap-backed `open()` path using the committed 224-byte `tests/models/mini.gguf` synthetic fixture; covers happy-path (version=3, tensor_count=0, metadata_count=4), insta snapshot, nonexistent-path error, and wrong-magic-bytes error. No model download required.
- **Backend Selection Property + Snapshot Tests** (PR #698): `crates/bitnet-common/tests/backend_selection_tests.rs` (13 tests, no feature gate) — 5 proptest properties for `BackendRequest::Display` invariants, `summary()` format contracts, `select_backend()` determinism; 6 unit tests for specific selection scenarios; 2 insta snapshots pinning the summary string and `BackendSelectionResult` Debug format
- **Property Tests for 6 SRP Microcrates** (PR #696): Integration-level proptest coverage for `bitnet-testing-policy-core` (7 tests: PolicySnapshot invariants), `bitnet-testing-scenarios-core` (8 tests: Display↔FromStr round-trips + CI contract), `bitnet-honest-compute` (11 property + 5 unit: mock detection, classification, kernel ID hygiene), `bitnet-rope` (7 property + 4 unit: Pythagorean identity, dimension invariants, error paths), `bitnet-validation` (10 property + 4 unit: LN name detection, ruleset bounds), `bitnet-feature-contract` (5 property + 3 unit: set-diff semantics, consistency invariants) — adds ~45 tests
- **Property Tests for `bitnet-bdd-grid-core`** (PR #694): 10 proptest properties — `TestingScenario`/`ExecutionEnvironment`/`BitnetFeature` Display→FromStr round-trips (all variants lossless); `FeatureSet` insert→contains, superset satisfaction, labels completeness; plus 4 unit error-case tests
- **Property Tests for `bitnet-runtime-feature-flags-core`** (PR #692): 6 proptest properties verifying `FeatureActivation → FeatureSet` conversion invariants — `cuda ⇒ gpu`, `cpu ⇒ inference+kernels+tokenizers`, `feature_line` prefix, default activation empty labels, and 2 unit stability tests
- **Property Tests for `bitnet-startup-contract-core`** (PR #691): 4 proptest properties verifying `ProfileContract` invariants — context round-trips, summary non-empty, `Observe` policy never fails `enforce()`, `is_compatible()` consistent with `state()`; plus 1 unit stability test (5 total)
- **Property Tests for `bitnet-transformer`** (PR #689): 4 proptest properties covering KVCache invariants — seq_len tracks N append operations correctly, clear/reset semantics, overflow (capacity) rejection, and initial zero state
- **Tracing Instrumentation for Template Detection** (PR #686): `bitnet-prompt-templates::TemplateType::detect()` now emits `debug!` on each branch (Llama3Chat, Instruct, Raw) and `warn!` on the Raw fallback; in-crate `#[traced_test]` unit tests verify log capture
- **`tracing-test` 0.2.6 Added to Workspace** (PR #686): Shared dev-dep for crates that test tracing output

### Fixed
- **`bitnet-runtime-feature-flags` Snapshot Test Names and Values** (PR #715): Renamed `feature_labels_without_features_is_empty` → `feature_labels_count_with_cpu_feature` (the old name was incorrect — tests always run with `--features cpu`); updated snapshot values to reflect 4 compiled features (`cpu`, `inference`, `kernels`, `tokenizers`).
- **Env-var race in `bitnet-startup-contract-core`** (PR #691): Replaced `unsafe { env::set_var/remove_var }` with `temp_env::with_var + #[serial(bitnet_env)]` in `evaluate_preserves_context_overrides` test
- **4 Previously-Ignored Tests Unblocked** (PR #686): `test_once_per_layer_warning_guards`, `test_kv_cache_warning_message_format` (unique layer indices avoid `Once` state collisions), `test_detection_logs_decision`, `test_fallback_logs_warning` (converted to behavioral assertions; log coverage in emitting crate)
- **Fixture Timeout: 5-min → ~1s** (PR #687): Reduced huge synthetic fixture allocations in `bitnet-quantization` — vocab fixtures (50257→512), large projection (1024×4096→512×1024), GGUF model layers (realistic dims→2×32) reduced fixture allocation from >2GB to <5MB total

### Added (prior)
- **Property Tests for `bitnet-logits`** (PR #683): 13 proptest properties verifying softmax sum-to-one / non-negativity / argmax-preservation, temperature scaling semantics (T=1.0 identity, argmax preservation), top-k filtering (≤k elements, k=0/k≥len no-ops), argmax correctness, and repetition penalty semantics (1.0 no-op, reduces positives, worsens negatives)
- **Property Tests for `bitnet-generation`** (PR #683): 8 tests (5 property + 3 unit) covering `check_stop` priority order (token-ID > EOS > max-tokens > stop-string), `max_tokens=0` disabling budget, determinism, and stop-string boundary matching
- **Property Tests for `bitnet-engine-core` standalone suite** (PR #683): 7 tests (5 property + 2 unit) in `tests/property_tests.rs` for `SessionConfig` JSON round-trips, `BackendInfo` JSON round-trips, `SessionMetrics` non-negativity, and default values
- **`Tokenizer::get_family_name()` trait method** (PR #673): Returns family hint (`"llama3"`, `"default"`, etc.) based on special tokens; used for auto-template detection
- **Property Tests for `bitnet-device-probe`** (PR #650): 4 proptest properties verifying GPU compiled idempotency, SIMD level determinism, CPU always available, and cuda_compiled consistency with `gpu_compiled()`
- **Property Tests for `bitnet-engine-core`** (PR #650): 3 proptest properties verifying `SessionConfig` JSON round-trips, `BackendInfo` JSON round-trips, and `SessionMetrics` non-negativity invariants
- **Property Tests for `bitnet-validation`** (PR #650): 7 proptest properties for `rules.rs` (detect_rules name invariants, envelope bounds, proj_rms consistency) and `names.rs` (is_ln_gamma keyword recognition, suffix requirement, determinism)
- **Property Tests for `bitnet-sampling`** (PR #650): 5 proptest properties for greedy argmax, index bounds, softmax distribution validity, top-k finite count, temperature=0 greedy equivalence
- **Property Tests for `bitnet-rope`** (PR #650): 4 proptest properties for valid input success, shape invariants, sin²+cos²=1 trig identity, odd-dim rejection
- **Property Tests for `bitnet-receipts`** (PR #650): 4 proptest properties for schema validation, real compute_path, valid kernel ID acceptance, empty kernel ID rejection
- **Property Tests for `bitnet-common/kernel_registry`** (PR #650): 4 proptest properties for no-duplicate backends, best_available reachability, CUDA preference, requires_gpu semantics
- **Property Tests for `bitnet-prompt-templates`** (PR #650): 4 proptest properties for user text inclusion, Raw identity, Instruct suffix, non-Raw stop sequences
- **Property Tests for `bitnet-tokenizers`** (PR #650): 4 proptest properties for BasicTokenizer ASCII round-trip, byte-count invariant, empty input, empty slice decode
- **Property Tests for `bitnet-honest-compute`** (PR #650): 4 proptest properties for valid ID acceptance, too-long ID rejection, compute_path validation, classify_compute_path correctness
- **Docs Archive Cleanup** (PR #650): 136 stale planning/sprint/spec documents moved to `docs/archive/`; `docs/explanation/` reduced to 13 user-facing Diataxis docs
- **Libfuzzer Crash Artifacts Gitignored**: Added `fuzz/crash-*`, `fuzz/slow-unit-*`, `fuzz/leak-*`, `fuzz/timeout-*` patterns to `.gitignore` to keep fuzz crash files out of the repo
- **Runtime Backend Selection** (PR #642): `BackendCapabilities` snapshot at CLI/server startup producing `requested=X detected=[…] selected=Y` log line and receipt field
- **CPU Golden Path E2E Tests** (PR #643): 5 deterministic end-to-end tests in `bitnet-inference` always running in PR CI without model download
- **SRP Microcrates Wired Into CI** (PR #644): `bitnet-logits`, `bitnet-gguf`, `bitnet-generation`, `bitnet-device-probe`, `bitnet-engine-core` added to CI test matrix
- **QK256 (GGML I2_S) Pure-Rust Support** (PR #640): GGUF loader detects and stores QK256 tensors; pure-Rust `gemv_qk256()` kernel; dual I2_S flavor detection; 17 comprehensive tests; no FFI required
- **Property Tests for SRP Microcrates** (PR #649): proptest suites for `bitnet-gguf` (header parsing, magic validation, arbitrary-byte safety) and `bitnet-generation` (stop criteria, max-token budget, stop-string matching)
- **GGUF Header Fuzz Target** (PR #649, `fuzz/fuzz_targets/gguf_header.rs`): dedicated libfuzzer target exercising `bitnet-gguf`'s lightweight header parser
- **Fuzz Target Registration** (PR #649): `architecture_detection`, `tl_lut_helper`, `tokenizer_discovery`, `vocab_size_extraction` fuzz targets were present but not registered in `fuzz/Cargo.toml`; now properly wired

### Fixed
- **Flaky SIMD throughput test** (PR #681): Removed brittle `throughput.is_finite()` assertion in `bitnet-kernels` quantized matmul throughput test; avoids intermittent failures on CI runners where elapsed time rounds to 0
- **Clippy warnings workspace-wide** (PR #682): Resolved all `clippy --all-targets --features cpu` warnings; zero-warning policy enforced for CPU feature gate
- **`ci-core.yml` paths filter excludes documentation-only PRs** (PR #684): Added `CLAUDE.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `.github/copilot-instructions.md` to the `paths:` trigger so docs-only PRs correctly run required status checks instead of being permanently blocked
- **Env-var race conditions eliminated across workspace** (PR #678): Replaced bare `unsafe { env::set_var/remove_var }` with `temp_env::with_var`/`with_vars`/`with_var_unset` + `#[serial(bitnet_env)]` in `bitnet-kernels` (3 tests in `issue_260_feature_gated_tests.rs`), `bitnet-models` (`test_iq2s_backend_selection`), `bitnet-trace` (5 integration tests + 1 unit test), and `bitnet-runtime-profile-contract-core` (4 tests); eliminates flaky test failures from cross-test env var races in parallel nextest runs
- **Template detection + KV cache init tests unblocked** (PR #673): 3 tests previously failing due to missing `get_family_name()` and wrong KV cache test logic now pass without `#[ignore]`
- **QK256, CLI, and simple inference tests unblocked** (PR #674): 9 TDD scaffold tests (`test_qk256_tolerance_*`, `test_help_text_snapshot`, `test_simple_real_inference`, etc.) activated by fixing stub implementations
- **Receipts property test mock filter** (PR #675): `proptest` strategy for valid kernel IDs excluded "mock"-containing strings that are rejected by honest-compute policy (`0.99f32 as f64` precision issue also fixed)
- **Issue #159 resolved — `MockGgufFileBuilder` writes real GGUFs** (PR #676): Replaced `b"mock_gguf_content"` placeholder with `GgufWriter`-based synthetic GGUF containing real transformer layer tensors; 4 previously ignored tests (`test_ac1_complete_transformer_weight_parsing_cpu`, `test_ac2_i2s/tl1/tl2_quantization_accuracy_cpu`) now pass
- **TL1/TL2 Quantizer Round-Trip Accuracy** (PR #641): LUT offset mismatch in `pack_2bit_values` caused clipping of codes 2–3; dequantize now maps ternary inputs within `[-1, 1]`
- **Security Audit** (PR #645): Updated `bytes` to 1.11.1 (RUSTSEC-2026-0007) and `time` to 0.3.47 (RUSTSEC-2026-0009); added documented accepted-risk entries for gix-date, rsa, bincode advisories; fixed LGPL false-positive in GPL license check; fixed supply-chain Cargo.lock verification command
- **GGUF Tokenizer Test Fixtures** (PR #648): Repaired three malformed GGUF fixtures (`kv_count` off-by-one, corrupted magic); rebuilt `llama3-with-hf-tokenizer.gguf` with correct 5-KV HF tokenizer metadata; added `.gitignore` exceptions so fixtures reach CI; fixed `get_u32_metadata` in `bitnet-models` to also accept `GgufValue::I32` (as generated by llama.cpp)
- **Gitignore Fix** (PR #649): `fuzz/` was globally gitignored, preventing fuzz source files from being tracked; replaced with specific artifact-only ignores for `fuzz/target/`, `fuzz/artifacts/`, `fuzz/coverage/`, `fuzz/corpus/`
- **Env-var race condition hardening** (PR #678): Replaced `unsafe { env::set_var / remove_var }` call sites without serial+RAII cleanup with `temp_env::with_var` + `#[serial(bitnet_env)]` where practical; duplicate `blake3` dev-dep removed; GPU cfg predicates unified to `any(feature = "gpu", feature = "cuda")`

### Documentation
- **CLAUDE.md + copilot-instructions.md accuracy refresh** (PR #680): Removed stale "Active Blockers" references to non-existent/closed issues (#254, #260, #439, #469); updated MSRV to 1.92.0; updated test counts (3,097 passing, 466 skipped); simplified ignored-test categorization to reflect actual reasons (model-gated, CUDA, slow, crossval, TDD scaffold); updated architecture doc with SRP microcrate section

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.10.0-rc.0] - 2025-10-17

### Added

- **Pure-Rust GGUF Tokenizer** ([feat/crossval-parity-harness](https://github.com/EffortlessSteven/BitNet-rs/tree/feat/crossval-parity-harness)):
  - Load tokenizers directly from GGUF metadata (SPM protobuf, BPE vocab+merges)
  - No external tokenizer files required for self-contained models
  - Auto-detection from `tokenizer.ggml.model` metadata
- **BPE ByteLevel Prefix Space Fix**:
  - Enable `add_prefix_space=true` for both BPE pre-tokenizer **and** decoder
  - Ensures consistent " What" tokenization with leading space marker
  - Proper handling of GPT-2 style Ġ (U+0120) prefix markers
- **BPE Piece-to-GGUF-ID Remapping**:
  - Maps HuggingFace token IDs to authoritative GGUF vocabulary IDs via `HashMap<String, u32>`
  - Prevents ID drift from HuggingFace's internal ID assignment
  - Uses GGUF `tokenizer.ggml.tokens` array as source of truth (index = token ID)
  - Model-aware token IDs: BitNet GGUF has "ĠWhat" at position 3639 (not universal 3923)
- **Receipt-Based Provenance** (crossval):
  - Tokenizer metadata: `merges_count` (BPE), `tokenizer_blob_sha256` (SPM)
  - Environment metadata: `target_cpu`, `cpu_features`, `libc`, `rayon_threads`, `seed`
  - C++ metadata: `llama_cpp_commit` (from BITNET_CPP_DIR)
  - Prompt hash: `blake3` for formatted prompt verification
  - Timeout receipts with diagnostic data (120s guard)
- **Model-Aware Golden Token Tests**:
  - Split fixtures: `golden_tokens_gpt2.json`, `golden_tokens_llama.json`, `golden_tokens_llama3.json`
  - Auto-select based on `tokenizer.ggml.model` from GGUF
  - Exact-match validation locks in BPE and SPM behavior
- **Optional tok-debug Diagnostics**:
  - Feature-gated `--features tok-debug` for piece→ID diagnostics
  - Dumps first 8 tokens: `hf_id`, `piece`, `gguf_id`
- **LLaMA-3 Chat Prompt Support**:
  - Multi-prompt support via `CROSSVAL_PROMPT_SET=math|chat|all`
  - Auto-detect `parse_special=true` for `<|start_header_id|>`, `<|eot_id|>`
  - Proper EOT vs EOS handling
- **CI Workflows**:
  - `parity-proof.yml`: Fast PR gate with receipt artifact upload
  - `nightly-parity-matrix.yml`: Prompt+quant matrix with dated archiving

### Fixed

- **BPE Token ID Mapping**:
  - Rust tokenizer now uses GGUF vocabulary IDs instead of HuggingFace internal IDs
  - Fixes token ID mismatches by looking up piece strings in GGUF `tokenizer.ggml.tokens` array
  - Handles space normalization: both ` What` and `ĠWhat` forms correctly mapped
- **FFI Memory Safety**:
  - Hardened batch lifecycle management in C++ shim
  - Two-call tokenization pattern (preflight + allocation)
  - Explicit safety contract documentation for FFI boundary
- **Deterministic Inference**:
  - Seeded runs with `BITNET_SEED` + `RAYON_NUM_THREADS=1`
  - Reproducible tokenization and generation
- **SPM Blob Reproducibility**:
  - SHA256 fingerprinting of SentencePiece protobuf
  - Validates tokenizer integrity across runs

### Documentation

- Added `docs/releases/v0.10.0-rc.0-summary.md`: Comprehensive release guide
- Updated `CLAUDE.md`: Document tok-debug feature and golden token tests
- CI workflow examples for parity validation

### Testing

- Parity test timeout increased: 60s → 120s for 2B+ models
- Golden token tests: 8 test cases across 3 tokenizer families
- FFI lifecycle test: 100x create/drop cycles (no crashes)

### Changed

- **Prompt Template Auto-Detection Default** ([PR #467](https://github.com/EffortlessMetrics/BitNet-rs/pull/467)):
  - Changed auto-detection fallback from `raw` to `instruct` template for better out-of-box experience with instruction-tuned models
  - Auto-detection now logs template selection at `info` level for better visibility
  - Use `--prompt-template raw` to explicitly request raw completion behavior if needed
- **Kernel Recorder Receipt Improvements** ([PR #467](https://github.com/EffortlessMetrics/BitNet-rs/pull/467)):
  - Kernel recorder now resets before each inference turn to track per-turn kernel usage
  - Receipt kernel lists are now sorted, deduplicated, and capped at 32 entries to prevent bloat
  - Documents that receipts track coarse kernel classes (e.g., `i2s_gemv`, `tl1_lut_q`) not individual calls
- **Weight Mapper Code Quality Improvements** ([#209](https://github.com/EffortlessSteven/BitNet-rs/pull/209)):
  - **Enhanced Pick Helper Function**: Refactored `pick` helper in `weight_mapper.rs` to return both key and tensor, eliminating code duplication
  - **Streamlined Tensor Alias Resolution**: Simplified embedding and lm_head tensor alias handling with improved maintainability
  - **Code Structure Optimization**: Enhanced internal helper functions while maintaining all existing functionality and backward compatibility
  - **Comprehensive Validation**: All 62 tests in bitnet-models package pass, ensuring no functional regressions
- **Cargo Configuration Cleanup** ([#113](https://github.com/EffortlessSteven/BitNet-rs/pull/113)):
  - Remove tool-generated metadata files (`.crates.toml`, `.crates2.json`) from version control
  - Commit `Cargo.lock` files for reproducible builds across environments
  - Standardize GPU feature aliases in cargo config to use `gpu` instead of `cuda`

### Added

- **Token-Level Stop Sequences** ([PR #467](https://github.com/EffortlessMetrics/BitNet-rs/pull/467)):
  - Added `stop_token_ids` field to `GenerationConfig` for fast token-level stop checking
  - Avoids expensive string decoding for common stop tokens like LLaMA-3's `<|eot_id|>`
  - Falls back to string-based stop sequence matching for compatibility
- **Enhanced CLI Help Text Testing** ([PR #467](https://github.com/EffortlessMetrics/BitNet-rs/pull/467)):
  - Help footer tests now use direct CLI builder instead of subprocess for faster, more reliable testing
  - Exposed `build_cli()` function in `bitnet-cli` for external test use

- **CPU Forward Pass with Autoregressive Generation** ([#462](https://github.com/EffortlessSteven/BitNet-rs/issues/462)):
  - **CPU Forward Pass**: Complete autoregressive generation from BOS token through logits with device-aware CPU inference
  - **CLI Inference**: `bitnet-cli infer` command with CPU backend support and deterministic inference
  - **TL LUT Helper**: Safe `bitnet_kernels::tl_lut::lut_index()` with checked arithmetic, overflow detection, and 100% mutation testing coverage
  - **Receipt CPU Validation**: Honest compute validation with CPU quantized kernel enforcement (i2s_*, tl1_*, tl2_*) and silent CPU fallback detection
  - **Enhanced Testing**: 91% overall mutation testing score (31/31 tests passing, 20/22 mutants killed)
- **WebAssembly (WASM) Compatibility Improvements** ([#170](https://github.com/EffortlessSteven/BitNet-rs/pull/170)):
  - **Enhanced WASM Build Compatibility**: Avoiding native dependency conflicts for seamless WebAssembly compilation
  - **Updated Tokenizers Configuration**: Added `unstable_wasm` feature for proper WebAssembly support alongside existing features
  - **Fixed Workspace Dependency Management**: Consistent dependency versions across all WASM-related crates
  - **Improved Browser Compatibility**: Proper feature gating and dependency management for browser environments
  - **SIMD Intrinsic Compatibility**: Fixed AVX2 intrinsics for WebAssembly target compatibility using portable alternatives
  - **Zero Breaking Changes**: All improvements maintain full backward compatibility with existing builds
- **Enhanced GGUF Tokenizer with Optimized Byte Mapping** ([#171](https://github.com/EffortlessSteven/BitNet-rs/pull/171)):
  - **O(1) Byte Lookup Performance**: Replaced HashMap with `byte_to_id[256]` array for optimized tokenization performance
  - **Improved UTF-8 Handling**: Enhanced byte buffer management in decode operations for robust text processing
  - **BOS Token Support**: Added BOS token support to BasicTokenizer with vocab boundary checks and enhanced safety
  - **Critical SPM Compilation Fix**: Resolved compilation error in SentencePiece tokenizer that prevented `spm` feature from working
  - **Enhanced token_to_piece Functionality**: Direct byte lookup for improved token-to-text conversion performance
  - **Comprehensive Test Coverage**: Added unit tests for BOS token handling, vocab overflow protection, and enhanced tokenizer functionality
  - **Backward Compatibility**: All enhancements maintain full API compatibility with existing tokenizer implementations
- **GPU Infrastructure Foundation - CUDA Context and Module Access** ([#199](https://github.com/EffortlessSteven/BitNet-rs/pull/199)):
  - **Public CUDA Infrastructure Access**: Exposed CUDA context and module through public `context()` and `module()` accessor methods
  - **Custom Kernel Loading Foundation**: Enables loading of specialized PTX kernels for domain-specific GPU operations
  - **Advanced Memory Management Support**: CUDA context access enables custom memory pools and advanced GPU memory operations
  - **Device-Aware Launch Optimization**: Integrated `calculate_optimal_launch_params()` in matrix multiplication operations replacing hardcoded 16x16 block sizes
  - **Dead Code Elimination**: Removed `#[allow(dead_code)]` attributes from infrastructure fields, ensuring active utilization
  - **GPU Infrastructure Sequence Foundation**: First phase of three-part GPU enhancement sequence (#199 → #202 → #206)
  - **Enhanced Testing Framework**: Added comprehensive GPU infrastructure tests for context access, module loading, and optimal launch parameters
  - **Production-Ready Architecture**: Maintains backward compatibility while enabling advanced GPU programming capabilities
  - **Multi-Stream Coordination Support**: Foundation for overlapped execution and advanced GPU orchestration
  - **Performance Monitoring Integration**: Enhanced GPU operations tracking with device-specific optimization metrics
- **Enhanced GGUF Tensor Alignment Validation** ([#210](https://github.com/EffortlessSteven/BitNet-rs/pull/210)):
  - **Comprehensive Tensor Offset Validation**: All tensor offsets are validated against `general.alignment` to ensure proper memory alignment
  - **Data Section Boundary Validation**: Validates that the tensor data section starts at properly aligned boundaries
  - **Metadata Consistency Checks**: Verifies that n_dims field matches actual dimensions array length preventing parsing corruption
  - **Enhanced Error Messages**: Detailed error messages include tensor names, offsets, and alignment requirements for easier debugging
  - **Memory Safety Improvements**: Prevents out-of-bounds access with robust bounds checking and detailed tensor information
  - **Malformed GGUF Detection**: Detects corrupted or non-standard GGUF files with comprehensive validation before processing
  - **Backward Compatibility**: Enhanced validation maintains full compatibility with existing valid GGUF files
  - **Performance Impact**: Negligible performance impact while significantly improving reliability and error detection
- **Enhanced SIMD Kernels with Optimized Memory Access** ([#174](https://github.com/EffortlessSteven/BitNet-rs/pull/174)):
  - **Improved Memory Operations**: Refactored SIMD store operations in I2S quantization using cleaner `_mm_storeu_si64` and `_mm_loadu_si64` intrinsics
  - **Cross-Platform Compatibility**: Added 7 comprehensive SIMD compatibility tests ensuring consistent behavior across x86_64 and ARM64 architectures
  - **Performance Validation**: Implemented SIMD/scalar parity validation for all quantization types with comprehensive accuracy testing
  - **Architecture-Specific Testing**: Enhanced data alignment scenario testing for robust SIMD operations across different memory layouts
  - **Benchmark Infrastructure**: Added 9 specialized benchmark suites for performance comparison between SIMD implementations
  - **Microbenchmark Framework**: Comprehensive performance baseline validation with automated SIMD optimization verification
  - **Enhanced Error Handling**: Improved cross-architecture support with proper CPU feature detection and graceful scalar fallback
- **Comprehensive NaN-Safe Sampling Pipeline** ([#184](https://github.com/EffortlessSteven/BitNet-rs/pull/184)):
  - **Automatic NaN Sanitization**: Converts NaN logits to negative infinity for predictable behavior
  - **Enhanced Top-K Filtering**: Pre-filters NaN values and uses safe partial_cmp() with fallback to Ordering::Equal
  - **Robust Top-P Filtering**: Sanitizes logits before probability calculation with graceful edge case handling
  - **Safe Sorting Operations**: Prevents panics from NaN comparisons with deterministic tie-breaking
  - **Comprehensive Test Coverage**: New tests for `test_top_k_filter_with_nan`, `test_top_p_filter_with_nan`, and `test_sample_with_nan_logits`
  - **Production Reliability**: Prevents runtime crashes from model output anomalies while maintaining streaming inference
- **Enhanced Prefill Functionality for Batch Inference** ([#187](https://github.com/EffortlessSteven/BitNet-rs/pull/187)):
  - **Explicit Prefill Integration**: Added `engine.prefill()` method for explicit cache warming and latency measurement in batch inference operations
  - **Structured Performance Metrics**: Enhanced `TimingMetrics` with separate measurement for prefill, decode, tokenization, and total inference time
  - **Comprehensive Throughput Calculations**: New `ThroughputMetrics` with tokens-per-second for prefill, decode, and end-to-end performance
  - **Production-Ready Error Handling**: Robust error handling for empty tokens, invalid tokens, and context length exceeded scenarios
  - **Comprehensive Test Coverage**: 13 specialized tests (8 unit + 5 integration) covering batch prefill operations, performance consistency, and error recovery
  - **Enhanced CLI Integration**: Updated inference commands with prefill timing support and structured JSON metrics output
  - **Mock Testing Infrastructure**: Comprehensive mock model and tokenizer with realistic timing for accurate performance validation
  - **Documentation Enhancement**: Detailed inline documentation with usage examples, performance benefits, and troubleshooting guides
  - **Backward Compatibility**: Zero breaking changes to existing API while adding enhanced functionality
- **PrefillEngine Trait Abstraction** ([#139](https://github.com/EffortlessSteven/BitNet-rs/pull/139)):
  - **Clean Dependency Injection**: Added PrefillEngine trait to enable proper mocking in CLI inference tests
  - **Async Support**: Trait methods support async/await pattern matching InferenceEngine API
  - **Test Infrastructure**: MockEngine implementation for isolated unit testing of inference pipelines
  - **Backward Compatible**: InferenceEngine implements PrefillEngine with existing functionality preserved
  - **Enhanced Testability**: Enables comprehensive unit testing of CLI batch inference without external dependencies
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
  - Integration with existing bitnet-rs kernel architecture and error handling patterns
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

### Added (0.3.0)

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

### Fixed (0.3.0)

- Model loading edge cases and error handling improvements
- Memory management optimizations for large models
- Cross-platform compatibility improvements

### Changed (0.3.0)

- Improved API ergonomics and error messages
- Enhanced documentation with more examples
- Streamlined build process and dependency management

## [0.2.0] - 2024-12-15

### Added (0.2.0)

- Basic quantization support (I2_S, TL1, TL2)
- GGUF format compatibility
- Python bindings with PyO3
- C API for llama.cpp compatibility
- Streaming inference capabilities
- Initial CUDA support

### Fixed (0.2.0)

- Memory safety improvements
- Performance optimizations
- Cross-validation accuracy

## [0.1.0] - 2024-11-01

### Added (0.1.0)

- Initial release
- Basic BitNet model loading and inference
- CPU-only quantization support
- Core API design and architecture


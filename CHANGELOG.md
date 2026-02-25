# Changelog

All notable changes to BitNet.rs will be documented in this file.

## [Unreleased]

### Added
- **Property Tests for `bitnet-transformer`** (PR #689): 4 proptest properties covering KVCache invariants — seq_len tracks N append operations correctly, clear/reset semantics, overflow (capacity) rejection, and initial zero state
- **Tracing Instrumentation for Template Detection** (PR #686): `bitnet-prompt-templates::TemplateType::detect()` now emits `debug!` on each branch (Llama3Chat, Instruct, Raw) and `warn!` on the Raw fallback; in-crate `#[traced_test]` unit tests verify log capture
- **`tracing-test` 0.2.6 Added to Workspace** (PR #686): Shared dev-dep for crates that test tracing output

### Fixed
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

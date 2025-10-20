# TDD Scaffold Implementation Guide: GPU, Tokenizer, and Other Tests

**Files**: Multiple test suites across bitnet-kernels, bitnet-tokenizers, bitnet-cli, bitnet-common, bitnet-models
**Total Scaffolds**: 64 scaffolds across 8 test files
**Coverage**: GPU kernels, tokenizer discovery/download, CLI QA, strict mode, real model loading

---

## Summary Table

| Category | File | Scaffolds | GPU Required | Network Required | Priority |
|----------|------|-----------|--------------|------------------|----------|
| GPU Kernels | gpu_quantization.rs | 6 | Yes | No | MEDIUM |
| GPU Integration | gpu_integration.rs | 6 | Yes | No | MEDIUM |
| Tokenizer Smoke | tokenization_smoke.rs | 7 | No | No (GGUF file) | HIGH |
| Tokenizer Download | test_ac4_smart_download_integration.rs | 15 | No | Yes | MEDIUM |
| Tokenizer Production | test_ac5_production_readiness.rs | 20 | No | No (GGUF files) | HIGH |
| CLI QA | qa_greedy_math_confidence.rs | 3 | No | No (GGUF file) | HIGH |
| Strict Mode | issue_260_strict_mode_tests.rs | 3 | No | No | MEDIUM |
| Real Model Loading | real_model_loading.rs | 4 | No | No (GGUF file) | HIGH |

---

## GPU Kernel Tests (bitnet-kernels)

### File: `crates/bitnet-kernels/tests/gpu_quantization.rs`

**Total Scaffolds**: 6 ignored tests
**GPU Dependency**: All require CUDA-capable hardware
**Feature Flag**: `#[cfg(feature = "gpu")]`

#### Scaffold 1: GPU I2S Quantization Test
**Lines**: 138-176
**Status**: Stub (unimplemented quantization)
**Priority**: MEDIUM (GPU-dependent)
**Requires**: 
- CUDA toolkit installed
- CUDA-capable GPU (compute capability 7.0+)
- Feature flag: `gpu`

**Test Coverage**:
- Medium-sized input quantization (256 elements)
- I2S quantization format validation
- GPU device stats collection
- Graceful CPU fallback handling

**Implementation Steps**:
1. Implement `DeviceAwareQuantizer::quantize()` for I2S on GPU
2. Add GPU kernel for I2S quantization (`i2s_gpu_quantize`)
3. Implement device stats collection
4. Add fallback detection and logging
5. Validate output tensor structure

**Success Criteria**:
- Quantization produces non-zero output
- GPU kernel ID appears in stats
- Fallback to CPU works gracefully if GPU unavailable

**Dependencies**:
- `bitnet_kernels::device_aware::DeviceAwareQuantizer`
- CUDA kernel implementation for I2S

---

#### Scaffold 2: GPU vs CPU Quantization Accuracy
**Lines**: 178-231
**Status**: Stub (accuracy comparison unimplemented)
**Priority**: MEDIUM (GPU validation)
**Requires**: CUDA hardware + CPU baseline

**Test Coverage**:
- Identical input quantization on CPU and GPU
- Numerical accuracy comparison
- Non-zero output validation
- Provider detection (GPU vs CPU fallback)

**Implementation Steps**:
1. Implement CPU I2S quantization kernel
2. Implement GPU I2S quantization kernel
3. Add numerical comparison utilities (cosine similarity, max error)
4. Validate both paths produce meaningful results
5. Log provider information for debugging

**Success Criteria**:
- Both CPU and GPU produce non-zero output
- If GPU active, results should be numerically close (tolerance <1e-5)
- Provider detection works correctly

**Validation Strategy**:
- Compare scales and quantized values
- Check for systematic bias
- Validate block-wise quantization consistency

---

#### Scaffold 3: GPU Quantization Fallback
**Lines**: 264-297
**Status**: Stub (fallback mechanism unimplemented)
**Priority**: MEDIUM (error recovery)

**Test Coverage**:
- Normal GPU operation
- Forced CPU fallback via `force_cpu_fallback()`
- Provider state change detection
- Post-fallback operation validation

**Implementation Steps**:
1. Implement `DeviceAwareQuantizer::force_cpu_fallback()`
2. Add provider state tracking
3. Implement automatic fallback on GPU errors
4. Test operation after fallback
5. Validate state consistency

**Success Criteria**:
- Provider changes from GPU to fallback
- Quantization still works after fallback
- No crashes or data corruption

---

#### Scaffold 4: GPU Memory Management
**Lines**: 316-346
**Status**: Stub (memory leak detection unimplemented)
**Priority**: MEDIUM (stability)

**Test Coverage**:
- Multiple quantization operations (10 iterations)
- Memory leak detection
- Device resource cleanup
- Error handling in loop

**Implementation Steps**:
1. Implement GPU memory allocation tracking
2. Add CUDA memory profiling hooks
3. Test repeated allocation/deallocation
4. Validate no memory leaks (use `cuda-memcheck` or similar)
5. Add cleanup verification

**Success Criteria**:
- All 10 iterations complete successfully
- No memory growth over iterations
- GPU memory fully released after test

**Debugging Tools**:
- CUDA error checking after each operation
- `nvidia-smi` monitoring during test
- Valgrind/cuda-memcheck integration

---

#### Scaffold 5: Concurrent GPU Operations
**Lines**: 348-400
**Status**: Stub (thread safety unimplemented)
**Priority**: MEDIUM (concurrency)

**Test Coverage**:
- 4 concurrent threads using GPU quantizer
- Thread safety validation
- CUDA context handling
- Error handling per thread

**Implementation Steps**:
1. Implement thread-safe CUDA context management
2. Add mutex/lock for GPU resource access
3. Test concurrent kernel launches
4. Validate no data races or corruption
5. Handle CUDA context per-thread or shared

**Success Criteria**:
- All 4 threads complete successfully
- No deadlocks or race conditions
- Correct results from each thread

**CUDA Considerations**:
- CUDA contexts are not thread-safe by default
- May need one context per thread or serialized access
- Test should detect and handle concurrency errors gracefully

---

#### Scaffold 6: GPU I2S Quantization Smoke Test
**Lines**: 138-176 (duplicate of Scaffold 1, different test name)
**Status**: See Scaffold 1
**Priority**: MEDIUM

---

### File: `crates/bitnet-kernels/tests/gpu_integration.rs`

**Total Scaffolds**: 6 ignored tests
**GPU Dependency**: All require CUDA hardware
**Feature Flag**: `#[cfg(any(feature = "gpu", feature = "cuda"))]`

#### Scaffold 7: Comprehensive GPU Validation
**Lines**: 19-104
**Status**: Partial (GpuValidator framework exists)
**Priority**: MEDIUM (integration test)

**Test Coverage**:
- CUDA availability check
- Numerical accuracy validation (4 test sizes)
- Performance benchmarking (3 sizes)
- Memory management validation
- Error handling validation

**Implementation Steps**:
1. Complete `GpuValidator::validate()` implementation
2. Implement accuracy tests with CPU baseline
3. Add performance benchmarking with GFLOPS calculation
4. Test memory leak detection
5. Validate error handling for edge cases

**Success Criteria**:
- All accuracy tests pass (max_error < 1e-6)
- GPU achieves >1.0 GFLOPS
- GPU speedup > 0.5x vs CPU (not slower)
- No memory leaks detected
- Error handling works correctly

**Performance Targets**:
- 256x256x256: >10 GFLOPS
- 512x512x512: >50 GFLOPS
- 1024x1024x1024: >100 GFLOPS (on modern GPU)

---

#### Scaffold 8: Quick Benchmark Integration
**Lines**: 165-180
**Status**: Partial (quick_benchmark exists)
**Priority**: MEDIUM (validation)

**Test Coverage**:
- Fast GPU validation
- Smoke test for CUDA availability
- Basic performance sanity check

**Implementation Steps**:
1. Ensure `quick_benchmark()` runs without errors
2. Add minimal validation (CUDA available, kernel runs)
3. Test with small matrix sizes
4. Validate output is reasonable

**Success Criteria**:
- Benchmark completes without errors
- CUDA kernel executes successfully
- Results are logged correctly

---

#### Scaffold 9: Large Matrix Performance
**Lines**: 182-222
**Status**: Partial (CudaKernel::matmul_i2s exists)
**Priority**: MEDIUM (stress test)

**Test Coverage**:
- 512x512x512, 1024x1024x1024, 2048x1024x512 matrices
- GFLOPS calculation
- Non-zero result validation
- Performance logging

**Implementation Steps**:
1. Test progressively larger matrices
2. Calculate GFLOPS for each size
3. Validate non-zero output
4. Check for performance scaling
5. Test memory allocation for large tensors

**Success Criteria**:
- All sizes complete successfully
- GFLOPS increases with matrix size
- Results contain non-zero values
- No out-of-memory errors

**Performance Expectations**:
- 512³: ~50ms, ~10 GFLOPS
- 1024³: ~400ms, ~50 GFLOPS
- 2048×1024×512: ~200ms, ~100 GFLOPS

---

#### Scaffold 10: Concurrent Kernel Usage
**Lines**: 224-274
**Status**: Stub (thread safety unimplemented)
**Priority**: MEDIUM (concurrency)

**Test Coverage**:
- 4 threads each create CUDA kernel
- Concurrent matrix multiplication
- CUDA context thread safety
- Error handling per thread

**Implementation Steps**:
1. Test per-thread kernel creation
2. Validate CUDA context management
3. Test concurrent kernel launches
4. Handle thread-local CUDA errors
5. Validate at least some threads succeed

**Success Criteria**:
- At least 1 thread succeeds (CUDA may limit concurrent contexts)
- No crashes or deadlocks
- Proper error handling for context conflicts

**Note**: CUDA contexts have limitations - test allows some failures if GPU restricts concurrency.

---

#### Scaffold 11-12: Memory Management and Error Handling
**Lines**: 106-163
**Status**: Partially implemented in comprehensive test
**Priority**: MEDIUM

**See Scaffold 7** for memory management and error handling validation.

---

## Tokenizer Tests (bitnet-tokenizers)

### File: `crates/bitnet-tokenizers/tests/tokenization_smoke.rs`

**Total Scaffolds**: 7 ignored tests
**Network Dependency**: No (requires GGUF file via env var)
**Environment Variable**: `CROSSVAL_GGUF`

#### Scaffold 13: Pure Rust Tokenizer from GGUF Smoke Test
**Lines**: 43-87
**Status**: Partial (RustGgufTokenizer exists)
**Priority**: HIGH (core functionality)

**Test Coverage**:
- Load tokenizer from GGUF metadata
- Encode basic text ("What is 2+2?")
- Validate token IDs within vocab bounds
- BOS token handling from metadata

**Implementation Steps**:
1. Test `RustGgufTokenizer::from_gguf()` with real GGUF
2. Validate encoding produces non-empty tokens
3. Check token IDs are within vocab size
4. Test BOS token hint from metadata

**Success Criteria**:
- Tokenizer loads from GGUF successfully
- Text encodes to ≥3 tokens
- All token IDs < vocab_size
- BOS hint extracted correctly

**Test Setup**:
```bash
export CROSSVAL_GGUF=models/model.gguf
cargo test -p bitnet-tokenizers --test tokenization_smoke pure_rust_tokenizer_from_gguf_smoke -- --ignored
```

---

#### Scaffold 14: BOS Token Handling
**Lines**: 89-153
**Status**: Partial (BOS detection exists)
**Priority**: HIGH (correctness)

**Test Coverage**:
- Encode without BOS (`add_bos=false`)
- Encode with BOS (`add_bos=true`)
- Verify BOS token presence
- Compare token counts

**Implementation Steps**:
1. Extract BOS token ID from GGUF
2. Test encoding with `add_bos=false`
3. Test encoding with `add_bos=true`
4. Validate first token is BOS when enabled
5. Check token count differences

**Success Criteria**:
- BOS token ID extracted correctly
- `add_bos=true` produces BOS at start
- `add_bos=false` does not include BOS (or differs from true)
- BOS hint from metadata works

---

#### Scaffold 15: Special Token Lookup
**Lines**: 155-222
**Status**: Partial (bos_eos_eot method exists)
**Priority**: HIGH (LLaMA-3 support)

**Test Coverage**:
- Extract BOS, EOS, EOT token IDs
- Validate special token IDs are accessible
- Test `bos_token_id()`, `eos_token_id()` methods
- Verify at least one special token configured

**Implementation Steps**:
1. Test `bos_eos_eot()` extraction
2. Validate BOS/EOS/EOT IDs match metadata
3. Test special token accessor methods
4. Handle models without EOT (non-LLaMA-3)

**Success Criteria**:
- BOS and EOS tokens extracted
- EOT token extracted for LLaMA-3 models
- At least one special token present
- Accessor methods return correct IDs

---

#### Scaffold 16: Parse Special EOT Handling
**Lines**: 224-282
**Status**: Stub (parse_special flag unimplemented)
**Priority**: HIGH (LLaMA-3 chat)

**Test Coverage**:
- Encode with `parse_special=false` (literal)
- Encode with `parse_special=true` (parse tokens)
- Validate EOT token parsing
- Check tokenizer kind (SPM vs BPE)

**Implementation Steps**:
1. Implement `parse_special` flag in `encode()`
2. Test literal encoding of "<|eot_id|>"
3. Test special token parsing of "<|eot_id|>"
4. Validate EOT token ID appears with `parse_special=true`
5. Handle SPM vs BPE differences

**Success Criteria**:
- `parse_special=false` treats "<|eot_id|>" as text
- `parse_special=true` parses it as token ID 128009
- EOT token appears in parsed output
- Encoding succeeds regardless of flag

**Note**: SPM tokenizers have built-in special token handling; BPE needs `parse_special` control.

---

#### Scaffold 17: Tokenizer Kind Detection
**Lines**: 284-334
**Status**: Partial (kind method exists)
**Priority**: HIGH (format detection)

**Test Coverage**:
- Detect SPM vs BPE tokenizer
- Validate kind matches metadata
- Test GGUF `tokenizer.ggml.model` field
- Check SPM/BPE feature gates

**Implementation Steps**:
1. Test `tokenizer.kind()` method
2. Extract `tokenizer.ggml.model` from GGUF
3. Validate kind matches metadata ("llama" -> SPM, "gpt2" -> BPE)
4. Check feature gate consistency

**Success Criteria**:
- Kind detected as `Spm` or `Bpe`
- Kind matches GGUF metadata
- SPM requires `spm` feature
- Metadata extraction works

---

#### Scaffold 18: Vocabulary Size Sanity
**Lines**: 346-402
**Status**: Partial (vocab_size method exists)
**Priority**: HIGH (bounds checking)

**Test Coverage**:
- Extract vocabulary size from GGUF
- Validate size within reasonable range (1K-200K)
- Check token IDs within vocab bounds
- Test SPM vs BPE typical sizes

**Implementation Steps**:
1. Test `tokenizer.vocab_size()` method
2. Validate size > 0
3. Check size within typical range (1000-200000)
4. Encode text and validate all token IDs < vocab_size

**Success Criteria**:
- Vocab size > 0
- Vocab size within 1K-200K range
- All encoded token IDs < vocab_size
- SPM typically ~32K, BPE ~50K

---

#### Scaffold 19: FFI Parity Tests (NOTE)
**Lines**: 336-344 (comment block)
**Status**: N/A (tests in crossval crate)
**Priority**: N/A

**Note**: FFI parity tests are in `crossval/tests/parity_bitnetcpp.rs`, not this file. See crossval crate for FFI validation.

---

### File: `crates/bitnet-tokenizers/tests/test_ac4_smart_download_integration.rs`

**Total Scaffolds**: 15 ignored tests
**Network Dependency**: Yes (HuggingFace Hub downloads)
**Feature Flag**: `#[cfg(feature = "cpu")]` (tokio::test)

#### Scaffold 20: Download Tokenizer from HuggingFace
**Lines**: 21-49
**Status**: Stub (SmartTokenizerDownload unimplemented)
**Priority**: MEDIUM (network-dependent)

**Test Coverage**:
- Download tokenizer.json from HuggingFace Hub
- Verify file exists after download
- Validate file size > 0
- Test with LLaMA-2 repo

**Implementation Steps**:
1. Implement `SmartTokenizerDownload::new()`
2. Implement `download_tokenizer()` async method
3. Add HuggingFace Hub API integration
4. Handle network errors gracefully
5. Validate downloaded file integrity

**Success Criteria**:
- Download succeeds or fails gracefully (network)
- Downloaded file exists and is non-empty
- Path returned points to valid file
- Network errors produce actionable messages

**Test Setup**:
```bash
cargo test -p bitnet-tokenizers --test test_ac4_smart_download_integration ac4_download_tokenizer_from_huggingface -- --ignored
```

---

#### Scaffold 21: Tokenizer Download Caching
**Lines**: 54-91
**Status**: Stub (caching unimplemented)
**Priority**: MEDIUM (performance)

**Test Coverage**:
- First download from network
- Second download from cache
- Compare download times
- Validate same file path returned

**Implementation Steps**:
1. Implement cache directory structure
2. Add cache key generation
3. Test first download (slower)
4. Test second download (faster)
5. Validate cache hit detection

**Success Criteria**:
- First download completes (or skips if network unavailable)
- Second download faster than first
- Same file path returned for both
- Cache hit logged correctly

---

#### Scaffold 22: Download Verification
**Lines**: 96-120
**Status**: Stub (verification unimplemented)
**Priority**: MEDIUM (integrity)

**Test Coverage**:
- Download tokenizer file
- Verify file is valid JSON
- Validate JSON structure
- Test integrity checks

**Implementation Steps**:
1. Download tokenizer file
2. Read file contents
3. Parse as JSON
4. Validate JSON schema (optional)
5. Check for required fields

**Success Criteria**:
- Downloaded file is valid JSON
- JSON parsing succeeds
- File integrity verified
- No corruption detected

---

#### Scaffold 23: Network Failure Handling
**Lines**: 125-154
**Status**: Stub (error handling unimplemented)
**Priority**: MEDIUM (robustness)

**Test Coverage**:
- Test with invalid repository
- Validate error message quality
- Test actionable error guidance
- Graceful failure handling

**Implementation Steps**:
1. Test download with invalid repo
2. Validate error is returned
3. Check error message is descriptive
4. Test error message length > 10 chars
5. Provide recovery suggestions

**Success Criteria**:
- Invalid repo produces error
- Error message is actionable
- Error message is non-empty
- Guidance provided for resolution

---

#### Scaffold 24: Download Retry Logic
**Lines**: 159-189
**Status**: Stub (retry unimplemented)
**Priority**: MEDIUM (resilience)

**Test Coverage**:
- Test retry on transient failures
- Validate retry transparency
- Test eventual success or failure
- Log retry attempts

**Implementation Steps**:
1. Implement retry mechanism (3 attempts)
2. Add exponential backoff
3. Test transient failure recovery
4. Log retry attempts
5. Fail gracefully after max retries

**Success Criteria**:
- Retries are transparent to caller
- Transient failures recovered automatically
- Max retries respected
- Success or failure logged correctly

---

#### Scaffold 25-32: Additional Download Tests
**Lines**: 191-492
**Status**: Stub (various download features)
**Priority**: MEDIUM-LOW

**Remaining scaffolds cover**:
- BitNet download infrastructure integration (25)
- Multiple files download (26)
- Download progress monitoring (27)
- Offline mode handling (28)
- Concurrent downloads (29)
- Vocabulary size validation (30)
- Cache management (31)
- Discovery integration (32)
- Error recovery strategies (33)

**Common Implementation Pattern**:
1. Implement `SmartTokenizerDownload` core
2. Add feature incrementally
3. Test with real network or mock
4. Validate error handling
5. Document network requirements

**Note**: Network-dependent tests should be run manually or in CI with network access.

---

### File: `crates/bitnet-tokenizers/tests/test_ac5_production_readiness.rs`

**Total Scaffolds**: 20 ignored tests
**Network Dependency**: No (requires GGUF fixtures)
**Priority**: HIGH (production validation)

#### Scaffold 33: CrossVal C++ Reference Tokenization
**Lines**: 21-49
**Status**: Stub (C++ FFI comparison unimplemented)
**Priority**: HIGH (parity validation)

**Test Coverage**:
- Load tokenizer from BitNet GGUF
- Encode test text with Rust tokenizer
- Compare with C++ reference (future)
- Validate non-empty tokens

**Implementation Steps**:
1. Load GGUF with embedded tokenizer
2. Extract tokenizer with `try_extract_embedded_tokenizer()`
3. Encode test text
4. (Future) Compare with C++ reference via FFI
5. Validate token count and IDs

**Success Criteria**:
- Tokenizer extracts successfully
- Encoding produces tokens
- Token IDs are valid
- (Future) Parity with C++ within tolerance

**Test Setup**:
```bash
# Requires GGUF fixture
cp models/model.gguf tests/fixtures/gguf/bitnet-b1.58-2B.gguf
cargo test -p bitnet-tokenizers --test test_ac5_production_readiness ac5_crossval_cpp_reference_tokenization -- --ignored
```

---

#### Scaffold 34: CrossVal Tokenization Parity
**Lines**: 54-88
**Status**: Stub (multi-input parity unimplemented)
**Priority**: HIGH (comprehensive validation)

**Test Coverage**:
- Simple English text
- Multi-language (中文, العربية)
- Numbers and symbols
- Code snippets
- Long text (100 repeats)

**Implementation Steps**:
1. Create diverse test cases
2. Tokenize each with Rust implementation
3. (Future) Compare with C++ reference
4. Validate all produce tokens
5. Check tolerance <1e-5 for numerical comparisons

**Success Criteria**:
- All test cases tokenize successfully
- Non-empty tokens for all inputs
- (Future) Parity with C++ reference
- Handles diverse input correctly

---

#### Scaffold 35: CrossVal Vocabulary Size Parity
**Lines**: 93-118
**Status**: Stub (vocab comparison unimplemented)
**Priority**: HIGH (correctness)

**Test Coverage**:
- LLaMA-2 32K vocab
- LLaMA-3 128K vocab
- GPT-2 50K vocab
- Validate exact match with C++ reference

**Implementation Steps**:
1. Load GGUF for each model type
2. Extract vocabulary size
3. Compare with expected values
4. (Future) Compare with C++ reference
5. Validate exact match

**Success Criteria**:
- LLaMA-2: 32000
- LLaMA-3: 128256
- GPT-2: 50257
- Exact match with C++ reference

**Note**: Requires GGUF fixtures for each model type.

---

#### Scaffold 36: GGUF Model Compatibility Rate (>99%)
**Lines**: 127-181
**Status**: Stub (compatibility testing unimplemented)
**Priority**: HIGH (production target)

**Test Coverage**:
- 10 diverse GGUF models
- Compatibility rate calculation
- Target: >99% compatibility
- Various architectures (LLaMA, Mistral, Phi, etc.)

**Implementation Steps**:
1. Create GGUF fixture collection
2. Test tokenizer discovery for each
3. Count compatible vs total
4. Calculate compatibility rate
5. Validate >99% target

**Success Criteria**:
- ≥99% compatibility rate (9/10 or better)
- All major architectures supported
- Actionable errors for incompatible models
- Comprehensive logging

**Fixture Requirements**:
- LLaMA-2 (Q4_0, Q8_0)
- LLaMA-3 instruct
- BitNet b1.58
- GPT-2
- Mistral, CodeLLaMA, Phi, Gemma, Qwen

---

#### Scaffold 37: GGUF Quantization Format Compatibility
**Lines**: 186-221
**Status**: Stub (format testing unimplemented)
**Priority**: HIGH (format support)

**Test Coverage**:
- Q4_0, Q4_1, Q5_0, Q5_1, Q8_0
- F16, F32
- Tokenizer extraction for all formats

**Implementation Steps**:
1. Create GGUF fixtures for each format
2. Test tokenizer discovery
3. Validate vocab size extraction
4. Log format compatibility
5. Handle format-specific edge cases

**Success Criteria**:
- All quantization formats supported
- Tokenizer extraction works for all
- Vocab size extracted correctly
- Format-specific handling works

---

#### Scaffold 38-52: Production Readiness Tests
**Lines**: 229-600
**Status**: Stub (various production features)
**Priority**: HIGH-MEDIUM

**Remaining scaffolds cover**:
- Discovery performance (<100ms) (38)
- Tokenization throughput (>10 iter/sec) (39)
- Comprehensive error messages (40)
- Documentation coverage (41)
- Production-scale model loading (42)
- Deterministic tokenization (43)
- Concurrent tokenization (44)
- Memory efficiency (45)
- Production error recovery (46)
- Comprehensive system integration (47)

**Common Implementation Pattern**:
1. Create GGUF fixtures for testing
2. Implement feature incrementally
3. Add performance benchmarks
4. Validate error handling
5. Test edge cases and stress scenarios

---

## CLI QA Tests (bitnet-cli)

### File: `crates/bitnet-cli/tests/qa_greedy_math_confidence.rs`

**Total Scaffolds**: 3 ignored tests
**Network Dependency**: No (requires GGUF file)
**Priority**: HIGH (user-facing validation)

#### Scaffold 53: Greedy Math Simple 2+2
**Lines**: 115-178
**Status**: Partial (CLI exists, needs model)
**Priority**: HIGH (smoke test)

**Test Coverage**:
- CLI execution with greedy sampling
- Math prompt: "2+2="
- Validate output contains "4"
- Deterministic inference (seed=42)

**Implementation Steps**:
1. Build bitnet-cli binary
2. Set BITNET_GGUF environment variable
3. Run CLI with `--greedy` flag
4. Parse output
5. Validate "4" appears in output

**Success Criteria**:
- CLI executes successfully
- Output contains "4"
- Deterministic (same output each run)
- Completes in <30 seconds

**Test Setup**:
```bash
cargo build -p bitnet-cli --no-default-features --features cpu,full-cli
export BITNET_GGUF=models/model.gguf
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
cargo test -p bitnet-cli --test qa_greedy_math_confidence test_greedy_math_simple_2plus2 -- --ignored
```

---

#### Scaffold 54: Greedy Math Q&A Format
**Lines**: 195-256
**Status**: Partial (CLI + template exists)
**Priority**: HIGH (Q&A validation)

**Test Coverage**:
- Instruct template Q&A format
- Question: "What is 2+2?"
- Validate output contains "4" or "four"
- Deterministic greedy sampling

**Implementation Steps**:
1. Build CLI with full-cli feature
2. Run with `--prompt-template instruct`
3. Parse Q&A output
4. Validate answer presence
5. Check for "4" or "four"

**Success Criteria**:
- CLI executes with instruct template
- Output contains "4" or "four"
- Q&A format respected
- Deterministic output

---

#### Scaffold 55: Greedy Deterministic Reproducibility
**Lines**: 280-355
**Status**: Partial (determinism needs validation)
**Priority**: HIGH (correctness)

**Test Coverage**:
- Two identical CLI runs
- Same seed (42)
- Compare outputs byte-for-byte
- Validate determinism

**Implementation Steps**:
1. Run CLI twice with same parameters
2. Capture both outputs
3. Compare outputs exactly
4. Validate identical results
5. Test with RAYON_NUM_THREADS=1

**Success Criteria**:
- Run 1 and Run 2 produce identical output
- Byte-for-byte match
- No stochastic variation
- Determinism holds across runs

**Note**: Requires single-threaded execution for full determinism.

---

#### Scaffold 56: Greedy Respects Stop Sequences
**Lines**: 379-447
**Status**: Partial (stop sequences exist)
**Priority**: MEDIUM (UX)

**Test Coverage**:
- Instruct template default stop sequences
- Request 100 tokens
- Validate generation stops early
- Check no stop sequence in output

**Implementation Steps**:
1. Run CLI with `--max-tokens 100`
2. Use instruct template
3. Count output tokens
4. Validate <50 tokens generated
5. Check stop sequences not present

**Success Criteria**:
- Generation stops before 100 tokens
- Word count <50 (approximate)
- Stop sequences ("\n\nQ:", "\n\nHuman:") not in output
- Early stopping works correctly

---

## Strict Mode Tests (bitnet-common)

### File: `crates/bitnet-common/tests/issue_260_strict_mode_tests.rs`

**Total Scaffolds**: 3 ignored tests (2 flaky)
**Network Dependency**: No
**Priority**: MEDIUM (infrastructure)

#### Scaffold 57: Strict Mode Validation Behavior
**Lines**: 108-182
**Status**: Stub (validation unimplemented)
**Priority**: MEDIUM (testing infrastructure)

**Test Coverage**:
- Mock inference path validation
- Real inference path validation
- Kernel availability validation
- Performance metrics validation

**Implementation Steps**:
1. Implement `StrictModeConfig::validate_inference_path()`
2. Implement `validate_kernel_availability()`
3. Implement `validate_performance_metrics()`
4. Add mock detection logic
5. Test failure scenarios

**Success Criteria**:
- Mock paths rejected in strict mode
- Real paths allowed in strict mode
- Missing kernels flagged
- Suspicious performance flagged

**Note**: TDD placeholder - drives implementation of strict mode validation.

---

#### Scaffold 58: Granular Strict Mode Configuration
**Lines**: 185-285
**Status**: Stub (granular config unimplemented)
**Priority**: MEDIUM (flexibility)

**Test Coverage**:
- Individual config options (FAIL_ON_MOCK, REQUIRE_QUANTIZATION, etc.)
- Partial strict configuration
- CI-specific enhanced mode
- Configuration inheritance

**Implementation Steps**:
1. Implement `StrictModeConfig::from_env_detailed()`
2. Add granular environment variable parsing
3. Implement CI enhanced mode
4. Test configuration combinations
5. Validate inheritance

**Success Criteria**:
- Individual flags work independently
- Partial configuration supported
- CI mode adds extra strictness
- All combinations tested

---

#### Scaffold 59: Cross-Crate Strict Mode Consistency
**Lines**: 295-377
**Status**: Flaky (env var pollution, issue #441)
**Priority**: MEDIUM (integration)

**Test Coverage**:
- All crates detect strict mode
- Configuration consistency across crates
- Coordinated validation
- Thread-safe access

**Implementation Steps**:
1. Implement `StrictModeEnforcer` in each crate
2. Add cross-crate coordination
3. Test configuration propagation
4. Validate thread safety
5. Fix environment variable pollution

**Success Criteria**:
- All 5 crates detect strict mode
- Configuration consistent across crates
- No validation failures
- Thread-safe operation

**Known Issues**:
- Flaky in workspace runs (issue #441)
- Environment variable conflicts
- Passes individually

---

## Real Model Loading Tests (bitnet-models)

### File: `crates/bitnet-models/tests/real_model_loading.rs`

**Total Scaffolds**: 4 ignored tests
**Network Dependency**: No (requires BITNET_GGUF)
**Priority**: HIGH (core functionality)

#### Scaffold 60: Real GGUF Model Loading with Validation
**Lines**: 63-100
**Status**: Partial (ProductionModelLoader exists)
**Priority**: HIGH (production loading)

**Test Coverage**:
- Load GGUF with strict validation
- Validate model structure
- Validate tensor alignment (32-byte)
- Validate quantization detection

**Implementation Steps**:
1. Set BITNET_GGUF environment variable
2. Create `ProductionLoadConfig` with strict validation
3. Load model with `ProductionModelLoader`
4. Validate model config (layers, dims, vocab)
5. Check tensor alignment
6. Verify quantization format detection

**Success Criteria**:
- Model loads within timeout (60s)
- Model structure valid (layers >0, dims >0)
- Tensor alignment validated
- Quantization format detected

**Test Setup**:
```bash
export BITNET_GGUF=models/model.gguf
cargo test -p bitnet-models --features inference --test real_model_loading test_real_gguf_model_loading_with_validation -- --ignored
```

---

#### Scaffold 61: Enhanced Tensor Alignment Validation
**Lines**: 105-122
**Status**: Partial (basic checks only)
**Priority**: MEDIUM (SIMD optimization)

**Test Coverage**:
- 32-byte tensor alignment
- SIMD efficiency (64-byte for hidden size)
- Detailed error reporting for misalignment

**Implementation Steps**:
1. Load model with strict validation
2. Access GGUF tensor metadata (future)
3. Validate alignment for each tensor
4. Check hidden_size % 64 == 0
5. Report alignment warnings

**Success Criteria**:
- Model loads successfully
- Basic alignment checks pass
- Warnings logged for non-optimal alignment
- (Future) Full tensor-level validation

**Note**: Full tensor alignment validation requires exposing GGUF internals.

---

#### Scaffold 62: Device-Aware Model Optimization
**Lines**: 127-145
**Status**: Partial (basic device config)
**Priority**: MEDIUM (GPU support)

**Test Coverage**:
- Optimal device configuration
- Batch size recommendations
- Device strategy selection
- GPU-specific optimizations

**Implementation Steps**:
1. Load model
2. Get optimal device config with `get_optimal_device_config()`
3. Validate device strategy present
4. Check batch size recommendation >0
5. Test GPU-specific paths (if GPU available)

**Success Criteria**:
- Device config generated
- Strategy is non-None
- Batch size >0
- GPU config works if GPU available

**Requires**: `--features inference,gpu` for GPU tests

---

#### Scaffold 63: Model Metadata Extraction Validation
**Lines**: 151-179
**Status**: Partial (config extraction works)
**Priority**: HIGH (correctness)

**Test Coverage**:
- Vocab size >0
- Hidden size >0
- Attention heads >0
- Layer count >0
- Dimension alignment

**Implementation Steps**:
1. Load model
2. Extract config with `model.config()`
3. Validate all dimensions >0
4. Check hidden_size % num_heads == 0
5. Validate KV heads configuration

**Success Criteria**:
- Vocab size >0
- Hidden size >0, divisible by num_heads
- Attention heads >0
- Layers >0, <200 (sanity check)
- KV heads configured correctly

---

#### Scaffold 64: Memory-Mapped Model Loading
**Lines**: 205-237
**Status**: Partial (mmap works by default)
**Priority**: MEDIUM (large models)

**Test Coverage**:
- Large file loading (>100MB)
- Memory mapping efficiency
- Load time <30s for large models
- Model structure validation

**Implementation Steps**:
1. Check file size ≥100MB
2. Load model with default mmap
3. Measure load time
4. Validate load_duration <30s
5. Check model structure

**Success Criteria**:
- File size ≥100MB
- Load time <30 seconds
- Model loads successfully
- Memory mapping used efficiently

**Note**: Skips test if file <100MB (not large enough for meaningful mmap test).

---

## Implementation Order Recommendation

### Phase 1: Core Functionality (HIGH Priority)
1. **Tokenizer Smoke Tests** (Scaffolds 13-19): Pure Rust tokenizer validation - no network, just GGUF file
   - Start with Scaffold 13 (basic encoding)
   - Then 14-15 (BOS/special tokens)
   - Then 16-18 (parse_special, kind, vocab)
2. **Real Model Loading** (Scaffolds 60, 63-64): Production model loader with validation
   - Start with 60 (basic loading)
   - Then 63 (metadata extraction)
   - Then 64 (mmap for large files)
3. **CLI QA Tests** (Scaffolds 53-55): User-facing greedy inference validation
   - Start with 53 (simple math)
   - Then 54 (Q&A format)
   - Then 55 (determinism)

### Phase 2: Production Readiness (MEDIUM Priority)
4. **Tokenizer Production Tests** (Scaffolds 33-47): Comprehensive validation, C++ parity
   - Start with 33-35 (crossval parity)
   - Then 36-37 (GGUF compatibility)
   - Then 38-39 (performance benchmarks)
5. **Strict Mode Tests** (Scaffolds 57-59): Testing infrastructure for mock elimination
   - Start with 57 (validation behavior)
   - Then 58 (granular config)
   - Then 59 (cross-crate consistency - fix flakiness first)
6. **Tokenizer Download Tests** (Scaffolds 20-32): Network-dependent, lower priority
   - Start with 20-21 (basic download + caching)
   - Then 22-24 (verification, error handling, retry)
   - Then 25-32 (advanced features)

### Phase 3: GPU Features (GPU-Dependent)
7. **GPU Quantization Tests** (Scaffolds 1-6): Requires CUDA hardware
   - Start with 1 (basic I2S)
   - Then 2 (CPU/GPU accuracy)
   - Then 3-6 (fallback, memory, concurrency)
8. **GPU Integration Tests** (Scaffolds 7-12): Comprehensive GPU validation
   - Start with 7 (comprehensive validation)
   - Then 8-9 (benchmarks)
   - Then 10-12 (concurrency, stress tests)

### Phase 4: Advanced Features (MEDIUM-LOW Priority)
9. **Device-Aware Optimization** (Scaffold 62): GPU-specific optimizations
10. **Tensor Alignment Validation** (Scaffold 61): Enhanced GGUF validation
11. **CLI Stop Sequences** (Scaffold 56): UX improvements

---

## Environment Requirements Summary

### Required for Most Tests
- **BITNET_GGUF**: Path to GGUF model file
  - Download: `cargo run -p xtask -- download-model`
  - Or set: `export BITNET_GGUF=models/model.gguf`

### GPU Tests (Scaffolds 1-12)
- **CUDA Toolkit**: Version 11.0+ or 12.0+
- **CUDA-Capable GPU**: Compute capability 7.0+ (Volta or newer)
- **Feature Flag**: `--features gpu` or `--features cuda`
- **Runtime Check**: `nvidia-smi` should show GPU

### Tokenizer Tests (Scaffolds 13-19)
- **CROSSVAL_GGUF**: Path to GGUF file with embedded tokenizer
  - Same as BITNET_GGUF: `export CROSSVAL_GGUF=$BITNET_GGUF`

### Network Tests (Scaffolds 20-32)
- **Internet Access**: HuggingFace Hub downloads
- **Async Runtime**: tokio (included in test)
- **Optional**: `BITNET_OFFLINE=1` to skip network tests

### CLI Tests (Scaffolds 53-56)
- **Built CLI**: `cargo build -p bitnet-cli --features cpu,full-cli`
- **BITNET_GGUF**: Path to model file
- **Determinism**: `BITNET_DETERMINISTIC=1 BITNET_SEED=42 RAYON_NUM_THREADS=1`

### Strict Mode Tests (Scaffolds 57-59)
- **BITNET_STRICT_MODE**: `1` or `true` to enable
- **No External Dependencies**: Pure environment variable testing

### Real Model Loading (Scaffolds 60-64)
- **BITNET_GGUF**: Required
- **Feature Flag**: `--features inference`
- **GPU Tests**: `--features inference,gpu` (Scaffold 62)

---

## Common Test Patterns

### Pattern 1: GGUF File Required
```bash
# Setup
export BITNET_GGUF=models/model.gguf
export CROSSVAL_GGUF=$BITNET_GGUF

# Run test
cargo test -p <crate> --test <test_file> <test_name> -- --ignored
```

### Pattern 2: GPU Required
```bash
# Check GPU
nvidia-smi

# Run test
cargo test -p bitnet-kernels --features gpu --test gpu_quantization <test_name> -- --ignored
```

### Pattern 3: Network Required
```bash
# Run test (may fail in CI without network)
cargo test -p bitnet-tokenizers --test test_ac4_smart_download_integration <test_name> -- --ignored

# Skip in CI
if [ -z "$CI" ]; then
  cargo test ... -- --ignored
fi
```

### Pattern 4: Deterministic Inference
```bash
# Setup environment
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export RAYON_NUM_THREADS=1

# Run test
cargo test -p bitnet-cli --test qa_greedy_math_confidence <test_name> -- --ignored
```

---

## CI Integration Strategy

### Tier 1: Always Run (No External Dependencies)
- Strict mode tests (57-58, skip 59 until fixed)
- Tokenizer smoke tests (13-19) with cached GGUF

### Tier 2: Run with Model Cache
- Real model loading (60, 63-64)
- CLI QA tests (53-55)
- Tokenizer production tests (33-47) with GGUF fixtures

### Tier 3: Run in GPU CI
- GPU quantization (1-6)
- GPU integration (7-12)
- Device-aware optimization (62)

### Tier 4: Manual / Nightly Only
- Tokenizer download tests (20-32) - network-dependent
- Large model mmap test (64) - requires >100MB file
- Concurrent GPU tests (6, 10-12) - may fail on some hardware

---

## Quick Reference: Test Counts by Priority

| Priority | Count | Description |
|----------|-------|-------------|
| **HIGH** | 26 | Core functionality, user-facing, production-critical |
| **MEDIUM** | 31 | Infrastructure, optimization, non-blocking |
| **LOW** | 7 | Advanced features, nice-to-have, future enhancements |

**Total**: 64 scaffolds

---

## Notes for Implementers

1. **Start with HIGH priority tests** - they validate core functionality
2. **GPU tests are MEDIUM priority** - require specific hardware, not blocking for most users
3. **Network tests are MEDIUM-LOW priority** - can fail in restricted environments
4. **Flaky tests (59)** - fix environment variable pollution before enabling in CI
5. **TDD pattern** - tests compile but fail until implementation provided
6. **Documentation** - each test includes manual testing instructions in comments

---

**Generated**: 2025-10-20
**Last Updated**: Sprint 4 TDD Scaffold Completion
**Status**: Comprehensive guide for 64 test scaffolds across 8 files

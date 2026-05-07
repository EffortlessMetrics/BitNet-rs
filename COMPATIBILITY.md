# BitNet-rs Compatibility

> **Note:** BitNet-rs is pre-alpha (v0.2.x). All compatibility contracts in this document are best-effort goals, not guarantees. Breaking changes may occur before v1.0.0.

This document describes the compatibility goals that BitNet-rs is working toward. Contracts described here become binding at v1.0.0.

## API Stability

### C/C++ FFI API (llama.cpp compatibility)

We target API compatibility with llama.cpp's C API. The following functions are implemented in `crates/bitnet-ffi/src/llama_compat.rs` but **not yet validated in CI integration tests**:

```c
// Model management — implemented, CI validation pending
llama_model* llama_load_model_from_file(const char* path, struct llama_model_params params);
void llama_free_model(llama_model* model);

// Context management — implemented, CI validation pending
llama_context* llama_new_context_with_model(llama_model* model, struct llama_context_params params);
void llama_free(llama_context* ctx);

// Tokenization — implemented, CI validation pending
int32_t llama_tokenize(const llama_model* model, const char* text, int32_t text_len,
                       int32_t* tokens, int32_t n_max_tokens, bool add_bos, bool special);

// Evaluation — implemented, CI validation pending
int llama_eval(llama_context* ctx, const int32_t* tokens, int32_t n_tokens,
               int32_t n_past, int32_t n_threads);

// Logits access — implemented, CI validation pending
float* llama_get_logits(llama_context* ctx);
```

**Error codes (target):**
- `-1`: Generic error
- `-2`: Invalid UTF-8
- `-3`: Tokenization failed
- `0`: Success
- `1`: Eval error

### Python API (llama-cpp-python compatibility)

> **Status: Scaffolded.** The Python API exists in `crates/bitnet-py` but is not validated end-to-end in CI. Do not depend on this API in its current state.

We target compatibility with llama-cpp-python. The intended API:

```python
# This import change is the ONLY change needed (once validated)
from bitnet.llama_compat import Llama  # was: from llama_cpp import Llama

llama = Llama(
    model_path="model.gguf",
    n_ctx=2048,
    n_batch=512,
    n_threads=4,
    n_gpu_layers=32,
)

tokens = llama.tokenize(text, add_bos=True, special=True)
output = llama(prompt, max_tokens=100, temperature=0.7)
```

## Tokenizer Compatibility

### Supported Tokenizer Types

BitNet-rs aims to handle all of the following tokenizer types:

1. **GPT-2 BPE** (including variants with missing metadata)
2. **Llama 3 BPE** (128k vocabulary GPT-2 variant)
3. **SentencePiece** (Llama 1/2 style)
4. **Tiktoken** (GPT-3.5/4 style)
5. **Falcon** tokenizer

CI currently tests GPT-2 BPE and Llama 3 BPE. Other tokenizer types are implemented but not yet covered in CI.

### Extended Tokenizer Handling

BitNet-rs handles some tokenizer configurations that break llama.cpp:

```yaml
# This configuration breaks llama.cpp but works in BitNet-rs
tokenizer.ggml.model: gpt2
tokenizer.ggml.pre: <missing>  # llama.cpp fails here
```

## GGUF Format Support

### Auto-fixing Capability

BitNet-rs automatically fixes the following GGUF issues during loading:

1. Missing `tokenizer.ggml.pre` for GPT-2 models
2. Missing `tokenizer.ggml.add_space_prefix`
3. Missing `tokenizer.ggml.byte_fallback`
4. Missing special token IDs (BOS, EOS, PAD, UNK)

### Model Compatibility

BitNet-rs loads:
- Models that llama.cpp can load
- **Plus** models that llama.cpp cannot load due to:
  - Missing tokenizer metadata
  - GPT-2 tokenizer without pre-tokenizer field
  - Vocabulary size mismatches (with warning)

### FFI Bridge (`--features ffi`)

BitNet-rs includes an FFI bridge for cross-validation against C++ implementations:

- **Quantization Bridge**: I2S, TL1, and TL2 quantization via C++ kernels
- **Validation**: Tools for comparing FFI vs Rust quantization accuracy
- **Feature Gated**: Optional `--features ffi` flag with graceful fallback when unavailable
- **Safety**: Safe Rust wrappers with proper error handling and memory management

### GGUF Format Versions

- **GGUF v2**: Supported with 32-byte default alignment and tensor alignment validation
- **GGUF v3 Standard**: Supported with alignment and data_offset fields
- **GGUF v3 Early Variant**: Handles files missing alignment/data_offset fields (e.g., Microsoft BitNet models). Demonstrated in manual testing with 1.2GB model; no automated fixture test yet.
  - Invalid `alignment` values (0 or non-power-of-two) are clamped to 32
  - Invalid `data_offset` values fall back to `align_up(kv_end, alignment)`
  - Format variant detected using header-only heuristics

## Test Coverage

### Test Files

- `crates/bitnet-ffi/tests/api_contract.rs` — C API contract tests (gated behind feature flags; not running in CI)
- `crates/bitnet-tokenizers/tests/tokenizer_contracts.rs` — Tokenizer contract tests
- `crates/bitnet-py/tests/test_llama_compat.py` — Python API contract tests (not running in CI)

### CI

- `.github/workflows/compatibility.yml` — Runs on every PR (Linux only)
- Compatibility jobs are informational (`continue-on-error: true`) — failures do not block merges
- macOS and Windows CI coverage is planned but not yet implemented
- MSRV: 1.93.0 (Rust 2024 edition)

## Performance Goals

> **Note:** Performance targets are aspirational during pre-alpha (v0.2.x). No formal benchmarks exist comparing BitNet-rs to llama.cpp.

We aim for:

1. **No performance regression** vs llama.cpp for supported operations (not yet benchmarked)
2. **Better performance** for (aspirational; not yet validated):
   - Model loading (memory-mapped)
   - Tokenization (especially GPT-2)
   - SIMD operations (hand-optimized AVX2; AVX-512 code paths exist but are not validated in CI)

## Hardware Compatibility

### CPU Support

**Base Requirements:**
- x86_64 with SSE2 (2001+) or ARM64 with NEON
- Minimum 2GB RAM for small models (1-3B parameters)
- 64-bit operating system (Linux, macOS, Windows)

**SIMD Acceleration:**
- **AVX2 (Intel Haswell 2013+, AMD Excavator 2015+)**: Automatic detection
- **AVX-512 (Intel Skylake-X 2017+, Ice Lake 2019+)**: Runtime detection; code paths exist but not yet validated in CI
  - Requires both AVX-512F (Foundation) and AVX-512BW (Byte and Word) instruction sets
- **NEON (ARM64/AArch64)**: Automatic detection on compatible ARM processors

### GPU Backend Compatibility

| Backend | Feature Flag | Min Hardware | Status |
|---------|-------------|-------------|--------|
| NVIDIA CUDA | `gpu` / `cuda` | Compute 6.0+ (Pascal) | Alpha (scaffolded; not validated end-to-end) |
| Intel OpenCL | `opencl` | Arc A-series | Experimental (CPU reference impl; real OpenCL not validated) |
| Apple Metal | `metal` | M1/M2/M3+ | Scaffold (CPU reference stub only) |
| Vulkan | `vulkan` | Vulkan 1.3 GPU | Scaffold (CPU reference stub only) |
| AMD ROCm | `rocm` | RDNA 2+ | Scaffold (CPU reference stub only) |

**Status definitions:**
- **Alpha**: Feature-gated code exists but is not validated end-to-end in CI. May produce incorrect results.
- **Experimental**: Has some functional code paths but needs significant testing.
- **Scaffold**: CPU reference stub only — no actual GPU kernel execution.

**Backend selection** (`--device`):
- `auto` (default): Selects CUDA if available, otherwise CPU
- Explicit: `cuda`, `opencl`, `cpu` (other backends are scaffolded)

### Operating System Support

**Tested in CI:**
- Ubuntu 22.04 with GCC (ci-core.yml, compatibility.yml)
- macOS ARM64 (apple-silicon.yml — clippy only, build/test in progress)

**Supported but not in CI:**
- Linux (x86_64, ARM64): CPU inference with SIMD
- macOS (Intel, Apple Silicon): CPU path
- Windows (x86_64): MSVC or GNU toolchains

## What We DON'T Guarantee

1. Bug-for-bug compatibility with llama.cpp bugs
2. Compatibility with undocumented llama.cpp behavior
3. Support for llama.cpp's internal/private APIs
4. Identical numerical outputs (within quantization bounds is sufficient)

## Versioning Policy

- **Major version bump (2.0.0)**: Breaking compatibility contracts
- **Minor version bump (0.3.0)**: New features, maintaining compatibility
- **Patch version bump (0.2.2)**: Bug fixes, no API changes

## API Support Status

### llama.cpp C API

| Function | Status | Notes |
|----------|--------|-------|
| `llama_load_model_from_file` | Implemented | CI validation pending |
| `llama_free_model` | Implemented | CI validation pending |
| `llama_new_context_with_model` | Implemented | CI validation pending |
| `llama_free` | Implemented | CI validation pending |
| `llama_tokenize` | Implemented | CI validation pending |
| `llama_eval` | Implemented | CI validation pending |
| `llama_get_logits` | Implemented | CI validation pending |
| `llama_get_embeddings` | Not implemented | Planned |
| `llama_batch_*` | Not implemented | Planned |
| `llama_kv_cache_*` | Not implemented | Planned |
| `llama_grammar_*` | Not planned | Use constraints API instead |
| `llama_sampling_*` | Not implemented | Rust SamplingStrategy exists; FFI wrapper not yet exposed |
| `llama_model_quantize` | Not implemented | Planned |

### Error Code Table

| Code | Meaning | llama.cpp Compatible |
|------|---------|---------------------|
| `0` | Success | Yes |
| `-1` | Generic error | Yes |
| `-2` | Invalid UTF-8 | Yes |
| `-3` | Tokenization failed | Yes |
| `-4` | Model not found | Extension |
| `-5` | Model load failed | Extension |
| `-6` | Inference failed | Extension |
| `-7` | Out of memory | Extension |

## Inference Path Design Goals

### Teacher-Forcing and Incremental Decoding Parity

BitNet-rs is designed so that teacher-forcing (full sequence processing) and incremental decoding produce identical results. This is a design goal, not yet formally validated:

- Correct causal masking in both paths
- Identical positional encoding application
- KV cache consistency
- Deterministic results regardless of inference path

## Validation Status

### Cross-Validation Framework

BitNet-rs includes a cross-validation framework (`crossval-per-token`) for comparing Rust inference against C++ reference implementations. Current status:

- Validation framework is implemented
- Cross-validation CI is non-blocking and label-gated (runs only on PRs with `crossval` label)
- All current baseline receipts show `cosine_similarity: null` and `cpp_available: false`
- No measured parity data exists yet

### GGUF v3 Early Variant

BitNet-rs loads the Microsoft BitNet model (GGUF v3 early variant, 1.2GB) that causes the C++ reference to crash. This was demonstrated in manual testing; no automated fixture test exists.

### Validation Framework Components

1. **Tokenizer Parity** — Exact token ID matching between Rust and HF tokenizers
2. **Logit Parity (Tau-b)** — Score-aware Kendall's tau-b for handling quantization ties
3. **NLL Parity** — Token-weighted mean matching industry standard
4. **Property-Based Testing** — Hypothesis framework for exhaustive testing

## Stability Timeline

- **Pre-v1.0 (current)**: APIs may change; compatibility is best-effort
- **v1.0.0 (planned)**: FFI API locked, tokenizer compatibility locked
- **Post-v1.0**: Additional APIs may be added; existing ones won't break

## Commitment

We aim to (pre-alpha goals; not yet guarantees):

1. **Minimize breaking changes** to the compatibility layer as APIs stabilize
2. **Handle edge-case models** that llama.cpp fails on (GGUF v3 early variants)
3. **Match or improve performance** vs bitnet.cpp for supported operations (benchmarking infrastructure pending)
4. **Keep core tests passing** — CI-Core gate blocks merges on test failures

## Contact

If you find a compatibility issue:

1. Check this document first
2. Open an issue with the `compatibility` label
3. Include the exact error and a minimal reproduction

---

**Status:** BitNet-rs targets llama.cpp API compatibility. The FFI layer is implemented but validation is ongoing. Breaking changes may occur before v1.0.0.

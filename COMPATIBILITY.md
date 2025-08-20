# BitNet.rs Compatibility

This document defines the compatibility contracts that BitNet.rs maintains. Any changes that break these contracts are considered breaking changes and will be avoided or require a major version bump.

## 🔒 API Stability

### C/C++ FFI API (llama.cpp compatibility)

We aim for **100% API compatibility** with llama.cpp's C API. The following functions will maintain their exact signatures:

```c
// Model management - LOCKED API
llama_model* llama_load_model_from_file(const char* path, struct llama_model_params params);
void llama_free_model(llama_model* model);

// Context management - LOCKED API
llama_context* llama_new_context_with_model(llama_model* model, struct llama_context_params params);
void llama_free(llama_context* ctx);

// Tokenization - LOCKED API
int32_t llama_tokenize(const llama_model* model, const char* text, int32_t text_len,
                       int32_t* tokens, int32_t n_max_tokens, bool add_bos, bool special);

// Evaluation - LOCKED API
int llama_eval(llama_context* ctx, const int32_t* tokens, int32_t n_tokens,
               int32_t n_past, int32_t n_threads);

// Logits access - LOCKED API
float* llama_get_logits(llama_context* ctx);
```

**Error codes are locked:**
- `-1`: Generic error
- `-2`: Invalid UTF-8
- `-3`: Tokenization failed
- `0`: Success
- `1`: Eval error

### Python API (llama-cpp-python compatibility)

We guarantee drop-in compatibility with llama-cpp-python. The following will always work:

```python
# This import change is the ONLY change needed
from bitnet.llama_compat import Llama  # was: from llama_cpp import Llama

# All these signatures are LOCKED
llama = Llama(
    model_path="model.gguf",
    n_ctx=2048,
    n_batch=512,
    n_threads=4,
    n_gpu_layers=32,
    # ... all other parameters
)

tokens = llama.tokenize(text, add_bos=True, special=True)
output = llama(prompt, max_tokens=100, temperature=0.7)
```

## 🛡️ Tokenizer Compatibility Guarantees

### Universal Tokenizer Support

We guarantee to handle ALL of the following tokenizer types:

1. **GPT-2 BPE** (including variants with missing metadata)
2. **Llama 3 BPE** (128k vocabulary GPT-2 variant)
3. **SentencePiece** (Llama 1/2 style)
4. **Tiktoken** (GPT-3.5/4 style)
5. **Falcon** tokenizer

### Breaking llama.cpp Compatibility

We **explicitly guarantee** to handle some tokenizers that break llama.cpp:

```yaml
# This configuration breaks llama.cpp but MUST work in BitNet.rs
tokenizer.ggml.model: gpt2
tokenizer.ggml.pre: <missing>  # llama.cpp fails here
```

## 📦 GGUF Format Guarantees

### Auto-fixing Capability

We guarantee to automatically fix the following GGUF issues:

1. Missing `tokenizer.ggml.pre` for GPT-2 models
2. Missing `tokenizer.ggml.add_space_prefix`
3. Missing `tokenizer.ggml.byte_fallback`
4. Missing special token IDs (BOS, EOS, PAD, UNK)

### Model Compatibility

We guarantee to load:
- All models that llama.cpp can load
- **PLUS** models that llama.cpp cannot load due to:
  - Missing tokenizer metadata
  - GPT-2 tokenizer without pre-tokenizer field
  - Vocabulary size mismatches (with warning)

### GGUF Format Support

- **GGUF v2 and v3 headers**: BitNet.rs accepts both versions with defensive parsing
  - For v3, invalid `alignment` values (0 or non-power-of-two) are clamped to 32
  - For v3, invalid `data_offset` values (past EOF, misaligned, or backwards) fall back to `align_up(kv_end, alignment)`
  - Maintains full backward compatibility with v2 files

## 🧪 Test Coverage Requirements

All compatibility features are protected by tests:

### Required Test Files
- `crates/bitnet-ffi/tests/api_contract.rs` - C API contracts
- `crates/bitnet-tokenizers/tests/tokenizer_contracts.rs` - Tokenizer contracts
- `crates/bitnet-models/tests/gguf_compatibility.rs` - GGUF fix contracts
- `crates/bitnet-py/tests/test_llama_compat.py` - Python API contracts

### CI Requirements
- `.github/workflows/compatibility.yml` - Runs on every PR
- Must test on Linux, macOS, Windows
- Must test Python 3.8-3.12
- Must test Rust stable and MSRV (1.89.0)

## 📊 Performance Guarantees

While not breaking compatibility, we guarantee:

1. **No performance regression** vs llama.cpp for supported operations
2. **Better performance** for:
   - Model loading (memory-mapped)
   - Tokenization (especially GPT-2)
   - SIMD operations (hand-optimized)

## 🚫 What We DON'T Guarantee

To be clear, we do NOT guarantee:

1. Bug-for-bug compatibility with llama.cpp bugs
2. Compatibility with undocumented llama.cpp behavior
3. Support for llama.cpp's internal/private APIs
4. Identical numerical outputs (within quantization bounds is sufficient)

## 📝 Versioning Policy

- **Major version bump (2.0.0)**: Only if we break compatibility contracts
- **Minor version bump (1.1.0)**: New features, maintaining compatibility
- **Patch version bump (1.0.1)**: Bug fixes, no API changes

## 🔄 Migration Promise

We promise that migrating from llama.cpp to BitNet.rs will always be:

### For C/C++ users:
```c
// Change 1: Include path
#include "bitnet_ffi.h"  // was: #include "llama.h"

// Change 2: Link library
-lbitnet_ffi  // was: -llama

// That's it! No code changes needed.
```

### For Python users:
```python
# Change 1: Import
from bitnet.llama_compat import Llama  # was: from llama_cpp import Llama

# That's it! No code changes needed.
```

## 📊 API Support Truth Table

### llama.cpp C API Support Status

| Function | Status | Notes |
|----------|--------|-------|
| `llama_load_model_from_file` | ✓ | Full support |
| `llama_free_model` | ✓ | Full support |
| `llama_new_context_with_model` | ✓ | Full support |
| `llama_free` | ✓ | Full support |
| `llama_tokenize` | ✓ | Full support |
| `llama_eval` | ✓ | Full support |
| `llama_get_logits` | ✓ | Full support |
| `llama_get_embeddings` | • | Planned for v1.1 |
| `llama_batch_*` | • | Planned for v1.2 |
| `llama_kv_cache_*` | • | Planned for v1.2 |
| `llama_grammar_*` | × | Not planned (use constraints API) |
| `llama_sampling_*` | ✓ | Full support |
| `llama_model_quantize` | • | Planned for v1.3 |

**Legend:**
- ✓ = Fully supported
- • = Planned/In progress
- × = Not planned (alternative provided)

### Error Code Table

| Code | Meaning | llama.cpp Compatible |
|------|---------|---------------------|
| `0` | Success | ✓ |
| `-1` | Generic error | ✓ |
| `-2` | Invalid UTF-8 | ✓ |
| `-3` | Tokenization failed | ✓ |
| `-4` | Model not found | Extension |
| `-5` | Model load failed | Extension |
| `-6` | Inference failed | Extension |
| `-7` | Out of memory | Extension |
| `-8` | Thread safety error | Extension |
| `-9` | Invalid model ID | Extension |
| `-10` | Context length exceeded | Extension |

## 🏆 Compatibility Advantages

BitNet.rs provides these advantages while maintaining compatibility:

1. **Memory safety** - No segfaults, guaranteed by Rust
2. **Better error messages** - Clear, actionable error messages
3. **Broader model support** - Handles models llama.cpp can't
4. **Integrated features** - HTTP server, streaming, async/await
5. **Cross-platform** - Better Windows support

## 📅 Stability Timeline

- **2024-01-01**: FFI API locked (v1.0.0)
- **2024-01-01**: Python API locked (v1.0.0)
- **2024-01-01**: Tokenizer compatibility locked (v1.0.0)
- **Future**: Additional APIs may be added, existing ones won't break

## 🤝 Commitment

We commit to:

1. **Never break existing code** that uses our compatibility layer
2. **Always handle certain models** that llama.cpp fails on
3. **Maintain or improve performance** vs bitnet.cpp
4. **Keep tests passing** - CI blocks merges if compatibility breaks

## 📞 Contact

If you find a compatibility issue:

1. Check this document first
2. Run the compatibility test suite
3. Open an issue with the `compatibility` label
4. Include the exact error and a minimal reproduction

---

**The Bottom Line:** If your code works with llama.cpp or llama-cpp-python today, it will work with BitNet.rs tomorrow, next month, and next year. That's our promise.
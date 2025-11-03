# BitNet.cpp API Requirements Specification

**Status**: DISCOVERED ✓  
**Date**: October 25, 2025  
**Location**: `/home/steven/.cache/bitnet_cpp/`

## Executive Summary

BitNet.cpp (v1.0+) is built on the **llama.cpp framework** and uses the **llama.cpp C API** for tokenization and inference. There is no separate "BitNet-only" public API; instead, BitNet adds quantization kernels to the GGML backend.

**Key Finding**: The wrapper in `crossval/src/bitnet_cpp_wrapper.cc` should use **llama.cpp API functions directly**, not a hypothetical BitNet-specific interface.

## Available Artifacts

### Location
- **Root**: `/home/steven/.cache/bitnet_cpp/`
- **Headers**: `3rdparty/llama.cpp/include/llama.h` (primary public API)
- **Build**: `build/lib/` (compiled libllama.so, libggml.so, etc.)

### Key Files
- Primary API header: `/home/steven/.cache/bitnet_cpp/3rdparty/llama.cpp/include/llama.h`
- BitNet kernels: `include/ggml-bitnet.h` (kernel registration, not needed for FFI)
- Example usage: `3rdparty/llama.cpp/examples/main/main.cpp` (942 lines)
- Common utilities: `3rdparty/llama.cpp/common/common.cpp`

## API Overview

All functions use the **llama.cpp C interface** defined in `llama.h`. The following sections document the minimal API needed for the cross-validation wrapper.

### 1. Backend Initialization

```c
void llama_backend_init(void);   // Call once at program start
void llama_backend_free(void);   // Call once at program end
```

### 2. Model Loading

```c
struct llama_model_params llama_model_default_params(void);

struct llama_model * llama_load_model_from_file(
    const char * path_model,
    struct llama_model_params params
);

void llama_free_model(struct llama_model * model);

int32_t llama_n_vocab(const struct llama_model * model);
```

**Model Parameters**:
```c
struct llama_model_params {
    int32_t n_gpu_layers;
    // ... other fields ...
    bool use_mmap;        // Recommended: true
    bool use_mlock;       // Recommended: false
    bool check_tensors;   // Recommended: false
};
```

### 3. Context Creation

```c
struct llama_context_params llama_context_default_params(void);

struct llama_context * llama_new_context_with_model(
    struct llama_model * model,
    struct llama_context_params params
);

void llama_free(struct llama_context * ctx);
```

**Context Parameters**:
```c
struct llama_context_params {
    uint32_t n_ctx;           // Context size
    uint32_t n_batch;         // Batch size (512 recommended)
    int32_t n_threads;        // CPU threads
    int32_t n_threads_batch;  // Batch threads
    bool logits_all;          // true for all-position logits
    // ... other fields ...
};
```

### 4. Tokenization

```c
// Returns: positive = token count, negative = -(count that would fit)
int32_t llama_tokenize(
    const struct llama_model * model,
    const char * text,
    int32_t text_len,
    llama_token * tokens,          // NULL for size query
    int32_t n_tokens_max,          // Ignored if tokens=NULL
    bool add_special,              // Add BOS token
    bool parse_special             // Recognize special tokens
);
```

**Two-Pass Pattern**:
1. Call with `tokens=NULL` to get size
2. Allocate buffer
3. Call with buffer to fill

### 5. Batch and Decoding

```c
struct llama_batch llama_batch_get_one(
    llama_token * tokens,
    int32_t n_tokens,
    llama_pos pos_0,       // Starting position
    llama_seq_id seq_id    // Sequence ID
);

int32_t llama_decode(
    struct llama_context * ctx,
    struct llama_batch batch
);

void llama_synchronize(struct llama_context * ctx);
```

### 6. Logits Extraction

```c
float * llama_get_logits(struct llama_context * ctx);
```

**Layout**: Row-major, shape [n_tokens][n_vocab]

## Implementation Pattern for Wrapper Functions

### For `crossval_bitnet_tokenize()`

```c
#include <llama.h>
#include <cstring>
#include <cstdio>

int crossval_bitnet_tokenize(
    const char* model_path,
    const char* prompt,
    int add_bos,
    int parse_special,
    int32_t* out_tokens,
    int32_t out_capacity,
    int32_t* out_len,
    char* err,
    int32_t err_len
) {
    // Load model
    struct llama_model_params mparams = llama_model_default_params();
    struct llama_model * model = llama_load_model_from_file(model_path, mparams);
    
    if (!model) {
        snprintf(err, err_len, "Failed to load model from %s", model_path);
        return -1;
    }
    
    // Query size
    int32_t n_tokens = llama_tokenize(
        model,
        prompt, (int32_t)strlen(prompt),
        NULL, 0,
        add_bos != 0,
        parse_special != 0
    );
    
    if (n_tokens < 0) n_tokens = -n_tokens;  // Handle error case
    
    *out_len = n_tokens;
    
    // Pass 1: Size query
    if (!out_tokens || out_capacity <= 0) {
        llama_free_model(model);
        return 0;
    }
    
    // Pass 2: Bounds check
    if (out_capacity < n_tokens) {
        snprintf(err, err_len,
                 "Buffer too small (need %d, got %d)",
                 n_tokens, out_capacity);
        llama_free_model(model);
        return -1;
    }
    
    // Fill buffer
    int32_t result = llama_tokenize(
        model,
        prompt, (int32_t)strlen(prompt),
        out_tokens,
        out_capacity,
        add_bos != 0,
        parse_special != 0
    );
    
    if (result < 0) {
        snprintf(err, err_len, "Tokenization failed");
        llama_free_model(model);
        return -1;
    }
    
    llama_free_model(model);
    return 0;
}
```

### For `crossval_bitnet_eval_with_tokens()`

```c
int crossval_bitnet_eval_with_tokens(
    const char* model_path,
    const int32_t* tokens,
    int32_t n_tokens,
    int32_t n_ctx,
    float* out_logits,
    int32_t logits_capacity,
    int32_t* out_rows,
    int32_t* out_cols,
    char* err,
    int32_t err_len
) {
    // Load model
    struct llama_model_params mparams = llama_model_default_params();
    mparams.use_mmap = true;
    
    struct llama_model * model = llama_load_model_from_file(model_path, mparams);
    if (!model) {
        snprintf(err, err_len, "Failed to load model from %s", model_path);
        return -1;
    }
    
    // Create context
    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.logits_all = true;  // Critical: compute all positions
    
    struct llama_context * ctx = llama_new_context_with_model(model, cparams);
    if (!ctx) {
        snprintf(err, err_len, "Failed to create context");
        llama_free_model(model);
        return -1;
    }
    
    // Get vocab size
    int32_t n_vocab = llama_n_vocab(model);
    
    // Decode
    struct llama_batch batch = llama_batch_get_one(
        (llama_token *)tokens,  // Cast away const
        n_tokens,
        0,
        0
    );
    
    int32_t decode_result = llama_decode(ctx, batch);
    if (decode_result != 0) {
        snprintf(err, err_len, "Decoding failed");
        llama_free(ctx);
        llama_free_model(model);
        return -1;
    }
    
    // Setup output shape
    *out_rows = n_tokens;
    *out_cols = n_vocab;
    
    // Pass 1: Shape query
    if (!out_logits || logits_capacity <= 0) {
        llama_free(ctx);
        llama_free_model(model);
        return 0;
    }
    
    // Pass 2: Bounds check
    int32_t total_elements = n_tokens * n_vocab;
    if (logits_capacity < total_elements) {
        snprintf(err, err_len,
                 "Logits buffer too small (need %d, got %d)",
                 total_elements, logits_capacity);
        llama_free(ctx);
        llama_free_model(model);
        return -1;
    }
    
    // Copy logits
    float * logits = llama_get_logits(ctx);
    if (!logits) {
        snprintf(err, err_len, "Failed to extract logits");
        llama_free(ctx);
        llama_free_model(model);
        return -1;
    }
    
    std::memcpy(out_logits, logits, total_elements * sizeof(float));
    
    llama_free(ctx);
    llama_free_model(model);
    return 0;
}
```

## Build Configuration

### Compiler Flags
```bash
-I/home/steven/.cache/bitnet_cpp/3rdparty/llama.cpp/include
```

### Linker Flags
```bash
-L/home/steven/.cache/bitnet_cpp/build/lib
-lllama
-lggml
```

### Environment
```bash
export BITNET_CPP_DIR=/home/steven/.cache/bitnet_cpp
export LD_LIBRARY_PATH=$BITNET_CPP_DIR/build/lib:$LD_LIBRARY_PATH
```

## Key API Differences from Commented Code

| Wrapper Comment | Actual API | Notes |
|---|---|---|
| `bitnet_get_tokenizer()` | Use `llama_tokenize()` on model | API directly on model |
| `bitnet_tokenize_text()` | Use `llama_tokenize()` on model | Same as above |
| `bitnet_eval_all_positions()` | Set `logits_all=true` in context params | Not a function, a parameter |
| `llama_batch_get_one()` | Exact name ✓ | Available in llama.h |
| `llama_get_logits()` | Exact name ✓ | Available in llama.h |
| Session/caching | Per-call load/unload | MVP design (inefficient but stateless) |

## Critical Implementation Notes

1. **Always set `logits_all = true`** in context params to get all-position logits in a single decode call
2. **Two-pass tokenization pattern** is essential: NULL query then buffer fill
3. **Cast away const** on tokens array for llama_decode (API takes non-const pointer)
4. **Per-call model loading** is inefficient but matches MVP stateless design
5. **Row-major logits layout**: use `logits[pos * n_vocab + vocab_idx]` for indexing

## Future Optimization Opportunities

1. **Session API**: Cache models/contexts between calls (v0.2)
2. **GPU support**: Set `n_gpu_layers > 0` for GPU acceleration
3. **Batch processing**: Use `llama_batch_init()` for multi-sequence batches
4. **Static context**: Load model once at startup instead of per-call

## Validation

- [x] BitNet.cpp installed and compiled
- [x] llama.cpp headers available
- [x] C API fully documented
- [x] Model loading functions confirmed
- [x] Tokenization API confirmed
- [x] Inference API confirmed
- [x] Logits extraction confirmed
- [x] Two-pass pattern compatible
- [ ] GPU pathway (future)

## References

- llama.h: `/home/steven/.cache/bitnet_cpp/3rdparty/llama.cpp/include/llama.h`
- BitNet.cpp: `/home/steven/.cache/bitnet_cpp/`
- Example: `/home/steven/.cache/bitnet_cpp/3rdparty/llama.cpp/examples/main/main.cpp`
- Common lib: `/home/steven/.cache/bitnet_cpp/3rdparty/llama.cpp/common/common.cpp`


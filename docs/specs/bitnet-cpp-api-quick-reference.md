# BitNet.cpp API Quick Reference

**TL;DR**: Use **llama.cpp C API** from `/home/steven/.cache/bitnet_cpp/3rdparty/llama.cpp/include/llama.h`

## One-Liner Functions

| Task | Function | Header |
|------|----------|--------|
| **Load model** | `llama_load_model_from_file(path, params)` | llama.h:420 |
| **Create context** | `llama_new_context_with_model(model, params)` | llama.h:427 |
| **Tokenize** | `llama_tokenize(model, text, len, tokens, max, add_bos, parse_special)` | llama.h:939 |
| **Decode** | `llama_decode(ctx, batch)` | llama.h:821 |
| **Get logits** | `llama_get_logits(ctx)` | llama.h:857 |
| **Get vocab** | `llama_n_vocab(model)` | llama.h:448 |
| **Free model** | `llama_free_model(model)` | llama.h:424 |
| **Free context** | `llama_free(ctx)` | llama.h:432 |
| **Batch helper** | `llama_batch_get_one(tokens, n, pos, seq_id)` | llama.h:788 |

## Critical Parameters

### Model Loading
```c
struct llama_model_params mparams = llama_model_default_params();
mparams.use_mmap = true;    // Enable memory mapping
mparams.n_gpu_layers = 0;   // CPU only (for now)
```

### Context Creation
```c
struct llama_context_params cparams = llama_context_default_params();
cparams.n_ctx = 2048;       // Context size
cparams.logits_all = true;  // ALL POSITIONS (critical!)
cparams.n_threads = 4;      // CPU threads
```

## Minimal Tokenization Example

```c
// Query size
int n_tokens = llama_tokenize(model, "hello", 5, NULL, 0, true, false);
if (n_tokens < 0) n_tokens = -n_tokens;

// Allocate
int32_t *tokens = malloc(n_tokens * sizeof(int32_t));

// Fill
llama_tokenize(model, "hello", 5, tokens, n_tokens, true, false);
```

## Minimal Inference Example

```c
// Setup
struct llama_batch batch = llama_batch_get_one(tokens, n_tokens, 0, 0);
int32_t n_vocab = llama_n_vocab(model);

// Decode
llama_decode(ctx, batch);

// Extract
float *logits = llama_get_logits(ctx);  // [n_tokens][n_vocab]
```

## Error Handling Pattern

```c
if (!model) {
    snprintf(err, err_len, "Failed to load model");
    return -1;
}

if (llama_decode(ctx, batch) != 0) {
    snprintf(err, err_len, "Decode failed");
    return -1;
}
```

## Includes Required

```c
#include <llama.h>
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
```

## Link Flags

```
-L/home/steven/.cache/bitnet_cpp/build/lib -lllama -lggml
```

## Memory Management Summary

| Resource | Allocate | Free |
|----------|----------|------|
| Model | `llama_load_model_from_file()` | `llama_free_model()` |
| Context | `llama_new_context_with_model()` | `llama_free()` |
| Tokens | `malloc()` | `free()` |
| Batch | `llama_batch_get_one()` | (stack-allocated, no free needed) |

## Logits Layout

```
logits = [position 0][vocab 0..n]
         [position 1][vocab 0..n]
         ...
         [position n][vocab 0..n]

Access: logits[pos * n_vocab + vocab_idx]
```

## Two-Pass API Pattern (Used in Wrapper)

### Tokenization
```c
// Pass 1: NULL query
int32_t n = llama_tokenize(..., NULL, ...);

// Pass 2: Buffer fill
llama_tokenize(..., buffer, ...);
```

### Logits
```c
// Pass 1: NULL query
if (!out_logits) {
    *out_rows = n_tokens;
    *out_cols = n_vocab;
    return 0;  // Shape query only
}

// Pass 2: Buffer fill
float *logits = llama_get_logits(ctx);
memcpy(out_logits, logits, n_tokens * n_vocab * sizeof(float));
```

## BuildConfiguration

```bash
# Headers
-I/home/steven/.cache/bitnet_cpp/3rdparty/llama.cpp/include

# Libraries
-L/home/steven/.cache/bitnet_cpp/build/lib -lllama -lggml

# Environment
export BITNET_CPP_DIR=/home/steven/.cache/bitnet_cpp
export LD_LIBRARY_PATH=$BITNET_CPP_DIR/build/lib:$LD_LIBRARY_PATH
```

## No BitNet-Specific API Needed

- ❌ `bitnet_load()`
- ❌ `bitnet_tokenize()`
- ❌ `bitnet_eval()`
- ✅ Use `llama_*` functions directly

BitNet kernels are registered with GGML backend automatically on model load.


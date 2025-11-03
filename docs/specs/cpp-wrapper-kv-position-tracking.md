# C++ Wrapper KV Position Tracking - Technical Specification

**Component**: BitNet.cpp/LLaMA.cpp C++ Wrapper Position Tracking
**Location**: `crossval/src/bitnet_cpp_wrapper.cc`, `crossval/src/cpp_bindings.rs`
**Dependencies**: llama.cpp API (libbitnet.so, libllama.so)
**Version**: 1.0.0
**Date**: 2025-10-25

## Executive Summary

This specification defines the implementation of manual KV cache position tracking in the C++ wrapper to replace the removed `llama_get_kv_cache_token_count()` API. The design enables multi-turn conversation support and autoregressive generation by maintaining explicit position state in the persistent context structure, providing a migration path from stateless single-batch evaluation (Socket 0) to persistent context evaluation (Socket 1).

### Current State (v0.1 MVP)

- **Working**: Stateless single-pass evaluation via `llama_batch_get_one()` + `llama_decode()`
- **Limitation**: Per-call stateless evaluation; KV cache not reused across calls
- **Performance**: Suitable for full-batch parity tests, but suboptimal for streaming/multi-turn
- **API Removed**: `llama_get_kv_cache_token_count()` no longer available in llama.cpp

### Problem Statement

The llama.cpp API removed `llama_get_kv_cache_token_count()`, which was historically used to track:
- How many tokens are already in the KV cache
- The position for the next batch of tokens
- Prevention of redundant re-evaluation in multi-turn scenarios

**Without position tracking**:
```
Call 1: eval([prompt_tokens])           // Processes all tokens, fills KV cache
Call 2: eval([prompt_tokens, new_token]) // RECOMPUTES prompt_tokens unnecessarily!
```

**With manual position tracking** (this spec):
```
Call 1: eval([prompt_tokens])      // n_past = len(prompt_tokens)
Call 2: eval([new_token])          // Only decodes new_token, uses cached context
```

### Solution: Manual n_past Tracking (Socket 1)

This specification extends the Socket 1 persistent context structure with:
1. **Manual position counter** (`n_past`) tracking tokens in KV cache
2. **Position validation** to prevent context overflow
3. **Reset semantics** for new conversations
4. **Migration path** from Socket 0 (stateless) to Socket 1 (persistent)

---

## 1. Problem Statement

### 1.1 Background: Removed llama.cpp API

**Historically available** (llama.cpp v0.1-v0.2):
```cpp
// Query how many tokens are in KV cache
int n_past = llama_get_kv_cache_token_count(ctx);
```

**Current status** (llama.cpp v3+):
- API removed to simplify internal implementation
- KV cache management now application responsibility
- Manual tracking required for streaming/multi-turn scenarios

### 1.2 Current Stateless Evaluation Limitations

**Socket 0 Pattern** (`crossval_bitnet_eval_with_tokens`, Lines 192-326):
```cpp
// Current approach: Load model, evaluate tokens, return logits
int crossval_bitnet_eval_with_tokens(
    const char* model_path,         // Loads model EVERY call
    const int32_t* tokens,
    int32_t n_tokens,
    // ...
);
```

**Performance implications**:
- Model loading overhead: 100-500ms per call
- KV cache cleared between calls
- No context reuse across evaluations

**Use case limitations**:
- ❌ Multi-turn conversations (context reset each call)
- ❌ Streaming generation (no position continuity)
- ✅ Full-batch parity tests (evaluates all tokens at once)
- ✅ One-shot inference (no context reuse needed)

### 1.3 Need for Position Tracking

**Multi-turn scenario** (without tracking):
```
User: "What is the capital of France?"
  eval([BOS, What, is, the, capital, of, France, ?])  // 8 tokens
  -> Generate: "Paris"

User: "What about Germany?"
  eval([BOS, What, is, the, capital, of, France, ?, Paris, What, about, Germany, ?])  // 13 tokens
  -> RECOMPUTES first 9 tokens (prompt + "Paris") unnecessarily!
```

**Multi-turn scenario** (with position tracking):
```
User: "What is the capital of France?"
  eval([BOS, What, is, the, capital, of, France, ?])  // n_past = 8
  -> Generate: "Paris" (eval 1 token at a time, updating n_past)
  -> n_past = 9

User: "What about Germany?"
  eval([What, about, Germany, ?])  // Only 4 NEW tokens
  -> Uses KV cache for positions 0-8 (already cached)
  -> Only decodes positions 9-12 (new tokens)
```

**Performance benefit**:
- Prompt prefill: Amortized cost (one-time per conversation)
- Token generation: 10-100× faster (no redundant recomputation)
- Memory: Efficient KV cache reuse

---

## 2. Technical Design

### 2.1 Manual Position Tracking Architecture

**Core concept**: Application maintains `n_past` counter in persistent context.

**Lifecycle**:
1. **Initialization**: `n_past = 0` (empty KV cache)
2. **Evaluation**: Decode `n_tokens`, advance `n_past += n_tokens`
3. **Reset**: `n_past = 0` + `llama_kv_cache_clear()` (new conversation)

**Position semantics**:
- `n_past`: Number of tokens already in KV cache (valid range: `[0, n_ctx]`)
- `n_tokens`: Number of new tokens to decode in current batch
- `n_ctx`: Maximum context window size (configured at init)

**Validation**:
```cpp
if (n_past + n_tokens > n_ctx) {
    return ERROR_KV_CACHE_OVERFLOW;
}
```

### 2.2 Context Structure Extension

**Current Socket 1 structure** (Lines 334-341):
```cpp
struct bitnet_context_t {
#ifdef BITNET_AVAILABLE
    llama_model* model;
    llama_context* ctx;
    int32_t n_ctx;           // Maximum context window
    int32_t n_gpu_layers;
#endif
};
```

**Extended structure** (this spec):
```cpp
struct bitnet_context_t {
#ifdef BITNET_AVAILABLE
    llama_model* model;
    llama_context* ctx;
    int32_t n_ctx;           // Maximum context window
    int32_t n_gpu_layers;
    int32_t n_past;          // [NEW] Manual position tracking
#endif
};
```

**Field semantics**:
- `n_past`: Number of tokens currently in KV cache (starts at 0)
- Updated **only after successful** `llama_decode()` call
- Reset to 0 when starting new conversation

### 2.3 Socket 0 vs Socket 1 Comparison

| Aspect | Socket 0 (Stateless) | Socket 1 (Persistent) |
|--------|---------------------|---------------------|
| **Model Loading** | Per-call (100-500ms) | One-time (amortized) |
| **KV Cache** | Cleared each call | Persistent across calls |
| **Position Tracking** | N/A (stateless) | Manual `n_past` counter |
| **Use Cases** | Parity tests, one-shot | Multi-turn, streaming |
| **Context Size** | Per-call parameter | Configured at init |
| **Performance** | Baseline (100%) | 10-100× faster (multi-turn) |

### 2.4 Position Tracking Across Evaluation Phases

**Phase 1: Prompt Prefill** (large batch, one-time cost):
```cpp
// Initial state: n_past = 0
int32_t prompt_tokens[] = {1, 1234, 5678, 9012};  // 4 tokens
int32_t n_prompt = 4;

// Evaluate prompt
llama_batch batch = llama_batch_get_one(prompt_tokens, n_prompt);
int result = llama_decode(ctx->ctx, batch);

// Update position AFTER successful decode
if (result == 0) {
    ctx->n_past += n_prompt;  // n_past = 4
}
```

**Phase 2: Token Generation** (single token, repeated):
```cpp
// State: n_past = 4 (prompt cached)
int32_t new_token = sample_from_logits(...);  // Sample next token

// Evaluate single token
llama_batch batch = llama_batch_get_one(&new_token, 1);
int result = llama_decode(ctx->ctx, batch);

// Update position
if (result == 0) {
    ctx->n_past += 1;  // n_past = 5
}

// Repeat for each generated token
```

**Phase 3: New Conversation** (reset):
```cpp
// Clear KV cache and reset position
ctx->n_past = 0;
llama_kv_cache_clear(ctx->ctx);  // If available in llama.cpp API
```

### 2.5 Position Validation and Overflow Handling

**Validation before decode**:
```cpp
int validate_position(const bitnet_context_t* ctx, int32_t n_tokens) {
    if (ctx->n_past < 0 || ctx->n_past > ctx->n_ctx) {
        return ERROR_INVALID_POSITION;  // Corrupted state
    }
    if (n_tokens <= 0) {
        return ERROR_INVALID_TOKEN_COUNT;
    }
    if (ctx->n_past + n_tokens > ctx->n_ctx) {
        return ERROR_KV_CACHE_OVERFLOW;  // Not enough space
    }
    return 0;
}
```

**Error handling patterns**:
```cpp
// Pattern 1: Reject overflow (strict)
if (ctx->n_past + n_tokens > ctx->n_ctx) {
    snprintf(err, err_len,
             "KV cache overflow: n_past=%d + n_tokens=%d > n_ctx=%d",
             ctx->n_past, n_tokens, ctx->n_ctx);
    return -1;
}

// Pattern 2: Auto-reset (lenient - NOT RECOMMENDED for v0.2)
if (ctx->n_past + n_tokens > ctx->n_ctx) {
    ctx->n_past = 0;
    llama_kv_cache_clear(ctx->ctx);
    // Continue with fresh context
}

// Pattern 3: Sliding window (advanced - v0.3+)
if (ctx->n_past + n_tokens > ctx->n_ctx) {
    // Shift KV cache window, discard old tokens
    // Requires llama.cpp API support
}
```

---

## 3. API Design

### 3.1 Context Initialization (Socket 1)

**Already implemented** (Lines 357-442), requires field initialization:

```cpp
int bitnet_cpp_init_context(
    bitnet_context_t** out_ctx,
    const char* model_path,
    int32_t n_ctx,
    int32_t n_gpu_layers,
    char* err,
    int32_t err_len
) {
    // ... existing model/context creation ...

    ctx->model = llama_load_model_from_file(model_path, model_params);
    ctx->ctx = llama_new_context_with_model(ctx->model, ctx_params);
    ctx->n_ctx = n_ctx;
    ctx->n_gpu_layers = n_gpu_layers;

    // [NEW] Initialize position tracking
    ctx->n_past = 0;  // Empty KV cache

    *out_ctx = ctx;
    return 0;
}
```

### 3.2 Evaluation with Position Tracking (Socket 3)

**Current implementation** (`bitnet_cpp_eval_with_context`, Lines 648-759):
```cpp
int bitnet_cpp_eval_with_context(
    const bitnet_context_t* ctx,
    const int32_t* tokens,
    int32_t n_tokens,
    int32_t seq_id,              // [UNUSED in v0.1]
    float* out_logits,
    int32_t logits_capacity,
    int32_t* out_rows,
    int32_t* out_cols,
    char* err,
    int32_t err_len
);
```

**Modified implementation** (with position tracking):
```cpp
int bitnet_cpp_eval_with_context(
    bitnet_context_t* ctx,           // [CHANGED] Non-const for n_past update
    const int32_t* tokens,
    int32_t n_tokens,
    int32_t seq_id,
    float* out_logits,
    int32_t logits_capacity,
    int32_t* out_rows,
    int32_t* out_cols,
    char* err,
    int32_t err_len
) {
    // ... existing validation ...

    // [NEW] Validate position before decode
    if (ctx->n_past + n_tokens > ctx->n_ctx) {
        snprintf(err, err_len,
                 "KV cache overflow: n_past=%d + n_tokens=%d > n_ctx=%d",
                 ctx->n_past, n_tokens, ctx->n_ctx);
        return -1;
    }

    // [NEW] Log position tracking (debug builds)
    #ifdef BITNET_DEBUG
    fprintf(stderr, "DEBUG: eval n_past=%d, n_tokens=%d, n_ctx=%d\n",
            ctx->n_past, n_tokens, ctx->n_ctx);
    #endif

    // Existing batch creation and decode
    llama_batch batch = llama_batch_get_one(const_cast<int32_t*>(tokens), n_tokens);
    int result = llama_decode(ctx->ctx, batch);

    if (result != 0) {
        snprintf(err, err_len, "Batch decode failed (result=%d)", result);
        return -1;
    }

    // [NEW] Update position AFTER successful decode
    ctx->n_past += n_tokens;

    // Existing logits extraction
    for (int32_t i = 0; i < n_tokens; ++i) {
        float* logits_for_pos = llama_get_logits_ith(ctx->ctx, i);
        std::memcpy(&out_logits[i * n_vocab], logits_for_pos, n_vocab * sizeof(float));
    }

    return 0;
}
```

**Breaking change**: `ctx` parameter must be **non-const** to allow `n_past` updates.

### 3.3 Context Reset (Socket 1 Extension)

**New function** (add to wrapper):
```cpp
/// Reset KV cache position (start new conversation)
///
/// Args:
///   ctx: Context handle to reset
///
/// Returns: 0 on success, -1 on error
int bitnet_cpp_reset_context(
    bitnet_context_t* ctx,
    char* err,
    int32_t err_len
) {
    if (!ctx || !err || err_len <= 0) {
        if (err && err_len > 0) {
            snprintf(err, err_len, "bitnet_cpp_reset_context: NULL parameter");
        }
        return -1;
    }

    err[0] = '\0';

#ifdef BITNET_STUB
    snprintf(err, err_len, "bitnet_cpp_reset_context: STUB mode");
    return -1;

#elif defined(BITNET_AVAILABLE)
    if (!ctx->ctx) {
        snprintf(err, err_len, "bitnet_cpp_reset_context: Invalid context");
        return -1;
    }

    // Reset position tracking
    ctx->n_past = 0;

    // Clear KV cache (if available in llama.cpp API)
    llama_kv_cache_clear(ctx->ctx);

    return 0;

#else
    #error "Must define either BITNET_STUB or BITNET_AVAILABLE"
#endif
}
```

### 3.4 Position Query (Socket 1 Extension - Optional)

**New function** (read-only query):
```cpp
/// Query current KV cache position
///
/// Args:
///   ctx: Context handle
///   out_n_past: [out] Current position (number of cached tokens)
///
/// Returns: 0 on success, -1 on error
int bitnet_cpp_get_position(
    const bitnet_context_t* ctx,
    int32_t* out_n_past,
    char* err,
    int32_t err_len
) {
    if (!ctx || !out_n_past || !err || err_len <= 0) {
        if (err && err_len > 0) {
            snprintf(err, err_len, "bitnet_cpp_get_position: NULL parameter");
        }
        return -1;
    }

    *out_n_past = 0;
    err[0] = '\0';

#ifdef BITNET_STUB
    snprintf(err, err_len, "bitnet_cpp_get_position: STUB mode");
    return -1;

#elif defined(BITNET_AVAILABLE)
    *out_n_past = ctx->n_past;
    return 0;

#else
    #error "Must define either BITNET_STUB or BITNET_AVAILABLE"
#endif
}
```

---

## 4. Implementation Strategy

### 4.1 Phase 1: Enable Socket 1 with Manual n_past (v0.2)

**Goals**:
- Add `n_past` field to `bitnet_context_t` structure
- Initialize to 0 in `bitnet_cpp_init_context()`
- Update after successful decode in `bitnet_cpp_eval_with_context()`
- Add validation for KV cache overflow

**Changes required**:
```diff
 struct bitnet_context_t {
 #ifdef BITNET_AVAILABLE
     llama_model* model;
     llama_context* ctx;
     int32_t n_ctx;
     int32_t n_gpu_layers;
+    int32_t n_past;          // Manual position tracking
 #endif
 };
```

```diff
 int bitnet_cpp_init_context(...) {
     // ... existing code ...
     ctx->n_ctx = n_ctx;
     ctx->n_gpu_layers = n_gpu_layers;
+    ctx->n_past = 0;  // Initialize empty KV cache
     *out_ctx = ctx;
     return 0;
 }
```

```diff
 int bitnet_cpp_eval_with_context(
-    const bitnet_context_t* ctx,
+    bitnet_context_t* ctx,  // Non-const for n_past update
     const int32_t* tokens,
     int32_t n_tokens,
     // ...
 ) {
     // ... existing validation ...

+    // Validate position
+    if (ctx->n_past + n_tokens > ctx->n_ctx) {
+        snprintf(err, err_len, "KV cache overflow: n_past=%d + n_tokens=%d > n_ctx=%d",
+                 ctx->n_past, n_tokens, ctx->n_ctx);
+        return -1;
+    }

     // ... existing decode ...

+    // Update position after successful decode
+    if (result == 0) {
+        ctx->n_past += n_tokens;
+    }

     return 0;
 }
```

**Testing approach**:
1. Verify initialization: `n_past = 0` after `bitnet_cpp_init_context()`
2. Verify increment: `n_past += n_tokens` after `bitnet_cpp_eval_with_context()`
3. Verify overflow: Error when `n_past + n_tokens > n_ctx`

### 4.2 Phase 2: Add Context Reset API (v0.2)

**Goals**:
- Implement `bitnet_cpp_reset_context()` for new conversations
- Clear KV cache via `llama_kv_cache_clear()` (if available)
- Reset `n_past = 0`

**New function**:
```cpp
int bitnet_cpp_reset_context(
    bitnet_context_t* ctx,
    char* err,
    int32_t err_len
);
```

**Rust FFI binding**:
```rust
#[cfg(feature = "ffi")]
unsafe extern "C" {
    fn bitnet_cpp_reset_context(
        ctx: *mut BitnetContext,
        err: *mut c_char,
        err_len: i32,
    ) -> c_int;
}
```

**Testing approach**:
1. Fill KV cache (`n_past = 100`)
2. Call `bitnet_cpp_reset_context()`
3. Verify `n_past = 0` after reset
4. Verify can evaluate new prompt without overflow

### 4.3 Phase 3: Migrate Existing Code to Socket 1 (v0.2)

**Goals**:
- Update `BitnetSession` wrapper to use persistent context
- Replace Socket 0 calls with Socket 1 calls
- Add position tracking to Rust wrapper

**Rust wrapper updates**:
```rust
pub struct BitnetSession {
    ctx: *mut BitnetContext,
    model_path: PathBuf,
    n_ctx: i32,
    n_past: i32,  // [NEW] Track position on Rust side (optional mirror)
}

impl BitnetSession {
    /// Evaluate tokens with automatic position tracking
    pub fn evaluate(&mut self, tokens: &[i32]) -> Result<Vec<Vec<f32>>> {
        // Validation
        if self.n_past + tokens.len() as i32 > self.n_ctx {
            return Err(anyhow::anyhow!(
                "KV cache overflow: n_past={} + n_tokens={} > n_ctx={}",
                self.n_past, tokens.len(), self.n_ctx
            ));
        }

        // Delegate to C++ wrapper (Socket 3)
        let logits = self.eval_with_context_raw(tokens)?;

        // Update Rust-side position tracking (optional mirror)
        self.n_past += tokens.len() as i32;

        Ok(logits)
    }

    /// Reset KV cache (start new conversation)
    pub fn reset(&mut self) -> Result<()> {
        let mut err_buf = vec![0u8; 512];

        let result = unsafe {
            bitnet_cpp_reset_context(
                self.ctx,
                err_buf.as_mut_ptr() as *mut c_char,
                err_buf.len() as i32,
            )
        };

        if result != 0 {
            let error_msg = extract_error_message(&err_buf);
            return Err(anyhow::anyhow!("Reset failed: {}", error_msg));
        }

        self.n_past = 0;  // Reset Rust-side tracking
        Ok(())
    }
}
```

**Migration checklist**:
- [ ] Update `BitnetSession::create()` to initialize `n_past = 0`
- [ ] Update `BitnetSession::evaluate()` to track and validate position
- [ ] Add `BitnetSession::reset()` for new conversations
- [ ] Update tests to use persistent sessions (no per-call model loading)
- [ ] Verify parity tests still pass (no behavior change for full-batch eval)

### 4.4 Phase 4: Deprecate Socket 0 (Optional - v0.3)

**Decision point**: Keep Socket 0 for backward compatibility or remove?

**Option A: Keep Socket 0** (recommended for v0.2):
- Maintains backward compatibility
- Useful for one-shot/stateless use cases
- No migration burden on existing code

**Option B: Remove Socket 0** (v0.3+):
- Simplifies API surface
- Forces users to persistent sessions (better performance)
- Requires migration guide

**Recommendation**: Keep Socket 0 in v0.2, evaluate deprecation for v0.3.

---

## 5. Code Examples

### 5.1 Before: Stateless Evaluation (Socket 0)

**Current approach**:
```rust
// Socket 0: Load model, evaluate, discard context
let tokens = vec![1, 2, 3, 4];
let logits = crossval_bitnet_eval_with_tokens(
    "model.gguf",
    &tokens,
    2048,  // n_ctx
    // ...
)?;

// Next call: RELOADS model (100-500ms overhead)
let tokens2 = vec![1, 2, 3, 4, 5];
let logits2 = crossval_bitnet_eval_with_tokens(
    "model.gguf",
    &tokens2,
    2048,
    // ...
)?;  // Redundant model load + token recomputation!
```

### 5.2 After: Persistent Context (Socket 1)

**New approach with position tracking**:
```rust
// Socket 1: Create persistent session (one-time model load)
let mut session = BitnetSession::create(
    Path::new("model.gguf"),
    2048,  // n_ctx
    0,     // n_gpu_layers
)?;

// Evaluate prompt (prefill)
let prompt_tokens = vec![1, 2, 3, 4];
let prompt_logits = session.evaluate(&prompt_tokens)?;
// After: session.n_past = 4 (prompt cached)

// Generate next token (efficient)
let next_token = vec![5];
let next_logits = session.evaluate(&next_token)?;
// After: session.n_past = 5 (only 1 new token decoded)

// Start new conversation
session.reset()?;
// After: session.n_past = 0 (KV cache cleared)
```

### 5.3 Position Initialization (Prompt)

**Pattern**: Prefill prompt, establish KV cache baseline
```cpp
// C++ side
bitnet_context_t* ctx = ...;  // n_past = 0

int32_t prompt[] = {1, 1234, 5678, 9012};  // 4 tokens
int32_t n_prompt = 4;

// Evaluate prompt in single batch
int result = bitnet_cpp_eval_with_context(
    ctx,
    prompt,
    n_prompt,
    0,  // seq_id
    logits_buffer,
    logits_capacity,
    &out_rows,
    &out_cols,
    err_buf,
    sizeof(err_buf)
);

// After successful eval: ctx->n_past = 4
```

```rust
// Rust side
let mut session = BitnetSession::create(...)?;

let prompt_tokens = vec![1, 1234, 5678, 9012];
let prompt_logits = session.evaluate(&prompt_tokens)?;

assert_eq!(session.n_past, 4);  // Position advanced
```

### 5.4 Position Increment (Autoregressive)

**Pattern**: Generate one token at a time, efficient KV cache reuse
```cpp
// C++ side (autoregressive loop)
bitnet_context_t* ctx = ...;  // n_past = 4 (prompt cached)

for (int step = 0; step < max_tokens; ++step) {
    // Sample next token from logits
    int32_t next_token = sample_from_logits(...);

    // Evaluate single token (uses cached context)
    int result = bitnet_cpp_eval_with_context(
        ctx,
        &next_token,
        1,  // n_tokens = 1
        0,  // seq_id
        logits_buffer,
        logits_capacity,
        &out_rows,
        &out_cols,
        err_buf,
        sizeof(err_buf)
    );

    // After: ctx->n_past increments by 1 each iteration
}
```

```rust
// Rust side (autoregressive loop)
let mut session = BitnetSession::create(...)?;

// Prefill prompt
session.evaluate(&prompt_tokens)?;

// Generate tokens
for step in 0..max_tokens {
    let next_token = sample_from_logits(...);
    let logits = session.evaluate(&[next_token])?;

    // Position automatically increments
    assert_eq!(session.n_past, prompt_tokens.len() as i32 + step + 1);
}
```

### 5.5 Position Reset (New Conversation)

**Pattern**: Clear KV cache, start fresh context
```cpp
// C++ side
bitnet_context_t* ctx = ...;  // n_past = 100 (previous conversation)

// Reset for new conversation
int result = bitnet_cpp_reset_context(ctx, err_buf, sizeof(err_buf));
if (result != 0) {
    fprintf(stderr, "Reset failed: %s\n", err_buf);
    return -1;
}

// After: ctx->n_past = 0, KV cache cleared

// Start new prompt
int32_t new_prompt[] = {1, 42, 43, 44};
result = bitnet_cpp_eval_with_context(ctx, new_prompt, 4, ...);
// After: ctx->n_past = 4
```

```rust
// Rust side
let mut session = BitnetSession::create(...)?;

// First conversation
session.evaluate(&conversation1_tokens)?;
// session.n_past = 100

// Reset for new conversation
session.reset()?;
assert_eq!(session.n_past, 0);

// Start new prompt
session.evaluate(&conversation2_tokens)?;
assert_eq!(session.n_past, conversation2_tokens.len() as i32);
```

### 5.6 Position Overflow Handling

**Pattern**: Validate before evaluation, reject overflow
```rust
let mut session = BitnetSession::create(
    Path::new("model.gguf"),
    512,  // n_ctx = 512
    0,
)?;

// Evaluate 500 tokens (OK)
let tokens1 = vec![1; 500];
session.evaluate(&tokens1)?;
assert_eq!(session.n_past, 500);

// Try to evaluate 20 more tokens (OVERFLOW!)
let tokens2 = vec![2; 20];
let result = session.evaluate(&tokens2);

// Should fail with overflow error
assert!(result.is_err());
assert!(result.unwrap_err().to_string().contains("KV cache overflow"));

// Position unchanged after failed evaluation
assert_eq!(session.n_past, 500);

// Solution: Reset and start fresh
session.reset()?;
session.evaluate(&tokens2)?;  // OK now
assert_eq!(session.n_past, 20);
```

---

## 6. Testing Requirements

### 6.1 Unit Tests: Position Tracking

**Test: Position initialization**
```rust
#[test]
fn test_position_init() -> Result<()> {
    let session = BitnetSession::create(
        Path::new("models/test.gguf"),
        512,
        0,
    )?;

    assert_eq!(session.n_past, 0);  // Empty KV cache
    Ok(())
}
```

**Test: Position increment**
```rust
#[test]
fn test_position_increment() -> Result<()> {
    let mut session = BitnetSession::create(...)?;

    // Evaluate 4 tokens
    let tokens1 = vec![1, 2, 3, 4];
    session.evaluate(&tokens1)?;
    assert_eq!(session.n_past, 4);

    // Evaluate 2 more tokens
    let tokens2 = vec![5, 6];
    session.evaluate(&tokens2)?;
    assert_eq!(session.n_past, 6);

    Ok(())
}
```

**Test: Position validation**
```rust
#[test]
fn test_position_overflow() -> Result<()> {
    let mut session = BitnetSession::create(
        Path::new("models/test.gguf"),
        10,  // Small context
        0,
    )?;

    // Fill context to capacity
    let tokens1 = vec![1; 10];
    session.evaluate(&tokens1)?;
    assert_eq!(session.n_past, 10);

    // Try to overflow (should fail)
    let tokens2 = vec![2];
    let result = session.evaluate(&tokens2);
    assert!(result.is_err());

    // Position unchanged
    assert_eq!(session.n_past, 10);

    Ok(())
}
```

### 6.2 Integration Tests: Multi-Turn Evaluation

**Test: Multi-turn conversation**
```rust
#[test]
fn test_multi_turn_conversation() -> Result<()> {
    let mut session = BitnetSession::create(...)?;

    // Turn 1: Prompt
    let prompt = vec![1, 2, 3, 4];
    let logits1 = session.evaluate(&prompt)?;
    assert_eq!(session.n_past, 4);

    // Turn 2: Generate response
    for _ in 0..5 {
        let next_token = sample_from_logits(&logits1[logits1.len() - 1]);
        let logits_new = session.evaluate(&[next_token])?;
        // Position increments each step
    }
    assert_eq!(session.n_past, 9);  // 4 prompt + 5 generated

    // Turn 3: Continue conversation
    let followup = vec![10, 11];
    session.evaluate(&followup)?;
    assert_eq!(session.n_past, 11);

    Ok(())
}
```

**Test: Context reset**
```rust
#[test]
fn test_context_reset() -> Result<()> {
    let mut session = BitnetSession::create(...)?;

    // First conversation
    session.evaluate(&vec![1, 2, 3, 4])?;
    assert_eq!(session.n_past, 4);

    // Reset
    session.reset()?;
    assert_eq!(session.n_past, 0);

    // New conversation (should not overflow)
    session.evaluate(&vec![5, 6, 7, 8])?;
    assert_eq!(session.n_past, 4);

    Ok(())
}
```

### 6.3 Parity Tests: Rust vs C++ with Position Tracking

**Test: Single-turn parity (backward compatibility)**
```rust
#[test]
fn test_single_turn_parity() -> Result<()> {
    let mut cpp_session = BitnetSession::create(...)?;

    // Evaluate entire sequence at once (like Socket 0)
    let tokens = vec![1, 2, 3, 4];
    let cpp_logits = cpp_session.evaluate(&tokens)?;

    // Compare with Rust implementation
    let rust_logits = eval_rust_inference(&tokens)?;

    // Should have exact parity
    assert_logits_match(&rust_logits, &cpp_logits)?;

    // Position tracking should work transparently
    assert_eq!(cpp_session.n_past, 4);

    Ok(())
}
```

**Test: Multi-turn parity**
```rust
#[test]
fn test_multi_turn_parity() -> Result<()> {
    let mut cpp_session = BitnetSession::create(...)?;

    // Turn 1: Prefill prompt
    let prompt = vec![1, 2, 3, 4];
    let cpp_logits1 = cpp_session.evaluate(&prompt)?;

    // Turn 2: Generate token
    let next_token = vec![5];
    let cpp_logits2 = cpp_session.evaluate(&next_token)?;

    // Compare with Rust (full evaluation)
    let rust_logits_full = eval_rust_inference(&vec![1, 2, 3, 4, 5])?;

    // Last position should match
    assert_logits_match(
        &rust_logits_full[rust_logits_full.len() - 1],
        &cpp_logits2[0],
    )?;

    Ok(())
}
```

### 6.4 Performance Tests: KV Cache Reuse

**Test: Measure overhead reduction**
```rust
#[test]
fn test_kv_cache_performance() -> Result<()> {
    let mut session = BitnetSession::create(...)?;

    // Measure: Full evaluation (no KV cache reuse)
    let prompt = vec![1; 100];
    let start1 = Instant::now();
    let logits1 = session.evaluate(&prompt)?;
    let time1 = start1.elapsed();

    // Measure: Incremental evaluation (with KV cache reuse)
    session.reset()?;
    session.evaluate(&prompt)?;  // Prefill

    let start2 = Instant::now();
    for _ in 0..10 {
        let next = vec![42];
        session.evaluate(&next)?;
    }
    let time2 = start2.elapsed();

    // Should be much faster (no prompt recomputation)
    eprintln!("Full eval (100 tokens): {:?}", time1);
    eprintln!("Incremental (10 tokens): {:?}", time2);
    assert!(time2 < time1 / 5);  // At least 5× faster

    Ok(())
}
```

---

## 7. Performance Implications

### 7.1 Socket 0 vs Socket 1 Overhead Analysis

**Socket 0 (Stateless)** - Baseline:
- Model loading: 100-500ms per call
- Token evaluation: ~5ms per token (CPU)
- KV cache: Cleared between calls
- **Total overhead**: 100-500ms fixed + 5ms/token

**Socket 1 (Persistent)** - Optimized:
- Model loading: One-time (amortized)
- Token evaluation: ~5ms per token (CPU)
- KV cache: Reused across calls
- **Total overhead**: 5ms/token (no fixed cost)

**Performance ratio** (multi-turn scenario):
```
# 10-turn conversation, 10 tokens per turn
Socket 0: (100ms + 10*5ms) * 10 = 1500ms
Socket 1: 100ms + 100*5ms       = 600ms
Speedup: 2.5×

# 100-turn conversation
Socket 0: (100ms + 10*5ms) * 100 = 15000ms
Socket 1: 100ms + 1000*5ms        = 5100ms
Speedup: 2.9×
```

### 7.2 Context Persistence Benefits

**KV Cache Reuse** (autoregressive generation):
- Without tracking: Recompute prompt every token (O(n²))
- With tracking: Compute prompt once, decode incrementally (O(n))

**Example**: Generate 100 tokens from 50-token prompt
- Without KV cache: 50 + 51 + 52 + ... + 150 = 10,050 token computations
- With KV cache: 50 (prefill) + 100 (decode) = 150 token computations
- **Speedup**: 67× (theoretical maximum)

**Real-world speedup** (observed):
- LLaMA-3 8B: 10-20× faster for generation (prompt caching)
- BitNet 2B: 5-15× faster for generation (depending on prompt length)

### 7.3 Memory Usage Considerations

**KV Cache Size**:
- Per token: 2 * n_layers * n_embd * sizeof(float)
- Example (LLaMA-3 8B): 2 * 32 * 4096 * 4 = 1MB per token
- Context size 2048: ~2GB KV cache memory

**Trade-off**:
- Stateless (Socket 0): No persistent memory, but redundant computation
- Persistent (Socket 1): Higher memory, but much faster inference

**Recommendation**: Use Socket 1 for all multi-turn scenarios (memory cost is justified).

---

## 8. Acceptance Criteria

### 8.1 Functional Requirements

**FR1: Position Initialization**
- [ ] `n_past = 0` after `bitnet_cpp_init_context()`
- [ ] Position query returns 0 on fresh context

**FR2: Position Increment**
- [ ] `n_past += n_tokens` after successful `bitnet_cpp_eval_with_context()`
- [ ] Position unchanged on failed evaluation

**FR3: Position Validation**
- [ ] Error when `n_past + n_tokens > n_ctx`
- [ ] Error message includes `n_past`, `n_tokens`, `n_ctx` values

**FR4: Context Reset**
- [ ] `n_past = 0` after `bitnet_cpp_reset_context()`
- [ ] KV cache cleared (if API available)
- [ ] Can evaluate new prompt after reset without overflow

**FR5: Multi-Turn Support**
- [ ] Prefill prompt, generate tokens incrementally
- [ ] No redundant prompt recomputation
- [ ] Position tracking accurate across turns

### 8.2 API Requirements

**AR1: Socket 1 Compatibility**
- [ ] Existing Socket 1 functions work with position tracking
- [ ] `bitnet_cpp_init_context()` initializes `n_past = 0`
- [ ] `bitnet_cpp_free_context()` releases resources correctly

**AR2: Socket 3 Compatibility**
- [ ] `bitnet_cpp_eval_with_context()` signature change (non-const `ctx`)
- [ ] Backward compatibility for full-batch evaluation (parity tests)
- [ ] Position tracking transparent to caller

**AR3: New APIs**
- [ ] `bitnet_cpp_reset_context()` implemented and tested
- [ ] `bitnet_cpp_get_position()` implemented (optional)
- [ ] FFI bindings match C function signatures

### 8.3 Parity Requirements

**PR1: Single-Turn Parity**
- [ ] Full-batch evaluation produces identical logits (Rust vs C++)
- [ ] Position tracking does not affect parity test results
- [ ] Cosine similarity > 0.9999 for all positions

**PR2: Multi-Turn Parity**
- [ ] Incremental evaluation (prompt + tokens) matches full evaluation
- [ ] Last position logits match between approaches
- [ ] No divergence introduced by position tracking

**PR3: Performance Parity**
- [ ] No performance regression in single-turn case
- [ ] 10-100× speedup in multi-turn case (expected)
- [ ] Parity with C++ reference implementation performance

### 8.4 Testing Requirements

**TR1: Unit Test Coverage**
- [ ] Position initialization test
- [ ] Position increment test
- [ ] Position validation test (overflow)
- [ ] Position reset test
- [ ] Position query test (if implemented)

**TR2: Integration Test Coverage**
- [ ] Multi-turn conversation test
- [ ] Context reset integration test
- [ ] Autoregressive generation test
- [ ] Performance benchmark (KV cache reuse)

**TR3: Parity Test Coverage**
- [ ] Single-turn parity test (backward compatibility)
- [ ] Multi-turn parity test (incremental vs full)
- [ ] Position tracking transparency test

### 8.5 Documentation Requirements

**DR1: API Documentation**
- [ ] Socket 1 functions documented with position tracking behavior
- [ ] Socket 3 signature change documented (non-const `ctx`)
- [ ] Reset API usage examples

**DR2: Migration Guide**
- [ ] Socket 0 → Socket 1 migration guide
- [ ] Example code for multi-turn scenarios
- [ ] Performance comparison (before/after)

**DR3: Architecture Documentation**
- [ ] Position tracking design rationale
- [ ] KV cache management strategy
- [ ] Overflow handling patterns

---

## 9. Migration Path

### 9.1 Backward Compatibility

**Socket 0 (Stateless)**: Keep as-is for backward compatibility
- No breaking changes to `crossval_bitnet_eval_with_tokens()`
- Useful for one-shot inference and parity tests
- Can coexist with Socket 1

**Socket 1 (Persistent)**: Extend with position tracking
- **Breaking change**: `bitnet_cpp_eval_with_context()` takes non-const `ctx`
- Rust wrapper must be updated to handle mutable context
- All existing Socket 1 users must update to new signature

**Migration timeline**:
- v0.2: Add position tracking to Socket 1, keep Socket 0 as fallback
- v0.3: Deprecate Socket 0 (optional), recommend Socket 1 for all use cases

### 9.2 Migration Guide for Users

**Before (Socket 0)**:
```rust
// Stateless evaluation (reloads model each call)
let logits1 = crossval_bitnet_eval_with_tokens("model.gguf", &tokens1, ...)?;
let logits2 = crossval_bitnet_eval_with_tokens("model.gguf", &tokens2, ...)?;
```

**After (Socket 1)**:
```rust
// Persistent session (one-time model load)
let mut session = BitnetSession::create(Path::new("model.gguf"), 2048, 0)?;

// Evaluate multiple batches (reuses model and KV cache)
let logits1 = session.evaluate(&tokens1)?;
let logits2 = session.evaluate(&tokens2)?;

// Reset when starting new conversation
session.reset()?;
```

**Migration checklist**:
1. Replace stateless calls with persistent session creation
2. Update mutable references for session methods
3. Add explicit reset calls between conversations
4. Validate context size (`n_ctx`) is sufficient for use case
5. Handle overflow errors gracefully

---

## 10. Risk Assessment

### 10.1 Breaking Changes

**Risk**: Socket 1 signature change (`const ctx` → `ctx`)
- **Impact**: All Socket 1 users must recompile and update Rust code
- **Mitigation**: Clear migration guide, deprecation warnings
- **Severity**: Medium (breaking change, but limited API surface)

### 10.2 Position Tracking Bugs

**Risk**: Incorrect `n_past` tracking causes divergence
- **Impact**: Parity tests fail, incorrect inference results
- **Mitigation**: Comprehensive unit tests, position validation
- **Severity**: High (correctness issue)

**Specific scenarios to test**:
- Position increment after failed decode (should not increment)
- Position validation edge cases (exactly at boundary)
- Position reset idempotency (reset twice should work)

### 10.3 KV Cache API Availability

**Risk**: `llama_kv_cache_clear()` not available in all llama.cpp versions
- **Impact**: Cannot fully clear KV cache on reset
- **Mitigation**: Conditional compilation, graceful degradation
- **Severity**: Low (can recreate context as fallback)

**Fallback strategy**:
```cpp
#ifdef LLAMA_HAS_KV_CACHE_CLEAR
    llama_kv_cache_clear(ctx->ctx);
#else
    // Fallback: Recreate context (slower, but correct)
    llama_free(ctx->ctx);
    ctx->ctx = llama_new_context_with_model(ctx->model, ctx_params);
#endif
```

### 10.4 Performance Regression

**Risk**: Position tracking overhead slows down single-turn inference
- **Impact**: Parity tests run slower
- **Mitigation**: Benchmark before/after, optimize hot paths
- **Severity**: Low (expected overhead is negligible)

**Benchmarking plan**:
- Measure single-turn latency before/after position tracking
- Target: < 1% overhead in single-turn case
- If exceeded: Profile and optimize validation logic

---

## 11. Future Work

### 11.1 Sliding Window KV Cache (v0.3+)

**Problem**: Fixed context size limits conversation length
**Solution**: Shift KV cache window when reaching capacity

```cpp
// Proposed API (requires llama.cpp support)
int llama_kv_cache_shift(
    llama_context* ctx,
    int n_shift  // Number of tokens to discard from beginning
);
```

**Usage**:
```cpp
// When approaching context limit:
if (ctx->n_past + n_tokens > ctx->n_ctx - 128) {
    int n_shift = ctx->n_ctx / 2;  // Shift half the context
    llama_kv_cache_shift(ctx->ctx, n_shift);
    ctx->n_past -= n_shift;
}
```

### 11.2 Multi-Sequence Batching (v0.3+)

**Problem**: Cannot process multiple conversations in parallel
**Solution**: Track position per sequence ID

```cpp
struct bitnet_context_t {
    llama_model* model;
    llama_context* ctx;
    int32_t n_ctx;
    int32_t n_gpu_layers;

    // [PROPOSED] Per-sequence position tracking
    std::unordered_map<int32_t, int32_t> seq_positions;
};
```

**API extension**:
```cpp
int bitnet_cpp_eval_with_sequence(
    bitnet_context_t* ctx,
    const int32_t* tokens,
    int32_t n_tokens,
    int32_t seq_id,          // Sequence identifier
    // ...
) {
    // Get position for this sequence
    int32_t n_past = ctx->seq_positions[seq_id];

    // Validate and decode
    // ...

    // Update per-sequence position
    ctx->seq_positions[seq_id] = n_past + n_tokens;
}
```

### 11.3 Speculative Decoding (v0.4+)

**Problem**: Autoregressive generation is sequential (no parallelism)
**Solution**: Speculative draft + verification with position rollback

```cpp
// Draft phase: Generate n tokens speculatively
int draft_tokens[10];
draft_model_generate(draft_tokens, 10);

// Verification phase: Check how many match
int n_verified = verify_against_main_model(draft_tokens, 10);

// Rollback position if draft was wrong
ctx->n_past = original_n_past + n_verified;
```

---

## 12. Appendices

### Appendix A: llama.cpp API Changes

**Removed APIs** (no longer available):
```cpp
int llama_get_kv_cache_token_count(llama_context* ctx);  // REMOVED
```

**Replacement**: Manual position tracking (this specification)

**Available KV Cache APIs** (llama.cpp v3+):
```cpp
void llama_kv_cache_clear(llama_context* ctx);
// Clears all KV cache entries (reset to empty)

// Note: llama_kv_cache_shift() may be available in future versions
```

### Appendix B: Position Tracking Invariants

**Invariant 1**: `0 ≤ n_past ≤ n_ctx`
- Position must be non-negative and within context bounds

**Invariant 2**: `n_past` increments only on successful decode
- If `llama_decode()` fails, `n_past` remains unchanged

**Invariant 3**: `n_past` corresponds to KV cache size
- `n_past` tracks number of tokens in KV cache
- KV cache contains exactly `n_past` token embeddings

**Invariant 4**: Reset clears both `n_past` and KV cache
- `n_past = 0` implies empty KV cache
- Empty KV cache implies `n_past = 0`

### Appendix C: Error Codes

**Proposed error codes** (for future typed error handling):
```cpp
#define BITNET_ERROR_KV_CACHE_OVERFLOW    -1001
#define BITNET_ERROR_INVALID_POSITION     -1002
#define BITNET_ERROR_CONTEXT_NOT_READY    -1003
```

**Error messages**:
```cpp
// Overflow
"KV cache overflow: n_past=%d + n_tokens=%d > n_ctx=%d"

// Invalid position
"Invalid position: n_past=%d (expected: 0 ≤ n_past ≤ n_ctx=%d)"

// Context not ready
"Context not initialized (call bitnet_cpp_init_context first)"
```

---

## Summary

This specification defines the implementation of manual KV cache position tracking in the C++ wrapper to replace the removed `llama_get_kv_cache_token_count()` API. The design:

1. **Extends Socket 1** with `n_past` field for explicit position tracking
2. **Modifies Socket 3** to validate and update position on each evaluation
3. **Adds reset API** for clearing KV cache and starting new conversations
4. **Maintains backward compatibility** with Socket 0 for stateless use cases
5. **Enables multi-turn support** with 10-100× performance improvement

**Key acceptance criteria**:
- Position correctly initialized, incremented, and validated
- Multi-turn conversations work without redundant recomputation
- Parity tests continue to pass (no correctness regression)
- No performance regression in single-turn case

**Implementation phases**:
1. v0.2: Add `n_past` field and validation logic
2. v0.2: Implement reset API and Rust wrapper updates
3. v0.2: Migrate existing code to use persistent sessions
4. v0.3: (Optional) Deprecate Socket 0 stateless API

**Next steps**: Implement Phase 1 (add `n_past` field and validation) and write unit tests for position tracking.

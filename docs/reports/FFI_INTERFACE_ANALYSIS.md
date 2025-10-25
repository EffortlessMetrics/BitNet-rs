# C++ FFI Interface Analysis for BitNet-rs

## Executive Summary

The BitNet-rs codebase has two distinct FFI layers:

1. **`bitnet-sys` crate** (lower level): Safe Rust wrappers around llama.cpp C API
   - Provides `Session`, `Context`, `Model` structs with unsafe C bindings
   - Includes methods to tokenize, evaluate tokens, and get logits
   - Supports both llama.cpp API AND custom `bitnet_c_shim.cc` functions

2. **`bitnet-crossval` crate** (higher level): Cross-validation comparison framework
   - Currently provides only a mock C wrapper (no integration with bitnet-sys yet)
   - Compares token generation and logits between Rust and C++ implementations

---

## 1. Current API Analysis

### 1.1 Current Token-Based API (Existing)

**Location**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-sys/src/wrapper.rs`

#### Session struct (Lines 330-394)
```rust
pub struct Session {
    pub model: Model,
    pub context: Context,
}

impl Session {
    /// Load a model and create a context with deterministic settings
    pub fn load_deterministic(model_path: &str) -> Result<Self> { ... }
    
    /// Tokenize text
    pub fn tokenize(&self, text: &str) -> Result<Vec<i32>> { ... }
    
    /// Evaluate tokens and return logits for LAST position only
    pub fn eval_and_get_logits(&mut self, tokens: &[i32], n_past: i32) -> Result<Vec<f32>> {
        self.context.eval(tokens, n_past)?;
        self.context.get_logits()  // Only returns logits for last position
    }
}
```

**Key Finding**: `eval_and_get_logits()` returns logits ONLY for the **last token position**. 
It does NOT return per-position logits (which would be Vec<Vec<f32>>).

#### Custom BitNet C Shim API (Lines 478-659)

The wrapper also provides custom BitNet functions that accept token IDs directly:

```rust
/// Tokenize text using custom C shim
pub fn bitnet_tokenize_text(
    model: &BitnetModel,
    text: &str,
    add_bos: bool,
    parse_special: bool,
) -> Result<Vec<i32>> { ... }

/// Evaluate tokens and get last-position logits
pub fn bitnet_eval_tokens(ctx: &BitnetContext, ids: &[i32], vocab_size: usize) -> Result<Vec<f32>> {
    if ids.is_empty() {
        return Err(CppError::LlamaError("Cannot eval empty token sequence".to_string()));
    }
    
    let mut logits = vec![0.0f32; vocab_size];
    
    let result = unsafe {
        crate::bindings::bitnet_eval(
            ctx.as_ptr(),
            ids.as_ptr(),            // Direct token IDs array
            ids.len() as i32,        // Token count
            logits.as_mut_ptr(),
            logits.len() as i32,
        )
    };
    
    if result != 0 {
        return Err(CppError::LlamaError(format!("Eval failed with code: {}", result)));
    }
    
    Ok(logits)  // Returns Vec<f32> for last position only
}

/// Prefill the context with prompt tokens (primes KV cache and sets n_past)
pub fn bitnet_prefill(ctx: &BitnetContext, ids: &[i32]) -> Result<()> { ... }

/// Vocabulary size
pub fn cpp_vocab_size(ctx: &BitnetContext) -> Result<usize> { ... }

/// Greedy decoding
pub fn cpp_decode_greedy(
    model: &BitnetModel,
    ctx: &BitnetContext,
    eos_id: i32,
    eot_id: Option<i32>,
    max_steps: usize,
    out: &mut [i32],  // Output buffer for generated token IDs
) -> Result<usize> { ... }
```

**Key Finding**: Custom BitNet shim ALREADY accepts token IDs directly!
- `bitnet_eval_tokens(&[i32])` - evaluates pre-tokenized sequence
- `bitnet_prefill(&[i32])` - primes KV cache with tokens
- `cpp_decode_greedy()` - generates greedy tokens

### 1.2 Current String-Based Flow

The current tokenization flow in crossval/tests/per_position_logits.rs (Lines 44-46):

```rust
let mut cpp_session = CppSession::load_deterministic(&model_path)?;
let prompt = "The capital of France is";
let tokens = cpp_session.tokenize(prompt)?;  // String → tokenize → Vec<i32>
let cpp_logits_last = cpp_session.eval_and_get_logits(&tokens, 0)?;  // Then eval
```

**Flow**: String prompt → tokenize() → token IDs → eval_and_get_logits()

### 1.3 Return Types Analysis

#### Logits Format
- **Single position**: `Vec<f32>` (vocab_size floats)
- **Multiple positions**: NOT CURRENTLY SUPPORTED in main Session API
  - `Context::get_logits_ith(i: i32)` exists (line 270) but returns `Vec<f32>`
  - `Context::get_all_logits(n_tokens)` exists (line 285) returning `Vec<Vec<f32>>`
  - BUT these are NOT exposed through Session public API

#### Token IDs Format
- Input: `&[i32]` (i32 for llama.cpp compatibility)
- Output from tokenize: `Vec<i32>`
- Output from generation: `Vec<i32>` (greedy) or buffer fill for decode_greedy

---

## 2. Extension Points Assessment

### 2.1 Direct Token ID Interface: Feasibility

**Status: ALREADY EXISTS** (partially implemented)

The custom BitNet shim already provides direct token ID evaluation:

```rust
// Currently available in bitnet-sys:
pub fn bitnet_eval_tokens(ctx: &BitnetContext, ids: &[i32], vocab_size: usize) -> Result<Vec<f32>>
pub fn bitnet_prefill(ctx: &BitnetContext, ids: &[i32]) -> Result<()>
pub fn cpp_decode_greedy(...) -> Result<usize>
```

**What's Missing for per-token logits**:

The wrapper currently returns only **last-position logits**. To get per-token logits:

```rust
// NEEDED: Add this to wrapper.rs
pub fn bitnet_eval_tokens_all_positions(
    ctx: &BitnetContext,
    ids: &[i32],
    vocab_size: usize,
) -> Result<Vec<Vec<f32>>> {
    // Evaluate tokens
    bitnet_eval(ctx, ids, ...)?;
    
    // Get logits for each position (requires new C shim function)
    // Currently bitnet_eval only returns last position
}
```

### 2.2 Current Crossval Integration

**Location**: `/home/steven/code/Rust/BitNet-rs/crossval/src/cpp_bindings.rs`

Current crossval FFI is **DECOUPLED** from bitnet-sys:

```rust
#[cfg(all(feature = "ffi", have_cpp))]
mod imp {
    // Mock C wrapper with ONLY string-based API:
    unsafe extern "C" {
        fn bitnet_cpp_create_model(model_path: *const c_char) -> *mut c_void;
        fn bitnet_cpp_generate(
            model: *mut c_void,
            prompt: *const c_char,      // String prompt, NOT tokens
            max_tokens: c_int,
            tokens_out: *mut u32,
            tokens_count: *mut c_int,
        ) -> c_int;
    }
    
    pub struct CppModel { ... }
    
    impl CppModel {
        pub fn generate(&self, prompt: &str, max_tokens: usize) -> Result<Vec<u32>> {
            // Internally tokenizes the prompt in C++
            // No token ID access
        }
    }
}
```

**Problem**: This doesn't actually connect to bitnet-sys! The C wrapper is mock only
(defined in `/home/steven/code/Rust/BitNet-rs/crossval/src/bitnet_cpp_wrapper.c`).

---

## 3. How Logits Are Currently Compared

### 3.1 Per-Position Comparison (per_position_logits.rs)

**Location**: `/home/steven/code/Rust/BitNet-rs/crossval/tests/per_position_logits.rs`

```rust
use bitnet_inference::eval_logits_once;  // Rust inference
use bitnet_sys::wrapper::{self, Session as CppSession};  // C++ via llama.cpp

// Line 52-53
let cpp_logits_last = cpp_session.eval_and_get_logits(&tokens, 0)?;  // Last position only
let rust_logits_last = eval_logits_once(&model_path, &tokens)?;       // Last position only

// Line 69
let divergence = compare_per_position_logits(&vec![rust_logits_last], &vec![cpp_logits_last]);
```

**Issue**: Currently wraps single-position results in Vec to use the per-position infrastructure!

### 3.2 Comparison Format

**From logits_compare.rs**:

```rust
pub fn compare_per_position_logits(
    rs_logits: &[Vec<f32>],   // Outer = positions, inner = vocab
    cpp_logits: &[Vec<f32>],
) -> LogitsDivergence {
    // Returns: first_divergence_token, per_token_cosine_sim, per_token_l2_dist, max_absolute_diff
}
```

**Current Return Types**:
- `first_divergence_token: Option<usize>` - token position where divergence occurred
- `per_token_cosine_sim: Vec<f32>` - cosine similarity per position
- `per_token_l2_dist: Vec<f32>` - L2 distance per position
- `max_absolute_diff: f32>` - max element-wise difference

---

## 4. Specific Function Signatures for Direct Token Interface

### 4.1 What Exists (bitnet-sys wrapper.rs)

#### llama.cpp API (via Context struct):
```rust
// Line 209
pub fn eval(&mut self, tokens: &[i32], n_past: i32) -> Result<()>

// Line 255-267
pub fn get_logits(&self) -> Result<Vec<f32>>

// Line 270-282
pub fn get_logits_ith(&self, i: i32) -> Result<Vec<f32>>

// Line 285-293
pub fn get_all_logits(&self, n_tokens: usize) -> Result<Vec<Vec<f32>>>
```

#### Custom BitNet C Shim API:
```rust
// Line 548-570
pub fn bitnet_eval_tokens(
    ctx: &BitnetContext,
    ids: &[i32],           // ← DIRECT TOKEN IDS
    vocab_size: usize,
) -> Result<Vec<f32>>      // ← Only last position

// Line 573-586
pub fn bitnet_prefill(
    ctx: &BitnetContext,
    ids: &[i32],           // ← DIRECT TOKEN IDS
) -> Result<()>
```

### 4.2 What's Missing for Multi-Position Logits

**Current limitation**: Both Session and BitNet shim APIs return only **last-position logits**.

To support per-token comparison, we would need:

```rust
// Option A: Extend Session API
impl Session {
    pub fn eval_and_get_all_logits(
        &mut self,
        tokens: &[i32],
        n_past: i32,
    ) -> Result<Vec<Vec<f32>>> {
        self.context.eval(tokens, n_past)?;
        self.context.get_all_logits(tokens.len())  // Already exists!
    }
}

// Option B: Add to BitNet C Shim
pub fn bitnet_eval_tokens_all_positions(
    ctx: &BitnetContext,
    ids: &[i32],
    vocab_size: usize,
) -> Result<Vec<Vec<f32>>> {
    // Call bitnet_eval to fill context logits
    // Then extract all positions
    // CAVEAT: bitnet_eval C function must fill context for all positions
}
```

---

## 5. Architecture Diagram

```
Current Architecture:

┌─ crossval (mock C wrapper) ─────────────────┐
│  bitnet_cpp_wrapper.c (dummy generation)    │
│  String prompt → tokenize → generate        │
│  CppModel::generate(&str) → Vec<u32>        │
└─────────────────────────────────────────────┘
                        (DECOUPLED)
                            ↓
┌─ bitnet-sys (real C++ bindings) ───────────────────────┐
│                                                        │
│  llama.cpp API:                                       │
│  ├─ Session::load_deterministic(path)                │
│  ├─ tokenize(text: &str) → Vec<i32>                  │
│  └─ eval_and_get_logits(tokens: &[i32]) → Vec<f32>  │
│                                                        │
│  Custom BitNet C Shim:                                │
│  ├─ bitnet_tokenize_text(text: &str)                 │
│  ├─ bitnet_eval_tokens(ids: &[i32]) → Vec<f32>      │
│  ├─ bitnet_prefill(ids: &[i32])                      │
│  └─ cpp_decode_greedy(ids: &[i32]) → usize          │
│                                                        │
│  Limitation: All return ONLY last-position logits!    │
└────────────────────────────────────────────────────────┘
            ↑
    Uses via tests
    (/per_position_logits.rs)
            ↑
┌─ bitnet-inference (Rust inference) ────┐
│ eval_logits_once(model: &str,         │
│                  tokens: &[i32])      │
│   → Vec<f32>  (last position)          │
└────────────────────────────────────────┘
```

---

## 6. Effort Assessment

### 6.1 Adding Direct Token Interface to crossval

**Current status**: crossval is decoupled from bitnet-sys; uses mock C wrapper

**To integrate with bitnet-sys**:
- **Effort: MEDIUM (2-3 hours)**
  1. Replace `CppModel` in crossval/cpp_bindings.rs to use `bitnet_sys::wrapper::Session`
  2. Add wrapper methods to expose token-based API
  3. Update tests to use new API

**Code changes** (rough estimate):
```rust
// In crossval/src/cpp_bindings.rs
// BEFORE:
pub struct CppModel {
    handle: *mut c_void,  // Mock C wrapper
}

// AFTER:
pub struct CppModel {
    session: bitnet_sys::wrapper::Session,  // Real C++ binding
}

impl CppModel {
    pub fn load<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let session = bitnet_sys::wrapper::Session::load_deterministic(...)?;
        Ok(CppModel { session })
    }
    
    pub fn eval_tokens(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        // Convert u32 to i32 for bitnet_sys
        let ids: Vec<i32> = tokens.iter().map(|t| *t as i32).collect();
        self.session.eval_and_get_logits(&ids, 0)
    }
}
```

### 6.2 Adding Per-Token Logits Support

**Current limitation**: wrapper returns only last-position logits

**To support multi-position logits**:
- **Effort: LOW (1-2 hours)**
  1. Session already exposes `Context::get_all_logits()` - just need to surface it
  2. Add Session method wrapper
  3. Update tests

**Code changes**:
```rust
// In bitnet-sys/src/wrapper.rs
impl Session {
    pub fn eval_and_get_all_logits(&mut self, tokens: &[i32], n_past: i32) -> Result<Vec<Vec<f32>>> {
        self.context.eval(tokens, n_past)?;
        self.context.get_all_logits(tokens.len())  // ← Already implemented (line 285)
    }
}
```

### 6.3 Adding Per-Token Logits to Custom BitNet C Shim

**Current limitation**: `bitnet_eval_tokens()` returns only last position

**To support multi-position**:
- **Effort: MEDIUM-HIGH (3-5 hours)**
  1. Modify C++ shim function to return all positions (or restructure API)
  2. Add Rust wrapper around new C function
  3. Test correctness

**Unknowns**:
- Does the C++ shim function `bitnet_eval()` fill context for all positions?
- What's the C function signature?
- Does it require KV cache pre-filling via `bitnet_prefill()`?

---

## 7. Recommended Path Forward

### Phase 1: Unify crossval with bitnet-sys (MEDIUM EFFORT)
1. Replace mock C wrapper with real bitnet-sys::wrapper::Session
2. Expose token-based API in crossval::CppModel
3. Update per_position_logits.rs tests to use real C++ backend

### Phase 2: Add Per-Token Logits (LOW EFFORT)
1. Surface Session::eval_and_get_all_logits() for llama.cpp API
2. Update per_position_logits.rs to collect all positions
3. Validate multi-token divergence detection works

### Phase 3: Custom BitNet Shim Per-Token (MEDIUM-HIGH EFFORT)
1. Investigate C++ shim signature and capabilities
2. If possible, extend to return multi-position logits
3. Add Rust wrapper
4. Test performance impact

---

## 8. Key Files for Reference

### FFI Definition:
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-sys/src/wrapper.rs` - Main wrapper API
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-sys/src/lib.rs` - Public exports

### Current Crossval:
- `/home/steven/code/Rust/BitNet-rs/crossval/src/cpp_bindings.rs` - Mock C++ wrapper
- `/home/steven/code/Rust/BitNet-rs/crossval/src/bitnet_cpp_wrapper.c` - C stub
- `/home/steven/code/Rust/BitNet-rs/crossval/tests/per_position_logits.rs` - Logits comparison tests

### Comparison Logic:
- `/home/steven/code/Rust/BitNet-rs/crossval/src/logits_compare.rs` - Per-position metrics
- `/home/steven/code/Rust/BitNet-rs/crossval/src/comparison.rs` - High-level comparison

### Inference:
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/parity.rs` - Rust eval_logits_once()

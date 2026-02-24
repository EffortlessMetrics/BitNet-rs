# Dispatcher Architecture Summary: Backend Routing for crossval-per-token

## Visual Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     crossval-per-token Command                      │
│                       (xtask/src/main.rs)                           │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 ├─ CLI Args parsing (lines 437-483)
                 │  └─ cpp_backend: Option<CppBackend>
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│           Backend Detection & Preflight (lines 2992-3004)           │
│                                                                     │
│  backend = cpp_backend.unwrap_or_else(                             │
│    || CppBackend::from_model_path(model_path)                      │
│  )                                                                  │
│                                                                     │
│  preflight_backend_libs(backend, verbose)?                         │
│  ✅ Auto-detection from path (bitnet/llama)                        │
│  ✅ Library availability check                                     │
└────────────────┬────────────────────────────────────────────────────┘
                 │
         ┌───────┼───────┐
         │       │       │
    [BitNet]  [Llama]  [Undetected]
         │       │       │
         └───────┼───────┘
                 │
                 ▼
    ┌────────────────────────────────────┐
    │  Prompt Formatting & Rust Tokenize │
    │  (lines 3006-3041)                 │
    │  ✅ Backend-agnostic               │
    └────────────────────────────────────┘
                 │
                 ▼
    ┌────────────────────────────────────┐
    │    ❌ [ROUTING POINT #1] ❌         │
    │  C++ Tokenization (line ~3074)     │
    │                                    │
    │  CURRENT: Hardcoded to llama.cpp   │
    │  NEEDS: Backend match statement    │
    └────────────┬───────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
    ┌──────────┐   ┌────────────────┐
    │ BitNet   │   │ Llama.cpp      │
    │(TODO)    │   │ (implemented)  │
    │          │   │                │
    │unimpl!() │   │load_deterministic()
    │          │   │session.tokenize()
    └──────────┘   └────────────────┘
        │                 │
        └────────┬────────┘
                 │
                 ▼
    ┌────────────────────────────────────┐
    │   Token Parity Validation          │
    │   (lines 3084-3092)                │
    │   ✅ Backend-agnostic              │
    └────────────────────────────────────┘
                 │
                 ▼
    ┌────────────────────────────────────┐
    │   Rust Logits Evaluation           │
    │   (lines 3095-3102)                │
    │   ✅ Pure Rust (all formats)       │
    └────────────────────────────────────┘
                 │
                 ▼
    ┌────────────────────────────────────┐
    │    ❌ [ROUTING POINT #2] ❌         │
    │  C++ Logits Evaluation (line ~3108)│
    │                                    │
    │  CURRENT: Hardcoded to llama.cpp   │
    │  NEEDS: Backend match statement    │
    └────────────┬───────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
    ┌──────────┐   ┌────────────────┐
    │ BitNet   │   │ Llama.cpp      │
    │(TODO)    │   │ (implemented)  │
    │          │   │                │
    │unimpl!() │   │session.context.eval()
    │          │   │context.get_all_logits()
    └──────────┘   └────────────────┘
        │                 │
        └────────┬────────┘
                 │
                 ▼
    ┌────────────────────────────────────┐
    │   Logits Comparison                │
    │   (lines 3120-3186)                │
    │   ✅ Backend-agnostic              │
    │   ✅ Generates diagnostics         │
    └────────────────────────────────────┘
```

---

## Backend Infrastructure (Already Complete)

### 1. Backend Enum
**File**: `xtask/src/crossval/backend.rs`

```rust
pub enum CppBackend {
    BitNet,   // microsoft-bitnet-b1.58-2B-4T-gguf
    Llama,    // llama-3, llama-2, etc.
}
```

**Methods Available**:
- `from_model_path(path: &Path) -> Self` - Auto-detect from path
- `name() -> &'static str` - "bitnet.cpp" or "llama.cpp"
- `required_libs() -> &[&'static str]` - Library requirements
- `setup_command() -> &'static str` - Setup instructions

---

### 2. Preflight Validation
**File**: `xtask/src/crossval/preflight.rs`

```rust
pub fn preflight_backend_libs(backend: CppBackend, verbose: bool) -> Result<()>
```

**Checks**:
- Compile-time library detection (CROSSVAL_HAS_BITNET, CROSSVAL_HAS_LLAMA)
- Returns error with setup instructions if missing
- Provides helpful messages for user

---

### 3. C++ FFI Wrapper (llama.cpp only)
**Module**: `bitnet_sys::wrapper`

```rust
bitnet_sys::wrapper::init_backend()
bitnet_sys::wrapper::Session::load_deterministic(model_path)
session.tokenize(prompt) -> Result<Vec<u32>>
session.context.eval(tokens, pos)
session.context.get_all_logits(len) -> Result<Vec<Vec<f32>>>
bitnet_sys::wrapper::free_backend()
```

---

## Two Routing Points

### Routing Point #1: C++ Tokenization (Line ~3074)

**Current Code**:
```rust
let mut cpp_session = bitnet_sys::wrapper::Session::load_deterministic(model_path_str)?;
let cpp_tokens = cpp_session.tokenize(&formatted_prompt)?;
```

**Required Change**:
```rust
let cpp_tokens = match backend {
    CppBackend::BitNet => unimplemented!("BitNet.cpp tokenization"),
    CppBackend::Llama => {
        // Move current code here
    }
};
```

**Call Site Function Signature**:
- Input: `&str` (model_path), `&str` (formatted_prompt)
- Output: `Vec<u32>` (token IDs)
- Error: `Result<Vec<u32>>`

---

### Routing Point #2: C++ Logits Evaluation (Line ~3108)

**Current Code**:
```rust
cpp_session.context.eval(&cpp_tokens, 0)?;
let cpp_logits = cpp_session.context.get_all_logits(cpp_tokens.len())?;
```

**Required Change**:
```rust
let cpp_logits = match backend {
    CppBackend::BitNet => unimplemented!("BitNet.cpp logits evaluation"),
    CppBackend::Llama => {
        // Move current code here (needs to re-initialize session)
    }
};
```

**Call Site Function Signature**:
- Input: `&[u32]` (tokens), `model_path_str: &str`
- Output: `Vec<Vec<f32>>` (logits per position)
- Error: `Result<Vec<Vec<f32>>>`

---

## Component Status Matrix

| Component | File | Status | Dependencies |
|-----------|------|--------|--------------|
| CLI Args | main.rs:437-483 | ✅ Ready | None |
| CppBackend enum | backend.rs:10-28 | ✅ Ready | clap |
| Auto-detection | backend.rs:50-61 | ✅ Ready | CppBackend enum |
| Preflight check | preflight.rs:34-79 | ✅ Ready | CppBackend enum |
| Preflight integration | main.rs:3004 | ✅ Ready | preflight.rs |
| Rust tokenization | main.rs:3032-3041 | ✅ Ready | bitnet_tokenizers |
| FFI availability check | main.rs:3043-3048 | ✅ Ready | bitnet_sys |
| **C++ tokenization dispatch** | main.rs:~3074 | ❌ NOT IMPLEMENTED | Backend match |
| Token parity validation | main.rs:3084-3092 | ✅ Ready | bitnet_crossval |
| Rust logits evaluation | main.rs:3097 | ✅ Ready | bitnet_inference |
| **C++ logits dispatch** | main.rs:~3108 | ❌ NOT IMPLEMENTED | Backend match |
| Logits comparison | main.rs:3122 | ✅ Ready | bitnet_crossval |
| Output formatting | main.rs:3125-3186 | ✅ Ready | serde_json |

---

## Implementation Checklist

### Phase 1: Insert Backend Match Statements

- [ ] **Task 1.1**: Add match statement for C++ tokenization (line ~3074)
  - Move existing llama.cpp code to `CppBackend::Llama` arm
  - Add `CppBackend::BitNet => unimplemented!()` placeholder
  - Test with llama model

- [ ] **Task 1.2**: Add match statement for C++ logits evaluation (line ~3108)
  - Move existing llama.cpp code to `CppBackend::Llama` arm
  - Note: Need to re-initialize session in this arm
  - Add `CppBackend::BitNet => unimplemented!()` placeholder
  - Test with llama model

- [ ] **Task 1.3**: Add diagnostic output
  - In verbose mode, print which backend path was selected
  - Print which C++ functions were called

### Phase 2: Implement BitNet.cpp FFI Wrappers

- [ ] **Task 2.1**: Create tokenize_bitnet() wrapper
  - Signature: `fn(model_path: &str, prompt: &str) -> Result<Vec<u32>>`
  - Wire into `CppBackend::BitNet` arm of tokenization match

- [ ] **Task 2.2**: Create evaluate_logits_bitnet() wrapper
  - Signature: `fn(model_path: &str, tokens: &[u32]) -> Result<Vec<Vec<f32>>>`
  - Wire into `CppBackend::BitNet` arm of logits match

- [ ] **Task 2.3**: Test with BitNet models

### Phase 3: Optimization (Optional)

- [ ] Add timing metrics for each backend
- [ ] Add verbose tracing of FFI calls
- [ ] Refactor to avoid session re-initialization

---

## Code Location Reference

```
/home/steven/code/Rust/BitNet-rs/
├── xtask/src/
│   ├── main.rs                    (crossval_per_token_cmd: lines 2974-3189)
│   ├── crossval/
│   │   ├── backend.rs             (CppBackend enum)
│   │   └── preflight.rs           (preflight_backend_libs)
│   └── lib.rs                     (module exports)
│
├── crossval/src/
│   ├── lib.rs                     (cross-validation framework)
│   ├── token_parity.rs            (validate_token_parity)
│   └── logits_compare.rs          (compare_per_position_logits)
│
├── crates/bitnet-inference/src/
│   └── parity.rs                  (eval_logits_all_positions)
│
└── crates/bitnet-kernels/src/
    └── (FFI wrapper to C++)
```

---

## Key Insight: Two Independent Paths

The routing decision is **made once** at the beginning (line 2993):

```rust
let backend = cpp_backend.unwrap_or_else(|| CppBackend::from_model_path(model_path));
```

After that:
- **Rust paths** are backend-agnostic (same code for all models)
- **C++ paths** need backend dispatch (different FFI calls)
- **Comparison paths** are backend-agnostic (just compares outputs)

This means:
- You only need to modify the C++ call sites
- You don't need to refactor the comparison logic
- BitNet models will work with the same Rust inference engine

---

## Minimal Implementation Strategy

To get this working with **minimal risk**:

1. Extract the current llama.cpp code into match arms
2. Use `unimplemented!()` for BitNet.cpp (clear placeholder)
3. Test with existing llama models (no regression)
4. Document where FFI wrappers are needed for Phase 2

This approach:
- Preserves all existing functionality
- Makes the infrastructure explicit
- Guides Phase 2 implementation
- Creates zero new dependencies


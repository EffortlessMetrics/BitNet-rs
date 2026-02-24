# Backend Routing Analysis: crossval-per-token Command

## Executive Summary

The `crossval-per-token` command in xtask already has **complete backend infrastructure** in place for dispatching between bitnet.cpp and llama.cpp. However, the **C++ tokenization and logits evaluation paths** are currently hardcoded to use only llama.cpp via `bitnet_sys::wrapper`. This document identifies exactly where backend-aware routing needs to be inserted.

---

## Architecture Overview

### CLI Entry Point
**File**: `/home/steven/code/Rust/BitNet-rs/xtask/src/main.rs` (lines 437-483)

```rust
#[cfg(feature = "inference")]
#[command(name = "crossval-per-token")]
CrossvalPerToken {
    model: PathBuf,                          // Model GGUF path
    tokenizer: PathBuf,                      // Tokenizer JSON path
    prompt: String,                          // Input prompt
    max_tokens: usize,                       // Generation budget (reserved)
    cos_tol: f32,                            // Cosine similarity threshold (0.999)
    format: String,                          // Output format (text/json)
    prompt_template: PromptTemplateArg,      // Template type
    system_prompt: Option<String>,           // System prompt (chat templates)
    cpp_backend: Option<CppBackend>,         // Backend selection ✅ AVAILABLE
    verbose: bool,                           // Diagnostic output
}
```

### Backend Enum
**File**: `/home/steven/code/Rust/BitNet-rs/xtask/src/crossval/backend.rs` (lines 10-106)

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum CppBackend {
    /// BitNet.cpp - for BitNet models (Microsoft)
    #[value(name = "bitnet")]
    BitNet,
    
    /// Llama.cpp - for LLaMA/compatible models
    #[value(name = "llama")]
    Llama,
}
```

#### Key Methods:
- `from_model_path(&Path) -> Self` - Auto-detect backend from model path
- `name() -> &'static str` - "bitnet.cpp" or "llama.cpp"
- `required_libs() -> &[&'static str]` - Lib requirements for preflight
- `setup_command() -> &'static str` - Setup instructions for user

### Handler Function
**File**: `/home/steven/code/Rust/BitNet-rs/xtask/src/main.rs` (lines 2974-3189)

```rust
#[cfg(feature = "inference")]
fn crossval_per_token_cmd(
    model_path: &Path,
    tokenizer_path: &Path,
    prompt: &str,
    _max_tokens: usize,
    cos_tol: f32,
    format: &str,
    prompt_template: PromptTemplateArg,
    _system_prompt: Option<&str>,
    cpp_backend: Option<CppBackend>,        // ← Backend passed in
    verbose: bool,
) -> Result<()>
```

---

## Current Execution Flow

### 1. Backend Selection (Already Implemented ✅)
**Lines 2992-3004**

```rust
// Auto-detect if not explicit
let backend = cpp_backend.unwrap_or_else(|| CppBackend::from_model_path(model_path));

if verbose {
    println!("Selected backend: {} ({})", backend.name(), ...);
}

// Preflight validation - checks if libraries are available
crate::crossval::preflight_backend_libs(backend, verbose)?;
```

**Status**: ✅ Working. Detects backend and validates availability.

### 2. Rust Tokenization (Already Working ✅)
**Lines 3032-3041**

```rust
println!("📝 Tokenizing prompt (Rust)...");
let tokenizer = bitnet_tokenizers::loader::load_tokenizer(tokenizer_path)?;
let tokens = tokenizer.encode(&formatted_prompt, add_bos, parse_special)?;
let token_ids: Vec<i32> = tokens.iter().map(|&id| id as i32).collect();
```

**Status**: ✅ Uses universal Rust tokenizer (works for all models).

### 3. C++ Availability Check (Already Implemented ✅)
**Lines 3043-3048**

```rust
if !bitnet_sys::is_available() {
    anyhow::bail!("C++ FFI not available...");
}
```

**Status**: ✅ Checks if FFI is compiled.

### 4. C++ Tokenization (HARDCODED to llama.cpp ❌)
**Lines 3050-3080** — **THIS IS WHERE ROUTING IS NEEDED**

```rust
println!("📝 Tokenizing prompt (C++)...");

// TODO(Phase D): Backend dispatch - replace with backend-aware tokenization
// match backend {
//     CppBackend::BitNet => { /* ... */ }
//     CppBackend::Llama => { /* ... */ }
// }

// CURRENT CODE (hardcoded llama.cpp):
let model_path_str = model_path.to_str()?;
bitnet_sys::wrapper::init_backend();
let _guard = scopeguard::guard((), |_| bitnet_sys::wrapper::free_backend());

let mut cpp_session = bitnet_sys::wrapper::Session::load_deterministic(model_path_str)?;
let cpp_tokens = cpp_session.tokenize(&formatted_prompt)?;
```

**Status**: ❌ Hardcoded to llama.cpp. Needs backend dispatch.

### 5. Token Parity Validation (Backend-Agnostic ✅)
**Lines 3081-3092**

```rust
println!("🔒 Validating token parity...");
let rust_tokens_u32: Vec<u32> = token_ids.iter().map(|&id| id as u32).collect();
bitnet_crossval::token_parity::validate_token_parity(
    &rust_tokens_u32, &cpp_tokens, prompt
)?;
println!("✓ Token sequences match");
```

**Status**: ✅ Works with any backend (just compares token sequences).

### 6. Rust Logits Evaluation (Already Working ✅)
**Lines 3095-3102**

```rust
println!("🦀 Evaluating Rust logits for all positions...");
let rust_logits = eval_logits_all_positions(model_path_str, &token_ids)?;
```

**Status**: ✅ Uses bitnet_inference::parity (Rust-native for all formats).

### 7. C++ Logits Evaluation (HARDCODED to llama.cpp ❌)
**Lines 3104-3117** — **THIS IS WHERE ROUTING IS NEEDED**

```rust
println!("🔧 Evaluating C++ logits for all positions...");

// CURRENT CODE (hardcoded llama.cpp Session):
cpp_session.context.eval(&cpp_tokens, 0)?;
let cpp_logits = cpp_session.context.get_all_logits(cpp_tokens.len())?;
```

**Status**: ❌ Hardcoded to llama.cpp. Needs backend dispatch.

### 8. Logits Comparison (Backend-Agnostic ✅)
**Lines 3120-3186**

```rust
println!("📊 Comparing logits per position...");
let divergence = compare_per_position_logits(&rust_logits, &cpp_logits);
// Output formatting and diagnostics...
```

**Status**: ✅ Works with any backend (just compares logits vectors).

---

## Identified Call Sites for Backend Routing

### Call Site #1: C++ Tokenization (Line 3074)
**Location**: `/home/steven/code/Rust/BitNet-rs/xtask/src/main.rs:3074`

```rust
// CURRENT (llama.cpp only):
let mut cpp_session = bitnet_sys::wrapper::Session::load_deterministic(model_path_str)?;
let cpp_tokens = cpp_session.tokenize(&formatted_prompt)?;
```

**Needs**: Backend-aware tokenization dispatch

**Signature**:
- Input: `&Path`, `&str` (model_path, formatted_prompt)
- Output: `Vec<u32>` (token IDs)
- Error: `anyhow::Result`

### Call Site #2: C++ Logits Evaluation (Lines 3108-3111)
**Location**: `/home/steven/code/Rust/BitNet-rs/xtask/src/main.rs:3108-3111`

```rust
// CURRENT (llama.cpp only):
cpp_session.context.eval(&cpp_tokens, 0)?;
let cpp_logits = cpp_session.context.get_all_logits(cpp_tokens.len())?;
```

**Needs**: Backend-aware logits evaluation dispatch

**Signature**:
- Input: `&[u32]` (tokens), model context
- Output: `Vec<Vec<f32>>` (logits per position)
- Error: `anyhow::Result`

---

## Current C++ Wrapper Integration (llama.cpp)

### bitnet_sys::wrapper API
**Module**: `bitnet_sys::wrapper` (FFI bridge to llama.cpp)

**Current usage pattern**:
```rust
// 1. Initialize backend
bitnet_sys::wrapper::init_backend();
let _guard = scopeguard::guard((), |_| bitnet_sys::wrapper::free_backend());

// 2. Load model session (deterministic)
let mut session = bitnet_sys::wrapper::Session::load_deterministic(model_path)?;

// 3. Tokenization
let tokens = session.tokenize(&prompt)?;  // Returns Vec<u32>

// 4. Evaluation
session.context.eval(&tokens, 0)?;        // Evaluate with logits_all=true
let logits = session.context.get_all_logits(tokens.len())?;  // Vec<Vec<f32>>
```

**Error Type**: Uses `anyhow::Result<T>`

**Lifetime**: Scoped with scopeguard - auto cleanup on drop

---

## Backend Match Statement Template

### Where to Insert (Line ~3050)

```rust
// Step 3: C++ tokenization
println!("📝 Tokenizing prompt (C++)...");

let model_path_str = model_path.to_str().ok_or_else(|| anyhow::anyhow!("Invalid model path"))?;

// Backend-aware tokenization dispatch
let cpp_tokens = match backend {
    CppBackend::BitNet => {
        // TODO: Implement bitnet.cpp FFI wrappers
        // Expected signature: tokenize_bitnet(&model_path_str, &formatted_prompt)?
        unimplemented!("BitNet.cpp tokenization - requires FFI wrapper layer")
    }
    CppBackend::Llama => {
        // Current implementation - llama.cpp
        bitnet_sys::wrapper::init_backend();
        let _guard = scopeguard::guard((), |_| bitnet_sys::wrapper::free_backend());
        let mut cpp_session = bitnet_sys::wrapper::Session::load_deterministic(model_path_str)?;
        cpp_session.tokenize(&formatted_prompt)?
    }
};

println!("Tokens: {} (C++)", cpp_tokens.len());
```

### For Logits Evaluation (Line ~3104)

```rust
println!("🔧 Evaluating C++ logits for all positions...");

// Backend-aware logits dispatch
let cpp_logits = match backend {
    CppBackend::BitNet => {
        // TODO: Implement bitnet.cpp FFI for logits evaluation
        // Expected to evaluate tokens and return Vec<Vec<f32>>
        unimplemented!("BitNet.cpp logits evaluation - requires FFI wrapper layer")
    }
    CppBackend::Llama => {
        // Current implementation - llama.cpp
        bitnet_sys::wrapper::init_backend();
        let _guard = scopeguard::guard((), |_| bitnet_sys::wrapper::free_backend());
        let mut cpp_session = bitnet_sys::wrapper::Session::load_deterministic(model_path_str)?;
        cpp_session.context.eval(&cpp_tokens, 0)?;
        cpp_session.context.get_all_logits(cpp_tokens.len())?
    }
};

println!("✓ C++: {} positions, vocab_size={}", ...);
```

---

## Crossval Module Exports

**File**: `/home/steven/code/Rust/BitNet-rs/crossval/src/lib.rs`

### Available Modules
- `cpp_bindings` - C++ interface abstraction
- `comparison` - Logits/token comparison utilities
- `logits_compare` - Per-position logits comparison
- `token_parity` - Token sequence validation
- `validation` - Multi-stage validation suite
- `fixtures` - Test fixture generation
- `score` - Scoring and metrics

### Key Functions Used in crossval-per-token
```rust
use bitnet_crossval::logits_compare::compare_per_position_logits;
use bitnet_crossval::token_parity::validate_token_parity;
use bitnet_inference::parity::eval_logits_all_positions;
```

---

## Error Handling Pattern

### Current Pattern
```rust
// Initialize with scopeguard for cleanup
bitnet_sys::wrapper::init_backend();
let _guard = scopeguard::guard((), |_| bitnet_sys::wrapper::free_backend());

// Operations use anyhow::Result<T>
let result: Vec<u32> = some_operation()?;

// Errors propagate with context
.ok_or_else(|| anyhow::anyhow!("descriptive error"))?
```

### For Backend Dispatch
- Use `match backend { ... }` for selection
- Each arm should handle its own initialization/cleanup
- Use `unimplemented!()` for future BitNet.cpp paths
- Errors propagate via `?` operator (anyhow::Result)

---

## Preflight Validation

**File**: `/home/steven/code/Rust/BitNet-rs/xtask/src/crossval/preflight.rs`

### What It Does
- Checks compile-time detection from `build.rs` environment variables
- `CROSSVAL_HAS_BITNET` - "true" if libbitnet* found
- `CROSSVAL_HAS_LLAMA` - "true" if libllama*/libggml* found
- Returns error with setup instructions if missing

### Usage in crossval_per_token_cmd
```rust
crate::crossval::preflight_backend_libs(backend, verbose)?;
```

**Status**: ✅ Already integrated at line 3004

---

## Implementation Roadmap

### Phase 1: Insert Backend Match (IMMEDIATE)
1. Replace hardcoded llama.cpp calls with `match backend { ... }`
2. Move existing llama.cpp code to `CppBackend::Llama` arm
3. Add `unimplemented!()` for `CppBackend::BitNet`
4. Test with llama models to ensure no regression

### Phase 2: Implement BitNet.cpp FFI (FUTURE)
1. Create FFI wrappers for bitnet.cpp tokenization
2. Create FFI wrappers for bitnet.cpp logits evaluation
3. Implement both arms of backend match
4. Test with BitNet models

### Phase 3: Add Diagnostic Output (OPTIONAL)
1. In verbose mode, print which backend functions were called
2. Add timing information for each backend path
3. Log FFI call arguments/returns for debugging

---

## Summary Table

| Component | Location | Status | Notes |
|-----------|----------|--------|-------|
| CLI Args | main.rs:437-483 | ✅ Ready | CppBackend enum available |
| Backend Detection | backend.rs:50-61 | ✅ Ready | Auto-detects from path |
| Backend Metadata | backend.rs:72-106 | ✅ Ready | name(), required_libs(), setup_command() |
| Preflight Check | main.rs:3004 | ✅ Ready | Validates library availability |
| Rust Tokenization | main.rs:3032-3041 | ✅ Ready | Universal tokenizer |
| C++ FFI Check | main.rs:3043-3048 | ✅ Ready | Checks if compiled |
| **C++ Tokenization** | main.rs:3074-3077 | ❌ HARDCODED | Needs backend dispatch |
| Token Parity Check | main.rs:3084-3092 | ✅ Ready | Backend-agnostic |
| Rust Logits | main.rs:3097 | ✅ Ready | Pure Rust path |
| **C++ Logits** | main.rs:3108-3111 | ❌ HARDCODED | Needs backend dispatch |
| Comparison | main.rs:3122 | ✅ Ready | Backend-agnostic |
| Output | main.rs:3125-3186 | ✅ Ready | Text/JSON formatting |

---

## Function Signatures to Match

### For BitNet.cpp Tokenization (Future)
```rust
// Expected signature to implement:
fn tokenize_bitnet(model_path: &str, prompt: &str) -> Result<Vec<u32>>
```

### For BitNet.cpp Logits (Future)
```rust
// Expected signature to implement:
fn evaluate_logits_bitnet(model_path: &str, tokens: &[u32]) -> Result<Vec<Vec<f32>>>
```

### Current llama.cpp Signatures (Via bitnet_sys::wrapper)
```rust
// Tokenization:
Session::load_deterministic(model_path: &str) -> Result<Session>
session.tokenize(prompt: &str) -> Result<Vec<u32>>

// Logits:
session.context.eval(tokens: &[u32], pos: usize) -> Result<()>
session.context.get_all_logits(len: usize) -> Result<Vec<Vec<f32>>>
```

---

## Key Insights

1. **Infrastructure is 90% Complete**: Backend enum, detection, preflight validation all exist
2. **Only Two Call Sites Need Routing**: C++ tokenization and logits evaluation
3. **Llama.cpp Path Already Working**: Just needs to be moved into `CppBackend::Llama` arm
4. **BitNet.cpp is Unimplemented**: Expected to be wired in Phase 2
5. **Error Handling Pattern**: Use anyhow::Result, cleanup with scopeguard
6. **No Shared Session**: Each backend path manages its own session lifecycle
7. **Token Sequences Must Match**: Token parity check ensures consistent input to logits evaluation


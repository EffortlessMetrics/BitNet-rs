# `crossval-per-token` Command Implementation Analysis

## Command Overview

**Location**: `/home/steven/code/Rust/BitNet-rs/xtask/src/main.rs` (lines 405-3041)

The `crossval-per-token` command performs deterministic per-token logits comparison between Rust and C++ implementations to identify divergence points during inference.

### Command Definition
**File**: `/home/steven/code/Rust/BitNet-rs/xtask/src/main.rs`
**Lines**: 405-430 (command struct), 2901-3041 (implementation)

```rust
#[cfg(feature = "inference")]
#[command(name = "crossval-per-token")]
CrossvalPerToken {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    tokenizer: PathBuf,
    #[arg(long)]
    prompt: String,
    #[arg(long, default_value_t = 4)]
    max_tokens: usize,
    #[arg(long, default_value_t = 0.999)]
    cos_tol: f32,
    #[arg(long, default_value = "text")]
    format: String,
},
```

---

## Current Flow Analysis

### 1. TOKENIZATION (Lines 2918-2926)

**Current Approach**: Uses Rust tokenizer only, no template support

```rust
// Line 2920-2922
let tokenizer = bitnet_tokenizers::loader::load_tokenizer(tokenizer_path)?;
let tokens = tokenizer.encode(prompt, false, false)?;  // add_bos=false, add_special=false
let token_ids: Vec<i32> = tokens.iter().map(|&id| id as u32).collect();
```

**Key Issues**:
- **No BOS token**: `add_bos=false` means no beginning-of-sequence token prepended
- **No special tokens**: `add_special=false` skips special token handling
- **No template**: Raw tokenization without prompt template formatting (unlike CLI)
- **No parity with CLI tokenization**: CLI uses auto-detected templates that can format prompts

**Comparison to CLI** (`crates/bitnet-cli/src/commands/inference.rs`):
- CLI uses prompt templates (auto-detected or explicit)
- CLI may prepend system prompts
- CLI may apply special formatting for Q&A or chat modes

---

### 2. RUST LOGITS EVALUATION (Lines 2929-2938)

**Function**: `eval_logits_all_positions()` from `bitnet_inference::parity`
**File**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/parity.rs` (lines 157-223)

**What it does**:
1. Loads GGUF model with `load_gguf_full()` (pure Rust, fail-closed)
2. Converts QK256 tensors to raw byte tensors with key remapping
3. Runs full forward pass for all token positions
4. Returns `Vec<Vec<f32>>` with shape: [n_positions][vocab_size]

```rust
// Lines 2931-2933
let model_path_str = model_path.to_str().ok_or_else(...)?;
let rust_logits = eval_logits_all_positions(model_path_str, &token_ids)?;
```

**Output**: Logits for each input position (not generated tokens)

---

### 3. C++ LOGITS EVALUATION (Lines 2940-2970)

**FFI Components**:
- **Wrapper**: `bitnet_sys::wrapper::Session` (safe wrapper around llama.cpp)
- **Location**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-sys/src/wrapper.rs`

**Key FFI Calls**:

| Function | Purpose | Lines |
|----------|---------|-------|
| `bitnet_sys::is_available()` | Check if C++ available | 2944 |
| `bitnet_sys::wrapper::init_backend()` | Initialize llama backend | 2951 |
| `Session::load_deterministic()` | Load model, 1 thread | 2954 |
| `cpp_session.tokenize(prompt)` | Tokenize with C++ llama.cpp | 2957 |
| `context.eval(&cpp_tokens, 0)` | Evaluate all tokens | 2960 |
| `context.get_all_logits(n_tokens)` | Get per-position logits | 2963 |

**Session Details** (lines 329-394 in wrapper.rs):
```rust
pub struct Session {
    pub model: Model,
    pub context: Context,
}

impl Session {
    pub fn load_deterministic(model_path: &str) -> Result<Self> {
        let model = Model::load(model_path)?;
        let context = Context::new(&model, 2048, 512, 1)?;  // 2048 context, 512 batch, 1 thread
        Ok(Session { model, context })
    }
}
```

**Deterministic Settings**:
- `n_ctx = 2048` (context window)
- `n_batch = 512` (batch size)
- `n_threads = 1` (single-threaded for reproducibility)
- `logits_all = true` (enable logits for all positions)

**C++ Tokenization** (wrapper.rs lines 144-186):
```rust
pub fn tokenize(&self, text: &str, add_special: bool) -> Result<Vec<i32>> {
    // Two-pass tokenization via llama_tokenize()
    // add_special=true (C++ side uses special token handling)
}
```

**C++ Logits Retrieval** (wrapper.rs lines 285-293):
```rust
pub fn get_all_logits(&self, n_tokens: usize) -> Result<Vec<Vec<f32>>> {
    // Calls get_logits_ith(i) for each position
    // Returns all logits via llama_get_logits_ith()
}
```

**Cleanup**: Uses `scopeguard::guard()` to ensure `free_backend()` is called

---

### 4. COMPARISON LOGIC (Lines 2972-3041)

**Function**: `compare_per_position_logits()` from `bitnet_crossval::logits_compare`
**File**: `/home/steven/code/Rust/BitNet-rs/crossval/src/logits_compare.rs` (lines 49-102)

**Metrics Computed**:

| Metric | Purpose | Threshold |
|--------|---------|-----------|
| **Cosine Similarity** | Direction similarity [0,1] | cos_tol (default 0.999) |
| **L2 Distance** | Euclidean distance | Informational |
| **Max Absolute Diff** | Peak value difference | Informational |
| **First Divergence Token** | Position index | Based on cosine < (1 - threshold) |

**Divergence Detection** (logits_compare.rs lines 91-93):
```rust
if first_divergence_token.is_none() && (1.0 - cosine_sim) > COSINE_SIMILARITY_THRESHOLD {
    first_divergence_token = Some(pos);
}
```
Uses `COSINE_SIMILARITY_THRESHOLD = 1e-4` (hardcoded), then checks against `cos_tol` in output

---

## FFI Interface Deep Dive

### bitnet_sys Crate Structure

**File**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-sys/src/lib.rs`

```
bitnet_sys
├── bindings (unsafe raw FFI from bindgen) [generated code]
├── wrapper (safe wrappers)
│   ├── Model struct -> llama_model*
│   ├── Context struct -> llama_context*
│   ├── Session struct (Model + Context)
│   └── Functions: init_backend, free_backend, get_version
└── safe (high-level API)
    └── ModelHandle, is_available(), generate()
```

### FFI Safety Pattern

**Memory Management**: Manual cleanup with explicit Drop impls

```rust
impl Drop for Model {
    fn drop(&mut self) {
        let ptr = std::mem::replace(&mut self.ptr, std::ptr::null_mut());
        if !ptr.is_null() {
            unsafe { llama_free_model(ptr); }
        }
    }
}
```

**Guard Pattern** (main.rs line 2952):
```rust
let _guard = scopeguard::guard((), |_| bitnet_sys::wrapper::free_backend());
// Ensures free_backend() called on scope exit, even if error occurs
```

### FFI Availability Check

**File**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-sys/src/lib.rs` (lines 71-77)

```rust
pub fn is_available() -> bool {
    panic::catch_unwind(|| {
        wrapper::get_version();  // Try a simple C++ call
    }).is_ok()
}
```

**Usage in crossval-per-token** (main.rs lines 2944-2948):
```rust
if !bitnet_sys::is_available() {
    anyhow::bail!(
        "C++ FFI not available. Compile with --features crossval or set BITNET_CPP_DIR"
    );
}
```

---

## Tokenization Differences: crossval-per-token vs CLI

### crossval-per-token Tokenization
**File**: xtask/src/main.rs lines 2920-2922

```rust
// Rust side: Raw tokenization, NO template, NO BOS
let tokens = tokenizer.encode(prompt, false, false)?;

// C++ side: Uses llama.cpp tokenize with special handling
let cpp_tokens = cpp_session.tokenize(prompt)?;  // add_special=true
```

### CLI Tokenization
**File**: crates/bitnet-cli/src/commands/inference.rs

- Uses **prompt templates** (auto-detected from GGUF metadata or explicit)
- May prepend **system prompts**
- May prepend **BOS token** depending on template
- Applies **special token formatting** (e.g., LLaMA-3 `<|start_header_id|>`)

### Impact on Parity
**Token Sequence Mismatch**:
- CLI: `[BOS, template_formatted_prompt_tokens, special_tokens]`
- crossval-per-token: `[raw_prompt_tokens]`
- C++: Depends on tokenizer config in llama.cpp

**Result**: Different token sequences → Different logits at every position → False divergence detection

---

## Token-Parity Pre-Gate Integration Points

### Current Integration Flow
```
crossval-per-token command
├── Load model (Rust GGUF loader)
├── Tokenize prompt (Rust tokenizer + C++ tokenizer)
├── Eval logits (Rust forward pass + C++ eval)
├── Compare logits (cosine similarity)
└── Report divergence (if cos_sim < threshold)
```

### Proposed Pre-Gate Integration

**Location 1: Before Rust Evaluation** (main.rs line 2929-2933)
```rust
// BEFORE:
let rust_logits = eval_logits_all_positions(model_path_str, &token_ids)?;

// AFTER: Add pre-gate check
let pre_gate_result = token_parity_pre_gate::check_tokens(&tokens)?;
if pre_gate_result.status != PreGateStatus::Pass {
    eprintln!("PRE-GATE: Tokenization mismatch detected");
    // Report and continue/abort
}
let rust_logits = eval_logits_all_positions(model_path_str, &token_ids)?;
```

**Location 2: Between Rust & C++ Evaluation** (main.rs line 2940-2963)
```rust
// BEFORE:
let cpp_logits = cpp_session.context.get_all_logits(cpp_tokens.len())?;

// AFTER: Add token sequence validation
validate_token_parity(&token_ids, &cpp_tokens)?;

let cpp_logits = cpp_session.context.get_all_logits(cpp_tokens.len())?;
```

**Location 3: Before Logits Comparison** (main.rs line 2972-2974)
```rust
// BEFORE:
let divergence = compare_per_position_logits(&rust_logits, &cpp_logits);

// AFTER: Add per-token logits pre-gate
let pre_gate = check_per_token_logits_baseline(&rust_logits);
if pre_gate.has_issues() {
    eprintln!("PRE-GATE: Rust logits have baseline issues");
}

let divergence = compare_per_position_logits(&rust_logits, &cpp_logits);
```

---

## Error Handling

### Current Approach
1. **FFI Availability Check** (line 2944): Bail if C++ not available
2. **Rust Model Load**: Fail-closed on any error
3. **Tokenization Errors**: Propagate with context
4. **Comparison Results**: Always report metrics, exit 1 if divergence

### Command Invocation Analysis

**Lines 895-898** (main.rs):
```rust
Commands::CrossvalPerToken { model, tokenizer, prompt, max_tokens, cos_tol, format } => {
    crossval_per_token_cmd(&model, &tokenizer, &prompt, max_tokens, cos_tol, &format)?;
},
```

**Exit Behavior**:
- Success (no divergence): Exit 0 (line 3036)
- Divergence detected: Exit 1 (line 3033)
- FFI unavailable: Error propagation with message
- Model load failure: Error propagation

---

## Key Crate Interdependencies

| Crate | Module | Purpose | Location |
|-------|--------|---------|----------|
| `bitnet_tokenizers` | `loader::load_tokenizer()` | Universal tokenizer loader | - |
| `bitnet_inference` | `parity::eval_logits_all_positions()` | Rust forward pass, all positions | `crates/bitnet-inference/src/parity.rs:157-223` |
| `bitnet_sys` | `wrapper::Session` | Safe C++ wrapper (llama.cpp) | `crates/bitnet-sys/src/wrapper.rs:329-394` |
| `bitnet_crossval` | `logits_compare::compare_per_position_logits()` | Per-position comparison | `crossval/src/logits_compare.rs:49-102` |

---

## Quick Reference: File Paths

| Component | File | Key Lines |
|-----------|------|-----------|
| **Command Definition** | `xtask/src/main.rs` | 405-430 |
| **Implementation** | `xtask/src/main.rs` | 2901-3041 |
| **Rust Logits** | `crates/bitnet-inference/src/parity.rs` | 157-223 |
| **C++ FFI Wrapper** | `crates/bitnet-sys/src/wrapper.rs` | 329-394, 144-186, 285-293 |
| **Logits Comparison** | `crossval/src/logits_compare.rs` | 49-102 |
| **FFI Availability** | `crates/bitnet-sys/src/lib.rs` | 71-77 |

---

## Summary Table

| Aspect | Current Implementation | Pre-Gate Integration Point |
|--------|---|---|
| **Tokenization** | Rust (raw) + C++ (llama.cpp) | Before eval: validate token parity |
| **Token BOS** | Rust: false, C++: varies | Check BOS handling consistency |
| **Special Tokens** | Rust: false, C++: true | Validate special token handling |
| **Prompt Templates** | None (raw prompt) | Support template-aware tokenization |
| **Rust Eval** | `eval_logits_all_positions()` | Add pre-gate on logits baseline |
| **C++ Eval** | `Session::load_deterministic()` | Validate settings (1 thread, logits_all) |
| **Comparison** | Cosine similarity @ cos_tol | Add pre-gate check before comparison |
| **Error Handling** | Fail on FFI unavailable | Enhance with detailed token diagnostics |


# Quick Implementation Guide: Backend Routing for crossval-per-token

## TL;DR - Where to Change

**File**: `/home/steven/code/Rust/BitNet-rs/xtask/src/main.rs`

**Two locations need `match backend { ... }` statements**:

1. **Line ~3074** - C++ Tokenization (currently hardcoded to llama)
2. **Line ~3108** - C++ Logits Evaluation (currently hardcoded to llama)

---

## Quick Reference: Current Hardcoded Code

### Hardcoded #1: C++ Tokenization (Lines 3068-3078)

```rust
let model_path_str =
    model_path.to_str().ok_or_else(|| anyhow::anyhow!("Invalid model path"))?;
bitnet_sys::wrapper::init_backend();
let _guard = scopeguard::guard((), |_| bitnet_sys::wrapper::free_backend());

let mut cpp_session = bitnet_sys::wrapper::Session::load_deterministic(model_path_str)?;

// Tokenize with C++ tokenizer using the same formatted prompt
let cpp_tokens = cpp_session.tokenize(&formatted_prompt)?;
```

**What it does**: Loads llama.cpp session and tokenizes the prompt

**Needs to become**: 
```rust
let cpp_tokens = match backend {
    CppBackend::BitNet => { /* TODO: bitnet.cpp path */ },
    CppBackend::Llama => { /* current code */ },
};
```

---

### Hardcoded #2: C++ Logits Evaluation (Lines 3104-3111)

```rust
// Evaluate all positions
cpp_session.context.eval(&cpp_tokens, 0)?;

// Get all logits (requires logits_all=true in context)
let cpp_logits = cpp_session.context.get_all_logits(cpp_tokens.len())?;
```

**What it does**: Uses the same cpp_session to evaluate and get logits

**Problem**: `cpp_session` only exists in the llama.cpp path

**Needs to become**:
```rust
let cpp_logits = match backend {
    CppBackend::BitNet => { /* TODO: bitnet.cpp path */ },
    CppBackend::Llama => {
        // Re-initialize session (current logic)
        bitnet_sys::wrapper::init_backend();
        let _guard = scopeguard::guard((), |_| bitnet_sys::wrapper::free_backend());
        let mut cpp_session = bitnet_sys::wrapper::Session::load_deterministic(model_path_str)?;
        cpp_session.context.eval(&cpp_tokens, 0)?;
        cpp_session.context.get_all_logits(cpp_tokens.len())?
    },
};
```

---

## Implementation Pattern

### For C++ Tokenization

**Insert at line ~3050 (before println!("📝 Tokenizing prompt (C++)...;"))**

```rust
// Step 3: C++ tokenization
println!("📝 Tokenizing prompt (C++)...");

let model_path_str =
    model_path.to_str().ok_or_else(|| anyhow::anyhow!("Invalid model path"))?;

let cpp_tokens = match backend {
    CppBackend::BitNet => {
        eprintln!("TODO: Implement bitnet.cpp tokenization");
        unimplemented!("BitNet.cpp tokenization not yet implemented")
    }
    CppBackend::Llama => {
        bitnet_sys::wrapper::init_backend();
        let _guard = scopeguard::guard((), |_| bitnet_sys::wrapper::free_backend());
        let mut cpp_session = bitnet_sys::wrapper::Session::load_deterministic(model_path_str)?;
        cpp_session.tokenize(&formatted_prompt)?
    }
};

println!("Tokens: {} (C++)", cpp_tokens.len());
println!();
```

---

### For C++ Logits Evaluation

**Replace lines 3104-3118 (the cpp_session.context calls)**

```rust
// Step 6: Get C++ logits
println!("🔧 Evaluating C++ logits for all positions...");

let cpp_logits = match backend {
    CppBackend::BitNet => {
        eprintln!("TODO: Implement bitnet.cpp logits evaluation");
        unimplemented!("BitNet.cpp logits evaluation not yet implemented")
    }
    CppBackend::Llama => {
        bitnet_sys::wrapper::init_backend();
        let _guard = scopeguard::guard((), |_| bitnet_sys::wrapper::free_backend());
        let mut cpp_session = bitnet_sys::wrapper::Session::load_deterministic(model_path_str)?;
        cpp_session.context.eval(&cpp_tokens, 0)?;
        cpp_session.context.get_all_logits(cpp_tokens.len())?
    }
};

println!(
    "✓ C++: {} positions, vocab_size={}",
    cpp_logits.len(),
    cpp_logits.first().map(|v| v.len()).unwrap_or(0)
);
println!();
```

---

## Minimal Change for Phase 1

If you want **minimal changes** that preserve current behavior:

1. Keep all existing llama.cpp code
2. Wrap it in `CppBackend::Llama => { ... }`
3. Add `CppBackend::BitNet => unimplemented!(...)`

This ensures:
- No regression for llama.cpp (existing code path unchanged)
- Clear placeholder for future BitNet.cpp implementation
- Proper backend selection infrastructure in place

---

## Testing After Implementation

### Test 1: Llama Model (Verify No Regression)
```bash
cargo run -p xtask -- crossval-per-token \
  --model models/llama-3-8b-instruct.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 4 \
  --verbose
```

**Expected**: Should work exactly as before (routes to llama.cpp)

### Test 2: BitNet Model (Should Fail with Helpful Message)
```bash
cargo run -p xtask -- crossval-per-token \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 4 \
  --verbose
```

**Expected**: Should fail with `thread 'main' panicked at 'BitNet.cpp tokenization not yet implemented'`

### Test 3: Auto-Detection
```bash
cargo run -p xtask -- crossval-per-token \
  --model models/llama-3-8b-instruct.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "Test" \
  --verbose
  # Note: NO --cpp-backend flag - should auto-detect Llama
```

**Expected**: Should print "Selected backend: llama.cpp (auto-detected)"

---

## Variable Scope Issue

**⚠️ Important**: The `cpp_session` variable only exists in the `CppBackend::Llama` arm.

If logits evaluation code comes AFTER the tokenization match, you need to handle this:

### Option A: Separate matches (cleaner)
```rust
let cpp_tokens = match backend { ... };
// ... code ...
let cpp_logits = match backend { ... };  // Separate match with separate session
```

### Option B: Nested code block
```rust
let (cpp_tokens, cpp_logits) = match backend {
    CppBackend::BitNet => unimplemented!(),
    CppBackend::Llama => {
        bitnet_sys::wrapper::init_backend();
        let _guard = scopeguard::guard((), |_| bitnet_sys::wrapper::free_backend());
        let mut session = bitnet_sys::wrapper::Session::load_deterministic(model_path_str)?;
        let tokens = session.tokenize(&formatted_prompt)?;
        session.context.eval(&tokens, 0)?;
        let logits = session.context.get_all_logits(tokens.len())?;
        (tokens, logits)
    }
};
```

**Recommendation**: Use Option A (separate matches) for clarity and future extensibility.

---

## Files to Modify

| File | Lines | Change |
|------|-------|--------|
| `/home/steven/code/Rust/BitNet-rs/xtask/src/main.rs` | ~3050-3080 | Add backend match for tokenization |
| `/home/steven/code/Rust/BitNet-rs/xtask/src/main.rs` | ~3104-3120 | Add backend match for logits |

**No other files need changes** - backend infrastructure is already in place!

---

## Imports Already Available

```rust
use crate::crossval::CppBackend;  // Already imported at top of main.rs
use bitnet_sys::wrapper;           // Already imported
use bitnet_inference::parity::eval_logits_all_positions;  // Already imported
```

No new imports needed!

---

## Error Messages

When BitNet.cpp is unimplemented, show helpful message:

```rust
CppBackend::BitNet => {
    eprintln!();
    eprintln!("⚠️  BitNet.cpp backend not yet supported in crossval-per-token");
    eprintln!();
    eprintln!("Setup BitNet.cpp reference:");
    eprintln!("  eval \"$(cargo run -p xtask -- setup-cpp-auto --bitnet --emit=sh)\"");
    eprintln!();
    eprintln!("Then rebuild xtask:");
    eprintln!("  cargo clean -p xtask && cargo build -p xtask --features crossval-all");
    eprintln!();
    anyhow::bail!("BitNet.cpp tokenization not yet implemented")
}
```

---

## Verification Checklist

- [ ] Backend match added to tokenization code (line ~3074)
- [ ] Backend match added to logits evaluation code (line ~3108)
- [ ] CppBackend::Llama arm contains existing llama.cpp code
- [ ] CppBackend::BitNet arm has unimplemented!() placeholder
- [ ] No variable scope issues (cpp_session not used outside match)
- [ ] Test with llama model - should pass
- [ ] Test with bitnet model - should fail gracefully
- [ ] Auto-detection works (--verbose flag confirms backend selection)


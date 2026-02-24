# Backend Routing Quick Reference Card

## The Two Code Changes Required

### Change 1: C++ Tokenization (Line ~3074)

**BEFORE:**
```rust
let model_path_str = model_path.to_str()?;
bitnet_sys::wrapper::init_backend();
let _guard = scopeguard::guard((), |_| bitnet_sys::wrapper::free_backend());
let mut cpp_session = bitnet_sys::wrapper::Session::load_deterministic(model_path_str)?;
let cpp_tokens = cpp_session.tokenize(&formatted_prompt)?;
```

**AFTER:**
```rust
let model_path_str = model_path.to_str()?;
let cpp_tokens = match backend {
    CppBackend::BitNet => {
        unimplemented!("BitNet.cpp tokenization - Phase 2")
    }
    CppBackend::Llama => {
        bitnet_sys::wrapper::init_backend();
        let _guard = scopeguard::guard((), |_| bitnet_sys::wrapper::free_backend());
        let mut cpp_session = bitnet_sys::wrapper::Session::load_deterministic(model_path_str)?;
        cpp_session.tokenize(&formatted_prompt)?
    }
};
```

---

### Change 2: C++ Logits Evaluation (Line ~3108)

**BEFORE:**
```rust
cpp_session.context.eval(&cpp_tokens, 0)?;
let cpp_logits = cpp_session.context.get_all_logits(cpp_tokens.len())?;
```

**AFTER:**
```rust
let cpp_logits = match backend {
    CppBackend::BitNet => {
        unimplemented!("BitNet.cpp logits evaluation - Phase 2")
    }
    CppBackend::Llama => {
        bitnet_sys::wrapper::init_backend();
        let _guard = scopeguard::guard((), |_| bitnet_sys::wrapper::free_backend());
        let mut cpp_session = bitnet_sys::wrapper::Session::load_deterministic(model_path_str)?;
        cpp_session.context.eval(&cpp_tokens, 0)?;
        cpp_session.context.get_all_logits(cpp_tokens.len())?
    }
};
```

---

## What's Already Working

| Component | Status | No Action Needed |
|-----------|--------|------------------|
| CLI arg parsing | ✅ | Yes |
| Backend enum | ✅ | Yes |
| Auto-detection | ✅ | Yes |
| Preflight check | ✅ | Yes |
| Rust tokenization | ✅ | Yes |
| Token parity | ✅ | Yes |
| Rust logits eval | ✅ | Yes |
| Comparison logic | ✅ | Yes |
| Output formatting | ✅ | Yes |

---

## Testing Commands

### Regression Test (Llama)
```bash
cargo run -p xtask -- crossval-per-token \
  --model models/llama-3-8b-instruct.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "What is 2+2?" --verbose
```
**Expected**: Works exactly as today

### Placeholder Test (BitNet)
```bash
cargo run -p xtask -- crossval-per-token \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "What is 2+2?"
```
**Expected**: Fails with "BitNet.cpp tokenization not yet implemented"

### Auto-Detection Test
```bash
cargo run -p xtask -- crossval-per-token \
  --model models/llama-3-8b-instruct.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "Test" --verbose
```
**Expected**: Prints "Selected backend: llama.cpp (auto-detected)"

---

## Files to Modify

| File | Lines | Change |
|------|-------|--------|
| xtask/src/main.rs | ~3074 | Add backend match for tokenization |
| xtask/src/main.rs | ~3108 | Add backend match for logits |

**Total changes**: 2 match statements
**Total files affected**: 1
**Lines added**: ~30
**Lines removed**: 0 (just wrapped in match arms)

---

## Imports (Already Available)

```rust
use crate::crossval::CppBackend;           // Already imported
use bitnet_sys::wrapper;                   // Already imported
use bitnet_inference::parity::eval_logits_all_positions;  // Already imported
```

No new imports needed!

---

## Error Handling Pattern (Established)

```rust
use anyhow::Result;

// Parse with context
let path = input.to_str()?;

// Operations return Result<T>
let tokens = session.tokenize(&prompt)?;

// Match on backend
let result = match backend {
    CppBackend::Llama => /* ... */?,
    CppBackend::BitNet => unimplemented!(),
};
```

---

## Key Insights

1. **No Risk**: Wrapping existing llama.cpp code in a match arm breaks nothing
2. **No New Dependencies**: Uses existing imports and error handling
3. **Clear Placeholder**: `unimplemented!()` for BitNet is explicit and helpful
4. **Minimal Diff**: Easy to review and understand
5. **Future-Proof**: Phase 2 just fills in the BitNet arm

---

## Phase 2 Preview (Future)

When BitNet.cpp FFI wrappers are ready, replace:

```rust
CppBackend::BitNet => {
    unimplemented!("BitNet.cpp tokenization - Phase 2")
}
```

With:

```rust
CppBackend::BitNet => {
    tokenize_bitnet(model_path_str, &formatted_prompt)?
}
```

(Similar for logits evaluation)

---

## Documentation Links

- **BACKEND_ROUTING_ANALYSIS.md** - Full architectural reference
- **QUICK_IMPLEMENTATION_GUIDE.md** - Detailed step-by-step guide
- **DISPATCHER_ARCHITECTURE_SUMMARY.md** - Visual flow diagrams
- **EXPLORATION_INDEX.md** - Complete file index

---

## Quick Facts

- Backend infrastructure: 90% complete
- C++ dispatch points: 2
- Code changes needed: 2 match statements
- Implementation time: 1-2 hours
- Risk level: Minimal
- Regression risk: None (existing code path preserved)
- Test strategy: 3 simple commands


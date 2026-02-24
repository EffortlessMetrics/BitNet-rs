# BitNet.cpp Integration - Quick Reference

## Setup

```bash
# Fetch and build C++ reference
cargo run -p xtask -- fetch-cpp

# Set environment (or let default ~/.cache/bitnet_cpp be used)
export BITNET_CPP_DIR=$HOME/.cache/bitnet_cpp
```

## Key Files

| Component | Location | Purpose |
|-----------|----------|---------|
| C API Header | `crates/bitnet-sys/include/bitnet_c.h` | FFI contract (model, tokenize, eval) |
| C++ Shim | `crates/bitnet-sys/csrc/bitnet_c_shim.cc` | llama.cpp → C FFI bridge |
| Rust Wrappers | `crates/bitnet-sys/src/wrapper.rs` | Safe Rust abstractions |
| Build Config | `crates/bitnet-sys/build.rs` | Links C++ libs, generates bindings |
| Parity Tests | `crossval/tests/parity_bitnetcpp.rs` | Rust vs C++ comparison |
| Session Utils | `crates/bitnet-inference/src/ffi_session.rs` | Reusable FFI session |

## Core APIs

### Safe Rust Wrappers (Recommended)

```rust
use bitnet_sys::wrapper::{Session, Model, Context};

// Load model
let session = Session::load_deterministic("path/to/model.gguf")?;

// Tokenize
let tokens = session.tokenize("Hello world")?;

// Evaluate and get logits
let logits = session.eval_and_get_logits(&tokens, 0)?;

// Generate tokens
let generated = session.generate_greedy("Prompt", 32)?;
```

### Custom C Shim (For Parity Tests)

```rust
use bitnet_sys::{BitnetModel, BitnetContext, bitnet_eval_tokens, cpp_vocab_size};

let model = BitnetModel::from_file("path/to/model.gguf")?;
let ctx = BitnetContext::new(&model, 4096, 1, 0)?;

let tokens = bitnet_tokenize_text(&model, "Hello", true, false)?;
let logits = bitnet_eval_tokens(&ctx, &tokens, vocab_size)?;
```

## Logits Extraction (Key for Debugging)

### Per-Position Logits (llama.cpp feature)

```rust
// Setup: Enable logits for all token positions
context.eval(tokens, 0)?;

// Get logits for last token (most common)
let logits_last = context.get_logits()?;

// Get logits for specific position (requires logits_all=true)
let logits_pos_0 = context.get_logits_ith(0)?;
let logits_pos_5 = context.get_logits_ith(5)?;

// Get all position logits at once
let all_logits = context.get_all_logits(tokens.len())?;
```

## Cross-Validation Testing

### Run Parity Tests

```bash
export CROSSVAL_GGUF=/path/to/model.gguf
export BITNET_CPP_DIR=$HOME/.cache/bitnet_cpp

# Run comprehensive parity test
cargo test -p crossval --features crossval,integration-tests \
  parity_bitnetcpp -- --nocapture

# Check receipt output
cat docs/baselines/$(date +%Y-%m-%d)/parity-bitnetcpp.json
```

### What Gets Compared

- **Tokenization:** Input text → token IDs (should be identical)
- **Single-step logits:** Forward pass produces identical logits (tolerance: 1e-4)
- **Multi-step generation:** Greedy token sequence must match exactly
- **Cosine similarity:** Measures numeric alignment across vocabulary

## Building with FFI

```bash
# With FFI support (requires C++ libraries)
cargo build --features ffi

# Without FFI (pure Rust - always works)
cargo build --no-default-features --features cpu

# Test with FFI
cargo test --features ffi -p bitnet-sys
```

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `BITNET_CPP_DIR` | Path to built C++ implementation | `$HOME/.cache/bitnet_cpp` |
| `BITNET_CPP_PATH` | Legacy fallback for above | (unused if BITNET_CPP_DIR set) |
| `CROSSVAL_GGUF` | Model for parity tests | (required for crossval tests) |
| `CROSSVAL_PROMPT` | Custom prompt for parity test | "Hello" |

## Troubleshooting

### "BITNET_CPP_DIR not set"
```bash
cargo run -p xtask -- fetch-cpp
export BITNET_CPP_DIR=$HOME/.cache/bitnet_cpp
```

### "Failed to load C++ model"
- Verify model path is correct and file exists
- Check that C++ library links are working: `ldd $(find ~/.cache/bitnet_cpp -name "*.so")`

### "Logits differ by..."
- Normal at ±1e-4 tolerance (floating point differences)
- Beyond 1e-3 suggests quantization or arithmetic mismatch
- Check both sides use same model and deterministic settings

### "parity test skipped"
- Need `CROSSVAL_GGUF` environment variable
- Test auto-skips if C++ library unavailable (expected when not building with FFI)

## What Can We Compare?

✅ **Can Compare Now:**
- Logits (final layer outputs)
- Token sequences (greedy generation)
- Tokenization outputs
- Vocabulary size and model properties

❌ **Cannot Compare (Not in llama.cpp API):**
- Intermediate layer activations
- Attention weights
- Hidden state dimensions
- Gradient flows

🔧 **Could Add (Would Require C++ Fork/Extension):**
- Per-layer hidden states
- Attention map extraction
- Intermediate layer comparisons

## Performance Notes

- **Memory:** ~1 GB for 2B model (Rust + C++ both loaded)
- **Speed:** C++ typically 2-5x faster than Rust MVP kernels
- **Determinism:** Requires `OMP_NUM_THREADS=1`, `GGML_NUM_THREADS=1` for reproducibility

## Example: Full Parity Check

```rust
use bitnet_sys::wrapper::Session as CppSession;
use bitnet_inference::eval_logits_once;

#[test]
fn test_parity() -> Result<()> {
    let model_path = "path/to/model.gguf";
    let prompt = "Hello world";
    
    // Rust inference
    let rust_tokens = bitnet_tokenizers::tokenize(prompt)?;
    let rust_logits = eval_logits_once(model_path, &rust_tokens)?;
    
    // C++ inference
    let mut cpp_session = CppSession::load_deterministic(model_path)?;
    let cpp_tokens = cpp_session.tokenize(prompt)?;
    let cpp_logits = cpp_session.eval_and_get_logits(&cpp_tokens, 0)?;
    
    // Compare
    assert_eq!(rust_tokens, cpp_tokens, "Tokenization mismatch");
    assert!(logits_similar(&rust_logits, &cpp_logits, 1e-4), "Logits differ");
    
    Ok(())
}
```

## Architecture Diagram

```
bitnet.cpp (C++ Reference)
    ↓
llama.cpp (Backend)
    ↓
bitnet_c_shim.cc (C Wrapper)
    ↓
bitnet-sys (FFI Bindings)
    ↓
bitnet-sys/wrapper.rs (Safe Rust)
    ↓
crossval (Test Infrastructure)
    ↓
parity_bitnetcpp.rs (Parity Tests)
```

## Key APIs Summary

**Model Lifecycle:**
- `BitnetModel::from_file()` - Load model
- `BitnetContext::new()` - Create inference context

**Evaluation:**
- `bitnet_eval_tokens()` - Forward pass, return logits
- `bitnet_prefill()` - Prime KV cache
- `cpp_decode_greedy()` - Full generation loop

**Extraction:**
- `bitnet_tokenize_text()` - Text → token IDs
- `llama_get_logits()` - Final layer output (last token)
- `llama_get_logits_ith()` - Final layer output (specific position)

**Utilities:**
- `cpp_vocab_size()` - Get vocab size
- `Session::load_deterministic()` - Easy setup


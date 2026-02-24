# Backend-Aware Error Messages - Quick Reference

## TL;DR

Token parity errors now display backend-specific troubleshooting guidance for BitNet.cpp and llama.cpp.

## Key Features

### 1. Backend Identification
Error headers clearly identify which C++ backend is being compared:
```
❌ Token Sequence Mismatch with C++ Backend: BitNet
```

### 2. Backend-Specific Troubleshooting

**BitNet Backend:**
- Model compatibility checks
- Tokenizer path validation
- Suggestion to try llama backend
- BOS token handling

**LLaMA Backend:**
- LLaMA tokenizer compatibility
- Architecture matching
- Suggestion to try bitnet backend
- Special token handling

### 3. Utility Methods

```rust
use bitnet_crossval::backend::CppBackend;

// Get setup command
let cmd = CppBackend::BitNet.setup_command();
// Returns: "eval \"$(cargo run -p xtask -- setup-cpp-auto --bitnet --emit=sh)\""

// Get required libraries
let libs = CppBackend::Llama.required_libs();
// Returns: &["libllama", "libggml"]
```

## Usage

### For End Users

When you encounter a token mismatch error, the error message will guide you through:

1. **Identify the backend** - Header shows BitNet or LLaMA
2. **Check backend-specific issues** - Model compatibility, tokenizer matching
3. **Try common fixes** - Template settings, BOS handling
4. **Copy-paste example command** - Includes correct backend flag

### For Developers

```rust
use bitnet_crossval::backend::CppBackend;
use bitnet_crossval::token_parity::TokenParityError;

let error = TokenParityError {
    rust_tokens: vec![1, 2, 3],
    cpp_tokens: vec![1, 2, 4],
    first_diff_index: 2,
    prompt: "test".to_string(),
    backend: CppBackend::BitNet,
};

// Error message automatically includes backend-specific guidance
let formatted = format_token_mismatch_error(&error);
```

## Testing

```bash
# Run all backend tests
cargo test -p bitnet-crossval --no-default-features backend

# Run token parity tests
cargo test -p bitnet-crossval --no-default-features token_parity

# View error message demo
cargo run -p bitnet-crossval --example backend_error_demo --no-default-features
```

## Architecture

Two separate `CppBackend` enums exist:

1. **crossval/src/backend.rs** - For error messages (this implementation)
2. **xtask/src/crossval/backend.rs** - For CLI integration

Both can interoperate via `from_name()` method.

## Files Modified

- `crossval/src/backend.rs` - Added utility methods
- `crossval/src/token_parity.rs` - Already backend-aware (unchanged)
- `crossval/examples/backend_error_demo.rs` - New demo example

## Test Coverage

- ✅ 35 unit tests
- ✅ 8 integration tests
- ✅ 6 doctests
- ✅ All existing tests pass

## Example Output

```
❌ Token Sequence Mismatch with C++ Backend: BitNet
Fix BOS/template before comparing logits

Rust tokens (4):
  [128000, 128000, 1229, 374]

C++ tokens (3):
  [128000, 1229, 374]

First diff at index: 1
Mismatch: Rust token=128000, C++ token=1229

Troubleshooting for BitNet backend:
  • Verify your model is BitNet-compatible (microsoft-bitnet-b1.58-2B-4T-gguf)
  • Check tokenizer path matches model format
  • Try --cpp-backend llama if this is a LLaMA model
  • Verify --prompt-template setting (current: auto)
  • Check BOS token handling with --dump-ids

Common fixes:
  • Use --prompt-template raw (disable template formatting)
  • Add --no-bos flag (if BOS is duplicated)
  • Check GGUF chat_template metadata

Example command:
  cargo run -p xtask -- crossval-per-token \
    --model <model.gguf> \
    --tokenizer <tokenizer.json> \
    --prompt "What is 2+2?" \
    --prompt-template raw \
    --cpp-backend bitnet \
    --max-tokens 4
```

---

**See also:**
- Full implementation summary: `BACKEND_AWARE_ERROR_MESSAGES_SUMMARY.md`
- Token parity specification: `docs/explanation/token-parity-pregate.md`
- Dual backend guide: `docs/explanation/dual-backend-crossval.md`

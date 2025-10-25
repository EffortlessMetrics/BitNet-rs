# BitNet.cpp Wrapper Implementation Guide

**Purpose**: Map TODO sections in `crossval/src/bitnet_cpp_wrapper.cc` to actual llama.cpp API calls

**Status**: Ready for implementation

## File Structure

### Current State
- **File**: `crossval/src/bitnet_cpp_wrapper.cc`
- **Mode**: STUB (returns helpful errors)
- **Target Mode**: AVAILABLE (uses llama.cpp API)
- **Feature Flag**: `BITNET_AVAILABLE` (vs `BITNET_STUB`)

## TODO-to-API Mapping

### Section 1: Tokenization (Lines 87-157)

**Current**: Lines 90-157 contain production-ready commented code pattern

**APIs to Implement**:
```
✓ llama_load_model_from_file()      Line 96 comment
✓ llama_tokenize()                  Line 109 comment
✓ llama_free_model()                Line 119 comment
```

**Implementation Steps**:
1. Replace `#ifdef BITNET_AVAILABLE` guard (line 87)
2. Uncomment and adapt lines 95-145
3. Ensure two-pass pattern:
   - Pass 1: NULL query returns count
   - Pass 2: Buffer fill copies tokens
4. Add error handling for:
   - Model load failure
   - Tokenization failure
   - Buffer size mismatch

**Key Parameter**:
- `add_bos = (add_bos != 0)` - convert C int to bool
- `parse_special = (parse_special != 0)` - same conversion

### Section 2: Evaluation (Lines 224-312)

**Current**: Lines 227-312 contain production-ready commented code pattern

**APIs to Implement**:
```
✓ llama_load_model_from_file()      Line 232 comment
✓ llama_new_context_with_model()    Line 242 comment
✓ llama_n_vocab()                   Line 251 comment
✓ llama_batch_get_one()             Line 260 comment
✓ llama_decode()                    Line 260 comment
✓ llama_get_logits()                Line 268 comment
✓ llama_free()                      Line 278 comment
✓ llama_free_model()                Line 279 comment
```

**Implementation Steps**:
1. Replace `#ifdef BITNET_AVAILABLE` guard (line 224)
2. Uncomment and adapt lines 230-300
3. **Critical**: Set `logits_all = true` in context params (line 241)
4. Handle both pass patterns:
   - Pass 1: Shape query (NULL output)
   - Pass 2: Buffer copy
5. Implement all-positions logits in single decode:
   - No loop needed if `logits_all=true`
   - `llama_get_logits()` returns [n_tokens][n_vocab] array

**Key Parameter**:
- `logits_all = true` - enables all-position logits from single decode

### Critical Changes from Comments

| Comment Code | Actual Implementation | Reason |
|---|---|---|
| `bitnet_get_tokenizer()` | Skip - use model directly | API difference |
| `bitnet_tokenize_text()` | Use `llama_tokenize()` | Actual API |
| `bitnet_eval_all_positions()` | Use `logits_all=true` param | Not a function |
| `std::vector<...>` | Already OK | C++ allowed |
| `llama_batch_get_one()` | Use as-is | Exact match |

## Feature Flags

### Build Flag
```bash
# Currently in STUB mode:
cargo build --features bitnet-cpp-stub

# To enable AVAILABLE mode (after implementation):
cargo build --features bitnet-cpp-available
```

### Compilation Control
```c
#ifdef BITNET_AVAILABLE
  // Implement this section
#elif BITNET_STUB
  // Current stub code
#else
  #error "Must define either BITNET_STUB or BITNET_AVAILABLE"
#endif
```

## Includes to Add

```cpp
#include <llama.h>          // llama.cpp C API
#include <vector>           // For token buffer
#include <cstring>          // memcpy, snprintf (already present)
#include <cstdio>           // Already present
#include <cstdint>          // Already present
```

## Recommended Implementation Order

### Phase 1: Tokenization Only
1. Focus on `crossval_bitnet_tokenize()`
2. Keep eval as STUB temporarily
3. Test tokenization end-to-end
4. Commit

### Phase 2: Evaluation
1. Implement `crossval_bitnet_eval_with_tokens()`
2. Test with single token, then multiple tokens
3. Validate logits shape and values
4. Commit

### Phase 3: Integration
1. Enable both functions in AVAILABLE mode
2. Run full cross-validation tests
3. Profile performance
4. Document results

## Testing Strategy

### Unit Tests
1. Create test with small BitNet model
2. Verify tokenization matches llama.cpp output
3. Verify logits shape is correct
4. Verify logits values are in reasonable range

### Integration Tests
1. Run `crossval_bitnet_tokenize()` on sample prompts
2. Run `crossval_bitnet_eval_with_tokens()` on tokenized input
3. Compare logits with direct llama.cpp inference

### Acceptance Criteria
- Two-pass pattern works for both functions
- Error messages are informative
- Memory is properly freed
- No leaks on error paths
- Logits match llama.cpp baseline

## Environment Setup

### Required
```bash
export BITNET_CPP_DIR=/home/steven/.cache/bitnet_cpp
export LD_LIBRARY_PATH=$BITNET_CPP_DIR/build/lib:$LD_LIBRARY_PATH
```

### Cargo.toml Updates (if needed)
```toml
[dependencies]
# No new deps - link time only

[build-dependencies]
# Might need cc crate if using build.rs for C++ compilation
```

### Build Configuration (build.rs)
```rust
// In crossval/build.rs
let bitnet_cpp_dir = env::var("BITNET_CPP_DIR").unwrap_or_else(|_| 
    "/home/steven/.cache/bitnet_cpp".to_string()
);

println!("cargo:rustc-link-search=native={}/build/lib", bitnet_cpp_dir);
println!("cargo:rustc-link-lib=llama");
println!("cargo:rustc-link-lib=ggml");
```

## Debugging Tips

### Compiler Errors
1. Check `llama.h` location: `-I$BITNET_CPP_DIR/3rdparty/llama.cpp/include`
2. Check library path: `-L$BITNET_CPP_DIR/build/lib`
3. Verify libraries exist: `ls -la $BITNET_CPP_DIR/build/lib/lib*.so`

### Runtime Errors
1. Use `RUST_LOG=debug` for detailed output
2. Set `LD_LIBRARY_PATH` correctly before running tests
3. Check error messages from wrapper functions (use `err` buffer)

### Logits Validation
1. Compare shape: `[n_tokens][n_vocab]`
2. Check for NaN/infinity values
3. Compare with direct llama.cpp output
4. Verify cosine similarity for cross-validation

## Performance Expectations

### Tokenization
- CPU-bound
- Depends on text length
- ~1000s of tokens/second typical

### Inference
- **CRITICAL**: Current implementation loads model per-call (inefficient)
- For 2B model on CPU: ~0.1-1 tok/s (depends on SIMD)
- Per-call overhead (load): ~100-500ms
- v0.2 should cache model

## Known Limitations

1. **Per-Call Load**: Model loaded/freed on every call
   - Fix in v0.2 with session API
   
2. **No GPU Support in MVP**: `n_gpu_layers = 0`
   - Will be added in v0.3
   
3. **Single Sequence Only**: No batch processing of sequences
   - Future enhancement using `llama_batch_init()`

## Files Modified

- `crossval/src/bitnet_cpp_wrapper.cc` - main implementation
- `crossval/src/cpp_bindings.rs` - may need minor updates to feature flags
- `crossval/build.rs` - build configuration
- `Cargo.toml` - feature flag definition (if new)

## Success Metrics

- [ ] Both wrapper functions compile with `BITNET_AVAILABLE`
- [ ] Tokenization matches llama.cpp output
- [ ] Logits shape is correct
- [ ] Cross-validation tests pass
- [ ] No memory leaks
- [ ] Error handling is robust


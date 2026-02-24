# BitNet.cpp API Discovery Summary

**Date**: October 25, 2025  
**Status**: ✓ COMPLETE - Ready for Implementation

## Key Finding

**BitNet.cpp uses the llama.cpp C API directly.** There is no separate BitNet-specific tokenization or inference API. The wrapper code should use llama.cpp functions, not hypothetical bitnet-specific ones.

## What Was Discovered

### Location
```
BitNet.cpp: /home/steven/.cache/bitnet_cpp/
Headers:   /home/steven/.cache/bitnet_cpp/3rdparty/llama.cpp/include/llama.h
Libraries: /home/steven/.cache/bitnet_cpp/build/lib/libllama.so, libggml.so
```

### API Identification
✓ Model loading: `llama_load_model_from_file()`  
✓ Context creation: `llama_new_context_with_model()`  
✓ Tokenization: `llama_tokenize()`  
✓ Inference: `llama_decode()`  
✓ Logits extraction: `llama_get_logits()`  
✓ Resource cleanup: `llama_free()`, `llama_free_model()`  

### Critical Insight
- Set **`logits_all = true`** in context parameters for all-position logits in single decode
- Use **two-pass pattern** for buffer negotiation (NULL query → size, then buffer fill)
- **Cast away const** on tokens array for llama_decode (API quirk)

## Deliverables Created

### 1. Full Specification
**File**: `docs/specs/bitnet-cpp-api-requirements.md` (371 lines)

Contents:
- Complete llama.cpp C API documentation
- Function signatures with parameter descriptions
- Two-pass buffer pattern explanation
- Implementation examples for both wrapper functions
- Build configuration
- Memory management guide
- Error handling patterns

### 2. Quick Reference
**File**: `docs/specs/bitnet-cpp-api-quick-reference.md` (162 lines)

Contents:
- One-liner function table
- Critical parameters
- Minimal code examples
- Memory management summary
- Logits layout and indexing
- Build flags quick reference

### 3. Implementation Guide
**File**: `docs/specs/bitnet-cpp-wrapper-implementation-guide.md` (280+ lines)

Contents:
- TODO-to-API mapping for wrapper.cc
- Line-by-line implementation steps
- Feature flag configuration
- Testing strategy
- Debugging tips
- Performance expectations
- Success metrics

## Implementation Roadmap

### Phase 1: Tokenization (Est. 1-2 hours)
1. Uncomment lines 95-145 in `bitnet_cpp_wrapper.cc`
2. Adapt to actual llama.cpp API (no hypothetical functions)
3. Add error handling
4. Test with sample prompt
5. Commit

### Phase 2: Inference (Est. 2-3 hours)
1. Uncomment lines 230-300 in `bitnet_cpp_wrapper.cc`
2. Set `logits_all = true` in context params
3. Verify all-positions logits in single decode
4. Test with tokenized input
5. Validate logits shape and values
6. Commit

### Phase 3: Integration & Validation (Est. 1-2 hours)
1. Enable both functions in AVAILABLE mode
2. Run cross-validation tests
3. Profile performance
4. Document results
5. Commit

## API Quick Facts

### Model Loading
```c
struct llama_model_params params = llama_model_default_params();
params.use_mmap = true;
struct llama_model *model = llama_load_model_from_file(path, params);
```

### Context Creation
```c
struct llama_context_params params = llama_context_default_params();
params.n_ctx = 2048;
params.logits_all = true;  // CRITICAL: all-position logits
struct llama_context *ctx = llama_new_context_with_model(model, params);
```

### Tokenization
```c
// Pass 1: Query size
int n = llama_tokenize(model, text, len, NULL, 0, true, false);

// Pass 2: Fill buffer
llama_tokenize(model, text, len, tokens, n, true, false);
```

### Inference
```c
struct llama_batch batch = llama_batch_get_one(tokens, n_tokens, 0, 0);
llama_decode(ctx, batch);
float *logits = llama_get_logits(ctx);  // [n_tokens][n_vocab]
```

## Critical Build Flags

```bash
# Headers
-I/home/steven/.cache/bitnet_cpp/3rdparty/llama.cpp/include

# Libraries
-L/home/steven/.cache/bitnet_cpp/build/lib -lllama -lggml

# Environment
export BITNET_CPP_DIR=/home/steven/.cache/bitnet_cpp
export LD_LIBRARY_PATH=$BITNET_CPP_DIR/build/lib:$LD_LIBRARY_PATH
```

## Next Steps

1. **Review docs/specs/bitnet-cpp-api-requirements.md** - full reference
2. **Review docs/specs/bitnet-cpp-wrapper-implementation-guide.md** - step-by-step
3. **Start Phase 1** - implement tokenization
4. **Run tests** - validate against llama.cpp baseline
5. **Proceed to Phase 2** - implement inference
6. **Full integration** - enable AVAILABLE mode

## No Surprises

The wrapper code had good commentary! The uncommented sections are mostly correct patterns. Key adjustments:
- No `bitnet_get_tokenizer()` - use model directly
- No `bitnet_tokenize_text()` - use `llama_tokenize()`
- No `bitnet_eval_all_positions()` - use context parameter instead

## Files Modified

- `docs/specs/bitnet-cpp-api-requirements.md` - NEW
- `docs/specs/bitnet-cpp-api-quick-reference.md` - NEW
- `docs/specs/bitnet-cpp-wrapper-implementation-guide.md` - NEW

## Success Criteria

- [x] BitNet.cpp location identified
- [x] API fully documented
- [x] Implementation patterns provided
- [x] Build configuration clear
- [x] Memory management documented
- [x] Error handling patterns defined
- [x] Test strategy provided
- [ ] (Future) Implementation complete and tested

---

**Time to implement**: ~4-7 hours total  
**Difficulty**: Low-Medium (straightforward API, clear patterns)  
**Risk**: Low (C FFI well-understood, two-pass pattern proven)  
**Blocking Issues**: None identified


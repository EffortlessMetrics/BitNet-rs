# BitNet.cpp API Discovery - Documentation Index

**Discovery Date**: October 25, 2025  
**Status**: COMPLETE - All APIs identified and documented

## Overview

This directory contains comprehensive documentation of the BitNet.cpp API discovered during G2 discovery phase. BitNet.cpp uses the llama.cpp C API; there is no separate BitNet-specific public interface.

## Files in This Directory

### 1. bitnet-cpp-api-requirements.md (Full Reference)
**Purpose**: Complete API documentation with implementation examples  
**Audience**: Developers implementing the wrapper  
**Length**: 371 lines

**Contents**:
- Executive summary
- Available artifacts location
- Complete API reference (6 sections)
- Model loading, context creation, tokenization, batching, decoding, logits
- Implementation patterns for both wrapper functions
- Build configuration and CMake integration
- API differences from commented code
- Potential issues and workarounds
- Validation checklist
- Recommended implementation order

**When to use**: For detailed implementation, full API reference, build setup

### 2. bitnet-cpp-api-quick-reference.md (Cheat Sheet)
**Purpose**: Quick lookup of functions and patterns  
**Audience**: All developers  
**Length**: 161 lines

**Contents**:
- One-liner function table with line numbers
- Critical model and context parameters
- Minimal working examples (tokenization, inference)
- Error handling pattern
- Required includes
- Memory management summary
- Logits layout and indexing
- Two-pass API patterns
- Build configuration quick reference
- What's NOT needed (no BitNet-specific API)

**When to use**: Quick lookup, code patterns, parameter reference

### 3. bitnet-cpp-wrapper-implementation-guide.md (Step-by-Step)
**Purpose**: Implementation roadmap with specific steps  
**Audience**: Developer implementing `bitnet_cpp_wrapper.cc`  
**Length**: 241 lines

**Contents**:
- File structure and current state
- TODO-to-API mapping for both functions
- Line-by-line implementation steps for tokenization (lines 87-157)
- Line-by-line implementation steps for evaluation (lines 224-312)
- Critical changes from commented code
- Feature flag configuration
- Recommended implementation phases (tokenization, then inference, then integration)
- Testing strategy (unit, integration, acceptance)
- Environment setup
- Build configuration details
- Debugging tips
- Performance expectations
- Known limitations and future improvements
- Success metrics

**When to use**: Planning implementation, step-by-step guidance, testing

## Quick Start Paths

### "I need to implement the wrapper now"
1. Read: Quick Reference (2 min)
2. Read: Implementation Guide Phase 1 (10 min)
3. Start coding tokenization
4. Reference: Full API Requirements as needed

### "I need to understand the API"
1. Read: Full API Requirements (20 min)
2. Review: Implementation examples (5 min)
3. Check: Quick Reference for specific functions

### "I'm debugging a specific function"
1. Search: Quick Reference for function
2. Look up: Full requirements for detailed docs
3. Check: Implementation guide for expected behavior

## Key Findings Summary

### Location
- **Root**: `/home/steven/.cache/bitnet_cpp/`
- **Headers**: `3rdparty/llama.cpp/include/llama.h`
- **Libraries**: `build/lib/libllama.so`, `libggml.so`

### Core API Functions Needed

| Task | Function | Status |
|------|----------|--------|
| Load model | `llama_load_model_from_file()` | ✓ Available |
| Create context | `llama_new_context_with_model()` | ✓ Available |
| Tokenize | `llama_tokenize()` | ✓ Available |
| Decode | `llama_decode()` | ✓ Available |
| Get logits | `llama_get_logits()` | ✓ Available |
| Get vocab size | `llama_n_vocab()` | ✓ Available |
| Create batch | `llama_batch_get_one()` | ✓ Available |
| Free resources | `llama_free()`, `llama_free_model()` | ✓ Available |

### Critical Parameters

**For Model Loading**:
```c
use_mmap = true      // Enable memory mapping
n_gpu_layers = 0     // CPU only (for MVP)
```

**For Context**:
```c
n_ctx = 2048         // Context size
logits_all = true    // CRITICAL: all-position logits
n_threads = 4        // CPU threads
```

### Two-Pass API Pattern
Both wrapper functions use two-pass buffer negotiation:
1. **Pass 1**: NULL buffer query returns size
2. **Pass 2**: Real buffer call copies data

### Implementation Estimate
- **Phase 1 (Tokenization)**: 1-2 hours
- **Phase 2 (Inference)**: 2-3 hours
- **Phase 3 (Integration)**: 1-2 hours
- **Total**: 4-7 hours

## Important Notes

### What's NOT Needed
- ❌ `bitnet_get_tokenizer()` - doesn't exist
- ❌ `bitnet_tokenize_text()` - doesn't exist
- ❌ `bitnet_eval_all_positions()` - doesn't exist

Use `llama_*` functions directly instead.

### What's Different from Comments
The uncommented code in `bitnet_cpp_wrapper.cc` mentions hypothetical functions. The actual API is:
- Tokenization: Use `llama_tokenize()` directly on model
- Inference: Use `logits_all=true` parameter, not a separate function
- All else: Matches the comments closely

### Performance Notes
- Current MVP loads model per-call (inefficient)
- Will be optimized in v0.2 with context caching
- Expected: ~0.1-1 tok/s for 2B models on CPU
- Per-call overhead: ~100-500ms (model load)

## Comprehensive Documentation Suite

### Complete BitNet.cpp Integration Guides

1. **[BitNet.cpp API Requirements](bitnet-cpp-api-requirements.md)** (Full Reference)
   - Complete API documentation with implementation examples
   - All llama.cpp functions needed for wrapper
   - Build configuration and setup

2. **[BitNet.cpp API Quick Reference](bitnet-cpp-api-quick-reference.md)** (Cheat Sheet)
   - Quick lookup of functions and patterns
   - Minimal working examples
   - Common parameters at a glance

3. **[BitNet.cpp Wrapper Implementation Guide](bitnet-cpp-wrapper-implementation-guide.md)** (Step-by-Step)
   - Implementation roadmap with specific steps
   - Line-by-line guidance for both FFI functions
   - Testing strategy and success criteria

4. **[BitNet.cpp AVAILABLE Mode Wiring Guide](bitnet-available-wiring.md)** ⭐
   - **Comprehensive wiring guide for production FFI integration**
   - Required headers and library dependencies
   - Build system configuration (build.rs)
   - Symbol visibility and linking best practices
   - Platform-specific notes (Linux, macOS, Windows)
   - Common compilation errors with fixes
   - Troubleshooting guide with diagnostics
   - Verification checklist

5. **[BitNet.cpp FFI Integration Sockets](bitnet-cpp-ffi-sockets.md)** 🆕 LATEST
   - **Technical specification for 6 missing FFI sockets**
   - Context initialization for persistent model loading
   - BitNet-specific tokenization and inference (optional, with llama.cpp fallback)
   - Session API for 10-100× performance improvements
   - dlopen loader architecture for runtime symbol resolution
   - Graceful degradation when symbols unavailable
   - Migration path from current per-call model loading to session API
   - Testing strategy and performance benchmarks

6. **[BitNet.cpp Session API](bitnet-session-api.md)** 📘
   - High-level session management design
   - Lifecycle management and resource cleanup
   - Integration with FFI sockets
   - Performance optimization strategies

### When to Use Which Guide

| Task | Recommended Guide |
|------|-------------------|
| Understanding the API | API Requirements (Full Reference) |
| Quick function lookup | API Quick Reference (Cheat Sheet) |
| Implementing wrapper code | Wrapper Implementation Guide |
| Build system integration | AVAILABLE Mode Wiring Guide ⭐ |
| Troubleshooting linker errors | AVAILABLE Mode Wiring Guide ⭐ |
| Platform-specific issues | AVAILABLE Mode Wiring Guide ⭐ |
| Symbol visibility problems | AVAILABLE Mode Wiring Guide ⭐ |
| Designing persistent context API | FFI Integration Sockets 🆕 |
| Runtime symbol resolution | FFI Integration Sockets 🆕 |
| Session management architecture | Session API + FFI Integration Sockets |
| Performance optimization (10-100×) | FFI Integration Sockets 🆕 |
| dlopen loader implementation | FFI Integration Sockets 🆕 |

## Related Documentation

See also in parent `/docs` directory:
- `BITNET_CPP_INTEGRATION_ANALYSIS.md` - Previous analysis
- `C_FFI_INTEGRATION_ANALYSIS.md` - FFI strategy
- `CROSSVAL.md` - Cross-validation framework

## Files to Modify

1. **`crossval/src/bitnet_cpp_wrapper.cc`**
   - Tokenization: Replace lines 87-157
   - Inference: Replace lines 224-312
   - Feature flag: `BITNET_AVAILABLE` vs `BITNET_STUB`

2. **`crossval/build.rs`** (minor)
   - Ensure llama.so, ggml.so linked
   - Set include path for llama.h

3. **`Cargo.toml`** (maybe)
   - Add feature flag definition if needed

## Success Criteria

All items should be checked before implementation is complete:

- [ ] Both functions compile with `BITNET_AVAILABLE`
- [ ] Tokenization output matches llama.cpp baseline
- [ ] Logits shape is `[n_tokens][n_vocab]`
- [ ] All-positions logits in single decode call
- [ ] Error handling for all failure paths
- [ ] Memory properly freed in all paths
- [ ] No memory leaks (valgrind clean)
- [ ] Integration tests pass
- [ ] Cross-validation tests pass
- [ ] Performance profiled and documented

## Implementation Workflow

```
1. Create feature-branch
   git checkout -b feat/bitnet-cpp-available-mode

2. Phase 1: Tokenization
   - Implement crossval_bitnet_tokenize()
   - Test tokenization end-to-end
   - Commit

3. Phase 2: Inference
   - Implement crossval_bitnet_eval_with_tokens()
   - Test inference end-to-end
   - Commit

4. Phase 3: Integration
   - Run cross-validation tests
   - Profile performance
   - Document results
   - Create pull request

5. Review & Merge
   - Code review
   - CI passes
   - Merge to main
```

## Getting Help

1. **Quick question?** → Check Quick Reference
2. **Implementation stuck?** → Check Implementation Guide
3. **API details?** → Check Full Requirements
4. **Specific error?** → Debug Tips section in Implementation Guide

## Document Versions

- v1.0 (Oct 25, 2025): Initial discovery, all APIs documented
- (Future versions as implementation progresses)

---

**Last Updated**: October 25, 2025  
**Next Update**: After Phase 1 implementation complete


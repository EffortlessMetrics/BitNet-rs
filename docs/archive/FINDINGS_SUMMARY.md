# Library Discovery & Linking - Investigation Summary

## Investigation Overview

This analysis explored the BitNet.rs build system's approach to discovering, linking, and selecting compute backends (CPU vs GPU, Rust vs C++).

**Duration**: Comprehensive source code review
**Scope**: Build scripts, FFI integration, feature gates, environment configuration
**Output**: 3 detailed reference documents + implementation roadmap

---

## Key Findings

### 1. Discovery System Architecture

The BitNet.rs build system uses a **staged discovery pattern**:

1. **Stage 1 - Optional**: Feature gates determine if FFI linking is attempted
2. **Stage 2 - Fallible**: Library search across standard CMake output locations
3. **Stage 3 - Graceful**: Mock implementations used if C++ libraries not found
4. **Stage 4 - Platform-aware**: Link platform-specific runtime dependencies

This is **NOT a hard requirement** - the system gracefully degrades to pure Rust implementations.

### 2. Library Categories

Three types of libraries are discovered:

| Category | Currently Linked | Location |
|----------|-----------------|----------|
| **BitNet-specific** | ❌ No | `build/lib/libbitnet.*` |
| **llama.cpp core** | ✅ Yes | `build/3rdparty/llama.cpp/src/libllama.*` |
| **GGML quantization** | ✅ Yes (optional) | `build/3rdparty/llama.cpp/ggml/src/libggml.*` |

### 3. Current Capabilities

**Working Well**:
- Automatic library discovery from standard CMake locations
- Environment variable overrides (BITNET_CPP_DIR, BITNET_CROSSVAL_LIBDIR)
- Feature-gated GPU/CPU compilation
- Platform-specific standard library linking
- RPATH support for runtime resolution
- Graceful fallback to mock implementations

**Not Implemented**:
- Detection of CUDA vs CPU-only builds
- Kernel capability registry
- Symbol analysis to determine available backends
- ABI compatibility validation
- Runtime backend selection
- Backend enforcement

### 4. Build Script Differences

Two main build scripts handle FFI:

**`crossval/build.rs`** - Permissive
- Searches for all libraries
- Links any found libraries
- Warns if none found, but continues
- Used by cross-validation tests
- **Behavior**: "Try to link, fail gracefully"

**`bitnet-sys/build.rs`** - Strict
- Requires BITNET_CPP_DIR to be set
- Requires build directory to exist
- Fails early if libraries not found and FFI enabled
- Generates bindgen bindings
- **Behavior**: "All or nothing"

### 5. Environment Variable Resolution Chain

```
BITNET_CROSSVAL_LIBDIR (highest priority, explicit)
  ↓
BITNET_CPP_DIR (primary, recommended)
  ↓
BITNET_CPP_PATH (legacy fallback)
  ↓
$HOME/.cache/bitnet_cpp (default location)
```

### 6. GPU vs CPU Detection

**Current**: Compile-time only via feature gates
```bash
cargo build --features gpu      # Links CUDA libraries
cargo build --features cpu      # CPU-only, no GPU linking
```

**Missing**: Runtime detection of whether CUDA is actually available

### 7. C Interface (FFI) Symbols

Six core symbols exposed by the shim:
- `bitnet_model_new_from_file()` - Load GGUF
- `bitnet_context_new()` - Create inference context
- `bitnet_tokenize()` - Text → token IDs
- `bitnet_eval()` - Get logits for tokens
- `bitnet_prefill()` - Prime KV cache
- `bitnet_decode_greedy()` - Generate tokens

These forward to llama.cpp C API internally.

---

## Critical Gaps for Dual-Backend Support

### Gap 1: Backend Variant Detection

**Problem**: Can't distinguish between:
- CUDA-enabled bitnet.cpp vs CPU-only
- Different quantization kernel implementations

**Impact**: May link incompatible C++ library at build time

**Solution**: Symbol analysis tool + cfg flags

### Gap 2: Kernel Capability Registry

**Problem**: No centralized way to query available kernels

**Impact**: Code must hard-code assumptions about backend availability

**Solution**: Create `KernelBackend` enum + capabilities struct

### Gap 3: Runtime Validation

**Problem**: No checks that loaded library is compatible

**Impact**: Silent failures if wrong backend loaded

**Solution**: Add FFI library validation at startup

### Gap 4: Symbol Availability

**Problem**: Can't query what functions are actually available in loaded library

**Impact**: Can't select optimized code paths based on actual library

**Solution**: Use `nm` command to analyze symbols

---

## Implementation Roadmap

A 5-phase plan to add dual-backend support:

### Phase 1 (1-2 weeks): Infrastructure
- Create kernel registry module
- Add symbol analysis tool to xtask
- Enhance build script logging
- Document current implementation

### Phase 2 (1 week): Build-time Detection
- Implement symbol analysis in build scripts
- Emit cfg flags based on detected backends
- Add optional symbol-analysis feature

### Phase 3 (1 week): Runtime Validation
- Add library validation at startup
- Implement backend capability detection
- Expose detection API

### Phase 4 (1 week): Testing & Enforcement
- Add backend selection tests
- Enhance CI validation
- Create compatibility matrix

### Phase 5 (Future): Extended Features
- Runtime kernel switching
- Dynamic library hot-reload
- Kernel capability reporting

**Total MVP**: 4-5 weeks for complete implementation

---

## Specific Code Locations

### Build Scripts
| File | Purpose | Lines |
|------|---------|-------|
| `crossval/build.rs` | Primary discovery | 40-94 |
| `bitnet-sys/build.rs` | FFI linking | 74-146 |
| `bitnet-kernels/build.rs` | GPU detection | 30-61 |

### C Interface
| File | Purpose |
|------|---------|
| `crates/bitnet-sys/include/bitnet_c.h` | FFI header |
| `crates/bitnet-sys/csrc/bitnet_c_shim.cc` | C++ implementation |
| `crates/bitnet-sys/src/wrapper.rs` | Safe Rust wrappers |

### Integration
| File | Purpose |
|------|---------|
| `xtask/src/cpp_setup_auto.rs` | Automated setup |
| `ci/fetch_bitnet_cpp.sh` | Build C++ reference |
| `scripts/diagnose_cpp.sh` | Diagnostic tool |

---

## Recommendations

### Immediate (Next Sprint)
1. Save reference documentation (DONE)
2. Review current limitations with team
3. Decide on Phase 1 priority

### Short-term (This Quarter)
1. Implement kernel registry module
2. Add symbol analysis tool
3. Enhance build script logging
4. Add basic backend detection

### Medium-term (Next Quarter)
1. Implement runtime validation
2. Add backend enforcement
3. Comprehensive testing
4. CI integration

### Long-term (Post-MVP)
1. Runtime kernel switching
2. Hot-reload support
3. Advanced capability reporting

---

## Success Metrics

After implementation, verify:

- [ ] All library discovery paths tested
- [ ] CI validates both CPU and CUDA builds
- [ ] Clear error messages on backend mismatch
- [ ] >90% test coverage for backend selection
- [ ] Documentation covers all configurations
- [ ] Build succeeds even without C++ libraries
- [ ] Runtime library validation prevents crashes

---

## Related Issues

The current build system partially addresses these issues:

- **#439** (Feature gate unification): ✅ Resolved - unified GPU predicate
- **#254** (Shape mismatch): ❌ Not addressed (quantization issue)
- **#260** (Mock elimination): ❌ Partially - mocks still used as fallback
- **#469** (Tokenizer parity): ❌ Not addressed (cross-validation issue)

This investigation enables **backend selection** to work correctly once above issues are resolved.

---

## Documentation Generated

Three reference documents created:

1. **library-discovery-and-linking.md** (4800+ words)
   - Comprehensive architecture overview
   - Detailed library search flowchart
   - Current capabilities and limitations
   - Build system changes needed

2. **library-discovery-quick-reference.md** (350+ words)
   - Quick lookup tables
   - TL;DR search chains
   - Common issues and fixes
   - Testing examples

3. **dual-backend-roadmap.md** (3000+ words)
   - 5-phase implementation plan
   - Code examples for each phase
   - Timeline and effort estimates
   - Success criteria

All documents saved to `/docs/reference/` for future reference.

---

## Conclusion

BitNet.rs implements a **sophisticated, flexible library discovery system** that gracefully handles:
- Optional FFI integration
- Multiple library formats
- Platform-specific linking
- Environment variable configuration
- Fallback to pure Rust implementations

The missing pieces for **full dual-backend support** are well-understood and can be implemented incrementally without breaking existing functionality. A 5-phase roadmap outlines the path forward with low risk.


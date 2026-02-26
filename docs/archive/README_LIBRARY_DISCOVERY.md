# Library Discovery & Linking Investigation

This directory contains comprehensive documentation of the BitNet-rs build system's approach to discovering, linking, and selecting compute backends.

## Documents in This Collection

### 1. **FINDINGS_SUMMARY.md** - START HERE
**Quick executive summary** of all findings (5 min read)
- Key findings
- Architecture overview
- Critical gaps
- Implementation roadmap
- Recommendations

### 2. **library-discovery-and-linking.md** - COMPREHENSIVE REFERENCE
**Complete technical documentation** of the discovery system (15-20 min read)
- Detailed library discovery status
- Where libraries are located
- How library selection works
- Available C interface symbols
- Build system changes needed
- Environment variable reference
- Build process flowchart

### 3. **library-discovery-quick-reference.md** - QUICK LOOKUP
**Fast reference tables and command examples** (5 min reference)
- TL;DR search chain
- Library categories table
- FFI available symbols
- Feature gate behavior
- Environment variable precedence
- Common issues & fixes
- Testing examples

### 4. **dual-backend-roadmap.md** - IMPLEMENTATION GUIDE
**Detailed 5-phase implementation plan** (10-15 min read)
- Current state assessment
- Phase 1-5 detailed plans
- Code examples and pseudocode
- Timeline and effort estimates
- Success criteria
- Dependencies and blockers

---

## Quick Navigation

### I Want To...

**Understand the current system**
→ Start with [FINDINGS_SUMMARY.md](FINDINGS_SUMMARY.md), then read [library-discovery-and-linking.md](library-discovery-and-linking.md)

**Get the library search chain quickly**
→ See [library-discovery-quick-reference.md](library-discovery-quick-reference.md) (TL;DR section)

**Debug a linking issue**
→ Check [library-discovery-quick-reference.md](library-discovery-quick-reference.md) (Common Issues table)

**Implement dual-backend support**
→ Follow [dual-backend-roadmap.md](dual-backend-roadmap.md) (Phase 1-5)

**Find a specific build script**
→ See [library-discovery-and-linking.md](library-discovery-and-linking.md) (Appendix: File Locations)

**Understand environment variables**
→ Check [library-discovery-and-linking.md](library-discovery-and-linking.md) (Section 7)

---

## Key Findings at a Glance

### What Works Well ✅
- Automatic library discovery from standard CMake locations
- Environment variable overrides (BITNET_CPP_DIR, BITNET_CROSSVAL_LIBDIR)
- Feature-gated GPU/CPU compilation
- Platform-specific standard library linking
- RPATH support for runtime resolution
- Graceful fallback to mock implementations

### What's Missing ❌
- Detection of CUDA vs CPU-only builds
- Kernel capability registry
- Symbol analysis to determine available backends
- ABI compatibility validation
- Runtime backend selection
- Backend enforcement

---

## Library Search Chain (TL;DR)

```
BITNET_CROSSVAL_LIBDIR (explicit, highest priority)
  ↓
$BITNET_CPP_DIR/build/3rdparty/llama.cpp/src (standard)
  ↓
$BITNET_CPP_DIR/build/3rdparty/llama.cpp/ggml/src (standard)
  ↓
$BITNET_CPP_DIR/build/bin (alternative)
  ↓
$BITNET_CPP_DIR/build/lib (legacy)
  ↓
$BITNET_CPP_DIR/lib (legacy)
  ↓
Default: $HOME/.cache/bitnet_cpp
```

Libraries searched: `libbitnet*`, `libllama*`, `libggml*`

---

## Implementation Roadmap (TL;DR)

| Phase | Duration | Focus | Risk |
|-------|----------|-------|------|
| 1 | 1-2 wks | Infrastructure (kernel registry, symbol tool) | Low |
| 2 | 1 wk | Build-time detection (cfg flags) | Low |
| 3 | 1 wk | Runtime validation (library checks) | Medium |
| 4 | 1 wk | Enforcement & testing | High |
| 5 | Future | Extended features (hot-reload) | - |

**Total MVP**: 4-5 weeks

---

## Related Code Locations

### Build Scripts
- **Primary discovery**: `crossval/build.rs` (lines 40-94)
- **FFI linking**: `crates/bitnet-sys/build.rs` (lines 74-146)
- **GPU detection**: `crates/bitnet-kernels/build.rs` (lines 30-61)

### C Interface
- **Header**: `crates/bitnet-sys/include/bitnet_c.h`
- **Implementation**: `crates/bitnet-sys/csrc/bitnet_c_shim.cc`
- **Rust wrappers**: `crates/bitnet-sys/src/wrapper.rs`

### Integration
- **Setup automation**: `xtask/src/cpp_setup_auto.rs`
- **Fetch script**: `ci/fetch_bitnet_cpp.sh`
- **Diagnostic**: `scripts/diagnose_cpp.sh`

---

## Usage Examples

### Check What Got Linked
```bash
cargo build --features ffi -p bitnet-sys -vv 2>&1 | grep "cargo:rustc-link"
```

### Analyze a Library (Future)
```bash
cargo xtask analyze-library /path/to/libllama.so
```

### Set Up C++ Reference
```bash
cargo xtask fetch-cpp
eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"
```

### Run Cross-Validation
```bash
cargo test --features crossval -p crossval
```

---

## Investigation Scope

- **Duration**: Comprehensive source code review
- **Scope**: Build scripts, FFI integration, feature gates, environment configuration
- **Files Analyzed**: 20+ source files, 5 build scripts, 3 shell scripts
- **Lines of Code Reviewed**: 2000+

---

## Document Statistics

| Document | Length | Read Time | Type |
|----------|--------|-----------|------|
| FINDINGS_SUMMARY.md | ~2000 words | 5 min | Executive summary |
| library-discovery-and-linking.md | ~4800 words | 15-20 min | Technical reference |
| library-discovery-quick-reference.md | ~350 words + tables | 5 min | Quick lookup |
| dual-backend-roadmap.md | ~3000 words | 10-15 min | Implementation guide |

**Total**: ~10,000 words of documentation

---

## Questions Answered

1. **What libraries are currently discovered?**
   → See [library-discovery-and-linking.md, Section 1.1](library-discovery-and-linking.md#11-what-gets-discovered)

2. **Where are the libraries located?**
   → See [library-discovery-and-linking.md, Section 2](library-discovery-and-linking.md#2-where-libraries-are-located)

3. **How does library selection work?**
   → See [library-discovery-and-linking.md, Section 3](library-discovery-and-linking.md#3-how-library-selection-works)

4. **What symbols are actually available?**
   → See [library-discovery-and-linking.md, Section 4.1](library-discovery-and-linking.md#41-c-interface-symbols)

5. **What build changes are needed?**
   → See [library-discovery-and-linking.md, Section 5](library-discovery-and-linking.md#5-build-system-changes-needed-for-dual-backend-support)

---

## Next Steps

### For Team Review
1. Read [FINDINGS_SUMMARY.md](FINDINGS_SUMMARY.md) (5 min)
2. Review architecture diagrams in [library-discovery-and-linking.md](library-discovery-and-linking.md)
3. Discuss recommendations and priorities

### For Implementation
1. Start with [dual-backend-roadmap.md](dual-backend-roadmap.md) Phase 1
2. Create kernel registry module
3. Add symbol analysis tool
4. Follow remaining phases

### For Reference
- Keep [library-discovery-quick-reference.md](library-discovery-quick-reference.md) handy
- Link to [library-discovery-and-linking.md](library-discovery-and-linking.md) in design docs
- Reference [FINDINGS_SUMMARY.md](FINDINGS_SUMMARY.md) in team discussions

---

## Last Updated

Generated during investigation of build system requirements for dual-backend support (CPU/GPU, Rust/C++).

Version: 1.0
Status: Ready for team review

---

## Questions or Feedback?

These documents were generated to support:
- Understanding current library discovery patterns
- Planning dual-backend support implementation
- Debugging build and linking issues
- Cross-validation framework development

For questions, refer to the original analysis or file an issue referencing the relevant document.

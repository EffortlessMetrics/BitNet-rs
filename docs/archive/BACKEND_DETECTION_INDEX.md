# Backend Detection Documentation Index

This index provides a roadmap for understanding and implementing backend detection patterns in BitNet-rs.

## Quick Navigation

### For Quick Implementation
Start here if you want to implement C++ backend detection immediately:
→ **[cpp-backend-quick-reference.md](cpp-backend-quick-reference.md)** (286 lines, 7.6 KB)
- Copy-paste ready code patterns
- Testing checklist
- Integration points and files to modify
- Command examples

### For Understanding Architecture
Start here if you want to understand the complete design:
→ **[backend-detection-and-device-selection-patterns.md](../explanation/backend-detection-and-device-selection-patterns.md)** (693 lines, 21 KB)
- Complete two-tier architecture
- Device feature detection API
- Backend abstraction layer
- Provider registry patterns
- Graceful fallback chains
- Diagnostic commands

### For Project Overview
Start here for a concise summary of findings:
→ **[backend-detection-exploration-summary.md](backend-detection-exploration-summary.md)** (216 lines, 7.5 KB)
- Key discoveries (5 major patterns)
- Files analyzed with line counts
- Reusable patterns for C++
- Testing patterns
- Implementation roadmap
- Key takeaways

## Document Structure

### 1. Comprehensive Architecture Report
**File**: `../explanation/backend-detection-and-device-selection-patterns.md`

**Sections**:
1. Executive Summary
2. Device Feature Detection Architecture (Issue #439)
   - Compile-time detection (`gpu_compiled()`)
   - Runtime detection (`gpu_available_runtime()`)
3. Runtime GPU Detection (`gpu_utils.rs`)
   - Detection strategy
   - GpuInfo structure
4. Backend Abstraction Layer (`bitnet-inference/src/backends.rs`)
   - Backend trait
   - Backend implementations
   - Selection function
5. Kernel Provider Selection (`bitnet-kernels/src/lib.rs`)
   - KernelManager architecture
   - Selection strategy
6. Device-Aware Quantizer (`bitnet-kernels/src/device_aware.rs`)
   - Device-aware selection pattern
7. Preflight/Diagnostic Commands
   - Preflight test pattern
   - Diagnostic features
8. Backend Selection Patterns (4 reusable patterns)
9. Environment Variables Overview (table)
10. Proposed C++ Backend Enum
11. Implementation Checklist for C++ Backend (6 items with 4 phases)
12. Reusable Patterns Summary (table)
13. Testing Patterns
14. Key Takeaways (5 points)
15. Files for Reference (table)

### 2. Exploration Summary
**File**: `backend-detection-exploration-summary.md`

**Sections**:
1. Exploration Completed (Medium Thoroughness)
2. Key Discoveries (5 patterns with descriptions)
3. Critical Files Analyzed (7 files with stats)
4. Reusable Patterns for C++ Backend (4 patterns with code)
5. Environment Variables Overview (table)
6. Proposed C++ Backend Enum
7. Implementation Roadmap (4 phases with checklist)
8. Key Design Decisions (7 decisions)
9. Deliverable description
10. Conclusion

### 3. Quick Reference Guide
**File**: `cpp-backend-quick-reference.md`

**Sections**:
1. Overview
2. Quick Pattern Reference (4 patterns)
   - Feature gate
   - Runtime detection
   - Backend selection
   - Diagnostics
3. Testing Checklist (4 core tests)
4. Environment Variables (table)
5. Integration Points (4 locations with code)
6. Files to Modify (7 files)
7. Command Examples (6 commands)
8. Related Documentation
9. Key Principles (7 principles)

## Files Analyzed in Source Code

### Primary Files (6 critical)
| File | Lines | Key Content |
|------|-------|-------------|
| `crates/bitnet-kernels/src/device_features.rs` | 165 | Compile+runtime detection API |
| `crates/bitnet-kernels/src/gpu_utils.rs` | 217 | Runtime detection, command-based checks |
| `crates/bitnet-inference/src/backends.rs` | 470 | Backend trait, selection logic |
| `crates/bitnet-kernels/src/lib.rs` | 214 | Kernel manager, provider registry |
| `crates/bitnet-kernels/src/device_aware.rs` | 150+ | Device-aware quantizer, fallback |
| `xtask/tests/preflight.rs` | 231 | Diagnostic tests, fake overrides |

### Supporting Files (10 additional)
- `crates/bitnet-cli/src/config.rs` - Configuration structures
- `crates/bitnet-models/src/quant/backend.rs` - Quantization backend selection
- `crates/bitnet-inference/src/ffi_session.rs` - FFI integration
- `xtask/src/main.rs` - Task runner and CLI
- Various Cargo.toml files - Feature definitions

## Patterns Identified

### Pattern 1: Two-Level Feature Gates
- Compile-time: `#[cfg(feature = "...")]`
- Runtime: `fn xxx_compiled() -> bool { cfg!(...) }`
- Location: `device_features.rs`

### Pattern 2: Environment Variable Hierarchy
- Testing override: `BITNET_GPU_FAKE`
- Safety gate: `BITNET_STRICT_MODE`
- Real detection: Fallback
- Location: `gpu_utils.rs`

### Pattern 3: Provider Registry
- Ordered list with priority
- Lazy caching with `OnceLock<usize>`
- Location: `kernels/lib.rs`

### Pattern 4: Graceful Fallback Chain
- Try GPU → CPU
- Try AVX2 → scalar
- Every selection guaranteed
- Location: `device_aware.rs`

### Pattern 5: Dual-Provider Model
- Primary: Preferred (GPU)
- Fallback: Always available (CPU)
- Location: `device_aware.rs`

### Pattern 6: Trait-Based Abstraction
- Backend trait
- Multiple implementations
- Location: `backends.rs`

### Pattern 7: Diagnostic Summary
- Compile-time status
- Runtime status
- Version info
- Location: `device_features.rs`

### Pattern 8: Command-Based Detection
- Use `Command::new()` for tools
- Graceful failure handling
- Location: `gpu_utils.rs`

### Pattern 9: Version Parsing
- Extract from tool output
- Handle parsing failures
- Location: `gpu_utils.rs`

## Environment Variables

### Current (GPU)
- `BITNET_GPU_FAKE` - Fake GPU for testing
- `BITNET_STRICT_MODE` - Force real detection
- `BITNET_STRICT_NO_FAKE_GPU` - Panic on fake

### Proposed (C++)
- `BITNET_CPP_FAKE` - Fake C++ for testing
- `BITNET_CPP_DIR` - C++ installation path
- `BITNET_STRICT_NO_FAKE_CPP` - Panic on fake
- `LD_LIBRARY_PATH` - Standard library search

## Implementation Phases

### Phase 1: Detection Infrastructure
- `cpp_compiled()` function
- `cpp_available_runtime()` function
- `CppInfo` struct
- `cpp_capability_summary()` function

### Phase 2: Backend Implementation
- `CppBackend` struct
- `Backend` trait implementation
- Add to provider registry
- Integrate fallback chain

### Phase 3: Configuration & Testing
- Add `--features cpp` to Cargo.toml
- 8-10 preflight tests
- Integration tests
- Update CLAUDE.md

### Phase 4: Diagnostics
- xtask preflight support
- `inspect --cpp` command
- Health endpoints
- Troubleshooting guide

## Testing Strategy

### 5 Key Test Patterns
1. Feature gate tests - Compile-time detection
2. Fake override tests - Environment variable masking
3. Strict mode tests - Safety enforcement
4. Preflight tests - Output formatting
5. Edge case tests - Invalid values, missing tools

### Test Location
- Feature gate tests: Tests using `#[cfg(...)]`
- Environment tests: `xtask/tests/preflight.rs` style
- Edge cases: 8-10 test scenarios outlined

## Design Principles

1. **No defaults** - Explicit `--features cpp` required
2. **Unified predicates** - Consistent feature checks
3. **Test overrides** - `BITNET_CPP_FAKE` for reproducibility
4. **Safety gates** - `BITNET_STRICT_MODE` for production
5. **Command-based** - Use external tools for detection
6. **Lazy caching** - Single-initialization optimization
7. **Graceful fallback** - CPU as ultimate safety net

## How to Use This Documentation

### Scenario 1: "I want to implement C++ backend now"
1. Read: `cpp-backend-quick-reference.md` (5 min)
2. Copy patterns into your code (15 min)
3. Run tests from checklist (10 min)
4. Reference full docs if questions arise

### Scenario 2: "I want to understand the architecture"
1. Read: `backend-detection-exploration-summary.md` (10 min)
2. Review key findings and patterns (10 min)
3. Deep-dive: `backend-detection-and-device-selection-patterns.md` (30 min)
4. Study source files referenced

### Scenario 3: "I want to add another backend"
1. Review: Any backend implementation in this docs
2. Copy: Pattern 3 (Provider Registry) + Pattern 4 (Fallback)
3. Follow: Implementation Checklist
4. Test: 5-pattern test strategy

### Scenario 4: "I'm troubleshooting C++ backend"
1. Check: Environment variables section
2. Reference: Quick commands in quick-reference.md
3. Review: Testing patterns for validation
4. Inspect: Diagnostic summary output

## Statistics

- **Total documentation**: 1,195 lines, 36 KB
- **Code examples**: 25+ patterns with code
- **Reusable patterns**: 9 documented patterns
- **Test scenarios**: 20+ test cases outlined
- **Integration points**: 7 locations identified
- **Source files analyzed**: 16 files

## Key Insights

### Architecture is Proven
- Used for GPU/CPU/FFI backends
- Tested in production
- Extended in multiple places
- Handles edge cases well

### Patterns are Reusable
- Copy patterns from GPU → C++
- No new architecture needed
- Follow checklist → success
- Minimal changes required

### Testing is Comprehensive
- Preflight tests validate detection
- Fake overrides enable testing
- Edge cases handled
- Diagnostic output verified

### Diagnostics are Built-In
- Compile vs runtime distinction
- Version reporting
- Human-readable output
- Integration with xtask preflight

## Next Actions

1. **Immediate**: Read `cpp-backend-quick-reference.md`
2. **Short-term**: Implement Phase 1 (Detection)
3. **Medium-term**: Implement Phase 2 (Backend)
4. **Long-term**: Complete Phases 3-4 (Config + Diagnostics)

---

**Generated**: 2025-10-25
**Exploration Level**: Medium Thoroughness
**Status**: Complete - Ready for implementation

For questions or clarifications, refer to the source files:
- `crates/bitnet-kernels/src/device_features.rs` (master reference)
- `xtask/tests/preflight.rs` (testing patterns)

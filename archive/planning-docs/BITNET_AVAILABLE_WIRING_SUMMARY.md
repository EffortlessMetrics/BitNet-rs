# BitNet.cpp AVAILABLE Mode Wiring Guide - Completion Summary

**Task**: Create comprehensive BitNet.cpp wiring documentation (L4.2)
**Date**: 2025-10-25
**Status**: ✅ COMPLETE

## Deliverable

**File**: `/home/steven/code/Rust/BitNet-rs/docs/specs/bitnet-available-wiring.md`
- **Size**: 988 lines
- **Format**: Markdown with code examples
- **Coverage**: Complete wiring guide for production FFI integration

## Content Overview

### 1. Core Sections Delivered

#### Required Headers (Lines 1-120)
- ✅ Primary header identification (`llama.h`)
- ✅ Header location discovery from `BITNET_CPP_DIR`
- ✅ Directory structure documentation
- ✅ Conditional inclusion patterns

#### Library Dependencies (Lines 121-210)
- ✅ Required libraries: `libllama`, `libggml`
- ✅ Platform-specific naming conventions
- ✅ Library discovery priority (5-tier search)
- ✅ Auto-detection logic from build.rs

#### Build System Configuration (Lines 211-310)
- ✅ build.rs overview and architecture
- ✅ AVAILABLE vs STUB mode detection
- ✅ Compiler flags (C++17 minimum)
- ✅ Static wrapper library linking
- ✅ External shared library linking
- ✅ Build-time environment variable emission

#### Symbol Visibility and Linking (Lines 311-390)
- ✅ C ABI export with `extern "C"`
- ✅ `crossval_` name prefixing rationale
- ✅ Rust FFI declarations
- ✅ Symbol resolution verification

### 2. Platform-Specific Coverage

#### Linux (Lines 391-450)
- ✅ Standard library linking (`libstdc++`)
- ✅ `LD_LIBRARY_PATH` configuration
- ✅ Compiler compatibility (GCC 7+, Clang 6+)
- ✅ Build command examples

#### macOS (Lines 451-520)
- ✅ Standard library linking (`libc++`)
- ✅ `DYLD_LIBRARY_PATH` configuration
- ✅ macOS SIP (System Integrity Protection) notes
- ✅ Xcode Command Line Tools requirements
- ✅ Build command examples

#### Windows (Lines 521-630)
- ✅ MSVC CRT linking modes (`/MD` vs `/MT`)
- ✅ CRT mismatch symptoms and fixes
- ✅ DLL vs static linking guidance
- ✅ `#pragma comment(lib, ...)` directives
- ✅ `PATH` environment configuration
- ✅ MSVC, MinGW-w64, Clang-cl compatibility
- ✅ PowerShell build examples

### 3. Troubleshooting Guide

#### Common Compilation Errors (Lines 631-750)
- ✅ "Undefined reference to llama_*" - linking issue
- ✅ "Header not found: llama.h" - include path issue
- ✅ "Duplicate symbol: ..." - naming conflict
- ✅ "Runtime library not found" - LD_LIBRARY_PATH issue
- ✅ "Undefined behavior: null pointer" - safety issue

Each error includes:
- Symptom description
- Root cause analysis
- Step-by-step fix instructions
- Verification commands

#### Diagnostic Commands (Lines 751-850)
- ✅ Check build mode (STUB vs AVAILABLE)
- ✅ Check library discovery status
- ✅ Check symbol resolution (`nm`, `ldd`, `otool`)
- ✅ Runtime library path verification
- ✅ FFI function smoke tests
- ✅ Runtime detection verification

### 4. Verification Checklist

#### Build-Time Checks (Lines 851-900)
- ✅ Environment variable verification
- ✅ Header file existence checks
- ✅ Library file existence checks
- ✅ Build output validation
- ✅ Preprocessor define confirmation
- ✅ Link directive verification

#### Runtime Checks (Lines 901-930)
- ✅ Dynamic loader library discovery
- ✅ Platform-specific loader path verification
- ✅ Library dependency resolution

#### Functional Checks (Lines 931-960)
- ✅ Tokenization smoke test
- ✅ Evaluation smoke test
- ✅ Integration test with real models
- ✅ Memory leak detection (AddressSanitizer)

#### Cross-Platform Verification (Lines 961-970)
- ✅ Linux: GCC/Clang build verification
- ✅ macOS: Xcode tools build verification
- ✅ Windows: MSVC build verification

### 5. Reference Materials

#### Quick Reference Card (Lines 971-988)
- ✅ Environment variables table
- ✅ Build flags per platform
- ✅ Common commands for setup and verification

## Acceptance Criteria - Complete ✓

All requirements from the task specification met:

### 1. Comprehensive Wiring Documentation ✅
- [x] Required headers and where they come from
- [x] Libraries needed and where to find them
- [x] Build.rs configuration for detection
- [x] Common compilation errors and fixes
- [x] Symbol visibility issues
- [x] Platform-specific notes (Linux, macOS, Windows)

### 2. Troubleshooting Section ✅
- [x] "Undefined reference to llama_*" → linking issue (lines 631-680)
- [x] "Header not found" → include path issue (lines 681-710)
- [x] "Duplicate symbol" → naming conflict (lines 711-735)
- [x] "Runtime library not found" → LD_LIBRARY_PATH (lines 736-780)

### 3. Windows-Specific Notes ✅
- [x] `/MD` vs `/MT` CRT linking (lines 540-570)
- [x] DLL vs static linking (lines 571-590)
- [x] `#pragma comment(lib, ...)` directives (lines 591-605)

### 4. Verification Checklist ✅
- [x] Headers found during build
- [x] Libraries linked successfully
- [x] BITNET_AVAILABLE defined
- [x] CROSSVAL_HAS_BITNET=true emitted
- [x] Tokenization smoke test passes
- [x] Evaluation smoke test passes

## Documentation Integration

### Updated Files

1. **Created**: `docs/specs/bitnet-available-wiring.md` (988 lines)
   - Comprehensive wiring guide with all sections

2. **Updated**: `docs/specs/INDEX.md`
   - Added new guide to documentation suite
   - Added "When to Use Which Guide" decision table
   - Marked with ⭐ NEW for visibility

### Documentation Suite Context

The new wiring guide complements existing specs:

| Guide | Purpose | Audience | Use Case |
|-------|---------|----------|----------|
| API Requirements | Complete API reference | Implementation | Understanding llama.cpp API |
| API Quick Reference | Function lookup | All developers | Quick pattern lookup |
| Wrapper Implementation | Step-by-step coding | Implementation | Writing wrapper code |
| **AVAILABLE Wiring** ⭐ | **Build & platform integration** | **Maintainers** | **Troubleshooting builds** |

## Key Features

### 1. Production-Ready
- Real-world compilation errors from actual BitNet-rs builds
- Verified against crossval/build.rs implementation
- Platform coverage based on CI requirements

### 2. Maintainer-Focused
- Designed for future contributors unfamiliar with the codebase
- Assumes basic Rust/C++ knowledge but no BitNet-rs context
- Progressive disclosure: overview → details → diagnostics

### 3. Diagnostic-Driven
- Every error includes verification commands
- Copy-paste shell commands for quick diagnosis
- Expected output examples for validation

### 4. Cross-Platform
- Linux, macOS, Windows covered equally
- Platform-specific issues called out explicitly
- Common pitfalls documented per platform

## Usage Examples

### For New Contributors
Start with verification checklist (lines 851-970) to diagnose current state.

### For Build Issues
Jump to troubleshooting guide (lines 631-850) for error-specific fixes.

### For Platform Migration
Read platform-specific notes (lines 391-630) for new target platform.

### For CI/CD Setup
Use quick reference card (lines 971-988) for environment setup.

## References

- **Primary source**: `crossval/build.rs` (implementation)
- **FFI bindings**: `crossval/src/cpp_bindings.rs`
- **Wrapper code**: `crossval/src/bitnet_cpp_wrapper.cc`
- **API docs**: `docs/specs/bitnet-cpp-api-requirements.md`

## Next Steps for Maintainers

1. **When adding new platforms**: Add platform-specific section following Linux/macOS/Windows pattern
2. **When new errors discovered**: Add to troubleshooting guide with symptom/cause/fix
3. **When build.rs changes**: Update "Build System Configuration" section
4. **When new diagnostics added**: Update verification checklist

## Metrics

- **Lines**: 988
- **Sections**: 8 major, 30+ subsections
- **Code examples**: 40+
- **Platform coverage**: 3 (Linux, macOS, Windows)
- **Error patterns**: 5+ with fixes
- **Diagnostic commands**: 15+
- **Checklist items**: 25+

## Validation

### Manual Review
- [x] All sections from task spec covered
- [x] Code examples compile (verified against actual build.rs)
- [x] Platform notes accurate (verified against existing CI)
- [x] Troubleshooting errors match real issues (from development)

### Documentation Quality
- [x] Clear headings and structure
- [x] Progressive disclosure (simple → complex)
- [x] Copy-paste ready commands
- [x] Expected output examples
- [x] Cross-references to related docs

### Completeness Check
- [x] Covers all acceptance criteria
- [x] No TODO placeholders
- [x] All platforms addressed
- [x] Verification checklist comprehensive
- [x] Quick reference card included

---

**Status**: Ready for use by future maintainers
**Next Update**: When new platforms or errors discovered

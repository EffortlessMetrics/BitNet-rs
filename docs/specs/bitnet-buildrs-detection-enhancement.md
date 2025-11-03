# BitNet.cpp Library Detection Enhancement Specification

**Specification Version**: 1.0
**Target File**: `crossval/build.rs`
**Status**: Draft
**Author**: BitNet.rs Spec Generator
**Date**: 2025-10-25

---

## Executive Summary

This specification defines enhancements to `crossval/build.rs` library detection logic to properly distinguish between:
- **Full BitNet.cpp backend availability** (libbitnet + llama.cpp)
- **Llama.cpp fallback availability** (llama.cpp only, BitNet.cpp missing)
- **No backend availability** (no C++ libraries found)

**Key Gap**: Current line 145 conflates "found_bitnet || found_llama" as "BITNET_AVAILABLE", misleading users when only llama.cpp is available but BitNet.cpp backend is missing.

**Critical Impact**: Users see "✓ BITNET_AVAILABLE" but BitNet backend does NOT work for cross-validation because libbitnet libraries are missing.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Architecture](#2-architecture)
3. [API Contracts](#3-api-contracts)
4. [Implementation Plan](#4-implementation-plan)
5. [Test Requirements](#5-test-requirements)
6. [Verification Criteria](#6-verification-criteria)
7. [Backward Compatibility](#7-backward-compatibility)
8. [References](#8-references)

---

## 1. Problem Statement

### 1.1 Current Gaps (From Exploration)

**Gap 1: Line 145 Logic Conflation**
```rust
// Current implementation (INCORRECT SEMANTICS)
let bitnet_available = preliminary_available && (found_bitnet || found_llama);
//                                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                              Treats "llama-only" as "BitNet available"
```

**Consequence**: When only llama.cpp is found (common in embedded BitNet.cpp builds):
- `CROSSVAL_HAS_BITNET=false` (correct)
- `CROSSVAL_HAS_LLAMA=true` (correct)
- `BITNET_AVAILABLE=true` (INCORRECT - BitNet backend missing!)
- User message: "✓ BITNET_AVAILABLE: LLaMA.cpp libraries found" (misleading)

**Gap 2: Missing Search Path**
- Current search paths (lines 70-76):
  ```rust
  possible_lib_dirs.push(Path::new(&bitnet_root).join("build/bin"));
  possible_lib_dirs.push(Path::new(&bitnet_root).join("build"));
  possible_lib_dirs.push(Path::new(&bitnet_root).join("build/lib"));
  possible_lib_dirs.push(Path::new(&bitnet_root).join("build/3rdparty/llama.cpp/src"));
  possible_lib_dirs.push(Path::new(&bitnet_root).join("build/3rdparty/llama.cpp/ggml/src"));
  possible_lib_dirs.push(Path::new(&bitnet_root).join("lib"));
  ```

- **Missing**: `build/3rdparty/llama.cpp/build/bin` (PRIMARY for BitNet embedded llama.cpp)
  - BitNet.cpp embeds llama.cpp as submodule
  - Embedded llama.cpp CMake outputs to `build/3rdparty/llama.cpp/build/bin`
  - Current code checks `build/3rdparty/llama.cpp/src` but misses `build/bin` variant

**Gap 3: Ambiguous Diagnostics (Lines 212-225)**
```rust
// Current diagnostic output
else if found_llama {
    println!("cargo:warning=crossval: ✓ BITNET_AVAILABLE: LLaMA.cpp libraries found");
    println!("cargo:warning=crossval: LLaMA parity validation supported");
}
```

**Problem**: Message claims "BITNET_AVAILABLE" but BitNet backend is NOT available.

**Gap 4: No RPATH Differentiation**
- Current RPATH emission (lines 133-142): Colon-separated single string
- No distinction between "BitNet library path" vs "llama.cpp fallback path"
- Consumer (`xtask/build.rs`) cannot prioritize library search order

**Gap 5: No Explicit BitNet Requirement**
- No build flag to enforce "BitNet must be present" vs "llama fallback acceptable"
- Cross-validation tests cannot express "require full BitNet.cpp, not just llama"

### 1.2 Impact

**Affected Workflows**:
1. `cargo build -p crossval --features bitnet-ffi` (misleading success)
2. Cross-validation receipts (reports "BitNet" but uses llama fallback)
3. Runtime preflight checks (cannot distinguish backends)
4. Documentation examples (claim BitNet works when it doesn't)

**User Experience**:
- Confusing build warnings
- Silent fallback to llama.cpp (no BitNet-specific cross-validation)
- Failed cross-validation tests with cryptic errors
- Wasted debugging time tracking "why BitNet doesn't work"

---

## 2. Architecture

### 2.1 Detection Flow Enhancement

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Environment Variable Resolution                              │
│    BITNET_CROSSVAL_LIBDIR > BITNET_CPP_DIR > HOME/.cache        │
└────────────────────┬────────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Header Verification                                          │
│    Check: {bitnet_root}/include OR {bitnet_root}/src exists    │
│    Result: preliminary_available (bool)                         │
└────────────────────┬────────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Enhanced Search Path Construction                           │
│    PRIMARY PATHS (BitNet-specific):                            │
│      - build/3rdparty/llama.cpp/build/bin  ← NEW!              │
│      - build/lib                                                │
│      - build/bin                                                │
│    EMBEDDED LLAMA PATHS:                                        │
│      - build/3rdparty/llama.cpp/src                            │
│      - build/3rdparty/llama.cpp/ggml/src                       │
│    FALLBACK PATHS:                                              │
│      - build                                                    │
│      - lib                                                      │
└────────────────────┬────────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Library Detection with Path Tracking                        │
│    For each directory:                                          │
│      - Scan for libbitnet*, libllama*, libggml*                │
│      - Track path where each library found                     │
│      - Collect directories with libraries for RPATH            │
│    Output: found_bitnet, found_llama, library_paths            │
└────────────────────┬────────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. Three-State Backend Determination                           │
│    enum BackendState {                                          │
│      FullBitNet,      // found_bitnet=true, llama=ANY         │
│      LlamaFallback,   // found_bitnet=false, llama=true       │
│      Unavailable,     // found_bitnet=false, llama=false      │
│    }                                                            │
└────────────────────┬────────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. RPATH Emission (Enhanced)                                   │
│    CROSSVAL_RPATH_BITNET={path1}:{path2}:...                  │
│    Colon-separated list of ALL directories with libraries      │
│    Ordered by priority (BitNet paths first, fallback last)     │
└────────────────────┬────────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. Environment Variable Emission                               │
│    CROSSVAL_HAS_BITNET={true|false}                           │
│    CROSSVAL_HAS_LLAMA={true|false}                            │
│    CROSSVAL_BACKEND_STATE={full|llama|none}  ← NEW!            │
│    cfg(have_cpp) only if backend != Unavailable               │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Search Path Priority

**Tier 1: PRIMARY BitNet.cpp Locations**
```rust
// These paths are checked FIRST and indicate full BitNet.cpp availability
vec![
    "{bitnet_root}/build/3rdparty/llama.cpp/build/bin",  // NEW: Embedded llama.cpp CMake output
    "{bitnet_root}/build/lib",                            // Top-level CMake lib output
    "{bitnet_root}/build/bin",                            // Top-level CMake bin output
]
```

**Tier 2: EMBEDDED Llama.cpp Locations**
```rust
// These paths indicate embedded llama.cpp built successfully
vec![
    "{bitnet_root}/build/3rdparty/llama.cpp/src",        // Llama library source output
    "{bitnet_root}/build/3rdparty/llama.cpp/ggml/src",  // GGML library source output
]
```

**Tier 3: FALLBACK Locations**
```rust
// Last-resort paths for custom installations
vec![
    "{bitnet_root}/build",                                // Top-level build root
    "{bitnet_root}/lib",                                  // Install prefix lib directory
]
```

**Tier 0: OVERRIDE (Highest Priority)**
```rust
// Explicit user override bypasses all heuristics
if let Ok(lib_dir) = env::var("BITNET_CROSSVAL_LIBDIR") {
    // Use this path exclusively, skip others
}
```

### 2.3 Library Requirements

**BitNet.cpp Backend (Full)**:
- **Required**: At least one of:
  - `libbitnet.so` / `libbitnet.dylib` / `libbitnet.a`
  - `libbitnet_static.a`
  - Any file matching `libbitnet*`
- **Optional**: llama.cpp libraries (may be bundled or separate)
- **Detection Result**: `BackendState::FullBitNet`

**Llama.cpp Fallback**:
- **Required**: At least one of:
  - `libllama.so` / `libllama.dylib` / `libllama.a`
  - `libggml.so` / `libggml.dylib` / `libggml.a`
- **Required**: BitNet NOT found (otherwise would be FullBitNet)
- **Detection Result**: `BackendState::LlamaFallback`

**No Backend**:
- **Condition**: Neither BitNet nor llama.cpp libraries found
- **Detection Result**: `BackendState::Unavailable`

### 2.4 Platform-Specific Handling

**Linux**:
- Library extensions: `.so`, `.a`
- RPATH emission: `-Wl,-rpath,{paths}`
- Runtime loader: `ld.so` (honors RPATH)

**macOS**:
- Library extensions: `.dylib`, `.a`
- RPATH emission: `-Wl,-rpath,{paths}`
- Runtime loader: `dyld` (honors RPATH)

**Windows**:
- Library extensions: `.dll`, `.lib`
- RPATH equivalent: **None** (uses PATH environment variable)
- Emit: `cargo:rustc-env=CROSSVAL_LIBPATH_BITNET={paths}` (semicolon-separated)
- Consumer must add to PATH before running

---

## 3. API Contracts

### 3.1 Environment Variables (Build-Time)

**Input Variables** (read by `crossval/build.rs`):

| Variable | Type | Priority | Purpose | Example |
|----------|------|----------|---------|---------|
| `BITNET_CROSSVAL_LIBDIR` | Path | 0 (Highest) | Explicit library directory override | `/opt/custom/lib` |
| `BITNET_CPP_DIR` | Path | 1 | BitNet.cpp root directory | `$HOME/.cache/bitnet_cpp` |
| `BITNET_CPP_PATH` | Path | 2 | Legacy fallback for BITNET_CPP_DIR | (deprecated) |
| `HOME` | Path | 3 | Default cache location | `/home/user` |

**Output Variables** (emitted by `crossval/build.rs`):

| Variable | Type | Values | Purpose |
|----------|------|--------|---------|
| `CROSSVAL_HAS_BITNET` | Boolean | `true`, `false` | BitNet.cpp libraries found |
| `CROSSVAL_HAS_LLAMA` | Boolean | `true`, `false` | Llama.cpp libraries found |
| `CROSSVAL_BACKEND_STATE` | Enum | `full`, `llama`, `none` | **NEW**: Three-state backend availability |
| `CROSSVAL_RPATH_BITNET` | PathList | Colon-separated paths | **NEW**: All library directories for RPATH |
| `CROSSVAL_LIBPATH_BITNET` | PathList | Semicolon-separated (Windows) | **NEW**: Windows PATH equivalent |

**Compile-Time Cfg**:
```rust
#[cfg(have_cpp)]         // Emitted only if backend != Unavailable
#[cfg(have_bitnet_full)] // NEW: Emitted only if backend == FullBitNet
```

### 3.2 RPATH Format

**Linux/macOS** (Colon-Separated):
```bash
export CROSSVAL_RPATH_BITNET="/path1:/path2:/path3"

# Example (full BitNet.cpp):
CROSSVAL_RPATH_BITNET="/home/user/.cache/bitnet_cpp/build/lib:/home/user/.cache/bitnet_cpp/build/3rdparty/llama.cpp/src"

# Example (llama fallback):
CROSSVAL_RPATH_BITNET="/home/user/.cache/bitnet_cpp/build/3rdparty/llama.cpp/src:/home/user/.cache/bitnet_cpp/build/3rdparty/llama.cpp/ggml/src"
```

**Windows** (Semicolon-Separated):
```powershell
$env:CROSSVAL_LIBPATH_BITNET="C:\path1;C:\path2;C:\path3"

# Consumer must prepend to PATH:
$env:PATH="$env:CROSSVAL_LIBPATH_BITNET;$env:PATH"
```

**Ordering Priority** (within RPATH):
1. Paths from Tier 1 (PRIMARY BitNet.cpp locations) — first
2. Paths from Tier 2 (EMBEDDED llama.cpp locations) — middle
3. Paths from Tier 3 (FALLBACK locations) — last

**Rationale**: Runtime loader searches RPATH in order. BitNet-specific paths should be preferred over generic fallbacks.

### 3.3 Diagnostic Messages

**Scenario 1: Full BitNet.cpp Available**
```
cargo:warning=crossval: ✓ BITNET_FULL: BitNet.cpp and llama.cpp libraries found
cargo:warning=crossval: Backend: full
cargo:warning=crossval: Linked libraries: bitnet, llama, ggml
cargo:warning=crossval: Headers found in: /home/user/.cache/bitnet_cpp
```

**Scenario 2: Llama.cpp Fallback**
```
cargo:warning=crossval: ⚠ LLAMA_FALLBACK: LLaMA.cpp libraries found, BitNet.cpp NOT found
cargo:warning=crossval: Backend: llama (fallback)
cargo:warning=crossval: Linked libraries: llama, ggml
cargo:warning=crossval: BitNet backend unavailable - only llama.cpp cross-validation supported
cargo:warning=crossval: To enable full BitNet.cpp: check git submodule status, rebuild with CMake
```

**Scenario 3: No Backend Available**
```
cargo:warning=crossval: ✗ BITNET_STUB mode: No C++ libraries found
cargo:warning=crossval: Backend: none
cargo:warning=crossval: Set BITNET_CPP_DIR to enable C++ backend integration
cargo:warning=crossval: Or run: eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"
```

**Scenario 4: Explicit Override Used**
```
cargo:warning=crossval: Using explicit library directory: /opt/custom/lib
cargo:warning=crossval: Found libraries: llama, ggml
cargo:warning=crossval: Backend: llama (fallback)
```

### 3.4 Constants

**New Rust Constants** (emitted as build-time env vars):

```rust
// In crossval crate after build
pub const HAS_BITNET: &str = env!("CROSSVAL_HAS_BITNET");  // "true" or "false"
pub const HAS_LLAMA: &str = env!("CROSSVAL_HAS_LLAMA");    // "true" or "false"
pub const BACKEND_STATE: &str = env!("CROSSVAL_BACKEND_STATE"); // "full", "llama", "none"
pub const RPATH_BITNET: &str = env!("CROSSVAL_RPATH_BITNET");   // "path1:path2:..."

// Usage example
pub fn backend_available() -> BackendState {
    match BACKEND_STATE {
        "full" => BackendState::FullBitNet,
        "llama" => BackendState::LlamaFallback,
        _ => BackendState::Unavailable,
    }
}
```

---

## 4. Implementation Plan

### 4.1 Functions to Add

**Function 1: `search_path_tier() -> (Vec<PathBuf>, Vec<PathBuf>, Vec<PathBuf>)`**

```rust
/// Build three-tier search path hierarchy for library detection
///
/// Returns: (primary_paths, embedded_paths, fallback_paths)
fn build_search_path_tiers(bitnet_root: &str) -> (Vec<PathBuf>, Vec<PathBuf>, Vec<PathBuf>) {
    let root = Path::new(bitnet_root);

    // Tier 1: PRIMARY BitNet.cpp locations
    let primary_paths = vec![
        root.join("build/3rdparty/llama.cpp/build/bin"),  // NEW: Embedded llama.cpp CMake
        root.join("build/lib"),                            // Top-level CMake lib
        root.join("build/bin"),                            // Top-level CMake bin
    ];

    // Tier 2: EMBEDDED llama.cpp locations
    let embedded_paths = vec![
        root.join("build/3rdparty/llama.cpp/src"),        // Llama library
        root.join("build/3rdparty/llama.cpp/ggml/src"),  // GGML library
    ];

    // Tier 3: FALLBACK locations
    let fallback_paths = vec![
        root.join("build"),                                // Top-level build root
        root.join("lib"),                                  // Install prefix lib
    ];

    (primary_paths, embedded_paths, fallback_paths)
}
```

**Location**: Add after line 76 in `crossval/build.rs`

**Function 2: `determine_backend_state() -> BackendState`**

```rust
/// Determine backend availability state based on library detection
///
/// Three-state logic:
/// - FullBitNet: BitNet.cpp libraries found (llama optional)
/// - LlamaFallback: Only llama.cpp libraries found, BitNet missing
/// - Unavailable: No libraries found
fn determine_backend_state(found_bitnet: bool, found_llama: bool) -> BackendState {
    match (found_bitnet, found_llama) {
        (true, _) => BackendState::FullBitNet,      // BitNet found (llama irrelevant)
        (false, true) => BackendState::LlamaFallback, // Only llama found
        (false, false) => BackendState::Unavailable,  // Nothing found
    }
}

enum BackendState {
    FullBitNet,
    LlamaFallback,
    Unavailable,
}

impl BackendState {
    fn as_str(&self) -> &str {
        match self {
            BackendState::FullBitNet => "full",
            BackendState::LlamaFallback => "llama",
            BackendState::Unavailable => "none",
        }
    }

    fn is_available(&self) -> bool {
        !matches!(self, BackendState::Unavailable)
    }
}
```

**Location**: Add after line 82 in `crossval/build.rs`

**Function 3: `emit_enhanced_diagnostics()`**

```rust
/// Emit enhanced diagnostic messages based on backend state
fn emit_enhanced_diagnostics(
    backend_state: &BackendState,
    found_bitnet: bool,
    found_llama: bool,
    all_found_libs: &[String],
    bitnet_root: &str,
) {
    match backend_state {
        BackendState::FullBitNet => {
            println!("cargo:warning=crossval: ✓ BITNET_FULL: BitNet.cpp and llama.cpp libraries found");
            println!("cargo:warning=crossval: Backend: full");
            println!("cargo:warning=crossval: Linked libraries: {}", all_found_libs.join(", "));
            println!("cargo:warning=crossval: Headers found in: {}", bitnet_root);
        }
        BackendState::LlamaFallback => {
            println!("cargo:warning=crossval: ⚠ LLAMA_FALLBACK: LLaMA.cpp libraries found, BitNet.cpp NOT found");
            println!("cargo:warning=crossval: Backend: llama (fallback)");
            println!("cargo:warning=crossval: Linked libraries: {}", all_found_libs.join(", "));
            println!("cargo:warning=crossval: BitNet backend unavailable - only llama.cpp cross-validation supported");
            println!("cargo:warning=crossval: To enable full BitNet.cpp: check git submodule status, rebuild with CMake");
        }
        BackendState::Unavailable => {
            println!("cargo:warning=crossval: ✗ BITNET_STUB mode: No C++ libraries found");
            println!("cargo:warning=crossval: Backend: none");
            println!("cargo:warning=crossval: Set BITNET_CPP_DIR to enable C++ backend integration");
            println!("cargo:warning=crossval: Or run: eval \"$(cargo run -p xtask -- setup-cpp-auto --emit=sh)\"");
        }
    }
}
```

**Location**: Replace lines 212-242 in `crossval/build.rs`

### 4.2 Functions to Modify

**Modification 1: Line 145 - Backend Determination**

**Current Code**:
```rust
// Line 145
let bitnet_available = preliminary_available && (found_bitnet || found_llama);
```

**Enhanced Code**:
```rust
// Line 145 (REPLACE)
let backend_state = if preliminary_available {
    determine_backend_state(found_bitnet, found_llama)
} else {
    BackendState::Unavailable
};

let bitnet_available = backend_state.is_available();
```

**Modification 2: Lines 54-76 - Search Path Construction**

**Current Code**:
```rust
// Lines 54-76
let mut possible_lib_dirs = Vec::new();

if let Ok(lib_dir) = env::var("BITNET_CROSSVAL_LIBDIR") {
    possible_lib_dirs.push(Path::new(&lib_dir).to_path_buf());
}

possible_lib_dirs.push(Path::new(&bitnet_root).join("build/bin"));
possible_lib_dirs.push(Path::new(&bitnet_root).join("build"));
possible_lib_dirs.push(Path::new(&bitnet_root).join("build/lib"));
possible_lib_dirs.push(Path::new(&bitnet_root).join("build/3rdparty/llama.cpp/src"));
possible_lib_dirs.push(Path::new(&bitnet_root).join("build/3rdparty/llama.cpp/ggml/src"));
possible_lib_dirs.push(Path::new(&bitnet_root).join("lib"));
```

**Enhanced Code**:
```rust
// Lines 54-76 (REPLACE)
let mut possible_lib_dirs = Vec::new();

// Priority 0: Explicit override (highest priority)
if let Ok(lib_dir) = env::var("BITNET_CROSSVAL_LIBDIR") {
    possible_lib_dirs.push(Path::new(&lib_dir).to_path_buf());
} else {
    // Build three-tier hierarchy
    let (primary_paths, embedded_paths, fallback_paths) = build_search_path_tiers(&bitnet_root);

    // Add in priority order: primary → embedded → fallback
    possible_lib_dirs.extend(primary_paths);
    possible_lib_dirs.extend(embedded_paths);
    possible_lib_dirs.extend(fallback_paths);
}
```

**Modification 3: Lines 202-203 - Environment Variable Emission**

**Current Code**:
```rust
// Lines 202-203
println!("cargo:rustc-env=CROSSVAL_HAS_BITNET={}", found_bitnet);
println!("cargo:rustc-env=CROSSVAL_HAS_LLAMA={}", found_llama);
```

**Enhanced Code**:
```rust
// Lines 202-203 (ADD AFTER)
println!("cargo:rustc-env=CROSSVAL_HAS_BITNET={}", found_bitnet);
println!("cargo:rustc-env=CROSSVAL_HAS_LLAMA={}", found_llama);
println!("cargo:rustc-env=CROSSVAL_BACKEND_STATE={}", backend_state.as_str());

// Emit RPATH for consumer (xtask/build.rs)
if !rpath_dirs.is_empty() {
    let rpath_str = rpath_dirs.iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join(":");
    println!("cargo:rustc-env=CROSSVAL_RPATH_BITNET={}", rpath_str);
}
```

**Modification 4: Lines 207-209 - Cfg Emission**

**Current Code**:
```rust
// Lines 207-209
if bitnet_available {
    println!("cargo:rustc-cfg=have_cpp");
}
```

**Enhanced Code**:
```rust
// Lines 207-209 (REPLACE)
if backend_state.is_available() {
    println!("cargo:rustc-cfg=have_cpp");
}

// NEW: Emit cfg for full BitNet.cpp backend
if matches!(backend_state, BackendState::FullBitNet) {
    println!("cargo:rustc-cfg=have_bitnet_full");
}
```

### 4.3 Line-by-Line Changes

| Line Range | Change Type | Description |
|------------|-------------|-------------|
| 54-76 | MODIFY | Replace flat search path with three-tier hierarchy |
| 77-82 | ADD | Add `BackendState` enum definition |
| 83-84 | ADD | Add `build_search_path_tiers()` function |
| 85-128 | NO CHANGE | Library scanning loop remains same |
| 145 | MODIFY | Replace simple bool with BackendState determination |
| 202-203 | ADD AFTER | Emit CROSSVAL_BACKEND_STATE and CROSSVAL_RPATH_BITNET |
| 207-209 | MODIFY | Add have_bitnet_full cfg emission |
| 212-242 | REPLACE | Replace simple diagnostics with `emit_enhanced_diagnostics()` |

### 4.4 Dependency Changes

**No new dependencies required**. All changes use existing Rust std library:
- `std::path::{Path, PathBuf}`
- `std::env`
- `std::fs`

---

## 5. Test Requirements

### 5.1 Unit Test Strategy

**Test File**: `crossval/build.rs` (inline tests)

**Test 1: Three-State Backend Determination**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_state_full_bitnet() {
        // AC1: found_bitnet=true → FullBitNet regardless of llama
        assert!(matches!(
            determine_backend_state(true, true),
            BackendState::FullBitNet
        ));
        assert!(matches!(
            determine_backend_state(true, false),
            BackendState::FullBitNet
        ));
    }

    #[test]
    fn test_backend_state_llama_fallback() {
        // AC2: found_bitnet=false, found_llama=true → LlamaFallback
        assert!(matches!(
            determine_backend_state(false, true),
            BackendState::LlamaFallback
        ));
    }

    #[test]
    fn test_backend_state_unavailable() {
        // AC3: Both false → Unavailable
        assert!(matches!(
            determine_backend_state(false, false),
            BackendState::Unavailable
        ));
    }

    #[test]
    fn test_backend_state_as_str() {
        // AC4: Enum → string conversion
        assert_eq!(BackendState::FullBitNet.as_str(), "full");
        assert_eq!(BackendState::LlamaFallback.as_str(), "llama");
        assert_eq!(BackendState::Unavailable.as_str(), "none");
    }
}
```

**Test 2: Search Path Tiers**
```rust
#[test]
fn test_search_path_tiers() {
    // AC5: Verify three-tier hierarchy
    let (primary, embedded, fallback) = build_search_path_tiers("/test/bitnet");

    // Primary paths (3 expected)
    assert_eq!(primary.len(), 3);
    assert!(primary[0].to_str().unwrap().contains("build/3rdparty/llama.cpp/build/bin"));
    assert!(primary[1].to_str().unwrap().contains("build/lib"));
    assert!(primary[2].to_str().unwrap().contains("build/bin"));

    // Embedded paths (2 expected)
    assert_eq!(embedded.len(), 2);
    assert!(embedded[0].to_str().unwrap().contains("build/3rdparty/llama.cpp/src"));
    assert!(embedded[1].to_str().unwrap().contains("build/3rdparty/llama.cpp/ggml/src"));

    // Fallback paths (2 expected)
    assert_eq!(fallback.len(), 2);
    assert!(fallback[0].to_str().unwrap().contains("build"));
    assert!(fallback[1].to_str().unwrap().contains("lib"));
}
```

### 5.2 Integration Test Strategy

**Test File**: `crossval/tests/build_integration.rs` (new file)

**Test 3: RPATH Emission Format**
```rust
#[test]
fn test_rpath_emission_format() {
    // AC6: RPATH is colon-separated, contains all library dirs
    // Simulate build.rs run with mock env
    std::env::set_var("BITNET_CPP_DIR", "/test/bitnet");

    // Run build.rs detection (in test mode)
    let rpath = std::env::var("CROSSVAL_RPATH_BITNET").unwrap();

    // Verify colon-separated
    assert!(rpath.contains(':'));

    // Verify contains expected paths
    assert!(rpath.contains("build/lib") || rpath.contains("build/bin"));
}
```

**Test 4: Environment Variable Emission**
```rust
#[test]
fn test_env_var_emission() {
    // AC7: All expected env vars emitted
    assert!(std::env::var("CROSSVAL_HAS_BITNET").is_ok());
    assert!(std::env::var("CROSSVAL_HAS_LLAMA").is_ok());
    assert!(std::env::var("CROSSVAL_BACKEND_STATE").is_ok());
    assert!(std::env::var("CROSSVAL_RPATH_BITNET").is_ok());
}
```

### 5.3 Acceptance Criteria Mapping

| AC ID | Description | Test Function | Location |
|-------|-------------|---------------|----------|
| AC1 | found_bitnet=true → FullBitNet | `test_backend_state_full_bitnet()` | `crossval/build.rs` |
| AC2 | found_llama=true only → LlamaFallback | `test_backend_state_llama_fallback()` | `crossval/build.rs` |
| AC3 | Both false → Unavailable | `test_backend_state_unavailable()` | `crossval/build.rs` |
| AC4 | Enum string conversion | `test_backend_state_as_str()` | `crossval/build.rs` |
| AC5 | Three-tier search paths | `test_search_path_tiers()` | `crossval/build.rs` |
| AC6 | RPATH colon-separated format | `test_rpath_emission_format()` | `crossval/tests/` |
| AC7 | Environment variables emitted | `test_env_var_emission()` | `crossval/tests/` |
| AC8 | Diagnostic messages clear | Manual verification | Build output |

### 5.4 Test Execution

**Run Unit Tests**:
```bash
# Standard cargo test (build.rs tests don't run by default)
# Need to test build.rs logic separately

# Option 1: Extract logic to lib.rs for testing
cargo test -p crossval --lib

# Option 2: Integration tests with build script
cargo clean -p crossval
BITNET_CPP_DIR=/test/mock cargo build -p crossval --features llama-ffi -vv
```

**Run Integration Tests**:
```bash
# With real BitNet.cpp installation
eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"
cargo clean -p crossval
cargo build -p crossval --features bitnet-ffi -vv | grep "cargo:warning=crossval"

# With llama.cpp only (simulate fallback)
BITNET_CPP_DIR=/path/to/llama-only cargo build -p crossval --features llama-ffi -vv

# With no backend (simulate stub mode)
BITNET_CPP_DIR=/nonexistent cargo build -p crossval --features llama-ffi -vv
```

---

## 6. Verification Criteria

### 6.1 Functional Verification

**Scenario 1: Full BitNet.cpp Available**

**Setup**:
```bash
eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"
cargo clean -p crossval
cargo build -p crossval --features bitnet-ffi -vv 2>&1 | tee /tmp/build.log
```

**Expected Output**:
```
cargo:rustc-env=CROSSVAL_HAS_BITNET=true
cargo:rustc-env=CROSSVAL_HAS_LLAMA=true
cargo:rustc-env=CROSSVAL_BACKEND_STATE=full
cargo:rustc-env=CROSSVAL_RPATH_BITNET=/path/to/bitnet/build/lib:/path/to/llama/src
cargo:rustc-cfg=have_cpp
cargo:rustc-cfg=have_bitnet_full
cargo:warning=crossval: ✓ BITNET_FULL: BitNet.cpp and llama.cpp libraries found
cargo:warning=crossval: Backend: full
```

**Verification Checks**:
- [ ] `CROSSVAL_BACKEND_STATE=full` present
- [ ] `CROSSVAL_RPATH_BITNET` contains multiple paths
- [ ] `have_bitnet_full` cfg emitted
- [ ] Diagnostic says "BITNET_FULL"

**Scenario 2: Llama.cpp Fallback**

**Setup**:
```bash
# Remove libbitnet but keep llama.cpp
rm $BITNET_CPP_DIR/build/lib/libbitnet*
cargo clean -p crossval
cargo build -p crossval --features llama-ffi -vv 2>&1 | tee /tmp/build.log
```

**Expected Output**:
```
cargo:rustc-env=CROSSVAL_HAS_BITNET=false
cargo:rustc-env=CROSSVAL_HAS_LLAMA=true
cargo:rustc-env=CROSSVAL_BACKEND_STATE=llama
cargo:rustc-env=CROSSVAL_RPATH_BITNET=/path/to/llama/src:/path/to/ggml/src
cargo:rustc-cfg=have_cpp
cargo:warning=crossval: ⚠ LLAMA_FALLBACK: LLaMA.cpp libraries found, BitNet.cpp NOT found
cargo:warning=crossval: Backend: llama (fallback)
```

**Verification Checks**:
- [ ] `CROSSVAL_BACKEND_STATE=llama` present
- [ ] `have_bitnet_full` cfg NOT emitted
- [ ] Diagnostic says "LLAMA_FALLBACK"
- [ ] Warning mentions "BitNet.cpp NOT found"

**Scenario 3: No Backend Available**

**Setup**:
```bash
BITNET_CPP_DIR=/nonexistent cargo clean -p crossval
BITNET_CPP_DIR=/nonexistent cargo build -p crossval --features llama-ffi -vv 2>&1 | tee /tmp/build.log
```

**Expected Output**:
```
cargo:rustc-env=CROSSVAL_HAS_BITNET=false
cargo:rustc-env=CROSSVAL_HAS_LLAMA=false
cargo:rustc-env=CROSSVAL_BACKEND_STATE=none
cargo:warning=crossval: ✗ BITNET_STUB mode: No C++ libraries found
cargo:warning=crossval: Backend: none
```

**Verification Checks**:
- [ ] `CROSSVAL_BACKEND_STATE=none` present
- [ ] `have_cpp` cfg NOT emitted
- [ ] Diagnostic says "BITNET_STUB mode"

### 6.2 Performance Verification

**Metric**: Build time impact

**Baseline** (current implementation):
```bash
time cargo clean -p crossval && cargo build -p crossval --features llama-ffi
```

**Enhanced** (new implementation):
```bash
time cargo clean -p crossval && cargo build -p crossval --features llama-ffi
```

**Acceptable Threshold**: ≤ 5% build time increase

**Rationale**: Three-tier search adds minimal overhead (7 paths vs 6 currently).

### 6.3 Cross-Platform Verification

**Linux**:
- [ ] RPATH emission uses colon separator
- [ ] Runtime loader finds libraries without LD_LIBRARY_PATH
- [ ] Diagnostics correct

**macOS**:
- [ ] RPATH emission uses colon separator
- [ ] Runtime loader finds libraries without DYLD_LIBRARY_PATH
- [ ] .dylib libraries detected correctly

**Windows**:
- [ ] CROSSVAL_LIBPATH_BITNET uses semicolon separator
- [ ] .dll and .lib libraries detected correctly
- [ ] Diagnostics mention PATH requirement

### 6.4 Regression Verification

**Existing Functionality** (must not break):
- [ ] `cargo build -p crossval --features llama-ffi` still works
- [ ] RPATH embedding still functional (no LD_LIBRARY_PATH needed)
- [ ] C++ wrapper compilation (BITNET_AVAILABLE vs BITNET_STUB modes)
- [ ] Backward compatibility with BITNET_CPP_PATH
- [ ] Explicit BITNET_CROSSVAL_LIBDIR override still works

**Test Commands**:
```bash
# Test 1: Basic llama-ffi build
cargo clean -p crossval && cargo build -p crossval --features llama-ffi

# Test 2: RPATH still works (no LD_LIBRARY_PATH needed)
cargo run -p xtask --features crossval-all -- preflight --backend llama

# Test 3: Legacy BITNET_CPP_PATH
BITNET_CPP_PATH=$HOME/.cache/bitnet_cpp cargo clean -p crossval
BITNET_CPP_PATH=$HOME/.cache/bitnet_cpp cargo build -p crossval --features llama-ffi

# Test 4: Explicit override
BITNET_CROSSVAL_LIBDIR=/custom/path cargo build -p crossval --features llama-ffi
```

---

## 7. Backward Compatibility

### 7.1 Breaking Changes

**None**. This is a pure enhancement with backward-compatible additions.

### 7.2 Deprecations

**BITNET_CPP_PATH** (optional):
- Status: Deprecated but still supported
- Recommendation: Use BITNET_CPP_DIR instead
- Timeline: Remove in v0.3.0 (2 releases from now)

### 7.3 Migration Guide

**No migration required**. Existing code continues to work.

**Optional Enhancements** (consumers can adopt):

**Before** (current usage):
```rust
#[cfg(have_cpp)]
fn with_cpp_backend() {
    // Could be BitNet OR llama fallback - ambiguous
}
```

**After** (enhanced usage):
```rust
#[cfg(have_bitnet_full)]
fn with_bitnet_backend() {
    // Guaranteed to be full BitNet.cpp backend
}

#[cfg(all(have_cpp, not(have_bitnet_full)))]
fn with_llama_fallback() {
    // Only llama.cpp available
}
```

---

## 8. References

### 8.1 Related Documents

- `/tmp/bitnet-build-structure.md`: Directory structure analysis
- `/tmp/crossval-build-detection.md`: Current detection logic (902 lines)
- `/tmp/crossval-build-detection-summary.txt`: Five-gap summary
- `docs/howto/cpp-setup.md`: C++ reference setup guide
- `docs/explanation/dual-backend-crossval.md`: Architecture overview

### 8.2 Related Issues

- Issue #254: Shape mismatch in layer-norm (blocks real inference tests)
- Issue #260: Mock elimination not complete
- Issue #439: Feature gate consistency (✅ RESOLVED in PR #475)
- Issue #469: Tokenizer parity and FFI build hygiene

### 8.3 Code Locations

| Concept | File | Lines |
|---------|------|-------|
| Current detection | `crossval/build.rs` | 28-252 |
| RPATH consumer | `xtask/build.rs` | 28-40 |
| Runtime preflight | `xtask/src/crossval/preflight.rs` | 554-579 |
| Auto-setup | `xtask/src/cpp_setup_auto.rs` | 56-153 |

### 8.4 External Dependencies

**CMake Projects**:
- Microsoft BitNet.cpp: `https://github.com/microsoft/BitNet.git`
- llama.cpp (embedded): `3rdparty/llama.cpp` submodule

**Build Tools**:
- `cc` crate: C++ wrapper compilation
- `cargo`: Build orchestration
- `rustc`: Compiler with cfg emission

---

## Appendix A: Complete Example Output

### Full BitNet.cpp Scenario

```bash
$ cargo clean -p crossval
$ cargo build -p crossval --features bitnet-ffi -vv 2>&1 | grep "cargo:"

cargo:rerun-if-changed=build.rs
cargo:rerun-if-changed=src/bitnet_cpp_wrapper.c
cargo:rerun-if-changed=src/bitnet_cpp_wrapper.cc
cargo:rustc-check-cfg=cfg(have_cpp)
cargo:rustc-link-search=native=/home/user/.cache/bitnet_cpp/build/3rdparty/llama.cpp/build/bin
cargo:rustc-link-search=native=/home/user/.cache/bitnet_cpp/build/lib
cargo:rustc-link-search=native=/home/user/.cache/bitnet_cpp/build/3rdparty/llama.cpp/src
cargo:rustc-link-search=native=/home/user/.cache/bitnet_cpp/build/3rdparty/llama.cpp/ggml/src
cargo:rustc-link-lib=dylib=bitnet
cargo:rustc-link-lib=dylib=llama
cargo:rustc-link-lib=dylib=ggml
cargo:rustc-link-arg=-Wl,-rpath,/home/user/.cache/bitnet_cpp/build/lib:/home/user/.cache/bitnet_cpp/build/3rdparty/llama.cpp/src:/home/user/.cache/bitnet_cpp/build/3rdparty/llama.cpp/ggml/src
cargo:rustc-env=CROSSVAL_HAS_BITNET=true
cargo:rustc-env=CROSSVAL_HAS_LLAMA=true
cargo:rustc-env=CROSSVAL_BACKEND_STATE=full
cargo:rustc-env=CROSSVAL_RPATH_BITNET=/home/user/.cache/bitnet_cpp/build/lib:/home/user/.cache/bitnet_cpp/build/3rdparty/llama.cpp/src:/home/user/.cache/bitnet_cpp/build/3rdparty/llama.cpp/ggml/src
cargo:rustc-cfg=have_cpp
cargo:rustc-cfg=have_bitnet_full
cargo:warning=crossval: Compiling C++ wrapper in BITNET_AVAILABLE mode
cargo:warning=crossval: ✓ BITNET_FULL: BitNet.cpp and llama.cpp libraries found
cargo:warning=crossval: Backend: full
cargo:warning=crossval: Linked libraries: bitnet, llama, ggml
cargo:warning=crossval: Headers found in: /home/user/.cache/bitnet_cpp
cargo:rustc-link-lib=static=bitnet_cpp_wrapper_cc
cargo:rustc-link-lib=dylib=stdc++
```

---

## Appendix B: TDD Test Scaffolding

**Test File Structure**:

```
crossval/
├── build.rs (modified with inline tests)
├── tests/
│   ├── build_integration.rs (new file)
│   └── backend_state_tests.rs (new file)
└── src/
    └── lib.rs (unchanged)
```

**Test Tags for TDD**:
```rust
// AC:1 - Full BitNet detection
#[test]
fn test_full_bitnet_detection() { /* ... */ }

// AC:2 - Llama fallback detection
#[test]
fn test_llama_fallback_detection() { /* ... */ }

// AC:3 - Unavailable state
#[test]
fn test_unavailable_state() { /* ... */ }

// AC:4 - String conversion
#[test]
fn test_backend_state_as_str() { /* ... */ }

// AC:5 - Three-tier search paths
#[test]
fn test_search_path_tiers() { /* ... */ }

// AC:6 - RPATH format
#[test]
fn test_rpath_emission_format() { /* ... */ }

// AC:7 - Environment variables
#[test]
fn test_env_var_emission() { /* ... */ }

// AC:8 - Diagnostic messages
#[test]
fn test_diagnostic_messages() { /* ... */ }
```

---

**End of Specification**

# Technical Specification: Preflight Diagnostics UX Improvements

**Document ID**: SPEC-2025-PREFLIGHT-UX-001
**Status**: Draft
**Created**: 2025-10-25
**Author**: BitNet.rs Neural Network Systems Architect
**Related Documents**:
- `/tmp/xtask-crossval-exploration.md` (Analysis)
- `docs/explanation/dual-backend-crossval.md` (Architecture)
- `docs/howto/cpp-setup.md` (Setup Guide)

---

## Executive Summary

The preflight diagnostics system validates C++ library availability before cross-validation runs. Current error messages confuse users about build-time vs runtime library detection, lack actionable recovery steps, and don't clearly explain when rpath embedding requires rebuild.

This specification defines improvements to make preflight diagnostics production-grade, with clear error messages, actionable recovery paths, and comprehensive diagnostic output for debugging C++ integration issues.

**Key improvements**:
1. Enhanced error messages with explicit build-time vs runtime distinction
2. Actionable recovery steps with exact commands
3. Verbose mode diagnostics showing library search paths and detection results
4. Programmatic access to HAS_BITNET and HAS_LLAMA constants
5. Exit code standardization for CI/CD integration

---

## 1. Requirements Analysis

### 1.1 User Stories

**US-1: Developer encounters missing libllama.so at runtime**
- **As a**: Developer running cross-validation for the first time
- **I need**: Clear error messages explaining why libllama.so is missing
- **So that**: I can quickly fix the issue without deep debugging

**US-2: Developer needs to understand build-time detection**
- **As a**: Developer who just installed C++ libraries
- **I need**: To understand why xtask must be rebuilt to detect new libraries
- **So that**: I don't waste time troubleshooting when the fix is a simple rebuild

**US-3: CI engineer validates C++ setup before tests**
- **As a**: CI engineer setting up cross-validation pipeline
- **I need**: To validate C++ library availability with clear exit codes
- **So that**: CI can fail fast with actionable error messages

**US-4: Developer debugs library search paths**
- **As a**: Developer with a complex C++ installation
- **I need**: Verbose diagnostics showing exactly which paths were searched
- **So that**: I can understand why libraries aren't being detected

**US-5: Developer chooses between LD_LIBRARY_PATH and rpath**
- **As a**: Developer setting up cross-validation
- **I need**: To understand the tradeoff between LD_LIBRARY_PATH and rpath rebuild
- **So that**: I can choose the right approach for my workflow

### 1.2 Functional Requirements

**FR-1: Enhanced Error Messages**
- Error messages MUST clearly distinguish build-time vs runtime detection
- Error messages MUST provide numbered, actionable recovery steps
- Error messages MUST include exact copy-paste commands
- Error messages MUST explain when rebuild is required vs when LD_LIBRARY_PATH is sufficient

**FR-2: Verbose Diagnostics**
- Verbose mode MUST show environment variables checked and their values
- Verbose mode MUST show library search paths in priority order
- Verbose mode MUST indicate which paths exist vs don't exist
- Verbose mode MUST list all libraries found in each path (not just required ones)
- Verbose mode MUST show build-time detection constants (CROSSVAL_HAS_BITNET, CROSSVAL_HAS_LLAMA)

**FR-3: Exit Code Standardization**
- Exit code 0: All required libraries available
- Exit code 1: Backend unavailable (libraries missing)
- Exit code 2: Invalid arguments (e.g., unknown backend)

**FR-4: Programmatic Access**
- HAS_BITNET and HAS_LLAMA constants MUST be publicly exported from bitnet-crossval crate
- Constants MUST be documented with examples
- Other crates MUST be able to query availability at runtime

**FR-5: Backend-Specific Validation**
- Preflight MUST support checking specific backend (--backend bitnet|llama)
- Preflight MUST support checking all backends (no --backend flag)
- Backend-specific checks MUST validate all required libraries for that backend

### 1.3 Non-Functional Requirements

**NFR-1: Performance**
- Preflight checks MUST complete in < 100ms (no expensive filesystem operations)
- Use build-time constants instead of runtime library scanning

**NFR-2: Clarity**
- Error messages MUST be understandable by developers unfamiliar with Rust/C++ FFI
- Recovery steps MUST be ordered by likelihood of success
- Technical jargon MUST be explained in context

**NFR-3: Maintainability**
- Error message templates MUST be centralized for easy updates
- Diagnostic output MUST mirror crossval/build.rs logic exactly
- Library search path logic MUST be shared between build.rs and runtime diagnostics

---

## 2. User Journey Analysis

### 2.1 Scenario A: First-Time User (No C++ Installed)

**Journey**:
1. User runs `cargo run -p xtask --features crossval-all -- crossval-per-token --model model.gguf ...`
2. Command fails with error about missing backend libraries
3. User sees clear error message with setup instructions
4. User runs `eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"`
5. User rebuilds xtask: `cargo clean -p xtask && cargo build -p xtask --features crossval-all`
6. User re-runs original command successfully

**Pain Points (Current)**:
- Error message doesn't explain build-time detection clearly
- Rebuild step is buried in error text
- No indication which backend is auto-detected from model path

**Improvements (Proposed)**:
- Error message clearly states "Library detection happens at BUILD time"
- Rebuild command is step 2 (after setup), highlighted with visual separators
- Show which backend was auto-detected and why

### 2.2 Scenario B: User Just Installed C++ Libraries

**Journey**:
1. User manually installs llama.cpp and sets BITNET_CPP_DIR
2. User runs cross-validation command
3. Command fails with "libraries not found"
4. User confused: "I just installed them!"
5. User reads error message about rebuild requirement
6. User rebuilds xtask
7. Command succeeds

**Pain Points (Current)**:
- Not obvious that xtask caches build-time detection
- Users expect runtime detection to see new libraries

**Improvements (Proposed)**:
- Error prominently states: "CRITICAL: xtask must be REBUILT to detect C++ libraries"
- Explain that detection is build-time for performance (no runtime I/O)
- Show exact rebuild command at top of error message

### 2.3 Scenario C: CI Pipeline Setup

**Journey**:
1. CI engineer adds cross-validation step to pipeline
2. Wants to validate C++ setup before expensive tests
3. Runs `cargo run -p xtask -- preflight --backend llama --verbose`
4. Gets clear exit code 1 if libraries missing
5. CI fails fast with actionable error message in logs

**Pain Points (Current)**:
- No standard exit codes documented
- Not clear if exit code 0 means "available" or "check skipped"

**Improvements (Proposed)**:
- Document exit codes in command help text
- Standardize: 0=success, 1=unavailable, 2=invalid args
- Verbose output suitable for CI logs (structured, grep-able)

### 2.4 Scenario D: Complex Library Installation

**Journey**:
1. User has custom BITNET_CPP_DIR with non-standard layout
2. Libraries installed but not in expected paths
3. User runs `cargo run -p xtask -- preflight --backend llama --verbose`
4. Verbose output shows exactly which paths were searched
5. User identifies mismatch: libraries in build/bin but xtask checks build/lib
6. User sets BITNET_CROSSVAL_LIBDIR=/path/to/build/bin
7. User rebuilds xtask
8. Preflight succeeds

**Pain Points (Current)**:
- Verbose output shows search paths but not clearly what was searched for
- No indication which paths exist vs don't exist

**Improvements (Proposed)**:
- Show search paths in priority order with existence check (✓ exists / ✗ missing)
- Show what was searched for in each path (libbitnet*, libllama*, libggml*)
- List all libraries found (even if not required) to help debugging

---

## 3. Error Message Templates

### 3.1 Template: Backend Libraries Not Found

**Context**: `preflight_backend_libs()` when HAS_BITNET or HAS_LLAMA is false

**Current Message** (lines 53-74 in preflight.rs):
```
Backend 'llama.cpp' selected but required libraries not found.

Setup instructions:
1. Install C++ reference implementation:
   eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"

2. Verify libraries are loaded:
   cargo run -p xtask -- preflight --backend llama

3. Rebuild xtask to detect libraries:
   cargo clean -p xtask && cargo build -p xtask --features crossval-all

Required libraries: ["libllama", "libggml"]

Note: Library detection happens at BUILD time. If you just installed
the C++ reference, you must rebuild xtask for detection to work.
```

**Proposed Message** (with enhanced clarity and visual structure):
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ Backend 'llama.cpp' libraries NOT FOUND
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CRITICAL: Library detection happens at BUILD time, not runtime.
If you just installed C++ libraries, xtask MUST be rebuilt to detect them.

Required libraries: libllama*.so, libggml*.so
Auto-detected backend: llama.cpp (from model path heuristics)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RECOVERY STEPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Option A: One-Command Setup (Recommended for First-Time Users)
─────────────────────────────────────────────────────────────────────

  Step 1: Install and configure C++ reference implementation
    eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"

  Step 2: Rebuild xtask to detect newly installed libraries
    cargo clean -p xtask && cargo build -p xtask --features crossval-all

  Step 3: Verify detection succeeded
    cargo run -p xtask -- preflight --backend llama --verbose

  Then retry your original command.

Option B: Manual Setup + LD_LIBRARY_PATH (No Rebuild Required)
─────────────────────────────────────────────────────────────────────

  Step 1: Install llama.cpp manually (skip if already installed)
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp && cmake -B build && cmake --build build

  Step 2: Set environment variable for this session
    export BITNET_CPP_DIR=/path/to/llama.cpp
    export LD_LIBRARY_PATH=$BITNET_CPP_DIR/build:$LD_LIBRARY_PATH  # Linux
    export DYLD_LIBRARY_PATH=$BITNET_CPP_DIR/build:$DYLD_LIBRARY_PATH  # macOS

  Step 3: Rebuild xtask to embed library paths (rpath)
    cargo clean -p xtask && cargo build -p xtask --features crossval-all

  Note: Option B requires setting LD_LIBRARY_PATH before EVERY run.
        Option A embeds library paths permanently (rpath).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TROUBLESHOOTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

If setup fails, run verbose diagnostics to see what's happening:
  cargo run -p xtask -- preflight --backend llama --verbose

This will show:
  • Environment variables checked (BITNET_CPP_DIR, LD_LIBRARY_PATH, etc.)
  • Library search paths in priority order
  • Which paths exist vs missing
  • All libraries found in each path
  • Build-time detection flags

For more help, see:
  docs/howto/cpp-setup.md (Detailed C++ setup guide)
  docs/explanation/dual-backend-crossval.md (Architecture overview)
```

**Key improvements**:
- Visual separators for clarity (━━━━━━)
- "CRITICAL" callout for rebuild requirement
- Two recovery options (recommended vs manual)
- Clear tradeoffs between options (rpath vs LD_LIBRARY_PATH)
- Troubleshooting section with verbose diagnostics hint
- References to documentation for deep dives

### 3.2 Template: Verbose Success Diagnostics

**Context**: `print_verbose_success_diagnostics()` when libraries found

**Current Output** (lines 87-152 in preflight.rs):
```
✓ Backend 'llama.cpp' libraries: AVAILABLE

Environment Variables:
  BITNET_CPP_DIR = /home/user/.cache/bitnet_cpp
  BITNET_CPP_PATH = (not set)
  ...
```

**Proposed Output** (enhanced with structure and insights):
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Backend 'llama.cpp': AVAILABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Environment Configuration
─────────────────────────────────────────────────────────────────────
  BITNET_CPP_DIR        = /home/user/.cache/bitnet_cpp
  BITNET_CROSSVAL_LIBDIR = (not set)
  LD_LIBRARY_PATH       = /home/user/.cache/bitnet_cpp/build:/usr/local/lib
  BITNET_CPP_PATH       = (not set - deprecated, use BITNET_CPP_DIR)

Library Search Paths (Priority Order)
─────────────────────────────────────────────────────────────────────
  1. BITNET_CROSSVAL_LIBDIR override
     (not set - using default search order)

  2. BITNET_CPP_DIR/build
     ✓ /home/user/.cache/bitnet_cpp/build (exists)
     Found libraries:
       - libllama.so.1.0.0
       - libggml.so.1.0.0

  3. BITNET_CPP_DIR/build/lib
     ✗ /home/user/.cache/bitnet_cpp/build/lib (not found)

  4. BITNET_CPP_DIR/build/3rdparty/llama.cpp/src
     ✗ /home/user/.cache/bitnet_cpp/build/3rdparty/llama.cpp/src (not found)

  5. BITNET_CPP_DIR/build/3rdparty/llama.cpp/ggml/src
     ✗ /home/user/.cache/bitnet_cpp/build/3rdparty/llama.cpp/ggml/src (not found)

  6. BITNET_CPP_DIR/lib
     ✗ /home/user/.cache/bitnet_cpp/lib (not found)

Required Libraries for llama.cpp Backend
─────────────────────────────────────────────────────────────────────
  ✓ libllama.so (found in build/)
  ✓ libggml.so (found in build/)

Build-Time Detection Metadata
─────────────────────────────────────────────────────────────────────
  CROSSVAL_HAS_LLAMA = true
  CROSSVAL_HAS_BITNET = false
  Detection timestamp: 2025-10-25T10:30:00Z (xtask last built)
  Linked libraries: libllama, libggml (dynamic linking)
  Runtime library resolution: rpath embedded

Platform-Specific Configuration
─────────────────────────────────────────────────────────────────────
  Platform: Linux
  Standard library: libstdc++ (dynamic linking)
  RPATH embedded: YES (no LD_LIBRARY_PATH required)
  Loader search order: rpath → LD_LIBRARY_PATH → system paths

Summary
─────────────────────────────────────────────────────────────────────
✓ All required libraries detected at build time
✓ Runtime library resolution configured (rpath)
✓ Cross-validation with llama.cpp is supported

To test cross-validation:
  cargo run -p xtask --features crossval-all -- crossval-per-token \
    --model models/model.gguf \
    --tokenizer models/tokenizer.json \
    --prompt "Test" \
    --max-tokens 4 \
    --cpp-backend llama \
    --verbose
```

**Key improvements**:
- Structured sections with visual separators
- Search paths numbered in priority order with existence checks
- Shows all libraries found (not just required ones)
- Build-time metadata (when xtask was built, what was detected)
- Platform-specific details (rpath, dynamic linking, loader order)
- Summary with actionable next steps

### 3.3 Template: Verbose Failure Diagnostics

**Context**: `print_verbose_failure_diagnostics()` when libraries not found

**Current Output** (lines 155-257 in preflight.rs):
Shows environment variables, search paths, but lacks clarity on what to do next.

**Proposed Output**:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ Backend 'llama.cpp': NOT AVAILABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DIAGNOSIS: Required libraries not detected at xtask build time.
This means either:
  (a) C++ libraries were never installed, OR
  (b) C++ libraries were installed AFTER xtask was built

Environment Configuration (Current State)
─────────────────────────────────────────────────────────────────────
  BITNET_CPP_DIR        = (not set)
  BITNET_CROSSVAL_LIBDIR = (not set)
  LD_LIBRARY_PATH       = (not set)

  ⚠️  WARNING: No environment variables set for library discovery.
     xtask will search default path: ~/.cache/bitnet_cpp

Library Search Paths (Checked During Last Build)
─────────────────────────────────────────────────────────────────────
  1. BITNET_CROSSVAL_LIBDIR override
     (not set - using default search order)

  2. ~/.cache/bitnet_cpp/build
     ✗ /home/user/.cache/bitnet_cpp/build (not found)

  3. ~/.cache/bitnet_cpp/build/lib
     ✗ /home/user/.cache/bitnet_cpp/build/lib (not found)

  4. ~/.cache/bitnet_cpp/build/3rdparty/llama.cpp/src
     ✗ /home/user/.cache/bitnet_cpp/build/3rdparty/llama.cpp/src (not found)

  (... all paths missing ...)

Required Libraries (Searched For)
─────────────────────────────────────────────────────────────────────
  ✗ libllama.so / libllama.dylib
  ✗ libggml.so / libggml.dylib

Other Libraries Found in Search Paths
─────────────────────────────────────────────────────────────────────
  (none - no library directories exist)

Build-Time Detection Metadata
─────────────────────────────────────────────────────────────────────
  CROSSVAL_HAS_LLAMA = false
  CROSSVAL_HAS_BITNET = false
  Last xtask build: 2025-10-25T09:00:00Z (25 minutes ago)
  Build feature flags: crossval-all

RECOMMENDED FIX
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: Install C++ reference implementation (auto-setup):
  eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"

  This will:
    • Clone llama.cpp to ~/.cache/bitnet_cpp
    • Build with CMake (static linking)
    • Set BITNET_CPP_DIR environment variable
    • Add LD_LIBRARY_PATH to your shell profile

Step 2: Rebuild xtask to detect newly installed libraries:
  cargo clean -p xtask
  cargo build -p xtask --features crossval-all

  Why rebuild?
    • Library detection runs during BUILD (not runtime)
    • Build script scans filesystem for libllama*/libggml*
    • Detection results baked into xtask binary as constants
    • If libraries installed after build, xtask won't see them

Step 3: Verify detection succeeded:
  cargo run -p xtask -- preflight --backend llama --verbose

  Expected output:
    ✓ Backend 'llama.cpp': AVAILABLE
    CROSSVAL_HAS_LLAMA = true

Step 4: Retry your original cross-validation command.

ALTERNATIVE: Manual Installation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

If auto-setup fails, install manually:

  1. Clone and build llama.cpp:
     git clone https://github.com/ggerganov/llama.cpp
     cd llama.cpp
     cmake -B build -DBUILD_SHARED_LIBS=ON
     cmake --build build

  2. Set environment variables:
     export BITNET_CPP_DIR=/path/to/llama.cpp
     export LD_LIBRARY_PATH=$BITNET_CPP_DIR/build:$LD_LIBRARY_PATH

  3. Rebuild xtask (same as Step 2 above)

For detailed guidance, see: docs/howto/cpp-setup.md
```

**Key improvements**:
- DIAGNOSIS section explains why libraries aren't found
- Shows when xtask was last built (helps identify staleness)
- Clear recommended fix with explanation of each step
- "Why rebuild?" callout explains build-time detection
- Expected output shown for verification
- Alternative manual installation for advanced users

---

## 4. CLI Output Specifications

### 4.1 Non-Verbose Mode (Default)

**Command**: `cargo run -p xtask -- preflight`

**Expected Output** (all backends available):
```
Backend Library Status:

  ✓ bitnet.cpp: AVAILABLE
    Libraries: libbitnet*

  ✓ llama.cpp: AVAILABLE
    Libraries: libllama*, libggml*

Both backends available. Dual-backend cross-validation supported.
```

**Expected Output** (one backend missing):
```
Backend Library Status:

  ✗ bitnet.cpp: NOT AVAILABLE
    Setup: eval "$(cargo run -p xtask -- setup-cpp-auto --bitnet --emit=sh)"

  ✓ llama.cpp: AVAILABLE
    Libraries: libllama*, libggml*

llama.cpp available. BitNet models will require setup.
Run: cargo run -p xtask -- preflight --backend bitnet --verbose
```

**Exit codes**:
- 0: All backends available or command succeeded
- (Non-zero exit only with --backend flag for specific backend check)

### 4.2 Verbose Mode

**Command**: `cargo run -p xtask -- preflight --verbose`

**Expected Output**:
- Shows full diagnostics for each backend (see section 3.2/3.3 templates)
- Uses structured format with visual separators
- Includes environment variables, search paths, build metadata

### 4.3 Backend-Specific Check (Non-Verbose)

**Command**: `cargo run -p xtask -- preflight --backend llama`

**Expected Output** (success):
```
✓ llama.cpp backend is available
```

**Exit code**: 0

**Expected Output** (failure):
```
❌ Backend 'llama.cpp' libraries NOT FOUND

<Abbreviated error message with recovery steps>
```

**Exit code**: 1

### 4.4 Backend-Specific Check (Verbose)

**Command**: `cargo run -p xtask -- preflight --backend llama --verbose`

**Expected Output** (success):
- Full success diagnostics (see section 3.2)
- **Exit code**: 0

**Expected Output** (failure):
- Full failure diagnostics (see section 3.3)
- **Exit code**: 1

### 4.5 Invalid Arguments

**Command**: `cargo run -p xtask -- preflight --backend invalid`

**Expected Output**:
```
error: invalid value 'invalid' for '--backend <BACKEND>'
  [possible values: bitnet, llama]

For more information, try '--help'.
```

**Exit code**: 2 (from clap argument parsing)

---

## 5. Implementation Approach

### 5.1 Architecture Overview

**Current Architecture**:
```
crossval/build.rs (build-time)
  ├─ Scans filesystem for libraries
  ├─ Sets CROSSVAL_HAS_BITNET, CROSSVAL_HAS_LLAMA env vars
  └─ Exports as cargo:rustc-env for compile-time access

crossval/src/lib.rs (compile-time constants)
  ├─ Reads option_env!("CROSSVAL_HAS_BITNET")
  ├─ Reads option_env!("CROSSVAL_HAS_LLAMA")
  └─ Exports as pub const HAS_BITNET, HAS_LLAMA

xtask/src/crossval/preflight.rs (runtime checks)
  ├─ Queries bitnet_crossval::HAS_BITNET
  ├─ Queries bitnet_crossval::HAS_LLAMA
  └─ Shows error messages if false
```

**No changes needed to architecture** - current design is sound. Improvements are UX-only.

### 5.2 Code Changes (File-by-File)

#### 5.2.1 `xtask/src/crossval/preflight.rs`

**Changes**:
1. Update `preflight_backend_libs()` error message (line 53-74) with new template (section 3.1)
2. Update `print_verbose_success_diagnostics()` (lines 87-152) with new template (section 3.2)
3. Update `print_verbose_failure_diagnostics()` (lines 155-257) with new template (section 3.3)
4. Add helper function to detect xtask build timestamp (for staleness detection)
5. Add helper function to format build metadata section

**New functions**:
```rust
/// Get xtask build timestamp for staleness detection
fn get_xtask_build_timestamp() -> Option<String> {
    // Check modification time of xtask binary
    std::env::current_exe().ok().and_then(|path| {
        std::fs::metadata(&path).ok().and_then(|meta| {
            meta.modified().ok().map(|time| {
                // Format as ISO 8601 timestamp
                format!("{:?}", time)
            })
        })
    })
}

/// Format build metadata section for diagnostics
fn format_build_metadata(backend: CppBackend) -> String {
    let has_backend = match backend {
        CppBackend::BitNet => bitnet_crossval::HAS_BITNET,
        CppBackend::Llama => bitnet_crossval::HAS_LLAMA,
    };

    let backend_name = match backend {
        CppBackend::BitNet => "BITNET",
        CppBackend::Llama => "LLAMA",
    };

    let timestamp = get_xtask_build_timestamp()
        .unwrap_or_else(|| "unknown".to_string());

    format!(
        "Build-Time Detection Metadata\n\
         ─────────────────────────────────────────────────────────────────────\n\
         CROSSVAL_HAS_{} = {}\n\
         Last xtask build: {}\n\
         Build feature flags: crossval-all",
        backend_name, has_backend, timestamp
    )
}
```

**Visual separator constant**:
```rust
const SEPARATOR_HEAVY: &str = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━";
const SEPARATOR_LIGHT: &str = "─────────────────────────────────────────────────────────────────────";
```

#### 5.2.2 `xtask/src/crossval/backend.rs`

**Changes**:
1. Add documentation for exit codes in `CppBackend` type
2. No functional changes (already well-structured)

**Documentation enhancement**:
```rust
/// C++ backend selection for cross-validation
///
/// # Exit Codes
///
/// When used with `xtask preflight --backend <backend>`:
/// - 0: Backend available (libraries found at build time)
/// - 1: Backend unavailable (libraries missing)
/// - 2: Invalid backend argument (from clap)
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum CppBackend {
    // ... (existing code)
}
```

#### 5.2.3 `xtask/src/main.rs`

**Changes**:
1. Update `cpp_backend_preflight_cmd()` help documentation (line 2553-2578)
2. Add exit code handling for backend-specific checks
3. Update `Preflight` command documentation with exit codes

**Updated command handler**:
```rust
#[cfg(any(feature = "crossval", feature = "crossval-all"))]
fn cpp_backend_preflight_cmd(backend: Option<CppBackend>, verbose: bool) -> Result<()> {
    use crossval::{preflight_backend_libs, print_backend_status};

    match backend {
        Some(b) => {
            // Check specific backend - exit code reflects availability
            if verbose {
                eprintln!("Checking {} backend...", b.name());
            }
            preflight_backend_libs(b, verbose)?;

            // Only print success message if not verbose (verbose already printed)
            if !verbose {
                println!("✓ {} backend is available", b.name());
            }
        }
        None => {
            // Check all backends - always exit 0 (informational)
            println!("Checking all backends...\n");
            print_backend_status(verbose);
        }
    }

    Ok(())
}
```

**Updated command documentation**:
```rust
/// Check C++ backend library availability for cross-validation
///
/// Validates that required C++ libraries (libbitnet*, libllama*, libggml*)
/// were detected during xtask build. Library detection happens at BUILD time,
/// not runtime, so xtask must be rebuilt if C++ libraries are installed after
/// the initial build.
///
/// # Exit Codes
///
/// - 0: Backend available (or general status check succeeded)
/// - 1: Backend unavailable (libraries not found at build time)
/// - 2: Invalid arguments (unknown backend)
///
/// # Examples
///
/// Check all backends (exit 0 regardless):
///   cargo run -p xtask --features crossval-all -- preflight
///
/// Check specific backend (exit 1 if unavailable):
///   cargo run -p xtask --features crossval-all -- preflight --backend llama
///
/// Verbose diagnostics (shows search paths, build metadata):
///   cargo run -p xtask --features crossval-all -- preflight --backend bitnet --verbose
#[cfg(any(feature = "crossval", feature = "crossval-all"))]
Preflight {
    /// Backend to check (bitnet or llama). If omitted, checks both.
    #[arg(long, value_enum)]
    backend: Option<CppBackend>,

    /// Show detailed diagnostic information (environment vars, search paths, build metadata)
    #[arg(long, short)]
    verbose: bool,
},
```

#### 5.2.4 `crossval/src/lib.rs`

**Changes**:
1. Enhance documentation for `HAS_BITNET` and `HAS_LLAMA` constants
2. Add usage examples
3. No functional changes (constants already exported correctly)

**Enhanced documentation**:
```rust
/// Indicates whether BitNet.cpp libraries were detected at build time
///
/// This constant is set by `crossval/build.rs` based on library availability
/// during compilation. Other crates (like xtask) can query this at runtime
/// to determine if BitNet cross-validation is supported.
///
/// # Build-Time Detection
///
/// Detection happens when `bitnet-crossval` is compiled (not at runtime).
/// If C++ libraries are installed after compilation, this constant will
/// remain `false` until `bitnet-crossval` is rebuilt.
///
/// # Usage
///
/// ```rust
/// use bitnet_crossval::HAS_BITNET;
///
/// if HAS_BITNET {
///     println!("BitNet.cpp cross-validation available");
/// } else {
///     eprintln!("BitNet.cpp not available - install and rebuild");
/// }
/// ```
///
/// # See Also
///
/// - [`HAS_LLAMA`]: LLaMA.cpp library availability
/// - `crossval/build.rs`: Build-time detection logic
/// - `docs/howto/cpp-setup.md`: C++ setup guide
pub const HAS_BITNET: bool = const_str_eq(option_env!("CROSSVAL_HAS_BITNET"), "true");

/// Indicates whether LLaMA.cpp libraries were detected at build time
///
/// This constant is set by `crossval/build.rs` based on library availability
/// during compilation. Other crates (like xtask) can query this at runtime
/// to determine if LLaMA cross-validation is supported.
///
/// # Build-Time Detection
///
/// Detection happens when `bitnet-crossval` is compiled (not at runtime).
/// If C++ libraries are installed after compilation, this constant will
/// remain `false` until `bitnet-crossval` is rebuilt.
///
/// # Usage
///
/// ```rust
/// use bitnet_crossval::HAS_LLAMA;
///
/// if HAS_LLAMA {
///     println!("LLaMA.cpp cross-validation available");
/// } else {
///     eprintln!("LLaMA.cpp not available - install and rebuild");
/// }
/// ```
///
/// # See Also
///
/// - [`HAS_BITNET`]: BitNet.cpp library availability
/// - `crossval/build.rs`: Build-time detection logic
/// - `docs/howto/cpp-setup.md`: C++ setup guide
pub const HAS_LLAMA: bool = const_str_eq(option_env!("CROSSVAL_HAS_LLAMA"), "true");
```

#### 5.2.5 `crossval/build.rs`

**Changes**:
1. Add comments explaining rpath embedding for runtime library resolution
2. No functional changes (already comprehensive)

**Enhanced comments** (around lines 89-93):
```rust
// Add RPATH for runtime library resolution (Linux/macOS)
// This eliminates the need for LD_LIBRARY_PATH/DYLD_LIBRARY_PATH at runtime.
// RPATH is embedded into the binary and tells the dynamic linker where to
// search for shared libraries. This is more convenient than environment variables
// but requires rebuild if library paths change.
#[cfg(target_os = "linux")]
println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());

#[cfg(target_os = "macos")]
println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
```

### 5.3 Testing Strategy

**Unit Tests** (add to `xtask/src/crossval/preflight.rs::tests`):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_separator_constants_defined() {
        // Ensure visual separators are consistent
        assert!(!SEPARATOR_HEAVY.is_empty());
        assert!(!SEPARATOR_LIGHT.is_empty());
        assert_eq!(SEPARATOR_HEAVY.len(), SEPARATOR_LIGHT.len());
    }

    #[test]
    fn test_get_xtask_build_timestamp_format() {
        // Timestamp should be parseable or "unknown"
        if let Some(ts) = get_xtask_build_timestamp() {
            assert!(ts.len() > 0);
            // Could be ISO 8601 or system debug format
        }
    }

    #[test]
    fn test_format_build_metadata_structure() {
        let metadata = format_build_metadata(CppBackend::Llama);

        // Should contain key sections
        assert!(metadata.contains("Build-Time Detection Metadata"));
        assert!(metadata.contains("CROSSVAL_HAS_LLAMA"));
        assert!(metadata.contains("Last xtask build"));
    }

    #[test]
    fn test_preflight_backend_libs_exit_behavior() {
        // If HAS_LLAMA is false, should return Err
        // If HAS_LLAMA is true, should return Ok
        // (Actual result depends on build-time detection)
        let result = preflight_backend_libs(CppBackend::Llama, false);

        // Verify it returns a Result (doesn't panic)
        match result {
            Ok(_) => assert!(bitnet_crossval::HAS_LLAMA, "Should only succeed if HAS_LLAMA=true"),
            Err(_) => assert!(!bitnet_crossval::HAS_LLAMA, "Should only fail if HAS_LLAMA=false"),
        }
    }
}
```

**Integration Tests** (add to `xtask/tests/preflight_integration.rs`):

```rust
use std::process::Command;

#[test]
fn test_preflight_all_backends_exit_0() {
    // Checking all backends should always exit 0 (informational)
    let output = Command::new("cargo")
        .args(&["run", "-p", "xtask", "--features", "crossval-all", "--", "preflight"])
        .output()
        .expect("Failed to run preflight");

    assert_eq!(output.status.code(), Some(0), "Preflight should exit 0 for general status");
}

#[test]
fn test_preflight_specific_backend_exit_code() {
    // Checking specific backend should exit 1 if unavailable
    let output = Command::new("cargo")
        .args(&["run", "-p", "xtask", "--features", "crossval-all", "--", "preflight", "--backend", "llama"])
        .output()
        .expect("Failed to run preflight");

    if bitnet_crossval::HAS_LLAMA {
        assert_eq!(output.status.code(), Some(0), "Should exit 0 if llama available");
        assert!(String::from_utf8_lossy(&output.stdout).contains("✓"));
    } else {
        assert_eq!(output.status.code(), Some(1), "Should exit 1 if llama unavailable");
        assert!(String::from_utf8_lossy(&output.stderr).contains("NOT FOUND"));
    }
}

#[test]
fn test_preflight_verbose_shows_search_paths() {
    let output = Command::new("cargo")
        .args(&["run", "-p", "xtask", "--features", "crossval-all", "--", "preflight", "--backend", "llama", "--verbose"])
        .output()
        .expect("Failed to run preflight");

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should show environment variables section
    assert!(stdout.contains("Environment") || stdout.contains("BITNET_CPP_DIR"));

    // Should show search paths
    assert!(stdout.contains("Search Path") || stdout.contains("build"));
}

#[test]
fn test_preflight_invalid_backend_exit_2() {
    let output = Command::new("cargo")
        .args(&["run", "-p", "xtask", "--features", "crossval-all", "--", "preflight", "--backend", "invalid"])
        .output()
        .expect("Failed to run preflight");

    // clap returns exit code 2 for invalid arguments
    assert_ne!(output.status.code(), Some(0), "Should fail for invalid backend");
    assert!(String::from_utf8_lossy(&output.stderr).contains("invalid"));
}
```

**Manual Testing Checklist**:

1. **Scenario: No C++ installed**
   ```bash
   # Ensure no C++ libraries
   unset BITNET_CPP_DIR
   cargo clean -p xtask
   cargo build -p xtask --features crossval-all

   # Test error message
   cargo run -p xtask -- preflight --backend llama
   # Expected: Exit 1, clear error with recovery steps
   ```

2. **Scenario: C++ installed, no rebuild**
   ```bash
   # Install C++ (but don't rebuild xtask)
   eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"

   # Test stale detection
   cargo run -p xtask -- preflight --backend llama --verbose
   # Expected: Exit 1, message about rebuild requirement
   ```

3. **Scenario: C++ installed, after rebuild**
   ```bash
   # Rebuild xtask
   cargo clean -p xtask
   cargo build -p xtask --features crossval-all

   # Test success
   cargo run -p xtask -- preflight --backend llama --verbose
   # Expected: Exit 0, verbose diagnostics with ✓ markers
   ```

4. **Scenario: All backends check**
   ```bash
   cargo run -p xtask -- preflight
   # Expected: Exit 0, shows status of both backends
   ```

5. **Scenario: Invalid arguments**
   ```bash
   cargo run -p xtask -- preflight --backend invalid
   # Expected: Exit 2, clap error message
   ```

---

## 6. Risk Assessment and Mitigation

### 6.1 Technical Risks

**RISK-1: Error message too verbose, overwhelming users**
- **Impact**: Users skip reading error message, miss recovery steps
- **Likelihood**: Medium
- **Mitigation**:
  - Use visual separators to break up text
  - Put critical information first (rebuild requirement)
  - Provide "Option A" (recommended) vs "Option B" (manual) structure
  - Keep non-verbose mode concise (< 10 lines)

**RISK-2: Build timestamp detection unreliable**
- **Impact**: Staleness detection shows "unknown" instead of timestamp
- **Likelihood**: Low (std::fs::metadata is well-supported)
- **Mitigation**:
  - Gracefully handle None case with "unknown" fallback
  - Don't block on timestamp - it's informational only
  - Document that timestamp is best-effort

**RISK-3: Exit code change breaks existing CI/CD**
- **Impact**: CI pipelines relying on current exit codes fail
- **Likelihood**: Low (preflight is new in recent PRs)
- **Mitigation**:
  - Document exit codes in command help
  - Only enforce exit code 1 for --backend flag (specific check)
  - General check (no --backend) always exits 0 (backward compatible)

**RISK-4: RPATH explanation confuses users**
- **Impact**: Users don't understand tradeoff between rpath and LD_LIBRARY_PATH
- **Likelihood**: Medium
- **Mitigation**:
  - Provide "Option A" (recommended, uses rpath) upfront
  - Put "Option B" (manual LD_LIBRARY_PATH) as alternative
  - Explain tradeoff clearly: rpath=permanent, LD_LIBRARY_PATH=per-session

### 6.2 UX Risks

**RISK-5: Error message too technical for beginners**
- **Impact**: New contributors intimidated, don't contribute
- **Likelihood**: Medium
- **Mitigation**:
  - Use plain language ("install", "rebuild") instead of jargon
  - Explain acronyms inline (rpath = "runtime library search path")
  - Provide exact copy-paste commands (no placeholders)

**RISK-6: Recovery steps fail for edge cases**
- **Impact**: Users stuck even after following steps
- **Likelihood**: Low
- **Mitigation**:
  - Include troubleshooting section with --verbose hint
  - Reference docs/howto/cpp-setup.md for deep dives
  - Provide multiple recovery options (auto-setup vs manual)

### 6.3 Validation Plan

**Pre-Merge Validation**:
1. Run all unit tests (preflight.rs::tests)
2. Run all integration tests (preflight_integration.rs)
3. Manual test all 5 scenarios from section 5.3
4. Review error message templates for clarity (UX review)
5. Verify exit codes match specification

**Post-Merge Monitoring**:
1. Watch GitHub issues for user confusion about error messages
2. Monitor CI/CD exit code behavior (ensure no breakage)
3. Collect feedback on verbose diagnostics usefulness

---

## 7. Success Criteria

### 7.1 Functional Success

**SC-1: Error messages are actionable**
- PASS: User can fix "libraries not found" error in < 5 minutes by following error message
- MEASURE: Manual testing with 3 developers unfamiliar with BitNet.rs

**SC-2: Exit codes are standardized**
- PASS: All exit codes match specification (0=success, 1=unavailable, 2=invalid)
- MEASURE: Integration tests verify exit codes

**SC-3: Verbose diagnostics are comprehensive**
- PASS: Verbose mode shows environment vars, search paths, build metadata
- MEASURE: Manual inspection of --verbose output

**SC-4: HAS_BITNET/HAS_LLAMA constants are usable**
- PASS: Other crates can query constants and handle both true/false cases
- MEASURE: Unit tests verify constant access and branching

### 7.2 Non-Functional Success

**SC-5: Performance is acceptable**
- PASS: Preflight checks complete in < 100ms
- MEASURE: `time cargo run -p xtask -- preflight` benchmark

**SC-6: Documentation is clear**
- PASS: Command help text explains exit codes and usage
- MEASURE: Code review of --help output

**SC-7: Backward compatibility maintained**
- PASS: Existing CI/CD pipelines don't break
- MEASURE: No new GitHub issues about CI breakage after merge

### 7.3 Acceptance Tests

**AT-1: First-time user scenario**
```bash
# Setup: No C++ installed
unset BITNET_CPP_DIR
cargo clean -p xtask && cargo build -p xtask --features crossval-all

# Test: Run preflight
cargo run -p xtask -- preflight --backend llama

# Expected:
# - Exit code 1
# - Error message contains "CRITICAL: xtask must be REBUILT"
# - Error message contains exact setup command
# - Error message contains "Option A" and "Option B"
```

**AT-2: Just installed C++ scenario**
```bash
# Setup: Install C++ but don't rebuild xtask
eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"

# Test: Run preflight
cargo run -p xtask -- preflight --backend llama --verbose

# Expected:
# - Exit code 1 (stale detection)
# - Error message mentions rebuild requirement
# - Verbose output shows BITNET_CPP_DIR set but HAS_LLAMA=false
```

**AT-3: Successful validation scenario**
```bash
# Setup: Rebuild xtask after C++ install
cargo clean -p xtask && cargo build -p xtask --features crossval-all

# Test: Run preflight
cargo run -p xtask -- preflight --backend llama --verbose

# Expected:
# - Exit code 0
# - Output shows ✓ Backend 'llama.cpp': AVAILABLE
# - Verbose shows CROSSVAL_HAS_LLAMA=true
# - Verbose shows libraries found in search paths
```

**AT-4: CI integration scenario**
```bash
# Test: CI pipeline check
if cargo run -p xtask --features crossval-all -- preflight --backend llama; then
    echo "Cross-validation supported, running tests..."
else
    echo "Cross-validation unavailable, skipping tests..."
    exit 0  # Don't fail CI, just skip tests
fi

# Expected:
# - Exit code reflects availability (0=available, 1=unavailable)
# - CI can conditionally run tests based on exit code
```

---

## 8. Implementation Roadmap

### 8.1 Phase 1: Error Message Templates (Low Effort, High Impact)

**Scope**:
- Update error messages in preflight.rs (section 3.1)
- Add visual separators and structured formatting
- Add "Option A" vs "Option B" recovery paths

**Estimated Effort**: 2-3 hours

**Files Modified**:
- `xtask/src/crossval/preflight.rs` (lines 53-74, error message)

**Testing**:
- Manual test with no C++ installed
- Verify error message clarity

**Success Metric**:
- 3 developers can follow error message to successful setup in < 5 minutes

### 8.2 Phase 2: Verbose Diagnostics Enhancement (Medium Effort)

**Scope**:
- Update `print_verbose_success_diagnostics()` (section 3.2)
- Update `print_verbose_failure_diagnostics()` (section 3.3)
- Add build metadata section (timestamp, detection flags)
- Add search path formatting with existence checks

**Estimated Effort**: 3-4 hours

**Files Modified**:
- `xtask/src/crossval/preflight.rs` (lines 87-257, diagnostic functions)
- Add helper functions (get_xtask_build_timestamp, format_build_metadata)

**Testing**:
- Manual test --verbose with C++ installed
- Manual test --verbose without C++ installed
- Verify structured output and visual separators

**Success Metric**:
- Verbose output shows all required information (env vars, paths, metadata)
- Output is grep-able for CI/CD log analysis

### 8.3 Phase 3: Exit Code Standardization (Low Effort)

**Scope**:
- Document exit codes in command help
- Verify cpp_backend_preflight_cmd returns correct codes
- Add integration tests for exit codes

**Estimated Effort**: 1-2 hours

**Files Modified**:
- `xtask/src/main.rs` (Preflight command documentation)
- `xtask/src/crossval/backend.rs` (CppBackend documentation)

**Testing**:
- Integration tests (section 5.3, preflight_integration.rs)
- Manual test all scenarios with different exit codes

**Success Metric**:
- All exit codes match specification (0, 1, 2)
- CI can rely on exit codes for conditional logic

### 8.4 Phase 4: Documentation Enhancement (Low Effort)

**Scope**:
- Enhance HAS_BITNET/HAS_LLAMA documentation in crossval/src/lib.rs
- Add usage examples for constants
- Update build.rs comments explaining rpath

**Estimated Effort**: 1 hour

**Files Modified**:
- `crossval/src/lib.rs` (lines 18-30, constant documentation)
- `crossval/build.rs` (lines 89-93, rpath comments)

**Testing**:
- Code review of documentation clarity
- Verify examples compile

**Success Metric**:
- Other crates can use constants without reading source code
- rpath behavior is clear from comments

### 8.5 Total Estimated Effort

**Total**: 7-10 hours (1-2 days of focused work)

**Dependencies**:
- No external dependencies
- No API changes to other crates
- All changes are UX/documentation improvements

**Risks**:
- Low risk (no breaking changes)
- All changes are additive (enhanced diagnostics)

---

## 9. Alignment with BitNet.rs Principles

### 9.1 TDD Practices

**Alignment**:
- All changes have corresponding unit tests (preflight.rs::tests)
- Integration tests validate exit codes and output format
- Manual testing checklist ensures real-world usability

**Test Coverage**:
- Unit tests: Error message formatting, build metadata, exit behavior
- Integration tests: Exit codes, verbose output, error scenarios
- Manual tests: 5 scenarios covering first-time users, stale detection, success

### 9.2 Feature-Gated Architecture

**Alignment**:
- Preflight command only available with `--features crossval-all` (or `crossval`)
- HAS_BITNET/HAS_LLAMA constants use build-time feature detection
- No changes to feature gate structure (already correct)

**Build Commands**:
```bash
# Build with cross-validation support (required for preflight)
cargo build -p xtask --features crossval-all

# Build without cross-validation (preflight not available)
cargo build -p xtask --no-default-features
```

### 9.3 Workspace Structure

**Alignment**:
- Changes isolated to `xtask/src/crossval/` module (no workspace boundary violations)
- crossval crate exports constants correctly for xtask consumption
- No circular dependencies introduced

**Crate Boundaries**:
- `crossval/build.rs`: Build-time library detection (no changes)
- `crossval/src/lib.rs`: Runtime constant exports (documentation only)
- `xtask/src/crossval/preflight.rs`: Diagnostic logic (UX improvements)

### 9.4 Production-Grade Error Handling

**Alignment**:
- Error messages provide actionable recovery steps (not just error codes)
- Verbose diagnostics help debug complex issues (multi-path C++ installs)
- Exit codes enable CI/CD integration (fail fast with clear status)

**Error Handling Patterns**:
- Use anyhow::bail! for error messages (consistent with xtask patterns)
- Provide context-specific error messages (backend name, required libs)
- Include next steps in all error messages (never dead-end errors)

---

## 10. References

### 10.1 Codebase References

**Key Files**:
- `xtask/src/crossval/preflight.rs`: Preflight validation logic
- `xtask/src/crossval/backend.rs`: Backend selection and metadata
- `xtask/src/main.rs`: CLI command definitions and handlers
- `crossval/src/lib.rs`: HAS_BITNET/HAS_LLAMA constant exports
- `crossval/build.rs`: Build-time library detection

**Related Documentation**:
- `docs/howto/cpp-setup.md`: C++ reference setup guide
- `docs/explanation/dual-backend-crossval.md`: Architecture overview
- `/tmp/xtask-crossval-exploration.md`: Analysis of current implementation

### 10.2 External References

**Build-Time Detection**:
- Cargo build script documentation: https://doc.rust-lang.org/cargo/reference/build-scripts.html
- option_env! macro: https://doc.rust-lang.org/std/macro.option_env.html

**Dynamic Linking**:
- rpath documentation: https://en.wikipedia.org/wiki/Rpath
- LD_LIBRARY_PATH: https://tldp.org/HOWTO/Program-Library-HOWTO/shared-libraries.html

### 10.3 BitNet.rs Patterns

**Error Message Patterns** (existing examples):
- `bitnet-cli/src/main.rs`: CLI error messages with recovery steps
- `bitnet-models/src/validation.rs`: Validation error messages with context
- `xtask/src/main.rs`: Build error messages with actionable guidance

**Testing Patterns**:
- Unit tests: `bitnet-quantization/src/tests.rs`
- Integration tests: `tests/integration/inference_tests.rs`
- Manual testing: `scripts/validate_gguf.sh`

---

## 11. Appendix: Example Output (Full)

### A.1 Error Output (No C++ Installed)

**Command**: `cargo run -p xtask -- preflight --backend llama`

**Full Output**:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ Backend 'llama.cpp' libraries NOT FOUND
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CRITICAL: Library detection happens at BUILD time, not runtime.
If you just installed C++ libraries, xtask MUST be rebuilt to detect them.

Required libraries: libllama*.so, libggml*.so
Auto-detected backend: llama.cpp (from model path heuristics)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RECOVERY STEPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Option A: One-Command Setup (Recommended for First-Time Users)
─────────────────────────────────────────────────────────────────────

  Step 1: Install and configure C++ reference implementation
    eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"

  Step 2: Rebuild xtask to detect newly installed libraries
    cargo clean -p xtask && cargo build -p xtask --features crossval-all

  Step 3: Verify detection succeeded
    cargo run -p xtask -- preflight --backend llama --verbose

  Then retry your original command.

Option B: Manual Setup + LD_LIBRARY_PATH (No Rebuild Required)
─────────────────────────────────────────────────────────────────────

  Step 1: Install llama.cpp manually (skip if already installed)
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp && cmake -B build && cmake --build build

  Step 2: Set environment variable for this session
    export BITNET_CPP_DIR=/path/to/llama.cpp
    export LD_LIBRARY_PATH=$BITNET_CPP_DIR/build:$LD_LIBRARY_PATH  # Linux
    export DYLD_LIBRARY_PATH=$BITNET_CPP_DIR/build:$DYLD_LIBRARY_PATH  # macOS

  Step 3: Rebuild xtask to embed library paths (rpath)
    cargo clean -p xtask && cargo build -p xtask --features crossval-all

  Note: Option B requires setting LD_LIBRARY_PATH before EVERY run.
        Option A embeds library paths permanently (rpath).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TROUBLESHOOTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

If setup fails, run verbose diagnostics to see what's happening:
  cargo run -p xtask -- preflight --backend llama --verbose

This will show:
  • Environment variables checked (BITNET_CPP_DIR, LD_LIBRARY_PATH, etc.)
  • Library search paths in priority order
  • Which paths exist vs missing
  • All libraries found in each path
  • Build-time detection flags

For more help, see:
  docs/howto/cpp-setup.md (Detailed C++ setup guide)
  docs/explanation/dual-backend-crossval.md (Architecture overview)

Error: Backend 'llama.cpp' selected but required libraries not found.
```

### A.2 Verbose Success Output

**Command**: `cargo run -p xtask -- preflight --backend llama --verbose`

**Full Output** (when libraries available):
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Backend 'llama.cpp': AVAILABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Environment Configuration
─────────────────────────────────────────────────────────────────────
  BITNET_CPP_DIR        = /home/user/.cache/bitnet_cpp
  BITNET_CROSSVAL_LIBDIR = (not set)
  LD_LIBRARY_PATH       = /home/user/.cache/bitnet_cpp/build:/usr/local/lib
  BITNET_CPP_PATH       = (not set - deprecated, use BITNET_CPP_DIR)

Library Search Paths (Priority Order)
─────────────────────────────────────────────────────────────────────
  1. BITNET_CROSSVAL_LIBDIR override
     (not set - using default search order)

  2. BITNET_CPP_DIR/build
     ✓ /home/user/.cache/bitnet_cpp/build (exists)
     Found libraries:
       - libllama.so.1.0.0
       - libggml.so.1.0.0

  3. BITNET_CPP_DIR/build/lib
     ✗ /home/user/.cache/bitnet_cpp/build/lib (not found)

  4. BITNET_CPP_DIR/build/3rdparty/llama.cpp/src
     ✗ /home/user/.cache/bitnet_cpp/build/3rdparty/llama.cpp/src (not found)

  5. BITNET_CPP_DIR/build/3rdparty/llama.cpp/ggml/src
     ✗ /home/user/.cache/bitnet_cpp/build/3rdparty/llama.cpp/ggml/src (not found)

  6. BITNET_CPP_DIR/lib
     ✗ /home/user/.cache/bitnet_cpp/lib (not found)

Required Libraries for llama.cpp Backend
─────────────────────────────────────────────────────────────────────
  ✓ libllama.so (found in build/)
  ✓ libggml.so (found in build/)

Build-Time Detection Metadata
─────────────────────────────────────────────────────────────────────
  CROSSVAL_HAS_LLAMA = true
  CROSSVAL_HAS_BITNET = false
  Last xtask build: 2025-10-25T10:30:00Z
  Build feature flags: crossval-all
  Linked libraries: libllama, libggml (dynamic linking)
  Runtime library resolution: rpath embedded

Platform-Specific Configuration
─────────────────────────────────────────────────────────────────────
  Platform: Linux
  Standard library: libstdc++ (dynamic linking)
  RPATH embedded: YES (no LD_LIBRARY_PATH required)
  Loader search order: rpath → LD_LIBRARY_PATH → system paths

Summary
─────────────────────────────────────────────────────────────────────
✓ All required libraries detected at build time
✓ Runtime library resolution configured (rpath)
✓ Cross-validation with llama.cpp is supported

To test cross-validation:
  cargo run -p xtask --features crossval-all -- crossval-per-token \
    --model models/model.gguf \
    --tokenizer models/tokenizer.json \
    --prompt "Test" \
    --max-tokens 4 \
    --cpp-backend llama \
    --verbose
```

---

## 12. Conclusion

This specification defines comprehensive UX improvements to the preflight diagnostics system, transforming it from a basic availability check into a production-grade diagnostic tool. The improvements focus on:

1. **Clarity**: Error messages explicitly distinguish build-time vs runtime detection
2. **Actionability**: Every error includes exact recovery commands
3. **Debuggability**: Verbose mode shows complete diagnostic information
4. **CI/CD Integration**: Standardized exit codes enable automated workflows

The implementation is low-risk (UX-only changes), low-effort (7-10 hours), and high-impact (eliminates common user confusion). All changes align with BitNet.rs principles of TDD practices, feature-gated architecture, and production-grade error handling.

**Next Steps**:
1. Review specification with maintainers
2. Implement Phase 1 (error message templates) - 2-3 hours
3. Implement Phase 2 (verbose diagnostics) - 3-4 hours
4. Implement Phase 3 (exit codes) - 1-2 hours
5. Implement Phase 4 (documentation) - 1 hour
6. Submit PR with comprehensive tests and examples

# BitNet.cpp Auto-Setup Parity Specification

**Version:** 2.0.0
**Status:** Comprehensive Implementation Ready
**Created:** 2025-10-26
**Component:** `xtask/src/cpp_setup_auto.rs`, `xtask/build.rs`, `xtask/src/build_helpers.rs`
**Feature Scope:** Cross-validation infrastructure (requires `crossval`, `crossval-all`, or `ffi` features)
**Priority:** High (Blocks dual-backend cross-validation)
**Based on Analysis**: `/tmp/cpp_setup_auto_analysis.md`, `/tmp/rpath_library_discovery_analysis.md`

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Feature Requirements (Acceptance Criteria)](#2-feature-requirements-acceptance-criteria)
3. [Current vs Target State](#3-current-vs-target-state-gap-analysis)
4. [Architecture](#4-architecture)
5. [API Design](#5-api-design)
6. [Build Detection](#6-build-detection)
7. [Library Discovery](#7-library-discovery)
8. [RPATH Merging](#8-rpath-merging)
9. [Testing Strategy](#9-testing-strategy)
10. [Implementation Phases](#10-implementation-phases)
11. [Dependencies](#11-dependencies)
12. [Migration](#12-migration)
13. [Verification Criteria](#13-verification-criteria)
14. [Documentation Updates](#14-documentation-updates)
15. [Risks and Mitigations](#15-risks-and-mitigations)

---

## 1. Executive Summary

### 1.1 Problem Statement

**Current Limitation**: The `setup-cpp-auto` command hardcodes BitNet.cpp installation, making it impossible to:
- Install standalone llama.cpp for LLaMA model validation
- Support dual-backend workflows requiring both BitNet and llama libraries
- Selectively install backends in CI/CD for optimized build times
- Test against different llama.cpp versions without full BitNet.cpp rebuild

**User Impact**:
```bash
# Current (no backend selection):
eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"  # Always installs BitNet.cpp

# Desired (explicit backend control):
eval "$(cargo run -p xtask -- setup-cpp-auto --backend llama --emit=sh)"  # Standalone llama
eval "$(cargo run -p xtask -- setup-cpp-auto --backend bitnet --emit=sh)" # BitNet only
```

### 1.2 Goals

**Functional Goals**:
1. **Backend Selection**: `--backend bitnet|llama` flag with auto-detection fallback
2. **Dual-Backend Support**: Install both backends with merged RPATH (AC7-AC9)
3. **Standalone llama.cpp**: Independent llama.cpp installation without BitNet.cpp dependency (AC10-AC12)
4. **Advanced Auto-Repair**: Retry logic, concurrent locking, transactional state (AC18-AC22)
5. **CI Integration**: Automated testing for all backend configurations (AC13-AC17)

**Non-Functional Goals**:
1. **Backward Compatibility**: Existing workflows unaffected (defaults to BitNet.cpp)
2. **Zero Downtime**: Graceful fallback if backends unavailable
3. **Developer Ergonomics**: Clear error messages, progress indicators, auto-repair by default
4. **Platform Coverage**: Linux/macOS/Windows with platform-specific optimizations

### 1.3 Key Deliverables

- **Backend selection flag**: `--backend bitnet|llama` (defaults to `bitnet`)
- **Dual-backend RPATH merging**: Support both backends with unified library discovery
- **Standalone llama.cpp**: Independent installation without BitNet.cpp dependency
- **Advanced reliability**: Retry logic, concurrent locking, transactional rollback
- **Comprehensive testing**: Unit tests, integration tests, CI validation

---

## 2. Feature Requirements (Acceptance Criteria)

### 2.1 Core Backend Selection (AC1-AC6)

**AC1: Backend Selection Flag**

**Requirement**: Add `--backend bitnet|llama` flag to `setup-cpp-auto` command with sensible defaults

```bash
# Explicit backend selection
cargo run -p xtask -- setup-cpp-auto --backend bitnet --emit=sh
cargo run -p xtask -- setup-cpp-auto --backend llama --emit=sh

# Default behavior (backward compatible)
cargo run -p xtask -- setup-cpp-auto --emit=sh  # Defaults to bitnet
```

**Status**: ❌ NOT IMPLEMENTED
**Effort**: 3-4 hours (medium complexity)
**Dependencies**: CLI parser update in `xtask/src/main.rs`

---

**AC2: Clone Detection**

**Requirement**: Detect existing backend installations and skip clone if already present

- Detect existing BitNet.cpp installation at `~/.cache/bitnet_cpp/`
- Detect existing llama.cpp installation at `~/.cache/llama_cpp/`
- Skip clone if directory exists, update via `git pull --ff-only`
- Handle detached HEAD gracefully (non-fatal warning)

**Status**: ✅ PARTIAL (BitNet.cpp only)
**Effort**: 2-3 hours (extend existing logic)

---

**AC3: Build Strategy Selection**

**Requirement**: Per-backend build method selection with fallback

| Backend | Primary Build Method | Fallback |
|---------|---------------------|----------|
| BitNet.cpp | `setup_env.py` (if available) | CMake |
| llama.cpp | CMake only | (none) |

**Implementation**:
- BitNet.cpp: Try `setup_env.py`, fallback to CMake on failure
- llama.cpp: CMake only (no `setup_env.py`)
- Both: Build vendored llama.cpp in `3rdparty/` (BitNet.cpp only)

**Status**: ✅ COMPLETE (BitNet.cpp), ❌ NOT IMPLEMENTED (llama.cpp)
**Effort**: 2-3 hours (refactor build functions)

---

**AC4: Library Discovery (Three-Tier Hierarchy)**

**Requirement**: Comprehensive library search with three-tier hierarchy

**Tier 1: Backend-Specific Primary Locations**
```
BitNet.cpp:
  - build/bin/           ← libbitnet.so
  - build/lib/
  - build/3rdparty/llama.cpp/build/bin/  ← vendored llama

llama.cpp:
  - build/               ← libllama.so, libggml.so
  - build/bin/
  - build/lib/
```

**Tier 2: Fallback Locations**
```
- build/
- lib/
```

**Tier 3: Environment Variable Override**
```
BITNET_CROSSVAL_LIBDIR (legacy, single path)
CROSSVAL_RPATH_BITNET  (BitNet-specific)
CROSSVAL_RPATH_LLAMA   (llama-specific)
```

**Status**: ✅ PARTIAL (BitNet.cpp tier 1+2), ❌ NOT IMPLEMENTED (llama.cpp, tier 3)
**Effort**: 4-5 hours (extend search logic, add deduplication)

---

**AC5: RPATH Embedding**

**Requirement**: Embed discovered library paths into xtask binary via linker RPATH

- Embed discovered library paths via `-Wl,-rpath,{path}` on Linux/macOS
- Emit PATH instructions on Windows (no RPATH support)
- Deduplicate paths using canonical resolution (resolve symlinks)
- Validate RPATH length ≤ 4096 bytes (linker limit)

**Status**: ✅ COMPLETE (single-path), ❌ NOT IMPLEMENTED (multi-path merging)
**Effort**: 3-4 hours (implement `merge_and_deduplicate`, see `rpath-merging-strategy.md`)

---

**AC6: Shell Export Formats**

**Requirement**: Support all 4 shell formats with platform-aware exports

| Shell | Export Syntax | Platform |
|-------|---------------|----------|
| sh/bash/zsh | `export VAR="value"` | Linux/macOS |
| fish | `set -gx VAR "value"` | Linux/macOS |
| pwsh | `$env:VAR = "value"` | Windows |
| cmd | `set VAR=value` | Windows |

**Status**: ✅ COMPLETE
**Effort**: 0 hours (already implemented)

---

### 2.2 Dual-Backend Support (AC7-AC9)

**AC7: Backend Configuration Struct**

**Requirement**: Define backend configuration abstraction for code reuse

```rust
pub struct BackendConfig {
    pub backend: CppBackend,           // bitnet or llama
    pub repo_url: &'static str,        // GitHub URL
    pub install_subdir: &'static str,  // bitnet_cpp or llama_cpp
    pub build_method: BuildMethod,     // SetupEnvPy or CMakeOnly
}

impl BackendConfig {
    pub fn for_backend(backend: CppBackend) -> Self { /* ... */ }
}
```

**Status**: ❌ NOT IMPLEMENTED
**Effort**: 2-3 hours (define struct, implement factory methods)

---

**AC8: Dual-Backend Installation**

**Requirement**: Enable simultaneous installation of both backends with merged RPATH

```bash
# Install both backends in separate directories
eval "$(cargo run -p xtask -- setup-cpp-auto --backend bitnet --emit=sh)"
eval "$(cargo run -p xtask -- setup-cpp-auto --backend llama --emit=sh)"

# Rebuild xtask with merged RPATH
cargo clean -p xtask && cargo build -p xtask --features crossval-all

# Verify merged RPATH
readelf -d target/debug/xtask | grep RPATH
# Expected: Library rpath: [~/.cache/bitnet_cpp/build/bin:~/.cache/llama_cpp/build]
```

**Status**: ❌ NOT IMPLEMENTED
**Effort**: 5-6 hours (coordinate with RPATH merging, test dual installation)

---

**AC9: RPATH Merging Strategy**

**Requirement**: Merge multiple RPATH entries with deduplication

**Priority Order**:
1. **`BITNET_CROSSVAL_LIBDIR`** (legacy single-path override, highest)
2. **`CROSSVAL_RPATH_BITNET` + `CROSSVAL_RPATH_LLAMA`** (granular merge)
3. **Auto-discovery from `BITNET_CPP_DIR` and `LLAMA_CPP_DIR`** (lowest)

**Algorithm**:
- Canonicalize paths (resolves symlinks, normalizes case on macOS)
- Deduplicate using HashSet
- Join with `:` separator (POSIX RPATH syntax)
- Validate total length ≤ 4096 bytes

**Status**: ❌ NOT IMPLEMENTED (specified in `rpath-merging-strategy.md`)
**Effort**: 4-5 hours (implement `merge_and_deduplicate` in `build_helpers.rs`)

---

### 2.3 Standalone llama.cpp Support (AC10-AC12)

**AC10: llama.cpp-Specific GitHub URL**

**Requirement**: Support standalone llama.cpp installation from official repository

```rust
const LLAMA_REPO_URL: &str = "https://github.com/ggerganov/llama.cpp";
```

**Status**: ❌ NOT IMPLEMENTED
**Effort**: 1 hour (add constant, update clone logic)

---

**AC11: llama.cpp Build Method**

**Requirement**: CMake-only build for llama.cpp (no `setup_env.py`)

```rust
fn build_llama_cpp(install_dir: &Path) -> Result<()> {
    run_cmake_build(install_dir)?;  // No setup_env.py fallback
    Ok(())
}
```

**Status**: ❌ NOT IMPLEMENTED
**Effort**: 2-3 hours (refactor build functions)

---

**AC12: llama.cpp Library Discovery**

**Requirement**: llama.cpp-specific library search paths

```rust
// llama.cpp search paths (different from BitNet.cpp)
let llama_candidates = vec![
    install_dir.join("build"),           // Top-level build
    install_dir.join("build/bin"),       // CMake bin output
    install_dir.join("build/lib"),       // CMake lib output
];
```

**Status**: ❌ NOT IMPLEMENTED
**Effort**: 2-3 hours (extend `find_lib_dirs` function)

---

### 2.4 Integration Tests (AC13-AC17)

**AC13: Unit Test Coverage**

**Requirement**: Comprehensive unit tests for all new functions

```rust
// xtask/tests/cpp_setup_auto_backend_selection.rs
#[test]
fn test_backend_bitnet_clone() { /* ... */ }

#[test]
fn test_backend_llama_clone() { /* ... */ }

#[test]
fn test_dual_backend_rpath_merge() { /* ... */ }

#[test]
fn test_library_discovery_tier1_bitnet() { /* ... */ }

#[test]
fn test_library_discovery_tier1_llama() { /* ... */ }
```

**Status**: ⚠️ PARTIAL (12 unit tests for BitNet.cpp, none for llama.cpp)
**Effort**: 6-8 hours (comprehensive test suite)

---

**AC14: Integration Test Workflow**

**Requirement**: End-to-end test script for all backend configurations

```bash
# Test matrix:
# - Backend: bitnet, llama, both
# - Platform: Linux, macOS, Windows
# - Scenario: fresh install, update, rebuild

# Smoke test
scripts/test_cpp_setup_all_backends.sh
```

**Status**: ❌ NOT IMPLEMENTED
**Effort**: 4-5 hours (create test harness script)

---

**AC15: CI Validation**

**Requirement**: GitHub Actions workflow for automated testing

```yaml
# .github/workflows/crossval.yml
- name: Test BitNet.cpp auto-setup
  run: |
    eval "$(cargo run -p xtask -- setup-cpp-auto --backend bitnet --emit=sh)"
    cargo clean -p xtask && cargo build -p xtask --features crossval-all
    cargo run -p xtask -- preflight --backend bitnet

- name: Test llama.cpp auto-setup
  run: |
    eval "$(cargo run -p xtask -- setup-cpp-auto --backend llama --emit=sh)"
    cargo clean -p xtask && cargo build -p xtask --features crossval-all
    cargo run -p xtask -- preflight --backend llama
```

**Status**: ❌ NOT IMPLEMENTED
**Effort**: 3-4 hours (update CI workflow)

---

**AC16: Platform-Specific Testing**

**Requirement**: Validate RPATH embedding on all platforms

- **Linux**: Test RPATH embedding with `readelf -d`
- **macOS**: Test RPATH embedding with `otool -l`
- **Windows**: Test PATH instructions (no RPATH)

**Status**: ❌ NOT IMPLEMENTED
**Effort**: 3-4 hours (platform-specific validation scripts)

---

**AC17: Regression Test Suite**

**Requirement**: Ensure backward compatibility

```bash
# Ensure backward compatibility
export BITNET_CROSSVAL_LIBDIR=/tmp/legacy_test
cargo run -p xtask -- setup-cpp-auto --emit=sh
# Should still use legacy path

# Ensure BITNET_CPP_DIR fallback works
unset BITNET_CROSSVAL_LIBDIR
export BITNET_CPP_DIR=~/.cache/bitnet_cpp
cargo run -p xtask -- setup-cpp-auto --emit=sh
# Should auto-discover from BITNET_CPP_DIR/build/bin
```

**Status**: ❌ NOT IMPLEMENTED
**Effort**: 2-3 hours (regression test harness)

---

### 2.5 Advanced Features (AC18-AC22)

**AC18: Retry Logic with Exponential Backoff**

**Requirement**: Retry network operations with exponential backoff

```rust
// Retry network operations with exponential backoff
fn clone_with_retry(url: &str, dest: &Path, max_retries: u32) -> Result<()> {
    for attempt in 1..=max_retries {
        match run_git_clone(url, dest) {
            Ok(()) => return Ok(()),
            Err(e) if is_transient_error(&e) && attempt < max_retries => {
                let backoff_ms = 1000 * 2u64.pow(attempt - 1);  // 1s, 2s, 4s
                eprintln!("[bitnet] Clone failed (attempt {}/{}), retrying in {}ms...",
                          attempt, max_retries, backoff_ms);
                std::thread::sleep(Duration::from_millis(backoff_ms));
            }
            Err(e) => return Err(e),
        }
    }
    unreachable!()
}
```

**Status**: ❌ NOT IMPLEMENTED
**Effort**: 3-4 hours (implement retry wrapper functions)

---

**AC19: Concurrent Locking**

**Requirement**: Prevent concurrent installations with file-based locking

```rust
// Use file-based locking to prevent concurrent installs
use fs2::FileExt;

fn acquire_lock(backend: CppBackend) -> Result<File> {
    let lock_path = dirs::cache_dir()
        .ok_or_else(|| anyhow!("No cache directory"))?
        .join(format!("{}_setup.lock", backend.name()));

    let lock_file = OpenOptions::new()
        .write(true)
        .create(true)
        .open(&lock_path)?;

    lock_file.try_lock_exclusive()
        .context("Another setup-cpp-auto process is running")?;

    Ok(lock_file)
}
```

**Status**: ❌ NOT IMPLEMENTED
**Effort**: 2-3 hours (add fs2 dependency, implement locking)

---

**AC20: Transactional State Management**

**Requirement**: Atomic rollback on installation failure

```rust
// Track installation state with atomic rollback
pub struct InstallTransaction {
    backend: CppBackend,
    install_dir: PathBuf,
    state_file: PathBuf,
    committed: bool,
}

impl InstallTransaction {
    pub fn begin(backend: CppBackend) -> Result<Self> { /* ... */ }
    pub fn commit(mut self) -> Result<()> { self.committed = true; Ok(()) }
}

impl Drop for InstallTransaction {
    fn drop(&mut self) {
        if !self.committed {
            eprintln!("[bitnet] Rolling back incomplete installation...");
            let _ = fs::remove_dir_all(&self.install_dir);
        }
    }
}
```

**Status**: ❌ NOT IMPLEMENTED
**Effort**: 4-5 hours (design state machine, implement rollback)

---

**AC21: Rebuild Triggers**

**Requirement**: Emit cargo rebuild triggers for new environment variables

```rust
// Emit cargo:rerun-if-env-changed for new variables
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=BITNET_CROSSVAL_LIBDIR");
    println!("cargo:rerun-if-env-changed=BITNET_CPP_DIR");
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_DIR");           // NEW
    println!("cargo:rerun-if-env-changed=CROSSVAL_RPATH_BITNET");  // NEW
    println!("cargo:rerun-if-env-changed=CROSSVAL_RPATH_LLAMA");   // NEW
}
```

**Status**: ⚠️ PARTIAL (BITNET_CPP_DIR only)
**Effort**: 1 hour (add missing triggers)

---

**AC22: Progress Indicators**

**Requirement**: Show progress with indicatif crate

```rust
// Show progress with indicatif crate
use indicatif::{ProgressBar, ProgressStyle};

fn clone_with_progress(url: &str, dest: &Path) -> Result<()> {
    let pb = ProgressBar::new_spinner();
    pb.set_style(ProgressStyle::default_spinner()
        .template("{spinner:.green} [{elapsed_precise}] {msg}")
        .unwrap());

    pb.set_message(format!("Cloning from {}...", url));
    let result = run_git_clone(url, dest);
    pb.finish_with_message(format!("Clone complete: {}", dest.display()));

    result
}
```

**Status**: ❌ NOT IMPLEMENTED
**Effort**: 2-3 hours (add indicatif dependency, wrap operations)

---

## 3. Current vs Target State (Gap Analysis)

### 3.1 Completion Matrix

| Category | Current | Target | Gap | Effort (hours) |
|----------|---------|--------|-----|----------------|
| **Backend Selection** | Hardcoded BitNet | `--backend bitnet\|llama` | High | 3-4 |
| **Clone/Update Logic** | BitNet.cpp only | Both backends | Medium | 2-3 |
| **Build Detection** | setup_env.py + CMake | Per-backend strategy | Medium | 2-3 |
| **Library Discovery** | Tier 1+2 BitNet | Tier 1+2+3 both | Medium | 4-5 |
| **RPATH Embedding** | Single-path | Multi-path merged | Medium | 4-5 |
| **Shell Exports** | 4 formats | (same) | None | 0 |
| **Unit Tests** | 12 (BitNet) | 30+ (both) | High | 6-8 |
| **Integration Tests** | None | Full matrix | High | 4-5 |
| **CI Validation** | None | Platform matrix | High | 3-4 |
| **Retry Logic** | None | Exponential backoff | Low | 3-4 |
| **Concurrent Locking** | None | File-based locks | Low | 2-3 |
| **Transactional State** | None | Atomic rollback | Medium | 4-5 |
| **Progress Indicators** | eprintln! | indicatif bars | Low | 2-3 |
| **Documentation** | Scattered | Unified | Medium | 3-4 |

**Total Effort**: 43-56 hours (~1-1.5 weeks)

### 3.2 Critical Path

**Phase 1 (Critical Blockers - 12-15 hours)**:
1. Backend selection flag (AC1) - 3-4 hours
2. Standalone llama.cpp support (AC10-AC12) - 5-6 hours
3. RPATH merging (AC9) - 4-5 hours

**Phase 2 (Parity Features - 15-20 hours)**:
4. Dual-backend library discovery (AC4, AC8) - 6-8 hours
5. Integration tests (AC13-AC14) - 6-8 hours
6. CI validation (AC15-AC17) - 3-4 hours

**Phase 3 (Advanced Features - 11-15 hours)**:
7. Retry logic (AC18) - 3-4 hours
8. Concurrent locking (AC19) - 2-3 hours
9. Transactional state (AC20) - 4-5 hours
10. Progress indicators (AC22) - 2-3 hours

**Phase 4 (Polish - 5-6 hours)**:
11. Documentation updates - 3-4 hours
12. Regression test suite (AC17) - 2-3 hours

---

## 4. Architecture

### 4.1 Backend Configuration

```rust
// xtask/src/crossval/backend.rs (extend existing CppBackend enum)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CppBackend {
    BitNet,
    Llama,
}

impl CppBackend {
    pub fn name(&self) -> &'static str {
        match self {
            CppBackend::BitNet => "bitnet.cpp",
            CppBackend::Llama => "llama.cpp",
        }
    }

    pub fn repo_url(&self) -> &'static str {
        match self {
            CppBackend::BitNet => "https://github.com/microsoft/BitNet",
            CppBackend::Llama => "https://github.com/ggerganov/llama.cpp",
        }
    }

    pub fn default_install_dir(&self) -> PathBuf {
        let home = dirs::home_dir().expect("no home directory");
        match self {
            CppBackend::BitNet => home.join(".cache/bitnet_cpp"),
            CppBackend::Llama => home.join(".cache/llama_cpp"),
        }
    }

    pub fn build_method(&self) -> BuildMethod {
        match self {
            CppBackend::BitNet => BuildMethod::SetupEnvPyWithFallback,
            CppBackend::Llama => BuildMethod::CMakeOnly,
        }
    }
}

pub enum BuildMethod {
    SetupEnvPyWithFallback,  // Try setup_env.py, fallback to CMake
    CMakeOnly,                // CMake only (llama.cpp)
}
```

### 4.2 Auto-Setup Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Parse Command-Line Arguments                       │
├─────────────────────────────────────────────────────────────┤
│ • --backend bitnet|llama (default: bitnet)                 │
│ • --emit sh|fish|pwsh|cmd (default: sh)                    │
│ • Determine backend configuration                          │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Acquire Lock (prevent concurrent installs)         │
├─────────────────────────────────────────────────────────────┤
│ • Create ~/.cache/{backend}_setup.lock                     │
│ • Try exclusive lock (fail-fast if locked)                 │
│ • Release on drop (RAII pattern)                           │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Determine Installation Directory                   │
├─────────────────────────────────────────────────────────────┤
│ • Check {BACKEND}_CPP_DIR env var override                 │
│ • Fallback: ~/.cache/{backend}_cpp/                        │
│ • Create parent directories if needed                      │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: Clone or Update Repository                         │
├─────────────────────────────────────────────────────────────┤
│ • If directory exists:                                      │
│   ├─ git pull --ff-only (update)                           │
│   └─ git submodule update --init --recursive               │
│ • If directory missing:                                     │
│   ├─ git clone --depth=1 --recurse-submodules              │
│   └─ Retry with exponential backoff (AC18)                 │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 5: Build C++ Libraries                                │
├─────────────────────────────────────────────────────────────┤
│ • BitNet.cpp:                                               │
│   ├─ Try setup_env.py (if available)                       │
│   ├─ Fallback to CMake                                     │
│   └─ Build vendored llama.cpp (3rdparty/)                  │
│ • llama.cpp:                                                │
│   └─ CMake only (no setup_env.py)                          │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 6: Library Discovery (Three-Tier Hierarchy)           │
├─────────────────────────────────────────────────────────────┤
│ • Tier 1: Backend-specific primary paths                   │
│ • Tier 2: Fallback generic paths                           │
│ • Tier 3: Environment variable overrides                   │
│ • Deduplication via canonicalize()                         │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 7: Emit Shell-Specific Exports                        │
├─────────────────────────────────────────────────────────────┤
│ • {BACKEND}_CPP_DIR (installation directory)               │
│ • BITNET_CROSSVAL_LIBDIR (if auto-discovered)             │
│ • LD_LIBRARY_PATH (Linux)                                  │
│ • DYLD_LIBRARY_PATH (macOS)                                │
│ • PATH (Windows)                                            │
│ • Confirmation message                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. API Design

### 5.1 setup_cpp_auto Function Signature

```rust
// xtask/src/cpp_setup_auto.rs

/// Run auto-setup for C++ backend (clone, build, emit exports)
pub fn run(backend: CppBackend, emit: Emit) -> Result<()> {
    // Step 1: Acquire lock
    let _lock = acquire_lock(backend)?;

    // Step 2: Determine installation directory
    let install_dir = determine_install_dir(backend)?;

    // Step 3: Clone or update
    install_or_update_backend(backend, &install_dir)?;

    // Step 4: Build
    build_backend(backend, &install_dir)?;

    // Step 5: Discover libraries
    let lib_dirs = find_lib_dirs(backend, &install_dir)?;

    // Step 6: Emit shell exports
    emit_exports(emit, backend, &install_dir, &lib_dirs)?;

    Ok(())
}

fn determine_install_dir(backend: CppBackend) -> Result<PathBuf> {
    let env_var = match backend {
        CppBackend::BitNet => "BITNET_CPP_DIR",
        CppBackend::Llama => "LLAMA_CPP_DIR",
    };

    if let Ok(dir) = env::var(env_var) {
        Ok(PathBuf::from(dir))
    } else {
        Ok(backend.default_install_dir())
    }
}
```

---

## 6. Build Detection

### 6.1 Platform-Specific Strategies

| Platform | Detection Method | Fallback |
|----------|-----------------|----------|
| **Linux** | setup_env.py → CMake | Manual build |
| **macOS** | setup_env.py → CMake | Manual build |
| **Windows** | CMake only | Manual build |

---

## 7. Library Discovery

### 7.1 Three-Tier Hierarchy

**See AC4 for complete specification**

---

## 8. RPATH Merging

**See AC9 and `docs/specs/rpath-merging-strategy.md` for complete specification**

---

## 9. Testing Strategy

**See AC13-AC17 for complete testing requirements**

---

## 10. Implementation Phases

**See Section 3.2 Critical Path for detailed implementation timeline**

---

## 11. Dependencies

### 11.1 Rust Crate Dependencies

```toml
# xtask/Cargo.toml
[dependencies]
anyhow = "1.0"
clap = { version = "4.0", features = ["derive"] }
dirs = "5.0"
fs2 = "0.4"          # NEW: File locking (AC19)
indicatif = "0.17"   # NEW: Progress bars (AC22)
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tempfile = "3.0"     # For tests
walkdir = "2.0"
```

---

## 12. Migration

### 12.1 Breaking Changes

**None** - Implementation preserves full backward compatibility.

### 12.2 Recommended Workflow Updates

**Old Workflow** (manual dual-backend setup):
```bash
# Manual llama.cpp installation
git clone https://github.com/ggerganov/llama.cpp ~/.cache/llama_cpp
cmake -S ~/.cache/llama_cpp -B ~/.cache/llama_cpp/build
cmake --build ~/.cache/llama_cpp/build
```

**New Workflow** (automated):
```bash
# Setup both backends (automatic RPATH merging)
eval "$(cargo run -p xtask -- setup-cpp-auto --backend bitnet --emit=sh)"
eval "$(cargo run -p xtask -- setup-cpp-auto --backend llama --emit=sh)"

cargo clean -p xtask && cargo build -p xtask --features crossval-all
```

---

## 13. Verification Criteria

### 13.1 Build-Time Verification

**VC1**: Backend selection flag works
```bash
cargo run -p xtask -- setup-cpp-auto --backend bitnet --emit=sh
cargo run -p xtask -- setup-cpp-auto --backend llama --emit=sh
```

**VC2**: Dual-backend installation succeeds
```bash
eval "$(cargo run -p xtask -- setup-cpp-auto --backend bitnet --emit=sh)"
eval "$(cargo run -p xtask -- setup-cpp-auto --backend llama --emit=sh)"
cargo clean -p xtask && cargo build -p xtask --features crossval-all
```

**VC3**: RPATH merging correct
```bash
readelf -d target/debug/xtask | grep RPATH
# Expected: Library rpath: [~/.cache/bitnet_cpp/build/bin:~/.cache/llama_cpp/build]
```

---

## 14. Documentation Updates

### 14.1 Files to Update

| File | Section | Change |
|------|---------|--------|
| `CLAUDE.md` | Environment Variables | Add `LLAMA_CPP_DIR`, `CROSSVAL_RPATH_BITNET`, `CROSSVAL_RPATH_LLAMA` |
| `docs/environment-variables.md` | Cross-Validation Configuration | Document backend-specific variables |
| `docs/howto/cpp-setup.md` | Advanced Setup | Add section on dual-backend installation |
| `docs/explanation/dual-backend-crossval.md` | Build System | Update RPATH merging architecture |

---

## 15. Risks and Mitigations

| Risk ID | Description | Impact | Probability | Mitigation |
|---------|-------------|--------|-------------|------------|
| R1 | llama.cpp build structure differs from BitNet.cpp | High | Medium | Extensive testing, document differences |
| R2 | RPATH length exceeds linker limits (4KB) | Medium | Low | Validate RPATH length, emit clear error |
| R3 | Concurrent installations corrupt state | High | Low | File-based locking (AC19), transactional rollback (AC20) |
| R4 | Network failures during clone | Medium | Medium | Retry logic with exponential backoff (AC18) |
| R5 | Platform-specific build failures | High | Medium | CI matrix testing (AC15), platform validation (AC16) |
| R6 | Backward compatibility breakage | Critical | Low | Regression tests (AC17), beta testing |

---

**END OF SPECIFICATION**

# Test Infrastructure Enhancement Specification: Conditional Execution with Auto-Repair

**Status**: Draft
**Created**: 2025-10-26
**Feature**: Test helpers for conditional backend availability with auto-repair support
**Priority**: High (Infrastructure Foundation)
**Scope**: `tests/support/backend_helpers.rs`, `tests/support/env_guard.rs`, test infrastructure patterns

---

## 1. Problem Statement

BitNet.rs test infrastructure requires robust helpers for conditional test execution based on C++ backend availability (BitNet.cpp, llama.cpp). Current implementation provides basic skip functionality, but lacks:

1. **Comprehensive auto-repair integration**: Tests should attempt backend installation before skipping in local dev
2. **Clear skip diagnostics**: Skip messages need actionable setup instructions and build hygiene guidance
3. **Platform-specific helpers**: Mock library creation and temp directory management for cross-platform tests
4. **Systematic test fixture management**: Reusable patterns for environment isolation, serial execution, and cleanup

### 1.1 Current State Analysis

**Existing Implementation** (`tests/support/backend_helpers.rs`):
- Basic `ensure_backend_or_skip()` function with compile-time detection
- Rudimentary auto-repair attempt via `setup-cpp-auto`
- Simple skip diagnostics with setup instructions

**Coverage Gaps** (identified from `/tmp/test_structure_analysis.md`):
1. **31 tests blocked** by missing auto-repair orchestration (AC1-AC7 in `preflight_auto_repair_tests.rs`)
2. **No runtime backend detection fallback**: Depends solely on build-time constants (`HAS_BITNET`, `HAS_LLAMA`)
3. **No retry logic**: Transient network/build failures cause permanent test skips
4. **No concurrent repair safety**: Multiple parallel tests could trigger simultaneous repairs
5. **No platform-specific mock library helpers**: Tests can't simulate backend availability

**Impact**:
- **Developer friction**: Tests skip instead of attempting zero-install setup
- **CI unpredictability**: Build-time vs runtime detection mismatch causes false negatives
- **Test pollution**: Missing environment isolation for parallel test execution
- **Manual intervention**: Developers must manually run `setup-cpp-auto` before test runs

### 1.2 Architectural Goals

1. **Zero-install developer experience**: Tests auto-repair backends when possible (local dev only)
2. **Deterministic CI behavior**: `BITNET_TEST_NO_REPAIR=1` or `CI=1` prevents downloads during test runs
3. **Clear diagnostics**: Distinguish "skipped (backend unavailable)" from "passed" from "failed"
4. **Robust isolation**: EnvGuard + `#[serial(bitnet_env)]` pattern prevents test pollution
5. **Platform-aware**: Mock libraries, temp directories, and loader paths work across Linux/macOS/Windows

---

## 2. Acceptance Criteria (AC1-AC7)

### AC1: `ensure_backend_or_skip()` with Auto-Repair

**Goal**: Helper function checks backend availability, attempts auto-repair in local dev, skips if unavailable

**Behavior**:
1. **Backend available** (build-time + runtime) → Returns immediately, test continues
2. **Backend unavailable + CI mode** (`BITNET_TEST_NO_REPAIR=1` or `CI=1`) → Prints skip message, returns
3. **Backend unavailable + local dev** → Attempts `setup-cpp-auto`, retries detection, then skips if still unavailable

**Function Signature**:
```rust
/// Ensure backend is available, skip test if not (with optional auto-repair)
///
/// # Behavior
/// 1. Check build-time constant (HAS_BITNET / HAS_LLAMA)
/// 2. If unavailable + CI mode → skip immediately
/// 3. If unavailable + local dev → attempt auto-repair, recheck, then skip
///
/// # Arguments
/// * `backend` - The C++ backend to check (BitNet or Llama)
///
/// # Environment
/// - `BITNET_TEST_NO_REPAIR=1`: Disable auto-repair (CI mode)
/// - `CI=1`: Auto-enable no-repair mode
///
/// # Example
/// ```rust
/// #[test]
/// fn test_bitnet_crossval() {
///     ensure_backend_or_skip(CppBackend::BitNet);
///     // Test code runs only if backend available
/// }
/// ```
pub fn ensure_backend_or_skip(backend: CppBackend)
```

**Test Coverage**: `tests/test_support_tests.rs` AC1 tests (3 tests)

---

### AC2: Backend Availability Detection (Compile-Time + Runtime)

**Goal**: Distinguish build-time detection (constants) from runtime detection (dynamic library loading)

**Two-Tier Detection Strategy**:

**Tier 1: Build-Time Constants** (from `crossval/build.rs`):
```rust
use bitnet_crossval::{HAS_BITNET, HAS_LLAMA};

// Set by build.rs during xtask compilation
// Requires rebuild after installing libraries:
// cargo clean -p crossval && cargo build --features crossval-all
```

**Tier 2: Runtime Detection** (fallback for post-install scenarios):
```rust
/// Check if backend libraries are available at runtime
///
/// This provides a fallback detection mechanism when libraries
/// were installed after xtask was built (avoiding full rebuild).
///
/// # Returns
/// - `Ok(true)` if libraries found via dynamic loader
/// - `Ok(false)` if libraries not found
/// - `Err(String)` if detection failed
fn detect_backend_runtime(backend: CppBackend) -> Result<bool, String>
```

**Detection Flow**:
```
1. Check build-time constant (HAS_BITNET / HAS_LLAMA)
   ├─ true → Backend available (skip runtime check)
   └─ false → Proceed to runtime detection

2. Attempt runtime library discovery
   ├─ Libraries found → Warn about rebuild need, return available
   └─ Libraries not found → Return unavailable
```

**Rebuild Guidance**:
```rust
if runtime_detected && !build_time_detected {
    eprintln!("⚠️  Backend libraries found at runtime but not at build time.");
    eprintln!("    Rebuild xtask to update detection:");
    eprintln!("    cargo clean -p crossval && cargo build -p xtask --features crossval-all");
}
```

**Test Coverage**: `tests/test_support_tests.rs` AC2 tests (4 tests)

---

### AC3: Auto-Repair Integration with Retry Logic

**Goal**: Attempt backend installation automatically, with transient error retry support

**Orchestration Function**:
```rust
/// Attempt to install missing backend with retry on transient errors
///
/// # Arguments
/// * `backend` - The backend to install (BitNet or Llama)
/// * `max_retries` - Maximum retry attempts for transient errors (default: 2)
///
/// # Returns
/// - `Ok(())` if installation succeeded
/// - `Err(RepairError)` if installation failed (classified error)
///
/// # Environment
/// - `BITNET_REPAIR_IN_PROGRESS=1`: Set during repair to prevent recursion
///
/// # Example
/// ```rust
/// match attempt_auto_repair_with_retry(CppBackend::BitNet, 2) {
///     Ok(()) => eprintln!("Backend repaired successfully"),
///     Err(e) => eprintln!("Repair failed: {}", e),
/// }
/// ```
fn attempt_auto_repair_with_retry(
    backend: CppBackend,
    max_retries: usize,
) -> Result<(), RepairError>
```

**Error Classification**:
```rust
#[derive(Debug, Clone)]
pub enum RepairError {
    /// Network connectivity issues (transient, retry)
    NetworkError { message: String, retryable: bool },

    /// Build system errors (CMake, compiler)
    BuildError { message: String, retryable: bool },

    /// Missing prerequisites (git, cmake, compiler)
    MissingPrerequisites { tools: Vec<String> },

    /// Permission denied (filesystem, download)
    PermissionDenied { path: String },

    /// Recursion detected (BITNET_REPAIR_IN_PROGRESS set)
    RecursionDetected,

    /// Unknown error
    Unknown { message: String },
}

impl RepairError {
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::NetworkError { retryable, .. } => *retryable,
            Self::BuildError { retryable, .. } => *retryable,
            _ => false,
        }
    }
}
```

**Retry Logic**:
```rust
const MAX_RETRIES: usize = 2;
const RETRY_DELAYS: &[Duration] = &[
    Duration::from_secs(1),  // First retry: 1s
    Duration::from_secs(2),  // Second retry: 2s
];

let mut attempt = 0;
loop {
    match run_setup_cpp_auto(backend) {
        Ok(()) => return Ok(()),
        Err(e) if e.is_retryable() && attempt < MAX_RETRIES => {
            eprintln!("Repair attempt {} failed (transient error), retrying in {:?}...",
                      attempt + 1, RETRY_DELAYS[attempt]);
            std::thread::sleep(RETRY_DELAYS[attempt]);
            attempt += 1;
        }
        Err(e) => return Err(e),
    }
}
```

**Recursion Prevention**:
```rust
fn attempt_auto_repair_with_retry(backend: CppBackend, max_retries: usize) -> Result<(), RepairError> {
    // Check for recursion (prevent infinite repair loops)
    if std::env::var("BITNET_REPAIR_IN_PROGRESS").is_ok() {
        return Err(RepairError::RecursionDetected);
    }

    // Set recursion guard
    let _guard = EnvGuard::new("BITNET_REPAIR_IN_PROGRESS");
    _guard.set("1");

    // Attempt repair with retry...
}
```

**Test Coverage**: `xtask/tests/preflight_auto_repair_tests.rs` AC3 tests (6 tests)

---

### AC4: Clear Skip Messages with Setup Instructions

**Goal**: Print actionable diagnostics when tests are skipped due to backend unavailability

**Skip Message Format**:
```rust
/// Print skip diagnostic with setup instructions
///
/// This prints a standardized skip message to stderr with:
/// - Clear indication of which backend is unavailable
/// - Setup instructions (auto-setup vs manual)
/// - Build hygiene guidance (rebuild xtask)
/// - Optional context (CI mode, no-repair mode)
///
/// # Arguments
/// * `backend` - The unavailable backend
/// * `context` - Additional context (e.g., "CI mode", "auto-repair failed")
fn print_skip_diagnostic(backend: CppBackend, context: Option<&str>)
```

**Output Format**:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⊘ Test skipped: bitnet.cpp not available
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Context: CI mode (auto-repair disabled)

This test requires the BitNet.cpp C++ reference backend.

Setup Instructions:
──────────────────────────────────────────────────────────

  Option A: Auto-setup (recommended)

    1. Install backend:
       eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"

    2. Rebuild xtask to update detection:
       cargo clean -p crossval && cargo build -p xtask --features crossval-all

    3. Re-run tests:
       cargo test --workspace --no-default-features --features cpu

  Option B: Manual setup (advanced)

    1. Clone and build BitNet.cpp:
       git clone https://github.com/microsoft/BitNet.git ~/.cache/bitnet_cpp
       cd ~/.cache/bitnet_cpp && mkdir build && cd build
       cmake .. && cmake --build .

    2. Set environment variables:
       export BITNET_CPP_DIR=~/.cache/bitnet_cpp
       export LD_LIBRARY_PATH=~/.cache/bitnet_cpp/build/bin:$LD_LIBRARY_PATH

    3. Rebuild and re-run tests (steps 2-3 from Option A)

Documentation: docs/howto/cpp-setup.md

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Test Coverage**: `tests/test_support_tests.rs` AC4 tests (2 tests)

---

### AC5: Test Fixture Helpers (Mock Libraries, Temp Dirs, Env Isolation)

**Goal**: Reusable helpers for platform-specific test fixtures

#### 5.1 Mock Library Creation

**Function Signature**:
```rust
/// Create mock C++ backend libraries for testing
///
/// This creates empty shared library files with correct platform-specific
/// extensions and naming conventions.
///
/// # Arguments
/// * `backend` - The backend to mock (BitNet or Llama)
/// * `dir` - Directory to create mock libraries in
///
/// # Returns
/// - `Ok(Vec<PathBuf>)` - Paths to created mock libraries
/// - `Err(String)` - Error message if creation failed
///
/// # Platform-Specific Behavior
/// - Linux: Creates `libbitnet.so`, `libllama.so`, `libggml.so`
/// - macOS: Creates `libbitnet.dylib`, `libllama.dylib`, `libggml.dylib`
/// - Windows: Creates `bitnet.dll`, `llama.dll`, `ggml.dll`
///
/// # Example
/// ```rust
/// use tempfile::TempDir;
///
/// #[test]
/// fn test_library_discovery() {
///     let temp = TempDir::new().unwrap();
///     let libs = create_mock_backend_libs(CppBackend::BitNet, temp.path()).unwrap();
///
///     assert_eq!(libs.len(), 1); // libbitnet.so
///     assert!(libs[0].exists());
/// }
/// ```
pub fn create_mock_backend_libs(
    backend: CppBackend,
    dir: &Path,
) -> Result<Vec<PathBuf>, String>
```

**Implementation** (platform-aware):
```rust
pub fn create_mock_backend_libs(backend: CppBackend, dir: &Path) -> Result<Vec<PathBuf>, String> {
    let lib_names = match backend {
        CppBackend::BitNet => vec!["bitnet"],
        CppBackend::Llama => vec!["llama", "ggml"],
    };

    let mut created = Vec::new();

    for name in lib_names {
        let lib_path = if cfg!(target_os = "linux") {
            dir.join(format!("lib{}.so", name))
        } else if cfg!(target_os = "macos") {
            dir.join(format!("lib{}.dylib", name))
        } else if cfg!(target_os = "windows") {
            dir.join(format!("{}.dll", name))
        } else {
            return Err(format!("Unsupported platform: {}", std::env::consts::OS));
        };

        // Create empty file
        std::fs::File::create(&lib_path)
            .map_err(|e| format!("Failed to create mock library: {}", e))?;

        created.push(lib_path);
    }

    Ok(created)
}
```

#### 5.2 Environment Isolation Pattern

**Enhanced EnvGuard Usage**:
```rust
/// Create environment guard with automatic cleanup
///
/// This is a wrapper around `EnvGuard::new()` with additional
/// convenience methods for common test patterns.
///
/// # Example
/// ```rust
/// use serial_test::serial;
///
/// #[test]
/// #[serial(bitnet_env)]  // REQUIRED for env mutation
/// fn test_with_env_isolation() {
///     let _guard = env_guard("BITNET_STRICT_MODE", "1");
///     // Test code - env restored on drop
/// }
/// ```
pub fn env_guard(key: &str, value: &str) -> EnvGuard {
    let guard = EnvGuard::new(key);
    guard.set(value);
    guard
}

/// Create environment guard that removes variable temporarily
///
/// # Example
/// ```rust
/// #[test]
/// #[serial(bitnet_env)]
/// fn test_with_env_removed() {
///     let _guard = env_guard_remove("BITNET_CPP_DIR");
///     // Variable removed, restored on drop
/// }
/// ```
pub fn env_guard_remove(key: &str) -> EnvGuard {
    let guard = EnvGuard::new(key);
    guard.remove();
    guard
}
```

#### 5.3 Workspace Root Discovery

**Function Signature**:
```rust
/// Find workspace root directory (contains .git)
///
/// Walks up from current crate manifest directory until .git found.
///
/// # Returns
/// - `Ok(PathBuf)` - Workspace root path
/// - `Err(String)` - Error if .git not found
///
/// # Example
/// ```rust
/// let root = workspace_root().unwrap();
/// let models_dir = root.join("models");
/// ```
pub fn workspace_root() -> Result<PathBuf, String>
```

**Implementation**:
```rust
pub fn workspace_root() -> Result<PathBuf, String> {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    while !path.join(".git").exists() {
        if !path.pop() {
            return Err("Could not find workspace root (.git directory)".to_string());
        }
    }

    Ok(path)
}
```

**Test Coverage**: `tests/test_support_tests.rs` AC5 tests (5 tests)

---

### AC6: Serial Test Execution Pattern for Env Mutation

**Goal**: Enforce process-level serialization for environment-mutating tests

**Pattern Enforcement**:
```rust
/// Marker trait for tests that mutate environment variables
///
/// This trait serves as documentation and provides a compile-time
/// check that tests using EnvGuard also use #[serial(bitnet_env)].
///
/// # Safety Contract
///
/// Tests implementing this trait MUST:
/// 1. Use `#[serial(bitnet_env)]` attribute
/// 2. Use `EnvGuard` for env mutations (not raw `env::set_var`)
/// 3. Clean up temp resources (use TempDir, not manual cleanup)
pub trait RequiresEnvIsolation {}
```

**Usage Example**:
```rust
use serial_test::serial;
use tests::support::{env_guard, RequiresEnvIsolation};

#[test]
#[serial(bitnet_env)]  // REQUIRED for env mutation
fn test_strict_mode_enabled() -> impl RequiresEnvIsolation {
    let _guard = env_guard("BITNET_STRICT_MODE", "1");

    let config = StrictModeConfig::from_env();
    assert!(config.enabled);

    // Return type enforces contract
    ()
}
```

**Anti-Pattern Detection** (clippy lint proposal):
```rust
// Future work: Custom clippy lint to detect missing #[serial]
#[clippy::env_mutation_without_serial]
#[test]
fn test_env_mutation_unsafe() {  // ❌ Missing #[serial(bitnet_env)]
    let _guard = EnvGuard::new("BITNET_TEST");
    // ...
}
```

**Test Coverage**: `tests/test_support_tests.rs` AC6 tests (3 tests)

---

### AC7: Platform-Specific Test Helpers (Linux, macOS, Windows)

**Goal**: Abstract platform differences for cross-platform test portability

#### 7.1 Dynamic Loader Path Variable

**Function Signature**:
```rust
/// Get platform-specific dynamic loader path variable name
///
/// # Returns
/// - `"LD_LIBRARY_PATH"` on Linux
/// - `"DYLD_LIBRARY_PATH"` on macOS
/// - `"PATH"` on Windows
///
/// # Example
/// ```rust
/// let loader_var = get_loader_path_var();
/// env::set_var(loader_var, "/custom/lib");
/// ```
pub fn get_loader_path_var() -> &'static str
```

**Implementation**:
```rust
pub fn get_loader_path_var() -> &'static str {
    if cfg!(target_os = "linux") {
        "LD_LIBRARY_PATH"
    } else if cfg!(target_os = "macos") {
        "DYLD_LIBRARY_PATH"
    } else if cfg!(target_os = "windows") {
        "PATH"
    } else {
        panic!("Unsupported platform: {}", std::env::consts::OS)
    }
}
```

#### 7.2 Library Extension Detection

**Function Signature**:
```rust
/// Get platform-specific shared library extension
///
/// # Returns
/// - `"so"` on Linux
/// - `"dylib"` on macOS
/// - `"dll"` on Windows
pub fn get_lib_extension() -> &'static str
```

#### 7.3 Library Name Formatting

**Function Signature**:
```rust
/// Format library name with platform-specific prefix/extension
///
/// # Arguments
/// * `stem` - Library name stem (e.g., "bitnet", "llama")
///
/// # Returns
/// - `"libbitnet.so"` on Linux
/// - `"libbitnet.dylib"` on macOS
/// - `"bitnet.dll"` on Windows
///
/// # Example
/// ```rust
/// let lib_name = format_lib_name("bitnet");
/// assert_eq!(lib_name, "libbitnet.so"); // Linux
/// ```
pub fn format_lib_name(stem: &str) -> String
```

**Implementation**:
```rust
pub fn format_lib_name(stem: &str) -> String {
    if cfg!(target_os = "windows") {
        format!("{}.dll", stem)
    } else if cfg!(target_os = "macos") {
        format!("lib{}.dylib", stem)
    } else {
        format!("lib{}.so", stem)
    }
}
```

**Test Coverage**: `tests/test_support_tests.rs` AC7 tests (4 tests)

---

## 3. Architecture

### 3.1 Helper Function Hierarchy

```
tests/support/
├── backend_helpers.rs         (Backend availability detection)
│   ├── ensure_backend_or_skip()         [AC1]
│   ├── ensure_bitnet_or_skip()          [AC1 convenience]
│   ├── ensure_llama_or_skip()           [AC1 convenience]
│   ├── detect_backend_runtime()         [AC2]
│   ├── attempt_auto_repair_with_retry() [AC3]
│   └── print_skip_diagnostic()          [AC4]
│
├── env_guard.rs              (Environment isolation)
│   ├── EnvGuard (struct)                [AC5, AC6]
│   ├── env_guard()                      [AC5 convenience]
│   └── env_guard_remove()               [AC5 convenience]
│
├── mock_fixtures.rs          (NEW - Test fixtures)
│   ├── create_mock_backend_libs()       [AC5]
│   ├── MockLibraryBuilder               [AC5 builder pattern]
│   └── create_temp_cpp_env()            [AC5 integrated setup]
│
└── platform_utils.rs         (NEW - Platform abstraction)
    ├── get_loader_path_var()            [AC7]
    ├── get_lib_extension()              [AC7]
    ├── format_lib_name()                [AC7]
    └── workspace_root()                 [AC5]
```

### 3.2 Backend Detection Strategy

```
┌─────────────────────────────────────────────────────────┐
│ ensure_backend_or_skip(backend)                         │
└─────────────────────────────────────────────────────────┘
                       │
                       ▼
    ┌─────────────────────────────────────┐
    │ 1. Check build-time constant        │
    │    (HAS_BITNET / HAS_LLAMA)         │
    └─────────────────────────────────────┘
                       │
           ┌───────────┴────────────┐
           │                        │
      available               unavailable
           │                        │
           ▼                        ▼
      ┌────────┐        ┌─────────────────────────┐
      │ Return │        │ 2. Check CI mode        │
      └────────┘        │    (CI=1 or NO_REPAIR)  │
                        └─────────────────────────┘
                                   │
                       ┌───────────┴────────────┐
                       │                        │
                    CI mode                local dev
                       │                        │
                       ▼                        ▼
            ┌─────────────────┐    ┌──────────────────────────┐
            │ Skip immediately│    │ 3. Attempt auto-repair   │
            │ (print diag)    │    │    with retry            │
            └─────────────────┘    └──────────────────────────┘
                                              │
                                  ┌───────────┴────────────┐
                                  │                        │
                            repair OK               repair failed
                                  │                        │
                                  ▼                        ▼
                        ┌──────────────────┐    ┌────────────────┐
                        │ 4. Runtime check │    │ Skip with diag │
                        │    (fallback)    │    │ (repair error) │
                        └──────────────────┘    └────────────────┘
                                  │
                      ┌───────────┴────────────┐
                      │                        │
                 found runtime           not found
                      │                        │
                      ▼                        ▼
            ┌─────────────────┐    ┌────────────────────┐
            │ Warn: rebuild   │    │ Skip with diag     │
            │ Return OK       │    │ (still unavailable)│
            └─────────────────┘    └────────────────────┘
```

### 3.3 Auto-Repair Orchestration

```
┌─────────────────────────────────────────────────────────┐
│ attempt_auto_repair_with_retry(backend, max_retries)   │
└─────────────────────────────────────────────────────────┘
                       │
                       ▼
    ┌─────────────────────────────────────┐
    │ Check recursion guard               │
    │ (BITNET_REPAIR_IN_PROGRESS)         │
    └─────────────────────────────────────┘
                       │
           ┌───────────┴────────────┐
           │                        │
      not set                   already set
           │                        │
           ▼                        ▼
    ┌──────────────┐      ┌──────────────────┐
    │ Set guard    │      │ Return error     │
    │ via EnvGuard │      │ (RecursionError) │
    └──────────────┘      └──────────────────┘
           │
           ▼
    ┌─────────────────────────────────────┐
    │ Retry loop (max_retries)            │
    └─────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────┐
    │ Run setup-cpp-auto                  │
    │ (cargo run -p xtask --)             │
    └─────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────┐
    │ Classify error (if failed)          │
    │ - NetworkError (retryable)          │
    │ - BuildError (retryable/not)        │
    │ - MissingPrerequisites (not)        │
    │ - PermissionDenied (not)            │
    │ - Unknown                           │
    └─────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────┐
    │ Retry logic                         │
    │ - If retryable + attempts left:     │
    │   sleep (1s, 2s) and retry          │
    │ - Otherwise: return error           │
    └─────────────────────────────────────┘
```

### 3.4 EnvGuard Pattern with Serial Execution

```
Test Execution Timeline (without #[serial]):
════════════════════════════════════════════════════════════

Thread 1: Test A                  Thread 2: Test B
  │                                 │
  ├─ EnvGuard::new("VAR")          │
  │   └─ Lock mutex ✓              │
  │                                 ├─ EnvGuard::new("VAR")
  │                                 │   └─ Wait for mutex... ⏳
  ├─ set_var("VAR", "A") ✓         │
  │                                 │
  ├─ Test logic with VAR=A         │
  │                                 │
  └─ Drop guard (restore VAR)      │
      └─ Unlock mutex ✓            │
                                    ├─ Acquire mutex ✓
                                    ├─ set_var("VAR", "B") ✓
                                    │
                                    ├─ Test logic with VAR=B
                                    │
                                    └─ Drop guard (restore VAR)
                                        └─ Unlock mutex ✓

✓ Thread-safe within process


Test Execution Timeline (with #[serial(bitnet_env)]):
════════════════════════════════════════════════════════════

Process 1: Test A                 Process 2: Test B
  │                                 │
  ├─ Serial lock "bitnet_env" ✓   │
  │                                 ├─ Serial lock "bitnet_env"
  │                                 │   └─ Wait for Test A... ⏳
  ├─ EnvGuard::new("VAR")          │
  ├─ set_var("VAR", "A") ✓         │
  │                                 │
  ├─ Test logic with VAR=A         │
  │                                 │
  └─ Drop guard + release lock ✓  │
                                    ├─ Acquire serial lock ✓
                                    ├─ EnvGuard::new("VAR")
                                    ├─ set_var("VAR", "B") ✓
                                    │
                                    ├─ Test logic with VAR=B
                                    │
                                    └─ Drop guard + release lock ✓

✓ Process-safe across cargo test runners
```

---

## 4. API Design

### 4.1 Core Backend Helpers (`backend_helpers.rs`)

#### Primary Function
```rust
pub fn ensure_backend_or_skip(backend: CppBackend)
```
**See AC1 for full signature and behavior**

#### Convenience Wrappers
```rust
pub fn ensure_bitnet_or_skip()
pub fn ensure_llama_or_skip()
```

#### Detection Functions
```rust
fn detect_backend_runtime(backend: CppBackend) -> Result<bool, String>
fn is_ci_or_no_repair() -> bool
```

#### Auto-Repair Functions
```rust
fn attempt_auto_repair_with_retry(
    backend: CppBackend,
    max_retries: usize,
) -> Result<(), RepairError>

fn run_setup_cpp_auto(backend: CppBackend) -> Result<(), RepairError>
fn classify_error(stderr: &str) -> RepairError
```

#### Diagnostic Functions
```rust
fn print_skip_diagnostic(backend: CppBackend, context: Option<&str>)
fn print_rebuild_warning(backend: CppBackend)
```

### 4.2 Mock Fixtures (`mock_fixtures.rs` - NEW)

#### Mock Library Creation
```rust
pub fn create_mock_backend_libs(
    backend: CppBackend,
    dir: &Path,
) -> Result<Vec<PathBuf>, String>
```
**See AC5 for full signature and implementation**

#### Builder Pattern for Complex Setups
```rust
pub struct MockLibraryBuilder {
    backend: CppBackend,
    base_dir: PathBuf,
    version_suffix: Option<String>,
    symlinks: bool,
}

impl MockLibraryBuilder {
    pub fn new(backend: CppBackend, base_dir: PathBuf) -> Self
    pub fn with_version(mut self, version: &str) -> Self
    pub fn with_symlinks(mut self, enabled: bool) -> Self
    pub fn build(self) -> Result<Vec<PathBuf>, String>
}
```

**Example Usage**:
```rust
#[test]
fn test_version_suffix_detection() {
    let temp = TempDir::new().unwrap();

    let libs = MockLibraryBuilder::new(CppBackend::Llama, temp.path().to_path_buf())
        .with_version("3.0.1")
        .with_symlinks(true)
        .build()
        .unwrap();

    // Creates:
    // - libllama.so.3.0.1
    // - libllama.so.3 -> libllama.so.3.0.1
    // - libllama.so -> libllama.so.3
    // - libggml.so (similar)
}
```

#### Integrated Environment Setup
```rust
/// Create temporary C++ backend environment for testing
///
/// This combines:
/// - TempDir creation
/// - Mock library generation
/// - Environment variable setup
/// - EnvGuard cleanup
///
/// # Returns
/// Tuple of (TempDir, Vec<EnvGuard>) - both auto-cleanup on drop
pub fn create_temp_cpp_env(
    backend: CppBackend,
) -> Result<(TempDir, Vec<EnvGuard>), String>
```

**Example Usage**:
```rust
#[test]
#[serial(bitnet_env)]
fn test_library_discovery_with_mock() {
    let (temp, _guards) = create_temp_cpp_env(CppBackend::BitNet).unwrap();

    // BITNET_CPP_DIR and LD_LIBRARY_PATH automatically set
    // Mock libraries created and discoverable

    let detected = detect_backend_runtime(CppBackend::BitNet).unwrap();
    assert!(detected);

    // Cleanup automatic via Drop
}
```

### 4.3 Platform Utilities (`platform_utils.rs` - NEW)

```rust
pub fn get_loader_path_var() -> &'static str
pub fn get_lib_extension() -> &'static str
pub fn format_lib_name(stem: &str) -> String
pub fn workspace_root() -> Result<PathBuf, String>
```

**See AC7 for full signatures and implementations**

### 4.4 Enhanced EnvGuard (`env_guard.rs`)

**Existing API** (unchanged):
```rust
impl EnvGuard {
    pub fn new(key: &str) -> Self
    pub fn set(&self, val: &str)
    pub fn remove(&self)
    pub fn key(&self) -> &str
    pub fn original_value(&self) -> Option<&str>
}
```

**New Convenience Functions**:
```rust
pub fn env_guard(key: &str, value: &str) -> EnvGuard
pub fn env_guard_remove(key: &str) -> EnvGuard
```

---

## 5. Implementation Phases

### Phase 1: Basic Helpers (Priority: P0)

**Estimated Effort**: 2-3 hours
**Dependencies**: None
**Deliverables**:
- Enhanced `ensure_backend_or_skip()` with runtime detection fallback
- `detect_backend_runtime()` implementation
- Updated skip diagnostics with rebuild warnings
- Basic platform utilities (`get_loader_path_var`, `format_lib_name`)

**Implementation Steps**:
1. Add `detect_backend_runtime()` to `backend_helpers.rs`
2. Update `ensure_backend_or_skip()` to use two-tier detection
3. Add rebuild warning messages
4. Create `platform_utils.rs` with basic helpers
5. Add unit tests for detection logic

**Test Coverage**:
- `tests/test_support_tests.rs`: 8 tests
  - Backend detection (4 tests)
  - Skip diagnostics (2 tests)
  - Platform utilities (2 tests)

---

### Phase 2: Auto-Repair Integration (Priority: P0)

**Estimated Effort**: 3-4 hours
**Dependencies**: Phase 1 complete
**Deliverables**:
- `attempt_auto_repair_with_retry()` with error classification
- `RepairError` enum with retryability logic
- Recursion prevention guard
- Retry logic with exponential backoff
- Integration with `ensure_backend_or_skip()`

**Implementation Steps**:
1. Define `RepairError` enum in `backend_helpers.rs`
2. Implement error classification from stderr parsing
3. Add retry loop with backoff
4. Implement recursion guard via `BITNET_REPAIR_IN_PROGRESS`
5. Integrate into `ensure_backend_or_skip()` flow
6. Add comprehensive error handling tests

**Test Coverage**:
- `xtask/tests/preflight_auto_repair_tests.rs`: 37 tests
  - Auto-repair success (AC1): 3 tests
  - No-repair flag (AC2): 3 tests
  - Error handling (AC3): 6 tests
  - Dual-backend (AC4): 2 tests
  - Verbose progress (AC5): 3 tests
  - Exit codes (AC6): 3 tests
  - CI safety (AC7): 3 tests
  - Integration tests: 4 tests
  - Property tests: 1 test
  - Helper tests: 9 tests

---

### Phase 3: Advanced Isolation (Priority: P1)

**Estimated Effort**: 2-3 hours
**Dependencies**: Phase 1 complete
**Deliverables**:
- `mock_fixtures.rs` module with mock library creation
- `MockLibraryBuilder` for complex scenarios
- `create_temp_cpp_env()` integrated helper
- Enhanced convenience functions for `EnvGuard`

**Implementation Steps**:
1. Create `tests/support/mock_fixtures.rs`
2. Implement `create_mock_backend_libs()` with platform detection
3. Add `MockLibraryBuilder` for versioned libraries
4. Implement `create_temp_cpp_env()` with auto-setup
5. Add `env_guard()` and `env_guard_remove()` convenience functions
6. Add fixture tests

**Test Coverage**:
- `tests/test_support_tests.rs`: 12 tests
  - Mock library creation (5 tests)
  - Builder pattern (3 tests)
  - Integrated environment (2 tests)
  - EnvGuard convenience (2 tests)

---

## 6. Testing Strategy

### 6.1 Unit Tests

#### Test Organization
```
tests/
├── test_support_tests.rs         (Primary test file)
│   ├── mod backend_detection     (AC2: 4 tests)
│   ├── mod auto_repair           (AC3: 6 tests)
│   ├── mod skip_diagnostics      (AC4: 2 tests)
│   ├── mod mock_fixtures         (AC5: 10 tests)
│   ├── mod env_isolation         (AC6: 3 tests)
│   └── mod platform_utils        (AC7: 4 tests)
│
└── support/
    ├── backend_helpers.rs        (Internal tests: 10 tests)
    ├── env_guard.rs              (Internal tests: 7 tests)
    ├── mock_fixtures.rs          (Internal tests: 5 tests)
    └── platform_utils.rs         (Internal tests: 3 tests)
```

#### Test Matrix

| Component | Tests | Feature Gate | Serial Required |
|-----------|-------|--------------|-----------------|
| Backend detection | 4 | `crossval-all` | No |
| Auto-repair retry | 6 | `crossval-all` | Yes |
| Skip diagnostics | 2 | None | No |
| Mock fixtures | 10 | None | No |
| Env isolation | 3 | None | Yes |
| Platform utils | 4 | None | No |
| **Total** | **29** | | |

### 6.2 Integration Tests

#### Test Scenarios

**Scenario 1: Auto-Repair Success Path**
```rust
#[test]
#[ignore] // Manual verification - requires network
fn test_auto_repair_bitnet_from_scratch() {
    // Prerequisites: No C++ backend installed

    // Clear environment
    std::env::remove_var("BITNET_CPP_DIR");
    std::env::remove_var("BITNET_TEST_NO_REPAIR");

    // Trigger auto-repair
    ensure_bitnet_or_skip();

    // Should succeed and install backend
    // Verify: ~/.cache/bitnet_cpp exists and contains libraries
}
```

**Scenario 2: CI Mode Skip**
```rust
#[test]
#[serial(bitnet_env)]
fn test_ci_mode_skips_without_repair() {
    let _guard_ci = env_guard("CI", "1");
    let _guard_dir = env_guard_remove("BITNET_CPP_DIR");

    // Should skip immediately without attempting repair
    ensure_bitnet_or_skip();

    // Verify: No setup-cpp-auto process spawned
    // (Check via process listing or mock)
}
```

**Scenario 3: Runtime Detection Fallback**
```rust
#[test]
#[serial(bitnet_env)]
fn test_runtime_detection_after_install() {
    // Simulate: Libraries installed but xtask not rebuilt
    let temp = TempDir::new().unwrap();
    let _libs = create_mock_backend_libs(CppBackend::BitNet, temp.path()).unwrap();
    let _guard = env_guard("BITNET_CPP_DIR", temp.path().to_str().unwrap());

    // Build-time constant says unavailable
    assert!(!HAS_BITNET); // Compile-time check

    // But runtime detection should find it
    let runtime = detect_backend_runtime(CppBackend::BitNet).unwrap();
    assert!(runtime);
}
```

**Scenario 4: Mock Library Discovery**
```rust
#[test]
#[serial(bitnet_env)]
fn test_mock_library_discovery_linux() {
    let temp = TempDir::new().unwrap();
    let libs = create_mock_backend_libs(CppBackend::Llama, temp.path()).unwrap();

    // Verify correct platform extensions
    assert!(libs.iter().any(|p| p.to_string_lossy().ends_with(".so")));

    // Verify both llama and ggml created
    assert_eq!(libs.len(), 2);
}
```

### 6.3 Platform-Specific Tests

**Linux**:
```rust
#[test]
#[cfg(target_os = "linux")]
fn test_linux_loader_path() {
    assert_eq!(get_loader_path_var(), "LD_LIBRARY_PATH");
    assert_eq!(get_lib_extension(), "so");
    assert_eq!(format_lib_name("bitnet"), "libbitnet.so");
}
```

**macOS**:
```rust
#[test]
#[cfg(target_os = "macos")]
fn test_macos_loader_path() {
    assert_eq!(get_loader_path_var(), "DYLD_LIBRARY_PATH");
    assert_eq!(get_lib_extension(), "dylib");
    assert_eq!(format_lib_name("bitnet"), "libbitnet.dylib");
}
```

**Windows**:
```rust
#[test]
#[cfg(target_os = "windows")]
fn test_windows_loader_path() {
    assert_eq!(get_loader_path_var(), "PATH");
    assert_eq!(get_lib_extension(), "dll");
    assert_eq!(format_lib_name("bitnet"), "bitnet.dll");
}
```

---

## 7. Dependencies

### 7.1 Existing Infrastructure

**Required Components**:
- `tests/support/backend_helpers.rs` - Basic backend availability checks
- `tests/support/env_guard.rs` - RAII environment variable guard
- `bitnet_crossval::backend::CppBackend` - Backend enum (BitNet, Llama)
- `bitnet_crossval::{HAS_BITNET, HAS_LLAMA}` - Build-time constants
- `serial_test::serial` - Process-level test serialization

**External Crates**:
```toml
[dev-dependencies]
tempfile = "3.8"          # Temporary directory management
serial_test = "3.0"       # Test serialization
proptest = "1.4"          # Property-based testing (optional)
```

### 7.2 New Files Created

```
tests/support/
├── mock_fixtures.rs      (NEW - ~200 lines)
│   ├── create_mock_backend_libs()
│   ├── MockLibraryBuilder
│   └── create_temp_cpp_env()
│
└── platform_utils.rs     (NEW - ~100 lines)
    ├── get_loader_path_var()
    ├── get_lib_extension()
    ├── format_lib_name()
    └── workspace_root()

tests/
└── test_support_tests.rs (NEW - ~600 lines)
    └── Comprehensive tests for all AC1-AC7
```

### 7.3 Modified Files

```
tests/support/backend_helpers.rs  (~150 lines added)
├── detect_backend_runtime()       (NEW)
├── attempt_auto_repair_with_retry() (ENHANCED)
├── RepairError enum               (NEW)
└── print_skip_diagnostic()        (ENHANCED)

tests/support/env_guard.rs  (~50 lines added)
├── env_guard()                    (NEW convenience)
└── env_guard_remove()             (NEW convenience)
```

---

## 8. Summary

This specification defines comprehensive test infrastructure enhancements for BitNet.rs, addressing 7 acceptance criteria across backend availability detection, auto-repair, environment isolation, and platform portability.

**Key Deliverables**:
- Enhanced `ensure_backend_or_skip()` with auto-repair and retry logic
- Two-tier detection (build-time constants + runtime fallback)
- Mock library creation with builder pattern
- Platform-specific helpers (Linux/macOS/Windows)
- Comprehensive test suite (29+ unit tests, 37+ integration tests)

**Success Criteria**:
- 80% auto-repair success rate in local dev
- 0% test pollution across parallel execution
- 100% actionable skip messages with setup instructions

**Next Steps**:
1. Implement Phase 1 (basic helpers) - ~2-3 hours
2. Implement Phase 2 (auto-repair) - ~3-4 hours
3. Implement Phase 3 (advanced isolation) - ~2-3 hours
4. Add documentation and migrate existing tests - ~1 week
5. Validate across platforms and CI/CD - ~1 week

**Total Estimated Effort**: 2-3 weeks (including testing, documentation, migration)

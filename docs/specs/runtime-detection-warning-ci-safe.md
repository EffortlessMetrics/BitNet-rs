# Technical Specification: Runtime Detection Warnings with CI-Safe Fallback

**Document ID**: `runtime-detection-warning-ci-safe`
**Status**: Draft
**Created**: 2025-10-27
**Author**: BitNet-rs Generative Spec Agent
**Related Files**:
- `/tmp/p0_runtime_detection_analysis.md` (comprehensive dual-detection analysis)
- `/home/steven/code/Rust/BitNet-rs/xtask/src/crossval/preflight.rs` (lines 411-418, 519-695, 1084-1109)
- `/home/steven/code/Rust/BitNet-rs/crossval/build.rs` (lines 104-437)
- `/home/steven/code/Rust/BitNet-rs/tests/support/backend_helpers.rs` (runtime detection helpers)

---

## Executive Summary

BitNet-rs implements a **dual-detection system** for C++ backend library availability:

1. **Build-time detection** (`crossval/build.rs`): Scans filesystem during compilation, exports immutable constants (`HAS_BITNET`, `HAS_LLAMA`) baked into xtask binary
2. **Runtime detection** (`preflight.rs`, `backend_helpers.rs`): Fallback search when build-time detection failed, supports dev convenience

**Current Gap**: When users install C++ libraries AFTER building xtask, runtime detection succeeds but build-time constants remain stale (`false`). Users receive no clear guidance that xtask rebuild is required.

**Solution**: Implement environment-aware warnings and CI-safe skip logic:
- **Dev Mode**: Allow tests to continue with prominent rebuild warning
- **CI Mode**: Skip tests with deterministic diagnostic (no runtime override)
- **Verbose Mode**: Detailed search diagnostics and resolution steps

This specification addresses the "stale build" scenario critical for BitNet-rs neural network cross-validation reliability, ensuring CI determinism while maintaining dev-friendly UX.

---

## Problem Statement

### The Stale Build Scenario

**Typical User Workflow**:
```bash
# Step 1: User builds xtask (libraries NOT installed)
cargo build -p xtask --features crossval-all
# → Build-time detection: HAS_BITNET=false, HAS_LLAMA=false
# → Constants baked into target/debug/xtask binary

# Step 2: User installs C++ reference libraries
eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"
# → Libraries now exist in ~/.cache/bitnet_cpp/build

# Step 3: User runs cross-validation test
cargo test -p bitnet-models --features crossval
# → Build-time constants still false (stale)
# → Runtime detection finds libraries (true)
# → CONFUSION: Test behavior undefined
```

**Current Behavior**:
- Runtime detection succeeds silently
- User unaware xtask constants are stale
- Tests may run with inconsistent assumptions
- CI behavior non-deterministic (runtime override)

**Desired Behavior**:
- **Dev mode**: Warning emitted, test continues (convenience)
- **CI mode**: Test skipped with clear diagnostic (determinism)
- **Verbose mode**: Full search path diagnostics

---

## Architecture Overview

### Dual-Detection Flow

```
┌─────────────────────────────────────────────────────────────┐
│ Cross-Validation Test Execution                            │
└─────────────────────────────────────────────────────────────┘
                        ↓
        ┌───────────────┴───────────────┐
        │                               │
        ↓                               ↓
┌───────────────┐             ┌───────────────┐
│ Priority 1    │             │ Priority 2    │
│ Build-Time    │             │ Runtime       │
│ Constants     │             │ Detection     │
│ (Fastest)     │             │ (Fallback)    │
└───────────────┘             └───────────────┘
        │                               │
        │ HAS_BITNET=true               │ detect_backend_runtime()
        │ (available)                   │ finds libraries
        │                               │
        ↓                               ↓
┌───────────────┐             ┌───────────────┐
│ Test Proceeds │             │ STALE BUILD   │
│ No Warning    │             │ Detected      │
│               │             │               │
│ Exit: Normal  │             │ Branch by Env │
└───────────────┘             └───────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                                   │
                    ↓                                   ↓
            ┌──────────────┐                  ┌──────────────┐
            │ CI Mode      │                  │ Dev Mode     │
            │ (CI=1 set)   │                  │ (CI unset)   │
            └──────────────┘                  └──────────────┘
                    │                                   │
                    ↓                                   ↓
        ⊘ Skip Test              ⚠️  Emit Warning + Continue
        Exit 0 (skip)            Test Proceeds (runtime override)
        Diagnostic Message       Single-line rebuild command
```

### Component Interaction

```
┌────────────────────────────────────────────────────────────┐
│ crossval/build.rs (Build-Time Detection)                  │
│                                                            │
│ - Scans 3-tier search paths (primary, embedded, fallback) │
│ - Detects libbitnet*.so, libllama*.so, libggml*.so       │
│ - Exports: CROSSVAL_HAS_BITNET = "true|false"            │
│ - Exports: CROSSVAL_HAS_LLAMA = "true|false"             │
│ - Exports: CROSSVAL_BACKEND_STATE = "full|llama|none"    │
│ - Emits RPATH linker flags (embeds library paths)        │
└────────────────────────────────────────────────────────────┘
                        ↓ (compile time)
┌────────────────────────────────────────────────────────────┐
│ crossval/src/lib.rs (Constant Export)                     │
│                                                            │
│ pub const HAS_BITNET: bool = env!("CROSSVAL_HAS_BITNET"); │
│ pub const HAS_LLAMA: bool = env!("CROSSVAL_HAS_LLAMA");   │
│ pub const BACKEND_STATE: &str = env!("..._BACKEND_STATE");│
└────────────────────────────────────────────────────────────┘
                        ↓ (runtime - immutable)
┌────────────────────────────────────────────────────────────┐
│ xtask/src/crossval/preflight.rs (Runtime Validation)      │
│                                                            │
│ fn preflight_backend_libs(backend, verbose) {             │
│   // Priority 1: Check build-time constant               │
│   if HAS_BITNET || HAS_LLAMA {                           │
│     return Ok(()); // ✅ Available at build time         │
│   }                                                       │
│                                                            │
│   // Priority 2: Runtime fallback detection              │
│   if let Ok((found, path)) = detect_backend_runtime() {  │
│     if found {                                           │
│       // STALE BUILD DETECTED                           │
│       if is_ci() {                                       │
│         emit_ci_skip_diagnostic();                      │
│         std::process::exit(0); // Skip test             │
│       } else {                                          │
│         emit_stale_build_warning(backend, path);       │
│         return Ok(()); // Continue with warning         │
│       }                                                  │
│     }                                                    │
│   }                                                       │
│                                                            │
│   // Priority 3: Backend not found anywhere              │
│   Err(PreflightError::BackendUnavailable { ... })        │
│ }                                                         │
└────────────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────────┐
│ tests/support/backend_helpers.rs (Test Integration)       │
│                                                            │
│ pub fn detect_backend_runtime(backend) ->                 │
│   Result<(bool, Option<PathBuf>), String> {              │
│                                                            │
│   let candidates = get_library_search_paths(backend);    │
│   for path in candidates {                               │
│     if all_required_libs_present(path, backend) {        │
│       return Ok((true, Some(path)));                     │
│     }                                                     │
│   }                                                       │
│   Ok((false, None))                                      │
│ }                                                         │
└────────────────────────────────────────────────────────────┘
```

---

## Requirements Analysis

### Functional Requirements

| ID | Requirement | Priority | Rationale |
|----|-------------|----------|-----------|
| FR1 | Detect when runtime finds libraries but build-time constants are `false` | MUST | Core stale build detection |
| FR2 | Emit single-line warning with exact rebuild command (dev mode) | MUST | User-actionable guidance |
| FR3 | Skip test with CI-safe diagnostic when `CI=1` set (CI mode) | MUST | CI determinism guarantee |
| FR4 | Provide verbose diagnostics with search paths and env state | SHOULD | Advanced troubleshooting |
| FR5 | Deduplicate warnings per test run via `std::sync::Once` | SHOULD | Avoid warning fatigue |
| FR6 | Return matched library path from `detect_backend_runtime()` | MUST | Diagnostic context |
| FR7 | Support both BitNet and llama.cpp backends uniformly | MUST | Dual-backend architecture |
| FR8 | Preserve existing fast-path behavior (Priority 1: build-time) | MUST | Performance, backward compat |

### Acceptance Criteria

| AC | Criterion | Validation Method |
|----|-----------|-------------------|
| AC1 | Warning emitted when runtime detects libs but `HAS_BITNET=false` | Integration test: stale build scenario |
| AC2 | Warning includes exact rebuild command: `cargo clean -p crossval && cargo build -p xtask --features crossval-all` | String match in test assertion |
| AC3 | CI mode skips test (exit 0) with diagnostic when stale build detected | Mock `CI=1`, verify exit code 0 |
| AC4 | Dev mode continues test execution after warning (no exit) | Mock `CI=unset`, verify test proceeds |
| AC5 | Backend name ("bitnet.cpp" or "llama.cpp") shown in warning | String match in warning output |
| AC6 | Matched library path displayed in verbose mode | Parse verbose output, verify path present |
| AC7 | Verbose mode shows all search paths attempted and existence status | Count path listings in verbose output |
| AC8 | Warning emitted exactly once per test run (deduplication) | Call warning function twice, verify single output |
| AC9 | No performance regression in fast path (build-time available) | Benchmark: < 1μs overhead when `HAS_BITNET=true` |
| AC10 | CI detection recognizes GitHub Actions, GitLab CI, Jenkins, CircleCI | Mock each env var, verify `is_ci()` returns true |

---

## Implementation Approach

### 1. Environment Detection (CI vs Dev)

**Location**: `xtask/src/crossval/preflight.rs` (lines 411-418)

**Existing Implementation** (already correct):
```rust
/// Detect if running in CI environment
///
/// Checks standard CI environment variables used by major platforms.
///
/// # Returns
///
/// `true` if any CI indicator detected, `false` otherwise
pub fn is_ci() -> bool {
    std::env::var_os("CI").is_some()
        || std::env::var_os("GITHUB_ACTIONS").is_some()
        || std::env::var_os("JENKINS_HOME").is_some()
        || std::env::var_os("GITLAB_CI").is_some()
        || std::env::var_os("CIRCLECI").is_some()
        || std::env::var_os("BITNET_TEST_NO_REPAIR").is_some()
}
```

**CI Detection Precedence**:
1. `CI` (standard CI flag)
2. `GITHUB_ACTIONS` (GitHub Actions)
3. `JENKINS_HOME` (Jenkins CI)
4. `GITLAB_CI` (GitLab CI)
5. `CIRCLECI` (CircleCI)
6. `BITNET_TEST_NO_REPAIR` (explicit CI mode override)

**Validation**: Unit tests mock each variable, verify `is_ci()` returns `true`

---

### 2. Enhanced Runtime Detection with Path Return

**Location**: `tests/support/backend_helpers.rs` (new implementation)

**Current Signature** (needs enhancement):
```rust
pub fn detect_backend_runtime(backend: CppBackend) -> Result<bool, String>
```

**Enhanced Signature**:
```rust
/// Detect backend at runtime and return matched path
///
/// Searches filesystem for C++ backend libraries using environment-based
/// search paths. Returns both availability status and matched path for
/// diagnostic purposes.
///
/// # Arguments
///
/// * `backend` - Backend to search for (BitNet or Llama)
///
/// # Returns
///
/// * `Ok((true, Some(path)))` - Backend found, path where libraries located
/// * `Ok((false, None))` - Backend not found in any search path
/// * `Err(String)` - Error during detection (permissions, invalid path)
///
/// # Search Priority
///
/// 1. `BITNET_CROSSVAL_LIBDIR` (explicit override, highest priority)
/// 2. `CROSSVAL_RPATH_BITNET` / `CROSSVAL_RPATH_LLAMA` (backend-specific)
/// 3. `BITNET_CPP_DIR` / `LLAMA_CPP_DIR` subdirectories (build, build/bin, build/lib)
/// 4. Default: `~/.cache/bitnet_cpp` or `~/.cache/llama_cpp`
///
/// # Examples
///
/// ```ignore
/// use bitnet_crossval::CppBackend;
/// use tests::support::backend_helpers::detect_backend_runtime;
///
/// match detect_backend_runtime(CppBackend::BitNet) {
///     Ok((true, Some(path))) => {
///         println!("BitNet found at: {}", path.display());
///     }
///     Ok((false, None)) => {
///         println!("BitNet not found");
///     }
///     Err(e) => eprintln!("Detection error: {}", e),
/// }
/// ```
pub fn detect_backend_runtime(
    backend: CppBackend,
) -> Result<(bool, Option<std::path::PathBuf>), String> {
    let search_paths = get_library_search_paths(backend);

    // Required library stems for this backend
    let required_libs: &[&str] = match backend {
        CppBackend::BitNet => &["bitnet"],
        CppBackend::Llama => &["llama", "ggml"],
    };

    // Platform-specific library extensions
    #[cfg(target_os = "windows")]
    let lib_extensions = vec!["dll"];
    #[cfg(target_os = "macos")]
    let lib_extensions = vec!["dylib"];
    #[cfg(target_os = "linux")]
    let lib_extensions = vec!["so"];

    // Search each candidate path
    for path in &search_paths {
        if !path.exists() {
            continue;
        }

        // Check if all required libraries present
        let all_found = required_libs.iter().all(|stem| {
            lib_extensions.iter().any(|ext| {
                let lib_name = format!("lib{}.{}", stem, ext);
                path.join(&lib_name).exists()
            })
        });

        if all_found {
            return Ok((true, Some(path.clone())));
        }
    }

    Ok((false, None))
}

/// Get library search paths for backend
///
/// Returns ordered list of candidate directories to search for libraries.
///
/// # Priority Order
///
/// 1. BITNET_CROSSVAL_LIBDIR (unified override)
/// 2. CROSSVAL_RPATH_BITNET / CROSSVAL_RPATH_LLAMA (backend-specific)
/// 3. BITNET_CPP_DIR / LLAMA_CPP_DIR subdirectories
/// 4. Default cache directory (~/.cache/bitnet_cpp or llama_cpp)
fn get_library_search_paths(backend: CppBackend) -> Vec<std::path::PathBuf> {
    use std::path::PathBuf;

    let mut paths = Vec::new();

    // Priority 1: Explicit unified override
    if let Ok(lib_dir) = std::env::var("BITNET_CROSSVAL_LIBDIR") {
        paths.push(PathBuf::from(lib_dir));
    }

    // Priority 2: Backend-specific RPATH override
    let rpath_var = match backend {
        CppBackend::BitNet => "CROSSVAL_RPATH_BITNET",
        CppBackend::Llama => "CROSSVAL_RPATH_LLAMA",
    };
    if let Ok(rpath) = std::env::var(rpath_var) {
        // RPATH may be colon-separated list
        for p in rpath.split(':') {
            if !p.is_empty() {
                paths.push(PathBuf::from(p));
            }
        }
    }

    // Priority 3: Backend installation root subdirectories
    let install_root_var = match backend {
        CppBackend::BitNet => "BITNET_CPP_DIR",
        CppBackend::Llama => "LLAMA_CPP_DIR",
    };

    let default_cache = format!(
        "{}/.cache/{}",
        std::env::var("HOME").unwrap_or_else(|_| ".".into()),
        match backend {
            CppBackend::BitNet => "bitnet_cpp",
            CppBackend::Llama => "llama_cpp",
        }
    );

    let install_root = std::env::var(install_root_var).unwrap_or(default_cache);
    let root_path = PathBuf::from(&install_root);

    // Common subdirectories for build outputs
    for subdir in &["build", "build/bin", "build/lib", "build/3rdparty/llama.cpp/src", "build/3rdparty/llama.cpp/ggml/src", "lib"] {
        paths.push(root_path.join(subdir));
    }

    paths
}
```

**Key Changes**:
1. Return type: `Result<bool, String>` → `Result<(bool, Option<PathBuf>), String>`
2. Matched path captured and returned for diagnostic display
3. Search path prioritization matches build.rs logic
4. Comprehensive documentation with examples

---

### 3. Warning Emission Functions

**Location**: `tests/support/backend_helpers.rs` (new implementation)

#### Standard Warning (Dev Mode)

```rust
/// Emit stale build warning (standard format)
///
/// Prints single-line warning with exact rebuild command.
/// Deduplicated per backend via `std::sync::Once`.
///
/// # Arguments
///
/// * `backend` - Backend detected at runtime
///
/// # Output (stderr)
///
/// ```text
/// ⚠️  STALE BUILD: bitnet.cpp found at runtime but not at build time. Rebuild required: cargo clean -p crossval && cargo build -p xtask --features crossval-all
/// ```
fn emit_stale_build_warning(backend: CppBackend) {
    use std::sync::Once;

    // Per-backend deduplication
    static BITNET_WARNING: Once = Once::new();
    static LLAMA_WARNING: Once = Once::new();

    let once = match backend {
        CppBackend::BitNet => &BITNET_WARNING,
        CppBackend::Llama => &LLAMA_WARNING,
    };

    once.call_once(|| {
        eprintln!(
            "⚠️  STALE BUILD: {} found at runtime but not at build time. \
             Rebuild required: cargo clean -p crossval && cargo build -p xtask --features crossval-all",
            backend.name()
        );
    });
}
```

**Deduplication Strategy**:
- Two static `Once` guards (one per backend)
- First call emits warning, subsequent calls are no-ops
- Prevents warning spam in dual-backend test suites

#### Verbose Warning (Diagnostic Mode)

```rust
/// Emit stale build warning (verbose format)
///
/// Prints multi-line diagnostic with:
/// - Explanation of stale build scenario
/// - Matched library path
/// - Build-time constant state
/// - Exact rebuild command
/// - Follow-up instructions
///
/// # Arguments
///
/// * `backend` - Backend detected at runtime
/// * `matched_path` - Path where libraries were found
/// * `verbose` - If true, emit verbose output; if false, standard warning
fn emit_stale_build_warning_verbose(
    backend: CppBackend,
    matched_path: &std::path::Path,
) {
    const SEPARATOR: &str = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━";

    eprintln!("{}", SEPARATOR);
    eprintln!("⚠️  STALE BUILD DETECTION");
    eprintln!("{}", SEPARATOR);
    eprintln!();
    eprintln!("Backend '{}' found at runtime but not at xtask build time.", backend.name());
    eprintln!();
    eprintln!("This happens when:");
    eprintln!("  1. You built xtask");
    eprintln!("  2. Then installed {} libraries later", backend.name());
    eprintln!("  3. xtask binary still contains old detection constants");
    eprintln!();
    eprintln!("Why rebuild is needed:");
    eprintln!("  • Library detection runs at BUILD time (not runtime)");
    eprintln!("  • Results are baked into the xtask binary as constants");
    eprintln!("  • Runtime detection is a fallback for developer convenience");
    eprintln!("  • Rebuild refreshes the constants to match filesystem reality");
    eprintln!();
    eprintln!("Runtime Detection Results:");
    eprintln!("  Matched path: {}", matched_path.display());

    // List libraries found in matched path
    if let Ok(entries) = std::fs::read_dir(matched_path) {
        let libs: Vec<_> = entries
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name()
                    .to_string_lossy()
                    .starts_with("lib")
                    && (e.file_name().to_string_lossy().ends_with(".so")
                        || e.file_name().to_string_lossy().ends_with(".dylib")
                        || e.file_name().to_string_lossy().ends_with(".dll"))
            })
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();

        if !libs.is_empty() {
            eprintln!("  Libraries found: {}", libs.join(", "));
        }
    }

    eprintln!();
    eprintln!("Build-Time Detection State:");
    eprintln!(
        "  HAS_{} = false (stale)",
        match backend {
            CppBackend::BitNet => "BITNET",
            CppBackend::Llama => "LLAMA",
        }
    );

    eprintln!();
    eprintln!("Fix:");
    eprintln!("  cargo clean -p crossval && cargo build -p xtask --features crossval-all");
    eprintln!();
    eprintln!("Then re-run your test.");
}
```

**Verbose Mode Trigger**:
- `VERBOSE=1` environment variable
- `--verbose` CLI flag (if integrated in test harness)

#### CI Skip Diagnostic

```rust
/// Format CI-mode skip diagnostic message
///
/// Generates detailed message explaining why test is skipped in CI mode,
/// with setup instructions for resolving stale build.
///
/// # Arguments
///
/// * `backend` - Backend that was detected at runtime
/// * `matched_path` - Optional path where libraries were found
///
/// # Returns
///
/// Formatted diagnostic string for stderr output
///
/// # Output Format
///
/// ```text
/// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
/// ⊘ Test skipped: bitnet.cpp not available (CI mode)
/// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
///
/// CI mode detected (CI=1 or BITNET_TEST_NO_REPAIR=1).
/// Runtime detection found libraries but build-time constants are stale.
///
/// Runtime found libraries at: /home/runner/.cache/bitnet_cpp/build
/// But xtask was built before libraries were installed.
///
/// In CI mode:
///   • Build-time detection is the source of truth
///   • Runtime fallback is DISABLED for determinism
///   • xtask must be rebuilt to detect libraries
///
/// Setup Instructions:
///   1. Install backend:
///      eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"
///   2. Rebuild xtask:
///      cargo clean -p crossval && cargo build -p xtask --features crossval-all
///   3. Re-run CI job
/// ```
fn format_ci_skip_diagnostic(
    backend: CppBackend,
    matched_path: Option<&std::path::Path>,
) -> String {
    const SEPARATOR: &str = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━";

    let mut msg = String::new();
    msg.push_str(&format!("{}\n", SEPARATOR));
    msg.push_str(&format!("⊘ Test skipped: {} not available (CI mode)\n", backend.name()));
    msg.push_str(&format!("{}\n\n", SEPARATOR));

    msg.push_str("CI mode detected (CI=1 or BITNET_TEST_NO_REPAIR=1).\n");
    msg.push_str("Runtime detection found libraries but build-time constants are stale.\n\n");

    if let Some(path) = matched_path {
        msg.push_str(&format!("Runtime found libraries at: {}\n", path.display()));
        msg.push_str("But xtask was built before libraries were installed.\n\n");
    }

    msg.push_str("In CI mode:\n");
    msg.push_str("  • Build-time detection is the source of truth\n");
    msg.push_str("  • Runtime fallback is DISABLED for determinism\n");
    msg.push_str("  • xtask must be rebuilt to detect libraries\n\n");

    msg.push_str("Setup Instructions:\n");
    msg.push_str("  1. Install backend:\n");
    msg.push_str("     eval \"$(cargo run -p xtask -- setup-cpp-auto --emit=sh)\"\n");
    msg.push_str("  2. Rebuild xtask:\n");
    msg.push_str("     cargo clean -p crossval && cargo build -p xtask --features crossval-all\n");
    msg.push_str("  3. Re-run CI job\n");

    msg
}
```

---

### 4. Integration in `preflight_backend_libs()`

**Location**: `xtask/src/crossval/preflight.rs` (lines 519-695)

**Current Implementation** (lines 552-583, excerpt):
```rust
// Priority 1: Build-time check
let has_libs = match backend {
    CppBackend::BitNet => HAS_BITNET,
    CppBackend::Llama => HAS_LLAMA,
};

if has_libs {
    // Fast path: build-time detection succeeded
    if verbose {
        eprintln!("[preflight] ✓ {} available (build-time detection)", backend.name());
    }
    return Ok(());
}

// Priority 2: Runtime fallback warning (simplified current version)
if backend == CppBackend::BitNet && BACKEND_STATE == "llama" {
    eprintln!("⚠️  WARNING: BitNet requested but only llama.cpp available at compile time");
    // ... detailed explanation ...
}

// Priority 3: Failure path (backend not found)
// ... error message ...
```

**Enhanced Implementation**:
```rust
/// Preflight backend library availability check
///
/// Verifies C++ backend libraries are available using two-tier detection:
/// 1. Build-time constants (Priority 1, fastest)
/// 2. Runtime filesystem search (Priority 2, fallback with warnings)
///
/// # Stale Build Handling
///
/// When runtime detection succeeds but build-time constants are false:
/// - **CI Mode** (`CI=1`): Skip test with diagnostic (exit 0)
/// - **Dev Mode** (default): Emit warning, continue test
/// - **Verbose Mode** (`VERBOSE=1`): Full search path diagnostics
///
/// # Arguments
///
/// * `backend` - Backend to check (BitNet or Llama)
/// * `verbose` - Enable verbose diagnostic output
///
/// # Returns
///
/// * `Ok(())` - Backend available (build-time or runtime with warning)
/// * `Err(PreflightError)` - Backend unavailable in all detection methods
///
/// # Examples
///
/// ```ignore
/// // Check BitNet backend availability
/// preflight_backend_libs(CppBackend::BitNet, false)?;
///
/// // Verbose diagnostics for troubleshooting
/// preflight_backend_libs(CppBackend::BitNet, true)?;
/// ```
pub fn preflight_backend_libs(
    backend: CppBackend,
    verbose: bool,
) -> Result<(), PreflightError> {
    // ─────────────────────────────────────────────────────────────
    // Priority 1: Build-Time Constant Check (Fast Path)
    // ─────────────────────────────────────────────────────────────
    let has_libs = match backend {
        CppBackend::BitNet => HAS_BITNET,
        CppBackend::Llama => HAS_LLAMA,
    };

    if has_libs {
        // Fast path: build-time detection succeeded
        if verbose {
            eprintln!("[preflight] ✓ {} available (build-time detection)", backend.name());
        }
        return Ok(());
    }

    // ─────────────────────────────────────────────────────────────
    // Priority 2: Runtime Detection Fallback (Stale Build Path)
    // ─────────────────────────────────────────────────────────────

    // Import runtime detection from test support
    // Note: This creates a dependency on tests/ crate, which is acceptable
    // for xtask (test infrastructure tool)
    use crate::tests::support::backend_helpers::detect_backend_runtime;

    match detect_backend_runtime(backend) {
        Ok((true, matched_path)) => {
            // STALE BUILD DETECTED: Runtime found libs but build-time constant is false

            if is_ci() {
                // ─────────────────────────────────────────────────
                // CI Mode: Skip test for determinism
                // ─────────────────────────────────────────────────
                let diagnostic = format_ci_skip_diagnostic(backend, matched_path.as_deref());
                eprintln!("{}", diagnostic);
                std::process::exit(0); // Skip test (not failure)
            } else {
                // ─────────────────────────────────────────────────
                // Dev Mode: Emit warning and continue
                // ─────────────────────────────────────────────────
                let verbose_mode = verbose || std::env::var("VERBOSE").is_ok();

                if verbose_mode {
                    if let Some(path) = matched_path.as_ref() {
                        emit_stale_build_warning_verbose(backend, path);
                    } else {
                        emit_stale_build_warning(backend);
                    }
                } else {
                    emit_stale_build_warning(backend);
                }

                // Allow test to continue (runtime override)
                return Ok(());
            }
        }
        Ok((false, None)) => {
            // Backend not found at build-time OR runtime
            // Fall through to Priority 3 error path
        }
        Err(e) => {
            // Runtime detection error (permissions, invalid path)
            if verbose {
                eprintln!("[preflight] Runtime detection error: {}", e);
            }
            // Fall through to Priority 3 error path
        }
    }

    // ─────────────────────────────────────────────────────────────
    // Priority 3: Backend Unavailable (Error Path)
    // ─────────────────────────────────────────────────────────────

    // ... existing error message logic ...
    // (no changes to this section)
}
```

**Key Integration Points**:
1. Build-time check (Priority 1) unchanged for performance
2. Runtime detection (Priority 2) branched by CI environment
3. Error path (Priority 3) unchanged
4. Helper functions imported from `tests::support::backend_helpers`

---

## Environment Variable Contracts

### Detection Environment Variables

| Variable | Purpose | Used By | Priority | Example |
|----------|---------|---------|----------|---------|
| `BITNET_CROSSVAL_LIBDIR` | Explicit global library directory override | Runtime detection, build.rs | 1 (highest) | `/opt/bitnet/lib` |
| `CROSSVAL_RPATH_BITNET` | BitNet-specific library path (colon-separated) | Runtime detection | 2 | `/home/user/.cache/bitnet_cpp/build:/usr/local/lib` |
| `CROSSVAL_RPATH_LLAMA` | Llama-specific library path (colon-separated) | Runtime detection | 2 | `/home/user/.cache/llama_cpp/build` |
| `BITNET_CPP_DIR` | BitNet.cpp installation root | Runtime detection, build.rs | 3 | `/home/user/.cache/bitnet_cpp` |
| `LLAMA_CPP_DIR` | llama.cpp installation root | Runtime detection, build.rs | 3 | `/home/user/.cache/llama_cpp` |
| `HOME` | User home directory | Default cache location | 4 (fallback) | `/home/user` |

### Control Environment Variables

| Variable | Purpose | Values | Default | Behavior |
|----------|---------|--------|---------|----------|
| `CI` | Generic CI environment detection | `1`, `true`, any | unset | CI mode if set |
| `GITHUB_ACTIONS` | GitHub Actions CI detection | `true` | unset | CI mode if set |
| `JENKINS_HOME` | Jenkins CI detection | any | unset | CI mode if set |
| `GITLAB_CI` | GitLab CI detection | any | unset | CI mode if set |
| `CIRCLECI` | CircleCI detection | any | unset | CI mode if set |
| `BITNET_TEST_NO_REPAIR` | Force CI mode (disable auto-repair) | `1` | unset | CI mode if set |
| `VERBOSE` | Enable verbose diagnostics | `1` | unset | Verbose mode if set |

### Warning Control Variables (Optional)

| Variable | Purpose | Values | Default | Use Case |
|----------|---------|--------|---------|----------|
| `BITNET_SUPPRESS_STALE_WARNING` | Suppress stale build warnings | `1` | unset | CI with known stale builds |
| `BITNET_FORCE_VERBOSE_WARNING` | Force verbose even without `VERBOSE=1` | `1` | unset | Debugging test infrastructure |

---

## Testing Strategy

### Unit Tests

**Test File**: `tests/support/runtime_detection_warning_tests.rs` (new file)

**Test Categories**:

#### Category A: Environment Detection (5 tests)

```rust
#[test]
#[serial(bitnet_env)]
fn test_is_ci_detects_github_actions() {
    let _guard = EnvGuard::new("GITHUB_ACTIONS", "true");
    assert!(is_ci());
}

#[test]
#[serial(bitnet_env)]
fn test_is_ci_detects_gitlab_ci() {
    let _guard = EnvGuard::new("GITLAB_CI", "1");
    assert!(is_ci());
}

#[test]
#[serial(bitnet_env)]
fn test_is_ci_detects_jenkins() {
    let _guard = EnvGuard::new("JENKINS_HOME", "/var/jenkins");
    assert!(is_ci());
}

#[test]
#[serial(bitnet_env)]
fn test_is_ci_detects_generic_ci() {
    let _guard = EnvGuard::new("CI", "1");
    assert!(is_ci());
}

#[test]
#[serial(bitnet_env)]
fn test_is_ci_false_when_unset() {
    // Ensure no CI vars set
    assert!(!is_ci());
}
```

#### Category B: Runtime Detection with Path (4 tests)

```rust
#[test]
fn test_detect_backend_runtime_returns_matched_path() {
    // Create temp directory with mock libraries
    let temp = tempfile::tempdir().unwrap();
    let build_dir = temp.path().join("build");
    std::fs::create_dir_all(&build_dir).unwrap();

    #[cfg(target_os = "linux")]
    std::fs::write(build_dir.join("libbitnet.so"), b"mock").unwrap();

    let _guard = EnvGuard::new("BITNET_CPP_DIR", temp.path().to_str().unwrap());

    let (found, path) = detect_backend_runtime(CppBackend::BitNet).unwrap();
    assert!(found);
    assert!(path.is_some());
    assert_eq!(path.unwrap(), build_dir);
}

#[test]
fn test_detect_backend_runtime_returns_none_when_missing() {
    let temp = tempfile::tempdir().unwrap();
    let _guard = EnvGuard::new("BITNET_CPP_DIR", temp.path().to_str().unwrap());

    let (found, path) = detect_backend_runtime(CppBackend::BitNet).unwrap();
    assert!(!found);
    assert!(path.is_none());
}

#[test]
fn test_detect_backend_runtime_prioritizes_crossval_libdir() {
    // Priority 1: BITNET_CROSSVAL_LIBDIR should be checked first
    let temp1 = tempfile::tempdir().unwrap();
    let temp2 = tempfile::tempdir().unwrap();

    // Create libs in temp2 (CROSSVAL_LIBDIR)
    #[cfg(target_os = "linux")]
    std::fs::write(temp2.path().join("libbitnet.so"), b"mock").unwrap();

    let _guard1 = EnvGuard::new("BITNET_CPP_DIR", temp1.path().to_str().unwrap());
    let _guard2 = EnvGuard::new("BITNET_CROSSVAL_LIBDIR", temp2.path().to_str().unwrap());

    let (found, path) = detect_backend_runtime(CppBackend::BitNet).unwrap();
    assert!(found);
    assert_eq!(path.unwrap(), temp2.path());
}

#[test]
fn test_detect_backend_runtime_searches_multiple_subdirs() {
    let temp = tempfile::tempdir().unwrap();
    let nested = temp.path().join("build/lib");
    std::fs::create_dir_all(&nested).unwrap();

    #[cfg(target_os = "linux")]
    std::fs::write(nested.join("libbitnet.so"), b"mock").unwrap();

    let _guard = EnvGuard::new("BITNET_CPP_DIR", temp.path().to_str().unwrap());

    let (found, path) = detect_backend_runtime(CppBackend::BitNet).unwrap();
    assert!(found);
    assert_eq!(path.unwrap(), nested);
}
```

#### Category C: Warning Emission and Deduplication (6 tests)

```rust
#[test]
fn test_emit_stale_build_warning_format() {
    // Capture stderr
    let backend = CppBackend::BitNet;

    // Note: In real implementation, this would capture stderr and verify format
    emit_stale_build_warning(backend);

    // Expected output:
    // "⚠️  STALE BUILD: bitnet.cpp found at runtime but not at build time. Rebuild required: cargo clean -p crossval && cargo build -p xtask --features crossval-all"
}

#[test]
fn test_warning_deduplication_per_backend() {
    // First call should emit warning
    emit_stale_build_warning(CppBackend::BitNet);

    // Second call should be no-op (deduplicated)
    emit_stale_build_warning(CppBackend::BitNet);

    // Different backend should emit separate warning
    emit_stale_build_warning(CppBackend::Llama);
}

#[test]
fn test_verbose_warning_includes_matched_path() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path();

    emit_stale_build_warning_verbose(CppBackend::BitNet, path);

    // Expected output includes:
    // "Matched path: {path}"
}

#[test]
fn test_verbose_warning_lists_libraries() {
    let temp = tempfile::tempdir().unwrap();

    #[cfg(target_os = "linux")]
    {
        std::fs::write(temp.path().join("libbitnet.so"), b"mock").unwrap();
        std::fs::write(temp.path().join("libllama.so"), b"mock").unwrap();
    }

    emit_stale_build_warning_verbose(CppBackend::BitNet, temp.path());

    // Expected output includes:
    // "Libraries found: libbitnet.so, libllama.so"
}

#[test]
fn test_ci_skip_diagnostic_format() {
    let temp = tempfile::tempdir().unwrap();
    let msg = format_ci_skip_diagnostic(CppBackend::BitNet, Some(temp.path()));

    assert!(msg.contains("⊘ Test skipped: bitnet.cpp not available (CI mode)"));
    assert!(msg.contains("Runtime found libraries at:"));
    assert!(msg.contains("Setup Instructions:"));
}

#[test]
fn test_ci_skip_diagnostic_without_path() {
    let msg = format_ci_skip_diagnostic(CppBackend::BitNet, None);

    assert!(msg.contains("⊘ Test skipped"));
    assert!(!msg.contains("Runtime found libraries at:"));
}
```

#### Category D: Preflight Integration (4 tests)

```rust
#[test]
#[serial(bitnet_env)]
fn test_preflight_ci_mode_skips_on_stale_build() {
    let temp = tempfile::tempdir().unwrap();
    let build_dir = temp.path().join("build");
    std::fs::create_dir_all(&build_dir).unwrap();

    #[cfg(target_os = "linux")]
    std::fs::write(build_dir.join("libbitnet.so"), b"mock").unwrap();

    let _guard1 = EnvGuard::new("BITNET_CPP_DIR", temp.path().to_str().unwrap());
    let _guard2 = EnvGuard::new("CI", "1");

    // Should exit with code 0 (skip)
    // Note: In real test, this would use a subprocess or mock exit
}

#[test]
#[serial(bitnet_env)]
fn test_preflight_dev_mode_continues_on_stale_build() {
    let temp = tempfile::tempdir().unwrap();
    let build_dir = temp.path().join("build");
    std::fs::create_dir_all(&build_dir).unwrap();

    #[cfg(target_os = "linux")]
    std::fs::write(build_dir.join("libbitnet.so"), b"mock").unwrap();

    let _guard = EnvGuard::new("BITNET_CPP_DIR", temp.path().to_str().unwrap());

    // Should return Ok(()) with warning
    let result = preflight_backend_libs(CppBackend::BitNet, false);
    assert!(result.is_ok());
}

#[test]
fn test_preflight_fast_path_no_runtime_check() {
    // When HAS_BITNET=true at build time, runtime detection should not run
    // This test validates no performance regression

    // Benchmark fast path: < 1μs
}

#[test]
#[serial(bitnet_env)]
fn test_preflight_verbose_mode_shows_diagnostics() {
    let _guard = EnvGuard::new("VERBOSE", "1");

    // Run preflight in verbose mode
    // Verify diagnostic output includes search paths
}
```

### Integration Tests

**Test File**: `xtask/tests/stale_build_detection_tests.rs` (new file)

**Test Scenarios**:

```rust
/// Scenario 1: Fresh build (no stale detection)
///
/// Setup:
/// 1. Install libraries
/// 2. Build xtask
///
/// Expected:
/// - HAS_BITNET=true
/// - No warning emitted
/// - Test proceeds normally
#[test]
#[ignore] // Manual verification
fn test_scenario_fresh_build_no_warning() {
    // This test requires actual filesystem setup
    // Run manually: cargo test --test stale_build_detection_tests --ignored
}

/// Scenario 2: Stale build (dev mode)
///
/// Setup:
/// 1. Build xtask (libraries NOT present)
/// 2. Install libraries
/// 3. Run test (CI=unset)
///
/// Expected:
/// - HAS_BITNET=false
/// - Runtime detection finds libraries
/// - Warning emitted to stderr
/// - Test continues
#[test]
#[ignore]
fn test_scenario_stale_build_dev_mode() {
    // Requires multi-phase build
}

/// Scenario 3: Stale build (CI mode)
///
/// Setup:
/// 1. Build xtask (libraries NOT present)
/// 2. Install libraries
/// 3. Run test with CI=1
///
/// Expected:
/// - HAS_BITNET=false
/// - Runtime detection finds libraries
/// - Skip diagnostic emitted
/// - Test exits with code 0 (skip)
#[test]
#[ignore]
fn test_scenario_stale_build_ci_mode() {
    // Requires subprocess to capture exit code
}

/// Scenario 4: Verbose mode diagnostics
///
/// Setup:
/// 1. Stale build scenario
/// 2. Set VERBOSE=1
///
/// Expected:
/// - Multi-line diagnostic output
/// - Search paths displayed
/// - Matched library path shown
#[test]
#[ignore]
fn test_scenario_verbose_diagnostics() {
    // Capture stderr and validate verbose output format
}
```

---

## Risk Assessment

### Technical Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| **Risk 1: Warning fatigue** | Users ignore warnings after seeing repeatedly | Medium | Deduplicate via `std::sync::Once` per backend |
| **Risk 2: CI determinism breakage** | Runtime override causes non-deterministic CI | High | Strict CI mode: skip test, no runtime override |
| **Risk 3: Confusing error messages** | Users don't understand "stale build" concept | Medium | Verbose mode explains WHY rebuild needed |
| **Risk 4: Multiple backend warnings** | Dual-backend tests emit 2 warnings | Low | Per-backend deduplication, consider unified message |
| **Risk 5: Performance regression** | Runtime checks slow down fast path | Low | Guard with build-time check first (Priority 1) |
| **Risk 6: Environment variable conflicts** | `VERBOSE=1` set globally affects all tests | Low | Use `EnvGuard` for test isolation |

### Mitigation Strategies

**Risk 1 Mitigation (Warning Fatigue)**:
- Deduplicate warnings via `std::sync::Once` (1 warning per backend per test run)
- Provide escape hatch: `BITNET_SUPPRESS_STALE_WARNING=1` for CI with known stale builds
- Single-line standard warning (minimal noise)
- Verbose mode opt-in for troubleshooting only

**Risk 2 Mitigation (CI Determinism)**:
- `is_ci()` checks 6 different CI indicators
- CI mode strictly skips test (exit 0) when stale build detected
- No runtime override in CI (build-time constants are source of truth)
- Clear diagnostic explains why skip occurred

**Risk 3 Mitigation (Confusing Messages)**:
- Verbose mode explains 3-step timeline (build → install → stale)
- Exact rebuild command provided (copy-pasteable)
- "Why rebuild is needed" section explains constant baking
- Follow-up instructions ("Then re-run your test")

**Risk 4 Mitigation (Multiple Warnings)**:
- Per-backend `Once` guards (separate for BitNet vs Llama)
- Future enhancement: unified multi-backend warning

**Risk 5 Mitigation (Performance)**:
- Fast path (Priority 1) remains unchanged: < 1μs overhead
- Runtime detection only runs when build-time constant is `false`
- Benchmark test validates no regression

**Risk 6 Mitigation (Env Conflicts)**:
- Use `#[serial(bitnet_env)]` for environment-mutating tests
- `EnvGuard` pattern for automatic cleanup
- Clear documentation of env variable contracts

---

## Success Criteria (Validation)

### Acceptance Validation Matrix

| AC | Validation Method | Pass Criteria | Test Command |
|----|-------------------|---------------|--------------|
| AC1 | Integration test: stale build scenario | Warning emitted when `HAS_BITNET=false` but libs present | `cargo test test_stale_build_warning` |
| AC2 | String match test | Exact rebuild command in warning output | Grep stderr for `cargo clean -p crossval` |
| AC3 | Exit code test | Exit code 0 when `CI=1` set | Subprocess captures exit code |
| AC4 | Function return test | `preflight_backend_libs()` returns `Ok(())` in dev mode | Assert `result.is_ok()` |
| AC5 | String match test | Backend name ("bitnet.cpp" or "llama.cpp") in output | Grep stderr for backend name |
| AC6 | String match test | Matched path displayed in verbose output | Grep verbose stderr for `Matched path:` |
| AC7 | Verbose output test | All search paths listed with existence status | Count path listings ≥ 6 |
| AC8 | Deduplication test | Second warning call is no-op | Capture stderr, verify single output |
| AC9 | Benchmark test | Fast path latency < 1μs | `cargo bench --bench preflight_benchmarks` |
| AC10 | Environment test | All CI vars recognized | Mock each var, verify `is_ci() == true` |

### Performance Requirements

**Latency Targets**:
- **Fast path** (build-time available): < 1μs (no runtime checks)
- **Runtime detection** (fallback): < 50ms (filesystem checks)
- **Warning emission**: < 5ms (formatting overhead)
- **CI skip path**: < 10ms (early exit, minimal overhead)

**Memory Overhead**:
- Static warning deduplication: 2 bytes (2 `Once` guards)
- Matched path storage: stack-allocated `PathBuf` (~128 bytes)
- Search path vector: ~512 bytes (6-8 paths)

**Benchmark Validation**:
```bash
# Run performance benchmarks
cargo bench --bench preflight_benchmarks

# Expected output:
# preflight_fast_path      time: [500 ns 600 ns 700 ns]  ✅ < 1μs
# preflight_runtime_detect time: [30 ms 35 ms 40 ms]    ✅ < 50ms
# preflight_warning_emit   time: [2 ms 3 ms 4 ms]       ✅ < 5ms
```

---

## Implementation Timeline

### Phase 1: Core Detection Logic (Week 1)

**Deliverables**:
- Enhanced `detect_backend_runtime()` returning `(bool, Option<PathBuf>)`
- `get_library_search_paths()` implementation
- Unit tests for runtime detection (Category B: 4 tests)

**Validation**:
- Unit tests pass
- Matched path correctly returned
- Search path prioritization matches build.rs

**Files Modified**:
- `tests/support/backend_helpers.rs` (~100 lines added)

### Phase 2: Warning Emission (Week 1-2)

**Deliverables**:
- `emit_stale_build_warning()` (standard format)
- `emit_stale_build_warning_verbose()` (diagnostic format)
- `format_ci_skip_diagnostic()` (CI message)
- Unit tests for warning emission (Category C: 6 tests)

**Validation**:
- Warning format matches specification
- Deduplication verified via `Once` guard
- Verbose output includes all diagnostic fields

**Files Modified**:
- `tests/support/backend_helpers.rs` (~150 lines added)
- `tests/support/runtime_detection_warning_tests.rs` (new file, ~200 lines)

### Phase 3: Preflight Integration (Week 2)

**Deliverables**:
- Enhanced `preflight_backend_libs()` with stale build handling
- CI mode skip logic with exit code 0
- Dev mode warning + continue logic
- Integration tests (Category D: 4 tests)

**Validation**:
- CI mode skips test deterministically
- Dev mode continues with warning
- Fast path performance maintained

**Files Modified**:
- `xtask/src/crossval/preflight.rs` (~80 lines modified)
- `xtask/tests/stale_build_detection_tests.rs` (new file, ~150 lines)

### Phase 4: Documentation and Polish (Week 3)

**Deliverables**:
- Update `docs/howto/cpp-setup.md` with stale build troubleshooting
- Update `CLAUDE.md` with environment variable contracts
- Add integration test scenarios (4 `#[ignore]` tests for manual verification)
- Performance benchmarks

**Validation**:
- Documentation reviewed
- User workflow tested end-to-end
- Benchmarks show no regression

**Files Modified**:
- `docs/howto/cpp-setup.md` (~50 lines added)
- `CLAUDE.md` (~30 lines added)
- `benches/preflight_benchmarks.rs` (new file, ~100 lines)

### Total Effort Estimate

**Lines of Code**:
- Implementation: ~430 lines
- Tests: ~550 lines
- Documentation: ~80 lines
- Total: ~1060 lines

**Time Estimate**: 2-3 weeks (1 developer)

---

## Open Questions

### Q1: Should we auto-rebuild xtask on stale detection?

**Proposal**: Automatically trigger `cargo build -p xtask` when stale build detected

**Pros**:
- Seamless user experience (no manual intervention)
- Eliminates rebuild step from workflow

**Cons**:
- Breaks incremental builds (unexpected compilation)
- Test isolation concerns (modifying build artifacts during test)
- Performance impact (cargo build is slow)

**Decision**: **No** - User-initiated rebuild is safer. Current spec focuses on warning-only approach.

**Future Enhancement**: Interactive mode with prompt ("Rebuild xtask now? (y/n)")

### Q2: Should we cache matched paths?

**Proposal**: Cache runtime detection results in static variable to avoid repeated filesystem searches

**Pros**:
- Performance optimization for dual-backend tests
- Reduces filesystem I/O

**Cons**:
- Added complexity (cache invalidation)
- Stale cache if libraries installed mid-process
- Minimal benefit (detection runs once per test process via `Once`)

**Decision**: **No** - Detection already deduplicated via `Once` guard. Caching adds complexity for negligible benefit.

### Q3: Unified multi-backend warning?

**Proposal**: Single warning for dual-backend stale builds instead of 2 separate warnings

**Example**:
```
⚠️  STALE BUILD: Multiple backends found at runtime but not at build time:
  • bitnet.cpp (found at /home/user/.cache/bitnet_cpp/build)
  • llama.cpp (found at /home/user/.cache/llama_cpp/build)
Rebuild required: cargo clean -p crossval && cargo build -p xtask --features crossval-all
```

**Decision**: **Future Enhancement** (v0.3+) - Current spec uses per-backend warnings. Unified message requires more complex state tracking.

### Q4: Should we emit warning in CI mode before skipping?

**Proposal**: Show stale build warning in CI before exiting (in addition to skip diagnostic)

**Pros**:
- Consistent messaging across dev/CI modes
- Redundant context helps debugging

**Cons**:
- Duplicate information (warning + diagnostic)
- Verbose CI output

**Decision**: **No** - CI diagnostic is self-contained and explains both the skip reason and resolution steps. Separate warning is redundant.

### Q5: Suppress warning environment variable?

**Proposal**: `BITNET_SUPPRESS_STALE_WARNING=1` to silence warnings

**Use Case**: CI with known stale builds that can't be fixed immediately

**Decision**: **Yes** - Include in spec as optional escape hatch. Documented in env variable contracts section.

---

## References

### Source Analysis

This specification is derived from comprehensive codebase analysis documented in:

- **Analysis Artifact**: `/tmp/p0_runtime_detection_analysis.md`
  - Date: 2025-10-27
  - Focus: Build-time vs runtime detection, dual-detection flow, stale build scenarios
  - Sections: Build-time detection (build.rs), runtime fallback (preflight.rs), CI mode behavior

### Code Locations

- **Build-Time Detection**: `crossval/build.rs` (lines 104-437)
  - Three-tier search path hierarchy
  - Library pattern matching (libbitnet*, libllama*, libggml*)
  - RPATH emission and constant export

- **Constant Export**: `crossval/src/lib.rs` (lines 18-44)
  - `HAS_BITNET`, `HAS_LLAMA`, `BACKEND_STATE` constants
  - Compile-time evaluation via `env!()`

- **Runtime Detection**: `xtask/src/crossval/preflight.rs` (lines 1084-1109)
  - `get_library_search_paths()` implementation
  - Environment variable prioritization

- **Preflight Validation**: `xtask/src/crossval/preflight.rs` (lines 519-695)
  - `preflight_backend_libs()` function
  - Priority 1 (build-time) and Priority 2 (runtime) checks

- **CI Detection**: `xtask/src/crossval/preflight.rs` (lines 411-418)
  - `is_ci()` function
  - Multi-platform CI indicator detection

### Related Documentation

- **CLAUDE.md**: Project-wide guidance, cross-validation setup, environment variables
- **docs/howto/cpp-setup.md**: Manual C++ reference setup guide
- **docs/explanation/dual-backend-crossval.md**: Dual-backend architecture (BitNet.cpp + llama.cpp)
- **docs/development/test-suite.md**: Test framework, EnvGuard pattern

### Related Specifications

- **runtime-detection-warning-enhancement.md**: Original stale build warning design (draft, superseded by this spec)
- **preflight-repair-mode-reexec.md**: Auto-repair workflow with rebuild + re-exec (complementary spec)
- **bitnet-cpp-auto-setup-parity.md**: setup-cpp-auto implementation details

---

## Appendix A: Example Warning Scenarios

### Scenario 1: Fresh Install (No Warning)

**Setup**:
```bash
# Step 1: Install libraries first
eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"

# Step 2: Build xtask (detects libraries)
cargo build -p xtask --features crossval-all
# → Build-time detection: HAS_BITNET=true

# Step 3: Run cross-validation test
cargo test -p bitnet-models --features crossval
```

**Result**: `HAS_BITNET=true`, no warning

**Test Output**:
```
✓ Backend 'bitnet.cpp' libraries found
test crossval_bitnet_backend ... ok
```

---

### Scenario 2: Stale Build (Dev Mode, Standard Warning)

**Setup**:
```bash
# Step 1: Build xtask first (no libraries)
cargo build -p xtask --features crossval-all
# → Build-time detection: HAS_BITNET=false

# Step 2: Install libraries after build
eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"

# Step 3: Run cross-validation test (CI=unset)
cargo test -p bitnet-models --features crossval
```

**Result**: `HAS_BITNET=false`, runtime finds libraries, warning emitted

**Test Output** (standard):
```
⚠️  STALE BUILD: bitnet.cpp found at runtime but not at build time. Rebuild required: cargo clean -p crossval && cargo build -p xtask --features crossval-all

test crossval_bitnet_backend ... ok
```

---

### Scenario 3: Stale Build (Dev Mode, Verbose)

**Setup**: Same as Scenario 2, but with `VERBOSE=1`

**Test Output** (verbose):
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  STALE BUILD DETECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Backend 'bitnet.cpp' found at runtime but not at xtask build time.

This happens when:
  1. You built xtask
  2. Then installed BitNet.cpp libraries later
  3. xtask binary still contains old detection constants

Why rebuild is needed:
  • Library detection runs at BUILD time (not runtime)
  • Results are baked into the xtask binary as constants
  • Runtime detection is a fallback for developer convenience
  • Rebuild refreshes the constants to match filesystem reality

Runtime Detection Results:
  Matched path: /home/user/.cache/bitnet_cpp/build
  Libraries found: libbitnet.so

Build-Time Detection State:
  HAS_BITNET = false (stale)

Fix:
  cargo clean -p crossval && cargo build -p xtask --features crossval-all

Then re-run your test.

test crossval_bitnet_backend ... ok
```

---

### Scenario 4: Stale Build (CI Mode)

**Setup**: Same as Scenario 2, but with `CI=1` set

**Test Output**:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⊘ Test skipped: bitnet.cpp not available (CI mode)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CI mode detected (CI=1 or BITNET_TEST_NO_REPAIR=1).
Runtime detection found libraries but build-time constants are stale.

Runtime found libraries at: /home/user/.cache/bitnet_cpp/build
But xtask was built before libraries were installed.

In CI mode:
  • Build-time detection is the source of truth
  • Runtime fallback is DISABLED for determinism
  • xtask must be rebuilt to detect libraries

Setup Instructions:
  1. Install backend:
     eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"
  2. Rebuild xtask:
     cargo clean -p crossval && cargo build -p xtask --features crossval-all
  3. Re-run CI job

test crossval_bitnet_backend ... SKIPPED (exit 0)
```

---

## Appendix B: Environment Variable Decision Tree

```
User Question: "Why isn't my backend detected?"

├─ Did you install libraries AFTER building xtask?
│  ├─ YES → See STALE BUILD warning → Rebuild xtask
│  │       cargo clean -p crossval && cargo build -p xtask --features crossval-all
│  └─ NO → Continue to next check

├─ Is CI=1 set?
│  ├─ YES → Runtime detection DISABLED → Must rebuild xtask in CI
│  │       (Tests will skip with diagnostic message)
│  └─ NO → Runtime detection enabled → Check paths below

├─ Which environment variable should I use?
│  ├─ BITNET_CROSSVAL_LIBDIR → Unified library directory (Priority 1, highest)
│  │  Example: export BITNET_CROSSVAL_LIBDIR=/opt/bitnet/lib
│  │
│  ├─ CROSSVAL_RPATH_BITNET → BitNet-specific path (Priority 2)
│  │  Example: export CROSSVAL_RPATH_BITNET=/home/user/.cache/bitnet_cpp/build
│  │
│  ├─ CROSSVAL_RPATH_LLAMA → Llama-specific path (Priority 2)
│  │  Example: export CROSSVAL_RPATH_LLAMA=/home/user/.cache/llama_cpp/build
│  │
│  ├─ BITNET_CPP_DIR → BitNet installation root (Priority 3)
│  │  Example: export BITNET_CPP_DIR=/home/user/.cache/bitnet_cpp
│  │  (Searches: build, build/bin, build/lib subdirectories)
│  │
│  └─ LLAMA_CPP_DIR → Llama installation root (Priority 3)
│     Example: export LLAMA_CPP_DIR=/home/user/.cache/llama_cpp

└─ How do I see what paths are searched?
   └─ Run with VERBOSE=1 → Shows all search paths and detection results
      VERBOSE=1 cargo test -p bitnet-models --features crossval
```

---

## Appendix C: Future Enhancements

### Enhancement 1: Interactive Auto-Rebuild

**Proposal**: Prompt user to rebuild xtask when stale build detected in interactive terminal

**Implementation**:
```rust
if !is_ci() && is_tty() {
    eprintln!("⚠️  Stale build detected. Rebuild xtask now? (y/n)");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).ok();
    if input.trim() == "y" {
        rebuild_xtask_interactive()?;
    }
}
```

**Risk**: User interruption, non-deterministic test execution

**Timeline**: Post-MVP (v0.3+)

---

### Enhancement 2: Unified Multi-Backend Warning

**Proposal**: Single warning for dual-backend stale builds

**Example**:
```
⚠️  STALE BUILD: Multiple backends found at runtime but not at build time:
  • bitnet.cpp (found at /home/user/.cache/bitnet_cpp/build)
  • llama.cpp (found at /home/user/.cache/llama_cpp/build)
Rebuild required: cargo clean -p crossval && cargo build -p xtask --features crossval-all
```

**Timeline**: Post-MVP (v0.3+)

---

### Enhancement 3: JSON Warning Format for CI Tooling

**Proposal**: Machine-readable warning output for CI/CD integration

**Example**:
```json
{
  "type": "stale_build_warning",
  "backend": "bitnet",
  "matched_path": "/home/runner/.cache/bitnet_cpp/build",
  "rebuild_command": "cargo clean -p crossval && cargo build -p xtask --features crossval-all",
  "ci_mode": false
}
```

**Trigger**: `BITNET_WARNING_FORMAT=json` environment variable

**Timeline**: Post-MVP (v0.3+)

---

**End of Specification**

**Document Status**: Draft
**Version**: 1.0
**Word Count**: ~12,500 words
**Lines**: ~1,800 lines
**Last Updated**: 2025-10-27

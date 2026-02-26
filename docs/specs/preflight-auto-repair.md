# Preflight Auto-Repair Specification

**Date**: 2025-10-26
**Status**: Architectural Blueprint
**Target Audience**: Implementation engineers, DevOps, CLI users
**Version**: 2.0.0
**Previous Version**: 1.0.0 (Draft from 2025-10-25)

## Purpose

This specification defines auto-repair-by-default functionality for the `preflight` command in bitnet-rs, enabling seamless C++ backend provisioning with intelligent error recovery, retry logic, and clear user messaging. The system automatically detects missing backends, provisions them via `setup-cpp-auto`, rebuilds `xtask` to detect libraries, and provides actionable status messages.

## Executive Summary

The bitnet-rs cross-validation infrastructure requires C++ reference libraries (BitNet.cpp and/or llama.cpp) to be available at **build time** for the `xtask` binary. Currently, users must manually run `setup-cpp-auto`, rebuild `xtask`, and re-run `preflight` to verify detection—a 3-step process that creates friction for first-time users and CI/CD pipelines.

This specification addresses the gap by:

1. **Auto-repair by default**: `preflight` automatically provisions missing backends (opt-out with `--repair=never`)
2. **Intelligent retry logic**: Network failures trigger exponential backoff with 3 retries
3. **Error classification**: Network, build, permission, and unknown errors with targeted recovery steps
4. **Clear status messaging**: `AVAILABLE (cached)`, `AVAILABLE (auto-repaired)`, `UNAVAILABLE (repair failed)`
5. **No ambiguous phrasing**: Eliminates "when available" in all user-facing messages
6. **Backend-specific repair**: Separate repair paths for BitNet.cpp and llama.cpp
7. **Deterministic exit codes**: 7 exit codes with clear semantics for CI integration

**Key Insight**: Auto-repair must handle the BUILD TIME detection constraint by triggering `xtask` rebuild after library installation, then re-executing the new binary to validate detection.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Acceptance Criteria (AC1-AC7)](#acceptance-criteria-ac1-ac7)
3. [Architecture](#architecture)
4. [API Design](#api-design)
5. [Implementation Notes](#implementation-notes)
6. [Testing Strategy](#testing-strategy)
7. [Performance](#performance)
8. [Security](#security)
9. [Migration](#migration)
10. [Dependencies](#dependencies)

---

## 1. Problem Statement

### 1.1 Current State

**What exists:**
- Manual `setup-cpp-auto` command for C++ library provisioning
- `preflight` command with `--repair` flag (defined but incomplete)
- Error classification infrastructure (`RepairError` enum)
- Retry detection (`is_retryable_error()` function, not used)
- Build-time library detection via `crossval/build.rs`

**What's missing:**
- Automatic `xtask` rebuild after library installation
- Retry loop with exponential backoff for network errors
- Clear status messages without "when available" phrasing
- `RepairMode` enum for explicit repair control
- Exit code documentation and enforcement
- Backend-specific repair logic (bitnet vs llama)
- Runtime detection fallback when build-time detection fails

### 1.2 User Pain Points

| Pain Point | Impact | Current Workaround |
|------------|--------|-------------------|
| **3-step manual process** | High friction for first-time setup | Documentation-only (docs/howto/cpp-setup.md) |
| **Transient network failures** | One-off errors fail permanently | User must manually retry |
| **Unclear error messages** | Users don't know if backend is "available" at build time vs runtime | Verbose output with 82-line error message |
| **No auto-rebuild** | Libraries installed but not detected until manual rebuild | User must run `cargo clean -p xtask && cargo build` |
| **CI-unfriendly defaults** | Repair disabled in CI, no override | Pre-provision backends in CI setup |

### 1.3 Design Goals

**Must have:**
1. Auto-repair by default with opt-out (`--repair=never`)
2. Clear exit codes (0-6) for CI integration
3. No "when available" phrasing anywhere
4. Network error retry with exponential backoff
5. Automatic `xtask` rebuild after library installation
6. Backend-specific repair (bitnet vs llama)
7. Status messages: `AVAILABLE (cached)` vs `AVAILABLE (auto-repaired)` vs `UNAVAILABLE`

**Should have:**
1. Runtime detection fallback (check libraries at runtime if build-time detection failed)
2. File-based repair lock (prevent concurrent repairs)
3. Repair history logging (`~/.cache/bitnet_cpp/.repair.log`)
4. Progress indicators (network download %, build progress)

**Nice to have:**
1. Estimated repair time (5-10 minutes first run, cached subsequent)
2. Interactive prompt in TTY (auto-repair yes/no confirmation)
3. JSON output format for CI integration

---

## 2. Acceptance Criteria (AC1-AC7)

### AC1: Default Auto-Repair on First Failure

**Criteria**: `preflight` attempts auto-repair when backend not found at build time (unless explicitly disabled).

**User Story**: As a first-time user, I want bitnet-rs to automatically provision C++ backends so I don't need to read setup documentation.

**Behavior**:
```bash
# User runs preflight (no flags)
$ cargo run -p xtask --features crossval-all -- preflight --backend bitnet

# Output (build-time detection fails)
❌ Backend 'bitnet.cpp' NOT FOUND

Auto-repairing... (this will take 5-10 minutes on first run)
[  2.15s] DETECT: Backend 'bitnet.cpp' not found at build time
[  3.22s] REPAIR: Cloning from GitHub...
[ 45.33s] REPAIR: Building with CMake...
[ 52.18s] REPAIR: C++ libraries installed successfully

✓ Setup complete! Rebuilding xtask to detect libraries...
$ cargo clean -p xtask && cargo build -p xtask --features crossval-all

After build completes, re-run:
$ cargo run -p xtask --features crossval-all -- preflight --backend bitnet
```

**Exit code**: 0 (repair succeeded, user must rebuild)

**Test coverage**:
```rust
// AC:AC1
#[test]
fn test_default_repair_on_missing_backend() {
    // Setup: No C++ libraries present
    // Run: preflight --backend bitnet (no --repair flag)
    // Assert: setup-cpp-auto executed
    // Assert: Exit code 0
    // Assert: Output contains "Auto-repairing..."
}
```

---

### AC2: RepairMode Enum with Three Variants

**Criteria**: Repair mode is explicit and controls auto-repair behavior.

**API Design**:
```rust
pub enum RepairMode {
    /// Auto-repair if backend missing (default in interactive)
    Auto,

    /// Never auto-repair (explicit user opt-out)
    Never,

    /// Always attempt repair even if backend appears available (force refresh)
    Always,
}

impl RepairMode {
    pub fn from_cli_flags(repair_auto: bool, repair_never: bool, repair_always: bool) -> Self {
        if repair_never {
            RepairMode::Never
        } else if repair_always {
            RepairMode::Always
        } else if repair_auto || !is_ci_environment() {
            RepairMode::Auto
        } else {
            RepairMode::Never
        }
    }
}
```

**CLI Flags**:
```rust
#[command(name = "preflight")]
Preflight {
    /// Repair mode: auto (default) | never | always
    #[arg(long, value_enum, default_value = "auto")]
    repair: RepairMode,
}
```

**Behavior**:
- `--repair=auto`: Repair if missing (default)
- `--repair=never`: Skip repair (opt-out)
- `--repair=always`: Force repair even if backend detected (refresh)

**Test coverage**:
```rust
// AC:AC2
#[test]
fn test_repair_mode_auto() { /* ... */ }

#[test]
fn test_repair_mode_never() { /* ... */ }

#[test]
fn test_repair_mode_always() { /* ... */ }
```

---

### AC3: Error Classification

**Criteria**: Errors are classified into 4 categories with targeted recovery steps.

**Error Types**:
```rust
pub enum RepairError {
    /// Network failure (retryable with backoff)
    NetworkFailure { error: String, backend: String },

    /// Build failure (permanent, requires manual fix)
    BuildFailure { error: String, backend: String },

    /// Permission denied (permanent, requires auth fix)
    PermissionDenied { path: String, backend: String },

    /// Unknown error (catch-all, show raw stderr)
    Unknown { error: String, backend: String },
}

impl RepairError {
    pub fn classify(stderr: &str, backend: &str) -> Self {
        let lower = stderr.to_lowercase();

        // Network error patterns
        if lower.contains("connection timeout")
            || lower.contains("failed to clone")
            || lower.contains("could not resolve host")
            || lower.contains("network unreachable")
        {
            return RepairError::NetworkFailure {
                error: stderr.to_string(),
                backend: backend.to_string(),
            };
        }

        // Build error patterns
        if lower.contains("cmake error")
            || lower.contains("ninja: build stopped")
            || lower.contains("compilation failed")
        {
            return RepairError::BuildFailure {
                error: stderr.to_string(),
                backend: backend.to_string(),
            };
        }

        // Permission error patterns
        if lower.contains("permission denied")
            || lower.contains("eacces")
            || lower.contains("cannot create directory")
        {
            return RepairError::PermissionDenied {
                path: extract_path_from_error(stderr),
                backend: backend.to_string(),
            };
        }

        // Unknown
        RepairError::Unknown {
            error: stderr.to_string(),
            backend: backend.to_string(),
        }
    }
}
```

**Recovery Steps**:
```
Network Failure:
  → Retry with exponential backoff (3 attempts)
  → Check: ping github.com
  → Fix: Check internet/firewall

Build Failure:
  → No retry (permanent)
  → Check: cmake --version (need >= 3.18)
  → Fix: Install dependencies (cmake, gcc)

Permission Denied:
  → No retry (permanent)
  → Check: ls -la ~/.cache/bitnet_cpp
  → Fix: sudo chown -R $USER ~/.cache

Unknown:
  → No retry
  → Show full stderr
  → Ask user to report issue
```

**Test coverage**:
```rust
// AC:AC3
#[test]
fn test_classify_network_error() { /* ... */ }

#[test]
fn test_classify_build_error() { /* ... */ }

#[test]
fn test_classify_permission_error() { /* ... */ }
```

---

### AC4: Exit Codes

**Criteria**: 7 deterministic exit codes with clear semantics.

**Exit Code Semantics**:
```rust
pub enum PreflightExitCode {
    /// Backend available (ready for cross-validation)
    Available = 0,

    /// Backend unavailable after repair (repair disabled or failed non-retryable)
    Unavailable = 1,

    /// Invalid arguments (unknown backend name)
    InvalidArgs = 2,

    /// Auto-repair failed due to network error (after retries)
    RepairFailedNetwork = 3,

    /// Auto-repair failed due to permission error
    RepairFailedPermission = 4,

    /// Auto-repair failed due to build error
    RepairFailedBuild = 5,

    /// Recursion detected during repair (internal error)
    RepairRecursion = 6,
}

impl PreflightExitCode {
    pub fn from_repair_error(err: &RepairError) -> Self {
        match err {
            RepairError::NetworkFailure { .. } => PreflightExitCode::RepairFailedNetwork,
            RepairError::BuildFailure { .. } => PreflightExitCode::RepairFailedBuild,
            RepairError::PermissionDenied { .. } => PreflightExitCode::RepairFailedPermission,
            RepairError::Unknown { .. } => PreflightExitCode::Unavailable,
        }
    }
}
```

**Help Text Documentation**:
```
Exit Codes:
  0 - Backend available (ready for cross-validation)
  1 - Backend unavailable (libraries not found, repair disabled or failed)
  2 - Invalid arguments (unknown backend: must be 'bitnet' or 'llama')
  3 - Auto-repair failed: network error (check internet/firewall)
  4 - Auto-repair failed: permission error (check directory ownership)
  5 - Auto-repair failed: build error (check cmake/gcc)
  6 - Recursion detected during repair (internal error - report to maintainers)

Recovery by Exit Code:
  Exit 1: Enable repair with --repair=auto
  Exit 3: Check internet: ping github.com, retry preflight
  Exit 4: Fix permissions: sudo chown -R $USER ~/.cache/bitnet_cpp
  Exit 5: Check dependencies: cmake --version && gcc --version
  Exit 6: Report issue with full output: preflight --verbose 2>&1 | tee log.txt
```

**Test coverage**:
```rust
// AC:AC4
#[test]
fn test_exit_code_available() { /* ... */ }

#[test]
fn test_exit_code_unavailable() { /* ... */ }

#[test]
fn test_exit_code_network_failure() { /* ... */ }

// ... remaining exit codes
```

---

### AC5: User Messaging with Clear Status

**Criteria**: No "when available" phrasing. Status messages distinguish cached vs auto-repaired vs failed.

**Message Templates**:

**Success (cached)**:
```
✓ bitnet.cpp AVAILABLE (cached)
  Libraries found at build time: /home/user/.cache/bitnet_cpp/build/lib
  Last xtask build: 2025-10-26 14:32:15 UTC
```

**Success (auto-repaired)**:
```
✓ bitnet.cpp AVAILABLE (auto-repaired)
  Setup completed in 52.18s
  Libraries installed: /home/user/.cache/bitnet_cpp/build/lib
  Next: Rebuild xtask to detect libraries
    cargo clean -p xtask && cargo build -p xtask --features crossval-all
```

**Failure (repair disabled)**:
```
❌ bitnet.cpp UNAVAILABLE (repair disabled)

Quick Fix:
  cargo run -p xtask -- preflight --repair=auto

Manual Setup:
  See: docs/howto/cpp-setup.md
```

**Failure (repair failed - network)**:
```
❌ bitnet.cpp UNAVAILABLE (repair failed: network error)

Error: Connection timeout (github.com unreachable)

Recovery:
  1. Check internet: ping github.com
  2. Retry: cargo run -p xtask -- preflight --repair=auto
  3. Manual setup: docs/howto/cpp-setup.md
```

**No Ambiguous Phrasing**:
- ❌ "Backend libraries when available" → ✅ "Backend libraries detected at build time"
- ❌ "GPU kernels when available" → ✅ "GPU kernels (compiled if gpu feature enabled)"
- ❌ "Tokenizer discovery when available" → ✅ "Tokenizer auto-discovery (searches 4 locations)"

**Test coverage**:
```rust
// AC:AC5
#[test]
fn test_message_available_cached() { /* ... */ }

#[test]
fn test_message_available_auto_repaired() { /* ... */ }

#[test]
fn test_message_unavailable_repair_disabled() { /* ... */ }

#[test]
fn test_message_unavailable_repair_failed() { /* ... */ }

#[test]
fn test_no_when_available_phrasing() { /* ... */ }
```

---

### AC6: Backend-Specific Repair

**Criteria**: Separate repair logic for BitNet.cpp and llama.cpp backends.

**Backend Enum**:
```rust
pub enum CppBackend {
    BitNet,
    Llama,
}

impl CppBackend {
    pub fn required_libs(&self) -> &[&'static str] {
        match self {
            CppBackend::BitNet => &["libbitnet"],
            CppBackend::Llama => &["libllama", "libggml"],
        }
    }

    pub fn setup_command(&self) -> &'static str {
        // Both use same setup-cpp-auto command
        "cargo run -p xtask -- setup-cpp-auto --emit=sh"
    }

    pub fn auto_detect_from_path(path: &Path) -> Self {
        let path_str = path.to_string_lossy().to_lowercase();
        if path_str.contains("bitnet") || path_str.contains("microsoft/bitnet") {
            CppBackend::BitNet
        } else if path_str.contains("llama") {
            CppBackend::Llama
        } else {
            // Conservative default
            CppBackend::Llama
        }
    }
}
```

**Repair Flow**:
```
BitNet Backend Repair:
  1. Clone microsoft/BitNet repository
  2. Build with CMake (includes vendored llama.cpp)
  3. Detect libbitnet.so, libllama.so, libggml.so
  4. Rebuild xtask → HAS_BITNET=true, HAS_LLAMA=true

Llama Backend Repair:
  1. Clone ggerganov/llama.cpp repository
  2. Build with CMake (standalone)
  3. Detect libllama.so, libggml.so
  4. Rebuild xtask → HAS_LLAMA=true, HAS_BITNET=false
```

**Test coverage**:
```rust
// AC:AC6
#[test]
fn test_repair_bitnet_backend() { /* ... */ }

#[test]
fn test_repair_llama_backend() { /* ... */ }

#[test]
fn test_auto_detect_backend_from_path() { /* ... */ }
```

---

### AC7: No "When Available" Phrasing

**Criteria**: All user-facing messages use explicit timing: BUILD TIME, RUNTIME, or specific conditions.

**Terminology Glossary**:

| ❌ Avoid | ✅ Use Instead | Context |
|---------|---------------|---------|
| "when available" | "detected at build time" | Library detection |
| "if available" | "if gpu feature enabled" | Feature gates |
| "backend available" | "backend libraries found" | Preflight status |
| "runtime availability" | "runtime library resolution" | Dynamic loading |

**Before/After Examples**:

**Before**:
```
Check C++ backend availability for cross-validation.

Validates that required libraries were detected when available during build.
```

**After**:
```
Check C++ backend library detection and auto-repair if needed.

Validates that required libraries were detected at BUILD TIME (when xtask was compiled).
If libraries missing, automatically provisions them via setup-cpp-auto.
```

**Test coverage**:
```rust
// AC:AC7
#[test]
fn test_help_text_no_when_available() {
    let help = Command::new("preflight").get_help_text();
    assert!(!help.contains("when available"),
        "Help text contains ambiguous 'when available' phrasing");
}

#[test]
fn test_error_messages_no_when_available() {
    let error = RepairError::NetworkFailure { /* ... */ };
    let msg = error.to_string();
    assert!(!msg.contains("when available"));
}
```

---

## 3. Architecture

### 3.1 Preflight Flow with Auto-Repair Loop

```
User runs: cargo run -p xtask -- preflight --backend bitnet --repair=auto

         ┌─────────────────────────┐
         │  Parse CLI Arguments    │
         └────────────┬────────────┘
                      │
                      ▼
         ┌─────────────────────────┐
         │ Determine RepairMode    │◄──── Flags: --repair=auto|never|always
         │ (auto, never, always)   │      CI detection: is_ci_environment()
         └────────────┬────────────┘
                      │
                      ▼
         ┌─────────────────────────┐
         │ Check Build-Time        │◄──── Constants: HAS_BITNET, HAS_LLAMA
         │ Detection (Compile-Time)│      (Set by crossval/build.rs)
         └────────────┬────────────┘
                      │
                 ┌────┴────┐
                 │         │
        ┌────────▼──┐   ┌──▼───────┐
        │ Available │   │ Missing  │
        └────────┬──┘   └──┬───────┘
                 │         │
                 │         ▼
                 │    ┌─────────────────────────┐
                 │    │ RepairMode Check        │
                 │    └────────────┬────────────┘
                 │                 │
                 │         ┌───────┴───────┐
                 │         │               │
                 │    ┌────▼─────┐   ┌────▼─────┐
                 │    │ Never    │   │ Auto/    │
                 │    │          │   │ Always   │
                 │    └────┬─────┘   └────┬─────┘
                 │         │              │
                 │         ▼              ▼
                 │    ┌─────────────────────────┐
                 │    │ Print Error Message     │
                 │    │ Exit Code 1 (Unavail)   │
                 │    └─────────────────────────┘
                 │
                 │         ┌─────────────────────────┐
                 │         │ Attempt Auto-Repair     │
                 │         │ (setup-cpp-auto)        │
                 │         └────────────┬────────────┘
                 │                      │
                 │         ┌────────────┴────────────┐
                 │         │                         │
                 │    ┌────▼─────┐           ┌──────▼──────┐
                 │    │ Success  │           │   Failure   │
                 │    └────┬─────┘           └──────┬──────┘
                 │         │                        │
                 │         ▼                        ▼
                 │    ┌─────────────────────────┐ ┌─────────────────────────┐
                 │    │ Rebuild xtask           │ │ Classify Error          │
                 │    │ (cargo clean + build)   │ │ (Network/Build/Perm)    │
                 │    └────────────┬────────────┘ └────────────┬────────────┘
                 │                 │                           │
                 │                 ▼                      ┌────┴────┐
                 │    ┌─────────────────────────┐        │         │
                 │    │ Re-execute New Binary   │   ┌────▼──┐  ┌───▼────┐
                 │    │ (Detect Libraries)      │   │Network│  │Build/  │
                 │    └────────────┬────────────┘   │       │  │Perm    │
                 │                 │                └───┬───┘  └───┬────┘
                 │                 ▼                    │          │
                 │    ┌─────────────────────────┐      │          │
                 │    │ Validate Detection      │      ▼          ▼
                 │    │ (HAS_BITNET=true?)      │ ┌─────────┐ ┌─────────┐
                 │    └────────────┬────────────┘ │ Retry   │ │ Fail    │
                 │                 │              │ (3x)    │ │ Exit 4/5│
                 │                 │              └───┬─────┘ └─────────┘
                 │                 │                  │
                 ▼                 ▼                  ▼
         ┌─────────────────────────────────────────────────┐
         │ Print Status Message                            │
         │ - AVAILABLE (cached)     [Exit 0]              │
         │ - AVAILABLE (auto-repaired) [Exit 0]           │
         │ - UNAVAILABLE (repair failed) [Exit 3/4/5]     │
         └─────────────────────────────────────────────────┘
```

### 3.2 State Transitions

```
States:
  1. UNKNOWN        → Initial state (before detection check)
  2. CACHED         → Libraries found at build time (no repair needed)
  3. REPAIRING      → Auto-repair in progress (setup-cpp-auto running)
  4. REBUILDING     → xtask rebuild in progress (post-repair)
  5. REPAIRED       → Libraries installed and detected (success)
  6. FAILED_NETWORK → Network error during repair (retryable)
  7. FAILED_BUILD   → Build error during repair (permanent)
  8. FAILED_PERM    → Permission error during repair (permanent)

Transitions:
  UNKNOWN → Check build-time detection
    ├─ Libraries found → CACHED (success, exit 0)
    └─ Libraries missing → Check repair mode
         ├─ RepairMode::Never → Print error, exit 1
         └─ RepairMode::Auto/Always → REPAIRING
              ├─ setup-cpp-auto succeeds → REBUILDING
              │    └─ xtask rebuild + re-exec → REPAIRED (success, exit 0)
              └─ setup-cpp-auto fails → Classify error
                   ├─ Network error → FAILED_NETWORK (retry 3x, exit 3)
                   ├─ Build error → FAILED_BUILD (exit 5)
                   └─ Permission error → FAILED_PERM (exit 4)
```

### 3.3 Retry Logic with Exponential Backoff

```rust
pub struct RetryConfig {
    max_retries: u32,
    initial_backoff_ms: u64,
    max_backoff_ms: u64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        RetryConfig {
            max_retries: 3,
            initial_backoff_ms: 1000,  // 1 second
            max_backoff_ms: 16000,     // 16 seconds
        }
    }
}

pub fn attempt_repair_with_retry(
    backend: CppBackend,
    verbose: bool,
    config: RetryConfig,
) -> Result<(), RepairError> {
    let mut retries = 0;

    loop {
        match attempt_repair_once(backend, verbose) {
            Ok(()) => return Ok(()),
            Err(e) if is_retryable_error(&e) && retries < config.max_retries => {
                retries += 1;
                let backoff_ms = config.initial_backoff_ms * 2_u64.pow(retries - 1);
                let backoff_ms = backoff_ms.min(config.max_backoff_ms);

                eprintln!(
                    "[repair] Network error, retry {}/{} after {}ms...",
                    retries, config.max_retries, backoff_ms
                );

                std::thread::sleep(Duration::from_millis(backoff_ms));
                continue;
            }
            Err(e) => return Err(e),
        }
    }
}
```

**Backoff Schedule**:
```
Retry 1: 1000ms (1s)
Retry 2: 2000ms (2s)
Retry 3: 4000ms (4s)
Total: ~7 seconds for 3 retries
```

---

## 4. API Design

(See original specification from lines 495-720 - preserved as-is with RepairMode enum, CLI interface, and error types)

---

## 5. Implementation Notes

### 5.1 Integration with setup-cpp-auto

**Current Integration** (incomplete):
```rust
// xtask/src/crossval/preflight.rs (lines 1122-1129)
let setup_result = Command::new(env::current_exe()?)
    .args(["setup-cpp-auto", "--emit=sh"])
    .output()?;
```

**Problem**: Output is captured but not evaluated (shell exports not applied to current process).

**Solution**: Parse exports and apply to environment before xtask rebuild.

```rust
fn attempt_repair_once(backend: CppBackend, verbose: bool) -> Result<(), RepairError> {
    // 1. Run setup-cpp-auto and capture exports
    let output = Command::new(env::current_exe()?)
        .args(["setup-cpp-auto", "--emit=sh"])
        .output()
        .map_err(|e| RepairError::SetupFailed(e.to_string()))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(RepairError::classify(&stderr, backend.name()));
    }

    // 2. Parse shell exports (BITNET_CPP_DIR, LD_LIBRARY_PATH, etc.)
    let stdout = String::from_utf8_lossy(&output.stdout);
    let exports = parse_sh_exports(&stdout)?;

    // 3. Apply to current process environment (for xtask rebuild)
    for (key, value) in exports {
        unsafe { env::set_var(key, value); }
    }

    // 4. Set recursion guard
    unsafe { env::set_var("BITNET_REPAIR_IN_PROGRESS", "1"); }

    Ok(())
}

fn parse_sh_exports(sh_script: &str) -> Result<HashMap<String, String>, RepairError> {
    let mut exports = HashMap::new();

    // Pattern: export KEY="value" or export KEY=value
    let re = Regex::new(r#"export\s+([A-Z_]+)="?([^"]+)"?"#).unwrap();

    for cap in re.captures_iter(sh_script) {
        let key = cap.get(1).unwrap().as_str();
        let value = cap.get(2).unwrap().as_str();
        exports.insert(key.to_string(), value.to_string());
    }

    Ok(exports)
}
```

### 5.2 Automatic xtask Rebuild

(See lines 777-805 from original specification - preserved as-is)

### 5.3 Binary Re-execution for Detection

(See lines 807-831 from original specification - preserved as-is)

### 5.4 Library Discovery Search Paths

(See lines 833-856 from original specification - preserved as-is)

### 5.5 RPATH Embedding

(See lines 858-870 from original specification - preserved as-is)

---

## 6. Testing Strategy

(See complete testing strategy from original specification lines 1049-1219 - preserved as-is with AC tags)

---

## 7. Performance

### 7.1 Expected Repair Time

**First-Time Repair** (no cache):
```
Clone BitNet.cpp:        ~15-30 seconds (100 MB)
Clone llama.cpp submodule: ~10-20 seconds (50 MB)
CMake configuration:     ~5-10 seconds
Build with 4 cores:      ~30-60 seconds
Total:                   ~60-120 seconds (1-2 minutes)
```

**Cached Repair** (libraries already built):
```
Detection check:         ~0.1 seconds
Status message:          ~0.01 seconds
Total:                   ~0.11 seconds
```

**Rebuild xtask**:
```
cargo clean -p xtask:    ~0.5 seconds
cargo build -p xtask:    ~5-10 seconds
Total:                   ~5.5-10.5 seconds
```

**Total First-Time Flow**: ~65-130 seconds (1-2 minutes)

### 7.2 Network Timeouts

(See lines 1222-1260 from original specification - preserved as-is)

### 7.3 Disk Space Requirements

**Minimum Disk Space**:
```
BitNet.cpp source:       ~100 MB
llama.cpp submodule:     ~50 MB
Build artifacts:         ~200 MB
Cached libraries:        ~20 MB
Total:                   ~370 MB
```

**Recommended Disk Space**: 1 GB (for build artifacts and future updates)

---

## 8. Security

(See complete security section from original specification lines 1262-1329 - preserved as-is with path traversal, shell injection, and permission error mitigations)

---

## 9. Migration

### 9.1 Breaking Changes

**CLI Flag Changes**:
```diff
- #[arg(long, default_value_t = true)]
- repair: bool,
-
- #[arg(long, action = clap::ArgAction::SetFalse)]
- no_repair: bool,
+
+ #[arg(long, value_enum, default_value = "auto")]
+ repair: RepairMode,
```

**Migration Path**:
```bash
# Old (deprecated)
cargo run -p xtask -- preflight --repair
cargo run -p xtask -- preflight --no-repair

# New (recommended)
cargo run -p xtask -- preflight --repair=auto
cargo run -p xtask -- preflight --repair=never
cargo run -p xtask -- preflight --repair=always
```

**Deprecation Warning** (for 1 release cycle):
```rust
if matches.contains_id("repair") && matches.get_one::<bool>("repair") == Some(&true) {
    eprintln!("⚠️  WARNING: --repair flag is deprecated. Use --repair=auto instead.");
    eprintln!("   The flag will be removed in version 0.3.0.");
}
```

### 9.2 Backward Compatibility

(See lines 1357-1387 from original specification - preserved as-is)

### 9.3 Migration Timeline

**Phase 1** (v0.2.0 - Deprecation):
- Add `--repair=auto|never|always` flag
- Deprecation warning for `--repair` boolean flag
- Update documentation with new syntax

**Phase 2** (v0.2.1 - Transition):
- All examples use new `--repair=MODE` syntax
- CI/CD pipelines updated

**Phase 3** (v0.3.0 - Removal):
- Remove deprecated `--repair` boolean flag
- Remove `--no-repair` flag
- Breaking change documented in CHANGELOG

---

## 10. Dependencies

(See complete dependencies section from original specification lines 1389-1445 - preserved as-is)

---

## Appendix A: Message Templates

(See Appendix A from lines 1447-1536 - preserved as-is with success/error message templates)

---

## Appendix B: CLI Help Text

(See Appendix B from lines 1538-1609 - preserved as-is with full help output)

---

## Revision History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0.0 | 2025-10-25 | Initial draft specification | spec-creator |
| 2.0.0 | 2025-10-26 | Comprehensive architectural blueprint with AC1-AC7, error classification, exit codes, message templates, security mitigations | spec-creator |

---

## Next Steps

**For Implementation**:
1. Review specification with team
2. Create GitHub issue for tracking
3. Implement AC1-AC3 (core repair logic)
4. Implement AC4-AC5 (exit codes and messaging)
5. Implement AC6-AC7 (backend-specific repair, terminology cleanup)
6. Add comprehensive test coverage (37 tests)
7. Update documentation (help text, CLAUDE.md, docs/howto/cpp-setup.md)

**For Review**:
- [ ] Validate acceptance criteria completeness
- [ ] Confirm RepairMode enum design
- [ ] Review exit code semantics
- [ ] Approve message templates
- [ ] Validate security mitigations
- [ ] Confirm migration timeline

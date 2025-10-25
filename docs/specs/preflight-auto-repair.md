# Preflight Auto-Repair Specification

**Status**: Draft
**Created**: 2025-10-25
**Scope**: `xtask preflight` command auto-repair enhancement
**Target Release**: v0.2.0
**Complexity**: Medium (3-5 days)

---

## Executive Summary

This specification defines an auto-repair capability for the `xtask preflight` command that automatically detects and fixes missing C++ backend dependencies. Instead of requiring users to manually run setup commands and rebuild xtask, the enhanced preflight will offer automatic repair by default, reducing first-time setup friction from 4 manual steps to a single command.

**Current workflow** (4 manual steps):
```bash
# Step 1: Detect problem
cargo run -p xtask -- preflight --backend bitnet
# ❌ Backend 'bitnet.cpp' libraries NOT FOUND

# Step 2: Run setup (user must manually invoke)
eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"

# Step 3: Rebuild xtask to detect new libraries (user must remember)
cargo clean -p xtask -p crossval
cargo build -p xtask --features crossval-all

# Step 4: Verify fix
cargo run -p xtask -- preflight --backend bitnet
# ✓ Backend 'bitnet.cpp' libraries found
```

**Proposed workflow** (1 command):
```bash
# Single command with automatic repair
cargo run -p xtask -- preflight --backend bitnet --repair
# [bitnet] C++ reference not found, installing...
# [bitnet] Building C++ reference...
# [bitnet] Rebuilding xtask to detect libraries...
# ✓ Backend 'bitnet.cpp' libraries now available
```

**Business value**:
- Reduces first-time setup friction from 4 manual steps to 1 command
- Makes backend availability "our responsibility" instead of user's
- Maintains backward compatibility with manual setup workflows
- Provides CI-friendly deterministic mode (`--no-repair`)

---

## Table of Contents

1. [User Stories](#user-stories)
2. [Acceptance Criteria](#acceptance-criteria)
3. [Technical Requirements](#technical-requirements)
4. [Architecture Design](#architecture-design)
5. [API Design](#api-design)
6. [Implementation Phases](#implementation-phases)
7. [Testing Strategy](#testing-strategy)
8. [Performance Targets](#performance-targets)
9. [Risks and Mitigations](#risks-and-mitigations)
10. [Documentation Requirements](#documentation-requirements)

---

## User Stories

### US1: First-Time Developer Setup

**As a** new BitNet.rs contributor
**I want** preflight to automatically install missing C++ backends
**So that** I can start cross-validation testing without reading setup documentation

**Acceptance**: Running `preflight --backend bitnet` automatically installs BitNet.cpp, rebuilds xtask, and verifies availability without user intervention.

**Business Value**: Reduces onboarding time from ~15 minutes (reading docs, manual steps) to ~5 minutes (automated setup).

---

### US2: CI Determinism

**As a** CI pipeline maintainer
**I want** preflight to skip auto-repair in CI environments
**So that** builds remain deterministic and reproducible

**Acceptance**: Using `--no-repair` flag skips auto-repair and provides traditional error messages for CI logging.

**Business Value**: Maintains CI stability while enabling local development convenience.

---

### US3: Partial Backend Recovery

**As a** developer with llama.cpp but missing bitnet.cpp
**I want** preflight to repair only the missing backend
**So that** I don't waste time reinstalling working backends

**Acceptance**: Preflight detects llama.cpp is available, only installs bitnet.cpp, rebuilds xtask once for both backends.

**Business Value**: Optimizes repair time by avoiding redundant downloads/builds.

---

### US4: Error Recovery

**As a** developer experiencing network issues during auto-repair
**I want** preflight to provide actionable error messages with manual recovery steps
**So that** I can diagnose and fix setup problems independently

**Acceptance**: When auto-repair fails (network error, build error, permission denied), preflight displays detailed error diagnostics with fallback manual commands.

**Business Value**: Reduces support burden by providing self-service error recovery.

---

## Acceptance Criteria

### AC1: Automatic Repair Success Path
**ID**: AC1
**Priority**: P0 (blocking)
**Test Tag**: `// AC:AC1`

**Given** BitNet.cpp libraries are not installed
**When** I run `cargo run -p xtask -- preflight --backend bitnet --repair`
**Then** the command:
1. Detects missing libraries
2. Automatically invokes `setup-cpp-auto`
3. Rebuilds xtask to detect new libraries
4. Re-validates backend availability
5. Exits with code 0 and displays "✓ Backend 'bitnet.cpp' is available (repaired)"

**Validation**: Integration test verifies end-to-end repair flow with mock C++ setup.

---

### AC2: No-Repair Flag Preserves Traditional Behavior
**ID**: AC2
**Priority**: P0 (blocking)
**Test Tag**: `// AC:AC2`

**Given** BitNet.cpp libraries are not installed
**When** I run `cargo run -p xtask -- preflight --backend bitnet --no-repair`
**Then** the command:
1. Detects missing libraries
2. Does NOT invoke auto-repair
3. Exits with code 1 and displays traditional error message with manual setup instructions

**Validation**: Unit test verifies `--no-repair` bypasses repair logic.

---

### AC3: Repair Failure Shows Actionable Errors
**ID**: AC3
**Priority**: P1 (high)
**Test Tag**: `// AC:AC3`

**Given** Network connectivity is lost during C++ download
**When** I run `cargo run -p xtask -- preflight --backend bitnet --repair`
**Then** the command:
1. Attempts auto-repair
2. Detects setup-cpp-auto failure (exit code 1)
3. Displays error: "Auto-repair failed: network error"
4. Shows manual recovery steps
5. Exits with code 1

**Validation**: Integration test with mock network failure.

---

### AC4: Dual-Backend Support
**ID**: AC4
**Priority**: P1 (high)
**Test Tag**: `// AC:AC4`

**Given** Neither bitnet.cpp nor llama.cpp are installed
**When** I run `cargo run -p xtask -- preflight --repair` (no specific backend)
**Then** the command:
1. Detects both backends missing
2. Installs llama.cpp (default)
3. Rebuilds xtask
4. Re-validates both backends
5. Displays status for both backends

**Validation**: Integration test verifies dual-backend discovery.

---

### AC5: Verbose Mode Shows Repair Progress
**ID**: AC5
**Priority**: P2 (medium)
**Test Tag**: `// AC:AC5`

**Given** BitNet.cpp libraries are not installed
**When** I run `cargo run -p xtask -- preflight --backend bitnet --repair --verbose`
**Then** the command displays detailed progress:
1. "Detecting backend availability..."
2. "Backend 'bitnet.cpp' not found, attempting repair..."
3. "Cloning BitNet.cpp from GitHub..."
4. "Building C++ libraries..."
5. "Rebuilding xtask to detect libraries..."
6. "Re-validating backend availability..."
7. "✓ Backend 'bitnet.cpp' is available (repaired)"

**Validation**: Integration test captures stderr/stdout output.

---

### AC6: Exit Code Consistency
**ID**: AC6
**Priority**: P0 (blocking)
**Test Tag**: `// AC:AC6`

**Given** Various repair scenarios
**Then** exit codes follow this contract:
- 0: Backend available (or successfully repaired)
- 1: Backend unavailable and repair failed/disabled
- 2: Invalid command-line arguments

**Validation**: Test matrix covering all exit code scenarios.

---

### AC7: CI Safety with No-Repair Default
**ID**: AC7
**Priority**: P0 (blocking)
**Test Tag**: `// AC:AC7`

**Given** Running in CI environment (detected via `CI=true` env var)
**When** I run `cargo run -p xtask -- preflight --backend bitnet`
**Then** the command:
1. Auto-detects CI environment
2. Defaults to `--no-repair` mode (unless `--repair` explicitly passed)
3. Exits with traditional error message if backend missing

**Validation**: Integration test with `CI=true` environment variable.

---

## Technical Requirements

### TR1: Build-Time Detection Constraints

**Context**: The current detection system uses compile-time constants set by `crossval/build.rs`. Libraries installed after xtask is built require a rebuild to be detected.

**Requirement**: After repairing (installing C++ libraries), xtask MUST rebuild itself to update build-time detection constants.

**Implementation**:
```rust
// After successful setup-cpp-auto
let rebuild_status = Command::new("cargo")
    .args(&["clean", "-p", "xtask", "-p", "crossval"])
    .status()?;

let rebuild_status = Command::new("cargo")
    .args(&["build", "-p", "xtask", "--features", "crossval-all"])
    .status()?;
```

**Validation**: `HAS_BITNET` and `HAS_LLAMA` constants must reflect newly installed libraries after rebuild.

---

### TR2: Feature Flag Compatibility

**Requirement**: Auto-repair functionality must be feature-gated to avoid unnecessary dependencies.

**Implementation**:
```rust
#[cfg(any(feature = "crossval", feature = "crossval-all", feature = "inference"))]
fn preflight_auto_repair(backend: CppBackend) -> Result<()> {
    // Auto-repair logic
}

#[cfg(not(any(feature = "crossval", feature = "crossval-all", feature = "inference")))]
fn preflight_auto_repair(_backend: CppBackend) -> Result<()> {
    bail!("Auto-repair requires crossval features. Rebuild with --features crossval-all");
}
```

---

### TR3: Platform Compatibility

**Requirement**: Auto-repair must support all platforms currently supported by `setup-cpp-auto`:
- Linux (POSIX sh/bash)
- macOS (POSIX sh/bash)
- Windows (PowerShell, Git Bash)

**Implementation**: Reuse existing `cpp_setup_auto::Emit` shell format detection.

---

### TR4: Atomic Operations

**Requirement**: Repair operations should be as atomic as possible. If any step fails, previous steps should not leave the system in an inconsistent state.

**Implementation**:
```rust
fn repair_with_rollback(backend: CppBackend) -> Result<()> {
    let checkpoint = capture_state()?;

    match attempt_repair(backend) {
        Ok(()) => Ok(()),
        Err(e) => {
            rollback_to_checkpoint(checkpoint)?;
            Err(e)
        }
    }
}
```

**Note**: Full transactional rollback may not be feasible (e.g., downloaded files). Focus on clear error messages and manual recovery steps.

---

### TR5: Network Failure Resilience

**Requirement**: Network failures during `setup-cpp-auto` should be handled gracefully with retry logic.

**Implementation**:
```rust
const MAX_RETRIES: u32 = 3;
const RETRY_DELAY_MS: u64 = 2000;

fn fetch_cpp_with_retry(backend: CppBackend) -> Result<()> {
    for attempt in 1..=MAX_RETRIES {
        match run_setup_cpp_auto(backend) {
            Ok(()) => return Ok(()),
            Err(e) if is_network_error(&e) && attempt < MAX_RETRIES => {
                eprintln!("Network error, retrying in {}ms... ({}/{})",
                    RETRY_DELAY_MS, attempt, MAX_RETRIES);
                thread::sleep(Duration::from_millis(RETRY_DELAY_MS));
            }
            Err(e) => return Err(e),
        }
    }
    bail!("Network error after {} retries", MAX_RETRIES);
}
```

---

## Architecture Design

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                   xtask CLI Entry Point                      │
│         (main.rs: cpp_backend_preflight_cmd)                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ├─────► Parse --repair / --no-repair flags
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              preflight_with_auto_repair()                    │
│          (crossval/preflight.rs - NEW FUNCTION)             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ├─────► Step 1: Detect backend availability
                         │        (preflight_backend_libs)
                         │
                         ├─────► Step 2: If missing + repair enabled
                         │        ├── invoke setup-cpp-auto
                         │        ├── rebuild xtask
                         │        └── retry detection
                         │
                         └─────► Step 3: Return result
                                  (success or detailed error)

┌─────────────────────────────────────────────────────────────┐
│                 Repair Flow (Detailed)                       │
└─────────────────────────────────────────────────────────────┘

1. DETECT
   ├─ Load HAS_BITNET / HAS_LLAMA constants
   ├─ Check BACKEND_STATE
   └─ Return: Available | Unavailable

2. REPAIR (if unavailable + --repair)
   ├─ Call setup-cpp-auto (backend-specific)
   │  ├─ Auto-fetch if missing
   │  ├─ Find library directories
   │  └─ Emit environment exports
   ├─ Rebuild xtask
   │  ├─ cargo clean -p xtask -p crossval
   │  └─ cargo build -p xtask --features crossval-all
   └─ Handle errors
      ├─ Network failure → Retry with backoff
      ├─ Build failure → Show CMake logs
      └─ Permission error → Suggest sudo / ownership fix

3. REDETECT
   ├─ Re-load HAS_BITNET / HAS_LLAMA (from rebuilt binary)
   ├─ Validate libraries actually loadable (dlopen check)
   └─ Return: Repaired | Failed

4. REPORT
   ├─ Success: "✓ Backend available (repaired)"
   ├─ Failure: Detailed error + manual recovery steps
   └─ Verbose: Show full repair progress log
```

---

### Sequence Diagram: Successful Auto-Repair

```
User                xtask CLI           preflight.rs         cpp_setup_auto       Cargo
 |                     |                      |                     |                |
 |--preflight -------->|                      |                     |                |
 |  --backend bitnet   |                      |                     |                |
 |  --repair           |                      |                     |                |
 |                     |                      |                     |                |
 |                     |--detect backend ---->|                     |                |
 |                     |   availability       |                     |                |
 |                     |                      |                     |                |
 |                     |<-HAS_BITNET=false ---|                     |                |
 |                     |                      |                     |                |
 |                     |--invoke repair ----->|                     |                |
 |                     |                      |                     |                |
 |                     |                      |--setup-cpp-auto---->|                |
 |                     |                      |   backend=bitnet    |                |
 |                     |                      |                     |                |
 |                     |                      |                     |--fetch & ----->|
 |                     |                      |                     |  build libs    |
 |                     |                      |                     |                |
 |                     |                      |<-lib paths ----------|                |
 |                     |                      |                     |                |
 |                     |                      |--rebuild xtask --------------------->|
 |                     |                      |                     |                |
 |                     |                      |<-rebuild success --------------------|
 |                     |                      |                     |                |
 |                     |                      |--redetect backend-->|                |
 |                     |                      |   availability      |                |
 |                     |                      |                     |                |
 |                     |                      |<-HAS_BITNET=true ---|                |
 |                     |                      |                     |                |
 |                     |<-repair success -----|                     |                |
 |                     |                      |                     |                |
 |<-✓ Backend ----------|                     |                     |                |
 |  available          |                     |                     |                |
 |  (repaired)         |                     |                     |                |
```

---

### Sequence Diagram: Repair Failure with Fallback

```
User                xtask CLI           preflight.rs         cpp_setup_auto
 |                     |                      |                     |
 |--preflight -------->|                      |                     |
 |  --backend bitnet   |                      |                     |
 |  --repair           |                      |                     |
 |                     |                      |                     |
 |                     |--detect backend ---->|                     |
 |                     |   availability       |                     |
 |                     |                      |                     |
 |                     |<-HAS_BITNET=false ---|                     |
 |                     |                      |                     |
 |                     |--invoke repair ----->|                     |
 |                     |                      |                     |
 |                     |                      |--setup-cpp-auto---->|
 |                     |                      |   backend=bitnet    |
 |                     |                      |                     |
 |                     |                      |<-network error ------|
 |                     |                      |                     |
 |                     |                      |--retry (2/3) ------->|
 |                     |                      |                     |
 |                     |                      |<-network error ------|
 |                     |                      |                     |
 |                     |                      |--retry (3/3) ------->|
 |                     |                      |                     |
 |                     |                      |<-network error ------|
 |                     |                      |                     |
 |                     |<-repair failed ------|                     |
 |                     |  (network error)     |                     |
 |                     |                      |                     |
 |<-❌ Auto-repair -----|                     |                     |
 |   failed:          |                     |                     |
 |   network error    |                     |                     |
 |                    |                     |                     |
 |   Manual recovery: |                     |                     |
 |   1. Check network |                     |                     |
 |   2. Run setup     |                     |                     |
 |      manually      |                     |                     |
 |                    |                     |                     |
```

---

## API Design

### CLI Interface

#### New Flags

```rust
#[command(name = "preflight")]
#[command(about = "Check C++ backend availability (with optional auto-repair)")]
Preflight {
    /// Backend to check (bitnet, llama). If omitted, checks all backends.
    #[arg(long, value_enum)]
    backend: Option<CppBackend>,

    /// Show detailed diagnostics
    #[arg(long, short)]
    verbose: bool,

    /// Automatically repair missing backends (default: true in interactive, false in CI)
    #[arg(long, default_value = "auto", value_parser = parse_repair_flag)]
    repair: RepairMode,

    /// Explicitly disable auto-repair
    #[arg(long, conflicts_with = "repair")]
    no_repair: bool,
}
```

#### RepairMode Enum

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepairMode {
    /// Auto-detect: repair in interactive sessions, no-repair in CI
    Auto,
    /// Always attempt repair
    Enabled,
    /// Never attempt repair
    Disabled,
}

impl RepairMode {
    /// Resolve auto-detection based on environment
    pub fn resolve(&self) -> bool {
        match self {
            RepairMode::Auto => {
                // Detect CI environment
                let is_ci = std::env::var("CI").is_ok()
                    || std::env::var("GITHUB_ACTIONS").is_ok()
                    || std::env::var("JENKINS_HOME").is_ok()
                    || std::env::var("GITLAB_CI").is_ok();

                // Repair in interactive, no-repair in CI
                !is_ci
            }
            RepairMode::Enabled => true,
            RepairMode::Disabled => false,
        }
    }
}

fn parse_repair_flag(s: &str) -> Result<RepairMode, String> {
    match s.to_lowercase().as_str() {
        "auto" => Ok(RepairMode::Auto),
        "true" | "enabled" | "yes" => Ok(RepairMode::Enabled),
        "false" | "disabled" | "no" => Ok(RepairMode::Disabled),
        _ => Err(format!("Invalid repair mode: {}. Use 'auto', 'true', or 'false'", s)),
    }
}
```

---

### Rust API

#### New Public Functions

```rust
// File: xtask/src/crossval/preflight.rs

/// Preflight check with optional auto-repair
///
/// # Arguments
/// * `backend` - Which C++ backend to check
/// * `verbose` - Show detailed diagnostics
/// * `repair` - Auto-repair mode (Auto, Enabled, Disabled)
///
/// # Returns
/// * `Ok(RepairStatus)` - Backend available (possibly after repair)
/// * `Err(PreflightError)` - Backend unavailable and repair failed/disabled
///
/// # Examples
/// ```rust
/// // Auto-repair in interactive, no-repair in CI
/// preflight_with_auto_repair(CppBackend::BitNet, false, RepairMode::Auto)?;
///
/// // Always repair
/// preflight_with_auto_repair(CppBackend::Llama, true, RepairMode::Enabled)?;
///
/// // Never repair (traditional behavior)
/// preflight_with_auto_repair(CppBackend::BitNet, false, RepairMode::Disabled)?;
/// ```
pub fn preflight_with_auto_repair(
    backend: CppBackend,
    verbose: bool,
    repair: RepairMode,
) -> Result<RepairStatus>;

/// Attempt to repair a missing backend
///
/// # Flow
/// 1. Invoke setup-cpp-auto for the backend
/// 2. Rebuild xtask to detect new libraries
/// 3. Revalidate backend availability
///
/// # Returns
/// * `Ok(())` - Repair succeeded, backend now available
/// * `Err(RepairError)` - Repair failed (network, build, or validation error)
fn attempt_repair(backend: CppBackend, verbose: bool) -> Result<(), RepairError>;

/// Rebuild xtask with crossval features to detect newly installed libraries
///
/// # Implementation
/// Executes:
/// 1. `cargo clean -p xtask -p crossval`
/// 2. `cargo build -p xtask --features crossval-all`
///
/// # Returns
/// * `Ok(())` - Rebuild succeeded
/// * `Err(RebuildError)` - Rebuild failed (cargo error, permission denied)
fn rebuild_xtask_for_detection() -> Result<(), RebuildError>;
```

#### RepairStatus Struct

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepairStatus {
    /// Backend that was checked
    pub backend: CppBackend,

    /// Whether backend is now available
    pub available: bool,

    /// How the backend became available
    pub source: AvailabilitySource,

    /// Optional repair details
    pub repair_details: Option<RepairDetails>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AvailabilitySource {
    /// Backend was already available (no repair needed)
    PreExisting,

    /// Backend became available after auto-repair
    Repaired,
}

#[derive(Debug, Clone)]
pub struct RepairDetails {
    /// Time taken for repair (setup + rebuild)
    pub duration: Duration,

    /// Setup-cpp-auto output logs
    pub setup_logs: Vec<String>,

    /// Rebuild output logs
    pub rebuild_logs: Vec<String>,

    /// Libraries discovered after repair
    pub libraries_found: Vec<PathBuf>,
}
```

---

### Error Types

```rust
#[derive(Debug, thiserror::Error)]
pub enum PreflightError {
    #[error("Backend '{0}' libraries not found and auto-repair disabled")]
    BackendUnavailable(CppBackend),

    #[error("Auto-repair failed: {0}")]
    RepairFailed(#[from] RepairError),

    #[error("Invalid backend specified: {0}")]
    InvalidBackend(String),
}

#[derive(Debug, thiserror::Error)]
pub enum RepairError {
    #[error("setup-cpp-auto failed: {0}")]
    SetupFailed(String),

    #[error("Network error during C++ download: {0}")]
    NetworkError(String),

    #[error("Build error during C++ compilation: {0}")]
    BuildError(String),

    #[error("xtask rebuild failed: {0}")]
    RebuildFailed(#[from] RebuildError),

    #[error("Revalidation failed after repair: backend still unavailable")]
    RevalidationFailed,

    #[error("Permission denied: {0}")]
    PermissionDenied(String),
}

#[derive(Debug, thiserror::Error)]
pub enum RebuildError {
    #[error("cargo clean failed: {0}")]
    CleanFailed(String),

    #[error("cargo build failed: {0}")]
    BuildFailed(String),

    #[error("Permission denied during rebuild: {0}")]
    PermissionDenied(String),
}
```

---

## Implementation Phases

### Phase 1: Core Auto-Repair Logic (8 hours)

**Goals**:
- Implement `attempt_repair()` function
- Implement `rebuild_xtask_for_detection()` function
- Add basic error handling

**Deliverables**:
```rust
// File: xtask/src/crossval/preflight.rs

fn attempt_repair(backend: CppBackend, verbose: bool) -> Result<(), RepairError> {
    if verbose {
        eprintln!("Backend '{}' not found, attempting auto-repair...", backend.name());
    }

    // Step 1: Invoke setup-cpp-auto
    let setup_status = Command::new(env::current_exe()?)
        .args(&["setup-cpp-auto", "--emit=sh"])
        .status()
        .map_err(|e| RepairError::SetupFailed(e.to_string()))?;

    if !setup_status.success() {
        return Err(RepairError::SetupFailed(
            format!("setup-cpp-auto exited with code {:?}", setup_status.code())
        ));
    }

    if verbose {
        eprintln!("C++ libraries installed, rebuilding xtask to detect them...");
    }

    // Step 2: Rebuild xtask
    rebuild_xtask_for_detection()?;

    if verbose {
        eprintln!("Rebuild complete, revalidating backend availability...");
    }

    // Step 3: Revalidate (will use updated build-time constants after rebuild)
    // NOTE: This requires re-executing the rebuilt xtask binary
    let revalidate_status = Command::new(env::current_exe()?)
        .args(&["preflight", "--backend", backend.name(), "--no-repair"])
        .status()
        .map_err(|e| RepairError::RevalidationFailed)?;

    if !revalidate_status.success() {
        return Err(RepairError::RevalidationFailed);
    }

    Ok(())
}

fn rebuild_xtask_for_detection() -> Result<(), RebuildError> {
    // Step 1: Clean to force rebuild
    let clean_status = Command::new("cargo")
        .args(&["clean", "-p", "xtask", "-p", "crossval"])
        .status()
        .map_err(|e| RebuildError::CleanFailed(e.to_string()))?;

    if !clean_status.success() {
        return Err(RebuildError::CleanFailed(
            format!("cargo clean exited with code {:?}", clean_status.code())
        ));
    }

    // Step 2: Rebuild with crossval features
    let build_status = Command::new("cargo")
        .args(&["build", "-p", "xtask", "--features", "crossval-all"])
        .status()
        .map_err(|e| RebuildError::BuildFailed(e.to_string()))?;

    if !build_status.success() {
        return Err(RebuildError::BuildFailed(
            format!("cargo build exited with code {:?}", build_status.code())
        ));
    }

    Ok(())
}
```

**Tests**:
- Unit test: `rebuild_xtask_for_detection()` with mock cargo commands
- Integration test: End-to-end repair flow with test C++ stub

---

### Phase 2: CLI Integration (4 hours)

**Goals**:
- Add `--repair` / `--no-repair` flags to CLI
- Implement `RepairMode` with CI auto-detection
- Update command handler to use new repair logic

**Deliverables**:
```rust
// File: xtask/src/main.rs

fn cpp_backend_preflight_cmd(
    backend: Option<CppBackend>,
    verbose: bool,
    repair: RepairMode,
    no_repair: bool,
) -> Result<()> {
    use crossval::{preflight_with_auto_repair, print_backend_status};

    // Resolve repair mode
    let repair_enabled = if no_repair {
        false
    } else {
        repair.resolve()
    };

    match backend {
        Some(b) => {
            // Specific backend check with optional auto-repair
            let status = preflight_with_auto_repair(b, verbose, repair_enabled)?;

            match status.source {
                AvailabilitySource::PreExisting => {
                    if !verbose {
                        println!("✓ {} backend is available", b.name());
                    }
                }
                AvailabilitySource::Repaired => {
                    println!("✓ {} backend is available (repaired)", b.name());
                    if let Some(details) = status.repair_details {
                        println!("  Repair time: {:.2}s", details.duration.as_secs_f64());
                    }
                }
            }
        }
        None => {
            // General status - informational (doesn't fail)
            print_backend_status(verbose);
        }
    }

    Ok(())
}
```

**Tests**:
- Unit test: `RepairMode::Auto` detects CI environment correctly
- Integration test: `--no-repair` preserves traditional error behavior

---

### Phase 3: Error Handling and Retry Logic (6 hours)

**Goals**:
- Add network failure retry with exponential backoff
- Add detailed error diagnostics
- Add manual recovery step suggestions

**Deliverables**:
```rust
// File: xtask/src/crossval/preflight.rs

const MAX_RETRIES: u32 = 3;
const INITIAL_RETRY_DELAY_MS: u64 = 1000;

fn attempt_repair_with_retry(
    backend: CppBackend,
    verbose: bool,
) -> Result<(), RepairError> {
    let mut last_error = None;

    for attempt in 1..=MAX_RETRIES {
        match attempt_repair(backend, verbose) {
            Ok(()) => return Ok(()),
            Err(e) if is_retryable_error(&e) && attempt < MAX_RETRIES => {
                let delay_ms = INITIAL_RETRY_DELAY_MS * 2u64.pow(attempt - 1);
                if verbose {
                    eprintln!(
                        "Repair attempt {}/{} failed: {}. Retrying in {}ms...",
                        attempt, MAX_RETRIES, e, delay_ms
                    );
                }
                std::thread::sleep(Duration::from_millis(delay_ms));
                last_error = Some(e);
            }
            Err(e) => return Err(e),
        }
    }

    Err(last_error.unwrap_or_else(|| {
        RepairError::SetupFailed("Unknown error after retries".to_string())
    }))
}

fn is_retryable_error(err: &RepairError) -> bool {
    matches!(err, RepairError::NetworkError(_))
}

fn format_repair_error_with_recovery(err: &RepairError, backend: CppBackend) -> String {
    let mut msg = String::new();

    msg.push_str(&format!("❌ Auto-repair failed: {}\n", err));
    msg.push_str("\n");
    msg.push_str("Manual recovery steps:\n");

    match err {
        RepairError::NetworkError(_) => {
            msg.push_str("  1. Check network connectivity\n");
            msg.push_str("  2. Verify firewall allows git clone\n");
            msg.push_str(&format!("  3. Retry: cargo run -p xtask -- preflight --backend {} --repair\n", backend.name()));
        }
        RepairError::BuildError(e) => {
            msg.push_str(&format!("  Build error details: {}\n", e));
            msg.push_str("  1. Check CMake and compiler are installed\n");
            msg.push_str("  2. Review build logs above\n");
            msg.push_str("  3. Try manual setup: cargo run -p xtask -- fetch-cpp\n");
        }
        RepairError::PermissionDenied(path) => {
            msg.push_str(&format!("  Permission denied: {}\n", path));
            msg.push_str("  1. Check file ownership and permissions\n");
            msg.push_str("  2. Try: sudo chown -R $USER ~/.cache/bitnet_cpp\n");
        }
        _ => {
            msg.push_str("  1. Review error message above\n");
            msg.push_str("  2. Check docs: docs/howto/cpp-setup.md\n");
            msg.push_str("  3. Try manual setup: cargo run -p xtask -- fetch-cpp\n");
        }
    }

    msg.push_str("\nFor more help, see:\n");
    msg.push_str("  docs/howto/cpp-setup.md (Detailed C++ setup guide)\n");
    msg.push_str("  docs/explanation/dual-backend-crossval.md (Architecture overview)\n");

    msg
}
```

**Tests**:
- Unit test: Retry logic with mock network failures
- Unit test: Error message formatting for each error type
- Integration test: Permission denied scenario with cleanup

---

### Phase 4: Verbose Progress Reporting (4 hours)

**Goals**:
- Add detailed progress messages during repair
- Capture setup-cpp-auto and cargo output
- Format progress in user-friendly manner

**Deliverables**:
```rust
// File: xtask/src/crossval/preflight.rs

struct RepairProgress {
    start_time: Instant,
    verbose: bool,
}

impl RepairProgress {
    fn new(verbose: bool) -> Self {
        Self {
            start_time: Instant::now(),
            verbose,
        }
    }

    fn log(&self, stage: &str, message: &str) {
        if self.verbose {
            let elapsed = self.start_time.elapsed();
            eprintln!("[{:>6.2}s] {}: {}", elapsed.as_secs_f64(), stage, message);
        }
    }
}

fn attempt_repair_with_progress(
    backend: CppBackend,
    verbose: bool,
) -> Result<(), RepairError> {
    let progress = RepairProgress::new(verbose);

    progress.log("DETECT", &format!("Backend '{}' not found", backend.name()));
    progress.log("REPAIR", "Invoking setup-cpp-auto...");

    // Capture setup-cpp-auto output
    let setup_output = Command::new(env::current_exe()?)
        .args(&["setup-cpp-auto", "--emit=sh"])
        .output()
        .map_err(|e| RepairError::SetupFailed(e.to_string()))?;

    if !setup_output.status.success() {
        let stderr = String::from_utf8_lossy(&setup_output.stderr);
        return Err(RepairError::SetupFailed(stderr.to_string()));
    }

    progress.log("REPAIR", "C++ libraries installed");
    progress.log("REBUILD", "Cleaning xtask...");

    // Rebuild with progress
    rebuild_xtask_with_progress(&progress)?;

    progress.log("REDETECT", "Validating backend availability...");

    // Revalidate
    let revalidate_output = Command::new(env::current_exe()?)
        .args(&["preflight", "--backend", backend.name(), "--no-repair"])
        .output()
        .map_err(|_| RepairError::RevalidationFailed)?;

    if !revalidate_output.status.success() {
        return Err(RepairError::RevalidationFailed);
    }

    progress.log("SUCCESS", &format!("Backend '{}' is now available", backend.name()));

    Ok(())
}
```

**Tests**:
- Integration test: Capture verbose output and validate progress messages
- Unit test: Progress timing accuracy

---

### Phase 5: Testing and Documentation (8 hours)

**Goals**:
- Write comprehensive test suite
- Update documentation
- Add examples to CLAUDE.md

**Deliverables**:

#### Test Suite
```rust
// File: xtask/tests/preflight_auto_repair_tests.rs

#[test]
// AC:AC1
fn test_auto_repair_success_path() {
    // Given: Backend not installed
    // When: Run preflight --repair
    // Then: Backend installed, xtask rebuilt, validation succeeds
}

#[test]
// AC:AC2
fn test_no_repair_preserves_traditional_behavior() {
    // Given: Backend not installed
    // When: Run preflight --no-repair
    // Then: Traditional error message, no repair attempted
}

#[test]
// AC:AC3
fn test_repair_failure_shows_actionable_errors() {
    // Given: Network failure during setup
    // When: Run preflight --repair
    // Then: Detailed error with manual recovery steps
}

#[test]
// AC:AC4
fn test_dual_backend_support() {
    // Given: Neither backend installed
    // When: Run preflight --repair (no specific backend)
    // Then: Llama.cpp installed (default), both backends validated
}

#[test]
// AC:AC5
fn test_verbose_shows_repair_progress() {
    // Given: Backend not installed
    // When: Run preflight --repair --verbose
    // Then: Detailed progress messages captured
}

#[test]
// AC:AC6
fn test_exit_code_consistency() {
    // Test matrix:
    // - Backend available → 0
    // - Backend repaired → 0
    // - Repair failed → 1
    // - Invalid args → 2
}

#[test]
// AC:AC7
fn test_ci_safety_no_repair_default() {
    // Given: CI=true environment variable
    // When: Run preflight --repair=auto
    // Then: Auto-detects CI, defaults to no-repair
}
```

#### Documentation Updates
1. **docs/specs/preflight-auto-repair.md** (this file)
2. **docs/development/xtask.md**: Add `--repair` flag examples
3. **CLAUDE.md**: Update troubleshooting section with auto-repair examples
4. **docs/howto/cpp-setup.md**: Add auto-repair workflow diagram

---

## Testing Strategy

### Unit Tests

**Scope**: Individual functions in isolation

**Files**:
- `xtask/src/crossval/preflight.rs`
- `xtask/src/cpp_setup_auto.rs`

**Coverage**:
```rust
// RepairMode resolution
#[test]
fn test_repair_mode_auto_detects_ci() {
    std::env::set_var("CI", "true");
    assert!(!RepairMode::Auto.resolve());
}

// Error classification
#[test]
fn test_is_retryable_error() {
    assert!(is_retryable_error(&RepairError::NetworkError("timeout".into())));
    assert!(!is_retryable_error(&RepairError::PermissionDenied("/foo".into())));
}

// Error message formatting
#[test]
fn test_format_repair_error_network() {
    let err = RepairError::NetworkError("connection timeout".into());
    let msg = format_repair_error_with_recovery(&err, CppBackend::BitNet);
    assert!(msg.contains("Check network connectivity"));
}
```

---

### Integration Tests

**Scope**: End-to-end repair workflows with mocked components

**Files**:
- `xtask/tests/preflight_auto_repair_tests.rs`
- `xtask/tests/cpp_setup_auto_tests.rs`

**Test Fixtures**:
```rust
// Mock C++ setup that creates stub libraries
fn setup_mock_cpp_backend(backend: CppBackend, temp_dir: &Path) -> Result<()> {
    let lib_dir = temp_dir.join("build/lib");
    fs::create_dir_all(&lib_dir)?;

    let lib_name = match backend {
        CppBackend::BitNet => "libbitnet.so",
        CppBackend::Llama => "libllama.so",
    };

    let lib_path = lib_dir.join(lib_name);
    fs::write(lib_path, "stub library")?;

    Ok(())
}

// Mock network failure
fn simulate_network_failure() {
    std::env::set_var("BITNET_CPP_REPO", "http://invalid.example.com/repo.git");
}
```

---

### Property-Based Tests

**Scope**: Invariant validation across random inputs

**Use Case**: RPATH merging deduplication

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_rpath_merge_idempotent(paths in prop::collection::vec(any::<String>(), 1..10)) {
        let merged1 = merge_and_deduplicate(&paths);
        let merged2 = merge_and_deduplicate(&[&merged1]);
        assert_eq!(merged1, merged2); // Merging is idempotent
    }
}
```

---

## Performance Targets

### Repair Time Budget

| Operation | Target | Acceptable | Notes |
|-----------|--------|------------|-------|
| **setup-cpp-auto** (fresh download) | 60s | 120s | Network-dependent |
| **setup-cpp-auto** (cached build) | 2s | 5s | File I/O only |
| **cargo clean** | 1s | 3s | Depends on target size |
| **cargo build xtask** | 15s | 30s | Depends on CPU |
| **Total repair (fresh)** | ~80s | ~160s | User-acceptable |
| **Total repair (cached)** | ~20s | ~40s | Fast feedback |

### Memory Usage

- **Peak memory during repair**: < 500 MB (cargo build dominates)
- **No memory leaks**: Validate with valgrind or sanitizers

### Network Bandwidth

- **C++ repo clone**: ~50 MB (first time)
- **Retry backoff**: Exponential (1s, 2s, 4s) to avoid server overload

---

## Risks and Mitigations

### Risk 1: Rebuild Recursion

**Risk**: Rebuilt xtask binary re-invokes preflight, causing infinite loop

**Likelihood**: Medium
**Impact**: High (infinite recursion, system hang)

**Mitigation**:
```rust
// Set environment variable to prevent recursion
fn attempt_repair(backend: CppBackend, verbose: bool) -> Result<(), RepairError> {
    // Check recursion guard
    if env::var("BITNET_REPAIR_IN_PROGRESS").is_ok() {
        return Err(RepairError::SetupFailed(
            "Repair already in progress (recursion detected)".to_string()
        ));
    }

    // Set guard
    env::set_var("BITNET_REPAIR_IN_PROGRESS", "1");

    // ... repair logic ...

    // Unset guard on completion
    env::remove_var("BITNET_REPAIR_IN_PROGRESS");

    Ok(())
}
```

---

### Risk 2: Partial Rebuild Leaves Stale Constants

**Risk**: Rebuild fails mid-way, leaving xtask binary in inconsistent state

**Likelihood**: Low
**Impact**: High (false positives in detection)

**Mitigation**:
- Use atomic rebuilds: `cargo clean -p xtask` before `cargo build -p xtask`
- Validate build success before revalidation
- Provide manual rollback instructions in error message

---

### Risk 3: Network Timeouts in CI

**Risk**: CI builds fail due to network issues during auto-repair

**Likelihood**: Medium
**Impact**: Medium (CI flakiness)

**Mitigation**:
- Auto-detect CI and default to `--no-repair`
- Allow explicit `--repair` override for CI debugging
- Recommend pre-installing C++ deps in CI Docker images

---

### Risk 4: Concurrent Repairs

**Risk**: Multiple processes attempt repair simultaneously, causing file conflicts

**Likelihood**: Low
**Impact**: Medium (repair failure, corrupted cache)

**Mitigation**:
- File locking in `setup-cpp-auto` (already implemented via `FileExt::lock()`)
- Add advisory lock in `attempt_repair()` using same mechanism

---

### Risk 5: Platform-Specific Failures

**Risk**: Repair works on Linux but fails on macOS/Windows

**Likelihood**: Medium
**Impact**: Medium (platform inconsistency)

**Mitigation**:
- Cross-platform testing in CI (Linux, macOS, Windows)
- Document platform-specific requirements (CMake, Git, compilers)
- Provide platform-specific error messages

---

## Documentation Requirements

### User-Facing Docs

1. **docs/development/xtask.md**
   - Update `preflight` command section
   - Add `--repair` / `--no-repair` flag documentation
   - Add examples:
     ```bash
     # Auto-repair (interactive default)
     cargo run -p xtask -- preflight --backend bitnet

     # Explicit no-repair (CI mode)
     cargo run -p xtask -- preflight --backend bitnet --no-repair

     # Verbose repair progress
     cargo run -p xtask -- preflight --backend bitnet --repair --verbose
     ```

2. **CLAUDE.md**
   - Update "Troubleshooting" section
   - Add auto-repair workflow example
   - Update first-time setup guide

3. **docs/howto/cpp-setup.md**
   - Add "Automatic Setup (Recommended)" section
   - Document manual setup as fallback
   - Add troubleshooting subsection for repair failures

4. **docs/explanation/dual-backend-crossval.md**
   - Add section on auto-repair architecture
   - Document detection → repair → redetect flow
   - Add sequence diagrams

---

### Developer Docs

1. **Architecture Decision Record (ADR)**
   - **Title**: Auto-Repair Default Behavior for C++ Backend Setup
   - **Status**: Proposed
   - **Context**: Users face 4-step manual setup process
   - **Decision**: Auto-repair by default in interactive, no-repair in CI
   - **Consequences**: Reduced setup friction, added complexity in error handling

2. **Inline Code Documentation**
   - Rustdoc comments for all public APIs
   - Examples in docstrings
   - Performance notes for expensive operations

---

## Success Metrics

### Quantitative

1. **Setup time reduction**: 4-step manual → 1-step auto (75% reduction)
2. **First-time success rate**: Target 90% success on first attempt
3. **Error recovery rate**: 80% of failures provide actionable fix
4. **Test coverage**: ≥90% line coverage for auto-repair code paths

### Qualitative

1. **User feedback**: Gather feedback from 5+ new contributors
2. **Documentation clarity**: Zero confusion on `--repair` vs `--no-repair`
3. **CI stability**: No flaky tests due to auto-repair in CI

---

## Future Enhancements (Out of Scope)

### FE1: Interactive Repair Confirmation

**Description**: Prompt user before starting repair (optional)

**Example**:
```
❌ Backend 'bitnet.cpp' libraries NOT FOUND

Attempt auto-repair? This will:
  1. Download and build C++ reference (~60s)
  2. Rebuild xtask to detect libraries (~15s)

Continue? [Y/n]:
```

**Effort**: 2 hours (CLI prompt handling)

---

### FE2: Repair Progress Bar

**Description**: Visual progress bar instead of verbose text

**Example**:
```
Repairing bitnet.cpp backend...
[=====>                    ] 25% - Building C++ libraries...
```

**Effort**: 4 hours (integrate indicatif crate)

---

### FE3: Backend Version Management

**Description**: Allow users to specify C++ backend version

**Example**:
```bash
cargo run -p xtask -- preflight --backend bitnet --repair --version v0.5.2
```

**Effort**: 6 hours (version detection, tag checkout)

---

### FE4: Repair Dry-Run Mode

**Description**: Show what repair would do without executing

**Example**:
```bash
cargo run -p xtask -- preflight --backend bitnet --repair --dry-run
# Would execute:
#   1. setup-cpp-auto --emit=sh
#   2. cargo clean -p xtask -p crossval
#   3. cargo build -p xtask --features crossval-all
```

**Effort**: 3 hours (command preview mode)

---

## Open Questions

1. **Q**: Should `--repair` be the default in all environments?
   **A**: No. Auto-detect: repair in interactive, no-repair in CI (via `RepairMode::Auto`).

2. **Q**: How to handle repair on Windows where RPATH doesn't exist?
   **A**: Repair still works (setup-cpp-auto emits PATH exports). Windows limitation documented separately.

3. **Q**: Should repair support both backends simultaneously?
   **A**: Yes. When no `--backend` specified, repair all missing backends.

4. **Q**: What happens if user cancels repair mid-way (Ctrl+C)?
   **A**: Partial state left on disk. Document manual cleanup steps. Consider adding signal handlers (future enhancement).

5. **Q**: Should repair cache downloaded artifacts?
   **A**: Yes. `setup-cpp-auto` already caches at `~/.cache/bitnet_cpp`. No changes needed.

---

## Glossary

- **Auto-repair**: Automatic installation and configuration of missing C++ backend dependencies
- **Backend**: C++ reference implementation (BitNet.cpp or llama.cpp)
- **Build-time constants**: Compile-time flags set by `crossval/build.rs` (HAS_BITNET, HAS_LLAMA)
- **Preflight**: Diagnostic command to check backend availability before cross-validation
- **RPATH**: Runtime library search path embedded in binary (Linux/macOS)
- **Repair**: Process of installing missing dependencies and rebuilding to detect them
- **Revalidation**: Re-checking backend availability after repair

---

## Appendix A: Example Error Messages

### Error 1: Network Failure

```
❌ Auto-repair failed: network error

Network error details:
  Failed to clone https://github.com/microsoft/BitNet.git
  Error: connection timeout after 30s

Manual recovery steps:
  1. Check network connectivity:
     ping github.com

  2. Verify firewall allows git clone:
     git clone https://github.com/microsoft/BitNet.git /tmp/test-clone

  3. If network is working, retry repair:
     cargo run -p xtask -- preflight --backend bitnet --repair

  4. If problem persists, try manual setup:
     cargo run -p xtask -- fetch-cpp --repo https://github.com/microsoft/BitNet.git

For more help, see:
  docs/howto/cpp-setup.md (Detailed C++ setup guide)
  docs/explanation/dual-backend-crossval.md (Architecture overview)
```

---

### Error 2: Build Failure

```
❌ Auto-repair failed: build error

Build error details:
  CMake configuration failed: CUDA not found

  CMake output:
  -- The C compiler identification is GNU 11.4.0
  -- Detecting CUDA compiler...
  CMake Error at CMakeLists.txt:42 (find_package):
    Could not find a package configuration file provided by "CUDA" with any
    of the following names:
      CUDAConfig.cmake
      cuda-config.cmake

Manual recovery steps:
  1. Check build requirements are installed:
     cmake --version  # Should be >= 3.18
     gcc --version    # Or clang

  2. For GPU support, install CUDA toolkit:
     sudo apt install nvidia-cuda-toolkit  # Ubuntu/Debian
     brew install cuda                      # macOS

  3. If CPU-only build desired, try:
     cargo run -p xtask -- fetch-cpp --backend cpu --cmake-flags "-DGGML_CUDA=OFF"

  4. Review full build logs in:
     ~/.cache/bitnet_cpp/build/CMakeOutput.log

For more help, see:
  docs/howto/cpp-setup.md (Detailed C++ setup guide)
  docs/GPU_SETUP.md (GPU-specific setup)
```

---

### Error 3: Permission Denied

```
❌ Auto-repair failed: permission denied

Permission error details:
  Cannot write to /home/user/.cache/bitnet_cpp
  Error: Permission denied (os error 13)

Manual recovery steps:
  1. Check directory ownership:
     ls -ld ~/.cache/bitnet_cpp

  2. Fix ownership if needed:
     sudo chown -R $USER:$USER ~/.cache/bitnet_cpp

  3. If running in restricted environment, specify custom directory:
     export BITNET_CPP_DIR=/tmp/bitnet_cpp
     cargo run -p xtask -- preflight --backend bitnet --repair

  4. Retry repair:
     cargo run -p xtask -- preflight --backend bitnet --repair

For more help, see:
  docs/howto/cpp-setup.md (Detailed C++ setup guide)
```

---

## Appendix B: Implementation Checklist

**Phase 1: Core Auto-Repair Logic** (8 hours)
- [ ] Implement `attempt_repair()` function
- [ ] Implement `rebuild_xtask_for_detection()` function
- [ ] Add `RepairError` and `RebuildError` types
- [ ] Write unit tests for repair logic
- [ ] Write integration test for end-to-end repair

**Phase 2: CLI Integration** (4 hours)
- [ ] Add `--repair` flag to `Preflight` command
- [ ] Add `--no-repair` flag to `Preflight` command
- [ ] Implement `RepairMode` enum with CI auto-detection
- [ ] Update `cpp_backend_preflight_cmd()` handler
- [ ] Write unit tests for flag parsing
- [ ] Write integration test for CI auto-detection

**Phase 3: Error Handling and Retry Logic** (6 hours)
- [ ] Implement retry logic with exponential backoff
- [ ] Add `is_retryable_error()` classification
- [ ] Implement `format_repair_error_with_recovery()`
- [ ] Add network error detection
- [ ] Add build error detection
- [ ] Add permission error detection
- [ ] Write unit tests for error handling
- [ ] Write integration tests for retry scenarios

**Phase 4: Verbose Progress Reporting** (4 hours)
- [ ] Implement `RepairProgress` struct
- [ ] Add progress logging to repair flow
- [ ] Capture setup-cpp-auto output
- [ ] Capture cargo build output
- [ ] Format progress messages
- [ ] Write integration test for verbose output

**Phase 5: Testing and Documentation** (8 hours)
- [ ] Write comprehensive test suite (7 integration tests)
- [ ] Achieve ≥90% line coverage
- [ ] Update `docs/development/xtask.md`
- [ ] Update `CLAUDE.md`
- [ ] Update `docs/howto/cpp-setup.md`
- [ ] Create ADR for auto-repair design
- [ ] Add inline Rustdoc comments
- [ ] Review and merge

**Total Estimated Effort**: 30 hours (~4 working days)

---

## Appendix C: Compatibility Matrix

| Platform | Auto-Repair Support | RPATH Support | Notes |
|----------|---------------------|---------------|-------|
| **Linux (Ubuntu 20.04+)** | ✓ Full | ✓ Yes | Primary development platform |
| **Linux (Debian 11+)** | ✓ Full | ✓ Yes | Tested in CI |
| **macOS (12+)** | ✓ Full | ✓ Yes | Requires Xcode command-line tools |
| **Windows 10/11 (Git Bash)** | ✓ Partial | ✗ No | Uses PATH instead of RPATH |
| **Windows (PowerShell)** | ✓ Partial | ✗ No | Native PowerShell support |
| **Windows (WSL2)** | ✓ Full | ✓ Yes | Treated as Linux |

**Notes**:
- Windows "Partial" support means repair works but requires PATH management
- macOS code signing may require additional user action (documented separately)
- WSL2 provides best Windows experience (full Linux compatibility)

---

## Document Metadata

- **Created**: 2025-10-25
- **Author**: BitNet.rs Spec Agent (Claude)
- **Status**: Draft (awaiting review)
- **Target Release**: v0.2.0
- **Estimated Effort**: 30 hours (4 working days)
- **Complexity**: Medium
- **Priority**: High (reduces setup friction significantly)
- **Dependencies**: None (builds on existing infrastructure)

---

**End of Specification**

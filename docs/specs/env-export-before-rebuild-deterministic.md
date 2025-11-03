# Environment Export Before Rebuild: Deterministic Auto-Repair

**Status**: Draft
**Created**: 2025-10-27
**Updated**: 2025-10-27
**Version**: 1.0.0
**Priority**: P0 (Critical)
**Complexity**: Medium
**Estimated Effort**: 5-8 hours

## Overview

This specification addresses the **environment export propagation gap** in BitNet.rs auto-repair workflow. Currently, `setup-cpp-auto` successfully installs C++ libraries and outputs shell exports (e.g., `export BITNET_CPP_DIR="/path"`), but these environment variables are **never applied** to the subsequent `rebuild_xtask()` subprocess. This causes the rebuild to inherit stale environment variables, preventing build.rs from detecting the newly-installed libraries.

**Impact**: The auto-repair workflow appears to succeed but produces a binary with `HAS_BITNET=false` (BITNET_STUB mode), breaking deterministic repair and requiring manual intervention.

## Problem Statement

### Current Broken Flow

```
┌─ setup-cpp-auto               ┌─ rebuild_xtask()
│  (installs libs)              │  (runs cargo build)
│  outputs: export VAR="..."    │
│           export PATH="..."   │
│                               │
└─→ [SUBPROCESS STDOUT]        ├─ Inherits PARENT env (stale!)
   [OUTPUT DISCARDED]           │
                                ├─ Does NOT see new BITNET_CPP_DIR
                                │
                                └─→ build.rs can't find libraries
                                    HAS_BITNET stays false
```

### The Three Gaps

| Gap | Location | Problem | Impact |
|-----|----------|---------|--------|
| **Gap 1** | `preflight.rs:1970-1990` | `setup-cpp-auto` stdout captured but never read | Env exports lost |
| **Gap 2** | `preflight.rs:1404` | `rebuild_xtask()` doesn't pass env to child cargo | Child doesn't see BITNET_CPP_DIR |
| **Gap 3** | `preflight.rs` | No parsing function exists | Can't extract `export VAR=value` lines |

### Why This Matters

**Symptom**: After successful auto-repair, user sees:
```
cargo:warning=crossval: ✗ BITNET_STUB mode: No C++ libraries found
cargo:warning=crossval: Set BITNET_CPP_DIR to enable C++ backend integration
```

**Expected**: After successful auto-repair, user should see:
```
cargo:warning=crossval: ✓ BITNET_FULL: BitNet.cpp and llama.cpp libraries found
cargo:warning=crossval: Backend: full
cargo:warning=crossval: Linked libraries: bitnet, llama, ggml
```

## Goals

1. **Parse shell exports** from `setup-cpp-auto` output (sh, fish, PowerShell formats)
2. **Apply environment variables** to current process before rebuild
3. **Pass environment** to child `cargo build` subprocess
4. **Achieve deterministic repair**: After repair + rebuild + re-exec, `HAS_BITNET=true`

## Non-Goals

- **Cross-platform shell execution**: We only parse exports, not execute arbitrary shell code
- **Dynamic loader configuration**: We pass `LD_LIBRARY_PATH` to cargo, but don't modify system linker
- **Persistent environment changes**: Changes are scoped to current process and children only

## Acceptance Criteria

### AC1: Parse Shell Exports

**Given**: Shell output from `setup-cpp-auto --emit=sh|fish|pwsh|cmd`
**When**: Calling `parse_sh_exports(output)`
**Then**: Returns `HashMap<String, String>` with extracted key=value pairs

**Test Coverage**:
1. Parse POSIX sh format: `export VAR="value"`
2. Parse POSIX sh with path expansion: `export PATH="/path:${PATH:-}"`
3. Parse fish format: `set -gx VAR "value"`
4. Parse PowerShell format: `$env:VAR = "value"`
5. Parse cmd.exe format: `set VAR=value`
6. Handle quoted values with spaces
7. Skip non-export lines (echo, comments)
8. Handle empty input gracefully

**Success Metrics**:
- All 5 shell formats parsed correctly
- Zero false positives (non-export lines ignored)
- Preserves value escaping and special characters

### AC2: Apply Environment Variables

**Given**: HashMap of environment variables from parse step
**When**: Calling `apply_env_exports(&exports)`
**Then**: Variables are set in current process AND available to child processes

**Test Coverage**:
1. Variables visible via `std::env::var()` after apply
2. Variables inherited by child `Command::spawn()`
3. Dynamic loader paths (LD_LIBRARY_PATH) correctly propagated
4. Windows PATH correctly propagated
5. Original values restored on EnvGuard drop (test isolation)

**Success Metrics**:
- 100% of parsed variables visible in current process
- Child processes inherit all applied variables
- No environment pollution between tests

### AC3: Integration with Repair Flow

**Given**: Successful `setup-cpp-auto` execution
**When**: Running `preflight_with_auto_repair()`
**Then**: Environment exports applied before `rebuild_xtask()`

**Test Coverage**:
1. `attempt_repair_once()` returns stdout containing exports
2. `preflight_with_auto_repair()` parses stdout
3. `rebuild_xtask_with_env()` receives parsed exports
4. Child cargo process receives BITNET_CPP_DIR environment variable
5. Verbose mode logs applied environment variables

**Success Metrics**:
- Zero data loss between setup-cpp-auto and rebuild
- All environment variables reach child cargo process
- Verbose mode shows human-readable env var listing

### AC4: Persistent Detection After Re-exec

**Given**: Completed repair + rebuild + re-exec cycle
**When**: Checking `HAS_BITNET` constant in re-exec child
**Then**: `HAS_BITNET=true` (no BITNET_STUB mode)

**Test Coverage**:
1. Integration test: full repair → rebuild → re-exec flow
2. Verify re-exec child sees BITNET_FULL mode
3. Verify no "BITNET_STUB mode" warning in cargo output
4. Verify `preflight_backend_libs()` succeeds in re-exec child

**Success Metrics**:
- Re-exec child reports "✓ bitnet.cpp AVAILABLE"
- Zero BITNET_STUB warnings after successful repair
- Deterministic: Same result on repeated runs

## Technical Design

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                   preflight_with_auto_repair()                  │
│                                                                   │
│  Step 1: Check backend availability                              │
│  Step 2: Determine repair needed (RepairMode)                    │
│  Step 3: Call attempt_repair_with_retry()                        │
│           ↓                                                       │
│           └─→ Returns stdout (shell exports)                     │
│                                                                   │
│  [NEW] Step 3b: Parse shell exports                              │
│           ↓                                                       │
│           parse_sh_exports(stdout) → HashMap<String, String>     │
│                                                                   │
│  [NEW] Step 3c: Apply to current process                         │
│           ↓                                                       │
│           apply_env_exports(&exports)                            │
│                                                                   │
│  Step 4: Rebuild with environment                                │
│           ↓                                                       │
│           rebuild_xtask_with_env(verbose, &exports)              │
│           ├─→ Command::new("cargo")                              │
│           ├─→ .env("BITNET_CPP_DIR", "...")                      │
│           ├─→ .env("LD_LIBRARY_PATH", "...")                     │
│           └─→ .status() → build succeeds with detection          │
│                                                                   │
│  Step 5: Re-exec with updated binary                             │
│           ↓                                                       │
│           reexec_current_command() → HAS_BITNET=true            │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow Diagram

```
setup-cpp-auto --emit=sh
        │
        ├─ stdout: "export BITNET_CPP_DIR=\"/path\"\n"
        │          "export LD_LIBRARY_PATH=\"/lib:${LD_LIBRARY_PATH:-}\"\n"
        │          "echo \"[bitnet] C++ ready\"\n"
        ↓
attempt_repair_once() returns String
        │
        ├─ Raw shell output (multi-line string)
        ↓
parse_sh_exports(&output) → Result<HashMap<String, String>>
        │
        ├─ HashMap {
        │     "BITNET_CPP_DIR": "/home/user/.cache/bitnet_cpp",
        │     "LD_LIBRARY_PATH": "/home/user/.cache/bitnet_cpp/build/bin:${LD_LIBRARY_PATH:-}",
        │     "BITNET_CROSSVAL_LIBDIR": "/home/user/.cache/bitnet_cpp/build/3rdparty/llama.cpp/build/bin",
        │   }
        ↓
apply_env_exports(&exports)
        │
        ├─ unsafe { env::set_var("BITNET_CPP_DIR", "...") }
        ├─ unsafe { env::set_var("LD_LIBRARY_PATH", "...") }
        ↓
rebuild_xtask_with_env(verbose, &exports)
        │
        ├─ Command::new("cargo")
        │     .args(["build", "-p", "xtask", "--features", "crossval-all"])
        │     .env("BITNET_CPP_DIR", "/home/user/.cache/bitnet_cpp")
        │     .env("LD_LIBRARY_PATH", "/home/user/.cache/bitnet_cpp/build/bin:...")
        │     .status()
        ↓
cargo build -p xtask (child process)
        │
        ├─ Inherits environment: BITNET_CPP_DIR set
        ├─ Runs crossval/build.rs
        │   ├─ env::var("BITNET_CPP_DIR") = Ok("/home/user/.cache/bitnet_cpp")
        │   ├─ Searches for libraries in detected paths
        │   └─ Finds libbitnet.so, libllama.so, libggml.so
        ├─ Emits: cargo:rustc-env=CROSSVAL_HAS_BITNET=true
        └─ Emits: cargo:rustc-env=CROSSVAL_BACKEND_STATE=full
        ↓
xtask binary rebuilt with HAS_BITNET=true
```

### Component Interfaces

#### 1. parse_sh_exports() Function

**Location**: `xtask/src/crossval/preflight.rs` (new function)

**Signature**:
```rust
/// Parse shell export format from setup-cpp-auto output
///
/// Handles multiple shell formats:
/// - POSIX sh/bash/zsh: export VAR="value"
/// - fish shell: set -gx VAR "value"
/// - PowerShell: $env:VAR = "value"
/// - cmd.exe: set VAR=value
///
/// # Arguments
///
/// * `output` - Raw shell output from setup-cpp-auto
///
/// # Returns
///
/// * `Ok(HashMap<String, String>)` - Parsed environment variables
/// * `Err(String)` - Parsing error with diagnostic message
///
/// # Example
///
/// ```rust
/// let output = r#"export BITNET_CPP_DIR="/path/to/bitnet"
/// export LD_LIBRARY_PATH="/path/to/libs:${LD_LIBRARY_PATH:-}"
/// echo "[bitnet] C++ ready""#;
///
/// let exports = parse_sh_exports(output)?;
/// assert_eq!(exports.get("BITNET_CPP_DIR"), Some(&"/path/to/bitnet".to_string()));
/// ```
fn parse_sh_exports(output: &str) -> Result<HashMap<String, String>, String>
```

**Algorithm**:
1. Split input by newlines
2. For each line, try patterns in priority order:
   - `export VAR="value"` (POSIX sh)
   - `set -gx VAR "value"` (fish)
   - `$env:VAR = "value"` (PowerShell)
   - `set VAR=value` (cmd.exe)
3. Extract key and value, handling quotes
4. Handle variable references (e.g., `${PATH:-}`) as literal strings
5. Skip non-export lines (echo, comments, empty)
6. Return HashMap or error on malformed input

**Edge Cases**:
- Quoted values with spaces: `export VAR="value with spaces"`
- Nested quotes: `export VAR="\"quoted\""`
- Variable references: `export PATH="/new:${PATH:-}"`
- Empty values: `export VAR=""`
- Non-ASCII characters: Preserve as UTF-8

#### 2. rebuild_xtask_with_env() Function

**Location**: `xtask/src/crossval/preflight.rs` (new function)

**Signature**:
```rust
/// Rebuild xtask with environment variables
///
/// Applies parsed environment variables from setup-cpp-auto to the cargo
/// build process, ensuring build.rs detection can find newly-installed libraries.
///
/// # Arguments
///
/// * `verbose` - If true, print progress messages and env var listing
/// * `exports` - Environment variables to apply to child cargo process
///
/// # Returns
///
/// * `Ok(())` - Rebuild succeeded
/// * `Err(RebuildError)` - Rebuild failed with diagnostic message
///
/// # Example
///
/// ```rust
/// let mut exports = HashMap::new();
/// exports.insert("BITNET_CPP_DIR".to_string(), "/path/to/bitnet".to_string());
///
/// rebuild_xtask_with_env(true, &exports)?;
/// ```
fn rebuild_xtask_with_env(
    verbose: bool,
    exports: &HashMap<String, String>,
) -> Result<(), RebuildError>
```

**Implementation**:
1. Create `Command::new("cargo")`
2. Add args: `["build", "-p", "xtask", "--features", "crossval-all"]`
3. For each (key, value) in exports:
   - Call `cmd.env(key, value)`
   - If verbose, print `[preflight]   KEY = VALUE`
4. Execute `.status()` and check exit code
5. Return `Ok(())` on success, `Err(RebuildError)` on failure

**Error Handling**:
- I/O error spawning cargo → `RebuildError::BuildFailed`
- Non-zero exit code → `RebuildError::BuildFailed` with exit code
- Preserve cargo stderr for diagnostics

#### 3. Modified attempt_repair_once()

**Location**: `xtask/src/crossval/preflight.rs:1951-1999` (MODIFIED)

**Changes**:
- **Return type**: `Result<(), RepairError>` → `Result<String, RepairError>`
- **Capture stdout**: `String::from_utf8_lossy(&setup_result.stdout).to_string()`
- **Return stdout**: `Ok(stdout)` instead of `Ok(())`

**Modified Signature**:
```rust
fn attempt_repair_once(backend: CppBackend, verbose: bool) -> Result<String, RepairError>
```

**Key Change** (Line 1990-1999):
```rust
// OLD: Discard stdout
if !setup_result.status.success() {
    let stderr = String::from_utf8_lossy(&setup_result.stderr);
    return Err(RepairError::classify(&stderr, backend.name()));
}
Ok(())

// NEW: Capture and return stdout
let stdout = String::from_utf8_lossy(&setup_result.stdout).to_string();

if !setup_result.status.success() {
    let stderr = String::from_utf8_lossy(&setup_result.stderr);
    return Err(RepairError::classify(&stderr, backend.name()));
}

progress.log("REPAIR", "C++ libraries installed successfully");
progress.log("REBUILD", "Next: Rebuild xtask to detect libraries");

Ok(stdout)  // ← Return the shell exports
```

#### 4. Modified preflight_with_auto_repair()

**Location**: `xtask/src/crossval/preflight.rs:1393-1407` (MODIFIED)

**Changes**:
1. Capture output from `attempt_repair_with_retry()` (now returns `Result<String, RepairError>`)
2. Parse shell exports with `parse_sh_exports()`
3. Apply to current process with `apply_env_exports()`
4. Pass to `rebuild_xtask_with_env()` instead of `rebuild_xtask()`

**Modified Code** (Line 1393-1407):
```rust
// OLD: Discard output
if let Err(e) = attempt_repair_with_retry(backend, verbose) {
    eprintln!("\n{}", e);
    bail!("Auto-repair failed for backend '{}'", backend.name());
}

if let Err(e) = rebuild_xtask(verbose) {
    eprintln!("\n{}", e);
    bail!("xtask rebuild failed after successful C++ setup");
}

// NEW: Capture, parse, apply, rebuild with env
let setup_output = match attempt_repair_with_retry(backend, verbose) {
    Ok(output) => output,
    Err(e) => {
        eprintln!("\n{}", e);
        bail!("Auto-repair failed for backend '{}'", backend.name());
    }
};

// Parse environment exports
let exports = match parse_sh_exports(&setup_output) {
    Ok(map) => map,
    Err(e) => {
        eprintln!("Failed to parse setup-cpp-auto output: {}", e);
        bail!("Environment parsing failed");
    }
};

// Apply to current process (for diagnostics/logging)
for (key, value) in &exports {
    unsafe {
        env::set_var(key, value);
    }
}

// Step 4: Rebuild xtask to pick up new detection (AC3)
if verbose {
    eprintln!("[repair] Step 2/3: Rebuilding xtask binary...");
}

if let Err(e) = rebuild_xtask_with_env(verbose, &exports) {
    eprintln!("\n{}", e);
    bail!("xtask rebuild failed after successful C++ setup");
}
```

### Implementation Patterns

#### Pattern 1: Parse Key-Value Helper

```rust
/// Helper: Parse key=value from a string
/// Returns (key, value) if found, handling quoted values
fn parse_key_value(s: &str, quote: char) -> Option<(&str, &str)> {
    let parts: Vec<&str> = s.splitn(2, '=').collect();
    if parts.len() != 2 {
        return None;
    }

    let key = parts[0].trim();
    let value_str = parts[1].trim();

    let value = if quote != '\0' {
        // Handle quoted values: "value" or 'value'
        value_str
            .strip_prefix(quote)
            .and_then(|s| s.strip_suffix(quote))
            .unwrap_or(value_str)
    } else {
        value_str
    };

    Some((key, value))
}
```

#### Pattern 2: POSIX sh Export Parsing

```rust
// Parse POSIX sh/bash/zsh format: export VAR="value"
if let Some(rest) = trimmed.strip_prefix("export ") {
    if let Some((key, value)) = parse_key_value(rest, '"') {
        exports.insert(key.to_string(), value.to_string());
        continue;
    }
}
```

#### Pattern 3: Fish Shell Parsing

```rust
// Parse fish shell format: set -gx VAR "value"
if trimmed.starts_with("set ") && trimmed.contains("-gx ") {
    let parts: Vec<&str> = trimmed.split_whitespace().collect();
    if parts.len() >= 4 && parts[1] == "-gx" {
        let key = parts[2];
        // Extract value after key, handling quotes
        let value_start = trimmed.find(key).unwrap() + key.len();
        let value_part = trimmed[value_start..].trim();
        if let Some(value) = value_part.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
            exports.insert(key.to_string(), value.to_string());
        }
    }
}
```

#### Pattern 4: PowerShell Parsing

```rust
// Parse PowerShell format: $env:VAR = "value"
if let Some(rest) = trimmed.strip_prefix("$env:") {
    if let Some((key, value)) = parse_key_value(rest, '"') {
        exports.insert(key.to_string(), value.to_string());
        continue;
    }
}
```

#### Pattern 5: Environment Application

```rust
/// Apply parsed environment variables to current process
///
/// WARNING: Uses unsafe env::set_var. Caller must ensure:
/// - Called from single-threaded context or with proper serialization
/// - Test isolation via EnvGuard or #[serial(bitnet_env)]
fn apply_env_exports(exports: &HashMap<String, String>) {
    for (key, value) in exports {
        // SAFETY: This is safe because:
        // 1. We're in the auto-repair flow before spawning child processes
        // 2. The calling test uses #[serial(bitnet_env)] for isolation
        // 3. These env vars are specifically for child cargo build process
        unsafe {
            env::set_var(key, value);
        }
    }
}
```

## Implementation Phases

### Phase 1: Pure Functions (Parsing)

**Deliverables**:
- `parse_sh_exports()` function with comprehensive unit tests
- `parse_key_value()` helper function
- Unit tests for all 5 shell formats (sh, fish, pwsh, cmd, edge cases)

**Test Strategy**:
```rust
#[cfg(test)]
mod parse_tests {
    use super::*;

    #[test]
    fn test_parse_posix_sh_export() {
        let output = r#"export BITNET_CPP_DIR="/home/user/.cache/bitnet_cpp""#;
        let exports = parse_sh_exports(output).unwrap();
        assert_eq!(exports.get("BITNET_CPP_DIR"), Some(&"/home/user/.cache/bitnet_cpp".to_string()));
    }

    #[test]
    fn test_parse_fish_set() {
        let output = r#"set -gx BITNET_CPP_DIR "/home/user/.cache/bitnet_cpp""#;
        let exports = parse_sh_exports(output).unwrap();
        assert_eq!(exports.get("BITNET_CPP_DIR"), Some(&"/home/user/.cache/bitnet_cpp".to_string()));
    }

    #[test]
    fn test_parse_powershell_env() {
        let output = r#"$env:BITNET_CPP_DIR = "/home/user/.cache/bitnet_cpp""#;
        let exports = parse_sh_exports(output).unwrap();
        assert_eq!(exports.get("BITNET_CPP_DIR"), Some(&"/home/user/.cache/bitnet_cpp".to_string()));
    }

    #[test]
    fn test_parse_multiple_exports() {
        let output = r#"export BITNET_CPP_DIR="/path1"
export LD_LIBRARY_PATH="/path2:/path3"
export BITNET_CROSSVAL_LIBDIR="/path4"
echo "[bitnet] C++ ready""#;
        let exports = parse_sh_exports(output).unwrap();
        assert_eq!(exports.len(), 3);
        assert_eq!(exports.get("BITNET_CPP_DIR"), Some(&"/path1".to_string()));
        assert_eq!(exports.get("LD_LIBRARY_PATH"), Some(&"/path2:/path3".to_string()));
        assert_eq!(exports.get("BITNET_CROSSVAL_LIBDIR"), Some(&"/path4".to_string()));
    }

    #[test]
    fn test_parse_skips_non_exports() {
        let output = r#"export BITNET_CPP_DIR="/path"
echo "[bitnet] C++ ready"
# This is a comment
export LD_LIBRARY_PATH="/lib""#;
        let exports = parse_sh_exports(output).unwrap();
        assert_eq!(exports.len(), 2);
        assert!(!exports.contains_key("echo"));
    }

    #[test]
    fn test_parse_quoted_values_with_spaces() {
        let output = r#"export PROJECT_NAME="BitNet C++ Backend""#;
        let exports = parse_sh_exports(output).unwrap();
        assert_eq!(exports.get("PROJECT_NAME"), Some(&"BitNet C++ Backend".to_string()));
    }

    #[test]
    fn test_parse_variable_references_preserved() {
        let output = r#"export LD_LIBRARY_PATH="/path:${LD_LIBRARY_PATH:-}""#;
        let exports = parse_sh_exports(output).unwrap();
        // Variable references are preserved as literal strings
        assert_eq!(exports.get("LD_LIBRARY_PATH"), Some(&"/path:${LD_LIBRARY_PATH:-}".to_string()));
    }

    #[test]
    fn test_parse_empty_input() {
        let output = "";
        let exports = parse_sh_exports(output).unwrap();
        assert_eq!(exports.len(), 0);
    }
}
```

**Validation**:
- All 8 test cases pass
- Code coverage ≥ 90% for parsing functions
- No panics on malformed input

### Phase 2: Integration (Modify Repair Flow)

**Deliverables**:
- Modify `attempt_repair_once()` to return stdout
- Modify `attempt_repair_with_retry()` to return stdout
- Add `apply_env_exports()` function
- Modify `preflight_with_auto_repair()` to integrate parsing

**Test Strategy**:
```rust
#[cfg(test)]
mod integration_tests {
    use super::*;
    use serial_test::serial;
    use tests::support::env_guard::EnvGuard;

    #[test]
    #[serial(bitnet_env)]
    fn test_env_exports_reach_child_process() {
        let _guard1 = EnvGuard::new("BITNET_CPP_DIR");
        let _guard2 = EnvGuard::new("LD_LIBRARY_PATH");

        let mut exports = HashMap::new();
        exports.insert("BITNET_CPP_DIR".to_string(), "/test/path".to_string());
        exports.insert("LD_LIBRARY_PATH".to_string(), "/test/lib".to_string());

        apply_env_exports(&exports);

        // Verify variables visible in current process
        assert_eq!(env::var("BITNET_CPP_DIR").unwrap(), "/test/path");
        assert_eq!(env::var("LD_LIBRARY_PATH").unwrap(), "/test/lib");

        // Verify variables passed to child process
        let output = Command::new("sh")
            .arg("-c")
            .arg("echo $BITNET_CPP_DIR")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("/test/path"));
    }

    #[test]
    #[serial(bitnet_env)]
    fn test_attempt_repair_once_returns_stdout() {
        // Mock setup-cpp-auto command that outputs exports
        // Verify stdout captured and returned
        // Note: Requires mock infrastructure or integration test with real setup-cpp-auto
        unimplemented!("Integration test for stdout capture");
    }
}
```

**Validation**:
- `attempt_repair_once()` returns stdout containing exports
- Environment variables applied to current process
- Child processes inherit applied variables
- EnvGuard restores original state after tests

### Phase 3: Wiring (rebuild_xtask_with_env)

**Deliverables**:
- Add `rebuild_xtask_with_env()` function
- Wire into `preflight_with_auto_repair()` flow
- Add verbose logging of applied environment variables

**Test Strategy**:
```rust
#[cfg(test)]
mod rebuild_tests {
    use super::*;

    #[test]
    fn test_rebuild_xtask_with_env_passes_vars() {
        let mut exports = HashMap::new();
        exports.insert("BITNET_CPP_DIR".to_string(), "/test/path".to_string());

        // Note: This is a smoke test - full validation requires real cargo build
        // The key assertion is that Command::env() is called for each export
        let result = rebuild_xtask_with_env(false, &exports);
        // In mock context, verify .env() calls made
        // In integration test, verify build.rs detection succeeds
    }

    #[test]
    fn test_rebuild_xtask_with_env_verbose_logging() {
        let mut exports = HashMap::new();
        exports.insert("BITNET_CPP_DIR".to_string(), "/test/path".to_string());

        // Capture stderr to verify verbose logging
        // Assert: Contains "[preflight]   BITNET_CPP_DIR = /test/path"
        unimplemented!("Verbose logging test");
    }
}
```

**Validation**:
- `rebuild_xtask_with_env()` applies environment to cargo subprocess
- Verbose mode prints human-readable env var listing
- Rebuild succeeds with applied environment

### Phase 4: End-to-End Validation

**Deliverables**:
- Integration test for full repair → rebuild → re-exec flow
- Verification of HAS_BITNET=true after re-exec
- Documentation updates in repair workflow docs

**Test Strategy**:
```rust
#[cfg(test)]
#[cfg(feature = "crossval-all")]
mod e2e_tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[ignore] // Requires real setup-cpp-auto and cargo build
    #[serial(bitnet_env)]
    fn test_full_repair_flow_deterministic() {
        // Clean state: Remove BITNET_CPP_DIR if set
        let _guard = EnvGuard::new("BITNET_CPP_DIR");

        // Step 1: Run preflight with auto-repair
        let result = preflight_with_auto_repair(
            CppBackend::BitNet,
            true,  // verbose
            RepairMode::Auto,
        );

        // Step 2: Verify repair succeeded
        assert!(result.is_ok(), "Auto-repair should succeed");

        // Step 3: Verify HAS_BITNET=true after repair
        // Note: This requires re-exec, so we check re-exec child output
        // Assert: Output contains "✓ bitnet.cpp AVAILABLE"
        // Assert: No "BITNET_STUB mode" warning

        // Step 4: Verify deterministic on second run
        let result2 = preflight_with_auto_repair(
            CppBackend::BitNet,
            true,
            RepairMode::Auto,
        );
        assert!(result2.is_ok(), "Second run should also succeed");
    }

    #[test]
    #[ignore]
    #[serial(bitnet_env)]
    fn test_rebuild_detects_libraries_after_env_application() {
        // Simulate: setup-cpp-auto installed libs at /tmp/test_bitnet
        // Set environment: BITNET_CPP_DIR=/tmp/test_bitnet
        // Run: rebuild_xtask_with_env
        // Verify: cargo output contains "✓ BITNET_FULL"
        unimplemented!("Build detection integration test");
    }
}
```

**Validation**:
- Full repair cycle succeeds with HAS_BITNET=true
- No BITNET_STUB warnings after successful repair
- Deterministic behavior on repeated runs
- Re-exec child reports "✓ bitnet.cpp AVAILABLE"

## Test Strategy

### Unit Tests (Phase 1)

**File**: `xtask/src/crossval/preflight.rs` (inline `#[cfg(test)] mod parse_tests`)

**Coverage**:
- Parse POSIX sh export format (5 variants)
- Parse fish shell format (2 variants)
- Parse PowerShell format (2 variants)
- Parse cmd.exe format (1 variant)
- Edge cases: quoted values, variable references, empty input, malformed lines
- Negative cases: non-export lines, comments, invalid syntax

**Test Count**: 8 core tests + 4 edge case tests = **12 unit tests**

**Execution**:
```bash
cargo test -p xtask parse_tests --no-default-features --features crossval-all
```

### Integration Tests (Phase 2)

**File**: `xtask/tests/env_export_integration_tests.rs` (new file)

**Coverage**:
- Environment variables reach current process after `apply_env_exports()`
- Child processes inherit applied variables
- `attempt_repair_once()` returns stdout containing exports
- EnvGuard restores state after tests (isolation)

**Test Count**: 4 integration tests

**Execution**:
```bash
cargo test -p xtask env_export_integration_tests --no-default-features --features crossval-all -- --test-threads=1
```

**Note**: Use `#[serial(bitnet_env)]` for all tests that mutate environment.

### End-to-End Tests (Phase 4)

**File**: `xtask/tests/preflight_auto_repair_tests.rs` (existing file, new tests added)

**Coverage**:
- Full repair → rebuild → re-exec flow with environment propagation
- Verify HAS_BITNET=true after re-exec
- Deterministic behavior on repeated runs
- Verbose mode logging verification

**Test Count**: 2 e2e tests (marked `#[ignore]` for CI)

**Execution**:
```bash
# Local only (requires real setup-cpp-auto and cargo build)
cargo test -p xtask test_full_repair_flow_deterministic --no-default-features --features crossval-all -- --ignored --test-threads=1
```

### Test Isolation Pattern

**Use EnvGuard for all environment-mutating tests**:

```rust
use tests::support::env_guard::EnvGuard;
use serial_test::serial;

#[test]
#[serial(bitnet_env)]
fn test_env_propagation() {
    let _guard = EnvGuard::new("BITNET_CPP_DIR");

    // Test code here - environment automatically restored on drop
    let mut exports = HashMap::new();
    exports.insert("BITNET_CPP_DIR".to_string(), "/test".to_string());
    apply_env_exports(&exports);

    assert_eq!(env::var("BITNET_CPP_DIR").unwrap(), "/test");

    // Guard drops here, restoring original state
}
```

**Why This Matters**:
- Prevents test pollution (env vars leaking between tests)
- Ensures deterministic test execution
- Allows parallel test execution with `#[serial(bitnet_env)]` group

## Risks & Mitigation

### Risk 1: Shell Format Compatibility

**Risk**: Different platforms produce different export formats, parser may miss variations

**Likelihood**: Medium
**Impact**: High (auto-repair fails silently)

**Mitigation**:
1. Test all 5 shell formats (sh, fish, pwsh, cmd, edge cases)
2. Fallback to default paths if parsing fails
3. Verbose logging of parsed exports for diagnostics
4. Unit tests with real setup-cpp-auto output samples

### Risk 2: Quote Handling Edge Cases

**Risk**: Complex quoting (nested quotes, escaped characters) may break parser

**Likelihood**: Low
**Impact**: Medium (values truncated or corrupted)

**Mitigation**:
1. Use robust parsing with `splitn()` and `strip_prefix/suffix()`
2. Test with complex values: spaces, special chars, nested quotes
3. Preserve literal strings (don't expand variables like `${PATH}`)
4. Error handling with diagnostic messages

### Risk 3: Dynamic Loader Path Propagation

**Risk**: `LD_LIBRARY_PATH` (Linux) or `DYLD_LIBRARY_PATH` (macOS) may not reach child cargo

**Likelihood**: Low
**Impact**: High (build succeeds but runtime linking fails)

**Mitigation**:
1. Explicitly pass dynamic loader vars via `Command::env()`
2. Test child process inheritance with subprocess spawn test
3. Document platform-specific loader vars in code comments
4. CI validation on Linux, macOS, Windows

### Risk 4: Windows PATH Handling

**Risk**: Windows uses `;` separator, may break path concatenation logic

**Likelihood**: Low
**Impact**: Medium (Windows builds fail)

**Mitigation**:
1. Preserve exact value from setup-cpp-auto (no path manipulation)
2. setup-cpp-auto already handles platform-specific separators
3. Test Windows builds in CI (if available)
4. Document Windows-specific behavior

### Risk 5: Environment Pollution in Tests

**Risk**: Env vars leak between tests, causing flaky failures

**Likelihood**: Medium (without EnvGuard)
**Impact**: High (CI instability)

**Mitigation**:
1. **Mandatory**: Use `#[serial(bitnet_env)]` on all env-mutating tests
2. **Mandatory**: Use `EnvGuard` to restore state after tests
3. Document test isolation requirements in code comments
4. CI validation with `--test-threads=1` to catch races

## Success Metrics

### Functional Metrics

1. **Parse Accuracy**: 100% of setup-cpp-auto exports correctly parsed (all 5 formats)
2. **Environment Propagation**: 100% of parsed vars visible in child cargo process
3. **Detection Success**: After repair + rebuild, `HAS_BITNET=true` (no BITNET_STUB)
4. **Deterministic Repair**: Same result on repeated runs (no env state leakage)

### Quality Metrics

1. **Test Coverage**: ≥ 90% line coverage for parsing and integration code
2. **Test Count**: 12 unit tests + 4 integration tests + 2 e2e tests = **18 tests**
3. **Zero Regression**: Existing preflight tests continue to pass
4. **Zero Environment Pollution**: All tests use EnvGuard + `#[serial(bitnet_env)]`

### Performance Metrics

1. **Parsing Overhead**: < 1ms for typical setup-cpp-auto output (~10 lines)
2. **Rebuild Time**: No measurable increase (env vars add negligible overhead)
3. **Re-exec Time**: No change (same as existing workflow)

## Related Work

### Existing Specifications

- **preflight-auto-repair.md**: Base specification for auto-repair workflow (this extends it)
- **bitnet-cpp-auto-setup-parity.md**: setup-cpp-auto implementation (source of exports)
- **docs/explanation/dual-backend-crossval.md**: C++ backend architecture

### Related Issues

- **Issue #439** (RESOLVED): GPU/CPU feature gate unification
- **Issue #469** (ACTIVE): Tokenizer parity and FFI build hygiene

### Dependencies

- **setup-cpp-auto**: Must emit shell exports (already implemented)
- **crossval/build.rs**: Must read BITNET_CPP_DIR at build time (already implemented)
- **EnvGuard**: Test isolation infrastructure (already implemented)

## Implementation Checklist

### Phase 1: Parsing (2-3 hours)
- [ ] Implement `parse_sh_exports()` function
- [ ] Implement `parse_key_value()` helper
- [ ] Write 12 unit tests (8 core + 4 edge cases)
- [ ] Verify 100% parse accuracy on all formats
- [ ] Code review for edge cases

### Phase 2: Integration (1-2 hours)
- [ ] Modify `attempt_repair_once()` to return stdout
- [ ] Modify `attempt_repair_with_retry()` to return stdout
- [ ] Implement `apply_env_exports()` function
- [ ] Write 4 integration tests with EnvGuard
- [ ] Verify environment propagation to child processes

### Phase 3: Wiring (1-2 hours)
- [ ] Implement `rebuild_xtask_with_env()` function
- [ ] Modify `preflight_with_auto_repair()` integration
- [ ] Add verbose logging of applied env vars
- [ ] Write rebuild integration tests
- [ ] Verify cargo subprocess receives env vars

### Phase 4: Validation (1-2 hours)
- [ ] Write 2 e2e tests (full repair flow)
- [ ] Verify HAS_BITNET=true after re-exec
- [ ] Test deterministic behavior on repeated runs
- [ ] Update documentation in repair workflow docs
- [ ] Final code review and merge

### Testing (Throughout)
- [ ] All tests pass with `--test-threads=1`
- [ ] All tests use `#[serial(bitnet_env)]` for env isolation
- [ ] EnvGuard used in all environment-mutating tests
- [ ] CI validation on Linux (primary platform)
- [ ] Zero test pollution (env state isolated)

## References

### Analysis Documents

- **Main Analysis**: `/home/steven/code/Rust/BitNet-rs/docs/analysis/env-export-build-gap-analysis.md`
- **Code Snippets**: `/home/steven/code/Rust/BitNet-rs/docs/analysis/env-gap-code-snippets.md`
- **Quick Reference**: `/home/steven/code/Rust/BitNet-rs/docs/analysis/QUICK-REF-env-gap.md`

### Source Files

- **setup-cpp-auto**: `xtask/src/cpp_setup_auto.rs:707-866` (emit_exports function)
- **Repair Flow**: `xtask/src/crossval/preflight.rs:1326-1936` (preflight_with_auto_repair)
- **Build Detection**: `crossval/build.rs:131-189` (compile_ffi function)
- **Test Infrastructure**: `tests/support/env_guard.rs` (EnvGuard implementation)

### Line Number References

| File | Function | Lines | Change Type |
|------|----------|-------|-------------|
| `preflight.rs` | `parse_sh_exports()` | NEW | Add function |
| `preflight.rs` | `rebuild_xtask_with_env()` | NEW | Add function |
| `preflight.rs` | `apply_env_exports()` | NEW | Add function |
| `preflight.rs` | `attempt_repair_once()` | 1951-1999 | Modify return type |
| `preflight.rs` | `preflight_with_auto_repair()` | 1393-1407 | Integrate parsing |

## Appendix: Example Output

### Before Fix (BROKEN)

```
[repair] Step 1/3: Installing C++ backend...
[  1.23s] DETECT: Backend 'bitnet.cpp' not found
[  1.45s] REPAIR: Invoking setup-cpp-auto...
[123.45s] REPAIR: C++ libraries installed successfully
[123.46s] REBUILD: Next: Rebuild xtask to detect libraries

[repair] Step 2/3: Rebuilding xtask binary...
   Compiling crossval v0.1.0
warning: crossval: ✗ BITNET_STUB mode: No C++ libraries found
warning: crossval: Set BITNET_CPP_DIR to enable C++ backend integration
   Compiling xtask v0.1.0
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 5.23s

[repair] Step 3/3: Re-executing with updated detection...

ERROR: Backend 'bitnet.cpp' still unavailable after repair
```

### After Fix (WORKING)

```
[repair] Step 1/3: Installing C++ backend...
[  1.23s] DETECT: Backend 'bitnet.cpp' not found
[  1.45s] REPAIR: Invoking setup-cpp-auto...
[123.45s] REPAIR: C++ libraries installed successfully
[123.46s] REBUILD: Next: Rebuild xtask to detect libraries

[repair] Step 2/3: Rebuilding xtask binary...
[preflight] Rebuilding xtask with environment variables...
[preflight]   BITNET_CPP_DIR = /home/user/.cache/bitnet_cpp
[preflight]   BITNET_CROSSVAL_LIBDIR = /home/user/.cache/bitnet_cpp/build/3rdparty/llama.cpp/build/bin
[preflight]   LD_LIBRARY_PATH = /home/user/.cache/bitnet_cpp/build/bin:/usr/local/lib
   Compiling crossval v0.1.0
warning: crossval: ✓ BITNET_FULL: BitNet.cpp and llama.cpp libraries found
warning: crossval: Backend: full
warning: crossval: Linked libraries: bitnet, llama, ggml
   Compiling xtask v0.1.0
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 5.34s
[preflight] ✓ Rebuild complete with environment variables

[repair] Step 3/3: Re-executing with updated detection...

✓ bitnet.cpp AVAILABLE (detected after repair)
```

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-10-27 | BitNet.rs Spec Agent | Initial specification based on gap analysis |

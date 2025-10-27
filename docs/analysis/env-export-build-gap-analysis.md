# Environment Export → build.rs → HAS_* Constants Flow Gap Analysis

**Status**: Gap identified and documented  
**Date**: 2025-10-27  
**Scope**: Environment variable propagation pipeline for C++ cross-validation setup

## Executive Summary

The auto-repair workflow has a critical **environment propagation gap**: when `setup-cpp-auto` outputs shell exports (e.g., `export BITNET_CPP_DIR="/path/to/bitnet_cpp"`), these exports are **NOT automatically applied to the child cargo build process**. This causes subsequent `rebuild_xtask()` invocations to inherit the parent process's stale environment, preventing the build.rs detection logic from discovering the newly-installed libraries.

### The Gap

```
setup-cpp-auto          rebuild_xtask()
(subprocess)            (child cargo build)
      |                       |
      v                       v
Installs libs           (env vars NOT inherited)
Outputs shell exports   ───X──→ cargo build doesn't see BITNET_CPP_DIR
      |
      └─ Shell output lost in subprocess
```

## Current Workflow (with Gap)

### 1. setup-cpp-auto Command Output

**Location**: `/home/steven/code/Rust/BitNet-rs/xtask/src/cpp_setup_auto.rs:787-866`

The `emit_exports()` function generates platform-specific environment variable exports:

#### POSIX sh/bash/zsh format (Emit::Sh)
```bash
export BITNET_CPP_DIR="/home/user/.cache/bitnet_cpp"
export BITNET_CROSSVAL_LIBDIR="/home/user/.cache/bitnet_cpp/build/3rdparty/llama.cpp/build/bin"
export LD_LIBRARY_PATH="/home/user/.cache/bitnet_cpp/build/bin:${LD_LIBRARY_PATH:-}"
echo "[bitnet] C++ ready at $BITNET_CPP_DIR"
```

#### Fish shell format (Emit::Fish)
```fish
set -gx BITNET_CPP_DIR "/home/user/.cache/bitnet_cpp"
set -gx BITNET_CROSSVAL_LIBDIR "/home/user/.cache/bitnet_cpp/build/3rdparty/llama.cpp/build/bin"
set -gx LD_LIBRARY_PATH "/home/user/.cache/bitnet_cpp/build/bin" $LD_LIBRARY_PATH
echo "[bitnet] C++ ready at $BITNET_CPP_DIR"
```

#### PowerShell format (Emit::Pwsh)
```powershell
$env:BITNET_CPP_DIR = "/home/user/.cache/bitnet_cpp"
$env:BITNET_CROSSVAL_LIBDIR = "/home/user/.cache/bitnet_cpp/build/3rdparty/llama.cpp/build/bin"
$env:PATH = "/home/user/.cache/bitnet_cpp/build/bin;" + $env:PATH
Write-Host "[bitnet] C++ ready at $env:BITNET_CPP_DIR"
```

**Key Environment Variables Emitted**:
- `BITNET_CPP_DIR`: Root installation directory for BitNet.cpp
- `BITNET_CROSSVAL_LIBDIR`: Library directory (auto-discovered or explicit)
- `LD_LIBRARY_PATH` (Linux) / `DYLD_LIBRARY_PATH` (macOS) / `PATH` (Windows): Dynamic loader search paths

### 2. Current Auto-Repair Flow (Incomplete)

**Location**: `/home/steven/code/Rust/BitNet-rs/xtask/src/crossval/preflight.rs:1326-1423`

Current flow in `preflight_with_auto_repair()`:

1. **Step 1**: Check if backend available (HAS_BITNET/HAS_LLAMA from build-time)
2. **Step 2**: Determine if repair needed (RepairMode logic)
3. **Step 3**: Call `attempt_repair_with_retry()` → runs `setup-cpp-auto --emit=sh`
   - **PROBLEM**: Output is subprocess stdout, NOT inherited by parent process
4. **Step 4**: Call `rebuild_xtask(verbose)` (Line 1404)
   - Runs: `cargo build -p xtask --features crossval-all`
   - **MISSING**: No environment variable application before this call
5. **Step 5**: Re-exec with new binary

**The Gap**:
```rust
// Line 1393-1404 in preflight_with_auto_repair()
if let Err(e) = attempt_repair_with_retry(backend, verbose) {
    // error handling...
}

// ← GAP: setup-cpp-auto output never parsed and applied ←

if let Err(e) = rebuild_xtask(verbose) {
    // rebuild inherits STALE env, doesn't see new BITNET_CPP_DIR
}
```

### 3. build.rs Detection Logic

**Location**: `/home/steven/code/Rust/BitNet-rs/crossval/build.rs:131-189`

The detection happens in `compile_ffi()`:

```rust
// Priority 0: Explicit BITNET_CROSSVAL_LIBDIR override (highest priority)
if let Ok(lib_dir) = env::var("BITNET_CROSSVAL_LIBDIR") {
    possible_lib_dirs.push(Path::new(&lib_dir).to_path_buf());
} else {
    // Priority 1-3: Use BITNET_CPP_DIR to search for libraries
    let bitnet_root = env::var("BITNET_CPP_DIR")
        .unwrap_or_else(|_| {
            format!("{}/.cache/bitnet_cpp", env::var("HOME").unwrap_or_else(|_| ".".into()))
        });
    
    // Search for libraries in tiered locations...
}
```

**Detection Depends On**:
- `BITNET_CROSSVAL_LIBDIR`: Explicit library directory (Priority 0)
- `BITNET_CPP_DIR`: Root directory for library search (Priority 1-3)
- Environment must be present **at cargo build time**

### 4. HAS_BITNET/HAS_LLAMA Constants

**Location**: `/home/steven/code/Rust/BitNet-rs/crossval/build.rs:340-344`

After detection, build.rs emits:

```rust
println!("cargo:rustc-env=CROSSVAL_HAS_BITNET={}", found_bitnet);
println!("cargo:rustc-env=CROSSVAL_HAS_LLAMA={}", found_llama);
println!("cargo:rustc-env=CROSSVAL_BACKEND_STATE={}", backend_state.as_str());
```

These become Rust constants that are checked in `preflight.rs:1340-1341, 1368-1370`.

## The Gap in Detail

### Gap 1: Shell Output Not Captured

**Problem**: `setup-cpp-auto` emits shell commands to stdout, but `attempt_repair_with_retry()` doesn't capture or parse them:

```rust
// Line 1970-1976 in attempt_repair_once()
let setup_result = Command::new(env::current_exe())
    .args(["setup-cpp-auto", "--emit=sh"])
    .output()  // ← Captures stdout/stderr
    .map_err(|e| RepairError::SetupFailed(...))?;

// Check exit status only, never reads setup_result.stdout
if !setup_result.status.success() {
    let stderr = String::from_utf8_lossy(&setup_result.stderr);
    return Err(RepairError::classify(&stderr, backend_name));
}

// ← setup_result.stdout (containing "export BITNET_CPP_DIR=...") is DISCARDED
```

### Gap 2: Environment Variables Not Applied

**Problem**: Even if output were captured, it's not applied to the current process environment before `rebuild_xtask()`:

```rust
// Line 1404 in preflight_with_auto_repair()
if let Err(e) = rebuild_xtask(verbose) {
    // ← rebuild_xtask() spawns a child cargo process
    // ← That child inherits parent's environment (BITNET_CPP_DIR NOT SET)
}
```

The `rebuild_xtask()` function spawns cargo without environment modification:

```rust
// Line 1622-1624 in rebuild_xtask()
let build_status = Command::new("cargo")
    .args(["build", "-p", "xtask", "--features", "crossval-all"])
    .status()  // ← No .env() calls, inherits parent env
    .map_err(|e: std::io::Error| RebuildError::BuildFailed(e.to_string()))?;
```

### Gap 3: No Env Parsing Infrastructure

**Problem**: No parsing function exists to extract environment variables from shell export output.

**Current State**:
- `cpp_setup_auto::run()` produces formatted shell commands
- No `parse_sh_exports()` function to extract key=value pairs
- No `apply_exports()` function to set them in current process

## Target Workflow (Post-Fix)

### Desired Flow

```
setup-cpp-auto                      rebuild_xtask()
(subprocess)                        (child cargo build)
      |                                   |
      v                                   v
Installs libs + outputs exports    Receives env vars
      |                                   |
      ├─ stdout: export BITNET_CPP_DIR=...
      ├─ stdout: export LD_LIBRARY_PATH=...
      |
      └──→ [PARSE]                   ← NEW STEP
           (parse_sh_exports)
           |
           └──→ [APPLY]              ← NEW STEP
                (apply_env_exports)
                |
                └──→ Command::new("cargo")
                     .env("BITNET_CPP_DIR", "...")
                     .env("LD_LIBRARY_PATH", "...")
                     .spawn()
                     |
                     └─→ build.rs detection SUCCEEDS
                         (finds libraries via BITNET_CPP_DIR)
```

## Specific Locations Needing Changes

### Location 1: Parse Shell Exports

**File**: `xtask/src/crossval/preflight.rs` (new function)

**Needed**:
```rust
/// Parse shell export format to HashMap
/// 
/// Parses output from setup-cpp-auto --emit=sh (or other formats)
/// Extracts lines like: export VAR="value"
/// 
/// Returns: HashMap<String, String> of environment variables
fn parse_sh_exports(shell_output: &str) -> HashMap<String, String> {
    // Parse lines like:
    // export BITNET_CPP_DIR="/path"
    // export LD_LIBRARY_PATH="/path:${LD_LIBRARY_PATH:-}"
    // set -gx BITNET_CPP_DIR "/path"  (fish)
    // $env:BITNET_CPP_DIR = "/path"   (pwsh)
}
```

**Reference**: Patterns in `cpp_setup_auto::emit_exports()` lines 795-865

### Location 2: Apply Parsed Exports

**File**: `xtask/src/crossval/preflight.rs` (new function)

**Needed**:
```rust
/// Apply parsed environment variables to current process and child spawns
/// 
/// After parse_sh_exports(), apply exported variables to:
/// 1. Current process (via unsafe env::set_var)
/// 2. All subsequent Command spawns
fn apply_env_exports(exports: &HashMap<String, String>) -> Result<()> {
    // For each key in exports:
    // - unsafe { env::set_var(key, value) }
    // OR create a Vec<(String, String)> for passing to .env() calls
}
```

### Location 3: Integrate into Auto-Repair Flow

**File**: `xtask/src/crossval/preflight.rs:1393-1407`

**Current Code**:
```rust
if let Err(e) = attempt_repair_with_retry(backend, verbose) {
    eprintln!("\n{}", e);
    bail!("Auto-repair failed for backend '{}'", backend.name());
}

// Step 4: Rebuild xtask to pick up new detection (AC3)
if verbose {
    eprintln!("[repair] Step 2/3: Rebuilding xtask binary...");
}

if let Err(e) = rebuild_xtask(verbose) {
    eprintln!("\n{}", e);
    bail!("xtask rebuild failed after successful C++ setup");
}
```

**Needed Change**:
```rust
// Capture setup-cpp-auto output (modify attempt_repair_with_retry to return it)
let setup_output = attempt_repair_with_retry(backend, verbose)?;

// Parse environment exports
let exports = parse_sh_exports(&setup_output)?;

// Apply to current process
apply_env_exports(&exports)?;

// Step 4: Rebuild with applied environment
if verbose {
    eprintln!("[repair] Step 2/3: Rebuilding xtask binary...");
}

if let Err(e) = rebuild_xtask_with_env(verbose, &exports) {
    eprintln!("\n{}", e);
    bail!("xtask rebuild failed");
}
```

### Location 4: Modify rebuild_xtask Signature

**File**: `xtask/src/crossval/preflight.rs:1617`

**Current**:
```rust
fn rebuild_xtask(verbose: bool) -> Result<(), RebuildError> {
    let build_status = Command::new("cargo")
        .args(["build", "-p", "xtask", "--features", "crossval-all"])
        .status()?;
    // ...
}
```

**Needed**:
```rust
fn rebuild_xtask_with_env(
    verbose: bool,
    exports: &HashMap<String, String>,
) -> Result<(), RebuildError> {
    let mut cmd = Command::new("cargo");
    cmd.args(["build", "-p", "xtask", "--features", "crossval-all"]);
    
    // Apply environment variables
    for (key, value) in exports {
        cmd.env(key, value);
    }
    
    let build_status = cmd.status()?;
    // ...
}
```

## Implementation Plan

### Phase 1: Create Parsing Infrastructure

1. **Create `parse_sh_exports()` function**
   - Input: Raw shell output from setup-cpp-auto
   - Output: HashMap<String, String>
   - Handle all shell formats (sh, fish, pwsh, cmd)
   - Handle variable references (e.g., `${LD_LIBRARY_PATH:-}`)

2. **Create `apply_env_exports()` function**
   - Input: HashMap from parse_sh_exports
   - Effects: Set environment variables in current process
   - Returns: Env var collection for passing to child processes

### Phase 2: Integrate into Repair Flow

1. **Modify `attempt_repair_with_retry()`**
   - Return captured stdout from setup-cpp-auto
   - Change return type to `Result<String, RepairError>`

2. **Modify `preflight_with_auto_repair()`**
   - Capture output from attempt_repair_with_retry
   - Parse and apply exports
   - Pass exports to rebuild_xtask

3. **Create `rebuild_xtask_with_env()`**
   - Accept exports HashMap
   - Apply to child cargo process

### Phase 3: Testing

1. **Unit tests** for parsing functions
   - Test each shell format
   - Test variable references
   - Test edge cases (empty values, special chars)

2. **Integration tests**
   - Verify env vars reach child process
   - Verify build.rs detection succeeds
   - Verify HAS_BITNET/HAS_LLAMA updated

3. **End-to-end test**
   - Full auto-repair workflow
   - Verify rebuilt binary has new constants

## Validation Criteria

### AC1: Parse Shell Exports
- [ ] Correctly parses sh format: `export VAR="value"`
- [ ] Correctly parses fish format: `set -gx VAR "value"`
- [ ] Correctly parses PowerShell format: `$env:VAR = "value"`
- [ ] Handles variable references: `${LD_LIBRARY_PATH:-}`
- [ ] Preserves value escaping

### AC2: Apply Environment Variables
- [ ] Sets variables in current process
- [ ] Makes variables available to child Command processes
- [ ] Handles path-like variables (e.g., LD_LIBRARY_PATH)
- [ ] Doesn't crash on invalid env values

### AC3: Integration with Repair
- [ ] setup-cpp-auto output captured correctly
- [ ] Exports applied before rebuild_xtask call
- [ ] Child cargo process receives BITNET_CPP_DIR
- [ ] build.rs detection succeeds with applied env

### AC4: PERSIST No BITNET_STUB Mode
- [ ] After repair + rebuild + re-exec, HAS_BITNET = true
- [ ] No "BITNET_STUB mode" warning after repair succeeds
- [ ] Preflight succeeds on second check

## Example Parse/Apply Output

### Input: setup-cpp-auto stdout
```
export BITNET_CPP_DIR="/home/user/.cache/bitnet_cpp"
export BITNET_CROSSVAL_LIBDIR="/home/user/.cache/bitnet_cpp/build/3rdparty/llama.cpp/build/bin"
export LD_LIBRARY_PATH="/home/user/.cache/bitnet_cpp/build/bin:${LD_LIBRARY_PATH:-}"
echo "[bitnet] C++ ready at $BITNET_CPP_DIR"
```

### After parse_sh_exports():
```rust
{
    "BITNET_CPP_DIR": "/home/user/.cache/bitnet_cpp",
    "BITNET_CROSSVAL_LIBDIR": "/home/user/.cache/bitnet_cpp/build/3rdparty/llama.cpp/build/bin",
    "LD_LIBRARY_PATH": "/home/user/.cache/bitnet_cpp/build/bin:${LD_LIBRARY_PATH:-}",
}
```

### After apply_env_exports() in rebuild_xtask_with_env:
```rust
let mut cmd = Command::new("cargo");
cmd.args(["build", "-p", "xtask", "--features", "crossval-all"]);
cmd.env("BITNET_CPP_DIR", "/home/user/.cache/bitnet_cpp");
cmd.env("BITNET_CROSSVAL_LIBDIR", "/home/user/.cache/bitnet_cpp/build/3rdparty/llama.cpp/build/bin");
cmd.env("LD_LIBRARY_PATH", "/home/user/.cache/bitnet_cpp/build/bin:${LD_LIBRARY_PATH:-}");
cmd.status()?;
```

## Related Files

- **Setup Function**: `/home/steven/code/Rust/BitNet-rs/xtask/src/cpp_setup_auto.rs:707-866`
- **Repair Flow**: `/home/steven/code/Rust/BitNet-rs/xtask/src/crossval/preflight.rs:1326-1936`
- **Build Detection**: `/home/steven/code/Rust/BitNet-rs/crossval/build.rs:131-189`
- **Tests**: `/home/steven/code/Rust/BitNet-rs/xtask/tests/preflight_auto_repair_tests.rs`

## Risk Assessment

### Low Risk (Parsing/Applying Exports)
- Isolated to new functions
- No changes to existing critical logic
- Reversible if issues found

### Medium Risk (Integration)
- Changes flow of attempt_repair_with_retry return value
- Affects auto-repair codepath (but only when used)
- Well-scoped to preflight_with_auto_repair

### Mitigation
- Comprehensive unit tests for parsing
- Integration tests for environment propagation
- Keep rebuild_xtask() as fallback (non-env version)
- Add verbose logging of env vars passed to cargo

## Deployment Notes

1. **Backward Compatibility**: Keep rebuild_xtask() callable without env
2. **Error Handling**: Graceful fallback if parsing fails
3. **User Communication**: Update docs with new workflow
4. **CI/CD**: Add tests to verify env propagation in automated repairs

## References

- **setup-cpp-auto**: `/home/steven/code/Rust/BitNet-rs/xtask/src/cpp_setup_auto.rs`
- **Emit Formats**: Lines 795-865 (sh, fish, pwsh, cmd)
- **Build Detection**: `/home/steven/code/Rust/BitNet-rs/crossval/build.rs:131-189`
- **Auto-Repair Flow**: `/home/steven/code/Rust/BitNet-rs/xtask/src/crossval/preflight.rs:1326-1936`

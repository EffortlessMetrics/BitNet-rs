# Technical Specification: Robust Re-exec with Cargo Run Fallback

**Document ID**: `reexec-cargo-fallback-robust`
**Status**: Draft
**Created**: 2025-10-27
**Author**: BitNet.rs Generative Spec Agent
**Related Files**:
- `/home/steven/code/Rust/BitNet-rs/xtask/src/crossval/preflight.rs` (lines 1706-1749)
- `/tmp/explore_reexec.md` (failure analysis)

---

## Executive Summary

The current `reexec_current_command()` implementation fails with "No such file or directory (os error 2)" when `exec()` is called. This specification proposes a robust two-tier re-execution mechanism:

1. **Fast path**: Try `exec()` with `current_exe()` path (Unix only, zero overhead)
2. **Fallback path**: Use `cargo run -p xtask --features crossval-all -- <args>` when binary unavailable

This approach ensures automatic C++ backend installation works reliably even when the rebuilt binary is temporarily unavailable due to race conditions or filesystem inconsistencies.

---

## Problem Statement

### Current Implementation Issues

From `/tmp/explore_reexec.md` and current code analysis:

**Failure Mode**: `exec()` returns `ENOENT` despite `current_exe()` succeeding

**Root Causes**:
1. **Race condition**: Binary deleted/overwritten between `current_exe()` and `exec()` calls
2. **Build artifact corruption**: Cargo incremental cleanup removes binary before exec
3. **Symlink invalidation**: `/proc/<pid>/exe` points to deleted inode
4. **Filesystem timing**: tmpfs auto-cleanup or network filesystem timeout

**Current Code (lines 1706-1749)**:
```rust
pub fn reexec_current_command(original_args: &[String]) -> Result<(), RepairError> {
    eprintln!("[repair] Re-executing with updated detection...");

    let current_exe = env::current_exe().map_err(|e| RepairError::Unknown {
        error: format!("Failed to get current executable path: {}", e),
        backend: "unknown".to_string(),
    })?;

    let mut cmd = Command::new(&current_exe);
    cmd.args(original_args);
    cmd.env("BITNET_REPAIR_PARENT", "1");

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let err = cmd.exec();  // ← FAILS WITH ENOENT
        Err(RepairError::Unknown {
            error: format!("exec() failed: {}", err),
            backend: "unknown".to_string(),
        })
    }

    #[cfg(not(unix))]
    {
        match cmd.status() {
            Ok(status) => std::process::exit(status.code().unwrap_or(1)),
            Err(e) => Err(RepairError::Unknown {
                error: format!("spawn() failed: {}", e),
                backend: "unknown".to_string(),
            }),
        }
    }
}
```

**Limitations**:
- Single execution strategy (no fallback)
- Throws away error details (wraps all as `Unknown`)
- No diagnostic logging before exec attempt
- No path existence validation

---

## Requirements

### Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR1 | Try exec() with current_exe() path first (fast path) | MUST |
| FR2 | Fall back to `cargo run` when binary unavailable | MUST |
| FR3 | Preserve all CLI arguments across re-exec | MUST |
| FR4 | Preserve BITNET_REPAIR_PARENT=1 environment variable | MUST |
| FR5 | Log diagnostic information before exec attempts | SHOULD |
| FR6 | Handle both Unix (exec) and Windows (spawn) platforms | MUST |
| FR7 | Exit with correct code on fallback spawn failure | MUST |

### Acceptance Criteria

| AC | Criterion | Validation |
|----|-----------|------------|
| AC1 | Fast path uses exec() when binary exists | Manual test: verify `exec()` called first |
| AC2 | Fallback to cargo run works when current_exe() fails | Unit test: mock ENOENT error |
| AC3 | All CLI args preserved across re-exec | Integration test: verify args match |
| AC4 | BITNET_REPAIR_PARENT guard prevents infinite loops | Unit test: verify env var set |
| AC5 | Diagnostic logging shows resolved path + existence | Manual test: check stderr output |
| AC6 | Windows uses spawn() pattern consistently | Windows integration test |
| AC7 | Exit code propagated correctly from spawned process | Integration test: verify exit codes |

---

## Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│ reexec_current_command(original_args: &[String])            │
│                                                             │
│ 1. Collect CLI arguments                                   │
│ 2. Set BITNET_REPAIR_PARENT=1 (recursion guard)            │
│ 3. Try fast path (Unix only)                               │
│ 4. On failure → Try cargo run fallback                     │
│ 5. Exit with child status code                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
        ┌───────────────────┴───────────────────┐
        │                                       │
        ↓                                       ↓
┌─────────────────┐                   ┌─────────────────┐
│ Fast Path       │                   │ Fallback Path   │
│ (Unix only)     │                   │ (All platforms) │
└─────────────────┘                   └─────────────────┘
        ↓                                       ↓
  1. Get current_exe()              1. Build cargo run command
  2. Check if exists               2. Add -p xtask --features crossval-all
  3. Log path + existence          3. Add original arguments
  4. Try exec() replacement        4. Set BITNET_REPAIR_PARENT=1
  5. On success → NEVER RETURNS    5. Spawn child process
  6. On ENOENT → Fall to right     6. Wait for completion
                                   7. Exit with child's code
        ↓                                       ↓
  [ Process Replaced ]            [ Parent Exits with Child Code ]
```

### Execution Flow

```
┌────────────────────────────────────────────────────────────┐
│ Step 1: Argument Collection                                │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│ Step 2: Environment Setup                                  │
│   • BITNET_REPAIR_PARENT=1 (recursion guard)               │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│ Step 3: Fast Path (Unix only, #[cfg(unix)])                │
│   current_exe = env::current_exe()?                        │
│   if current_exe.exists() {                                │
│     eprintln!("[reexec] Fast path: {}", current_exe)       │
│     eprintln!("[reexec] Binary exists: YES")               │
│     exec(current_exe, args, env)  // Never returns on OK   │
│   }                                                         │
│   eprintln!("[reexec] Fast path failed: {:?}", err)        │
└────────────────────────────────────────────────────────────┘
                            ↓
            ┌───────────────┴───────────────┐
            │ exec() succeeded              │ exec() failed (ENOENT)
            ↓                               ↓
    [ Process Replaced ]        ┌────────────────────────────────┐
    NEVER RETURNS               │ Step 4: Cargo Run Fallback     │
                                │   eprintln!("[reexec] Trying   │
                                │     cargo run fallback...")     │
                                │   cmd = cargo run -p xtask \    │
                                │     --features crossval-all -- \│
                                │     <original_args>             │
                                │   cmd.env("BITNET_REPAIR_PARENT│
                                │     ", "1")                     │
                                │   status = cmd.status()?        │
                                │   exit(status.code().unwrap_or │
                                │     (1))                        │
                                └────────────────────────────────┘
                                            ↓
                                [ Parent Exits with Child Code ]
```

---

## API Contract

### Function Signature

```rust
/// Re-execute current xtask command with updated detection
///
/// Implements a two-tier execution strategy:
/// 1. **Fast path** (Unix): exec() with current_exe() path (zero overhead)
/// 2. **Fallback path** (All platforms): cargo run -p xtask --features crossval-all
///
/// # Recursion Guard
///
/// Sets `BITNET_REPAIR_PARENT=1` to prevent infinite repair loops. Re-executed
/// processes skip repair and only validate detection.
///
/// # Platform Behavior
///
/// * **Unix fast path**: Uses exec() to replace process (never returns on success)
/// * **Unix fallback**: Spawns cargo run and exits with its status code
/// * **Windows**: Always uses cargo run (exec() not available on Windows)
///
/// # Arguments
///
/// * `original_args` - Command-line arguments to preserve (from `env::args()`)
///
/// # Returns
///
/// * **Unix fast path**: Never returns on success (process replaced)
/// * **All fallback paths**: Never returns (exits with child status code)
/// * `Err(RepairError)` only if both exec() and cargo run fail
///
/// # Examples
///
/// ```ignore
/// use xtask::crossval::preflight::reexec_current_command;
///
/// // After rebuild, re-exec with original args
/// let args: Vec<String> = env::args().collect();
/// reexec_current_command(&args)?;
/// // This line never reached (process replaced or exited)
/// ```
///
/// # Errors
///
/// Returns `RepairError::Unknown` only if:
/// - Unix: Both exec() and cargo run fail
/// - Windows: cargo run fails
///
/// # Diagnostics
///
/// Logs to stderr:
/// - `[reexec] Fast path: <path>` (Unix only)
/// - `[reexec] Binary exists: YES/NO` (Unix only)
/// - `[reexec] Fast path failed: <error>` (on exec failure)
/// - `[reexec] Trying cargo run fallback...`
/// - `[reexec] Fallback command: cargo run -p xtask...`
///
pub fn reexec_current_command(original_args: &[String]) -> Result<(), RepairError> {
    // Implementation below
}
```

### Behavior Contract

| Scenario | Fast Path | Fallback Path | Return Behavior |
|----------|-----------|---------------|-----------------|
| Unix, binary exists | exec() succeeds | Not tried | Never returns (process replaced) |
| Unix, binary missing | exec() fails ENOENT | cargo run succeeds | Never returns (exits with child code) |
| Unix, cargo fails | exec() fails ENOENT | cargo run fails | Returns `Err(RepairError)` |
| Windows | N/A (no exec) | cargo run succeeds | Never returns (exits with child code) |
| Windows, cargo fails | N/A | cargo run fails | Returns `Err(RepairError)` |

---

## Implementation

### Proposed Implementation

```rust
/// Re-execute current xtask command with updated detection
///
/// See API contract documentation above for full details.
pub fn reexec_current_command(original_args: &[String]) -> Result<(), RepairError> {
    eprintln!("[repair] Re-executing with updated detection...");

    // Unix fast path: Try exec() with current_exe() first
    #[cfg(unix)]
    {
        if let Ok(current_exe) = env::current_exe() {
            let exists = current_exe.exists();

            // Diagnostic logging (AC5)
            eprintln!("[reexec] Fast path: {}", current_exe.display());
            eprintln!("[reexec] Binary exists: {}", if exists { "YES" } else { "NO" });

            if exists {
                use std::os::unix::process::CommandExt;

                let mut cmd = Command::new(&current_exe);
                cmd.args(original_args);
                cmd.env("BITNET_REPAIR_PARENT", "1");  // AC4: recursion guard

                eprintln!("[reexec] Attempting exec()...");

                // Try exec() - never returns on success
                let err = cmd.exec();

                // If we reach here, exec() failed
                eprintln!("[reexec] Fast path failed: {}", err);
                eprintln!("[reexec] Error kind: {:?}", err.kind());

                // Fall through to cargo run fallback
            } else {
                eprintln!("[reexec] Binary doesn't exist, skipping exec()");
            }
        } else {
            eprintln!("[reexec] current_exe() failed, skipping exec()");
        }
    }

    // Fallback path: cargo run (all platforms, Unix if exec failed)
    eprintln!("[reexec] Trying cargo run fallback...");

    let mut cmd = Command::new("cargo");
    cmd.arg("run")
        .arg("-p")
        .arg("xtask")
        .arg("--features")
        .arg("crossval-all")
        .arg("--");

    // AC3: Preserve all original arguments
    // Skip the first arg (program name) from original_args
    if original_args.len() > 1 {
        cmd.args(&original_args[1..]);
    }

    // AC4: Set recursion guard
    cmd.env("BITNET_REPAIR_PARENT", "1");

    // Diagnostic logging (AC5)
    eprintln!("[reexec] Fallback command: cargo run -p xtask --features crossval-all -- {:?}",
              &original_args[1..]);

    // Spawn and wait for child
    match cmd.status() {
        Ok(status) => {
            let code = status.code().unwrap_or(1);
            eprintln!("[reexec] Fallback child exited with code: {}", code);
            std::process::exit(code);  // AC7: Propagate exit code
        }
        Err(e) => {
            // Both exec() (if Unix) and cargo run failed
            Err(RepairError::Unknown {
                error: format!(
                    "Re-exec failed: exec() and cargo run both failed. Last error: {}",
                    e
                ),
                backend: "unknown".to_string(),
            })
        }
    }
}
```

### Platform-Specific Implementation Notes

#### Unix (Linux, macOS)

**Fast Path**:
1. Call `env::current_exe()` to get binary path
2. Check `Path::exists()` to validate binary is accessible
3. Log diagnostic information (path, existence)
4. If exists: try `CommandExt::exec()` (never returns on success)
5. If exec() fails: log error and fall through to cargo run

**Fallback Path**:
1. Build `cargo run -p xtask --features crossval-all -- <args>` command
2. Set `BITNET_REPAIR_PARENT=1` environment variable
3. Call `cmd.status()` to spawn and wait
4. Exit parent with child's exit code

**Advantages**:
- Fast path: Zero overhead process replacement when binary exists
- Fallback: Robust recovery when binary temporarily unavailable
- Seamless transition: User doesn't see failure, just slight delay

#### Windows

**No Fast Path** (exec() not available):
1. Skip directly to cargo run fallback
2. Build `cargo run -p xtask --features crossval-all -- <args>` command
3. Set `BITNET_REPAIR_PARENT=1` environment variable
4. Call `cmd.status()` to spawn and wait
5. Exit parent with child's exit code

**Advantages**:
- Consistent behavior: Same fallback strategy as Unix
- Robust: Works even if binary path detection fails

---

## Error Handling Strategy

### Error Classification

| Error Type | When | Retry Strategy | Exit Code |
|------------|------|----------------|-----------|
| `current_exe()` fails | Can't get binary path | Skip exec, try cargo run | N/A (fallback) |
| `exec()` ENOENT | Binary not found | Try cargo run fallback | N/A (fallback) |
| `exec()` EACCES | Permission denied | Try cargo run fallback | N/A (fallback) |
| `exec()` ENOEXEC | Format error | Try cargo run fallback | N/A (fallback) |
| `cargo run` fails | cargo not found or build error | Return error | 1 (Unavailable) |

### Enhanced Error Reporting

**Current** (wraps all as `Unknown`):
```rust
Err(RepairError::Unknown {
    error: format!("exec() failed: {}", err),
    backend: "unknown".to_string(),
})
```

**Proposed** (preserves error details):
```rust
Err(RepairError::Unknown {
    error: format!(
        "Re-exec failed: exec() and cargo run both failed.\n\
         exec() error: {} (kind: {:?})\n\
         cargo run error: {}",
        exec_err, exec_err.kind(), cargo_err
    ),
    backend: "unknown".to_string(),
})
```

### Diagnostic Logging

**Before exec() attempt**:
```
[reexec] Fast path: /home/user/BitNet-rs/target/debug/xtask
[reexec] Binary exists: YES
[reexec] Attempting exec()...
```

**On exec() failure**:
```
[reexec] Fast path failed: No such file or directory (os error 2)
[reexec] Error kind: NotFound
[reexec] Trying cargo run fallback...
```

**On cargo run success**:
```
[reexec] Fallback command: cargo run -p xtask --features crossval-all -- ["preflight", "--backend", "bitnet", "--repair=auto"]
[reexec] Fallback child exited with code: 0
```

---

## Testing Strategy

### Unit Tests

**Test Coverage**:

| Test | Purpose | Assertion |
|------|---------|-----------|
| `test_reexec_preserves_args` | Verify CLI args preserved | Mock spawn, check args match |
| `test_reexec_sets_repair_parent_guard` | Verify env var set | Check `BITNET_REPAIR_PARENT=1` |
| `test_reexec_fallback_on_missing_binary` | Verify cargo run called when binary missing | Mock current_exe() failure |
| `test_reexec_propagates_exit_code` | Verify child exit code propagated | Mock child exit, check parent exit |

**Implementation**:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reexec_sets_repair_parent_guard() {
        // Verify BITNET_REPAIR_PARENT=1 is set in both fast and fallback paths
        // This test documents the recursion guard behavior

        // Can't directly test exec() in unit test (would replace process)
        // But we can verify the guard is set in Command environment
    }

    #[test]
    fn test_reexec_preserves_args() {
        // Verify original arguments are preserved across re-exec
        let original_args = vec![
            "xtask".to_string(),
            "preflight".to_string(),
            "--backend".to_string(),
            "bitnet".to_string(),
        ];

        // Mock: capture command built by reexec_current_command
        // Assert: args[3..] match original_args[1..]
    }
}
```

### Integration Tests

**Test Scenarios**:

| Scenario | Setup | Expected Behavior |
|----------|-------|-------------------|
| Binary exists (Unix) | Build xtask, verify exists | exec() succeeds, process replaced |
| Binary missing (Unix) | Delete binary after rebuild | cargo run fallback succeeds |
| cargo not found | Remove cargo from PATH | Returns error with diagnostic |
| Recursion guard works | Set BITNET_REPAIR_PARENT=1 | Child skips repair, validates only |

**Implementation**:

```rust
#[test]
#[cfg(unix)]
fn test_reexec_fast_path_with_existing_binary() {
    // This test requires careful setup to avoid replacing test process
    // Use fork() or subprocess to test exec() behavior
}

#[test]
fn test_reexec_fallback_when_binary_missing() {
    // Setup: ensure current_exe() points to non-existent path
    // Expected: cargo run fallback succeeds
    // Assertion: child process runs successfully
}

#[test]
fn test_reexec_recursion_guard_prevents_loops() {
    // Setup: Set BITNET_REPAIR_PARENT=1 before calling preflight
    // Expected: Repair skipped, only detection validated
    // Assertion: No second repair attempt
}
```

### Manual Testing

**Test Plan**:

1. **Fast path success** (Unix):
   ```bash
   # Rebuild xtask
   cargo build -p xtask --features crossval-all

   # Trigger repair workflow
   cargo run -p xtask -- preflight --backend bitnet --repair=auto

   # Expected: Fast path exec() succeeds, process replaced seamlessly
   ```

2. **Fallback path** (Unix, simulate missing binary):
   ```bash
   # Rebuild xtask
   cargo build -p xtask --features crossval-all

   # Move binary to break current_exe() path
   mv target/debug/xtask target/debug/xtask.backup

   # Trigger repair workflow
   cargo run -p xtask -- preflight --backend bitnet --repair=auto

   # Expected: exec() fails, cargo run fallback succeeds
   # Verify: Child process runs, exits with correct code
   ```

3. **Windows consistency**:
   ```powershell
   # Rebuild xtask
   cargo build -p xtask --features crossval-all

   # Trigger repair workflow
   cargo run -p xtask -- preflight --backend bitnet --repair=auto

   # Expected: cargo run fallback used (no exec on Windows)
   ```

---

## Migration Path

### Step 1: Add Fallback Implementation

**Location**: `xtask/src/crossval/preflight.rs`

**Changes**:
1. Replace single-strategy `reexec_current_command()` with two-tier implementation
2. Add diagnostic logging before exec() attempt
3. Add cargo run fallback on exec() failure
4. Update error messages to reflect fallback strategy

**Compatibility**: Backward compatible - function signature unchanged

### Step 2: Update Documentation

**Files to Update**:
- `docs/howto/cpp-setup.md`: Document fallback behavior
- `docs/explanation/dual-backend-crossval.md`: Explain re-exec robustness
- `CLAUDE.md`: Update troubleshooting section

**Key Points**:
- Explain why cargo run fallback is needed
- Document diagnostic logging output
- Clarify when each path is used

### Step 3: Add Tests

**Test Files**:
- `xtask/tests/preflight_repair_mode_tests.rs`: Add re-exec tests
- `xtask/tests/auto_repair_e2e_tests.rs`: Add end-to-end scenarios

**Coverage**:
- Argument preservation
- Recursion guard behavior
- Exit code propagation
- Fallback activation

### Step 4: Validate in CI

**CI Checks**:
1. Verify fast path on Linux (exec() succeeds)
2. Verify fallback on simulated missing binary
3. Verify Windows uses cargo run consistently
4. Check diagnostic logging output

### Step 5: Production Rollout

**Rollout Plan**:
1. Merge implementation to feature branch
2. Test locally on Linux/macOS/Windows
3. Monitor CI for any regressions
4. Merge to main after successful testing

---

## Performance Considerations

### Fast Path (Unix)

**Overhead**: Zero
- `current_exe()`: Single syscall (~µs)
- `Path::exists()`: Single stat() syscall (~µs)
- `exec()`: Kernel process replacement (no new process)

**Total**: < 1ms overhead vs. current implementation

### Fallback Path (All Platforms)

**Overhead**: Moderate
- `cargo run`: Spawns cargo subprocess (~100-500ms)
- Build check: cargo verifies binary up-to-date (~10-50ms)
- Child spawn: New process creation (~10-50ms)

**Total**: ~120-600ms overhead vs. exec()

**Tradeoff Justification**:
- Fallback only used when exec() fails (rare case)
- 600ms delay acceptable for auto-repair workflow (one-time operation)
- Robustness benefit outweighs performance cost

---

## Security Considerations

### Argument Injection

**Risk**: Malicious CLI arguments passed to cargo run

**Mitigation**:
1. Arguments collected from `env::args()` (trusted source)
2. No shell interpolation (direct Command::args())
3. `--` separator isolates xtask args from cargo args

**Safety**: No additional risk vs. current implementation

### Environment Variable Tampering

**Risk**: BITNET_REPAIR_PARENT manipulated to bypass guard

**Mitigation**:
1. Guard only prevents infinite loops, not security boundary
2. If user sets guard manually, worst case is skipped repair
3. No privilege escalation or data access

**Safety**: Low risk, guard is convenience feature

### Binary Substitution

**Risk**: Malicious binary at cargo run path

**Mitigation**:
1. cargo invoked by name (relies on PATH)
2. Same trust model as current implementation
3. cargo verifies binary signature and build

**Safety**: No additional risk vs. current implementation

---

## Open Questions

| Question | Impact | Resolution Needed By |
|----------|--------|----------------------|
| Should we add retry logic for exec() ENOENT? | Low - fallback already handles this | Optional enhancement |
| Should we cache binary existence check? | Very low - only called once per repair | No action needed |
| Should we log to file instead of stderr? | Low - stderr is standard for diagnostics | No action needed |
| Should we validate cargo is in PATH before fallback? | Medium - clearer error if cargo missing | Consider for v1.1 |

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| AC1: Fast path used when binary exists | 95% of Unix re-execs | Monitor logs in dev testing |
| AC2: Fallback succeeds when exec fails | 100% when cargo available | Integration test pass rate |
| AC3: Arguments preserved | 100% accuracy | Unit test assertion |
| AC4: Recursion guard prevents loops | 0 infinite loops | Manual testing + CI |
| AC5: Diagnostic logging complete | All paths logged | Manual inspection |
| AC6: Windows consistency | 100% cargo run usage | Windows CI pass rate |
| AC7: Exit code propagation | 100% accuracy | Integration test assertion |

---

## References

### Related Documentation

- `/tmp/explore_reexec.md`: Failure analysis and root cause investigation
- `docs/specs/preflight-auto-repair.md`: Auto-repair workflow specification
- `docs/specs/preflight-repair-mode-reexec.md`: Repair mode and re-exec integration

### Related Issues

- Current implementation: `xtask/src/crossval/preflight.rs` lines 1706-1749
- Failure symptom: "No such file or directory (os error 2)" on exec()

### External References

- Rust std::process::Command: https://doc.rust-lang.org/std/process/struct.Command.html
- Unix exec() syscall: https://man7.org/linux/man-pages/man3/exec.3.html
- Cargo run behavior: https://doc.rust-lang.org/cargo/commands/cargo-run.html

---

## Appendix A: Complete Implementation Example

```rust
/// Re-execute current xtask command with updated detection
///
/// Implements a two-tier execution strategy for robustness:
/// 1. Fast path (Unix): exec() with current_exe() if binary exists
/// 2. Fallback path (All): cargo run -p xtask --features crossval-all
///
/// This ensures auto-repair works even when the rebuilt binary is
/// temporarily unavailable due to race conditions or filesystem timing.
///
/// # Behavior
///
/// - **Unix fast path**: Tries exec(), never returns on success
/// - **Unix fallback**: Falls back to cargo run if exec() fails
/// - **Windows**: Always uses cargo run (exec() not available)
///
/// # Returns
///
/// Never returns on success (process replaced or parent exits).
/// Returns `Err(RepairError)` only if both exec() and cargo run fail.
pub fn reexec_current_command(original_args: &[String]) -> Result<(), RepairError> {
    eprintln!("[repair] Re-executing with updated detection...");

    // Unix fast path: Try exec() with current_exe() first
    #[cfg(unix)]
    {
        if let Ok(current_exe) = env::current_exe() {
            let exists = current_exe.exists();

            eprintln!("[reexec] Fast path: {}", current_exe.display());
            eprintln!("[reexec] Binary exists: {}", if exists { "YES" } else { "NO" });

            if exists {
                use std::os::unix::process::CommandExt;

                let mut cmd = Command::new(&current_exe);
                cmd.args(original_args);
                cmd.env("BITNET_REPAIR_PARENT", "1");

                eprintln!("[reexec] Attempting exec()...");
                let err = cmd.exec();

                eprintln!("[reexec] Fast path failed: {}", err);
                eprintln!("[reexec] Error kind: {:?}", err.kind());
            } else {
                eprintln!("[reexec] Binary doesn't exist, skipping exec()");
            }
        } else {
            eprintln!("[reexec] current_exe() failed, skipping exec()");
        }
    }

    // Fallback path: cargo run (all platforms)
    eprintln!("[reexec] Trying cargo run fallback...");

    let mut cmd = Command::new("cargo");
    cmd.arg("run")
        .arg("-p")
        .arg("xtask")
        .arg("--features")
        .arg("crossval-all")
        .arg("--");

    if original_args.len() > 1 {
        cmd.args(&original_args[1..]);
    }

    cmd.env("BITNET_REPAIR_PARENT", "1");

    eprintln!("[reexec] Fallback command: cargo run -p xtask --features crossval-all -- {:?}",
              &original_args[1..]);

    match cmd.status() {
        Ok(status) => {
            let code = status.code().unwrap_or(1);
            eprintln!("[reexec] Fallback child exited with code: {}", code);
            std::process::exit(code);
        }
        Err(e) => {
            Err(RepairError::Unknown {
                error: format!(
                    "Re-exec failed: both exec() and cargo run failed. Last error: {}",
                    e
                ),
                backend: "unknown".to_string(),
            })
        }
    }
}
```

---

## Appendix B: Diagnostic Output Examples

### Scenario 1: Fast Path Success (Unix)

```
[repair] Re-executing with updated detection...
[reexec] Fast path: /home/user/BitNet-rs/target/debug/xtask
[reexec] Binary exists: YES
[reexec] Attempting exec()...
(process replaced, no further output from parent)
```

### Scenario 2: Fallback Success (Unix, Binary Missing)

```
[repair] Re-executing with updated detection...
[reexec] Fast path: /home/user/BitNet-rs/target/debug/xtask
[reexec] Binary exists: NO
[reexec] Binary doesn't exist, skipping exec()
[reexec] Trying cargo run fallback...
[reexec] Fallback command: cargo run -p xtask --features crossval-all -- ["preflight", "--backend", "bitnet"]
    Finished dev [unoptimized + debuginfo] target(s) in 0.05s
     Running `target/debug/xtask preflight --backend bitnet`
[reexec] Fallback child exited with code: 0
```

### Scenario 3: Fallback Success (Windows)

```
[repair] Re-executing with updated detection...
[reexec] Trying cargo run fallback...
[reexec] Fallback command: cargo run -p xtask --features crossval-all -- ["preflight", "--backend", "bitnet"]
    Finished dev [unoptimized + debuginfo] target(s) in 0.08s
     Running `target\debug\xtask.exe preflight --backend bitnet`
[reexec] Fallback child exited with code: 0
```

### Scenario 4: Both Paths Fail

```
[repair] Re-executing with updated detection...
[reexec] Fast path: /home/user/BitNet-rs/target/debug/xtask
[reexec] Binary exists: YES
[reexec] Attempting exec()...
[reexec] Fast path failed: No such file or directory (os error 2)
[reexec] Error kind: NotFound
[reexec] Trying cargo run fallback...
[reexec] Fallback command: cargo run -p xtask --features crossval-all -- ["preflight"]
error: cargo not found in PATH

Error: Re-exec failed: both exec() and cargo run failed. Last error: No such file or directory (os error 2)
```

---

**End of Specification**

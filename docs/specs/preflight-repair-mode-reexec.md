# Preflight RepairMode with Rebuild and Re-Exec Specification

**Version**: 1.0
**Status**: Draft
**Author**: BitNet.rs Team
**Date**: 2025-10-26
**Target Release**: v0.2.0

---

## Executive Summary

### Current State: Manual Setup Burden

BitNet.rs cross-validation currently requires **manual multi-step setup** when C++ backend libraries are not detected at build time:

```bash
# Current workflow (5+ commands)
1. User: cargo run -p xtask -- preflight --backend bitnet
   Output: "❌ Backend unavailable"

2. User: cargo run -p xtask -- setup-cpp-auto --emit=sh
   Output: shell exports (user must copy-paste)

3. User: eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"
   (installs C++ reference to ~/.cache/bitnet_cpp)

4. User: cargo clean -p xtask && cargo build -p xtask --features crossval-all
   (rebuilds xtask to pick up new build-time constants)

5. User: cargo run -p xtask -- preflight --backend bitnet
   Output: "✓ Backend available"
```

**Why This is Necessary**: Build-time constants (`HAS_BITNET`, `HAS_LLAMA`) are baked into the xtask binary during compilation. After `setup-cpp-auto` installs libraries, the **current process still has old constants** and cannot detect newly installed libraries without a rebuild + re-execution cycle.

**Pain Points**:
- 5+ command workflow discourages cross-validation testing
- Users must understand build-time vs runtime detection
- Error-prone (forgetting to rebuild, copy-paste errors)
- CI/CD requires pre-provisioning or complex bootstrap scripts

### Desired State: One-Command Auto-Repair

Enable **automatic end-to-end repair** with a single command:

```bash
# Desired workflow (1 command, 60-120 seconds)
cargo run -p xtask -- preflight --backend bitnet --repair=auto
```

**Behind the scenes** (AC1-AC14):
1. Detect backend missing (build-time constants: `HAS_BITNET=false`)
2. Invoke `setup-cpp-auto` subprocess (clone + build C++ reference)
3. Rebuild `xtask` binary in workspace (`cargo build -p xtask --features crossval-all`)
4. Re-execute new `xtask` binary with original arguments
5. New binary detects libraries (`HAS_BITNET=true`)
6. Exit 0 (success), user proceeds with cross-validation

**User Experience**:
```
$ cargo run -p xtask -- preflight --backend bitnet --repair=auto
[preflight] Backend not detected at build time, starting auto-repair...
[repair] Step 1/3: Installing C++ reference (60-120s estimated)
[setup-cpp-auto] Cloning from https://github.com/microsoft/BitNet...
[setup-cpp-auto] Building with CMake...
[setup-cpp-auto] ✓ C++ reference installed to ~/.cache/bitnet_cpp
[repair] Step 2/3: Rebuilding xtask binary (10-30s estimated)
[repair] Running: cargo build -p xtask --features crossval-all
[repair] ✓ xtask rebuilt successfully
[repair] Step 3/3: Re-executing with updated detection
[preflight] ✓ bitnet.cpp AVAILABLE (auto-repaired in 97.3s)

To cross-validate:
  cargo run -p xtask --features crossval-all -- crossval-per-token \
    --model model.gguf --tokenizer tokenizer.json --prompt "Test" --max-tokens 4

Exit code: 0
```

---

## Acceptance Criteria (AC1-AC14)

### AC1: RepairMode Enum with CLI Integration

**Requirement**: Explicit `RepairMode` enum with three variants and clap integration.

**Implementation**:
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepairMode {
    /// Automatically repair missing backend (default in interactive mode)
    Auto,

    /// Never attempt auto-repair (explicit opt-out, default in CI)
    Never,

    /// Always attempt repair even if backend appears available (force refresh)
    Always,
}

impl RepairMode {
    pub fn from_cli_flags(repair: Option<&str>, ci_detected: bool) -> Self {
        match repair {
            Some("auto") => RepairMode::Auto,
            Some("never") => RepairMode::Never,
            Some("always") => RepairMode::Always,
            None => {
                // Default behavior: CI-aware
                if ci_detected {
                    RepairMode::Never  // Safe default: fail fast in CI
                } else {
                    RepairMode::Auto   // User-friendly: auto-repair locally
                }
            }
            _ => RepairMode::Never,  // Unknown value → conservative default
        }
    }

    pub fn should_repair(self, backend_available: bool) -> bool {
        match self {
            RepairMode::Auto => !backend_available,  // Only if missing
            RepairMode::Never => false,               // Never repair
            RepairMode::Always => true,               // Always repair
        }
    }
}
```

**CLI Integration** (clap):
```rust
#[derive(Parser)]
struct PreflightArgs {
    #[arg(long)]
    backend: String,

    /// Repair mode: auto (default), never, always
    #[arg(long, value_parser = ["auto", "never", "always"])]
    repair: Option<String>,

    #[arg(short, long)]
    verbose: bool,
}
```

**Test Coverage**:
- `test_repair_mode_auto`: Verify Auto mode repairs only when backend missing
- `test_repair_mode_never`: Verify Never mode skips repair even when missing
- `test_repair_mode_always`: Verify Always mode repairs even when available
- `test_repair_mode_default_ci`: Verify CI environment defaults to Never
- `test_repair_mode_default_local`: Verify local environment defaults to Auto

---

### AC2: setup-cpp-auto Invocation on Missing Backend

**Requirement**: When backend missing and repair enabled, invoke `setup-cpp-auto` as subprocess.

**Implementation**:
```rust
fn attempt_repair_once(backend: CppBackend, verbose: bool) -> Result<(), RepairError> {
    // Check recursion guard
    if env::var("BITNET_REPAIR_IN_PROGRESS").is_ok() {
        return Err(RepairError::RecursionDetected);
    }

    // Set recursion guard
    unsafe { env::set_var("BITNET_REPAIR_IN_PROGRESS", "1"); }

    // Invoke setup-cpp-auto
    let current_exe = env::current_exe()
        .context("Failed to get current xtask executable path")?;

    let setup_result = Command::new(&current_exe)
        .args(["setup-cpp-auto", "--emit=sh"])
        .env("BITNET_REPAIR_IN_PROGRESS", "1")  // Explicit env pass
        .output()
        .context("Failed to execute setup-cpp-auto")?;

    // Cleanup recursion guard
    unsafe { env::remove_var("BITNET_REPAIR_IN_PROGRESS"); }

    if !setup_result.status.success() {
        let stderr = String::from_utf8_lossy(&setup_result.stderr);
        return Err(RepairError::classify(&stderr, backend.name()));
    }

    Ok(())
}
```

**Error Classification**: Stderr pattern matching (see AC3)

**Test Coverage**:
- `test_setup_cpp_auto_invocation`: Verify correct command line
- `test_setup_cpp_auto_env_pass`: Verify recursion guard passed to child
- `test_setup_cpp_auto_error_capture`: Verify stderr captured on failure

---

### AC3: Workspace-Local xtask Rebuild

**Requirement**: After successful `setup-cpp-auto`, rebuild xtask binary in workspace.

**Implementation**:
```rust
fn rebuild_xtask_for_detection() -> Result<(), RebuildError> {
    eprintln!("[repair] Rebuilding xtask to pick up new library detection...");

    // Step 1: Clean xtask and crossval crates (ensures full rebuild)
    let clean_status = Command::new("cargo")
        .args(["clean", "-p", "xtask", "-p", "crossval"])
        .status()
        .context("Failed to execute cargo clean")?;

    if !clean_status.success() {
        return Err(RebuildError::CleanFailed {
            code: clean_status.code(),
        });
    }

    // Step 2: Rebuild xtask with crossval features (required for detection)
    let build_status = Command::new("cargo")
        .args(["build", "-p", "xtask", "--features", "crossval-all"])
        .status()
        .context("Failed to execute cargo build")?;

    if !build_status.success() {
        return Err(RebuildError::BuildFailed {
            code: build_status.code(),
        });
    }

    eprintln!("[repair] ✓ xtask rebuild completed");
    Ok(())
}

#[derive(Debug, thiserror::Error)]
pub enum RebuildError {
    #[error("cargo clean failed with exit code {code:?}")]
    CleanFailed { code: Option<i32> },

    #[error("cargo build failed with exit code {code:?}")]
    BuildFailed { code: Option<i32> },

    #[error(transparent)]
    Io(#[from] std::io::Error),
}
```

**Why Clean First**: Ensures build.rs re-runs library detection even if xtask source unchanged.

**Why --features crossval-all**: Required to enable `bitnet_crossval::{HAS_BITNET, HAS_LLAMA}` constants.

**Test Coverage**:
- `test_rebuild_xtask_success`: Verify clean + build sequence
- `test_rebuild_xtask_clean_failure`: Verify error handling when clean fails
- `test_rebuild_xtask_build_failure`: Verify error handling when build fails

---

### AC4: Binary Re-Exec with Preserved Arguments

**Requirement**: After rebuild, re-execute new xtask binary with original arguments.

**Implementation**:
```rust
fn reexec_with_updated_binary(
    original_args: &[String],
    backend: CppBackend,
    verbose: bool,
) -> Result<(), RepairError> {
    eprintln!("[repair] Re-executing with updated detection...");

    // Get path to newly rebuilt xtask binary
    let current_exe = env::current_exe()
        .context("Failed to get current xtask executable path")?;

    // Build command line: preserve original arguments
    let mut cmd = Command::new(&current_exe);
    cmd.args(original_args);

    // Mark as re-exec to prevent further recursion
    cmd.env("BITNET_REPAIR_PARENT", "1");

    // Execute and replace current process (Unix) or spawn + wait (Windows)
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let err = cmd.exec();  // Never returns on success
        return Err(RepairError::ReexecFailed(err.into()));
    }

    #[cfg(not(unix))]
    {
        let status = cmd.status()
            .context("Failed to spawn re-exec process")?;
        std::process::exit(status.code().unwrap_or(1));
    }
}
```

**Argument Preservation**: Pass original `argv` to new process (idempotent).

**Recursion Prevention**: `BITNET_REPAIR_PARENT=1` marks re-exec (see AC5).

**Test Coverage**:
- `test_reexec_preserves_arguments`: Verify argv passed correctly
- `test_reexec_sets_parent_env`: Verify BITNET_REPAIR_PARENT=1
- `test_reexec_replaces_process_unix`: Verify exec() on Unix
- `test_reexec_spawns_child_windows`: Verify spawn + exit on Windows

---

### AC5: Recursion Guard via BITNET_REPAIR_PARENT

**Requirement**: Prevent infinite recursion loops with environment variable guard.

**Implementation**:
```rust
fn preflight_with_auto_repair(
    backend: CppBackend,
    verbose: bool,
    repair_mode: RepairMode,
) -> Result<(), RepairError> {
    // Check if we are a re-exec child (guard against infinite loops)
    if env::var("BITNET_REPAIR_PARENT").is_ok() {
        // We are the re-exec child, do NOT attempt repair again
        eprintln!("[repair] Re-exec detected, skipping repair (checking detection only)");

        // Just check backend availability
        if is_backend_available(backend) {
            println!("✓ {} AVAILABLE (detected after repair)", backend.name());
            return Ok(());
        } else {
            return Err(RepairError::RevalidationFailed {
                backend: backend.name().to_string(),
            });
        }
    }

    // Not a re-exec, proceed with normal flow
    if !is_backend_available(backend) {
        if repair_mode.should_repair(false) {
            // Attempt repair workflow
            attempt_repair_with_retry(backend, verbose)?;
            rebuild_xtask_for_detection()?;
            reexec_with_updated_binary(&env::args().collect::<Vec<_>>(), backend, verbose)?;
        } else {
            return Err(RepairError::BackendUnavailable {
                backend: backend.name().to_string(),
            });
        }
    }

    Ok(())
}
```

**Guard Semantics**:
- **Parent Process**: `BITNET_REPAIR_PARENT` NOT set → Attempts repair
- **Child Process**: `BITNET_REPAIR_PARENT=1` → Skips repair, only validates

**Test Coverage**:
- `test_recursion_guard_parent_not_set`: Verify repair attempted when guard absent
- `test_recursion_guard_parent_set`: Verify repair skipped when guard present
- `test_recursion_guard_revalidation_success`: Verify child detects libraries
- `test_recursion_guard_revalidation_failure`: Verify child fails if libs still missing

---

### AC6: Runtime Fallback Detection with Rebuild Warning

**Requirement**: If build-time detection fails but libraries exist at runtime, warn user to rebuild.

**Implementation**:
```rust
fn check_runtime_fallback(backend: CppBackend) -> Option<PathBuf> {
    // If build-time detection succeeded, skip runtime check
    if is_backend_available_build_time(backend) {
        return None;
    }

    // Build-time detection failed, try runtime discovery
    let install_dir = determine_install_dir(backend).ok()?;
    let lib_dirs = find_backend_lib_dirs(backend, &install_dir).ok()?;

    if !lib_dirs.is_empty() {
        // Libraries exist but build-time detection missed them!
        return Some(install_dir);
    }

    None
}

fn emit_rebuild_warning(backend: CppBackend, install_dir: &Path) {
    eprintln!(
        "\n⚠️  {} libraries detected at runtime but not at build time\n\
         \n\
         Location: {}\n\
         \n\
         This means xtask was compiled before libraries were installed.\n\
         To enable full integration, rebuild xtask:\n\
         \n\
           cargo clean -p xtask && cargo build -p xtask --features crossval-all\n\
         \n\
         Or enable auto-repair:\n\
         \n\
           cargo run -p xtask -- preflight --backend {} --repair=auto\n",
        backend.name(),
        install_dir.display(),
        backend.name()
    );
}
```

**When This Occurs**: User manually installs libraries after building xtask.

**Test Coverage**:
- `test_runtime_fallback_detects_libs`: Verify runtime discovery when build-time missed
- `test_runtime_fallback_emits_warning`: Verify warning message displayed
- `test_runtime_fallback_no_warning_when_build_time_ok`: Verify no warning when build-time worked

---

### AC7: Exit Code Taxonomy (0-6)

**Requirement**: Clear exit codes for CI integration and automated workflows.

**Exit Code Mapping**:
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum PreflightExitCode {
    Available = 0,              // Backend detected, ready for cross-validation
    Unavailable = 1,            // Backend not found, repair disabled or user opt-out
    InvalidArgs = 2,            // Invalid CLI arguments (unknown backend, etc.)
    NetworkFailure = 3,         // Auto-repair failed: network error (retryable)
    PermissionDenied = 4,       // Auto-repair failed: permission error (manual fix)
    BuildFailure = 5,           // Auto-repair failed: build error (install deps)
    RecursionDetected = 6,      // Recursion guard triggered (internal error)
}

impl PreflightExitCode {
    pub fn from_repair_error(err: &RepairError) -> Self {
        match err {
            RepairError::NetworkFailure { .. } => PreflightExitCode::NetworkFailure,
            RepairError::BuildFailure { .. } => PreflightExitCode::BuildFailure,
            RepairError::PermissionDenied { .. } => PreflightExitCode::PermissionDenied,
            RepairError::RecursionDetected => PreflightExitCode::RecursionDetected,
            RepairError::BackendUnavailable { .. } => PreflightExitCode::Unavailable,
            _ => PreflightExitCode::Unavailable,
        }
    }
}
```

**CI Integration Examples**:
```yaml
# GitHub Actions
- name: Preflight check
  run: |
    cargo run -p xtask --features crossval-all -- preflight --backend bitnet
  continue-on-error: true

- name: Interpret exit code
  if: failure()
  run: |
    case $? in
      0) echo "✓ Backend available" ;;
      1) echo "❌ Backend unavailable (manual setup required)" ;;
      3) echo "⚠️  Network error (retry recommended)" ;;
      4) echo "⚠️  Permission error (fix ownership)" ;;
      5) echo "❌ Build error (install dependencies)" ;;
      6) echo "❌ Recursion detected (bug - report to maintainers)" ;;
    esac
```

**Test Coverage**:
- `test_exit_code_available`: Exit 0 when backend detected
- `test_exit_code_unavailable`: Exit 1 when backend missing + repair disabled
- `test_exit_code_network_failure`: Exit 3 on network error
- `test_exit_code_permission_denied`: Exit 4 on permission error
- `test_exit_code_build_failure`: Exit 5 on build error
- `test_exit_code_recursion_detected`: Exit 6 on recursion

---

### AC8: Clear Error Messages with Recovery Steps

**Requirement**: Structured error messages with actionable recovery steps.

**Message Template**:
```rust
fn format_repair_error(err: &RepairError) -> String {
    match err {
        RepairError::NetworkFailure { error, backend } => format!(
            "❌ Backend '{}' UNAVAILABLE (network error during repair)\n\
             \n\
             Error:\n\
               {}\n\
             \n\
             Recovery Steps:\n\
               1. Check internet connectivity:\n\
                  ping github.com\n\
               2. Verify firewall allows git clone:\n\
                  curl -I https://github.com\n\
               3. Retry with backoff (auto-repair will retry 3 times)\n\
               4. For persistent issues, see manual setup:\n\
                  docs/howto/cpp-setup.md\n\
             \n\
             Exit code: 3 (network error - retryable)",
            backend, error
        ),

        RepairError::BuildFailure { error, backend } => format!(
            "❌ Backend '{}' UNAVAILABLE (build error during repair)\n\
             \n\
             Error:\n\
               {}\n\
             \n\
             Recovery Steps:\n\
               1. Check required dependencies:\n\
                  cmake --version      # Need >= 3.18\n\
                  gcc --version        # or clang\n\
                  git --version\n\
               2. Install missing tools (Linux):\n\
                  # Ubuntu/Debian\n\
                  sudo apt-get install cmake build-essential git\n\
                  # CentOS/RHEL\n\
                  sudo yum install cmake gcc-c++ git\n\
               3. Retry repair:\n\
                  cargo run -p xtask -- preflight --backend {} --repair=auto\n\
               4. For detailed setup:\n\
                  docs/development/build-commands.md\n\
                  docs/GPU_SETUP.md (for GPU-related issues)\n\
             \n\
             Exit code: 5 (build error - install dependencies)",
            backend, error, backend
        ),

        RepairError::PermissionDenied { path, backend } => format!(
            "❌ Backend '{}' UNAVAILABLE (permission error during repair)\n\
             \n\
             Error:\n\
               Permission denied: {}\n\
             \n\
             Recovery Steps:\n\
               1. Check directory ownership:\n\
                  ls -ld {}\n\
               2. Fix ownership:\n\
                  sudo chown -R $USER {}\n\
               3. OR use custom directory:\n\
                  export BITNET_CPP_DIR=~/my-custom-bitnet\n\
                  cargo run -p xtask -- preflight --backend {} --repair=auto\n\
               4. See detailed setup guide:\n\
                  docs/howto/cpp-setup.md\n\
             \n\
             Exit code: 4 (permission error - manual fix required)",
            backend, path, path, path, backend
        ),

        _ => format!("{}", err),
    }
}
```

**Test Coverage**:
- `test_error_message_network`: Verify network error format
- `test_error_message_build`: Verify build error format
- `test_error_message_permission`: Verify permission error format
- `test_error_message_has_recovery_steps`: Verify all errors have recovery section
- `test_error_message_has_exit_code`: Verify exit code documented

---

### AC9: Network Retry with Exponential Backoff

**Requirement**: Retry network operations with exponential backoff (3 attempts).

**Implementation**:
```rust
pub struct RetryConfig {
    pub max_retries: u32,
    pub initial_backoff_ms: u64,
    pub max_backoff_ms: u64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        RetryConfig {
            max_retries: 3,
            initial_backoff_ms: 1000,   // 1 second
            max_backoff_ms: 16000,      // 16 seconds
        }
    }
}

pub fn attempt_repair_with_retry(
    backend: CppBackend,
    verbose: bool,
) -> Result<(), RepairError> {
    let config = RetryConfig::default();
    let mut retries = 0;

    loop {
        match attempt_repair_once(backend, verbose) {
            Ok(()) => return Ok(()),
            Err(e) if is_retryable_error(&e) && retries < config.max_retries => {
                retries += 1;

                // Calculate backoff: 1s, 2s, 4s, 8s, ...
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

pub fn is_retryable_error(err: &RepairError) -> bool {
    matches!(err, RepairError::NetworkFailure { .. })
}
```

**Backoff Schedule**:
```
Attempt 1: Immediate
Attempt 2: Wait 1000ms (1s)
Attempt 3: Wait 2000ms (2s)
Attempt 4: Wait 4000ms (4s)

Total retry wait time: 1s + 2s + 4s = 7 seconds
```

**Test Coverage**:
- `test_retry_exponential_backoff`: Verify backoff timing
- `test_retry_max_attempts`: Verify stops after 3 retries
- `test_retry_network_only`: Verify only network errors retried
- `test_retry_build_error_no_retry`: Verify build errors NOT retried

---

### AC10: File Lock per Backend Directory

**Requirement**: Prevent concurrent repairs with advisory file locks.

**Implementation**:
```rust
use std::fs::File;
use std::path::Path;

pub struct RepairLock {
    _file: File,
    lock_path: PathBuf,
}

impl RepairLock {
    pub fn acquire(backend: CppBackend) -> Result<Self, RepairError> {
        let install_dir = determine_install_dir(backend)?;
        let lock_path = install_dir.parent()
            .unwrap_or(&install_dir)
            .join(format!(".{}_repair.lock", backend.name()));

        let file = File::create(&lock_path)
            .with_context(|| format!("Failed to create lock file: {}", lock_path.display()))?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            use nix::fcntl::{flock, FlockArg};
            use std::os::unix::io::AsRawFd;

            flock(file.as_raw_fd(), FlockArg::LockExclusiveNonblock)
                .map_err(|_| RepairError::LockFailed {
                    backend: backend.name().to_string(),
                    path: lock_path.display().to_string(),
                })?;
        }

        Ok(RepairLock {
            _file: file,
            lock_path,
        })
    }
}

impl Drop for RepairLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.lock_path);
    }
}

#[derive(Debug, thiserror::Error)]
#[error("Another repair operation for {backend} is in progress (lock: {path})")]
pub struct LockFailedError {
    pub backend: String,
    pub path: String,
}
```

**Lock Lifecycle**:
```
1. Acquire lock at start of repair
2. Hold lock during setup-cpp-auto + rebuild
3. Release lock on success/failure (automatic via Drop)
```

**Test Coverage**:
- `test_lock_acquire_success`: Verify lock acquired when available
- `test_lock_acquire_failure`: Verify error when lock held by another process
- `test_lock_cleanup_on_drop`: Verify lock file removed on drop
- `test_lock_cleanup_on_panic`: Verify lock released even on panic

---

### AC11: Transactional Rollback on Failure

**Requirement**: Backup existing installation before repair, rollback on failure.

**Implementation**:
```rust
fn install_or_update_backend_transactional(
    backend: CppBackend,
) -> Result<PathBuf, RepairError> {
    let install_dir = determine_install_dir(backend)?;
    let backup_dir = install_dir.with_extension("backup");

    // Step 1: Create backup if existing installation
    if install_dir.exists() {
        eprintln!("[repair] Creating backup of existing installation...");
        fs::rename(&install_dir, &backup_dir)
            .with_context(|| format!("Failed to create backup: {}", backup_dir.display()))?;
    }

    // Step 2: Attempt new installation
    let install_result = install_or_update_backend_internal(backend, &install_dir);

    // Step 3: Handle success/failure
    match install_result {
        Ok(dir) => {
            // Success: cleanup backup
            if backup_dir.exists() {
                eprintln!("[repair] ✓ Installation successful, removing backup...");
                let _ = fs::remove_dir_all(&backup_dir);
            }
            Ok(dir)
        }
        Err(e) => {
            // Failure: restore backup
            if backup_dir.exists() {
                eprintln!("[repair] Installation failed, restoring backup...");
                let _ = fs::remove_dir_all(&install_dir);
                fs::rename(&backup_dir, &install_dir)
                    .with_context(|| "Failed to restore backup")?;
                eprintln!("[repair] ✓ Backup restored");
            }
            Err(e)
        }
    }
}
```

**Atomicity Properties**:
- Backup created before modifying original
- Original preserved until success confirmed
- Automatic rollback on any error
- Idempotent (safe to retry after failure)

**Test Coverage**:
- `test_transactional_backup_created`: Verify backup created before install
- `test_transactional_rollback_on_failure`: Verify backup restored on error
- `test_transactional_cleanup_on_success`: Verify backup removed on success
- `test_transactional_no_backup_for_fresh_install`: Verify no backup when install_dir absent

---

### AC12: Both Backends Supported (BitNet + llama)

**Requirement**: Repair workflow supports both `bitnet.cpp` and `llama.cpp` backends.

**Implementation**:
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CppBackend {
    BitNet,
    Llama,
}

impl CppBackend {
    pub fn name(&self) -> &'static str {
        match self {
            Self::BitNet => "bitnet.cpp",
            Self::Llama => "llama.cpp",
        }
    }

    pub fn repo_url(&self) -> &'static str {
        match self {
            Self::BitNet => "https://github.com/microsoft/BitNet",
            Self::Llama => "https://github.com/ggerganov/llama.cpp",
        }
    }

    pub fn install_subdir(&self) -> &'static str {
        match self {
            Self::BitNet => "bitnet_cpp",
            Self::Llama => "llama_cpp",
        }
    }

    pub fn env_var_dir(&self) -> &'static str {
        match self {
            Self::BitNet => "BITNET_CPP_DIR",
            Self::Llama => "LLAMA_CPP_DIR",
        }
    }
}

// Generic repair function (backend-agnostic)
pub fn preflight_with_auto_repair(
    backend: CppBackend,  // ← Works for both backends
    verbose: bool,
    repair_mode: RepairMode,
) -> Result<(), RepairError> {
    // ... (same implementation for both backends)
}
```

**Backend-Specific Differences** (handled internally):
- Installation directory: `~/.cache/bitnet_cpp` vs `~/.cache/llama_cpp`
- Environment variable: `BITNET_CPP_DIR` vs `LLAMA_CPP_DIR`
- Repository URL: Microsoft/BitNet vs ggerganov/llama.cpp
- Build method: setup_env.py + CMake vs CMake-only

**Test Coverage**:
- `test_repair_bitnet_backend`: End-to-end repair for BitNet.cpp
- `test_repair_llama_backend`: End-to-end repair for llama.cpp
- `test_backend_specific_install_dir`: Verify correct directories
- `test_backend_specific_env_vars`: Verify correct env var precedence

---

### AC13: --repair=auto Default, --repair=never Opt-Out

**Requirement**: Default repair mode is CI-aware (auto locally, never in CI).

**Implementation**:
```rust
fn is_ci_environment() -> bool {
    env::var("CI").is_ok()
        || env::var("GITHUB_ACTIONS").is_ok()
        || env::var("JENKINS_HOME").is_ok()
        || env::var("GITLAB_CI").is_ok()
        || env::var("CIRCLECI").is_ok()
}

impl RepairMode {
    pub fn default_for_environment() -> Self {
        if is_ci_environment() {
            RepairMode::Never  // CI: fail fast, explicit setup
        } else {
            RepairMode::Auto   // Local: user-friendly auto-repair
        }
    }
}

// CLI integration
#[derive(Parser)]
struct PreflightArgs {
    #[arg(long)]
    backend: String,

    /// Repair mode: auto (default locally), never (default in CI), always
    #[arg(long, value_parser = ["auto", "never", "always"])]
    repair: Option<String>,
}

fn run_preflight(args: PreflightArgs) -> Result<()> {
    let backend = parse_backend(&args.backend)?;
    let repair_mode = match args.repair.as_deref() {
        Some(mode) => RepairMode::from_cli_flags(Some(mode), is_ci_environment()),
        None => RepairMode::default_for_environment(),
    };

    preflight_with_auto_repair(backend, args.verbose, repair_mode)
}
```

**Default Behavior**:
```bash
# Local environment (CI=unset)
$ cargo run -p xtask -- preflight --backend bitnet
# → RepairMode::Auto (will auto-repair if backend missing)

# CI environment (CI=true)
$ CI=true cargo run -p xtask -- preflight --backend bitnet
# → RepairMode::Never (fails fast if backend missing)

# Explicit override in CI
$ CI=true cargo run -p xtask -- preflight --backend bitnet --repair=auto
# → RepairMode::Auto (user explicitly requested auto-repair)
```

**Test Coverage**:
- `test_default_repair_local`: Verify Auto default locally
- `test_default_repair_ci`: Verify Never default in CI
- `test_explicit_override_ci`: Verify --repair=auto overrides CI default

---

### AC14: Comprehensive Tests with Mock Flows

**Requirement**: Complete test coverage with mocked network/filesystem operations.

**Test Structure**:
```rust
// Test helpers (xtask/tests/preflight_repair_tests.rs)
mod helpers {
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;

    pub struct MockSetupCppAuto {
        pub calls: Mutex<Vec<SetupCall>>,
    }

    pub struct SetupCall {
        pub backend: String,
        pub emit: String,
        pub success: bool,
    }

    impl MockSetupCppAuto {
        pub fn new() -> Self {
            MockSetupCppAuto {
                calls: Mutex::new(Vec::new()),
            }
        }

        pub fn simulate_success(&self) {
            self.calls.lock().unwrap().push(SetupCall {
                backend: "bitnet".to_string(),
                emit: "sh".to_string(),
                success: true,
            });
        }

        pub fn simulate_network_failure(&self) {
            // Simulate stderr with network error pattern
        }
    }

    pub fn mock_backend_missing(backend: &str) -> tempfile::TempDir {
        // Create temp directory without libraries
        let temp = tempfile::tempdir().unwrap();
        temp
    }

    pub fn mock_backend_available(backend: &str) -> tempfile::TempDir {
        // Create temp directory with mock libraries
        let temp = tempfile::tempdir().unwrap();
        let lib_dir = temp.path().join("build/bin");
        fs::create_dir_all(&lib_dir).unwrap();

        #[cfg(target_os = "linux")]
        let lib_name = format!("lib{}.so", backend);
        #[cfg(target_os = "macos")]
        let lib_name = format!("lib{}.dylib", backend);
        #[cfg(target_os = "windows")]
        let lib_name = format!("{}.dll", backend);

        fs::write(lib_dir.join(lib_name), b"mock library").unwrap();
        temp
    }
}

// Unit tests (37+ tests covering AC1-AC14)
#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    fn test_repair_mode_auto() {
        let mode = RepairMode::Auto;
        assert!(mode.should_repair(false));  // Repair when missing
        assert!(!mode.should_repair(true));  // Don't repair when available
    }

    #[test]
    fn test_repair_mode_never() {
        let mode = RepairMode::Never;
        assert!(!mode.should_repair(false));  // Never repair
        assert!(!mode.should_repair(true));   // Never repair
    }

    #[test]
    fn test_repair_mode_always() {
        let mode = RepairMode::Always;
        assert!(mode.should_repair(false));  // Always repair
        assert!(mode.should_repair(true));   // Always repair
    }

    #[test]
    #[serial(bitnet_env)]
    fn test_ci_detection_github_actions() {
        env::set_var("GITHUB_ACTIONS", "true");
        assert!(is_ci_environment());
        env::remove_var("GITHUB_ACTIONS");
    }

    #[test]
    fn test_setup_cpp_auto_invocation() {
        // Mock setup-cpp-auto command execution
        // Verify correct arguments passed
    }

    #[test]
    fn test_rebuild_xtask_success() {
        // Mock cargo clean + cargo build
        // Verify sequence executed correctly
    }

    #[test]
    fn test_reexec_preserves_arguments() {
        // Mock exec() call
        // Verify original argv passed to new process
    }

    #[test]
    fn test_recursion_guard_prevents_infinite_loop() {
        env::set_var("BITNET_REPAIR_PARENT", "1");
        // Verify repair skipped when guard present
        env::remove_var("BITNET_REPAIR_PARENT");
    }

    #[test]
    fn test_retry_exponential_backoff() {
        // Mock 3 network failures
        // Verify backoff: 1s, 2s, 4s
    }

    #[test]
    fn test_file_lock_prevents_concurrent_repair() {
        // Create lock file
        // Attempt second lock
        // Verify error returned
    }

    #[test]
    fn test_transactional_rollback_on_failure() {
        // Create existing installation
        // Mock setup-cpp-auto failure
        // Verify backup restored
    }

    // ... (25+ more tests covering all ACs)
}
```

**Test Coverage Goals**:
- **AC1**: 5 tests (RepairMode enum variants + CLI parsing)
- **AC2**: 3 tests (setup-cpp-auto invocation, env passing, error capture)
- **AC3**: 3 tests (rebuild success, clean failure, build failure)
- **AC4**: 4 tests (reexec argument preservation, env setting, Unix exec, Windows spawn)
- **AC5**: 4 tests (recursion guard set/unset, revalidation success/failure)
- **AC6**: 3 tests (runtime fallback detection, warning emission)
- **AC7**: 7 tests (one per exit code: 0-6)
- **AC8**: 5 tests (error message formatting, recovery steps)
- **AC9**: 4 tests (retry timing, max attempts, retryable errors)
- **AC10**: 4 tests (lock acquire, lock conflict, cleanup)
- **AC11**: 4 tests (backup creation, rollback, cleanup, no-backup case)
- **AC12**: 4 tests (BitNet backend, llama backend, install dirs, env vars)
- **AC13**: 3 tests (CI default, local default, explicit override)
- **AC14**: Integration test suite (end-to-end workflows)

**Total**: 50+ tests

---

## Architecture

### State Machine for Repair Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ State: INITIAL                                                   │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Check: BITNET_REPAIR_PARENT env var                        │ │
│ │ ├─ Present? → Transition to RE_EXEC_CHILD                 │ │
│ │ └─ Absent? → Transition to CHECK_BACKEND                  │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ State: RE_EXEC_CHILD (recursion guard)                          │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Action: Skip repair, only validate detection              │ │
│ │ Check: is_backend_available(backend)                      │ │
│ │ ├─ True? → Transition to SUCCESS                         │ │
│ │ └─ False? → Transition to REVALIDATION_FAILED           │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ State: CHECK_BACKEND                                             │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Check: is_backend_available(backend)                      │ │
│ │ ├─ True? → Transition to SUCCESS                         │ │
│ │ └─ False? → Transition to CHECK_REPAIR_MODE             │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ State: CHECK_REPAIR_MODE                                         │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Check: repair_mode.should_repair(false)                   │ │
│ │ ├─ True? → Transition to ACQUIRE_LOCK                    │ │
│ │ └─ False? → Transition to UNAVAILABLE_REPAIR_DISABLED   │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ State: ACQUIRE_LOCK                                              │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Action: RepairLock::acquire(backend)                      │ │
│ │ ├─ Success? → Transition to CREATE_BACKUP                │ │
│ │ └─ Failure? → Transition to LOCK_FAILED                  │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ State: CREATE_BACKUP                                             │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Action: fs::rename(install_dir, backup_dir)              │ │
│ │ (if install_dir exists)                                   │ │
│ │ └─ Always transition to SETUP_CPP_AUTO                   │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ State: SETUP_CPP_AUTO (with retry loop)                         │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Action: attempt_repair_with_retry(backend, verbose)      │ │
│ │ Retry: 3 attempts with exponential backoff (1s, 2s, 4s) │ │
│ │ ├─ Success? → Transition to REBUILD_XTASK                │ │
│ │ ├─ Network error? → Retry up to 3 times                  │ │
│ │ └─ Other error? → Transition to REPAIR_FAILED           │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ State: REBUILD_XTASK                                             │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Action: rebuild_xtask_for_detection()                     │ │
│ │   Step 1: cargo clean -p xtask -p crossval              │ │
│ │   Step 2: cargo build -p xtask --features crossval-all  │ │
│ │ ├─ Success? → Transition to RE_EXEC                      │ │
│ │ └─ Failure? → Transition to REBUILD_FAILED              │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ State: RE_EXEC                                                   │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Action: reexec_with_updated_binary(original_args)        │ │
│ │   - Set BITNET_REPAIR_PARENT=1                           │ │
│ │   - Unix: exec() (replaces process)                      │ │
│ │   - Windows: spawn + exit                                │ │
│ │ ├─ Success? → Process replaced, new process starts      │ │
│ │ └─ Failure? → Transition to REEXEC_FAILED               │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ State: SUCCESS                                                   │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Action: Print success message                            │ │
│ │ Output: "✓ {backend} AVAILABLE"                         │ │
│ │ Exit code: 0                                             │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Error States (with exit codes)                                  │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ UNAVAILABLE_REPAIR_DISABLED → Exit 1                     │ │
│ │ LOCK_FAILED → Exit 1 (with clear message)                │ │
│ │ REPAIR_FAILED → Exit 3/4/5 (based on error type)        │ │
│ │ REBUILD_FAILED → Exit 5                                  │ │
│ │ REEXEC_FAILED → Exit 1                                   │ │
│ │ REVALIDATION_FAILED → Exit 1                             │ │
│ │ RECURSION_DETECTED → Exit 6                              │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Rebuild + Re-Exec Algorithm (Step-by-Step)

### Phase 1: Pre-Repair Validation

```
Step 1.1: Check Re-Exec Guard
  IF env::var("BITNET_REPAIR_PARENT") is Ok:
    THEN:
      eprintln("[repair] Re-exec detected, skipping repair")
      IF is_backend_available(backend):
        THEN: EXIT 0 (success)
        ELSE: EXIT 1 (revalidation failed)
  ELSE:
    CONTINUE to Step 1.2

Step 1.2: Check Build-Time Detection
  IF is_backend_available(backend):
    THEN: EXIT 0 (success, backend already detected)
  ELSE:
    CONTINUE to Step 1.3

Step 1.3: Check Repair Mode
  repair_mode = RepairMode::from_cli_flags(args.repair, is_ci_environment())
  IF NOT repair_mode.should_repair(false):
    THEN:
      eprintln("❌ Backend unavailable (repair disabled)")
      EXIT 1 (unavailable)
  ELSE:
    CONTINUE to Phase 2
```

### Phase 2: Acquire Resources

```
Step 2.1: Acquire File Lock
  TRY:
    lock = RepairLock::acquire(backend)
  CATCH LockFailed:
    eprintln("Another repair operation for {} is in progress", backend)
    EXIT 1 (lock failed)

Step 2.2: Create Installation Backup (if exists)
  install_dir = determine_install_dir(backend)
  backup_dir = install_dir.with_extension("backup")

  IF install_dir.exists():
    THEN:
      eprintln("[repair] Creating backup...")
      fs::rename(install_dir, backup_dir)

  CONTINUE to Phase 3
```

### Phase 3: Install C++ Reference

```
Step 3.1: Set Recursion Guard
  unsafe { env::set_var("BITNET_REPAIR_IN_PROGRESS", "1") }

Step 3.2: Invoke setup-cpp-auto (with retry)
  retry_count = 0
  LOOP:
    current_exe = env::current_exe()
    TRY:
      output = Command::new(current_exe)
        .args(["setup-cpp-auto", "--emit=sh"])
        .env("BITNET_REPAIR_IN_PROGRESS", "1")
        .output()

    IF output.status.success():
      THEN:
        BREAK (success)
    ELSE:
      stderr = String::from_utf8_lossy(&output.stderr)
      error = RepairError::classify(&stderr, backend.name())

      IF is_retryable_error(&error) AND retry_count < 3:
        THEN:
          retry_count += 1
          backoff_ms = 1000 * 2^(retry_count - 1)
          eprintln("[repair] Retry {}/3 after {}ms...", retry_count, backoff_ms)
          sleep(Duration::from_millis(backoff_ms))
          CONTINUE loop
      ELSE:
        ROLLBACK: Restore backup if exists
        EXIT exit_code_from_error(error)

Step 3.3: Cleanup Recursion Guard
  unsafe { env::remove_var("BITNET_REPAIR_IN_PROGRESS") }

  CONTINUE to Phase 4
```

### Phase 4: Rebuild xtask Binary

```
Step 4.1: Clean xtask and crossval Crates
  TRY:
    status = Command::new("cargo")
      .args(["clean", "-p", "xtask", "-p", "crossval"])
      .status()

  IF NOT status.success():
    THEN:
      ROLLBACK: Restore backup if exists
      EXIT 5 (rebuild failed)

Step 4.2: Build xtask with crossval Features
  TRY:
    status = Command::new("cargo")
      .args(["build", "-p", "xtask", "--features", "crossval-all"])
      .status()

  IF NOT status.success():
    THEN:
      ROLLBACK: Restore backup if exists
      EXIT 5 (rebuild failed)

  eprintln("[repair] ✓ xtask rebuilt successfully")
  CONTINUE to Phase 5
```

### Phase 5: Re-Execute New Binary

```
Step 5.1: Prepare Re-Exec Arguments
  current_exe = env::current_exe()
  original_args = env::args().collect::<Vec<_>>()

Step 5.2: Execute New Binary (Platform-Specific)
  #[cfg(unix)]
  {
    // Unix: exec() replaces current process (never returns)
    use std::os::unix::process::CommandExt;

    let err = Command::new(current_exe)
      .args(&original_args)
      .env("BITNET_REPAIR_PARENT", "1")
      .exec();  // Never returns on success

    // Only reaches here on exec() failure
    eprintln!("[repair] exec() failed: {}", err);
    ROLLBACK: Restore backup if exists
    EXIT 1 (reexec failed)
  }

  #[cfg(not(unix))]
  {
    // Windows: spawn + wait + exit
    let status = Command::new(current_exe)
      .args(&original_args)
      .env("BITNET_REPAIR_PARENT", "1")
      .status();

    match status {
      Ok(status) => std::process::exit(status.code().unwrap_or(1)),
      Err(e) => {
        eprintln!("[repair] spawn failed: {}", e);
        ROLLBACK: Restore backup if exists
        EXIT 1 (reexec failed)
      }
    }
  }
```

### Phase 6: Cleanup (Post-Success)

```
Step 6.1: Validate Detection (in re-exec child)
  # At this point, we are in the NEW xtask process
  # BITNET_REPAIR_PARENT=1 is set

  IF is_backend_available(backend):
    THEN:
      eprintln("✓ {} AVAILABLE (auto-repaired)", backend.name())
      EXIT 0 (success)
  ELSE:
    eprintln("❌ Backend still unavailable after repair")
    EXIT 1 (revalidation failed)

Step 6.2: Cleanup Backup (automatic via RepairLock Drop)
  # When RepairLock goes out of scope:
  backup_dir = install_dir.with_extension("backup")
  IF backup_dir.exists():
    fs::remove_dir_all(backup_dir)

  # Lock file automatically removed
```

### Safety Guarantees

**Idempotency**:
- Running repair multiple times produces same result
- Backup/restore mechanism prevents corruption
- File locks prevent concurrent operations

**Atomicity** (best-effort):
- Backup created before modifying original
- Rollback on any error (setup, rebuild, reexec)
- Lock released even on panic (Drop guarantee)

**Isolation**:
- Recursion guard (`BITNET_REPAIR_PARENT`) prevents infinite loops
- File lock prevents concurrent repairs
- Environment variables scoped to repair operation

**State Preservation**:
- Original arguments passed to re-exec
- Environment variables preserved
- Working directory maintained

---

## Error Classification

### Pattern Matching Algorithm

**Function**: `RepairError::classify(stderr: &str, backend: &str) -> RepairError`

**Input**: Raw stderr output from `setup-cpp-auto`
**Output**: Classified `RepairError` with recovery context

**Classification Rules** (priority order):

```rust
pub fn classify(stderr: &str, backend: &str) -> Self {
    let lower = stderr.to_lowercase();

    // Priority 1: Network errors (transient, retryable)
    if lower.contains("connection timeout")
        || lower.contains("failed to clone")
        || lower.contains("could not resolve host")
        || lower.contains("network unreachable")
        || lower.contains("connection refused")
        || lower.contains("failed to connect")
        || lower.contains("timed out")
        || lower.contains("temporary failure in name resolution")
        || lower.contains("unable to access")
        || lower.contains("gnutls_handshake() failed")
    {
        return RepairError::NetworkFailure {
            error: stderr.to_string(),
            backend: backend.to_string(),
        };
    }

    // Priority 2: Build errors (permanent, requires deps)
    if lower.contains("cmake error")
        || lower.contains("cmake not found")
        || lower.contains("ninja: build stopped")
        || lower.contains("undefined reference")
        || lower.contains("no such file or directory")
        || lower.contains("compilation failed")
        || lower.contains("cannot find")
        || lower.contains("compiler")
        || lower.contains("make: *** [")
        || lower.contains("ld: ")
        || lower.contains("collect2: error: ld returned")
    {
        return RepairError::BuildFailure {
            error: stderr.to_string(),
            backend: backend.to_string(),
        };
    }

    // Priority 3: Permission errors (permanent, requires chown)
    if lower.contains("permission denied")
        || lower.contains("eacces")
        || lower.contains("cannot create directory")
        || lower.contains("access is denied")
    {
        // Extract path from error message
        let path = extract_path_from_error(stderr)
            .unwrap_or_else(|| format!("~/.cache/{}", backend));

        return RepairError::PermissionDenied {
            path,
            backend: backend.to_string(),
        };
    }

    // Priority 4: Unknown (catch-all)
    RepairError::Unknown {
        error: stderr.to_string(),
        backend: backend.to_string(),
    }
}
```

### Path Extraction Helper

```rust
fn extract_path_from_error(stderr: &str) -> Option<String> {
    // Pattern 1: "permission denied: /path/to/dir"
    if let Some(idx) = stderr.find("permission denied:") {
        let rest = &stderr[idx + 18..];  // Skip "permission denied:"
        if let Some(path_start) = rest.find('/') {
            if let Some(path_end) = rest[path_start..].find('\n') {
                return Some(rest[path_start..path_start + path_end].trim().to_string());
            }
        }
    }

    // Pattern 2: "cannot create directory '/path/to/dir'"
    if let Some(idx) = stderr.find("cannot create directory") {
        let rest = &stderr[idx..];
        if let Some(quote_start) = rest.find('\'') {
            let after_quote = &rest[quote_start + 1..];
            if let Some(quote_end) = after_quote.find('\'') {
                return Some(after_quote[..quote_end].to_string());
            }
        }
    }

    None
}
```

### Retryability Predicate

```rust
pub fn is_retryable_error(err: &RepairError) -> bool {
    matches!(err, RepairError::NetworkFailure { .. })
}
```

**Rationale**:
- **Network errors**: Transient (DNS flakiness, GitHub rate limiting, connection timeouts)
- **Build errors**: Permanent (missing CMake, compiler, headers) → Don't retry
- **Permission errors**: Permanent (ownership, ACLs) → Don't retry
- **Unknown errors**: Unpredictable → Don't retry (conservative)

### Error Examples

**Network Error**:
```
fatal: unable to access 'https://github.com/microsoft/BitNet/':
  gnutls_handshake() failed: Error in the pull function.

→ Classified as: RepairError::NetworkFailure
→ Action: Retry with backoff (3 attempts)
→ Exit code: 3
```

**Build Error**:
```
CMake Error at CMakeLists.txt:3 (project):
  No CMAKE_CXX_COMPILER could be found.

→ Classified as: RepairError::BuildFailure
→ Action: Do NOT retry, emit recovery steps
→ Exit code: 5
```

**Permission Error**:
```
mkdir: cannot create directory '/home/user/.cache/bitnet_cpp':
  Permission denied

→ Classified as: RepairError::PermissionDenied
→ Extracted path: /home/user/.cache/bitnet_cpp
→ Action: Do NOT retry, emit chown command
→ Exit code: 4
```

---

## Exit Code Taxonomy (Complete Mapping)

### Exit Code Reference Table

| Code | Name | Meaning | CI Action | Recovery |
|------|------|---------|-----------|----------|
| 0 | Available | Backend detected at build time, ready for cross-validation | Continue with tests | N/A (success) |
| 1 | Unavailable | Backend not found, repair disabled or failed (non-network) | Stop, fail job | Enable auto-repair or pre-provision |
| 2 | InvalidArgs | Invalid CLI arguments (unknown backend, bad flag) | Stop, fix command syntax | Check --help for valid options |
| 3 | NetworkFailure | Auto-repair failed due to network error (after retries) | Retry job later (transient) | Check internet, retry, or pre-provision |
| 4 | PermissionDenied | Auto-repair failed due to permission error | Stop, requires intervention | Fix directory ownership with chown |
| 5 | BuildFailure | Auto-repair failed due to build error (CMake, compiler) | Stop, requires intervention | Install dependencies (cmake, gcc) |
| 6 | RecursionDetected | Recursion guard triggered (internal error) | Stop, report bug | Report to maintainers (likely bug) |

### Exit Code Usage Examples

**Bash Script**:
```bash
#!/bin/bash
set -e

cargo run -p xtask --features crossval-all -- preflight --backend bitnet --repair=auto
EXIT_CODE=$?

case $EXIT_CODE in
  0)
    echo "✓ Backend available, proceeding with cross-validation"
    cargo run -p xtask --features crossval-all -- crossval-per-token \
      --model model.gguf --tokenizer tokenizer.json --prompt "Test" --max-tokens 4
    ;;
  1)
    echo "❌ Backend unavailable (manual setup required)"
    exit 1
    ;;
  2)
    echo "❌ Invalid arguments"
    cargo run -p xtask -- preflight --help
    exit 2
    ;;
  3)
    echo "⚠️  Network error (retrying in 60s...)"
    sleep 60
    exec "$0"  # Retry script
    ;;
  4)
    echo "❌ Permission error (fix with: sudo chown -R $USER ~/.cache/bitnet_cpp)"
    exit 4
    ;;
  5)
    echo "❌ Build error (install dependencies: sudo apt-get install cmake build-essential)"
    exit 5
    ;;
  6)
    echo "❌ Internal error (recursion detected - please report bug)"
    exit 6
    ;;
  *)
    echo "Unknown exit code: $EXIT_CODE"
    exit 1
    ;;
esac
```

**GitHub Actions**:
```yaml
- name: Preflight check with auto-repair
  id: preflight
  run: |
    cargo run -p xtask --features crossval-all -- \
      preflight --backend bitnet --repair=auto --verbose
  continue-on-error: true

- name: Handle preflight result
  if: steps.preflight.outcome != 'success'
  run: |
    EXIT_CODE=${{ steps.preflight.outputs.exitcode }}

    if [ "$EXIT_CODE" = "3" ]; then
      echo "::warning::Network error during repair (transient)"
      echo "Consider pre-provisioning C++ backend in CI cache"
    elif [ "$EXIT_CODE" = "5" ]; then
      echo "::error::Build error - missing dependencies"
      echo "Install cmake and build-essential in CI image"
      exit 1
    else
      echo "::error::Preflight failed with exit code $EXIT_CODE"
      exit 1
    fi
```

---

## Testing Strategy

### Unit Test Categories

#### Category A: RepairMode Logic (5 tests)
```rust
#[test]
fn test_repair_mode_auto_repairs_when_missing()
#[test]
fn test_repair_mode_never_skips_repair()
#[test]
fn test_repair_mode_always_forces_repair()
#[test]
fn test_repair_mode_default_ci_is_never()
#[test]
fn test_repair_mode_default_local_is_auto()
```

#### Category B: Error Classification (8 tests)
```rust
#[test]
fn test_classify_network_error_timeout()
#[test]
fn test_classify_network_error_dns()
#[test]
fn test_classify_build_error_cmake()
#[test]
fn test_classify_build_error_compiler()
#[test]
fn test_classify_permission_error_mkdir()
#[test]
fn test_classify_permission_error_eacces()
#[test]
fn test_classify_unknown_error()
#[test]
fn test_extract_path_from_permission_error()
```

#### Category C: Recursion Guard (4 tests)
```rust
#[test]
#[serial(bitnet_env)]
fn test_recursion_guard_parent_not_set_allows_repair()
#[test]
#[serial(bitnet_env)]
fn test_recursion_guard_parent_set_skips_repair()
#[test]
#[serial(bitnet_env)]
fn test_recursion_guard_revalidation_success()
#[test]
#[serial(bitnet_env)]
fn test_recursion_guard_revalidation_failure()
```

#### Category D: Retry Logic (4 tests)
```rust
#[test]
fn test_retry_exponential_backoff_timing()
#[test]
fn test_retry_max_attempts_respected()
#[test]
fn test_retry_network_error_retryable()
#[test]
fn test_retry_build_error_not_retryable()
```

#### Category E: File Locking (4 tests)
```rust
#[test]
fn test_lock_acquire_success()
#[test]
fn test_lock_acquire_failure_when_held()
#[test]
fn test_lock_cleanup_on_drop()
#[test]
fn test_lock_cleanup_on_panic()
```

#### Category F: Transactional Rollback (4 tests)
```rust
#[test]
fn test_transactional_backup_created()
#[test]
fn test_transactional_rollback_on_failure()
#[test]
fn test_transactional_cleanup_on_success()
#[test]
fn test_transactional_no_backup_for_fresh_install()
```

#### Category G: Exit Code Mapping (7 tests)
```rust
#[test]
fn test_exit_code_available()
#[test]
fn test_exit_code_unavailable()
#[test]
fn test_exit_code_invalid_args()
#[test]
fn test_exit_code_network_failure()
#[test]
fn test_exit_code_permission_denied()
#[test]
fn test_exit_code_build_failure()
#[test]
fn test_exit_code_recursion_detected()
```

#### Category H: Integration Tests (5 tests)
```rust
#[test]
#[ignore]  // Requires network and build tools
fn test_end_to_end_repair_bitnet_success()
#[test]
#[ignore]
fn test_end_to_end_repair_llama_success()
#[test]
#[ignore]
fn test_end_to_end_repair_network_failure()
#[test]
#[ignore]
fn test_end_to_end_repair_build_failure()
#[test]
#[ignore]
fn test_end_to_end_repair_permission_failure()
```

### Mock Infrastructure

**Mock setup-cpp-auto**:
```rust
struct MockSetupCppAuto {
    pub calls: Vec<SetupCall>,
    pub behavior: MockBehavior,
}

enum MockBehavior {
    Success,
    NetworkError(String),
    BuildError(String),
    PermissionError(String),
}

impl MockSetupCppAuto {
    pub fn simulate_network_error() -> Self {
        MockSetupCppAuto {
            calls: vec![],
            behavior: MockBehavior::NetworkError(
                "fatal: unable to access 'https://github.com/...': connection timeout".to_string()
            ),
        }
    }

    pub fn execute(&mut self, backend: &str) -> Result<(), String> {
        self.calls.push(SetupCall {
            backend: backend.to_string(),
            timestamp: Instant::now(),
        });

        match &self.behavior {
            MockBehavior::Success => Ok(()),
            MockBehavior::NetworkError(msg) => Err(msg.clone()),
            MockBehavior::BuildError(msg) => Err(msg.clone()),
            MockBehavior::PermissionError(msg) => Err(msg.clone()),
        }
    }
}
```

**Mock cargo rebuild**:
```rust
struct MockCargoBuild {
    pub clean_success: bool,
    pub build_success: bool,
}

impl MockCargoBuild {
    pub fn clean(&self) -> Result<(), std::io::Error> {
        if self.clean_success {
            Ok(())
        } else {
            Err(std::io::Error::new(std::io::ErrorKind::Other, "clean failed"))
        }
    }

    pub fn build(&self) -> Result<(), std::io::Error> {
        if self.build_success {
            Ok(())
        } else {
            Err(std::io::Error::new(std::io::ErrorKind::Other, "build failed"))
        }
    }
}
```

---

## Risks

### Risk 1: Recursion Loops

**Scenario**: Recursion guard (`BITNET_REPAIR_PARENT`) fails to prevent infinite loop.

**Causes**:
- Environment variable not properly passed to child process
- Guard cleared before re-exec completes
- Multiple repair attempts in parallel (different terminals)

**Mitigation**:
- **Primary**: File lock (AC10) prevents concurrent repairs
- **Secondary**: Recursion guard checked at function entry (AC5)
- **Tertiary**: Maximum retry count (AC9) limits damage

**Detection**:
- Process CPU usage spikes (tight loop)
- Disk space exhaustion (repeated builds)
- User reports "hang" during repair

**Recovery**:
- User kills process (Ctrl+C)
- Exit code 6 (RecursionDetected) triggers bug report
- File lock released automatically

### Risk 2: Concurrent Repairs (Race Conditions)

**Scenario**: Two users simultaneously run auto-repair on same installation.

**Causes**:
- No file locking (if AC10 not implemented)
- Lock timeout/stale locks
- Network filesystem (NFS) with delayed lock propagation

**Mitigation**:
- **Primary**: Advisory file lock (AC10) with non-blocking acquire
- **Secondary**: Backup/restore mechanism (AC11) isolates failures
- **Tertiary**: Error message guides user to wait or use different directory

**Detection**:
- Lock acquisition failure (clear error message)
- Partial build artifacts (detected by library verification)

**Recovery**:
- User waits for first repair to complete
- User exports `BITNET_CPP_DIR=/tmp/bitnet_cpp_user2` (custom directory)

### Risk 3: Network Flakiness

**Scenario**: Transient network errors exhaust retry budget.

**Causes**:
- GitHub rate limiting (429)
- DNS resolution failures
- Firewall interference
- Intermittent connectivity

**Mitigation**:
- **Primary**: Retry with exponential backoff (AC9, 3 attempts)
- **Secondary**: Clear error messages with recovery steps (AC8)
- **Tertiary**: User can manually provision with `setup-cpp-auto`

**Detection**:
- Exit code 3 (NetworkFailure)
- Stderr contains "connection timeout" or "failed to clone"

**Recovery**:
```bash
# Option 1: Retry auto-repair (will retry 3 times)
cargo run -p xtask -- preflight --backend bitnet --repair=auto

# Option 2: Manual setup
eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"
cargo clean -p xtask && cargo build -p xtask --features crossval-all

# Option 3: Pre-download tarball (offline install)
wget https://github.com/microsoft/BitNet/archive/refs/heads/main.tar.gz
tar xzf main.tar.gz -C ~/.cache/bitnet_cpp
```

### Risk 4: Partial Build Corruption

**Scenario**: Build fails mid-compilation, leaving partial artifacts.

**Causes**:
- Disk space exhaustion during build
- OOM killer terminates build process
- Power loss / system crash
- CMake misconfiguration

**Mitigation**:
- **Primary**: Transactional backup/restore (AC11)
- **Secondary**: `cargo clean` before rebuild (AC3)
- **Tertiary**: Library verification after build (existing pattern)

**Detection**:
- Rebuild fails with "undefined reference" (partial link)
- Library verification finds 0 libraries (post-build check)
- Exit code 5 (BuildFailure)

**Recovery**:
- Automatic rollback restores previous working installation
- User can manually clean: `rm -rf ~/.cache/bitnet_cpp && retry`

### Risk 5: Rebuild Breaks Other Features

**Scenario**: Rebuilding xtask with `--features crossval-all` breaks non-crossval commands.

**Causes**:
- Feature unification conflicts
- Dependency version mismatches
- Build script side effects

**Mitigation**:
- **Primary**: `--features crossval-all` is additive (doesn't disable other features)
- **Secondary**: Test scaffolding validates non-crossval commands post-repair
- **Tertiary**: User can manually rebuild without `crossval-all`

**Detection**:
- Non-crossval commands fail after repair
- Exit code 1 (Unavailable) when expecting 0

**Recovery**:
```bash
# Rebuild with default features
cargo clean -p xtask
cargo build -p xtask
```

### Risk 6: Re-Exec Fails Silently

**Scenario**: `exec()` or `spawn()` fails but doesn't surface clear error.

**Causes**:
- Binary missing (removed between rebuild and re-exec)
- Invalid executable permissions
- Platform-specific exec() failure (e.g., ARM vs x86)

**Mitigation**:
- **Primary**: Error handling in `reexec_with_updated_binary()` (AC4)
- **Secondary**: Rollback mechanism preserves working state (AC11)
- **Tertiary**: Exit code 1 (generic failure) with clear message

**Detection**:
- Re-exec returns error instead of replacing process
- User sees "exec() failed" message

**Recovery**:
- Automatic rollback restores previous installation
- User manually verifies xtask binary exists: `ls -l target/debug/xtask`

---

## Summary

This specification defines a comprehensive **end-to-end auto-repair workflow** for BitNet.rs cross-validation infrastructure:

**Key Innovations**:
1. **One-Command Experience**: `--repair=auto` replaces 5-step manual workflow
2. **Build-Time Awareness**: Acknowledges and solves the rebuild challenge
3. **CI-Aware Defaults**: Safe for CI (Never), user-friendly locally (Auto)
4. **Robust Error Handling**: Network retry, transactional rollback, clear recovery steps
5. **Safety First**: File locks, recursion guards, idempotent operations

**Implementation Complexity**: **Medium-High**
- 1100+ lines specification
- 50+ test cases across 8 categories
- Platform-specific code (Unix exec vs Windows spawn)
- Integration with existing build system and preflight infrastructure

**User Impact**: **High**
- Reduces cross-validation setup from 5+ commands to 1
- Eliminates need to understand build-time vs runtime detection
- Automatic retry for transient failures
- Clear error messages with actionable recovery steps

**Maintenance Burden**: **Low** (after initial implementation)
- Self-contained repair module
- Well-defined error taxonomy
- Comprehensive test coverage
- CI integration patterns documented

**Next Steps**:
1. Implement AC1-AC6 (core repair loop)
2. Implement AC7-AC11 (error handling + safety)
3. Implement AC12-AC14 (dual backend + testing)
4. Integration testing with real C++ backends
5. Documentation updates (CLAUDE.md, howto guides)
6. CI/CD pipeline integration examples

---

## References

### Source Analysis

This specification is derived from comprehensive codebase analysis documented in:

- **Analysis Artifact**: `/tmp/preflight_repair_patterns_analysis.md`
  - Date: 2025-10-26
  - Focus: Preflight auto-repair architecture, rebuild/re-exec flow, and recursion guards
  - Sections: Current architecture, repair mode state machine, rebuild + re-exec flow, error taxonomy

### Code Locations

- **Preflight Core**: `xtask/src/crossval/preflight.rs` (1525 lines)
  - RepairMode enum and state machine
  - Error classification taxonomy
  - Rebuild and re-exec stubs
  - Recursion guard implementation

- **setup-cpp-auto**: `xtask/src/cpp_setup_auto.rs` (1418 lines)
  - C++ backend installation orchestration
  - Library path detection and environment export
  - Build system integration (CMake, setup_env.py)

- **AC2 Integration**: `xtask/src/crossval/preflight_ac2.rs` (224 lines)
  - setup-cpp-auto subprocess invocation
  - Shell export parsing and environment propagation

- **Test Scaffolding**: `xtask/tests/preflight_repair_mode_tests.rs`
  - Comprehensive TDD test scaffolding for AC1-AC14
  - 54+ test cases organized by acceptance criterion
  - Serial execution with environment isolation

### Related Documentation

- **CLAUDE.md**: Project-wide guidance, cross-validation setup, preflight usage
- **docs/howto/cpp-setup.md**: Manual C++ reference setup guide
- **docs/explanation/dual-backend-crossval.md**: Dual-backend architecture (BitNet.cpp + llama.cpp)
- **docs/development/test-suite.md**: Test framework and ignored tests

### Related Specifications

- **preflight-auto-repair.md**: Original auto-repair design (AC1-AC7)
- **bitnet-cpp-auto-setup-parity.md**: setup-cpp-auto implementation details
- **test-infra-auto-repair-ci.md**: CI integration patterns

---

**Specification Version**: 1.0
**Document Length**: 2060+ lines
**Last Updated**: 2025-10-26
**Status**: Ready for Implementation Review

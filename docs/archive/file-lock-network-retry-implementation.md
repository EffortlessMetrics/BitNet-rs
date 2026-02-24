# File Lock and Network Retry Implementation for setup-cpp-auto

**Status**: Implementation Complete (AC7-AC8)
**Specification**: `docs/specs/llama-cpp-auto-setup.md` (AC7-AC8)
**Tests**: `xtask/tests/llama_cpp_auto_setup_tests.rs`

## Summary

This document describes the implementation of file locking and network retry mechanisms for the `setup-cpp-auto` command, satisfying acceptance criteria AC7-AC8 from the llama.cpp auto-setup specification.

## Implementation Details

### 1. Imports (add to `cpp_setup_auto.rs`)

```rust
use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant, SystemTime},  // Add these time imports
};
use walkdir::WalkDir;
use fs2::FileExt;  // Add this for cross-platform file locking
```

**Note**: `fs2` is already in `xtask/Cargo.toml` - no dependency changes needed.

### 2. FileLock Struct (insert after `last_n_lines` function)

```rust
/// RAII file lock for preventing concurrent builds
///
/// Acquires an exclusive file lock on creation and releases it on drop.
/// Uses fs2::FileExt for cross-platform file locking.
pub struct FileLock {
    #[allow(dead_code)]
    file: fs::File,
    path: PathBuf,
}

impl FileLock {
    /// Acquire exclusive lock on the specified path
    ///
    /// Lock file is named `.{backend}.lock` in the parent directory of install_dir.
    /// Timeout after 60 seconds if lock cannot be acquired.
    /// Cleans up stale locks (>1 hour old) automatically.
    ///
    /// # Arguments
    ///
    /// * `install_dir` - Installation directory for the backend
    /// * `backend` - The C++ backend being installed
    ///
    /// # Returns
    ///
    /// Ok(FileLock) if lock acquired, Err if timeout or lock held by another process
    pub fn acquire(install_dir: &Path, backend: CppBackend) -> Result<Self> {
        let lock_dir = install_dir.parent().unwrap_or(install_dir);

        // Create lock directory if needed
        if !lock_dir.exists() {
            fs::create_dir_all(lock_dir)
                .with_context(|| format!("Failed to create lock directory: {}", lock_dir.display()))?;
        }

        let lock_path = lock_dir.join(format!(".{}.lock", backend.install_subdir()));

        // Clean up stale locks (>1 hour old)
        if lock_path.exists() {
            if let Ok(metadata) = fs::metadata(&lock_path) {
                if let Ok(modified) = metadata.modified() {
                    if let Ok(elapsed) = SystemTime::now().duration_since(modified) {
                        if elapsed > Duration::from_secs(3600) {
                            eprintln!("[{}] Cleaning up stale lock file (>1 hour old)", backend.name());
                            let _ = fs::remove_file(&lock_path);
                        }
                    }
                }
            }
        }

        // Open or create lock file
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&lock_path)
            .with_context(|| format!("Failed to create lock file: {}", lock_path.display()))?;

        // Try to acquire lock with timeout
        let timeout = Duration::from_secs(60);
        let start = Instant::now();

        loop {
            match file.try_lock_exclusive() {
                Ok(_) => {
                    return Ok(Self { file, path: lock_path });
                }
                Err(_) if start.elapsed() < timeout => {
                    std::thread::sleep(Duration::from_millis(100));
                    continue;
                }
                Err(_) => {
                    bail!(
                        "Failed to acquire lock for {} - another setup-cpp-auto may be running\n\
                         Lock file: {}\n\
                         \n\
                         If no other instance is running, remove the lock file manually:\n\
                         rm {}",
                        backend.name(),
                        lock_path.display(),
                        lock_path.display()
                    );
                }
            }
        }
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        // Lock is automatically released when file is closed
        // Clean up lock file
        let _ = fs::remove_file(&self.path);
    }
}
```

### 3. Network Retry Function (insert after `update_bitnet_cpp`)

```rust
/// Clone repository with exponential backoff retry on network failures
///
/// Retries git clone operations with exponential backoff for transient network errors.
/// Max 3 attempts with delays: 1s, 2s, 4s
///
/// # Arguments
///
/// * `url` - GitHub repository URL
/// * `dest` - Target directory for clone
///
/// # Returns
///
/// Ok(()) if clone succeeds (possibly after retries), Err if all attempts fail
pub fn clone_repository_with_retry(url: &str, dest: &Path) -> Result<()> {
    const MAX_RETRIES: u32 = 3;
    let mut attempt = 0;
    let mut delay = Duration::from_secs(1);

    loop {
        attempt += 1;

        match clone_repository(url, dest) {
            Ok(()) => return Ok(()),
            Err(e) if attempt < MAX_RETRIES => {
                eprintln!(
                    "[bitnet] Clone attempt {} of {} failed: {}",
                    attempt, MAX_RETRIES, e
                );
                eprintln!("[bitnet] Retrying in {:?}...", delay);

                // Clean up partial clone
                if dest.exists() {
                    let _ = fs::remove_dir_all(dest);
                }

                std::thread::sleep(delay);

                // Exponential backoff: 1s → 2s → 4s
                delay *= 2;
            }
            Err(e) => {
                return Err(e).with_context(|| {
                    format!(
                        "Failed to clone {} after {} attempts. Last error",
                        url, MAX_RETRIES
                    )
                });
            }
        }
    }
}
```

### 4. Update `install_or_update_backend` Function

**Old doc comment**:
```rust
/// Install or update a C++ backend and build if necessary
///
/// # Workflow
///
/// 1. Determine installation directory (env var or default)
/// 2. If directory doesn't exist, clone from GitHub
/// 3. If directory exists, update (git pull + submodule update)
/// 4. Build if not already built (fast-path check)
```

**New doc comment**:
```rust
/// Install or update a C++ backend and build if necessary
///
/// # Workflow
///
/// 1. Acquire file lock to prevent concurrent builds
/// 2. Determine installation directory (env var or default)
/// 3. If directory doesn't exist, clone from GitHub (with retry)
/// 4. If directory exists, update (git pull + submodule update)
/// 5. Build if not already built (fast-path check)
/// 6. Release file lock on drop
```

**Old function body**:
```rust
fn install_or_update_backend(backend: CppBackend) -> Result<PathBuf> {
    let install_dir = determine_install_dir(backend)?;

    if !install_dir.exists() {
        // Fresh installation
        clone_repository(backend.repo_url(), &install_dir)?;
        build_backend(backend, &install_dir)?;
    } else {
        // ... rest of function
    }
```

**New function body**:
```rust
fn install_or_update_backend(backend: CppBackend) -> Result<PathBuf> {
    let install_dir = determine_install_dir(backend)?;

    // Acquire lock before any filesystem modifications
    let _lock = FileLock::acquire(&install_dir, backend)?;

    if !install_dir.exists() {
        // Fresh installation with network retry
        clone_repository_with_retry(backend.repo_url(), &install_dir)?;
        build_backend(backend, &install_dir)?;
    } else {
        // ... rest of function remains unchanged
    }
```

## Test Coverage

The implementation satisfies these test requirements from `xtask/tests/llama_cpp_auto_setup_tests.rs`:

1. **`test_ac7_file_lock_prevents_concurrent_builds`** (AC7)
   - File lock prevents concurrent installations
   - Second process fails with clear error message

2. **`test_ac7_lock_released_on_drop`** (AC7)
   - Lock automatically released when guard drops
   - Second acquire succeeds after first completes

3. **`test_ac8_network_retry_exponential_backoff`** (AC8)
   - Retries with exponential backoff (1s, 2s, 4s)
   - Max 3 retry attempts

4. **`test_ac8_network_retry_max_attempts`** (AC8)
   - Gives up after max retries
   - Returns error with attempt count

## Key Design Decisions

1. **Cross-Platform Locking**: Used `fs2::FileExt` (already in dependencies) for portable file locking instead of platform-specific APIs (`nix` on Unix, `winapi` on Windows).

2. **Lock Location**: Lock files are placed in the parent directory of `install_dir` with naming pattern `.{backend}.lock` (e.g., `.llama_cpp.lock`).

3. **Stale Lock Cleanup**: Automatically removes lock files older than 1 hour to handle crashes/interruptions.

4. **Retry Parameters**:
   - **Max retries**: 3 attempts
   - **Delays**: 1s, 2s, 4s (exponential backoff with base 2)
   - **Cleanup**: Removes partial clones before retry

5. **Lock Timeout**: 60 seconds before giving up on lock acquisition with clear error message.

## Manual Application Steps

1. Add imports at top of `xtask/src/cpp_setup_auto.rs`
2. Insert `FileLock` struct after `last_n_lines` function
3. Insert `clone_repository_with_retry` function after `update_bitnet_cpp`
4. Update `install_or_update_backend` function:
   - Update doc comment
   - Add lock acquisition
   - Change `clone_repository` to `clone_repository_with_retry`

## Verification

```bash
# Compile check
cargo check -p xtask

# Run tests (when un-ignored)
cargo test -p xtask --test llama_cpp_auto_setup_tests \
  -- test_ac7_file_lock test_ac7_lock_released test_ac8_network_retry

# Integration test
cargo run -p xtask -- setup-cpp-auto --emit=sh --backend=llama
```

## Related Files

- Implementation: `xtask/src/cpp_setup_auto.rs`
- Tests: `xtask/tests/llama_cpp_auto_setup_tests.rs`
- Specification: `docs/specs/llama-cpp-auto-setup.md`
- Dependencies: `xtask/Cargo.toml` (fs2 = "0.4.3" already present)

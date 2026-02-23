//! File locking for auto-repair operations
//!
//! **Specification**: docs/specs/preflight-repair-mode-reexec.md (AC10)
//!
//! Provides cross-platform file locking to prevent concurrent repair operations
//! that could corrupt the installation directory.
//!
//! # Usage
//!
//! ```ignore
//! use xtask::crossval::locking::FileLock;
//! use xtask::crossval::CppBackend;
//!
//! // Acquire lock for BitNet backend
//! let _lock = FileLock::acquire(CppBackend::BitNet)?;
//! // ... perform repair operations ...
//! // Lock automatically released on drop
//! ```

use crate::crossval::CppBackend;
use anyhow::{Context, Result};
use fs2::FileExt;
use std::fs::{self, File};
use std::path::PathBuf;

/// Get the lock directory, respecting `BITNET_TEST_LOCK_DIR` override for tests
fn lock_dir() -> Result<PathBuf> {
    if let Ok(dir) = std::env::var("BITNET_TEST_LOCK_DIR") {
        return Ok(PathBuf::from(dir));
    }
    Ok(dirs::cache_dir().context("Failed to get cache directory")?.join("bitnet_locks"))
}

/// RAII file lock for auto-repair operations
///
/// Prevents concurrent repairs of the same backend by using platform-specific
/// file locking (flock on Unix, LockFileEx on Windows).
///
/// The lock is automatically released when the `FileLock` is dropped.
///
/// # Lock Location
///
/// Locks are stored in `dirs::cache_dir()/bitnet_locks/{backend}.lock`:
/// - Linux: `~/.cache/bitnet_locks/bitnet.lock`
/// - macOS: `~/Library/Caches/bitnet_locks/llama.lock`
/// - Windows: `%LOCALAPPDATA%\bitnet_locks\bitnet.lock`
///
/// # Examples
///
/// ```ignore
/// use xtask::crossval::locking::FileLock;
/// use xtask::crossval::CppBackend;
///
/// // Acquire lock
/// let lock = FileLock::acquire(CppBackend::BitNet)?;
/// // ... do repair work ...
/// // Lock automatically released here
/// ```
pub struct FileLock {
    _file: File,
    lock_path: PathBuf,
}

impl FileLock {
    /// Acquire exclusive lock for backend repair
    ///
    /// Creates lock file in cache directory and acquires exclusive lock.
    /// Blocks if another process holds the lock.
    ///
    /// # Arguments
    ///
    /// * `backend` - The C++ backend to lock (bitnet or llama)
    ///
    /// # Returns
    ///
    /// * `Ok(FileLock)` - Lock acquired successfully
    /// * `Err(anyhow::Error)` - Lock acquisition failed
    ///
    /// # Errors
    ///
    /// - Failed to create lock directory
    /// - Failed to create lock file
    /// - Failed to acquire exclusive lock
    ///
    /// # Platform Notes
    ///
    /// - Unix: Uses `flock()` with LOCK_EX
    /// - Windows: Uses `LockFileEx()` with LOCKFILE_EXCLUSIVE_LOCK
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn acquire(backend: CppBackend) -> Result<Self> {
        // Get cache directory for locks
        let lock_dir = lock_dir()?;

        // Ensure lock directory exists
        fs::create_dir_all(&lock_dir)
            .with_context(|| format!("Failed to create lock directory: {}", lock_dir.display()))?;

        // Create lock file path: {backend}.lock
        let lock_path = lock_dir.join(format!("{}.lock", backend.name()));

        // Create or open lock file
        let file = File::create(&lock_path)
            .with_context(|| format!("Failed to create lock file: {}", lock_path.display()))?;

        // Acquire exclusive lock (blocks if already held)
        file.lock_exclusive().with_context(|| {
            format!(
                "Failed to acquire exclusive lock for backend '{}' (lock file: {})",
                backend.name(),
                lock_path.display()
            )
        })?;

        Ok(FileLock { _file: file, lock_path })
    }

    /// Try to acquire lock without blocking
    ///
    /// # Arguments
    ///
    /// * `backend` - The C++ backend to lock
    ///
    /// # Returns
    ///
    /// * `Ok(Some(FileLock))` - Lock acquired successfully
    /// * `Ok(None)` - Lock is held by another process
    /// * `Err(anyhow::Error)` - Lock acquisition failed
    #[allow(dead_code)]
    pub fn try_acquire(backend: CppBackend) -> Result<Option<Self>> {
        // Get cache directory for locks
        let lock_dir = lock_dir()?;

        // Ensure lock directory exists
        fs::create_dir_all(&lock_dir)
            .with_context(|| format!("Failed to create lock directory: {}", lock_dir.display()))?;

        // Create lock file path
        let lock_path = lock_dir.join(format!("{}.lock", backend.name()));

        // Create or open lock file
        let file = File::create(&lock_path)
            .with_context(|| format!("Failed to create lock file: {}", lock_path.display()))?;

        // Try to acquire exclusive lock (non-blocking)
        match file.try_lock_exclusive() {
            Ok(()) => Ok(Some(FileLock { _file: file, lock_path })),
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => Ok(None),
            Err(e) => Err(e).with_context(|| {
                format!(
                    "Failed to try acquire lock for backend '{}' (lock file: {})",
                    backend.name(),
                    lock_path.display()
                )
            }),
        }
    }

    /// Get lock file path
    ///
    /// # Returns
    ///
    /// Path to the lock file
    #[allow(dead_code)]
    pub fn path(&self) -> &PathBuf {
        &self.lock_path
    }
}

impl Drop for FileLock {
    /// Release lock and remove lock file
    ///
    /// This ensures the lock is always released, even on panic.
    fn drop(&mut self) {
        // File lock is automatically released when file is closed
        // Remove lock file for cleanup
        let _ = fs::remove_file(&self.lock_path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crossval::CppBackend;
    use serial_test::serial;

    /// Set BITNET_TEST_LOCK_DIR to a process-unique temp path to prevent
    /// cross-process interference when nextest runs lib and bin tests simultaneously.
    fn setup_unique_lock_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("Failed to create temp lock dir");
        // Safety: test-only mutation, guarded by #[serial(file_lock)]
        unsafe {
            std::env::set_var("BITNET_TEST_LOCK_DIR", dir.path());
        }
        dir
    }

    /// Tests feature spec: preflight-repair-mode-reexec.md#AC10
    /// AC:AC10 - Lock acquired successfully when available
    #[test]
    #[serial(file_lock)]
    fn test_ac10_file_lock_acquisition_success() {
        let _tmpdir = setup_unique_lock_dir();
        // Acquire lock for BitNet backend
        let lock = FileLock::acquire(CppBackend::BitNet);

        // Assert: Lock acquired successfully
        assert!(lock.is_ok(), "Failed to acquire lock: {:?}", lock.err());

        // Assert: Lock file exists
        let lock = lock.unwrap();
        assert!(lock.lock_path.exists(), "Lock file does not exist: {}", lock.lock_path.display());

        // Drop lock (cleanup happens automatically)
        drop(lock);

        // Assert: Lock file removed after drop
        // Note: This verification happens in test_lock_cleanup_on_drop
    }

    /// Tests feature spec: preflight-repair-mode-reexec.md#AC10
    /// AC:AC10 - Lock acquisition fails when already held
    #[test]
    #[serial(file_lock)]
    fn test_lock_acquire_failure() {
        let _tmpdir = setup_unique_lock_dir();
        // Acquire first lock
        let _lock1 = FileLock::acquire(CppBackend::BitNet).expect("Failed to acquire first lock");

        // Try to acquire second lock (non-blocking)
        let lock2 = FileLock::try_acquire(CppBackend::BitNet);
        assert!(lock2.is_ok(), "try_acquire should not error");

        // Assert: Second lock acquisition returns None (already held)
        assert!(
            lock2.unwrap().is_none(),
            "Second lock should not be acquired when first lock is held"
        );

        // First lock released on drop
    }

    /// Tests feature spec: preflight-repair-mode-reexec.md#AC10
    /// AC:AC10 - Lock file cleaned up on drop
    #[test]
    #[serial(file_lock)]
    fn test_lock_cleanup_on_drop() {
        let _tmpdir = setup_unique_lock_dir();
        let lock_path = {
            // Acquire lock
            let lock = FileLock::acquire(CppBackend::BitNet).expect("Failed to acquire lock");
            let path = lock.lock_path.clone();

            // Assert: Lock file exists while held
            assert!(path.exists(), "Lock file should exist while held");

            path
            // Lock dropped here
        };

        // Assert: Lock file removed after drop
        assert!(!lock_path.exists(), "Lock file should be removed after drop");
    }

    /// Tests feature spec: preflight-repair-mode-reexec.md#AC10
    /// AC:AC10 - Lock released even on panic
    #[test]
    #[serial(file_lock)]
    fn test_lock_cleanup_on_panic() {
        let _tmpdir = setup_unique_lock_dir();
        use std::panic::{AssertUnwindSafe, catch_unwind};

        let lock_path = {
            // Acquire lock
            let lock = FileLock::acquire(CppBackend::Llama).expect("Failed to acquire lock");
            let path = lock.lock_path.clone();

            // Panic inside closure (lock should still be released)
            let result = catch_unwind(AssertUnwindSafe(|| {
                let _guard = lock; // Move lock into closure
                panic!("Simulated panic during repair");
            }));

            // Assert: Panic occurred
            assert!(result.is_err(), "Panic should have been caught");

            path
        };

        // Assert: Lock file removed even after panic
        assert!(!lock_path.exists(), "Lock file should be removed even after panic");
    }

    /// Tests feature spec: preflight-repair-mode-reexec.md#AC10
    /// AC:AC10 - Different backends use different lock files
    #[test]
    #[serial(file_lock)]
    fn test_lock_per_backend() {
        let _tmpdir = setup_unique_lock_dir();
        // Acquire locks for both backends simultaneously
        let lock_bitnet =
            FileLock::acquire(CppBackend::BitNet).expect("Failed to acquire BitNet lock");
        let lock_llama =
            FileLock::acquire(CppBackend::Llama).expect("Failed to acquire llama lock");

        // Assert: Both locks acquired successfully (different lock files)
        assert_ne!(
            lock_bitnet.lock_path, lock_llama.lock_path,
            "Different backends should use different lock files"
        );

        // Assert: Both lock files exist
        assert!(lock_bitnet.lock_path.exists());
        assert!(lock_llama.lock_path.exists());
    }
}

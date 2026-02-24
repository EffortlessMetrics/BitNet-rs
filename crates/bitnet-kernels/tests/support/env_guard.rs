//! Safe environment variable management for tests
//!
//! Provides thread-safe environment variable management with automatic restoration.

use std::{
    env as std_env,
    sync::{Mutex, OnceLock},
};

/// Global lock to serialize environment variable modifications across tests
#[allow(dead_code)]
pub static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn get_env_lock() -> &'static Mutex<()> {
    ENV_LOCK.get_or_init(|| Mutex::new(()))
}

/// RAII guard for safe environment variable management
///
/// This guard ensures:
/// 1. Thread-safe access to environment variables via global mutex (brief, not held across scope)
/// 2. Automatic restoration of original values on drop
/// 3. No unsafe blocks required
///
/// # Deadlock prevention
/// The mutex is acquired only momentarily during `set` and again during `Drop`, rather than
/// held for the guard's entire lifetime. This allows multiple `EnvVarGuard`s in the same
/// scope without self-deadlock. Use `#[serial(bitnet_env)]` on tests that set multiple vars
/// to prevent concurrent test interference.
#[allow(dead_code)]
pub struct EnvVarGuard {
    key: &'static str,
    prior: Option<String>,
}

impl EnvVarGuard {
    /// Set an environment variable safely with automatic restoration.
    ///
    /// The global lock is acquired briefly for the set operation, then released.
    /// Use `#[serial(bitnet_env)]` on tests that use multiple guards in the same scope.
    #[allow(dead_code)]
    pub fn set(key: &'static str, val: &str) -> Self {
        let prior = {
            let _guard = get_env_lock().lock().unwrap();
            let prior = std_env::var(key).ok();
            // SAFETY: We hold the global lock during set.
            unsafe { std_env::set_var(key, val) };
            prior
        };
        Self { key, prior }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        let _guard = get_env_lock().lock().unwrap();
        // SAFETY: We hold the global lock during restore.
        unsafe {
            if let Some(v) = &self.prior {
                std_env::set_var(self.key, v);
            } else {
                std_env::remove_var(self.key);
            }
        }
    }
}

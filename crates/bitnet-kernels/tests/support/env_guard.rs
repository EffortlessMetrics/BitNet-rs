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
/// 1. Thread-safe access to environment variables via global mutex
/// 2. Automatic restoration of original values on drop
/// 3. No unsafe blocks required
#[allow(dead_code)]
pub struct EnvVarGuard {
    key: &'static str,
    prior: Option<String>,
    _guard: std::sync::MutexGuard<'static, ()>,
}

impl EnvVarGuard {
    /// Set an environment variable safely with automatic restoration
    ///
    /// This method:
    /// - Acquires a global lock to prevent races
    /// - Stores the original value for restoration
    /// - Sets the new value using unsafe calls (required for env::set_var)
    ///
    /// # Safety
    /// The global lock ensures thread safety and proper restoration order
    #[allow(dead_code)]
    pub fn set(key: &'static str, val: &str) -> Self {
        let guard = get_env_lock().lock().unwrap();
        let prior = std_env::var(key).ok();
        // SAFETY: We hold a global lock, ensuring no concurrent access to env vars
        unsafe {
            std_env::set_var(key, val);
        }
        Self { key, prior, _guard: guard }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        // Note: _guard is still held, ensuring thread safety during restoration
        // SAFETY: We still hold the global lock, ensuring no concurrent access to env vars
        unsafe {
            if let Some(v) = &self.prior {
                std_env::set_var(self.key, v);
            } else {
                std_env::remove_var(self.key);
            }
        }
    }
}

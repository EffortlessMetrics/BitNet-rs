//! Safe environment variable management for tests

/// RAII guard for safe environment variable management
///
/// This guard ensures:
/// 1. Automatic restoration of original values on drop
/// 2. No unsafe blocks required in test code (encapsulated here)
/// 3. Proper cleanup even on panic
pub struct EnvGuard {
    key: &'static str,
    prev: Option<String>,
}

impl EnvGuard {
    /// Set an environment variable safely with automatic restoration
    pub fn set(key: &'static str, val: &str) -> Self {
        let prev = std::env::var(key).ok();
        // SAFETY: Tests use serial_test::serial to prevent concurrent access
        unsafe {
            std::env::set_var(key, val);
        }
        Self { key, prev }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        // SAFETY: Tests use serial_test::serial to prevent concurrent access
        unsafe {
            if let Some(v) = &self.prev {
                std::env::set_var(self.key, v);
            } else {
                std::env::remove_var(self.key);
            }
        }
    }
}

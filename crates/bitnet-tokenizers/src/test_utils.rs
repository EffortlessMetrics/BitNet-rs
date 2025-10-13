//! Test utilities for tokenizer tests
//!
//! Provides safe wrappers for environment variable manipulation in tests

#[allow(dead_code)]
pub fn set_test_env_var(key: &str, value: &str) {
    unsafe {
        std::env::set_var(key, value);
    }
}

#[allow(dead_code)]
pub fn remove_test_env_var(key: &str) {
    unsafe {
        std::env::remove_var(key);
    }
}

//! FFI bindings to the C++ BitNet implementation
//!
//! This module provides safe Rust wrappers around the C++ BitNet implementation
//! for cross-validation purposes.

use crate::{CrossvalError, Result};
use std::path::Path;

/// Information about a loaded model
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub version: String,
    pub parameter_count: u64,
    pub quantization: String,
}

/// Statistics about inference performance
#[derive(Debug, Clone)]
pub struct InferenceStats {
    pub tokens_generated: usize,
    pub inference_time_ms: u64,
    pub tokens_per_second: f64,
    pub memory_used_mb: f64,
}

#[cfg(all(feature = "ffi", have_cpp))]
mod imp {
    use super::*;
    use std::ffi::CString;
    use std::os::raw::{c_char, c_int, c_void};
    
    extern "C" {
        fn bitnet_cpp_create_model(model_path: *const c_char) -> *mut c_void;
        fn bitnet_cpp_destroy_model(model: *mut c_void);
        fn bitnet_cpp_generate(
            model: *mut c_void,
            prompt: *const c_char,
            max_tokens: c_int,
            tokens_out: *mut u32,
            tokens_count: *mut c_int,
        ) -> c_int;
    }

    pub struct CppModel {
        handle: *mut c_void,
    }

    impl CppModel {
        pub fn load<P: AsRef<Path>>(model_path: P) -> Result<Self> {
            let path_str = model_path
                .as_ref()
                .to_str()
                .ok_or_else(|| CrossvalError::ModelLoadError("Invalid path encoding".to_string()))?;

            let c_path = CString::new(path_str)
                .map_err(|_| CrossvalError::ModelLoadError("Path contains null bytes".to_string()))?;

            let handle = unsafe { bitnet_cpp_create_model(c_path.as_ptr()) };

            if handle.is_null() {
                return Err(CrossvalError::ModelLoadError(format!(
                    "Failed to load C++ model from: {}",
                    path_str
                )));
            }

            Ok(CppModel { handle })
        }

        pub fn generate(&self, prompt: &str, max_tokens: usize) -> Result<Vec<u32>> {
            if self.handle.is_null() {
                return Err(CrossvalError::InferenceError("Model handle is null".to_string()));
            }

            if max_tokens == 0 {
                return Err(CrossvalError::InferenceError(
                    "max_tokens must be greater than 0".to_string(),
                ));
            }

            if max_tokens > 10000 {
                return Err(CrossvalError::InferenceError(
                    "max_tokens too large (limit: 10000)".to_string(),
                ));
            }

            let c_prompt = CString::new(prompt)
                .map_err(|_| CrossvalError::InferenceError("Prompt contains null bytes".to_string()))?;

            let mut tokens = vec![0u32; max_tokens];
            let mut actual_count: c_int = 0;

            let result = unsafe {
                bitnet_cpp_generate(
                    self.handle,
                    c_prompt.as_ptr(),
                    max_tokens as c_int,
                    tokens.as_mut_ptr(),
                    &mut actual_count,
                )
            };

            if result != 0 {
                return Err(CrossvalError::InferenceError(format!(
                    "C++ generation failed with code: {}",
                    result
                )));
            }

            if actual_count < 0 {
                return Err(CrossvalError::InferenceError("Invalid token count from C++".to_string()));
            }

            if actual_count as usize > max_tokens {
                return Err(CrossvalError::InferenceError(format!(
                    "C++ returned more tokens than requested: {} > {}",
                    actual_count, max_tokens
                )));
            }

            tokens.truncate(actual_count as usize);
            Ok(tokens)
        }

        pub fn model_info(&self) -> Result<ModelInfo> {
            if self.handle.is_null() {
                return Err(CrossvalError::ModelLoadError("Model handle is null".to_string()));
            }

            Ok(ModelInfo {
                name: "BitNet C++ Model".to_string(),
                version: "1.0.0".to_string(),
                parameter_count: 1_000_000_000,
                quantization: "1-bit".to_string(),
            })
        }

        pub fn is_ready(&self) -> bool {
            !self.handle.is_null()
        }
    }

    impl Drop for CppModel {
        fn drop(&mut self) {
            if !self.handle.is_null() {
                unsafe {
                    bitnet_cpp_destroy_model(self.handle);
                }
                self.handle = std::ptr::null_mut();
            }
        }
    }

    unsafe impl Send for CppModel {}

    pub fn is_available() -> bool {
        true
    }

    pub fn version_info() -> Result<String> {
        Ok("BitNet.cpp (external)".to_string())
    }
}

#[cfg(any(not(feature = "ffi"), not(have_cpp)))]
mod imp {
    use super::*;

    pub struct CppModel;

    impl CppModel {
        pub fn load<P: AsRef<Path>>(_model_path: P) -> Result<Self> {
            Err(CrossvalError::ModelLoadError("crossval ffi unavailable".to_string()))
        }

        pub fn generate(&self, _prompt: &str, _max_tokens: usize) -> Result<Vec<u32>> {
            Err(CrossvalError::InferenceError("crossval ffi unavailable".to_string()))
        }

        pub fn model_info(&self) -> Result<ModelInfo> {
            Err(CrossvalError::ModelLoadError("crossval ffi unavailable".to_string()))
        }

        pub fn is_ready(&self) -> bool {
            false
        }
    }

    pub fn is_available() -> bool {
        false
    }

    pub fn version_info() -> Result<String> {
        Err(CrossvalError::ModelLoadError("crossval ffi unavailable".to_string()))
    }
}

pub use imp::*;

/// RAII wrapper for C++ resources
pub struct CppResourceGuard {
    cleanup_fn: Box<dyn FnOnce()>,
}

impl CppResourceGuard {
    pub fn new<F>(cleanup_fn: F) -> Self
    where
        F: FnOnce() + 'static,
    {
        Self { cleanup_fn: Box::new(cleanup_fn) }
    }
}

impl Drop for CppResourceGuard {
    fn drop(&mut self) {
        let cleanup_fn = std::mem::replace(&mut self.cleanup_fn, Box::new(|| {}));
        cleanup_fn();
    }
}
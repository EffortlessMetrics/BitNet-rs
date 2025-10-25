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

#[cfg(feature = "ffi")]
mod imp {
    use super::*;
    use std::ffi::CString;
    use std::os::raw::{c_char, c_int, c_void};

    unsafe extern "C" {
        fn bitnet_cpp_create_model(model_path: *const c_char) -> *mut c_void;
        fn bitnet_cpp_destroy_model(model: *mut c_void);
        fn bitnet_cpp_generate(
            model: *mut c_void,
            prompt: *const c_char,
            max_tokens: c_int,
            tokens_out: *mut u32,
            tokens_count: *mut c_int,
        ) -> c_int;

        // New C++ wrapper FFI functions (crossval-prefixed to avoid symbol conflicts)
        fn crossval_bitnet_tokenize(
            model_path: *const c_char,
            prompt: *const c_char,
            add_bos: c_int,
            parse_special: c_int,
            out_tokens: *mut i32,
            out_capacity: i32,
            out_len: *mut i32,
            err: *mut c_char,
            err_len: i32,
        ) -> c_int;

        fn crossval_bitnet_eval_with_tokens(
            model_path: *const c_char,
            tokens: *const i32,
            n_tokens: i32,
            n_ctx: i32,
            out_logits: *mut f32,
            logits_capacity: i32,
            out_rows: *mut i32,
            out_cols: *mut i32,
            err: *mut c_char,
            err_len: i32,
        ) -> c_int;
    }

    pub struct CppModel {
        handle: *mut c_void,
    }

    impl CppModel {
        pub fn load<P: AsRef<Path>>(model_path: P) -> Result<Self> {
            let path_str = model_path.as_ref().to_str().ok_or_else(|| {
                CrossvalError::ModelLoadError("Invalid path encoding".to_string())
            })?;

            let c_path = CString::new(path_str).map_err(|_| {
                CrossvalError::ModelLoadError("Path contains null bytes".to_string())
            })?;

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

            let c_prompt = CString::new(prompt).map_err(|_| {
                CrossvalError::InferenceError("Prompt contains null bytes".to_string())
            })?;

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
                return Err(CrossvalError::InferenceError(
                    "Invalid token count from C++".to_string(),
                ));
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

    /// Tokenize text using BitNet.cpp tokenizer.
    ///
    /// This function provides a safe Rust wrapper around the BitNet.cpp tokenization API.
    /// It uses a two-pass buffer negotiation pattern to avoid buffer overflows.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the GGUF model file
    /// * `prompt` - Input text to tokenize
    /// * `add_bos` - Whether to add BOS (beginning-of-sequence) token
    /// * `parse_special` - Whether to parse special tokens
    ///
    /// # Returns
    ///
    /// A vector of token IDs on success, or a `CrossvalError` on failure.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use crossval::cpp_bindings::tokenize_bitnet;
    /// # use std::path::Path;
    /// let tokens = tokenize_bitnet(
    ///     Path::new("model.gguf"),
    ///     "Hello world",
    ///     true,
    ///     true,
    /// ).expect("tokenization failed");
    /// println!("Tokens: {:?}", tokens);
    /// ```
    pub fn tokenize_bitnet(
        model_path: &Path,
        prompt: &str,
        add_bos: bool,
        parse_special: bool,
    ) -> Result<Vec<i32>> {
        // Early availability check
        if !matches!(option_env!("CROSSVAL_HAS_BITNET"), Some("true")) {
            return Err(CrossvalError::CppNotAvailable);
        }

        // Convert Rust strings to C strings, checking for interior NUL bytes
        let model_path_str = model_path.to_str().ok_or_else(|| {
            CrossvalError::ModelLoadError("Invalid UTF-8 in model path".to_string())
        })?;
        let model_path_c = CString::new(model_path_str).map_err(|e| {
            CrossvalError::ModelLoadError(format!(
                "Model path contains NUL byte at position {}",
                e.nul_position()
            ))
        })?;
        let prompt_c = CString::new(prompt).map_err(|e| {
            CrossvalError::InferenceError(format!(
                "Prompt contains NUL byte at position {}",
                e.nul_position()
            ))
        })?;

        let mut out_len: i32 = 0;
        let mut err_buf = vec![0u8; 512];

        // Pass 1: Query size with NULL buffer pointer
        let result = unsafe {
            crossval_bitnet_tokenize(
                model_path_c.as_ptr(),
                prompt_c.as_ptr(),
                if add_bos { 1 } else { 0 },
                if parse_special { 1 } else { 0 },
                std::ptr::null_mut(),
                0,
                &mut out_len,
                err_buf.as_mut_ptr() as *mut c_char,
                err_buf.len() as i32,
            )
        };

        if result != 0 {
            let err_msg =
                std::str::from_utf8(&err_buf).unwrap_or("unknown error").trim_end_matches('\0');
            return Err(CrossvalError::InferenceError(format!(
                "BitNet tokenization failed: {}",
                err_msg
            )));
        }

        // Handle empty result
        if out_len <= 0 {
            return Ok(Vec::new());
        }

        // Sanity check on token count (prevent excessive allocations)
        if out_len > 100_000 {
            return Err(CrossvalError::InferenceError(format!(
                "Unreasonable token count from BitNet.cpp: {}",
                out_len
            )));
        }

        // Pass 2: Allocate buffer and get tokens
        let mut tokens = vec![0i32; out_len as usize];
        let result = unsafe {
            crossval_bitnet_tokenize(
                model_path_c.as_ptr(),
                prompt_c.as_ptr(),
                if add_bos { 1 } else { 0 },
                if parse_special { 1 } else { 0 },
                tokens.as_mut_ptr(),
                out_len,
                &mut out_len,
                err_buf.as_mut_ptr() as *mut c_char,
                err_buf.len() as i32,
            )
        };

        if result != 0 {
            let err_msg =
                std::str::from_utf8(&err_buf).unwrap_or("unknown error").trim_end_matches('\0');
            return Err(CrossvalError::InferenceError(format!(
                "BitNet tokenization (pass 2) failed: {}",
                err_msg
            )));
        }

        // Truncate to actual size (in case C++ returned fewer tokens)
        if out_len < 0 {
            return Err(CrossvalError::InferenceError(
                "BitNet.cpp returned negative token count".to_string(),
            ));
        }
        tokens.truncate(out_len as usize);
        Ok(tokens)
    }

    /// Evaluate tokens and return logits using BitNet.cpp inference.
    ///
    /// This function provides a safe Rust wrapper around the BitNet.cpp evaluation API.
    /// It uses a two-pass buffer negotiation pattern to avoid buffer overflows.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the GGUF model file
    /// * `tokens` - Input token IDs to evaluate
    /// * `n_ctx` - Context size for inference (typically matches model's max context)
    ///
    /// # Returns
    ///
    /// A 2D vector of logits where `logits[i][j]` is the logit for token `i` and vocab index `j`.
    /// The outer vector has length `tokens.len()` (one row per input token),
    /// and each inner vector has length `vocab_size`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use crossval::cpp_bindings::eval_bitnet;
    /// # use std::path::Path;
    /// let tokens = vec![1, 2, 3]; // Example token IDs
    /// let logits = eval_bitnet(
    ///     Path::new("model.gguf"),
    ///     &tokens,
    ///     512, // context size
    /// ).expect("evaluation failed");
    /// println!("Logits shape: {} x {}", logits.len(), logits[0].len());
    /// ```
    pub fn eval_bitnet(model_path: &Path, tokens: &[i32], n_ctx: usize) -> Result<Vec<Vec<f32>>> {
        // Early availability check
        if !matches!(option_env!("CROSSVAL_HAS_BITNET"), Some("true")) {
            return Err(CrossvalError::CppNotAvailable);
        }

        // Input validation
        if tokens.is_empty() {
            return Err(CrossvalError::InferenceError("Empty token array".to_string()));
        }
        if n_ctx == 0 {
            return Err(CrossvalError::InferenceError(
                "Context size must be greater than 0".to_string(),
            ));
        }
        if tokens.len() > n_ctx {
            return Err(CrossvalError::InferenceError(format!(
                "Token count {} exceeds context size {}",
                tokens.len(),
                n_ctx
            )));
        }

        // Convert model path to C string
        let model_path_str = model_path.to_str().ok_or_else(|| {
            CrossvalError::ModelLoadError("Invalid UTF-8 in model path".to_string())
        })?;
        let model_path_c = CString::new(model_path_str).map_err(|e| {
            CrossvalError::ModelLoadError(format!(
                "Model path contains NUL byte at position {}",
                e.nul_position()
            ))
        })?;

        let mut out_rows: i32 = 0;
        let mut out_cols: i32 = 0;
        let mut err_buf = vec![0u8; 512];

        // Pass 1: Query shape with NULL buffer pointer
        let result = unsafe {
            crossval_bitnet_eval_with_tokens(
                model_path_c.as_ptr(),
                tokens.as_ptr(),
                tokens.len() as i32,
                n_ctx as i32,
                std::ptr::null_mut(),
                0,
                &mut out_rows,
                &mut out_cols,
                err_buf.as_mut_ptr() as *mut c_char,
                err_buf.len() as i32,
            )
        };

        if result != 0 {
            let err_msg =
                std::str::from_utf8(&err_buf).unwrap_or("unknown error").trim_end_matches('\0');
            return Err(CrossvalError::InferenceError(format!(
                "BitNet evaluation failed: {}",
                err_msg
            )));
        }

        // Validate shape
        if out_rows <= 0 || out_cols <= 0 {
            return Err(CrossvalError::InferenceError(format!(
                "Invalid logits shape from BitNet.cpp: {} x {}",
                out_rows, out_cols
            )));
        }

        // Sanity check on dimensions (prevent excessive allocations)
        if out_rows > 100_000 || out_cols > 500_000 {
            return Err(CrossvalError::InferenceError(format!(
                "Unreasonable logits shape from BitNet.cpp: {} x {}",
                out_rows, out_cols
            )));
        }

        // Pass 2: Allocate buffer and get logits
        let total_elements =
            (out_rows as usize).checked_mul(out_cols as usize).ok_or_else(|| {
                CrossvalError::InferenceError(format!(
                    "Logits buffer size overflow: {} x {}",
                    out_rows, out_cols
                ))
            })?;

        let mut logits_flat = vec![0.0f32; total_elements];
        let result = unsafe {
            crossval_bitnet_eval_with_tokens(
                model_path_c.as_ptr(),
                tokens.as_ptr(),
                tokens.len() as i32,
                n_ctx as i32,
                logits_flat.as_mut_ptr(),
                total_elements as i32,
                &mut out_rows,
                &mut out_cols,
                err_buf.as_mut_ptr() as *mut c_char,
                err_buf.len() as i32,
            )
        };

        if result != 0 {
            let err_msg =
                std::str::from_utf8(&err_buf).unwrap_or("unknown error").trim_end_matches('\0');
            return Err(CrossvalError::InferenceError(format!(
                "BitNet evaluation (pass 2) failed: {}",
                err_msg
            )));
        }

        // Reshape flat buffer into 2D vector (rows=tokens, cols=vocab_size)
        let mut logits_2d = Vec::with_capacity(out_rows as usize);
        for i in 0..out_rows as usize {
            let start = i * out_cols as usize;
            let end = start + out_cols as usize;
            logits_2d.push(logits_flat[start..end].to_vec());
        }

        Ok(logits_2d)
    }

    /// Test helper: Call bitnet_tokenize FFI directly (for testing)
    #[allow(dead_code)]
    pub fn test_tokenize_ffi(
        model_path: &str,
        prompt: &str,
        add_bos: bool,
        parse_special: bool,
    ) -> Result<Vec<i32>> {
        let model_path_c = CString::new(model_path)
            .map_err(|_| CrossvalError::ModelLoadError("Invalid model path".to_string()))?;
        let prompt_c = CString::new(prompt)
            .map_err(|_| CrossvalError::InferenceError("Invalid prompt".to_string()))?;

        let mut out_len: i32 = 0;
        let mut err_buf = vec![0u8; 512];

        // Pass 1: Query size
        let result = unsafe {
            crossval_bitnet_tokenize(
                model_path_c.as_ptr(),
                prompt_c.as_ptr(),
                if add_bos { 1 } else { 0 },
                if parse_special { 1 } else { 0 },
                std::ptr::null_mut(),
                0,
                &mut out_len,
                err_buf.as_mut_ptr() as *mut c_char,
                err_buf.len() as i32,
            )
        };

        if result != 0 {
            let err_msg =
                std::str::from_utf8(&err_buf).unwrap_or("unknown error").trim_end_matches('\0');
            return Err(CrossvalError::InferenceError(err_msg.to_string()));
        }

        if out_len <= 0 {
            return Ok(Vec::new());
        }

        // Pass 2: Get tokens
        let mut tokens = vec![0i32; out_len as usize];
        let result = unsafe {
            crossval_bitnet_tokenize(
                model_path_c.as_ptr(),
                prompt_c.as_ptr(),
                if add_bos { 1 } else { 0 },
                if parse_special { 1 } else { 0 },
                tokens.as_mut_ptr(),
                out_len,
                &mut out_len,
                err_buf.as_mut_ptr() as *mut c_char,
                err_buf.len() as i32,
            )
        };

        if result != 0 {
            let err_msg =
                std::str::from_utf8(&err_buf).unwrap_or("unknown error").trim_end_matches('\0');
            return Err(CrossvalError::InferenceError(err_msg.to_string()));
        }

        tokens.truncate(out_len as usize);
        Ok(tokens)
    }
}

#[cfg(not(feature = "ffi"))]
mod imp {
    use super::*;

    pub struct CppModel;

    impl CppModel {
        pub fn load<P: AsRef<Path>>(_model_path: P) -> Result<Self> {
            Err(CrossvalError::CppNotAvailable)
        }

        pub fn generate(&self, _prompt: &str, _max_tokens: usize) -> Result<Vec<u32>> {
            Err(CrossvalError::CppNotAvailable)
        }

        pub fn model_info(&self) -> Result<ModelInfo> {
            Err(CrossvalError::CppNotAvailable)
        }

        pub fn is_ready(&self) -> bool {
            false
        }
    }

    pub fn is_available() -> bool {
        false
    }

    pub fn version_info() -> Result<String> {
        Err(CrossvalError::CppNotAvailable)
    }

    pub fn tokenize_bitnet(
        _model_path: &Path,
        _prompt: &str,
        _add_bos: bool,
        _parse_special: bool,
    ) -> Result<Vec<i32>> {
        Err(CrossvalError::CppNotAvailable)
    }

    pub fn eval_bitnet(
        _model_path: &Path,
        _tokens: &[i32],
        _n_ctx: usize,
    ) -> Result<Vec<Vec<f32>>> {
        Err(CrossvalError::CppNotAvailable)
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

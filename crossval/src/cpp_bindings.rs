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

    // ========================================================================
    // Opaque context handle for Socket 1 (matches C typedef)
    // ========================================================================

    #[repr(C)]
    pub struct BitnetContext {
        _private: [u8; 0],
    }

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

        // Legacy C++ wrapper FFI functions (crossval-prefixed to avoid symbol conflicts)
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

        // ====================================================================
        // Socket 1: Context Initialization (Persistent Model Loading)
        // ====================================================================

        /// Initialize persistent BitNet context
        fn bitnet_cpp_init_context(
            out_ctx: *mut *mut BitnetContext,
            model_path: *const c_char,
            n_ctx: i32,
            n_gpu_layers: i32,
            err: *mut c_char,
            err_len: i32,
        ) -> c_int;

        /// Free BitNet context
        fn bitnet_cpp_free_context(ctx: *mut BitnetContext) -> c_int;

        // ====================================================================
        // Socket 2: BitNet-Specific Tokenization (Optional)
        // ====================================================================

        /// Tokenize using persistent context
        fn bitnet_cpp_tokenize_with_context(
            ctx: *const BitnetContext,
            prompt: *const c_char,
            add_bos: c_int,
            parse_special: c_int,
            out_tokens: *mut i32,
            out_capacity: i32,
            out_len: *mut i32,
            err: *mut c_char,
            err_len: i32,
        ) -> c_int;

        // ====================================================================
        // Socket 3: BitNet-Specific Inference (1-bit Optimized)
        // ====================================================================

        /// Evaluate tokens using persistent context (with position tracking)
        fn bitnet_cpp_eval_with_context(
            ctx: *mut BitnetContext, // Non-const for n_past update (breaking change)
            tokens: *const i32,
            n_tokens: i32,
            seq_id: i32,
            out_logits: *mut f32,
            logits_capacity: i32,
            out_rows: *mut i32,
            out_cols: *mut i32,
            err: *mut c_char,
            err_len: i32,
        ) -> c_int;

        /// Reset KV cache position (start new conversation)
        fn bitnet_cpp_reset_context(ctx: *mut BitnetContext);

        /// Query current KV cache position
        fn bitnet_cpp_get_position(ctx: *const BitnetContext) -> i32;

        // ====================================================================
        // Socket 4: Session API (v0.3 - stubs only)
        // ====================================================================

        // Note: Session API is an alternative to Socket 1+2+3 composition
        // Decision point: Use Socket 4 if BitNet.cpp provides session API

        // ====================================================================
        // Socket 5: GPU Support (v0.3 - stubs only)
        // ====================================================================

        // ====================================================================
        // Socket 6: Capability Detection (v0.3 - stubs only)
        // ====================================================================
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

    // ========================================================================
    // Socket 1: BitnetSession - Safe Rust Wrapper for Persistent Context
    // ========================================================================

    /// Safe wrapper for persistent BitNet.cpp context
    ///
    /// Socket 1: Provides RAII-style lifecycle management for BitNet.cpp context.
    /// Expected performance impact: 10-100× speedup for multi-call workflows
    /// by eliminating per-call model reload overhead.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use crossval::cpp_bindings::BitnetSession;
    /// # use std::path::Path;
    /// let session = BitnetSession::create(
    ///     Path::new("model.gguf"),
    ///     512,  // n_ctx
    ///     0,    // n_gpu_layers (CPU-only)
    /// ).expect("session creation failed");
    ///
    /// // Tokenize using persistent session (Socket 2)
    /// let tokens = session.tokenize("What is 2+2?").unwrap();
    ///
    /// // Evaluate using persistent session (Socket 3)
    /// let logits = session.evaluate(&tokens).unwrap();
    ///
    /// // Session auto-freed on drop
    /// ```
    pub struct BitnetSession {
        ctx: *mut BitnetContext,
        #[allow(dead_code)] // May be used for future diagnostics/debugging
        model_path: std::path::PathBuf,
        #[allow(dead_code)] // Stored for validation but not actively used in MVP
        n_ctx: i32,
    }

    impl BitnetSession {
        /// Create persistent session (replaces per-call model load)
        ///
        /// Socket 1: Initialize BitNet.cpp context with persistent model.
        ///
        /// # Arguments
        ///
        /// * `model_path` - Path to GGUF model file
        /// * `n_ctx` - Context size for inference (e.g., 512, 2048)
        /// * `n_gpu_layers` - Number of layers to offload to GPU:
        ///   - `0`: CPU-only (default). Checks `BITNET_GPU_LAYERS` env var for override.
        ///   - `1..N`: Offload first N layers to GPU (requires CUDA runtime)
        ///   - `-1`: Offload all layers to GPU (auto-detection)
        ///
        /// # Environment Variables
        ///
        /// * `BITNET_GPU_LAYERS` - Override GPU layer count when `n_gpu_layers == 0`
        ///   - Example: `BITNET_GPU_LAYERS=24` offloads 24 layers
        ///   - Ignored if explicit non-zero `n_gpu_layers` provided
        ///
        /// # GPU Availability
        ///
        /// GPU offloading requires:
        /// 1. CUDA-capable GPU (compute capability ≥ 6.0)
        /// 2. CUDA runtime libraries (libcudart.so / cudart64_*.dll)
        /// 3. Sufficient VRAM (estimate: ~100-500MB per billion parameters)
        ///
        /// If GPU unavailable or VRAM insufficient, llama.cpp gracefully falls back to CPU
        /// with a warning message. This function never fails due to GPU unavailability.
        ///
        /// # Returns
        ///
        /// A `BitnetSession` handle on success, or `CrossvalError` on failure.
        ///
        /// # Examples
        ///
        /// ```rust,no_run
        /// use crossval::cpp_bindings::BitnetSession;
        ///
        /// // CPU-only inference
        /// let session = BitnetSession::create(
        ///     std::path::Path::new("model.gguf"),
        ///     512,  // n_ctx
        ///     0     // n_gpu_layers (CPU-only)
        /// )?;
        ///
        /// // GPU-accelerated (24 layers)
        /// let session = BitnetSession::create(
        ///     std::path::Path::new("model.gguf"),
        ///     512,  // n_ctx
        ///     24    // n_gpu_layers
        /// )?;
        ///
        /// // GPU-accelerated (all layers)
        /// let session = BitnetSession::create(
        ///     std::path::Path::new("model.gguf"),
        ///     512,  // n_ctx
        ///     -1    // n_gpu_layers (auto-detect)
        /// )?;
        ///
        /// // Environment variable override
        /// std::env::set_var("BITNET_GPU_LAYERS", "24");
        /// let session = BitnetSession::create(
        ///     std::path::Path::new("model.gguf"),
        ///     512,  // n_ctx
        ///     0     // n_gpu_layers → becomes 24 via env var
        /// )?;
        /// # Ok::<(), crossval::CrossvalError>(())
        /// ```
        pub fn create(model_path: &std::path::Path, n_ctx: i32, n_gpu_layers: i32) -> Result<Self> {
            // Early availability check
            if !matches!(option_env!("CROSSVAL_HAS_BITNET"), Some("true")) {
                return Err(CrossvalError::CppNotAvailable);
            }

            // Apply environment variable override if n_gpu_layers == 0
            // Three-level precedence: API > BITNET_GPU_LAYERS env > default 0
            let effective_gpu_layers = if n_gpu_layers == 0 {
                std::env::var("BITNET_GPU_LAYERS")
                    .ok()
                    .and_then(|s| s.trim().parse::<i32>().ok())
                    .unwrap_or(0)
            } else {
                n_gpu_layers
            };

            let model_path_c = CString::new(model_path.to_str().ok_or_else(|| {
                CrossvalError::ModelLoadError("Invalid UTF-8 in model path".to_string())
            })?)
            .map_err(|e| {
                CrossvalError::ModelLoadError(format!(
                    "Model path contains NUL byte at position {}",
                    e.nul_position()
                ))
            })?;

            let mut err_buf = vec![0u8; 512];
            let mut ctx_ptr: *mut BitnetContext = std::ptr::null_mut();

            let result = unsafe {
                bitnet_cpp_init_context(
                    &mut ctx_ptr,
                    model_path_c.as_ptr(),
                    n_ctx,
                    effective_gpu_layers,
                    err_buf.as_mut_ptr() as *mut c_char,
                    err_buf.len() as i32,
                )
            };

            if result != 0 {
                let error_msg =
                    std::str::from_utf8(&err_buf).unwrap_or("unknown error").trim_end_matches('\0');
                return Err(CrossvalError::InferenceError(error_msg.to_string()));
            }

            if ctx_ptr.is_null() {
                return Err(CrossvalError::InferenceError(
                    "bitnet_cpp_init_context returned null context".into(),
                ));
            }

            Ok(Self { ctx: ctx_ptr, model_path: model_path.to_path_buf(), n_ctx })
        }

        /// Tokenize using persistent session (Socket 2)
        ///
        /// Socket 2: Use BitNet-native tokenization if available, otherwise
        /// falls back to llama.cpp tokenization.
        ///
        /// # Arguments
        ///
        /// * `prompt` - Input text to tokenize
        ///
        /// # Returns
        ///
        /// A vector of token IDs on success, or `CrossvalError` on failure.
        pub fn tokenize(&self, prompt: &str) -> Result<Vec<i32>> {
            self.tokenize_ex(prompt, true, false)
        }

        /// Tokenize with explicit BOS and special token flags
        ///
        /// Socket 2: Extended tokenization control.
        ///
        /// # Arguments
        ///
        /// * `prompt` - Input text to tokenize
        /// * `add_bos` - Whether to add BOS (beginning-of-sequence) token
        /// * `parse_special` - Whether to parse special tokens
        pub fn tokenize_ex(
            &self,
            prompt: &str,
            add_bos: bool,
            parse_special: bool,
        ) -> Result<Vec<i32>> {
            let prompt_c = CString::new(prompt).map_err(|e| {
                CrossvalError::InferenceError(format!(
                    "Prompt contains NUL byte at position {}",
                    e.nul_position()
                ))
            })?;

            let mut out_len: i32 = 0;
            let mut err_buf = vec![0u8; 512];

            // Pass 1: Query size
            let result = unsafe {
                bitnet_cpp_tokenize_with_context(
                    self.ctx,
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

            if out_len <= 0 {
                return Ok(Vec::new());
            }

            // Pass 2: Get tokens
            let mut tokens = vec![0i32; out_len as usize];
            let result = unsafe {
                bitnet_cpp_tokenize_with_context(
                    self.ctx,
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

            tokens.truncate(out_len as usize);
            Ok(tokens)
        }

        /// Evaluate tokens using persistent session (Socket 3)
        ///
        /// Socket 3: Use BitNet-optimized 1-bit inference kernels with position tracking.
        ///
        /// # Arguments
        ///
        /// * `tokens` - Input token IDs to evaluate
        ///
        /// # Returns
        ///
        /// A 2D vector of logits where `logits[i][j]` is the logit for position `i`
        /// and vocab index `j`. Returns all-position logits (not just last token).
        ///
        /// # Note
        ///
        /// Position tracking is automatic - `n_past` is updated after successful evaluation.
        pub fn evaluate(&mut self, tokens: &[i32]) -> Result<Vec<Vec<f32>>> {
            self.evaluate_with_seq_id(tokens, 0)
        }

        /// Evaluate with explicit sequence ID (for future batch processing)
        ///
        /// Socket 3: Extended evaluation control with seq_id parameter and position tracking.
        ///
        /// # Arguments
        ///
        /// * `tokens` - Input token IDs
        /// * `seq_id` - Sequence ID for batch processing (0 for single sequence)
        pub fn evaluate_with_seq_id(
            &mut self,
            tokens: &[i32],
            seq_id: i32,
        ) -> Result<Vec<Vec<f32>>> {
            if tokens.is_empty() {
                return Err(CrossvalError::InferenceError("Empty token array".to_string()));
            }

            let mut out_rows: i32 = 0;
            let mut out_cols: i32 = 0;
            let mut err_buf = vec![0u8; 512];

            // Pass 1: Query shape
            let result = unsafe {
                bitnet_cpp_eval_with_context(
                    self.ctx,
                    tokens.as_ptr(),
                    tokens.len() as i32,
                    seq_id,
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

            if out_rows <= 0 || out_cols <= 0 {
                return Err(CrossvalError::InferenceError(format!(
                    "Invalid logits shape from BitNet.cpp: {} x {}",
                    out_rows, out_cols
                )));
            }

            // Pass 2: Get logits
            let total_elements =
                (out_rows as usize).checked_mul(out_cols as usize).ok_or_else(|| {
                    CrossvalError::InferenceError(format!(
                        "Logits buffer size overflow: {} x {}",
                        out_rows, out_cols
                    ))
                })?;

            let mut logits_flat = vec![0.0f32; total_elements];
            let result = unsafe {
                bitnet_cpp_eval_with_context(
                    self.ctx,
                    tokens.as_ptr(),
                    tokens.len() as i32,
                    seq_id,
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

            // Reshape flat buffer into 2D vector
            let mut logits_2d = Vec::with_capacity(out_rows as usize);
            for i in 0..out_rows as usize {
                let start = i * out_cols as usize;
                let end = start + out_cols as usize;
                logits_2d.push(logits_flat[start..end].to_vec());
            }

            Ok(logits_2d)
        }

        /// Reset KV cache position (start new conversation)
        ///
        /// Socket 1 Extension: Clears KV cache and resets position tracking.
        ///
        /// # Returns
        ///
        /// Ok(()) on success, or CrossvalError on failure.
        ///
        /// # Example
        ///
        /// ```no_run
        /// # use crossval::cpp_bindings::BitnetSession;
        /// # use std::path::Path;
        /// let mut session = BitnetSession::create(Path::new("model.gguf"), 512, 0)?;
        ///
        /// // First conversation
        /// session.evaluate(&vec![1, 2, 3, 4])?;
        ///
        /// // Reset for new conversation
        /// session.reset()?;
        ///
        /// // Start new prompt
        /// session.evaluate(&vec![5, 6, 7, 8])?;
        /// # Ok::<(), crossval::CrossvalError>(())
        /// ```
        pub fn reset(&mut self) -> Result<()> {
            if self.ctx.is_null() {
                return Err(CrossvalError::InferenceError("Cannot reset NULL context".to_string()));
            }

            unsafe {
                bitnet_cpp_reset_context(self.ctx);
            }

            Ok(())
        }

        /// Query current KV cache position
        ///
        /// Socket 1 Extension: Read-only query for position tracking state.
        ///
        /// # Returns
        ///
        /// Current position (number of cached tokens), or error if context is NULL.
        ///
        /// # Example
        ///
        /// ```no_run
        /// # use crossval::cpp_bindings::BitnetSession;
        /// # use std::path::Path;
        /// let mut session = BitnetSession::create(Path::new("model.gguf"), 512, 0)?;
        ///
        /// // After evaluating tokens
        /// session.evaluate(&vec![1, 2, 3, 4])?;
        /// let position = session.get_position()?;
        /// assert_eq!(position, 4);
        /// # Ok::<(), crossval::CrossvalError>(())
        /// ```
        pub fn get_position(&self) -> Result<i32> {
            if self.ctx.is_null() {
                return Err(CrossvalError::InferenceError(
                    "Cannot query position from NULL context".to_string(),
                ));
            }

            let position = unsafe { bitnet_cpp_get_position(self.ctx) };

            if position < 0 {
                return Err(CrossvalError::InferenceError(
                    "Invalid position returned from C++ wrapper".to_string(),
                ));
            }

            Ok(position)
        }
    }

    impl Drop for BitnetSession {
        fn drop(&mut self) {
            if !self.ctx.is_null() {
                unsafe {
                    let _ = bitnet_cpp_free_context(self.ctx);
                }
                self.ctx = std::ptr::null_mut();
            }
        }
    }

    unsafe impl Send for BitnetSession {}

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

//! Safe wrappers for llama.cpp C API
//!
//! This module provides safe Rust wrappers around the llama.cpp C API
//! for cross-validation against the Microsoft BitNet implementation.

use crate::bindings::*;
use std::ffi::{CStr, CString};
use std::ptr;
use std::slice;

/// Error type for C++ FFI operations
#[derive(Debug, thiserror::Error)]
pub enum CppError {
    #[error("Null pointer returned from C++")]
    NullPointer,

    #[error("Invalid UTF-8 string: {0}")]
    InvalidUtf8(#[from] std::str::Utf8Error),

    #[error("Invalid C string: {0}")]
    InvalidCString(#[from] std::ffi::NulError),

    #[error("LLAMA error: {0}")]
    LlamaError(String),

    #[error("Model loading failed: {0}")]
    ModelLoadError(String),
}

pub type Result<T> = std::result::Result<T, CppError>;

/// Initialize the llama backend
pub fn init_backend() {
    unsafe {
        llama_backend_init();
    }
}

/// Free the llama backend
pub fn free_backend() {
    unsafe {
        llama_backend_free();
    }
}

/// Get the llama.cpp version string
pub fn get_version() -> String {
    unsafe {
        let version = llama_print_system_info();
        if version.is_null() {
            "unknown".to_string()
        } else {
            CStr::from_ptr(version).to_string_lossy().to_string()
        }
    }
}

/// Safe wrapper for llama_model
pub struct Model {
    ptr: *mut llama_model,
}

impl Model {
    /// Load a model from a GGUF file
    pub fn load(path: &str) -> Result<Self> {
        let c_path = CString::new(path)?;

        // Create default model params
        let params = unsafe { llama_model_default_params() };

        // Load the model
        let ptr = unsafe { llama_load_model_from_file(c_path.as_ptr(), params) };

        if ptr.is_null() {
            return Err(CppError::ModelLoadError(format!("Failed to load model from: {}", path)));
        }

        Ok(Model { ptr })
    }

    /// Get the number of tokens in the model's vocabulary
    pub fn n_vocab(&self) -> i32 {
        unsafe { llama_n_vocab(self.ptr) }
    }

    /// Get the model's context size
    pub fn n_ctx_train(&self) -> i32 {
        unsafe { llama_n_ctx_train(self.ptr) }
    }

    /// Get the model's embedding dimension
    pub fn n_embd(&self) -> i32 {
        unsafe { llama_n_embd(self.ptr) }
    }

    /// Get raw pointer for use with C API
    pub(crate) fn as_ptr(&self) -> *mut llama_model {
        self.ptr
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                llama_free_model(self.ptr);
            }
        }
    }
}

// Safety: llama models are thread-safe for reading
unsafe impl Send for Model {}
unsafe impl Sync for Model {}

/// Safe wrapper for llama_context
pub struct Context {
    ptr: *mut llama_context,
}

impl Context {
    /// Create a new context from a model with deterministic settings
    pub fn new(model: &Model, n_ctx: u32, n_batch: u32, n_threads: i32) -> Result<Self> {
        let mut params = unsafe { llama_context_default_params() };
        params.n_ctx = n_ctx;
        params.n_batch = n_batch;
        params.n_threads = n_threads;
        params.n_threads_batch = n_threads;
        // Note: seed is not a field in llama_context_params in this version
        params.logits_all = true; // Get logits for all tokens

        let ptr = unsafe { llama_new_context_with_model(model.as_ptr(), params) };

        if ptr.is_null() {
            return Err(CppError::NullPointer);
        }

        Ok(Context { ptr })
    }

    /// Tokenize text into token IDs
    pub fn tokenize(&self, text: &str, add_special: bool) -> Result<Vec<i32>> {
        let c_text = CString::new(text)?;
        let model = unsafe { llama_get_model(self.ptr) };

        // First call to get the number of tokens
        let n_tokens = unsafe {
            llama_tokenize(
                model,
                c_text.as_ptr(),
                text.len() as i32,
                ptr::null_mut(),
                0,
                add_special,
                false, // parse_special
            )
        };

        if n_tokens < 0 {
            return Err(CppError::LlamaError("Tokenization failed".to_string()));
        }

        // Second call to get the actual tokens
        let mut tokens = vec![0i32; n_tokens as usize];
        let actual_n = unsafe {
            llama_tokenize(
                model,
                c_text.as_ptr(),
                text.len() as i32,
                tokens.as_mut_ptr(),
                tokens.len() as i32,
                add_special,
                false,
            )
        };

        if actual_n < 0 {
            return Err(CppError::LlamaError("Tokenization failed".to_string()));
        }

        tokens.truncate(actual_n as usize);
        Ok(tokens)
    }

    /// Decode tokens back to text
    pub fn decode(&self, tokens: &[i32]) -> Result<String> {
        let model = unsafe { llama_get_model(self.ptr) };
        let mut result = String::new();

        for &token in tokens {
            let token_str = unsafe {
                let ptr = llama_token_get_text(model, token);
                if ptr.is_null() {
                    continue;
                }
                CStr::from_ptr(ptr)
            };

            result.push_str(&token_str.to_string_lossy());
        }

        Ok(result)
    }

    /// Evaluate tokens and get logits using official API helpers
    pub fn eval(&mut self, tokens: &[i32], n_past: i32) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }

        // Create batch
        let mut batch = unsafe { llama_batch_init(tokens.len() as i32, 0, 1) };

        // Prepare seq_ids on the stack (more efficient than heap allocation)
        let mut seq_ids: [llama_seq_id; 1] = [0];
        let seq_ids_ptr = seq_ids.as_mut_ptr();

        // Populate the batch fields directly (no llama_batch_add in this version)
        unsafe {
            // Set tokens
            for (i, &token) in tokens.iter().enumerate() {
                *batch.token.add(i) = token;
                *batch.pos.add(i) = n_past + i as i32;
                *batch.n_seq_id.add(i) = 1;

                // Point to the same stack-allocated seq_id array for all tokens
                // This is safe because the array lives through the llama_decode call
                *batch.seq_id.add(i) = seq_ids_ptr;

                // Request logits for all tokens (could optimize to only last token if needed)
                *batch.logits.add(i) = 1;
            }
            batch.n_tokens = tokens.len() as i32;
        }

        // Decode the batch
        let result = unsafe { llama_decode(self.ptr, batch) };

        // Free the batch
        unsafe {
            llama_batch_free(batch);
        }

        if result != 0 {
            return Err(CppError::LlamaError(format!("Decode failed with code: {}", result)));
        }

        Ok(())
    }

    /// Get logits from the last evaluation for a specific token position
    pub fn get_logits(&self) -> Result<Vec<f32>> {
        let model = unsafe { llama_get_model(self.ptr) };
        let n_vocab = unsafe { llama_n_vocab(model) };

        let logits_ptr = unsafe { llama_get_logits(self.ptr) };
        if logits_ptr.is_null() {
            return Err(CppError::NullPointer);
        }

        let logits = unsafe { slice::from_raw_parts(logits_ptr, n_vocab as usize) };

        Ok(logits.to_vec())
    }

    /// Get logits for a specific token index (requires logits_all=true)
    pub fn get_logits_ith(&self, i: i32) -> Result<Vec<f32>> {
        let model = unsafe { llama_get_model(self.ptr) };
        let n_vocab = unsafe { llama_n_vocab(model) };

        let logits_ptr = unsafe { llama_get_logits_ith(self.ptr, i) };
        if logits_ptr.is_null() {
            return Err(CppError::NullPointer);
        }

        let logits = unsafe { slice::from_raw_parts(logits_ptr, n_vocab as usize) };

        Ok(logits.to_vec())
    }

    /// Get all logits for each token position (requires logits_all=true)
    pub fn get_all_logits(&self, n_tokens: usize) -> Result<Vec<Vec<f32>>> {
        let mut all_logits = Vec::with_capacity(n_tokens);

        for i in 0..n_tokens {
            all_logits.push(self.get_logits_ith(i as i32)?);
        }

        Ok(all_logits)
    }

    /// Sample a token from logits using greedy sampling
    pub fn sample_greedy(&self, logits: &[f32]) -> i32 {
        // Manual argmax that gracefully handles NaN values instead of
        // relying on `partial_cmp` which would panic on NaN.
        let mut max_i = 0;
        let mut max_val = f32::NEG_INFINITY;

        for (i, &val) in logits.iter().enumerate() {
            if val.is_nan() {
                continue;
            }
            if val > max_val {
                max_val = val;
                max_i = i;
            }
        }

        max_i as i32
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                llama_free(self.ptr);
            }
        }
    }
}

/// Combined session for easy use in tests with deterministic settings
pub struct Session {
    pub model: Model,
    pub context: Context,
}

impl Session {
    /// Load a model and create a context with deterministic settings
    pub fn load(model_path: &str, n_ctx: u32, n_batch: u32, n_threads: i32) -> Result<Self> {
        let model = Model::load(model_path)?;
        let context = Context::new(&model, n_ctx, n_batch, n_threads)?;
        Ok(Session { model, context })
    }

    /// Load with default deterministic settings for cross-validation
    pub fn load_deterministic(model_path: &str) -> Result<Self> {
        let model = Model::load(model_path)?;
        let context = Context::new(&model, 2048, 512, 1)?; // Single thread, deterministic
        Ok(Session { model, context })
    }

    /// Tokenize text
    pub fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
        self.context.tokenize(text, true)
    }

    /// Decode tokens to text
    pub fn decode(&self, tokens: &[i32]) -> Result<String> {
        self.context.decode(tokens)
    }

    /// Evaluate tokens and return logits
    pub fn eval_and_get_logits(&mut self, tokens: &[i32], n_past: i32) -> Result<Vec<f32>> {
        self.context.eval(tokens, n_past)?;
        self.context.get_logits()
    }

    /// Generate tokens greedily
    pub fn generate_greedy(&mut self, prompt: &str, max_tokens: usize) -> Result<Vec<i32>> {
        // Tokenize and evaluate the prompt
        let prompt_tokens = self.tokenize(prompt)?;
        let prompt_len = prompt_tokens.len();
        self.context.eval(&prompt_tokens, 0)?;

        let mut generated = Vec::new();

        // Generate new tokens
        for i in 0..max_tokens {
            let logits = self.context.get_logits()?;
            let next_token = self.context.sample_greedy(&logits);

            // Check for EOS
            let eos_token = unsafe { llama_token_eos(self.model.as_ptr()) };
            if next_token == eos_token {
                break;
            }

            generated.push(next_token);

            // Evaluate the new token
            self.context.eval(&[next_token], (prompt_len + i) as i32)?;
        }

        Ok(generated)
    }
}

// ============================================================================
// Wrappers for custom bitnet_c_shim.cc functions (for parity testing)
// ============================================================================

/// Safe wrapper for bitnet_model_t from bitnet_c_shim.cc
pub struct BitnetModel {
    ptr: *mut crate::bindings::bitnet_model_t,
}

impl BitnetModel {
    /// Load a model from a GGUF file using the custom C shim
    pub fn from_file(path: &str) -> Result<Self> {
        let c_path = CString::new(path)?;
        let ptr = unsafe { crate::bindings::bitnet_model_new_from_file(c_path.as_ptr()) };

        if ptr.is_null() {
            return Err(CppError::ModelLoadError(format!("Failed to load model from: {}", path)));
        }

        Ok(BitnetModel { ptr })
    }

    /// Get raw pointer for use with C API
    pub(crate) fn as_ptr(&self) -> *mut crate::bindings::bitnet_model_t {
        self.ptr
    }
}

impl Drop for BitnetModel {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                crate::bindings::bitnet_model_free(self.ptr);
            }
        }
    }
}

unsafe impl Send for BitnetModel {}
unsafe impl Sync for BitnetModel {}

/// Safe wrapper for bitnet_ctx_t from bitnet_c_shim.cc
pub struct BitnetContext {
    ptr: *mut crate::bindings::bitnet_ctx_t,
}

impl BitnetContext {
    /// Create a new context with specified parameters
    pub fn new(model: &BitnetModel, n_ctx: i32, n_threads: i32, seed: i32) -> Result<Self> {
        let params = crate::bindings::bitnet_params_t { n_ctx, n_threads, seed, rope_freq: 1.0 };

        let ptr = unsafe { crate::bindings::bitnet_context_new(model.as_ptr(), &params) };

        if ptr.is_null() {
            return Err(CppError::NullPointer);
        }

        Ok(BitnetContext { ptr })
    }

    /// Get raw pointer for use with C API
    pub(crate) fn as_ptr(&self) -> *mut crate::bindings::bitnet_ctx_t {
        self.ptr
    }
}

impl Drop for BitnetContext {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                crate::bindings::bitnet_context_free(self.ptr);
            }
        }
    }
}

/// Tokenize text using the custom C shim
pub fn bitnet_tokenize_text(
    model: &BitnetModel,
    text: &str,
    add_bos: bool,
    add_special: bool,
) -> Result<Vec<i32>> {
    let c_text = CString::new(text)?;

    // First call to get the number of tokens (pass nullptr for out_ids)
    let n_tokens = unsafe {
        crate::bindings::bitnet_tokenize(
            model.as_ptr(),
            c_text.as_ptr(),
            add_bos as i32,
            add_special as i32,
            ptr::null_mut(),
            0,
        )
    };

    if n_tokens < 0 {
        return Err(CppError::LlamaError(format!("Tokenization failed with code: {}", n_tokens)));
    }

    if n_tokens == 0 {
        return Ok(Vec::new());
    }

    // Second call to get the actual tokens
    let mut tokens = vec![0i32; n_tokens as usize];
    let actual_n = unsafe {
        crate::bindings::bitnet_tokenize(
            model.as_ptr(),
            c_text.as_ptr(),
            add_bos as i32,
            add_special as i32,
            tokens.as_mut_ptr(),
            tokens.len() as i32,
        )
    };

    if actual_n < 0 {
        return Err(CppError::LlamaError(format!("Tokenization failed with code: {}", actual_n)));
    }

    tokens.truncate(actual_n as usize);
    Ok(tokens)
}

/// Evaluate tokens and get last-position logits using the custom C shim
pub fn bitnet_eval_tokens(ctx: &BitnetContext, ids: &[i32], vocab_size: usize) -> Result<Vec<f32>> {
    if ids.is_empty() {
        return Err(CppError::LlamaError("Cannot eval empty token sequence".to_string()));
    }

    let mut logits = vec![0.0f32; vocab_size];

    let result = unsafe {
        crate::bindings::bitnet_eval(
            ctx.as_ptr(),
            ids.as_ptr(),
            ids.len() as i32,
            logits.as_mut_ptr(),
            logits.len() as i32,
        )
    };

    if result != 0 {
        return Err(CppError::LlamaError(format!("Eval failed with code: {}", result)));
    }

    Ok(logits)
}

/// Perform greedy decoding using the custom C shim
pub fn bitnet_decode_greedy(
    ctx: &BitnetContext,
    initial_ids: &[i32],
    max_new_tokens: usize,
    eos_id: i32,
) -> Result<Vec<i32>> {
    if max_new_tokens == 0 {
        return Ok(Vec::new());
    }

    // Allocate buffer for generated tokens
    let mut gen_ids = vec![0i32; max_new_tokens];

    let n_generated = unsafe {
        crate::bindings::bitnet_decode_greedy(
            ctx.as_ptr(),
            gen_ids.as_mut_ptr(),
            max_new_tokens as i32,
            eos_id,
            0.0, // temperature (greedy)
        )
    };

    if n_generated < 0 {
        return Err(CppError::LlamaError(format!("Decode failed with code: {}", n_generated)));
    }

    gen_ids.truncate(n_generated as usize);
    Ok(gen_ids)
}

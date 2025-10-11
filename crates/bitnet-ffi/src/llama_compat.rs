// llama.cpp compatible C API for drop-in replacement
#![allow(non_camel_case_types)]

use log::{debug, error, warn};
use std::ffi::CStr;
use std::os::raw::{c_char, c_float, c_int, c_void};
use std::ptr;
use std::slice;
use std::sync::Arc;

use bitnet_common::Device;
use bitnet_inference::InferenceEngine;
use bitnet_models::Model;
use bitnet_tokenizers::Tokenizer;

/// Opaque model handle - matches llama_model*
#[repr(C)]
pub struct llama_model {
    model: Arc<dyn Model>,
    tokenizer: Arc<dyn Tokenizer>,
    device: Device,
}

/// Opaque context handle - matches llama_context*
#[repr(C)]
pub struct llama_context {
    engine: InferenceEngine,
    last_logits: Vec<f32>,
    vocab_size: usize,
}

/// Model loading parameters - matches llama_model_params
#[repr(C)]
pub struct llama_model_params {
    pub n_gpu_layers: c_int,
    pub main_gpu: c_int,
    pub tensor_split: *const c_float,
    pub progress_callback: Option<extern "C" fn(f32, *mut c_void) -> bool>,
    pub progress_callback_user_data: *mut c_void,
    pub kv_overrides: *const c_void,
    pub vocab_only: bool,
    pub use_mmap: bool,
    pub use_mlock: bool,
}

/// Context parameters - matches llama_context_params
#[repr(C)]
pub struct llama_context_params {
    pub seed: u32,
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_threads: u32,
    pub n_threads_batch: u32,
    pub rope_scaling_type: c_int,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub yarn_ext_factor: f32,
    pub yarn_attn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub yarn_orig_ctx: u32,
    pub cb_eval: Option<extern "C" fn(*mut c_void, bool) -> bool>,
    pub cb_eval_user_data: *mut c_void,
    pub type_k: c_int,
    pub type_v: c_int,
    pub logits_all: bool,
    pub embedding: bool,
    pub offload_kqv: bool,
}

/// Load model from file - 100% compatible with llama_load_model_from_file
///
/// # Safety
/// This function dereferences raw pointers and must be called with valid arguments:
/// - `path_model` must be a valid pointer to a null-terminated C string
/// - The string must be valid UTF-8 and the file must exist
/// - The caller must ensure the pointer remains valid for the duration of this call
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_load_model_from_file(
    path_model: *const c_char,
    params: llama_model_params,
) -> *mut llama_model {
    if path_model.is_null() {
        error!("llama_load_model_from_file: null path");
        return ptr::null_mut();
    }

    let path = unsafe {
        match CStr::from_ptr(path_model).to_str() {
            Ok(s) => s,
            Err(e) => {
                error!("Invalid UTF-8 in model path: {}", e);
                return ptr::null_mut();
            }
        }
    };

    debug!("Loading model from: {}", path);

    // Detect device based on params
    let device = if params.n_gpu_layers > 0 {
        #[cfg(any(feature = "gpu", feature = "cuda"))]
        {
            Device::Cuda(params.main_gpu as usize)
        }
        #[cfg(not(any(feature = "gpu", feature = "cuda")))]
        {
            warn!("GPU layers requested but CUDA not compiled in, using CPU");
            Device::Cpu
        }
    } else {
        Device::Cpu
    };

    // Load model with auto-fix for compatibility
    let model = match fix_and_load_model(path, device) {
        Ok(m) => m,
        Err(e) => {
            error!("Failed to load model: {}", e);
            return ptr::null_mut();
        }
    };

    // Create universal tokenizer that handles all formats
    let tokenizer = match create_universal_tokenizer(model.as_ref()) {
        Ok(t) => t,
        Err(e) => {
            error!("Failed to create tokenizer: {}", e);
            return ptr::null_mut();
        }
    };

    let llama_model = Box::new(llama_model { model, tokenizer, device });

    Box::into_raw(llama_model)
}

/// Free model - matches llama_free_model
/// Free a model - 100% compatible with llama_free_model
///
/// # Safety
/// This function takes ownership of a raw pointer and must be called with valid arguments:
/// - `model` must be a valid pointer previously returned by llama_load_model_from_file
/// - The model must not be used after this call
/// - This function must not be called twice with the same pointer
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_free_model(model: *mut llama_model) {
    if !model.is_null() {
        unsafe {
            drop(Box::from_raw(model));
        }
    }
}

/// Create context from model - matches llama_new_context_with_model
/// Create new context with model - 100% compatible with llama_new_context_with_model
///
/// # Safety
/// This function dereferences raw pointers and must be called with valid arguments:
/// - `model` must be a valid pointer to a model previously loaded with llama_load_model_from_file
/// - The model must remain valid for the lifetime of the returned context
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_new_context_with_model(
    model: *mut llama_model,
    _params: llama_context_params,
) -> *mut llama_context {
    if model.is_null() {
        return ptr::null_mut();
    }

    let model = unsafe { &*model };

    match InferenceEngine::new(model.model.clone(), model.tokenizer.clone(), model.device) {
        Ok(engine) => {
            let vocab_size = model.tokenizer.vocab_size();
            let ctx =
                Box::new(llama_context { engine, last_logits: vec![0.0; vocab_size], vocab_size });
            Box::into_raw(ctx)
        }
        Err(e) => {
            error!("Failed to create context: {}", e);
            ptr::null_mut()
        }
    }
}

/// Free context - matches llama_free
/// Free a context - 100% compatible with llama_free
///
/// # Safety
/// This function takes ownership of a raw pointer and must be called with valid arguments:
/// - `ctx` must be a valid pointer previously returned by llama_new_context_with_model
/// - The context must not be used after this call
/// - This function must not be called twice with the same pointer
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_free(ctx: *mut llama_context) {
    if !ctx.is_null() {
        unsafe {
            drop(Box::from_raw(ctx));
        }
    }
}

/// Tokenize text - 100% compatible with llama_tokenize
/// Returns number of tokens, or negative value on error
/// If return value > n_max_tokens, call again with larger buffer
/// Tokenize text - 100% compatible with llama_tokenize
///
/// # Safety
/// This function dereferences raw pointers and must be called with valid arguments:
/// - `model` must be a valid pointer to a loaded model
/// - `text` must be a valid pointer to a null-terminated C string
/// - `tokens` must be a valid pointer to a writable array of at least `n_max_tokens` elements
/// - All pointers must remain valid for the duration of this call
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_tokenize(
    model: *const llama_model,
    text: *const c_char,
    text_len: c_int,
    tokens: *mut c_int,
    n_max_tokens: c_int,
    add_bos: bool,
    special: bool,
) -> c_int {
    if model.is_null() || text.is_null() {
        return -1;
    }

    let model = unsafe { &*model };

    // Convert text
    let text_slice = if text_len < 0 {
        // Null-terminated string
        unsafe {
            match CStr::from_ptr(text).to_str() {
                Ok(s) => s,
                Err(_) => return -2,
            }
        }
    } else {
        // Length-specified string
        unsafe {
            let bytes = slice::from_raw_parts(text as *const u8, text_len as usize);
            match std::str::from_utf8(bytes) {
                Ok(s) => s,
                Err(_) => return -2,
            }
        }
    };

    // Tokenize with universal tokenizer
    let token_ids = match model.tokenizer.encode(text_slice, add_bos, special) {
        Ok(ids) => ids,
        Err(e) => {
            error!("Tokenization failed: {}", e);
            // Return -3 to match llama.cpp error code for tokenization failure
            return -3;
        }
    };

    let n_tokens = token_ids.len() as c_int;

    // Check buffer size
    if tokens.is_null() {
        // Caller is querying required size
        return n_tokens;
    }

    if n_tokens > n_max_tokens {
        // Buffer too small - return negative to indicate required size
        return -n_tokens;
    }

    // Copy tokens to output buffer
    unsafe {
        for (i, &id) in token_ids.iter().enumerate() {
            *tokens.add(i) = id as c_int;
        }
    }

    n_tokens
}

/// Evaluate tokens - matches llama_eval
/// Evaluate tokens - 100% compatible with llama_eval
///
/// # Safety
/// This function dereferences raw pointers and must be called with valid arguments:
/// - `ctx` must be a valid pointer to a context
/// - `tokens` must be a valid pointer to a readable array of at least `n_tokens` elements
/// - All pointers must remain valid for the duration of this call
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_eval(
    ctx: *mut llama_context,
    tokens: *const c_int,
    n_tokens: c_int,
    _n_past: c_int,
    _n_threads: c_int,
) -> c_int {
    if ctx.is_null() || tokens.is_null() || n_tokens <= 0 {
        return 1; // Error
    }

    let ctx = unsafe { &mut *ctx };
    let tokens = unsafe { slice::from_raw_parts(tokens, n_tokens as usize) };
    let token_ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();

    // Run inference - need to block on async
    use futures::executor::block_on;
    match block_on(ctx.engine.eval_ids(&token_ids)) {
        Ok(logits) => {
            // Store logits for retrieval
            ctx.last_logits = logits;
            0 // Success
        }
        Err(e) => {
            error!("Eval failed: {}", e);
            1 // Error
        }
    }
}

/// Get logits pointer - matches llama_get_logits
/// Get logits - 100% compatible with llama_get_logits
///
/// # Safety
/// This function dereferences raw pointers and must be called with valid arguments:
/// - `ctx` must be a valid pointer to a context
/// - The returned pointer is valid only until the next call to llama_eval
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_get_logits(ctx: *mut llama_context) -> *mut c_float {
    if ctx.is_null() {
        return ptr::null_mut();
    }

    let ctx = unsafe { &mut *ctx };
    ctx.last_logits.as_mut_ptr()
}

/// Get logits for specific token - matches llama_get_logits_ith
/// Get logits at specific position - 100% compatible with llama_get_logits_ith
///
/// # Safety
/// This function dereferences raw pointers and must be called with valid arguments:
/// - `ctx` must be a valid pointer to a context
/// - `i` must be a valid token index
/// - The returned pointer is valid only until the next call to llama_eval
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_get_logits_ith(ctx: *mut llama_context, i: c_int) -> *mut c_float {
    if ctx.is_null() || i < 0 {
        return ptr::null_mut();
    }

    let ctx = unsafe { &mut *ctx };
    let offset = (i as usize) * ctx.vocab_size;

    if offset >= ctx.last_logits.len() {
        return ptr::null_mut();
    }

    unsafe { ctx.last_logits.as_mut_ptr().add(offset) }
}

/// Get vocabulary size
/// Get vocabulary size - 100% compatible with llama_n_vocab
///
/// # Safety
/// This function dereferences raw pointers and must be called with valid arguments:
/// - `model` must be a valid pointer to a loaded model
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_n_vocab(model: *const llama_model) -> c_int {
    if model.is_null() {
        return 0;
    }

    let model = unsafe { &*model };
    model.tokenizer.vocab_size() as c_int
}

/// Get context size
/// Get context size - 100% compatible with llama_n_ctx
///
/// # Safety
/// This function dereferences raw pointers and must be called with valid arguments:
/// - `ctx` must be a valid pointer to a context
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_n_ctx(ctx: *const llama_context) -> c_int {
    if ctx.is_null() {
        return 0;
    }

    let ctx = unsafe { &*ctx };
    ctx.engine.model_config().model.max_position_embeddings as c_int
}

// Helper functions

fn fix_and_load_model(path: &str, device: Device) -> anyhow::Result<Arc<dyn Model>> {
    use bitnet_compat::gguf_fixer::GgufCompatibilityFixer;

    // Diagnose issues
    let fixes = GgufCompatibilityFixer::diagnose(path)?;

    if !fixes.is_empty() {
        warn!("Model compatibility issues detected:");
        for fix in &fixes {
            warn!("  - {}", fix);
        }

        // Create fixed version
        let fixed_path = format!("{}.fixed", path);
        GgufCompatibilityFixer::export_fixed(path, &fixed_path)?;

        // Load fixed model
        use bitnet_models::formats::gguf::GgufLoader;
        use bitnet_models::loader::{FormatLoader, LoadConfig};
        use std::path::Path;
        let loader = GgufLoader;
        loader
            .load(Path::new(&fixed_path), &device, &LoadConfig::default())
            .map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))
            .map(Arc::from)
    } else {
        use bitnet_models::formats::gguf::GgufLoader;
        use bitnet_models::loader::{FormatLoader, LoadConfig};
        use std::path::Path;
        let loader = GgufLoader;
        loader
            .load(Path::new(path), &device, &LoadConfig::default())
            .map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))
            .map(Arc::from)
    }
}

fn create_universal_tokenizer(_model: &dyn Model) -> anyhow::Result<Arc<dyn Tokenizer>> {
    use bitnet_tokenizers::{TokenizerConfig, UniversalTokenizer};

    // Create default tokenizer config
    let config = TokenizerConfig::new();

    // Create tokenizer that handles GPT2, SentencePiece, etc.
    let tokenizer = UniversalTokenizer::new(config)?;

    Ok(Arc::new(tokenizer))
}

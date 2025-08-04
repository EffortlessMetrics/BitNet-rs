//! Core C API functions with exact signature compatibility
//!
//! This module implements the core C API functions that match the existing BitNet C++
//! bindings exactly, providing a drop-in replacement with enhanced functionality.

use crate::{
    BitNetCError, BitNetCModel, BitNetCConfig, BitNetCInferenceConfig,
    set_last_error, clear_last_error, get_model_manager, get_inference_manager
};
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_float, c_uint, c_ulong};
use std::ptr;

/// ABI version for compatibility checking
pub const BITNET_ABI_VERSION: u32 = 1;

/// Error codes matching existing C++ API
pub const BITNET_SUCCESS: c_int = 0;
pub const BITNET_ERROR_INVALID_ARGUMENT: c_int = -1;
pub const BITNET_ERROR_MODEL_NOT_FOUND: c_int = -2;
pub const BITNET_ERROR_MODEL_LOAD_FAILED: c_int = -3;
pub const BITNET_ERROR_INFERENCE_FAILED: c_int = -4;
pub const BITNET_ERROR_OUT_OF_MEMORY: c_int = -5;
pub const BITNET_ERROR_THREAD_SAFETY: c_int = -6;
pub const BITNET_ERROR_INVALID_MODEL_ID: c_int = -7;
pub const BITNET_ERROR_CONTEXT_LENGTH_EXCEEDED: c_int = -8;
pub const BITNET_ERROR_UNSUPPORTED_OPERATION: c_int = -9;
pub const BITNET_ERROR_INTERNAL: c_int = -10;

/// Get ABI version for compatibility validation
/// 
/// Returns the current ABI version number. Applications should check this
/// to ensure compatibility with the expected API version.
/// 
/// # Returns
/// Current ABI version number
#[no_mangle]
pub extern "C" fn bitnet_abi_version() -> c_uint {
    BITNET_ABI_VERSION
}

/// Get library version string
/// 
/// Returns a null-terminated string containing the library version.
/// The returned pointer is valid for the lifetime of the program.
/// 
/// # Returns
/// Pointer to null-terminated version string
#[no_mangle]
pub extern "C" fn bitnet_version() -> *const c_char {
    static VERSION: &str = "0.1.0\0";
    VERSION.as_ptr() as *const c_char
}

/// Initialize the BitNet library
/// 
/// Must be called before any other BitNet functions. This function is thread-safe
/// and can be called multiple times safely.
/// 
/// # Returns
/// BITNET_SUCCESS on success, error code on failure
#[no_mangle]
pub extern "C" fn bitnet_init() -> c_int {
    clear_last_error();
    
    match crate::memory::initialize_memory_manager() {
        Ok(_) => {
            match crate::threading::initialize_thread_pool() {
                Ok(_) => BITNET_SUCCESS,
                Err(e) => {
                    set_last_error(BitNetCError::ThreadSafety(format!("Failed to initialize thread pool: {}", e)));
                    BITNET_ERROR_THREAD_SAFETY
                }
            }
        }
        Err(e) => {
            set_last_error(BitNetCError::OutOfMemory(format!("Failed to initialize memory manager: {}", e)));
            BITNET_ERROR_OUT_OF_MEMORY
        }
    }
}

/// Cleanup and shutdown the BitNet library
/// 
/// Should be called when the library is no longer needed. After calling this
/// function, no other BitNet functions should be called except bitnet_init().
/// This function is thread-safe.
/// 
/// # Returns
/// BITNET_SUCCESS on success, error code on failure
#[no_mangle]
pub extern "C" fn bitnet_cleanup() -> c_int {
    clear_last_error();
    
    // Cleanup in reverse order of initialization
    if let Err(e) = crate::threading::cleanup_thread_pool() {
        set_last_error(BitNetCError::Internal(format!("Failed to cleanup thread pool: {}", e)));
        return BITNET_ERROR_INTERNAL;
    }
    
    if let Err(e) = crate::memory::cleanup_memory_manager() {
        set_last_error(BitNetCError::Internal(format!("Failed to cleanup memory manager: {}", e)));
        return BITNET_ERROR_INTERNAL;
    }
    
    BITNET_SUCCESS
}

/// Load a model from file with exact signature compatibility
/// 
/// Loads a BitNet model from the specified file path. The model format is
/// automatically detected based on the file extension and content.
/// 
/// # Arguments
/// * `path` - Null-terminated string containing the path to the model file
/// 
/// # Returns
/// Model ID (>= 0) on success, negative error code on failure
#[no_mangle]
pub extern "C" fn bitnet_model_load(path: *const c_char) -> c_int {
    clear_last_error();
    
    if path.is_null() {
        set_last_error(BitNetCError::InvalidArgument("path cannot be null".to_string()));
        return BITNET_ERROR_INVALID_ARGUMENT;
    }
    
    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(BitNetCError::InvalidArgument(format!("Invalid UTF-8 in path: {}", e)));
            return BITNET_ERROR_INVALID_ARGUMENT;
        }
    };
    
    match get_model_manager().load_model(path_str) {
        Ok(model_id) => model_id as c_int,
        Err(e) => {
            set_last_error(e);
            match get_last_error() {
                Some(BitNetCError::ModelNotFound(_)) => BITNET_ERROR_MODEL_NOT_FOUND,
                Some(BitNetCError::ModelLoadFailed(_)) => BITNET_ERROR_MODEL_LOAD_FAILED,
                Some(BitNetCError::OutOfMemory(_)) => BITNET_ERROR_OUT_OF_MEMORY,
                _ => BITNET_ERROR_INTERNAL,
            }
        }
    }
}

/// Load a model with configuration
/// 
/// Loads a BitNet model with the specified configuration. This provides more
/// control over the loading process than bitnet_model_load().
/// 
/// # Arguments
/// * `path` - Null-terminated string containing the path to the model file
/// * `config` - Pointer to model configuration structure
/// 
/// # Returns
/// Model ID (>= 0) on success, negative error code on failure
#[no_mangle]
pub extern "C" fn bitnet_model_load_with_config(
    path: *const c_char,
    config: *const BitNetCConfig,
) -> c_int {
    clear_last_error();
    
    if path.is_null() {
        set_last_error(BitNetCError::InvalidArgument("path cannot be null".to_string()));
        return BITNET_ERROR_INVALID_ARGUMENT;
    }
    
    if config.is_null() {
        set_last_error(BitNetCError::InvalidArgument("config cannot be null".to_string()));
        return BITNET_ERROR_INVALID_ARGUMENT;
    }
    
    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(BitNetCError::InvalidArgument(format!("Invalid UTF-8 in path: {}", e)));
            return BITNET_ERROR_INVALID_ARGUMENT;
        }
    };
    
    let config_ref = unsafe { &*config };
    
    match get_model_manager().load_model_with_config(path_str, config_ref) {
        Ok(model_id) => model_id as c_int,
        Err(e) => {
            set_last_error(e);
            match get_last_error() {
                Some(BitNetCError::ModelNotFound(_)) => BITNET_ERROR_MODEL_NOT_FOUND,
                Some(BitNetCError::ModelLoadFailed(_)) => BITNET_ERROR_MODEL_LOAD_FAILED,
                Some(BitNetCError::OutOfMemory(_)) => BITNET_ERROR_OUT_OF_MEMORY,
                _ => BITNET_ERROR_INTERNAL,
            }
        }
    }
}

/// Free a loaded model with exact signature compatibility
/// 
/// Frees the resources associated with a loaded model. After calling this
/// function, the model ID becomes invalid and should not be used.
/// 
/// # Arguments
/// * `model_id` - Model ID returned by bitnet_model_load()
/// 
/// # Returns
/// BITNET_SUCCESS on success, error code on failure
#[no_mangle]
pub extern "C" fn bitnet_model_free(model_id: c_int) -> c_int {
    clear_last_error();
    
    if model_id < 0 {
        set_last_error(BitNetCError::InvalidArgument("model_id must be non-negative".to_string()));
        return BITNET_ERROR_INVALID_ARGUMENT;
    }
    
    match get_model_manager().free_model(model_id as u32) {
        Ok(_) => BITNET_SUCCESS,
        Err(e) => {
            set_last_error(e);
            match get_last_error() {
                Some(BitNetCError::InvalidModelId(_)) => BITNET_ERROR_INVALID_MODEL_ID,
                _ => BITNET_ERROR_INTERNAL,
            }
        }
    }
}

/// Run inference with exact signature compatibility
/// 
/// Generates text from the given prompt using the specified model.
/// The output buffer must be large enough to hold the generated text.
/// 
/// # Arguments
/// * `model_id` - Model ID returned by bitnet_model_load()
/// * `prompt` - Null-terminated input prompt string
/// * `output` - Buffer to store the generated text (null-terminated)
/// * `max_len` - Maximum length of the output buffer (including null terminator)
/// 
/// # Returns
/// Number of characters written (excluding null terminator) on success, negative error code on failure
#[no_mangle]
pub extern "C" fn bitnet_inference(
    model_id: c_int,
    prompt: *const c_char,
    output: *mut c_char,
    max_len: usize,
) -> c_int {
    clear_last_error();
    
    if model_id < 0 {
        set_last_error(BitNetCError::InvalidArgument("model_id must be non-negative".to_string()));
        return BITNET_ERROR_INVALID_ARGUMENT;
    }
    
    if prompt.is_null() {
        set_last_error(BitNetCError::InvalidArgument("prompt cannot be null".to_string()));
        return BITNET_ERROR_INVALID_ARGUMENT;
    }
    
    if output.is_null() {
        set_last_error(BitNetCError::InvalidArgument("output cannot be null".to_string()));
        return BITNET_ERROR_INVALID_ARGUMENT;
    }
    
    if max_len == 0 {
        set_last_error(BitNetCError::InvalidArgument("max_len must be greater than 0".to_string()));
        return BITNET_ERROR_INVALID_ARGUMENT;
    }
    
    let prompt_str = match unsafe { CStr::from_ptr(prompt) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(BitNetCError::InvalidArgument(format!("Invalid UTF-8 in prompt: {}", e)));
            return BITNET_ERROR_INVALID_ARGUMENT;
        }
    };
    
    match get_inference_manager().generate(model_id as u32, prompt_str, max_len) {
        Ok(generated_text) => {
            let bytes_to_copy = std::cmp::min(generated_text.len(), max_len - 1);
            unsafe {
                ptr::copy_nonoverlapping(
                    generated_text.as_ptr(),
                    output as *mut u8,
                    bytes_to_copy,
                );
                *output.add(bytes_to_copy) = 0; // Null terminator
            }
            bytes_to_copy as c_int
        }
        Err(e) => {
            set_last_error(e);
            match get_last_error() {
                Some(BitNetCError::InvalidModelId(_)) => BITNET_ERROR_INVALID_MODEL_ID,
                Some(BitNetCError::InferenceFailed(_)) => BITNET_ERROR_INFERENCE_FAILED,
                Some(BitNetCError::ContextLengthExceeded(_)) => BITNET_ERROR_CONTEXT_LENGTH_EXCEEDED,
                Some(BitNetCError::OutOfMemory(_)) => BITNET_ERROR_OUT_OF_MEMORY,
                _ => BITNET_ERROR_INTERNAL,
            }
        }
    }
}

/// Run inference with configuration
/// 
/// Generates text from the given prompt using the specified model and configuration.
/// This provides more control over the generation process than bitnet_inference().
/// 
/// # Arguments
/// * `model_id` - Model ID returned by bitnet_model_load()
/// * `prompt` - Null-terminated input prompt string
/// * `config` - Pointer to inference configuration structure
/// * `output` - Buffer to store the generated text (null-terminated)
/// * `max_len` - Maximum length of the output buffer (including null terminator)
/// 
/// # Returns
/// Number of characters written (excluding null terminator) on success, negative error code on failure
#[no_mangle]
pub extern "C" fn bitnet_inference_with_config(
    model_id: c_int,
    prompt: *const c_char,
    config: *const BitNetCInferenceConfig,
    output: *mut c_char,
    max_len: usize,
) -> c_int {
    clear_last_error();
    
    if model_id < 0 {
        set_last_error(BitNetCError::InvalidArgument("model_id must be non-negative".to_string()));
        return BITNET_ERROR_INVALID_ARGUMENT;
    }
    
    if prompt.is_null() {
        set_last_error(BitNetCError::InvalidArgument("prompt cannot be null".to_string()));
        return BITNET_ERROR_INVALID_ARGUMENT;
    }
    
    if config.is_null() {
        set_last_error(BitNetCError::InvalidArgument("config cannot be null".to_string()));
        return BITNET_ERROR_INVALID_ARGUMENT;
    }
    
    if output.is_null() {
        set_last_error(BitNetCError::InvalidArgument("output cannot be null".to_string()));
        return BITNET_ERROR_INVALID_ARGUMENT;
    }
    
    if max_len == 0 {
        set_last_error(BitNetCError::InvalidArgument("max_len must be greater than 0".to_string()));
        return BITNET_ERROR_INVALID_ARGUMENT;
    }
    
    let prompt_str = match unsafe { CStr::from_ptr(prompt) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(BitNetCError::InvalidArgument(format!("Invalid UTF-8 in prompt: {}", e)));
            return BITNET_ERROR_INVALID_ARGUMENT;
        }
    };
    
    let config_ref = unsafe { &*config };
    
    match get_inference_manager().generate_with_config(model_id as u32, prompt_str, config_ref, max_len) {
        Ok(generated_text) => {
            let bytes_to_copy = std::cmp::min(generated_text.len(), max_len - 1);
            unsafe {
                ptr::copy_nonoverlapping(
                    generated_text.as_ptr(),
                    output as *mut u8,
                    bytes_to_copy,
                );
                *output.add(bytes_to_copy) = 0; // Null terminator
            }
            bytes_to_copy as c_int
        }
        Err(e) => {
            set_last_error(e);
            match get_last_error() {
                Some(BitNetCError::InvalidModelId(_)) => BITNET_ERROR_INVALID_MODEL_ID,
                Some(BitNetCError::InferenceFailed(_)) => BITNET_ERROR_INFERENCE_FAILED,
                Some(BitNetCError::ContextLengthExceeded(_)) => BITNET_ERROR_CONTEXT_LENGTH_EXCEEDED,
                Some(BitNetCError::OutOfMemory(_)) => BITNET_ERROR_OUT_OF_MEMORY,
                _ => BITNET_ERROR_INTERNAL,
            }
        }
    }
}

/// Get the last error message
/// 
/// Returns a detailed error message for the last error that occurred.
/// The returned pointer is valid until the next BitNet function call.
/// 
/// # Returns
/// Pointer to null-terminated error message, or null if no error occurred
#[no_mangle]
pub extern "C" fn bitnet_get_last_error() -> *const c_char {
    match get_last_error() {
        Some(error) => {
            let error_msg = format!("{}\0", error);
            // Store in thread-local storage to ensure lifetime
            thread_local! {
                static ERROR_MSG: std::cell::RefCell<Option<CString>> = std::cell::RefCell::new(None);
            }
            
            ERROR_MSG.with(|msg| {
                let cstring = CString::new(error_msg).unwrap_or_else(|_| {
                    CString::new("Error message contains null bytes").unwrap()
                });
                let ptr = cstring.as_ptr();
                *msg.borrow_mut() = Some(cstring);
                ptr
            })
        }
        None => ptr::null(),
    }
}

/// Clear the last error
/// 
/// Clears the last error state. After calling this function,
/// bitnet_get_last_error() will return null until another error occurs.
#[no_mangle]
pub extern "C" fn bitnet_clear_last_error() {
    clear_last_error();
}

/// Check if a model is loaded
/// 
/// Checks whether a model with the given ID is currently loaded.
/// 
/// # Arguments
/// * `model_id` - Model ID to check
/// 
/// # Returns
/// 1 if model is loaded, 0 if not loaded, negative error code on failure
#[no_mangle]
pub extern "C" fn bitnet_model_is_loaded(model_id: c_int) -> c_int {
    clear_last_error();
    
    if model_id < 0 {
        set_last_error(BitNetCError::InvalidArgument("model_id must be non-negative".to_string()));
        return BITNET_ERROR_INVALID_ARGUMENT;
    }
    
    match get_model_manager().is_model_loaded(model_id as u32) {
        Ok(is_loaded) => if is_loaded { 1 } else { 0 },
        Err(e) => {
            set_last_error(e);
            BITNET_ERROR_INTERNAL
        }
    }
}

/// Get model information
/// 
/// Retrieves information about a loaded model.
/// 
/// # Arguments
/// * `model_id` - Model ID
/// * `info` - Pointer to structure to fill with model information
/// 
/// # Returns
/// BITNET_SUCCESS on success, error code on failure
#[no_mangle]
pub extern "C" fn bitnet_model_get_info(
    model_id: c_int,
    info: *mut BitNetCModel,
) -> c_int {
    clear_last_error();
    
    if model_id < 0 {
        set_last_error(BitNetCError::InvalidArgument("model_id must be non-negative".to_string()));
        return BITNET_ERROR_INVALID_ARGUMENT;
    }
    
    if info.is_null() {
        set_last_error(BitNetCError::InvalidArgument("info cannot be null".to_string()));
        return BITNET_ERROR_INVALID_ARGUMENT;
    }
    
    match get_model_manager().get_model_info(model_id as u32) {
        Ok(model_info) => {
            unsafe {
                *info = model_info;
            }
            BITNET_SUCCESS
        }
        Err(e) => {
            set_last_error(e);
            match get_last_error() {
                Some(BitNetCError::InvalidModelId(_)) => BITNET_ERROR_INVALID_MODEL_ID,
                _ => BITNET_ERROR_INTERNAL,
            }
        }
    }
}

/// Set the number of threads for CPU inference
/// 
/// Sets the number of threads to use for CPU-based inference operations.
/// This affects all models and inference operations.
/// 
/// # Arguments
/// * `num_threads` - Number of threads to use (0 for auto-detection)
/// 
/// # Returns
/// BITNET_SUCCESS on success, error code on failure
#[no_mangle]
pub extern "C" fn bitnet_set_num_threads(num_threads: c_uint) -> c_int {
    clear_last_error();
    
    match crate::threading::set_num_threads(num_threads as usize) {
        Ok(_) => BITNET_SUCCESS,
        Err(e) => {
            set_last_error(BitNetCError::ThreadSafety(format!("Failed to set thread count: {}", e)));
            BITNET_ERROR_THREAD_SAFETY
        }
    }
}

/// Get the current number of threads
/// 
/// Returns the current number of threads being used for CPU inference.
/// 
/// # Returns
/// Number of threads currently in use
#[no_mangle]
pub extern "C" fn bitnet_get_num_threads() -> c_uint {
    crate::threading::get_num_threads() as c_uint
}

/// Enable or disable GPU acceleration
/// 
/// Enables or disables GPU acceleration for inference operations.
/// This setting affects all subsequent model loading and inference operations.
/// 
/// # Arguments
/// * `enable` - 1 to enable GPU, 0 to disable
/// 
/// # Returns
/// BITNET_SUCCESS on success, error code on failure
#[no_mangle]
pub extern "C" fn bitnet_set_gpu_enabled(enable: c_int) -> c_int {
    clear_last_error();
    
    match get_inference_manager().set_gpu_enabled(enable != 0) {
        Ok(_) => BITNET_SUCCESS,
        Err(e) => {
            set_last_error(e);
            BITNET_ERROR_UNSUPPORTED_OPERATION
        }
    }
}

/// Check if GPU acceleration is available
/// 
/// Checks whether GPU acceleration is available on the current system.
/// 
/// # Returns
/// 1 if GPU is available, 0 if not available
#[no_mangle]
pub extern "C" fn bitnet_is_gpu_available() -> c_int {
    if get_inference_manager().is_gpu_available() { 1 } else { 0 }
}
//! Stub implementation of C++ FFI functions for testing
//!
//! This module provides stub implementations of the C++ FFI functions
//! that would normally be provided by linking against the BitNet.cpp library.
//! These stubs allow the code to compile and run basic tests without
//! requiring the actual C++ implementation to be available.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int, c_uint};
use std::ptr;

use super::cpp_implementation::{
    BitNetCppHandle, CppInferenceConfig, CppInferenceResult, CppModelInfo, CppPerformanceMetrics,
};

// Stub implementations of the C++ FFI functions
// These will be used when the actual C++ library is not available

#[no_mangle]
pub extern "C" fn bitnet_cpp_create() -> *mut BitNetCppHandle {
    // Return a non-null pointer to indicate "success"
    // In a real implementation, this would allocate and return a handle
    0x1 as *mut BitNetCppHandle
}

#[no_mangle]
pub extern "C" fn bitnet_cpp_destroy(_handle: *mut BitNetCppHandle) {
    // Stub: no-op
}

#[no_mangle]
pub extern "C" fn bitnet_cpp_is_available() -> c_int {
    // Return 0 to indicate not available in stub mode
    0
}

#[no_mangle]
pub extern "C" fn bitnet_cpp_load_model(
    _handle: *mut BitNetCppHandle,
    _path: *const c_char,
) -> c_int {
    // Return error code to indicate failure in stub mode
    -1
}

#[no_mangle]
pub extern "C" fn bitnet_cpp_unload_model(_handle: *mut BitNetCppHandle) -> c_int {
    // Return success
    0
}

#[no_mangle]
pub extern "C" fn bitnet_cpp_is_model_loaded(_handle: *mut BitNetCppHandle) -> c_int {
    // Return false (no model loaded)
    0
}

#[no_mangle]
pub extern "C" fn bitnet_cpp_get_model_info(_handle: *mut BitNetCppHandle) -> CppModelInfo {
    // Return empty model info
    CppModelInfo {
        name: ptr::null(),
        format: 0,
        size_bytes: 0,
        parameter_count: 0,
        context_length: 0,
        vocabulary_size: 0,
    }
}

#[no_mangle]
pub extern "C" fn bitnet_cpp_tokenize(
    _handle: *mut BitNetCppHandle,
    _text: *const c_char,
    tokens: *mut *mut c_uint,
    token_count: *mut c_uint,
) -> c_int {
    // Return empty token list
    unsafe {
        *tokens = ptr::null_mut();
        *token_count = 0;
    }
    -1 // Error code
}

#[no_mangle]
pub extern "C" fn bitnet_cpp_detokenize(
    _handle: *mut BitNetCppHandle,
    _tokens: *const c_uint,
    _token_count: c_uint,
    text: *mut *mut c_char,
) -> c_int {
    // Return empty string
    unsafe {
        *text = ptr::null_mut();
    }
    -1 // Error code
}

#[no_mangle]
pub extern "C" fn bitnet_cpp_inference(
    _handle: *mut BitNetCppHandle,
    _tokens: *const c_uint,
    _token_count: c_uint,
    _config: *const CppInferenceConfig,
    result: *mut CppInferenceResult,
) -> c_int {
    // Return empty result
    unsafe {
        (*result).tokens = ptr::null_mut();
        (*result).token_count = 0;
        (*result).text = ptr::null();
        (*result).duration_ms = 0;
        (*result).memory_usage = 0;
    }
    -1 // Error code
}

#[no_mangle]
pub extern "C" fn bitnet_cpp_get_metrics(_handle: *mut BitNetCppHandle) -> CppPerformanceMetrics {
    // Return empty metrics
    CppPerformanceMetrics {
        model_load_time_ms: 0,
        tokenization_time_ms: 0,
        inference_time_ms: 0,
        peak_memory: 0,
        tokens_per_second: 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bitnet_cpp_reset_metrics(_handle: *mut BitNetCppHandle) {
    // Stub: no-op
}

#[no_mangle]
pub extern "C" fn bitnet_cpp_cleanup(_handle: *mut BitNetCppHandle) -> c_int {
    // Return success
    0
}

#[no_mangle]
pub extern "C" fn bitnet_cpp_free_string(_ptr: *mut c_char) {
    // Stub: no-op (nothing to free in stub mode)
}

#[no_mangle]
pub extern "C" fn bitnet_cpp_free_tokens(_ptr: *mut c_uint) {
    // Stub: no-op (nothing to free in stub mode)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stub_functions() {
        // Test that stub functions don't crash
        let handle = bitnet_cpp_create();
        assert!(!handle.is_null());

        assert_eq!(bitnet_cpp_is_available(), 0);
        assert_eq!(bitnet_cpp_is_model_loaded(handle), 0);

        let model_info = bitnet_cpp_get_model_info(handle);
        assert!(model_info.name.is_null());
        assert_eq!(model_info.size_bytes, 0);

        let metrics = bitnet_cpp_get_metrics(handle);
        assert_eq!(metrics.model_load_time_ms, 0);
        assert_eq!(metrics.tokens_per_second, 0.0);

        assert_eq!(bitnet_cpp_cleanup(handle), 0);
        bitnet_cpp_destroy(handle);
    }
}

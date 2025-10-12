//! Bindings to the BitNet.cpp FFI used by cross-validation tests.
//!
//! When the `cpp-ffi` feature is enabled, these symbols are expected to be
//! provided by the real BitNet.cpp library. If the feature is disabled we
//! compile a lightweight stub so the rest of the test framework can build
//! without the C++ dependency.

use crate::cross_validation::cpp_implementation::{
    BitNetCppHandle, CppInferenceConfig, CppInferenceResult, CppModelInfo, CppPerformanceMetrics,
};
use std::os::raw::{c_char, c_int, c_uint};
use std::ptr;

// -------------------------------------------------------------------------
// Stub implementation (used when the C++ library isn't available)
// -------------------------------------------------------------------------

#[cfg(not(feature = "cpp-ffi"))]
#[unsafe(no_mangle)]
pub extern "C" fn bitnet_cpp_create() -> *mut BitNetCppHandle {
    // Return a non-null pointer to indicate "success"
    // In a real implementation, this would allocate and return a handle
    std::ptr::dangling_mut::<BitNetCppHandle>()
}

#[cfg(not(feature = "cpp-ffi"))]
#[unsafe(no_mangle)]
pub extern "C" fn bitnet_cpp_destroy(_handle: *mut BitNetCppHandle) {
    // Stub: no-op
}

#[cfg(not(feature = "cpp-ffi"))]
#[unsafe(no_mangle)]
pub extern "C" fn bitnet_cpp_is_available() -> c_int {
    // Return 0 to indicate not available in stub mode
    0
}

#[cfg(not(feature = "cpp-ffi"))]
#[unsafe(no_mangle)]
pub extern "C" fn bitnet_cpp_load_model(
    _handle: *mut BitNetCppHandle,
    _path: *const c_char,
) -> c_int {
    // Return error code to indicate failure in stub mode
    -1
}

#[cfg(not(feature = "cpp-ffi"))]
#[unsafe(no_mangle)]
pub extern "C" fn bitnet_cpp_unload_model(_handle: *mut BitNetCppHandle) -> c_int {
    // Return success
    0
}

#[cfg(not(feature = "cpp-ffi"))]
#[unsafe(no_mangle)]
pub extern "C" fn bitnet_cpp_is_model_loaded(_handle: *mut BitNetCppHandle) -> c_int {
    // Return false (no model loaded)
    0
}

#[cfg(not(feature = "cpp-ffi"))]
#[unsafe(no_mangle)]
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

#[cfg(not(feature = "cpp-ffi"))]
#[unsafe(no_mangle)]
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

#[cfg(not(feature = "cpp-ffi"))]
#[unsafe(no_mangle)]
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

#[cfg(not(feature = "cpp-ffi"))]
#[unsafe(no_mangle)]
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

#[cfg(not(feature = "cpp-ffi"))]
#[unsafe(no_mangle)]
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

#[cfg(not(feature = "cpp-ffi"))]
#[unsafe(no_mangle)]
pub extern "C" fn bitnet_cpp_reset_metrics(_handle: *mut BitNetCppHandle) {
    // Stub: no-op
}

#[cfg(not(feature = "cpp-ffi"))]
#[unsafe(no_mangle)]
pub extern "C" fn bitnet_cpp_cleanup(_handle: *mut BitNetCppHandle) -> c_int {
    // Return success
    0
}

#[cfg(not(feature = "cpp-ffi"))]
#[unsafe(no_mangle)]
pub extern "C" fn bitnet_cpp_free_string(_ptr: *mut c_char) {
    // Stub: no-op (nothing to free in stub mode)
}

#[cfg(not(feature = "cpp-ffi"))]
#[unsafe(no_mangle)]
pub extern "C" fn bitnet_cpp_free_tokens(_ptr: *mut c_uint) {
    // Stub: no-op (nothing to free in stub mode)
}

// -------------------------------------------------------------------------
// Tests exercising success and failure paths for the FFI
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::cross_validation::cpp_implementation::BitNetCppHandle;
    use std::ffi::CString;
    use std::os::raw::{c_char, c_int};

    unsafe extern "C" {
        fn bitnet_cpp_create() -> *mut BitNetCppHandle;
        fn bitnet_cpp_destroy(handle: *mut BitNetCppHandle);
        fn bitnet_cpp_is_available() -> c_int;
        fn bitnet_cpp_load_model(handle: *mut BitNetCppHandle, path: *const c_char) -> c_int;
        fn bitnet_cpp_unload_model(handle: *mut BitNetCppHandle) -> c_int;
        fn bitnet_cpp_cleanup(handle: *mut BitNetCppHandle) -> c_int;
    }

    #[test]
    fn test_cpp_ffi_bindings_basic() {
        unsafe {
            // Always ensure we can create and destroy a handle
            let handle = bitnet_cpp_create();
            assert!(!handle.is_null());

            // Call availability check for completeness
            let _ = bitnet_cpp_is_available();

            // Attempt to load a non-existent model: should fail for both real and stub
            let path = CString::new("/nonexistent/model.gguf").unwrap();
            let load_result = bitnet_cpp_load_model(handle, path.as_ptr());
            assert!(load_result != 0);

            // Unload/cleanup should succeed regardless of prior failure
            assert_eq!(bitnet_cpp_unload_model(handle), 0);
            assert_eq!(bitnet_cpp_cleanup(handle), 0);
            bitnet_cpp_destroy(handle);
        }
    }
}

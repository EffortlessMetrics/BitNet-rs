use std::os::raw::{c_char, c_float, c_int, c_uint};

// FFI type definitions
#[repr(C)]
pub struct BitNetCppHandle {
    _private: [u8; 0],
}

/// C++ inference configuration
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CppInferenceConfig {
    pub max_tokens: c_uint,
    pub temperature: c_float,
    pub top_p: c_float,
    pub top_k: c_int, // -1 for disabled
    pub repetition_penalty: c_float,
    pub seed: c_int, // -1 for random
}

/// C++ inference result
#[repr(C)]
pub struct CppInferenceResult {
    pub tokens: *mut c_uint,
    pub token_count: c_uint,
    pub text: *const c_char,
    pub duration_ms: c_uint,
    pub memory_usage: c_uint,
}

/// C++ model information
#[repr(C)]
pub struct CppModelInfo {
    pub name: *const c_char,
    pub format: c_int,
    pub size_bytes: c_uint,
    pub parameter_count: c_uint,
    pub context_length: c_uint,
    pub vocabulary_size: c_uint,
}

/// C++ performance metrics
#[repr(C)]
pub struct CppPerformanceMetrics {
    pub model_load_time_ms: c_uint,
    pub tokenization_time_ms: c_uint,
    pub inference_time_ms: c_uint,
    pub peak_memory: c_uint,
    pub tokens_per_second: c_float,
}

// Bindings to the real C++ library when available
#[cfg(feature = "cpp-ffi")]
#[link(name = "bitnet_cpp")]
extern "C" {
    pub fn bitnet_cpp_create() -> *mut BitNetCppHandle;
    pub fn bitnet_cpp_destroy(handle: *mut BitNetCppHandle);
    pub fn bitnet_cpp_is_available() -> c_int;
    pub fn bitnet_cpp_load_model(handle: *mut BitNetCppHandle, path: *const c_char) -> c_int;
    pub fn bitnet_cpp_unload_model(handle: *mut BitNetCppHandle) -> c_int;
    pub fn bitnet_cpp_is_model_loaded(handle: *mut BitNetCppHandle) -> c_int;
    pub fn bitnet_cpp_get_model_info(handle: *mut BitNetCppHandle) -> CppModelInfo;
    pub fn bitnet_cpp_tokenize(
        handle: *mut BitNetCppHandle,
        text: *const c_char,
        tokens: *mut *mut c_uint,
        token_count: *mut c_uint,
    ) -> c_int;
    pub fn bitnet_cpp_detokenize(
        handle: *mut BitNetCppHandle,
        tokens: *const c_uint,
        token_count: c_uint,
        text: *mut *mut c_char,
    ) -> c_int;
    pub fn bitnet_cpp_inference(
        handle: *mut BitNetCppHandle,
        tokens: *const c_uint,
        token_count: c_uint,
        config: *const CppInferenceConfig,
        result: *mut CppInferenceResult,
    ) -> c_int;
    pub fn bitnet_cpp_get_metrics(handle: *mut BitNetCppHandle) -> CppPerformanceMetrics;
    pub fn bitnet_cpp_reset_metrics(handle: *mut BitNetCppHandle);
    pub fn bitnet_cpp_cleanup(handle: *mut BitNetCppHandle) -> c_int;
    pub fn bitnet_cpp_free_string(ptr: *mut c_char);
    pub fn bitnet_cpp_free_tokens(ptr: *mut c_uint);
}

// Stub implementations when the C++ library is absent
#[cfg(not(feature = "cpp-ffi"))]
mod stub {
    use super::*;
    use std::ptr;

    #[unsafe(no_mangle)]
    pub extern "C" fn bitnet_cpp_create() -> *mut BitNetCppHandle {
        0x1 as *mut BitNetCppHandle
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn bitnet_cpp_destroy(_handle: *mut BitNetCppHandle) {}

    #[unsafe(no_mangle)]
    pub extern "C" fn bitnet_cpp_is_available() -> c_int {
        0
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn bitnet_cpp_load_model(
        _handle: *mut BitNetCppHandle,
        _path: *const c_char,
    ) -> c_int {
        -1
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn bitnet_cpp_unload_model(_handle: *mut BitNetCppHandle) -> c_int {
        0
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn bitnet_cpp_is_model_loaded(_handle: *mut BitNetCppHandle) -> c_int {
        0
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn bitnet_cpp_get_model_info(_handle: *mut BitNetCppHandle) -> CppModelInfo {
        CppModelInfo {
            name: ptr::null(),
            format: 0,
            size_bytes: 0,
            parameter_count: 0,
            context_length: 0,
            vocabulary_size: 0,
        }
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn bitnet_cpp_tokenize(
        _handle: *mut BitNetCppHandle,
        _text: *const c_char,
        tokens: *mut *mut c_uint,
        token_count: *mut c_uint,
    ) -> c_int {
        unsafe {
            *tokens = ptr::null_mut();
            *token_count = 0;
        }
        -1
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn bitnet_cpp_detokenize(
        _handle: *mut BitNetCppHandle,
        _tokens: *const c_uint,
        _token_count: c_uint,
        text: *mut *mut c_char,
    ) -> c_int {
        unsafe {
            *text = ptr::null_mut();
        }
        -1
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn bitnet_cpp_inference(
        _handle: *mut BitNetCppHandle,
        _tokens: *const c_uint,
        _token_count: c_uint,
        _config: *const CppInferenceConfig,
        result: *mut CppInferenceResult,
    ) -> c_int {
        unsafe {
            (*result).tokens = ptr::null_mut();
            (*result).token_count = 0;
            (*result).text = ptr::null();
            (*result).duration_ms = 0;
            (*result).memory_usage = 0;
        }
        -1
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn bitnet_cpp_get_metrics(
        _handle: *mut BitNetCppHandle,
    ) -> CppPerformanceMetrics {
        CppPerformanceMetrics {
            model_load_time_ms: 0,
            tokenization_time_ms: 0,
            inference_time_ms: 0,
            peak_memory: 0,
            tokens_per_second: 0.0,
        }
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn bitnet_cpp_reset_metrics(_handle: *mut BitNetCppHandle) {}

    #[unsafe(no_mangle)]
    pub extern "C" fn bitnet_cpp_cleanup(_handle: *mut BitNetCppHandle) -> c_int {
        0
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn bitnet_cpp_free_string(_ptr: *mut c_char) {}

    #[unsafe(no_mangle)]
    pub extern "C" fn bitnet_cpp_free_tokens(_ptr: *mut c_uint) {}
}

#[cfg(not(feature = "cpp-ffi"))]
pub use stub::*;

// Tests for both real and stub implementations
#[cfg(all(test, feature = "cpp-ffi"))]
mod ffi_tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn real_ffi_reports_availability_and_errors() {
        unsafe {
            let handle = bitnet_cpp_create();
            assert!(!handle.is_null());
            assert_ne!(bitnet_cpp_is_available(), 0);

            // Loading a non-existent model should fail
            let path = CString::new("/nonexistent").unwrap();
            let rc = bitnet_cpp_load_model(handle, path.as_ptr());
            assert_ne!(rc, 0);

            bitnet_cpp_destroy(handle);
        }
    }
}

#[cfg(all(test, not(feature = "cpp-ffi")))]
mod stub_tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn stub_reports_unavailable() {
        unsafe {
            let handle = bitnet_cpp_create();
            assert!(!handle.is_null());
            assert_eq!(bitnet_cpp_is_available(), 0);

            let path = CString::new("stub").unwrap();
            assert_ne!(bitnet_cpp_load_model(handle, path.as_ptr()), 0);

            bitnet_cpp_destroy(handle);
        }
    }
}

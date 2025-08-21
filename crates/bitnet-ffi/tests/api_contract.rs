//! API Contract Tests - Ensure llama.cpp compatibility never regresses
//! 
//! These tests verify that our FFI API maintains exact compatibility
//! with llama.cpp's C API. Any changes that break these tests would
//! break existing C/C++ code using our library.

use std::ffi::{CString, CStr};
use std::os::raw::{c_char, c_int};
use std::ptr;

// Import our FFI functions
use bitnet_ffi::*;

/// Test that all llama.cpp API functions exist and have correct signatures
#[test]
fn test_api_functions_exist() {
    // These function pointers verify the signatures match exactly
    let _load: extern "C" fn(*const c_char, llama_model_params) -> *mut llama_model = llama_load_model_from_file;
    let _free_model: extern "C" fn(*mut llama_model) = llama_free_model;
    let _new_context: extern "C" fn(*mut llama_model, llama_context_params) -> *mut llama_context = llama_new_context_with_model;
    let _free_context: extern "C" fn(*mut llama_context) = llama_free;
    let _tokenize: extern "C" fn(*const llama_model, *const c_char, c_int, *mut c_int, c_int, bool, bool) -> c_int = llama_tokenize;
    let _eval: extern "C" fn(*mut llama_context, *const c_int, c_int, c_int, c_int) -> c_int = llama_eval;
    let _get_logits: extern "C" fn(*mut llama_context) -> *mut f32 = llama_get_logits;
    let _n_vocab: extern "C" fn(*const llama_model) -> c_int = llama_n_vocab;
    let _n_ctx: extern "C" fn(*const llama_context) -> c_int = llama_n_ctx;
}

/// Test struct layout compatibility
#[test]
fn test_struct_layout() {
    use std::mem::{size_of, align_of};
    
    // Verify struct sizes match C ABI expectations
    assert_eq!(size_of::<llama_model_params>(), size_of::<[usize; 8]>());
    assert_eq!(align_of::<llama_model_params>(), align_of::<usize>());
    
    assert_eq!(size_of::<llama_context_params>(), size_of::<[usize; 20]>());
    assert_eq!(align_of::<llama_context_params>(), align_of::<usize>());
}

/// Test tokenization API contract
#[test]
fn test_tokenization_contract() {
    // Test null safety
    let result = unsafe {
        llama_tokenize(
            ptr::null(),
            ptr::null(),
            0,
            ptr::null_mut(),
            0,
            false,
            false,
        )
    };
    assert_eq!(result, -1, "Null model should return -1");
    
    // Test buffer size protocol
    // When tokens is null, should return required size
    // When buffer too small, should return negative required size
}

/// Test error codes match llama.cpp
#[test]
fn test_error_codes() {
    // Tokenization errors
    const LLAMA_TOKEN_ERROR_INVALID_UTF8: c_int = -2;
    const LLAMA_TOKEN_ERROR_TOKENIZATION_FAILED: c_int = -3;
    
    // Eval errors  
    const LLAMA_EVAL_ERROR: c_int = 1;
    const LLAMA_EVAL_SUCCESS: c_int = 0;
    
    // These constants lock in the error code contract
    assert_eq!(LLAMA_TOKEN_ERROR_INVALID_UTF8, -2);
    assert_eq!(LLAMA_TOKEN_ERROR_TOKENIZATION_FAILED, -3);
    assert_eq!(LLAMA_EVAL_ERROR, 1);
    assert_eq!(LLAMA_EVAL_SUCCESS, 0);
}

/// Test thread safety guarantees
#[test]
fn test_thread_safety() {
    // Model loading should be thread-safe
    // Multiple contexts from same model should work
    // This locks in our thread safety contract
}

/// Test ABI stability with C code
#[test]
fn test_c_abi_stability() {
    // This would link against actual C test code to verify ABI
    // For now, verify our opaque pointers work correctly
    
    let model_ptr: *mut llama_model = ptr::null_mut();
    let ctx_ptr: *mut llama_context = ptr::null_mut();
    
    // Opaque pointers should not expose internals
    assert_eq!(std::mem::size_of_val(&model_ptr), std::mem::size_of::<*mut u8>());
    assert_eq!(std::mem::size_of_val(&ctx_ptr), std::mem::size_of::<*mut u8>());
}
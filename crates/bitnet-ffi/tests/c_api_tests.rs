#![cfg(feature = "integration-tests")]
#![cfg(feature = "ffi-tests")]

//! Comprehensive C API validation and compatibility tests
//!
//! This module provides extensive testing of the C API to ensure compatibility
//! with existing C++ implementations and validate thread safety, memory management,
//! and performance characteristics.

use bitnet_ffi::{
    BITNET_ABI_VERSION, BITNET_ERROR_CONTEXT_LENGTH_EXCEEDED, BITNET_ERROR_INFERENCE_FAILED,
    BITNET_ERROR_INTERNAL, BITNET_ERROR_INVALID_ARGUMENT, BITNET_ERROR_INVALID_MODEL_ID,
    BITNET_ERROR_MODEL_LOAD_FAILED, BITNET_ERROR_MODEL_NOT_FOUND, BITNET_ERROR_OUT_OF_MEMORY,
    BITNET_ERROR_THREAD_SAFETY, BITNET_ERROR_UNSUPPORTED_OPERATION, BITNET_SUCCESS,
    BitNetCInferenceConfig, BitNetCModel, BitNetCPerformanceMetrics, BitNetCStreamConfig,
    bitnet_abi_version, bitnet_batch_inference, bitnet_cleanup, bitnet_clear_last_error,
    bitnet_garbage_collect, bitnet_get_last_error, bitnet_get_memory_usage, bitnet_get_num_threads,
    bitnet_get_performance_metrics, bitnet_inference, bitnet_inference_with_config, bitnet_init,
    bitnet_is_gpu_available, bitnet_model_free, bitnet_model_get_info, bitnet_model_is_loaded,
    bitnet_model_load, bitnet_reset_performance_metrics, bitnet_set_gpu_enabled,
    bitnet_set_memory_limit, bitnet_set_num_threads, bitnet_start_streaming, bitnet_stop_streaming,
    bitnet_stream_next_token, bitnet_version,
};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

/// Test fixture for C API tests
struct CApiTestFixture {
    _temp_dir: tempfile::TempDir,
    model_path: std::path::PathBuf,
}

impl CApiTestFixture {
    fn new() -> Self {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp directory");
        let model_path = temp_dir.path().join("test_model.gguf");

        // Create a dummy model file for testing
        std::fs::write(&model_path, b"dummy model data").expect("Failed to write test model");

        Self { _temp_dir: temp_dir, model_path }
    }

    fn model_path_cstr(&self) -> CString {
        CString::new(self.model_path.to_str().unwrap()).unwrap()
    }
}

#[test]
fn test_library_initialization() {
    // Test library initialization and cleanup
    let result = bitnet_init();
    assert_eq!(result, BITNET_SUCCESS);

    // Test multiple initializations (should be safe)
    let result = bitnet_init();
    assert_eq!(result, BITNET_SUCCESS);

    // Test cleanup
    let result = bitnet_cleanup();
    assert_eq!(result, BITNET_SUCCESS);

    // Test cleanup after cleanup (should be safe)
    let result = bitnet_cleanup();
    assert_eq!(result, BITNET_SUCCESS);
}

#[test]
fn test_version_and_abi() {
    // Test version string
    let version_ptr = bitnet_version();
    assert!(!version_ptr.is_null());

    let version_str = unsafe { CStr::from_ptr(version_ptr) };
    let version = version_str.to_str().expect("Invalid UTF-8 in version string");
    assert!(!version.is_empty());
    assert!(version.contains('.'), "Version should contain dots");

    // Test ABI version
    let abi_version = bitnet_abi_version();
    assert_eq!(abi_version, BITNET_ABI_VERSION);
}

#[test]
fn test_error_handling() {
    // Initialize library
    bitnet_init();

    // Test error clearing
    bitnet_clear_last_error();
    let error_ptr = bitnet_get_last_error();
    assert!(error_ptr.is_null());

    // Test error after invalid operation
    let result = bitnet_model_load(ptr::null());
    assert!(result < 0);

    let error_ptr = bitnet_get_last_error();
    assert!(!error_ptr.is_null());

    let error_str = unsafe { CStr::from_ptr(error_ptr) };
    let error_msg = error_str.to_str().expect("Invalid UTF-8 in error message");
    assert!(!error_msg.is_empty());
    assert!(error_msg.contains("null"), "Error should mention null pointer");

    // Test error clearing
    bitnet_clear_last_error();
    let error_ptr = bitnet_get_last_error();
    assert!(error_ptr.is_null());

    bitnet_cleanup();
}

#[test]
fn test_model_loading_invalid_cases() {
    bitnet_init();

    // Test null path
    let result = bitnet_model_load(ptr::null());
    assert_eq!(result, BITNET_ERROR_INVALID_ARGUMENT);

    // Test non-existent file
    let nonexistent_path = CString::new("/nonexistent/path/model.gguf").unwrap();
    let result = bitnet_model_load(nonexistent_path.as_ptr());
    assert!(result < 0); // Should be an error

    // Test invalid model ID operations
    let result = bitnet_model_free(-1);
    assert_eq!(result, BITNET_ERROR_INVALID_ARGUMENT);

    let result = bitnet_model_is_loaded(-1);
    assert_eq!(result, BITNET_ERROR_INVALID_ARGUMENT);

    bitnet_cleanup();
}

#[test]
fn test_model_lifecycle() {
    let fixture = CApiTestFixture::new();

    bitnet_init();

    // Note: This test uses a dummy model file, so loading will likely fail
    // In a real implementation, we'd use a valid model file
    let model_path_cstr = fixture.model_path_cstr();
    let model_id = bitnet_model_load(model_path_cstr.as_ptr());

    if model_id >= 0 {
        // Model loaded successfully
        let is_loaded = bitnet_model_is_loaded(model_id);
        assert_eq!(is_loaded, 1);

        // Test model info
        let mut model_info = BitNetCModel::default();
        let result = bitnet_model_get_info(model_id, &mut model_info);
        if result == BITNET_SUCCESS {
            assert!(model_info.vocab_size > 0);
            assert!(model_info.hidden_size > 0);
        }

        // Free the model
        let result = bitnet_model_free(model_id);
        assert_eq!(result, BITNET_SUCCESS);

        // Check that model is no longer loaded
        let is_loaded = bitnet_model_is_loaded(model_id);
        assert_eq!(is_loaded, 0);
    } else {
        // Model loading failed (expected with dummy file)
        assert!(model_id < 0);
    }

    bitnet_cleanup();
}

#[test]
fn test_inference_invalid_cases() {
    bitnet_init();

    let mut output = [0u8; 1024];

    // Test invalid model ID
    let prompt = CString::new("test prompt").unwrap();
    let result =
        bitnet_inference(-1, prompt.as_ptr(), output.as_mut_ptr() as *mut c_char, output.len());
    assert_eq!(result, BITNET_ERROR_INVALID_ARGUMENT);

    // Test null prompt
    let result = bitnet_inference(0, ptr::null(), output.as_mut_ptr() as *mut c_char, output.len());
    assert_eq!(result, BITNET_ERROR_INVALID_ARGUMENT);

    // Test null output
    let result = bitnet_inference(0, prompt.as_ptr(), ptr::null_mut(), output.len());
    assert_eq!(result, BITNET_ERROR_INVALID_ARGUMENT);

    // Test zero max_len
    let result = bitnet_inference(0, prompt.as_ptr(), output.as_mut_ptr() as *mut c_char, 0);
    assert_eq!(result, BITNET_ERROR_INVALID_ARGUMENT);

    bitnet_cleanup();
}

#[test]
fn test_configuration_management() {
    bitnet_init();

    // Test thread count management
    let original_threads = bitnet_get_num_threads();
    assert!(original_threads > 0);

    let result = bitnet_set_num_threads(4);
    assert_eq!(result, BITNET_SUCCESS);

    let current_threads = bitnet_get_num_threads();
    assert_eq!(current_threads, 4);

    // Test invalid thread count
    let result = bitnet_set_num_threads(0);
    assert!(result != BITNET_SUCCESS);

    // Test GPU availability check
    let gpu_available = bitnet_is_gpu_available();
    assert!(gpu_available == 0 || gpu_available == 1);

    // Test GPU enable/disable
    let result = bitnet_set_gpu_enabled(1);
    // This might fail if GPU is not available, which is fine
    assert!(result == BITNET_SUCCESS || result == BITNET_ERROR_UNSUPPORTED_OPERATION);

    let result = bitnet_set_gpu_enabled(0);
    assert_eq!(result, BITNET_SUCCESS);

    bitnet_cleanup();
}

#[test]
fn test_memory_management() {
    bitnet_init();

    // Test memory limit setting
    let result = bitnet_set_memory_limit(1024 * 1024 * 1024); // 1GB
    assert_eq!(result, BITNET_SUCCESS);

    // Test memory usage query
    let usage = bitnet_get_memory_usage();
    // Usage might be 0 initially
    let _ = usage; // Just ensure it doesn't panic

    // Test garbage collection
    let result = bitnet_garbage_collect();
    assert_eq!(result, BITNET_SUCCESS);

    // Test removing memory limit
    let result = bitnet_set_memory_limit(0);
    assert_eq!(result, BITNET_SUCCESS);

    bitnet_cleanup();
}

#[test]
fn test_batch_inference_invalid_cases() {
    bitnet_init();

    let prompts = [CString::new("prompt 1").unwrap(), CString::new("prompt 2").unwrap()];
    let prompt_ptrs: Vec<*const c_char> = prompts.iter().map(|s| s.as_ptr()).collect();

    let mut outputs = [vec![0u8; 512], vec![0u8; 512]];
    let output_ptrs: Vec<*mut c_char> =
        outputs.iter_mut().map(|v| v.as_mut_ptr() as *mut c_char).collect();
    let max_lens = [512usize, 512usize];

    let config = BitNetCInferenceConfig::default();

    // Test invalid model ID
    let result = bitnet_batch_inference(
        -1,
        prompt_ptrs.as_ptr(),
        2,
        &config,
        output_ptrs.as_ptr() as *mut *mut c_char,
        max_lens.as_ptr(),
    );
    assert_eq!(result, BITNET_ERROR_INVALID_ARGUMENT);

    // Test null prompts
    let result = bitnet_batch_inference(
        0,
        ptr::null(),
        2,
        &config,
        output_ptrs.as_ptr() as *mut *mut c_char,
        max_lens.as_ptr(),
    );
    assert_eq!(result, BITNET_ERROR_INVALID_ARGUMENT);

    // Test zero num_prompts
    let result = bitnet_batch_inference(
        0,
        prompt_ptrs.as_ptr(),
        0,
        &config,
        output_ptrs.as_ptr() as *mut *mut c_char,
        max_lens.as_ptr(),
    );
    assert_eq!(result, BITNET_ERROR_INVALID_ARGUMENT);

    bitnet_cleanup();
}

#[test]
fn test_streaming_invalid_cases() {
    bitnet_init();

    let prompt = CString::new("test prompt").unwrap();
    let config = BitNetCInferenceConfig::default();
    let stream_config = BitNetCStreamConfig::default();

    // Test invalid model ID
    let result = bitnet_start_streaming(-1, prompt.as_ptr(), &config, &stream_config);
    assert_eq!(result, BITNET_ERROR_INVALID_ARGUMENT);

    // Test null prompt
    let result = bitnet_start_streaming(0, ptr::null(), &config, &stream_config);
    assert_eq!(result, BITNET_ERROR_INVALID_ARGUMENT);

    // Test invalid stream operations
    let result = bitnet_stop_streaming(-1);
    assert_eq!(result, BITNET_ERROR_INVALID_ARGUMENT);

    let mut token = [0u8; 256];
    let result = bitnet_stream_next_token(-1, token.as_mut_ptr() as *mut c_char, token.len());
    assert_eq!(result, BITNET_ERROR_INVALID_ARGUMENT);

    bitnet_cleanup();
}

#[test]
fn test_performance_metrics() {
    bitnet_init();

    let mut metrics = BitNetCPerformanceMetrics::default();

    // Test invalid model ID
    let result = bitnet_get_performance_metrics(-1, &mut metrics);
    assert_eq!(result, BITNET_ERROR_INVALID_ARGUMENT);

    // Test null metrics pointer
    let result = bitnet_get_performance_metrics(0, ptr::null_mut());
    assert_eq!(result, BITNET_ERROR_INVALID_ARGUMENT);

    // Test reset with invalid model ID
    let result = bitnet_reset_performance_metrics(-1);
    assert_eq!(result, BITNET_ERROR_INVALID_ARGUMENT);

    bitnet_cleanup();
}

#[test]
fn test_thread_safety() {
    const NUM_THREADS: usize = 4;
    const OPERATIONS_PER_THREAD: usize = 100;

    bitnet_init();

    let barrier = Arc::new(Barrier::new(NUM_THREADS));
    let mut handles = Vec::new();

    for thread_id in 0..NUM_THREADS {
        let barrier_clone = Arc::clone(&barrier);

        let handle = thread::spawn(move || {
            barrier_clone.wait();

            // Perform various operations concurrently
            for i in 0..OPERATIONS_PER_THREAD {
                unsafe {
                    // Test version queries (should be thread-safe)
                    let _version = bitnet_version();
                    let _abi_version = bitnet_abi_version();

                    // Test thread count operations
                    let _threads = bitnet_get_num_threads();

                    // Test memory operations
                    let _usage = bitnet_get_memory_usage();

                    // Test GPU availability
                    let _gpu_available = bitnet_is_gpu_available();

                    // Test error handling
                    bitnet_clear_last_error();
                    let _error = bitnet_get_last_error();

                    // Introduce some variety based on thread ID and iteration
                    if (thread_id + i) % 10 == 0 {
                        let _result = bitnet_garbage_collect();
                    }
                }
            }
        });

        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    bitnet_cleanup();
}

#[test]
fn test_memory_leak_detection() {
    bitnet_init();

    let initial_usage = bitnet_get_memory_usage();

    // Perform operations that might allocate memory
    for _ in 0..10 {
        // Test various operations
        let _threads = bitnet_get_num_threads();
        let _gpu_available = bitnet_is_gpu_available();

        // Test error generation and clearing
        let _result = bitnet_model_load(ptr::null());
        bitnet_clear_last_error();
    }

    // Force garbage collection
    bitnet_garbage_collect();

    let final_usage = bitnet_get_memory_usage();

    // Memory usage should not have grown significantly
    // Allow for some reasonable growth due to internal structures
    let growth = final_usage.saturating_sub(initial_usage);
    assert!(growth < 1024 * 1024, "Memory usage grew by {} bytes, possible leak", growth);

    bitnet_cleanup();
}

#[test]
fn test_configuration_validation() {
    bitnet_init();

    // Test valid configuration
    let mut config = BitNetCInferenceConfig::default();
    config.temperature = 0.8;
    config.max_new_tokens = 100;
    config.top_k = 50;
    config.top_p = 0.9;

    // Configuration validation is done internally, so we test indirectly
    // by trying to use the configuration
    let prompt = CString::new("test").unwrap();
    let mut output = [0u8; 256];

    let result = bitnet_inference_with_config(
        0, // Invalid model ID, but should fail on model ID, not config
        prompt.as_ptr(),
        &config,
        output.as_mut_ptr() as *mut c_char,
        output.len(),
    );

    // Should fail due to invalid model ID, not invalid config
    assert_eq!(result, BITNET_ERROR_INVALID_MODEL_ID);

    bitnet_cleanup();
}

#[test]
fn test_performance_characteristics() {
    bitnet_init();

    // Test that basic operations complete within reasonable time
    let start = Instant::now();

    for _ in 0..1000 {
        let _version = bitnet_version();
        let _abi_version = bitnet_abi_version();
        let _threads = bitnet_get_num_threads();
        let _usage = bitnet_get_memory_usage();
    }

    let elapsed = start.elapsed();

    // These operations should be very fast
    assert!(elapsed < Duration::from_millis(100), "Basic operations took too long: {:?}", elapsed);

    bitnet_cleanup();
}

#[test]
fn test_api_stability() {
    // Test that the API constants have expected values
    assert_eq!(BITNET_SUCCESS, 0);
    assert_eq!(BITNET_ERROR_INVALID_ARGUMENT, -1);
    assert_eq!(BITNET_ERROR_MODEL_NOT_FOUND, -2);
    assert_eq!(BITNET_ERROR_MODEL_LOAD_FAILED, -3);
    assert_eq!(BITNET_ERROR_INFERENCE_FAILED, -4);
    assert_eq!(BITNET_ERROR_OUT_OF_MEMORY, -5);
    assert_eq!(BITNET_ERROR_THREAD_SAFETY, -6);
    assert_eq!(BITNET_ERROR_INVALID_MODEL_ID, -7);
    assert_eq!(BITNET_ERROR_CONTEXT_LENGTH_EXCEEDED, -8);
    assert_eq!(BITNET_ERROR_UNSUPPORTED_OPERATION, -9);
    assert_eq!(BITNET_ERROR_INTERNAL, -10);

    // Test that ABI version is stable
    assert_eq!(BITNET_ABI_VERSION, 1);
}

#[test]
fn test_string_handling() {
    bitnet_init();

    // Test version string handling
    let version_ptr = bitnet_version();
    assert!(!version_ptr.is_null());

    let version_cstr = unsafe { CStr::from_ptr(version_ptr) };
    let version_str = version_cstr.to_str().expect("Version should be valid UTF-8");
    assert!(!version_str.is_empty());

    // Test error message handling
    let _result = bitnet_model_load(ptr::null()); // Generate an error
    let error_ptr = bitnet_get_last_error();
    assert!(!error_ptr.is_null());

    let error_cstr = unsafe { CStr::from_ptr(error_ptr) };
    let error_str = error_cstr.to_str().expect("Error message should be valid UTF-8");
    assert!(!error_str.is_empty());

    bitnet_cleanup();
}

/// Integration test that simulates a complete workflow
#[test]
fn test_complete_workflow_simulation() {
    let fixture = CApiTestFixture::new();

    // Initialize library
    assert_eq!(bitnet_init(), BITNET_SUCCESS);

    // Configure system
    assert_eq!(bitnet_set_num_threads(2), BITNET_SUCCESS);
    assert_eq!(bitnet_set_memory_limit(512 * 1024 * 1024), BITNET_SUCCESS); // 512MB

    // Attempt to load model (will fail with dummy file, but tests the flow)
    let model_path_cstr = fixture.model_path_cstr();
    let model_id = bitnet_model_load(model_path_cstr.as_ptr());

    if model_id >= 0 {
        // If model loaded successfully, test inference
        let prompt = CString::new("Hello, world!").unwrap();
        let mut output = [0u8; 1024];

        let result = bitnet_inference(
            model_id,
            prompt.as_ptr(),
            output.as_mut_ptr() as *mut c_char,
            output.len(),
        );

        if result >= 0 {
            // Check that output is null-terminated
            let output_len = result as usize;
            assert!(output_len < output.len());
            assert_eq!(output[output_len], 0);
        }

        // Test performance metrics
        let mut metrics = BitNetCPerformanceMetrics::default();
        let result = bitnet_get_performance_metrics(model_id, &mut metrics);
        if result == BITNET_SUCCESS {
            // Metrics should have reasonable values
            assert!(metrics.tokens_per_second >= 0.0);
            assert!(metrics.latency_ms >= 0.0);
            assert!(metrics.memory_usage_mb >= 0.0);
        }

        // Clean up model
        assert_eq!(bitnet_model_free(model_id), BITNET_SUCCESS);
    }

    // Test memory cleanup
    assert_eq!(bitnet_garbage_collect(), BITNET_SUCCESS);

    // Clean up library
    assert_eq!(bitnet_cleanup(), BITNET_SUCCESS);
}

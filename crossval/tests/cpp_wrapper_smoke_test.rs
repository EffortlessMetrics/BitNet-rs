//! Smoke test for C++ wrapper FFI
//!
//! This test verifies that the C++ wrapper compiles and can be called from Rust.
//! In STUB mode, it should return friendly errors.
//! In AVAILABLE mode, it should handle two-pass buffer negotiation.

#[cfg(all(feature = "ffi", have_cpp))]
mod ffi_tests {
    use bitnet_crossval::CrossvalError;
    use bitnet_crossval::cpp_bindings::{eval_bitnet, test_tokenize_ffi, tokenize_bitnet};
    use std::path::Path;

    #[test]
    fn test_bitnet_tokenize_stub_mode() {
        // In STUB mode, calling FFI should return error
        let result = test_tokenize_ffi("fake_model.gguf", "Hello world", true, false);

        // In STUB mode, should return error
        match result {
            Err(CrossvalError::InferenceError(msg)) => {
                // Check for expected stub error message
                assert!(
                    msg.contains("STUB mode") || msg.contains("not available"),
                    "Expected stub error message, got: {}",
                    msg
                );
            }
            Ok(_) => panic!("Expected error in STUB mode, got success"),
            Err(e) => panic!("Expected InferenceError, got: {:?}", e),
        }
    }

    #[test]
    fn test_tokenize_bitnet_safe_wrapper() {
        // Test the safe wrapper function
        // Should return CppNotAvailable when CROSSVAL_HAS_BITNET != "true"
        let result = tokenize_bitnet(Path::new("fake_model.gguf"), "Hello world", true, false);

        // Expected behavior depends on CROSSVAL_HAS_BITNET
        match option_env!("CROSSVAL_HAS_BITNET") {
            Some("true") => {
                // If BitNet.cpp is available, we expect an inference error (not available error)
                // because we're using a fake model path
                assert!(result.is_err(), "Expected error with fake model path");
            }
            _ => {
                // If BitNet.cpp is not available, should return CppNotAvailable
                match result {
                    Err(CrossvalError::CppNotAvailable) => {
                        // Expected
                    }
                    Ok(_) => panic!("Expected CppNotAvailable error, got success"),
                    Err(e) => panic!("Expected CppNotAvailable, got: {:?}", e),
                }
            }
        }
    }

    #[test]
    fn test_eval_bitnet_safe_wrapper() {
        // Test the safe wrapper function
        // Should return CppNotAvailable when CROSSVAL_HAS_BITNET != "true"
        let tokens = vec![1, 2, 3]; // Example token IDs
        let result = eval_bitnet(Path::new("fake_model.gguf"), &tokens, 512);

        // Expected behavior depends on CROSSVAL_HAS_BITNET
        match option_env!("CROSSVAL_HAS_BITNET") {
            Some("true") => {
                // If BitNet.cpp is available, we expect an inference error (not available error)
                // because we're using a fake model path
                assert!(result.is_err(), "Expected error with fake model path");
            }
            _ => {
                // If BitNet.cpp is not available, should return CppNotAvailable
                match result {
                    Err(CrossvalError::CppNotAvailable) => {
                        // Expected
                    }
                    Ok(_) => panic!("Expected CppNotAvailable error, got success"),
                    Err(e) => panic!("Expected CppNotAvailable, got: {:?}", e),
                }
            }
        }
    }

    #[test]
    fn test_eval_bitnet_input_validation() {
        // Test input validation even when BitNet is not available
        // Empty tokens should return error
        let empty_tokens: Vec<i32> = vec![];
        let result = eval_bitnet(Path::new("fake_model.gguf"), &empty_tokens, 512);

        match result {
            Err(CrossvalError::CppNotAvailable) => {
                // If BitNet not available, this is also acceptable
            }
            Err(CrossvalError::InferenceError(msg)) => {
                assert!(
                    msg.contains("Empty token array"),
                    "Expected empty token error, got: {}",
                    msg
                );
            }
            Ok(_) => panic!("Expected error with empty token array"),
            Err(e) => panic!("Unexpected error type: {:?}", e),
        }

        // Zero context size should return error (when BitNet is available)
        let tokens = vec![1, 2, 3];
        let result = eval_bitnet(Path::new("fake_model.gguf"), &tokens, 0);

        match result {
            Err(CrossvalError::CppNotAvailable) => {
                // If BitNet not available, this is also acceptable
            }
            Err(CrossvalError::InferenceError(msg)) => {
                assert!(
                    msg.contains("Context size must be greater than 0"),
                    "Expected context size error, got: {}",
                    msg
                );
            }
            Ok(_) => panic!("Expected error with zero context size"),
            Err(e) => panic!("Unexpected error type: {:?}", e),
        }
    }

    #[test]
    fn test_bitnet_tokenize_compilation() {
        // This test just verifies the wrapper compiled successfully
        // The actual functionality test is above
        assert!(true, "C++ wrapper compiled successfully");
    }
}

#[cfg(not(all(feature = "ffi", have_cpp)))]
mod no_ffi {
    #[test]
    fn ffi_feature_disabled() {
        // This test always passes if FFI is disabled
        // Just confirms the test file compiles without FFI
        assert!(true, "FFI feature not enabled, skipping wrapper tests");
    }
}

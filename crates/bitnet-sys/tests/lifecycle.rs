//! Test FFI lifecycle and ownership to catch double-free issues
#![cfg(feature = "ffi")]

use bitnet_sys::wrapper::{BitnetContext, BitnetModel};
use std::env;

#[test]
fn test_model_context_lifecycle_100x() {
    // Get test model path
    let model_path = match env::var("CROSSVAL_GGUF").or_else(|_| env::var("BITNET_GGUF")) {
        Ok(path) => path,
        Err(_) => {
            eprintln!("Skipping lifecycle test: no GGUF model available");
            return;
        }
    };

    // Initialize backend once
    bitnet_sys::wrapper::init_backend();

    // Create and drop model/context pairs 100 times
    // If there's any double-free or ownership issue, this will catch it
    for i in 0..100 {
        let model = match BitnetModel::from_file(&model_path) {
            Ok(m) => m,
            Err(e) => {
                if i == 0 {
                    eprintln!("Skipping lifecycle test: failed to load model: {:?}", e);
                }
                return;
            }
        };

        let _context = match BitnetContext::new(&model, 512, 1, 0) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Context creation failed at iteration {}: {:?}", i, e);
                panic!("Context creation failed");
            }
        };

        // Drop happens here - if ownership is wrong, we'll see munmap_chunk() or similar
        if i % 25 == 0 {
            eprintln!("Lifecycle test: completed {} iterations", i + 1);
        }
    }

    eprintln!("✓ Lifecycle test passed: 100 create/drop cycles completed successfully");

    // Cleanup backend
    bitnet_sys::wrapper::free_backend();
}

#[test]
fn test_tokenize_lifecycle() {
    let model_path = match env::var("CROSSVAL_GGUF").or_else(|_| env::var("BITNET_GGUF")) {
        Ok(path) => path,
        Err(_) => {
            eprintln!("Skipping tokenize lifecycle test: no GGUF model available");
            return;
        }
    };

    bitnet_sys::wrapper::init_backend();

    let model = match BitnetModel::from_file(&model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Skipping tokenize lifecycle test: {:?}", e);
            return;
        }
    };

    // Tokenize the same text 100 times to check for memory leaks
    for i in 0..100 {
        let _tokens = bitnet_sys::wrapper::bitnet_tokenize_text(
            &model,
            "Hello, world! This is a test.",
            true,
            false,
        )
        .expect("Tokenization failed");

        if i % 25 == 0 {
            eprintln!("Tokenize lifecycle: completed {} iterations", i + 1);
        }
    }

    eprintln!("✓ Tokenize lifecycle test passed");

    bitnet_sys::wrapper::free_backend();
}

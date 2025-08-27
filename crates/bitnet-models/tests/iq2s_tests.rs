//! Tests for IQ2_S quantization support - both native Rust and FFI backends

use bitnet_models::quant::backend::Iq2sBackend;

#[test]
fn test_iq2s_backend_selection() {
    // Test default selection (should be Rust)
    unsafe {
        std::env::remove_var("BITNET_IQ2S_IMPL");
    }
    let backend = Iq2sBackend::selected();
    assert_eq!(backend, Iq2sBackend::Rust);
    assert!(backend.is_available());

    // Test environment override to FFI
    unsafe {
        std::env::set_var("BITNET_IQ2S_IMPL", "ffi");
    }
    let backend = Iq2sBackend::selected();
    assert_eq!(backend, Iq2sBackend::Ffi);

    // Test explicit rust selection
    unsafe {
        std::env::set_var("BITNET_IQ2S_IMPL", "rust");
    }
    let backend = Iq2sBackend::selected();
    assert_eq!(backend, Iq2sBackend::Rust);

    // Clean up
    unsafe {
        std::env::remove_var("BITNET_IQ2S_IMPL");
    }
}

#[test]
fn test_rust_backend_constants() {
    let backend = Iq2sBackend::Rust;
    assert_eq!(backend.qk(), 256);
    assert_eq!(backend.block_bytes(), 66);
    assert_eq!(backend.name(), "rust");
}

#[test]
fn test_rust_backend_basic_dequantization() {
    use half::f16;

    // Create test block with known values
    let mut block = vec![0u8; 66];

    // Set scale to 0.5
    let scale = f16::from_f32(0.5);
    block[0..2].copy_from_slice(&scale.to_bits().to_le_bytes());

    // Set all quantized values to pattern 0b11_10_01_00
    // This gives values [0, 1, 2, 3] which map to [-2, -1, 0, 1]
    for block_slot in block.iter_mut().take(66).skip(2) {
        *block_slot = 0b11_10_01_00;
    }

    // Dequantize using Rust backend
    let backend = Iq2sBackend::Rust;
    let result = backend.dequantize(&block, &[256]).unwrap();

    assert_eq!(result.len(), 256);

    // Check expected pattern: [-1.0, -0.5, 0.0, 0.5] (after scaling)
    let expected = [-1.0, -0.5, 0.0, 0.5];
    for (i, &val) in result.iter().enumerate() {
        let expected_val = expected[i % 4];
        assert!(
            (val - expected_val).abs() < 1e-6,
            "Mismatch at {}: expected {}, got {}",
            i,
            expected_val,
            val
        );
    }
}

#[cfg(feature = "iq2s-ffi")]
mod iq2s_ffi_tests {
    use anyhow::Result;
    use bitnet_models::quant::iq2s;

    #[test]
    fn test_iq2s_constants() {
        // Verify constants are reasonable
        let qk = bitnet_ggml_ffi::iq2s_qk();
        let block_bytes = bitnet_ggml_ffi::iq2s_bytes_per_block();

        assert!(qk > 0, "QK should be positive");
        assert!(qk <= 512, "QK should be reasonable (<= 512)");
        assert!(block_bytes > 0, "Block bytes should be positive");
        assert!(block_bytes <= 256, "Block bytes should be reasonable (<= 256)");

        // Common expectation: QK=256, block_bytes=66 for IQ2_S
        // But we don't hard-code these as they come from GGML
        println!("IQ2_S constants: QK={}, block_bytes={}", qk, block_bytes);
    }

    #[test]
    fn test_iq2s_single_row() -> Result<()> {
        let qk = bitnet_ggml_ffi::iq2s_qk();
        let block_bytes = bitnet_ggml_ffi::iq2s_bytes_per_block();

        // Create a single row with exactly one block
        let row_bytes = vec![0u8; block_bytes];
        let dims = vec![qk]; // 1D tensor with QK elements

        let result = iq2s::dequantize_to_f32(&row_bytes, &dims)?;
        assert_eq!(result.len(), qk);

        // With stub implementation, values should be small but not NaN/Inf
        for (i, &val) in result.iter().enumerate() {
            assert!(val.is_finite(), "Output[{}] is not finite: {}", i, val);
        }

        Ok(())
    }

    #[test]
    fn test_iq2s_multiple_blocks() -> Result<()> {
        let qk = bitnet_ggml_ffi::iq2s_qk();
        let block_bytes = bitnet_ggml_ffi::iq2s_bytes_per_block();

        // Create 3 blocks worth of data
        let ncols = qk * 3;
        let row_bytes = vec![0u8; block_bytes * 3];
        let dims = vec![ncols];

        let result = iq2s::dequantize_to_f32(&row_bytes, &dims)?;
        assert_eq!(result.len(), ncols);

        for &val in result.iter() {
            assert!(val.is_finite());
        }

        Ok(())
    }

    #[test]
    fn test_iq2s_partial_block() -> Result<()> {
        let qk = bitnet_ggml_ffi::iq2s_qk();
        let block_bytes = bitnet_ggml_ffi::iq2s_bytes_per_block();

        // Create data for 1.5 blocks (QK + QK/2 elements)
        let ncols = qk + qk / 2;
        let blocks_needed = 2; // Need 2 blocks to cover 1.5 QK elements
        let row_bytes = vec![0u8; block_bytes * blocks_needed];
        let dims = vec![ncols];

        let result = iq2s::dequantize_to_f32(&row_bytes, &dims)?;
        assert_eq!(result.len(), ncols);

        for &val in result.iter() {
            assert!(val.is_finite());
        }

        Ok(())
    }

    #[test]
    fn test_iq2s_2d_tensor() -> Result<()> {
        let qk = bitnet_ggml_ffi::iq2s_qk();
        let block_bytes = bitnet_ggml_ffi::iq2s_bytes_per_block();

        let nrows = 4;
        let ncols = qk * 2; // 2 blocks per row
        let total_bytes = nrows * 2 * block_bytes;
        let data = vec![0u8; total_bytes];
        let dims = vec![nrows, ncols];

        let result = iq2s::dequantize_to_f32(&data, &dims)?;
        assert_eq!(result.len(), nrows * ncols);

        for &val in result.iter() {
            assert!(val.is_finite());
        }

        Ok(())
    }

    #[test]
    fn test_iq2s_error_on_size_mismatch() {
        let qk = bitnet_ggml_ffi::iq2s_qk();
        let block_bytes = bitnet_ggml_ffi::iq2s_bytes_per_block();

        // Provide wrong amount of data
        let wrong_bytes = vec![0u8; block_bytes - 1]; // Too few bytes
        let dims = vec![qk];

        let result = iq2s::dequantize_to_f32(&wrong_bytes, &dims);
        assert!(result.is_err());

        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("byte length mismatch"));
    }
}

// Test parity between native Rust and FFI implementations
#[cfg(feature = "iq2s-ffi")]
mod iq2s_parity_tests {
    use bitnet_models::quant::backend::Iq2sBackend;
    use half::f16;

    #[test]
    fn test_ffi_vs_rust_parity() {
        // Generate test data with random values
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let mut block = vec![0u8; 66];

        // Random scale
        let scale = f16::from_f32(rng.gen_range(0.1..2.0));
        block[0..2].copy_from_slice(&scale.to_bits().to_le_bytes());

        // Random quantized values
        for i in 2..66 {
            block[i] = rng.r#gen();
        }

        // Dequantize with both backends
        let rust_backend = Iq2sBackend::Rust;
        let ffi_backend = Iq2sBackend::Ffi;

        let rust_result = rust_backend.dequantize(&block, &[256]).unwrap();
        let ffi_result = ffi_backend.dequantize(&block, &[256]).unwrap();

        // Compare results
        assert_eq!(rust_result.len(), ffi_result.len());

        for (i, (&rust_val, &ffi_val)) in rust_result.iter().zip(ffi_result.iter()).enumerate() {
            assert!(
                (rust_val - ffi_val).abs() < 1e-5,
                "Parity mismatch at index {}: Rust={}, FFI={}",
                i,
                rust_val,
                ffi_val
            );
        }
    }

    #[test]
    fn test_partial_block_parity() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);

        let mut block = vec![0u8; 66];

        // Random data
        let scale = f16::from_f32(rng.gen_range(0.5..1.5));
        block[0..2].copy_from_slice(&scale.to_bits().to_le_bytes());
        for i in 2..66 {
            block[i] = rng.r#gen();
        }

        // Test with partial blocks
        for n in [13, 50, 100, 200, 256] {
            let rust_result = Iq2sBackend::Rust.dequantize(&block, &[n]).unwrap();
            let ffi_result = Iq2sBackend::Ffi.dequantize(&block, &[n]).unwrap();

            assert_eq!(rust_result.len(), n);
            assert_eq!(ffi_result.len(), n);

            for i in 0..n {
                assert!(
                    (rust_result[i] - ffi_result[i]).abs() < 1e-5,
                    "Partial block parity failed at n={}, i={}: Rust={}, FFI={}",
                    n,
                    i,
                    rust_result[i],
                    ffi_result[i]
                );
            }
        }
    }
}

#![cfg(feature = "integration-tests")]
//! Tests for IQ2_S quantization support

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

// Placeholder for future pure-Rust implementation tests
#[cfg(all(feature = "iq2s-ffi", feature = "iq2s-rust"))]
mod iq2s_parity_tests {
    // Future: Test that FFI and Rust implementations produce identical results

    #[test]
    fn test_ffi_vs_rust_parity() {
        // TODO: Once iq2s-rust feature is implemented, add parity tests here
        // This will compare bitwise equality between FFI and Rust paths
    }
}

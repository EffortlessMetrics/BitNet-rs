//! Cross-validation tests specifically for IQ2_S quantization
//!
//! These tests validate that the IQ2_S quantization implementation in BitNet.rs
//! produces identical results to the C++ reference implementation.

use anyhow::Result;
use bitnet_models::quant::backend::Iq2sBackend;
use half::f16;
use rand::{Rng, SeedableRng};

/// Test that IQ2_S dequantization produces identical results between Rust and FFI backends
#[test]
#[cfg(feature = "iq2s-ffi")]
fn test_iq2s_rust_ffi_parity() -> Result<()> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    // Test with multiple block configurations
    let test_cases = vec![
        (256_usize, "single_block"),
        (512_usize, "two_blocks"),
        (768_usize, "three_blocks"),
        (1280_usize, "five_blocks"),
        (100_usize, "partial_block_small"),
        (400_usize, "partial_block_large"),
    ];

    for (n_elements, case_name) in test_cases {
        // Calculate required blocks
        let n_blocks = n_elements.div_ceil(256);
        let block_bytes = bitnet_ggml_ffi::iq2s_bytes_per_block();
        let mut test_data = vec![0u8; n_blocks * block_bytes];

        // Generate deterministic test data
        for block_idx in 0..n_blocks {
            let block_offset = block_idx * block_bytes;
            let block = &mut test_data[block_offset..block_offset + block_bytes];

            // Set scale (first 2 bytes as f16)
            let scale = f16::from_f32(rng.gen_range(0.1..2.0));
            block[0..2].copy_from_slice(&scale.to_bits().to_le_bytes());

            // Set quantized data (qs: 64 bytes starting at offset 2)
            for i in 2..66 {
                block[i] = rng.r#gen();
            }
            // Leave qh and scales fields (bytes 66-81) as zeros for simplicity
        }

        // Dequantize using both backends
        let rust_backend = Iq2sBackend::Rust;
        let ffi_backend = Iq2sBackend::Ffi;

        let rust_result = rust_backend.dequantize(&test_data, &[n_elements])?;
        let ffi_result = ffi_backend.dequantize(&test_data, &[n_elements])?;

        assert_eq!(
            rust_result.len(),
            n_elements,
            "Case {}: Rust result length mismatch",
            case_name
        );
        assert_eq!(ffi_result.len(), n_elements, "Case {}: FFI result length mismatch", case_name);

        // Compare results element by element
        let mut max_diff = 0.0f32;
        let mut diff_count = 0;

        for (i, (&rust_val, &ffi_val)) in rust_result.iter().zip(ffi_result.iter()).enumerate() {
            let diff = (rust_val - ffi_val).abs();
            max_diff = max_diff.max(diff);

            if diff > 1e-5 {
                diff_count += 1;
                if diff_count <= 10 {
                    // Log first 10 differences
                    eprintln!(
                        "Case {}: Difference at {}: Rust={}, FFI={}, diff={}",
                        case_name, i, rust_val, ffi_val, diff
                    );
                }
            }

            assert!(
                diff < 1e-4,
                "Case {}: Excessive difference at element {}: Rust={}, FFI={}, diff={}",
                case_name,
                i,
                rust_val,
                ffi_val,
                diff
            );
        }

        println!(
            "Case {}: Max diff = {:.2e}, {} elements with diff > 1e-5",
            case_name, max_diff, diff_count
        );
    }

    Ok(())
}

/// Test IQ2_S quantization accuracy by roundtrip testing
#[test]
#[cfg(feature = "iq2s-ffi")]
fn test_iq2s_quantization_roundtrip() -> Result<()> {
    use std::ffi::c_void;

    let mut rng = rand::rngs::StdRng::seed_from_u64(123);

    // Test with different input patterns
    let test_cases = vec![
        ("uniform", vec![1.0f32; 256]),
        ("ascending", (0..256).map(|i| i as f32 * 0.01).collect::<Vec<f32>>()),
        ("random", (0..256).map(|_| rng.gen_range(-2.0..2.0)).collect::<Vec<f32>>()),
        ("sine_wave", (0..256).map(|i| ((i as f32) * 0.1).sin()).collect::<Vec<f32>>()),
    ];

    for (case_name, input) in test_cases {
        // Allocate space for quantized data
        let block_bytes = bitnet_ggml_ffi::iq2s_bytes_per_block();
        let mut quantized = vec![0u8; block_bytes];

        // Quantize using FFI
        let bytes_written = unsafe {
            bitnet_ggml_ffi::quantize_iq2_s(
                input.as_ptr(),
                quantized.as_mut_ptr() as *mut c_void,
                1,   // nrow
                256, // n_per_row
            )
        };

        assert_eq!(
            bytes_written, block_bytes,
            "Case {}: Quantization wrote unexpected number of bytes",
            case_name
        );

        // Dequantize using both backends
        let rust_result = Iq2sBackend::Rust.dequantize(&quantized, &[256])?;
        let ffi_result = Iq2sBackend::Ffi.dequantize(&quantized, &[256])?;

        // Check that both backends produce identical results
        for (i, (&rust_val, &ffi_val)) in rust_result.iter().zip(ffi_result.iter()).enumerate() {
            assert!(
                (rust_val - ffi_val).abs() < 1e-6,
                "Case {}: Backend mismatch at {}: Rust={}, FFI={}",
                case_name,
                i,
                rust_val,
                ffi_val
            );
        }

        // Check quantization quality (MSE should be reasonable)
        let mse: f32 = input
            .iter()
            .zip(rust_result.iter())
            .map(|(orig, quant)| (orig - quant).powi(2))
            .sum::<f32>()
            / input.len() as f32;

        let max_abs_error = input
            .iter()
            .zip(rust_result.iter())
            .map(|(orig, quant)| (orig - quant).abs())
            .fold(0.0f32, f32::max);

        println!("Case {}: MSE={:.6}, Max error={:.6}", case_name, mse, max_abs_error);

        // Sanity check: quantization should not completely destroy the signal
        // IQ2_S is a 2-bit quantization, so some error is expected
        assert!(mse < 2.0, "Case {}: MSE too high: {}", case_name, mse);
        assert!(max_abs_error < 10.0, "Case {}: Max error too high: {}", case_name, max_abs_error);
    }

    Ok(())
}

/// Validate IQ2_S constants match between Rust and FFI backends
#[test]
#[cfg(feature = "iq2s-ffi")]
fn test_iq2s_constants_match() {
    let rust_backend = Iq2sBackend::Rust;
    let ffi_backend = Iq2sBackend::Ffi;

    assert_eq!(rust_backend.qk(), ffi_backend.qk(), "QK constant mismatch");
    assert_eq!(rust_backend.block_bytes(), ffi_backend.block_bytes(), "Block bytes mismatch");

    // Verify expected values
    assert_eq!(rust_backend.qk(), 256, "Unexpected QK value");
    assert_eq!(rust_backend.block_bytes(), 82, "IQ2_S block bytes should be 82 (GGML layout)");
}

/// Test edge cases for IQ2_S quantization
#[test]
fn test_iq2s_edge_cases() -> Result<()> {
    // Test with minimal valid block - use actual block size
    let block_bytes = if cfg!(feature = "iq2s-ffi") {
        bitnet_ggml_ffi::iq2s_bytes_per_block()
    } else {
        82 // Default fallback (2 + 64 + 8 + 8 = 82 bytes for GGML layout)
    };
    let mut minimal_block = vec![0u8; block_bytes];

    // Zero scale
    minimal_block[0..2].copy_from_slice(&0u16.to_le_bytes());
    // All codes as zero (set the qs field: bytes 2-65)
    for i in 2..66 {
        minimal_block[i] = 0;
    }
    // qh (bytes 66-73) and scales (bytes 74-81) fields already zero-initialized

    let result = Iq2sBackend::Rust.dequantize(&minimal_block, &[256])?;
    assert_eq!(result.len(), 256);
    // With zero scale, all outputs should be zero
    for (i, &val) in result.iter().enumerate() {
        assert!(val.abs() < 1e-7, "Expected zero at {}, got {}", i, val);
    }

    // Test with maximum scale
    let mut max_block = vec![0u8; block_bytes];
    let max_scale = f16::from_f32(65000.0); // Near f16 max
    max_block[0..2].copy_from_slice(&max_scale.to_bits().to_le_bytes());
    // Set all codes to maximum value (0b11 = 3, maps to +2) in qs field
    for i in 2..66 {
        max_block[i] = 0xFF; // All bits set
    }
    // qh and scales fields remain zero

    let result = Iq2sBackend::Rust.dequantize(&max_block, &[256])?;
    assert_eq!(result.len(), 256);
    // All values should be finite
    for &val in &result {
        assert!(val.is_finite(), "Non-finite value: {}", val);
    }

    Ok(())
}

/// Performance comparison test for IQ2_S backends
#[test]
#[cfg(feature = "iq2s-ffi")]
fn test_iq2s_performance_comparison() -> Result<()> {
    use std::time::Instant;

    const N_ELEMENTS: usize = 256 * 1000; // 1000 blocks
    const N_ITERATIONS: usize = 10;

    // Generate test data
    let n_blocks = N_ELEMENTS.div_ceil(256);
    let block_bytes = bitnet_ggml_ffi::iq2s_bytes_per_block();
    let mut test_data = vec![0u8; n_blocks * block_bytes];
    let mut rng = rand::rngs::StdRng::seed_from_u64(789);

    for block in test_data.chunks_mut(block_bytes) {
        let scale = f16::from_f32(rng.gen_range(0.5..1.5));
        block[0..2].copy_from_slice(&scale.to_bits().to_le_bytes());
        for i in 2..66 {
            block[i] = rng.r#gen();
        }
        // Leave the rest as zeros
    }

    // Benchmark Rust backend
    let start = Instant::now();
    for _ in 0..N_ITERATIONS {
        let _result = Iq2sBackend::Rust.dequantize(&test_data, &[N_ELEMENTS])?;
    }
    let rust_duration = start.elapsed();

    // Benchmark FFI backend
    let start = Instant::now();
    for _ in 0..N_ITERATIONS {
        let _result = Iq2sBackend::Ffi.dequantize(&test_data, &[N_ELEMENTS])?;
    }
    let ffi_duration = start.elapsed();

    let rust_throughput = (N_ELEMENTS * N_ITERATIONS) as f64 / rust_duration.as_secs_f64();
    let ffi_throughput = (N_ELEMENTS * N_ITERATIONS) as f64 / ffi_duration.as_secs_f64();

    println!("IQ2_S Performance Comparison:");
    println!("  Rust backend: {:.2} elements/sec", rust_throughput);
    println!("  FFI backend:  {:.2} elements/sec", ffi_throughput);
    println!("  Ratio: {:.2}x", rust_throughput / ffi_throughput);

    // Both should achieve reasonable throughput
    assert!(rust_throughput > 1e6, "Rust backend too slow: {} elements/sec", rust_throughput);
    assert!(ffi_throughput > 1e6, "FFI backend too slow: {} elements/sec", ffi_throughput);

    Ok(())
}

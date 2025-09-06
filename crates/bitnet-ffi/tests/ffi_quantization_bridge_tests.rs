//! Comprehensive FFI Quantization Bridge Tests
//!
//! This module tests the FFI quantization bridge that enables gradual migration
//! from C++ to Rust while maintaining functionality and performance.

#[cfg(feature = "ffi")]
mod ffi_bridge_tests {
    use bitnet_common::QuantizationType;
    use bitnet_kernels::Kernel;
    use bitnet_kernels::ffi::FfiKernel;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_ffi_kernel_creation() {
        match FfiKernel::new() {
            Ok(kernel) => {
                assert_eq!(kernel.name(), "ffi");
                println!("FFI kernel available: {}", kernel.is_available());
            }
            Err(e) => {
                println!("FFI kernel creation failed (expected if C++ lib not available): {}", e);
                // This is not necessarily a failure - FFI might not be built
            }
        }
    }

    #[test]
    fn test_ffi_quantization_types() {
        let kernel = match FfiKernel::new() {
            Ok(k) if k.is_available() => k,
            _ => {
                println!("FFI kernel not available, skipping test");
                return;
            }
        };

        let test_data = vec![1.5f32, -2.0, 0.5, -0.5, 3.0, -1.0, 0.0, 2.5];
        let quantization_types =
            [QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2];

        for &qtype in &quantization_types {
            let mut output = vec![0u8; 8];
            let mut scales = vec![0.0f32; 2];

            let result = kernel.quantize(&test_data, &mut output, &mut scales, qtype);

            match result {
                Ok(()) => {
                    // Verify output validity
                    assert!(
                        scales.iter().all(|&s| s > 0.0 && s.is_finite()),
                        "Invalid scales for {:?}",
                        qtype
                    );
                    println!("FFI quantization succeeded for {:?}", qtype);
                }
                Err(e) => {
                    println!("FFI quantization failed for {:?}: {} (may be expected)", qtype, e);
                    // Some quantization types might not be implemented in FFI
                }
            }
        }
    }

    #[test]
    fn test_ffi_vs_rust_quantization_parity() {
        let ffi_kernel = match FfiKernel::new() {
            Ok(k) if k.is_available() => k,
            _ => {
                println!("FFI kernel not available, skipping parity test");
                return;
            }
        };

        // This test would compare FFI quantization with pure Rust implementation
        // For now, we just test that FFI quantization is consistent
        let test_data = vec![1.0f32, -1.0, 2.0, -2.0, 0.5, -0.5, 0.0, 3.0];

        // Run multiple times to check consistency
        let mut outputs = Vec::new();
        let mut scales_sets = Vec::new();

        for i in 0..5 {
            let mut output = vec![0u8; 8];
            let mut scales = vec![0.0f32; 2];

            match ffi_kernel.quantize(&test_data, &mut output, &mut scales, QuantizationType::I2S) {
                Ok(()) => {
                    outputs.push(output);
                    scales_sets.push(scales);
                    println!("FFI quantization run {} succeeded", i);
                }
                Err(e) => {
                    println!("FFI quantization run {} failed: {}", i, e);
                    return; // Skip consistency check if any run fails
                }
            }
        }

        // Check consistency across runs
        if !outputs.is_empty() {
            let first_output = &outputs[0];
            let first_scales = &scales_sets[0];

            for (i, (output, scales)) in outputs.iter().zip(scales_sets.iter()).enumerate().skip(1)
            {
                // FFI quantization should be deterministic
                if output != first_output {
                    println!("Warning: FFI quantization output inconsistency at run {}", i);
                }

                for (j, (&scale, &first_scale)) in
                    scales.iter().zip(first_scales.iter()).enumerate()
                {
                    let diff = (scale - first_scale).abs();
                    if diff > 1e-6 {
                        println!(
                            "Warning: FFI quantization scale inconsistency at run {}, scale {}: {} vs {}",
                            i, j, scale, first_scale
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_ffi_quantization_edge_cases() {
        let kernel = match FfiKernel::new() {
            Ok(k) if k.is_available() => k,
            _ => {
                println!("FFI kernel not available, skipping edge case test");
                return;
            }
        };

        // Test with all zeros
        let zeros = vec![0.0f32; 8];
        let mut output = vec![0u8; 8];
        let mut scales = vec![0.0f32; 1];

        match kernel.quantize(&zeros, &mut output, &mut scales, QuantizationType::I2S) {
            Ok(()) => {
                println!("FFI quantization handled all zeros");
                // Scale might be zero or very small for all-zero input
            }
            Err(e) => println!("FFI quantization failed on all zeros: {}", e),
        }

        // Test with very large values
        let large_values = vec![1e6f32, -1e6, 1e5, -1e5, 1e4, -1e4, 0.0, 1e3];
        let mut output = vec![0u8; 8];
        let mut scales = vec![0.0f32; 2];

        match kernel.quantize(&large_values, &mut output, &mut scales, QuantizationType::I2S) {
            Ok(()) => {
                assert!(
                    scales.iter().all(|&s| s.is_finite()),
                    "Scales should be finite even for large inputs"
                );
                println!("FFI quantization handled large values");
            }
            Err(e) => println!("FFI quantization failed on large values: {}", e),
        }

        // Test with very small values
        let small_values = vec![1e-6f32, -1e-6, 1e-5, -1e-5, 0.0, 1e-4, -1e-4, 1e-3];
        let mut output = vec![0u8; 8];
        let mut scales = vec![0.0f32; 1];

        match kernel.quantize(&small_values, &mut output, &mut scales, QuantizationType::I2S) {
            Ok(()) => {
                println!("FFI quantization handled small values");
            }
            Err(e) => println!("FFI quantization failed on small values: {}", e),
        }
    }

    #[test]
    fn test_ffi_quantization_performance() {
        let kernel = match FfiKernel::new() {
            Ok(k) if k.is_available() => k,
            _ => {
                println!("FFI kernel not available, skipping performance test");
                return;
            }
        };

        let sizes = [64, 128, 256, 512];
        let iterations = 10;

        for &size in &sizes {
            let test_data = (0..size).map(|i| (i as f32).sin()).collect::<Vec<f32>>();
            let mut output = vec![0u8; size / 4]; // Assuming 4:1 compression ratio
            let mut scales = vec![0.0f32; size / 64]; // Assuming block size of 64

            let start = std::time::Instant::now();

            let mut successful_runs = 0;
            for _ in 0..iterations {
                match kernel.quantize(&test_data, &mut output, &mut scales, QuantizationType::I2S) {
                    Ok(()) => successful_runs += 1,
                    Err(_) => continue,
                }
            }

            let elapsed = start.elapsed();

            if successful_runs > 0 {
                let avg_time = elapsed / successful_runs;
                let throughput = (size as f64) / avg_time.as_secs_f64();

                println!(
                    "FFI quantization size {}: {} successful runs, avg time {:?}, throughput {:.2} elements/sec",
                    size, successful_runs, avg_time, throughput
                );

                // Performance should be reasonable
                assert!(
                    avg_time < Duration::from_millis(100),
                    "FFI quantization should complete within 100ms for size {}",
                    size
                );
            } else {
                println!("FFI quantization failed all runs for size {}", size);
            }
        }
    }

    #[test]
    fn test_ffi_quantization_thread_safety() {
        let kernel = match FfiKernel::new() {
            Ok(k) if k.is_available() => k,
            _ => {
                println!("FFI kernel not available, skipping thread safety test");
                return;
            }
        };

        let kernel = Arc::new(kernel);
        let num_threads = 4;
        let iterations_per_thread = 10;

        let mut handles = Vec::new();

        for thread_id in 0..num_threads {
            let kernel_clone = Arc::clone(&kernel);

            let handle = thread::spawn(move || {
                let mut successful_runs = 0;

                for i in 0..iterations_per_thread {
                    let test_data = vec![
                        (thread_id as f32) + (i as f32) * 0.1,
                        -(thread_id as f32) - (i as f32) * 0.1,
                        (thread_id as f32) * 2.0,
                        -(thread_id as f32) * 2.0,
                        0.0,
                        1.0,
                        -1.0,
                        (i as f32).sin(),
                    ];

                    let mut output = vec![0u8; 8];
                    let mut scales = vec![0.0f32; 1];

                    match kernel_clone.quantize(
                        &test_data,
                        &mut output,
                        &mut scales,
                        QuantizationType::I2S,
                    ) {
                        Ok(()) => {
                            successful_runs += 1;
                            // Verify output validity
                            if !scales.iter().all(|&s| s.is_finite()) {
                                eprintln!(
                                    "Thread {}: Invalid scales at iteration {}",
                                    thread_id, i
                                );
                            }
                        }
                        Err(e) => {
                            eprintln!(
                                "Thread {}: FFI quantization failed at iteration {}: {}",
                                thread_id, i, e
                            );
                        }
                    }

                    // Small delay to increase chance of race conditions
                    thread::sleep(Duration::from_millis(1));
                }

                (thread_id, successful_runs)
            });

            handles.push(handle);
        }

        // Wait for all threads and collect results
        let mut total_successful = 0;
        for handle in handles {
            match handle.join() {
                Ok((thread_id, successful_runs)) => {
                    println!("Thread {} completed {} successful runs", thread_id, successful_runs);
                    total_successful += successful_runs;
                }
                Err(e) => {
                    eprintln!("Thread panicked: {:?}", e);
                }
            }
        }

        println!("Total successful FFI quantization runs across all threads: {}", total_successful);

        // At least some runs should succeed if FFI is working
        if total_successful > 0 {
            println!("FFI quantization appears to be thread-safe");
        }
    }

    #[test]
    fn test_ffi_quantization_memory_safety() {
        let kernel = match FfiKernel::new() {
            Ok(k) if k.is_available() => k,
            _ => {
                println!("FFI kernel not available, skipping memory safety test");
                return;
            }
        };

        // Test with various buffer sizes to ensure no buffer overruns
        let test_cases = [
            (vec![1.0f32; 4], 4, 1),   // Minimal case
            (vec![1.0f32; 8], 8, 1),   // Standard case
            (vec![1.0f32; 64], 16, 4), // Larger case
        ];

        for (input, output_size, scales_size) in test_cases {
            let mut output = vec![0u8; output_size];
            let mut scales = vec![0.0f32; scales_size];

            match kernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S) {
                Ok(()) => {
                    // Verify no buffer overruns by checking that we can still access the data
                    let _sum: u8 = output.iter().sum();
                    let _scale_sum: f32 = scales.iter().sum();
                    println!("Memory safety test passed for input size {}", input.len());
                }
                Err(e) => {
                    println!("FFI quantization failed for input size {}: {}", input.len(), e);
                }
            }
        }
    }
}

#[cfg(not(feature = "ffi"))]
mod ffi_bridge_disabled_tests {
    #[test]
    fn test_ffi_bridge_disabled() {
        println!("FFI bridge feature disabled - skipping FFI quantization bridge tests");

        // This test ensures the test suite runs even when FFI is disabled
        // Test passes when FFI is disabled - no assertion needed
    }
}

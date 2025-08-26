//! GPU smoke test with CPU parity checking
//!
//! This test verifies GPU functionality and compares results with CPU
//! to ensure correctness. The test is designed to be fast and reliable,
//! suitable for CI/CD pipelines.

#![cfg(feature = "cuda")]

use bitnet_kernels::{KernelProvider, CudaKernel, FallbackKernel};
use std::env;

/// Tolerance for floating-point comparison
const DEFAULT_TOLERANCE: f32 = 0.99;

/// Get test configuration from environment
fn get_test_config() -> (String, f32) {
    let size = env::var("GPU_TEST_SIZE").unwrap_or_else(|_| "tiny".to_string());
    let tolerance = env::var("GPU_TEST_TOLERANCE")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(DEFAULT_TOLERANCE);
    (size, tolerance)
}

/// Check if GPU is available (skip test if not)
fn check_gpu_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        use bitnet_kernels::gpu_utils;
        gpu_utils::gpu_available()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Generate test data based on size
fn generate_test_data(size: &str) -> (Vec<f32>, Vec<u8>, Vec<f32>, usize, usize, usize) {
    let (m, n, k) = match size {
        "tiny" => (16, 16, 16),
        "small" => (64, 64, 64),
        "medium" => (256, 256, 256),
        _ => (16, 16, 16), // Default to tiny
    };

    // Generate deterministic test data
    let mut a = vec![0i8; m * k];
    let mut b = vec![0u8; k * n];
    let c = vec![0.0f32; m * n];

    // Fill with simple patterns for reproducibility
    for i in 0..a.len() {
        a[i] = ((i % 3) as i8) - 1; // -1, 0, 1 pattern
    }

    for i in 0..b.len() {
        b[i] = (i % 256) as u8;
    }

    // Convert i8 to f32 for CPU computation
    let a_f32: Vec<f32> = a.iter().map(|&x| x as f32).collect();

    (a_f32, b, c, m, n, k)
}

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        if norm_a == norm_b {
            1.0 // Both zero vectors
        } else {
            0.0 // One is zero
        }
    } else {
        dot_product / (norm_a * norm_b)
    }
}

/// Mean squared error between two vectors
fn mse(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");

    let sum_sq_diff: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum();

    sum_sq_diff / (a.len() as f32)
}

#[test]
fn gpu_smoke_test() {
    // Skip if GPU not available
    if !check_gpu_available() {
        eprintln!("Skipping GPU smoke test: No GPU available");
        return;
    }

    let (size, tolerance) = get_test_config();
    eprintln!("Running GPU smoke test with size={}, tolerance={}", size, tolerance);

    // Generate test data
    let (a, b, mut c_cpu, m, n, k) = generate_test_data(&size);
    let mut c_gpu = c_cpu.clone();

    // Convert for GPU (assuming i8 for BitNet)
    let a_i8: Vec<i8> = a.iter().map(|&x| x as i8).collect();

    // Run CPU version using fallback kernel
    let cpu_provider = FallbackKernel;
    assert!(cpu_provider.is_available(), "CPU provider should always be available");

    eprintln!("Running CPU computation...");
    let cpu_start = std::time::Instant::now();
    cpu_provider.matmul_i2s(&a_i8, &b, &mut c_cpu, m, n, k).expect("CPU matmul failed");
    let cpu_time = cpu_start.elapsed();
    eprintln!("CPU time: {:?}", cpu_time);

    // Run GPU version
    #[cfg(feature = "cuda")]
    {
        let gpu_provider = CudaKernel::new().expect("Failed to create GPU provider");
        assert!(gpu_provider.is_available(), "GPU provider should be available");

        eprintln!("Running GPU computation...");
        let gpu_start = std::time::Instant::now();
        gpu_provider.matmul_i2s(&a_i8, &b, &mut c_gpu, m, n, k).expect("GPU matmul failed");
        let gpu_time = gpu_start.elapsed();
        eprintln!("GPU time: {:?}", gpu_time);

        // Calculate speedup
        let speedup = cpu_time.as_secs_f32() / gpu_time.as_secs_f32();
        eprintln!("Speedup: {:.2}x", speedup);
    }

    // Compare results
    let similarity = cosine_similarity(&c_cpu, &c_gpu);
    let error = mse(&c_cpu, &c_gpu);

    eprintln!("Cosine similarity: {:.6}", similarity);
    eprintln!("MSE: {:.6}", error);

    // Check tolerance
    assert!(
        similarity >= tolerance,
        "GPU-CPU similarity {} below tolerance {}",
        similarity,
        tolerance
    );

    // Also check MSE is reasonable
    let max_mse = (1.0 - tolerance) * 100.0; // Rough heuristic
    assert!(error <= max_mse, "GPU-CPU MSE {} above threshold {}", error, max_mse);

    eprintln!("✅ GPU smoke test passed!");
}

#[test]
fn gpu_determinism_test() {
    if !check_gpu_available() {
        eprintln!("Skipping GPU determinism test: No GPU available");
        return;
    }

    // Run the same computation multiple times and verify identical results
    let (a, b, _, m, n, k) = generate_test_data("tiny");
    let a_i8: Vec<i8> = a.iter().map(|&x| x as i8).collect();

    #[cfg(feature = "cuda")]
    {
        let gpu_provider = CudaKernel::new().expect("Failed to create GPU provider");

        let mut results = Vec::new();
        for i in 0..3 {
            let mut c = vec![0.0f32; m * n];
            gpu_provider.matmul_i2s(&a_i8, &b, &mut c, m, n, k).expect("GPU matmul failed");
            results.push(c);
            eprintln!("Run {}: first element = {}", i + 1, results[i][0]);
        }

        // All runs should produce identical results
        for i in 1..results.len() {
            assert_eq!(results[0], results[i], "GPU results not deterministic: run 0 != run {}", i);
        }

        eprintln!("✅ GPU determinism test passed!");
    }
}

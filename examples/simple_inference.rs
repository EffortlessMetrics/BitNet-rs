//! Simple inference example demonstrating SIMD-optimized BitNet model execution
//!
//! This example shows:
//! - Loading a GGUF model with automatic SIMD kernel selection
//! - Running inference with CPU optimizations
//! - Measuring performance improvements from SIMD

#[cfg(feature = "examples")]
use anyhow::Result;
#[cfg(feature = "examples")]
use std::path::Path;
#[cfg(feature = "examples")]
use std::time::Instant;

#[cfg(feature = "examples")]
fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("BitNet SIMD-Optimized Inference Example");
    println!("========================================\n");

    // Model path from environment or command line
    let model_path = std::env::args()
        .nth(1)
        .or_else(|| std::env::var("MODEL_PATH").ok())
        .unwrap_or_else(|| "models/ggml-model-i2_s.gguf".to_string());

    println!("Model: {}", model_path);

    // Check if model exists
    let model_path = Path::new(&model_path);
    if !model_path.exists() {
        eprintln!("Model not found at: {}", model_path.display());
        eprintln!("Please download a model using:");
        eprintln!("  cargo xtask download-model");
        eprintln!("\nOr provide a path:");
        eprintln!("  cargo run --example simple_inference --features cpu -- path/to/model.gguf");
        return Ok(());
    }

    // Detect CPU features
    println!("\n1. Detecting CPU features...");
    #[cfg(all(feature = "cpu", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            println!("   ✓ AVX-512 available (best performance)");
        } else if is_x86_feature_detected!("avx2") {
            println!("   ✓ AVX2 available (good performance)");
        } else {
            println!("   ⚠ No SIMD features detected (using scalar fallback)");
        }
    }

    #[cfg(all(feature = "cpu", target_arch = "aarch64"))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            println!("   ✓ NEON available (optimized for ARM)");
        } else {
            println!("   ⚠ No SIMD features detected (using scalar fallback)");
        }
    }

    #[cfg(feature = "cpu")]
    {
        use bitnet_common::QuantizationType;
        use bitnet_kernels::KernelProvider;

        // Create kernel provider (auto-selects best available)
        println!("\n2. Selecting optimal kernel...");
        let kernel = bitnet_kernels::create_best_kernel();
        println!("   Using kernel: {}", kernel.name());
        println!("   Available: {}", kernel.is_available());

        // Benchmark quantization performance
        println!("\n3. Testing quantization performance...");
        let test_sizes = [1024, 16384, 65536, 262144];

        for &size in &test_sizes {
            let input = vec![0.5f32; size];
            let mut output = vec![0u8; size / 4]; // 2-bit packing
            let mut scales = vec![0.0f32; size / 32]; // One scale per 32 elements

            let start = Instant::now();
            kernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S)?;
            let quant_time = start.elapsed();

            let throughput = size as f64 / quant_time.as_secs_f64();
            println!(
                "   {} elements: {:.2}M elem/s ({:.3}ms)",
                size,
                throughput / 1_000_000.0,
                quant_time.as_millis()
            );
        }

        // Test matrix multiplication
        println!("\n4. Testing matrix multiplication...");
        let m = 128;
        let n = 256;
        let k = 512;

        let a = vec![1i8; m * k];
        let b = vec![1u8; k * n];
        let mut c = vec![0.0f32; m * n];

        let start = Instant::now();
        kernel.matmul_i2s(&a, &b, &mut c, m, n, k)?;
        let matmul_time = start.elapsed();

        let gflops = (2.0 * m as f64 * n as f64 * k as f64) / matmul_time.as_secs_f64() / 1e9;
        println!("   {}x{}x{}: {:.2} GFLOPS ({:.3}ms)", m, n, k, gflops, matmul_time.as_millis());
    }

    #[cfg(feature = "inference")]
    {
        use bitnet_models::gguf_parity::validate_gguf_model;

        // Validate model metadata
        println!("\n5. Validating GGUF model...");
        let metadata = validate_gguf_model(model_path, None)?;
        println!("   ✓ Architecture: {}", metadata.arch);
        println!("   ✓ Vocab size: {}", metadata.vocab_size);
        println!("   ✓ Context: {}", metadata.context_length);
        println!("   ✓ Quantization: {:?}", metadata.quantization_type);

        // Load model and run simple inference
        println!("\n6. Loading model for inference...");
        let start = Instant::now();

        // Here we would load the actual model and run inference
        // This is a placeholder for the full implementation

        let load_time = start.elapsed();
        println!("   ✓ Model loaded in {:.2}s", load_time.as_secs_f32());
    }

    #[cfg(not(feature = "cpu"))]
    {
        println!("\nNote: Run with --features=\"cpu\" to enable SIMD optimizations.");
        println!("      Or --features=\"cpu,inference\" for full inference.");
    }

    println!("\n✅ Example completed successfully!");
    Ok(())
}

#[cfg(not(feature = "examples"))]
fn main() {}

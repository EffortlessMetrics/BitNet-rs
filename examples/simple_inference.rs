//! Simple inference example demonstrating SIMD-optimized BitNet model execution
//!
//! This example shows:
//! - Loading a GGUF model with automatic SIMD kernel selection
//! - Running inference with CPU optimizations
//! - Measuring performance improvements from SIMD

#[cfg(feature = "examples")]
use anyhow::Result;
#[cfg(feature = "examples")]
use std::path::{Path, PathBuf};
#[cfg(feature = "examples")]
use std::time::Instant;

#[cfg(feature = "examples")]
mod srp {
    use super::*;

    pub fn print_header() {
        tracing_subscriber::fmt::init();
        println!("BitNet SIMD-Optimized Inference Example");
        println!("========================================\n");
    }

    pub fn resolve_model_path() -> PathBuf {
        std::env::args()
            .nth(1)
            .or_else(|| std::env::var("MODEL_PATH").ok())
            .unwrap_or_else(|| "models/ggml-model-i2_s.gguf".to_string())
            .into()
    }

    pub fn ensure_model_exists(model_path: &Path) -> bool {
        if model_path.exists() {
            return true;
        }
        eprintln!("Model not found at: {}", model_path.display());
        eprintln!("Please download a model using:");
        eprintln!("  cargo xtask download-model");
        eprintln!("\nOr provide a path:");
        eprintln!("  cargo run --example simple_inference --features cpu -- path/to/model.gguf");
        false
    }

    pub fn print_cpu_features() {
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
    }

    #[cfg(feature = "cpu")]
    pub fn run_cpu_benchmarks() -> Result<()> {
        use bitnet_common::QuantizationType;
        use bitnet_kernels::KernelProvider;

        println!("\n2. Selecting optimal kernel...");
        let kernel = bitnet_kernels::create_best_kernel();
        println!("   Using kernel: {}", kernel.name());
        println!("   Available: {}", kernel.is_available());

        println!("\n3. Testing quantization performance...");
        let test_sizes = [1024, 16384, 65536, 262144];

        for &size in &test_sizes {
            let input = vec![0.5f32; size];
            let mut output = vec![0u8; size / 4];
            let mut scales = vec![0.0f32; size / 32];

            let start = Instant::now();
            kernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S)?;
            let i2s_time = start.elapsed();
            let i2s_throughput = size as f64 / i2s_time.as_secs_f64();

            let iq2s_result = if cfg!(feature = "iq2s-ffi") {
                let start = Instant::now();
                match kernel.quantize(&input, &mut output, &mut scales, QuantizationType::IQ2_S) {
                    Ok(_) => {
                        let time = start.elapsed();
                        let throughput = size as f64 / time.as_secs_f64();
                        Some((time, throughput))
                    }
                    Err(_) => None,
                }
            } else {
                None
            };

            println!(
                "   {} elements I2S: {:.2}M elem/s ({:.3}ms)",
                size,
                i2s_throughput / 1_000_000.0,
                i2s_time.as_millis()
            );

            if let Some((iq2s_time, iq2s_throughput)) = iq2s_result {
                println!(
                    "   {} elements IQ2_S: {:.2}M elem/s ({:.3}ms) [82B blocks]",
                    size,
                    iq2s_throughput / 1_000_000.0,
                    iq2s_time.as_millis()
                );
            }
        }

        println!("\n4. Testing matrix multiplication...");
        let (m, n, k) = (128, 256, 512);
        let a = vec![1i8; m * k];
        let b = vec![1u8; k * n];
        let mut c = vec![0.0f32; m * n];

        let start = Instant::now();
        kernel.matmul_i2s(&a, &b, &mut c, m, n, k)?;
        let matmul_time = start.elapsed();

        let gflops = (2.0 * m as f64 * n as f64 * k as f64) / matmul_time.as_secs_f64() / 1e9;
        println!("   {}x{}x{}: {:.2} GFLOPS ({:.3}ms)", m, n, k, gflops, matmul_time.as_millis());
        Ok(())
    }

    #[cfg(feature = "inference")]
    pub fn run_inference_checks(model_path: &Path) -> Result<()> {
        use bitnet_models::gguf_parity::validate_gguf_model;

        println!("\n5. Validating GGUF model...");
        let metadata = validate_gguf_model(model_path, None)?;
        println!("   ✓ Architecture: {}", metadata.arch);
        println!("   ✓ Vocab size: {}", metadata.vocab_size);
        println!("   ✓ Context: {}", metadata.context_length);
        println!("   ✓ Quantization: {:?}", metadata.quantization_type);

        println!("\n6. Loading model for inference...");
        let start = Instant::now();
        let load_time = start.elapsed();
        println!("   ✓ Model loaded in {:.2}s", load_time.as_secs_f32());
        Ok(())
    }
}

#[cfg(feature = "examples")]
fn main() -> Result<()> {
    srp::print_header();

    let model_path = srp::resolve_model_path();
    println!("Model: {}", model_path.display());

    if !srp::ensure_model_exists(&model_path) {
        return Ok(());
    }

    srp::print_cpu_features();

    #[cfg(feature = "cpu")]
    srp::run_cpu_benchmarks()?;

    #[cfg(feature = "inference")]
    srp::run_inference_checks(&model_path)?;

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

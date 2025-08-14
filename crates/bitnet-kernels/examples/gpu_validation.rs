//! GPU validation and benchmarking CLI utility
//!
//! This example provides a command-line interface for running comprehensive
//! GPU kernel validation and performance benchmarking.
//!
//! Usage:
//!   cargo run --example gpu_validation --features cuda
//!   cargo run --example gpu_validation --features cuda -- --benchmark-only
//!   cargo run --example gpu_validation --features cuda -- --validation-only

#[cfg(feature = "cuda")]
use bitnet_kernels::gpu::{
    cuda_device_count, is_cuda_available, print_benchmark_results, print_validation_results,
    BenchmarkConfig, CudaKernel, GpuBenchmark, GpuValidator, ValidationConfig,
};
use std::env;

fn main() {
    #[cfg(not(feature = "cuda"))]
    {
        println!("❌ This example requires the 'cuda' feature to be enabled");
        println!("Run with: cargo run --example gpu_validation --features cuda");
        std::process::exit(1);
    }

    #[cfg(feature = "cuda")]
    run_gpu_validation();
}

#[cfg(feature = "cuda")]
fn run_gpu_validation() {
    // Initialize logging
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    println!("🚀 BitNet GPU Kernel Validation and Benchmarking Tool");
    println!("====================================================");

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let validation_only = args.contains(&"--validation-only".to_string());
    let benchmark_only = args.contains(&"--benchmark-only".to_string());
    let quick_mode = args.contains(&"--quick".to_string());

    // Check CUDA availability
    println!("\n🔍 Checking CUDA availability...");
    if !is_cuda_available() {
        println!("❌ CUDA is not available on this system");
        println!("   Please ensure:");
        println!("   - NVIDIA GPU is installed");
        println!("   - CUDA drivers are installed");
        println!("   - CUDA toolkit is installed");
        std::process::exit(1);
    }

    let device_count = cuda_device_count();
    println!("✅ CUDA is available with {} device(s)", device_count);

    // Test basic kernel creation
    println!("\n🧪 Testing basic kernel creation...");
    match CudaKernel::new() {
        Ok(kernel) => {
            println!("✅ CUDA kernel created successfully");
            println!("   Device info: {:?}", kernel.device_info());
        }
        Err(e) => {
            println!("❌ Failed to create CUDA kernel: {}", e);
            std::process::exit(1);
        }
    }

    // Run validation if requested
    if !benchmark_only {
        println!("\n📊 Running GPU kernel validation...");
        run_validation(quick_mode);
    }

    // Run benchmarks if requested
    if !validation_only {
        println!("\n🏃 Running GPU kernel benchmarks...");
        run_benchmarks(quick_mode);
    }

    println!("\n🎉 GPU validation and benchmarking completed!");
    println!("\nFor more detailed testing, run:");
    println!("  cargo test --features cuda --ignored gpu_integration");
}

fn run_validation(quick_mode: bool) {
    let config = if quick_mode {
        ValidationConfig {
            test_sizes: vec![(128, 128, 128), (256, 256, 256)],
            benchmark_iterations: 20,
            tolerance: 1e-6,
            check_memory_leaks: true,
            test_mixed_precision: false,
        }
    } else {
        ValidationConfig {
            test_sizes: vec![
                (64, 64, 64),
                (128, 128, 128),
                (256, 256, 256),
                (512, 512, 512),
                (1024, 1024, 1024),
            ],
            benchmark_iterations: 100,
            tolerance: 1e-6,
            check_memory_leaks: true,
            test_mixed_precision: false,
        }
    };

    let validator = GpuValidator::with_config(config);

    match validator.validate() {
        Ok(results) => {
            print_validation_results(&results);

            if !results.success {
                println!("\n❌ GPU validation failed!");
                std::process::exit(1);
            } else {
                println!("\n✅ GPU validation passed!");
            }
        }
        Err(e) => {
            println!("❌ GPU validation error: {}", e);
            std::process::exit(1);
        }
    }
}

fn run_benchmarks(quick_mode: bool) {
    let config = if quick_mode {
        BenchmarkConfig {
            test_sizes: vec![(256, 256, 256), (512, 512, 512)],
            warmup_iterations: 5,
            benchmark_iterations: 20,
            include_cpu_comparison: true,
            test_data_patterns: false,
        }
    } else {
        BenchmarkConfig {
            test_sizes: vec![
                (128, 128, 128),
                (256, 256, 256),
                (512, 512, 512),
                (1024, 1024, 1024),
                (2048, 1024, 512),
                (1024, 2048, 512),
            ],
            warmup_iterations: 10,
            benchmark_iterations: 100,
            include_cpu_comparison: true,
            test_data_patterns: false,
        }
    };

    let benchmark = GpuBenchmark::with_config(config);

    match benchmark.run() {
        Ok(results) => {
            print_benchmark_results(&results);

            // Provide performance analysis
            println!("\n📈 Performance Analysis:");
            if results.summary.avg_speedup > 10.0 {
                println!(
                    "🚀 Excellent GPU acceleration! Average speedup: {:.1}x",
                    results.summary.avg_speedup
                );
            } else if results.summary.avg_speedup > 5.0 {
                println!(
                    "✅ Very good GPU acceleration! Average speedup: {:.1}x",
                    results.summary.avg_speedup
                );
            } else if results.summary.avg_speedup > 2.0 {
                println!(
                    "✅ Good GPU acceleration! Average speedup: {:.1}x",
                    results.summary.avg_speedup
                );
            } else if results.summary.avg_speedup > 1.0 {
                println!(
                    "⚠️  Modest GPU acceleration. Average speedup: {:.1}x",
                    results.summary.avg_speedup
                );
                println!("   Consider optimizing GPU kernels or checking GPU utilization");
            } else {
                println!(
                    "❌ GPU is slower than CPU! Average speedup: {:.1}x",
                    results.summary.avg_speedup
                );
                println!("   This indicates a problem with the GPU implementation");
            }

            if results.summary.peak_gflops > 500.0 {
                println!(
                    "🚀 Excellent computational throughput: {:.0} GFLOPS",
                    results.summary.peak_gflops
                );
            } else if results.summary.peak_gflops > 100.0 {
                println!(
                    "✅ Good computational throughput: {:.0} GFLOPS",
                    results.summary.peak_gflops
                );
            } else if results.summary.peak_gflops > 50.0 {
                println!(
                    "⚠️  Moderate computational throughput: {:.0} GFLOPS",
                    results.summary.peak_gflops
                );
            } else {
                println!(
                    "❌ Low computational throughput: {:.0} GFLOPS",
                    results.summary.peak_gflops
                );
            }
        }
        Err(e) => {
            println!("❌ GPU benchmark error: {}", e);
            std::process::exit(1);
        }
    }
}

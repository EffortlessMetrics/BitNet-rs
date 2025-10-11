//! Simple GPU validation test
//!
//! This example provides a basic test of GPU kernel functionality
//! without the complex test infrastructure.

#[cfg(any(feature = "gpu", feature = "cuda"))]
use bitnet_kernels::KernelProvider;
#[cfg(any(feature = "gpu", feature = "cuda"))]
use bitnet_kernels::gpu::{CudaKernel, cuda_device_count, is_cuda_available};

fn main() {
    #[cfg(not(any(feature = "gpu", feature = "cuda")))]
    {
        println!("❌ This example requires the 'cuda' feature to be enabled");
        println!("Run with: cargo run --example simple_gpu_test --features cuda");
    }

    #[cfg(any(feature = "gpu", feature = "cuda"))]
    run_simple_gpu_test();
}

#[cfg(any(feature = "gpu", feature = "cuda"))]
fn run_simple_gpu_test() {
    println!("🚀 Simple GPU Kernel Test");
    println!("========================");

    // Check CUDA availability
    println!("\n🔍 Checking CUDA availability...");
    if !is_cuda_available() {
        println!("❌ CUDA is not available on this system");
        return;
    }

    let device_count = cuda_device_count();
    println!("✅ CUDA is available with {} device(s)", device_count);

    // Test basic kernel creation
    println!("\n🧪 Testing kernel creation...");
    let kernel = match CudaKernel::new() {
        Ok(k) => {
            println!("✅ CUDA kernel created successfully");
            println!("   Device info: {:?}", k.device_info());
            k
        }
        Err(e) => {
            println!("❌ Failed to create CUDA kernel: {}", e);
            return;
        }
    };

    // Test basic matrix multiplication
    println!("\n🔢 Testing matrix multiplication...");
    let test_sizes = [(4, 4, 4), (8, 8, 8), (16, 16, 16)];

    for &(m, n, k) in &test_sizes {
        println!("   Testing {}x{}x{} matrix...", m, n, k);

        // Create test data
        let a: Vec<i8> = (0..m * k).map(|i| ((i % 3) as i8) - 1).collect(); // -1, 0, 1
        let b: Vec<u8> = (0..k * n).map(|i| (i % 2) as u8).collect(); // 0, 1
        let mut c = vec![0.0f32; m * n];

        // Run kernel
        match kernel.matmul_i2s(&a, &b, &mut c, m, n, k) {
            Ok(_) => {
                let nonzero_count = c.iter().filter(|&&x| x != 0.0).count();
                println!("     ✅ Success! {} non-zero results", nonzero_count);

                if nonzero_count > 0 {
                    println!("     Sample results: {:?}", &c[0..4.min(c.len())]);
                } else {
                    println!("     ⚠️  All results are zero - may indicate an issue");
                }
            }
            Err(e) => {
                println!("     ❌ Failed: {}", e);
            }
        }
    }

    // Test performance
    println!("\n⏱️  Testing performance...");
    let (m, n, k) = (256, 256, 256);
    let a: Vec<i8> = (0..m * k).map(|i| ((i % 3) as i8) - 1).collect();
    let b: Vec<u8> = (0..k * n).map(|i| (i % 2) as u8).collect();
    let mut c = vec![0.0f32; m * n];

    let iterations = 10;
    let start = std::time::Instant::now();

    for _ in 0..iterations {
        if let Err(e) = kernel.matmul_i2s(&a, &b, &mut c, m, n, k) {
            println!("   ❌ Performance test failed: {}", e);
            return;
        }
    }

    let elapsed = start.elapsed();
    let avg_time = elapsed.as_secs_f64() / iterations as f64;
    let operations = 2.0 * m as f64 * n as f64 * k as f64;
    let gflops = operations / (avg_time * 1e9);

    println!("   ✅ Average time: {:.2}ms", avg_time * 1000.0);
    println!("   ✅ Performance: {:.1} GFLOPS", gflops);

    println!("\n🎉 GPU kernel test completed successfully!");
}

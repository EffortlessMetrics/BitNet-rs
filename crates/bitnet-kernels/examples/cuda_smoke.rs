//! Simple CUDA smoke test to verify GPU functionality

#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    use bitnet_kernels::KernelProvider;
    use bitnet_kernels::gpu::cuda::CudaKernel;

    println!("CUDA Smoke Test");
    println!("================");

    // Check CUDA availability
    if !bitnet_kernels::gpu::is_cuda_available() {
        eprintln!("CUDA is not available on this system");
        return Ok(());
    }

    let device_count = bitnet_kernels::gpu::cuda_device_count();
    println!("Found {} CUDA device(s)", device_count);

    // Create kernel on device 0
    let kernel = match CudaKernel::new() {
        Ok(k) => {
            println!("✓ Created CUDA kernel on device 0");
            k
        }
        Err(e) => {
            eprintln!("✗ Failed to create CUDA kernel: {}", e);
            return Err(e.into());
        }
    };

    // Get device info
    println!("\nDevice Information:");
    println!("  Name: {}", kernel.name());
    println!("  Available: {}", kernel.is_available());
    println!("  Device info: {:?}", kernel.device_info());

    // Test small matrix multiplication
    println!("\nTesting small matrix multiplication...");
    let m = 4usize;
    let n = 4usize;
    let k = 8usize;

    // Create test data
    let a: Vec<i8> = (0..m * k).map(|i| ((i % 3) as i8) - 1).collect();
    let s: Vec<u8> = (0..k * n).map(|i| (i % 2) as u8).collect();
    let mut c: Vec<f32> = vec![0.0; m * n];

    println!("  Matrix dimensions: A[{}x{}] × S[{}x{}] = C[{}x{}]", m, k, k, n, m, n);

    // Run matrix multiplication
    match kernel.matmul_i2s(&a, &s, &mut c, m, n, k) {
        Ok(()) => {
            println!("✓ Matrix multiplication completed successfully");

            // Display first few results
            println!("\nFirst 8 results:");
            for (i, val) in c.iter().take(8).enumerate() {
                println!("  C[{}] = {:.4}", i, val);
            }

            // Verify results are non-zero and finite
            let non_zero = c.iter().filter(|&&x| x != 0.0).count();
            let all_finite = c.iter().all(|&x| x.is_finite());

            println!("\nValidation:");
            println!("  Non-zero elements: {}/{}", non_zero, c.len());
            println!("  All finite: {}", all_finite);

            if non_zero > 0 && all_finite {
                println!("✓ Results validated successfully");
            } else {
                println!("✗ Results validation failed");
            }
        }
        Err(e) => {
            eprintln!("✗ Matrix multiplication failed: {}", e);
            return Err(e.into());
        }
    }

    // Test performance stats
    println!("\nPerformance Statistics:");
    let stats = kernel.performance_stats();
    println!("  Kernel launches: {}", stats.total_kernel_launches);
    println!("  Total execution time: {:.2} ms", stats.total_execution_time_ms);
    println!("  H2D transfers: {}", stats.memory_transfers_host_to_device);
    println!("  D2H transfers: {}", stats.memory_transfers_device_to_host);

    println!("\n✓ CUDA smoke test completed successfully!");
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This example requires the 'cuda' feature to be enabled.");
    eprintln!("Run with: cargo run --example cuda_smoke --features cuda");
    std::process::exit(1);
}

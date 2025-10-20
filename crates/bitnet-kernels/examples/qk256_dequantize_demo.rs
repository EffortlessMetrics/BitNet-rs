//! QK256 dequantization demonstration
//!
//! This example shows how to use the AVX2-optimized QK256 dequantization kernel.

use bitnet_kernels::{KernelProvider, cpu::Avx2Kernel};

fn main() -> anyhow::Result<()> {
    println!("QK256 Dequantization Demo\n");

    // Create AVX2 kernel instance
    let kernel = Avx2Kernel;

    // Check if AVX2 is available
    if !kernel.is_available() {
        println!("⚠️  AVX2 not available, using scalar fallback");
    } else {
        println!("✅ AVX2 available - using SIMD acceleration");
    }

    // Create test data: 2 blocks of QK256 (512 elements total)
    const QK256_BLOCK: usize = 256;
    const QK256_PACKED_BYTES: usize = 64; // 256 elements * 2 bits / 8 bits/byte
    let num_blocks = 2;

    // Pack quantized data: all codes = 2 (→ +1.0 with LUT)
    // Pattern: 0b_10_10_10_10 = 0xAA
    let quantized = vec![0xAAu8 as i8; num_blocks * QK256_PACKED_BYTES];

    // Scales for each block (multiply the LUT weights)
    let scales = vec![2.0f32, 3.0f32];

    println!("\nInput:");
    println!("  Quantized blocks: {}", num_blocks);
    println!("  Packed bytes: {} (total)", quantized.len());
    println!("  Scales: {:?}", scales);

    // Dequantize using AVX2 kernel
    let result = kernel.dequantize_qk256(&quantized, &scales, 256)?;

    println!("\nOutput:");
    println!("  Dequantized elements: {}", result.len());
    println!("  First 16 values: {:?}", &result[..16]);
    println!("  Last 16 values: {:?}", &result[result.len() - 16..]);

    // Verify correctness
    // Code 2 → LUT[2] = +1.0
    // Block 0: weight = 1.0 * scale[0] = 1.0 * 2.0 = 2.0
    // Block 1: weight = 1.0 * scale[1] = 1.0 * 3.0 = 3.0
    let expected_block0 = 2.0f32;
    let expected_block1 = 3.0f32;

    let block0_ok = result[..QK256_BLOCK].iter().all(|&v| (v - expected_block0).abs() < 1e-5);
    let block1_ok = result[QK256_BLOCK..].iter().all(|&v| (v - expected_block1).abs() < 1e-5);

    println!("\nValidation:");
    println!("  Block 0 (scale={:.1}): {} ✅", scales[0], if block0_ok { "PASS" } else { "FAIL" });
    println!("  Block 1 (scale={:.1}): {} ✅", scales[1], if block1_ok { "PASS" } else { "FAIL" });

    if block0_ok && block1_ok {
        println!("\n✅ QK256 dequantization successful!");
    } else {
        println!("\n❌ Validation failed!");
        return Err(anyhow::anyhow!("Validation failed"));
    }

    Ok(())
}

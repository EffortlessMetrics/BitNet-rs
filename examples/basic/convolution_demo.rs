//! Convolution Demo
//!
//! This example demonstrates the 2D convolution functionality in BitNet-rs,
//! showing both full-precision and quantized convolution operations.

use bitnet_common::{QuantizationType, Result};
use bitnet_kernels::convolution::{Conv2DParams, conv2d, conv2d_quantized};

fn main() -> Result<()> {
    println!("🔬 BitNet-rs Convolution Demo");
    println!("=============================\n");

    // Demo 1: Basic 2D Convolution
    basic_convolution_demo()?;

    // Demo 2: Convolution with stride and padding
    stride_padding_demo()?;

    // Demo 3: Quantized convolution with I2S
    quantized_convolution_demo()?;

    // Demo 4: Multiple quantization types comparison
    quantization_comparison_demo()?;

    println!("✅ All convolution demos completed successfully!");
    Ok(())
}

/// Demonstrates basic 2D convolution operation
fn basic_convolution_demo() -> Result<()> {
    println!("📋 Demo 1: Basic 2D Convolution");
    println!("--------------------------------");

    // Create a simple 3x3 input with 1 channel
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

    // Edge detection kernel (Sobel-like)
    let weight = vec![-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];

    // Output will be 1x1 (no padding, 3x3->3x3->1x1)
    let mut output = vec![0.0; 1];

    conv2d(
        &input,
        &weight,
        None, // No bias
        &mut output,
        Conv2DParams::default(),
        (1, 1, 3, 3), // Input: 1 batch, 1 channel, 3x3
        (1, 1, 3, 3), // Weight: 1 out_ch, 1 in_ch, 3x3
    )?;

    println!("Input (3x3):");
    print_tensor_2d(&input, 3, 3);

    println!("\nKernel (3x3 edge detector):");
    print_tensor_2d(&weight, 3, 3);

    println!("\nOutput (1x1): {:.2}", output[0]);
    println!("✓ Basic convolution completed\n");

    Ok(())
}

/// Demonstrates convolution with stride and padding
fn stride_padding_demo() -> Result<()> {
    println!("📋 Demo 2: Stride and Padding");
    println!("------------------------------");

    // 4x4 input
    let input =
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];

    // Simple 2x2 averaging kernel
    let weight = vec![0.25, 0.25, 0.25, 0.25];

    // With stride 2, output will be 2x2
    let mut output = vec![0.0; 4];

    let params = Conv2DParams {
        stride: (2, 2),  // Stride of 2 in both dimensions
        padding: (0, 0), // No padding
        dilation: (1, 1),
    };

    conv2d(
        &input,
        &weight,
        None,
        &mut output,
        params,
        (1, 1, 4, 4), // Input: 1 batch, 1 channel, 4x4
        (1, 1, 2, 2), // Weight: 1 out_ch, 1 in_ch, 2x2
    )?;

    println!("Input (4x4):");
    print_tensor_2d(&input, 4, 4);

    println!("\nAveraging kernel (2x2):");
    print_tensor_2d(&weight, 2, 2);

    println!("\nOutput with stride=2 (2x2):");
    print_tensor_2d(&output, 2, 2);

    println!("✓ Stride and padding demo completed\n");

    Ok(())
}

/// Demonstrates quantized convolution with I2S quantization
fn quantized_convolution_demo() -> Result<()> {
    println!("📋 Demo 3: Quantized Convolution (I2S)");
    println!("--------------------------------------");

    // Simple 2x2 input
    let input = vec![1.0, 2.0, 3.0, 4.0];

    // I2S quantized weights representing [-2, -1, 1, 2]
    // Packed as: 00|01|10|11 = 0b00011011 = 0x1B
    let weight_quantized = vec![0x1B];

    // Scale factor for the single output channel
    let weight_scales = vec![0.5];

    let mut output = vec![0.0; 1];

    conv2d_quantized(
        &input,
        &weight_quantized,
        &weight_scales,
        None,
        &mut output,
        Conv2DParams::default(),
        (1, 1, 2, 2), // Input: 1 batch, 1 channel, 2x2
        (1, 1, 2, 2), // Weight: 1 out_ch, 1 in_ch, 2x2
        QuantizationType::I2S,
    )?;

    println!("Input (2x2): {:?}", input);
    println!("Quantized weights (I2S): [-2, -1, 1, 2] * scale({})", weight_scales[0]);
    println!("Raw quantized byte: 0x{:02X}", weight_quantized[0]);
    println!("Output: {:.2}", output[0]);

    // Manual calculation for verification
    let manual_result = 1.0 * (-2.0 * 0.5) + 2.0 * (-0.5) + 3.0 * (1.0 * 0.5) + 4.0 * (2.0 * 0.5);
    println!("Manual calculation: 1×(-1) + 2×(-0.5) + 3×0.5 + 4×1 = {:.2}", manual_result);

    println!("✓ Quantized convolution completed\n");

    Ok(())
}

/// Compares different quantization types
fn quantization_comparison_demo() -> Result<()> {
    println!("📋 Demo 4: Quantization Types Comparison");
    println!("----------------------------------------");

    let input = vec![1.0, -1.0, 2.0, -2.0]; // 2x2 input with positive and negative values
    let weight_scales = vec![1.0];
    let mut output_i2s = vec![0.0; 1];
    let mut output_tl1 = vec![0.0; 1];
    let mut output_tl2 = vec![0.0; 1];

    // I2S: [-2, -1, 1, 2] packed in one byte
    let weight_i2s = vec![0x1B]; // 00|01|10|11

    // TL1: [0, 64, 192, 255] -> linear mapping to [-1, -0.5, 0.5, 1]
    let weight_tl1 = vec![0, 64, 192, 255];

    // TL2: Same quantized values, different dequantization
    let weight_tl2 = vec![0, 64, 192, 255];

    // Test I2S
    conv2d_quantized(
        &input,
        &weight_i2s,
        &weight_scales,
        None,
        &mut output_i2s,
        Conv2DParams::default(),
        (1, 1, 2, 2),
        (1, 1, 2, 2),
        QuantizationType::I2S,
    )?;

    // Test TL1
    conv2d_quantized(
        &input,
        &weight_tl1,
        &weight_scales,
        None,
        &mut output_tl1,
        Conv2DParams::default(),
        (1, 1, 2, 2),
        (1, 1, 2, 2),
        QuantizationType::TL1,
    )?;

    // Test TL2
    conv2d_quantized(
        &input,
        &weight_tl2,
        &weight_scales,
        None,
        &mut output_tl2,
        Conv2DParams::default(),
        (1, 1, 2, 2),
        (1, 1, 2, 2),
        QuantizationType::TL2,
    )?;

    println!("Input: {:?}", input);
    println!("Results:");
    println!("  I2S quantization:  {:.4}", output_i2s[0]);
    println!("  TL1 quantization:  {:.4}", output_tl1[0]);
    println!("  TL2 quantization:  {:.4}", output_tl2[0]);

    println!("\nQuantization schemes:");
    println!("  I2S: 2-bit signed values [-2, -1, 1, 2]");
    println!("  TL1: Linear mapping from [0, 255] to [-1, 1]");
    println!("  TL2: Non-linear mapping for enhanced precision");

    println!("✓ Quantization comparison completed\n");

    Ok(())
}

/// Helper function to print a 2D tensor in a readable format
fn print_tensor_2d(data: &[f32], height: usize, width: usize) {
    for row in 0..height {
        print!("  ");
        for col in 0..width {
            let idx = row * width + col;
            print!("{:6.1}", data[idx]);
        }
        println!();
    }
}

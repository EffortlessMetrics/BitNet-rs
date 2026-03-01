//! Scalar reference kernels used as portable fallback implementations.

use bitnet_common::{BitNetError, KernelError, QuantizationType, Result};

/// Scalar matrix multiplication for I2_S packed weights.
pub fn matmul_i2s(a: &[i8], b: &[u8], c: &mut [f32], m: usize, n: usize, k: usize) -> Result<()> {
    if a.len() != m * k {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("Matrix A dimension mismatch: expected {}, got {}", m * k, a.len()),
        }));
    }
    if b.len() != k * n {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("Matrix B dimension mismatch: expected {}, got {}", k * n, b.len()),
        }));
    }
    if c.len() != m * n {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("Matrix C dimension mismatch: expected {}, got {}", m * n, c.len()),
        }));
    }

    c.fill(0.0);
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += (a[i * k + l] as f32) * (b[l * n + j] as f32);
            }
            c[i * n + j] = sum;
        }
    }
    Ok(())
}

/// Scalar quantization shared by fallback implementations.
pub fn quantize(
    input: &[f32],
    output: &mut [u8],
    scales: &mut [f32],
    qtype: QuantizationType,
) -> Result<()> {
    output.fill(0);
    match qtype {
        QuantizationType::I2S => quantize_i2s(input, output, scales),
        QuantizationType::TL1 => {
            quantize_lut(input, output, scales, 64, [-1.0, -0.33, 0.33, 1.0], "TL1")
        }
        QuantizationType::TL2 => {
            quantize_lut(input, output, scales, 128, [-1.2, -0.4, 0.4, 1.2], "TL2")
        }
    }
}

fn quantize_i2s(input: &[f32], output: &mut [u8], scales: &mut [f32]) -> Result<()> {
    const BLOCK_SIZE: usize = 32;
    let num_blocks = input.len().div_ceil(BLOCK_SIZE);
    validate_quant_buffers(input.len(), output.len(), scales.len(), num_blocks, "I2_S")?;

    for (block_idx, scale) in scales.iter_mut().enumerate().take(num_blocks) {
        let start = block_idx * BLOCK_SIZE;
        let end = (start + BLOCK_SIZE).min(input.len());
        let block = &input[start..end];

        let max_val = block.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        *scale = if max_val > 1e-8 { max_val / 1.5 } else { 1.0 };

        for (i, &val) in block.iter().enumerate() {
            let normalized = val / *scale;
            let quantized = if normalized > 0.5 {
                1u8
            } else if normalized < -0.5 {
                3u8
            } else {
                0u8
            };

            let global_idx = start + i;
            let byte_idx = global_idx / 4;
            let bit_offset = (global_idx % 4) * 2;
            output[byte_idx] |= quantized << bit_offset;
        }
    }

    Ok(())
}

fn quantize_lut(
    input: &[f32],
    output: &mut [u8],
    scales: &mut [f32],
    block_size: usize,
    lut: [f32; 4],
    label: &str,
) -> Result<()> {
    let num_blocks = input.len().div_ceil(block_size);
    validate_quant_buffers(input.len(), output.len(), scales.len(), num_blocks, label)?;

    for (block_idx, scale) in scales.iter_mut().enumerate().take(num_blocks) {
        let start = block_idx * block_size;
        let end = (start + block_size).min(input.len());
        let block = &input[start..end];

        let max_val = block.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        *scale = if max_val > 1e-8 { max_val / 1.5 } else { 1.0 };

        for (i, &val) in block.iter().enumerate() {
            let normalized = val / *scale;
            let mut best_idx = 0usize;
            let mut best_dist = (normalized - lut[0]).abs();
            for (idx, &lut_val) in lut.iter().enumerate().skip(1) {
                let dist = (normalized - lut_val).abs();
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = idx;
                }
            }

            let global_idx = start + i;
            let byte_idx = global_idx / 4;
            let bit_offset = (global_idx % 4) * 2;
            output[byte_idx] |= (best_idx as u8) << bit_offset;
        }
    }

    Ok(())
}

fn validate_quant_buffers(
    input_len: usize,
    output_len: usize,
    scales_len: usize,
    num_blocks: usize,
    label: &str,
) -> Result<()> {
    let min_output = input_len.div_ceil(4);
    if output_len < min_output {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!(
                "Output buffer too small for {label}: expected {min_output}, got {output_len}"
            ),
        }));
    }
    if scales_len < num_blocks {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("Scales buffer too small: expected {num_blocks}, got {scales_len}"),
        }));
    }
    Ok(())
}

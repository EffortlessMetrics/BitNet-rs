//! CPU reference implementations for GPU quantization kernels.
//!
//! These functions mirror the OpenCL QK256, TL1, and TL2 kernels exactly,
//! providing a ground-truth for validating GPU kernel correctness.

/// QK256 block size (elements per block).
pub const QK256_BLOCK_SIZE: usize = 256;
/// QK256 packed bytes per block (256 elements * 2 bits / 8 bits).
pub const QK256_BYTES_PER_BLOCK: usize = 64;

/// Decode a 2-bit representation to ternary value.
fn ternary_decode(bits: u8) -> f32 {
    match bits & 0x03 {
        0x01 => 1.0,
        0x03 => -1.0,
        _ => 0.0,
    }
}

// --- QK256 reference implementations ---

/// Dequantize QK256 blocks: unpack 2-bit ternary values and apply scales.
pub fn qk256_dequantize_ref(
    packed_data: &[u8],
    scales: &[f32],
    num_blocks: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; num_blocks * QK256_BLOCK_SIZE];

    for block_id in 0..num_blocks {
        let scale = scales[block_id];
        let data_offset = block_id * QK256_BYTES_PER_BLOCK;

        for elem in 0..QK256_BLOCK_SIZE {
            let byte_idx = data_offset + elem / 4;
            let bit_shift = (elem % 4) * 2;
            let bits = (packed_data[byte_idx] >> bit_shift) & 0x03;
            let value = ternary_decode(bits);
            output[block_id * QK256_BLOCK_SIZE + elem] = value * scale;
        }
    }

    output
}

/// QK256 matrix-vector multiply: packed weights * activations -> output.
pub fn qk256_matmul_ref(
    weights: &[u8],
    weight_scales: &[f32],
    activations: &[f32],
    n: usize,
    k: usize,
) -> Vec<f32> {
    let blocks_per_row = k / QK256_BLOCK_SIZE;
    let mut output = vec![0.0f32; n];

    for row in 0..n {
        let mut sum = 0.0f32;

        for b in 0..blocks_per_row {
            let block_idx = row * blocks_per_row + b;
            let scale = weight_scales[block_idx];
            let data_offset = block_idx * QK256_BYTES_PER_BLOCK;
            let k_base = b * QK256_BLOCK_SIZE;

            for byte_i in 0..QK256_BYTES_PER_BLOCK {
                for sub in 0..4u32 {
                    let ki = k_base + byte_i * 4 + sub as usize;
                    if ki >= k {
                        break;
                    }
                    let bits = (weights[data_offset + byte_i] >> (sub * 2)) & 0x03;
                    let w = ternary_decode(bits);
                    sum += w * scale * activations[ki];
                }
            }
        }

        output[row] = sum;
    }

    output
}

/// Apply per-block scales to integer dot products.
pub fn qk256_apply_scales_ref(accumulated: &[i32], scales: &[f32]) -> Vec<f32> {
    accumulated
        .iter()
        .zip(scales.iter())
        .map(|(&acc, &scale)| acc as f32 * scale)
        .collect()
}

// --- TL1 reference implementations ---

/// Pack float32 values to TL1 ternary format with per-group scales.
pub fn tl1_pack_ref(input: &[f32], group_size: usize) -> (Vec<u8>, Vec<f32>) {
    let n = input.len();
    let num_groups = (n + group_size - 1) / group_size;
    let mut packed = vec![0u8; (n + 3) / 4];
    let mut scales = vec![0.0f32; num_groups];

    for group_id in 0..num_groups {
        let start = group_id * group_size;
        let end = (start + group_size).min(n);

        let absmax = input[start..end]
            .iter()
            .map(|x| x.abs())
            .fold(0.0f32, f32::max);
        let scale = if absmax > 0.0 { absmax } else { 1.0 };
        scales[group_id] = scale;

        let mut i = start;
        while i < end {
            let mut byte_val = 0u8;
            for sub in 0..4 {
                if i + sub >= end {
                    break;
                }
                let normalized = input[i + sub] / scale;
                let ternary: u8 = if normalized > 0.5 {
                    0x01
                } else if normalized < -0.5 {
                    0x03
                } else {
                    0x00
                };
                byte_val |= ternary << (sub * 2);
            }
            packed[i / 4] = byte_val;
            i += 4;
        }
    }

    (packed, scales)
}

/// Unpack TL1 ternary format to float32.
pub fn tl1_unpack_ref(packed: &[u8], scales: &[f32], n: usize, group_size: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; n];

    for i in 0..n {
        let group_id = i / group_size;
        let scale = scales[group_id];
        let byte_idx = i / 4;
        let bit_shift = (i % 4) * 2;
        let bits = (packed[byte_idx] >> bit_shift) & 0x03;
        output[i] = ternary_decode(bits) * scale;
    }

    output
}

/// TL1 matrix-vector multiply with per-row scales.
pub fn tl1_matmul_ref(
    weights: &[u8],
    scales: &[f32],
    input: &[f32],
    n: usize,
    k: usize,
) -> Vec<f32> {
    let k_packed = k / 4;
    let mut output = vec![0.0f32; n];

    for row in 0..n {
        let mut sum = 0.0f32;

        for kp in 0..k_packed {
            let packed = weights[row * k_packed + kp];
            for sub in 0..4u32 {
                let ki = kp * 4 + sub as usize;
                if ki >= k {
                    break;
                }
                let bits = (packed >> (sub * 2)) & 0x03;
                let w = ternary_decode(bits);
                sum += w * input[ki];
            }
        }

        output[row] = sum * scales[row];
    }

    output
}

// --- TL2 reference implementations ---

/// TL2 dequantize with per-group scales.
pub fn tl2_dequantize_ref(
    packed_data: &[u8],
    group_scales: &[f32],
    n: usize,
    group_size: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; n];

    for i in 0..n {
        let group_id = i / group_size;
        let scale = group_scales[group_id];
        let byte_idx = i / 4;
        let bit_shift = (i % 4) * 2;
        let bits = (packed_data[byte_idx] >> bit_shift) & 0x03;
        output[i] = ternary_decode(bits) * scale;
    }

    output
}

/// TL2 quantize: float32 to packed ternary with per-group scales.
pub fn tl2_quantize_ref(input: &[f32], group_size: usize) -> (Vec<u8>, Vec<f32>) {
    let n = input.len();
    let num_groups = (n + group_size - 1) / group_size;
    let mut packed = vec![0u8; (n + 3) / 4];
    let mut group_scales = vec![0.0f32; num_groups];

    for group_id in 0..num_groups {
        let start = group_id * group_size;
        let end = (start + group_size).min(n);

        let absmax = input[start..end]
            .iter()
            .map(|x| x.abs())
            .fold(0.0f32, f32::max);
        let scale = if absmax > 0.0 { absmax } else { 1.0 };
        group_scales[group_id] = scale;

        let mut i = start;
        while i < end {
            let mut byte_val = 0u8;
            for sub in 0..4 {
                if i + sub >= end {
                    break;
                }
                let normalized = input[i + sub] / scale;
                let ternary: u8 = if normalized > 0.5 {
                    0x01
                } else if normalized < -0.5 {
                    0x03
                } else {
                    0x00
                };
                byte_val |= ternary << (sub * 2);
            }
            packed[i / 4] = byte_val;
            i += 4;
        }
    }

    (packed, group_scales)
}

/// TL2 matrix-vector multiply with per-group scales.
pub fn tl2_matmul_ref(
    weights: &[u8],
    group_scales: &[f32],
    input: &[f32],
    n: usize,
    k: usize,
    group_size: usize,
) -> Vec<f32> {
    let k_packed = k / 4;
    let groups_per_row = (k + group_size - 1) / group_size;
    let mut output = vec![0.0f32; n];

    for row in 0..n {
        let mut sum = 0.0f32;

        for kp in 0..k_packed {
            let packed = weights[row * k_packed + kp];
            let k_base = kp * 4;

            for sub in 0..4u32 {
                let ki = k_base + sub as usize;
                if ki >= k {
                    break;
                }
                let group_id = row * groups_per_row + ki / group_size;
                let scale = group_scales[group_id];
                let bits = (packed >> (sub * 2)) & 0x03;
                let w = ternary_decode(bits);
                sum += w * scale * input[ki];
            }
        }

        output[row] = sum;
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ternary_encode_decode_roundtrip() {
        assert_eq!(ternary_decode(0x01), 1.0);
        assert_eq!(ternary_decode(0x03), -1.0);
        assert_eq!(ternary_decode(0x00), 0.0);
    }

    #[test]
    fn qk256_dequantize_all_zeros() {
        let packed = vec![0u8; QK256_BYTES_PER_BLOCK];
        let scales = vec![1.0f32];
        let output = qk256_dequantize_ref(&packed, &scales, 1);
        assert!(output.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn qk256_dequantize_positive_ones() {
        // All elements = +1 (0b01 repeated): 0b01010101 = 0x55
        let packed = vec![0x55u8; QK256_BYTES_PER_BLOCK];
        let scales = vec![2.0f32];
        let output = qk256_dequantize_ref(&packed, &scales, 1);
        assert_eq!(output.len(), QK256_BLOCK_SIZE);
        assert!(output.iter().all(|&v| (v - 2.0).abs() < 1e-6));
    }

    #[test]
    fn qk256_dequantize_negative_ones() {
        // All elements = -1 (0b11 repeated): 0xFF
        let packed = vec![0xFFu8; QK256_BYTES_PER_BLOCK];
        let scales = vec![3.0f32];
        let output = qk256_dequantize_ref(&packed, &scales, 1);
        assert!(output.iter().all(|&v| (v + 3.0).abs() < 1e-6));
    }

    #[test]
    fn qk256_apply_scales_ref_correctness() {
        let acc = vec![10, -5, 0, 7];
        let scales = vec![0.5, 2.0, 1.0, 0.1];
        let result = qk256_apply_scales_ref(&acc, &scales);
        assert!((result[0] - 5.0).abs() < 1e-6);
        assert!((result[1] + 10.0).abs() < 1e-6);
        assert!((result[2] - 0.0).abs() < 1e-6);
        assert!((result[3] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn tl1_pack_unpack_roundtrip() {
        let input = vec![1.0, -1.0, 0.0, 0.8, -0.9, 0.1, 0.7, -0.6];
        let group_size = 8;
        let (packed, scales) = tl1_pack_ref(&input, group_size);
        let output = tl1_unpack_ref(&packed, &scales, input.len(), group_size);
        let scale = scales[0];
        assert!((output[0] - scale).abs() < 1e-6); // 1.0 -> +1*scale
        assert!((output[1] + scale).abs() < 1e-6); // -1.0 -> -1*scale
        assert!((output[2] - 0.0).abs() < 1e-6); // 0.0 -> 0
    }

    #[test]
    fn tl1_matmul_simple() {
        // Row 0: [+1, -1, 0, +1] = packed 0b01_00_11_01 = 0x4D
        // Row 1: [0, +1, +1, 0]  = packed 0b00_01_01_00 = 0x14
        let weights = vec![0x4Du8, 0x14u8];
        let scales = vec![1.0, 1.0];
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = tl1_matmul_ref(&weights, &scales, &input, 2, 4);
        // Row 0: 1*1 + (-1)*2 + 0*3 + 1*4 = 3
        assert!((output[0] - 3.0).abs() < 1e-6);
        // Row 1: 0*1 + 1*2 + 1*3 + 0*4 = 5
        assert!((output[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn tl2_quantize_dequantize_roundtrip() {
        let input = vec![0.8, -0.9, 0.1, 0.7, -0.6, 0.0, 1.0, -1.0];
        let group_size = 4;
        let (packed, scales) = tl2_quantize_ref(&input, group_size);
        let output = tl2_dequantize_ref(&packed, &scales, input.len(), group_size);
        let s0 = scales[0]; // absmax of [0.8, -0.9, 0.1, 0.7] = 0.9
        assert!((output[0] - s0).abs() < 1e-6); // 0.8/0.9 > 0.5 -> +1*s0
        assert!((output[1] + s0).abs() < 1e-6); // -0.9/0.9 < -0.5 -> -1*s0
    }

    #[test]
    fn tl2_matmul_with_groups() {
        // 1x8 weight, group_size=4, so 2 groups per row
        // Group 0: [+1, -1, +1, 0] scale=2.0 -> packed 0b00_01_11_01 = 0x1D
        // Group 1: [0, +1, -1, +1] scale=3.0 -> packed 0b01_11_01_00 = 0x74
        let weights = vec![0x1Du8, 0x74u8];
        let group_scales = vec![2.0, 3.0];
        let input = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let output = tl2_matmul_ref(&weights, &group_scales, &input, 1, 8, 4);
        // Group 0: 2*(1 + (-1) + 1 + 0) = 2
        // Group 1: 3*(0 + 1 + (-1) + 1) = 3
        assert!((output[0] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn qk256_matmul_identity_like() {
        let k = QK256_BLOCK_SIZE;
        let n = 1;
        // All weights = +1 (0x55), scale = 1.0
        let weights = vec![0x55u8; QK256_BYTES_PER_BLOCK];
        let scales = vec![1.0f32];
        let activations: Vec<f32> = (0..k).map(|i| i as f32).collect();
        let output = qk256_matmul_ref(&weights, &scales, &activations, n, k);
        // Sum of 0..255 = 255*256/2 = 32640
        assert!((output[0] - 32640.0).abs() < 1e-2);
    }
}
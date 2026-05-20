//! Small selected-device A770 OpenCL fixtures.
//!
//! These helpers describe the diagnostic `matmul_i2s` fixture used by the A770
//! smoke/parity lane. They are not official BitNet QK256 production semantics.

pub const A770_MATMUL_I2S_M: usize = 2;
pub const A770_MATMUL_I2S_N: usize = 3;
pub const A770_MATMUL_I2S_K: usize = 8;
pub const A770_QK256_SCALED_ROWS: usize = 2;
pub const A770_QK256_SCALED_COLS: usize = 256;
pub const A770_QK256_PACKED_BYTES_PER_BLOCK: usize = 64;
pub const A770_QK256_SCALED_ROW_STRIDE_BYTES: usize = A770_QK256_PACKED_BYTES_PER_BLOCK;
pub const A770_QK256_SCALED_ACT_SCALE: f32 = 1.0;
pub const A770_QK256_SCALED_WEIGHT_SCALE: f32 = 0.25;

/// Int8 activation matrix in row-major `[M, K]` order.
pub const A770_MATMUL_I2S_ACTIVATIONS: [i8; A770_MATMUL_I2S_M * A770_MATMUL_I2S_K] = [
    1, -2, 3, 4, -5, 6, 7, -8, //
    -3, 5, -7, 9, 11, -13, 15, -17,
];

/// I2_S weight matrix in row-major `[K, N]` order before packing.
pub const A770_MATMUL_I2S_WEIGHTS: [i8; A770_MATMUL_I2S_K * A770_MATMUL_I2S_N] = [
    1, 0, -1, //
    0, 1, 1, //
    -1, 1, 0, //
    1, -1, 0, //
    0, -1, 1, //
    1, 0, -1, //
    -1, 1, 1, //
    0, -1, 1,
];

/// Pre-quantized I8_S activation row for the QK256 scaled fixture.
///
/// This fixture starts after activation quantization. It exercises the official
/// grouped QK256 byte layout and `(dot - activation_sum) / activation_scale *
/// weight_scale` correction, but it does not prove GPU-resident activation
/// quantization or full BitNet inference.
pub fn a770_qk256_scaled_i8s_activations() -> Vec<i8> {
    let pattern = [-127, -13, -3, -1, 0, 1, 7, 31, 64, 96, 127];
    (0..A770_QK256_SCALED_COLS).map(|idx| pattern[idx % pattern.len()]).collect()
}

pub fn a770_qk256_scaled_activation_sum() -> i32 {
    a770_qk256_scaled_i8s_activations().iter().map(|&value| value as i32).sum()
}

pub fn a770_qk256_scaled_codes() -> Vec<u8> {
    let mut codes = Vec::with_capacity(A770_QK256_SCALED_ROWS * A770_QK256_SCALED_COLS);
    for row in 0..A770_QK256_SCALED_ROWS {
        for col in 0..A770_QK256_SCALED_COLS {
            let code = match (row, col % 8) {
                (0, 0 | 5) => 0,
                (0, 1 | 6) => 1,
                (0, 2 | 7) => 2,
                (0, _) => 3,
                (_, 0 | 4) => 2,
                (_, 1 | 5) => 0,
                (_, 2 | 6) => 3,
                (_, _) => 1,
            };
            codes.push(code);
        }
    }
    codes
}

pub fn pack_a770_qk256_scaled_weights() -> Result<Vec<u8>, String> {
    pack_qk256_grouped_codes(
        &a770_qk256_scaled_codes(),
        A770_QK256_SCALED_ROWS,
        A770_QK256_SCALED_COLS,
    )
}

pub fn a770_qk256_scaled_cpu_reference() -> Result<Vec<f32>, String> {
    let q = a770_qk256_scaled_i8s_activations();
    let act_sum = a770_qk256_scaled_activation_sum();
    let packed = pack_a770_qk256_scaled_weights()?;
    let mut output = Vec::with_capacity(A770_QK256_SCALED_ROWS);
    for row in 0..A770_QK256_SCALED_ROWS {
        let row_start = row * A770_QK256_SCALED_ROW_STRIDE_BYTES;
        let row_end = row_start + A770_QK256_SCALED_ROW_STRIDE_BYTES;
        let int_dot = qk256_i8s_int_dot(&packed[row_start..row_end], &q, A770_QK256_SCALED_COLS);
        output.push(
            ((int_dot - act_sum) as f32 / A770_QK256_SCALED_ACT_SCALE)
                * A770_QK256_SCALED_WEIGHT_SCALE,
        );
    }
    Ok(output)
}

pub fn pack_a770_matmul_i2s_weights() -> Result<Vec<u8>, String> {
    pack_i2s_k_by_n_weights(&A770_MATMUL_I2S_WEIGHTS, A770_MATMUL_I2S_N, A770_MATMUL_I2S_K)
}

pub fn a770_matmul_i2s_cpu_reference() -> Vec<f32> {
    cpu_matmul_i2s_reference(
        &A770_MATMUL_I2S_ACTIVATIONS,
        &A770_MATMUL_I2S_WEIGHTS,
        A770_MATMUL_I2S_M,
        A770_MATMUL_I2S_N,
        A770_MATMUL_I2S_K,
    )
}

pub fn pack_i2s_k_by_n_weights(weights: &[i8], n: usize, k: usize) -> Result<Vec<u8>, String> {
    if n == 0 || k == 0 {
        return Err("I2_S fixture dimensions must be non-zero".to_string());
    }
    if weights.len() != n * k {
        return Err(format!(
            "I2_S fixture has {} weights, expected {} for K={k}, N={n}",
            weights.len(),
            n * k
        ));
    }
    if !k.is_multiple_of(4) {
        return Err(format!(
            "I2_S fixture K must be a multiple of 4 for the current matmul_i2s kernel, got {k}"
        ));
    }

    let k_packed = k / 4;
    let mut packed = vec![0u8; k_packed * n];
    for kp in 0..k_packed {
        for col in 0..n {
            let mut byte = 0u8;
            for sub in 0..4 {
                let depth = kp * 4 + sub;
                if depth >= k {
                    break;
                }
                let weight = weights[depth * n + col];
                byte |= encode_i2s_weight(weight)? << (sub * 2);
            }
            packed[kp * n + col] = byte;
        }
    }
    Ok(packed)
}

pub fn pack_qk256_grouped_codes(codes: &[u8], rows: usize, cols: usize) -> Result<Vec<u8>, String> {
    if rows == 0 || cols == 0 {
        return Err("QK256 fixture dimensions must be non-zero".to_string());
    }
    if !cols.is_multiple_of(256) {
        return Err(format!("QK256 fixture cols must be a multiple of 256, got {cols}"));
    }
    if codes.len() != rows * cols {
        return Err(format!(
            "QK256 fixture has {} codes, expected {} for rows={rows}, cols={cols}",
            codes.len(),
            rows * cols
        ));
    }

    let row_stride = (cols / 256) * A770_QK256_PACKED_BYTES_PER_BLOCK;
    let mut packed = vec![0u8; rows * row_stride];
    for row in 0..rows {
        for block in 0..(cols / 256) {
            let code_base = row * cols + block * 256;
            let byte_base = row * row_stride + block * A770_QK256_PACKED_BYTES_PER_BLOCK;
            pack_qk256_block(
                &codes[code_base..code_base + 256],
                &mut packed[byte_base..byte_base + 64],
            )?;
        }
    }
    Ok(packed)
}

fn pack_qk256_block(codes: &[u8], out: &mut [u8]) -> Result<(), String> {
    debug_assert_eq!(codes.len(), 256);
    debug_assert_eq!(out.len(), A770_QK256_PACKED_BYTES_PER_BLOCK);
    for chunk in 0..2 {
        let byte_base = chunk * 32;
        let elem_base = chunk * 128;
        for gp in 0..32 {
            let mut byte = 0u8;
            for lane in 0..4 {
                let code = codes[elem_base + lane * 32 + gp];
                if code > 3 {
                    return Err(format!("unsupported QK256 code {code}"));
                }
                byte |= code << (6 - lane * 2);
            }
            out[byte_base + gp] = byte;
        }
    }
    Ok(())
}

pub fn qk256_grouped_code(row_bytes: &[u8], col: usize) -> u8 {
    let block = col / 256;
    let offset = col % 256;
    let chunk = offset / 128;
    let lane = (offset % 128) / 32;
    let gp = offset % 32;
    let byte_index = block * A770_QK256_PACKED_BYTES_PER_BLOCK + chunk * 32 + gp;
    (row_bytes[byte_index] >> (6 - lane * 2)) & 0x03
}

fn qk256_i8s_int_dot(row_bytes: &[u8], q: &[i8], cols: usize) -> i32 {
    let mut int_dot = 0i32;
    for (col, &activation) in q.iter().enumerate().take(cols) {
        int_dot += qk256_grouped_code(row_bytes, col) as i32 * activation as i32;
    }
    int_dot
}

pub fn cpu_matmul_i2s_reference(
    activations: &[i8],
    weights: &[i8],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for depth in 0..k {
                let activation = activations[row * k + depth] as f32;
                let weight = weights[depth * n + col] as f32;
                sum += activation * weight;
            }
            output[row * n + col] = sum;
        }
    }
    output
}

fn encode_i2s_weight(value: i8) -> Result<u8, String> {
    match value {
        1 => Ok(0x01),
        -1 => Ok(0x03),
        0 => Ok(0x00),
        other => Err(format!("unsupported I2_S fixture weight {other}")),
    }
}

#[cfg(test)]
fn decode_i2s_weight(bits: u8) -> i8 {
    match bits & 0x03 {
        0x01 => 1,
        0x03 => -1,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a770_fixture_packs_weights_in_kernel_k_by_n_order() -> Result<(), String> {
        let packed = pack_a770_matmul_i2s_weights()?;
        assert_eq!(packed.len(), A770_MATMUL_I2S_K.div_ceil(4) * A770_MATMUL_I2S_N);

        for depth in 0..A770_MATMUL_I2S_K {
            for col in 0..A770_MATMUL_I2S_N {
                let kp = depth / 4;
                let sub = depth % 4;
                let byte = packed[kp * A770_MATMUL_I2S_N + col];
                let decoded = decode_i2s_weight(byte >> (sub * 2));
                assert_eq!(decoded, A770_MATMUL_I2S_WEIGHTS[depth * A770_MATMUL_I2S_N + col]);
            }
        }
        Ok(())
    }

    #[test]
    fn a770_fixture_cpu_reference_matches_contract_shape() {
        let expected = vec![1.0, 17.0, -15.0, -15.0, 10.0, 30.0];
        assert_eq!(a770_matmul_i2s_cpu_reference(), expected);
    }

    #[test]
    fn a770_fixture_rejects_non_i2s_weight_values() {
        let err =
            pack_i2s_k_by_n_weights(&[2, 0, 0, 0], 1, 4).expect_err("invalid I2_S value rejected");
        assert!(err.contains("unsupported I2_S fixture weight 2"));
    }

    #[test]
    fn a770_fixture_rejects_non_kernel_depth_multiple() {
        let err =
            pack_i2s_k_by_n_weights(&[0, 1, -1, 0, 1], 1, 5).expect_err("non-multiple K rejected");
        assert!(err.contains("K must be a multiple of 4"));
    }

    #[test]
    fn a770_qk256_fixture_packs_grouped_bitnet_layout() -> Result<(), String> {
        let codes = a770_qk256_scaled_codes();
        let packed = pack_a770_qk256_scaled_weights()?;
        assert_eq!(packed.len(), A770_QK256_SCALED_ROWS * A770_QK256_SCALED_ROW_STRIDE_BYTES);

        for row in 0..A770_QK256_SCALED_ROWS {
            let row_start = row * A770_QK256_SCALED_ROW_STRIDE_BYTES;
            let row_bytes = &packed[row_start..row_start + A770_QK256_SCALED_ROW_STRIDE_BYTES];
            for col in 0..A770_QK256_SCALED_COLS {
                assert_eq!(
                    qk256_grouped_code(row_bytes, col),
                    codes[row * A770_QK256_SCALED_COLS + col]
                );
            }
        }
        Ok(())
    }

    #[test]
    fn a770_qk256_scaled_cpu_reference_locks_scale_sum_formula() -> Result<(), String> {
        let expected = vec![510.75, 506.0];
        assert_eq!(a770_qk256_scaled_activation_sum(), 4043);
        assert_eq!(a770_qk256_scaled_cpu_reference()?, expected);
        Ok(())
    }

    #[test]
    fn a770_qk256_fixture_rejects_bad_codes() {
        let err = pack_qk256_grouped_codes(&[4; 256], 1, 256).expect_err("bad code rejected");
        assert!(err.contains("unsupported QK256 code 4"));
    }
}

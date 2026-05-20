//! Small selected-device A770 OpenCL fixtures.
//!
//! These helpers describe the diagnostic `matmul_i2s` fixture used by the A770
//! smoke/parity lane. They are not official BitNet QK256 production semantics.

pub const A770_MATMUL_I2S_M: usize = 2;
pub const A770_MATMUL_I2S_N: usize = 3;
pub const A770_MATMUL_I2S_K: usize = 8;

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
}

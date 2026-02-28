// Extended property-based tests for OpenCL kernel CPU reference implementations.
//
// Each test validates mathematical invariants that any correct implementation
// must satisfy, regardless of precision or hardware backend.

use proptest::prelude::*;

// ============================================================================
// CPU reference implementations mirroring the OpenCL kernels
// ============================================================================

/// CPU reference matmul: C[m][n] = sum_k(A_ternary[m][k] * B[k][n])
fn ref_matmul_i2s(a_packed: &[u8], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for i in 0..k {
                let byte_idx = (row * k + i) / 4;
                let sub = (row * k + i) % 4;
                let bits = if byte_idx < a_packed.len() {
                    (a_packed[byte_idx] >> (sub * 2)) & 0x03
                } else {
                    0
                };
                let w: f32 = match bits {
                    0x01 => 1.0,
                    0x03 => -1.0,
                    _ => 0.0,
                };
                if i * n + col < b.len() {
                    sum += w * b[i * n + col];
                }
            }
            c[row * n + col] = sum;
        }
    }
    c
}

/// CPU reference RMS normalization.
fn ref_rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len();
    let sum_sq: f32 = input.iter().map(|x| x * x).sum();
    let rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();
    input.iter().zip(weight.iter()).map(|(&x, &w)| x * rms * w).collect()
}

/// CPU reference RoPE (Rotary Position Embedding).
fn ref_rope(input: &[f32], position: usize, head_dim: usize) -> Vec<f32> {
    let mut output = input.to_vec();
    for i in (0..head_dim).step_by(2) {
        if i + 1 >= input.len() {
            break;
        }
        let freq = 1.0 / (10000.0f64.powf(i as f64 / head_dim as f64));
        let angle = position as f64 * freq;
        let cos_val = angle.cos() as f32;
        let sin_val = angle.sin() as f32;
        let x0 = input[i];
        let x1 = input[i + 1];
        output[i] = x0 * cos_val - x1 * sin_val;
        output[i + 1] = x0 * sin_val + x1 * cos_val;
    }
    output
}

/// CPU reference softmax.
fn ref_softmax(input: &[f32]) -> Vec<f32> {
    let max_val = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = input.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        vec![1.0 / input.len() as f32; input.len()]
    } else {
        exps.iter().map(|&e| e / sum).collect()
    }
}

/// CPU reference attention: softmax(Q * K^T / sqrt(d)) * V
fn ref_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    causal: bool,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; seq_len * head_dim];

    for i in 0..seq_len {
        let mut scores = vec![0.0f32; seq_len];
        for j in 0..seq_len {
            if causal && j > i {
                scores[j] = f32::NEG_INFINITY;
            } else {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[i * head_dim + d] * k[j * head_dim + d];
                }
                scores[j] = dot * scale;
            }
        }

        let weights = ref_softmax(&scores);

        for d in 0..head_dim {
            let mut sum = 0.0f32;
            for j in 0..seq_len {
                sum += weights[j] * v[j * head_dim + d];
            }
            output[i * head_dim + d] = sum;
        }
    }
    output
}

/// CPU reference SiLU activation: x * sigmoid(x)
fn ref_silu(input: &[f32]) -> Vec<f32> {
    input
        .iter()
        .map(|&x| {
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            x * sigmoid
        })
        .collect()
}

/// CPU reference embedding lookup.
fn ref_embedding_lookup(table: &[f32], indices: &[usize], dim: usize) -> Option<Vec<f32>> {
    let vocab_size = table.len() / dim;
    let mut output = Vec::with_capacity(indices.len() * dim);
    for &idx in indices {
        if idx >= vocab_size {
            return None;
        }
        output.extend_from_slice(&table[idx * dim..(idx + 1) * dim]);
    }
    Some(output)
}

/// Pack ternary values into bytes (I2_S encoding).
fn pack_ternary(values: &[i8]) -> Vec<u8> {
    values
        .chunks(4)
        .map(|chunk| {
            let mut packed = 0u8;
            for (i, &v) in chunk.iter().enumerate() {
                let bits: u8 = match v {
                    1 => 0b01,
                    -1 => 0b11,
                    _ => 0b00,
                };
                packed |= bits << (i * 2);
            }
            packed
        })
        .collect()
}

/// Unpack bytes back to ternary values.
fn unpack_ternary(packed: &[u8], count: usize) -> Vec<i8> {
    let mut result = Vec::with_capacity(count);
    for &byte in packed {
        for sub in 0..4 {
            if result.len() >= count {
                break;
            }
            let bits = (byte >> (sub * 2)) & 0x03;
            result.push(match bits {
                0x01 => 1,
                0x03 => -1,
                _ => 0,
            });
        }
    }
    result
}

// ============================================================================
// Helper: generate small float vectors
// ============================================================================

fn vec_f32(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-10.0f32..10.0f32, min_len..=max_len)
}

// ============================================================================
// 1. Matmul: output dimensions, zero weights, linearity
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn matmul_output_dimensions(
        m in 1usize..=8,
        n in 1usize..=8,
        k in 1usize..=16,
    ) {
        let a_vals: Vec<i8> = (0..m * k).map(|i| [0, 1, -1][i % 3]).collect();
        let a_packed = pack_ternary(&a_vals);
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();
        let c = ref_matmul_i2s(&a_packed, &b, m, n, k);
        prop_assert_eq!(c.len(), m * n, "output must be M×N");
        for &val in &c {
            prop_assert!(val.is_finite(), "matmul output must be finite");
        }
    }

    #[test]
    fn matmul_zero_weights_produce_zero_output(
        m in 1usize..=8,
        n in 1usize..=8,
        k in 1usize..=16,
    ) {
        let a_packed = vec![0u8; (m * k + 3) / 4];
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let c = ref_matmul_i2s(&a_packed, &b, m, n, k);
        for &val in &c {
            prop_assert!((val).abs() < 1e-6, "zero weights must produce zero output, got {}", val);
        }
    }

    #[test]
    fn matmul_linearity_in_b(
        m in 1usize..=4,
        n in 1usize..=4,
        k in 1usize..=8,
        scale in 0.1f32..5.0,
    ) {
        let a_vals: Vec<i8> = (0..m * k).map(|i| [1, -1, 0][i % 3]).collect();
        let a_packed = pack_ternary(&a_vals);
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1 + 0.01).collect();

        let c1 = ref_matmul_i2s(&a_packed, &b, m, n, k);
        let b_scaled: Vec<f32> = b.iter().map(|&v| v * scale).collect();
        let c2 = ref_matmul_i2s(&a_packed, &b_scaled, m, n, k);

        for (i, (&v1, &v2)) in c1.iter().zip(c2.iter()).enumerate() {
            let expected = v1 * scale;
            prop_assert!(
                (v2 - expected).abs() < 1e-3,
                "linearity violated at index {}: {} * {} != {}", i, v1, scale, v2
            );
        }
    }
}

// ============================================================================
// 2. RMSNorm: unit RMS and direction preservation
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn rmsnorm_output_has_unit_rms_with_unit_weights(
        input in vec_f32(2, 64).prop_filter("non-zero input", |v| {
            v.iter().any(|&x| x.abs() > 1e-6)
        })
    ) {
        let weight = vec![1.0f32; input.len()];
        let eps = 1e-5;
        let output = ref_rms_norm(&input, &weight, eps);

        let n = output.len() as f32;
        let sum_sq: f32 = output.iter().map(|x| x * x).sum();
        let rms = (sum_sq / n).sqrt();

        prop_assert!(
            (rms - 1.0).abs() < 0.1,
            "RMS should be ~1.0, got {} for input len {}", rms, input.len()
        );
    }

    #[test]
    fn rmsnorm_preserves_direction(
        input in vec_f32(2, 64).prop_filter("non-zero input", |v| {
            v.iter().any(|&x| x.abs() > 1e-6)
        })
    ) {
        let weight = vec![1.0f32; input.len()];
        let eps = 1e-5;
        let output = ref_rms_norm(&input, &weight, eps);

        for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
            if inp.abs() > 1e-4 {
                prop_assert_eq!(
                    inp.signum() as i32,
                    out.signum() as i32,
                    "direction not preserved at index {}", i
                );
            }
        }
    }

    #[test]
    fn rmsnorm_output_is_finite(
        input in vec_f32(1, 64),
        eps in 1e-8f32..1e-2,
    ) {
        let weight = vec![1.0f32; input.len()];
        let output = ref_rms_norm(&input, &weight, eps);
        for (i, &val) in output.iter().enumerate() {
            prop_assert!(val.is_finite(), "non-finite at index {}: {}", i, val);
        }
    }
}

// ============================================================================
// 3. RoPE: orthogonal rotation (norm preservation) and periodicity
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn rope_preserves_vector_norm(
        dim in (1usize..=32).prop_map(|d| d * 2),
        position in 0usize..128,
    ) {
        let input: Vec<f32> = (0..dim).map(|i| ((i + 1) as f32) * 0.1).collect();
        let output = ref_rope(&input, position, dim);

        let input_norm: f32 = input.iter().map(|x| x * x).sum::<f32>().sqrt();
        let output_norm: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt();

        prop_assert!(
            (input_norm - output_norm).abs() < 1e-4,
            "RoPE should preserve norm: input={}, output={}", input_norm, output_norm
        );
    }

    #[test]
    fn rope_at_position_zero_is_identity(
        dim in (1usize..=32).prop_map(|d| d * 2),
    ) {
        let input: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.3 + 0.1).collect();
        let output = ref_rope(&input, 0, dim);

        for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
            prop_assert!(
                (inp - out).abs() < 1e-5,
                "position 0 should be identity at index {}: {} != {}", i, inp, out
            );
        }
    }

    #[test]
    fn rope_output_is_finite(
        dim in (1usize..=16).prop_map(|d| d * 2),
        position in 0usize..256,
    ) {
        let input: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.5).collect();
        let output = ref_rope(&input, position, dim);
        for &val in &output {
            prop_assert!(val.is_finite(), "RoPE output must be finite");
        }
    }
}

// ============================================================================
// 4. Softmax: sums to 1.0, values in [0,1], monotonic
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn softmax_sums_to_one(input in vec_f32(1, 32)) {
        let output = ref_softmax(&input);
        let sum: f32 = output.iter().sum();
        prop_assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax sum should be 1.0, got {}", sum
        );
    }

    #[test]
    fn softmax_all_values_in_unit_interval(input in vec_f32(1, 32)) {
        let output = ref_softmax(&input);
        for (i, &val) in output.iter().enumerate() {
            prop_assert!(
                val >= 0.0 && val <= 1.0,
                "softmax[{}] = {} not in [0,1]", i, val
            );
        }
    }

    #[test]
    fn softmax_monotonic(base in vec_f32(2, 16)) {
        let input: Vec<f32> = base.iter().enumerate()
            .map(|(i, &v)| v + i as f32 * 0.001)
            .collect();

        let output = ref_softmax(&input);

        let mut indexed: Vec<(usize, f32)> = input.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        for window in indexed.windows(2) {
            let (i1, _) = window[0];
            let (i2, _) = window[1];
            prop_assert!(
                output[i1] <= output[i2] + 1e-7,
                "monotonicity: softmax[{}]={} > softmax[{}]={}",
                i1, output[i1], i2, output[i2]
            );
        }
    }
}

// ============================================================================
// 5. Attention: uniform weights for identical Q=K, causal mask zeros future
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn attention_identical_qk_uniform_weights(
        seq_len in 1usize..=8,
        head_dim in (1usize..=8).prop_map(|d| d * 2),
    ) {
        let qk: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i % head_dim) as f32) * 0.1)
            .collect();
        let v: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| (i as f32) * 0.01)
            .collect();

        let output = ref_attention(&qk, &qk, &v, seq_len, head_dim, false);
        prop_assert_eq!(output.len(), seq_len * head_dim);

        if seq_len > 0 {
            let v_mean: Vec<f32> = (0..head_dim)
                .map(|d| {
                    (0..seq_len).map(|s| v[s * head_dim + d]).sum::<f32>() / seq_len as f32
                })
                .collect();

            for i in 0..seq_len {
                for d in 0..head_dim {
                    let got = output[i * head_dim + d];
                    let expected = v_mean[d];
                    prop_assert!(
                        (got - expected).abs() < 1e-4,
                        "uniform attention mismatch at [{},{}]: {} != {}",
                        i, d, got, expected
                    );
                }
            }
        }
    }

    #[test]
    fn attention_causal_mask_zeros_future(
        seq_len in 2usize..=8,
        head_dim in (1usize..=4).prop_map(|d| d * 2),
    ) {
        let q: Vec<f32> = (0..seq_len * head_dim).map(|_| 0.5).collect();
        let k: Vec<f32> = (0..seq_len * head_dim).map(|_| 0.5).collect();
        let v: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| {
                let row = i / head_dim;
                (row + 1) as f32
            })
            .collect();

        let output = ref_attention(&q, &k, &v, seq_len, head_dim, true);

        // Position 0 attends only to itself → output row 0 == V row 0
        for d in 0..head_dim {
            let got = output[d];
            let expected = v[d];
            prop_assert!(
                (got - expected).abs() < 1e-4,
                "causal pos 0 should only see itself: {} != {}", got, expected
            );
        }
    }

    #[test]
    fn attention_output_is_finite(
        seq_len in 1usize..=8,
        head_dim in (1usize..=4).prop_map(|d| d * 2),
    ) {
        let data: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32) * 0.1).collect();
        let output = ref_attention(&data, &data, &data, seq_len, head_dim, false);
        for &val in &output {
            prop_assert!(val.is_finite(), "attention output must be finite");
        }
    }
}

// ============================================================================
// 6. SiLU: silu(0)=0, silu(x)>0 for x>0, continuous
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn silu_zero_maps_to_zero(_dummy in 0..1i32) {
        let output = ref_silu(&[0.0]);
        prop_assert!(
            output[0].abs() < 1e-7,
            "silu(0) should be 0, got {}", output[0]
        );
    }

    #[test]
    fn silu_positive_for_positive_input(x in 0.01f32..100.0) {
        let output = ref_silu(&[x]);
        prop_assert!(
            output[0] > 0.0,
            "silu({}) should be positive, got {}", x, output[0]
        );
    }

    #[test]
    fn silu_continuity(x in -10.0f32..10.0) {
        let delta = 1e-4f32;
        let y1 = ref_silu(&[x])[0];
        let y2 = ref_silu(&[x + delta])[0];
        prop_assert!(
            (y2 - y1).abs() < 1.0,
            "silu not continuous: silu({})={}, silu({})={}", x, y1, x + delta, y2
        );
    }

    #[test]
    fn silu_output_is_finite(input in vec_f32(1, 64)) {
        let output = ref_silu(&input);
        for (i, &val) in output.iter().enumerate() {
            prop_assert!(val.is_finite(), "silu output[{}] is not finite: {}", i, val);
        }
    }
}

// ============================================================================
// 7. Embedding: lookup returns correct row, OOB handled
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn embedding_lookup_returns_correct_row(
        vocab_size in 2usize..=16,
        dim in 2usize..=16,
        idx in 0usize..16,
    ) {
        let idx = idx % vocab_size;
        let table: Vec<f32> = (0..vocab_size * dim)
            .map(|i| i as f32 * 0.01)
            .collect();

        let result = ref_embedding_lookup(&table, &[idx], dim).unwrap();
        let expected = &table[idx * dim..(idx + 1) * dim];
        prop_assert_eq!(&result, expected, "embedding lookup returned wrong row");
    }

    #[test]
    fn embedding_oob_returns_none(
        vocab_size in 1usize..=8,
        dim in 1usize..=8,
    ) {
        let table: Vec<f32> = vec![0.0; vocab_size * dim];
        let result = ref_embedding_lookup(&table, &[vocab_size], dim);
        prop_assert!(result.is_none(), "OOB index should return None");
    }

    #[test]
    fn embedding_multiple_lookups(
        vocab_size in 2usize..=8,
        dim in 2usize..=8,
        count in 1usize..=8,
    ) {
        let table: Vec<f32> = (0..vocab_size * dim).map(|i| i as f32).collect();
        let indices: Vec<usize> = (0..count).map(|i| i % vocab_size).collect();
        let result = ref_embedding_lookup(&table, &indices, dim).unwrap();
        prop_assert_eq!(result.len(), count * dim);
    }
}

// ============================================================================
// 8. Ternary pack/unpack: round-trip preserves all values
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn ternary_extended_roundtrip(
        values in prop::collection::vec(
            prop::sample::select(vec![-1i8, 0, 1]), 1..256
        )
    ) {
        let packed = pack_ternary(&values);
        let unpacked = unpack_ternary(&packed, values.len());
        prop_assert_eq!(&values, &unpacked, "round-trip failed");
    }

    #[test]
    fn ternary_packed_size_is_ceil_div_4(count in 1usize..512) {
        let values: Vec<i8> = (0..count).map(|i| [0, 1, -1][i % 3]).collect();
        let packed = pack_ternary(&values);
        prop_assert_eq!(packed.len(), (count + 3) / 4);
    }

    #[test]
    fn ternary_unpack_values_always_valid(data in prop::collection::vec(0u8..=255, 1..128)) {
        let unpacked = unpack_ternary(&data, data.len() * 4);
        for &v in &unpacked {
            prop_assert!(
                v == -1 || v == 0 || v == 1,
                "unpacked value {} is not ternary", v
            );
        }
    }

    #[test]
    fn ternary_pack_is_deterministic(
        values in prop::collection::vec(
            prop::sample::select(vec![-1i8, 0, 1]), 1..64
        )
    ) {
        let packed1 = pack_ternary(&values);
        let packed2 = pack_ternary(&values);
        prop_assert_eq!(packed1, packed2, "packing must be deterministic");
    }
}

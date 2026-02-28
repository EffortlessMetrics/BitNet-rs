// Golden output regression tests for OpenCL kernel reference implementations.
//
// Each test computes output for a fixed input and asserts the exact expected
// result (within floating-point tolerance). These tests catch regressions if
// the kernel logic changes.

const TOL: f32 = 1e-6;

fn assert_approx_eq(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{}: length mismatch ({} vs {})",
        label,
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < TOL,
            "{} mismatch at index {}: actual={}, expected={} (diff={})",
            label,
            i,
            a,
            e,
            (a - e).abs()
        );
    }
}

fn ref_matmul_i2s(a_packed: &[u8], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for i in 0..k {
                let byte_idx = (row * k + i) / 4;
                let sub = (row * k + i) % 4;
                let bits = (a_packed[byte_idx] >> (sub * 2)) & 0x03;
                let w: f32 = match bits {
                    0x01 => 1.0,
                    0x03 => -1.0,
                    _ => 0.0,
                };
                sum += w * b[i * n + col];
            }
            c[row * n + col] = sum;
        }
    }
    c
}

fn ref_rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len();
    let sum_sq: f32 = input.iter().map(|x| x * x).sum();
    let rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();
    input
        .iter()
        .zip(weight.iter())
        .map(|(&x, &w)| x * rms * w)
        .collect()
}

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

fn ref_softmax(input: &[f32]) -> Vec<f32> {
    let max_val = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = input.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

fn ref_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; seq_len * head_dim];
    for i in 0..seq_len {
        let mut scores = vec![0.0f32; seq_len];
        for j in 0..seq_len {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[i * head_dim + d] * k[j * head_dim + d];
            }
            scores[j] = dot * scale;
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

fn ref_silu(input: &[f32]) -> Vec<f32> {
    input
        .iter()
        .map(|&x| {
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            x * sigmoid
        })
        .collect()
}

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

// === 1. Matmul golden tests ===

#[test]
fn golden_matmul_4x4() {
    let a_vals: Vec<i8> = vec![1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    let a_packed = pack_ternary(&a_vals);
    #[rustfmt::skip]
    let b = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    let c = ref_matmul_i2s(&a_packed, &b, 4, 4, 4);
    #[rustfmt::skip]
    let expected = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    assert_approx_eq(&c, &expected, "matmul identity");
}

#[test]
fn golden_matmul_negation() {
    let a_vals: Vec<i8> = vec![-1, -1, -1, -1, -1, -1, -1, -1];
    let a_packed = pack_ternary(&a_vals);
    #[rustfmt::skip]
    let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let c = ref_matmul_i2s(&a_packed, &b, 2, 2, 4);
    let expected = vec![-16.0, -20.0, -16.0, -20.0];
    assert_approx_eq(&c, &expected, "matmul negation");
}

// === 2. RMSNorm golden tests ===

#[test]
fn golden_rmsnorm() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0];
    let eps = 1e-5;
    let output = ref_rms_norm(&input, &weight, eps);
    let rms = (30.0f32 / 4.0 + eps).sqrt();
    let scale = 1.0 / rms;
    let expected = vec![1.0 * scale, 2.0 * scale, 3.0 * scale, 4.0 * scale];
    assert_approx_eq(&output, &expected, "rmsnorm [1,2,3,4]");
}

#[test]
fn golden_rmsnorm_with_weights() {
    let input = vec![2.0, 4.0];
    let weight = vec![0.5, 2.0];
    let eps = 1e-5;
    let output = ref_rms_norm(&input, &weight, eps);
    let rms = ((4.0 + 16.0) / 2.0 + eps).sqrt();
    let scale = 1.0 / rms;
    let expected = vec![2.0 * scale * 0.5, 4.0 * scale * 2.0];
    assert_approx_eq(&output, &expected, "rmsnorm with weights");
}

// === 3. RoPE golden tests ===

#[test]
fn golden_rope_position_0() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let output = ref_rope(&input, 0, 4);
    assert_approx_eq(&output, &input, "rope position 0 (identity)");
}

#[test]
fn golden_rope_position_1() {
    let input = vec![1.0, 0.0, 0.0, 1.0];
    let output = ref_rope(&input, 1, 4);
    let cos0 = 1.0f64.cos() as f32;
    let sin0 = 1.0f64.sin() as f32;
    let cos1 = 0.01f64.cos() as f32;
    let sin1 = 0.01f64.sin() as f32;
    let expected = vec![
        1.0 * cos0 - 0.0 * sin0,
        1.0 * sin0 + 0.0 * cos0,
        0.0 * cos1 - 1.0 * sin1,
        0.0 * sin1 + 1.0 * cos1,
    ];
    assert_approx_eq(&output, &expected, "rope position 1");
}

// === 4. Softmax golden tests ===

#[test]
fn golden_softmax() {
    let input = vec![1.0, 2.0, 3.0];
    let output = ref_softmax(&input);
    let e0 = (-2.0f32).exp();
    let e1 = (-1.0f32).exp();
    let e2 = (0.0f32).exp();
    let sum = e0 + e1 + e2;
    let expected = vec![e0 / sum, e1 / sum, e2 / sum];
    assert_approx_eq(&output, &expected, "softmax [1,2,3]");
}

#[test]
fn golden_softmax_uniform() {
    let input = vec![5.0, 5.0, 5.0, 5.0];
    let output = ref_softmax(&input);
    let expected = vec![0.25, 0.25, 0.25, 0.25];
    assert_approx_eq(&output, &expected, "softmax uniform");
}

// === 5. Attention golden tests ===

#[test]
fn golden_attention_2_token() {
    let head_dim = 2;
    let seq_len = 2;
    let q = vec![1.0, 0.0, 0.0, 1.0];
    let k = vec![1.0, 0.0, 0.0, 1.0];
    let v = vec![1.0, 2.0, 3.0, 4.0];
    let output = ref_attention(&q, &k, &v, seq_len, head_dim);
    let scale = 1.0 / (2.0f32).sqrt();
    let w0 = ref_softmax(&[1.0 * scale, 0.0 * scale]);
    let out0_0 = w0[0] * 1.0 + w0[1] * 3.0;
    let out0_1 = w0[0] * 2.0 + w0[1] * 4.0;
    let w1 = ref_softmax(&[0.0 * scale, 1.0 * scale]);
    let out1_0 = w1[0] * 1.0 + w1[1] * 3.0;
    let out1_1 = w1[0] * 2.0 + w1[1] * 4.0;
    let expected = vec![out0_0, out0_1, out1_0, out1_1];
    assert_approx_eq(&output, &expected, "attention 2-token");
}

#[test]
fn golden_attention_uniform() {
    let qk = vec![1.0, 1.0, 1.0, 1.0];
    let v = vec![2.0, 4.0, 6.0, 8.0];
    let output = ref_attention(&qk, &qk, &v, 2, 2);
    let expected = vec![4.0, 6.0, 4.0, 6.0];
    assert_approx_eq(&output, &expected, "attention uniform");
}

// === 6. SiLU golden test ===

#[test]
fn golden_silu() {
    let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let output = ref_silu(&input);
    let expected: Vec<f32> = input
        .iter()
        .map(|&x| {
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            x * sigmoid
        })
        .collect();
    assert_approx_eq(&output, &expected, "silu [-2,-1,0,1,2]");
    assert!((output[2]).abs() < 1e-7, "silu(0) must be 0");
    assert!(output[3] > 0.73 && output[3] < 0.74);
    assert!(output[4] > 1.76 && output[4] < 1.77);
}

// === 7. Ternary pack golden tests ===

#[test]
fn golden_ternary_pack() {
    let values: Vec<i8> = vec![0, 1, -1, 0, 1, -1, 1, 0];
    let packed = pack_ternary(&values);
    assert_eq!(packed, vec![0x34u8, 0x1Du8]);
}

#[test]
fn golden_ternary_pack_all_patterns() {
    assert_eq!(pack_ternary(&[1, 1, 1, 1]), vec![0x55]);
    assert_eq!(pack_ternary(&[-1, -1, -1, -1]), vec![0xFF]);
    assert_eq!(pack_ternary(&[0, 0, 0, 0]), vec![0x00]);
    assert_eq!(pack_ternary(&[1]), vec![0x01]);
    assert_eq!(pack_ternary(&[-1]), vec![0x03]);
    assert_eq!(pack_ternary(&[0]), vec![0x00]);
}

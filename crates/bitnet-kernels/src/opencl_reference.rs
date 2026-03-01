//! CPU reference implementations of OpenCL kernels for validation.
//!
//! These functions implement the exact same algorithms as the .cl kernel
//! sources, providing ground truth for correctness testing.

/// CPU reference: naive matmul C = alpha * A @ B + beta * C
pub fn matmul_naive_ref(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = alpha * sum + beta * c[i * n + j];
        }
    }
}

/// CPU reference: batched matmul
pub fn matmul_batched_ref(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    batch_count: usize,
) {
    let stride_a = m * k;
    let stride_b = k * n;
    let stride_c = m * n;
    for batch in 0..batch_count {
        let a_off = batch * stride_a;
        let b_off = batch * stride_b;
        let c_off = batch * stride_c;
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a[a_off + i * k + kk] * b[b_off + kk * n + j];
                }
                c[c_off + i * n + j] = sum;
            }
        }
    }
}

/// CPU reference: softmax (row-wise)
pub fn softmax_row_ref(input: &[f32], output: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let row_start = r * cols;
        let mut max_val = input[row_start];
        for j in 1..cols {
            max_val = max_val.max(input[row_start + j]);
        }
        let mut sum = 0.0f32;
        for j in 0..cols {
            let e = (input[row_start + j] - max_val).exp();
            output[row_start + j] = e;
            sum += e;
        }
        let inv_sum = 1.0 / sum;
        for j in 0..cols {
            output[row_start + j] *= inv_sum;
        }
    }
}

/// CPU reference: softmax with temperature
pub fn softmax_temperature_ref(
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    temperature: f32,
) {
    let inv_temp = 1.0 / temperature;
    for r in 0..rows {
        let row_start = r * cols;
        let mut max_val = input[row_start] * inv_temp;
        for j in 1..cols {
            max_val = max_val.max(input[row_start + j] * inv_temp);
        }
        let mut sum = 0.0f32;
        for j in 0..cols {
            let e = (input[row_start + j] * inv_temp - max_val).exp();
            output[row_start + j] = e;
            sum += e;
        }
        let inv_sum = 1.0 / sum;
        for j in 0..cols {
            output[row_start + j] *= inv_sum;
        }
    }
}

/// CPU reference: layer norm
pub fn layer_norm_ref(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    output: &mut [f32],
    rows: usize,
    hidden_size: usize,
    eps: f32,
) {
    for r in 0..rows {
        let base = r * hidden_size;
        let mut mean = 0.0f32;
        for i in 0..hidden_size {
            mean += input[base + i];
        }
        mean /= hidden_size as f32;
        let mut var = 0.0f32;
        for i in 0..hidden_size {
            let diff = input[base + i] - mean;
            var += diff * diff;
        }
        var /= hidden_size as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for i in 0..hidden_size {
            output[base + i] = (input[base + i] - mean) * inv_std * gamma[i] + beta[i];
        }
    }
}

/// CPU reference: RMS norm
pub fn rms_norm_ref(
    input: &[f32],
    gamma: &[f32],
    output: &mut [f32],
    rows: usize,
    hidden_size: usize,
    eps: f32,
) {
    for r in 0..rows {
        let base = r * hidden_size;
        let mut sum_sq = 0.0f32;
        for i in 0..hidden_size {
            sum_sq += input[base + i] * input[base + i];
        }
        let rms = (sum_sq / hidden_size as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;
        for i in 0..hidden_size {
            output[base + i] = input[base + i] * inv_rms * gamma[i];
        }
    }
}

/// CPU reference: SiLU activation
pub fn silu_ref(x: &[f32], y: &mut [f32]) {
    for i in 0..x.len() {
        y[i] = x[i] / (1.0 + (-x[i]).exp());
    }
}

/// CPU reference: GELU activation (approximate)
pub fn gelu_ref(x: &[f32], y: &mut [f32]) {
    for i in 0..x.len() {
        let v = x[i];
        let cdf = 0.5 * (1.0 + (0.7978845608_f32 * (v + 0.044715 * v * v * v)).tanh());
        y[i] = v * cdf;
    }
}

/// CPU reference: I2_S dequantization
pub fn dequantize_i2s_ref(packed: &[u8], scales: &[f32], block_size: usize) -> Vec<f32> {
    let mut output = Vec::new();
    for (byte_idx, &byte) in packed.iter().enumerate() {
        for j in 0..4 {
            let val = (byte >> (j * 2)) & 0x03;
            let fval = (val as i32 - 1) as f32;
            let elem_idx = byte_idx * 4 + j;
            let block_idx = elem_idx / block_size;
            let scale = scales.get(block_idx).copied().unwrap_or(1.0);
            output.push(fval * scale);
        }
    }
    output
}

/// CPU reference: vector addition C = A + B
pub fn vec_add_ref(a: &[f32], b: &[f32], c: &mut [f32]) {
    for i in 0..a.len() {
        c[i] = a[i] + b[i];
    }
}

/// CPU reference: ReLU activation
pub fn relu_ref(x: &[f32], y: &mut [f32]) {
    for i in 0..x.len() {
        y[i] = x[i].max(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    fn assert_approx_slice(actual: &[f32], expected: &[f32], tol: f32, msg: &str) {
        assert_eq!(actual.len(), expected.len(), "{msg}: length mismatch");
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                approx_eq(*a, *e, tol),
                "{msg}: index {i}: actual={a}, expected={e}, diff={}",
                (a - e).abs()
            );
        }
    }

    // ================================================================
    // Matmul tests
    // ================================================================

    #[test]
    fn matmul_identity_2x2() {
        let a = [1.0, 0.0, 0.0, 1.0]; // I₂
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut c = [0.0f32; 4];
        matmul_naive_ref(&a, &b, &mut c, 2, 2, 2, 1.0, 0.0);
        assert_approx_slice(&c, &b, EPS, "identity matmul");
    }

    #[test]
    fn matmul_zero_matrix() {
        let a = [0.0f32; 4];
        let b = [1.0, 2.0, 3.0, 4.0];
        let mut c = [0.0f32; 4];
        matmul_naive_ref(&a, &b, &mut c, 2, 2, 2, 1.0, 0.0);
        assert_approx_slice(&c, &[0.0; 4], EPS, "zero matmul");
    }

    #[test]
    fn matmul_known_2x2() {
        // [1, 2; 3, 4] @ [5, 6; 7, 8] = [19, 22; 43, 50]
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut c = [0.0f32; 4];
        matmul_naive_ref(&a, &b, &mut c, 2, 2, 2, 1.0, 0.0);
        assert_approx_slice(&c, &[19.0, 22.0, 43.0, 50.0], EPS, "known 2x2");
    }

    #[test]
    fn matmul_known_2x3_3x2() {
        // [1,2,3; 4,5,6] @ [7,8; 9,10; 11,12] = [58,64; 139,154]
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut c = [0.0f32; 4];
        matmul_naive_ref(&a, &b, &mut c, 2, 2, 3, 1.0, 0.0);
        assert_approx_slice(&c, &[58.0, 64.0, 139.0, 154.0], EPS, "2x3 @ 3x2");
    }

    #[test]
    fn matmul_alpha_scaling() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut c = [0.0f32; 4];
        matmul_naive_ref(&a, &b, &mut c, 2, 2, 2, 2.0, 0.0);
        assert_approx_slice(&c, &[38.0, 44.0, 86.0, 100.0], EPS, "alpha=2");
    }

    #[test]
    fn matmul_beta_accumulation() {
        let a = [1.0, 0.0, 0.0, 1.0]; // I₂
        let b = [1.0, 0.0, 0.0, 1.0]; // I₂
        let mut c = [10.0, 20.0, 30.0, 40.0];
        matmul_naive_ref(&a, &b, &mut c, 2, 2, 2, 1.0, 1.0);
        // C = 1*I + 1*[10,20;30,40] = [11,20;30,41]
        assert_approx_slice(&c, &[11.0, 20.0, 30.0, 41.0], EPS, "beta=1");
    }

    #[test]
    fn matmul_alpha_zero_beta_one() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut c = [100.0, 200.0, 300.0, 400.0];
        matmul_naive_ref(&a, &b, &mut c, 2, 2, 2, 0.0, 1.0);
        assert_approx_slice(&c, &[100.0, 200.0, 300.0, 400.0], EPS, "alpha=0, beta=1");
    }

    #[test]
    fn matmul_1x1() {
        let a = [3.0];
        let b = [4.0];
        let mut c = [0.0f32; 1];
        matmul_naive_ref(&a, &b, &mut c, 1, 1, 1, 1.0, 0.0);
        assert_approx_slice(&c, &[12.0], EPS, "1x1");
    }

    #[test]
    fn matmul_row_times_col() {
        // [1,2,3] @ [4;5;6] = [32]
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let mut c = [0.0f32; 1];
        matmul_naive_ref(&a, &b, &mut c, 1, 1, 3, 1.0, 0.0);
        assert_approx_slice(&c, &[32.0], EPS, "dot product");
    }

    #[test]
    fn matmul_negative_values() {
        let a = [-1.0, -2.0, -3.0, -4.0];
        let b = [1.0, 2.0, 3.0, 4.0];
        let mut c = [0.0f32; 4];
        matmul_naive_ref(&a, &b, &mut c, 2, 2, 2, 1.0, 0.0);
        assert_approx_slice(&c, &[-7.0, -10.0, -15.0, -22.0], EPS, "negative");
    }

    #[test]
    fn matmul_rectangular_3x1_1x3() {
        // [1;2;3] @ [4,5,6] = 3x3 outer product
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let mut c = [0.0f32; 9];
        matmul_naive_ref(&a, &b, &mut c, 3, 3, 1, 1.0, 0.0);
        let expected = [4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 12.0, 15.0, 18.0];
        assert_approx_slice(&c, &expected, EPS, "outer product");
    }

    #[test]
    fn matmul_combined_alpha_beta() {
        let a = [1.0, 0.0, 0.0, 1.0]; // I₂
        let b = [2.0, 3.0, 4.0, 5.0];
        let mut c = [10.0, 10.0, 10.0, 10.0];
        // C = 3*(I@B) + 2*C_old = 3*[2,3;4,5] + 2*[10,10;10,10]
        matmul_naive_ref(&a, &b, &mut c, 2, 2, 2, 3.0, 2.0);
        assert_approx_slice(&c, &[26.0, 29.0, 32.0, 35.0], EPS, "alpha=3 beta=2");
    }

    // ================================================================
    // Batched matmul tests
    // ================================================================

    #[test]
    fn batched_matmul_single_batch() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut c = [0.0f32; 4];
        matmul_batched_ref(&a, &b, &mut c, 2, 2, 2, 1);
        assert_approx_slice(&c, &[19.0, 22.0, 43.0, 50.0], EPS, "batch=1");
    }

    #[test]
    fn batched_matmul_two_batches() {
        // batch0: I₂ @ [1,2;3,4] = [1,2;3,4]
        // batch1: [2,0;0,2] @ [1,1;1,1] = [2,2;2,2]
        let a = [1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0];
        let b = [1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0];
        let mut c = [0.0f32; 8];
        matmul_batched_ref(&a, &b, &mut c, 2, 2, 2, 2);
        assert_approx_slice(&c, &[1.0, 2.0, 3.0, 4.0, 2.0, 2.0, 2.0, 2.0], EPS, "batch=2");
    }

    #[test]
    fn batched_matmul_zero_batch_is_noop() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut c = [99.0f32; 4];
        matmul_batched_ref(&a, &b, &mut c, 2, 2, 2, 0);
        // c should remain untouched
        assert_approx_slice(&c, &[99.0; 4], EPS, "batch=0 noop");
    }

    // ================================================================
    // Softmax tests
    // ================================================================

    #[test]
    fn softmax_single_element() {
        let input = [42.0];
        let mut output = [0.0f32; 1];
        softmax_row_ref(&input, &mut output, 1, 1);
        assert_approx_slice(&output, &[1.0], EPS, "softmax single");
    }

    #[test]
    fn softmax_uniform_input() {
        let input = [1.0, 1.0, 1.0, 1.0];
        let mut output = [0.0f32; 4];
        softmax_row_ref(&input, &mut output, 1, 4);
        assert_approx_slice(&output, &[0.25, 0.25, 0.25, 0.25], EPS, "softmax uniform");
    }

    #[test]
    fn softmax_row_sums_to_one() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut output = [0.0f32; 5];
        softmax_row_ref(&input, &mut output, 1, 5);
        let sum: f32 = output.iter().sum();
        assert!(approx_eq(sum, 1.0, EPS), "sum={sum}");
    }

    #[test]
    fn softmax_monotonicity() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = [0.0f32; 4];
        softmax_row_ref(&input, &mut output, 1, 4);
        for i in 0..3 {
            assert!(
                output[i] < output[i + 1],
                "softmax should preserve order: output[{}]={} >= output[{}]={}",
                i,
                output[i],
                i + 1,
                output[i + 1]
            );
        }
    }

    #[test]
    fn softmax_all_outputs_nonnegative() {
        let input = [-10.0, -5.0, 0.0, 5.0, 10.0];
        let mut output = [0.0f32; 5];
        softmax_row_ref(&input, &mut output, 1, 5);
        for (i, &v) in output.iter().enumerate() {
            assert!(v >= 0.0, "softmax output[{i}]={v} should be >= 0");
            assert!(v.is_finite(), "softmax output[{i}] should be finite");
        }
        // Elements with reasonable range should be strictly positive
        assert!(output[2] > 0.0, "softmax(0.0) should be > 0");
        assert!(output[4] > 0.0, "softmax(max) should be > 0");
    }

    #[test]
    fn softmax_multi_row() {
        let input = [1.0, 2.0, 3.0, 4.0, 10.0, 10.0, 10.0, 10.0];
        let mut output = [0.0f32; 8];
        softmax_row_ref(&input, &mut output, 2, 4);
        let sum0: f32 = output[..4].iter().sum();
        let sum1: f32 = output[4..].iter().sum();
        assert!(approx_eq(sum0, 1.0, EPS), "row 0 sum={sum0}");
        assert!(approx_eq(sum1, 1.0, EPS), "row 1 sum={sum1}");
        // Row 1 is uniform
        assert_approx_slice(&output[4..], &[0.25, 0.25, 0.25, 0.25], EPS, "row1 uniform");
    }

    #[test]
    fn softmax_large_values_no_overflow() {
        let input = [1000.0, 1001.0, 1002.0];
        let mut output = [0.0f32; 3];
        softmax_row_ref(&input, &mut output, 1, 3);
        let sum: f32 = output.iter().sum();
        assert!(approx_eq(sum, 1.0, EPS), "sum={sum} (large values)");
        for v in &output {
            assert!(v.is_finite(), "overflow check");
        }
    }

    #[test]
    fn softmax_negative_values() {
        let input = [-3.0, -2.0, -1.0];
        let mut output = [0.0f32; 3];
        softmax_row_ref(&input, &mut output, 1, 3);
        let sum: f32 = output.iter().sum();
        assert!(approx_eq(sum, 1.0, EPS), "sum={sum} (negative values)");
    }

    #[test]
    fn softmax_known_values() {
        // softmax([0, 0]) = [0.5, 0.5]
        let input = [0.0, 0.0];
        let mut output = [0.0f32; 2];
        softmax_row_ref(&input, &mut output, 1, 2);
        assert_approx_slice(&output, &[0.5, 0.5], EPS, "softmax [0,0]");
    }

    // ================================================================
    // Softmax with temperature tests
    // ================================================================

    #[test]
    fn softmax_temperature_1_equals_normal() {
        let input = [1.0, 2.0, 3.0];
        let mut out_normal = [0.0f32; 3];
        let mut out_temp = [0.0f32; 3];
        softmax_row_ref(&input, &mut out_normal, 1, 3);
        softmax_temperature_ref(&input, &mut out_temp, 1, 3, 1.0);
        assert_approx_slice(&out_temp, &out_normal, EPS, "temp=1");
    }

    #[test]
    fn softmax_high_temperature_flattens() {
        let input = [1.0, 5.0];
        let mut out_low = [0.0f32; 2];
        let mut out_high = [0.0f32; 2];
        softmax_temperature_ref(&input, &mut out_low, 1, 2, 0.1);
        softmax_temperature_ref(&input, &mut out_high, 1, 2, 100.0);
        // High temperature should make distribution more uniform
        let diff_low = (out_low[0] - out_low[1]).abs();
        let diff_high = (out_high[0] - out_high[1]).abs();
        assert!(
            diff_high < diff_low,
            "high temp diff={diff_high} should be < low temp diff={diff_low}"
        );
    }

    #[test]
    fn softmax_low_temperature_sharpens() {
        let input = [1.0, 2.0, 3.0];
        let mut output = [0.0f32; 3];
        softmax_temperature_ref(&input, &mut output, 1, 3, 0.01);
        // Very low temperature: max element gets almost all probability
        assert!(output[2] > 0.99, "max should dominate: {}", output[2]);
    }

    #[test]
    fn softmax_temperature_sums_to_one() {
        let input = [0.5, 1.5, -0.5, 2.0];
        let mut output = [0.0f32; 4];
        softmax_temperature_ref(&input, &mut output, 1, 4, 0.5);
        let sum: f32 = output.iter().sum();
        assert!(approx_eq(sum, 1.0, EPS), "temp softmax sum={sum}");
    }

    // ================================================================
    // Layer norm tests
    // ================================================================

    #[test]
    fn layer_norm_zero_mean_output() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let gamma = [1.0; 4];
        let beta = [0.0; 4];
        let mut output = [0.0f32; 4];
        layer_norm_ref(&input, &gamma, &beta, &mut output, 1, 4, 1e-5);
        let mean: f32 = output.iter().sum::<f32>() / 4.0;
        assert!(approx_eq(mean, 0.0, 1e-4), "output mean={mean} should be ~0");
    }

    #[test]
    fn layer_norm_unit_variance() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let gamma = [1.0; 4];
        let beta = [0.0; 4];
        let mut output = [0.0f32; 4];
        layer_norm_ref(&input, &gamma, &beta, &mut output, 1, 4, 1e-5);
        let mean: f32 = output.iter().sum::<f32>() / 4.0;
        let var: f32 = output.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / 4.0;
        assert!(approx_eq(var, 1.0, 1e-3), "output var={var} should be ~1");
    }

    #[test]
    fn layer_norm_constant_input() {
        let input = [5.0; 4];
        let gamma = [1.0; 4];
        let beta = [0.0; 4];
        let mut output = [0.0f32; 4];
        layer_norm_ref(&input, &gamma, &beta, &mut output, 1, 4, 1e-5);
        // Constant input → zero after normalization (zero variance → eps prevents div-by-zero)
        for (i, &v) in output.iter().enumerate() {
            assert!(v.abs() < 0.01, "constant input: output[{i}]={v}");
        }
    }

    #[test]
    fn layer_norm_gamma_scaling() {
        let input = [0.0, 1.0, 2.0, 3.0];
        let gamma = [2.0; 4];
        let beta = [0.0; 4];
        let mut out_g1 = [0.0f32; 4];
        let mut out_g2 = [0.0f32; 4];
        layer_norm_ref(&input, &[1.0; 4], &beta, &mut out_g1, 1, 4, 1e-5);
        layer_norm_ref(&input, &gamma, &beta, &mut out_g2, 1, 4, 1e-5);
        for i in 0..4 {
            assert!(
                approx_eq(out_g2[i], 2.0 * out_g1[i], 1e-4),
                "gamma scaling: [{i}] {} vs {}",
                out_g2[i],
                2.0 * out_g1[i]
            );
        }
    }

    #[test]
    fn layer_norm_beta_offset() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let gamma = [1.0; 4];
        let beta = [10.0; 4];
        let mut out_b0 = [0.0f32; 4];
        let mut out_b10 = [0.0f32; 4];
        layer_norm_ref(&input, &gamma, &[0.0; 4], &mut out_b0, 1, 4, 1e-5);
        layer_norm_ref(&input, &gamma, &beta, &mut out_b10, 1, 4, 1e-5);
        for i in 0..4 {
            assert!(
                approx_eq(out_b10[i], out_b0[i] + 10.0, 1e-4),
                "beta offset: [{i}] {} vs {}",
                out_b10[i],
                out_b0[i] + 10.0
            );
        }
    }

    #[test]
    fn layer_norm_multi_row() {
        let input = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let gamma = [1.0; 4];
        let beta = [0.0; 4];
        let mut output = [0.0f32; 8];
        layer_norm_ref(&input, &gamma, &beta, &mut output, 2, 4, 1e-5);
        // Each row should have zero mean
        let mean0: f32 = output[..4].iter().sum::<f32>() / 4.0;
        let mean1: f32 = output[4..].iter().sum::<f32>() / 4.0;
        assert!(approx_eq(mean0, 0.0, 1e-4), "row 0 mean={mean0}");
        assert!(approx_eq(mean1, 0.0, 1e-4), "row 1 mean={mean1}");
    }

    #[test]
    fn layer_norm_known_two_elements() {
        // input=[0, 2], mean=1, var=1, inv_std=1
        // normalized = [-1, 1] with gamma=1, beta=0
        let input = [0.0, 2.0];
        let gamma = [1.0; 2];
        let beta = [0.0; 2];
        let mut output = [0.0f32; 2];
        layer_norm_ref(&input, &gamma, &beta, &mut output, 1, 2, 0.0);
        assert_approx_slice(&output, &[-1.0, 1.0], EPS, "layer norm [0,2]");
    }

    // ================================================================
    // RMS norm tests
    // ================================================================

    #[test]
    fn rms_norm_known_values() {
        // input=[3, 4], rms=sqrt((9+16)/2)=sqrt(12.5)=3.5355
        // output = [3/3.5355, 4/3.5355] * gamma
        let input = [3.0, 4.0];
        let gamma = [1.0; 2];
        let mut output = [0.0f32; 2];
        rms_norm_ref(&input, &gamma, &mut output, 1, 2, 0.0);
        let rms = (12.5f32).sqrt();
        assert_approx_slice(&output, &[3.0 / rms, 4.0 / rms], 1e-4, "rms norm [3,4]");
    }

    #[test]
    fn rms_norm_unit_input() {
        // input=[1,1,1,1], rms=sqrt(4/4)=1, output=[1,1,1,1]*gamma
        let input = [1.0; 4];
        let gamma = [1.0; 4];
        let mut output = [0.0f32; 4];
        rms_norm_ref(&input, &gamma, &mut output, 1, 4, 0.0);
        assert_approx_slice(&output, &[1.0; 4], 1e-4, "rms norm ones");
    }

    #[test]
    fn rms_norm_gamma_scaling() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut out_g1 = [0.0f32; 4];
        let mut out_g3 = [0.0f32; 4];
        rms_norm_ref(&input, &[1.0; 4], &mut out_g1, 1, 4, 1e-5);
        rms_norm_ref(&input, &[3.0; 4], &mut out_g3, 1, 4, 1e-5);
        for i in 0..4 {
            assert!(
                approx_eq(out_g3[i], 3.0 * out_g1[i], 1e-4),
                "rms gamma: [{i}] {} vs {}",
                out_g3[i],
                3.0 * out_g1[i]
            );
        }
    }

    #[test]
    fn rms_norm_multi_row() {
        let input = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0];
        let gamma = [1.0; 4];
        let mut output = [0.0f32; 8];
        rms_norm_ref(&input, &gamma, &mut output, 2, 4, 1e-5);
        // Row 0: rms=sqrt(1/4)=0.5, output=[2, 0, 0, 0]
        let rms0 = (0.25f32 + 1e-5).sqrt();
        assert!(approx_eq(output[0], 1.0 / rms0, 1e-3), "row0[0]={}", output[0]);
    }

    #[test]
    fn rms_norm_zero_input_with_eps() {
        let input = [0.0; 4];
        let gamma = [1.0; 4];
        let mut output = [0.0f32; 4];
        rms_norm_ref(&input, &gamma, &mut output, 1, 4, 1e-5);
        // All zeros normalized should still be zero (0 * inv_rms = 0)
        assert_approx_slice(&output, &[0.0; 4], EPS, "rms norm zeros");
    }

    #[test]
    fn rms_norm_preserves_sign() {
        let input = [-2.0, 3.0, -4.0, 5.0];
        let gamma = [1.0; 4];
        let mut output = [0.0f32; 4];
        rms_norm_ref(&input, &gamma, &mut output, 1, 4, 1e-5);
        assert!(output[0] < 0.0, "negative preserved");
        assert!(output[1] > 0.0, "positive preserved");
        assert!(output[2] < 0.0, "negative preserved");
        assert!(output[3] > 0.0, "positive preserved");
    }

    // ================================================================
    // SiLU activation tests
    // ================================================================

    #[test]
    fn silu_zero() {
        let x = [0.0];
        let mut y = [0.0f32; 1];
        silu_ref(&x, &mut y);
        assert_approx_slice(&y, &[0.0], EPS, "silu(0)=0");
    }

    #[test]
    fn silu_positive_large() {
        let x = [10.0];
        let mut y = [0.0f32; 1];
        silu_ref(&x, &mut y);
        // silu(x) ≈ x for large positive x
        assert!(approx_eq(y[0], 10.0, 0.001), "silu(10) ≈ 10, got {}", y[0]);
    }

    #[test]
    fn silu_negative_large() {
        let x = [-10.0];
        let mut y = [0.0f32; 1];
        silu_ref(&x, &mut y);
        // silu(x) ≈ 0 for large negative x
        assert!(y[0].abs() < 0.001, "silu(-10) ≈ 0, got {}", y[0]);
    }

    #[test]
    fn silu_known_value() {
        // silu(1) = 1 / (1 + exp(-1)) = 1/(1+0.3679) = 0.7311
        let x = [1.0];
        let mut y = [0.0f32; 1];
        silu_ref(&x, &mut y);
        assert!(approx_eq(y[0], 0.7311, 0.001), "silu(1) ≈ 0.7311, got {}", y[0]);
    }

    #[test]
    fn silu_monotonic() {
        let x: Vec<f32> = (-20..=20).map(|i| i as f32 * 0.5).collect();
        let mut y = vec![0.0f32; x.len()];
        silu_ref(&x, &mut y);
        // SiLU is monotonically increasing for x > ~-0.278, but across
        // the full range from -10 to 10, each output should be ≤ the next
        // for inputs spaced far enough apart
        for i in 1..y.len() {
            if x[i] > 0.0 {
                assert!(y[i] >= y[i - 1], "silu monotonic at x={}: {} < {}", x[i], y[i], y[i - 1]);
            }
        }
    }

    #[test]
    fn silu_batch() {
        let x = [-1.0, 0.0, 1.0, 2.0];
        let mut y = [0.0f32; 4];
        silu_ref(&x, &mut y);
        assert!(y[0] < 0.0, "silu(-1) < 0");
        assert_approx_slice(&y[1..2], &[0.0], EPS, "silu(0) = 0");
        assert!(y[2] > 0.0, "silu(1) > 0");
        assert!(y[3] > y[2], "silu(2) > silu(1)");
    }

    // ================================================================
    // GELU activation tests
    // ================================================================

    #[test]
    fn gelu_zero() {
        let x = [0.0];
        let mut y = [0.0f32; 1];
        gelu_ref(&x, &mut y);
        assert_approx_slice(&y, &[0.0], EPS, "gelu(0)=0");
    }

    #[test]
    fn gelu_positive_large() {
        let x = [10.0];
        let mut y = [0.0f32; 1];
        gelu_ref(&x, &mut y);
        assert!(approx_eq(y[0], 10.0, 0.01), "gelu(10) ≈ 10, got {}", y[0]);
    }

    #[test]
    fn gelu_negative_large() {
        let x = [-10.0];
        let mut y = [0.0f32; 1];
        gelu_ref(&x, &mut y);
        assert!(y[0].abs() < 0.01, "gelu(-10) ≈ 0, got {}", y[0]);
    }

    #[test]
    fn gelu_known_value() {
        // gelu(1) ≈ 0.8412 (approximate form)
        let x = [1.0];
        let mut y = [0.0f32; 1];
        gelu_ref(&x, &mut y);
        assert!(approx_eq(y[0], 0.8412, 0.002), "gelu(1) ≈ 0.8412, got {}", y[0]);
    }

    #[test]
    fn gelu_monotonic_positive() {
        let x: Vec<f32> = (0..=20).map(|i| i as f32 * 0.5).collect();
        let mut y = vec![0.0f32; x.len()];
        gelu_ref(&x, &mut y);
        for i in 1..y.len() {
            assert!(y[i] >= y[i - 1], "gelu monotonic at x={}: {} < {}", x[i], y[i], y[i - 1]);
        }
    }

    #[test]
    fn gelu_symmetry_near_zero() {
        // gelu(-x) ≈ -gelu(x) only holds approximately near zero
        let x_pos = [0.5];
        let x_neg = [-0.5];
        let mut y_pos = [0.0f32; 1];
        let mut y_neg = [0.0f32; 1];
        gelu_ref(&x_pos, &mut y_pos);
        gelu_ref(&x_neg, &mut y_neg);
        // Not exact antisymmetry, but gelu(-x) + gelu(x) ≈ 0 doesn't hold,
        // instead gelu(x) + gelu(-x) = x (approximately for large |x|)
        assert!(y_neg[0] < 0.0, "gelu(-0.5) should be negative");
        assert!(y_pos[0] > 0.0, "gelu(0.5) should be positive");
    }

    #[test]
    fn gelu_batch() {
        let x = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut y = [0.0f32; 5];
        gelu_ref(&x, &mut y);
        assert!(y[0] < 0.0, "gelu(-2) < 0");
        assert!(y[1] < 0.0, "gelu(-1) < 0");
        assert_approx_slice(&y[2..3], &[0.0], EPS, "gelu(0) = 0");
        assert!(y[3] > 0.0, "gelu(1) > 0");
        assert!(y[4] > y[3], "gelu(2) > gelu(1)");
    }

    // ================================================================
    // I2_S dequantization tests
    // ================================================================

    #[test]
    fn dequant_i2s_zero_packed() {
        // 0x00 → four values of (0-1) = -1
        let packed = [0x00];
        let scales = [1.0];
        let result = dequantize_i2s_ref(&packed, &scales, 4);
        assert_approx_slice(&result, &[-1.0, -1.0, -1.0, -1.0], EPS, "deq 0x00");
    }

    #[test]
    fn dequant_i2s_all_ones() {
        // 0x55 = 0b01_01_01_01 → four values of (1-1) = 0
        let packed = [0x55];
        let scales = [1.0];
        let result = dequantize_i2s_ref(&packed, &scales, 4);
        assert_approx_slice(&result, &[0.0, 0.0, 0.0, 0.0], EPS, "deq 0x55");
    }

    #[test]
    fn dequant_i2s_all_twos() {
        // 0xAA = 0b10_10_10_10 → four values of (2-1) = 1
        let packed = [0xAA];
        let scales = [1.0];
        let result = dequantize_i2s_ref(&packed, &scales, 4);
        assert_approx_slice(&result, &[1.0, 1.0, 1.0, 1.0], EPS, "deq 0xAA");
    }

    #[test]
    fn dequant_i2s_all_threes() {
        // 0xFF = 0b11_11_11_11 → four values of (3-1) = 2
        let packed = [0xFF];
        let scales = [1.0];
        let result = dequantize_i2s_ref(&packed, &scales, 4);
        assert_approx_slice(&result, &[2.0, 2.0, 2.0, 2.0], EPS, "deq 0xFF");
    }

    #[test]
    fn dequant_i2s_mixed_byte() {
        // 0b11_10_01_00 = 0xE4
        // pos0: 0b00=0 → -1, pos1: 0b01=1 → 0, pos2: 0b10=2 → 1, pos3: 0b11=3 → 2
        let packed = [0xE4];
        let scales = [1.0];
        let result = dequantize_i2s_ref(&packed, &scales, 4);
        assert_approx_slice(&result, &[-1.0, 0.0, 1.0, 2.0], EPS, "deq mixed");
    }

    #[test]
    fn dequant_i2s_scale_application() {
        let packed = [0xAA]; // all 2-1=1
        let scales = [2.5];
        let result = dequantize_i2s_ref(&packed, &scales, 4);
        assert_approx_slice(&result, &[2.5, 2.5, 2.5, 2.5], EPS, "deq scaled");
    }

    #[test]
    fn dequant_i2s_multi_block() {
        // Two bytes, block_size=4 → each byte is its own block
        let packed = [0xAA, 0x00]; // block0: all 1, block1: all -1
        let scales = [1.0, 3.0];
        let result = dequantize_i2s_ref(&packed, &scales, 4);
        assert_eq!(result.len(), 8);
        assert_approx_slice(&result[..4], &[1.0, 1.0, 1.0, 1.0], EPS, "block 0");
        assert_approx_slice(&result[4..], &[-3.0, -3.0, -3.0, -3.0], EPS, "block 1");
    }

    #[test]
    fn dequant_i2s_zero_scale() {
        let packed = [0xFF];
        let scales = [0.0];
        let result = dequantize_i2s_ref(&packed, &scales, 4);
        assert_approx_slice(&result, &[0.0, 0.0, 0.0, 0.0], EPS, "zero scale");
    }

    #[test]
    fn dequant_i2s_negative_scale() {
        let packed = [0xAA]; // all 2-1=1
        let scales = [-2.0];
        let result = dequantize_i2s_ref(&packed, &scales, 4);
        assert_approx_slice(&result, &[-2.0, -2.0, -2.0, -2.0], EPS, "neg scale");
    }

    // ================================================================
    // Vector addition tests
    // ================================================================

    #[test]
    fn vec_add_basic() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let mut c = [0.0f32; 3];
        vec_add_ref(&a, &b, &mut c);
        assert_approx_slice(&c, &[5.0, 7.0, 9.0], EPS, "vec add");
    }

    #[test]
    fn vec_add_zeros() {
        let a = [1.0, 2.0, 3.0];
        let b = [0.0; 3];
        let mut c = [0.0f32; 3];
        vec_add_ref(&a, &b, &mut c);
        assert_approx_slice(&c, &a, EPS, "vec add zero");
    }

    #[test]
    fn vec_add_negation() {
        let a = [1.0, -2.0, 3.0];
        let b = [-1.0, 2.0, -3.0];
        let mut c = [0.0f32; 3];
        vec_add_ref(&a, &b, &mut c);
        assert_approx_slice(&c, &[0.0, 0.0, 0.0], EPS, "vec add negation");
    }

    // ================================================================
    // ReLU tests
    // ================================================================

    #[test]
    fn relu_positive() {
        let x = [1.0, 2.0, 3.0];
        let mut y = [0.0f32; 3];
        relu_ref(&x, &mut y);
        assert_approx_slice(&y, &x, EPS, "relu positive");
    }

    #[test]
    fn relu_negative() {
        let x = [-1.0, -2.0, -3.0];
        let mut y = [0.0f32; 3];
        relu_ref(&x, &mut y);
        assert_approx_slice(&y, &[0.0, 0.0, 0.0], EPS, "relu negative");
    }

    #[test]
    fn relu_mixed() {
        let x = [-1.0, 0.0, 1.0, -0.5, 0.5];
        let mut y = [0.0f32; 5];
        relu_ref(&x, &mut y);
        assert_approx_slice(&y, &[0.0, 0.0, 1.0, 0.0, 0.5], EPS, "relu mixed");
    }

    // ================================================================
    // Property-style tests with random inputs
    // ================================================================

    #[test]
    fn matmul_associativity_small() {
        // (A @ B) @ C ≈ A @ (B @ C) for 2x2 matrices
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let cc = [9.0, 10.0, 11.0, 12.0];
        let mut ab = [0.0f32; 4];
        matmul_naive_ref(&a, &b, &mut ab, 2, 2, 2, 1.0, 0.0);
        let mut ab_c = [0.0f32; 4];
        matmul_naive_ref(&ab, &cc, &mut ab_c, 2, 2, 2, 1.0, 0.0);
        let mut bc = [0.0f32; 4];
        matmul_naive_ref(&b, &cc, &mut bc, 2, 2, 2, 1.0, 0.0);
        let mut a_bc = [0.0f32; 4];
        matmul_naive_ref(&a, &bc, &mut a_bc, 2, 2, 2, 1.0, 0.0);
        assert_approx_slice(&ab_c, &a_bc, 1e-3, "matmul associativity");
    }

    #[test]
    fn softmax_sum_one_various_sizes() {
        for size in [1, 2, 3, 5, 8, 16, 32] {
            let input: Vec<f32> = (0..size).map(|i| (i as f32) * 0.3 - 1.0).collect();
            let mut output = vec![0.0f32; size];
            softmax_row_ref(&input, &mut output, 1, size);
            let sum: f32 = output.iter().sum();
            assert!(approx_eq(sum, 1.0, 1e-4), "softmax sum={sum} for size={size}");
        }
    }

    #[test]
    fn softmax_invariant_to_constant_shift() {
        let input1 = [1.0, 2.0, 3.0];
        let input2 = [101.0, 102.0, 103.0]; // shifted by 100
        let mut out1 = [0.0f32; 3];
        let mut out2 = [0.0f32; 3];
        softmax_row_ref(&input1, &mut out1, 1, 3);
        softmax_row_ref(&input2, &mut out2, 1, 3);
        assert_approx_slice(&out1, &out2, 1e-4, "softmax shift invariance");
    }

    #[test]
    fn layer_norm_invariant_to_affine_transform() {
        // layer_norm(a*x + b) with gamma=1, beta=0 should produce
        // same result as layer_norm(x) with gamma=1, beta=0
        let input1 = [1.0, 2.0, 3.0, 4.0];
        let input2: Vec<f32> = input1.iter().map(|&x| 5.0 * x + 100.0).collect();
        let gamma = [1.0; 4];
        let beta = [0.0; 4];
        let mut out1 = [0.0f32; 4];
        let mut out2 = [0.0f32; 4];
        layer_norm_ref(&input1, &gamma, &beta, &mut out1, 1, 4, 1e-5);
        layer_norm_ref(&input2, &gamma, &beta, &mut out2, 1, 4, 1e-5);
        assert_approx_slice(&out1, &out2, 1e-3, "layer norm affine invariance");
    }

    #[test]
    fn matmul_transpose_property() {
        // (A @ B)^T = B^T @ A^T - verify via known values
        // A = [1,2; 3,4], B = [5,6; 7,8]
        // A@B = [19,22; 43,50]
        // (A@B)^T = [19,43; 22,50]
        // B^T = [5,7; 6,8], A^T = [1,3; 2,4]
        // B^T @ A^T = [5*1+7*2, 5*3+7*4; 6*1+8*2, 6*3+8*4]
        //           = [19, 43; 22, 50] ✓
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut ab = [0.0f32; 4];
        matmul_naive_ref(&a, &b, &mut ab, 2, 2, 2, 1.0, 0.0);
        let ab_t = [ab[0], ab[2], ab[1], ab[3]];
        let bt = [5.0, 7.0, 6.0, 8.0];
        let at = [1.0, 3.0, 2.0, 4.0];
        let mut btat = [0.0f32; 4];
        matmul_naive_ref(&bt, &at, &mut btat, 2, 2, 2, 1.0, 0.0);
        assert_approx_slice(&ab_t, &btat, EPS, "transpose property");
    }

    #[test]
    fn silu_approaches_identity_for_positive() {
        // For large x, silu(x) ≈ x
        for &x_val in &[5.0, 10.0, 20.0, 50.0] {
            let x = [x_val];
            let mut y = [0.0f32; 1];
            silu_ref(&x, &mut y);
            let relative_error = (y[0] - x_val).abs() / x_val;
            assert!(relative_error < 0.01, "silu({x_val}) rel error = {relative_error}");
        }
    }

    #[test]
    fn gelu_approaches_identity_for_positive() {
        for &x_val in &[5.0, 10.0, 20.0] {
            let x = [x_val];
            let mut y = [0.0f32; 1];
            gelu_ref(&x, &mut y);
            let relative_error = (y[0] - x_val).abs() / x_val;
            assert!(relative_error < 0.01, "gelu({x_val}) rel error = {relative_error}");
        }
    }

    #[test]
    fn dequant_i2s_roundtrip_encode_decode() {
        // Encode known ternary values and verify decode
        let values: Vec<i8> = vec![1, -1, 0, 1, -1, -1, 0, 0];
        let mut packed = Vec::new();
        for chunk in values.chunks(4) {
            let mut byte = 0u8;
            for (i, &v) in chunk.iter().enumerate() {
                // Encode: val+1 gives 0→-1, 1→0, 2→+1
                let encoded = (v + 1) as u8;
                byte |= encoded << (i * 2);
            }
            packed.push(byte);
        }
        let scales = [1.0; 2]; // 2 blocks of 4
        let result = dequantize_i2s_ref(&packed, &scales, 4);
        // Decode maps val-1: 0→-1, 1→0, 2→+1
        let expected: Vec<f32> = values.iter().map(|&v| v as f32).collect();
        assert_approx_slice(&result, &expected, EPS, "roundtrip");
    }

    #[test]
    fn matmul_4x4_larger() {
        // 4x4 identity @ any matrix = same matrix
        let mut ident = [0.0f32; 16];
        for i in 0..4 {
            ident[i * 4 + i] = 1.0;
        }
        let b: Vec<f32> = (1..=16).map(|i| i as f32).collect();
        let mut c = [0.0f32; 16];
        matmul_naive_ref(&ident, &b, &mut c, 4, 4, 4, 1.0, 0.0);
        assert_approx_slice(&c, &b, EPS, "4x4 identity");
    }

    #[test]
    fn softmax_extreme_dominance() {
        // One very large value should dominate
        let input = [0.0, 0.0, 100.0, 0.0];
        let mut output = [0.0f32; 4];
        softmax_row_ref(&input, &mut output, 1, 4);
        assert!(output[2] > 0.99, "dominant value gets ~1.0: {}", output[2]);
    }

    #[test]
    fn rms_norm_scale_invariance() {
        // rms_norm(c*x) = c*x / rms(c*x) = c*x / (|c|*rms(x)) = sign(c) * x / rms(x)
        // So for positive c: rms_norm(c*x) = rms_norm(x)
        let input1 = [1.0, 2.0, 3.0, 4.0];
        let input2: Vec<f32> = input1.iter().map(|&x| 5.0 * x).collect();
        let gamma = [1.0; 4];
        let mut out1 = [0.0f32; 4];
        let mut out2 = [0.0f32; 4];
        rms_norm_ref(&input1, &gamma, &mut out1, 1, 4, 0.0);
        rms_norm_ref(&input2, &gamma, &mut out2, 1, 4, 0.0);
        assert_approx_slice(&out1, &out2, 1e-4, "rms norm scale invariance");
    }

    #[test]
    fn matmul_distributive() {
        // A @ (B + C) = A @ B + A @ C
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [1.0, 0.0, 0.0, 1.0];
        let cc = [0.0, 1.0, 1.0, 0.0];
        let b_plus_c: Vec<f32> = b.iter().zip(cc.iter()).map(|(x, y)| x + y).collect();
        let mut a_bpc = [0.0f32; 4];
        matmul_naive_ref(&a, &b_plus_c, &mut a_bpc, 2, 2, 2, 1.0, 0.0);
        let mut a_b = [0.0f32; 4];
        matmul_naive_ref(&a, &b, &mut a_b, 2, 2, 2, 1.0, 0.0);
        let mut a_c = [0.0f32; 4];
        matmul_naive_ref(&a, &cc, &mut a_c, 2, 2, 2, 1.0, 0.0);
        let sum: Vec<f32> = a_b.iter().zip(a_c.iter()).map(|(x, y)| x + y).collect();
        assert_approx_slice(&a_bpc, &sum, EPS, "distributive");
    }
}

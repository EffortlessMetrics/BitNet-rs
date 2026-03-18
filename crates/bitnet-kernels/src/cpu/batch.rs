//! Contiguous-layout CPU batch operations for efficient multi-input inference.
//!
//! All functions operate on flat `&[f32]` buffers where individual batch
//! elements are laid out contiguously.  This layout is cache-friendly and
//! maps naturally to GPU tensor semantics.

use bitnet_common::{BitNetError, KernelError, Result};

// ── Batched matrix multiply ────────────────────────────────────────────

/// Batch matrix multiply: for each batch element compute `C = A * B`.
///
/// `a` has shape `[batch, m, k]`, `b` has shape `[batch, k, n]`, and the
/// result has shape `[batch, m, n]`.  All matrices are **row-major**.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] when the input slices do
/// not match the declared dimensions.
pub fn batched_matmul(
    a: &[f32],
    b: &[f32],
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>> {
    let a_batch_len = m * k;
    let b_batch_len = k * n;
    let c_batch_len = m * n;

    if a.len() != batch * a_batch_len {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!(
                "dimension mismatch: expected {}, got {}",
                batch * a_batch_len,
                a.len()
            ),
        }));
    }
    if b.len() != batch * b_batch_len {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!(
                "dimension mismatch: expected {}, got {}",
                batch * b_batch_len,
                b.len()
            ),
        }));
    }

    let mut out = vec![0.0f32; batch * c_batch_len];

    for bi in 0..batch {
        let a_off = bi * a_batch_len;
        let b_off = bi * b_batch_len;
        let c_off = bi * c_batch_len;
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[a_off + i * k + p] * b[b_off + p * n + j];
                }
                out[c_off + i * n + j] = sum;
            }
        }
    }

    Ok(out)
}

// ── Batched softmax ────────────────────────────────────────────────────

/// Row-wise softmax over a `[batch, seq_len]` tensor.
///
/// Each row of length `seq_len` is independently normalised.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] when `input.len() !=
/// batch * seq_len`.
pub fn batched_softmax(input: &[f32], batch: usize, seq_len: usize) -> Result<Vec<f32>> {
    let total = batch * seq_len;
    if input.len() != total {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("dimension mismatch: expected {}, got {}", total, input.len()),
        }));
    }

    let mut out = vec![0.0f32; total];

    for bi in 0..batch {
        let off = bi * seq_len;
        let row = &input[off..off + seq_len];

        // Numerically stable softmax: subtract max before exp.
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let mut sum_exp = 0.0f32;
        for j in 0..seq_len {
            let e = (row[j] - max_val).exp();
            out[off + j] = e;
            sum_exp += e;
        }

        if sum_exp > 0.0 {
            for j in 0..seq_len {
                out[off + j] /= sum_exp;
            }
        }
    }

    Ok(out)
}

// ── Batched layer norm ─────────────────────────────────────────────────

/// Contiguous-layout layer normalisation over a `[batch, dim]` tensor.
///
/// For each batch element `x` of length `dim`:
///
/// ```text
/// y = gamma * (x - mean) / sqrt(var + eps) + beta
/// ```
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] when slice lengths are
/// inconsistent.
pub fn batched_layer_norm(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    batch: usize,
    dim: usize,
    eps: f32,
) -> Result<Vec<f32>> {
    let total = batch * dim;
    if input.len() != total {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("dimension mismatch: expected {}, got {}", total, input.len()),
        }));
    }
    if gamma.len() != dim {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("dimension mismatch: expected {}, got {}", dim, gamma.len()),
        }));
    }
    if beta.len() != dim {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("dimension mismatch: expected {}, got {}", dim, beta.len()),
        }));
    }

    let mut out = vec![0.0f32; total];

    for bi in 0..batch {
        let off = bi * dim;
        let row = &input[off..off + dim];

        let mean = row.iter().sum::<f32>() / dim as f32;
        let var = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
        let inv_std = 1.0 / (var + eps).sqrt();

        for j in 0..dim {
            out[off + j] = gamma[j] * (row[j] - mean) * inv_std + beta[j];
        }
    }

    Ok(out)
}

// ── Batched element-wise add ───────────────────────────────────────────

/// Element-wise addition of two `[batch, dim]` tensors.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] when slice lengths are
/// inconsistent.
pub fn batched_add(a: &[f32], b: &[f32], batch: usize, dim: usize) -> Result<Vec<f32>> {
    let total = batch * dim;
    if a.len() != total {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("dimension mismatch: expected {}, got {}", total, a.len()),
        }));
    }
    if b.len() != total {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("dimension mismatch: expected {}, got {}", total, b.len()),
        }));
    }

    Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < tol,
                "mismatch at index {i}: {x} vs {y} (diff {})",
                (x - y).abs()
            );
        }
    }

    // ── matmul tests ───────────────────────────────────────────────────

    #[test]
    fn matmul_identity_batch1() {
        // A=[[1,2],[3,4]], B=I => C=A
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let c = batched_matmul(&a, &b, 1, 2, 2, 2).unwrap();
        approx_eq(&c, &[1.0, 2.0, 3.0, 4.0], EPS);
    }

    #[test]
    fn matmul_known_values() {
        // [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c = batched_matmul(&a, &b, 1, 2, 2, 2).unwrap();
        approx_eq(&c, &[19.0, 22.0, 43.0, 50.0], EPS);
    }

    #[test]
    fn matmul_batch2_equals_individual() {
        let a1 = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let b1 = vec![5.0, 6.0, 7.0, 8.0];
        let a2 = vec![2.0, 0.0, 0.0, 3.0];
        let b2 = vec![1.0, 1.0, 1.0, 1.0];

        let c1 = batched_matmul(&a1, &b1, 1, 2, 2, 2).unwrap();
        let c2 = batched_matmul(&a2, &b2, 1, 2, 2, 2).unwrap();

        let a_cat: Vec<f32> = [a1, a2].concat();
        let b_cat: Vec<f32> = [b1, b2].concat();
        let c_batched = batched_matmul(&a_cat, &b_cat, 2, 2, 2, 2).unwrap();

        let expected: Vec<f32> = [c1, c2].concat();
        approx_eq(&c_batched, &expected, EPS);
    }

    #[test]
    fn matmul_non_square() {
        // A: 1x3, B: 3x2 => C: 1x2
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let c = batched_matmul(&a, &b, 1, 1, 3, 2).unwrap();
        // [1*1+2*0+3*1, 1*0+2*1+3*1] = [4, 5]
        approx_eq(&c, &[4.0, 5.0], EPS);
    }

    #[test]
    fn matmul_dim_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!(batched_matmul(&a, &b, 1, 2, 2, 2).is_err());
    }

    #[test]
    fn matmul_many_batches() {
        let batch = 8;
        let m = 2;
        let k = 3;
        let n = 2;
        let a = vec![1.0; batch * m * k];
        let b = vec![1.0; batch * k * n];
        let c = batched_matmul(&a, &b, batch, m, k, n).unwrap();
        assert_eq!(c.len(), batch * m * n);
        // Each element = sum of k ones = 3.0
        for &v in &c {
            assert!((v - k as f32).abs() < EPS);
        }
    }

    #[test]
    fn matmul_dim1() {
        // scalar multiply: 1x1 * 1x1
        let c = batched_matmul(&[3.0], &[4.0], 1, 1, 1, 1).unwrap();
        approx_eq(&c, &[12.0], EPS);
    }

    // ── softmax tests ──────────────────────────────────────────────────

    #[test]
    fn softmax_uniform() {
        let input = vec![0.0, 0.0, 0.0, 0.0];
        let out = batched_softmax(&input, 1, 4).unwrap();
        approx_eq(&out, &[0.25, 0.25, 0.25, 0.25], EPS);
    }

    #[test]
    fn softmax_sums_to_one() {
        let input = vec![1.0, 2.0, 3.0];
        let out = batched_softmax(&input, 1, 3).unwrap();
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < EPS);
    }

    #[test]
    fn softmax_batch2_equals_individual() {
        let row1 = vec![1.0, 2.0, 3.0];
        let row2 = vec![4.0, 0.0, -1.0];

        let s1 = batched_softmax(&row1, 1, 3).unwrap();
        let s2 = batched_softmax(&row2, 1, 3).unwrap();

        let input: Vec<f32> = [row1, row2].concat();
        let batched = batched_softmax(&input, 2, 3).unwrap();

        let expected: Vec<f32> = [s1, s2].concat();
        approx_eq(&batched, &expected, EPS);
    }

    #[test]
    fn softmax_large_values_stable() {
        // Numerical stability: large values should not overflow.
        let input = vec![1000.0, 1001.0, 1002.0];
        let out = batched_softmax(&input, 1, 3).unwrap();
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < EPS);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn softmax_dim1() {
        let out = batched_softmax(&[42.0], 1, 1).unwrap();
        approx_eq(&out, &[1.0], EPS);
    }

    #[test]
    fn softmax_dim_mismatch() {
        assert!(batched_softmax(&[1.0, 2.0], 1, 3).is_err());
    }

    // ── layer norm tests ───────────────────────────────────────────────

    #[test]
    fn layer_norm_zero_mean_unit_var() {
        // Input already normalised: [−1, 1], gamma=1, beta=0.
        let input = vec![-1.0, 1.0];
        let gamma = vec![1.0, 1.0];
        let beta = vec![0.0, 0.0];
        let out = batched_layer_norm(&input, &gamma, &beta, 1, 2, EPS).unwrap();
        // mean=0, var=1 → output ≈ input
        approx_eq(&out, &[-1.0, 1.0], 1e-3);
    }

    #[test]
    fn layer_norm_shift_and_scale() {
        // Constant input → mean=c, var=0 → output = beta.
        let input = vec![5.0, 5.0, 5.0];
        let gamma = vec![2.0, 2.0, 2.0];
        let beta = vec![1.0, 1.0, 1.0];
        let out = batched_layer_norm(&input, &gamma, &beta, 1, 3, EPS).unwrap();
        // (x - mean)/std = 0 for all, so output = beta
        approx_eq(&out, &[1.0, 1.0, 1.0], 1e-3);
    }

    #[test]
    fn layer_norm_batch2_equals_individual() {
        let gamma = vec![1.0, 1.0, 1.0];
        let beta = vec![0.0, 0.0, 0.0];

        let r1 = vec![1.0, 2.0, 3.0];
        let r2 = vec![4.0, 5.0, 6.0];

        let ln1 = batched_layer_norm(&r1, &gamma, &beta, 1, 3, EPS).unwrap();
        let ln2 = batched_layer_norm(&r2, &gamma, &beta, 1, 3, EPS).unwrap();

        let input: Vec<f32> = [r1, r2].concat();
        let batched = batched_layer_norm(&input, &gamma, &beta, 2, 3, EPS).unwrap();

        let expected: Vec<f32> = [ln1, ln2].concat();
        approx_eq(&batched, &expected, EPS);
    }

    #[test]
    fn layer_norm_dim_mismatch_input() {
        let gamma = vec![1.0, 1.0];
        let beta = vec![0.0, 0.0];
        assert!(batched_layer_norm(&[1.0], &gamma, &beta, 1, 2, EPS).is_err());
    }

    #[test]
    fn layer_norm_dim_mismatch_gamma() {
        let input = vec![1.0, 2.0];
        let gamma = [1.0];
        let beta = vec![0.0, 0.0];
        assert!(batched_layer_norm(&input, &gamma, &beta, 1, 2, EPS).is_err());
    }

    #[test]
    fn layer_norm_dim1() {
        // dim=1: always normalises to 0, then output = beta
        let out = batched_layer_norm(&[99.0], &[2.0], &[7.0], 1, 1, EPS).unwrap();
        approx_eq(&out, &[7.0], 1e-3);
    }

    // ── add tests ──────────────────────────────────────────────────────

    #[test]
    fn add_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![10.0, 20.0, 30.0, 40.0];
        let c = batched_add(&a, &b, 2, 2).unwrap();
        approx_eq(&c, &[11.0, 22.0, 33.0, 44.0], EPS);
    }

    #[test]
    fn add_batch2_equals_individual() {
        let a1 = vec![1.0, 2.0];
        let b1 = vec![3.0, 4.0];
        let a2 = vec![5.0, 6.0];
        let b2 = vec![7.0, 8.0];

        let c1 = batched_add(&a1, &b1, 1, 2).unwrap();
        let c2 = batched_add(&a2, &b2, 1, 2).unwrap();

        let a_cat: Vec<f32> = [a1, a2].concat();
        let b_cat: Vec<f32> = [b1, b2].concat();
        let batched = batched_add(&a_cat, &b_cat, 2, 2).unwrap();

        let expected: Vec<f32> = [c1, c2].concat();
        approx_eq(&batched, &expected, EPS);
    }

    #[test]
    fn add_dim_mismatch() {
        assert!(batched_add(&[1.0], &[1.0, 2.0], 1, 2).is_err());
    }

    #[test]
    fn add_dim1() {
        let c = batched_add(&[3.0], &[7.0], 1, 1).unwrap();
        approx_eq(&c, &[10.0], EPS);
    }

    #[test]
    fn add_zeros() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.0, 0.0, 0.0];
        let c = batched_add(&a, &b, 1, 3).unwrap();
        approx_eq(&c, &a, EPS);
    }

    #[test]
    fn add_many_batches() {
        let batch = 16;
        let dim = 4;
        let a = vec![1.0; batch * dim];
        let b = vec![2.0; batch * dim];
        let c = batched_add(&a, &b, batch, dim).unwrap();
        assert!(c.iter().all(|&v| (v - 3.0).abs() < EPS));
    }

    // ── property: batch == concat of individuals ───────────────────────

    #[test]
    fn matmul_property_batch_equals_concat() {
        let batch = 4;
        let (m, k, n) = (3, 2, 4);
        let a: Vec<f32> = (0..batch * m * k).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..batch * k * n).map(|i| (i as f32) * 0.2 - 1.0).collect();

        let batched = batched_matmul(&a, &b, batch, m, k, n).unwrap();

        let mut concat = Vec::new();
        for bi in 0..batch {
            let a_slice = &a[bi * m * k..(bi + 1) * m * k];
            let b_slice = &b[bi * k * n..(bi + 1) * k * n];
            let single = batched_matmul(a_slice, b_slice, 1, m, k, n).unwrap();
            concat.extend_from_slice(&single);
        }

        approx_eq(&batched, &concat, 1e-4);
    }

    #[test]
    fn softmax_property_batch_equals_concat() {
        let batch = 5;
        let seq = 6;
        let input: Vec<f32> = (0..batch * seq).map(|i| (i as f32) * 0.3 - 2.0).collect();

        let batched = batched_softmax(&input, batch, seq).unwrap();

        let mut concat = Vec::new();
        for bi in 0..batch {
            let row = &input[bi * seq..(bi + 1) * seq];
            let single = batched_softmax(row, 1, seq).unwrap();
            concat.extend_from_slice(&single);
        }

        approx_eq(&batched, &concat, EPS);
    }

    #[test]
    fn layer_norm_property_batch_equals_concat() {
        let batch = 4;
        let dim = 5;
        let gamma: Vec<f32> = (0..dim).map(|i| 0.5 + i as f32 * 0.1).collect();
        let beta: Vec<f32> = (0..dim).map(|i| i as f32 * 0.01).collect();
        let input: Vec<f32> = (0..batch * dim).map(|i| (i as f32) * 0.7 - 3.0).collect();

        let batched = batched_layer_norm(&input, &gamma, &beta, batch, dim, EPS).unwrap();

        let mut concat = Vec::new();
        for bi in 0..batch {
            let row = &input[bi * dim..(bi + 1) * dim];
            let single = batched_layer_norm(row, &gamma, &beta, 1, dim, EPS).unwrap();
            concat.extend_from_slice(&single);
        }

        approx_eq(&batched, &concat, 1e-4);
    }

    #[test]
    fn add_property_batch_equals_concat() {
        let batch = 6;
        let dim = 3;
        let a: Vec<f32> = (0..batch * dim).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..batch * dim).map(|i| -(i as f32)).collect();

        let batched = batched_add(&a, &b, batch, dim).unwrap();

        let mut concat = Vec::new();
        for bi in 0..batch {
            let a_slice = &a[bi * dim..(bi + 1) * dim];
            let b_slice = &b[bi * dim..(bi + 1) * dim];
            let single = batched_add(a_slice, b_slice, 1, dim).unwrap();
            concat.extend_from_slice(&single);
        }

        approx_eq(&batched, &concat, EPS);
    }
}

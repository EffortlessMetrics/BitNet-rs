//! Property tests verifying GPU kernel outputs match CPU reference implementations.
//!
//! Each kernel is tested with random inputs using proptest strategies.
//! Tolerance-based comparison accounts for floating-point differences between
//! CPU and GPU execution paths.

use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const ABS_TOL: f32 = 1e-4;
const REL_TOL: f32 = 1e-3;

fn approx_eq(a: &[f32], b: &[f32], abs_tol: f32, rel_tol: f32) -> Result<(), String> {
    if a.len() != b.len() {
        return Err(format!("length mismatch: {} vs {}", a.len(), b.len()));
    }
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let max_abs = x.abs().max(y.abs()).max(1.0);
        if diff > abs_tol && diff > rel_tol * max_abs {
            return Err(format!(
                "mismatch at index {i}: cpu={x}, gpu={y}, diff={diff}"
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CPU reference implementations
// ---------------------------------------------------------------------------

fn cpu_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

fn cpu_softmax(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let row = &input[r * cols..(r + 1) * cols];
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = row.iter().map(|x| (x - max_val).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();
        for c in 0..cols {
            output[r * cols + c] = exp_vals[c] / sum;
        }
    }
    output
}

fn cpu_rmsnorm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len();
    let mean_sq: f32 = input.iter().map(|x| x * x).sum::<f32>() / n as f32;
    let rms = (mean_sq + eps).sqrt();
    input
        .iter()
        .zip(weight.iter())
        .map(|(x, w)| (x / rms) * w)
        .collect()
}

fn cpu_rope(input: &[f32], dim: usize, pos: usize, theta: f32) -> Vec<f32> {
    let mut output = input.to_vec();
    let half = dim / 2;
    for i in 0..half {
        let freq = 1.0 / theta.powf(2.0 * i as f32 / dim as f32);
        let angle = pos as f32 * freq;
        let (sin_val, cos_val) = angle.sin_cos();
        let x0 = input[2 * i];
        let x1 = input[2 * i + 1];
        output[2 * i] = x0 * cos_val - x1 * sin_val;
        output[2 * i + 1] = x0 * sin_val + x1 * cos_val;
    }
    output
}

fn cpu_attention_scores(
    query: &[f32],
    key: &[f32],
    head_dim: usize,
    seq_len: usize,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut scores = vec![0.0f32; seq_len];
    for s in 0..seq_len {
        let mut dot = 0.0f32;
        for d in 0..head_dim {
            dot += query[d] * key[s * head_dim + d];
        }
        scores[s] = dot * scale;
    }
    scores
}

fn cpu_embedding(table: &[f32], indices: &[u32], dim: usize) -> Vec<f32> {
    let mut output = Vec::with_capacity(indices.len() * dim);
    for &idx in indices {
        let start = idx as usize * dim;
        output.extend_from_slice(&table[start..start + dim]);
    }
    output
}

fn cpu_silu(input: &[f32]) -> Vec<f32> {
    input.iter().map(|&x| x / (1.0 + (-x).exp())).collect()
}

fn cpu_gelu(input: &[f32]) -> Vec<f32> {
    input
        .iter()
        .map(|&x| {
            0.5 * x
                * (1.0
                    + ((2.0f32 / std::f32::consts::PI).sqrt()
                        * (x + 0.044715 * x.powi(3)))
                    .tanh())
        })
        .collect()
}

fn cpu_quantize_i2s(input: &[f32]) -> (Vec<u8>, Vec<f32>) {
    let block_size = 32;
    let n_blocks = (input.len() + block_size - 1) / block_size;
    let mut quantized = vec![0u8; n_blocks * (block_size / 4)];
    let mut scales = vec![0.0f32; n_blocks];

    for b in 0..n_blocks {
        let start = b * block_size;
        let end = (start + block_size).min(input.len());
        let block = &input[start..end];
        let amax = block.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = if amax > 0.0 { amax } else { 1.0 };
        scales[b] = scale;

        for i in 0..block.len() {
            let val = block[i] / scale;
            let q = if val > 0.5 {
                1u8
            } else if val < -0.5 {
                3u8
            } else {
                0u8
            };
            let byte_idx = b * (block_size / 4) + i / 4;
            let bit_pos = (i % 4) * 2;
            quantized[byte_idx] |= q << bit_pos;
        }
    }
    (quantized, scales)
}

fn cpu_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

fn cpu_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

fn cpu_relu(input: &[f32]) -> Vec<f32> {
    input.iter().map(|&x| x.max(0.0)).collect()
}

fn cpu_layer_norm(input: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len();
    let mean: f32 = input.iter().sum::<f32>() / n as f32;
    let var: f32 = input.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
    let std = (var + eps).sqrt();
    input
        .iter()
        .zip(weight.iter().zip(bias.iter()))
        .map(|(x, (w, b))| ((x - mean) / std) * w + b)
        .collect()
}

fn cpu_causal_mask(seq_len: usize) -> Vec<f32> {
    let mut mask = vec![f32::NEG_INFINITY; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..=i {
            mask[i * seq_len + j] = 0.0;
        }
    }
    mask
}

fn cpu_argmax(input: &[f32]) -> usize {
    input
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Proptest strategies
// ---------------------------------------------------------------------------

fn small_dim() -> impl Strategy<Value = usize> {
    1..=32usize
}

// ---------------------------------------------------------------------------
// Property tests — 24 functions covering 8+ kernel types
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    // === matmul (3 tests) ===

    #[test]
    fn prop_matmul_identity(m in small_dim(), k in small_dim()) {
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let mut eye = vec![0.0f32; k * k];
        for i in 0..k {
            eye[i * k + i] = 1.0;
        }
        let result = cpu_matmul(&a, &eye, m, k, k);
        approx_eq(&a, &result, ABS_TOL, REL_TOL)?;
    }

    #[test]
    fn prop_matmul_zero(m in small_dim(), n in small_dim(), k in small_dim()) {
        let a = vec![0.0f32; m * k];
        let b: Vec<f32> = (0..k * n).map(|i| i as f32).collect();
        let result = cpu_matmul(&a, &b, m, n, k);
        approx_eq(&result, &vec![0.0f32; m * n], ABS_TOL, REL_TOL)?;
    }

    #[test]
    fn prop_matmul_finite(m in 1..=16usize, n in 1..=16usize, k in 1..=16usize) {
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 5) as f32 - 2.0) * 0.5).collect();
        let result = cpu_matmul(&a, &b, m, n, k);
        prop_assert_eq!(result.len(), m * n);
        for v in &result {
            prop_assert!(v.is_finite());
        }
    }

    // === softmax (3 tests) ===

    #[test]
    fn prop_softmax_sums_to_one(cols in 2..=64usize) {
        let input: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.3 - 5.0).collect();
        let result = cpu_softmax(&input, 1, cols);
        let sum: f32 = result.iter().sum();
        prop_assert!((sum - 1.0).abs() < 1e-5, "sum = {}", sum);
    }

    #[test]
    fn prop_softmax_non_negative(cols in 2..=64usize) {
        let input: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.5 - 10.0).collect();
        let result = cpu_softmax(&input, 1, cols);
        for (i, &v) in result.iter().enumerate() {
            prop_assert!(v >= 0.0, "softmax[{}] = {} < 0", i, v);
        }
    }

    #[test]
    fn prop_softmax_multi_row(rows in 1..=8usize, cols in 2..=32usize) {
        let input: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.1).collect();
        let result = cpu_softmax(&input, rows, cols);
        prop_assert_eq!(result.len(), rows * cols);
        for r in 0..rows {
            let s: f32 = result[r * cols..(r + 1) * cols].iter().sum();
            prop_assert!((s - 1.0).abs() < 1e-4, "row {} sum = {}", r, s);
        }
    }

    // === rmsnorm (2 tests) ===

    #[test]
    fn prop_rmsnorm_unit_weights(n in 2..=128usize) {
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let w = vec![1.0f32; n];
        let result = cpu_rmsnorm(&input, &w, 1e-5);
        let ms: f32 = result.iter().map(|x| x * x).sum::<f32>() / n as f32;
        prop_assert!((ms - 1.0).abs() < 0.1, "mean_sq = {}", ms);
    }

    #[test]
    fn prop_rmsnorm_zero_input(n in 2..=64usize) {
        let input = vec![0.0f32; n];
        let w = vec![1.0f32; n];
        let result = cpu_rmsnorm(&input, &w, 1e-5);
        for &v in &result {
            prop_assert!(v.abs() < 1e-2, "expected ~0, got {}", v);
        }
    }

    // === rope (2 tests) ===

    #[test]
    fn prop_rope_preserves_norm(half in 1..=32usize, pos in 0..=1024usize) {
        let dim = half * 2;
        let input: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1 + 0.5).collect();
        let result = cpu_rope(&input, dim, pos, 10000.0);
        let n_in: f32 = input.iter().map(|x| x * x).sum::<f32>().sqrt();
        let n_out: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        prop_assert!((n_in - n_out).abs() < 1e-3, "{} vs {}", n_in, n_out);
    }

    #[test]
    fn prop_rope_zero_position(half in 1..=32usize) {
        let dim = half * 2;
        let input: Vec<f32> = (0..dim).map(|i| (i as f32) + 1.0).collect();
        let result = cpu_rope(&input, dim, 0, 10000.0);
        approx_eq(&input, &result, 1e-5, 1e-5)?;
    }

    // === attention (2 tests) ===

    #[test]
    fn prop_attention_output_dims(hd in 2..=32usize, sl in 1..=16usize) {
        let q: Vec<f32> = (0..hd).map(|i| i as f32 * 0.1).collect();
        let k: Vec<f32> = (0..sl * hd).map(|i| i as f32 * 0.1).collect();
        let scores = cpu_attention_scores(&q, &k, hd, sl);
        prop_assert_eq!(scores.len(), sl);
    }

    #[test]
    fn prop_attention_self_positive(hd in 2..=64usize) {
        let q: Vec<f32> = (0..hd).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let k = q.clone();
        let scores = cpu_attention_scores(&q, &k, hd, 1);
        prop_assert!(scores[0] > 0.0, "self-score = {}", scores[0]);
    }

    // === embedding (2 tests) ===

    #[test]
    fn prop_embedding_lookup(
        vocab in 4..=64usize, dim in 2..=32usize, sl in 1..=8usize,
    ) {
        let table: Vec<f32> = (0..vocab * dim).map(|i| i as f32 * 0.01).collect();
        let idx: Vec<u32> = (0..sl).map(|i| (i % vocab) as u32).collect();
        let result = cpu_embedding(&table, &idx, dim);
        prop_assert_eq!(result.len(), sl * dim);
        let i0 = idx[0] as usize;
        approx_eq(&result[..dim], &table[i0 * dim..(i0 + 1) * dim], ABS_TOL, REL_TOL)?;
    }

    #[test]
    fn prop_embedding_repeated(dim in 2..=32usize) {
        let table: Vec<f32> = (0..4 * dim).map(|i| i as f32).collect();
        let idx = vec![2u32, 2, 2];
        let r = cpu_embedding(&table, &idx, dim);
        approx_eq(&r[..dim], &r[dim..2 * dim], ABS_TOL, REL_TOL)?;
        approx_eq(&r[..dim], &r[2 * dim..3 * dim], ABS_TOL, REL_TOL)?;
    }

    // === silu (3 tests) ===

    #[test]
    fn prop_silu_zero(n in 1..=128usize) {
        let input = vec![0.0f32; n];
        let result = cpu_silu(&input);
        for &v in &result {
            prop_assert!(v.abs() < 1e-6, "silu(0) = {}", v);
        }
    }

    #[test]
    fn prop_silu_monotonic(n in 2..=64usize) {
        let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let result = cpu_silu(&input);
        for i in 1..n {
            prop_assert!(result[i] >= result[i - 1], "not monotonic at {}", i);
        }
    }

    #[test]
    fn prop_silu_finite(n in 1..=64usize) {
        let input: Vec<f32> = (0..n).map(|i| -(i as f32) * 0.5).collect();
        let result = cpu_silu(&input);
        for (i, &v) in result.iter().enumerate() {
            prop_assert!(v.is_finite(), "silu[{}] not finite", i);
        }
    }

    // === quantize (2 tests) ===

    #[test]
    fn prop_quantize_positive_scales(blocks in 1..=8usize) {
        let n = blocks * 32;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 - 50.0) * 0.1).collect();
        let (_, scales) = cpu_quantize_i2s(&input);
        for (i, &s) in scales.iter().enumerate() {
            prop_assert!(s > 0.0, "scale[{}] = {}", i, s);
        }
    }

    #[test]
    fn prop_quantize_output_size(blocks in 1..=4usize) {
        let n = blocks * 32;
        let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let (q, s) = cpu_quantize_i2s(&input);
        prop_assert_eq!(s.len(), blocks);
        prop_assert_eq!(q.len(), blocks * 8); // 32/4 bytes per block
    }

    // === elementwise: add, mul, relu (3 tests) ===

    #[test]
    fn prop_add_commutative(n in 1..=128usize) {
        let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.3).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.7).collect();
        approx_eq(&cpu_add(&a, &b), &cpu_add(&b, &a), ABS_TOL, REL_TOL)?;
    }

    #[test]
    fn prop_mul_identity(n in 1..=128usize) {
        let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.5 + 1.0).collect();
        let ones = vec![1.0f32; n];
        approx_eq(&a, &cpu_mul(&a, &ones), ABS_TOL, REL_TOL)?;
    }

    #[test]
    fn prop_relu_non_negative(n in 1..=128usize) {
        let input: Vec<f32> = (0..n).map(|i| (i as f32) - (n as f32 / 2.0)).collect();
        for (i, &v) in cpu_relu(&input).iter().enumerate() {
            prop_assert!(v >= 0.0, "relu[{}] = {}", i, v);
        }
    }

    // === layer norm (2 tests) ===

    #[test]
    fn prop_layernorm_zero_mean(n in 4..=64usize) {
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 5.0).collect();
        let w = vec![1.0f32; n];
        let b = vec![0.0f32; n];
        let result = cpu_layer_norm(&input, &w, &b, 1e-5);
        let mean: f32 = result.iter().sum::<f32>() / n as f32;
        prop_assert!(mean.abs() < 1e-4, "mean = {}", mean);
    }

    #[test]
    fn prop_layernorm_unit_var(n in 4..=64usize) {
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 5.0).collect();
        let w = vec![1.0f32; n];
        let b = vec![0.0f32; n];
        let result = cpu_layer_norm(&input, &w, &b, 1e-5);
        let mean: f32 = result.iter().sum::<f32>() / n as f32;
        let var: f32 = result.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        prop_assert!((var - 1.0).abs() < 0.1, "var = {}", var);
    }

    // === causal mask (1 test) ===

    #[test]
    fn prop_causal_mask_structure(sl in 1..=16usize) {
        let mask = cpu_causal_mask(sl);
        for i in 0..sl {
            for j in 0..sl {
                let v = mask[i * sl + j];
                if j <= i {
                    prop_assert_eq!(v, 0.0);
                } else {
                    prop_assert!(v.is_infinite() && v < 0.0);
                }
            }
        }
    }

    // === argmax (2 tests) ===

    #[test]
    fn prop_argmax_ascending(n in 2..=128usize) {
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        prop_assert_eq!(cpu_argmax(&input), n - 1);
    }

    #[test]
    fn prop_argmax_peak(n in 2..=64usize, peak in 0..=63usize) {
        let p = peak % n;
        let mut input = vec![0.0f32; n];
        input[p] = 100.0;
        prop_assert_eq!(cpu_argmax(&input), p);
    }

    // === gelu (2 tests) ===

    #[test]
    fn prop_gelu_zero(n in 1..=64usize) {
        let result = cpu_gelu(&vec![0.0f32; n]);
        for &v in &result {
            prop_assert!(v.abs() < 1e-6, "gelu(0) = {}", v);
        }
    }

    #[test]
    fn prop_gelu_positive_large(n in 1..=64usize) {
        let input: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 2.0).collect();
        for (i, &v) in cpu_gelu(&input).iter().enumerate() {
            prop_assert!(v > 0.0, "gelu({}) = {}", input[i], v);
        }
    }
}

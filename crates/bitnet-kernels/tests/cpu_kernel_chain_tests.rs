#![allow(dead_code, unused_imports, unused_variables, unused_unsafe, unsafe_op_in_unsafe_fn)]
#![cfg(feature = "cpu")]

//! End-to-end chain smoke tests that wire real CPU kernels together with
//! synthetic data.  No model file needed — exercises the actual kernel
//! dispatch paths (rms_norm → matmul → softmax).

use bitnet_kernels::cpu::layer_norm::{LayerNormConfig, layer_norm, rms_norm};
use bitnet_kernels::cpu::simd_matmul::{SimdMatmulConfig, simd_matmul_f32};
use bitnet_kernels::cpu::softmax::{softmax_f32, softmax_topk};

// ── helpers ────────────────────────────────────────────────────────────

/// Deterministic PRNG (xorshift32) so the test is reproducible without
/// pulling in `rand`.
struct Xorshift32(u32);

impl Xorshift32 {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        // map to roughly [-1, 1]
        (self.0 as f32) / (u32::MAX as f32) * 2.0 - 1.0
    }

    fn fill(&mut self, buf: &mut [f32]) {
        for v in buf.iter_mut() {
            *v = self.next_f32();
        }
    }
}

fn assert_no_nan_inf(data: &[f32], label: &str) {
    for (i, &v) in data.iter().enumerate() {
        assert!(v.is_finite(), "{label}[{i}] is not finite: {v}");
    }
}

// ── tests ──────────────────────────────────────────────────────────────

/// Chain: rms_norm → matmul → softmax.
///
/// Verifies that real CPU kernels compose correctly on synthetic data and
/// produce a valid probability distribution (sums to ~1, all in [0,1]).
#[test]
fn test_rms_norm_then_matmul_then_softmax_chain() {
    const DIM: usize = 128;
    let mut rng = Xorshift32(42);

    // 1. Synthetic input & weight vectors
    let mut input = vec![0.0f32; DIM];
    rng.fill(&mut input);

    let gamma: Vec<f32> = (0..DIM).map(|_| 1.0 + rng.next_f32() * 0.1).collect();

    // 2. RMS-norm
    let config =
        LayerNormConfig { normalized_shape: vec![DIM], eps: 1e-5, elementwise_affine: true };
    let normed = rms_norm(&input, &gamma, &config).expect("rms_norm failed");
    assert_eq!(normed.len(), DIM);
    assert_no_nan_inf(&normed, "rms_norm output");

    // 3. MatMul:  (1×DIM) @ (DIM×DIM) → (1×DIM)
    let mut weight_matrix = vec![0.0f32; DIM * DIM];
    rng.fill(&mut weight_matrix);
    // Scale weights down to keep values reasonable
    for w in weight_matrix.iter_mut() {
        *w *= 0.1;
    }

    let cfg = SimdMatmulConfig::new(1, DIM, DIM);
    let mut matmul_out = vec![0.0f32; DIM];
    simd_matmul_f32(&normed, &weight_matrix, &mut matmul_out, &cfg)
        .expect("simd_matmul_f32 failed");
    assert_no_nan_inf(&matmul_out, "matmul output");

    // 4. Softmax
    let mut probs = vec![0.0f32; DIM];
    softmax_f32(&matmul_out, &mut probs).expect("softmax_f32 failed");
    assert_no_nan_inf(&probs, "softmax output");

    // Verify valid probability distribution
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax output should sum to ~1.0, got {sum}");
    for (i, &p) in probs.iter().enumerate() {
        assert!((0.0..=1.0).contains(&p), "softmax output[{i}] = {p} not in [0,1]");
    }
}

/// Apply layer_norm to data containing extreme values and verify
/// numerical stability (no NaN / Inf).
#[test]
fn test_layer_norm_numerical_stability() {
    const DIM: usize = 64;

    let extreme_inputs: &[&[f32]] = &[
        // large positive / negative
        &{
            let mut v = vec![1e6_f32; DIM];
            v[0] = -1e6;
            v[DIM / 2] = 0.0;
            v
        },
        // tiny values near zero
        &vec![1e-7_f32; DIM],
        // mixed scales
        &{
            let mut v = vec![0.0f32; DIM];
            for (i, x) in v.iter_mut().enumerate() {
                *x = if i % 2 == 0 { 1e4 } else { -1e4 };
            }
            v
        },
    ];

    let gamma = vec![1.0f32; DIM];
    let beta = vec![0.0f32; DIM];
    let config =
        LayerNormConfig { normalized_shape: vec![DIM], eps: 1e-5, elementwise_affine: true };

    for (idx, input) in extreme_inputs.iter().enumerate() {
        let out = layer_norm(input, &gamma, Some(&beta), &config).expect("layer_norm failed");
        assert_eq!(out.len(), DIM);
        assert_no_nan_inf(&out, &format!("layer_norm extreme case {idx}"));
    }
}

/// Apply softmax to random logits and verify that top-k selection
/// preserves ordering and probability invariants.
#[test]
fn test_softmax_then_top_k_selection() {
    const VOCAB: usize = 256;
    const K: usize = 10;
    let mut rng = Xorshift32(7);

    let mut logits = vec![0.0f32; VOCAB];
    rng.fill(&mut logits);
    // Plant a clear winner so top-k is deterministic
    logits[42] = 10.0;

    // Full softmax first — sanity check
    let mut full_probs = vec![0.0f32; VOCAB];
    softmax_f32(&logits, &mut full_probs).expect("softmax_f32 failed");

    let full_sum: f32 = full_probs.iter().sum();
    assert!((full_sum - 1.0).abs() < 1e-5, "full softmax sum = {full_sum}");
    assert_no_nan_inf(&full_probs, "full softmax");

    // top-k softmax — only top K logits contribute
    let mut topk_probs = vec![0.0f32; VOCAB];
    softmax_topk(&logits, &mut topk_probs, K).expect("softmax_topk failed");
    assert_no_nan_inf(&topk_probs, "topk softmax");

    // Exactly K values should be non-zero
    let nonzero_count = topk_probs.iter().filter(|&&p| p > 0.0).count();
    assert_eq!(
        nonzero_count, K,
        "expected {K} non-zero values in top-k output, got {nonzero_count}"
    );

    // Those K values should still form a valid distribution
    let topk_sum: f32 = topk_probs.iter().sum();
    assert!((topk_sum - 1.0).abs() < 1e-5, "top-k softmax sum = {topk_sum}");

    // The planted winner (index 42) must be in the top-k
    assert!(topk_probs[42] > 0.0, "planted winner at index 42 should be in top-k");

    // Verify ordering: sort non-zero entries descending and confirm
    // they match the largest full-softmax entries
    let mut topk_indices: Vec<usize> = (0..VOCAB).filter(|&i| topk_probs[i] > 0.0).collect();
    topk_indices.sort_by(|&a, &b| {
        full_probs[b].partial_cmp(&full_probs[a]).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut full_sorted: Vec<usize> = (0..VOCAB).collect();
    full_sorted.sort_by(|&a, &b| {
        full_probs[b].partial_cmp(&full_probs[a]).unwrap_or(std::cmp::Ordering::Equal)
    });

    // The top-k indices should be the same as the top K from full sort
    let expected_topk: Vec<usize> = full_sorted[..K].to_vec();
    let mut actual_topk = topk_indices.clone();
    actual_topk.sort();
    let mut expected_sorted = expected_topk.clone();
    expected_sorted.sort();
    assert_eq!(
        actual_topk, expected_sorted,
        "top-k indices don't match expected top-K from full softmax"
    );
}

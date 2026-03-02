#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::fusion::{
    fused_add_normalize, fused_gelu_linear, fused_rmsnorm_linear, fused_scale_add,
    fused_softmax_mask,
};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct FusionInput {
    n_hint: u8,
    out_dim_hint: u8,
    eps_raw: [u8; 4],
    scale_raw: [u8; 4],
    data_a: Vec<u8>,
    data_b: Vec<u8>,
    data_w: Vec<u8>,
    data_gamma: Vec<u8>,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    data.chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn bytes_to_f32_single(b: &[u8; 4]) -> f32 {
    f32::from_le_bytes(*b)
}

fuzz_target!(|input: FusionInput| {
    let n = (input.n_hint as usize % 32) + 1;
    let out_dim = (input.out_dim_hint as usize % 8) + 1;

    let a = bytes_to_f32(&input.data_a, n);
    let b = bytes_to_f32(&input.data_b, n);
    let gamma = bytes_to_f32(&input.data_gamma, n);
    let weight = bytes_to_f32(&input.data_w, out_dim * n);

    if a.len() < n || b.len() < n {
        return;
    }
    if !a[..n].iter().chain(b[..n].iter()).all(|x| x.is_finite()) {
        return;
    }

    // --- fused_scale_add ---
    {
        let scale = bytes_to_f32_single(&input.scale_raw);
        if scale.is_finite() {
            if let Ok(out) = fused_scale_add(&a[..n], &b[..n], scale) {
                assert_eq!(out.len(), n);
                for (i, &v) in out.iter().enumerate() {
                    assert!(v.is_finite(), "fused_scale_add non-finite at {i}: {v}");
                }
            }
        }
    }

    // --- fused_add_normalize ---
    if gamma.len() >= n && gamma[..n].iter().all(|x| x.is_finite()) {
        let eps = bytes_to_f32_single(&input.eps_raw).abs();
        if eps.is_finite() && eps > 0.0 {
            if let Ok(out) = fused_add_normalize(&a[..n], &b[..n], &gamma[..n], eps) {
                assert_eq!(out.len(), n);
            }
        }
    }

    // --- fused_softmax_mask ---
    {
        // Use a and b as scores and mask
        if let Ok(out) = fused_softmax_mask(&a[..n], &b[..n], 1.0) {
            assert_eq!(out.len(), n);
            // Softmax output should sum to ~1.0
            let sum: f32 = out.iter().sum();
            if sum.is_finite() {
                assert!(
                    (sum - 1.0).abs() < 1e-3 || sum == 0.0,
                    "softmax sum = {sum}, expected ~1.0"
                );
            }
            // All outputs should be >= 0
            for (i, &v) in out.iter().enumerate() {
                assert!(v >= 0.0, "softmax output[{i}] = {v} < 0");
            }
        }
    }

    // --- fused_rmsnorm_linear ---
    if weight.len() >= out_dim * n
        && gamma.len() >= n
        && gamma[..n].iter().all(|x| x.is_finite())
        && weight[..out_dim * n].iter().all(|x| x.is_finite())
    {
        let eps = bytes_to_f32_single(&input.eps_raw).abs();
        if eps.is_finite() && eps > 0.0 {
            if let Ok(out) = fused_rmsnorm_linear(&a[..n], &weight[..out_dim * n], &gamma[..n], eps)
            {
                assert_eq!(out.len(), out_dim);
            }
        }
    }

    // --- fused_gelu_linear ---
    if weight.len() >= out_dim * n && weight[..out_dim * n].iter().all(|x| x.is_finite()) {
        // Without bias
        if let Ok(out) = fused_gelu_linear(&a[..n], &weight[..out_dim * n], &[]) {
            assert_eq!(out.len(), out_dim);
        }
        // With bias
        let bias = bytes_to_f32(&input.data_b, out_dim);
        if bias.len() >= out_dim && bias[..out_dim].iter().all(|x| x.is_finite()) {
            if let Ok(out) = fused_gelu_linear(&a[..n], &weight[..out_dim * n], &bias[..out_dim]) {
                assert_eq!(out.len(), out_dim);
            }
        }
    }

    // --- empty input should error ---
    {
        let empty: &[f32] = &[];
        assert!(fused_scale_add(empty, empty, 1.0).is_err());
        assert!(fused_softmax_mask(empty, empty, 1.0).is_err());
    }
});

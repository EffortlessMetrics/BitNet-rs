#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::fusion::{
    fused_add_normalize, fused_gelu_linear, fused_rmsnorm_linear, fused_scale_add,
    fused_softmax_mask,
};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct FusionInput {
    data_a: Vec<u8>,
    data_b: Vec<u8>,
    data_gamma: Vec<u8>,
    eps_byte: u8,
    scale_byte: u8,
    op: u8,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned]
        .chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn clamp_finite(v: &mut [f32]) {
    for x in v.iter_mut() {
        if !x.is_finite() {
            *x = 0.0;
        }
    }
}

fuzz_target!(|input: FusionInput| {
    let mut a = bytes_to_f32(&input.data_a, 256);
    let mut b = bytes_to_f32(&input.data_b, 256);
    let mut gamma = bytes_to_f32(&input.data_gamma, 256);

    if a.is_empty() {
        return;
    }

    // Clamp to finite to focus on logic bugs, not NaN propagation
    clamp_finite(&mut a);
    clamp_finite(&mut b);
    clamp_finite(&mut gamma);

    let eps = (input.eps_byte as f32 / 255.0) * 0.1 + 1e-8;
    let scale = (input.scale_byte as f32 / 255.0) * 4.0 - 2.0;
    let n = a.len();

    match input.op % 5 {
        0 => {
            // fused_rmsnorm_linear: input, weight, gamma must be same length
            b.resize(n, 0.0);
            gamma.resize(n, 1.0);
            let _ = fused_rmsnorm_linear(&a, &b, &gamma, eps);
        }
        1 => {
            // fused_gelu_linear: input, weight, bias must be same length
            b.resize(n, 0.0);
            gamma.resize(n, 0.0);
            let _ = fused_gelu_linear(&a, &b, &gamma);
        }
        2 => {
            // fused_softmax_mask: scores and mask must be same length
            b.resize(n, 0.0);
            let _ = fused_softmax_mask(&a, &b, scale);
        }
        3 => {
            // fused_add_normalize: a, b, gamma must be same length
            b.resize(n, 0.0);
            gamma.resize(n, 1.0);
            let _ = fused_add_normalize(&a, &b, &gamma, eps);
        }
        4 => {
            // fused_scale_add: a and b must be same length
            b.resize(n, 0.0);
            let _ = fused_scale_add(&a, &b, scale);
        }
        _ => unreachable!(),
    }
});

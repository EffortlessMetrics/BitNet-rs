#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::activations::{
    ActivationType, activate, activate_inplace, elu, gelu, gelu_tanh, gelu_vec, hard_sigmoid,
    hard_swish, leaky_relu, mish, mish_vec, quick_gelu, relu, selu, sigmoid, silu, silu_vec,
    softplus, swish, tanh_act,
};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct ActivationInput {
    data: Vec<u8>,
    alpha_byte: u8,
    beta_byte: u8,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned]
        .chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fuzz_target!(|input: ActivationInput| {
    let values = bytes_to_f32(&input.data, 256);
    if values.is_empty() {
        return;
    }

    let alpha = (input.alpha_byte as f32 / 255.0) * 2.0 - 1.0;
    let beta = (input.beta_byte as f32 / 255.0) * 4.0;

    // Exercise every scalar activation on each value
    for &x in &values {
        let _ = relu(x);
        let _ = leaky_relu(x, alpha);
        let _ = sigmoid(x);
        let _ = tanh_act(x);
        let _ = gelu(x);
        let _ = gelu_tanh(x);
        let _ = silu(x);
        let _ = swish(x, beta);
        let _ = hard_sigmoid(x);
        let _ = hard_swish(x);
        let _ = softplus(x);
        let _ = mish(x);
        let _ = elu(x, alpha);
        let _ = selu(x);
        let _ = quick_gelu(x);
    }

    // Exercise vectorized paths
    let _ = gelu_vec(&values);
    let _ = silu_vec(&values);
    let _ = mish_vec(&values);

    // Exercise activate / activate_inplace with each ActivationType
    let types = [
        ActivationType::ReLU,
        ActivationType::LeakyReLU(alpha),
        ActivationType::GELU,
        ActivationType::GELUTanh,
        ActivationType::SiLU,
        ActivationType::Swish(beta),
        ActivationType::Sigmoid,
        ActivationType::Tanh,
        ActivationType::HardSigmoid,
        ActivationType::HardSwish,
        ActivationType::Mish,
        ActivationType::Softplus,
        ActivationType::ELU(alpha),
        ActivationType::SELU,
        ActivationType::QuickGELU,
    ];

    for act in &types {
        let out = activate(&values, *act);
        assert_eq!(out.len(), values.len());

        // In-place variant should produce same results
        let mut buf = values.clone();
        activate_inplace(&mut buf, *act);
        for (a, b) in out.iter().zip(buf.iter()) {
            assert!(
                (a - b).abs() < 1e-6 || (a.is_nan() && b.is_nan()),
                "activate vs activate_inplace mismatch for {act:?}: {a} vs {b}"
            );
        }
    }

    // Invariant: ReLU output is non-negative for finite input
    let relu_out = activate(&values, ActivationType::ReLU);
    for (i, &v) in relu_out.iter().enumerate() {
        if v.is_finite() {
            assert!(v >= 0.0, "ReLU produced negative output {v} at index {i}");
        }
    }

    // Invariant: Sigmoid output is in [0, 1] for finite input
    let sig_out = activate(&values, ActivationType::Sigmoid);
    for (i, &v) in sig_out.iter().enumerate() {
        if v.is_finite() {
            assert!((-1e-6..=1.0 + 1e-6).contains(&v), "Sigmoid out of range {v} at index {i}");
        }
    }
});

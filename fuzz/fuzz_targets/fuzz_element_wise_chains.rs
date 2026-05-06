#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::activations::{ActivationType, activate, activate_inplace};
use bitnet_kernels::cpu::fusion::{fused_add_normalize, fused_scale_add};
use bitnet_kernels::cpu::residual::{add_residual, add_residual_scaled};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct ElementWiseChainInput {
    data_a: Vec<u8>,
    data_b: Vec<u8>,
    dim: u8,
    chain_ops: Vec<u8>,
    scale_byte: u8,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned]
        .chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn select_activation(sel: u8) -> ActivationType {
    match sel % 10 {
        0 => ActivationType::ReLU,
        1 => ActivationType::GELU,
        2 => ActivationType::SiLU,
        3 => ActivationType::Sigmoid,
        4 => ActivationType::Tanh,
        5 => ActivationType::HardSigmoid,
        6 => ActivationType::HardSwish,
        7 => ActivationType::Mish,
        8 => ActivationType::Softplus,
        9 => ActivationType::SELU,
        _ => unreachable!(),
    }
}

fuzz_target!(|input: ElementWiseChainInput| {
    let dim = (input.dim as usize % 64) + 2;

    let a_raw = bytes_to_f32(&input.data_a, dim);
    let b_raw = bytes_to_f32(&input.data_b, dim);
    if a_raw.len() < dim || b_raw.len() < dim {
        return;
    }
    let a = &a_raw[..dim];
    let b = &b_raw[..dim];

    // Skip non-finite inputs.
    if a.iter().chain(b.iter()).any(|x| !x.is_finite()) {
        return;
    }

    let scale = (input.scale_byte as f32 / 255.0) * 4.0 - 2.0;

    let ops = if input.chain_ops.is_empty() { &[0u8][..] } else { &input.chain_ops[..] };

    let mut current = a.to_vec();

    for &op_sel in ops.iter().take(8) {
        match op_sel % 5 {
            0 => {
                // Activation function (element-wise).
                let act = select_activation(op_sel / 5);
                current = activate(&current, act);
                for (i, &v) in current.iter().enumerate() {
                    assert!(!v.is_nan(), "activation {act:?} NaN at {i}");
                }
            }
            1 => {
                // Residual add.
                let mut buf = current.clone();
                if buf.len() == b.len() && add_residual(&mut buf, b).is_ok() {
                    current = buf;
                }
            }
            2 => {
                // Scaled residual add.
                let mut buf = current.clone();
                if buf.len() == b.len()
                    && scale.is_finite()
                    && add_residual_scaled(&mut buf, b, scale).is_ok()
                {
                    current = buf;
                }
            }
            3 => {
                // Fused scale+add.
                if current.len() == b.len()
                    && scale.is_finite()
                    && let Ok(out) = fused_scale_add(&current, b, scale)
                {
                    current = out;
                }
            }
            4 => {
                // Fused add+normalize.
                if current.len() == b.len() && current.len() >= 2 {
                    let gamma = vec![1.0f32; current.len()];
                    if let Ok(out) = fused_add_normalize(&current, b, &gamma, 1e-5) {
                        current = out;
                    }
                }
            }
            _ => unreachable!(),
        }

        // Bail out early if values become non-finite to avoid cascading issues.
        if current.iter().any(|x| !x.is_finite()) {
            return;
        }
    }

    // Invariant: output length must match input length.
    assert_eq!(current.len(), dim, "chain changed output length");

    // Invariant: activate and activate_inplace must agree.
    let act = select_activation(ops[0]);
    let alloc_out = activate(a, act);
    let mut inplace_out = a.to_vec();
    activate_inplace(&mut inplace_out, act);
    for (i, (&al, &ip)) in alloc_out.iter().zip(inplace_out.iter()).enumerate() {
        assert!(
            (al - ip).abs() < 1e-6
                || (al.is_nan() && ip.is_nan())
                || (al.is_infinite() && ip.is_infinite() && al.signum() == ip.signum()),
            "activate vs inplace mismatch at {i}: {al} vs {ip}"
        );
    }
});

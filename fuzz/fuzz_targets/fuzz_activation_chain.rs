#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::activations::{ActivationType, activate, activate_inplace};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct ActivationChainInput {
    data: Vec<u8>,
    /// Each byte selects an activation function.
    chain: Vec<u8>,
    leaky_alpha: u8,
    elu_alpha: u8,
    swish_beta: u8,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned]
        .chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn select_activation(
    selector: u8,
    leaky_alpha: f32,
    elu_alpha: f32,
    swish_beta: f32,
) -> ActivationType {
    match selector % 15 {
        0 => ActivationType::ReLU,
        1 => ActivationType::LeakyReLU(leaky_alpha),
        2 => ActivationType::GELU,
        3 => ActivationType::GELUTanh,
        4 => ActivationType::SiLU,
        5 => ActivationType::Swish(swish_beta),
        6 => ActivationType::Sigmoid,
        7 => ActivationType::Tanh,
        8 => ActivationType::HardSigmoid,
        9 => ActivationType::HardSwish,
        10 => ActivationType::Mish,
        11 => ActivationType::Softplus,
        12 => ActivationType::ELU(elu_alpha),
        13 => ActivationType::SELU,
        14 => ActivationType::QuickGELU,
        _ => unreachable!(),
    }
}

fuzz_target!(|input: ActivationChainInput| {
    let mut values = bytes_to_f32(&input.data, 128);
    if values.is_empty() {
        return;
    }

    // Filter to finite values.
    values.retain(|x| x.is_finite());
    if values.is_empty() {
        return;
    }

    let leaky_alpha = (input.leaky_alpha as f32 / 255.0) * 0.5;
    let elu_alpha = (input.elu_alpha as f32 / 255.0) * 2.0 + 0.01;
    let swish_beta = (input.swish_beta as f32 / 255.0) * 3.0 + 0.1;

    let chain_len = input.chain.len().min(6).max(1);

    // Apply chain of activations using the allocating API.
    let mut current = values.clone();
    for &sel in input.chain.iter().take(chain_len) {
        let act = select_activation(sel, leaky_alpha, elu_alpha, swish_beta);
        current = activate(&current, act);

        // After each step, verify no NaN.
        for (i, &val) in current.iter().enumerate() {
            assert!(!val.is_nan(), "NaN after activation {act:?} at index {i}");
        }
    }

    // The final output should have the same length as the input.
    assert_eq!(current.len(), values.len(), "chain changed output length");

    // Apply chain of activations using the in-place API.
    let mut inplace = values.clone();
    for &sel in input.chain.iter().take(chain_len) {
        let act = select_activation(sel, leaky_alpha, elu_alpha, swish_beta);
        activate_inplace(&mut inplace, act);

        for (i, &val) in inplace.iter().enumerate() {
            assert!(!val.is_nan(), "NaN (inplace) after {act:?} at index {i}");
        }
    }

    // Both paths must produce identical results.
    assert_eq!(current.len(), inplace.len(), "allocating vs inplace length mismatch");
    for (i, (&a, &b)) in current.iter().zip(inplace.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6
                || (a.is_infinite() && b.is_infinite() && a.signum() == b.signum()),
            "allocating vs inplace divergence at {i}: {a} vs {b}"
        );
    }

    // Invariant: bounded activations (Sigmoid, Tanh, HardSigmoid) stay in range.
    for &sel in &[6u8, 7, 8] {
        let act = select_activation(sel, leaky_alpha, elu_alpha, swish_beta);
        let out = activate(&values, act);
        let (lo, hi) = match sel {
            6 => (0.0f32, 1.0),  // Sigmoid
            7 => (-1.0f32, 1.0), // Tanh
            8 => (0.0f32, 1.0),  // HardSigmoid
            _ => unreachable!(),
        };
        for (i, &val) in out.iter().enumerate() {
            if val.is_finite() {
                assert!(
                    val >= lo - 1e-6 && val <= hi + 1e-6,
                    "{act:?} out of range [{lo}, {hi}] at {i}: {val}"
                );
            }
        }
    }
});

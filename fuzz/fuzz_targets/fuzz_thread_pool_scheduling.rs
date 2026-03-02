#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::batch::{
    batched_add, batched_layer_norm, batched_matmul, batched_softmax,
};
use libfuzzer_sys::fuzz_target;

/// Fuzz work scheduling by exercising batched operations with varying task
/// counts, simulating the work-distribution patterns of a thread pool.
#[derive(Arbitrary, Debug)]
struct ThreadPoolInput {
    batch_size: u8,
    dim: u8,
    inner_dim: u8,
    data: Vec<u8>,
    weights: Vec<u8>,
    gamma_data: Vec<u8>,
    beta_data: Vec<u8>,
    op_sequence: Vec<u8>,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned]
        .chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fuzz_target!(|input: ThreadPoolInput| {
    let batch = (input.batch_size as usize % 8) + 1;
    let dim = (input.dim as usize % 32) + 2;
    let inner = (input.inner_dim as usize % 16) + 2;
    let total = batch * dim;

    let data = bytes_to_f32(&input.data, total);
    if data.len() < total {
        return;
    }
    let data = &data[..total];

    // Skip non-finite inputs.
    if data.iter().any(|x| !x.is_finite()) {
        return;
    }

    // Execute a fuzz-selected sequence of batched operations.
    let ops = if input.op_sequence.is_empty() { &[0u8][..] } else { &input.op_sequence[..] };

    for &op_sel in ops.iter().take(8) {
        match op_sel % 4 {
            0 => {
                // Batched softmax: must not panic, output must be valid probabilities.
                if let Ok(out) = batched_softmax(data, batch, dim) {
                    assert_eq!(out.len(), total);
                    for (i, &v) in out.iter().enumerate() {
                        assert!(!v.is_nan(), "batched_softmax NaN at {i}");
                        assert!(v >= 0.0, "batched_softmax negative at {i}: {v}");
                    }
                    // Each batch row should sum to ~1.0.
                    for bi in 0..batch {
                        let row_sum: f32 = out[bi * dim..(bi + 1) * dim].iter().sum();
                        if row_sum.is_finite() {
                            assert!((row_sum - 1.0).abs() < 1e-3, "softmax row {bi} sum={row_sum}");
                        }
                    }
                }
            }
            1 => {
                // Batched matmul: a=[batch, 1, dim], b=[batch, dim, inner] → [batch, 1, inner].
                let weight_count = batch * dim * inner;
                let weights = bytes_to_f32(&input.weights, weight_count);
                if weights.len() >= weight_count
                    && weights[..weight_count].iter().all(|x| x.is_finite())
                {
                    // a is [batch * 1 * dim], b is [batch * dim * inner]
                    if let Ok(out) =
                        batched_matmul(data, &weights[..weight_count], batch, 1, dim, inner)
                    {
                        assert_eq!(out.len(), batch * inner);
                    }
                }
            }
            2 => {
                // Batched layer norm: must not panic.
                let gamma = bytes_to_f32(&input.gamma_data, dim);
                let beta = bytes_to_f32(&input.beta_data, dim);
                if gamma.len() >= dim
                    && beta.len() >= dim
                    && gamma[..dim].iter().all(|x| x.is_finite())
                    && beta[..dim].iter().all(|x| x.is_finite())
                {
                    if let Ok(out) =
                        batched_layer_norm(data, &gamma[..dim], &beta[..dim], batch, dim, 1e-5)
                    {
                        assert_eq!(out.len(), total);
                        for (i, &v) in out.iter().enumerate() {
                            assert!(!v.is_nan(), "batched_layer_norm NaN at {i}");
                        }
                    }
                }
            }
            3 => {
                // Batched add: must not panic.
                let b_data = bytes_to_f32(&input.weights, total);
                if b_data.len() >= total && b_data[..total].iter().all(|x| x.is_finite()) {
                    if let Ok(out) = batched_add(data, &b_data[..total], batch, dim) {
                        assert_eq!(out.len(), total);
                        for (i, &v) in out.iter().enumerate() {
                            assert!(v.is_finite(), "batched_add non-finite at {i}");
                        }
                    }
                }
            }
            _ => unreachable!(),
        }
    }

    // Invariant: running the same operation twice yields identical results.
    if let Ok(out1) = batched_softmax(data, batch, dim) {
        if let Ok(out2) = batched_softmax(data, batch, dim) {
            for (i, (&a, &b)) in out1.iter().zip(out2.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-7 || (a.is_nan() && b.is_nan()),
                    "determinism violated at {i}: {a} vs {b}"
                );
            }
        }
    }
});

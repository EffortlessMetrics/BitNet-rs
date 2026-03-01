#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct LayerNormInput {
    dim: u8,
    batch_size: u8,
    data: Vec<u8>,
    gamma_data: Vec<u8>,
    beta_data: Vec<u8>,
    eps: u8,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned]
        .chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn layer_norm(input: &[f32], gamma: &[f32], beta: &[f32], dim: usize, eps: f32) -> Vec<f32> {
    let mean: f32 = input.iter().sum::<f32>() / dim as f32;
    let variance: f32 = input.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / dim as f32;
    let inv_std = 1.0 / (variance + eps).sqrt();

    input
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let normalized = (x - mean) * inv_std;
            normalized * gamma[i] + beta[i]
        })
        .collect()
}

fn rms_norm(input: &[f32], gamma: &[f32], dim: usize, eps: f32) -> Vec<f32> {
    let rms: f32 = (input.iter().map(|x| x * x).sum::<f32>() / dim as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;

    input.iter().enumerate().map(|(i, &x)| x * inv_rms * gamma[i]).collect()
}

fuzz_target!(|input: LayerNormInput| {
    let dim = (input.dim as usize % 64) + 2; // min 2 for meaningful stats
    let batch_size = (input.batch_size as usize % 8) + 1;
    let eps_val = 1e-5 * (1.0 + input.eps as f32); // Always positive epsilon

    let all_data = bytes_to_f32(&input.data, batch_size * dim);
    let gamma_raw = bytes_to_f32(&input.gamma_data, dim);
    let beta_raw = bytes_to_f32(&input.beta_data, dim);

    if all_data.len() < batch_size * dim || gamma_raw.len() < dim || beta_raw.len() < dim {
        return;
    }

    let gamma = &gamma_raw[..dim];
    let beta = &beta_raw[..dim];

    // Skip non-finite inputs
    if all_data[..batch_size * dim]
        .iter()
        .chain(gamma.iter())
        .chain(beta.iter())
        .any(|x| !x.is_finite())
    {
        return;
    }

    for b in 0..batch_size {
        let row = &all_data[b * dim..(b + 1) * dim];

        // --- LayerNorm ---
        let ln_output = layer_norm(row, gamma, beta, dim, eps_val);

        // Invariant 1: Output dimension matches input
        assert_eq!(
            ln_output.len(),
            dim,
            "layer_norm output dim mismatch: expected {dim}, got {}",
            ln_output.len()
        );

        // Invariant 2: No NaN/Inf in output
        for (i, &val) in ln_output.iter().enumerate() {
            assert!(val.is_finite(), "layer_norm non-finite at batch={b} idx={i}: {val}");
        }

        // Invariant 3: With gamma=1 and beta=0, output has ~zero mean and ~unit variance
        let ones = vec![1.0f32; dim];
        let zeros = vec![0.0f32; dim];
        let normalized = layer_norm(row, &ones, &zeros, dim, eps_val);

        let mean: f32 = normalized.iter().sum::<f32>() / dim as f32;
        assert!(
            mean.abs() < 1e-3,
            "normalized mean should be ~0, got {mean} (batch={b}, dim={dim})"
        );

        if dim >= 4 {
            let var: f32 =
                normalized.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / dim as f32;
            assert!(
                (var - 1.0).abs() < 0.1,
                "normalized variance should be ~1.0, got {var} (batch={b}, dim={dim})"
            );
        }

        // --- RMSNorm ---
        let rms_output = rms_norm(row, gamma, dim, eps_val);

        // Invariant 4: RMSNorm output has correct dimension
        assert_eq!(
            rms_output.len(),
            dim,
            "rms_norm output dim mismatch: expected {dim}, got {}",
            rms_output.len()
        );

        // Invariant 5: RMSNorm output is finite
        for (i, &val) in rms_output.iter().enumerate() {
            assert!(val.is_finite(), "rms_norm non-finite at batch={b} idx={i}: {val}");
        }
    }

    // Invariant 6: Constant input produces constant output (after normalization)
    let constant_input = vec![3.14f32; dim];
    let ones = vec![1.0f32; dim];
    let zeros = vec![0.0f32; dim];
    let const_ln = layer_norm(&constant_input, &ones, &zeros, dim, eps_val);
    let first = const_ln[0];
    for (i, &val) in const_ln.iter().enumerate() {
        let diff = (val - first).abs();
        assert!(
            diff < 1e-5,
            "constant input should produce constant output, idx={i}: {first} vs {val}"
        );
    }
});

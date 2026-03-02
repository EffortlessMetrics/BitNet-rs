#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::ffn::{FfnActivation, FfnConfig, ffn_forward, gated_ffn_forward};
use libfuzzer_sys::fuzz_target;

/// Fuzz FFN forward passes with arbitrary dimensions and weights.
#[derive(Arbitrary, Debug)]
struct FfnInput {
    /// Hidden dimension selector (clamped to reasonable range).
    hidden_dim: u8,
    /// Intermediate dimension selector.
    intermediate_dim: u8,
    /// Activation selector (mod 3).
    activation: u8,
    /// Whether to use gated variant.
    gated: bool,
    /// Raw weight bytes.
    raw_weights: Vec<u8>,
}

fn bytes_to_f32(raw: &[u8], count: usize) -> Vec<f32> {
    let aligned = (raw.len() / 4) * 4;
    let mut out: Vec<f32> = raw[..aligned]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    out.resize(count, 0.0);
    out.truncate(count);
    out
}

fuzz_target!(|input: FfnInput| {
    // Clamp dimensions to avoid huge allocations but exercise real code paths.
    let hidden = (input.hidden_dim as usize % 32) + 1;
    let inter = (input.intermediate_dim as usize % 32) + 1;
    let activation = match input.activation % 3 {
        0 => FfnActivation::GeLU,
        1 => FfnActivation::SiLU,
        _ => FfnActivation::ReLU,
    };

    let config = match FfnConfig::new(hidden, inter, activation) {
        Ok(c) => c,
        Err(_) => return,
    };

    let x = bytes_to_f32(&input.raw_weights, hidden);
    let w_up_len = inter * hidden;
    let w_down_len = hidden * inter;

    // Slice remaining bytes for weights.
    let offset = hidden * 4;
    let remaining =
        if input.raw_weights.len() > offset { &input.raw_weights[offset..] } else { &[] };

    if input.gated {
        let total = w_up_len + w_up_len + w_down_len; // gate + up + down
        let all_weights = bytes_to_f32(remaining, total);
        let w_gate = &all_weights[..w_up_len];
        let w_up = &all_weights[w_up_len..w_up_len * 2];
        let w_down = &all_weights[w_up_len * 2..];
        let _ = gated_ffn_forward(&x, w_gate, w_up, w_down, &config);
    } else {
        let total = w_up_len + w_down_len;
        let all_weights = bytes_to_f32(remaining, total);
        let w_up = &all_weights[..w_up_len];
        let w_down = &all_weights[w_up_len..];
        let _ = ffn_forward(&x, w_up, w_down, &config);
    }
});

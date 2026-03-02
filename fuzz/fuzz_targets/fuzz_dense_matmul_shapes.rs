#![no_main]

//! Fuzz DenseLinear forward pass with adversarial shape combinations.
//!
//! Exercises the real `bitnet_inference::dense_forward::DenseLinear` matmul
//! path, including batched inputs, bias, and SwiGLU FFN sub-modules.

use arbitrary::Arbitrary;
use bitnet_inference::dense_forward::DenseLinear;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct DenseMatmulInput {
    /// Batch dimension (clamped).
    batch: u8,
    /// Input features (clamped).
    in_features: u8,
    /// Output features (clamped).
    out_features: u8,
    /// Raw weight bytes reinterpreted as f32.
    weight_bytes: Vec<u8>,
    /// Raw input bytes reinterpreted as f32.
    input_bytes: Vec<u8>,
    /// Whether to include bias.
    use_bias: bool,
    /// Raw bias bytes reinterpreted as f32.
    bias_bytes: Vec<u8>,
}

const MAX_DIM: usize = 64;

fn bytes_to_f32_clamped(bytes: &[u8], count: usize) -> Vec<f32> {
    let aligned = (bytes.len() / 4) * 4;
    let mut result: Vec<f32> = bytes[..aligned]
        .chunks_exact(4)
        .map(|b| {
            let v = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            if v.is_nan() || v.is_infinite() { 0.0 } else { v.clamp(-1e4, 1e4) }
        })
        .collect();
    result.resize(count, 0.0);
    result
}

fuzz_target!(|input: DenseMatmulInput| {
    let batch = (input.batch as usize % 8) + 1;
    let in_f = (input.in_features as usize % MAX_DIM) + 1;
    let out_f = (input.out_features as usize % MAX_DIM) + 1;

    let weight = bytes_to_f32_clamped(&input.weight_bytes, in_f * out_f);
    let x = bytes_to_f32_clamped(&input.input_bytes, batch * in_f);
    let bias =
        if input.use_bias { Some(bytes_to_f32_clamped(&input.bias_bytes, out_f)) } else { None };

    let layer = DenseLinear::new(weight, bias, in_f, out_f);
    let out = layer.forward(&x);

    // Invariant 1: output length = batch * out_features.
    assert_eq!(out.len(), batch * out_f, "output shape mismatch");

    // Invariant 2: finite inputs → finite outputs.
    for (i, &v) in out.iter().enumerate() {
        assert!(v.is_finite(), "non-finite output at index {i}: {v}");
    }

    // Invariant 3: zero input → output equals bias (or zero).
    let zero_x = vec![0.0f32; batch * in_f];
    let zero_out = layer.forward(&zero_x);
    for b in 0..batch {
        for o in 0..out_f {
            let val = zero_out[b * out_f + o];
            if let Some(ref bias) = layer.bias {
                let diff = (val - bias[o]).abs();
                assert!(diff < 1e-5, "zero input: out[{b},{o}]={val} != bias {}", bias[o]);
            } else {
                assert_eq!(val, 0.0, "zero input without bias should be 0.0");
            }
        }
    }
});

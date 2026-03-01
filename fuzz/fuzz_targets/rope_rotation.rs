#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct RopeInput {
    /// Head dimension (will be clamped to even value 2..=64).
    dim: u8,
    /// Sequence position (clamped to 0..=255).
    position: u8,
    /// RoPE base theta.
    base: f32,
    /// Raw vector data (interpreted as f32 pairs).
    data: Vec<u8>,
}

fuzz_target!(|input: RopeInput| {
    use bitnet_rope::build_tables;

    // Clamp dim to a small even number >= 2.
    let dim = (((input.dim as usize) % 32) + 1) * 2; // 2, 4, ..., 64
    let position = input.position as usize;
    let max_seq_len = position + 1;

    // Sanitise base: must be finite and positive.
    let base = input.base.abs();
    if !base.is_finite() || base <= 0.0 {
        return;
    }

    let tables = match build_tables(dim, max_seq_len, base) {
        Ok(t) => t,
        Err(_) => return, // Invalid inputs are expected; no panic = success.
    };

    // Parse fuzz data into an f32 vector of length `dim`.
    let aligned_len = (input.data.len() / 4) * 4;
    let raw: Vec<f32> = input.data[..aligned_len]
        .chunks_exact(4)
        .take(256)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    if raw.len() < dim {
        return;
    }
    let vec = &raw[..dim];

    // Skip vectors with non-finite components (norm is meaningless).
    if vec.iter().any(|x| !x.is_finite()) {
        return;
    }

    let half = tables.half_dim;
    let row_offset = position * half;
    if row_offset + half > tables.sin.len() {
        return;
    }

    let sin_row = &tables.sin[row_offset..row_offset + half];
    let cos_row = &tables.cos[row_offset..row_offset + half];

    // Apply RoPE rotation: for each pair (x_i, x_{i+half}),
    //   out_i       = x_i * cos - x_{i+half} * sin
    //   out_{i+half}= x_i * sin + x_{i+half} * cos
    let mut rotated = vec![0.0f32; dim];
    for i in 0..half {
        let x0 = vec[i];
        let x1 = vec[i + half];
        let s = sin_row[i];
        let c = cos_row[i];
        rotated[i] = x0 * c - x1 * s;
        rotated[i + half] = x0 * s + x1 * c;
    }

    // Invariant 1: No NaN or Inf in output.
    for (i, &v) in rotated.iter().enumerate() {
        assert!(v.is_finite(), "RoPE output non-finite at index {i}: {v}");
    }

    // Invariant 2: Norm preservation — rotation must not change vector magnitude.
    let norm_in: f32 = vec.iter().map(|x| x * x).sum();
    let norm_out: f32 = rotated.iter().map(|x| x * x).sum();

    if norm_in > 1e-12 {
        let ratio = norm_out / norm_in;
        assert!(
            (ratio - 1.0).abs() < 1e-4,
            "Norm not preserved: in={norm_in} out={norm_out} ratio={ratio}"
        );
    }
});

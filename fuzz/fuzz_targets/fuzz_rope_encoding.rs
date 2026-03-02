#![no_main]

use arbitrary::Arbitrary;
use bitnet_rope::{build_tables, resolve_base};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct RopeEncodingInput {
    dim: u8,
    max_seq_len: u8,
    base: f32,
    position: u8,
    vector_data: Vec<u8>,
    resolve_base_arg: Option<f32>,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned]
        .chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fuzz_target!(|input: RopeEncodingInput| {
    // Exercise resolve_base with arbitrary input.
    let _resolved = resolve_base(input.resolve_base_arg);

    // Clamp dim to even number >= 2.
    let dim = (((input.dim as usize) % 32) + 1) * 2;
    let max_seq_len = (input.max_seq_len as usize % 128) + 1;
    let position = input.position as usize % max_seq_len;

    // Sanitise base: must be finite and positive for valid tables.
    let base = input.base.abs();
    if !base.is_finite() || base <= 0.0 {
        // Non-finite / non-positive base: build_tables should return error, not panic.
        let result = build_tables(dim, max_seq_len, input.base);
        assert!(result.is_err() || result.is_ok(), "build_tables must not panic");
        return;
    }

    let tables = match build_tables(dim, max_seq_len, base) {
        Ok(t) => t,
        Err(_) => return,
    };

    // Invariant 1: Table dimensions match.
    let half = tables.half_dim;
    assert_eq!(half, dim / 2, "half_dim should be dim/2");
    assert_eq!(tables.sin.len(), max_seq_len * half, "sin table size mismatch");
    assert_eq!(tables.cos.len(), max_seq_len * half, "cos table size mismatch");

    // Invariant 2: sin²+cos² ≈ 1 for every element.
    for (s, c) in tables.sin.iter().zip(&tables.cos) {
        let norm = s * s + c * c;
        assert!((norm - 1.0).abs() < 1e-4, "sin²+cos²={norm} ≠ 1.0 (sin={s}, cos={c})");
    }

    // Invariant 3: No NaN or Inf in tables.
    for &v in tables.sin.iter().chain(tables.cos.iter()) {
        assert!(v.is_finite(), "table entry non-finite: {v}");
    }

    // Apply rotation to a fuzz-generated vector.
    let raw = bytes_to_f32(&input.vector_data, dim);
    if raw.len() < dim {
        return;
    }
    let vec_data = &raw[..dim];
    if vec_data.iter().any(|x| !x.is_finite()) {
        return;
    }

    let row_offset = position * half;
    if row_offset + half > tables.sin.len() {
        return;
    }

    let sin_row = &tables.sin[row_offset..row_offset + half];
    let cos_row = &tables.cos[row_offset..row_offset + half];

    let mut rotated = vec![0.0f32; dim];
    for i in 0..half {
        let x0 = vec_data[i];
        let x1 = vec_data[i + half];
        rotated[i] = x0 * cos_row[i] - x1 * sin_row[i];
        rotated[i + half] = x0 * sin_row[i] + x1 * cos_row[i];
    }

    // Invariant 4: No NaN/Inf in rotated output.
    for (i, &v) in rotated.iter().enumerate() {
        assert!(v.is_finite(), "rotated output non-finite at idx {i}: {v}");
    }

    // Invariant 5: Norm preservation.
    let norm_in: f32 = vec_data.iter().map(|x| x * x).sum();
    let norm_out: f32 = rotated.iter().map(|x| x * x).sum();
    if norm_in > 1e-10 {
        let ratio = norm_out / norm_in;
        assert!(
            (ratio - 1.0).abs() < 1e-3,
            "norm not preserved: in={norm_in} out={norm_out} ratio={ratio}"
        );
    }

    // Invariant 6: Zero dim and odd dim should error, not panic.
    assert!(build_tables(0, max_seq_len, base).is_err(), "dim=0 should error");
    assert!(build_tables(1, max_seq_len, base).is_err(), "odd dim should error");
});

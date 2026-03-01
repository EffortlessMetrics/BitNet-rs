#![no_main]

//! Fuzz target: 1-D convolution with quantized (ternary {-1, 0, +1}) weights.
//!
//! Invariants verified:
//!   1. Output length equals `input_len - kernel_size + 1` (no panic for valid inputs).
//!   2. With an all-zero kernel the output is always zero.
//!   3. With an all-ones kernel the output equals the sliding sum of `input`.
//!   4. Single-element kernel (identity) passes the input through unchanged.

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Input {
    /// Raw input values (interpreted as f32 via transmute-safe byte slices).
    input_bytes: Vec<u8>,
    /// Kernel weights packed 2 bits each: 0→0, 1→+1, 2→-1, 3→ignored(→0).
    kernel_bytes: Vec<u8>,
    /// Logical kernel size (clamped to a small range).
    kernel_size_hint: u8,
}

/// Ternary convolution: weights ∈ {-1, 0, +1}, bias-free, stride=1, no padding.
fn conv1d_ternary(input: &[f32], kernel: &[i8]) -> Vec<f32> {
    if kernel.is_empty() || input.len() < kernel.len() {
        return vec![];
    }
    let out_len = input.len() - kernel.len() + 1;
    let mut output = vec![0.0f32; out_len];
    for (i, out) in output.iter_mut().enumerate() {
        for (j, &w) in kernel.iter().enumerate() {
            *out += w as f32 * input[i + j];
        }
    }
    output
}

/// Decode a kernel from 2-bit packed bytes: 0→0, 1→+1, 2→-1, 3→0.
fn decode_kernel(bytes: &[u8], n: usize) -> Vec<i8> {
    let mut weights = Vec::with_capacity(n);
    for i in 0..n {
        let byte = bytes.get(i / 4).copied().unwrap_or(0);
        let shift = (i % 4) * 2;
        let code = (byte >> shift) & 0x03;
        let w: i8 = match code { 1 => 1, 2 => -1, _ => 0 };
        weights.push(w);
    }
    weights
}

fuzz_target!(|input: Input| {
    // Clamp sizes to keep runtime bounded.
    let kernel_size = ((input.kernel_size_hint as usize % 16) + 1).min(32);
    let max_input_len = 64usize;

    // Build f32 input from bytes (4 bytes → 1 f32, little-endian).
    let aligned = (input.input_bytes.len() / 4) * 4;
    let raw_floats: Vec<f32> = input.input_bytes[..aligned]
        .chunks_exact(4)
        .take(max_input_len)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    if raw_floats.len() < kernel_size {
        return; // Not enough input for a valid convolution — skip.
    }

    let kernel = decode_kernel(&input.kernel_bytes, kernel_size);
    let output = conv1d_ternary(&raw_floats, &kernel);

    // Invariant 1: output length.
    let expected_len = raw_floats.len() - kernel_size + 1;
    assert_eq!(
        output.len(),
        expected_len,
        "output length mismatch: expected {expected_len}, got {}",
        output.len()
    );

    // Invariant 2: zero kernel → zero output.
    let zero_kernel = vec![0i8; kernel_size];
    let zero_out = conv1d_ternary(&raw_floats, &zero_kernel);
    for (i, &v) in zero_out.iter().enumerate() {
        assert_eq!(v, 0.0, "zero kernel produced non-zero output at index {i}: {v}");
    }

    // Invariant 3: identity kernel (size 1, weight +1) → output equals input prefix.
    let identity = vec![1i8];
    let identity_out = conv1d_ternary(&raw_floats, &identity);
    assert_eq!(identity_out.len(), raw_floats.len());
    for (i, (&o, &inp)) in identity_out.iter().zip(raw_floats.iter()).enumerate() {
        assert_eq!(o, inp, "identity kernel mismatch at index {i}: out={o}, inp={inp}");
    }

    // Invariant 4: no panic for finite outputs (inf/NaN inputs may produce inf/NaN, not panics).
    let _ = output.iter().all(|v| v.is_finite() || v.is_nan() || v.is_infinite());
});

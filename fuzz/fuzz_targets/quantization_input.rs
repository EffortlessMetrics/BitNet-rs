#![no_main]

use arbitrary::Arbitrary;
use bitnet_common::QuantizationType;
use bitnet_quantization::{
    Quantize, QuantizedTensor,
    i2s_qk256::{
        I2SQk256NoScale, QK256_BLOCK, QK256_PACKED_BYTES, code_to_f32, gemv_qk256_row,
        unpack_qk256_block,
    },
};
use libfuzzer_sys::fuzz_target;

/// Structured input for I2_S / QK256 dequantization fuzzing.
#[derive(Arbitrary, Debug)]
struct FuzzInput {
    /// Raw packed bytes for a QuantizedTensor (I2_S, 32-element blocks).
    i2s_data: Vec<u8>,
    /// Scale factors accompanying the I2_S data.
    i2s_scales: Vec<f32>,
    /// Shape dimensions (will be validated before use).
    shape: Vec<usize>,
    /// Block size hint (1–256; clamped internally).
    block_size: u8,

    /// Rows for QK256 matrix construction (clamped to a safe range).
    qk256_rows: u8,
    /// Cols for QK256 matrix construction (clamped to a safe range, multiples of 256 checked).
    qk256_cols: u16,
    /// Raw packed bytes for the QK256 matrix.
    qk256_qs: Vec<u8>,

    /// A single 64-byte block for unpack_qk256_block.
    qk256_block: [u8; QK256_PACKED_BYTES],
    /// A 2-bit code byte (0–3) for code_to_f32.
    code_byte: u8,

    /// Input activations for gemv_qk256_row.
    activations: Vec<f32>,
}

fuzz_target!(|input: FuzzInput| {
    // ── Path 1: QuantizedTensor::dequantize for I2_S ─────────────────────────
    // Limit to avoid OOM; individual pieces are checked separately.
    if !input.i2s_data.is_empty()
        && !input.i2s_scales.is_empty()
        && !input.shape.is_empty()
        && input.i2s_data.len() <= 65_536
        && input.i2s_scales.len() <= 4_096
    {
        // Validate shape so it doesn't overflow usize.
        let total: Option<usize> =
            input.shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d));
        if let Some(numel) = total {
            if numel > 0 && numel <= 65_536 {
                let block_size = (input.block_size as usize).max(1).min(256);
                let qt = QuantizedTensor::new_with_params(
                    input.i2s_data.clone(),
                    input.i2s_scales.iter().map(|&s| if s.is_finite() { s } else { 1.0 }).collect(),
                    None,
                    input.shape.clone(),
                    QuantizationType::I2S,
                    block_size,
                );
                // Must not panic; an Err return is acceptable.
                let _ = qt.dequantize();

                // Also exercise TL1 and TL2 paths on the same raw bytes.
                let qt_tl1 = QuantizedTensor::new_with_params(
                    input.i2s_data.clone(),
                    qt.scales.clone(),
                    None,
                    input.shape.clone(),
                    QuantizationType::TL1,
                    block_size,
                );
                let _ = qt_tl1.dequantize();

                let qt_tl2 = QuantizedTensor::new_with_params(
                    input.i2s_data.clone(),
                    qt.scales.clone(),
                    None,
                    input.shape.clone(),
                    QuantizationType::TL2,
                    block_size,
                );
                let _ = qt_tl2.dequantize();
            }
        }
    }

    // ── Path 2: I2SQk256NoScale construction ─────────────────────────────────
    // Use small, safe dimensions to avoid OOM while still covering the API.
    let rows = (input.qk256_rows as usize).max(1).min(64);
    let cols = (input.qk256_cols as usize).max(1).min(1024);

    // Attempt construction with the fuzzed byte slice (may fail with Err).
    let _ = I2SQk256NoScale::new(rows, cols, input.qk256_qs.clone());

    // Also try a correctly-sized buffer so we always reach the internals.
    let blocks_per_row = cols.div_ceil(QK256_BLOCK);
    let expected_bytes = rows * blocks_per_row * QK256_PACKED_BYTES;
    if expected_bytes <= 131_072 {
        // Build a buffer from cyclic repetition of the fuzz bytes (or zeros).
        let mut qs_buf = vec![0u8; expected_bytes];
        if !input.qk256_qs.is_empty() {
            for (dst, src) in qs_buf.iter_mut().zip(input.qk256_qs.iter().cycle()) {
                *dst = *src;
            }
        }
        if let Ok(mat) = I2SQk256NoScale::new(rows, cols, qs_buf) {
            // Exercise row_bytes for each row.
            for r in 0..rows {
                let _ = mat.row_bytes(r);
            }

            // ── Path 3: gemv_qk256_row ────────────────────────────────────────
            let act: Vec<f32> = if input.activations.is_empty() {
                vec![1.0f32; cols]
            } else {
                input
                    .activations
                    .iter()
                    .cycle()
                    .take(cols)
                    .map(|&v| if v.is_finite() { v } else { 0.0 })
                    .collect()
            };
            let row_bytes = mat.row_bytes(0);
            let _ = gemv_qk256_row(row_bytes, &act, cols);
        }
    }

    // ── Path 4: unpack_qk256_block ────────────────────────────────────────────
    let mut out = [0u8; QK256_BLOCK];
    unpack_qk256_block(&input.qk256_block, &mut out);
    // Verify every output code is in range [0, 3].
    for &code in out.iter() {
        debug_assert!(code <= 3, "unpack produced out-of-range code {code}");
    }

    // ── Path 5: code_to_f32 ───────────────────────────────────────────────────
    // Only codes 0–3 are meaningful; exercise all 256 byte values without panicking.
    let _ = code_to_f32(input.code_byte & 0x03);
    let _ = code_to_f32(input.code_byte); // must not panic even for out-of-range
});

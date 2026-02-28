//! Compatibility dispatch wrapper for legacy QK256 GEMV callsites.
//!
//! The actively maintained QK256 implementation lives in `i2s_qk256` and already
//! performs runtime AVX2 detection with a scalar fallback. This module keeps the
//! previous API shape (`qk256_gemv`) while delegating execution to that canonical
//! implementation.

use crate::i2s_qk256;

/// QK256 block size (256 elements per quantized block)
pub const QK256: usize = 256;

/// Enum describing which QK256 back-end is active at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qk256Backend {
    /// AVX2 SIMD implementation.
    Avx2,
    /// Portable scalar fallback.
    Scalar,
}

/// Returns the back-end that `qk256_gemv` / `i2s_qk256::gemv_qk256` would use.
pub fn active_backend() -> Qk256Backend {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return Qk256Backend::Avx2;
        }
    }
    Qk256Backend::Scalar
}

/// Validate QK256 GEMV inputs without running the kernel.
///
/// Returns `Ok(())` if the inputs are consistent, or a descriptive error string.
pub fn validate_qk256_inputs(
    output_len: usize,
    rows: usize,
    cols: usize,
    packed_len: usize,
    scales_len: usize,
    activations_len: usize,
) -> std::result::Result<(), String> {
    if output_len != rows {
        return Err(format!("Output length {output_len} != rows {rows}"));
    }
    if activations_len != cols {
        return Err(format!("Activations length {activations_len} != cols {cols}"));
    }
    if !cols.is_multiple_of(QK256) {
        return Err(format!("Cols {cols} is not a multiple of QK256={QK256}"));
    }
    let blocks_per_row = cols / QK256;
    let expected_packed = rows * cols / 4;
    let expected_scales = rows * blocks_per_row;
    if packed_len != expected_packed {
        return Err(format!(
            "Packed weight size {packed_len} != expected {expected_packed} (rows={rows}, cols={cols})"
        ));
    }
    if scales_len != expected_scales {
        return Err(format!(
            "Scales length {scales_len} != expected {expected_scales} (rows={rows}, blocks_per_row={blocks_per_row})"
        ));
    }
    Ok(())
}

/// Non-panicking variant of [`qk256_gemv`].
///
/// Returns `Err(String)` on dimension mismatches instead of panicking.
pub fn qk256_gemv_checked(
    output: &mut [f32],
    rows: usize,
    cols: usize,
    packed: &[u8],
    scales: &[f32],
    activations: &[f32],
) -> std::result::Result<(), String> {
    validate_qk256_inputs(output.len(), rows, cols, packed.len(), scales.len(), activations.len())?;
    let blocks_per_row = cols / QK256;
    let row_stride_bytes = blocks_per_row * i2s_qk256::QK256_PACKED_BYTES;
    i2s_qk256::gemv_qk256(packed, activations, output, rows, cols, row_stride_bytes)
        .map_err(|e| e.to_string())
}

/// Legacy QK256 GEMV entry-point.
///
/// The legacy signature exposes an explicit `scales` tensor, but the current
/// GGML I2_S (QK256 no-scale) format used in BitNet-rs does not consume it.
///
/// This function validates the legacy input contract and forwards to
/// [`i2s_qk256::gemv_qk256`], which dispatches AVX2/scalar at runtime.
pub fn qk256_gemv(
    output: &mut [f32],
    rows: usize,
    cols: usize,
    packed: &[u8],
    scales: &[f32],
    activations: &[f32],
) {
    assert_eq!(output.len(), rows, "Output length mismatch");
    assert_eq!(activations.len(), cols, "Activation length mismatch");
    assert_eq!(cols % QK256, 0, "Cols must be multiple of QK256={}", QK256);

    let blocks_per_row = cols / QK256;
    let expected_packed_len = rows * cols / 4;
    let expected_scales_len = rows * blocks_per_row;

    assert_eq!(packed.len(), expected_packed_len, "Packed weight size mismatch");
    assert_eq!(scales.len(), expected_scales_len, "Scales length mismatch");

    let row_stride_bytes = blocks_per_row * i2s_qk256::QK256_PACKED_BYTES;

    i2s_qk256::gemv_qk256(packed, activations, output, rows, cols, row_stride_bytes)
        .expect("qk256_gemv: internal dispatch failed after input validation (this is a bug)");
}

/// Scalar legacy QK256 GEMV (kept for benchmark compatibility).
///
/// This preserves the historical 2-bit mapping used by this legacy interface:
/// `00→-1, 01→0, 10→1, 11→-1`, followed by per-block scaling.
pub fn qk256_gemv_scalar(
    output: &mut [f32],
    rows: usize,
    cols: usize,
    packed: &[u8],
    scales: &[f32],
    activations: &[f32],
) {
    assert_eq!(output.len(), rows, "Output length mismatch");
    assert_eq!(activations.len(), cols, "Activation length mismatch");
    assert_eq!(cols % QK256, 0, "Cols must be multiple of QK256={}", QK256);

    let blocks_per_row = cols / QK256;
    let expected_packed_len = rows * cols / 4;
    let expected_scales_len = rows * blocks_per_row;

    assert_eq!(packed.len(), expected_packed_len, "Packed weight size mismatch");
    assert_eq!(scales.len(), expected_scales_len, "Scales length mismatch");

    for (row_idx, output_elem) in output.iter_mut().enumerate().take(rows) {
        let mut row_sum = 0.0f32;

        for block_idx in 0..blocks_per_row {
            let global_block = row_idx * blocks_per_row + block_idx;
            let scale = scales[global_block];
            let byte_offset = global_block * QK256 / 4;
            let act_offset = block_idx * QK256;

            let mut block_sum = 0.0f32;
            for elem in 0..QK256 {
                let byte_idx = byte_offset + elem / 4;
                let bit_shift = (elem % 4) * 2;
                let two_bit = (packed[byte_idx] >> bit_shift) & 0b11;

                let signed_val = match two_bit {
                    0b00 => -1.0f32,
                    0b01 => 0.0f32,
                    0b10 => 1.0f32,
                    0b11 => -1.0f32,
                    _ => unreachable!(),
                };

                block_sum += signed_val * activations[act_offset + elem];
            }

            row_sum += block_sum * scale;
        }

        *output_elem = row_sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qk256_gemv_smoke() {
        let rows = 256;
        let cols = 256;
        let mut output = vec![0.0f32; rows];
        // Code 1 (0b01) maps to -1.0 in i2s_qk256; with activations at 0.5 this
        // yields a stable, non-zero smoke assertion.
        let packed = vec![0x55u8; rows * cols / 4];
        let scales = vec![1.0f32; rows * cols / QK256];
        let activations = vec![0.5f32; cols];

        qk256_gemv(&mut output, rows, cols, &packed, &scales, &activations);

        assert!(output.iter().all(|&x| x == -128.0));
    }

    #[test]
    #[should_panic(expected = "Cols must be multiple of QK256")]
    fn test_qk256_gemv_invalid_cols() {
        let rows = 1;
        let cols = 255;
        let mut output = vec![0.0f32; rows];
        let packed = vec![0u8; rows * (cols / 4)];
        let scales = vec![1.0f32; rows * (cols / QK256)];
        let activations = vec![0.5f32; cols];

        qk256_gemv(&mut output, rows, cols, &packed, &scales, &activations);
    }

    #[test]
    fn test_active_backend_returns_known_variant() {
        let backend = active_backend();
        assert!(backend == Qk256Backend::Avx2 || backend == Qk256Backend::Scalar);
    }

    #[test]
    fn test_validate_qk256_inputs_ok() {
        let rows = 256;
        let cols = 256;
        assert!(validate_qk256_inputs(rows, rows, cols, rows * cols / 4, rows, cols).is_ok());
    }

    #[test]
    fn test_validate_qk256_inputs_bad_cols() {
        let result = validate_qk256_inputs(1, 1, 255, 0, 0, 255);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not a multiple"));
    }

    #[test]
    fn test_qk256_gemv_checked_ok() {
        let rows = 256;
        let cols = 256;
        let mut output = vec![0.0f32; rows];
        let packed = vec![0x55u8; rows * cols / 4];
        let scales = vec![1.0f32; rows * cols / QK256];
        let activations = vec![0.5f32; cols];
        let result = qk256_gemv_checked(&mut output, rows, cols, &packed, &scales, &activations);
        assert!(result.is_ok());
    }

    #[test]
    fn test_qk256_gemv_checked_err() {
        let mut output = vec![0.0f32; 1];
        let result = qk256_gemv_checked(&mut output, 1, 255, &[], &[], &[0.0; 255]);
        assert!(result.is_err());
    }
}

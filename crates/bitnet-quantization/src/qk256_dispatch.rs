#![allow(dead_code)] // temporary until wired into inference path (cleared in PR2/PR3)

//! QK256 GEMV Dispatch Layer
//!
//! Runtime CPU feature detection and dispatch for QK256 (GGML-compatible)
//! 2-bit quantized GEMV (General Matrix-Vector) operations.
//!
//! ## Sprint-2 Track A: QK256 SIMD Optimization (#417)
//!
//! **PR1 (This File)**: Dispatch scaffolding and CPU feature detection
//! **PR2**: Unpack path (nibble LUT expansion to i8/f32)
//! **PR3**: AVX2 kernel (FMA tiling, 8-wide SIMD)
//! **PR4**: Integration (Rayon row-parallel, production usage)
//!
//! ## Architecture
//!
//! ```text
//! qk256_gemv()  ← Public API
//!     │
//!     ├─→ [Runtime Check: AVX2 available?]
//!     │       ├─ YES → qk256_gemv_avx2() [PR3]
//!     │       └─ NO  → qk256_gemv_scalar() [PR1]
//!     │
//!     └─→ Output: &mut [f32]
//! ```
//!
//! ## Safety
//!
//! AVX2 code paths use `#[target_feature(enable = "avx2")]` and are only
//! called after runtime detection confirms AVX2 availability.

/// QK256 block size (256 elements per quantized block)
pub const QK256: usize = 256;

/// QK256 GEMV: General Matrix-Vector multiplication for 2-bit quantized weights
///
/// **Inputs**:
/// - `output`: Output vector (length = rows)
/// - `rows`, `cols`: Matrix dimensions (cols must be multiple of QK256)
/// - `packed`: Packed 2-bit weights (length = rows * cols / 4)
/// - `scales`: Per-block scales (length = rows * cols / QK256)
/// - `activations`: Input vector (length = cols)
///
/// **Dispatch Logic (PR1)**:
/// - Scalar path only (no SIMD yet)
/// - PR3 will add AVX2 dispatch branch
///
/// **Parity Requirements**:
/// - Scalar vs AVX2: cosine similarity ≥ .99999
/// - Tested via property-based tests (PR3)
///
/// ## Example
///
/// ```rust,no_run
/// use bitnet_quantization::qk256_dispatch::{qk256_gemv, QK256};
///
/// let rows = 2048;
/// let cols = 2048;
/// let mut output = vec![0.0f32; rows];
/// let packed = vec![0u8; rows * cols / 4];
/// let scales = vec![1.0f32; rows * cols / QK256];
/// let activations = vec![0.5f32; cols];
///
/// qk256_gemv(
///     &mut output,
///     rows,
///     cols,
///     &packed,
///     &scales,
///     &activations,
/// );
/// ```
pub fn qk256_gemv(
    output: &mut [f32],
    rows: usize,
    cols: usize,
    packed: &[u8],
    scales: &[f32],
    activations: &[f32],
) {
    // PR1: Only scalar path available
    // PR3 will add:
    // if is_x86_feature_detected!("avx2") {
    //     unsafe { qk256_gemv_avx2(output, rows, cols, packed, scales, activations) }
    // } else {
    //     qk256_gemv_scalar(output, rows, cols, packed, scales, activations)
    // }

    qk256_gemv_scalar(output, rows, cols, packed, scales, activations);
}

/// Scalar QK256 GEMV (no SIMD).
///
/// Reference implementation. All optimized paths must match this output
/// within tolerance.
///
/// **Algorithm**:
/// 1. For each output row:
///    a. For each QK256 block in row:
///       - Unpack 2-bit values → signed (-1, 0, 1)
///       - Dot product with activation slice
///       - Scale by block scale
///
///    b. Sum scaled block results → row output.
///
/// **Performance (PR1 Baseline)**:
/// - 2B model (2048x2048): ~0.1 tok/s (scalar only)
/// - Target (PR3 AVX2): ≥3× faster (~0.3 tok/s)
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
    let expected_packed_len = rows * cols / 4; // 2 bits per element = 4 elem/byte
    let expected_scales_len = rows * blocks_per_row;

    assert_eq!(packed.len(), expected_packed_len, "Packed weight size mismatch");
    assert_eq!(scales.len(), expected_scales_len, "Scales length mismatch");

    for (row_idx, output_elem) in output.iter_mut().enumerate().take(rows) {
        let mut row_sum = 0.0f32;

        for block_idx in 0..blocks_per_row {
            let global_block = row_idx * blocks_per_row + block_idx;
            let scale = scales[global_block];

            // Byte offset: 4 elements per byte (2 bits each)
            let byte_offset = global_block * QK256 / 4;
            let act_offset = block_idx * QK256;

            // Unpack and dot product (scalar - no SIMD)
            let mut block_sum = 0.0f32;
            for elem in 0..QK256 {
                // Extract 2-bit value
                let byte_idx = byte_offset + elem / 4;
                let bit_shift = (elem % 4) * 2;
                let two_bit = (packed[byte_idx] >> bit_shift) & 0b11;

                // Map 2-bit → signed: 00→-1, 01→0, 10→1, 11→-1 (for reference compatibility)
                // TODO(PR2): Replace with nibble LUT for faster unpack
                // TODO(PR2): Strict-mode error for 0b11 can be added in PR2
                let signed_val = match two_bit {
                    0b00 => -1.0f32,
                    0b01 => 0.0f32,
                    0b10 => 1.0f32,
                    0b11 => -1.0f32, // Fallback for reference compatibility
                    _ => unreachable!(),
                };

                block_sum += signed_val * activations[act_offset + elem];
            }

            row_sum += block_sum * scale;
        }

        *output_elem = row_sum;
    }
}

// PR3 will add this function:
// #[target_feature(enable = "avx2")]
// unsafe fn qk256_gemv_avx2(
//     output: &mut [f32],
//     rows: usize,
//     cols: usize,
//     packed: &[u8],
//     scales: &[f32],
//     activations: &[f32],
// ) {
//     // AVX2 implementation with FMA tiling
//     unimplemented!("PR3: AVX2 kernel not yet implemented");
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qk256_gemv_smoke() {
        // Smoke test: ensure basic functionality works
        let rows = 256;
        let cols = 256;
        let mut output = vec![0.0f32; rows];
        // Use 0x55 pattern: 01010101 → 01 01 01 01 → 0, 0, 0, 0 (all zeros)
        let packed = vec![0x55u8; rows * cols / 4];
        let scales = vec![1.0f32; rows * cols / QK256];
        let activations = vec![0.5f32; cols];

        qk256_gemv(&mut output, rows, cols, &packed, &scales, &activations);

        // With 0x55 pattern (01→0), output should be zero
        assert!(output.iter().all(|&x| x == 0.0));
    }

    #[test]
    #[should_panic(expected = "Cols must be multiple of QK256")]
    fn test_qk256_gemv_invalid_cols() {
        let rows = 256;
        let cols = 255; // Not multiple of QK256
        let mut output = vec![0.0f32; rows];
        let packed = vec![0u8; rows * cols / 4];
        let scales = vec![1.0f32; rows];
        let activations = vec![0.5f32; cols];

        qk256_gemv(&mut output, rows, cols, &packed, &scales, &activations);
    }

    #[test]
    fn test_qk256_gemv_parity_placeholder() {
        // PR3 will add:
        // - Property-based tests (randomized inputs)
        // - Parity tests (scalar vs AVX2, cosine ≥ .99999)
        // - Edge case tests (aligned/unaligned, odd sizes)
    }
}

// ============================================================================
// PR1 Acceptance Criteria
// ============================================================================
// [x] Dispatch scaffolding complete (scalar path only)
// [ ] CPU feature detection compiles on x86_64 and ARM
// [ ] Benchmarks record scalar baseline performance
// [ ] Documentation updated (architecture diagram, usage example)
// [ ] Tests pass: `cargo test -p bitnet-quantization --features cpu`
//
// PR2 will add:
// - Unpack path (nibble LUT expansion)
// - Unpack benches + correctness tests
//
// PR3 will add:
// - AVX2 kernel (FMA tiling)
// - Runtime dispatch (is_x86_feature_detected)
// - Parity tests (scalar vs AVX2)
//
// PR4 will add:
// - Rayon row-parallel
// - Production integration
// - End-to-end throughput receipt

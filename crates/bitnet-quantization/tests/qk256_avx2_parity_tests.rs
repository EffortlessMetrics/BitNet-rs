//! AVX2-vs-scalar parity tests for QK256 GEMV operations.
//!
//! Validates that the AVX2 dispatch path produces results identical
//! (within tolerance) to the scalar path for various input patterns.

#![cfg(all(test, feature = "cpu"))]

use bitnet_quantization::i2s_qk256::{
    QK256_BLOCK, QK256_PACKED_BYTES, code_to_f32, gemv_qk256, gemv_qk256_row, unpack_qk256_block,
};
use bitnet_quantization::i2s_qk256_avx2::gemv_qk256_avx2;

/// Pack 256 2-bit codes into 64 bytes (4 codes per byte, LSB first).
fn pack_codes(codes: &[u8; 256]) -> [u8; 64] {
    let mut packed = [0u8; 64];
    for i in 0..256 {
        packed[i / 4] |= (codes[i] & 0x03) << ((i % 4) * 2);
    }
    packed
}

/// Build contiguous row-major quantized data for multi-row tests.
fn build_qs_data(row_codes: &[Vec<u8>], cols: usize) -> (Vec<u8>, usize) {
    let blocks_per_row = cols.div_ceil(QK256_BLOCK);
    let row_stride = blocks_per_row * QK256_PACKED_BYTES;
    let mut qs = vec![0u8; row_codes.len() * row_stride];
    for (r, codes) in row_codes.iter().enumerate() {
        for blk in 0..blocks_per_row {
            let mut block_codes = [0u8; 256];
            for j in 0..QK256_BLOCK {
                let col = blk * QK256_BLOCK + j;
                if col < cols {
                    block_codes[j] = codes[col] & 0x03;
                }
            }
            let packed = pack_codes(&block_codes);
            let off = r * row_stride + blk * QK256_PACKED_BYTES;
            qs[off..off + QK256_PACKED_BYTES].copy_from_slice(&packed);
        }
    }
    (qs, row_stride)
}

/// Compute expected dot product manually from codes and x.
fn manual_dot(codes: &[u8], x: &[f32], cols: usize) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..cols {
        acc += code_to_f32(codes[i] & 0x03) * x[i];
    }
    acc
}

/// Compare scalar `gemv_qk256_row` against dispatched `gemv_qk256`
/// and (on x86_64 with AVX2) the explicit AVX2 path.
fn assert_parity(
    qs_data: &[u8],
    x: &[f32],
    rows: usize,
    cols: usize,
    row_stride: usize,
    tolerance: f32,
) {
    // Scalar per-row results
    let y_scalar: Vec<f32> = (0..rows)
        .map(|r| {
            let off = r * row_stride;
            gemv_qk256_row(&qs_data[off..off + row_stride], x, cols)
        })
        .collect();

    // Dispatched path (auto-selects AVX2 or scalar)
    let mut y_dispatch = vec![0.0f32; rows];
    gemv_qk256(qs_data, x, &mut y_dispatch, rows, cols, row_stride).expect("gemv_qk256 dispatch");
    for (i, (s, d)) in y_scalar.iter().zip(y_dispatch.iter()).enumerate() {
        assert!(
            (s - d).abs() < tolerance,
            "row {i}: scalar={s} dispatch={d} diff={}",
            (s - d).abs()
        );
    }

    // Explicit AVX2 path (only on x86_64 with runtime detection)
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        let mut y_avx2 = vec![0.0f32; rows];
        gemv_qk256_avx2(qs_data, x, &mut y_avx2, rows, cols, row_stride).expect("avx2 gemv");
        for (i, (s, a)) in y_scalar.iter().zip(y_avx2.iter()).enumerate() {
            assert!(
                (s - a).abs() < tolerance,
                "row {i}: scalar={s} avx2={a} diff={}",
                (s - a).abs()
            );
        }
    }
}

// ── Test 1: single row, all codes=2 (+1), uniform x=1.0 ──

#[test]
fn test_gemv_single_row_all_ones() {
    let cols = 256;
    let codes = vec![2u8; cols]; // code 2 → +1.0
    let x = vec![1.0f32; cols];
    let (qs, stride) = build_qs_data(&[codes], cols);

    let scalar = gemv_qk256_row(&qs, &x, cols);
    assert!((scalar - cols as f32).abs() < 1e-3, "expected {}, got {scalar}", cols as f32);
    assert_parity(&qs, &x, 1, cols, stride, 1e-5);
}

// ── Test 2: single row, all codes=1 (-1), uniform x=1.0 ──

#[test]
fn test_gemv_single_row_all_neg_ones() {
    let cols = 256;
    let codes = vec![1u8; cols]; // code 1 → -1.0
    let x = vec![1.0f32; cols];
    let (qs, stride) = build_qs_data(&[codes], cols);

    let scalar = gemv_qk256_row(&qs, &x, cols);
    assert!((scalar - (-(cols as f32))).abs() < 1e-3, "expected {}, got {scalar}", -(cols as f32));
    assert_parity(&qs, &x, 1, cols, stride, 1e-5);
}

// ── Test 3: single row, alternating codes [0,1,2,3], ramp x ──

#[test]
fn test_gemv_single_row_mixed_codes() {
    let cols = 256;
    let codes: Vec<u8> = (0..cols).map(|i| (i % 4) as u8).collect();
    let x: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.01).collect();
    let (qs, stride) = build_qs_data(&[codes.clone()], cols);

    let expected = manual_dot(&codes, &x, cols);
    let scalar = gemv_qk256_row(&qs, &x, cols);
    assert!((scalar - expected).abs() < 1e-3, "expected {expected}, got {scalar}");
    assert_parity(&qs, &x, 1, cols, stride, 1e-5);
}

// ── Test 4: 4 rows × 256 cols, random-ish patterns ──

#[test]
fn test_gemv_multi_row_parity() {
    let cols = 256;
    let rows = 4;
    let row_codes: Vec<Vec<u8>> =
        (0..rows).map(|r| (0..cols).map(|c| ((r * 7 + c * 13 + 5) % 4) as u8).collect()).collect();
    let x: Vec<f32> = (0..cols).map(|i| ((i as f32) - 128.0) * 0.1).collect();
    let (qs, stride) = build_qs_data(&row_codes, cols);

    assert_parity(&qs, &x, rows, cols, stride, 1e-3);
}

// ── Test 5: 16 rows × 512 cols (2 blocks per row) ──

#[test]
fn test_gemv_multi_row_large() {
    let cols = 512;
    let rows = 16;
    let row_codes: Vec<Vec<u8>> =
        (0..rows).map(|r| (0..cols).map(|c| ((r * 3 + c * 11 + 1) % 4) as u8).collect()).collect();
    let x: Vec<f32> = (0..cols).map(|i| (i as f32).sin()).collect();
    let (qs, stride) = build_qs_data(&row_codes, cols);

    assert_parity(&qs, &x, rows, cols, stride, 1e-3);
}

// ── Test 6: cols not a multiple of 256 (tail handling) ──

#[test]
fn test_gemv_unaligned_cols() {
    let cols = 300; // not a multiple of 256
    let rows = 2;
    let row_codes: Vec<Vec<u8>> =
        (0..rows).map(|r| (0..cols).map(|c| ((r + c * 7) % 4) as u8).collect()).collect();
    let x: Vec<f32> = (0..cols).map(|i| i as f32 * 0.01).collect();
    let (qs, stride) = build_qs_data(&row_codes, cols);

    assert_parity(&qs, &x, rows, cols, stride, 1e-3);
}

// ── Test 7: single row — gemv_qk256_row vs gemv_qk256[0] ──

#[test]
fn test_gemv_row_vs_full() {
    let cols = 256;
    let codes: Vec<u8> = (0..cols).map(|i| ((i * 3 + 2) % 4) as u8).collect();
    let x: Vec<f32> = (0..cols).map(|i| i as f32 * 0.05).collect();
    let (qs, stride) = build_qs_data(&[codes], cols);

    let row_result = gemv_qk256_row(&qs, &x, cols);
    let mut full_result = [0.0f32; 1];
    gemv_qk256(&qs, &x, &mut full_result, 1, cols, stride).expect("gemv_qk256");

    assert!((row_result - full_result[0]).abs() < 1e-3, "row={row_result} full={}", full_result[0]);
}

// ── Test 8: pack → unpack roundtrip ──

#[test]
fn test_unpack_code_roundtrip() {
    let mut codes = [0u8; 256];
    for i in 0..256 {
        codes[i] = (i % 4) as u8;
    }
    let packed = pack_codes(&codes);
    let mut unpacked = [0u8; QK256_BLOCK];
    unpack_qk256_block(<&[u8; QK256_PACKED_BYTES]>::try_from(&packed[..]).unwrap(), &mut unpacked);
    for i in 0..256 {
        assert_eq!(
            unpacked[i], codes[i],
            "mismatch at index {i}: expected {}, got {}",
            codes[i], unpacked[i]
        );
    }
}

// ── Test 9: code_to_f32 mapping ──

#[test]
fn test_code_to_f32_mapping() {
    assert_eq!(code_to_f32(0), -2.0);
    assert_eq!(code_to_f32(1), -1.0);
    assert_eq!(code_to_f32(2), 1.0);
    assert_eq!(code_to_f32(3), 2.0);
}

// ── Test 10: all x=0.0 → y=0.0 ──

#[test]
fn test_gemv_zero_input() {
    let cols = 256;
    let rows = 4;
    let row_codes: Vec<Vec<u8>> =
        (0..rows).map(|r| (0..cols).map(|c| ((r + c) % 4) as u8).collect()).collect();
    let x = vec![0.0f32; cols];
    let (qs, stride) = build_qs_data(&row_codes, cols);

    let mut y = vec![0.0f32; rows];
    gemv_qk256(&qs, &x, &mut y, rows, cols, stride).expect("gemv_qk256 zero");
    for (i, &val) in y.iter().enumerate() {
        assert!(val.abs() < 1e-10, "row {i}: expected 0.0, got {val}");
    }
    assert_parity(&qs, &x, rows, cols, stride, 1e-10);
}

// ── Test 11: large x values — no overflow, parity holds ──

#[test]
fn test_gemv_large_values() {
    let cols = 256;
    let rows = 2;
    let row_codes: Vec<Vec<u8>> =
        (0..rows).map(|r| (0..cols).map(|c| ((r * 5 + c) % 4) as u8).collect()).collect();
    let x = vec![1e4_f32; cols];
    let (qs, stride) = build_qs_data(&row_codes, cols);

    let mut y = vec![0.0f32; rows];
    gemv_qk256(&qs, &x, &mut y, rows, cols, stride).expect("gemv_qk256 large");
    for &val in &y {
        assert!(val.is_finite(), "output must be finite: {val}");
    }
    assert_parity(&qs, &x, rows, cols, stride, 1.0);
}

// ── Test 12: minimum col count (cols=4) edge case ──

#[test]
fn test_gemv_single_element_rows() {
    let cols = 4; // minimum meaningful size
    let rows = 2;
    // codes: row0=[0,1,2,3] → weights [-2,-1,+1,+2]
    // codes: row1=[3,3,3,3] → weights [+2,+2,+2,+2]
    let row_codes = vec![vec![0u8, 1, 2, 3], vec![3u8, 3, 3, 3]];
    let x = vec![1.0f32; cols];
    let (qs, stride) = build_qs_data(&row_codes, cols);

    // Row 0: (-2)*1 + (-1)*1 + 1*1 + 2*1 = 0
    // Row 1: 2*1 + 2*1 + 2*1 + 2*1 = 8
    let mut y = vec![0.0f32; rows];
    gemv_qk256(&qs, &x, &mut y, rows, cols, stride).expect("gemv_qk256 small");
    assert!(y[0].abs() < 1e-5, "row 0 expected 0, got {}", y[0]);
    assert!((y[1] - 8.0).abs() < 1e-5, "row 1 expected 8, got {}", y[1]);
    assert_parity(&qs, &x, rows, cols, stride, 1e-5);
}

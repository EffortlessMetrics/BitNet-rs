//! AVX2-vs-scalar parity tests for QK256 GEMV operations.
//!
//! Validates that the AVX2 dispatch path produces results identical
//! (within tolerance) to the scalar path for various input patterns.

#![cfg(all(test, feature = "cpu"))]

use bitnet_quantization::i2s_qk256::{
    QK256_AVX2_GEMV_KERNEL_ID, QK256_BLOCK, QK256_PACKED_BYTES, QK256_SCALAR_GEMV_KERNEL_ID,
    code_to_f32, gemv_qk256, gemv_qk256_row, gemv_qk256_with_kernel_selection,
    select_qk256_gemv_kernel, unpack_qk256_block,
};
#[cfg(target_arch = "x86_64")]
use bitnet_quantization::i2s_qk256_avx2::gemv_qk256_avx2;

/// Pack 256 2-bit codes into BitNet.cpp I2_S grouped QK256 layout.
fn pack_codes(codes: &[u8; 256]) -> [u8; 64] {
    let mut packed = [0u8; 64];
    for (i, &code) in codes.iter().enumerate() {
        let chunk = i / 128;
        let chunk_pos = i % 128;
        let lane = chunk_pos / 32;
        let gp = chunk_pos % 32;
        let byte_idx = chunk * 32 + gp;
        packed[byte_idx] |= (code & 0x03) << (6 - lane * 2);
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
            for (j, bc) in block_codes.iter_mut().enumerate().take(QK256_BLOCK) {
                let col = blk * QK256_BLOCK + j;
                if col < cols {
                    *bc = codes[col] & 0x03;
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
    for (&code, &xi) in codes[..cols].iter().zip(x[..cols].iter()) {
        acc += code_to_f32(code & 0x03) * xi;
    }
    acc
}

fn deterministic_row_codes(rows: usize, cols: usize, salt: usize) -> Vec<Vec<u8>> {
    (0..rows)
        .map(|row| {
            (0..cols).map(|col| ((row * 17 + col * 13 + col / 7 + salt * 5) & 0x03) as u8).collect()
        })
        .collect()
}

fn deterministic_activation(cols: usize, salt: usize) -> Vec<f32> {
    (0..cols)
        .map(|i| {
            let centered = ((i * 31 + salt * 19) % 257) as f32 - 128.0;
            let sign = if (i + salt).is_multiple_of(11) { -1.0 } else { 1.0 };
            sign * centered / 37.0
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
enum CodePattern {
    AllZeroCodes,
    Repeating0123,
    AlternatingRows,
    PseudoRandom,
}

impl CodePattern {
    fn name(self) -> &'static str {
        match self {
            Self::AllZeroCodes => "all_zero_codes",
            Self::Repeating0123 => "repeating_0123",
            Self::AlternatingRows => "alternating_rows",
            Self::PseudoRandom => "pseudo_random",
        }
    }
}

fn row_codes_for_pattern(pattern: CodePattern, row: usize, cols: usize) -> Vec<u8> {
    (0..cols)
        .map(|col| match pattern {
            CodePattern::AllZeroCodes => 0,
            CodePattern::Repeating0123 => (col % 4) as u8,
            CodePattern::AlternatingRows => ((row + col) % 4) as u8,
            CodePattern::PseudoRandom => {
                let mixed = (row as u64)
                    .wrapping_mul(0x9E37_79B9)
                    .wrapping_add((col as u64).wrapping_mul(0x85EB_CA6B))
                    .wrapping_add(0xC2B2_AE35);
                ((mixed ^ (mixed >> 13) ^ (mixed >> 29)) & 0x03) as u8
            }
        })
        .collect()
}

fn deterministic_prng_activation(cols: usize) -> Vec<f32> {
    (0..cols)
        .map(|i| {
            let mixed =
                (i as u64).wrapping_mul(0xD1B5_4A32).wrapping_add(0x94D0_49BB).rotate_left(17);
            let bucket = (mixed % 2049) as f32 - 1024.0;
            bucket / 512.0
        })
        .collect()
}

#[cfg(target_arch = "x86_64")]
fn avx2_selection_available() -> bool {
    cfg!(feature = "avx2") && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
}

fn assert_close_vectors(label: &str, expected: &[f32], actual: &[f32], tolerance: f32) {
    assert_eq!(expected.len(), actual.len(), "{label}: vector length mismatch");
    for (idx, (&expected, &actual)) in expected.iter().zip(actual.iter()).enumerate() {
        assert!(
            (expected - actual).abs() <= tolerance,
            "{label} row {idx}: expected={expected} actual={actual} diff={}",
            (expected - actual).abs()
        );
    }
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
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
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

fn assert_hardened_parity_case(rows: usize, cols: usize, salt: usize, tolerance: f32) {
    let row_codes = deterministic_row_codes(rows, cols, salt);
    let (qs, stride) = build_qs_data(&row_codes, cols);
    let x = deterministic_activation(cols, salt);

    let mut y_scalar = vec![0.0f32; rows];
    let scalar_selection = gemv_qk256_with_kernel_selection(
        &qs,
        &x,
        &mut y_scalar,
        rows,
        cols,
        stride,
        Some(QK256_SCALAR_GEMV_KERNEL_ID),
        true,
    )
    .expect("forced scalar selection must run");
    assert_eq!(scalar_selection.selected_kernel, QK256_SCALAR_GEMV_KERNEL_ID);
    assert!(!scalar_selection.fallback_used);

    let mut y_auto = vec![0.0f32; rows];
    let auto_selection =
        gemv_qk256_with_kernel_selection(&qs, &x, &mut y_auto, rows, cols, stride, None, false)
            .expect("auto selection must run");
    assert!(matches!(
        auto_selection.selected_kernel,
        QK256_SCALAR_GEMV_KERNEL_ID | QK256_AVX2_GEMV_KERNEL_ID
    ));
    assert!(!auto_selection.fallback_used, "auto selection is not fallback");
    assert_close_vectors("auto-vs-scalar", &y_scalar, &y_auto, tolerance);

    let mut y_auto_repeat = vec![0.0f32; rows];
    gemv_qk256(&qs, &x, &mut y_auto_repeat, rows, cols, stride).expect("repeat auto dispatch");
    assert_eq!(y_auto, y_auto_repeat, "auto GEMV must be repeatable for rows={rows} cols={cols}");

    #[cfg(target_arch = "x86_64")]
    {
        if avx2_selection_available() {
            let mut y_strict_avx2 = vec![0.0f32; rows];
            let avx2_selection = gemv_qk256_with_kernel_selection(
                &qs,
                &x,
                &mut y_strict_avx2,
                rows,
                cols,
                stride,
                Some(QK256_AVX2_GEMV_KERNEL_ID),
                true,
            )
            .expect("strict AVX2 selection must run on AVX2/FMA hosts");
            assert_eq!(avx2_selection.selected_kernel, QK256_AVX2_GEMV_KERNEL_ID);
            assert!(!avx2_selection.fallback_used);
            assert_close_vectors("strict-avx2-vs-scalar", &y_scalar, &y_strict_avx2, tolerance);

            let mut y_direct_avx2 = vec![0.0f32; rows];
            gemv_qk256_avx2(&qs, &x, &mut y_direct_avx2, rows, cols, stride)
                .expect("direct AVX2 GEMV");
            assert_close_vectors("direct-avx2-vs-strict-avx2", &y_strict_avx2, &y_direct_avx2, 0.0);

            let mut y_direct_avx2_repeat = vec![0.0f32; rows];
            gemv_qk256_avx2(&qs, &x, &mut y_direct_avx2_repeat, rows, cols, stride)
                .expect("repeat direct AVX2 GEMV");
            assert_eq!(
                y_direct_avx2, y_direct_avx2_repeat,
                "direct AVX2 GEMV must be repeatable for rows={rows} cols={cols}"
            );
        } else {
            let mut y_strict_avx2 = vec![0.0f32; rows];
            let err = gemv_qk256_with_kernel_selection(
                &qs,
                &x,
                &mut y_strict_avx2,
                rows,
                cols,
                stride,
                Some(QK256_AVX2_GEMV_KERNEL_ID),
                true,
            )
            .expect_err("strict AVX2 selection must fail if AVX2/FMA is unavailable");
            assert!(err.to_string().contains("cannot fall back"));
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        let mut y_strict_avx2 = vec![0.0f32; rows];
        let err = gemv_qk256_with_kernel_selection(
            &qs,
            &x,
            &mut y_strict_avx2,
            rows,
            cols,
            stride,
            Some(QK256_AVX2_GEMV_KERNEL_ID),
            true,
        )
        .expect_err("strict AVX2 selection must fail on non-x86_64 hosts");
        assert!(err.to_string().contains("cannot fall back"));
    }
}

#[test]
fn test_requested_scalar_selection_executes_scalar() {
    let rows = 2;
    let cols = 256;
    let row_codes = vec![vec![2u8; cols], vec![1u8; cols]];
    let (qs, stride) = build_qs_data(&row_codes, cols);
    let x: Vec<f32> = (0..cols).map(|i| (i as f32 + 1.0) * 0.125).collect();
    let mut y = vec![0.0f32; rows];

    let selection = gemv_qk256_with_kernel_selection(
        &qs,
        &x,
        &mut y,
        rows,
        cols,
        stride,
        Some(QK256_SCALAR_GEMV_KERNEL_ID),
        true,
    )
    .expect("forced scalar selection");

    assert_eq!(selection.requested_kernel, Some(QK256_SCALAR_GEMV_KERNEL_ID));
    assert_eq!(selection.selected_kernel, QK256_SCALAR_GEMV_KERNEL_ID);
    assert!(!selection.fallback_used);
    assert_eq!(selection.fallback_reason, None);

    assert!((y[0] - manual_dot(&row_codes[0], &x, cols)).abs() < 1e-4);
    assert!((y[1] - manual_dot(&row_codes[1], &x, cols)).abs() < 1e-4);
}

#[test]
fn test_requested_avx2_selection_is_explicit() {
    let selection = select_qk256_gemv_kernel(Some(QK256_AVX2_GEMV_KERNEL_ID), false)
        .expect("non-strict requested AVX2 selection");

    if selection.selected_kernel == QK256_AVX2_GEMV_KERNEL_ID {
        assert!(!selection.fallback_used);
        assert_eq!(selection.fallback_reason, None);
        assert!(selection.cpu_features.contains(&"avx2"));
        assert!(selection.cpu_features.contains(&"fma"));
    } else {
        assert_eq!(selection.selected_kernel, QK256_SCALAR_GEMV_KERNEL_ID);
        assert!(selection.fallback_used);
        assert_eq!(selection.fallback_reason.as_deref(), Some("avx2/fma unavailable"));
    }
}

#[test]
fn test_requested_avx2_strict_matches_runtime_availability() {
    let selection = select_qk256_gemv_kernel(Some(QK256_AVX2_GEMV_KERNEL_ID), true);

    match selection {
        Ok(selection) => {
            assert_eq!(selection.selected_kernel, QK256_AVX2_GEMV_KERNEL_ID);
            assert!(!selection.fallback_used);
            assert!(selection.cpu_features.contains(&"avx2"));
            assert!(selection.cpu_features.contains(&"fma"));
        }
        Err(err) => {
            assert!(err.to_string().contains("cannot fall back"));
        }
    }
}

#[test]
fn test_hardened_avx2_parity_rows_tails_patterns_and_repeats() {
    for (rows, cols, salt) in [(1, 256, 3), (2, 257, 5), (5, 511, 7), (9, 768, 11), (13, 1025, 17)]
    {
        assert_hardened_parity_case(rows, cols, salt, 1e-3);
    }
}

#[test]
fn test_avx2_parity_requested_shape_pattern_matrix_1e4() {
    let patterns = [
        CodePattern::AllZeroCodes,
        CodePattern::Repeating0123,
        CodePattern::AlternatingRows,
        CodePattern::PseudoRandom,
    ];

    for rows in [1usize, 2, 7, 32] {
        for cols in [256usize, 300, 512, 513, 1024] {
            let x = deterministic_prng_activation(cols);
            for pattern in patterns {
                let row_codes: Vec<Vec<u8>> =
                    (0..rows).map(|row| row_codes_for_pattern(pattern, row, cols)).collect();
                let (qs, stride) = build_qs_data(&row_codes, cols);

                let mut y_scalar = vec![0.0f32; rows];
                gemv_qk256_with_kernel_selection(
                    &qs,
                    &x,
                    &mut y_scalar,
                    rows,
                    cols,
                    stride,
                    Some(QK256_SCALAR_GEMV_KERNEL_ID),
                    true,
                )
                .expect("forced scalar GEMV");

                let mut y_scalar_repeat = vec![0.0f32; rows];
                gemv_qk256_with_kernel_selection(
                    &qs,
                    &x,
                    &mut y_scalar_repeat,
                    rows,
                    cols,
                    stride,
                    Some(QK256_SCALAR_GEMV_KERNEL_ID),
                    true,
                )
                .expect("repeat forced scalar GEMV");
                assert_eq!(
                    y_scalar,
                    y_scalar_repeat,
                    "scalar repeat mismatch rows={rows} cols={cols} pattern={}",
                    pattern.name()
                );

                #[cfg(target_arch = "x86_64")]
                if avx2_selection_available() {
                    let mut y_avx2 = vec![0.0f32; rows];
                    let selection = gemv_qk256_with_kernel_selection(
                        &qs,
                        &x,
                        &mut y_avx2,
                        rows,
                        cols,
                        stride,
                        Some(QK256_AVX2_GEMV_KERNEL_ID),
                        true,
                    )
                    .expect("forced AVX2 GEMV");
                    assert_eq!(selection.selected_kernel, QK256_AVX2_GEMV_KERNEL_ID);
                    assert!(!selection.fallback_used);

                    let mut y_avx2_repeat = vec![0.0f32; rows];
                    gemv_qk256_with_kernel_selection(
                        &qs,
                        &x,
                        &mut y_avx2_repeat,
                        rows,
                        cols,
                        stride,
                        Some(QK256_AVX2_GEMV_KERNEL_ID),
                        true,
                    )
                    .expect("repeat forced AVX2 GEMV");
                    assert_eq!(
                        y_avx2,
                        y_avx2_repeat,
                        "AVX2 repeat mismatch rows={rows} cols={cols} pattern={}",
                        pattern.name()
                    );

                    assert_close_vectors(
                        "requested-shape-pattern-avx2-vs-scalar",
                        &y_scalar,
                        &y_avx2,
                        1e-4,
                    );
                } else {
                    let mut y_strict_avx2 = vec![0.0f32; rows];
                    let err = gemv_qk256_with_kernel_selection(
                        &qs,
                        &x,
                        &mut y_strict_avx2,
                        rows,
                        cols,
                        stride,
                        Some(QK256_AVX2_GEMV_KERNEL_ID),
                        true,
                    )
                    .expect_err("strict AVX2 must fail when AVX2/FMA is unavailable");
                    assert!(err.to_string().contains("cannot fall back"));
                }
            }
        }
    }
}

// ── Test 1: single row, all codes=2 (+1), uniform x=1.0 ──

#[test]
fn test_strict_avx2_full_block_position_identity() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(target_arch = "x86_64"))]
    {
        eprintln!("Skipping AVX2 position identity test - not x86_64");
        return Ok(());
    }

    #[cfg(target_arch = "x86_64")]
    {
        if !avx2_selection_available() {
            eprintln!("Skipping AVX2 position identity test - AVX2/FMA unavailable");
            return Ok(());
        }

        let rows = 1;
        let cols = QK256_BLOCK;
        let x: Vec<f32> = (0..cols).map(|idx| (idx as f32 + 0.25) / 17.0).collect();

        for target_col in 0..cols {
            for (code, expected_sign) in [(2u8, 1.0f32), (0u8, -1.0f32)] {
                let mut codes = vec![1u8; cols];
                codes[target_col] = code;
                let (qs, stride) = build_qs_data(&[codes], cols);
                let mut y = vec![0.0f32; rows];

                let selection = gemv_qk256_with_kernel_selection(
                    &qs,
                    &x,
                    &mut y,
                    rows,
                    cols,
                    stride,
                    Some(QK256_AVX2_GEMV_KERNEL_ID),
                    true,
                )?;

                assert_eq!(selection.selected_kernel, QK256_AVX2_GEMV_KERNEL_ID);
                assert!(!selection.fallback_used);
                let expected = expected_sign * x[target_col];
                assert!(
                    (y[0] - expected).abs() <= 1e-6,
                    "target_col={target_col} code={code}: expected={expected} actual={}",
                    y[0]
                );
            }
        }
    }

    Ok(())
}

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

// ── Test 2: single row, all codes=0 (-1), uniform x=1.0 ──

#[test]
fn test_gemv_single_row_all_neg_ones() {
    let cols = 256;
    let codes = vec![0u8; cols]; // code 0 → -1.0
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
    let (qs, stride) = build_qs_data(std::slice::from_ref(&codes), cols);

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
    for (i, code) in codes.iter_mut().enumerate() {
        *code = (i % 4) as u8;
    }
    let packed = pack_codes(&codes);
    let mut unpacked = [0u8; QK256_BLOCK];
    unpack_qk256_block(<&[u8; QK256_PACKED_BYTES]>::try_from(&packed[..]).unwrap(), &mut unpacked);
    for (i, (&u, &c)) in unpacked.iter().zip(codes.iter()).enumerate() {
        assert_eq!(u, c, "mismatch at index {i}: expected {c}, got {u}");
    }
}

// ── Test 9: code_to_f32 mapping ──

#[test]
fn test_code_to_f32_mapping() {
    assert_eq!(code_to_f32(0), -1.0);
    assert_eq!(code_to_f32(1), 0.0);
    assert_eq!(code_to_f32(2), 1.0);
    assert_eq!(code_to_f32(3), 0.0);
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
    // codes: row0=[0,1,2,3] → weights [-1,0,+1,0]
    // codes: row1=[2,2,2,2] → weights [+1,+1,+1,+1]
    let row_codes = vec![vec![0u8, 1, 2, 3], vec![2u8, 2, 2, 2]];
    let x = vec![1.0f32; cols];
    let (qs, stride) = build_qs_data(&row_codes, cols);

    // Row 0: (-1)*1 + 0*1 + 1*1 + 0*1 = 0
    // Row 1: 1*1 + 1*1 + 1*1 + 1*1 = 4
    let mut y = vec![0.0f32; rows];
    gemv_qk256(&qs, &x, &mut y, rows, cols, stride).expect("gemv_qk256 small");
    assert!(y[0].abs() < 1e-5, "row 0 expected 0, got {}", y[0]);
    assert!((y[1] - 4.0).abs() < 1e-5, "row 1 expected 4, got {}", y[1]);
    assert_parity(&qs, &x, rows, cols, stride, 1e-5);
}

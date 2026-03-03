//! Property-based tests comparing SIMD-dispatched vs scalar results.

#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap, clippy::cast_precision_loss)]

use bitnet_avx512_gemv::*;
use proptest::prelude::*;

// ── f32 GEMV properties ───────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn f32_gemv_scalar_matches_dispatch(
        rows in 1usize..=16,
        cols in 1usize..=64,
    ) {
        let n = rows * cols;
        // Generate data deterministically from rows/cols.
        let matrix: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) % 201) as f32 - 100.0).collect();
        let vector: Vec<f32> = (0..cols).map(|i| ((i * 13 + 5) % 201) as f32 - 100.0).collect();
        let p = GemvParams::new(rows, cols, &matrix, &vector);
        let dispatched = gemv(&p);
        let scalar = f32_gemv::gemv_scalar(&p);
        for (a, b) in dispatched.iter().zip(scalar.iter()) {
            prop_assert!((a - b).abs() < 1.0, "f32 GEMV mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn f32_dot_scalar_matches_dispatch(n in 0usize..=256) {
        let a: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) % 201) as f32 - 100.0).collect();
        let b: Vec<f32> = (0..n).map(|i| ((i * 13 + 5) % 201) as f32 - 100.0).collect();
        let dispatched = dot::dot_f32(&a, &b);
        let scalar = dot::dot_f32_scalar(&a, &b);
        prop_assert!((dispatched - scalar).abs() < 1.0,
            "f32 dot mismatch: {} vs {}", dispatched, scalar);
    }

    #[test]
    fn f32_gemv_zero_vector_gives_zero(rows in 1usize..=8, cols in 1usize..=32) {
        let matrix: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
        let vector = vec![0.0_f32; cols];
        let p = GemvParams::new(rows, cols, &matrix, &vector);
        let result = gemv(&p);
        for &v in &result {
            prop_assert!((v).abs() < f32::EPSILON, "expected zero, got {v}");
        }
    }

    #[test]
    fn f32_gemv_identity(n in 1usize..=16) {
        let mut eye = vec![0.0_f32; n * n];
        for i in 0..n {
            eye[i * n + i] = 1.0;
        }
        let v: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let p = GemvParams::new(n, n, &eye, &v);
        let r = gemv(&p);
        for (i, &val) in r.iter().enumerate() {
            prop_assert!((val - i as f32).abs() < f32::EPSILON,
                "identity failed at {i}: got {val}");
        }
    }
}

// ── i8 GEMV properties ───────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn i8_gemv_scalar_matches_dispatch(
        rows in 1usize..=16,
        cols in 1usize..=64,
    ) {
        let n = rows * cols;
        let matrix: Vec<i8> = (0..n).map(|i| (((i * 7 + 3) % 256) as i16 - 128) as i8).collect();
        let vector: Vec<i8> = (0..cols).map(|i| (((i * 13 + 5) % 256) as i16 - 128) as i8).collect();
        let p = I8GemvParams::new(rows, cols, &matrix, &vector);
        let dispatched = gemv_i8(&p);
        let scalar = i8_gemv::gemv_i8_scalar(&p);
        prop_assert_eq!(dispatched, scalar, "i8 GEMV mismatch");
    }

    #[test]
    fn i8_dot_scalar_matches_dispatch(n in 0usize..=256) {
        let a: Vec<i8> = (0..n).map(|i| (((i * 7 + 3) % 256) as i16 - 128) as i8).collect();
        let b: Vec<i8> = (0..n).map(|i| (((i * 13 + 5) % 256) as i16 - 128) as i8).collect();
        let dispatched = dot::dot_i8(&a, &b);
        let scalar = dot::dot_i8_scalar(&a, &b);
        prop_assert_eq!(dispatched, scalar, "i8 dot mismatch");
    }

    #[test]
    fn i8_gemv_zero_vector_gives_zero(rows in 1usize..=8, cols in 1usize..=32) {
        let matrix: Vec<i8> = (0..rows * cols).map(|i| (i % 127) as i8).collect();
        let vector = vec![0_i8; cols];
        let p = I8GemvParams::new(rows, cols, &matrix, &vector);
        let result = gemv_i8(&p);
        for &v in &result {
            prop_assert_eq!(v, 0, "expected zero, got {}", v);
        }
    }

    #[test]
    fn i8_gemv_identity(n in 1usize..=16) {
        let mut eye = vec![0_i8; n * n];
        for i in 0..n {
            eye[i * n + i] = 1;
        }
        let v: Vec<i8> = (0..n).map(|i| i as i8).collect();
        let p = I8GemvParams::new(n, n, &eye, &v);
        let r = gemv_i8(&p);
        for (i, &val) in r.iter().enumerate() {
            prop_assert_eq!(val, i as i32, "identity failed at {}: got {}", i, val);
        }
    }
}

// ── Binary GEMV properties ────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn binary_gemv_scalar_matches_dispatch(
        rows in 1usize..=8,
        col_bytes in 1usize..=8,
    ) {
        let cols = col_bytes * 8;
        let packed: Vec<u8> = (0..rows * col_bytes).map(|i| (i * 37) as u8).collect();
        let vector: Vec<f32> = (0..cols).map(|i| ((i * 7) % 100) as f32 * 0.1).collect();
        let p = BinaryGemvParams::new(rows, cols, &packed, &vector);
        let dispatched = gemv_binary(&p);
        let scalar = binary_gemv::gemv_binary_scalar(&p);
        for (a, b) in dispatched.iter().zip(scalar.iter()) {
            prop_assert!((a - b).abs() < 1e-3,
                "binary GEMV mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn binary_all_ones_row_sums_vector(col_bytes in 1usize..=8) {
        let cols = col_bytes * 8;
        let packed = vec![0xFF_u8; col_bytes];
        let vector: Vec<f32> = (0..cols).map(|i| i as f32).collect();
        let p = BinaryGemvParams::new(1, cols, &packed, &vector);
        let r = gemv_binary(&p);
        let expected: f32 = vector.iter().sum();
        prop_assert!((r[0] - expected).abs() < 1e-3,
            "all-ones should sum vector: {} vs {}", r[0], expected);
    }

    #[test]
    fn binary_all_zeros_row_negates_sum(col_bytes in 1usize..=8) {
        let cols = col_bytes * 8;
        let packed = vec![0x00_u8; col_bytes];
        let vector: Vec<f32> = (0..cols).map(|i| i as f32).collect();
        let p = BinaryGemvParams::new(1, cols, &packed, &vector);
        let r = gemv_binary(&p);
        let expected: f32 = -vector.iter().sum::<f32>();
        prop_assert!((r[0] - expected).abs() < 1e-3,
            "all-zeros should negate sum: {} vs {}", r[0], expected);
    }
}

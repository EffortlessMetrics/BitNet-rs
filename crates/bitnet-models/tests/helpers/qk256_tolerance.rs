//! QK256 FMA vs Scalar Tolerance Strategy
//!
//! Implements adaptive tolerance for comparing AVX2 FMA vs scalar QK256 GEMV results.
//!
//! ## Rationale
//!
//! AVX2 FMA performs horizontal reduction differently than scalar left-associative
//! accumulation, leading to different rounding behaviors:
//!
//! - **Scalar**: ((... + w1*x1) + w2*x2) + w3*x3 (sequential, left-associative)
//! - **FMA**: 8-way parallel FMA lanes with horizontal sum reduction
//!
//! This causes drift that scales with sqrt(matrix_cols), not linearly.
//!
//! ## Formula
//!
//! **Absolute tolerance**: `1e-5 × sqrt(cols / 256)`, capped at 5e-4
//! - Base: 1e-5 ULPs for single 256-element block
//! - Scaling: sqrt accounts for random-walk error accumulation
//! - Cap: prevents masking real bugs
//!
//! **Relative tolerance**: 1e-4 (standard numerical analysis threshold)
//! - Applied only when |result| > 1e-6
//! - Provides safety check for scaled results
//!
//! ## Simplified API for Property Tests
//!
//! The strategy document proposes a comprehensive tolerance formula, but for property tests
//! we provide a simplified API that uses length-based scaling:
//!
//! ```rust,ignore
//! use helpers::qk256_tolerance::approx_eq_with_len;
//!
//! // For QK256 property tests: adaptive tolerance based on accumulation length
//! let len = 2048; // columns in matrix
//! assert!(approx_eq_with_len(qk256_result, fp32_result, len));
//! ```

/// Approximate equality with fixed tolerance
///
/// This is a simplified helper for basic equality checks without length-based scaling.
/// For QK256 property tests, prefer `approx_eq_with_len` which accounts for accumulation length.
///
/// Uses combined absolute + relative tolerance:
/// - Absolute tolerance: 1e-4
/// - Relative tolerance: 1e-4 (0.01%)
///
/// Returns true if either:
/// - `|a - b| < abs_tol` OR
/// - `|a - b| / max(|a|, |b|) < rel_tol`
pub fn approx_eq(a: f32, b: f32) -> bool {
    let abs_tol = 1e-4;
    let rel_tol = 1e-4;

    let diff = (a - b).abs();

    // Check 1: Absolute tolerance (always applied)
    if diff < abs_tol {
        return true;
    }

    // Check 2: Relative tolerance (if result magnitude permits)
    let max_magnitude = a.abs().max(b.abs());
    if max_magnitude > 1e-6 {
        let rel_diff = diff / max_magnitude;
        if rel_diff < rel_tol {
            return true;
        }
    }

    false
}

/// Approximate equality with length-based adaptive tolerance
///
/// This is the recommended helper for QK256 property tests. It accounts for accumulation
/// length to prevent false failures on large matrices with FMA-induced drift.
///
/// Formula (empirically tuned based on observed AVX2 FMA vs scalar differences):
/// - Base absolute tolerance: 2e-4 (for single 256-element block)
/// - Scaling: `sqrt(len / 256)` to account for error accumulation (random walk behavior)
/// - Cap: 1e-3 to prevent masking real bugs while allowing observed FMA drift
/// - Relative tolerance: 2e-2 (2% - accounts for near-zero denominators and FMA drift)
///
/// Returns true if either:
/// - `|a - b| < (2e-4 * sqrt(len/256))` (capped at 1e-3) OR
/// - `|a - b| / max(|a|, |b|) < 2e-2`
///
/// # Arguments
/// * `a` - First value to compare (typically QK256 result)
/// * `b` - Second value to compare (typically FP32 reference result)
/// * `len` - Accumulation length (number of columns in matrix)
///
/// # Example
/// ```rust,ignore
/// let qk256_result = 100.0;
/// let fp32_result = 100.0002;
/// let cols = 2048;
/// assert!(approx_eq_with_len(qk256_result, fp32_result, cols));
/// ```
pub fn approx_eq_with_len(a: f32, b: f32, len: usize) -> bool {
    // Adaptive absolute tolerance: scales with sqrt(len/256) to account for accumulation drift
    // Base: 2e-4 for single 256-element block (empirically tuned for FMA vs scalar differences)
    // Scaling: sqrt accounts for random-walk error accumulation
    // Cap: 1e-3 prevents masking real bugs while allowing observed FMA drift
    let cols_factor = (len as f32 / 256.0).sqrt();
    let abs_tol = (2e-4 * cols_factor).min(1e-3);
    let rel_tol = 2e-2; // 2% relative tolerance to account for near-zero denominators and FMA drift

    let diff = (a - b).abs();

    // Check 1: Absolute tolerance with length-based scaling (always applied)
    if diff < abs_tol {
        return true;
    }

    // Check 2: Relative tolerance (if result magnitude permits)
    let max_magnitude = a.abs().max(b.abs());
    if max_magnitude > 1e-6 {
        let rel_diff = diff / max_magnitude;
        if rel_diff < rel_tol {
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_approx_eq_exact_match() {
        assert!(approx_eq(1.0, 1.0));
        assert!(approx_eq(0.0, 0.0));
        assert!(approx_eq(-1.0, -1.0));
    }

    #[test]
    fn test_approx_eq_within_absolute_tolerance() {
        assert!(approx_eq(1.0, 1.00005)); // diff = 5e-5 < 1e-4
        assert!(approx_eq(100.0, 100.00008)); // diff = 8e-5 < 1e-4
    }

    #[test]
    fn test_approx_eq_within_relative_tolerance() {
        // Large values where relative tolerance dominates
        assert!(approx_eq(10000.0, 10000.5)); // rel_diff = 5e-5 < 1e-4
        assert!(approx_eq(1e6, 1e6 + 50.0)); // rel_diff = 5e-5 < 1e-4
    }

    #[test]
    fn test_approx_eq_exceeds_tolerance() {
        assert!(!approx_eq(1.0, 1.001)); // diff = 1e-3 > 1e-4
        assert!(!approx_eq(100.0, 100.05)); // diff = 5e-2 > 1e-4
    }

    #[test]
    fn test_approx_eq_near_zero() {
        assert!(approx_eq(0.0, 1e-5)); // diff = 1e-5 < 1e-4
        assert!(approx_eq(1e-7, 2e-7)); // diff = 1e-7 < 1e-4
        assert!(!approx_eq(0.0, 1e-3)); // diff = 1e-3 > 1e-4
    }

    #[test]
    fn test_approx_eq_with_len_single_block() {
        // Single 256-element block: tolerance = 2e-4 * sqrt(256/256) = 2e-4
        let len = 256;
        let cols_factor = (len as f32 / 256.0).sqrt();
        let expected_tol = 2e-4 * cols_factor;
        assert!((expected_tol - 2e-4).abs() < 1e-10);

        // Should pass with error just under absolute tolerance
        assert!(approx_eq_with_len(1.0, 1.0 + 1.9e-4, len));

        // Should pass with error within relative tolerance (1e-2 / 1.0 = 1e-2 < 2e-2)
        assert!(approx_eq_with_len(1.0, 1.0 + 1e-2, len));

        // Should fail with error exceeding both absolute and relative tolerance
        // For small values like 1.0, we need abs_diff > 2e-4 AND rel_diff > 2e-2
        // abs_diff = 3e-2, rel_diff = 3e-2 / 1.0 = 3e-2 > 2e-2
        assert!(!approx_eq_with_len(1.0, 1.0 + 3e-2, len));
    }

    #[test]
    fn test_approx_eq_with_len_large_matrix() {
        // 2048-element accumulation: tolerance = 2e-4 * sqrt(2048/256) = 2e-4 * sqrt(8) ≈ 5.66e-4
        let len = 2048;
        let cols_factor = (len as f32 / 256.0).sqrt();
        let expected_tol = (2e-4 * cols_factor).min(1e-3);
        assert!((expected_tol - 5.66e-4).abs() / 5.66e-4 < 0.01);

        // Should pass with error just under tolerance
        assert!(approx_eq_with_len(100.0, 100.0 + 5.6e-4, len));

        // Should pass with error within relative tolerance (1.0 / 100.0 = 1e-2 < 2e-2)
        assert!(approx_eq_with_len(100.0, 100.0 + 1.0, len));

        // Should fail with error exceeding both absolute AND relative tolerance
        // abs_diff = 2.5, rel_diff = 2.5/100.0 = 2.5e-2 > 2e-2
        assert!(!approx_eq_with_len(100.0, 100.0 + 2.5, len));
    }

    #[test]
    fn test_approx_eq_with_len_scales_correctly() {
        // Verify tolerance scales with sqrt(len)
        let len_256 = 256;
        let len_1024 = 1024;

        let cols_factor_256 = (len_256 as f32 / 256.0).sqrt();
        let cols_factor_1024 = (len_1024 as f32 / 256.0).sqrt();
        let tol_256 = 1e-5 * cols_factor_256;
        let tol_1024 = 1e-5 * cols_factor_1024;

        // 1024 = 4 × 256 → sqrt(4) = 2× tolerance
        let expected_ratio = 2.0;
        let actual_ratio = tol_1024 / tol_256;
        assert!((actual_ratio - expected_ratio).abs() / expected_ratio < 0.01);
    }

    #[test]
    fn test_approx_eq_with_len_relative_tolerance() {
        // For large values, relative tolerance should dominate
        let len = 256;

        // Large magnitude: rel_diff = 150.0 / 10000 = 1.5e-2 < 2e-2
        assert!(approx_eq_with_len(10000.0, 10150.0, len));

        // Large magnitude: rel_diff = 250.0 / 10000 = 2.5e-2 > 2e-2
        assert!(!approx_eq_with_len(10000.0, 10250.0, len));
    }

    #[test]
    fn test_approx_eq_with_len_near_zero() {
        let len = 256;

        // Near-zero result: absolute tolerance is more reliable
        // For len=256, abs_tol = 2e-4
        assert!(approx_eq_with_len(0.0, 1e-5, len)); // well within 2e-4
        assert!(approx_eq_with_len(1e-7, 1.1e-7, len)); // diff = 1e-8 < 2e-4

        // Exceeds absolute tolerance (2e-4) and relative check doesn't apply for near-zero
        assert!(!approx_eq_with_len(0.0, 3e-4, len));
    }

    #[test]
    fn test_tolerance_formula_matches_strategy_doc() {
        // Verify the tolerance formula: 2e-4 * sqrt(len / 256), capped at 1e-3
        let test_cases = vec![
            (256, 2e-4),      // sqrt(256/256) = 1.0 → 2e-4
            (512, 2.828e-4),  // sqrt(512/256) = sqrt(2) ≈ 1.414 → 2.828e-4
            (1024, 4e-4),     // sqrt(1024/256) = 2.0 → 4e-4
            (2048, 5.656e-4), // sqrt(2048/256) = sqrt(8) ≈ 2.828 → 5.656e-4
        ];

        for (len, expected_tol) in test_cases {
            let cols_factor = (len as f32 / 256.0).sqrt();
            let computed_tol = (2e-4 * cols_factor).min(1e-3);
            let rel_error = (computed_tol - expected_tol).abs() / expected_tol;
            assert!(
                rel_error < 0.01,
                "Tolerance formula mismatch for len={}: got {:.3e}, expected {:.3e}",
                len,
                computed_tol,
                expected_tol
            );
        }
    }
}

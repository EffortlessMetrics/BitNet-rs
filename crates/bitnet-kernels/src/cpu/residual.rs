//! CPU residual connection operations.
//!
//! Provides in-place residual addition for transformer architectures
//! (`x += sublayer(x)`), with optional scaling and dropout masking.
//! Loops are written for auto-vectorization by LLVM.

use bitnet_common::{BitNetError, KernelError, Result};

fn length_mismatch(output_len: usize, residual_len: usize) -> BitNetError {
    BitNetError::Kernel(KernelError::InvalidArguments {
        reason: format!("output length ({output_len}) must equal residual length ({residual_len})"),
    })
}

fn mask_length_mismatch(output_len: usize, mask_len: usize) -> BitNetError {
    BitNetError::Kernel(KernelError::InvalidArguments {
        reason: format!("output length ({output_len}) must equal dropout_mask length ({mask_len})"),
    })
}

/// In-place residual addition: `output[i] += residual[i]`.
///
/// # Errors
///
/// Returns an error if `output` and `residual` have different lengths.
#[inline]
pub fn add_residual(output: &mut [f32], residual: &[f32]) -> Result<()> {
    if output.len() != residual.len() {
        return Err(length_mismatch(output.len(), residual.len()));
    }
    // Simple loop — LLVM auto-vectorizes with `-C opt-level>=2`.
    for (o, &r) in output.iter_mut().zip(residual.iter()) {
        *o += r;
    }
    Ok(())
}

/// Scaled residual addition: `output[i] += scale * residual[i]`.
///
/// # Errors
///
/// Returns an error if `output` and `residual` have different lengths.
#[inline]
pub fn add_residual_scaled(output: &mut [f32], residual: &[f32], scale: f32) -> Result<()> {
    if output.len() != residual.len() {
        return Err(length_mismatch(output.len(), residual.len()));
    }
    for (o, &r) in output.iter_mut().zip(residual.iter()) {
        *o += scale * r;
    }
    Ok(())
}

/// Masked residual addition (dropout): adds `residual[i]` only where
/// `dropout_mask[i]` is `true`.
///
/// # Errors
///
/// Returns an error if the three slices have different lengths.
#[inline]
pub fn add_residual_with_dropout(
    output: &mut [f32],
    residual: &[f32],
    dropout_mask: &[bool],
) -> Result<()> {
    if output.len() != residual.len() {
        return Err(length_mismatch(output.len(), residual.len()));
    }
    if output.len() != dropout_mask.len() {
        return Err(mask_length_mismatch(output.len(), dropout_mask.len()));
    }
    for ((o, &r), &keep) in output.iter_mut().zip(residual.iter()).zip(dropout_mask.iter()) {
        if keep {
            *o += r;
        }
    }
    Ok(())
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── add_residual ───────────────────────────────────────────

    #[test]
    fn add_residual_known_values() {
        let mut output = vec![1.0, 2.0, 3.0, 4.0];
        let residual = vec![0.5, -0.5, 1.0, -1.0];
        add_residual(&mut output, &residual).unwrap();
        assert_eq!(output, vec![1.5, 1.5, 4.0, 3.0]);
    }

    #[test]
    fn add_residual_zero_length() {
        let mut output: Vec<f32> = vec![];
        let residual: Vec<f32> = vec![];
        add_residual(&mut output, &residual).unwrap();
        assert!(output.is_empty());
    }

    #[test]
    fn add_residual_all_zeros_residual() {
        let mut output = vec![1.0, 2.0, 3.0];
        let residual = vec![0.0, 0.0, 0.0];
        add_residual(&mut output, &residual).unwrap();
        assert_eq!(output, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn add_residual_length_mismatch() {
        let mut output = vec![1.0, 2.0];
        let residual = [1.0];
        assert!(add_residual(&mut output, &residual).is_err());
    }

    #[test]
    fn add_residual_then_subtract_roundtrip() {
        let original = vec![4.0_f32, -2.0, 0.0, 42.0];
        let residual = vec![1.0, -1.0, 0.5, -0.5];
        let mut output = original.clone();
        add_residual(&mut output, &residual).unwrap();
        // Subtract to recover original.
        let neg: Vec<f32> = residual.iter().map(|r| -r).collect();
        add_residual(&mut output, &neg).unwrap();
        for (a, b) in output.iter().zip(original.iter()) {
            assert!((a - b).abs() < f32::EPSILON);
        }
    }

    // ── add_residual_scaled ────────────────────────────────────

    #[test]
    fn add_residual_scaled_known_values() {
        let mut output = vec![1.0, 2.0, 3.0];
        let residual = vec![2.0, 4.0, 6.0];
        add_residual_scaled(&mut output, &residual, 0.5).unwrap();
        assert_eq!(output, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn add_residual_scaled_zero_scale() {
        let mut output = vec![1.0, 2.0, 3.0];
        let original = output.clone();
        let residual = vec![100.0, 200.0, 300.0];
        add_residual_scaled(&mut output, &residual, 0.0).unwrap();
        assert_eq!(output, original);
    }

    #[test]
    fn add_residual_scaled_unit_scale() {
        let mut output = vec![1.0, 2.0];
        let residual = vec![10.0, 20.0];
        add_residual_scaled(&mut output, &residual, 1.0).unwrap();
        assert_eq!(output, vec![11.0, 22.0]);
    }

    #[test]
    fn add_residual_scaled_linearity() {
        // scale * (a + b) == scale * a + scale * b
        let base = [0.0_f32; 4];
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let scale = 0.3;

        // Apply a then b with same scale.
        let mut combined = base.clone();
        add_residual_scaled(&mut combined, &a, scale).unwrap();
        add_residual_scaled(&mut combined, &b, scale).unwrap();

        // Apply (a+b) with same scale.
        let ab: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
        let mut single = base;
        add_residual_scaled(&mut single, &ab, scale).unwrap();

        for (x, y) in combined.iter().zip(single.iter()) {
            assert!((x - y).abs() < 1e-6);
        }
    }

    #[test]
    fn add_residual_scaled_zero_length() {
        let mut output: Vec<f32> = vec![];
        add_residual_scaled(&mut output, &[], 2.0).unwrap();
        assert!(output.is_empty());
    }

    #[test]
    fn add_residual_scaled_length_mismatch() {
        let mut output = [1.0];
        assert!(add_residual_scaled(&mut output, &[1.0, 2.0], 1.0).is_err());
    }

    // ── add_residual_with_dropout ──────────────────────────────

    #[test]
    fn add_residual_with_dropout_known_values() {
        let mut output = vec![1.0, 2.0, 3.0, 4.0];
        let residual = vec![10.0, 20.0, 30.0, 40.0];
        let mask = vec![true, false, true, false];
        add_residual_with_dropout(&mut output, &residual, &mask).unwrap();
        assert_eq!(output, vec![11.0, 2.0, 33.0, 4.0]);
    }

    #[test]
    fn add_residual_with_dropout_all_kept() {
        let mut output = vec![1.0, 2.0];
        let residual = vec![3.0, 4.0];
        let mask = vec![true, true];
        add_residual_with_dropout(&mut output, &residual, &mask).unwrap();
        assert_eq!(output, vec![4.0, 6.0]);
    }

    #[test]
    fn add_residual_with_dropout_all_dropped() {
        let mut output = vec![1.0, 2.0];
        let original = output.clone();
        let residual = vec![100.0, 200.0];
        let mask = vec![false, false];
        add_residual_with_dropout(&mut output, &residual, &mask).unwrap();
        assert_eq!(output, original);
    }

    #[test]
    fn add_residual_with_dropout_zero_length() {
        let mut output: Vec<f32> = vec![];
        add_residual_with_dropout(&mut output, &[], &[]).unwrap();
        assert!(output.is_empty());
    }

    #[test]
    fn add_residual_with_dropout_residual_mismatch() {
        let mut output = vec![1.0, 2.0];
        assert!(add_residual_with_dropout(&mut output, &[1.0], &[true, true]).is_err());
    }

    #[test]
    fn add_residual_with_dropout_mask_mismatch() {
        let mut output = vec![1.0, 2.0];
        assert!(add_residual_with_dropout(&mut output, &[1.0, 2.0], &[true]).is_err());
    }

    // ── Property: larger vectors (auto-vectorization path) ─────

    #[test]
    fn add_residual_large_vector() {
        let n = 1024;
        let mut output = vec![1.0_f32; n];
        let residual = vec![0.5_f32; n];
        add_residual(&mut output, &residual).unwrap();
        assert!(output.iter().all(|&v| (v - 1.5).abs() < f32::EPSILON));
    }

    #[test]
    fn add_residual_scaled_large_vector() {
        let n = 1024;
        let mut output = vec![0.0_f32; n];
        let residual: Vec<f32> = (0..n).map(|i| i as f32).collect();
        add_residual_scaled(&mut output, &residual, 2.0).unwrap();
        for (i, &v) in output.iter().enumerate() {
            assert!((v - 2.0 * i as f32).abs() < f32::EPSILON);
        }
    }
}

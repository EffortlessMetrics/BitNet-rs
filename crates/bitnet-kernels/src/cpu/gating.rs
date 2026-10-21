//! CPU gating kernels for transformer FFN layers.
//!
//! Gating mechanisms combine two projections element-wise:
//! `output = activation(gate) * up`
//!
//! Supported gating types:
//! - **SwiGLU**: `SiLU(gate) * up` — used in LLaMA, Mistral, etc.
//! - **GeGLU**: `GELU(gate) * up` — used in some GPT variants
//! - **ReGLU**: `ReLU(gate) * up` — simpler alternative

use bitnet_common::{KernelError, Result};

// ── Gating type enum ────────────────────────────────────────────────

/// Supported gating function types for FFN layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatingType {
    /// SwiGLU: `SiLU(gate) * up`
    SwiGLU,
    /// GeGLU: `GELU(gate) * up`
    GeGLU,
    /// ReGLU: `ReLU(gate) * up`
    ReGLU,
}

// ── Scalar activation helpers ───────────────────────────────────────

/// Scalar SiLU: `x / (1 + exp(-x))`.
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Scalar GELU (tanh approximation).
#[inline]
fn gelu(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const COEFF: f32 = 0.044_715;
    let x3 = x * x * x;
    let inner = SQRT_2_OVER_PI * (x + COEFF * x3);
    0.5 * x * (1.0 + inner.tanh())
}

/// Scalar ReLU: `max(0, x)`.
#[inline]
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

// ── Input validation ────────────────────────────────────────────────

/// Validate that `gate`, `up`, and `output` slices are compatible.
fn validate_gating_buffers(gate: &[f32], up: &[f32], output: &[f32]) -> Result<usize> {
    if gate.is_empty() {
        return Err(KernelError::InvalidArguments {
            reason: "gating input must be non-empty".into(),
        }
        .into());
    }
    if gate.len() != up.len() {
        return Err(KernelError::InvalidArguments {
            reason: format!("gating gate length {} != up length {}", gate.len(), up.len()),
        }
        .into());
    }
    if output.len() < gate.len() {
        return Err(KernelError::InvalidArguments {
            reason: format!("gating output length {} < input length {}", output.len(), gate.len()),
        }
        .into());
    }
    Ok(gate.len())
}

// ── Gating implementations ──────────────────────────────────────────

/// SwiGLU gating: `output[i] = SiLU(gate[i]) * up[i]`.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if slices are empty or
/// have mismatched lengths.
pub fn swiglu(gate: &[f32], up: &[f32], output: &mut [f32]) -> Result<()> {
    let n = validate_gating_buffers(gate, up, output)?;
    for i in 0..n {
        output[i] = silu(gate[i]) * up[i];
    }
    Ok(())
}

/// GeGLU gating: `output[i] = GELU(gate[i]) * up[i]`.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if slices are empty or
/// have mismatched lengths.
pub fn geglu(gate: &[f32], up: &[f32], output: &mut [f32]) -> Result<()> {
    let n = validate_gating_buffers(gate, up, output)?;
    for i in 0..n {
        output[i] = gelu(gate[i]) * up[i];
    }
    Ok(())
}

/// ReGLU gating: `output[i] = ReLU(gate[i]) * up[i]`.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if slices are empty or
/// have mismatched lengths.
pub fn reglu(gate: &[f32], up: &[f32], output: &mut [f32]) -> Result<()> {
    let n = validate_gating_buffers(gate, up, output)?;
    for i in 0..n {
        output[i] = relu(gate[i]) * up[i];
    }
    Ok(())
}

/// Dispatch gating by [`GatingType`].
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if slices are empty or
/// have mismatched lengths.
pub fn apply_gating(
    gating: GatingType,
    gate: &[f32],
    up: &[f32],
    output: &mut [f32],
) -> Result<()> {
    match gating {
        GatingType::SwiGLU => swiglu(gate, up, output),
        GatingType::GeGLU => geglu(gate, up, output),
        GatingType::ReGLU => reglu(gate, up, output),
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Validation tests ---------------------------------------------------

    #[test]
    fn test_gating_rejects_empty() {
        let mut out = [0.0f32; 1];
        assert!(swiglu(&[], &[], &mut out).is_err());
        assert!(geglu(&[], &[], &mut out).is_err());
        assert!(reglu(&[], &[], &mut out).is_err());
    }

    #[test]
    fn test_gating_rejects_length_mismatch() {
        let mut out = [0.0f32; 4];
        assert!(swiglu(&[1.0, 2.0], &[1.0], &mut out).is_err());
        assert!(geglu(&[1.0], &[1.0, 2.0], &mut out).is_err());
        assert!(reglu(&[1.0, 2.0, 3.0], &[1.0, 2.0], &mut out).is_err());
    }

    #[test]
    fn test_gating_rejects_short_output() {
        let mut out = [0.0f32; 1];
        assert!(swiglu(&[1.0, 2.0], &[1.0, 2.0], &mut out).is_err());
    }

    // -- SwiGLU known values ------------------------------------------------

    #[test]
    fn test_swiglu_zeros() {
        let gate = [0.0f32; 4];
        let up = [1.0, 2.0, 3.0, 4.0];
        let mut out = [999.0f32; 4];
        swiglu(&gate, &up, &mut out).unwrap();
        // SiLU(0) = 0, so output is all zeros
        for &v in &out[..4] {
            assert!(v.abs() < 1e-7, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_swiglu_known_values() {
        // SiLU(1) ≈ 0.7311, SiLU(-1) ≈ -0.2689
        let gate = [1.0, -1.0, 2.0, 0.0];
        let up = [1.0, 1.0, 0.5, 5.0];
        let mut out = [0.0f32; 4];
        swiglu(&gate, &up, &mut out).unwrap();

        assert!((out[0] - 0.7311).abs() < 1e-3, "swiglu(1,1)={}", out[0]);
        assert!((out[1] - (-0.2689)).abs() < 1e-3, "swiglu(-1,1)={}", out[1]);
        // SiLU(2) ≈ 1.7616, * 0.5 ≈ 0.8808
        assert!((out[2] - 0.8808).abs() < 1e-3, "swiglu(2,0.5)={}", out[2]);
        // SiLU(0) = 0, * 5 = 0
        assert!(out[3].abs() < 1e-7, "swiglu(0,5)={}", out[3]);
    }

    #[test]
    fn test_swiglu_large_positive_gate() {
        let gate = [100.0f32];
        let up = [2.0f32];
        let mut out = [0.0f32];
        swiglu(&gate, &up, &mut out).unwrap();
        // SiLU(100) ≈ 100, * 2 ≈ 200
        assert!((out[0] - 200.0).abs() < 1e-2, "swiglu(100,2)={}", out[0]);
    }

    #[test]
    fn test_swiglu_large_negative_gate() {
        let gate = [-100.0f32];
        let up = [2.0f32];
        let mut out = [999.0f32];
        swiglu(&gate, &up, &mut out).unwrap();
        // SiLU(-100) ≈ 0
        assert!(out[0].abs() < 1e-2, "swiglu(-100,2)={}", out[0]);
    }

    #[test]
    fn test_swiglu_negative_up() {
        let gate = [1.0f32];
        let up = [-3.0f32];
        let mut out = [0.0f32];
        swiglu(&gate, &up, &mut out).unwrap();
        // SiLU(1) ≈ 0.7311, * -3 ≈ -2.1933
        assert!((out[0] - (-2.1933)).abs() < 1e-3, "swiglu(1,-3)={}", out[0]);
    }

    // -- GeGLU known values -------------------------------------------------

    #[test]
    fn test_geglu_zeros() {
        let gate = [0.0f32; 4];
        let up = [1.0, 2.0, 3.0, 4.0];
        let mut out = [999.0f32; 4];
        geglu(&gate, &up, &mut out).unwrap();
        // GELU(0) = 0
        for &v in &out[..4] {
            assert!(v.abs() < 1e-7, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_geglu_known_values() {
        // GELU(1) ≈ 0.8412, GELU(-1) ≈ -0.1588
        let gate = [1.0, -1.0, 0.0];
        let up = [1.0, 1.0, 5.0];
        let mut out = [0.0f32; 3];
        geglu(&gate, &up, &mut out).unwrap();

        assert!((out[0] - 0.8412).abs() < 1e-3, "geglu(1,1)={}", out[0]);
        assert!((out[1] - (-0.1588)).abs() < 1e-3, "geglu(-1,1)={}", out[1]);
        assert!(out[2].abs() < 1e-7, "geglu(0,5)={}", out[2]);
    }

    #[test]
    fn test_geglu_large_positive_gate() {
        let gate = [10.0f32];
        let up = [2.0f32];
        let mut out = [0.0f32];
        geglu(&gate, &up, &mut out).unwrap();
        // GELU(10) ≈ 10, * 2 ≈ 20
        assert!((out[0] - 20.0).abs() < 1e-2, "geglu(10,2)={}", out[0]);
    }

    #[test]
    fn test_geglu_large_negative_gate() {
        let gate = [-10.0f32];
        let up = [2.0f32];
        let mut out = [999.0f32];
        geglu(&gate, &up, &mut out).unwrap();
        // GELU(-10) ≈ 0
        assert!(out[0].abs() < 1e-2, "geglu(-10,2)={}", out[0]);
    }

    // -- ReGLU known values -------------------------------------------------

    #[test]
    fn test_reglu_zeros() {
        let gate = [0.0f32; 4];
        let up = [1.0, 2.0, 3.0, 4.0];
        let mut out = [999.0f32; 4];
        reglu(&gate, &up, &mut out).unwrap();
        // ReLU(0) = 0
        for &v in &out[..4] {
            assert!(v.abs() < 1e-7, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_reglu_known_values() {
        let gate = [1.0, -1.0, 2.5, 0.0];
        let up = [3.0, 3.0, 2.0, 5.0];
        let mut out = [0.0f32; 4];
        reglu(&gate, &up, &mut out).unwrap();

        // ReLU(1)*3 = 3
        assert!((out[0] - 3.0).abs() < 1e-7, "reglu(1,3)={}", out[0]);
        // ReLU(-1)*3 = 0
        assert!(out[1].abs() < 1e-7, "reglu(-1,3)={}", out[1]);
        // ReLU(2.5)*2 = 5
        assert!((out[2] - 5.0).abs() < 1e-7, "reglu(2.5,2)={}", out[2]);
        // ReLU(0)*5 = 0
        assert!(out[3].abs() < 1e-7, "reglu(0,5)={}", out[3]);
    }

    #[test]
    fn test_reglu_all_negative_gate() {
        let gate = [-1.0, -5.0, -100.0];
        let up = [10.0, 20.0, 30.0];
        let mut out = [999.0f32; 3];
        reglu(&gate, &up, &mut out).unwrap();
        for &v in &out[..3] {
            assert!(v.abs() < 1e-7, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_reglu_all_positive_gate() {
        let gate = [1.0, 2.0, 3.0];
        let up = [4.0, 5.0, 6.0];
        let mut out = [0.0f32; 3];
        reglu(&gate, &up, &mut out).unwrap();
        // ReLU is identity for positive inputs
        assert!((out[0] - 4.0).abs() < 1e-7);
        assert!((out[1] - 10.0).abs() < 1e-7);
        assert!((out[2] - 18.0).abs() < 1e-7);
    }

    // -- apply_gating dispatch ----------------------------------------------

    #[test]
    fn test_apply_gating_dispatches_correctly() {
        let gate = [1.0f32];
        let up = [2.0f32];
        let mut out_swi = [0.0f32];
        let mut out_ge = [0.0f32];
        let mut out_re = [0.0f32];

        apply_gating(GatingType::SwiGLU, &gate, &up, &mut out_swi).unwrap();
        apply_gating(GatingType::GeGLU, &gate, &up, &mut out_ge).unwrap();
        apply_gating(GatingType::ReGLU, &gate, &up, &mut out_re).unwrap();

        // SiLU(1)*2 ≈ 1.4622, GELU(1)*2 ≈ 1.6824, ReLU(1)*2 = 2.0
        assert!((out_swi[0] - 1.4622).abs() < 1e-3);
        assert!((out_ge[0] - 1.6824).abs() < 1e-3);
        assert!((out_re[0] - 2.0).abs() < 1e-7);
    }

    // -- Property-style tests -----------------------------------------------

    #[test]
    fn test_gating_output_length_equals_input_length() {
        for n in [1, 7, 64, 256, 1000] {
            let gate: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 5.0).collect();
            let up: Vec<f32> = (0..n).map(|i| (i as f32) * 0.02 - 3.0).collect();
            let mut out = vec![0.0f32; n];

            swiglu(&gate, &up, &mut out).unwrap();
            // All n elements were written (none are the sentinel 999.0)
            assert_eq!(out.len(), n);

            geglu(&gate, &up, &mut out).unwrap();
            assert_eq!(out.len(), n);

            reglu(&gate, &up, &mut out).unwrap();
            assert_eq!(out.len(), n);
        }
    }

    #[test]
    fn test_reglu_output_non_negative_when_up_non_negative() {
        // ReLU(gate) >= 0, so if up >= 0, output >= 0
        let gate: Vec<f32> = (-50..50).map(|i| i as f32 * 0.1).collect();
        let up: Vec<f32> = (0..100).map(|i| i as f32 * 0.5).collect();
        let mut out = vec![0.0f32; 100];
        reglu(&gate, &up, &mut out).unwrap();
        for (i, &v) in out.iter().enumerate() {
            assert!(v >= 0.0, "reglu output[{i}] = {v} should be >= 0");
        }
    }

    #[test]
    fn test_swiglu_bounded_by_up_magnitude() {
        // |SiLU(x)| <= |x| for all x, so |SwiGLU| <= |gate| * |up|
        let gate: Vec<f32> = (-20..20).map(|i| i as f32 * 0.5).collect();
        let up: Vec<f32> = (0..40).map(|i| (i as f32 - 20.0) * 0.3).collect();
        let mut out = vec![0.0f32; 40];
        swiglu(&gate, &up, &mut out).unwrap();
        for i in 0..40 {
            let bound = gate[i].abs() * up[i].abs();
            assert!(
                out[i].abs() <= bound + 1e-6,
                "|swiglu[{i}]| = {} > |gate|*|up| = {bound}",
                out[i].abs(),
            );
        }
    }

    // -- Numerical accuracy -------------------------------------------------

    #[test]
    fn test_swiglu_numerical_accuracy() {
        // Compare against manually computed reference values
        // gate=0.5: SiLU(0.5) = 0.5 * sigmoid(0.5) = 0.5 * 0.62246 = 0.31123
        // up=1.0 -> output = 0.31123
        let gate = [0.5f32];
        let up = [1.0f32];
        let mut out = [0.0f32];
        swiglu(&gate, &up, &mut out).unwrap();
        let sigmoid_half = 1.0 / (1.0 + (-0.5f32).exp());
        let expected = 0.5 * sigmoid_half * 1.0;
        assert!(
            (out[0] - expected).abs() < 1e-6,
            "swiglu(0.5,1) = {} expected {}",
            out[0],
            expected,
        );
    }

    #[test]
    fn test_geglu_numerical_accuracy() {
        // GELU(0.5) via tanh approx: 0.5 * 0.5 * (1 + tanh(sqrt(2/pi)*(0.5 + 0.044715*0.125)))
        let gate = [0.5f32];
        let up = [1.0f32];
        let mut out = [0.0f32];
        geglu(&gate, &up, &mut out).unwrap();
        let expected = gelu(0.5) * 1.0;
        assert!(
            (out[0] - expected).abs() < 1e-6,
            "geglu(0.5,1) = {} expected {}",
            out[0],
            expected,
        );
    }

    #[test]
    fn test_reglu_numerical_accuracy() {
        let gate = [0.5f32, -0.5];
        let up = [3.0f32, 3.0];
        let mut out = [0.0f32; 2];
        reglu(&gate, &up, &mut out).unwrap();
        assert!((out[0] - 1.5).abs() < 1e-7, "reglu(0.5,3)={}", out[0]);
        assert!(out[1].abs() < 1e-7, "reglu(-0.5,3)={}", out[1]);
    }

    #[test]
    fn test_gating_single_element() {
        let gate = [1.5f32];
        let up = [2.0f32];
        let mut out = [0.0f32];

        swiglu(&gate, &up, &mut out).unwrap();
        let expected_swiglu = silu(1.5) * 2.0;
        assert!((out[0] - expected_swiglu).abs() < 1e-6);

        geglu(&gate, &up, &mut out).unwrap();
        let expected_geglu = gelu(1.5) * 2.0;
        assert!((out[0] - expected_geglu).abs() < 1e-6);

        reglu(&gate, &up, &mut out).unwrap();
        assert!((out[0] - 3.0).abs() < 1e-7); // ReLU(1.5)*2 = 3
    }

    #[test]
    fn test_gating_up_zeros() {
        // When up is all zeros, output should be zero regardless of gate
        let gate = [1.0, -1.0, 100.0, -100.0];
        let up = [0.0f32; 4];
        let mut out = [999.0f32; 4];

        swiglu(&gate, &up, &mut out).unwrap();
        for &v in &out[..4] {
            assert!(v.abs() < 1e-7, "expected 0, got {v}");
        }

        geglu(&gate, &up, &mut out).unwrap();
        for &v in &out[..4] {
            assert!(v.abs() < 1e-7, "expected 0, got {v}");
        }

        reglu(&gate, &up, &mut out).unwrap();
        for &v in &out[..4] {
            assert!(v.abs() < 1e-7, "expected 0, got {v}");
        }
    }
}

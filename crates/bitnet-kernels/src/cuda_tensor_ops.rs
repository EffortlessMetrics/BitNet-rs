//! CUDA-style element-wise tensor operations with CPU reference implementations.
//!
//! Provides unary/binary elementwise ops, fused multiply-add, broadcasting,
//! and tensor comparison utilities suitable for cross-validation against GPU kernels.

use std::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by tensor operations.
#[derive(Debug, Clone, PartialEq)]
pub enum TensorOpError {
    /// Operand shapes are incompatible for the requested broadcast.
    ShapeMismatch { a_len: usize, b_len: usize },
    /// An input slice was unexpectedly empty.
    EmptyTensor,
    /// Division by zero encountered at the given index.
    DivisionByZero { index: usize },
    /// A domain error (e.g. log of negative number).
    DomainError { op: String, index: usize, value: f32 },
}

impl fmt::Display for TensorOpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch { a_len, b_len } => {
                write!(f, "shape mismatch: a.len()={a_len}, b.len()={b_len}")
            }
            Self::EmptyTensor => write!(f, "empty tensor"),
            Self::DivisionByZero { index } => {
                write!(f, "division by zero at index {index}")
            }
            Self::DomainError { op, index, value } => {
                write!(f, "domain error in {op} at index {index}: value={value}")
            }
        }
    }
}

impl std::error::Error for TensorOpError {}

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Element-wise operations supported on tensors.
#[derive(Debug, Clone, PartialEq)]
pub enum TensorOp {
    // Binary arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,

    // Unary
    Abs,
    Neg,
    Sqrt,
    Rsqrt,
    Exp,
    Log,
    Tanh,
    Sigmoid,
    Gelu,
    Silu,
    Relu,
    LeakyRelu(f32),
    Clamp { min: f32, max: f32 },
}

/// Broadcasting rules for binary operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BroadcastRule {
    /// Both tensors have the same length – no broadcast needed.
    NoBroadcast,
    /// `b` is a scalar (len 1) broadcast to the shape of `a`.
    ScalarB,
    /// `a` is a scalar (len 1) broadcast to the shape of `b`.
    ScalarA,
}

/// Result of comparing two tensors element-wise.
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    pub all_close: bool,
    pub max_abs_diff: f32,
    pub max_rel_diff: f32,
    pub num_mismatches: usize,
    pub first_mismatch_index: Option<usize>,
}

// ---------------------------------------------------------------------------
// Broadcast helpers
// ---------------------------------------------------------------------------

/// Determine the broadcast rule for two slices, or return an error.
pub fn resolve_broadcast(a: &[f32], b: &[f32]) -> Result<BroadcastRule, TensorOpError> {
    if a.len() == b.len() {
        Ok(BroadcastRule::NoBroadcast)
    } else if b.len() == 1 {
        Ok(BroadcastRule::ScalarB)
    } else if a.len() == 1 {
        Ok(BroadcastRule::ScalarA)
    } else {
        Err(TensorOpError::ShapeMismatch { a_len: a.len(), b_len: b.len() })
    }
}

// ---------------------------------------------------------------------------
// Core math helpers (scalar)
// ---------------------------------------------------------------------------

#[inline]
fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn gelu_f32(x: f32) -> f32 {
    // Approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    x * 0.5 * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

#[inline]
fn silu_f32(x: f32) -> f32 {
    x * sigmoid_f32(x)
}

// ---------------------------------------------------------------------------
// Unary operations
// ---------------------------------------------------------------------------

/// Apply a unary [`TensorOp`] element-wise to `input`.
///
/// Returns `Err` for binary-only ops (`Add`, `Sub`, `Mul`, `Div`, `Max`, `Min`).
pub fn elementwise_unary(op: &TensorOp, input: &[f32]) -> Result<Vec<f32>, TensorOpError> {
    let out = match op {
        TensorOp::Abs => input.iter().map(|x| x.abs()).collect(),
        TensorOp::Neg => input.iter().map(|x| -x).collect(),
        TensorOp::Sqrt => input.iter().map(|x| x.sqrt()).collect(),
        TensorOp::Rsqrt => input.iter().map(|x| 1.0 / x.sqrt()).collect(),
        TensorOp::Exp => input.iter().map(|x| x.exp()).collect(),
        TensorOp::Log => input.iter().map(|x| x.ln()).collect(),
        TensorOp::Tanh => input.iter().map(|x| x.tanh()).collect(),
        TensorOp::Sigmoid => input.iter().map(|x| sigmoid_f32(*x)).collect(),
        TensorOp::Gelu => input.iter().map(|x| gelu_f32(*x)).collect(),
        TensorOp::Silu => input.iter().map(|x| silu_f32(*x)).collect(),
        TensorOp::Relu => input.iter().map(|x| x.max(0.0)).collect(),
        TensorOp::LeakyRelu(alpha) => {
            let a = *alpha;
            input.iter().map(|x| if *x >= 0.0 { *x } else { a * x }).collect()
        }
        TensorOp::Clamp { min, max } => input.iter().map(|x| x.clamp(*min, *max)).collect(),
        // Binary-only ops are not valid for unary application.
        TensorOp::Add
        | TensorOp::Sub
        | TensorOp::Mul
        | TensorOp::Div
        | TensorOp::Max
        | TensorOp::Min => {
            return Err(TensorOpError::DomainError { op: format!("{op:?}"), index: 0, value: 0.0 });
        }
    };
    Ok(out)
}

// ---------------------------------------------------------------------------
// Binary operations
// ---------------------------------------------------------------------------

/// Apply a binary [`TensorOp`] element-wise to `a` and `b`, with broadcasting.
pub fn elementwise_binary(op: &TensorOp, a: &[f32], b: &[f32]) -> Result<Vec<f32>, TensorOpError> {
    let rule = resolve_broadcast(a, b)?;

    let len = match rule {
        BroadcastRule::NoBroadcast | BroadcastRule::ScalarB => a.len(),
        BroadcastRule::ScalarA => b.len(),
    };

    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let va = match rule {
            BroadcastRule::ScalarA => a[0],
            _ => a[i],
        };
        let vb = match rule {
            BroadcastRule::ScalarB => b[0],
            _ => b[i],
        };
        let val = match op {
            TensorOp::Add => va + vb,
            TensorOp::Sub => va - vb,
            TensorOp::Mul => va * vb,
            TensorOp::Div => va / vb,
            TensorOp::Max => va.max(vb),
            TensorOp::Min => va.min(vb),
            // Unary ops: apply op(a) and ignore b (lenient).
            TensorOp::Abs => va.abs(),
            TensorOp::Neg => -va,
            TensorOp::Sqrt => va.sqrt(),
            TensorOp::Rsqrt => 1.0 / va.sqrt(),
            TensorOp::Exp => va.exp(),
            TensorOp::Log => va.ln(),
            TensorOp::Tanh => va.tanh(),
            TensorOp::Sigmoid => sigmoid_f32(va),
            TensorOp::Gelu => gelu_f32(va),
            TensorOp::Silu => silu_f32(va),
            TensorOp::Relu => va.max(0.0),
            TensorOp::LeakyRelu(alpha) => {
                if va >= 0.0 {
                    va
                } else {
                    alpha * va
                }
            }
            TensorOp::Clamp { min, max } => va.clamp(*min, *max),
        };
        out.push(val);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Fused / compound operations
// ---------------------------------------------------------------------------

/// Fused multiply-add: `a * b + c` element-wise (all same length).
pub fn fused_multiply_add(a: &[f32], b: &[f32], c: &[f32]) -> Result<Vec<f32>, TensorOpError> {
    if a.len() != b.len() || b.len() != c.len() {
        return Err(TensorOpError::ShapeMismatch { a_len: a.len(), b_len: b.len() });
    }
    Ok(a.iter().zip(b.iter()).zip(c.iter()).map(|((&va, &vb), &vc)| va.mul_add(vb, vc)).collect())
}

/// `input * scale + shift` element-wise (all same length).
pub fn scale_and_shift(
    input: &[f32],
    scale: &[f32],
    shift: &[f32],
) -> Result<Vec<f32>, TensorOpError> {
    fused_multiply_add(input, scale, shift)
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

/// Compare two tensors element-wise with relative and absolute tolerance.
pub fn compare_tensors(
    a: &[f32],
    b: &[f32],
    rtol: f32,
    atol: f32,
) -> Result<ComparisonResult, TensorOpError> {
    if a.len() != b.len() {
        return Err(TensorOpError::ShapeMismatch { a_len: a.len(), b_len: b.len() });
    }

    let mut max_abs_diff: f32 = 0.0;
    let mut max_rel_diff: f32 = 0.0;
    let mut num_mismatches: usize = 0;
    let mut first_mismatch_index: Option<usize> = None;

    for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
        let abs_diff = (va - vb).abs();
        let rel_diff = if va.abs() > f32::EPSILON { abs_diff / va.abs() } else { abs_diff };
        if abs_diff > max_abs_diff {
            max_abs_diff = abs_diff;
        }
        if rel_diff > max_rel_diff {
            max_rel_diff = rel_diff;
        }
        let close = abs_diff <= atol + rtol * vb.abs();
        if !close {
            num_mismatches += 1;
            if first_mismatch_index.is_none() {
                first_mismatch_index = Some(i);
            }
        }
    }

    Ok(ComparisonResult {
        all_close: num_mismatches == 0,
        max_abs_diff,
        max_rel_diff,
        num_mismatches,
        first_mismatch_index,
    })
}

// ---------------------------------------------------------------------------
// Batch operations
// ---------------------------------------------------------------------------

/// Apply a sequence of independent unary operations, returning one output per entry.
pub fn batch_elementwise(ops: &[(&TensorOp, &[f32])]) -> Result<Vec<Vec<f32>>, TensorOpError> {
    ops.iter().map(|(op, data)| elementwise_unary(op, data)).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol || (a.is_nan() && b.is_nan())
    }

    fn assert_vec_approx(actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(approx_eq(*a, *e, tol), "index {i}: actual={a}, expected={e}, tol={tol}");
        }
    }

    // ── unary ops ───────────────────────────────────────────────────────

    #[test]
    fn test_abs() {
        let r = elementwise_unary(&TensorOp::Abs, &[-1.0, 0.0, 3.5]).unwrap();
        assert_vec_approx(&r, &[1.0, 0.0, 3.5], 1e-6);
    }

    #[test]
    fn test_neg() {
        let r = elementwise_unary(&TensorOp::Neg, &[1.0, -2.0, 0.0]).unwrap();
        assert_vec_approx(&r, &[-1.0, 2.0, 0.0], 1e-6);
    }

    #[test]
    fn test_sqrt() {
        let r = elementwise_unary(&TensorOp::Sqrt, &[4.0, 9.0, 0.0]).unwrap();
        assert_vec_approx(&r, &[2.0, 3.0, 0.0], 1e-6);
    }

    #[test]
    fn test_rsqrt() {
        let r = elementwise_unary(&TensorOp::Rsqrt, &[4.0, 16.0]).unwrap();
        assert_vec_approx(&r, &[0.5, 0.25], 1e-6);
    }

    #[test]
    fn test_exp() {
        let r = elementwise_unary(&TensorOp::Exp, &[0.0, 1.0]).unwrap();
        assert_vec_approx(&r, &[1.0, std::f32::consts::E], 1e-5);
    }

    #[test]
    fn test_log() {
        let r = elementwise_unary(&TensorOp::Log, &[1.0, std::f32::consts::E]).unwrap();
        assert_vec_approx(&r, &[0.0, 1.0], 1e-5);
    }

    #[test]
    fn test_tanh() {
        let r = elementwise_unary(&TensorOp::Tanh, &[0.0, 1.0, -1.0]).unwrap();
        assert_vec_approx(&r, &[0.0, 1.0_f32.tanh(), (-1.0_f32).tanh()], 1e-6);
    }

    #[test]
    fn test_sigmoid() {
        let r = elementwise_unary(&TensorOp::Sigmoid, &[0.0]).unwrap();
        assert_vec_approx(&r, &[0.5], 1e-6);
    }

    #[test]
    fn test_sigmoid_extremes() {
        let r = elementwise_unary(&TensorOp::Sigmoid, &[100.0, -100.0]).unwrap();
        assert!(r[0] > 0.999);
        assert!(r[1] < 0.001);
    }

    #[test]
    fn test_gelu() {
        // GELU(0) == 0
        let r = elementwise_unary(&TensorOp::Gelu, &[0.0, 1.0]).unwrap();
        assert_vec_approx(&r, &[0.0, gelu_f32(1.0)], 1e-5);
    }

    #[test]
    fn test_silu() {
        let r = elementwise_unary(&TensorOp::Silu, &[0.0, 1.0, -1.0]).unwrap();
        assert_vec_approx(&r, &[0.0, silu_f32(1.0), silu_f32(-1.0)], 1e-6);
    }

    #[test]
    fn test_relu() {
        let r = elementwise_unary(&TensorOp::Relu, &[-2.0, 0.0, 3.0]).unwrap();
        assert_vec_approx(&r, &[0.0, 0.0, 3.0], 1e-6);
    }

    #[test]
    fn test_leaky_relu() {
        let r = elementwise_unary(&TensorOp::LeakyRelu(0.1), &[-10.0, 0.0, 5.0]).unwrap();
        assert_vec_approx(&r, &[-1.0, 0.0, 5.0], 1e-6);
    }

    #[test]
    fn test_clamp() {
        let r = elementwise_unary(&TensorOp::Clamp { min: -1.0, max: 1.0 }, &[-5.0, 0.5, 10.0])
            .unwrap();
        assert_vec_approx(&r, &[-1.0, 0.5, 1.0], 1e-6);
    }

    // ── binary ops ──────────────────────────────────────────────────────

    #[test]
    fn test_add() {
        let r = elementwise_binary(&TensorOp::Add, &[1.0, 2.0], &[3.0, 4.0]).unwrap();
        assert_vec_approx(&r, &[4.0, 6.0], 1e-6);
    }

    #[test]
    fn test_sub() {
        let r = elementwise_binary(&TensorOp::Sub, &[5.0, 3.0], &[1.0, 2.0]).unwrap();
        assert_vec_approx(&r, &[4.0, 1.0], 1e-6);
    }

    #[test]
    fn test_mul() {
        let r = elementwise_binary(&TensorOp::Mul, &[2.0, 3.0], &[4.0, 5.0]).unwrap();
        assert_vec_approx(&r, &[8.0, 15.0], 1e-6);
    }

    #[test]
    fn test_div() {
        let r = elementwise_binary(&TensorOp::Div, &[6.0, 9.0], &[2.0, 3.0]).unwrap();
        assert_vec_approx(&r, &[3.0, 3.0], 1e-6);
    }

    #[test]
    fn test_max_binary() {
        let r = elementwise_binary(&TensorOp::Max, &[1.0, 5.0], &[3.0, 2.0]).unwrap();
        assert_vec_approx(&r, &[3.0, 5.0], 1e-6);
    }

    #[test]
    fn test_min_binary() {
        let r = elementwise_binary(&TensorOp::Min, &[1.0, 5.0], &[3.0, 2.0]).unwrap();
        assert_vec_approx(&r, &[1.0, 2.0], 1e-6);
    }

    // ── broadcasting ────────────────────────────────────────────────────

    #[test]
    fn test_broadcast_scalar_b() {
        let r = elementwise_binary(&TensorOp::Add, &[1.0, 2.0, 3.0], &[10.0]).unwrap();
        assert_vec_approx(&r, &[11.0, 12.0, 13.0], 1e-6);
    }

    #[test]
    fn test_broadcast_scalar_a() {
        let r = elementwise_binary(&TensorOp::Mul, &[2.0], &[3.0, 4.0, 5.0]).unwrap();
        assert_vec_approx(&r, &[6.0, 8.0, 10.0], 1e-6);
    }

    #[test]
    fn test_broadcast_shape_mismatch() {
        let r = elementwise_binary(&TensorOp::Add, &[1.0, 2.0], &[3.0, 4.0, 5.0]);
        assert!(r.is_err());
        match r.unwrap_err() {
            TensorOpError::ShapeMismatch { a_len, b_len } => {
                assert_eq!(a_len, 2);
                assert_eq!(b_len, 3);
            }
            e => panic!("unexpected error: {e}"),
        }
    }

    #[test]
    fn test_resolve_broadcast_no_broadcast() {
        assert_eq!(
            resolve_broadcast(&[1.0, 2.0], &[3.0, 4.0]).unwrap(),
            BroadcastRule::NoBroadcast
        );
    }

    #[test]
    fn test_resolve_broadcast_scalar_b() {
        assert_eq!(resolve_broadcast(&[1.0, 2.0], &[5.0]).unwrap(), BroadcastRule::ScalarB);
    }

    #[test]
    fn test_resolve_broadcast_scalar_a() {
        assert_eq!(resolve_broadcast(&[5.0], &[1.0, 2.0]).unwrap(), BroadcastRule::ScalarA);
    }

    // ── fused_multiply_add / scale_and_shift ────────────────────────────

    #[test]
    fn test_fused_multiply_add() {
        let r = fused_multiply_add(&[2.0, 3.0], &[4.0, 5.0], &[1.0, 1.0]).unwrap();
        assert_vec_approx(&r, &[9.0, 16.0], 1e-6);
    }

    #[test]
    fn test_fma_shape_mismatch() {
        let r = fused_multiply_add(&[1.0], &[2.0, 3.0], &[4.0]);
        assert!(r.is_err());
    }

    #[test]
    fn test_scale_and_shift() {
        let r = scale_and_shift(&[1.0, 2.0], &[3.0, 4.0], &[0.5, 0.5]).unwrap();
        // 1*3+0.5, 2*4+0.5
        assert_vec_approx(&r, &[3.5, 8.5], 1e-6);
    }

    // ── compare_tensors ─────────────────────────────────────────────────

    #[test]
    fn test_compare_identical() {
        let a = [1.0, 2.0, 3.0];
        let r = compare_tensors(&a, &a, 1e-5, 1e-8).unwrap();
        assert!(r.all_close);
        assert_eq!(r.num_mismatches, 0);
    }

    #[test]
    fn test_compare_within_tol() {
        let a = [1.0, 2.0, 3.0];
        let b = [1.0 + 1e-7, 2.0, 3.0];
        let r = compare_tensors(&a, &b, 1e-5, 1e-6).unwrap();
        assert!(r.all_close);
    }

    #[test]
    fn test_compare_mismatch() {
        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 2.5, 3.0];
        let r = compare_tensors(&a, &b, 1e-5, 1e-5).unwrap();
        assert!(!r.all_close);
        assert_eq!(r.num_mismatches, 1);
        assert_eq!(r.first_mismatch_index, Some(1));
    }

    #[test]
    fn test_compare_shape_mismatch() {
        let r = compare_tensors(&[1.0], &[1.0, 2.0], 0.0, 0.0);
        assert!(r.is_err());
    }

    // ── batch_elementwise ───────────────────────────────────────────────

    #[test]
    fn test_batch_elementwise_basic() {
        let ops: Vec<(&TensorOp, &[f32])> =
            vec![(&TensorOp::Abs, &[-1.0, -2.0]), (&TensorOp::Relu, &[-1.0, 3.0])];
        let results = batch_elementwise(&ops).unwrap();
        assert_vec_approx(&results[0], &[1.0, 2.0], 1e-6);
        assert_vec_approx(&results[1], &[0.0, 3.0], 1e-6);
    }

    #[test]
    fn test_batch_elementwise_empty_list() {
        let ops: Vec<(&TensorOp, &[f32])> = vec![];
        let results = batch_elementwise(&ops).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_batch_fails_on_binary_op() {
        let ops: Vec<(&TensorOp, &[f32])> = vec![(&TensorOp::Add, &[1.0])];
        assert!(batch_elementwise(&ops).is_err());
    }

    // ── edge cases: empty tensors ───────────────────────────────────────

    #[test]
    fn test_unary_empty() {
        let r = elementwise_unary(&TensorOp::Relu, &[]).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn test_binary_empty() {
        let r = elementwise_binary(&TensorOp::Add, &[], &[]).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn test_fma_empty() {
        let r = fused_multiply_add(&[], &[], &[]).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn test_compare_empty() {
        let r = compare_tensors(&[], &[], 0.0, 0.0).unwrap();
        assert!(r.all_close);
        assert_eq!(r.num_mismatches, 0);
    }

    // ── edge cases: NaN / Inf ───────────────────────────────────────────

    #[test]
    fn test_nan_propagation_add() {
        let r = elementwise_binary(&TensorOp::Add, &[f32::NAN], &[1.0]).unwrap();
        assert!(r[0].is_nan());
    }

    #[test]
    fn test_inf_add() {
        let r = elementwise_binary(&TensorOp::Add, &[f32::INFINITY], &[1.0]).unwrap();
        assert!(r[0].is_infinite() && r[0].is_sign_positive());
    }

    #[test]
    fn test_neg_inf_relu() {
        let r = elementwise_unary(&TensorOp::Relu, &[f32::NEG_INFINITY]).unwrap();
        assert_eq!(r[0], 0.0);
    }

    #[test]
    fn test_sqrt_negative_nan() {
        let r = elementwise_unary(&TensorOp::Sqrt, &[-1.0]).unwrap();
        assert!(r[0].is_nan());
    }

    #[test]
    fn test_log_zero() {
        let r = elementwise_unary(&TensorOp::Log, &[0.0]).unwrap();
        assert!(r[0].is_infinite() && r[0].is_sign_negative());
    }

    #[test]
    fn test_div_by_zero_inf() {
        let r = elementwise_binary(&TensorOp::Div, &[1.0], &[0.0]).unwrap();
        assert!(r[0].is_infinite());
    }

    #[test]
    fn test_exp_large() {
        let r = elementwise_unary(&TensorOp::Exp, &[1000.0]).unwrap();
        assert!(r[0].is_infinite());
    }

    #[test]
    fn test_tanh_large() {
        let r = elementwise_unary(&TensorOp::Tanh, &[100.0, -100.0]).unwrap();
        assert!((r[0] - 1.0).abs() < 1e-6);
        assert!((r[1] + 1.0).abs() < 1e-6);
    }

    // ── misc ────────────────────────────────────────────────────────────

    #[test]
    fn test_leaky_relu_zero_alpha() {
        let r = elementwise_unary(&TensorOp::LeakyRelu(0.0), &[-5.0, 0.0, 5.0]).unwrap();
        assert_vec_approx(&r, &[0.0, 0.0, 5.0], 1e-6);
    }

    #[test]
    fn test_clamp_wide_range() {
        let r =
            elementwise_unary(&TensorOp::Clamp { min: -100.0, max: 100.0 }, &[-200.0, 0.0, 200.0])
                .unwrap();
        assert_vec_approx(&r, &[-100.0, 0.0, 100.0], 1e-6);
    }

    #[test]
    fn test_gelu_negative() {
        let r = elementwise_unary(&TensorOp::Gelu, &[-3.0]).unwrap();
        // GELU(-3) ≈ -0.00404 (very small negative)
        assert!(r[0] < 0.0);
        assert!(r[0] > -0.1);
    }

    #[test]
    fn test_silu_negative() {
        let r = elementwise_unary(&TensorOp::Silu, &[-5.0]).unwrap();
        assert!(r[0] < 0.0);
    }

    #[test]
    fn test_binary_unary_op_uses_a() {
        // Binary call with a unary op applies op to `a` values only.
        let r = elementwise_binary(&TensorOp::Abs, &[-3.0, 4.0], &[99.0, 99.0]).unwrap();
        assert_vec_approx(&r, &[3.0, 4.0], 1e-6);
    }

    #[test]
    fn test_error_display() {
        let e = TensorOpError::ShapeMismatch { a_len: 2, b_len: 3 };
        let s = format!("{e}");
        assert!(s.contains("2"));
        assert!(s.contains("3"));
    }

    #[test]
    fn test_comparison_result_fields() {
        let a = [0.0, 1.0, 2.0, 3.0];
        let b = [0.0, 1.0, 2.0, 4.0];
        let r = compare_tensors(&a, &b, 0.0, 0.0).unwrap();
        assert!(!r.all_close);
        assert_eq!(r.max_abs_diff, 1.0);
        assert_eq!(r.first_mismatch_index, Some(3));
    }

    #[test]
    fn test_scale_and_shift_identity() {
        let input = [1.0, 2.0, 3.0];
        let ones = [1.0, 1.0, 1.0];
        let zeros = [0.0, 0.0, 0.0];
        let r = scale_and_shift(&input, &ones, &zeros).unwrap();
        assert_vec_approx(&r, &input, 1e-6);
    }

    #[test]
    fn test_fma_with_negative() {
        let r = fused_multiply_add(&[-1.0], &[2.0], &[3.0]).unwrap();
        assert_vec_approx(&r, &[1.0], 1e-6);
    }

    // ── proptest ────────────────────────────────────────────────────────

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// relu(x) >= 0 for all finite x.
            #[test]
            fn relu_non_negative(x in -1e6_f32..1e6_f32) {
                let r = elementwise_unary(&TensorOp::Relu, &[x]).unwrap();
                prop_assert!(r[0] >= 0.0, "relu({x}) = {} < 0", r[0]);
            }

            /// sigmoid(x) ∈ [0, 1] for all finite x.
            #[test]
            fn sigmoid_range(x in -500.0_f32..500.0_f32) {
                let r = elementwise_unary(&TensorOp::Sigmoid, &[x]).unwrap();
                prop_assert!(r[0] >= 0.0 && r[0] <= 1.0,
                    "sigmoid({x}) = {} outside [0,1]", r[0]);
            }

            /// add is commutative: a + b == b + a.
            #[test]
            fn add_commutative(
                a in proptest::collection::vec(-1e4_f32..1e4, 1..64),
                b in proptest::collection::vec(-1e4_f32..1e4, 1..64),
            ) {
                // Only test when lengths match.
                if a.len() == b.len() {
                    let ab = elementwise_binary(&TensorOp::Add, &a, &b).unwrap();
                    let ba = elementwise_binary(&TensorOp::Add, &b, &a).unwrap();
                    for (i, (x, y)) in ab.iter().zip(ba.iter()).enumerate() {
                        prop_assert!((x - y).abs() < 1e-4,
                            "commutative violation at {i}: {x} vs {y}");
                    }
                }
            }

            /// fma(a,b,c) ≈ a*b + c for finite values.
            #[test]
            fn fma_matches_manual(
                vals in proptest::collection::vec(-100.0_f32..100.0, 3..33),
            ) {
                let n = vals.len() / 3;
                if n == 0 { return Ok(()); }
                let a = &vals[..n];
                let b = &vals[n..2*n];
                let c = &vals[2*n..3*n];
                let fma_result = fused_multiply_add(a, b, c).unwrap();
                for i in 0..n {
                    let expected = a[i] * b[i] + c[i];
                    prop_assert!((fma_result[i] - expected).abs() < 1e-3,
                        "fma mismatch at {i}: {} vs {expected}", fma_result[i]);
                }
            }

            /// abs(x) >= 0 and abs(x) == abs(-x) for all finite x.
            #[test]
            fn abs_symmetry(x in -1e6_f32..1e6_f32) {
                let pos = elementwise_unary(&TensorOp::Abs, &[x]).unwrap();
                let neg = elementwise_unary(&TensorOp::Abs, &[-x]).unwrap();
                prop_assert!(pos[0] >= 0.0);
                prop_assert!((pos[0] - neg[0]).abs() < 1e-6,
                    "|{x}|={} != |{}|={}", pos[0], -x, neg[0]);
            }
        }
    }
}

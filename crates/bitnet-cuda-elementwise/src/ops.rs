//! Core element-wise arithmetic operations.
//!
//! Every public function has a CPU reference implementation that is always
//! compiled.  When the `gpu` or `cuda` feature is active a CUDA kernel
//! launcher stub is included (actual PTX dispatch is a future extension).

use crate::broadcast::BroadcastShape;
use crate::error::{ElementWiseError, Result};

// ── helpers ────────────────────────────────────────────────────────

/// Apply a binary `op` element-wise, respecting `shape` broadcast rules.
fn broadcast_binop(
    lhs: &[f32],
    rhs: &[f32],
    shape: BroadcastShape,
    op: fn(f32, f32) -> f32,
) -> Vec<f32> {
    let len = shape.output_len();
    let mut out = Vec::with_capacity(len);
    match shape {
        BroadcastShape::Same(_) => {
            for (l, r) in lhs.iter().zip(rhs.iter()) {
                out.push(op(*l, *r));
            }
        }
        BroadcastShape::ScalarLeft(n) => {
            let s = lhs[0];
            for r in &rhs[..n] {
                out.push(op(s, *r));
            }
        }
        BroadcastShape::ScalarRight(n) => {
            let s = rhs[0];
            for l in &lhs[..n] {
                out.push(op(*l, s));
            }
        }
        BroadcastShape::VectorRight { total_len, rhs_len } => {
            for i in 0..total_len {
                out.push(op(lhs[i], rhs[i % rhs_len]));
            }
        }
        BroadcastShape::VectorLeft { total_len, lhs_len } => {
            for i in 0..total_len {
                out.push(op(lhs[i % lhs_len], rhs[i]));
            }
        }
    }
    out
}

/// Apply a binary `op` **in-place** on `lhs`, broadcasting `rhs`.
fn broadcast_binop_inplace(
    lhs: &mut [f32],
    rhs: &[f32],
    shape: BroadcastShape,
    op: fn(f32, f32) -> f32,
) {
    match shape {
        BroadcastShape::Same(_) => {
            for (l, r) in lhs.iter_mut().zip(rhs.iter()) {
                *l = op(*l, *r);
            }
        }
        BroadcastShape::ScalarRight(_) => {
            let s = rhs[0];
            for v in lhs.iter_mut() {
                *v = op(*v, s);
            }
        }
        BroadcastShape::VectorRight { rhs_len, .. } => {
            for i in 0..lhs.len() {
                lhs[i] = op(lhs[i], rhs[i % rhs_len]);
            }
        }
        // ScalarLeft / VectorLeft cannot be applied in-place on `lhs`
        // because lhs is the smaller operand.  We expand into lhs length
        // which is the only valid semantic for in-place mutation.
        BroadcastShape::ScalarLeft(_) | BroadcastShape::VectorLeft { .. } => {
            // For in-place, lhs must be the larger operand.
            // This case is unreachable when called from the public API
            // because we validate shapes before calling.
            unreachable!("in-place ops require lhs to be the larger operand");
        }
    }
}

// ── GPU stubs ──────────────────────────────────────────────────────

/// Marker module for future CUDA kernel launchers.
#[cfg(any(feature = "gpu", feature = "cuda"))]
mod cuda {
    /// Placeholder: launch an element-wise add kernel on the GPU.
    pub fn launch_add(_lhs: &[f32], _rhs: &[f32], _out: &mut [f32]) {
        // Future: cuLaunchKernel for elementwise_add.ptx
        unimplemented!("CUDA element-wise add kernel not yet wired");
    }

    /// Placeholder: launch an element-wise FMA kernel on the GPU.
    pub fn launch_fma(_a: &[f32], _b: &[f32], _c: &[f32], _out: &mut [f32]) {
        unimplemented!("CUDA FMA kernel not yet wired");
    }
}

// ── public API ─────────────────────────────────────────────────────

/// Element-wise addition with broadcasting.
///
/// # Errors
///
/// Returns an error if the shapes cannot be broadcast.
pub fn add(lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>> {
    let shape = BroadcastShape::resolve(lhs.len(), rhs.len())?;
    Ok(broadcast_binop(lhs, rhs, shape, |a, b| a + b))
}

/// In-place element-wise addition: `lhs[i] += rhs[broadcast(i)]`.
///
/// # Errors
///
/// Returns an error if `rhs` cannot be broadcast into `lhs`.
pub fn add_inplace(lhs: &mut [f32], rhs: &[f32]) -> Result<()> {
    let shape = resolve_inplace(lhs.len(), rhs.len())?;
    broadcast_binop_inplace(lhs, rhs, shape, |a, b| a + b);
    Ok(())
}

/// Element-wise subtraction with broadcasting.
///
/// # Errors
///
/// Returns an error if the shapes cannot be broadcast.
pub fn sub(lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>> {
    let shape = BroadcastShape::resolve(lhs.len(), rhs.len())?;
    Ok(broadcast_binop(lhs, rhs, shape, |a, b| a - b))
}

/// In-place element-wise subtraction: `lhs[i] -= rhs[broadcast(i)]`.
///
/// # Errors
///
/// Returns an error if `rhs` cannot be broadcast into `lhs`.
pub fn sub_inplace(lhs: &mut [f32], rhs: &[f32]) -> Result<()> {
    let shape = resolve_inplace(lhs.len(), rhs.len())?;
    broadcast_binop_inplace(lhs, rhs, shape, |a, b| a - b);
    Ok(())
}

/// Element-wise multiplication with broadcasting.
///
/// # Errors
///
/// Returns an error if the shapes cannot be broadcast.
pub fn mul(lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>> {
    let shape = BroadcastShape::resolve(lhs.len(), rhs.len())?;
    Ok(broadcast_binop(lhs, rhs, shape, |a, b| a * b))
}

/// In-place element-wise multiplication: `lhs[i] *= rhs[broadcast(i)]`.
///
/// # Errors
///
/// Returns an error if `rhs` cannot be broadcast into `lhs`.
pub fn mul_inplace(lhs: &mut [f32], rhs: &[f32]) -> Result<()> {
    let shape = resolve_inplace(lhs.len(), rhs.len())?;
    broadcast_binop_inplace(lhs, rhs, shape, |a, b| a * b);
    Ok(())
}

/// Element-wise division with broadcasting.
///
/// NaN propagation: if the divisor contains `NaN`, the result is `NaN` at
/// that position.  An explicit zero in the divisor triggers an error *unless*
/// the numerator is also zero (0/0 → `NaN` is IEEE-754 compliant).
///
/// # Errors
///
/// Returns [`ElementWiseError::DivisionByZero`] when a finite non-zero
/// numerator would be divided by exactly zero.
/// Returns a shape error if the operands cannot be broadcast.
pub fn div(lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>> {
    let shape = BroadcastShape::resolve(lhs.len(), rhs.len())?;
    check_div_by_zero(lhs, rhs, shape)?;
    Ok(broadcast_binop(lhs, rhs, shape, |a, b| a / b))
}

/// In-place element-wise division: `lhs[i] /= rhs[broadcast(i)]`.
///
/// # Errors
///
/// Returns [`ElementWiseError::DivisionByZero`] or a shape error.
pub fn div_inplace(lhs: &mut [f32], rhs: &[f32]) -> Result<()> {
    let shape = resolve_inplace(lhs.len(), rhs.len())?;
    check_div_by_zero(lhs, rhs, shape)?;
    broadcast_binop_inplace(lhs, rhs, shape, |a, b| a / b);
    Ok(())
}

/// Fused multiply-add: `a * b + c`, element-wise with broadcasting.
///
/// All three operands are broadcast against each other pairwise; the output
/// length is the maximum of the three.
///
/// # Errors
///
/// Returns [`ElementWiseError::FmaLengthMismatch`] when the three lengths
/// cannot be reconciled via broadcast.
pub fn fma(a: &[f32], b: &[f32], c: &[f32]) -> Result<Vec<f32>> {
    let ab_shape = BroadcastShape::resolve(a.len(), b.len()).map_err(|_| {
        ElementWiseError::FmaLengthMismatch { a_len: a.len(), b_len: b.len(), c_len: c.len() }
    })?;
    let ab_len = ab_shape.output_len();
    let ab = broadcast_binop(a, b, ab_shape, |x, y| x * y);

    let final_shape = BroadcastShape::resolve(ab_len, c.len()).map_err(|_| {
        ElementWiseError::FmaLengthMismatch { a_len: a.len(), b_len: b.len(), c_len: c.len() }
    })?;
    Ok(broadcast_binop(&ab, c, final_shape, |x, y| x + y))
}

/// In-place fused multiply-add: `a[i] = a[i] * b[broadcast(i)] + c[broadcast(i)]`.
///
/// # Errors
///
/// Returns [`ElementWiseError::FmaLengthMismatch`] on incompatible lengths.
pub fn fma_inplace(a: &mut [f32], b: &[f32], c: &[f32]) -> Result<()> {
    let ab_shape = resolve_inplace(a.len(), b.len()).map_err(|_| {
        ElementWiseError::FmaLengthMismatch { a_len: a.len(), b_len: b.len(), c_len: c.len() }
    })?;
    let c_shape = resolve_inplace(a.len(), c.len()).map_err(|_| {
        ElementWiseError::FmaLengthMismatch { a_len: a.len(), b_len: b.len(), c_len: c.len() }
    })?;

    // Apply multiply then add in-place.
    broadcast_binop_inplace(a, b, ab_shape, |x, y| x * y);
    broadcast_binop_inplace(a, c, c_shape, |x, y| x + y);
    Ok(())
}

// ── internal helpers ───────────────────────────────────────────────

/// Resolve broadcast for in-place ops where `lhs` must be the larger operand.
fn resolve_inplace(lhs_len: usize, rhs_len: usize) -> Result<BroadcastShape> {
    let shape = BroadcastShape::resolve(lhs_len, rhs_len)?;
    match shape {
        BroadcastShape::Same(_)
        | BroadcastShape::ScalarRight(_)
        | BroadcastShape::VectorRight { .. } => Ok(shape),
        BroadcastShape::ScalarLeft(_) | BroadcastShape::VectorLeft { .. } => {
            Err(ElementWiseError::ShapeMismatch { lhs: vec![lhs_len], rhs: vec![rhs_len] })
        }
    }
}

/// Check for hard division-by-zero (finite nonzero / 0.0).
fn check_div_by_zero(lhs: &[f32], rhs: &[f32], shape: BroadcastShape) -> Result<()> {
    let len = shape.output_len();
    for i in 0..len {
        let d = match shape {
            BroadcastShape::Same(_)
            | BroadcastShape::ScalarLeft(_)
            | BroadcastShape::VectorLeft { .. } => rhs[i],
            BroadcastShape::ScalarRight(_) => rhs[0],
            BroadcastShape::VectorRight { rhs_len, .. } => rhs[i % rhs_len],
        };
        let n = match shape {
            BroadcastShape::ScalarLeft(_) => lhs[0],
            BroadcastShape::Same(_)
            | BroadcastShape::ScalarRight(_)
            | BroadcastShape::VectorRight { .. } => lhs[i],
            BroadcastShape::VectorLeft { lhs_len, .. } => lhs[i % lhs_len],
        };
        if d == 0.0 && n.is_finite() && n != 0.0 {
            return Err(ElementWiseError::DivisionByZero);
        }
    }
    Ok(())
}

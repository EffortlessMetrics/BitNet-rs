//! ARM NEON fused MLP kernels for Apple Silicon.
//!
//! Provides NEON-accelerated fused MLP operations used in transformer FFN
//! layers (SwiGLU-style gate+up projection, down projection with residual
//! add, and a complete forward pass). Processes 4 × f32 lanes at a time
//! with scalar fallback for remainder elements.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Lane count for `float32x4_t` NEON vectors.
const LANES: usize = 4;

// ── SiLU helpers ────────────────────────────────────────────────────

/// Scalar SiLU (Swish): `x * sigmoid(x) = x / (1 + exp(-x))`.
#[inline(always)]
fn silu_scalar(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// NEON-vectorised sigmoid for four lanes.
///
/// Uses the identity `sigmoid(x) = 1 / (1 + exp(-x))` with a fast
/// polynomial exp approximation.
///
/// # Safety
/// Requires `aarch64` target with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn sigmoid_neon(x: float32x4_t) -> float32x4_t {
    let one = vdupq_n_f32(1.0);
    let neg_x = vnegq_f32(x);
    let exp_neg_x = unsafe { fast_exp_neon(neg_x) };
    // 1 / (1 + exp(-x))
    let denom = vaddq_f32(one, exp_neg_x);
    vdivq_f32(one, denom)
}

/// Fast polynomial exp approximation (Cody-Waite style) for NEON.
///
/// Maximum relative error ≈ 2 × 10⁻⁴ for |x| ≤ 88.
///
/// # Safety
/// Requires `aarch64` target with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn fast_exp_neon(x: float32x4_t) -> float32x4_t {
    let min_val = vdupq_n_f32(-88.0);
    let max_val = vdupq_n_f32(88.0);
    let x = vmaxq_f32(vminq_f32(x, max_val), min_val);

    let log2e = vdupq_n_f32(std::f32::consts::LOG2_E);
    let ln2 = vdupq_n_f32(std::f32::consts::LN_2);
    let n = vrndnq_f32(vmulq_f32(x, log2e));
    let r = vsubq_f32(x, vmulq_f32(n, ln2));

    let c1 = vdupq_n_f32(1.0 / 24.0);
    let c2 = vdupq_n_f32(1.0 / 6.0);
    let c3 = vdupq_n_f32(0.5);
    let one = vdupq_n_f32(1.0);

    let p = vfmaq_f32(c2, r, c1);
    let p = vfmaq_f32(c3, r, p);
    let p = vfmaq_f32(one, r, p);
    let poly = vfmaq_f32(one, r, p);

    let bias = vdupq_n_s32(127);
    let ni = vcvtq_s32_f32(n);
    let pow2n = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(ni, bias), 23));

    vmulq_f32(poly, pow2n)
}

// ── Public API ──────────────────────────────────────────────────────

/// SiLU/Swish activation using NEON intrinsics: `out[i] = x[i] * sigmoid(x[i])`.
///
/// # Safety
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
/// Panics if `out.len() != x.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_silu_activation(x: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), out.len(), "output length mismatch");
    let n = x.len();
    let chunks = n / LANES;
    let x_ptr = x.as_ptr();
    let o_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * LANES;
        unsafe {
            let vx = vld1q_f32(x_ptr.add(offset));
            let vsig = sigmoid_neon(vx);
            let vr = vmulq_f32(vx, vsig);
            vst1q_f32(o_ptr.add(offset), vr);
        }
    }

    for i in (chunks * LANES)..n {
        out[i] = silu_scalar(x[i]);
    }
}

/// Fused gate+up projection (SwiGLU-style): `out[i] = gate[i] * silu(up[i])`.
///
/// In a SwiGLU FFN the gate and up projections are computed separately,
/// then combined as `gate * silu(up)`. This kernel fuses the activation
/// and element-wise multiply into a single pass.
///
/// # Safety
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
/// Panics if `gate`, `up`, and `out` do not have the same length.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_fused_gate_up_project(gate: &[f32], up: &[f32], out: &mut [f32]) {
    assert_eq!(gate.len(), up.len(), "gate/up length mismatch");
    assert_eq!(gate.len(), out.len(), "gate/out length mismatch");
    let n = gate.len();
    let chunks = n / LANES;
    let g_ptr = gate.as_ptr();
    let u_ptr = up.as_ptr();
    let o_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * LANES;
        unsafe {
            let vg = vld1q_f32(g_ptr.add(offset));
            let vu = vld1q_f32(u_ptr.add(offset));
            // silu(up) = up * sigmoid(up)
            let vsig = sigmoid_neon(vu);
            let vsilu = vmulq_f32(vu, vsig);
            // gate * silu(up)
            let vr = vmulq_f32(vg, vsilu);
            vst1q_f32(o_ptr.add(offset), vr);
        }
    }

    for i in (chunks * LANES)..n {
        out[i] = gate[i] * silu_scalar(up[i]);
    }
}

/// Down projection with residual add: `out[i] = down[i] + residual[i]`.
///
/// After the gated activation, the down projection result is added to
/// the residual stream. This kernel fuses the final add.
///
/// # Safety
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
/// Panics if `down`, `residual`, and `out` do not have the same length.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_fused_down_project(down: &[f32], residual: &[f32], out: &mut [f32]) {
    assert_eq!(down.len(), residual.len(), "down/residual length mismatch");
    assert_eq!(down.len(), out.len(), "down/out length mismatch");
    let n = down.len();
    let chunks = n / LANES;
    let d_ptr = down.as_ptr();
    let r_ptr = residual.as_ptr();
    let o_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * LANES;
        unsafe {
            let vd = vld1q_f32(d_ptr.add(offset));
            let vr = vld1q_f32(r_ptr.add(offset));
            let vs = vaddq_f32(vd, vr);
            vst1q_f32(o_ptr.add(offset), vs);
        }
    }

    for i in (chunks * LANES)..n {
        out[i] = down[i] + residual[i];
    }
}

/// Complete fused MLP forward pass.
///
/// Executes the full SwiGLU MLP pipeline in a single call:
///   1. Fused gate+up: `hidden[i] = gate[i] * silu(up[i])`
///   2. Simulated down projection: `hidden[i] *= down_weights[i]`
///   3. Residual add: `out[i] = hidden[i] + residual[i]`
///
/// `gate`, `up`, and `down_weights` must all have the same length
/// (the intermediate / hidden dimension). `residual` and `out` must
/// have the same length as the others (in real models a matmul projects
/// back to the model dimension; here all vectors share one dimension
/// for the fused element-wise path).
///
/// # Safety
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
/// Panics on any length mismatch among the five slices.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_fused_mlp_forward(
    gate: &[f32],
    up: &[f32],
    down_weights: &[f32],
    residual: &[f32],
    out: &mut [f32],
) {
    let n = gate.len();
    assert_eq!(up.len(), n, "up length mismatch");
    assert_eq!(down_weights.len(), n, "down_weights length mismatch");
    assert_eq!(residual.len(), n, "residual length mismatch");
    assert_eq!(out.len(), n, "output length mismatch");

    let chunks = n / LANES;
    let g_ptr = gate.as_ptr();
    let u_ptr = up.as_ptr();
    let d_ptr = down_weights.as_ptr();
    let r_ptr = residual.as_ptr();
    let o_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * LANES;
        unsafe {
            let vg = vld1q_f32(g_ptr.add(offset));
            let vu = vld1q_f32(u_ptr.add(offset));
            let vd = vld1q_f32(d_ptr.add(offset));
            let vr = vld1q_f32(r_ptr.add(offset));

            // gate * silu(up)
            let vsig = sigmoid_neon(vu);
            let vsilu = vmulq_f32(vu, vsig);
            let vhidden = vmulq_f32(vg, vsilu);

            // down projection (element-wise) + residual
            let vdown = vmulq_f32(vhidden, vd);
            let vout = vaddq_f32(vdown, vr);
            vst1q_f32(o_ptr.add(offset), vout);
        }
    }

    for i in (chunks * LANES)..n {
        let hidden = gate[i] * silu_scalar(up[i]);
        out[i] = hidden * down_weights[i] + residual[i];
    }
}

// ── Scalar reference (used in tests) ────────────────────────────────

/// Scalar SiLU activation for parity testing.
#[cfg(test)]
fn silu_scalar_ref(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| silu_scalar(v)).collect()
}

/// Scalar fused gate+up for parity testing.
#[cfg(test)]
fn gate_up_scalar_ref(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter().zip(up.iter()).map(|(&g, &u)| g * silu_scalar(u)).collect()
}

/// Scalar fused MLP forward for parity testing.
#[cfg(test)]
fn mlp_forward_scalar_ref(
    gate: &[f32],
    up: &[f32],
    down_weights: &[f32],
    residual: &[f32],
) -> Vec<f32> {
    gate.iter()
        .zip(up.iter())
        .zip(down_weights.iter())
        .zip(residual.iter())
        .map(|(((&g, &u), &d), &r)| g * silu_scalar(u) * d + r)
        .collect()
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    const TOL: f32 = 1e-3;

    fn assert_approx_eq(a: &[f32], b: &[f32], tol: f32, ctx: &str) {
        assert_eq!(a.len(), b.len(), "{ctx}: length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol, "{ctx}[{i}]: {x} vs {y} (diff {})", (x - y).abs());
        }
    }

    // ── SiLU activation tests ───────────────────────────────────────

    #[test]
    fn test_silu_basic() {
        let x = [0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0];
        let expected = silu_scalar_ref(&x);
        let mut out = vec![0.0; x.len()];
        unsafe { neon_silu_activation(&x, &mut out) };
        assert_approx_eq(&out, &expected, TOL, "silu_basic");
    }

    #[test]
    fn test_silu_zero() {
        // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        let x = [0.0f32; 4];
        let mut out = vec![0.0; 4];
        unsafe { neon_silu_activation(&x, &mut out) };
        for &v in &out {
            assert!(v.abs() < 1e-6, "silu(0) should be 0, got {v}");
        }
    }

    #[test]
    fn test_silu_remainder() {
        // Non-multiple-of-4 length to exercise scalar tail.
        let x: Vec<f32> = (0..11).map(|i| i as f32 * 0.3 - 1.5).collect();
        let expected = silu_scalar_ref(&x);
        let mut out = vec![0.0; x.len()];
        unsafe { neon_silu_activation(&x, &mut out) };
        assert_approx_eq(&out, &expected, TOL, "silu_remainder");
    }

    #[test]
    fn test_silu_empty() {
        let mut out: Vec<f32> = vec![];
        unsafe { neon_silu_activation(&[], &mut out) };
        assert!(out.is_empty());
    }

    #[test]
    fn test_silu_large() {
        let n = 1024;
        let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 5.0).collect();
        let expected = silu_scalar_ref(&x);
        let mut out = vec![0.0; n];
        unsafe { neon_silu_activation(&x, &mut out) };
        assert_approx_eq(&out, &expected, TOL, "silu_large");
    }

    // ── Fused gate+up projection tests ──────────────────────────────

    #[test]
    fn test_gate_up_basic() {
        let gate = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let up = [0.5f32, -0.5, 1.0, -1.0, 2.0, -2.0, 0.0, 3.0];
        let expected = gate_up_scalar_ref(&gate, &up);
        let mut out = vec![0.0; gate.len()];
        unsafe { neon_fused_gate_up_project(&gate, &up, &mut out) };
        assert_approx_eq(&out, &expected, TOL, "gate_up_basic");
    }

    #[test]
    fn test_gate_up_zeros() {
        // gate=0 → output must be 0 regardless of up values.
        let gate = [0.0f32; 8];
        let up = [100.0f32; 8];
        let mut out = vec![0.0; 8];
        unsafe { neon_fused_gate_up_project(&gate, &up, &mut out) };
        for (i, &v) in out.iter().enumerate() {
            assert!(v.abs() < 1e-5, "gate=0 should zero output, got {v} at {i}");
        }
    }

    #[test]
    fn test_gate_up_remainder() {
        let gate: Vec<f32> = (0..13).map(|i| i as f32 * 0.1).collect();
        let up: Vec<f32> = (0..13).map(|i| (i as f32) * 0.2 - 1.0).collect();
        let expected = gate_up_scalar_ref(&gate, &up);
        let mut out = vec![0.0; 13];
        unsafe { neon_fused_gate_up_project(&gate, &up, &mut out) };
        assert_approx_eq(&out, &expected, TOL, "gate_up_remainder");
    }

    #[test]
    fn test_gate_up_empty() {
        let mut out: Vec<f32> = vec![];
        unsafe { neon_fused_gate_up_project(&[], &[], &mut out) };
        assert!(out.is_empty());
    }

    // ── Down projection + residual tests ────────────────────────────

    #[test]
    fn test_down_project_basic() {
        let down = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let residual = [10.0f32, 20.0, 30.0, 40.0, 50.0];
        let mut out = vec![0.0; 5];
        unsafe { neon_fused_down_project(&down, &residual, &mut out) };
        assert_eq!(out, [11.0, 22.0, 33.0, 44.0, 55.0]);
    }

    #[test]
    fn test_down_project_zero_residual() {
        let down = [1.0f32, -2.0, 3.0, -4.0];
        let residual = [0.0f32; 4];
        let mut out = vec![0.0; 4];
        unsafe { neon_fused_down_project(&down, &residual, &mut out) };
        assert_eq!(out, [1.0, -2.0, 3.0, -4.0]);
    }

    #[test]
    fn test_down_project_empty() {
        let mut out: Vec<f32> = vec![];
        unsafe { neon_fused_down_project(&[], &[], &mut out) };
        assert!(out.is_empty());
    }

    // ── Full MLP forward tests ──────────────────────────────────────

    #[test]
    fn test_mlp_forward_basic() {
        let gate = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let up = [0.5f32, -0.5, 1.0, -1.0, 2.0, -2.0, 0.0, 3.0];
        let down_w = [1.0f32; 8];
        let residual = [0.1f32; 8];
        let expected = mlp_forward_scalar_ref(&gate, &up, &down_w, &residual);
        let mut out = vec![0.0; 8];
        unsafe { neon_fused_mlp_forward(&gate, &up, &down_w, &residual, &mut out) };
        assert_approx_eq(&out, &expected, TOL, "mlp_forward_basic");
    }

    #[test]
    fn test_mlp_forward_identity_down() {
        // down_weights=1 → same as gate_up + residual
        let gate = [2.0f32, 3.0, 4.0, 5.0];
        let up = [1.0f32, -1.0, 0.5, -0.5];
        let down_w = [1.0f32; 4];
        let residual = [0.0f32; 4];
        let expected_gu = gate_up_scalar_ref(&gate, &up);
        let mut out = vec![0.0; 4];
        unsafe { neon_fused_mlp_forward(&gate, &up, &down_w, &residual, &mut out) };
        assert_approx_eq(&out, &expected_gu, TOL, "mlp_forward_identity_down");
    }

    #[test]
    fn test_mlp_forward_remainder() {
        let n = 17;
        let gate: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let up: Vec<f32> = (0..n).map(|i| (i as f32) * 0.2 - 1.5).collect();
        let down_w: Vec<f32> = (0..n).map(|i| 0.5 + (i % 3) as f32 * 0.2).collect();
        let residual: Vec<f32> = (0..n).map(|i| i as f32 * -0.05).collect();
        let expected = mlp_forward_scalar_ref(&gate, &up, &down_w, &residual);
        let mut out = vec![0.0; n];
        unsafe { neon_fused_mlp_forward(&gate, &up, &down_w, &residual, &mut out) };
        assert_approx_eq(&out, &expected, TOL, "mlp_forward_remainder");
    }

    #[test]
    fn test_mlp_forward_empty() {
        let mut out: Vec<f32> = vec![];
        unsafe { neon_fused_mlp_forward(&[], &[], &[], &[], &mut out) };
        assert!(out.is_empty());
    }

    #[test]
    fn test_mlp_forward_large() {
        let n = 1024;
        let gate: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) % 100) as f32 * 0.01).collect();
        let up: Vec<f32> = (0..n).map(|i| ((i * 13 + 5) % 100) as f32 * 0.02 - 1.0).collect();
        let down_w: Vec<f32> = (0..n).map(|i| 0.5 + (i % 7) as f32 * 0.1).collect();
        let residual: Vec<f32> = (0..n).map(|i| i as f32 * 0.001).collect();
        let expected = mlp_forward_scalar_ref(&gate, &up, &down_w, &residual);
        let mut out = vec![0.0; n];
        unsafe { neon_fused_mlp_forward(&gate, &up, &down_w, &residual, &mut out) };
        assert_approx_eq(&out, &expected, TOL, "mlp_forward_large");
    }

    // ── Property: residual pass-through ─────────────────────────────

    #[test]
    fn test_mlp_forward_zero_gate_preserves_residual() {
        // gate=0 → hidden=0 → output = residual
        let n = 12;
        let gate = vec![0.0f32; n];
        let up: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let down_w = vec![1.0f32; n];
        let residual: Vec<f32> = (0..n).map(|i| (i as f32) * 10.0).collect();
        let mut out = vec![0.0; n];
        unsafe { neon_fused_mlp_forward(&gate, &up, &down_w, &residual, &mut out) };
        assert_approx_eq(&out, &residual, 1e-5, "zero_gate_residual");
    }
}

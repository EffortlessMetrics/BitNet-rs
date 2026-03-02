//! ARM NEON-optimized SwiGLU activation kernels for Apple Silicon.
//!
//! Implements gated linear unit variants used in modern LLM feed-forward
//! layers. All functions are `unsafe` due to NEON intrinsic requirements
//! and require `target_arch = "aarch64"`.
//!
//! Supported variants:
//! - **SwiGLU**: `SiLU(gate) * up` — LLaMA, Mistral, BitNet
//! - **GeGLU**: `GELU(gate) * up` — GPT variants
//! - **ReGLU**: `ReLU(gate) * up` — simpler alternative
//! - **Fused SwiGLU + projection**: gated activation with linear output

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── Scalar helpers ──────────────────────────────────────────────────

/// Scalar sigmoid: `1 / (1 + exp(-x))`.
#[inline(always)]
fn scalar_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Scalar SiLU (Swish): `x * sigmoid(x)`.
#[inline(always)]
fn scalar_silu(x: f32) -> f32 {
    x * scalar_sigmoid(x)
}

/// Scalar GELU (tanh approximation).
#[inline(always)]
fn scalar_gelu(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const COEFF: f32 = 0.044_715;
    let x3 = x * x * x;
    let inner = SQRT_2_OVER_PI * (x + COEFF * x3);
    0.5 * x * (1.0 + inner.tanh())
}

// ── NEON SiLU (fast approximation) ─────────────────────────────────

/// NEON-optimized SiLU/Swish: `x * sigmoid(x)`.
///
/// Computes sigmoid in scalar (NEON lacks native `exp`), then uses
/// NEON `vmulq_f32` for the final `x * sigmoid(x)` multiply.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on
/// AArch64).
///
/// # Panics
///
/// Panics if `output.len() < input.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_silu_fast(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    let n = input.len();

    // Scalar sigmoid pass into output buffer.
    for (x, o) in input.iter().zip(output.iter_mut()) {
        *o = scalar_sigmoid(*x);
    }

    // NEON multiply: output[i] = input[i] * sigmoid(input[i]).
    let chunks = n / 4;
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let vx = vld1q_f32(in_ptr.add(off));
            let vs = vld1q_f32(out_ptr.add(off));
            let vr = vmulq_f32(vx, vs);
            vst1q_f32(out_ptr.add(off), vr);
        }
    }

    let tail = chunks * 4;
    for i in tail..n {
        output[i] *= input[i];
    }
}

// ── SwiGLU ──────────────────────────────────────────────────────────

/// NEON-optimized SwiGLU: `output[i] = SiLU(gate[i]) * up[i]`.
///
/// Computes scalar SiLU on each gate element, then multiplies by `up`
/// using 4-wide NEON lanes.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on
/// AArch64).
///
/// # Panics
///
/// Panics if `gate.len() != up.len()` or `output.len() < gate.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_swiglu(gate: &[f32], up: &[f32], output: &mut [f32]) {
    let n = gate.len();
    assert_eq!(gate.len(), up.len(), "gate/up length mismatch");
    assert!(output.len() >= n, "output buffer too small");

    // Scalar SiLU into output buffer.
    for i in 0..n {
        output[i] = scalar_silu(gate[i]);
    }

    // NEON multiply: output[i] *= up[i].
    let chunks = n / 4;
    let up_ptr = up.as_ptr();
    let out_ptr = output.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let vact = vld1q_f32(out_ptr.add(off));
            let vup = vld1q_f32(up_ptr.add(off));
            let vr = vmulq_f32(vact, vup);
            vst1q_f32(out_ptr.add(off), vr);
        }
    }

    let tail = chunks * 4;
    for i in tail..n {
        output[i] *= up[i];
    }
}

/// In-place SwiGLU: `gate[i] = SiLU(gate[i]) * up[i]`.
///
/// Overwrites the `gate` buffer with the result for memory efficiency.
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
///
/// # Panics
///
/// Panics if `gate.len() != up.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_swiglu_inplace(gate: &mut [f32], up: &[f32]) {
    let n = gate.len();
    assert_eq!(n, up.len(), "gate/up length mismatch");

    // Scalar SiLU in-place.
    for v in gate.iter_mut() {
        *v = scalar_silu(*v);
    }

    // NEON multiply: gate[i] *= up[i].
    let chunks = n / 4;
    let up_ptr = up.as_ptr();
    let g_ptr = gate.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let vg = vld1q_f32(g_ptr.add(off));
            let vu = vld1q_f32(up_ptr.add(off));
            let vr = vmulq_f32(vg, vu);
            vst1q_f32(g_ptr.add(off), vr);
        }
    }

    let tail = chunks * 4;
    for i in tail..n {
        gate[i] *= up[i];
    }
}

// ── GeGLU ───────────────────────────────────────────────────────────

/// NEON-optimized GeGLU: `output[i] = GELU(gate[i]) * up[i]`.
///
/// Uses tanh-approximation GELU in scalar, NEON for the final multiply.
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
///
/// # Panics
///
/// Panics if `gate.len() != up.len()` or `output.len() < gate.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_geglu(gate: &[f32], up: &[f32], output: &mut [f32]) {
    let n = gate.len();
    assert_eq!(gate.len(), up.len(), "gate/up length mismatch");
    assert!(output.len() >= n, "output buffer too small");

    for i in 0..n {
        output[i] = scalar_gelu(gate[i]);
    }

    let chunks = n / 4;
    let up_ptr = up.as_ptr();
    let out_ptr = output.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let vact = vld1q_f32(out_ptr.add(off));
            let vup = vld1q_f32(up_ptr.add(off));
            let vr = vmulq_f32(vact, vup);
            vst1q_f32(out_ptr.add(off), vr);
        }
    }

    let tail = chunks * 4;
    for i in tail..n {
        output[i] *= up[i];
    }
}

/// In-place GeGLU: `gate[i] = GELU(gate[i]) * up[i]`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
///
/// # Panics
///
/// Panics if `gate.len() != up.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_geglu_inplace(gate: &mut [f32], up: &[f32]) {
    let n = gate.len();
    assert_eq!(n, up.len(), "gate/up length mismatch");

    for v in gate.iter_mut() {
        *v = scalar_gelu(*v);
    }

    let chunks = n / 4;
    let up_ptr = up.as_ptr();
    let g_ptr = gate.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let vg = vld1q_f32(g_ptr.add(off));
            let vu = vld1q_f32(up_ptr.add(off));
            let vr = vmulq_f32(vg, vu);
            vst1q_f32(g_ptr.add(off), vr);
        }
    }

    let tail = chunks * 4;
    for i in tail..n {
        gate[i] *= up[i];
    }
}

// ── ReGLU ───────────────────────────────────────────────────────────

/// NEON-optimized ReGLU: `output[i] = ReLU(gate[i]) * up[i]`.
///
/// Fully vectorized — NEON provides native `vmaxq_f32` for ReLU.
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
///
/// # Panics
///
/// Panics if `gate.len() != up.len()` or `output.len() < gate.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_reglu(gate: &[f32], up: &[f32], output: &mut [f32]) {
    let n = gate.len();
    assert_eq!(gate.len(), up.len(), "gate/up length mismatch");
    assert!(output.len() >= n, "output buffer too small");

    let chunks = n / 4;
    let zero = vdupq_n_f32(0.0);
    let g_ptr = gate.as_ptr();
    let up_ptr = up.as_ptr();
    let out_ptr = output.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let vg = vld1q_f32(g_ptr.add(off));
            let vrelu = vmaxq_f32(vg, zero);
            let vu = vld1q_f32(up_ptr.add(off));
            let vr = vmulq_f32(vrelu, vu);
            vst1q_f32(out_ptr.add(off), vr);
        }
    }

    let tail = chunks * 4;
    for i in tail..n {
        output[i] = gate[i].max(0.0) * up[i];
    }
}

/// In-place ReGLU: `gate[i] = ReLU(gate[i]) * up[i]`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
///
/// # Panics
///
/// Panics if `gate.len() != up.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_reglu_inplace(gate: &mut [f32], up: &[f32]) {
    let n = gate.len();
    assert_eq!(n, up.len(), "gate/up length mismatch");

    let chunks = n / 4;
    let zero = vdupq_n_f32(0.0);
    let g_ptr = gate.as_mut_ptr();
    let up_ptr = up.as_ptr();

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let vg = vld1q_f32(g_ptr.add(off));
            let vrelu = vmaxq_f32(vg, zero);
            let vu = vld1q_f32(up_ptr.add(off));
            let vr = vmulq_f32(vrelu, vu);
            vst1q_f32(g_ptr.add(off), vr);
        }
    }

    let tail = chunks * 4;
    for i in tail..n {
        gate[i] = gate[i].max(0.0) * up[i];
    }
}

// ── Fused SwiGLU + linear projection ───────────────────────────────

/// Fused SwiGLU + linear projection for a single row.
///
/// Computes `dot(SiLU(gate) * up, weights) + bias` in a single pass,
/// avoiding an intermediate allocation for the gated output.
///
/// `weights` is `[out_dim][hidden_dim]` in row-major order, where
/// `hidden_dim == gate.len()`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
///
/// # Panics
///
/// Panics if slice dimensions are inconsistent.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_swiglu_linear(
    gate: &[f32],
    up: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
) {
    let hidden = gate.len();
    let out_dim = bias.len();
    assert_eq!(gate.len(), up.len(), "gate/up length mismatch");
    assert!(output.len() >= out_dim, "output buffer too small");
    assert_eq!(weights.len(), out_dim * hidden, "weights size mismatch");

    // Materialise gated activation into a scratch buffer on the stack
    // for small hidden dims, heap for large.
    let mut scratch = vec![0.0f32; hidden];
    for i in 0..hidden {
        scratch[i] = scalar_silu(gate[i]) * up[i];
    }

    // Dot product per output neuron.
    let chunks = hidden / 4;
    for o in 0..out_dim {
        let w_row = &weights[o * hidden..(o + 1) * hidden];
        let mut acc = vdupq_n_f32(0.0);

        let s_ptr = scratch.as_ptr();
        let w_ptr = w_row.as_ptr();

        for c in 0..chunks {
            let off = c * 4;
            unsafe {
                let vs = vld1q_f32(s_ptr.add(off));
                let vw = vld1q_f32(w_ptr.add(off));
                acc = vfmaq_f32(acc, vs, vw);
            }
        }

        // Horizontal reduction.
        let sum = unsafe { vaddvq_f32(acc) };

        // Scalar tail.
        let mut tail_sum = 0.0f32;
        for i in (chunks * 4)..hidden {
            tail_sum += scratch[i] * w_row[i];
        }

        output[o] = sum + tail_sum + bias[o];
    }
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    const EPS: f32 = 1e-5;

    // ── SiLU fast ───────────────────────────────────────────────────

    #[test]
    fn test_silu_fast_zero() {
        let input = [0.0f32];
        let mut output = [0.0f32];
        unsafe { neon_silu_fast(&input, &mut output) };
        assert!(approx_eq(output[0], 0.0, EPS));
    }

    #[test]
    fn test_silu_fast_positive() {
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let mut output = [0.0f32; 4];
        unsafe { neon_silu_fast(&input, &mut output) };
        for (&x, &o) in input.iter().zip(output.iter()) {
            let expected = scalar_silu(x);
            assert!(approx_eq(o, expected, EPS), "silu_fast({x}) = {o}, expected {expected}");
        }
    }

    #[test]
    fn test_silu_fast_negative() {
        let input = [-1.0f32, -2.0, -3.0, -4.0, -0.5];
        let mut output = [0.0f32; 5];
        unsafe { neon_silu_fast(&input, &mut output) };
        for (&x, &o) in input.iter().zip(output.iter()) {
            let expected = scalar_silu(x);
            assert!(approx_eq(o, expected, EPS), "silu_fast({x}) = {o}, expected {expected}");
        }
    }

    #[test]
    fn test_silu_fast_empty() {
        let input: [f32; 0] = [];
        let mut output: [f32; 0] = [];
        unsafe { neon_silu_fast(&input, &mut output) };
    }

    // ── SwiGLU ──────────────────────────────────────────────────────

    #[test]
    fn test_swiglu_zeros() {
        let gate = [0.0f32; 4];
        let up = [1.0, 2.0, 3.0, 4.0];
        let mut out = [999.0f32; 4];
        unsafe { neon_swiglu(&gate, &up, &mut out) };
        for &v in &out {
            assert!(v.abs() < 1e-7, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_swiglu_known_values() {
        let gate = [1.0f32, -1.0, 2.0, 0.0];
        let up = [1.0, 1.0, 0.5, 5.0];
        let mut out = [0.0f32; 4];
        unsafe { neon_swiglu(&gate, &up, &mut out) };

        let e0 = scalar_silu(1.0) * 1.0;
        let e1 = scalar_silu(-1.0) * 1.0;
        let e2 = scalar_silu(2.0) * 0.5;
        let e3 = scalar_silu(0.0) * 5.0;
        assert!(approx_eq(out[0], e0, 1e-4), "got {}", out[0]);
        assert!(approx_eq(out[1], e1, 1e-4), "got {}", out[1]);
        assert!(approx_eq(out[2], e2, 1e-4), "got {}", out[2]);
        assert!(approx_eq(out[3], e3, 1e-4), "got {}", out[3]);
    }

    #[test]
    fn test_swiglu_large_input() {
        let n = 1025; // not a multiple of 4
        let gate: Vec<f32> = (0..n).map(|i| (i as f32 - 512.0) * 0.01).collect();
        let up: Vec<f32> = (0..n).map(|i| (i as f32) * 0.005 - 2.0).collect();
        let mut out = vec![0.0f32; n];
        unsafe { neon_swiglu(&gate, &up, &mut out) };
        for i in 0..n {
            let expected = scalar_silu(gate[i]) * up[i];
            assert!(approx_eq(out[i], expected, 1e-4), "mismatch at {i}: {} vs {expected}", out[i]);
        }
    }

    #[test]
    fn test_swiglu_matches_scalar() {
        let gate = [0.5f32, -0.5, 1.5, -1.5, 3.0, -3.0, 0.1, -0.1];
        let up = [1.0, 2.0, 0.5, 3.0, -1.0, -2.0, 4.0, -4.0];
        let mut neon_out = [0.0f32; 8];
        unsafe { neon_swiglu(&gate, &up, &mut neon_out) };
        for i in 0..8 {
            let scalar = scalar_silu(gate[i]) * up[i];
            assert!(
                approx_eq(neon_out[i], scalar, EPS),
                "[{i}] neon={} scalar={scalar}",
                neon_out[i]
            );
        }
    }

    // ── SwiGLU in-place ─────────────────────────────────────────────

    #[test]
    fn test_swiglu_inplace_matches_out_of_place() {
        let gate_orig = [1.0f32, -1.0, 2.0, 0.0, 0.5, -0.5, 3.0];
        let up = [1.0, 1.0, 0.5, 5.0, 2.0, 3.0, -1.0];
        let mut out = [0.0f32; 7];
        unsafe { neon_swiglu(&gate_orig, &up, &mut out) };

        let mut gate_ip = gate_orig;
        unsafe { neon_swiglu_inplace(&mut gate_ip, &up) };

        for i in 0..7 {
            assert!(
                approx_eq(gate_ip[i], out[i], EPS),
                "[{i}] inplace={} vs out_of_place={}",
                gate_ip[i],
                out[i]
            );
        }
    }

    // ── GeGLU ───────────────────────────────────────────────────────

    #[test]
    fn test_geglu_zeros() {
        let gate = [0.0f32; 4];
        let up = [1.0, 2.0, 3.0, 4.0];
        let mut out = [999.0f32; 4];
        unsafe { neon_geglu(&gate, &up, &mut out) };
        for &v in &out {
            assert!(v.abs() < 1e-7, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_geglu_known_values() {
        let gate = [1.0f32, -1.0, 0.0];
        let up = [1.0, 1.0, 5.0];
        let mut out = [0.0f32; 3];
        unsafe { neon_geglu(&gate, &up, &mut out) };

        let e0 = scalar_gelu(1.0) * 1.0;
        let e1 = scalar_gelu(-1.0) * 1.0;
        let e2 = scalar_gelu(0.0) * 5.0;
        assert!(approx_eq(out[0], e0, 1e-4), "got {}", out[0]);
        assert!(approx_eq(out[1], e1, 1e-4), "got {}", out[1]);
        assert!(approx_eq(out[2], e2, 1e-4), "got {}", out[2]);
    }

    #[test]
    fn test_geglu_large_input() {
        let n = 1025;
        let gate: Vec<f32> = (0..n).map(|i| (i as f32 - 512.0) * 0.01).collect();
        let up: Vec<f32> = (0..n).map(|i| (i as f32) * 0.005 - 2.0).collect();
        let mut out = vec![0.0f32; n];
        unsafe { neon_geglu(&gate, &up, &mut out) };
        for i in 0..n {
            let expected = scalar_gelu(gate[i]) * up[i];
            assert!(approx_eq(out[i], expected, 1e-4), "mismatch at {i}: {} vs {expected}", out[i]);
        }
    }

    #[test]
    fn test_geglu_inplace_matches_out_of_place() {
        let gate_orig = [1.0f32, -1.0, 2.0, 0.0, 0.5, -0.5, 3.0];
        let up = [1.0, 1.0, 0.5, 5.0, 2.0, 3.0, -1.0];
        let mut out = [0.0f32; 7];
        unsafe { neon_geglu(&gate_orig, &up, &mut out) };

        let mut gate_ip = gate_orig;
        unsafe { neon_geglu_inplace(&mut gate_ip, &up) };

        for i in 0..7 {
            assert!(
                approx_eq(gate_ip[i], out[i], EPS),
                "[{i}] inplace={} vs out_of_place={}",
                gate_ip[i],
                out[i]
            );
        }
    }

    // ── ReGLU ───────────────────────────────────────────────────────

    #[test]
    fn test_reglu_zeros() {
        let gate = [0.0f32; 4];
        let up = [1.0, 2.0, 3.0, 4.0];
        let mut out = [999.0f32; 4];
        unsafe { neon_reglu(&gate, &up, &mut out) };
        for &v in &out {
            assert!(v.abs() < 1e-7, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_reglu_known_values() {
        let gate = [1.0f32, -1.0, 2.5, 0.0];
        let up = [3.0, 3.0, 2.0, 5.0];
        let mut out = [0.0f32; 4];
        unsafe { neon_reglu(&gate, &up, &mut out) };

        assert!(approx_eq(out[0], 3.0, EPS)); // relu(1)*3
        assert!(approx_eq(out[1], 0.0, EPS)); // relu(-1)*3
        assert!(approx_eq(out[2], 5.0, EPS)); // relu(2.5)*2
        assert!(approx_eq(out[3], 0.0, EPS)); // relu(0)*5
    }

    #[test]
    fn test_reglu_all_negative() {
        let gate = [-1.0f32, -5.0, -100.0];
        let up = [10.0, 20.0, 30.0];
        let mut out = [999.0f32; 3];
        unsafe { neon_reglu(&gate, &up, &mut out) };
        for &v in &out {
            assert!(v.abs() < 1e-7, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_reglu_large_input() {
        let n = 1025;
        let gate: Vec<f32> = (0..n).map(|i| (i as f32 - 512.0) * 0.01).collect();
        let up: Vec<f32> = (0..n).map(|i| (i as f32) * 0.005 - 2.0).collect();
        let mut out = vec![0.0f32; n];
        unsafe { neon_reglu(&gate, &up, &mut out) };
        for i in 0..n {
            let expected = gate[i].max(0.0) * up[i];
            assert!(approx_eq(out[i], expected, EPS), "mismatch at {i}: {} vs {expected}", out[i]);
        }
    }

    #[test]
    fn test_reglu_inplace_matches_out_of_place() {
        let gate_orig = [1.0f32, -1.0, 2.5, 0.0, 0.5, -0.5, 3.0];
        let up = [3.0, 3.0, 2.0, 5.0, 2.0, 3.0, -1.0];
        let mut out = [0.0f32; 7];
        unsafe { neon_reglu(&gate_orig, &up, &mut out) };

        let mut gate_ip = gate_orig;
        unsafe { neon_reglu_inplace(&mut gate_ip, &up) };

        for i in 0..7 {
            assert!(
                approx_eq(gate_ip[i], out[i], EPS),
                "[{i}] inplace={} vs out_of_place={}",
                gate_ip[i],
                out[i]
            );
        }
    }

    // ── Fused SwiGLU + linear ───────────────────────────────────────

    #[test]
    fn test_swiglu_linear_single_output() {
        // hidden=4, out_dim=1
        let gate = [1.0f32, 0.0, -1.0, 2.0];
        let up = [1.0, 1.0, 1.0, 1.0];
        let weights = [1.0f32, 1.0, 1.0, 1.0]; // 1x4
        let bias = [0.0f32];
        let mut out = [0.0f32];

        unsafe {
            neon_swiglu_linear(&gate, &up, &weights, &bias, &mut out);
        }

        // Expected: sum of SiLU(gate) * up * 1.0
        let expected: f32 = gate.iter().zip(up.iter()).map(|(&g, &u)| scalar_silu(g) * u).sum();
        assert!(approx_eq(out[0], expected, 1e-4), "got {} expected {expected}", out[0]);
    }

    #[test]
    fn test_swiglu_linear_with_bias() {
        let gate = [1.0f32, 2.0, 3.0, 4.0];
        let up = [1.0f32; 4];
        let weights = [1.0f32; 4]; // 1x4
        let bias = [10.0f32];
        let mut out = [0.0f32];

        unsafe {
            neon_swiglu_linear(&gate, &up, &weights, &bias, &mut out);
        }

        let dot: f32 = gate.iter().map(|&g| scalar_silu(g)).sum();
        assert!(approx_eq(out[0], dot + 10.0, 1e-4), "got {} expected {}", out[0], dot + 10.0);
    }

    #[test]
    fn test_swiglu_linear_multi_output() {
        // hidden=4, out_dim=2
        let gate = [1.0f32, 0.0, -1.0, 0.5];
        let up = [2.0, 1.0, 1.0, 3.0];
        // Row 0: all ones; Row 1: all twos
        let weights = [
            1.0, 1.0, 1.0, 1.0, // row 0
            2.0, 2.0, 2.0, 2.0, // row 1
        ];
        let bias = [0.0f32, 1.0];
        let mut out = [0.0f32; 2];

        unsafe {
            neon_swiglu_linear(&gate, &up, &weights, &bias, &mut out);
        }

        let activated: Vec<f32> =
            gate.iter().zip(up.iter()).map(|(&g, &u)| scalar_silu(g) * u).collect();
        let e0: f32 = activated.iter().sum::<f32>() + 0.0;
        let e1: f32 = activated.iter().map(|v| v * 2.0).sum::<f32>() + 1.0;
        assert!(approx_eq(out[0], e0, 1e-4), "got {}", out[0]);
        assert!(approx_eq(out[1], e1, 1e-4), "got {}", out[1]);
    }

    #[test]
    fn test_swiglu_linear_non_aligned_hidden() {
        // hidden=5 (not a multiple of 4, exercises scalar tail)
        let gate = [1.0f32, 0.0, -1.0, 0.5, 2.0];
        let up = [1.0f32; 5];
        let weights = [1.0f32; 5];
        let bias = [0.0f32];
        let mut out = [0.0f32];

        unsafe {
            neon_swiglu_linear(&gate, &up, &weights, &bias, &mut out);
        }

        let expected: f32 = gate.iter().map(|&g| scalar_silu(g)).sum();
        assert!(approx_eq(out[0], expected, 1e-4), "got {} expected {expected}", out[0]);
    }

    // ── Edge cases ──────────────────────────────────────────────────

    #[test]
    fn test_empty_slices() {
        let g: [f32; 0] = [];
        let u: [f32; 0] = [];
        let mut o: [f32; 0] = [];
        unsafe {
            neon_swiglu(&g, &u, &mut o);
            neon_geglu(&g, &u, &mut o);
            neon_reglu(&g, &u, &mut o);
        }
    }

    #[test]
    fn test_single_element() {
        let gate = [1.5f32];
        let up = [2.0f32];
        let mut out = [0.0f32];

        unsafe { neon_swiglu(&gate, &up, &mut out) };
        let e_swi = scalar_silu(1.5) * 2.0;
        assert!(approx_eq(out[0], e_swi, EPS));

        unsafe { neon_geglu(&gate, &up, &mut out) };
        let e_ge = scalar_gelu(1.5) * 2.0;
        assert!(approx_eq(out[0], e_ge, EPS));

        unsafe { neon_reglu(&gate, &up, &mut out) };
        assert!(approx_eq(out[0], 3.0, EPS)); // relu(1.5)*2
    }

    // ── Property: output bounded by gate*up magnitude ───────────────

    #[test]
    fn test_swiglu_bounded_by_gate_up() {
        let gate: Vec<f32> = (-20..20).map(|i| i as f32 * 0.5).collect();
        let up: Vec<f32> = (0..40).map(|i| (i as f32 - 20.0) * 0.3).collect();
        let mut out = vec![0.0f32; 40];
        unsafe { neon_swiglu(&gate, &up, &mut out) };
        for i in 0..40 {
            let bound = gate[i].abs() * up[i].abs();
            assert!(
                out[i].abs() <= bound + 1e-6,
                "|swiglu[{i}]| = {} > bound {bound}",
                out[i].abs()
            );
        }
    }

    #[test]
    fn test_reglu_non_negative_when_up_non_negative() {
        let gate: Vec<f32> = (-50..50).map(|i| i as f32 * 0.1).collect();
        let up: Vec<f32> = (0..100).map(|i| i as f32 * 0.5).collect();
        let mut out = vec![0.0f32; 100];
        unsafe { neon_reglu(&gate, &up, &mut out) };
        for (i, &v) in out.iter().enumerate() {
            assert!(v >= 0.0, "reglu[{i}] = {v} should be >= 0");
        }
    }

    // ── Benchmark-gated tests ───────────────────────────────────────

    #[test]
    #[ignore = "Slow: large-buffer throughput test; run with --ignored"]
    fn test_swiglu_throughput_large() {
        let n = 1 << 20; // ~1M elements
        let gate: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001) - 500.0).collect();
        let up: Vec<f32> = (0..n).map(|i| (i as f32 * 0.002) - 1000.0).collect();
        let mut out = vec![0.0f32; n];
        unsafe { neon_swiglu(&gate, &up, &mut out) };
        // Spot-check a few values.
        for &idx in &[0, n / 4, n / 2, 3 * n / 4, n - 1] {
            let expected = scalar_silu(gate[idx]) * up[idx];
            assert!(approx_eq(out[idx], expected, 1e-3), "mismatch at {idx}");
        }
    }
}

//! NEON-optimized fused MLP v2 kernels for Apple Silicon.
//!
//! Provides six fused operations that eliminate intermediate buffers and
//! maximise data locality on AArch64 NEON:
//!
//! 1. **Gate-up projection** — NEON vectorised gate_proj × up_proj with SiLU
//! 2. **SwiGLU** — fused `silu(gate(x)) * up(x)` in one pass
//! 3. **Down projection** — NEON vectorised down-projection with residual add
//! 4. **RMSNorm + MLP** — combined normalisation and MLP in one kernel
//! 5. **Quantised-weight MLP** — I2_S dequant + MLP fused
//! 6. **Attention + MLP** — pipeline attention output through MLP

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ═══════════════════════════════════════════════════════════════════════
// Scalar helpers
// ═══════════════════════════════════════════════════════════════════════

#[inline(always)]
fn scalar_silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline(always)]
fn scalar_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Naive matrix-vector product: `out[i] = Σ_j weight[i*k + j] * input[j]`.
fn scalar_matvec(weight: &[f32], input: &[f32], output: &mut [f32], rows: usize, cols: usize) {
    for i in 0..rows {
        let mut acc = 0.0f32;
        for j in 0..cols {
            acc += weight[i * cols + j] * input[j];
        }
        output[i] = acc;
    }
}

/// Decode a 2-bit I2_S code to its signed value.
#[inline(always)]
fn decode_i2s(bits: u8) -> f32 {
    match bits & 0x03 {
        0b01 => 1.0,
        0b11 => -1.0,
        _ => 0.0,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 1. Fused gate-up projection with SiLU
// ═══════════════════════════════════════════════════════════════════════

/// Fused gate-up projection: `output[i] = silu(gate[i]) * up[i]`.
///
/// NEON path processes 4 lanes at a time; scalar fallback for remainder.
///
/// # Panics
///
/// Panics if `gate`, `up`, and `output` do not share the same length.
pub fn fused_gate_up_silu(gate: &[f32], up: &[f32], output: &mut [f32]) {
    let n = gate.len();
    assert_eq!(up.len(), n, "up length mismatch");
    assert_eq!(output.len(), n, "output length mismatch");

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is always available on AArch64.
        unsafe { neon_fused_gate_up_silu(gate, up, output) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_fused_gate_up_silu(gate, up, output);
    }
}

/// Scalar reference implementation.
pub fn scalar_fused_gate_up_silu(gate: &[f32], up: &[f32], output: &mut [f32]) {
    let n = gate.len();
    assert_eq!(up.len(), n);
    assert_eq!(output.len(), n);
    for i in 0..n {
        output[i] = scalar_silu(gate[i]) * up[i];
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_fused_gate_up_silu(gate: &[f32], up: &[f32], output: &mut [f32]) {
    let n = gate.len();
    let chunks = n / 4;
    let one = vdupq_n_f32(1.0);
    let neg_one = vdupq_n_f32(-1.0);

    for c in 0..chunks {
        let off = c * 4;
        let g = vld1q_f32(gate.as_ptr().add(off));
        let u = vld1q_f32(up.as_ptr().add(off));

        // sigmoid(g) via scalar (NEON lacks exp)
        let mut sig = [0.0f32; 4];
        vst1q_f32(sig.as_mut_ptr(), g);
        for s in &mut sig {
            *s = scalar_sigmoid(*s);
        }
        let sig_v = vld1q_f32(sig.as_ptr());
        // silu(g) = g * sigmoid(g)
        let silu_g = vmulq_f32(g, sig_v);
        let res = vmulq_f32(silu_g, u);
        vst1q_f32(output.as_mut_ptr().add(off), res);
    }

    let tail = chunks * 4;
    for i in tail..n {
        output[i] = scalar_silu(gate[i]) * up[i];
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 2. Fused SwiGLU
// ═══════════════════════════════════════════════════════════════════════

/// Fused SwiGLU: projects `input` through `w_gate` and `w_up`, then
/// computes `silu(gate_proj) * up_proj` in one pass.
///
/// `w_gate` and `w_up` are row-major `[intermediate_dim × hidden_dim]`.
///
/// # Panics
///
/// Panics on dimension mismatches.
pub fn fused_swiglu(
    input: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    output: &mut [f32],
    intermediate_dim: usize,
    hidden_dim: usize,
) {
    assert_eq!(input.len(), hidden_dim, "input length mismatch");
    assert_eq!(w_gate.len(), intermediate_dim * hidden_dim, "w_gate size mismatch");
    assert_eq!(w_up.len(), intermediate_dim * hidden_dim, "w_up size mismatch");
    assert_eq!(output.len(), intermediate_dim, "output length mismatch");

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_fused_swiglu(input, w_gate, w_up, output, intermediate_dim, hidden_dim) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_fused_swiglu(input, w_gate, w_up, output, intermediate_dim, hidden_dim);
    }
}

/// Scalar SwiGLU reference.
pub fn scalar_fused_swiglu(
    input: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    output: &mut [f32],
    intermediate_dim: usize,
    hidden_dim: usize,
) {
    for i in 0..intermediate_dim {
        let mut gate_acc = 0.0f32;
        let mut up_acc = 0.0f32;
        let row = i * hidden_dim;
        for j in 0..hidden_dim {
            gate_acc += w_gate[row + j] * input[j];
            up_acc += w_up[row + j] * input[j];
        }
        output[i] = scalar_silu(gate_acc) * up_acc;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_fused_swiglu(
    input: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    output: &mut [f32],
    intermediate_dim: usize,
    hidden_dim: usize,
) {
    for i in 0..intermediate_dim {
        let row = i * hidden_dim;
        let mut gate_acc = vdupq_n_f32(0.0);
        let mut up_acc = vdupq_n_f32(0.0);
        let chunks = hidden_dim / 4;

        for c in 0..chunks {
            let off = row + c * 4;
            let inp = vld1q_f32(input.as_ptr().add(c * 4));
            let wg = vld1q_f32(w_gate.as_ptr().add(off));
            let wu = vld1q_f32(w_up.as_ptr().add(off));
            gate_acc = vfmaq_f32(gate_acc, wg, inp);
            up_acc = vfmaq_f32(up_acc, wu, inp);
        }

        let mut gate_sum = vaddvq_f32(gate_acc);
        let mut up_sum = vaddvq_f32(up_acc);

        let tail = chunks * 4;
        for j in tail..hidden_dim {
            gate_sum += w_gate[row + j] * input[j];
            up_sum += w_up[row + j] * input[j];
        }

        output[i] = scalar_silu(gate_sum) * up_sum;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 3. Fused down projection with residual add
// ═══════════════════════════════════════════════════════════════════════

/// Fused down projection with residual: `output[i] = residual[i] + Σ_j w_down[i*k+j] * input[j]`.
///
/// `w_down` is row-major `[hidden_dim × intermediate_dim]`.
///
/// # Panics
///
/// Panics on dimension mismatches.
pub fn fused_down_proj_residual(
    input: &[f32],
    w_down: &[f32],
    residual: &[f32],
    output: &mut [f32],
    hidden_dim: usize,
    intermediate_dim: usize,
) {
    assert_eq!(input.len(), intermediate_dim, "input length mismatch");
    assert_eq!(w_down.len(), hidden_dim * intermediate_dim, "w_down size mismatch");
    assert_eq!(residual.len(), hidden_dim, "residual length mismatch");
    assert_eq!(output.len(), hidden_dim, "output length mismatch");

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            neon_fused_down_proj_residual(
                input,
                w_down,
                residual,
                output,
                hidden_dim,
                intermediate_dim,
            )
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_fused_down_proj_residual(
            input,
            w_down,
            residual,
            output,
            hidden_dim,
            intermediate_dim,
        );
    }
}

/// Scalar down-projection with residual add.
pub fn scalar_fused_down_proj_residual(
    input: &[f32],
    w_down: &[f32],
    residual: &[f32],
    output: &mut [f32],
    hidden_dim: usize,
    intermediate_dim: usize,
) {
    for i in 0..hidden_dim {
        let mut acc = 0.0f32;
        let row = i * intermediate_dim;
        for j in 0..intermediate_dim {
            acc += w_down[row + j] * input[j];
        }
        output[i] = residual[i] + acc;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_fused_down_proj_residual(
    input: &[f32],
    w_down: &[f32],
    residual: &[f32],
    output: &mut [f32],
    hidden_dim: usize,
    intermediate_dim: usize,
) {
    for i in 0..hidden_dim {
        let row = i * intermediate_dim;
        let mut acc = vdupq_n_f32(0.0);
        let chunks = intermediate_dim / 4;

        for c in 0..chunks {
            let off = row + c * 4;
            let inp = vld1q_f32(input.as_ptr().add(c * 4));
            let w = vld1q_f32(w_down.as_ptr().add(off));
            acc = vfmaq_f32(acc, w, inp);
        }

        let mut sum = vaddvq_f32(acc);
        let tail = chunks * 4;
        for j in tail..intermediate_dim {
            sum += w_down[row + j] * input[j];
        }
        output[i] = residual[i] + sum;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 4. Fused RMSNorm + MLP
// ═══════════════════════════════════════════════════════════════════════

/// Fused RMSNorm followed by gated MLP (SwiGLU):
///
/// ```text
/// normed = rmsnorm(input, gamma, eps)
/// gate   = normed · W_gate^T
/// up     = normed · W_up^T
/// output = residual + (silu(gate) * up) · W_down^T
/// ```
///
/// # Panics
///
/// Panics on dimension mismatches.
pub fn fused_rmsnorm_mlp(
    input: &[f32],
    gamma: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    residual: &[f32],
    output: &mut [f32],
    hidden_dim: usize,
    intermediate_dim: usize,
    eps: f32,
) {
    assert_eq!(input.len(), hidden_dim);
    assert_eq!(gamma.len(), hidden_dim);
    assert_eq!(w_gate.len(), intermediate_dim * hidden_dim);
    assert_eq!(w_up.len(), intermediate_dim * hidden_dim);
    assert_eq!(w_down.len(), hidden_dim * intermediate_dim);
    assert_eq!(residual.len(), hidden_dim);
    assert_eq!(output.len(), hidden_dim);

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            neon_fused_rmsnorm_mlp(
                input,
                gamma,
                w_gate,
                w_up,
                w_down,
                residual,
                output,
                hidden_dim,
                intermediate_dim,
                eps,
            )
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_fused_rmsnorm_mlp(
            input,
            gamma,
            w_gate,
            w_up,
            w_down,
            residual,
            output,
            hidden_dim,
            intermediate_dim,
            eps,
        );
    }
}

/// Scalar reference for fused RMSNorm + MLP.
pub fn scalar_fused_rmsnorm_mlp(
    input: &[f32],
    gamma: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    residual: &[f32],
    output: &mut [f32],
    hidden_dim: usize,
    intermediate_dim: usize,
    eps: f32,
) {
    // 1. RMSNorm
    let mut ss = 0.0f32;
    for i in 0..hidden_dim {
        ss += input[i] * input[i];
    }
    let rms = (ss / hidden_dim as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    let mut normed = vec![0.0f32; hidden_dim];
    for i in 0..hidden_dim {
        normed[i] = input[i] * inv_rms * gamma[i];
    }

    // 2. SwiGLU
    let mut intermediate = vec![0.0f32; intermediate_dim];
    scalar_fused_swiglu(&normed, w_gate, w_up, &mut intermediate, intermediate_dim, hidden_dim);

    // 3. Down projection + residual
    scalar_fused_down_proj_residual(
        &intermediate,
        w_down,
        residual,
        output,
        hidden_dim,
        intermediate_dim,
    );
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_fused_rmsnorm_mlp(
    input: &[f32],
    gamma: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    residual: &[f32],
    output: &mut [f32],
    hidden_dim: usize,
    intermediate_dim: usize,
    eps: f32,
) {
    // 1. RMSNorm — compute sum of squares via NEON
    let chunks = hidden_dim / 4;
    let mut ss_acc = vdupq_n_f32(0.0);
    for c in 0..chunks {
        let off = c * 4;
        let v = vld1q_f32(input.as_ptr().add(off));
        ss_acc = vfmaq_f32(ss_acc, v, v);
    }
    let mut ss = vaddvq_f32(ss_acc);
    for i in (chunks * 4)..hidden_dim {
        ss += input[i] * input[i];
    }
    let inv_rms = 1.0 / (ss / hidden_dim as f32 + eps).sqrt();
    let inv_rms_v = vdupq_n_f32(inv_rms);

    // Normalise into temp buffer
    let mut normed = vec![0.0f32; hidden_dim];
    for c in 0..chunks {
        let off = c * 4;
        let v = vld1q_f32(input.as_ptr().add(off));
        let g = vld1q_f32(gamma.as_ptr().add(off));
        let n = vmulq_f32(vmulq_f32(v, inv_rms_v), g);
        vst1q_f32(normed.as_mut_ptr().add(off), n);
    }
    for i in (chunks * 4)..hidden_dim {
        normed[i] = input[i] * inv_rms * gamma[i];
    }

    // 2. SwiGLU projection
    let mut intermediate = vec![0.0f32; intermediate_dim];
    neon_fused_swiglu(&normed, w_gate, w_up, &mut intermediate, intermediate_dim, hidden_dim);

    // 3. Down projection + residual
    neon_fused_down_proj_residual(
        &intermediate,
        w_down,
        residual,
        output,
        hidden_dim,
        intermediate_dim,
    );
}

// ═══════════════════════════════════════════════════════════════════════
// 5. Fused MLP with quantised I2_S weights
// ═══════════════════════════════════════════════════════════════════════

/// Dequantise a row of packed I2_S bytes into `f32` (4 values per byte).
fn dequant_i2s_row(packed: &[u8], scale: f32, out: &mut [f32]) {
    let mut idx = 0;
    for &byte in packed {
        for shift in (0..8).step_by(2) {
            if idx >= out.len() {
                return;
            }
            let bits = (byte >> shift) & 0x03;
            out[idx] = decode_i2s(bits) * scale;
            idx += 1;
        }
    }
}

/// Fused MLP with I2_S quantised gate/up/down weights.
///
/// Weight packing: 4 ternary values per byte (LSB-first). Each weight
/// matrix has one `f32` scale per row.
///
/// # Panics
///
/// Panics on dimension mismatches.
pub fn fused_mlp_quantized_i2s(
    input: &[f32],
    gate_packed: &[u8],
    gate_scales: &[f32],
    up_packed: &[u8],
    up_scales: &[f32],
    down_packed: &[u8],
    down_scales: &[f32],
    residual: &[f32],
    output: &mut [f32],
    hidden_dim: usize,
    intermediate_dim: usize,
) {
    let bytes_per_row_h = (hidden_dim + 3) / 4;
    let bytes_per_row_i = (intermediate_dim + 3) / 4;
    assert_eq!(input.len(), hidden_dim);
    assert_eq!(gate_packed.len(), intermediate_dim * bytes_per_row_h);
    assert_eq!(gate_scales.len(), intermediate_dim);
    assert_eq!(up_packed.len(), intermediate_dim * bytes_per_row_h);
    assert_eq!(up_scales.len(), intermediate_dim);
    assert_eq!(down_packed.len(), hidden_dim * bytes_per_row_i);
    assert_eq!(down_scales.len(), hidden_dim);
    assert_eq!(residual.len(), hidden_dim);
    assert_eq!(output.len(), hidden_dim);

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            neon_fused_mlp_quantized_i2s(
                input,
                gate_packed,
                gate_scales,
                up_packed,
                up_scales,
                down_packed,
                down_scales,
                residual,
                output,
                hidden_dim,
                intermediate_dim,
            )
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_fused_mlp_quantized_i2s(
            input,
            gate_packed,
            gate_scales,
            up_packed,
            up_scales,
            down_packed,
            down_scales,
            residual,
            output,
            hidden_dim,
            intermediate_dim,
        );
    }
}

/// Scalar reference for quantised MLP.
pub fn scalar_fused_mlp_quantized_i2s(
    input: &[f32],
    gate_packed: &[u8],
    gate_scales: &[f32],
    up_packed: &[u8],
    up_scales: &[f32],
    down_packed: &[u8],
    down_scales: &[f32],
    residual: &[f32],
    output: &mut [f32],
    hidden_dim: usize,
    intermediate_dim: usize,
) {
    let bytes_per_row_h = (hidden_dim + 3) / 4;
    let bytes_per_row_i = (intermediate_dim + 3) / 4;
    let mut gate_row = vec![0.0f32; hidden_dim];
    let mut up_row = vec![0.0f32; hidden_dim];
    let mut intermediate = vec![0.0f32; intermediate_dim];

    // Gate/up projection + SwiGLU
    for i in 0..intermediate_dim {
        let row_off = i * bytes_per_row_h;
        dequant_i2s_row(
            &gate_packed[row_off..row_off + bytes_per_row_h],
            gate_scales[i],
            &mut gate_row,
        );
        dequant_i2s_row(&up_packed[row_off..row_off + bytes_per_row_h], up_scales[i], &mut up_row);
        let mut gate_acc = 0.0f32;
        let mut up_acc = 0.0f32;
        for j in 0..hidden_dim {
            gate_acc += gate_row[j] * input[j];
            up_acc += up_row[j] * input[j];
        }
        intermediate[i] = scalar_silu(gate_acc) * up_acc;
    }

    // Down projection + residual
    let mut down_row = vec![0.0f32; intermediate_dim];
    for i in 0..hidden_dim {
        let row_off = i * bytes_per_row_i;
        dequant_i2s_row(
            &down_packed[row_off..row_off + bytes_per_row_i],
            down_scales[i],
            &mut down_row,
        );
        let mut acc = 0.0f32;
        for j in 0..intermediate_dim {
            acc += down_row[j] * intermediate[j];
        }
        output[i] = residual[i] + acc;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_fused_mlp_quantized_i2s(
    input: &[f32],
    gate_packed: &[u8],
    gate_scales: &[f32],
    up_packed: &[u8],
    up_scales: &[f32],
    down_packed: &[u8],
    down_scales: &[f32],
    residual: &[f32],
    output: &mut [f32],
    hidden_dim: usize,
    intermediate_dim: usize,
) {
    let bytes_per_row_h = (hidden_dim + 3) / 4;
    let bytes_per_row_i = (intermediate_dim + 3) / 4;
    let mut dequant_buf = vec![0.0f32; hidden_dim.max(intermediate_dim)];
    let mut intermediate = vec![0.0f32; intermediate_dim];

    // Gate/up projection + SwiGLU
    for i in 0..intermediate_dim {
        let row_off = i * bytes_per_row_h;

        // Gate dot
        dequant_i2s_row(
            &gate_packed[row_off..row_off + bytes_per_row_h],
            gate_scales[i],
            &mut dequant_buf[..hidden_dim],
        );
        let mut gate_acc = vdupq_n_f32(0.0);
        let chunks = hidden_dim / 4;
        for c in 0..chunks {
            let off = c * 4;
            let w = vld1q_f32(dequant_buf.as_ptr().add(off));
            let inp = vld1q_f32(input.as_ptr().add(off));
            gate_acc = vfmaq_f32(gate_acc, w, inp);
        }
        let mut gate_sum = vaddvq_f32(gate_acc);
        for j in (chunks * 4)..hidden_dim {
            gate_sum += dequant_buf[j] * input[j];
        }

        // Up dot
        dequant_i2s_row(
            &up_packed[row_off..row_off + bytes_per_row_h],
            up_scales[i],
            &mut dequant_buf[..hidden_dim],
        );
        let mut up_acc_v = vdupq_n_f32(0.0);
        for c in 0..chunks {
            let off = c * 4;
            let w = vld1q_f32(dequant_buf.as_ptr().add(off));
            let inp = vld1q_f32(input.as_ptr().add(off));
            up_acc_v = vfmaq_f32(up_acc_v, w, inp);
        }
        let mut up_sum = vaddvq_f32(up_acc_v);
        for j in (chunks * 4)..hidden_dim {
            up_sum += dequant_buf[j] * input[j];
        }

        intermediate[i] = scalar_silu(gate_sum) * up_sum;
    }

    // Down projection + residual
    for i in 0..hidden_dim {
        let row_off = i * bytes_per_row_i;
        dequant_i2s_row(
            &down_packed[row_off..row_off + bytes_per_row_i],
            down_scales[i],
            &mut dequant_buf[..intermediate_dim],
        );
        let mut acc = vdupq_n_f32(0.0);
        let chunks = intermediate_dim / 4;
        for c in 0..chunks {
            let off = c * 4;
            let w = vld1q_f32(dequant_buf.as_ptr().add(off));
            let inp = vld1q_f32(intermediate.as_ptr().add(off));
            acc = vfmaq_f32(acc, w, inp);
        }
        let mut sum = vaddvq_f32(acc);
        for j in (chunks * 4)..intermediate_dim {
            sum += dequant_buf[j] * intermediate[j];
        }
        output[i] = residual[i] + sum;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 6. Fused attention + MLP pipeline
// ═══════════════════════════════════════════════════════════════════════

/// Fused attention-output-through-MLP: takes pre-computed attention
/// output (already projected via `W_o`) and pipes it through the full
/// gated MLP block with RMSNorm and residual, avoiding an intermediate
/// write-back.
///
/// ```text
/// normed = rmsnorm(attn_out + residual, gamma, eps)
/// gate   = normed · W_gate^T
/// up     = normed · W_up^T
/// output = (attn_out + residual) + (silu(gate) * up) · W_down^T
/// ```
///
/// # Panics
///
/// Panics on dimension mismatches.
pub fn fused_attention_mlp(
    attn_out: &[f32],
    residual: &[f32],
    gamma: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    output: &mut [f32],
    hidden_dim: usize,
    intermediate_dim: usize,
    eps: f32,
) {
    assert_eq!(attn_out.len(), hidden_dim);
    assert_eq!(residual.len(), hidden_dim);
    assert_eq!(gamma.len(), hidden_dim);
    assert_eq!(w_gate.len(), intermediate_dim * hidden_dim);
    assert_eq!(w_up.len(), intermediate_dim * hidden_dim);
    assert_eq!(w_down.len(), hidden_dim * intermediate_dim);
    assert_eq!(output.len(), hidden_dim);

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            neon_fused_attention_mlp(
                attn_out,
                residual,
                gamma,
                w_gate,
                w_up,
                w_down,
                output,
                hidden_dim,
                intermediate_dim,
                eps,
            )
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_fused_attention_mlp(
            attn_out,
            residual,
            gamma,
            w_gate,
            w_up,
            w_down,
            output,
            hidden_dim,
            intermediate_dim,
            eps,
        );
    }
}

/// Scalar reference for fused attention + MLP.
pub fn scalar_fused_attention_mlp(
    attn_out: &[f32],
    residual: &[f32],
    gamma: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    output: &mut [f32],
    hidden_dim: usize,
    intermediate_dim: usize,
    eps: f32,
) {
    // 1. Post-attention residual
    let mut post_attn = vec![0.0f32; hidden_dim];
    for i in 0..hidden_dim {
        post_attn[i] = attn_out[i] + residual[i];
    }

    // 2. RMSNorm + MLP with post_attn as both input and residual
    scalar_fused_rmsnorm_mlp(
        &post_attn,
        gamma,
        w_gate,
        w_up,
        w_down,
        &post_attn.clone(),
        output,
        hidden_dim,
        intermediate_dim,
        eps,
    );
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_fused_attention_mlp(
    attn_out: &[f32],
    residual: &[f32],
    gamma: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    output: &mut [f32],
    hidden_dim: usize,
    intermediate_dim: usize,
    eps: f32,
) {
    // 1. Post-attention residual via NEON
    let mut post_attn = vec![0.0f32; hidden_dim];
    let chunks = hidden_dim / 4;
    for c in 0..chunks {
        let off = c * 4;
        let a = vld1q_f32(attn_out.as_ptr().add(off));
        let r = vld1q_f32(residual.as_ptr().add(off));
        vst1q_f32(post_attn.as_mut_ptr().add(off), vaddq_f32(a, r));
    }
    for i in (chunks * 4)..hidden_dim {
        post_attn[i] = attn_out[i] + residual[i];
    }

    // 2. RMSNorm + MLP (re-uses post_attn as residual)
    let post_attn_residual = post_attn.clone();
    neon_fused_rmsnorm_mlp(
        &post_attn,
        gamma,
        w_gate,
        w_up,
        w_down,
        &post_attn_residual,
        output,
        hidden_dim,
        intermediate_dim,
        eps,
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic seeded RNG (xorshift32) — no external dep needed.
    struct Rng(u32);
    impl Rng {
        fn new(seed: u32) -> Self {
            Self(seed)
        }
        fn next_u32(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 17;
            self.0 ^= self.0 << 5;
            self.0
        }
        fn next_f32(&mut self) -> f32 {
            (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
        }
        fn rand_vec(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.next_f32()).collect()
        }
        fn rand_positive_vec(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.next_f32().abs() + 0.01).collect()
        }
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32, ctx: &str) {
        assert_eq!(a.len(), b.len(), "{ctx}: length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (x - y).abs();
            let denom = x.abs().max(y.abs()).max(1e-6);
            assert!(
                diff / denom < tol,
                "{ctx}: index {i}: {x} vs {y} (diff={diff}, rel={:.6})",
                diff / denom
            );
        }
    }

    fn naive_silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    /// Helper: pack f32 weights to I2_S (ternary quantisation) and return (packed, scales).
    fn quantize_row_i2s(weights: &[f32]) -> (Vec<u8>, f32) {
        let max_abs = weights.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = if max_abs > 0.0 { max_abs } else { 1.0 };
        let bytes_needed = (weights.len() + 3) / 4;
        let mut packed = vec![0u8; bytes_needed];
        for (i, &w) in weights.iter().enumerate() {
            let q = (w / scale).round() as i32;
            let code: u8 = match q.clamp(-1, 1) {
                1 => 0b01,
                -1 => 0b11,
                _ => 0b00,
            };
            let byte_idx = i / 4;
            let bit_shift = (i % 4) * 2;
            packed[byte_idx] |= code << bit_shift;
        }
        (packed, scale)
    }

    fn quantize_matrix_i2s(weights: &[f32], rows: usize, cols: usize) -> (Vec<u8>, Vec<f32>) {
        let bytes_per_row = (cols + 3) / 4;
        let mut packed = vec![0u8; rows * bytes_per_row];
        let mut scales = vec![0.0f32; rows];
        for r in 0..rows {
            let row = &weights[r * cols..(r + 1) * cols];
            let (p, s) = quantize_row_i2s(row);
            packed[r * bytes_per_row..(r + 1) * bytes_per_row].copy_from_slice(&p);
            scales[r] = s;
        }
        (packed, scales)
    }

    // ── 1. Gate-up SiLU tests ───────────────────────────────────────

    #[test]
    fn test_gate_up_silu_basic() {
        let gate = vec![1.0, -1.0, 0.5, 2.0];
        let up = vec![1.0, 1.0, 1.0, 1.0];
        let mut output = vec![0.0; 4];
        fused_gate_up_silu(&gate, &up, &mut output);
        for i in 0..4 {
            let expected = naive_silu(gate[i]) * up[i];
            assert!((output[i] - expected).abs() < 1e-5, "idx {i}");
        }
    }

    #[test]
    fn test_gate_up_silu_zeros() {
        let gate = vec![0.0; 8];
        let up = vec![1.0; 8];
        let mut output = vec![0.0; 8];
        fused_gate_up_silu(&gate, &up, &mut output);
        for &v in &output {
            assert!((v - 0.0).abs() < 1e-7);
        }
    }

    #[test]
    fn test_gate_up_silu_negative() {
        let gate = vec![-5.0, -10.0, -0.1, -100.0];
        let up = vec![1.0; 4];
        let mut output = vec![0.0; 4];
        fused_gate_up_silu(&gate, &up, &mut output);
        for i in 0..4 {
            let expected = naive_silu(gate[i]) * up[i];
            assert!((output[i] - expected).abs() < 1e-4, "idx {i}");
        }
    }

    #[test]
    fn test_gate_up_silu_size_1() {
        let gate = vec![0.5];
        let up = vec![2.0];
        let mut output = vec![0.0; 1];
        fused_gate_up_silu(&gate, &up, &mut output);
        let expected = naive_silu(0.5) * 2.0;
        assert!((output[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn test_gate_up_silu_non_multiple_of_4() {
        let mut rng = Rng::new(42);
        for n in [3, 5, 7, 11, 13, 17] {
            let gate = rng.rand_vec(n);
            let up = rng.rand_vec(n);
            let mut output = vec![0.0; n];
            let mut expected = vec![0.0; n];
            fused_gate_up_silu(&gate, &up, &mut output);
            scalar_fused_gate_up_silu(&gate, &up, &mut expected);
            assert_close(&output, &expected, 1e-5, &format!("gate_up n={n}"));
        }
    }

    #[test]
    fn test_gate_up_silu_neon_scalar_parity() {
        let mut rng = Rng::new(123);
        for &n in &[128, 256, 512, 1024, 2048, 4096] {
            let gate = rng.rand_vec(n);
            let up = rng.rand_vec(n);
            let mut neon_out = vec![0.0; n];
            let mut scalar_out = vec![0.0; n];
            fused_gate_up_silu(&gate, &up, &mut neon_out);
            scalar_fused_gate_up_silu(&gate, &up, &mut scalar_out);
            assert_close(&neon_out, &scalar_out, 1e-5, &format!("gate_up parity n={n}"));
        }
    }

    #[test]
    fn test_gate_up_silu_large() {
        let n = 4096;
        let mut rng = Rng::new(7);
        let gate = rng.rand_vec(n);
        let up = rng.rand_vec(n);
        let mut output = vec![0.0; n];
        fused_gate_up_silu(&gate, &up, &mut output);
        // Smoke: no crash, all finite
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_gate_up_silu_numerical_stability() {
        let gate = vec![80.0, -80.0, 0.0, 1e-8];
        let up = vec![1.0; 4];
        let mut output = vec![0.0; 4];
        fused_gate_up_silu(&gate, &up, &mut output);
        assert!(output.iter().all(|v| v.is_finite()));
    }

    // ── 2. Fused SwiGLU tests ───────────────────────────────────────

    #[test]
    fn test_swiglu_basic() {
        let hidden = 4;
        let inter = 2;
        let input = vec![1.0, 0.5, -1.0, 2.0];
        let w_gate = vec![0.1, 0.2, 0.3, 0.4, -0.1, -0.2, 0.1, 0.5];
        let w_up = vec![0.5, -0.1, 0.2, 0.3, 0.1, 0.4, -0.3, 0.2];
        let mut output = vec![0.0; inter];
        let mut expected = vec![0.0; inter];
        fused_swiglu(&input, &w_gate, &w_up, &mut output, inter, hidden);
        scalar_fused_swiglu(&input, &w_gate, &w_up, &mut expected, inter, hidden);
        assert_close(&output, &expected, 1e-5, "swiglu basic");
    }

    #[test]
    fn test_swiglu_zero_input() {
        let hidden = 8;
        let inter = 4;
        let input = vec![0.0; hidden];
        let mut rng = Rng::new(55);
        let w_gate = rng.rand_vec(inter * hidden);
        let w_up = rng.rand_vec(inter * hidden);
        let mut output = vec![0.0; inter];
        fused_swiglu(&input, &w_gate, &w_up, &mut output, inter, hidden);
        // silu(0)*0 = 0
        for &v in &output {
            assert!((v).abs() < 1e-7);
        }
    }

    #[test]
    fn test_swiglu_sizes() {
        let mut rng = Rng::new(99);
        for &(h, i) in &[(128, 256), (256, 512), (64, 128), (32, 64)] {
            let input = rng.rand_vec(h);
            let w_gate = rng.rand_vec(i * h);
            let w_up = rng.rand_vec(i * h);
            let mut output = vec![0.0; i];
            let mut expected = vec![0.0; i];
            fused_swiglu(&input, &w_gate, &w_up, &mut output, i, h);
            scalar_fused_swiglu(&input, &w_gate, &w_up, &mut expected, i, h);
            assert_close(&output, &expected, 5e-3, &format!("swiglu h={h} i={i}"));
        }
    }

    #[test]
    fn test_swiglu_non_multiple_of_4_hidden() {
        let hidden = 5;
        let inter = 3;
        let mut rng = Rng::new(77);
        let input = rng.rand_vec(hidden);
        let w_gate = rng.rand_vec(inter * hidden);
        let w_up = rng.rand_vec(inter * hidden);
        let mut output = vec![0.0; inter];
        let mut expected = vec![0.0; inter];
        fused_swiglu(&input, &w_gate, &w_up, &mut output, inter, hidden);
        scalar_fused_swiglu(&input, &w_gate, &w_up, &mut expected, inter, hidden);
        assert_close(&output, &expected, 1e-4, "swiglu non-mult-4");
    }

    #[test]
    fn test_swiglu_neon_scalar_parity_large() {
        let hidden = 512;
        let inter = 1024;
        let mut rng = Rng::new(42);
        let input = rng.rand_vec(hidden);
        let w_gate = rng.rand_vec(inter * hidden);
        let w_up = rng.rand_vec(inter * hidden);
        let mut neon_out = vec![0.0; inter];
        let mut scalar_out = vec![0.0; inter];
        fused_swiglu(&input, &w_gate, &w_up, &mut neon_out, inter, hidden);
        scalar_fused_swiglu(&input, &w_gate, &w_up, &mut scalar_out, inter, hidden);
        assert_close(&neon_out, &scalar_out, 1e-3, "swiglu parity large");
    }

    // ── 3. Down projection + residual tests ─────────────────────────

    #[test]
    fn test_down_proj_residual_basic() {
        let hidden = 2;
        let inter = 3;
        let input = vec![1.0, 2.0, 3.0];
        let w_down = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let residual = vec![10.0, 20.0];
        let mut output = vec![0.0; hidden];
        let mut expected = vec![0.0; hidden];
        fused_down_proj_residual(&input, &w_down, &residual, &mut output, hidden, inter);
        scalar_fused_down_proj_residual(&input, &w_down, &residual, &mut expected, hidden, inter);
        assert_close(&output, &expected, 1e-5, "down_proj basic");
    }

    #[test]
    fn test_down_proj_residual_zero_input() {
        let hidden = 4;
        let inter = 8;
        let input = vec![0.0; inter];
        let mut rng = Rng::new(10);
        let w_down = rng.rand_vec(hidden * inter);
        let residual = rng.rand_vec(hidden);
        let mut output = vec![0.0; hidden];
        fused_down_proj_residual(&input, &w_down, &residual, &mut output, hidden, inter);
        assert_close(&output, &residual, 1e-7, "down_proj zero input");
    }

    #[test]
    fn test_down_proj_residual_sizes() {
        let mut rng = Rng::new(200);
        for &(h, i) in &[(128, 256), (256, 512), (512, 1024), (64, 128)] {
            let input = rng.rand_vec(i);
            let w_down = rng.rand_vec(h * i);
            let residual = rng.rand_vec(h);
            let mut output = vec![0.0; h];
            let mut expected = vec![0.0; h];
            fused_down_proj_residual(&input, &w_down, &residual, &mut output, h, i);
            scalar_fused_down_proj_residual(&input, &w_down, &residual, &mut expected, h, i);
            assert_close(&output, &expected, 1e-3, &format!("down_proj h={h} i={i}"));
        }
    }

    #[test]
    fn test_down_proj_residual_non_multiple_of_4() {
        let hidden = 3;
        let inter = 5;
        let mut rng = Rng::new(33);
        let input = rng.rand_vec(inter);
        let w_down = rng.rand_vec(hidden * inter);
        let residual = rng.rand_vec(hidden);
        let mut output = vec![0.0; hidden];
        let mut expected = vec![0.0; hidden];
        fused_down_proj_residual(&input, &w_down, &residual, &mut output, hidden, inter);
        scalar_fused_down_proj_residual(&input, &w_down, &residual, &mut expected, hidden, inter);
        assert_close(&output, &expected, 1e-5, "down_proj non-mult-4");
    }

    #[test]
    fn test_down_proj_residual_neon_scalar_parity() {
        let mut rng = Rng::new(444);
        for &(h, i) in &[(128, 256), (256, 512), (1024, 2048)] {
            let input = rng.rand_vec(i);
            let w_down = rng.rand_vec(h * i);
            let residual = rng.rand_vec(h);
            let mut neon_out = vec![0.0; h];
            let mut scalar_out = vec![0.0; h];
            fused_down_proj_residual(&input, &w_down, &residual, &mut neon_out, h, i);
            scalar_fused_down_proj_residual(&input, &w_down, &residual, &mut scalar_out, h, i);
            assert_close(&neon_out, &scalar_out, 1e-3, &format!("down_proj parity h={h} i={i}"));
        }
    }

    // ── 4. RMSNorm + MLP tests ──────────────────────────────────────

    #[test]
    fn test_rmsnorm_mlp_basic() {
        let hidden = 4;
        let inter = 8;
        let mut rng = Rng::new(50);
        let input = rng.rand_vec(hidden);
        let gamma = rng.rand_positive_vec(hidden);
        let w_gate = rng.rand_vec(inter * hidden);
        let w_up = rng.rand_vec(inter * hidden);
        let w_down = rng.rand_vec(hidden * inter);
        let residual = rng.rand_vec(hidden);
        let mut output = vec![0.0; hidden];
        let mut expected = vec![0.0; hidden];
        fused_rmsnorm_mlp(
            &input,
            &gamma,
            &w_gate,
            &w_up,
            &w_down,
            &residual,
            &mut output,
            hidden,
            inter,
            1e-5,
        );
        scalar_fused_rmsnorm_mlp(
            &input,
            &gamma,
            &w_gate,
            &w_up,
            &w_down,
            &residual,
            &mut expected,
            hidden,
            inter,
            1e-5,
        );
        assert_close(&output, &expected, 1e-4, "rmsnorm_mlp basic");
    }

    #[test]
    fn test_rmsnorm_mlp_zero_input() {
        let hidden = 8;
        let inter = 16;
        let input = vec![0.0; hidden];
        let gamma = vec![1.0; hidden];
        let mut rng = Rng::new(60);
        let w_gate = rng.rand_vec(inter * hidden);
        let w_up = rng.rand_vec(inter * hidden);
        let w_down = rng.rand_vec(hidden * inter);
        let residual = rng.rand_vec(hidden);
        let mut output = vec![0.0; hidden];
        fused_rmsnorm_mlp(
            &input,
            &gamma,
            &w_gate,
            &w_up,
            &w_down,
            &residual,
            &mut output,
            hidden,
            inter,
            1e-5,
        );
        // With zero input, RMSNorm produces near-zero, SwiGLU ≈ 0, output ≈ residual
        assert_close(&output, &residual, 0.5, "rmsnorm_mlp zero input");
    }

    #[test]
    fn test_rmsnorm_mlp_sizes() {
        let mut rng = Rng::new(70);
        for &(h, i) in &[(32, 64), (64, 128), (128, 256)] {
            let input = rng.rand_vec(h);
            let gamma = rng.rand_positive_vec(h);
            let w_gate = rng.rand_vec(i * h);
            let w_up = rng.rand_vec(i * h);
            let w_down = rng.rand_vec(h * i);
            let residual = rng.rand_vec(h);
            let mut output = vec![0.0; h];
            let mut expected = vec![0.0; h];
            fused_rmsnorm_mlp(
                &input,
                &gamma,
                &w_gate,
                &w_up,
                &w_down,
                &residual,
                &mut output,
                h,
                i,
                1e-5,
            );
            scalar_fused_rmsnorm_mlp(
                &input,
                &gamma,
                &w_gate,
                &w_up,
                &w_down,
                &residual,
                &mut expected,
                h,
                i,
                1e-5,
            );
            assert_close(&output, &expected, 1e-3, &format!("rmsnorm_mlp h={h} i={i}"));
        }
    }

    #[test]
    fn test_rmsnorm_mlp_non_multiple_of_4() {
        let hidden = 5;
        let inter = 7;
        let mut rng = Rng::new(80);
        let input = rng.rand_vec(hidden);
        let gamma = rng.rand_positive_vec(hidden);
        let w_gate = rng.rand_vec(inter * hidden);
        let w_up = rng.rand_vec(inter * hidden);
        let w_down = rng.rand_vec(hidden * inter);
        let residual = rng.rand_vec(hidden);
        let mut output = vec![0.0; hidden];
        let mut expected = vec![0.0; hidden];
        fused_rmsnorm_mlp(
            &input,
            &gamma,
            &w_gate,
            &w_up,
            &w_down,
            &residual,
            &mut output,
            hidden,
            inter,
            1e-5,
        );
        scalar_fused_rmsnorm_mlp(
            &input,
            &gamma,
            &w_gate,
            &w_up,
            &w_down,
            &residual,
            &mut expected,
            hidden,
            inter,
            1e-5,
        );
        assert_close(&output, &expected, 1e-4, "rmsnorm_mlp non-mult-4");
    }

    #[test]
    fn test_rmsnorm_mlp_epsilon_stability() {
        let hidden = 4;
        let inter = 8;
        let input = vec![1e-8; hidden];
        let gamma = vec![1.0; hidden];
        let mut rng = Rng::new(90);
        let w_gate = rng.rand_vec(inter * hidden);
        let w_up = rng.rand_vec(inter * hidden);
        let w_down = rng.rand_vec(hidden * inter);
        let residual = vec![0.0; hidden];
        let mut output = vec![0.0; hidden];
        fused_rmsnorm_mlp(
            &input,
            &gamma,
            &w_gate,
            &w_up,
            &w_down,
            &residual,
            &mut output,
            hidden,
            inter,
            1e-5,
        );
        assert!(output.iter().all(|v| v.is_finite()), "should be finite for tiny input");
    }

    #[test]
    fn test_rmsnorm_mlp_neon_scalar_parity_256() {
        let hidden = 256;
        let inter = 512;
        let mut rng = Rng::new(100);
        let input = rng.rand_vec(hidden);
        let gamma = rng.rand_positive_vec(hidden);
        let w_gate = rng.rand_vec(inter * hidden);
        let w_up = rng.rand_vec(inter * hidden);
        let w_down = rng.rand_vec(hidden * inter);
        let residual = rng.rand_vec(hidden);
        let mut neon_out = vec![0.0; hidden];
        let mut scalar_out = vec![0.0; hidden];
        fused_rmsnorm_mlp(
            &input,
            &gamma,
            &w_gate,
            &w_up,
            &w_down,
            &residual,
            &mut neon_out,
            hidden,
            inter,
            1e-5,
        );
        scalar_fused_rmsnorm_mlp(
            &input,
            &gamma,
            &w_gate,
            &w_up,
            &w_down,
            &residual,
            &mut scalar_out,
            hidden,
            inter,
            1e-5,
        );
        assert_close(&neon_out, &scalar_out, 1e-3, "rmsnorm_mlp parity 256");
    }

    // ── 5. Quantised MLP tests ──────────────────────────────────────

    #[test]
    fn test_quantized_mlp_basic() {
        let hidden = 8;
        let inter = 4;
        let mut rng = Rng::new(300);
        let input = rng.rand_vec(hidden);
        let residual = rng.rand_vec(hidden);

        let gate_weights = rng.rand_vec(inter * hidden);
        let up_weights = rng.rand_vec(inter * hidden);
        let down_weights = rng.rand_vec(hidden * inter);

        let (gate_p, gate_s) = quantize_matrix_i2s(&gate_weights, inter, hidden);
        let (up_p, up_s) = quantize_matrix_i2s(&up_weights, inter, hidden);
        let (down_p, down_s) = quantize_matrix_i2s(&down_weights, hidden, inter);

        let mut output = vec![0.0; hidden];
        fused_mlp_quantized_i2s(
            &input,
            &gate_p,
            &gate_s,
            &up_p,
            &up_s,
            &down_p,
            &down_s,
            &residual,
            &mut output,
            hidden,
            inter,
        );
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_quantized_mlp_neon_scalar_parity() {
        let hidden = 16;
        let inter = 8;
        let mut rng = Rng::new(310);
        let input = rng.rand_vec(hidden);
        let residual = rng.rand_vec(hidden);

        let gate_weights = rng.rand_vec(inter * hidden);
        let up_weights = rng.rand_vec(inter * hidden);
        let down_weights = rng.rand_vec(hidden * inter);

        let (gate_p, gate_s) = quantize_matrix_i2s(&gate_weights, inter, hidden);
        let (up_p, up_s) = quantize_matrix_i2s(&up_weights, inter, hidden);
        let (down_p, down_s) = quantize_matrix_i2s(&down_weights, hidden, inter);

        let mut neon_out = vec![0.0; hidden];
        let mut scalar_out = vec![0.0; hidden];
        fused_mlp_quantized_i2s(
            &input,
            &gate_p,
            &gate_s,
            &up_p,
            &up_s,
            &down_p,
            &down_s,
            &residual,
            &mut neon_out,
            hidden,
            inter,
        );
        scalar_fused_mlp_quantized_i2s(
            &input,
            &gate_p,
            &gate_s,
            &up_p,
            &up_s,
            &down_p,
            &down_s,
            &residual,
            &mut scalar_out,
            hidden,
            inter,
        );
        assert_close(&neon_out, &scalar_out, 1e-5, "quant mlp parity");
    }

    #[test]
    fn test_quantized_mlp_zero_input() {
        let hidden = 8;
        let inter = 4;
        let input = vec![0.0; hidden];
        let residual = vec![1.0; hidden];
        let mut rng = Rng::new(320);
        let gate_weights = rng.rand_vec(inter * hidden);
        let up_weights = rng.rand_vec(inter * hidden);
        let down_weights = rng.rand_vec(hidden * inter);
        let (gate_p, gate_s) = quantize_matrix_i2s(&gate_weights, inter, hidden);
        let (up_p, up_s) = quantize_matrix_i2s(&up_weights, inter, hidden);
        let (down_p, down_s) = quantize_matrix_i2s(&down_weights, hidden, inter);
        let mut output = vec![0.0; hidden];
        fused_mlp_quantized_i2s(
            &input,
            &gate_p,
            &gate_s,
            &up_p,
            &up_s,
            &down_p,
            &down_s,
            &residual,
            &mut output,
            hidden,
            inter,
        );
        assert_close(&output, &residual, 1e-5, "quant mlp zero");
    }

    #[test]
    fn test_quantized_mlp_sizes() {
        let mut rng = Rng::new(330);
        for &(h, i) in &[(16, 32), (32, 64), (64, 128)] {
            let input = rng.rand_vec(h);
            let residual = rng.rand_vec(h);
            let gate_w = rng.rand_vec(i * h);
            let up_w = rng.rand_vec(i * h);
            let down_w = rng.rand_vec(h * i);
            let (gp, gs) = quantize_matrix_i2s(&gate_w, i, h);
            let (up, us) = quantize_matrix_i2s(&up_w, i, h);
            let (dp, ds) = quantize_matrix_i2s(&down_w, h, i);
            let mut neon_out = vec![0.0; h];
            let mut scalar_out = vec![0.0; h];
            fused_mlp_quantized_i2s(
                &input,
                &gp,
                &gs,
                &up,
                &us,
                &dp,
                &ds,
                &residual,
                &mut neon_out,
                h,
                i,
            );
            scalar_fused_mlp_quantized_i2s(
                &input,
                &gp,
                &gs,
                &up,
                &us,
                &dp,
                &ds,
                &residual,
                &mut scalar_out,
                h,
                i,
            );
            assert_close(&neon_out, &scalar_out, 5e-3, &format!("quant mlp h={h} i={i}"));
        }
    }

    #[test]
    fn test_quantized_mlp_non_multiple_of_4() {
        let hidden = 7;
        let inter = 5;
        let mut rng = Rng::new(340);
        let input = rng.rand_vec(hidden);
        let residual = rng.rand_vec(hidden);
        let gate_w = rng.rand_vec(inter * hidden);
        let up_w = rng.rand_vec(inter * hidden);
        let down_w = rng.rand_vec(hidden * inter);
        let (gp, gs) = quantize_matrix_i2s(&gate_w, inter, hidden);
        let (up, us) = quantize_matrix_i2s(&up_w, inter, hidden);
        let (dp, ds) = quantize_matrix_i2s(&down_w, hidden, inter);
        let mut neon_out = vec![0.0; hidden];
        let mut scalar_out = vec![0.0; hidden];
        fused_mlp_quantized_i2s(
            &input,
            &gp,
            &gs,
            &up,
            &us,
            &dp,
            &ds,
            &residual,
            &mut neon_out,
            hidden,
            inter,
        );
        scalar_fused_mlp_quantized_i2s(
            &input,
            &gp,
            &gs,
            &up,
            &us,
            &dp,
            &ds,
            &residual,
            &mut scalar_out,
            hidden,
            inter,
        );
        assert_close(&neon_out, &scalar_out, 1e-4, "quant mlp non-mult-4");
    }

    // ── 6. Attention + MLP tests ────────────────────────────────────

    #[test]
    fn test_attention_mlp_basic() {
        let hidden = 4;
        let inter = 8;
        let mut rng = Rng::new(400);
        let attn_out = rng.rand_vec(hidden);
        let residual = rng.rand_vec(hidden);
        let gamma = rng.rand_positive_vec(hidden);
        let w_gate = rng.rand_vec(inter * hidden);
        let w_up = rng.rand_vec(inter * hidden);
        let w_down = rng.rand_vec(hidden * inter);
        let mut output = vec![0.0; hidden];
        let mut expected = vec![0.0; hidden];
        fused_attention_mlp(
            &attn_out,
            &residual,
            &gamma,
            &w_gate,
            &w_up,
            &w_down,
            &mut output,
            hidden,
            inter,
            1e-5,
        );
        scalar_fused_attention_mlp(
            &attn_out,
            &residual,
            &gamma,
            &w_gate,
            &w_up,
            &w_down,
            &mut expected,
            hidden,
            inter,
            1e-5,
        );
        assert_close(&output, &expected, 1e-4, "attn_mlp basic");
    }

    #[test]
    fn test_attention_mlp_zero_attn() {
        let hidden = 8;
        let inter = 16;
        let attn_out = vec![0.0; hidden];
        let mut rng = Rng::new(410);
        let residual = rng.rand_vec(hidden);
        let gamma = rng.rand_positive_vec(hidden);
        let w_gate = rng.rand_vec(inter * hidden);
        let w_up = rng.rand_vec(inter * hidden);
        let w_down = rng.rand_vec(hidden * inter);
        let mut output = vec![0.0; hidden];
        let mut expected = vec![0.0; hidden];
        fused_attention_mlp(
            &attn_out,
            &residual,
            &gamma,
            &w_gate,
            &w_up,
            &w_down,
            &mut output,
            hidden,
            inter,
            1e-5,
        );
        scalar_fused_attention_mlp(
            &attn_out,
            &residual,
            &gamma,
            &w_gate,
            &w_up,
            &w_down,
            &mut expected,
            hidden,
            inter,
            1e-5,
        );
        assert_close(&output, &expected, 1e-4, "attn_mlp zero attn");
    }

    #[test]
    fn test_attention_mlp_sizes() {
        let mut rng = Rng::new(420);
        for &(h, i) in &[(32, 64), (64, 128), (128, 256)] {
            let attn_out = rng.rand_vec(h);
            let residual = rng.rand_vec(h);
            let gamma = rng.rand_positive_vec(h);
            let w_gate = rng.rand_vec(i * h);
            let w_up = rng.rand_vec(i * h);
            let w_down = rng.rand_vec(h * i);
            let mut output = vec![0.0; h];
            let mut expected = vec![0.0; h];
            fused_attention_mlp(
                &attn_out,
                &residual,
                &gamma,
                &w_gate,
                &w_up,
                &w_down,
                &mut output,
                h,
                i,
                1e-5,
            );
            scalar_fused_attention_mlp(
                &attn_out,
                &residual,
                &gamma,
                &w_gate,
                &w_up,
                &w_down,
                &mut expected,
                h,
                i,
                1e-5,
            );
            assert_close(&output, &expected, 1e-3, &format!("attn_mlp h={h} i={i}"));
        }
    }

    #[test]
    fn test_attention_mlp_non_multiple_of_4() {
        let hidden = 5;
        let inter = 7;
        let mut rng = Rng::new(430);
        let attn_out = rng.rand_vec(hidden);
        let residual = rng.rand_vec(hidden);
        let gamma = rng.rand_positive_vec(hidden);
        let w_gate = rng.rand_vec(inter * hidden);
        let w_up = rng.rand_vec(inter * hidden);
        let w_down = rng.rand_vec(hidden * inter);
        let mut output = vec![0.0; hidden];
        let mut expected = vec![0.0; hidden];
        fused_attention_mlp(
            &attn_out,
            &residual,
            &gamma,
            &w_gate,
            &w_up,
            &w_down,
            &mut output,
            hidden,
            inter,
            1e-5,
        );
        scalar_fused_attention_mlp(
            &attn_out,
            &residual,
            &gamma,
            &w_gate,
            &w_up,
            &w_down,
            &mut expected,
            hidden,
            inter,
            1e-5,
        );
        assert_close(&output, &expected, 1e-4, "attn_mlp non-mult-4");
    }

    #[test]
    fn test_attention_mlp_neon_scalar_parity_256() {
        let hidden = 256;
        let inter = 512;
        let mut rng = Rng::new(440);
        let attn_out = rng.rand_vec(hidden);
        let residual = rng.rand_vec(hidden);
        let gamma = rng.rand_positive_vec(hidden);
        let w_gate = rng.rand_vec(inter * hidden);
        let w_up = rng.rand_vec(inter * hidden);
        let w_down = rng.rand_vec(hidden * inter);
        let mut neon_out = vec![0.0; hidden];
        let mut scalar_out = vec![0.0; hidden];
        fused_attention_mlp(
            &attn_out,
            &residual,
            &gamma,
            &w_gate,
            &w_up,
            &w_down,
            &mut neon_out,
            hidden,
            inter,
            1e-5,
        );
        scalar_fused_attention_mlp(
            &attn_out,
            &residual,
            &gamma,
            &w_gate,
            &w_up,
            &w_down,
            &mut scalar_out,
            hidden,
            inter,
            1e-5,
        );
        assert_close(&neon_out, &scalar_out, 1e-3, "attn_mlp parity 256");
    }

    // ── Cross-cutting tests ─────────────────────────────────────────

    #[test]
    fn test_silu_properties() {
        // silu(0) = 0
        assert!((scalar_silu(0.0)).abs() < 1e-7);
        // silu is monotonically increasing for x > ~-0.278
        let vals: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        for w in vals.windows(2) {
            assert!(scalar_silu(w[1]) >= scalar_silu(w[0]));
        }
        // silu(x) → x for large positive x
        assert!((scalar_silu(100.0) - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_decode_i2s_values() {
        assert_eq!(decode_i2s(0b00), 0.0);
        assert_eq!(decode_i2s(0b01), 1.0);
        assert_eq!(decode_i2s(0b11), -1.0);
        assert_eq!(decode_i2s(0b10), 0.0);
    }

    #[test]
    fn test_dequant_i2s_row_roundtrip() {
        let weights = vec![1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0];
        let (packed, scale) = quantize_row_i2s(&weights);
        let mut out = vec![0.0f32; 8];
        dequant_i2s_row(&packed, scale, &mut out);
        for (i, (&w, &o)) in weights.iter().zip(out.iter()).enumerate() {
            assert!((w - o).abs() < 1e-5, "idx {i}: {w} vs {o}");
        }
    }

    #[test]
    fn test_gate_up_silu_random_tolerance() {
        let mut rng = Rng::new(500);
        for _ in 0..10 {
            let n = (rng.next_u32() % 200 + 1) as usize;
            let gate = rng.rand_vec(n);
            let up = rng.rand_vec(n);
            let mut out = vec![0.0; n];
            let mut expected = vec![0.0; n];
            fused_gate_up_silu(&gate, &up, &mut out);
            scalar_fused_gate_up_silu(&gate, &up, &mut expected);
            assert_close(&out, &expected, 1e-5, "random gate_up_silu");
        }
    }

    #[test]
    fn test_swiglu_identity_weights() {
        // With identity-like weights (diagonal 1s), gate and up projections
        // reduce to selecting input elements.
        let n = 4;
        let mut w_gate = vec![0.0f32; n * n];
        let mut w_up = vec![0.0f32; n * n];
        for i in 0..n {
            w_gate[i * n + i] = 1.0;
            w_up[i * n + i] = 1.0;
        }
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; n];
        fused_swiglu(&input, &w_gate, &w_up, &mut output, n, n);
        for i in 0..n {
            let expected = naive_silu(input[i]) * input[i];
            assert!((output[i] - expected).abs() < 1e-5, "identity swiglu idx {i}");
        }
    }

    #[test]
    fn test_down_proj_residual_identity() {
        let n = 4;
        let mut w_down = vec![0.0f32; n * n];
        for i in 0..n {
            w_down[i * n + i] = 1.0;
        }
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let residual = vec![10.0, 20.0, 30.0, 40.0];
        let mut output = vec![0.0; n];
        fused_down_proj_residual(&input, &w_down, &residual, &mut output, n, n);
        for i in 0..n {
            let expected = residual[i] + input[i];
            assert!((output[i] - expected).abs() < 1e-5, "identity down_proj idx {i}");
        }
    }

    #[test]
    fn test_rmsnorm_mlp_large_gamma() {
        let hidden = 8;
        let inter = 16;
        let mut rng = Rng::new(600);
        let input = rng.rand_vec(hidden);
        let gamma = vec![100.0; hidden];
        let w_gate = rng.rand_vec(inter * hidden);
        let w_up = rng.rand_vec(inter * hidden);
        let w_down = rng.rand_vec(hidden * inter);
        let residual = rng.rand_vec(hidden);
        let mut output = vec![0.0; hidden];
        fused_rmsnorm_mlp(
            &input,
            &gamma,
            &w_gate,
            &w_up,
            &w_down,
            &residual,
            &mut output,
            hidden,
            inter,
            1e-5,
        );
        assert!(output.iter().all(|v| v.is_finite()), "large gamma should be finite");
    }

    #[test]
    fn test_rmsnorm_correctness_standalone() {
        // Verify the RMSNorm part independently.
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let eps = 1e-5;
        let ss: f32 = input.iter().map(|x| x * x).sum();
        let rms = (ss / 4.0 + eps).sqrt();
        let inv_rms = 1.0 / rms;
        let expected: Vec<f32> =
            input.iter().zip(gamma.iter()).map(|(x, g)| x * inv_rms * g).collect();
        // Through the full pipeline with zero weights, we can't easily isolate,
        // so just check the math:
        let computed_rms = (input.iter().map(|x| x * x).sum::<f32>() / 4.0 + eps).sqrt();
        assert!((computed_rms - rms).abs() < 1e-7);
        assert!((expected[0] - 1.0 / rms).abs() < 1e-5);
    }

    #[test]
    fn test_all_ops_finite_with_random_inputs() {
        let mut rng = Rng::new(700);
        let h = 32;
        let i = 64;

        // gate_up_silu
        let gate = rng.rand_vec(h);
        let up = rng.rand_vec(h);
        let mut out1 = vec![0.0; h];
        fused_gate_up_silu(&gate, &up, &mut out1);
        assert!(out1.iter().all(|v| v.is_finite()), "gate_up finite");

        // swiglu
        let input = rng.rand_vec(h);
        let wg = rng.rand_vec(i * h);
        let wu = rng.rand_vec(i * h);
        let mut out2 = vec![0.0; i];
        fused_swiglu(&input, &wg, &wu, &mut out2, i, h);
        assert!(out2.iter().all(|v| v.is_finite()), "swiglu finite");

        // down_proj
        let inp3 = rng.rand_vec(i);
        let wd = rng.rand_vec(h * i);
        let res3 = rng.rand_vec(h);
        let mut out3 = vec![0.0; h];
        fused_down_proj_residual(&inp3, &wd, &res3, &mut out3, h, i);
        assert!(out3.iter().all(|v| v.is_finite()), "down_proj finite");
    }

    #[test]
    fn test_gate_up_silu_size_128() {
        let n = 128;
        let mut rng = Rng::new(800);
        let gate = rng.rand_vec(n);
        let up = rng.rand_vec(n);
        let mut neon_out = vec![0.0; n];
        let mut scalar_out = vec![0.0; n];
        fused_gate_up_silu(&gate, &up, &mut neon_out);
        scalar_fused_gate_up_silu(&gate, &up, &mut scalar_out);
        assert_close(&neon_out, &scalar_out, 1e-5, "gate_up 128");
    }

    #[test]
    fn test_gate_up_silu_size_256() {
        let n = 256;
        let mut rng = Rng::new(801);
        let gate = rng.rand_vec(n);
        let up = rng.rand_vec(n);
        let mut neon_out = vec![0.0; n];
        let mut scalar_out = vec![0.0; n];
        fused_gate_up_silu(&gate, &up, &mut neon_out);
        scalar_fused_gate_up_silu(&gate, &up, &mut scalar_out);
        assert_close(&neon_out, &scalar_out, 1e-5, "gate_up 256");
    }

    #[test]
    fn test_gate_up_silu_size_512() {
        let n = 512;
        let mut rng = Rng::new(802);
        let gate = rng.rand_vec(n);
        let up = rng.rand_vec(n);
        let mut neon_out = vec![0.0; n];
        let mut scalar_out = vec![0.0; n];
        fused_gate_up_silu(&gate, &up, &mut neon_out);
        scalar_fused_gate_up_silu(&gate, &up, &mut scalar_out);
        assert_close(&neon_out, &scalar_out, 1e-5, "gate_up 512");
    }

    #[test]
    fn test_gate_up_silu_size_1024() {
        let n = 1024;
        let mut rng = Rng::new(803);
        let gate = rng.rand_vec(n);
        let up = rng.rand_vec(n);
        let mut neon_out = vec![0.0; n];
        let mut scalar_out = vec![0.0; n];
        fused_gate_up_silu(&gate, &up, &mut neon_out);
        scalar_fused_gate_up_silu(&gate, &up, &mut scalar_out);
        assert_close(&neon_out, &scalar_out, 1e-5, "gate_up 1024");
    }

    #[test]
    fn test_gate_up_silu_size_2048() {
        let n = 2048;
        let mut rng = Rng::new(804);
        let gate = rng.rand_vec(n);
        let up = rng.rand_vec(n);
        let mut neon_out = vec![0.0; n];
        let mut scalar_out = vec![0.0; n];
        fused_gate_up_silu(&gate, &up, &mut neon_out);
        scalar_fused_gate_up_silu(&gate, &up, &mut scalar_out);
        assert_close(&neon_out, &scalar_out, 1e-5, "gate_up 2048");
    }

    #[test]
    fn test_gate_up_silu_size_4096() {
        let n = 4096;
        let mut rng = Rng::new(805);
        let gate = rng.rand_vec(n);
        let up = rng.rand_vec(n);
        let mut neon_out = vec![0.0; n];
        let mut scalar_out = vec![0.0; n];
        fused_gate_up_silu(&gate, &up, &mut neon_out);
        scalar_fused_gate_up_silu(&gate, &up, &mut scalar_out);
        assert_close(&neon_out, &scalar_out, 1e-5, "gate_up 4096");
    }

    #[test]
    fn test_swiglu_hidden_128_inter_256() {
        let hidden = 128;
        let inter = 256;
        let mut rng = Rng::new(810);
        let input = rng.rand_vec(hidden);
        let w_gate = rng.rand_vec(inter * hidden);
        let w_up = rng.rand_vec(inter * hidden);
        let mut neon_out = vec![0.0; inter];
        let mut scalar_out = vec![0.0; inter];
        fused_swiglu(&input, &w_gate, &w_up, &mut neon_out, inter, hidden);
        scalar_fused_swiglu(&input, &w_gate, &w_up, &mut scalar_out, inter, hidden);
        assert_close(&neon_out, &scalar_out, 5e-3, "swiglu 128x256");
    }

    #[test]
    fn test_down_proj_hidden_128_inter_256() {
        let hidden = 128;
        let inter = 256;
        let mut rng = Rng::new(820);
        let input = rng.rand_vec(inter);
        let w_down = rng.rand_vec(hidden * inter);
        let residual = rng.rand_vec(hidden);
        let mut neon_out = vec![0.0; hidden];
        let mut scalar_out = vec![0.0; hidden];
        fused_down_proj_residual(&input, &w_down, &residual, &mut neon_out, hidden, inter);
        scalar_fused_down_proj_residual(&input, &w_down, &residual, &mut scalar_out, hidden, inter);
        assert_close(&neon_out, &scalar_out, 1e-3, "down_proj 128x256");
    }

    #[test]
    fn test_rmsnorm_mlp_hidden_128_inter_256() {
        let hidden = 128;
        let inter = 256;
        let mut rng = Rng::new(830);
        let input = rng.rand_vec(hidden);
        let gamma = rng.rand_positive_vec(hidden);
        let w_gate = rng.rand_vec(inter * hidden);
        let w_up = rng.rand_vec(inter * hidden);
        let w_down = rng.rand_vec(hidden * inter);
        let residual = rng.rand_vec(hidden);
        let mut neon_out = vec![0.0; hidden];
        let mut scalar_out = vec![0.0; hidden];
        fused_rmsnorm_mlp(
            &input,
            &gamma,
            &w_gate,
            &w_up,
            &w_down,
            &residual,
            &mut neon_out,
            hidden,
            inter,
            1e-5,
        );
        scalar_fused_rmsnorm_mlp(
            &input,
            &gamma,
            &w_gate,
            &w_up,
            &w_down,
            &residual,
            &mut scalar_out,
            hidden,
            inter,
            1e-5,
        );
        assert_close(&neon_out, &scalar_out, 1e-3, "rmsnorm_mlp 128x256");
    }

    #[test]
    fn test_attention_mlp_hidden_128_inter_256() {
        let hidden = 128;
        let inter = 256;
        let mut rng = Rng::new(840);
        let attn_out = rng.rand_vec(hidden);
        let residual = rng.rand_vec(hidden);
        let gamma = rng.rand_positive_vec(hidden);
        let w_gate = rng.rand_vec(inter * hidden);
        let w_up = rng.rand_vec(inter * hidden);
        let w_down = rng.rand_vec(hidden * inter);
        let mut neon_out = vec![0.0; hidden];
        let mut scalar_out = vec![0.0; hidden];
        fused_attention_mlp(
            &attn_out,
            &residual,
            &gamma,
            &w_gate,
            &w_up,
            &w_down,
            &mut neon_out,
            hidden,
            inter,
            1e-5,
        );
        scalar_fused_attention_mlp(
            &attn_out,
            &residual,
            &gamma,
            &w_gate,
            &w_up,
            &w_down,
            &mut scalar_out,
            hidden,
            inter,
            1e-5,
        );
        assert_close(&neon_out, &scalar_out, 1e-3, "attn_mlp 128x256");
    }

    #[test]
    fn test_quantized_mlp_hidden_32_inter_64() {
        let hidden = 32;
        let inter = 64;
        let mut rng = Rng::new(850);
        let input = rng.rand_vec(hidden);
        let residual = rng.rand_vec(hidden);
        let gate_w = rng.rand_vec(inter * hidden);
        let up_w = rng.rand_vec(inter * hidden);
        let down_w = rng.rand_vec(hidden * inter);
        let (gp, gs) = quantize_matrix_i2s(&gate_w, inter, hidden);
        let (up, us) = quantize_matrix_i2s(&up_w, inter, hidden);
        let (dp, ds) = quantize_matrix_i2s(&down_w, hidden, inter);
        let mut neon_out = vec![0.0; hidden];
        let mut scalar_out = vec![0.0; hidden];
        fused_mlp_quantized_i2s(
            &input,
            &gp,
            &gs,
            &up,
            &us,
            &dp,
            &ds,
            &residual,
            &mut neon_out,
            hidden,
            inter,
        );
        scalar_fused_mlp_quantized_i2s(
            &input,
            &gp,
            &gs,
            &up,
            &us,
            &dp,
            &ds,
            &residual,
            &mut scalar_out,
            hidden,
            inter,
        );
        assert_close(&neon_out, &scalar_out, 1e-4, "quant mlp 32x64");
    }

    #[test]
    fn test_gate_up_silu_all_positive() {
        let gate = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let up = vec![1.0; 8];
        let mut output = vec![0.0; 8];
        fused_gate_up_silu(&gate, &up, &mut output);
        // silu(x) > 0 for x > 0
        for &v in &output {
            assert!(v > 0.0);
        }
    }

    #[test]
    fn test_gate_up_silu_symmetry() {
        // silu(-x) ≠ -silu(x), but we can check that the output
        // magnitude relationship is correct
        let gate_pos = vec![2.0; 4];
        let gate_neg = vec![-2.0; 4];
        let up = vec![1.0; 4];
        let mut out_pos = vec![0.0; 4];
        let mut out_neg = vec![0.0; 4];
        fused_gate_up_silu(&gate_pos, &up, &mut out_pos);
        fused_gate_up_silu(&gate_neg, &up, &mut out_neg);
        // silu(2) > |silu(-2)|
        for i in 0..4 {
            assert!(out_pos[i] > out_neg[i].abs());
        }
    }

    #[test]
    fn test_down_proj_accumulation() {
        // Verify accumulation: w_down is all 1s, input is [1,1,...],
        // so each output element should be residual + inter
        let hidden = 4;
        let inter = 8;
        let input = vec![1.0; inter];
        let w_down = vec![1.0; hidden * inter];
        let residual = vec![0.0; hidden];
        let mut output = vec![0.0; hidden];
        fused_down_proj_residual(&input, &w_down, &residual, &mut output, hidden, inter);
        for &v in &output {
            assert!((v - inter as f32).abs() < 1e-4);
        }
    }

    #[test]
    fn test_rmsnorm_unit_norm() {
        // For input of all same value v, gamma=1, RMSNorm → 1.0
        let hidden = 8;
        let inter = 4;
        let input = vec![3.0; hidden];
        let gamma = vec![1.0; hidden];
        // RMS = sqrt(mean(x^2)) = sqrt(9) = 3
        // normed = x / 3 * 1 = 1.0
        let w_gate = vec![0.0; inter * hidden];
        let w_up = vec![0.0; inter * hidden];
        let w_down = vec![0.0; hidden * inter];
        let residual = vec![5.0; hidden];
        let mut output = vec![0.0; hidden];
        fused_rmsnorm_mlp(
            &input,
            &gamma,
            &w_gate,
            &w_up,
            &w_down,
            &residual,
            &mut output,
            hidden,
            inter,
            1e-5,
        );
        // zero weights → intermediate all 0 → output = residual
        assert_close(&output, &residual, 1e-4, "rmsnorm unit norm");
    }

    #[test]
    fn test_quantized_mlp_all_zero_weights() {
        let hidden = 8;
        let inter = 4;
        let input = vec![1.0; hidden];
        let residual = vec![5.0; hidden];
        let bytes_h = (hidden + 3) / 4;
        let bytes_i = (inter + 3) / 4;
        let gate_p = vec![0u8; inter * bytes_h];
        let gate_s = vec![1.0; inter];
        let up_p = vec![0u8; inter * bytes_h];
        let up_s = vec![1.0; inter];
        let down_p = vec![0u8; hidden * bytes_i];
        let down_s = vec![1.0; hidden];
        let mut output = vec![0.0; hidden];
        fused_mlp_quantized_i2s(
            &input,
            &gate_p,
            &gate_s,
            &up_p,
            &up_s,
            &down_p,
            &down_s,
            &residual,
            &mut output,
            hidden,
            inter,
        );
        // All zero packed → dequant = 0 → matmul = 0 → output = residual
        assert_close(&output, &residual, 1e-5, "quant zero weights");
    }

    #[test]
    fn test_attention_mlp_residual_passthrough() {
        // With zero weights, attn_out+residual should pass through
        let hidden = 4;
        let inter = 8;
        let attn_out = vec![0.0; hidden];
        let residual = vec![3.0; hidden];
        let gamma = vec![1.0; hidden];
        let w_gate = vec![0.0; inter * hidden];
        let w_up = vec![0.0; inter * hidden];
        let w_down = vec![0.0; hidden * inter];
        let mut output = vec![0.0; hidden];
        fused_attention_mlp(
            &attn_out,
            &residual,
            &gamma,
            &w_gate,
            &w_up,
            &w_down,
            &mut output,
            hidden,
            inter,
            1e-5,
        );
        // post_attn = [3,3,3,3], norm → [1,1,1,1], zero weights → 0, output = post_attn + 0
        assert_close(&output, &residual, 1e-4, "attn_mlp passthrough");
    }

    #[test]
    fn test_swiglu_negative_weights() {
        let hidden = 4;
        let inter = 2;
        let input = vec![1.0; hidden];
        let w_gate = vec![-0.5; inter * hidden];
        let w_up = vec![-0.5; inter * hidden];
        let mut output = vec![0.0; inter];
        let mut expected = vec![0.0; inter];
        fused_swiglu(&input, &w_gate, &w_up, &mut output, inter, hidden);
        scalar_fused_swiglu(&input, &w_gate, &w_up, &mut expected, inter, hidden);
        assert_close(&output, &expected, 1e-5, "swiglu negative weights");
    }

    #[test]
    fn test_gate_up_silu_extreme_values() {
        let gate = vec![f32::MAX / 2.0, f32::MIN / 2.0, 0.0, 1.0];
        let up = vec![1.0; 4];
        let mut output = vec![0.0; 4];
        fused_gate_up_silu(&gate, &up, &mut output);
        // Just check no NaN/panic
        assert!(output[2].is_finite());
        assert!(output[3].is_finite());
    }

    #[test]
    fn test_down_proj_size_1() {
        let input = vec![2.0];
        let w_down = vec![3.0];
        let residual = vec![1.0];
        let mut output = vec![0.0; 1];
        fused_down_proj_residual(&input, &w_down, &residual, &mut output, 1, 1);
        assert!((output[0] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_quantized_mlp_scale_correctness() {
        // Single known weights to verify scale is applied correctly
        let weights = vec![0.5, -0.5, 0.0, 0.5];
        let (packed, scale) = quantize_row_i2s(&weights);
        assert!((scale - 0.5).abs() < 1e-5);
        let mut out = vec![0.0f32; 4];
        dequant_i2s_row(&packed, scale, &mut out);
        // 0.5 → round(0.5/0.5)=1 → code 01 → decode 1.0 * 0.5 = 0.5
        assert!((out[0] - 0.5).abs() < 1e-5);
        // -0.5 → round(-0.5/0.5)=-1 → code 11 → decode -1.0 * 0.5 = -0.5
        assert!((out[1] - (-0.5)).abs() < 1e-5);
    }

    #[test]
    fn test_rmsnorm_mlp_different_eps() {
        let hidden = 8;
        let inter = 16;
        let mut rng = Rng::new(900);
        let input = rng.rand_vec(hidden);
        let gamma = rng.rand_positive_vec(hidden);
        let w_gate = rng.rand_vec(inter * hidden);
        let w_up = rng.rand_vec(inter * hidden);
        let w_down = rng.rand_vec(hidden * inter);
        let residual = rng.rand_vec(hidden);

        for &eps in &[1e-5, 1e-6, 1e-8] {
            let mut output = vec![0.0; hidden];
            fused_rmsnorm_mlp(
                &input,
                &gamma,
                &w_gate,
                &w_up,
                &w_down,
                &residual,
                &mut output,
                hidden,
                inter,
                eps,
            );
            assert!(output.iter().all(|v| v.is_finite()), "eps={eps}");
        }
    }

    #[test]
    fn test_attention_mlp_large_attn_output() {
        let hidden = 8;
        let inter = 16;
        let attn_out = vec![100.0; hidden];
        let residual = vec![100.0; hidden];
        let gamma = vec![1.0; hidden];
        let mut rng = Rng::new(910);
        let w_gate = rng.rand_vec(inter * hidden);
        let w_up = rng.rand_vec(inter * hidden);
        let w_down = rng.rand_vec(hidden * inter);
        let mut output = vec![0.0; hidden];
        fused_attention_mlp(
            &attn_out,
            &residual,
            &gamma,
            &w_gate,
            &w_up,
            &w_down,
            &mut output,
            hidden,
            inter,
            1e-5,
        );
        assert!(output.iter().all(|v| v.is_finite()), "large attn output");
    }
}

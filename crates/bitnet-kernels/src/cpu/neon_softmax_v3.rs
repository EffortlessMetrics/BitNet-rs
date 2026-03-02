//! Advanced NEON-optimized softmax v3 kernel for Apple Silicon (aarch64).
//!
//! Provides six softmax variants with NEON SIMD acceleration:
//! 1. Standard row-wise softmax
//! 2. Numerically stable log-softmax
//! 3. Online/streaming softmax (Milakov & Gimelshein single-pass algorithm)
//! 4. Fused softmax with attention mask
//! 5. Temperature-scaled softmax for sampling
//! 6. Grouped softmax for multi-head attention
//!
//! Each function has an `unsafe fn neon_*` NEON path, a `fn scalar_*` fallback,
//! and a public dispatcher that selects at runtime via
//! `is_aarch64_feature_detected!("neon")`.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Lane count for `float32x4_t` NEON vectors.
const LANES: usize = 4;

// ── Fast exp approximation ──────────────────────────────────────────────

/// Scalar fast exp approximation (degree-4 Cody–Waite polynomial).
/// Maximum relative error ≈ 2 × 10⁻⁴ for |x| ≤ 20.
#[inline(always)]
fn fast_exp_scalar(x: f32) -> f32 {
    let x = x.clamp(-88.0, 88.0);
    let n = (x * std::f32::consts::LOG2_E).round();
    let r = x - n * std::f32::consts::LN_2;
    let poly = 1.0 + r * (1.0 + r * (0.5 + r * (1.0 / 6.0 + r * (1.0 / 24.0))));
    poly * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

/// NEON vectorised fast exp for four lanes.
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

// ═══════════════════════════════════════════════════════════════════════
// 1. Standard softmax
// ═══════════════════════════════════════════════════════════════════════

/// NEON-accelerated softmax: `output[i] = exp(input[i] - max) / Σ exp(…)`.
///
/// # Safety
/// Requires `aarch64` target with NEON available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_softmax_f32(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    if len == 0 {
        return;
    }

    // Pass 1: find max
    let chunks = len / LANES;
    let remainder = len % LANES;
    let ptr = input.as_ptr();
    let mut max_vec = vdupq_n_f32(f32::NEG_INFINITY);
    for i in 0..chunks {
        let v = unsafe { vld1q_f32(ptr.add(i * LANES)) };
        max_vec = vmaxq_f32(max_vec, v);
    }
    let mut max_val = vmaxvq_f32(max_vec);
    for i in 0..remainder {
        max_val = max_val.max(input[chunks * LANES + i]);
    }

    // Pass 2: exp(x - max) and sum
    let max_v = vdupq_n_f32(max_val);
    let mut sum_vec = vdupq_n_f32(0.0);
    let out_ptr = output.as_mut_ptr();
    for i in 0..chunks {
        let v = unsafe { vld1q_f32(ptr.add(i * LANES)) };
        let shifted = vsubq_f32(v, max_v);
        let e = unsafe { fast_exp_neon(shifted) };
        sum_vec = vaddq_f32(sum_vec, e);
        unsafe { vst1q_f32(out_ptr.add(i * LANES), e) };
    }
    let mut sum_val = vaddvq_f32(sum_vec);
    let tail_start = chunks * LANES;
    for i in 0..remainder {
        let e = fast_exp_scalar(input[tail_start + i] - max_val);
        output[tail_start + i] = e;
        sum_val += e;
    }

    // Pass 3: normalize
    let inv_sum = 1.0 / sum_val;
    let inv_sum_v = vdupq_n_f32(inv_sum);
    for i in 0..chunks {
        let e = unsafe { vld1q_f32(out_ptr.add(i * LANES)) };
        let r = vmulq_f32(e, inv_sum_v);
        unsafe { vst1q_f32(out_ptr.add(i * LANES), r) };
    }
    for i in 0..remainder {
        output[tail_start + i] *= inv_sum;
    }
}

/// Scalar fallback softmax.
fn scalar_softmax_f32(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    if len == 0 {
        return;
    }
    let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for i in 0..len {
        let e = fast_exp_scalar(input[i] - max_val);
        output[i] = e;
        sum += e;
    }
    let inv = 1.0 / sum;
    for i in 0..len {
        output[i] *= inv;
    }
}

/// Row-wise softmax with automatic NEON dispatch.
///
/// # Panics
/// Panics if `output.len() < input.len()`.
pub fn softmax_f32(input: &[f32], output: &mut [f32]) {
    assert!(
        output.len() >= input.len(),
        "output buffer too small: {} < {}",
        output.len(),
        input.len()
    );
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { neon_softmax_f32(input, output) };
            return;
        }
    }
    scalar_softmax_f32(input, output);
}

// ═══════════════════════════════════════════════════════════════════════
// 2. Log-softmax
// ═══════════════════════════════════════════════════════════════════════

/// NEON-accelerated log-softmax: `output[i] = (input[i] - max) - ln(Σ exp(…))`.
///
/// # Safety
/// Requires `aarch64` target with NEON available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_log_softmax_f32(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    if len == 0 {
        return;
    }

    let chunks = len / LANES;
    let remainder = len % LANES;
    let ptr = input.as_ptr();

    // Pass 1: max
    let mut max_vec = vdupq_n_f32(f32::NEG_INFINITY);
    for i in 0..chunks {
        let v = unsafe { vld1q_f32(ptr.add(i * LANES)) };
        max_vec = vmaxq_f32(max_vec, v);
    }
    let mut max_val = vmaxvq_f32(max_vec);
    for i in 0..remainder {
        max_val = max_val.max(input[chunks * LANES + i]);
    }

    // Pass 2: sum of exp(x - max)
    let max_v = vdupq_n_f32(max_val);
    let mut sum_vec = vdupq_n_f32(0.0);
    for i in 0..chunks {
        let v = unsafe { vld1q_f32(ptr.add(i * LANES)) };
        let shifted = vsubq_f32(v, max_v);
        let e = unsafe { fast_exp_neon(shifted) };
        sum_vec = vaddq_f32(sum_vec, e);
    }
    let mut sum_val = vaddvq_f32(sum_vec);
    let tail_start = chunks * LANES;
    for i in 0..remainder {
        sum_val += fast_exp_scalar(input[tail_start + i] - max_val);
    }

    // Pass 3: output = (x - max) - ln(sum)
    let log_sum = sum_val.ln();
    let log_sum_v = vdupq_n_f32(log_sum);
    let out_ptr = output.as_mut_ptr();
    for i in 0..chunks {
        let v = unsafe { vld1q_f32(ptr.add(i * LANES)) };
        let shifted = vsubq_f32(v, max_v);
        let result = vsubq_f32(shifted, log_sum_v);
        unsafe { vst1q_f32(out_ptr.add(i * LANES), result) };
    }
    for i in 0..remainder {
        output[tail_start + i] = (input[tail_start + i] - max_val) - log_sum;
    }
}

/// Scalar fallback log-softmax.
fn scalar_log_softmax_f32(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    if len == 0 {
        return;
    }
    let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for i in 0..len {
        sum += fast_exp_scalar(input[i] - max_val);
    }
    let log_sum = sum.ln();
    for i in 0..len {
        output[i] = (input[i] - max_val) - log_sum;
    }
}

/// Numerically stable log-softmax with automatic NEON dispatch.
///
/// # Panics
/// Panics if `output.len() < input.len()`.
pub fn log_softmax_f32(input: &[f32], output: &mut [f32]) {
    assert!(
        output.len() >= input.len(),
        "output buffer too small: {} < {}",
        output.len(),
        input.len()
    );
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { neon_log_softmax_f32(input, output) };
            return;
        }
    }
    scalar_log_softmax_f32(input, output);
}

// ═══════════════════════════════════════════════════════════════════════
// 3. Online softmax (Milakov & Gimelshein single-pass algorithm)
// ═══════════════════════════════════════════════════════════════════════

/// NEON-accelerated online/streaming softmax.
///
/// Single-pass algorithm: tracks running max and running sum-of-exp,
/// then normalises in a second pass. This avoids a separate max-finding
/// pass, reducing memory traffic.
///
/// # Safety
/// Requires `aarch64` target with NEON available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_online_softmax_f32(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    if len == 0 {
        return;
    }

    let chunks = len / LANES;
    let remainder = len % LANES;
    let ptr = input.as_ptr();

    // Online pass: maintain running max and running sum of exp(x - max).
    let mut max_vec = vdupq_n_f32(f32::NEG_INFINITY);
    let mut sum_vec = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let v = unsafe { vld1q_f32(ptr.add(i * LANES)) };
        let old_max = max_vec;
        max_vec = vmaxq_f32(max_vec, v);
        // Rescale running sum: sum *= exp(old_max - new_max)
        let correction = vsubq_f32(old_max, max_vec);
        let scale = unsafe { fast_exp_neon(correction) };
        sum_vec = vmulq_f32(sum_vec, scale);
        // Add new contributions
        let shifted = vsubq_f32(v, max_vec);
        let e = unsafe { fast_exp_neon(shifted) };
        sum_vec = vaddq_f32(sum_vec, e);
    }

    // Reduce NEON lanes to scalar running state.
    let mut max_val = vmaxvq_f32(max_vec);

    // Merge four independent online accumulators.
    let mut running_sum = 0.0f32;
    if chunks > 0 {
        let mut lane_maxes = [0.0f32; LANES];
        let mut lane_sums = [0.0f32; LANES];
        unsafe { vst1q_f32(lane_maxes.as_mut_ptr(), max_vec) };
        unsafe { vst1q_f32(lane_sums.as_mut_ptr(), sum_vec) };
        for lane in 0..LANES {
            let correction = fast_exp_scalar(lane_maxes[lane] - max_val);
            running_sum += lane_sums[lane] * correction;
        }
    }

    // Scalar tail with online update.
    let tail_start = chunks * LANES;
    for i in 0..remainder {
        let val = input[tail_start + i];
        if val > max_val {
            running_sum *= fast_exp_scalar(max_val - val);
            max_val = val;
        }
        running_sum += fast_exp_scalar(val - max_val);
    }

    // Normalisation pass: output[i] = exp(input[i] - max) / sum
    let inv_sum = 1.0 / running_sum;
    let max_v = vdupq_n_f32(max_val);
    let inv_sum_v = vdupq_n_f32(inv_sum);
    let out_ptr = output.as_mut_ptr();
    for i in 0..chunks {
        let v = unsafe { vld1q_f32(ptr.add(i * LANES)) };
        let shifted = vsubq_f32(v, max_v);
        let e = unsafe { fast_exp_neon(shifted) };
        let r = vmulq_f32(e, inv_sum_v);
        unsafe { vst1q_f32(out_ptr.add(i * LANES), r) };
    }
    for i in 0..remainder {
        let e = fast_exp_scalar(input[tail_start + i] - max_val);
        output[tail_start + i] = e * inv_sum;
    }
}

/// Scalar fallback online softmax.
fn scalar_online_softmax_f32(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    if len == 0 {
        return;
    }

    // Single-pass: online max + sum tracking.
    let mut max_val = f32::NEG_INFINITY;
    let mut sum = 0.0f32;
    for i in 0..len {
        let val = input[i];
        if val > max_val {
            sum *= fast_exp_scalar(max_val - val);
            max_val = val;
        }
        sum += fast_exp_scalar(val - max_val);
    }

    // Normalisation pass.
    let inv = 1.0 / sum;
    for i in 0..len {
        output[i] = fast_exp_scalar(input[i] - max_val) * inv;
    }
}

/// Online/streaming softmax (Milakov & Gimelshein) with automatic NEON dispatch.
///
/// # Panics
/// Panics if `output.len() < input.len()`.
pub fn online_softmax_f32(input: &[f32], output: &mut [f32]) {
    assert!(
        output.len() >= input.len(),
        "output buffer too small: {} < {}",
        output.len(),
        input.len()
    );
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { neon_online_softmax_f32(input, output) };
            return;
        }
    }
    scalar_online_softmax_f32(input, output);
}

// ═══════════════════════════════════════════════════════════════════════
// 4. Fused softmax with attention mask
// ═══════════════════════════════════════════════════════════════════════

/// NEON-accelerated fused softmax with mask.
///
/// Masked positions (where `mask[i] == true`) are replaced with `neg_inf`
/// before applying softmax, effectively zeroing those positions in the output.
///
/// # Safety
/// Requires `aarch64` target with NEON available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_fused_softmax_mask_f32(
    input: &[f32],
    mask: &[bool],
    output: &mut [f32],
    neg_inf: f32,
) {
    let len = input.len();
    if len == 0 {
        return;
    }

    let chunks = len / LANES;
    let remainder = len % LANES;
    let ptr = input.as_ptr();
    let neg_inf_v = vdupq_n_f32(neg_inf);

    // Pass 1: apply mask and find max
    let mut max_vec = vdupq_n_f32(f32::NEG_INFINITY);
    let out_ptr = output.as_mut_ptr();

    for i in 0..chunks {
        let v = unsafe { vld1q_f32(ptr.add(i * LANES)) };
        let base = i * LANES;
        // Build mask for this chunk — apply per-lane.
        let mut masked = [0.0f32; LANES];
        for lane in 0..LANES {
            masked[lane] = if mask[base + lane] { neg_inf } else { input[base + lane] };
        }
        let masked_v = unsafe { vld1q_f32(masked.as_ptr()) };
        unsafe { vst1q_f32(out_ptr.add(base), masked_v) };
        max_vec = vmaxq_f32(max_vec, masked_v);
        let _ = v; // original loaded but we used masked version
    }
    let mut max_val = vmaxvq_f32(max_vec);
    let tail_start = chunks * LANES;
    for i in 0..remainder {
        let idx = tail_start + i;
        let val = if mask[idx] { neg_inf } else { input[idx] };
        output[idx] = val;
        max_val = max_val.max(val);
    }

    // Pass 2: exp(masked - max) and sum
    let max_v = vdupq_n_f32(max_val);
    let mut sum_vec = vdupq_n_f32(0.0);
    for i in 0..chunks {
        let v = unsafe { vld1q_f32(out_ptr.add(i * LANES)) };
        let shifted = vsubq_f32(v, max_v);
        let e = unsafe { fast_exp_neon(shifted) };
        sum_vec = vaddq_f32(sum_vec, e);
        unsafe { vst1q_f32(out_ptr.add(i * LANES), e) };
    }
    let mut sum_val = vaddvq_f32(sum_vec);
    for i in 0..remainder {
        let idx = tail_start + i;
        let e = fast_exp_scalar(output[idx] - max_val);
        output[idx] = e;
        sum_val += e;
    }

    // Pass 3: normalize
    let inv_sum = if sum_val > 0.0 { 1.0 / sum_val } else { 0.0 };
    let inv_sum_v = vdupq_n_f32(inv_sum);
    for i in 0..chunks {
        let e = unsafe { vld1q_f32(out_ptr.add(i * LANES)) };
        let r = vmulq_f32(e, inv_sum_v);
        unsafe { vst1q_f32(out_ptr.add(i * LANES), r) };
    }
    for i in 0..remainder {
        output[tail_start + i] *= inv_sum;
    }
}

/// Scalar fallback fused softmax with mask.
fn scalar_fused_softmax_mask_f32(
    input: &[f32],
    mask: &[bool],
    output: &mut [f32],
    neg_inf: f32,
) {
    let len = input.len();
    if len == 0 {
        return;
    }

    // Apply mask
    let mut max_val = f32::NEG_INFINITY;
    for i in 0..len {
        let val = if mask[i] { neg_inf } else { input[i] };
        output[i] = val;
        max_val = max_val.max(val);
    }

    // Exp and sum
    let mut sum = 0.0f32;
    for i in 0..len {
        let e = fast_exp_scalar(output[i] - max_val);
        output[i] = e;
        sum += e;
    }

    // Normalize
    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for i in 0..len {
        output[i] *= inv;
    }
}

/// Softmax with attention mask applied, with automatic NEON dispatch.
///
/// Positions where `mask[i] == true` are set to `neg_inf` before softmax,
/// effectively zeroing them in the output distribution.
///
/// # Panics
/// Panics if `output.len() < input.len()` or `mask.len() < input.len()`.
pub fn fused_softmax_mask_f32(
    input: &[f32],
    mask: &[bool],
    output: &mut [f32],
    neg_inf: f32,
) {
    assert!(
        output.len() >= input.len(),
        "output buffer too small: {} < {}",
        output.len(),
        input.len()
    );
    assert!(
        mask.len() >= input.len(),
        "mask buffer too small: {} < {}",
        mask.len(),
        input.len()
    );
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { neon_fused_softmax_mask_f32(input, mask, output, neg_inf) };
            return;
        }
    }
    scalar_fused_softmax_mask_f32(input, mask, output, neg_inf);
}

// ═══════════════════════════════════════════════════════════════════════
// 5. Temperature-scaled softmax
// ═══════════════════════════════════════════════════════════════════════

/// NEON-accelerated temperature-scaled softmax.
///
/// `output[i] = exp((input[i] - max) / temperature) / Σ exp(…)`
///
/// # Safety
/// Requires `aarch64` target with NEON available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_softmax_temperature_f32(input: &[f32], output: &mut [f32], inv_temp: f32) {
    let len = input.len();
    if len == 0 {
        return;
    }

    let chunks = len / LANES;
    let remainder = len % LANES;
    let ptr = input.as_ptr();

    // Pass 1: find max
    let mut max_vec = vdupq_n_f32(f32::NEG_INFINITY);
    for i in 0..chunks {
        let v = unsafe { vld1q_f32(ptr.add(i * LANES)) };
        max_vec = vmaxq_f32(max_vec, v);
    }
    let mut max_val = vmaxvq_f32(max_vec);
    for i in 0..remainder {
        max_val = max_val.max(input[chunks * LANES + i]);
    }

    // Pass 2: exp((x - max) * inv_temp) and sum
    let max_v = vdupq_n_f32(max_val);
    let inv_temp_v = vdupq_n_f32(inv_temp);
    let mut sum_vec = vdupq_n_f32(0.0);
    let out_ptr = output.as_mut_ptr();

    for i in 0..chunks {
        let v = unsafe { vld1q_f32(ptr.add(i * LANES)) };
        let shifted = vsubq_f32(v, max_v);
        let scaled = vmulq_f32(shifted, inv_temp_v);
        let e = unsafe { fast_exp_neon(scaled) };
        sum_vec = vaddq_f32(sum_vec, e);
        unsafe { vst1q_f32(out_ptr.add(i * LANES), e) };
    }
    let mut sum_val = vaddvq_f32(sum_vec);
    let tail_start = chunks * LANES;
    for i in 0..remainder {
        let e = fast_exp_scalar((input[tail_start + i] - max_val) * inv_temp);
        output[tail_start + i] = e;
        sum_val += e;
    }

    // Pass 3: normalize
    let inv_sum = 1.0 / sum_val;
    let inv_sum_v = vdupq_n_f32(inv_sum);
    for i in 0..chunks {
        let e = unsafe { vld1q_f32(out_ptr.add(i * LANES)) };
        let r = vmulq_f32(e, inv_sum_v);
        unsafe { vst1q_f32(out_ptr.add(i * LANES), r) };
    }
    for i in 0..remainder {
        output[tail_start + i] *= inv_sum;
    }
}

/// Scalar fallback temperature-scaled softmax.
fn scalar_softmax_temperature_f32(input: &[f32], output: &mut [f32], inv_temp: f32) {
    let len = input.len();
    if len == 0 {
        return;
    }
    let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for i in 0..len {
        let e = fast_exp_scalar((input[i] - max_val) * inv_temp);
        output[i] = e;
        sum += e;
    }
    let inv = 1.0 / sum;
    for i in 0..len {
        output[i] *= inv;
    }
}

/// Temperature-scaled softmax with automatic NEON dispatch.
///
/// When `temperature` is very close to zero (< 1e-7), falls back to
/// argmax (one-hot output). Temperature must be non-negative.
///
/// # Panics
/// Panics if `output.len() < input.len()` or `temperature` is negative.
pub fn softmax_temperature_f32(input: &[f32], output: &mut [f32], temperature: f32) {
    assert!(
        output.len() >= input.len(),
        "output buffer too small: {} < {}",
        output.len(),
        input.len()
    );
    assert!(temperature >= 0.0, "temperature must be non-negative");
    let len = input.len();
    if len == 0 {
        return;
    }

    // Near-zero temperature → argmax (one-hot).
    if temperature < 1e-7 {
        let mut max_idx = 0;
        let mut max_val = f32::NEG_INFINITY;
        for i in 0..len {
            if input[i] > max_val {
                max_val = input[i];
                max_idx = i;
            }
        }
        for i in 0..len {
            output[i] = 0.0;
        }
        output[max_idx] = 1.0;
        return;
    }

    let inv_temp = 1.0 / temperature;

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { neon_softmax_temperature_f32(input, output, inv_temp) };
            return;
        }
    }
    scalar_softmax_temperature_f32(input, output, inv_temp);
}

// ═══════════════════════════════════════════════════════════════════════
// 6. Grouped softmax (for multi-head attention)
// ═══════════════════════════════════════════════════════════════════════

/// Softmax applied independently to consecutive groups.
///
/// # Panics
/// Panics if `output.len() < input.len()`, `group_size == 0`, or
/// `input.len()` is not divisible by `group_size`.
pub fn grouped_softmax_f32(input: &[f32], group_size: usize, output: &mut [f32]) {
    let len = input.len();
    assert!(
        output.len() >= len,
        "output buffer too small: {} < {}",
        output.len(),
        len
    );
    assert!(group_size > 0, "group_size must be > 0");
    assert!(
        len % group_size == 0,
        "input length {} not divisible by group_size {}",
        len,
        group_size
    );

    let num_groups = len / group_size;
    for g in 0..num_groups {
        let start = g * group_size;
        let end = start + group_size;
        softmax_f32(&input[start..end], &mut output[start..end]);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference softmax using f64 for accuracy comparison.
    fn reference_softmax_f64(input: &[f32]) -> Vec<f32> {
        if input.is_empty() {
            return vec![];
        }
        let max_val = input.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b as f64));
        let exps: Vec<f64> = input.iter().map(|&x| ((x as f64) - max_val).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|&e| (e / sum) as f32).collect()
    }

    /// Reference log-softmax using f64.
    fn reference_log_softmax_f64(input: &[f32]) -> Vec<f32> {
        if input.is_empty() {
            return vec![];
        }
        let max_val = input.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b as f64));
        let exps: Vec<f64> = input.iter().map(|&x| ((x as f64) - max_val).exp()).collect();
        let sum: f64 = exps.iter().sum();
        let log_sum = sum.ln();
        input
            .iter()
            .map(|&x| ((x as f64 - max_val) - log_sum) as f32)
            .collect()
    }

    /// Tolerance for fast_exp approximation (relative error ~2e-4).
    const APPROX_TOL: f32 = 5e-3;

    fn assert_close(a: &[f32], b: &[f32], tol: f32, msg: &str) {
        assert_eq!(a.len(), b.len(), "{msg}: length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (x - y).abs();
            let denom = x.abs().max(y.abs()).max(1e-12);
            assert!(
                diff / denom < tol || diff < tol,
                "{msg}: index {i}: {x} vs {y} (diff={diff})"
            );
        }
    }

    // ── 1. Standard softmax tests ──────────────────────────────────────

    #[test]
    fn test_softmax_empty() {
        let input: Vec<f32> = vec![];
        let mut output = vec![];
        softmax_f32(&input, &mut output);
        assert!(output.is_empty());
    }

    #[test]
    fn test_softmax_single() {
        let input = vec![5.0];
        let mut output = vec![0.0];
        softmax_f32(&input, &mut output);
        assert!((output[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_two_elements() {
        let input = vec![1.0, 2.0];
        let mut output = vec![0.0; 2];
        softmax_f32(&input, &mut output);
        let expected = reference_softmax_f64(&input);
        assert_close(&output, &expected, APPROX_TOL, "softmax_two");
    }

    #[test]
    fn test_softmax_uniform() {
        let input = vec![1.0; 8];
        let mut output = vec![0.0; 8];
        softmax_f32(&input, &mut output);
        for &v in &output {
            assert!((v - 0.125).abs() < 1e-5, "uniform should give 1/n");
        }
    }

    #[test]
    fn test_softmax_vs_reference_small() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut output = vec![0.0; 5];
        softmax_f32(&input, &mut output);
        let expected = reference_softmax_f64(&input);
        assert_close(&output, &expected, APPROX_TOL, "softmax_small");
    }

    #[test]
    fn test_softmax_vs_reference_8() {
        let input: Vec<f32> = (0..8).map(|i| i as f32 * 0.5).collect();
        let mut output = vec![0.0; 8];
        softmax_f32(&input, &mut output);
        let expected = reference_softmax_f64(&input);
        assert_close(&output, &expected, APPROX_TOL, "softmax_8");
    }

    #[test]
    fn test_softmax_vs_reference_non_aligned() {
        let input: Vec<f32> = (0..11).map(|i| i as f32 - 5.0).collect();
        let mut output = vec![0.0; 11];
        softmax_f32(&input, &mut output);
        let expected = reference_softmax_f64(&input);
        assert_close(&output, &expected, APPROX_TOL, "softmax_11");
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let input = vec![0.1, 0.5, -0.3, 2.0, 1.5, -1.0, 0.7, 3.0, -2.0];
        let mut output = vec![0.0; 9];
        softmax_f32(&input, &mut output);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "sum = {sum}");
    }

    #[test]
    fn test_softmax_all_negative() {
        let input = vec![-10.0, -20.0, -30.0, -5.0];
        let mut output = vec![0.0; 4];
        softmax_f32(&input, &mut output);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
        // -5.0 should be the largest probability
        assert!(output[3] > output[0]);
    }

    #[test]
    fn test_softmax_large_values() {
        let input = vec![80.0, 81.0, 82.0, 83.0];
        let mut output = vec![0.0; 4];
        softmax_f32(&input, &mut output);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3, "sum = {sum}");
        assert!(output[3] > output[2]);
    }

    #[test]
    fn test_softmax_large_negative_values() {
        let input = vec![-80.0, -81.0, -82.0, -83.0];
        let mut output = vec![0.0; 4];
        softmax_f32(&input, &mut output);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3, "sum = {sum}");
    }

    #[test]
    fn test_softmax_mixed_extreme() {
        let input = vec![-50.0, 0.0, 50.0];
        let mut output = vec![0.0; 3];
        softmax_f32(&input, &mut output);
        // 50.0 should dominate
        assert!(output[2] > 0.99);
    }

    #[test]
    fn test_softmax_large_array() {
        let input: Vec<f32> = (0..256).map(|i| (i as f32 / 25.6) - 5.0).collect();
        let mut output = vec![0.0; 256];
        softmax_f32(&input, &mut output);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3, "sum = {sum}");
    }

    // ── 2. Log-softmax tests ───────────────────────────────────────────

    #[test]
    fn test_log_softmax_empty() {
        let input: Vec<f32> = vec![];
        let mut output = vec![];
        log_softmax_f32(&input, &mut output);
        assert!(output.is_empty());
    }

    #[test]
    fn test_log_softmax_single() {
        let input = vec![42.0];
        let mut output = vec![0.0];
        log_softmax_f32(&input, &mut output);
        assert!(output[0].abs() < 1e-4, "log_softmax(single) should be ~0");
    }

    #[test]
    fn test_log_softmax_vs_reference() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut output = vec![0.0; 5];
        log_softmax_f32(&input, &mut output);
        let expected = reference_log_softmax_f64(&input);
        assert_close(&output, &expected, APPROX_TOL, "log_softmax_ref");
    }

    #[test]
    fn test_log_softmax_values_nonpositive() {
        let input = vec![1.0, 3.0, 2.0, 0.5, 4.0, -1.0, 2.5, 1.5];
        let mut output = vec![0.0; 8];
        log_softmax_f32(&input, &mut output);
        for (i, &v) in output.iter().enumerate() {
            assert!(v <= 1e-5, "log_softmax[{i}] = {v} should be <= 0");
        }
    }

    #[test]
    fn test_log_softmax_exp_sums_to_one() {
        let input = vec![0.5, -0.5, 1.0, -1.0, 0.0];
        let mut output = vec![0.0; 5];
        log_softmax_f32(&input, &mut output);
        let sum: f32 = output.iter().map(|&x| fast_exp_scalar(x)).sum();
        assert!((sum - 1.0).abs() < 1e-3, "exp(log_softmax) sum = {sum}");
    }

    #[test]
    fn test_log_softmax_uniform() {
        let n = 8;
        let input = vec![0.0; n];
        let mut output = vec![0.0; n];
        log_softmax_f32(&input, &mut output);
        let expected = -(n as f32).ln();
        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - expected).abs() < APPROX_TOL,
                "log_softmax uniform[{i}] = {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_log_softmax_non_aligned() {
        let input: Vec<f32> = (0..7).map(|i| i as f32).collect();
        let mut output = vec![0.0; 7];
        log_softmax_f32(&input, &mut output);
        let expected = reference_log_softmax_f64(&input);
        assert_close(&output, &expected, APPROX_TOL, "log_softmax_7");
    }

    #[test]
    fn test_log_softmax_large_negative() {
        let input = vec![-100.0, -50.0, -10.0, -1.0];
        let mut output = vec![0.0; 4];
        log_softmax_f32(&input, &mut output);
        for &v in &output {
            assert!(v <= 1e-5);
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_log_softmax_consistent_with_softmax() {
        let input = vec![2.0, 1.0, 0.5, 3.0, -1.0];
        let mut sm = vec![0.0; 5];
        let mut lsm = vec![0.0; 5];
        softmax_f32(&input, &mut sm);
        log_softmax_f32(&input, &mut lsm);
        for i in 0..5 {
            let ln_sm = sm[i].ln();
            assert!(
                (ln_sm - lsm[i]).abs() < APPROX_TOL,
                "ln(softmax)[{i}]={ln_sm} vs log_softmax[{i}]={}",
                lsm[i]
            );
        }
    }

    // ── 3. Online softmax tests ────────────────────────────────────────

    #[test]
    fn test_online_softmax_empty() {
        let input: Vec<f32> = vec![];
        let mut output = vec![];
        online_softmax_f32(&input, &mut output);
    }

    #[test]
    fn test_online_softmax_single() {
        let input = vec![7.0];
        let mut output = vec![0.0];
        online_softmax_f32(&input, &mut output);
        assert!((output[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_online_softmax_vs_standard() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut standard = vec![0.0; 8];
        let mut online = vec![0.0; 8];
        softmax_f32(&input, &mut standard);
        online_softmax_f32(&input, &mut online);
        assert_close(&online, &standard, APPROX_TOL, "online_vs_standard");
    }

    #[test]
    fn test_online_softmax_vs_standard_non_aligned() {
        let input: Vec<f32> = (0..13).map(|i| i as f32 * 0.3 - 2.0).collect();
        let mut standard = vec![0.0; 13];
        let mut online = vec![0.0; 13];
        softmax_f32(&input, &mut standard);
        online_softmax_f32(&input, &mut online);
        assert_close(&online, &standard, APPROX_TOL, "online_vs_standard_13");
    }

    #[test]
    fn test_online_softmax_sums_to_one() {
        let input = vec![0.3, -1.5, 2.7, 0.0, -0.5, 1.2, -3.0, 4.1, 0.9];
        let mut output = vec![0.0; 9];
        online_softmax_f32(&input, &mut output);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3, "sum = {sum}");
    }

    #[test]
    fn test_online_softmax_large_values() {
        let input = vec![50.0, 51.0, 52.0, 53.0];
        let mut output = vec![0.0; 4];
        online_softmax_f32(&input, &mut output);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_online_softmax_uniform() {
        let input = vec![3.0; 16];
        let mut output = vec![0.0; 16];
        online_softmax_f32(&input, &mut output);
        for &v in &output {
            assert!((v - 1.0 / 16.0).abs() < 1e-4);
        }
    }

    // ── 4. Fused softmax mask tests ────────────────────────────────────

    #[test]
    fn test_fused_mask_empty() {
        let input: Vec<f32> = vec![];
        let mask: Vec<bool> = vec![];
        let mut output = vec![];
        fused_softmax_mask_f32(&input, &mask, &mut output, f32::NEG_INFINITY);
    }

    #[test]
    fn test_fused_mask_none_masked() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![false; 4];
        let mut output = vec![0.0; 4];
        let mut expected = vec![0.0; 4];
        fused_softmax_mask_f32(&input, &mask, &mut output, f32::NEG_INFINITY);
        softmax_f32(&input, &mut expected);
        assert_close(&output, &expected, APPROX_TOL, "no_mask");
    }

    #[test]
    fn test_fused_mask_all_masked() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![true; 4];
        let mut output = vec![0.0; 4];
        fused_softmax_mask_f32(&input, &mask, &mut output, -1e9);
        // When all masked with finite neg_inf, all get the same value → uniform.
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3, "all masked sum = {sum}");
        for &v in &output {
            assert!((v - 0.25).abs() < 1e-3);
        }
    }

    #[test]
    fn test_fused_mask_alternating() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mask = vec![true, false, true, false, true, false, true, false];
        let mut output = vec![0.0; 8];
        fused_softmax_mask_f32(&input, &mask, &mut output, -1e9);
        // Masked positions should be ~0
        for i in (0..8).step_by(2) {
            assert!(output[i] < 1e-4, "masked[{i}] = {}", output[i]);
        }
        // Unmasked should sum to ~1
        let unmasked_sum: f32 = (0..8).step_by(2).map(|i| output[i + 1]).sum();
        assert!((unmasked_sum - 1.0).abs() < 1e-2, "unmasked_sum = {unmasked_sum}");
    }

    #[test]
    fn test_fused_mask_single_unmasked() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mask = vec![true, true, false, true, true];
        let mut output = vec![0.0; 5];
        fused_softmax_mask_f32(&input, &mask, &mut output, -1e9);
        // Only index 2 is unmasked — should get ~1.0
        assert!(output[2] > 0.99, "single unmasked = {}", output[2]);
    }

    #[test]
    fn test_fused_mask_non_aligned() {
        let input: Vec<f32> = (0..9).map(|i| i as f32).collect();
        let mask = vec![false, true, false, true, false, true, false, true, false];
        let mut output = vec![0.0; 9];
        fused_softmax_mask_f32(&input, &mask, &mut output, f32::NEG_INFINITY);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_fused_mask_neg_inf_values() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![true, false, true, false];
        let mut output = vec![0.0; 4];
        fused_softmax_mask_f32(&input, &mask, &mut output, -1e30);
        assert!(output[0] < 1e-6);
        assert!(output[2] < 1e-6);
    }

    // ── 5. Temperature-scaled softmax tests ────────────────────────────

    #[test]
    fn test_temperature_one() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut temp1 = vec![0.0; 4];
        let mut standard = vec![0.0; 4];
        softmax_temperature_f32(&input, &mut temp1, 1.0);
        softmax_f32(&input, &mut standard);
        assert_close(&temp1, &standard, APPROX_TOL, "temp=1");
    }

    #[test]
    fn test_temperature_high_uniform() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        softmax_temperature_f32(&input, &mut output, 100.0);
        // High temperature → nearly uniform.
        for &v in &output {
            assert!((v - 0.25).abs() < 0.05, "high temp should be ~uniform: {v}");
        }
    }

    #[test]
    fn test_temperature_low_peaked() {
        let input = vec![1.0, 2.0, 3.0, 10.0];
        let mut output = vec![0.0; 4];
        softmax_temperature_f32(&input, &mut output, 0.01);
        // Low temperature → peaked at max.
        assert!(output[3] > 0.99, "low temp peak = {}", output[3]);
    }

    #[test]
    fn test_temperature_zero_argmax() {
        let input = vec![1.0, 5.0, 3.0, 2.0];
        let mut output = vec![0.0; 4];
        softmax_temperature_f32(&input, &mut output, 0.0);
        assert!((output[1] - 1.0).abs() < 1e-6);
        assert!(output[0].abs() < 1e-6);
        assert!(output[2].abs() < 1e-6);
        assert!(output[3].abs() < 1e-6);
    }

    #[test]
    fn test_temperature_sums_to_one() {
        let input = vec![0.5, -1.0, 2.0, 0.0, 1.5];
        for temp in [0.1, 0.5, 1.0, 2.0, 10.0] {
            let mut output = vec![0.0; 5];
            softmax_temperature_f32(&input, &mut output, temp);
            let sum: f32 = output.iter().sum();
            assert!((sum - 1.0).abs() < 1e-3, "temp={temp} sum={sum}");
        }
    }

    #[test]
    fn test_temperature_empty() {
        let input: Vec<f32> = vec![];
        let mut output = vec![];
        softmax_temperature_f32(&input, &mut output, 1.0);
    }

    #[test]
    fn test_temperature_single() {
        let input = vec![42.0];
        let mut output = vec![0.0];
        softmax_temperature_f32(&input, &mut output, 2.5);
        assert!((output[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_temperature_very_high() {
        let input = vec![1.0, 100.0, -100.0, 50.0];
        let mut output = vec![0.0; 4];
        softmax_temperature_f32(&input, &mut output, 1e6);
        for &v in &output {
            assert!((v - 0.25).abs() < 0.01, "very high temp: {v}");
        }
    }

    #[test]
    #[should_panic(expected = "temperature must be non-negative")]
    fn test_temperature_negative_panics() {
        let input = vec![1.0, 2.0];
        let mut output = vec![0.0; 2];
        softmax_temperature_f32(&input, &mut output, -1.0);
    }

    #[test]
    fn test_temperature_monotonic_sharpening() {
        let input = vec![1.0, 3.0, 2.0, 0.0];
        let mut out_high = vec![0.0; 4];
        let mut out_low = vec![0.0; 4];
        softmax_temperature_f32(&input, &mut out_high, 5.0);
        softmax_temperature_f32(&input, &mut out_low, 0.5);
        // Lower temp → higher max prob
        let max_high = out_high.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let max_low = out_low.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(max_low > max_high, "low temp should be sharper");
    }

    // ── 6. Grouped softmax tests ──────────────────────────────────────

    #[test]
    fn test_grouped_size_1() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        grouped_softmax_f32(&input, 1, &mut output);
        for &v in &output {
            assert!((v - 1.0).abs() < 1e-6, "group_size=1: {v}");
        }
    }

    #[test]
    fn test_grouped_full_length() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        let mut expected = vec![0.0; 4];
        grouped_softmax_f32(&input, 4, &mut output);
        softmax_f32(&input, &mut expected);
        assert_close(&output, &expected, APPROX_TOL, "grouped_full");
    }

    #[test]
    fn test_grouped_two_groups() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let mut output = vec![0.0; 8];
        grouped_softmax_f32(&input, 4, &mut output);

        // Each group should sum to 1.
        let sum1: f32 = output[0..4].iter().sum();
        let sum2: f32 = output[4..8].iter().sum();
        assert!((sum1 - 1.0).abs() < 1e-3, "group1 sum = {sum1}");
        assert!((sum2 - 1.0).abs() < 1e-3, "group2 sum = {sum2}");
    }

    #[test]
    fn test_grouped_prime_size() {
        let input: Vec<f32> = (0..15).map(|i| i as f32).collect();
        let mut output = vec![0.0; 15];
        grouped_softmax_f32(&input, 3, &mut output);
        for g in 0..5 {
            let sum: f32 = output[g * 3..(g + 1) * 3].iter().sum();
            assert!((sum - 1.0).abs() < 1e-3, "group {g} sum = {sum}");
        }
    }

    #[test]
    fn test_grouped_power_of_two() {
        let input: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
        let mut output = vec![0.0; 32];
        grouped_softmax_f32(&input, 8, &mut output);
        for g in 0..4 {
            let sum: f32 = output[g * 8..(g + 1) * 8].iter().sum();
            assert!((sum - 1.0).abs() < 1e-3, "group {g} sum = {sum}");
        }
    }

    #[test]
    fn test_grouped_groups_independent() {
        // Changing one group should not affect another.
        let input1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input2 = vec![1.0, 2.0, 3.0, 100.0, 200.0, 300.0];
        let mut out1 = vec![0.0; 6];
        let mut out2 = vec![0.0; 6];
        grouped_softmax_f32(&input1, 3, &mut out1);
        grouped_softmax_f32(&input2, 3, &mut out2);
        // First group should be identical.
        assert_close(&out1[0..3], &out2[0..3], 1e-6, "groups_independent");
    }

    #[test]
    #[should_panic(expected = "group_size must be > 0")]
    fn test_grouped_zero_size_panics() {
        let input = vec![1.0, 2.0];
        let mut output = vec![0.0; 2];
        grouped_softmax_f32(&input, 0, &mut output);
    }

    #[test]
    #[should_panic(expected = "not divisible")]
    fn test_grouped_non_divisible_panics() {
        let input = vec![1.0, 2.0, 3.0];
        let mut output = vec![0.0; 3];
        grouped_softmax_f32(&input, 2, &mut output);
    }

    #[test]
    fn test_grouped_size_5() {
        let input: Vec<f32> = (0..25).map(|i| (i as f32) - 12.0).collect();
        let mut output = vec![0.0; 25];
        grouped_softmax_f32(&input, 5, &mut output);
        for g in 0..5 {
            let sum: f32 = output[g * 5..(g + 1) * 5].iter().sum();
            assert!((sum - 1.0).abs() < 1e-3, "group {g} sum = {sum}");
        }
    }

    // ── Cross-function & property tests ────────────────────────────────

    #[test]
    fn test_softmax_all_outputs_non_negative() {
        let input = vec![-5.0, -3.0, 0.0, 2.0, 7.0, -10.0, 4.0, 1.0];
        let mut output = vec![0.0; 8];
        softmax_f32(&input, &mut output);
        for (i, &v) in output.iter().enumerate() {
            assert!(v >= 0.0, "softmax[{i}] = {v} should be >= 0");
        }
    }

    #[test]
    fn test_log_softmax_consistency_with_online() {
        let input = vec![2.0, 4.0, 1.0, 3.0, 5.0, 0.5, -1.0, 2.5];
        let mut online = vec![0.0; 8];
        let mut standard = vec![0.0; 8];
        online_softmax_f32(&input, &mut online);
        softmax_f32(&input, &mut standard);
        assert_close(&online, &standard, APPROX_TOL, "online_vs_standard_8");
    }

    #[test]
    fn test_softmax_preserves_ordering() {
        let input = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let mut output = vec![0.0; 5];
        softmax_f32(&input, &mut output);
        // Softmax preserves relative ordering of inputs.
        assert!(output[3] > output[4]); // 5 > 4
        assert!(output[4] > output[1]); // 4 > 3
        assert!(output[1] > output[2]); // 3 > 2
        assert!(output[2] > output[0]); // 2 > 1
    }

    #[test]
    fn test_online_softmax_preserves_ordering() {
        let input = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let mut output = vec![0.0; 5];
        online_softmax_f32(&input, &mut output);
        assert!(output[3] > output[4]);
        assert!(output[4] > output[1]);
    }

    #[test]
    fn test_softmax_idempotent_structure() {
        // Applying softmax twice should still sum to 1.
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut first = vec![0.0; 4];
        let mut second = vec![0.0; 4];
        softmax_f32(&input, &mut first);
        softmax_f32(&first, &mut second);
        let sum: f32 = second.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_all_functions_handle_3_elements() {
        let input = vec![1.0, 2.0, 3.0];
        let mask = vec![false, false, false];
        let mut out = vec![0.0; 3];

        softmax_f32(&input, &mut out);
        assert!((out.iter().sum::<f32>() - 1.0).abs() < 1e-3);

        log_softmax_f32(&input, &mut out);
        assert!(out.iter().all(|&v| v <= 1e-5));

        online_softmax_f32(&input, &mut out);
        assert!((out.iter().sum::<f32>() - 1.0).abs() < 1e-3);

        fused_softmax_mask_f32(&input, &mask, &mut out, f32::NEG_INFINITY);
        assert!((out.iter().sum::<f32>() - 1.0).abs() < 1e-3);

        softmax_temperature_f32(&input, &mut out, 1.0);
        assert!((out.iter().sum::<f32>() - 1.0).abs() < 1e-3);

        grouped_softmax_f32(&input, 3, &mut out);
        assert!((out.iter().sum::<f32>() - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_all_functions_handle_16_elements() {
        let input: Vec<f32> = (0..16).map(|i| i as f32 * 0.5 - 4.0).collect();
        let mask = vec![false; 16];
        let mut out = vec![0.0; 16];

        softmax_f32(&input, &mut out);
        assert!((out.iter().sum::<f32>() - 1.0).abs() < 1e-3);

        log_softmax_f32(&input, &mut out);
        assert!(out.iter().all(|&v| v <= 1e-5));

        online_softmax_f32(&input, &mut out);
        assert!((out.iter().sum::<f32>() - 1.0).abs() < 1e-3);

        fused_softmax_mask_f32(&input, &mask, &mut out, f32::NEG_INFINITY);
        assert!((out.iter().sum::<f32>() - 1.0).abs() < 1e-3);

        softmax_temperature_f32(&input, &mut out, 0.5);
        assert!((out.iter().sum::<f32>() - 1.0).abs() < 1e-3);

        grouped_softmax_f32(&input, 4, &mut out);
        for g in 0..4 {
            let s: f32 = out[g * 4..(g + 1) * 4].iter().sum();
            assert!((s - 1.0).abs() < 1e-3);
        }
    }

    #[test]
    fn test_scalar_softmax_matches_dispatcher() {
        let input = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let mut dispatched = vec![0.0; 5];
        let mut scalar = vec![0.0; 5];
        softmax_f32(&input, &mut dispatched);
        scalar_softmax_f32(&input, &mut scalar);
        assert_close(&dispatched, &scalar, APPROX_TOL, "scalar_matches_dispatch");
    }

    #[test]
    fn test_scalar_log_softmax_matches_dispatcher() {
        let input = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let mut dispatched = vec![0.0; 5];
        let mut scalar = vec![0.0; 5];
        log_softmax_f32(&input, &mut dispatched);
        scalar_log_softmax_f32(&input, &mut scalar);
        assert_close(
            &dispatched,
            &scalar,
            APPROX_TOL,
            "scalar_log_matches_dispatch",
        );
    }

    #[test]
    fn test_scalar_online_softmax_matches_dispatcher() {
        let input = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let mut dispatched = vec![0.0; 5];
        let mut scalar = vec![0.0; 5];
        online_softmax_f32(&input, &mut dispatched);
        scalar_online_softmax_f32(&input, &mut scalar);
        assert_close(
            &dispatched,
            &scalar,
            APPROX_TOL,
            "scalar_online_matches_dispatch",
        );
    }

    #[test]
    fn test_fused_mask_sums_to_one_with_partial_mask() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mask = vec![true, false, true, false, true, false, true, false];
        let mut output = vec![0.0; 8];
        fused_softmax_mask_f32(&input, &mask, &mut output, -1e9);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-2, "partial mask sum = {sum}");
    }

    #[test]
    fn test_temperature_with_non_aligned_length() {
        let input: Vec<f32> = (0..7).map(|i| i as f32).collect();
        let mut output = vec![0.0; 7];
        softmax_temperature_f32(&input, &mut output, 2.0);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3, "non-aligned temp sum = {sum}");
    }

    #[test]
    fn test_grouped_large_group_count() {
        let input: Vec<f32> = (0..100).map(|i| (i as f32) * 0.1 - 5.0).collect();
        let mut output = vec![0.0; 100];
        grouped_softmax_f32(&input, 10, &mut output);
        for g in 0..10 {
            let sum: f32 = output[g * 10..(g + 1) * 10].iter().sum();
            assert!((sum - 1.0).abs() < 1e-3, "group {g} sum = {sum}");
        }
    }

    #[test]
    fn test_softmax_all_zeros() {
        let input = vec![0.0; 8];
        let mut output = vec![0.0; 8];
        softmax_f32(&input, &mut output);
        for &v in &output {
            assert!((v - 0.125).abs() < 1e-5);
        }
    }

    #[test]
    fn test_online_softmax_all_same() {
        let input = vec![5.5; 12];
        let mut output = vec![0.0; 12];
        online_softmax_f32(&input, &mut output);
        let expected = 1.0 / 12.0;
        for &v in &output {
            assert!((v - expected).abs() < 1e-4);
        }
    }

    #[test]
    fn test_fused_mask_with_large_neg_inf() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![true, false, false, true];
        let mut output = vec![0.0; 4];
        fused_softmax_mask_f32(&input, &mask, &mut output, -1e38);
        // Masked positions negligible
        assert!(output[0] < 1e-6);
        assert!(output[3] < 1e-6);
        // Unmasked sum to ~1
        let s = output[1] + output[2];
        assert!((s - 1.0).abs() < 1e-3);
    }
}

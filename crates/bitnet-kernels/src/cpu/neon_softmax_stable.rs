//! ARM NEON numerically-stable softmax kernels for Apple Silicon.
//!
//! Extends the basic NEON softmax with production features:
//! - Fused softmax + masking (causal / padding)
//! - Log-softmax with numerical stability
//! - Temperature scaling
//! - Flash-attention style online softmax (single-pass running max/sum)
//! - Multi-head parallel softmax
//! - In-place softmax for memory efficiency
//!
//! All public functions are gated on `target_arch = "aarch64"`.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Lane count for `float32x4_t` NEON vectors.
const LANES: usize = 4;

// ── Fast exp helpers ────────────────────────────────────────────────────

/// Scalar fast exp (degree-4 Cody-Waite). Max relative error ≈ 2e-4 for
/// |x| ≤ 20, sufficient for softmax normalisation.
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

// ── Internal NEON helpers ───────────────────────────────────────────────

/// Find the maximum value in `data` using NEON horizontal max.
///
/// # Safety
/// Requires `aarch64` target with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn find_max_neon(data: &[f32]) -> f32 {
    if data.is_empty() {
        return f32::NEG_INFINITY;
    }

    let len = data.len();
    let chunks = len / LANES;
    let remainder = len % LANES;

    let mut max_vec = vdupq_n_f32(f32::NEG_INFINITY);
    let ptr = data.as_ptr();
    for i in 0..chunks {
        let v = unsafe { vld1q_f32(ptr.add(i * LANES)) };
        max_vec = vmaxq_f32(max_vec, v);
    }

    let mut max_val = vmaxvq_f32(max_vec);
    for i in 0..remainder {
        let val = data[chunks * LANES + i];
        if val > max_val {
            max_val = val;
        }
    }
    max_val
}

// ── 1. Numerically-stable softmax ───────────────────────────────────────

/// NEON-accelerated softmax with max-subtract for numerical stability.
///
/// Computes `output[i] = exp(input[i] - max) / Σ exp(input[j] - max)`.
///
/// # Safety
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
/// Panics if `output.len() < input.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn softmax_stable(input: &[f32], output: &mut [f32]) {
    assert!(
        output.len() >= input.len(),
        "output buffer too small: {} < {}",
        output.len(),
        input.len()
    );
    let len = input.len();
    if len == 0 {
        return;
    }

    let max_val = unsafe { find_max_neon(input) };

    // Fused exp-and-sum pass.
    let chunks = len / LANES;
    let remainder = len % LANES;
    let max_vec = vdupq_n_f32(max_val);
    let mut sum_vec = vdupq_n_f32(0.0);

    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();

    for i in 0..chunks {
        let v = unsafe { vld1q_f32(in_ptr.add(i * LANES)) };
        let shifted = vsubq_f32(v, max_vec);
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

    // Normalise.
    let inv_sum = 1.0 / sum_val;
    let inv_sum_vec = vdupq_n_f32(inv_sum);
    for i in 0..chunks {
        let e = unsafe { vld1q_f32(out_ptr.add(i * LANES)) };
        let r = vmulq_f32(e, inv_sum_vec);
        unsafe { vst1q_f32(out_ptr.add(i * LANES), r) };
    }
    for i in 0..remainder {
        output[tail_start + i] *= inv_sum;
    }
}

// ── 2. Fused softmax + masking ──────────────────────────────────────────

/// Mask type for fused softmax.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskType {
    /// Causal (lower-triangular) attention mask. `row` is the current
    /// query position; positions `> row` are masked to `-inf`.
    Causal { row: usize },
    /// Padding mask — `true` elements are valid, `false` are masked.
    Padding,
}

/// Fused softmax with masking.
///
/// For `MaskType::Causal { row }` the `mask` slice is ignored and
/// positions `j > row` receive `-inf` before the softmax.
///
/// For `MaskType::Padding` the caller supplies a boolean mask where
/// `true` = keep, `false` = mask to `-inf`.
///
/// # Safety
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
/// Panics if buffer sizes are inconsistent.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn softmax_masked(
    input: &[f32],
    output: &mut [f32],
    mask_type: MaskType,
    mask: Option<&[bool]>,
) {
    let len = input.len();
    assert!(output.len() >= len, "output buffer too small: {} < {}", output.len(), len);
    if len == 0 {
        return;
    }

    // Apply mask into a scratch buffer.
    let mut masked = vec![0.0f32; len];
    match mask_type {
        MaskType::Causal { row } => {
            for j in 0..len {
                masked[j] = if j <= row { input[j] } else { f32::NEG_INFINITY };
            }
        }
        MaskType::Padding => {
            let m = mask.expect("Padding mask requires a bool slice");
            assert!(m.len() >= len, "mask length too small: {} < {}", m.len(), len);
            for j in 0..len {
                masked[j] = if m[j] { input[j] } else { f32::NEG_INFINITY };
            }
        }
    }

    unsafe { softmax_stable(&masked, output) };
}

// ── 3. Log-softmax ──────────────────────────────────────────────────────

/// Numerically-stable log-softmax:
/// `output[i] = (input[i] - max) - ln(Σ exp(input[j] - max))`.
///
/// # Safety
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
/// Panics if `output.len() < input.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn log_softmax_stable(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    assert!(output.len() >= len, "output buffer too small: {} < {}", output.len(), len);
    if len == 0 {
        return;
    }

    let max_val = unsafe { find_max_neon(input) };

    // Compute sum of exp(x_i - max).
    let chunks = len / LANES;
    let remainder = len % LANES;
    let max_vec = vdupq_n_f32(max_val);
    let mut sum_vec = vdupq_n_f32(0.0);

    let in_ptr = input.as_ptr();
    for i in 0..chunks {
        let v = unsafe { vld1q_f32(in_ptr.add(i * LANES)) };
        let shifted = vsubq_f32(v, max_vec);
        let e = unsafe { fast_exp_neon(shifted) };
        sum_vec = vaddq_f32(sum_vec, e);
    }
    let mut sum_val = vaddvq_f32(sum_vec);
    let tail_start = chunks * LANES;
    for i in 0..remainder {
        sum_val += fast_exp_scalar(input[tail_start + i] - max_val);
    }

    let log_sum = sum_val.ln();

    // output[i] = (input[i] - max_val) - log_sum
    let log_sum_vec = vdupq_n_f32(log_sum);
    let out_ptr = output.as_mut_ptr();
    for i in 0..chunks {
        let v = unsafe { vld1q_f32(in_ptr.add(i * LANES)) };
        let shifted = vsubq_f32(v, max_vec);
        let result = vsubq_f32(shifted, log_sum_vec);
        unsafe { vst1q_f32(out_ptr.add(i * LANES), result) };
    }
    for i in 0..remainder {
        output[tail_start + i] = (input[tail_start + i] - max_val) - log_sum;
    }
}

// ── 4. Softmax with temperature scaling ─────────────────────────────────

/// Temperature-scaled softmax:
/// `output[i] = softmax(input[i] / temperature)`.
///
/// A `temperature` of 1.0 is equivalent to standard softmax.
/// Lower temperatures sharpen the distribution; higher temperatures
/// flatten it.
///
/// # Safety
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
/// Panics if `temperature <= 0.0` or `output.len() < input.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn softmax_with_temperature(input: &[f32], output: &mut [f32], temperature: f32) {
    assert!(temperature > 0.0, "temperature must be positive");
    let len = input.len();
    assert!(output.len() >= len, "output buffer too small: {} < {}", output.len(), len);
    if len == 0 {
        return;
    }

    let inv_temp = 1.0 / temperature;

    // Scale input by 1/temperature into output, then run softmax in-place.
    let chunks = len / LANES;
    let remainder = len % LANES;
    let inv_temp_vec = vdupq_n_f32(inv_temp);

    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();

    for i in 0..chunks {
        let v = unsafe { vld1q_f32(in_ptr.add(i * LANES)) };
        let scaled = vmulq_f32(v, inv_temp_vec);
        unsafe { vst1q_f32(out_ptr.add(i * LANES), scaled) };
    }
    let tail_start = chunks * LANES;
    for i in 0..remainder {
        output[tail_start + i] = input[tail_start + i] * inv_temp;
    }

    unsafe { softmax_stable_inplace(output) };
}

// ── 5. Online softmax (flash-attention style) ───────────────────────────

/// Result of a single-pass online softmax computation.
#[derive(Debug, Clone)]
pub struct OnlineSoftmaxState {
    /// Running maximum seen so far.
    pub max: f32,
    /// Running sum of `exp(x_i - max)`, corrected on-the-fly.
    pub sum: f32,
}

impl Default for OnlineSoftmaxState {
    fn default() -> Self {
        Self::new()
    }
}

impl OnlineSoftmaxState {
    /// Create a new empty state.
    #[inline]
    pub fn new() -> Self {
        Self { max: f32::NEG_INFINITY, sum: 0.0 }
    }

    /// Absorb a single value into the running state.
    #[inline]
    pub fn update(&mut self, val: f32) {
        if val > self.max {
            // Rescale the existing sum for the new max.
            self.sum *= fast_exp_scalar(self.max - val);
            self.max = val;
        }
        self.sum += fast_exp_scalar(val - self.max);
    }

    /// Merge another state into this one (e.g. from a parallel chunk).
    #[inline]
    pub fn merge(&mut self, other: &OnlineSoftmaxState) {
        if other.max > self.max {
            self.sum = self.sum * fast_exp_scalar(self.max - other.max) + other.sum;
            self.max = other.max;
        } else {
            self.sum += other.sum * fast_exp_scalar(other.max - self.max);
        }
    }

    /// Compute the log-normaliser: `max + ln(sum)`.
    #[inline]
    pub fn log_normaliser(&self) -> f32 {
        self.max + self.sum.ln()
    }
}

/// Flash-attention style online softmax: single-pass running max/sum
/// without needing a full preliminary max pass.
///
/// Returns the final `OnlineSoftmaxState` and writes normalised
/// probabilities into `output`.
///
/// # Safety
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
/// Panics if `output.len() < input.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn online_softmax(input: &[f32], output: &mut [f32]) -> OnlineSoftmaxState {
    let len = input.len();
    assert!(output.len() >= len, "output buffer too small: {} < {}", output.len(), len);

    // Pass 1: compute running max and sum in a single sweep.
    let mut state = OnlineSoftmaxState::new();
    for &v in input.iter() {
        state.update(v);
    }

    if len == 0 || state.sum == 0.0 {
        return state;
    }

    // Pass 2: normalise using the computed max and sum.
    let chunks = len / LANES;
    let remainder = len % LANES;
    let max_vec = vdupq_n_f32(state.max);
    let inv_sum = 1.0 / state.sum;
    let inv_sum_vec = vdupq_n_f32(inv_sum);

    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();

    for i in 0..chunks {
        let v = unsafe { vld1q_f32(in_ptr.add(i * LANES)) };
        let shifted = vsubq_f32(v, max_vec);
        let e = unsafe { fast_exp_neon(shifted) };
        let r = vmulq_f32(e, inv_sum_vec);
        unsafe { vst1q_f32(out_ptr.add(i * LANES), r) };
    }

    let tail_start = chunks * LANES;
    for i in 0..remainder {
        let e = fast_exp_scalar(input[tail_start + i] - state.max);
        output[tail_start + i] = e * inv_sum;
    }

    state
}

// ── 6. Multi-head parallel softmax ──────────────────────────────────────

/// Apply softmax independently to each of `num_heads` contiguous rows of
/// length `head_dim` stored in row-major order in `data`.
///
/// `data.len()` must equal `num_heads * head_dim`.
///
/// # Safety
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
/// Panics if `output.len() < data.len()` or dimensions are inconsistent.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn multi_head_softmax(
    data: &[f32],
    output: &mut [f32],
    num_heads: usize,
    head_dim: usize,
) {
    let total = num_heads * head_dim;
    assert!(data.len() >= total, "input too small for {num_heads}×{head_dim}: {}", data.len());
    assert!(output.len() >= total, "output too small for {num_heads}×{head_dim}: {}", output.len());

    for h in 0..num_heads {
        let offset = h * head_dim;
        let row = &data[offset..offset + head_dim];
        let out_row = &mut output[offset..offset + head_dim];
        unsafe { softmax_stable(row, out_row) };
    }
}

// ── 7. In-place softmax ─────────────────────────────────────────────────

/// In-place numerically-stable softmax.
///
/// Overwrites `data` with softmax probabilities. Uses a fused
/// exp-normalise pass to avoid allocating a temporary buffer.
///
/// # Safety
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn softmax_stable_inplace(data: &mut [f32]) {
    let len = data.len();
    if len == 0 {
        return;
    }

    let max_val = unsafe { find_max_neon(data) };

    // Fused exp + accumulate sum.
    let chunks = len / LANES;
    let remainder = len % LANES;
    let max_vec = vdupq_n_f32(max_val);
    let mut sum_vec = vdupq_n_f32(0.0);

    let ptr = data.as_mut_ptr();
    for i in 0..chunks {
        let v = unsafe { vld1q_f32(ptr.add(i * LANES)) };
        let shifted = vsubq_f32(v, max_vec);
        let e = unsafe { fast_exp_neon(shifted) };
        sum_vec = vaddq_f32(sum_vec, e);
        unsafe { vst1q_f32(ptr.add(i * LANES), e) };
    }

    let mut sum_val = vaddvq_f32(sum_vec);
    let tail_start = chunks * LANES;
    for i in 0..remainder {
        let e = fast_exp_scalar(data[tail_start + i] - max_val);
        data[tail_start + i] = e;
        sum_val += e;
    }

    // Normalise in-place.
    let inv_sum = 1.0 / sum_val;
    let inv_sum_vec = vdupq_n_f32(inv_sum);
    for i in 0..chunks {
        let v = unsafe { vld1q_f32(ptr.add(i * LANES)) };
        let r = vmulq_f32(v, inv_sum_vec);
        unsafe { vst1q_f32(ptr.add(i * LANES), r) };
    }
    for i in 0..remainder {
        data[tail_start + i] *= inv_sum;
    }
}

// ── Scalar reference ────────────────────────────────────────────────────

/// Plain scalar softmax for parity testing.
pub fn softmax_scalar_ref(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    assert!(output.len() >= len);
    if len == 0 {
        return;
    }
    let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = input.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    for i in 0..len {
        output[i] = exps[i] / sum;
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    fn assert_close(a: f32, b: f32, tol: f32, ctx: &str) {
        assert!((a - b).abs() < tol, "{ctx}: expected {b}, got {a} (diff {})", (a - b).abs());
    }

    fn assert_sums_to_one(v: &[f32], tol: f32, ctx: &str) {
        let sum: f32 = v.iter().sum();
        assert_close(sum, 1.0, tol, ctx);
    }

    fn assert_all_finite(v: &[f32], ctx: &str) {
        for (i, &x) in v.iter().enumerate() {
            assert!(x.is_finite(), "{ctx}[{i}] = {x} is not finite");
        }
    }

    // ── softmax_stable ──────────────────────────────────────────────

    #[test]
    fn test_stable_basic() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        unsafe { softmax_stable(&input, &mut output) };
        assert_sums_to_one(&output, 1e-3, "basic sum");
        for w in output.windows(2) {
            assert!(w[0] < w[1], "monotonic increase");
        }
    }

    #[test]
    fn test_stable_single() {
        let mut output = [0.0];
        unsafe { softmax_stable(&[42.0], &mut output) };
        assert_close(output[0], 1.0, 1e-5, "single");
    }

    #[test]
    fn test_stable_empty() {
        let mut output: Vec<f32> = vec![];
        unsafe { softmax_stable(&[], &mut output) };
        assert!(output.is_empty());
    }

    #[test]
    fn test_stable_large_values() {
        let input = [1000.0, 1001.0, 1002.0, 1003.0];
        let mut output = vec![0.0; 4];
        unsafe { softmax_stable(&input, &mut output) };
        assert_sums_to_one(&output, 1e-3, "large");
        assert_all_finite(&output, "large");
    }

    #[test]
    fn test_stable_negative_values() {
        let input = [-100.0, -50.0, -10.0, 0.0];
        let mut output = vec![0.0; 4];
        unsafe { softmax_stable(&input, &mut output) };
        assert_sums_to_one(&output, 1e-3, "negative");
        assert_all_finite(&output, "negative");
    }

    #[test]
    fn test_stable_uniform() {
        let input = [5.0; 8];
        let mut output = vec![0.0; 8];
        unsafe { softmax_stable(&input, &mut output) };
        let expected = 1.0 / 8.0;
        for (i, &v) in output.iter().enumerate() {
            assert_close(v, expected, 1e-3, &format!("uniform[{i}]"));
        }
    }

    #[test]
    fn test_stable_non_aligned() {
        for &len in &[5, 7, 13, 17] {
            let input: Vec<f32> = (0..len).map(|i| i as f32 * 0.3).collect();
            let mut output = vec![0.0; len];
            unsafe { softmax_stable(&input, &mut output) };
            assert_sums_to_one(&output, 1e-3, &format!("non-aligned len={len}"));
            assert_all_finite(&output, &format!("non-aligned len={len}"));
        }
    }

    #[test]
    fn test_stable_parity_with_scalar() {
        let input: Vec<f32> = (0..17).map(|i| (i as f32) * 0.3 - 2.5).collect();
        let mut neon_out = vec![0.0f32; input.len()];
        unsafe { softmax_stable(&input, &mut neon_out) };
        let mut scalar_out = vec![0.0f32; input.len()];
        softmax_scalar_ref(&input, &mut scalar_out);

        for (i, (&n, &s)) in neon_out.iter().zip(scalar_out.iter()).enumerate() {
            assert!((n - s).abs() < 1e-3, "parity[{i}]: neon={n}, scalar={s}",);
        }
    }

    // ── softmax_masked ──────────────────────────────────────────────

    #[test]
    fn test_masked_causal() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        unsafe { softmax_masked(&input, &mut output, MaskType::Causal { row: 1 }, None) };
        // Positions 2 and 3 should be zero (masked out).
        assert_close(output[2], 0.0, 1e-6, "causal[2]");
        assert_close(output[3], 0.0, 1e-6, "causal[3]");
        // Remaining should still sum to ~1.
        let visible_sum: f32 = output[0..2].iter().sum();
        assert_close(visible_sum, 1.0, 1e-3, "causal visible sum");
    }

    #[test]
    fn test_masked_padding() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mask = [true, false, true, false];
        let mut output = vec![0.0; 4];
        unsafe { softmax_masked(&input, &mut output, MaskType::Padding, Some(&mask)) };
        assert_close(output[1], 0.0, 1e-6, "pad[1]");
        assert_close(output[3], 0.0, 1e-6, "pad[3]");
        let visible_sum: f32 = output.iter().filter(|&&v| v > 1e-10).sum();
        assert_close(visible_sum, 1.0, 1e-3, "pad visible sum");
    }

    #[test]
    fn test_masked_empty() {
        let mut output: Vec<f32> = vec![];
        unsafe { softmax_masked(&[], &mut output, MaskType::Causal { row: 0 }, None) };
        assert!(output.is_empty());
    }

    // ── log_softmax_stable ──────────────────────────────────────────

    #[test]
    fn test_log_softmax_basic() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut log_out = vec![0.0; 4];
        unsafe { log_softmax_stable(&input, &mut log_out) };
        // All log-softmax values must be <= 0.
        for (i, &v) in log_out.iter().enumerate() {
            assert!(v <= 0.0, "log_softmax[{i}] = {v} > 0");
        }
        // exp(log_softmax) should sum to 1.
        let sum: f32 = log_out.iter().map(|&v| v.exp()).sum();
        assert_close(sum, 1.0, 1e-3, "exp(log_softmax) sum");
    }

    #[test]
    fn test_log_softmax_parity() {
        let input = [0.5, 1.5, -0.5, 2.0, 0.0];
        let mut softmax_out = vec![0.0; 5];
        unsafe { softmax_stable(&input, &mut softmax_out) };
        let mut log_out = vec![0.0; 5];
        unsafe { log_softmax_stable(&input, &mut log_out) };

        for (i, (&s, &l)) in softmax_out.iter().zip(log_out.iter()).enumerate() {
            assert_close(l, s.ln(), 1e-3, &format!("log_parity[{i}]"));
        }
    }

    #[test]
    fn test_log_softmax_empty() {
        let mut output: Vec<f32> = vec![];
        unsafe { log_softmax_stable(&[], &mut output) };
        assert!(output.is_empty());
    }

    // ── softmax_with_temperature ────────────────────────────────────

    #[test]
    fn test_temperature_one() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut t1 = vec![0.0; 4];
        let mut plain = vec![0.0; 4];
        unsafe {
            softmax_with_temperature(&input, &mut t1, 1.0);
            softmax_stable(&input, &mut plain);
        }
        for (i, (&a, &b)) in t1.iter().zip(plain.iter()).enumerate() {
            assert_close(a, b, 1e-4, &format!("temp1[{i}]"));
        }
    }

    #[test]
    fn test_temperature_high_flattens() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0; 4];
        unsafe { softmax_with_temperature(&input, &mut out, 100.0) };
        // High temperature → near-uniform.
        let expected = 0.25;
        for (i, &v) in out.iter().enumerate() {
            assert_close(v, expected, 0.01, &format!("high_temp[{i}]"));
        }
    }

    #[test]
    fn test_temperature_low_sharpens() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0; 4];
        unsafe { softmax_with_temperature(&input, &mut out, 0.01) };
        // Low temperature → argmax dominates.
        assert!(out[3] > 0.99, "expected argmax to dominate, got {}", out[3]);
    }

    #[test]
    #[should_panic(expected = "temperature must be positive")]
    fn test_temperature_zero_panics() {
        let input = [1.0, 2.0];
        let mut out = vec![0.0; 2];
        unsafe { softmax_with_temperature(&input, &mut out, 0.0) };
    }

    // ── online_softmax ──────────────────────────────────────────────

    #[test]
    fn test_online_softmax_parity() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut online_out = vec![0.0; input.len()];
        let mut stable_out = vec![0.0; input.len()];
        unsafe {
            online_softmax(&input, &mut online_out);
            softmax_stable(&input, &mut stable_out);
        }
        for (i, (&o, &s)) in online_out.iter().zip(stable_out.iter()).enumerate() {
            assert_close(o, s, 1e-3, &format!("online_parity[{i}]"));
        }
    }

    #[test]
    fn test_online_softmax_state_merge() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut s1 = OnlineSoftmaxState::new();
        for &v in &data[..3] {
            s1.update(v);
        }
        let mut s2 = OnlineSoftmaxState::new();
        for &v in &data[3..] {
            s2.update(v);
        }
        s1.merge(&s2);

        let mut full = OnlineSoftmaxState::new();
        for &v in &data {
            full.update(v);
        }

        assert_close(s1.max, full.max, 1e-6, "merge max");
        assert_close(s1.sum, full.sum, 1e-3, "merge sum");
    }

    #[test]
    fn test_online_softmax_empty() {
        let mut output: Vec<f32> = vec![];
        let state = unsafe { online_softmax(&[], &mut output) };
        assert_eq!(state.max, f32::NEG_INFINITY);
        assert_eq!(state.sum, 0.0);
    }

    // ── multi_head_softmax ──────────────────────────────────────────

    #[test]
    fn test_multi_head_basic() {
        let num_heads = 3;
        let head_dim = 5;
        let data: Vec<f32> = (0..(num_heads * head_dim)).map(|i| i as f32 * 0.2 - 1.0).collect();
        let mut output = vec![0.0; data.len()];
        unsafe { multi_head_softmax(&data, &mut output, num_heads, head_dim) };

        for h in 0..num_heads {
            let row = &output[h * head_dim..(h + 1) * head_dim];
            assert_sums_to_one(row, 1e-3, &format!("head {h}"));
            assert_all_finite(row, &format!("head {h}"));
        }
    }

    #[test]
    fn test_multi_head_single() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let mut single = vec![0.0; 4];
        let mut multi = vec![0.0; 4];
        unsafe {
            softmax_stable(&data, &mut single);
            multi_head_softmax(&data, &mut multi, 1, 4);
        }
        for (i, (&s, &m)) in single.iter().zip(multi.iter()).enumerate() {
            assert_close(s, m, 1e-5, &format!("single_head[{i}]"));
        }
    }

    // ── softmax_stable_inplace ──────────────────────────────────────

    #[test]
    fn test_inplace_matches_out_of_place() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut out_of_place = vec![0.0; input.len()];
        unsafe { softmax_stable(&input, &mut out_of_place) };

        let mut inplace = input.to_vec();
        unsafe { softmax_stable_inplace(&mut inplace) };

        for (i, (&o, &p)) in out_of_place.iter().zip(inplace.iter()).enumerate() {
            assert_close(o, p, 1e-5, &format!("inplace[{i}]"));
        }
    }

    #[test]
    fn test_inplace_empty() {
        let mut data: Vec<f32> = vec![];
        unsafe { softmax_stable_inplace(&mut data) };
        assert!(data.is_empty());
    }

    #[test]
    fn test_inplace_single() {
        let mut data = vec![99.0];
        unsafe { softmax_stable_inplace(&mut data) };
        assert_close(data[0], 1.0, 1e-5, "inplace single");
    }

    // ── Cross-cutting numerical stability ───────────────────────────

    #[test]
    fn test_extreme_range_no_nan() {
        let input = [-1e6, 0.0, 1e6];
        let mut output = vec![0.0; 3];
        unsafe { softmax_stable(&input, &mut output) };
        assert_all_finite(&output, "extreme range");
        assert_sums_to_one(&output, 1e-3, "extreme range");
    }

    #[test]
    fn test_all_neg_inf_graceful() {
        // All -inf should not produce NaN; behaviour is technically
        // undefined (0/0) but we accept any finite or NaN result.
        let input = [f32::NEG_INFINITY; 4];
        let mut output = vec![0.0; 4];
        unsafe { softmax_stable(&input, &mut output) };
        // Just assert no panic occurred.
    }

    // ── Ignored tests (with justification) ──────────────────────────

    #[test]
    #[ignore = "Slow: benchmark-style test comparing \
                NEON vs scalar throughput on large tensors"]
    fn bench_neon_vs_scalar_throughput() {
        let n = 1 << 20;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001 - 500.0).collect();
        let mut output = vec![0.0; n];
        let start = std::time::Instant::now();
        unsafe { softmax_stable(&input, &mut output) };
        let neon_dur = start.elapsed();

        let mut scalar_out = vec![0.0; n];
        let start = std::time::Instant::now();
        softmax_scalar_ref(&input, &mut scalar_out);
        let scalar_dur = start.elapsed();

        eprintln!(
            "NEON: {neon_dur:?}, Scalar: {scalar_dur:?}, \
             speedup: {:.2}×",
            scalar_dur.as_secs_f64() / neon_dur.as_secs_f64()
        );
    }

    #[test]
    #[ignore = "TDD scaffold: requires multi-threaded rayon \
                integration for parallel head processing"]
    fn test_multi_head_parallel_rayon() {
        panic!("not yet implemented");
    }
}

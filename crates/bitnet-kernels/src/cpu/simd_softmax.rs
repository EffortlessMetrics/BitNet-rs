//! SIMD-optimized softmax variants with temperature scaling, masking, and diagnostics.
//!
//! Builds on the core softmax module to provide:
//! - Numerically stable softmax with explicit SIMD vectorization
//! - Online (streaming) softmax for single-pass computation
//! - Temperature-scaled softmax with argmax fallback
//! - Top-k filtered softmax
//! - Masked and causal attention softmax
//! - Multi-head batched softmax
//! - Log-softmax (out-of-place and in-place)
//! - Diagnostic softmax with entropy and range reporting
//!
//! On x86-64 with AVX2, the hot loops are vectorized 8-wide; a scalar
//! fallback handles all other architectures and tail elements.
#![allow(unsafe_op_in_unsafe_fn)]

use bitnet_common::{BitNetError, KernelError, Result};

// ── AVX2 helpers ────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[allow(clippy::wildcard_imports)]
use std::arch::x86_64::*;

/// 8-wide AVX2 horizontal max → scalar.
///
/// # Safety
/// Caller must ensure AVX2 is available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hmax_avx2(v: __m256) -> f32 {
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let m128 = _mm_max_ps(lo128, hi128);
    let m64 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
    let m32 = _mm_max_ss(m64, _mm_shuffle_ps(m64, m64, 1));
    _mm_cvtss_f32(m32)
}

/// 8-wide AVX2 horizontal sum → scalar.
///
/// # Safety
/// Caller must ensure AVX2 is available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_avx2(v: __m256) -> f32 {
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let s128 = _mm_add_ps(lo128, hi128);
    let s64 = _mm_add_ps(s128, _mm_movehl_ps(s128, s128));
    let s32 = _mm_add_ss(s64, _mm_shuffle_ps(s64, s64, 1));
    _mm_cvtss_f32(s32)
}

/// Fast scalar exp with clamping to avoid inf/NaN.
#[inline(always)]
fn fast_exp(x: f32) -> f32 {
    x.clamp(-88.0, 88.0).exp()
}

/// Vectorized exp(x) for 8×f32 using AVX2 (Cephes-style polynomial).
///
/// Uses the identity exp(x) = 2^(x * log2(e)) and splits into integer
/// and fractional parts.  The fractional part is approximated with a
/// degree-5 minimax polynomial (Cephes coefficients).
///
/// Accuracy: max relative error < 2e-7 over [-88, 88].
///
/// # Safety
/// Caller must ensure AVX2 + FMA are available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::excessive_precision)] // Cody-Waite constants need exact bit patterns
#[inline]
unsafe fn exp_avx2(x: __m256) -> __m256 {
    let lo = _mm256_set1_ps(-88.376_26_f32);
    let hi = _mm256_set1_ps(88.376_26_f32);
    let x = _mm256_min_ps(_mm256_max_ps(x, lo), hi);

    let log2e = _mm256_set1_ps(std::f32::consts::LOG2_E);
    let t = _mm256_mul_ps(x, log2e);
    let n = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    let ln2_hi = _mm256_set1_ps(0.693_145_751_953_125_f32);
    let ln2_lo = _mm256_set1_ps(1.428_606_765_330_187_e-6_f32);
    let f = _mm256_sub_ps(_mm256_sub_ps(x, _mm256_mul_ps(n, ln2_hi)), _mm256_mul_ps(n, ln2_lo));

    let c5 = _mm256_set1_ps(1.987_569_1e-4);
    let c4 = _mm256_set1_ps(1.398_199_9e-3);
    let c3 = _mm256_set1_ps(8.333_452e-3);
    let c2 = _mm256_set1_ps(4.166_579_6e-2);
    let c1 = _mm256_set1_ps(1.666_666_6e-1);
    let c0 = _mm256_set1_ps(5.000_000_2e-1);
    let one = _mm256_set1_ps(1.0);

    let mut p = _mm256_fmadd_ps(c5, f, c4);
    p = _mm256_fmadd_ps(p, f, c3);
    p = _mm256_fmadd_ps(p, f, c2);
    p = _mm256_fmadd_ps(p, f, c1);
    p = _mm256_fmadd_ps(p, f, c0);
    p = _mm256_fmadd_ps(p, _mm256_mul_ps(f, f), _mm256_add_ps(f, one));

    let ni = _mm256_cvtps_epi32(n);
    let pow2n =
        _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(ni, _mm256_set1_epi32(127)), 23));
    _mm256_mul_ps(p, pow2n)
}

// ── Core scalar implementation ──────────────────────────────────────────

/// Find max of a slice (scalar).
fn scalar_max(data: &[f32]) -> f32 {
    data.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

/// Numerically-stable softmax written to `output` (scalar path).
fn simd_softmax_scalar(input: &[f32], output: &mut [f32]) {
    if input.is_empty() {
        return;
    }
    let max_val = scalar_max(input);
    let mut sum = 0.0f32;
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        let e = fast_exp(x - max_val);
        *o = e;
        sum += e;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for o in output.iter_mut() {
            *o *= inv;
        }
    }
}

// ── AVX2 softmax core ──────────────────────────────────────────────────

/// Numerically-stable softmax using AVX2 intrinsics.
///
/// # Safety
/// Caller must ensure AVX2 + FMA are available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn simd_softmax_avx2(input: &[f32], output: &mut [f32]) {
    let n = input.len();
    if n == 0 {
        return;
    }

    // ── pass 1: find max ────────────────────────────────────────────────
    let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);
    let chunks = n / 8;
    let inp = input.as_ptr();

    for i in 0..chunks {
        let v = _mm256_loadu_ps(inp.add(i * 8));
        vmax = _mm256_max_ps(vmax, v);
    }
    let mut max_val = hmax_avx2(vmax);
    for i in (chunks * 8)..n {
        max_val = max_val.max(*inp.add(i));
    }

    // ── pass 2: exp(x - max) and accumulate sum ─────────────────────────
    let vmax_bc = _mm256_set1_ps(max_val);
    let mut vsum = _mm256_setzero_ps();
    let outp = output.as_mut_ptr();

    for i in 0..chunks {
        let v = _mm256_loadu_ps(inp.add(i * 8));
        let shifted = _mm256_sub_ps(v, vmax_bc);
        let exp_v = exp_avx2(shifted);
        _mm256_storeu_ps(outp.add(i * 8), exp_v);
        vsum = _mm256_add_ps(vsum, exp_v);
    }
    let mut sum_exp = hsum_avx2(vsum);
    for i in (chunks * 8)..n {
        let e = fast_exp(*inp.add(i) - max_val);
        *outp.add(i) = e;
        sum_exp += e;
    }

    // ── pass 3: normalize ───────────────────────────────────────────────
    if sum_exp > 0.0 {
        let inv = _mm256_set1_ps(1.0 / sum_exp);
        for i in 0..chunks {
            let v = _mm256_loadu_ps(outp.add(i * 8));
            _mm256_storeu_ps(outp.add(i * 8), _mm256_mul_ps(v, inv));
        }
        let inv_s = 1.0 / sum_exp;
        for i in (chunks * 8)..n {
            *outp.add(i) *= inv_s;
        }
    }
}

/// In-place numerically-stable softmax using AVX2 intrinsics.
///
/// # Safety
/// Caller must ensure AVX2 + FMA are available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn simd_softmax_avx2_inplace(data: &mut [f32]) {
    let n = data.len();
    if n == 0 {
        return;
    }

    let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);
    let chunks = n / 8;
    let ptr = data.as_mut_ptr();

    for i in 0..chunks {
        let v = _mm256_loadu_ps(ptr.add(i * 8));
        vmax = _mm256_max_ps(vmax, v);
    }
    let mut max_val = hmax_avx2(vmax);
    for i in (chunks * 8)..n {
        max_val = max_val.max(*ptr.add(i));
    }

    let vmax_bc = _mm256_set1_ps(max_val);
    let mut vsum = _mm256_setzero_ps();

    for i in 0..chunks {
        let v = _mm256_loadu_ps(ptr.add(i * 8));
        let shifted = _mm256_sub_ps(v, vmax_bc);
        let exp_v = exp_avx2(shifted);
        _mm256_storeu_ps(ptr.add(i * 8), exp_v);
        vsum = _mm256_add_ps(vsum, exp_v);
    }
    let mut sum_exp = hsum_avx2(vsum);
    for i in (chunks * 8)..n {
        let e = fast_exp(*ptr.add(i) - max_val);
        *ptr.add(i) = e;
        sum_exp += e;
    }

    if sum_exp > 0.0 {
        let inv = _mm256_set1_ps(1.0 / sum_exp);
        for i in 0..chunks {
            let v = _mm256_loadu_ps(ptr.add(i * 8));
            _mm256_storeu_ps(ptr.add(i * 8), _mm256_mul_ps(v, inv));
        }
        let inv_s = 1.0 / sum_exp;
        for i in (chunks * 8)..n {
            *ptr.add(i) *= inv_s;
        }
    }
}

/// Dispatch softmax to AVX2 or scalar.
fn dispatch_softmax(input: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: feature detection above guarantees AVX2 + FMA.
            unsafe { simd_softmax_avx2(input, output) };
            return;
        }
    }
    simd_softmax_scalar(input, output);
}

// ── Public API ──────────────────────────────────────────────────────────

// ── 1. Numerically stable softmax with SIMD ─────────────────────────────

/// Numerically-stable softmax over `input`, written to `output`.
///
/// Uses AVX2 when available on x86-64, otherwise falls back to scalar.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] when `input.len() != output.len()`.
pub fn simd_softmax(input: &[f32], output: &mut [f32]) -> Result<()> {
    if input.len() != output.len() {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("input/output length mismatch: {} vs {}", input.len(), output.len()),
        }));
    }
    dispatch_softmax(input, output);
    Ok(())
}

/// In-place numerically-stable SIMD softmax.
pub fn simd_softmax_inplace(data: &mut [f32]) -> Result<()> {
    if data.is_empty() {
        return Ok(());
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: feature detection above guarantees AVX2 + FMA.
            unsafe { simd_softmax_avx2_inplace(data) };
            return Ok(());
        }
    }
    let input: Vec<f32> = data.to_vec();
    simd_softmax_scalar(&input, data);
    Ok(())
}

// ── 2. Online softmax (Milakov & Gimelshein single-pass) ────────────────

/// Online (streaming) softmax — single-pass numerically-stable algorithm.
///
/// Maintains a running max and correction factor so the full output can be
/// produced in one scan without a separate max-finding pass.
///
/// Reference: Milakov & Gimelshein, "Online normalizer calculation for softmax", 2018.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] on length mismatch.
pub fn simd_softmax_online(input: &[f32], output: &mut [f32]) -> Result<()> {
    if input.len() != output.len() {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("input/output length mismatch: {} vs {}", input.len(), output.len()),
        }));
    }
    if input.is_empty() {
        return Ok(());
    }

    // Single-pass: track running max and denominator.
    let mut running_max = f32::NEG_INFINITY;
    let mut running_sum = 0.0f32;

    for &x in input {
        if x > running_max {
            running_sum *= fast_exp(running_max - x);
            running_max = x;
        }
        running_sum += fast_exp(x - running_max);
    }

    // Write normalized output.
    let log_sum = running_max + running_sum.ln();
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = fast_exp(x - log_sum);
    }
    Ok(())
}

// ── 3. Temperature-scaled softmax ───────────────────────────────────────

/// Temperature-scaled softmax: `softmax(x / temperature)`.
///
/// When `temperature` is very close to zero (`< 1e-7`), the output is a
/// one-hot vector at the argmax position. Very large temperatures produce
/// a near-uniform distribution.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] on length mismatch or
/// negative temperature.
pub fn simd_softmax_temperature(input: &[f32], output: &mut [f32], temperature: f32) -> Result<()> {
    if input.len() != output.len() {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("input/output length mismatch: {} vs {}", input.len(), output.len()),
        }));
    }
    if temperature < 0.0 {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("temperature must be non-negative, got {temperature}"),
        }));
    }
    if input.is_empty() {
        return Ok(());
    }

    // Near-zero temperature → one-hot at argmax.
    if temperature < 1e-7 {
        output.fill(0.0);
        let argmax = input
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        output[argmax] = 1.0;
        return Ok(());
    }

    // Scale then delegate to SIMD softmax.
    let scaled: Vec<f32> = input.iter().map(|&x| x / temperature).collect();
    dispatch_softmax(&scaled, output);
    Ok(())
}

// ── 4. Top-k filtered softmax ───────────────────────────────────────────

/// Top-K softmax: computes softmax only over the `k` largest elements,
/// setting all others to 0. The surviving values sum to ~1.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] on length mismatch or `k == 0`.
pub fn simd_softmax_topk(input: &[f32], output: &mut [f32], k: usize) -> Result<()> {
    if input.len() != output.len() {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("input/output length mismatch: {} vs {}", input.len(), output.len()),
        }));
    }
    if k == 0 {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: "k must be > 0".to_string(),
        }));
    }
    if input.is_empty() {
        return Ok(());
    }

    let effective_k = k.min(input.len());

    // Find the k-th largest value via partial sort on indices.
    let mut indices: Vec<usize> = (0..input.len()).collect();
    indices.select_nth_unstable_by(effective_k.saturating_sub(1), |&a, &b| {
        input[b].partial_cmp(&input[a]).unwrap_or(std::cmp::Ordering::Equal)
    });

    let top_k_indices = &indices[..effective_k];
    let mut mask = vec![false; input.len()];
    for &idx in top_k_indices {
        mask[idx] = true;
    }

    simd_softmax_masked(input, output, &mask)
}

// ── 5. Masked softmax (causal attention mask support) ───────────────────

/// Masked softmax: positions where `mask[i]` is `false` are set to 0 in the
/// output; remaining positions receive a valid softmax distribution that sums
/// to ~1.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] on length mismatch.
pub fn simd_softmax_masked(input: &[f32], output: &mut [f32], mask: &[bool]) -> Result<()> {
    if input.len() != output.len() || input.len() != mask.len() {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!(
                "length mismatch: input={}, output={}, mask={}",
                input.len(),
                output.len(),
                mask.len()
            ),
        }));
    }
    if input.is_empty() {
        return Ok(());
    }

    // Build a masked copy with NEG_INFINITY for masked positions.
    let masked: Vec<f32> = input
        .iter()
        .zip(mask.iter())
        .map(|(&x, &m)| if m { x } else { f32::NEG_INFINITY })
        .collect();

    dispatch_softmax(&masked, output);

    // Ensure masked positions are exactly 0 (exp(-inf) may give tiny values).
    for (o, &m) in output.iter_mut().zip(mask.iter()) {
        if !m {
            *o = 0.0;
        }
    }
    Ok(())
}

/// Causal (lower-triangular) softmax for attention.
///
/// For a `seq_len × seq_len` matrix stored row-major, applies softmax per
/// row with a causal mask: position `(row, col)` is visible iff `col <= row`.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] when the buffer length does
/// not equal `seq_len * seq_len`.
pub fn simd_softmax_causal(input: &[f32], output: &mut [f32], seq_len: usize) -> Result<()> {
    let total = seq_len * seq_len;
    if input.len() != total || output.len() != total {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!(
                "buffer length mismatch: input={}, output={}, expected={}",
                input.len(),
                output.len(),
                total
            ),
        }));
    }

    for row in 0..seq_len {
        let off = row * seq_len;
        let row_in = &input[off..off + seq_len];
        let row_out = &mut output[off..off + seq_len];

        // Build causal mask: col <= row is visible.
        let mask: Vec<bool> = (0..seq_len).map(|col| col <= row).collect();
        simd_softmax_masked(row_in, row_out, &mask)?;
    }
    Ok(())
}

// ── 6. Multi-head batched softmax ───────────────────────────────────────

/// Multi-head batched softmax: applies softmax independently to each row
/// across `num_heads` heads, each of shape `[seq_len, seq_len]`.
///
/// Total buffer size: `num_heads * seq_len * seq_len`.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] on size mismatch or zero dimensions.
pub fn simd_softmax_multi_head(
    input: &[f32],
    output: &mut [f32],
    num_heads: usize,
    seq_len: usize,
) -> Result<()> {
    if num_heads == 0 || seq_len == 0 {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!(
                "num_heads and seq_len must be > 0, got num_heads={num_heads}, seq_len={seq_len}"
            ),
        }));
    }
    let total = num_heads * seq_len * seq_len;
    if input.len() != total || output.len() != total {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!(
                "buffer length mismatch: input={}, output={}, expected={}",
                input.len(),
                output.len(),
                total
            ),
        }));
    }

    let head_size = seq_len * seq_len;
    for h in 0..num_heads {
        let head_off = h * head_size;
        for row in 0..seq_len {
            let off = head_off + row * seq_len;
            let row_in = &input[off..off + seq_len];
            let row_out = &mut output[off..off + seq_len];
            dispatch_softmax(row_in, row_out);
        }
    }
    Ok(())
}

/// Multi-head batched softmax with causal masking.
///
/// Combines multi-head batching with causal (lower-triangular) mask.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] on size mismatch or zero dimensions.
pub fn simd_softmax_multi_head_causal(
    input: &[f32],
    output: &mut [f32],
    num_heads: usize,
    seq_len: usize,
) -> Result<()> {
    if num_heads == 0 || seq_len == 0 {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!(
                "num_heads and seq_len must be > 0, got num_heads={num_heads}, seq_len={seq_len}"
            ),
        }));
    }
    let total = num_heads * seq_len * seq_len;
    if input.len() != total || output.len() != total {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!(
                "buffer length mismatch: input={}, output={}, expected={}",
                input.len(),
                output.len(),
                total
            ),
        }));
    }

    let head_size = seq_len * seq_len;
    for h in 0..num_heads {
        let head_off = h * head_size;
        simd_softmax_causal(
            &input[head_off..head_off + head_size],
            &mut output[head_off..head_off + head_size],
            seq_len,
        )?;
    }
    Ok(())
}

// ── 7. Log-softmax ──────────────────────────────────────────────────────

/// Numerically-stable log-softmax: `log_softmax(x)_i = x_i - max - log(Σ exp(x_j - max))`.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] on length mismatch.
pub fn simd_log_softmax(input: &[f32], output: &mut [f32]) -> Result<()> {
    if input.len() != output.len() {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("input/output length mismatch: {} vs {}", input.len(), output.len()),
        }));
    }
    if input.is_empty() {
        return Ok(());
    }

    let max_val = scalar_max(input);
    let mut sum_exp = 0.0f32;
    for &x in input {
        sum_exp += fast_exp(x - max_val);
    }
    let log_sum_exp = max_val + sum_exp.ln();
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = x - log_sum_exp;
    }
    Ok(())
}

/// In-place log-softmax.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] when data contains only NaN
/// (practically never occurs).
pub fn simd_log_softmax_inplace(data: &mut [f32]) -> Result<()> {
    if data.is_empty() {
        return Ok(());
    }

    let max_val = scalar_max(data);
    let mut sum_exp = 0.0f32;
    for &x in data.iter() {
        sum_exp += fast_exp(x - max_val);
    }
    let log_sum_exp = max_val + sum_exp.ln();
    for x in data.iter_mut() {
        *x -= log_sum_exp;
    }
    Ok(())
}

// ── 8. Softmax with diagnostics ─────────────────────────────────────────

/// Diagnostic information about a softmax computation.
#[derive(Debug, Clone)]
pub struct SoftmaxDiagnostics {
    /// Shannon entropy of the output distribution: `−Σ p_i log(p_i)`.
    pub entropy: f32,
    /// Maximum probability in the output.
    pub max_prob: f32,
    /// Minimum probability in the output.
    pub min_prob: f32,
    /// Whether any input was NaN.
    pub has_nan: bool,
    /// Whether any input was ±infinity.
    pub has_inf: bool,
    /// Range of input logits: `max(input) - min(input)`.
    pub logit_range: f32,
}

/// Softmax with diagnostic reporting.
///
/// Computes numerically-stable softmax and returns a [`SoftmaxDiagnostics`]
/// struct with entropy, NaN/inf detection, and probability range information.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] on length mismatch.
pub fn simd_softmax_with_diagnostics(
    input: &[f32],
    output: &mut [f32],
) -> Result<SoftmaxDiagnostics> {
    if input.len() != output.len() {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("input/output length mismatch: {} vs {}", input.len(), output.len()),
        }));
    }

    let has_nan = input.iter().any(|x| x.is_nan());
    let has_inf = input.iter().any(|x| x.is_infinite());

    if input.is_empty() {
        return Ok(SoftmaxDiagnostics {
            entropy: 0.0,
            max_prob: 0.0,
            min_prob: 0.0,
            has_nan,
            has_inf,
            logit_range: 0.0,
        });
    }

    // Filter out NaN/inf for logit range and softmax computation.
    let finite_vals: Vec<f32> = input.iter().copied().filter(|x| x.is_finite()).collect();
    let logit_range = if finite_vals.is_empty() {
        0.0
    } else {
        let lo = finite_vals.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = finite_vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        hi - lo
    };

    // Replace NaN/inf for softmax computation.
    let clean: Vec<f32> = input
        .iter()
        .map(|&x| {
            if x.is_nan() {
                0.0
            } else if x == f32::INFINITY {
                88.0
            } else if x == f32::NEG_INFINITY {
                -88.0
            } else {
                x
            }
        })
        .collect();

    dispatch_softmax(&clean, output);

    // Compute diagnostics from output.
    let max_prob = output.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min_prob = output.iter().copied().fold(f32::INFINITY, f32::min);

    // Shannon entropy: −Σ p_i log(p_i), treating 0·log(0) = 0.
    let entropy = output.iter().map(|&p| if p > 0.0 { -p * p.ln() } else { 0.0 }).sum::<f32>();

    Ok(SoftmaxDiagnostics { entropy, max_prob, min_prob, has_nan, has_inf, logit_range })
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: assert all values sum to ~1.
    fn assert_sums_to_one(v: &[f32], tol: f32) {
        let s: f32 = v.iter().sum();
        assert!((s - 1.0).abs() < tol, "expected sum ≈ 1.0, got {s} (delta {})", (s - 1.0).abs());
    }

    /// Helper: assert all values are non-negative.
    fn assert_non_negative(v: &[f32]) {
        for (i, &x) in v.iter().enumerate() {
            assert!(x >= 0.0, "output[{i}] = {x} < 0");
        }
    }

    // ── 1. simd_softmax basic ───────────────────────────────────────────

    #[test]
    fn test_simd_softmax_basic() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = [0.0; 4];
        simd_softmax(&input, &mut output).unwrap();
        assert_sums_to_one(&output, 1e-6);
        assert_non_negative(&output);
        // Monotonically increasing.
        for i in 1..output.len() {
            assert!(output[i] > output[i - 1]);
        }
    }

    #[test]
    fn test_simd_softmax_single_element() {
        let input = [42.0];
        let mut output = [0.0; 1];
        simd_softmax(&input, &mut output).unwrap();
        assert!((output[0] - 1.0).abs() < 1e-7);
    }

    #[test]
    fn test_simd_softmax_empty() {
        let input: [f32; 0] = [];
        let mut output: Vec<f32> = vec![];
        simd_softmax(&input, &mut output).unwrap();
    }

    #[test]
    fn test_simd_softmax_all_same() {
        let input = [5.0; 8];
        let mut output = [0.0; 8];
        simd_softmax(&input, &mut output).unwrap();
        assert_sums_to_one(&output, 1e-6);
        for &x in &output {
            assert!((x - 0.125).abs() < 1e-6, "expected uniform 1/8, got {x}");
        }
    }

    #[test]
    fn test_simd_softmax_length_mismatch() {
        let input = [1.0, 2.0];
        let mut output = [0.0; 3];
        assert!(simd_softmax(&input, &mut output).is_err());
    }

    #[test]
    fn test_simd_softmax_large_positive() {
        let input = [1000.0, 1001.0, 1002.0];
        let mut output = [0.0; 3];
        simd_softmax(&input, &mut output).unwrap();
        assert_sums_to_one(&output, 1e-5);
        for &x in &output {
            assert!(x.is_finite(), "non-finite output with large inputs");
        }
    }

    #[test]
    fn test_simd_softmax_large_negative() {
        let input = [-1000.0, -999.0, -998.0];
        let mut output = [0.0; 3];
        simd_softmax(&input, &mut output).unwrap();
        assert_sums_to_one(&output, 1e-5);
        assert_non_negative(&output);
    }

    #[test]
    fn test_simd_softmax_mixed_extreme() {
        let input = [-1000.0, 0.0, 1000.0];
        let mut output = [0.0; 3];
        simd_softmax(&input, &mut output).unwrap();
        assert_sums_to_one(&output, 1e-5);
        assert!(output[2] > 0.99);
    }

    #[test]
    fn test_softmax_stability_no_nan() {
        let input = [f32::MAX / 2.0, f32::MAX / 2.0];
        let mut output = [0.0; 2];
        simd_softmax(&input, &mut output).unwrap();
        for &x in &output {
            assert!(!x.is_nan(), "NaN in output");
            assert!(x.is_finite(), "non-finite output");
        }
    }

    #[test]
    fn test_softmax_zeros() {
        let input = [0.0; 5];
        let mut output = [0.0; 5];
        simd_softmax(&input, &mut output).unwrap();
        assert_sums_to_one(&output, 1e-6);
        for &x in &output {
            assert!((x - 0.2).abs() < 1e-6);
        }
    }

    #[test]
    fn test_softmax_negative_values() {
        let input = [-3.0, -2.0, -1.0];
        let mut output = [0.0; 3];
        simd_softmax(&input, &mut output).unwrap();
        assert_sums_to_one(&output, 1e-6);
        assert!(output[2] > output[1]);
        assert!(output[1] > output[0]);
    }

    #[test]
    fn test_softmax_identical_large() {
        let input = [100.0; 16];
        let mut output = [0.0; 16];
        simd_softmax(&input, &mut output).unwrap();
        assert_sums_to_one(&output, 1e-5);
        for &x in &output {
            assert!((x - 1.0 / 16.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_softmax_alternating_extreme() {
        let input = [-88.0, 88.0, -88.0, 88.0];
        let mut output = [0.0; 4];
        simd_softmax(&input, &mut output).unwrap();
        assert_sums_to_one(&output, 1e-5);
        assert!(output[0] < 1e-10);
        assert!(output[2] < 1e-10);
    }

    #[test]
    fn test_softmax_gradual_range() {
        let input: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let mut output = [0.0; 32];
        simd_softmax(&input, &mut output).unwrap();
        assert_sums_to_one(&output, 1e-5);
        for i in 1..32 {
            assert!(output[i] > output[i - 1]);
        }
    }

    // ── SIMD vs scalar equivalence ──────────────────────────────────────

    #[test]
    fn test_simd_vs_scalar_small() {
        let input = [0.5, -1.0, 2.0, 0.0, 1.5];
        let mut scalar_out = [0.0; 5];
        simd_softmax_scalar(&input, &mut scalar_out);
        let mut simd_out = [0.0; 5];
        simd_softmax(&input, &mut simd_out).unwrap();
        for (a, b) in scalar_out.iter().zip(simd_out.iter()) {
            assert!((a - b).abs() < 1e-6, "scalar={a}, simd={b}");
        }
    }

    #[test]
    fn test_simd_vs_scalar_exact_8() {
        // Exactly 8 elements — single AVX2 pass with no tail.
        let input = [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0];
        let mut scalar_out = [0.0; 8];
        simd_softmax_scalar(&input, &mut scalar_out);
        let mut simd_out = [0.0; 8];
        simd_softmax(&input, &mut simd_out).unwrap();
        for (a, b) in scalar_out.iter().zip(simd_out.iter()) {
            assert!((a - b).abs() < 1e-6, "scalar={a}, simd={b}");
        }
    }

    #[test]
    fn test_simd_vs_scalar_large() {
        let input: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1) - 5.0).collect();
        let mut scalar_out = [0.0; 100];
        simd_softmax_scalar(&input, &mut scalar_out);
        let mut simd_out = [0.0; 100];
        simd_softmax(&input, &mut simd_out).unwrap();
        for (i, (a, b)) in scalar_out.iter().zip(simd_out.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "index {i}: scalar={a}, simd={b}");
        }
    }

    // ── In-place ────────────────────────────────────────────────────────

    #[test]
    fn test_inplace_basic() {
        let mut data = vec![1.0, 2.0, 3.0];
        simd_softmax_inplace(&mut data).unwrap();
        assert_sums_to_one(&data, 1e-6);
    }

    #[test]
    fn test_inplace_empty() {
        let mut data: Vec<f32> = vec![];
        simd_softmax_inplace(&mut data).unwrap();
    }

    #[test]
    fn test_inplace_matches_out_of_place() {
        let input = vec![0.5, -1.0, 2.0, 0.0, 1.5];
        let mut out1 = [0.0; 5];
        simd_softmax(&input, &mut out1).unwrap();
        let mut out2 = input.clone();
        simd_softmax_inplace(&mut out2).unwrap();
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!((a - b).abs() < 1e-7, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_inplace_single() {
        let mut data = [99.0];
        simd_softmax_inplace(&mut data).unwrap();
        assert!((data[0] - 1.0).abs() < 1e-7);
    }

    // ── 2. Online softmax ───────────────────────────────────────────────

    #[test]
    fn test_online_basic() {
        let input = [1.0, 2.0, 3.0];
        let mut out_online = [0.0; 3];
        simd_softmax_online(&input, &mut out_online).unwrap();
        assert_sums_to_one(&out_online, 1e-5);
        assert_non_negative(&out_online);
    }

    #[test]
    fn test_online_matches_standard() {
        let input = [0.5, -1.0, 2.0, 0.0, 1.5, -0.5, 3.0, 1.0];
        let mut std_out = [0.0; 8];
        simd_softmax(&input, &mut std_out).unwrap();
        let mut online_out = [0.0; 8];
        simd_softmax_online(&input, &mut online_out).unwrap();
        for (i, (a, b)) in std_out.iter().zip(online_out.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "index {i}: standard={a}, online={b}");
        }
    }

    #[test]
    fn test_online_empty() {
        let input: [f32; 0] = [];
        let mut output: Vec<f32> = vec![];
        simd_softmax_online(&input, &mut output).unwrap();
    }

    #[test]
    fn test_online_single_element() {
        let input = [7.0];
        let mut output = [0.0; 1];
        simd_softmax_online(&input, &mut output).unwrap();
        assert!((output[0] - 1.0).abs() < 1e-7);
    }

    #[test]
    fn test_online_length_mismatch() {
        let input = [1.0, 2.0];
        let mut output = [0.0; 3];
        assert!(simd_softmax_online(&input, &mut output).is_err());
    }

    #[test]
    fn test_online_all_same() {
        let input = [3.0; 4];
        let mut output = [0.0; 4];
        simd_softmax_online(&input, &mut output).unwrap();
        for &x in &output {
            assert!((x - 0.25).abs() < 1e-5);
        }
    }

    #[test]
    fn test_online_large_values() {
        let input = [500.0, 501.0, 502.0];
        let mut output = [0.0; 3];
        simd_softmax_online(&input, &mut output).unwrap();
        assert_sums_to_one(&output, 1e-4);
        for &x in &output {
            assert!(x.is_finite());
        }
    }

    // ── 3. Temperature softmax ──────────────────────────────────────────

    #[test]
    fn test_temperature_one_is_identity() {
        let input = [1.0, 2.0, 3.0];
        let mut out_t1 = [0.0; 3];
        simd_softmax_temperature(&input, &mut out_t1, 1.0).unwrap();
        let mut out_plain = [0.0; 3];
        simd_softmax(&input, &mut out_plain).unwrap();
        for (a, b) in out_t1.iter().zip(out_plain.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_temperature_zero_is_argmax() {
        let input = [1.0, 5.0, 3.0, 2.0];
        let mut output = [0.0; 4];
        simd_softmax_temperature(&input, &mut output, 0.0).unwrap();
        assert!((output[1] - 1.0).abs() < 1e-7, "argmax should be 1.0");
        assert!(output[0].abs() < 1e-7);
        assert!(output[2].abs() < 1e-7);
        assert!(output[3].abs() < 1e-7);
    }

    #[test]
    fn test_temperature_high_approaches_uniform() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = [0.0; 4];
        simd_softmax_temperature(&input, &mut output, 1e6).unwrap();
        let expected = 1.0 / 4.0;
        for &x in &output {
            assert!((x - expected).abs() < 1e-3, "expected near-uniform with high temp, got {x}");
        }
    }

    #[test]
    fn test_temperature_low_sharpens() {
        let input = [1.0, 2.0, 3.0];
        let mut out_sharp = [0.0; 3];
        simd_softmax_temperature(&input, &mut out_sharp, 0.1).unwrap();
        // With low temperature, the largest element dominates.
        assert!(out_sharp[2] > 0.99);
    }

    #[test]
    fn test_temperature_negative_error() {
        let input = [1.0, 2.0];
        let mut output = [0.0; 2];
        assert!(simd_softmax_temperature(&input, &mut output, -1.0).is_err());
    }

    #[test]
    fn test_temperature_empty() {
        let input: [f32; 0] = [];
        let mut output: Vec<f32> = vec![];
        simd_softmax_temperature(&input, &mut output, 1.0).unwrap();
    }

    #[test]
    fn test_temperature_length_mismatch() {
        let input = [1.0, 2.0];
        let mut output = [0.0; 3];
        assert!(simd_softmax_temperature(&input, &mut output, 1.0).is_err());
    }

    #[test]
    fn test_temperature_preserves_order() {
        let input = [1.0, 3.0, 2.0];
        let mut output = [0.0; 3];
        simd_softmax_temperature(&input, &mut output, 0.5).unwrap();
        assert!(output[1] > output[2]);
        assert!(output[2] > output[0]);
    }

    #[test]
    fn test_temperature_moderate_value() {
        let input = [1.0, 2.0, 3.0];
        let mut out = [0.0; 3];
        simd_softmax_temperature(&input, &mut out, 2.0).unwrap();
        assert_sums_to_one(&out, 1e-6);
        // Higher temperature → more uniform → smaller gap.
        let mut out_t1 = [0.0; 3];
        simd_softmax_temperature(&input, &mut out_t1, 1.0).unwrap();
        let gap_t2 = out[2] - out[0];
        let gap_t1 = out_t1[2] - out_t1[0];
        assert!(gap_t2 < gap_t1, "temp=2 should be more uniform than temp=1");
    }

    #[test]
    fn test_temperature_with_large_input() {
        let input = [500.0, 501.0, 502.0];
        let mut output = [0.0; 3];
        simd_softmax_temperature(&input, &mut output, 0.5).unwrap();
        assert_sums_to_one(&output, 1e-5);
        for &x in &output {
            assert!(x.is_finite());
        }
    }

    // ── 4. Top-k softmax ────────────────────────────────────────────────

    #[test]
    fn test_topk_basic() {
        let input = [1.0, 4.0, 2.0, 3.0];
        let mut output = [0.0; 4];
        simd_softmax_topk(&input, &mut output, 2).unwrap();
        // Elements at index 1 (4.0) and 3 (3.0) should be non-zero.
        assert!(output[1] > 0.0);
        assert!(output[3] > 0.0);
        assert!((output[0]).abs() < 1e-10);
        assert!((output[2]).abs() < 1e-10);
        let s: f32 = output.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_topk_k_equals_n() {
        let input = [1.0, 2.0, 3.0];
        let mut output = [0.0; 3];
        simd_softmax_topk(&input, &mut output, 3).unwrap();
        assert_sums_to_one(&output, 1e-6);
        assert_non_negative(&output);
    }

    #[test]
    fn test_topk_k_greater_than_n() {
        let input = [1.0, 2.0];
        let mut output = [0.0; 2];
        simd_softmax_topk(&input, &mut output, 10).unwrap();
        assert_sums_to_one(&output, 1e-6);
    }

    #[test]
    fn test_topk_k_is_one() {
        let input = [1.0, 5.0, 3.0];
        let mut output = [0.0; 3];
        simd_softmax_topk(&input, &mut output, 1).unwrap();
        assert!((output[1] - 1.0).abs() < 1e-7);
        assert!(output[0].abs() < 1e-10);
        assert!(output[2].abs() < 1e-10);
    }

    #[test]
    fn test_topk_k_zero_error() {
        let input = [1.0, 2.0];
        let mut output = [0.0; 2];
        assert!(simd_softmax_topk(&input, &mut output, 0).is_err());
    }

    #[test]
    fn test_topk_empty() {
        let input: [f32; 0] = [];
        let mut output: Vec<f32> = vec![];
        simd_softmax_topk(&input, &mut output, 1).unwrap();
    }

    #[test]
    fn test_topk_length_mismatch() {
        let input = [1.0, 2.0];
        let mut output = [0.0; 3];
        assert!(simd_softmax_topk(&input, &mut output, 1).is_err());
    }

    #[test]
    fn test_topk_single_element() {
        let input = [5.0];
        let mut output = [0.0; 1];
        simd_softmax_topk(&input, &mut output, 1).unwrap();
        assert!((output[0] - 1.0).abs() < 1e-7);
    }

    #[test]
    fn test_topk_with_ties() {
        let input = [3.0, 3.0, 3.0, 1.0];
        let mut output = [0.0; 4];
        simd_softmax_topk(&input, &mut output, 2).unwrap();
        // Exactly 2 elements should be non-zero.
        let nonzero: usize = output.iter().filter(|&&x| x > 1e-10).count();
        assert_eq!(nonzero, 2);
        let s: f32 = output.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
    }

    // ── 5. Masked softmax ───────────────────────────────────────────────

    #[test]
    fn test_masked_basic() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mask = [true, false, true, false];
        let mut output = [0.0; 4];
        simd_softmax_masked(&input, &mut output, &mask).unwrap();
        assert!((output[1]).abs() < 1e-10, "masked position should be 0");
        assert!((output[3]).abs() < 1e-10, "masked position should be 0");
        assert!(output[0] > 0.0);
        assert!(output[2] > 0.0);
        let s: f32 = output.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_masked_all_true() {
        let input = [1.0, 2.0, 3.0];
        let mask = [true, true, true];
        let mut output = [0.0; 3];
        simd_softmax_masked(&input, &mut output, &mask).unwrap();
        assert_sums_to_one(&output, 1e-6);
    }

    #[test]
    fn test_masked_all_false() {
        let input = [1.0, 2.0, 3.0];
        let mask = [false, false, false];
        let mut output = [0.0; 3];
        simd_softmax_masked(&input, &mut output, &mask).unwrap();
        for &x in &output {
            assert!(x.abs() < 1e-10);
        }
    }

    #[test]
    fn test_masked_empty() {
        let input: [f32; 0] = [];
        let mask: [bool; 0] = [];
        let mut output: Vec<f32> = vec![];
        simd_softmax_masked(&input, &mut output, &mask).unwrap();
    }

    #[test]
    fn test_masked_length_mismatch() {
        let input = [1.0, 2.0];
        let mask = [true];
        let mut output = [0.0; 2];
        assert!(simd_softmax_masked(&input, &mut output, &mask).is_err());
    }

    #[test]
    fn test_masked_single_visible() {
        let input = [1.0, 5.0, 3.0];
        let mask = [false, true, false];
        let mut output = [0.0; 3];
        simd_softmax_masked(&input, &mut output, &mask).unwrap();
        assert!((output[1] - 1.0).abs() < 1e-7);
        assert!(output[0].abs() < 1e-10);
        assert!(output[2].abs() < 1e-10);
    }

    // ── 5b. Causal softmax ──────────────────────────────────────────────

    #[test]
    fn test_causal_1x1() {
        let input = [5.0];
        let mut output = [0.0; 1];
        simd_softmax_causal(&input, &mut output, 1).unwrap();
        assert!((output[0] - 1.0).abs() < 1e-7);
    }

    #[test]
    fn test_causal_2x2() {
        // Row 0: only col 0 visible → [1.0, 0.0]
        // Row 1: both visible → softmax([3, 4])
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = [0.0; 4];
        simd_softmax_causal(&input, &mut output, 2).unwrap();
        assert!((output[0] - 1.0).abs() < 1e-7);
        assert!(output[1].abs() < 1e-10);
        assert_sums_to_one(&output[2..4], 1e-5);
    }

    #[test]
    fn test_causal_3x3() {
        let input = [0.0; 9];
        let mut output = [0.0; 9];
        simd_softmax_causal(&input, &mut output, 3).unwrap();
        // Row 0: 1 visible → [1.0, 0, 0]
        assert!((output[0] - 1.0).abs() < 1e-7);
        assert!(output[1].abs() < 1e-10);
        assert!(output[2].abs() < 1e-10);
        // Row 1: 2 visible → [0.5, 0.5, 0]
        assert!((output[3] - 0.5).abs() < 1e-5);
        assert!((output[4] - 0.5).abs() < 1e-5);
        assert!(output[5].abs() < 1e-10);
        // Row 2: 3 visible → [1/3, 1/3, 1/3]
        for &x in &output[6..9] {
            assert!((x - 1.0 / 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_causal_dim_mismatch() {
        let input = [1.0, 2.0, 3.0]; // length 3 ≠ 2*2
        let mut output = [0.0; 3];
        assert!(simd_softmax_causal(&input, &mut output, 2).is_err());
    }

    #[test]
    fn test_causal_future_tokens_zero() {
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut output = [0.0; 16];
        simd_softmax_causal(&input, &mut output, 4).unwrap();
        // Verify future tokens are zero.
        for row in 0..4 {
            for col in (row + 1)..4 {
                assert!(
                    output[row * 4 + col].abs() < 1e-10,
                    "future position ({row},{col}) should be 0, got {}",
                    output[row * 4 + col]
                );
            }
        }
    }

    #[test]
    fn test_causal_uniform_input() {
        let input = [1.0; 9];
        let mut output = [0.0; 9];
        simd_softmax_causal(&input, &mut output, 3).unwrap();
        // Row 0: [1.0, 0, 0]
        assert!((output[0] - 1.0).abs() < 1e-5);
        // Row 1: [0.5, 0.5, 0]
        assert!((output[3] - 0.5).abs() < 1e-5);
        assert!((output[4] - 0.5).abs() < 1e-5);
    }

    // ── 6. Multi-head batched softmax ───────────────────────────────────

    #[test]
    fn test_multi_head_basic() {
        let input = [1.0; 8]; // 2 heads × 2×2
        let mut output = [0.0; 8];
        simd_softmax_multi_head(&input, &mut output, 2, 2).unwrap();
        for row_start in (0..8).step_by(2) {
            assert_sums_to_one(&output[row_start..row_start + 2], 1e-5);
        }
    }

    #[test]
    fn test_multi_head_independent_heads() {
        // Two heads with different values should produce independent results.
        let mut input = [0.0; 8];
        // Head 0: [[1,2],[3,4]]
        input[0..4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        // Head 1: [[5,6],[7,8]]
        input[4..8].copy_from_slice(&[5.0, 6.0, 7.0, 8.0]);

        let mut output = [0.0; 8];
        simd_softmax_multi_head(&input, &mut output, 2, 2).unwrap();

        // Head 0 row 0
        assert_sums_to_one(&output[0..2], 1e-5);
        // Head 0 row 1
        assert_sums_to_one(&output[2..4], 1e-5);
        // Head 1 row 0
        assert_sums_to_one(&output[4..6], 1e-5);
        // Head 1 row 1
        assert_sums_to_one(&output[6..8], 1e-5);
    }

    #[test]
    fn test_multi_head_size_mismatch() {
        let input = [1.0; 10];
        let mut output = [0.0; 10];
        // 2 heads × 2×2 = 8 ≠ 10
        assert!(simd_softmax_multi_head(&input, &mut output, 2, 2).is_err());
    }

    #[test]
    fn test_multi_head_zero_heads_error() {
        let input = [1.0; 4];
        let mut output = [0.0; 4];
        assert!(simd_softmax_multi_head(&input, &mut output, 0, 2).is_err());
    }

    #[test]
    fn test_multi_head_zero_seq_error() {
        let input: Vec<f32> = vec![];
        let mut output: Vec<f32> = vec![];
        assert!(simd_softmax_multi_head(&input, &mut output, 2, 0).is_err());
    }

    // ── 6b. Multi-head causal ───────────────────────────────────────────

    #[test]
    fn test_multi_head_causal_basic() {
        let input = [0.0; 8]; // 2 heads × 2×2
        let mut output = [0.0; 8];
        simd_softmax_multi_head_causal(&input, &mut output, 2, 2).unwrap();
        // Row 0 of each head: [1.0, 0.0]
        assert!((output[0] - 1.0).abs() < 1e-5);
        assert!(output[1].abs() < 1e-10);
        assert!((output[4] - 1.0).abs() < 1e-5);
        assert!(output[5].abs() < 1e-10);
    }

    #[test]
    fn test_multi_head_causal_size_mismatch() {
        let input = [1.0; 5];
        let mut output = [0.0; 5];
        assert!(simd_softmax_multi_head_causal(&input, &mut output, 1, 2).is_err());
    }

    #[test]
    fn test_multi_head_causal_matches_per_head() {
        // Multi-head causal should equal applying causal per-head.
        let input: Vec<f32> = (0..18).map(|i| i as f32 * 0.1).collect(); // 2 heads × 3×3
        let mut mh_out = [0.0; 18];
        simd_softmax_multi_head_causal(&input, &mut mh_out, 2, 3).unwrap();

        let mut h0_out = [0.0; 9];
        simd_softmax_causal(&input[0..9], &mut h0_out, 3).unwrap();
        let mut h1_out = [0.0; 9];
        simd_softmax_causal(&input[9..18], &mut h1_out, 3).unwrap();

        for (a, b) in mh_out[0..9].iter().zip(h0_out.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
        for (a, b) in mh_out[9..18].iter().zip(h1_out.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    // ── 7. Log-softmax ──────────────────────────────────────────────────

    #[test]
    fn test_log_softmax_basic() {
        let input = [1.0, 2.0, 3.0];
        let mut output = [0.0; 3];
        simd_log_softmax(&input, &mut output).unwrap();
        for &x in &output {
            assert!(x <= 0.0, "log_softmax value {x} > 0");
        }
        let exp_sum: f32 = output.iter().map(|&x| x.exp()).sum();
        assert!((exp_sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_log_softmax_identity() {
        let input = [0.5, -1.0, 2.0, 0.0];
        let mut sm = [0.0; 4];
        simd_softmax(&input, &mut sm).unwrap();
        let log_sm: Vec<f32> = sm.iter().map(|&x| x.ln()).collect();

        let mut lsm = [0.0; 4];
        simd_log_softmax(&input, &mut lsm).unwrap();

        for (a, b) in log_sm.iter().zip(lsm.iter()) {
            assert!((a - b).abs() < 1e-5, "log(softmax) vs log_softmax: {a} vs {b}");
        }
    }

    #[test]
    fn test_log_softmax_empty() {
        let input: [f32; 0] = [];
        let mut output: Vec<f32> = vec![];
        simd_log_softmax(&input, &mut output).unwrap();
    }

    #[test]
    fn test_log_softmax_length_mismatch() {
        let input = [1.0];
        let mut output = [0.0; 2];
        assert!(simd_log_softmax(&input, &mut output).is_err());
    }

    #[test]
    fn test_log_softmax_single() {
        let input = [5.0];
        let mut output = [0.0; 1];
        simd_log_softmax(&input, &mut output).unwrap();
        assert!((output[0]).abs() < 1e-7, "log_softmax of single element should be 0");
    }

    #[test]
    fn test_log_softmax_large_values() {
        let input = [1000.0, 1001.0, 1002.0];
        let mut output = [0.0; 3];
        simd_log_softmax(&input, &mut output).unwrap();
        for &x in &output {
            assert!(x.is_finite(), "non-finite log_softmax with large inputs");
        }
        let exp_sum: f32 = output.iter().map(|&x| x.exp()).sum();
        assert!((exp_sum - 1.0).abs() < 1e-4);
    }

    // ── 7b. In-place log-softmax ────────────────────────────────────────

    #[test]
    fn test_log_softmax_inplace_basic() {
        let input = [1.0, 2.0, 3.0];
        let mut out1 = [0.0; 3];
        simd_log_softmax(&input, &mut out1).unwrap();

        let mut data = input.to_vec();
        simd_log_softmax_inplace(&mut data).unwrap();

        for (a, b) in out1.iter().zip(data.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
    }

    #[test]
    fn test_log_softmax_inplace_empty() {
        let mut data: Vec<f32> = vec![];
        simd_log_softmax_inplace(&mut data).unwrap();
    }

    #[test]
    fn test_log_softmax_inplace_all_negative() {
        let mut data = vec![-5.0, -3.0, -1.0];
        simd_log_softmax_inplace(&mut data).unwrap();
        for &x in &data {
            assert!(x <= 0.0);
            assert!(x.is_finite());
        }
    }

    // ── 8. Diagnostics softmax ──────────────────────────────────────────

    #[test]
    fn test_diagnostics_basic() {
        let input = [1.0, 2.0, 3.0];
        let mut output = [0.0; 3];
        let diag = simd_softmax_with_diagnostics(&input, &mut output).unwrap();
        assert_sums_to_one(&output, 1e-6);
        assert!(!diag.has_nan);
        assert!(!diag.has_inf);
        assert!(diag.entropy > 0.0);
        assert!((diag.logit_range - 2.0).abs() < 1e-7);
    }

    #[test]
    fn test_diagnostics_empty() {
        let input: [f32; 0] = [];
        let mut output: Vec<f32> = vec![];
        let diag = simd_softmax_with_diagnostics(&input, &mut output).unwrap();
        assert!(!diag.has_nan);
        assert!(!diag.has_inf);
        assert!((diag.entropy).abs() < 1e-7);
    }

    #[test]
    fn test_diagnostics_nan_input() {
        let input = [1.0, f32::NAN, 3.0];
        let mut output = [0.0; 3];
        let diag = simd_softmax_with_diagnostics(&input, &mut output).unwrap();
        assert!(diag.has_nan);
    }

    #[test]
    fn test_diagnostics_inf_input() {
        let input = [1.0, f32::INFINITY, 3.0];
        let mut output = [0.0; 3];
        let diag = simd_softmax_with_diagnostics(&input, &mut output).unwrap();
        assert!(diag.has_inf);
    }

    #[test]
    fn test_diagnostics_neg_inf_input() {
        let input = [1.0, f32::NEG_INFINITY, 3.0];
        let mut output = [0.0; 3];
        let diag = simd_softmax_with_diagnostics(&input, &mut output).unwrap();
        assert!(diag.has_inf);
    }

    #[test]
    fn test_diagnostics_entropy_uniform() {
        let input = [0.0; 4];
        let mut output = [0.0; 4];
        let diag = simd_softmax_with_diagnostics(&input, &mut output).unwrap();
        // Uniform: entropy = log(4) ≈ 1.386
        let expected = (4.0f32).ln();
        assert!(
            (diag.entropy - expected).abs() < 0.01,
            "entropy should be ~{expected}, got {}",
            diag.entropy
        );
    }

    #[test]
    fn test_diagnostics_entropy_peaked() {
        let input = [0.0, 0.0, 100.0];
        let mut output = [0.0; 3];
        let diag = simd_softmax_with_diagnostics(&input, &mut output).unwrap();
        // Peaked: entropy should be near 0.
        assert!(
            diag.entropy < 0.01,
            "peaked distribution should have near-zero entropy, got {}",
            diag.entropy
        );
    }

    #[test]
    fn test_diagnostics_max_prob() {
        let input = [0.0, 0.0, 100.0];
        let mut output = [0.0; 3];
        let diag = simd_softmax_with_diagnostics(&input, &mut output).unwrap();
        assert!((diag.max_prob - 1.0).abs() < 1e-5, "max_prob should be ~1.0");
    }

    #[test]
    fn test_diagnostics_min_prob() {
        let input = [0.0; 4];
        let mut output = [0.0; 4];
        let diag = simd_softmax_with_diagnostics(&input, &mut output).unwrap();
        assert!((diag.min_prob - 0.25).abs() < 1e-5, "min_prob should be 0.25 for uniform");
    }

    #[test]
    fn test_diagnostics_logit_range() {
        let input = [-10.0, 0.0, 10.0];
        let mut output = [0.0; 3];
        let diag = simd_softmax_with_diagnostics(&input, &mut output).unwrap();
        assert!((diag.logit_range - 20.0).abs() < 1e-7, "logit_range should be 20");
    }

    #[test]
    fn test_diagnostics_length_mismatch() {
        let input = [1.0, 2.0];
        let mut output = [0.0; 3];
        assert!(simd_softmax_with_diagnostics(&input, &mut output).is_err());
    }

    #[test]
    fn test_diagnostics_single_element() {
        let input = [42.0];
        let mut output = [0.0; 1];
        let diag = simd_softmax_with_diagnostics(&input, &mut output).unwrap();
        assert!((output[0] - 1.0).abs() < 1e-7);
        assert!((diag.entropy).abs() < 1e-7, "single element → entropy = 0");
        assert!((diag.logit_range).abs() < 1e-7, "single element → range = 0");
    }

    // ── Additional edge case tests ──────────────────────────────────────

    #[test]
    fn test_simd_softmax_non_multiple_of_8() {
        // 13 elements: 1 full AVX2 chunk + 5 tail.
        let input: Vec<f32> = (0..13).map(|i| i as f32).collect();
        let mut output = [0.0; 13];
        simd_softmax(&input, &mut output).unwrap();
        assert_sums_to_one(&output, 1e-5);
        assert_non_negative(&output);
    }

    #[test]
    fn test_simd_softmax_exact_16() {
        // 16 elements: exactly 2 AVX2 chunks, no tail.
        let input: Vec<f32> = (0..16).map(|i| (i as f32) * 0.5 - 4.0).collect();
        let mut output = [0.0; 16];
        simd_softmax(&input, &mut output).unwrap();
        assert_sums_to_one(&output, 1e-5);
    }

    #[test]
    fn test_online_matches_simd_standard() {
        // Online and standard should produce very close results.
        let input: Vec<f32> = (0..20).map(|i| (i as f32) * 0.3 - 3.0).collect();
        let mut std_out = [0.0; 20];
        simd_softmax(&input, &mut std_out).unwrap();
        let mut online_out = [0.0; 20];
        simd_softmax_online(&input, &mut online_out).unwrap();
        for (i, (a, b)) in std_out.iter().zip(online_out.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "index {i}: standard={a}, online={b}");
        }
    }
}

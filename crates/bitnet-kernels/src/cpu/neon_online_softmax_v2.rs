//! NEON-optimized online softmax v2 for Apple Silicon.
//! Numerically stable streaming softmax computation using
//! the Online Normalizer trick for large vocabularies.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Lane count for `float32x4_t` NEON vectors.
const LANES: usize = 4;

/// Default chunk size for cache-friendly softmax (16 KiB worth of f32).
const DEFAULT_CHUNK: usize = 4096;

// ── Fast exp approximation ─────────────────────────────────────────────

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
    unsafe {
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
}

// ── Horizontal NEON reductions ─────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn hsum_f32(v: float32x4_t) -> f32 {
    unsafe { vaddvq_f32(v) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn hmax_f32(v: float32x4_t) -> f32 {
    unsafe { vmaxvq_f32(v) }
}

// ════════════════════════════════════════════════════════════════════════
// 1. online_softmax_v2 — streaming 3-pass softmax
// ════════════════════════════════════════════════════════════════════════

/// Streaming 3-pass softmax: find-max, exp-sum, normalise.
///
/// Uses NEON intrinsics on aarch64 with automatic scalar fallback.
pub fn online_softmax_v2(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len(), "input/output length mismatch");
    if input.is_empty() {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                online_softmax_v2_neon(input, output);
            }
            return;
        }
    }

    online_softmax_v2_scalar(input, output);
}

/// NEON implementation of 3-pass streaming softmax.
///
/// # Safety
/// Requires aarch64 target with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn online_softmax_v2_neon(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    let ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let chunks = len / LANES;
    let tail = len % LANES;

    // Pass 1: find global max.
    let mut max_vec = unsafe { vdupq_n_f32(f32::NEG_INFINITY) };
    for i in 0..chunks {
        let v = unsafe { vld1q_f32(ptr.add(i * LANES)) };
        max_vec = unsafe { vmaxq_f32(max_vec, v) };
    }
    let mut max_val = unsafe { hmax_f32(max_vec) };
    for i in (chunks * LANES)..len {
        max_val = max_val.max(unsafe { *input.get_unchecked(i) });
    }

    // Pass 2: compute exp(x - max) and accumulate sum.
    let max_splat = unsafe { vdupq_n_f32(max_val) };
    let mut sum_vec = unsafe { vdupq_n_f32(0.0) };
    for i in 0..chunks {
        unsafe {
            let v = vld1q_f32(ptr.add(i * LANES));
            let shifted = vsubq_f32(v, max_splat);
            let e = fast_exp_neon(shifted);
            vst1q_f32(out_ptr.add(i * LANES), e);
            sum_vec = vaddq_f32(sum_vec, e);
        }
    }
    let mut sum_val = unsafe { hsum_f32(sum_vec) };
    for i in 0..tail {
        let idx = chunks * LANES + i;
        let e = fast_exp_scalar(unsafe { *input.get_unchecked(idx) } - max_val);
        unsafe { *output.get_unchecked_mut(idx) = e };
        sum_val += e;
    }

    // Pass 3: normalise.
    if sum_val == 0.0 {
        let uniform = 1.0 / len as f32;
        for o in output.iter_mut() {
            *o = uniform;
        }
        return;
    }
    let inv_sum = 1.0 / sum_val;
    let inv_splat = unsafe { vdupq_n_f32(inv_sum) };
    for i in 0..chunks {
        unsafe {
            let v = vld1q_f32(out_ptr.add(i * LANES));
            vst1q_f32(out_ptr.add(i * LANES), vmulq_f32(v, inv_splat));
        }
    }
    for i in 0..tail {
        let idx = chunks * LANES + i;
        unsafe { *output.get_unchecked_mut(idx) *= inv_sum };
    }
}

/// Scalar fallback for 3-pass streaming softmax.
pub fn online_softmax_v2_scalar(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len(), "input/output length mismatch");
    if input.is_empty() {
        return;
    }

    let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let mut sum = 0.0f32;
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        let e = fast_exp_scalar(x - max_val);
        *o = e;
        sum += e;
    }

    if sum == 0.0 {
        let uniform = 1.0 / input.len() as f32;
        for o in output.iter_mut() {
            *o = uniform;
        }
        return;
    }
    let inv = 1.0 / sum;
    for o in output.iter_mut() {
        *o *= inv;
    }
}

// ════════════════════════════════════════════════════════════════════════
// 2. online_log_softmax — log-domain softmax
// ════════════════════════════════════════════════════════════════════════

/// Log-domain softmax: `log_softmax(x_i) = x_i - max - ln(sum(exp(x_j - max)))`.
///
/// More numerically stable than `log(softmax(x))`.
pub fn online_log_softmax(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len(), "input/output length mismatch");
    if input.is_empty() {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                online_log_softmax_neon(input, output);
            }
            return;
        }
    }

    online_log_softmax_scalar(input, output);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn online_log_softmax_neon(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    let ptr = input.as_ptr();
    let chunks = len / LANES;

    // Pass 1: find max.
    let mut max_vec = unsafe { vdupq_n_f32(f32::NEG_INFINITY) };
    for i in 0..chunks {
        let v = unsafe { vld1q_f32(ptr.add(i * LANES)) };
        max_vec = unsafe { vmaxq_f32(max_vec, v) };
    }
    let mut max_val = unsafe { hmax_f32(max_vec) };
    for i in (chunks * LANES)..len {
        max_val = max_val.max(unsafe { *input.get_unchecked(i) });
    }

    // Pass 2: sum of exp(x - max).
    let max_splat = unsafe { vdupq_n_f32(max_val) };
    let mut sum_vec = unsafe { vdupq_n_f32(0.0) };
    for i in 0..chunks {
        unsafe {
            let v = vld1q_f32(ptr.add(i * LANES));
            let shifted = vsubq_f32(v, max_splat);
            sum_vec = vaddq_f32(sum_vec, fast_exp_neon(shifted));
        }
    }
    let mut sum_val = unsafe { hsum_f32(sum_vec) };
    for i in (chunks * LANES)..len {
        sum_val += fast_exp_scalar(unsafe { *input.get_unchecked(i) } - max_val);
    }

    // Pass 3: output = x - max - ln(sum).
    let log_sum = sum_val.ln();
    let log_sum_splat = unsafe { vdupq_n_f32(log_sum) };
    let out_ptr = output.as_mut_ptr();
    for i in 0..chunks {
        unsafe {
            let v = vld1q_f32(ptr.add(i * LANES));
            let shifted = vsubq_f32(v, max_splat);
            vst1q_f32(out_ptr.add(i * LANES), vsubq_f32(shifted, log_sum_splat));
        }
    }
    for i in (chunks * LANES)..len {
        unsafe {
            *output.get_unchecked_mut(i) = *input.get_unchecked(i) - max_val - log_sum;
        }
    }
}

/// Scalar fallback for log-domain softmax.
pub fn online_log_softmax_scalar(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len(), "input/output length mismatch");
    if input.is_empty() {
        return;
    }

    let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = input.iter().map(|&x| fast_exp_scalar(x - max_val)).sum();
    let log_sum = sum.ln();

    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = x - max_val - log_sum;
    }
}

// ════════════════════════════════════════════════════════════════════════
// 3. fused_softmax_mask — softmax with causal/padding mask fusion
// ════════════════════════════════════════════════════════════════════════

/// Fused softmax with boolean mask. Masked positions are set to -∞ before
/// the softmax, avoiding a separate masking pass.
///
/// `mask[i]` = `true` means the position is **valid**; `false` means masked out.
pub fn fused_softmax_mask(input: &[f32], mask: &[bool], output: &mut [f32]) {
    assert_eq!(input.len(), output.len(), "input/output length mismatch");
    assert_eq!(input.len(), mask.len(), "input/mask length mismatch");
    if input.is_empty() {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                fused_softmax_mask_neon(input, mask, output);
            }
            return;
        }
    }

    fused_softmax_mask_scalar(input, mask, output);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn fused_softmax_mask_neon(input: &[f32], mask: &[bool], output: &mut [f32]) {
    let len = input.len();
    let ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let chunks = len / LANES;
    let tail = len % LANES;
    let neg_inf = f32::NEG_INFINITY;

    // Pass 1: find max (masked).
    let mut max_vec = unsafe { vdupq_n_f32(neg_inf) };
    for i in 0..chunks {
        let base = i * LANES;
        let v = unsafe { vld1q_f32(ptr.add(base)) };
        let mut mask_bits: [u32; LANES] = [0; LANES];
        for j in 0..LANES {
            if unsafe { *mask.get_unchecked(base + j) } {
                mask_bits[j] = 0xFFFF_FFFF;
            }
        }
        unsafe {
            let mask_vec = vld1q_u32(mask_bits.as_ptr());
            let neg_inf_vec = vdupq_n_f32(neg_inf);
            let masked_v = vbslq_f32(mask_vec, v, neg_inf_vec);
            max_vec = vmaxq_f32(max_vec, masked_v);
        }
    }
    let mut max_val = unsafe { hmax_f32(max_vec) };
    for i in 0..tail {
        let idx = chunks * LANES + i;
        if unsafe { *mask.get_unchecked(idx) } {
            max_val = max_val.max(unsafe { *input.get_unchecked(idx) });
        }
    }

    // If everything is masked, output zeros.
    if max_val == neg_inf {
        for o in output.iter_mut() {
            *o = 0.0;
        }
        return;
    }

    // Pass 2: exp(x - max) for valid positions, 0 for masked.
    let max_splat = unsafe { vdupq_n_f32(max_val) };
    let zero_vec = unsafe { vdupq_n_f32(0.0) };
    let mut sum_vec = unsafe { vdupq_n_f32(0.0) };
    for i in 0..chunks {
        let base = i * LANES;
        unsafe {
            let v = vld1q_f32(ptr.add(base));
            let shifted = vsubq_f32(v, max_splat);
            let e = fast_exp_neon(shifted);
            let mut mask_bits: [u32; LANES] = [0; LANES];
            for j in 0..LANES {
                if *mask.get_unchecked(base + j) {
                    mask_bits[j] = 0xFFFF_FFFF;
                }
            }
            let mask_vec = vld1q_u32(mask_bits.as_ptr());
            let masked_e = vbslq_f32(mask_vec, e, zero_vec);
            vst1q_f32(out_ptr.add(base), masked_e);
            sum_vec = vaddq_f32(sum_vec, masked_e);
        }
    }
    let mut sum_val = unsafe { hsum_f32(sum_vec) };
    for i in 0..tail {
        let idx = chunks * LANES + i;
        if unsafe { *mask.get_unchecked(idx) } {
            let e = fast_exp_scalar(unsafe { *input.get_unchecked(idx) } - max_val);
            unsafe { *output.get_unchecked_mut(idx) = e };
            sum_val += e;
        } else {
            unsafe { *output.get_unchecked_mut(idx) = 0.0 };
        }
    }

    // Pass 3: normalise.
    if sum_val == 0.0 {
        return;
    }
    let inv_sum = 1.0 / sum_val;
    let inv_splat = unsafe { vdupq_n_f32(inv_sum) };
    for i in 0..chunks {
        let base = i * LANES;
        unsafe {
            let v = vld1q_f32(out_ptr.add(base));
            vst1q_f32(out_ptr.add(base), vmulq_f32(v, inv_splat));
        }
    }
    for i in 0..tail {
        let idx = chunks * LANES + i;
        unsafe { *output.get_unchecked_mut(idx) *= inv_sum };
    }
}

/// Scalar fallback for fused softmax with mask.
pub fn fused_softmax_mask_scalar(input: &[f32], mask: &[bool], output: &mut [f32]) {
    assert_eq!(input.len(), output.len(), "input/output length mismatch");
    assert_eq!(input.len(), mask.len(), "input/mask length mismatch");
    if input.is_empty() {
        return;
    }

    let max_val = input
        .iter()
        .zip(mask.iter())
        .filter(|(_, m)| **m)
        .map(|(x, _)| *x)
        .fold(f32::NEG_INFINITY, f32::max);

    if max_val == f32::NEG_INFINITY {
        for o in output.iter_mut() {
            *o = 0.0;
        }
        return;
    }

    let mut sum = 0.0f32;
    for (i, (o, &x)) in output.iter_mut().zip(input.iter()).enumerate() {
        if mask[i] {
            let e = fast_exp_scalar(x - max_val);
            *o = e;
            sum += e;
        } else {
            *o = 0.0;
        }
    }

    if sum == 0.0 {
        return;
    }
    let inv = 1.0 / sum;
    for o in output.iter_mut() {
        *o *= inv;
    }
}

// ════════════════════════════════════════════════════════════════════════
// 4. top_k_softmax — partial softmax over top-k elements
// ════════════════════════════════════════════════════════════════════════

/// Partial softmax: only the top-k elements receive probability mass,
/// the rest are zeroed. Returns softmax values in-place in `output`.
pub fn top_k_softmax(input: &[f32], k: usize, output: &mut [f32]) {
    assert_eq!(input.len(), output.len(), "input/output length mismatch");
    if input.is_empty() || k == 0 {
        for o in output.iter_mut() {
            *o = 0.0;
        }
        return;
    }

    let effective_k = k.min(input.len());

    // Find the k-th largest value via full sort of indices.
    let mut indices: Vec<usize> = (0..input.len()).collect();
    indices.sort_unstable_by(|&a, &b| {
        input[b].partial_cmp(&input[a]).unwrap_or(std::cmp::Ordering::Equal)
    });
    let threshold = input[indices[effective_k - 1]];

    // Build mask: top-k positions are valid (handle ties by capping at k).
    let mut selected = vec![false; input.len()];
    let mut count = 0;
    for &idx in &indices {
        if count >= effective_k {
            break;
        }
        if input[idx] >= threshold {
            selected[idx] = true;
            count += 1;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                top_k_softmax_neon(input, &selected, output);
            }
            return;
        }
    }

    top_k_softmax_scalar(input, &selected, output);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn top_k_softmax_neon(input: &[f32], selected: &[bool], output: &mut [f32]) {
    // Reuse the fused mask path.
    unsafe { fused_softmax_mask_neon(input, selected, output) }
}

/// Scalar fallback for top-k softmax.
pub fn top_k_softmax_scalar(input: &[f32], selected: &[bool], output: &mut [f32]) {
    fused_softmax_mask_scalar(input, selected, output);
}

// ════════════════════════════════════════════════════════════════════════
// 5. chunked_softmax — cache-friendly chunked computation
// ════════════════════════════════════════════════════════════════════════

/// Softmax over a large array in cache-friendly chunks.
///
/// For very large vocabularies (100k+), working in chunks that fit in L1/L2
/// cache reduces TLB misses.
pub fn chunked_softmax(input: &[f32], output: &mut [f32]) {
    chunked_softmax_with_size(input, output, DEFAULT_CHUNK);
}

/// Chunked softmax with configurable chunk size.
pub fn chunked_softmax_with_size(input: &[f32], output: &mut [f32], chunk_size: usize) {
    assert_eq!(input.len(), output.len(), "input/output length mismatch");
    if input.is_empty() {
        return;
    }
    let chunk_size = chunk_size.max(1);

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                chunked_softmax_neon(input, output, chunk_size);
            }
            return;
        }
    }

    chunked_softmax_scalar(input, output, chunk_size);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn chunked_softmax_neon(input: &[f32], output: &mut [f32], chunk_size: usize) {
    let len = input.len();

    // Phase 1: per-chunk local max.
    let n_chunks = (len + chunk_size - 1) / chunk_size;
    let mut chunk_max = vec![f32::NEG_INFINITY; n_chunks];
    let mut chunk_sum = vec![0.0f32; n_chunks];

    for c in 0..n_chunks {
        let start = c * chunk_size;
        let end = (start + chunk_size).min(len);
        let clen = end - start;
        let ptr = unsafe { input.as_ptr().add(start) };
        let simd_chunks = clen / LANES;

        let mut max_vec = unsafe { vdupq_n_f32(f32::NEG_INFINITY) };
        for i in 0..simd_chunks {
            let v = unsafe { vld1q_f32(ptr.add(i * LANES)) };
            max_vec = unsafe { vmaxq_f32(max_vec, v) };
        }
        let mut local_max = unsafe { hmax_f32(max_vec) };
        for i in (simd_chunks * LANES)..clen {
            local_max = local_max.max(unsafe { *input.get_unchecked(start + i) });
        }
        chunk_max[c] = local_max;
    }

    let global_max = chunk_max.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Phase 2: per-chunk exp and sum (relative to global max).
    for c in 0..n_chunks {
        let start = c * chunk_size;
        let end = (start + chunk_size).min(len);
        let clen = end - start;
        let ptr = unsafe { input.as_ptr().add(start) };
        let optr = unsafe { output.as_mut_ptr().add(start) };
        let simd_chunks = clen / LANES;

        let max_splat = unsafe { vdupq_n_f32(global_max) };
        let mut sum_vec = unsafe { vdupq_n_f32(0.0) };
        for i in 0..simd_chunks {
            unsafe {
                let v = vld1q_f32(ptr.add(i * LANES));
                let shifted = vsubq_f32(v, max_splat);
                let e = fast_exp_neon(shifted);
                vst1q_f32(optr.add(i * LANES), e);
                sum_vec = vaddq_f32(sum_vec, e);
            }
        }
        let mut local_sum = unsafe { hsum_f32(sum_vec) };
        for i in (simd_chunks * LANES)..clen {
            let e = fast_exp_scalar(unsafe { *input.get_unchecked(start + i) } - global_max);
            unsafe { *output.get_unchecked_mut(start + i) = e };
            local_sum += e;
        }
        chunk_sum[c] = local_sum;
    }

    let total_sum: f32 = chunk_sum.iter().sum();

    // Phase 3: global normalisation.
    if total_sum == 0.0 {
        let uniform = 1.0 / len as f32;
        for o in output.iter_mut() {
            *o = uniform;
        }
        return;
    }
    let inv_sum = 1.0 / total_sum;
    let inv_splat = unsafe { vdupq_n_f32(inv_sum) };
    for c in 0..n_chunks {
        let start = c * chunk_size;
        let end = (start + chunk_size).min(len);
        let clen = end - start;
        let optr = unsafe { output.as_mut_ptr().add(start) };
        let simd_chunks = clen / LANES;

        for i in 0..simd_chunks {
            unsafe {
                let v = vld1q_f32(optr.add(i * LANES));
                vst1q_f32(optr.add(i * LANES), vmulq_f32(v, inv_splat));
            }
        }
        for i in (simd_chunks * LANES)..clen {
            unsafe { *output.get_unchecked_mut(start + i) *= inv_sum };
        }
    }
}

/// Scalar fallback for chunked softmax.
pub fn chunked_softmax_scalar(input: &[f32], output: &mut [f32], chunk_size: usize) {
    assert_eq!(input.len(), output.len(), "input/output length mismatch");
    if input.is_empty() {
        return;
    }
    let chunk_size = chunk_size.max(1);
    let len = input.len();

    // Phase 1: global max across chunks.
    let mut global_max = f32::NEG_INFINITY;
    for chunk in input.chunks(chunk_size) {
        let local_max = chunk.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        global_max = global_max.max(local_max);
    }

    // Phase 2: exp and accumulate sum.
    let mut total_sum = 0.0f32;
    for (in_chunk, out_chunk) in input.chunks(chunk_size).zip(output.chunks_mut(chunk_size)) {
        for (o, &x) in out_chunk.iter_mut().zip(in_chunk.iter()) {
            let e = fast_exp_scalar(x - global_max);
            *o = e;
            total_sum += e;
        }
    }

    // Phase 3: normalise.
    if total_sum == 0.0 {
        let uniform = 1.0 / len as f32;
        for o in output.iter_mut() {
            *o = uniform;
        }
        return;
    }
    let inv = 1.0 / total_sum;
    for o in output.iter_mut() {
        *o *= inv;
    }
}

// ════════════════════════════════════════════════════════════════════════
// 6. temperature_softmax — softmax with temperature scaling
// ════════════════════════════════════════════════════════════════════════

/// Softmax with temperature scaling: `softmax(x / T)`.
///
/// Temperature > 1.0 flattens the distribution; < 1.0 sharpens it.
/// Temperature of 0.0 is treated as greedy (argmax gets 1.0).
pub fn temperature_softmax(input: &[f32], temperature: f32, output: &mut [f32]) {
    assert_eq!(input.len(), output.len(), "input/output length mismatch");
    if input.is_empty() {
        return;
    }

    // Greedy: temperature ≈ 0.
    if temperature.abs() < 1e-8 {
        for o in output.iter_mut() {
            *o = 0.0;
        }
        let mut max_val = f32::NEG_INFINITY;
        let mut max_idx = 0;
        for (i, &x) in input.iter().enumerate() {
            if x > max_val {
                max_val = x;
                max_idx = i;
            }
        }
        output[max_idx] = 1.0;
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                temperature_softmax_neon(input, temperature, output);
            }
            return;
        }
    }

    temperature_softmax_scalar(input, temperature, output);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn temperature_softmax_neon(input: &[f32], temperature: f32, output: &mut [f32]) {
    let len = input.len();
    let ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let chunks = len / LANES;
    let tail = len % LANES;
    let inv_temp = 1.0 / temperature;
    let inv_temp_splat = unsafe { vdupq_n_f32(inv_temp) };

    // Pass 1: find max of scaled values.
    let mut max_vec = unsafe { vdupq_n_f32(f32::NEG_INFINITY) };
    for i in 0..chunks {
        unsafe {
            let v = vld1q_f32(ptr.add(i * LANES));
            let scaled = vmulq_f32(v, inv_temp_splat);
            max_vec = vmaxq_f32(max_vec, scaled);
        }
    }
    let mut max_val = unsafe { hmax_f32(max_vec) };
    for i in (chunks * LANES)..len {
        max_val = max_val.max(unsafe { *input.get_unchecked(i) } * inv_temp);
    }

    // Pass 2: exp((x / T) - max) and accumulate sum.
    let max_splat = unsafe { vdupq_n_f32(max_val) };
    let mut sum_vec = unsafe { vdupq_n_f32(0.0) };
    for i in 0..chunks {
        unsafe {
            let v = vld1q_f32(ptr.add(i * LANES));
            let scaled = vmulq_f32(v, inv_temp_splat);
            let shifted = vsubq_f32(scaled, max_splat);
            let e = fast_exp_neon(shifted);
            vst1q_f32(out_ptr.add(i * LANES), e);
            sum_vec = vaddq_f32(sum_vec, e);
        }
    }
    let mut sum_val = unsafe { hsum_f32(sum_vec) };
    for i in 0..tail {
        let idx = chunks * LANES + i;
        let e = fast_exp_scalar(unsafe { *input.get_unchecked(idx) } * inv_temp - max_val);
        unsafe { *output.get_unchecked_mut(idx) = e };
        sum_val += e;
    }

    // Pass 3: normalise.
    if sum_val == 0.0 {
        let uniform = 1.0 / len as f32;
        for o in output.iter_mut() {
            *o = uniform;
        }
        return;
    }
    let inv_sum = 1.0 / sum_val;
    let inv_splat = unsafe { vdupq_n_f32(inv_sum) };
    for i in 0..chunks {
        unsafe {
            let v = vld1q_f32(out_ptr.add(i * LANES));
            vst1q_f32(out_ptr.add(i * LANES), vmulq_f32(v, inv_splat));
        }
    }
    for i in 0..tail {
        let idx = chunks * LANES + i;
        unsafe { *output.get_unchecked_mut(idx) *= inv_sum };
    }
}

/// Scalar fallback for temperature-scaled softmax.
pub fn temperature_softmax_scalar(input: &[f32], temperature: f32, output: &mut [f32]) {
    assert_eq!(input.len(), output.len(), "input/output length mismatch");
    if input.is_empty() {
        return;
    }

    if temperature.abs() < 1e-8 {
        for o in output.iter_mut() {
            *o = 0.0;
        }
        let mut max_val = f32::NEG_INFINITY;
        let mut max_idx = 0;
        for (i, &x) in input.iter().enumerate() {
            if x > max_val {
                max_val = x;
                max_idx = i;
            }
        }
        output[max_idx] = 1.0;
        return;
    }

    let inv_temp = 1.0 / temperature;
    let max_val = input
        .iter()
        .map(|&x| x * inv_temp)
        .fold(f32::NEG_INFINITY, f32::max);

    let mut sum = 0.0f32;
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        let e = fast_exp_scalar(x * inv_temp - max_val);
        *o = e;
        sum += e;
    }

    if sum == 0.0 {
        let uniform = 1.0 / input.len() as f32;
        for o in output.iter_mut() {
            *o = uniform;
        }
        return;
    }
    let inv = 1.0 / sum;
    for o in output.iter_mut() {
        *o *= inv;
    }
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference (naive) softmax for correctness checking.
    fn naive_softmax(input: &[f32]) -> Vec<f32> {
        if input.is_empty() {
            return vec![];
        }
        let max = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = input.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|&e| e / sum).collect()
    }

    /// Reference log-softmax.
    fn naive_log_softmax(input: &[f32]) -> Vec<f32> {
        if input.is_empty() {
            return vec![];
        }
        let max = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = input.iter().map(|&x| (x - max).exp()).sum();
        let log_sum = sum.ln();
        input.iter().map(|&x| x - max - log_sum).collect()
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32, msg: &str) {
        assert_eq!(a.len(), b.len(), "{msg}: length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < tol,
                "{msg}: index {i}: {x} vs {y} (diff {})",
                (x - y).abs()
            );
        }
    }

    fn sums_to_one(v: &[f32], tol: f32) {
        let s: f32 = v.iter().sum();
        assert!(
            (s - 1.0).abs() < tol,
            "sum = {s}, expected 1.0 (diff {})",
            (s - 1.0).abs()
        );
    }

    fn all_non_negative(v: &[f32]) {
        for (i, x) in v.iter().enumerate() {
            assert!(*x >= 0.0, "index {i}: {x} < 0");
        }
    }

    // ── online_softmax_v2 tests ────────────────────────────────────────

    #[test]
    fn test_softmax_v2_basic() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        online_softmax_v2(&input, &mut output);
        let expected = naive_softmax(&input);
        assert_close(&output, &expected, 1e-3, "basic softmax");
    }

    #[test]
    fn test_softmax_v2_single_element() {
        let input = [42.0];
        let mut output = vec![0.0; 1];
        online_softmax_v2(&input, &mut output);
        assert!((output[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_v2_two_elements() {
        let input = [0.0, 0.0];
        let mut output = vec![0.0; 2];
        online_softmax_v2(&input, &mut output);
        assert_close(&output, &[0.5, 0.5], 1e-5, "equal elements");
    }

    #[test]
    fn test_softmax_v2_empty() {
        let input: [f32; 0] = [];
        let mut output: Vec<f32> = vec![];
        online_softmax_v2(&input, &mut output);
        assert!(output.is_empty());
    }

    #[test]
    fn test_softmax_v2_large_values() {
        let input = [1000.0, 1001.0, 1002.0, 999.0];
        let mut output = vec![0.0; 4];
        online_softmax_v2(&input, &mut output);
        sums_to_one(&output, 1e-3);
        all_non_negative(&output);
    }

    #[test]
    fn test_softmax_v2_negative_values() {
        let input = [-1.0, -2.0, -3.0, -4.0, -5.0];
        let mut output = vec![0.0; 5];
        online_softmax_v2(&input, &mut output);
        let expected = naive_softmax(&input);
        assert_close(&output, &expected, 1e-3, "negative values");
    }

    #[test]
    fn test_softmax_v2_mixed_values() {
        let input = [-10.0, 0.0, 10.0, 20.0, -20.0, 5.0, 15.0];
        let mut output = vec![0.0; 7];
        online_softmax_v2(&input, &mut output);
        sums_to_one(&output, 1e-3);
        all_non_negative(&output);
    }

    #[test]
    fn test_softmax_v2_non_multiple_of_4() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut output = vec![0.0; 7];
        online_softmax_v2(&input, &mut output);
        let expected = naive_softmax(&input);
        assert_close(&output, &expected, 1e-3, "7 elements");
    }

    #[test]
    fn test_softmax_v2_exactly_4() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        online_softmax_v2(&input, &mut output);
        sums_to_one(&output, 1e-3);
    }

    #[test]
    fn test_softmax_v2_scalar_matches_dispatch() {
        let input = [1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0, 9.0];
        let mut out_dispatch = vec![0.0; input.len()];
        let mut out_scalar = vec![0.0; input.len()];
        online_softmax_v2(&input, &mut out_dispatch);
        online_softmax_v2_scalar(&input, &mut out_scalar);
        assert_close(&out_dispatch, &out_scalar, 1e-3, "dispatch vs scalar");
    }

    #[test]
    fn test_softmax_v2_identical_values() {
        let input = [5.0; 8];
        let mut output = vec![0.0; 8];
        online_softmax_v2(&input, &mut output);
        let expected = 1.0 / 8.0;
        for o in &output {
            assert!((*o - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn test_softmax_v2_large_array() {
        let n = 1024;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let mut output = vec![0.0; n];
        online_softmax_v2(&input, &mut output);
        sums_to_one(&output, 1e-2);
        all_non_negative(&output);
    }

    #[test]
    fn test_softmax_v2_very_large_spread() {
        let input = [-80.0, 0.0, 80.0];
        let mut output = vec![0.0; 3];
        online_softmax_v2(&input, &mut output);
        assert!(output[2] > 0.99);
        sums_to_one(&output, 1e-3);
    }

    #[test]
    fn test_softmax_v2_monotonicity() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut output = vec![0.0; 5];
        online_softmax_v2(&input, &mut output);
        for i in 1..output.len() {
            assert!(output[i] >= output[i - 1], "not monotone at {i}");
        }
    }

    // ── online_log_softmax tests ───────────────────────────────────────

    #[test]
    fn test_log_softmax_basic() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        online_log_softmax(&input, &mut output);
        let expected = naive_log_softmax(&input);
        assert_close(&output, &expected, 1e-3, "log softmax basic");
    }

    #[test]
    fn test_log_softmax_single() {
        let input = [42.0];
        let mut output = vec![0.0; 1];
        online_log_softmax(&input, &mut output);
        assert!(output[0].abs() < 1e-5, "log(1.0) = 0");
    }

    #[test]
    fn test_log_softmax_all_negative() {
        let input = [-1.0, -2.0, -3.0, -4.0, -5.0];
        let mut output = vec![0.0; 5];
        online_log_softmax(&input, &mut output);
        for o in &output {
            assert!(*o <= 0.0, "log softmax must be ≤ 0, got {o}");
        }
    }

    #[test]
    fn test_log_softmax_exp_sums_to_one() {
        let input = [2.0, 4.0, 6.0, 8.0, 10.0];
        let mut output = vec![0.0; 5];
        online_log_softmax(&input, &mut output);
        let sum_exp: f32 = output.iter().map(|x| x.exp()).sum();
        assert!((sum_exp - 1.0).abs() < 1e-3, "exp(log_softmax) should sum to 1");
    }

    #[test]
    fn test_log_softmax_scalar_matches_dispatch() {
        let input = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0];
        let mut out_dispatch = vec![0.0; input.len()];
        let mut out_scalar = vec![0.0; input.len()];
        online_log_softmax(&input, &mut out_dispatch);
        online_log_softmax_scalar(&input, &mut out_scalar);
        assert_close(&out_dispatch, &out_scalar, 1e-3, "log softmax dispatch vs scalar");
    }

    #[test]
    fn test_log_softmax_large_values() {
        let input = [500.0, 501.0, 502.0];
        let mut output = vec![0.0; 3];
        online_log_softmax(&input, &mut output);
        for o in &output {
            assert!(o.is_finite(), "log softmax should be finite");
        }
    }

    #[test]
    fn test_log_softmax_non_multiple_of_4() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut output = vec![0.0; 5];
        online_log_softmax(&input, &mut output);
        let expected = naive_log_softmax(&input);
        assert_close(&output, &expected, 1e-3, "log softmax 5 elements");
    }

    #[test]
    fn test_log_softmax_empty() {
        let input: [f32; 0] = [];
        let mut output: Vec<f32> = vec![];
        online_log_softmax(&input, &mut output);
        assert!(output.is_empty());
    }

    // ── fused_softmax_mask tests ───────────────────────────────────────

    #[test]
    fn test_fused_mask_all_valid() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mask = [true, true, true, true];
        let mut output = vec![0.0; 4];
        fused_softmax_mask(&input, &mask, &mut output);
        let expected = naive_softmax(&input);
        assert_close(&output, &expected, 1e-3, "all valid mask");
    }

    #[test]
    fn test_fused_mask_partial() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mask = [true, false, true, false];
        let mut output = vec![0.0; 4];
        fused_softmax_mask(&input, &mask, &mut output);
        assert!((output[1]).abs() < 1e-6, "masked position should be 0");
        assert!((output[3]).abs() < 1e-6, "masked position should be 0");
        let valid_sum = output[0] + output[2];
        assert!((valid_sum - 1.0).abs() < 1e-3, "valid positions should sum to 1");
    }

    #[test]
    fn test_fused_mask_all_masked() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mask = [false, false, false, false];
        let mut output = vec![0.0; 4];
        fused_softmax_mask(&input, &mask, &mut output);
        for o in &output {
            assert!((*o).abs() < 1e-6, "all masked → all zero");
        }
    }

    #[test]
    fn test_fused_mask_single_valid() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mask = [false, false, true, false];
        let mut output = vec![0.0; 4];
        fused_softmax_mask(&input, &mask, &mut output);
        assert!((output[2] - 1.0).abs() < 1e-5);
        assert!(output[0].abs() < 1e-6);
        assert!(output[1].abs() < 1e-6);
        assert!(output[3].abs() < 1e-6);
    }

    #[test]
    fn test_fused_mask_scalar_matches_dispatch() {
        let input = [1.0, 3.0, 5.0, 7.0, 2.0, 4.0];
        let mask = [true, false, true, true, false, true];
        let mut out_dispatch = vec![0.0; input.len()];
        let mut out_scalar = vec![0.0; input.len()];
        fused_softmax_mask(&input, &mask, &mut out_dispatch);
        fused_softmax_mask_scalar(&input, &mask, &mut out_scalar);
        assert_close(&out_dispatch, &out_scalar, 1e-3, "mask dispatch vs scalar");
    }

    #[test]
    fn test_fused_mask_non_multiple_of_4() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mask = [true, true, false, true, true, false, true];
        let mut output = vec![0.0; 7];
        fused_softmax_mask(&input, &mask, &mut output);
        assert!(output[2].abs() < 1e-6);
        assert!(output[5].abs() < 1e-6);
        let valid_sum: f32 = output.iter().sum();
        assert!((valid_sum - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_fused_mask_causal_pattern() {
        let input = [0.5, 0.3, 0.8, 0.1];
        let mask = [true, true, true, false];
        let mut output = vec![0.0; 4];
        fused_softmax_mask(&input, &mask, &mut output);
        assert!(output[3].abs() < 1e-6, "future position masked");
        let valid_sum = output[0] + output[1] + output[2];
        assert!((valid_sum - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_fused_mask_empty() {
        let input: [f32; 0] = [];
        let mask: [bool; 0] = [];
        let mut output: Vec<f32> = vec![];
        fused_softmax_mask(&input, &mask, &mut output);
        assert!(output.is_empty());
    }

    // ── top_k_softmax tests ────────────────────────────────────────────

    #[test]
    fn test_top_k_all() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        top_k_softmax(&input, 4, &mut output);
        let expected = naive_softmax(&input);
        assert_close(&output, &expected, 1e-3, "top-k all");
    }

    #[test]
    fn test_top_k_one() {
        let input = [1.0, 5.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        top_k_softmax(&input, 1, &mut output);
        assert!((output[1] - 1.0).abs() < 1e-5, "top-1 should be 1.0");
        assert!(output[0].abs() < 1e-6);
        assert!(output[2].abs() < 1e-6);
        assert!(output[3].abs() < 1e-6);
    }

    #[test]
    fn test_top_k_two() {
        let input = [1.0, 5.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        top_k_softmax(&input, 2, &mut output);
        assert!(output[0].abs() < 1e-6, "not in top-2");
        assert!(output[2].abs() < 1e-6, "not in top-2");
        let valid_sum = output[1] + output[3];
        assert!((valid_sum - 1.0).abs() < 1e-3);
        assert!(output[1] > output[3], "5.0 should have higher prob than 4.0");
    }

    #[test]
    fn test_top_k_zero() {
        let input = [1.0, 2.0, 3.0];
        let mut output = vec![0.0; 3];
        top_k_softmax(&input, 0, &mut output);
        for o in &output {
            assert!(o.abs() < 1e-6);
        }
    }

    #[test]
    fn test_top_k_exceeds_len() {
        let input = [1.0, 2.0, 3.0];
        let mut output = vec![0.0; 3];
        top_k_softmax(&input, 100, &mut output);
        let expected = naive_softmax(&input);
        assert_close(&output, &expected, 1e-3, "k > len");
    }

    #[test]
    fn test_top_k_sums_to_one() {
        let input: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let mut output = vec![0.0; 20];
        top_k_softmax(&input, 5, &mut output);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3, "top-k softmax should sum to 1");
    }

    #[test]
    fn test_top_k_non_selected_are_zero() {
        let input = [10.0, 1.0, 2.0, 3.0, 9.0, 4.0, 8.0, 5.0];
        let mut output = vec![0.0; 8];
        top_k_softmax(&input, 3, &mut output);
        let zero_indices = [1, 2, 3, 5, 7];
        for &i in &zero_indices {
            assert!(output[i].abs() < 1e-6, "idx {i} not in top-3");
        }
    }

    // ── chunked_softmax tests ──────────────────────────────────────────

    #[test]
    fn test_chunked_matches_regular() {
        let input: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
        let mut out_regular = vec![0.0; 64];
        let mut out_chunked = vec![0.0; 64];
        online_softmax_v2(&input, &mut out_regular);
        chunked_softmax_with_size(&input, &mut out_chunked, 16);
        assert_close(&out_chunked, &out_regular, 1e-3, "chunked vs regular");
    }

    #[test]
    fn test_chunked_small_chunk() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        chunked_softmax_with_size(&input, &mut output, 2);
        let expected = naive_softmax(&input);
        assert_close(&output, &expected, 1e-3, "chunk size 2");
    }

    #[test]
    fn test_chunked_single_chunk() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        chunked_softmax_with_size(&input, &mut output, 100);
        let expected = naive_softmax(&input);
        assert_close(&output, &expected, 1e-3, "single chunk");
    }

    #[test]
    fn test_chunked_chunk_size_1() {
        let input = [1.0, 2.0, 3.0];
        let mut output = vec![0.0; 3];
        chunked_softmax_with_size(&input, &mut output, 1);
        let expected = naive_softmax(&input);
        assert_close(&output, &expected, 1e-3, "chunk size 1");
    }

    #[test]
    fn test_chunked_large_array() {
        let n = 8192;
        let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.001).sin()).collect();
        let mut output = vec![0.0; n];
        chunked_softmax(&input, &mut output);
        sums_to_one(&output, 1e-2);
        all_non_negative(&output);
    }

    #[test]
    fn test_chunked_scalar_matches_dispatch() {
        let input: Vec<f32> = (0..100).map(|i| (i as f32) * 0.05).collect();
        let mut out_dispatch = vec![0.0; 100];
        let mut out_scalar = vec![0.0; 100];
        chunked_softmax_with_size(&input, &mut out_dispatch, 32);
        chunked_softmax_scalar(&input, &mut out_scalar, 32);
        assert_close(&out_dispatch, &out_scalar, 1e-3, "chunked dispatch vs scalar");
    }

    #[test]
    fn test_chunked_empty() {
        let input: [f32; 0] = [];
        let mut output: Vec<f32> = vec![];
        chunked_softmax(&input, &mut output);
        assert!(output.is_empty());
    }

    #[test]
    fn test_chunked_default_chunk() {
        let n = 256;
        let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut output = vec![0.0; n];
        chunked_softmax(&input, &mut output);
        sums_to_one(&output, 1e-2);
    }

    // ── temperature_softmax tests ──────────────────────────────────────

    #[test]
    fn test_temperature_one() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        temperature_softmax(&input, 1.0, &mut output);
        let expected = naive_softmax(&input);
        assert_close(&output, &expected, 1e-3, "temperature 1.0");
    }

    #[test]
    fn test_temperature_high_flattens() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut out_t1 = vec![0.0; 4];
        let mut out_t10 = vec![0.0; 4];
        temperature_softmax(&input, 1.0, &mut out_t1);
        temperature_softmax(&input, 10.0, &mut out_t10);
        let max_t1 = out_t1.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let max_t10 = out_t10.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(max_t10 < max_t1, "high temp should flatten");
        sums_to_one(&out_t10, 1e-3);
    }

    #[test]
    fn test_temperature_low_sharpens() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut out_t1 = vec![0.0; 4];
        let mut out_t01 = vec![0.0; 4];
        temperature_softmax(&input, 1.0, &mut out_t1);
        temperature_softmax(&input, 0.1, &mut out_t01);
        let max_t1 = out_t1.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let max_t01 = out_t01.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(max_t01 > max_t1, "low temp should sharpen");
        sums_to_one(&out_t01, 1e-3);
    }

    #[test]
    fn test_temperature_zero_greedy() {
        let input = [1.0, 5.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        temperature_softmax(&input, 0.0, &mut output);
        assert!((output[1] - 1.0).abs() < 1e-6, "argmax gets 1.0");
        assert!(output[0].abs() < 1e-6);
        assert!(output[2].abs() < 1e-6);
        assert!(output[3].abs() < 1e-6);
    }

    #[test]
    fn test_temperature_negative_near_zero() {
        let input = [1.0, 2.0, 3.0];
        let mut output = vec![0.0; 3];
        temperature_softmax(&input, 1e-10, &mut output);
        assert!((output[2] - 1.0).abs() < 1e-6, "near-zero → greedy");
    }

    #[test]
    fn test_temperature_scalar_matches_dispatch() {
        let input = [1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0, 9.0];
        let mut out_dispatch = vec![0.0; input.len()];
        let mut out_scalar = vec![0.0; input.len()];
        temperature_softmax(&input, 0.5, &mut out_dispatch);
        temperature_softmax_scalar(&input, 0.5, &mut out_scalar);
        assert_close(&out_dispatch, &out_scalar, 1e-3, "temp dispatch vs scalar");
    }

    #[test]
    fn test_temperature_sums_to_one() {
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
        for &temp in &[0.1, 0.5, 1.0, 2.0, 5.0, 10.0] {
            let mut output = vec![0.0; 16];
            temperature_softmax(&input, temp, &mut output);
            sums_to_one(&output, 1e-3);
        }
    }

    #[test]
    fn test_temperature_empty() {
        let input: [f32; 0] = [];
        let mut output: Vec<f32> = vec![];
        temperature_softmax(&input, 1.0, &mut output);
        assert!(output.is_empty());
    }

    // ── Cross-function consistency tests ───────────────────────────────

    #[test]
    fn test_log_softmax_consistent_with_softmax() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut softmax_out = vec![0.0; 8];
        let mut log_softmax_out = vec![0.0; 8];
        online_softmax_v2(&input, &mut softmax_out);
        online_log_softmax(&input, &mut log_softmax_out);
        for i in 0..8 {
            let log_of_softmax = softmax_out[i].ln();
            assert!(
                (log_of_softmax - log_softmax_out[i]).abs() < 1e-2,
                "index {i}: ln(softmax) = {log_of_softmax}, log_softmax = {}",
                log_softmax_out[i]
            );
        }
    }

    #[test]
    fn test_softmax_v2_matches_chunked() {
        let input: Vec<f32> = (0..128).map(|i| (i as f32 * 0.03).sin()).collect();
        let mut out_v2 = vec![0.0; 128];
        let mut out_chunked = vec![0.0; 128];
        online_softmax_v2(&input, &mut out_v2);
        chunked_softmax_with_size(&input, &mut out_chunked, 32);
        assert_close(&out_v2, &out_chunked, 1e-3, "v2 vs chunked");
    }

    #[test]
    fn test_mask_all_true_matches_regular() {
        let input = [2.0, 4.0, 6.0, 8.0, 10.0];
        let mask = [true; 5];
        let mut out_masked = vec![0.0; 5];
        let mut out_regular = vec![0.0; 5];
        fused_softmax_mask(&input, &mask, &mut out_masked);
        online_softmax_v2(&input, &mut out_regular);
        assert_close(&out_masked, &out_regular, 1e-3, "all-true mask vs regular");
    }

    #[test]
    fn test_temperature_one_matches_regular() {
        let input = [2.0, 4.0, 6.0, 8.0, 10.0];
        let mut out_temp = vec![0.0; 5];
        let mut out_regular = vec![0.0; 5];
        temperature_softmax(&input, 1.0, &mut out_temp);
        online_softmax_v2(&input, &mut out_regular);
        assert_close(&out_temp, &out_regular, 1e-3, "temp=1 vs regular");
    }

    // ── Numerical stability edge cases ─────────────────────────────────

    #[test]
    fn test_stability_all_same_large() {
        let input = [1e30; 8];
        let mut output = vec![0.0; 8];
        online_softmax_v2(&input, &mut output);
        let expected = 1.0 / 8.0;
        for o in &output {
            assert!((*o - expected).abs() < 1e-5, "all same large");
        }
    }

    #[test]
    fn test_stability_all_same_negative_large() {
        let input = [-1e30; 8];
        let mut output = vec![0.0; 8];
        online_softmax_v2(&input, &mut output);
        let expected = 1.0 / 8.0;
        for o in &output {
            assert!((*o - expected).abs() < 1e-5, "all same negative large");
        }
    }

    #[test]
    fn test_stability_no_nan_inf() {
        let input = [f32::MAX / 2.0, 0.0, f32::MIN / 2.0, 1.0];
        let mut output = vec![0.0; 4];
        online_softmax_v2(&input, &mut output);
        for o in &output {
            assert!(o.is_finite(), "output must be finite, got {o}");
            assert!(!o.is_nan(), "output must not be NaN");
        }
    }

    #[test]
    fn test_stability_log_softmax_no_nan() {
        let input = [1e10, -1e10, 0.0, 1.0];
        let mut output = vec![0.0; 4];
        online_log_softmax(&input, &mut output);
        for o in &output {
            assert!(o.is_finite(), "log softmax must be finite, got {o}");
        }
    }

    #[test]
    fn test_stability_temperature_extreme() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        temperature_softmax(&input, 100.0, &mut output);
        sums_to_one(&output, 1e-3);
        let expected = 0.25;
        for o in &output {
            assert!((*o - expected).abs() < 0.05, "very high temp → ~uniform");
        }
    }

    #[test]
    fn test_stability_large_vocab() {
        let n = 32000;
        let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.001).sin() * 10.0).collect();
        let mut output = vec![0.0; n];
        online_softmax_v2(&input, &mut output);
        sums_to_one(&output, 5e-2);
        all_non_negative(&output);
    }

    #[test]
    fn test_stability_very_large_vocab_chunked() {
        let n = 100_000;
        let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0001).cos() * 5.0).collect();
        let mut output = vec![0.0; n];
        chunked_softmax(&input, &mut output);
        sums_to_one(&output, 0.1);
        all_non_negative(&output);
    }
}

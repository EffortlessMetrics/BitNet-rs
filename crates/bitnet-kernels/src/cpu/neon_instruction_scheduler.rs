#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::let_and_return)]
//! ARM NEON instruction scheduling primitives for Apple Silicon pipeline optimization.
//!
//! Implements instruction reordering patterns that maximize NEON pipeline
//! utilization: interleaved loads, fused scale+bias, dual-issue accumulation,
//! pipelined reduction, and software prefetch with computation overlap.
//! Each operation processes data in 4-wide (`float32x4_t`) chunks with
//! scalar fallback for tail elements.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON lane count for `float32x4_t`.
const LANES: usize = 4;

/// Deinterleave two f32 streams using `vld2q_f32`.
///
/// Reads `data` as pairs of interleaved values with the given `stride` and
/// writes the two deinterleaved streams into `out_a` and `out_b`.
/// When `stride == 1`, consecutive pairs `(a0, b0, a1, b1, …)` are split.
///
/// # Panics
///
/// Panics if `out_a` or `out_b` is shorter than the number of produced elements.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_interleaved_load_f32(
    data: &[f32],
    stride: usize,
    out_a: &mut [f32],
    out_b: &mut [f32],
) {
    if data.is_empty() || stride == 0 {
        return;
    }

    // Number of (a, b) pairs we can extract.
    let pair_count = data.len() / (2 * stride);
    assert!(out_a.len() >= pair_count, "out_a too short");
    assert!(out_b.len() >= pair_count, "out_b too short");

    if stride == 1 {
        // Fast path — contiguous interleaved pairs, use vld2q_f32.
        let ptr = data.as_ptr();
        let chunks = pair_count / LANES;
        let remainder = pair_count % LANES;

        for i in 0..chunks {
            let base = i * LANES * 2;
            let pair = unsafe { vld2q_f32(ptr.add(base)) };
            unsafe {
                vst1q_f32(out_a.as_mut_ptr().add(i * LANES), pair.0);
                vst1q_f32(out_b.as_mut_ptr().add(i * LANES), pair.1);
            }
        }

        let scalar_start = chunks * LANES;
        for j in 0..remainder {
            let idx = (scalar_start + j) * 2;
            out_a[scalar_start + j] = data[idx];
            out_b[scalar_start + j] = data[idx + 1];
        }
    } else {
        // General stride — scalar path.
        for i in 0..pair_count {
            let base = i * 2 * stride;
            out_a[i] = data[base];
            out_b[i] = data[base + stride];
        }
    }
}

/// Fused scale and bias: `data[i] = data[i] * scale[i] + bias[i]` using `vfmaq_f32`.
///
/// Processes four elements at a time. `scale` and `bias` must be at least as
/// long as `data`.
///
/// # Panics
///
/// Panics if `scale` or `bias` is shorter than `data`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_fused_scale_bias_f32(data: &mut [f32], scale: &[f32], bias: &[f32]) {
    let len = data.len();
    assert!(scale.len() >= len, "scale too short");
    assert!(bias.len() >= len, "bias too short");

    if len == 0 {
        return;
    }

    let d_ptr = data.as_mut_ptr();
    let s_ptr = scale.as_ptr();
    let b_ptr = bias.as_ptr();

    let chunks = len / LANES;
    let remainder = len % LANES;

    for i in 0..chunks {
        let off = i * LANES;
        unsafe {
            let vd = vld1q_f32(d_ptr.add(off));
            let vs = vld1q_f32(s_ptr.add(off));
            let vb = vld1q_f32(b_ptr.add(off));
            // vfmaq_f32(a, b, c) = a + b * c  →  bias + data * scale
            let res = vfmaq_f32(vb, vd, vs);
            vst1q_f32(d_ptr.add(off), res);
        }
    }

    let tail_start = chunks * LANES;
    for j in 0..remainder {
        let idx = tail_start + j;
        data[idx] = data[idx] * scale[idx] + bias[idx];
    }
}

/// Dual-issue accumulate: `out[i] = a[i]*b[i] + c[i]*d[i]`.
///
/// Uses two back-to-back `vfmaq_f32` to expose instruction-level parallelism
/// on out-of-order NEON pipelines.
///
/// # Panics
///
/// Panics if any input slice is shorter than `out`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_dual_accumulate_f32(
    a: &[f32],
    b: &[f32],
    c: &[f32],
    d: &[f32],
    out: &mut [f32],
) {
    let len = out.len();
    assert!(a.len() >= len, "a too short");
    assert!(b.len() >= len, "b too short");
    assert!(c.len() >= len, "c too short");
    assert!(d.len() >= len, "d too short");

    if len == 0 {
        return;
    }

    let chunks = len / LANES;
    let remainder = len % LANES;

    for i in 0..chunks {
        let off = i * LANES;
        unsafe {
            let va = vld1q_f32(a.as_ptr().add(off));
            let vb = vld1q_f32(b.as_ptr().add(off));
            let vc = vld1q_f32(c.as_ptr().add(off));
            let vd = vld1q_f32(d.as_ptr().add(off));

            // First FMA: acc = 0 + a * b
            let acc1 = vmulq_f32(va, vb);
            // Second FMA: acc = acc1 + c * d  (dual-issue opportunity)
            let acc2 = vfmaq_f32(acc1, vc, vd);

            vst1q_f32(out.as_mut_ptr().add(off), acc2);
        }
    }

    let tail_start = chunks * LANES;
    for j in 0..remainder {
        let idx = tail_start + j;
        out[idx] = a[idx] * b[idx] + c[idx] * d[idx];
    }
}

/// Pipelined reduction that maximises throughput via multiple accumulators.
///
/// Uses four independent `float32x4_t` accumulators to hide FMA latency,
/// then collapses to a scalar with `vaddvq_f32`. Returns `0.0` for empty input.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_pipelined_reduce_f32(data: &[f32]) -> f32 {
    let len = data.len();
    if len == 0 {
        return 0.0;
    }

    let ptr = data.as_ptr();

    // Four accumulators for pipeline filling.
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);

    let unroll = LANES * 4; // 16 elements per iteration
    let main_iters = len / unroll;
    let after_main = main_iters * unroll;

    for i in 0..main_iters {
        let base = i * unroll;
        unsafe {
            acc0 = vaddq_f32(acc0, vld1q_f32(ptr.add(base)));
            acc1 = vaddq_f32(acc1, vld1q_f32(ptr.add(base + LANES)));
            acc2 = vaddq_f32(acc2, vld1q_f32(ptr.add(base + LANES * 2)));
            acc3 = vaddq_f32(acc3, vld1q_f32(ptr.add(base + LANES * 3)));
        }
    }

    // Handle remaining 4-wide chunks.
    let remaining = len - after_main;
    let extra_chunks = remaining / LANES;
    for i in 0..extra_chunks {
        unsafe {
            acc0 = vaddq_f32(acc0, vld1q_f32(ptr.add(after_main + i * LANES)));
        }
    }

    // Collapse accumulators.
    let t0 = vaddq_f32(acc0, acc1);
    let t1 = vaddq_f32(acc2, acc3);
    let combined = vaddq_f32(t0, t1);
    let mut sum = vaddvq_f32(combined);

    // Scalar tail.
    let scalar_start = after_main + extra_chunks * LANES;
    for i in scalar_start..len {
        sum += unsafe { *ptr.add(i) };
    }

    sum
}

/// Software prefetch with computation overlap.
///
/// Copies `data` into `out` while issuing prefetch hints `prefetch_offset`
/// elements ahead, giving the memory subsystem time to bring cache lines in
/// before they are needed.
///
/// # Panics
///
/// Panics if `out` is shorter than `data`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_prefetch_load_f32(data: &[f32], prefetch_offset: usize, out: &mut [f32]) {
    let len = data.len();
    assert!(out.len() >= len, "out too short");

    if len == 0 {
        return;
    }

    let src = data.as_ptr();
    let dst = out.as_mut_ptr();

    let chunks = len / LANES;
    let remainder = len % LANES;

    for i in 0..chunks {
        let off = i * LANES;

        // Issue prefetch for a future cache line via inline assembly.
        let pf_off = off + prefetch_offset;
        if pf_off < len {
            unsafe {
                let pf_ptr = src.add(pf_off);
                #[cfg(target_arch = "aarch64")]
                std::arch::asm!("prfm pldl1keep, [{ptr}]", ptr = in(reg) pf_ptr, options(nostack, preserves_flags));
            }
        }

        unsafe {
            let v = vld1q_f32(src.add(off));
            vst1q_f32(dst.add(off), v);
        }
    }

    let tail_start = chunks * LANES;
    for j in 0..remainder {
        let idx = tail_start + j;
        out[idx] = data[idx];
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to call unsafe neon functions inside tests.
    const EPSILON: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    // ── interleaved_load ────────────────────────────────────────────────

    #[test]
    fn interleaved_load_empty() {
        let data: [f32; 0] = [];
        let mut out_a = [0.0f32; 0];
        let mut out_b = [0.0f32; 0];
        unsafe {
            neon_interleaved_load_f32(&data, 1, &mut out_a, &mut out_b);
        }
    }

    #[test]
    fn interleaved_load_single_pair() {
        let data = [1.0f32, 2.0];
        let mut out_a = [0.0f32; 1];
        let mut out_b = [0.0f32; 1];
        unsafe {
            neon_interleaved_load_f32(&data, 1, &mut out_a, &mut out_b);
        }
        assert!(approx_eq(out_a[0], 1.0));
        assert!(approx_eq(out_b[0], 2.0));
    }

    #[test]
    fn interleaved_load_aligned_4_pairs() {
        // 4 pairs = 8 elements → exactly one NEON chunk
        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let mut out_a = [0.0f32; 4];
        let mut out_b = [0.0f32; 4];
        unsafe {
            neon_interleaved_load_f32(&data, 1, &mut out_a, &mut out_b);
        }
        assert_eq!(out_a, vec![0.0, 2.0, 4.0, 6.0]);
        assert_eq!(out_b, vec![1.0, 3.0, 5.0, 7.0]);
    }

    #[test]
    fn interleaved_load_unaligned_5_pairs() {
        let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let mut out_a = [0.0f32; 5];
        let mut out_b = [0.0f32; 5];
        unsafe {
            neon_interleaved_load_f32(&data, 1, &mut out_a, &mut out_b);
        }
        assert_eq!(out_a, vec![0.0, 2.0, 4.0, 6.0, 8.0]);
        assert_eq!(out_b, vec![1.0, 3.0, 5.0, 7.0, 9.0]);
    }

    #[test]
    fn interleaved_load_stride_2() {
        // stride=2: pairs at offsets (0, 2), (4, 6)
        let data = [10.0f32, 0.0, 20.0, 0.0, 30.0, 0.0, 40.0, 0.0];
        let mut out_a = [0.0f32; 2];
        let mut out_b = [0.0f32; 2];
        unsafe {
            neon_interleaved_load_f32(&data, 2, &mut out_a, &mut out_b);
        }
        assert!(approx_eq(out_a[0], 10.0));
        assert!(approx_eq(out_b[0], 20.0));
        assert!(approx_eq(out_a[1], 30.0));
        assert!(approx_eq(out_b[1], 40.0));
    }

    #[test]
    fn interleaved_load_stride_3() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out_a = [0.0f32; 1];
        let mut out_b = [0.0f32; 1];
        unsafe {
            neon_interleaved_load_f32(&data, 3, &mut out_a, &mut out_b);
        }
        assert!(approx_eq(out_a[0], 1.0));
        assert!(approx_eq(out_b[0], 4.0));
    }

    #[test]
    fn interleaved_load_large_data() {
        let n = 256;
        let data: Vec<f32> = (0..n * 2).map(|i| i as f32).collect();
        let mut out_a = vec![0.0f32; n];
        let mut out_b = vec![0.0f32; n];
        unsafe {
            neon_interleaved_load_f32(&data, 1, &mut out_a, &mut out_b);
        }
        for i in 0..n {
            assert!(approx_eq(out_a[i], (i * 2) as f32));
            assert!(approx_eq(out_b[i], (i * 2 + 1) as f32));
        }
    }

    #[test]
    fn interleaved_load_zeros() {
        let data = [0.0f32; 16];
        let mut out_a = [1.0f32; 8];
        let mut out_b = [1.0f32; 8];
        unsafe {
            neon_interleaved_load_f32(&data, 1, &mut out_a, &mut out_b);
        }
        assert!(out_a.iter().all(|&x| x == 0.0));
        assert!(out_b.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn interleaved_load_negative() {
        let data = [-1.0f32, -2.0, -3.0, -4.0];
        let mut out_a = [0.0f32; 2];
        let mut out_b = [0.0f32; 2];
        unsafe {
            neon_interleaved_load_f32(&data, 1, &mut out_a, &mut out_b);
        }
        assert!(approx_eq(out_a[0], -1.0));
        assert!(approx_eq(out_b[0], -2.0));
        assert!(approx_eq(out_a[1], -3.0));
        assert!(approx_eq(out_b[1], -4.0));
    }

    #[test]
    fn interleaved_load_mixed_sign() {
        let data = [1.0f32, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0];
        let mut out_a = [0.0f32; 4];
        let mut out_b = [0.0f32; 4];
        unsafe {
            neon_interleaved_load_f32(&data, 1, &mut out_a, &mut out_b);
        }
        assert_eq!(out_a, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(out_b, vec![-1.0, -2.0, -3.0, -4.0]);
    }

    #[test]
    fn interleaved_load_zero_stride_noop() {
        let data = [1.0f32, 2.0];
        let mut out_a = [99.0f32; 1];
        let mut out_b = [99.0f32; 1];
        unsafe {
            neon_interleaved_load_f32(&data, 0, &mut out_a, &mut out_b);
        }
        // stride=0 returns immediately — outputs unchanged.
        assert!(approx_eq(out_a[0], 99.0));
        assert!(approx_eq(out_b[0], 99.0));
    }

    // ── fused_scale_bias ────────────────────────────────────────────────

    #[test]
    fn fused_scale_bias_identity() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let scale = [1.0; 4];
        let bias = [0.0; 4];
        unsafe { neon_fused_scale_bias_f32(&mut data, &scale, &bias) };
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn fused_scale_bias_zero_bias() {
        let mut data = vec![2.0, 3.0, 4.0, 5.0];
        let scale = [2.0; 4];
        let bias = [0.0; 4];
        unsafe { neon_fused_scale_bias_f32(&mut data, &scale, &bias) };
        assert_eq!(data, vec![4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn fused_scale_bias_zero_scale() {
        let mut data = [100.0; 4];
        let scale = [0.0; 4];
        let bias = [5.0; 4];
        unsafe { neon_fused_scale_bias_f32(&mut data, &scale, &bias) };
        assert_eq!(data, vec![5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn fused_scale_bias_negative() {
        let mut data = vec![1.0, -1.0, 2.0, -2.0];
        let scale = [-1.0; 4];
        let bias = [0.0; 4];
        unsafe { neon_fused_scale_bias_f32(&mut data, &scale, &bias) };
        assert_eq!(data, vec![-1.0, 1.0, -2.0, 2.0]);
    }

    #[test]
    fn fused_scale_bias_large() {
        let n = 129; // not a multiple of 4
        let mut data = vec![2.0f32; n];
        let scale = vec![3.0f32; n];
        let bias = vec![1.0f32; n];
        unsafe { neon_fused_scale_bias_f32(&mut data, &scale, &bias) };
        for &v in &data {
            assert!(approx_eq(v, 7.0)); // 2*3+1
        }
    }

    #[test]
    fn fused_scale_bias_ones() {
        let mut data = [1.0; 8];
        let scale = [1.0; 8];
        let bias = [1.0; 8];
        unsafe { neon_fused_scale_bias_f32(&mut data, &scale, &bias) };
        for &v in &data {
            assert!(approx_eq(v, 2.0));
        }
    }

    #[test]
    fn fused_scale_bias_alternating() {
        let mut data = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let scale = [2.0; 8];
        let bias = [0.5; 8];
        unsafe { neon_fused_scale_bias_f32(&mut data, &scale, &bias) };
        let expected = vec![2.5, 0.5, 2.5, 0.5, 2.5, 0.5, 2.5, 0.5];
        assert_eq!(data, expected);
    }

    #[test]
    fn fused_scale_bias_precision() {
        let mut data = vec![0.1, 0.2, 0.3, 0.4];
        let scale = [10.0; 4];
        let bias = [0.0; 4];
        unsafe { neon_fused_scale_bias_f32(&mut data, &scale, &bias) };
        for (i, &v) in data.iter().enumerate() {
            let expected = (i + 1) as f32 * 0.1 * 10.0;
            assert!(approx_eq(v, expected));
        }
    }

    #[test]
    fn fused_scale_bias_mixed_sign() {
        let mut data = vec![1.0, -2.0, 3.0, -4.0, 5.0];
        let scale = vec![1.0, -1.0, 1.0, -1.0, 1.0];
        let bias = [10.0; 5];
        unsafe { neon_fused_scale_bias_f32(&mut data, &scale, &bias) };
        assert_eq!(data, vec![11.0, 12.0, 13.0, 14.0, 15.0]);
    }

    #[test]
    fn fused_scale_bias_empty() {
        let mut data: Vec<f32> = vec![];
        let scale: Vec<f32> = vec![];
        let bias: Vec<f32> = vec![];
        unsafe { neon_fused_scale_bias_f32(&mut data, &scale, &bias) };
        assert!(data.is_empty());
    }

    // ── dual_accumulate ─────────────────────────────────────────────────

    #[test]
    fn dual_accumulate_zeros() {
        let z = [0.0f32; 8];
        let mut out = [99.0f32; 8];
        unsafe { neon_dual_accumulate_f32(&z, &z, &z, &z, &mut out) };
        assert!(out.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn dual_accumulate_ones() {
        let o = [1.0f32; 8];
        let mut out = [0.0f32; 8];
        unsafe { neon_dual_accumulate_f32(&o, &o, &o, &o, &mut out) };
        // 1*1 + 1*1 = 2
        assert!(out.iter().all(|&x| approx_eq(x, 2.0)));
    }

    #[test]
    fn dual_accumulate_identity() {
        let a = [3.0f32; 4];
        let b = [1.0f32; 4];
        let c = [0.0f32; 4];
        let d = [0.0f32; 4];
        let mut out = [0.0f32; 4];
        unsafe { neon_dual_accumulate_f32(&a, &b, &c, &d, &mut out) };
        assert!(out.iter().all(|&x| approx_eq(x, 3.0)));
    }

    #[test]
    fn dual_accumulate_negative() {
        let a = [-1.0f32; 4];
        let b = [2.0f32; 4];
        let c = [3.0f32; 4];
        let d = [-1.0f32; 4];
        let mut out = [0.0f32; 4];
        unsafe { neon_dual_accumulate_f32(&a, &b, &c, &d, &mut out) };
        // -1*2 + 3*(-1) = -5
        assert!(out.iter().all(|&x| approx_eq(x, -5.0)));
    }

    #[test]
    fn dual_accumulate_large() {
        let n = 131; // prime, not multiple of 4
        let a = vec![2.0f32; n];
        let b = vec![3.0f32; n];
        let c = vec![4.0f32; n];
        let d = vec![5.0f32; n];
        let mut out = vec![0.0f32; n];
        unsafe { neon_dual_accumulate_f32(&a, &b, &c, &d, &mut out) };
        // 2*3 + 4*5 = 26
        assert!(out.iter().all(|&x| approx_eq(x, 26.0)));
    }

    #[test]
    fn dual_accumulate_alternating() {
        let a = vec![1.0, -1.0, 1.0, -1.0];
        let b = [1.0; 4];
        let c = [1.0; 4];
        let d = vec![1.0, -1.0, 1.0, -1.0];
        let mut out = [0.0f32; 4];
        unsafe { neon_dual_accumulate_f32(&a, &b, &c, &d, &mut out) };
        // a*b + c*d: [1+1, -1-1, 1+1, -1-1]
        assert_eq!(out, vec![2.0, -2.0, 2.0, -2.0]);
    }

    #[test]
    fn dual_accumulate_precision() {
        let a = [0.1f32; 4];
        let b = [0.2f32; 4];
        let c = [0.3f32; 4];
        let d = [0.4f32; 4];
        let mut out = [0.0f32; 4];
        unsafe { neon_dual_accumulate_f32(&a, &b, &c, &d, &mut out) };
        // 0.1*0.2 + 0.3*0.4 = 0.02 + 0.12 = 0.14
        for &v in &out {
            assert!(approx_eq(v, 0.14));
        }
    }

    #[test]
    fn dual_accumulate_overflow_safe() {
        let big = [1e18f32; 4];
        let one = [1.0f32; 4];
        let z = [0.0f32; 4];
        let mut out = [0.0f32; 4];
        unsafe { neon_dual_accumulate_f32(&big, &one, &z, &z, &mut out) };
        assert!(out.iter().all(|&x| approx_eq(x, 1e18)));
    }

    #[test]
    fn dual_accumulate_mixed() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let c = [0.5; 5];
        let d = [2.0; 5];
        let mut out = [0.0f32; 5];
        unsafe { neon_dual_accumulate_f32(&a, &b, &c, &d, &mut out) };
        // [5+1, 8+1, 9+1, 8+1, 5+1] = [6, 9, 10, 9, 6]
        assert_eq!(out, vec![6.0, 9.0, 10.0, 9.0, 6.0]);
    }

    #[test]
    fn dual_accumulate_empty() {
        let e: Vec<f32> = vec![];
        let mut out: Vec<f32> = vec![];
        unsafe { neon_dual_accumulate_f32(&e, &e, &e, &e, &mut out) };
        assert!(out.is_empty());
    }

    // ── pipelined_reduce ────────────────────────────────────────────────

    #[test]
    fn pipelined_reduce_empty() {
        let data: [f32; 0] = [];
        let result = unsafe { neon_pipelined_reduce_f32(&data) };
        assert!(approx_eq(result, 0.0));
    }

    #[test]
    fn pipelined_reduce_single() {
        let data = [42.0f32];
        let result = unsafe { neon_pipelined_reduce_f32(&data) };
        assert!(approx_eq(result, 42.0));
    }

    #[test]
    fn pipelined_reduce_pair() {
        let data = [1.0f32, 2.0];
        let result = unsafe { neon_pipelined_reduce_f32(&data) };
        assert!(approx_eq(result, 3.0));
    }

    #[test]
    fn pipelined_reduce_quad() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let result = unsafe { neon_pipelined_reduce_f32(&data) };
        assert!(approx_eq(result, 10.0));
    }

    #[test]
    fn pipelined_reduce_8_elements() {
        let data: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let result = unsafe { neon_pipelined_reduce_f32(&data) };
        assert!(approx_eq(result, 36.0));
    }

    #[test]
    fn pipelined_reduce_16_elements() {
        let data = [1.0f32; 16];
        let result = unsafe { neon_pipelined_reduce_f32(&data) };
        assert!(approx_eq(result, 16.0));
    }

    #[test]
    fn pipelined_reduce_32_elements() {
        let data = [0.5f32; 32];
        let result = unsafe { neon_pipelined_reduce_f32(&data) };
        assert!(approx_eq(result, 16.0));
    }

    #[test]
    fn pipelined_reduce_negative() {
        let data = [-1.0f32; 4];
        let result = unsafe { neon_pipelined_reduce_f32(&data) };
        assert!(approx_eq(result, -4.0));
    }

    #[test]
    fn pipelined_reduce_mixed() {
        let data = [1.0f32, -1.0, 2.0, -2.0];
        let result = unsafe { neon_pipelined_reduce_f32(&data) };
        assert!(approx_eq(result, 0.0));
    }

    #[test]
    fn pipelined_reduce_large() {
        let n = 1024;
        let data = vec![1.0f32; n];
        let result = unsafe { neon_pipelined_reduce_f32(&data) };
        assert!(approx_eq(result, n as f32));
    }

    #[test]
    fn pipelined_reduce_alternating() {
        let data: Vec<f32> = (0..64).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let result = unsafe { neon_pipelined_reduce_f32(&data) };
        assert!(approx_eq(result, 0.0));
    }

    #[test]
    fn pipelined_reduce_precision() {
        let data: Vec<f32> = (1..=100).map(|i| i as f32).collect();
        let result = unsafe { neon_pipelined_reduce_f32(&data) };
        let expected = 5050.0f32;
        assert!((result - expected).abs() < 1.0);
    }

    // ── prefetch_load ───────────────────────────────────────────────────

    #[test]
    fn prefetch_load_basic() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut out = [0.0f32; 8];
        unsafe { neon_prefetch_load_f32(&data, 8, &mut out) };
        assert_eq!(data, out);
    }

    #[test]
    fn prefetch_load_zero_offset() {
        let data = [10.0f32; 4];
        let mut out = [0.0f32; 4];
        unsafe { neon_prefetch_load_f32(&data, 0, &mut out) };
        assert_eq!(data, out);
    }

    #[test]
    fn prefetch_load_large_offset() {
        let data = [3.14f32; 8];
        let mut out = [0.0f32; 8];
        unsafe { neon_prefetch_load_f32(&data, 1000, &mut out) };
        assert_eq!(data, out);
    }

    #[test]
    fn prefetch_load_aligned() {
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut out = [0.0f32; 16];
        unsafe { neon_prefetch_load_f32(&data, 16, &mut out) };
        assert_eq!(data, out);
    }

    #[test]
    fn prefetch_load_unaligned() {
        let data: Vec<f32> = (0..7).map(|i| i as f32).collect();
        let mut out = [0.0f32; 7];
        unsafe { neon_prefetch_load_f32(&data, 4, &mut out) };
        assert_eq!(data, out);
    }

    #[test]
    fn prefetch_load_sequential() {
        let data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let mut out = [0.0f32; 32];
        unsafe { neon_prefetch_load_f32(&data, 4, &mut out) };
        assert_eq!(data, out);
    }

    #[test]
    fn prefetch_load_boundary() {
        let data = vec![f32::MAX, f32::MIN, 0.0, f32::EPSILON];
        let mut out = [0.0f32; 4];
        unsafe { neon_prefetch_load_f32(&data, 2, &mut out) };
        assert_eq!(data, out);
    }

    #[test]
    fn prefetch_load_empty() {
        let data: [f32; 0] = [];
        let mut out: [f32; 0] = [];
        unsafe { neon_prefetch_load_f32(&data, 4, &mut out) };
    }

    #[test]
    fn prefetch_load_ones() {
        let data = [1.0f32; 17];
        let mut out = [0.0f32; 17];
        unsafe { neon_prefetch_load_f32(&data, 8, &mut out) };
        assert_eq!(data, out);
    }

    #[test]
    fn prefetch_load_pattern() {
        let data: Vec<f32> = (0..20).map(|i| (i * i) as f32).collect();
        let mut out = [0.0f32; 20];
        unsafe { neon_prefetch_load_f32(&data, 4, &mut out) };
        assert_eq!(data, out);
    }

    // ── integration ─────────────────────────────────────────────────────

    #[test]
    fn integration_interleave_then_scale_bias() {
        // Deinterleave then fuse scale+bias on one stream.
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut out_a = [0.0f32; 8];
        let mut out_b = [0.0f32; 8];
        unsafe {
            neon_interleaved_load_f32(&data, 1, &mut out_a, &mut out_b);
        }
        let scale = [2.0f32; 8];
        let bias = [1.0f32; 8];
        unsafe {
            neon_fused_scale_bias_f32(&mut out_a, &scale, &bias);
        }
        // out_a was [0,2,4,6,8,10,12,14] → 0*2+1=1, 2*2+1=5, ...
        let expected = vec![1.0, 5.0, 9.0, 13.0, 17.0, 21.0, 25.0, 29.0];
        assert_eq!(out_a, expected);
    }

    #[test]
    fn integration_dual_accumulate_then_reduce() {
        let a = [1.0f32; 8];
        let b = [2.0f32; 8];
        let c = [3.0f32; 8];
        let d = [4.0f32; 8];
        let mut acc = [0.0f32; 8];
        unsafe { neon_dual_accumulate_f32(&a, &b, &c, &d, &mut acc) };
        // Each element = 1*2 + 3*4 = 14
        let sum = unsafe { neon_pipelined_reduce_f32(&acc) };
        assert!(approx_eq(sum, 14.0 * 8.0));
    }

    #[test]
    fn integration_prefetch_then_reduce() {
        let data: Vec<f32> = (1..=16).map(|i| i as f32).collect();
        let mut buf = [0.0f32; 16];
        unsafe { neon_prefetch_load_f32(&data, 8, &mut buf) };
        let sum = unsafe { neon_pipelined_reduce_f32(&buf) };
        assert!(approx_eq(sum, 136.0)); // sum 1..=16
    }

    #[test]
    fn integration_end_to_end_pipeline() {
        // Full pipeline: prefetch → deinterleave → scale+bias → dual-accumulate → reduce
        let raw: Vec<f32> = (0..32).map(|i| i as f32).collect();

        // 1. Prefetch load
        let mut fetched = [0.0f32; 32];
        unsafe { neon_prefetch_load_f32(&raw, 8, &mut fetched) };

        // 2. Deinterleave
        let mut stream_a = [0.0f32; 16];
        let mut stream_b = [0.0f32; 16];
        unsafe { neon_interleaved_load_f32(&fetched, 1, &mut stream_a, &mut stream_b) };

        // 3. Scale+bias on stream_a
        let scale = [1.0f32; 16];
        let bias = [0.5f32; 16];
        unsafe { neon_fused_scale_bias_f32(&mut stream_a, &scale, &bias) };

        // 4. Dual-accumulate: stream_a*1 + stream_b*1
        let ones = [1.0f32; 16];
        let mut combined = [0.0f32; 16];
        unsafe {
            neon_dual_accumulate_f32(&stream_a, &ones, &stream_b, &ones, &mut combined);
        }

        // 5. Reduce
        let total = unsafe { neon_pipelined_reduce_f32(&combined) };

        // Verify: stream_a = [0,2,4,...,30]*1+0.5 = [0.5,2.5,...,30.5]
        // stream_b = [1,3,5,...,31]
        // combined[i] = stream_a[i] + stream_b[i]
        let expected: f32 = (0..16)
            .map(|i| {
                let a = (i * 2) as f32 + 0.5;
                let b = (i * 2 + 1) as f32;
                a + b
            })
            .sum();
        assert!((total - expected).abs() < 1.0);
    }

    #[test]
    fn integration_scale_bias_then_reduce() {
        let mut data: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let scale = [2.0f32; 8];
        let bias = [-1.0f32; 8];
        unsafe { neon_fused_scale_bias_f32(&mut data, &scale, &bias) };
        let sum = unsafe { neon_pipelined_reduce_f32(&data) };
        // Each element: i*2 - 1 → 1,3,5,7,9,11,13,15 → sum = 64
        assert!(approx_eq(sum, 64.0));
    }

    #[test]
    fn pipelined_reduce_17_elements() {
        // 17 = 16 + 1 → exercises main loop + single-element tail
        let data: Vec<f32> = (1..=17).map(|i| i as f32).collect();
        let result = unsafe { neon_pipelined_reduce_f32(&data) };
        assert!(approx_eq(result, 153.0)); // 17*18/2
    }

    #[test]
    fn dual_accumulate_sequential_values() {
        let a: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let b: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let c = [0.0f32; 8];
        let d = [0.0f32; 8];
        let mut out = [0.0f32; 8];
        unsafe { neon_dual_accumulate_f32(&a, &b, &c, &d, &mut out) };
        // out[i] = (i+1)^2
        for i in 0..8 {
            let expected = ((i + 1) * (i + 1)) as f32;
            assert!(approx_eq(out[i], expected));
        }
    }

    #[test]
    fn interleaved_load_3_pairs() {
        // 3 pairs = 6 elements → not a multiple of 4, exercises tail
        let data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0f32];
        let mut out_a = [0.0f32; 3];
        let mut out_b = [0.0f32; 3];
        unsafe { neon_interleaved_load_f32(&data, 1, &mut out_a, &mut out_b) };
        assert_eq!(out_a, vec![10.0, 30.0, 50.0]);
        assert_eq!(out_b, vec![20.0, 40.0, 60.0]);
    }
}

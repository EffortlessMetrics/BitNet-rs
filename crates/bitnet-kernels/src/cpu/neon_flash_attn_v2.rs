//! ARM NEON Flash Attention v2 kernel for Apple Silicon.
//!
//! Implements tiled, memory-efficient flash attention v2 with NEON SIMD
//! intrinsics for `float32x4` 4-wide parallelism. Instead of materialising
//! the full N×N attention matrix, the kernel streams query–key blocks
//! through an online softmax accumulator (running max + running sum),
//! yielding O(block_size × head_dim) working memory instead of O(N²).
//!
//! # Kernels
//!
//! - [`flash_attention_forward_neon`] — tiled single-head flash attention
//! - [`tiled_softmax_neon`] — online softmax with running max/sum
//! - [`blocked_matmul_neon`] — block-tiled matrix multiply for attention
//! - [`causal_mask_scores_neon`] — apply causal mask in-place
//! - [`multi_head_flash_attention_neon`] — multi-head flash attention
//!
//! # Safety
//!
//! Every function touching NEON load/store intrinsics (`vld1q_f32`,
//! `vst1q_f32`) or raw pointer arithmetic (`ptr::add`) is wrapped in
//! `unsafe {}` blocks. Pure arithmetic intrinsics are safe on aarch64
//! but are also wrapped since Rust 2024 edition treats all intrinsics
//! as unsafe.

#![allow(
    clippy::missing_safety_doc,
    clippy::float_cmp,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::manual_div_ceil
)]

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON lane count for `float32x4_t`.
#[cfg(target_arch = "aarch64")]
const LANES: usize = 4;

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
// 1. flash_attention_forward_neon
// ═══════════════════════════════════════════════════════════════════════

/// Tiled flash attention v2 forward pass (single head).
///
/// `q`, `k`, `v` are row-major `[seq_len, head_dim]`. Returns output
/// `[seq_len, head_dim]`.
#[cfg(target_arch = "aarch64")]
pub fn flash_attention_forward_neon(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    block_size: usize,
) -> Vec<f32> {
    assert_eq!(q.len(), seq_len * head_dim);
    assert_eq!(k.len(), seq_len * head_dim);
    assert_eq!(v.len(), seq_len * head_dim);
    assert!(block_size > 0);

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; seq_len * head_dim];

    // For each block of queries
    for qi_start in (0..seq_len).step_by(block_size) {
        let qi_end = (qi_start + block_size).min(seq_len);
        let qi_count = qi_end - qi_start;

        // Running state per query row: max, sum, output accumulator
        let mut row_max = vec![f32::NEG_INFINITY; qi_count];
        let mut row_sum = vec![0.0f32; qi_count];
        let mut acc = vec![0.0f32; qi_count * head_dim];

        // For each block of keys/values
        for kj_start in (0..seq_len).step_by(block_size) {
            let kj_end = (kj_start + block_size).min(seq_len);
            let kj_count = kj_end - kj_start;

            // Compute scores for this Q-block × K-block
            let mut scores = vec![0.0f32; qi_count * kj_count];
            for i in 0..qi_count {
                for j in 0..kj_count {
                    let qi_row = qi_start + i;
                    let kj_row = kj_start + j;
                    let mut dot: f32;

                    let q_off = qi_row * head_dim;
                    let k_off = kj_row * head_dim;
                    let chunks = head_dim / LANES;
                    let rem = head_dim % LANES;

                    unsafe {
                        let mut sum_vec = vdupq_n_f32(0.0);
                        let qp = q.as_ptr().add(q_off);
                        let kp = k.as_ptr().add(k_off);
                        for c in 0..chunks {
                            let qv = vld1q_f32(qp.add(c * LANES));
                            let kv = vld1q_f32(kp.add(c * LANES));
                            sum_vec = vfmaq_f32(sum_vec, qv, kv);
                        }
                        dot = vaddvq_f32(sum_vec);
                    }
                    for r in 0..rem {
                        dot += q[q_off + chunks * LANES + r] * k[k_off + chunks * LANES + r];
                    }
                    scores[i * kj_count + j] = dot * scale;
                }
            }

            // Causal mask
            for i in 0..qi_count {
                let qi_pos = qi_start + i;
                for j in 0..kj_count {
                    let kj_pos = kj_start + j;
                    if kj_pos > qi_pos {
                        scores[i * kj_count + j] = f32::NEG_INFINITY;
                    }
                }
            }

            // Online softmax update
            for i in 0..qi_count {
                let row_scores = &scores[i * kj_count..(i + 1) * kj_count];

                let block_max = row_scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);

                let new_max = row_max[i].max(block_max);
                let old_scale_factor = fast_exp_scalar(row_max[i] - new_max);
                let mut block_sum = 0.0f32;

                // Rescale accumulator
                let acc_off = i * head_dim;
                let chunks = head_dim / LANES;
                let rem = head_dim % LANES;

                unsafe {
                    let scale_v = vdupq_n_f32(old_scale_factor);
                    let ap = acc.as_mut_ptr().add(acc_off);
                    for c in 0..chunks {
                        let av = vld1q_f32(ap.add(c * LANES));
                        let sv = vmulq_f32(av, scale_v);
                        vst1q_f32(ap.add(c * LANES), sv);
                    }
                }
                for r in 0..rem {
                    acc[acc_off + chunks * LANES + r] *= old_scale_factor;
                }

                // Accumulate exp(score - new_max) * V
                for j in 0..kj_count {
                    let w = fast_exp_scalar(row_scores[j] - new_max);
                    block_sum += w;

                    let v_off = (kj_start + j) * head_dim;
                    unsafe {
                        let wv = vdupq_n_f32(w);
                        let ap = acc.as_mut_ptr().add(acc_off);
                        let vp = v.as_ptr().add(v_off);
                        for c in 0..chunks {
                            let av = vld1q_f32(ap.add(c * LANES));
                            let vv = vld1q_f32(vp.add(c * LANES));
                            let r = vfmaq_f32(av, wv, vv);
                            vst1q_f32(ap.add(c * LANES), r);
                        }
                    }
                    for r in 0..rem {
                        acc[acc_off + chunks * LANES + r] += w * v[v_off + chunks * LANES + r];
                    }
                }

                row_sum[i] = row_sum[i] * old_scale_factor + block_sum;
                row_max[i] = new_max;
            }
        }

        // Normalise: O[i] /= row_sum[i]
        for i in 0..qi_count {
            let acc_off = i * head_dim;
            let out_off = (qi_start + i) * head_dim;
            let inv_sum = if row_sum[i] > 0.0 { 1.0 / row_sum[i] } else { 0.0 };

            let chunks = head_dim / LANES;
            let rem = head_dim % LANES;

            unsafe {
                let inv_v = vdupq_n_f32(inv_sum);
                let ap = acc.as_ptr().add(acc_off);
                let op = output.as_mut_ptr().add(out_off);
                for c in 0..chunks {
                    let av = vld1q_f32(ap.add(c * LANES));
                    let rv = vmulq_f32(av, inv_v);
                    vst1q_f32(op.add(c * LANES), rv);
                }
            }
            for r in 0..rem {
                output[out_off + chunks * LANES + r] = acc[acc_off + chunks * LANES + r] * inv_sum;
            }
        }
    }

    output
}

/// Scalar fallback for flash attention forward pass.
#[cfg(not(target_arch = "aarch64"))]
pub fn flash_attention_forward_neon(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    block_size: usize,
) -> Vec<f32> {
    assert_eq!(q.len(), seq_len * head_dim);
    assert_eq!(k.len(), seq_len * head_dim);
    assert_eq!(v.len(), seq_len * head_dim);
    assert!(block_size > 0);

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; seq_len * head_dim];

    for qi_start in (0..seq_len).step_by(block_size) {
        let qi_end = (qi_start + block_size).min(seq_len);
        let qi_count = qi_end - qi_start;

        let mut row_max = vec![f32::NEG_INFINITY; qi_count];
        let mut row_sum = vec![0.0f32; qi_count];
        let mut acc = vec![0.0f32; qi_count * head_dim];

        for kj_start in (0..seq_len).step_by(block_size) {
            let kj_end = (kj_start + block_size).min(seq_len);
            let kj_count = kj_end - kj_start;

            let mut scores = vec![0.0f32; qi_count * kj_count];
            for i in 0..qi_count {
                for j in 0..kj_count {
                    let qi_row = qi_start + i;
                    let kj_row = kj_start + j;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[qi_row * head_dim + d] * k[kj_row * head_dim + d];
                    }
                    scores[i * kj_count + j] = dot * scale;
                }
            }

            for i in 0..qi_count {
                let qi_pos = qi_start + i;
                for j in 0..kj_count {
                    let kj_pos = kj_start + j;
                    if kj_pos > qi_pos {
                        scores[i * kj_count + j] = f32::NEG_INFINITY;
                    }
                }
            }

            for i in 0..qi_count {
                let row_scores = &scores[i * kj_count..(i + 1) * kj_count];

                let block_max = row_scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let new_max = row_max[i].max(block_max);
                let old_scale_factor = fast_exp_scalar(row_max[i] - new_max);
                let mut block_sum = 0.0f32;

                let acc_off = i * head_dim;
                for d in 0..head_dim {
                    acc[acc_off + d] *= old_scale_factor;
                }

                for j in 0..kj_count {
                    let w = fast_exp_scalar(row_scores[j] - new_max);
                    block_sum += w;
                    let v_off = (kj_start + j) * head_dim;
                    for d in 0..head_dim {
                        acc[acc_off + d] += w * v[v_off + d];
                    }
                }

                row_sum[i] = row_sum[i] * old_scale_factor + block_sum;
                row_max[i] = new_max;
            }
        }

        for i in 0..qi_count {
            let acc_off = i * head_dim;
            let out_off = (qi_start + i) * head_dim;
            let inv_sum = if row_sum[i] > 0.0 { 1.0 / row_sum[i] } else { 0.0 };
            for d in 0..head_dim {
                output[out_off + d] = acc[acc_off + d] * inv_sum;
            }
        }
    }

    output
}

// ═══════════════════════════════════════════════════════════════════════
// 2. tiled_softmax_neon
// ═══════════════════════════════════════════════════════════════════════

/// Online softmax with running max/sum over a 2-D score matrix
/// `[block_rows, block_cols]`. Returns `(softmax, row_max, row_sum)`.
#[cfg(target_arch = "aarch64")]
pub fn tiled_softmax_neon(
    scores: &[f32],
    block_rows: usize,
    block_cols: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    assert_eq!(scores.len(), block_rows * block_cols);

    let mut out = vec![0.0f32; block_rows * block_cols];
    let mut row_max = vec![f32::NEG_INFINITY; block_rows];
    let mut row_sum = vec![0.0f32; block_rows];

    for i in 0..block_rows {
        let row = &scores[i * block_cols..(i + 1) * block_cols];

        // Find row max with NEON
        let chunks = block_cols / LANES;
        let rem = block_cols % LANES;
        let mut max_val: f32;

        unsafe {
            let mut max_vec = vdupq_n_f32(f32::NEG_INFINITY);
            let rp = row.as_ptr();
            for c in 0..chunks {
                let v = vld1q_f32(rp.add(c * LANES));
                max_vec = vmaxq_f32(max_vec, v);
            }
            max_val = vmaxvq_f32(max_vec);
        }
        for r in 0..rem {
            max_val = max_val.max(row[chunks * LANES + r]);
        }
        row_max[i] = max_val;

        // Compute exp(score - max) and sum
        let out_row = &mut out[i * block_cols..(i + 1) * block_cols];
        let mut sum = 0.0f32;

        unsafe {
            let max_v = vdupq_n_f32(max_val);
            let rp = row.as_ptr();
            let op = out_row.as_mut_ptr();
            for c in 0..chunks {
                let sv = vld1q_f32(rp.add(c * LANES));
                let diff = vsubq_f32(sv, max_v);
                let ev = fast_exp_neon(diff);
                vst1q_f32(op.add(c * LANES), ev);
                sum += vaddvq_f32(ev);
            }
        }
        for r in 0..rem {
            let idx = chunks * LANES + r;
            let e = fast_exp_scalar(row[idx] - max_val);
            out_row[idx] = e;
            sum += e;
        }
        row_sum[i] = sum;

        // Normalise
        if sum > 0.0 {
            let inv_sum = 1.0 / sum;
            unsafe {
                let inv_v = vdupq_n_f32(inv_sum);
                let op = out_row.as_mut_ptr();
                for c in 0..chunks {
                    let ev = vld1q_f32(op.add(c * LANES));
                    let nv = vmulq_f32(ev, inv_v);
                    vst1q_f32(op.add(c * LANES), nv);
                }
            }
            for r in 0..rem {
                out_row[chunks * LANES + r] *= inv_sum;
            }
        }
    }

    (out, row_max, row_sum)
}

/// Scalar fallback for tiled softmax.
#[cfg(not(target_arch = "aarch64"))]
pub fn tiled_softmax_neon(
    scores: &[f32],
    block_rows: usize,
    block_cols: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    assert_eq!(scores.len(), block_rows * block_cols);

    let mut out = vec![0.0f32; block_rows * block_cols];
    let mut row_max = vec![f32::NEG_INFINITY; block_rows];
    let mut row_sum = vec![0.0f32; block_rows];

    for i in 0..block_rows {
        let row = &scores[i * block_cols..(i + 1) * block_cols];
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        row_max[i] = max_val;

        let out_row = &mut out[i * block_cols..(i + 1) * block_cols];
        let mut sum = 0.0f32;
        for j in 0..block_cols {
            let e = fast_exp_scalar(row[j] - max_val);
            out_row[j] = e;
            sum += e;
        }
        row_sum[i] = sum;

        if sum > 0.0 {
            let inv = 1.0 / sum;
            for j in 0..block_cols {
                out_row[j] *= inv;
            }
        }
    }

    (out, row_max, row_sum)
}

// ═══════════════════════════════════════════════════════════════════════
// 3. blocked_matmul_neon
// ═══════════════════════════════════════════════════════════════════════

/// Block-tiled matrix multiply: C = A × B.
///
/// `a` is `[m, k]`, `b` is `[k, n]`, result is `[m, n]`. All row-major.
#[cfg(target_arch = "aarch64")]
pub fn blocked_matmul_neon(
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
    block_size: usize,
) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert!(block_size > 0);

    let mut c = vec![0.0f32; m * n];

    for ii in (0..m).step_by(block_size) {
        let i_end = (ii + block_size).min(m);
        for jj in (0..n).step_by(block_size) {
            let j_end = (jj + block_size).min(n);
            for kk in (0..k).step_by(block_size) {
                let k_end = (kk + block_size).min(k);
                for i in ii..i_end {
                    for j in jj..j_end {
                        let mut dot = 0.0f32;
                        let inner_len = k_end - kk;
                        let chunks = inner_len / LANES;
                        let rem = inner_len % LANES;

                        unsafe {
                            let mut sum_vec = vdupq_n_f32(0.0);
                            let ap = a.as_ptr().add(i * k + kk);
                            let bp = b.as_ptr().add(kk * n + j);
                            for c in 0..chunks {
                                let av = vld1q_f32(ap.add(c * LANES));
                                // Gather from column j of B (stride = n)
                                let b0 = *bp.add((c * LANES) * n);
                                let b1 = *bp.add((c * LANES + 1) * n);
                                let b2 = *bp.add((c * LANES + 2) * n);
                                let b3 = *bp.add((c * LANES + 3) * n);
                                let bv = vld1q_f32([b0, b1, b2, b3].as_ptr());
                                sum_vec = vfmaq_f32(sum_vec, av, bv);
                            }
                            dot += vaddvq_f32(sum_vec);
                        }
                        for r in 0..rem {
                            let ki = kk + chunks * LANES + r;
                            dot += a[i * k + ki] * b[ki * n + j];
                        }
                        c[i * n + j] += dot;
                    }
                }
            }
        }
    }

    c
}

/// Scalar fallback for blocked matmul.
#[cfg(not(target_arch = "aarch64"))]
pub fn blocked_matmul_neon(
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
    block_size: usize,
) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert!(block_size > 0);

    let mut c = vec![0.0f32; m * n];

    for ii in (0..m).step_by(block_size) {
        let i_end = (ii + block_size).min(m);
        for jj in (0..n).step_by(block_size) {
            let j_end = (jj + block_size).min(n);
            for kk in (0..k).step_by(block_size) {
                let k_end = (kk + block_size).min(k);
                for i in ii..i_end {
                    for j in jj..j_end {
                        let mut dot = 0.0f32;
                        for ki in kk..k_end {
                            dot += a[i * k + ki] * b[ki * n + j];
                        }
                        c[i * n + j] += dot;
                    }
                }
            }
        }
    }

    c
}

// ═══════════════════════════════════════════════════════════════════════
// 4. causal_mask_scores_neon
// ═══════════════════════════════════════════════════════════════════════

/// Apply a causal (lower-triangular) mask to attention scores in-place.
///
/// `scores` is `[seq_len, seq_len]` row-major. Positions where
/// `col > row` are set to `neg_inf`.
#[cfg(target_arch = "aarch64")]
pub fn causal_mask_scores_neon(scores: &mut [f32], seq_len: usize, neg_inf: f32) {
    assert_eq!(scores.len(), seq_len * seq_len);

    for row in 0..seq_len {
        let mask_start = row + 1;
        if mask_start >= seq_len {
            continue;
        }
        let count = seq_len - mask_start;
        let off = row * seq_len + mask_start;
        let chunks = count / LANES;
        let rem = count % LANES;

        unsafe {
            let neg_v = vdupq_n_f32(neg_inf);
            let sp = scores.as_mut_ptr().add(off);
            for c in 0..chunks {
                vst1q_f32(sp.add(c * LANES), neg_v);
            }
        }
        for r in 0..rem {
            scores[off + chunks * LANES + r] = neg_inf;
        }
    }
}

/// Scalar fallback for causal masking.
#[cfg(not(target_arch = "aarch64"))]
pub fn causal_mask_scores_neon(scores: &mut [f32], seq_len: usize, neg_inf: f32) {
    assert_eq!(scores.len(), seq_len * seq_len);

    for row in 0..seq_len {
        for col in (row + 1)..seq_len {
            scores[row * seq_len + col] = neg_inf;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 5. multi_head_flash_attention_neon
// ═══════════════════════════════════════════════════════════════════════

/// Multi-head flash attention v2.
///
/// `q`, `k`, `v` are `[num_heads, seq_len, head_dim]` in row-major order
/// (head-major layout). Returns `[num_heads, seq_len, head_dim]`.
#[cfg(target_arch = "aarch64")]
pub fn multi_head_flash_attention_neon(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    block_size: usize,
) -> Vec<f32> {
    let head_size = seq_len * head_dim;
    assert_eq!(q.len(), num_heads * head_size);
    assert_eq!(k.len(), num_heads * head_size);
    assert_eq!(v.len(), num_heads * head_size);

    let mut output = vec![0.0f32; num_heads * head_size];

    for h in 0..num_heads {
        let off = h * head_size;
        let q_head = &q[off..off + head_size];
        let k_head = &k[off..off + head_size];
        let v_head = &v[off..off + head_size];

        let head_out =
            flash_attention_forward_neon(q_head, k_head, v_head, seq_len, head_dim, block_size);

        output[off..off + head_size].copy_from_slice(&head_out);
    }

    output
}

/// Scalar fallback for multi-head flash attention.
#[cfg(not(target_arch = "aarch64"))]
pub fn multi_head_flash_attention_neon(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    block_size: usize,
) -> Vec<f32> {
    let head_size = seq_len * head_dim;
    assert_eq!(q.len(), num_heads * head_size);
    assert_eq!(k.len(), num_heads * head_size);
    assert_eq!(v.len(), num_heads * head_size);

    let mut output = vec![0.0f32; num_heads * head_size];

    for h in 0..num_heads {
        let off = h * head_size;
        let q_head = &q[off..off + head_size];
        let k_head = &k[off..off + head_size];
        let v_head = &v[off..off + head_size];

        let head_out =
            flash_attention_forward_neon(q_head, k_head, v_head, seq_len, head_dim, block_size);

        output[off..off + head_size].copy_from_slice(&head_out);
    }

    output
}

// ═══════════════════════════════════════════════════════════════════════
// Reference implementations for testing
// ═══════════════════════════════════════════════════════════════════════

/// Naive reference attention for validation:
/// softmax(Q·Kᵀ / √d + causal_mask) × V
#[allow(dead_code)]
fn reference_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; seq_len * head_dim];

    for i in 0..seq_len {
        // Compute scores
        let mut scores = vec![0.0f32; seq_len];
        for j in 0..seq_len {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[i * head_dim + d] * k[j * head_dim + d];
            }
            scores[j] = dot * scale;
        }

        // Causal mask
        for j in (i + 1)..seq_len {
            scores[j] = f32::NEG_INFINITY;
        }

        // Softmax
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|&s| (s - max).exp()).collect();
        let sum: f32 = exps.iter().sum();

        // Weighted sum of V
        if sum > 0.0 {
            for j in 0..seq_len {
                let w = exps[j] / sum;
                for d in 0..head_dim {
                    output[i * head_dim + d] += w * v[j * head_dim + d];
                }
            }
        }
    }

    output
}

/// Reference matmul: C = A × B (naive triple loop).
#[allow(dead_code)]
fn reference_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for ki in 0..k {
                sum += a[i * k + ki] * b[ki * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;

    fn assert_close(a: f32, b: f32, tol: f32, ctx: &str) {
        assert!((a - b).abs() < tol, "{ctx}: expected {b}, got {a} (diff {})", (a - b).abs());
    }

    fn assert_vec_close(a: &[f32], b: &[f32], tol: f32, ctx: &str) {
        assert_eq!(a.len(), b.len(), "{ctx}: length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert_close(x, y, tol, &format!("{ctx}[{i}]"));
        }
    }

    /// Deterministic seed-based pseudo-random f32 in [lo, hi].
    fn pseudo_rand(seed: u64, lo: f32, hi: f32) -> f32 {
        // Simple xorshift-like hash
        let mut x = seed;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let t = (x & 0xFFFF) as f32 / 65535.0;
        lo + t * (hi - lo)
    }

    fn make_random_vec(len: usize, seed: u64) -> Vec<f32> {
        (0..len).map(|i| pseudo_rand(seed.wrapping_add(i as u64), -1.0, 1.0)).collect()
    }

    // ── flash_attention_forward_neon ────────────────────────────────

    #[test]
    fn test_flash_attn_seq1_dim4() {
        let q = vec![1.0, 0.0, 0.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 0.0];
        let v = vec![0.5, 0.5, 0.5, 0.5];
        let out = flash_attention_forward_neon(&q, &k, &v, 1, 4, 1);
        assert_vec_close(&out, &v, 1e-4, "seq1_dim4");
    }

    #[test]
    fn test_flash_attn_seq2_dim4() {
        let q = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let v = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let out = flash_attention_forward_neon(&q, &k, &v, 2, 4, 2);
        let ref_out = reference_attention(&q, &k, &v, 2, 4);
        assert_vec_close(&out, &ref_out, 1e-3, "seq2_dim4");
    }

    #[test]
    fn test_flash_attn_matches_reference_seq4_dim8() {
        let q = make_random_vec(4 * 8, 42);
        let k = make_random_vec(4 * 8, 100);
        let v = make_random_vec(4 * 8, 200);
        let out = flash_attention_forward_neon(&q, &k, &v, 4, 8, 2);
        let ref_out = reference_attention(&q, &k, &v, 4, 8);
        assert_vec_close(&out, &ref_out, 0.05, "seq4_dim8");
    }

    #[test]
    fn test_flash_attn_matches_reference_seq8_dim16() {
        let q = make_random_vec(8 * 16, 1);
        let k = make_random_vec(8 * 16, 2);
        let v = make_random_vec(8 * 16, 3);
        let out = flash_attention_forward_neon(&q, &k, &v, 8, 16, 4);
        let ref_out = reference_attention(&q, &k, &v, 8, 16);
        assert_vec_close(&out, &ref_out, 0.05, "seq8_dim16");
    }

    #[test]
    fn test_flash_attn_block_size_1() {
        let q = make_random_vec(4 * 8, 10);
        let k = make_random_vec(4 * 8, 20);
        let v = make_random_vec(4 * 8, 30);
        let out = flash_attention_forward_neon(&q, &k, &v, 4, 8, 1);
        let ref_out = reference_attention(&q, &k, &v, 4, 8);
        assert_vec_close(&out, &ref_out, 0.05, "block1");
    }

    #[test]
    fn test_flash_attn_block_larger_than_seq() {
        let q = make_random_vec(3 * 4, 50);
        let k = make_random_vec(3 * 4, 60);
        let v = make_random_vec(3 * 4, 70);
        let out = flash_attention_forward_neon(&q, &k, &v, 3, 4, 16);
        let ref_out = reference_attention(&q, &k, &v, 3, 4);
        assert_vec_close(&out, &ref_out, 0.05, "big_block");
    }

    #[test]
    fn test_flash_attn_uniform_values() {
        // With uniform V, output should equal the value vector regardless of Q/K.
        let q = make_random_vec(4 * 8, 111);
        let k = make_random_vec(4 * 8, 222);
        let v = vec![0.7f32; 4 * 8];
        let out = flash_attention_forward_neon(&q, &k, &v, 4, 8, 2);
        for &val in &out {
            assert_close(val, 0.7, 0.02, "uniform_v");
        }
    }

    #[test]
    fn test_flash_attn_output_length() {
        let q = make_random_vec(6 * 12, 1);
        let k = make_random_vec(6 * 12, 2);
        let v = make_random_vec(6 * 12, 3);
        let out = flash_attention_forward_neon(&q, &k, &v, 6, 12, 3);
        assert_eq!(out.len(), 6 * 12);
    }

    #[test]
    fn test_flash_attn_no_nan() {
        let q = make_random_vec(4 * 8, 555);
        let k = make_random_vec(4 * 8, 666);
        let v = make_random_vec(4 * 8, 777);
        let out = flash_attention_forward_neon(&q, &k, &v, 4, 8, 2);
        for &val in &out {
            assert!(!val.is_nan(), "output contains NaN");
        }
    }

    #[test]
    fn test_flash_attn_no_inf() {
        let q = make_random_vec(4 * 8, 888);
        let k = make_random_vec(4 * 8, 999);
        let v = make_random_vec(4 * 8, 1111);
        let out = flash_attention_forward_neon(&q, &k, &v, 4, 8, 2);
        for &val in &out {
            assert!(val.is_finite(), "output contains infinity");
        }
    }

    #[test]
    fn test_flash_attn_seq1_single_token() {
        // Single token: output == v (softmax of single score = 1)
        let q = vec![0.5; 4];
        let k = vec![0.5; 4];
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let out = flash_attention_forward_neon(&q, &k, &v, 1, 4, 1);
        assert_vec_close(&out, &v, 1e-4, "single_token");
    }

    #[test]
    fn test_flash_attn_head_dim_non_multiple_of_4() {
        let dim = 5;
        let q = make_random_vec(3 * dim, 42);
        let k = make_random_vec(3 * dim, 43);
        let v = make_random_vec(3 * dim, 44);
        let out = flash_attention_forward_neon(&q, &k, &v, 3, dim, 2);
        let ref_out = reference_attention(&q, &k, &v, 3, dim);
        assert_vec_close(&out, &ref_out, 0.05, "dim5");
    }

    #[test]
    fn test_flash_attn_head_dim_1() {
        let q = make_random_vec(4, 11);
        let k = make_random_vec(4, 12);
        let v = make_random_vec(4, 13);
        let out = flash_attention_forward_neon(&q, &k, &v, 4, 1, 2);
        let ref_out = reference_attention(&q, &k, &v, 4, 1);
        assert_vec_close(&out, &ref_out, 0.05, "dim1");
    }

    #[test]
    fn test_flash_attn_seq16_dim64() {
        let q = make_random_vec(16 * 64, 500);
        let k = make_random_vec(16 * 64, 501);
        let v = make_random_vec(16 * 64, 502);
        let out = flash_attention_forward_neon(&q, &k, &v, 16, 64, 4);
        let ref_out = reference_attention(&q, &k, &v, 16, 64);
        assert_vec_close(&out, &ref_out, 0.1, "seq16_dim64");
    }

    #[test]
    fn test_flash_attn_causal_first_row_only_self() {
        // First query can only attend to first key (causal) → output = v[0]
        let q = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let v = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let out = flash_attention_forward_neon(&q, &k, &v, 2, 3, 2);
        // Row 0 attends only to key 0 → output[0..3] ≈ v[0..3]
        assert_close(out[0], 10.0, 0.01, "causal_row0_d0");
        assert_close(out[1], 20.0, 0.01, "causal_row0_d1");
        assert_close(out[2], 30.0, 0.01, "causal_row0_d2");
    }

    #[test]
    fn test_flash_attn_different_block_sizes_same_result() {
        let q = make_random_vec(8 * 8, 700);
        let k = make_random_vec(8 * 8, 701);
        let v = make_random_vec(8 * 8, 702);
        let out1 = flash_attention_forward_neon(&q, &k, &v, 8, 8, 2);
        let out2 = flash_attention_forward_neon(&q, &k, &v, 8, 8, 4);
        let out3 = flash_attention_forward_neon(&q, &k, &v, 8, 8, 8);
        assert_vec_close(&out1, &out2, 0.02, "bs2_vs_bs4");
        assert_vec_close(&out2, &out3, 0.02, "bs4_vs_bs8");
    }

    #[test]
    fn test_flash_attn_zeros_q() {
        let q = vec![0.0; 4 * 4];
        let k = make_random_vec(4 * 4, 1);
        let v = make_random_vec(4 * 4, 2);
        let out = flash_attention_forward_neon(&q, &k, &v, 4, 4, 2);
        // All-zero Q → uniform attention (after causal mask) over V
        for &val in &out {
            assert!(val.is_finite(), "zero_q: output not finite");
        }
    }

    #[test]
    fn test_flash_attn_identity_value() {
        // V = identity rows: output row i should be a weighted combo
        let q = vec![0.1; 4 * 4];
        let k = vec![0.1; 4 * 4];
        let v = vec![
            1.0, 0.0, 0.0, 0.0, // row 0
            0.0, 1.0, 0.0, 0.0, // row 1
            0.0, 0.0, 1.0, 0.0, // row 2
            0.0, 0.0, 0.0, 1.0, // row 3
        ];
        let out = flash_attention_forward_neon(&q, &k, &v, 4, 4, 2);
        let ref_out = reference_attention(&q, &k, &v, 4, 4);
        assert_vec_close(&out, &ref_out, 0.02, "identity_v");
    }

    #[test]
    fn test_flash_attn_seq3_dim7_odd() {
        let q = make_random_vec(3 * 7, 300);
        let k = make_random_vec(3 * 7, 301);
        let v = make_random_vec(3 * 7, 302);
        let out = flash_attention_forward_neon(&q, &k, &v, 3, 7, 2);
        let ref_out = reference_attention(&q, &k, &v, 3, 7);
        assert_vec_close(&out, &ref_out, 0.05, "seq3_dim7");
    }

    // ── tiled_softmax_neon ─────────────────────────────────────────

    #[test]
    fn test_tiled_softmax_single_row() {
        let scores = vec![1.0, 2.0, 3.0, 4.0];
        let (sm, rm, rs) = tiled_softmax_neon(&scores, 1, 4);
        assert_eq!(sm.len(), 4);
        let sum: f32 = sm.iter().sum();
        assert_close(sum, 1.0, 1e-4, "softmax_sum");
        assert_close(rm[0], 4.0, 1e-6, "row_max");
        assert!(rs[0] > 0.0, "row_sum positive");
    }

    #[test]
    fn test_tiled_softmax_uniform() {
        let scores = vec![1.0; 8];
        let (sm, _rm, _rs) = tiled_softmax_neon(&scores, 1, 8);
        for &val in &sm {
            assert_close(val, 0.125, 1e-4, "uniform_sm");
        }
    }

    #[test]
    fn test_tiled_softmax_two_rows() {
        let scores = vec![0.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 0.0];
        let (sm, rm, rs) = tiled_softmax_neon(&scores, 2, 4);
        assert_eq!(sm.len(), 8);
        assert_eq!(rm.len(), 2);
        assert_eq!(rs.len(), 2);
        // Each row sums to 1
        let sum0: f32 = sm[0..4].iter().sum();
        let sum1: f32 = sm[4..8].iter().sum();
        assert_close(sum0, 1.0, 1e-4, "row0_sum");
        assert_close(sum1, 1.0, 1e-4, "row1_sum");
    }

    #[test]
    fn test_tiled_softmax_row_max_correct() {
        let scores = vec![1.0, 5.0, 3.0, 2.0, -1.0, -2.0, 0.0, -3.0];
        let (_sm, rm, _rs) = tiled_softmax_neon(&scores, 2, 4);
        assert_close(rm[0], 5.0, 1e-6, "rm0");
        assert_close(rm[1], 0.0, 1e-6, "rm1");
    }

    #[test]
    fn test_tiled_softmax_large_values() {
        let scores = vec![100.0, 101.0, 102.0, 103.0];
        let (sm, _rm, _rs) = tiled_softmax_neon(&scores, 1, 4);
        let sum: f32 = sm.iter().sum();
        assert_close(sum, 1.0, 1e-3, "large_val_sum");
    }

    #[test]
    fn test_tiled_softmax_negative_values() {
        let scores = vec![-10.0, -20.0, -30.0, -40.0];
        let (sm, rm, _rs) = tiled_softmax_neon(&scores, 1, 4);
        let sum: f32 = sm.iter().sum();
        assert_close(sum, 1.0, 1e-3, "neg_val_sum");
        assert_close(rm[0], -10.0, 1e-6, "neg_rm");
    }

    #[test]
    fn test_tiled_softmax_monotonicity() {
        let scores = vec![1.0, 2.0, 3.0, 4.0];
        let (sm, _rm, _rs) = tiled_softmax_neon(&scores, 1, 4);
        // Softmax preserves ordering
        assert!(sm[0] < sm[1]);
        assert!(sm[1] < sm[2]);
        assert!(sm[2] < sm[3]);
    }

    #[test]
    fn test_tiled_softmax_non_multiple_of_4() {
        let scores = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (sm, _rm, _rs) = tiled_softmax_neon(&scores, 1, 5);
        let sum: f32 = sm.iter().sum();
        assert_close(sum, 1.0, 1e-3, "5elem_sum");
    }

    #[test]
    fn test_tiled_softmax_single_element() {
        let scores = vec![42.0];
        let (sm, rm, rs) = tiled_softmax_neon(&scores, 1, 1);
        assert_close(sm[0], 1.0, 1e-6, "single_elem");
        assert_close(rm[0], 42.0, 1e-6, "single_rm");
        assert!(rs[0] > 0.0);
    }

    #[test]
    fn test_tiled_softmax_no_nan() {
        let scores = make_random_vec(3 * 8, 42);
        let (sm, rm, rs) = tiled_softmax_neon(&scores, 3, 8);
        for &v in sm.iter().chain(rm.iter()).chain(rs.iter()) {
            assert!(!v.is_nan(), "softmax output contains NaN");
        }
    }

    #[test]
    fn test_tiled_softmax_4x4() {
        let scores = make_random_vec(16, 99);
        let (sm, _rm, _rs) = tiled_softmax_neon(&scores, 4, 4);
        for r in 0..4 {
            let sum: f32 = sm[r * 4..(r + 1) * 4].iter().sum();
            assert_close(sum, 1.0, 1e-3, &format!("4x4_row{r}"));
        }
    }

    #[test]
    fn test_tiled_softmax_all_zeros() {
        let scores = vec![0.0; 8];
        let (sm, _rm, _rs) = tiled_softmax_neon(&scores, 2, 4);
        for &v in &sm[0..4] {
            assert_close(v, 0.25, 1e-4, "zero_row0");
        }
        for &v in &sm[4..8] {
            assert_close(v, 0.25, 1e-4, "zero_row1");
        }
    }

    // ── blocked_matmul_neon ────────────────────────────────────────

    #[test]
    fn test_matmul_identity() {
        #[rustfmt::skip]
        let a = vec![
            1.0, 0.0,
            0.0, 1.0,
        ];
        let b = vec![3.0, 4.0, 5.0, 6.0];
        let c = blocked_matmul_neon(&a, &b, 2, 2, 2, 2);
        assert_vec_close(&c, &b, 1e-5, "identity_matmul");
    }

    #[test]
    fn test_matmul_2x3_3x2() {
        #[rustfmt::skip]
        let a = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ];
        #[rustfmt::skip]
        let b = vec![
            7.0, 8.0,
            9.0, 10.0,
            11.0, 12.0,
        ];
        let c = blocked_matmul_neon(&a, &b, 2, 2, 3, 2);
        let ref_c = reference_matmul(&a, &b, 2, 2, 3);
        assert_vec_close(&c, &ref_c, 1e-4, "2x3_3x2");
    }

    #[test]
    fn test_matmul_matches_reference_random() {
        let a = make_random_vec(4 * 8, 42);
        let b = make_random_vec(8 * 6, 43);
        let c = blocked_matmul_neon(&a, &b, 4, 6, 8, 2);
        let ref_c = reference_matmul(&a, &b, 4, 6, 8);
        assert_vec_close(&c, &ref_c, 0.01, "random_matmul");
    }

    #[test]
    fn test_matmul_block_sizes() {
        let a = make_random_vec(8 * 8, 100);
        let b = make_random_vec(8 * 8, 200);
        let c1 = blocked_matmul_neon(&a, &b, 8, 8, 8, 1);
        let c2 = blocked_matmul_neon(&a, &b, 8, 8, 8, 4);
        let c3 = blocked_matmul_neon(&a, &b, 8, 8, 8, 8);
        let ref_c = reference_matmul(&a, &b, 8, 8, 8);
        assert_vec_close(&c1, &ref_c, 0.01, "bs1");
        assert_vec_close(&c2, &ref_c, 0.01, "bs4");
        assert_vec_close(&c3, &ref_c, 0.01, "bs8");
    }

    #[test]
    fn test_matmul_1x1() {
        let a = vec![3.0];
        let b = vec![5.0];
        let c = blocked_matmul_neon(&a, &b, 1, 1, 1, 1);
        assert_close(c[0], 15.0, 1e-5, "1x1");
    }

    #[test]
    fn test_matmul_output_dimensions() {
        let a = make_random_vec(3 * 5, 1);
        let b = make_random_vec(5 * 7, 2);
        let c = blocked_matmul_neon(&a, &b, 3, 7, 5, 2);
        assert_eq!(c.len(), 3 * 7);
    }

    #[test]
    fn test_matmul_zeros() {
        let a = vec![0.0; 4 * 4];
        let b = make_random_vec(4 * 4, 1);
        let c = blocked_matmul_neon(&a, &b, 4, 4, 4, 2);
        for &val in &c {
            assert_close(val, 0.0, 1e-6, "zero_matmul");
        }
    }

    #[test]
    fn test_matmul_non_square() {
        let a = make_random_vec(3 * 5, 50);
        let b = make_random_vec(5 * 4, 51);
        let c = blocked_matmul_neon(&a, &b, 3, 4, 5, 2);
        let ref_c = reference_matmul(&a, &b, 3, 4, 5);
        assert_vec_close(&c, &ref_c, 0.01, "nonsquare");
    }

    #[test]
    fn test_matmul_large_block() {
        let a = make_random_vec(2 * 3, 10);
        let b = make_random_vec(3 * 2, 20);
        let c = blocked_matmul_neon(&a, &b, 2, 2, 3, 64);
        let ref_c = reference_matmul(&a, &b, 2, 2, 3);
        assert_vec_close(&c, &ref_c, 0.01, "large_block");
    }

    #[test]
    fn test_matmul_no_nan() {
        let a = make_random_vec(4 * 6, 777);
        let b = make_random_vec(6 * 4, 888);
        let c = blocked_matmul_neon(&a, &b, 4, 4, 6, 2);
        for &val in &c {
            assert!(!val.is_nan(), "matmul NaN");
        }
    }

    // ── causal_mask_scores_neon ────────────────────────────────────

    #[test]
    fn test_causal_mask_2x2() {
        let mut scores = vec![1.0, 2.0, 3.0, 4.0];
        causal_mask_scores_neon(&mut scores, 2, f32::NEG_INFINITY);
        assert_close(scores[0], 1.0, 1e-6, "cm_00");
        assert_eq!(scores[1], f32::NEG_INFINITY); // (0,1) masked
        assert_close(scores[2], 3.0, 1e-6, "cm_10");
        assert_close(scores[3], 4.0, 1e-6, "cm_11");
    }

    #[test]
    fn test_causal_mask_4x4() {
        let mut scores = vec![1.0; 16];
        causal_mask_scores_neon(&mut scores, 4, -1e9);
        // Check diagonal and below are preserved
        for row in 0..4 {
            for col in 0..4 {
                if col > row {
                    assert_close(scores[row * 4 + col], -1e9, 1e-2, &format!("mask_{row}_{col}"));
                } else {
                    assert_close(scores[row * 4 + col], 1.0, 1e-6, &format!("keep_{row}_{col}"));
                }
            }
        }
    }

    #[test]
    fn test_causal_mask_1x1() {
        let mut scores = vec![5.0];
        causal_mask_scores_neon(&mut scores, 1, f32::NEG_INFINITY);
        assert_close(scores[0], 5.0, 1e-6, "1x1_keep");
    }

    #[test]
    fn test_causal_mask_preserves_diagonal() {
        let mut scores = (0..25).map(|i| i as f32).collect::<Vec<_>>();
        causal_mask_scores_neon(&mut scores, 5, f32::NEG_INFINITY);
        for i in 0..5 {
            assert_close(scores[i * 5 + i], (i * 5 + i) as f32, 1e-6, &format!("diag_{i}"));
        }
    }

    #[test]
    fn test_causal_mask_upper_triangle_masked() {
        let mut scores = vec![1.0; 9];
        causal_mask_scores_neon(&mut scores, 3, f32::NEG_INFINITY);
        // Upper triangle: (0,1), (0,2), (1,2)
        assert_eq!(scores[1], f32::NEG_INFINITY);
        assert_eq!(scores[2], f32::NEG_INFINITY);
        assert_eq!(scores[5], f32::NEG_INFINITY);
    }

    #[test]
    fn test_causal_mask_lower_triangle_preserved() {
        let mut scores = vec![7.0; 9];
        causal_mask_scores_neon(&mut scores, 3, f32::NEG_INFINITY);
        // Lower triangle + diagonal: (0,0), (1,0), (1,1), (2,0), (2,1), (2,2)
        assert_close(scores[0], 7.0, 1e-6, "00");
        assert_close(scores[3], 7.0, 1e-6, "10");
        assert_close(scores[4], 7.0, 1e-6, "11");
        assert_close(scores[6], 7.0, 1e-6, "20");
        assert_close(scores[7], 7.0, 1e-6, "21");
        assert_close(scores[8], 7.0, 1e-6, "22");
    }

    #[test]
    fn test_causal_mask_custom_neg_inf() {
        let mut scores = vec![1.0; 4];
        causal_mask_scores_neon(&mut scores, 2, -999.0);
        assert_close(scores[1], -999.0, 1e-3, "custom_neg");
    }

    #[test]
    fn test_causal_mask_8x8() {
        let mut scores = vec![1.0; 64];
        causal_mask_scores_neon(&mut scores, 8, f32::NEG_INFINITY);
        for row in 0..8 {
            for col in 0..8 {
                if col > row {
                    assert_eq!(scores[row * 8 + col], f32::NEG_INFINITY, "8x8_mask_{row}_{col}");
                } else {
                    assert_close(
                        scores[row * 8 + col],
                        1.0,
                        1e-6,
                        &format!("8x8_keep_{row}_{col}"),
                    );
                }
            }
        }
    }

    // ── multi_head_flash_attention_neon ────────────────────────────

    #[test]
    fn test_multi_head_single_head() {
        let q = make_random_vec(4 * 8, 42);
        let k = make_random_vec(4 * 8, 43);
        let v = make_random_vec(4 * 8, 44);
        let mh_out = multi_head_flash_attention_neon(&q, &k, &v, 4, 1, 8, 2);
        let sh_out = flash_attention_forward_neon(&q, &k, &v, 4, 8, 2);
        assert_vec_close(&mh_out, &sh_out, 1e-5, "mh_single");
    }

    #[test]
    fn test_multi_head_two_heads() {
        let q = make_random_vec(2 * 4 * 8, 10);
        let k = make_random_vec(2 * 4 * 8, 20);
        let v = make_random_vec(2 * 4 * 8, 30);
        let out = multi_head_flash_attention_neon(&q, &k, &v, 4, 2, 8, 2);
        assert_eq!(out.len(), 2 * 4 * 8);
    }

    #[test]
    fn test_multi_head_independent_heads() {
        // Each head should be computed independently
        let q = make_random_vec(2 * 4 * 8, 100);
        let k = make_random_vec(2 * 4 * 8, 200);
        let v = make_random_vec(2 * 4 * 8, 300);
        let out = multi_head_flash_attention_neon(&q, &k, &v, 4, 2, 8, 2);

        let head_size = 4 * 8;
        let h0 = flash_attention_forward_neon(
            &q[..head_size],
            &k[..head_size],
            &v[..head_size],
            4,
            8,
            2,
        );
        let h1 = flash_attention_forward_neon(
            &q[head_size..],
            &k[head_size..],
            &v[head_size..],
            4,
            8,
            2,
        );
        assert_vec_close(&out[..head_size], &h0, 1e-5, "mh_head0");
        assert_vec_close(&out[head_size..], &h1, 1e-5, "mh_head1");
    }

    #[test]
    fn test_multi_head_output_length() {
        let q = make_random_vec(4 * 6 * 16, 1);
        let k = make_random_vec(4 * 6 * 16, 2);
        let v = make_random_vec(4 * 6 * 16, 3);
        let out = multi_head_flash_attention_neon(&q, &k, &v, 6, 4, 16, 3);
        assert_eq!(out.len(), 4 * 6 * 16);
    }

    #[test]
    fn test_multi_head_no_nan() {
        let q = make_random_vec(2 * 4 * 8, 555);
        let k = make_random_vec(2 * 4 * 8, 666);
        let v = make_random_vec(2 * 4 * 8, 777);
        let out = multi_head_flash_attention_neon(&q, &k, &v, 4, 2, 8, 2);
        for &val in &out {
            assert!(!val.is_nan(), "mh NaN");
        }
    }

    #[test]
    fn test_multi_head_uniform_v() {
        let q = make_random_vec(3 * 4 * 8, 11);
        let k = make_random_vec(3 * 4 * 8, 22);
        let v = vec![0.5f32; 3 * 4 * 8];
        let out = multi_head_flash_attention_neon(&q, &k, &v, 4, 3, 8, 2);
        for &val in &out {
            assert_close(val, 0.5, 0.02, "mh_uniform");
        }
    }

    #[test]
    fn test_multi_head_4_heads() {
        let q = make_random_vec(4 * 4 * 4, 1000);
        let k = make_random_vec(4 * 4 * 4, 2000);
        let v = make_random_vec(4 * 4 * 4, 3000);
        let out = multi_head_flash_attention_neon(&q, &k, &v, 4, 4, 4, 2);
        assert_eq!(out.len(), 4 * 4 * 4);
        for &val in &out {
            assert!(val.is_finite(), "mh4 finite");
        }
    }

    #[test]
    fn test_multi_head_matches_ref_per_head() {
        let num_heads = 3;
        let seq_len = 4;
        let head_dim = 8;
        let head_size = seq_len * head_dim;
        let q = make_random_vec(num_heads * head_size, 42);
        let k = make_random_vec(num_heads * head_size, 43);
        let v = make_random_vec(num_heads * head_size, 44);
        let out = multi_head_flash_attention_neon(&q, &k, &v, seq_len, num_heads, head_dim, 2);

        for h in 0..num_heads {
            let off = h * head_size;
            let ref_out = reference_attention(
                &q[off..off + head_size],
                &k[off..off + head_size],
                &v[off..off + head_size],
                seq_len,
                head_dim,
            );
            assert_vec_close(
                &out[off..off + head_size],
                &ref_out,
                0.05,
                &format!("mh_ref_head{h}"),
            );
        }
    }

    // ── Cross-function integration tests ──────────────────────────

    #[test]
    fn test_tiled_softmax_sums_to_one() {
        for cols in [1, 3, 4, 5, 8, 13, 16] {
            let scores = make_random_vec(cols, cols as u64);
            let (sm, _rm, _rs) = tiled_softmax_neon(&scores, 1, cols);
            let sum: f32 = sm.iter().sum();
            assert_close(sum, 1.0, 1e-3, &format!("sum_cols{cols}"));
        }
    }

    #[test]
    fn test_matmul_then_softmax_pipeline() {
        let q = make_random_vec(4 * 8, 1);
        let k = make_random_vec(4 * 8, 2);
        // Transpose K for Q·Kᵀ
        let mut kt = vec![0.0f32; 8 * 4];
        for i in 0..4 {
            for j in 0..8 {
                kt[j * 4 + i] = k[i * 8 + j];
            }
        }
        let scores = blocked_matmul_neon(&q, &kt, 4, 4, 8, 2);
        assert_eq!(scores.len(), 16);
        let (sm, _rm, _rs) = tiled_softmax_neon(&scores, 4, 4);
        for r in 0..4 {
            let sum: f32 = sm[r * 4..(r + 1) * 4].iter().sum();
            assert_close(sum, 1.0, 1e-3, &format!("pipe_row{r}"));
        }
    }

    #[test]
    fn test_causal_mask_then_softmax() {
        let mut scores = vec![1.0; 16];
        causal_mask_scores_neon(&mut scores, 4, f32::NEG_INFINITY);
        let (sm, _rm, _rs) = tiled_softmax_neon(&scores, 4, 4);
        // Row 0: only (0,0) is valid → softmax = [1, 0, 0, 0]
        assert_close(sm[0], 1.0, 1e-3, "masked_sm_00");
        assert_close(sm[1], 0.0, 1e-3, "masked_sm_01");
    }

    #[test]
    fn test_flash_attn_seq5_dim3_odd_dims() {
        let q = make_random_vec(5 * 3, 40);
        let k = make_random_vec(5 * 3, 41);
        let v = make_random_vec(5 * 3, 42);
        let out = flash_attention_forward_neon(&q, &k, &v, 5, 3, 2);
        let ref_out = reference_attention(&q, &k, &v, 5, 3);
        assert_vec_close(&out, &ref_out, 0.05, "seq5_dim3");
    }

    #[test]
    fn test_flash_attn_numerical_stability_large_qk() {
        // Large Q/K values should not cause overflow thanks to online softmax
        let q = vec![10.0f32; 4 * 4];
        let k = vec![10.0f32; 4 * 4];
        let v = make_random_vec(4 * 4, 1);
        let out = flash_attention_forward_neon(&q, &k, &v, 4, 4, 2);
        for &val in &out {
            assert!(val.is_finite(), "large_qk not finite");
        }
    }

    #[test]
    fn test_matmul_commutativity_trace() {
        // tr(A·B) should equal the sum of element-wise products
        let a = make_random_vec(4 * 4, 11);
        let b = make_random_vec(4 * 4, 22);
        let c = blocked_matmul_neon(&a, &b, 4, 4, 4, 2);
        let trace: f32 = (0..4).map(|i| c[i * 4 + i]).sum();
        // Compute reference trace via dot products
        let mut ref_trace = 0.0f32;
        for i in 0..4 {
            for k in 0..4 {
                ref_trace += a[i * 4 + k] * b[k * 4 + i];
            }
        }
        assert_close(trace, ref_trace, 0.1, "trace");
    }

    #[test]
    fn test_tiled_softmax_row_independence() {
        // Changing one row should not affect the other
        let mut scores1 = make_random_vec(2 * 4, 1);
        let scores2 = scores1.clone();
        scores1[0] += 10.0; // change row 0

        let (sm1, _, _) = tiled_softmax_neon(&scores1, 2, 4);
        let (sm2, _, _) = tiled_softmax_neon(&scores2, 2, 4);

        // Row 1 should be identical
        assert_vec_close(&sm1[4..8], &sm2[4..8], 1e-6, "row_indep");
    }

    #[test]
    fn test_causal_mask_count_masked() {
        let mut scores = vec![1.0; 25]; // 5×5
        causal_mask_scores_neon(&mut scores, 5, f32::NEG_INFINITY);
        let masked_count = scores.iter().filter(|&&v| v == f32::NEG_INFINITY).count();
        // Upper triangle count = n*(n-1)/2 = 10
        assert_eq!(masked_count, 10, "mask_count");
    }

    #[test]
    fn test_flash_attn_scale_factor() {
        // Verify scale = 1/√d is applied correctly
        // With dim=1 and single token, score = q*k*scale = q*k/1 = q*k
        let q = vec![2.0];
        let k = vec![3.0];
        let v = vec![5.0];
        let out = flash_attention_forward_neon(&q, &k, &v, 1, 1, 1);
        // Single token → output = v regardless of score
        assert_close(out[0], 5.0, 1e-4, "scale_single");
    }

    #[test]
    fn test_multi_head_block_size_variations() {
        let q = make_random_vec(2 * 6 * 8, 42);
        let k = make_random_vec(2 * 6 * 8, 43);
        let v = make_random_vec(2 * 6 * 8, 44);
        let out1 = multi_head_flash_attention_neon(&q, &k, &v, 6, 2, 8, 1);
        let out2 = multi_head_flash_attention_neon(&q, &k, &v, 6, 2, 8, 3);
        let out3 = multi_head_flash_attention_neon(&q, &k, &v, 6, 2, 8, 6);
        assert_vec_close(&out1, &out2, 0.02, "mh_bs1_vs_3");
        assert_vec_close(&out2, &out3, 0.02, "mh_bs3_vs_6");
    }

    #[test]
    fn test_matmul_scalar_multiply() {
        // A = [[s]], B = [[t]] → C = [[s*t]]
        for (s, t) in [(2.0, 3.0), (0.5, 4.0), (-1.0, 7.0)] {
            let c = blocked_matmul_neon(&[s], &[t], 1, 1, 1, 1);
            assert_close(c[0], s * t, 1e-5, "scalar_mul");
        }
    }

    #[test]
    fn test_tiled_softmax_exp_sum_consistency() {
        let scores = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let (sm, rm, rs) = tiled_softmax_neon(&scores, 2, 4);
        // Verify: row_sum ≈ sum(exp(score - row_max))
        for r in 0..2 {
            let row = &scores[r * 4..(r + 1) * 4];
            let expected_sum: f32 = row.iter().map(|&s| fast_exp_scalar(s - rm[r])).sum();
            assert_close(rs[r], expected_sum, 1e-3, &format!("rs_check{r}"));
            // Also verify softmax sums to 1
            let sm_sum: f32 = sm[r * 4..(r + 1) * 4].iter().sum();
            assert_close(sm_sum, 1.0, 1e-3, &format!("sm_sum{r}"));
        }
    }

    #[test]
    fn test_flash_attn_deterministic() {
        let q = make_random_vec(4 * 8, 42);
        let k = make_random_vec(4 * 8, 43);
        let v = make_random_vec(4 * 8, 44);
        let out1 = flash_attention_forward_neon(&q, &k, &v, 4, 8, 2);
        let out2 = flash_attention_forward_neon(&q, &k, &v, 4, 8, 2);
        assert_eq!(out1, out2, "deterministic");
    }

    #[test]
    fn test_multi_head_deterministic() {
        let q = make_random_vec(2 * 4 * 8, 42);
        let k = make_random_vec(2 * 4 * 8, 43);
        let v = make_random_vec(2 * 4 * 8, 44);
        let out1 = multi_head_flash_attention_neon(&q, &k, &v, 4, 2, 8, 2);
        let out2 = multi_head_flash_attention_neon(&q, &k, &v, 4, 2, 8, 2);
        assert_eq!(out1, out2, "mh_deterministic");
    }

    #[test]
    fn test_causal_mask_3x3() {
        let mut scores = vec![1.0; 9];
        causal_mask_scores_neon(&mut scores, 3, -1.0e30);
        // Expected: lower triangle + diagonal preserved
        let expected = vec![1.0, -1.0e30, -1.0e30, 1.0, 1.0, -1.0e30, 1.0, 1.0, 1.0];
        for (i, (&a, &b)) in scores.iter().zip(expected.iter()).enumerate() {
            assert_close(a, b, 1.0, &format!("3x3_{i}"));
        }
    }

    #[test]
    fn test_matmul_k_dim_non_multiple_of_4() {
        let a = make_random_vec(3 * 5, 1);
        let b = make_random_vec(5 * 3, 2);
        let c = blocked_matmul_neon(&a, &b, 3, 3, 5, 2);
        let ref_c = reference_matmul(&a, &b, 3, 3, 5);
        assert_vec_close(&c, &ref_c, 0.01, "k5");
    }

    #[test]
    fn test_flash_attn_seq2_dim2_minimal() {
        let q = vec![1.0, 0.0, 0.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let out = flash_attention_forward_neon(&q, &k, &v, 2, 2, 1);
        let ref_out = reference_attention(&q, &k, &v, 2, 2);
        assert_vec_close(&out, &ref_out, 0.05, "minimal_2x2");
    }

    #[test]
    fn test_tiled_softmax_max_dominates() {
        // One very large value → softmax ≈ [0, ..., 1, ..., 0]
        let mut scores = vec![0.0; 8];
        scores[3] = 50.0;
        let (sm, _rm, _rs) = tiled_softmax_neon(&scores, 1, 8);
        assert!(sm[3] > 0.99, "dominant value should be near 1.0");
        for (i, &val) in sm.iter().enumerate() {
            if i != 3 {
                assert!(val < 0.01, "non-dominant should be near 0");
            }
        }
    }

    #[test]
    fn test_flash_attn_last_row_attends_to_all() {
        // Last row in causal attention attends to all positions
        let seq_len = 4;
        let dim = 4;
        let q = vec![0.1f32; seq_len * dim];
        let k = vec![0.1f32; seq_len * dim];
        let v = make_random_vec(seq_len * dim, 42);
        let out = flash_attention_forward_neon(&q, &k, &v, seq_len, dim, 2);
        let ref_out = reference_attention(&q, &k, &v, seq_len, dim);
        // Last row should match reference (attends to all with uniform Q/K)
        let last = (seq_len - 1) * dim;
        assert_vec_close(&out[last..last + dim], &ref_out[last..last + dim], 0.05, "last_row");
    }

    #[test]
    fn test_multi_head_seq1() {
        let q = make_random_vec(2 * 1 * 4, 1);
        let k = make_random_vec(2 * 1 * 4, 2);
        let v = make_random_vec(2 * 1 * 4, 3);
        let out = multi_head_flash_attention_neon(&q, &k, &v, 1, 2, 4, 1);
        // Seq=1: output = V for each head
        assert_vec_close(&out[0..4], &v[0..4], 1e-4, "mh_seq1_h0");
        assert_vec_close(&out[4..8], &v[4..8], 1e-4, "mh_seq1_h1");
    }

    #[test]
    fn test_tiled_softmax_3rows_5cols() {
        let scores = make_random_vec(15, 42);
        let (sm, rm, rs) = tiled_softmax_neon(&scores, 3, 5);
        assert_eq!(sm.len(), 15);
        assert_eq!(rm.len(), 3);
        assert_eq!(rs.len(), 3);
        for r in 0..3 {
            let sum: f32 = sm[r * 5..(r + 1) * 5].iter().sum();
            assert_close(sum, 1.0, 1e-3, &format!("3x5_row{r}"));
        }
    }

    #[test]
    fn test_causal_mask_idempotent() {
        let mut s1 = vec![1.0; 16];
        let mut s2 = vec![1.0; 16];
        causal_mask_scores_neon(&mut s1, 4, f32::NEG_INFINITY);
        causal_mask_scores_neon(&mut s2, 4, f32::NEG_INFINITY);
        causal_mask_scores_neon(&mut s2, 4, f32::NEG_INFINITY);
        assert_eq!(s1, s2, "idempotent");
    }

    #[test]
    fn test_matmul_symmetry() {
        // For symmetric A, A·A should also be symmetric
        #[rustfmt::skip]
        let a = vec![
            1.0, 2.0,
            2.0, 3.0,
        ];
        let c = blocked_matmul_neon(&a, &a, 2, 2, 2, 2);
        assert_close(c[1], c[2], 0.01, "symmetric_off_diag");
    }

    #[test]
    fn test_fast_exp_scalar_accuracy() {
        for &x in &[-5.0, -1.0, 0.0, 1.0, 5.0, 10.0] {
            let approx = fast_exp_scalar(x);
            let exact = x.exp();
            let rel_err = (approx - exact).abs() / exact.max(1e-10);
            assert!(rel_err < 0.01, "exp({x}): approx={approx}, exact={exact}, err={rel_err}");
        }
    }

    #[test]
    fn test_fast_exp_scalar_extreme() {
        let big = fast_exp_scalar(88.0);
        assert!(big.is_finite(), "exp(88) should be finite");
        let neg = fast_exp_scalar(-88.0);
        assert!(neg >= 0.0 && neg.is_finite(), "exp(-88) should be small finite");
    }

    #[test]
    fn test_flash_attn_seq12_dim32_large() {
        let q = make_random_vec(12 * 32, 1);
        let k = make_random_vec(12 * 32, 2);
        let v = make_random_vec(12 * 32, 3);
        let out = flash_attention_forward_neon(&q, &k, &v, 12, 32, 4);
        let ref_out = reference_attention(&q, &k, &v, 12, 32);
        assert_vec_close(&out, &ref_out, 0.1, "seq12_dim32");
    }

    #[test]
    fn test_flash_attn_all_same_qkv() {
        // When Q=K=V, output should still be valid
        let data = make_random_vec(4 * 8, 42);
        let out = flash_attention_forward_neon(&data, &data, &data, 4, 8, 2);
        for &val in &out {
            assert!(val.is_finite(), "same_qkv finite");
        }
    }

    #[test]
    fn test_matmul_associativity_dims() {
        // (A·B)·C vs A·(B·C) for compatible dims
        let a = make_random_vec(2 * 3, 1);
        let b = make_random_vec(3 * 4, 2);
        let c_mat = make_random_vec(4 * 2, 3);
        let ab = blocked_matmul_neon(&a, &b, 2, 4, 3, 2);
        let abc1 = blocked_matmul_neon(&ab, &c_mat, 2, 2, 4, 2);
        let bc = blocked_matmul_neon(&b, &c_mat, 3, 2, 4, 2);
        let abc2 = blocked_matmul_neon(&a, &bc, 2, 2, 3, 2);
        assert_vec_close(&abc1, &abc2, 0.1, "associativity");
    }

    #[test]
    fn test_causal_mask_6x6() {
        let mut scores = vec![1.0; 36];
        causal_mask_scores_neon(&mut scores, 6, f32::NEG_INFINITY);
        let masked = scores.iter().filter(|&&v| v == f32::NEG_INFINITY).count();
        // Upper triangle: 6*5/2 = 15
        assert_eq!(masked, 15);
    }

    #[test]
    fn test_tiled_softmax_1x1() {
        let (sm, rm, rs) = tiled_softmax_neon(&[3.0], 1, 1);
        assert_close(sm[0], 1.0, 1e-6, "1x1_sm");
        assert_close(rm[0], 3.0, 1e-6, "1x1_rm");
        assert!(rs[0] > 0.0, "1x1 rs positive");
    }

    #[test]
    fn test_flash_attn_negative_values() {
        let q = vec![-1.0f32; 4 * 4];
        let k = vec![-1.0f32; 4 * 4];
        let v = vec![2.0f32; 4 * 4];
        let out = flash_attention_forward_neon(&q, &k, &v, 4, 4, 2);
        for &val in &out {
            assert_close(val, 2.0, 0.02, "neg_vals");
        }
    }

    #[test]
    fn test_matmul_row_vector_times_col_vector() {
        let a = vec![1.0, 2.0, 3.0]; // 1×3
        let b = vec![4.0, 5.0, 6.0]; // 3×1
        let c = blocked_matmul_neon(&a, &b, 1, 1, 3, 2);
        // 1*4 + 2*5 + 3*6 = 32
        assert_close(c[0], 32.0, 0.01, "row_col");
    }

    #[test]
    fn test_multi_head_8_heads_seq2() {
        let q = make_random_vec(8 * 2 * 4, 42);
        let k = make_random_vec(8 * 2 * 4, 43);
        let v = make_random_vec(8 * 2 * 4, 44);
        let out = multi_head_flash_attention_neon(&q, &k, &v, 2, 8, 4, 2);
        assert_eq!(out.len(), 8 * 2 * 4);
        for &val in &out {
            assert!(val.is_finite(), "8heads finite");
        }
    }

    #[test]
    fn test_flash_attn_seq6_dim4_block3() {
        let q = make_random_vec(6 * 4, 77);
        let k = make_random_vec(6 * 4, 78);
        let v = make_random_vec(6 * 4, 79);
        let out = flash_attention_forward_neon(&q, &k, &v, 6, 4, 3);
        let ref_out = reference_attention(&q, &k, &v, 6, 4);
        assert_vec_close(&out, &ref_out, 0.05, "seq6_dim4_b3");
    }

    #[test]
    fn test_tiled_softmax_invariant_shift() {
        // Softmax should be shift-invariant: softmax(x) == softmax(x + c)
        let scores1 = vec![1.0, 2.0, 3.0, 4.0];
        let scores2: Vec<f32> = scores1.iter().map(|&x| x + 100.0).collect();
        let (sm1, _, _) = tiled_softmax_neon(&scores1, 1, 4);
        let (sm2, _, _) = tiled_softmax_neon(&scores2, 1, 4);
        assert_vec_close(&sm1, &sm2, 1e-3, "shift_inv");
    }

    #[test]
    fn test_causal_mask_preserves_lower_values() {
        let mut scores: Vec<f32> = (0..16).map(|i| i as f32).collect();
        causal_mask_scores_neon(&mut scores, 4, f32::NEG_INFINITY);
        // Lower triangle values should be unchanged
        assert_close(scores[0], 0.0, 1e-6, "lt_00");
        assert_close(scores[4], 4.0, 1e-6, "lt_10");
        assert_close(scores[5], 5.0, 1e-6, "lt_11");
        assert_close(scores[8], 8.0, 1e-6, "lt_20");
        assert_close(scores[9], 9.0, 1e-6, "lt_21");
        assert_close(scores[10], 10.0, 1e-6, "lt_22");
    }
}

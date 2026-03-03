//! ARM NEON batch processing v2 kernels for Apple Silicon.
//!
//! Provides optimized batched linear-algebra primitives:
//! - Batched matrix multiplication (`batch_matmul_neon`)
//! - Batched element-wise addition (`batch_add_neon`)
//! - Batched scalar multiplication (`batch_scale_neon`)
//! - Batch normalization with running statistics (`batch_norm_neon`)
//!
//! All hot paths process 4 × `f32` NEON lanes with scalar tail fallback.
//! Input tensors are laid out as flat `[batch_size, rows, cols]` in row-major order.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON lane width for `float32x4_t`.
const LANES: usize = 4;

// ── Configuration ─────────────────────────────────────────────────

/// Describes the shape of a batched tensor workload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchConfig {
    /// Number of independent samples in the batch.
    pub batch_size: usize,
    /// Sequence (row) length of each sample.
    pub seq_len: usize,
    /// Hidden (column) dimension of each sample.
    pub hidden_dim: usize,
}

impl BatchConfig {
    /// Number of elements in a single sample matrix.
    #[inline]
    pub fn sample_elems(&self) -> usize {
        self.seq_len * self.hidden_dim
    }

    /// Total number of elements across the entire batch.
    #[inline]
    pub fn total_elems(&self) -> usize {
        self.batch_size * self.sample_elems()
    }
}

/// Parameters for [`batch_norm_neon`].
#[derive(Debug, Clone)]
pub struct BatchNormParams<'a> {
    /// Input data, `[batch_size, features]` row-major.
    pub data: &'a [f32],
    /// Number of samples.
    pub batch_size: usize,
    /// Number of features per sample.
    pub features: usize,
    /// Per-feature scale.
    pub gamma: &'a [f32],
    /// Per-feature shift.
    pub beta: &'a [f32],
    /// Numerical stability constant.
    pub eps: f32,
    /// EMA momentum for running statistics.
    pub momentum: f32,
}

// ── Batched matrix multiplication ─────────────────────────────────

/// Batched matrix multiplication: `C[b] = A[b] × B[b]` for each sample in the batch.
///
/// * `a` — `[batch_size, M, K]` row-major
/// * `b_mat` — `[batch_size, K, N]` row-major
/// * returns `[batch_size, M, N]` row-major
///
/// Uses NEON FMA for the inner dot-product accumulation.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
///
/// Panics if `a.len()` or `b_mat.len()` does not match the expected shape.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn batch_matmul_neon(
    a: &[f32],
    b_mat: &[f32],
    batch_size: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let a_sample = m * k;
    let b_sample = k * n;
    let c_sample = m * n;
    assert_eq!(a.len(), batch_size * a_sample, "a shape mismatch");
    assert_eq!(b_mat.len(), batch_size * b_sample, "b shape mismatch");

    let mut c = vec![0.0f32; batch_size * c_sample];

    for batch in 0..batch_size {
        let a_off = batch * a_sample;
        let b_off = batch * b_sample;
        let c_off = batch * c_sample;

        for row in 0..m {
            for col in 0..n {
                let mut acc = vdupq_n_f32(0.0);
                let full = k / LANES;

                for t in 0..full {
                    let idx_a = a_off + row * k + t * LANES;
                    let idx_b_base = b_off + col;
                    // Gather b column values into a NEON register.
                    let bv = unsafe {
                        let b0 = b_mat[idx_b_base + (t * LANES) * n];
                        let b1 = b_mat[idx_b_base + (t * LANES + 1) * n];
                        let b2 = b_mat[idx_b_base + (t * LANES + 2) * n];
                        let b3 = b_mat[idx_b_base + (t * LANES + 3) * n];
                        let arr: [f32; 4] = [b0, b1, b2, b3];
                        vld1q_f32(arr.as_ptr())
                    };
                    let av = unsafe { vld1q_f32(a.as_ptr().add(idx_a)) };
                    acc = vfmaq_f32(acc, av, bv);
                }

                // Horizontal reduction of the NEON accumulator.
                let sum = vgetq_lane_f32::<0>(acc)
                    + vgetq_lane_f32::<1>(acc)
                    + vgetq_lane_f32::<2>(acc)
                    + vgetq_lane_f32::<3>(acc);

                // Scalar tail.
                let mut scalar_sum = sum;
                for t in (full * LANES)..k {
                    scalar_sum += a[a_off + row * k + t] * b_mat[b_off + t * n + col];
                }

                c[c_off + row * n + col] = scalar_sum;
            }
        }
    }
    c
}

// ── Batched element-wise addition ─────────────────────────────────

/// Batched element-wise addition: `out[i] = a[i] + b[i]`.
///
/// Both `a` and `b` must have identical lengths.  Uses NEON 4-wide adds
/// with scalar tail.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn batch_add_neon(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "batch_add length mismatch");
    let len = a.len();
    let mut out = vec![0.0f32; len];

    let full = len / LANES;

    for i in 0..full {
        let off = i * LANES;
        unsafe {
            let va = vld1q_f32(a.as_ptr().add(off));
            let vb = vld1q_f32(b.as_ptr().add(off));
            let vs = vaddq_f32(va, vb);
            vst1q_f32(out.as_mut_ptr().add(off), vs);
        }
    }

    let base = full * LANES;
    for i in 0..len - base {
        out[base + i] = a[base + i] + b[base + i];
    }
    out
}

// ── Batched scalar multiplication ─────────────────────────────────

/// Batched scalar multiplication: `out[i] = data[i] * scale`.
///
/// Broadcasts `scale` across NEON lanes for 4-wide multiply.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn batch_scale_neon(data: &[f32], scale: f32) -> Vec<f32> {
    let len = data.len();
    let mut out = vec![0.0f32; len];

    let vs = vdupq_n_f32(scale);
    let full = len / LANES;

    for i in 0..full {
        let off = i * LANES;
        unsafe {
            let vd = vld1q_f32(data.as_ptr().add(off));
            let vr = vmulq_f32(vd, vs);
            vst1q_f32(out.as_mut_ptr().add(off), vr);
        }
    }

    let base = full * LANES;
    for i in 0..len - base {
        out[base + i] = data[base + i] * scale;
    }
    out
}

// ── Batch normalization ───────────────────────────────────────────

/// Batch normalization with optional running-statistics update.
///
/// Input shape: `[batch_size, features]` (row-major).
///
/// For each feature `f`:
///   1. Compute batch mean `μ_f` and variance `σ²_f`.
///   2. Normalize: `x̂ = (x - μ) / sqrt(σ² + eps)`.
///   3. Scale + shift: `y = γ · x̂ + β`.
///   4. If `running_mean` / `running_var` are provided, update them with
///      exponential moving average using `momentum`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
///
/// Panics if any of the per-feature slices (`gamma`, `beta`, etc.) have a
/// length other than `features`, or if `data` does not have exactly
/// `batch_size * features` elements.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn batch_norm_neon(
    params: &BatchNormParams<'_>,
    running_mean: Option<&mut [f32]>,
    running_var: Option<&mut [f32]>,
) -> Vec<f32> {
    let BatchNormParams { data, batch_size, features, gamma, beta, eps, momentum } = *params;

    assert_eq!(data.len(), batch_size * features, "data shape mismatch");
    assert_eq!(gamma.len(), features, "gamma length mismatch");
    assert_eq!(beta.len(), features, "beta length mismatch");

    // ── Compute per-feature mean ──────────────────────────────────
    let mut mean = vec![0.0f32; features];
    for b in 0..batch_size {
        let row = &data[b * features..(b + 1) * features];
        let full = features / LANES;
        for i in 0..full {
            let off = i * LANES;
            unsafe {
                let vm = vld1q_f32(mean.as_ptr().add(off));
                let vr = vld1q_f32(row.as_ptr().add(off));
                let vs = vaddq_f32(vm, vr);
                vst1q_f32(mean.as_mut_ptr().add(off), vs);
            }
        }
        let base = full * LANES;
        for i in 0..features - base {
            mean[base + i] += row[base + i];
        }
    }
    let inv_n = 1.0 / batch_size as f32;
    let vinv = vdupq_n_f32(inv_n);
    {
        let full = features / LANES;
        for i in 0..full {
            let off = i * LANES;
            unsafe {
                let vm = vld1q_f32(mean.as_ptr().add(off));
                let vr = vmulq_f32(vm, vinv);
                vst1q_f32(mean.as_mut_ptr().add(off), vr);
            }
        }
        let base = full * LANES;
        for i in 0..features - base {
            mean[base + i] *= inv_n;
        }
    }

    // ── Compute per-feature variance ──────────────────────────────
    let mut var = vec![0.0f32; features];
    for b in 0..batch_size {
        let row = &data[b * features..(b + 1) * features];
        let full = features / LANES;
        for i in 0..full {
            let off = i * LANES;
            unsafe {
                let vr = vld1q_f32(row.as_ptr().add(off));
                let vm = vld1q_f32(mean.as_ptr().add(off));
                let diff = vsubq_f32(vr, vm);
                let sq = vmulq_f32(diff, diff);
                let vv = vld1q_f32(var.as_ptr().add(off));
                let vs = vaddq_f32(vv, sq);
                vst1q_f32(var.as_mut_ptr().add(off), vs);
            }
        }
        let base = full * LANES;
        for i in 0..features - base {
            let diff = row[base + i] - mean[base + i];
            var[base + i] += diff * diff;
        }
    }
    {
        let full = features / LANES;
        for i in 0..full {
            let off = i * LANES;
            unsafe {
                let vv = vld1q_f32(var.as_ptr().add(off));
                let vr = vmulq_f32(vv, vinv);
                vst1q_f32(var.as_mut_ptr().add(off), vr);
            }
        }
        let base = full * LANES;
        for i in 0..features - base {
            var[base + i] *= inv_n;
        }
    }

    // ── Normalize, scale, shift ───────────────────────────────────
    let mut out = vec![0.0f32; data.len()];
    // Pre-compute inverse standard deviation per feature.
    let mut inv_std = vec![0.0f32; features];
    for f in 0..features {
        inv_std[f] = 1.0 / (var[f] + eps).sqrt();
    }

    for b in 0..batch_size {
        let row_off = b * features;
        let full = features / LANES;
        for i in 0..full {
            let off = i * LANES;
            unsafe {
                let vx = vld1q_f32(data.as_ptr().add(row_off + off));
                let vm = vld1q_f32(mean.as_ptr().add(off));
                let vis = vld1q_f32(inv_std.as_ptr().add(off));
                let vg = vld1q_f32(gamma.as_ptr().add(off));
                let vb = vld1q_f32(beta.as_ptr().add(off));

                let diff = vsubq_f32(vx, vm);
                let normed = vmulq_f32(diff, vis);
                let scaled = vfmaq_f32(vb, vg, normed);
                vst1q_f32(out.as_mut_ptr().add(row_off + off), scaled);
            }
        }
        let base = full * LANES;
        for i in 0..features - base {
            let x = data[row_off + base + i];
            let normed = (x - mean[base + i]) * inv_std[base + i];
            out[row_off + base + i] = gamma[base + i] * normed + beta[base + i];
        }
    }

    // ── Update running statistics (EMA) ───────────────────────────
    if let Some(rm) = running_mean {
        assert_eq!(rm.len(), features, "running_mean length mismatch");
        let vm_momentum = vdupq_n_f32(momentum);
        let vm_complement = vdupq_n_f32(1.0 - momentum);
        let full = features / LANES;
        for i in 0..full {
            let off = i * LANES;
            unsafe {
                let vold = vld1q_f32(rm.as_ptr().add(off));
                let vnew = vld1q_f32(mean.as_ptr().add(off));
                let updated = vfmaq_f32(vmulq_f32(vold, vm_complement), vm_momentum, vnew);
                vst1q_f32(rm.as_mut_ptr().add(off), updated);
            }
        }
        let base = full * LANES;
        for i in 0..features - base {
            rm[base + i] = (1.0 - momentum) * rm[base + i] + momentum * mean[base + i];
        }
    }

    if let Some(rv) = running_var {
        assert_eq!(rv.len(), features, "running_var length mismatch");
        let vm_momentum = vdupq_n_f32(momentum);
        let vm_complement = vdupq_n_f32(1.0 - momentum);
        let full = features / LANES;
        for i in 0..full {
            let off = i * LANES;
            unsafe {
                let vold = vld1q_f32(rv.as_ptr().add(off));
                let vnew = vld1q_f32(var.as_ptr().add(off));
                let updated = vfmaq_f32(vmulq_f32(vold, vm_complement), vm_momentum, vnew);
                vst1q_f32(rv.as_mut_ptr().add(off), updated);
            }
        }
        let base = full * LANES;
        for i in 0..features - base {
            rv[base + i] = (1.0 - momentum) * rv[base + i] + momentum * var[base + i];
        }
    }

    out
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    // ── helpers ───────────────────────────────────────────────────

    /// Scalar reference matmul for a single (M,K)×(K,N) pair.
    fn ref_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut s = 0.0f32;
                for t in 0..k {
                    s += a[row * k + t] * b[t * n + col];
                }
                c[row * n + col] = s;
            }
        }
        c
    }

    fn assert_approx_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch {} vs {}", a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() <= tol,
                "mismatch at index {i}: {x} vs {y} (diff {})",
                (x - y).abs()
            );
        }
    }

    fn make_norm_params<'a>(
        data: &'a [f32],
        batch_size: usize,
        features: usize,
        gamma: &'a [f32],
        beta: &'a [f32],
        eps: f32,
        momentum: f32,
    ) -> BatchNormParams<'a> {
        BatchNormParams { data, batch_size, features, gamma, beta, eps, momentum }
    }

    // ── BatchConfig ──────────────────────────────────────────────

    #[test]
    fn test_batch_config_sample_elems() {
        let cfg = BatchConfig { batch_size: 4, seq_len: 8, hidden_dim: 16 };
        assert_eq!(cfg.sample_elems(), 128);
    }

    #[test]
    fn test_batch_config_total_elems() {
        let cfg = BatchConfig { batch_size: 4, seq_len: 8, hidden_dim: 16 };
        assert_eq!(cfg.total_elems(), 512);
    }

    #[test]
    fn test_batch_config_single_sample() {
        let cfg = BatchConfig { batch_size: 1, seq_len: 1, hidden_dim: 1 };
        assert_eq!(cfg.sample_elems(), 1);
        assert_eq!(cfg.total_elems(), 1);
    }

    #[test]
    fn test_batch_config_large() {
        let cfg = BatchConfig { batch_size: 64, seq_len: 512, hidden_dim: 768 };
        assert_eq!(cfg.total_elems(), 64 * 512 * 768);
    }

    #[test]
    fn test_batch_config_zero_batch() {
        let cfg = BatchConfig { batch_size: 0, seq_len: 8, hidden_dim: 16 };
        assert_eq!(cfg.total_elems(), 0);
    }

    #[test]
    fn test_batch_config_equality() {
        let a = BatchConfig { batch_size: 2, seq_len: 4, hidden_dim: 8 };
        let b = BatchConfig { batch_size: 2, seq_len: 4, hidden_dim: 8 };
        assert_eq!(a, b);
    }

    #[test]
    fn test_batch_config_copy() {
        let a = BatchConfig { batch_size: 2, seq_len: 4, hidden_dim: 8 };
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn test_batch_config_debug() {
        let cfg = BatchConfig { batch_size: 1, seq_len: 2, hidden_dim: 3 };
        let dbg = format!("{cfg:?}");
        assert!(dbg.contains("BatchConfig"));
    }

    // ── batch_matmul_neon ────────────────────────────────────────

    #[test]
    fn test_matmul_identity_2x2() {
        let a = vec![1.0, 0.0, 0.0, 1.0]; // I₂
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c = unsafe { batch_matmul_neon(&a, &b, 1, 2, 2, 2) };
        assert_approx_eq(&c, &b, 1e-5);
    }

    #[test]
    fn test_matmul_zero_matrix() {
        let a = vec![0.0; 4];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let c = unsafe { batch_matmul_neon(&a, &b, 1, 2, 2, 2) };
        assert_approx_eq(&c, &[0.0; 4], 1e-6);
    }

    #[test]
    fn test_matmul_batch_1_square() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let expected = ref_matmul(&a, &b, 2, 2, 2);
        let c = unsafe { batch_matmul_neon(&a, &b, 1, 2, 2, 2) };
        assert_approx_eq(&c, &expected, 1e-4);
    }

    #[test]
    fn test_matmul_batch_2() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0];
        let c = unsafe { batch_matmul_neon(&a, &b, 2, 2, 2, 2) };
        // Batch 0: A*I = A
        assert_approx_eq(&c[0..4], &[1.0, 2.0, 3.0, 4.0], 1e-5);
        // Batch 1: A*2I = 2A
        assert_approx_eq(&c[4..8], &[10.0, 12.0, 14.0, 16.0], 1e-5);
    }

    #[test]
    fn test_matmul_non_square() {
        // (1,3) × (3,2) → (1,2)
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let expected = ref_matmul(&a, &b, 1, 3, 2);
        let c = unsafe { batch_matmul_neon(&a, &b, 1, 1, 3, 2) };
        assert_approx_eq(&c, &expected, 1e-4);
    }

    #[test]
    fn test_matmul_k_gt_4() {
        // K=5 exercises the NEON loop + scalar tail.
        let a: Vec<f32> = (0..10).map(|x| x as f32).collect(); // 2×5
        let b: Vec<f32> = (0..15).map(|x| x as f32).collect(); // 5×3
        let expected = ref_matmul(&a, &b, 2, 5, 3);
        let c = unsafe { batch_matmul_neon(&a, &b, 1, 2, 5, 3) };
        assert_approx_eq(&c, &expected, 1e-3);
    }

    #[test]
    fn test_matmul_k_exact_4() {
        // K=4 → full NEON, no tail.
        let a: Vec<f32> = (0..8).map(|x| x as f32).collect(); // 2×4
        let b: Vec<f32> = (0..12).map(|x| x as f32).collect(); // 4×3
        let expected = ref_matmul(&a, &b, 2, 4, 3);
        let c = unsafe { batch_matmul_neon(&a, &b, 1, 2, 4, 3) };
        assert_approx_eq(&c, &expected, 1e-3);
    }

    #[test]
    fn test_matmul_k_8() {
        let a: Vec<f32> = (0..16).map(|x| x as f32).collect(); // 2×8
        let b: Vec<f32> = (0..24).map(|x| x as f32).collect(); // 8×3
        let expected = ref_matmul(&a, &b, 2, 8, 3);
        let c = unsafe { batch_matmul_neon(&a, &b, 1, 2, 8, 3) };
        assert_approx_eq(&c, &expected, 1e-2);
    }

    #[test]
    fn test_matmul_1x1() {
        let a = vec![3.0];
        let b = vec![7.0];
        let c = unsafe { batch_matmul_neon(&a, &b, 1, 1, 1, 1) };
        assert_approx_eq(&c, &[21.0], 1e-5);
    }

    #[test]
    fn test_matmul_batch_4() {
        let m = 2;
        let k = 3;
        let n = 2;
        let batch = 4;
        let a: Vec<f32> = (0..(batch * m * k) as u32).map(|x| x as f32 * 0.1).collect();
        let b: Vec<f32> = (0..(batch * k * n) as u32).map(|x| x as f32 * 0.1).collect();
        let c = unsafe { batch_matmul_neon(&a, &b, batch, m, k, n) };
        for bi in 0..batch {
            let a_s = &a[bi * m * k..(bi + 1) * m * k];
            let b_s = &b[bi * k * n..(bi + 1) * k * n];
            let exp = ref_matmul(a_s, b_s, m, k, n);
            assert_approx_eq(&c[bi * m * n..(bi + 1) * m * n], &exp, 1e-3);
        }
    }

    #[test]
    fn test_matmul_negative_values() {
        let a = vec![-1.0, -2.0, -3.0, -4.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let c = unsafe { batch_matmul_neon(&a, &b, 1, 2, 2, 2) };
        assert_approx_eq(&c, &a, 1e-5);
    }

    #[test]
    fn test_matmul_large_k() {
        let k = 17;
        let a: Vec<f32> = (0..k).map(|i| (i + 1) as f32).collect();
        let b: Vec<f32> = vec![1.0; k];
        let c = unsafe { batch_matmul_neon(&a, &b, 1, 1, k, 1) };
        let expected = (1..=k as u32).map(|x| x as f32).sum::<f32>();
        assert_approx_eq(&c, &[expected], 1e-2);
    }

    // ── batch_add_neon ───────────────────────────────────────────

    #[test]
    fn test_add_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![10.0, 20.0, 30.0, 40.0];
        let c = unsafe { batch_add_neon(&a, &b) };
        assert_approx_eq(&c, &[11.0, 22.0, 33.0, 44.0], 1e-6);
    }

    #[test]
    fn test_add_zeros() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![0.0; 4];
        let c = unsafe { batch_add_neon(&a, &b) };
        assert_approx_eq(&c, &a, 1e-6);
    }

    #[test]
    fn test_add_negative() {
        let a = vec![1.0, -2.0, 3.0, -4.0];
        let b = vec![-1.0, 2.0, -3.0, 4.0];
        let c = unsafe { batch_add_neon(&a, &b) };
        assert_approx_eq(&c, &[0.0; 4], 1e-6);
    }

    #[test]
    fn test_add_tail() {
        // len=5 → 4 NEON + 1 scalar tail
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let c = unsafe { batch_add_neon(&a, &b) };
        assert_approx_eq(&c, &[11.0, 22.0, 33.0, 44.0, 55.0], 1e-6);
    }

    #[test]
    fn test_add_single() {
        let a = vec![42.0];
        let b = vec![58.0];
        let c = unsafe { batch_add_neon(&a, &b) };
        assert_approx_eq(&c, &[100.0], 1e-6);
    }

    #[test]
    fn test_add_empty() {
        let c = unsafe { batch_add_neon(&[], &[]) };
        assert!(c.is_empty());
    }

    #[test]
    fn test_add_large() {
        let n = 1024;
        let a: Vec<f32> = (0..n).map(|x| x as f32).collect();
        let b: Vec<f32> = (0..n).map(|x| x as f32 * 2.0).collect();
        let c = unsafe { batch_add_neon(&a, &b) };
        let expected: Vec<f32> = (0..n).map(|x| x as f32 * 3.0).collect();
        assert_approx_eq(&c, &expected, 1e-4);
    }

    #[test]
    fn test_add_len_7() {
        let a: Vec<f32> = (0..7).map(|x| x as f32).collect();
        let b: Vec<f32> = vec![1.0; 7];
        let c = unsafe { batch_add_neon(&a, &b) };
        let expected: Vec<f32> = (0..7).map(|x| x as f32 + 1.0).collect();
        assert_approx_eq(&c, &expected, 1e-6);
    }

    #[test]
    fn test_add_len_3() {
        // Pure scalar tail (no NEON loop).
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let c = unsafe { batch_add_neon(&a, &b) };
        assert_approx_eq(&c, &[5.0, 7.0, 9.0], 1e-6);
    }

    // ── batch_scale_neon ─────────────────────────────────────────

    #[test]
    fn test_scale_basic() {
        let d = vec![1.0, 2.0, 3.0, 4.0];
        let c = unsafe { batch_scale_neon(&d, 2.0) };
        assert_approx_eq(&c, &[2.0, 4.0, 6.0, 8.0], 1e-6);
    }

    #[test]
    fn test_scale_zero() {
        let d = vec![1.0, 2.0, 3.0, 4.0];
        let c = unsafe { batch_scale_neon(&d, 0.0) };
        assert_approx_eq(&c, &[0.0; 4], 1e-6);
    }

    #[test]
    fn test_scale_one() {
        let d = vec![5.0, 6.0, 7.0, 8.0];
        let c = unsafe { batch_scale_neon(&d, 1.0) };
        assert_approx_eq(&c, &d, 1e-6);
    }

    #[test]
    fn test_scale_negative() {
        let d = vec![1.0, -2.0, 3.0, -4.0];
        let c = unsafe { batch_scale_neon(&d, -1.0) };
        assert_approx_eq(&c, &[-1.0, 2.0, -3.0, 4.0], 1e-6);
    }

    #[test]
    fn test_scale_tail() {
        let d = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let c = unsafe { batch_scale_neon(&d, 0.5) };
        assert_approx_eq(&c, &[1.0, 2.0, 3.0, 4.0, 5.0], 1e-6);
    }

    #[test]
    fn test_scale_empty() {
        let c = unsafe { batch_scale_neon(&[], 99.0) };
        assert!(c.is_empty());
    }

    #[test]
    fn test_scale_large() {
        let n = 513;
        let d: Vec<f32> = (0..n).map(|x| x as f32).collect();
        let c = unsafe { batch_scale_neon(&d, 3.0) };
        let expected: Vec<f32> = (0..n).map(|x| x as f32 * 3.0).collect();
        assert_approx_eq(&c, &expected, 1e-4);
    }

    #[test]
    fn test_scale_fractional() {
        let d = vec![10.0, 20.0, 30.0, 40.0];
        let c = unsafe { batch_scale_neon(&d, 0.1) };
        assert_approx_eq(&c, &[1.0, 2.0, 3.0, 4.0], 1e-5);
    }

    #[test]
    fn test_scale_single() {
        let c = unsafe { batch_scale_neon(&[7.0], 3.0) };
        assert_approx_eq(&c, &[21.0], 1e-6);
    }

    // ── batch_norm_neon ──────────────────────────────────────────

    #[test]
    fn test_norm_identity_gamma_beta() {
        // gamma=1, beta=0 → pure normalization.
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2×4
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let params = make_norm_params(&data, 2, 4, &gamma, &beta, 1e-5, 0.1);
        let out = unsafe { batch_norm_neon(&params, None, None) };
        for f in 0..4 {
            let m = (data[f] + data[4 + f]) / 2.0;
            let v = ((data[f] - m).powi(2) + (data[4 + f] - m).powi(2)) / 2.0;
            let inv = 1.0 / (v + 1e-5_f32).sqrt();
            let n0 = (data[f] - m) * inv;
            let n1 = (data[4 + f] - m) * inv;
            assert!((out[f] - n0).abs() < 1e-4, "f={f}");
            assert!((out[4 + f] - n1).abs() < 1e-4, "f={f}");
        }
    }

    #[test]
    fn test_norm_gamma_scale() {
        let data = vec![2.0, 4.0, 6.0, 8.0]; // 1×4
        let gamma = vec![2.0; 4];
        let beta = vec![0.0; 4];
        let params = make_norm_params(&data, 1, 4, &gamma, &beta, 1e-5, 0.1);
        let out = unsafe { batch_norm_neon(&params, None, None) };
        // batch_size=1 → mean=x, var=0 → normalized=(x-x)/sqrt(eps)=0 → out=0
        assert_approx_eq(&out, &[0.0; 4], 1e-3);
    }

    #[test]
    fn test_norm_beta_shift() {
        let data = vec![0.0; 8]; // 2×4 — all zeros
        let gamma = vec![1.0; 4];
        let beta = vec![5.0; 4];
        let params = make_norm_params(&data, 2, 4, &gamma, &beta, 1e-5, 0.1);
        let out = unsafe { batch_norm_neon(&params, None, None) };
        assert_approx_eq(&out, &[5.0; 8], 1e-4);
    }

    #[test]
    fn test_norm_running_mean_update() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2×4
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let mut rm = vec![0.0f32; 4];
        let momentum = 0.1;
        let params = make_norm_params(&data, 2, 4, &gamma, &beta, 1e-5, momentum);
        let _ = unsafe { batch_norm_neon(&params, Some(&mut rm), None) };
        for f in 0..4 {
            let m = (data[f] + data[4 + f]) / 2.0;
            let expected = momentum * m;
            assert!((rm[f] - expected).abs() < 1e-4, "rm[{f}]={} expected={expected}", rm[f]);
        }
    }

    #[test]
    fn test_norm_running_var_update() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2×4
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let mut rv = vec![1.0f32; 4]; // initial running var = 1
        let momentum = 0.1;
        let params = make_norm_params(&data, 2, 4, &gamma, &beta, 1e-5, momentum);
        let _ = unsafe { batch_norm_neon(&params, None, Some(&mut rv)) };
        for f in 0..4 {
            let m = (data[f] + data[4 + f]) / 2.0;
            let v = ((data[f] - m).powi(2) + (data[4 + f] - m).powi(2)) / 2.0;
            let expected = (1.0 - momentum) * 1.0 + momentum * v;
            assert!((rv[f] - expected).abs() < 1e-4, "rv[{f}]={} expected={expected}", rv[f]);
        }
    }

    #[test]
    fn test_norm_batch_1() {
        // With a single sample, mean=x, var=0, so output = beta.
        let data = vec![10.0, 20.0, 30.0, 40.0];
        let gamma = vec![1.0; 4];
        let beta = vec![7.0; 4];
        let params = make_norm_params(&data, 1, 4, &gamma, &beta, 1e-5, 0.1);
        let out = unsafe { batch_norm_neon(&params, None, None) };
        assert_approx_eq(&out, &[7.0; 4], 1e-2);
    }

    #[test]
    fn test_norm_features_not_multiple_of_4() {
        let features = 5;
        let batch = 3;
        let data: Vec<f32> = (0..(batch * features) as u32).map(|x| x as f32).collect();
        let gamma = vec![1.0; features];
        let beta = vec![0.0; features];
        let params = make_norm_params(&data, batch, features, &gamma, &beta, 1e-5, 0.1);
        let out = unsafe { batch_norm_neon(&params, None, None) };
        assert_eq!(out.len(), batch * features);
        for &v in &out {
            assert!(v.is_finite(), "non-finite value in output");
        }
    }

    #[test]
    fn test_norm_features_1() {
        let data = vec![2.0, 4.0, 6.0]; // 3 samples × 1 feature
        let gamma = vec![1.0];
        let beta = vec![0.0];
        let params = make_norm_params(&data, 3, 1, &gamma, &beta, 1e-5, 0.1);
        let out = unsafe { batch_norm_neon(&params, None, None) };
        let m = 4.0f32;
        let v = ((2.0 - m).powi(2) + (4.0 - m).powi(2) + (6.0 - m).powi(2)) / 3.0;
        let inv = 1.0 / (v + 1e-5_f32).sqrt();
        for (i, &x) in data.iter().enumerate() {
            let expected = (x - m) * inv;
            assert!((out[i] - expected).abs() < 1e-4);
        }
    }

    #[test]
    fn test_norm_large_batch() {
        let features = 8;
        let batch = 128;
        let data: Vec<f32> = (0..(batch * features) as u32).map(|x| (x % 17) as f32).collect();
        let gamma = vec![1.0; features];
        let beta = vec![0.0; features];
        let params = make_norm_params(&data, batch, features, &gamma, &beta, 1e-5, 0.1);
        let out = unsafe { batch_norm_neon(&params, None, None) };
        assert_eq!(out.len(), batch * features);
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_norm_constant_input() {
        // All values identical → var=0, normalized = 0, out = beta.
        let data = vec![3.0; 20]; // 4×5
        let gamma = vec![2.0; 5];
        let beta = vec![1.0; 5];
        let params = make_norm_params(&data, 4, 5, &gamma, &beta, 1e-5, 0.1);
        let out = unsafe { batch_norm_neon(&params, None, None) };
        assert_approx_eq(&out, &[1.0; 20], 1e-3);
    }

    #[test]
    fn test_norm_eps_stability() {
        // Near-zero variance should not produce NaN/Inf.
        let data = vec![1.0, 1.0, 1.0, 1.0, 1.0000001, 1.0000001, 1.0000001, 1.0000001];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let params = make_norm_params(&data, 2, 4, &gamma, &beta, 1e-5, 0.1);
        let out = unsafe { batch_norm_neon(&params, None, None) };
        for &v in &out {
            assert!(v.is_finite(), "eps instability produced non-finite value");
        }
    }

    #[test]
    fn test_norm_negative_values() {
        let data = vec![-1.0, -2.0, -3.0, -4.0, 1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let params = make_norm_params(&data, 2, 4, &gamma, &beta, 1e-5, 0.1);
        let out = unsafe { batch_norm_neon(&params, None, None) };
        for f in 0..4 {
            assert!((out[f] + out[4 + f]).abs() < 1e-4, "symmetry broken at feature {f}");
        }
    }

    #[test]
    fn test_norm_running_both_update() {
        let data = vec![1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0]; // 2×4
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let mut rm = vec![0.0f32; 4];
        let mut rv = vec![0.0f32; 4];
        let params = make_norm_params(&data, 2, 4, &gamma, &beta, 1e-5, 0.1);
        let _ = unsafe { batch_norm_neon(&params, Some(&mut rm), Some(&mut rv)) };
        for f in 0..4 {
            assert!(rm[f].is_finite());
            assert!(rv[f].is_finite());
            assert!(rv[f] >= 0.0, "variance must be non-negative");
        }
    }

    #[test]
    fn test_norm_momentum_zero() {
        // momentum=0 → running stats should not change.
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let mut rm = vec![99.0f32; 4];
        let params = make_norm_params(&data, 2, 4, &gamma, &beta, 1e-5, 0.0);
        let _ = unsafe { batch_norm_neon(&params, Some(&mut rm), None) };
        assert_approx_eq(&rm, &[99.0; 4], 1e-6);
    }

    #[test]
    fn test_norm_momentum_one() {
        // momentum=1 → running stats fully replaced by batch stats.
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2×4
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let mut rm = vec![99.0f32; 4];
        let params = make_norm_params(&data, 2, 4, &gamma, &beta, 1e-5, 1.0);
        let _ = unsafe { batch_norm_neon(&params, Some(&mut rm), None) };
        assert_approx_eq(&rm, &[3.0, 4.0, 5.0, 6.0], 1e-4);
    }

    // ── combined / integration ───────────────────────────────────

    #[test]
    fn test_add_then_scale() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![4.0, 3.0, 2.0, 1.0];
        let sum = unsafe { batch_add_neon(&a, &b) };
        let result = unsafe { batch_scale_neon(&sum, 0.5) };
        assert_approx_eq(&result, &[2.5, 2.5, 2.5, 2.5], 1e-6);
    }

    #[test]
    fn test_scale_then_add() {
        let a = vec![2.0, 4.0, 6.0, 8.0];
        let scaled = unsafe { batch_scale_neon(&a, 0.5) };
        let b = vec![1.0; 4];
        let result = unsafe { batch_add_neon(&scaled, &b) };
        assert_approx_eq(&result, &[2.0, 3.0, 4.0, 5.0], 1e-6);
    }

    #[test]
    fn test_matmul_then_add_bias() {
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let c = unsafe { batch_matmul_neon(&a, &b, 1, 2, 2, 2) };
        let bias = vec![10.0, 20.0, 10.0, 20.0];
        let result = unsafe { batch_add_neon(&c, &bias) };
        assert_approx_eq(&result, &[12.0, 23.0, 14.0, 25.0], 1e-5);
    }

    #[test]
    fn test_norm_then_scale() {
        let data = vec![-1.0, 1.0, -1.0, 1.0]; // 2×2
        let gamma = vec![1.0; 2];
        let beta = vec![0.0; 2];
        let params = make_norm_params(&data, 2, 2, &gamma, &beta, 1e-5, 0.1);
        let normed = unsafe { batch_norm_neon(&params, None, None) };
        let scaled = unsafe { batch_scale_neon(&normed, 2.0) };
        assert_eq!(scaled.len(), 4);
        for &v in &scaled {
            assert!(v.is_finite());
        }
    }

    // ── edge cases & stress ──────────────────────────────────────

    #[test]
    fn test_matmul_batch_1_wide() {
        // (1,16) × (16,1) → (1,1) — exercises multiple NEON iterations.
        let k = 16;
        let a = vec![1.0; k];
        let b = vec![1.0; k];
        let c = unsafe { batch_matmul_neon(&a, &b, 1, 1, k, 1) };
        assert_approx_eq(&c, &[k as f32], 1e-4);
    }

    #[test]
    fn test_add_exact_multiple_of_lanes() {
        let n = 16;
        let a: Vec<f32> = (0..n).map(|x| x as f32).collect();
        let b: Vec<f32> = vec![1.0; n];
        let c = unsafe { batch_add_neon(&a, &b) };
        let expected: Vec<f32> = (0..n).map(|x| x as f32 + 1.0).collect();
        assert_approx_eq(&c, &expected, 1e-6);
    }

    #[test]
    fn test_scale_large_value() {
        let d = vec![1e10, -1e10, 1e-10, -1e-10];
        let c = unsafe { batch_scale_neon(&d, 2.0) };
        assert_approx_eq(&c, &[2e10, -2e10, 2e-10, -2e-10], 1e4);
    }

    #[test]
    fn test_norm_features_16() {
        let features = 16;
        let batch = 4;
        let data: Vec<f32> = (0..(batch * features) as u32).map(|x| x as f32).collect();
        let gamma = vec![1.0; features];
        let beta = vec![0.0; features];
        let params = make_norm_params(&data, batch, features, &gamma, &beta, 1e-5, 0.1);
        let out = unsafe { batch_norm_neon(&params, None, None) };
        assert_eq!(out.len(), batch * features);
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_norm_batch_64() {
        let features = 4;
        let batch = 64;
        let data: Vec<f32> = (0..(batch * features) as u32).map(|x| x as f32 * 0.01).collect();
        let gamma = vec![1.0; features];
        let beta = vec![0.0; features];
        let mut rm = vec![0.0f32; features];
        let mut rv = vec![0.0f32; features];
        let params = make_norm_params(&data, batch, features, &gamma, &beta, 1e-5, 0.1);
        let out = unsafe { batch_norm_neon(&params, Some(&mut rm), Some(&mut rv)) };
        assert_eq!(out.len(), batch * features);
        for &v in &out {
            assert!(v.is_finite());
        }
        for &v in &rm {
            assert!(v.is_finite());
        }
        for &v in &rv {
            assert!(v.is_finite());
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn test_matmul_batch_8_rectangular() {
        let batch = 8;
        let m = 3;
        let k = 5;
        let n = 2;
        let a: Vec<f32> = (0..(batch * m * k) as u32).map(|x| (x as f32) * 0.01).collect();
        let b: Vec<f32> = (0..(batch * k * n) as u32).map(|x| (x as f32) * 0.01).collect();
        let c = unsafe { batch_matmul_neon(&a, &b, batch, m, k, n) };
        assert_eq!(c.len(), batch * m * n);
        for bi in 0..batch {
            let a_s = &a[bi * m * k..(bi + 1) * m * k];
            let b_s = &b[bi * k * n..(bi + 1) * k * n];
            let exp = ref_matmul(a_s, b_s, m, k, n);
            assert_approx_eq(&c[bi * m * n..(bi + 1) * m * n], &exp, 1e-2);
        }
    }

    #[test]
    fn test_add_commutative() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let ab = unsafe { batch_add_neon(&a, &b) };
        let ba = unsafe { batch_add_neon(&b, &a) };
        assert_approx_eq(&ab, &ba, 1e-6);
    }

    #[test]
    fn test_scale_distributive_over_add() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let s = 3.0;
        let sum = unsafe { batch_add_neon(&a, &b) };
        let lhs = unsafe { batch_scale_neon(&sum, s) };
        let sa = unsafe { batch_scale_neon(&a, s) };
        let sb = unsafe { batch_scale_neon(&b, s) };
        let rhs = unsafe { batch_add_neon(&sa, &sb) };
        assert_approx_eq(&lhs, &rhs, 1e-4);
    }

    #[test]
    fn test_batch_config_clone() {
        let cfg = BatchConfig { batch_size: 16, seq_len: 32, hidden_dim: 64 };
        let cfg2 = cfg.clone();
        assert_eq!(cfg, cfg2);
    }

    #[test]
    fn test_matmul_associative_scaling() {
        // (scale*A) * B should ≈ scale * (A*B)
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let s = 2.0;
        let sa = unsafe { batch_scale_neon(&a, s) };
        let lhs = unsafe { batch_matmul_neon(&sa, &b, 1, 2, 2, 2) };
        let ab = unsafe { batch_matmul_neon(&a, &b, 1, 2, 2, 2) };
        let rhs = unsafe { batch_scale_neon(&ab, s) };
        assert_approx_eq(&lhs, &rhs, 1e-3);
    }

    #[test]
    fn test_norm_idempotent_constant() {
        // Normalizing already-constant data twice should give the same result.
        let data = vec![5.0; 12]; // 3×4
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let params1 = make_norm_params(&data, 3, 4, &gamma, &beta, 1e-5, 0.1);
        let out1 = unsafe { batch_norm_neon(&params1, None, None) };
        let params2 = make_norm_params(&out1, 3, 4, &gamma, &beta, 1e-5, 0.1);
        let out2 = unsafe { batch_norm_neon(&params2, None, None) };
        assert_approx_eq(&out1, &out2, 1e-3);
    }
}

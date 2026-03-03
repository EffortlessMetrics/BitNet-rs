//! NEON-optimized tensor parallel v2 kernel for Apple Silicon.
//!
//! Provides tensor parallelism primitives for multi-core Apple Silicon
//! inference: all-reduce sum, scatter, gather, K-dimension partitioned
//! matmul, and reduce-scatter — each with a scalar fallback.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── scalar fallbacks ───────────────────────────────────────────────

/// Scalar fallback: element-wise accumulate `local` into `global`.
pub fn all_reduce_sum_scalar(local: &[f32], global: &mut [f32], len: usize) {
    assert!(
        len <= local.len() && len <= global.len(),
        "all_reduce_sum_scalar: len {len} exceeds slice lengths ({}, {})",
        local.len(),
        global.len(),
    );
    for i in 0..len {
        global[i] += local[i];
    }
}

/// Scalar fallback: split `input` into equal `partition_size` chunks.
pub fn scatter_scalar(input: &[f32], outputs: &mut [&mut [f32]], partition_size: usize) {
    for (p, out) in outputs.iter_mut().enumerate() {
        let start = p * partition_size;
        let end = (start + partition_size).min(input.len());
        let copy_len = end.saturating_sub(start).min(out.len());
        out[..copy_len].copy_from_slice(&input[start..start + copy_len]);
    }
}

/// Scalar fallback: concatenate partition slices into `output`.
pub fn gather_scalar(inputs: &[&[f32]], output: &mut [f32], partition_size: usize) {
    for (p, inp) in inputs.iter().enumerate() {
        let start = p * partition_size;
        let copy_len = partition_size.min(inp.len()).min(output.len().saturating_sub(start));
        output[start..start + copy_len].copy_from_slice(&inp[..copy_len]);
    }
}

/// Scalar fallback: partial matmul over K-range `[k_start..k_end)`.
///
/// Computes C += A[:,k_start..k_end] * B[k_start..k_end,:] (row-major).
pub fn partition_matmul_scalar(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    k_start: usize,
    k_end: usize,
) {
    assert!(k_end <= k, "partition_matmul_scalar: k_end {k_end} > k {k}");
    assert!(k_start <= k_end, "partition_matmul_scalar: k_start {k_start} > k_end {k_end}");
    assert!(a.len() >= m * k, "partition_matmul_scalar: a too short");
    assert!(b.len() >= k * n, "partition_matmul_scalar: b too short");
    assert!(c.len() >= m * n, "partition_matmul_scalar: c too short");

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in k_start..k_end {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] += sum;
        }
    }
}

/// Scalar fallback: reduce-scatter — reduce full input then write the
/// partition owned by `partition_id`.
pub fn reduce_scatter_scalar(
    input: &[f32],
    output: &mut [f32],
    num_partitions: usize,
    partition_id: usize,
) {
    assert!(num_partitions > 0, "reduce_scatter_scalar: num_partitions must be > 0");
    assert!(
        partition_id < num_partitions,
        "reduce_scatter_scalar: partition_id {partition_id} >= num_partitions {num_partitions}",
    );
    let total = input.len();
    let partition_size = total / num_partitions;
    let start = partition_id * partition_size;
    let end = if partition_id == num_partitions - 1 { total } else { start + partition_size };
    let copy_len = (end - start).min(output.len());
    output[..copy_len].copy_from_slice(&input[start..start + copy_len]);
}

// ── NEON-optimised implementations ─────────────────────────────────

/// NEON-accelerated element-wise accumulate: `global[i] += local[i]`.
///
/// Processes 4 floats per iteration via `vaddq_f32`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn all_reduce_sum_neon(local: &[f32], global: &mut [f32], len: usize) {
    assert!(
        len <= local.len() && len <= global.len(),
        "all_reduce_sum_neon: len {len} exceeds slice lengths ({}, {})",
        local.len(),
        global.len(),
    );
    let l_ptr = local.as_ptr();
    let g_ptr = global.as_mut_ptr();
    let chunks = len / 4;
    let remainder = len % 4;

    for i in 0..chunks {
        let offset = i * 4;
        unsafe {
            let lv = vld1q_f32(l_ptr.add(offset));
            let gv = vld1q_f32(g_ptr.add(offset));
            let sv = vaddq_f32(gv, lv);
            vst1q_f32(g_ptr.add(offset), sv);
        }
    }
    let tail = chunks * 4;
    for i in 0..remainder {
        global[tail + i] += local[tail + i];
    }
}

/// NEON-accelerated scatter: split `input` into equal `partition_size`
/// chunks across `outputs`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn scatter_neon(input: &[f32], outputs: &mut [&mut [f32]], partition_size: usize) {
    let in_ptr = input.as_ptr();
    for (p, out) in outputs.iter_mut().enumerate() {
        let start = p * partition_size;
        let end = (start + partition_size).min(input.len());
        let copy_len = end.saturating_sub(start).min(out.len());
        let o_ptr = out.as_mut_ptr();
        let chunks = copy_len / 4;
        let remainder = copy_len % 4;

        for i in 0..chunks {
            let offset = i * 4;
            unsafe {
                let v = vld1q_f32(in_ptr.add(start + offset));
                vst1q_f32(o_ptr.add(offset), v);
            }
        }
        let tail = chunks * 4;
        for i in 0..remainder {
            out[tail + i] = input[start + tail + i];
        }
    }
}

/// NEON-accelerated gather: merge partition slices into `output`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn gather_neon(inputs: &[&[f32]], output: &mut [f32], partition_size: usize) {
    let o_ptr = output.as_mut_ptr();
    for (p, inp) in inputs.iter().enumerate() {
        let start = p * partition_size;
        let copy_len = partition_size.min(inp.len()).min(output.len().saturating_sub(start));
        let chunks = copy_len / 4;
        let remainder = copy_len % 4;
        let i_ptr = inp.as_ptr();

        for i in 0..chunks {
            let offset = i * 4;
            unsafe {
                let v = vld1q_f32(i_ptr.add(offset));
                vst1q_f32(o_ptr.add(start + offset), v);
            }
        }
        let tail = chunks * 4;
        for i in 0..remainder {
            output[start + tail + i] = inp[tail + i];
        }
    }
}

/// NEON-accelerated partial matmul for K-dimension partitioning.
///
/// Computes C += A[:,k_start..k_end] * B[k_start..k_end,:] (row-major)
/// using `vfmaq_f32` for the inner dot product.
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn partition_matmul_neon(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    k_start: usize,
    k_end: usize,
) {
    assert!(k_end <= k, "partition_matmul_neon: k_end {k_end} > k {k}");
    assert!(k_start <= k_end, "partition_matmul_neon: k_start {k_start} > k_end {k_end}");
    assert!(a.len() >= m * k, "partition_matmul_neon: a too short");
    assert!(b.len() >= k * n, "partition_matmul_neon: b too short");
    assert!(c.len() >= m * n, "partition_matmul_neon: c too short");

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let c_ptr = c.as_mut_ptr();

    for i in 0..m {
        let a_row = i * k;
        let c_row = i * n;
        // For each column of C, vectorise across K.
        for j in 0..n {
            let k_len = k_end - k_start;
            let chunks = k_len / 4;
            let remainder = k_len % 4;

            unsafe {
                let mut acc = vdupq_n_f32(0.0);
                for c_idx in 0..chunks {
                    let kk = k_start + c_idx * 4;
                    let av = vld1q_f32(a_ptr.add(a_row + kk));
                    // B is row-major: need b[kk][j], b[kk+1][j], ...
                    let bv = vsetq_lane_f32::<3>(
                        *b_ptr.add((kk + 3) * n + j),
                        vsetq_lane_f32::<2>(
                            *b_ptr.add((kk + 2) * n + j),
                            vsetq_lane_f32::<1>(
                                *b_ptr.add((kk + 1) * n + j),
                                vdupq_n_f32(*b_ptr.add(kk * n + j)),
                            ),
                        ),
                    );
                    acc = vfmaq_f32(acc, av, bv);
                }

                let mut sum = vaddvq_f32(acc);
                let tail_start = k_start + chunks * 4;
                for r in 0..remainder {
                    let kk = tail_start + r;
                    sum += *a_ptr.add(a_row + kk) * *b_ptr.add(kk * n + j);
                }
                *c_ptr.add(c_row + j) += sum;
            }
        }
    }
}

/// NEON-accelerated reduce-scatter: copies the partition slice belonging
/// to `partition_id` into `output` using vectorised loads/stores.
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn reduce_scatter_neon(
    input: &[f32],
    output: &mut [f32],
    num_partitions: usize,
    partition_id: usize,
) {
    assert!(num_partitions > 0, "reduce_scatter_neon: num_partitions must be > 0");
    assert!(
        partition_id < num_partitions,
        "reduce_scatter_neon: partition_id {partition_id} >= num_partitions {num_partitions}",
    );
    let total = input.len();
    let partition_size = total / num_partitions;
    let start = partition_id * partition_size;
    let end = if partition_id == num_partitions - 1 { total } else { start + partition_size };
    let copy_len = (end - start).min(output.len());

    let in_ptr = input.as_ptr();
    let o_ptr = output.as_mut_ptr();
    let chunks = copy_len / 4;
    let remainder = copy_len % 4;

    for i in 0..chunks {
        let offset = i * 4;
        unsafe {
            let v = vld1q_f32(in_ptr.add(start + offset));
            vst1q_f32(o_ptr.add(offset), v);
        }
    }
    let tail = chunks * 4;
    for i in 0..remainder {
        output[tail + i] = input[start + tail + i];
    }
}

// ── tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: fill vec with a pattern.
    fn pattern(len: usize, seed: f32) -> Vec<f32> {
        (0..len).map(|i| seed + i as f32 * 0.5).collect()
    }

    // Helper: naive reference matmul C = A * B (full K).
    fn naive_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for kk in 0..k {
                    s += a[i * k + kk] * b[kk * n + j];
                }
                c[i * n + j] = s;
            }
        }
        c
    }

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    fn assert_slices_approx(a: &[f32], b: &[f32], eps: f32, msg: &str) {
        assert_eq!(a.len(), b.len(), "{msg}: length mismatch {} vs {}", a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(approx_eq(x, y, eps), "{msg}: index {i}: {x} vs {y} (eps={eps})");
        }
    }

    // ── all_reduce_sum scalar ──────────────────────────────────────

    #[test]
    fn test_all_reduce_sum_scalar_basic() {
        let local = vec![1.0, 2.0, 3.0, 4.0];
        let mut global = vec![10.0, 20.0, 30.0, 40.0];
        all_reduce_sum_scalar(&local, &mut global, 4);
        assert_eq!(global, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_all_reduce_sum_scalar_partial_len() {
        let local = vec![1.0, 2.0, 3.0, 4.0];
        let mut global = vec![10.0, 20.0, 30.0, 40.0];
        all_reduce_sum_scalar(&local, &mut global, 2);
        assert_eq!(global, vec![11.0, 22.0, 30.0, 40.0]);
    }

    #[test]
    fn test_all_reduce_sum_scalar_zero_len() {
        let local = vec![1.0];
        let mut global = vec![10.0];
        all_reduce_sum_scalar(&local, &mut global, 0);
        assert_eq!(global, vec![10.0]);
    }

    #[test]
    fn test_all_reduce_sum_scalar_large() {
        let n = 1024;
        let local = vec![1.0; n];
        let mut global = vec![0.0; n];
        all_reduce_sum_scalar(&local, &mut global, n);
        for &v in &global {
            assert_eq!(v, 1.0);
        }
    }

    #[test]
    fn test_all_reduce_sum_scalar_odd_len() {
        let local = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut global = vec![0.0; 5];
        all_reduce_sum_scalar(&local, &mut global, 5);
        assert_eq!(global, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    // ── all_reduce_sum NEON ────────────────────────────────────────

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_all_reduce_sum_neon_basic() {
        let local = vec![1.0, 2.0, 3.0, 4.0];
        let mut global = vec![10.0, 20.0, 30.0, 40.0];
        unsafe { all_reduce_sum_neon(&local, &mut global, 4) };
        assert_eq!(global, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_all_reduce_sum_neon_remainder() {
        let local = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut global = vec![0.0; 7];
        unsafe { all_reduce_sum_neon(&local, &mut global, 7) };
        assert_eq!(global, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_all_reduce_sum_neon_zero_len() {
        let local = vec![1.0];
        let mut global = vec![10.0];
        unsafe { all_reduce_sum_neon(&local, &mut global, 0) };
        assert_eq!(global, vec![10.0]);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_all_reduce_sum_neon_parity_with_scalar() {
        let n = 137;
        let local = pattern(n, 0.1);
        let mut g_neon = pattern(n, 1.0);
        let mut g_scalar = g_neon.clone();
        unsafe { all_reduce_sum_neon(&local, &mut g_neon, n) };
        all_reduce_sum_scalar(&local, &mut g_scalar, n);
        assert_slices_approx(&g_neon, &g_scalar, 1e-5, "all_reduce_sum parity");
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_all_reduce_sum_neon_large() {
        let n = 4096;
        let local = vec![0.25; n];
        let mut global = vec![0.75; n];
        unsafe { all_reduce_sum_neon(&local, &mut global, n) };
        for &v in &global {
            assert!(approx_eq(v, 1.0, 1e-6));
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_all_reduce_sum_neon_multiple_rounds() {
        let n = 16;
        let mut global = vec![0.0; n];
        for _ in 0..4 {
            let local = vec![1.0; n];
            unsafe { all_reduce_sum_neon(&local, &mut global, n) };
        }
        for &v in &global {
            assert!(approx_eq(v, 4.0, 1e-6));
        }
    }

    // ── scatter scalar ─────────────────────────────────────────────

    #[test]
    fn test_scatter_scalar_2_partitions() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut p0 = vec![0.0; 2];
        let mut p1 = vec![0.0; 2];
        {
            let mut outputs: Vec<&mut [f32]> = vec![&mut p0, &mut p1];
            scatter_scalar(&input, &mut outputs, 2);
        }
        assert_eq!(p0, vec![1.0, 2.0]);
        assert_eq!(p1, vec![3.0, 4.0]);
    }

    #[test]
    fn test_scatter_scalar_4_partitions() {
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut parts: Vec<Vec<f32>> = (0..4).map(|_| vec![0.0; 4]).collect();
        {
            let mut slices: Vec<&mut [f32]> = parts.iter_mut().map(|p| p.as_mut_slice()).collect();
            scatter_scalar(&input, &mut slices, 4);
        }
        for p in 0..4 {
            let expected: Vec<f32> = (0..4).map(|i| (p * 4 + i) as f32).collect();
            assert_eq!(parts[p], expected);
        }
    }

    #[test]
    fn test_scatter_scalar_single_partition() {
        let input = vec![1.0, 2.0, 3.0];
        let mut p0 = vec![0.0; 3];
        {
            let mut outputs: Vec<&mut [f32]> = vec![&mut p0];
            scatter_scalar(&input, &mut outputs, 3);
        }
        assert_eq!(p0, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_scatter_scalar_empty() {
        let input: Vec<f32> = vec![];
        let mut p0: Vec<f32> = vec![];
        {
            let mut outputs: Vec<&mut [f32]> = vec![&mut p0];
            scatter_scalar(&input, &mut outputs, 0);
        }
        assert!(p0.is_empty());
    }

    // ── scatter NEON ───────────────────────────────────────────────

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_scatter_neon_2_partitions() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut p0 = vec![0.0; 4];
        let mut p1 = vec![0.0; 4];
        unsafe {
            let mut outputs: Vec<&mut [f32]> = vec![&mut p0, &mut p1];
            scatter_neon(&input, &mut outputs, 4);
        }
        assert_eq!(p0, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(p1, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_scatter_neon_4_partitions() {
        let input: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let mut parts: Vec<Vec<f32>> = (0..4).map(|_| vec![0.0; 8]).collect();
        unsafe {
            let mut slices: Vec<&mut [f32]> = parts.iter_mut().map(|p| p.as_mut_slice()).collect();
            scatter_neon(&input, &mut slices, 8);
        }
        for p in 0..4 {
            let expected: Vec<f32> = (0..8).map(|i| (p * 8 + i) as f32).collect();
            assert_eq!(parts[p], expected);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_scatter_neon_odd_partition_size() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut p0 = vec![0.0; 3];
        let mut p1 = vec![0.0; 3];
        unsafe {
            let mut outputs: Vec<&mut [f32]> = vec![&mut p0, &mut p1];
            scatter_neon(&input, &mut outputs, 3);
        }
        assert_eq!(p0, vec![1.0, 2.0, 3.0]);
        assert_eq!(p1, vec![4.0, 5.0, 6.0]);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_scatter_neon_parity_with_scalar() {
        let n = 64;
        let np = 4;
        let ps = n / np;
        let input = pattern(n, 0.0);
        let mut neon_parts: Vec<Vec<f32>> = (0..np).map(|_| vec![0.0; ps]).collect();
        let mut scal_parts: Vec<Vec<f32>> = (0..np).map(|_| vec![0.0; ps]).collect();
        unsafe {
            let mut slices: Vec<&mut [f32]> =
                neon_parts.iter_mut().map(|p| p.as_mut_slice()).collect();
            scatter_neon(&input, &mut slices, ps);
        }
        {
            let mut slices: Vec<&mut [f32]> =
                scal_parts.iter_mut().map(|p| p.as_mut_slice()).collect();
            scatter_scalar(&input, &mut slices, ps);
        }
        for p in 0..np {
            assert_slices_approx(&neon_parts[p], &scal_parts[p], 1e-6, "scatter parity");
        }
    }

    // ── gather scalar ──────────────────────────────────────────────

    #[test]
    fn test_gather_scalar_2_partitions() {
        let p0 = vec![1.0, 2.0];
        let p1 = vec![3.0, 4.0];
        let inputs: Vec<&[f32]> = vec![&p0, &p1];
        let mut output = vec![0.0; 4];
        gather_scalar(&inputs, &mut output, 2);
        assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_gather_scalar_4_partitions() {
        let parts: Vec<Vec<f32>> = (0..4).map(|p| vec![p as f32; 4]).collect();
        let inputs: Vec<&[f32]> = parts.iter().map(|p| p.as_slice()).collect();
        let mut output = vec![0.0; 16];
        gather_scalar(&inputs, &mut output, 4);
        for p in 0..4 {
            for i in 0..4 {
                assert_eq!(output[p * 4 + i], p as f32);
            }
        }
    }

    #[test]
    fn test_gather_scalar_single_partition() {
        let p0 = vec![5.0, 6.0, 7.0];
        let inputs: Vec<&[f32]> = vec![&p0];
        let mut output = vec![0.0; 3];
        gather_scalar(&inputs, &mut output, 3);
        assert_eq!(output, vec![5.0, 6.0, 7.0]);
    }

    // ── gather NEON ────────────────────────────────────────────────

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_gather_neon_2_partitions() {
        let p0 = vec![1.0, 2.0, 3.0, 4.0];
        let p1 = vec![5.0, 6.0, 7.0, 8.0];
        let inputs: Vec<&[f32]> = vec![&p0, &p1];
        let mut output = vec![0.0; 8];
        unsafe { gather_neon(&inputs, &mut output, 4) };
        assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_gather_neon_4_partitions() {
        let parts: Vec<Vec<f32>> =
            (0..4).map(|p| (0..8).map(|i| (p * 8 + i) as f32).collect()).collect();
        let inputs: Vec<&[f32]> = parts.iter().map(|p| p.as_slice()).collect();
        let mut output = vec![0.0; 32];
        unsafe { gather_neon(&inputs, &mut output, 8) };
        let expected: Vec<f32> = (0..32).map(|i| i as f32).collect();
        assert_eq!(output, expected);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_gather_neon_odd_size() {
        let p0 = vec![1.0, 2.0, 3.0];
        let p1 = vec![4.0, 5.0, 6.0];
        let inputs: Vec<&[f32]> = vec![&p0, &p1];
        let mut output = vec![0.0; 6];
        unsafe { gather_neon(&inputs, &mut output, 3) };
        assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_gather_neon_parity_with_scalar() {
        let n = 64;
        let np = 8;
        let ps = n / np;
        let parts: Vec<Vec<f32>> = (0..np).map(|p| pattern(ps, p as f32)).collect();
        let inputs: Vec<&[f32]> = parts.iter().map(|p| p.as_slice()).collect();
        let mut neon_out = vec![0.0; n];
        let mut scal_out = vec![0.0; n];
        unsafe { gather_neon(&inputs, &mut neon_out, ps) };
        gather_scalar(&inputs, &mut scal_out, ps);
        assert_slices_approx(&neon_out, &scal_out, 1e-6, "gather parity");
    }

    // ── scatter + gather round-trip ────────────────────────────────

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_scatter_gather_roundtrip_neon() {
        let n = 128;
        let np = 4;
        let ps = n / np;
        let input = pattern(n, 3.14);
        let mut parts: Vec<Vec<f32>> = (0..np).map(|_| vec![0.0; ps]).collect();
        unsafe {
            let mut slices: Vec<&mut [f32]> = parts.iter_mut().map(|p| p.as_mut_slice()).collect();
            scatter_neon(&input, &mut slices, ps);
        }
        let inputs: Vec<&[f32]> = parts.iter().map(|p| p.as_slice()).collect();
        let mut output = vec![0.0; n];
        unsafe { gather_neon(&inputs, &mut output, ps) };
        assert_slices_approx(&output, &input, 1e-6, "scatter-gather roundtrip");
    }

    #[test]
    fn test_scatter_gather_roundtrip_scalar() {
        let n = 100;
        let np = 4;
        let ps = n / np;
        let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let mut parts: Vec<Vec<f32>> = (0..np).map(|_| vec![0.0; ps]).collect();
        {
            let mut slices: Vec<&mut [f32]> = parts.iter_mut().map(|p| p.as_mut_slice()).collect();
            scatter_scalar(&input, &mut slices, ps);
        }
        let inputs: Vec<&[f32]> = parts.iter().map(|p| p.as_slice()).collect();
        let mut output = vec![0.0; n];
        gather_scalar(&inputs, &mut output, ps);
        // Only ps*np elements are roundtripped for evenly divisible input
        assert_slices_approx(&output, &input, 1e-6, "scatter-gather roundtrip scalar");
    }

    // ── partition_matmul scalar ────────────────────────────────────

    #[test]
    fn test_partition_matmul_scalar_full_k() {
        let m = 2;
        let n = 3;
        let k = 4;
        let a: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.5).collect();
        let expected = naive_matmul(&a, &b, m, n, k);
        let mut c = vec![0.0; m * n];
        partition_matmul_scalar(&a, &b, &mut c, m, n, k, 0, k);
        assert_slices_approx(&c, &expected, 1e-4, "partition_matmul full K");
    }

    #[test]
    fn test_partition_matmul_scalar_split_k() {
        let m = 2;
        let n = 3;
        let k = 8;
        let a: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.5).collect();
        let expected = naive_matmul(&a, &b, m, n, k);
        let mut c = vec![0.0; m * n];
        partition_matmul_scalar(&a, &b, &mut c, m, n, k, 0, 4);
        partition_matmul_scalar(&a, &b, &mut c, m, n, k, 4, 8);
        assert_slices_approx(&c, &expected, 1e-4, "partition_matmul split K");
    }

    #[test]
    fn test_partition_matmul_scalar_identity() {
        let n = 3;
        let a: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let b: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut c = vec![0.0; n * n];
        partition_matmul_scalar(&a, &b, &mut c, n, n, n, 0, n);
        assert_slices_approx(&c, &b, 1e-4, "matmul identity");
    }

    #[test]
    fn test_partition_matmul_scalar_zero_range() {
        let m = 2;
        let n = 2;
        let k = 4;
        let a = vec![1.0; m * k];
        let b = vec![1.0; k * n];
        let mut c = vec![0.0; m * n];
        partition_matmul_scalar(&a, &b, &mut c, m, n, k, 2, 2);
        assert_eq!(c, vec![0.0; m * n]);
    }

    #[test]
    fn test_partition_matmul_scalar_4_partitions() {
        let m = 4;
        let n = 4;
        let k = 16;
        let a = pattern(m * k, 0.1);
        let b = pattern(k * n, 0.2);
        let expected = naive_matmul(&a, &b, m, n, k);
        let mut c = vec![0.0; m * n];
        let pk = k / 4;
        for p in 0..4 {
            partition_matmul_scalar(&a, &b, &mut c, m, n, k, p * pk, (p + 1) * pk);
        }
        assert_slices_approx(&c, &expected, 1e-2, "partition_matmul 4-part");
    }

    // ── partition_matmul NEON ──────────────────────────────────────

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_partition_matmul_neon_full_k() {
        let m = 2;
        let n = 3;
        let k = 4;
        let a: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.5).collect();
        let expected = naive_matmul(&a, &b, m, n, k);
        let mut c = vec![0.0; m * n];
        unsafe { partition_matmul_neon(&a, &b, &mut c, m, n, k, 0, k) };
        assert_slices_approx(&c, &expected, 1e-3, "neon partition_matmul full K");
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_partition_matmul_neon_split_k() {
        let m = 2;
        let n = 3;
        let k = 8;
        let a: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.5).collect();
        let expected = naive_matmul(&a, &b, m, n, k);
        let mut c = vec![0.0; m * n];
        unsafe {
            partition_matmul_neon(&a, &b, &mut c, m, n, k, 0, 4);
            partition_matmul_neon(&a, &b, &mut c, m, n, k, 4, 8);
        }
        assert_slices_approx(&c, &expected, 1e-3, "neon partition_matmul split K");
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_partition_matmul_neon_parity_with_scalar() {
        let m = 4;
        let n = 5;
        let k = 16;
        let a = pattern(m * k, 0.1);
        let b = pattern(k * n, 0.2);
        let mut c_neon = vec![0.0; m * n];
        let mut c_scal = vec![0.0; m * n];
        unsafe { partition_matmul_neon(&a, &b, &mut c_neon, m, n, k, 0, k) };
        partition_matmul_scalar(&a, &b, &mut c_scal, m, n, k, 0, k);
        assert_slices_approx(&c_neon, &c_scal, 1e-2, "matmul neon vs scalar parity");
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_partition_matmul_neon_4_partitions() {
        let m = 4;
        let n = 4;
        let k = 16;
        let a = pattern(m * k, 0.1);
        let b = pattern(k * n, 0.2);
        let expected = naive_matmul(&a, &b, m, n, k);
        let mut c = vec![0.0; m * n];
        let pk = k / 4;
        for p in 0..4 {
            unsafe { partition_matmul_neon(&a, &b, &mut c, m, n, k, p * pk, (p + 1) * pk) };
        }
        assert_slices_approx(&c, &expected, 1e-2, "neon partition_matmul 4-part");
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_partition_matmul_neon_identity() {
        let n = 4;
        let mut id = vec![0.0; n * n];
        for i in 0..n {
            id[i * n + i] = 1.0;
        }
        let b = pattern(n * n, 1.0);
        let mut c = vec![0.0; n * n];
        unsafe { partition_matmul_neon(&id, &b, &mut c, n, n, n, 0, n) };
        assert_slices_approx(&c, &b, 1e-4, "neon matmul identity");
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_partition_matmul_neon_zero_range() {
        let m = 2;
        let n = 2;
        let k = 4;
        let a = vec![1.0; m * k];
        let b = vec![1.0; k * n];
        let mut c = vec![0.0; m * n];
        unsafe { partition_matmul_neon(&a, &b, &mut c, m, n, k, 2, 2) };
        assert_eq!(c, vec![0.0; m * n]);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_partition_matmul_neon_8_partitions() {
        let m = 4;
        let n = 4;
        let k = 32;
        let a = pattern(m * k, 0.05);
        let b = pattern(k * n, 0.1);
        let expected = naive_matmul(&a, &b, m, n, k);
        let mut c = vec![0.0; m * n];
        let pk = k / 8;
        for p in 0..8 {
            unsafe { partition_matmul_neon(&a, &b, &mut c, m, n, k, p * pk, (p + 1) * pk) };
        }
        assert_slices_approx(&c, &expected, 0.5, "neon partition_matmul 8-part");
    }

    // ── reduce_scatter scalar ──────────────────────────────────────

    #[test]
    fn test_reduce_scatter_scalar_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 2];
        reduce_scatter_scalar(&input, &mut output, 2, 0);
        assert_eq!(output, vec![1.0, 2.0]);
    }

    #[test]
    fn test_reduce_scatter_scalar_second_partition() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 2];
        reduce_scatter_scalar(&input, &mut output, 2, 1);
        assert_eq!(output, vec![3.0, 4.0]);
    }

    #[test]
    fn test_reduce_scatter_scalar_4_partitions() {
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
        for pid in 0..4 {
            let mut output = vec![0.0; 4];
            reduce_scatter_scalar(&input, &mut output, 4, pid);
            let expected: Vec<f32> = (0..4).map(|i| (pid * 4 + i) as f32).collect();
            assert_eq!(output, expected);
        }
    }

    #[test]
    fn test_reduce_scatter_scalar_single_partition() {
        let input = vec![1.0, 2.0, 3.0];
        let mut output = vec![0.0; 3];
        reduce_scatter_scalar(&input, &mut output, 1, 0);
        assert_eq!(output, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_reduce_scatter_scalar_last_gets_remainder() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut output = vec![0.0; 3];
        reduce_scatter_scalar(&input, &mut output, 2, 1);
        // partition_size = 5/2 = 2, last partition gets [2..5]
        assert_eq!(output, vec![3.0, 4.0, 5.0]);
    }

    // ── reduce_scatter NEON ────────────────────────────────────────

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_reduce_scatter_neon_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = vec![0.0; 4];
        unsafe { reduce_scatter_neon(&input, &mut output, 2, 0) };
        assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_reduce_scatter_neon_second_partition() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = vec![0.0; 4];
        unsafe { reduce_scatter_neon(&input, &mut output, 2, 1) };
        assert_eq!(output, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_reduce_scatter_neon_4_partitions() {
        let input: Vec<f32> = (0..32).map(|i| i as f32).collect();
        for pid in 0..4 {
            let mut output = vec![0.0; 8];
            unsafe { reduce_scatter_neon(&input, &mut output, 4, pid) };
            let expected: Vec<f32> = (0..8).map(|i| (pid * 8 + i) as f32).collect();
            assert_eq!(output, expected);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_reduce_scatter_neon_parity_with_scalar() {
        let n = 96;
        let np = 4;
        let input = pattern(n, 0.5);
        for pid in 0..np {
            let ps = n / np;
            let out_len = if pid == np - 1 { n - pid * ps } else { ps };
            let mut neon_out = vec![0.0; out_len];
            let mut scal_out = vec![0.0; out_len];
            unsafe { reduce_scatter_neon(&input, &mut neon_out, np, pid) };
            reduce_scatter_scalar(&input, &mut scal_out, np, pid);
            assert_slices_approx(&neon_out, &scal_out, 1e-6, "reduce_scatter parity");
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_reduce_scatter_neon_single_partition() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut output = vec![0.0; 5];
        unsafe { reduce_scatter_neon(&input, &mut output, 1, 0) };
        assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_reduce_scatter_neon_odd_size() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut output = vec![0.0; 4];
        // partition_size = 7/2 = 3; last partition gets [3..7] = 4 elements
        unsafe { reduce_scatter_neon(&input, &mut output, 2, 1) };
        assert_eq!(output, vec![4.0, 5.0, 6.0, 7.0]);
    }

    // ── combined / integration tests ───────────────────────────────

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_scatter_reduce_gather_pipeline() {
        // Simulate: scatter → per-partition matmul → gather
        let n = 16;
        let np = 2;
        let ps = n / np;
        let input = pattern(n, 1.0);
        let mut parts: Vec<Vec<f32>> = (0..np).map(|_| vec![0.0; ps]).collect();
        unsafe {
            let mut slices: Vec<&mut [f32]> = parts.iter_mut().map(|p| p.as_mut_slice()).collect();
            scatter_neon(&input, &mut slices, ps);
        }
        // "Process" each partition: double values
        for part in &mut parts {
            for v in part.iter_mut() {
                *v *= 2.0;
            }
        }
        let inputs: Vec<&[f32]> = parts.iter().map(|p| p.as_slice()).collect();
        let mut output = vec![0.0; n];
        unsafe { gather_neon(&inputs, &mut output, ps) };
        let expected: Vec<f32> = input.iter().map(|v| v * 2.0).collect();
        assert_slices_approx(&output, &expected, 1e-5, "scatter-process-gather");
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_all_reduce_then_reduce_scatter() {
        let n = 32;
        let np = 4;
        let local = vec![1.0; n];
        let mut global = vec![0.0; n];
        // Simulate 4 ranks contributing
        for _ in 0..np {
            unsafe { all_reduce_sum_neon(&local, &mut global, n) };
        }
        for &v in &global {
            assert!(approx_eq(v, np as f32, 1e-5));
        }
        // Then reduce-scatter to partition 2
        let mut out = vec![0.0; n / np];
        unsafe { reduce_scatter_neon(&global, &mut out, np, 2) };
        for &v in &out {
            assert!(approx_eq(v, np as f32, 1e-5));
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_partitioned_matmul_matches_full() {
        let m = 3;
        let n = 3;
        let k = 12;
        let a = pattern(m * k, 0.1);
        let b = pattern(k * n, 0.2);
        let expected = naive_matmul(&a, &b, m, n, k);
        // Two-partition K-split using NEON
        let mut c = vec![0.0; m * n];
        unsafe {
            partition_matmul_neon(&a, &b, &mut c, m, n, k, 0, 6);
            partition_matmul_neon(&a, &b, &mut c, m, n, k, 6, 12);
        }
        assert_slices_approx(&c, &expected, 0.1, "two-partition matmul");
    }

    #[test]
    fn test_scalar_fallback_compiles_on_all_platforms() {
        // Ensure all scalar functions compile and run on any target
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let mut g = vec![0.0; 4];
        all_reduce_sum_scalar(&a, &mut g, 4);
        assert_eq!(g, a);

        let mut p0 = vec![0.0; 2];
        let mut p1 = vec![0.0; 2];
        {
            let mut outs: Vec<&mut [f32]> = vec![&mut p0, &mut p1];
            scatter_scalar(&a, &mut outs, 2);
        }
        assert_eq!(p0, vec![1.0, 2.0]);
        assert_eq!(p1, vec![3.0, 4.0]);

        let inputs: Vec<&[f32]> = vec![&p0, &p1];
        let mut out = vec![0.0; 4];
        gather_scalar(&inputs, &mut out, 2);
        assert_eq!(out, a);

        let mut rs = vec![0.0; 2];
        reduce_scatter_scalar(&a, &mut rs, 2, 1);
        assert_eq!(rs, vec![3.0, 4.0]);
    }

    #[test]
    fn test_partition_matmul_scalar_1x1() {
        let a = vec![3.0];
        let b = vec![4.0];
        let mut c = vec![0.0];
        partition_matmul_scalar(&a, &b, &mut c, 1, 1, 1, 0, 1);
        assert!(approx_eq(c[0], 12.0, 1e-6));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_partition_matmul_neon_1x1() {
        let a = vec![3.0];
        let b = vec![4.0];
        let mut c = vec![0.0];
        unsafe { partition_matmul_neon(&a, &b, &mut c, 1, 1, 1, 0, 1) };
        assert!(approx_eq(c[0], 12.0, 1e-6));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_scatter_neon_8_partitions() {
        let n = 64;
        let np = 8;
        let ps = n / np;
        let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut parts: Vec<Vec<f32>> = (0..np).map(|_| vec![0.0; ps]).collect();
        unsafe {
            let mut slices: Vec<&mut [f32]> = parts.iter_mut().map(|p| p.as_mut_slice()).collect();
            scatter_neon(&input, &mut slices, ps);
        }
        for p in 0..np {
            let expected: Vec<f32> = (0..ps).map(|i| (p * ps + i) as f32).collect();
            assert_eq!(parts[p], expected);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_gather_neon_8_partitions() {
        let n = 64;
        let np = 8;
        let ps = n / np;
        let parts: Vec<Vec<f32>> =
            (0..np).map(|p| (0..ps).map(|i| (p * ps + i) as f32).collect()).collect();
        let inputs: Vec<&[f32]> = parts.iter().map(|p| p.as_slice()).collect();
        let mut output = vec![0.0; n];
        unsafe { gather_neon(&inputs, &mut output, ps) };
        let expected: Vec<f32> = (0..n).map(|i| i as f32).collect();
        assert_eq!(output, expected);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_reduce_scatter_neon_8_partitions() {
        let n = 64;
        let np = 8;
        let ps = n / np;
        let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
        for pid in 0..np {
            let mut output = vec![0.0; ps];
            unsafe { reduce_scatter_neon(&input, &mut output, np, pid) };
            let expected: Vec<f32> = (0..ps).map(|i| (pid * ps + i) as f32).collect();
            assert_eq!(output, expected, "partition {pid}");
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_all_reduce_sum_neon_exact_4() {
        let local = vec![0.5, 1.0, 1.5, 2.0];
        let mut global = vec![0.5, 1.0, 1.5, 2.0];
        unsafe { all_reduce_sum_neon(&local, &mut global, 4) };
        assert_eq!(global, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_scatter_scalar_8_partitions() {
        let n = 80;
        let np = 8;
        let ps = n / np;
        let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut parts: Vec<Vec<f32>> = (0..np).map(|_| vec![0.0; ps]).collect();
        {
            let mut slices: Vec<&mut [f32]> = parts.iter_mut().map(|p| p.as_mut_slice()).collect();
            scatter_scalar(&input, &mut slices, ps);
        }
        for p in 0..np {
            let expected: Vec<f32> = (0..ps).map(|i| (p * ps + i) as f32).collect();
            assert_eq!(parts[p], expected);
        }
    }

    #[test]
    fn test_gather_scalar_8_partitions() {
        let np = 8;
        let ps = 10;
        let parts: Vec<Vec<f32>> =
            (0..np).map(|p| (0..ps).map(|i| (p * ps + i) as f32).collect()).collect();
        let inputs: Vec<&[f32]> = parts.iter().map(|p| p.as_slice()).collect();
        let mut output = vec![0.0; np * ps];
        gather_scalar(&inputs, &mut output, ps);
        let expected: Vec<f32> = (0..np * ps).map(|i| i as f32).collect();
        assert_eq!(output, expected);
    }

    #[test]
    fn test_reduce_scatter_scalar_8_partitions() {
        let n = 80;
        let np = 8;
        let ps = n / np;
        let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
        for pid in 0..np {
            let mut output = vec![0.0; ps];
            reduce_scatter_scalar(&input, &mut output, np, pid);
            let expected: Vec<f32> = (0..ps).map(|i| (pid * ps + i) as f32).collect();
            assert_eq!(output, expected, "scalar partition {pid}");
        }
    }

    #[test]
    fn test_partition_matmul_scalar_2_partitions() {
        let m = 2;
        let n = 2;
        let k = 8;
        let a = vec![1.0; m * k];
        let b = vec![1.0; k * n];
        let expected = naive_matmul(&a, &b, m, n, k);
        let mut c = vec![0.0; m * n];
        partition_matmul_scalar(&a, &b, &mut c, m, n, k, 0, 4);
        partition_matmul_scalar(&a, &b, &mut c, m, n, k, 4, 8);
        assert_slices_approx(&c, &expected, 1e-4, "scalar 2-partition matmul");
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_partition_matmul_neon_2_partitions() {
        let m = 2;
        let n = 2;
        let k = 8;
        let a = vec![1.0; m * k];
        let b = vec![1.0; k * n];
        let expected = naive_matmul(&a, &b, m, n, k);
        let mut c = vec![0.0; m * n];
        unsafe {
            partition_matmul_neon(&a, &b, &mut c, m, n, k, 0, 4);
            partition_matmul_neon(&a, &b, &mut c, m, n, k, 4, 8);
        }
        assert_slices_approx(&c, &expected, 1e-3, "neon 2-partition matmul");
    }
}

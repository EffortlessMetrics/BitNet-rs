//! ARM NEON tensor parallelism kernels for Apple Silicon.
//!
//! Provides NEON-optimized primitives for distributing tensor
//! computations across multiple workers:
//!
//! - **AllReduce sum/max** — lane-parallel reductions across partitions
//! - **Scatter / Gather** — split and merge tensors along an axis
//! - **Column-parallel linear** — partition weight columns across workers
//! - **Row-parallel linear** — partition weight rows across workers
//! - **Pipeline stage boundary** — activation transfer between stages

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── AllReduce Sum ──────────────────────────────────────────────────

/// NEON-accelerated element-wise sum reduction across tensor partitions.
///
/// Each partition in `partitions` must have the same length as `output`.
/// The result at each position is the sum of corresponding elements
/// across all partitions.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
///
/// * Any partition length != `output.len()`
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_allreduce_sum(partitions: &[&[f32]], output: &mut [f32]) {
    let n = output.len();
    for (i, part) in partitions.iter().enumerate() {
        assert_eq!(
            part.len(),
            n,
            "neon_allreduce_sum: partition {i} length {} != output length {n}",
            part.len(),
        );
    }

    // Zero the output.
    let out_ptr = output.as_mut_ptr();
    let chunks = n / 4;
    let remainder = n % 4;

    for c in 0..chunks {
        let offset = c * 4;
        let mut acc = vdupq_n_f32(0.0);
        for part in partitions {
            let v = unsafe { vld1q_f32(part.as_ptr().add(offset)) };
            acc = vaddq_f32(acc, v);
        }
        unsafe { vst1q_f32(out_ptr.add(offset), acc) };
    }

    // Scalar tail.
    let tail_start = chunks * 4;
    for i in 0..remainder {
        let idx = tail_start + i;
        let mut sum = 0.0f32;
        for part in partitions {
            sum += part[idx];
        }
        unsafe { *out_ptr.add(idx) = sum };
    }
}

// ── AllReduce Max ──────────────────────────────────────────────────

/// NEON-accelerated element-wise max reduction across tensor partitions.
///
/// Each partition must have the same length as `output`. The result at
/// each position is the maximum of corresponding elements across all
/// partitions. Useful for distributed argmax.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
///
/// * Any partition length != `output.len()`
/// * `partitions` is empty
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_allreduce_max(partitions: &[&[f32]], output: &mut [f32]) {
    assert!(!partitions.is_empty(), "neon_allreduce_max: partitions must not be empty",);
    let n = output.len();
    for (i, part) in partitions.iter().enumerate() {
        assert_eq!(
            part.len(),
            n,
            "neon_allreduce_max: partition {i} length {} != output length {n}",
            part.len(),
        );
    }

    let out_ptr = output.as_mut_ptr();
    let chunks = n / 4;
    let remainder = n % 4;

    for c in 0..chunks {
        let offset = c * 4;
        let mut acc = unsafe { vld1q_f32(partitions[0].as_ptr().add(offset)) };
        for part in &partitions[1..] {
            let v = unsafe { vld1q_f32(part.as_ptr().add(offset)) };
            acc = vmaxq_f32(acc, v);
        }
        unsafe { vst1q_f32(out_ptr.add(offset), acc) };
    }

    let tail_start = chunks * 4;
    for i in 0..remainder {
        let idx = tail_start + i;
        let mut mx = partitions[0][idx];
        for part in &partitions[1..] {
            if part[idx] > mx {
                mx = part[idx];
            }
        }
        unsafe { *out_ptr.add(idx) = mx };
    }
}

// ── Tensor Scatter ─────────────────────────────────────────────────

/// Split `input` into `n_partitions` equal chunks along the leading
/// (flattened) axis and copy each chunk into the corresponding output
/// using NEON stores.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
///
/// * `input.len()` is not divisible by `n_partitions`
/// * `outputs.len() != n_partitions`
/// * Any output slice length != `input.len() / n_partitions`
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_tensor_scatter(input: &[f32], outputs: &mut [&mut [f32]], n_partitions: usize) {
    assert!(n_partitions > 0, "neon_tensor_scatter: n_partitions must be > 0",);
    assert_eq!(
        outputs.len(),
        n_partitions,
        "neon_tensor_scatter: outputs.len() ({}) != n_partitions ({n_partitions})",
        outputs.len(),
    );
    assert_eq!(
        input.len() % n_partitions,
        0,
        "neon_tensor_scatter: input length {} not divisible by {n_partitions}",
        input.len(),
    );

    let chunk_size = input.len() / n_partitions;
    for (p, out) in outputs.iter_mut().enumerate() {
        assert_eq!(
            out.len(),
            chunk_size,
            "neon_tensor_scatter: output {p} length {} != chunk_size {chunk_size}",
            out.len(),
        );

        let src = unsafe { input.as_ptr().add(p * chunk_size) };
        let dst = out.as_mut_ptr();
        let chunks = chunk_size / 4;
        let remainder = chunk_size % 4;

        for c in 0..chunks {
            let v = unsafe { vld1q_f32(src.add(c * 4)) };
            unsafe { vst1q_f32(dst.add(c * 4), v) };
        }
        let tail = chunks * 4;
        for i in 0..remainder {
            unsafe { *dst.add(tail + i) = *src.add(tail + i) };
        }
    }
}

// ── Tensor Gather ──────────────────────────────────────────────────

/// Merge `partitions` into `output` by concatenation, copying each
/// partition into the corresponding region of the output buffer.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
///
/// * Total length of all partitions != `output.len()`
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_tensor_gather(partitions: &[&[f32]], output: &mut [f32]) {
    let total: usize = partitions.iter().map(|p| p.len()).sum();
    assert_eq!(
        total,
        output.len(),
        "neon_tensor_gather: total partition length {total} != output length {}",
        output.len(),
    );

    let out_ptr = output.as_mut_ptr();
    let mut offset = 0usize;
    for part in partitions {
        let len = part.len();
        let src = part.as_ptr();
        let dst = unsafe { out_ptr.add(offset) };
        let chunks = len / 4;
        let remainder = len % 4;

        for c in 0..chunks {
            let v = unsafe { vld1q_f32(src.add(c * 4)) };
            unsafe { vst1q_f32(dst.add(c * 4), v) };
        }
        let tail = chunks * 4;
        for i in 0..remainder {
            unsafe { *dst.add(tail + i) = *src.add(tail + i) };
        }
        offset += len;
    }
}

// ── Column-Parallel Linear ─────────────────────────────────────────

/// Column-parallel linear: partition weight columns across workers.
///
/// Computes `output = input × W_partᵀ` where `W_part` is the column
/// slice `[partition_start..partition_end, :]` of the full weight
/// matrix stored in row-major order `[out_features, in_features]`.
///
/// `input` is `[batch_size, in_features]`.
/// `output` is `[batch_size, partition_cols]`.
///
/// NEON accelerates the inner dot-product accumulation.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
///
/// * Dimension mismatches between input / weight / output shapes.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_column_parallel_linear(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    batch_size: usize,
    in_features: usize,
    partition_start: usize,
    partition_end: usize,
) {
    let partition_cols = partition_end - partition_start;
    assert_eq!(
        input.len(),
        batch_size * in_features,
        "neon_column_parallel_linear: input size mismatch",
    );
    assert_eq!(
        output.len(),
        batch_size * partition_cols,
        "neon_column_parallel_linear: output size mismatch",
    );
    if let Some(b) = bias {
        assert!(b.len() >= partition_end, "neon_column_parallel_linear: bias too short",);
    }

    let in_ptr = input.as_ptr();
    let w_ptr = weight.as_ptr();
    let o_ptr = output.as_mut_ptr();
    let chunks = in_features / 4;
    let remainder = in_features % 4;

    for b in 0..batch_size {
        let x_row = unsafe { in_ptr.add(b * in_features) };
        for col in 0..partition_cols {
            let w_row_idx = partition_start + col;
            let w_row = unsafe { w_ptr.add(w_row_idx * in_features) };

            let mut acc = vdupq_n_f32(0.0);
            for c in 0..chunks {
                let xv = unsafe { vld1q_f32(x_row.add(c * 4)) };
                let wv = unsafe { vld1q_f32(w_row.add(c * 4)) };
                acc = vfmaq_f32(acc, xv, wv);
            }
            let mut dot = vaddvq_f32(acc);

            let tail = chunks * 4;
            for i in 0..remainder {
                dot += unsafe { *x_row.add(tail + i) * *w_row.add(tail + i) };
            }

            if let Some(b_slice) = bias {
                dot += b_slice[partition_start + col];
            }

            unsafe { *o_ptr.add(b * partition_cols + col) = dot };
        }
    }
}

// ── Row-Parallel Linear ────────────────────────────────────────────

/// Row-parallel linear: partition weight rows across workers.
///
/// Computes `output += input_part × W_partᵀ` where `input_part` is
/// the worker's slice of the input along the feature dimension and
/// `W_part` is the corresponding weight row slice.
///
/// `input_part` shape: `[batch_size, partition_features]`
/// `weight_part` shape: `[out_features, partition_features]`
/// `output` shape: `[batch_size, out_features]` — **accumulated** into.
///
/// After all workers execute, a final allreduce-sum across outputs
/// produces the correct result.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
///
/// * Dimension mismatches.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_row_parallel_linear(
    input_part: &[f32],
    weight_part: &[f32],
    output: &mut [f32],
    batch_size: usize,
    partition_features: usize,
    out_features: usize,
) {
    assert_eq!(
        input_part.len(),
        batch_size * partition_features,
        "neon_row_parallel_linear: input_part size mismatch",
    );
    assert_eq!(
        weight_part.len(),
        out_features * partition_features,
        "neon_row_parallel_linear: weight_part size mismatch",
    );
    assert_eq!(
        output.len(),
        batch_size * out_features,
        "neon_row_parallel_linear: output size mismatch",
    );

    let x_ptr = input_part.as_ptr();
    let w_ptr = weight_part.as_ptr();
    let o_ptr = output.as_mut_ptr();
    let chunks = partition_features / 4;
    let remainder = partition_features % 4;

    for b in 0..batch_size {
        let x_row = unsafe { x_ptr.add(b * partition_features) };
        for col in 0..out_features {
            let w_row = unsafe { w_ptr.add(col * partition_features) };

            let mut acc = vdupq_n_f32(0.0);
            for c in 0..chunks {
                let xv = unsafe { vld1q_f32(x_row.add(c * 4)) };
                let wv = unsafe { vld1q_f32(w_row.add(c * 4)) };
                acc = vfmaq_f32(acc, xv, wv);
            }
            let mut dot = vaddvq_f32(acc);

            let tail = chunks * 4;
            for i in 0..remainder {
                dot += unsafe { *x_row.add(tail + i) * *w_row.add(tail + i) };
            }

            // Accumulate (not overwrite) — caller allreduces later.
            let out_idx = b * out_features + col;
            unsafe { *o_ptr.add(out_idx) += dot };
        }
    }
}

// ── Pipeline Stage Boundary ────────────────────────────────────────

/// Activation buffer for pipeline-parallel stage boundaries.
///
/// Holds a contiguous activation tensor and metadata so the next
/// pipeline stage can pick up where the previous one left off.
#[cfg(target_arch = "aarch64")]
#[derive(Debug, Clone)]
pub struct PipelineStageBuffer {
    /// Activation data transferred between stages.
    pub activations: Vec<f32>,
    /// Source stage index (0-based).
    pub src_stage: usize,
    /// Destination stage index.
    pub dst_stage: usize,
    /// Batch size carried by this activation.
    pub batch_size: usize,
    /// Feature dimension of the activation tensor.
    pub feature_dim: usize,
}

#[cfg(target_arch = "aarch64")]
impl PipelineStageBuffer {
    /// Create a new pipeline stage buffer by copying activations from
    /// the source slice using NEON loads/stores.
    ///
    /// # Safety
    ///
    /// Caller must ensure `neon` target feature is available.
    #[target_feature(enable = "neon")]
    pub unsafe fn transfer(
        activations: &[f32],
        src_stage: usize,
        dst_stage: usize,
        batch_size: usize,
        feature_dim: usize,
    ) -> Self {
        assert_eq!(
            activations.len(),
            batch_size * feature_dim,
            "PipelineStageBuffer::transfer: activation size mismatch \
             (expected {}, got {})",
            batch_size * feature_dim,
            activations.len(),
        );

        let mut buf = vec![0.0f32; activations.len()];
        let src = activations.as_ptr();
        let dst = buf.as_mut_ptr();
        let len = activations.len();
        let chunks = len / 4;
        let remainder = len % 4;

        for c in 0..chunks {
            let v = unsafe { vld1q_f32(src.add(c * 4)) };
            unsafe { vst1q_f32(dst.add(c * 4), v) };
        }
        let tail = chunks * 4;
        for i in 0..remainder {
            unsafe { *dst.add(tail + i) = *src.add(tail + i) };
        }

        Self { activations: buf, src_stage, dst_stage, batch_size, feature_dim }
    }

    /// Receive: copy activations into the provided output buffer.
    ///
    /// # Safety
    ///
    /// Caller must ensure `neon` target feature is available.
    #[target_feature(enable = "neon")]
    pub unsafe fn receive_into(&self, output: &mut [f32]) {
        assert_eq!(
            output.len(),
            self.activations.len(),
            "PipelineStageBuffer::receive_into: output size mismatch",
        );

        let src = self.activations.as_ptr();
        let dst = output.as_mut_ptr();
        let len = self.activations.len();
        let chunks = len / 4;
        let remainder = len % 4;

        for c in 0..chunks {
            let v = unsafe { vld1q_f32(src.add(c * 4)) };
            unsafe { vst1q_f32(dst.add(c * 4), v) };
        }
        let tail = chunks * 4;
        for i in 0..remainder {
            unsafe { *dst.add(tail + i) = *src.add(tail + i) };
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    // ── AllReduce Sum tests ────────────────────────────────────────

    #[test]
    fn test_allreduce_sum_two_partitions() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0f32, 20.0, 30.0, 40.0, 50.0];
        let mut out = [0.0f32; 5];
        unsafe { neon_allreduce_sum(&[&a, &b], &mut out) };
        assert_eq!(out.to_vec(), vec![11.0, 22.0, 33.0, 44.0, 55.0]);
    }

    #[test]
    fn test_allreduce_sum_three_partitions() {
        let a = [1.0f32; 8];
        let b = [2.0f32; 8];
        let c = [3.0f32; 8];
        let mut out = [0.0f32; 8];
        unsafe { neon_allreduce_sum(&[&a, &b, &c], &mut out) };
        assert_eq!(out.to_vec(), vec![6.0; 8]);
    }

    #[test]
    fn test_allreduce_sum_single_partition() {
        let a = vec![7.0f32, 8.0, 9.0];
        let mut out = [0.0f32; 3];
        unsafe { neon_allreduce_sum(&[&a], &mut out) };
        assert_eq!(out.to_vec(), vec![7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_allreduce_sum_empty() {
        let mut out: Vec<f32> = vec![];
        unsafe { neon_allreduce_sum(&[], &mut out) };
        assert!(out.is_empty());
    }

    // ── AllReduce Max tests ────────────────────────────────────────

    #[test]
    fn test_allreduce_max_basic() {
        let a = vec![1.0f32, 5.0, 3.0, 7.0, 2.0];
        let b = vec![4.0f32, 2.0, 6.0, 1.0, 8.0];
        let mut out = [0.0f32; 5];
        unsafe { neon_allreduce_max(&[&a, &b], &mut out) };
        assert_eq!(out.to_vec(), vec![4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_allreduce_max_negative_values() {
        let a = vec![-1.0f32, -5.0, -3.0, -7.0];
        let b = vec![-4.0f32, -2.0, -6.0, -1.0];
        let mut out = [0.0f32; 4];
        unsafe { neon_allreduce_max(&[&a, &b], &mut out) };
        assert_eq!(out.to_vec(), vec![-1.0, -2.0, -3.0, -1.0]);
    }

    #[test]
    fn test_allreduce_max_three_partitions() {
        let a = vec![1.0f32, 9.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![8.0f32, 2.0, 7.0, 4.0, 5.0, 6.0, 1.0, 3.0];
        let c = vec![5.0f32, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0];
        let mut out = [0.0f32; 8];
        unsafe { neon_allreduce_max(&[&a, &b, &c], &mut out) };
        assert_eq!(out.to_vec(), vec![8.0, 9.0, 7.0, 5.0, 5.0, 6.0, 7.0, 8.0]);
    }

    // ── Tensor Scatter tests ───────────────────────────────────────

    #[test]
    fn test_scatter_even_split() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut p0 = [0.0f32; 3];
        let mut p1 = [0.0f32; 3];
        let mut outputs: Vec<&mut [f32]> = vec![&mut p0, &mut p1];
        unsafe { neon_tensor_scatter(&input, &mut outputs, 2) };
        assert_eq!(p0.to_vec(), vec![1.0, 2.0, 3.0]);
        assert_eq!(p1.to_vec(), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_scatter_four_partitions() {
        let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let mut p0 = [0.0f32; 4];
        let mut p1 = [0.0f32; 4];
        let mut p2 = [0.0f32; 4];
        let mut p3 = [0.0f32; 4];
        let mut outputs: Vec<&mut [f32]> = vec![&mut p0, &mut p1, &mut p2, &mut p3];
        unsafe { neon_tensor_scatter(&input, &mut outputs, 4) };
        assert_eq!(p0.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(p1.to_vec(), vec![5.0, 6.0, 7.0, 8.0]);
        assert_eq!(p2.to_vec(), vec![9.0, 10.0, 11.0, 12.0]);
        assert_eq!(p3.to_vec(), vec![13.0, 14.0, 15.0, 16.0]);
    }

    // ── Tensor Gather tests ────────────────────────────────────────

    #[test]
    fn test_gather_basic() {
        let p0 = vec![1.0f32, 2.0, 3.0];
        let p1 = vec![4.0f32, 5.0, 6.0];
        let mut out = [0.0f32; 6];
        unsafe { neon_tensor_gather(&[&p0, &p1], &mut out) };
        assert_eq!(out.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_scatter_gather_roundtrip() {
        let input: Vec<f32> = (0..20).map(|x| x as f32).collect();
        let mut p0 = [0.0f32; 5];
        let mut p1 = [0.0f32; 5];
        let mut p2 = [0.0f32; 5];
        let mut p3 = [0.0f32; 5];
        let mut outputs: Vec<&mut [f32]> = vec![&mut p0, &mut p1, &mut p2, &mut p3];
        unsafe { neon_tensor_scatter(&input, &mut outputs, 4) };

        let mut restored = [0.0f32; 20];
        unsafe { neon_tensor_gather(&[&p0, &p1, &p2, &p3], &mut restored) };
        assert_eq!(input, restored);
    }

    // ── Column-Parallel Linear tests ───────────────────────────────

    #[test]
    fn test_column_parallel_identity() {
        // Identity-like: weight = I_{4×4}, partition [0..2)
        // input [1,4], output [1,2]
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        #[rustfmt::skip]
        let weight = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let mut out = [0.0f32; 2];
        unsafe {
            neon_column_parallel_linear(&input, &weight, None, &mut out, 1, 4, 0, 2);
        }
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);
    }

    #[test]
    fn test_column_parallel_with_bias() {
        let input = vec![1.0f32, 1.0, 1.0, 1.0];
        let weight = vec![
            1.0, 1.0, 1.0, 1.0, // row 0 → dot = 4
            2.0, 2.0, 2.0, 2.0, // row 1 → dot = 8
        ];
        let bias = vec![0.5, 1.0];
        let mut out = [0.0f32; 2];
        unsafe {
            neon_column_parallel_linear(&input, &weight, Some(&bias), &mut out, 1, 4, 0, 2);
        }
        assert_eq!(out.to_vec(), vec![4.5, 9.0]);
    }

    // ── Row-Parallel Linear tests ──────────────────────────────────

    #[test]
    fn test_row_parallel_accumulate() {
        // weight_part [2, 2], input_part [1, 2], output [1, 2]
        let input_part = vec![1.0f32, 2.0];
        let weight_part = vec![
            1.0, 0.0, // row 0 → dot = 1
            0.0, 1.0, // row 1 → dot = 2
        ];
        let mut out = vec![10.0f32, 20.0]; // pre-existing values
        unsafe {
            neon_row_parallel_linear(&input_part, &weight_part, &mut out, 1, 2, 2);
        }
        // 10 + 1 = 11, 20 + 2 = 22
        assert_eq!(out, vec![11.0, 22.0]);
    }

    #[test]
    fn test_row_parallel_two_workers() {
        // Full weight [2, 4], split into two row-parallel workers
        // Worker 0: features [0..2), Worker 1: features [2..4)
        let input = vec![1.0f32, 2.0, 3.0, 4.0]; // [1, 4]
        let w0 = vec![
            1.0, 0.0, // out_feat 0, part_feat [0..2)
            0.0, 1.0, // out_feat 1, part_feat [0..2)
        ];
        let w1 = vec![
            1.0, 0.0, // out_feat 0, part_feat [2..4)
            0.0, 1.0, // out_feat 1, part_feat [2..4)
        ];

        let mut out = [0.0f32; 2];
        unsafe {
            neon_row_parallel_linear(&input[0..2], &w0, &mut out, 1, 2, 2);
            neon_row_parallel_linear(&input[2..4], &w1, &mut out, 1, 2, 2);
        }
        // out[0] = 1*1 + 2*0 + 3*1 + 4*0 = 4
        // out[1] = 1*0 + 2*1 + 3*0 + 4*1 = 6
        assert_eq!(out.to_vec(), vec![4.0, 6.0]);
    }

    // ── Pipeline Stage Boundary tests ──────────────────────────────

    #[test]
    fn test_pipeline_transfer_roundtrip() {
        let activations = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let buf = unsafe { PipelineStageBuffer::transfer(&activations, 0, 1, 2, 3) };
        assert_eq!(buf.src_stage, 0);
        assert_eq!(buf.dst_stage, 1);
        assert_eq!(buf.batch_size, 2);
        assert_eq!(buf.feature_dim, 3);
        assert_eq!(buf.activations, activations);

        let mut received = [0.0f32; 6];
        unsafe { buf.receive_into(&mut received) };
        assert_eq!(received.to_vec(), activations);
    }

    #[test]
    fn test_pipeline_large_transfer() {
        let n = 1024;
        let activations: Vec<f32> = (0..n).map(|x| x as f32 * 0.01).collect();
        let buf = unsafe { PipelineStageBuffer::transfer(&activations, 3, 4, 32, 32) };
        let mut received = vec![0.0f32; n];
        unsafe { buf.receive_into(&mut received) };
        assert_eq!(received, activations);
    }

    // ── Edge-case / stress tests ───────────────────────────────────

    #[test]
    fn test_allreduce_sum_large_unaligned() {
        // 17 elements: 4 NEON chunks + 1 scalar tail
        let a: Vec<f32> = (0..17).map(|x| x as f32).collect();
        let b: Vec<f32> = (0..17).map(|x| (x * 2) as f32).collect();
        let mut out = [0.0f32; 17];
        unsafe { neon_allreduce_sum(&[&a, &b], &mut out) };
        for i in 0..17 {
            assert_eq!(out[i], (i + i * 2) as f32, "mismatch at {i}");
        }
    }

    #[test]
    #[should_panic(expected = "partition 1 length")]
    fn test_allreduce_sum_mismatched_lengths() {
        let a = [1.0f32; 4];
        let b = [1.0f32; 5]; // wrong length
        let mut out = [0.0f32; 4];
        unsafe { neon_allreduce_sum(&[&a, &b], &mut out) };
    }

    #[test]
    #[should_panic(expected = "partitions must not be empty")]
    fn test_allreduce_max_empty_partitions() {
        let mut out = [0.0f32; 4];
        unsafe { neon_allreduce_max(&[], &mut out) };
    }
}

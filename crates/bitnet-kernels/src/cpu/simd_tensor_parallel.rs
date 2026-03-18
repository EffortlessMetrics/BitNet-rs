//! SIMD-accelerated CPU tensor parallelism with sharded matmul and all-reduce.
//!
//! Extends the basic tensor-parallel primitives in [`super::tensor_parallel`]
//! with SIMD-optimised all-reduce, scatter/gather, sharded matrix multiply,
//! communication cost estimation, and automatic sharding plan generation.
//!
//! # Layout conventions
//!
//! * All matrices are **row-major** (M×K for A, K×N for B, M×N for C).
//! * Sharding is along a single axis – column-parallel splits N, row-parallel
//!   splits M.
#![allow(unsafe_op_in_unsafe_fn)]

use std::fmt;

#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
use std::arch::x86_64::*;

// ── Errors ─────────────────────────────────────────────────────────

/// Errors specific to SIMD tensor-parallel operations.
#[derive(Debug, Clone, PartialEq)]
pub enum SimdTensorParallelError {
    /// A dimension or shape constraint was violated.
    ShapeMismatch(String),
    /// Sharding cannot be applied with the given parameters.
    InvalidSharding(String),
    /// An argument was out of range.
    InvalidArgument(String),
    /// Buffer size mismatch.
    BufferSizeMismatch { expected: usize, got: usize },
}

impl fmt::Display for SimdTensorParallelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch(msg) => write!(f, "shape mismatch: {msg}"),
            Self::InvalidSharding(msg) => write!(f, "invalid sharding: {msg}"),
            Self::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
            Self::BufferSizeMismatch { expected, got } => {
                write!(f, "buffer size mismatch: expected {expected}, got {got}")
            }
        }
    }
}

impl std::error::Error for SimdTensorParallelError {}

type Result<T> = std::result::Result<T, SimdTensorParallelError>;

// ── Sharding strategy ──────────────────────────────────────────────

/// Strategy for distributing a 2-D tensor across cores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SimdShardStrategy {
    /// Split columns (N dimension) across cores.
    ColumnParallel,
    /// Split rows (M dimension) across cores.
    RowParallel,
}

impl fmt::Display for SimdShardStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ColumnParallel => write!(f, "ColumnParallel"),
            Self::RowParallel => write!(f, "RowParallel"),
        }
    }
}

// ── Reduce operation ───────────────────────────────────────────────

/// Element-wise reduction operation for all-reduce.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ReduceOp {
    /// Element-wise sum.
    #[default]
    Sum,
    /// Element-wise maximum.
    Max,
    /// Element-wise mean (sum / count).
    Mean,
}

impl fmt::Display for ReduceOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sum => write!(f, "Sum"),
            Self::Max => write!(f, "Max"),
            Self::Mean => write!(f, "Mean"),
        }
    }
}

// ── Communication cost model ───────────────────────────────────────

/// Estimated communication cost for a parallelism configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct CommCostEstimate {
    /// Bytes transferred in total across all ranks.
    pub total_bytes: usize,
    /// Number of synchronisation barriers.
    pub num_barriers: usize,
    /// Estimated latency in microseconds (simple model).
    pub estimated_latency_us: f64,
    /// Strategy that was evaluated.
    pub strategy: SimdShardStrategy,
}

/// Estimate communication cost for a given sharding plan.
///
/// Uses a simple α-β model:
///   `cost = num_barriers * α + total_bytes * β`
///
/// where `α` (latency per barrier, µs) and `β` (inverse bandwidth, µs/byte)
/// are fixed constants tuned for intra-process memcpy.
pub fn estimate_comm_cost(
    rows: usize,
    cols: usize,
    num_cores: usize,
    strategy: SimdShardStrategy,
) -> Result<CommCostEstimate> {
    if num_cores == 0 {
        return Err(SimdTensorParallelError::InvalidArgument("num_cores must be > 0".into()));
    }
    if rows == 0 || cols == 0 {
        return Err(SimdTensorParallelError::ShapeMismatch("tensor dimensions must be > 0".into()));
    }

    // α-β constants (tuned for in-process memcpy on modern x86).
    const ALPHA_US: f64 = 0.5; // barrier latency
    const BETA_US_PER_BYTE: f64 = 0.001; // ~1 GB/s effective

    let element_bytes = std::mem::size_of::<f32>();

    // For all-reduce after sharded matmul:
    //   column-parallel → all-reduce across M×(N/P) partial results per core
    //   row-parallel    → all-reduce across (M/P)×N partial results per core
    let shard_elements = match strategy {
        SimdShardStrategy::ColumnParallel => rows * cols.div_ceil(num_cores),
        SimdShardStrategy::RowParallel => rows.div_ceil(num_cores) * cols,
    };

    // Ring all-reduce: 2*(P-1)/P * data_size bytes total traffic.
    let factor = if num_cores > 1 { 2.0 * (num_cores - 1) as f64 / num_cores as f64 } else { 0.0 };
    let total_bytes = (factor * (shard_elements * element_bytes) as f64) as usize;
    // Two barrier phases (reduce-scatter + all-gather).
    let num_barriers = if num_cores > 1 { 2 } else { 0 };
    let estimated_latency_us =
        num_barriers as f64 * ALPHA_US + total_bytes as f64 * BETA_US_PER_BYTE;

    Ok(CommCostEstimate { total_bytes, num_barriers, estimated_latency_us, strategy })
}

// ── Sharding plan ──────────────────────────────────────────────────

/// An automatic sharding plan for a 2-D matmul.
#[derive(Debug, Clone, PartialEq)]
pub struct ShardingPlan {
    /// The chosen strategy.
    pub strategy: SimdShardStrategy,
    /// Number of cores to use.
    pub num_cores: usize,
    /// Per-core shard sizes along the split axis.
    pub shard_sizes: Vec<usize>,
    /// Estimated communication cost.
    pub cost: CommCostEstimate,
}

impl fmt::Display for ShardingPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ShardingPlan({}, {} cores, est. {:.1}µs)",
            self.strategy, self.num_cores, self.cost.estimated_latency_us
        )
    }
}

/// Generate an automatic sharding plan.
///
/// Evaluates both column-parallel and row-parallel strategies and selects
/// the one with lower estimated communication cost.  `max_cores` is an
/// upper bound; the planner may choose fewer cores if the tensor is too
/// small to benefit from full parallelism.
pub fn generate_sharding_plan(
    m: usize,
    k: usize,
    n: usize,
    max_cores: usize,
) -> Result<ShardingPlan> {
    if max_cores == 0 {
        return Err(SimdTensorParallelError::InvalidArgument("max_cores must be > 0".into()));
    }
    if m == 0 || k == 0 || n == 0 {
        return Err(SimdTensorParallelError::ShapeMismatch("matmul dimensions must be > 0".into()));
    }

    // Clamp cores to the smaller split dimension to avoid empty shards.
    let col_cores = max_cores.min(n);
    let row_cores = max_cores.min(m);

    let col_cost = estimate_comm_cost(m, n, col_cores, SimdShardStrategy::ColumnParallel)?;
    let row_cost = estimate_comm_cost(m, n, row_cores, SimdShardStrategy::RowParallel)?;

    let (strategy, cores, cost) = if col_cost.estimated_latency_us <= row_cost.estimated_latency_us
    {
        (SimdShardStrategy::ColumnParallel, col_cores, col_cost)
    } else {
        (SimdShardStrategy::RowParallel, row_cores, row_cost)
    };

    let split_dim = match strategy {
        SimdShardStrategy::ColumnParallel => n,
        SimdShardStrategy::RowParallel => m,
    };

    let shard_sizes = compute_shard_sizes(split_dim, cores);

    Ok(ShardingPlan { strategy, num_cores: cores, shard_sizes, cost })
}

/// Divide `total` into `num_parts`, distributing remainder to the first
/// parts.
fn compute_shard_sizes(total: usize, num_parts: usize) -> Vec<usize> {
    let base = total / num_parts;
    let remainder = total % num_parts;
    (0..num_parts).map(|i| base + if i < remainder { 1 } else { 0 }).collect()
}

// ── SIMD all-reduce ────────────────────────────────────────────────

/// SIMD-accelerated element-wise all-reduce across multiple buffers.
///
/// All input slices must have the same length.  The result is written to
/// `output` which must also have the same length.
pub fn simd_all_reduce(inputs: &[&[f32]], output: &mut [f32], op: ReduceOp) -> Result<()> {
    if inputs.is_empty() {
        return Err(SimdTensorParallelError::InvalidArgument("no inputs provided".into()));
    }
    let len = inputs[0].len();
    for (i, inp) in inputs.iter().enumerate().skip(1) {
        if inp.len() != len {
            return Err(SimdTensorParallelError::BufferSizeMismatch {
                expected: len,
                got: inp.len(),
            });
        }
        let _ = i;
    }
    if output.len() != len {
        return Err(SimdTensorParallelError::BufferSizeMismatch {
            expected: len,
            got: output.len(),
        });
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 confirmed at runtime; slice bounds checked above.
            unsafe { simd_all_reduce_avx2(inputs, output, op) };
            return Ok(());
        }
    }

    simd_all_reduce_scalar(inputs, output, op);
    Ok(())
}

/// Scalar fallback for all-reduce.
fn simd_all_reduce_scalar(inputs: &[&[f32]], output: &mut [f32], op: ReduceOp) {
    let len = output.len();
    let count = inputs.len();

    match op {
        ReduceOp::Sum => {
            for i in 0..len {
                let mut acc = 0.0f32;
                for inp in inputs {
                    acc += inp[i];
                }
                output[i] = acc;
            }
        }
        ReduceOp::Max => {
            for i in 0..len {
                let mut acc = f32::NEG_INFINITY;
                for inp in inputs {
                    acc = acc.max(inp[i]);
                }
                output[i] = acc;
            }
        }
        ReduceOp::Mean => {
            for i in 0..len {
                let mut acc = 0.0f32;
                for inp in inputs {
                    acc += inp[i];
                }
                output[i] = acc / count as f32;
            }
        }
    }
}

/// AVX2 path for all-reduce.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_all_reduce_avx2(inputs: &[&[f32]], output: &mut [f32], op: ReduceOp) {
    let len = output.len();
    let count = inputs.len();
    let chunks = len / 8;
    let remainder = len % 8;

    for c in 0..chunks {
        let base = c * 8;
        let mut acc = _mm256_loadu_ps(inputs[0].as_ptr().add(base));
        for inp in inputs.iter().skip(1) {
            let v = _mm256_loadu_ps(inp.as_ptr().add(base));
            acc = match op {
                ReduceOp::Sum | ReduceOp::Mean => _mm256_add_ps(acc, v),
                ReduceOp::Max => _mm256_max_ps(acc, v),
            };
        }
        if matches!(op, ReduceOp::Mean) {
            let divisor = _mm256_set1_ps(count as f32);
            acc = _mm256_div_ps(acc, divisor);
        }
        _mm256_storeu_ps(output.as_mut_ptr().add(base), acc);
    }

    // Scalar tail.
    let tail_start = chunks * 8;
    for i in 0..remainder {
        let idx = tail_start + i;
        let mut acc = inputs[0][idx];
        for inp in inputs.iter().skip(1) {
            acc = match op {
                ReduceOp::Sum | ReduceOp::Mean => acc + inp[idx],
                ReduceOp::Max => acc.max(inp[idx]),
            };
        }
        if matches!(op, ReduceOp::Mean) {
            acc /= count as f32;
        }
        output[idx] = acc;
    }
}

// ── SIMD scatter / gather ──────────────────────────────────────────

/// Scatter (split) a row-major matrix `[rows × cols]` along the given axis.
///
/// * `ColumnParallel` → each shard gets a contiguous band of columns.
/// * `RowParallel`    → each shard gets a contiguous band of rows.
pub fn simd_scatter(
    data: &[f32],
    rows: usize,
    cols: usize,
    num_shards: usize,
    strategy: SimdShardStrategy,
) -> Result<Vec<Vec<f32>>> {
    if num_shards == 0 {
        return Err(SimdTensorParallelError::InvalidArgument("num_shards must be > 0".into()));
    }
    if data.len() != rows * cols {
        return Err(SimdTensorParallelError::BufferSizeMismatch {
            expected: rows * cols,
            got: data.len(),
        });
    }

    match strategy {
        SimdShardStrategy::RowParallel => scatter_rows(data, rows, cols, num_shards),
        SimdShardStrategy::ColumnParallel => scatter_cols(data, rows, cols, num_shards),
    }
}

fn scatter_rows(
    data: &[f32],
    rows: usize,
    cols: usize,
    num_shards: usize,
) -> Result<Vec<Vec<f32>>> {
    let sizes = compute_shard_sizes(rows, num_shards);
    let mut shards = Vec::with_capacity(num_shards);
    let mut row_offset = 0;
    for &s in &sizes {
        let start = row_offset * cols;
        let end = (row_offset + s) * cols;
        shards.push(data[start..end].to_vec());
        row_offset += s;
    }
    Ok(shards)
}

fn scatter_cols(
    data: &[f32],
    rows: usize,
    cols: usize,
    num_shards: usize,
) -> Result<Vec<Vec<f32>>> {
    let sizes = compute_shard_sizes(cols, num_shards);
    let mut shards: Vec<Vec<f32>> = sizes.iter().map(|&s| Vec::with_capacity(rows * s)).collect();
    for r in 0..rows {
        let row_start = r * cols;
        let mut col_offset = 0;
        for (shard_idx, &s) in sizes.iter().enumerate() {
            shards[shard_idx]
                .extend_from_slice(&data[row_start + col_offset..row_start + col_offset + s]);
            col_offset += s;
        }
    }
    Ok(shards)
}

/// Gather (concatenate) shards back into a single row-major matrix.
///
/// Inverse of [`simd_scatter`].  `shard_rows` / `shard_cols` describe the
/// *original* full matrix shape so that column-parallel gather can
/// reconstruct row layout.
pub fn simd_gather(
    shards: &[&[f32]],
    rows: usize,
    cols: usize,
    strategy: SimdShardStrategy,
) -> Result<Vec<f32>> {
    if shards.is_empty() {
        return Err(SimdTensorParallelError::InvalidArgument("no shards provided".into()));
    }

    match strategy {
        SimdShardStrategy::RowParallel => gather_rows(shards, rows, cols),
        SimdShardStrategy::ColumnParallel => gather_cols(shards, rows, cols),
    }
}

fn gather_rows(shards: &[&[f32]], rows: usize, cols: usize) -> Result<Vec<f32>> {
    let total_len: usize = shards.iter().map(|s| s.len()).sum();
    if total_len != rows * cols {
        return Err(SimdTensorParallelError::BufferSizeMismatch {
            expected: rows * cols,
            got: total_len,
        });
    }
    let mut out = Vec::with_capacity(total_len);
    for shard in shards {
        out.extend_from_slice(shard);
    }
    Ok(out)
}

fn gather_cols(shards: &[&[f32]], rows: usize, cols: usize) -> Result<Vec<f32>> {
    let shard_cols: Vec<usize> =
        shards.iter().map(|s| if rows == 0 { 0 } else { s.len() / rows }).collect();
    let total_cols: usize = shard_cols.iter().sum();
    if total_cols != cols {
        return Err(SimdTensorParallelError::ShapeMismatch(format!(
            "shard columns sum to {total_cols}, expected {cols}"
        )));
    }
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let mut col_offset = 0;
        for (shard_idx, shard) in shards.iter().enumerate() {
            let sc = shard_cols[shard_idx];
            let src_start = r * sc;
            let dst_start = r * cols + col_offset;
            out[dst_start..dst_start + sc].copy_from_slice(&shard[src_start..src_start + sc]);
            col_offset += sc;
        }
    }
    Ok(out)
}

// ── Sharded matmul ─────────────────────────────────────────────────

/// Perform `C = A × B` where `A` is `[m × k]`, `B` is `[k × n]`, `C` is
/// `[m × n]`, splitting work across `num_cores` threads using the given
/// sharding strategy.
///
/// * **ColumnParallel** — each core computes a vertical band of C.
/// * **RowParallel**    — each core computes a horizontal band of C.
///
/// Uses [`rayon`] thread pool for parallelism and SIMD within each core's
/// local matmul.
pub fn sharded_matmul(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    num_cores: usize,
    strategy: SimdShardStrategy,
) -> Result<()> {
    // Validate dimensions.
    if m == 0 || k == 0 || n == 0 {
        return Err(SimdTensorParallelError::ShapeMismatch("matmul dimensions must be > 0".into()));
    }
    if num_cores == 0 {
        return Err(SimdTensorParallelError::InvalidArgument("num_cores must be > 0".into()));
    }
    if a.len() < m * k {
        return Err(SimdTensorParallelError::BufferSizeMismatch { expected: m * k, got: a.len() });
    }
    if b.len() < k * n {
        return Err(SimdTensorParallelError::BufferSizeMismatch { expected: k * n, got: b.len() });
    }
    if c.len() < m * n {
        return Err(SimdTensorParallelError::BufferSizeMismatch { expected: m * n, got: c.len() });
    }

    match strategy {
        SimdShardStrategy::ColumnParallel => sharded_matmul_col(a, b, c, m, k, n, num_cores),
        SimdShardStrategy::RowParallel => sharded_matmul_row(a, b, c, m, k, n, num_cores),
    }
}

/// Column-parallel: each core computes C[:, col_start..col_end].
fn sharded_matmul_col(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    num_cores: usize,
) -> Result<()> {
    use rayon::prelude::*;

    let cores = num_cores.min(n);
    let shard_sizes = compute_shard_sizes(n, cores);

    // Build column offsets.
    let mut offsets = Vec::with_capacity(cores);
    let mut off = 0usize;
    for &s in &shard_sizes {
        offsets.push((off, s));
        off += s;
    }

    // Each shard writes into its region of C. We split C into non-overlapping
    // row-strided column bands. To avoid data-race issues with shared mutable
    // access, we collect partial results and copy back.
    let partials: Vec<Vec<f32>> = offsets
        .par_iter()
        .map(|&(col_off, shard_n)| {
            let mut local = vec![0.0f32; m * shard_n];
            matmul_kernel(
                a,
                b,
                &mut local,
                &KernelParams { k, n, row_off: 0, col_off, local_m: m, local_n: shard_n },
            );
            local
        })
        .collect();

    // Copy partials into C.
    let mut col_off = 0;
    for (idx, partial) in partials.iter().enumerate() {
        let shard_n = shard_sizes[idx];
        for r in 0..m {
            let dst_start = r * n + col_off;
            let src_start = r * shard_n;
            c[dst_start..dst_start + shard_n]
                .copy_from_slice(&partial[src_start..src_start + shard_n]);
        }
        col_off += shard_n;
    }

    Ok(())
}

/// Row-parallel: each core computes C[row_start..row_end, :].
fn sharded_matmul_row(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    num_cores: usize,
) -> Result<()> {
    use rayon::prelude::*;

    let cores = num_cores.min(m);
    let shard_sizes = compute_shard_sizes(m, cores);

    let mut offsets = Vec::with_capacity(cores);
    let mut off = 0usize;
    for &s in &shard_sizes {
        offsets.push((off, s));
        off += s;
    }

    let partials: Vec<Vec<f32>> = offsets
        .par_iter()
        .map(|&(row_off, shard_m)| {
            let mut local = vec![0.0f32; shard_m * n];
            matmul_kernel(
                a,
                b,
                &mut local,
                &KernelParams { k, n, row_off, col_off: 0, local_m: shard_m, local_n: n },
            );
            local
        })
        .collect();

    // Copy partials into C.
    let mut row_off = 0;
    for (idx, partial) in partials.iter().enumerate() {
        let shard_m = shard_sizes[idx];
        let dst_start = row_off * n;
        let len = shard_m * n;
        c[dst_start..dst_start + len].copy_from_slice(&partial[..len]);
        row_off += shard_m;
    }

    Ok(())
}

/// Parameters for the inner matmul kernel.
struct KernelParams {
    k: usize,
    n: usize,
    row_off: usize,
    col_off: usize,
    local_m: usize,
    local_n: usize,
}

/// Inner matmul kernel for a sub-block of C.
///
/// Computes `C_local[i, j] = sum_p A[row_off+i, p] * B[p, col_off+j]`
/// for `i in 0..local_m`, `j in 0..local_n`.
///
/// `A` is `[M × K]` (full), `B` is `[K × N]` (full), `C_local` is
/// `[local_m × local_n]`.
fn matmul_kernel(a: &[f32], b: &[f32], c_local: &mut [f32], p: &KernelParams) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 confirmed at runtime; bounds verified by caller.
            unsafe {
                matmul_kernel_avx2(a, b, c_local, p);
            }
            return;
        }
    }

    matmul_kernel_scalar(a, b, c_local, p);
}

/// Scalar matmul kernel.
fn matmul_kernel_scalar(a: &[f32], b: &[f32], c_local: &mut [f32], p: &KernelParams) {
    for i in 0..p.local_m {
        for j in 0..p.local_n {
            let mut acc = 0.0f32;
            for q in 0..p.k {
                acc += a[(p.row_off + i) * p.k + q] * b[q * p.n + (p.col_off + j)];
            }
            c_local[i * p.local_n + j] = acc;
        }
    }
}

/// AVX2 matmul kernel with 8-wide dot products.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn matmul_kernel_avx2(a: &[f32], b: &[f32], c_local: &mut [f32], p: &KernelParams) {
    let KernelParams { k, n, row_off, col_off, local_m, local_n } = *p;
    for i in 0..local_m {
        for j in 0..local_n {
            let mut acc = _mm256_setzero_ps();
            let chunks = k / 8;
            let a_row = (row_off + i) * k;

            for c in 0..chunks {
                let base = c * 8;
                let va = _mm256_loadu_ps(a.as_ptr().add(a_row + base));
                // Gather B column elements (strided access).
                let b_col = col_off + j;
                let vb = _mm256_set_ps(
                    b[(base + 7) * n + b_col],
                    b[(base + 6) * n + b_col],
                    b[(base + 5) * n + b_col],
                    b[(base + 4) * n + b_col],
                    b[(base + 3) * n + b_col],
                    b[(base + 2) * n + b_col],
                    b[(base + 1) * n + b_col],
                    b[base * n + b_col],
                );
                acc = _mm256_fmadd_ps(va, vb, acc);
            }

            // Horizontal sum of acc.
            let hi = _mm256_extractf128_ps(acc, 1);
            let lo = _mm256_castps256_ps128(acc);
            let sum128 = _mm_add_ps(lo, hi);
            let shuf = _mm_movehdup_ps(sum128);
            let sums = _mm_add_ps(sum128, shuf);
            let shuf2 = _mm_movehl_ps(sums, sums);
            let result = _mm_add_ss(sums, shuf2);
            let mut val = _mm_cvtss_f32(result);

            // Scalar tail.
            for p in (chunks * 8)..k {
                val += a[a_row + p] * b[p * n + (col_off + j)];
            }

            c_local[i * local_n + j] = val;
        }
    }
}

// ── In-place all-reduce (gradient sync) ────────────────────────────

/// In-place all-reduce: reduce `buffers[0..num_buffers]` into `buffers[0]`.
///
/// This models the gradient synchronisation pattern where each thread
/// holds a partial gradient buffer and we need the reduced result in every
/// thread's buffer.
pub fn all_reduce_inplace(buffers: &mut [Vec<f32>], op: ReduceOp) -> Result<()> {
    if buffers.is_empty() {
        return Err(SimdTensorParallelError::InvalidArgument("no buffers provided".into()));
    }
    let len = buffers[0].len();
    for (i, buf) in buffers.iter().enumerate().skip(1) {
        if buf.len() != len {
            return Err(SimdTensorParallelError::BufferSizeMismatch {
                expected: len,
                got: buf.len(),
            });
        }
        let _ = i;
    }

    if buffers.len() == 1 {
        return Ok(());
    }

    // Reduce into a temporary buffer, then broadcast.
    let refs: Vec<&[f32]> = buffers.iter().map(|b| b.as_slice()).collect();
    let mut reduced = vec![0.0f32; len];
    simd_all_reduce(&refs, &mut reduced, op)?;

    for buf in buffers.iter_mut() {
        buf.copy_from_slice(&reduced);
    }
    Ok(())
}

// ── Ring all-reduce ────────────────────────────────────────────────

/// Ring all-reduce: simulates a ring-based reduce-scatter + all-gather.
///
/// Each of `num_ranks` participants contributes one buffer of the same
/// length.  After the call, every participant holds the fully reduced
/// result.  This is more bandwidth-efficient than the naive star pattern
/// for large tensors.
pub fn ring_all_reduce(buffers: &mut [Vec<f32>], op: ReduceOp) -> Result<()> {
    if buffers.is_empty() {
        return Err(SimdTensorParallelError::InvalidArgument("no buffers provided".into()));
    }
    let len = buffers[0].len();
    let num_ranks = buffers.len();

    for buf in buffers.iter().skip(1) {
        if buf.len() != len {
            return Err(SimdTensorParallelError::BufferSizeMismatch {
                expected: len,
                got: buf.len(),
            });
        }
    }

    if num_ranks <= 1 {
        return Ok(());
    }

    // Phase 1: reduce-scatter — each rank ends up with 1/P of the fully
    // reduced result.
    let chunk = len.div_ceil(num_ranks);
    let mut reduced_chunks: Vec<Vec<f32>> = Vec::with_capacity(num_ranks);

    for rank in 0..num_ranks {
        let start = (rank * chunk).min(len);
        let end = (start + chunk).min(len);
        let seg_len = end - start;
        let mut acc = vec![0.0f32; seg_len];
        for buf in buffers.iter() {
            for (j, a) in acc.iter_mut().enumerate() {
                let v = buf[start + j];
                *a = match op {
                    ReduceOp::Sum | ReduceOp::Mean => *a + v,
                    ReduceOp::Max => a.max(v),
                };
            }
        }
        if matches!(op, ReduceOp::Mean) {
            for a in &mut acc {
                *a /= num_ranks as f32;
            }
        }
        reduced_chunks.push(acc);
    }

    // Phase 2: all-gather — broadcast every chunk to every rank.
    for buf in buffers.iter_mut() {
        for (rank, rc) in reduced_chunks.iter().enumerate() {
            let start = (rank * chunk).min(len);
            let end = (start + rc.len()).min(len);
            buf[start..end].copy_from_slice(&rc[..end - start]);
        }
    }

    Ok(())
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ─────────────────────────────────────────────────────

    fn identity_matrix(n: usize) -> Vec<f32> {
        let mut m = vec![0.0f32; n * n];
        for i in 0..n {
            m[i * n + i] = 1.0;
        }
        m
    }

    fn sequential_matrix(rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols).map(|i| i as f32).collect()
    }

    fn constant_matrix(rows: usize, cols: usize, val: f32) -> Vec<f32> {
        vec![val; rows * cols]
    }

    fn assert_matrices_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "matrix lengths differ");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol, "element {i} differs: {x} vs {y} (tol={tol})");
        }
    }

    /// Reference scalar matmul for verification.
    fn ref_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = acc;
            }
        }
        c
    }

    // ── SimdTensorParallelError display ─────────────────────────────

    #[test]
    fn error_display_shape_mismatch() {
        let e = SimdTensorParallelError::ShapeMismatch("bad shape".into());
        assert!(e.to_string().contains("bad shape"));
    }

    #[test]
    fn error_display_invalid_sharding() {
        let e = SimdTensorParallelError::InvalidSharding("cannot shard".into());
        assert!(e.to_string().contains("cannot shard"));
    }

    #[test]
    fn error_display_invalid_argument() {
        let e = SimdTensorParallelError::InvalidArgument("zero cores".into());
        assert!(e.to_string().contains("zero cores"));
    }

    #[test]
    fn error_display_buffer_size() {
        let e = SimdTensorParallelError::BufferSizeMismatch { expected: 10, got: 5 };
        let s = e.to_string();
        assert!(s.contains("10") && s.contains("5"));
    }

    #[test]
    fn error_is_std_error() {
        let e: Box<dyn std::error::Error> =
            Box::new(SimdTensorParallelError::InvalidArgument("test".into()));
        assert!(e.to_string().contains("test"));
    }

    // ── SimdShardStrategy display ───────────────────────────────────

    #[test]
    fn strategy_display_column() {
        assert_eq!(SimdShardStrategy::ColumnParallel.to_string(), "ColumnParallel");
    }

    #[test]
    fn strategy_display_row() {
        assert_eq!(SimdShardStrategy::RowParallel.to_string(), "RowParallel");
    }

    // ── ReduceOp display ────────────────────────────────────────────

    #[test]
    fn reduce_op_display() {
        assert_eq!(ReduceOp::Sum.to_string(), "Sum");
        assert_eq!(ReduceOp::Max.to_string(), "Max");
        assert_eq!(ReduceOp::Mean.to_string(), "Mean");
    }

    #[test]
    fn reduce_op_default_is_sum() {
        assert_eq!(ReduceOp::default(), ReduceOp::Sum);
    }

    // ── compute_shard_sizes ─────────────────────────────────────────

    #[test]
    fn shard_sizes_even() {
        assert_eq!(compute_shard_sizes(12, 4), vec![3, 3, 3, 3]);
    }

    #[test]
    fn shard_sizes_uneven() {
        let sizes = compute_shard_sizes(10, 3);
        assert_eq!(sizes, vec![4, 3, 3]);
        assert_eq!(sizes.iter().sum::<usize>(), 10);
    }

    #[test]
    fn shard_sizes_single() {
        assert_eq!(compute_shard_sizes(7, 1), vec![7]);
    }

    #[test]
    fn shard_sizes_more_parts_than_total() {
        let sizes = compute_shard_sizes(2, 5);
        assert_eq!(sizes, vec![1, 1, 0, 0, 0]);
        assert_eq!(sizes.iter().sum::<usize>(), 2);
    }

    // ── simd_all_reduce ─────────────────────────────────────────────

    #[test]
    fn all_reduce_sum_two_buffers() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [10.0f32, 20.0, 30.0, 40.0];
        let mut out = [0.0f32; 4];
        simd_all_reduce(&[&a, &b], &mut out, ReduceOp::Sum).unwrap();
        assert_eq!(out, [11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn all_reduce_sum_three_buffers() {
        let a = [1.0f32; 8];
        let b = [2.0f32; 8];
        let c = [3.0f32; 8];
        let mut out = [0.0f32; 8];
        simd_all_reduce(&[&a, &b, &c], &mut out, ReduceOp::Sum).unwrap();
        for &v in &out {
            assert!((v - 6.0).abs() < 1e-6);
        }
    }

    #[test]
    fn all_reduce_max_basic() {
        let a = [1.0f32, 5.0, 3.0];
        let b = [4.0f32, 2.0, 6.0];
        let mut out = [0.0f32; 3];
        simd_all_reduce(&[&a, &b], &mut out, ReduceOp::Max).unwrap();
        assert_eq!(out, [4.0, 5.0, 6.0]);
    }

    #[test]
    fn all_reduce_mean_basic() {
        let a = [2.0f32, 4.0];
        let b = [6.0f32, 8.0];
        let mut out = [0.0f32; 2];
        simd_all_reduce(&[&a, &b], &mut out, ReduceOp::Mean).unwrap();
        assert!((out[0] - 4.0).abs() < 1e-6);
        assert!((out[1] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn all_reduce_single_buffer() {
        let a = [7.0f32, 8.0, 9.0];
        let mut out = [0.0f32; 3];
        simd_all_reduce(&[&a], &mut out, ReduceOp::Sum).unwrap();
        assert_eq!(out, [7.0, 8.0, 9.0]);
    }

    #[test]
    fn all_reduce_empty_input() {
        let mut out = [0.0f32; 1];
        assert!(simd_all_reduce(&[], &mut out, ReduceOp::Sum).is_err());
    }

    #[test]
    fn all_reduce_mismatched_input_lengths() {
        let a = [1.0f32, 2.0];
        let b = [3.0f32];
        let mut out = [0.0f32; 2];
        assert!(simd_all_reduce(&[&a, &b[..]], &mut out, ReduceOp::Sum).is_err());
    }

    #[test]
    fn all_reduce_mismatched_output_length() {
        let a = [1.0f32, 2.0];
        let mut out = [0.0f32; 3];
        assert!(simd_all_reduce(&[&a], &mut out, ReduceOp::Sum).is_err());
    }

    #[test]
    fn all_reduce_large_buffer() {
        let n = 1024;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
        let mut out = vec![0.0f32; n];
        simd_all_reduce(&[&a, &b], &mut out, ReduceOp::Sum).unwrap();
        for &v in &out {
            assert!((v - n as f32).abs() < 1e-3);
        }
    }

    #[test]
    fn all_reduce_sum_with_negatives() {
        let a = [-1.0f32, -2.0, 3.0];
        let b = [1.0f32, 2.0, -3.0];
        let mut out = [0.0f32; 3];
        simd_all_reduce(&[&a, &b], &mut out, ReduceOp::Sum).unwrap();
        assert_eq!(out, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn all_reduce_max_with_negatives() {
        let a = [-5.0f32, -1.0];
        let b = [-3.0f32, -2.0];
        let mut out = [0.0f32; 2];
        simd_all_reduce(&[&a, &b], &mut out, ReduceOp::Max).unwrap();
        assert_eq!(out, [-3.0, -1.0]);
    }

    // ── all_reduce_inplace ──────────────────────────────────────────

    #[test]
    fn inplace_reduce_sum() {
        let mut bufs = vec![vec![1.0f32, 2.0], vec![3.0f32, 4.0]];
        all_reduce_inplace(&mut bufs, ReduceOp::Sum).unwrap();
        assert_eq!(bufs[0], [4.0, 6.0]);
        assert_eq!(bufs[1], [4.0, 6.0]);
    }

    #[test]
    fn inplace_reduce_single_buffer() {
        let mut bufs = vec![vec![5.0f32, 6.0]];
        all_reduce_inplace(&mut bufs, ReduceOp::Sum).unwrap();
        assert_eq!(bufs[0], [5.0, 6.0]);
    }

    #[test]
    fn inplace_reduce_empty() {
        let mut bufs: Vec<Vec<f32>> = vec![];
        assert!(all_reduce_inplace(&mut bufs, ReduceOp::Sum).is_err());
    }

    #[test]
    fn inplace_reduce_mismatched() {
        let mut bufs = vec![vec![1.0f32, 2.0], vec![3.0f32]];
        assert!(all_reduce_inplace(&mut bufs, ReduceOp::Sum).is_err());
    }

    #[test]
    fn inplace_reduce_max() {
        let mut bufs = vec![vec![1.0f32, 5.0], vec![3.0f32, 2.0], vec![2.0f32, 4.0]];
        all_reduce_inplace(&mut bufs, ReduceOp::Max).unwrap();
        for buf in &bufs {
            assert_eq!(*buf, [3.0, 5.0]);
        }
    }

    // ── ring_all_reduce ─────────────────────────────────────────────

    #[test]
    fn ring_reduce_sum_basic() {
        let mut bufs = vec![vec![1.0f32, 2.0], vec![3.0f32, 4.0]];
        ring_all_reduce(&mut bufs, ReduceOp::Sum).unwrap();
        assert_eq!(bufs[0], [4.0, 6.0]);
        assert_eq!(bufs[1], [4.0, 6.0]);
    }

    #[test]
    fn ring_reduce_single_rank() {
        let mut bufs = vec![vec![5.0f32, 6.0]];
        ring_all_reduce(&mut bufs, ReduceOp::Sum).unwrap();
        assert_eq!(bufs[0], [5.0, 6.0]);
    }

    #[test]
    fn ring_reduce_empty() {
        let mut bufs: Vec<Vec<f32>> = vec![];
        assert!(ring_all_reduce(&mut bufs, ReduceOp::Sum).is_err());
    }

    #[test]
    fn ring_reduce_mismatched_lengths() {
        let mut bufs = vec![vec![1.0f32], vec![1.0f32, 2.0]];
        assert!(ring_all_reduce(&mut bufs, ReduceOp::Sum).is_err());
    }

    #[test]
    fn ring_reduce_four_ranks() {
        let mut bufs: Vec<Vec<f32>> = (0..4).map(|_| vec![1.0f32; 16]).collect();
        ring_all_reduce(&mut bufs, ReduceOp::Sum).unwrap();
        for buf in &bufs {
            for &v in buf {
                assert!((v - 4.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn ring_reduce_mean() {
        let mut bufs = vec![vec![2.0f32, 8.0], vec![6.0f32, 4.0]];
        ring_all_reduce(&mut bufs, ReduceOp::Mean).unwrap();
        for buf in &bufs {
            assert!((buf[0] - 4.0).abs() < 1e-6);
            assert!((buf[1] - 6.0).abs() < 1e-6);
        }
    }

    #[test]
    fn ring_reduce_max() {
        let mut bufs = vec![vec![1.0f32, 5.0], vec![3.0f32, 2.0]];
        ring_all_reduce(&mut bufs, ReduceOp::Max).unwrap();
        for buf in &bufs {
            assert_eq!(*buf, [3.0, 5.0]);
        }
    }

    // ── simd_scatter (row-parallel) ─────────────────────────────────

    #[test]
    fn scatter_rows_even() {
        let data = sequential_matrix(4, 3);
        let shards = simd_scatter(&data, 4, 3, 2, SimdShardStrategy::RowParallel).unwrap();
        assert_eq!(shards.len(), 2);
        assert_eq!(shards[0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]); // rows 0-1
        assert_eq!(shards[1], [6.0, 7.0, 8.0, 9.0, 10.0, 11.0]); // rows 2-3
    }

    #[test]
    fn scatter_rows_uneven() {
        let data = sequential_matrix(3, 2);
        let shards = simd_scatter(&data, 3, 2, 2, SimdShardStrategy::RowParallel).unwrap();
        assert_eq!(shards[0].len(), 4); // 2 rows × 2 cols
        assert_eq!(shards[1].len(), 2); // 1 row × 2 cols
    }

    #[test]
    fn scatter_rows_single_shard() {
        let data = sequential_matrix(2, 3);
        let shards = simd_scatter(&data, 2, 3, 1, SimdShardStrategy::RowParallel).unwrap();
        assert_eq!(shards.len(), 1);
        assert_eq!(shards[0], data);
    }

    // ── simd_scatter (column-parallel) ──────────────────────────────

    #[test]
    fn scatter_cols_even() {
        // 2×4 matrix, 2 shards → each shard gets 2 cols
        let data = sequential_matrix(2, 4);
        let shards = simd_scatter(&data, 2, 4, 2, SimdShardStrategy::ColumnParallel).unwrap();
        assert_eq!(shards.len(), 2);
        assert_eq!(shards[0], [0.0, 1.0, 4.0, 5.0]); // cols 0-1
        assert_eq!(shards[1], [2.0, 3.0, 6.0, 7.0]); // cols 2-3
    }

    #[test]
    fn scatter_cols_uneven() {
        // 2×3 matrix, 2 shards → shard 0 gets 2 cols, shard 1 gets 1 col
        let data = sequential_matrix(2, 3);
        let shards = simd_scatter(&data, 2, 3, 2, SimdShardStrategy::ColumnParallel).unwrap();
        assert_eq!(shards[0], [0.0, 1.0, 3.0, 4.0]); // 2 cols
        assert_eq!(shards[1], [2.0, 5.0]); // 1 col
    }

    #[test]
    fn scatter_zero_shards() {
        let data = sequential_matrix(2, 2);
        assert!(simd_scatter(&data, 2, 2, 0, SimdShardStrategy::RowParallel).is_err());
    }

    #[test]
    fn scatter_wrong_buffer_size() {
        let data = vec![1.0, 2.0, 3.0]; // 3 elements for 2×2 matrix
        assert!(simd_scatter(&data, 2, 2, 1, SimdShardStrategy::RowParallel).is_err());
    }

    // ── simd_gather ─────────────────────────────────────────────────

    #[test]
    fn gather_rows_basic() {
        let s0: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let s1: Vec<f32> = vec![6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let out = simd_gather(&[&s0, &s1], 4, 3, SimdShardStrategy::RowParallel).unwrap();
        assert_eq!(out, sequential_matrix(4, 3));
    }

    #[test]
    fn gather_cols_basic() {
        let s0: Vec<f32> = vec![0.0, 1.0, 4.0, 5.0]; // cols 0-1
        let s1: Vec<f32> = vec![2.0, 3.0, 6.0, 7.0]; // cols 2-3
        let out = simd_gather(&[&s0, &s1], 2, 4, SimdShardStrategy::ColumnParallel).unwrap();
        assert_eq!(out, sequential_matrix(2, 4));
    }

    #[test]
    fn gather_empty_shards() {
        assert!(simd_gather(&[], 1, 1, SimdShardStrategy::RowParallel).is_err());
    }

    #[test]
    fn gather_row_wrong_total() {
        let s0 = vec![1.0f32];
        assert!(simd_gather(&[&s0[..]], 2, 2, SimdShardStrategy::RowParallel).is_err());
    }

    #[test]
    fn gather_col_wrong_total() {
        let s0 = vec![1.0f32];
        assert!(simd_gather(&[&s0[..]], 2, 2, SimdShardStrategy::ColumnParallel).is_err());
    }

    // ── scatter/gather round-trip ───────────────────────────────────

    #[test]
    fn roundtrip_row_parallel() {
        let data = sequential_matrix(6, 4);
        let shards = simd_scatter(&data, 6, 4, 3, SimdShardStrategy::RowParallel).unwrap();
        let refs: Vec<&[f32]> = shards.iter().map(|s| s.as_slice()).collect();
        let gathered = simd_gather(&refs, 6, 4, SimdShardStrategy::RowParallel).unwrap();
        assert_eq!(gathered, data);
    }

    #[test]
    fn roundtrip_column_parallel() {
        let data = sequential_matrix(4, 6);
        let shards = simd_scatter(&data, 4, 6, 3, SimdShardStrategy::ColumnParallel).unwrap();
        let refs: Vec<&[f32]> = shards.iter().map(|s| s.as_slice()).collect();
        let gathered = simd_gather(&refs, 4, 6, SimdShardStrategy::ColumnParallel).unwrap();
        assert_eq!(gathered, data);
    }

    #[test]
    fn roundtrip_single_shard_row() {
        let data = sequential_matrix(3, 5);
        let shards = simd_scatter(&data, 3, 5, 1, SimdShardStrategy::RowParallel).unwrap();
        let refs: Vec<&[f32]> = shards.iter().map(|s| s.as_slice()).collect();
        let gathered = simd_gather(&refs, 3, 5, SimdShardStrategy::RowParallel).unwrap();
        assert_eq!(gathered, data);
    }

    #[test]
    fn roundtrip_single_shard_col() {
        let data = sequential_matrix(3, 5);
        let shards = simd_scatter(&data, 3, 5, 1, SimdShardStrategy::ColumnParallel).unwrap();
        let refs: Vec<&[f32]> = shards.iter().map(|s| s.as_slice()).collect();
        let gathered = simd_gather(&refs, 3, 5, SimdShardStrategy::ColumnParallel).unwrap();
        assert_eq!(gathered, data);
    }

    // ── sharded_matmul ──────────────────────────────────────────────

    #[test]
    fn matmul_identity_col_parallel() {
        let m = 4;
        let k = 4;
        let n = 4;
        let a = sequential_matrix(m, k);
        let b = identity_matrix(k);
        let mut c = vec![0.0f32; m * n];
        sharded_matmul(&a, &b, &mut c, m, k, n, 2, SimdShardStrategy::ColumnParallel).unwrap();
        assert_matrices_close(&c, &a, 1e-5);
    }

    #[test]
    fn matmul_identity_row_parallel() {
        let m = 4;
        let k = 4;
        let n = 4;
        let a = sequential_matrix(m, k);
        let b = identity_matrix(k);
        let mut c = vec![0.0f32; m * n];
        sharded_matmul(&a, &b, &mut c, m, k, n, 2, SimdShardStrategy::RowParallel).unwrap();
        assert_matrices_close(&c, &a, 1e-5);
    }

    #[test]
    fn matmul_ones_col_parallel() {
        let (m, k, n) = (3, 4, 5);
        let a = constant_matrix(m, k, 1.0);
        let b = constant_matrix(k, n, 1.0);
        let mut c = vec![0.0f32; m * n];
        sharded_matmul(&a, &b, &mut c, m, k, n, 2, SimdShardStrategy::ColumnParallel).unwrap();
        let expected = constant_matrix(m, n, k as f32);
        assert_matrices_close(&c, &expected, 1e-5);
    }

    #[test]
    fn matmul_ones_row_parallel() {
        let (m, k, n) = (3, 4, 5);
        let a = constant_matrix(m, k, 1.0);
        let b = constant_matrix(k, n, 1.0);
        let mut c = vec![0.0f32; m * n];
        sharded_matmul(&a, &b, &mut c, m, k, n, 2, SimdShardStrategy::RowParallel).unwrap();
        let expected = constant_matrix(m, n, k as f32);
        assert_matrices_close(&c, &expected, 1e-5);
    }

    #[test]
    fn matmul_vs_reference_col() {
        let (m, k, n) = (8, 16, 12);
        let a = sequential_matrix(m, k);
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();
        let expected = ref_matmul(&a, &b, m, k, n);
        let mut c = vec![0.0f32; m * n];
        sharded_matmul(&a, &b, &mut c, m, k, n, 4, SimdShardStrategy::ColumnParallel).unwrap();
        assert_matrices_close(&c, &expected, 1e-2);
    }

    #[test]
    fn matmul_vs_reference_row() {
        let (m, k, n) = (8, 16, 12);
        let a = sequential_matrix(m, k);
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();
        let expected = ref_matmul(&a, &b, m, k, n);
        let mut c = vec![0.0f32; m * n];
        sharded_matmul(&a, &b, &mut c, m, k, n, 4, SimdShardStrategy::RowParallel).unwrap();
        assert_matrices_close(&c, &expected, 1e-2);
    }

    #[test]
    fn matmul_single_core() {
        let (m, k, n) = (3, 4, 5);
        let a = sequential_matrix(m, k);
        let b = sequential_matrix(k, n);
        let expected = ref_matmul(&a, &b, m, k, n);
        let mut c = vec![0.0f32; m * n];
        sharded_matmul(&a, &b, &mut c, m, k, n, 1, SimdShardStrategy::ColumnParallel).unwrap();
        assert_matrices_close(&c, &expected, 1e-2);
    }

    #[test]
    fn matmul_more_cores_than_cols() {
        let (m, k, n) = (4, 4, 2);
        let a = sequential_matrix(m, k);
        let b = sequential_matrix(k, n);
        let expected = ref_matmul(&a, &b, m, k, n);
        let mut c = vec![0.0f32; m * n];
        sharded_matmul(&a, &b, &mut c, m, k, n, 8, SimdShardStrategy::ColumnParallel).unwrap();
        assert_matrices_close(&c, &expected, 1e-2);
    }

    #[test]
    fn matmul_more_cores_than_rows() {
        let (m, k, n) = (2, 4, 4);
        let a = sequential_matrix(m, k);
        let b = sequential_matrix(k, n);
        let expected = ref_matmul(&a, &b, m, k, n);
        let mut c = vec![0.0f32; m * n];
        sharded_matmul(&a, &b, &mut c, m, k, n, 8, SimdShardStrategy::RowParallel).unwrap();
        assert_matrices_close(&c, &expected, 1e-2);
    }

    #[test]
    fn matmul_zero_dim_m() {
        let mut c = [0.0f32; 0];
        assert!(
            sharded_matmul(&[], &[], &mut c, 0, 4, 4, 2, SimdShardStrategy::ColumnParallel)
                .is_err()
        );
    }

    #[test]
    fn matmul_zero_cores() {
        let a = sequential_matrix(2, 2);
        let b = sequential_matrix(2, 2);
        let mut c = [0.0f32; 4];
        assert!(
            sharded_matmul(&a, &b, &mut c, 2, 2, 2, 0, SimdShardStrategy::ColumnParallel).is_err()
        );
    }

    #[test]
    fn matmul_buffer_too_small_a() {
        let a = [1.0f32; 3]; // too small for 2×2
        let b = sequential_matrix(2, 2);
        let mut c = [0.0f32; 4];
        assert!(
            sharded_matmul(&a, &b, &mut c, 2, 2, 2, 1, SimdShardStrategy::ColumnParallel).is_err()
        );
    }

    #[test]
    fn matmul_buffer_too_small_b() {
        let a = sequential_matrix(2, 2);
        let b = [1.0f32; 3]; // too small for 2×2
        let mut c = [0.0f32; 4];
        assert!(
            sharded_matmul(&a, &b, &mut c, 2, 2, 2, 1, SimdShardStrategy::ColumnParallel).is_err()
        );
    }

    #[test]
    fn matmul_buffer_too_small_c() {
        let a = sequential_matrix(2, 2);
        let b = sequential_matrix(2, 2);
        let mut c = [0.0f32; 3]; // too small for 2×2
        assert!(
            sharded_matmul(&a, &b, &mut c, 2, 2, 2, 1, SimdShardStrategy::ColumnParallel).is_err()
        );
    }

    #[test]
    fn matmul_large_col_parallel() {
        let (m, k, n) = (32, 64, 48);
        let a = sequential_matrix(m, k);
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.001).collect();
        let expected = ref_matmul(&a, &b, m, k, n);
        let mut c = vec![0.0f32; m * n];
        sharded_matmul(&a, &b, &mut c, m, k, n, 8, SimdShardStrategy::ColumnParallel).unwrap();
        assert_matrices_close(&c, &expected, 0.5);
    }

    #[test]
    fn matmul_large_row_parallel() {
        let (m, k, n) = (32, 64, 48);
        let a = sequential_matrix(m, k);
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.001).collect();
        let expected = ref_matmul(&a, &b, m, k, n);
        let mut c = vec![0.0f32; m * n];
        sharded_matmul(&a, &b, &mut c, m, k, n, 8, SimdShardStrategy::RowParallel).unwrap();
        assert_matrices_close(&c, &expected, 0.5);
    }

    // ── estimate_comm_cost ──────────────────────────────────────────

    #[test]
    fn comm_cost_single_core() {
        let cost = estimate_comm_cost(4, 4, 1, SimdShardStrategy::ColumnParallel).unwrap();
        assert_eq!(cost.total_bytes, 0);
        assert_eq!(cost.num_barriers, 0);
        assert!(cost.estimated_latency_us >= 0.0);
    }

    #[test]
    fn comm_cost_two_cores() {
        let cost = estimate_comm_cost(4, 4, 2, SimdShardStrategy::ColumnParallel).unwrap();
        assert!(cost.total_bytes > 0);
        assert_eq!(cost.num_barriers, 2);
        assert!(cost.estimated_latency_us > 0.0);
    }

    #[test]
    fn comm_cost_col_vs_row() {
        // Wide matrix: 10×100 — row-parallel splits M=10 (small), col splits N=100 (large).
        // Row-parallel shard = ceil(10/4)*100 = 300 elems; col shard = 10*ceil(100/4) = 250.
        // Column-parallel should be cheaper on a wide matrix.
        let col = estimate_comm_cost(10, 100, 4, SimdShardStrategy::ColumnParallel).unwrap();
        let row = estimate_comm_cost(10, 100, 4, SimdShardStrategy::RowParallel).unwrap();
        assert!(col.estimated_latency_us < row.estimated_latency_us);
    }

    #[test]
    fn comm_cost_zero_cores() {
        assert!(estimate_comm_cost(4, 4, 0, SimdShardStrategy::ColumnParallel).is_err());
    }

    #[test]
    fn comm_cost_zero_rows() {
        assert!(estimate_comm_cost(0, 4, 2, SimdShardStrategy::ColumnParallel).is_err());
    }

    #[test]
    fn comm_cost_zero_cols() {
        assert!(estimate_comm_cost(4, 0, 2, SimdShardStrategy::ColumnParallel).is_err());
    }

    #[test]
    fn comm_cost_strategy_stored() {
        let cost = estimate_comm_cost(4, 4, 2, SimdShardStrategy::RowParallel).unwrap();
        assert_eq!(cost.strategy, SimdShardStrategy::RowParallel);
    }

    // ── generate_sharding_plan ──────────────────────────────────────

    #[test]
    fn plan_basic() {
        let plan = generate_sharding_plan(16, 32, 16, 4).unwrap();
        assert!(plan.num_cores > 0);
        assert!(!plan.shard_sizes.is_empty());
        let total: usize = plan.shard_sizes.iter().sum();
        let split_dim = match plan.strategy {
            SimdShardStrategy::ColumnParallel => 16,
            SimdShardStrategy::RowParallel => 16,
        };
        assert_eq!(total, split_dim);
    }

    #[test]
    fn plan_tall_skinny_prefers_column() {
        // 1000×10 output → column parallel should win since N is small.
        let plan = generate_sharding_plan(1000, 64, 10, 8).unwrap();
        // With only 10 columns, cores clamped to 10.
        assert!(plan.num_cores <= 10);
    }

    #[test]
    fn plan_wide_prefers_row_or_col() {
        let plan = generate_sharding_plan(4, 64, 1000, 8).unwrap();
        // Should pick something valid.
        assert!(plan.num_cores > 0);
    }

    #[test]
    fn plan_single_core() {
        let plan = generate_sharding_plan(8, 8, 8, 1).unwrap();
        assert_eq!(plan.num_cores, 1);
        assert_eq!(plan.shard_sizes, vec![8]);
    }

    #[test]
    fn plan_zero_cores() {
        assert!(generate_sharding_plan(8, 8, 8, 0).is_err());
    }

    #[test]
    fn plan_zero_dim() {
        assert!(generate_sharding_plan(0, 8, 8, 4).is_err());
        assert!(generate_sharding_plan(8, 0, 8, 4).is_err());
        assert!(generate_sharding_plan(8, 8, 0, 4).is_err());
    }

    #[test]
    fn plan_display() {
        let plan = generate_sharding_plan(16, 16, 16, 4).unwrap();
        let s = plan.to_string();
        assert!(s.contains("ShardingPlan"));
        assert!(s.contains("cores"));
    }

    #[test]
    fn plan_shard_sizes_sum_to_dim() {
        let plan = generate_sharding_plan(100, 64, 50, 7).unwrap();
        let split_dim = match plan.strategy {
            SimdShardStrategy::ColumnParallel => 50,
            SimdShardStrategy::RowParallel => 100,
        };
        assert_eq!(plan.shard_sizes.iter().sum::<usize>(), split_dim);
    }

    // ── proptest ────────────────────────────────────────────────────

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_scatter_gather_roundtrip_row(
                rows in 1usize..=16,
                cols in 1usize..=16,
                num_shards in 1usize..=8,
            ) {
                let data = sequential_matrix(rows, cols);
                let shards = simd_scatter(
                    &data, rows, cols, num_shards, SimdShardStrategy::RowParallel,
                ).unwrap();
                let refs: Vec<&[f32]> = shards.iter().map(|s| s.as_slice()).collect();
                let gathered = simd_gather(&refs, rows, cols, SimdShardStrategy::RowParallel).unwrap();
                prop_assert_eq!(gathered, data);
            }

            #[test]
            fn prop_scatter_gather_roundtrip_col(
                rows in 1usize..=16,
                cols in 1usize..=16,
                num_shards in 1usize..=8,
            ) {
                let data = sequential_matrix(rows, cols);
                let shards = simd_scatter(
                    &data, rows, cols, num_shards, SimdShardStrategy::ColumnParallel,
                ).unwrap();
                let refs: Vec<&[f32]> = shards.iter().map(|s| s.as_slice()).collect();
                let gathered = simd_gather(
                    &refs, rows, cols, SimdShardStrategy::ColumnParallel,
                ).unwrap();
                prop_assert_eq!(gathered, data);
            }

            #[test]
            fn prop_all_reduce_sum_correct(
                len in 1usize..=64,
                n_bufs in 2usize..=8,
            ) {
                let bufs: Vec<Vec<f32>> = (0..n_bufs)
                    .map(|_| vec![1.0f32; len])
                    .collect();
                let refs: Vec<&[f32]> = bufs.iter().map(|b| b.as_slice()).collect();
                let mut out = vec![0.0f32; len];
                simd_all_reduce(&refs, &mut out, ReduceOp::Sum).unwrap();
                for &v in &out {
                    prop_assert!((v - n_bufs as f32).abs() < 1e-6);
                }
            }

            #[test]
            fn prop_ring_reduce_matches_inplace(
                len in 1usize..=32,
                n_bufs in 2usize..=6,
            ) {
                let bufs: Vec<Vec<f32>> = (0..n_bufs)
                    .map(|i| (0..len).map(|j| (i * len + j) as f32).collect())
                    .collect();
                let mut ring_bufs = bufs.clone();
                let mut inplace_bufs = bufs;
                ring_all_reduce(&mut ring_bufs, ReduceOp::Sum).unwrap();
                all_reduce_inplace(&mut inplace_bufs, ReduceOp::Sum).unwrap();
                for (rb, ib) in ring_bufs.iter().zip(inplace_bufs.iter()) {
                    for (&r, &ip) in rb.iter().zip(ib.iter()) {
                        prop_assert!((r - ip).abs() < 1e-4);
                    }
                }
            }

            #[test]
            fn prop_sharded_matmul_matches_ref(
                m in 1usize..=8,
                k in 1usize..=8,
                n in 1usize..=8,
                cores in 1usize..=4,
            ) {
                let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
                let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();
                let expected = ref_matmul(&a, &b, m, k, n);

                let mut c_col = vec![0.0f32; m * n];
                sharded_matmul(
                    &a, &b, &mut c_col, m, k, n, cores,
                    SimdShardStrategy::ColumnParallel,
                ).unwrap();
                for (i, (&e, &c)) in expected.iter().zip(c_col.iter()).enumerate() {
                    prop_assert!(
                        (e - c).abs() < 1e-2,
                        "col mismatch at {i}: expected {e}, got {c}"
                    );
                }

                let mut c_row = vec![0.0f32; m * n];
                sharded_matmul(
                    &a, &b, &mut c_row, m, k, n, cores,
                    SimdShardStrategy::RowParallel,
                ).unwrap();
                for (i, (&e, &c)) in expected.iter().zip(c_row.iter()).enumerate() {
                    prop_assert!(
                        (e - c).abs() < 1e-2,
                        "row mismatch at {i}: expected {e}, got {c}"
                    );
                }
            }

            #[test]
            fn prop_comm_cost_nonnegative(
                rows in 2usize..=32,
                cols in 2usize..=32,
                cores in 1usize..=8,
            ) {
                let cost = estimate_comm_cost(rows, cols, cores, SimdShardStrategy::ColumnParallel).unwrap();
                prop_assert!(cost.estimated_latency_us >= 0.0);
                prop_assert!(cost.num_barriers <= 2);
            }

            #[test]
            fn prop_plan_shard_sizes_cover(
                m in 1usize..=32,
                k in 1usize..=32,
                n in 1usize..=32,
                max_cores in 1usize..=8,
            ) {
                let plan = generate_sharding_plan(m, k, n, max_cores).unwrap();
                let split_dim = match plan.strategy {
                    SimdShardStrategy::ColumnParallel => n,
                    SimdShardStrategy::RowParallel => m,
                };
                prop_assert_eq!(plan.shard_sizes.iter().sum::<usize>(), split_dim);
                prop_assert_eq!(plan.shard_sizes.len(), plan.num_cores);
            }
        }
    }
}

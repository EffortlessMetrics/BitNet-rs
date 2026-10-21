//! OpenCL scatter and gather operations for sparse/indexed tensor access.
//!
//! Provides CPU reference implementations and embedded OpenCL kernel source for:
//!
//! - **Gather** — index-based element selection along a specified axis
//!   (`torch.gather` semantics)
//! - **Scatter** — index-based element placement along a specified axis
//!   (`torch.scatter` semantics)
//! - **Scatter with reduction** — scatter with add, mul, max, min, or mean
//! - **Index select** — select full rows/columns by an index tensor
//! - **Masked fill** — fill positions where a boolean mask is true
//! - **Masked select** — compact elements where a boolean mask is true
//! - **Top-k select** — partial sort returning top-k values with indices
//! - **Stats** — throughput, element count, and bandwidth tracking
//!
//! All operations have pure-Rust CPU fallbacks and do **not** require an
//! OpenCL runtime. The OpenCL kernel source is embedded at compile time for
//! future GPU dispatch on Intel / AMD / other OpenCL-capable devices.

use std::fmt;
use std::time::Instant;

use bitnet_common::{KernelError, Result};

// ── OpenCL kernel source ─────────────────────────────────────────

/// OpenCL kernel source for scatter and gather operations.
pub const SCATTER_GATHER_CL: &str = include_str!("gpu/kernels/scatter_gather.cl");

// ── Reduction mode ───────────────────────────────────────────────

/// Reduction applied when scattering overlapping indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScatterReduce {
    /// Overwrite destination (last-write-wins for duplicates).
    Assign,
    /// Add source to destination.
    Add,
    /// Multiply source into destination.
    Mul,
    /// Keep the maximum of source and destination.
    Max,
    /// Keep the minimum of source and destination.
    Min,
    /// Average all values scattered to the same position.
    Mean,
}

impl ScatterReduce {
    /// Identity element such that `combine(identity, x) == x`.
    pub fn identity(self) -> f32 {
        match self {
            Self::Assign | Self::Add | Self::Mean => 0.0,
            Self::Mul => 1.0,
            Self::Max => f32::NEG_INFINITY,
            Self::Min => f32::INFINITY,
        }
    }

    /// Combine two values according to the reduction.
    #[inline]
    fn combine(self, dst: f32, src: f32) -> f32 {
        match self {
            Self::Assign => src,
            Self::Add | Self::Mean => dst + src,
            Self::Mul => dst * src,
            Self::Max => dst.max(src),
            Self::Min => dst.min(src),
        }
    }
}

impl fmt::Display for ScatterReduce {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Assign => "assign",
            Self::Add => "add",
            Self::Mul => "mul",
            Self::Max => "max",
            Self::Min => "min",
            Self::Mean => "mean",
        };
        f.write_str(s)
    }
}

// ── GatherOp ─────────────────────────────────────────────────────

/// Index-based gather along a specified axis (like `torch.gather`).
///
/// For a 2-D source `[S_rows, S_cols]` and index matrix `[I_rows, I_cols]`:
/// - **axis 0**: `output[i][j] = src[indices[i][j]][j]`
/// - **axis 1**: `output[i][j] = src[i][indices[i][j]]`
#[derive(Debug, Clone)]
pub struct GatherOp {
    /// Source tensor shape `[rows, cols]`.
    pub src_shape: (usize, usize),
    /// Index tensor shape `[rows, cols]`.
    pub idx_shape: (usize, usize),
    /// Axis along which to gather (0 or 1).
    pub axis: usize,
    /// Whether to error on out-of-bounds indices.
    pub bounds_check: bool,
}

impl GatherOp {
    /// Create a new gather operation.
    ///
    /// # Errors
    ///
    /// Returns an error if `axis > 1` or shapes are incompatible.
    pub fn new(
        src_shape: (usize, usize),
        idx_shape: (usize, usize),
        axis: usize,
        bounds_check: bool,
    ) -> Result<Self> {
        if axis > 1 {
            return Err(KernelError::InvalidArguments {
                reason: format!("gather axis must be 0 or 1, got {axis}"),
            }
            .into());
        }
        if axis == 0 && idx_shape.1 != src_shape.1 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "gather axis 0: idx cols ({}) must equal src cols ({})",
                    idx_shape.1, src_shape.1,
                ),
            }
            .into());
        }
        if axis == 1 && idx_shape.0 != src_shape.0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "gather axis 1: idx rows ({}) must equal src rows ({})",
                    idx_shape.0, src_shape.0,
                ),
            }
            .into());
        }
        Ok(Self { src_shape, idx_shape, axis, bounds_check })
    }

    /// Number of output elements.
    pub fn output_len(&self) -> usize {
        self.idx_shape.0 * self.idx_shape.1
    }

    /// Execute gather on CPU.
    pub fn execute(&self, src: &[f32], indices: &[usize], output: &mut [f32]) -> Result<()> {
        let (s_rows, s_cols) = self.src_shape;
        let (i_rows, i_cols) = self.idx_shape;
        let out_len = self.output_len();

        if src.len() < s_rows * s_cols {
            return Err(KernelError::InvalidArguments {
                reason: format!("gather src len {} < {}", src.len(), s_rows * s_cols),
            }
            .into());
        }
        if indices.len() < out_len {
            return Err(KernelError::InvalidArguments {
                reason: format!("gather indices len {} < {}", indices.len(), out_len),
            }
            .into());
        }
        if output.len() < out_len {
            return Err(KernelError::InvalidArguments {
                reason: format!("gather output len {} < {}", output.len(), out_len),
            }
            .into());
        }

        let bound = if self.axis == 0 { s_rows } else { s_cols };

        for i in 0..i_rows {
            for j in 0..i_cols {
                let flat = i * i_cols + j;
                let idx = indices[flat];
                if self.bounds_check && idx >= bound {
                    return Err(KernelError::InvalidArguments {
                        reason: format!(
                            "gather index {idx} out of bounds \
                             (axis {}, size {bound})",
                            self.axis,
                        ),
                    }
                    .into());
                }
                let clamped = idx.min(bound.saturating_sub(1));
                let src_off =
                    if self.axis == 0 { clamped * s_cols + j } else { i * s_cols + clamped };
                output[flat] = src[src_off];
            }
        }
        Ok(())
    }
}

// ── ScatterOp ────────────────────────────────────────────────────

/// Index-based scatter along a specified axis (like `torch.scatter`).
///
/// For a 2-D destination `[D_rows, D_cols]` and source `[I_rows, I_cols]`:
/// - **axis 0**: `dst[indices[i][j]][j] = reduce(dst[…], src[i][j])`
/// - **axis 1**: `dst[i][indices[i][j]] = reduce(dst[…], src[i][j])`
#[derive(Debug, Clone)]
pub struct ScatterOp {
    /// Destination tensor shape `[rows, cols]`.
    pub dst_shape: (usize, usize),
    /// Source/index tensor shape `[rows, cols]`.
    pub idx_shape: (usize, usize),
    /// Axis along which to scatter (0 or 1).
    pub axis: usize,
    /// Reduction mode for overlapping indices.
    pub reduce: ScatterReduce,
    /// Whether to error on out-of-bounds indices.
    pub bounds_check: bool,
}

impl ScatterOp {
    /// Create a new scatter operation.
    pub fn new(
        dst_shape: (usize, usize),
        idx_shape: (usize, usize),
        axis: usize,
        reduce: ScatterReduce,
        bounds_check: bool,
    ) -> Result<Self> {
        if axis > 1 {
            return Err(KernelError::InvalidArguments {
                reason: format!("scatter axis must be 0 or 1, got {axis}"),
            }
            .into());
        }
        if axis == 0 && idx_shape.1 != dst_shape.1 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "scatter axis 0: idx cols ({}) must equal dst cols ({})",
                    idx_shape.1, dst_shape.1,
                ),
            }
            .into());
        }
        if axis == 1 && idx_shape.0 != dst_shape.0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "scatter axis 1: idx rows ({}) must equal dst rows ({})",
                    idx_shape.0, dst_shape.0,
                ),
            }
            .into());
        }
        Ok(Self { dst_shape, idx_shape, axis, reduce, bounds_check })
    }

    /// Execute scatter on CPU.
    ///
    /// For `ScatterReduce::Mean`, callers must pre-fill `dst` with zeros
    /// and provide `counts` to receive per-position scatter counts. After
    /// execution, divide each `dst[i]` by `counts[i]` where `counts[i] > 0`.
    pub fn execute(
        &self,
        src: &[f32],
        indices: &[usize],
        dst: &mut [f32],
        mut counts: Option<&mut [u32]>,
    ) -> Result<()> {
        let (d_rows, d_cols) = self.dst_shape;
        let (i_rows, i_cols) = self.idx_shape;
        let elem_count = i_rows * i_cols;

        if src.len() < elem_count {
            return Err(KernelError::InvalidArguments {
                reason: format!("scatter src len {} < {}", src.len(), elem_count),
            }
            .into());
        }
        if indices.len() < elem_count {
            return Err(KernelError::InvalidArguments {
                reason: format!("scatter indices len {} < {}", indices.len(), elem_count),
            }
            .into());
        }
        if dst.len() < d_rows * d_cols {
            return Err(KernelError::InvalidArguments {
                reason: format!("scatter dst len {} < {}", dst.len(), d_rows * d_cols),
            }
            .into());
        }

        let bound = if self.axis == 0 { d_rows } else { d_cols };

        for i in 0..i_rows {
            for j in 0..i_cols {
                let flat = i * i_cols + j;
                let idx = indices[flat];
                if self.bounds_check && idx >= bound {
                    return Err(KernelError::InvalidArguments {
                        reason: format!(
                            "scatter index {idx} out of bounds \
                             (axis {}, size {bound})",
                            self.axis,
                        ),
                    }
                    .into());
                }
                let clamped = idx.min(bound.saturating_sub(1));
                let dst_off =
                    if self.axis == 0 { clamped * d_cols + j } else { i * d_cols + clamped };
                dst[dst_off] = self.reduce.combine(dst[dst_off], src[flat]);
                if let Some(c) = counts.as_deref_mut()
                    && dst_off < c.len()
                {
                    c[dst_off] += 1;
                }
            }
        }
        Ok(())
    }

    /// Execute scatter with mean reduction, returning the averaged result.
    pub fn execute_mean(&self, src: &[f32], indices: &[usize], dst: &mut [f32]) -> Result<()> {
        let dst_len = self.dst_shape.0 * self.dst_shape.1;
        let mut counts = vec![0u32; dst_len];
        // Ensure we use Add semantics for accumulation.
        let add_op = ScatterOp { reduce: ScatterReduce::Mean, ..self.clone() };
        add_op.execute(src, indices, dst, Some(&mut counts))?;
        for (val, &cnt) in dst.iter_mut().zip(counts.iter()) {
            if cnt > 0 {
                *val /= cnt as f32;
            }
        }
        Ok(())
    }
}

// ── IndexSelect ──────────────────────────────────────────────────

/// Select full rows (or columns) from a 2-D tensor by an index vector.
///
/// Given `src [S_rows, S_cols]` and `indices [N]`, produces
/// `output [N, S_cols]` where `output[i] = src[indices[i]]`.
#[derive(Debug, Clone)]
pub struct IndexSelect {
    /// Source tensor shape `[rows, cols]`.
    pub src_shape: (usize, usize),
    /// Whether to error on out-of-bounds indices.
    pub bounds_check: bool,
}

impl IndexSelect {
    pub fn new(src_shape: (usize, usize), bounds_check: bool) -> Self {
        Self { src_shape, bounds_check }
    }

    /// Execute index selection on CPU.
    pub fn execute(&self, src: &[f32], indices: &[usize], output: &mut [f32]) -> Result<()> {
        let (s_rows, s_cols) = self.src_shape;
        let n = indices.len();
        let out_len = n * s_cols;

        if src.len() < s_rows * s_cols {
            return Err(KernelError::InvalidArguments {
                reason: format!("index_select src len {} < {}", src.len(), s_rows * s_cols),
            }
            .into());
        }
        if output.len() < out_len {
            return Err(KernelError::InvalidArguments {
                reason: format!("index_select output len {} < {}", output.len(), out_len),
            }
            .into());
        }

        for (out_row, &idx) in indices.iter().enumerate() {
            if self.bounds_check && idx >= s_rows {
                return Err(KernelError::InvalidArguments {
                    reason: format!("index_select index {idx} >= rows {s_rows}"),
                }
                .into());
            }
            let clamped = idx.min(s_rows.saturating_sub(1));
            let src_start = clamped * s_cols;
            let dst_start = out_row * s_cols;
            output[dst_start..dst_start + s_cols]
                .copy_from_slice(&src[src_start..src_start + s_cols]);
        }
        Ok(())
    }
}

// ── MaskedFill ───────────────────────────────────────────────────

/// Fill positions where a boolean mask is true with a constant value.
///
/// `output[i] = if mask[i] { fill_value } else { input[i] }`
#[derive(Debug, Clone)]
pub struct MaskedFill {
    /// Value to write where the mask is true.
    pub fill_value: f32,
}

impl MaskedFill {
    pub fn new(fill_value: f32) -> Self {
        Self { fill_value }
    }

    /// Execute masked fill on CPU.
    pub fn execute(&self, input: &[f32], mask: &[bool], output: &mut [f32]) -> Result<()> {
        if input.len() != mask.len() {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "masked_fill: input len {} != mask len {}",
                    input.len(),
                    mask.len(),
                ),
            }
            .into());
        }
        if output.len() < input.len() {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "masked_fill: output len {} < input len {}",
                    output.len(),
                    input.len(),
                ),
            }
            .into());
        }
        for i in 0..input.len() {
            output[i] = if mask[i] { self.fill_value } else { input[i] };
        }
        Ok(())
    }

    /// In-place masked fill.
    pub fn execute_inplace(&self, data: &mut [f32], mask: &[bool]) -> Result<()> {
        if data.len() != mask.len() {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "masked_fill inplace: data len {} != mask len {}",
                    data.len(),
                    mask.len(),
                ),
            }
            .into());
        }
        for i in 0..data.len() {
            if mask[i] {
                data[i] = self.fill_value;
            }
        }
        Ok(())
    }
}

// ── MaskedSelect ─────────────────────────────────────────────────

/// Compact elements from `input` where `mask[i]` is true.
///
/// Returns a dense vector of selected elements.
#[derive(Debug, Clone)]
pub struct MaskedSelect;

impl MaskedSelect {
    /// Execute masked select on CPU, returning compacted elements.
    pub fn execute(input: &[f32], mask: &[bool]) -> Result<Vec<f32>> {
        if input.len() != mask.len() {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "masked_select: input len {} != mask len {}",
                    input.len(),
                    mask.len(),
                ),
            }
            .into());
        }
        let result: Vec<f32> = input
            .iter()
            .zip(mask.iter())
            .filter_map(|(&v, &m)| if m { Some(v) } else { None })
            .collect();
        Ok(result)
    }

    /// Count how many elements would be selected.
    pub fn count_selected(mask: &[bool]) -> usize {
        mask.iter().filter(|&&m| m).count()
    }

    /// Execute masked select into a pre-allocated output buffer.
    pub fn execute_into(input: &[f32], mask: &[bool], output: &mut [f32]) -> Result<usize> {
        if input.len() != mask.len() {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "masked_select: input len {} != mask len {}",
                    input.len(),
                    mask.len(),
                ),
            }
            .into());
        }
        let mut pos = 0;
        for i in 0..input.len() {
            if mask[i] {
                if pos >= output.len() {
                    return Err(KernelError::InvalidArguments {
                        reason: format!(
                            "masked_select: output buffer too small \
                             ({} < needed)",
                            output.len(),
                        ),
                    }
                    .into());
                }
                output[pos] = input[i];
                pos += 1;
            }
        }
        Ok(pos)
    }
}

// ── TopKSelect ───────────────────────────────────────────────────

/// Result of a top-k selection: values and their original indices.
#[derive(Debug, Clone, PartialEq)]
pub struct TopKResult {
    /// Top-k values in descending order.
    pub values: Vec<f32>,
    /// Original indices of the top-k values.
    pub indices: Vec<usize>,
}

/// Select the top-k largest (or smallest) values with their indices.
///
/// Uses a partial sort (selection) algorithm — O(n + k log k).
#[derive(Debug, Clone)]
pub struct TopKSelect {
    /// Number of elements to select.
    pub k: usize,
    /// If true, select the k *largest*; if false, the k *smallest*.
    pub largest: bool,
    /// If true, the returned values are sorted.
    pub sorted: bool,
}

impl TopKSelect {
    pub fn new(k: usize, largest: bool, sorted: bool) -> Self {
        Self { k, largest, sorted }
    }

    /// Execute top-k selection on CPU.
    pub fn execute(&self, input: &[f32]) -> Result<TopKResult> {
        if self.k == 0 {
            return Ok(TopKResult { values: vec![], indices: vec![] });
        }
        if self.k > input.len() {
            return Err(KernelError::InvalidArguments {
                reason: format!("topk: k ({}) > input len ({})", self.k, input.len()),
            }
            .into());
        }

        // Build (value, index) pairs.
        let mut pairs: Vec<(f32, usize)> =
            input.iter().copied().enumerate().map(|(i, v)| (v, i)).collect();

        // Partial sort: move top-k to front.
        if self.largest {
            pairs.select_nth_unstable_by(self.k - 1, |a, b| {
                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            pairs.select_nth_unstable_by(self.k - 1, |a, b| {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        let mut selected: Vec<(f32, usize)> = pairs[..self.k].to_vec();

        if self.sorted {
            if self.largest {
                selected.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            } else {
                selected.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            }
        }

        let values: Vec<f32> = selected.iter().map(|p| p.0).collect();
        let indices: Vec<usize> = selected.iter().map(|p| p.1).collect();
        Ok(TopKResult { values, indices })
    }

    /// Execute top-k per row of a 2-D tensor `[rows, cols]`.
    pub fn execute_2d(&self, input: &[f32], rows: usize, cols: usize) -> Result<Vec<TopKResult>> {
        if input.len() < rows * cols {
            return Err(KernelError::InvalidArguments {
                reason: format!("topk_2d: input len {} < rows*cols {}", input.len(), rows * cols),
            }
            .into());
        }
        let mut results = Vec::with_capacity(rows);
        for r in 0..rows {
            let row_start = r * cols;
            let row = &input[row_start..row_start + cols];
            results.push(self.execute(row)?);
        }
        Ok(results)
    }
}

// ── ScatterGatherStats ───────────────────────────────────────────

/// Throughput and bandwidth statistics for scatter/gather operations.
#[derive(Debug, Clone)]
pub struct ScatterGatherStats {
    /// Total elements processed.
    pub element_count: u64,
    /// Wall-clock duration.
    pub elapsed: std::time::Duration,
    /// Bytes read from source.
    pub bytes_read: u64,
    /// Bytes written to destination.
    pub bytes_written: u64,
}

impl ScatterGatherStats {
    /// Create stats from a timed operation.
    pub fn new(
        element_count: u64,
        elapsed: std::time::Duration,
        bytes_read: u64,
        bytes_written: u64,
    ) -> Self {
        Self { element_count, elapsed, bytes_read, bytes_written }
    }

    /// Elements per second.
    pub fn throughput(&self) -> f64 {
        let secs = self.elapsed.as_secs_f64();
        if secs > 0.0 { self.element_count as f64 / secs } else { 0.0 }
    }

    /// Effective bandwidth in bytes per second (read + write).
    pub fn bandwidth_bytes_per_sec(&self) -> f64 {
        let secs = self.elapsed.as_secs_f64();
        if secs > 0.0 { (self.bytes_read + self.bytes_written) as f64 / secs } else { 0.0 }
    }

    /// Effective bandwidth in GiB/s.
    pub fn bandwidth_gib_per_sec(&self) -> f64 {
        self.bandwidth_bytes_per_sec() / (1024.0 * 1024.0 * 1024.0)
    }
}

impl fmt::Display for ScatterGatherStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} elems in {:.3}ms ({:.2} Melem/s, {:.2} GiB/s)",
            self.element_count,
            self.elapsed.as_secs_f64() * 1000.0,
            self.throughput() / 1e6,
            self.bandwidth_gib_per_sec(),
        )
    }
}

// ── Convenience: timed gather ────────────────────────────────────

/// Execute a gather and return stats.
pub fn gather_timed(
    op: &GatherOp,
    src: &[f32],
    indices: &[usize],
    output: &mut [f32],
) -> Result<ScatterGatherStats> {
    let start = Instant::now();
    op.execute(src, indices, output)?;
    let elapsed = start.elapsed();
    let n = op.output_len() as u64;
    let bytes_r = n * 4; // f32 reads
    let bytes_w = n * 4; // f32 writes
    Ok(ScatterGatherStats::new(n, elapsed, bytes_r, bytes_w))
}

/// Execute a scatter and return stats.
pub fn scatter_timed(
    op: &ScatterOp,
    src: &[f32],
    indices: &[usize],
    dst: &mut [f32],
) -> Result<ScatterGatherStats> {
    let start = Instant::now();
    op.execute(src, indices, dst, None)?;
    let elapsed = start.elapsed();
    let n = (op.idx_shape.0 * op.idx_shape.1) as u64;
    let bytes_r = n * 4;
    let bytes_w = n * 4;
    Ok(ScatterGatherStats::new(n, elapsed, bytes_r, bytes_w))
}

// ══════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── OpenCL source ────────────────────────────────────────────

    #[test]
    fn kernel_source_is_non_empty() {
        assert!(!SCATTER_GATHER_CL.is_empty(), "OpenCL source must be embedded");
    }

    #[test]
    fn kernel_source_contains_gather() {
        assert!(SCATTER_GATHER_CL.contains("gather_axis0"));
        assert!(SCATTER_GATHER_CL.contains("gather_axis1"));
    }

    #[test]
    fn kernel_source_contains_scatter() {
        assert!(SCATTER_GATHER_CL.contains("scatter_assign"));
        assert!(SCATTER_GATHER_CL.contains("scatter_add"));
    }

    #[test]
    fn kernel_source_contains_index_select() {
        assert!(SCATTER_GATHER_CL.contains("index_select_kernel"));
    }

    #[test]
    fn kernel_source_contains_masked_fill() {
        assert!(SCATTER_GATHER_CL.contains("masked_fill_kernel"));
    }

    // ── ScatterReduce ────────────────────────────────────────────

    #[test]
    fn reduce_identity_values() {
        assert_eq!(ScatterReduce::Assign.identity(), 0.0);
        assert_eq!(ScatterReduce::Add.identity(), 0.0);
        assert_eq!(ScatterReduce::Mul.identity(), 1.0);
        assert_eq!(ScatterReduce::Max.identity(), f32::NEG_INFINITY);
        assert_eq!(ScatterReduce::Min.identity(), f32::INFINITY);
        assert_eq!(ScatterReduce::Mean.identity(), 0.0);
    }

    #[test]
    fn reduce_combine_assign() {
        assert_eq!(ScatterReduce::Assign.combine(99.0, 5.0), 5.0);
    }

    #[test]
    fn reduce_combine_add() {
        assert_eq!(ScatterReduce::Add.combine(10.0, 3.0), 13.0);
    }

    #[test]
    fn reduce_combine_mul() {
        assert_eq!(ScatterReduce::Mul.combine(4.0, 3.0), 12.0);
    }

    #[test]
    fn reduce_combine_max() {
        assert_eq!(ScatterReduce::Max.combine(4.0, 7.0), 7.0);
        assert_eq!(ScatterReduce::Max.combine(7.0, 4.0), 7.0);
    }

    #[test]
    fn reduce_combine_min() {
        assert_eq!(ScatterReduce::Min.combine(4.0, 7.0), 4.0);
        assert_eq!(ScatterReduce::Min.combine(7.0, 4.0), 4.0);
    }

    #[test]
    fn reduce_display() {
        assert_eq!(format!("{}", ScatterReduce::Add), "add");
        assert_eq!(format!("{}", ScatterReduce::Mean), "mean");
    }

    // ── GatherOp construction ────────────────────────────────────

    #[test]
    fn gather_op_new_axis0() {
        let op = GatherOp::new((4, 3), (2, 3), 0, true).unwrap();
        assert_eq!(op.output_len(), 6);
    }

    #[test]
    fn gather_op_new_axis1() {
        let op = GatherOp::new((4, 3), (4, 2), 1, true).unwrap();
        assert_eq!(op.output_len(), 8);
    }

    #[test]
    fn gather_op_rejects_axis2() {
        assert!(GatherOp::new((4, 3), (2, 3), 2, true).is_err());
    }

    #[test]
    fn gather_op_rejects_shape_mismatch_axis0() {
        // axis 0: idx cols must equal src cols
        assert!(GatherOp::new((4, 3), (2, 5), 0, true).is_err());
    }

    #[test]
    fn gather_op_rejects_shape_mismatch_axis1() {
        // axis 1: idx rows must equal src rows
        assert!(GatherOp::new((4, 3), (2, 2), 1, true).is_err());
    }

    // ── GatherOp execution ───────────────────────────────────────

    #[test]
    fn gather_axis0_basic() {
        // src 3×2: [[10,11],[20,21],[30,31]]
        let src = [10.0, 11.0, 20.0, 21.0, 30.0, 31.0];
        let indices = [2, 0]; // 1×2
        let op = GatherOp::new((3, 2), (1, 2), 0, true).unwrap();
        let mut out = [0.0f32; 2];
        op.execute(&src, &indices, &mut out).unwrap();
        assert_eq!(out, [30.0, 11.0]);
    }

    #[test]
    fn gather_axis1_basic() {
        // src 2×4: [[0,1,2,3],[4,5,6,7]]
        let src: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let indices = [3, 1, 0, 2]; // 2×2
        let op = GatherOp::new((2, 4), (2, 2), 1, true).unwrap();
        let mut out = [0.0f32; 4];
        op.execute(&src, &indices, &mut out).unwrap();
        assert_eq!(out, [3.0, 1.0, 4.0, 6.0]);
    }

    #[test]
    fn gather_single_element() {
        let src = [42.0];
        let op = GatherOp::new((1, 1), (1, 1), 0, true).unwrap();
        let mut out = [0.0f32; 1];
        op.execute(&src, &[0], &mut out).unwrap();
        assert_eq!(out[0], 42.0);
    }

    #[test]
    fn gather_oob_with_bounds_check() {
        let src = [1.0, 2.0, 3.0, 4.0]; // 2×2
        let indices = [5, 0]; // 5 is OOB for 2 rows
        let op = GatherOp::new((2, 2), (1, 2), 0, true).unwrap();
        let mut out = [0.0f32; 2];
        assert!(op.execute(&src, &indices, &mut out).is_err());
    }

    #[test]
    fn gather_oob_without_bounds_check_clamps() {
        let src = [1.0, 2.0, 3.0, 4.0]; // 2×2
        let indices = [5, 0];
        let op = GatherOp::new((2, 2), (1, 2), 0, false).unwrap();
        let mut out = [0.0f32; 2];
        op.execute(&src, &indices, &mut out).unwrap();
        // clamped to row 1 → src[1*2+0]=3.0
        assert_eq!(out[0], 3.0);
        assert_eq!(out[1], 2.0);
    }

    #[test]
    fn gather_large_tensor() {
        let rows = 100;
        let cols = 64;
        let src: Vec<f32> = (0..(rows * cols) as u32).map(|x| x as f32).collect();
        let n_sel = 50;
        let indices: Vec<usize> = (0..n_sel).flat_map(|i| vec![i * 2; cols]).collect();
        let op = GatherOp::new((rows, cols), (n_sel, cols), 0, true).unwrap();
        let mut out = vec![0.0f32; n_sel * cols];
        op.execute(&src, &indices, &mut out).unwrap();
        // First row should be src row 0
        for j in 0..cols {
            assert_eq!(out[j], src[j]);
        }
        // Second row should be src row 2
        for j in 0..cols {
            assert_eq!(out[cols + j], src[2 * cols + j]);
        }
    }

    #[test]
    fn gather_all_same_index() {
        let src = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]; // 3×2
        let indices = [1, 1, 1, 1]; // 2×2, all point to row 1
        let op = GatherOp::new((3, 2), (2, 2), 0, true).unwrap();
        let mut out = [0.0f32; 4];
        op.execute(&src, &indices, &mut out).unwrap();
        assert_eq!(out, [30.0, 40.0, 30.0, 40.0]);
    }

    #[test]
    fn gather_src_too_small_error() {
        let src = [1.0]; // too small for (2,2)
        let op = GatherOp::new((2, 2), (1, 2), 0, false).unwrap();
        let mut out = [0.0f32; 2];
        assert!(op.execute(&src, &[0, 0], &mut out).is_err());
    }

    #[test]
    fn gather_indices_too_small_error() {
        let src = [1.0, 2.0, 3.0, 4.0];
        let op = GatherOp::new((2, 2), (1, 2), 0, false).unwrap();
        let mut out = [0.0f32; 2];
        assert!(op.execute(&src, &[0], &mut out).is_err());
    }

    #[test]
    fn gather_output_too_small_error() {
        let src = [1.0, 2.0, 3.0, 4.0];
        let op = GatherOp::new((2, 2), (1, 2), 0, false).unwrap();
        let mut out = [0.0f32; 1];
        assert!(op.execute(&src, &[0, 0], &mut out).is_err());
    }

    // ── ScatterOp construction ───────────────────────────────────

    #[test]
    fn scatter_op_rejects_axis2() {
        assert!(ScatterOp::new((3, 2), (1, 2), 2, ScatterReduce::Assign, true).is_err());
    }

    #[test]
    fn scatter_op_rejects_shape_mismatch_axis0() {
        assert!(ScatterOp::new((3, 2), (1, 3), 0, ScatterReduce::Assign, true).is_err());
    }

    #[test]
    fn scatter_op_rejects_shape_mismatch_axis1() {
        assert!(ScatterOp::new((3, 2), (4, 1), 1, ScatterReduce::Assign, true).is_err());
    }

    // ── ScatterOp execution ──────────────────────────────────────

    #[test]
    fn scatter_assign_axis0() {
        let src = [10.0, 11.0]; // 1×2
        let indices = [2, 2];
        let op = ScatterOp::new((3, 2), (1, 2), 0, ScatterReduce::Assign, true).unwrap();
        let mut dst = [0.0f32; 6];
        op.execute(&src, &indices, &mut dst, None).unwrap();
        assert_eq!(dst, [0.0, 0.0, 0.0, 0.0, 10.0, 11.0]);
    }

    #[test]
    fn scatter_add_accumulates() {
        let src = [1.0, 2.0, 3.0, 4.0]; // 2×2
        let indices = [0, 0, 0, 0]; // all target row 0
        let op = ScatterOp::new((2, 2), (2, 2), 0, ScatterReduce::Add, true).unwrap();
        let mut dst = [0.0f32; 4];
        op.execute(&src, &indices, &mut dst, None).unwrap();
        assert_eq!(dst[0], 4.0); // 1 + 3
        assert_eq!(dst[1], 6.0); // 2 + 4
    }

    #[test]
    fn scatter_mul_accumulates() {
        let src = [2.0, 3.0, 4.0, 5.0]; // 2×2
        let indices = [0, 0, 0, 0]; // all target row 0
        let op = ScatterOp::new((2, 2), (2, 2), 0, ScatterReduce::Mul, true).unwrap();
        let mut dst = [1.0f32; 4]; // identity for mul
        op.execute(&src, &indices, &mut dst, None).unwrap();
        assert_eq!(dst[0], 8.0); // 2 * 4
        assert_eq!(dst[1], 15.0); // 3 * 5
    }

    #[test]
    fn scatter_max_keeps_max() {
        let src = [5.0, 1.0, 3.0, 9.0]; // 2×2
        let indices = [0, 0, 0, 0];
        let op = ScatterOp::new((2, 2), (2, 2), 0, ScatterReduce::Max, true).unwrap();
        let mut dst = [f32::NEG_INFINITY; 4];
        op.execute(&src, &indices, &mut dst, None).unwrap();
        assert_eq!(dst[0], 5.0);
        assert_eq!(dst[1], 9.0);
    }

    #[test]
    fn scatter_min_keeps_min() {
        let src = [5.0, 1.0, 3.0, 9.0]; // 2×2
        let indices = [0, 0, 0, 0];
        let op = ScatterOp::new((2, 2), (2, 2), 0, ScatterReduce::Min, true).unwrap();
        let mut dst = [f32::INFINITY; 4];
        op.execute(&src, &indices, &mut dst, None).unwrap();
        assert_eq!(dst[0], 3.0);
        assert_eq!(dst[1], 1.0);
    }

    #[test]
    fn scatter_axis1() {
        let src = [10.0, 20.0]; // 2×1
        let indices = [2, 0]; // row0→col2, row1→col0
        let op = ScatterOp::new((2, 3), (2, 1), 1, ScatterReduce::Assign, true).unwrap();
        let mut dst = [0.0f32; 6];
        op.execute(&src, &indices, &mut dst, None).unwrap();
        assert_eq!(dst, [0.0, 0.0, 10.0, 20.0, 0.0, 0.0]);
    }

    #[test]
    fn scatter_oob_with_bounds_check() {
        let src = [1.0, 2.0];
        let indices = [99, 0];
        let op = ScatterOp::new((2, 2), (1, 2), 0, ScatterReduce::Assign, true).unwrap();
        let mut dst = [0.0f32; 4];
        assert!(op.execute(&src, &indices, &mut dst, None).is_err());
    }

    #[test]
    fn scatter_src_too_small_error() {
        let src = [1.0];
        let op = ScatterOp::new((2, 2), (1, 2), 0, ScatterReduce::Assign, false).unwrap();
        let mut dst = [0.0f32; 4];
        assert!(op.execute(&src, &[0, 0], &mut dst, None).is_err());
    }

    #[test]
    fn scatter_dst_too_small_error() {
        let src = [1.0, 2.0];
        let op = ScatterOp::new((2, 2), (1, 2), 0, ScatterReduce::Assign, false).unwrap();
        let mut dst = [0.0f32; 2]; // need 4
        assert!(op.execute(&src, &[0, 0], &mut dst, None).is_err());
    }

    #[test]
    fn scatter_mean_basic() {
        // Scatter values [10, 20, 30] to row 0 → mean = 20
        let src = [10.0, 20.0, 30.0]; // 3×1
        let indices = [0, 0, 0]; // all target row 0
        let op = ScatterOp::new((2, 1), (3, 1), 0, ScatterReduce::Mean, true).unwrap();
        let mut dst = [0.0f32; 2];
        op.execute_mean(&src, &indices, &mut dst).unwrap();
        assert!((dst[0] - 20.0).abs() < 1e-6);
    }

    #[test]
    fn scatter_with_counts() {
        let src = [1.0, 2.0, 3.0]; // 3×1
        let indices = [0, 0, 1];
        let op = ScatterOp::new((2, 1), (3, 1), 0, ScatterReduce::Add, true).unwrap();
        let mut dst = [0.0f32; 2];
        let mut counts = [0u32; 2];
        op.execute(&src, &indices, &mut dst, Some(&mut counts)).unwrap();
        assert_eq!(dst[0], 3.0); // 1 + 2
        assert_eq!(dst[1], 3.0);
        assert_eq!(counts[0], 2);
        assert_eq!(counts[1], 1);
    }

    // ── Gather/scatter roundtrip ─────────────────────────────────

    #[test]
    fn gather_scatter_roundtrip_identity() {
        // If indices are a permutation, gather→scatter should recover
        // the original (for axis 0 with unique row indices).
        let src = [10.0, 11.0, 20.0, 21.0, 30.0, 31.0]; // 3×2
        let perm = [2, 0, 1]; // full permutation of rows
        let cols = 2;

        // Gather: select rows in permuted order
        let gather_indices: Vec<usize> = perm.iter().flat_map(|&r| vec![r; cols]).collect();
        let gather_op = GatherOp::new((3, cols), (3, cols), 0, true).unwrap();
        let mut gathered = [0.0f32; 6];
        gather_op.execute(&src, &gather_indices, &mut gathered).unwrap();

        // Scatter back: gathered[i] came from src[perm[i]],
        // so scatter gathered[i] to position perm[i].
        let scatter_indices: Vec<usize> = perm.iter().flat_map(|&r| vec![r; cols]).collect();
        let scatter_op =
            ScatterOp::new((3, cols), (3, cols), 0, ScatterReduce::Assign, true).unwrap();
        let mut result = [0.0f32; 6];
        scatter_op.execute(&gathered, &scatter_indices, &mut result, None).unwrap();

        assert_eq!(result, src);
    }

    #[test]
    fn gather_scatter_roundtrip_axis1() {
        // Permute columns.
        let src = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2×3
        let rows = 2;
        let cols = 3;
        let col_perm = [2, 0, 1]; // column permutation

        // Gather axis 1
        let gather_indices: Vec<usize> = (0..rows).flat_map(|_| col_perm.iter().copied()).collect();
        let gather_op = GatherOp::new((rows, cols), (rows, cols), 1, true).unwrap();
        let mut gathered = [0.0f32; 6];
        gather_op.execute(&src, &gather_indices, &mut gathered).unwrap();

        // Scatter back: gathered[i][j] came from src[i][col_perm[j]],
        // so scatter gathered[i][j] to column col_perm[j].
        let scatter_indices: Vec<usize> =
            (0..rows).flat_map(|_| col_perm.iter().copied()).collect();
        let scatter_op =
            ScatterOp::new((rows, cols), (rows, cols), 1, ScatterReduce::Assign, true).unwrap();
        let mut result = [0.0f32; 6];
        scatter_op.execute(&gathered, &scatter_indices, &mut result, None).unwrap();

        assert_eq!(result, src);
    }

    // ── IndexSelect ──────────────────────────────────────────────

    #[test]
    fn index_select_basic() {
        let src: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let sel = IndexSelect::new((4, 3), true);
        let indices = [3, 1];
        let mut out = [0.0f32; 6];
        sel.execute(&src, &indices, &mut out).unwrap();
        assert_eq!(out, [9.0, 10.0, 11.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn index_select_single_row() {
        let src = [1.0, 2.0, 3.0];
        let sel = IndexSelect::new((1, 3), true);
        let mut out = [0.0f32; 3];
        sel.execute(&src, &[0], &mut out).unwrap();
        assert_eq!(out, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn index_select_duplicate_indices() {
        let src = [10.0, 20.0, 30.0, 40.0]; // 2×2
        let sel = IndexSelect::new((2, 2), true);
        let indices = [1, 1, 0];
        let mut out = [0.0f32; 6];
        sel.execute(&src, &indices, &mut out).unwrap();
        assert_eq!(out, [30.0, 40.0, 30.0, 40.0, 10.0, 20.0]);
    }

    #[test]
    fn index_select_oob_error() {
        let src = [1.0, 2.0];
        let sel = IndexSelect::new((1, 2), true);
        let mut out = [0.0f32; 2];
        assert!(sel.execute(&src, &[5], &mut out).is_err());
    }

    #[test]
    fn index_select_empty_indices() {
        let src = [1.0, 2.0, 3.0, 4.0];
        let sel = IndexSelect::new((2, 2), true);
        let indices: &[usize] = &[];
        let mut out: Vec<f32> = vec![];
        sel.execute(&src, indices, &mut out).unwrap();
    }

    #[test]
    fn index_select_src_too_small() {
        let src = [1.0]; // not enough for (2,2)
        let sel = IndexSelect::new((2, 2), true);
        let mut out = [0.0f32; 2];
        assert!(sel.execute(&src, &[0], &mut out).is_err());
    }

    #[test]
    fn index_select_output_too_small() {
        let src = [1.0, 2.0, 3.0, 4.0];
        let sel = IndexSelect::new((2, 2), true);
        let mut out = [0.0f32; 1]; // need 2
        assert!(sel.execute(&src, &[0], &mut out).is_err());
    }

    // ── MaskedFill ───────────────────────────────────────────────

    #[test]
    fn masked_fill_basic() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mask = [false, true, false, true, false];
        let filler = MaskedFill::new(-999.0);
        let mut out = [0.0f32; 5];
        filler.execute(&input, &mask, &mut out).unwrap();
        assert_eq!(out, [1.0, -999.0, 3.0, -999.0, 5.0]);
    }

    #[test]
    fn masked_fill_all_true() {
        let input = [1.0, 2.0, 3.0];
        let mask = [true, true, true];
        let filler = MaskedFill::new(0.0);
        let mut out = [0.0f32; 3];
        filler.execute(&input, &mask, &mut out).unwrap();
        assert_eq!(out, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn masked_fill_all_false() {
        let input = [1.0, 2.0, 3.0];
        let mask = [false, false, false];
        let filler = MaskedFill::new(0.0);
        let mut out = [0.0f32; 3];
        filler.execute(&input, &mask, &mut out).unwrap();
        assert_eq!(out, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn masked_fill_empty() {
        let input: &[f32] = &[];
        let mask: &[bool] = &[];
        let filler = MaskedFill::new(42.0);
        let mut out: Vec<f32> = vec![];
        filler.execute(input, mask, &mut out).unwrap();
    }

    #[test]
    fn masked_fill_length_mismatch() {
        let input = [1.0, 2.0];
        let mask = [true];
        let filler = MaskedFill::new(0.0);
        let mut out = [0.0f32; 2];
        assert!(filler.execute(&input, &mask, &mut out).is_err());
    }

    #[test]
    fn masked_fill_output_too_small() {
        let input = [1.0, 2.0];
        let mask = [true, false];
        let filler = MaskedFill::new(0.0);
        let mut out = [0.0f32; 1];
        assert!(filler.execute(&input, &mask, &mut out).is_err());
    }

    #[test]
    fn masked_fill_inplace() {
        let mut data = [1.0, 2.0, 3.0, 4.0];
        let mask = [false, true, true, false];
        let filler = MaskedFill::new(-1.0);
        filler.execute_inplace(&mut data, &mask).unwrap();
        assert_eq!(data, [1.0, -1.0, -1.0, 4.0]);
    }

    #[test]
    fn masked_fill_inplace_mismatch() {
        let mut data = [1.0, 2.0];
        let mask = [true, false, true];
        let filler = MaskedFill::new(0.0);
        assert!(filler.execute_inplace(&mut data, &mask).is_err());
    }

    #[test]
    fn masked_fill_neg_infinity() {
        let input = [1.0, 2.0, 3.0];
        let mask = [true, false, true];
        let filler = MaskedFill::new(f32::NEG_INFINITY);
        let mut out = [0.0f32; 3];
        filler.execute(&input, &mask, &mut out).unwrap();
        assert_eq!(out[0], f32::NEG_INFINITY);
        assert_eq!(out[1], 2.0);
        assert_eq!(out[2], f32::NEG_INFINITY);
    }

    // ── MaskedSelect ─────────────────────────────────────────────

    #[test]
    fn masked_select_basic() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mask = [true, false, true, false, true];
        let result = MaskedSelect::execute(&input, &mask).unwrap();
        assert_eq!(result, [1.0, 3.0, 5.0]);
    }

    #[test]
    fn masked_select_all_true() {
        let input = [10.0, 20.0, 30.0];
        let mask = [true, true, true];
        let result = MaskedSelect::execute(&input, &mask).unwrap();
        assert_eq!(result, [10.0, 20.0, 30.0]);
    }

    #[test]
    fn masked_select_all_false() {
        let input = [10.0, 20.0, 30.0];
        let mask = [false, false, false];
        let result = MaskedSelect::execute(&input, &mask).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn masked_select_empty() {
        let result = MaskedSelect::execute(&[], &[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn masked_select_length_mismatch() {
        assert!(MaskedSelect::execute(&[1.0, 2.0], &[true]).is_err());
    }

    #[test]
    fn masked_select_count() {
        let mask = [true, false, true, true, false];
        assert_eq!(MaskedSelect::count_selected(&mask), 3);
    }

    #[test]
    fn masked_select_into_buffer() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mask = [false, true, false, true];
        let mut out = [0.0f32; 2];
        let n = MaskedSelect::execute_into(&input, &mask, &mut out).unwrap();
        assert_eq!(n, 2);
        assert_eq!(out, [2.0, 4.0]);
    }

    #[test]
    fn masked_select_into_buffer_too_small() {
        let input = [1.0, 2.0, 3.0];
        let mask = [true, true, true];
        let mut out = [0.0f32; 1]; // too small
        assert!(MaskedSelect::execute_into(&input, &mask, &mut out).is_err());
    }

    // ── TopKSelect ───────────────────────────────────────────────

    #[test]
    fn topk_basic_largest() {
        let input = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let sel = TopKSelect::new(3, true, true);
        let result = sel.execute(&input).unwrap();
        assert_eq!(result.values, [9.0, 6.0, 5.0]);
        // Verify indices point to correct values.
        for (&v, &i) in result.values.iter().zip(result.indices.iter()) {
            assert_eq!(v, input[i]);
        }
    }

    #[test]
    fn topk_basic_smallest() {
        let input = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let sel = TopKSelect::new(3, false, true);
        let result = sel.execute(&input).unwrap();
        assert_eq!(result.values, [1.0, 1.0, 2.0]);
        for (&v, &i) in result.values.iter().zip(result.indices.iter()) {
            assert_eq!(v, input[i]);
        }
    }

    #[test]
    fn topk_k_equals_len() {
        let input = [5.0, 3.0, 1.0];
        let sel = TopKSelect::new(3, true, true);
        let result = sel.execute(&input).unwrap();
        assert_eq!(result.values, [5.0, 3.0, 1.0]);
    }

    #[test]
    fn topk_k_equals_1() {
        let input = [10.0, 20.0, 5.0];
        let sel = TopKSelect::new(1, true, true);
        let result = sel.execute(&input).unwrap();
        assert_eq!(result.values, [20.0]);
        assert_eq!(result.indices, [1]);
    }

    #[test]
    fn topk_k_zero() {
        let sel = TopKSelect::new(0, true, true);
        let result = sel.execute(&[1.0, 2.0, 3.0]).unwrap();
        assert!(result.values.is_empty());
        assert!(result.indices.is_empty());
    }

    #[test]
    fn topk_k_exceeds_len() {
        let sel = TopKSelect::new(5, true, true);
        assert!(sel.execute(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn topk_sorted_descending() {
        let input = [1.0, 5.0, 3.0, 7.0, 2.0];
        let sel = TopKSelect::new(4, true, true);
        let result = sel.execute(&input).unwrap();
        // Must be sorted descending.
        for i in 1..result.values.len() {
            assert!(result.values[i - 1] >= result.values[i]);
        }
    }

    #[test]
    fn topk_sorted_ascending() {
        let input = [1.0, 5.0, 3.0, 7.0, 2.0];
        let sel = TopKSelect::new(4, false, true);
        let result = sel.execute(&input).unwrap();
        // Must be sorted ascending.
        for i in 1..result.values.len() {
            assert!(result.values[i - 1] <= result.values[i]);
        }
    }

    #[test]
    fn topk_unsorted() {
        let input = [1.0, 5.0, 3.0, 7.0, 2.0];
        let sel = TopKSelect::new(3, true, false);
        let result = sel.execute(&input).unwrap();
        assert_eq!(result.values.len(), 3);
        // All values should come from the top-3 {5, 7, 3} set.
        let mut sorted = result.values.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        assert_eq!(sorted, [7.0, 5.0, 3.0]);
    }

    #[test]
    fn topk_indices_are_valid() {
        let input = [100.0, 200.0, 50.0, 300.0, 150.0];
        let sel = TopKSelect::new(3, true, true);
        let result = sel.execute(&input).unwrap();
        for &i in &result.indices {
            assert!(i < input.len());
        }
    }

    #[test]
    fn topk_duplicate_values() {
        let input = [5.0, 5.0, 5.0, 5.0];
        let sel = TopKSelect::new(2, true, true);
        let result = sel.execute(&input).unwrap();
        assert_eq!(result.values, [5.0, 5.0]);
    }

    #[test]
    fn topk_negative_values() {
        let input = [-3.0, -1.0, -4.0, -1.0, -5.0];
        let sel = TopKSelect::new(2, true, true);
        let result = sel.execute(&input).unwrap();
        assert_eq!(result.values, [-1.0, -1.0]);
    }

    #[test]
    fn topk_with_nan_does_not_panic() {
        let input = [1.0, f32::NAN, 3.0, 2.0];
        let sel = TopKSelect::new(2, true, true);
        // Should not panic; NaN ordering is implementation-defined.
        let _ = sel.execute(&input);
    }

    #[test]
    fn topk_2d_basic() {
        let input = [
            3.0, 1.0, 4.0, 1.0, // row 0
            5.0, 9.0, 2.0, 6.0, // row 1
        ];
        let sel = TopKSelect::new(2, true, true);
        let results = sel.execute_2d(&input, 2, 4).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].values, [4.0, 3.0]);
        assert_eq!(results[1].values, [9.0, 6.0]);
    }

    #[test]
    fn topk_2d_input_too_small() {
        let sel = TopKSelect::new(2, true, true);
        assert!(sel.execute_2d(&[1.0], 2, 4).is_err());
    }

    // ── ScatterGatherStats ───────────────────────────────────────

    #[test]
    fn stats_throughput() {
        let stats = ScatterGatherStats::new(1000, std::time::Duration::from_secs(1), 4000, 4000);
        assert!((stats.throughput() - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn stats_bandwidth() {
        let stats = ScatterGatherStats::new(1000, std::time::Duration::from_secs(1), 4000, 4000);
        assert!((stats.bandwidth_bytes_per_sec() - 8000.0).abs() < 1e-6);
    }

    #[test]
    fn stats_zero_duration() {
        let stats = ScatterGatherStats::new(1000, std::time::Duration::ZERO, 4000, 4000);
        assert_eq!(stats.throughput(), 0.0);
        assert_eq!(stats.bandwidth_bytes_per_sec(), 0.0);
    }

    #[test]
    fn stats_display() {
        let stats =
            ScatterGatherStats::new(1000, std::time::Duration::from_millis(100), 4000, 4000);
        let s = format!("{stats}");
        assert!(s.contains("1000 elems"));
        assert!(s.contains("GiB/s"));
    }

    #[test]
    fn stats_gib_per_sec() {
        let gib = 1024.0 * 1024.0 * 1024.0;
        let stats = ScatterGatherStats::new(1, std::time::Duration::from_secs(1), gib as u64, 0);
        assert!((stats.bandwidth_gib_per_sec() - 1.0).abs() < 1e-6);
    }

    // ── Timed helpers ────────────────────────────────────────────

    #[test]
    fn gather_timed_returns_stats() {
        let src = [10.0, 20.0, 30.0, 40.0]; // 2×2
        let op = GatherOp::new((2, 2), (1, 2), 0, true).unwrap();
        let mut out = [0.0f32; 2];
        let stats = gather_timed(&op, &src, &[1, 0], &mut out).unwrap();
        assert_eq!(stats.element_count, 2);
        assert!(stats.elapsed.as_nanos() > 0);
        assert_eq!(out, [30.0, 20.0]);
    }

    #[test]
    fn scatter_timed_returns_stats() {
        let src = [5.0, 6.0];
        let op = ScatterOp::new((2, 2), (1, 2), 0, ScatterReduce::Assign, true).unwrap();
        let mut dst = [0.0f32; 4];
        let stats = scatter_timed(&op, &src, &[1, 1], &mut dst).unwrap();
        assert_eq!(stats.element_count, 2);
        assert_eq!(dst, [0.0, 0.0, 5.0, 6.0]);
    }

    // ── Property-style tests ─────────────────────────────────────

    #[test]
    fn property_gather_preserves_values() {
        // Every element in the output must exist in the source.
        let src: Vec<f32> = (0..20).map(|x| x as f32 * 1.5).collect();
        let indices: Vec<usize> = (0..10).map(|i| i % 4).collect();
        let op = GatherOp::new((4, 5), (2, 5), 0, true).unwrap();
        let mut out = [0.0f32; 10];
        op.execute(&src, &indices, &mut out).unwrap();
        for &v in &out {
            assert!(src.contains(&v));
        }
    }

    #[test]
    fn property_scatter_add_commutative() {
        // Order of scattering shouldn't matter for Add.
        let src1 = [1.0, 2.0];
        let src2 = [3.0, 4.0];
        let idx1 = [0, 0]; // 1×2
        let idx2 = [0, 0];

        let op1 = ScatterOp::new((2, 2), (1, 2), 0, ScatterReduce::Add, true).unwrap();
        let op2 = ScatterOp::new((2, 2), (1, 2), 0, ScatterReduce::Add, true).unwrap();

        // Order 1: src1 then src2
        let mut dst_a = [0.0f32; 4];
        op1.execute(&src1, &idx1, &mut dst_a, None).unwrap();
        op2.execute(&src2, &idx2, &mut dst_a, None).unwrap();

        // Order 2: src2 then src1
        let mut dst_b = [0.0f32; 4];
        op2.execute(&src2, &idx2, &mut dst_b, None).unwrap();
        op1.execute(&src1, &idx1, &mut dst_b, None).unwrap();

        assert_eq!(dst_a, dst_b);
    }

    #[test]
    fn property_masked_fill_select_inverse() {
        // Fill with sentinel, then select non-sentinel = original
        // masked positions.
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mask = [true, false, true, false, true];
        let sentinel = -999.0;

        let filler = MaskedFill::new(sentinel);
        let mut filled = [0.0f32; 5];
        filler.execute(&input, &mask, &mut filled).unwrap();

        // Invert mask for select
        let inv_mask: Vec<bool> = mask.iter().map(|&m| !m).collect();
        let selected = MaskedSelect::execute(&filled, &inv_mask).unwrap();
        // Selected should be the untouched values.
        assert_eq!(selected, [2.0, 4.0]);
    }

    #[test]
    fn property_topk_indices_unique_for_distinct_values() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let sel = TopKSelect::new(3, true, true);
        let result = sel.execute(&input).unwrap();
        let mut indices = result.indices.clone();
        indices.sort();
        indices.dedup();
        assert_eq!(indices.len(), 3);
    }

    #[test]
    fn property_gather_identity_with_sequential_indices() {
        // Gathering with indices [0,1,2,...] = identity.
        let src: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let indices: Vec<usize> = (0..4).flat_map(|r| vec![r; 3]).collect();
        let op = GatherOp::new((4, 3), (4, 3), 0, true).unwrap();
        let mut out = [0.0f32; 12];
        op.execute(&src, &indices, &mut out).unwrap();
        assert_eq!(&out[..], &src[..]);
    }

    #[test]
    fn property_index_select_all_rows() {
        // Selecting all rows in order = identity.
        let src: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let sel = IndexSelect::new((4, 2), true);
        let indices = [0, 1, 2, 3];
        let mut out = [0.0f32; 8];
        sel.execute(&src, &indices, &mut out).unwrap();
        assert_eq!(&out[..], &src[..]);
    }

    #[test]
    fn property_index_select_reverse() {
        let src = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3×2
        let sel = IndexSelect::new((3, 2), true);
        let indices = [2, 1, 0]; // reverse
        let mut out = [0.0f32; 6];
        sel.execute(&src, &indices, &mut out).unwrap();
        assert_eq!(out, [5.0, 6.0, 3.0, 4.0, 1.0, 2.0]);
    }
}

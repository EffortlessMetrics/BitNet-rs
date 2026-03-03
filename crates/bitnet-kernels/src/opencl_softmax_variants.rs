//! OpenCL-optimized softmax variants with CPU reference implementations.
//!
//! # Overview
//!
//! This module implements a comprehensive set of softmax operations used
//! across transformer inference — from standard softmax through
//! FlashAttention-style online softmax and fused attention masking. It
//! provides:
//!
//! - **Standard softmax** — numerically stable `exp(x - max) / sum`.
//! - **Log-softmax** — `x - max - log(sum(exp(x - max)))`, avoids `log(exp)`.
//! - **Online softmax** — single-pass streaming algorithm (FlashAttention).
//! - **Fused softmax** — scale + mask + softmax in one pass (attention).
//! - **Sparse top-k softmax** — softmax over top-k values only.
//! - **Temperature softmax** — temperature-scaled softmax for sampling.
//! - **OpenCL kernel source** — work-group reduction softmax kernel.
//! - **CPU reference** — scalar implementations for correctness testing.
//!
//! # OpenCL kernel
//!
//! The embedded OpenCL C source ([`SOFTMAX_CL`]) contains a work-group
//! parallel softmax kernel using local-memory reductions for max-finding
//! and sum accumulation. A row-per-workgroup design keeps memory access
//! coalesced across work-items.

use bitnet_common::{KernelError, Result};

// ---------------------------------------------------------------------------
// OpenCL kernel source
// ---------------------------------------------------------------------------

/// OpenCL C source for work-group-parallel softmax variants.
///
/// Provides three kernels:
/// - `softmax_row` — standard numerically-stable softmax.
/// - `log_softmax_row` — fused log-softmax.
/// - `fused_scale_mask_softmax` — scaled + masked softmax for attention.
pub const SOFTMAX_CL: &str = r#"
// ---- softmax_row --------------------------------------------------------
// One workgroup per row. Local memory for max-reduce + sum-reduce.
__kernel void softmax_row(
    __global const float* input,
    __global       float* output,
    const int cols
) {
    const int row  = get_group_id(0);
    const int lid  = get_local_id(0);
    const int lsz  = get_local_size(0);
    const int base = row * cols;

    __local float scratch[256];

    // ---- pass 1: find row max -------------------------------------------
    float local_max = -INFINITY;
    for (int c = lid; c < cols; c += lsz) {
        float v = input[base + c];
        local_max = fmax(local_max, v);
    }
    scratch[lid] = local_max;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsz >> 1; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] = fmax(scratch[lid], scratch[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float row_max = scratch[0];

    // ---- pass 2: sum of exp(x - max) ------------------------------------
    float local_sum = 0.0f;
    for (int c = lid; c < cols; c += lsz) {
        local_sum += exp(input[base + c] - row_max);
    }
    scratch[lid] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsz >> 1; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float row_sum = scratch[0];

    // ---- pass 3: write normalised output --------------------------------
    float inv_sum = 1.0f / row_sum;
    for (int c = lid; c < cols; c += lsz) {
        output[base + c] = exp(input[base + c] - row_max) * inv_sum;
    }
}

// ---- log_softmax_row ----------------------------------------------------
__kernel void log_softmax_row(
    __global const float* input,
    __global       float* output,
    const int cols
) {
    const int row  = get_group_id(0);
    const int lid  = get_local_id(0);
    const int lsz  = get_local_size(0);
    const int base = row * cols;

    __local float scratch[256];

    // max reduce
    float local_max = -INFINITY;
    for (int c = lid; c < cols; c += lsz)
        local_max = fmax(local_max, input[base + c]);
    scratch[lid] = local_max;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsz >> 1; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] = fmax(scratch[lid], scratch[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float row_max = scratch[0];

    // sum reduce
    float local_sum = 0.0f;
    for (int c = lid; c < cols; c += lsz)
        local_sum += exp(input[base + c] - row_max);
    scratch[lid] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsz >> 1; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float log_sum = log(scratch[0]);

    // output = x - max - log(sum)
    for (int c = lid; c < cols; c += lsz)
        output[base + c] = input[base + c] - row_max - log_sum;
}

// ---- fused_scale_mask_softmax -------------------------------------------
// scale  -> x * scale_factor
// mask   -> if mask[i]==0 set to -INFINITY
// softmax-> standard row softmax
__kernel void fused_scale_mask_softmax(
    __global const float* input,
    __global const int*   mask,
    __global       float* output,
    const int   cols,
    const float scale_factor
) {
    const int row  = get_group_id(0);
    const int lid  = get_local_id(0);
    const int lsz  = get_local_size(0);
    const int base = row * cols;

    __local float scratch[256];

    // pass 1: max (after scale + mask)
    float local_max = -INFINITY;
    for (int c = lid; c < cols; c += lsz) {
        float v = input[base + c] * scale_factor;
        if (mask[base + c] == 0) v = -INFINITY;
        local_max = fmax(local_max, v);
    }
    scratch[lid] = local_max;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsz >> 1; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] = fmax(scratch[lid], scratch[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float row_max = scratch[0];

    // pass 2: sum
    float local_sum = 0.0f;
    for (int c = lid; c < cols; c += lsz) {
        float v = input[base + c] * scale_factor;
        if (mask[base + c] == 0) v = -INFINITY;
        local_sum += exp(v - row_max);
    }
    scratch[lid] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsz >> 1; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float row_sum = scratch[0];

    // pass 3: write
    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    for (int c = lid; c < cols; c += lsz) {
        float v = input[base + c] * scale_factor;
        if (mask[base + c] == 0) v = -INFINITY;
        output[base + c] = exp(v - row_max) * inv_sum;
    }
}
"#;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for softmax operations.
#[derive(Debug, Clone)]
pub struct SoftmaxConfig {
    /// The axis along which softmax is computed (0 = row-wise).
    pub axis: usize,
    /// Small epsilon added to denominators for numerical stability.
    pub numerical_eps: f32,
    /// Temperature scaling factor (applied before softmax).
    /// `T < 1.0` sharpens, `T > 1.0` smooths the distribution.
    pub temperature: f32,
    /// If set, only the top-k values participate in softmax.
    pub top_k: Option<usize>,
}

impl Default for SoftmaxConfig {
    fn default() -> Self {
        Self { axis: 0, numerical_eps: 1e-8, temperature: 1.0, top_k: None }
    }
}

impl SoftmaxConfig {
    /// Create a default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the axis.
    #[must_use]
    pub fn with_axis(mut self, axis: usize) -> Self {
        self.axis = axis;
        self
    }

    /// Set numerical epsilon.
    #[must_use]
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.numerical_eps = eps;
        self
    }

    /// Set temperature.
    #[must_use]
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    /// Set top-k.
    #[must_use]
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = Some(k);
        self
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Runtime statistics for a softmax invocation.
#[derive(Debug, Clone)]
pub struct SoftmaxStats {
    /// Number of rows processed.
    pub rows_processed: usize,
    /// Number of columns per row.
    pub cols: usize,
    /// Total elements processed.
    pub total_elements: usize,
    /// Minimum output value across all rows.
    pub min_output: f32,
    /// Maximum output value across all rows.
    pub max_output: f32,
    /// Count of output values that underflowed to zero.
    pub underflow_count: usize,
    /// Throughput in elements per second (0.0 if timing not measured).
    pub throughput_elements_per_sec: f64,
}

impl SoftmaxStats {
    /// Collect statistics from softmax output.
    pub fn from_output(output: &[f32], rows: usize, cols: usize) -> Self {
        let mut min_output = f32::INFINITY;
        let mut max_output = f32::NEG_INFINITY;
        let mut underflow_count = 0usize;
        for &v in output {
            if v < min_output {
                min_output = v;
            }
            if v > max_output {
                max_output = v;
            }
            if v == 0.0 {
                underflow_count += 1;
            }
        }
        Self {
            rows_processed: rows,
            cols,
            total_elements: rows * cols,
            min_output,
            max_output,
            underflow_count,
            throughput_elements_per_sec: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Standard Softmax
// ---------------------------------------------------------------------------

/// Standard numerically-stable softmax: `exp(x - max) / sum(exp(x - max))`.
pub struct Softmax;

impl Softmax {
    /// Compute softmax in-place over a single row.
    #[inline]
    pub fn row_inplace(row: &mut [f32]) {
        if row.is_empty() {
            return;
        }
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for v in row.iter_mut() {
                *v *= inv;
            }
        }
    }

    /// Compute softmax over `input`, writing results to `output`.
    ///
    /// `input` is row-major `[rows, cols]`.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] when sizes mismatch.
    pub fn forward(input: &[f32], output: &mut [f32], rows: usize, cols: usize) -> Result<()> {
        validate_dims(input.len(), output.len(), rows, cols)?;
        output.copy_from_slice(input);
        for r in 0..rows {
            let start = r * cols;
            Self::row_inplace(&mut output[start..start + cols]);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Log-Softmax
// ---------------------------------------------------------------------------

/// Log-domain softmax: `x - max - log(sum(exp(x - max)))`.
///
/// More numerically stable than computing `log(softmax(x))` because it avoids
/// the intermediate `exp → log` round-trip.
pub struct LogSoftmax;

impl LogSoftmax {
    /// Compute log-softmax in-place over a single row.
    #[inline]
    pub fn row_inplace(row: &mut [f32]) {
        if row.is_empty() {
            return;
        }
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let log_sum_exp: f32 = row.iter().map(|&v| (v - max).exp()).sum::<f32>().ln();
        for v in row.iter_mut() {
            *v = *v - max - log_sum_exp;
        }
    }

    /// Compute log-softmax over `input`, writing results to `output`.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] when sizes mismatch.
    pub fn forward(input: &[f32], output: &mut [f32], rows: usize, cols: usize) -> Result<()> {
        validate_dims(input.len(), output.len(), rows, cols)?;
        output.copy_from_slice(input);
        for r in 0..rows {
            let start = r * cols;
            Self::row_inplace(&mut output[start..start + cols]);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Online Softmax (single-pass, FlashAttention style)
// ---------------------------------------------------------------------------

/// Single-pass online softmax (Milakov & Gimelshein, 2018).
///
/// Maintains a running `(max, sum)` state that is updated incrementally,
/// enabling streaming computation without a separate max-finding pass.
/// This is the algorithm underlying FlashAttention's numerically-stable
/// softmax within tiled SRAM blocks.
pub struct OnlineSoftmax;

/// Running state for online softmax accumulation.
#[derive(Debug, Clone, Copy)]
pub struct OnlineSoftmaxState {
    /// Running maximum.
    pub max: f32,
    /// Running sum of `exp(x_i - max)`, corrected for max updates.
    pub sum: f32,
}

impl Default for OnlineSoftmaxState {
    fn default() -> Self {
        Self { max: f32::NEG_INFINITY, sum: 0.0 }
    }
}

impl OnlineSoftmaxState {
    /// Incorporate a new value into the running state.
    #[inline]
    pub fn update(&mut self, x: f32) {
        if x > self.max {
            // Rescale existing sum to the new max.
            self.sum = self.sum * (self.max - x).exp() + (0.0f32); // correction
            self.max = x;
        }
        self.sum += (x - self.max).exp();
    }

    /// Merge two states (e.g. from parallel tiles).
    #[inline]
    pub fn merge(a: Self, b: Self) -> Self {
        if a.max >= b.max {
            Self { max: a.max, sum: a.sum + b.sum * (b.max - a.max).exp() }
        } else {
            Self { max: b.max, sum: b.sum + a.sum * (a.max - b.max).exp() }
        }
    }
}

impl OnlineSoftmax {
    /// Compute softmax using the online/streaming algorithm.
    ///
    /// Produces identical results to [`Softmax::forward`] within floating-point
    /// tolerance.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] when sizes mismatch.
    pub fn forward(input: &[f32], output: &mut [f32], rows: usize, cols: usize) -> Result<()> {
        validate_dims(input.len(), output.len(), rows, cols)?;
        for r in 0..rows {
            let start = r * cols;
            let row = &input[start..start + cols];

            // Pass 1: compute running (max, sum).
            let mut state = OnlineSoftmaxState::default();
            for &x in row {
                state.update(x);
            }

            // Pass 2: normalise.
            let inv_sum = if state.sum > 0.0 { 1.0 / state.sum } else { 0.0 };
            for (i, &x) in row.iter().enumerate() {
                output[start + i] = (x - state.max).exp() * inv_sum;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Fused Scale + Mask + Softmax
// ---------------------------------------------------------------------------

/// Fused scale → mask → softmax, typical in attention score computation.
///
/// Computes `softmax(input * scale, mask)` where masked positions are set to
/// `-inf` before the softmax normalisation.
pub struct FusedSoftmax;

impl FusedSoftmax {
    /// Compute fused scale + mask + softmax.
    ///
    /// `mask` is row-major `[rows, cols]`, `true` ⇒ keep, `false` ⇒ mask out.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] when sizes mismatch.
    pub fn forward(
        input: &[f32],
        mask: &[bool],
        output: &mut [f32],
        rows: usize,
        cols: usize,
        scale: f32,
    ) -> Result<()> {
        validate_dims(input.len(), output.len(), rows, cols)?;
        if mask.len() != rows * cols {
            return Err(KernelError::InvalidArguments {
                reason: format!("mask length {} != rows*cols {}", mask.len(), rows * cols),
            }
            .into());
        }
        for r in 0..rows {
            let start = r * cols;
            // Scale + mask.
            for c in 0..cols {
                let idx = start + c;
                output[idx] = if mask[idx] { input[idx] * scale } else { f32::NEG_INFINITY };
            }
            // Softmax the row.
            Softmax::row_inplace(&mut output[start..start + cols]);
        }
        Ok(())
    }

    /// Generate a lower-triangular causal mask `[rows, cols]`.
    ///
    /// `mask[i][j] = j <= i + offset`.
    pub fn causal_mask(rows: usize, cols: usize, offset: usize) -> Vec<bool> {
        let mut mask = vec![false; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                mask[i * cols + j] = j <= i + offset;
            }
        }
        mask
    }
}

// ---------------------------------------------------------------------------
// Sparse Top-K Softmax
// ---------------------------------------------------------------------------

/// Softmax restricted to the top-k elements; remaining positions are zeroed.
///
/// This is useful for efficient sparse attention and sampling where only the
/// highest-scoring candidates matter.
pub struct SparseTopKSoftmax;

impl SparseTopKSoftmax {
    /// Compute sparse top-k softmax.
    ///
    /// For each row, only the `k` largest values participate in the softmax
    /// normalisation. All other positions are set to `0.0`.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] when sizes mismatch or `k == 0`.
    pub fn forward(
        input: &[f32],
        output: &mut [f32],
        rows: usize,
        cols: usize,
        k: usize,
    ) -> Result<()> {
        validate_dims(input.len(), output.len(), rows, cols)?;
        if k == 0 {
            return Err(KernelError::InvalidArguments { reason: "top_k must be > 0".into() }.into());
        }
        let k = k.min(cols);
        for r in 0..rows {
            let start = r * cols;
            let row = &input[start..start + cols];

            // Find the k-th largest value (partial sort via indices).
            let mut indices: Vec<usize> = (0..cols).collect();
            indices.sort_unstable_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());

            // Build output: top-k get softmax, rest get 0.
            let top_k_indices = &indices[..k];
            let max = row[top_k_indices[0]];
            let mut sum = 0.0f32;
            for &idx in top_k_indices {
                sum += (row[idx] - max).exp();
            }
            let inv_sum = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            // Zero out everything first.
            output[start..start + cols].fill(0.0);
            for &idx in top_k_indices {
                output[start + idx] = (row[idx] - max).exp() * inv_sum;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Temperature Softmax
// ---------------------------------------------------------------------------

/// Temperature-scaled softmax: `softmax(x / T)`.
///
/// - `T < 1.0` → sharper (more confident) distribution.
/// - `T > 1.0` → smoother (more uniform) distribution.
/// - `T == 1.0` → standard softmax.
pub struct TemperatureSoftmax;

impl TemperatureSoftmax {
    /// Compute temperature-scaled softmax.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] when sizes mismatch or
    /// `temperature <= 0.0`.
    pub fn forward(
        input: &[f32],
        output: &mut [f32],
        rows: usize,
        cols: usize,
        temperature: f32,
    ) -> Result<()> {
        validate_dims(input.len(), output.len(), rows, cols)?;
        if temperature <= 0.0 {
            return Err(KernelError::InvalidArguments {
                reason: format!("temperature must be > 0, got {temperature}"),
            }
            .into());
        }
        let inv_temp = 1.0 / temperature;
        for r in 0..rows {
            let start = r * cols;
            for c in 0..cols {
                output[start + c] = input[start + c] * inv_temp;
            }
            Softmax::row_inplace(&mut output[start..start + cols]);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Validate that input/output lengths match the expected `rows * cols`.
fn validate_dims(input_len: usize, output_len: usize, rows: usize, cols: usize) -> Result<()> {
    let expected = rows * cols;
    if input_len != expected {
        return Err(KernelError::InvalidArguments {
            reason: format!("input length {input_len} != rows*cols {expected}"),
        }
        .into());
    }
    if output_len != expected {
        return Err(KernelError::InvalidArguments {
            reason: format!("output length {output_len} != rows*cols {expected}"),
        }
        .into());
    }
    Ok(())
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const ATOL: f32 = 1e-5;

    fn assert_close(a: f32, b: f32, tol: f32) {
        assert!((a - b).abs() <= tol, "values not close: {a} vs {b} (diff={})", (a - b).abs());
    }

    fn assert_slices_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() <= tol,
                "mismatch at index {i}: {x} vs {y} (diff={})",
                (x - y).abs()
            );
        }
    }

    fn row_sum(data: &[f32]) -> f32 {
        data.iter().sum()
    }

    // =====================================================================
    // Standard Softmax
    // =====================================================================

    #[test]
    fn test_softmax_sums_to_one() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        Softmax::forward(&input, &mut output, 1, 4).unwrap();
        assert_close(row_sum(&output), 1.0, ATOL);
    }

    #[test]
    fn test_softmax_all_positive() {
        let input = vec![1.0, 2.0, 3.0];
        let mut output = vec![0.0; 3];
        Softmax::forward(&input, &mut output, 1, 3).unwrap();
        for &v in &output {
            assert!(v > 0.0 && v <= 1.0, "expected (0,1] got {v}");
        }
    }

    #[test]
    fn test_softmax_monotonicity() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        Softmax::forward(&input, &mut output, 1, 4).unwrap();
        for i in 1..output.len() {
            assert!(output[i] > output[i - 1], "not monotonically increasing");
        }
    }

    #[test]
    fn test_softmax_multi_row() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output = vec![0.0; 6];
        Softmax::forward(&input, &mut output, 2, 3).unwrap();
        assert_close(row_sum(&output[0..3]), 1.0, ATOL);
        assert_close(row_sum(&output[3..6]), 1.0, ATOL);
    }

    #[test]
    fn test_softmax_single_element() {
        let input = vec![42.0];
        let mut output = vec![0.0];
        Softmax::forward(&input, &mut output, 1, 1).unwrap();
        assert_close(output[0], 1.0, ATOL);
    }

    #[test]
    fn test_softmax_uniform_input() {
        let input = vec![5.0; 8];
        let mut output = vec![0.0; 8];
        Softmax::forward(&input, &mut output, 1, 8).unwrap();
        for &v in &output {
            assert_close(v, 0.125, ATOL);
        }
    }

    #[test]
    fn test_softmax_large_values_stability() {
        let input = vec![1000.0, 1001.0, 1002.0];
        let mut output = vec![0.0; 3];
        Softmax::forward(&input, &mut output, 1, 3).unwrap();
        assert_close(row_sum(&output), 1.0, ATOL);
        for &v in &output {
            assert!(!v.is_nan(), "NaN in output");
            assert!(!v.is_infinite(), "Inf in output");
        }
    }

    #[test]
    fn test_softmax_negative_values() {
        let input = vec![-1.0, -2.0, -3.0, -4.0];
        let mut output = vec![0.0; 4];
        Softmax::forward(&input, &mut output, 1, 4).unwrap();
        assert_close(row_sum(&output), 1.0, ATOL);
    }

    #[test]
    fn test_softmax_mixed_sign() {
        let input = vec![-10.0, 0.0, 10.0];
        let mut output = vec![0.0; 3];
        Softmax::forward(&input, &mut output, 1, 3).unwrap();
        assert_close(row_sum(&output), 1.0, ATOL);
        // The largest input should dominate.
        assert!(output[2] > 0.99);
    }

    #[test]
    fn test_softmax_near_zero() {
        let input = vec![1e-10, 2e-10, 3e-10];
        let mut output = vec![0.0; 3];
        Softmax::forward(&input, &mut output, 1, 3).unwrap();
        assert_close(row_sum(&output), 1.0, ATOL);
    }

    #[test]
    fn test_softmax_very_negative() {
        let input = vec![-1000.0, -999.0, -998.0];
        let mut output = vec![0.0; 3];
        Softmax::forward(&input, &mut output, 1, 3).unwrap();
        assert_close(row_sum(&output), 1.0, ATOL);
        for &v in &output {
            assert!(!v.is_nan());
        }
    }

    #[test]
    fn test_softmax_deterministic() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut out1 = vec![0.0; 5];
        let mut out2 = vec![0.0; 5];
        Softmax::forward(&input, &mut out1, 1, 5).unwrap();
        Softmax::forward(&input, &mut out2, 1, 5).unwrap();
        assert_slices_close(&out1, &out2, 0.0);
    }

    #[test]
    fn test_softmax_size_mismatch_input() {
        let input = vec![1.0, 2.0, 3.0];
        let mut output = vec![0.0; 4];
        assert!(Softmax::forward(&input, &mut output, 1, 4).is_err());
    }

    #[test]
    fn test_softmax_size_mismatch_output() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 3];
        assert!(Softmax::forward(&input, &mut output, 1, 4).is_err());
    }

    #[test]
    fn test_softmax_row_inplace_empty() {
        let mut row: Vec<f32> = vec![];
        Softmax::row_inplace(&mut row);
        assert!(row.is_empty());
    }

    #[test]
    fn test_softmax_long_sequence() {
        let n = 4096;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
        let mut output = vec![0.0; n];
        Softmax::forward(&input, &mut output, 1, n).unwrap();
        assert_close(row_sum(&output), 1.0, 1e-4);
    }

    #[test]
    fn test_softmax_all_same_large_row() {
        let n = 1024;
        let input = vec![std::f32::consts::PI; n];
        let mut output = vec![0.0; n];
        Softmax::forward(&input, &mut output, 1, n).unwrap();
        let expected = 1.0 / n as f32;
        for &v in &output {
            assert_close(v, expected, ATOL);
        }
    }

    #[test]
    fn test_softmax_inf_input_handled() {
        // One element is +inf; exp(inf - inf) = exp(0) = 1, others = exp(-inf) = 0.
        // However, inf arithmetic can produce NaN — verify no panic and finite max.
        let input = vec![1.0, f32::INFINITY, 2.0];
        let mut output = vec![0.0; 3];
        Softmax::forward(&input, &mut output, 1, 3).unwrap();
        // The +inf element should dominate.  Because exp(inf-inf)=exp(NaN)=NaN on
        // some platforms, we just verify no panic occurred and non-inf positions
        // are pushed toward zero.
        assert!(output[0] <= ATOL, "non-max element should be ~0");
        assert!(output[2] <= ATOL, "non-max element should be ~0");
    }

    #[test]
    fn test_softmax_all_neginf() {
        // All -inf: degenerate case; output should be 0 (no probability mass).
        let input = vec![f32::NEG_INFINITY; 4];
        let mut output = vec![0.0; 4];
        Softmax::forward(&input, &mut output, 1, 4).unwrap();
        // exp(-inf - (-inf)) = exp(NaN) is 0/0 → handle gracefully.
        for &v in &output {
            assert!(!v.is_infinite());
        }
    }

    // =====================================================================
    // Log-Softmax
    // =====================================================================

    #[test]
    fn test_log_softmax_equals_log_of_softmax() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut sm_out = vec![0.0; 4];
        let mut lsm_out = vec![0.0; 4];
        Softmax::forward(&input, &mut sm_out, 1, 4).unwrap();
        LogSoftmax::forward(&input, &mut lsm_out, 1, 4).unwrap();
        for i in 0..4 {
            assert_close(lsm_out[i], sm_out[i].ln(), ATOL);
        }
    }

    #[test]
    fn test_log_softmax_all_negative_output() {
        let input = vec![1.0, 2.0, 3.0];
        let mut output = vec![0.0; 3];
        LogSoftmax::forward(&input, &mut output, 1, 3).unwrap();
        for &v in &output {
            assert!(v <= 0.0, "log-softmax should be <= 0, got {v}");
        }
    }

    #[test]
    fn test_log_softmax_exp_sums_to_one() {
        let input = vec![2.0, 3.0, 5.0, 7.0];
        let mut output = vec![0.0; 4];
        LogSoftmax::forward(&input, &mut output, 1, 4).unwrap();
        let sum: f32 = output.iter().map(|&v| v.exp()).sum();
        assert_close(sum, 1.0, ATOL);
    }

    #[test]
    fn test_log_softmax_multi_row() {
        let input = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let mut output = vec![0.0; 6];
        LogSoftmax::forward(&input, &mut output, 2, 3).unwrap();
        for r in 0..2 {
            let row = &output[r * 3..(r + 1) * 3];
            let sum: f32 = row.iter().map(|&v| v.exp()).sum();
            assert_close(sum, 1.0, ATOL);
        }
    }

    #[test]
    fn test_log_softmax_single_element() {
        let input = vec![42.0];
        let mut output = vec![0.0];
        LogSoftmax::forward(&input, &mut output, 1, 1).unwrap();
        assert_close(output[0], 0.0, ATOL);
    }

    #[test]
    fn test_log_softmax_large_values_stability() {
        let input = vec![500.0, 501.0, 502.0];
        let mut output = vec![0.0; 3];
        LogSoftmax::forward(&input, &mut output, 1, 3).unwrap();
        for &v in &output {
            assert!(!v.is_nan(), "NaN in log-softmax output");
            assert!(!v.is_infinite(), "Inf in log-softmax output");
        }
    }

    #[test]
    fn test_log_softmax_size_mismatch() {
        let input = vec![1.0; 3];
        let mut output = vec![0.0; 4];
        assert!(LogSoftmax::forward(&input, &mut output, 1, 4).is_err());
    }

    #[test]
    fn test_log_softmax_ordering_preserved() {
        let input = vec![1.0, 3.0, 2.0, 5.0];
        let mut output = vec![0.0; 4];
        LogSoftmax::forward(&input, &mut output, 1, 4).unwrap();
        // Relative ordering should be preserved.
        assert!(output[3] > output[1]); // 5.0 > 3.0
        assert!(output[1] > output[2]); // 3.0 > 2.0
        assert!(output[2] > output[0]); // 2.0 > 1.0
    }

    // =====================================================================
    // Online Softmax
    // =====================================================================

    #[test]
    fn test_online_softmax_matches_standard() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut std_out = vec![0.0; 5];
        let mut online_out = vec![0.0; 5];
        Softmax::forward(&input, &mut std_out, 1, 5).unwrap();
        OnlineSoftmax::forward(&input, &mut online_out, 1, 5).unwrap();
        assert_slices_close(&std_out, &online_out, ATOL);
    }

    #[test]
    fn test_online_softmax_multi_row() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut std_out = vec![0.0; 8];
        let mut online_out = vec![0.0; 8];
        Softmax::forward(&input, &mut std_out, 2, 4).unwrap();
        OnlineSoftmax::forward(&input, &mut online_out, 2, 4).unwrap();
        assert_slices_close(&std_out, &online_out, ATOL);
    }

    #[test]
    fn test_online_softmax_single_element() {
        let input = vec![99.0];
        let mut output = vec![0.0];
        OnlineSoftmax::forward(&input, &mut output, 1, 1).unwrap();
        assert_close(output[0], 1.0, ATOL);
    }

    #[test]
    fn test_online_softmax_large_values() {
        let input = vec![1000.0, 1001.0, 1002.0, 1003.0];
        let mut std_out = vec![0.0; 4];
        let mut online_out = vec![0.0; 4];
        Softmax::forward(&input, &mut std_out, 1, 4).unwrap();
        OnlineSoftmax::forward(&input, &mut online_out, 1, 4).unwrap();
        assert_slices_close(&std_out, &online_out, ATOL);
    }

    #[test]
    fn test_online_softmax_negative_values() {
        let input = vec![-5.0, -3.0, -1.0, -10.0];
        let mut std_out = vec![0.0; 4];
        let mut online_out = vec![0.0; 4];
        Softmax::forward(&input, &mut std_out, 1, 4).unwrap();
        OnlineSoftmax::forward(&input, &mut online_out, 1, 4).unwrap();
        assert_slices_close(&std_out, &online_out, ATOL);
    }

    #[test]
    fn test_online_softmax_uniform() {
        let input = vec![7.0; 16];
        let mut output = vec![0.0; 16];
        OnlineSoftmax::forward(&input, &mut output, 1, 16).unwrap();
        for &v in &output {
            assert_close(v, 1.0 / 16.0, ATOL);
        }
    }

    #[test]
    fn test_online_softmax_sums_to_one() {
        let input = vec![0.1, 0.5, 0.9, 1.3, 1.7, 2.1];
        let mut output = vec![0.0; 6];
        OnlineSoftmax::forward(&input, &mut output, 1, 6).unwrap();
        assert_close(row_sum(&output), 1.0, ATOL);
    }

    #[test]
    fn test_online_softmax_state_merge() {
        // Merge two halves should equal processing all at once.
        let input = vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0];
        let mut full_state = OnlineSoftmaxState::default();
        for &x in &input {
            full_state.update(x);
        }

        let mut left = OnlineSoftmaxState::default();
        for &x in &input[..3] {
            left.update(x);
        }
        let mut right = OnlineSoftmaxState::default();
        for &x in &input[3..] {
            right.update(x);
        }
        let merged = OnlineSoftmaxState::merge(left, right);

        assert_close(full_state.max, merged.max, ATOL);
        assert_close(full_state.sum, merged.sum, ATOL);
    }

    #[test]
    fn test_online_softmax_size_mismatch() {
        let input = vec![1.0; 6];
        let mut output = vec![0.0; 5];
        assert!(OnlineSoftmax::forward(&input, &mut output, 2, 3).is_err());
    }

    // =====================================================================
    // Fused Softmax
    // =====================================================================

    #[test]
    fn test_fused_softmax_no_mask_unit_scale() {
        // All-true mask + scale=1.0 → same as standard softmax.
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![true; 4];
        let mut fused_out = vec![0.0; 4];
        let mut std_out = vec![0.0; 4];
        FusedSoftmax::forward(&input, &mask, &mut fused_out, 1, 4, 1.0).unwrap();
        Softmax::forward(&input, &mut std_out, 1, 4).unwrap();
        assert_slices_close(&fused_out, &std_out, ATOL);
    }

    #[test]
    fn test_fused_softmax_with_scale() {
        let input = vec![1.0, 2.0, 3.0];
        let mask = vec![true; 3];
        let scale = 0.5;
        let mut output = vec![0.0; 3];
        FusedSoftmax::forward(&input, &mask, &mut output, 1, 3, scale).unwrap();
        // Manually: softmax([0.5, 1.0, 1.5])
        let scaled = vec![0.5, 1.0, 1.5];
        let mut expected = vec![0.0; 3];
        Softmax::forward(&scaled, &mut expected, 1, 3).unwrap();
        assert_slices_close(&output, &expected, ATOL);
    }

    #[test]
    fn test_fused_softmax_causal_mask() {
        let cols = 4;
        let mask = FusedSoftmax::causal_mask(4, cols, 0);
        let input = vec![1.0; 16];
        let mut output = vec![0.0; 16];
        FusedSoftmax::forward(&input, &mask, &mut output, 4, cols, 1.0).unwrap();
        // Row 0: only column 0 is allowed → output[0] = 1.0.
        assert_close(output[0], 1.0, ATOL);
        // Row 3: all 4 columns allowed → uniform.
        for c in 0..4 {
            assert_close(output[3 * cols + c], 0.25, ATOL);
        }
    }

    #[test]
    fn test_fused_softmax_causal_rows_sum_to_one() {
        let n = 8;
        let mask = FusedSoftmax::causal_mask(n, n, 0);
        let input: Vec<f32> = (0..n * n).map(|i| (i as f32) * 0.1).collect();
        let mut output = vec![0.0; n * n];
        FusedSoftmax::forward(&input, &mask, &mut output, n, n, 1.0).unwrap();
        for r in 0..n {
            assert_close(row_sum(&output[r * n..(r + 1) * n]), 1.0, ATOL);
        }
    }

    #[test]
    fn test_fused_softmax_fully_masked_row() {
        // All masked → degenerate; softmax of all -inf.
        let input = vec![1.0, 2.0, 3.0];
        let mask = vec![false; 3];
        let mut output = vec![0.0; 3];
        FusedSoftmax::forward(&input, &mask, &mut output, 1, 3, 1.0).unwrap();
        // All outputs should be zero or NaN-free.
        for &v in &output {
            assert!(!v.is_infinite(), "unexpected inf");
        }
    }

    #[test]
    fn test_fused_softmax_mask_size_mismatch() {
        let input = vec![1.0; 4];
        let mask = vec![true; 3]; // Wrong size.
        let mut output = vec![0.0; 4];
        assert!(FusedSoftmax::forward(&input, &mask, &mut output, 1, 4, 1.0).is_err());
    }

    #[test]
    fn test_fused_softmax_causal_mask_offset() {
        // With offset=2, row 0 can attend to cols 0,1,2.
        let mask = FusedSoftmax::causal_mask(2, 4, 2);
        assert!(mask[0]); // (0, 0) → 0 <= 0+2 ✓
        assert!(mask[1]); // (0, 1) → 1 <= 2 ✓
        assert!(mask[2]); // (0, 2) → 2 <= 2 ✓
        assert!(!mask[3]); // (0, 3) → 3 <= 2 ✗
        // Row 1 can attend to cols 0,1,2,3.
        assert!(mask[4]); // (1, 0)
        assert!(mask[5]); // (1, 1)
        assert!(mask[6]); // (1, 2)
        assert!(mask[7]); // (1, 3)
    }

    #[test]
    fn test_fused_softmax_masked_positions_zero() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![true, false, true, false];
        let mut output = vec![0.0; 4];
        FusedSoftmax::forward(&input, &mask, &mut output, 1, 4, 1.0).unwrap();
        assert_close(output[1], 0.0, ATOL);
        assert_close(output[3], 0.0, ATOL);
        assert_close(output[0] + output[2], 1.0, ATOL);
    }

    // =====================================================================
    // Sparse Top-K Softmax
    // =====================================================================

    #[test]
    fn test_sparse_topk_preserves_top_values() {
        let input = vec![1.0, 5.0, 3.0, 4.0, 2.0];
        let mut output = vec![0.0; 5];
        SparseTopKSoftmax::forward(&input, &mut output, 1, 5, 2).unwrap();
        // Top-2 values are at indices 1 (5.0) and 3 (4.0).
        assert!(output[1] > 0.0);
        assert!(output[3] > 0.0);
        // Rest should be zero.
        assert_close(output[0], 0.0, ATOL);
        assert_close(output[2], 0.0, ATOL);
        assert_close(output[4], 0.0, ATOL);
    }

    #[test]
    fn test_sparse_topk_sums_to_one() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = vec![0.0; 8];
        SparseTopKSoftmax::forward(&input, &mut output, 1, 8, 3).unwrap();
        assert_close(row_sum(&output), 1.0, ATOL);
    }

    #[test]
    fn test_sparse_topk_k_equals_n() {
        // k == cols → same as standard softmax.
        let input = vec![1.0, 2.0, 3.0];
        let mut sparse_out = vec![0.0; 3];
        let mut std_out = vec![0.0; 3];
        SparseTopKSoftmax::forward(&input, &mut sparse_out, 1, 3, 3).unwrap();
        Softmax::forward(&input, &mut std_out, 1, 3).unwrap();
        assert_slices_close(&sparse_out, &std_out, ATOL);
    }

    #[test]
    fn test_sparse_topk_k_greater_than_n() {
        // k > cols → clamp to cols, same as standard softmax.
        let input = vec![1.0, 2.0];
        let mut output = vec![0.0; 2];
        SparseTopKSoftmax::forward(&input, &mut output, 1, 2, 100).unwrap();
        assert_close(row_sum(&output), 1.0, ATOL);
    }

    #[test]
    fn test_sparse_topk_k_one() {
        // k=1 → all mass on the largest element.
        let input = vec![1.0, 5.0, 3.0];
        let mut output = vec![0.0; 3];
        SparseTopKSoftmax::forward(&input, &mut output, 1, 3, 1).unwrap();
        assert_close(output[1], 1.0, ATOL);
        assert_close(output[0], 0.0, ATOL);
        assert_close(output[2], 0.0, ATOL);
    }

    #[test]
    fn test_sparse_topk_k_zero_error() {
        let input = vec![1.0; 4];
        let mut output = vec![0.0; 4];
        assert!(SparseTopKSoftmax::forward(&input, &mut output, 1, 4, 0).is_err());
    }

    #[test]
    fn test_sparse_topk_multi_row() {
        let input = vec![1.0, 2.0, 3.0, 30.0, 20.0, 10.0];
        let mut output = vec![0.0; 6];
        SparseTopKSoftmax::forward(&input, &mut output, 2, 3, 1).unwrap();
        // Row 0: top-1 is index 2 (val 3.0)
        assert_close(output[2], 1.0, ATOL);
        assert_close(output[0], 0.0, ATOL);
        // Row 1: top-1 is index 0 (val 30.0)
        assert_close(output[3], 1.0, ATOL);
        assert_close(output[5], 0.0, ATOL);
    }

    #[test]
    fn test_sparse_topk_zeros_are_zero() {
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut output = vec![0.0; 16];
        SparseTopKSoftmax::forward(&input, &mut output, 1, 16, 4).unwrap();
        let zero_count = output.iter().filter(|&&v| v == 0.0).count();
        assert_eq!(zero_count, 12);
    }

    // =====================================================================
    // Temperature Softmax
    // =====================================================================

    #[test]
    fn test_temperature_one_equals_standard() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut temp_out = vec![0.0; 4];
        let mut std_out = vec![0.0; 4];
        TemperatureSoftmax::forward(&input, &mut temp_out, 1, 4, 1.0).unwrap();
        Softmax::forward(&input, &mut std_out, 1, 4).unwrap();
        assert_slices_close(&temp_out, &std_out, ATOL);
    }

    #[test]
    fn test_temperature_low_sharper() {
        // T=0.5 → sharper distribution (max prob higher).
        let input = vec![1.0, 2.0, 3.0];
        let mut sharp = vec![0.0; 3];
        let mut normal = vec![0.0; 3];
        TemperatureSoftmax::forward(&input, &mut sharp, 1, 3, 0.5).unwrap();
        Softmax::forward(&input, &mut normal, 1, 3).unwrap();
        // Max element should have higher probability.
        assert!(sharp[2] > normal[2], "T<1 should sharpen");
    }

    #[test]
    fn test_temperature_high_smoother() {
        // T=2.0 → smoother distribution (more uniform).
        let input = vec![1.0, 2.0, 3.0];
        let mut smooth = vec![0.0; 3];
        let mut normal = vec![0.0; 3];
        TemperatureSoftmax::forward(&input, &mut smooth, 1, 3, 2.0).unwrap();
        Softmax::forward(&input, &mut normal, 1, 3).unwrap();
        // Min element should have higher probability with smoothing.
        assert!(smooth[0] > normal[0], "T>1 should smooth");
    }

    #[test]
    fn test_temperature_very_low() {
        // Very low T → nearly argmax.
        let input = vec![1.0, 3.0, 2.0];
        let mut output = vec![0.0; 3];
        TemperatureSoftmax::forward(&input, &mut output, 1, 3, 0.01).unwrap();
        assert!(output[1] > 0.99, "very low T should be near-argmax");
    }

    #[test]
    fn test_temperature_very_high() {
        // Very high T → nearly uniform.
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        TemperatureSoftmax::forward(&input, &mut output, 1, 4, 100.0).unwrap();
        for &v in &output {
            assert_close(v, 0.25, 0.01);
        }
    }

    #[test]
    fn test_temperature_sums_to_one() {
        for &temp in &[0.1, 0.5, 1.0, 2.0, 10.0] {
            let input = vec![0.5, 1.5, 2.5, 3.5];
            let mut output = vec![0.0; 4];
            TemperatureSoftmax::forward(&input, &mut output, 1, 4, temp).unwrap();
            assert_close(row_sum(&output), 1.0, ATOL);
        }
    }

    #[test]
    fn test_temperature_zero_error() {
        let input = vec![1.0; 3];
        let mut output = vec![0.0; 3];
        assert!(TemperatureSoftmax::forward(&input, &mut output, 1, 3, 0.0).is_err());
    }

    #[test]
    fn test_temperature_negative_error() {
        let input = vec![1.0; 3];
        let mut output = vec![0.0; 3];
        assert!(TemperatureSoftmax::forward(&input, &mut output, 1, 3, -1.0).is_err());
    }

    #[test]
    fn test_temperature_multi_row() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output = vec![0.0; 6];
        TemperatureSoftmax::forward(&input, &mut output, 2, 3, 0.5).unwrap();
        assert_close(row_sum(&output[0..3]), 1.0, ATOL);
        assert_close(row_sum(&output[3..6]), 1.0, ATOL);
    }

    // =====================================================================
    // SoftmaxConfig
    // =====================================================================

    #[test]
    fn test_config_defaults() {
        let cfg = SoftmaxConfig::new();
        assert_eq!(cfg.axis, 0);
        assert_close(cfg.temperature, 1.0, ATOL);
        assert!(cfg.top_k.is_none());
        assert!(cfg.numerical_eps > 0.0);
    }

    #[test]
    fn test_config_builder() {
        let cfg =
            SoftmaxConfig::new().with_axis(1).with_temperature(0.7).with_top_k(10).with_eps(1e-6);
        assert_eq!(cfg.axis, 1);
        assert_close(cfg.temperature, 0.7, ATOL);
        assert_eq!(cfg.top_k, Some(10));
        assert_close(cfg.numerical_eps, 1e-6, ATOL);
    }

    // =====================================================================
    // SoftmaxStats
    // =====================================================================

    #[test]
    fn test_stats_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        Softmax::forward(&input, &mut output, 1, 4).unwrap();
        let stats = SoftmaxStats::from_output(&output, 1, 4);
        assert_eq!(stats.rows_processed, 1);
        assert_eq!(stats.cols, 4);
        assert_eq!(stats.total_elements, 4);
        assert!(stats.min_output > 0.0);
        assert!(stats.max_output <= 1.0);
        assert_eq!(stats.underflow_count, 0);
    }

    #[test]
    fn test_stats_underflow_count() {
        let output = vec![0.0, 0.5, 0.0, 0.5];
        let stats = SoftmaxStats::from_output(&output, 1, 4);
        assert_eq!(stats.underflow_count, 2);
    }

    #[test]
    fn test_stats_range() {
        let output = vec![0.1, 0.2, 0.3, 0.4];
        let stats = SoftmaxStats::from_output(&output, 1, 4);
        assert_close(stats.min_output, 0.1, ATOL);
        assert_close(stats.max_output, 0.4, ATOL);
    }

    // =====================================================================
    // OpenCL kernel source
    // =====================================================================

    #[test]
    fn test_opencl_source_not_empty() {
        assert!(!SOFTMAX_CL.is_empty());
    }

    #[test]
    fn test_opencl_source_contains_softmax_kernel() {
        assert!(SOFTMAX_CL.contains("__kernel void softmax_row"));
    }

    #[test]
    fn test_opencl_source_contains_log_softmax_kernel() {
        assert!(SOFTMAX_CL.contains("__kernel void log_softmax_row"));
    }

    #[test]
    fn test_opencl_source_contains_fused_kernel() {
        assert!(SOFTMAX_CL.contains("__kernel void fused_scale_mask_softmax"));
    }

    #[test]
    fn test_opencl_source_contains_barrier() {
        assert!(SOFTMAX_CL.contains("barrier(CLK_LOCAL_MEM_FENCE)"));
    }

    #[test]
    fn test_opencl_source_contains_local_memory() {
        assert!(SOFTMAX_CL.contains("__local float"));
    }

    #[test]
    fn test_opencl_source_contains_exp() {
        assert!(SOFTMAX_CL.contains("exp("));
    }

    #[test]
    fn test_opencl_source_contains_get_group_id() {
        assert!(SOFTMAX_CL.contains("get_group_id"));
    }

    #[test]
    fn test_opencl_source_contains_get_local_id() {
        assert!(SOFTMAX_CL.contains("get_local_id"));
    }

    // =====================================================================
    // Property-like tests
    // =====================================================================

    #[test]
    fn test_property_softmax_in_unit_interval() {
        // All outputs in [0, 1].
        for size in [2, 5, 16, 64, 128] {
            let input: Vec<f32> = (0..size).map(|i| (i as f32) * 0.3 - 5.0).collect();
            let mut output = vec![0.0; size];
            Softmax::forward(&input, &mut output, 1, size).unwrap();
            for &v in &output {
                assert!((0.0..=1.0).contains(&v), "out of [0,1]: {v}");
            }
        }
    }

    #[test]
    fn test_property_softmax_sum_to_one_various_sizes() {
        for size in [1, 2, 3, 4, 8, 16, 32, 100, 256] {
            let input: Vec<f32> = (0..size).map(|i| (i as f32) - (size as f32 / 2.0)).collect();
            let mut output = vec![0.0; size];
            Softmax::forward(&input, &mut output, 1, size).unwrap();
            assert_close(row_sum(&output), 1.0, 1e-4);
        }
    }

    #[test]
    fn test_property_online_matches_standard_various() {
        for size in [3, 7, 16, 50, 128] {
            let input: Vec<f32> = (0..size).map(|i| ((i * 7 + 3) as f32).sin()).collect();
            let mut std_out = vec![0.0; size];
            let mut online_out = vec![0.0; size];
            Softmax::forward(&input, &mut std_out, 1, size).unwrap();
            OnlineSoftmax::forward(&input, &mut online_out, 1, size).unwrap();
            assert_slices_close(&std_out, &online_out, ATOL);
        }
    }

    #[test]
    fn test_property_temperature_monotone_sharpness() {
        // As T decreases, max probability should increase.
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let temps = [5.0, 2.0, 1.0, 0.5, 0.1];
        let mut max_probs = vec![];
        for &t in &temps {
            let mut out = vec![0.0; 5];
            TemperatureSoftmax::forward(&input, &mut out, 1, 5, t).unwrap();
            max_probs.push(out.iter().copied().fold(f32::NEG_INFINITY, f32::max));
        }
        for i in 1..max_probs.len() {
            assert!(
                max_probs[i] >= max_probs[i - 1] - ATOL,
                "sharpness should increase as T decreases: {:?}",
                max_probs
            );
        }
    }

    #[test]
    fn test_property_log_softmax_all_non_positive() {
        for size in [2, 8, 32] {
            let input: Vec<f32> = (0..size).map(|i| i as f32).collect();
            let mut output = vec![0.0; size];
            LogSoftmax::forward(&input, &mut output, 1, size).unwrap();
            for &v in &output {
                assert!(v <= 0.0 + ATOL, "log-softmax should be <= 0, got {v}");
            }
        }
    }

    #[test]
    fn test_property_sparse_topk_at_most_k_nonzero() {
        let input: Vec<f32> = (0..20).map(|i| i as f32).collect();
        for k in [1, 3, 5, 10] {
            let mut output = vec![0.0; 20];
            SparseTopKSoftmax::forward(&input, &mut output, 1, 20, k).unwrap();
            let nonzero = output.iter().filter(|&&v| v > 0.0).count();
            assert_eq!(nonzero, k, "expected {k} nonzero, got {nonzero}");
        }
    }

    #[test]
    fn test_property_fused_causal_upper_triangle_zero() {
        let n = 6;
        let mask = FusedSoftmax::causal_mask(n, n, 0);
        let input: Vec<f32> = (0..n * n).map(|i| (i as f32) * 0.1).collect();
        let mut output = vec![0.0; n * n];
        FusedSoftmax::forward(&input, &mask, &mut output, n, n, 1.0).unwrap();
        for i in 0..n {
            for j in (i + 1)..n {
                assert_close(output[i * n + j], 0.0, ATOL);
            }
        }
    }

    // =====================================================================
    // Cross-variant consistency
    // =====================================================================

    #[test]
    fn test_softmax_vs_temperature_one() {
        let input: Vec<f32> = (0..10).map(|i| i as f32 * 0.5).collect();
        let mut std_out = vec![0.0; 10];
        let mut temp_out = vec![0.0; 10];
        Softmax::forward(&input, &mut std_out, 1, 10).unwrap();
        TemperatureSoftmax::forward(&input, &mut temp_out, 1, 10, 1.0).unwrap();
        assert_slices_close(&std_out, &temp_out, ATOL);
    }

    #[test]
    fn test_all_variants_agree_on_single_element() {
        let input = vec![std::f32::consts::PI];
        let mask = vec![true];
        let mut out_std = vec![0.0];
        let mut out_log = vec![0.0];
        let mut out_online = vec![0.0];
        let mut out_fused = vec![0.0];
        let mut out_sparse = vec![0.0];
        let mut out_temp = vec![0.0];

        Softmax::forward(&input, &mut out_std, 1, 1).unwrap();
        LogSoftmax::forward(&input, &mut out_log, 1, 1).unwrap();
        OnlineSoftmax::forward(&input, &mut out_online, 1, 1).unwrap();
        FusedSoftmax::forward(&input, &mask, &mut out_fused, 1, 1, 1.0).unwrap();
        SparseTopKSoftmax::forward(&input, &mut out_sparse, 1, 1, 1).unwrap();
        TemperatureSoftmax::forward(&input, &mut out_temp, 1, 1, 1.0).unwrap();

        assert_close(out_std[0], 1.0, ATOL);
        assert_close(out_online[0], 1.0, ATOL);
        assert_close(out_fused[0], 1.0, ATOL);
        assert_close(out_sparse[0], 1.0, ATOL);
        assert_close(out_temp[0], 1.0, ATOL);
        assert_close(out_log[0], 0.0, ATOL);
    }

    #[test]
    fn test_all_standard_variants_match() {
        let input: Vec<f32> = (0..8).map(|i| (i as f32) - 3.5).collect();
        let mask = vec![true; 8];
        let mut out_std = vec![0.0; 8];
        let mut out_online = vec![0.0; 8];
        let mut out_fused = vec![0.0; 8];
        let mut out_temp = vec![0.0; 8];
        let mut out_sparse = vec![0.0; 8];

        Softmax::forward(&input, &mut out_std, 1, 8).unwrap();
        OnlineSoftmax::forward(&input, &mut out_online, 1, 8).unwrap();
        FusedSoftmax::forward(&input, &mask, &mut out_fused, 1, 8, 1.0).unwrap();
        TemperatureSoftmax::forward(&input, &mut out_temp, 1, 8, 1.0).unwrap();
        SparseTopKSoftmax::forward(&input, &mut out_sparse, 1, 8, 8).unwrap();

        assert_slices_close(&out_std, &out_online, ATOL);
        assert_slices_close(&out_std, &out_fused, ATOL);
        assert_slices_close(&out_std, &out_temp, ATOL);
        assert_slices_close(&out_std, &out_sparse, ATOL);
    }
}

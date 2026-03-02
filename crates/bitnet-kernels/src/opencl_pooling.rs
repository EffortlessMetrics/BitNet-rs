//! OpenCL pooling operations for sequence/spatial data with CPU reference.
//!
//! # Overview
//!
//! This module implements pooling operations commonly used to reduce sequence
//! or spatial dimensions in neural network inference:
//!
//! - **Max pooling** — sliding-window maximum with optional index tracking.
//! - **Average pooling** — sliding-window mean with `count_include_pad` option.
//! - **Global pooling** — reduce an entire sequence to a single value (max/avg).
//! - **Adaptive pooling** — pool to a target output size regardless of input length.
//! - **Lp pooling** — Lp-norm pooling (L1 = mean-absolute, L2 = RMS-like).
//!
//! # CPU reference
//!
//! All operations have scalar CPU implementations for correctness testing and
//! non-GPU environments.
//!
//! # OpenCL kernel
//!
//! [`POOLING_CL`] contains OpenCL C source for max-pool, avg-pool, and
//! global-pool kernels suitable for GPU dispatch.

use bitnet_common::{KernelError, Result};

// ---------------------------------------------------------------------------
// Pool kind enum
// ---------------------------------------------------------------------------

/// Variant of pooling operation.
#[derive(Debug, Clone, PartialEq)]
pub enum PoolKind {
    /// Maximum value in each window.
    Max,
    /// Arithmetic mean of each window.
    Average,
    /// Weighted average (weights supplied externally).
    WeightedAverage,
    /// Lp-norm pooling: `(sum |x|^p)^(1/p)`.
    LpPool(f32),
    /// Adaptive max pooling to a target output size.
    AdaptiveMax,
    /// Adaptive average pooling to a target output size.
    AdaptiveAvg,
}

// ---------------------------------------------------------------------------
// 1-D pooling configuration
// ---------------------------------------------------------------------------

/// Configuration for 1-D pooling.
#[derive(Debug, Clone)]
pub struct Pool1dConfig {
    /// Width of the pooling window.
    pub kernel_size: usize,
    /// Step between successive windows.
    pub stride: usize,
    /// Zero-padding added to each side of the input.
    pub padding: usize,
}

impl Pool1dConfig {
    /// Create a new 1-D pooling config.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if `kernel_size` or `stride`
    /// is zero.
    pub fn new(kernel_size: usize, stride: usize, padding: usize) -> Result<Self> {
        if kernel_size == 0 || stride == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "kernel_size and stride must be > 0: kernel_size={kernel_size}, stride={stride}"
                ),
            }
            .into());
        }
        Ok(Self { kernel_size, stride, padding })
    }

    /// Compute the output length for a given input length.
    pub fn output_len(&self, input_len: usize) -> usize {
        if input_len + 2 * self.padding < self.kernel_size {
            return 0;
        }
        (input_len + 2 * self.padding - self.kernel_size) / self.stride + 1
    }
}

// ---------------------------------------------------------------------------
// MaxPool1d
// ---------------------------------------------------------------------------

/// 1-D max pooling with optional index tracking.
#[derive(Debug, Clone)]
pub struct MaxPool1d {
    /// Pooling configuration.
    pub config: Pool1dConfig,
}

impl MaxPool1d {
    /// Create a new max-pool operator.
    pub fn new(config: Pool1dConfig) -> Self {
        Self { config }
    }

    /// Run max pooling over `input` (length = `input_len`), writing to `output`.
    ///
    /// If `indices` is `Some`, records the index of the maximum element in each
    /// window.
    ///
    /// # Errors
    ///
    /// Returns an error if output buffer is too small.
    pub fn forward(
        &self,
        input: &[f32],
        output: &mut [f32],
        indices: Option<&mut [usize]>,
    ) -> Result<()> {
        let input_len = input.len();
        let out_len = self.config.output_len(input_len);
        if output.len() < out_len {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "output buffer too small: need {out_len}, got {}",
                    output.len()
                ),
            }
            .into());
        }
        if let Some(ref idx) = indices
            && idx.len() < out_len
        {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "indices buffer too small: need {out_len}, got {}",
                    idx.len()
                ),
            }
            .into());
        }
        max_pool1d_ref(input, output, indices, &self.config);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// AvgPool1d
// ---------------------------------------------------------------------------

/// 1-D average pooling.
#[derive(Debug, Clone)]
pub struct AvgPool1d {
    /// Pooling configuration.
    pub config: Pool1dConfig,
    /// If `true`, padded zeros count toward the divisor.
    pub count_include_pad: bool,
}

impl AvgPool1d {
    /// Create a new average-pool operator.
    pub fn new(config: Pool1dConfig, count_include_pad: bool) -> Self {
        Self { config, count_include_pad }
    }

    /// Run average pooling over `input`, writing to `output`.
    ///
    /// # Errors
    ///
    /// Returns an error if output buffer is too small.
    pub fn forward(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        let input_len = input.len();
        let out_len = self.config.output_len(input_len);
        if output.len() < out_len {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "output buffer too small: need {out_len}, got {}",
                    output.len()
                ),
            }
            .into());
        }
        avg_pool1d_ref(input, output, &self.config, self.count_include_pad);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// GlobalPool
// ---------------------------------------------------------------------------

/// Global pooling: reduce an entire sequence to one value per channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlobalPoolKind {
    /// Maximum over the sequence.
    Max,
    /// Arithmetic mean over the sequence.
    Average,
}

/// Global pooling operator.
#[derive(Debug, Clone)]
pub struct GlobalPool {
    /// Which reduction to apply.
    pub kind: GlobalPoolKind,
}

impl GlobalPool {
    /// Create a new global pool.
    pub fn new(kind: GlobalPoolKind) -> Self {
        Self { kind }
    }

    /// Pool `input` (shape `[batch, seq_len]`) to `output` (shape `[batch]`).
    ///
    /// # Errors
    ///
    /// Returns an error if `seq_len` is zero or buffers are mismatched.
    pub fn forward(
        &self,
        input: &[f32],
        output: &mut [f32],
        batch: usize,
        seq_len: usize,
    ) -> Result<()> {
        if seq_len == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "seq_len must be > 0 for global pooling".into(),
            }
            .into());
        }
        if input.len() < batch * seq_len {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "input length {} < batch({batch}) * seq_len({seq_len})",
                    input.len()
                ),
            }
            .into());
        }
        if output.len() < batch {
            return Err(KernelError::InvalidArguments {
                reason: format!("output length {} < batch({batch})", output.len()),
            }
            .into());
        }
        global_pool_ref(input, output, batch, seq_len, self.kind);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// AdaptivePool
// ---------------------------------------------------------------------------

/// Adaptive pooling: pool to a fixed target output size.
#[derive(Debug, Clone)]
pub struct AdaptivePool {
    /// Target output length.
    pub output_size: usize,
    /// Whether to use max or average.
    pub kind: GlobalPoolKind,
}

impl AdaptivePool {
    /// Create a new adaptive pool targeting `output_size` elements.
    ///
    /// # Errors
    ///
    /// Returns an error if `output_size` is zero.
    pub fn new(output_size: usize, kind: GlobalPoolKind) -> Result<Self> {
        if output_size == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "adaptive pool output_size must be > 0".into(),
            }
            .into());
        }
        Ok(Self { output_size, kind })
    }

    /// Pool `input` of length `input_len` to `output` of length `output_size`.
    ///
    /// # Errors
    ///
    /// Returns an error if `input_len < output_size` or buffers are too small.
    pub fn forward(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        let input_len = input.len();
        if input_len < self.output_size {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "input_len({input_len}) < output_size({})",
                    self.output_size
                ),
            }
            .into());
        }
        if output.len() < self.output_size {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "output buffer too small: need {}, got {}",
                    self.output_size,
                    output.len()
                ),
            }
            .into());
        }
        adaptive_pool_ref(input, output, input_len, self.output_size, self.kind);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// LpPool
// ---------------------------------------------------------------------------

/// Lp-norm pooling: `(sum |x_i|^p / n)^(1/p)`.
#[derive(Debug, Clone)]
pub struct LpPool {
    /// Pooling configuration.
    pub config: Pool1dConfig,
    /// The exponent `p` (must be ≥ 1).
    pub p: f32,
}

impl LpPool {
    /// Create a new Lp-pool operator.
    ///
    /// # Errors
    ///
    /// Returns an error if `p < 1.0`.
    pub fn new(config: Pool1dConfig, p: f32) -> Result<Self> {
        if p < 1.0 {
            return Err(KernelError::InvalidArguments {
                reason: format!("Lp pool exponent p must be >= 1.0, got {p}"),
            }
            .into());
        }
        Ok(Self { config, p })
    }

    /// Run Lp-norm pooling over `input`, writing to `output`.
    ///
    /// # Errors
    ///
    /// Returns an error if output buffer is too small.
    pub fn forward(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        let input_len = input.len();
        let out_len = self.config.output_len(input_len);
        if output.len() < out_len {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "output buffer too small: need {out_len}, got {}",
                    output.len()
                ),
            }
            .into());
        }
        lp_pool1d_ref(input, output, &self.config, self.p);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// PoolStats
// ---------------------------------------------------------------------------

/// Statistics for a pooling operation.
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Number of input elements.
    pub input_len: usize,
    /// Number of output elements.
    pub output_len: usize,
    /// Reduction ratio: `input_len / output_len`.
    pub reduction_ratio: f32,
    /// Throughput in elements per second (if timed externally).
    pub throughput_eps: Option<f64>,
}

impl PoolStats {
    /// Compute pool statistics.
    pub fn new(input_len: usize, output_len: usize) -> Self {
        let reduction_ratio = if output_len > 0 {
            input_len as f32 / output_len as f32
        } else {
            0.0
        };
        Self { input_len, output_len, reduction_ratio, throughput_eps: None }
    }

    /// Set throughput from elapsed time.
    pub fn with_elapsed(mut self, elapsed_secs: f64) -> Self {
        if elapsed_secs > 0.0 {
            self.throughput_eps = Some(self.input_len as f64 / elapsed_secs);
        }
        self
    }
}

// ---------------------------------------------------------------------------
// OpenCL kernel source
// ---------------------------------------------------------------------------

/// OpenCL kernel source for pooling operations.
pub const POOLING_CL: &str = r#"
// ============================================================
// Pooling kernels for 1-D sequence / spatial data
// ============================================================

/// Max pool 1-D.
/// Each work-item computes one output element.
__kernel void max_pool1d(
    __global const float* input,
    __global float* output,
    __global int* indices,
    const int input_len,
    const int kernel_size,
    const int stride,
    const int padding,
    const int write_indices
) {
    int gid = get_global_id(0);
    int start = gid * stride - padding;

    float max_val = -INFINITY;
    int max_idx = 0;

    for (int k = 0; k < kernel_size; k++) {
        int pos = start + k;
        if (pos >= 0 && pos < input_len) {
            float v = input[pos];
            if (v > max_val) {
                max_val = v;
                max_idx = pos;
            }
        }
    }
    output[gid] = max_val;
    if (write_indices) {
        indices[gid] = max_idx;
    }
}

/// Average pool 1-D.
__kernel void avg_pool1d(
    __global const float* input,
    __global float* output,
    const int input_len,
    const int kernel_size,
    const int stride,
    const int padding,
    const int count_include_pad
) {
    int gid = get_global_id(0);
    int start = gid * stride - padding;

    float sum = 0.0f;
    int count = 0;

    for (int k = 0; k < kernel_size; k++) {
        int pos = start + k;
        if (pos >= 0 && pos < input_len) {
            sum += input[pos];
            count++;
        }
    }
    int divisor = count_include_pad ? kernel_size : count;
    output[gid] = (divisor > 0) ? (sum / (float)divisor) : 0.0f;
}

/// Global max pool over a sequence.
/// One work-item per batch element.
__kernel void global_max_pool(
    __global const float* input,
    __global float* output,
    const int seq_len
) {
    int batch = get_global_id(0);
    int base = batch * seq_len;

    float max_val = -INFINITY;
    for (int i = 0; i < seq_len; i++) {
        float v = input[base + i];
        if (v > max_val) {
            max_val = v;
        }
    }
    output[batch] = max_val;
}

/// Global average pool over a sequence.
/// One work-item per batch element.
__kernel void global_avg_pool(
    __global const float* input,
    __global float* output,
    const int seq_len
) {
    int batch = get_global_id(0);
    int base = batch * seq_len;

    float sum = 0.0f;
    for (int i = 0; i < seq_len; i++) {
        sum += input[base + i];
    }
    output[batch] = sum / (float)seq_len;
}

/// Lp-norm pool 1-D.
__kernel void lp_pool1d(
    __global const float* input,
    __global float* output,
    const int input_len,
    const int kernel_size,
    const int stride,
    const int padding,
    const float p
) {
    int gid = get_global_id(0);
    int start = gid * stride - padding;

    float sum = 0.0f;
    int count = 0;

    for (int k = 0; k < kernel_size; k++) {
        int pos = start + k;
        if (pos >= 0 && pos < input_len) {
            sum += pow(fabs(input[pos]), p);
            count++;
        }
    }
    float inv_p = 1.0f / p;
    output[gid] = (count > 0) ? pow(sum / (float)count, inv_p) : 0.0f;
}
"#;

// ---------------------------------------------------------------------------
// CPU reference implementations
// ---------------------------------------------------------------------------

/// CPU reference: 1-D max pooling.
fn max_pool1d_ref(
    input: &[f32],
    output: &mut [f32],
    mut indices: Option<&mut [usize]>,
    config: &Pool1dConfig,
) {
    let input_len = input.len();
    let out_len = config.output_len(input_len);

    for (i, out_val) in output[..out_len].iter_mut().enumerate() {
        let start = (i * config.stride) as isize - config.padding as isize;
        let mut max_val = f32::NEG_INFINITY;
        let mut max_idx: usize = 0;

        for k in 0..config.kernel_size {
            let pos = start + k as isize;
            if pos >= 0 && (pos as usize) < input_len {
                let v = input[pos as usize];
                if v > max_val {
                    max_val = v;
                    max_idx = pos as usize;
                }
            }
        }
        *out_val = max_val;
        if let Some(ref mut idx) = indices {
            idx[i] = max_idx;
        }
    }
}

/// CPU reference: 1-D average pooling.
fn avg_pool1d_ref(
    input: &[f32],
    output: &mut [f32],
    config: &Pool1dConfig,
    count_include_pad: bool,
) {
    let input_len = input.len();
    let out_len = config.output_len(input_len);

    for (i, out_val) in output[..out_len].iter_mut().enumerate() {
        let start = (i * config.stride) as isize - config.padding as isize;
        let mut sum = 0.0_f32;
        let mut count = 0_usize;

        for k in 0..config.kernel_size {
            let pos = start + k as isize;
            if pos >= 0 && (pos as usize) < input_len {
                sum += input[pos as usize];
                count += 1;
            }
        }
        let divisor = if count_include_pad { config.kernel_size } else { count };
        *out_val = if divisor > 0 { sum / divisor as f32 } else { 0.0 };
    }
}

/// CPU reference: global pooling.
fn global_pool_ref(
    input: &[f32],
    output: &mut [f32],
    batch: usize,
    seq_len: usize,
    kind: GlobalPoolKind,
) {
    for (b, out_val) in output[..batch].iter_mut().enumerate() {
        let base = b * seq_len;
        let slice = &input[base..base + seq_len];
        *out_val = match kind {
            GlobalPoolKind::Max => slice.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            GlobalPoolKind::Average => {
                let sum: f32 = slice.iter().sum();
                sum / seq_len as f32
            }
        };
    }
}

/// CPU reference: adaptive pooling.
fn adaptive_pool_ref(
    input: &[f32],
    output: &mut [f32],
    input_len: usize,
    output_size: usize,
    kind: GlobalPoolKind,
) {
    for (i, out_val) in output[..output_size].iter_mut().enumerate() {
        // Compute window boundaries (same as PyTorch adaptive pooling).
        let start = (i * input_len) / output_size;
        let end = ((i + 1) * input_len) / output_size;
        let window = &input[start..end];

        *out_val = match kind {
            GlobalPoolKind::Max => window.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            GlobalPoolKind::Average => {
                let sum: f32 = window.iter().sum();
                if window.is_empty() { 0.0 } else { sum / window.len() as f32 }
            }
        };
    }
}

/// CPU reference: Lp-norm pooling.
fn lp_pool1d_ref(input: &[f32], output: &mut [f32], config: &Pool1dConfig, p: f32) {
    let input_len = input.len();
    let out_len = config.output_len(input_len);
    let inv_p = 1.0 / p;

    for (i, out_val) in output[..out_len].iter_mut().enumerate() {
        let start = (i * config.stride) as isize - config.padding as isize;
        let mut sum = 0.0_f32;
        let mut count = 0_usize;

        for k in 0..config.kernel_size {
            let pos = start + k as isize;
            if pos >= 0 && (pos as usize) < input_len {
                sum += input[pos as usize].abs().powf(p);
                count += 1;
            }
        }
        *out_val = if count > 0 { (sum / count as f32).powf(inv_p) } else { 0.0 };
    }
}

/// CPU reference: weighted average pooling.
pub fn weighted_avg_pool1d_ref(
    input: &[f32],
    weights: &[f32],
    output: &mut [f32],
    config: &Pool1dConfig,
) -> Result<()> {
    let input_len = input.len();
    let out_len = config.output_len(input_len);
    if weights.len() < config.kernel_size {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "weights length {} < kernel_size {}",
                weights.len(),
                config.kernel_size
            ),
        }
        .into());
    }
    if output.len() < out_len {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "output buffer too small: need {out_len}, got {}",
                output.len()
            ),
        }
        .into());
    }

    for (i, out_val) in output[..out_len].iter_mut().enumerate() {
        let start = (i * config.stride) as isize - config.padding as isize;
        let mut wsum = 0.0_f32;
        let mut wdiv = 0.0_f32;

        for (k, &w) in weights[..config.kernel_size].iter().enumerate() {
            let pos = start + k as isize;
            if pos >= 0 && (pos as usize) < input_len {
                wsum += input[pos as usize] * w;
                wdiv += w;
            }
        }
        *out_val = if wdiv > 0.0 { wsum / wdiv } else { 0.0 };
    }
    Ok(())
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-6;

    fn assert_near(actual: f32, expected: f32, tol: f32, label: &str) {
        assert!(
            (actual - expected).abs() <= tol,
            "{label}: expected {expected}, got {actual} (diff={})",
            (actual - expected).abs()
        );
    }

    // ── PoolKind enum ────────────────────────────────────────────

    #[test]
    fn test_pool_kind_variants() {
        let kinds = [
            PoolKind::Max,
            PoolKind::Average,
            PoolKind::WeightedAverage,
            PoolKind::LpPool(2.0),
            PoolKind::AdaptiveMax,
            PoolKind::AdaptiveAvg,
        ];
        assert_eq!(kinds.len(), 6);
    }

    #[test]
    fn test_pool_kind_lp_equality() {
        assert_eq!(PoolKind::LpPool(2.0), PoolKind::LpPool(2.0));
        assert_ne!(PoolKind::LpPool(1.0), PoolKind::LpPool(2.0));
    }

    #[test]
    fn test_pool_kind_debug() {
        let s = format!("{:?}", PoolKind::Max);
        assert!(s.contains("Max"));
    }

    // ── Pool1dConfig ─────────────────────────────────────────────

    #[test]
    fn test_config_output_len_basic() {
        // kernel=3, stride=1, pad=0, input=5 → (5-3)/1+1 = 3
        let cfg = Pool1dConfig::new(3, 1, 0).unwrap();
        assert_eq!(cfg.output_len(5), 3);
    }

    #[test]
    fn test_config_output_len_with_stride() {
        // kernel=3, stride=2, pad=0, input=7 → (7-3)/2+1 = 3
        let cfg = Pool1dConfig::new(3, 2, 0).unwrap();
        assert_eq!(cfg.output_len(7), 3);
    }

    #[test]
    fn test_config_output_len_with_padding() {
        // kernel=3, stride=1, pad=1, input=5 → (5+2-3)/1+1 = 5
        let cfg = Pool1dConfig::new(3, 1, 1).unwrap();
        assert_eq!(cfg.output_len(5), 5);
    }

    #[test]
    fn test_config_output_len_kernel_eq_input() {
        // kernel=5, stride=1, pad=0, input=5 → 1
        let cfg = Pool1dConfig::new(5, 1, 0).unwrap();
        assert_eq!(cfg.output_len(5), 1);
    }

    #[test]
    fn test_config_output_len_input_smaller_than_kernel() {
        // kernel=5, stride=1, pad=0, input=3 → 0
        let cfg = Pool1dConfig::new(5, 1, 0).unwrap();
        assert_eq!(cfg.output_len(3), 0);
    }

    #[test]
    fn test_config_kernel_size_1() {
        // kernel=1, stride=1, pad=0, input=5 → 5
        let cfg = Pool1dConfig::new(1, 1, 0).unwrap();
        assert_eq!(cfg.output_len(5), 5);
    }

    #[test]
    fn test_config_zero_kernel_error() {
        assert!(Pool1dConfig::new(0, 1, 0).is_err());
    }

    #[test]
    fn test_config_zero_stride_error() {
        assert!(Pool1dConfig::new(3, 0, 0).is_err());
    }

    #[test]
    fn test_config_non_overlapping() {
        // kernel=2, stride=3 → non-overlapping with gaps
        let cfg = Pool1dConfig::new(2, 3, 0).unwrap();
        // input=9 → (9-2)/3+1 = 3
        assert_eq!(cfg.output_len(9), 3);
    }

    // ── MaxPool1d correctness ────────────────────────────────────

    #[test]
    fn test_maxpool_basic() {
        let input = [1.0, 3.0, 2.0, 5.0, 4.0];
        let cfg = Pool1dConfig::new(3, 1, 0).unwrap();
        let pool = MaxPool1d::new(cfg);
        let mut output = vec![0.0; 3];
        pool.forward(&input, &mut output, None).unwrap();
        assert_eq!(output, vec![3.0, 5.0, 5.0]);
    }

    #[test]
    fn test_maxpool_with_indices() {
        let input = [1.0, 3.0, 2.0, 5.0, 4.0];
        let cfg = Pool1dConfig::new(3, 1, 0).unwrap();
        let pool = MaxPool1d::new(cfg);
        let mut output = vec![0.0; 3];
        let mut indices = vec![0_usize; 3];
        pool.forward(&input, &mut output, Some(&mut indices)).unwrap();
        assert_eq!(output, vec![3.0, 5.0, 5.0]);
        assert_eq!(indices, vec![1, 3, 3]);
    }

    #[test]
    fn test_maxpool_stride2() {
        let input = [1.0, 5.0, 3.0, 7.0, 2.0, 6.0];
        // kernel=2, stride=2 → (6-2)/2+1 = 3
        let cfg = Pool1dConfig::new(2, 2, 0).unwrap();
        let pool = MaxPool1d::new(cfg);
        let mut output = vec![0.0; 3];
        pool.forward(&input, &mut output, None).unwrap();
        assert_eq!(output, vec![5.0, 7.0, 6.0]);
    }

    #[test]
    fn test_maxpool_kernel_size_1() {
        let input = [2.0, 4.0, 1.0, 3.0];
        let cfg = Pool1dConfig::new(1, 1, 0).unwrap();
        let pool = MaxPool1d::new(cfg);
        let mut output = vec![0.0; 4];
        pool.forward(&input, &mut output, None).unwrap();
        assert_eq!(output, vec![2.0, 4.0, 1.0, 3.0]);
    }

    #[test]
    fn test_maxpool_with_padding() {
        // input=[1,2,3], kernel=3, stride=1, pad=1 → 3 outputs
        // window[-1,0,1] → max(0*, 1, 2) = 2  (* padded zero)
        // window[0,1,2]  → max(1, 2, 3) = 3
        // window[1,2,3]  → max(2, 3, 0*) = 3
        let input = [1.0, 2.0, 3.0];
        let cfg = Pool1dConfig::new(3, 1, 1).unwrap();
        let pool = MaxPool1d::new(cfg);
        let mut output = vec![0.0; 3];
        pool.forward(&input, &mut output, None).unwrap();
        assert_eq!(output, vec![2.0, 3.0, 3.0]);
    }

    #[test]
    fn test_maxpool_negative_values() {
        let input = [-5.0, -3.0, -7.0, -1.0, -4.0];
        let cfg = Pool1dConfig::new(3, 1, 0).unwrap();
        let pool = MaxPool1d::new(cfg);
        let mut output = vec![0.0; 3];
        pool.forward(&input, &mut output, None).unwrap();
        assert_eq!(output, vec![-3.0, -1.0, -1.0]);
    }

    #[test]
    fn test_maxpool_single_element_window() {
        let input = [42.0];
        let cfg = Pool1dConfig::new(1, 1, 0).unwrap();
        let pool = MaxPool1d::new(cfg);
        let mut output = vec![0.0; 1];
        pool.forward(&input, &mut output, None).unwrap();
        assert_eq!(output, vec![42.0]);
    }

    #[test]
    fn test_maxpool_output_too_small() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let cfg = Pool1dConfig::new(3, 1, 0).unwrap();
        let pool = MaxPool1d::new(cfg);
        let mut output = vec![0.0; 1]; // need 3
        assert!(pool.forward(&input, &mut output, None).is_err());
    }

    #[test]
    fn test_maxpool_indices_too_small() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let cfg = Pool1dConfig::new(3, 1, 0).unwrap();
        let pool = MaxPool1d::new(cfg);
        let mut output = vec![0.0; 3];
        let mut indices = vec![0_usize; 1]; // need 3
        assert!(pool.forward(&input, &mut output, Some(&mut indices)).is_err());
    }

    #[test]
    fn test_maxpool_non_overlapping() {
        // kernel=2, stride=3 → windows: [0,1], [3,4], [6,7]
        let input = [1.0, 9.0, 0.0, 8.0, 2.0, 0.0, 7.0, 3.0, 0.0];
        let cfg = Pool1dConfig::new(2, 3, 0).unwrap();
        let pool = MaxPool1d::new(cfg);
        let out_len = pool.config.output_len(input.len());
        let mut output = vec![0.0; out_len];
        pool.forward(&input, &mut output, None).unwrap();
        assert_eq!(output, vec![9.0, 8.0, 7.0]);
    }

    // ── AvgPool1d correctness ────────────────────────────────────

    #[test]
    fn test_avgpool_basic() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let cfg = Pool1dConfig::new(3, 1, 0).unwrap();
        let pool = AvgPool1d::new(cfg, false);
        let mut output = vec![0.0; 3];
        pool.forward(&input, &mut output).unwrap();
        assert_near(output[0], 2.0, EPS, "avg[0]");
        assert_near(output[1], 3.0, EPS, "avg[1]");
        assert_near(output[2], 4.0, EPS, "avg[2]");
    }

    #[test]
    fn test_avgpool_stride2() {
        let input = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
        let cfg = Pool1dConfig::new(2, 2, 0).unwrap();
        let pool = AvgPool1d::new(cfg, false);
        let mut output = vec![0.0; 3];
        pool.forward(&input, &mut output).unwrap();
        assert_near(output[0], 3.0, EPS, "avg[0]");
        assert_near(output[1], 7.0, EPS, "avg[1]");
        assert_near(output[2], 11.0, EPS, "avg[2]");
    }

    #[test]
    fn test_avgpool_count_include_pad_true() {
        // pad=1, kernel=3 → first window is [pad, 1, 2]
        // With count_include_pad: (0+1+2)/3 = 1.0
        let input = [1.0, 2.0, 3.0];
        let cfg = Pool1dConfig::new(3, 1, 1).unwrap();
        let pool = AvgPool1d::new(cfg, true);
        let mut output = vec![0.0; 3];
        pool.forward(&input, &mut output).unwrap();
        assert_near(output[0], (0.0 + 1.0 + 2.0) / 3.0, EPS, "avg_inc[0]");
        assert_near(output[1], (1.0 + 2.0 + 3.0) / 3.0, EPS, "avg_inc[1]");
        assert_near(output[2], (2.0 + 3.0 + 0.0) / 3.0, EPS, "avg_inc[2]");
    }

    #[test]
    fn test_avgpool_count_exclude_pad() {
        // Same setup but count_include_pad=false → first window divides by 2
        let input = [1.0, 2.0, 3.0];
        let cfg = Pool1dConfig::new(3, 1, 1).unwrap();
        let pool = AvgPool1d::new(cfg, false);
        let mut output = vec![0.0; 3];
        pool.forward(&input, &mut output).unwrap();
        assert_near(output[0], (1.0 + 2.0) / 2.0, EPS, "avg_exc[0]");
        assert_near(output[1], (1.0 + 2.0 + 3.0) / 3.0, EPS, "avg_exc[1]");
        assert_near(output[2], (2.0 + 3.0) / 2.0, EPS, "avg_exc[2]");
    }

    #[test]
    fn test_avgpool_kernel_size_1() {
        let input = [10.0, 20.0, 30.0];
        let cfg = Pool1dConfig::new(1, 1, 0).unwrap();
        let pool = AvgPool1d::new(cfg, false);
        let mut output = vec![0.0; 3];
        pool.forward(&input, &mut output).unwrap();
        assert_eq!(output, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_avgpool_output_too_small() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let cfg = Pool1dConfig::new(3, 1, 0).unwrap();
        let pool = AvgPool1d::new(cfg, false);
        let mut output = vec![0.0; 1];
        assert!(pool.forward(&input, &mut output).is_err());
    }

    // ── GlobalPool ───────────────────────────────────────────────

    #[test]
    fn test_global_max_single_batch() {
        let input = [1.0, 5.0, 3.0, 7.0, 2.0];
        let gp = GlobalPool::new(GlobalPoolKind::Max);
        let mut output = [0.0_f32; 1];
        gp.forward(&input, &mut output, 1, 5).unwrap();
        assert_eq!(output[0], 7.0);
    }

    #[test]
    fn test_global_avg_single_batch() {
        let input = [2.0, 4.0, 6.0, 8.0, 10.0];
        let gp = GlobalPool::new(GlobalPoolKind::Average);
        let mut output = [0.0_f32; 1];
        gp.forward(&input, &mut output, 1, 5).unwrap();
        assert_near(output[0], 6.0, EPS, "global_avg");
    }

    #[test]
    fn test_global_max_multi_batch() {
        // batch=2, seq=3
        let input = [1.0, 9.0, 3.0, 7.0, 2.0, 5.0];
        let gp = GlobalPool::new(GlobalPoolKind::Max);
        let mut output = [0.0_f32; 2];
        gp.forward(&input, &mut output, 2, 3).unwrap();
        assert_eq!(output[0], 9.0);
        assert_eq!(output[1], 7.0);
    }

    #[test]
    fn test_global_avg_multi_batch() {
        let input = [3.0, 6.0, 9.0, 10.0, 20.0, 30.0];
        let gp = GlobalPool::new(GlobalPoolKind::Average);
        let mut output = [0.0_f32; 2];
        gp.forward(&input, &mut output, 2, 3).unwrap();
        assert_near(output[0], 6.0, EPS, "global_avg[0]");
        assert_near(output[1], 20.0, EPS, "global_avg[1]");
    }

    #[test]
    fn test_global_pool_seq_len_zero_error() {
        let gp = GlobalPool::new(GlobalPoolKind::Max);
        let mut output = [0.0_f32; 1];
        assert!(gp.forward(&[], &mut output, 1, 0).is_err());
    }

    #[test]
    fn test_global_pool_input_too_small() {
        let gp = GlobalPool::new(GlobalPoolKind::Max);
        let mut output = [0.0_f32; 2];
        assert!(gp.forward(&[1.0, 2.0, 3.0], &mut output, 2, 3).is_err());
    }

    #[test]
    fn test_global_pool_output_too_small() {
        let gp = GlobalPool::new(GlobalPoolKind::Max);
        let mut output = [0.0_f32; 1];
        assert!(gp.forward(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &mut output, 2, 3).is_err());
    }

    #[test]
    fn test_global_max_negative_values() {
        let input = [-5.0, -3.0, -8.0, -1.0];
        let gp = GlobalPool::new(GlobalPoolKind::Max);
        let mut output = [0.0_f32; 1];
        gp.forward(&input, &mut output, 1, 4).unwrap();
        assert_eq!(output[0], -1.0);
    }

    #[test]
    fn test_global_avg_single_element() {
        let input = [42.0];
        let gp = GlobalPool::new(GlobalPoolKind::Average);
        let mut output = [0.0_f32; 1];
        gp.forward(&input, &mut output, 1, 1).unwrap();
        assert_near(output[0], 42.0, EPS, "global_avg_single");
    }

    // ── AdaptivePool ─────────────────────────────────────────────

    #[test]
    fn test_adaptive_max_halve() {
        // input=6, output=3 → windows [0,1], [2,3], [4,5]
        let input = [1.0, 4.0, 2.0, 6.0, 3.0, 5.0];
        let ap = AdaptivePool::new(3, GlobalPoolKind::Max).unwrap();
        let mut output = vec![0.0; 3];
        ap.forward(&input, &mut output).unwrap();
        assert_eq!(output, vec![4.0, 6.0, 5.0]);
    }

    #[test]
    fn test_adaptive_avg_halve() {
        let input = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
        let ap = AdaptivePool::new(3, GlobalPoolKind::Average).unwrap();
        let mut output = vec![0.0; 3];
        ap.forward(&input, &mut output).unwrap();
        assert_near(output[0], 3.0, EPS, "adap_avg[0]");
        assert_near(output[1], 7.0, EPS, "adap_avg[1]");
        assert_near(output[2], 11.0, EPS, "adap_avg[2]");
    }

    #[test]
    fn test_adaptive_output_size_1() {
        // Reduce to single element = global pool
        let input = [1.0, 5.0, 3.0, 7.0, 2.0];
        let ap = AdaptivePool::new(1, GlobalPoolKind::Max).unwrap();
        let mut output = vec![0.0; 1];
        ap.forward(&input, &mut output).unwrap();
        assert_eq!(output[0], 7.0);
    }

    #[test]
    fn test_adaptive_output_eq_input() {
        // output_size = input_len → each window has 1 element
        let input = [1.0, 2.0, 3.0, 4.0];
        let ap = AdaptivePool::new(4, GlobalPoolKind::Average).unwrap();
        let mut output = vec![0.0; 4];
        ap.forward(&input, &mut output).unwrap();
        assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_adaptive_output_size_zero_error() {
        assert!(AdaptivePool::new(0, GlobalPoolKind::Max).is_err());
    }

    #[test]
    fn test_adaptive_input_smaller_than_output() {
        let input = [1.0, 2.0];
        let ap = AdaptivePool::new(5, GlobalPoolKind::Max).unwrap();
        let mut output = vec![0.0; 5];
        assert!(ap.forward(&input, &mut output).is_err());
    }

    #[test]
    fn test_adaptive_output_buffer_too_small() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let ap = AdaptivePool::new(3, GlobalPoolKind::Max).unwrap();
        let mut output = vec![0.0; 2];
        assert!(ap.forward(&input, &mut output).is_err());
    }

    #[test]
    fn test_adaptive_uneven_split() {
        // input=7, output=3 → windows: [0,2), [2,4), [4,7)
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let ap = AdaptivePool::new(3, GlobalPoolKind::Max).unwrap();
        let mut output = vec![0.0; 3];
        ap.forward(&input, &mut output).unwrap();
        assert_eq!(output[0], 2.0); // max(1,2)
        assert_eq!(output[1], 4.0); // max(3,4)
        assert_eq!(output[2], 7.0); // max(5,6,7)
    }

    // ── LpPool ───────────────────────────────────────────────────

    #[test]
    fn test_lp_pool_l1() {
        // L1: (sum|x|/n)^1 = mean of absolute values
        let input = [1.0, -2.0, 3.0, -4.0, 5.0];
        let cfg = Pool1dConfig::new(3, 1, 0).unwrap();
        let pool = LpPool::new(cfg, 1.0).unwrap();
        let mut output = vec![0.0; 3];
        pool.forward(&input, &mut output).unwrap();
        // window[0,1,2]: (1+2+3)/3 = 2.0
        assert_near(output[0], 2.0, EPS, "lp1[0]");
        // window[1,2,3]: (2+3+4)/3 = 3.0
        assert_near(output[1], 3.0, EPS, "lp1[1]");
        // window[2,3,4]: (3+4+5)/3 = 4.0
        assert_near(output[2], 4.0, EPS, "lp1[2]");
    }

    #[test]
    fn test_lp_pool_l2() {
        // L2: (sum x^2 / n)^(1/2) = RMS
        let input = [3.0, 4.0];
        let cfg = Pool1dConfig::new(2, 1, 0).unwrap();
        let pool = LpPool::new(cfg, 2.0).unwrap();
        let mut output = vec![0.0; 1];
        pool.forward(&input, &mut output).unwrap();
        let expected = ((9.0 + 16.0) / 2.0_f32).sqrt();
        assert_near(output[0], expected, EPS, "lp2");
    }

    #[test]
    fn test_lp_pool_l2_uniform() {
        // All same values → Lp = that value for any p
        let input = [5.0, 5.0, 5.0, 5.0];
        let cfg = Pool1dConfig::new(2, 2, 0).unwrap();
        let pool = LpPool::new(cfg, 2.0).unwrap();
        let mut output = vec![0.0; 2];
        pool.forward(&input, &mut output).unwrap();
        assert_near(output[0], 5.0, EPS, "lp2_uniform[0]");
        assert_near(output[1], 5.0, EPS, "lp2_uniform[1]");
    }

    #[test]
    fn test_lp_pool_invalid_p() {
        let cfg = Pool1dConfig::new(3, 1, 0).unwrap();
        assert!(LpPool::new(cfg, 0.5).is_err());
    }

    #[test]
    fn test_lp_pool_output_too_small() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let cfg = Pool1dConfig::new(3, 1, 0).unwrap();
        let pool = LpPool::new(cfg, 2.0).unwrap();
        let mut output = vec![0.0; 1];
        assert!(pool.forward(&input, &mut output).is_err());
    }

    #[test]
    fn test_lp_pool_large_p() {
        // Large p → approaches max (use small values to avoid f32 overflow)
        let input = [0.1, 1.0, 0.2];
        let cfg = Pool1dConfig::new(3, 1, 0).unwrap();
        let pool = LpPool::new(cfg, 20.0).unwrap();
        let mut output = vec![0.0; 1];
        pool.forward(&input, &mut output).unwrap();
        // Should be close to 1.0 (the max)
        assert!((output[0] - 1.0).abs() < 0.1, "large p should approach max, got {}", output[0]);
    }

    // ── WeightedAvg ──────────────────────────────────────────────

    #[test]
    fn test_weighted_avg_uniform_weights() {
        // Uniform weights = regular average
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let weights = [1.0, 1.0, 1.0];
        let cfg = Pool1dConfig::new(3, 1, 0).unwrap();
        let mut output = vec![0.0; 3];
        weighted_avg_pool1d_ref(&input, &weights, &mut output, &cfg).unwrap();
        assert_near(output[0], 2.0, EPS, "wavg[0]");
        assert_near(output[1], 3.0, EPS, "wavg[1]");
        assert_near(output[2], 4.0, EPS, "wavg[2]");
    }

    #[test]
    fn test_weighted_avg_nonuniform() {
        // weights=[1,2,1], input=[1,2,3] → (1*1 + 2*2 + 1*3) / (1+2+1) = 8/4 = 2.0
        let input = [1.0, 2.0, 3.0];
        let weights = [1.0, 2.0, 1.0];
        let cfg = Pool1dConfig::new(3, 1, 0).unwrap();
        let mut output = vec![0.0; 1];
        weighted_avg_pool1d_ref(&input, &weights, &mut output, &cfg).unwrap();
        assert_near(output[0], 2.0, EPS, "wavg_nonuniform");
    }

    #[test]
    fn test_weighted_avg_weights_too_short() {
        let input = [1.0, 2.0, 3.0];
        let weights = [1.0]; // kernel=3, need 3 weights
        let cfg = Pool1dConfig::new(3, 1, 0).unwrap();
        let mut output = vec![0.0; 1];
        assert!(weighted_avg_pool1d_ref(&input, &weights, &mut output, &cfg).is_err());
    }

    #[test]
    fn test_weighted_avg_output_too_small() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let weights = [1.0, 1.0, 1.0];
        let cfg = Pool1dConfig::new(3, 1, 0).unwrap();
        let mut output = vec![0.0; 1]; // need 3
        assert!(weighted_avg_pool1d_ref(&input, &weights, &mut output, &cfg).is_err());
    }

    // ── PoolStats ────────────────────────────────────────────────

    #[test]
    fn test_pool_stats_basic() {
        let stats = PoolStats::new(100, 50);
        assert_eq!(stats.input_len, 100);
        assert_eq!(stats.output_len, 50);
        assert_near(stats.reduction_ratio, 2.0, EPS, "reduction_ratio");
        assert!(stats.throughput_eps.is_none());
    }

    #[test]
    fn test_pool_stats_with_elapsed() {
        let stats = PoolStats::new(1000, 100).with_elapsed(0.5);
        assert!(stats.throughput_eps.is_some());
        assert_near(stats.throughput_eps.unwrap() as f32, 2000.0, 1.0, "throughput");
    }

    #[test]
    fn test_pool_stats_zero_output() {
        let stats = PoolStats::new(100, 0);
        assert_near(stats.reduction_ratio, 0.0, EPS, "zero_output_ratio");
    }

    #[test]
    fn test_pool_stats_zero_elapsed() {
        let stats = PoolStats::new(100, 10).with_elapsed(0.0);
        assert!(stats.throughput_eps.is_none());
    }

    // ── OpenCL kernel source ─────────────────────────────────────

    #[test]
    fn test_cl_source_not_empty() {
        assert!(!POOLING_CL.is_empty());
    }

    #[test]
    fn test_cl_source_has_max_pool() {
        assert!(POOLING_CL.contains("__kernel void max_pool1d"));
    }

    #[test]
    fn test_cl_source_has_avg_pool() {
        assert!(POOLING_CL.contains("__kernel void avg_pool1d"));
    }

    #[test]
    fn test_cl_source_has_global_max() {
        assert!(POOLING_CL.contains("__kernel void global_max_pool"));
    }

    #[test]
    fn test_cl_source_has_global_avg() {
        assert!(POOLING_CL.contains("__kernel void global_avg_pool"));
    }

    #[test]
    fn test_cl_source_has_lp_pool() {
        assert!(POOLING_CL.contains("__kernel void lp_pool1d"));
    }

    #[test]
    fn test_cl_source_has_get_global_id() {
        assert!(POOLING_CL.contains("get_global_id"));
    }

    // ── Property tests ───────────────────────────────────────────

    #[test]
    fn test_property_maxpool_leq_max_input() {
        // max_pool output element must be ≤ max(input)
        let input = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let max_in = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let cfg = Pool1dConfig::new(3, 1, 0).unwrap();
        let pool = MaxPool1d::new(cfg);
        let out_len = pool.config.output_len(input.len());
        let mut output = vec![0.0; out_len];
        pool.forward(&input, &mut output, None).unwrap();
        for (i, &v) in output.iter().enumerate() {
            assert!(v <= max_in, "max_pool[{i}]={v} > max(input)={max_in}");
        }
    }

    #[test]
    fn test_property_maxpool_geq_min_window() {
        // Each output ≥ elements actually in its window
        let input = [3.0, 1.0, 4.0, 1.0, 5.0];
        let cfg = Pool1dConfig::new(3, 1, 0).unwrap();
        let pool = MaxPool1d::new(cfg);
        let mut output = vec![0.0; 3];
        pool.forward(&input, &mut output, None).unwrap();
        for (i, &v) in output.iter().enumerate() {
            let window = &input[i..i + 3];
            let min_w = window.iter().copied().fold(f32::INFINITY, f32::min);
            assert!(v >= min_w, "max_pool[{i}]={v} < min(window)={min_w}");
        }
    }

    #[test]
    fn test_property_avgpool_in_range() {
        // avg_pool output must be between min and max of input
        let input = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let min_in = input.iter().copied().fold(f32::INFINITY, f32::min);
        let max_in = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let cfg = Pool1dConfig::new(3, 1, 0).unwrap();
        let pool = AvgPool1d::new(cfg, false);
        let out_len = pool.config.output_len(input.len());
        let mut output = vec![0.0; out_len];
        pool.forward(&input, &mut output).unwrap();
        for (i, &v) in output.iter().enumerate() {
            assert!(
                v >= min_in && v <= max_in,
                "avg_pool[{i}]={v} not in [{min_in}, {max_in}]"
            );
        }
    }

    #[test]
    fn test_property_adaptive_output_size() {
        let input: Vec<f32> = (0..20).map(|i| i as f32).collect();
        for target in [1, 3, 5, 10, 20] {
            let ap = AdaptivePool::new(target, GlobalPoolKind::Average).unwrap();
            let mut output = vec![0.0; target];
            ap.forward(&input, &mut output).unwrap();
            assert_eq!(output.len(), target);
        }
    }

    #[test]
    fn test_property_global_max_eq_max() {
        let input = [2.0, 8.0, 1.0, 6.0, 3.0];
        let expected_max = 8.0;
        let gp = GlobalPool::new(GlobalPoolKind::Max);
        let mut output = [0.0_f32; 1];
        gp.forward(&input, &mut output, 1, 5).unwrap();
        assert_eq!(output[0], expected_max);
    }

    #[test]
    fn test_property_global_avg_eq_mean() {
        let input = [2.0, 4.0, 6.0, 8.0, 10.0];
        let expected_mean: f32 = input.iter().sum::<f32>() / input.len() as f32;
        let gp = GlobalPool::new(GlobalPoolKind::Average);
        let mut output = [0.0_f32; 1];
        gp.forward(&input, &mut output, 1, 5).unwrap();
        assert_near(output[0], expected_mean, EPS, "global_mean_property");
    }

    #[test]
    fn test_property_pool_stats_ratio() {
        for (inp, out) in [(100, 25), (64, 32), (10, 1)] {
            let stats = PoolStats::new(inp, out);
            let expected = inp as f32 / out as f32;
            assert_near(stats.reduction_ratio, expected, EPS, &format!("{inp}/{out}"));
        }
    }

    // ── Stride > kernel_size (non-overlapping with gaps) ─────────

    #[test]
    fn test_maxpool_stride_gt_kernel() {
        // kernel=2, stride=4 → windows at [0,1], [4,5]
        let input = [1.0, 9.0, 0.0, 0.0, 8.0, 2.0, 0.0, 0.0];
        let cfg = Pool1dConfig::new(2, 4, 0).unwrap();
        let pool = MaxPool1d::new(cfg);
        let out_len = pool.config.output_len(input.len());
        let mut output = vec![0.0; out_len];
        pool.forward(&input, &mut output, None).unwrap();
        assert_eq!(output, vec![9.0, 8.0]);
    }

    #[test]
    fn test_avgpool_stride_gt_kernel() {
        let input = [2.0, 4.0, 0.0, 0.0, 6.0, 8.0, 0.0, 0.0];
        let cfg = Pool1dConfig::new(2, 4, 0).unwrap();
        let pool = AvgPool1d::new(cfg, false);
        let out_len = pool.config.output_len(input.len());
        let mut output = vec![0.0; out_len];
        pool.forward(&input, &mut output).unwrap();
        assert_near(output[0], 3.0, EPS, "avg_gap[0]");
        assert_near(output[1], 7.0, EPS, "avg_gap[1]");
    }

    // ── Edge case: empty output ──────────────────────────────────

    #[test]
    fn test_maxpool_empty_output() {
        // input_len(2) < kernel_size(5) → 0 output
        let input = [1.0, 2.0];
        let cfg = Pool1dConfig::new(5, 1, 0).unwrap();
        let pool = MaxPool1d::new(cfg);
        let mut output = vec![];
        pool.forward(&input, &mut output, None).unwrap();
        assert!(output.is_empty());
    }

    #[test]
    fn test_avgpool_empty_output() {
        let input = [1.0, 2.0];
        let cfg = Pool1dConfig::new(5, 1, 0).unwrap();
        let pool = AvgPool1d::new(cfg, false);
        let mut output = vec![];
        pool.forward(&input, &mut output).unwrap();
        assert!(output.is_empty());
    }

    // ── Misc ─────────────────────────────────────────────────────

    #[test]
    fn test_maxpool_all_equal() {
        let input = [7.0; 6];
        let cfg = Pool1dConfig::new(3, 1, 0).unwrap();
        let pool = MaxPool1d::new(cfg);
        let mut output = vec![0.0; 4];
        pool.forward(&input, &mut output, None).unwrap();
        assert!(output.iter().all(|&v| v == 7.0));
    }

    #[test]
    fn test_avgpool_all_equal() {
        let input = [7.0; 6];
        let cfg = Pool1dConfig::new(3, 1, 0).unwrap();
        let pool = AvgPool1d::new(cfg, false);
        let mut output = vec![0.0; 4];
        pool.forward(&input, &mut output).unwrap();
        for &v in &output {
            assert_near(v, 7.0, EPS, "avg_equal");
        }
    }

    #[test]
    fn test_weighted_avg_with_padding() {
        // kernel=3, stride=1, pad=1, input=[4,6,8], weights=[1,2,1]
        // window[-1,0,1]: val(0,4,6), w(1,2,1) → (0*1 + 4*2 + 6*1)/(1+2+1) = 14/4 = 3.5
        //   but padded positions don't contribute: (4*2 + 6*1)/(2+1) = 14/3 ≈ 4.667
        let input = [4.0, 6.0, 8.0];
        let weights = [1.0, 2.0, 1.0];
        let cfg = Pool1dConfig::new(3, 1, 1).unwrap();
        let mut output = vec![0.0; 3];
        weighted_avg_pool1d_ref(&input, &weights, &mut output, &cfg).unwrap();
        // First window: positions -1(pad),0,1 → valid: 0,1 → w[1]*4 + w[2]*6 = 8+6=14, wdiv=3
        assert_near(output[0], 14.0 / 3.0, EPS, "wpad[0]");
        // Middle: all valid → w[0]*4+w[1]*6+w[2]*8 = 4+12+8=24, wdiv=4
        assert_near(output[1], 24.0 / 4.0, EPS, "wpad[1]");
        // Last: positions 1,2,3(pad) → valid: 1,2 → w[0]*6+w[1]*8 = 6+16=22, wdiv=3
        assert_near(output[2], 22.0 / 3.0, EPS, "wpad[2]");
    }

    #[test]
    fn test_adaptive_max_single_element_output() {
        // output=1 → global max
        let input = [10.0, 2.0, 8.0, 1.0, 9.0];
        let ap = AdaptivePool::new(1, GlobalPoolKind::Max).unwrap();
        let mut output = vec![0.0; 1];
        ap.forward(&input, &mut output).unwrap();
        assert_eq!(output[0], 10.0);
    }

    #[test]
    fn test_lp_pool_p1_with_stride() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let cfg = Pool1dConfig::new(2, 2, 0).unwrap();
        let pool = LpPool::new(cfg, 1.0).unwrap();
        let mut output = vec![0.0; 3];
        pool.forward(&input, &mut output).unwrap();
        assert_near(output[0], 1.5, EPS, "lp1_s2[0]");
        assert_near(output[1], 3.5, EPS, "lp1_s2[1]");
        assert_near(output[2], 5.5, EPS, "lp1_s2[2]");
    }
}

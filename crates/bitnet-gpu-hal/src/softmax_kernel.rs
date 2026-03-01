//! Module stub - implementation pending merge from feature branch
//! Numerically stable softmax kernel implementations for GPU inference.
//!
//! Provides configurable softmax variants including standard, log, sparse,
//! online, and flash softmax with temperature scaling, causal masking,
//! and top-k filtering.

use std::time::Instant;

// ---------------------------------------------------------------------------
// Configuration & Types
// ---------------------------------------------------------------------------

/// Configuration for softmax computation.
#[derive(Debug, Clone)]
pub struct SoftmaxConfig {
    /// Temperature scaling factor (applied before softmax).
    pub temperature: f64,
    /// Small epsilon for numerical stability.
    pub epsilon: f64,
    /// Dimension of the input vectors.
    pub dimension: usize,
    /// Whether to apply a causal (upper-triangular) mask.
    pub causal_mask: bool,
}

impl Default for SoftmaxConfig {
    fn default() -> Self {
        Self { temperature: 1.0, epsilon: 1e-10, dimension: 0, causal_mask: false }
    }
}

/// Enum of supported softmax variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoftmaxType {
    /// Standard softmax: `exp(x_i) / Σ exp(x_j)`.
    Standard,
    /// Log-softmax: `x_i - log(Σ exp(x_j))`, more numerically stable for NLL.
    LogSoftmax,
    /// Sparse softmax: zeros out values below a threshold after softmax.
    SparseSoftmax,
    /// Online softmax: single-pass streaming algorithm.
    OnlineSoftmax,
    /// Flash softmax: tiled computation for memory efficiency.
    FlashSoftmax,
}

// ---------------------------------------------------------------------------
// Numerical Stabilizer
// ---------------------------------------------------------------------------

/// Utilities for numerically stable softmax computation.
pub struct NumericalStabilizer;

impl NumericalStabilizer {
    /// Subtract the maximum value from each element for overflow prevention.
    pub fn subtract_max(input: &[f64]) -> Vec<f64> {
        let max_val = input.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        input.iter().map(|&x| x - max_val).collect()
    }

    /// Compute log-sum-exp in a numerically stable way.
    ///
    /// `log(Σ exp(x_i)) = max(x) + log(Σ exp(x_i - max(x)))`
    pub fn log_sum_exp(input: &[f64]) -> f64 {
        if input.is_empty() {
            return f64::NEG_INFINITY;
        }
        let max_val = input.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if max_val == f64::NEG_INFINITY {
            return f64::NEG_INFINITY;
        }
        let sum_exp: f64 = input.iter().map(|&x| (x - max_val).exp()).sum();
        max_val + sum_exp.ln()
    }
}

// ---------------------------------------------------------------------------
// Core Softmax Kernel
// ---------------------------------------------------------------------------

/// Core softmax computation engine.
pub struct SoftmaxKernel;

impl SoftmaxKernel {
    /// Compute standard softmax: subtract max → exp → sum → divide.
    pub fn compute(input: &[f64], config: &SoftmaxConfig) -> Vec<f64> {
        if input.is_empty() {
            return Vec::new();
        }
        let shifted = NumericalStabilizer::subtract_max(input);
        let exps: Vec<f64> = shifted.iter().map(|&x| x.exp()).collect();
        let sum: f64 = exps.iter().sum::<f64>() + config.epsilon;
        exps.iter().map(|&e| e / sum).collect()
    }

    /// Compute log-softmax: `x_i - log(Σ exp(x_j))`.
    pub fn compute_log(input: &[f64], _config: &SoftmaxConfig) -> Vec<f64> {
        if input.is_empty() {
            return Vec::new();
        }
        let lse = NumericalStabilizer::log_sum_exp(input);
        input.iter().map(|&x| x - lse).collect()
    }

    /// Compute sparse softmax, zeroing values below `threshold`.
    pub fn compute_sparse(input: &[f64], config: &SoftmaxConfig, threshold: f64) -> Vec<f64> {
        let mut result = Self::compute(input, config);
        for v in &mut result {
            if *v < threshold {
                *v = 0.0;
            }
        }
        // Re-normalize.
        let sum: f64 = result.iter().sum::<f64>() + config.epsilon;
        if sum > config.epsilon {
            for v in &mut result {
                *v /= sum;
            }
        }
        result
    }

    /// Online (streaming) softmax – single-pass algorithm.
    pub fn compute_online(input: &[f64], _config: &SoftmaxConfig) -> Vec<f64> {
        if input.is_empty() {
            return Vec::new();
        }
        // First pass: find max and sum of exps in one sweep.
        let mut max_val = f64::NEG_INFINITY;
        let mut sum_exp = 0.0_f64;
        for &x in input {
            if x > max_val {
                // Rescale previously accumulated sum.
                sum_exp *= (max_val - x).exp();
                max_val = x;
            }
            sum_exp += (x - max_val).exp();
        }
        // Second pass: compute outputs.
        input.iter().map(|&x| (x - max_val).exp() / sum_exp).collect()
    }

    /// Flash softmax – tiled computation for memory efficiency.
    pub fn compute_flash(input: &[f64], _config: &SoftmaxConfig, tile_size: usize) -> Vec<f64> {
        if input.is_empty() {
            return Vec::new();
        }
        let tile_size = tile_size.max(1);

        // Pass 1: compute global max and sum of exps tile-by-tile.
        let mut global_max = f64::NEG_INFINITY;
        let mut global_sum = 0.0_f64;

        for chunk in input.chunks(tile_size) {
            let tile_max = chunk.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if tile_max > global_max {
                global_sum *= (global_max - tile_max).exp();
                global_max = tile_max;
            }
            let tile_sum: f64 = chunk.iter().map(|&x| (x - global_max).exp()).sum();
            global_sum += tile_sum;
        }

        // Pass 2: produce output.
        input.iter().map(|&x| (x - global_max).exp() / global_sum).collect()
    }

    /// Dispatch to the requested softmax variant.
    pub fn dispatch(input: &[f64], config: &SoftmaxConfig, softmax_type: SoftmaxType) -> Vec<f64> {
        match softmax_type {
            SoftmaxType::Standard => Self::compute(input, config),
            SoftmaxType::LogSoftmax => Self::compute_log(input, config),
            SoftmaxType::SparseSoftmax => Self::compute_sparse(input, config, 0.01),
            SoftmaxType::OnlineSoftmax => Self::compute_online(input, config),
            SoftmaxType::FlashSoftmax => Self::compute_flash(input, config, 64),
        }
    }
}

// ---------------------------------------------------------------------------
// Temperature Softmax
// ---------------------------------------------------------------------------

/// Applies temperature scaling before softmax.
pub struct TemperatureSoftmax;

impl TemperatureSoftmax {
    /// Compute softmax with temperature: softmax(x / T).
    pub fn compute(input: &[f64], config: &SoftmaxConfig) -> Vec<f64> {
        assert!(config.temperature > 0.0, "temperature must be positive");
        let scaled: Vec<f64> = input.iter().map(|&x| x / config.temperature).collect();
        SoftmaxKernel::compute(&scaled, config)
    }
}

// ---------------------------------------------------------------------------
// Causal Softmax
// ---------------------------------------------------------------------------

/// Applies a causal (upper-triangular) mask before softmax.
///
/// For position `i`, future positions `j > i` are masked to negative infinity.
pub struct CausalSoftmax;

impl CausalSoftmax {
    /// Compute softmax over a square attention matrix with causal masking.
    ///
    /// `matrix` is row-major with dimensions `seq_len × seq_len`.
    pub fn compute(matrix: &[f64], seq_len: usize, config: &SoftmaxConfig) -> Vec<f64> {
        assert_eq!(matrix.len(), seq_len * seq_len);
        let mut output = Vec::with_capacity(matrix.len());
        for row in 0..seq_len {
            let start = row * seq_len;
            let mut row_data: Vec<f64> = matrix[start..start + seq_len].to_vec();
            // Mask future positions.
            for v in &mut row_data[(row + 1)..seq_len] {
                *v = f64::NEG_INFINITY;
            }
            output.extend(SoftmaxKernel::compute(&row_data, config));
        }
        output
    }
}

// ---------------------------------------------------------------------------
// Top-K Softmax
// ---------------------------------------------------------------------------

/// Zeros out all but the top-k values, then applies softmax.
pub struct TopKSoftmax;

impl TopKSoftmax {
    pub fn compute(input: &[f64], k: usize, config: &SoftmaxConfig) -> Vec<f64> {
        if input.is_empty() || k == 0 {
            return vec![0.0; input.len()];
        }
        let k = k.min(input.len());

        // Find the k-th largest value.
        let mut sorted: Vec<f64> = input.to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = sorted[k - 1];

        // Keep only values >= threshold.
        let filtered: Vec<f64> =
            input.iter().map(|&x| if x >= threshold { x } else { f64::NEG_INFINITY }).collect();
        SoftmaxKernel::compute(&filtered, config)
    }
}

// ---------------------------------------------------------------------------
// Softmax Gradient (backward pass)
// ---------------------------------------------------------------------------

/// Backward pass for softmax (Jacobian–vector product).
pub struct SoftmaxGradient;

impl SoftmaxGradient {
    /// Compute the gradient of loss w.r.t. softmax input given `softmax_output`
    /// and upstream `grad_output`.
    ///
    /// `dx_i = s_i * (dL/ds_i - Σ_j s_j * dL/ds_j)`
    pub fn compute(softmax_output: &[f64], grad_output: &[f64]) -> Vec<f64> {
        assert_eq!(softmax_output.len(), grad_output.len());
        let dot: f64 = softmax_output.iter().zip(grad_output.iter()).map(|(&s, &g)| s * g).sum();
        softmax_output.iter().zip(grad_output.iter()).map(|(&s, &g)| s * (g - dot)).collect()
    }

    /// Compute gradient of log-softmax.
    ///
    /// `dx_i = dL/ds_i - softmax(x)_i * Σ_j dL/ds_j`
    pub fn compute_log(softmax_output: &[f64], grad_output: &[f64]) -> Vec<f64> {
        assert_eq!(softmax_output.len(), grad_output.len());
        let sum_grad: f64 = grad_output.iter().sum();
        // softmax_output here are log-softmax values; exponentiate to get probs.
        let probs: Vec<f64> = softmax_output.iter().map(|&x| x.exp()).collect();
        grad_output.iter().zip(probs.iter()).map(|(&g, &p)| p.mul_add(-sum_grad, g)).collect()
    }
}

// ---------------------------------------------------------------------------
// Batch Softmax
// ---------------------------------------------------------------------------

/// Processes multiple sequences (rows) in a batch.
pub struct BatchSoftmax;

impl BatchSoftmax {
    /// Apply softmax independently to each row of a batch.
    ///
    /// `batch` is row-major with shape `[batch_size, dim]`.
    pub fn compute(
        batch: &[f64],
        batch_size: usize,
        dim: usize,
        config: &SoftmaxConfig,
    ) -> Vec<f64> {
        assert_eq!(batch.len(), batch_size * dim);
        let mut output = Vec::with_capacity(batch.len());
        for i in 0..batch_size {
            let row = &batch[i * dim..(i + 1) * dim];
            output.extend(SoftmaxKernel::compute(row, config));
        }
        output
    }

    /// Apply softmax with a specific variant to each row.
    pub fn compute_typed(
        batch: &[f64],
        batch_size: usize,
        dim: usize,
        config: &SoftmaxConfig,
        softmax_type: SoftmaxType,
    ) -> Vec<f64> {
        assert_eq!(batch.len(), batch_size * dim);
        let mut output = Vec::with_capacity(batch.len());
        for i in 0..batch_size {
            let row = &batch[i * dim..(i + 1) * dim];
            output.extend(SoftmaxKernel::dispatch(row, config, softmax_type));
        }
        output
    }
}

// ---------------------------------------------------------------------------
// Softmax Profiler
// ---------------------------------------------------------------------------

/// Tracks computation time and numerical stability statistics.
#[derive(Debug, Clone)]
pub struct SoftmaxProfiler {
    /// Total number of softmax calls profiled.
    pub call_count: u64,
    /// Cumulative wall-clock time (seconds).
    pub total_time_secs: f64,
    /// Maximum absolute deviation of any output row sum from 1.0.
    pub max_sum_deviation: f64,
    /// Minimum output probability observed (excluding zeros in sparse).
    pub min_probability: f64,
    /// Maximum output probability observed.
    pub max_probability: f64,
}

impl Default for SoftmaxProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl SoftmaxProfiler {
    pub const fn new() -> Self {
        Self {
            call_count: 0,
            total_time_secs: 0.0,
            max_sum_deviation: 0.0,
            min_probability: f64::INFINITY,
            max_probability: f64::NEG_INFINITY,
        }
    }

    /// Run softmax on `input`, record timing and stability stats, return output.
    pub fn profile_compute(&mut self, input: &[f64], config: &SoftmaxConfig) -> Vec<f64> {
        let start = Instant::now();
        let output = SoftmaxKernel::compute(input, config);
        self.total_time_secs += start.elapsed().as_secs_f64();
        self.update_stats(&output);
        output
    }

    /// Run softmax on a batch, record stats.
    pub fn profile_batch(
        &mut self,
        batch: &[f64],
        batch_size: usize,
        dim: usize,
        config: &SoftmaxConfig,
    ) -> Vec<f64> {
        let start = Instant::now();
        let output = BatchSoftmax::compute(batch, batch_size, dim, config);
        self.total_time_secs += start.elapsed().as_secs_f64();
        for row in output.chunks(dim) {
            self.update_stats(row);
        }
        output
    }

    fn update_stats(&mut self, output: &[f64]) {
        self.call_count += 1;
        let sum: f64 = output.iter().sum();
        let deviation = (sum - 1.0).abs();
        if deviation > self.max_sum_deviation {
            self.max_sum_deviation = deviation;
        }
        for &v in output {
            if v > 0.0 && v < self.min_probability {
                self.min_probability = v;
            }
            if v > self.max_probability {
                self.max_probability = v;
            }
        }
    }

    /// Average time per call in seconds.
    pub fn avg_time_secs(&self) -> f64 {
        if self.call_count == 0 {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            let count = self.call_count as f64;
            self.total_time_secs / count
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> SoftmaxConfig {
        SoftmaxConfig::default()
    }

    fn assert_sums_to_one(v: &[f64], tol: f64) {
        let sum: f64 = v.iter().sum();
        assert!((sum - 1.0).abs() < tol, "expected sum ≈ 1.0, got {sum}");
    }

    fn assert_all_non_negative(v: &[f64]) {
        for (i, &x) in v.iter().enumerate() {
            assert!(x >= 0.0, "element {i} is negative: {x}");
        }
    }

    // -----------------------------------------------------------------------
    // SoftmaxConfig tests
    // -----------------------------------------------------------------------

    #[test]
    fn config_default_values() {
        let c = SoftmaxConfig::default();
        assert!((c.temperature - 1.0).abs() < f64::EPSILON);
        assert!(c.epsilon > 0.0);
        assert_eq!(c.dimension, 0);
        assert!(!c.causal_mask);
    }

    #[test]
    fn config_custom() {
        let c =
            SoftmaxConfig { temperature: 0.5, epsilon: 1e-8, dimension: 128, causal_mask: true };
        assert!((c.temperature - 0.5).abs() < f64::EPSILON);
        assert!(c.causal_mask);
    }

    // -----------------------------------------------------------------------
    // SoftmaxType tests
    // -----------------------------------------------------------------------

    #[test]
    fn softmax_type_equality() {
        assert_eq!(SoftmaxType::Standard, SoftmaxType::Standard);
        assert_ne!(SoftmaxType::Standard, SoftmaxType::LogSoftmax);
    }

    #[test]
    fn softmax_type_clone() {
        let t = SoftmaxType::FlashSoftmax;
        let t2 = t;
        assert_eq!(t, t2);
    }

    #[test]
    fn softmax_type_debug() {
        let s = format!("{:?}", SoftmaxType::OnlineSoftmax);
        assert!(s.contains("OnlineSoftmax"));
    }

    // -----------------------------------------------------------------------
    // NumericalStabilizer tests
    // -----------------------------------------------------------------------

    #[test]
    fn stabilizer_subtract_max_basic() {
        let input = vec![1.0, 2.0, 3.0];
        let result = NumericalStabilizer::subtract_max(&input);
        assert!((result[2]).abs() < f64::EPSILON); // max element -> 0
        assert!((result[0] - (-2.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn stabilizer_subtract_max_single() {
        let result = NumericalStabilizer::subtract_max(&[42.0]);
        assert!((result[0]).abs() < f64::EPSILON);
    }

    #[test]
    fn stabilizer_subtract_max_all_same() {
        let input = vec![5.0; 4];
        let result = NumericalStabilizer::subtract_max(&input);
        for v in &result {
            assert!(v.abs() < f64::EPSILON);
        }
    }

    #[test]
    fn stabilizer_subtract_max_negative() {
        let input = vec![-10.0, -20.0, -5.0];
        let result = NumericalStabilizer::subtract_max(&input);
        assert!((result[2]).abs() < f64::EPSILON); // -5 is max
        assert!((result[0] - (-5.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn stabilizer_log_sum_exp_basic() {
        let input = vec![1.0, 2.0, 3.0];
        let lse = NumericalStabilizer::log_sum_exp(&input);
        let naive = input.iter().map(|x| x.exp()).sum::<f64>().ln();
        assert!((lse - naive).abs() < 1e-10);
    }

    #[test]
    fn stabilizer_log_sum_exp_large_values() {
        let input = vec![1000.0, 1001.0, 1002.0];
        let lse = NumericalStabilizer::log_sum_exp(&input);
        // Should not overflow; naive would produce inf.
        assert!(lse.is_finite());
        assert!(lse > 1000.0);
    }

    #[test]
    fn stabilizer_log_sum_exp_very_negative() {
        let input = vec![-1000.0, -999.0, -998.0];
        let lse = NumericalStabilizer::log_sum_exp(&input);
        assert!(lse.is_finite());
        assert!(lse < -997.0);
    }

    #[test]
    fn stabilizer_log_sum_exp_empty() {
        assert_eq!(NumericalStabilizer::log_sum_exp(&[]), f64::NEG_INFINITY);
    }

    #[test]
    fn stabilizer_log_sum_exp_single() {
        let lse = NumericalStabilizer::log_sum_exp(&[5.0]);
        assert!((lse - 5.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // SoftmaxKernel::compute tests
    // -----------------------------------------------------------------------

    #[test]
    fn kernel_basic() {
        let input = vec![1.0, 2.0, 3.0];
        let out = SoftmaxKernel::compute(&input, &default_config());
        assert_sums_to_one(&out, 1e-6);
        assert_all_non_negative(&out);
        // Monotonicity: larger input → larger probability.
        assert!(out[2] > out[1]);
        assert!(out[1] > out[0]);
    }

    #[test]
    fn kernel_empty() {
        let out = SoftmaxKernel::compute(&[], &default_config());
        assert!(out.is_empty());
    }

    #[test]
    fn kernel_single_element() {
        let out = SoftmaxKernel::compute(&[42.0], &default_config());
        assert_eq!(out.len(), 1);
        assert!((out[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn kernel_all_same_values() {
        let input = vec![3.0; 5];
        let out = SoftmaxKernel::compute(&input, &default_config());
        assert_sums_to_one(&out, 1e-6);
        // All should be equal.
        for v in &out {
            assert!((v - 0.2).abs() < 1e-6);
        }
    }

    #[test]
    fn kernel_large_values_no_overflow() {
        let input = vec![1000.0, 1001.0, 1002.0];
        let out = SoftmaxKernel::compute(&input, &default_config());
        assert_sums_to_one(&out, 1e-6);
        for v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn kernel_very_negative_values() {
        let input = vec![-1000.0, -999.0, -998.0];
        let out = SoftmaxKernel::compute(&input, &default_config());
        assert_sums_to_one(&out, 1e-6);
        for v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn kernel_mixed_extreme_values() {
        let input = vec![-500.0, 0.0, 500.0];
        let out = SoftmaxKernel::compute(&input, &default_config());
        assert_sums_to_one(&out, 1e-6);
        // The largest value should dominate.
        assert!(out[2] > 0.99);
    }

    #[test]
    fn kernel_two_elements() {
        let input = vec![0.0, 0.0];
        let out = SoftmaxKernel::compute(&input, &default_config());
        assert!((out[0] - 0.5).abs() < 1e-6);
        assert!((out[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn kernel_zeros() {
        let input = vec![0.0; 4];
        let out = SoftmaxKernel::compute(&input, &default_config());
        assert_sums_to_one(&out, 1e-6);
        for v in &out {
            assert!((v - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn kernel_extremely_negative_all() {
        let input = vec![-1e15; 3];
        let out = SoftmaxKernel::compute(&input, &default_config());
        assert_sums_to_one(&out, 1e-5);
    }

    // -----------------------------------------------------------------------
    // Log-softmax tests
    // -----------------------------------------------------------------------

    #[test]
    fn log_softmax_basic() {
        let input = vec![1.0, 2.0, 3.0];
        let log_out = SoftmaxKernel::compute_log(&input, &default_config());
        // All log-softmax values should be <= 0.
        for v in &log_out {
            assert!(*v <= 0.0 + 1e-10);
        }
    }

    #[test]
    fn log_softmax_vs_log_of_softmax() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let log_out = SoftmaxKernel::compute_log(&input, &default_config());
        let standard = SoftmaxKernel::compute(&input, &default_config());
        for (l, s) in log_out.iter().zip(standard.iter()) {
            assert!((l - s.ln()).abs() < 1e-6, "log_softmax={l}, log(softmax)={}", s.ln());
        }
    }

    #[test]
    fn log_softmax_empty() {
        let out = SoftmaxKernel::compute_log(&[], &default_config());
        assert!(out.is_empty());
    }

    #[test]
    fn log_softmax_single() {
        let out = SoftmaxKernel::compute_log(&[5.0], &default_config());
        assert!((out[0]).abs() < 1e-10);
    }

    #[test]
    fn log_softmax_large_values() {
        let input = vec![1000.0, 1001.0, 1002.0];
        let out = SoftmaxKernel::compute_log(&input, &default_config());
        for v in &out {
            assert!(v.is_finite());
        }
        // exp(log_softmax) should sum to 1.
        let probs: Vec<f64> = out.iter().map(|x| x.exp()).collect();
        assert_sums_to_one(&probs, 1e-6);
    }

    #[test]
    fn log_softmax_exp_sums_to_one() {
        let input = vec![0.5, 1.5, -0.5, 2.0];
        let log_out = SoftmaxKernel::compute_log(&input, &default_config());
        let sum: f64 = log_out.iter().map(|x| x.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Sparse softmax tests
    // -----------------------------------------------------------------------

    #[test]
    fn sparse_softmax_zeros_small() {
        let input = vec![10.0, 0.0, 0.0, 0.0];
        let out = SoftmaxKernel::compute_sparse(&input, &default_config(), 0.01);
        // First element should dominate; others may be zero.
        assert!(out[0] > 0.9);
        // Should still roughly sum to 1.
        let sum: f64 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5 || sum < 1e-5 + 1.0);
    }

    #[test]
    fn sparse_softmax_all_equal() {
        let input = vec![1.0; 4];
        let out = SoftmaxKernel::compute_sparse(&input, &default_config(), 0.01);
        // All equal → all kept → should sum to 1.
        assert_sums_to_one(&out, 1e-5);
    }

    // -----------------------------------------------------------------------
    // Online softmax tests
    // -----------------------------------------------------------------------

    #[test]
    fn online_softmax_matches_standard() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let standard = SoftmaxKernel::compute(&input, &default_config());
        let online = SoftmaxKernel::compute_online(&input, &default_config());
        for (s, o) in standard.iter().zip(online.iter()) {
            assert!((s - o).abs() < 1e-10, "standard={s}, online={o}");
        }
    }

    #[test]
    fn online_softmax_empty() {
        assert!(SoftmaxKernel::compute_online(&[], &default_config()).is_empty());
    }

    #[test]
    fn online_softmax_single() {
        let out = SoftmaxKernel::compute_online(&[7.0], &default_config());
        assert!((out[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn online_softmax_sums_to_one() {
        let input = vec![0.1, 0.2, 0.3, 0.4];
        let out = SoftmaxKernel::compute_online(&input, &default_config());
        assert_sums_to_one(&out, 1e-10);
    }

    #[test]
    fn online_softmax_large_values() {
        let input = vec![500.0, 501.0, 502.0];
        let out = SoftmaxKernel::compute_online(&input, &default_config());
        assert_sums_to_one(&out, 1e-10);
        for v in &out {
            assert!(v.is_finite());
        }
    }

    // -----------------------------------------------------------------------
    // Flash softmax tests
    // -----------------------------------------------------------------------

    #[test]
    fn flash_softmax_matches_standard() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let standard = SoftmaxKernel::compute(&input, &default_config());
        let flash = SoftmaxKernel::compute_flash(&input, &default_config(), 3);
        for (s, f) in standard.iter().zip(flash.iter()) {
            assert!((s - f).abs() < 1e-10, "standard={s}, flash={f}");
        }
    }

    #[test]
    fn flash_softmax_tile_size_1() {
        let input = vec![1.0, 2.0, 3.0];
        let standard = SoftmaxKernel::compute(&input, &default_config());
        let flash = SoftmaxKernel::compute_flash(&input, &default_config(), 1);
        for (s, f) in standard.iter().zip(flash.iter()) {
            assert!((s - f).abs() < 1e-10);
        }
    }

    #[test]
    fn flash_softmax_tile_larger_than_input() {
        let input = vec![1.0, 2.0];
        let standard = SoftmaxKernel::compute(&input, &default_config());
        let flash = SoftmaxKernel::compute_flash(&input, &default_config(), 100);
        for (s, f) in standard.iter().zip(flash.iter()) {
            assert!((s - f).abs() < 1e-10);
        }
    }

    #[test]
    fn flash_softmax_empty() {
        assert!(SoftmaxKernel::compute_flash(&[], &default_config(), 4).is_empty());
    }

    // -----------------------------------------------------------------------
    // Temperature softmax tests
    // -----------------------------------------------------------------------

    #[test]
    fn temperature_1_matches_standard() {
        let input = vec![1.0, 2.0, 3.0];
        let config = default_config();
        let standard = SoftmaxKernel::compute(&input, &config);
        let temp = TemperatureSoftmax::compute(&input, &config);
        for (s, t) in standard.iter().zip(temp.iter()) {
            assert!((s - t).abs() < 1e-10);
        }
    }

    #[test]
    fn high_temperature_more_uniform() {
        let input = vec![1.0, 2.0, 3.0];
        let low_config = SoftmaxConfig { temperature: 0.1, ..default_config() };
        let high_config = SoftmaxConfig { temperature: 100.0, ..default_config() };
        let low = TemperatureSoftmax::compute(&input, &low_config);
        let high = TemperatureSoftmax::compute(&input, &high_config);
        // High temp → more uniform → entropy is higher → max prob is lower.
        let max_low = low.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let max_high = high.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_high < max_low);
    }

    #[test]
    fn low_temperature_more_peaky() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let config = SoftmaxConfig { temperature: 0.01, ..default_config() };
        let out = TemperatureSoftmax::compute(&input, &config);
        // The max element should get almost all probability.
        assert!(out[3] > 0.99);
    }

    #[test]
    fn temperature_sums_to_one() {
        let input = vec![0.5, 1.5, 2.5];
        for t in [0.1, 0.5, 1.0, 2.0, 10.0] {
            let config = SoftmaxConfig { temperature: t, ..default_config() };
            let out = TemperatureSoftmax::compute(&input, &config);
            assert_sums_to_one(&out, 1e-5);
        }
    }

    #[test]
    #[should_panic(expected = "temperature must be positive")]
    fn temperature_zero_panics() {
        let config = SoftmaxConfig { temperature: 0.0, ..default_config() };
        TemperatureSoftmax::compute(&[1.0, 2.0], &config);
    }

    #[test]
    fn temperature_monotonicity() {
        // As temperature decreases, the argmax probability should increase.
        let input = vec![1.0, 3.0, 2.0];
        let mut prev_max = 0.0_f64;
        for t in [10.0, 5.0, 1.0, 0.5, 0.1] {
            let config = SoftmaxConfig { temperature: t, ..default_config() };
            let out = TemperatureSoftmax::compute(&input, &config);
            let cur_max = out.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            assert!(cur_max >= prev_max - 1e-10);
            prev_max = cur_max;
        }
    }

    // -----------------------------------------------------------------------
    // Causal softmax tests
    // -----------------------------------------------------------------------

    #[test]
    fn causal_mask_zeros_future() {
        let seq_len = 3;
        let matrix = vec![1.0; seq_len * seq_len];
        let out = CausalSoftmax::compute(&matrix, seq_len, &default_config());
        // Row 0: only position 0 visible → [1, 0, 0].
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!(out[1].abs() < 1e-6);
        assert!(out[2].abs() < 1e-6);
        // Row 1: positions 0,1 visible → [0.5, 0.5, 0].
        assert!((out[3] - 0.5).abs() < 1e-6);
        assert!((out[4] - 0.5).abs() < 1e-6);
        assert!(out[5].abs() < 1e-6);
        // Row 2: all visible → [1/3, 1/3, 1/3].
        assert!((out[6] - 1.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn causal_mask_rows_sum_to_one() {
        let seq_len = 5;
        let matrix = vec![0.5; seq_len * seq_len];
        let out = CausalSoftmax::compute(&matrix, seq_len, &default_config());
        for row in 0..seq_len {
            let row_sum: f64 = out[row * seq_len..(row + 1) * seq_len].iter().sum();
            assert!((row_sum - 1.0).abs() < 1e-6, "row {row} sum = {row_sum}");
        }
    }

    #[test]
    fn causal_mask_future_positions_zero() {
        let seq_len = 4;
        let matrix = vec![1.0; seq_len * seq_len];
        let out = CausalSoftmax::compute(&matrix, seq_len, &default_config());
        for row in 0..seq_len {
            for col in (row + 1)..seq_len {
                assert!(out[row * seq_len + col].abs() < 1e-10, "row={row}, col={col} should be 0");
            }
        }
    }

    #[test]
    fn causal_mask_single_token() {
        let out = CausalSoftmax::compute(&[5.0], 1, &default_config());
        assert!((out[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn causal_mask_two_tokens() {
        let matrix = vec![1.0, 2.0, 3.0, 4.0];
        let out = CausalSoftmax::compute(&matrix, 2, &default_config());
        // Row 0: only [1.0] visible → [1.0, 0.0]
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!(out[1].abs() < 1e-6);
        // Row 1: [3.0, 4.0] visible → softmax([3,4])
        let expected = SoftmaxKernel::compute(&[3.0, 4.0], &default_config());
        assert!((out[2] - expected[0]).abs() < 1e-6);
        assert!((out[3] - expected[1]).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Top-K softmax tests
    // -----------------------------------------------------------------------

    #[test]
    fn topk_basic() {
        let input = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        let out = TopKSoftmax::compute(&input, 2, &default_config());
        assert_sums_to_one(&out, 1e-6);
        // Only indices 1 (5.0) and 4 (4.0) should be non-zero.
        assert!(out[1] > 0.0);
        assert!(out[4] > 0.0);
        assert!(out[0] < 1e-10);
        assert!(out[3] < 1e-10);
    }

    #[test]
    fn topk_k_equals_len() {
        let input = vec![1.0, 2.0, 3.0];
        let out = TopKSoftmax::compute(&input, 3, &default_config());
        let standard = SoftmaxKernel::compute(&input, &default_config());
        for (a, b) in out.iter().zip(standard.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn topk_k_greater_than_len() {
        let input = vec![1.0, 2.0];
        let out = TopKSoftmax::compute(&input, 10, &default_config());
        assert_sums_to_one(&out, 1e-6);
    }

    #[test]
    fn topk_k_is_1() {
        let input = vec![1.0, 3.0, 2.0];
        let out = TopKSoftmax::compute(&input, 1, &default_config());
        assert!((out[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn topk_k_is_0() {
        let input = vec![1.0, 2.0, 3.0];
        let out = TopKSoftmax::compute(&input, 0, &default_config());
        for v in &out {
            assert!(v.abs() < f64::EPSILON);
        }
    }

    #[test]
    fn topk_all_same() {
        let input = vec![2.0; 5];
        let out = TopKSoftmax::compute(&input, 3, &default_config());
        // All values equal → all kept (threshold = 2.0, everything matches).
        assert_sums_to_one(&out, 1e-5);
    }

    #[test]
    fn topk_sums_to_one() {
        let input = vec![0.1, 0.5, 0.3, 0.8, 0.2];
        for k in 1..=5 {
            let out = TopKSoftmax::compute(&input, k, &default_config());
            assert_sums_to_one(&out, 1e-5);
        }
    }

    // -----------------------------------------------------------------------
    // Gradient tests
    // -----------------------------------------------------------------------

    #[test]
    fn gradient_basic_shape() {
        let softmax_out = SoftmaxKernel::compute(&[1.0, 2.0, 3.0], &default_config());
        let grad = vec![1.0, 0.0, 0.0];
        let dx = SoftmaxGradient::compute(&softmax_out, &grad);
        assert_eq!(dx.len(), 3);
    }

    #[test]
    fn gradient_sums_to_zero() {
        // The Jacobian of softmax has the property that gradients sum to zero.
        let softmax_out = SoftmaxKernel::compute(&[1.0, 2.0, 3.0, 4.0], &default_config());
        let grad = vec![0.5, -0.3, 0.8, -0.1];
        let dx = SoftmaxGradient::compute(&softmax_out, &grad);
        let sum: f64 = dx.iter().sum();
        assert!(sum.abs() < 1e-10, "gradient sum = {sum}");
    }

    #[test]
    fn gradient_finite_differences() {
        let input = vec![1.0, 2.0, 3.0];
        let config = default_config();
        let eps = 1e-5;
        let grad_output = vec![1.0, 0.5, -0.5];

        let softmax_out = SoftmaxKernel::compute(&input, &config);
        let analytical = SoftmaxGradient::compute(&softmax_out, &grad_output);

        // Numerical gradient via finite differences.
        for i in 0..input.len() {
            let mut input_plus = input.clone();
            let mut input_minus = input.clone();
            input_plus[i] += eps;
            input_minus[i] -= eps;

            let out_plus = SoftmaxKernel::compute(&input_plus, &config);
            let out_minus = SoftmaxKernel::compute(&input_minus, &config);

            let numerical: f64 = out_plus
                .iter()
                .zip(out_minus.iter())
                .zip(grad_output.iter())
                .map(|((&p, &m), &g)| (p - m) / (2.0 * eps) * g)
                .sum();

            assert!(
                (analytical[i] - numerical).abs() < 1e-5,
                "i={i}: analytical={}, numerical={numerical}",
                analytical[i]
            );
        }
    }

    #[test]
    fn gradient_log_softmax_shape() {
        let log_out = SoftmaxKernel::compute_log(&[1.0, 2.0, 3.0], &default_config());
        let grad = vec![1.0, 0.0, 0.0];
        let dx = SoftmaxGradient::compute_log(&log_out, &grad);
        assert_eq!(dx.len(), 3);
    }

    #[test]
    fn gradient_log_softmax_finite_differences() {
        let input = vec![1.0, 2.0, 3.0];
        let config = default_config();
        let eps = 1e-5;
        let grad_output = vec![1.0, -0.5, 0.3];

        let log_out = SoftmaxKernel::compute_log(&input, &config);
        let analytical = SoftmaxGradient::compute_log(&log_out, &grad_output);

        for i in 0..input.len() {
            let mut input_plus = input.clone();
            let mut input_minus = input.clone();
            input_plus[i] += eps;
            input_minus[i] -= eps;

            let out_plus = SoftmaxKernel::compute_log(&input_plus, &config);
            let out_minus = SoftmaxKernel::compute_log(&input_minus, &config);

            let numerical: f64 = out_plus
                .iter()
                .zip(out_minus.iter())
                .zip(grad_output.iter())
                .map(|((&p, &m), &g)| (p - m) / (2.0 * eps) * g)
                .sum();

            assert!(
                (analytical[i] - numerical).abs() < 1e-4,
                "i={i}: analytical={}, numerical={numerical}",
                analytical[i]
            );
        }
    }

    // -----------------------------------------------------------------------
    // Batch softmax tests
    // -----------------------------------------------------------------------

    #[test]
    fn batch_each_row_sums_to_one() {
        let batch = vec![
            1.0, 2.0, 3.0, // row 0
            4.0, 5.0, 6.0, // row 1
            0.0, 0.0, 0.0, // row 2
        ];
        let out = BatchSoftmax::compute(&batch, 3, 3, &default_config());
        assert_eq!(out.len(), 9);
        for row in 0..3 {
            let row_sum: f64 = out[row * 3..(row + 1) * 3].iter().sum();
            assert!((row_sum - 1.0).abs() < 1e-6, "row {row} sum = {row_sum}");
        }
    }

    #[test]
    fn batch_matches_individual() {
        let batch = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let config = default_config();
        let batch_out = BatchSoftmax::compute(&batch, 2, 3, &config);

        let row0 = SoftmaxKernel::compute(&batch[0..3], &config);
        let row1 = SoftmaxKernel::compute(&batch[3..6], &config);

        for (a, b) in batch_out[0..3].iter().zip(row0.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
        for (a, b) in batch_out[3..6].iter().zip(row1.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn batch_single_row() {
        let batch = vec![1.0, 2.0, 3.0];
        let out = BatchSoftmax::compute(&batch, 1, 3, &default_config());
        let single = SoftmaxKernel::compute(&batch, &default_config());
        assert_eq!(out, single);
    }

    #[test]
    fn batch_typed_log_softmax() {
        let batch = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let config = default_config();
        let out = BatchSoftmax::compute_typed(&batch, 2, 3, &config, SoftmaxType::LogSoftmax);
        assert_eq!(out.len(), 6);
        // Each row: exp(log_softmax) sums to 1.
        for row in 0..2 {
            let row_sum: f64 = out[row * 3..(row + 1) * 3].iter().map(|x| x.exp()).sum();
            assert!((row_sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn batch_typed_online() {
        let batch = vec![1.0, 2.0, 3.0, 4.0];
        let config = default_config();
        let online = BatchSoftmax::compute_typed(&batch, 2, 2, &config, SoftmaxType::OnlineSoftmax);
        let standard = BatchSoftmax::compute(&batch, 2, 2, &config);
        for (o, s) in online.iter().zip(standard.iter()) {
            assert!((o - s).abs() < 1e-10);
        }
    }

    // -----------------------------------------------------------------------
    // Dispatch tests
    // -----------------------------------------------------------------------

    #[test]
    fn dispatch_all_variants() {
        let input = vec![1.0, 2.0, 3.0];
        let config = default_config();
        for variant in [
            SoftmaxType::Standard,
            SoftmaxType::LogSoftmax,
            SoftmaxType::SparseSoftmax,
            SoftmaxType::OnlineSoftmax,
            SoftmaxType::FlashSoftmax,
        ] {
            let out = SoftmaxKernel::dispatch(&input, &config, variant);
            assert_eq!(out.len(), 3, "variant {variant:?} produced wrong length");
        }
    }

    #[test]
    fn dispatch_standard_matches_compute() {
        let input = vec![0.5, 1.5, 2.5];
        let config = default_config();
        let dispatched = SoftmaxKernel::dispatch(&input, &config, SoftmaxType::Standard);
        let direct = SoftmaxKernel::compute(&input, &config);
        assert_eq!(dispatched, direct);
    }

    // -----------------------------------------------------------------------
    // Profiler tests
    // -----------------------------------------------------------------------

    #[test]
    fn profiler_tracks_calls() {
        let mut profiler = SoftmaxProfiler::new();
        let config = default_config();
        profiler.profile_compute(&[1.0, 2.0, 3.0], &config);
        profiler.profile_compute(&[4.0, 5.0, 6.0], &config);
        assert_eq!(profiler.call_count, 2);
        assert!(profiler.total_time_secs >= 0.0);
    }

    #[test]
    fn profiler_default() {
        let p = SoftmaxProfiler::default();
        assert_eq!(p.call_count, 0);
        assert!((p.avg_time_secs()).abs() < f64::EPSILON);
    }

    #[test]
    fn profiler_sum_deviation_small() {
        let mut profiler = SoftmaxProfiler::new();
        let config = default_config();
        profiler.profile_compute(&[1.0, 2.0, 3.0], &config);
        assert!(profiler.max_sum_deviation < 1e-5);
    }

    #[test]
    fn profiler_probability_range() {
        let mut profiler = SoftmaxProfiler::new();
        let config = default_config();
        profiler.profile_compute(&[1.0, 2.0, 3.0], &config);
        assert!(profiler.min_probability > 0.0);
        assert!(profiler.max_probability <= 1.0);
        assert!(profiler.min_probability <= profiler.max_probability);
    }

    #[test]
    fn profiler_batch() {
        let mut profiler = SoftmaxProfiler::new();
        let batch = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let config = default_config();
        let out = profiler.profile_batch(&batch, 2, 3, &config);
        assert_eq!(out.len(), 6);
        assert_eq!(profiler.call_count, 2);
    }

    #[test]
    fn profiler_avg_time() {
        let mut profiler = SoftmaxProfiler::new();
        let config = default_config();
        profiler.profile_compute(&[1.0; 100], &config);
        profiler.profile_compute(&[2.0; 100], &config);
        assert!(profiler.avg_time_secs() >= 0.0);
        assert!(profiler.avg_time_secs() <= profiler.total_time_secs);
    }

    // -----------------------------------------------------------------------
    // Sum-to-1 invariant for all softmax types
    // -----------------------------------------------------------------------

    #[test]
    fn all_non_log_variants_sum_to_one() {
        let input = vec![0.5, 1.0, 1.5, 2.0];
        let config = default_config();
        for variant in [
            SoftmaxType::Standard,
            SoftmaxType::SparseSoftmax,
            SoftmaxType::OnlineSoftmax,
            SoftmaxType::FlashSoftmax,
        ] {
            let out = SoftmaxKernel::dispatch(&input, &config, variant);
            assert_sums_to_one(&out, 1e-5);
            assert_all_non_negative(&out);
        }
    }

    #[test]
    fn log_softmax_exp_sums_to_one_all_inputs() {
        for input in [vec![1.0, 2.0, 3.0], vec![0.0; 5], vec![-1.0, -2.0, -3.0], vec![100.0, 200.0]]
        {
            let out = SoftmaxKernel::compute_log(&input, &default_config());
            let sum: f64 = out.iter().map(|x| x.exp()).sum();
            assert!((sum - 1.0).abs() < 1e-5, "input={input:?}, exp(log_softmax) sum={sum}");
        }
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn edge_very_large_dimension() {
        let input = vec![1.0; 10_000];
        let out = SoftmaxKernel::compute(&input, &default_config());
        assert_sums_to_one(&out, 1e-5);
        let expected = 1.0 / 10_000.0;
        for v in &out {
            assert!((v - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn edge_alternating_extreme() {
        let mut input = vec![0.0; 6];
        for i in 0..6 {
            input[i] = if i % 2 == 0 { 100.0 } else { -100.0 };
        }
        let out = SoftmaxKernel::compute(&input, &default_config());
        assert_sums_to_one(&out, 1e-5);
        // Even indices should dominate.
        for (i, v) in out.iter().enumerate() {
            if i % 2 == 0 {
                assert!(*v > 0.3);
            } else {
                assert!(*v < 1e-10);
            }
        }
    }

    #[test]
    fn edge_one_hot_like() {
        let input = vec![-1e10, -1e10, 0.0, -1e10];
        let out = SoftmaxKernel::compute(&input, &default_config());
        assert!((out[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn edge_descending_order() {
        let input: Vec<f64> = (0..10).rev().map(|i| i as f64).collect();
        let out = SoftmaxKernel::compute(&input, &default_config());
        assert_sums_to_one(&out, 1e-6);
        // Should be in descending probability order.
        for i in 0..9 {
            assert!(out[i] >= out[i + 1] - 1e-10);
        }
    }

    #[test]
    fn edge_ascending_order() {
        let input: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let out = SoftmaxKernel::compute(&input, &default_config());
        // Should be in ascending probability order.
        for i in 0..9 {
            assert!(out[i] <= out[i + 1] + 1e-10);
        }
    }

    // -----------------------------------------------------------------------
    // Proptest
    // -----------------------------------------------------------------------

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        fn finite_f64() -> impl Strategy<Value = f64> {
            -500.0..500.0_f64
        }

        proptest! {
            #[test]
            fn softmax_sums_to_one(input in proptest::collection::vec(finite_f64(), 1..100)) {
                let out = SoftmaxKernel::compute(&input, &default_config());
                let sum: f64 = out.iter().sum();
                prop_assert!((sum - 1.0).abs() < 1e-5, "sum = {}", sum);
            }

            #[test]
            fn softmax_all_non_negative(input in proptest::collection::vec(finite_f64(), 1..100)) {
                let out = SoftmaxKernel::compute(&input, &default_config());
                for (i, &v) in out.iter().enumerate() {
                    prop_assert!(v >= 0.0, "element {} is {}", i, v);
                }
            }

            #[test]
            fn online_matches_standard(input in proptest::collection::vec(finite_f64(), 1..50)) {
                let config = default_config();
                let standard = SoftmaxKernel::compute(&input, &config);
                let online = SoftmaxKernel::compute_online(&input, &config);
                for (s, o) in standard.iter().zip(online.iter()) {
                    prop_assert!((s - o).abs() < 1e-8, "standard={}, online={}", s, o);
                }
            }

            #[test]
            fn flash_matches_standard(
                input in proptest::collection::vec(finite_f64(), 1..50),
                tile_size in 1_usize..20
            ) {
                let config = default_config();
                let standard = SoftmaxKernel::compute(&input, &config);
                let flash = SoftmaxKernel::compute_flash(&input, &config, tile_size);
                for (s, f) in standard.iter().zip(flash.iter()) {
                    prop_assert!((s - f).abs() < 1e-8, "standard={}, flash={}", s, f);
                }
            }

            #[test]
            fn log_softmax_exp_sums_to_one(input in proptest::collection::vec(finite_f64(), 1..50)) {
                let out = SoftmaxKernel::compute_log(&input, &default_config());
                let sum: f64 = out.iter().map(|x| x.exp()).sum();
                prop_assert!((sum - 1.0).abs() < 1e-5, "exp(log_softmax) sum = {}", sum);
            }

            #[test]
            fn temperature_preserves_sum(
                input in proptest::collection::vec(finite_f64(), 1..50),
                temp in 0.01_f64..100.0
            ) {
                let config = SoftmaxConfig { temperature: temp, ..default_config() };
                let out = TemperatureSoftmax::compute(&input, &config);
                let sum: f64 = out.iter().sum();
                prop_assert!((sum - 1.0).abs() < 1e-5, "sum = {} at temp = {}", sum, temp);
            }

            #[test]
            fn gradient_sums_to_zero(input in proptest::collection::vec(finite_f64(), 2..20)) {
                let config = default_config();
                let softmax_out = SoftmaxKernel::compute(&input, &config);
                let grad_output: Vec<f64> = input.iter().map(|x| x.sin()).collect();
                let dx = SoftmaxGradient::compute(&softmax_out, &grad_output);
                let sum: f64 = dx.iter().sum();
                prop_assert!(sum.abs() < 1e-8, "gradient sum = {}", sum);
            }

            #[test]
            fn topk_sums_to_one(
                input in proptest::collection::vec(finite_f64(), 1..30),
                k in 1_usize..30
            ) {
                let out = TopKSoftmax::compute(&input, k, &default_config());
                let sum: f64 = out.iter().sum();
                prop_assert!((sum - 1.0).abs() < 1e-5, "sum = {} for k = {}", sum, k);
            }

            #[test]
            fn batch_each_row_sums_to_one(
                dim in 1_usize..20,
                batch_size in 1_usize..10
            ) {
                let batch: Vec<f64> = (0..dim * batch_size).map(|i| (i as f64) * 0.1).collect();
                let out = BatchSoftmax::compute(&batch, batch_size, dim, &default_config());
                for row in 0..batch_size {
                    let sum: f64 = out[row * dim..(row + 1) * dim].iter().sum();
                    prop_assert!((sum - 1.0).abs() < 1e-5, "row {} sum = {}", row, sum);
                }
            }

            #[test]
            fn causal_future_is_zero(seq_len in 1_usize..10) {
                let matrix = vec![1.0; seq_len * seq_len];
                let out = CausalSoftmax::compute(&matrix, seq_len, &default_config());
                for row in 0..seq_len {
                    for col in (row + 1)..seq_len {
                        let val = out[row * seq_len + col];
                        prop_assert!(val.abs() < 1e-10, "row={}, col={}, val={}", row, col, val);
                    }
                }
            }
        }
    }
}

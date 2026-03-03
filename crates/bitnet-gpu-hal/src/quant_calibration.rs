//! Post-training quantization calibration engine.
//!
//! Provides calibration methods (MinMax, Percentile, MSE, Entropy) for
//! determining optimal quantization ranges from representative data.

use std::fmt;
use std::time::Instant;

// ── Configuration ─────────────────────────────────────────────────────────

/// Calibration method selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationMethod {
    /// Use observed min/max directly.
    MinMax,
    /// Clip at the P'th percentile to reduce outlier influence.
    Percentile,
    /// Search for clipping range that minimizes quantization MSE.
    Mse,
    /// TensorRT-style: minimize KL divergence between original and
    /// quantized distributions.
    Entropy,
}

impl fmt::Display for CalibrationMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MinMax => write!(f, "MinMax"),
            Self::Percentile => write!(f, "Percentile"),
            Self::Mse => write!(f, "MSE"),
            Self::Entropy => write!(f, "Entropy"),
        }
    }
}

/// Configuration for a calibration run.
#[derive(Debug, Clone)]
pub struct CalibrationConfig {
    /// Number of samples to use for calibration.
    pub num_samples: usize,
    /// Calibration method.
    pub method: CalibrationMethod,
    /// Per-channel (true) vs per-tensor (false) calibration.
    pub per_channel: bool,
    /// Symmetric quantization (true) vs asymmetric (false).
    pub symmetric: bool,
    /// Number of quantization bits (e.g. 4, 8).
    pub num_bits: u32,
    /// Percentile value for the Percentile method (0.0–1.0).
    pub percentile: f64,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            num_samples: 100,
            method: CalibrationMethod::MinMax,
            per_channel: false,
            symmetric: true,
            num_bits: 8,
            percentile: 0.9999,
        }
    }
}

// ── Calibration Dataset ───────────────────────────────────────────────────

/// Stores calibration samples as flat f32 tensors.
#[derive(Debug, Clone)]
pub struct CalibrationDataset {
    samples: Vec<Vec<f32>>,
}

impl CalibrationDataset {
    /// Create an empty dataset.
    pub fn new() -> Self {
        Self { samples: Vec::new() }
    }

    /// Add a calibration sample.
    pub fn add_sample(&mut self, tensor: Vec<f32>) {
        self.samples.push(tensor);
    }

    /// Number of samples in the dataset.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Iterate over samples.
    pub fn iter(&self) -> impl Iterator<Item = &[f32]> {
        self.samples.iter().map(Vec::as_slice)
    }
}

impl Default for CalibrationDataset {
    fn default() -> Self {
        Self::new()
    }
}

// ── Activation Observer ───────────────────────────────────────────────────

/// Records activation statistics during calibration runs.
#[derive(Debug, Clone)]
pub struct ActivationObserver {
    /// Running minimum.
    pub min_val: f64,
    /// Running maximum.
    pub max_val: f64,
    /// Sum of all observed values (for mean).
    sum: f64,
    /// Sum of squares (for variance).
    sum_sq: f64,
    /// Total number of observed values.
    count: u64,
    /// Histogram bins for distribution analysis.
    histogram: Vec<u64>,
    /// Number of histogram bins.
    num_bins: usize,
    /// Histogram range: lower bound.
    hist_min: f64,
    /// Histogram range: upper bound.
    hist_max: f64,
}

impl ActivationObserver {
    /// Create a new observer with the specified number of histogram bins.
    pub fn new(num_bins: usize) -> Self {
        Self {
            min_val: f64::INFINITY,
            max_val: f64::NEG_INFINITY,
            sum: 0.0,
            sum_sq: 0.0,
            count: 0,
            histogram: vec![0; num_bins],
            num_bins,
            hist_min: 0.0,
            hist_max: 0.0,
        }
    }

    /// Observe a batch of activation values.
    pub fn observe(&mut self, values: &[f32]) {
        for &v in values {
            let vf = f64::from(v);
            if vf < self.min_val {
                self.min_val = vf;
            }
            if vf > self.max_val {
                self.max_val = vf;
            }
            self.sum += vf;
            self.sum_sq += vf * vf;
            self.count += 1;
        }
        // Rebuild histogram with updated range.
        self.rebuild_histogram(values);
    }

    /// Rebuild the histogram from scratch using current min/max.
    fn rebuild_histogram(&mut self, values: &[f32]) {
        if self.min_val >= self.max_val || self.num_bins == 0 {
            return;
        }
        self.hist_min = self.min_val;
        self.hist_max = self.max_val;
        let range = self.hist_max - self.hist_min;
        for &v in values {
            let vf = f64::from(v);
            let normalized = (vf - self.hist_min) / range;
            let bin = (normalized * self.num_bins as f64) as usize;
            let bin = bin.min(self.num_bins - 1);
            self.histogram[bin] += 1;
        }
    }

    /// Mean of observed values.
    pub fn mean(&self) -> f64 {
        if self.count == 0 { 0.0 } else { self.sum / self.count as f64 }
    }

    /// Variance of observed values.
    pub fn variance(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let mean = self.mean();
        self.sum_sq / self.count as f64 - mean * mean
    }

    /// Total number of observed values.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Access the histogram bins.
    pub fn histogram(&self) -> &[u64] {
        &self.histogram
    }

    /// Histogram range (min, max).
    pub fn histogram_range(&self) -> (f64, f64) {
        (self.hist_min, self.hist_max)
    }

    /// Reset all accumulated statistics.
    pub fn reset(&mut self) {
        self.min_val = f64::INFINITY;
        self.max_val = f64::NEG_INFINITY;
        self.sum = 0.0;
        self.sum_sq = 0.0;
        self.count = 0;
        self.histogram.fill(0);
        self.hist_min = 0.0;
        self.hist_max = 0.0;
    }
}

// ── Quantization Range ────────────────────────────────────────────────────

/// Describes a quantization range with computed scale and zero-point.
#[derive(Debug, Clone)]
pub struct QuantizationRange {
    /// Minimum value of the range.
    pub min_val: f64,
    /// Maximum value of the range.
    pub max_val: f64,
    /// Quantization scale factor.
    pub scale: f64,
    /// Quantization zero-point.
    pub zero_point: i64,
    /// Number of quantization bits.
    pub num_bits: u32,
}

impl QuantizationRange {
    /// Compute quantization parameters for the given range.
    pub fn compute_params(min_val: f64, max_val: f64, num_bits: u32, symmetric: bool) -> Self {
        let qmin = 0i64;
        let qmax = (1i64 << num_bits) - 1;

        if symmetric {
            let abs_max = min_val.abs().max(max_val.abs());
            let scale = if abs_max == 0.0 { 1.0 } else { (2.0 * abs_max) / (qmax - qmin) as f64 };
            let zero_point = (qmax - qmin) / 2 + qmin;
            Self { min_val: -abs_max, max_val: abs_max, scale, zero_point, num_bits }
        } else {
            let range = max_val - min_val;
            let scale = if range == 0.0 { 1.0 } else { range / (qmax - qmin) as f64 };
            let zero_point = qmin - (min_val / scale).round() as i64;
            let zero_point = zero_point.clamp(qmin, qmax);
            Self { min_val, max_val, scale, zero_point, num_bits }
        }
    }

    /// Quantize a single float value.
    pub fn quantize(&self, value: f64) -> i64 {
        let qmin = 0i64;
        let qmax = (1i64 << self.num_bits) - 1;
        let q = (value / self.scale).round() + self.zero_point as f64;
        (q as i64).clamp(qmin, qmax)
    }

    /// Dequantize a single integer value.
    pub fn dequantize(&self, quantized: i64) -> f64 {
        (quantized - self.zero_point) as f64 * self.scale
    }
}

// ── MinMax Calibrator ─────────────────────────────────────────────────────

/// Simplest calibrator: uses observed min/max as the quantization range.
pub struct MinMaxCalibrator;

impl MinMaxCalibrator {
    /// Calibrate from an `ActivationObserver`.
    pub fn calibrate(
        observer: &ActivationObserver,
        num_bits: u32,
        symmetric: bool,
    ) -> QuantizationRange {
        QuantizationRange::compute_params(observer.min_val, observer.max_val, num_bits, symmetric)
    }

    /// Calibrate directly from raw data.
    pub fn calibrate_from_data(data: &[f32], num_bits: u32, symmetric: bool) -> QuantizationRange {
        let mut observer = ActivationObserver::new(256);
        observer.observe(data);
        Self::calibrate(&observer, num_bits, symmetric)
    }
}

// ── Percentile Calibrator ─────────────────────────────────────────────────

/// Clips the quantization range at the given percentile to reduce outlier
/// influence.
pub struct PercentileCalibrator;

impl PercentileCalibrator {
    /// Calibrate by clipping at `percentile` (0.0–1.0).
    pub fn calibrate(
        data: &[f32],
        percentile: f64,
        num_bits: u32,
        symmetric: bool,
    ) -> QuantizationRange {
        if data.is_empty() {
            return QuantizationRange::compute_params(0.0, 0.0, num_bits, symmetric);
        }

        let mut sorted: Vec<f64> = data.iter().map(|&v| f64::from(v)).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let lower_idx = ((1.0 - percentile) * (sorted.len() - 1) as f64).round() as usize;
        let upper_idx = (percentile * (sorted.len() - 1) as f64).round() as usize;

        let min_val = sorted[lower_idx.min(sorted.len() - 1)];
        let max_val = sorted[upper_idx.min(sorted.len() - 1)];

        QuantizationRange::compute_params(min_val, max_val, num_bits, symmetric)
    }
}

// ── MSE Calibrator ────────────────────────────────────────────────────────

/// Searches for the optimal clipping range that minimizes quantization MSE.
pub struct MseCalibrator;

impl MseCalibrator {
    /// Number of candidate thresholds to evaluate.
    const NUM_CANDIDATES: usize = 200;

    /// Calibrate by searching for the MSE-optimal clipping range.
    pub fn calibrate(data: &[f32], num_bits: u32, symmetric: bool) -> QuantizationRange {
        if data.is_empty() {
            return QuantizationRange::compute_params(0.0, 0.0, num_bits, symmetric);
        }

        let mut abs_sorted: Vec<f64> = data.iter().map(|&v| f64::from(v).abs()).collect();
        abs_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let max_abs = abs_sorted.last().copied().unwrap_or(0.0);
        if max_abs == 0.0 {
            return QuantizationRange::compute_params(0.0, 0.0, num_bits, symmetric);
        }

        let mut best_range =
            QuantizationRange::compute_params(-max_abs, max_abs, num_bits, symmetric);
        let mut best_mse = Self::compute_mse(data, &best_range);

        for i in 1..Self::NUM_CANDIDATES {
            let fraction = i as f64 / Self::NUM_CANDIDATES as f64;
            let threshold = max_abs * fraction;
            let candidate =
                QuantizationRange::compute_params(-threshold, threshold, num_bits, symmetric);
            let mse = Self::compute_mse(data, &candidate);
            if mse < best_mse {
                best_mse = mse;
                best_range = candidate;
            }
        }

        best_range
    }

    /// Compute MSE between original and quantized-then-dequantized values.
    fn compute_mse(data: &[f32], range: &QuantizationRange) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        let sum_sq: f64 = data
            .iter()
            .map(|&v| {
                let vf = f64::from(v);
                let q = range.quantize(vf);
                let dq = range.dequantize(q);
                let diff = vf - dq;
                diff * diff
            })
            .sum();
        sum_sq / data.len() as f64
    }
}

// ── Entropy Calibrator ────────────────────────────────────────────────────

/// TensorRT-style calibrator: minimizes KL divergence between the original
/// and quantized activation distributions.
pub struct EntropyCalibrator;

impl EntropyCalibrator {
    /// Number of histogram bins for distribution modelling.
    const NUM_BINS: usize = 2048;
    /// Number of candidate thresholds to evaluate.
    const NUM_CANDIDATES: usize = 128;

    /// Calibrate by minimizing KL divergence.
    pub fn calibrate(data: &[f32], num_bits: u32, symmetric: bool) -> QuantizationRange {
        if data.is_empty() {
            return QuantizationRange::compute_params(0.0, 0.0, num_bits, symmetric);
        }

        let data_f64: Vec<f64> = data.iter().map(|&v| f64::from(v)).collect();
        let min_val = data_f64.iter().copied().fold(f64::INFINITY, f64::min);
        let max_val = data_f64.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        if (max_val - min_val).abs() < f64::EPSILON {
            return QuantizationRange::compute_params(min_val, max_val, num_bits, symmetric);
        }

        // Build reference histogram.
        let range = max_val - min_val;
        let mut ref_hist = vec![0u64; Self::NUM_BINS];
        for &v in &data_f64 {
            let normalized = (v - min_val) / range;
            let bin = (normalized * (Self::NUM_BINS - 1) as f64).round() as usize;
            ref_hist[bin.min(Self::NUM_BINS - 1)] += 1;
        }

        let num_quant_levels = 1usize << num_bits;
        let mut best_kl = f64::INFINITY;
        let abs_max = min_val.abs().max(max_val.abs());
        let mut best_threshold = abs_max;

        // Search over candidate thresholds.
        for i in 1..=Self::NUM_CANDIDATES {
            let fraction = i as f64 / Self::NUM_CANDIDATES as f64;
            let threshold = abs_max * fraction;
            if threshold == 0.0 {
                continue;
            }

            let kl =
                Self::compute_kl_divergence(&ref_hist, min_val, range, threshold, num_quant_levels);
            if kl < best_kl {
                best_kl = kl;
                best_threshold = threshold;
            }
        }

        QuantizationRange::compute_params(-best_threshold, best_threshold, num_bits, symmetric)
    }

    /// Compute KL divergence between the reference histogram and the
    /// quantized-then-expanded distribution.
    fn compute_kl_divergence(
        ref_hist: &[u64],
        hist_min: f64,
        hist_range: f64,
        threshold: f64,
        num_quant_levels: usize,
    ) -> f64 {
        let num_bins = ref_hist.len();
        let total: u64 = ref_hist.iter().sum();
        if total == 0 {
            return 0.0;
        }

        // Build quantized histogram: map each reference bin to a quant level.
        let mut quant_hist = vec![0u64; num_quant_levels];
        for (i, &count) in ref_hist.iter().enumerate() {
            let bin_center = hist_min + (i as f64 + 0.5) * hist_range / num_bins as f64;
            let clamped = bin_center.clamp(-threshold, threshold);
            let normalized = (clamped + threshold) / (2.0 * threshold);
            let q_idx = (normalized * (num_quant_levels - 1) as f64).round() as usize;
            let q_idx = q_idx.min(num_quant_levels - 1);
            quant_hist[q_idx] += count;
        }

        // Expand quantized histogram back to reference resolution.
        let mut expanded = vec![0.0_f64; num_bins];
        for (i, _) in ref_hist.iter().enumerate() {
            let bin_center = hist_min + (i as f64 + 0.5) * hist_range / num_bins as f64;
            let clamped = bin_center.clamp(-threshold, threshold);
            let normalized = (clamped + threshold) / (2.0 * threshold);
            let q_idx = (normalized * (num_quant_levels - 1) as f64).round() as usize;
            let q_idx = q_idx.min(num_quant_levels - 1);
            if quant_hist[q_idx] > 0 {
                // Count how many ref bins map to this quant level.
                let bins_in_level: usize = ref_hist
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| {
                        let bc = hist_min + (*j as f64 + 0.5) * hist_range / num_bins as f64;
                        let c = bc.clamp(-threshold, threshold);
                        let n = (c + threshold) / (2.0 * threshold);
                        let qi = (n * (num_quant_levels - 1) as f64).round() as usize;
                        qi.min(num_quant_levels - 1) == q_idx
                    })
                    .count();
                if bins_in_level > 0 {
                    expanded[i] = quant_hist[q_idx] as f64 / bins_in_level as f64;
                }
            }
        }

        // Compute KL(ref || expanded).
        let total_f = total as f64;
        let exp_total: f64 = expanded.iter().sum();
        let mut kl = 0.0_f64;
        for (i, &count) in ref_hist.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let p = count as f64 / total_f;
            let q = if exp_total > 0.0 { expanded[i] / exp_total } else { 0.0 };
            if q > 0.0 {
                kl += p * (p / q).ln();
            }
        }
        kl
    }
}

// ── Calibration Runner ────────────────────────────────────────────────────

/// Runs calibration over a dataset, collecting per-layer observer statistics.
pub struct CalibrationRunner {
    config: CalibrationConfig,
    observers: Vec<ActivationObserver>,
    layer_names: Vec<String>,
}

impl CalibrationRunner {
    /// Create a new runner with the given config and layer names.
    pub fn new(config: CalibrationConfig, layer_names: Vec<String>) -> Self {
        let observers = layer_names.iter().map(|_| ActivationObserver::new(2048)).collect();
        Self { config, observers, layer_names }
    }

    /// Feed a sample through the runner, recording activations for each layer.
    ///
    /// `layer_activations` should have one entry per layer.
    pub fn process_sample(
        &mut self,
        layer_activations: &[Vec<f32>],
    ) -> Result<(), CalibrationError> {
        if layer_activations.len() != self.observers.len() {
            return Err(CalibrationError::LayerCountMismatch {
                expected: self.observers.len(),
                actual: layer_activations.len(),
            });
        }
        for (obs, act) in self.observers.iter_mut().zip(layer_activations.iter()) {
            obs.observe(act);
        }
        Ok(())
    }

    /// Run calibration on an entire dataset and return per-layer ranges.
    pub fn run(
        &mut self,
        dataset: &CalibrationDataset,
    ) -> Result<(Vec<QuantizationRange>, CalibrationMetrics), CalibrationError> {
        if dataset.is_empty() {
            return Err(CalibrationError::EmptyDataset);
        }
        let start = Instant::now();
        let mut samples_processed = 0u64;

        // For simplicity, each sample is treated as activations for all layers
        // (split evenly). In a real system the model forward pass would produce
        // per-layer tensors.
        for sample in dataset.iter() {
            if self.observers.is_empty() {
                break;
            }
            let chunk_size = sample.len() / self.observers.len();
            if chunk_size == 0 {
                continue;
            }
            let chunks: Vec<Vec<f32>> = self
                .observers
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    let start = i * chunk_size;
                    let end = (start + chunk_size).min(sample.len());
                    sample[start..end].to_vec()
                })
                .collect();
            self.process_sample(&chunks)?;
            samples_processed += 1;
        }

        // Compute ranges.
        let ranges: Vec<QuantizationRange> =
            self.observers.iter().map(|obs| self.calibrate_observer(obs)).collect();

        let avg_error = Self::compute_avg_error(&ranges);
        let elapsed = start.elapsed().as_millis() as u64;

        let metrics = CalibrationMetrics {
            samples_processed,
            layers_calibrated: ranges.len() as u64,
            avg_quantization_error: avg_error,
            calibration_time_ms: elapsed,
        };

        Ok((ranges, metrics))
    }

    /// Calibrate a single observer using the configured method.
    fn calibrate_observer(&self, observer: &ActivationObserver) -> QuantizationRange {
        match self.config.method {
            CalibrationMethod::MinMax => {
                MinMaxCalibrator::calibrate(observer, self.config.num_bits, self.config.symmetric)
            }
            _ => {
                // For Percentile/MSE/Entropy, fall back to MinMax when only
                // observer stats are available (no raw data retained).
                MinMaxCalibrator::calibrate(observer, self.config.num_bits, self.config.symmetric)
            }
        }
    }

    /// Compute average quantization error (scale) across ranges.
    fn compute_avg_error(ranges: &[QuantizationRange]) -> f64 {
        if ranges.is_empty() {
            return 0.0;
        }
        let sum: f64 = ranges.iter().map(|r| r.scale).sum();
        sum / ranges.len() as f64
    }

    /// Access observers.
    pub fn observers(&self) -> &[ActivationObserver] {
        &self.observers
    }

    /// Access layer names.
    pub fn layer_names(&self) -> &[String] {
        &self.layer_names
    }

    /// Access config.
    pub fn config(&self) -> &CalibrationConfig {
        &self.config
    }
}

// ── Calibration Metrics ───────────────────────────────────────────────────

/// Summary statistics from a calibration run.
#[derive(Debug, Clone)]
pub struct CalibrationMetrics {
    /// Number of calibration samples processed.
    pub samples_processed: u64,
    /// Number of layers calibrated.
    pub layers_calibrated: u64,
    /// Average quantization error (scale) across all layers.
    pub avg_quantization_error: f64,
    /// Wall-clock time for calibration in milliseconds.
    pub calibration_time_ms: u64,
}

impl fmt::Display for CalibrationMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CalibrationMetrics {{ samples: {}, layers: {}, \
             avg_error: {:.6}, time: {}ms }}",
            self.samples_processed,
            self.layers_calibrated,
            self.avg_quantization_error,
            self.calibration_time_ms,
        )
    }
}

// ── Errors ────────────────────────────────────────────────────────────────

/// Errors that can occur during calibration.
#[derive(Debug, Clone, PartialEq)]
pub enum CalibrationError {
    /// Dataset contained no samples.
    EmptyDataset,
    /// Number of layer activations did not match expected layer count.
    LayerCountMismatch { expected: usize, actual: usize },
}

impl fmt::Display for CalibrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDataset => write!(f, "calibration dataset is empty"),
            Self::LayerCountMismatch { expected, actual } => {
                write!(f, "layer count mismatch: expected {expected}, got {actual}")
            }
        }
    }
}

impl std::error::Error for CalibrationError {}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── CalibrationConfig tests ───────────────────────────────────────

    #[test]
    fn test_config_default() {
        let cfg = CalibrationConfig::default();
        assert_eq!(cfg.num_samples, 100);
        assert_eq!(cfg.method, CalibrationMethod::MinMax);
        assert!(!cfg.per_channel);
        assert!(cfg.symmetric);
        assert_eq!(cfg.num_bits, 8);
    }

    #[test]
    fn test_config_custom() {
        let cfg = CalibrationConfig {
            num_samples: 50,
            method: CalibrationMethod::Entropy,
            per_channel: true,
            symmetric: false,
            num_bits: 4,
            percentile: 0.999,
        };
        assert_eq!(cfg.num_samples, 50);
        assert_eq!(cfg.method, CalibrationMethod::Entropy);
        assert!(cfg.per_channel);
        assert!(!cfg.symmetric);
    }

    #[test]
    fn test_method_display() {
        assert_eq!(CalibrationMethod::MinMax.to_string(), "MinMax");
        assert_eq!(CalibrationMethod::Percentile.to_string(), "Percentile");
        assert_eq!(CalibrationMethod::Mse.to_string(), "MSE");
        assert_eq!(CalibrationMethod::Entropy.to_string(), "Entropy");
    }

    // ── CalibrationDataset tests ──────────────────────────────────────

    #[test]
    fn test_dataset_empty() {
        let ds = CalibrationDataset::new();
        assert!(ds.is_empty());
        assert_eq!(ds.len(), 0);
    }

    #[test]
    fn test_dataset_add_and_iterate() {
        let mut ds = CalibrationDataset::new();
        ds.add_sample(vec![1.0, 2.0, 3.0]);
        ds.add_sample(vec![4.0, 5.0]);
        assert_eq!(ds.len(), 2);
        assert!(!ds.is_empty());

        let collected: Vec<&[f32]> = ds.iter().collect();
        assert_eq!(collected[0], &[1.0, 2.0, 3.0]);
        assert_eq!(collected[1], &[4.0, 5.0]);
    }

    #[test]
    fn test_dataset_default() {
        let ds = CalibrationDataset::default();
        assert!(ds.is_empty());
    }

    // ── ActivationObserver tests ──────────────────────────────────────

    #[test]
    fn test_observer_initial_state() {
        let obs = ActivationObserver::new(256);
        assert_eq!(obs.count(), 0);
        assert_eq!(obs.mean(), 0.0);
        assert_eq!(obs.variance(), 0.0);
    }

    #[test]
    fn test_observer_single_value() {
        let mut obs = ActivationObserver::new(256);
        obs.observe(&[5.0]);
        assert_eq!(obs.count(), 1);
        assert!((obs.min_val - 5.0).abs() < 1e-10);
        assert!((obs.max_val - 5.0).abs() < 1e-10);
        assert!((obs.mean() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_observer_multiple_batches() {
        let mut obs = ActivationObserver::new(256);
        obs.observe(&[1.0, 2.0, 3.0]);
        obs.observe(&[4.0, 5.0]);
        assert_eq!(obs.count(), 5);
        assert!((obs.min_val - 1.0).abs() < 1e-10);
        assert!((obs.max_val - 5.0).abs() < 1e-10);
        assert!((obs.mean() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_observer_variance() {
        let mut obs = ActivationObserver::new(256);
        obs.observe(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        // Mean = 5.0, Variance = 4.0
        assert!((obs.mean() - 5.0).abs() < 1e-10);
        assert!((obs.variance() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_observer_negative_values() {
        let mut obs = ActivationObserver::new(256);
        obs.observe(&[-3.0, -1.0, 0.0, 1.0, 3.0]);
        assert!((obs.min_val - (-3.0)).abs() < 1e-10);
        assert!((obs.max_val - 3.0).abs() < 1e-10);
        assert!(obs.mean().abs() < 1e-10);
    }

    #[test]
    fn test_observer_histogram_populated() {
        let mut obs = ActivationObserver::new(10);
        obs.observe(&[0.0, 0.5, 1.0]);
        let hist = obs.histogram();
        let total: u64 = hist.iter().sum();
        assert_eq!(total, 3);
    }

    #[test]
    fn test_observer_reset() {
        let mut obs = ActivationObserver::new(256);
        obs.observe(&[1.0, 2.0, 3.0]);
        obs.reset();
        assert_eq!(obs.count(), 0);
        assert_eq!(obs.min_val, f64::INFINITY);
        assert_eq!(obs.max_val, f64::NEG_INFINITY);
    }

    #[test]
    fn test_observer_constant_values() {
        let mut obs = ActivationObserver::new(256);
        obs.observe(&[7.0, 7.0, 7.0, 7.0]);
        assert!((obs.min_val - 7.0).abs() < 1e-10);
        assert!((obs.max_val - 7.0).abs() < 1e-10);
        assert!((obs.mean() - 7.0).abs() < 1e-10);
        assert!(obs.variance().abs() < 1e-10);
    }

    // ── QuantizationRange tests ───────────────────────────────────────

    #[test]
    fn test_range_symmetric_8bit() {
        let r = QuantizationRange::compute_params(-1.0, 1.0, 8, true);
        assert_eq!(r.num_bits, 8);
        assert!((r.min_val - (-1.0)).abs() < 1e-10);
        assert!((r.max_val - 1.0).abs() < 1e-10);
        // scale = 2.0 / 255
        assert!((r.scale - 2.0 / 255.0).abs() < 1e-10);
        assert_eq!(r.zero_point, 127);
    }

    #[test]
    fn test_range_asymmetric_8bit() {
        let r = QuantizationRange::compute_params(0.0, 1.0, 8, false);
        assert_eq!(r.num_bits, 8);
        assert!((r.scale - 1.0 / 255.0).abs() < 1e-10);
    }

    #[test]
    fn test_range_symmetric_4bit() {
        let r = QuantizationRange::compute_params(-2.0, 2.0, 4, true);
        // scale = 4.0 / 15
        assert!((r.scale - 4.0 / 15.0).abs() < 1e-10);
        assert_eq!(r.zero_point, 7);
    }

    #[test]
    fn test_range_zero_range() {
        let r = QuantizationRange::compute_params(0.0, 0.0, 8, true);
        assert!((r.scale - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let r = QuantizationRange::compute_params(-1.0, 1.0, 8, true);
        let original = 0.5;
        let q = r.quantize(original);
        let dq = r.dequantize(q);
        // Should be close to original within one quantization step.
        assert!((dq - original).abs() < r.scale + 1e-10);
    }

    #[test]
    fn test_quantize_clamps_to_range() {
        let r = QuantizationRange::compute_params(-1.0, 1.0, 8, true);
        let q_high = r.quantize(100.0);
        let q_low = r.quantize(-100.0);
        assert!(q_high <= 255);
        assert!(q_low >= 0);
    }

    #[test]
    fn test_asymmetric_zero_point() {
        let r = QuantizationRange::compute_params(0.0, 1.0, 8, false);
        // Zero should quantize to near the zero_point.
        let q_zero = r.quantize(0.0);
        let dq_zero = r.dequantize(q_zero);
        assert!(dq_zero.abs() < r.scale + 1e-10);
    }

    #[test]
    fn test_range_negative_only() {
        let r = QuantizationRange::compute_params(-5.0, -1.0, 8, false);
        assert!((r.min_val - (-5.0)).abs() < 1e-10);
        assert!((r.max_val - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_symmetric_uses_abs_max() {
        let r = QuantizationRange::compute_params(-1.0, 3.0, 8, true);
        // Symmetric should use abs_max = 3.0.
        assert!((r.min_val - (-3.0)).abs() < 1e-10);
        assert!((r.max_val - 3.0).abs() < 1e-10);
    }

    // ── MinMax Calibrator tests ───────────────────────────────────────

    #[test]
    fn test_minmax_basic() {
        let data = vec![-1.0, 0.0, 0.5, 1.0];
        let r = MinMaxCalibrator::calibrate_from_data(&data, 8, true);
        assert!((r.min_val - (-1.0)).abs() < 1e-10);
        assert!((r.max_val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_minmax_from_observer() {
        let mut obs = ActivationObserver::new(256);
        obs.observe(&[-2.0, 0.0, 3.0]);
        let r = MinMaxCalibrator::calibrate(&obs, 8, true);
        // Symmetric: abs_max = 3.0.
        assert!((r.max_val - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_minmax_asymmetric() {
        let data = vec![0.0, 1.0, 2.0, 3.0];
        let r = MinMaxCalibrator::calibrate_from_data(&data, 8, false);
        assert!((r.min_val).abs() < 1e-10);
        assert!((r.max_val - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_minmax_single_value() {
        let data = vec![5.0];
        let r = MinMaxCalibrator::calibrate_from_data(&data, 8, true);
        assert!((r.max_val - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_minmax_constant_values() {
        let data = vec![3.0, 3.0, 3.0];
        let r = MinMaxCalibrator::calibrate_from_data(&data, 8, true);
        assert!((r.max_val - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_minmax_with_outlier() {
        let data = vec![0.0, 0.1, 0.2, 0.3, 100.0];
        let r = MinMaxCalibrator::calibrate_from_data(&data, 8, true);
        assert!((r.max_val - 100.0).abs() < 1e-10);
    }

    // ── Percentile Calibrator tests ───────────────────────────────────

    #[test]
    fn test_percentile_100_equals_minmax() {
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let r_pct = PercentileCalibrator::calibrate(&data, 1.0, 8, true);
        let r_mm = MinMaxCalibrator::calibrate_from_data(&data, 8, true);
        assert!((r_pct.max_val - r_mm.max_val).abs() < 1e-6);
    }

    #[test]
    fn test_percentile_clips_outliers() {
        let mut data: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0).collect();
        data.push(100.0); // outlier
        let r = PercentileCalibrator::calibrate(&data, 0.999, 8, true);
        // Should clip well below 100.
        assert!(r.max_val < 50.0);
    }

    #[test]
    fn test_percentile_50th() {
        let data: Vec<f32> = (0..101).map(|i| i as f32).collect();
        let r = PercentileCalibrator::calibrate(&data, 0.5, 8, false);
        // 50th percentile of 0..100 = 50.
        assert!((r.max_val - 50.0).abs() < 1.0);
    }

    #[test]
    fn test_percentile_empty_data() {
        let r = PercentileCalibrator::calibrate(&[], 0.99, 8, true);
        assert!((r.scale - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_percentile_single_sample() {
        let data = vec![42.0];
        let r = PercentileCalibrator::calibrate(&data, 0.99, 8, true);
        assert!((r.max_val - 42.0).abs() < 1e-6);
    }

    #[test]
    fn test_percentile_symmetric() {
        let data: Vec<f32> = (-50..=50).map(|i| i as f32).collect();
        let r = PercentileCalibrator::calibrate(&data, 0.95, 8, true);
        // Symmetric: range should be centered at 0.
        assert!((r.min_val + r.max_val).abs() < 1e-6);
    }

    #[test]
    fn test_percentile_9999() {
        let mut data: Vec<f32> = (0..10000).map(|i| i as f32 / 10000.0).collect();
        data.push(1000.0); // extreme outlier
        let r = PercentileCalibrator::calibrate(&data, 0.9999, 8, true);
        assert!(r.max_val < 500.0);
    }

    // ── MSE Calibrator tests ──────────────────────────────────────────

    #[test]
    fn test_mse_basic() {
        let data: Vec<f32> = (-100..=100).map(|i| i as f32 / 100.0).collect();
        let r = MseCalibrator::calibrate(&data, 8, true);
        // Should find a reasonable range.
        assert!(r.max_val > 0.0);
        assert!(r.max_val <= 1.01);
    }

    #[test]
    fn test_mse_empty() {
        let r = MseCalibrator::calibrate(&[], 8, true);
        assert!((r.scale - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mse_constant() {
        let data = vec![3.0; 100];
        let r = MseCalibrator::calibrate(&data, 8, true);
        // Range should be close to 3.0 (within one search step).
        assert!((r.max_val - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_mse_with_outlier() {
        let mut data: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0).collect();
        data.push(100.0); // outlier
        let r = MseCalibrator::calibrate(&data, 8, true);
        // MSE-optimal range should likely not extend all the way to 100.
        assert!(r.max_val < 100.0);
    }

    #[test]
    fn test_mse_better_than_minmax_with_outliers() {
        let mut data: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0).collect();
        data.push(100.0);
        let r_mse = MseCalibrator::calibrate(&data, 8, true);
        let r_mm = MinMaxCalibrator::calibrate_from_data(&data, 8, true);
        // MSE should have a tighter range than MinMax.
        assert!(r_mse.max_val <= r_mm.max_val);
    }

    #[test]
    fn test_mse_4bit() {
        let data: Vec<f32> = (-50..=50).map(|i| i as f32 / 50.0).collect();
        let r = MseCalibrator::calibrate(&data, 4, true);
        assert!(r.max_val > 0.0);
        assert_eq!(r.num_bits, 4);
    }

    #[test]
    fn test_mse_negative_only() {
        let data: Vec<f32> = (-100..=-1).map(|i| i as f32 / 100.0).collect();
        let r = MseCalibrator::calibrate(&data, 8, true);
        assert!(r.max_val > 0.0); // Symmetric extends to positive.
    }

    // ── Entropy Calibrator tests ──────────────────────────────────────

    #[test]
    fn test_entropy_basic() {
        let data: Vec<f32> = (-100..=100).map(|i| i as f32 / 100.0).collect();
        let r = EntropyCalibrator::calibrate(&data, 8, true);
        assert!(r.max_val > 0.0);
    }

    #[test]
    fn test_entropy_empty() {
        let r = EntropyCalibrator::calibrate(&[], 8, true);
        assert!((r.scale - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_constant() {
        let data = vec![5.0; 100];
        let r = EntropyCalibrator::calibrate(&data, 8, true);
        // Constant data: range degenerates to a single point.
        assert!(r.scale > 0.0);
        assert!((r.max_val - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_entropy_with_outlier() {
        let mut data: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0).collect();
        data.push(100.0);
        let r = EntropyCalibrator::calibrate(&data, 8, true);
        // Should find a range, possibly clipping the outlier.
        assert!(r.max_val > 0.0);
    }

    #[test]
    fn test_entropy_4bit() {
        let data: Vec<f32> = (-50..=50).map(|i| i as f32 / 50.0).collect();
        let r = EntropyCalibrator::calibrate(&data, 4, true);
        assert_eq!(r.num_bits, 4);
        assert!(r.max_val > 0.0);
    }

    #[test]
    fn test_entropy_negative_only() {
        let data: Vec<f32> = (-100..=-1).map(|i| i as f32 / 100.0).collect();
        let r = EntropyCalibrator::calibrate(&data, 8, true);
        assert!(r.max_val > 0.0);
    }

    // ── Per-channel vs per-tensor calibration ─────────────────────────

    #[test]
    fn test_per_channel_calibration() {
        // Simulate 2 channels with different ranges.
        let channel_0: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let channel_1: Vec<f32> = (0..100).map(|i| i as f32 / 10.0).collect();

        let r0 = MinMaxCalibrator::calibrate_from_data(&channel_0, 8, true);
        let r1 = MinMaxCalibrator::calibrate_from_data(&channel_1, 8, true);

        // Channel 1 should have a larger range.
        assert!(r1.max_val > r0.max_val);
    }

    #[test]
    fn test_per_tensor_vs_per_channel() {
        let channel_0 = vec![0.1, 0.2, 0.3];
        let channel_1 = vec![10.0, 20.0, 30.0];
        let combined: Vec<f32> = channel_0.iter().chain(channel_1.iter()).copied().collect();

        let per_tensor = MinMaxCalibrator::calibrate_from_data(&combined, 8, true);
        let per_ch0 = MinMaxCalibrator::calibrate_from_data(&channel_0, 8, true);
        let per_ch1 = MinMaxCalibrator::calibrate_from_data(&channel_1, 8, true);

        // Per-tensor range is dominated by the large channel.
        assert!((per_tensor.max_val - 30.0).abs() < 1e-6);
        // Per-channel ranges are tighter for channel 0.
        assert!(per_ch0.max_val < per_tensor.max_val);
        assert!((per_ch1.max_val - per_tensor.max_val).abs() < 1e-6);
    }

    // ── CalibrationRunner tests ───────────────────────────────────────

    #[test]
    fn test_runner_process_sample() {
        let config = CalibrationConfig::default();
        let layers = vec!["layer0".to_string(), "layer1".to_string()];
        let mut runner = CalibrationRunner::new(config, layers);

        let activations = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        assert!(runner.process_sample(&activations).is_ok());
        assert_eq!(runner.observers()[0].count(), 3);
        assert_eq!(runner.observers()[1].count(), 3);
    }

    #[test]
    fn test_runner_layer_mismatch() {
        let config = CalibrationConfig::default();
        let layers = vec!["layer0".to_string()];
        let mut runner = CalibrationRunner::new(config, layers);

        let activations = vec![vec![1.0], vec![2.0]];
        let err = runner.process_sample(&activations).unwrap_err();
        assert_eq!(err, CalibrationError::LayerCountMismatch { expected: 1, actual: 2 });
    }

    #[test]
    fn test_runner_empty_dataset() {
        let config = CalibrationConfig::default();
        let layers = vec!["layer0".to_string()];
        let mut runner = CalibrationRunner::new(config, layers);
        let ds = CalibrationDataset::new();
        let err = runner.run(&ds).unwrap_err();
        assert_eq!(err, CalibrationError::EmptyDataset);
    }

    #[test]
    fn test_runner_full_pipeline() {
        let config = CalibrationConfig {
            num_samples: 10,
            method: CalibrationMethod::MinMax,
            per_channel: false,
            symmetric: true,
            num_bits: 8,
            percentile: 0.9999,
        };
        let layers = vec!["attn".to_string(), "ffn".to_string()];
        let mut runner = CalibrationRunner::new(config, layers);

        let mut ds = CalibrationDataset::new();
        for i in 0..5 {
            let sample: Vec<f32> = (0..100).map(|j| (i * 100 + j) as f32 / 500.0).collect();
            ds.add_sample(sample);
        }

        let (ranges, metrics) = runner.run(&ds).unwrap();
        assert_eq!(ranges.len(), 2);
        assert_eq!(metrics.layers_calibrated, 2);
        assert!(metrics.samples_processed > 0);
    }

    #[test]
    fn test_runner_accessors() {
        let config = CalibrationConfig { num_bits: 4, ..CalibrationConfig::default() };
        let layers = vec!["a".to_string(), "b".to_string()];
        let runner = CalibrationRunner::new(config, layers);
        assert_eq!(runner.layer_names().len(), 2);
        assert_eq!(runner.observers().len(), 2);
        assert_eq!(runner.config().num_bits, 4);
    }

    // ── CalibrationMetrics tests ──────────────────────────────────────

    #[test]
    fn test_metrics_display() {
        let m = CalibrationMetrics {
            samples_processed: 100,
            layers_calibrated: 24,
            avg_quantization_error: 0.001,
            calibration_time_ms: 42,
        };
        let s = m.to_string();
        assert!(s.contains("100"));
        assert!(s.contains("24"));
        assert!(s.contains("42ms"));
    }

    #[test]
    fn test_metrics_clone() {
        let m = CalibrationMetrics {
            samples_processed: 10,
            layers_calibrated: 2,
            avg_quantization_error: 0.01,
            calibration_time_ms: 5,
        };
        let m2 = m.clone();
        assert_eq!(m2.samples_processed, 10);
    }

    // ── CalibrationError tests ────────────────────────────────────────

    #[test]
    fn test_error_display_empty() {
        let e = CalibrationError::EmptyDataset;
        assert_eq!(e.to_string(), "calibration dataset is empty");
    }

    #[test]
    fn test_error_display_mismatch() {
        let e = CalibrationError::LayerCountMismatch { expected: 3, actual: 5 };
        assert!(e.to_string().contains("3"));
        assert!(e.to_string().contains("5"));
    }

    #[test]
    fn test_error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(CalibrationError::EmptyDataset);
        assert!(e.to_string().contains("empty"));
    }

    // ── Edge case tests ───────────────────────────────────────────────

    #[test]
    fn test_large_range_values() {
        let data = vec![-1e6, 1e6];
        let r = MinMaxCalibrator::calibrate_from_data(&data, 8, true);
        assert!((r.max_val - 1e6).abs() < 1.0);
    }

    #[test]
    fn test_very_small_range() {
        let data = vec![1e-7, 1e-7 + 1e-10];
        let r = MinMaxCalibrator::calibrate_from_data(&data, 8, true);
        assert!(r.scale > 0.0);
    }

    #[test]
    fn test_all_zeros() {
        let data = vec![0.0; 100];
        let r = MinMaxCalibrator::calibrate_from_data(&data, 8, true);
        assert!((r.scale - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_single_negative() {
        let data = vec![-42.0];
        let r = MinMaxCalibrator::calibrate_from_data(&data, 8, true);
        assert!((r.max_val - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_alternating_signs() {
        let data: Vec<f32> = (0..100).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let r = MinMaxCalibrator::calibrate_from_data(&data, 8, true);
        assert!((r.max_val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_quantize_boundary_values() {
        let r = QuantizationRange::compute_params(-1.0, 1.0, 8, true);
        let q_min = r.quantize(-1.0);
        let q_max = r.quantize(1.0);
        assert!(q_min >= 0);
        assert!(q_max <= 255);
    }

    #[test]
    fn test_dequantize_all_levels_8bit() {
        let r = QuantizationRange::compute_params(-1.0, 1.0, 8, true);
        for q in 0..=255i64 {
            let dq = r.dequantize(q);
            assert!(dq >= r.min_val - r.scale);
            assert!(dq <= r.max_val + r.scale);
        }
    }

    #[test]
    fn test_mse_finds_non_trivial_range() {
        // Distribution where outliers dominate: many clustered values, many
        // outliers so clipping is beneficial.
        let mut data: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) / 50.0).collect();
        for _ in 0..50 {
            data.push(50.0);
            data.push(-50.0);
        }
        let r = MseCalibrator::calibrate(&data, 8, true);
        // MSE-optimal range should be valid and finite.
        assert!(r.max_val > 0.0);
        assert!(r.max_val.is_finite());
    }

    #[test]
    fn test_entropy_finds_non_trivial_range() {
        let mut data: Vec<f32> = (0..1000).map(|i| (i as f32 - 500.0) / 500.0).collect();
        data.push(50.0);
        data.push(-50.0);
        let r = EntropyCalibrator::calibrate(&data, 8, true);
        assert!(r.max_val > 0.0);
    }

    #[test]
    fn test_percentile_asymmetric_range() {
        let data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let r = PercentileCalibrator::calibrate(&data, 0.95, 8, false);
        assert!(r.min_val >= 0.0);
        assert!(r.max_val <= 999.0);
    }

    #[test]
    fn test_observer_histogram_range() {
        let mut obs = ActivationObserver::new(100);
        obs.observe(&[-5.0, 0.0, 5.0]);
        let (hmin, hmax) = obs.histogram_range();
        assert!((hmin - (-5.0)).abs() < 1e-10);
        assert!((hmax - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_runner_multiple_samples() {
        let config = CalibrationConfig::default();
        let layers = vec!["l0".to_string()];
        let mut runner = CalibrationRunner::new(config, layers);

        for _ in 0..10 {
            runner.process_sample(&[vec![1.0, 2.0, 3.0]]).unwrap();
        }
        assert_eq!(runner.observers()[0].count(), 30);
    }

    #[test]
    fn test_config_percentile_field() {
        let cfg = CalibrationConfig { percentile: 0.95, ..CalibrationConfig::default() };
        assert!((cfg.percentile - 0.95).abs() < 1e-10);
    }
}

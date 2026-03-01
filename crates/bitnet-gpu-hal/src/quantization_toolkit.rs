//! Module stub - implementation pending merge from feature branch
//! Quantization toolkit for GPU-accelerated inference.
//!
//! Provides absmax, min-max, and per-channel quantization with calibration,
//! error measurement (MSE / SNR / cosine similarity), mixed-precision
//! selection, dequantization kernels, and a unified [`QuantizationToolkit`]
//! entry-point.
//!
//! All routines have CPU reference implementations; GPU dispatch is a future
//! extension behind the `gpu` feature gate.

use std::fmt;

// ── Configuration ───────────────────────────────────────────────────────────

/// Quantization method selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantMethod {
    /// Symmetric absmax quantization.
    Absmax,
    /// Asymmetric min-max quantization.
    MinMax,
    /// Per-channel quantization (falls back to per-tensor when channel count is 1).
    PerChannel,
}

impl fmt::Display for QuantMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Absmax => write!(f, "absmax"),
            Self::MinMax => write!(f, "minmax"),
            Self::PerChannel => write!(f, "per-channel"),
        }
    }
}

/// Calibration strategy for post-training quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationStrategy {
    /// No calibration — use raw tensor statistics.
    None,
    /// Use min/max of calibration samples.
    MinMax,
    /// Use a running exponential moving average of statistics.
    MovingAverage { window: usize },
    /// Use percentile clipping to reduce outlier impact.
    Percentile { pct: u32 },
}

/// Configuration for a quantization pass.
#[derive(Debug, Clone)]
pub struct QuantConfig {
    /// Quantization method to apply.
    pub method: QuantMethod,
    /// Target bit-width (e.g. 1, 2, 4, 8).
    pub bits: u8,
    /// Whether to use symmetric range (`[-max, max]`).
    pub symmetric: bool,
    /// Whether to quantize per-channel instead of per-tensor.
    pub per_channel: bool,
    /// Calibration strategy.
    pub calibration: CalibrationStrategy,
}

impl QuantConfig {
    /// Create a new quantization config.
    pub fn new(method: QuantMethod, bits: u8) -> Self {
        Self {
            method,
            bits,
            symmetric: matches!(method, QuantMethod::Absmax | QuantMethod::PerChannel),
            per_channel: method == QuantMethod::PerChannel,
            calibration: CalibrationStrategy::None,
        }
    }

    /// Builder: set symmetric flag.
    pub fn with_symmetric(mut self, symmetric: bool) -> Self {
        self.symmetric = symmetric;
        self
    }

    /// Builder: set per-channel flag.
    pub fn with_per_channel(mut self, per_channel: bool) -> Self {
        self.per_channel = per_channel;
        self
    }

    /// Builder: set calibration strategy.
    pub fn with_calibration(mut self, calibration: CalibrationStrategy) -> Self {
        self.calibration = calibration;
        self
    }

    /// Maximum representable integer for the configured bit-width (unsigned).
    pub fn max_quant_val(&self) -> i64 {
        (1i64 << self.bits) - 1
    }

    /// Quantization range `[qmin, qmax]`.
    pub fn quant_range(&self) -> (i64, i64) {
        if self.symmetric {
            let half = (1i64 << (self.bits - 1)) - 1;
            (-half, half)
        } else {
            (0, self.max_quant_val())
        }
    }
}

impl Default for QuantConfig {
    fn default() -> Self {
        Self::new(QuantMethod::Absmax, 8)
    }
}

// ── Absmax quantizer ────────────────────────────────────────────────────────

/// Symmetric absmax quantizer: `scale = max(|x|) / qmax`.
#[derive(Debug, Clone)]
pub struct AbsmaxQuantizer {
    config: QuantConfig,
}

impl AbsmaxQuantizer {
    /// Create a new absmax quantizer with the given bit-width.
    pub fn new(bits: u8) -> Self {
        Self { config: QuantConfig::new(QuantMethod::Absmax, bits) }
    }

    /// Return the underlying config.
    pub fn config(&self) -> &QuantConfig {
        &self.config
    }

    /// Compute the absmax scale for `data`.
    pub fn compute_scale(&self, data: &[f32]) -> f32 {
        if data.is_empty() {
            return 1.0;
        }
        let absmax = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let (_, qmax) = self.config.quant_range();
        if absmax == 0.0 { 1.0 } else { absmax / qmax as f32 }
    }

    /// Quantize `data` to `i8` values, returning `(quantized, scale)`.
    pub fn quantize(&self, data: &[f32]) -> (Vec<i8>, f32) {
        let scale = self.compute_scale(data);
        let (qmin, qmax) = self.config.quant_range();
        let quantized = data
            .iter()
            .map(|&v| (v / scale).round().clamp(qmin as f32, qmax as f32) as i8)
            .collect();
        (quantized, scale)
    }

    /// Dequantize `i8` values back to `f32`.
    pub fn dequantize(&self, quantized: &[i8], scale: f32) -> Vec<f32> {
        quantized.iter().map(|&q| q as f32 * scale).collect()
    }
}

// ── MinMax quantizer ────────────────────────────────────────────────────────

/// Asymmetric min-max quantizer: maps `[min, max]` → `[0, 2^bits - 1]`.
#[derive(Debug, Clone)]
pub struct MinMaxQuantizer {
    config: QuantConfig,
}

impl MinMaxQuantizer {
    /// Create a new min-max quantizer with the given bit-width.
    pub fn new(bits: u8) -> Self {
        Self { config: QuantConfig::new(QuantMethod::MinMax, bits).with_symmetric(false) }
    }

    /// Return the underlying config.
    pub fn config(&self) -> &QuantConfig {
        &self.config
    }

    /// Compute `(scale, zero_point)` for `data`.
    pub fn compute_params(&self, data: &[f32]) -> (f32, f32) {
        if data.is_empty() {
            return (1.0, 0.0);
        }
        let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let (qmin, qmax) = self.config.quant_range();
        let range = max - min;
        if range == 0.0 {
            return (1.0, -min);
        }
        let scale = range / (qmax - qmin) as f32;
        let zero_point = -(min / scale) + qmin as f32;
        (scale, zero_point)
    }

    /// Quantize `data` to `u8` values, returning `(quantized, scale, zero_point)`.
    pub fn quantize(&self, data: &[f32]) -> (Vec<u8>, f32, f32) {
        let (scale, zp) = self.compute_params(data);
        let (qmin, qmax) = self.config.quant_range();
        let quantized = data
            .iter()
            .map(|&v| (v / scale + zp).round().clamp(qmin as f32, qmax as f32) as u8)
            .collect();
        (quantized, scale, zp)
    }

    /// Dequantize `u8` values back to `f32`.
    pub fn dequantize(&self, quantized: &[u8], scale: f32, zero_point: f32) -> Vec<f32> {
        quantized.iter().map(|&q| (q as f32 - zero_point) * scale).collect()
    }
}

// ── Per-channel quantizer ───────────────────────────────────────────────────

/// Per-channel quantizer: computes independent scales per output channel.
///
/// Falls back to per-tensor (single channel) when `channels == 1`.
#[derive(Debug, Clone)]
pub struct PerChannelQuantizer {
    config: QuantConfig,
}

impl PerChannelQuantizer {
    /// Create a per-channel quantizer with the given bit-width.
    pub fn new(bits: u8) -> Self {
        Self { config: QuantConfig::new(QuantMethod::PerChannel, bits) }
    }

    /// Return the underlying config.
    pub fn config(&self) -> &QuantConfig {
        &self.config
    }

    /// Quantize a 2-D tensor stored row-major as `[channels × channel_size]`.
    ///
    /// Returns `(quantized_flat, scales)` where `scales.len() == channels`.
    pub fn quantize(&self, data: &[f32], channels: usize) -> (Vec<i8>, Vec<f32>) {
        assert!(channels > 0, "channels must be > 0");
        assert!(data.len() % channels == 0, "data length must be divisible by channels");
        let ch_size = data.len() / channels;
        let (qmin, qmax) = self.config.quant_range();
        let mut quantized = Vec::with_capacity(data.len());
        let mut scales = Vec::with_capacity(channels);

        for ch in 0..channels {
            let start = ch * ch_size;
            let end = start + ch_size;
            let slice = &data[start..end];
            let absmax = slice.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = if absmax == 0.0 { 1.0 } else { absmax / qmax as f32 };
            scales.push(scale);
            for &v in slice {
                let q = (v / scale).round().clamp(qmin as f32, qmax as f32) as i8;
                quantized.push(q);
            }
        }
        (quantized, scales)
    }

    /// Dequantize per-channel quantized values.
    pub fn dequantize(&self, quantized: &[i8], scales: &[f32], channels: usize) -> Vec<f32> {
        assert!(channels > 0);
        assert_eq!(scales.len(), channels);
        assert!(quantized.len() % channels == 0);
        let ch_size = quantized.len() / channels;
        let mut output = Vec::with_capacity(quantized.len());
        for ch in 0..channels {
            let start = ch * ch_size;
            let end = start + ch_size;
            let scale = scales[ch];
            for &q in &quantized[start..end] {
                output.push(q as f32 * scale);
            }
        }
        output
    }
}

// ── Calibration dataset ─────────────────────────────────────────────────────

/// Accumulates calibration samples for post-training quantization.
#[derive(Debug, Clone)]
pub struct CalibrationDataset {
    samples: Vec<Vec<f32>>,
    strategy: CalibrationStrategy,
    running_min: f32,
    running_max: f32,
    sample_count: usize,
}

impl CalibrationDataset {
    /// Create an empty calibration dataset with the given strategy.
    pub fn new(strategy: CalibrationStrategy) -> Self {
        Self {
            samples: Vec::new(),
            strategy,
            running_min: f32::INFINITY,
            running_max: f32::NEG_INFINITY,
            sample_count: 0,
        }
    }

    /// Add a calibration sample (activation tensor snapshot).
    pub fn add_sample(&mut self, sample: Vec<f32>) {
        for &v in &sample {
            if v < self.running_min {
                self.running_min = v;
            }
            if v > self.running_max {
                self.running_max = v;
            }
        }
        self.sample_count += 1;
        self.samples.push(sample);
    }

    /// Number of samples collected so far.
    pub fn len(&self) -> usize {
        self.sample_count
    }

    /// Whether the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.sample_count == 0
    }

    /// The calibration strategy in use.
    pub fn strategy(&self) -> CalibrationStrategy {
        self.strategy
    }

    /// Observed global min across all samples.
    pub fn global_min(&self) -> f32 {
        self.running_min
    }

    /// Observed global max across all samples.
    pub fn global_max(&self) -> f32 {
        self.running_max
    }

    /// Compute the calibrated `(min, max)` range using the configured strategy.
    pub fn calibrated_range(&self) -> (f32, f32) {
        if self.is_empty() {
            return (0.0, 0.0);
        }
        match self.strategy {
            CalibrationStrategy::None | CalibrationStrategy::MinMax => {
                (self.running_min, self.running_max)
            }
            CalibrationStrategy::MovingAverage { window } => {
                let n = self.samples.len();
                let start = if n > window { n - window } else { 0 };
                let recent = &self.samples[start..];
                let mut rmin = f32::INFINITY;
                let mut rmax = f32::NEG_INFINITY;
                for s in recent {
                    for &v in s {
                        if v < rmin {
                            rmin = v;
                        }
                        if v > rmax {
                            rmax = v;
                        }
                    }
                }
                (rmin, rmax)
            }
            CalibrationStrategy::Percentile { pct } => {
                let mut all_vals: Vec<f32> =
                    self.samples.iter().flat_map(|s| s.iter().copied()).collect();
                all_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                if all_vals.is_empty() {
                    return (0.0, 0.0);
                }
                let pct_f = pct as f64 / 100.0;
                let lo_idx = ((1.0 - pct_f) / 2.0 * all_vals.len() as f64).floor() as usize;
                let hi_idx = ((1.0 - (1.0 - pct_f) / 2.0) * all_vals.len() as f64).ceil() as usize;
                let lo_idx = lo_idx.min(all_vals.len() - 1);
                let hi_idx = hi_idx.min(all_vals.len() - 1);
                (all_vals[lo_idx], all_vals[hi_idx])
            }
        }
    }
}

// ── Quantization error metrics ──────────────────────────────────────────────

/// Error metrics comparing original vs. dequantized tensors.
#[derive(Debug, Clone)]
pub struct QuantizationError {
    /// Mean Squared Error.
    pub mse: f64,
    /// Signal-to-Noise Ratio in dB.
    pub snr_db: f64,
    /// Cosine similarity (1.0 = identical direction).
    pub cosine_similarity: f64,
    /// Maximum absolute error.
    pub max_abs_error: f64,
}

impl QuantizationError {
    /// Compute error metrics between `original` and `reconstructed` tensors.
    pub fn compute(original: &[f32], reconstructed: &[f32]) -> Self {
        assert_eq!(original.len(), reconstructed.len(), "tensor lengths must match");
        let n = original.len() as f64;
        if n == 0.0 {
            return Self {
                mse: 0.0,
                snr_db: f64::INFINITY,
                cosine_similarity: 1.0,
                max_abs_error: 0.0,
            };
        }

        let mut sum_sq_err = 0.0f64;
        let mut sum_sq_sig = 0.0f64;
        let mut dot = 0.0f64;
        let mut mag_a = 0.0f64;
        let mut mag_b = 0.0f64;
        let mut max_err = 0.0f64;

        for (&o, &r) in original.iter().zip(reconstructed.iter()) {
            let o64 = o as f64;
            let r64 = r as f64;
            let err = o64 - r64;
            sum_sq_err += err * err;
            sum_sq_sig += o64 * o64;
            dot += o64 * r64;
            mag_a += o64 * o64;
            mag_b += r64 * r64;
            let ae = err.abs();
            if ae > max_err {
                max_err = ae;
            }
        }

        let mse = sum_sq_err / n;
        let snr_db = if sum_sq_err == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (sum_sq_sig / sum_sq_err).log10()
        };
        let denom = (mag_a.sqrt()) * (mag_b.sqrt());
        let cosine_similarity = if denom == 0.0 { 1.0 } else { dot / denom };

        Self { mse, snr_db, cosine_similarity, max_abs_error: max_err }
    }
}

impl fmt::Display for QuantizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MSE={:.6e}  SNR={:.2} dB  cos={:.6}  max_err={:.6e}",
            self.mse, self.snr_db, self.cosine_similarity, self.max_abs_error
        )
    }
}

// ── Mixed-precision selector ────────────────────────────────────────────────

/// Per-layer sensitivity record.
#[derive(Debug, Clone)]
pub struct LayerSensitivity {
    /// Layer name or index.
    pub name: String,
    /// Error metric when quantized at each candidate bit-width.
    pub errors: Vec<(u8, QuantizationError)>,
}

/// Selects per-layer bit-width based on sensitivity analysis.
#[derive(Debug, Clone)]
pub struct MixedPrecisionSelector {
    /// Candidate bit-widths (e.g. `[2, 4, 8]`).
    pub candidates: Vec<u8>,
    /// Maximum allowed MSE per layer.
    pub mse_threshold: f64,
    /// Minimum required cosine similarity per layer.
    pub cosine_threshold: f64,
}

impl MixedPrecisionSelector {
    /// Create a selector with candidate bit-widths and quality thresholds.
    pub fn new(candidates: Vec<u8>, mse_threshold: f64, cosine_threshold: f64) -> Self {
        Self { candidates, mse_threshold, cosine_threshold }
    }

    /// Select the lowest bit-width for a layer that meets quality thresholds.
    ///
    /// Returns `(bits, error)`.  Falls back to the highest candidate if none
    /// meet the threshold.
    pub fn select(&self, sensitivity: &LayerSensitivity) -> (u8, QuantizationError) {
        // Sort candidates ascending (prefer fewer bits).
        let mut cands: Vec<u8> = self.candidates.clone();
        cands.sort();

        for &bits in &cands {
            if let Some((_, err)) = sensitivity.errors.iter().find(|(b, _)| *b == bits) {
                if err.mse <= self.mse_threshold && err.cosine_similarity >= self.cosine_threshold {
                    return (bits, err.clone());
                }
            }
        }

        // Fallback: highest bit-width.
        let fallback_bits = *cands.last().unwrap_or(&8);
        if let Some((_, err)) = sensitivity.errors.iter().find(|(b, _)| *b == fallback_bits) {
            (fallback_bits, err.clone())
        } else {
            (
                fallback_bits,
                QuantizationError {
                    mse: 0.0,
                    snr_db: f64::INFINITY,
                    cosine_similarity: 1.0,
                    max_abs_error: 0.0,
                },
            )
        }
    }

    /// Analyze a layer at all candidate bit-widths and select the best.
    ///
    /// `quantize_fn(data, bits)` should return the dequantized reconstruction.
    pub fn analyze_layer<F>(
        &self,
        name: &str,
        data: &[f32],
        quantize_fn: F,
    ) -> (u8, LayerSensitivity)
    where
        F: Fn(&[f32], u8) -> Vec<f32>,
    {
        let errors: Vec<(u8, QuantizationError)> = self
            .candidates
            .iter()
            .map(|&bits| {
                let recon = quantize_fn(data, bits);
                let err = QuantizationError::compute(data, &recon);
                (bits, err)
            })
            .collect();
        let sensitivity = LayerSensitivity { name: name.to_string(), errors };
        let (bits, _) = self.select(&sensitivity);
        (bits, sensitivity)
    }
}

// ── Dequantization kernel ───────────────────────────────────────────────────

/// CPU dequantization kernels optimised for inference.
///
/// In the future, GPU-dispatch variants will sit behind the `gpu` feature gate.
#[derive(Debug, Clone)]
pub struct DequantKernel {
    bits: u8,
}

impl DequantKernel {
    /// Create a dequant kernel for the given bit-width.
    pub fn new(bits: u8) -> Self {
        Self { bits }
    }

    /// Bit-width this kernel targets.
    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// Dequantize symmetric absmax-quantized `i8` values in-place into `output`.
    pub fn dequant_absmax(&self, quantized: &[i8], scale: f32, output: &mut [f32]) {
        assert_eq!(quantized.len(), output.len());
        for (o, &q) in output.iter_mut().zip(quantized.iter()) {
            *o = q as f32 * scale;
        }
    }

    /// Dequantize asymmetric min-max-quantized `u8` values in-place into `output`.
    pub fn dequant_minmax(
        &self,
        quantized: &[u8],
        scale: f32,
        zero_point: f32,
        output: &mut [f32],
    ) {
        assert_eq!(quantized.len(), output.len());
        for (o, &q) in output.iter_mut().zip(quantized.iter()) {
            *o = (q as f32 - zero_point) * scale;
        }
    }

    /// Dequantize per-channel symmetric `i8` values in-place.
    pub fn dequant_per_channel(
        &self,
        quantized: &[i8],
        scales: &[f32],
        channels: usize,
        output: &mut [f32],
    ) {
        assert_eq!(quantized.len(), output.len());
        assert!(channels > 0);
        assert_eq!(scales.len(), channels);
        let ch_size = quantized.len() / channels;
        for ch in 0..channels {
            let start = ch * ch_size;
            let end = start + ch_size;
            let s = scales[ch];
            for i in start..end {
                output[i] = quantized[i] as f32 * s;
            }
        }
    }

    /// Fused dequantize + bias-add for absmax-quantized values.
    pub fn dequant_absmax_bias(
        &self,
        quantized: &[i8],
        scale: f32,
        bias: &[f32],
        output: &mut [f32],
    ) {
        assert_eq!(quantized.len(), output.len());
        assert_eq!(bias.len(), output.len());
        for i in 0..output.len() {
            output[i] = quantized[i] as f32 * scale + bias[i];
        }
    }
}

// ── Quantization report ─────────────────────────────────────────────────────

/// Per-layer entry in a [`QuantizationReport`].
#[derive(Debug, Clone)]
pub struct LayerReport {
    /// Layer name.
    pub name: String,
    /// Bit-width chosen for this layer.
    pub bits: u8,
    /// Error metrics at the chosen bit-width.
    pub error: QuantizationError,
    /// Method used.
    pub method: QuantMethod,
}

/// Aggregated quantization quality report.
#[derive(Debug, Clone)]
pub struct QuantizationReport {
    /// Per-layer reports.
    pub layers: Vec<LayerReport>,
}

impl QuantizationReport {
    /// Create an empty report.
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Add a layer entry.
    pub fn add_layer(
        &mut self,
        name: impl Into<String>,
        bits: u8,
        error: QuantizationError,
        method: QuantMethod,
    ) {
        self.layers.push(LayerReport { name: name.into(), bits, error, method });
    }

    /// Average MSE across all layers.
    pub fn avg_mse(&self) -> f64 {
        if self.layers.is_empty() {
            return 0.0;
        }
        self.layers.iter().map(|l| l.error.mse).sum::<f64>() / self.layers.len() as f64
    }

    /// Average cosine similarity across all layers.
    pub fn avg_cosine(&self) -> f64 {
        if self.layers.is_empty() {
            return 1.0;
        }
        self.layers.iter().map(|l| l.error.cosine_similarity).sum::<f64>()
            / self.layers.len() as f64
    }

    /// Worst-case (maximum) MSE across all layers.
    pub fn max_mse(&self) -> f64 {
        self.layers.iter().map(|l| l.error.mse).fold(0.0, f64::max)
    }

    /// Worst-case (minimum) cosine similarity across all layers.
    pub fn min_cosine(&self) -> f64 {
        self.layers.iter().map(|l| l.error.cosine_similarity).fold(1.0, f64::min)
    }

    /// Number of layers in the report.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }
}

impl Default for QuantizationReport {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for QuantizationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Quantization Report ({} layers)", self.layers.len())?;
        writeln!(f, "  avg MSE      = {:.6e}", self.avg_mse())?;
        writeln!(f, "  avg cosine   = {:.6}", self.avg_cosine())?;
        writeln!(f, "  worst MSE    = {:.6e}", self.max_mse())?;
        writeln!(f, "  worst cosine = {:.6}", self.min_cosine())?;
        for l in &self.layers {
            writeln!(f, "  [{:>3}-bit {}] {}: {}", l.bits, l.method, l.name, l.error)?;
        }
        Ok(())
    }
}

// ── Unified toolkit ─────────────────────────────────────────────────────────

/// Unified entry-point for all quantization operations.
///
/// Wraps all quantizer variants, calibration, error measurement,
/// mixed-precision selection, dequantization kernels, and reporting.
#[derive(Debug, Clone)]
pub struct QuantizationToolkit {
    config: QuantConfig,
}

impl QuantizationToolkit {
    /// Create a toolkit with the given configuration.
    pub fn new(config: QuantConfig) -> Self {
        Self { config }
    }

    /// Return the active configuration.
    pub fn config(&self) -> &QuantConfig {
        &self.config
    }

    /// Quantize `data` using the configured method, returning `(i8 values, scale)`.
    ///
    /// For `MinMax` the zero-point is folded into the returned scale by
    /// returning a symmetric approximation (useful for uniform APIs).
    pub fn quantize(&self, data: &[f32]) -> (Vec<i8>, f32) {
        match self.config.method {
            QuantMethod::Absmax => {
                let q = AbsmaxQuantizer::new(self.config.bits);
                q.quantize(data)
            }
            QuantMethod::MinMax => {
                let q = MinMaxQuantizer::new(self.config.bits);
                let (quant_u8, scale, zp) = q.quantize(data);
                // Convert u8 to i8 by shifting around zero_point.
                let quantized: Vec<i8> = quant_u8
                    .iter()
                    .map(|&v| (v as i16 - zp.round() as i16).clamp(-128, 127) as i8)
                    .collect();
                (quantized, scale)
            }
            QuantMethod::PerChannel => {
                // Default to 1-channel (per-tensor) when called via the unified API.
                let q = PerChannelQuantizer::new(self.config.bits);
                let (quantized, scales) = q.quantize(data, 1);
                (quantized, scales[0])
            }
        }
    }

    /// Dequantize `i8` values back to `f32`.
    pub fn dequantize(&self, quantized: &[i8], scale: f32) -> Vec<f32> {
        match self.config.method {
            QuantMethod::Absmax | QuantMethod::PerChannel => {
                let q = AbsmaxQuantizer::new(self.config.bits);
                q.dequantize(quantized, scale)
            }
            QuantMethod::MinMax => {
                // Symmetric approximation for dequant.
                quantized.iter().map(|&q| q as f32 * scale).collect()
            }
        }
    }

    /// Quantize then immediately dequantize (roundtrip).
    pub fn roundtrip(&self, data: &[f32]) -> Vec<f32> {
        let (q, s) = self.quantize(data);
        self.dequantize(&q, s)
    }

    /// Measure quantization error for `data`.
    pub fn measure_error(&self, data: &[f32]) -> QuantizationError {
        let recon = self.roundtrip(data);
        QuantizationError::compute(data, &recon)
    }

    /// Build a per-channel quantization report for `data` split into `channels`.
    pub fn per_channel_report(
        &self,
        data: &[f32],
        channels: usize,
        layer_prefix: &str,
    ) -> QuantizationReport {
        assert!(channels > 0);
        assert!(data.len() % channels == 0);
        let ch_size = data.len() / channels;
        let mut report = QuantizationReport::new();
        for ch in 0..channels {
            let start = ch * ch_size;
            let end = start + ch_size;
            let slice = &data[start..end];
            let err = self.measure_error(slice);
            report.add_layer(
                format!("{layer_prefix}/ch{ch}"),
                self.config.bits,
                err,
                self.config.method,
            );
        }
        report
    }

    /// Create a [`DequantKernel`] for the toolkit's bit-width.
    pub fn dequant_kernel(&self) -> DequantKernel {
        DequantKernel::new(self.config.bits)
    }

    /// Create a [`CalibrationDataset`] using the toolkit's calibration strategy.
    pub fn calibration_dataset(&self) -> CalibrationDataset {
        CalibrationDataset::new(self.config.calibration)
    }

    /// Create a [`MixedPrecisionSelector`] with the given candidates and thresholds.
    pub fn mixed_precision_selector(
        candidates: Vec<u8>,
        mse_threshold: f64,
        cosine_threshold: f64,
    ) -> MixedPrecisionSelector {
        MixedPrecisionSelector::new(candidates, mse_threshold, cosine_threshold)
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: approximate f32 equality.
    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    // ── QuantConfig tests ───────────────────────────────────────────────

    #[test]
    fn config_default_is_8bit_absmax() {
        let c = QuantConfig::default();
        assert_eq!(c.method, QuantMethod::Absmax);
        assert_eq!(c.bits, 8);
        assert!(c.symmetric);
    }

    #[test]
    fn config_max_quant_val_8bit() {
        let c = QuantConfig::new(QuantMethod::Absmax, 8);
        assert_eq!(c.max_quant_val(), 255);
    }

    #[test]
    fn config_max_quant_val_4bit() {
        let c = QuantConfig::new(QuantMethod::Absmax, 4);
        assert_eq!(c.max_quant_val(), 15);
    }

    #[test]
    fn config_max_quant_val_2bit() {
        let c = QuantConfig::new(QuantMethod::Absmax, 2);
        assert_eq!(c.max_quant_val(), 3);
    }

    #[test]
    fn config_quant_range_symmetric_8bit() {
        let c = QuantConfig::new(QuantMethod::Absmax, 8);
        assert_eq!(c.quant_range(), (-127, 127));
    }

    #[test]
    fn config_quant_range_asymmetric_8bit() {
        let c = QuantConfig::new(QuantMethod::MinMax, 8).with_symmetric(false);
        assert_eq!(c.quant_range(), (0, 255));
    }

    #[test]
    fn config_builder_per_channel() {
        let c = QuantConfig::new(QuantMethod::Absmax, 4).with_per_channel(true);
        assert!(c.per_channel);
    }

    #[test]
    fn config_builder_calibration() {
        let c = QuantConfig::new(QuantMethod::Absmax, 8)
            .with_calibration(CalibrationStrategy::Percentile { pct: 99 });
        assert!(matches!(c.calibration, CalibrationStrategy::Percentile { pct: 99 }));
    }

    #[test]
    fn quant_method_display() {
        assert_eq!(format!("{}", QuantMethod::Absmax), "absmax");
        assert_eq!(format!("{}", QuantMethod::MinMax), "minmax");
        assert_eq!(format!("{}", QuantMethod::PerChannel), "per-channel");
    }

    #[test]
    fn config_quant_range_symmetric_4bit() {
        let c = QuantConfig::new(QuantMethod::Absmax, 4);
        assert_eq!(c.quant_range(), (-7, 7));
    }

    // ── Absmax quantizer tests ──────────────────────────────────────────

    #[test]
    fn absmax_roundtrip_zeros() {
        let q = AbsmaxQuantizer::new(8);
        let data = vec![0.0; 10];
        let (quant, scale) = q.quantize(&data);
        let recon = q.dequantize(&quant, scale);
        for v in &recon {
            assert!(approx_eq(*v, 0.0, 1e-6));
        }
    }

    #[test]
    fn absmax_roundtrip_identity_at_integers() {
        let q = AbsmaxQuantizer::new(8);
        let data = vec![1.0, -1.0, 0.5, -0.5, 0.0];
        let (quant, scale) = q.quantize(&data);
        let recon = q.dequantize(&quant, scale);
        for (&o, &r) in data.iter().zip(recon.iter()) {
            assert!(approx_eq(o, r, 0.02), "orig={o} recon={r}");
        }
    }

    #[test]
    fn absmax_scale_positive() {
        let q = AbsmaxQuantizer::new(8);
        let data = vec![3.0, -2.0, 1.0];
        let scale = q.compute_scale(&data);
        assert!(scale > 0.0);
    }

    #[test]
    fn absmax_quantized_within_range() {
        let q = AbsmaxQuantizer::new(8);
        let data: Vec<f32> = (-100..=100).map(|i| i as f32 * 0.01).collect();
        let (quant, _) = q.quantize(&data);
        for &v in &quant {
            assert!((v as i16) >= -127 && (v as i16) <= 127);
        }
    }

    #[test]
    fn absmax_empty_input() {
        let q = AbsmaxQuantizer::new(8);
        let (quant, scale) = q.quantize(&[]);
        assert!(quant.is_empty());
        assert_eq!(scale, 1.0);
    }

    #[test]
    fn absmax_single_value() {
        let q = AbsmaxQuantizer::new(8);
        let (quant, scale) = q.quantize(&[5.0]);
        assert_eq!(quant.len(), 1);
        let recon = q.dequantize(&quant, scale);
        assert!(approx_eq(recon[0], 5.0, 0.05));
    }

    #[test]
    fn absmax_large_range() {
        let q = AbsmaxQuantizer::new(8);
        let data = vec![-1000.0, 0.0, 1000.0];
        let (quant, scale) = q.quantize(&data);
        let recon = q.dequantize(&quant, scale);
        assert!(approx_eq(recon[0], -1000.0, 10.0));
        assert!(approx_eq(recon[2], 1000.0, 10.0));
    }

    #[test]
    fn absmax_4bit_range() {
        let q = AbsmaxQuantizer::new(4);
        let data: Vec<f32> = (-10..=10).map(|i| i as f32 * 0.1).collect();
        let (quant, _) = q.quantize(&data);
        for &v in &quant {
            assert!(v >= -7 && v <= 7);
        }
    }

    #[test]
    fn absmax_2bit_range() {
        let q = AbsmaxQuantizer::new(2);
        let data = vec![-1.0, 0.0, 1.0];
        let (quant, _) = q.quantize(&data);
        for &v in &quant {
            assert!(v >= -1 && v <= 1);
        }
    }

    #[test]
    fn absmax_negative_only() {
        let q = AbsmaxQuantizer::new(8);
        let data = vec![-5.0, -3.0, -1.0];
        let (quant, scale) = q.quantize(&data);
        let recon = q.dequantize(&quant, scale);
        for (&o, &r) in data.iter().zip(recon.iter()) {
            assert!(approx_eq(o, r, 0.1));
        }
    }

    #[test]
    fn absmax_config_accessor() {
        let q = AbsmaxQuantizer::new(4);
        assert_eq!(q.config().bits, 4);
        assert_eq!(q.config().method, QuantMethod::Absmax);
    }

    // ── MinMax quantizer tests ──────────────────────────────────────────

    #[test]
    fn minmax_roundtrip_basic() {
        let q = MinMaxQuantizer::new(8);
        let data = vec![0.0, 0.5, 1.0];
        let (quant, scale, zp) = q.quantize(&data);
        let recon = q.dequantize(&quant, scale, zp);
        for (&o, &r) in data.iter().zip(recon.iter()) {
            assert!(approx_eq(o, r, 0.01), "orig={o} recon={r}");
        }
    }

    #[test]
    fn minmax_roundtrip_negative() {
        let q = MinMaxQuantizer::new(8);
        let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let (quant, scale, zp) = q.quantize(&data);
        let recon = q.dequantize(&quant, scale, zp);
        for (&o, &r) in data.iter().zip(recon.iter()) {
            assert!(approx_eq(o, r, 0.05), "orig={o} recon={r}");
        }
    }

    #[test]
    fn minmax_empty_input() {
        let q = MinMaxQuantizer::new(8);
        let (quant, scale, _) = q.quantize(&[]);
        assert!(quant.is_empty());
        assert_eq!(scale, 1.0);
    }

    #[test]
    fn minmax_constant_input() {
        let q = MinMaxQuantizer::new(8);
        let data = vec![3.0, 3.0, 3.0];
        let (quant, scale, zp) = q.quantize(&data);
        let recon = q.dequantize(&quant, scale, zp);
        for &r in &recon {
            assert!(approx_eq(r, 3.0, 0.1));
        }
    }

    #[test]
    fn minmax_quantized_in_range() {
        let q = MinMaxQuantizer::new(8);
        let data: Vec<f32> = (-50..=50).map(|i| i as f32 * 0.1).collect();
        let (quant, _, _) = q.quantize(&data);
        for &v in &quant {
            assert!((v as u16) <= 255);
        }
    }

    #[test]
    fn minmax_4bit() {
        let q = MinMaxQuantizer::new(4);
        let data = vec![0.0, 0.5, 1.0];
        let (quant, _, _) = q.quantize(&data);
        for &v in &quant {
            assert!(v <= 15);
        }
    }

    #[test]
    fn minmax_scale_zero_point() {
        let q = MinMaxQuantizer::new(8);
        let (scale, zp) = q.compute_params(&[0.0, 1.0]);
        assert!(scale > 0.0);
        assert!(zp >= 0.0);
    }

    #[test]
    fn minmax_config_accessor() {
        let q = MinMaxQuantizer::new(8);
        assert_eq!(q.config().method, QuantMethod::MinMax);
        assert!(!q.config().symmetric);
    }

    // ── Per-channel quantizer tests ─────────────────────────────────────

    #[test]
    fn perchannel_single_channel_matches_absmax() {
        let pc = PerChannelQuantizer::new(8);
        let am = AbsmaxQuantizer::new(8);
        let data = vec![1.0, -2.0, 3.0, -4.0];
        let (q_pc, scales) = pc.quantize(&data, 1);
        let (q_am, scale_am) = am.quantize(&data);
        assert_eq!(q_pc, q_am);
        assert!(approx_eq(scales[0], scale_am, 1e-6));
    }

    #[test]
    fn perchannel_two_channels() {
        let q = PerChannelQuantizer::new(8);
        let data = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let (quant, scales) = q.quantize(&data, 2);
        assert_eq!(scales.len(), 2);
        assert_eq!(quant.len(), 6);
        let recon = q.dequantize(&quant, &scales, 2);
        for (&o, &r) in data.iter().zip(recon.iter()) {
            assert!(approx_eq(o, r, 0.5), "orig={o} recon={r}");
        }
    }

    #[test]
    fn perchannel_four_channels() {
        let q = PerChannelQuantizer::new(8);
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let (quant, scales) = q.quantize(&data, 4);
        assert_eq!(scales.len(), 4);
        let recon = q.dequantize(&quant, &scales, 4);
        for (&o, &r) in data.iter().zip(recon.iter()) {
            assert!(approx_eq(o, r, 0.15), "orig={o} recon={r}");
        }
    }

    #[test]
    #[should_panic(expected = "channels must be > 0")]
    fn perchannel_zero_channels_panics() {
        let q = PerChannelQuantizer::new(8);
        q.quantize(&[1.0], 0);
    }

    #[test]
    #[should_panic(expected = "data length must be divisible by channels")]
    fn perchannel_mismatched_length_panics() {
        let q = PerChannelQuantizer::new(8);
        q.quantize(&[1.0, 2.0, 3.0], 2);
    }

    #[test]
    fn perchannel_all_zeros() {
        let q = PerChannelQuantizer::new(8);
        let data = vec![0.0; 8];
        let (quant, scales) = q.quantize(&data, 2);
        let recon = q.dequantize(&quant, &scales, 2);
        for v in &recon {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn perchannel_config_accessor() {
        let q = PerChannelQuantizer::new(4);
        assert_eq!(q.config().method, QuantMethod::PerChannel);
        assert!(q.config().per_channel);
    }

    // ── Calibration dataset tests ───────────────────────────────────────

    #[test]
    fn calibration_empty() {
        let cd = CalibrationDataset::new(CalibrationStrategy::None);
        assert!(cd.is_empty());
        assert_eq!(cd.len(), 0);
    }

    #[test]
    fn calibration_add_samples() {
        let mut cd = CalibrationDataset::new(CalibrationStrategy::MinMax);
        cd.add_sample(vec![1.0, 2.0, 3.0]);
        cd.add_sample(vec![-1.0, 5.0]);
        assert_eq!(cd.len(), 2);
        assert!(!cd.is_empty());
    }

    #[test]
    fn calibration_global_min_max() {
        let mut cd = CalibrationDataset::new(CalibrationStrategy::MinMax);
        cd.add_sample(vec![1.0, 2.0]);
        cd.add_sample(vec![-3.0, 0.5]);
        assert!(approx_eq(cd.global_min(), -3.0, 1e-6));
        assert!(approx_eq(cd.global_max(), 2.0, 1e-6));
    }

    #[test]
    fn calibration_range_none_strategy() {
        let mut cd = CalibrationDataset::new(CalibrationStrategy::None);
        cd.add_sample(vec![-1.0, 2.0]);
        let (lo, hi) = cd.calibrated_range();
        assert!(approx_eq(lo, -1.0, 1e-6));
        assert!(approx_eq(hi, 2.0, 1e-6));
    }

    #[test]
    fn calibration_range_minmax_strategy() {
        let mut cd = CalibrationDataset::new(CalibrationStrategy::MinMax);
        cd.add_sample(vec![-5.0, 10.0]);
        let (lo, hi) = cd.calibrated_range();
        assert!(approx_eq(lo, -5.0, 1e-6));
        assert!(approx_eq(hi, 10.0, 1e-6));
    }

    #[test]
    fn calibration_range_moving_average() {
        let mut cd = CalibrationDataset::new(CalibrationStrategy::MovingAverage { window: 2 });
        cd.add_sample(vec![-100.0, 100.0]); // will be outside window
        cd.add_sample(vec![-1.0, 1.0]);
        cd.add_sample(vec![-2.0, 2.0]);
        let (lo, hi) = cd.calibrated_range();
        // Only last 2 samples should be considered.
        assert!(approx_eq(lo, -2.0, 1e-6));
        assert!(approx_eq(hi, 2.0, 1e-6));
    }

    #[test]
    fn calibration_range_percentile() {
        let mut cd = CalibrationDataset::new(CalibrationStrategy::Percentile { pct: 90 });
        let sample: Vec<f32> = (0..100).map(|i| i as f32).collect();
        cd.add_sample(sample);
        let (lo, hi) = cd.calibrated_range();
        // 90th percentile: should clip the extremes.
        assert!(lo >= 0.0);
        assert!(hi <= 99.0);
    }

    #[test]
    fn calibration_empty_range() {
        let cd = CalibrationDataset::new(CalibrationStrategy::MinMax);
        let (lo, hi) = cd.calibrated_range();
        assert_eq!(lo, 0.0);
        assert_eq!(hi, 0.0);
    }

    #[test]
    fn calibration_strategy_accessor() {
        let cd = CalibrationDataset::new(CalibrationStrategy::MovingAverage { window: 5 });
        assert!(matches!(cd.strategy(), CalibrationStrategy::MovingAverage { window: 5 }));
    }

    // ── QuantizationError tests ─────────────────────────────────────────

    #[test]
    fn error_identical_tensors() {
        let data = vec![1.0, 2.0, 3.0];
        let err = QuantizationError::compute(&data, &data);
        assert_eq!(err.mse, 0.0);
        assert_eq!(err.snr_db, f64::INFINITY);
        assert!((err.cosine_similarity - 1.0).abs() < 1e-10);
        assert_eq!(err.max_abs_error, 0.0);
    }

    #[test]
    fn error_known_mse() {
        let orig = vec![1.0, 2.0, 3.0, 4.0];
        let recon = vec![1.1, 2.1, 3.1, 4.1];
        let err = QuantizationError::compute(&orig, &recon);
        // MSE = mean(0.01) = 0.01
        assert!((err.mse - 0.01).abs() < 1e-4);
    }

    #[test]
    fn error_cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let err = QuantizationError::compute(&a, &b);
        assert!(err.cosine_similarity.abs() < 1e-10);
    }

    #[test]
    fn error_cosine_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let err = QuantizationError::compute(&a, &b);
        assert!((err.cosine_similarity - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn error_max_abs() {
        let orig = vec![0.0, 0.0, 0.0];
        let recon = vec![0.1, 0.5, 0.2];
        let err = QuantizationError::compute(&orig, &recon);
        assert!((err.max_abs_error - 0.5).abs() < 1e-10);
    }

    #[test]
    fn error_empty_tensors() {
        let err = QuantizationError::compute(&[], &[]);
        assert_eq!(err.mse, 0.0);
        assert_eq!(err.cosine_similarity, 1.0);
    }

    #[test]
    fn error_display() {
        let err = QuantizationError {
            mse: 0.001,
            snr_db: 30.0,
            cosine_similarity: 0.999,
            max_abs_error: 0.01,
        };
        let s = format!("{err}");
        assert!(s.contains("MSE="));
        assert!(s.contains("SNR="));
    }

    #[test]
    fn error_snr_positive_for_small_error() {
        let orig: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        let recon: Vec<f32> = orig.iter().map(|&v| v + 0.001).collect();
        let err = QuantizationError::compute(&orig, &recon);
        assert!(err.snr_db > 0.0);
    }

    #[test]
    #[should_panic(expected = "tensor lengths must match")]
    fn error_mismatched_lengths_panics() {
        QuantizationError::compute(&[1.0], &[1.0, 2.0]);
    }

    // ── Mixed-precision selector tests ──────────────────────────────────

    #[test]
    fn mixed_precision_picks_lowest_meeting_threshold() {
        let sel = MixedPrecisionSelector::new(vec![2, 4, 8], 0.01, 0.99);
        let sensitivity = LayerSensitivity {
            name: "layer0".to_string(),
            errors: vec![
                (
                    2,
                    QuantizationError {
                        mse: 0.1,
                        snr_db: 10.0,
                        cosine_similarity: 0.9,
                        max_abs_error: 0.5,
                    },
                ),
                (
                    4,
                    QuantizationError {
                        mse: 0.005,
                        snr_db: 23.0,
                        cosine_similarity: 0.995,
                        max_abs_error: 0.05,
                    },
                ),
                (
                    8,
                    QuantizationError {
                        mse: 0.0001,
                        snr_db: 40.0,
                        cosine_similarity: 0.9999,
                        max_abs_error: 0.001,
                    },
                ),
            ],
        };
        let (bits, _) = sel.select(&sensitivity);
        assert_eq!(bits, 4);
    }

    #[test]
    fn mixed_precision_fallback_to_highest() {
        let sel = MixedPrecisionSelector::new(vec![2, 4, 8], 0.0001, 0.9999);
        let sensitivity = LayerSensitivity {
            name: "layer0".to_string(),
            errors: vec![
                (
                    2,
                    QuantizationError {
                        mse: 1.0,
                        snr_db: 0.0,
                        cosine_similarity: 0.5,
                        max_abs_error: 2.0,
                    },
                ),
                (
                    4,
                    QuantizationError {
                        mse: 0.5,
                        snr_db: 3.0,
                        cosine_similarity: 0.8,
                        max_abs_error: 1.0,
                    },
                ),
                (
                    8,
                    QuantizationError {
                        mse: 0.01,
                        snr_db: 20.0,
                        cosine_similarity: 0.99,
                        max_abs_error: 0.1,
                    },
                ),
            ],
        };
        let (bits, _) = sel.select(&sensitivity);
        assert_eq!(bits, 8);
    }

    #[test]
    fn mixed_precision_analyze_layer() {
        let sel = MixedPrecisionSelector::new(vec![4, 8], 0.1, 0.9);
        let data: Vec<f32> = (0..64).map(|i| (i as f32).sin()).collect();
        let (bits, sensitivity) = sel.analyze_layer("test_layer", &data, |d, b| {
            let q = AbsmaxQuantizer::new(b);
            let (quant, scale) = q.quantize(d);
            q.dequantize(&quant, scale)
        });
        assert!(bits == 4 || bits == 8);
        assert_eq!(sensitivity.name, "test_layer");
        assert_eq!(sensitivity.errors.len(), 2);
    }

    // ── DequantKernel tests ─────────────────────────────────────────────

    #[test]
    fn dequant_kernel_absmax_basic() {
        let k = DequantKernel::new(8);
        let quantized = vec![127i8, -127, 0, 64];
        let scale = 0.01;
        let mut output = vec![0.0f32; 4];
        k.dequant_absmax(&quantized, scale, &mut output);
        assert!(approx_eq(output[0], 1.27, 1e-4));
        assert!(approx_eq(output[1], -1.27, 1e-4));
        assert!(approx_eq(output[2], 0.0, 1e-6));
        assert!(approx_eq(output[3], 0.64, 1e-4));
    }

    #[test]
    fn dequant_kernel_minmax_basic() {
        let k = DequantKernel::new(8);
        let quantized = vec![0u8, 128, 255];
        let scale = 0.01;
        let zp = 128.0;
        let mut output = vec![0.0f32; 3];
        k.dequant_minmax(&quantized, scale, zp, &mut output);
        assert!(approx_eq(output[0], -1.28, 1e-4));
        assert!(approx_eq(output[1], 0.0, 1e-4));
        assert!(approx_eq(output[2], 1.27, 1e-4));
    }

    #[test]
    fn dequant_kernel_per_channel() {
        let k = DequantKernel::new(8);
        let quantized = vec![100i8, 50, -100, -50];
        let scales = vec![0.01, 0.02];
        let mut output = vec![0.0f32; 4];
        k.dequant_per_channel(&quantized, &scales, 2, &mut output);
        assert!(approx_eq(output[0], 1.0, 1e-4));
        assert!(approx_eq(output[1], 0.5, 1e-4));
        assert!(approx_eq(output[2], -2.0, 1e-4));
        assert!(approx_eq(output[3], -1.0, 1e-4));
    }

    #[test]
    fn dequant_kernel_absmax_bias() {
        let k = DequantKernel::new(8);
        let quantized = vec![100i8, -100];
        let scale = 0.01;
        let bias = vec![1.0, 2.0];
        let mut output = vec![0.0f32; 2];
        k.dequant_absmax_bias(&quantized, scale, &bias, &mut output);
        assert!(approx_eq(output[0], 2.0, 1e-4));
        assert!(approx_eq(output[1], 1.0, 1e-4));
    }

    #[test]
    fn dequant_kernel_bits_accessor() {
        let k = DequantKernel::new(4);
        assert_eq!(k.bits(), 4);
    }

    // ── QuantizationReport tests ────────────────────────────────────────

    #[test]
    fn report_empty() {
        let r = QuantizationReport::new();
        assert_eq!(r.layer_count(), 0);
        assert_eq!(r.avg_mse(), 0.0);
        assert_eq!(r.avg_cosine(), 1.0);
    }

    #[test]
    fn report_add_layers() {
        let mut r = QuantizationReport::new();
        r.add_layer(
            "layer0",
            8,
            QuantizationError {
                mse: 0.01,
                snr_db: 20.0,
                cosine_similarity: 0.99,
                max_abs_error: 0.05,
            },
            QuantMethod::Absmax,
        );
        r.add_layer(
            "layer1",
            4,
            QuantizationError {
                mse: 0.05,
                snr_db: 13.0,
                cosine_similarity: 0.95,
                max_abs_error: 0.1,
            },
            QuantMethod::MinMax,
        );
        assert_eq!(r.layer_count(), 2);
        assert!((r.avg_mse() - 0.03).abs() < 1e-10);
        assert!((r.avg_cosine() - 0.97).abs() < 1e-10);
    }

    #[test]
    fn report_max_mse() {
        let mut r = QuantizationReport::new();
        r.add_layer(
            "a",
            8,
            QuantizationError {
                mse: 0.01,
                snr_db: 20.0,
                cosine_similarity: 0.99,
                max_abs_error: 0.05,
            },
            QuantMethod::Absmax,
        );
        r.add_layer(
            "b",
            8,
            QuantizationError {
                mse: 0.1,
                snr_db: 10.0,
                cosine_similarity: 0.9,
                max_abs_error: 0.5,
            },
            QuantMethod::Absmax,
        );
        assert!((r.max_mse() - 0.1).abs() < 1e-10);
        assert!((r.min_cosine() - 0.9).abs() < 1e-10);
    }

    #[test]
    fn report_display() {
        let mut r = QuantizationReport::new();
        r.add_layer(
            "layer0",
            8,
            QuantizationError {
                mse: 0.001,
                snr_db: 30.0,
                cosine_similarity: 0.999,
                max_abs_error: 0.01,
            },
            QuantMethod::Absmax,
        );
        let s = format!("{r}");
        assert!(s.contains("Quantization Report"));
        assert!(s.contains("layer0"));
    }

    #[test]
    fn report_default() {
        let r = QuantizationReport::default();
        assert_eq!(r.layer_count(), 0);
    }

    // ── QuantizationToolkit tests ───────────────────────────────────────

    #[test]
    fn toolkit_absmax_roundtrip() {
        let tk = QuantizationToolkit::new(QuantConfig::new(QuantMethod::Absmax, 8));
        let data: Vec<f32> = (0..32).map(|i| (i as f32 * 0.1).sin()).collect();
        let recon = tk.roundtrip(&data);
        for (&o, &r) in data.iter().zip(recon.iter()) {
            assert!(approx_eq(o, r, 0.02), "orig={o} recon={r}");
        }
    }

    #[test]
    fn toolkit_minmax_quantize() {
        let tk = QuantizationToolkit::new(QuantConfig::new(QuantMethod::MinMax, 8));
        let data = vec![0.0, 0.5, 1.0];
        let (quant, scale) = tk.quantize(&data);
        assert_eq!(quant.len(), 3);
        assert!(scale > 0.0);
    }

    #[test]
    fn toolkit_perchannel_quantize() {
        let tk = QuantizationToolkit::new(QuantConfig::new(QuantMethod::PerChannel, 8));
        let data = vec![1.0, -1.0, 0.5, -0.5];
        let (quant, scale) = tk.quantize(&data);
        assert_eq!(quant.len(), 4);
        assert!(scale > 0.0);
    }

    #[test]
    fn toolkit_measure_error_low_for_8bit() {
        let tk = QuantizationToolkit::new(QuantConfig::new(QuantMethod::Absmax, 8));
        let data: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let err = tk.measure_error(&data);
        assert!(err.mse < 0.001, "MSE={}", err.mse);
        assert!(err.cosine_similarity > 0.99);
    }

    #[test]
    fn toolkit_4bit_higher_error_than_8bit() {
        let data: Vec<f32> = (0..256).map(|i| (i as f32 * 0.05).sin()).collect();
        let tk8 = QuantizationToolkit::new(QuantConfig::new(QuantMethod::Absmax, 8));
        let tk4 = QuantizationToolkit::new(QuantConfig::new(QuantMethod::Absmax, 4));
        let err8 = tk8.measure_error(&data);
        let err4 = tk4.measure_error(&data);
        assert!(
            err4.mse >= err8.mse,
            "4-bit MSE ({}) should be >= 8-bit MSE ({})",
            err4.mse,
            err8.mse
        );
    }

    #[test]
    fn toolkit_per_channel_report() {
        let tk = QuantizationToolkit::new(QuantConfig::new(QuantMethod::Absmax, 8));
        let data: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let report = tk.per_channel_report(&data, 4, "attn");
        assert_eq!(report.layer_count(), 4);
        assert!(report.layers[0].name.starts_with("attn/ch"));
    }

    #[test]
    fn toolkit_dequant_kernel_bits_match() {
        let tk = QuantizationToolkit::new(QuantConfig::new(QuantMethod::Absmax, 4));
        let k = tk.dequant_kernel();
        assert_eq!(k.bits(), 4);
    }

    #[test]
    fn toolkit_calibration_dataset_strategy() {
        let tk = QuantizationToolkit::new(
            QuantConfig::new(QuantMethod::Absmax, 8)
                .with_calibration(CalibrationStrategy::Percentile { pct: 95 }),
        );
        let cd = tk.calibration_dataset();
        assert!(matches!(cd.strategy(), CalibrationStrategy::Percentile { pct: 95 }));
    }

    #[test]
    fn toolkit_mixed_precision_selector() {
        let sel = QuantizationToolkit::mixed_precision_selector(vec![4, 8], 0.01, 0.99);
        assert_eq!(sel.candidates.len(), 2);
    }

    #[test]
    fn toolkit_config_accessor() {
        let cfg = QuantConfig::new(QuantMethod::MinMax, 4);
        let tk = QuantizationToolkit::new(cfg.clone());
        assert_eq!(tk.config().bits, 4);
        assert_eq!(tk.config().method, QuantMethod::MinMax);
    }

    // ── Roundtrip fidelity matrix ───────────────────────────────────────

    #[test]
    fn roundtrip_absmax_8bit_fidelity() {
        let q = AbsmaxQuantizer::new(8);
        let data: Vec<f32> = (0..128).map(|i| (i as f32 * 0.05).sin()).collect();
        let (quant, scale) = q.quantize(&data);
        let recon = q.dequantize(&quant, scale);
        let err = QuantizationError::compute(&data, &recon);
        assert!(err.mse < 1e-4);
        assert!(err.cosine_similarity > 0.999);
    }

    #[test]
    fn roundtrip_absmax_4bit_fidelity() {
        let q = AbsmaxQuantizer::new(4);
        let data: Vec<f32> = (0..128).map(|i| (i as f32 * 0.05).sin()).collect();
        let (quant, scale) = q.quantize(&data);
        let recon = q.dequantize(&quant, scale);
        let err = QuantizationError::compute(&data, &recon);
        assert!(err.mse < 0.01);
        assert!(err.cosine_similarity > 0.99);
    }

    #[test]
    fn roundtrip_minmax_8bit_fidelity() {
        let q = MinMaxQuantizer::new(8);
        let data: Vec<f32> = (0..128).map(|i| (i as f32 * 0.05).sin()).collect();
        let (quant, scale, zp) = q.quantize(&data);
        let recon = q.dequantize(&quant, scale, zp);
        let err = QuantizationError::compute(&data, &recon);
        assert!(err.mse < 1e-4);
        assert!(err.cosine_similarity > 0.999);
    }

    #[test]
    fn roundtrip_perchannel_8bit_fidelity() {
        let q = PerChannelQuantizer::new(8);
        let data: Vec<f32> = (0..128).map(|i| (i as f32 * 0.05).sin()).collect();
        let (quant, scales) = q.quantize(&data, 4);
        let recon = q.dequantize(&quant, &scales, 4);
        let err = QuantizationError::compute(&data, &recon);
        assert!(err.mse < 1e-4);
        assert!(err.cosine_similarity > 0.999);
    }

    // ── Edge-case tests ─────────────────────────────────────────────────

    #[test]
    fn absmax_tiny_values() {
        let q = AbsmaxQuantizer::new(8);
        let data = vec![1e-10, -1e-10, 5e-11];
        let (quant, scale) = q.quantize(&data);
        let recon = q.dequantize(&quant, scale);
        let err = QuantizationError::compute(&data, &recon);
        assert!(err.cosine_similarity > 0.9);
    }

    #[test]
    fn absmax_large_values() {
        let q = AbsmaxQuantizer::new(8);
        let data = vec![1e6, -1e6, 5e5];
        let (quant, scale) = q.quantize(&data);
        let recon = q.dequantize(&quant, scale);
        for (&o, &r) in data.iter().zip(recon.iter()) {
            let rel_err = ((o - r) / o).abs();
            assert!(rel_err < 0.01, "relative error too large: {rel_err}");
        }
    }

    #[test]
    fn minmax_single_element() {
        let q = MinMaxQuantizer::new(8);
        let (quant, scale, zp) = q.quantize(&[42.0]);
        let recon = q.dequantize(&quant, scale, zp);
        assert!(approx_eq(recon[0], 42.0, 1.0));
    }

    #[test]
    fn perchannel_many_channels() {
        let q = PerChannelQuantizer::new(8);
        let data: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
        let (quant, scales) = q.quantize(&data, 16);
        assert_eq!(scales.len(), 16);
        let recon = q.dequantize(&quant, &scales, 16);
        let err = QuantizationError::compute(&data, &recon);
        assert!(err.cosine_similarity > 0.99);
    }

    #[test]
    fn toolkit_roundtrip_preserves_sign() {
        let tk = QuantizationToolkit::new(QuantConfig::new(QuantMethod::Absmax, 8));
        let data = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let recon = tk.roundtrip(&data);
        for (&o, &r) in data.iter().zip(recon.iter()) {
            if o != 0.0 {
                assert!(o.signum() == r.signum(), "sign mismatch: orig={o} recon={r}");
            }
        }
    }

    #[test]
    fn calibration_percentile_clipping() {
        let mut cd = CalibrationDataset::new(CalibrationStrategy::Percentile { pct: 80 });
        let mut sample: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        sample.push(1_000_000.0); // outlier
        cd.add_sample(sample);
        let (_lo, hi) = cd.calibrated_range();
        assert!(hi < 1_000_000.0, "percentile should clip outlier");
    }

    #[test]
    fn error_snr_higher_at_higher_bits() {
        let data: Vec<f32> = (0..256).map(|i| (i as f32 * 0.05).sin()).collect();
        let q8 = AbsmaxQuantizer::new(8);
        let q4 = AbsmaxQuantizer::new(4);
        let (qv8, s8) = q8.quantize(&data);
        let (qv4, s4) = q4.quantize(&data);
        let err8 = QuantizationError::compute(&data, &q8.dequantize(&qv8, s8));
        let err4 = QuantizationError::compute(&data, &q4.dequantize(&qv4, s4));
        assert!(err8.snr_db > err4.snr_db);
    }

    #[test]
    fn dequant_kernel_matches_quantizer_output() {
        let q = AbsmaxQuantizer::new(8);
        let k = DequantKernel::new(8);
        let data = vec![1.0, -1.0, 0.5, -0.5];
        let (quant, scale) = q.quantize(&data);
        let recon_q = q.dequantize(&quant, scale);
        let mut recon_k = vec![0.0f32; 4];
        k.dequant_absmax(&quant, scale, &mut recon_k);
        assert_eq!(recon_q, recon_k);
    }

    #[test]
    fn report_worst_layer_identified() {
        let mut r = QuantizationReport::new();
        for i in 0..5 {
            let mse = 0.001 * (i as f64 + 1.0);
            r.add_layer(
                format!("layer{i}"),
                8,
                QuantizationError {
                    mse,
                    snr_db: 30.0 - i as f64,
                    cosine_similarity: 1.0 - mse,
                    max_abs_error: mse * 10.0,
                },
                QuantMethod::Absmax,
            );
        }
        assert!((r.max_mse() - 0.005).abs() < 1e-10);
        assert!((r.min_cosine() - 0.995).abs() < 1e-10);
    }

    #[test]
    fn toolkit_absmax_dequantize() {
        let tk = QuantizationToolkit::new(QuantConfig::new(QuantMethod::Absmax, 8));
        let data = vec![1.0, -1.0, 0.5];
        let (quant, scale) = tk.quantize(&data);
        let recon = tk.dequantize(&quant, scale);
        for (&o, &r) in data.iter().zip(recon.iter()) {
            assert!(approx_eq(o, r, 0.02));
        }
    }

    #[test]
    fn toolkit_minmax_dequantize() {
        let tk = QuantizationToolkit::new(QuantConfig::new(QuantMethod::MinMax, 8));
        let data = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let (quant, scale) = tk.quantize(&data);
        let recon = tk.dequantize(&quant, scale);
        assert_eq!(recon.len(), data.len());
    }

    #[test]
    fn calibration_moving_average_single_window() {
        let mut cd = CalibrationDataset::new(CalibrationStrategy::MovingAverage { window: 1 });
        cd.add_sample(vec![-100.0, 100.0]);
        cd.add_sample(vec![-1.0, 1.0]);
        let (lo, hi) = cd.calibrated_range();
        assert!(approx_eq(lo, -1.0, 1e-6));
        assert!(approx_eq(hi, 1.0, 1e-6));
    }

    #[test]
    fn mixed_precision_empty_errors() {
        let sel = MixedPrecisionSelector::new(vec![4, 8], 0.01, 0.99);
        let sensitivity = LayerSensitivity { name: "empty".to_string(), errors: vec![] };
        let (bits, _) = sel.select(&sensitivity);
        assert_eq!(bits, 8); // fallback
    }

    #[test]
    fn perchannel_dequant_matches_manual() {
        let q = PerChannelQuantizer::new(8);
        let data = vec![1.0, 2.0, 10.0, 20.0];
        let (quant, scales) = q.quantize(&data, 2);
        let recon = q.dequantize(&quant, &scales, 2);
        // Channel 0: max=2.0, scale=2.0/127
        // Channel 1: max=20.0, scale=20.0/127
        for (&o, &r) in data.iter().zip(recon.iter()) {
            assert!(approx_eq(o, r, 0.5));
        }
    }

    #[test]
    fn toolkit_2bit_quantization() {
        let tk = QuantizationToolkit::new(QuantConfig::new(QuantMethod::Absmax, 2));
        let data = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let (quant, _scale) = tk.quantize(&data);
        for &v in &quant {
            assert!(v >= -1 && v <= 1, "2-bit value out of range: {v}");
        }
    }
}

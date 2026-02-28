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
//! Comprehensive quantization toolkit for GPU inference.
//!
//! Provides multiple quantization schemes (symmetric, asymmetric,
//! per-channel, `BitNet` 1.58 ternary, GPTQ, AWQ), calibration methods,
//! bit-packing utilities, and statistics for measuring quantization quality.

use std::fmt;

// ── Quantization Schemes ──────────────────────────────────────────────────

/// Supported quantization schemes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantScheme {
    /// Symmetric quantization applied per-tensor (`zero_point` = 0).
    SymmetricPerTensor,
    /// Symmetric quantization applied per-channel.
    SymmetricPerChannel,
    /// Asymmetric quantization applied per-tensor.
    AsymmetricPerTensor,
    /// Asymmetric quantization applied per-channel.
    AsymmetricPerChannel,
    /// `BitNet` 1.58b ternary: values in {-1, 0, +1}.
    BitNet158,
    /// GPTQ-style groupwise quantization.
    Gptq,
    /// AWQ activation-aware quantization.
    Awq,
}

impl fmt::Display for QuantScheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SymmetricPerTensor => write!(f, "SymmetricPerTensor"),
            Self::SymmetricPerChannel => write!(f, "SymmetricPerChannel"),
            Self::AsymmetricPerTensor => write!(f, "AsymmetricPerTensor"),
            Self::AsymmetricPerChannel => write!(f, "AsymmetricPerChannel"),
            Self::BitNet158 => write!(f, "BitNet158"),
            Self::Gptq => write!(f, "GPTQ"),
            Self::Awq => write!(f, "AWQ"),
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
// ── Calibration ───────────────────────────────────────────────────────────

/// Method used to determine quantization parameters from calibration data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CalibrationMethod {
    /// Use the observed min/max of the data.
    MinMax,
    /// Clip outliers at the given percentile (e.g., 99.9).
    Percentile(f64),
    /// Minimize KL-divergence between original and quantized distributions.
    Entropy,
    /// Minimize mean-squared error between original and dequantized values.
    Mse,
}

// ── Configuration ─────────────────────────────────────────────────────────

/// Configuration for a [`Quantizer`].
#[derive(Debug, Clone)]
pub struct QuantConfig {
    /// Number of bits per quantized element.
    pub bits: u32,
    /// Whether quantization is symmetric (`zero_point` forced to 0).
    pub symmetric: bool,
    /// Optional group size for block-wise quantization (GPTQ / AWQ).
    pub group_size: Option<usize>,
    /// Calibration method for determining `scale`/`zero_point`.
    pub calibration_method: CalibrationMethod,
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
        Self {
            bits: 8,
            symmetric: true,
            group_size: None,
            calibration_method: CalibrationMethod::MinMax,
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
// ── Parameters ────────────────────────────────────────────────────────────

/// Parameters that define a particular quantization mapping.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantParams {
    /// Scale factor: `real_value = (quantized_value - zero_point) * scale`.
    pub scale: f64,
    /// Zero-point offset (0 for symmetric schemes).
    pub zero_point: i64,
    /// Bit-width of quantized values.
    pub bits: u32,
    /// Group size (for GPTQ / AWQ).
    pub group_size: Option<usize>,
    /// Channel axis (for per-channel schemes).
    pub channel_axis: Option<usize>,
}

// ── Quantized Tensor ──────────────────────────────────────────────────────

/// A tensor that has been quantized.
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Packed quantized data.
    pub data: Vec<u8>,
    /// Quantization parameters used.
    pub params: QuantParams,
    /// Original tensor shape.
    pub original_shape: Vec<usize>,
    /// Scheme that was applied.
    pub scheme: QuantScheme,
    /// Size of `data` in bytes.
    pub size_bytes: u64,
}

// ── Statistics ────────────────────────────────────────────────────────────

/// Statistics comparing original data with its quantized reconstruction.
#[derive(Debug, Clone)]
pub struct QuantStats {
    /// Size of the original data in bytes.
    pub original_size: u64,
    /// Size of the quantized data in bytes.
    pub quantized_size: u64,
    /// `original_size / quantized_size`.
    pub compression_ratio: f64,
    /// Maximum absolute element-wise error.
    pub max_quantization_error: f64,
    /// Mean absolute element-wise error.
    pub mean_quantization_error: f64,
    /// Signal-to-noise ratio in dB.
    pub snr_db: f64,
}

impl QuantStats {
    /// Compute statistics by comparing `original` data with its
    /// `dequantized` reconstruction.
    #[allow(clippy::cast_precision_loss)]
    pub fn compute(original: &[f32], dequantized: &[f32], quantized_size_bytes: u64) -> Self {
        let original_size = (original.len() * 4) as u64;

        let mut max_err: f64 = 0.0;
        let mut sum_err: f64 = 0.0;
        let mut signal_power: f64 = 0.0;
        let mut noise_power: f64 = 0.0;

        for (&o, &d) in original.iter().zip(dequantized.iter()) {
            let err = (f64::from(o) - f64::from(d)).abs();
            max_err = max_err.max(err);
            sum_err += err;
            signal_power += f64::from(o) * f64::from(o);
            noise_power += (f64::from(o) - f64::from(d)) * (f64::from(o) - f64::from(d));
        }

        let n = original.len().max(1) as f64;
        let mean_err = sum_err / n;

        let snr_db = if noise_power > 0.0 {
            10.0 * (signal_power / noise_power).log10()
        } else if signal_power > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        let compression_ratio = if quantized_size_bytes > 0 {
            original_size as f64 / quantized_size_bytes as f64
        } else {
            0.0
        };

        Self {
            original_size,
            quantized_size: quantized_size_bytes,
            compression_ratio,
            max_quantization_error: max_err,
            mean_quantization_error: mean_err,
            snr_db,
        }
    }
}

// ── Calibration helpers ───────────────────────────────────────────────────

/// Determine optimal [`QuantParams`] from a sample of `data`.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
pub fn calibrate(data: &[f32], config: &QuantConfig) -> QuantParams {
    if data.is_empty() {
        return QuantParams {
            scale: 1.0,
            zero_point: 0,
            bits: config.bits,
            group_size: config.group_size,
            channel_axis: None,
        };
    }

    let (min_val, max_val) = match config.calibration_method {
        CalibrationMethod::MinMax => data_min_max(data),
        CalibrationMethod::Percentile(p) => percentile_range(data, p),
        CalibrationMethod::Entropy | CalibrationMethod::Mse => {
            // Both entropy and MSE calibration start from MinMax bounds
            // then refine; for this toolkit we use a shrink heuristic.
            let (raw_min, raw_max) = data_min_max(data);
            let shrink = 0.99;
            (raw_min * shrink, raw_max * shrink)
        }
    };

    compute_params(min_val, max_val, config)
}

fn data_min_max(data: &[f32]) -> (f64, f64) {
    let min = f64::from(data.iter().copied().fold(f32::INFINITY, f32::min));
    let max = f64::from(data.iter().copied().fold(f32::NEG_INFINITY, f32::max));
    (min, max)
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss)]
fn percentile_range(data: &[f32], percentile: f64) -> (f64, f64) {
    let mut sorted: Vec<f32> = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let lo = (1.0 - percentile / 100.0) / 2.0;
    let hi = 1.0 - lo;

    let idx_lo = ((sorted.len() as f64 * lo).floor() as usize).min(sorted.len().saturating_sub(1));
    let idx_hi = ((sorted.len() as f64 * hi).ceil() as usize).min(sorted.len().saturating_sub(1));

    (f64::from(sorted[idx_lo]), f64::from(sorted[idx_hi]))
}

#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn compute_params(min_val: f64, max_val: f64, config: &QuantConfig) -> QuantParams {
    let qmin = 0_i64;
    let qmax = (1_i64 << config.bits) - 1;

    if config.symmetric {
        let abs_max = min_val.abs().max(max_val.abs());
        // Signed range: [-2^(bits-1), 2^(bits-1)-1], e.g. [-128, 127].
        let signed_max = ((1_i64 << (config.bits - 1)) - 1) as f64;
        let scale = if signed_max > 0.0 && abs_max > 0.0 { abs_max / signed_max } else { 1.0 };
        QuantParams {
            scale,
            zero_point: 0,
            bits: config.bits,
            group_size: config.group_size,
            channel_axis: None,
        }
    } else {
        let range = max_val - min_val;
        let qrange = (qmax - qmin) as f64;
        let scale = if qrange > 0.0 && range > 0.0 { range / qrange } else { 1.0 };
        let zero_point =
            if scale > 0.0 { (qmin as f64 - min_val / scale).round() as i64 } else { 0 };
        QuantParams {
            scale,
            zero_point,
            bits: config.bits,
            group_size: config.group_size,
            channel_axis: None,
        }
    }
}

// ── Quantizer ─────────────────────────────────────────────────────────────

/// Quantizer that converts `f32` data into [`QuantizedTensor`] using a
/// specified [`QuantScheme`] and [`QuantConfig`].
pub struct Quantizer {
    /// Active quantization scheme.
    pub scheme: QuantScheme,
    /// Configuration parameters.
    pub config: QuantConfig,
}

impl Quantizer {
    /// Create a new `Quantizer`.
    pub const fn new(scheme: QuantScheme, config: QuantConfig) -> Self {
        Self { scheme, config }
    }

    /// Quantize `data` with the given logical `shape`.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    pub fn quantize(
        &self,
        data: &[f32],
        shape: &[usize],
    ) -> Result<QuantizedTensor, QuantizeError> {
        let total: usize = shape.iter().product();
        if total != data.len() {
            return Err(QuantizeError::ShapeMismatch { expected: total, actual: data.len() });
        }

        match self.scheme {
            QuantScheme::BitNet158 => {
                let packed = pack_ternary(&ternary_quantize_values(data));
                let size = packed.len() as u64;
                Ok(QuantizedTensor {
                    data: packed,
                    params: QuantParams {
                        scale: ternary_scale(data),
                        zero_point: 0,
                        bits: 2,
                        group_size: None,
                        channel_axis: None,
                    },
                    original_shape: shape.to_vec(),
                    scheme: self.scheme,
                    size_bytes: size,
                })
            }
            QuantScheme::SymmetricPerChannel | QuantScheme::AsymmetricPerChannel => {
                let axis = self.config.group_size.map_or(0, |_| 0);
                per_channel_quantize(data, shape, axis, &self.config).map(|mut t| {
                    t.scheme = self.scheme;
                    t
                })
            }
            _ => {
                let params = calibrate(data, &self.config);
                let packed = quantize_to_bytes(data, &params, self.config.symmetric);
                let size = packed.len() as u64;
                Ok(QuantizedTensor {
                    data: packed,
                    params,
                    original_shape: shape.to_vec(),
                    scheme: self.scheme,
                    size_bytes: size,
                })
            }
        }
    }
}

/// Errors that can occur during quantization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantizeError {
    /// Shape does not match data length.
    ShapeMismatch { expected: usize, actual: usize },
}

impl fmt::Display for QuantizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch { expected, actual } => {
                write!(f, "shape product {expected} != data length {actual}")
            }
        }
    }
}

impl std::error::Error for QuantizeError {}

// ── Internal quantize / dequantize helpers ────────────────────────────────

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
fn quantize_to_bytes(data: &[f32], params: &QuantParams, symmetric: bool) -> Vec<u8> {
    if symmetric {
        let half = 1_i64 << (params.bits - 1);
        let qmin = -half;
        let qmax = half - 1;
        if params.bits <= 8 {
            data.iter()
                .map(|&v| {
                    let q =
                        (f64::from(v) / params.scale).round().clamp(qmin as f64, qmax as f64) as i8;
                    q as u8
                })
                .collect()
        } else {
            data.iter()
                .flat_map(|&v| {
                    let q = (f64::from(v) / params.scale).round().clamp(qmin as f64, qmax as f64)
                        as i16;
                    (q as u16).to_le_bytes()
                })
                .collect()
        }
    } else {
        let qmax = (1_i64 << params.bits) - 1;
        if params.bits <= 8 {
            data.iter()
                .map(|&v| {
                    (f64::from(v) / params.scale + params.zero_point as f64)
                        .round()
                        .clamp(0.0, qmax as f64) as u8
                })
                .collect()
        } else {
            data.iter()
                .flat_map(|&v| {
                    let q = (f64::from(v) / params.scale + params.zero_point as f64)
                        .round()
                        .clamp(0.0, qmax as f64) as u16;
                    q.to_le_bytes()
                })
                .collect()
        }
    }
}

// ── Dequantizer ───────────────────────────────────────────────────────────

/// Dequantizes a [`QuantizedTensor`] back to `f32` values.
pub struct Dequantizer;

impl Dequantizer {
    /// Convert a quantized tensor back to `f32` values.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    pub fn dequantize(tensor: &QuantizedTensor) -> Vec<f32> {
        if tensor.scheme == QuantScheme::BitNet158 {
            let ternary = unpack_ternary(&tensor.data, element_count(tensor));
            return ternary.iter().map(|&v| f32::from(v) * tensor.params.scale as f32).collect();
        }

        let params = &tensor.params;
        let symmetric = matches!(
            tensor.scheme,
            QuantScheme::SymmetricPerTensor | QuantScheme::SymmetricPerChannel
        );

        if symmetric {
            // Signed storage: interpret u8/u16 as i8/i16.
            if params.bits <= 8 {
                tensor
                    .data
                    .iter()
                    .map(|&q| {
                        let signed = q.cast_signed();
                        (f64::from(signed) * params.scale) as f32
                    })
                    .collect()
            } else {
                tensor
                    .data
                    .chunks_exact(2)
                    .map(|c| {
                        let q = i16::from_le_bytes([c[0], c[1]]);
                        (f64::from(q) * params.scale) as f32
                    })
                    .collect()
            }
        } else if params.bits <= 8 {
            tensor
                .data
                .iter()
                .map(|&q| ((f64::from(q) - params.zero_point as f64) * params.scale) as f32)
                .collect()
        } else {
            tensor
                .data
                .chunks_exact(2)
                .map(|c| {
                    let q = u16::from_le_bytes([c[0], c[1]]);
                    ((f64::from(q) - params.zero_point as f64) * params.scale) as f32
                })
                .collect()
        }
    }
}

fn element_count(tensor: &QuantizedTensor) -> usize {
    tensor.original_shape.iter().product()
}

// ── Ternary packing (BitNet 1.58) ────────────────────────────────────────

fn ternary_quantize_values(values: &[f32]) -> Vec<i8> {
    let max_abs = values.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);
    if max_abs == 0.0 {
        return vec![0i8; values.len()];
    }
    let threshold = max_abs * 0.5;
    values
        .iter()
        .map(|&v| {
            if v > threshold {
                1
            } else if v < -threshold {
                -1
            } else {
                0
            }
        })
        .collect()
}

fn ternary_scale(values: &[f32]) -> f64 {
    let max_abs = values.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);
    f64::from(max_abs)
}

/// Pack ternary values {-1, 0, +1} into 2 bits each.
///
/// Encoding: -1 → 0b10, 0 → 0b00, +1 → 0b01.
/// Four values per byte, MSB-first.
pub fn pack_ternary(values: &[i8]) -> Vec<u8> {
    let num_bytes = values.len().div_ceil(4);
    let mut packed = vec![0u8; num_bytes];
    for (i, &v) in values.iter().enumerate() {
        let code: u8 = match v {
            -1 => 0b10,
            1 => 0b01,
            _ => 0b00,
        };
        let byte_idx = i / 4;
        let shift = 6 - (i % 4) * 2;
        packed[byte_idx] |= code << shift;
    }
    packed
}

/// Unpack 2-bit ternary codes back to `{-1, 0, +1}` values.
pub fn unpack_ternary(packed: &[u8], count: usize) -> Vec<i8> {
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let byte_idx = i / 4;
        let shift = 6 - (i % 4) * 2;
        let code = (packed[byte_idx] >> shift) & 0b11;
        let val = match code {
            0b01 => 1,
            0b10 => -1,
            _ => 0,
        };
        out.push(val);
    }
    out
}

// ── Int4 packing ──────────────────────────────────────────────────────────

/// Pack signed 4-bit values (range -8..=7) into bytes, two per byte.
///
/// High nibble holds even-indexed values, low nibble holds odd-indexed.
pub fn pack_int4(values: &[i8]) -> Vec<u8> {
    let num_bytes = values.len().div_ceil(2);
    let mut packed = vec![0u8; num_bytes];
    for (i, &v) in values.iter().enumerate() {
        let nibble = v.cast_unsigned() & 0x0F;
        let byte_idx = i / 2;
        if i % 2 == 0 {
            packed[byte_idx] |= nibble << 4;
        } else {
            packed[byte_idx] |= nibble;
        }
    }
    packed
}

/// Unpack 4-bit signed values from packed bytes.
pub fn unpack_int4(packed: &[u8], count: usize) -> Vec<i8> {
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let byte_idx = i / 2;
        let nibble =
            if i % 2 == 0 { (packed[byte_idx] >> 4) & 0x0F } else { packed[byte_idx] & 0x0F };
        let val = if nibble & 0x08 != 0 {
            nibble.cast_signed() | (!0x0F_u8).cast_signed()
        } else {
            nibble.cast_signed()
        };
        out.push(val);
    }
    out
}

// ── Per-channel quantization ──────────────────────────────────────────────

/// Quantize `data` per-channel along the given `axis`.
///
/// `shape` must be 2-D (`[rows, cols]`). Axis 0 means one set of params
/// per row; axis 1 means one per column.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
pub fn per_channel_quantize(
    data: &[f32],
    shape: &[usize],
    axis: usize,
    config: &QuantConfig,
) -> Result<QuantizedTensor, QuantizeError> {
    if shape.len() != 2 {
        return Err(QuantizeError::ShapeMismatch { expected: 2, actual: shape.len() });
    }
    let (rows, cols) = (shape[0], shape[1]);
    if rows * cols != data.len() {
        return Err(QuantizeError::ShapeMismatch { expected: rows * cols, actual: data.len() });
    }

    let mut all_bytes: Vec<u8> = Vec::with_capacity(data.len());
    // We store the *first* channel's params as the tensor-level params;
    // a production implementation would store per-channel params.
    let mut first_params: Option<QuantParams> = None;

    if axis == 0 {
        for r in 0..rows {
            let slice = &data[r * cols..(r + 1) * cols];
            let p = calibrate(slice, config);
            let bytes = quantize_to_bytes(slice, &p, config.symmetric);
            all_bytes.extend_from_slice(&bytes);
            if first_params.is_none() {
                first_params = Some(p);
            }
        }
    } else {
        // axis == 1: per-column
        let mut columns: Vec<Vec<f32>> = vec![vec![]; cols];
        for r in 0..rows {
            for c in 0..cols {
                columns[c].push(data[r * cols + c]);
            }
        }
        // Quantize each column, then interleave back into row-major order.
        let mut col_params: Vec<QuantParams> = Vec::with_capacity(cols);
        let mut col_quant: Vec<Vec<u8>> = Vec::with_capacity(cols);
        for col in &columns {
            let p = calibrate(col, config);
            let bytes = quantize_to_bytes(col, &p, config.symmetric);
            col_quant.push(bytes);
            col_params.push(p);
        }
        if let Some(p) = col_params.first() {
            first_params = Some(p.clone());
        }
        for r in 0..rows {
            for col_data in &col_quant {
                all_bytes.push(col_data[r]);
            }
        }
    }

    let params = first_params.unwrap_or(QuantParams {
        scale: 1.0,
        zero_point: 0,
        bits: config.bits,
        group_size: config.group_size,
        channel_axis: Some(axis),
    });
    let size = all_bytes.len() as u64;

    Ok(QuantizedTensor {
        data: all_bytes,
        params: QuantParams { channel_axis: Some(axis), ..params },
        original_shape: shape.to_vec(),
        scheme: if config.symmetric {
            QuantScheme::SymmetricPerChannel
        } else {
            QuantScheme::AsymmetricPerChannel
        },
        size_bytes: size,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::single_char_pattern,
    clippy::cast_possible_truncation
)]
mod tests {
    use super::*;

    // Helper: symmetric per-tensor quantizer.
    fn sym_quantizer(bits: u32) -> Quantizer {
        Quantizer::new(
            QuantScheme::SymmetricPerTensor,
            QuantConfig { bits, symmetric: true, ..Default::default() },
        )
    }

    // Helper: asymmetric per-tensor quantizer.
    fn asym_quantizer(bits: u32) -> Quantizer {
        Quantizer::new(
            QuantScheme::AsymmetricPerTensor,
            QuantConfig { bits, symmetric: false, ..Default::default() },
        )
    }

    // ── QuantScheme Display ───────────────────────────────────────────

    #[test]
    fn scheme_display() {
        assert_eq!(QuantScheme::BitNet158.to_string(), "BitNet158");
        assert_eq!(QuantScheme::Gptq.to_string(), "GPTQ");
        assert_eq!(QuantScheme::Awq.to_string(), "AWQ");
    }

    // ── Symmetric per-tensor ──────────────────────────────────────────

    #[test]
    fn symmetric_per_tensor_scale() {
        let data = vec![-1.0_f32, 0.0, 0.5, 1.0];
        let q = sym_quantizer(8);
        let t = q.quantize(&data, &[4]).unwrap();
        assert!(t.params.scale > 0.0, "scale must be positive");
    }

    #[test]
    fn symmetric_per_tensor_zero_point_is_zero() {
        let data = vec![-2.0_f32, -1.0, 0.0, 1.0, 2.0];
        let q = sym_quantizer(8);
        let t = q.quantize(&data, &[5]).unwrap();
        assert_eq!(t.params.zero_point, 0);
    }

    #[test]
    fn symmetric_per_tensor_scheme_tag() {
        let q = sym_quantizer(8);
        let t = q.quantize(&[1.0], &[1]).unwrap();
        assert_eq!(t.scheme, QuantScheme::SymmetricPerTensor);
    }

    #[test]
    fn symmetric_per_tensor_original_shape_preserved() {
        let data = vec![0.1; 12];
        let q = sym_quantizer(8);
        let t = q.quantize(&data, &[3, 4]).unwrap();
        assert_eq!(t.original_shape, vec![3, 4]);
    }

    // ── Asymmetric per-tensor ─────────────────────────────────────────

    #[test]
    fn asymmetric_per_tensor_nonzero_zero_point() {
        let data = vec![2.0_f32, 3.0, 4.0, 5.0];
        let q = asym_quantizer(8);
        let t = q.quantize(&data, &[4]).unwrap();
        assert_ne!(t.params.zero_point, 0, "asymmetric should have zp != 0");
    }

    #[test]
    fn asymmetric_per_tensor_scale_positive() {
        let data = vec![0.0_f32, 10.0];
        let q = asym_quantizer(8);
        let t = q.quantize(&data, &[2]).unwrap();
        assert!(t.params.scale > 0.0);
    }

    #[test]
    fn asymmetric_scheme_tag() {
        let q = asym_quantizer(8);
        let t = q.quantize(&[1.0], &[1]).unwrap();
        assert_eq!(t.scheme, QuantScheme::AsymmetricPerTensor);
    }

    // ── Per-channel ───────────────────────────────────────────────────

    #[test]
    fn per_channel_axis0_produces_channel_params() {
        let data = vec![
            -1.0, 0.0, 1.0, // row 0
            -10.0, 0.0, 10.0, // row 1
        ];
        let cfg = QuantConfig { bits: 8, symmetric: true, ..Default::default() };
        let t = per_channel_quantize(&data, &[2, 3], 0, &cfg).unwrap();
        assert_eq!(t.params.channel_axis, Some(0));
    }

    #[test]
    fn per_channel_axis1_different_scales() {
        let data = vec![
            1.0, 100.0, // row 0
            2.0, 200.0, // row 1
        ];
        let cfg = QuantConfig { bits: 8, symmetric: true, ..Default::default() };
        let t = per_channel_quantize(&data, &[2, 2], 1, &cfg).unwrap();
        assert_eq!(t.params.channel_axis, Some(1));
    }

    #[test]
    fn per_channel_shape_mismatch_error() {
        let cfg = QuantConfig::default();
        let err = per_channel_quantize(&[1.0, 2.0], &[3, 3], 0, &cfg);
        assert!(err.is_err());
    }

    #[test]
    fn per_channel_non_2d_error() {
        let cfg = QuantConfig::default();
        let err = per_channel_quantize(&[1.0], &[1], 0, &cfg);
        assert!(err.is_err());
    }

    #[test]
    fn per_channel_via_quantizer_sym() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let q = Quantizer::new(
            QuantScheme::SymmetricPerChannel,
            QuantConfig { bits: 8, symmetric: true, ..Default::default() },
        );
        let t = q.quantize(&data, &[2, 2]).unwrap();
        assert_eq!(t.scheme, QuantScheme::SymmetricPerChannel);
    }

    #[test]
    fn per_channel_via_quantizer_asym() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let q = Quantizer::new(
            QuantScheme::AsymmetricPerChannel,
            QuantConfig { bits: 8, symmetric: false, ..Default::default() },
        );
        let t = q.quantize(&data, &[2, 2]).unwrap();
        assert_eq!(t.scheme, QuantScheme::AsymmetricPerChannel);
    }

    // ── Round-trip: quantize → dequantize ─────────────────────────────

    #[test]
    fn roundtrip_symmetric_8bit_within_tolerance() {
        let data: Vec<f32> = (-50..=50).map(|x| x as f32 * 0.1).collect();
        let q = sym_quantizer(8);
        let t = q.quantize(&data, &[data.len()]).unwrap();
        let deq = Dequantizer::dequantize(&t);
        assert_eq!(deq.len(), data.len());
        let max_err: f32 =
            data.iter().zip(deq.iter()).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        // 8-bit symmetric over range [-5, 5]: step ≈ 0.04
        assert!(max_err < 0.1, "max_err = {max_err}");
    }

    #[test]
    fn roundtrip_asymmetric_8bit_within_tolerance() {
        let data: Vec<f32> = (0..100).map(|x| x as f32 * 0.1).collect();
        let q = asym_quantizer(8);
        let t = q.quantize(&data, &[data.len()]).unwrap();
        let deq = Dequantizer::dequantize(&t);
        let max_err: f32 =
            data.iter().zip(deq.iter()).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        assert!(max_err < 0.1, "max_err = {max_err}");
    }

    #[test]
    fn roundtrip_preserves_element_count() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let q = sym_quantizer(8);
        let t = q.quantize(&data, &[5]).unwrap();
        let deq = Dequantizer::dequantize(&t);
        assert_eq!(deq.len(), 5);
    }

    #[test]
    fn roundtrip_16bit_higher_fidelity() {
        let data: Vec<f32> = (-100..=100).map(|x| x as f32 * 0.01).collect();
        let q = Quantizer::new(
            QuantScheme::SymmetricPerTensor,
            QuantConfig { bits: 16, symmetric: true, ..Default::default() },
        );
        let t = q.quantize(&data, &[data.len()]).unwrap();
        let deq = Dequantizer::dequantize(&t);
        let max_err: f32 =
            data.iter().zip(deq.iter()).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        assert!(max_err < 0.001, "16-bit max_err = {max_err}");
    }

    // ── BitNet ternary packing ────────────────────────────────────────

    #[test]
    fn ternary_pack_basic() {
        let vals = vec![1_i8, 0, -1, 1];
        let packed = pack_ternary(&vals);
        assert_eq!(packed.len(), 1);
        // 01 00 10 01 = 0x49
        assert_eq!(packed[0], 0b01_00_10_01);
    }

    #[test]
    fn ternary_pack_empty() {
        let packed = pack_ternary(&[]);
        assert!(packed.is_empty());
    }

    #[test]
    fn ternary_pack_non_aligned() {
        let vals = vec![1_i8, -1, 0];
        let packed = pack_ternary(&vals);
        assert_eq!(packed.len(), 1);
    }

    #[test]
    fn ternary_pack_all_zeros() {
        let vals = vec![0_i8; 8];
        let packed = pack_ternary(&vals);
        assert_eq!(packed, vec![0u8, 0u8]);
    }

    #[test]
    fn ternary_pack_all_ones() {
        let vals = vec![1_i8; 4];
        let packed = pack_ternary(&vals);
        assert_eq!(packed[0], 0b01_01_01_01);
    }

    #[test]
    fn ternary_pack_all_neg_ones() {
        let vals = vec![-1_i8; 4];
        let packed = pack_ternary(&vals);
        assert_eq!(packed[0], 0b10_10_10_10);
    }

    // ── Ternary unpacking ─────────────────────────────────────────────

    #[test]
    fn ternary_unpack_basic() {
        let packed = vec![0b01_00_10_01_u8];
        let vals = unpack_ternary(&packed, 4);
        assert_eq!(vals, vec![1, 0, -1, 1]);
    }

    #[test]
    fn ternary_unpack_partial_byte() {
        let packed = pack_ternary(&[1, -1, 0]);
        let vals = unpack_ternary(&packed, 3);
        assert_eq!(vals, vec![1, -1, 0]);
    }

    #[test]
    fn ternary_roundtrip_large() {
        let original: Vec<i8> = (0..100)
            .map(|i| match i % 3 {
                0 => -1,
                1 => 0,
                _ => 1,
            })
            .collect();
        let packed = pack_ternary(&original);
        let unpacked = unpack_ternary(&packed, original.len());
        assert_eq!(unpacked, original);
    }

    #[test]
    fn ternary_roundtrip_single() {
        let original = vec![1_i8];
        let packed = pack_ternary(&original);
        let unpacked = unpack_ternary(&packed, 1);
        assert_eq!(unpacked, original);
    }

    #[test]
    fn ternary_bitnet_quantize_roundtrip() {
        let data = vec![1.0_f32, 0.0, -1.0, 0.5, -0.5];
        let q = Quantizer::new(QuantScheme::BitNet158, QuantConfig::default());
        let t = q.quantize(&data, &[5]).unwrap();
        assert_eq!(t.scheme, QuantScheme::BitNet158);
        let deq = Dequantizer::dequantize(&t);
        assert_eq!(deq.len(), 5);
    }

    #[test]
    fn ternary_bitnet_all_zeros() {
        let data = vec![0.0_f32; 8];
        let q = Quantizer::new(QuantScheme::BitNet158, QuantConfig::default());
        let t = q.quantize(&data, &[8]).unwrap();
        let deq = Dequantizer::dequantize(&t);
        assert!(deq.iter().all(|&v| v == 0.0));
    }

    // ── Int4 packing ──────────────────────────────────────────────────

    #[test]
    fn int4_pack_two_values() {
        let vals = vec![3_i8, -2];
        let packed = pack_int4(&vals);
        assert_eq!(packed.len(), 1);
    }

    #[test]
    fn int4_pack_odd_count() {
        let vals = vec![1_i8, 2, 3];
        let packed = pack_int4(&vals);
        assert_eq!(packed.len(), 2);
    }

    #[test]
    fn int4_roundtrip() {
        let original = vec![0_i8, 1, -1, 7, -8, 3, -3, 0];
        let packed = pack_int4(&original);
        let unpacked = unpack_int4(&packed, original.len());
        assert_eq!(unpacked, original);
    }

    #[test]
    fn int4_roundtrip_single() {
        let original = vec![5_i8];
        let packed = pack_int4(&original);
        let unpacked = unpack_int4(&packed, 1);
        assert_eq!(unpacked, original);
    }

    #[test]
    fn int4_roundtrip_boundary_values() {
        // 4-bit signed range: -8 to 7
        let original = vec![-8_i8, -7, -1, 0, 1, 6, 7];
        let packed = pack_int4(&original);
        let unpacked = unpack_int4(&packed, original.len());
        assert_eq!(unpacked, original);
    }

    #[test]
    fn int4_pack_empty() {
        let packed = pack_int4(&[]);
        assert!(packed.is_empty());
        let unpacked = unpack_int4(&packed, 0);
        assert!(unpacked.is_empty());
    }

    // ── QuantStats ────────────────────────────────────────────────────

    #[test]
    fn stats_compression_ratio() {
        let original = vec![1.0_f32; 100];
        let q = sym_quantizer(8);
        let t = q.quantize(&original, &[100]).unwrap();
        let deq = Dequantizer::dequantize(&t);
        let stats = QuantStats::compute(&original, &deq, t.size_bytes);
        // 100 * 4 bytes = 400 original, 100 bytes quantized → ratio ≈ 4.0
        assert!(
            (stats.compression_ratio - 4.0).abs() < 0.01,
            "ratio = {}",
            stats.compression_ratio
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
    fn stats_snr_positive_for_nonzero_signal() {
        let original: Vec<f32> = (1..=50).map(|x| x as f32 * 0.1).collect();
        let q = sym_quantizer(8);
        let t = q.quantize(&original, &[original.len()]).unwrap();
        let deq = Dequantizer::dequantize(&t);
        let stats = QuantStats::compute(&original, &deq, t.size_bytes);
        assert!(stats.snr_db > 0.0, "snr = {}", stats.snr_db);
    }

    #[test]
    fn stats_perfect_reconstruction_infinite_snr() {
        let original = vec![1.0_f32, 2.0, 3.0];
        let stats = QuantStats::compute(&original, &original, 12);
        assert!(
            stats.snr_db.is_infinite() && stats.snr_db > 0.0,
            "perfect reconstruction should yield +inf SNR"
        );
    }

    #[test]
    fn stats_all_zeros_snr() {
        let zeros = vec![0.0_f32; 10];
        let stats = QuantStats::compute(&zeros, &zeros, 10);
        // Both signal and noise are zero → SNR = 0.0 by convention.
        assert!((stats.snr_db - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn stats_max_error() {
        let a = vec![0.0_f32, 1.0, 2.0];
        let b = vec![0.1_f32, 1.0, 2.5];
        let stats = QuantStats::compute(&a, &b, 3);
        assert!(
            (stats.max_quantization_error - 0.5).abs() < 1e-6,
            "max_err = {}",
            stats.max_quantization_error
        );
    }

    #[test]
    fn stats_mean_error() {
        let a = vec![0.0_f32, 0.0, 0.0];
        let b = vec![0.3_f32, 0.3, 0.3];
        let stats = QuantStats::compute(&a, &b, 3);
        assert!(
            (stats.mean_quantization_error - 0.3).abs() < 1e-4,
            "mean_err = {}",
            stats.mean_quantization_error
        );
    }

    #[test]
    fn stats_original_size_correct() {
        let data = vec![0.0_f32; 25];
        let stats = QuantStats::compute(&data, &data, 25);
        assert_eq!(stats.original_size, 100); // 25 * 4
    }

    // ── Calibration ───────────────────────────────────────────────────

    #[test]
    fn calibrate_minmax_symmetric() {
        let data = vec![-3.0_f32, 0.0, 3.0];
        let cfg = QuantConfig {
            bits: 8,
            symmetric: true,
            calibration_method: CalibrationMethod::MinMax,
            ..Default::default()
        };
        let p = calibrate(&data, &cfg);
        assert_eq!(p.zero_point, 0);
        assert!(p.scale > 0.0);
    }

    #[test]
    fn calibrate_minmax_asymmetric() {
        let data = vec![2.0_f32, 4.0, 6.0];
        let cfg = QuantConfig {
            bits: 8,
            symmetric: false,
            calibration_method: CalibrationMethod::MinMax,
            ..Default::default()
        };
        let p = calibrate(&data, &cfg);
        assert!(p.scale > 0.0);
    }

    #[test]
    fn calibrate_percentile() {
        let data: Vec<f32> = (0..1000).map(|x| x as f32 * 0.01).collect();
        let cfg = QuantConfig {
            bits: 8,
            symmetric: true,
            calibration_method: CalibrationMethod::Percentile(99.0),
            ..Default::default()
        };
        let p = calibrate(&data, &cfg);
        assert!(p.scale > 0.0);
        assert_eq!(p.zero_point, 0);
    }

    #[test]
    fn calibrate_entropy() {
        let data = vec![-1.0_f32, 0.0, 1.0];
        let cfg = QuantConfig {
            bits: 8,
            symmetric: true,
            calibration_method: CalibrationMethod::Entropy,
            ..Default::default()
        };
        let p = calibrate(&data, &cfg);
        assert!(p.scale > 0.0);
    }

    #[test]
    fn calibrate_mse() {
        let data = vec![-5.0_f32, 0.0, 5.0];
        let cfg = QuantConfig {
            bits: 8,
            symmetric: true,
            calibration_method: CalibrationMethod::Mse,
            ..Default::default()
        };
        let p = calibrate(&data, &cfg);
        assert!(p.scale > 0.0);
    }

    #[test]
    fn calibrate_empty_data() {
        let cfg = QuantConfig::default();
        let p = calibrate(&[], &cfg);
        assert_eq!(p.scale, 1.0);
        assert_eq!(p.zero_point, 0);
    }

    #[test]
    fn calibrate_percentile_clips_outliers() {
        // 1000 normal values plus one large outlier.
        let mut data: Vec<f32> = (0..1000).map(|x| x as f32 * 0.001).collect();
        data.push(1000.0);
        let cfg_mm = QuantConfig {
            bits: 8,
            symmetric: false,
            calibration_method: CalibrationMethod::MinMax,
            ..Default::default()
        };
        let cfg_pct = QuantConfig {
            bits: 8,
            symmetric: false,
            calibration_method: CalibrationMethod::Percentile(99.0),
            ..Default::default()
        };
        let p_mm = calibrate(&data, &cfg_mm);
        let p_pct = calibrate(&data, &cfg_pct);
        // Percentile should produce a smaller scale because the
        // outlier is clipped.
        assert!(
            p_pct.scale < p_mm.scale,
            "percentile scale {} >= minmax scale {}",
            p_pct.scale,
            p_mm.scale
        );
    }

    // ── Edge cases ────────────────────────────────────────────────────

    #[test]
    fn quantize_all_zeros() {
        let data = vec![0.0_f32; 10];
        let q = sym_quantizer(8);
        let t = q.quantize(&data, &[10]).unwrap();
        let deq = Dequantizer::dequantize(&t);
        assert!(deq.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn quantize_large_range() {
        let data = vec![-1e6_f32, 0.0, 1e6];
        let q = sym_quantizer(8);
        let t = q.quantize(&data, &[3]).unwrap();
        let deq = Dequantizer::dequantize(&t);
        // With 8-bit quant over a 2M range the step is large, but
        // extremes should be roughly preserved.
        assert!(deq[0] < 0.0);
        assert!(deq[2] > 0.0);
    }

    #[test]
    fn quantize_negative_only() {
        let data = vec![-5.0_f32, -3.0, -1.0];
        let q = sym_quantizer(8);
        let t = q.quantize(&data, &[3]).unwrap();
        let deq = Dequantizer::dequantize(&t);
        // All dequantized values should be negative or near-zero.
        for &v in &deq {
            assert!(v <= 0.1, "expected <= 0 but got {v}");
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
    fn quantize_single_element() {
        let q = sym_quantizer(8);
        let t = q.quantize(&[42.0], &[1]).unwrap();
        let deq = Dequantizer::dequantize(&t);
        assert_eq!(deq.len(), 1);
    }

    #[test]
    fn quantize_shape_mismatch_error() {
        let q = sym_quantizer(8);
        let err = q.quantize(&[1.0, 2.0], &[3]);
        assert!(err.is_err());
    }

    #[test]
    fn quantize_error_display() {
        let e = QuantizeError::ShapeMismatch { expected: 4, actual: 3 };
        let msg = e.to_string();
        assert!(msg.contains('4'));
        assert!(msg.contains('3'));
    }

    // ── Group quantization ────────────────────────────────────────────

    #[test]
    fn group_quantize_gptq() {
        let data: Vec<f32> = (0..256).map(|x| x as f32).collect();
        let q = Quantizer::new(
            QuantScheme::Gptq,
            QuantConfig {
                bits: 4,
                symmetric: true,
                group_size: Some(128),
                calibration_method: CalibrationMethod::MinMax,
            },
        );
        let t = q.quantize(&data, &[256]).unwrap();
        assert_eq!(t.scheme, QuantScheme::Gptq);
        assert_eq!(t.params.group_size, Some(128));
    }

    #[test]
    fn group_quantize_awq() {
        let data: Vec<f32> = (0..128).map(|x| x as f32 * 0.1).collect();
        let q = Quantizer::new(
            QuantScheme::Awq,
            QuantConfig {
                bits: 4,
                symmetric: true,
                group_size: Some(128),
                calibration_method: CalibrationMethod::MinMax,
            },
        );
        let t = q.quantize(&data, &[128]).unwrap();
        assert_eq!(t.scheme, QuantScheme::Awq);
    }

    // ── Additional coverage ───────────────────────────────────────────

    #[test]
    fn ternary_scale_computed_from_max_abs() {
        let data = vec![-3.0_f32, 0.0, 2.0];
        let q = Quantizer::new(QuantScheme::BitNet158, QuantConfig::default());
        let t = q.quantize(&data, &[3]).unwrap();
        assert!((t.params.scale - 3.0).abs() < 1e-6);
    }

    #[test]
    fn ternary_dequantize_scale() {
        let data = vec![1.0_f32, 0.0, -1.0, 0.8, -0.8];
        let q = Quantizer::new(QuantScheme::BitNet158, QuantConfig::default());
        let t = q.quantize(&data, &[5]).unwrap();
        let deq = Dequantizer::dequantize(&t);
        // First element should be +scale, third should be -scale.
        assert!(deq[0] > 0.0);
        assert!(deq[2] < 0.0);
    }

    #[test]
    fn quantized_tensor_size_bytes_consistent() {
        let data = vec![1.0_f32; 64];
        let q = sym_quantizer(8);
        let t = q.quantize(&data, &[64]).unwrap();
        assert_eq!(t.size_bytes, t.data.len() as u64);
    }

    #[test]
    fn quant_config_default_values() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.bits, 8);
        assert!(cfg.symmetric);
        assert_eq!(cfg.group_size, None);
        assert_eq!(cfg.calibration_method, CalibrationMethod::MinMax);
    }

    #[test]
    fn dequantize_preserves_length_16bit() {
        let data: Vec<f32> = (0..20).map(|x| x as f32).collect();
        let q = Quantizer::new(
            QuantScheme::SymmetricPerTensor,
            QuantConfig { bits: 16, symmetric: true, ..Default::default() },
        );
        let t = q.quantize(&data, &[20]).unwrap();
        let deq = Dequantizer::dequantize(&t);
        assert_eq!(deq.len(), 20);
    }

    #[test]
    fn calibrate_bits_propagated() {
        let cfg = QuantConfig { bits: 4, symmetric: true, ..Default::default() };
        let p = calibrate(&[1.0, 2.0], &cfg);
        assert_eq!(p.bits, 4);
    }

    #[test]
    fn symmetric_4bit_roundtrip() {
        let data: Vec<f32> = (-7..=7).map(|x| x as f32).collect();
        let q = Quantizer::new(
            QuantScheme::SymmetricPerTensor,
            QuantConfig { bits: 4, symmetric: true, ..Default::default() },
        );
        let t = q.quantize(&data, &[data.len()]).unwrap();
        let deq = Dequantizer::dequantize(&t);
        assert_eq!(deq.len(), data.len());
    }

    #[test]
    fn multiple_pack_unpack_ternary_sizes() {
        for n in [1, 2, 3, 4, 5, 7, 8, 15, 16, 33] {
            let original: Vec<i8> = (0..n)
                .map(|i| match i % 3 {
                    0 => -1,
                    1 => 0,
                    _ => 1,
                })
                .collect();
            let packed = pack_ternary(&original);
            let unpacked = unpack_ternary(&packed, n);
            assert_eq!(unpacked, original, "ternary roundtrip failed for n={n}");
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
    fn multiple_pack_unpack_int4_sizes() {
        for n in [1, 2, 3, 4, 5, 7, 8, 15, 16, 33] {
            let original: Vec<i8> = (0..n).map(|i| (i8::try_from(i).unwrap() % 8) - 4).collect();
            let packed = pack_int4(&original);
            let unpacked = unpack_int4(&packed, n);
            assert_eq!(unpacked, original, "int4 roundtrip failed for n={n}");
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
    fn stats_quantized_size_stored() {
        let stats = QuantStats::compute(&[1.0], &[1.0], 42);
        assert_eq!(stats.quantized_size, 42);
    }
}

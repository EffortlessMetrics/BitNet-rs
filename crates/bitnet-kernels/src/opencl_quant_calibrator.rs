//! Quantization calibration system for post-training quantization.
//!
//! Collects activation statistics (min, max, histogram), computes optimal
//! scale/zero-point, simulates quantization error, and automatically selects
//! the best scheme per layer.  All operations have CPU reference
//! implementations — no OpenCL runtime required.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Quantization scheme
// ---------------------------------------------------------------------------

/// Granularity of quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantScheme {
    /// One scale/zero-point for the entire tensor.
    PerTensor,
    /// One scale/zero-point per output channel.
    PerChannel,
    /// One scale/zero-point per group of `group_size` elements.
    PerGroup(usize),
}

impl fmt::Display for QuantScheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PerTensor => write!(f, "PerTensor"),
            Self::PerChannel => write!(f, "PerChannel"),
            Self::PerGroup(g) => write!(f, "PerGroup({g})"),
        }
    }
}

// ---------------------------------------------------------------------------
// Range method
// ---------------------------------------------------------------------------

/// Method used to determine the quantization range from collected statistics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantRangeMethod {
    /// Use the observed min/max directly.
    MinMax,
    /// Clip to the given percentile (e.g., 99.99).
    Percentile(f64),
    /// Minimise mean-squared-error between original and quantized values.
    Mse,
    /// Minimise KL-divergence between the original and quantized distributions.
    Entropy,
}

impl fmt::Display for QuantRangeMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MinMax => write!(f, "MinMax"),
            Self::Percentile(p) => write!(f, "Percentile({p:.2})"),
            Self::Mse => write!(f, "MSE"),
            Self::Entropy => write!(f, "Entropy(KL)"),
        }
    }
}

// ---------------------------------------------------------------------------
// Calibration dataset
// ---------------------------------------------------------------------------

/// Collects activation statistics over a calibration dataset.
#[derive(Debug, Clone)]
pub struct CalibrationDataset {
    /// Running min per-sample.
    pub min: f64,
    /// Running max per-sample.
    pub max: f64,
    /// Sum of values (for mean).
    pub sum: f64,
    /// Sum of squared values (for variance).
    pub sum_sq: f64,
    /// Number of values observed.
    pub count: u64,
    /// Histogram bins.
    pub histogram: Vec<u64>,
    /// Histogram bin edges (len = histogram.len() + 1).
    pub bin_edges: Vec<f64>,
    /// Sorted sample reservoir for percentile computation.
    reservoir: Vec<f32>,
    /// Maximum reservoir size.
    reservoir_cap: usize,
    /// Deterministic counter for reservoir sampling.
    sample_index: u64,
}

impl CalibrationDataset {
    /// Create a new dataset with `n_bins` histogram bins and reservoir capacity.
    pub fn new(n_bins: usize, reservoir_cap: usize) -> Self {
        Self {
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            sum: 0.0,
            sum_sq: 0.0,
            count: 0,
            histogram: vec![0u64; n_bins],
            bin_edges: Vec::new(),
            reservoir: Vec::new(),
            reservoir_cap,
            sample_index: 0,
        }
    }

    /// Record a batch of activations.  On the first batch the histogram range
    /// is initialised; subsequent batches accumulate into the existing bins.
    pub fn record(&mut self, values: &[f32]) {
        if values.is_empty() {
            return;
        }

        for &v in values {
            let vf = v as f64;
            if vf < self.min {
                self.min = vf;
            }
            if vf > self.max {
                self.max = vf;
            }
            self.sum += vf;
            self.sum_sq += vf * vf;
            self.count += 1;

            // Reservoir sampling (deterministic modular replacement).
            if self.reservoir.len() < self.reservoir_cap {
                self.reservoir.push(v);
            } else if self.reservoir_cap > 0 {
                let idx = (self.sample_index as usize) % self.reservoir_cap;
                self.reservoir[idx] = v;
            }
            self.sample_index += 1;
        }

        // (Re-)build histogram if edges are not yet set.
        if self.bin_edges.is_empty() && self.count > 0 {
            self.rebuild_histogram();
        }

        // Accumulate into bins (including the first batch after rebuild).
        if !self.bin_edges.is_empty() {
            let n_bins = self.histogram.len();
            let lo = self.bin_edges[0];
            let hi = self.bin_edges[n_bins];
            let range = hi - lo;
            if range <= 0.0 {
                self.histogram[0] += values.len() as u64;
            } else {
                for &v in values {
                    let vf = v as f64;
                    let idx = ((vf - lo) / range * n_bins as f64).floor() as usize;
                    let idx = idx.min(n_bins - 1);
                    self.histogram[idx] += 1;
                }
            }
        }
    }

    /// Rebuild histogram from current min/max.
    fn rebuild_histogram(&mut self) {
        let n_bins = self.histogram.len();
        let lo = self.min;
        let hi = self.max;
        let range = if (hi - lo).abs() < f64::EPSILON { 1.0 } else { hi - lo };
        self.bin_edges = (0..=n_bins).map(|i| lo + range * (i as f64 / n_bins as f64)).collect();
        self.histogram.fill(0);
    }

    /// Mean of observed values.
    pub fn mean(&self) -> f64 {
        if self.count == 0 { 0.0 } else { self.sum / self.count as f64 }
    }

    /// Variance (population).
    pub fn variance(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let mean = self.mean();
        self.sum_sq / self.count as f64 - mean * mean
    }

    /// Percentile from the reservoir (0.0–100.0).
    pub fn percentile(&self, p: f64) -> f64 {
        if self.reservoir.is_empty() {
            return 0.0;
        }
        let mut sorted: Vec<f32> = self.reservoir.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((p / 100.0) * (sorted.len() - 1) as f64)
            .round()
            .clamp(0.0, (sorted.len() - 1) as f64) as usize;
        sorted[idx] as f64
    }
}

// ---------------------------------------------------------------------------
// Calibration observer
// ---------------------------------------------------------------------------

/// Hooks into layer outputs to collect per-layer statistics.
#[derive(Debug)]
pub struct CalibrationObserver {
    /// Per-layer calibration datasets, keyed by layer name.
    pub layers: HashMap<String, CalibrationDataset>,
    /// Number of histogram bins used for new layers.
    pub n_bins: usize,
    /// Reservoir capacity per layer.
    pub reservoir_cap: usize,
}

impl CalibrationObserver {
    pub fn new(n_bins: usize, reservoir_cap: usize) -> Self {
        Self { layers: HashMap::new(), n_bins, reservoir_cap }
    }

    /// Record activations for a named layer.
    pub fn observe(&mut self, layer_name: &str, values: &[f32]) {
        let dataset = self
            .layers
            .entry(layer_name.to_string())
            .or_insert_with(|| CalibrationDataset::new(self.n_bins, self.reservoir_cap));
        dataset.record(values);
    }

    /// Number of observed layers.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Retrieve statistics for a specific layer.
    pub fn get_layer_stats(&self, layer_name: &str) -> Option<&CalibrationDataset> {
        self.layers.get(layer_name)
    }
}

// ---------------------------------------------------------------------------
// Scale computation
// ---------------------------------------------------------------------------

/// Computed scale and zero-point for symmetric/asymmetric quantization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScaleZeroPoint {
    pub scale: f64,
    pub zero_point: i32,
}

/// Computes optimal scale and zero-point from collected statistics.
pub struct ScaleComputer;

impl ScaleComputer {
    /// Compute scale/zero-point for signed 8-bit symmetric quantization.
    pub fn compute_symmetric(min_val: f64, max_val: f64, bits: u32) -> ScaleZeroPoint {
        let qmax = (1i64 << (bits - 1)) - 1;
        let abs_max = min_val.abs().max(max_val.abs());
        let scale = if abs_max == 0.0 { 1.0 } else { abs_max / qmax as f64 };
        ScaleZeroPoint { scale, zero_point: 0 }
    }

    /// Compute scale/zero-point for unsigned 8-bit asymmetric quantization.
    pub fn compute_asymmetric(min_val: f64, max_val: f64, bits: u32) -> ScaleZeroPoint {
        let qmin = 0i64;
        let qmax = (1i64 << bits) - 1;
        // Ensure the range includes zero so zero_point stays representable.
        let min_val = min_val.min(0.0);
        let max_val = max_val.max(0.0);
        let range = max_val - min_val;
        let scale = if range == 0.0 { 1.0 } else { range / (qmax - qmin) as f64 };
        let zero_point = if range == 0.0 {
            0
        } else {
            ((-min_val / scale).round() as i64).clamp(qmin, qmax) as i32
        };
        ScaleZeroPoint { scale, zero_point }
    }

    /// Compute scale/zero-point using a specific range method and dataset.
    pub fn compute(
        dataset: &CalibrationDataset,
        method: QuantRangeMethod,
        bits: u32,
        symmetric: bool,
    ) -> ScaleZeroPoint {
        let (rmin, rmax) = Self::determine_range(dataset, method);
        if symmetric {
            Self::compute_symmetric(rmin, rmax, bits)
        } else {
            Self::compute_asymmetric(rmin, rmax, bits)
        }
    }

    /// Determine the effective min/max range from the dataset using the given method.
    pub fn determine_range(dataset: &CalibrationDataset, method: QuantRangeMethod) -> (f64, f64) {
        match method {
            QuantRangeMethod::MinMax => (dataset.min, dataset.max),
            QuantRangeMethod::Percentile(p) => {
                let lo = dataset.percentile(100.0 - p);
                let hi = dataset.percentile(p);
                (lo, hi)
            }
            QuantRangeMethod::Mse => {
                // Iterative search: try shrinking the range and pick the one
                // with lowest MSE when simulating quantisation.
                Self::mse_optimal_range(dataset)
            }
            QuantRangeMethod::Entropy => Self::entropy_optimal_range(dataset),
        }
    }

    /// MSE-optimal range: test 40 candidate thresholds, return the best.
    fn mse_optimal_range(dataset: &CalibrationDataset) -> (f64, f64) {
        if dataset.reservoir.is_empty() {
            return (dataset.min, dataset.max);
        }
        let full_max = dataset.min.abs().max(dataset.max.abs());
        let mut best_mse = f64::MAX;
        let mut best_threshold = full_max;
        let steps = 40;
        for i in 1..=steps {
            let threshold = full_max * (i as f64 / steps as f64);
            let mse = Self::simulate_mse(&dataset.reservoir, threshold, 8);
            if mse < best_mse {
                best_mse = mse;
                best_threshold = threshold;
            }
        }
        (-best_threshold, best_threshold)
    }

    /// Entropy (KL-divergence) optimal range via histogram-based search.
    fn entropy_optimal_range(dataset: &CalibrationDataset) -> (f64, f64) {
        if dataset.histogram.iter().all(|&c| c == 0) {
            return (dataset.min, dataset.max);
        }
        let n_bins = dataset.histogram.len();
        let total: u64 = dataset.histogram.iter().sum();
        if total == 0 {
            return (dataset.min, dataset.max);
        }

        // Reference distribution P (normalised histogram).
        let p_dist: Vec<f64> = dataset.histogram.iter().map(|&c| c as f64 / total as f64).collect();

        let mut best_kl = f64::MAX;
        let mut best_end = n_bins;
        let start_bin = n_bins / 2; // search from midpoint outward

        for end in start_bin..=n_bins {
            // Build a candidate quantised distribution Q with `end` bins.
            let q_dist = Self::build_quantised_distribution(&p_dist, end, 256);
            let kl = Self::kl_divergence(&p_dist[..end], &q_dist);
            if kl < best_kl {
                best_kl = kl;
                best_end = end;
            }
        }

        // Map best_end back to a value range.
        let lo = dataset.bin_edges.first().copied().unwrap_or(dataset.min);
        let hi_edge = dataset.bin_edges.get(best_end).copied().unwrap_or(dataset.max);
        // Mirror for symmetric.
        let abs_max = lo.abs().max(hi_edge.abs());
        (-abs_max, abs_max)
    }

    /// Build a quantised histogram by merging `src_bins` into `n_quant_levels`.
    fn build_quantised_distribution(p: &[f64], src_bins: usize, n_quant_levels: usize) -> Vec<f64> {
        if src_bins == 0 {
            return Vec::new();
        }
        let bins_per_level = (src_bins as f64 / n_quant_levels as f64).max(1.0);
        let mut q = vec![0.0f64; src_bins];
        for (i, qi) in q.iter_mut().enumerate() {
            let level = (i as f64 / bins_per_level).floor() as usize;
            let level = level.min(n_quant_levels - 1);
            let start = (level as f64 * bins_per_level).floor() as usize;
            let end = ((level + 1) as f64 * bins_per_level).ceil() as usize;
            let end = end.min(src_bins);
            let count = end - start;
            if count > 0 {
                // Uniform redistribution within the merged bin.
                let merged_sum: f64 = p[start..end].iter().sum();
                *qi = merged_sum / count as f64;
            }
        }
        // Normalise.
        let total: f64 = q.iter().sum();
        if total > 0.0 {
            for v in &mut q {
                *v /= total;
            }
        }
        q
    }

    /// KL divergence D(P || Q).  Bins where P=0 are skipped.
    fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
        let eps = 1e-12;
        let len = p.len().min(q.len());
        let mut kl = 0.0f64;
        for i in 0..len {
            if p[i] > eps {
                let qi = if q[i] > eps { q[i] } else { eps };
                kl += p[i] * (p[i] / qi).ln();
            }
        }
        kl
    }

    /// Simulate MSE for symmetric quantisation with a given threshold.
    fn simulate_mse(values: &[f32], threshold: f64, bits: u32) -> f64 {
        let qmax = ((1i64 << (bits - 1)) - 1) as f64;
        let scale = threshold / qmax;
        if scale == 0.0 {
            return 0.0;
        }
        let mut mse = 0.0f64;
        for &v in values {
            let vf = v as f64;
            let q = (vf / scale).round().clamp(-qmax, qmax);
            let deq = q * scale;
            let err = vf - deq;
            mse += err * err;
        }
        mse / values.len() as f64
    }
}

// ---------------------------------------------------------------------------
// Quantization simulator
// ---------------------------------------------------------------------------

/// Error metrics from a quantisation simulation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantErrorMetrics {
    /// Mean squared error.
    pub mse: f64,
    /// Maximum absolute error.
    pub max_error: f64,
    /// Signal-to-noise ratio in dB.
    pub snr_db: f64,
    /// Signal-to-quantisation-noise ratio in dB.
    pub sqnr_db: f64,
    /// Mean absolute error.
    pub mae: f64,
}

impl fmt::Display for QuantErrorMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MSE={:.6e}  MAE={:.6e}  max_err={:.6e}  SNR={:.2}dB  SQNR={:.2}dB",
            self.mse, self.mae, self.max_error, self.snr_db, self.sqnr_db,
        )
    }
}

/// Simulates quantization error for a given scale/zero-point.
pub struct QuantSimulator;

impl QuantSimulator {
    /// Quantize a single value (signed symmetric, 8-bit).
    #[inline]
    pub fn quantize_symmetric(value: f32, scale: f64, bits: u32) -> i32 {
        let qmax = ((1i64 << (bits - 1)) - 1) as f64;
        ((value as f64) / scale).round().clamp(-qmax, qmax) as i32
    }

    /// Dequantize a single value (signed symmetric).
    #[inline]
    pub fn dequantize_symmetric(qval: i32, scale: f64) -> f32 {
        (qval as f64 * scale) as f32
    }

    /// Quantize a single value (asymmetric).
    #[inline]
    pub fn quantize_asymmetric(value: f32, scale: f64, zero_point: i32, bits: u32) -> i32 {
        let qmax = ((1i64 << bits) - 1) as f64;
        ((value as f64) / scale + zero_point as f64).round().clamp(0.0, qmax) as i32
    }

    /// Dequantize a single value (asymmetric).
    #[inline]
    pub fn dequantize_asymmetric(qval: i32, scale: f64, zero_point: i32) -> f32 {
        ((qval - zero_point) as f64 * scale) as f32
    }

    /// Simulate quantisation round-trip and compute error metrics.
    pub fn simulate(
        values: &[f32],
        sz: ScaleZeroPoint,
        bits: u32,
        symmetric: bool,
    ) -> QuantErrorMetrics {
        if values.is_empty() {
            return QuantErrorMetrics {
                mse: 0.0,
                max_error: 0.0,
                snr_db: 0.0,
                sqnr_db: 0.0,
                mae: 0.0,
            };
        }
        let mut sum_sq_err = 0.0f64;
        let mut sum_abs_err = 0.0f64;
        let mut max_err = 0.0f64;
        let mut sum_sq_signal = 0.0f64;

        for &v in values {
            let deq = if symmetric {
                let q = Self::quantize_symmetric(v, sz.scale, bits);
                Self::dequantize_symmetric(q, sz.scale)
            } else {
                let q = Self::quantize_asymmetric(v, sz.scale, sz.zero_point, bits);
                Self::dequantize_asymmetric(q, sz.scale, sz.zero_point)
            };
            let err = (v - deq) as f64;
            sum_sq_err += err * err;
            sum_abs_err += err.abs();
            if err.abs() > max_err {
                max_err = err.abs();
            }
            sum_sq_signal += (v as f64) * (v as f64);
        }

        let n = values.len() as f64;
        let mse = sum_sq_err / n;
        let mae = sum_abs_err / n;
        let snr_db =
            if mse > 0.0 { 10.0 * (sum_sq_signal / n / mse).log10() } else { f64::INFINITY };
        let sqnr_db = snr_db; // SQNR = SNR for quantisation noise

        QuantErrorMetrics { mse, max_error: max_err, snr_db, sqnr_db, mae }
    }

    /// Per-channel simulation: values shape is (channels, elements_per_channel).
    pub fn simulate_per_channel(
        values: &[f32],
        channels: usize,
        method: QuantRangeMethod,
        bits: u32,
        symmetric: bool,
    ) -> Vec<QuantErrorMetrics> {
        if channels == 0 || values.is_empty() {
            return Vec::new();
        }
        let elems_per_ch = values.len() / channels;
        let mut results = Vec::with_capacity(channels);
        for ch in 0..channels {
            let start = ch * elems_per_ch;
            let end = start + elems_per_ch;
            let ch_data = &values[start..end];
            let mut ds = CalibrationDataset::new(64, ch_data.len());
            ds.record(ch_data);
            let sz = ScaleComputer::compute(&ds, method, bits, symmetric);
            results.push(Self::simulate(ch_data, sz, bits, symmetric));
        }
        results
    }
}

// ---------------------------------------------------------------------------
// Calibration report
// ---------------------------------------------------------------------------

/// Per-layer quantisation quality report.
#[derive(Debug, Clone)]
pub struct LayerReport {
    pub layer_name: String,
    pub scheme: QuantScheme,
    pub method: QuantRangeMethod,
    pub bits: u32,
    pub symmetric: bool,
    pub metrics: QuantErrorMetrics,
    pub scale_zero: ScaleZeroPoint,
    pub recommendation: String,
}

impl fmt::Display for LayerReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {} {}bit {}: {}  rec={}",
            self.layer_name, self.scheme, self.bits, self.method, self.metrics, self.recommendation,
        )
    }
}

/// Full calibration report across all observed layers.
#[derive(Debug)]
pub struct CalibrationReport {
    pub layers: Vec<LayerReport>,
}

impl CalibrationReport {
    /// Generate a report from an observer using the given configuration.
    pub fn generate(
        observer: &CalibrationObserver,
        scheme: QuantScheme,
        method: QuantRangeMethod,
        bits: u32,
        symmetric: bool,
    ) -> Self {
        let mut layers = Vec::new();
        let mut sorted_names: Vec<&String> = observer.layers.keys().collect();
        sorted_names.sort();

        for name in sorted_names {
            let dataset = &observer.layers[name];
            let sz = ScaleComputer::compute(dataset, method, bits, symmetric);
            let metrics = QuantSimulator::simulate(&dataset.reservoir, sz, bits, symmetric);
            let recommendation = Self::recommend(&metrics, bits);
            layers.push(LayerReport {
                layer_name: name.clone(),
                scheme,
                method,
                bits,
                symmetric,
                metrics,
                scale_zero: sz,
                recommendation,
            });
        }
        Self { layers }
    }

    fn recommend(metrics: &QuantErrorMetrics, bits: u32) -> String {
        if metrics.sqnr_db > 40.0 {
            "excellent".to_string()
        } else if metrics.sqnr_db > 25.0 {
            "good".to_string()
        } else if metrics.sqnr_db > 15.0 && bits <= 4 {
            "acceptable for low-bit".to_string()
        } else if metrics.sqnr_db > 10.0 {
            "marginal — consider higher bits or per-channel".to_string()
        } else {
            "poor — increase bits or use per-channel quantisation".to_string()
        }
    }

    /// Rank layers by quantisation sensitivity (worst first).
    pub fn sensitivity_ranking(&self) -> Vec<&LayerReport> {
        let mut ranked: Vec<&LayerReport> = self.layers.iter().collect();
        ranked.sort_by(|a, b| {
            a.metrics.sqnr_db.partial_cmp(&b.metrics.sqnr_db).unwrap_or(std::cmp::Ordering::Equal)
        });
        ranked
    }

    /// Layers whose SQNR falls below the given threshold.
    pub fn sensitive_layers(&self, sqnr_threshold_db: f64) -> Vec<&LayerReport> {
        self.layers.iter().filter(|l| l.metrics.sqnr_db < sqnr_threshold_db).collect()
    }
}

// ---------------------------------------------------------------------------
// Auto-quantizer
// ---------------------------------------------------------------------------

/// Automatically selects the best quantisation scheme per layer based on
/// sensitivity analysis.
pub struct AutoQuantizer {
    /// Target bits for "safe" layers.
    pub default_bits: u32,
    /// Bits to fall back to for sensitive layers.
    pub fallback_bits: u32,
    /// SQNR threshold (dB) below which a layer is "sensitive".
    pub sensitivity_threshold: f64,
    /// Whether to use symmetric quantisation.
    pub symmetric: bool,
}

impl Default for AutoQuantizer {
    fn default() -> Self {
        Self { default_bits: 8, fallback_bits: 16, sensitivity_threshold: 20.0, symmetric: true }
    }
}

/// Per-layer decision from the auto-quantizer.
#[derive(Debug, Clone)]
pub struct LayerQuantDecision {
    pub layer_name: String,
    pub scheme: QuantScheme,
    pub bits: u32,
    pub method: QuantRangeMethod,
    pub metrics: QuantErrorMetrics,
    pub sensitive: bool,
}

impl fmt::Display for LayerQuantDecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {}bit {} {} sensitive={}",
            self.layer_name, self.bits, self.scheme, self.method, self.sensitive,
        )
    }
}

impl AutoQuantizer {
    pub fn new(
        default_bits: u32,
        fallback_bits: u32,
        sensitivity_threshold: f64,
        symmetric: bool,
    ) -> Self {
        Self { default_bits, fallback_bits, sensitivity_threshold, symmetric }
    }

    /// Analyse the observer and produce per-layer decisions.
    pub fn analyse(&self, observer: &CalibrationObserver) -> Vec<LayerQuantDecision> {
        let methods = [
            QuantRangeMethod::MinMax,
            QuantRangeMethod::Percentile(99.99),
            QuantRangeMethod::Mse,
            QuantRangeMethod::Entropy,
        ];

        let mut decisions = Vec::new();
        let mut sorted_names: Vec<&String> = observer.layers.keys().collect();
        sorted_names.sort();

        for name in sorted_names {
            let dataset = &observer.layers[name];

            // Try each method at default bits, pick the one with best SQNR.
            let mut best_method = QuantRangeMethod::MinMax;
            let mut best_metrics: Option<QuantErrorMetrics> = None;

            for &method in &methods {
                let sz = ScaleComputer::compute(dataset, method, self.default_bits, self.symmetric);
                let m = QuantSimulator::simulate(
                    &dataset.reservoir,
                    sz,
                    self.default_bits,
                    self.symmetric,
                );
                if best_metrics.is_none() || m.sqnr_db > best_metrics.unwrap().sqnr_db {
                    best_metrics = Some(m);
                    best_method = method;
                }
            }

            let metrics = best_metrics.unwrap_or(QuantErrorMetrics {
                mse: 0.0,
                max_error: 0.0,
                snr_db: 0.0,
                sqnr_db: 0.0,
                mae: 0.0,
            });

            let sensitive = metrics.sqnr_db < self.sensitivity_threshold;
            let bits = if sensitive { self.fallback_bits } else { self.default_bits };

            // Try per-channel for sensitive layers.
            let scheme = if sensitive { QuantScheme::PerChannel } else { QuantScheme::PerTensor };

            decisions.push(LayerQuantDecision {
                layer_name: name.clone(),
                scheme,
                bits,
                method: best_method,
                metrics,
                sensitive,
            });
        }
        decisions
    }
}

// ---------------------------------------------------------------------------
// OpenCL kernel source (embedded, no runtime dependency)
// ---------------------------------------------------------------------------

/// OpenCL C kernel source for quantisation simulation on GPU.
pub const QUANT_CALIBRATOR_CL: &str = r#"
// Symmetric quantise → dequantise round-trip and per-element error.
__kernel void quant_simulate_symmetric(
    __global const float* input,
    __global float* output,      // dequantised values
    __global float* errors,      // per-element squared error
    const float scale,
    const float inv_scale,
    const float qmax,
    const int n)
{
    int gid = get_global_id(0);
    if (gid >= n) return;

    float v = input[gid];
    float q = round(v * inv_scale);
    q = clamp(q, -qmax, qmax);
    float deq = q * scale;
    output[gid] = deq;
    float err = v - deq;
    errors[gid] = err * err;
}

// Asymmetric quantise → dequantise round-trip.
__kernel void quant_simulate_asymmetric(
    __global const float* input,
    __global float* output,
    __global float* errors,
    const float scale,
    const float inv_scale,
    const int zero_point,
    const float qmax,
    const int n)
{
    int gid = get_global_id(0);
    if (gid >= n) return;

    float v = input[gid];
    float q = round(v * inv_scale + (float)zero_point);
    q = clamp(q, 0.0f, qmax);
    float deq = (q - (float)zero_point) * scale;
    output[gid] = deq;
    float err = v - deq;
    errors[gid] = err * err;
}

// Histogram bin counting (atomics).
__kernel void histogram_collect(
    __global const float* input,
    __global int* histogram,
    const float lo,
    const float inv_range,
    const int n_bins,
    const int n)
{
    int gid = get_global_id(0);
    if (gid >= n) return;

    float v = input[gid];
    int idx = (int)floor((v - lo) * inv_range * (float)n_bins);
    idx = clamp(idx, 0, n_bins - 1);
    atomic_add(&histogram[idx], 1);
}

// Min/max reduction (work-group level, first pass).
__kernel void minmax_reduce(
    __global const float* input,
    __global float* mins,
    __global float* maxs,
    __local float* local_min,
    __local float* local_max,
    const int n)
{
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int group_size = get_local_size(0);

    float val = (gid < n) ? input[gid] : 0.0f;
    local_min[lid] = (gid < n) ? val : INFINITY;
    local_max[lid] = (gid < n) ? val : -INFINITY;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            if (local_min[lid + stride] < local_min[lid])
                local_min[lid] = local_min[lid + stride];
            if (local_max[lid + stride] > local_max[lid])
                local_max[lid] = local_max[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        int group_id = get_group_id(0);
        mins[group_id] = local_min[0];
        maxs[group_id] = local_max[0];
    }
}
"#;

// ---------------------------------------------------------------------------
// CPU reference implementations that mirror the OpenCL kernels
// ---------------------------------------------------------------------------

/// CPU reference: symmetric quantise → dequantise, returns (output, errors).
pub fn cpu_quant_simulate_symmetric(input: &[f32], scale: f64, bits: u32) -> (Vec<f32>, Vec<f32>) {
    let qmax = ((1i64 << (bits - 1)) - 1) as f64;
    let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };
    let mut output = Vec::with_capacity(input.len());
    let mut errors = Vec::with_capacity(input.len());
    for &v in input {
        let vf = v as f64;
        let q = (vf * inv_scale).round().clamp(-qmax, qmax);
        let deq = q * scale;
        output.push(deq as f32);
        let err = vf - deq;
        errors.push((err * err) as f32);
    }
    (output, errors)
}

/// CPU reference: asymmetric quantise → dequantise, returns (output, errors).
pub fn cpu_quant_simulate_asymmetric(
    input: &[f32],
    scale: f64,
    zero_point: i32,
    bits: u32,
) -> (Vec<f32>, Vec<f32>) {
    let qmax = ((1i64 << bits) - 1) as f64;
    let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };
    let mut output = Vec::with_capacity(input.len());
    let mut errors = Vec::with_capacity(input.len());
    for &v in input {
        let vf = v as f64;
        let q = (vf * inv_scale + zero_point as f64).round().clamp(0.0, qmax);
        let deq = (q - zero_point as f64) * scale;
        output.push(deq as f32);
        let err = vf - deq;
        errors.push((err * err) as f32);
    }
    (output, errors)
}

/// CPU reference: histogram bin counting.
pub fn cpu_histogram_collect(input: &[f32], lo: f64, hi: f64, n_bins: usize) -> Vec<u64> {
    let range = hi - lo;
    let inv_range = if range == 0.0 { 0.0 } else { 1.0 / range };
    let mut histogram = vec![0u64; n_bins];
    for &v in input {
        let idx = ((v as f64 - lo) * inv_range * n_bins as f64).floor() as isize;
        let idx = idx.clamp(0, (n_bins - 1) as isize) as usize;
        histogram[idx] += 1;
    }
    histogram
}

/// CPU reference: min/max reduction.
pub fn cpu_minmax_reduce(input: &[f32]) -> (f32, f32) {
    if input.is_empty() {
        return (0.0, 0.0);
    }
    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;
    for &v in input {
        if v < min_val {
            min_val = v;
        }
        if v > max_val {
            max_val = v;
        }
    }
    (min_val, max_val)
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers --

    fn make_data(n: usize, f: impl Fn(usize) -> f32) -> Vec<f32> {
        (0..n).map(f).collect()
    }

    fn ramp(n: usize) -> Vec<f32> {
        make_data(n, |i| i as f32 / n as f32)
    }

    fn uniform(n: usize, v: f32) -> Vec<f32> {
        vec![v; n]
    }

    fn normal_ish(n: usize) -> Vec<f32> {
        // Deterministic pseudo-normal via simple LCG + Box–Muller-ish mapping.
        let mut vals = Vec::with_capacity(n);
        let mut seed: u64 = 42;
        for _ in 0..n {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = (seed >> 33) as f64 / (1u64 << 31) as f64;
            // Map uniform to rough bell via inverse-ish transform.
            let v = (u - 0.5) * 6.0; // range ≈ [-3, 3]
            vals.push(v as f32);
        }
        vals
    }

    fn data_with_outliers(n: usize, outlier_val: f32) -> Vec<f32> {
        let mut d = ramp(n);
        d[0] = -outlier_val;
        d[n - 1] = outlier_val;
        d
    }

    // ---- CalibrationDataset ----

    #[test]
    fn test_dataset_empty() {
        let ds = CalibrationDataset::new(64, 1024);
        assert_eq!(ds.count, 0);
        assert_eq!(ds.mean(), 0.0);
        assert_eq!(ds.variance(), 0.0);
    }

    #[test]
    fn test_dataset_single_value() {
        let mut ds = CalibrationDataset::new(64, 1024);
        ds.record(&[3.14]);
        assert_eq!(ds.count, 1);
        assert!((ds.min - 3.14).abs() < 1e-6);
        assert!((ds.max - 3.14).abs() < 1e-6);
        assert!((ds.mean() - 3.14).abs() < 1e-6);
    }

    #[test]
    fn test_dataset_min_max() {
        let mut ds = CalibrationDataset::new(64, 4096);
        let data = make_data(1000, |i| (i as f32 - 500.0) * 0.01);
        ds.record(&data);
        assert!((ds.min - (-5.0)).abs() < 0.01);
        assert!((ds.max - 4.99).abs() < 0.02);
    }

    #[test]
    fn test_dataset_mean_variance() {
        let mut ds = CalibrationDataset::new(64, 4096);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        ds.record(&data);
        assert!((ds.mean() - 3.0).abs() < 1e-9);
        assert!((ds.variance() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_dataset_histogram_nonzero() {
        let mut ds = CalibrationDataset::new(10, 4096);
        ds.record(&ramp(1000));
        let total: u64 = ds.histogram.iter().sum();
        assert!(total > 0, "histogram should have counts");
    }

    #[test]
    fn test_dataset_multiple_records() {
        let mut ds = CalibrationDataset::new(64, 4096);
        ds.record(&[1.0, 2.0]);
        ds.record(&[3.0, 4.0]);
        assert_eq!(ds.count, 4);
        assert!((ds.min - 1.0).abs() < 1e-9);
        assert!((ds.max - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_dataset_percentile_basic() {
        let mut ds = CalibrationDataset::new(64, 4096);
        let data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        ds.record(&data);
        let p50 = ds.percentile(50.0);
        assert!((p50 - 500.0).abs() < 5.0, "median ~500, got {p50}");
    }

    #[test]
    fn test_dataset_percentile_extremes() {
        let mut ds = CalibrationDataset::new(64, 4096);
        let data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        ds.record(&data);
        assert!(ds.percentile(0.0) < 1.0);
        assert!(ds.percentile(100.0) > 998.0);
    }

    #[test]
    fn test_dataset_empty_record() {
        let mut ds = CalibrationDataset::new(64, 1024);
        ds.record(&[]);
        assert_eq!(ds.count, 0);
    }

    #[test]
    fn test_dataset_negative_values() {
        let mut ds = CalibrationDataset::new(64, 4096);
        ds.record(&[-10.0, -5.0, 0.0, 5.0, 10.0]);
        assert!((ds.min - (-10.0)).abs() < 1e-9);
        assert!((ds.max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_dataset_reservoir_overflow() {
        let mut ds = CalibrationDataset::new(64, 10);
        let data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        ds.record(&data);
        assert_eq!(ds.reservoir.len(), 10);
        assert_eq!(ds.count, 1000);
    }

    // ---- CalibrationObserver ----

    #[test]
    fn test_observer_basic() {
        let mut obs = CalibrationObserver::new(64, 1024);
        obs.observe("layer0", &[1.0, 2.0, 3.0]);
        obs.observe("layer1", &[4.0, 5.0, 6.0]);
        assert_eq!(obs.layer_count(), 2);
    }

    #[test]
    fn test_observer_accumulates() {
        let mut obs = CalibrationObserver::new(64, 1024);
        obs.observe("attn.q", &[1.0]);
        obs.observe("attn.q", &[2.0]);
        let stats = obs.get_layer_stats("attn.q").unwrap();
        assert_eq!(stats.count, 2);
    }

    #[test]
    fn test_observer_missing_layer() {
        let obs = CalibrationObserver::new(64, 1024);
        assert!(obs.get_layer_stats("nonexistent").is_none());
    }

    // ---- ScaleComputer ----

    #[test]
    fn test_symmetric_scale_basic() {
        let sz = ScaleComputer::compute_symmetric(-1.0, 1.0, 8);
        assert!((sz.scale - 1.0 / 127.0).abs() < 1e-9);
        assert_eq!(sz.zero_point, 0);
    }

    #[test]
    fn test_symmetric_scale_zero_range() {
        let sz = ScaleComputer::compute_symmetric(0.0, 0.0, 8);
        assert_eq!(sz.scale, 1.0);
        assert_eq!(sz.zero_point, 0);
    }

    #[test]
    fn test_asymmetric_scale_basic() {
        let sz = ScaleComputer::compute_asymmetric(0.0, 1.0, 8);
        assert!((sz.scale - 1.0 / 255.0).abs() < 1e-9);
        assert_eq!(sz.zero_point, 0);
    }

    #[test]
    fn test_asymmetric_scale_negative_range() {
        let sz = ScaleComputer::compute_asymmetric(-1.0, 1.0, 8);
        assert!((sz.scale - 2.0 / 255.0).abs() < 1e-6);
        assert!(sz.zero_point > 0);
    }

    #[test]
    fn test_scale_4bit_symmetric() {
        let sz = ScaleComputer::compute_symmetric(-1.0, 1.0, 4);
        let qmax = (1i64 << 3) - 1; // 7
        assert!((sz.scale - 1.0 / qmax as f64).abs() < 1e-9);
    }

    #[test]
    fn test_scale_from_dataset_minmax() {
        let mut ds = CalibrationDataset::new(64, 4096);
        ds.record(&[-2.0, 0.0, 2.0]);
        let sz = ScaleComputer::compute(&ds, QuantRangeMethod::MinMax, 8, true);
        assert!((sz.scale - 2.0 / 127.0).abs() < 1e-9);
    }

    #[test]
    fn test_scale_from_dataset_percentile() {
        let mut ds = CalibrationDataset::new(64, 4096);
        let data = normal_ish(10000);
        ds.record(&data);
        let sz_mm = ScaleComputer::compute(&ds, QuantRangeMethod::MinMax, 8, true);
        let sz_pc = ScaleComputer::compute(&ds, QuantRangeMethod::Percentile(99.0), 8, true);
        // Percentile should give a tighter range (smaller scale).
        assert!(sz_pc.scale <= sz_mm.scale, "percentile scale should be <= minmax scale");
    }

    #[test]
    fn test_scale_mse_method() {
        let mut ds = CalibrationDataset::new(64, 4096);
        ds.record(&data_with_outliers(1000, 100.0));
        let sz_mm = ScaleComputer::compute(&ds, QuantRangeMethod::MinMax, 8, true);
        let sz_mse = ScaleComputer::compute(&ds, QuantRangeMethod::Mse, 8, true);
        // MSE-optimal should not be wider than MinMax (it clips outliers).
        assert!(sz_mse.scale <= sz_mm.scale + 1e-6);
    }

    #[test]
    fn test_scale_entropy_method() {
        let mut ds = CalibrationDataset::new(128, 4096);
        ds.record(&normal_ish(5000));
        let sz = ScaleComputer::compute(&ds, QuantRangeMethod::Entropy, 8, true);
        assert!(sz.scale > 0.0);
    }

    #[test]
    fn test_determine_range_minmax() {
        let mut ds = CalibrationDataset::new(64, 4096);
        ds.record(&[-3.0, 0.0, 7.0]);
        let (lo, hi) = ScaleComputer::determine_range(&ds, QuantRangeMethod::MinMax);
        assert!((lo - (-3.0)).abs() < 1e-9);
        assert!((hi - 7.0).abs() < 1e-9);
    }

    // ---- QuantSimulator ----

    #[test]
    fn test_quantize_dequantize_symmetric_roundtrip() {
        let scale = 1.0 / 127.0;
        for v in [-1.0f32, -0.5, 0.0, 0.5, 1.0] {
            let q = QuantSimulator::quantize_symmetric(v, scale, 8);
            let deq = QuantSimulator::dequantize_symmetric(q, scale);
            assert!((v - deq).abs() < scale as f32 + 1e-6, "roundtrip for {v}");
        }
    }

    #[test]
    fn test_quantize_dequantize_asymmetric_roundtrip() {
        let sz = ScaleComputer::compute_asymmetric(0.0, 1.0, 8);
        for v in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let q = QuantSimulator::quantize_asymmetric(v, sz.scale, sz.zero_point, 8);
            let deq = QuantSimulator::dequantize_asymmetric(q, sz.scale, sz.zero_point);
            assert!((v - deq).abs() < sz.scale as f32 + 1e-6, "roundtrip for {v}");
        }
    }

    #[test]
    fn test_simulate_empty() {
        let sz = ScaleZeroPoint { scale: 0.01, zero_point: 0 };
        let m = QuantSimulator::simulate(&[], sz, 8, true);
        assert_eq!(m.mse, 0.0);
    }

    #[test]
    fn test_simulate_perfect_zero() {
        let data = uniform(100, 0.0);
        let sz = ScaleComputer::compute_symmetric(-1.0, 1.0, 8);
        let m = QuantSimulator::simulate(&data, sz, 8, true);
        assert_eq!(m.mse, 0.0);
        assert_eq!(m.max_error, 0.0);
    }

    #[test]
    fn test_simulate_snr_finite() {
        let data = normal_ish(1000);
        let mut ds = CalibrationDataset::new(64, 4096);
        ds.record(&data);
        let sz = ScaleComputer::compute(&ds, QuantRangeMethod::MinMax, 8, true);
        let m = QuantSimulator::simulate(&data, sz, 8, true);
        assert!(m.snr_db.is_finite(), "SNR should be finite");
        assert!(m.snr_db > 0.0, "SNR should be positive for non-zero signal");
    }

    #[test]
    fn test_simulate_sqnr_equals_snr() {
        let data = ramp(500);
        let mut ds = CalibrationDataset::new(64, 4096);
        ds.record(&data);
        let sz = ScaleComputer::compute(&ds, QuantRangeMethod::MinMax, 8, true);
        let m = QuantSimulator::simulate(&data, sz, 8, true);
        assert!((m.sqnr_db - m.snr_db).abs() < 1e-12);
    }

    #[test]
    fn test_simulate_mse_bounded() {
        let data = ramp(1000);
        let sz = ScaleComputer::compute_symmetric(0.0, 1.0, 8);
        let m = QuantSimulator::simulate(&data, sz, 8, true);
        // MSE should be small for 8-bit quantisation of unit-range data.
        assert!(m.mse < 1e-4, "MSE too large: {}", m.mse);
    }

    #[test]
    fn test_simulate_max_error_bounded() {
        let data = ramp(1000);
        let sz = ScaleComputer::compute_symmetric(0.0, 1.0, 8);
        let m = QuantSimulator::simulate(&data, sz, 8, true);
        // Max error should be at most 0.5 * scale.
        let half_step = sz.scale * 0.5 + 1e-9;
        assert!(m.max_error <= half_step + 1e-6, "max_err={}", m.max_error);
    }

    #[test]
    fn test_simulate_mae_less_than_max() {
        let data = normal_ish(1000);
        let mut ds = CalibrationDataset::new(64, 4096);
        ds.record(&data);
        let sz = ScaleComputer::compute(&ds, QuantRangeMethod::MinMax, 8, true);
        let m = QuantSimulator::simulate(&data, sz, 8, true);
        assert!(m.mae <= m.max_error + 1e-12);
    }

    #[test]
    fn test_simulate_higher_bits_less_error() {
        let data = normal_ish(1000);
        let mut ds = CalibrationDataset::new(64, 4096);
        ds.record(&data);
        let sz8 = ScaleComputer::compute(&ds, QuantRangeMethod::MinMax, 8, true);
        let sz4 = ScaleComputer::compute(&ds, QuantRangeMethod::MinMax, 4, true);
        let m8 = QuantSimulator::simulate(&data, sz8, 8, true);
        let m4 = QuantSimulator::simulate(&data, sz4, 4, true);
        assert!(m8.mse <= m4.mse, "8-bit should have <= MSE than 4-bit");
    }

    #[test]
    fn test_simulate_asymmetric_positive() {
        let data = ramp(500);
        let sz = ScaleComputer::compute_asymmetric(0.0, 1.0, 8);
        let m = QuantSimulator::simulate(&data, sz, 8, false);
        assert!(m.mse < 1e-4);
    }

    // ---- Per-channel simulation ----

    #[test]
    fn test_per_channel_simulation() {
        let channels = 4;
        let per_ch = 256;
        let mut data = Vec::with_capacity(channels * per_ch);
        for ch in 0..channels {
            let scale = (ch + 1) as f32;
            for i in 0..per_ch {
                data.push((i as f32 / per_ch as f32 - 0.5) * scale);
            }
        }
        let results = QuantSimulator::simulate_per_channel(
            &data,
            channels,
            QuantRangeMethod::MinMax,
            8,
            true,
        );
        assert_eq!(results.len(), channels);
        for m in &results {
            assert!(m.mse < 0.01);
        }
    }

    #[test]
    fn test_per_channel_vs_per_tensor() {
        let channels = 4;
        let per_ch = 256;
        let mut data = Vec::with_capacity(channels * per_ch);
        for ch in 0..channels {
            let scale = (ch + 1) as f32 * 10.0;
            for i in 0..per_ch {
                data.push((i as f32 / per_ch as f32 - 0.5) * scale);
            }
        }
        // Per-tensor
        let mut ds_all = CalibrationDataset::new(64, data.len());
        ds_all.record(&data);
        let sz_all = ScaleComputer::compute(&ds_all, QuantRangeMethod::MinMax, 8, true);
        let m_all = QuantSimulator::simulate(&data, sz_all, 8, true);
        // Per-channel
        let per_ch_results = QuantSimulator::simulate_per_channel(
            &data,
            channels,
            QuantRangeMethod::MinMax,
            8,
            true,
        );
        let avg_per_ch_mse: f64 =
            per_ch_results.iter().map(|m| m.mse).sum::<f64>() / channels as f64;
        // Per-channel should have lower average MSE.
        assert!(
            avg_per_ch_mse <= m_all.mse + 1e-9,
            "per-channel MSE {avg_per_ch_mse} should be <= per-tensor MSE {}",
            m_all.mse
        );
    }

    #[test]
    fn test_per_channel_empty() {
        let results =
            QuantSimulator::simulate_per_channel(&[], 0, QuantRangeMethod::MinMax, 8, true);
        assert!(results.is_empty());
    }

    // ---- CalibrationReport ----

    #[test]
    fn test_report_generation() {
        let mut obs = CalibrationObserver::new(64, 4096);
        obs.observe("layer.0", &normal_ish(1000));
        obs.observe("layer.1", &normal_ish(1000));
        let report = CalibrationReport::generate(
            &obs,
            QuantScheme::PerTensor,
            QuantRangeMethod::MinMax,
            8,
            true,
        );
        assert_eq!(report.layers.len(), 2);
    }

    #[test]
    fn test_report_sorted_by_name() {
        let mut obs = CalibrationObserver::new(64, 4096);
        obs.observe("z_layer", &ramp(100));
        obs.observe("a_layer", &ramp(100));
        let report = CalibrationReport::generate(
            &obs,
            QuantScheme::PerTensor,
            QuantRangeMethod::MinMax,
            8,
            true,
        );
        assert_eq!(report.layers[0].layer_name, "a_layer");
        assert_eq!(report.layers[1].layer_name, "z_layer");
    }

    #[test]
    fn test_report_sensitivity_ranking() {
        let mut obs = CalibrationObserver::new(64, 4096);
        obs.observe("good_layer", &ramp(1000));
        // A layer with extreme outliers should be harder to quantise.
        obs.observe("bad_layer", &data_with_outliers(1000, 1000.0));
        let report = CalibrationReport::generate(
            &obs,
            QuantScheme::PerTensor,
            QuantRangeMethod::MinMax,
            8,
            true,
        );
        let ranking = report.sensitivity_ranking();
        // worst (lowest SQNR) should come first
        assert_eq!(ranking[0].layer_name, "bad_layer");
    }

    #[test]
    fn test_report_sensitive_layers() {
        let mut obs = CalibrationObserver::new(64, 4096);
        obs.observe("ok", &ramp(1000));
        obs.observe("tricky", &data_with_outliers(1000, 1000.0));
        let report = CalibrationReport::generate(
            &obs,
            QuantScheme::PerTensor,
            QuantRangeMethod::MinMax,
            8,
            true,
        );
        let sensitive = report.sensitive_layers(40.0);
        assert!(!sensitive.is_empty());
    }

    #[test]
    fn test_report_display() {
        let mut obs = CalibrationObserver::new(64, 4096);
        obs.observe("fc1", &ramp(100));
        let report = CalibrationReport::generate(
            &obs,
            QuantScheme::PerTensor,
            QuantRangeMethod::MinMax,
            8,
            true,
        );
        let s = format!("{}", report.layers[0]);
        assert!(s.contains("fc1"));
    }

    #[test]
    fn test_report_recommendation_strings() {
        let excellent = QuantErrorMetrics {
            mse: 1e-10,
            max_error: 1e-5,
            snr_db: 50.0,
            sqnr_db: 50.0,
            mae: 1e-6,
        };
        let poor =
            QuantErrorMetrics { mse: 1.0, max_error: 5.0, snr_db: 5.0, sqnr_db: 5.0, mae: 0.5 };
        let rec_e = CalibrationReport::recommend(&excellent, 8);
        let rec_p = CalibrationReport::recommend(&poor, 8);
        assert_eq!(rec_e, "excellent");
        assert!(rec_p.contains("poor"));
    }

    // ---- AutoQuantizer ----

    #[test]
    fn test_auto_quantizer_default() {
        let aq = AutoQuantizer::default();
        assert_eq!(aq.default_bits, 8);
        assert_eq!(aq.fallback_bits, 16);
    }

    #[test]
    fn test_auto_quantizer_analyse_basic() {
        let mut obs = CalibrationObserver::new(64, 4096);
        obs.observe("layer.0", &ramp(1000));
        obs.observe("layer.1", &normal_ish(1000));
        let aq = AutoQuantizer::default();
        let decisions = aq.analyse(&obs);
        assert_eq!(decisions.len(), 2);
    }

    #[test]
    fn test_auto_quantizer_sensitive_layer_gets_more_bits() {
        let mut obs = CalibrationObserver::new(64, 4096);
        obs.observe("easy", &ramp(1000));
        obs.observe("hard", &data_with_outliers(1000, 1000.0));
        let aq = AutoQuantizer::new(8, 16, 40.0, true);
        let decisions = aq.analyse(&obs);
        let hard = decisions.iter().find(|d| d.layer_name == "hard").unwrap();
        assert!(hard.sensitive, "outlier layer should be sensitive");
        assert!(hard.bits > 8, "sensitive layer should get more bits");
    }

    #[test]
    fn test_auto_quantizer_nonsensitive_stays() {
        let mut obs = CalibrationObserver::new(64, 4096);
        obs.observe("easy", &ramp(1000));
        let aq = AutoQuantizer::new(8, 16, 5.0, true);
        let decisions = aq.analyse(&obs);
        let easy = decisions.iter().find(|d| d.layer_name == "easy").unwrap();
        assert!(!easy.sensitive);
        assert_eq!(easy.bits, 8);
    }

    #[test]
    fn test_auto_quantizer_scheme_selection() {
        let mut obs = CalibrationObserver::new(64, 4096);
        obs.observe("easy", &ramp(1000));
        obs.observe("hard", &data_with_outliers(1000, 1000.0));
        let aq = AutoQuantizer::new(8, 16, 40.0, true);
        let decisions = aq.analyse(&obs);
        let easy = decisions.iter().find(|d| d.layer_name == "easy").unwrap();
        let hard = decisions.iter().find(|d| d.layer_name == "hard").unwrap();
        assert_eq!(easy.scheme, QuantScheme::PerTensor);
        assert_eq!(hard.scheme, QuantScheme::PerChannel);
    }

    #[test]
    fn test_auto_quantizer_display() {
        let d = LayerQuantDecision {
            layer_name: "fc".into(),
            scheme: QuantScheme::PerTensor,
            bits: 8,
            method: QuantRangeMethod::MinMax,
            metrics: QuantErrorMetrics {
                mse: 0.0,
                max_error: 0.0,
                snr_db: 60.0,
                sqnr_db: 60.0,
                mae: 0.0,
            },
            sensitive: false,
        };
        let s = format!("{d}");
        assert!(s.contains("fc"));
    }

    // ---- CPU reference kernels ----

    #[test]
    fn test_cpu_symmetric_simulate() {
        let input = vec![-1.0f32, 0.0, 0.5, 1.0];
        let scale = 1.0 / 127.0;
        let (output, errors) = cpu_quant_simulate_symmetric(&input, scale, 8);
        assert_eq!(output.len(), 4);
        assert_eq!(errors.len(), 4);
        // Zero should roundtrip perfectly.
        assert!((output[1]).abs() < 1e-9);
        assert!((errors[1]).abs() < 1e-9);
    }

    #[test]
    fn test_cpu_asymmetric_simulate() {
        let input = vec![0.0f32, 0.25, 0.5, 0.75, 1.0];
        let sz = ScaleComputer::compute_asymmetric(0.0, 1.0, 8);
        let (output, errors) = cpu_quant_simulate_asymmetric(&input, sz.scale, sz.zero_point, 8);
        assert_eq!(output.len(), 5);
        for &e in &errors {
            assert!(e < 1e-4, "error too large: {e}");
        }
    }

    #[test]
    fn test_cpu_histogram_collect() {
        let input = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let hist = cpu_histogram_collect(&input, 0.0, 1.0, 4);
        assert_eq!(hist.len(), 4);
        let total: u64 = hist.iter().sum();
        assert_eq!(total, 5);
    }

    #[test]
    fn test_cpu_histogram_single_bin() {
        let input = vec![0.5, 0.5, 0.5];
        let hist = cpu_histogram_collect(&input, 0.0, 1.0, 1);
        assert_eq!(hist, vec![3]);
    }

    #[test]
    fn test_cpu_minmax_reduce() {
        let input = vec![3.0, -2.0, 7.0, 1.0];
        let (min, max) = cpu_minmax_reduce(&input);
        assert_eq!(min, -2.0);
        assert_eq!(max, 7.0);
    }

    #[test]
    fn test_cpu_minmax_reduce_empty() {
        let (min, max) = cpu_minmax_reduce(&[]);
        assert_eq!(min, 0.0);
        assert_eq!(max, 0.0);
    }

    #[test]
    fn test_cpu_minmax_single() {
        let (min, max) = cpu_minmax_reduce(&[42.0]);
        assert_eq!(min, 42.0);
        assert_eq!(max, 42.0);
    }

    // ---- OpenCL kernel source ----

    #[test]
    fn test_opencl_source_not_empty() {
        assert!(!QUANT_CALIBRATOR_CL.is_empty());
    }

    #[test]
    fn test_opencl_source_contains_kernels() {
        assert!(QUANT_CALIBRATOR_CL.contains("quant_simulate_symmetric"));
        assert!(QUANT_CALIBRATOR_CL.contains("quant_simulate_asymmetric"));
        assert!(QUANT_CALIBRATOR_CL.contains("histogram_collect"));
        assert!(QUANT_CALIBRATOR_CL.contains("minmax_reduce"));
    }

    // ---- QuantScheme / QuantRangeMethod Display ----

    #[test]
    fn test_quant_scheme_display() {
        assert_eq!(format!("{}", QuantScheme::PerTensor), "PerTensor");
        assert_eq!(format!("{}", QuantScheme::PerChannel), "PerChannel");
        assert_eq!(format!("{}", QuantScheme::PerGroup(128)), "PerGroup(128)");
    }

    #[test]
    fn test_quant_range_method_display() {
        assert_eq!(format!("{}", QuantRangeMethod::MinMax), "MinMax");
        assert!(format!("{}", QuantRangeMethod::Percentile(99.5)).contains("99.50"));
        assert_eq!(format!("{}", QuantRangeMethod::Mse), "MSE");
        assert!(format!("{}", QuantRangeMethod::Entropy).contains("KL"));
    }

    // ---- Edge cases ----

    #[test]
    fn test_uniform_data_quantises_perfectly() {
        let data = uniform(100, 0.5);
        let mut ds = CalibrationDataset::new(64, 4096);
        ds.record(&data);
        let sz = ScaleComputer::compute(&ds, QuantRangeMethod::MinMax, 8, true);
        let m = QuantSimulator::simulate(&data, sz, 8, true);
        // All identical values → zero quantisation error.
        assert!(m.mse < 1e-10, "uniform data mse={}", m.mse);
    }

    #[test]
    fn test_single_value_quantises() {
        let data = vec![42.0f32];
        let mut ds = CalibrationDataset::new(64, 1024);
        ds.record(&data);
        let sz = ScaleComputer::compute(&ds, QuantRangeMethod::MinMax, 8, true);
        let m = QuantSimulator::simulate(&data, sz, 8, true);
        assert!(m.max_error < 1.0);
    }

    #[test]
    fn test_outlier_dominated_range() {
        let data = data_with_outliers(1000, 1000.0);
        let mut ds = CalibrationDataset::new(64, 4096);
        ds.record(&data);
        let sz_mm = ScaleComputer::compute(&ds, QuantRangeMethod::MinMax, 8, true);
        let sz_mse = ScaleComputer::compute(&ds, QuantRangeMethod::Mse, 8, true);
        let m_mm = QuantSimulator::simulate(&data, sz_mm, 8, true);
        let m_mse = QuantSimulator::simulate(&data, sz_mse, 8, true);
        // MSE-optimal should give lower overall MSE (clips outliers).
        assert!(m_mse.mse <= m_mm.mse + 1e-6, "mse_opt {} vs mm {}", m_mse.mse, m_mm.mse);
    }

    #[test]
    fn test_very_small_values() {
        let data = make_data(100, |i| i as f32 * 1e-7);
        let mut ds = CalibrationDataset::new(64, 4096);
        ds.record(&data);
        let sz = ScaleComputer::compute(&ds, QuantRangeMethod::MinMax, 8, true);
        assert!(sz.scale > 0.0);
    }

    #[test]
    fn test_very_large_values() {
        let data = make_data(100, |i| i as f32 * 1e6);
        let mut ds = CalibrationDataset::new(64, 4096);
        ds.record(&data);
        let sz = ScaleComputer::compute(&ds, QuantRangeMethod::MinMax, 8, true);
        assert!(sz.scale > 0.0);
    }

    // ---- Property-like tests ----

    #[test]
    fn test_quant_error_always_bounded_symmetric() {
        for seed_offset in 0..5 {
            let data: Vec<f32> = (0..500)
                .map(|i| {
                    let x = (i + seed_offset * 500) as f64 * 0.007;
                    (x.sin() * 3.0) as f32
                })
                .collect();
            let mut ds = CalibrationDataset::new(64, 4096);
            ds.record(&data);
            let sz = ScaleComputer::compute(&ds, QuantRangeMethod::MinMax, 8, true);
            let m = QuantSimulator::simulate(&data, sz, 8, true);
            // Max error should never exceed scale.
            assert!(
                m.max_error <= sz.scale + 1e-6,
                "seed_offset={seed_offset}: max_error={} > scale={}",
                m.max_error,
                sz.scale
            );
        }
    }

    #[test]
    fn test_quant_error_always_bounded_asymmetric() {
        for seed_offset in 0..5 {
            let data: Vec<f32> = (0..500)
                .map(|i| {
                    let x = (i + seed_offset * 500) as f64 * 0.007;
                    ((x.sin() + 1.0) * 2.0) as f32 // [0, 4] range
                })
                .collect();
            let mut ds = CalibrationDataset::new(64, 4096);
            ds.record(&data);
            let sz = ScaleComputer::compute(&ds, QuantRangeMethod::MinMax, 8, false);
            let m = QuantSimulator::simulate(&data, sz, 8, false);
            assert!(
                m.max_error <= sz.scale + 1e-6,
                "seed_offset={seed_offset}: max_error={} > scale={}",
                m.max_error,
                sz.scale
            );
        }
    }

    #[test]
    fn test_snr_increases_with_bits() {
        let data = normal_ish(2000);
        let mut ds = CalibrationDataset::new(64, 4096);
        ds.record(&data);
        let mut prev_snr = f64::NEG_INFINITY;
        for bits in [2, 4, 8, 16] {
            let sz = ScaleComputer::compute(&ds, QuantRangeMethod::MinMax, bits, true);
            let m = QuantSimulator::simulate(&data, sz, bits, true);
            assert!(
                m.snr_db >= prev_snr - 0.1,
                "SNR should increase with bits: {bits}bit snr={:.2} < prev={:.2}",
                m.snr_db,
                prev_snr,
            );
            prev_snr = m.snr_db;
        }
    }

    #[test]
    fn test_mse_decreases_with_bits() {
        let data = normal_ish(2000);
        let mut ds = CalibrationDataset::new(64, 4096);
        ds.record(&data);
        let mut prev_mse = f64::MAX;
        for bits in [2, 4, 8, 16] {
            let sz = ScaleComputer::compute(&ds, QuantRangeMethod::MinMax, bits, true);
            let m = QuantSimulator::simulate(&data, sz, bits, true);
            assert!(
                m.mse <= prev_mse + 1e-9,
                "MSE should decrease with bits: {bits}bit mse={:.6e} > prev={:.6e}",
                m.mse,
                prev_mse,
            );
            prev_mse = m.mse;
        }
    }

    // ---- Misc coverage ----

    #[test]
    fn test_quant_error_metrics_display() {
        let m = QuantErrorMetrics {
            mse: 1e-5,
            max_error: 1e-3,
            snr_db: 45.0,
            sqnr_db: 45.0,
            mae: 5e-4,
        };
        let s = format!("{m}");
        assert!(s.contains("SNR"));
        assert!(s.contains("MSE"));
    }

    #[test]
    fn test_scale_zero_point_debug() {
        let sz = ScaleZeroPoint { scale: 0.01, zero_point: 128 };
        let s = format!("{sz:?}");
        assert!(s.contains("scale"));
        assert!(s.contains("zero_point"));
    }

    #[test]
    fn test_kl_divergence_identical() {
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let kl = ScaleComputer::kl_divergence(&p, &p);
        assert!(kl < 1e-10, "KL(P||P) should be ~0, got {kl}");
    }

    #[test]
    fn test_kl_divergence_different() {
        let p = vec![0.5, 0.5, 0.0, 0.0];
        let q = vec![0.25, 0.25, 0.25, 0.25];
        let kl = ScaleComputer::kl_divergence(&p, &q);
        assert!(kl > 0.0, "KL should be > 0 for different distributions");
    }
}

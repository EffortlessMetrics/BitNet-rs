//! Quantization calibration for Intel Arc A770 GPU inference.
//!
//! Calibrate quantization scales and zero-points using representative data,
//! supporting multiple quantization schemes (symmetric, asymmetric, per-channel,
//! per-group, per-token).
//!
//! # Overview
//!
//! - [`QuantScheme`]: quantization granularity (symmetric, asymmetric, per-channel, …)
//! - [`CalibrationMethod`]: algorithm for deriving scale/zero-point (MinMax, Percentile, …)
//! - [`CalibrationStats`]: accumulated statistics (min, max, mean, variance, histogram)
//! - [`QuantParams`]: computed quantization parameters (scale, zero-point, bits)
//! - [`CalibrationSession`]: orchestrates calibration over multiple data batches
//! - [`ScaleComputer`]: derives optimal scales from collected statistics
//! - [`QuantizationError`]: measures quantization fidelity (MSE, SNR, max/mean error)

use std::fmt;

// ── Quantization scheme ─────────────────────────────────────────────────────

/// Quantization granularity / symmetry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantScheme {
    /// Symmetric range: `[-max_abs, +max_abs]`, zero-point = 0.
    Symmetric,
    /// Asymmetric range: `[min, max]`, non-zero zero-point.
    Asymmetric,
    /// Per-output-channel scales (one scale per row).
    PerChannel,
    /// Per-group scales with a fixed group size.
    PerGroup(usize),
    /// Per-token scales (one scale per input token / row in activation).
    PerToken,
}

impl fmt::Display for QuantScheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Symmetric => write!(f, "Symmetric"),
            Self::Asymmetric => write!(f, "Asymmetric"),
            Self::PerChannel => write!(f, "PerChannel"),
            Self::PerGroup(g) => write!(f, "PerGroup({g})"),
            Self::PerToken => write!(f, "PerToken"),
        }
    }
}

// ── Calibration method ──────────────────────────────────────────────────────

/// Algorithm used to derive quantization parameters from observed data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CalibrationMethod {
    /// Use the observed min/max directly.
    MinMax,
    /// Clip to the given percentile (e.g. 99.99 → 0.9999).
    Percentile(f32),
    /// Minimise mean-squared quantization error.
    Mse,
    /// Minimise KL-divergence (entropy) between original and quantized distributions.
    Entropy,
    /// Exponential moving average of min/max across batches.
    MovingAverage,
}

impl fmt::Display for CalibrationMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MinMax => write!(f, "MinMax"),
            Self::Percentile(p) => write!(f, "Percentile({p:.4})"),
            Self::Mse => write!(f, "MSE"),
            Self::Entropy => write!(f, "Entropy"),
            Self::MovingAverage => write!(f, "MovingAverage"),
        }
    }
}

// ── Calibration statistics ──────────────────────────────────────────────────

/// Per-tensor statistics accumulated during calibration.
#[derive(Debug, Clone)]
pub struct CalibrationStats {
    /// Running minimum value.
    pub min_val: f32,
    /// Running maximum value.
    pub max_val: f32,
    /// Running mean (Welford).
    pub mean: f64,
    /// Running variance (Welford, population).
    pub variance: f64,
    /// 256-bin histogram spanning `[min_val, max_val]`.
    pub histogram: Vec<u64>,
    /// Number of individual values observed.
    count: u64,
    /// Welford M2 accumulator.
    m2: f64,
    /// EMA min (for MovingAverage method).
    ema_min: f32,
    /// EMA max (for MovingAverage method).
    ema_max: f32,
}

impl CalibrationStats {
    /// Create empty statistics.
    pub fn new() -> Self {
        Self {
            min_val: f32::INFINITY,
            max_val: f32::NEG_INFINITY,
            mean: 0.0,
            variance: 0.0,
            histogram: vec![0u64; 256],
            count: 0,
            m2: 0.0,
            ema_min: f32::INFINITY,
            ema_max: f32::NEG_INFINITY,
        }
    }

    /// Number of values observed so far.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// EMA min (for MovingAverage method).
    pub fn ema_min(&self) -> f32 {
        self.ema_min
    }

    /// EMA max (for MovingAverage method).
    pub fn ema_max(&self) -> f32 {
        self.ema_max
    }

    /// Observe a batch of values, updating all running statistics.
    pub fn observe(&mut self, data: &[f32]) {
        for &v in data {
            self.observe_one(v);
        }
        self.rebuild_histogram(data);
    }

    /// Observe a single value (min/max/mean/variance).
    fn observe_one(&mut self, v: f32) {
        if v < self.min_val {
            self.min_val = v;
        }
        if v > self.max_val {
            self.max_val = v;
        }
        self.count += 1;
        let delta = v as f64 - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = v as f64 - self.mean;
        self.m2 += delta * delta2;
        self.variance = if self.count > 1 { self.m2 / self.count as f64 } else { 0.0 };
    }

    /// Update the EMA min/max with a smoothing factor.
    pub fn update_ema(&mut self, alpha: f32) {
        if self.ema_min == f32::INFINITY {
            self.ema_min = self.min_val;
            self.ema_max = self.max_val;
        } else {
            self.ema_min = alpha * self.min_val + (1.0 - alpha) * self.ema_min;
            self.ema_max = alpha * self.max_val + (1.0 - alpha) * self.ema_max;
        }
    }

    /// Rebuild the 256-bin histogram over the current `[min_val, max_val]` range.
    fn rebuild_histogram(&mut self, data: &[f32]) {
        if self.min_val >= self.max_val {
            // Constant tensor — put everything in bin 0.
            self.histogram = vec![0u64; 256];
            self.histogram[0] = data.len() as u64;
            return;
        }
        self.histogram = vec![0u64; 256];
        let range = self.max_val - self.min_val;
        for &v in data {
            let bin = ((v - self.min_val) / range * 255.0).clamp(0.0, 255.0) as usize;
            self.histogram[bin] += 1;
        }
    }

    /// Compute the percentile value from the histogram.
    pub fn percentile(&self, p: f32) -> f32 {
        let total: u64 = self.histogram.iter().sum();
        if total == 0 {
            return 0.0;
        }
        let target = (p * total as f32) as u64;
        let mut cumulative = 0u64;
        for (i, &count) in self.histogram.iter().enumerate() {
            cumulative += count;
            if cumulative >= target {
                let range = self.max_val - self.min_val;
                return self.min_val + (i as f32 / 255.0) * range;
            }
        }
        self.max_val
    }
}

impl Default for CalibrationStats {
    fn default() -> Self {
        Self::new()
    }
}

// ── Quantization parameters ─────────────────────────────────────────────────

/// Computed quantization parameters for a single tensor/group.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantParams {
    /// Scale factor: `real_value = (quantized_value - zero_point) * scale`.
    pub scale: f32,
    /// Zero-point offset in quantized domain.
    pub zero_point: i32,
    /// Bit-width of the quantized representation.
    pub bits: u8,
    /// Quantization scheme used.
    pub scheme: QuantScheme,
}

impl QuantParams {
    /// Quantize a single float value.
    pub fn quantize(&self, value: f32) -> i32 {
        let qmin = self.qmin();
        let qmax = self.qmax();
        if self.scale == 0.0 {
            return self.zero_point;
        }
        let q = (value / self.scale).round() as i32 + self.zero_point;
        q.clamp(qmin, qmax)
    }

    /// Dequantize a single quantized value back to float.
    pub fn dequantize(&self, quantized: i32) -> f32 {
        (quantized - self.zero_point) as f32 * self.scale
    }

    /// Minimum representable quantized value.
    fn qmin(&self) -> i32 {
        match self.scheme {
            QuantScheme::Symmetric
            | QuantScheme::PerChannel
            | QuantScheme::PerGroup(_)
            | QuantScheme::PerToken => -(1i32 << (self.bits - 1)),
            QuantScheme::Asymmetric => 0,
        }
    }

    /// Maximum representable quantized value.
    fn qmax(&self) -> i32 {
        match self.scheme {
            QuantScheme::Symmetric
            | QuantScheme::PerChannel
            | QuantScheme::PerGroup(_)
            | QuantScheme::PerToken => (1i32 << (self.bits - 1)) - 1,
            QuantScheme::Asymmetric => (1i32 << self.bits) - 1,
        }
    }
}

impl fmt::Display for QuantParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "QuantParams(scale={:.6}, zp={}, bits={}, {})",
            self.scale, self.zero_point, self.bits, self.scheme
        )
    }
}

// ── Scale computer ──────────────────────────────────────────────────────────

/// Derives optimal quantization parameters from calibration statistics.
pub struct ScaleComputer;

impl ScaleComputer {
    /// Compute [`QuantParams`] for the given scheme, method, and stats.
    pub fn compute(
        stats: &CalibrationStats,
        scheme: QuantScheme,
        method: CalibrationMethod,
        bits: u8,
    ) -> QuantParams {
        let (min_val, max_val) = Self::effective_range(stats, method);
        match scheme {
            QuantScheme::Symmetric
            | QuantScheme::PerChannel
            | QuantScheme::PerGroup(_)
            | QuantScheme::PerToken => Self::symmetric_params(min_val, max_val, bits, scheme),
            QuantScheme::Asymmetric => Self::asymmetric_params(min_val, max_val, bits),
        }
    }

    /// Compute per-group parameters: one [`QuantParams`] per group.
    pub fn compute_per_group(
        data: &[f32],
        group_size: usize,
        method: CalibrationMethod,
        bits: u8,
    ) -> Vec<QuantParams> {
        data.chunks(group_size)
            .map(|chunk| {
                let mut stats = CalibrationStats::new();
                stats.observe(chunk);
                Self::compute(&stats, QuantScheme::PerGroup(group_size), method, bits)
            })
            .collect()
    }

    /// Compute per-token parameters (one per row).
    pub fn compute_per_token(
        data: &[f32],
        cols: usize,
        method: CalibrationMethod,
        bits: u8,
    ) -> Vec<QuantParams> {
        data.chunks(cols)
            .map(|row| {
                let mut stats = CalibrationStats::new();
                stats.observe(row);
                Self::compute(&stats, QuantScheme::PerToken, method, bits)
            })
            .collect()
    }

    /// Determine the effective `[min, max]` range per the calibration method.
    fn effective_range(stats: &CalibrationStats, method: CalibrationMethod) -> (f32, f32) {
        match method {
            CalibrationMethod::MinMax => (stats.min_val, stats.max_val),
            CalibrationMethod::Percentile(p) => {
                let lo = stats.percentile(1.0 - p);
                let hi = stats.percentile(p);
                (lo, hi)
            }
            CalibrationMethod::Mse => Self::mse_optimal_range(stats),
            CalibrationMethod::Entropy => Self::entropy_optimal_range(stats),
            CalibrationMethod::MovingAverage => (stats.ema_min(), stats.ema_max()),
        }
    }

    /// Symmetric: `scale = max_abs / (2^(bits-1) - 1)`, zero-point = 0.
    fn symmetric_params(min_val: f32, max_val: f32, bits: u8, scheme: QuantScheme) -> QuantParams {
        let max_abs = min_val.abs().max(max_val.abs());
        let qmax = (1i32 << (bits - 1)) - 1;
        let scale = if qmax == 0 { 1.0 } else { max_abs / qmax as f32 };
        QuantParams { scale, zero_point: 0, bits, scheme }
    }

    /// Asymmetric: `scale = (max - min) / (2^bits - 1)`, zero-point computed.
    fn asymmetric_params(min_val: f32, max_val: f32, bits: u8) -> QuantParams {
        let qmax = (1i32 << bits) - 1;
        let range = max_val - min_val;
        let scale = if qmax == 0 || range == 0.0 { 1.0 } else { range / qmax as f32 };
        let zero_point = if scale == 0.0 { 0 } else { (-(min_val / scale)).round() as i32 };
        let zero_point = zero_point.clamp(0, qmax);
        QuantParams { scale, zero_point, bits, scheme: QuantScheme::Asymmetric }
    }

    /// Search for the `[min, max]` sub-range that minimises MSE.
    fn mse_optimal_range(stats: &CalibrationStats) -> (f32, f32) {
        let full_min = stats.min_val;
        let full_max = stats.max_val;
        if full_min >= full_max {
            return (full_min, full_max);
        }
        let steps = 40;
        let mut best_range = (full_min, full_max);
        let mut best_mse = f64::MAX;
        for i in 0..=steps {
            let frac = i as f32 / steps as f32;
            let shrink = frac * 0.2; // shrink up to 20%
            let lo = full_min + shrink * (full_max - full_min);
            let hi = full_max - shrink * (full_max - full_min);
            if lo >= hi {
                continue;
            }
            let mse = Self::estimate_histogram_mse(stats, lo, hi, 8);
            if mse < best_mse {
                best_mse = mse;
                best_range = (lo, hi);
            }
        }
        best_range
    }

    /// Search for the `[min, max]` sub-range that minimises KL-divergence.
    fn entropy_optimal_range(stats: &CalibrationStats) -> (f32, f32) {
        let full_min = stats.min_val;
        let full_max = stats.max_val;
        if full_min >= full_max {
            return (full_min, full_max);
        }
        let steps = 40;
        let mut best_range = (full_min, full_max);
        let mut best_kl = f64::MAX;
        for i in 0..=steps {
            let frac = i as f32 / steps as f32;
            let shrink = frac * 0.2;
            let lo = full_min + shrink * (full_max - full_min);
            let hi = full_max - shrink * (full_max - full_min);
            if lo >= hi {
                continue;
            }
            let kl = Self::estimate_histogram_kl(stats, lo, hi, 8);
            if kl < best_kl {
                best_kl = kl;
                best_range = (lo, hi);
            }
        }
        best_range
    }

    /// Estimate MSE from histogram when quantizing to `[lo, hi]`.
    fn estimate_histogram_mse(stats: &CalibrationStats, lo: f32, hi: f32, bits: u8) -> f64 {
        let n_bins = stats.histogram.len();
        let total: u64 = stats.histogram.iter().sum();
        if total == 0 {
            return 0.0;
        }
        let orig_range = stats.max_val - stats.min_val;
        if orig_range == 0.0 {
            return 0.0;
        }
        let qmax = ((1u32 << bits) - 1) as f32;
        let q_range = hi - lo;
        if q_range <= 0.0 {
            return f64::MAX;
        }
        let scale = q_range / qmax;
        let mut mse = 0.0f64;
        for (i, &count) in stats.histogram.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let val = stats.min_val + (i as f32 / (n_bins - 1) as f32) * orig_range;
            let clamped = val.clamp(lo, hi);
            let q = ((clamped - lo) / scale).round() * scale + lo;
            let err = (val - q) as f64;
            mse += err * err * count as f64;
        }
        mse / total as f64
    }

    /// Estimate KL-divergence from histogram when quantizing to `[lo, hi]`.
    fn estimate_histogram_kl(stats: &CalibrationStats, lo: f32, hi: f32, bits: u8) -> f64 {
        let n_bins = stats.histogram.len();
        let total: u64 = stats.histogram.iter().sum();
        if total == 0 {
            return 0.0;
        }
        let orig_range = stats.max_val - stats.min_val;
        if orig_range == 0.0 {
            return 0.0;
        }
        let n_quant_bins = 1usize << bits;
        let q_range = hi - lo;
        if q_range <= 0.0 {
            return f64::MAX;
        }
        // Build quantized distribution.
        let mut q_hist = vec![0u64; n_quant_bins];
        for (i, &count) in stats.histogram.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let val = stats.min_val + (i as f32 / (n_bins - 1) as f32) * orig_range;
            let clamped = val.clamp(lo, hi);
            let qbin = ((clamped - lo) / q_range * (n_quant_bins - 1) as f32).round() as usize;
            let qbin = qbin.min(n_quant_bins - 1);
            q_hist[qbin] += count;
        }
        // Map quantized bins back to original resolution.
        let mut q_expanded = vec![0.0f64; n_bins];
        for (qi, &qcount) in q_hist.iter().enumerate() {
            if qcount == 0 {
                continue;
            }
            let center_val = lo + (qi as f32 / (n_quant_bins - 1) as f32) * q_range;
            let orig_bin =
                ((center_val - stats.min_val) / orig_range * (n_bins - 1) as f32).round() as usize;
            let orig_bin = orig_bin.min(n_bins - 1);
            q_expanded[orig_bin] += qcount as f64;
        }
        // KL(P || Q).
        let eps = 1e-10;
        let total_f = total as f64;
        let q_total: f64 = q_expanded.iter().sum();
        let mut kl = 0.0f64;
        for (i, &count) in stats.histogram.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let p = count as f64 / total_f;
            let q = (q_expanded[i] / q_total).max(eps);
            kl += p * (p / q).ln();
        }
        kl
    }
}

// ── Calibration session ─────────────────────────────────────────────────────

/// Orchestrates calibration over multiple data batches.
#[derive(Debug)]
pub struct CalibrationSession {
    /// Per-tensor statistics keyed by name.
    pub stats: std::collections::HashMap<String, CalibrationStats>,
    /// Number of batches observed.
    pub num_samples: usize,
    /// Calibration method.
    pub method: CalibrationMethod,
    /// EMA smoothing factor (used when method = MovingAverage).
    pub ema_alpha: f32,
}

impl CalibrationSession {
    /// Create a new session with the given calibration method.
    pub fn new(method: CalibrationMethod) -> Self {
        Self { stats: std::collections::HashMap::new(), num_samples: 0, method, ema_alpha: 0.1 }
    }

    /// Builder: set the EMA smoothing factor.
    pub fn with_ema_alpha(mut self, alpha: f32) -> Self {
        self.ema_alpha = alpha;
        self
    }

    /// Record a batch of observed values for the named tensor.
    pub fn observe(&mut self, tensor_name: &str, data: &[f32]) {
        let entry = self.stats.entry(tensor_name.to_string()).or_default();
        entry.observe(data);
        if matches!(self.method, CalibrationMethod::MovingAverage) {
            entry.update_ema(self.ema_alpha);
        }
        self.num_samples += 1;
    }

    /// Compute quantization parameters for the named tensor.
    pub fn compute_params(
        &self,
        tensor_name: &str,
        scheme: QuantScheme,
        bits: u8,
    ) -> Option<QuantParams> {
        self.stats.get(tensor_name).map(|s| ScaleComputer::compute(s, scheme, self.method, bits))
    }

    /// Compute parameters for all observed tensors.
    pub fn compute_all_params(
        &self,
        scheme: QuantScheme,
        bits: u8,
    ) -> std::collections::HashMap<String, QuantParams> {
        self.stats
            .iter()
            .map(|(name, s)| {
                let p = ScaleComputer::compute(s, scheme, self.method, bits);
                (name.clone(), p)
            })
            .collect()
    }
}

// ── Quantization error measurement ─────────────────────────────────────────

/// Quantization fidelity metrics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantizationError {
    /// Mean squared error.
    pub mse: f64,
    /// Signal-to-noise ratio in dB.
    pub snr_db: f64,
    /// Maximum absolute error.
    pub max_error: f32,
    /// Mean absolute error.
    pub mean_error: f32,
}

impl QuantizationError {
    /// Measure quantization error for the given data and params.
    pub fn measure(data: &[f32], params: &QuantParams) -> Self {
        if data.is_empty() {
            return Self { mse: 0.0, snr_db: f64::INFINITY, max_error: 0.0, mean_error: 0.0 };
        }
        let mut sum_sq_err = 0.0f64;
        let mut sum_abs_err = 0.0f64;
        let mut max_err: f32 = 0.0;
        let mut signal_power = 0.0f64;

        for &v in data {
            let q = params.quantize(v);
            let dq = params.dequantize(q);
            let err = (v - dq).abs();
            sum_sq_err += (err as f64) * (err as f64);
            sum_abs_err += err as f64;
            if err > max_err {
                max_err = err;
            }
            signal_power += (v as f64) * (v as f64);
        }

        let n = data.len() as f64;
        let mse = sum_sq_err / n;
        let snr_db =
            if mse > 0.0 { 10.0 * ((signal_power / n) / mse).log10() } else { f64::INFINITY };

        Self { mse, snr_db, max_error: max_err, mean_error: (sum_abs_err / n) as f32 }
    }

    /// Measure per-group quantization error.
    pub fn measure_per_group(data: &[f32], params_list: &[QuantParams], group_size: usize) -> Self {
        if data.is_empty() {
            return Self { mse: 0.0, snr_db: f64::INFINITY, max_error: 0.0, mean_error: 0.0 };
        }
        let mut sum_sq_err = 0.0f64;
        let mut sum_abs_err = 0.0f64;
        let mut max_err: f32 = 0.0;
        let mut signal_power = 0.0f64;

        for (chunk, params) in data.chunks(group_size).zip(params_list.iter()) {
            for &v in chunk {
                let q = params.quantize(v);
                let dq = params.dequantize(q);
                let err = (v - dq).abs();
                sum_sq_err += (err as f64) * (err as f64);
                sum_abs_err += err as f64;
                if err > max_err {
                    max_err = err;
                }
                signal_power += (v as f64) * (v as f64);
            }
        }

        let n = data.len() as f64;
        let mse = sum_sq_err / n;
        let snr_db =
            if mse > 0.0 { 10.0 * ((signal_power / n) / mse).log10() } else { f64::INFINITY };

        Self { mse, snr_db, max_error: max_err, mean_error: (sum_abs_err / n) as f32 }
    }
}

impl fmt::Display for QuantizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MSE={:.6e}, SNR={:.2}dB, max_err={:.6}, mean_err={:.6}",
            self.mse, self.snr_db, self.max_error, self.mean_error
        )
    }
}

// ── CPU reference: quantize / dequantize a whole slice ──────────────────────

/// Quantize a slice of floats using the given parameters (CPU reference).
pub fn quantize_slice(data: &[f32], params: &QuantParams) -> Vec<i32> {
    data.iter().map(|&v| params.quantize(v)).collect()
}

/// Dequantize a slice of quantized values back to floats (CPU reference).
pub fn dequantize_slice(quantized: &[i32], params: &QuantParams) -> Vec<f32> {
    quantized.iter().map(|&q| params.dequantize(q)).collect()
}

// ═════════════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ─────────────────────────────────────────────────────────

    /// Deterministic pseudo-random data in `[lo, hi]`.
    fn pseudo_random(n: usize, lo: f32, hi: f32, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                // xorshift64
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let t = (state as f64) / (u64::MAX as f64);
                lo + (hi - lo) * t as f32
            })
            .collect()
    }

    fn assert_close(a: f32, b: f32, tol: f32, msg: &str) {
        assert!((a - b).abs() <= tol, "{msg}: {a} vs {b} (diff={}, tol={tol})", (a - b).abs());
    }

    // ── QuantScheme Display ─────────────────────────────────────────────

    #[test]
    fn quant_scheme_display() {
        assert_eq!(QuantScheme::Symmetric.to_string(), "Symmetric");
        assert_eq!(QuantScheme::Asymmetric.to_string(), "Asymmetric");
        assert_eq!(QuantScheme::PerChannel.to_string(), "PerChannel");
        assert_eq!(QuantScheme::PerGroup(64).to_string(), "PerGroup(64)");
        assert_eq!(QuantScheme::PerToken.to_string(), "PerToken");
    }

    // ── CalibrationMethod Display ───────────────────────────────────────

    #[test]
    fn calibration_method_display() {
        assert_eq!(CalibrationMethod::MinMax.to_string(), "MinMax");
        assert!(CalibrationMethod::Percentile(0.9999).to_string().contains("Percentile"));
        assert_eq!(CalibrationMethod::Mse.to_string(), "MSE");
        assert_eq!(CalibrationMethod::Entropy.to_string(), "Entropy");
        assert_eq!(CalibrationMethod::MovingAverage.to_string(), "MovingAverage");
    }

    // ── CalibrationStats ────────────────────────────────────────────────

    #[test]
    fn stats_empty() {
        let s = CalibrationStats::new();
        assert_eq!(s.count(), 0);
        assert_eq!(s.min_val, f32::INFINITY);
        assert_eq!(s.max_val, f32::NEG_INFINITY);
    }

    #[test]
    fn stats_single_value() {
        let mut s = CalibrationStats::new();
        s.observe(&[42.0]);
        assert_eq!(s.count(), 1);
        assert_eq!(s.min_val, 42.0);
        assert_eq!(s.max_val, 42.0);
        assert!((s.mean - 42.0).abs() < 1e-6);
        assert!(s.variance < 1e-6);
    }

    #[test]
    fn stats_multiple_values() {
        let mut s = CalibrationStats::new();
        s.observe(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(s.count(), 5);
        assert_eq!(s.min_val, 1.0);
        assert_eq!(s.max_val, 5.0);
        assert_close(s.mean as f32, 3.0, 1e-5, "mean");
        // Population variance of [1,2,3,4,5] = 2.0
        assert_close(s.variance as f32, 2.0, 1e-5, "variance");
    }

    #[test]
    fn stats_histogram_256_bins() {
        let mut s = CalibrationStats::new();
        let data: Vec<f32> = (0..1000).map(|i| i as f32 / 999.0).collect();
        s.observe(&data);
        assert_eq!(s.histogram.len(), 256);
        let total: u64 = s.histogram.iter().sum();
        assert_eq!(total, 1000);
    }

    #[test]
    fn stats_percentile_basic() {
        let mut s = CalibrationStats::new();
        let data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        s.observe(&data);
        // 50th percentile should be near 500
        let p50 = s.percentile(0.5);
        assert!(p50 > 400.0 && p50 < 600.0, "p50={p50}");
        // 99th percentile should be near 990
        let p99 = s.percentile(0.99);
        assert!(p99 > 900.0, "p99={p99}");
    }

    #[test]
    fn stats_ema_update() {
        let mut s = CalibrationStats::new();
        s.observe(&[-5.0, 5.0]);
        s.update_ema(0.1);
        assert_eq!(s.ema_min(), -5.0); // first update copies directly
        assert_eq!(s.ema_max(), 5.0);

        // Simulate new range [-3, 3]
        s.min_val = -3.0;
        s.max_val = 3.0;
        s.update_ema(0.1);
        // EMA should move toward new values
        assert!(s.ema_min() > -5.0, "ema_min={}", s.ema_min());
        assert!(s.ema_max() < 5.0, "ema_max={}", s.ema_max());
    }

    // ── Symmetric quantization ──────────────────────────────────────────

    #[test]
    fn symmetric_int8_basic() {
        let mut s = CalibrationStats::new();
        s.observe(&[-1.0, 0.0, 1.0]);
        let p = ScaleComputer::compute(&s, QuantScheme::Symmetric, CalibrationMethod::MinMax, 8);
        assert_eq!(p.zero_point, 0);
        assert_eq!(p.bits, 8);
        // scale = 1.0 / 127
        assert_close(p.scale, 1.0 / 127.0, 1e-6, "symmetric scale");
    }

    #[test]
    fn symmetric_quantize_dequantize_roundtrip() {
        let mut s = CalibrationStats::new();
        s.observe(&[-2.0, 2.0]);
        let p = ScaleComputer::compute(&s, QuantScheme::Symmetric, CalibrationMethod::MinMax, 8);
        let q = p.quantize(1.0);
        let dq = p.dequantize(q);
        assert_close(dq, 1.0, p.scale, "roundtrip");
    }

    #[test]
    fn symmetric_zero_maps_to_zero() {
        let mut s = CalibrationStats::new();
        s.observe(&[-10.0, 10.0]);
        let p = ScaleComputer::compute(&s, QuantScheme::Symmetric, CalibrationMethod::MinMax, 8);
        assert_eq!(p.quantize(0.0), 0);
        assert_eq!(p.dequantize(0), 0.0);
    }

    // ── Asymmetric quantization ─────────────────────────────────────────

    #[test]
    fn asymmetric_int8_basic() {
        let mut s = CalibrationStats::new();
        s.observe(&[0.0, 1.0]);
        let p = ScaleComputer::compute(&s, QuantScheme::Asymmetric, CalibrationMethod::MinMax, 8);
        assert_eq!(p.bits, 8);
        // scale = 1.0 / 255
        assert_close(p.scale, 1.0 / 255.0, 1e-6, "asymmetric scale");
        // zero_point should map 0.0 → 0
        assert_eq!(p.zero_point, 0);
    }

    #[test]
    fn asymmetric_negative_range() {
        let mut s = CalibrationStats::new();
        s.observe(&[-2.0, 6.0]);
        let p = ScaleComputer::compute(&s, QuantScheme::Asymmetric, CalibrationMethod::MinMax, 8);
        let expected_zp = (-(-2.0f32) / p.scale).round() as i32;
        assert_eq!(p.zero_point, expected_zp.clamp(0, 255));
    }

    #[test]
    fn asymmetric_quantize_clamps() {
        let mut s = CalibrationStats::new();
        s.observe(&[0.0, 1.0]);
        let p = ScaleComputer::compute(&s, QuantScheme::Asymmetric, CalibrationMethod::MinMax, 8);
        // Values outside [0,1] should be clamped
        let q_low = p.quantize(-10.0);
        let q_high = p.quantize(10.0);
        assert!(q_low >= 0);
        assert!(q_high <= 255);
    }

    // ── PerChannel ──────────────────────────────────────────────────────

    #[test]
    fn per_channel_same_as_symmetric() {
        let mut s = CalibrationStats::new();
        s.observe(&[-3.0, 3.0]);
        let p = ScaleComputer::compute(&s, QuantScheme::PerChannel, CalibrationMethod::MinMax, 8);
        assert_eq!(p.zero_point, 0);
        assert_eq!(p.scheme, QuantScheme::PerChannel);
    }

    // ── PerGroup ────────────────────────────────────────────────────────

    #[test]
    fn per_group_multiple_groups() {
        let data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
        let params = ScaleComputer::compute_per_group(&data, 64, CalibrationMethod::MinMax, 8);
        assert_eq!(params.len(), 4);
        for p in &params {
            assert_eq!(p.bits, 8);
            assert_eq!(p.zero_point, 0);
            assert!(p.scale > 0.0);
        }
    }

    #[test]
    fn per_group_error_lower_than_global() {
        let data = pseudo_random(256, -1.0, 1.0, 12345);
        // Global symmetric
        let mut stats = CalibrationStats::new();
        stats.observe(&data);
        let global_p =
            ScaleComputer::compute(&stats, QuantScheme::Symmetric, CalibrationMethod::MinMax, 8);
        let global_err = QuantizationError::measure(&data, &global_p);

        // Per-group (group_size=32)
        let group_params =
            ScaleComputer::compute_per_group(&data, 32, CalibrationMethod::MinMax, 8);
        let group_err = QuantizationError::measure_per_group(&data, &group_params, 32);

        // Per-group should be comparable or better; allow tiny fp tolerance.
        assert!(
            group_err.mse <= global_err.mse * 1.05 + 1e-8,
            "per-group MSE {} much worse than global MSE {}",
            group_err.mse,
            global_err.mse
        );
    }

    // ── PerToken ────────────────────────────────────────────────────────

    #[test]
    fn per_token_two_rows() {
        let data = vec![
            -1.0, 0.0, 1.0, 0.5, // row 0
            -10.0, 0.0, 10.0, 5.0, // row 1
        ];
        let params = ScaleComputer::compute_per_token(&data, 4, CalibrationMethod::MinMax, 8);
        assert_eq!(params.len(), 2);
        // Row 1 has larger range → larger scale
        assert!(params[1].scale > params[0].scale);
    }

    // ── CalibrationMethod: Percentile ───────────────────────────────────

    #[test]
    fn percentile_clips_outliers() {
        let mut data: Vec<f32> = (0..998).map(|i| i as f32 / 997.0).collect();
        data.push(100.0); // outlier
        data.push(-100.0); // outlier
        let mut s = CalibrationStats::new();
        s.observe(&data);
        let p_full =
            ScaleComputer::compute(&s, QuantScheme::Symmetric, CalibrationMethod::MinMax, 8);
        let p_clip = ScaleComputer::compute(
            &s,
            QuantScheme::Symmetric,
            CalibrationMethod::Percentile(0.99),
            8,
        );
        // Percentile should yield smaller scale (ignoring outliers)
        assert!(
            p_clip.scale < p_full.scale,
            "clipped scale {} should be < full scale {}",
            p_clip.scale,
            p_full.scale
        );
    }

    #[test]
    fn percentile_100_equals_minmax() {
        let data = pseudo_random(500, -5.0, 5.0, 999);
        let mut s = CalibrationStats::new();
        s.observe(&data);
        let p_mm = ScaleComputer::compute(&s, QuantScheme::Symmetric, CalibrationMethod::MinMax, 8);
        let p_p100 = ScaleComputer::compute(
            &s,
            QuantScheme::Symmetric,
            CalibrationMethod::Percentile(1.0),
            8,
        );
        assert_close(p_mm.scale, p_p100.scale, 1e-5, "percentile(1.0) ≈ minmax");
    }

    // ── CalibrationMethod: MSE ──────────────────────────────────────────

    #[test]
    fn mse_method_produces_valid_params() {
        let data = pseudo_random(1000, -2.0, 2.0, 42);
        let mut s = CalibrationStats::new();
        s.observe(&data);
        let p = ScaleComputer::compute(&s, QuantScheme::Symmetric, CalibrationMethod::Mse, 8);
        assert!(p.scale > 0.0);
        assert_eq!(p.bits, 8);
    }

    #[test]
    fn mse_method_scale_not_larger_than_minmax() {
        let data = pseudo_random(1000, -2.0, 2.0, 77);
        let mut s = CalibrationStats::new();
        s.observe(&data);
        let mm = ScaleComputer::compute(&s, QuantScheme::Symmetric, CalibrationMethod::MinMax, 8);
        let mse = ScaleComputer::compute(&s, QuantScheme::Symmetric, CalibrationMethod::Mse, 8);
        assert!(
            mse.scale <= mm.scale + 1e-6,
            "MSE scale {} > MinMax scale {}",
            mse.scale,
            mm.scale
        );
    }

    // ── CalibrationMethod: Entropy ──────────────────────────────────────

    #[test]
    fn entropy_method_produces_valid_params() {
        let data = pseudo_random(1000, -3.0, 3.0, 123);
        let mut s = CalibrationStats::new();
        s.observe(&data);
        let p = ScaleComputer::compute(&s, QuantScheme::Symmetric, CalibrationMethod::Entropy, 8);
        assert!(p.scale > 0.0);
        assert_eq!(p.bits, 8);
    }

    #[test]
    fn entropy_method_scale_not_larger_than_minmax() {
        let data = pseudo_random(1000, -3.0, 3.0, 456);
        let mut s = CalibrationStats::new();
        s.observe(&data);
        let mm = ScaleComputer::compute(&s, QuantScheme::Symmetric, CalibrationMethod::MinMax, 8);
        let ent = ScaleComputer::compute(&s, QuantScheme::Symmetric, CalibrationMethod::Entropy, 8);
        assert!(
            ent.scale <= mm.scale + 1e-6,
            "Entropy scale {} > MinMax scale {}",
            ent.scale,
            mm.scale
        );
    }

    // ── CalibrationMethod: MovingAverage ────────────────────────────────

    #[test]
    fn moving_average_converges() {
        let mut session =
            CalibrationSession::new(CalibrationMethod::MovingAverage).with_ema_alpha(0.3);
        for i in 0..20 {
            let data = pseudo_random(100, -1.0, 1.0, 1000 + i);
            session.observe("tensor_a", &data);
        }
        let s = session.stats.get("tensor_a").unwrap();
        assert!(s.ema_min() < 0.0, "ema_min={}", s.ema_min());
        assert!(s.ema_max() > 0.0, "ema_max={}", s.ema_max());
    }

    #[test]
    fn moving_average_ema_tracks_shift() {
        let mut session =
            CalibrationSession::new(CalibrationMethod::MovingAverage).with_ema_alpha(0.5);
        // Initial range [-1, 1]
        session.observe("t", &[-1.0, 1.0]);
        let ema_max_1 = session.stats.get("t").unwrap().ema_max();

        // Shift to [-10, 10]
        for _ in 0..10 {
            session.observe("t", &[-10.0, 10.0]);
        }
        let ema_max_2 = session.stats.get("t").unwrap().ema_max();
        assert!(ema_max_2 > ema_max_1, "EMA should track toward 10");
    }

    // ── CalibrationSession ──────────────────────────────────────────────

    #[test]
    fn session_tracks_multiple_tensors() {
        let mut session = CalibrationSession::new(CalibrationMethod::MinMax);
        session.observe("weight", &[-1.0, 1.0]);
        session.observe("activation", &[0.0, 5.0]);
        assert_eq!(session.stats.len(), 2);
        assert_eq!(session.num_samples, 2);
    }

    #[test]
    fn session_compute_params() {
        let mut session = CalibrationSession::new(CalibrationMethod::MinMax);
        session.observe("w", &[-2.0, 0.0, 2.0]);
        let p = session.compute_params("w", QuantScheme::Symmetric, 8).unwrap();
        assert_close(p.scale, 2.0 / 127.0, 1e-6, "session compute");
    }

    #[test]
    fn session_compute_all() {
        let mut session = CalibrationSession::new(CalibrationMethod::MinMax);
        session.observe("a", &[-1.0, 1.0]);
        session.observe("b", &[-5.0, 5.0]);
        let all = session.compute_all_params(QuantScheme::Symmetric, 8);
        assert_eq!(all.len(), 2);
        assert!(all.contains_key("a"));
        assert!(all.contains_key("b"));
        assert!(all["b"].scale > all["a"].scale);
    }

    #[test]
    fn session_missing_tensor_returns_none() {
        let session = CalibrationSession::new(CalibrationMethod::MinMax);
        assert!(session.compute_params("missing", QuantScheme::Symmetric, 8).is_none());
    }

    // ── QuantizationError measurement ───────────────────────────────────

    #[test]
    fn error_zero_for_exact_values() {
        let p = QuantParams { scale: 1.0, zero_point: 0, bits: 8, scheme: QuantScheme::Symmetric };
        let data = vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let err = QuantizationError::measure(&data, &p);
        assert!(err.mse < 1e-12, "mse={}", err.mse);
        assert_eq!(err.max_error, 0.0);
    }

    #[test]
    fn error_positive_for_fine_grained_data() {
        let data = pseudo_random(1000, -1.0, 1.0, 7777);
        let mut s = CalibrationStats::new();
        s.observe(&data);
        let p = ScaleComputer::compute(&s, QuantScheme::Symmetric, CalibrationMethod::MinMax, 8);
        let err = QuantizationError::measure(&data, &p);
        assert!(err.mse > 0.0);
        assert!(err.snr_db > 0.0);
        assert!(err.max_error > 0.0);
        assert!(err.mean_error > 0.0);
    }

    #[test]
    fn error_snr_higher_with_more_bits() {
        let data = pseudo_random(1000, -1.0, 1.0, 8888);
        let mut s = CalibrationStats::new();
        s.observe(&data);
        let p4 = ScaleComputer::compute(&s, QuantScheme::Symmetric, CalibrationMethod::MinMax, 4);
        let p8 = ScaleComputer::compute(&s, QuantScheme::Symmetric, CalibrationMethod::MinMax, 8);
        let e4 = QuantizationError::measure(&data, &p4);
        let e8 = QuantizationError::measure(&data, &p8);
        assert!(
            e8.snr_db > e4.snr_db,
            "8-bit SNR ({:.1}) should be > 4-bit SNR ({:.1})",
            e8.snr_db,
            e4.snr_db
        );
    }

    #[test]
    fn error_mse_decreases_with_more_bits() {
        let data = pseudo_random(500, -2.0, 2.0, 3333);
        let mut s = CalibrationStats::new();
        s.observe(&data);
        let p2 = ScaleComputer::compute(&s, QuantScheme::Symmetric, CalibrationMethod::MinMax, 2);
        let p4 = ScaleComputer::compute(&s, QuantScheme::Symmetric, CalibrationMethod::MinMax, 4);
        let p8 = ScaleComputer::compute(&s, QuantScheme::Symmetric, CalibrationMethod::MinMax, 8);
        let e2 = QuantizationError::measure(&data, &p2);
        let e4 = QuantizationError::measure(&data, &p4);
        let e8 = QuantizationError::measure(&data, &p8);
        assert!(e8.mse < e4.mse, "8-bit MSE < 4-bit MSE");
        assert!(e4.mse < e2.mse, "4-bit MSE < 2-bit MSE");
    }

    #[test]
    fn error_empty_data() {
        let p = QuantParams { scale: 1.0, zero_point: 0, bits: 8, scheme: QuantScheme::Symmetric };
        let err = QuantizationError::measure(&[], &p);
        assert_eq!(err.mse, 0.0);
        assert_eq!(err.max_error, 0.0);
    }

    #[test]
    fn error_display() {
        let err = QuantizationError { mse: 1e-4, snr_db: 40.0, max_error: 0.01, mean_error: 0.005 };
        let s = err.to_string();
        assert!(s.contains("MSE="));
        assert!(s.contains("SNR="));
    }

    // ── Edge cases ──────────────────────────────────────────────────────

    #[test]
    fn edge_constant_tensor() {
        let mut s = CalibrationStats::new();
        s.observe(&[5.0, 5.0, 5.0, 5.0]);
        let p = ScaleComputer::compute(&s, QuantScheme::Symmetric, CalibrationMethod::MinMax, 8);
        assert!(p.scale > 0.0);
        let dq = p.dequantize(p.quantize(5.0));
        assert_close(dq, 5.0, p.scale + 1e-6, "constant tensor roundtrip");
    }

    #[test]
    fn edge_all_zeros() {
        let mut s = CalibrationStats::new();
        s.observe(&[0.0, 0.0, 0.0]);
        let p = ScaleComputer::compute(&s, QuantScheme::Symmetric, CalibrationMethod::MinMax, 8);
        assert_eq!(p.quantize(0.0), 0);
    }

    #[test]
    fn edge_very_large_range() {
        let mut s = CalibrationStats::new();
        s.observe(&[-1e6, 1e6]);
        let p = ScaleComputer::compute(&s, QuantScheme::Symmetric, CalibrationMethod::MinMax, 8);
        assert!(p.scale > 1000.0, "scale={}", p.scale);
        let q = p.quantize(999_999.0);
        let dq = p.dequantize(q);
        assert_close(dq, 999_999.0, p.scale * 2.0, "large range roundtrip");
    }

    #[test]
    fn edge_single_negative_value() {
        let mut s = CalibrationStats::new();
        s.observe(&[-7.5]);
        let p = ScaleComputer::compute(&s, QuantScheme::Symmetric, CalibrationMethod::MinMax, 8);
        assert!(p.scale > 0.0);
    }

    // ── quantize / dequantize slices ────────────────────────────────────

    #[test]
    fn quantize_dequantize_slice_roundtrip() {
        let data = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let mut s = CalibrationStats::new();
        s.observe(&data);
        let p = ScaleComputer::compute(&s, QuantScheme::Symmetric, CalibrationMethod::MinMax, 8);
        let quantized = quantize_slice(&data, &p);
        let recovered = dequantize_slice(&quantized, &p);
        for (orig, rec) in data.iter().zip(recovered.iter()) {
            assert_close(*rec, *orig, p.scale, "slice roundtrip");
        }
    }

    // ── Property: quantize→dequantize error bounded ─────────────────────

    #[test]
    fn property_roundtrip_error_bounded_symmetric() {
        let data = pseudo_random(2000, -5.0, 5.0, 54321);
        let mut s = CalibrationStats::new();
        s.observe(&data);
        let p = ScaleComputer::compute(&s, QuantScheme::Symmetric, CalibrationMethod::MinMax, 8);
        let predicted_err = QuantizationError::measure(&data, &p);
        let max_per_elem = p.scale / 2.0 + 1e-6;
        for &v in &data {
            let q = p.quantize(v);
            let dq = p.dequantize(q);
            let err = (v - dq).abs();
            assert!(err <= max_per_elem, "elem error {err} > bound {max_per_elem} for v={v}");
        }
        assert!(predicted_err.max_error <= max_per_elem);
    }

    #[test]
    fn property_roundtrip_error_bounded_asymmetric() {
        let data = pseudo_random(2000, 0.0, 10.0, 99999);
        let mut s = CalibrationStats::new();
        s.observe(&data);
        let p = ScaleComputer::compute(&s, QuantScheme::Asymmetric, CalibrationMethod::MinMax, 8);
        let max_per_elem = p.scale / 2.0 + 1e-5;
        for &v in &data {
            let q = p.quantize(v);
            let dq = p.dequantize(q);
            let err = (v - dq).abs();
            assert!(err <= max_per_elem, "elem error {err} > bound {max_per_elem} for v={v}");
        }
    }

    #[test]
    fn property_per_group_error_bounded() {
        let data = pseudo_random(256, -3.0, 3.0, 11111);
        let group_size = 32;
        let params =
            ScaleComputer::compute_per_group(&data, group_size, CalibrationMethod::MinMax, 8);
        for (chunk, p) in data.chunks(group_size).zip(params.iter()) {
            let bound = p.scale / 2.0 + 1e-6;
            for &v in chunk {
                let q = p.quantize(v);
                let dq = p.dequantize(q);
                assert!((v - dq).abs() <= bound);
            }
        }
    }

    // ── QuantParams Display ─────────────────────────────────────────────

    #[test]
    fn quant_params_display() {
        let p =
            QuantParams { scale: 0.015748, zero_point: 0, bits: 8, scheme: QuantScheme::Symmetric };
        let s = p.to_string();
        assert!(s.contains("scale="));
        assert!(s.contains("bits=8"));
        assert!(s.contains("Symmetric"));
    }

    // ── Histogram edge cases ────────────────────────────────────────────

    #[test]
    fn histogram_constant_all_in_bin_zero() {
        let mut s = CalibrationStats::new();
        s.observe(&[3.14; 100]);
        assert_eq!(s.histogram[0], 100);
        let rest_sum: u64 = s.histogram[1..].iter().sum();
        assert_eq!(rest_sum, 0);
    }

    #[test]
    fn histogram_two_values_at_extremes() {
        let mut s = CalibrationStats::new();
        s.observe(&[0.0, 1.0]);
        assert!(s.histogram[0] >= 1);
        assert!(s.histogram[255] >= 1);
    }

    // ── Bit-width variations ────────────────────────────────────────────

    #[test]
    fn symmetric_4bit() {
        let mut s = CalibrationStats::new();
        s.observe(&[-1.0, 1.0]);
        let p = ScaleComputer::compute(&s, QuantScheme::Symmetric, CalibrationMethod::MinMax, 4);
        assert_eq!(p.bits, 4);
        // qmax = 7, scale = 1.0/7
        assert_close(p.scale, 1.0 / 7.0, 1e-5, "4-bit scale");
    }

    #[test]
    fn asymmetric_4bit() {
        let mut s = CalibrationStats::new();
        s.observe(&[0.0, 1.0]);
        let p = ScaleComputer::compute(&s, QuantScheme::Asymmetric, CalibrationMethod::MinMax, 4);
        assert_eq!(p.bits, 4);
        // qmax = 15, scale = 1.0/15
        assert_close(p.scale, 1.0 / 15.0, 1e-5, "4-bit asym scale");
    }

    // ── Per-group error measurement ─────────────────────────────────────

    #[test]
    fn per_group_error_measurement() {
        let data = pseudo_random(128, -1.0, 1.0, 5555);
        let group_size = 32;
        let params =
            ScaleComputer::compute_per_group(&data, group_size, CalibrationMethod::MinMax, 8);
        let err = QuantizationError::measure_per_group(&data, &params, group_size);
        assert!(err.mse >= 0.0);
        assert!(err.snr_db > 0.0);
    }

    #[test]
    fn per_group_error_empty_data() {
        let err = QuantizationError::measure_per_group(&[], &[], 32);
        assert_eq!(err.mse, 0.0);
    }
}

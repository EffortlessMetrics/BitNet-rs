//! Quantization calibration for determining optimal scale/zero-point.
//!
//! Collects activation statistics during calibration passes and
//! computes optimal quantization parameters for INT8/INT4.

/// Quantization bit width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitWidth {
    Int4,
    Int8,
}

impl BitWidth {
    pub fn bits(&self) -> u32 {
        match self {
            BitWidth::Int4 => 4,
            BitWidth::Int8 => 8,
        }
    }

    pub fn max_int(&self) -> i32 {
        (1 << (self.bits() - 1)) - 1
    }

    pub fn min_int(&self) -> i32 {
        -(1 << (self.bits() - 1))
    }

    pub fn range(&self) -> i32 {
        self.max_int() - self.min_int()
    }
}

/// Quantization parameters for a tensor.
#[derive(Debug, Clone)]
pub struct QuantParams {
    pub scale: f32,
    pub zero_point: i32,
    pub bit_width: BitWidth,
}

impl QuantParams {
    /// Quantize a float value.
    pub fn quantize(&self, val: f32) -> i32 {
        if self.scale == 0.0 {
            return 0;
        }
        let q = (val / self.scale).round() as i32 + self.zero_point;
        q.clamp(self.bit_width.min_int(), self.bit_width.max_int())
    }

    /// Dequantize an integer value.
    pub fn dequantize(&self, val: i32) -> f32 {
        (val - self.zero_point) as f32 * self.scale
    }
}

/// Statistics collector for calibration.
#[derive(Debug, Clone)]
pub struct CalibrationStats {
    pub min_val: f32,
    pub max_val: f32,
    pub sum: f64,
    pub sum_sq: f64,
    pub count: u64,
}

impl CalibrationStats {
    pub fn new() -> Self {
        Self { min_val: f32::INFINITY, max_val: f32::NEG_INFINITY, sum: 0.0, sum_sq: 0.0, count: 0 }
    }

    /// Update with a batch of values.
    pub fn update(&mut self, values: &[f32]) {
        for &v in values {
            if v < self.min_val {
                self.min_val = v;
            }
            if v > self.max_val {
                self.max_val = v;
            }
            self.sum += v as f64;
            self.sum_sq += (v as f64) * (v as f64);
            self.count += 1;
        }
    }

    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.sum / self.count as f64
    }

    pub fn variance(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let mean = self.mean();
        self.sum_sq / self.count as f64 - mean * mean
    }

    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Range of observed values.
    pub fn range(&self) -> f32 {
        if self.count == 0 {
            return 0.0;
        }
        self.max_val - self.min_val
    }

    /// Merge with another stats collector.
    pub fn merge(&mut self, other: &CalibrationStats) {
        if other.count == 0 {
            return;
        }
        self.min_val = self.min_val.min(other.min_val);
        self.max_val = self.max_val.max(other.max_val);
        self.sum += other.sum;
        self.sum_sq += other.sum_sq;
        self.count += other.count;
    }
}

impl Default for CalibrationStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute symmetric quantization params (zero_point = 0).
pub fn symmetric_params(stats: &CalibrationStats, bw: BitWidth) -> QuantParams {
    let abs_max = stats.min_val.abs().max(stats.max_val.abs());
    let scale = if abs_max == 0.0 { 1.0 } else { abs_max / bw.max_int() as f32 };
    QuantParams { scale, zero_point: 0, bit_width: bw }
}

/// Compute asymmetric quantization params.
pub fn asymmetric_params(stats: &CalibrationStats, bw: BitWidth) -> QuantParams {
    let range = stats.range();
    let scale = if range == 0.0 { 1.0 } else { range / bw.range() as f32 };
    let zero_point = (bw.min_int() as f32 - stats.min_val / scale).round() as i32;
    QuantParams { scale, zero_point, bit_width: bw }
}

/// Quantize a slice of f32 values.
pub fn quantize_slice(data: &[f32], params: &QuantParams) -> Vec<i32> {
    data.iter().map(|&v| params.quantize(v)).collect()
}

/// Dequantize a slice of i32 values.
pub fn dequantize_slice(data: &[i32], params: &QuantParams) -> Vec<f32> {
    data.iter().map(|&v| params.dequantize(v)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_width() {
        assert_eq!(BitWidth::Int8.max_int(), 127);
        assert_eq!(BitWidth::Int8.min_int(), -128);
        assert_eq!(BitWidth::Int4.max_int(), 7);
        assert_eq!(BitWidth::Int4.min_int(), -8);
    }

    #[test]
    fn test_symmetric_roundtrip() {
        let mut stats = CalibrationStats::new();
        stats.update(&[-1.0, 0.0, 0.5, 1.0]);
        let params = symmetric_params(&stats, BitWidth::Int8);
        let q = params.quantize(0.5);
        let dq = params.dequantize(q);
        assert!((dq - 0.5).abs() < 0.02);
    }

    #[test]
    fn test_asymmetric_roundtrip() {
        let mut stats = CalibrationStats::new();
        stats.update(&[0.0, 1.0, 2.0, 3.0]);
        let params = asymmetric_params(&stats, BitWidth::Int8);
        let q = params.quantize(1.5);
        let dq = params.dequantize(q);
        assert!((dq - 1.5).abs() < 0.05);
    }

    #[test]
    fn test_clamping() {
        let params = QuantParams { scale: 0.01, zero_point: 0, bit_width: BitWidth::Int8 };
        let q = params.quantize(1000.0);
        assert!(q <= 127);
    }

    #[test]
    fn test_stats_basic() {
        let mut s = CalibrationStats::new();
        s.update(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(s.min_val, 1.0);
        assert_eq!(s.max_val, 4.0);
        assert_eq!(s.count, 4);
        assert!((s.mean() - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_stats_variance() {
        let mut s = CalibrationStats::new();
        s.update(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        assert!((s.mean() - 5.0).abs() < 1e-6);
        assert!((s.variance() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_stats_merge() {
        let mut a = CalibrationStats::new();
        a.update(&[1.0, 2.0]);
        let mut b = CalibrationStats::new();
        b.update(&[3.0, 4.0]);
        a.merge(&b);
        assert_eq!(a.count, 4);
        assert_eq!(a.min_val, 1.0);
        assert_eq!(a.max_val, 4.0);
    }

    #[test]
    fn test_quantize_slice() {
        let params = symmetric_params(
            &{
                let mut s = CalibrationStats::new();
                s.update(&[-1.0, 1.0]);
                s
            },
            BitWidth::Int8,
        );
        let q = quantize_slice(&[-1.0, 0.0, 1.0], &params);
        assert_eq!(q.len(), 3);
        assert_eq!(q[0], -127);
        assert_eq!(q[1], 0);
        assert_eq!(q[2], 127);
    }

    #[test]
    fn test_dequantize_slice() {
        let params = QuantParams { scale: 0.1, zero_point: 0, bit_width: BitWidth::Int8 };
        let dq = dequantize_slice(&[10, -10, 0], &params);
        assert!((dq[0] - 1.0).abs() < 0.01);
        assert!((dq[1] - (-1.0)).abs() < 0.01);
        assert!((dq[2]).abs() < 0.01);
    }

    #[test]
    fn test_zero_scale() {
        let params = QuantParams { scale: 0.0, zero_point: 0, bit_width: BitWidth::Int8 };
        assert_eq!(params.quantize(5.0), 0);
    }

    #[test]
    fn test_empty_stats() {
        let s = CalibrationStats::new();
        assert_eq!(s.count, 0);
        assert_eq!(s.mean(), 0.0);
        assert_eq!(s.range(), 0.0);
    }

    #[test]
    fn test_int4_range() {
        let mut stats = CalibrationStats::new();
        stats.update(&[-1.0, 1.0]);
        let params = symmetric_params(&stats, BitWidth::Int4);
        let q = params.quantize(1.0);
        assert_eq!(q, 7);
    }
}

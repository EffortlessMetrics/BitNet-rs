//! Weight tensor statistics analysis.
//!
//! Compute distribution statistics for model weights to detect
//! anomalies, guide quantization, and validate model loading.

/// Statistics for a single weight tensor.
#[derive(Debug, Clone)]
pub struct WeightStats {
    pub name: String,
    pub shape: Vec<usize>,
    pub min: f32,
    pub max: f32,
    pub mean: f64,
    pub std_dev: f64,
    pub num_zeros: usize,
    pub num_elements: usize,
}

impl WeightStats {
    /// Compute statistics for a weight tensor.
    pub fn compute(name: impl Into<String>, shape: &[usize], data: &[f32]) -> Self {
        let n = data.len();
        if n == 0 {
            return Self {
                name: name.into(),
                shape: shape.to_vec(),
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                std_dev: 0.0,
                num_zeros: 0,
                num_elements: 0,
            };
        }

        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        let mut zeros = 0usize;

        for &v in data {
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
            sum += v as f64;
            sum_sq += (v as f64) * (v as f64);
            if v == 0.0 {
                zeros += 1;
            }
        }

        let mean = sum / n as f64;
        let variance = (sum_sq / n as f64) - (mean * mean);
        let std_dev = if variance > 0.0 { variance.sqrt() } else { 0.0 };

        Self {
            name: name.into(),
            shape: shape.to_vec(),
            min,
            max,
            mean,
            std_dev,
            num_zeros: zeros,
            num_elements: n,
        }
    }

    pub fn range(&self) -> f32 {
        self.max - self.min
    }

    pub fn sparsity(&self) -> f64 {
        if self.num_elements == 0 {
            return 0.0;
        }
        self.num_zeros as f64 / self.num_elements as f64
    }

    pub fn abs_max(&self) -> f32 {
        self.min.abs().max(self.max.abs())
    }

    /// Check if the distribution looks suspicious.
    pub fn is_suspicious(&self) -> bool {
        if self.num_elements == 0 {
            return true;
        }
        // All zeros
        if self.num_zeros == self.num_elements {
            return true;
        }
        // Extremely large values
        if self.abs_max() > 1e6 {
            return true;
        }
        // NaN-like (min > max only if data is NaN)
        if self.min > self.max {
            return true;
        }
        false
    }
}

/// Aggregate statistics across all weight tensors.
#[derive(Debug)]
pub struct ModelWeightReport {
    pub tensors: Vec<WeightStats>,
}

impl ModelWeightReport {
    pub fn new() -> Self {
        Self { tensors: Vec::new() }
    }

    pub fn add(&mut self, stats: WeightStats) {
        self.tensors.push(stats);
    }

    pub fn total_elements(&self) -> usize {
        self.tensors.iter().map(|t| t.num_elements).sum()
    }

    pub fn total_parameters_millions(&self) -> f64 {
        self.total_elements() as f64 / 1e6
    }

    pub fn total_zeros(&self) -> usize {
        self.tensors.iter().map(|t| t.num_zeros).sum()
    }

    pub fn overall_sparsity(&self) -> f64 {
        let total = self.total_elements();
        if total == 0 {
            return 0.0;
        }
        self.total_zeros() as f64 / total as f64
    }

    pub fn suspicious_tensors(&self) -> Vec<&WeightStats> {
        self.tensors.iter().filter(|t| t.is_suspicious()).collect()
    }

    pub fn global_min(&self) -> f32 {
        self.tensors.iter().map(|t| t.min).fold(f32::INFINITY, f32::min)
    }

    pub fn global_max(&self) -> f32 {
        self.tensors.iter().map(|t| t.max).fold(f32::NEG_INFINITY, f32::max)
    }

    pub fn summary(&self) -> String {
        format!(
            "{} tensors, {:.1}M params, sparsity {:.1}%, range [{:.4}, {:.4}]",
            self.tensors.len(),
            self.total_parameters_millions(),
            self.overall_sparsity() * 100.0,
            self.global_min(),
            self.global_max(),
        )
    }
}

impl Default for ModelWeightReport {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_basic() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let s = WeightStats::compute("test", &[4], &data);
        assert_eq!(s.min, 1.0);
        assert_eq!(s.max, 4.0);
        assert!((s.mean - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_compute_zeros() {
        let data = vec![0.0f32, 1.0, 0.0, 2.0, 0.0];
        let s = WeightStats::compute("t", &[5], &data);
        assert_eq!(s.num_zeros, 3);
        assert!((s.sparsity() - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_empty_data() {
        let s = WeightStats::compute("empty", &[0], &[]);
        assert_eq!(s.num_elements, 0);
        assert_eq!(s.sparsity(), 0.0);
    }

    #[test]
    fn test_range() {
        let data = vec![-5.0f32, 10.0];
        let s = WeightStats::compute("t", &[2], &data);
        assert_eq!(s.range(), 15.0);
    }

    #[test]
    fn test_abs_max() {
        let data = vec![-10.0f32, 5.0];
        let s = WeightStats::compute("t", &[2], &data);
        assert_eq!(s.abs_max(), 10.0);
    }

    #[test]
    fn test_suspicious_all_zeros() {
        let data = vec![0.0f32; 10];
        let s = WeightStats::compute("t", &[10], &data);
        assert!(s.is_suspicious());
    }

    #[test]
    fn test_suspicious_large() {
        let data = vec![1e7f32];
        let s = WeightStats::compute("t", &[1], &data);
        assert!(s.is_suspicious());
    }

    #[test]
    fn test_not_suspicious() {
        let data = vec![-0.5f32, 0.0, 0.5, 1.0];
        let s = WeightStats::compute("t", &[4], &data);
        assert!(!s.is_suspicious());
    }

    #[test]
    fn test_report() {
        let mut r = ModelWeightReport::new();
        r.add(WeightStats::compute("a", &[10], &vec![1.0f32; 10]));
        r.add(WeightStats::compute("b", &[20], &vec![0.5f32; 20]));
        assert_eq!(r.total_elements(), 30);
        assert_eq!(r.tensors.len(), 2);
    }

    #[test]
    fn test_report_summary() {
        let mut r = ModelWeightReport::new();
        r.add(WeightStats::compute("w", &[1000000], &vec![0.1f32; 100]));
        let s = r.summary();
        assert!(s.contains("tensors"));
    }

    #[test]
    fn test_global_min_max() {
        let mut r = ModelWeightReport::new();
        r.add(WeightStats::compute("a", &[2], &[-5.0f32, 3.0]));
        r.add(WeightStats::compute("b", &[2], &[-2.0f32, 10.0]));
        assert_eq!(r.global_min(), -5.0);
        assert_eq!(r.global_max(), 10.0);
    }

    #[test]
    fn test_suspicious_tensors() {
        let mut r = ModelWeightReport::new();
        r.add(WeightStats::compute("ok", &[2], &[0.1, 0.2]));
        r.add(WeightStats::compute("bad", &[2], &[0.0, 0.0]));
        assert_eq!(r.suspicious_tensors().len(), 1);
    }
}

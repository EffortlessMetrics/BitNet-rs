//! Weight statistics collector.
//!
//! Compute per-layer statistics (min/max/mean/std) for model weights.

/// Statistics for a single tensor.
#[derive(Debug, Clone)]
pub struct TensorStats {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std_dev: f64,
    pub num_zeros: usize,
    pub total_elements: usize,
}

impl TensorStats {
    pub fn sparsity(&self) -> f64 {
        if self.total_elements == 0 {
            return 0.0;
        }
        self.num_zeros as f64 / self.total_elements as f64
    }

    pub fn range(&self) -> f64 {
        self.max - self.min
    }

    pub fn is_all_zeros(&self) -> bool {
        self.num_zeros == self.total_elements
    }
}

/// Compute statistics from f32 data.
pub fn compute_stats(name: &str, shape: &[usize], dtype: &str, data: &[f32]) -> TensorStats {
    if data.is_empty() {
        return TensorStats {
            name: name.to_string(),
            shape: shape.to_vec(),
            dtype: dtype.to_string(),
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            std_dev: 0.0,
            num_zeros: 0,
            total_elements: 0,
        };
    }

    let n = data.len();
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut sum = 0.0_f64;
    let mut num_zeros = 0usize;

    for &v in data {
        let v64 = v as f64;
        if v64 < min {
            min = v64;
        }
        if v64 > max {
            max = v64;
        }
        sum += v64;
        if v == 0.0 {
            num_zeros += 1;
        }
    }

    let mean = sum / n as f64;
    let var_sum: f64 = data
        .iter()
        .map(|&v| {
            let diff = v as f64 - mean;
            diff * diff
        })
        .sum();
    let std_dev = (var_sum / n as f64).sqrt();

    TensorStats {
        name: name.to_string(),
        shape: shape.to_vec(),
        dtype: dtype.to_string(),
        min,
        max,
        mean,
        std_dev,
        num_zeros,
        total_elements: n,
    }
}

/// Model-level weight statistics.
#[derive(Debug, Clone)]
pub struct ModelWeightStats {
    pub tensors: Vec<TensorStats>,
    pub total_params: u64,
    pub total_bytes: u64,
}

impl ModelWeightStats {
    pub fn new() -> Self {
        Self { tensors: Vec::new(), total_params: 0, total_bytes: 0 }
    }

    pub fn add(&mut self, stats: TensorStats, bytes_per_element: usize) {
        self.total_params += stats.total_elements as u64;
        self.total_bytes += (stats.total_elements * bytes_per_element) as u64;
        self.tensors.push(stats);
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Find tensors with suspiciously high sparsity.
    pub fn high_sparsity(&self, threshold: f64) -> Vec<&TensorStats> {
        self.tensors.iter().filter(|t| t.sparsity() > threshold).collect()
    }

    /// Find tensors with all zeros (likely uninitialized).
    pub fn all_zero_tensors(&self) -> Vec<&TensorStats> {
        self.tensors.iter().filter(|t| t.is_all_zeros()).collect()
    }

    /// Largest tensors by element count.
    pub fn largest(&self, n: usize) -> Vec<&TensorStats> {
        let mut sorted: Vec<_> = self.tensors.iter().collect();
        sorted.sort_by(|a, b| b.total_elements.cmp(&a.total_elements));
        sorted.truncate(n);
        sorted
    }

    /// Overall mean absolute value.
    pub fn overall_mean_abs(&self) -> f64 {
        if self.tensors.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.tensors.iter().map(|t| t.mean.abs() * t.total_elements as f64).sum();
        sum / self.total_params as f64
    }
}

impl Default for ModelWeightStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary for display.
#[derive(Debug)]
pub struct WeightSummary {
    pub total_tensors: usize,
    pub total_params: u64,
    pub total_bytes: u64,
    pub sparse_count: usize,
    pub zero_count: usize,
}

impl ModelWeightStats {
    pub fn summary(&self) -> WeightSummary {
        WeightSummary {
            total_tensors: self.tensors.len(),
            total_params: self.total_params,
            total_bytes: self.total_bytes,
            sparse_count: self.high_sparsity(0.5).len(),
            zero_count: self.all_zero_tensors().len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_stats_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s = compute_stats("test", &[5], "f32", &data);
        assert_eq!(s.total_elements, 5);
        assert!((s.mean - 3.0).abs() < 0.01);
        assert!((s.min - 1.0).abs() < 0.01);
        assert!((s.max - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_stats_empty() {
        let s = compute_stats("empty", &[], "f32", &[]);
        assert_eq!(s.total_elements, 0);
        assert_eq!(s.mean, 0.0);
    }

    #[test]
    fn test_sparsity() {
        let data = vec![0.0, 0.0, 1.0, 0.0];
        let s = compute_stats("sparse", &[4], "f32", &data);
        assert!((s.sparsity() - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_std_dev() {
        let data = vec![2.0, 2.0, 2.0];
        let s = compute_stats("const", &[3], "f32", &data);
        assert!(s.std_dev < 0.001);
    }

    #[test]
    fn test_range() {
        let data = vec![-1.0, 5.0];
        let s = compute_stats("r", &[2], "f32", &data);
        assert!((s.range() - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_is_all_zeros() {
        let data = vec![0.0, 0.0, 0.0];
        let s = compute_stats("z", &[3], "f32", &data);
        assert!(s.is_all_zeros());
    }

    #[test]
    fn test_model_stats_add() {
        let mut ms = ModelWeightStats::new();
        let s = compute_stats("w", &[4], "f32", &[1.0, 2.0, 3.0, 4.0]);
        ms.add(s, 4);
        assert_eq!(ms.tensor_count(), 1);
        assert_eq!(ms.total_params, 4);
        assert_eq!(ms.total_bytes, 16);
    }

    #[test]
    fn test_high_sparsity() {
        let mut ms = ModelWeightStats::new();
        ms.add(compute_stats("a", &[4], "f32", &[0.0, 0.0, 0.0, 1.0]), 4);
        ms.add(compute_stats("b", &[4], "f32", &[1.0, 2.0, 3.0, 4.0]), 4);
        assert_eq!(ms.high_sparsity(0.5).len(), 1);
    }

    #[test]
    fn test_largest() {
        let mut ms = ModelWeightStats::new();
        ms.add(compute_stats("small", &[2], "f32", &[1.0, 2.0]), 4);
        ms.add(compute_stats("big", &[4], "f32", &[1.0, 2.0, 3.0, 4.0]), 4);
        let l = ms.largest(1);
        assert_eq!(l[0].name, "big");
    }

    #[test]
    fn test_summary() {
        let mut ms = ModelWeightStats::new();
        ms.add(compute_stats("w", &[3], "f32", &[1.0, 2.0, 3.0]), 4);
        let s = ms.summary();
        assert_eq!(s.total_tensors, 1);
        assert_eq!(s.total_params, 3);
    }

    #[test]
    fn test_overall_mean_abs() {
        let mut ms = ModelWeightStats::new();
        ms.add(compute_stats("a", &[2], "f32", &[1.0, -1.0]), 4);
        // mean of [1, -1] = 0, so overall_mean_abs contribution is 0
        assert!(ms.overall_mean_abs() < 0.01);
    }

    #[test]
    fn test_all_zero_tensors() {
        let mut ms = ModelWeightStats::new();
        ms.add(compute_stats("z", &[3], "f32", &[0.0, 0.0, 0.0]), 4);
        ms.add(compute_stats("nz", &[2], "f32", &[1.0, 0.0]), 4);
        assert_eq!(ms.all_zero_tensors().len(), 1);
    }
}

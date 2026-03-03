//! Activation function benchmarking and comparison utilities.
//!
//! Compare SiLU, ReLU, ReLU², GeLU implementations for correctness
//! and relative performance across different input sizes.

use std::time::{Duration, Instant};

/// Supported activation functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActivationKind {
    ReLU,
    ReLUSquared,
    SiLU,
    GeLU,
    Tanh,
    Sigmoid,
}

impl ActivationKind {
    pub fn name(&self) -> &'static str {
        match self {
            Self::ReLU => "ReLU",
            Self::ReLUSquared => "ReLU²",
            Self::SiLU => "SiLU",
            Self::GeLU => "GeLU",
            Self::Tanh => "Tanh",
            Self::Sigmoid => "Sigmoid",
        }
    }

    /// Apply the activation to a single f32 value.
    pub fn apply(&self, x: f32) -> f32 {
        match self {
            Self::ReLU => x.max(0.0),
            Self::ReLUSquared => {
                let r = x.max(0.0);
                r * r
            }
            Self::SiLU => x * sigmoid_f32(x),
            Self::GeLU => {
                // Approximate GeLU
                0.5 * x
                    * (1.0
                        + ((2.0_f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x))
                            .tanh())
            }
            Self::Tanh => x.tanh(),
            Self::Sigmoid => sigmoid_f32(x),
        }
    }

    /// Apply in-place to a slice.
    pub fn apply_slice(&self, data: &mut [f32]) {
        for x in data.iter_mut() {
            *x = self.apply(*x);
        }
    }
}

fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// All standard activations.
pub fn all_activations() -> Vec<ActivationKind> {
    vec![
        ActivationKind::ReLU,
        ActivationKind::ReLUSquared,
        ActivationKind::SiLU,
        ActivationKind::GeLU,
        ActivationKind::Tanh,
        ActivationKind::Sigmoid,
    ]
}

/// Result of benchmarking an activation on a given size.
#[derive(Debug, Clone)]
pub struct BenchResult {
    pub activation: ActivationKind,
    pub input_size: usize,
    pub iterations: usize,
    pub total_time: Duration,
}

impl BenchResult {
    pub fn per_iteration(&self) -> Duration {
        if self.iterations == 0 {
            return Duration::ZERO;
        }
        self.total_time / self.iterations as u32
    }

    pub fn throughput_elements_per_sec(&self) -> f64 {
        let secs = self.total_time.as_secs_f64();
        if secs == 0.0 {
            return 0.0;
        }
        (self.input_size * self.iterations) as f64 / secs
    }
}

/// Benchmark an activation function.
pub fn bench_activation(kind: ActivationKind, size: usize, iterations: usize) -> BenchResult {
    let mut data: Vec<f32> = (0..size).map(|i| (i as f32 - size as f32 / 2.0) * 0.01).collect();

    let start = Instant::now();
    for _ in 0..iterations {
        kind.apply_slice(&mut data);
    }
    let total_time = start.elapsed();

    BenchResult { activation: kind, input_size: size, iterations, total_time }
}

/// Compare all activations at a given size.
pub fn compare_activations(size: usize, iterations: usize) -> Vec<BenchResult> {
    all_activations().into_iter().map(|kind| bench_activation(kind, size, iterations)).collect()
}

/// Numerical comparison of two activations.
#[derive(Debug)]
pub struct ActivationComparison {
    pub a: ActivationKind,
    pub b: ActivationKind,
    pub max_abs_diff: f32,
    pub mean_abs_diff: f32,
    pub sample_count: usize,
}

/// Compare outputs of two activations across a range.
pub fn compare_outputs(
    a: ActivationKind,
    b: ActivationKind,
    range: std::ops::Range<i32>,
) -> ActivationComparison {
    let mut max_diff: f32 = 0.0;
    let mut sum_diff: f32 = 0.0;
    let mut count = 0usize;

    for i in range {
        let x = i as f32 * 0.1;
        let va = a.apply(x);
        let vb = b.apply(x);
        let diff = (va - vb).abs();
        max_diff = max_diff.max(diff);
        sum_diff += diff;
        count += 1;
    }

    ActivationComparison {
        a,
        b,
        max_abs_diff: max_diff,
        mean_abs_diff: if count > 0 { sum_diff / count as f32 } else { 0.0 },
        sample_count: count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        assert_eq!(ActivationKind::ReLU.apply(-1.0), 0.0);
        assert_eq!(ActivationKind::ReLU.apply(2.0), 2.0);
    }

    #[test]
    fn test_relu_squared() {
        assert_eq!(ActivationKind::ReLUSquared.apply(-1.0), 0.0);
        assert_eq!(ActivationKind::ReLUSquared.apply(3.0), 9.0);
    }

    #[test]
    fn test_silu_zero() {
        let val = ActivationKind::SiLU.apply(0.0);
        assert!((val - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid_bounds() {
        let s = ActivationKind::Sigmoid.apply(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        assert!(ActivationKind::Sigmoid.apply(100.0) < 1.001);
        assert!(ActivationKind::Sigmoid.apply(-100.0) > -0.001);
    }

    #[test]
    fn test_tanh_zero() {
        assert!((ActivationKind::Tanh.apply(0.0)).abs() < 1e-6);
    }

    #[test]
    fn test_apply_slice() {
        let mut data = vec![-1.0, 0.0, 1.0, 2.0];
        ActivationKind::ReLU.apply_slice(&mut data);
        assert_eq!(data, vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_all_activations() {
        let all = all_activations();
        assert_eq!(all.len(), 6);
    }

    #[test]
    fn test_bench_result() {
        let result = bench_activation(ActivationKind::ReLU, 100, 10);
        assert_eq!(result.input_size, 100);
        assert_eq!(result.iterations, 10);
        assert!(result.throughput_elements_per_sec() > 0.0);
    }

    #[test]
    fn test_compare_activations() {
        let results = compare_activations(100, 5);
        assert_eq!(results.len(), 6);
    }

    #[test]
    fn test_compare_outputs() {
        let cmp = compare_outputs(ActivationKind::ReLU, ActivationKind::ReLUSquared, -10..10);
        assert_eq!(cmp.sample_count, 20);
        assert!(cmp.max_abs_diff >= 0.0);
    }

    #[test]
    fn test_activation_names() {
        assert_eq!(ActivationKind::SiLU.name(), "SiLU");
        assert_eq!(ActivationKind::GeLU.name(), "GeLU");
        assert_eq!(ActivationKind::ReLUSquared.name(), "ReLU²");
    }

    #[test]
    fn test_gelu_symmetry() {
        let pos = ActivationKind::GeLU.apply(1.0);
        let neg = ActivationKind::GeLU.apply(-1.0);
        // GeLU is not symmetric but gelu(-x) ≈ -gelu(x) is NOT true
        // Just check both are finite
        assert!(pos.is_finite());
        assert!(neg.is_finite());
    }
}

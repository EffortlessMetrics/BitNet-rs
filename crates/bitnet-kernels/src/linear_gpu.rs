//! GPU-accelerated linear projection for transformer inference.
//! Provides a CPU reference implementation with the same layout as the OpenCL kernels.

/// Configuration for a linear projection layer.
#[derive(Debug, Clone)]
pub struct LinearConfig {
    pub in_features: usize,
    pub out_features: usize,
    pub has_bias: bool,
}

/// GPU linear projection with CPU reference implementation.
/// Weight layout: row-major [out_features * in_features] matching the OpenCL kernel.
#[derive(Debug)]
pub struct GpuLinearProj {
    config: LinearConfig,
    weights: Vec<f32>,
    bias: Option<Vec<f32>>,
}

impl GpuLinearProj {
    /// Create a new linear projection.
    ///
    /// # Panics
    /// Panics if weight/bias dimensions do not match config.
    pub fn new(config: LinearConfig, weights: Vec<f32>, bias: Option<Vec<f32>>) -> Self {
        let expected_w = config.out_features * config.in_features;
        assert_eq!(
            weights.len(),
            expected_w,
            "weight length {} != out*in {}",
            weights.len(),
            expected_w,
        );
        if let Some(ref b) = bias {
            assert_eq!(b.len(), config.out_features, "bias length mismatch");
        }
        if config.has_bias {
            assert!(bias.is_some(), "config.has_bias but no bias provided");
        }
        Self { config, weights, bias }
    }

    /// Forward pass for a single input vector (CPU reference path).
    /// input length must equal in_features. Returns vector of length out_features.
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.config.in_features, "input length mismatch");
        let mut output = vec![0.0f32; self.config.out_features];
        for row in 0..self.config.out_features {
            let mut acc = 0.0f32;
            let base = row * self.config.in_features;
            for k in 0..self.config.in_features {
                acc += self.weights[base + k] * input[k];
            }
            if let Some(ref b) = self.bias {
                acc += b[row];
            }
            output[row] = acc;
        }
        output
    }

    /// Batched forward: input [batch_size * in_features] -> output [batch_size * out_features].
    pub fn forward_batched(&self, input: &[f32], batch_size: usize) -> Vec<f32> {
        assert_eq!(input.len(), batch_size * self.config.in_features);
        let mut output = vec![0.0f32; batch_size * self.config.out_features];
        for b in 0..batch_size {
            let inp = &input[b * self.config.in_features..(b + 1) * self.config.in_features];
            let out_slice = self.forward(inp);
            output[b * self.config.out_features..(b + 1) * self.config.out_features]
                .copy_from_slice(&out_slice);
        }
        output
    }

    pub fn in_features(&self) -> usize {
        self.config.in_features
    }
    pub fn out_features(&self) -> usize {
        self.config.out_features
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_proj(n: usize) -> GpuLinearProj {
        // Identity matrix weights, zero bias
        let mut weights = vec![0.0f32; n * n];
        for i in 0..n {
            weights[i * n + i] = 1.0;
        }
        GpuLinearProj::new(
            LinearConfig { in_features: n, out_features: n, has_bias: true },
            weights,
            Some(vec![0.0; n]),
        )
    }

    #[test]
    fn identity_forward() {
        let proj = identity_proj(4);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(proj.forward(&input), input);
    }

    #[test]
    fn forward_with_bias() {
        // 2x3 weights: [[1,0,0],[0,1,0]], bias [10, 20]
        let proj = GpuLinearProj::new(
            LinearConfig { in_features: 3, out_features: 2, has_bias: true },
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            Some(vec![10.0, 20.0]),
        );
        let out = proj.forward(&[5.0, 7.0, 9.0]);
        assert_eq!(out, vec![15.0, 27.0]);
    }

    #[test]
    fn forward_no_bias() {
        let proj = GpuLinearProj::new(
            LinearConfig { in_features: 2, out_features: 2, has_bias: false },
            vec![2.0, 0.0, 0.0, 3.0],
            None,
        );
        assert_eq!(proj.forward(&[4.0, 5.0]), vec![8.0, 15.0]);
    }

    #[test]
    fn batched_forward() {
        let proj = identity_proj(2);
        let input = vec![1.0, 2.0, 3.0, 4.0]; // batch=2
        let out = proj.forward_batched(&input, 2);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn projection_reduces_dimension() {
        // 1x4 weights: sum of inputs
        let proj = GpuLinearProj::new(
            LinearConfig { in_features: 4, out_features: 1, has_bias: false },
            vec![1.0, 1.0, 1.0, 1.0],
            None,
        );
        assert_eq!(proj.forward(&[1.0, 2.0, 3.0, 4.0]), vec![10.0]);
    }

    #[test]
    fn projection_expands_dimension() {
        // 3x1 weights: replicate input
        let proj = GpuLinearProj::new(
            LinearConfig { in_features: 1, out_features: 3, has_bias: false },
            vec![1.0, 2.0, 3.0],
            None,
        );
        assert_eq!(proj.forward(&[5.0]), vec![5.0, 10.0, 15.0]);
    }

    #[test]
    #[should_panic(expected = "weight length")]
    fn wrong_weight_dims() {
        GpuLinearProj::new(
            LinearConfig { in_features: 3, out_features: 2, has_bias: false },
            vec![0.0; 5],
            None,
        );
    }

    #[test]
    #[should_panic(expected = "input length")]
    fn wrong_input_length() {
        let proj = identity_proj(4);
        proj.forward(&[1.0, 2.0]);
    }

    #[test]
    fn accessors() {
        let proj = GpuLinearProj::new(
            LinearConfig { in_features: 64, out_features: 128, has_bias: false },
            vec![0.0; 64 * 128],
            None,
        );
        assert_eq!(proj.in_features(), 64);
        assert_eq!(proj.out_features(), 128);
    }

    #[test]
    fn large_matmul_correctness() {
        // All-ones weight (4x8) times [1..=8] should give [36.0; 4]
        let proj = GpuLinearProj::new(
            LinearConfig { in_features: 8, out_features: 4, has_bias: false },
            vec![1.0; 32],
            None,
        );
        let input: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let out = proj.forward(&input);
        assert_eq!(out, vec![36.0, 36.0, 36.0, 36.0]);
    }
}

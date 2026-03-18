//! CPU fallback for linear projection (`y = x · Wᵀ + bias`).
//!
//! This module provides the shared [`LinearConfig`], buffer validation,
//! a naive O(batch × out × in) reference implementation ([`linear_cpu`]),
//! and a unified [`linear_forward`] dispatcher that prefers the GPU path
//! (see [`crate::cuda::linear`]) when available.

use bitnet_common::{BitNetError, KernelError, Result};

// ── Launch configuration ──────────────────────────────────────────────

/// Configuration for a linear projection layer.
///
/// Describes the shape `y = x · Wᵀ + bias` where:
///
/// - `x` is `[batch_size, in_features]`
/// - `W` is `[out_features, in_features]`
/// - `bias` is an optional `[out_features]` vector
/// - `y` is `[batch_size, out_features]`
#[derive(Debug, Clone)]
pub struct LinearConfig {
    /// Number of input features (inner / reduction dimension).
    pub in_features: usize,
    /// Number of output features.
    pub out_features: usize,
    /// Batch count (number of input rows).
    pub batch_size: usize,
    /// Whether a bias vector is present.
    pub has_bias: bool,
    /// CUDA tile size in the M (batch) dimension.
    pub tile_m: u32,
    /// CUDA tile size in the N (out_features) dimension.
    pub tile_n: u32,
    /// CUDA tile size in the K (in_features) dimension.
    pub tile_k: u32,
    /// Number of threads per block.
    pub threads_per_block: u32,
    /// Bytes of dynamic shared memory.
    pub shared_mem_bytes: u32,
}

impl Default for LinearConfig {
    fn default() -> Self {
        Self {
            in_features: 1,
            out_features: 1,
            batch_size: 1,
            has_bias: false,
            tile_m: 32,
            tile_n: 32,
            tile_k: 32,
            threads_per_block: 256,
            shared_mem_bytes: 8192,
        }
    }
}

impl LinearConfig {
    /// Create a config for the given layer dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error if any dimension is zero.
    pub fn new(batch_size: usize, in_features: usize, out_features: usize) -> Result<Self> {
        if batch_size == 0 || in_features == 0 || out_features == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "linear dimensions must be non-zero: \
                     batch_size={batch_size}, in_features={in_features}, \
                     out_features={out_features}"
                ),
            }
            .into());
        }

        let tile_m = 32u32;
        let tile_n = 32u32;
        let tile_k = 32u32;
        let shared = (tile_m * tile_k + tile_k * tile_n) * 4;

        Ok(Self {
            batch_size,
            in_features,
            out_features,
            tile_m,
            tile_n,
            tile_k,
            shared_mem_bytes: shared,
            ..Self::default()
        })
    }

    /// Enable or disable bias.
    pub fn with_bias(mut self, has_bias: bool) -> Self {
        self.has_bias = has_bias;
        self
    }

    /// Compute the CUDA grid dimensions `(grid_x, grid_y, 1)`.
    pub fn grid_dim(&self) -> (u32, u32, u32) {
        let grid_x = (self.out_features as u32).div_ceil(self.tile_n);
        let grid_y = (self.batch_size as u32).div_ceil(self.tile_m);
        (grid_x, grid_y, 1)
    }

    /// Compute the CUDA block dimensions.
    pub fn block_dim(&self) -> (u32, u32, u32) {
        (self.threads_per_block, 1, 1)
    }
}

// ── Validation ────────────────────────────────────────────────────────

fn validate_linear_buffers(
    x: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &[f32],
    config: &LinearConfig,
) -> Result<()> {
    let x_required = config.batch_size * config.in_features;
    let w_required = config.out_features * config.in_features;
    let out_required = config.batch_size * config.out_features;

    if x.len() < x_required {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("x buffer too small: expected >= {x_required}, got {}", x.len()),
        }));
    }
    if weight.len() < w_required {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!(
                "weight buffer too small: expected >= {w_required}, got {}",
                weight.len()
            ),
        }));
    }
    if output.len() < out_required {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!(
                "output buffer too small: expected >= {out_required}, got {}",
                output.len()
            ),
        }));
    }
    if let Some(b) = bias
        && b.len() < config.out_features
    {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!(
                "bias buffer too small: expected >= {}, got {}",
                config.out_features,
                b.len()
            ),
        }));
    }
    Ok(())
}

// ── CPU fallback ──────────────────────────────────────────────────────

/// Naive linear projection (CPU fallback).
///
/// Computes `y = x · Wᵀ + bias` for each row in the batch.
///
/// # Layout
///
/// - `x`: row-major `[batch_size, in_features]` f32
/// - `weight`: row-major `[out_features, in_features]` f32
/// - `bias`: optional `[out_features]` f32
/// - `output`: row-major `[batch_size, out_features]` f32
///
/// # Errors
///
/// Returns an error if buffer sizes are inconsistent with the config.
pub fn linear_cpu(
    x: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    config: &LinearConfig,
) -> Result<()> {
    validate_linear_buffers(x, weight, bias, output, config)?;

    let batch = config.batch_size;
    let in_f = config.in_features;
    let out_f = config.out_features;

    for b in 0..batch {
        let x_off = b * in_f;
        let o_off = b * out_f;

        for j in 0..out_f {
            let mut acc = 0.0f32;
            let w_off = j * in_f;
            for k in 0..in_f {
                acc += x[x_off + k] * weight[w_off + k];
            }
            if let Some(bias) = bias {
                acc += bias[j];
            }
            output[o_off + j] = acc;
        }
    }
    Ok(())
}

// ── Unified dispatch ──────────────────────────────────────────────────

/// Linear projection with automatic dispatch: GPU if available, else CPU
/// fallback.
pub fn linear_forward(
    x: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    config: &LinearConfig,
) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime()
            && let Ok(()) = crate::cuda::linear::launch_linear(x, weight, bias, output, config)
        {
            return Ok(());
        }
    }
    linear_cpu(x, weight, bias, output, config)
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ────────────────────────────────────────────────────

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol, "mismatch at {i}: {x} vs {y} (tol {tol})");
        }
    }

    // ── config tests ──────────────────────────────────────────────

    #[test]
    fn test_linear_config_defaults() {
        let cfg = LinearConfig::default();
        assert_eq!(cfg.in_features, 1);
        assert_eq!(cfg.out_features, 1);
        assert_eq!(cfg.batch_size, 1);
        assert!(!cfg.has_bias);
    }

    #[test]
    fn test_linear_config_new() {
        let cfg = LinearConfig::new(4, 8, 16).unwrap();
        assert_eq!(cfg.batch_size, 4);
        assert_eq!(cfg.in_features, 8);
        assert_eq!(cfg.out_features, 16);
    }

    #[test]
    fn test_linear_config_rejects_zero_dims() {
        assert!(LinearConfig::new(0, 8, 16).is_err());
        assert!(LinearConfig::new(4, 0, 16).is_err());
        assert!(LinearConfig::new(4, 8, 0).is_err());
    }

    #[test]
    fn test_linear_config_with_bias() {
        let cfg = LinearConfig::new(2, 3, 4).unwrap().with_bias(true);
        assert!(cfg.has_bias);
    }

    #[test]
    fn test_linear_config_grid_dim() {
        let cfg = LinearConfig::new(64, 128, 96).unwrap();
        let (gx, gy, gz) = cfg.grid_dim();
        assert_eq!(gx, 3); // ceil(96/32)
        assert_eq!(gy, 2); // ceil(64/32)
        assert_eq!(gz, 1);
    }

    // ── known input/output: 2×3 → 2×4 ────────────────────────────

    #[test]
    fn test_linear_known_2x3_to_2x4_no_bias() {
        // x: [2, 3], W: [4, 3]
        // y = x · W^T → [2, 4]
        #[rustfmt::skip]
        let x = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ];
        #[rustfmt::skip]
        let weight = vec![
            1.0, 0.0, 0.0, // W row 0
            0.0, 1.0, 0.0, // W row 1
            0.0, 0.0, 1.0, // W row 2
            1.0, 1.0, 1.0, // W row 3
        ];
        // y[0] = [x[0]·W[0], x[0]·W[1], x[0]·W[2], x[0]·W[3]]
        //      = [1, 2, 3, 6]
        // y[1] = [4, 5, 6, 15]
        let expected = vec![1.0, 2.0, 3.0, 6.0, 4.0, 5.0, 6.0, 15.0];

        let cfg = LinearConfig::new(2, 3, 4).unwrap();
        let mut out = [0.0f32; 8];
        linear_cpu(&x, &weight, None, &mut out, &cfg).unwrap();
        assert_close(&out, &expected, 1e-6);
    }

    #[test]
    fn test_linear_known_2x3_to_2x4_with_bias() {
        #[rustfmt::skip]
        let x = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ];
        #[rustfmt::skip]
        let weight = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            1.0, 1.0, 1.0,
        ];
        let bias = vec![10.0, 20.0, 30.0, 40.0];
        // y[0] = [1+10, 2+20, 3+30, 6+40] = [11, 22, 33, 46]
        // y[1] = [4+10, 5+20, 6+30, 15+40] = [14, 25, 36, 55]
        let expected = vec![11.0, 22.0, 33.0, 46.0, 14.0, 25.0, 36.0, 55.0];

        let cfg = LinearConfig::new(2, 3, 4).unwrap().with_bias(true);
        let mut out = [0.0f32; 8];
        linear_cpu(&x, &weight, Some(&bias), &mut out, &cfg).unwrap();
        assert_close(&out, &expected, 1e-6);
    }

    // ── batch dimension handling ──────────────────────────────────

    #[test]
    fn test_linear_single_batch() {
        let x = vec![1.0, 2.0];
        let weight = vec![3.0, 4.0]; // [1, 2] → [1, 1]
        let cfg = LinearConfig::new(1, 2, 1).unwrap();
        let mut out = [0.0f32; 1];
        linear_cpu(&x, &weight, None, &mut out, &cfg).unwrap();
        // 1*3 + 2*4 = 11
        assert_close(&out, &[11.0], 1e-6);
    }

    #[test]
    fn test_linear_batch_4() {
        // 4 rows, 2 in, 3 out
        #[rustfmt::skip]
        let x = vec![
            1.0, 0.0,
            0.0, 1.0,
            1.0, 1.0,
            2.0, 3.0,
        ];
        #[rustfmt::skip]
        let weight = vec![
            1.0, 0.0, // W[0]: select in[0]
            0.0, 1.0, // W[1]: select in[1]
            1.0, 1.0, // W[2]: sum
        ];
        // y[0] = [1, 0, 1]
        // y[1] = [0, 1, 1]
        // y[2] = [1, 1, 2]
        // y[3] = [2, 3, 5]
        #[rustfmt::skip]
        let expected = vec![
            1.0, 0.0, 1.0,
            0.0, 1.0, 1.0,
            1.0, 1.0, 2.0,
            2.0, 3.0, 5.0,
        ];

        let cfg = LinearConfig::new(4, 2, 3).unwrap();
        let mut out = [0.0f32; 12];
        linear_cpu(&x, &weight, None, &mut out, &cfg).unwrap();
        assert_close(&out, &expected, 1e-6);
    }

    // ── zero bias / no bias ──────────────────────────────────────

    #[test]
    fn test_linear_zero_bias_same_as_no_bias() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weight = vec![1.0, 1.0, 1.0, 0.5, 0.5, 0.5];
        let zero_bias = vec![0.0, 0.0];

        let cfg_no = LinearConfig::new(2, 3, 2).unwrap();
        let cfg_zero = LinearConfig::new(2, 3, 2).unwrap().with_bias(true);

        let mut out_no = [0.0f32; 4];
        let mut out_zero = [0.0f32; 4];

        linear_cpu(&x, &weight, None, &mut out_no, &cfg_no).unwrap();
        linear_cpu(&x, &weight, Some(&zero_bias), &mut out_zero, &cfg_zero).unwrap();
        assert_close(&out_no, &out_zero, 1e-6);
    }

    #[test]
    fn test_linear_bias_only_adds_to_output() {
        // Zero input + bias → output == bias broadcast
        let x = [0.0f32; 6]; // [2, 3]
        let weight = [1.0f32; 6]; // [2, 3]
        let bias = vec![7.0, 11.0];

        let cfg = LinearConfig::new(2, 3, 2).unwrap().with_bias(true);
        let mut out = [0.0f32; 4];
        linear_cpu(&x, &weight, Some(&bias), &mut out, &cfg).unwrap();
        // 0·W + bias → [7, 11, 7, 11]
        assert_close(&out, &[7.0, 11.0, 7.0, 11.0], 1e-6);
    }

    // ── identity-like projection ──────────────────────────────────

    #[test]
    fn test_linear_identity_projection() {
        // W = I₃, no bias → y = x
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
        #[rustfmt::skip]
        let weight = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let cfg = LinearConfig::new(2, 3, 3).unwrap();
        let mut out = [0.0f32; 6];
        linear_cpu(&x, &weight, None, &mut out, &cfg).unwrap();
        assert_close(&out, &x, 1e-6);
    }

    #[test]
    fn test_linear_identity_projection_with_bias() {
        let x = vec![1.0, 2.0, 3.0]; // [1, 3]
        #[rustfmt::skip]
        let weight = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let bias = vec![10.0, 20.0, 30.0];
        let cfg = LinearConfig::new(1, 3, 3).unwrap().with_bias(true);
        let mut out = [0.0f32; 3];
        linear_cpu(&x, &weight, Some(&bias), &mut out, &cfg).unwrap();
        assert_close(&out, &[11.0, 22.0, 33.0], 1e-6);
    }

    // ── buffer validation ─────────────────────────────────────────

    #[test]
    fn test_linear_x_buffer_too_small() {
        let cfg = LinearConfig::new(2, 4, 3).unwrap();
        let x = [1.0f32; 4]; // need 8
        let w = [1.0f32; 12];
        let mut out = [0.0f32; 6];
        assert!(linear_cpu(&x, &w, None, &mut out, &cfg).is_err());
    }

    #[test]
    fn test_linear_weight_buffer_too_small() {
        let cfg = LinearConfig::new(2, 4, 3).unwrap();
        let x = [1.0f32; 8];
        let w = [1.0f32; 8]; // need 12
        let mut out = [0.0f32; 6];
        assert!(linear_cpu(&x, &w, None, &mut out, &cfg).is_err());
    }

    #[test]
    fn test_linear_output_buffer_too_small() {
        let cfg = LinearConfig::new(2, 4, 3).unwrap();
        let x = [1.0f32; 8];
        let w = [1.0f32; 12];
        let mut out = [0.0f32; 3]; // need 6
        assert!(linear_cpu(&x, &w, None, &mut out, &cfg).is_err());
    }

    #[test]
    fn test_linear_bias_buffer_too_small() {
        let cfg = LinearConfig::new(2, 4, 3).unwrap().with_bias(true);
        let x = [1.0f32; 8];
        let w = [1.0f32; 12];
        let bias = [1.0f32; 2]; // need 3
        let mut out = [0.0f32; 6];
        assert!(linear_cpu(&x, &w, Some(&bias), &mut out, &cfg).is_err());
    }

    // ── unified dispatch ──────────────────────────────────────────

    #[test]
    fn test_linear_forward_falls_back_to_cpu() {
        let x = vec![1.0, 2.0, 3.0];
        #[rustfmt::skip]
        let weight = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let cfg = LinearConfig::new(1, 3, 3).unwrap();
        let mut out = [0.0f32; 3];
        linear_forward(&x, &weight, None, &mut out, &cfg).unwrap();
        assert_close(&out, &x, 1e-6);
    }

    // ── property: output shape invariant ──────────────────────────

    #[test]
    fn test_linear_output_shape_invariant() {
        for (batch, inf, outf) in [(1, 1, 1), (1, 8, 4), (4, 3, 7), (16, 32, 16)] {
            let x = vec![1.0f32; batch * inf];
            let w = vec![0.5f32; outf * inf];
            let cfg = LinearConfig::new(batch, inf, outf).unwrap();
            let mut out = vec![0.0f32; batch * outf];
            linear_cpu(&x, &w, None, &mut out, &cfg).unwrap();
            assert_eq!(out.len(), batch * outf);
        }
    }

    // ── property: linearity f(ax) = a·f(x) for bias=0 ────────────

    #[test]
    fn test_linear_linearity_scalar_multiple() {
        let scale = 3.5f32;
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
        let x_scaled: Vec<f32> = x.iter().map(|v| v * scale).collect();
        #[rustfmt::skip]
        let weight = vec![
            0.1, 0.2, 0.3,
            0.4, 0.5, 0.6,
        ];

        let cfg = LinearConfig::new(2, 3, 2).unwrap();

        let mut out_x = [0.0f32; 4];
        let mut out_sx = [0.0f32; 4];

        linear_cpu(&x, &weight, None, &mut out_x, &cfg).unwrap();
        linear_cpu(&x_scaled, &weight, None, &mut out_sx, &cfg).unwrap();

        // f(a·x) should equal a·f(x)
        let scaled_out: Vec<f32> = out_x.iter().map(|v| v * scale).collect();
        assert_close(&out_sx, &scaled_out, 1e-4);
    }

    // ── property: additivity f(a+b) = f(a) + f(b) for bias=0 ────

    #[test]
    fn test_linear_additivity() {
        let a = vec![1.0, 2.0, 3.0]; // [1, 3]
        let b = vec![4.0, 5.0, 6.0]; // [1, 3]
        let ab: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
        let weight = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]; // [2, 3]

        let cfg = LinearConfig::new(1, 3, 2).unwrap();

        let mut out_a = [0.0f32; 2];
        let mut out_b = [0.0f32; 2];
        let mut out_ab = [0.0f32; 2];

        linear_cpu(&a, &weight, None, &mut out_a, &cfg).unwrap();
        linear_cpu(&b, &weight, None, &mut out_b, &cfg).unwrap();
        linear_cpu(&ab, &weight, None, &mut out_ab, &cfg).unwrap();

        let sum: Vec<f32> = out_a.iter().zip(out_b.iter()).map(|(x, y)| x + y).collect();
        assert_close(&out_ab, &sum, 1e-4);
    }

    // ── 1×1 edge case ─────────────────────────────────────────────

    #[test]
    fn test_linear_1x1() {
        let x = vec![5.0f32];
        let w = vec![3.0f32];
        let cfg = LinearConfig::new(1, 1, 1).unwrap();
        let mut out = [0.0f32; 1];
        linear_cpu(&x, &w, None, &mut out, &cfg).unwrap();
        assert_close(&out, &[15.0], 1e-6);
    }

    #[test]
    fn test_linear_1x1_with_bias() {
        let x = vec![5.0f32];
        let w = vec![3.0f32];
        let bias = vec![2.0f32];
        let cfg = LinearConfig::new(1, 1, 1).unwrap().with_bias(true);
        let mut out = [0.0f32; 1];
        linear_cpu(&x, &w, Some(&bias), &mut out, &cfg).unwrap();
        assert_close(&out, &[17.0], 1e-6);
    }

    // ── large batch stress ────────────────────────────────────────

    #[test]
    fn test_linear_large_batch() {
        let (batch, inf, outf) = (64, 16, 8);
        let x: Vec<f32> = (0..batch * inf).map(|i| (i as f32 * 0.01).sin()).collect();
        let w: Vec<f32> = (0..outf * inf).map(|i| (i as f32 * 0.02).cos()).collect();
        let cfg = LinearConfig::new(batch, inf, outf).unwrap();
        let mut out = vec![0.0f32; batch * outf];
        linear_cpu(&x, &w, None, &mut out, &cfg).unwrap();

        // Verify each row independently via naive computation.
        for b in 0..batch {
            for j in 0..outf {
                let mut expected = 0.0f32;
                for k in 0..inf {
                    expected += x[b * inf + k] * w[j * inf + k];
                }
                let actual = out[b * outf + j];
                assert!(
                    (actual - expected).abs() < 1e-3,
                    "mismatch at batch={b}, out={j}: {actual} vs {expected}"
                );
            }
        }
    }
}

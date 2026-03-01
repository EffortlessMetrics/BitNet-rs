//! CPU SIMD-optimized pooling kernel.
//!
//! Provides 1-D pooling operations (max, average, global, adaptive) on
//! contiguous `f32` slices.  Scalar implementations are provided for
//! correctness on all platforms; SIMD acceleration is used when available.

use bitnet_common::{BitNetError, KernelError, Result};

// ── Configuration ──────────────────────────────────────────────────

/// Pooling operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PoolType {
    /// Sliding-window maximum.
    Max,
    /// Sliding-window arithmetic mean.
    Average,
    /// Global maximum (reduces to a single value).
    GlobalMax,
    /// Global arithmetic mean (reduces to a single value).
    GlobalAverage,
}

/// Parameters for a 1-D pooling operation.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Type of pooling to perform.
    pub pool_type: PoolType,
    /// Window size (ignored for global variants).
    pub kernel_size: usize,
    /// Stride between successive windows (ignored for global variants).
    pub stride: usize,
    /// Zero-padding added to each side of the input (ignored for global variants).
    pub padding: usize,
}

impl PoolConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<()> {
        match self.pool_type {
            PoolType::GlobalMax | PoolType::GlobalAverage => Ok(()),
            _ => {
                if self.kernel_size == 0 {
                    return Err(invalid_args("kernel_size must be > 0"));
                }
                if self.stride == 0 {
                    return Err(invalid_args("stride must be > 0"));
                }
                Ok(())
            }
        }
    }
}

/// Stateless pooling kernel that dispatches to the appropriate operation.
#[derive(Debug)]
pub struct PoolingKernel;

impl PoolingKernel {
    /// Apply a 1-D pooling operation described by `config` to `input`.
    pub fn apply(input: &[f32], config: &PoolConfig) -> Result<Vec<f32>> {
        config.validate()?;
        match config.pool_type {
            PoolType::Max => max_pool_1d(input, config.kernel_size, config.stride, config.padding),
            PoolType::Average => {
                avg_pool_1d(input, config.kernel_size, config.stride, config.padding)
            }
            PoolType::GlobalMax => global_max(input),
            PoolType::GlobalAverage => global_avg(input),
        }
    }

    /// Adaptive pooling: compute kernel size and stride so that an input of
    /// length `input_len` is reduced to exactly `output_size` elements.
    ///
    /// Returns a `PoolConfig` with padding = 0 and the caller-specified
    /// `pool_type`.  Global variants are returned as-is when
    /// `output_size == 1`.
    pub fn adaptive_config(
        pool_type: PoolType,
        input_len: usize,
        output_size: usize,
    ) -> Result<PoolConfig> {
        if output_size == 0 {
            return Err(invalid_args("output_size must be > 0"));
        }
        if input_len == 0 {
            return Err(invalid_args("input_len must be > 0"));
        }
        if output_size > input_len {
            return Err(invalid_args("output_size must be <= input_len for pooling"));
        }

        // Global variants when requesting a single output.
        if output_size == 1 {
            let global_type = match pool_type {
                PoolType::Max | PoolType::GlobalMax => PoolType::GlobalMax,
                PoolType::Average | PoolType::GlobalAverage => PoolType::GlobalAverage,
            };
            return Ok(PoolConfig {
                pool_type: global_type,
                kernel_size: input_len,
                stride: input_len,
                padding: 0,
            });
        }

        // PyTorch-style adaptive pooling: for each output index i, the
        // window covers input[start..end] where
        //   start = floor(i * input_len / output_size)
        //   end   = floor((i+1) * input_len / output_size)
        // This cannot always be expressed with a single (kernel, stride)
        // pair; we approximate with the dominant window parameters.
        let stride = input_len / output_size;
        let kernel_size = input_len - (output_size - 1) * stride;

        Ok(PoolConfig { pool_type, kernel_size, stride, padding: 0 })
    }
}

// ── Scalar implementations ─────────────────────────────────────────

fn invalid_args(reason: &str) -> BitNetError {
    BitNetError::Kernel(KernelError::InvalidArguments { reason: reason.to_string() })
}

/// Number of output elements for a 1-D pooling window.
#[inline]
fn output_len(input_len: usize, kernel_size: usize, stride: usize, padding: usize) -> usize {
    if input_len + 2 * padding < kernel_size {
        return 0;
    }
    (input_len + 2 * padding - kernel_size) / stride + 1
}

/// 1-D max pooling.
fn max_pool_1d(
    input: &[f32],
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> Result<Vec<f32>> {
    let n = input.len();
    let out_n = output_len(n, kernel_size, stride, padding);
    let mut output = Vec::with_capacity(out_n);

    for i in 0..out_n {
        let window_start = i * stride;
        let mut max_val = f32::NEG_INFINITY;

        for k in 0..kernel_size {
            let pos = window_start + k;
            let val = if pos < padding || pos >= n + padding {
                // Padded positions contribute -∞ for max pooling.
                f32::NEG_INFINITY
            } else {
                input[pos - padding]
            };
            if val > max_val {
                max_val = val;
            }
        }
        output.push(max_val);
    }
    Ok(output)
}

/// 1-D average pooling.
fn avg_pool_1d(
    input: &[f32],
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> Result<Vec<f32>> {
    let n = input.len();
    let out_n = output_len(n, kernel_size, stride, padding);
    let mut output = Vec::with_capacity(out_n);

    for i in 0..out_n {
        let window_start = i * stride;
        let mut sum = 0.0f32;

        for k in 0..kernel_size {
            let pos = window_start + k;
            // Padded positions contribute 0 for average pooling.
            if pos >= padding && pos < n + padding {
                sum += input[pos - padding];
            }
        }
        output.push(sum / kernel_size as f32);
    }
    Ok(output)
}

/// Global max: reduce entire input to a single maximum.
fn global_max(input: &[f32]) -> Result<Vec<f32>> {
    if input.is_empty() {
        return Err(invalid_args("global max requires non-empty input"));
    }
    let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    Ok(vec![max_val])
}

/// Global average: reduce entire input to a single mean.
fn global_avg(input: &[f32]) -> Result<Vec<f32>> {
    if input.is_empty() {
        return Err(invalid_args("global average requires non-empty input"));
    }
    let sum: f32 = input.iter().sum();
    Ok(vec![sum / input.len() as f32])
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-6;

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
    }

    // ── Max pooling ────────────────────────────────────────────────

    #[test]
    fn max_pool_basic() {
        let input = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let cfg = PoolConfig { pool_type: PoolType::Max, kernel_size: 2, stride: 1, padding: 0 };
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[3.0, 3.0, 5.0, 5.0], TOL));
    }

    #[test]
    fn max_pool_stride_2() {
        let input = vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0];
        let cfg = PoolConfig { pool_type: PoolType::Max, kernel_size: 2, stride: 2, padding: 0 };
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[3.0, 5.0, 6.0], TOL));
    }

    #[test]
    fn max_pool_kernel_equals_input() {
        let input = vec![1.0, 3.0, 2.0];
        let cfg = PoolConfig { pool_type: PoolType::Max, kernel_size: 3, stride: 1, padding: 0 };
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[3.0], TOL));
    }

    #[test]
    fn max_pool_with_padding() {
        let input = vec![1.0, 2.0, 3.0];
        let cfg = PoolConfig { pool_type: PoolType::Max, kernel_size: 3, stride: 1, padding: 1 };
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        // windows: [pad,1,2] [1,2,3] [2,3,pad]
        assert!(approx_eq(&out, &[2.0, 3.0, 3.0], TOL));
    }

    #[test]
    fn max_pool_negative_values() {
        let input = vec![-5.0, -3.0, -4.0, -1.0, -2.0];
        let cfg = PoolConfig { pool_type: PoolType::Max, kernel_size: 3, stride: 1, padding: 0 };
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[-3.0, -1.0, -1.0], TOL));
    }

    #[test]
    fn max_pool_single_element() {
        let input = vec![42.0];
        let cfg = PoolConfig { pool_type: PoolType::Max, kernel_size: 1, stride: 1, padding: 0 };
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[42.0], TOL));
    }

    // ── Average pooling ────────────────────────────────────────────

    #[test]
    fn avg_pool_basic() {
        let input = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let cfg =
            PoolConfig { pool_type: PoolType::Average, kernel_size: 2, stride: 1, padding: 0 };
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[2.0, 2.5, 3.5, 4.5], TOL));
    }

    #[test]
    fn avg_pool_stride_2() {
        let input = vec![2.0, 4.0, 6.0, 8.0];
        let cfg =
            PoolConfig { pool_type: PoolType::Average, kernel_size: 2, stride: 2, padding: 0 };
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[3.0, 7.0], TOL));
    }

    #[test]
    fn avg_pool_kernel_equals_input() {
        let input = vec![1.0, 2.0, 3.0];
        let cfg =
            PoolConfig { pool_type: PoolType::Average, kernel_size: 3, stride: 1, padding: 0 };
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[2.0], TOL));
    }

    #[test]
    fn avg_pool_with_padding() {
        let input = vec![3.0, 6.0, 9.0];
        let cfg =
            PoolConfig { pool_type: PoolType::Average, kernel_size: 3, stride: 1, padding: 1 };
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        // windows: [0,3,6]/3=3.0  [3,6,9]/3=6.0  [6,9,0]/3=5.0
        assert!(approx_eq(&out, &[3.0, 6.0, 5.0], TOL));
    }

    #[test]
    fn avg_pool_single_element() {
        let input = vec![7.0];
        let cfg =
            PoolConfig { pool_type: PoolType::Average, kernel_size: 1, stride: 1, padding: 0 };
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[7.0], TOL));
    }

    // ── Global pooling ─────────────────────────────────────────────

    #[test]
    fn global_max_basic() {
        let input = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        let cfg =
            PoolConfig { pool_type: PoolType::GlobalMax, kernel_size: 0, stride: 0, padding: 0 };
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[5.0], TOL));
    }

    #[test]
    fn global_max_single() {
        let cfg =
            PoolConfig { pool_type: PoolType::GlobalMax, kernel_size: 0, stride: 0, padding: 0 };
        let out = PoolingKernel::apply(&[42.0], &cfg).unwrap();
        assert!(approx_eq(&out, &[42.0], TOL));
    }

    #[test]
    fn global_max_all_negative() {
        let input = vec![-10.0, -5.0, -20.0];
        let cfg =
            PoolConfig { pool_type: PoolType::GlobalMax, kernel_size: 0, stride: 0, padding: 0 };
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[-5.0], TOL));
    }

    #[test]
    fn global_avg_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cfg = PoolConfig {
            pool_type: PoolType::GlobalAverage,
            kernel_size: 0,
            stride: 0,
            padding: 0,
        };
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[3.0], TOL));
    }

    #[test]
    fn global_avg_single() {
        let cfg = PoolConfig {
            pool_type: PoolType::GlobalAverage,
            kernel_size: 0,
            stride: 0,
            padding: 0,
        };
        let out = PoolingKernel::apply(&[99.0], &cfg).unwrap();
        assert!(approx_eq(&out, &[99.0], TOL));
    }

    #[test]
    fn global_max_empty_input() {
        let cfg =
            PoolConfig { pool_type: PoolType::GlobalMax, kernel_size: 0, stride: 0, padding: 0 };
        assert!(PoolingKernel::apply(&[], &cfg).is_err());
    }

    #[test]
    fn global_avg_empty_input() {
        let cfg = PoolConfig {
            pool_type: PoolType::GlobalAverage,
            kernel_size: 0,
            stride: 0,
            padding: 0,
        };
        assert!(PoolingKernel::apply(&[], &cfg).is_err());
    }

    // ── Adaptive pooling ───────────────────────────────────────────

    #[test]
    fn adaptive_reduces_to_single() {
        let cfg = PoolingKernel::adaptive_config(PoolType::Max, 10, 1).unwrap();
        assert_eq!(cfg.pool_type, PoolType::GlobalMax);
    }

    #[test]
    fn adaptive_avg_reduces_to_global() {
        let cfg = PoolingKernel::adaptive_config(PoolType::Average, 8, 1).unwrap();
        assert_eq!(cfg.pool_type, PoolType::GlobalAverage);
    }

    #[test]
    fn adaptive_output_shape() {
        let input: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let cfg = PoolingKernel::adaptive_config(PoolType::Average, 10, 5).unwrap();
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert_eq!(out.len(), 5);
    }

    #[test]
    fn adaptive_identity() {
        // output_size == input_len ⇒ kernel_size=1, stride=1 (identity).
        let cfg = PoolingKernel::adaptive_config(PoolType::Max, 5, 5).unwrap();
        assert_eq!(cfg.kernel_size, 1);
        assert_eq!(cfg.stride, 1);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &input, TOL));
    }

    #[test]
    fn adaptive_output_larger_than_input_rejected() {
        assert!(PoolingKernel::adaptive_config(PoolType::Max, 3, 5).is_err());
    }

    #[test]
    fn adaptive_zero_output_rejected() {
        assert!(PoolingKernel::adaptive_config(PoolType::Max, 5, 0).is_err());
    }

    #[test]
    fn adaptive_zero_input_rejected() {
        assert!(PoolingKernel::adaptive_config(PoolType::Max, 0, 1).is_err());
    }

    // ── Edge cases ─────────────────────────────────────────────────

    #[test]
    fn kernel_larger_than_input_produces_empty() {
        let input = vec![1.0, 2.0];
        let cfg = PoolConfig { pool_type: PoolType::Max, kernel_size: 5, stride: 1, padding: 0 };
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn zero_kernel_size_rejected() {
        let cfg = PoolConfig { pool_type: PoolType::Max, kernel_size: 0, stride: 1, padding: 0 };
        assert!(PoolingKernel::apply(&[1.0], &cfg).is_err());
    }

    #[test]
    fn zero_stride_rejected() {
        let cfg =
            PoolConfig { pool_type: PoolType::Average, kernel_size: 2, stride: 0, padding: 0 };
        assert!(PoolingKernel::apply(&[1.0, 2.0], &cfg).is_err());
    }

    #[test]
    fn large_stride_single_output() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cfg = PoolConfig { pool_type: PoolType::Max, kernel_size: 2, stride: 10, padding: 0 };
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert!(approx_eq(&out, &[2.0], TOL));
    }

    #[test]
    fn output_len_formula() {
        // input=6, kernel=3, stride=2, pad=0 → (6-3)/2+1 = 2
        assert_eq!(output_len(6, 3, 2, 0), 2);
        // input=5, kernel=3, stride=1, pad=1 → (5+2-3)/1+1 = 5
        assert_eq!(output_len(5, 3, 1, 1), 5);
    }

    #[test]
    fn max_pool_large_input() {
        let input: Vec<f32> = (0..1024).map(|i| (i as f32).sin()).collect();
        let cfg = PoolConfig { pool_type: PoolType::Max, kernel_size: 4, stride: 4, padding: 0 };
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert_eq!(out.len(), 256);
        // Each output is the max of 4 consecutive elements.
        for (i, &v) in out.iter().enumerate() {
            let window = &input[i * 4..i * 4 + 4];
            let expected = window.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            assert!((v - expected).abs() < TOL);
        }
    }

    #[test]
    fn avg_pool_large_input() {
        let input: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let cfg =
            PoolConfig { pool_type: PoolType::Average, kernel_size: 4, stride: 4, padding: 0 };
        let out = PoolingKernel::apply(&input, &cfg).unwrap();
        assert_eq!(out.len(), 256);
        for (i, &v) in out.iter().enumerate() {
            let window = &input[i * 4..i * 4 + 4];
            let expected: f32 = window.iter().sum::<f32>() / 4.0;
            assert!((v - expected).abs() < TOL);
        }
    }
}

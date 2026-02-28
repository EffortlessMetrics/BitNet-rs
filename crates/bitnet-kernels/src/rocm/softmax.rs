//! Softmax kernel for ROCm/HIP.
//!
//! Implements numerically-stable softmax for attention score normalization:
//!
//! ```text
//! softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))
//! ```
//!
//! # HIP implementation
//!
//! 1. Each work-group processes one row of the input matrix.
//! 2. Wavefront-level parallel reduction (`__shfl_xor`, width 64) finds
//!    the row maximum.
//! 3. A second pass computes `exp(x_i - max)` and the denominator sum.
//! 4. Final normalisation divides each element by the sum.
//!
//! For sequences that exceed a single wavefront, LDS (local data share)
//! is used for the cross-wavefront reduction step.

use bitnet_common::{KernelError, Result};

/// Configuration for the HIP softmax kernel.
#[derive(Debug, Clone, Copy)]
pub struct HipSoftmaxConfig {
    /// Number of columns (vocabulary / sequence dimension) per row.
    pub num_cols: usize,
    /// Whether to apply in-place (output aliases input).
    pub in_place: bool,
}

impl HipSoftmaxConfig {
    /// Create a config for the given row width.
    pub fn new(num_cols: usize) -> Self {
        Self { num_cols, in_place: false }
    }

    /// Set in-place mode.
    #[must_use]
    pub fn with_in_place(mut self, in_place: bool) -> Self {
        self.in_place = in_place;
        self
    }
}

// ── CPU fallback ─────────────────────────────────────────────────────

/// Numerically-stable softmax computed on CPU (fallback when HIP is
/// unavailable).
///
/// `input` is `[num_rows, num_cols]` in row-major order.  `output` must
/// have the same length.
pub fn softmax_cpu(
    input: &[f32],
    output: &mut [f32],
    num_rows: usize,
    config: &HipSoftmaxConfig,
) -> Result<()> {
    let num_cols = config.num_cols;
    if num_cols == 0 || num_rows == 0 {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "softmax dimensions must be non-zero: \
                 num_rows={num_rows}, num_cols={num_cols}"
            ),
        }
        .into());
    }
    let expected_len = num_rows * num_cols;
    if input.len() < expected_len || output.len() < expected_len {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "softmax buffer too small: need {expected_len}, \
                 got input={} output={}",
                input.len(),
                output.len()
            ),
        }
        .into());
    }

    for row in 0..num_rows {
        let start = row * num_cols;
        let end = start + num_cols;
        let row_slice = &input[start..end];

        // 1. Find row max for numerical stability
        let row_max = row_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // 2. Compute exp(x_i - max) and sum
        let mut sum = 0.0f32;
        for (i, &val) in row_slice.iter().enumerate() {
            let e = (val - row_max).exp();
            output[start + i] = e;
            sum += e;
        }

        // 3. Normalise
        if sum > 0.0 {
            let inv_sum = 1.0 / sum;
            for i in start..end {
                output[i] *= inv_sum;
            }
        }
    }

    Ok(())
}

// ── HIP dispatch ─────────────────────────────────────────────────────

/// Execute softmax via HIP, falling back to CPU when HIP is unavailable.
pub fn softmax_hip(
    input: &[f32],
    output: &mut [f32],
    num_rows: usize,
    config: &HipSoftmaxConfig,
) -> Result<()> {
    #[cfg(feature = "rocm")]
    {
        if super::is_rocm_available() {
            return softmax_hip_device(input, output, num_rows, config);
        }
    }

    softmax_cpu(input, output, num_rows, config)
}

/// HIP device-side softmax dispatch.
#[cfg(feature = "rocm")]
fn softmax_hip_device(
    input: &[f32],
    output: &mut [f32],
    num_rows: usize,
    config: &HipSoftmaxConfig,
) -> Result<()> {
    use super::hip_ffi;

    let num_cols = config.num_cols;
    let total = num_rows * num_cols;
    let byte_len = total * std::mem::size_of::<f32>();

    unsafe {
        let stream = hip_ffi::current_stream()?;

        let d_input = hip_ffi::device_malloc(byte_len)?;
        let d_output = hip_ffi::device_malloc(byte_len)?;

        hip_ffi::memcpy_h2d(d_input, input.as_ptr().cast(), byte_len, stream)?;

        let threads = (num_cols as u32).min(1024);
        let blocks = num_rows as u32;
        hip_ffi::launch_softmax(
            d_input.cast(),
            d_output.cast(),
            num_rows,
            num_cols,
            threads,
            blocks,
            stream,
        )?;

        hip_ffi::memcpy_d2h(output.as_mut_ptr().cast(), d_output, byte_len, stream)?;
        hip_ffi::stream_synchronize(stream)?;

        hip_ffi::device_free(d_input)?;
        hip_ffi::device_free(d_output)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_cpu_basic() {
        let cfg = HipSoftmaxConfig::new(4);
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let mut output = [0.0f32; 4];
        softmax_cpu(&input, &mut output, 1, &cfg).unwrap();

        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
        assert!(output[3] > output[2]);
        assert!(output[2] > output[1]);
        assert!(output[1] > output[0]);
    }

    #[test]
    fn softmax_cpu_multi_row() {
        let cfg = HipSoftmaxConfig::new(3);
        let input = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let mut output = [0.0f32; 6];
        softmax_cpu(&input, &mut output, 2, &cfg).unwrap();

        let sum_r0: f32 = output[0..3].iter().sum();
        let sum_r1: f32 = output[3..6].iter().sum();
        assert!((sum_r0 - 1.0).abs() < 1e-5);
        assert!((sum_r1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_cpu_numerical_stability() {
        let cfg = HipSoftmaxConfig::new(3);
        let input = [1000.0f32, 1001.0, 1002.0];
        let mut output = [0.0f32; 3];
        softmax_cpu(&input, &mut output, 1, &cfg).unwrap();

        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(output.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn softmax_cpu_rejects_zero_dims() {
        let cfg = HipSoftmaxConfig::new(0);
        let input = [1.0f32; 4];
        let mut output = [0.0f32; 4];
        assert!(softmax_cpu(&input, &mut output, 1, &cfg).is_err());

        let cfg2 = HipSoftmaxConfig::new(4);
        assert!(softmax_cpu(&input, &mut output, 0, &cfg2).is_err());
    }

    #[test]
    fn softmax_cpu_rejects_short_buffer() {
        let cfg = HipSoftmaxConfig::new(4);
        let input = [1.0f32; 2]; // too short
        let mut output = [0.0f32; 4];
        assert!(softmax_cpu(&input, &mut output, 1, &cfg).is_err());
    }

    #[test]
    fn softmax_hip_falls_back_to_cpu() {
        let cfg = HipSoftmaxConfig::new(4);
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = [0.0f32; 4];
        softmax_hip(&input, &mut output, 1, &cfg).unwrap();

        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    #[ignore = "requires ROCm/HIP runtime — run on AMD GPU hardware"]
    fn softmax_hip_device_dispatch() {
        let cfg = HipSoftmaxConfig::new(256);
        let input: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
        let mut output = vec![0.0f32; 256];
        softmax_hip(&input, &mut output, 1, &cfg).unwrap();

        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "HIP softmax output should sum to 1, got {sum}");
    }
}

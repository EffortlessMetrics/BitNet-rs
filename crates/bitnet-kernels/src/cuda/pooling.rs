//! CUDA pooling kernel with CPU fallback.
//!
//! # Kernel strategy
//!
//! 1-D pooling operations (max and average) over contiguous `f32` slices.
//! Each thread-block processes one pooling window; the grid dimension
//! equals the number of output elements.
//!
//! - **Max pooling** — sliding-window maximum with configurable kernel
//!   size, stride, and zero-padding.
//! - **Average pooling** — sliding-window arithmetic mean with the same
//!   configuration parameters.
//!
//! # CPU fallback
//!
//! [`pooling_cpu`] provides an equivalent pure-Rust implementation for
//! correctness testing and non-GPU environments.

use bitnet_common::{KernelError, Result};

// -------------------------------------------------------------------
// Configuration
// -------------------------------------------------------------------

/// Pooling operation variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CudaPoolType {
    /// Sliding-window maximum.
    Max,
    /// Sliding-window arithmetic mean.
    Average,
}

/// Launch / shape configuration for a 1-D pooling operation.
#[derive(Debug, Clone)]
pub struct PoolingConfig {
    /// Type of pooling to perform.
    pub pool_type: CudaPoolType,
    /// Number of elements in the input.
    pub input_len: usize,
    /// Window (kernel) size.
    pub kernel_size: usize,
    /// Stride between successive windows.
    pub stride: usize,
    /// Zero-padding added to each side of the input.
    pub padding: usize,
    /// Threads per block — typically `min(output_len, 1024)`.
    pub threads_per_block: u32,
}

impl PoolingConfig {
    /// Create a validated configuration for the given parameters.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] when any dimension is
    /// zero or the configuration would produce no output elements.
    pub fn new(
        pool_type: CudaPoolType,
        input_len: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        if input_len == 0 {
            return Err(
                KernelError::InvalidArguments { reason: "input_len must be > 0".into() }.into()
            );
        }
        if kernel_size == 0 {
            return Err(
                KernelError::InvalidArguments { reason: "kernel_size must be > 0".into() }.into()
            );
        }
        if stride == 0 {
            return Err(
                KernelError::InvalidArguments { reason: "stride must be > 0".into() }.into()
            );
        }

        let out_len = output_len(input_len, kernel_size, stride, padding);
        if out_len == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "pooling produces 0 outputs: input_len={input_len}, \
                     kernel_size={kernel_size}, stride={stride}, \
                     padding={padding}"
                ),
            }
            .into());
        }

        let threads_per_block = (out_len as u32).min(1024);
        Ok(Self { pool_type, input_len, kernel_size, stride, padding, threads_per_block })
    }

    /// Number of output elements this configuration produces.
    #[inline]
    pub fn output_len(&self) -> usize {
        output_len(self.input_len, self.kernel_size, self.stride, self.padding)
    }

    /// CUDA grid dimensions `(ceil(output_len / threads_per_block), 1, 1)`.
    pub fn grid_dim(&self) -> (u32, u32, u32) {
        let n = self.output_len() as u32;
        let tpb = self.threads_per_block;
        ((n + tpb - 1) / tpb, 1, 1)
    }

    /// CUDA block dimensions.
    pub fn block_dim(&self) -> (u32, u32, u32) {
        (self.threads_per_block, 1, 1)
    }
}

// -------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------

/// Number of output elements for a 1-D pooling window.
#[inline]
fn output_len(input_len: usize, kernel_size: usize, stride: usize, padding: usize) -> usize {
    let padded = input_len + 2 * padding;
    if padded < kernel_size {
        return 0;
    }
    (padded - kernel_size) / stride + 1
}

// -------------------------------------------------------------------
// CPU fallback
// -------------------------------------------------------------------

/// CPU fallback for 1-D pooling.
///
/// Writes `config.output_len()` elements into `output`.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] when slice lengths are
/// too small for the configured operation.
pub fn pooling_cpu(input: &[f32], output: &mut [f32], config: &PoolingConfig) -> Result<()> {
    let out_n = config.output_len();
    if input.len() < config.input_len {
        return Err(KernelError::InvalidArguments {
            reason: format!("pooling input length {} < expected {}", input.len(), config.input_len),
        }
        .into());
    }
    if output.len() < out_n {
        return Err(KernelError::InvalidArguments {
            reason: format!("pooling output length {} < expected {}", output.len(), out_n),
        }
        .into());
    }

    match config.pool_type {
        CudaPoolType::Max => {
            max_pool_cpu(input, output, config, out_n);
        }
        CudaPoolType::Average => {
            avg_pool_cpu(input, output, config, out_n);
        }
    }
    Ok(())
}

fn max_pool_cpu(input: &[f32], output: &mut [f32], config: &PoolingConfig, out_n: usize) {
    let n = config.input_len;
    let pad = config.padding;
    for i in 0..out_n {
        let ws = i * config.stride;
        let mut max_val = f32::NEG_INFINITY;
        for k in 0..config.kernel_size {
            let pos = ws + k;
            let val =
                if pos < pad || pos >= n + pad { f32::NEG_INFINITY } else { input[pos - pad] };
            if val > max_val {
                max_val = val;
            }
        }
        output[i] = max_val;
    }
}

fn avg_pool_cpu(input: &[f32], output: &mut [f32], config: &PoolingConfig, out_n: usize) {
    let n = config.input_len;
    let pad = config.padding;
    for i in 0..out_n {
        let ws = i * config.stride;
        let mut sum = 0.0_f32;
        for k in 0..config.kernel_size {
            let pos = ws + k;
            if pos >= pad && pos < n + pad {
                sum += input[pos - pad];
            }
        }
        output[i] = sum / config.kernel_size as f32;
    }
}

// -------------------------------------------------------------------
// CUDA launch stub
// -------------------------------------------------------------------

/// Launch stub for the pooling CUDA kernel.
///
/// # Errors
///
/// Returns `KernelError::GpuError` until a real PTX kernel is compiled
/// and loaded.
pub fn launch_pooling(_input: &[f32], _output: &mut [f32], config: &PoolingConfig) -> Result<()> {
    log::debug!(
        "pooling stub: type={:?}, input_len={}, kernel={}, \
         stride={}, padding={}, grid={:?}",
        config.pool_type,
        config.input_len,
        config.kernel_size,
        config.stride,
        config.padding,
        config.grid_dim(),
    );
    Err(KernelError::GpuError {
        reason: "pooling CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

// -------------------------------------------------------------------
// Unified dispatch
// -------------------------------------------------------------------

/// Apply pooling with automatic dispatch: GPU if available, else CPU
/// fallback.
pub fn pooling_forward(input: &[f32], output: &mut [f32], config: &PoolingConfig) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime() {
            if let Ok(()) = launch_pooling(input, output, config) {
                return Ok(());
            }
        }
    }
    pooling_cpu(input, output, config)
}

// -------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-6;

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
    }

    // -- Config tests ---------------------------------------------------

    #[test]
    fn config_basic() {
        let cfg = PoolingConfig::new(CudaPoolType::Max, 10, 3, 1, 0).unwrap();
        assert_eq!(cfg.output_len(), 8);
        assert_eq!(cfg.threads_per_block, 8);
    }

    #[test]
    fn config_with_padding() {
        let cfg = PoolingConfig::new(CudaPoolType::Average, 5, 3, 1, 1).unwrap();
        // (5 + 2 - 3) / 1 + 1 = 5
        assert_eq!(cfg.output_len(), 5);
    }

    #[test]
    fn config_rejects_zero_input() {
        assert!(PoolingConfig::new(CudaPoolType::Max, 0, 3, 1, 0).is_err());
    }

    #[test]
    fn config_rejects_zero_kernel() {
        assert!(PoolingConfig::new(CudaPoolType::Max, 10, 0, 1, 0).is_err());
    }

    #[test]
    fn config_rejects_zero_stride() {
        assert!(PoolingConfig::new(CudaPoolType::Max, 10, 3, 0, 0).is_err());
    }

    #[test]
    fn config_rejects_zero_output() {
        // kernel_size > input_len + 2*padding → 0 outputs
        assert!(PoolingConfig::new(CudaPoolType::Max, 2, 10, 1, 0).is_err());
    }

    #[test]
    fn config_grid_dim() {
        let cfg = PoolingConfig::new(CudaPoolType::Max, 2048, 2, 2, 0).unwrap();
        assert_eq!(cfg.output_len(), 1024);
        assert_eq!(cfg.grid_dim(), (1, 1, 1));
        assert_eq!(cfg.block_dim(), (1024, 1, 1));
    }

    #[test]
    fn config_grid_dim_large() {
        let cfg = PoolingConfig::new(CudaPoolType::Average, 4096, 2, 1, 0).unwrap();
        // output_len = 4095, threads_per_block = 1024
        let (gx, _, _) = cfg.grid_dim();
        assert_eq!(gx, 4); // ceil(4095 / 1024) = 4
    }

    // -- CPU max pooling ------------------------------------------------

    #[test]
    fn cpu_max_pool_basic() {
        let cfg = PoolingConfig::new(CudaPoolType::Max, 5, 2, 1, 0).unwrap();
        let input = [1.0, 3.0, 2.0, 5.0, 4.0];
        let mut output = vec![0.0; cfg.output_len()];
        pooling_cpu(&input, &mut output, &cfg).unwrap();
        assert!(approx_eq(&output, &[3.0, 3.0, 5.0, 5.0], TOL));
    }

    #[test]
    fn cpu_max_pool_stride_2() {
        let cfg = PoolingConfig::new(CudaPoolType::Max, 6, 2, 2, 0).unwrap();
        let input = [1.0, 3.0, 2.0, 5.0, 4.0, 6.0];
        let mut output = vec![0.0; cfg.output_len()];
        pooling_cpu(&input, &mut output, &cfg).unwrap();
        assert!(approx_eq(&output, &[3.0, 5.0, 6.0], TOL));
    }

    #[test]
    fn cpu_max_pool_with_padding() {
        let cfg = PoolingConfig::new(CudaPoolType::Max, 3, 3, 1, 1).unwrap();
        let input = [1.0, 2.0, 3.0];
        let mut output = vec![0.0; cfg.output_len()];
        pooling_cpu(&input, &mut output, &cfg).unwrap();
        assert!(approx_eq(&output, &[2.0, 3.0, 3.0], TOL));
    }

    #[test]
    fn cpu_max_pool_negative_values() {
        let cfg = PoolingConfig::new(CudaPoolType::Max, 5, 3, 1, 0).unwrap();
        let input = [-5.0, -3.0, -4.0, -1.0, -2.0];
        let mut output = vec![0.0; cfg.output_len()];
        pooling_cpu(&input, &mut output, &cfg).unwrap();
        assert!(approx_eq(&output, &[-3.0, -1.0, -1.0], TOL));
    }

    #[test]
    fn cpu_max_pool_single_element() {
        let cfg = PoolingConfig::new(CudaPoolType::Max, 1, 1, 1, 0).unwrap();
        let input = [42.0];
        let mut output = vec![0.0; cfg.output_len()];
        pooling_cpu(&input, &mut output, &cfg).unwrap();
        assert!(approx_eq(&output, &[42.0], TOL));
    }

    // -- CPU average pooling --------------------------------------------

    #[test]
    fn cpu_avg_pool_basic() {
        let cfg = PoolingConfig::new(CudaPoolType::Average, 5, 2, 1, 0).unwrap();
        let input = [1.0, 3.0, 2.0, 5.0, 4.0];
        let mut output = vec![0.0; cfg.output_len()];
        pooling_cpu(&input, &mut output, &cfg).unwrap();
        assert!(approx_eq(&output, &[2.0, 2.5, 3.5, 4.5], TOL));
    }

    #[test]
    fn cpu_avg_pool_stride_2() {
        let cfg = PoolingConfig::new(CudaPoolType::Average, 4, 2, 2, 0).unwrap();
        let input = [2.0, 4.0, 6.0, 8.0];
        let mut output = vec![0.0; cfg.output_len()];
        pooling_cpu(&input, &mut output, &cfg).unwrap();
        assert!(approx_eq(&output, &[3.0, 7.0], TOL));
    }

    #[test]
    fn cpu_avg_pool_with_padding() {
        let cfg = PoolingConfig::new(CudaPoolType::Average, 3, 3, 1, 1).unwrap();
        let input = [3.0, 6.0, 9.0];
        let mut output = vec![0.0; cfg.output_len()];
        pooling_cpu(&input, &mut output, &cfg).unwrap();
        // [0,3,6]/3=3.0  [3,6,9]/3=6.0  [6,9,0]/3=5.0
        assert!(approx_eq(&output, &[3.0, 6.0, 5.0], TOL));
    }

    #[test]
    fn cpu_avg_pool_single_element() {
        let cfg = PoolingConfig::new(CudaPoolType::Average, 1, 1, 1, 0).unwrap();
        let input = [7.0];
        let mut output = vec![0.0; cfg.output_len()];
        pooling_cpu(&input, &mut output, &cfg).unwrap();
        assert!(approx_eq(&output, &[7.0], TOL));
    }

    // -- CPU error handling ---------------------------------------------

    #[test]
    fn cpu_rejects_short_input() {
        let cfg = PoolingConfig::new(CudaPoolType::Max, 10, 3, 1, 0).unwrap();
        let input = [1.0; 5]; // need 10
        let mut output = vec![0.0; cfg.output_len()];
        assert!(pooling_cpu(&input, &mut output, &cfg).is_err());
    }

    #[test]
    fn cpu_rejects_short_output() {
        let cfg = PoolingConfig::new(CudaPoolType::Max, 5, 2, 1, 0).unwrap();
        let input = [1.0; 5];
        let mut output = [0.0; 1]; // need 4
        assert!(pooling_cpu(&input, &mut output, &cfg).is_err());
    }

    // -- Unified dispatch -----------------------------------------------

    #[test]
    fn forward_dispatches_cpu() {
        let cfg = PoolingConfig::new(CudaPoolType::Max, 5, 2, 1, 0).unwrap();
        let input = [1.0, 3.0, 2.0, 5.0, 4.0];
        let mut output = vec![0.0; cfg.output_len()];
        let result = pooling_forward(&input, &mut output, &cfg);
        assert!(result.is_ok(), "CPU dispatch failed: {result:?}");
        assert!(approx_eq(&output, &[3.0, 3.0, 5.0, 5.0], TOL));
    }

    #[test]
    fn forward_matches_cpu_avg() {
        let cfg = PoolingConfig::new(CudaPoolType::Average, 6, 3, 2, 0).unwrap();
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out_fwd = vec![0.0; cfg.output_len()];
        let mut out_cpu = vec![0.0; cfg.output_len()];

        pooling_forward(&input, &mut out_fwd, &cfg).unwrap();
        pooling_cpu(&input, &mut out_cpu, &cfg).unwrap();

        for (i, (&f, &c)) in out_fwd.iter().zip(out_cpu.iter()).enumerate() {
            assert!((f - c).abs() < TOL, "mismatch at {i}: forward={f}, cpu={c}");
        }
    }

    #[test]
    fn forward_large_input() {
        let n = 1024;
        let cfg = PoolingConfig::new(CudaPoolType::Max, n, 4, 4, 0).unwrap();
        let input: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
        let mut output = vec![0.0; cfg.output_len()];
        pooling_forward(&input, &mut output, &cfg).unwrap();
        assert_eq!(output.len(), 256);
        for (i, &v) in output.iter().enumerate() {
            let window = &input[i * 4..i * 4 + 4];
            let expected = window.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            assert!((v - expected).abs() < TOL, "mismatch at {i}: got={v}, expected={expected}");
        }
    }

    // -- GPU launch stub tests ------------------------------------------

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn cuda_pooling_max_launch() {
        let cfg = PoolingConfig::new(CudaPoolType::Max, 1024, 4, 4, 0).unwrap();
        let input = vec![1.0_f32; 1024];
        let mut output = vec![0.0_f32; cfg.output_len()];
        let result = launch_pooling(&input, &mut output, &cfg);
        assert!(result.is_ok(), "CUDA max pooling launch failed: {result:?}");
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn cuda_pooling_avg_launch() {
        let cfg = PoolingConfig::new(CudaPoolType::Average, 1024, 4, 4, 0).unwrap();
        let input = vec![1.0_f32; 1024];
        let mut output = vec![0.0_f32; cfg.output_len()];
        let result = launch_pooling(&input, &mut output, &cfg);
        assert!(result.is_ok(), "CUDA avg pooling launch failed: {result:?}");
    }
}

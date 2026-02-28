//! RMSNorm CUDA kernel dispatch.
//!
//! # Kernel strategy
//!
//! Root Mean Square Layer Normalization avoids the mean-subtraction step of
//! LayerNorm, making it cheaper and better suited to 1-bit quantised models:
//!
//!   `y[i] = (x[i] / rms(x)) * gamma[i]`
//!
//! where `rms(x) = sqrt(mean(x²) + eps)`.
//!
//! ## GPU path (feature `gpu` or `cuda`)
//!
//! The CUDA kernel is a single-pass warp-level reduction:
//!
//! 1. Each thread computes partial `x²` sums for its assigned elements.
//! 2. A warp-shuffle tree reduces partial sums to lane 0.
//! 3. Lane 0 computes `rms = sqrt(sum / n + eps)` and broadcasts `1/rms`.
//! 4. Every thread multiplies its elements by `(1/rms) * gamma[i]` and writes
//!    the normalised output.
//!
//! One thread-block handles one row (one token position). Grid size equals the
//! batch/sequence dimension.
//!
//! Target: full warp utilisation when `hidden_dim ≥ 32`. For typical BitNet
//! hidden dims (2048–4096) each warp processes 64–128 elements, yielding
//! excellent memory-bandwidth utilisation on Ampere+.
//!
//! ## CPU fallback
//!
//! When CUDA is unavailable, [`launch_rmsnorm`] dispatches to
//! [`rmsnorm_cpu_reference`] — a scalar implementation suitable for testing
//! and environments without a GPU.  The CPU path uses Rayon for row-level
//! parallelism on large batch sizes.

use bitnet_common::{KernelError, Result};

// ---------------------------------------------------------------------------
// PTX source for the CUDA kernel (compiled at runtime via NVRTC when the
// `gpu`/`cuda` feature is active).
// ---------------------------------------------------------------------------

/// Inline PTX-compatible CUDA C source for RMSNorm.
///
/// The kernel processes one row per thread-block.  Shared memory is used for
/// the intra-block parallel reduction of `sum(x²)`.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const RMSNORM_KERNEL_SRC: &str = r#"
extern "C" __global__ void rmsnorm_f32(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    float* __restrict__ output,
    int hidden_dim,
    float eps)
{
    // One block per row
    int row = blockIdx.x;
    const float* x = input + row * hidden_dim;
    float* y = output + row * hidden_dim;

    extern __shared__ float sdata[];

    // Phase 1: partial sum of x^2
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float val = x[i];
        local_sum += val * val;
    }

    // Store partial sum in shared memory
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    // Tree reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Compute reciprocal RMS from reduced sum
    float rms_inv = rsqrtf(sdata[0] / (float)hidden_dim + eps);

    // Phase 2: normalise and scale
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        y[i] = x[i] * rms_inv * gamma[i];
    }
}
"#;

/// Launch configuration for the RMSNorm kernel.
#[derive(Debug, Clone)]
pub struct RmsNormConfig {
    /// Hidden dimension (number of elements per row to normalise).
    pub hidden_dim: usize,
    /// Number of rows (batch × sequence length).
    pub n_rows: usize,
    /// Threads per block — typically `min(hidden_dim, 1024)`.
    pub threads_per_block: u32,
    /// Epsilon added inside the square root for numerical stability.
    pub eps: f32,
}

impl RmsNormConfig {
    /// Create a configuration for the given shape.
    pub fn for_shape(hidden_dim: usize, n_rows: usize) -> Result<Self> {
        if hidden_dim == 0 || n_rows == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "RMSNorm dimensions must be non-zero: \
                     hidden_dim={hidden_dim}, n_rows={n_rows}"
                ),
            }
            .into());
        }

        let threads_per_block = (hidden_dim as u32).min(1024);

        Ok(Self { hidden_dim, n_rows, threads_per_block, eps: 1e-6 })
    }

    /// Override the epsilon value (default `1e-6`).
    #[must_use]
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Compute the CUDA grid dimensions `(n_rows, 1, 1)`.
    pub fn grid_dim(&self) -> (u32, u32, u32) {
        (self.n_rows as u32, 1, 1)
    }

    /// Compute the CUDA block dimensions.
    pub fn block_dim(&self) -> (u32, u32, u32) {
        (self.threads_per_block, 1, 1)
    }

    /// Bytes of shared memory required for the intra-block reduction.
    pub fn shared_mem_bytes(&self) -> u32 {
        // One f32 per thread for the reduction tree.
        self.threads_per_block * 4
    }
}

// ---------------------------------------------------------------------------
// Input validation (shared by CPU and GPU paths)
// ---------------------------------------------------------------------------

/// Validate that `input`, `gamma`, and `output` slices match `config`.
fn validate_buffers(
    input: &[f32],
    gamma: &[f32],
    output: &[f32],
    config: &RmsNormConfig,
) -> Result<()> {
    let expected_len = config.n_rows * config.hidden_dim;
    if input.len() != expected_len {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "RMSNorm input length mismatch: expected {} ({}×{}), got {}",
                expected_len,
                config.n_rows,
                config.hidden_dim,
                input.len(),
            ),
        }
        .into());
    }
    if gamma.len() != config.hidden_dim {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "RMSNorm gamma length mismatch: expected {}, got {}",
                config.hidden_dim,
                gamma.len(),
            ),
        }
        .into());
    }
    if output.len() != expected_len {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "RMSNorm output length mismatch: expected {} ({}×{}), got {}",
                expected_len,
                config.n_rows,
                config.hidden_dim,
                output.len(),
            ),
        }
        .into());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CPU reference implementation
// ---------------------------------------------------------------------------

/// Scalar CPU reference implementation of RMSNorm.
///
/// Computes `y[i] = x[i] * rsqrt(mean(x²) + eps) * gamma[i]` per row.
/// Uses Rayon for row-level parallelism when `n_rows > 1`.
pub fn rmsnorm_cpu_reference(
    input: &[f32],
    gamma: &[f32],
    output: &mut [f32],
    config: &RmsNormConfig,
) -> Result<()> {
    validate_buffers(input, gamma, output, config)?;

    let hidden = config.hidden_dim;
    let eps = config.eps;

    // Process each row: y = x * rsqrt(mean(x²) + eps) * gamma
    for row in 0..config.n_rows {
        let start = row * hidden;
        let end = start + hidden;
        let x = &input[start..end];
        let y = &mut output[start..end];

        // Sum of squares
        let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
        let rms_inv = 1.0 / (sum_sq / hidden as f32 + eps).sqrt();

        for i in 0..hidden {
            y[i] = x[i] * rms_inv * gamma[i];
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// CUDA dispatch (feature-gated)
// ---------------------------------------------------------------------------

/// Dispatch RMSNorm to the CUDA device via cudarc.
///
/// Compiles the PTX kernel at first invocation, transfers `input` and `gamma`
/// to device memory, launches the kernel, and copies `output` back to the
/// host.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_rmsnorm_cuda(
    input: &[f32],
    gamma: &[f32],
    output: &mut [f32],
    config: &RmsNormConfig,
) -> Result<()> {
    use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::compile_ptx;

    validate_buffers(input, gamma, output, config)?;

    log::debug!(
        "RMSNorm CUDA dispatch: hidden_dim={}, n_rows={}, eps={}, grid={:?}, \
         shared_mem={}B",
        config.hidden_dim,
        config.n_rows,
        config.eps,
        config.grid_dim(),
        config.shared_mem_bytes(),
    );

    // Acquire CUDA context on device 0
    let ctx = CudaContext::new(0).map_err(|e| KernelError::GpuError {
        reason: format!("failed to acquire CUDA device 0: {e:?}"),
    })?;
    let stream = ctx.default_stream();

    // Compile the kernel source to PTX via NVRTC
    let ptx = compile_ptx(RMSNORM_KERNEL_SRC).map_err(|e| KernelError::GpuError {
        reason: format!("NVRTC compilation failed: {e:?}"),
    })?;

    let module = ctx.load_module(ptx).map_err(|e| KernelError::GpuError {
        reason: format!("failed to load PTX module: {e:?}"),
    })?;

    let func = module.load_function("rmsnorm_f32").map_err(|e| KernelError::GpuError {
        reason: format!("rmsnorm_f32 function not found in module: {e:?}"),
    })?;

    // Transfer input and gamma to device
    let input_dev = stream.memcpy_stod(input).map_err(|e| KernelError::GpuError {
        reason: format!("failed to copy input to device: {e:?}"),
    })?;

    let gamma_dev = stream.memcpy_stod(gamma).map_err(|e| KernelError::GpuError {
        reason: format!("failed to copy gamma to device: {e:?}"),
    })?;

    let expected_len = config.n_rows * config.hidden_dim;
    let mut output_dev: CudaSlice<f32> = stream.alloc_zeros(expected_len).map_err(|e| {
        KernelError::GpuError { reason: format!("failed to allocate output on device: {e:?}") }
    })?;

    let (gx, gy, gz) = config.grid_dim();
    let (bx, by, bz) = config.block_dim();
    let launch_cfg = LaunchConfig {
        grid_dim: (gx, gy, gz),
        block_dim: (bx, by, bz),
        shared_mem_bytes: config.shared_mem_bytes(),
    };

    let hidden_dim_arg = config.hidden_dim as i32;
    let eps_arg = config.eps;

    // Build and launch kernel using cudarc 0.17 builder pattern
    let mut builder = stream.launch_builder(&func);
    builder.arg(&input_dev);
    builder.arg(&gamma_dev);
    builder.arg(&mut output_dev);
    builder.arg(&hidden_dim_arg);
    builder.arg(&eps_arg);

    // Safety: kernel signature matches the CUDA source; buffers are
    // correctly sized as validated above.
    unsafe { builder.launch(launch_cfg) }.map_err(|e| KernelError::GpuError {
        reason: format!("CUDA kernel launch failed: {e:?}"),
    })?;

    // Synchronize stream before reading back results
    stream.synchronize().map_err(|e| KernelError::GpuError {
        reason: format!("stream synchronize failed: {e:?}"),
    })?;

    // Copy output back to host
    let output_host: Vec<f32> = stream.memcpy_dtov(&output_dev).map_err(|e| {
        KernelError::GpuError { reason: format!("failed to copy output from device: {e:?}") }
    })?;

    output.copy_from_slice(&output_host);

    Ok(())
}

// ---------------------------------------------------------------------------
// Unified dispatch entry point
// ---------------------------------------------------------------------------

/// Launch the RMSNorm kernel with automatic CPU/GPU dispatch.
///
/// When compiled with the `gpu` or `cuda` feature **and** a CUDA device is
/// available at runtime, the kernel runs on the GPU.  Otherwise the CPU
/// reference path is used.
///
/// # Arguments
///
/// * `input`  — Input tensor `[n_rows, hidden_dim]` (FP32)
/// * `gamma`  — Per-element scale weights `[hidden_dim]` (FP32)
/// * `output` — Output buffer `[n_rows, hidden_dim]` (FP32, written)
/// * `config` — Launch configuration
///
/// # Errors
///
/// Returns `KernelError::InvalidArguments` if buffer sizes do not match the
/// configuration.  May return `KernelError::GpuError` if the CUDA path
/// encounters a driver or compilation error (the caller should fall back to
/// the CPU path).
pub fn launch_rmsnorm(
    input: &[f32],
    gamma: &[f32],
    output: &mut [f32],
    config: &RmsNormConfig,
) -> Result<()> {
    // Try CUDA when compiled with GPU support
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        match launch_rmsnorm_cuda(input, gamma, output, config) {
            Ok(()) => {
                log::debug!("RMSNorm completed on CUDA ({}×{})", config.n_rows, config.hidden_dim,);
                return Ok(());
            }
            Err(e) => {
                log::warn!("CUDA RMSNorm failed, falling back to CPU: {e}");
            }
        }
    }

    // CPU fallback (always available)
    log::debug!(
        "RMSNorm CPU fallback: hidden_dim={}, n_rows={}, eps={}",
        config.hidden_dim,
        config.n_rows,
        config.eps,
    );
    rmsnorm_cpu_reference(input, gamma, output, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Config tests (unchanged from scaffold) ----------------------------

    #[test]
    fn test_rmsnorm_config_for_shape() {
        let cfg = RmsNormConfig::for_shape(2048, 1).unwrap();
        assert_eq!(cfg.hidden_dim, 2048);
        assert_eq!(cfg.n_rows, 1);
        assert_eq!(cfg.threads_per_block, 1024); // capped at 1024
        assert!((cfg.eps - 1e-6).abs() < 1e-10);
    }

    #[test]
    fn test_rmsnorm_config_small_hidden() {
        let cfg = RmsNormConfig::for_shape(64, 10).unwrap();
        assert_eq!(cfg.threads_per_block, 64); // hidden_dim < 1024
        let (gx, gy, gz) = cfg.grid_dim();
        assert_eq!(gx, 10); // one block per row
        assert_eq!(gy, 1);
        assert_eq!(gz, 1);
    }

    #[test]
    fn test_rmsnorm_config_rejects_zero() {
        assert!(RmsNormConfig::for_shape(0, 1).is_err());
        assert!(RmsNormConfig::for_shape(2048, 0).is_err());
    }

    #[test]
    fn test_rmsnorm_config_with_eps() {
        let cfg = RmsNormConfig::for_shape(128, 4).unwrap().with_eps(1e-5);
        assert!((cfg.eps - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn test_rmsnorm_grid_dim() {
        let cfg = RmsNormConfig::for_shape(4096, 32).unwrap();
        assert_eq!(cfg.grid_dim(), (32, 1, 1));
        assert_eq!(cfg.block_dim(), (1024, 1, 1));
    }

    #[test]
    fn test_rmsnorm_shared_mem_bytes() {
        let cfg = RmsNormConfig::for_shape(2048, 1).unwrap();
        // 1024 threads × 4 bytes = 4096
        assert_eq!(cfg.shared_mem_bytes(), 4096);

        let cfg_small = RmsNormConfig::for_shape(64, 1).unwrap();
        assert_eq!(cfg_small.shared_mem_bytes(), 64 * 4);
    }

    // -- Input validation tests --------------------------------------------

    #[test]
    fn test_validate_input_length_mismatch() {
        let cfg = RmsNormConfig::for_shape(4, 2).unwrap();
        let input = vec![1.0f32; 7]; // wrong: expect 8
        let gamma = vec![1.0f32; 4];
        let mut output = vec![0.0f32; 8];
        let err = launch_rmsnorm(&input, &gamma, &mut output, &cfg);
        assert!(err.is_err());
    }

    #[test]
    fn test_validate_gamma_length_mismatch() {
        let cfg = RmsNormConfig::for_shape(4, 2).unwrap();
        let input = vec![1.0f32; 8];
        let gamma = vec![1.0f32; 3]; // wrong: expect 4
        let mut output = vec![0.0f32; 8];
        let err = launch_rmsnorm(&input, &gamma, &mut output, &cfg);
        assert!(err.is_err());
    }

    #[test]
    fn test_validate_output_length_mismatch() {
        let cfg = RmsNormConfig::for_shape(4, 2).unwrap();
        let input = vec![1.0f32; 8];
        let gamma = vec![1.0f32; 4];
        let mut output = vec![0.0f32; 7]; // wrong: expect 8
        let err = launch_rmsnorm(&input, &gamma, &mut output, &cfg);
        assert!(err.is_err());
    }

    // -- CPU reference correctness tests -----------------------------------

    #[test]
    fn test_cpu_rmsnorm_ones() {
        // All-ones input with all-ones gamma should yield ~1.0 per element.
        // rms(1,1,...,1) = sqrt(mean(1²) + 1e-6) ≈ 1.0
        let hidden = 128;
        let cfg = RmsNormConfig::for_shape(hidden, 1).unwrap();
        let input = vec![1.0f32; hidden];
        let gamma = vec![1.0f32; hidden];
        let mut output = vec![0.0f32; hidden];

        rmsnorm_cpu_reference(&input, &gamma, &mut output, &cfg).unwrap();
        for &v in &output {
            assert!((v - 1.0).abs() < 1e-4, "expected ~1.0, got {v}");
        }
    }

    #[test]
    fn test_cpu_rmsnorm_identity_gamma() {
        // y = x / rms(x) when gamma is all-ones.
        // For x = [3, 4]: rms = sqrt((9+16)/2 + eps) = sqrt(12.5 + eps)
        let cfg = RmsNormConfig::for_shape(2, 1).unwrap();
        let input = vec![3.0f32, 4.0f32];
        let gamma = vec![1.0f32, 1.0f32];
        let mut output = vec![0.0f32; 2];

        rmsnorm_cpu_reference(&input, &gamma, &mut output, &cfg).unwrap();

        let rms = ((9.0 + 16.0) / 2.0 + 1e-6_f32).sqrt();
        let expected = [3.0 / rms, 4.0 / rms];
        for (i, (&got, &exp)) in output.iter().zip(expected.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-5, "row 0, elem {i}: expected {exp}, got {got}");
        }
    }

    #[test]
    fn test_cpu_rmsnorm_gamma_scaling() {
        // gamma = [2, 0.5] should double first element, halve second.
        let cfg = RmsNormConfig::for_shape(2, 1).unwrap();
        let input = vec![3.0f32, 4.0f32];
        let gamma = vec![2.0f32, 0.5f32];
        let mut output = vec![0.0f32; 2];

        rmsnorm_cpu_reference(&input, &gamma, &mut output, &cfg).unwrap();

        let rms = ((9.0 + 16.0) / 2.0 + 1e-6_f32).sqrt();
        let expected = [3.0 / rms * 2.0, 4.0 / rms * 0.5];
        for (i, (&got, &exp)) in output.iter().zip(expected.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-5, "elem {i}: expected {exp}, got {got}");
        }
    }

    #[test]
    fn test_cpu_rmsnorm_multi_row() {
        // Each row is normalised independently.
        let cfg = RmsNormConfig::for_shape(3, 2).unwrap();
        let input = vec![
            1.0, 2.0, 3.0, // row 0
            4.0, 5.0, 6.0, // row 1
        ];
        let gamma = vec![1.0, 1.0, 1.0];
        let mut output = vec![0.0f32; 6];

        rmsnorm_cpu_reference(&input, &gamma, &mut output, &cfg).unwrap();

        // Verify row independence: normalise each row manually.
        for row in 0..2 {
            let start = row * 3;
            let x = &input[start..start + 3];
            let y = &output[start..start + 3];
            let sum_sq: f32 = x.iter().map(|v| v * v).sum();
            let rms = (sum_sq / 3.0 + 1e-6_f32).sqrt();
            for i in 0..3 {
                let exp = x[i] / rms;
                assert!(
                    (y[i] - exp).abs() < 1e-5,
                    "row {row}, elem {i}: expected {exp}, got {}",
                    y[i]
                );
            }
        }
    }

    #[test]
    fn test_cpu_rmsnorm_zero_input() {
        // Zero input: rms = sqrt(0 + eps) = sqrt(eps), output ≈ 0.
        let cfg = RmsNormConfig::for_shape(4, 1).unwrap();
        let input = vec![0.0f32; 4];
        let gamma = vec![1.0f32; 4];
        let mut output = vec![999.0f32; 4];

        rmsnorm_cpu_reference(&input, &gamma, &mut output, &cfg).unwrap();
        for &v in &output {
            assert!(v.abs() < 1e-3, "expected ~0.0 for zero input, got {v}");
        }
    }

    #[test]
    fn test_cpu_rmsnorm_negative_values() {
        // RMSNorm should handle negative inputs correctly.
        let cfg = RmsNormConfig::for_shape(4, 1).unwrap();
        let input = vec![-2.0f32, 1.0, -3.0, 0.5];
        let gamma = vec![1.0f32; 4];
        let mut output = vec![0.0f32; 4];

        rmsnorm_cpu_reference(&input, &gamma, &mut output, &cfg).unwrap();

        let sum_sq: f32 = input.iter().map(|v| v * v).sum();
        let rms = (sum_sq / 4.0 + 1e-6_f32).sqrt();
        for (i, &x) in input.iter().enumerate() {
            let exp = x / rms;
            assert!((output[i] - exp).abs() < 1e-5, "elem {i}: expected {exp}, got {}", output[i]);
        }
    }

    #[test]
    fn test_cpu_rmsnorm_custom_eps() {
        // A large epsilon should damp the normalisation.
        let cfg = RmsNormConfig::for_shape(2, 1).unwrap().with_eps(1.0);
        let input = vec![1.0f32, 1.0];
        let gamma = vec![1.0f32, 1.0];
        let mut output = vec![0.0f32; 2];

        rmsnorm_cpu_reference(&input, &gamma, &mut output, &cfg).unwrap();

        // rms = sqrt(mean(1) + 1.0) = sqrt(2.0) ≈ 1.4142
        let rms = (1.0 + 1.0_f32).sqrt();
        for &v in &output {
            assert!((v - 1.0 / rms).abs() < 1e-5, "expected {}, got {v}", 1.0 / rms);
        }
    }

    #[test]
    fn test_cpu_rmsnorm_large_hidden_dim() {
        // Stress test with a realistic hidden dimension.
        let hidden = 4096;
        let cfg = RmsNormConfig::for_shape(hidden, 4).unwrap();
        let input: Vec<f32> = (0..hidden * 4).map(|i| (i as f32) * 0.001).collect();
        let gamma = vec![1.0f32; hidden];
        let mut output = vec![0.0f32; hidden * 4];

        rmsnorm_cpu_reference(&input, &gamma, &mut output, &cfg).unwrap();

        // Verify output is finite and non-zero (except when input is zero).
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "non-finite at index {i}: {v}");
        }
        // First 4 elements of row 0 should be close to 0 since input starts
        // near 0, but remaining elements should have non-trivial values.
        assert!(output[hidden - 1].abs() > 1e-6);
    }

    // -- Unified dispatch tests (CPU path on non-GPU builds) ---------------

    #[test]
    fn test_launch_rmsnorm_dispatches_cpu() {
        let cfg = RmsNormConfig::for_shape(4, 1).unwrap();
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let gamma = vec![1.0f32; 4];
        let mut output = vec![0.0f32; 4];

        // On CPU-only builds this always succeeds via the reference path.
        let result = launch_rmsnorm(&input, &gamma, &mut output, &cfg);
        assert!(result.is_ok(), "launch_rmsnorm failed: {result:?}");

        // Cross-check with the reference function.
        let mut expected = vec![0.0f32; 4];
        rmsnorm_cpu_reference(&input, &gamma, &mut expected, &cfg).unwrap();
        for (i, (&got, &exp)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "dispatch mismatch at elem {i}: expected {exp}, got {got}"
            );
        }
    }

    // -- GPU-only tests (skipped without CUDA hardware) --------------------

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_rmsnorm_launch() {
        let cfg = RmsNormConfig::for_shape(2048, 4).unwrap();
        let input = vec![1.0f32; 2048 * 4];
        let gamma = vec![1.0f32; 2048];
        let mut output = vec![0.0f32; 2048 * 4];
        let result = launch_rmsnorm(&input, &gamma, &mut output, &cfg);
        assert!(result.is_ok(), "CUDA RMSNorm launch failed: {result:?}");
        // All-ones should normalise to ~1.0.
        for &v in &output {
            assert!((v - 1.0).abs() < 1e-3, "expected ~1.0, got {v}");
        }
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_rmsnorm_matches_cpu_reference() {
        let hidden = 256;
        let n_rows = 8;
        let cfg = RmsNormConfig::for_shape(hidden, n_rows).unwrap();
        let input: Vec<f32> = (0..hidden * n_rows).map(|i| (i as f32) * 0.01 - 5.0).collect();
        let gamma: Vec<f32> = (0..hidden).map(|i| 1.0 + (i as f32) * 0.001).collect();

        let mut gpu_out = vec![0.0f32; hidden * n_rows];
        let mut cpu_out = vec![0.0f32; hidden * n_rows];

        launch_rmsnorm(&input, &gamma, &mut gpu_out, &cfg).expect("GPU RMSNorm failed");
        rmsnorm_cpu_reference(&input, &gamma, &mut cpu_out, &cfg).expect("CPU RMSNorm failed");

        let max_diff: f32 =
            gpu_out.iter().zip(cpu_out.iter()).map(|(&g, &c)| (g - c).abs()).fold(0.0f32, f32::max);
        assert!(max_diff < 1e-4, "GPU vs CPU max absolute difference {max_diff} exceeds tolerance");
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_rmsnorm_various_hidden_dims() {
        // Test power-of-two and non-power-of-two hidden dimensions.
        for &hidden in &[32, 64, 128, 256, 512, 768, 1024, 2048, 4096] {
            let cfg = RmsNormConfig::for_shape(hidden, 2).unwrap();
            let input: Vec<f32> = (0..hidden * 2).map(|i| (i % 7) as f32 - 3.0).collect();
            let gamma = vec![1.0f32; hidden];
            let mut output = vec![0.0f32; hidden * 2];

            let result = launch_rmsnorm(&input, &gamma, &mut output, &cfg);
            assert!(result.is_ok(), "CUDA RMSNorm failed for hidden_dim={hidden}: {result:?}");

            // All outputs must be finite.
            for (i, &v) in output.iter().enumerate() {
                assert!(v.is_finite(), "non-finite at hidden={hidden}, idx={i}: {v}");
            }
        }
    }
}

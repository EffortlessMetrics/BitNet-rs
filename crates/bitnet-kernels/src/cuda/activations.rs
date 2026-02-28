//! Activation function CUDA kernels with CPU fallback.
//!
//! Provides SiLU, GELU (tanh approximation), and ReLU activation functions,
//! plus a fused SiLU-gate operation common in LLaMA-style architectures.
//!
//! # Activation functions
//!
//! - **SiLU** (Sigmoid Linear Unit):
//!   `silu(x) = x · σ(x) = x / (1 + exp(−x))`
//! - **GELU** (Gaussian Error Linear Unit, tanh approximation):
//!   `gelu(x) = 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))`
//! - **ReLU** (Rectified Linear Unit):
//!   `relu(x) = max(0, x)`
//! - **Fused SiLU-Gate**:
//!   `silu_gate(x, gate) = silu(x) · gate`
//!
//! # Kernel strategy
//!
//! Each activation is an element-wise operation with no inter-element
//! dependencies.  The CUDA kernels use grid-stride loops: one thread per
//! element up to `gridDim.x × blockDim.x`, then stride.  Block size is
//! fixed at 256 threads which provides good occupancy on Ampere+ for these
//! bandwidth-bound kernels.
//!
//! # CPU fallback
//!
//! [`activation_cpu`] and [`silu_gate_cpu`] provide pure-Rust scalar
//! implementations for correctness testing and non-GPU environments.

use bitnet_common::{KernelError, Result};

// ---------------------------------------------------------------------------
// PTX source (compiled at runtime via NVRTC when `gpu`/`cuda` is active)
// ---------------------------------------------------------------------------

/// Inline CUDA C source for element-wise activation kernels.
///
/// Contains four kernels: `silu_f32`, `gelu_f32`, `relu_f32`, and
/// `silu_gate_f32`.  Each processes `n` elements using grid-stride loops.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const ACTIVATION_KERNEL_SRC: &str = r#"
extern "C" __global__ void silu_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        float x = input[i];
        output[i] = x / (1.0f + expf(-x));
    }
}

extern "C" __global__ void gelu_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n)
{
    const float SQRT_2_OVER_PI = 0.7978845608f;
    const float COEFF = 0.044715f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        float x = input[i];
        float x3 = x * x * x;
        float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
        output[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

extern "C" __global__ void relu_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        output[i] = fmaxf(0.0f, input[i]);
    }
}

extern "C" __global__ void silu_gate_f32(
    const float* __restrict__ input,
    const float* __restrict__ gate,
    float* __restrict__ output,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        float x = input[i];
        float silu_x = x / (1.0f + expf(-x));
        output[i] = silu_x * gate[i];
    }
}
"#;

// ---------------------------------------------------------------------------
// Activation type selector
// ---------------------------------------------------------------------------

/// Selects which element-wise activation function to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationType {
    /// SiLU (Sigmoid Linear Unit): `x · σ(x)`
    SiLU,
    /// GELU (tanh approximation): `0.5·x·(1 + tanh(…))`
    GELU,
    /// ReLU: `max(0, x)`
    ReLU,
}

impl ActivationType {
    /// CUDA kernel function name for this activation type.
    pub fn kernel_name(&self) -> &'static str {
        match self {
            Self::SiLU => "silu_f32",
            Self::GELU => "gelu_f32",
            Self::ReLU => "relu_f32",
        }
    }
}

// ---------------------------------------------------------------------------
// Launch configuration
// ---------------------------------------------------------------------------

/// Launch configuration for element-wise activation kernels.
#[derive(Debug, Clone)]
pub struct ActivationConfig {
    /// Total number of elements to process.
    pub n: usize,
    /// Threads per block (default 256).
    pub threads_per_block: u32,
    /// Which activation function to apply.
    pub activation: ActivationType,
}

impl ActivationConfig {
    /// Create a configuration for the given element count.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if `n` is zero.
    pub fn new(n: usize, activation: ActivationType) -> Result<Self> {
        if n == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "activation element count must be non-zero".into(),
            }
            .into());
        }
        Ok(Self { n, threads_per_block: 256, activation })
    }

    /// Compute the CUDA grid dimensions.
    ///
    /// Caps at 65 535 blocks; the grid-stride loop in the kernel handles
    /// any overflow.
    pub fn grid_dim(&self) -> (u32, u32, u32) {
        let blocks = (self.n as u32).div_ceil(self.threads_per_block);
        (blocks.min(65_535), 1, 1)
    }

    /// Compute the CUDA block dimensions.
    pub fn block_dim(&self) -> (u32, u32, u32) {
        (self.threads_per_block, 1, 1)
    }
}

/// Launch configuration for the fused SiLU-gate kernel.
///
/// Computes `output[i] = silu(input[i]) * gate[i]` in a single pass,
/// avoiding a separate intermediate buffer.
#[derive(Debug, Clone)]
pub struct SiluGateConfig {
    /// Total number of elements.
    pub n: usize,
    /// Threads per block.
    pub threads_per_block: u32,
}

impl SiluGateConfig {
    /// Create a configuration for the given element count.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if `n` is zero.
    pub fn new(n: usize) -> Result<Self> {
        if n == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "silu_gate element count must be non-zero".into(),
            }
            .into());
        }
        Ok(Self { n, threads_per_block: 256 })
    }

    /// Compute the CUDA grid dimensions.
    pub fn grid_dim(&self) -> (u32, u32, u32) {
        let blocks = (self.n as u32).div_ceil(self.threads_per_block);
        (blocks.min(65_535), 1, 1)
    }

    /// Compute the CUDA block dimensions.
    pub fn block_dim(&self) -> (u32, u32, u32) {
        (self.threads_per_block, 1, 1)
    }
}

// ---------------------------------------------------------------------------
// Scalar helpers (used by CPU fallback)
// ---------------------------------------------------------------------------

/// Scalar SiLU: `x / (1 + exp(-x))`.
#[inline]
fn silu_scalar(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Scalar GELU (tanh approximation).
#[inline]
fn gelu_scalar(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const COEFF: f32 = 0.044_715;
    let x3 = x * x * x;
    let inner = SQRT_2_OVER_PI * (x + COEFF * x3);
    0.5 * x * (1.0 + inner.tanh())
}

/// Scalar ReLU: `max(0, x)`.
#[inline]
fn relu_scalar(x: f32) -> f32 {
    x.max(0.0)
}

// ---------------------------------------------------------------------------
// Input validation
// ---------------------------------------------------------------------------

/// Validate that `input` and `output` lengths are sufficient.
fn validate_activation_buffers(input: &[f32], output: &[f32], n: usize) -> Result<()> {
    if input.len() < n {
        return Err(KernelError::InvalidArguments {
            reason: format!("activation input length {} < expected {n}", input.len(),),
        }
        .into());
    }
    if output.len() < n {
        return Err(KernelError::InvalidArguments {
            reason: format!("activation output length {} < expected {n}", output.len(),),
        }
        .into());
    }
    Ok(())
}

/// Validate buffers for the fused SiLU-gate kernel.
fn validate_silu_gate_buffers(input: &[f32], gate: &[f32], output: &[f32], n: usize) -> Result<()> {
    validate_activation_buffers(input, output, n)?;
    if gate.len() < n {
        return Err(KernelError::InvalidArguments {
            reason: format!("silu_gate gate length {} < expected {n}", gate.len(),),
        }
        .into());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CPU fallback implementations
// ---------------------------------------------------------------------------

/// Element-wise activation on the CPU.
///
/// Applies the activation selected by [`ActivationConfig::activation`] to
/// each of the first `config.n` elements.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if the slice lengths are
/// smaller than `config.n`.
pub fn activation_cpu(input: &[f32], output: &mut [f32], config: &ActivationConfig) -> Result<()> {
    validate_activation_buffers(input, output, config.n)?;

    let apply: fn(f32) -> f32 = match config.activation {
        ActivationType::SiLU => silu_scalar,
        ActivationType::GELU => gelu_scalar,
        ActivationType::ReLU => relu_scalar,
    };

    for i in 0..config.n {
        output[i] = apply(input[i]);
    }
    Ok(())
}

/// Fused SiLU-gate on the CPU.
///
/// Computes `output[i] = silu(input[i]) * gate[i]` for the first
/// `config.n` elements.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if any slice is shorter than
/// `config.n`.
pub fn silu_gate_cpu(
    input: &[f32],
    gate: &[f32],
    output: &mut [f32],
    config: &SiluGateConfig,
) -> Result<()> {
    validate_silu_gate_buffers(input, gate, output, config.n)?;

    for i in 0..config.n {
        output[i] = silu_scalar(input[i]) * gate[i];
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CUDA dispatch (feature-gated)
// ---------------------------------------------------------------------------

/// Dispatch an activation kernel to the CUDA device via cudarc.
///
/// Compiles the PTX source at first invocation, transfers data to device,
/// launches the selected kernel, and copies output back.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_activation_cuda(
    input: &[f32],
    output: &mut [f32],
    config: &ActivationConfig,
) -> Result<()> {
    use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::compile_ptx;

    validate_activation_buffers(input, output, config.n)?;

    log::debug!(
        "Activation CUDA dispatch: type={:?}, n={}, grid={:?}",
        config.activation,
        config.n,
        config.grid_dim(),
    );

    let ctx = CudaContext::new(0).map_err(|e| KernelError::GpuError {
        reason: format!("failed to acquire CUDA device 0: {e:?}"),
    })?;
    let stream = ctx.default_stream();

    let ptx = compile_ptx(ACTIVATION_KERNEL_SRC).map_err(|e| KernelError::GpuError {
        reason: format!("NVRTC compilation failed: {e:?}"),
    })?;
    let module = ctx.load_module(ptx).map_err(|e| KernelError::GpuError {
        reason: format!("failed to load PTX module: {e:?}"),
    })?;
    let func = module.load_function(config.activation.kernel_name()).map_err(|e| {
        KernelError::GpuError {
            reason: format!("{} function not found: {e:?}", config.activation.kernel_name()),
        }
    })?;

    let input_dev = stream.memcpy_stod(input).map_err(|e| KernelError::GpuError {
        reason: format!("failed to copy input to device: {e:?}"),
    })?;
    let mut output_dev: CudaSlice<f32> = stream.alloc_zeros(config.n).map_err(|e| {
        KernelError::GpuError { reason: format!("failed to allocate output on device: {e:?}") }
    })?;

    let (gx, gy, gz) = config.grid_dim();
    let (bx, by, bz) = config.block_dim();
    let launch_cfg =
        LaunchConfig { grid_dim: (gx, gy, gz), block_dim: (bx, by, bz), shared_mem_bytes: 0 };
    let n_arg = config.n as i32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&input_dev);
    builder.arg(&mut output_dev);
    builder.arg(&n_arg);

    // Safety: kernel signature matches the CUDA source; buffers are
    // correctly sized as validated above.
    unsafe { builder.launch(launch_cfg) }.map_err(|e| KernelError::GpuError {
        reason: format!("CUDA kernel launch failed: {e:?}"),
    })?;

    stream.synchronize().map_err(|e| KernelError::GpuError {
        reason: format!("stream synchronize failed: {e:?}"),
    })?;

    let host: Vec<f32> = stream.memcpy_dtov(&output_dev).map_err(|e| KernelError::GpuError {
        reason: format!("failed to copy output from device: {e:?}"),
    })?;
    output[..config.n].copy_from_slice(&host[..config.n]);

    Ok(())
}

/// Dispatch the fused SiLU-gate kernel to CUDA.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_silu_gate_cuda(
    input: &[f32],
    gate: &[f32],
    output: &mut [f32],
    config: &SiluGateConfig,
) -> Result<()> {
    use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::compile_ptx;

    validate_silu_gate_buffers(input, gate, output, config.n)?;

    log::debug!("SiLU-gate CUDA dispatch: n={}, grid={:?}", config.n, config.grid_dim(),);

    let ctx = CudaContext::new(0).map_err(|e| KernelError::GpuError {
        reason: format!("failed to acquire CUDA device 0: {e:?}"),
    })?;
    let stream = ctx.default_stream();

    let ptx = compile_ptx(ACTIVATION_KERNEL_SRC).map_err(|e| KernelError::GpuError {
        reason: format!("NVRTC compilation failed: {e:?}"),
    })?;
    let module = ctx.load_module(ptx).map_err(|e| KernelError::GpuError {
        reason: format!("failed to load PTX module: {e:?}"),
    })?;
    let func = module.load_function("silu_gate_f32").map_err(|e| KernelError::GpuError {
        reason: format!("silu_gate_f32 function not found: {e:?}"),
    })?;

    let input_dev = stream.memcpy_stod(input).map_err(|e| KernelError::GpuError {
        reason: format!("failed to copy input to device: {e:?}"),
    })?;
    let gate_dev = stream.memcpy_stod(gate).map_err(|e| KernelError::GpuError {
        reason: format!("failed to copy gate to device: {e:?}"),
    })?;
    let mut output_dev: CudaSlice<f32> = stream.alloc_zeros(config.n).map_err(|e| {
        KernelError::GpuError { reason: format!("failed to allocate output on device: {e:?}") }
    })?;

    let (gx, gy, gz) = config.grid_dim();
    let (bx, by, bz) = config.block_dim();
    let launch_cfg =
        LaunchConfig { grid_dim: (gx, gy, gz), block_dim: (bx, by, bz), shared_mem_bytes: 0 };
    let n_arg = config.n as i32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&input_dev);
    builder.arg(&gate_dev);
    builder.arg(&mut output_dev);
    builder.arg(&n_arg);

    // Safety: kernel signature matches the CUDA source; buffers are
    // correctly sized as validated above.
    unsafe { builder.launch(launch_cfg) }.map_err(|e| KernelError::GpuError {
        reason: format!("CUDA kernel launch failed: {e:?}"),
    })?;

    stream.synchronize().map_err(|e| KernelError::GpuError {
        reason: format!("stream synchronize failed: {e:?}"),
    })?;

    let host: Vec<f32> = stream.memcpy_dtov(&output_dev).map_err(|e| KernelError::GpuError {
        reason: format!("failed to copy output from device: {e:?}"),
    })?;
    output[..config.n].copy_from_slice(&host[..config.n]);

    Ok(())
}

// ---------------------------------------------------------------------------
// Unified dispatch entry points
// ---------------------------------------------------------------------------

/// Launch an element-wise activation with automatic CPU/GPU dispatch.
///
/// When compiled with `gpu` or `cuda` features **and** a CUDA device is
/// available at runtime, the kernel runs on the GPU.  Otherwise the CPU
/// fallback is used.
///
/// # Arguments
///
/// * `input`  — Input tensor (FP32, at least `config.n` elements)
/// * `output` — Output buffer (FP32, at least `config.n` elements)
/// * `config` — Launch configuration including activation type
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if buffer sizes are too small.
pub fn launch_activation(
    input: &[f32],
    output: &mut [f32],
    config: &ActivationConfig,
) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        match launch_activation_cuda(input, output, config) {
            Ok(()) => {
                log::debug!(
                    "Activation {:?} completed on CUDA (n={})",
                    config.activation,
                    config.n,
                );
                return Ok(());
            }
            Err(e) => {
                log::warn!("CUDA activation failed, falling back to CPU: {e}");
            }
        }
    }

    log::debug!("Activation {:?} CPU fallback (n={})", config.activation, config.n,);
    activation_cpu(input, output, config)
}

/// Launch the fused SiLU-gate kernel with automatic CPU/GPU dispatch.
///
/// Computes `output[i] = silu(input[i]) * gate[i]` in a single pass.
///
/// # Arguments
///
/// * `input`  — Input tensor (FP32)
/// * `gate`   — Gate tensor (FP32, same length)
/// * `output` — Output buffer (FP32)
/// * `config` — Launch configuration
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if any buffer is too small.
pub fn launch_silu_gate(
    input: &[f32],
    gate: &[f32],
    output: &mut [f32],
    config: &SiluGateConfig,
) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        match launch_silu_gate_cuda(input, gate, output, config) {
            Ok(()) => {
                log::debug!("SiLU-gate completed on CUDA (n={})", config.n,);
                return Ok(());
            }
            Err(e) => {
                log::warn!("CUDA SiLU-gate failed, falling back to CPU: {e}");
            }
        }
    }

    log::debug!("SiLU-gate CPU fallback (n={})", config.n);
    silu_gate_cpu(input, gate, output, config)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Config tests -------------------------------------------------------

    #[test]
    fn test_activation_config_basic() {
        let cfg = ActivationConfig::new(1024, ActivationType::SiLU).unwrap();
        assert_eq!(cfg.n, 1024);
        assert_eq!(cfg.threads_per_block, 256);
        assert_eq!(cfg.activation, ActivationType::SiLU);
    }

    #[test]
    fn test_activation_config_rejects_zero() {
        assert!(ActivationConfig::new(0, ActivationType::ReLU).is_err());
    }

    #[test]
    fn test_activation_config_grid_dim_small() {
        let cfg = ActivationConfig::new(100, ActivationType::GELU).unwrap();
        // ceil(100 / 256) = 1
        assert_eq!(cfg.grid_dim(), (1, 1, 1));
        assert_eq!(cfg.block_dim(), (256, 1, 1));
    }

    #[test]
    fn test_activation_config_grid_dim_large() {
        // 256 * 65535 = 16_776_960; anything above should cap at 65535
        let cfg = ActivationConfig::new(20_000_000, ActivationType::SiLU).unwrap();
        assert_eq!(cfg.grid_dim().0, 65_535);
    }

    #[test]
    fn test_silu_gate_config_basic() {
        let cfg = SiluGateConfig::new(512).unwrap();
        assert_eq!(cfg.n, 512);
        assert_eq!(cfg.threads_per_block, 256);
    }

    #[test]
    fn test_silu_gate_config_rejects_zero() {
        assert!(SiluGateConfig::new(0).is_err());
    }

    #[test]
    fn test_kernel_name_mapping() {
        assert_eq!(ActivationType::SiLU.kernel_name(), "silu_f32");
        assert_eq!(ActivationType::GELU.kernel_name(), "gelu_f32");
        assert_eq!(ActivationType::ReLU.kernel_name(), "relu_f32");
    }

    // -- SiLU CPU correctness -----------------------------------------------

    #[test]
    fn test_cpu_silu_known_values() {
        let cfg = ActivationConfig::new(5, ActivationType::SiLU).unwrap();
        let input = [0.0, 1.0, -1.0, 2.0, -2.0];
        let mut output = [0.0f32; 5];
        activation_cpu(&input, &mut output, &cfg).unwrap();

        // silu(0) = 0
        assert!(output[0].abs() < 1e-6, "silu(0)={}", output[0]);
        // silu(1) = 1/(1+e^-1) ≈ 0.7311
        assert!((output[1] - 0.7311).abs() < 1e-3, "silu(1)={}", output[1]);
        // silu(-1) = -1/(1+e^1) ≈ -0.2689
        assert!((output[2] - (-0.2689)).abs() < 1e-3, "silu(-1)={}", output[2]);
        // silu(x) is odd-symmetric-ish: silu(-x) = -x * σ(-x)
        // silu(2) > 0 and silu(-2) < 0
        assert!(output[3] > 0.0);
        assert!(output[4] < 0.0);
    }

    #[test]
    fn test_cpu_silu_zero() {
        let cfg = ActivationConfig::new(1, ActivationType::SiLU).unwrap();
        let input = [0.0f32];
        let mut output = [999.0f32];
        activation_cpu(&input, &mut output, &cfg).unwrap();
        assert!(output[0].abs() < 1e-7);
    }

    #[test]
    fn test_cpu_silu_large_positive() {
        let cfg = ActivationConfig::new(1, ActivationType::SiLU).unwrap();
        let input = [100.0f32];
        let mut output = [0.0f32];
        activation_cpu(&input, &mut output, &cfg).unwrap();
        // silu(100) ≈ 100 (sigmoid saturates to 1)
        assert!((output[0] - 100.0).abs() < 1e-3, "silu(100)={}", output[0]);
    }

    #[test]
    fn test_cpu_silu_large_negative() {
        let cfg = ActivationConfig::new(1, ActivationType::SiLU).unwrap();
        let input = [-100.0f32];
        let mut output = [999.0f32];
        activation_cpu(&input, &mut output, &cfg).unwrap();
        // silu(-100) ≈ 0 (sigmoid saturates to 0)
        assert!(output[0].abs() < 1e-3, "silu(-100)={}", output[0]);
    }

    // -- GELU CPU correctness -----------------------------------------------

    #[test]
    fn test_cpu_gelu_known_values() {
        let cfg = ActivationConfig::new(3, ActivationType::GELU).unwrap();
        let input = [0.0, 1.0, -1.0];
        let mut output = [0.0f32; 3];
        activation_cpu(&input, &mut output, &cfg).unwrap();

        // gelu(0) = 0
        assert!(output[0].abs() < 1e-6, "gelu(0)={}", output[0]);
        // gelu(1) ≈ 0.8412
        assert!((output[1] - 0.8412).abs() < 1e-3, "gelu(1)={}", output[1]);
        // gelu(-1) ≈ -0.1588
        assert!((output[2] - (-0.1588)).abs() < 1e-3, "gelu(-1)={}", output[2]);
    }

    #[test]
    fn test_cpu_gelu_large_positive() {
        let cfg = ActivationConfig::new(1, ActivationType::GELU).unwrap();
        let input = [10.0f32];
        let mut output = [0.0f32];
        activation_cpu(&input, &mut output, &cfg).unwrap();
        // gelu(10) ≈ 10 (tanh saturates to 1)
        assert!((output[0] - 10.0).abs() < 1e-3, "gelu(10)={}", output[0]);
    }

    #[test]
    fn test_cpu_gelu_large_negative() {
        let cfg = ActivationConfig::new(1, ActivationType::GELU).unwrap();
        let input = [-10.0f32];
        let mut output = [999.0f32];
        activation_cpu(&input, &mut output, &cfg).unwrap();
        // gelu(-10) ≈ 0 (tanh saturates to -1)
        assert!(output[0].abs() < 1e-3, "gelu(-10)={}", output[0]);
    }

    // -- ReLU CPU correctness -----------------------------------------------

    #[test]
    fn test_cpu_relu_known_values() {
        let cfg = ActivationConfig::new(5, ActivationType::ReLU).unwrap();
        let input = [-2.0, -0.5, 0.0, 0.5, 2.0];
        let mut output = [999.0f32; 5];
        activation_cpu(&input, &mut output, &cfg).unwrap();

        assert!((output[0]).abs() < 1e-7); // relu(-2) = 0
        assert!((output[1]).abs() < 1e-7); // relu(-0.5) = 0
        assert!((output[2]).abs() < 1e-7); // relu(0) = 0
        assert!((output[3] - 0.5).abs() < 1e-7); // relu(0.5) = 0.5
        assert!((output[4] - 2.0).abs() < 1e-7); // relu(2) = 2
    }

    #[test]
    fn test_cpu_relu_all_negative() {
        let cfg = ActivationConfig::new(4, ActivationType::ReLU).unwrap();
        let input = [-5.0, -1.0, -0.001, -100.0];
        let mut output = [999.0f32; 4];
        activation_cpu(&input, &mut output, &cfg).unwrap();
        for &v in &output {
            assert!(v.abs() < 1e-7, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_cpu_relu_all_positive() {
        let cfg = ActivationConfig::new(3, ActivationType::ReLU).unwrap();
        let input = [1.0, 42.0, 0.001];
        let mut output = [0.0f32; 3];
        activation_cpu(&input, &mut output, &cfg).unwrap();
        for (i, &v) in output.iter().enumerate() {
            assert!((v - input[i]).abs() < 1e-7, "relu({})={v}", input[i]);
        }
    }

    // -- Edge cases (shared across activations) -----------------------------

    #[test]
    fn test_cpu_activation_nan_propagation() {
        // SiLU and GELU propagate NaN through their computations.
        for act in [ActivationType::SiLU, ActivationType::GELU] {
            let cfg = ActivationConfig::new(1, act).unwrap();
            let input = [f32::NAN];
            let mut output = [0.0f32];
            activation_cpu(&input, &mut output, &cfg).unwrap();
            assert!(output[0].is_nan(), "{act:?}(NaN) should be NaN, got {}", output[0]);
        }
        // ReLU uses f32::max which returns the non-NaN operand per IEEE 754
        // max semantics in Rust, so relu(NaN) = max(0, NaN) = 0.
        let cfg = ActivationConfig::new(1, ActivationType::ReLU).unwrap();
        let input = [f32::NAN];
        let mut output = [999.0f32];
        activation_cpu(&input, &mut output, &cfg).unwrap();
        assert!(output[0].abs() < 1e-7, "relu(NaN) expected 0 (IEEE max), got {}", output[0]);
    }

    #[test]
    fn test_cpu_activation_single_element() {
        for act in [ActivationType::SiLU, ActivationType::GELU, ActivationType::ReLU] {
            let cfg = ActivationConfig::new(1, act).unwrap();
            let input = [1.0f32];
            let mut output = [0.0f32];
            activation_cpu(&input, &mut output, &cfg).unwrap();
            assert!(output[0].is_finite(), "{act:?}(1.0) not finite");
        }
    }

    #[test]
    fn test_cpu_activation_large_vector() {
        let n = 8192;
        for act in [ActivationType::SiLU, ActivationType::GELU, ActivationType::ReLU] {
            let cfg = ActivationConfig::new(n, act).unwrap();
            let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 40.0).collect();
            let mut output = vec![0.0f32; n];
            activation_cpu(&input, &mut output, &cfg).unwrap();
            for (i, &v) in output.iter().enumerate() {
                assert!(v.is_finite(), "{act:?} non-finite at index {i}: {v}");
            }
        }
    }

    // -- Buffer validation --------------------------------------------------

    #[test]
    fn test_cpu_activation_rejects_short_input() {
        let cfg = ActivationConfig::new(4, ActivationType::SiLU).unwrap();
        let input = [1.0f32; 2]; // need 4
        let mut output = [0.0f32; 4];
        assert!(activation_cpu(&input, &mut output, &cfg).is_err());
    }

    #[test]
    fn test_cpu_activation_rejects_short_output() {
        let cfg = ActivationConfig::new(4, ActivationType::GELU).unwrap();
        let input = [1.0f32; 4];
        let mut output = [0.0f32; 2]; // need 4
        assert!(activation_cpu(&input, &mut output, &cfg).is_err());
    }

    // -- Fused SiLU-gate CPU ------------------------------------------------

    #[test]
    fn test_cpu_silu_gate_basic() {
        let cfg = SiluGateConfig::new(4).unwrap();
        let input = [1.0f32, 2.0, -1.0, 0.0];
        let gate = [1.0f32, 0.5, 2.0, 3.0];
        let mut output = [0.0f32; 4];
        silu_gate_cpu(&input, &gate, &mut output, &cfg).unwrap();

        for i in 0..4 {
            let expected = silu_scalar(input[i]) * gate[i];
            assert!(
                (output[i] - expected).abs() < 1e-6,
                "elem {i}: expected {expected}, got {}",
                output[i]
            );
        }
    }

    #[test]
    fn test_cpu_silu_gate_zero_gate() {
        let cfg = SiluGateConfig::new(3).unwrap();
        let input = [1.0f32, 2.0, 3.0];
        let gate = [0.0f32; 3];
        let mut output = [999.0f32; 3];
        silu_gate_cpu(&input, &gate, &mut output, &cfg).unwrap();
        for &v in &output {
            assert!(v.abs() < 1e-7, "expected 0 with zero gate, got {v}");
        }
    }

    #[test]
    fn test_cpu_silu_gate_identity_gate() {
        // gate = 1.0 should match plain silu
        let n = 5;
        let cfg_gate = SiluGateConfig::new(n).unwrap();
        let cfg_silu = ActivationConfig::new(n, ActivationType::SiLU).unwrap();
        let input = [0.5f32, -0.5, 1.0, -1.0, 0.0];
        let gate = [1.0f32; 5];
        let mut out_gate = [0.0f32; 5];
        let mut out_silu = [0.0f32; 5];

        silu_gate_cpu(&input, &gate, &mut out_gate, &cfg_gate).unwrap();
        activation_cpu(&input, &mut out_silu, &cfg_silu).unwrap();

        for i in 0..n {
            assert!(
                (out_gate[i] - out_silu[i]).abs() < 1e-7,
                "silu_gate with gate=1 differs from silu at {i}"
            );
        }
    }

    #[test]
    fn test_cpu_silu_gate_rejects_short_gate() {
        let cfg = SiluGateConfig::new(4).unwrap();
        let input = [1.0f32; 4];
        let gate = [1.0f32; 2]; // too short
        let mut output = [0.0f32; 4];
        assert!(silu_gate_cpu(&input, &gate, &mut output, &cfg).is_err());
    }

    // -- Unified dispatch (CPU path on non-GPU builds) ----------------------

    #[test]
    fn test_launch_activation_dispatches_cpu_silu() {
        let cfg = ActivationConfig::new(4, ActivationType::SiLU).unwrap();
        let input = [1.0f32, -1.0, 0.0, 2.0];
        let mut output = [0.0f32; 4];

        launch_activation(&input, &mut output, &cfg).unwrap();

        let mut expected = [0.0f32; 4];
        activation_cpu(&input, &mut expected, &cfg).unwrap();
        for i in 0..4 {
            assert!((output[i] - expected[i]).abs() < 1e-7, "dispatch mismatch at {i}");
        }
    }

    #[test]
    fn test_launch_activation_dispatches_cpu_gelu() {
        let cfg = ActivationConfig::new(3, ActivationType::GELU).unwrap();
        let input = [1.0f32, -1.0, 0.5];
        let mut output = [0.0f32; 3];

        launch_activation(&input, &mut output, &cfg).unwrap();

        let mut expected = [0.0f32; 3];
        activation_cpu(&input, &mut expected, &cfg).unwrap();
        for i in 0..3 {
            assert!((output[i] - expected[i]).abs() < 1e-7, "dispatch mismatch at {i}");
        }
    }

    #[test]
    fn test_launch_activation_dispatches_cpu_relu() {
        let cfg = ActivationConfig::new(5, ActivationType::ReLU).unwrap();
        let input = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut output = [0.0f32; 5];

        launch_activation(&input, &mut output, &cfg).unwrap();

        let mut expected = [0.0f32; 5];
        activation_cpu(&input, &mut expected, &cfg).unwrap();
        for i in 0..5 {
            assert!((output[i] - expected[i]).abs() < 1e-7, "dispatch mismatch at {i}");
        }
    }

    #[test]
    fn test_launch_silu_gate_dispatches_cpu() {
        let cfg = SiluGateConfig::new(4).unwrap();
        let input = [1.0f32, 2.0, -1.0, 0.0];
        let gate = [0.5f32, 1.0, 2.0, 0.0];
        let mut output = [0.0f32; 4];

        launch_silu_gate(&input, &gate, &mut output, &cfg).unwrap();

        let mut expected = [0.0f32; 4];
        silu_gate_cpu(&input, &gate, &mut expected, &cfg).unwrap();
        for i in 0..4 {
            assert!((output[i] - expected[i]).abs() < 1e-7, "dispatch mismatch at {i}");
        }
    }

    // -- GPU-only tests (skipped without CUDA hardware) ---------------------

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_silu_launch() {
        let cfg = ActivationConfig::new(4096, ActivationType::SiLU).unwrap();
        let input: Vec<f32> = (0..4096).map(|i| (i as f32) * 0.01 - 20.0).collect();
        let mut output = vec![0.0f32; 4096];
        let result = launch_activation(&input, &mut output, &cfg);
        assert!(result.is_ok(), "CUDA SiLU launch failed: {result:?}");
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_gelu_launch() {
        let cfg = ActivationConfig::new(4096, ActivationType::GELU).unwrap();
        let input: Vec<f32> = (0..4096).map(|i| (i as f32) * 0.01 - 20.0).collect();
        let mut output = vec![0.0f32; 4096];
        let result = launch_activation(&input, &mut output, &cfg);
        assert!(result.is_ok(), "CUDA GELU launch failed: {result:?}");
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_silu_gate_launch() {
        let cfg = SiluGateConfig::new(2048).unwrap();
        let input = vec![1.0f32; 2048];
        let gate = vec![0.5f32; 2048];
        let mut output = vec![0.0f32; 2048];
        let result = launch_silu_gate(&input, &gate, &mut output, &cfg);
        assert!(result.is_ok(), "CUDA SiLU-gate launch failed: {result:?}");
    }
}

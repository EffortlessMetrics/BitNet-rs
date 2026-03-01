//! Gating CUDA kernels with CPU fallback for transformer FFN layers.
//!
//! Gating mechanisms combine two projections element-wise:
//! `output = activation(gate) * up`
//!
//! Provides three fused gating operations:
//! - **SwiGLU**: `SiLU(gate) * up` — used in LLaMA, Mistral, etc.
//! - **GeGLU**: `GELU(gate) * up` — used in some GPT variants
//! - **ReGLU**: `ReLU(gate) * up` — simpler alternative
//!
//! # Kernel strategy
//!
//! Each gating op is element-wise with no inter-element dependencies.
//! CUDA kernels use grid-stride loops with 256 threads per block,
//! matching the convention in [`super::activations`].
//!
//! # CPU fallback
//!
//! [`gating_cpu`] provides a pure-Rust scalar implementation that
//! delegates to [`crate::cpu::gating`] for non-GPU environments.

use bitnet_common::{KernelError, Result};

// ---------------------------------------------------------------------------
// PTX source (compiled at runtime via NVRTC when `gpu`/`cuda` is active)
// ---------------------------------------------------------------------------

/// Inline CUDA C source for fused gating kernels.
///
/// Contains three kernels: `swiglu_f32`, `geglu_f32`, `reglu_f32`.
/// Each processes `n` elements using grid-stride loops, computing
/// `output[i] = activation(gate[i]) * up[i]`.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const GATING_KERNEL_SRC: &str = r#"
extern "C" __global__ void swiglu_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ output,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        float g = gate[i];
        float silu_g = g / (1.0f + expf(-g));
        output[i] = silu_g * up[i];
    }
}

extern "C" __global__ void geglu_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ output,
    int n)
{
    const float SQRT_2_OVER_PI = 0.7978845608f;
    const float COEFF = 0.044715f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        float g = gate[i];
        float g3 = g * g * g;
        float inner = SQRT_2_OVER_PI * (g + COEFF * g3);
        float gelu_g = 0.5f * g * (1.0f + tanhf(inner));
        output[i] = gelu_g * up[i];
    }
}

extern "C" __global__ void reglu_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ output,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        output[i] = fmaxf(0.0f, gate[i]) * up[i];
    }
}
"#;

// ---------------------------------------------------------------------------
// Gating type selector
// ---------------------------------------------------------------------------

/// Selects which gating function to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatingType {
    /// SwiGLU: `SiLU(gate) * up`
    SwiGLU,
    /// GeGLU: `GELU(gate) * up`
    GeGLU,
    /// ReGLU: `ReLU(gate) * up`
    ReGLU,
}

impl GatingType {
    /// CUDA kernel function name for this gating type.
    pub fn kernel_name(&self) -> &'static str {
        match self {
            Self::SwiGLU => "swiglu_f32",
            Self::GeGLU => "geglu_f32",
            Self::ReGLU => "reglu_f32",
        }
    }
}

// ---------------------------------------------------------------------------
// Launch configuration
// ---------------------------------------------------------------------------

/// Launch configuration for gating kernels.
#[derive(Debug, Clone)]
pub struct GatingConfig {
    /// Total number of elements to process.
    pub n: usize,
    /// Threads per block (default 256).
    pub threads_per_block: u32,
    /// Which gating function to apply.
    pub gating: GatingType,
}

impl GatingConfig {
    /// Create a configuration for the given element count.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if `n` is zero.
    pub fn new(n: usize, gating: GatingType) -> Result<Self> {
        if n == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "gating element count must be non-zero".into(),
            }
            .into());
        }
        Ok(Self { n, threads_per_block: 256, gating })
    }

    /// Compute the CUDA grid dimensions.
    ///
    /// Caps at 65 535 blocks; the grid-stride loop handles overflow.
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
// Input validation
// ---------------------------------------------------------------------------

/// Validate buffers for gating kernels.
fn validate_gating_buffers(gate: &[f32], up: &[f32], output: &[f32], n: usize) -> Result<()> {
    if gate.len() < n {
        return Err(KernelError::InvalidArguments {
            reason: format!("gating gate length {} < expected {n}", gate.len()),
        }
        .into());
    }
    if up.len() < n {
        return Err(KernelError::InvalidArguments {
            reason: format!("gating up length {} < expected {n}", up.len()),
        }
        .into());
    }
    if output.len() < n {
        return Err(KernelError::InvalidArguments {
            reason: format!("gating output length {} < expected {n}", output.len()),
        }
        .into());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CPU fallback
// ---------------------------------------------------------------------------

/// Gating on the CPU via [`crate::cpu::gating`].
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if buffer sizes are too small.
pub fn gating_cpu(
    gate: &[f32],
    up: &[f32],
    output: &mut [f32],
    config: &GatingConfig,
) -> Result<()> {
    validate_gating_buffers(gate, up, output, config.n)?;
    let cpu_gating = match config.gating {
        GatingType::SwiGLU => crate::cpu::gating::GatingType::SwiGLU,
        GatingType::GeGLU => crate::cpu::gating::GatingType::GeGLU,
        GatingType::ReGLU => crate::cpu::gating::GatingType::ReGLU,
    };
    crate::cpu::gating::apply_gating(
        cpu_gating,
        &gate[..config.n],
        &up[..config.n],
        &mut output[..config.n],
    )
}

// ---------------------------------------------------------------------------
// CUDA dispatch (feature-gated)
// ---------------------------------------------------------------------------

/// Dispatch a gating kernel to the CUDA device via cudarc.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_gating_cuda(
    gate: &[f32],
    up: &[f32],
    output: &mut [f32],
    config: &GatingConfig,
) -> Result<()> {
    use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::compile_ptx;

    validate_gating_buffers(gate, up, output, config.n)?;

    log::debug!(
        "Gating CUDA dispatch: type={:?}, n={}, grid={:?}",
        config.gating,
        config.n,
        config.grid_dim(),
    );

    let ctx = CudaContext::new(0).map_err(|e| KernelError::GpuError {
        reason: format!("failed to acquire CUDA device 0: {e:?}"),
    })?;
    let stream = ctx.default_stream();

    let ptx = compile_ptx(GATING_KERNEL_SRC).map_err(|e| KernelError::GpuError {
        reason: format!("NVRTC compilation failed: {e:?}"),
    })?;
    let module = ctx.load_module(ptx).map_err(|e| KernelError::GpuError {
        reason: format!("failed to load PTX module: {e:?}"),
    })?;
    let func =
        module.load_function(config.gating.kernel_name()).map_err(|e| KernelError::GpuError {
            reason: format!("{} function not found: {e:?}", config.gating.kernel_name()),
        })?;

    let gate_dev = stream.memcpy_stod(gate).map_err(|e| KernelError::GpuError {
        reason: format!("failed to copy gate to device: {e:?}"),
    })?;
    let up_dev = stream.memcpy_stod(up).map_err(|e| KernelError::GpuError {
        reason: format!("failed to copy up to device: {e:?}"),
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
    builder.arg(&gate_dev);
    builder.arg(&up_dev);
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
// Unified dispatch entry point
// ---------------------------------------------------------------------------

/// Launch a gating kernel with automatic CPU/GPU dispatch.
///
/// When compiled with `gpu` or `cuda` features **and** a CUDA device is
/// available at runtime, the kernel runs on the GPU. Otherwise the CPU
/// fallback is used.
///
/// # Arguments
///
/// * `gate`   — Gate projection tensor (FP32, at least `config.n` elements)
/// * `up`     — Up projection tensor (FP32, at least `config.n` elements)
/// * `output` — Output buffer (FP32, at least `config.n` elements)
/// * `config` — Launch configuration including gating type
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if buffer sizes are too small.
pub fn launch_gating(
    gate: &[f32],
    up: &[f32],
    output: &mut [f32],
    config: &GatingConfig,
) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        match launch_gating_cuda(gate, up, output, config) {
            Ok(()) => {
                log::debug!("Gating {:?} completed on CUDA (n={})", config.gating, config.n,);
                return Ok(());
            }
            Err(e) => {
                log::warn!("CUDA gating failed, falling back to CPU: {e}");
            }
        }
    }

    log::debug!("Gating {:?} CPU fallback (n={})", config.gating, config.n);
    gating_cpu(gate, up, output, config)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gating_config_basic() {
        let cfg = GatingConfig::new(1024, GatingType::SwiGLU).unwrap();
        assert_eq!(cfg.n, 1024);
        assert_eq!(cfg.threads_per_block, 256);
        assert_eq!(cfg.gating, GatingType::SwiGLU);
    }

    #[test]
    fn test_gating_config_rejects_zero() {
        assert!(GatingConfig::new(0, GatingType::SwiGLU).is_err());
        assert!(GatingConfig::new(0, GatingType::GeGLU).is_err());
        assert!(GatingConfig::new(0, GatingType::ReGLU).is_err());
    }

    #[test]
    fn test_gating_config_grid_dim_small() {
        let cfg = GatingConfig::new(100, GatingType::GeGLU).unwrap();
        assert_eq!(cfg.grid_dim(), (1, 1, 1));
        assert_eq!(cfg.block_dim(), (256, 1, 1));
    }

    #[test]
    fn test_gating_config_grid_dim_large() {
        let cfg = GatingConfig::new(20_000_000, GatingType::ReGLU).unwrap();
        assert_eq!(cfg.grid_dim().0, 65_535);
    }

    #[test]
    fn test_kernel_name_mapping() {
        assert_eq!(GatingType::SwiGLU.kernel_name(), "swiglu_f32");
        assert_eq!(GatingType::GeGLU.kernel_name(), "geglu_f32");
        assert_eq!(GatingType::ReGLU.kernel_name(), "reglu_f32");
    }

    #[test]
    fn test_gating_cpu_swiglu() {
        let cfg = GatingConfig::new(2, GatingType::SwiGLU).unwrap();
        let gate = [1.0f32, 0.0];
        let up = [2.0f32, 5.0];
        let mut out = [0.0f32; 2];
        gating_cpu(&gate, &up, &mut out, &cfg).unwrap();
        // SiLU(1)*2 ≈ 1.4622
        assert!((out[0] - 1.4622).abs() < 1e-3);
        // SiLU(0)*5 = 0
        assert!(out[1].abs() < 1e-7);
    }

    #[test]
    fn test_gating_cpu_geglu() {
        let cfg = GatingConfig::new(2, GatingType::GeGLU).unwrap();
        let gate = [1.0f32, 0.0];
        let up = [2.0f32, 5.0];
        let mut out = [0.0f32; 2];
        gating_cpu(&gate, &up, &mut out, &cfg).unwrap();
        // GELU(1)*2 ≈ 1.6824
        assert!((out[0] - 1.6824).abs() < 1e-3);
        assert!(out[1].abs() < 1e-7);
    }

    #[test]
    fn test_gating_cpu_reglu() {
        let cfg = GatingConfig::new(2, GatingType::ReGLU).unwrap();
        let gate = [1.0f32, -1.0];
        let up = [2.0f32, 5.0];
        let mut out = [0.0f32; 2];
        gating_cpu(&gate, &up, &mut out, &cfg).unwrap();
        assert!((out[0] - 2.0).abs() < 1e-7);
        assert!(out[1].abs() < 1e-7);
    }

    #[test]
    fn test_gating_cpu_validation() {
        let cfg = GatingConfig::new(4, GatingType::SwiGLU).unwrap();
        let gate = [1.0f32, 2.0]; // too short
        let up = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 4];
        assert!(gating_cpu(&gate, &up, &mut out, &cfg).is_err());
    }

    #[test]
    fn test_launch_gating_uses_cpu_fallback() {
        let cfg = GatingConfig::new(3, GatingType::SwiGLU).unwrap();
        let gate = [1.0f32, 0.0, -1.0];
        let up = [1.0f32, 1.0, 1.0];
        let mut out = [0.0f32; 3];
        launch_gating(&gate, &up, &mut out, &cfg).unwrap();
        // Just verify it runs without error and produces plausible output
        assert!((out[0] - 0.7311).abs() < 1e-3);
        assert!(out[1].abs() < 1e-7);
        assert!((out[2] - (-0.2689)).abs() < 1e-3);
    }
}

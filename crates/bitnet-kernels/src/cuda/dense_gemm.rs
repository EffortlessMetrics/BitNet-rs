//! Dense regular-LLM CUDA GEMM smoke fixture.
//!
//! This module is deliberately narrower than the existing dense matmul
//! scaffold. It provides the first fallback-free CUDA FP16 GEMM smoke/parity
//! fixture for the `dense_regular_llm_cuda` lane. It is not a BitNet packed
//! I2_S/QK256 proof and it is not a general dense GGUF inference path.

use bitnet_common::{KernelError, Result};

#[cfg(feature = "cuda")]
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::Ptx;
#[cfg(feature = "cuda")]
use std::sync::Arc;

use super::half_precision::f32_to_f16_bits;
use super::matmul::{MatmulConfig, MatmulDtype, matmul_f16_cpu};

/// Kernel ID recorded by dense regular-LLM CUDA GEMM receipts.
pub const CUDA_DENSE_F16_GEMM_KERNEL_ID: &str = "dense_f16_gemm_cuda";

/// Fixture ID for the first dense regular-LLM CUDA GEMM parity smoke.
pub const CUDA_DENSE_F16_GEMM_FIXTURE_ID: &str = "dense_f16_gemm_m2_n3_k4";

/// CPU reference backend recorded by dense CUDA parity receipts.
pub const CUDA_DENSE_GEMM_REFERENCE_BACKEND: &str = "amd-9950x3d-cpu-avx512";

/// RTX 5070 Ti CUDA backend recorded by dense CUDA parity receipts.
pub const CUDA_DENSE_GEMM_TARGET_BACKEND: &str = "nvidia-rtx-5070-ti-cuda";

/// Tolerance for the deterministic FP16 smoke fixture.
pub const CUDA_DENSE_F16_GEMM_TOLERANCE: f32 = 0.002;

#[cfg(feature = "cuda")]
const CUDA_DENSE_F16_GEMM_PTX: &str = r#"
.version 8.0
.target sm_80
.address_size 64

.visible .entry dense_f16_gemm_cuda(
    .param .u64 dense_f16_gemm_cuda_param_0,
    .param .u64 dense_f16_gemm_cuda_param_1,
    .param .u64 dense_f16_gemm_cuda_param_2,
    .param .u32 dense_f16_gemm_cuda_param_3,
    .param .u32 dense_f16_gemm_cuda_param_4,
    .param .u32 dense_f16_gemm_cuda_param_5
)
{
    .reg .pred  %p<2>;
    .reg .b16   %h<3>;
    .reg .b32   %r<18>;
    .reg .b64   %rd<12>;
    .reg .f32   %f<4>;

    ld.param.u64    %rd1, [dense_f16_gemm_cuda_param_0];
    ld.param.u64    %rd2, [dense_f16_gemm_cuda_param_1];
    ld.param.u64    %rd3, [dense_f16_gemm_cuda_param_2];
    ld.param.u32    %r1, [dense_f16_gemm_cuda_param_3];
    ld.param.u32    %r2, [dense_f16_gemm_cuda_param_4];
    ld.param.u32    %r3, [dense_f16_gemm_cuda_param_5];

    mov.u32         %r4, %ctaid.x;
    mov.u32         %r5, %ntid.x;
    mov.u32         %r6, %tid.x;
    mad.lo.s32      %r7, %r4, %r5, %r6;
    mul.lo.s32      %r8, %r1, %r2;
    setp.ge.s32     %p1, %r7, %r8;
    @%p1 bra        DONE;

    div.u32         %r9, %r7, %r2;
    rem.u32         %r10, %r7, %r2;
    mov.f32         %f1, 0f00000000;
    mov.u32         %r11, 0;

LOOP:
    setp.ge.s32     %p1, %r11, %r3;
    @%p1 bra        STORE;
    mad.lo.s32      %r12, %r9, %r3, %r11;
    mad.lo.s32      %r13, %r11, %r2, %r10;
    mul.wide.u32    %rd4, %r12, 2;
    add.s64         %rd5, %rd1, %rd4;
    mul.wide.u32    %rd6, %r13, 2;
    add.s64         %rd7, %rd2, %rd6;
    ld.global.u16   %h1, [%rd5];
    ld.global.u16   %h2, [%rd7];
    cvt.f32.f16     %f2, %h1;
    cvt.f32.f16     %f3, %h2;
    fma.rn.f32      %f1, %f2, %f3, %f1;
    add.s32         %r11, %r11, 1;
    bra             LOOP;

STORE:
    mul.wide.u32    %rd8, %r7, 4;
    add.s64         %rd9, %rd3, %rd8;
    st.global.f32   [%rd9], %f1;

DONE:
    ret;
}
"#;

/// CUDA execution counters for the dense FP16 GEMM smoke fixture.
#[derive(Debug, Clone, PartialEq)]
pub struct CudaDenseGemmStats {
    /// CUDA kernel identifier.
    pub kernel_id: &'static str,
    /// Number of CUDA kernel invocations.
    pub invocations: u64,
    /// CPU fallback invocations under strict CUDA.
    pub fallback_invocations: u64,
    /// Host-to-device bytes copied for the fixture.
    pub host_to_device_bytes: u64,
    /// Device-to-host bytes copied for the fixture.
    pub device_to_host_bytes: u64,
    /// CUDA kernel launches.
    pub kernel_launches: u64,
    /// Optional measured kernel time.
    pub kernel_time_ms: Option<f64>,
}

/// Dense CUDA GEMM parity result against the CPU reference.
#[derive(Debug, Clone, PartialEq)]
pub struct CudaDenseGemmParity {
    /// Fixture identifier.
    pub fixture_id: &'static str,
    /// CPU reference backend.
    pub reference_backend: &'static str,
    /// CUDA target backend.
    pub target_backend: &'static str,
    /// CUDA kernel identifier.
    pub kernel_id: &'static str,
    /// Maximum absolute error against the CPU reference.
    pub max_abs_error: f32,
    /// Mean absolute error against the CPU reference.
    pub mean_abs_error: f32,
    /// Fixture tolerance.
    pub tolerance: f32,
    /// Whether the fixture passed tolerance.
    pub passed: bool,
    /// CUDA execution counters.
    pub stats: CudaDenseGemmStats,
}

/// Aggregate CUDA execution counters for a persistent dense GEMM fixture session.
#[derive(Debug, Clone, PartialEq)]
pub struct CudaDenseGemmPersistentStats {
    /// CUDA kernel identifier.
    pub kernel_id: &'static str,
    /// Repeated fixture launches in the session.
    pub runs: u64,
    /// Number of CUDA kernel invocations.
    pub invocations: u64,
    /// CPU fallback invocations under strict CUDA.
    pub fallback_invocations: u64,
    /// Host-to-device bytes copied once while constructing the session.
    pub host_to_device_bytes: u64,
    /// Device-to-host bytes copied for parity checks across all runs.
    pub device_to_host_bytes: u64,
    /// CUDA kernel launches across the session.
    pub kernel_launches: u64,
    /// CUDA context creations for the session.
    pub context_creations: u64,
    /// CUDA module loads for the session.
    pub module_loads: u64,
    /// Input tensor uploads performed while constructing the session.
    pub input_uploads: u64,
    /// Output tensor allocations performed while constructing the session.
    pub output_allocations: u64,
    /// Reusable CUDA handles retained by the session.
    pub persistent_handle_count: u64,
    /// Per-run host-to-device bytes after the session is constructed.
    pub per_run_host_to_device_bytes: u64,
    /// Optional measured aggregate kernel time.
    pub kernel_time_ms: Option<f64>,
}

/// Persistent dense CUDA GEMM parity result against the CPU reference.
#[derive(Debug, Clone, PartialEq)]
pub struct CudaDenseGemmPersistentParity {
    /// Fixture identifier.
    pub fixture_id: &'static str,
    /// CPU reference backend.
    pub reference_backend: &'static str,
    /// CUDA target backend.
    pub target_backend: &'static str,
    /// CUDA kernel identifier.
    pub kernel_id: &'static str,
    /// Repeated fixture launches in the session.
    pub runs: u64,
    /// Maximum absolute error across all runs.
    pub max_abs_error: f32,
    /// Mean absolute error across all compared output values.
    pub mean_abs_error: f32,
    /// Fixture tolerance.
    pub tolerance: f32,
    /// Whether every run passed tolerance.
    pub passed: bool,
    /// Aggregate persistent-session counters.
    pub stats: CudaDenseGemmPersistentStats,
}

/// Deterministic dense FP16 GEMM smoke fixture.
pub fn dense_f16_gemm_fixture() -> Result<(Vec<u16>, Vec<u16>, MatmulConfig)> {
    let a_f32 = [1.0, -2.0, 0.5, 3.0, 0.0, 4.0, -1.0, 2.0];
    let b_f32 = [1.0, 0.5, -1.0, 2.0, -2.0, 0.0, -0.5, 1.0, 2.0, 3.0, 0.25, -1.0];
    let a = a_f32.iter().map(|value| f32_to_f16_bits(*value)).collect::<Vec<_>>();
    let b = b_f32.iter().map(|value| f32_to_f16_bits(*value)).collect::<Vec<_>>();
    let cfg = MatmulConfig::for_shape(2, 3, 4)?.with_dtype(MatmulDtype::F16);
    Ok((a, b, cfg))
}

/// CPU reference output for the deterministic dense FP16 GEMM smoke fixture.
pub fn dense_f16_gemm_cpu_reference() -> Result<Vec<f32>> {
    let (a, b, cfg) = dense_f16_gemm_fixture()?;
    let mut out = vec![0.0f32; cfg.m * cfg.n];
    matmul_f16_cpu(&a, &b, &mut out, &cfg)?;
    Ok(out)
}

/// Run the deterministic dense FP16 GEMM smoke fixture on CUDA and compare it
/// against the CPU reference.
///
/// # Errors
///
/// Returns an error if CUDA/NVRTC is unavailable, the fixture cannot launch, or
/// the fixture buffers are invalid.
pub fn run_dense_f16_gemm_cuda_parity(device_index: usize) -> Result<CudaDenseGemmParity> {
    let (a, b, cfg) = dense_f16_gemm_fixture()?;
    let expected = dense_f16_gemm_cpu_reference()?;
    let mut actual = vec![0.0f32; cfg.m * cfg.n];
    let stats = launch_dense_f16_gemm_cuda(device_index, &a, &b, &mut actual, &cfg)?;
    let (max_abs_error, mean_abs_error) = compare_outputs(&expected, &actual)?;
    Ok(CudaDenseGemmParity {
        fixture_id: CUDA_DENSE_F16_GEMM_FIXTURE_ID,
        reference_backend: CUDA_DENSE_GEMM_REFERENCE_BACKEND,
        target_backend: CUDA_DENSE_GEMM_TARGET_BACKEND,
        kernel_id: CUDA_DENSE_F16_GEMM_KERNEL_ID,
        max_abs_error,
        mean_abs_error,
        tolerance: CUDA_DENSE_F16_GEMM_TOLERANCE,
        passed: max_abs_error <= CUDA_DENSE_F16_GEMM_TOLERANCE,
        stats,
    })
}

/// Run repeated deterministic dense FP16 GEMM launches through one persistent
/// CUDA fixture session and compare every launch against the CPU reference.
///
/// # Errors
///
/// Returns an error if CUDA/NVRTC is unavailable, the fixture cannot launch, or
/// `runs` is zero.
pub fn run_dense_f16_gemm_cuda_persistent_parity(
    device_index: usize,
    runs: usize,
) -> Result<CudaDenseGemmPersistentParity> {
    if runs == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "dense CUDA GEMM persistent fixture requires at least one run".into(),
        }
        .into());
    }

    #[cfg(feature = "cuda")]
    {
        return run_dense_f16_gemm_cuda_persistent_parity_impl(device_index, runs);
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = device_index;
        Err(KernelError::DeviceUnavailable {
            reason: "CUDA dense FP16 GEMM persistent fixture requires the cuda feature".to_string(),
        }
        .into())
    }
}

/// Launch the dense FP16 GEMM CUDA smoke kernel.
///
/// This is a strict CUDA fixture: unsupported shapes return an error instead
/// of falling back to CPU.
pub fn launch_dense_f16_gemm_cuda(
    device_index: usize,
    a: &[u16],
    b: &[u16],
    output: &mut [f32],
    config: &MatmulConfig,
) -> Result<CudaDenseGemmStats> {
    validate_dense_f16_gemm_inputs(a, b, output, config)?;

    #[cfg(feature = "cuda")]
    {
        return launch_dense_f16_gemm_cuda_impl(device_index, a, b, output, config);
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = device_index;
        Err(KernelError::DeviceUnavailable {
            reason: "CUDA dense FP16 GEMM requires the cuda feature".to_string(),
        }
        .into())
    }
}

fn validate_dense_f16_gemm_inputs(
    a: &[u16],
    b: &[u16],
    output: &[f32],
    config: &MatmulConfig,
) -> Result<()> {
    if config.dtype != MatmulDtype::F16 {
        return Err(KernelError::InvalidArguments {
            reason: "dense CUDA GEMM smoke fixture requires MatmulDtype::F16".into(),
        }
        .into());
    }
    if config.batch_size != 1 {
        return Err(KernelError::InvalidArguments {
            reason: "dense CUDA GEMM smoke fixture currently supports batch_size=1".into(),
        }
        .into());
    }
    if config.transpose_a || config.transpose_b {
        return Err(KernelError::InvalidArguments {
            reason: "dense CUDA GEMM smoke fixture currently requires non-transposed operands"
                .into(),
        }
        .into());
    }
    let a_required = checked_mul(config.m, config.k, "A")?;
    let b_required = checked_mul(config.k, config.n, "B")?;
    let out_required = checked_mul(config.m, config.n, "output")?;
    if a.len() < a_required {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "dense CUDA GEMM A buffer too short: expected >= {a_required}, got {}",
                a.len()
            ),
        }
        .into());
    }
    if b.len() < b_required {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "dense CUDA GEMM B buffer too short: expected >= {b_required}, got {}",
                b.len()
            ),
        }
        .into());
    }
    if output.len() < out_required {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "dense CUDA GEMM output buffer too short: expected >= {out_required}, got {}",
                output.len()
            ),
        }
        .into());
    }
    validate_i32_arg(config.m, "m")?;
    validate_i32_arg(config.n, "n")?;
    validate_i32_arg(config.k, "k")?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn launch_dense_f16_gemm_cuda_impl(
    device_index: usize,
    a: &[u16],
    b: &[u16],
    output: &mut [f32],
    config: &MatmulConfig,
) -> Result<CudaDenseGemmStats> {
    let ctx = CudaContext::new(device_index).map_err(|err| KernelError::GpuError {
        reason: format!("failed to create CUDA context for dense FP16 GEMM: {err:?}"),
    })?;
    let stream = ctx.default_stream();
    let ptx = compile_dense_f16_gemm_ptx()?;
    let module = ctx.load_module(ptx).map_err(|err| KernelError::GpuError {
        reason: format!("failed to load dense FP16 GEMM CUDA module: {err:?}"),
    })?;
    let function = module.load_function(CUDA_DENSE_F16_GEMM_KERNEL_ID).map_err(|err| {
        KernelError::GpuError {
            reason: format!("failed to load dense FP16 GEMM CUDA kernel: {err:?}"),
        }
    })?;

    let a_len = checked_mul(config.m, config.k, "A")?;
    let b_len = checked_mul(config.k, config.n, "B")?;
    let output_len = checked_mul(config.m, config.n, "output")?;

    let a_dev = stream.memcpy_stod(&a[..a_len]).map_err(|err| KernelError::GpuError {
        reason: format!("failed to copy dense FP16 GEMM A to device: {err:?}"),
    })?;
    let b_dev = stream.memcpy_stod(&b[..b_len]).map_err(|err| KernelError::GpuError {
        reason: format!("failed to copy dense FP16 GEMM B to device: {err:?}"),
    })?;
    let mut output_dev: CudaSlice<f32> =
        stream.alloc_zeros(output_len).map_err(|err| KernelError::GpuError {
            reason: format!("failed to allocate dense FP16 GEMM output on device: {err:?}"),
        })?;

    let threads_per_block = 128u32;
    let launch_config = LaunchConfig {
        grid_dim: ((output_len as u32).div_ceil(threads_per_block), 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(&function);
    builder.arg(&a_dev);
    builder.arg(&b_dev);
    builder.arg(&mut output_dev);
    let m_arg = i32::try_from(config.m).map_err(|_| KernelError::InvalidArguments {
        reason: format!("dense CUDA GEMM m exceeds i32: {}", config.m),
    })?;
    let n_arg = i32::try_from(config.n).map_err(|_| KernelError::InvalidArguments {
        reason: format!("dense CUDA GEMM n exceeds i32: {}", config.n),
    })?;
    let k_arg = i32::try_from(config.k).map_err(|_| KernelError::InvalidArguments {
        reason: format!("dense CUDA GEMM k exceeds i32: {}", config.k),
    })?;
    builder.arg(&m_arg);
    builder.arg(&n_arg);
    builder.arg(&k_arg);

    unsafe { builder.launch(launch_config) }.map_err(|err| KernelError::GpuError {
        reason: format!("failed to launch dense FP16 GEMM CUDA kernel: {err:?}"),
    })?;
    stream.synchronize().map_err(|err| KernelError::GpuError {
        reason: format!("failed to synchronize dense FP16 GEMM CUDA kernel: {err:?}"),
    })?;

    let output_host: Vec<f32> =
        stream.memcpy_dtov(&output_dev).map_err(|err| KernelError::GpuError {
            reason: format!("failed to copy dense FP16 GEMM output from device: {err:?}"),
        })?;
    output[..output_len].copy_from_slice(&output_host[..output_len]);

    Ok(CudaDenseGemmStats {
        kernel_id: CUDA_DENSE_F16_GEMM_KERNEL_ID,
        invocations: 1,
        fallback_invocations: 0,
        host_to_device_bytes: bytes_for::<u16>(a_len + b_len)?,
        device_to_host_bytes: bytes_for::<f32>(output_len)?,
        kernel_launches: 1,
        kernel_time_ms: None,
    })
}

#[cfg(feature = "cuda")]
fn run_dense_f16_gemm_cuda_persistent_parity_impl(
    device_index: usize,
    runs: usize,
) -> Result<CudaDenseGemmPersistentParity> {
    let (a, b, cfg) = dense_f16_gemm_fixture()?;
    let expected = dense_f16_gemm_cpu_reference()?;
    let mut session = CudaDenseGemmFixtureSession::new(device_index, &a, &b, &cfg)?;

    let mut max_abs_error = 0.0f32;
    let mut sum_abs_error = 0.0f32;
    let mut compared_values = 0usize;
    for _ in 0..runs {
        let mut actual = vec![0.0f32; cfg.m * cfg.n];
        session.launch(&mut actual)?;
        let (run_max_abs, run_mean_abs) = compare_outputs(&expected, &actual)?;
        max_abs_error = max_abs_error.max(run_max_abs);
        sum_abs_error += run_mean_abs * expected.len() as f32;
        compared_values += expected.len();
    }

    let runs_u64 = u64::try_from(runs).map_err(|_| KernelError::InvalidArguments {
        reason: format!("dense CUDA GEMM persistent run count exceeds u64: {runs}"),
    })?;
    let mean_abs_error =
        if compared_values == 0 { 0.0 } else { sum_abs_error / compared_values as f32 };
    let passed = max_abs_error <= CUDA_DENSE_F16_GEMM_TOLERANCE;

    Ok(CudaDenseGemmPersistentParity {
        fixture_id: CUDA_DENSE_F16_GEMM_FIXTURE_ID,
        reference_backend: CUDA_DENSE_GEMM_REFERENCE_BACKEND,
        target_backend: CUDA_DENSE_GEMM_TARGET_BACKEND,
        kernel_id: CUDA_DENSE_F16_GEMM_KERNEL_ID,
        runs: runs_u64,
        max_abs_error,
        mean_abs_error,
        tolerance: CUDA_DENSE_F16_GEMM_TOLERANCE,
        passed,
        stats: CudaDenseGemmPersistentStats {
            kernel_id: CUDA_DENSE_F16_GEMM_KERNEL_ID,
            runs: runs_u64,
            invocations: runs_u64,
            fallback_invocations: 0,
            host_to_device_bytes: session.host_to_device_bytes,
            device_to_host_bytes: session.device_to_host_bytes * runs_u64,
            kernel_launches: runs_u64,
            context_creations: 1,
            module_loads: 1,
            input_uploads: 2,
            output_allocations: 1,
            persistent_handle_count: 3,
            per_run_host_to_device_bytes: 0,
            kernel_time_ms: None,
        },
    })
}

#[cfg(feature = "cuda")]
struct CudaDenseGemmFixtureSession {
    stream: Arc<CudaStream>,
    function: CudaFunction,
    a_dev: CudaSlice<u16>,
    b_dev: CudaSlice<u16>,
    output_dev: CudaSlice<f32>,
    config: MatmulConfig,
    output_len: usize,
    host_to_device_bytes: u64,
    device_to_host_bytes: u64,
}

#[cfg(feature = "cuda")]
impl CudaDenseGemmFixtureSession {
    fn new(device_index: usize, a: &[u16], b: &[u16], config: &MatmulConfig) -> Result<Self> {
        let output_len = checked_mul(config.m, config.n, "output")?;
        validate_dense_f16_gemm_inputs(a, b, &vec![0.0f32; output_len], config)?;

        let ctx = CudaContext::new(device_index).map_err(|err| KernelError::GpuError {
            reason: format!(
                "failed to create CUDA context for persistent dense FP16 GEMM: {err:?}"
            ),
        })?;
        let stream = ctx.default_stream();
        let ptx = compile_dense_f16_gemm_ptx()?;
        let module = ctx.load_module(ptx).map_err(|err| KernelError::GpuError {
            reason: format!("failed to load persistent dense FP16 GEMM CUDA module: {err:?}"),
        })?;
        let function = module.load_function(CUDA_DENSE_F16_GEMM_KERNEL_ID).map_err(|err| {
            KernelError::GpuError {
                reason: format!("failed to load persistent dense FP16 GEMM CUDA kernel: {err:?}"),
            }
        })?;

        let a_len = checked_mul(config.m, config.k, "A")?;
        let b_len = checked_mul(config.k, config.n, "B")?;
        let a_dev = stream.memcpy_stod(&a[..a_len]).map_err(|err| KernelError::GpuError {
            reason: format!("failed to copy persistent dense FP16 GEMM A to device: {err:?}"),
        })?;
        let b_dev = stream.memcpy_stod(&b[..b_len]).map_err(|err| KernelError::GpuError {
            reason: format!("failed to copy persistent dense FP16 GEMM B to device: {err:?}"),
        })?;
        let output_dev: CudaSlice<f32> =
            stream.alloc_zeros(output_len).map_err(|err| KernelError::GpuError {
                reason: format!(
                    "failed to allocate persistent dense FP16 GEMM output on device: {err:?}"
                ),
            })?;

        Ok(Self {
            stream,
            function,
            a_dev,
            b_dev,
            output_dev,
            config: config.clone(),
            output_len,
            host_to_device_bytes: bytes_for::<u16>(a_len + b_len)?,
            device_to_host_bytes: bytes_for::<f32>(output_len)?,
        })
    }

    fn launch(&mut self, output: &mut [f32]) -> Result<()> {
        if output.len() < self.output_len {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "persistent dense CUDA GEMM output buffer too short: expected >= {}, got {}",
                    self.output_len,
                    output.len()
                ),
            }
            .into());
        }

        let threads_per_block = 128u32;
        let launch_config = LaunchConfig {
            grid_dim: ((self.output_len as u32).div_ceil(threads_per_block), 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = self.stream.launch_builder(&self.function);
        builder.arg(&self.a_dev);
        builder.arg(&self.b_dev);
        builder.arg(&mut self.output_dev);
        let m_arg = i32::try_from(self.config.m).map_err(|_| KernelError::InvalidArguments {
            reason: format!("persistent dense CUDA GEMM m exceeds i32: {}", self.config.m),
        })?;
        let n_arg = i32::try_from(self.config.n).map_err(|_| KernelError::InvalidArguments {
            reason: format!("persistent dense CUDA GEMM n exceeds i32: {}", self.config.n),
        })?;
        let k_arg = i32::try_from(self.config.k).map_err(|_| KernelError::InvalidArguments {
            reason: format!("persistent dense CUDA GEMM k exceeds i32: {}", self.config.k),
        })?;
        builder.arg(&m_arg);
        builder.arg(&n_arg);
        builder.arg(&k_arg);

        unsafe { builder.launch(launch_config) }.map_err(|err| KernelError::GpuError {
            reason: format!("failed to launch persistent dense FP16 GEMM CUDA kernel: {err:?}"),
        })?;
        self.stream.synchronize().map_err(|err| KernelError::GpuError {
            reason: format!(
                "failed to synchronize persistent dense FP16 GEMM CUDA kernel: {err:?}"
            ),
        })?;

        let output_host: Vec<f32> =
            self.stream.memcpy_dtov(&self.output_dev).map_err(|err| KernelError::GpuError {
                reason: format!(
                    "failed to copy persistent dense FP16 GEMM output from device: {err:?}"
                ),
            })?;
        output[..self.output_len].copy_from_slice(&output_host[..self.output_len]);
        Ok(())
    }
}

#[cfg(feature = "cuda")]
fn compile_dense_f16_gemm_ptx() -> Result<Ptx> {
    Ok(Ptx::from_src(CUDA_DENSE_F16_GEMM_PTX))
}

fn compare_outputs(expected: &[f32], actual: &[f32]) -> Result<(f32, f32)> {
    if expected.len() != actual.len() {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "dense CUDA GEMM parity length mismatch: expected {}, got {}",
                expected.len(),
                actual.len()
            ),
        }
        .into());
    }
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f32;
    for (expected, actual) in expected.iter().zip(actual) {
        let abs = (expected - actual).abs();
        max_abs = max_abs.max(abs);
        sum_abs += abs;
    }
    Ok((max_abs, sum_abs / expected.len() as f32))
}

fn checked_mul(lhs: usize, rhs: usize, label: &str) -> Result<usize> {
    lhs.checked_mul(rhs).ok_or_else(|| {
        KernelError::InvalidArguments {
            reason: format!("dense CUDA GEMM {label} length overflow: {lhs} * {rhs}"),
        }
        .into()
    })
}

#[cfg(feature = "cuda")]
fn bytes_for<T>(count: usize) -> Result<u64> {
    count
        .checked_mul(std::mem::size_of::<T>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| {
            KernelError::InvalidArguments {
                reason: format!("dense CUDA GEMM byte count overflow for {count} elements"),
            }
            .into()
        })
}

fn validate_i32_arg(value: usize, label: &str) -> Result<()> {
    i32::try_from(value).map(|_| ()).map_err(|_| {
        KernelError::InvalidArguments {
            reason: format!("dense CUDA GEMM {label} exceeds i32: {value}"),
        }
        .into()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};
    use std::error::Error;
    use std::io;
    use std::path::{Path, PathBuf};

    const RUN_ENV: &str = "BITNET_RUN_RTX5070TI_DENSE_CUDA_GEMM";
    const PERSISTENT_RUN_ENV: &str = "BITNET_RUN_RTX5070TI_DENSE_CUDA_GEMM_SESSION";
    const RECEIPT_ENV: &str = "BITNET_RTX5070TI_DENSE_CUDA_GEMM_RECEIPT";
    const PERSISTENT_RECEIPT_ENV: &str = "BITNET_RTX5070TI_DENSE_CUDA_GEMM_SESSION_RECEIPT";
    const ARTIFACT_PATH_ENV: &str = "BITNET_RTX5070TI_DENSE_CUDA_GEMM_ARTIFACT_PATH";
    const PERSISTENT_ARTIFACT_PATH_ENV: &str =
        "BITNET_RTX5070TI_DENSE_CUDA_GEMM_SESSION_ARTIFACT_PATH";
    const TIMESTAMP_ENV: &str = "BITNET_RTX5070TI_DENSE_CUDA_GEMM_TIMESTAMP_UTC";
    const PERSISTENT_RUNS_ENV: &str = "BITNET_RTX5070TI_DENSE_CUDA_GEMM_SESSION_RUNS";
    const DEVICE_INDEX_ENV: &str = "BITNET_RTX5070TI_CUDA_DEVICE_INDEX";
    const MACHINE_ID: &str = "windows-9950x3d-rtx5070ti";
    const HARDWARE_LANE: &str = "nvidia-rtx-5070-ti-cuda";

    #[test]
    fn dense_f16_fixture_cpu_reference_is_stable() {
        let reference = dense_f16_gemm_cpu_reference().unwrap();
        assert_eq!(reference.len(), 6);
        assert_eq!(reference, vec![5.75, 5.75, -3.0, 14.5, -8.5, -4.0]);
    }

    #[test]
    fn dense_f16_launch_rejects_non_f16_dtype_before_cuda() {
        let (a, b, mut cfg) = dense_f16_gemm_fixture().unwrap();
        cfg.dtype = MatmulDtype::F32;
        let mut output = vec![0.0; cfg.m * cfg.n];

        let err = launch_dense_f16_gemm_cuda(0, &a, &b, &mut output, &cfg).unwrap_err();

        assert!(err.to_string().contains("MatmulDtype::F16"), "unexpected error: {err}");
    }

    #[test]
    fn dense_f16_launch_rejects_hidden_batch_before_cuda() {
        let (a, b, mut cfg) = dense_f16_gemm_fixture().unwrap();
        cfg.batch_size = 2;
        let mut output = vec![0.0; cfg.m * cfg.n];

        let err = launch_dense_f16_gemm_cuda(0, &a, &b, &mut output, &cfg).unwrap_err();

        assert!(err.to_string().contains("batch_size=1"), "unexpected error: {err}");
    }

    #[test]
    fn dense_f16_launch_rejects_transpose_before_cuda() {
        let (a, b, mut cfg) = dense_f16_gemm_fixture().unwrap();
        cfg.transpose_a = true;
        let mut output = vec![0.0; cfg.m * cfg.n];

        let err = launch_dense_f16_gemm_cuda(0, &a, &b, &mut output, &cfg).unwrap_err();

        assert!(err.to_string().contains("non-transposed"), "unexpected error: {err}");
    }

    #[test]
    fn dense_f16_persistent_launch_rejects_zero_runs_before_cuda() {
        let err = run_dense_f16_gemm_cuda_persistent_parity(0, 0).unwrap_err();

        assert!(err.to_string().contains("at least one run"), "unexpected error: {err}");
    }

    #[test]
    fn dense_f16_gemm_receipt_contract_preserves_dense_boundary() {
        let parity = synthetic_passed_parity();
        let receipt = dense_gemm_receipt_json(
            &parity,
            None,
            "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/dense-f16-gemm-parity.json",
            "2026-05-08T00:00:00Z",
        );

        assert_eq!(receipt["artifact_kind"], "dense_regular_llm_cuda");
        assert_eq!(receipt["requested_backend"], HARDWARE_LANE);
        assert_eq!(receipt["selected_backend"], HARDWARE_LANE);
        assert_eq!(receipt["runtime_api"], "cuda");
        assert_eq!(receipt["fallback_used"], false);
        assert_eq!(receipt["speedup_claim"], false);
        assert_eq!(receipt["execution_path"]["model_class"], "dense_regular_llm");
        assert_eq!(receipt["execution_path"]["bitnet_packed_kernel_proof"], false);
        assert_eq!(receipt["execution_path"]["qk256_proof"], false);
        assert_eq!(receipt["kernel_stats"][0]["kernel_id"], CUDA_DENSE_F16_GEMM_KERNEL_ID);
        assert_eq!(receipt["kernel_stats"][0]["fallback_invocations"], 0);
        assert_eq!(receipt["parity"]["reference_backend"], CUDA_DENSE_GEMM_REFERENCE_BACKEND);
        assert_eq!(receipt["parity"]["target_backend"], CUDA_DENSE_GEMM_TARGET_BACKEND);
        assert_eq!(receipt["parity"]["fixture_id"], CUDA_DENSE_F16_GEMM_FIXTURE_ID);
        assert_eq!(receipt["parity"]["passed"], true);
        assert_eq!(receipt["claim_boundary"]["dense_regular_llm_cuda_claimed"], true);
        assert_eq!(receipt["claim_boundary"]["bitnet_packed_i2s_qk256_proof"], false);
        assert_eq!(receipt["claim_boundary"]["speedup_claim"], false);
        assert_eq!(receipt["claim_boundary"]["full_cuda_residency_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["dense_gguf_inference_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["persistent_session_residency_claimed"], false);
        assert_eq!(receipt["tensor_residency"]["scope"], "single_dense_f16_gemm_fixture");
        assert_eq!(receipt["tensor_residency"]["input_tensors_uploaded_once"], true);
        assert_eq!(receipt["tensor_residency"]["output_tensor_cuda_resident_during_kernel"], true);
        assert_eq!(receipt["tensor_residency"]["full_cuda_residency_claimed"], false);
        assert_eq!(receipt["tensor_residency"]["inputs"].as_array().unwrap().len(), 2);
        assert_eq!(
            receipt["tensor_residency"]["transfer_accounting"]["host_to_device_bytes"],
            parity.stats.host_to_device_bytes
        );
    }

    #[test]
    fn dense_f16_gemm_persistent_receipt_contract_preserves_dense_boundary() {
        let parity = synthetic_passed_persistent_parity();
        let receipt = dense_persistent_gemm_receipt_json(
            &parity,
            None,
            "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/dense-f16-gemm-persistent.json",
            "2026-05-08T00:00:00Z",
        );

        assert_eq!(receipt["artifact_kind"], "dense_regular_llm_cuda");
        assert_eq!(receipt["claim"], "dense_regular_llm_cuda_persistent_fixture_residency_tested");
        assert_eq!(receipt["requested_backend"], HARDWARE_LANE);
        assert_eq!(receipt["selected_backend"], HARDWARE_LANE);
        assert_eq!(receipt["runtime_api"], "cuda");
        assert_eq!(receipt["fallback_used"], false);
        assert_eq!(receipt["speedup_claim"], false);
        assert_eq!(receipt["execution_path"]["model_class"], "dense_regular_llm");
        assert_eq!(receipt["execution_path"]["bitnet_packed_kernel_proof"], false);
        assert_eq!(receipt["execution_path"]["qk256_proof"], false);
        assert_eq!(receipt["claim_boundary"]["dense_regular_llm_cuda_claimed"], true);
        assert_eq!(receipt["claim_boundary"]["dense_tensor_residency_claimed"], true);
        assert_eq!(receipt["claim_boundary"]["persistent_session_residency_claimed"], true);
        assert_eq!(receipt["claim_boundary"]["dense_gguf_inference_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["speedup_claim"], false);
        assert_eq!(receipt["claim_boundary"]["full_cuda_residency_claimed"], false);
        assert_eq!(
            receipt["tensor_residency"]["scope"],
            "persistent_dense_f16_gemm_fixture_session"
        );
        assert_eq!(receipt["tensor_residency"]["persistent_session_residency_claimed"], true);
        assert_eq!(receipt["tensor_residency"]["input_tensors_uploaded_once"], true);
        assert_eq!(receipt["tensor_residency"]["per_run_host_to_device_bytes"], 0);
        assert_eq!(
            receipt["tensor_residency"]["allocation"]["persistent_handle_count"],
            parity.stats.persistent_handle_count
        );
        assert_eq!(receipt["persistent_session"]["repeated_runs"], parity.stats.runs);
        assert_eq!(receipt["persistent_session"]["context_creations"], 1);
        assert_eq!(receipt["persistent_session"]["module_loads"], 1);
    }

    #[test]
    fn live_rtx5070ti_dense_f16_cuda_gemm_matches_cpu_reference_when_enabled()
    -> std::result::Result<(), Box<dyn Error>> {
        if std::env::var(RUN_ENV).as_deref() != Ok("1") {
            eprintln!("skipping live dense CUDA GEMM parity; set {RUN_ENV}=1 to run it");
            return Ok(());
        }

        let device_index = selected_device_index()?;
        let probe = bitnet_device_probe::probe_nvidia_cuda(Some(device_index));
        if !probe.available {
            return Err(io_error(format!(
                "CUDA-DENSE-002 requires CUDA probe success: {:?}",
                probe.failure_reason
            )));
        }

        let parity = run_dense_f16_gemm_cuda_parity(device_index)?;
        if !is_rtx5070ti_device_name(probe.selected_device_name.as_deref().unwrap_or_default()) {
            return Err(io_error(format!(
                "CUDA-DENSE-002 requires NVIDIA GeForce RTX 5070 Ti; found '{}'",
                probe.selected_device_name.as_deref().unwrap_or("unknown")
            )));
        }

        let artifact_path = std::env::var(ARTIFACT_PATH_ENV).unwrap_or_else(|_| {
            "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/dense-f16-gemm-parity.json"
                .to_string()
        });
        let timestamp_utc =
            std::env::var(TIMESTAMP_ENV).unwrap_or_else(|_| "2026-05-08T00:00:00Z".to_string());
        let receipt_json =
            dense_gemm_receipt_json(&parity, Some(&probe), &artifact_path, &timestamp_utc);

        if let Ok(path) = std::env::var(RECEIPT_ENV) {
            write_json_file(&path, &receipt_json)?;
        }
        println!("{}", serde_json::to_string_pretty(&receipt_json)?);

        if !parity.passed {
            return Err(io_error(format!(
                "CUDA-DENSE-002 dense FP16 GEMM parity failed: max_abs_error={} tolerance={}",
                parity.max_abs_error, parity.tolerance
            )));
        }

        Ok(())
    }

    #[test]
    fn live_rtx5070ti_dense_f16_cuda_gemm_persistent_session_when_enabled()
    -> std::result::Result<(), Box<dyn Error>> {
        if std::env::var(PERSISTENT_RUN_ENV).as_deref() != Ok("1") {
            eprintln!(
                "skipping live dense CUDA GEMM persistent session; set {PERSISTENT_RUN_ENV}=1 to run it"
            );
            return Ok(());
        }

        let device_index = selected_device_index()?;
        let runs = selected_persistent_runs()?;
        let probe = bitnet_device_probe::probe_nvidia_cuda(Some(device_index));
        if !probe.available {
            return Err(io_error(format!(
                "CUDA-DENSE-004 requires CUDA probe success: {:?}",
                probe.failure_reason
            )));
        }

        let parity = run_dense_f16_gemm_cuda_persistent_parity(device_index, runs)?;
        if !is_rtx5070ti_device_name(probe.selected_device_name.as_deref().unwrap_or_default()) {
            return Err(io_error(format!(
                "CUDA-DENSE-004 requires NVIDIA GeForce RTX 5070 Ti; found '{}'",
                probe.selected_device_name.as_deref().unwrap_or("unknown")
            )));
        }

        let artifact_path = std::env::var(PERSISTENT_ARTIFACT_PATH_ENV).unwrap_or_else(|_| {
            "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/dense-f16-gemm-persistent.json"
                .to_string()
        });
        let timestamp_utc =
            std::env::var(TIMESTAMP_ENV).unwrap_or_else(|_| "2026-05-08T00:00:00Z".to_string());
        let receipt_json = dense_persistent_gemm_receipt_json(
            &parity,
            Some(&probe),
            &artifact_path,
            &timestamp_utc,
        );

        if let Ok(path) = std::env::var(PERSISTENT_RECEIPT_ENV) {
            write_json_file(&path, &receipt_json)?;
        }
        println!("{}", serde_json::to_string_pretty(&receipt_json)?);

        if !parity.passed {
            return Err(io_error(format!(
                "CUDA-DENSE-004 persistent dense FP16 GEMM parity failed: max_abs_error={} tolerance={}",
                parity.max_abs_error, parity.tolerance
            )));
        }

        Ok(())
    }

    fn synthetic_passed_parity() -> CudaDenseGemmParity {
        CudaDenseGemmParity {
            fixture_id: CUDA_DENSE_F16_GEMM_FIXTURE_ID,
            reference_backend: CUDA_DENSE_GEMM_REFERENCE_BACKEND,
            target_backend: CUDA_DENSE_GEMM_TARGET_BACKEND,
            kernel_id: CUDA_DENSE_F16_GEMM_KERNEL_ID,
            max_abs_error: 0.0,
            mean_abs_error: 0.0,
            tolerance: CUDA_DENSE_F16_GEMM_TOLERANCE,
            passed: true,
            stats: CudaDenseGemmStats {
                kernel_id: CUDA_DENSE_F16_GEMM_KERNEL_ID,
                invocations: 1,
                fallback_invocations: 0,
                host_to_device_bytes: 40,
                device_to_host_bytes: 24,
                kernel_launches: 1,
                kernel_time_ms: None,
            },
        }
    }

    fn synthetic_passed_persistent_parity() -> CudaDenseGemmPersistentParity {
        CudaDenseGemmPersistentParity {
            fixture_id: CUDA_DENSE_F16_GEMM_FIXTURE_ID,
            reference_backend: CUDA_DENSE_GEMM_REFERENCE_BACKEND,
            target_backend: CUDA_DENSE_GEMM_TARGET_BACKEND,
            kernel_id: CUDA_DENSE_F16_GEMM_KERNEL_ID,
            runs: 3,
            max_abs_error: 0.0,
            mean_abs_error: 0.0,
            tolerance: CUDA_DENSE_F16_GEMM_TOLERANCE,
            passed: true,
            stats: CudaDenseGemmPersistentStats {
                kernel_id: CUDA_DENSE_F16_GEMM_KERNEL_ID,
                runs: 3,
                invocations: 3,
                fallback_invocations: 0,
                host_to_device_bytes: 40,
                device_to_host_bytes: 72,
                kernel_launches: 3,
                context_creations: 1,
                module_loads: 1,
                input_uploads: 2,
                output_allocations: 1,
                persistent_handle_count: 3,
                per_run_host_to_device_bytes: 0,
                kernel_time_ms: None,
            },
        }
    }

    fn selected_device_index() -> std::result::Result<usize, Box<dyn Error>> {
        match std::env::var(DEVICE_INDEX_ENV) {
            Ok(value) => value.parse::<usize>().map_err(|error| {
                io_error(format!("{DEVICE_INDEX_ENV} must be a non-negative integer: {error}"))
            }),
            Err(_) => Ok(0),
        }
    }

    fn selected_persistent_runs() -> std::result::Result<usize, Box<dyn Error>> {
        match std::env::var(PERSISTENT_RUNS_ENV) {
            Ok(value) => value.parse::<usize>().map_err(|error| {
                io_error(format!("{PERSISTENT_RUNS_ENV} must be a positive integer: {error}"))
            }),
            Err(_) => Ok(3),
        }
    }

    fn is_rtx5070ti_device_name(name: &str) -> bool {
        let compact = name
            .chars()
            .filter(|ch| ch.is_ascii_alphanumeric())
            .collect::<String>()
            .to_ascii_lowercase();

        compact.contains("nvidia") && compact.contains("rtx5070ti")
    }

    fn dense_gemm_receipt_json(
        parity: &CudaDenseGemmParity,
        probe: Option<&bitnet_device_probe::NvidiaCudaProbe>,
        artifact_path: &str,
        timestamp_utc: &str,
    ) -> Value {
        let cuda = match probe {
            Some(probe) => json!({
                "available": probe.available,
                "device_count": probe.device_count,
                "device_index": probe.selected_device_index.unwrap_or(0),
                "device_name": probe.selected_device_name.clone().unwrap_or_else(|| "unknown".into()),
                "compute_capability": probe.compute_capability.clone().unwrap_or_else(|| "12.0".into()),
                "driver_version": probe.driver_version.clone().unwrap_or_else(|| "unknown".into()),
                "cuda_runtime_version": probe.cuda_runtime_version.clone().unwrap_or_else(|| "unknown".into()),
                "cuda_toolkit_version": probe.cuda_toolkit_version.clone().unwrap_or_else(|| "unknown".into()),
                "nvrtc_version": probe.nvrtc_version.clone().unwrap_or_else(|| "unknown".into()),
                "nvml_available": probe.nvml_available,
                "vram_bytes": probe.vram_bytes.unwrap_or(1),
                "power_limit_watts": probe.power_limit_watts,
                "power_draw_watts": probe.power_draw_watts,
                "temperature_c": probe.temperature_c,
            }),
            None => json!({
                "available": true,
                "device_count": 1,
                "device_index": 0,
                "device_name": "NVIDIA GeForce RTX 5070 Ti",
                "compute_capability": "12.0",
                "driver_version": "591.86",
                "cuda_runtime_version": "12.9",
                "cuda_toolkit_version": "12.9",
                "nvrtc_version": "12.9",
                "nvml_available": true,
                "vram_bytes": 17094475776_u64,
                "power_limit_watts": 300.0,
                "power_draw_watts": 34.97,
                "temperature_c": 38.0,
            }),
        };

        let claim = if artifact_path.contains("residency") {
            "dense_regular_llm_cuda_tensor_residency_tested"
        } else {
            "dense_regular_llm_cuda_gemm_parity_tested"
        };

        json!({
            "schema": 1,
            "artifact_kind": "dense_regular_llm_cuda",
            "artifact_path": artifact_path,
            "claim": claim,
            "machine_id": MACHINE_ID,
            "hardware_lane": HARDWARE_LANE,
            "timestamp_utc": timestamp_utc,
            "requested_backend": HARDWARE_LANE,
            "selected_backend": HARDWARE_LANE,
            "runtime_api": "cuda",
            "fallback_used": false,
            "fallback_backend": null,
            "fallback_reason": null,
            "speedup_claim": false,
            "cuda": cuda,
            "model": {
                "model_family": "qwen",
                "artifact_kind": "dense_gguf",
                "file": "dense-f16-gemm-smoke-fixture",
                "sha256": "0".repeat(64)
            },
            "execution_path": {
                "model_class": "dense_regular_llm",
                "kernel_family": "dense_fp16_gemm",
                "quantization_family": "fp16_dense",
                "bitnet_packed_kernel_proof": false,
                "qk256_proof": false
            },
            "kernel_stats": [{
                "kernel_id": parity.stats.kernel_id,
                "invocations": parity.stats.invocations,
                "fallback_invocations": parity.stats.fallback_invocations,
                "host_to_device_bytes": parity.stats.host_to_device_bytes,
                "device_to_host_bytes": parity.stats.device_to_host_bytes,
                "kernel_launches": parity.stats.kernel_launches,
                "kernel_time_ms": parity.stats.kernel_time_ms
            }],
            "parity": {
                "reference_backend": parity.reference_backend,
                "target_backend": parity.target_backend,
                "kernel_id": parity.kernel_id,
                "fixture_id": parity.fixture_id,
                "max_abs_error": parity.max_abs_error,
                "mean_abs_error": parity.mean_abs_error,
                "passed": parity.passed,
                "tolerance": parity.tolerance,
                "tolerance_source": "CUDA-DENSE-002 deterministic FP16 smoke fixture"
            },
            "claim_boundary": {
                "dense_regular_llm_cuda_claimed": true,
                "dense_tensor_residency_claimed": claim == "dense_regular_llm_cuda_tensor_residency_tested",
                "dense_gguf_inference_claimed": false,
                "bitnet_packed_i2s_qk256_proof": false,
                "speedup_claim": false,
                "persistent_session_residency_claimed": false,
                "full_cuda_residency_claimed": false
            },
            "tensor_residency": {
                "schema_version": "1.0.0",
                "scope": "single_dense_f16_gemm_fixture",
                "model_class": "dense_regular_llm",
                "fixture_id": parity.fixture_id,
                "dense_tensor_residency_claimed": true,
                "dense_gguf_inference_claimed": false,
                "persistent_session_residency_claimed": false,
                "full_cuda_residency_claimed": false,
                "input_tensors_uploaded_once": true,
                "output_tensor_cuda_resident_during_kernel": true,
                "host_device_transfer_accounting_matches_kernel_stats": true,
                "inputs": [
                    {
                        "name": "a",
                        "dtype": "f16",
                        "shape": [2, 4],
                        "host_bytes": 16,
                        "device_residency": "cuda_device_buffer",
                        "upload_count": 1,
                        "reuse_scope": "single_fixture_launch"
                    },
                    {
                        "name": "b",
                        "dtype": "f16",
                        "shape": [4, 3],
                        "host_bytes": 24,
                        "device_residency": "cuda_device_buffer",
                        "upload_count": 1,
                        "reuse_scope": "single_fixture_launch"
                    }
                ],
                "outputs": [
                    {
                        "name": "c",
                        "dtype": "f32",
                        "shape": [2, 3],
                        "device_residency": "cuda_device_buffer",
                        "device_to_host_bytes": parity.stats.device_to_host_bytes,
                        "download_scope": "parity_check_only"
                    }
                ],
                "allocation": {
                    "device_buffer_count": 3,
                    "temporary_workspace_bytes": 0,
                    "persistent_handle_count": 0,
                    "persistent_handles_claimed": false
                },
                "transfer_accounting": {
                    "status": "measured",
                    "host_to_device_bytes": parity.stats.host_to_device_bytes,
                    "device_to_host_bytes": parity.stats.device_to_host_bytes
                }
            },
            "error": null
        })
    }

    fn dense_persistent_gemm_receipt_json(
        parity: &CudaDenseGemmPersistentParity,
        probe: Option<&bitnet_device_probe::NvidiaCudaProbe>,
        artifact_path: &str,
        timestamp_utc: &str,
    ) -> Value {
        let cuda = match probe {
            Some(probe) => json!({
                "available": probe.available,
                "device_count": probe.device_count,
                "device_index": probe.selected_device_index.unwrap_or(0),
                "device_name": probe.selected_device_name.clone().unwrap_or_else(|| "unknown".into()),
                "compute_capability": probe.compute_capability.clone().unwrap_or_else(|| "12.0".into()),
                "driver_version": probe.driver_version.clone().unwrap_or_else(|| "unknown".into()),
                "cuda_runtime_version": probe.cuda_runtime_version.clone().unwrap_or_else(|| "unknown".into()),
                "cuda_toolkit_version": probe.cuda_toolkit_version.clone().unwrap_or_else(|| "unknown".into()),
                "nvrtc_version": probe.nvrtc_version.clone().unwrap_or_else(|| "unknown".into()),
                "nvml_available": probe.nvml_available,
                "vram_bytes": probe.vram_bytes.unwrap_or(1),
                "power_limit_watts": probe.power_limit_watts,
                "power_draw_watts": probe.power_draw_watts,
                "temperature_c": probe.temperature_c,
            }),
            None => json!({
                "available": true,
                "device_count": 1,
                "device_index": 0,
                "device_name": "NVIDIA GeForce RTX 5070 Ti",
                "compute_capability": "12.0",
                "driver_version": "591.86",
                "cuda_runtime_version": "12.9",
                "cuda_toolkit_version": "12.9",
                "nvrtc_version": "12.9",
                "nvml_available": true,
                "vram_bytes": 17094475776_u64,
                "power_limit_watts": 300.0,
                "power_draw_watts": 34.97,
                "temperature_c": 38.0,
            }),
        };

        json!({
            "schema": 1,
            "artifact_kind": "dense_regular_llm_cuda",
            "artifact_path": artifact_path,
            "claim": "dense_regular_llm_cuda_persistent_fixture_residency_tested",
            "machine_id": MACHINE_ID,
            "hardware_lane": HARDWARE_LANE,
            "timestamp_utc": timestamp_utc,
            "requested_backend": HARDWARE_LANE,
            "selected_backend": HARDWARE_LANE,
            "runtime_api": "cuda",
            "fallback_used": false,
            "fallback_backend": null,
            "fallback_reason": null,
            "speedup_claim": false,
            "cuda": cuda,
            "model": {
                "model_family": "qwen",
                "artifact_kind": "dense_gguf",
                "file": "dense-f16-gemm-persistent-fixture",
                "sha256": "0".repeat(64)
            },
            "execution_path": {
                "model_class": "dense_regular_llm",
                "kernel_family": "dense_fp16_gemm",
                "quantization_family": "fp16_dense",
                "bitnet_packed_kernel_proof": false,
                "qk256_proof": false
            },
            "kernel_stats": [{
                "kernel_id": parity.stats.kernel_id,
                "invocations": parity.stats.invocations,
                "fallback_invocations": parity.stats.fallback_invocations,
                "host_to_device_bytes": parity.stats.host_to_device_bytes,
                "device_to_host_bytes": parity.stats.device_to_host_bytes,
                "kernel_launches": parity.stats.kernel_launches,
                "kernel_time_ms": parity.stats.kernel_time_ms
            }],
            "parity": {
                "reference_backend": parity.reference_backend,
                "target_backend": parity.target_backend,
                "kernel_id": parity.kernel_id,
                "fixture_id": parity.fixture_id,
                "runs": parity.runs,
                "max_abs_error": parity.max_abs_error,
                "mean_abs_error": parity.mean_abs_error,
                "passed": parity.passed,
                "tolerance": parity.tolerance,
                "tolerance_source": "CUDA-DENSE-004 persistent deterministic FP16 fixture"
            },
            "claim_boundary": {
                "dense_regular_llm_cuda_claimed": true,
                "dense_tensor_residency_claimed": true,
                "dense_gguf_inference_claimed": false,
                "bitnet_packed_i2s_qk256_proof": false,
                "speedup_claim": false,
                "persistent_session_residency_claimed": true,
                "full_cuda_residency_claimed": false
            },
            "persistent_session": {
                "schema_version": "1.0.0",
                "scope": "persistent_dense_f16_gemm_fixture_session",
                "repeated_runs": parity.stats.runs,
                "context_creations": parity.stats.context_creations,
                "module_loads": parity.stats.module_loads,
                "kernel_launches": parity.stats.kernel_launches,
                "input_uploads": parity.stats.input_uploads,
                "output_allocations": parity.stats.output_allocations,
                "persistent_handle_count": parity.stats.persistent_handle_count,
                "per_run_host_to_device_bytes": parity.stats.per_run_host_to_device_bytes,
                "dense_gguf_inference_claimed": false,
                "full_cuda_residency_claimed": false,
                "speedup_claim": false
            },
            "tensor_residency": {
                "schema_version": "1.0.0",
                "scope": "persistent_dense_f16_gemm_fixture_session",
                "model_class": "dense_regular_llm",
                "fixture_id": parity.fixture_id,
                "dense_tensor_residency_claimed": true,
                "dense_gguf_inference_claimed": false,
                "persistent_session_residency_claimed": true,
                "full_cuda_residency_claimed": false,
                "input_tensors_uploaded_once": true,
                "output_tensor_cuda_resident_during_kernel": true,
                "host_device_transfer_accounting_matches_kernel_stats": true,
                "per_run_host_to_device_bytes": parity.stats.per_run_host_to_device_bytes,
                "inputs": [
                    {
                        "name": "a",
                        "dtype": "f16",
                        "shape": [2, 4],
                        "host_bytes": 16,
                        "device_residency": "cuda_device_buffer",
                        "upload_count": 1,
                        "reuse_scope": "persistent_fixture_session"
                    },
                    {
                        "name": "b",
                        "dtype": "f16",
                        "shape": [4, 3],
                        "host_bytes": 24,
                        "device_residency": "cuda_device_buffer",
                        "upload_count": 1,
                        "reuse_scope": "persistent_fixture_session"
                    }
                ],
                "outputs": [
                    {
                        "name": "c",
                        "dtype": "f32",
                        "shape": [2, 3],
                        "device_residency": "cuda_device_buffer",
                        "device_to_host_bytes": parity.stats.device_to_host_bytes,
                        "download_scope": "parity_check_each_run"
                    }
                ],
                "allocation": {
                    "device_buffer_count": 3,
                    "temporary_workspace_bytes": 0,
                    "persistent_handle_count": parity.stats.persistent_handle_count,
                    "persistent_handles_claimed": true
                },
                "transfer_accounting": {
                    "status": "measured",
                    "host_to_device_bytes": parity.stats.host_to_device_bytes,
                    "device_to_host_bytes": parity.stats.device_to_host_bytes
                }
            },
            "error": null
        })
    }

    fn write_json_file(path: &str, value: &Value) -> std::result::Result<(), Box<dyn Error>> {
        let output_path = workspace_relative_path(path);
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(output_path, serde_json::to_string_pretty(value)?)?;
        Ok(())
    }

    fn workspace_relative_path(path: &str) -> PathBuf {
        let path = Path::new(path);
        if path.is_absolute() {
            return path.to_path_buf();
        }

        Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join(path)
    }

    fn io_error(message: String) -> Box<dyn Error> {
        Box::new(io::Error::other(message))
    }
}

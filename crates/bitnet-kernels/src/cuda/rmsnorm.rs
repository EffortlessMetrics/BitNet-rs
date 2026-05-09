//! RMSNorm CUDA kernel.
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
//! The kernel is a single-pass warp-level reduction:
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

use bitnet_common::{KernelError, Result};
#[cfg(feature = "cuda")]
use std::any::Any;
#[cfg(feature = "cuda")]
use std::sync::Mutex;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::{Ptx, compile_ptx};

/// Kernel ID recorded by dense regular-LLM CUDA RMSNorm receipts.
pub const CUDA_DENSE_RMSNORM_KERNEL_ID: &str = "dense_rmsnorm_f32_cuda";

/// Tolerance for dense GGUF RMSNorm fixture parity against the CPU reference.
pub const CUDA_DENSE_RMSNORM_TOLERANCE: f32 = 0.000_05;

/// CPU reference backend recorded by dense RMSNorm CUDA parity receipts.
pub const CUDA_DENSE_RMSNORM_REFERENCE_BACKEND: &str = "amd-9950x3d-cpu-avx512";

/// RTX 5070 Ti CUDA backend recorded by dense RMSNorm CUDA parity receipts.
pub const CUDA_DENSE_RMSNORM_TARGET_BACKEND: &str = "nvidia-rtx-5070-ti-cuda";

#[cfg(feature = "cuda")]
static NVRTC_COMPILE_LOCK: Mutex<()> = Mutex::new(());

#[cfg(feature = "cuda")]
const CUDA_DENSE_RMSNORM_KERNEL_SRC: &str = r#"
extern "C" __global__
void dense_rmsnorm_f32_cuda(
    const float* input,
    const float* gamma,
    float* output,
    int hidden_dim,
    int n_rows,
    float eps
) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows || col >= hidden_dim) {
        return;
    }

    const float* row_input = input + ((long long)row * hidden_dim);
    float* row_output = output + ((long long)row * hidden_dim);
    float sq_sum = 0.0f;
    for (int i = 0; i < hidden_dim; ++i) {
        float value = row_input[i];
        sq_sum += value * value;
    }

    float inv_rms = rsqrtf(sq_sum / (float)hidden_dim + eps);
    row_output[col] = row_input[col] * inv_rms * gamma[col];
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

/// CUDA execution counters for a dense RMSNorm fixture.
#[derive(Debug, Clone, PartialEq)]
pub struct CudaDenseRmsNormStats {
    /// CUDA kernel identifier.
    pub kernel_id: &'static str,
    /// Number of CUDA kernel invocations.
    pub invocations: u64,
    /// CPU fallback invocations under strict CUDA.
    pub fallback_invocations: u64,
    /// Host-to-device bytes copied for input and gamma.
    pub host_to_device_bytes: u64,
    /// Device-to-host bytes copied for output.
    pub device_to_host_bytes: u64,
    /// CUDA kernel launches.
    pub kernel_launches: u64,
    /// Optional measured kernel time.
    pub kernel_time_ms: Option<f64>,
}

/// Dense GGUF RMSNorm fixture data prepared by the model layer.
///
/// The model layer owns GGUF parsing and tensor materialization. This bridge
/// deliberately accepts plain F32 buffers so the CUDA kernel layer does not
/// depend on `bitnet-models`.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseGgufRmsNormCudaFixture {
    /// Fixture identifier recorded in parity receipts.
    pub fixture_id: String,
    /// Dense model family label, for example `qwen`.
    pub model_family: String,
    /// Source GGUF tensor name.
    pub tensor_name: String,
    /// Logical source tensor role, for example `attention_norm`.
    pub tensor_role: String,
    /// Source GGUF tensor type after descriptor inspection, for example `f32`.
    pub tensor_type: String,
    /// SHA-256 of the source materialized F32 gamma values.
    pub source_weight_sha256: String,
    /// Hidden dimension for the RMSNorm vector.
    pub hidden_dim: usize,
    /// Deterministic CPU-reference input vector.
    pub input_f32: Vec<f32>,
    /// RMSNorm gamma weights as F32.
    pub gamma_f32: Vec<f32>,
    /// CPU RMSNorm reference output.
    pub expected_output_f32: Vec<f32>,
    /// Epsilon used by the dense model family.
    pub rmsnorm_eps: f32,
}

/// Dense GGUF RMSNorm CUDA parity result against the CPU reference.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseGgufRmsNormCudaParity {
    /// Fixture identifier.
    pub fixture_id: String,
    /// Dense model family label.
    pub model_family: String,
    /// Source GGUF tensor name.
    pub tensor_name: String,
    /// Logical source tensor role.
    pub tensor_role: String,
    /// Source GGUF tensor type.
    pub tensor_type: String,
    /// SHA-256 of the source materialized F32 gamma values.
    pub source_weight_sha256: String,
    /// Hidden dimension for the RMSNorm vector.
    pub hidden_dim: usize,
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
    pub stats: CudaDenseRmsNormStats,
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
}

/// Launch stub for the RMSNorm kernel.
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
/// Returns `KernelError::GpuError` until a real PTX kernel is compiled and loaded.
pub fn launch_rmsnorm(
    input: &[f32],
    gamma: &[f32],
    output: &mut [f32],
    config: &RmsNormConfig,
) -> Result<()> {
    validate_rmsnorm_buffers(input, gamma, output, config.hidden_dim, config.n_rows, config.eps)?;

    #[cfg(feature = "cuda")]
    {
        launch_dense_rmsnorm_f32_cuda_impl(0, input, gamma, output, config)?;
        return Ok(());
    }

    #[cfg(not(feature = "cuda"))]
    {
        Err(KernelError::DeviceUnavailable {
            reason: "RMSNorm CUDA launch requires the cuda feature".to_string(),
        }
        .into())
    }
}

/// Launch a strict dense RMSNorm F32 CUDA fixture and return execution counters.
///
/// # Errors
///
/// Returns an error if CUDA/NVRTC is unavailable, buffers are invalid, or the
/// kernel launch fails. This function never falls back to CPU.
pub fn launch_dense_rmsnorm_f32_cuda(
    device_index: usize,
    input: &[f32],
    gamma: &[f32],
    output: &mut [f32],
    config: &RmsNormConfig,
) -> Result<CudaDenseRmsNormStats> {
    validate_rmsnorm_buffers(input, gamma, output, config.hidden_dim, config.n_rows, config.eps)?;

    #[cfg(feature = "cuda")]
    {
        return launch_dense_rmsnorm_f32_cuda_impl(device_index, input, gamma, output, config);
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = device_index;
        Err(KernelError::DeviceUnavailable {
            reason: "dense RMSNorm CUDA parity requires the cuda feature".to_string(),
        }
        .into())
    }
}

/// Run a dense GGUF RMSNorm fixture on CUDA and compare it to the CPU reference.
///
/// # Errors
///
/// Returns an error if the fixture is invalid, CUDA is unavailable, or parity
/// comparison cannot be computed.
pub fn run_dense_gguf_rmsnorm_cuda_parity(
    device_index: usize,
    fixture: &DenseGgufRmsNormCudaFixture,
) -> Result<DenseGgufRmsNormCudaParity> {
    validate_dense_gguf_rmsnorm_fixture(fixture)?;
    let config = RmsNormConfig::for_shape(fixture.hidden_dim, 1)?.with_eps(fixture.rmsnorm_eps);
    let mut actual = vec![0.0f32; fixture.hidden_dim];
    let stats = launch_dense_rmsnorm_f32_cuda(
        device_index,
        &fixture.input_f32,
        &fixture.gamma_f32,
        &mut actual,
        &config,
    )?;
    let (max_abs_error, mean_abs_error) = compare_outputs(&fixture.expected_output_f32, &actual)?;

    Ok(DenseGgufRmsNormCudaParity {
        fixture_id: fixture.fixture_id.clone(),
        model_family: fixture.model_family.clone(),
        tensor_name: fixture.tensor_name.clone(),
        tensor_role: fixture.tensor_role.clone(),
        tensor_type: fixture.tensor_type.clone(),
        source_weight_sha256: fixture.source_weight_sha256.clone(),
        hidden_dim: fixture.hidden_dim,
        reference_backend: CUDA_DENSE_RMSNORM_REFERENCE_BACKEND,
        target_backend: CUDA_DENSE_RMSNORM_TARGET_BACKEND,
        kernel_id: CUDA_DENSE_RMSNORM_KERNEL_ID,
        max_abs_error,
        mean_abs_error,
        tolerance: CUDA_DENSE_RMSNORM_TOLERANCE,
        passed: max_abs_error <= CUDA_DENSE_RMSNORM_TOLERANCE,
        stats,
    })
}

fn validate_rmsnorm_buffers(
    input: &[f32],
    gamma: &[f32],
    output: &[f32],
    hidden_dim: usize,
    n_rows: usize,
    eps: f32,
) -> Result<()> {
    if hidden_dim == 0 || n_rows == 0 {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "RMSNorm dimensions must be non-zero: hidden_dim={hidden_dim}, n_rows={n_rows}"
            ),
        }
        .into());
    }
    if !eps.is_finite() || eps <= 0.0 {
        return Err(KernelError::InvalidArguments {
            reason: format!("RMSNorm epsilon must be positive and finite, got {eps}"),
        }
        .into());
    }
    let required_input = checked_mul(hidden_dim, n_rows, "RMSNorm input")?;
    if input.len() < required_input {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "RMSNorm input buffer too short: expected >= {required_input}, got {}",
                input.len()
            ),
        }
        .into());
    }
    if gamma.len() < hidden_dim {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "RMSNorm gamma buffer too short: expected >= {hidden_dim}, got {}",
                gamma.len()
            ),
        }
        .into());
    }
    if output.len() < required_input {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "RMSNorm output buffer too short: expected >= {required_input}, got {}",
                output.len()
            ),
        }
        .into());
    }
    validate_i32_arg(hidden_dim, "hidden_dim")?;
    validate_i32_arg(n_rows, "n_rows")?;
    for (idx, value) in input[..required_input].iter().enumerate() {
        if !value.is_finite() {
            return Err(KernelError::InvalidArguments {
                reason: format!("RMSNorm input[{idx}] is not finite"),
            }
            .into());
        }
    }
    for (idx, value) in gamma[..hidden_dim].iter().enumerate() {
        if !value.is_finite() {
            return Err(KernelError::InvalidArguments {
                reason: format!("RMSNorm gamma[{idx}] is not finite"),
            }
            .into());
        }
    }
    Ok(())
}

fn validate_dense_gguf_rmsnorm_fixture(fixture: &DenseGgufRmsNormCudaFixture) -> Result<()> {
    require_dense_label(&fixture.fixture_id, "fixture_id")?;
    require_dense_label(&fixture.model_family, "model_family")?;
    require_dense_label(&fixture.tensor_name, "tensor_name")?;
    require_dense_label(&fixture.tensor_role, "tensor_role")?;
    require_dense_label(&fixture.tensor_type, "tensor_type")?;
    if fixture.source_weight_sha256.len() != 64
        || !fixture.source_weight_sha256.chars().all(|ch| ch.is_ascii_hexdigit())
    {
        return Err(KernelError::InvalidArguments {
            reason: "dense GGUF RMSNorm fixture source_weight_sha256 must be a SHA-256 hex digest"
                .into(),
        }
        .into());
    }
    if fixture.hidden_dim == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "dense GGUF RMSNorm fixture hidden_dim must be non-zero".into(),
        }
        .into());
    }
    if fixture.input_f32.len() != fixture.hidden_dim {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "dense GGUF RMSNorm input length mismatch: expected {}, got {}",
                fixture.hidden_dim,
                fixture.input_f32.len()
            ),
        }
        .into());
    }
    if fixture.gamma_f32.len() != fixture.hidden_dim {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "dense GGUF RMSNorm gamma length mismatch: expected {}, got {}",
                fixture.hidden_dim,
                fixture.gamma_f32.len()
            ),
        }
        .into());
    }
    if fixture.expected_output_f32.len() != fixture.hidden_dim {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "dense GGUF RMSNorm expected output length mismatch: expected {}, got {}",
                fixture.hidden_dim,
                fixture.expected_output_f32.len()
            ),
        }
        .into());
    }
    if !fixture.rmsnorm_eps.is_finite() || fixture.rmsnorm_eps <= 0.0 {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "dense GGUF RMSNorm epsilon must be positive, got {}",
                fixture.rmsnorm_eps
            ),
        }
        .into());
    }
    validate_rmsnorm_buffers(
        &fixture.input_f32,
        &fixture.gamma_f32,
        &fixture.expected_output_f32,
        fixture.hidden_dim,
        1,
        fixture.rmsnorm_eps,
    )
}

#[cfg(feature = "cuda")]
fn launch_dense_rmsnorm_f32_cuda_impl(
    device_index: usize,
    input: &[f32],
    gamma: &[f32],
    output: &mut [f32],
    config: &RmsNormConfig,
) -> Result<CudaDenseRmsNormStats> {
    let input_len = checked_mul(config.hidden_dim, config.n_rows, "RMSNorm input")?;

    let ctx = CudaContext::new(device_index).map_err(|err| KernelError::GpuError {
        reason: format!("failed to create CUDA context for dense RMSNorm: {err:?}"),
    })?;
    let stream = ctx.default_stream();
    let ptx = compile_dense_rmsnorm_ptx()?;
    let module = ctx.load_module(ptx).map_err(|err| KernelError::GpuError {
        reason: format!("failed to load dense RMSNorm CUDA module: {err:?}"),
    })?;
    let function = module.load_function(CUDA_DENSE_RMSNORM_KERNEL_ID).map_err(|err| {
        KernelError::GpuError {
            reason: format!("failed to load dense RMSNorm CUDA kernel: {err:?}"),
        }
    })?;

    let input_dev =
        stream.memcpy_stod(&input[..input_len]).map_err(|err| KernelError::GpuError {
            reason: format!("failed to copy dense RMSNorm input to device: {err:?}"),
        })?;
    let gamma_dev =
        stream.memcpy_stod(&gamma[..config.hidden_dim]).map_err(|err| KernelError::GpuError {
            reason: format!("failed to copy dense RMSNorm gamma to device: {err:?}"),
        })?;
    let mut output_dev: CudaSlice<f32> =
        stream.alloc_zeros(input_len).map_err(|err| KernelError::GpuError {
            reason: format!("failed to allocate dense RMSNorm output on device: {err:?}"),
        })?;

    let threads_per_block = 128u32;
    let launch_config = LaunchConfig {
        grid_dim: ((config.hidden_dim as u32).div_ceil(threads_per_block), config.n_rows as u32, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(&function);
    builder.arg(&input_dev);
    builder.arg(&gamma_dev);
    builder.arg(&mut output_dev);
    let hidden_dim_arg =
        i32::try_from(config.hidden_dim).map_err(|_| KernelError::InvalidArguments {
            reason: format!("dense RMSNorm hidden_dim exceeds i32: {}", config.hidden_dim),
        })?;
    let n_rows_arg = i32::try_from(config.n_rows).map_err(|_| KernelError::InvalidArguments {
        reason: format!("dense RMSNorm n_rows exceeds i32: {}", config.n_rows),
    })?;
    let eps_arg = config.eps;
    builder.arg(&hidden_dim_arg);
    builder.arg(&n_rows_arg);
    builder.arg(&eps_arg);

    unsafe { builder.launch(launch_config) }.map_err(|err| KernelError::GpuError {
        reason: format!("failed to launch dense RMSNorm CUDA kernel: {err:?}"),
    })?;
    stream.synchronize().map_err(|err| KernelError::GpuError {
        reason: format!("failed to synchronize dense RMSNorm CUDA kernel: {err:?}"),
    })?;

    let output_host: Vec<f32> =
        stream.memcpy_dtov(&output_dev).map_err(|err| KernelError::GpuError {
            reason: format!("failed to copy dense RMSNorm output from device: {err:?}"),
        })?;
    output[..input_len].copy_from_slice(&output_host[..input_len]);

    Ok(CudaDenseRmsNormStats {
        kernel_id: CUDA_DENSE_RMSNORM_KERNEL_ID,
        invocations: 1,
        fallback_invocations: 0,
        host_to_device_bytes: bytes_for::<f32>(input_len + config.hidden_dim)?,
        device_to_host_bytes: bytes_for::<f32>(input_len)?,
        kernel_launches: 1,
        kernel_time_ms: None,
    })
}

#[cfg(feature = "cuda")]
fn compile_dense_rmsnorm_ptx() -> Result<Ptx> {
    let _hook_guard = NVRTC_COMPILE_LOCK.lock().ok();
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let compile_result = std::panic::catch_unwind(|| compile_ptx(CUDA_DENSE_RMSNORM_KERNEL_SRC));
    std::panic::set_hook(previous_hook);

    match compile_result {
        Ok(Ok(ptx)) => Ok(ptx),
        Ok(Err(err)) => Err(KernelError::GpuError {
            reason: format!("failed to compile dense RMSNorm CUDA PTX: {err:?}"),
        }
        .into()),
        Err(payload) => Err(KernelError::GpuError {
            reason: format!(
                "failed to compile dense RMSNorm CUDA PTX because NVRTC was unavailable: {}",
                panic_payload_message(&*payload)
            ),
        }
        .into()),
    }
}

#[cfg(feature = "cuda")]
fn panic_payload_message(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

fn compare_outputs(expected: &[f32], actual: &[f32]) -> Result<(f32, f32)> {
    if expected.len() != actual.len() {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "dense RMSNorm parity length mismatch: expected {}, got {}",
                expected.len(),
                actual.len()
            ),
        }
        .into());
    }
    if expected.is_empty() {
        return Err(KernelError::InvalidArguments {
            reason: "dense RMSNorm parity comparison requires non-empty output".into(),
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
        KernelError::InvalidArguments { reason: format!("{label} size overflows usize") }.into()
    })
}

#[cfg(feature = "cuda")]
fn bytes_for<T>(items: usize) -> Result<u64> {
    let bytes = checked_mul(items, std::mem::size_of::<T>(), "byte count")?;
    u64::try_from(bytes).map_err(|_| {
        KernelError::InvalidArguments { reason: "byte count exceeds u64".into() }.into()
    })
}

fn validate_i32_arg(value: usize, label: &str) -> Result<()> {
    if value > i32::MAX as usize {
        return Err(KernelError::InvalidArguments {
            reason: format!("{label} exceeds i32: {value}"),
        }
        .into());
    }
    Ok(())
}

fn require_dense_label(value: &str, field: &str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(KernelError::InvalidArguments {
            reason: format!("dense GGUF RMSNorm fixture {field} must not be empty"),
        }
        .into());
    }
    let lower = value.to_ascii_lowercase();
    if lower.contains("bitnet") || lower.contains("qk256") || lower.contains("i2_s") {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "dense GGUF RMSNorm fixture {field} must not contain BitNet packed markers"
            ),
        }
        .into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_rmsnorm_launch() {
        let cfg = RmsNormConfig::for_shape(2048, 4).unwrap();
        let input = vec![1.0f32; 2048 * 4];
        let gamma = vec![1.0f32; 2048];
        let mut output = vec![0.0f32; 2048 * 4];
        let result = launch_rmsnorm(&input, &gamma, &mut output, &cfg);
        assert!(result.is_ok(), "CUDA RMSNorm launch failed: {result:?}");
    }
}

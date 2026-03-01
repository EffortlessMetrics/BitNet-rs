//! CUDA kernel fusion for common operation pairs.
//!
//! Fusing consecutive operations into a single GPU kernel eliminates
//! intermediate global-memory round-trips and kernel launch overhead.
//! Each fused kernel is numerically equivalent (within FP32 tolerance)
//! to running the constituent operations separately.
//!
//! # Fused operations
//!
//! | Fused kernel | Components | Benefit |
//! |---|---|---|
//! | **RMSNorm+Linear** | RMSNorm → matmul | 1 read of `input` instead of 2 |
//! | **GELU+Linear** | GELU → matmul + bias | skip activation buffer |
//! | **Softmax+Mask** | mask + scale + softmax | single-pass row reduction |
//! | **Add+RMSNorm** | residual add → RMSNorm | fuse residual into norm |
//! | **Scale+Add** | `a + scale * b` | pure memory-bound fusion |
//!
//! # Kernel strategy
//!
//! Vector-length operations (scale-add, add-normalize) use grid-stride
//! loops with 256 threads per block.  Matmul-fused operations
//! (rmsnorm-linear, gelu-linear) use one block per output row and
//! reduce within the block.  Softmax-mask uses one block per row with
//! a two-pass (max then exp-sum) warp reduction.
//!
//! # CPU fallback
//!
//! Every fused operation has a pure-Rust scalar fallback that is always
//! compiled and used when no GPU is available.  The CUDA launch
//! functions are gated behind
//! `#[cfg(any(feature = "gpu", feature = "cuda"))]`.

use std::fmt;

use bitnet_common::{KernelError, Result};

// ───────────────────────────────────────────────────────────────────
// PTX source
// ───────────────────────────────────────────────────────────────────

/// Inline CUDA C source for fused kernels.
///
/// Contains five kernels that fuse common operation pairs found in
/// BitNet transformer inference.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const FUSION_KERNEL_SRC: &str = r#"
// Fused RMSNorm + linear projection.
// Grid: (out_dim, 1, 1)  Block: (tpb, 1, 1)
// Shared memory: tpb * sizeof(float)
extern "C" __global__ void fused_rmsnorm_linear_f32(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int n,
    int out_dim,
    float eps)
{
    int row = blockIdx.x;
    if (row >= out_dim) return;
    extern __shared__ float sdata[];

    // Phase 1: partial sum-of-squares for RMSNorm.
    float local_ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = input[i];
        local_ss += v * v;
    }
    sdata[threadIdx.x] = local_ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float inv_rms = rsqrtf(sdata[0] / (float)n + eps);

    // Phase 2: dot(weight_row, normed_input).
    const float* w = weight + (long long)row * n;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        acc += w[i] * (input[i] * gamma[i] * inv_rms);
    }
    sdata[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) output[row] = sdata[0];
}

// Fused GELU + linear projection.
// Grid: (out_dim, 1, 1)  Block: (tpb, 1, 1)
extern "C" __global__ void fused_gelu_linear_f32(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int n,
    int out_dim,
    int has_bias)
{
    int row = blockIdx.x;
    if (row >= out_dim) return;
    extern __shared__ float sdata[];
    const float SQRT_2_OVER_PI = 0.7978845608f;
    const float COEFF = 0.044715f;

    const float* w = weight + (long long)row * n;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float x = input[i];
        float x3 = x * x * x;
        float g = 0.5f * x
            * (1.0f + tanhf(SQRT_2_OVER_PI * (x + COEFF * x3)));
        acc += w[i] * g;
    }
    sdata[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float val = sdata[0];
        if (has_bias) val += bias[row];
        output[row] = val;
    }
}

// Fused mask + scale + softmax (one row).
// Grid: (1, 1, 1)  Block: (tpb, 1, 1)
extern "C" __global__ void fused_softmax_mask_f32(
    const float* __restrict__ scores,
    const float* __restrict__ mask,
    float* __restrict__ output,
    int n,
    float scale)
{
    extern __shared__ float sdata[];

    // Pass 1: find max.
    float local_max = -1e30f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = (scores[i] + mask[i]) * scale;
        if (v > local_max) local_max = v;
    }
    sdata[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] =
                fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    float max_val = sdata[0];

    // Pass 2: exp and sum.
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = expf((scores[i] + mask[i]) * scale - max_val);
        output[i] = v;
        local_sum += v;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float inv_sum = (sdata[0] > 0.0f) ? (1.0f / sdata[0]) : 0.0f;

    // Pass 3: normalize.
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        output[i] *= inv_sum;
    }
}

// Fused residual-add + RMSNorm.
// Grid: (1, 1, 1)  Block: (tpb, 1, 1)
extern "C" __global__ void fused_add_rmsnorm_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ gamma,
    float* __restrict__ output,
    int n,
    float eps)
{
    extern __shared__ float sdata[];
    float local_ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float c = a[i] + b[i];
        output[i] = c;
        local_ss += c * c;
    }
    sdata[threadIdx.x] = local_ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float inv_rms = rsqrtf(sdata[0] / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        output[i] = output[i] * gamma[i] * inv_rms;
    }
}

// Fused scale-and-add: output = a + scale * b.
extern "C" __global__ void fused_scale_add_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ output,
    int n,
    float scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        output[i] = a[i] + scale * b[i];
    }
}
"#;

// ───────────────────────────────────────────────────────────────────
// Configuration
// ───────────────────────────────────────────────────────────────────

/// Controls which fused CUDA kernels are eligible for dispatch.
#[derive(Debug, Clone)]
pub struct FusionConfig {
    /// Fuse RMSNorm + linear projection into a single kernel.
    pub enable_rmsnorm_linear: bool,
    /// Fuse GELU activation + linear projection.
    pub enable_gelu_linear: bool,
    /// Fuse masking + scaling + softmax.
    pub enable_softmax_mask: bool,
    /// Minimum element count before fusion is attempted.
    pub min_fusion_size: usize,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            enable_rmsnorm_linear: true,
            enable_gelu_linear: true,
            enable_softmax_mask: true,
            min_fusion_size: 32,
        }
    }
}

impl FusionConfig {
    /// All fusions disabled — useful as a comparison baseline.
    pub fn disabled() -> Self {
        Self {
            enable_rmsnorm_linear: false,
            enable_gelu_linear: false,
            enable_softmax_mask: false,
            min_fusion_size: usize::MAX,
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> std::result::Result<(), FusionError> {
        if self.min_fusion_size == 0 {
            return Err(FusionError::InvalidConfig("min_fusion_size must be > 0".into()));
        }
        Ok(())
    }
}

// ───────────────────────────────────────────────────────────────────
// Fused operation identifier
// ───────────────────────────────────────────────────────────────────

/// Identifies a specific fused operation for logging / profiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusedOp {
    /// Fused RMSNorm followed by linear projection.
    RmsNormLinear,
    /// Fused GELU activation followed by linear projection.
    GeluLinear,
    /// Fused additive-mask + scale + softmax.
    SoftmaxMask,
    /// Fused residual addition followed by RMSNorm.
    AddNormalize,
    /// Fused `a + scale * b`.
    ScaleAndAdd,
}

impl FusedOp {
    /// CUDA kernel function name for this fused operation.
    pub fn kernel_name(&self) -> &'static str {
        match self {
            Self::RmsNormLinear => "fused_rmsnorm_linear_f32",
            Self::GeluLinear => "fused_gelu_linear_f32",
            Self::SoftmaxMask => "fused_softmax_mask_f32",
            Self::AddNormalize => "fused_add_rmsnorm_f32",
            Self::ScaleAndAdd => "fused_scale_add_f32",
        }
    }
}

impl fmt::Display for FusedOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RmsNormLinear => write!(f, "RmsNormLinear"),
            Self::GeluLinear => write!(f, "GeluLinear"),
            Self::SoftmaxMask => write!(f, "SoftmaxMask"),
            Self::AddNormalize => write!(f, "AddNormalize"),
            Self::ScaleAndAdd => write!(f, "ScaleAndAdd"),
        }
    }
}

// ───────────────────────────────────────────────────────────────────
// Error type
// ───────────────────────────────────────────────────────────────────

/// Errors specific to fusion kernel dispatch.
#[derive(Debug, Clone, PartialEq)]
pub enum FusionError {
    /// Tensor dimensions do not match.
    DimensionMismatch { expected: usize, got: usize },
    /// Configuration is invalid.
    InvalidConfig(String),
    /// An input tensor is empty.
    EmptyInput,
}

impl fmt::Display for FusionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::InvalidConfig(msg) => {
                write!(f, "invalid config: {msg}")
            }
            Self::EmptyInput => write!(f, "empty input"),
        }
    }
}

impl std::error::Error for FusionError {}

// ───────────────────────────────────────────────────────────────────
// Launch configuration helpers
// ───────────────────────────────────────────────────────────────────

/// Launch configuration for fused matmul-style kernels where one
/// block processes one output row.
#[derive(Debug, Clone)]
pub struct FusedMatmulLaunchConfig {
    /// Input vector length.
    pub n: usize,
    /// Number of output rows.
    pub out_dim: usize,
    /// Threads per block (capped at 256).
    pub threads_per_block: u32,
}

impl FusedMatmulLaunchConfig {
    /// Create a launch config for the given dimensions.
    pub fn new(n: usize, out_dim: usize) -> Result<Self> {
        if n == 0 || out_dim == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "fused matmul dims must be non-zero: \
                     n={n}, out_dim={out_dim}"
                ),
            }
            .into());
        }
        let threads_per_block = (n as u32).min(256);
        Ok(Self { n, out_dim, threads_per_block })
    }

    /// CUDA grid dimensions `(out_dim, 1, 1)`.
    pub fn grid_dim(&self) -> (u32, u32, u32) {
        (self.out_dim as u32, 1, 1)
    }

    /// CUDA block dimensions.
    pub fn block_dim(&self) -> (u32, u32, u32) {
        (self.threads_per_block, 1, 1)
    }

    /// Shared memory in bytes (one float per thread).
    pub fn shared_mem_bytes(&self) -> u32 {
        self.threads_per_block * 4
    }
}

/// Launch configuration for fused element-wise kernels.
#[derive(Debug, Clone)]
pub struct FusedElementwiseLaunchConfig {
    /// Total number of elements.
    pub n: usize,
    /// Threads per block (fixed at 256).
    pub threads_per_block: u32,
}

impl FusedElementwiseLaunchConfig {
    /// Create a launch config for the given element count.
    pub fn new(n: usize) -> Result<Self> {
        if n == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "element count must be non-zero".into(),
            }
            .into());
        }
        Ok(Self { n, threads_per_block: 256 })
    }

    /// CUDA grid dimensions.
    pub fn grid_dim(&self) -> (u32, u32, u32) {
        let blocks = (self.n as u32 + self.threads_per_block - 1) / self.threads_per_block;
        (blocks, 1, 1)
    }

    /// CUDA block dimensions.
    pub fn block_dim(&self) -> (u32, u32, u32) {
        (self.threads_per_block, 1, 1)
    }
}

// ───────────────────────────────────────────────────────────────────
// CPU fallback implementations
// ───────────────────────────────────────────────────────────────────

/// Scalar GELU approximation (tanh variant).
#[inline]
fn gelu(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const COEFF: f32 = 0.044_715;
    let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// Fused RMSNorm + linear projection (CPU fallback).
///
/// * `input`  — `[n]` row vector
/// * `weight` — `[out_dim × n]` row-major weight matrix
/// * `gamma`  — `[n]` per-element RMSNorm scale
/// * `output` — `[out_dim]` (written)
/// * `eps`    — RMSNorm epsilon
pub fn fused_rmsnorm_linear_cpu(
    input: &[f32],
    weight: &[f32],
    gamma: &[f32],
    output: &mut [f32],
    eps: f32,
) -> Result<()> {
    let n = input.len();
    if n == 0 {
        return Err(
            KernelError::InvalidArguments { reason: "input must be non-empty".into() }.into()
        );
    }
    if gamma.len() != n {
        return Err(KernelError::InvalidArguments {
            reason: format!("gamma length {} != input length {n}", gamma.len()),
        }
        .into());
    }
    if weight.len() % n != 0 {
        return Err(KernelError::InvalidArguments {
            reason: format!("weight length {} not divisible by n={n}", weight.len()),
        }
        .into());
    }
    let out_dim = weight.len() / n;
    if output.len() < out_dim {
        return Err(KernelError::InvalidArguments {
            reason: format!("output length {} < out_dim {out_dim}", output.len()),
        }
        .into());
    }

    let sum_sq: f32 = input.iter().map(|&x| x * x).sum();
    let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();

    for (o, row) in output.iter_mut().zip(weight.chunks_exact(n)) {
        let mut acc = 0.0f32;
        for ((&w, &x), &g) in row.iter().zip(input).zip(gamma) {
            acc += w * (x * g * inv_rms);
        }
        *o = acc;
    }
    Ok(())
}

/// Fused GELU + linear projection (CPU fallback).
///
/// * `input`  — `[n]`
/// * `weight` — `[out_dim × n]` row-major
/// * `bias`   — `[out_dim]` or empty for no bias
/// * `output` — `[out_dim]` (written)
pub fn fused_gelu_linear_cpu(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    output: &mut [f32],
) -> Result<()> {
    let n = input.len();
    if n == 0 {
        return Err(
            KernelError::InvalidArguments { reason: "input must be non-empty".into() }.into()
        );
    }
    if weight.len() % n != 0 {
        return Err(KernelError::InvalidArguments {
            reason: format!("weight length {} not divisible by n={n}", weight.len()),
        }
        .into());
    }
    let out_dim = weight.len() / n;
    if !bias.is_empty() && bias.len() != out_dim {
        return Err(KernelError::InvalidArguments {
            reason: format!("bias length {} != out_dim {out_dim}", bias.len()),
        }
        .into());
    }
    if output.len() < out_dim {
        return Err(KernelError::InvalidArguments {
            reason: format!("output length {} < out_dim {out_dim}", output.len()),
        }
        .into());
    }

    let activated: Vec<f32> = input.iter().map(|&x| gelu(x)).collect();

    for (i, row) in weight.chunks_exact(n).enumerate() {
        let mut acc = 0.0f32;
        for (&w, &a) in row.iter().zip(&activated) {
            acc += w * a;
        }
        if !bias.is_empty() {
            acc += bias[i];
        }
        output[i] = acc;
    }
    Ok(())
}

/// Fused mask + scale + softmax (CPU fallback).
///
/// * `scores` — `[n]`
/// * `mask`   — `[n]` additive mask (0=keep, large-neg=mask out)
/// * `output` — `[n]` (written)
/// * `scale`  — scalar multiplier
pub fn fused_softmax_mask_cpu(
    scores: &[f32],
    mask: &[f32],
    output: &mut [f32],
    scale: f32,
) -> Result<()> {
    let n = scores.len();
    if n == 0 {
        return Err(
            KernelError::InvalidArguments { reason: "scores must be non-empty".into() }.into()
        );
    }
    if mask.len() != n {
        return Err(KernelError::InvalidArguments {
            reason: format!("mask length {} != scores length {n}", mask.len()),
        }
        .into());
    }
    if output.len() < n {
        return Err(KernelError::InvalidArguments {
            reason: format!("output length {} < n={n}", output.len()),
        }
        .into());
    }

    let mut max_val = f32::NEG_INFINITY;
    for (&s, &m) in scores.iter().zip(mask) {
        let v = (s + m) * scale;
        if v > max_val {
            max_val = v;
        }
    }

    let mut sum = 0.0f32;
    for ((&s, &m), o) in scores.iter().zip(mask).zip(output.iter_mut()) {
        let v = ((s + m) * scale - max_val).exp();
        *o = v;
        sum += v;
    }

    if sum > 0.0 {
        let inv = 1.0 / sum;
        for o in output[..n].iter_mut() {
            *o *= inv;
        }
    }
    Ok(())
}

/// Fused residual addition + RMSNorm (CPU fallback).
///
/// * `a`, `b` — `[n]`
/// * `gamma`  — `[n]`
/// * `output` — `[n]` (written)
/// * `eps`    — normalisation epsilon
pub fn fused_add_rmsnorm_cpu(
    a: &[f32],
    b: &[f32],
    gamma: &[f32],
    output: &mut [f32],
    eps: f32,
) -> Result<()> {
    let n = a.len();
    if n == 0 {
        return Err(
            KernelError::InvalidArguments { reason: "input must be non-empty".into() }.into()
        );
    }
    if b.len() != n {
        return Err(KernelError::InvalidArguments {
            reason: format!("b length {} != a length {n}", b.len()),
        }
        .into());
    }
    if gamma.len() != n {
        return Err(KernelError::InvalidArguments {
            reason: format!("gamma length {} != a length {n}", gamma.len()),
        }
        .into());
    }
    if output.len() < n {
        return Err(KernelError::InvalidArguments {
            reason: format!("output length {} < n={n}", output.len()),
        }
        .into());
    }

    let mut sum_sq = 0.0f32;
    for (((&ai, &bi), _g), o) in a.iter().zip(b).zip(gamma).zip(output.iter_mut()) {
        let c = ai + bi;
        sum_sq += c * c;
        *o = c;
    }

    let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();

    for (o, &g) in output[..n].iter_mut().zip(gamma) {
        *o *= g * inv_rms;
    }
    Ok(())
}

/// Fused scale-and-add: `output = a + scale * b` (CPU fallback).
///
/// * `a`, `b` — `[n]`
/// * `output` — `[n]` (written)
/// * `scale`  — scalar multiplier for `b`
pub fn fused_scale_add_cpu(a: &[f32], b: &[f32], output: &mut [f32], scale: f32) -> Result<()> {
    let n = a.len();
    if n == 0 {
        return Err(
            KernelError::InvalidArguments { reason: "input must be non-empty".into() }.into()
        );
    }
    if b.len() != n {
        return Err(KernelError::InvalidArguments {
            reason: format!("b length {} != a length {n}", b.len()),
        }
        .into());
    }
    if output.len() < n {
        return Err(KernelError::InvalidArguments {
            reason: format!("output length {} < n={n}", output.len()),
        }
        .into());
    }

    for ((&ai, &bi), o) in a.iter().zip(b).zip(output.iter_mut()) {
        *o = ai + scale * bi;
    }
    Ok(())
}

// ───────────────────────────────────────────────────────────────────
// CUDA launch stubs
// ───────────────────────────────────────────────────────────────────

/// Launch the fused RMSNorm + linear CUDA kernel.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_fused_rmsnorm_linear_cuda(
    _input: &[f32],
    _weight: &[f32],
    _gamma: &[f32],
    _output: &mut [f32],
    config: &FusedMatmulLaunchConfig,
    _eps: f32,
) -> Result<()> {
    log::debug!(
        "fused RMSNorm+linear CUDA: n={}, out_dim={}, grid={:?}",
        config.n,
        config.out_dim,
        config.grid_dim(),
    );
    Err(KernelError::GpuError {
        reason: "fused RMSNorm+linear CUDA kernel not yet \
                 compiled — scaffold only"
            .into(),
    }
    .into())
}

/// Launch the fused GELU + linear CUDA kernel.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_fused_gelu_linear_cuda(
    _input: &[f32],
    _weight: &[f32],
    _bias: &[f32],
    _output: &mut [f32],
    config: &FusedMatmulLaunchConfig,
) -> Result<()> {
    log::debug!(
        "fused GELU+linear CUDA: n={}, out_dim={}, grid={:?}",
        config.n,
        config.out_dim,
        config.grid_dim(),
    );
    Err(KernelError::GpuError {
        reason: "fused GELU+linear CUDA kernel not yet \
                 compiled — scaffold only"
            .into(),
    }
    .into())
}

/// Launch the fused softmax + mask CUDA kernel.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_fused_softmax_mask_cuda(
    _scores: &[f32],
    _mask: &[f32],
    _output: &mut [f32],
    config: &FusedElementwiseLaunchConfig,
    _scale: f32,
) -> Result<()> {
    log::debug!("fused softmax+mask CUDA: n={}, grid={:?}", config.n, config.grid_dim(),);
    Err(KernelError::GpuError {
        reason: "fused softmax+mask CUDA kernel not yet \
                 compiled — scaffold only"
            .into(),
    }
    .into())
}

/// Launch the fused add + RMSNorm CUDA kernel.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_fused_add_rmsnorm_cuda(
    _a: &[f32],
    _b: &[f32],
    _gamma: &[f32],
    _output: &mut [f32],
    config: &FusedElementwiseLaunchConfig,
    _eps: f32,
) -> Result<()> {
    log::debug!("fused add+RMSNorm CUDA: n={}, grid={:?}", config.n, config.grid_dim(),);
    Err(KernelError::GpuError {
        reason: "fused add+RMSNorm CUDA kernel not yet \
                 compiled — scaffold only"
            .into(),
    }
    .into())
}

/// Launch the fused scale-and-add CUDA kernel.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_fused_scale_add_cuda(
    _a: &[f32],
    _b: &[f32],
    _output: &mut [f32],
    config: &FusedElementwiseLaunchConfig,
    _scale: f32,
) -> Result<()> {
    log::debug!("fused scale+add CUDA: n={}, grid={:?}", config.n, config.grid_dim(),);
    Err(KernelError::GpuError {
        reason: "fused scale+add CUDA kernel not yet \
                 compiled — scaffold only"
            .into(),
    }
    .into())
}

// ───────────────────────────────────────────────────────────────────
// Unified dispatch (CPU fallback / CUDA)
// ───────────────────────────────────────────────────────────────────

/// Dispatch fused RMSNorm + linear: CUDA when available, else CPU.
pub fn fused_rmsnorm_linear(
    input: &[f32],
    weight: &[f32],
    gamma: &[f32],
    output: &mut [f32],
    eps: f32,
) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if let Ok(cfg) = FusedMatmulLaunchConfig::new(input.len(), output.len()) {
            if launch_fused_rmsnorm_linear_cuda(input, weight, gamma, output, &cfg, eps).is_ok() {
                return Ok(());
            }
        }
    }
    fused_rmsnorm_linear_cpu(input, weight, gamma, output, eps)
}

/// Dispatch fused GELU + linear: CUDA when available, else CPU.
pub fn fused_gelu_linear(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    output: &mut [f32],
) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if let Ok(cfg) = FusedMatmulLaunchConfig::new(input.len(), output.len()) {
            if launch_fused_gelu_linear_cuda(input, weight, bias, output, &cfg).is_ok() {
                return Ok(());
            }
        }
    }
    fused_gelu_linear_cpu(input, weight, bias, output)
}

/// Dispatch fused softmax + mask: CUDA when available, else CPU.
pub fn fused_softmax_mask(
    scores: &[f32],
    mask: &[f32],
    output: &mut [f32],
    scale: f32,
) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if let Ok(cfg) = FusedElementwiseLaunchConfig::new(scores.len()) {
            if launch_fused_softmax_mask_cuda(scores, mask, output, &cfg, scale).is_ok() {
                return Ok(());
            }
        }
    }
    fused_softmax_mask_cpu(scores, mask, output, scale)
}

/// Dispatch fused residual-add + RMSNorm.
pub fn fused_add_rmsnorm(
    a: &[f32],
    b: &[f32],
    gamma: &[f32],
    output: &mut [f32],
    eps: f32,
) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if let Ok(cfg) = FusedElementwiseLaunchConfig::new(a.len()) {
            if launch_fused_add_rmsnorm_cuda(a, b, gamma, output, &cfg, eps).is_ok() {
                return Ok(());
            }
        }
    }
    fused_add_rmsnorm_cpu(a, b, gamma, output, eps)
}

/// Dispatch fused scale-and-add.
pub fn fused_scale_add(a: &[f32], b: &[f32], output: &mut [f32], scale: f32) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if let Ok(cfg) = FusedElementwiseLaunchConfig::new(a.len()) {
            if launch_fused_scale_add_cuda(a, b, output, &cfg, scale).is_ok() {
                return Ok(());
            }
        }
    }
    fused_scale_add_cpu(a, b, output, scale)
}

// ───────────────────────────────────────────────────────────────────
// Tests
// ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;
    const TOL: f32 = 1e-5;

    fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
    }

    // Reference (unfused) helpers ───────────────────────────────────

    fn ref_rmsnorm_linear(input: &[f32], weight: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
        let n = input.len();
        let out_dim = weight.len() / n;
        let sum_sq: f32 = input.iter().map(|&x| x * x).sum();
        let rms = (sum_sq / n as f32 + eps).sqrt();
        let normed: Vec<f32> = input.iter().zip(gamma).map(|(&x, &g)| x * g / rms).collect();
        let mut out = vec![0.0f32; out_dim];
        for (o, row) in out.iter_mut().zip(weight.chunks_exact(n)) {
            *o = row.iter().zip(&normed).map(|(&w, &x)| w * x).sum();
        }
        out
    }

    fn ref_gelu_linear(input: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
        let n = input.len();
        let out_dim = weight.len() / n;
        let act: Vec<f32> = input.iter().map(|&x| gelu(x)).collect();
        let mut out = vec![0.0f32; out_dim];
        for (i, row) in weight.chunks_exact(n).enumerate() {
            let mut acc: f32 = row.iter().zip(&act).map(|(&w, &a)| w * a).sum();
            if !bias.is_empty() {
                acc += bias[i];
            }
            out[i] = acc;
        }
        out
    }

    fn ref_softmax_mask(scores: &[f32], mask: &[f32], scale: f32) -> Vec<f32> {
        let scaled: Vec<f32> = scores.iter().zip(mask).map(|(&s, &m)| (s + m) * scale).collect();
        let mx = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scaled.iter().map(|&x| (x - mx).exp()).collect();
        let sum: f32 = exps.iter().sum();
        if sum > 0.0 { exps.iter().map(|&e| e / sum).collect() } else { exps }
    }

    fn ref_add_rmsnorm(a: &[f32], b: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
        let combined: Vec<f32> = a.iter().zip(b).map(|(&x, &y)| x + y).collect();
        let n = combined.len();
        let ss: f32 = combined.iter().map(|&x| x * x).sum();
        let rms = (ss / n as f32 + eps).sqrt();
        combined.iter().zip(gamma).map(|(&c, &g)| c * g / rms).collect()
    }

    // ── FusionConfig ───────────────────────────────────────────────

    #[test]
    fn config_default_enables_all() {
        let cfg = FusionConfig::default();
        assert!(cfg.enable_rmsnorm_linear);
        assert!(cfg.enable_gelu_linear);
        assert!(cfg.enable_softmax_mask);
        assert_eq!(cfg.min_fusion_size, 32);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn config_disabled() {
        let cfg = FusionConfig::disabled();
        assert!(!cfg.enable_rmsnorm_linear);
        assert!(!cfg.enable_gelu_linear);
        assert!(!cfg.enable_softmax_mask);
        assert_eq!(cfg.min_fusion_size, usize::MAX);
    }

    #[test]
    fn config_zero_min_rejected() {
        let cfg = FusionConfig { min_fusion_size: 0, ..FusionConfig::default() };
        assert!(cfg.validate().is_err());
    }

    // ── FusedOp ────────────────────────────────────────────────────

    #[test]
    fn fused_op_display() {
        assert_eq!(FusedOp::RmsNormLinear.to_string(), "RmsNormLinear");
        assert_eq!(FusedOp::GeluLinear.to_string(), "GeluLinear");
        assert_eq!(FusedOp::SoftmaxMask.to_string(), "SoftmaxMask");
        assert_eq!(FusedOp::AddNormalize.to_string(), "AddNormalize");
        assert_eq!(FusedOp::ScaleAndAdd.to_string(), "ScaleAndAdd");
    }

    #[test]
    fn fused_op_kernel_names() {
        assert_eq!(FusedOp::RmsNormLinear.kernel_name(), "fused_rmsnorm_linear_f32");
        assert_eq!(FusedOp::GeluLinear.kernel_name(), "fused_gelu_linear_f32");
        assert_eq!(FusedOp::SoftmaxMask.kernel_name(), "fused_softmax_mask_f32");
        assert_eq!(FusedOp::AddNormalize.kernel_name(), "fused_add_rmsnorm_f32");
        assert_eq!(FusedOp::ScaleAndAdd.kernel_name(), "fused_scale_add_f32");
    }

    // ── Launch configs ─────────────────────────────────────────────

    #[test]
    fn matmul_config_basic() {
        let c = FusedMatmulLaunchConfig::new(128, 64).unwrap();
        assert_eq!(c.n, 128);
        assert_eq!(c.out_dim, 64);
        assert_eq!(c.threads_per_block, 128);
        assert_eq!(c.grid_dim(), (64, 1, 1));
        assert_eq!(c.block_dim(), (128, 1, 1));
        assert_eq!(c.shared_mem_bytes(), 128 * 4);
    }

    #[test]
    fn matmul_config_caps_threads() {
        let c = FusedMatmulLaunchConfig::new(4096, 2).unwrap();
        assert_eq!(c.threads_per_block, 256);
    }

    #[test]
    fn matmul_config_rejects_zero() {
        assert!(FusedMatmulLaunchConfig::new(0, 4).is_err());
        assert!(FusedMatmulLaunchConfig::new(4, 0).is_err());
    }

    #[test]
    fn elem_config_basic() {
        let c = FusedElementwiseLaunchConfig::new(1024).unwrap();
        assert_eq!(c.n, 1024);
        assert_eq!(c.grid_dim(), (4, 1, 1));
        assert_eq!(c.block_dim(), (256, 1, 1));
    }

    #[test]
    fn elem_config_non_aligned() {
        let c = FusedElementwiseLaunchConfig::new(300).unwrap();
        assert_eq!(c.grid_dim(), (2, 1, 1));
    }

    #[test]
    fn elem_config_rejects_zero() {
        assert!(FusedElementwiseLaunchConfig::new(0).is_err());
    }

    // ── fused_rmsnorm_linear_cpu ───────────────────────────────────

    #[test]
    fn rmsnorm_linear_matches_ref() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let weight = vec![0.5, -0.5, 0.25, 0.1, 0.1, 0.2, 0.3, 0.4];
        let mut out = vec![0.0f32; 2];
        fused_rmsnorm_linear_cpu(&input, &weight, &gamma, &mut out, EPS).unwrap();
        let exp = ref_rmsnorm_linear(&input, &weight, &gamma, EPS);
        assert!(max_abs_err(&out, &exp) < TOL);
    }

    #[test]
    fn rmsnorm_linear_single() {
        let mut out = vec![0.0f32; 1];
        fused_rmsnorm_linear_cpu(&[3.0], &[2.0], &[1.0], &mut out, EPS).unwrap();
        let exp = ref_rmsnorm_linear(&[3.0], &[2.0], &[1.0], EPS);
        assert!(max_abs_err(&out, &exp) < TOL);
    }

    #[test]
    fn rmsnorm_linear_empty() {
        let mut out = vec![0.0f32; 1];
        assert!(fused_rmsnorm_linear_cpu(&[], &[1.0], &[], &mut out, EPS).is_err());
    }

    #[test]
    fn rmsnorm_linear_gamma_mismatch() {
        let mut out = vec![0.0f32; 2];
        assert!(fused_rmsnorm_linear_cpu(&[1.0, 2.0], &[1.0; 4], &[1.0], &mut out, EPS,).is_err());
    }

    #[test]
    fn rmsnorm_linear_large() {
        let n = 256;
        let od = 64;
        let inp: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let g = vec![1.0f32; n];
        let w: Vec<f32> = (0..od * n).map(|i| (i as f32) * 0.001).collect();
        let mut out = vec![0.0f32; od];
        fused_rmsnorm_linear_cpu(&inp, &w, &g, &mut out, EPS).unwrap();
        let exp = ref_rmsnorm_linear(&inp, &w, &g, EPS);
        assert!(max_abs_err(&out, &exp) < 1e-3);
    }

    // ── fused_gelu_linear_cpu ──────────────────────────────────────

    #[test]
    fn gelu_linear_matches_ref() {
        let inp = vec![1.0, -1.0, 0.5, -0.5];
        let w = vec![0.2, 0.3, 0.4, 0.5, -0.1, 0.6, 0.7, -0.2];
        let bias = vec![0.01, -0.01];
        let mut out = vec![0.0f32; 2];
        fused_gelu_linear_cpu(&inp, &w, &bias, &mut out).unwrap();
        let exp = ref_gelu_linear(&inp, &w, &bias);
        assert!(max_abs_err(&out, &exp) < TOL);
    }

    #[test]
    fn gelu_linear_no_bias() {
        let inp = vec![1.0, 2.0];
        let w = vec![0.5, 0.5, -0.5, -0.5];
        let mut out = vec![0.0f32; 2];
        fused_gelu_linear_cpu(&inp, &w, &[], &mut out).unwrap();
        let exp = ref_gelu_linear(&inp, &w, &[]);
        assert!(max_abs_err(&out, &exp) < TOL);
    }

    #[test]
    fn gelu_linear_zero_input() {
        let inp = vec![0.0; 4];
        let w = vec![1.0; 8];
        let bias = vec![0.5, 0.5];
        let mut out = vec![0.0f32; 2];
        fused_gelu_linear_cpu(&inp, &w, &bias, &mut out).unwrap();
        for (v, b) in out.iter().zip(&bias) {
            assert!((v - b).abs() < TOL, "{v} vs {b}");
        }
    }

    #[test]
    fn gelu_linear_empty() {
        let mut out = vec![0.0f32; 1];
        assert!(fused_gelu_linear_cpu(&[], &[], &[], &mut out).is_err());
    }

    #[test]
    fn gelu_linear_bias_mismatch() {
        let mut out = vec![0.0f32; 1];
        assert!(fused_gelu_linear_cpu(&[1.0], &[1.0], &[1.0, 2.0], &mut out,).is_err());
    }

    // ── fused_softmax_mask_cpu ─────────────────────────────────────

    #[test]
    fn softmax_mask_matches_ref() {
        let sc = vec![1.0, 2.0, 3.0, 4.0];
        let m = vec![0.0, 0.0, -1e9, 0.0];
        let mut out = vec![0.0f32; 4];
        fused_softmax_mask_cpu(&sc, &m, &mut out, 0.5).unwrap();
        let exp = ref_softmax_mask(&sc, &m, 0.5);
        assert!(max_abs_err(&out, &exp) < TOL);
    }

    #[test]
    fn softmax_mask_sums_to_one() {
        let sc = vec![2.0, 1.0, 0.5, 3.0];
        let m = vec![0.0; 4];
        let mut out = vec![0.0f32; 4];
        fused_softmax_mask_cpu(&sc, &m, &mut out, 1.0).unwrap();
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < TOL, "sum = {sum}");
    }

    #[test]
    fn softmax_mask_fully_masked() {
        let sc = vec![1.0, 2.0, 3.0];
        let m = vec![-1e30; 3];
        let mut out = vec![0.0f32; 3];
        fused_softmax_mask_cpu(&sc, &m, &mut out, 1.0).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn softmax_mask_single() {
        let mut out = vec![0.0f32; 1];
        fused_softmax_mask_cpu(&[5.0], &[0.0], &mut out, 1.0).unwrap();
        assert!((out[0] - 1.0).abs() < TOL);
    }

    #[test]
    fn softmax_mask_empty() {
        let mut out = vec![0.0f32; 1];
        assert!(fused_softmax_mask_cpu(&[], &[], &mut out, 1.0).is_err());
    }

    #[test]
    fn softmax_mask_dim_mismatch() {
        let mut out = vec![0.0f32; 2];
        assert!(fused_softmax_mask_cpu(&[1.0, 2.0], &[0.0], &mut out, 1.0,).is_err());
    }

    #[test]
    fn softmax_mask_numerical_stability() {
        let sc = vec![1000.0, 1001.0, 1002.0];
        let m = vec![0.0; 3];
        let mut out = vec![0.0f32; 3];
        fused_softmax_mask_cpu(&sc, &m, &mut out, 1.0).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < TOL);
    }

    // ── fused_add_rmsnorm_cpu ──────────────────────────────────────

    #[test]
    fn add_rmsnorm_matches_ref() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![0.5, -0.5, 0.25, -0.25];
        let g = vec![1.0; 4];
        let mut out = vec![0.0f32; 4];
        fused_add_rmsnorm_cpu(&a, &b, &g, &mut out, EPS).unwrap();
        let exp = ref_add_rmsnorm(&a, &b, &g, EPS);
        assert!(max_abs_err(&out, &exp) < TOL);
    }

    #[test]
    fn add_rmsnorm_single() {
        let mut out = vec![0.0f32; 1];
        fused_add_rmsnorm_cpu(&[3.0], &[1.0], &[1.0], &mut out, EPS).unwrap();
        let exp = ref_add_rmsnorm(&[3.0], &[1.0], &[1.0], EPS);
        assert!(max_abs_err(&out, &exp) < TOL);
    }

    #[test]
    fn add_rmsnorm_zero_residual() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.0; 3];
        let g = vec![1.0; 3];
        let mut out = vec![0.0f32; 3];
        fused_add_rmsnorm_cpu(&a, &b, &g, &mut out, EPS).unwrap();
        let exp = ref_add_rmsnorm(&a, &b, &g, EPS);
        assert!(max_abs_err(&out, &exp) < TOL);
    }

    #[test]
    fn add_rmsnorm_empty() {
        let mut out = vec![0.0f32; 1];
        assert!(fused_add_rmsnorm_cpu(&[], &[], &[], &mut out, EPS).is_err());
    }

    #[test]
    fn add_rmsnorm_b_mismatch() {
        let mut out = vec![0.0f32; 2];
        assert!(fused_add_rmsnorm_cpu(&[1.0, 2.0], &[1.0], &[1.0, 2.0], &mut out, EPS,).is_err());
    }

    #[test]
    fn add_rmsnorm_gamma_mismatch() {
        let mut out = vec![0.0f32; 1];
        assert!(fused_add_rmsnorm_cpu(&[1.0], &[1.0], &[1.0, 2.0], &mut out, EPS,).is_err());
    }

    // ── fused_scale_add_cpu ────────────────────────────────────────

    #[test]
    fn scale_add_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let mut out = vec![0.0f32; 3];
        fused_scale_add_cpu(&a, &b, &mut out, 0.5).unwrap();
        assert!(max_abs_err(&out, &[3.0, 4.5, 6.0]) < TOL);
    }

    #[test]
    fn scale_add_zero_scale() {
        let a = vec![1.0, 2.0];
        let b = vec![100.0, 200.0];
        let mut out = vec![0.0f32; 2];
        fused_scale_add_cpu(&a, &b, &mut out, 0.0).unwrap();
        assert!(max_abs_err(&out, &a) < TOL);
    }

    #[test]
    fn scale_add_negative() {
        let a = vec![10.0, 20.0];
        let b = vec![10.0, 20.0];
        let mut out = vec![0.0f32; 2];
        fused_scale_add_cpu(&a, &b, &mut out, -1.0).unwrap();
        assert!(max_abs_err(&out, &[0.0, 0.0]) < TOL);
    }

    #[test]
    fn scale_add_single() {
        let mut out = vec![0.0f32; 1];
        fused_scale_add_cpu(&[3.0], &[4.0], &mut out, 2.0).unwrap();
        assert!((out[0] - 11.0).abs() < TOL);
    }

    #[test]
    fn scale_add_empty() {
        let mut out = vec![0.0f32; 1];
        assert!(fused_scale_add_cpu(&[], &[], &mut out, 1.0).is_err());
    }

    #[test]
    fn scale_add_dim_mismatch() {
        let mut out = vec![0.0f32; 2];
        assert!(fused_scale_add_cpu(&[1.0, 2.0], &[1.0], &mut out, 1.0).is_err());
    }

    #[test]
    fn scale_add_various_sizes() {
        for &n in &[1, 7, 8, 15, 16, 31, 32, 64, 128, 256, 1024] {
            let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..n).map(|i| -(i as f32)).collect();
            let mut out = vec![0.0f32; n];
            fused_scale_add_cpu(&a, &b, &mut out, 1.0).unwrap();
            assert!(out.iter().all(|&v| v.abs() < TOL), "failed for n={n}");
        }
    }

    // ── Unified dispatch (CPU fallback) ────────────────────────────

    #[test]
    fn dispatch_rmsnorm_linear() {
        let inp = vec![1.0, 2.0, 3.0, 4.0];
        let g = vec![1.0; 4];
        let w = vec![0.5, -0.5, 0.25, 0.1, 0.1, 0.2, 0.3, 0.4];
        let mut out = vec![0.0f32; 2];
        fused_rmsnorm_linear(&inp, &w, &g, &mut out, EPS).unwrap();
        let exp = ref_rmsnorm_linear(&inp, &w, &g, EPS);
        assert!(max_abs_err(&out, &exp) < TOL);
    }

    #[test]
    fn dispatch_gelu_linear() {
        let inp = vec![1.0, -1.0, 0.5, -0.5];
        let w = vec![0.2, 0.3, 0.4, 0.5, -0.1, 0.6, 0.7, -0.2];
        let bias = vec![0.01, -0.01];
        let mut out = vec![0.0f32; 2];
        fused_gelu_linear(&inp, &w, &bias, &mut out).unwrap();
        let exp = ref_gelu_linear(&inp, &w, &bias);
        assert!(max_abs_err(&out, &exp) < TOL);
    }

    #[test]
    fn dispatch_softmax_mask() {
        let sc = vec![1.0, 2.0, 3.0, 4.0];
        let m = vec![0.0; 4];
        let mut out = vec![0.0f32; 4];
        fused_softmax_mask(&sc, &m, &mut out, 1.0).unwrap();
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < TOL);
    }

    #[test]
    fn dispatch_add_rmsnorm() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.5, -0.5, 0.25];
        let g = vec![1.0; 3];
        let mut out = vec![0.0f32; 3];
        fused_add_rmsnorm(&a, &b, &g, &mut out, EPS).unwrap();
        let exp = ref_add_rmsnorm(&a, &b, &g, EPS);
        assert!(max_abs_err(&out, &exp) < TOL);
    }

    #[test]
    fn dispatch_scale_add() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let mut out = vec![0.0f32; 3];
        fused_scale_add(&a, &b, &mut out, 0.5).unwrap();
        assert!(max_abs_err(&out, &[3.0, 4.5, 6.0]) < TOL);
    }

    // ── CUDA launch tests (GPU hardware required) ──────────────────

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn cuda_fused_rmsnorm_linear_launch() {
        let inp = vec![1.0f32; 128];
        let g = vec![1.0f32; 128];
        let w = vec![0.01f32; 64 * 128];
        let mut out = vec![0.0f32; 64];
        let cfg = FusedMatmulLaunchConfig::new(128, 64).unwrap();
        let r = launch_fused_rmsnorm_linear_cuda(&inp, &w, &g, &mut out, &cfg, 1e-5);
        assert!(r.is_ok(), "launch failed: {r:?}");
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn cuda_fused_gelu_linear_launch() {
        let inp = vec![1.0f32; 128];
        let w = vec![0.01f32; 64 * 128];
        let bias = vec![0.0f32; 64];
        let mut out = vec![0.0f32; 64];
        let cfg = FusedMatmulLaunchConfig::new(128, 64).unwrap();
        let r = launch_fused_gelu_linear_cuda(&inp, &w, &bias, &mut out, &cfg);
        assert!(r.is_ok(), "launch failed: {r:?}");
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn cuda_fused_softmax_mask_launch() {
        let sc = vec![1.0f32; 256];
        let m = vec![0.0f32; 256];
        let mut out = vec![0.0f32; 256];
        let cfg = FusedElementwiseLaunchConfig::new(256).unwrap();
        let r = launch_fused_softmax_mask_cuda(&sc, &m, &mut out, &cfg, 1.0);
        assert!(r.is_ok(), "launch failed: {r:?}");
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn cuda_fused_add_rmsnorm_launch() {
        let a = vec![1.0f32; 256];
        let b = vec![0.5f32; 256];
        let g = vec![1.0f32; 256];
        let mut out = vec![0.0f32; 256];
        let cfg = FusedElementwiseLaunchConfig::new(256).unwrap();
        let r = launch_fused_add_rmsnorm_cuda(&a, &b, &g, &mut out, &cfg, 1e-5);
        assert!(r.is_ok(), "launch failed: {r:?}");
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn cuda_fused_scale_add_launch() {
        let a = vec![1.0f32; 256];
        let b = vec![2.0f32; 256];
        let mut out = vec![0.0f32; 256];
        let cfg = FusedElementwiseLaunchConfig::new(256).unwrap();
        let r = launch_fused_scale_add_cuda(&a, &b, &mut out, &cfg, 0.5);
        assert!(r.is_ok(), "launch failed: {r:?}");
    }
}

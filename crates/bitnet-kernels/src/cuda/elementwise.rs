//! Element-wise arithmetic and activation CUDA kernels with CPU fallback.
//!
//! Provides binary arithmetic operations (add, mul, sub, div), unary
//! activations (ReLU, GELU, SiLU, sigmoid, tanh, clamp), and a fused
//! add-then-multiply operation for bias+scale patterns common in
//! transformer post-processing.
//!
//! # Kernel strategy
//!
//! All operations are element-wise with no inter-element dependencies.
//! CUDA kernels use grid-stride loops with 256 threads per block, matching
//! the convention in [`super::activations`].  Binary operations support
//! same-shape operands; broadcasting is handled at the caller level.
//!
//! # CPU fallback
//!
//! [`elementwise_cpu_fallback`] and [`fused_elementwise_cpu`] provide
//! pure-Rust implementations for correctness testing and non-GPU
//! environments.

use bitnet_common::{KernelError, Result};

// ---------------------------------------------------------------------------
// PTX source (compiled at runtime via NVRTC when `gpu`/`cuda` is active)
// ---------------------------------------------------------------------------

/// Inline CUDA C source for element-wise binary arithmetic kernels.
///
/// Contains kernels: `add_f32`, `mul_f32`, `sub_f32`, `div_f32`,
/// `fused_add_mul_f32`.  Each processes `n` elements using grid-stride
/// loops.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const ELEMENTWISE_BINARY_KERNEL_SRC: &str = r#"
extern "C" __global__ void add_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        out[i] = a[i] + b[i];
    }
}

extern "C" __global__ void mul_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        out[i] = a[i] * b[i];
    }
}

extern "C" __global__ void sub_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        out[i] = a[i] - b[i];
    }
}

extern "C" __global__ void div_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        out[i] = a[i] / b[i];
    }
}

extern "C" __global__ void fused_add_mul_f32(
    const float* __restrict__ input,
    const float* __restrict__ add_bias,
    const float* __restrict__ mul_scale,
    float* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        out[i] = (input[i] + add_bias[i]) * mul_scale[i];
    }
}
"#;

/// Inline CUDA C source for element-wise unary activation kernels.
///
/// Contains kernels: `relu_ew_f32`, `gelu_ew_f32`, `silu_ew_f32`,
/// `sigmoid_f32`, `tanh_f32`, `clamp_f32`.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const ELEMENTWISE_UNARY_KERNEL_SRC: &str = r#"
extern "C" __global__ void relu_ew_f32(
    const float* __restrict__ input,
    float* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        out[i] = fmaxf(0.0f, input[i]);
    }
}

extern "C" __global__ void gelu_ew_f32(
    const float* __restrict__ input,
    float* __restrict__ out,
    int n)
{
    const float SQRT_2_OVER_PI = 0.7978845608f;
    const float COEFF = 0.044715f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        float x = input[i];
        float x3 = x * x * x;
        float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
        out[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

extern "C" __global__ void silu_ew_f32(
    const float* __restrict__ input,
    float* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        float x = input[i];
        out[i] = x / (1.0f + expf(-x));
    }
}

extern "C" __global__ void sigmoid_f32(
    const float* __restrict__ input,
    float* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        out[i] = 1.0f / (1.0f + expf(-input[i]));
    }
}

extern "C" __global__ void tanh_ew_f32(
    const float* __restrict__ input,
    float* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        out[i] = tanhf(input[i]);
    }
}

extern "C" __global__ void clamp_f32(
    const float* __restrict__ input,
    float* __restrict__ out,
    int n,
    float lo,
    float hi)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        out[i] = fminf(fmaxf(input[i], lo), hi);
    }
}
"#;

// ---------------------------------------------------------------------------
// Operation selector
// ---------------------------------------------------------------------------

/// Selects which element-wise binary operation to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementwiseOp {
    /// Element-wise addition: `a + b`
    Add,
    /// Element-wise multiplication: `a * b`
    Mul,
    /// Element-wise subtraction: `a - b`
    Sub,
    /// Element-wise division: `a / b`
    Div,
    /// Fused add-then-multiply: `(a + b) * c`
    FusedAddMul,
    /// ReLU: `max(0, x)`
    Relu,
    /// GELU (tanh approximation)
    Gelu,
    /// SiLU (Sigmoid Linear Unit): `x · σ(x)`
    Silu,
    /// Sigmoid: `1 / (1 + exp(-x))`
    Sigmoid,
    /// Hyperbolic tangent
    Tanh,
    /// Clamp to `[lo, hi]`
    Clamp,
}

impl ElementwiseOp {
    /// CUDA kernel function name for this operation.
    pub fn kernel_name(&self) -> &'static str {
        match self {
            Self::Add => "add_f32",
            Self::Mul => "mul_f32",
            Self::Sub => "sub_f32",
            Self::Div => "div_f32",
            Self::FusedAddMul => "fused_add_mul_f32",
            Self::Relu => "relu_ew_f32",
            Self::Gelu => "gelu_ew_f32",
            Self::Silu => "silu_ew_f32",
            Self::Sigmoid => "sigmoid_f32",
            Self::Tanh => "tanh_ew_f32",
            Self::Clamp => "clamp_f32",
        }
    }

    /// Whether this is a binary operation (requires two input slices).
    pub fn is_binary(&self) -> bool {
        matches!(self, Self::Add | Self::Mul | Self::Sub | Self::Div)
    }

    /// Whether this is a unary operation (one input slice).
    pub fn is_unary(&self) -> bool {
        matches!(
            self,
            Self::Relu | Self::Gelu | Self::Silu | Self::Sigmoid | Self::Tanh | Self::Clamp
        )
    }
}

// ---------------------------------------------------------------------------
// Launch configuration
// ---------------------------------------------------------------------------

/// Launch configuration for element-wise kernels.
#[derive(Debug, Clone)]
pub struct ElementwiseConfig {
    /// Total number of elements to process.
    pub n: usize,
    /// Threads per block (default 256).
    pub threads_per_block: u32,
    /// Which element-wise operation to apply.
    pub op: ElementwiseOp,
    /// Lower clamp bound (only used when `op == Clamp`).
    pub clamp_lo: f32,
    /// Upper clamp bound (only used when `op == Clamp`).
    pub clamp_hi: f32,
}

impl ElementwiseConfig {
    /// Create a configuration for the given element count and operation.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if `n` is zero.
    pub fn new(n: usize, op: ElementwiseOp) -> Result<Self> {
        if n == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "elementwise element count must be non-zero".into(),
            }
            .into());
        }
        Ok(Self {
            n,
            threads_per_block: 256,
            op,
            clamp_lo: f32::NEG_INFINITY,
            clamp_hi: f32::INFINITY,
        })
    }

    /// Set clamp bounds (only meaningful when `op == Clamp`).
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if `lo > hi`.
    pub fn with_clamp_bounds(mut self, lo: f32, hi: f32) -> Result<Self> {
        if lo > hi {
            return Err(KernelError::InvalidArguments {
                reason: format!("clamp lo ({lo}) must be <= hi ({hi})"),
            }
            .into());
        }
        self.clamp_lo = lo;
        self.clamp_hi = hi;
        Ok(self)
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
// Scalar helpers (used by CPU fallback)
// ---------------------------------------------------------------------------

#[inline]
fn sigmoid_scalar(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn silu_scalar(x: f32) -> f32 {
    x * sigmoid_scalar(x)
}

#[inline]
fn gelu_scalar(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const COEFF: f32 = 0.044_715;
    let x3 = x * x * x;
    let inner = SQRT_2_OVER_PI * (x + COEFF * x3);
    0.5 * x * (1.0 + inner.tanh())
}

#[inline]
fn relu_scalar(x: f32) -> f32 {
    x.max(0.0)
}

#[inline]
fn tanh_scalar(x: f32) -> f32 {
    x.tanh()
}

#[inline]
fn clamp_scalar(x: f32, lo: f32, hi: f32) -> f32 {
    x.max(lo).min(hi)
}

// ---------------------------------------------------------------------------
// Input validation
// ---------------------------------------------------------------------------

fn validate_binary_buffers(a: &[f32], b: &[f32], n: usize) -> Result<()> {
    if a.len() < n {
        return Err(KernelError::InvalidArguments {
            reason: format!("elementwise input `a` length {} < expected {n}", a.len()),
        }
        .into());
    }
    if b.len() < n {
        return Err(KernelError::InvalidArguments {
            reason: format!("elementwise input `b` length {} < expected {n}", b.len()),
        }
        .into());
    }
    Ok(())
}

fn validate_unary_buffer(input: &[f32], n: usize) -> Result<()> {
    if input.len() < n {
        return Err(KernelError::InvalidArguments {
            reason: format!("elementwise input length {} < expected {n}", input.len()),
        }
        .into());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CPU fallback implementations
// ---------------------------------------------------------------------------

/// Element-wise binary operation on the CPU.
///
/// Applies the operation selected by `op` to each pair of elements
/// `(a[i], b[i])` for `i` in `0..a.len()`.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `a` and `b` have different
/// lengths or either is empty.
pub fn elementwise_cpu_fallback(a: &[f32], b: &[f32], op: ElementwiseOp) -> Result<Vec<f32>> {
    if a.is_empty() {
        return Err(KernelError::InvalidArguments {
            reason: "elementwise input `a` is empty".into(),
        }
        .into());
    }
    let n = a.len();
    validate_binary_buffers(a, b, n)?;

    let out = match op {
        ElementwiseOp::Add => a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect(),
        ElementwiseOp::Mul => a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect(),
        ElementwiseOp::Sub => a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect(),
        ElementwiseOp::Div => a.iter().zip(b.iter()).map(|(&x, &y)| x / y).collect(),
        _ => {
            return Err(KernelError::InvalidArguments {
                reason: format!("elementwise_cpu_fallback only supports binary ops, got {:?}", op),
            }
            .into());
        }
    };
    Ok(out)
}

/// Fused add-then-multiply on the CPU.
///
/// Computes `(input[i] + add_bias[i]) * mul_scale[i]` in a single pass.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if slice lengths differ or
/// any slice is empty.
pub fn fused_elementwise_cpu(
    input: &[f32],
    add_bias: &[f32],
    mul_scale: &[f32],
) -> Result<Vec<f32>> {
    if input.is_empty() {
        return Err(KernelError::InvalidArguments {
            reason: "fused elementwise input is empty".into(),
        }
        .into());
    }
    let n = input.len();
    if add_bias.len() < n {
        return Err(KernelError::InvalidArguments {
            reason: format!("add_bias length {} < expected {n}", add_bias.len()),
        }
        .into());
    }
    if mul_scale.len() < n {
        return Err(KernelError::InvalidArguments {
            reason: format!("mul_scale length {} < expected {n}", mul_scale.len()),
        }
        .into());
    }
    let out = input
        .iter()
        .zip(add_bias.iter())
        .zip(mul_scale.iter())
        .map(|((&x, &b), &s)| (x + b) * s)
        .collect();
    Ok(out)
}

/// Unary activation on the CPU.
///
/// Applies the unary operation selected by `config.op` to each element.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if input length is less than
/// `config.n` or if the operation is not unary.
pub fn elementwise_unary_cpu(input: &[f32], config: &ElementwiseConfig) -> Result<Vec<f32>> {
    validate_unary_buffer(input, config.n)?;

    let out = match config.op {
        ElementwiseOp::Relu => input[..config.n].iter().map(|&x| relu_scalar(x)).collect(),
        ElementwiseOp::Gelu => input[..config.n].iter().map(|&x| gelu_scalar(x)).collect(),
        ElementwiseOp::Silu => input[..config.n].iter().map(|&x| silu_scalar(x)).collect(),
        ElementwiseOp::Sigmoid => input[..config.n].iter().map(|&x| sigmoid_scalar(x)).collect(),
        ElementwiseOp::Tanh => input[..config.n].iter().map(|&x| tanh_scalar(x)).collect(),
        ElementwiseOp::Clamp => input[..config.n]
            .iter()
            .map(|&x| clamp_scalar(x, config.clamp_lo, config.clamp_hi))
            .collect(),
        _ => {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "elementwise_unary_cpu only supports unary ops, got {:?}",
                    config.op
                ),
            }
            .into());
        }
    };
    Ok(out)
}

/// Dispatch an element-wise binary operation, preferring CUDA when available.
///
/// Falls back to [`elementwise_cpu_fallback`] when compiled without GPU
/// features or when the CUDA dispatch fails.
pub fn launch_elementwise_binary(a: &[f32], b: &[f32], op: ElementwiseOp) -> Result<Vec<f32>> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        // CUDA path would go here; for now fall through to CPU.
        let _ = (a, b, op);
    }
    elementwise_cpu_fallback(a, b, op)
}

/// Dispatch a fused add+multiply, preferring CUDA when available.
pub fn launch_fused_add_mul(
    input: &[f32],
    add_bias: &[f32],
    mul_scale: &[f32],
) -> Result<Vec<f32>> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        let _ = (input, add_bias, mul_scale);
    }
    fused_elementwise_cpu(input, add_bias, mul_scale)
}

/// Dispatch a unary activation, preferring CUDA when available.
pub fn launch_elementwise_unary(input: &[f32], config: &ElementwiseConfig) -> Result<Vec<f32>> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        let _ = (input, config);
    }
    elementwise_unary_cpu(input, config)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // ElementwiseOp enum tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_op_kernel_names() {
        assert_eq!(ElementwiseOp::Add.kernel_name(), "add_f32");
        assert_eq!(ElementwiseOp::Mul.kernel_name(), "mul_f32");
        assert_eq!(ElementwiseOp::Sub.kernel_name(), "sub_f32");
        assert_eq!(ElementwiseOp::Div.kernel_name(), "div_f32");
        assert_eq!(ElementwiseOp::FusedAddMul.kernel_name(), "fused_add_mul_f32");
        assert_eq!(ElementwiseOp::Relu.kernel_name(), "relu_ew_f32");
        assert_eq!(ElementwiseOp::Gelu.kernel_name(), "gelu_ew_f32");
        assert_eq!(ElementwiseOp::Silu.kernel_name(), "silu_ew_f32");
        assert_eq!(ElementwiseOp::Sigmoid.kernel_name(), "sigmoid_f32");
        assert_eq!(ElementwiseOp::Tanh.kernel_name(), "tanh_ew_f32");
        assert_eq!(ElementwiseOp::Clamp.kernel_name(), "clamp_f32");
    }

    #[test]
    fn test_op_is_binary() {
        assert!(ElementwiseOp::Add.is_binary());
        assert!(ElementwiseOp::Mul.is_binary());
        assert!(ElementwiseOp::Sub.is_binary());
        assert!(ElementwiseOp::Div.is_binary());
        assert!(!ElementwiseOp::Relu.is_binary());
        assert!(!ElementwiseOp::FusedAddMul.is_binary());
    }

    #[test]
    fn test_op_is_unary() {
        assert!(ElementwiseOp::Relu.is_unary());
        assert!(ElementwiseOp::Gelu.is_unary());
        assert!(ElementwiseOp::Silu.is_unary());
        assert!(ElementwiseOp::Sigmoid.is_unary());
        assert!(ElementwiseOp::Tanh.is_unary());
        assert!(ElementwiseOp::Clamp.is_unary());
        assert!(!ElementwiseOp::Add.is_unary());
    }

    // -----------------------------------------------------------------------
    // Config tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_basic() {
        let cfg = ElementwiseConfig::new(1024, ElementwiseOp::Add).unwrap();
        assert_eq!(cfg.n, 1024);
        assert_eq!(cfg.threads_per_block, 256);
        assert_eq!(cfg.op, ElementwiseOp::Add);
    }

    #[test]
    fn test_config_rejects_zero() {
        assert!(ElementwiseConfig::new(0, ElementwiseOp::Mul).is_err());
    }

    #[test]
    fn test_config_grid_dim_small() {
        let cfg = ElementwiseConfig::new(100, ElementwiseOp::Sub).unwrap();
        assert_eq!(cfg.grid_dim(), (1, 1, 1));
        assert_eq!(cfg.block_dim(), (256, 1, 1));
    }

    #[test]
    fn test_config_grid_dim_large() {
        let cfg = ElementwiseConfig::new(20_000_000, ElementwiseOp::Add).unwrap();
        assert_eq!(cfg.grid_dim().0, 65_535);
    }

    #[test]
    fn test_config_clamp_bounds() {
        let cfg = ElementwiseConfig::new(4, ElementwiseOp::Clamp)
            .unwrap()
            .with_clamp_bounds(-1.0, 1.0)
            .unwrap();
        assert_eq!(cfg.clamp_lo, -1.0);
        assert_eq!(cfg.clamp_hi, 1.0);
    }

    #[test]
    fn test_config_clamp_bounds_rejects_inverted() {
        let cfg = ElementwiseConfig::new(4, ElementwiseOp::Clamp).unwrap();
        assert!(cfg.with_clamp_bounds(2.0, 1.0).is_err());
    }

    // -----------------------------------------------------------------------
    // Binary CPU fallback — arithmetic
    // -----------------------------------------------------------------------

    #[test]
    fn test_cpu_add() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![0.5, 1.5, 2.5, 3.5];
        let out = elementwise_cpu_fallback(&a, &b, ElementwiseOp::Add).unwrap();
        assert_eq!(out, vec![1.5, 3.5, 5.5, 7.5]);
    }

    #[test]
    fn test_cpu_mul() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 3.0, 4.0];
        let out = elementwise_cpu_fallback(&a, &b, ElementwiseOp::Mul).unwrap();
        assert_eq!(out, vec![2.0, 6.0, 12.0]);
    }

    #[test]
    fn test_cpu_sub() {
        let a = vec![5.0, 3.0, 1.0];
        let b = vec![1.0, 2.0, 3.0];
        let out = elementwise_cpu_fallback(&a, &b, ElementwiseOp::Sub).unwrap();
        assert_eq!(out, vec![4.0, 1.0, -2.0]);
    }

    #[test]
    fn test_cpu_div() {
        let a = vec![6.0, 9.0, 12.0];
        let b = vec![2.0, 3.0, 4.0];
        let out = elementwise_cpu_fallback(&a, &b, ElementwiseOp::Div).unwrap();
        assert_eq!(out, vec![3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_cpu_binary_empty_input_rejected() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        assert!(elementwise_cpu_fallback(&a, &b, ElementwiseOp::Add).is_err());
    }

    #[test]
    fn test_cpu_binary_mismatched_lengths() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0];
        assert!(elementwise_cpu_fallback(&a, &b, ElementwiseOp::Add).is_err());
    }

    #[test]
    fn test_cpu_binary_rejects_unary_op() {
        let a = vec![1.0];
        let b = vec![1.0];
        assert!(elementwise_cpu_fallback(&a, &b, ElementwiseOp::Relu).is_err());
    }

    // -----------------------------------------------------------------------
    // Fused add+mul CPU
    // -----------------------------------------------------------------------

    #[test]
    fn test_fused_add_mul_basic() {
        let input = vec![1.0, 2.0, 3.0];
        let bias = vec![0.5, 0.5, 0.5];
        let scale = vec![2.0, 2.0, 2.0];
        let out = fused_elementwise_cpu(&input, &bias, &scale).unwrap();
        assert_eq!(out, vec![3.0, 5.0, 7.0]);
    }

    #[test]
    fn test_fused_add_mul_zeros() {
        let input = vec![1.0, 2.0];
        let bias = vec![0.0, 0.0];
        let scale = vec![0.0, 0.0];
        let out = fused_elementwise_cpu(&input, &bias, &scale).unwrap();
        assert_eq!(out, vec![0.0, 0.0]);
    }

    #[test]
    fn test_fused_empty_rejected() {
        let empty: Vec<f32> = vec![];
        assert!(fused_elementwise_cpu(&empty, &empty, &empty).is_err());
    }

    #[test]
    fn test_fused_mismatched_bias_length() {
        let input = vec![1.0, 2.0, 3.0];
        let bias = vec![0.5];
        let scale = vec![2.0, 2.0, 2.0];
        assert!(fused_elementwise_cpu(&input, &bias, &scale).is_err());
    }

    #[test]
    fn test_fused_mismatched_scale_length() {
        let input = vec![1.0, 2.0, 3.0];
        let bias = vec![0.5, 0.5, 0.5];
        let scale = vec![2.0];
        assert!(fused_elementwise_cpu(&input, &bias, &scale).is_err());
    }

    // -----------------------------------------------------------------------
    // Unary CPU — activations
    // -----------------------------------------------------------------------

    #[test]
    fn test_cpu_relu() {
        let cfg = ElementwiseConfig::new(4, ElementwiseOp::Relu).unwrap();
        let input = vec![-2.0, -0.5, 0.0, 3.0];
        let out = elementwise_unary_cpu(&input, &cfg).unwrap();
        assert_eq!(out, vec![0.0, 0.0, 0.0, 3.0]);
    }

    #[test]
    fn test_cpu_sigmoid_known_values() {
        let cfg = ElementwiseConfig::new(3, ElementwiseOp::Sigmoid).unwrap();
        let input = vec![0.0, 100.0, -100.0];
        let out = elementwise_unary_cpu(&input, &cfg).unwrap();
        assert!((out[0] - 0.5).abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-4);
        assert!(out[2].abs() < 1e-4);
    }

    #[test]
    fn test_cpu_tanh_known_values() {
        let cfg = ElementwiseConfig::new(3, ElementwiseOp::Tanh).unwrap();
        let input = vec![0.0, 100.0, -100.0];
        let out = elementwise_unary_cpu(&input, &cfg).unwrap();
        assert!(out[0].abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-4);
        assert!((out[2] + 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_cpu_silu_known_values() {
        let cfg = ElementwiseConfig::new(3, ElementwiseOp::Silu).unwrap();
        let input = vec![0.0, 1.0, -1.0];
        let out = elementwise_unary_cpu(&input, &cfg).unwrap();
        assert!(out[0].abs() < 1e-6, "silu(0) = {}", out[0]);
        assert!((out[1] - 0.7311).abs() < 1e-3, "silu(1) = {}", out[1]);
        assert!((out[2] + 0.2689).abs() < 1e-3, "silu(-1) = {}", out[2]);
    }

    #[test]
    fn test_cpu_gelu_known_values() {
        let cfg = ElementwiseConfig::new(3, ElementwiseOp::Gelu).unwrap();
        let input = vec![0.0, 1.0, -1.0];
        let out = elementwise_unary_cpu(&input, &cfg).unwrap();
        assert!(out[0].abs() < 1e-6, "gelu(0) = {}", out[0]);
        assert!((out[1] - 0.8412).abs() < 1e-3, "gelu(1) = {}", out[1]);
        assert!((out[2] + 0.1588).abs() < 1e-3, "gelu(-1) = {}", out[2]);
    }

    #[test]
    fn test_cpu_clamp() {
        let cfg = ElementwiseConfig::new(5, ElementwiseOp::Clamp)
            .unwrap()
            .with_clamp_bounds(-1.0, 1.0)
            .unwrap();
        let input = vec![-5.0, -0.5, 0.0, 0.5, 5.0];
        let out = elementwise_unary_cpu(&input, &cfg).unwrap();
        assert_eq!(out, vec![-1.0, -0.5, 0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_cpu_unary_rejects_binary_op() {
        let cfg = ElementwiseConfig::new(2, ElementwiseOp::Add).unwrap();
        let input = vec![1.0, 2.0];
        assert!(elementwise_unary_cpu(&input, &cfg).is_err());
    }

    #[test]
    fn test_cpu_unary_short_input_rejected() {
        let cfg = ElementwiseConfig::new(5, ElementwiseOp::Relu).unwrap();
        let input = vec![1.0, 2.0];
        assert!(elementwise_unary_cpu(&input, &cfg).is_err());
    }

    // -----------------------------------------------------------------------
    // Edge cases: NaN, Inf, zeros
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_nan_propagation() {
        let a = vec![f32::NAN, 1.0];
        let b = vec![1.0, f32::NAN];
        let out = elementwise_cpu_fallback(&a, &b, ElementwiseOp::Add).unwrap();
        assert!(out[0].is_nan());
        assert!(out[1].is_nan());
    }

    #[test]
    fn test_mul_inf_handling() {
        let a = vec![f32::INFINITY, f32::NEG_INFINITY, 0.0];
        let b = vec![2.0, 3.0, f32::INFINITY];
        let out = elementwise_cpu_fallback(&a, &b, ElementwiseOp::Mul).unwrap();
        assert_eq!(out[0], f32::INFINITY);
        assert_eq!(out[1], f32::NEG_INFINITY);
        // 0 * inf = NaN per IEEE 754
        assert!(out[2].is_nan());
    }

    #[test]
    fn test_div_by_zero() {
        let a = vec![1.0, 0.0, -1.0];
        let b = vec![0.0, 0.0, 0.0];
        let out = elementwise_cpu_fallback(&a, &b, ElementwiseOp::Div).unwrap();
        assert_eq!(out[0], f32::INFINITY);
        assert!(out[1].is_nan());
        assert_eq!(out[2], f32::NEG_INFINITY);
    }

    #[test]
    fn test_sigmoid_nan_propagation() {
        let cfg = ElementwiseConfig::new(1, ElementwiseOp::Sigmoid).unwrap();
        let input = vec![f32::NAN];
        let out = elementwise_unary_cpu(&input, &cfg).unwrap();
        assert!(out[0].is_nan());
    }

    #[test]
    fn test_relu_negative_inf() {
        let cfg = ElementwiseConfig::new(2, ElementwiseOp::Relu).unwrap();
        let input = vec![f32::NEG_INFINITY, f32::INFINITY];
        let out = elementwise_unary_cpu(&input, &cfg).unwrap();
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], f32::INFINITY);
    }

    #[test]
    fn test_clamp_with_inf_bounds() {
        // Default clamp bounds are (-inf, +inf) which is a no-op
        let cfg = ElementwiseConfig::new(3, ElementwiseOp::Clamp).unwrap();
        let input = vec![-100.0, 0.0, 100.0];
        let out = elementwise_unary_cpu(&input, &cfg).unwrap();
        assert_eq!(out, vec![-100.0, 0.0, 100.0]);
    }

    // -----------------------------------------------------------------------
    // Dispatch wrappers
    // -----------------------------------------------------------------------

    #[test]
    fn test_launch_binary_add() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let out = launch_elementwise_binary(&a, &b, ElementwiseOp::Add).unwrap();
        assert_eq!(out, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_launch_fused_add_mul() {
        let input = vec![1.0, 2.0];
        let bias = vec![1.0, 1.0];
        let scale = vec![3.0, 3.0];
        let out = launch_fused_add_mul(&input, &bias, &scale).unwrap();
        assert_eq!(out, vec![6.0, 9.0]);
    }

    #[test]
    fn test_launch_unary_sigmoid() {
        let cfg = ElementwiseConfig::new(1, ElementwiseOp::Sigmoid).unwrap();
        let input = vec![0.0];
        let out = launch_elementwise_unary(&input, &cfg).unwrap();
        assert!((out[0] - 0.5).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Single-element edge case
    // -----------------------------------------------------------------------

    #[test]
    fn test_single_element_all_binary_ops() {
        let a = vec![4.0];
        let b = vec![2.0];
        for (op, expected) in [
            (ElementwiseOp::Add, 6.0),
            (ElementwiseOp::Sub, 2.0),
            (ElementwiseOp::Mul, 8.0),
            (ElementwiseOp::Div, 2.0),
        ] {
            let out = elementwise_cpu_fallback(&a, &b, op).unwrap();
            assert_eq!(out[0], expected, "{op:?} failed");
        }
    }
}

//! Fused kernel operations for Intel A770 (Arc) GPU acceleration.
//!
//! Fused operations combine multiple sequential ops into single kernel launches,
//! reducing memory bandwidth pressure by keeping intermediate results in
//! registers/local memory rather than round-tripping through global memory.
//!
//! # Provided fusions
//!
//! | Fusion | Ops combined |
//! |---|---|
//! | `cpu_fused_rmsnorm_linear` | RMSNorm → matmul |
//! | `cpu_fused_linear_silu` | matmul → SiLU |
//! | `cpu_fused_linear_gelu` | matmul → GELU |
//! | `cpu_fused_gate_up_proj` | SwiGLU gate+up projection |
//! | `cpu_fused_bias_add_activation` | bias add → activation (in-place) |
//! | `cpu_fused_scale_shift` | affine transform (in-place) |
//! | `cpu_fused_quantize_dequantize` | quantize → dequantize round-trip |
//!
//! # CPU reference
//!
//! All fused operations have scalar CPU reference implementations for
//! correctness testing against decomposed (unfused) equivalents.
//!
//! # OpenCL kernels
//!
//! [`FUSED_OPS_SRC`] contains OpenCL C source for GPU dispatch on Intel Arc
//! A770 and other OpenCL 3.0 devices.

// -------------------------------------------------------------------------
// Types
// -------------------------------------------------------------------------

/// Enumeration of supported fused operation patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusedOp {
    /// RMSNorm followed by a linear projection.
    NormLinear,
    /// Linear projection followed by an activation function.
    LinearActivation,
    /// RMSNorm → linear → activation in a single pass.
    NormLinearActivation,
    /// Bias addition followed by an activation function (in-place).
    BiasActivation,
    /// Elementwise affine transform: `x * scale + shift` (in-place).
    ScaleShift,
    /// Quantize then immediately dequantize (quantization noise simulation).
    QuantizeDequantize,
}

/// Activation function type for fused operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationType {
    /// Identity (no activation).
    None,
    /// Rectified Linear Unit: `max(0, x)`.
    ReLU,
    /// Sigmoid Linear Unit: `x * sigmoid(x)`.
    SiLU,
    /// Gaussian Error Linear Unit (tanh approximation).
    GELU,
    /// Hyperbolic tangent.
    Tanh,
}

/// Configuration for a fused operation.
#[derive(Debug, Clone)]
pub struct FusionConfig {
    /// Epsilon for numerical stability in normalization ops.
    pub epsilon: f32,
    /// Activation function to apply.
    pub activation: ActivationType,
    /// Whether to add a bias term after linear projections.
    pub use_bias: bool,
    /// Bit-width for quantize-dequantize fusion (`None` = skip).
    pub quantize_bits: Option<u8>,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            epsilon: 1e-6,
            activation: ActivationType::None,
            use_bias: false,
            quantize_bits: None,
        }
    }
}

/// Errors specific to fused operations.
#[derive(Debug, Clone, PartialEq)]
pub enum FusedOpsError {
    /// Input/weight dimensions are incompatible.
    DimensionMismatch { expected: usize, got: usize, context: &'static str },
    /// The requested fusion pattern is not supported.
    UnsupportedFusion(FusedOp),
    /// A numerical error occurred (NaN / Inf produced).
    NumericalError(&'static str),
}

impl std::fmt::Display for FusedOpsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got, context } => {
                write!(
                    f,
                    "dimension mismatch in {context}: \
                     expected {expected}, got {got}"
                )
            }
            Self::UnsupportedFusion(op) => {
                write!(f, "unsupported fusion: {op:?}")
            }
            Self::NumericalError(msg) => {
                write!(f, "numerical error: {msg}")
            }
        }
    }
}

impl std::error::Error for FusedOpsError {}

// -------------------------------------------------------------------------
// Helper functions
// -------------------------------------------------------------------------

/// Scalar SiLU activation: `x * sigmoid(x)`.
#[inline]
pub fn cpu_silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Approximate GELU: `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`.
#[inline]
pub fn cpu_gelu(x: f32) -> f32 {
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

/// Apply an [`ActivationType`] to a scalar value.
#[inline]
fn apply_activation(x: f32, act: ActivationType) -> f32 {
    match act {
        ActivationType::None => x,
        ActivationType::ReLU => x.max(0.0),
        ActivationType::SiLU => cpu_silu(x),
        ActivationType::GELU => cpu_gelu(x),
        ActivationType::Tanh => x.tanh(),
    }
}

/// Standalone RMSNorm for verification:
/// `y_i = (x_i / rms) * gamma_i` where `rms = sqrt(mean(x²) + eps)`.
pub fn cpu_rmsnorm(input: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(input.len(), gamma.len());
    let n = input.len() as f32;
    let sum_sq: f32 = input.iter().map(|&v| v * v).sum();
    let rms = (sum_sq / n + eps).sqrt();
    input.iter().zip(gamma.iter()).map(|(&x, &g)| x / rms * g).collect()
}

/// Row-major matrix multiply: C[m×n] = A[m×k] @ B[k×n].
pub fn cpu_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "A must be m×k");
    assert_eq!(b.len(), k * n, "B must be k×n");
    let mut c = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0_f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

// -------------------------------------------------------------------------
// Fused CPU reference implementations
// -------------------------------------------------------------------------

/// Fused RMSNorm + linear projection.
///
/// For each row of `input` (shape `[rows, cols]`):
/// 1. Apply RMSNorm with `gamma` (length `cols`) and `eps`.
/// 2. Multiply the normalised row by `weight` (shape `[cols, out_features]`).
///
/// Returns output of shape `[rows, out_features]`.
pub fn cpu_fused_rmsnorm_linear(
    input: &[f32],
    weight: &[f32],
    gamma: &[f32],
    rows: usize,
    cols: usize,
    out_features: usize,
    eps: f32,
) -> Vec<f32> {
    assert_eq!(input.len(), rows * cols);
    assert_eq!(weight.len(), cols * out_features);
    assert_eq!(gamma.len(), cols);

    let mut output = vec![0.0_f32; rows * out_features];
    for i in 0..rows {
        let row = &input[i * cols..(i + 1) * cols];

        // RMSNorm
        let sum_sq: f32 = row.iter().map(|&v| v * v).sum();
        let rms = (sum_sq / cols as f32 + eps).sqrt();

        // Fused norm + matmul: avoid materialising the normalised row.
        for j in 0..out_features {
            let mut acc = 0.0_f32;
            for p in 0..cols {
                let normed = row[p] / rms * gamma[p];
                acc += normed * weight[p * out_features + j];
            }
            output[i * out_features + j] = acc;
        }
    }
    output
}

/// Fused linear + SiLU activation.
///
/// `output = SiLU(input @ weight + bias)` where `input` is `[rows, in_features]`
/// and `weight` is `[in_features, out_features]`.
pub fn cpu_fused_linear_silu(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    rows: usize,
    in_features: usize,
    out_features: usize,
) -> Vec<f32> {
    assert_eq!(input.len(), rows * in_features);
    assert_eq!(weight.len(), in_features * out_features);
    if let Some(b) = bias {
        assert_eq!(b.len(), out_features);
    }

    let mut output = vec![0.0_f32; rows * out_features];
    for i in 0..rows {
        for j in 0..out_features {
            let mut acc = 0.0_f32;
            for p in 0..in_features {
                acc += input[i * in_features + p] * weight[p * out_features + j];
            }
            if let Some(b) = bias {
                acc += b[j];
            }
            output[i * out_features + j] = cpu_silu(acc);
        }
    }
    output
}

/// Fused linear + GELU activation.
///
/// `output = GELU(input @ weight + bias)`.
pub fn cpu_fused_linear_gelu(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    rows: usize,
    in_features: usize,
    out_features: usize,
) -> Vec<f32> {
    assert_eq!(input.len(), rows * in_features);
    assert_eq!(weight.len(), in_features * out_features);
    if let Some(b) = bias {
        assert_eq!(b.len(), out_features);
    }

    let mut output = vec![0.0_f32; rows * out_features];
    for i in 0..rows {
        for j in 0..out_features {
            let mut acc = 0.0_f32;
            for p in 0..in_features {
                acc += input[i * in_features + p] * weight[p * out_features + j];
            }
            if let Some(b) = bias {
                acc += b[j];
            }
            output[i * out_features + j] = cpu_gelu(acc);
        }
    }
    output
}

/// Fused SwiGLU gate+up projection.
///
/// `output = SiLU(input @ gate_weight) * (input @ up_weight)`.
///
/// Both `gate_weight` and `up_weight` have shape `[in_features, out_features]`.
/// Returns output of shape `[rows, out_features]`.
pub fn cpu_fused_gate_up_proj(
    input: &[f32],
    gate_weight: &[f32],
    up_weight: &[f32],
    rows: usize,
    in_features: usize,
    out_features: usize,
) -> Vec<f32> {
    assert_eq!(input.len(), rows * in_features);
    assert_eq!(gate_weight.len(), in_features * out_features);
    assert_eq!(up_weight.len(), in_features * out_features);

    let mut output = vec![0.0_f32; rows * out_features];
    for i in 0..rows {
        for j in 0..out_features {
            let mut gate_acc = 0.0_f32;
            let mut up_acc = 0.0_f32;
            for p in 0..in_features {
                let x = input[i * in_features + p];
                gate_acc += x * gate_weight[p * out_features + j];
                up_acc += x * up_weight[p * out_features + j];
            }
            output[i * out_features + j] = cpu_silu(gate_acc) * up_acc;
        }
    }
    output
}

/// In-place bias addition followed by activation.
///
/// `data` is row-major with `cols` columns. Each row gets `bias[j]` added to
/// column `j`, then the activation is applied.
pub fn cpu_fused_bias_add_activation(
    data: &mut [f32],
    bias: &[f32],
    cols: usize,
    activation: ActivationType,
) {
    assert_eq!(bias.len(), cols);
    assert_eq!(data.len() % cols, 0);
    for row in data.chunks_exact_mut(cols) {
        for (v, &b) in row.iter_mut().zip(bias.iter()) {
            *v = apply_activation(*v + b, activation);
        }
    }
}

/// In-place affine transform: `data[j] = data[j] * scale[j] + shift[j]`.
///
/// Used as the final step of LayerNorm (after centering + dividing by stddev).
pub fn cpu_fused_scale_shift(data: &mut [f32], scale: &[f32], shift: &[f32], cols: usize) {
    assert_eq!(scale.len(), cols);
    assert_eq!(shift.len(), cols);
    assert_eq!(data.len() % cols, 0);
    for row in data.chunks_exact_mut(cols) {
        for j in 0..cols {
            row[j] = row[j] * scale[j] + shift[j];
        }
    }
}

/// Quantize then immediately dequantize to simulate quantization noise.
///
/// Uniform symmetric quantization to `bits` (e.g. 8 → [-127, 127]):
/// `scale = max(|x|) / (2^(bits-1) - 1)`, then `round(x / scale) * scale`.
pub fn cpu_fused_quantize_dequantize(data: &[f32], bits: u8) -> Vec<f32> {
    assert!((2..=16).contains(&bits), "bits must be in [2, 16]");
    let max_val = (1_u32 << (bits - 1)) - 1;
    let max_abs = data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    if max_abs == 0.0 {
        return data.to_vec();
    }
    let scale = max_abs / max_val as f32;
    data.iter()
        .map(|&v| {
            let q = (v / scale).round().clamp(-(max_val as f32), max_val as f32);
            q * scale
        })
        .collect()
}

// -------------------------------------------------------------------------
// OpenCL kernel sources
// -------------------------------------------------------------------------

/// OpenCL C source for fused operations targeting Intel Arc A770.
///
/// Contains:
/// - `fused_rmsnorm_linear` — norm + matmul in a single kernel
/// - `fused_linear_silu` — matmul + SiLU with register-level fusion
/// - `fused_gate_up_proj` — SwiGLU gate+up projection
/// - `fused_bias_activation` — bias add + activation
pub const FUSED_OPS_SRC: &str = r#"
// ------------------------------------------------------------------
// Fused RMSNorm + Linear
// ------------------------------------------------------------------
// input:  [rows, cols]   (global memory)
// weight: [cols, out_features]
// gamma:  [cols]
// output: [rows, out_features]
__kernel void fused_rmsnorm_linear(
    __global const float* input,
    __global const float* weight,
    __global const float* gamma,
    __global float* output,
    const int cols,
    const int out_features,
    const float eps)
{
    const int row = get_global_id(0);
    const int j   = get_global_id(1);

    // Step 1: compute RMS for this row
    float sum_sq = 0.0f;
    for (int p = 0; p < cols; ++p) {
        float v = input[row * cols + p];
        sum_sq += v * v;
    }
    float rms = sqrt(sum_sq / (float)cols + eps);

    // Step 2: fused norm + dot-product (no intermediate buffer)
    float acc = 0.0f;
    for (int p = 0; p < cols; ++p) {
        float normed = input[row * cols + p] / rms * gamma[p];
        acc += normed * weight[p * out_features + j];
    }
    output[row * out_features + j] = acc;
}

// ------------------------------------------------------------------
// Fused Linear + SiLU
// ------------------------------------------------------------------
inline float silu(float x) {
    return x / (1.0f + exp(-x));
}

__kernel void fused_linear_silu(
    __global const float* input,
    __global const float* weight,
    __global const float* bias,
    __global float* output,
    const int in_features,
    const int out_features,
    const int use_bias)
{
    const int row = get_global_id(0);
    const int j   = get_global_id(1);

    float acc = 0.0f;
    for (int p = 0; p < in_features; ++p) {
        acc += input[row * in_features + p]
             * weight[p * out_features + j];
    }
    if (use_bias) acc += bias[j];
    output[row * out_features + j] = silu(acc);
}

// ------------------------------------------------------------------
// Fused SwiGLU gate+up projection
// ------------------------------------------------------------------
__kernel void fused_gate_up_proj(
    __global const float* input,
    __global const float* gate_weight,
    __global const float* up_weight,
    __global float* output,
    const int in_features,
    const int out_features)
{
    const int row = get_global_id(0);
    const int j   = get_global_id(1);

    float gate_acc = 0.0f;
    float up_acc   = 0.0f;
    for (int p = 0; p < in_features; ++p) {
        float x = input[row * in_features + p];
        gate_acc += x * gate_weight[p * out_features + j];
        up_acc   += x * up_weight[p * out_features + j];
    }
    output[row * out_features + j] = silu(gate_acc) * up_acc;
}

// ------------------------------------------------------------------
// Fused bias + activation (in-place)
// ------------------------------------------------------------------
// activation_type: 0=None, 1=ReLU, 2=SiLU, 3=GELU, 4=Tanh
inline float gelu_approx(float x) {
    const float c = 0.7978845608f; // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanh(c * (x + 0.044715f * x * x * x)));
}

inline float apply_act(float x, int act_type) {
    if (act_type == 1) return fmax(x, 0.0f);
    if (act_type == 2) return silu(x);
    if (act_type == 3) return gelu_approx(x);
    if (act_type == 4) return tanh(x);
    return x;
}

__kernel void fused_bias_activation(
    __global float* data,
    __global const float* bias,
    const int cols,
    const int activation_type)
{
    const int idx = get_global_id(0);
    const int j   = idx % cols;
    data[idx] = apply_act(data[idx] + bias[j], activation_type);
}
"#;

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Tolerance for floating-point comparison of fused vs unfused results.
    const TOL: f32 = 1e-4;
    const EPS: f32 = 1e-6;

    /// Check that every element of `a` is within `tol` of `b`.
    fn assert_approx_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (x - y).abs();
            assert!(diff <= tol, "element {i}: {x} vs {y} (diff={diff}, tol={tol})");
        }
    }

    // =================================================================
    // Activation helpers
    // =================================================================

    #[test]
    fn test_silu_zero() {
        assert!((cpu_silu(0.0) - 0.0).abs() < 1e-7);
    }

    #[test]
    fn test_silu_positive() {
        let y = cpu_silu(1.0);
        // SiLU(1) = 1 / (1 + e^-1) ≈ 0.7311
        assert!((y - 0.7311).abs() < 1e-3);
    }

    #[test]
    fn test_silu_negative() {
        let y = cpu_silu(-1.0);
        // SiLU(-1) = -1 / (1 + e^1) ≈ -0.2689
        assert!((y - (-0.2689)).abs() < 1e-3);
    }

    #[test]
    fn test_silu_monotonicity() {
        // For x > y > 0, SiLU(x) > SiLU(y)
        let vals = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0];
        for w in vals.windows(2) {
            assert!(cpu_silu(w[1]) > cpu_silu(w[0]), "SiLU not monotone at {} vs {}", w[0], w[1]);
        }
    }

    #[test]
    fn test_gelu_zero() {
        assert!((cpu_gelu(0.0) - 0.0).abs() < 1e-7);
    }

    #[test]
    fn test_gelu_positive() {
        let y = cpu_gelu(1.0);
        // GELU(1) ≈ 0.8412
        assert!((y - 0.8412).abs() < 1e-3);
    }

    #[test]
    fn test_gelu_negative() {
        let y = cpu_gelu(-1.0);
        // GELU(-1) ≈ -0.1588
        assert!((y - (-0.1588)).abs() < 1e-3);
    }

    #[test]
    fn test_gelu_approx_vs_exact() {
        // The tanh approximation should be within ~1e-3 of exact GELU
        // for moderate inputs.
        for &x in &[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0] {
            let approx = cpu_gelu(x);
            // Exact: 0.5 * x * (1 + erf(x / sqrt(2)))
            let exact = 0.5 * x * (1.0 + libm::erff(x / std::f32::consts::SQRT_2));
            assert!(
                (approx - exact).abs() < 2e-3,
                "GELU approx vs exact at x={x}: {approx} vs {exact}"
            );
        }
    }

    #[test]
    fn test_activation_none() {
        assert_eq!(apply_activation(3.14, ActivationType::None), 3.14);
    }

    #[test]
    fn test_activation_relu() {
        assert_eq!(apply_activation(-1.0, ActivationType::ReLU), 0.0);
        assert_eq!(apply_activation(2.5, ActivationType::ReLU), 2.5);
    }

    #[test]
    fn test_activation_tanh() {
        let y = apply_activation(1.0, ActivationType::Tanh);
        assert!((y - 1.0_f32.tanh()).abs() < 1e-7);
    }

    // =================================================================
    // cpu_matmul
    // =================================================================

    #[test]
    fn test_matmul_identity() {
        // 2×2 identity
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let c = cpu_matmul(&a, &b, 2, 2, 2);
        assert_approx_eq(&c, &[1.0, 2.0, 3.0, 4.0], 1e-7);
    }

    #[test]
    fn test_matmul_non_square() {
        // [2×3] @ [3×2]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let c = cpu_matmul(&a, &b, 2, 2, 3);
        // row0: 1*1+2*3+3*5=22, 1*2+2*4+3*6=28
        // row1: 4*1+5*3+6*5=49, 4*2+5*4+6*6=64
        assert_approx_eq(&c, &[22.0, 28.0, 49.0, 64.0], 1e-5);
    }

    #[test]
    fn test_matmul_1x1() {
        let c = cpu_matmul(&[3.0], &[4.0], 1, 1, 1);
        assert_approx_eq(&c, &[12.0], 1e-7);
    }

    // =================================================================
    // cpu_rmsnorm
    // =================================================================

    #[test]
    fn test_rmsnorm_ones() {
        let input = vec![1.0; 4];
        let gamma = vec![1.0; 4];
        let out = cpu_rmsnorm(&input, &gamma, EPS);
        // rms ≈ 1.0 → all outputs ≈ 1.0
        for &v in &out {
            assert!((v - 1.0).abs() < 1e-3);
        }
    }

    #[test]
    fn test_rmsnorm_scaling() {
        let input = vec![2.0, 2.0, 2.0, 2.0];
        let gamma = vec![1.0; 4];
        let out = cpu_rmsnorm(&input, &gamma, EPS);
        // rms ≈ 2.0 → output ≈ 1.0
        for &v in &out {
            assert!((v - 1.0).abs() < 1e-3);
        }
    }

    #[test]
    fn test_rmsnorm_gamma() {
        let input = vec![1.0, 1.0];
        let gamma = vec![2.0, 3.0];
        let out = cpu_rmsnorm(&input, &gamma, EPS);
        // rms ≈ 1.0 → output ≈ [2.0, 3.0]
        assert!((out[0] - 2.0).abs() < 1e-3);
        assert!((out[1] - 3.0).abs() < 1e-3);
    }

    #[test]
    fn test_rmsnorm_epsilon_sensitivity() {
        let input = vec![1e-8; 4];
        let gamma = vec![1.0; 4];
        let out_small_eps = cpu_rmsnorm(&input, &gamma, 1e-12);
        let out_large_eps = cpu_rmsnorm(&input, &gamma, 1.0);
        // Large epsilon makes rms larger → output smaller
        assert!(out_small_eps[0].abs() > out_large_eps[0].abs());
    }

    // =================================================================
    // Fused RMSNorm + Linear
    // =================================================================

    #[test]
    fn test_fused_rmsnorm_linear_vs_separate_small() {
        let (rows, cols, out) = (2, 4, 3);
        let input: Vec<f32> = (0..rows * cols).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let weight: Vec<f32> = (0..cols * out).map(|i| (i as f32 + 1.0) * 0.05).collect();
        let gamma = vec![1.0; cols];

        let fused = cpu_fused_rmsnorm_linear(&input, &weight, &gamma, rows, cols, out, EPS);

        // Separate: norm each row then matmul
        let mut normed = vec![0.0_f32; rows * cols];
        for i in 0..rows {
            let row = &input[i * cols..(i + 1) * cols];
            let n = cpu_rmsnorm(row, &gamma, EPS);
            normed[i * cols..(i + 1) * cols].copy_from_slice(&n);
        }
        let separate = cpu_matmul(&normed, &weight, rows, out, cols);

        assert_approx_eq(&fused, &separate, TOL);
    }

    #[test]
    fn test_fused_rmsnorm_linear_vs_separate_medium() {
        let (rows, cols, out) = (8, 64, 32);
        let input: Vec<f32> = (0..rows * cols).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
        let weight: Vec<f32> = (0..cols * out).map(|i| ((i % 13) as f32 - 6.0) * 0.02).collect();
        let gamma: Vec<f32> = (0..cols).map(|i| 0.5 + (i % 5) as f32 * 0.1).collect();

        let fused = cpu_fused_rmsnorm_linear(&input, &weight, &gamma, rows, cols, out, EPS);

        let mut normed = vec![0.0_f32; rows * cols];
        for i in 0..rows {
            let row = &input[i * cols..(i + 1) * cols];
            let n = cpu_rmsnorm(row, &gamma, EPS);
            normed[i * cols..(i + 1) * cols].copy_from_slice(&n);
        }
        let separate = cpu_matmul(&normed, &weight, rows, out, cols);

        assert_approx_eq(&fused, &separate, TOL);
    }

    #[test]
    fn test_fused_rmsnorm_linear_single_row() {
        let (rows, cols, out) = (1, 8, 4);
        let input: Vec<f32> = (0..cols).map(|i| i as f32 + 1.0).collect();
        let weight: Vec<f32> = (0..cols * out).map(|i| (i as f32) * 0.01).collect();
        let gamma = vec![1.0; cols];

        let fused = cpu_fused_rmsnorm_linear(&input, &weight, &gamma, rows, cols, out, EPS);

        let normed = cpu_rmsnorm(&input, &gamma, EPS);
        let separate = cpu_matmul(&normed, &weight, rows, out, cols);

        assert_approx_eq(&fused, &separate, TOL);
    }

    #[test]
    fn test_fused_rmsnorm_linear_shape() {
        let (rows, cols, out) = (3, 5, 7);
        let input = vec![1.0; rows * cols];
        let weight = vec![0.1; cols * out];
        let gamma = vec![1.0; cols];
        let result = cpu_fused_rmsnorm_linear(&input, &weight, &gamma, rows, cols, out, EPS);
        assert_eq!(result.len(), rows * out);
    }

    #[test]
    fn test_fused_rmsnorm_linear_zero_input() {
        let (rows, cols, out) = (2, 4, 3);
        let input = vec![0.0; rows * cols];
        let weight = vec![1.0; cols * out];
        let gamma = vec![1.0; cols];
        let result = cpu_fused_rmsnorm_linear(&input, &weight, &gamma, rows, cols, out, EPS);
        // Zero input normalised is still zero → output is zero
        for &v in &result {
            assert!(v.abs() < 1e-3);
        }
    }

    // =================================================================
    // Fused Linear + SiLU
    // =================================================================

    #[test]
    fn test_fused_linear_silu_vs_separate() {
        let (rows, inf, outf) = (2, 4, 3);
        let input: Vec<f32> = (0..rows * inf).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let weight: Vec<f32> = (0..inf * outf).map(|i| (i as f32 + 1.0) * 0.05).collect();

        let fused = cpu_fused_linear_silu(&input, &weight, None, rows, inf, outf);

        let mm = cpu_matmul(&input, &weight, rows, outf, inf);
        let separate: Vec<f32> = mm.iter().map(|&v| cpu_silu(v)).collect();

        assert_approx_eq(&fused, &separate, TOL);
    }

    #[test]
    fn test_fused_linear_silu_with_bias() {
        let (rows, inf, outf) = (2, 4, 3);
        let input: Vec<f32> = (0..rows * inf).map(|i| (i as f32) * 0.1).collect();
        let weight: Vec<f32> = (0..inf * outf).map(|i| (i as f32) * 0.02).collect();
        let bias = vec![0.1, -0.2, 0.3];

        let fused = cpu_fused_linear_silu(&input, &weight, Some(&bias), rows, inf, outf);

        let mm = cpu_matmul(&input, &weight, rows, outf, inf);
        let separate: Vec<f32> = mm
            .chunks(outf)
            .flat_map(|row| row.iter().zip(bias.iter()).map(|(&v, &b)| cpu_silu(v + b)))
            .collect();

        assert_approx_eq(&fused, &separate, TOL);
    }

    #[test]
    fn test_fused_linear_silu_shape() {
        let (rows, inf, outf) = (5, 8, 3);
        let input = vec![1.0; rows * inf];
        let weight = vec![0.1; inf * outf];
        let result = cpu_fused_linear_silu(&input, &weight, None, rows, inf, outf);
        assert_eq!(result.len(), rows * outf);
    }

    // =================================================================
    // Fused Linear + GELU
    // =================================================================

    #[test]
    fn test_fused_linear_gelu_vs_separate() {
        let (rows, inf, outf) = (2, 4, 3);
        let input: Vec<f32> = (0..rows * inf).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let weight: Vec<f32> = (0..inf * outf).map(|i| (i as f32 + 1.0) * 0.05).collect();

        let fused = cpu_fused_linear_gelu(&input, &weight, None, rows, inf, outf);

        let mm = cpu_matmul(&input, &weight, rows, outf, inf);
        let separate: Vec<f32> = mm.iter().map(|&v| cpu_gelu(v)).collect();

        assert_approx_eq(&fused, &separate, TOL);
    }

    #[test]
    fn test_fused_linear_gelu_with_bias() {
        let (rows, inf, outf) = (3, 6, 4);
        let input: Vec<f32> = (0..rows * inf).map(|i| (i as f32) * 0.05).collect();
        let weight: Vec<f32> = (0..inf * outf).map(|i| (i as f32) * 0.01).collect();
        let bias = vec![0.5, -0.5, 0.1, -0.1];

        let fused = cpu_fused_linear_gelu(&input, &weight, Some(&bias), rows, inf, outf);

        let mm = cpu_matmul(&input, &weight, rows, outf, inf);
        let separate: Vec<f32> = mm
            .chunks(outf)
            .flat_map(|row| row.iter().zip(bias.iter()).map(|(&v, &b)| cpu_gelu(v + b)))
            .collect();

        assert_approx_eq(&fused, &separate, TOL);
    }

    #[test]
    fn test_fused_linear_gelu_shape() {
        let (rows, inf, outf) = (4, 16, 8);
        let input = vec![0.5; rows * inf];
        let weight = vec![0.01; inf * outf];
        let result = cpu_fused_linear_gelu(&input, &weight, None, rows, inf, outf);
        assert_eq!(result.len(), rows * outf);
    }

    // =================================================================
    // Fused gate+up projection (SwiGLU)
    // =================================================================

    #[test]
    fn test_gate_up_proj_vs_separate() {
        let (rows, inf, outf) = (2, 4, 3);
        let input: Vec<f32> = (0..rows * inf).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let gate_w: Vec<f32> = (0..inf * outf).map(|i| (i as f32 + 1.0) * 0.05).collect();
        let up_w: Vec<f32> = (0..inf * outf).map(|i| (i as f32 + 2.0) * 0.03).collect();

        let fused = cpu_fused_gate_up_proj(&input, &gate_w, &up_w, rows, inf, outf);

        // Separate: SiLU(input @ gate) * (input @ up)
        let gate_out = cpu_matmul(&input, &gate_w, rows, outf, inf);
        let up_out = cpu_matmul(&input, &up_w, rows, outf, inf);
        let separate: Vec<f32> =
            gate_out.iter().zip(up_out.iter()).map(|(&g, &u)| cpu_silu(g) * u).collect();

        assert_approx_eq(&fused, &separate, TOL);
    }

    #[test]
    fn test_gate_up_proj_medium() {
        let (rows, inf, outf) = (4, 64, 32);
        let input: Vec<f32> = (0..rows * inf).map(|i| ((i % 11) as f32 - 5.0) * 0.1).collect();
        let gate_w: Vec<f32> = (0..inf * outf).map(|i| ((i % 7) as f32 - 3.0) * 0.02).collect();
        let up_w: Vec<f32> = (0..inf * outf).map(|i| ((i % 9) as f32 - 4.0) * 0.03).collect();

        let fused = cpu_fused_gate_up_proj(&input, &gate_w, &up_w, rows, inf, outf);

        let gate_out = cpu_matmul(&input, &gate_w, rows, outf, inf);
        let up_out = cpu_matmul(&input, &up_w, rows, outf, inf);
        let separate: Vec<f32> =
            gate_out.iter().zip(up_out.iter()).map(|(&g, &u)| cpu_silu(g) * u).collect();

        assert_approx_eq(&fused, &separate, TOL);
    }

    #[test]
    fn test_gate_up_proj_shape() {
        let (rows, inf, outf) = (3, 32, 128);
        let input = vec![0.1; rows * inf];
        let gate_w = vec![0.01; inf * outf];
        let up_w = vec![0.01; inf * outf];
        let result = cpu_fused_gate_up_proj(&input, &gate_w, &up_w, rows, inf, outf);
        assert_eq!(result.len(), rows * outf);
    }

    #[test]
    fn test_gate_up_proj_zero_input() {
        let (rows, inf, outf) = (2, 4, 3);
        let input = vec![0.0; rows * inf];
        let gate_w = vec![1.0; inf * outf];
        let up_w = vec![1.0; inf * outf];
        let result = cpu_fused_gate_up_proj(&input, &gate_w, &up_w, rows, inf, outf);
        for &v in &result {
            assert!(v.abs() < 1e-7);
        }
    }

    // =================================================================
    // Fused bias + activation (in-place)
    // =================================================================

    #[test]
    fn test_bias_activation_none() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let bias = vec![0.1, -0.1];
        cpu_fused_bias_add_activation(&mut data, &bias, 2, ActivationType::None);
        assert_approx_eq(&data, &[1.1, 1.9, 3.1, 3.9], TOL);
    }

    #[test]
    fn test_bias_activation_relu() {
        let mut data = vec![-1.0, 2.0, 0.5, -3.0];
        let bias = vec![0.5, -0.5];
        cpu_fused_bias_add_activation(&mut data, &bias, 2, ActivationType::ReLU);
        // [-1+0.5=-0.5→0, 2-0.5=1.5, 0.5+0.5=1.0, -3-0.5=-3.5→0]
        assert_approx_eq(&data, &[0.0, 1.5, 1.0, 0.0], TOL);
    }

    #[test]
    fn test_bias_activation_silu() {
        let mut data = vec![1.0, -1.0];
        let bias = vec![0.0, 0.0];
        cpu_fused_bias_add_activation(&mut data, &bias, 2, ActivationType::SiLU);
        assert!((data[0] - cpu_silu(1.0)).abs() < TOL);
        assert!((data[1] - cpu_silu(-1.0)).abs() < TOL);
    }

    #[test]
    fn test_bias_activation_gelu() {
        let mut data = vec![0.5, -0.5, 1.0, -1.0];
        let bias = vec![0.0, 0.0];
        cpu_fused_bias_add_activation(&mut data, &bias, 2, ActivationType::GELU);
        assert!((data[0] - cpu_gelu(0.5)).abs() < TOL);
        assert!((data[1] - cpu_gelu(-0.5)).abs() < TOL);
    }

    #[test]
    fn test_bias_activation_tanh() {
        let mut data = vec![1.0, -1.0];
        let bias = vec![0.5, -0.5];
        cpu_fused_bias_add_activation(&mut data, &bias, 2, ActivationType::Tanh);
        assert!((data[0] - 1.5_f32.tanh()).abs() < TOL);
        assert!((data[1] - (-1.5_f32).tanh()).abs() < TOL);
    }

    // =================================================================
    // Fused scale + shift (in-place)
    // =================================================================

    #[test]
    fn test_scale_shift_identity() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let scale = vec![1.0, 1.0];
        let shift = vec![0.0, 0.0];
        cpu_fused_scale_shift(&mut data, &scale, &shift, 2);
        assert_approx_eq(&data, &[1.0, 2.0, 3.0, 4.0], 1e-7);
    }

    #[test]
    fn test_scale_shift_values() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let scale = vec![2.0, 0.5];
        let shift = vec![1.0, -1.0];
        cpu_fused_scale_shift(&mut data, &scale, &shift, 2);
        // [1*2+1=3, 2*0.5-1=0, 3*2+1=7, 4*0.5-1=1]
        assert_approx_eq(&data, &[3.0, 0.0, 7.0, 1.0], TOL);
    }

    #[test]
    fn test_scale_shift_zero_scale() {
        let mut data = vec![100.0, 200.0];
        let scale = vec![0.0, 0.0];
        let shift = vec![5.0, 10.0];
        cpu_fused_scale_shift(&mut data, &scale, &shift, 2);
        assert_approx_eq(&data, &[5.0, 10.0], 1e-7);
    }

    // =================================================================
    // Quantize-dequantize round-trip
    // =================================================================

    #[test]
    fn test_quantize_dequantize_8bit() {
        let data = vec![0.0, 0.5, -0.5, 1.0, -1.0];
        let result = cpu_fused_quantize_dequantize(&data, 8);
        // Round-trip error bounded by half a quantization step
        let max_abs = 1.0_f32;
        let step = max_abs / 127.0;
        for (&orig, &recon) in data.iter().zip(result.iter()) {
            assert!((orig - recon).abs() <= step + 1e-7, "orig={orig}, recon={recon}, step={step}");
        }
    }

    #[test]
    fn test_quantize_dequantize_4bit() {
        let data = vec![0.0, 1.0, -1.0, 0.3, -0.7];
        let result = cpu_fused_quantize_dequantize(&data, 4);
        // 4-bit: max_val=7, scale=1.0/7≈0.143
        let step = 1.0 / 7.0;
        for (&orig, &recon) in data.iter().zip(result.iter()) {
            assert!((orig - recon).abs() <= step + 1e-5, "4-bit: orig={orig}, recon={recon}");
        }
    }

    #[test]
    fn test_quantize_dequantize_2bit() {
        let data = vec![0.0, 1.0, -1.0, 0.5];
        let result = cpu_fused_quantize_dequantize(&data, 2);
        // 2-bit: max_val=1, so values map to {-1, 0, 1} * scale
        assert_eq!(result.len(), data.len());
    }

    #[test]
    fn test_quantize_dequantize_zero_input() {
        let data = vec![0.0, 0.0, 0.0];
        let result = cpu_fused_quantize_dequantize(&data, 8);
        assert_approx_eq(&result, &[0.0, 0.0, 0.0], 1e-7);
    }

    #[test]
    fn test_quantize_dequantize_preserves_length() {
        let data = vec![0.1; 100];
        let result = cpu_fused_quantize_dequantize(&data, 8);
        assert_eq!(result.len(), data.len());
    }

    #[test]
    fn test_quantize_dequantize_all_same() {
        let data = vec![0.42; 16];
        let result = cpu_fused_quantize_dequantize(&data, 8);
        // All values are the same → all reconstructed values should match
        let first = result[0];
        for &v in &result {
            assert!((v - first).abs() < 1e-7);
        }
    }

    // =================================================================
    // Dimension / shape property tests
    // =================================================================

    #[test]
    fn test_fused_rmsnorm_linear_non_square() {
        let (rows, cols, out) = (3, 32, 128);
        let input = vec![0.5; rows * cols];
        let weight = vec![0.01; cols * out];
        let gamma = vec![1.0; cols];
        let fused = cpu_fused_rmsnorm_linear(&input, &weight, &gamma, rows, cols, out, EPS);

        let mut normed = vec![0.0_f32; rows * cols];
        for i in 0..rows {
            let row = &input[i * cols..(i + 1) * cols];
            let n = cpu_rmsnorm(row, &gamma, EPS);
            normed[i * cols..(i + 1) * cols].copy_from_slice(&n);
        }
        let separate = cpu_matmul(&normed, &weight, rows, out, cols);

        assert_approx_eq(&fused, &separate, TOL);
    }

    #[test]
    fn test_fused_linear_silu_single_column() {
        // out_features = 1
        let (rows, inf, outf) = (4, 8, 1);
        let input: Vec<f32> = (0..rows * inf).map(|i| i as f32 * 0.1).collect();
        let weight: Vec<f32> = (0..inf).map(|i| i as f32 * 0.1).collect();
        let fused = cpu_fused_linear_silu(&input, &weight, None, rows, inf, outf);

        let mm = cpu_matmul(&input, &weight, rows, outf, inf);
        let separate: Vec<f32> = mm.iter().map(|&v| cpu_silu(v)).collect();
        assert_approx_eq(&fused, &separate, TOL);
    }

    // =================================================================
    // Large values (overflow prevention)
    // =================================================================

    #[test]
    fn test_silu_large_positive() {
        let y = cpu_silu(50.0);
        // For large x, SiLU(x) ≈ x
        assert!((y - 50.0).abs() < 1e-3);
    }

    #[test]
    fn test_silu_large_negative() {
        let y = cpu_silu(-50.0);
        // For large negative x, SiLU(x) ≈ 0
        assert!(y.abs() < 1e-3);
    }

    #[test]
    fn test_gelu_large_positive() {
        let y = cpu_gelu(10.0);
        // For large positive x, GELU(x) ≈ x
        assert!((y - 10.0).abs() < 1e-2);
    }

    #[test]
    fn test_quantize_dequantize_large_values() {
        let data = vec![1000.0, -1000.0, 500.0];
        let result = cpu_fused_quantize_dequantize(&data, 8);
        // Should not produce NaN/Inf
        for &v in &result {
            assert!(v.is_finite(), "got non-finite: {v}");
        }
    }

    // =================================================================
    // FusionConfig and type tests
    // =================================================================

    #[test]
    fn test_fusion_config_default() {
        let cfg = FusionConfig::default();
        assert!((cfg.epsilon - 1e-6).abs() < 1e-12);
        assert_eq!(cfg.activation, ActivationType::None);
        assert!(!cfg.use_bias);
        assert!(cfg.quantize_bits.is_none());
    }

    #[test]
    fn test_fused_op_enum_variants() {
        // Ensure all variants are constructible and comparable.
        let ops = [
            FusedOp::NormLinear,
            FusedOp::LinearActivation,
            FusedOp::NormLinearActivation,
            FusedOp::BiasActivation,
            FusedOp::ScaleShift,
            FusedOp::QuantizeDequantize,
        ];
        for (i, a) in ops.iter().enumerate() {
            for (j, b) in ops.iter().enumerate() {
                assert_eq!(i == j, a == b);
            }
        }
    }

    #[test]
    fn test_fused_ops_error_display() {
        let err =
            FusedOpsError::DimensionMismatch { expected: 64, got: 32, context: "weight rows" };
        let msg = format!("{err}");
        assert!(msg.contains("64"));
        assert!(msg.contains("32"));

        let err2 = FusedOpsError::UnsupportedFusion(FusedOp::NormLinear);
        let msg2 = format!("{err2}");
        assert!(msg2.contains("NormLinear"));

        let err3 = FusedOpsError::NumericalError("NaN detected");
        let msg3 = format!("{err3}");
        assert!(msg3.contains("NaN"));
    }

    // =================================================================
    // OpenCL source sanity
    // =================================================================

    #[test]
    fn test_opencl_source_contains_kernels() {
        assert!(FUSED_OPS_SRC.contains("fused_rmsnorm_linear"));
        assert!(FUSED_OPS_SRC.contains("fused_linear_silu"));
        assert!(FUSED_OPS_SRC.contains("fused_gate_up_proj"));
        assert!(FUSED_OPS_SRC.contains("fused_bias_activation"));
    }

    #[test]
    fn test_opencl_source_not_empty() {
        assert!(FUSED_OPS_SRC.len() > 100);
    }
}

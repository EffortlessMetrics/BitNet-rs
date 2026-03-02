//! CPU gradient computation operations for backward pass support.
//!
//! Provides element-wise activation gradients, softmax / layer-norm
//! backward passes, matmul backward, cross-entropy backward, and
//! gradient clipping utilities needed for future training support.

use std::fmt;

// ── Error type ─────────────────────────────────────────────────────

/// Gradient computation errors.
#[derive(Debug)]
pub enum GradientError {
    /// Shape / length mismatch between buffers.
    DimensionMismatch { expected: usize, got: usize },
    /// Semantically invalid gradient (e.g. NaN in output).
    InvalidGradient(String),
    /// Operation would produce non-finite values.
    NumericalInstability(String),
}

impl fmt::Display for GradientError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::InvalidGradient(msg) => write!(f, "invalid gradient: {msg}"),
            Self::NumericalInstability(msg) => write!(f, "numerical instability: {msg}"),
        }
    }
}

impl std::error::Error for GradientError {}

/// Convenience alias.
pub type GradResult<T> = std::result::Result<T, GradientError>;

// ── Helpers ────────────────────────────────────────────────────────

fn check_len(a: usize, b: usize, name: &str) -> GradResult<()> {
    if a != b {
        return Err(GradientError::DimensionMismatch { expected: a, got: b });
    }
    if a == 0 {
        return Err(GradientError::InvalidGradient(format!("{name}: zero-length input")));
    }
    Ok(())
}

// ── Activation gradients ───────────────────────────────────────────

/// ReLU backward: `grad_input[i] = grad_output[i] * (input[i] > 0)`.
pub fn relu_backward(grad_output: &[f32], input: &[f32], grad_input: &mut [f32]) -> GradResult<()> {
    check_len(grad_output.len(), input.len(), "relu_backward")?;
    check_len(grad_output.len(), grad_input.len(), "relu_backward")?;
    for ((go, x), gi) in grad_output.iter().zip(input.iter()).zip(grad_input.iter_mut()) {
        *gi = if *x > 0.0 { *go } else { 0.0 };
    }
    Ok(())
}

/// GELU backward (tanh approximation).
///
/// Uses `GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))` and its
/// analytically-derived gradient.
pub fn gelu_backward(grad_output: &[f32], input: &[f32], grad_input: &mut [f32]) -> GradResult<()> {
    check_len(grad_output.len(), input.len(), "gelu_backward")?;
    check_len(grad_output.len(), grad_input.len(), "gelu_backward")?;
    const SQRT_2_OVER_PI: f32 = 0.797_884_6; // sqrt(2/π)
    const COEFF: f32 = 0.044_715;
    for ((go, x), gi) in grad_output.iter().zip(input.iter()).zip(grad_input.iter_mut()) {
        // tanh approximation: GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
        let x3 = *x * *x * *x;
        let inner = SQRT_2_OVER_PI * (*x + COEFF * x3);
        let t = inner.tanh();
        let sech2 = 1.0 - t * t;
        let d_inner = SQRT_2_OVER_PI * (1.0 + 3.0 * COEFF * *x * *x);
        *gi = *go * (0.5 * (1.0 + t) + 0.5 * *x * sech2 * d_inner);
    }
    Ok(())
}

/// SiLU (Swish) backward: `σ(x) + x·σ(x)·(1 − σ(x))`.
pub fn silu_backward(grad_output: &[f32], input: &[f32], grad_input: &mut [f32]) -> GradResult<()> {
    check_len(grad_output.len(), input.len(), "silu_backward")?;
    check_len(grad_output.len(), grad_input.len(), "silu_backward")?;
    for ((go, x), gi) in grad_output.iter().zip(input.iter()).zip(grad_input.iter_mut()) {
        let sig = 1.0 / (1.0 + (-*x).exp());
        *gi = *go * (sig + *x * sig * (1.0 - sig));
    }
    Ok(())
}

/// Sigmoid backward: `grad * output * (1 − output)`.
///
/// `output` is the forward-pass sigmoid output, not the raw input.
pub fn sigmoid_backward(
    grad_output: &[f32],
    output: &[f32],
    grad_input: &mut [f32],
) -> GradResult<()> {
    check_len(grad_output.len(), output.len(), "sigmoid_backward")?;
    check_len(grad_output.len(), grad_input.len(), "sigmoid_backward")?;
    for ((go, o), gi) in grad_output.iter().zip(output.iter()).zip(grad_input.iter_mut()) {
        *gi = *go * *o * (1.0 - *o);
    }
    Ok(())
}

/// Tanh backward: `grad * (1 − output²)`.
///
/// `output` is the forward-pass tanh output.
pub fn tanh_backward(
    grad_output: &[f32],
    output: &[f32],
    grad_input: &mut [f32],
) -> GradResult<()> {
    check_len(grad_output.len(), output.len(), "tanh_backward")?;
    check_len(grad_output.len(), grad_input.len(), "tanh_backward")?;
    for ((go, o), gi) in grad_output.iter().zip(output.iter()).zip(grad_input.iter_mut()) {
        *gi = *go * (1.0 - *o * *o);
    }
    Ok(())
}

// ── Softmax backward ───────────────────────────────────────────────

/// Softmax backward via Jacobian-vector product.
///
/// For a single row of softmax output `s`, the JVP is:
///   `grad_input = s * (grad_output − dot(grad_output, s))`
///
/// `grad_output` and `output` are flat `[batch_size × num_classes]`.
pub fn softmax_backward(
    grad_output: &[f32],
    output: &[f32],
    num_classes: usize,
    grad_input: &mut [f32],
) -> GradResult<()> {
    check_len(grad_output.len(), output.len(), "softmax_backward")?;
    check_len(grad_output.len(), grad_input.len(), "softmax_backward")?;
    if num_classes == 0 {
        return Err(GradientError::InvalidGradient(
            "softmax_backward: num_classes must be > 0".into(),
        ));
    }
    if !grad_output.len().is_multiple_of(num_classes) {
        return Err(GradientError::DimensionMismatch {
            expected: (grad_output.len() / num_classes) * num_classes,
            got: grad_output.len(),
        });
    }
    let batch_size = grad_output.len() / num_classes;
    for b in 0..batch_size {
        let start = b * num_classes;
        let end = start + num_classes;
        let go = &grad_output[start..end];
        let s = &output[start..end];
        let dot: f32 = go.iter().zip(s.iter()).map(|(g, si)| g * si).sum();
        for j in 0..num_classes {
            grad_input[start + j] = s[j] * (go[j] - dot);
        }
    }
    Ok(())
}

// ── Layer-norm backward ────────────────────────────────────────────

/// Layer normalization backward pass.
///
/// Computes `grad_input`, `grad_weight`, and `grad_bias` given the
/// saved forward-pass statistics (`mean`, `rstd`).
///
/// All flat buffers are `[batch_size × norm_size]` for `grad_output`
/// and `input`; `weight`, `grad_weight`, `grad_bias` are `[norm_size]`
/// (per-feature); `mean` and `rstd` are `[batch_size]`.
#[allow(clippy::too_many_arguments)]
pub fn layer_norm_backward(
    grad_output: &[f32],
    input: &[f32],
    weight: &[f32],
    mean: &[f32],
    rstd: &[f32],
    norm_size: usize,
    grad_input: &mut [f32],
    grad_weight: &mut [f32],
    grad_bias: &mut [f32],
) -> GradResult<()> {
    check_len(grad_output.len(), input.len(), "layer_norm_backward")?;
    check_len(grad_output.len(), grad_input.len(), "layer_norm_backward")?;
    check_len(weight.len(), norm_size, "layer_norm_backward weight")?;
    check_len(grad_weight.len(), norm_size, "layer_norm_backward grad_weight")?;
    check_len(grad_bias.len(), norm_size, "layer_norm_backward grad_bias")?;
    if norm_size == 0 || !grad_output.len().is_multiple_of(norm_size) {
        return Err(GradientError::InvalidGradient(
            "layer_norm_backward: invalid norm_size".into(),
        ));
    }
    let batch_size = grad_output.len() / norm_size;
    check_len(mean.len(), batch_size, "layer_norm_backward mean")?;
    check_len(rstd.len(), batch_size, "layer_norm_backward rstd")?;

    let n = norm_size as f32;

    // Zero accumulators.
    grad_weight.iter_mut().for_each(|v| *v = 0.0);
    grad_bias.iter_mut().for_each(|v| *v = 0.0);

    for b in 0..batch_size {
        let off = b * norm_size;
        let go = &grad_output[off..off + norm_size];
        let x = &input[off..off + norm_size];
        let mu = mean[b];
        let rs = rstd[b];

        // Accumulate grad_weight and grad_bias.
        for j in 0..norm_size {
            let x_hat = (x[j] - mu) * rs;
            grad_weight[j] += go[j] * x_hat;
            grad_bias[j] += go[j];
        }

        // grad_input for this row.
        let mut sum_go_w: f32 = 0.0;
        let mut sum_go_w_xhat: f32 = 0.0;
        for j in 0..norm_size {
            let gw = go[j] * weight[j];
            sum_go_w += gw;
            sum_go_w_xhat += gw * (x[j] - mu) * rs;
        }
        for j in 0..norm_size {
            let x_hat = (x[j] - mu) * rs;
            grad_input[off + j] =
                rs * (go[j] * weight[j] - sum_go_w / n - x_hat * sum_go_w_xhat / n);
        }
    }
    Ok(())
}

// ── Matmul backward ────────────────────────────────────────────────

/// Matrix multiply backward: `Y = A @ B`.
///
/// - `grad_a = grad_output @ Bᵀ`
/// - `grad_b = Aᵀ @ grad_output`
///
/// Dimensions: A is `[m × k]`, B is `[k × n]`, Y/grad_output is `[m × n]`.
pub fn matmul_backward(
    grad_output: &[f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    grad_a: &mut [f32],
    grad_b: &mut [f32],
) -> GradResult<()> {
    check_len(grad_output.len(), m * n, "matmul_backward grad_output")?;
    check_len(a.len(), m * k, "matmul_backward A")?;
    check_len(b.len(), k * n, "matmul_backward B")?;
    check_len(grad_a.len(), m * k, "matmul_backward grad_a")?;
    check_len(grad_b.len(), k * n, "matmul_backward grad_b")?;

    // grad_a = grad_output @ B^T  (m×n @ n×k → m×k)
    for i in 0..m {
        for j in 0..k {
            let mut sum = 0.0_f32;
            for p in 0..n {
                sum += grad_output[i * n + p] * b[j * n + p];
            }
            grad_a[i * k + j] = sum;
        }
    }

    // grad_b = A^T @ grad_output  (k×m @ m×n → k×n)
    grad_b.iter_mut().for_each(|v| *v = 0.0);
    for i in 0..m {
        for j in 0..n {
            let go = grad_output[i * n + j];
            for p in 0..k {
                grad_b[p * n + j] += a[i * k + p] * go;
            }
        }
    }
    Ok(())
}

// ── Cross-entropy backward ─────────────────────────────────────────

/// Cross-entropy loss backward (softmax + NLL combined).
///
/// `logits` is `[batch_size × num_classes]`, `targets` holds class indices.
/// The gradient is `softmax(logits) − one_hot(targets)`, scaled by
/// `grad_output` (upstream scalar, typically `1/batch_size`).
pub fn cross_entropy_backward(
    grad_output: f32,
    logits: &[f32],
    targets: &[usize],
    num_classes: usize,
    grad_input: &mut [f32],
) -> GradResult<()> {
    if targets.is_empty() {
        return Err(GradientError::InvalidGradient("cross_entropy_backward: empty targets".into()));
    }
    let batch_size = targets.len();
    check_len(logits.len(), batch_size * num_classes, "cross_entropy_backward logits")?;
    check_len(grad_input.len(), batch_size * num_classes, "cross_entropy_backward grad_input")?;

    for (i, &target) in targets.iter().enumerate() {
        if target >= num_classes {
            return Err(GradientError::InvalidGradient(format!(
                "cross_entropy_backward: target[{i}]={target} >= num_classes={num_classes}"
            )));
        }
        let off = i * num_classes;
        let row = &logits[off..off + num_classes];

        // Stable softmax.
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = row.iter().map(|&v| (v - max_val).exp()).sum();
        for j in 0..num_classes {
            let softmax_j = (row[j] - max_val).exp() / sum_exp;
            let indicator = if j == target { 1.0 } else { 0.0 };
            grad_input[off + j] = grad_output * (softmax_j - indicator);
        }
    }
    Ok(())
}

// ── Gradient clipping ──────────────────────────────────────────────

/// Clip gradients by global L2 norm, returning the original total norm.
///
/// If the total norm exceeds `max_norm`, every element is scaled by
/// `max_norm / total_norm`.  Otherwise the buffer is unchanged.
pub fn gradient_clip_norm(gradients: &mut [f32], max_norm: f32) -> GradResult<f32> {
    if max_norm <= 0.0 {
        return Err(GradientError::InvalidGradient(
            "gradient_clip_norm: max_norm must be > 0".into(),
        ));
    }
    let total_norm: f32 = gradients.iter().map(|g| g * g).sum::<f32>().sqrt();
    if total_norm > max_norm {
        let scale = max_norm / total_norm;
        gradients.iter_mut().for_each(|g| *g *= scale);
    }
    Ok(total_norm)
}

/// Clip gradients element-wise to `[−max_value, max_value]`.
pub fn gradient_clip_value(gradients: &mut [f32], max_value: f32) -> GradResult<()> {
    if max_value <= 0.0 {
        return Err(GradientError::InvalidGradient(
            "gradient_clip_value: max_value must be > 0".into(),
        ));
    }
    gradients.iter_mut().for_each(|g| *g = g.clamp(-max_value, max_value));
    Ok(())
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-4;
    /// Larger tolerance for finite-difference checks.
    const FD_TOL: f32 = 5e-3;
    const EPS_FD: f32 = 1e-4;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    // ── Numerical gradient helper (finite differences) ─────────

    /// Central-difference numerical gradient: `(f(x+ε) − f(x−ε)) / 2ε`.
    fn numerical_grad<F: Fn(&[f32]) -> f32>(f: &F, x: &[f32], idx: usize) -> f32 {
        let mut x_plus = x.to_vec();
        let mut x_minus = x.to_vec();
        x_plus[idx] += EPS_FD;
        x_minus[idx] -= EPS_FD;
        (f(&x_plus) - f(&x_minus)) / (2.0 * EPS_FD)
    }

    // ── ReLU backward ──────────────────────────────────────────

    #[test]
    fn relu_backward_basic() {
        let go = [1.0, 1.0, 1.0, 1.0];
        let input = [2.0, -1.0, 0.0, 3.0];
        let mut gi = [0.0_f32; 4];
        relu_backward(&go, &input, &mut gi).unwrap();
        assert_eq!(gi, [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn relu_backward_scaled_grad() {
        let go = [0.5, -0.3, 2.0];
        let input = [1.0, -1.0, 0.5];
        let mut gi = [0.0_f32; 3];
        relu_backward(&go, &input, &mut gi).unwrap();
        assert_eq!(gi, [0.5, 0.0, 2.0]);
    }

    #[test]
    fn relu_backward_numerical() {
        let input = [2.0_f32, -1.0, 0.5, -3.0];
        let relu_fwd = |x: &[f32]| -> f32 { x.iter().map(|v| v.max(0.0)).sum() };
        let go = [1.0; 4];
        let mut gi = [0.0_f32; 4];
        relu_backward(&go, &input, &mut gi).unwrap();
        for i in 0..input.len() {
            let ng = numerical_grad(&relu_fwd, &input, i);
            assert!(approx(gi[i], ng, FD_TOL), "idx {i}: analytic={} numerical={ng}", gi[i]);
        }
    }

    // ── GELU backward ──────────────────────────────────────────

    #[test]
    fn gelu_backward_at_zero() {
        // GELU'(0) = 0.5 (exactly: Φ(0) + 0·φ(0) = 0.5)
        let go = [1.0];
        let input = [0.0];
        let mut gi = [0.0_f32; 1];
        gelu_backward(&go, &input, &mut gi).unwrap();
        assert!(approx(gi[0], 0.5, TOL), "got {}", gi[0]);
    }

    #[test]
    fn gelu_backward_numerical() {
        let input = [-1.0_f32, 0.0, 0.5, 2.0];
        let gelu_fwd = |x: &[f32]| -> f32 {
            x.iter()
                .map(|v| {
                    let inner = 0.797_884_6_f32 * (*v + 0.044_715 * *v * *v * *v);
                    0.5 * v * (1.0 + inner.tanh())
                })
                .sum()
        };
        let go = [1.0; 4];
        let mut gi = [0.0_f32; 4];
        gelu_backward(&go, &input, &mut gi).unwrap();
        for i in 0..input.len() {
            let ng = numerical_grad(&gelu_fwd, &input, i);
            assert!(approx(gi[i], ng, FD_TOL), "idx {i}: analytic={} numerical={ng}", gi[i]);
        }
    }

    // ── SiLU backward ──────────────────────────────────────────

    #[test]
    fn silu_backward_at_zero() {
        // SiLU'(0) = σ(0) + 0·σ(0)·(1-σ(0)) = 0.5
        let go = [1.0];
        let input = [0.0];
        let mut gi = [0.0_f32; 1];
        silu_backward(&go, &input, &mut gi).unwrap();
        assert!(approx(gi[0], 0.5, TOL), "got {}", gi[0]);
    }

    #[test]
    fn silu_backward_numerical() {
        let input = [-2.0_f32, -0.5, 0.0, 1.0, 3.0];
        let silu_fwd = |x: &[f32]| -> f32 { x.iter().map(|v| v / (1.0 + (-v).exp())).sum() };
        let go = [1.0; 5];
        let mut gi = [0.0_f32; 5];
        silu_backward(&go, &input, &mut gi).unwrap();
        for i in 0..input.len() {
            let ng = numerical_grad(&silu_fwd, &input, i);
            assert!(approx(gi[i], ng, FD_TOL), "idx {i}: analytic={} numerical={ng}", gi[i]);
        }
    }

    // ── Sigmoid backward ───────────────────────────────────────

    #[test]
    fn sigmoid_backward_known_values() {
        // σ(0)=0.5, grad = 1 * 0.5 * 0.5 = 0.25
        let go = [1.0];
        let output = [0.5];
        let mut gi = [0.0_f32; 1];
        sigmoid_backward(&go, &output, &mut gi).unwrap();
        assert!(approx(gi[0], 0.25, TOL), "got {}", gi[0]);
    }

    #[test]
    fn sigmoid_backward_saturation() {
        // At extreme values, gradient → 0.
        let go = [1.0, 1.0];
        let output = [0.999, 0.001];
        let mut gi = [0.0_f32; 2];
        sigmoid_backward(&go, &output, &mut gi).unwrap();
        assert!(gi[0] < 0.01, "saturated high: {}", gi[0]);
        assert!(gi[1] < 0.01, "saturated low: {}", gi[1]);
    }

    #[test]
    fn sigmoid_backward_numerical() {
        let input = [-2.0_f32, 0.0, 1.5];
        let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
        let output: Vec<f32> = input.iter().map(|x| sigmoid(*x)).collect();
        let sig_sum = |x: &[f32]| -> f32 { x.iter().map(|v| sigmoid(*v)).sum() };
        let go = [1.0; 3];
        let mut gi = [0.0_f32; 3];
        sigmoid_backward(&go, &output, &mut gi).unwrap();
        for i in 0..input.len() {
            let ng = numerical_grad(&sig_sum, &input, i);
            assert!(approx(gi[i], ng, FD_TOL), "idx {i}: analytic={} numerical={ng}", gi[i]);
        }
    }

    // ── Tanh backward ──────────────────────────────────────────

    #[test]
    fn tanh_backward_at_zero() {
        // tanh(0)=0, grad = 1 * (1 - 0) = 1
        let go = [1.0];
        let output = [0.0];
        let mut gi = [0.0_f32; 1];
        tanh_backward(&go, &output, &mut gi).unwrap();
        assert!(approx(gi[0], 1.0, TOL), "got {}", gi[0]);
    }

    #[test]
    fn tanh_backward_numerical() {
        let input = [-1.5_f32, 0.0, 0.8, 2.0];
        let output: Vec<f32> = input.iter().map(|x| x.tanh()).collect();
        let tanh_sum = |x: &[f32]| -> f32 { x.iter().map(|v| v.tanh()).sum() };
        let go = [1.0; 4];
        let mut gi = [0.0_f32; 4];
        tanh_backward(&go, &output, &mut gi).unwrap();
        for i in 0..input.len() {
            let ng = numerical_grad(&tanh_sum, &input, i);
            assert!(approx(gi[i], ng, FD_TOL), "idx {i}: analytic={} numerical={ng}", gi[i]);
        }
    }

    // ── Softmax backward ───────────────────────────────────────

    #[test]
    fn softmax_backward_uniform() {
        // Uniform softmax: all 0.25.  grad_output = [1,0,0,0]
        // dot = 0.25;  gi[0] = 0.25*(1-0.25) = 0.1875, gi[j≠0] = -0.0625
        let output = [0.25_f32; 4];
        let go = [1.0, 0.0, 0.0, 0.0];
        let mut gi = [0.0_f32; 4];
        softmax_backward(&go, &output, 4, &mut gi).unwrap();
        assert!(approx(gi[0], 0.1875, TOL), "got {}", gi[0]);
        for j in 1..4 {
            assert!(approx(gi[j], -0.0625, TOL), "got {}", gi[j]);
        }
    }

    #[test]
    fn softmax_backward_batch() {
        let output = [0.5_f32, 0.5, 0.5, 0.5];
        let go = [1.0, 0.0, 0.0, 1.0];
        let mut gi = [0.0_f32; 4];
        softmax_backward(&go, &output, 2, &mut gi).unwrap();
        assert!(approx(gi[0], 0.25, TOL));
        assert!(approx(gi[1], -0.25, TOL));
        assert!(approx(gi[2], -0.25, TOL));
        assert!(approx(gi[3], 0.25, TOL));
    }

    #[test]
    fn softmax_backward_numerical() {
        let logits = [1.0_f32, 2.0, 0.5];
        let max_l = 2.0_f32;
        let sum_exp: f32 = logits.iter().map(|x| (x - max_l).exp()).sum();
        let sm: Vec<f32> = logits.iter().map(|x| (x - max_l).exp() / sum_exp).collect();

        let softmax_fn = |x: &[f32]| -> f32 {
            let mx = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let se: f32 = x.iter().map(|v| (v - mx).exp()).sum();
            // Return log of first class (arbitrary differentiable scalar).
            ((x[0] - mx).exp() / se).ln()
        };

        // Analytical: d(log(s_0))/d(x_j) = δ_{0j} − s_j
        for j in 0..3 {
            let ng = numerical_grad(&softmax_fn, &logits, j);
            let analytic = if j == 0 { 1.0 - sm[0] } else { -sm[j] };
            assert!(approx(analytic, ng, FD_TOL), "idx {j}: a={analytic} n={ng}");
        }
    }

    // ── Layer-norm backward ────────────────────────────────────

    #[test]
    fn layer_norm_backward_identity_weight() {
        let norm_size = 3;
        let input = [1.0_f32, 2.0, 3.0];
        let weight = [1.0_f32; 3];
        let mean = [2.0_f32];
        let var: f32 = (1.0 + 0.0 + 1.0) / 3.0;
        let rstd = [1.0 / (var + 1e-5_f32).sqrt()];

        let go = [1.0_f32, 0.0, 0.0];
        let mut gi = [0.0_f32; 3];
        let mut gw = [0.0_f32; 3];
        let mut gb = [0.0_f32; 3];
        layer_norm_backward(
            &go, &input, &weight, &mean, &rstd, norm_size, &mut gi, &mut gw, &mut gb,
        )
        .unwrap();

        // grad_bias should equal go.
        assert!(approx(gb[0], 1.0, TOL));
        assert!(approx(gb[1], 0.0, TOL));
        assert!(approx(gb[2], 0.0, TOL));

        // grad_input should sum to ~0 (LayerNorm property).
        let gi_sum: f32 = gi.iter().sum();
        assert!(approx(gi_sum, 0.0, TOL), "gi sum={gi_sum}");
    }

    #[test]
    fn layer_norm_backward_batch() {
        let norm_size = 2;
        let input = [1.0_f32, 3.0, 2.0, 4.0];
        let weight = [1.0_f32, 1.0];
        let mean = [2.0_f32, 3.0];
        let var0: f32 = ((1.0 - 2.0_f32).powi(2) + (3.0 - 2.0_f32).powi(2)) / 2.0;
        let var1: f32 = ((2.0 - 3.0_f32).powi(2) + (4.0 - 3.0_f32).powi(2)) / 2.0;
        let rstd = [1.0 / (var0 + 1e-5_f32).sqrt(), 1.0 / (var1 + 1e-5_f32).sqrt()];

        let go = [1.0_f32, 1.0, 1.0, 1.0];
        let mut gi = [0.0_f32; 4];
        let mut gw = [0.0_f32; 2];
        let mut gb = [0.0_f32; 2];
        layer_norm_backward(
            &go, &input, &weight, &mean, &rstd, norm_size, &mut gi, &mut gw, &mut gb,
        )
        .unwrap();

        // When go is all-ones, grad_input should be ~0 for each row.
        let sum0: f32 = gi[0] + gi[1];
        let sum1: f32 = gi[2] + gi[3];
        assert!(approx(sum0, 0.0, TOL), "row0 sum={sum0}");
        assert!(approx(sum1, 0.0, TOL), "row1 sum={sum1}");
    }

    // ── Matmul backward ────────────────────────────────────────

    #[test]
    fn matmul_backward_identity() {
        // A=I, B=[[1,2],[3,4]], go=I → grad_a = B^T, grad_b = I
        let go = [1.0, 0.0, 0.0, 1.0];
        let a = [1.0, 0.0, 0.0, 1.0];
        let b = [1.0, 2.0, 3.0, 4.0];
        let mut ga = [0.0_f32; 4];
        let mut gb = [0.0_f32; 4];
        matmul_backward(&go, &a, &b, 2, 2, 2, &mut ga, &mut gb).unwrap();
        assert!(approx(ga[0], 1.0, TOL));
        assert!(approx(ga[1], 3.0, TOL));
        assert!(approx(ga[2], 2.0, TOL));
        assert!(approx(ga[3], 4.0, TOL));
        assert!(approx(gb[0], 1.0, TOL));
        assert!(approx(gb[1], 0.0, TOL));
        assert!(approx(gb[2], 0.0, TOL));
        assert!(approx(gb[3], 1.0, TOL));
    }

    #[test]
    fn matmul_backward_non_square() {
        // A=[2,3], B=[3,2], go=[2,2] all-ones
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let go = [1.0; 4];
        let mut ga = [0.0_f32; 6];
        let mut gb = [0.0_f32; 6];
        matmul_backward(&go, &a, &b, 2, 3, 2, &mut ga, &mut gb).unwrap();
        // ga[0,0] = 1*1+1*2=3, ga[0,1]=1*3+1*4=7, ga[0,2]=1*5+1*6=11
        assert!(approx(ga[0], 3.0, TOL));
        assert!(approx(ga[1], 7.0, TOL));
        assert!(approx(ga[2], 11.0, TOL));
    }

    #[test]
    fn matmul_backward_numerical() {
        let a = [1.0_f32, 2.0, 3.0, 4.0];
        let b = [5.0_f32, 6.0, 7.0, 8.0];
        let matmul_sum_a = |a_flat: &[f32]| -> f32 {
            let mut y = [0.0_f32; 4];
            for i in 0..2 {
                for j in 0..2 {
                    for p in 0..2 {
                        y[i * 2 + j] += a_flat[i * 2 + p] * b[p * 2 + j];
                    }
                }
            }
            y.iter().sum()
        };
        let go = [1.0_f32; 4];
        let mut ga = [0.0_f32; 4];
        let mut gb = [0.0_f32; 4];
        matmul_backward(&go, &a, &b, 2, 2, 2, &mut ga, &mut gb).unwrap();
        for i in 0..4 {
            let ng = numerical_grad(&matmul_sum_a, &a, i);
            // Matmul accumulates FD error across sums; use relative tolerance.
            let tol = 0.01 * ga[i].abs().max(1.0);
            assert!(approx(ga[i], ng, tol), "idx {i}: analytic={} numerical={ng}", ga[i]);
        }
    }

    // ── Cross-entropy backward ─────────────────────────────────

    #[test]
    fn cross_entropy_backward_basic() {
        let logits = [1.0_f32, 2.0, 0.5];
        let targets = [1_usize];
        let mut gi = [0.0_f32; 3];
        cross_entropy_backward(1.0, &logits, &targets, 3, &mut gi).unwrap();
        assert!(gi[1] < 0.0, "target class grad should be negative, got {}", gi[1]);
        assert!(gi[0] > 0.0);
        assert!(gi[2] > 0.0);
        let sum: f32 = gi.iter().sum();
        assert!(approx(sum, 0.0, TOL), "sum={sum}");
    }

    #[test]
    fn cross_entropy_backward_numerical() {
        let logits = [1.0_f32, 3.0, 0.5];
        let targets = [1_usize];
        let ce_loss = |x: &[f32]| -> f32 {
            let mx = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let se: f32 = x.iter().map(|v| (v - mx).exp()).sum();
            mx + se.ln() - x[targets[0]]
        };
        let mut gi = [0.0_f32; 3];
        cross_entropy_backward(1.0, &logits, &targets, 3, &mut gi).unwrap();
        for i in 0..3 {
            let ng = numerical_grad(&ce_loss, &logits, i);
            assert!(approx(gi[i], ng, FD_TOL), "idx {i}: analytic={} numerical={ng}", gi[i]);
        }
    }

    #[test]
    fn cross_entropy_backward_batch() {
        let logits = [1.0_f32, 2.0, 3.0, 4.0];
        let targets = [0_usize, 1];
        let mut gi = [0.0_f32; 4];
        cross_entropy_backward(0.5, &logits, &targets, 2, &mut gi).unwrap();
        let sum0: f32 = gi[0] + gi[1];
        let sum1: f32 = gi[2] + gi[3];
        assert!(approx(sum0, 0.0, TOL), "row0 sum={sum0}");
        assert!(approx(sum1, 0.0, TOL), "row1 sum={sum1}");
    }

    // ── Gradient clipping ──────────────────────────────────────

    #[test]
    fn clip_norm_no_change() {
        let mut g = [1.0_f32, 0.0, 0.0];
        let norm = gradient_clip_norm(&mut g, 5.0).unwrap();
        assert!(approx(norm, 1.0, TOL));
        assert!(approx(g[0], 1.0, TOL));
    }

    #[test]
    fn clip_norm_scales_down() {
        let mut g = [3.0_f32, 4.0];
        let norm = gradient_clip_norm(&mut g, 1.0).unwrap();
        assert!(approx(norm, 5.0, TOL));
        let new_norm: f32 = g.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(approx(new_norm, 1.0, TOL), "new_norm={new_norm}");
    }

    #[test]
    fn clip_norm_preserves_direction() {
        let mut g = [3.0_f32, 4.0];
        gradient_clip_norm(&mut g, 1.0).unwrap();
        assert!(approx(g[0] / g[1], 0.75, TOL));
    }

    #[test]
    fn clip_norm_invalid_max() {
        let mut g = [1.0_f32];
        assert!(gradient_clip_norm(&mut g, 0.0).is_err());
        assert!(gradient_clip_norm(&mut g, -1.0).is_err());
    }

    #[test]
    fn clip_value_basic() {
        let mut g = [-5.0_f32, 0.5, 3.0, -0.1];
        gradient_clip_value(&mut g, 1.0).unwrap();
        assert_eq!(g, [-1.0, 0.5, 1.0, -0.1]);
    }

    #[test]
    fn clip_value_invalid_max() {
        let mut g = [1.0_f32];
        assert!(gradient_clip_value(&mut g, 0.0).is_err());
        assert!(gradient_clip_value(&mut g, -1.0).is_err());
    }

    // ── Chain rule composition ──────────────────────────────────

    #[test]
    fn chain_rule_sigmoid_then_tanh() {
        // y = tanh(sigmoid(x))
        let x_vals = [-1.0_f32, 0.0, 1.0, 2.0];
        let sig: Vec<f32> = x_vals.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
        let tanh_out: Vec<f32> = sig.iter().map(|s| s.tanh()).collect();

        let go = vec![1.0_f32; 4];
        let mut gi_tanh = vec![0.0_f32; 4];
        tanh_backward(&go, &tanh_out, &mut gi_tanh).unwrap();

        let mut gi_final = vec![0.0_f32; 4];
        sigmoid_backward(&gi_tanh, &sig, &mut gi_final).unwrap();

        let composed = |x: &[f32]| -> f32 {
            x.iter()
                .map(|v| {
                    let s = 1.0 / (1.0 + (-v).exp());
                    s.tanh()
                })
                .sum()
        };
        for i in 0..x_vals.len() {
            let ng = numerical_grad(&composed, &x_vals, i);
            assert!(
                approx(gi_final[i], ng, FD_TOL),
                "idx {i}: analytic={} numerical={ng}",
                gi_final[i]
            );
        }
    }

    #[test]
    fn chain_rule_relu_then_silu() {
        // y = silu(relu(x))
        let x_vals = [-1.0_f32, 0.5, 2.0];
        let relu_out: Vec<f32> = x_vals.iter().map(|x| x.max(0.0)).collect();

        let go = vec![1.0_f32; 3];
        let mut gi_silu = vec![0.0_f32; 3];
        silu_backward(&go, &relu_out, &mut gi_silu).unwrap();

        let mut gi_final = vec![0.0_f32; 3];
        relu_backward(&gi_silu, &x_vals, &mut gi_final).unwrap();

        let composed = |x: &[f32]| -> f32 {
            x.iter()
                .map(|v| {
                    let r = v.max(0.0);
                    r / (1.0 + (-r).exp())
                })
                .sum()
        };
        // Skip x=-1 (relu not differentiable at 0; x=-1 has grad 0).
        for i in [1, 2] {
            let ng = numerical_grad(&composed, &x_vals, i);
            assert!(
                approx(gi_final[i], ng, FD_TOL),
                "idx {i}: analytic={} numerical={ng}",
                gi_final[i]
            );
        }
    }

    // ── Error cases ────────────────────────────────────────────

    #[test]
    fn dimension_mismatch_relu() {
        let go = [1.0; 3];
        let input = [1.0; 2];
        let mut gi = [0.0; 3];
        assert!(relu_backward(&go, &input, &mut gi).is_err());
    }

    #[test]
    fn dimension_mismatch_softmax() {
        let go = [1.0; 5];
        let output = [0.2; 5];
        let mut gi = [0.0; 5];
        assert!(softmax_backward(&go, &output, 3, &mut gi).is_err());
    }

    #[test]
    fn cross_entropy_backward_empty_targets() {
        let mut gi = [0.0; 3];
        assert!(cross_entropy_backward(1.0, &[1.0, 2.0, 3.0], &[], 3, &mut gi).is_err());
    }

    #[test]
    fn cross_entropy_backward_target_out_of_range() {
        let mut gi = [0.0; 3];
        assert!(cross_entropy_backward(1.0, &[1.0, 2.0, 3.0], &[5], 3, &mut gi).is_err());
    }

    #[test]
    fn layer_norm_backward_invalid_norm_size() {
        let go = [1.0; 3];
        let input = [1.0; 3];
        let w = [1.0; 3];
        let mean = [0.0; 1];
        let rstd = [1.0; 1];
        let mut gi = [0.0; 3];
        let mut gw = [0.0; 2]; // wrong size
        let mut gb = [0.0; 3];
        assert!(
            layer_norm_backward(&go, &input, &w, &mean, &rstd, 3, &mut gi, &mut gw, &mut gb,)
                .is_err()
        );
    }

    #[test]
    fn matmul_backward_dimension_mismatch() {
        let go = [1.0; 4];
        let a = [1.0; 3]; // wrong
        let b = [1.0; 4];
        let mut ga = [0.0; 3];
        let mut gb = [0.0; 4];
        assert!(matmul_backward(&go, &a, &b, 2, 2, 2, &mut ga, &mut gb).is_err());
    }

    #[test]
    fn error_display_coverage() {
        let e1 = GradientError::DimensionMismatch { expected: 4, got: 3 };
        assert!(e1.to_string().contains("4"));
        let e2 = GradientError::InvalidGradient("test".into());
        assert!(e2.to_string().contains("test"));
        let e3 = GradientError::NumericalInstability("overflow".into());
        assert!(e3.to_string().contains("overflow"));
        // Verify Error trait is implemented.
        let _: &dyn std::error::Error = &e1;
    }

    #[test]
    fn softmax_backward_zero_classes() {
        let go = [];
        let output = [];
        let mut gi = [];
        assert!(softmax_backward(&go, &output, 0, &mut gi).is_err());
    }
}

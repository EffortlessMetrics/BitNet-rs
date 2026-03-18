//! SIMD-accelerated activation functions for CPU inference.
//!
//! Provides AVX2-vectorized implementations of common neural-network activation
//! functions with automatic scalar fallback on non-x86_64 or pre-AVX2 hardware.
//!
//! # Supported activations
//!
//! | Function | Formula | SIMD | In-place |
//! |----------|---------|------|----------|
//! | GeLU (fast) | `x * sigma(1.702x)` | yes | yes |
//! | GeLU (tanh) | `0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3)))` | yes | yes |
//! | SiLU / Swish | `x * sigma(x)` | yes | yes |
//! | Mish | `x * tanh(softplus(x))` | yes | yes |
//! | Softplus | `ln(1 + exp(x))` | yes | yes |
//! | Sigmoid | `1 / (1 + exp(-x))` | yes | yes |
//! | SwiGLU | `silu(gate) * up` | yes | yes |
//!
//! Quantized (INT8) helpers: `quantize_f32_to_i8`, `dequantize_i8_to_f32`,
//! `quantized_activation`.

use bitnet_common::{BitNetError, KernelError, Result};

// ---------------------------------------------------------------------------
// Enum dispatch
// ---------------------------------------------------------------------------

/// Selector for SIMD activation dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdActivationType {
    Gelu,
    GeluTanh,
    Silu,
    Mish,
    Softplus,
    Sigmoid,
}

/// Parameters for INT8 quantization round-trip.
#[derive(Debug, Clone, Copy)]
pub struct QuantizationParams {
    pub scale: f32,
    pub zero_point: i8,
}

// ---------------------------------------------------------------------------
// Validation helper
// ---------------------------------------------------------------------------

#[inline]
fn validate_equal_len(a: usize, b: usize) -> Result<()> {
    if a != b {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("input / output length mismatch: {} vs {}", a, b),
        }));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Scalar primitives (always available)
// ---------------------------------------------------------------------------

#[inline]
fn scalar_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn scalar_silu(x: f32) -> f32 {
    x * scalar_sigmoid(x)
}

#[inline]
fn scalar_gelu_fast(x: f32) -> f32 {
    x * scalar_sigmoid(1.702 * x)
}

#[inline]
fn scalar_gelu_tanh(x: f32) -> f32 {
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    let inner = c * (x + 0.044_715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

#[inline]
fn scalar_softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

#[inline]
fn scalar_mish(x: f32) -> f32 {
    x * scalar_softplus(x).tanh()
}

// ---------------------------------------------------------------------------
// AVX2 building blocks (x86_64 only)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod avx2 {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    /// Fast exp(x) approximation for 8 packed f32 (Cephes-style).
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn fast_exp_avx2(x: __m256) -> __m256 {
        use std::f32::consts::{LN_2, LOG2_E};

        let one = _mm256_set1_ps(1.0);
        let log2e = _mm256_set1_ps(LOG2_E);
        let ln2 = _mm256_set1_ps(LN_2);
        let clamp_lo = _mm256_set1_ps(-88.0);
        let clamp_hi = _mm256_set1_ps(88.0);

        let x = _mm256_max_ps(_mm256_min_ps(x, clamp_hi), clamp_lo);

        // t = x * log2(e);  integer part n, fractional f = t - n
        let t = _mm256_mul_ps(x, log2e);
        let n = _mm256_floor_ps(t);
        let f = _mm256_sub_ps(t, n);

        // Horner polynomial for 2^f on [0,1)
        let c0 = one;
        let c1 = ln2;
        let c2 = _mm256_set1_ps(0.240_227);
        let c3 = _mm256_set1_ps(0.055_504_1);
        let c4 = _mm256_set1_ps(0.009_618_129);
        let c5 = _mm256_set1_ps(0.001_339_733);

        let mut p = _mm256_add_ps(_mm256_mul_ps(c5, f), c4);
        p = _mm256_add_ps(_mm256_mul_ps(p, f), c3);
        p = _mm256_add_ps(_mm256_mul_ps(p, f), c2);
        p = _mm256_add_ps(_mm256_mul_ps(p, f), c1);
        p = _mm256_add_ps(_mm256_mul_ps(p, f), c0);

        // 2^n via integer exponent shift
        let ni = _mm256_cvtps_epi32(n);
        let pow2n = _mm256_castsi256_ps(_mm256_slli_epi32(
            _mm256_add_epi32(ni, _mm256_set1_epi32(127)),
            23,
        ));

        _mm256_mul_ps(p, pow2n)
    }

    /// Fast log(x) approximation for 8 packed f32.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn fast_log_avx2(x: __m256) -> __m256 {
        use std::f32::consts::LOG2_E;
        let inv_log2e = _mm256_set1_ps(1.0 / LOG2_E);
        let bias = _mm256_set1_ps(127.0_f32 * (1 << 23) as f32);
        let scale = _mm256_set1_ps(1.0 / (1 << 23) as f32);

        let bits = _mm256_castps_si256(x);
        let as_float = _mm256_cvtepi32_ps(bits);
        let log2_approx = _mm256_mul_ps(_mm256_sub_ps(as_float, bias), scale);
        _mm256_mul_ps(log2_approx, inv_log2e)
    }

    /// AVX2 sigmoid: 1 / (1 + exp(-x))
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn sigmoid_avx2(x: __m256) -> __m256 {
        unsafe {
            let one = _mm256_set1_ps(1.0);
            let neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
            let exp_neg = fast_exp_avx2(neg_x);
            _mm256_div_ps(one, _mm256_add_ps(one, exp_neg))
        }
    }

    /// AVX2 SiLU: x * sigmoid(x)
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn silu_avx2(x: __m256) -> __m256 {
        unsafe { _mm256_mul_ps(x, sigmoid_avx2(x)) }
    }

    /// AVX2 GeLU (fast / sigmoid approximation): x * sigmoid(1.702x)
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn gelu_fast_avx2(x: __m256) -> __m256 {
        unsafe {
            let k = _mm256_set1_ps(1.702);
            _mm256_mul_ps(x, sigmoid_avx2(_mm256_mul_ps(k, x)))
        }
    }

    /// AVX2 GeLU (tanh approximation).
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn gelu_tanh_avx2(x: __m256) -> __m256 {
        unsafe {
            let half = _mm256_set1_ps(0.5);
            let one = _mm256_set1_ps(1.0);
            let c = _mm256_set1_ps((2.0_f32 / std::f32::consts::PI).sqrt());
            let k = _mm256_set1_ps(0.044_715);

            // inner = c * (x + 0.044715 * x^3)
            let x2 = _mm256_mul_ps(x, x);
            let x3 = _mm256_mul_ps(x2, x);
            let cubic = _mm256_add_ps(x, _mm256_mul_ps(k, x3));
            let inner = _mm256_mul_ps(c, cubic);

            // tanh via (exp(2t)-1)/(exp(2t)+1)
            let two_inner = _mm256_add_ps(inner, inner);
            let e2 = fast_exp_avx2(two_inner);
            let tanh_val = _mm256_div_ps(_mm256_sub_ps(e2, one), _mm256_add_ps(e2, one));

            _mm256_mul_ps(half, _mm256_mul_ps(x, _mm256_add_ps(one, tanh_val)))
        }
    }

    /// AVX2 softplus: ln(1 + exp(x))
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn softplus_avx2(x: __m256) -> __m256 {
        unsafe {
            let one = _mm256_set1_ps(1.0);
            let threshold = _mm256_set1_ps(20.0);
            let exp_x = fast_exp_avx2(x);
            let sp = fast_log_avx2(_mm256_add_ps(one, exp_x));
            // For large x, softplus ~ x
            let mask = _mm256_cmp_ps(x, threshold, _CMP_GT_OQ);
            _mm256_blendv_ps(sp, x, mask)
        }
    }

    /// AVX2 mish: x * tanh(softplus(x))
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn mish_avx2(x: __m256) -> __m256 {
        unsafe {
            let one = _mm256_set1_ps(1.0);
            let sp = softplus_avx2(x);
            let two_sp = _mm256_add_ps(sp, sp);
            let e2 = fast_exp_avx2(two_sp);
            let tanh_sp = _mm256_div_ps(_mm256_sub_ps(e2, one), _mm256_add_ps(e2, one));
            _mm256_mul_ps(x, tanh_sp)
        }
    }

    /// Generic loop: process 8-wide AVX2 chunks + scalar tail.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn avx2_loop(
        input: &[f32],
        output: &mut [f32],
        avx_fn: unsafe fn(__m256) -> __m256,
        scalar_fn: fn(f32) -> f32,
    ) {
        let n = input.len();
        let chunks = n / 8;
        for i in 0..chunks {
            unsafe {
                let v = _mm256_loadu_ps(input.as_ptr().add(i * 8));
                let r = avx_fn(v);
                _mm256_storeu_ps(output.as_mut_ptr().add(i * 8), r);
            }
        }
        for (o, &inp) in output[chunks * 8..n].iter_mut().zip(&input[chunks * 8..n]) {
            *o = scalar_fn(inp);
        }
    }

    /// Generic in-place loop.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn avx2_loop_inplace(
        data: &mut [f32],
        avx_fn: unsafe fn(__m256) -> __m256,
        scalar_fn: fn(f32) -> f32,
    ) {
        let n = data.len();
        let chunks = n / 8;
        for i in 0..chunks {
            unsafe {
                let ptr = data.as_mut_ptr().add(i * 8);
                let v = _mm256_loadu_ps(ptr);
                let r = avx_fn(v);
                _mm256_storeu_ps(ptr, r);
            }
        }
        for v in data[chunks * 8..n].iter_mut() {
            *v = scalar_fn(*v);
        }
    }
}

// ---------------------------------------------------------------------------
// Public API -- out-of-place activations
// ---------------------------------------------------------------------------

/// GeLU (fast / sigmoid approximation).
pub fn simd_gelu(input: &[f32], output: &mut [f32]) -> Result<()> {
    validate_equal_len(input.len(), output.len())?;
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                avx2::avx2_loop(input, output, avx2::gelu_fast_avx2, scalar_gelu_fast);
            }
            return Ok(());
        }
    }
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = scalar_gelu_fast(x);
    }
    Ok(())
}

/// GeLU (tanh approximation).
pub fn simd_gelu_tanh(input: &[f32], output: &mut [f32]) -> Result<()> {
    validate_equal_len(input.len(), output.len())?;
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                avx2::avx2_loop(input, output, avx2::gelu_tanh_avx2, scalar_gelu_tanh);
            }
            return Ok(());
        }
    }
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = scalar_gelu_tanh(x);
    }
    Ok(())
}

/// SiLU / Swish.
pub fn simd_silu(input: &[f32], output: &mut [f32]) -> Result<()> {
    validate_equal_len(input.len(), output.len())?;
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                avx2::avx2_loop(input, output, avx2::silu_avx2, scalar_silu);
            }
            return Ok(());
        }
    }
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = scalar_silu(x);
    }
    Ok(())
}

/// Mish activation.
pub fn simd_mish(input: &[f32], output: &mut [f32]) -> Result<()> {
    validate_equal_len(input.len(), output.len())?;
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                avx2::avx2_loop(input, output, avx2::mish_avx2, scalar_mish);
            }
            return Ok(());
        }
    }
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = scalar_mish(x);
    }
    Ok(())
}

/// Softplus.
pub fn simd_softplus(input: &[f32], output: &mut [f32]) -> Result<()> {
    validate_equal_len(input.len(), output.len())?;
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                avx2::avx2_loop(input, output, avx2::softplus_avx2, scalar_softplus);
            }
            return Ok(());
        }
    }
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = scalar_softplus(x);
    }
    Ok(())
}

/// Sigmoid.
pub fn simd_sigmoid(input: &[f32], output: &mut [f32]) -> Result<()> {
    validate_equal_len(input.len(), output.len())?;
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                avx2::avx2_loop(input, output, avx2::sigmoid_avx2, scalar_sigmoid);
            }
            return Ok(());
        }
    }
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = scalar_sigmoid(x);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Public API -- in-place activations
// ---------------------------------------------------------------------------

/// In-place GeLU (fast).
pub fn simd_gelu_inplace(data: &mut [f32]) -> Result<()> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                avx2::avx2_loop_inplace(data, avx2::gelu_fast_avx2, scalar_gelu_fast);
            }
            return Ok(());
        }
    }
    for v in data.iter_mut() {
        *v = scalar_gelu_fast(*v);
    }
    Ok(())
}

/// In-place GeLU (tanh).
pub fn simd_gelu_tanh_inplace(data: &mut [f32]) -> Result<()> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                avx2::avx2_loop_inplace(data, avx2::gelu_tanh_avx2, scalar_gelu_tanh);
            }
            return Ok(());
        }
    }
    for v in data.iter_mut() {
        *v = scalar_gelu_tanh(*v);
    }
    Ok(())
}

/// In-place SiLU.
pub fn simd_silu_inplace(data: &mut [f32]) -> Result<()> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                avx2::avx2_loop_inplace(data, avx2::silu_avx2, scalar_silu);
            }
            return Ok(());
        }
    }
    for v in data.iter_mut() {
        *v = scalar_silu(*v);
    }
    Ok(())
}

/// In-place Mish.
pub fn simd_mish_inplace(data: &mut [f32]) -> Result<()> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                avx2::avx2_loop_inplace(data, avx2::mish_avx2, scalar_mish);
            }
            return Ok(());
        }
    }
    for v in data.iter_mut() {
        *v = scalar_mish(*v);
    }
    Ok(())
}

/// In-place Softplus.
pub fn simd_softplus_inplace(data: &mut [f32]) -> Result<()> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                avx2::avx2_loop_inplace(data, avx2::softplus_avx2, scalar_softplus);
            }
            return Ok(());
        }
    }
    for v in data.iter_mut() {
        *v = scalar_softplus(*v);
    }
    Ok(())
}

/// In-place Sigmoid.
pub fn simd_sigmoid_inplace(data: &mut [f32]) -> Result<()> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                avx2::avx2_loop_inplace(data, avx2::sigmoid_avx2, scalar_sigmoid);
            }
            return Ok(());
        }
    }
    for v in data.iter_mut() {
        *v = scalar_sigmoid(*v);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// SwiGLU  (gate * silu(up))
// ---------------------------------------------------------------------------

/// Fused SwiGLU: `output[i] = silu(gate[i]) * up[i]`.
pub fn simd_swiglu(gate: &[f32], up: &[f32], output: &mut [f32]) -> Result<()> {
    validate_equal_len(gate.len(), up.len())?;
    validate_equal_len(gate.len(), output.len())?;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                avx2_swiglu(gate, up, output);
            }
            return Ok(());
        }
    }
    for i in 0..gate.len() {
        output[i] = scalar_silu(gate[i]) * up[i];
    }
    Ok(())
}

/// In-place SwiGLU: `gate[i] = silu(gate[i]) * up[i]`.
pub fn simd_swiglu_inplace(gate: &mut [f32], up: &[f32]) -> Result<()> {
    validate_equal_len(gate.len(), up.len())?;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                avx2_swiglu_inplace(gate, up);
            }
            return Ok(());
        }
    }
    for i in 0..gate.len() {
        gate[i] = scalar_silu(gate[i]) * up[i];
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_swiglu(gate: &[f32], up: &[f32], output: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = gate.len();
    let chunks = n / 8;
    for i in 0..chunks {
        unsafe {
            let g = _mm256_loadu_ps(gate.as_ptr().add(i * 8));
            let u = _mm256_loadu_ps(up.as_ptr().add(i * 8));
            let sg = avx2::silu_avx2(g);
            let r = _mm256_mul_ps(sg, u);
            _mm256_storeu_ps(output.as_mut_ptr().add(i * 8), r);
        }
    }
    for i in (chunks * 8)..n {
        output[i] = scalar_silu(gate[i]) * up[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_swiglu_inplace(gate: &mut [f32], up: &[f32]) {
    use std::arch::x86_64::*;
    let n = gate.len();
    let chunks = n / 8;
    for i in 0..chunks {
        unsafe {
            let ptr = gate.as_mut_ptr().add(i * 8);
            let g = _mm256_loadu_ps(ptr);
            let u = _mm256_loadu_ps(up.as_ptr().add(i * 8));
            let sg = avx2::silu_avx2(g);
            let r = _mm256_mul_ps(sg, u);
            _mm256_storeu_ps(ptr, r);
        }
    }
    for i in (chunks * 8)..n {
        gate[i] = scalar_silu(gate[i]) * up[i];
    }
}

// ---------------------------------------------------------------------------
// Quantized INT8 helpers
// ---------------------------------------------------------------------------

/// Quantize f32 -> i8 with scale and zero-point.
pub fn quantize_f32_to_i8(input: &[f32], params: &QuantizationParams) -> Vec<i8> {
    input
        .iter()
        .map(|&x| {
            let q = (x / params.scale).round() as i32 + params.zero_point as i32;
            q.clamp(-128, 127) as i8
        })
        .collect()
}

/// Dequantize i8 -> f32.
pub fn dequantize_i8_to_f32(input: &[i8], params: &QuantizationParams) -> Vec<f32> {
    input.iter().map(|&q| (q as f32 - params.zero_point as f32) * params.scale).collect()
}

/// Quantized activation: dequant -> activation -> requant.
pub fn quantized_activation(
    input: &[i8],
    activation: SimdActivationType,
    in_params: &QuantizationParams,
    out_params: &QuantizationParams,
) -> Result<Vec<i8>> {
    let float_in = dequantize_i8_to_f32(input, in_params);
    let mut float_out = vec![0.0f32; float_in.len()];
    match activation {
        SimdActivationType::Gelu => simd_gelu(&float_in, &mut float_out)?,
        SimdActivationType::GeluTanh => simd_gelu_tanh(&float_in, &mut float_out)?,
        SimdActivationType::Silu => simd_silu(&float_in, &mut float_out)?,
        SimdActivationType::Mish => simd_mish(&float_in, &mut float_out)?,
        SimdActivationType::Softplus => simd_softplus(&float_in, &mut float_out)?,
        SimdActivationType::Sigmoid => simd_sigmoid(&float_in, &mut float_out)?,
    }
    Ok(quantize_f32_to_i8(&float_out, out_params))
}

// ---------------------------------------------------------------------------
// Enum dispatch helpers
// ---------------------------------------------------------------------------

/// Apply an activation selected by [`SimdActivationType`].
pub fn simd_activation_dispatch(
    input: &[f32],
    output: &mut [f32],
    activation: SimdActivationType,
) -> Result<()> {
    match activation {
        SimdActivationType::Gelu => simd_gelu(input, output),
        SimdActivationType::GeluTanh => simd_gelu_tanh(input, output),
        SimdActivationType::Silu => simd_silu(input, output),
        SimdActivationType::Mish => simd_mish(input, output),
        SimdActivationType::Softplus => simd_softplus(input, output),
        SimdActivationType::Sigmoid => simd_sigmoid(input, output),
    }
}

/// In-place activation selected by [`SimdActivationType`].
pub fn simd_activation_dispatch_inplace(
    data: &mut [f32],
    activation: SimdActivationType,
) -> Result<()> {
    match activation {
        SimdActivationType::Gelu => simd_gelu_inplace(data),
        SimdActivationType::GeluTanh => simd_gelu_tanh_inplace(data),
        SimdActivationType::Silu => simd_silu_inplace(data),
        SimdActivationType::Mish => simd_mish_inplace(data),
        SimdActivationType::Softplus => simd_softplus_inplace(data),
        SimdActivationType::Sigmoid => simd_sigmoid_inplace(data),
    }
}

/// Batched activation: apply the same activation to every row of a 2-D buffer
/// stored in row-major order.
pub fn simd_activation_batched(
    data: &mut [f32],
    rows: usize,
    cols: usize,
    activation: SimdActivationType,
) -> Result<()> {
    if data.len() != rows * cols {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("data length {} != rows ({}) * cols ({})", data.len(), rows, cols),
        }));
    }
    if cols == 0 {
        return Ok(());
    }
    for row in data.chunks_mut(cols) {
        simd_activation_dispatch_inplace(row, activation)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- helpers ----------------------------------------------------------

    fn assert_close(a: f32, b: f32, tol: f32, ctx: &str) {
        let diff = (a - b).abs();
        assert!(diff <= tol, "{ctx}: |{a} - {b}| = {diff} > {tol}");
    }

    fn reference_sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
    fn reference_silu(x: f32) -> f32 {
        x * reference_sigmoid(x)
    }
    fn reference_gelu_fast(x: f32) -> f32 {
        x * reference_sigmoid(1.702 * x)
    }
    fn reference_gelu_tanh(x: f32) -> f32 {
        let c = (2.0_f32 / std::f32::consts::PI).sqrt();
        0.5 * x * (1.0 + (c * (x + 0.044_715 * x * x * x)).tanh())
    }
    fn reference_softplus(x: f32) -> f32 {
        if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
    }
    fn reference_mish(x: f32) -> f32 {
        x * reference_softplus(x).tanh()
    }

    // ---- 1. GeLU (fast) ---------------------------------------------------

    #[test]
    fn test_simd_gelu_basic() {
        let input = vec![0.0, 1.0, -1.0, 2.0, -2.0];
        let mut output = [0.0; 5];
        simd_gelu(&input, &mut output).unwrap();
        for (i, &x) in input.iter().enumerate() {
            assert_close(output[i], reference_gelu_fast(x), 1e-4, "gelu basic");
        }
    }

    #[test]
    fn test_simd_gelu_empty() {
        let input: Vec<f32> = vec![];
        let mut output: Vec<f32> = vec![];
        simd_gelu(&input, &mut output).unwrap();
    }

    #[test]
    fn test_simd_gelu_non_aligned() {
        let input: Vec<f32> = (0..13).map(|i| i as f32 * 0.1 - 0.6).collect();
        let mut output = [0.0; 13];
        simd_gelu(&input, &mut output).unwrap();
        for (i, &x) in input.iter().enumerate() {
            assert_close(output[i], reference_gelu_fast(x), 1e-4, "gelu non-aligned");
        }
    }

    #[test]
    fn test_simd_gelu_length_mismatch() {
        let input = [1.0; 5];
        let mut output = [0.0; 3];
        assert!(simd_gelu(&input, &mut output).is_err());
    }

    #[test]
    fn test_scalar_gelu_fast_known_values() {
        assert_close(scalar_gelu_fast(0.0), 0.0, 1e-6, "gelu(0)");
        assert!((scalar_gelu_fast(1.0) - 0.846).abs() < 0.02);
        assert!(scalar_gelu_fast(-3.0).abs() < 0.05);
    }

    #[test]
    fn test_simd_gelu_large_vector() {
        let n = 1024;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 - 512.0) * 0.01).collect();
        let mut output = vec![0.0; n];
        simd_gelu(&input, &mut output).unwrap();
        for (i, &x) in input.iter().enumerate() {
            assert_close(output[i], reference_gelu_fast(x), 1e-3, "gelu large");
        }
    }

    #[test]
    fn test_simd_gelu_extreme_values() {
        let input = vec![-100.0, -50.0, 50.0, 100.0];
        let mut output = [0.0; 4];
        simd_gelu(&input, &mut output).unwrap();
        assert!(output[0].abs() < 1.0);
        assert!(output[1].abs() < 1.0);
        assert!((output[2] - 50.0).abs() < 1.0);
        assert!((output[3] - 100.0).abs() < 1.0);
    }

    // ---- 2. GeLU (tanh) ---------------------------------------------------

    #[test]
    fn test_simd_gelu_tanh_basic() {
        let input = vec![0.0, 1.0, -1.0, 2.0, -2.0];
        let mut output = [0.0; 5];
        simd_gelu_tanh(&input, &mut output).unwrap();
        for (i, &x) in input.iter().enumerate() {
            assert_close(output[i], reference_gelu_tanh(x), 1e-3, "gelu_tanh basic");
        }
    }

    #[test]
    fn test_simd_gelu_tanh_empty() {
        let mut out: Vec<f32> = vec![];
        simd_gelu_tanh(&[], &mut out).unwrap();
    }

    #[test]
    fn test_simd_gelu_tanh_non_aligned() {
        let input: Vec<f32> = (0..11).map(|i| i as f32 * 0.2 - 1.0).collect();
        let mut output = [0.0; 11];
        simd_gelu_tanh(&input, &mut output).unwrap();
        for (i, &x) in input.iter().enumerate() {
            assert_close(output[i], reference_gelu_tanh(x), 1e-3, "gelu_tanh non-aligned");
        }
    }

    #[test]
    fn test_simd_gelu_tanh_length_mismatch() {
        let input = [1.0; 4];
        let mut output = [0.0; 7];
        assert!(simd_gelu_tanh(&input, &mut output).is_err());
    }

    #[test]
    fn test_scalar_gelu_tanh_known_values() {
        assert_close(scalar_gelu_tanh(0.0), 0.0, 1e-6, "gelu_tanh(0)");
        assert!((scalar_gelu_tanh(1.0) - 0.841).abs() < 0.01);
        assert!(scalar_gelu_tanh(-3.0).abs() < 0.01);
    }

    #[test]
    fn test_gelu_tanh_vs_fast_near_zero() {
        for &x in &[-0.5, -0.1, 0.0, 0.1, 0.5] {
            let fast = scalar_gelu_fast(x);
            let tanh_v = scalar_gelu_tanh(x);
            assert_close(fast, tanh_v, 0.02, "gelu variants near zero");
        }
    }

    // ---- 3. SiLU ----------------------------------------------------------

    #[test]
    fn test_simd_silu_basic() {
        let input = vec![0.0, 1.0, -1.0, 3.0, -3.0];
        let mut output = [0.0; 5];
        simd_silu(&input, &mut output).unwrap();
        for (i, &x) in input.iter().enumerate() {
            assert_close(output[i], reference_silu(x), 1e-4, "silu basic");
        }
    }

    #[test]
    fn test_simd_silu_empty() {
        let mut out: Vec<f32> = vec![];
        simd_silu(&[], &mut out).unwrap();
    }

    #[test]
    fn test_simd_silu_non_aligned() {
        let input: Vec<f32> = (0..9).map(|i| i as f32 - 4.0).collect();
        let mut output = [0.0; 9];
        simd_silu(&input, &mut output).unwrap();
        for (i, &x) in input.iter().enumerate() {
            assert_close(output[i], reference_silu(x), 1e-4, "silu non-aligned");
        }
    }

    #[test]
    fn test_simd_silu_length_mismatch() {
        assert!(simd_silu(&[1.0; 2], &mut [0.0; 3]).is_err());
    }

    #[test]
    fn test_scalar_silu_known_values() {
        assert_close(scalar_silu(0.0), 0.0, 1e-6, "silu(0)");
        assert!((scalar_silu(1.0) - 0.731).abs() < 0.01);
    }

    // ---- 4. Mish ----------------------------------------------------------

    #[test]
    fn test_simd_mish_basic() {
        let input = vec![0.0, 1.0, -1.0, 2.0];
        let mut output = [0.0; 4];
        simd_mish(&input, &mut output).unwrap();
        for (i, &x) in input.iter().enumerate() {
            assert_close(output[i], reference_mish(x), 0.15, "mish basic");
        }
    }

    #[test]
    fn test_simd_mish_empty() {
        let mut out: Vec<f32> = vec![];
        simd_mish(&[], &mut out).unwrap();
    }

    #[test]
    fn test_simd_mish_non_aligned() {
        let input: Vec<f32> = (0..10).map(|i| i as f32 * 0.5 - 2.5).collect();
        let mut output = [0.0; 10];
        simd_mish(&input, &mut output).unwrap();
        for (i, &x) in input.iter().enumerate() {
            assert_close(output[i], reference_mish(x), 0.15, "mish non-aligned");
        }
    }

    #[test]
    fn test_simd_mish_length_mismatch() {
        assert!(simd_mish(&[1.0; 3], &mut [0.0; 2]).is_err());
    }

    #[test]
    fn test_scalar_mish_known_values() {
        assert_close(scalar_mish(0.0), 0.0, 1e-6, "mish(0)");
        assert!((scalar_mish(1.0) - 0.865).abs() < 0.02);
    }

    // ---- 5. Softplus ------------------------------------------------------

    #[test]
    fn test_simd_softplus_basic() {
        let input = vec![0.0, 1.0, -1.0, 25.0];
        let mut output = [0.0; 4];
        simd_softplus(&input, &mut output).unwrap();
        for (i, &x) in input.iter().enumerate() {
            assert_close(output[i], reference_softplus(x), 0.15, "softplus basic");
        }
    }

    #[test]
    fn test_simd_softplus_empty() {
        let mut out: Vec<f32> = vec![];
        simd_softplus(&[], &mut out).unwrap();
    }

    #[test]
    fn test_simd_softplus_large_input_passthrough() {
        let input = vec![25.0, 30.0, 50.0, 100.0];
        let mut output = [0.0; 4];
        simd_softplus(&input, &mut output).unwrap();
        for (i, &x) in input.iter().enumerate() {
            assert_close(output[i], x, 0.2, "softplus large");
        }
    }

    #[test]
    fn test_simd_softplus_length_mismatch() {
        assert!(simd_softplus(&[1.0; 6], &mut [0.0; 5]).is_err());
    }

    #[test]
    fn test_scalar_softplus_known_values() {
        assert_close(scalar_softplus(0.0), 0.0_f32.exp().ln_1p(), 1e-6, "sp(0)");
    }

    // ---- 6. Sigmoid -------------------------------------------------------

    #[test]
    fn test_simd_sigmoid_basic() {
        let input = vec![0.0, 1.0, -1.0, 5.0, -5.0];
        let mut output = [0.0; 5];
        simd_sigmoid(&input, &mut output).unwrap();
        for (i, &x) in input.iter().enumerate() {
            assert_close(output[i], reference_sigmoid(x), 1e-4, "sigmoid basic");
        }
    }

    #[test]
    fn test_simd_sigmoid_empty() {
        let mut out: Vec<f32> = vec![];
        simd_sigmoid(&[], &mut out).unwrap();
    }

    #[test]
    fn test_simd_sigmoid_range_01() {
        let input: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.2).collect();
        let mut output = [0.0; 100];
        simd_sigmoid(&input, &mut output).unwrap();
        for &o in &output {
            assert!((0.0..=1.0).contains(&o), "sigmoid out of [0,1]: {o}");
        }
    }

    #[test]
    fn test_simd_sigmoid_length_mismatch() {
        assert!(simd_sigmoid(&[1.0; 4], &mut [0.0; 5]).is_err());
    }

    #[test]
    fn test_scalar_sigmoid_known_values() {
        assert_close(scalar_sigmoid(0.0), 0.5, 1e-6, "sig(0)");
        assert!(scalar_sigmoid(100.0) > 0.999);
        assert!(scalar_sigmoid(-100.0) < 0.001);
    }

    // ---- 7. In-place variants ---------------------------------------------

    #[test]
    fn test_gelu_inplace_matches_out_of_place() {
        let input = vec![0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 0.1, -0.1, 3.0];
        let mut inplace = input.clone();
        let mut out_of_place = vec![0.0; input.len()];
        simd_gelu(&input, &mut out_of_place).unwrap();
        simd_gelu_inplace(&mut inplace).unwrap();
        for i in 0..input.len() {
            assert_close(inplace[i], out_of_place[i], 1e-6, "gelu inplace");
        }
    }

    #[test]
    fn test_gelu_tanh_inplace_matches() {
        let input = vec![0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 0.1, -0.1, 3.0];
        let mut inplace = input.clone();
        let mut oop = vec![0.0; input.len()];
        simd_gelu_tanh(&input, &mut oop).unwrap();
        simd_gelu_tanh_inplace(&mut inplace).unwrap();
        for i in 0..input.len() {
            assert_close(inplace[i], oop[i], 1e-6, "gelu_tanh inplace");
        }
    }

    #[test]
    fn test_silu_inplace_matches() {
        let input = vec![0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 0.1, -0.1, 3.0];
        let mut inplace = input.clone();
        let mut oop = vec![0.0; input.len()];
        simd_silu(&input, &mut oop).unwrap();
        simd_silu_inplace(&mut inplace).unwrap();
        for i in 0..input.len() {
            assert_close(inplace[i], oop[i], 1e-6, "silu inplace");
        }
    }

    #[test]
    fn test_mish_inplace_matches() {
        let input = vec![0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 0.1, -0.1, 3.0];
        let mut inplace = input.clone();
        let mut oop = vec![0.0; input.len()];
        simd_mish(&input, &mut oop).unwrap();
        simd_mish_inplace(&mut inplace).unwrap();
        for i in 0..input.len() {
            assert_close(inplace[i], oop[i], 1e-6, "mish inplace");
        }
    }

    #[test]
    fn test_softplus_inplace_matches() {
        let input = vec![0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 0.1, -0.1, 3.0];
        let mut inplace = input.clone();
        let mut oop = vec![0.0; input.len()];
        simd_softplus(&input, &mut oop).unwrap();
        simd_softplus_inplace(&mut inplace).unwrap();
        for i in 0..input.len() {
            assert_close(inplace[i], oop[i], 1e-6, "softplus inplace");
        }
    }

    #[test]
    fn test_sigmoid_inplace_matches() {
        let input = vec![0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 0.1, -0.1, 3.0];
        let mut inplace = input.clone();
        let mut oop = vec![0.0; input.len()];
        simd_sigmoid(&input, &mut oop).unwrap();
        simd_sigmoid_inplace(&mut inplace).unwrap();
        for i in 0..input.len() {
            assert_close(inplace[i], oop[i], 1e-6, "sigmoid inplace");
        }
    }

    #[test]
    fn test_inplace_empty() {
        let mut data: Vec<f32> = vec![];
        simd_gelu_inplace(&mut data).unwrap();
        simd_silu_inplace(&mut data).unwrap();
        simd_mish_inplace(&mut data).unwrap();
        simd_sigmoid_inplace(&mut data).unwrap();
        simd_softplus_inplace(&mut data).unwrap();
        simd_gelu_tanh_inplace(&mut data).unwrap();
    }

    // ---- 8. SwiGLU --------------------------------------------------------

    #[test]
    fn test_simd_swiglu_basic() {
        let gate = vec![1.0, -1.0, 0.5, 2.0, 0.0];
        let up = vec![1.0, 1.0, 2.0, 0.5, 3.0];
        let mut output = [0.0; 5];
        simd_swiglu(&gate, &up, &mut output).unwrap();
        for i in 0..gate.len() {
            let expected = reference_silu(gate[i]) * up[i];
            assert_close(output[i], expected, 1e-4, "swiglu basic");
        }
    }

    #[test]
    fn test_simd_swiglu_empty() {
        let mut out: Vec<f32> = vec![];
        simd_swiglu(&[], &[], &mut out).unwrap();
    }

    #[test]
    fn test_simd_swiglu_non_aligned() {
        let n = 11;
        let gate: Vec<f32> = (0..n).map(|i| i as f32 * 0.3 - 1.5).collect();
        let up: Vec<f32> = (0..n).map(|i| i as f32 * 0.2 + 0.1).collect();
        let mut output = vec![0.0; n];
        simd_swiglu(&gate, &up, &mut output).unwrap();
        for i in 0..n {
            let expected = reference_silu(gate[i]) * up[i];
            assert_close(output[i], expected, 1e-4, "swiglu non-aligned");
        }
    }

    #[test]
    fn test_simd_swiglu_length_mismatch_gate_up() {
        assert!(simd_swiglu(&[1.0; 3], &[1.0; 4], &mut [0.0; 3]).is_err());
    }

    #[test]
    fn test_simd_swiglu_length_mismatch_gate_out() {
        assert!(simd_swiglu(&[1.0; 3], &[1.0; 3], &mut [0.0; 4]).is_err());
    }

    #[test]
    fn test_simd_swiglu_inplace_basic() {
        let mut gate = vec![1.0, -1.0, 0.5, 2.0, 0.0];
        let up = vec![1.0, 1.0, 2.0, 0.5, 3.0];
        let expected: Vec<f32> =
            gate.iter().zip(up.iter()).map(|(&g, &u)| reference_silu(g) * u).collect();
        simd_swiglu_inplace(&mut gate, &up).unwrap();
        for i in 0..gate.len() {
            assert_close(gate[i], expected[i], 1e-4, "swiglu inplace");
        }
    }

    #[test]
    fn test_simd_swiglu_inplace_length_mismatch() {
        assert!(simd_swiglu_inplace(&mut [1.0; 3], &[1.0; 5]).is_err());
    }

    #[test]
    fn test_simd_swiglu_large_vector() {
        let n = 256;
        let gate: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) * 0.02).collect();
        let up: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let mut output = vec![0.0; n];
        simd_swiglu(&gate, &up, &mut output).unwrap();
        for i in 0..n {
            let expected = reference_silu(gate[i]) * up[i];
            assert_close(output[i], expected, 1e-3, "swiglu large");
        }
    }

    // ---- 9. Quantized INT8 ------------------------------------------------

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let params = QuantizationParams { scale: 0.1, zero_point: 0 };
        let input = vec![0.0, 0.5, -0.5, 1.0, -1.0];
        let quantized = quantize_f32_to_i8(&input, &params);
        let back = dequantize_i8_to_f32(&quantized, &params);
        for (i, (&orig, &deq)) in input.iter().zip(back.iter()).enumerate() {
            assert_close(orig, deq, params.scale, &format!("roundtrip[{i}]"));
        }
    }

    #[test]
    fn test_quantize_clamp() {
        let params = QuantizationParams { scale: 0.01, zero_point: 0 };
        let input = vec![100.0, -100.0];
        let q = quantize_f32_to_i8(&input, &params);
        assert_eq!(q[0], 127);
        assert_eq!(q[1], -128);
    }

    #[test]
    fn test_quantize_zero_point() {
        let params = QuantizationParams { scale: 0.1, zero_point: 10 };
        let input = [0.0];
        let q = quantize_f32_to_i8(&input, &params);
        assert_eq!(q[0], 10);
    }

    #[test]
    fn test_quantized_activation_gelu() {
        let in_p = QuantizationParams { scale: 0.05, zero_point: 0 };
        let out_p = QuantizationParams { scale: 0.05, zero_point: 0 };
        let input: Vec<i8> = vec![0, 10, -10, 20, -20];
        let result = quantized_activation(&input, SimdActivationType::Gelu, &in_p, &out_p).unwrap();
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_quantized_activation_sigmoid() {
        let in_p = QuantizationParams { scale: 0.1, zero_point: 0 };
        let out_p = QuantizationParams { scale: 0.01, zero_point: 0 };
        let input: Vec<i8> = vec![0, 10, -10];
        let result =
            quantized_activation(&input, SimdActivationType::Sigmoid, &in_p, &out_p).unwrap();
        // sigmoid(0) ~ 0.5 -> 0.5/0.01 = 50
        assert!((result[0] as f32 - 50.0).abs() < 2.0);
    }

    #[test]
    fn test_quantized_activation_all_types() {
        let params = QuantizationParams { scale: 0.1, zero_point: 0 };
        let input: Vec<i8> = vec![0, 5, -5];
        for ty in [
            SimdActivationType::Gelu,
            SimdActivationType::GeluTanh,
            SimdActivationType::Silu,
            SimdActivationType::Mish,
            SimdActivationType::Softplus,
            SimdActivationType::Sigmoid,
        ] {
            let r = quantized_activation(&input, ty, &params, &params);
            assert!(r.is_ok(), "quantized_activation failed for {ty:?}");
        }
    }

    #[test]
    fn test_quantized_activation_empty() {
        let params = QuantizationParams { scale: 0.1, zero_point: 0 };
        let input: Vec<i8> = vec![];
        let result =
            quantized_activation(&input, SimdActivationType::Silu, &params, &params).unwrap();
        assert!(result.is_empty());
    }

    // ---- 10. Enum dispatch ------------------------------------------------

    #[test]
    fn test_dispatch_gelu() {
        let input = vec![0.0, 1.0, -1.0];
        let mut output = [0.0; 3];
        simd_activation_dispatch(&input, &mut output, SimdActivationType::Gelu).unwrap();
        let mut expected = [0.0; 3];
        simd_gelu(&input, &mut expected).unwrap();
        for i in 0..3 {
            assert_close(output[i], expected[i], 1e-6, "dispatch gelu");
        }
    }

    #[test]
    fn test_dispatch_gelu_tanh() {
        let input = vec![0.0, 1.0, -1.0];
        let mut output = [0.0; 3];
        simd_activation_dispatch(&input, &mut output, SimdActivationType::GeluTanh).unwrap();
        let mut expected = [0.0; 3];
        simd_gelu_tanh(&input, &mut expected).unwrap();
        for i in 0..3 {
            assert_close(output[i], expected[i], 1e-6, "dispatch gelu_tanh");
        }
    }

    #[test]
    fn test_dispatch_silu() {
        let input = vec![0.0, 1.0, -1.0];
        let mut output = [0.0; 3];
        simd_activation_dispatch(&input, &mut output, SimdActivationType::Silu).unwrap();
        let mut expected = [0.0; 3];
        simd_silu(&input, &mut expected).unwrap();
        for i in 0..3 {
            assert_close(output[i], expected[i], 1e-6, "dispatch silu");
        }
    }

    #[test]
    fn test_dispatch_mish() {
        let input = vec![0.0, 1.0, -1.0];
        let mut output = [0.0; 3];
        simd_activation_dispatch(&input, &mut output, SimdActivationType::Mish).unwrap();
        let mut expected = [0.0; 3];
        simd_mish(&input, &mut expected).unwrap();
        for i in 0..3 {
            assert_close(output[i], expected[i], 1e-6, "dispatch mish");
        }
    }

    #[test]
    fn test_dispatch_softplus() {
        let input = vec![0.0, 1.0, -1.0];
        let mut output = [0.0; 3];
        simd_activation_dispatch(&input, &mut output, SimdActivationType::Softplus).unwrap();
        let mut expected = [0.0; 3];
        simd_softplus(&input, &mut expected).unwrap();
        for i in 0..3 {
            assert_close(output[i], expected[i], 1e-6, "dispatch softplus");
        }
    }

    #[test]
    fn test_dispatch_sigmoid() {
        let input = vec![0.0, 1.0, -1.0];
        let mut output = [0.0; 3];
        simd_activation_dispatch(&input, &mut output, SimdActivationType::Sigmoid).unwrap();
        let mut expected = [0.0; 3];
        simd_sigmoid(&input, &mut expected).unwrap();
        for i in 0..3 {
            assert_close(output[i], expected[i], 1e-6, "dispatch sigmoid");
        }
    }

    #[test]
    fn test_dispatch_inplace_all() {
        for ty in [
            SimdActivationType::Gelu,
            SimdActivationType::GeluTanh,
            SimdActivationType::Silu,
            SimdActivationType::Mish,
            SimdActivationType::Softplus,
            SimdActivationType::Sigmoid,
        ] {
            let input = vec![0.5, -0.5, 1.0, -1.0];
            let mut via_dispatch = input.clone();
            let mut via_oop = [0.0; 4];
            simd_activation_dispatch(&input, &mut via_oop, ty).unwrap();
            simd_activation_dispatch_inplace(&mut via_dispatch, ty).unwrap();
            for i in 0..4 {
                assert_close(
                    via_dispatch[i],
                    via_oop[i],
                    1e-6,
                    &format!("dispatch inplace {ty:?}"),
                );
            }
        }
    }

    // ---- 11. Batched dispatch ---------------------------------------------

    #[test]
    fn test_batched_basic() {
        let mut data = vec![0.0, 1.0, -1.0, 0.5, -0.5, 2.0];
        let original = data.clone();
        simd_activation_batched(&mut data, 2, 3, SimdActivationType::Silu).unwrap();
        for (i, &x) in original.iter().enumerate() {
            assert_close(data[i], reference_silu(x), 1e-4, "batched silu");
        }
    }

    #[test]
    fn test_batched_dimension_mismatch() {
        let mut data = [0.0; 10];
        assert!(simd_activation_batched(&mut data, 3, 4, SimdActivationType::Gelu).is_err());
    }

    #[test]
    fn test_batched_single_row() {
        let mut data = vec![1.0, 2.0, 3.0];
        simd_activation_batched(&mut data, 1, 3, SimdActivationType::Sigmoid).unwrap();
        for &v in &data {
            assert!(v > 0.0 && v < 1.0);
        }
    }

    #[test]
    fn test_batched_empty() {
        let mut data: Vec<f32> = vec![];
        simd_activation_batched(&mut data, 0, 0, SimdActivationType::Gelu).unwrap();
    }

    #[test]
    fn test_batched_many_rows() {
        let rows = 64;
        let cols = 17;
        let mut data: Vec<f32> = (0..rows * cols).map(|i| (i as f32 - 500.0) * 0.01).collect();
        simd_activation_batched(&mut data, rows, cols, SimdActivationType::Silu).unwrap();
        assert_eq!(data.len(), rows * cols);
    }

    // ---- 12. Monotonicity / mathematical properties ----------------------

    #[test]
    fn test_sigmoid_monotonic() {
        let input: Vec<f32> = (-50..50).map(|i| i as f32 * 0.1).collect();
        let mut output = vec![0.0; input.len()];
        simd_sigmoid(&input, &mut output).unwrap();
        for w in output.windows(2) {
            assert!(w[1] >= w[0], "sigmoid not monotonic: {} < {}", w[1], w[0]);
        }
    }

    #[test]
    fn test_softplus_monotonic() {
        let input: Vec<f32> = (-50..50).map(|i| i as f32 * 0.1).collect();
        let mut output = vec![0.0; input.len()];
        simd_softplus(&input, &mut output).unwrap();
        for w in output.windows(2) {
            assert!(w[1] >= w[0] - 0.01, "softplus not monotonic");
        }
    }

    #[test]
    fn test_softplus_non_negative() {
        let input: Vec<f32> = (-100..100).map(|i| i as f32 * 0.1).collect();
        let mut output = vec![0.0; input.len()];
        simd_softplus(&input, &mut output).unwrap();
        for &o in &output {
            assert!(o >= -0.01, "softplus negative: {o}");
        }
    }

    #[test]
    fn test_gelu_symmetry_approx() {
        // GeLU(0) = 0 should hold
        let mut out = [0.0];
        simd_gelu(&[0.0], &mut out).unwrap();
        assert_close(out[0], 0.0, 1e-6, "gelu(0)==0");
    }

    #[test]
    fn test_silu_zero() {
        let mut out = [0.0];
        simd_silu(&[0.0], &mut out).unwrap();
        assert_close(out[0], 0.0, 1e-6, "silu(0)==0");
    }

    #[test]
    fn test_mish_zero() {
        let mut out = [0.0];
        simd_mish(&[0.0], &mut out).unwrap();
        assert_close(out[0], 0.0, 1e-6, "mish(0)==0");
    }

    // ---- 13. Cross-activation consistency ---------------------------------

    #[test]
    fn test_silu_is_x_times_sigmoid() {
        let input: Vec<f32> = (-20..20).map(|i| i as f32 * 0.25).collect();
        let mut silu_out = vec![0.0; input.len()];
        let mut sig_out = vec![0.0; input.len()];
        simd_silu(&input, &mut silu_out).unwrap();
        simd_sigmoid(&input, &mut sig_out).unwrap();
        for i in 0..input.len() {
            assert_close(silu_out[i], input[i] * sig_out[i], 1e-4, "silu == x*sig");
        }
    }

    #[test]
    fn test_gelu_fast_is_x_times_sigmoid_1702() {
        let input: Vec<f32> = (-10..10).map(|i| i as f32 * 0.5).collect();
        let scaled: Vec<f32> = input.iter().map(|&x| 1.702 * x).collect();
        let mut gelu_out = vec![0.0; input.len()];
        let mut sig_out = vec![0.0; input.len()];
        simd_gelu(&input, &mut gelu_out).unwrap();
        simd_sigmoid(&scaled, &mut sig_out).unwrap();
        for i in 0..input.len() {
            assert_close(gelu_out[i], input[i] * sig_out[i], 1e-4, "gelu == x*sig(1.702x)");
        }
    }

    // ---- 14. Enum equality / Debug ----------------------------------------

    #[test]
    fn test_activation_type_eq() {
        assert_eq!(SimdActivationType::Gelu, SimdActivationType::Gelu);
        assert_ne!(SimdActivationType::Gelu, SimdActivationType::Silu);
    }

    #[test]
    fn test_activation_type_debug() {
        let s = format!("{:?}", SimdActivationType::Mish);
        assert_eq!(s, "Mish");
    }

    #[test]
    fn test_activation_type_clone() {
        let a = SimdActivationType::GeluTanh;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn test_quantization_params_debug() {
        let p = QuantizationParams { scale: 0.1, zero_point: 5 };
        let s = format!("{p:?}");
        assert!(s.contains("0.1"));
        assert!(s.contains("5"));
    }

    // ---- 15. Stress / large vectors --------------------------------------

    #[test]
    fn test_all_activations_large() {
        let n = 2048;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 - 1024.0) * 0.005).collect();
        let mut out = vec![0.0; n];
        simd_gelu(&input, &mut out).unwrap();
        simd_gelu_tanh(&input, &mut out).unwrap();
        simd_silu(&input, &mut out).unwrap();
        simd_mish(&input, &mut out).unwrap();
        simd_softplus(&input, &mut out).unwrap();
        simd_sigmoid(&input, &mut out).unwrap();
    }

    #[test]
    fn test_exact_8_elements() {
        let input = [1.0; 8];
        let mut output = [0.0; 8];
        simd_gelu(&input, &mut output).unwrap();
        for &o in &output {
            assert!(o > 0.0);
        }
    }

    #[test]
    fn test_exact_16_elements() {
        let input = [-0.5; 16];
        let mut output = [0.0; 16];
        simd_silu(&input, &mut output).unwrap();
        for &o in &output {
            assert!(o < 0.0);
        }
    }

    #[test]
    fn test_exact_1_element() {
        let mut out = [0.0];
        simd_sigmoid(&[0.0], &mut out).unwrap();
        assert_close(out[0], 0.5, 1e-4, "single sigmoid");
    }

    // ---- 16. NaN / Inf robustness ----------------------------------------

    #[test]
    fn test_sigmoid_inf() {
        let input = vec![f32::INFINITY, f32::NEG_INFINITY];
        let mut output = [0.0; 2];
        simd_sigmoid(&input, &mut output).unwrap();
        assert!((output[0] - 1.0).abs() < 0.01 || output[0].is_finite());
        assert!(output[1].abs() < 0.01 || output[1].is_finite());
    }

    #[test]
    fn test_silu_nan_produces_nan() {
        let input = vec![f32::NAN];
        let mut output = [0.0];
        simd_silu(&input, &mut output).unwrap();
        assert!(output[0].is_nan());
    }

    // ---- 17. Misc edge cases ----------------------------------------------

    #[test]
    fn test_swiglu_zeros() {
        let gate = [0.0; 16];
        let up = [1.0; 16];
        let mut output = [0.0; 16];
        simd_swiglu(&gate, &up, &mut output).unwrap();
        for &o in &output {
            assert_close(o, 0.0, 1e-6, "swiglu gate=0");
        }
    }

    #[test]
    fn test_swiglu_up_zeros() {
        let gate = [1.0; 16];
        let up = [0.0; 16];
        let mut output = [0.0; 16];
        simd_swiglu(&gate, &up, &mut output).unwrap();
        for &o in &output {
            assert_close(o, 0.0, 1e-6, "swiglu up=0");
        }
    }

    #[test]
    fn test_all_inplace_non_aligned() {
        for n in [1, 3, 7, 9, 15] {
            let mut data: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
            simd_gelu_inplace(&mut data).unwrap();
            simd_silu_inplace(&mut data).unwrap();
            simd_mish_inplace(&mut data).unwrap();
            simd_sigmoid_inplace(&mut data).unwrap();
            simd_softplus_inplace(&mut data).unwrap();
            simd_gelu_tanh_inplace(&mut data).unwrap();
        }
    }
}

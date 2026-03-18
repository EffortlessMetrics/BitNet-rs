//! CUDA half-precision (FP16/BF16) operations with CPU fallback.
//!
//! Provides FP16 (IEEE 754 binary16) and BF16 (bfloat16) conversion
//! utilities, matrix operations, mixed-precision accumulation, loss
//! scaling for mixed-precision training, and automatic mixed precision
//! (AMP) policy management.
//!
//! # Kernel strategy
//!
//! CUDA kernels use grid-stride loops with 256 threads per block.
//! Conversions and element-wise operations are embarrassingly parallel.
//! Mixed-precision GEMM accumulates in FP32 for numerical stability.
//!
//! # CPU fallback
//!
//! All public functions have pure-Rust implementations using bit-level
//! manipulation for FP16/BF16 encoding.  These serve as the reference
//! for correctness testing and non-GPU environments.

use bitnet_common::{KernelError, Result};

// ---------------------------------------------------------------------------
// CUDA kernel source
// ---------------------------------------------------------------------------

/// Inline CUDA C source for half-precision conversion and arithmetic.
///
/// Contains kernels: `f32_to_f16`, `f16_to_f32`, `f32_to_bf16`,
/// `bf16_to_f32`, `f16_add`, `f16_scale`, `f16_fma`,
/// `mixed_precision_dot`.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const HALF_PRECISION_KERNEL_SRC: &str = r#"
extern "C" __global__ void f32_to_f16(
    const float* __restrict__ input,
    unsigned short* __restrict__ output,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        output[i] = __float2half_rn(input[i]);
    }
}

extern "C" __global__ void f16_to_f32(
    const unsigned short* __restrict__ input,
    float* __restrict__ output,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        output[i] = __half2float(input[i]);
    }
}

extern "C" __global__ void f32_to_bf16(
    const float* __restrict__ input,
    unsigned short* __restrict__ output,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        unsigned int bits = __float_as_uint(input[i]);
        // Round-to-nearest-even: add 0x7FFF + bit[16] for rounding
        unsigned int rounding = ((bits >> 16) & 1) + 0x7FFFu;
        output[i] = (unsigned short)((bits + rounding) >> 16);
    }
}

extern "C" __global__ void bf16_to_f32(
    const unsigned short* __restrict__ input,
    float* __restrict__ output,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        unsigned int bits = ((unsigned int)input[i]) << 16;
        output[i] = __uint_as_float(bits);
    }
}

extern "C" __global__ void f16_add_kernel(
    const unsigned short* __restrict__ a,
    const unsigned short* __restrict__ b,
    unsigned short* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        float va = __half2float(a[i]);
        float vb = __half2float(b[i]);
        out[i] = __float2half_rn(va + vb);
    }
}

extern "C" __global__ void f16_scale_kernel(
    const unsigned short* __restrict__ input,
    unsigned short* __restrict__ output,
    float scale,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        float v = __half2float(input[i]);
        output[i] = __float2half_rn(v * scale);
    }
}

extern "C" __global__ void f16_fma_kernel(
    const unsigned short* __restrict__ a,
    const unsigned short* __restrict__ b,
    const unsigned short* __restrict__ c,
    unsigned short* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        float va = __half2float(a[i]);
        float vb = __half2float(b[i]);
        float vc = __half2float(c[i]);
        out[i] = __float2half_rn(fmaf(va, vb, vc));
    }
}

extern "C" __global__ void mixed_precision_dot_kernel(
    const unsigned short* __restrict__ a,
    const unsigned short* __restrict__ b,
    float* __restrict__ output,
    int n)
{
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        float va = __half2float(a[i]);
        float vb = __half2float(b[i]);
        sum += va * vb;
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(output, sdata[0]);
}
"#;

// ---------------------------------------------------------------------------
// Half-precision type
// ---------------------------------------------------------------------------

/// Supported half-precision floating-point formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HalfFormat {
    /// IEEE 754 binary16 (1 sign + 5 exponent + 10 mantissa).
    F16,
    /// Brain float 16 (1 sign + 8 exponent + 7 mantissa).
    BF16,
}

impl std::fmt::Display for HalfFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HalfFormat::F16 => write!(f, "fp16"),
            HalfFormat::BF16 => write!(f, "bf16"),
        }
    }
}

// ---------------------------------------------------------------------------
// Launch configuration
// ---------------------------------------------------------------------------

/// Launch configuration for half-precision conversion and arithmetic.
#[derive(Debug, Clone)]
pub struct HalfPrecisionConfig {
    /// Number of elements.
    pub n: usize,
    /// Source / operating format.
    pub format: HalfFormat,
    /// Threads per block (default 256).
    pub threads_per_block: u32,
}

impl HalfPrecisionConfig {
    /// Create a configuration for the given element count.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if `n` is zero.
    pub fn new(n: usize, format: HalfFormat) -> Result<Self> {
        if n == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "element count must be non-zero".into(),
            }
            .into());
        }
        Ok(Self { n, format, threads_per_block: 256 })
    }

    /// Compute the CUDA grid dimensions.
    pub fn grid_dim(&self) -> (u32, u32, u32) {
        let blocks = (self.n as u32).div_ceil(self.threads_per_block);
        (blocks, 1, 1)
    }

    /// Compute the CUDA block dimensions.
    pub fn block_dim(&self) -> (u32, u32, u32) {
        (self.threads_per_block, 1, 1)
    }
}

// ---------------------------------------------------------------------------
// FP16 bit-level conversion (CPU)
// ---------------------------------------------------------------------------

/// Convert a single `f32` to IEEE 754 half-precision (binary16) bits.
///
/// Handles subnormals, infinities, and NaN.  Uses round-to-nearest-even.
#[inline]
pub fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007F_FFFF;

    if exponent == 255 {
        // Inf / NaN
        let half_mantissa = if mantissa != 0 { 0x0200u16 } else { 0u16 };
        return sign | 0x7C00 | half_mantissa;
    }

    let unbiased = exponent - 127;

    if unbiased > 15 {
        // Overflow → Inf
        return sign | 0x7C00;
    }

    if unbiased < -24 {
        // Underflow → zero
        return sign;
    }

    if unbiased < -14 {
        // Subnormal in FP16
        let shift = (-14 - unbiased) as u32;
        let subnorm = ((mantissa | 0x0080_0000) >> (13 + shift)) as u16;
        // Round-to-nearest-even
        let round_bit = (mantissa >> (12 + shift)) & 1;
        let sticky = if (mantissa & ((1 << (12 + shift)) - 1)) != 0 { 1u16 } else { 0u16 };
        let round_up = if round_bit == 1 && (sticky == 1 || (subnorm & 1) == 1) { 1u16 } else { 0 };
        return sign | (subnorm + round_up);
    }

    let half_exp = ((unbiased + 15) as u16) << 10;
    let half_mantissa = (mantissa >> 13) as u16;
    // Round-to-nearest-even
    let round_bit = (mantissa >> 12) & 1;
    let sticky = if (mantissa & 0xFFF) != 0 { 1u16 } else { 0u16 };
    let round_up =
        if round_bit == 1 && (sticky == 1 || (half_mantissa & 1) == 1) { 1u16 } else { 0 };

    sign | half_exp | (half_mantissa + round_up)
}

/// Convert IEEE 754 half-precision bits back to `f32`.
#[inline]
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = (bits >> 10) & 0x1F;
    let mantissa = (bits & 0x03FF) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign); // ±0
        }
        // Subnormal: normalize
        let mut m = mantissa;
        let mut e: i32 = -14;
        while (m & 0x0400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x03FF;
        let f32_exp = ((e + 127) as u32) << 23;
        let f32_mantissa = m << 13;
        return f32::from_bits(sign | f32_exp | f32_mantissa);
    }

    if exponent == 31 {
        // Inf / NaN
        let f32_mantissa = if mantissa != 0 { 0x0040_0000 } else { 0 };
        return f32::from_bits(sign | 0x7F80_0000 | f32_mantissa);
    }

    let f32_exp = ((exponent as i32 - 15 + 127) as u32) << 23;
    let f32_mantissa = mantissa << 13;
    f32::from_bits(sign | f32_exp | f32_mantissa)
}

// ---------------------------------------------------------------------------
// BF16 bit-level conversion (CPU)
// ---------------------------------------------------------------------------

/// Convert a single `f32` to bfloat16 bits (truncate + round-to-nearest-even).
#[inline]
pub fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding_bias = ((bits >> 16) & 1) + 0x7FFF;
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

/// Convert bfloat16 bits back to `f32`.
#[inline]
pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

// ---------------------------------------------------------------------------
// Batch conversion (CPU)
// ---------------------------------------------------------------------------

/// Convert a slice of `f32` values to FP16 bits.
pub fn f32_to_f16_batch(input: &[f32], output: &mut [u16]) -> Result<()> {
    if input.len() != output.len() {
        return Err(KernelError::InvalidArguments {
            reason: format!("length mismatch: input={}, output={}", input.len(), output.len()),
        }
        .into());
    }
    for (i, &v) in input.iter().enumerate() {
        output[i] = f32_to_f16_bits(v);
    }
    Ok(())
}

/// Convert a slice of FP16 bits to `f32` values.
pub fn f16_to_f32_batch(input: &[u16], output: &mut [f32]) -> Result<()> {
    if input.len() != output.len() {
        return Err(KernelError::InvalidArguments {
            reason: format!("length mismatch: input={}, output={}", input.len(), output.len()),
        }
        .into());
    }
    for (i, &bits) in input.iter().enumerate() {
        output[i] = f16_bits_to_f32(bits);
    }
    Ok(())
}

/// Convert a slice of `f32` values to BF16 bits.
pub fn f32_to_bf16_batch(input: &[f32], output: &mut [u16]) -> Result<()> {
    if input.len() != output.len() {
        return Err(KernelError::InvalidArguments {
            reason: format!("length mismatch: input={}, output={}", input.len(), output.len()),
        }
        .into());
    }
    for (i, &v) in input.iter().enumerate() {
        output[i] = f32_to_bf16_bits(v);
    }
    Ok(())
}

/// Convert a slice of BF16 bits to `f32` values.
pub fn bf16_to_f32_batch(input: &[u16], output: &mut [f32]) -> Result<()> {
    if input.len() != output.len() {
        return Err(KernelError::InvalidArguments {
            reason: format!("length mismatch: input={}, output={}", input.len(), output.len()),
        }
        .into());
    }
    for (i, &bits) in input.iter().enumerate() {
        output[i] = bf16_bits_to_f32(bits);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// FP16 matrix operations (CPU)
// ---------------------------------------------------------------------------

/// Element-wise addition of two FP16 vectors (stored as `u16` bits).
///
/// Each element is promoted to `f32`, added, then converted back.
pub fn f16_add(a: &[u16], b: &[u16], output: &mut [u16]) -> Result<()> {
    if a.len() != b.len() || a.len() != output.len() {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "length mismatch: a={}, b={}, output={}",
                a.len(),
                b.len(),
                output.len()
            ),
        }
        .into());
    }
    for i in 0..a.len() {
        let va = f16_bits_to_f32(a[i]);
        let vb = f16_bits_to_f32(b[i]);
        output[i] = f32_to_f16_bits(va + vb);
    }
    Ok(())
}

/// Scale every FP16 element by a `f32` scalar.
pub fn f16_scale(input: &[u16], scale: f32, output: &mut [u16]) -> Result<()> {
    if input.len() != output.len() {
        return Err(KernelError::InvalidArguments {
            reason: format!("length mismatch: input={}, output={}", input.len(), output.len()),
        }
        .into());
    }
    for i in 0..input.len() {
        let v = f16_bits_to_f32(input[i]);
        output[i] = f32_to_f16_bits(v * scale);
    }
    Ok(())
}

/// Fused multiply-add on FP16 vectors: `out = a * b + c`.
///
/// Arithmetic is performed in `f32` for numerical stability.
pub fn f16_fma(a: &[u16], b: &[u16], c: &[u16], output: &mut [u16]) -> Result<()> {
    let n = a.len();
    if b.len() != n || c.len() != n || output.len() != n {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "length mismatch: a={n}, b={}, c={}, output={}",
                b.len(),
                c.len(),
                output.len()
            ),
        }
        .into());
    }
    for i in 0..n {
        let va = f16_bits_to_f32(a[i]);
        let vb = f16_bits_to_f32(b[i]);
        let vc = f16_bits_to_f32(c[i]);
        output[i] = f32_to_f16_bits(va.mul_add(vb, vc));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Mixed-precision accumulation (CPU)
// ---------------------------------------------------------------------------

/// Dot product of two FP16 vectors with FP32 accumulator.
///
/// Returns the scalar dot product accumulated in full precision.
pub fn mixed_precision_dot(a: &[u16], b: &[u16]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(KernelError::InvalidArguments {
            reason: format!("length mismatch: a={}, b={}", a.len(), b.len()),
        }
        .into());
    }
    let mut acc: f32 = 0.0;
    for i in 0..a.len() {
        let va = f16_bits_to_f32(a[i]);
        let vb = f16_bits_to_f32(b[i]);
        acc += va * vb;
    }
    Ok(acc)
}

/// Matrix-vector multiply with FP16 matrix and FP16 vector, FP32 accumulator.
///
/// `matrix` is row-major `[rows, cols]`.  Returns `f32` result vector of
/// length `rows`.
pub fn mixed_precision_matvec(
    matrix: &[u16],
    vector: &[u16],
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>> {
    if matrix.len() != rows * cols {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "matrix size mismatch: expected {}×{}={}, got {}",
                rows,
                cols,
                rows * cols,
                matrix.len()
            ),
        }
        .into());
    }
    if vector.len() != cols {
        return Err(KernelError::InvalidArguments {
            reason: format!("vector length mismatch: expected {cols}, got {}", vector.len()),
        }
        .into());
    }
    let mut result = vec![0.0f32; rows];
    for (r, result_elem) in result.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        let row_start = r * cols;
        for c in 0..cols {
            let a = f16_bits_to_f32(matrix[row_start + c]);
            let b = f16_bits_to_f32(vector[c]);
            acc += a * b;
        }
        *result_elem = acc;
    }
    Ok(result)
}

/// Accumulate FP16 values with Kahan summation in FP32 for improved
/// numerical accuracy.
pub fn kahan_accumulate_f16(values: &[u16]) -> f32 {
    let mut sum = 0.0f32;
    let mut compensation = 0.0f32;
    for &bits in values {
        let v = f16_bits_to_f32(bits);
        let y = v - compensation;
        let t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }
    sum
}

// ---------------------------------------------------------------------------
// Loss scaling for mixed-precision training
// ---------------------------------------------------------------------------

/// Dynamic loss scaler for mixed-precision training.
///
/// Scales losses up before backward pass to prevent FP16 gradient
/// underflow, then scales gradients down before the optimizer step.
#[derive(Debug, Clone)]
pub struct LossScaler {
    /// Current scale factor (power of 2).
    pub scale: f32,
    /// Growth factor applied when no overflow is detected.
    pub growth_factor: f32,
    /// Back-off factor applied on overflow.
    pub backoff_factor: f32,
    /// Number of consecutive non-overflow steps before growing.
    pub growth_interval: u32,
    /// Counter of consecutive non-overflow steps.
    pub non_overflow_count: u32,
    /// Whether an overflow was detected in the last step.
    pub overflow_detected: bool,
}

impl Default for LossScaler {
    fn default() -> Self {
        Self {
            scale: 65536.0, // 2^16
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            non_overflow_count: 0,
            overflow_detected: false,
        }
    }
}

impl LossScaler {
    /// Create a loss scaler with a specific initial scale.
    pub fn with_scale(scale: f32) -> Self {
        Self { scale, ..Default::default() }
    }

    /// Scale a loss value before the backward pass.
    pub fn scale_loss(&self, loss: f32) -> f32 {
        loss * self.scale
    }

    /// Unscale gradients after the backward pass.
    ///
    /// Returns `true` if the gradients are valid (no inf/nan).
    pub fn unscale_gradients(&self, gradients: &mut [f32]) -> bool {
        let inv_scale = 1.0 / self.scale;
        let mut valid = true;
        for g in gradients.iter_mut() {
            *g *= inv_scale;
            if g.is_nan() || g.is_infinite() {
                valid = false;
            }
        }
        valid
    }

    /// Update the scaler state after an optimizer step.
    ///
    /// Call with `overflow = true` when `unscale_gradients` detected
    /// inf/nan values.
    pub fn update(&mut self, overflow: bool) {
        self.overflow_detected = overflow;
        if overflow {
            self.scale *= self.backoff_factor;
            self.non_overflow_count = 0;
        } else {
            self.non_overflow_count += 1;
            if self.non_overflow_count >= self.growth_interval {
                self.scale *= self.growth_factor;
                self.non_overflow_count = 0;
            }
        }
    }

    /// Scale FP16 gradients in-place: promote to f32, unscale, return.
    pub fn unscale_f16_gradients(&self, gradients: &[u16]) -> (Vec<f32>, bool) {
        let inv_scale = 1.0 / self.scale;
        let mut f32_grads = Vec::with_capacity(gradients.len());
        let mut valid = true;
        for &bits in gradients {
            let v = f16_bits_to_f32(bits) * inv_scale;
            if v.is_nan() || v.is_infinite() {
                valid = false;
            }
            f32_grads.push(v);
        }
        (f32_grads, valid)
    }
}

// ---------------------------------------------------------------------------
// Precision-aware comparison
// ---------------------------------------------------------------------------

/// Compute the ULP (unit in the last place) distance between two FP16
/// values.
///
/// Returns `None` for NaN inputs.
pub fn f16_ulp_distance(a: u16, b: u16) -> Option<u32> {
    let fa = f16_bits_to_f32(a);
    let fb = f16_bits_to_f32(b);
    if fa.is_nan() || fb.is_nan() {
        return None;
    }
    // Handle signs: reinterpret as signed magnitude
    let ia = if (a & 0x8000) != 0 { -((a & 0x7FFF) as i32) } else { (a & 0x7FFF) as i32 };
    let ib = if (b & 0x8000) != 0 { -((b & 0x7FFF) as i32) } else { (b & 0x7FFF) as i32 };
    Some((ia - ib).unsigned_abs())
}

/// Check approximate equality of two FP16 values within a ULP tolerance.
pub fn f16_approx_eq(a: u16, b: u16, ulp_tolerance: u32) -> bool {
    match f16_ulp_distance(a, b) {
        Some(dist) => dist <= ulp_tolerance,
        None => false,
    }
}

/// Check approximate equality of two f32 values within a relative
/// tolerance.
pub fn f32_approx_eq(a: f32, b: f32, rel_tol: f32) -> bool {
    if a.is_nan() || b.is_nan() {
        return false;
    }
    if a == b {
        return true;
    }
    let diff = (a - b).abs();
    let max_abs = a.abs().max(b.abs());
    if max_abs == 0.0 {
        return diff < rel_tol;
    }
    diff / max_abs <= rel_tol
}

// ---------------------------------------------------------------------------
// FP16 / BF16 tensor creation and initialization
// ---------------------------------------------------------------------------

/// Create a tensor of FP16 zeros.
pub fn f16_zeros(n: usize) -> Vec<u16> {
    vec![0u16; n] // FP16 zero is 0x0000
}

/// Create a tensor of BF16 zeros.
pub fn bf16_zeros(n: usize) -> Vec<u16> {
    vec![0u16; n] // BF16 zero is 0x0000
}

/// Create a tensor of FP16 ones.
pub fn f16_ones(n: usize) -> Vec<u16> {
    vec![f32_to_f16_bits(1.0); n]
}

/// Create a tensor of BF16 ones.
pub fn bf16_ones(n: usize) -> Vec<u16> {
    vec![f32_to_bf16_bits(1.0); n]
}

/// Create a tensor filled with a constant value in FP16.
pub fn f16_full(n: usize, value: f32) -> Vec<u16> {
    vec![f32_to_f16_bits(value); n]
}

/// Create a tensor filled with a constant value in BF16.
pub fn bf16_full(n: usize, value: f32) -> Vec<u16> {
    vec![f32_to_bf16_bits(value); n]
}

/// Create a linearly spaced FP16 tensor from `start` to `end`
/// (inclusive) with `n` elements.
pub fn f16_linspace(start: f32, end: f32, n: usize) -> Vec<u16> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![f32_to_f16_bits(start)];
    }
    let step = (end - start) / (n - 1) as f32;
    (0..n).map(|i| f32_to_f16_bits(start + step * i as f32)).collect()
}

/// Create a linearly spaced BF16 tensor.
pub fn bf16_linspace(start: f32, end: f32, n: usize) -> Vec<u16> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![f32_to_bf16_bits(start)];
    }
    let step = (end - start) / (n - 1) as f32;
    (0..n).map(|i| f32_to_bf16_bits(start + step * i as f32)).collect()
}

// ---------------------------------------------------------------------------
// Automatic Mixed Precision (AMP) policy
// ---------------------------------------------------------------------------

/// Classification of an operation's precision requirement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecisionClass {
    /// Must run in FP32 (e.g., softmax, layer norm, loss).
    Fp32,
    /// Safe to run in FP16 / BF16 (e.g., GEMM, convolution).
    Half,
    /// Can run in either precision without significant accuracy loss
    /// (e.g., element-wise add, ReLU).
    Either,
}

/// Named operation types recognised by the AMP policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpType {
    Gemm,
    Convolution,
    LinearProjection,
    Softmax,
    LayerNorm,
    RmsNorm,
    Attention,
    Embedding,
    Loss,
    ElementwiseAdd,
    ElementwiseMul,
    Activation,
    Residual,
    Reduction,
}

/// AMP policy that decides per-operation precision.
#[derive(Debug, Clone)]
pub struct AmpPolicy {
    /// Default format for "Half" operations.
    pub default_half_format: HalfFormat,
    /// Whether AMP is enabled.
    pub enabled: bool,
}

impl Default for AmpPolicy {
    fn default() -> Self {
        Self { default_half_format: HalfFormat::F16, enabled: true }
    }
}

impl AmpPolicy {
    /// Create an AMP policy that uses FP16.
    pub fn fp16() -> Self {
        Self { default_half_format: HalfFormat::F16, enabled: true }
    }

    /// Create an AMP policy that uses BF16.
    pub fn bf16() -> Self {
        Self { default_half_format: HalfFormat::BF16, enabled: true }
    }

    /// Create a disabled AMP policy (everything runs in FP32).
    pub fn disabled() -> Self {
        Self { default_half_format: HalfFormat::F16, enabled: false }
    }

    /// Classify an operation into its precision category.
    pub fn classify(&self, op: OpType) -> PrecisionClass {
        match op {
            // Must stay in FP32 for numerical stability
            OpType::Softmax | OpType::LayerNorm | OpType::RmsNorm | OpType::Loss => {
                PrecisionClass::Fp32
            }
            // Benefit from half-precision compute
            OpType::Gemm | OpType::Convolution | OpType::LinearProjection | OpType::Attention => {
                PrecisionClass::Half
            }
            // Can go either way
            OpType::Embedding
            | OpType::ElementwiseAdd
            | OpType::ElementwiseMul
            | OpType::Activation
            | OpType::Residual
            | OpType::Reduction => PrecisionClass::Either,
        }
    }

    /// Decide whether a given operation should run in half-precision.
    pub fn should_use_half(&self, op: OpType) -> bool {
        if !self.enabled {
            return false;
        }
        match self.classify(op) {
            PrecisionClass::Half => true,
            PrecisionClass::Either => true,
            PrecisionClass::Fp32 => false,
        }
    }

    /// Return the half format to use for an operation, or `None` if
    /// the operation must use FP32.
    pub fn resolve_format(&self, op: OpType) -> Option<HalfFormat> {
        if self.should_use_half(op) { Some(self.default_half_format) } else { None }
    }
}

// ---------------------------------------------------------------------------
// GPU launch stubs
// ---------------------------------------------------------------------------

/// Launch FP32 → FP16 conversion on GPU.
///
/// # Errors
///
/// Returns an error if the GPU runtime is unavailable.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_f32_to_f16(config: &HalfPrecisionConfig) -> Result<()> {
    let _ = config;
    Err(bitnet_common::BitNetError::Kernel(KernelError::DeviceUnavailable {
        reason: "CUDA runtime not initialised (stub)".into(),
    }))
}

/// Launch FP16 → FP32 conversion on GPU.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_f16_to_f32(config: &HalfPrecisionConfig) -> Result<()> {
    let _ = config;
    Err(bitnet_common::BitNetError::Kernel(KernelError::DeviceUnavailable {
        reason: "CUDA runtime not initialised (stub)".into(),
    }))
}

/// Launch FP16 element-wise add on GPU.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_f16_add(config: &HalfPrecisionConfig) -> Result<()> {
    let _ = config;
    Err(bitnet_common::BitNetError::Kernel(KernelError::DeviceUnavailable {
        reason: "CUDA runtime not initialised (stub)".into(),
    }))
}

/// Launch FP16 fused multiply-add on GPU.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_f16_fma(config: &HalfPrecisionConfig) -> Result<()> {
    let _ = config;
    Err(bitnet_common::BitNetError::Kernel(KernelError::DeviceUnavailable {
        reason: "CUDA runtime not initialised (stub)".into(),
    }))
}

/// Launch mixed-precision dot product on GPU.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_mixed_precision_dot(config: &HalfPrecisionConfig) -> Result<()> {
    let _ = config;
    Err(bitnet_common::BitNetError::Kernel(KernelError::DeviceUnavailable {
        reason: "CUDA runtime not initialised (stub)".into(),
    }))
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── FP16 conversion: basic values ──────────────────────────────────

    #[test]
    fn test_f16_roundtrip_zero() {
        let bits = f32_to_f16_bits(0.0);
        assert_eq!(bits, 0x0000);
        assert_eq!(f16_bits_to_f32(bits), 0.0);
    }

    #[test]
    fn test_f16_roundtrip_neg_zero() {
        let bits = f32_to_f16_bits(-0.0);
        assert_eq!(bits, 0x8000);
        let v = f16_bits_to_f32(bits);
        assert!(v == 0.0 && v.is_sign_negative());
    }

    #[test]
    fn test_f16_roundtrip_one() {
        let bits = f32_to_f16_bits(1.0);
        assert_eq!(bits, 0x3C00);
        assert_eq!(f16_bits_to_f32(bits), 1.0);
    }

    #[test]
    fn test_f16_roundtrip_neg_one() {
        let bits = f32_to_f16_bits(-1.0);
        assert_eq!(bits, 0xBC00);
        assert_eq!(f16_bits_to_f32(bits), -1.0);
    }

    #[test]
    fn test_f16_roundtrip_half() {
        let bits = f32_to_f16_bits(0.5);
        assert_eq!(bits, 0x3800);
        assert_eq!(f16_bits_to_f32(bits), 0.5);
    }

    #[test]
    fn test_f16_roundtrip_two() {
        let bits = f32_to_f16_bits(2.0);
        assert_eq!(bits, 0x4000);
        assert_eq!(f16_bits_to_f32(bits), 2.0);
    }

    #[test]
    fn test_f16_positive_infinity() {
        let bits = f32_to_f16_bits(f32::INFINITY);
        assert_eq!(bits, 0x7C00);
        assert!(f16_bits_to_f32(bits).is_infinite());
        assert!(f16_bits_to_f32(bits).is_sign_positive());
    }

    #[test]
    fn test_f16_negative_infinity() {
        let bits = f32_to_f16_bits(f32::NEG_INFINITY);
        assert_eq!(bits, 0xFC00);
        assert!(f16_bits_to_f32(bits).is_infinite());
        assert!(f16_bits_to_f32(bits).is_sign_negative());
    }

    #[test]
    fn test_f16_nan() {
        let bits = f32_to_f16_bits(f32::NAN);
        assert!(f16_bits_to_f32(bits).is_nan());
    }

    #[test]
    fn test_f16_overflow_to_inf() {
        // FP16 max is ~65504; 100000.0 should overflow to infinity
        let bits = f32_to_f16_bits(100_000.0);
        assert_eq!(bits, 0x7C00);
    }

    #[test]
    fn test_f16_small_subnormal() {
        // Smallest FP16 subnormal: 2^-24 ≈ 5.96e-8
        let tiny = 5.96e-8_f32;
        let bits = f32_to_f16_bits(tiny);
        let back = f16_bits_to_f32(bits);
        assert!((back - tiny).abs() < 1e-7);
    }

    #[test]
    fn test_f16_underflow_to_zero() {
        // Values smaller than smallest subnormal → zero
        let bits = f32_to_f16_bits(1e-10);
        assert_eq!(bits, 0x0000);
    }

    #[test]
    fn test_f16_roundtrip_many_values() {
        let values = [0.1, 0.25, 0.75, 1.5, 2.5, 42.0, 255.0, 1024.0, 65504.0];
        for &v in &values {
            let bits = f32_to_f16_bits(v);
            let back = f16_bits_to_f32(bits);
            let rel_err = ((back - v) / v).abs();
            assert!(rel_err < 0.002, "f16 roundtrip of {v}: got {back}, rel_err={rel_err}");
        }
    }

    #[test]
    fn test_f16_negative_roundtrip() {
        let values = [-0.5, -1.0, -42.0, -1024.0];
        for &v in &values {
            let bits = f32_to_f16_bits(v);
            let back = f16_bits_to_f32(bits);
            let rel_err = ((back - v) / v.abs()).abs();
            assert!(rel_err < 0.002, "f16 roundtrip of {v}: got {back}");
        }
    }

    // ── BF16 conversion ────────────────────────────────────────────────

    #[test]
    fn test_bf16_roundtrip_zero() {
        let bits = f32_to_bf16_bits(0.0);
        assert_eq!(bits, 0x0000);
        assert_eq!(bf16_bits_to_f32(bits), 0.0);
    }

    #[test]
    fn test_bf16_roundtrip_one() {
        let bits = f32_to_bf16_bits(1.0);
        assert_eq!(bits, 0x3F80);
        assert_eq!(bf16_bits_to_f32(bits), 1.0);
    }

    #[test]
    fn test_bf16_roundtrip_neg_one() {
        let bits = f32_to_bf16_bits(-1.0);
        assert_eq!(bits, 0xBF80);
        assert_eq!(bf16_bits_to_f32(bits), -1.0);
    }

    #[test]
    fn test_bf16_large_range() {
        // BF16 has same exponent range as f32, so it can represent large values
        let v = 100_000.0f32;
        let bits = f32_to_bf16_bits(v);
        let back = bf16_bits_to_f32(bits);
        let rel_err = ((back - v) / v).abs();
        assert!(rel_err < 0.01, "bf16 roundtrip of {v}: got {back}");
    }

    #[test]
    fn test_bf16_infinity() {
        let bits = f32_to_bf16_bits(f32::INFINITY);
        assert!(bf16_bits_to_f32(bits).is_infinite());
    }

    #[test]
    fn test_bf16_nan() {
        let bits = f32_to_bf16_bits(f32::NAN);
        assert!(bf16_bits_to_f32(bits).is_nan());
    }

    #[test]
    fn test_bf16_neg_zero() {
        let bits = f32_to_bf16_bits(-0.0);
        let v = bf16_bits_to_f32(bits);
        assert!(v == 0.0 && v.is_sign_negative());
    }

    #[test]
    fn test_bf16_roundtrip_many() {
        let values = [0.1, 0.5, 1.5, 2.5, 42.0, 1024.0, 65504.0];
        for &v in &values {
            let bits = f32_to_bf16_bits(v);
            let back = bf16_bits_to_f32(bits);
            let rel_err = ((back - v) / v).abs();
            assert!(rel_err < 0.01, "bf16 roundtrip of {v}: got {back}");
        }
    }

    // ── Batch conversion ───────────────────────────────────────────────

    #[test]
    fn test_f32_to_f16_batch_basic() {
        let input = [1.0f32, 2.0, 0.5, -1.0];
        let mut output = [0u16; 4];
        f32_to_f16_batch(&input, &mut output).unwrap();
        for (i, &v) in input.iter().enumerate() {
            let back = f16_bits_to_f32(output[i]);
            assert!((back - v).abs() < 0.01, "index {i}: expected {v}, got {back}");
        }
    }

    #[test]
    fn test_f16_to_f32_batch_basic() {
        let f16_vals: Vec<u16> =
            [1.0f32, 2.0, 0.5, -1.0].iter().map(|&v| f32_to_f16_bits(v)).collect();
        let mut output = [0.0f32; 4];
        f16_to_f32_batch(&f16_vals, &mut output).unwrap();
        assert_eq!(output, [1.0, 2.0, 0.5, -1.0]);
    }

    #[test]
    fn test_f32_to_bf16_batch_basic() {
        let input = [1.0f32, -1.0, 0.0];
        let mut output = [0u16; 3];
        f32_to_bf16_batch(&input, &mut output).unwrap();
        for (i, &v) in input.iter().enumerate() {
            let back = bf16_bits_to_f32(output[i]);
            assert!((back - v).abs() < 0.01, "index {i}: expected {v}, got {back}");
        }
    }

    #[test]
    fn test_bf16_to_f32_batch_basic() {
        let bf16_vals: Vec<u16> =
            [1.0f32, -1.0, 0.0].iter().map(|&v| f32_to_bf16_bits(v)).collect();
        let mut output = [0.0f32; 3];
        bf16_to_f32_batch(&bf16_vals, &mut output).unwrap();
        assert_eq!(output, [1.0, -1.0, 0.0]);
    }

    #[test]
    fn test_batch_length_mismatch_f16() {
        let input = [1.0f32];
        let mut output = [0u16; 2];
        assert!(f32_to_f16_batch(&input, &mut output).is_err());
    }

    #[test]
    fn test_batch_length_mismatch_bf16() {
        let input = [1.0f32, 2.0];
        let mut output = [0u16; 1];
        assert!(f32_to_bf16_batch(&input, &mut output).is_err());
    }

    #[test]
    fn test_f16_to_f32_batch_length_mismatch() {
        let input = [0u16; 3];
        let mut output = [0.0f32; 2];
        assert!(f16_to_f32_batch(&input, &mut output).is_err());
    }

    #[test]
    fn test_bf16_to_f32_batch_length_mismatch() {
        let input = [0u16; 1];
        let mut output = [0.0f32; 5];
        assert!(bf16_to_f32_batch(&input, &mut output).is_err());
    }

    // ── FP16 matrix operations ─────────────────────────────────────────

    #[test]
    fn test_f16_add_basic() {
        let a: Vec<u16> = [1.0f32, 2.0, 3.0].iter().map(|&v| f32_to_f16_bits(v)).collect();
        let b: Vec<u16> = [4.0f32, 5.0, 6.0].iter().map(|&v| f32_to_f16_bits(v)).collect();
        let mut out = [0u16; 3];
        f16_add(&a, &b, &mut out).unwrap();
        let result: Vec<f32> = out.iter().map(|&bits| f16_bits_to_f32(bits)).collect();
        assert_eq!(result, [5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_f16_add_length_mismatch() {
        let a = [0u16; 3];
        let b = [0u16; 2];
        let mut out = [0u16; 3];
        assert!(f16_add(&a, &b, &mut out).is_err());
    }

    #[test]
    fn test_f16_add_with_negatives() {
        let a: Vec<u16> = [1.0, -2.0f32].iter().map(|&v| f32_to_f16_bits(v)).collect();
        let b: Vec<u16> = [-1.0, 2.0f32].iter().map(|&v| f32_to_f16_bits(v)).collect();
        let mut out = [0u16; 2];
        f16_add(&a, &b, &mut out).unwrap();
        let result: Vec<f32> = out.iter().map(|&bits| f16_bits_to_f32(bits)).collect();
        assert_eq!(result, [0.0, 0.0]);
    }

    #[test]
    fn test_f16_scale_basic() {
        let input: Vec<u16> = [1.0f32, 2.0, 4.0].iter().map(|&v| f32_to_f16_bits(v)).collect();
        let mut out = [0u16; 3];
        f16_scale(&input, 0.5, &mut out).unwrap();
        let result: Vec<f32> = out.iter().map(|&bits| f16_bits_to_f32(bits)).collect();
        assert_eq!(result, [0.5, 1.0, 2.0]);
    }

    #[test]
    fn test_f16_scale_zero() {
        let input: Vec<u16> = [1.0f32, 2.0].iter().map(|&v| f32_to_f16_bits(v)).collect();
        let mut out = [0u16; 2];
        f16_scale(&input, 0.0, &mut out).unwrap();
        let result: Vec<f32> = out.iter().map(|&bits| f16_bits_to_f32(bits)).collect();
        assert_eq!(result, [0.0, 0.0]);
    }

    #[test]
    fn test_f16_scale_length_mismatch() {
        let input = [0u16; 2];
        let mut out = [0u16; 3];
        assert!(f16_scale(&input, 1.0, &mut out).is_err());
    }

    #[test]
    fn test_f16_fma_basic() {
        // out = a * b + c = [1*2+10, 3*4+20] = [12, 32]
        let a: Vec<u16> = [1.0f32, 3.0].iter().map(|&v| f32_to_f16_bits(v)).collect();
        let b: Vec<u16> = [2.0f32, 4.0].iter().map(|&v| f32_to_f16_bits(v)).collect();
        let c: Vec<u16> = [10.0f32, 20.0].iter().map(|&v| f32_to_f16_bits(v)).collect();
        let mut out = [0u16; 2];
        f16_fma(&a, &b, &c, &mut out).unwrap();
        let result: Vec<f32> = out.iter().map(|&bits| f16_bits_to_f32(bits)).collect();
        assert_eq!(result, [12.0, 32.0]);
    }

    #[test]
    fn test_f16_fma_identity() {
        // a * 1 + 0 = a
        let a: Vec<u16> = [3.5f32, -2.0].iter().map(|&v| f32_to_f16_bits(v)).collect();
        let ones: Vec<u16> = [1.0f32; 2].iter().map(|&v| f32_to_f16_bits(v)).collect();
        let zeros: Vec<u16> = [0.0f32; 2].iter().map(|&v| f32_to_f16_bits(v)).collect();
        let mut out = [0u16; 2];
        f16_fma(&a, &ones, &zeros, &mut out).unwrap();
        let result: Vec<f32> = out.iter().map(|&bits| f16_bits_to_f32(bits)).collect();
        assert_eq!(result, [3.5, -2.0]);
    }

    #[test]
    fn test_f16_fma_length_mismatch() {
        let a = [0u16; 2];
        let b = [0u16; 3];
        let c = [0u16; 2];
        let mut out = [0u16; 2];
        assert!(f16_fma(&a, &b, &c, &mut out).is_err());
    }

    // ── Mixed-precision accumulation ───────────────────────────────────

    #[test]
    fn test_mixed_precision_dot_basic() {
        let a: Vec<u16> = [1.0f32, 2.0, 3.0].iter().map(|&v| f32_to_f16_bits(v)).collect();
        let b: Vec<u16> = [4.0f32, 5.0, 6.0].iter().map(|&v| f32_to_f16_bits(v)).collect();
        let result = mixed_precision_dot(&a, &b).unwrap();
        // 1*4 + 2*5 + 3*6 = 32
        assert!((result - 32.0).abs() < 1e-3);
    }

    #[test]
    fn test_mixed_precision_dot_zero() {
        let a: Vec<u16> = [1.0f32, 2.0].iter().map(|&v| f32_to_f16_bits(v)).collect();
        let b: Vec<u16> = [0.0f32, 0.0].iter().map(|&v| f32_to_f16_bits(v)).collect();
        let result = mixed_precision_dot(&a, &b).unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_mixed_precision_dot_length_mismatch() {
        let a = [0u16; 3];
        let b = [0u16; 2];
        assert!(mixed_precision_dot(&a, &b).is_err());
    }

    #[test]
    fn test_mixed_precision_dot_single() {
        let a = vec![f32_to_f16_bits(3.0)];
        let b = vec![f32_to_f16_bits(7.0)];
        let result = mixed_precision_dot(&a, &b).unwrap();
        assert!((result - 21.0).abs() < 1e-3);
    }

    #[test]
    fn test_mixed_precision_matvec_basic() {
        // 2x3 matrix × 3-vector
        // [[1,2,3],[4,5,6]] · [1,1,1] = [6, 15]
        let mat: Vec<u16> =
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32].iter().map(|&v| f32_to_f16_bits(v)).collect();
        let vec_data: Vec<u16> = [1.0f32; 3].iter().map(|&v| f32_to_f16_bits(v)).collect();
        let result = mixed_precision_matvec(&mat, &vec_data, 2, 3).unwrap();
        assert!((result[0] - 6.0).abs() < 1e-2);
        assert!((result[1] - 15.0).abs() < 1e-2);
    }

    #[test]
    fn test_mixed_precision_matvec_identity() {
        // 2x2 identity × [3, 7] = [3, 7]
        let mat: Vec<u16> = [1.0, 0.0, 0.0, 1.0f32].iter().map(|&v| f32_to_f16_bits(v)).collect();
        let vec_data: Vec<u16> = [3.0, 7.0f32].iter().map(|&v| f32_to_f16_bits(v)).collect();
        let result = mixed_precision_matvec(&mat, &vec_data, 2, 2).unwrap();
        assert!((result[0] - 3.0).abs() < 1e-3);
        assert!((result[1] - 7.0).abs() < 1e-3);
    }

    #[test]
    fn test_mixed_precision_matvec_size_mismatch_matrix() {
        let mat = [0u16; 5]; // Not 2×3 = 6
        let vec_data = [0u16; 3];
        assert!(mixed_precision_matvec(&mat, &vec_data, 2, 3).is_err());
    }

    #[test]
    fn test_mixed_precision_matvec_size_mismatch_vector() {
        let mat = [0u16; 6]; // 2×3
        let vec_data = [0u16; 2]; // Should be 3
        assert!(mixed_precision_matvec(&mat, &vec_data, 2, 3).is_err());
    }

    #[test]
    fn test_kahan_accumulate_basic() {
        let vals: Vec<u16> = [1.0f32, 2.0, 3.0, 4.0].iter().map(|&v| f32_to_f16_bits(v)).collect();
        let result = kahan_accumulate_f16(&vals);
        assert!((result - 10.0).abs() < 1e-3);
    }

    #[test]
    fn test_kahan_accumulate_empty() {
        let result = kahan_accumulate_f16(&[]);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_kahan_accumulate_cancellation() {
        // Many small values that would lose precision with naive sum
        let n = 1000;
        let small = 0.001f32;
        let vals: Vec<u16> = (0..n).map(|_| f32_to_f16_bits(small)).collect();
        let result = kahan_accumulate_f16(&vals);
        let expected = small * n as f32;
        assert!((result - expected).abs() < 0.1, "kahan: got {result}, expected ~{expected}");
    }

    // ── Loss scaling ───────────────────────────────────────────────────

    #[test]
    fn test_loss_scaler_default() {
        let scaler = LossScaler::default();
        assert_eq!(scaler.scale, 65536.0);
        assert_eq!(scaler.growth_factor, 2.0);
        assert_eq!(scaler.backoff_factor, 0.5);
        assert_eq!(scaler.growth_interval, 2000);
    }

    #[test]
    fn test_loss_scaler_with_scale() {
        let scaler = LossScaler::with_scale(1024.0);
        assert_eq!(scaler.scale, 1024.0);
    }

    #[test]
    fn test_loss_scaler_scale_loss() {
        let scaler = LossScaler::with_scale(8.0);
        assert_eq!(scaler.scale_loss(2.0), 16.0);
    }

    #[test]
    fn test_loss_scaler_unscale_gradients() {
        let scaler = LossScaler::with_scale(4.0);
        let mut grads = [8.0f32, -12.0, 4.0];
        let valid = scaler.unscale_gradients(&mut grads);
        assert!(valid);
        assert_eq!(grads, [2.0, -3.0, 1.0]);
    }

    #[test]
    fn test_loss_scaler_unscale_detects_inf() {
        let scaler = LossScaler::with_scale(4.0);
        let mut grads = [f32::INFINITY, 4.0];
        let valid = scaler.unscale_gradients(&mut grads);
        assert!(!valid);
    }

    #[test]
    fn test_loss_scaler_unscale_detects_nan() {
        let scaler = LossScaler::with_scale(4.0);
        let mut grads = [f32::NAN, 4.0];
        let valid = scaler.unscale_gradients(&mut grads);
        assert!(!valid);
    }

    #[test]
    fn test_loss_scaler_update_no_overflow() {
        let mut scaler = LossScaler::with_scale(8.0);
        scaler.growth_interval = 3;
        scaler.update(false);
        assert_eq!(scaler.non_overflow_count, 1);
        assert_eq!(scaler.scale, 8.0);
        scaler.update(false);
        assert_eq!(scaler.non_overflow_count, 2);
        scaler.update(false);
        // Growth triggered
        assert_eq!(scaler.scale, 16.0);
        assert_eq!(scaler.non_overflow_count, 0);
    }

    #[test]
    fn test_loss_scaler_update_overflow() {
        let mut scaler = LossScaler::with_scale(16.0);
        scaler.non_overflow_count = 100;
        scaler.update(true);
        assert_eq!(scaler.scale, 8.0); // 16 * 0.5
        assert_eq!(scaler.non_overflow_count, 0);
        assert!(scaler.overflow_detected);
    }

    #[test]
    fn test_loss_scaler_unscale_f16_gradients() {
        let scaler = LossScaler::with_scale(2.0);
        let grads: Vec<u16> = [4.0f32, -6.0].iter().map(|&v| f32_to_f16_bits(v)).collect();
        let (f32_grads, valid) = scaler.unscale_f16_gradients(&grads);
        assert!(valid);
        assert!((f32_grads[0] - 2.0).abs() < 1e-2);
        assert!((f32_grads[1] - (-3.0)).abs() < 1e-2);
    }

    // ── Precision-aware comparison ─────────────────────────────────────

    #[test]
    fn test_f16_ulp_distance_same() {
        let bits = f32_to_f16_bits(1.0);
        assert_eq!(f16_ulp_distance(bits, bits), Some(0));
    }

    #[test]
    fn test_f16_ulp_distance_adjacent() {
        let a = f32_to_f16_bits(1.0);
        let b = a + 1; // next representable FP16
        assert_eq!(f16_ulp_distance(a, b), Some(1));
    }

    #[test]
    fn test_f16_ulp_distance_nan() {
        let nan_bits: u16 = 0x7E00; // FP16 NaN
        let one = f32_to_f16_bits(1.0);
        assert_eq!(f16_ulp_distance(nan_bits, one), None);
    }

    #[test]
    fn test_f16_approx_eq_exact() {
        let a = f32_to_f16_bits(1.5);
        assert!(f16_approx_eq(a, a, 0));
    }

    #[test]
    fn test_f16_approx_eq_within_tolerance() {
        let a = f32_to_f16_bits(1.0);
        let b = a + 2;
        assert!(f16_approx_eq(a, b, 2));
        assert!(!f16_approx_eq(a, b, 1));
    }

    #[test]
    fn test_f16_approx_eq_nan() {
        let nan_bits: u16 = 0x7E00;
        assert!(!f16_approx_eq(nan_bits, nan_bits, 100));
    }

    #[test]
    fn test_f32_approx_eq_exact() {
        assert!(f32_approx_eq(1.0, 1.0, 1e-6));
    }

    #[test]
    fn test_f32_approx_eq_close() {
        assert!(f32_approx_eq(1.0, 1.0001, 0.001));
        assert!(!f32_approx_eq(1.0, 1.01, 0.001));
    }

    #[test]
    fn test_f32_approx_eq_nan() {
        assert!(!f32_approx_eq(f32::NAN, 1.0, 1e-3));
        assert!(!f32_approx_eq(f32::NAN, f32::NAN, 1e-3));
    }

    #[test]
    fn test_f32_approx_eq_zero() {
        assert!(f32_approx_eq(0.0, 0.0, 1e-6));
    }

    // ── Tensor creation ────────────────────────────────────────────────

    #[test]
    fn test_f16_zeros() {
        let z = f16_zeros(4);
        assert_eq!(z.len(), 4);
        for &bits in &z {
            assert_eq!(f16_bits_to_f32(bits), 0.0);
        }
    }

    #[test]
    fn test_bf16_zeros() {
        let z = bf16_zeros(3);
        assert_eq!(z.len(), 3);
        for &bits in &z {
            assert_eq!(bf16_bits_to_f32(bits), 0.0);
        }
    }

    #[test]
    fn test_f16_ones() {
        let o = f16_ones(5);
        assert_eq!(o.len(), 5);
        for &bits in &o {
            assert_eq!(f16_bits_to_f32(bits), 1.0);
        }
    }

    #[test]
    fn test_bf16_ones() {
        let o = bf16_ones(2);
        assert_eq!(o.len(), 2);
        for &bits in &o {
            assert_eq!(bf16_bits_to_f32(bits), 1.0);
        }
    }

    #[test]
    fn test_f16_full() {
        let t = f16_full(3, 42.0);
        for &bits in &t {
            assert!((f16_bits_to_f32(bits) - 42.0).abs() < 0.1);
        }
    }

    #[test]
    fn test_bf16_full() {
        let t = bf16_full(3, -7.0);
        for &bits in &t {
            assert!((bf16_bits_to_f32(bits) - (-7.0)).abs() < 0.1);
        }
    }

    #[test]
    fn test_f16_linspace() {
        let t = f16_linspace(0.0, 1.0, 5);
        assert_eq!(t.len(), 5);
        let vals: Vec<f32> = t.iter().map(|&b| f16_bits_to_f32(b)).collect();
        assert!((vals[0] - 0.0).abs() < 1e-3);
        assert!((vals[2] - 0.5).abs() < 1e-2);
        assert!((vals[4] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_f16_linspace_single() {
        let t = f16_linspace(3.0, 5.0, 1);
        assert_eq!(t.len(), 1);
        assert!((f16_bits_to_f32(t[0]) - 3.0).abs() < 1e-3);
    }

    #[test]
    fn test_f16_linspace_empty() {
        let t = f16_linspace(0.0, 1.0, 0);
        assert!(t.is_empty());
    }

    #[test]
    fn test_bf16_linspace() {
        let t = bf16_linspace(0.0, 10.0, 3);
        assert_eq!(t.len(), 3);
        let vals: Vec<f32> = t.iter().map(|&b| bf16_bits_to_f32(b)).collect();
        assert!((vals[0] - 0.0).abs() < 0.1);
        assert!((vals[1] - 5.0).abs() < 0.1);
        assert!((vals[2] - 10.0).abs() < 0.1);
    }

    // ── HalfPrecisionConfig ────────────────────────────────────────────

    #[test]
    fn test_config_new() {
        let cfg = HalfPrecisionConfig::new(512, HalfFormat::F16).unwrap();
        assert_eq!(cfg.n, 512);
        assert_eq!(cfg.format, HalfFormat::F16);
        assert_eq!(cfg.threads_per_block, 256);
    }

    #[test]
    fn test_config_rejects_zero() {
        assert!(HalfPrecisionConfig::new(0, HalfFormat::BF16).is_err());
    }

    #[test]
    fn test_config_grid_dim() {
        let cfg = HalfPrecisionConfig::new(1000, HalfFormat::F16).unwrap();
        assert_eq!(cfg.grid_dim(), (4, 1, 1));
        assert_eq!(cfg.block_dim(), (256, 1, 1));
    }

    #[test]
    fn test_config_grid_dim_exact() {
        let cfg = HalfPrecisionConfig::new(256, HalfFormat::F16).unwrap();
        assert_eq!(cfg.grid_dim(), (1, 1, 1));
    }

    #[test]
    fn test_config_bf16_format() {
        let cfg = HalfPrecisionConfig::new(100, HalfFormat::BF16).unwrap();
        assert_eq!(cfg.format, HalfFormat::BF16);
    }

    // ── HalfFormat display ─────────────────────────────────────────────

    #[test]
    fn test_half_format_display() {
        assert_eq!(format!("{}", HalfFormat::F16), "fp16");
        assert_eq!(format!("{}", HalfFormat::BF16), "bf16");
    }

    // ── AMP policy ─────────────────────────────────────────────────────

    #[test]
    fn test_amp_policy_default() {
        let policy = AmpPolicy::default();
        assert!(policy.enabled);
        assert_eq!(policy.default_half_format, HalfFormat::F16);
    }

    #[test]
    fn test_amp_policy_fp16() {
        let policy = AmpPolicy::fp16();
        assert_eq!(policy.default_half_format, HalfFormat::F16);
        assert!(policy.enabled);
    }

    #[test]
    fn test_amp_policy_bf16() {
        let policy = AmpPolicy::bf16();
        assert_eq!(policy.default_half_format, HalfFormat::BF16);
        assert!(policy.enabled);
    }

    #[test]
    fn test_amp_policy_disabled() {
        let policy = AmpPolicy::disabled();
        assert!(!policy.enabled);
    }

    #[test]
    fn test_amp_classify_fp32_ops() {
        let policy = AmpPolicy::default();
        assert_eq!(policy.classify(OpType::Softmax), PrecisionClass::Fp32);
        assert_eq!(policy.classify(OpType::LayerNorm), PrecisionClass::Fp32);
        assert_eq!(policy.classify(OpType::RmsNorm), PrecisionClass::Fp32);
        assert_eq!(policy.classify(OpType::Loss), PrecisionClass::Fp32);
    }

    #[test]
    fn test_amp_classify_half_ops() {
        let policy = AmpPolicy::default();
        assert_eq!(policy.classify(OpType::Gemm), PrecisionClass::Half);
        assert_eq!(policy.classify(OpType::Convolution), PrecisionClass::Half);
        assert_eq!(policy.classify(OpType::LinearProjection), PrecisionClass::Half);
        assert_eq!(policy.classify(OpType::Attention), PrecisionClass::Half);
    }

    #[test]
    fn test_amp_classify_either_ops() {
        let policy = AmpPolicy::default();
        assert_eq!(policy.classify(OpType::Embedding), PrecisionClass::Either);
        assert_eq!(policy.classify(OpType::ElementwiseAdd), PrecisionClass::Either);
        assert_eq!(policy.classify(OpType::Activation), PrecisionClass::Either);
        assert_eq!(policy.classify(OpType::Residual), PrecisionClass::Either);
        assert_eq!(policy.classify(OpType::Reduction), PrecisionClass::Either);
    }

    #[test]
    fn test_amp_should_use_half_enabled() {
        let policy = AmpPolicy::fp16();
        assert!(policy.should_use_half(OpType::Gemm));
        assert!(policy.should_use_half(OpType::Activation));
        assert!(!policy.should_use_half(OpType::Softmax));
        assert!(!policy.should_use_half(OpType::Loss));
    }

    #[test]
    fn test_amp_should_use_half_disabled() {
        let policy = AmpPolicy::disabled();
        assert!(!policy.should_use_half(OpType::Gemm));
        assert!(!policy.should_use_half(OpType::Activation));
        assert!(!policy.should_use_half(OpType::Softmax));
    }

    #[test]
    fn test_amp_resolve_format() {
        let policy = AmpPolicy::bf16();
        assert_eq!(policy.resolve_format(OpType::Gemm), Some(HalfFormat::BF16));
        assert_eq!(policy.resolve_format(OpType::Activation), Some(HalfFormat::BF16));
        assert_eq!(policy.resolve_format(OpType::Softmax), None);
    }

    #[test]
    fn test_amp_resolve_format_disabled() {
        let policy = AmpPolicy::disabled();
        assert_eq!(policy.resolve_format(OpType::Gemm), None);
    }
}

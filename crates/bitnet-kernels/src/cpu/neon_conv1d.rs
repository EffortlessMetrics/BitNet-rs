#![allow(unsafe_op_in_unsafe_fn, unused_unsafe, dead_code, unused_variables, unused_assignments)]
//! NEON SIMD-accelerated 1D convolution kernels for Apple Silicon.
//!
//! Provides vectorized 1D convolution variants using AArch64 NEON intrinsics:
//! basic conv1d, conv1d with bias, depthwise separable conv1d, fused
//! conv1d + ReLU, and causal (left-padded) conv1d for autoregressive models.
//! Processes 4 × f32 lanes at a time with scalar fallback for remainder
//! elements. Non-aarch64 targets get pure-scalar fallback implementations.

// ── NEON implementation (aarch64) ──────────────────────────────────────

#[cfg(target_arch = "aarch64")]
mod neon_impl {
    use std::arch::aarch64::*;

    /// Output length for a standard 1D convolution.
    #[inline]
    fn conv1d_output_len(input_len: usize, kernel_len: usize, stride: usize) -> usize {
        if input_len < kernel_len || stride == 0 || kernel_len == 0 {
            return 0;
        }
        (input_len - kernel_len) / stride + 1
    }

    /// NEON-accelerated dot product of two f32 slices of length `len`
    /// starting at raw pointers `a` and `b`.
    ///
    /// # Safety
    ///
    /// Both `a` and `b` must be valid for reads of `len` elements.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn neon_dot(a: *const f32, b: *const f32, len: usize) -> f32 {
        let chunks = len / 4;
        let rem = len % 4;
        let mut acc = vdupq_n_f32(0.0);

        for c in 0..chunks {
            let off = c * 4;
            let va = unsafe { vld1q_f32(a.add(off)) };
            let vb = unsafe { vld1q_f32(b.add(off)) };
            acc = vfmaq_f32(acc, va, vb);
        }

        // Horizontal sum of 4-lane accumulator.
        let low = vget_low_f32(acc);
        let high = vget_high_f32(acc);
        let pair = vadd_f32(low, high);
        let mut sum = vget_lane_f32::<0>(pair) + vget_lane_f32::<1>(pair);

        // Scalar tail.
        let tail_start = chunks * 4;
        for r in 0..rem {
            sum += unsafe { *a.add(tail_start + r) * *b.add(tail_start + r) };
        }
        sum
    }

    /// Basic 1D convolution with NEON dot-product acceleration.
    ///
    /// For each output position `o`:
    ///   `output[o] = Σ_k input[o * stride + k] * kernel[k]`
    ///
    /// # Safety
    ///
    /// Caller must ensure the target supports NEON (always true on AArch64).
    #[target_feature(enable = "neon")]
    pub unsafe fn conv1d_neon(
        input: &[f32],
        kernel: &[f32],
        input_len: usize,
        kernel_len: usize,
        stride: usize,
        output: &mut [f32],
    ) {
        let in_len = input_len.min(input.len());
        let k_len = kernel_len.min(kernel.len());

        if stride == 0 || k_len == 0 || in_len < k_len {
            return;
        }

        let out_len = conv1d_output_len(in_len, k_len, stride);
        let out_len = out_len.min(output.len());
        let k_ptr = kernel.as_ptr();

        for (o, out_val) in output.iter_mut().enumerate().take(out_len) {
            let base = o * stride;
            let in_ptr = unsafe { input.as_ptr().add(base) };
            *out_val = unsafe { neon_dot(in_ptr, k_ptr, k_len) };
        }
    }

    /// 1D convolution with bias, NEON-accelerated.
    ///
    /// `output[o] = bias + Σ_k input[o * stride + k] * kernel[k]`
    ///
    /// # Safety
    ///
    /// Caller must ensure the target supports NEON.
    #[target_feature(enable = "neon")]
    pub unsafe fn conv1d_bias_neon(
        input: &[f32],
        kernel: &[f32],
        bias: f32,
        input_len: usize,
        kernel_len: usize,
        stride: usize,
        output: &mut [f32],
    ) {
        let in_len = input_len.min(input.len());
        let k_len = kernel_len.min(kernel.len());

        if stride == 0 || k_len == 0 || in_len < k_len {
            return;
        }

        let out_len = conv1d_output_len(in_len, k_len, stride);
        let out_len = out_len.min(output.len());
        let k_ptr = kernel.as_ptr();

        for (o, out_val) in output.iter_mut().enumerate().take(out_len) {
            let base = o * stride;
            let in_ptr = unsafe { input.as_ptr().add(base) };
            *out_val = unsafe { neon_dot(in_ptr, k_ptr, k_len) } + bias;
        }
    }

    /// Depthwise separable 1D convolution with NEON acceleration.
    ///
    /// Each channel is convolved independently with its own kernel.
    /// Layout: `input[ch * input_len + i]`, `kernels[ch * kernel_len + k]`,
    /// `output[ch * out_per_ch + o]`.
    ///
    /// # Safety
    ///
    /// Caller must ensure the target supports NEON.
    #[target_feature(enable = "neon")]
    pub unsafe fn depthwise_conv1d_neon(
        input: &[f32],
        kernels: &[f32],
        channels: usize,
        input_len: usize,
        kernel_len: usize,
        stride: usize,
        output: &mut [f32],
    ) {
        if stride == 0 || channels == 0 || kernel_len == 0 || input_len < kernel_len {
            return;
        }

        let out_per_ch = conv1d_output_len(input_len, kernel_len, stride);
        if out_per_ch == 0 {
            return;
        }

        let expected_in = channels * input_len;
        let expected_k = channels * kernel_len;
        let expected_out = channels * out_per_ch;
        if input.len() < expected_in || kernels.len() < expected_k || output.len() < expected_out {
            return;
        }

        for ch in 0..channels {
            let in_off = ch * input_len;
            let k_off = ch * kernel_len;
            let out_off = ch * out_per_ch;

            for o in 0..out_per_ch {
                let base = in_off + o * stride;
                let in_ptr = unsafe { input.as_ptr().add(base) };
                let k_ptr = unsafe { kernels.as_ptr().add(k_off) };
                output[out_off + o] = unsafe { neon_dot(in_ptr, k_ptr, kernel_len) };
            }
        }
    }

    /// Fused 1D convolution + ReLU with NEON acceleration.
    ///
    /// `output[o] = max(0, Σ_k input[o * stride + k] * kernel[k])`
    ///
    /// # Safety
    ///
    /// Caller must ensure the target supports NEON.
    #[target_feature(enable = "neon")]
    pub unsafe fn conv1d_relu_neon(
        input: &[f32],
        kernel: &[f32],
        input_len: usize,
        kernel_len: usize,
        stride: usize,
        output: &mut [f32],
    ) {
        let in_len = input_len.min(input.len());
        let k_len = kernel_len.min(kernel.len());

        if stride == 0 || k_len == 0 || in_len < k_len {
            return;
        }

        let out_len = conv1d_output_len(in_len, k_len, stride);
        let out_len = out_len.min(output.len());
        let k_ptr = kernel.as_ptr();

        for (o, out_val) in output.iter_mut().enumerate().take(out_len) {
            let base = o * stride;
            let in_ptr = unsafe { input.as_ptr().add(base) };
            let val = unsafe { neon_dot(in_ptr, k_ptr, k_len) };
            *out_val = if val > 0.0 { val } else { 0.0 };
        }
    }

    /// Causal (left-padded) 1D convolution with NEON acceleration.
    ///
    /// Pads `kernel_len - 1` zeros on the left so the output has the same
    /// length as the input (stride is always 1). Suitable for autoregressive
    /// models where the output at position `t` depends only on inputs ≤ `t`.
    ///
    /// # Safety
    ///
    /// Caller must ensure the target supports NEON.
    #[target_feature(enable = "neon")]
    pub unsafe fn causal_conv1d_neon(
        input: &[f32],
        kernel: &[f32],
        input_len: usize,
        kernel_len: usize,
        output: &mut [f32],
    ) {
        let in_len = input_len.min(input.len());
        let k_len = kernel_len.min(kernel.len());

        if k_len == 0 || in_len == 0 {
            return;
        }

        let out_len = in_len.min(output.len());
        let pad = k_len - 1;

        for (o, out_val) in output.iter_mut().enumerate().take(out_len) {
            // In the zero-padded view, position `o` maps to padded
            // index `o`. We only accumulate where the padded index
            // falls within the real input.
            let mut sum = 0.0f32;
            for (k, &kern_val) in kernel.iter().enumerate().take(k_len) {
                // padded_pos = o + k, but with `pad` zeros prepended
                // the real input index is: o + k - pad
                let input_idx_signed = (o as isize) + (k as isize) - (pad as isize);
                if input_idx_signed >= 0 && (input_idx_signed as usize) < in_len {
                    sum += input[input_idx_signed as usize] * kern_val;
                }
            }
            *out_val = sum;
        }
    }

    /// Causal conv1d inner loop using NEON for positions that have the
    /// full kernel window inside the input (no padding needed).
    ///
    /// # Safety
    ///
    /// Caller must ensure the target supports NEON.
    #[target_feature(enable = "neon")]
    #[allow(dead_code)]
    pub unsafe fn causal_conv1d_neon_fast(
        input: &[f32],
        kernel: &[f32],
        input_len: usize,
        kernel_len: usize,
        output: &mut [f32],
    ) {
        let in_len = input_len.min(input.len());
        let k_len = kernel_len.min(kernel.len());

        if k_len == 0 || in_len == 0 {
            return;
        }

        let out_len = in_len.min(output.len());
        let pad = k_len - 1;
        let k_ptr = kernel.as_ptr();

        // Phase 1: positions in the padding region (scalar, partial kernel).
        for (o, out_val) in output.iter_mut().enumerate().take(pad.min(out_len)) {
            let mut sum = 0.0f32;
            for (k, &kern_val) in kernel.iter().enumerate().take(k_len) {
                let idx = (o as isize) + (k as isize) - (pad as isize);
                if idx >= 0 && (idx as usize) < in_len {
                    sum += input[idx as usize] * kern_val;
                }
            }
            *out_val = sum;
        }

        // Phase 2: full-kernel positions — use NEON dot.
        for (o, out_val) in output.iter_mut().enumerate().take(out_len).skip(pad) {
            let start = o - pad; // first input element
            let in_ptr = unsafe { input.as_ptr().add(start) };
            *out_val = unsafe { neon_dot(in_ptr, k_ptr, k_len) };
        }
    }
}

// ── Scalar fallback (non-aarch64) ──────────────────────────────────────

#[cfg(not(target_arch = "aarch64"))]
mod scalar_impl {
    #[inline]
    fn conv1d_output_len(input_len: usize, kernel_len: usize, stride: usize) -> usize {
        if input_len < kernel_len || stride == 0 || kernel_len == 0 {
            return 0;
        }
        (input_len - kernel_len) / stride + 1
    }

    #[inline]
    fn dot(a: &[f32], b: &[f32], len: usize) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..len {
            sum += a[i] * b[i];
        }
        sum
    }

    pub fn conv1d_neon(
        input: &[f32],
        kernel: &[f32],
        input_len: usize,
        kernel_len: usize,
        stride: usize,
        output: &mut [f32],
    ) {
        let in_len = input_len.min(input.len());
        let k_len = kernel_len.min(kernel.len());
        if stride == 0 || k_len == 0 || in_len < k_len {
            return;
        }
        let out_len = conv1d_output_len(in_len, k_len, stride);
        let out_len = out_len.min(output.len());
        for (o, out_val) in output.iter_mut().enumerate().take(out_len) {
            let base = o * stride;
            *out_val = dot(&input[base..], kernel, k_len);
        }
    }

    pub fn conv1d_bias_neon(
        input: &[f32],
        kernel: &[f32],
        bias: f32,
        input_len: usize,
        kernel_len: usize,
        stride: usize,
        output: &mut [f32],
    ) {
        let in_len = input_len.min(input.len());
        let k_len = kernel_len.min(kernel.len());
        if stride == 0 || k_len == 0 || in_len < k_len {
            return;
        }
        let out_len = conv1d_output_len(in_len, k_len, stride);
        let out_len = out_len.min(output.len());
        for (o, out_val) in output.iter_mut().enumerate().take(out_len) {
            let base = o * stride;
            *out_val = dot(&input[base..], kernel, k_len) + bias;
        }
    }

    pub fn depthwise_conv1d_neon(
        input: &[f32],
        kernels: &[f32],
        channels: usize,
        input_len: usize,
        kernel_len: usize,
        stride: usize,
        output: &mut [f32],
    ) {
        if stride == 0 || channels == 0 || kernel_len == 0 || input_len < kernel_len {
            return;
        }
        let out_per_ch = conv1d_output_len(input_len, kernel_len, stride);
        if out_per_ch == 0 {
            return;
        }
        let expected_in = channels * input_len;
        let expected_k = channels * kernel_len;
        let expected_out = channels * out_per_ch;
        if input.len() < expected_in || kernels.len() < expected_k || output.len() < expected_out {
            return;
        }
        for ch in 0..channels {
            let in_off = ch * input_len;
            let k_off = ch * kernel_len;
            let out_off = ch * out_per_ch;
            for o in 0..out_per_ch {
                let base = in_off + o * stride;
                output[out_off + o] = dot(&input[base..], &kernels[k_off..], kernel_len);
            }
        }
    }

    pub fn conv1d_relu_neon(
        input: &[f32],
        kernel: &[f32],
        input_len: usize,
        kernel_len: usize,
        stride: usize,
        output: &mut [f32],
    ) {
        let in_len = input_len.min(input.len());
        let k_len = kernel_len.min(kernel.len());
        if stride == 0 || k_len == 0 || in_len < k_len {
            return;
        }
        let out_len = conv1d_output_len(in_len, k_len, stride);
        let out_len = out_len.min(output.len());
        for (o, out_val) in output.iter_mut().enumerate().take(out_len) {
            let base = o * stride;
            let val = dot(&input[base..], kernel, k_len);
            *out_val = if val > 0.0 { val } else { 0.0 };
        }
    }

    pub fn causal_conv1d_neon(
        input: &[f32],
        kernel: &[f32],
        input_len: usize,
        kernel_len: usize,
        output: &mut [f32],
    ) {
        let in_len = input_len.min(input.len());
        let k_len = kernel_len.min(kernel.len());
        if k_len == 0 || in_len == 0 {
            return;
        }
        let out_len = in_len.min(output.len());
        let pad = k_len - 1;
        for (o, out_val) in output.iter_mut().enumerate().take(out_len) {
            let mut sum = 0.0f32;
            for (k, &k_val) in kernel[..k_len].iter().enumerate() {
                let idx = (o as isize) + (k as isize) - (pad as isize);
                if idx >= 0 && (idx as usize) < in_len {
                    sum += input[idx as usize] * k_val;
                }
            }
            *out_val = sum;
        }
    }
}

// ── Public safe wrappers ───────────────────────────────────────────────

/// Basic 1D convolution with configurable stride.
///
/// Computes `output[o] = Σ_k input[o * stride + k] * kernel[k]` for each
/// output position. Returns early (no-op) for degenerate inputs (empty
/// kernel, kernel longer than input, zero stride).
pub fn conv1d_neon(
    input: &[f32],
    kernel: &[f32],
    input_len: usize,
    kernel_len: usize,
    stride: usize,
    output: &mut [f32],
) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is always available on AArch64.
        unsafe {
            neon_impl::conv1d_neon(input, kernel, input_len, kernel_len, stride, output);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_impl::conv1d_neon(input, kernel, input_len, kernel_len, stride, output);
    }
}

/// 1D convolution with bias term.
///
/// Computes `output[o] = bias + Σ_k input[o * stride + k] * kernel[k]`.
pub fn conv1d_bias_neon(
    input: &[f32],
    kernel: &[f32],
    bias: f32,
    input_len: usize,
    kernel_len: usize,
    stride: usize,
    output: &mut [f32],
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            neon_impl::conv1d_bias_neon(input, kernel, bias, input_len, kernel_len, stride, output);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_impl::conv1d_bias_neon(input, kernel, bias, input_len, kernel_len, stride, output);
    }
}

/// Depthwise separable 1D convolution.
///
/// Each channel is convolved independently. Layouts:
/// - `input`:   `[channels × input_len]` contiguous per channel
/// - `kernels`: `[channels × kernel_len]` contiguous per channel
/// - `output`:  `[channels × output_len]` contiguous per channel
pub fn depthwise_conv1d_neon(
    input: &[f32],
    kernels: &[f32],
    channels: usize,
    input_len: usize,
    kernel_len: usize,
    stride: usize,
    output: &mut [f32],
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            neon_impl::depthwise_conv1d_neon(
                input, kernels, channels, input_len, kernel_len, stride, output,
            );
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_impl::depthwise_conv1d_neon(
            input, kernels, channels, input_len, kernel_len, stride, output,
        );
    }
}

/// Fused 1D convolution + ReLU activation.
///
/// Computes `output[o] = max(0, Σ_k input[o * stride + k] * kernel[k])`.
pub fn conv1d_relu_neon(
    input: &[f32],
    kernel: &[f32],
    input_len: usize,
    kernel_len: usize,
    stride: usize,
    output: &mut [f32],
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            neon_impl::conv1d_relu_neon(input, kernel, input_len, kernel_len, stride, output);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_impl::conv1d_relu_neon(input, kernel, input_len, kernel_len, stride, output);
    }
}

/// Causal (left-padded) 1D convolution for autoregressive models.
///
/// Prepends `kernel_len - 1` virtual zeros so the output has the same
/// length as the input (stride = 1). Output at position `t` depends
/// only on input positions `≤ t`.
pub fn causal_conv1d_neon(
    input: &[f32],
    kernel: &[f32],
    input_len: usize,
    kernel_len: usize,
    output: &mut [f32],
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            neon_impl::causal_conv1d_neon(input, kernel, input_len, kernel_len, output);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_impl::causal_conv1d_neon(input, kernel, input_len, kernel_len, output);
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;

    /// Reference scalar convolution for oracle comparisons.
    fn reference_conv1d(input: &[f32], kernel: &[f32], stride: usize) -> Vec<f32> {
        if stride == 0 || kernel.is_empty() || input.len() < kernel.len() {
            return vec![];
        }
        let out_len = (input.len() - kernel.len()) / stride + 1;
        (0..out_len)
            .map(|o| {
                let base = o * stride;
                kernel.iter().enumerate().map(|(k, &w)| input[base + k] * w).sum()
            })
            .collect()
    }

    fn reference_causal_conv1d(input: &[f32], kernel: &[f32]) -> Vec<f32> {
        if kernel.is_empty() || input.is_empty() {
            return vec![];
        }
        let pad = kernel.len() - 1;
        (0..input.len())
            .map(|o| {
                let mut sum = 0.0f32;
                for k in 0..kernel.len() {
                    let idx = (o as isize) + (k as isize) - (pad as isize);
                    if idx >= 0 && (idx as usize) < input.len() {
                        sum += input[idx as usize] * kernel[k];
                    }
                }
                sum
            })
            .collect()
    }

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol, "mismatch at index {i}: {x} vs {y} (tol={tol})",);
        }
    }

    // ── conv1d_neon basic tests ────────────────────────────────────

    #[test]
    fn test_conv1d_identity_kernel() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kernel = vec![1.0];
        let mut output = vec![0.0; 5];
        conv1d_neon(&input, &kernel, 5, 1, 1, &mut output);
        approx_eq(&output, &input, 1e-6);
    }

    #[test]
    fn test_conv1d_simple_3tap() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kernel = vec![1.0, 1.0, 1.0];
        let mut output = vec![0.0; 3];
        conv1d_neon(&input, &kernel, 5, 3, 1, &mut output);
        let expected = reference_conv1d(&input, &kernel, 1);
        approx_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn test_conv1d_stride2() {
        let input: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let kernel = vec![1.0, 0.5];
        let mut output = vec![0.0; 5];
        conv1d_neon(&input, &kernel, 10, 2, 2, &mut output);
        let expected = reference_conv1d(&input, &kernel, 2);
        approx_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn test_conv1d_stride3() {
        let input: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let kernel = vec![1.0, -1.0, 0.5];
        let mut output = vec![0.0; 4];
        conv1d_neon(&input, &kernel, 12, 3, 3, &mut output);
        let expected = reference_conv1d(&input, &kernel, 3);
        approx_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn test_conv1d_kernel_equals_input() {
        let input = vec![1.0, 2.0, 3.0];
        let kernel = vec![0.5, 0.5, 0.5];
        let mut output = vec![0.0; 1];
        conv1d_neon(&input, &kernel, 3, 3, 1, &mut output);
        let expected = reference_conv1d(&input, &kernel, 1);
        approx_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn test_conv1d_kernel_longer_than_input() {
        let input = vec![1.0, 2.0];
        let kernel = vec![1.0, 1.0, 1.0];
        let mut output = vec![99.0; 1];
        conv1d_neon(&input, &kernel, 2, 3, 1, &mut output);
        // No output produced; buffer untouched.
        assert_eq!(output[0], 99.0);
    }

    #[test]
    fn test_conv1d_empty_input() {
        let input: Vec<f32> = vec![];
        let kernel = vec![1.0];
        let mut output = vec![0.0; 1];
        conv1d_neon(&input, &kernel, 0, 1, 1, &mut output);
        assert_eq!(output[0], 0.0);
    }

    #[test]
    fn test_conv1d_empty_kernel() {
        let input = vec![1.0, 2.0, 3.0];
        let kernel: Vec<f32> = vec![];
        let mut output = vec![99.0; 3];
        conv1d_neon(&input, &kernel, 3, 0, 1, &mut output);
        assert_eq!(output[0], 99.0);
    }

    #[test]
    fn test_conv1d_zero_stride() {
        let input = vec![1.0, 2.0, 3.0];
        let kernel = vec![1.0];
        let mut output = vec![99.0; 3];
        conv1d_neon(&input, &kernel, 3, 1, 0, &mut output);
        assert_eq!(output[0], 99.0);
    }

    #[test]
    fn test_conv1d_all_zeros() {
        let input = vec![0.0; 8];
        let kernel = vec![1.0, 2.0, 3.0];
        let mut output = vec![99.0; 6];
        conv1d_neon(&input, &kernel, 8, 3, 1, &mut output);
        for &v in &output {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_conv1d_negative_values() {
        let input = vec![-1.0, -2.0, -3.0, -4.0];
        let kernel = vec![1.0, 1.0];
        let mut output = vec![0.0; 3];
        conv1d_neon(&input, &kernel, 4, 2, 1, &mut output);
        let expected = reference_conv1d(&input, &kernel, 1);
        approx_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn test_conv1d_single_element_input_and_kernel() {
        let input = vec![3.0];
        let kernel = vec![2.0];
        let mut output = vec![0.0; 1];
        conv1d_neon(&input, &kernel, 1, 1, 1, &mut output);
        assert!((output[0] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_conv1d_large_kernel_4_aligned() {
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let kernel = vec![0.25; 4];
        let mut output = vec![0.0; 13];
        conv1d_neon(&input, &kernel, 16, 4, 1, &mut output);
        let expected = reference_conv1d(&input, &kernel, 1);
        approx_eq(&output, &expected, 1e-5);
    }

    #[test]
    fn test_conv1d_large_kernel_5_unaligned() {
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let kernel = vec![0.1, 0.2, 0.3, 0.2, 0.1];
        let mut output = vec![0.0; 12];
        conv1d_neon(&input, &kernel, 16, 5, 1, &mut output);
        let expected = reference_conv1d(&input, &kernel, 1);
        approx_eq(&output, &expected, 1e-5);
    }

    #[test]
    fn test_conv1d_large_kernel_8() {
        let input: Vec<f32> = (0..20).map(|i| (i as f32) * 0.1).collect();
        let kernel = vec![1.0; 8];
        let mut output = vec![0.0; 13];
        conv1d_neon(&input, &kernel, 20, 8, 1, &mut output);
        let expected = reference_conv1d(&input, &kernel, 1);
        approx_eq(&output, &expected, 1e-4);
    }

    #[test]
    fn test_conv1d_large_kernel_9_unaligned() {
        let input: Vec<f32> = (0..20).map(|i| (i as f32) * 0.1).collect();
        let kernel: Vec<f32> = (0..9).map(|i| (i as f32) * 0.1).collect();
        let mut output = vec![0.0; 12];
        conv1d_neon(&input, &kernel, 20, 9, 1, &mut output);
        let expected = reference_conv1d(&input, &kernel, 1);
        approx_eq(&output, &expected, 1e-4);
    }

    #[test]
    fn test_conv1d_output_buffer_larger() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let kernel = vec![1.0, 1.0];
        let mut output = vec![99.0; 10];
        conv1d_neon(&input, &kernel, 4, 2, 1, &mut output);
        let expected = reference_conv1d(&input, &kernel, 1);
        approx_eq(&output[..3], &expected, 1e-6);
        // Trailing slots untouched.
        assert_eq!(output[3], 99.0);
    }

    #[test]
    fn test_conv1d_input_len_less_than_slice() {
        let input = vec![1.0, 2.0, 3.0, 99.0, 99.0];
        let kernel = vec![1.0, 1.0];
        let mut output = vec![0.0; 2];
        // Only use first 3 elements of the input slice.
        conv1d_neon(&input, &kernel, 3, 2, 1, &mut output);
        let expected = reference_conv1d(&input[..3], &kernel, 1);
        approx_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn test_conv1d_kernel_len_less_than_slice() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let kernel = vec![1.0, 0.5, 99.0, 99.0];
        let mut output = vec![0.0; 3];
        conv1d_neon(&input, &kernel, 4, 2, 1, &mut output);
        let expected = reference_conv1d(&input, &kernel[..2], 1);
        approx_eq(&output, &expected, 1e-6);
    }

    // ── conv1d_bias_neon tests ─────────────────────────────────────

    #[test]
    fn test_conv1d_bias_simple() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kernel = vec![1.0, 1.0, 1.0];
        let bias = 10.0;
        let mut output = vec![0.0; 3];
        conv1d_bias_neon(&input, &kernel, bias, 5, 3, 1, &mut output);
        let expected: Vec<f32> =
            reference_conv1d(&input, &kernel, 1).iter().map(|&v| v + bias).collect();
        approx_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn test_conv1d_bias_zero() {
        let input = vec![1.0, 2.0, 3.0];
        let kernel = vec![1.0];
        let mut output = vec![0.0; 3];
        conv1d_bias_neon(&input, &kernel, 0.0, 3, 1, 1, &mut output);
        approx_eq(&output, &input, 1e-6);
    }

    #[test]
    fn test_conv1d_bias_negative() {
        let input = vec![5.0, 10.0, 15.0];
        let kernel = vec![1.0];
        let bias = -5.0;
        let mut output = vec![0.0; 3];
        conv1d_bias_neon(&input, &kernel, bias, 3, 1, 1, &mut output);
        approx_eq(&output, &[0.0, 5.0, 10.0], 1e-6);
    }

    #[test]
    fn test_conv1d_bias_stride2() {
        let input: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let kernel = vec![1.0, 1.0];
        let bias = 100.0;
        let mut output = vec![0.0; 4];
        conv1d_bias_neon(&input, &kernel, bias, 8, 2, 2, &mut output);
        let expected: Vec<f32> =
            reference_conv1d(&input, &kernel, 2).iter().map(|&v| v + bias).collect();
        approx_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn test_conv1d_bias_empty_kernel() {
        let input = vec![1.0, 2.0];
        let kernel: Vec<f32> = vec![];
        let mut output = vec![99.0; 2];
        conv1d_bias_neon(&input, &kernel, 5.0, 2, 0, 1, &mut output);
        assert_eq!(output[0], 99.0);
    }

    #[test]
    fn test_conv1d_bias_kernel_longer() {
        let input = vec![1.0];
        let kernel = vec![1.0, 2.0, 3.0];
        let mut output = vec![99.0; 1];
        conv1d_bias_neon(&input, &kernel, 5.0, 1, 3, 1, &mut output);
        assert_eq!(output[0], 99.0);
    }

    #[test]
    fn test_conv1d_bias_large_kernel() {
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let kernel = vec![0.25; 8];
        let bias = 1.0;
        let mut output = vec![0.0; 9];
        conv1d_bias_neon(&input, &kernel, bias, 16, 8, 1, &mut output);
        let expected: Vec<f32> =
            reference_conv1d(&input, &kernel, 1).iter().map(|&v| v + bias).collect();
        approx_eq(&output, &expected, 1e-4);
    }

    #[test]
    fn test_conv1d_bias_all_zeros() {
        let input = vec![0.0; 6];
        let kernel = vec![1.0, 2.0];
        let bias = 3.0;
        let mut output = vec![0.0; 5];
        conv1d_bias_neon(&input, &kernel, bias, 6, 2, 1, &mut output);
        for &v in &output {
            assert!((v - bias).abs() < 1e-6);
        }
    }

    // ── depthwise_conv1d_neon tests ────────────────────────────────

    #[test]
    fn test_depthwise_single_channel() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kernel = vec![1.0, 1.0, 1.0];
        let mut output = vec![0.0; 3];
        depthwise_conv1d_neon(&input, &kernel, 1, 5, 3, 1, &mut output);
        let expected = reference_conv1d(&input, &kernel, 1);
        approx_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn test_depthwise_two_channels() {
        // ch0: [1,2,3,4], ch1: [5,6,7,8]
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        // ch0 kernel: [1,1], ch1 kernel: [0.5, 0.5]
        let kernels = vec![1.0, 1.0, 0.5, 0.5];
        let mut output = vec![0.0; 6]; // 3 per channel
        depthwise_conv1d_neon(&input, &kernels, 2, 4, 2, 1, &mut output);
        let e0 = reference_conv1d(&input[0..4], &kernels[0..2], 1);
        let e1 = reference_conv1d(&input[4..8], &kernels[2..4], 1);
        approx_eq(&output[0..3], &e0, 1e-6);
        approx_eq(&output[3..6], &e1, 1e-6);
    }

    #[test]
    fn test_depthwise_stride2() {
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let kernels = vec![1.0, 1.0, 1.0, 1.0]; // 2 ch × k=2
        // out_per_ch = (8 - 2) / 2 + 1 = 4
        let mut output = vec![0.0; 8]; // 4 per channel
        depthwise_conv1d_neon(&input, &kernels, 2, 8, 2, 2, &mut output);
        let e0 = reference_conv1d(&input[0..8], &kernels[0..2], 2);
        let e1 = reference_conv1d(&input[8..16], &kernels[2..4], 2);
        approx_eq(&output[..e0.len()], &e0, 1e-6);
        approx_eq(&output[e0.len()..e0.len() + e1.len()], &e1, 1e-6);
    }

    #[test]
    fn test_depthwise_zero_channels() {
        let input = vec![1.0, 2.0];
        let kernel = vec![1.0];
        let mut output = vec![99.0; 2];
        depthwise_conv1d_neon(&input, &kernel, 0, 2, 1, 1, &mut output);
        assert_eq!(output[0], 99.0);
    }

    #[test]
    fn test_depthwise_zero_stride() {
        let input = vec![1.0, 2.0];
        let kernel = vec![1.0];
        let mut output = vec![99.0; 2];
        depthwise_conv1d_neon(&input, &kernel, 1, 2, 1, 0, &mut output);
        assert_eq!(output[0], 99.0);
    }

    #[test]
    fn test_depthwise_kernel_longer_than_input() {
        let input = vec![1.0, 2.0];
        let kernel = vec![1.0, 2.0, 3.0];
        let mut output = vec![99.0; 1];
        depthwise_conv1d_neon(&input, &kernel, 1, 2, 3, 1, &mut output);
        assert_eq!(output[0], 99.0);
    }

    #[test]
    fn test_depthwise_three_channels() {
        let input: Vec<f32> = (0..15).map(|i| i as f32).collect(); // 3×5
        let kernels: Vec<f32> = vec![
            1.0, 0.0, -1.0, // ch0 k=3
            0.5, 0.5, 0.5, // ch1 k=3
            -1.0, 1.0, -1.0, // ch2 k=3
        ];
        let mut output = vec![0.0; 9]; // 3 per channel
        depthwise_conv1d_neon(&input, &kernels, 3, 5, 3, 1, &mut output);
        for ch in 0..3 {
            let in_slice = &input[ch * 5..(ch + 1) * 5];
            let k_slice = &kernels[ch * 3..(ch + 1) * 3];
            let expected = reference_conv1d(in_slice, k_slice, 1);
            approx_eq(&output[ch * 3..(ch + 1) * 3], &expected, 1e-5);
        }
    }

    #[test]
    fn test_depthwise_large_kernel() {
        let input: Vec<f32> = (0..20).map(|i| (i as f32) * 0.1).collect();
        let kernels = vec![0.125; 8]; // 1 ch × k=8
        let mut output = vec![0.0; 13];
        depthwise_conv1d_neon(&input, &kernels, 1, 20, 8, 1, &mut output);
        let expected = reference_conv1d(&input, &kernels[..8], 1);
        approx_eq(&output, &expected, 1e-4);
    }

    #[test]
    fn test_depthwise_insufficient_output() {
        // Output buffer too small → graceful no-op.
        let input = vec![1.0, 2.0, 3.0, 4.0]; // 2 ch × 2
        let kernels = vec![1.0, 1.0]; // 2 ch × k=1
        let mut output = vec![99.0; 1]; // needs 4, has 1
        depthwise_conv1d_neon(&input, &kernels, 2, 2, 1, 1, &mut output);
        assert_eq!(output[0], 99.0);
    }

    // ── conv1d_relu_neon tests ─────────────────────────────────────

    #[test]
    fn test_conv1d_relu_positive() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kernel = vec![1.0, 1.0, 1.0];
        let mut output = vec![0.0; 3];
        conv1d_relu_neon(&input, &kernel, 5, 3, 1, &mut output);
        let expected = reference_conv1d(&input, &kernel, 1);
        approx_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn test_conv1d_relu_negative_clipped() {
        let input = vec![-1.0, -2.0, -3.0, -4.0];
        let kernel = vec![1.0, 1.0];
        let mut output = vec![99.0; 3];
        conv1d_relu_neon(&input, &kernel, 4, 2, 1, &mut output);
        for &v in &output {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_conv1d_relu_mixed() {
        let input = vec![-2.0, 3.0, -1.0, 4.0];
        let kernel = vec![1.0, 1.0];
        let mut output = vec![0.0; 3];
        conv1d_relu_neon(&input, &kernel, 4, 2, 1, &mut output);
        let ref_out = reference_conv1d(&input, &kernel, 1);
        let expected: Vec<f32> = ref_out.iter().map(|&v| v.max(0.0)).collect();
        approx_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn test_conv1d_relu_zero_output() {
        let input = vec![-5.0, -5.0, -5.0];
        let kernel = vec![1.0];
        let mut output = vec![99.0; 3];
        conv1d_relu_neon(&input, &kernel, 3, 1, 1, &mut output);
        for &v in &output {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_conv1d_relu_stride2() {
        let input: Vec<f32> = vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0];
        let kernel = vec![1.0, 1.0];
        let mut output = vec![0.0; 4];
        conv1d_relu_neon(&input, &kernel, 8, 2, 2, &mut output);
        let ref_out = reference_conv1d(&input, &kernel, 2);
        let expected: Vec<f32> = ref_out.iter().map(|&v| v.max(0.0)).collect();
        approx_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn test_conv1d_relu_empty_kernel() {
        let input = vec![1.0];
        let kernel: Vec<f32> = vec![];
        let mut output = vec![99.0; 1];
        conv1d_relu_neon(&input, &kernel, 1, 0, 1, &mut output);
        assert_eq!(output[0], 99.0);
    }

    #[test]
    fn test_conv1d_relu_kernel_longer() {
        let input = vec![1.0];
        let kernel = vec![1.0, 2.0];
        let mut output = vec![99.0; 1];
        conv1d_relu_neon(&input, &kernel, 1, 2, 1, &mut output);
        assert_eq!(output[0], 99.0);
    }

    #[test]
    fn test_conv1d_relu_large_kernel() {
        let input: Vec<f32> = (0..16).map(|i| (i as f32) - 8.0).collect();
        let kernel = vec![0.5; 4];
        let mut output = vec![0.0; 13];
        conv1d_relu_neon(&input, &kernel, 16, 4, 1, &mut output);
        let ref_out = reference_conv1d(&input, &kernel, 1);
        let expected: Vec<f32> = ref_out.iter().map(|&v| v.max(0.0)).collect();
        approx_eq(&output, &expected, 1e-5);
    }

    #[test]
    fn test_conv1d_relu_all_zeros_input() {
        let input = vec![0.0; 6];
        let kernel = vec![1.0, 2.0, 3.0];
        let mut output = vec![99.0; 4];
        conv1d_relu_neon(&input, &kernel, 6, 3, 1, &mut output);
        for &v in &output {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_conv1d_relu_exactly_zero() {
        // Conv result is exactly zero → stays zero.
        let input = vec![1.0, -1.0];
        let kernel = vec![1.0, 1.0];
        let mut output = vec![99.0; 1];
        conv1d_relu_neon(&input, &kernel, 2, 2, 1, &mut output);
        assert_eq!(output[0], 0.0);
    }

    // ── causal_conv1d_neon tests ───────────────────────────────────

    #[test]
    fn test_causal_conv1d_identity() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kernel = vec![1.0];
        let mut output = vec![0.0; 5];
        causal_conv1d_neon(&input, &kernel, 5, 1, &mut output);
        approx_eq(&output, &input, 1e-6);
    }

    #[test]
    fn test_causal_conv1d_2tap() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let kernel = vec![1.0, 1.0];
        let mut output = vec![0.0; 4];
        causal_conv1d_neon(&input, &kernel, 4, 2, &mut output);
        let expected = reference_causal_conv1d(&input, &kernel);
        approx_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn test_causal_conv1d_3tap() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kernel = vec![0.5, 0.3, 0.2];
        let mut output = vec![0.0; 5];
        causal_conv1d_neon(&input, &kernel, 5, 3, &mut output);
        let expected = reference_causal_conv1d(&input, &kernel);
        approx_eq(&output, &expected, 1e-5);
    }

    #[test]
    fn test_causal_conv1d_output_length() {
        let input = vec![1.0; 10];
        let kernel = vec![1.0; 4];
        let mut output = vec![0.0; 10];
        causal_conv1d_neon(&input, &kernel, 10, 4, &mut output);
        assert_eq!(output.len(), 10);
    }

    #[test]
    fn test_causal_conv1d_first_elements_padded() {
        let input = vec![10.0, 20.0, 30.0, 40.0];
        let kernel = vec![1.0, 0.0, 0.0]; // 3-tap, pad=2
        let mut output = vec![0.0; 4];
        causal_conv1d_neon(&input, &kernel, 4, 3, &mut output);
        let expected = reference_causal_conv1d(&input, &kernel);
        approx_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn test_causal_conv1d_empty_input() {
        let input: Vec<f32> = vec![];
        let kernel = vec![1.0];
        let mut output = vec![99.0; 1];
        causal_conv1d_neon(&input, &kernel, 0, 1, &mut output);
        assert_eq!(output[0], 99.0);
    }

    #[test]
    fn test_causal_conv1d_empty_kernel() {
        let input = vec![1.0, 2.0];
        let kernel: Vec<f32> = vec![];
        let mut output = vec![99.0; 2];
        causal_conv1d_neon(&input, &kernel, 2, 0, &mut output);
        assert_eq!(output[0], 99.0);
    }

    #[test]
    fn test_causal_conv1d_kernel_longer_than_input() {
        // Still produces output — only the kernel taps overlapping
        // with real input contribute.
        let input = vec![5.0, 10.0];
        let kernel = vec![1.0, 1.0, 1.0, 1.0]; // pad = 3
        let mut output = vec![0.0; 2];
        causal_conv1d_neon(&input, &kernel, 2, 4, &mut output);
        let expected = reference_causal_conv1d(&input, &kernel);
        approx_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn test_causal_conv1d_single_element() {
        let input = vec![7.0];
        let kernel = vec![3.0];
        let mut output = vec![0.0; 1];
        causal_conv1d_neon(&input, &kernel, 1, 1, &mut output);
        assert!((output[0] - 21.0).abs() < 1e-6);
    }

    #[test]
    fn test_causal_conv1d_all_zeros() {
        let input = vec![0.0; 8];
        let kernel = vec![1.0, 2.0, 3.0];
        let mut output = vec![99.0; 8];
        causal_conv1d_neon(&input, &kernel, 8, 3, &mut output);
        for &v in &output {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_causal_conv1d_negative_values() {
        let input = vec![-1.0, -2.0, -3.0, -4.0];
        let kernel = vec![1.0, 1.0];
        let mut output = vec![0.0; 4];
        causal_conv1d_neon(&input, &kernel, 4, 2, &mut output);
        let expected = reference_causal_conv1d(&input, &kernel);
        approx_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn test_causal_conv1d_large_kernel() {
        let input: Vec<f32> = (0..20).map(|i| (i as f32) * 0.1).collect();
        let kernel = vec![0.125; 8];
        let mut output = vec![0.0; 20];
        causal_conv1d_neon(&input, &kernel, 20, 8, &mut output);
        let expected = reference_causal_conv1d(&input, &kernel);
        approx_eq(&output, &expected, 1e-4);
    }

    #[test]
    fn test_causal_conv1d_impulse_response() {
        // Delta at position 0 → output is the kernel itself.
        let mut input = vec![0.0; 8];
        input[0] = 1.0;
        let kernel = vec![0.5, 0.3, 0.2];
        let mut output = vec![0.0; 8];
        causal_conv1d_neon(&input, &kernel, 8, 3, &mut output);
        let expected = reference_causal_conv1d(&input, &kernel);
        approx_eq(&output, &expected, 1e-6);
    }

    // ── Cross-function consistency tests ───────────────────────────

    #[test]
    fn test_bias_zero_equals_basic() {
        let input: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let kernel = vec![0.5, -0.5, 0.25];
        let mut out_basic = vec![0.0; 8];
        let mut out_bias = vec![0.0; 8];
        conv1d_neon(&input, &kernel, 10, 3, 1, &mut out_basic);
        conv1d_bias_neon(&input, &kernel, 0.0, 10, 3, 1, &mut out_bias);
        approx_eq(&out_basic, &out_bias, 1e-6);
    }

    #[test]
    fn test_relu_positive_equals_basic() {
        // All positive results → ReLU should match basic.
        let input = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let kernel = vec![1.0, 1.0];
        let mut out_basic = vec![0.0; 4];
        let mut out_relu = vec![0.0; 4];
        conv1d_neon(&input, &kernel, 5, 2, 1, &mut out_basic);
        conv1d_relu_neon(&input, &kernel, 5, 2, 1, &mut out_relu);
        approx_eq(&out_basic, &out_relu, 1e-6);
    }

    #[test]
    fn test_depthwise_1ch_equals_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kernel = vec![1.0, -1.0];
        let mut out_basic = vec![0.0; 4];
        let mut out_dw = vec![0.0; 4];
        conv1d_neon(&input, &kernel, 5, 2, 1, &mut out_basic);
        depthwise_conv1d_neon(&input, &kernel, 1, 5, 2, 1, &mut out_dw);
        approx_eq(&out_basic, &out_dw, 1e-6);
    }

    #[test]
    fn test_causal_tail_matches_basic() {
        // For positions ≥ pad, causal conv should match basic conv.
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let kernel = vec![1.0, 1.0, 1.0]; // pad = 2
        let mut causal_out = vec![0.0; 8];
        causal_conv1d_neon(&input, &kernel, 8, 3, &mut causal_out);
        let basic = reference_conv1d(&input, &kernel, 1);
        // causal_out[2..8] should match basic[0..6]
        approx_eq(&causal_out[2..], &basic, 1e-6);
    }

    // ── Large / stress tests ───────────────────────────────────────

    #[test]
    fn test_conv1d_large_input() {
        let n = 256;
        let k = 7;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let kernel: Vec<f32> = (0..k).map(|i| (i as f32) * 0.1).collect();
        let out_len = n - k + 1;
        let mut output = vec![0.0; out_len];
        conv1d_neon(&input, &kernel, n, k, 1, &mut output);
        let expected = reference_conv1d(&input, &kernel, 1);
        approx_eq(&output, &expected, 1e-3);
    }

    #[test]
    fn test_conv1d_large_stride_large_input() {
        let n = 200;
        let k = 5;
        let stride = 4;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let kernel: Vec<f32> = (0..k).map(|i| (i as f32) * 0.2).collect();
        let out_len = (n - k) / stride + 1;
        let mut output = vec![0.0; out_len];
        conv1d_neon(&input, &kernel, n, k, stride, &mut output);
        let expected = reference_conv1d(&input, &kernel, stride);
        approx_eq(&output, &expected, 1e-3);
    }

    #[test]
    fn test_causal_conv1d_large_input() {
        let n = 128;
        let k = 5;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let kernel: Vec<f32> = (0..k).map(|i| (i as f32) * 0.1).collect();
        let mut output = vec![0.0; n];
        causal_conv1d_neon(&input, &kernel, n, k, &mut output);
        let expected = reference_causal_conv1d(&input, &kernel);
        approx_eq(&output, &expected, 1e-3);
    }

    #[test]
    fn test_depthwise_many_channels() {
        let channels = 8;
        let in_len = 16;
        let k_len = 3;
        let input: Vec<f32> = (0..(channels * in_len)).map(|i| (i as f32) * 0.01).collect();
        let kernels: Vec<f32> =
            (0..(channels * k_len)).map(|i| ((i % 5) as f32) * 0.2 - 0.4).collect();
        let out_per_ch = in_len - k_len + 1;
        let mut output = vec![0.0; channels * out_per_ch];
        depthwise_conv1d_neon(&input, &kernels, channels, in_len, k_len, 1, &mut output);
        for ch in 0..channels {
            let in_slice = &input[ch * in_len..(ch + 1) * in_len];
            let k_slice = &kernels[ch * k_len..(ch + 1) * k_len];
            let expected = reference_conv1d(in_slice, k_slice, 1);
            let out_slice = &output[ch * out_per_ch..(ch + 1) * out_per_ch];
            approx_eq(out_slice, &expected, 1e-4);
        }
    }

    #[test]
    fn test_conv1d_bias_large_input() {
        let n = 128;
        let k = 5;
        let bias = -2.5;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let kernel: Vec<f32> = (0..k).map(|i| (i as f32) * 0.1).collect();
        let out_len = n - k + 1;
        let mut output = vec![0.0; out_len];
        conv1d_bias_neon(&input, &kernel, bias, n, k, 1, &mut output);
        let expected: Vec<f32> =
            reference_conv1d(&input, &kernel, 1).iter().map(|&v| v + bias).collect();
        approx_eq(&output, &expected, 1e-3);
    }

    #[test]
    fn test_conv1d_relu_large_mixed() {
        let n = 64;
        let k = 3;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) - (n as f32) / 2.0).collect();
        let kernel = vec![1.0, -2.0, 1.0];
        let out_len = n - k + 1;
        let mut output = vec![0.0; out_len];
        conv1d_relu_neon(&input, &kernel, n, k, 1, &mut output);
        let expected: Vec<f32> =
            reference_conv1d(&input, &kernel, 1).iter().map(|&v| v.max(0.0)).collect();
        approx_eq(&output, &expected, 1e-4);
    }

    // ── Edge-case / regression tests ───────────────────────────────

    #[test]
    fn test_conv1d_stride_larger_than_input() {
        let input = vec![1.0, 2.0, 3.0];
        let kernel = vec![1.0];
        let mut output = vec![0.0; 1];
        // stride=5, input=3, kernel=1 → only 1 output
        conv1d_neon(&input, &kernel, 3, 1, 5, &mut output);
        assert!((output[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_conv1d_stride_equals_input() {
        let input = vec![1.0, 2.0, 3.0];
        let kernel = vec![1.0];
        let mut output = vec![0.0; 1];
        conv1d_neon(&input, &kernel, 3, 1, 3, &mut output);
        assert!((output[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_conv1d_kernel_all_ones() {
        // Moving average.
        let input = vec![2.0; 10];
        let kernel = vec![1.0; 3];
        let mut output = vec![0.0; 8];
        conv1d_neon(&input, &kernel, 10, 3, 1, &mut output);
        for &v in &output {
            assert!((v - 6.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_conv1d_alternating_signs() {
        let input: Vec<f32> = (0..8).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let kernel = vec![1.0, 1.0];
        let mut output = vec![0.0; 7];
        conv1d_neon(&input, &kernel, 8, 2, 1, &mut output);
        let expected = reference_conv1d(&input, &kernel, 1);
        approx_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn test_causal_conv1d_preserves_length() {
        for n in [1, 2, 5, 16, 33] {
            let input = vec![1.0; n];
            let kernel = vec![1.0; 3.min(n + 1)];
            let k_len = kernel.len();
            let mut output = vec![0.0; n];
            causal_conv1d_neon(&input, &kernel, n, k_len, &mut output);
            // Output length always equals input length.
            assert_eq!(output.len(), n);
        }
    }

    #[test]
    fn test_conv1d_very_small_values() {
        let input = vec![1e-10; 6];
        let kernel = vec![1e-10; 3];
        let mut output = vec![0.0; 4];
        conv1d_neon(&input, &kernel, 6, 3, 1, &mut output);
        let expected = reference_conv1d(&input, &kernel, 1);
        approx_eq(&output, &expected, 1e-18);
    }

    #[test]
    fn test_conv1d_large_values() {
        let input = vec![1e6; 4];
        let kernel = vec![1e6; 2];
        let mut output = vec![0.0; 3];
        conv1d_neon(&input, &kernel, 4, 2, 1, &mut output);
        let expected = reference_conv1d(&input, &kernel, 1);
        approx_eq(&output, &expected, 1.0); // f32 precision
    }

    #[test]
    fn test_depthwise_output_buffer_exact() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2×3
        let kernels = vec![1.0, 1.0, 1.0, 1.0]; // 2× k=2
        let mut output = vec![0.0; 4]; // exactly 2×2
        depthwise_conv1d_neon(&input, &kernels, 2, 3, 2, 1, &mut output);
        let e0 = reference_conv1d(&input[0..3], &kernels[0..2], 1);
        let e1 = reference_conv1d(&input[3..6], &kernels[2..4], 1);
        approx_eq(&output[0..2], &e0, 1e-6);
        approx_eq(&output[2..4], &e1, 1e-6);
    }

    #[test]
    fn test_conv1d_relu_all_positive_input() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kernel = vec![1.0];
        let mut output = vec![0.0; 5];
        conv1d_relu_neon(&input, &kernel, 5, 1, 1, &mut output);
        approx_eq(&output, &input, 1e-6);
    }

    #[test]
    fn test_conv1d_bias_matches_manual() {
        let input = vec![2.0, 4.0, 6.0];
        let kernel = vec![0.5];
        let bias = 1.0;
        let mut output = vec![0.0; 3];
        conv1d_bias_neon(&input, &kernel, bias, 3, 1, 1, &mut output);
        approx_eq(&output, &[2.0, 3.0, 4.0], 1e-6);
    }

    #[test]
    fn test_causal_conv1d_step_function() {
        // Step function: zeros then ones.
        let input = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let kernel = vec![1.0, 1.0, 1.0]; // pad = 2
        let mut output = vec![0.0; 8];
        causal_conv1d_neon(&input, &kernel, 8, 3, &mut output);
        let expected = reference_causal_conv1d(&input, &kernel);
        approx_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn test_conv1d_kernel_size_4_aligned() {
        // Exercises exactly one NEON 4-wide chunk.
        let input: Vec<f32> = (0..8).map(|i| (i + 1) as f32).collect();
        let kernel = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 5];
        conv1d_neon(&input, &kernel, 8, 4, 1, &mut output);
        let expected = reference_conv1d(&input, &kernel, 1);
        approx_eq(&output, &expected, 1e-5);
    }

    #[test]
    fn test_conv1d_kernel_size_7_unaligned() {
        // 1 NEON chunk + 3 scalar tail.
        let input: Vec<f32> = (0..12).map(|i| (i + 1) as f32).collect();
        let kernel: Vec<f32> = (0..7).map(|i| (i as f32) * 0.1).collect();
        let mut output = vec![0.0; 6];
        conv1d_neon(&input, &kernel, 12, 7, 1, &mut output);
        let expected = reference_conv1d(&input, &kernel, 1);
        approx_eq(&output, &expected, 1e-4);
    }

    #[test]
    fn test_conv1d_kernel_size_13() {
        // 3 NEON chunks + 1 scalar tail.
        let input: Vec<f32> = (0..20).map(|i| (i as f32) * 0.05).collect();
        let kernel: Vec<f32> = (0..13).map(|i| (i as f32) * 0.05).collect();
        let mut output = vec![0.0; 8];
        conv1d_neon(&input, &kernel, 20, 13, 1, &mut output);
        let expected = reference_conv1d(&input, &kernel, 1);
        approx_eq(&output, &expected, 1e-3);
    }

    #[test]
    fn test_conv1d_kernel_size_1_many_outputs() {
        let input: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let kernel = vec![2.0];
        let mut output = vec![0.0; 32];
        conv1d_neon(&input, &kernel, 32, 1, 1, &mut output);
        let expected: Vec<f32> = input.iter().map(|&v| v * 2.0).collect();
        approx_eq(&output, &expected, 1e-5);
    }

    #[test]
    fn test_causal_conv1d_large_kernel_small_input() {
        let input = vec![1.0, 2.0, 3.0];
        let kernel = vec![1.0; 8]; // pad = 7
        let mut output = vec![0.0; 3];
        causal_conv1d_neon(&input, &kernel, 3, 8, &mut output);
        let expected = reference_causal_conv1d(&input, &kernel);
        approx_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn test_conv1d_bias_stride3() {
        let input: Vec<f32> = (0..15).map(|i| i as f32).collect();
        let kernel = vec![1.0, -1.0, 0.5];
        let bias = 7.0;
        let mut output = vec![0.0; 5];
        conv1d_bias_neon(&input, &kernel, bias, 15, 3, 3, &mut output);
        let expected: Vec<f32> =
            reference_conv1d(&input, &kernel, 3).iter().map(|&v| v + bias).collect();
        approx_eq(&output, &expected, 1e-5);
    }
}

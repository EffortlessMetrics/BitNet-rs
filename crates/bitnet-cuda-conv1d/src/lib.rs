//! CUDA-accelerated 1D convolution operations for neural network inference.
//!
//! Provides standard, depthwise, grouped, and dilated 1D convolution with
//! multiple padding modes (`Valid`, `Same`, `Full`). The CPU path uses an
//! im2col + GEMM strategy for cache-friendly computation; the GPU path is
//! gated behind `#[cfg(any(feature = "gpu", feature = "cuda"))]`.
//!
//! # Quick start
//!
//! ```
//! use bitnet_cuda_conv1d::{Conv1dConfig, PaddingMode, conv1d};
//!
//! let input  = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
//! let kernel = vec![1.0f32, 0.0, -1.0];
//! let cfg = Conv1dConfig::new(1, 1, 3).with_padding(PaddingMode::Valid);
//! let out = conv1d(&input, &kernel, &cfg);
//! assert_eq!(out.len(), 3);
//! ```

// ── Padding ─────────────────────────────────────────────────────────────

/// Padding mode applied before convolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingMode {
    /// No padding — output length = `(input_len - kernel_size) / stride + 1`.
    Valid,
    /// Zero-pad so the output has the same length as `ceil(input_len / stride)`.
    Same,
    /// Zero-pad so every input element is covered by every kernel position.
    /// Output length = `input_len + kernel_size - 1` (stride = 1, dilation = 1).
    Full,
}

// ── Configuration ───────────────────────────────────────────────────────

/// Describes a 1-D convolution operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Conv1dConfig {
    /// Number of input channels.
    pub in_channels: usize,
    /// Number of output channels.
    pub out_channels: usize,
    /// Spatial size of each kernel.
    pub kernel_size: usize,
    /// Stride of the convolution.
    pub stride: usize,
    /// Dilation factor (spacing between kernel elements).
    pub dilation: usize,
    /// Number of groups for grouped / depthwise convolution.
    pub groups: usize,
    /// Padding mode.
    pub padding: PaddingMode,
}

impl Conv1dConfig {
    /// Create a configuration with sensible defaults (stride=1, dilation=1,
    /// groups=1, `Valid` padding).
    #[must_use]
    pub const fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride: 1,
            dilation: 1,
            groups: 1,
            padding: PaddingMode::Valid,
        }
    }

    /// Set the stride.
    #[must_use]
    pub const fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    /// Set the dilation factor.
    #[must_use]
    pub const fn with_dilation(mut self, dilation: usize) -> Self {
        self.dilation = dilation;
        self
    }

    /// Set the number of groups (use `in_channels` for depthwise).
    #[must_use]
    pub const fn with_groups(mut self, groups: usize) -> Self {
        self.groups = groups;
        self
    }

    /// Set the padding mode.
    #[must_use]
    pub const fn with_padding(mut self, padding: PaddingMode) -> Self {
        self.padding = padding;
        self
    }

    /// Effective kernel size accounting for dilation.
    #[must_use]
    pub const fn effective_kernel_size(&self) -> usize {
        (self.kernel_size - 1) * self.dilation + 1
    }

    /// Number of input channels per group.
    #[must_use]
    pub const fn in_channels_per_group(&self) -> usize {
        self.in_channels / self.groups
    }

    /// Number of output channels per group.
    #[must_use]
    pub const fn out_channels_per_group(&self) -> usize {
        self.out_channels / self.groups
    }

    /// Total padding (both sides combined) required for the chosen mode given
    /// `input_len`.
    #[must_use]
    #[allow(clippy::manual_div_ceil)]
    pub const fn total_padding(&self, input_len: usize) -> usize {
        match self.padding {
            PaddingMode::Valid => 0,
            PaddingMode::Same => {
                let ek = self.effective_kernel_size();
                let out_len = (input_len + self.stride - 1) / self.stride;
                let needed = (out_len - 1) * self.stride + ek;
                needed.saturating_sub(input_len)
            }
            PaddingMode::Full => {
                let ek = self.effective_kernel_size();
                2 * (ek - 1)
            }
        }
    }

    /// Compute the output spatial length for a given `input_len`.
    #[must_use]
    pub const fn output_len(&self, input_len: usize) -> usize {
        let total_pad = self.total_padding(input_len);
        let padded = input_len + total_pad;
        let ek = self.effective_kernel_size();
        if padded < ek {
            return 0;
        }
        (padded - ek) / self.stride + 1
    }
}

// ── im2col helper ───────────────────────────────────────────────────────

/// Build an im2col matrix from a single-channel 1-D signal.
///
/// Returns a column-major matrix of shape `(kernel_size, output_len)` stored
/// row-major (each row is one kernel tap across output positions).
#[must_use]
pub fn im2col_1d(
    input: &[f32],
    input_len: usize,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
    pad_left: usize,
    pad_right: usize,
) -> Vec<f32> {
    let ek = (kernel_size - 1) * dilation + 1;
    let total_pad = pad_left + pad_right;
    let out_len =
        if input_len + total_pad < ek { 0 } else { (input_len + total_pad - ek) / stride + 1 };
    let mut cols = vec![0.0f32; kernel_size * out_len];
    for o in 0..out_len {
        for k in 0..kernel_size {
            let idx = o * stride + k * dilation;
            let signed_idx = idx.cast_signed() - pad_left.cast_signed();
            if signed_idx >= 0 && signed_idx.cast_unsigned() < input_len {
                cols[k * out_len + o] = input[signed_idx.cast_unsigned()];
            }
        }
    }
    cols
}

// ── GEMM (tiny reference) ───────────────────────────────────────────────

/// Minimal row-major `C += A * B` where A is `(m, k)`, B is `(k, n)`.
#[allow(clippy::many_single_char_names)]
fn gemm_add(c: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for p in 0..k {
            let a_val = a[i * k + p];
            for j in 0..n {
                c[i * n + j] = a_val.mul_add(b[p * n + j], c[i * n + j]);
            }
        }
    }
}

// ── Public API: conv1d (im2col + GEMM) ─────────────────────────────────

/// Perform a 1-D convolution on a multi-channel signal.
///
/// ## Layout
///
/// * `input`  – `[in_channels, input_len]` row-major.
/// * `kernel` – `[out_channels, in_channels / groups, kernel_size]` row-major.
/// * Returns  – `[out_channels, output_len]` row-major.
///
/// The function supports grouped and depthwise convolution controlled by
/// `cfg.groups`.
#[must_use]
pub fn conv1d(input: &[f32], kernel: &[f32], cfg: &Conv1dConfig) -> Vec<f32> {
    assert!(cfg.in_channels > 0, "in_channels must be > 0");
    assert!(cfg.out_channels > 0, "out_channels must be > 0");
    assert!(cfg.kernel_size > 0, "kernel_size must be > 0");
    assert!(cfg.stride > 0, "stride must be > 0");
    assert!(cfg.dilation > 0, "dilation must be > 0");
    assert!(cfg.groups > 0, "groups must be > 0");
    assert!(cfg.in_channels.is_multiple_of(cfg.groups), "in_channels must be divisible by groups");
    assert!(
        cfg.out_channels.is_multiple_of(cfg.groups),
        "out_channels must be divisible by groups"
    );

    let in_c_per_g = cfg.in_channels_per_group();
    let out_c_per_g = cfg.out_channels_per_group();
    let input_len = input.len() / cfg.in_channels;
    let out_len = cfg.output_len(input_len);

    if out_len == 0 {
        return Vec::new();
    }

    let total_pad = cfg.total_padding(input_len);
    let pad_left = total_pad / 2;
    let pad_right = total_pad - pad_left;

    let mut output = vec![0.0f32; cfg.out_channels * out_len];

    for g in 0..cfg.groups {
        // Combined im2col block for all input channels in this group.
        let col_rows = in_c_per_g * cfg.kernel_size;
        let mut col = vec![0.0f32; col_rows * out_len];

        for ic in 0..in_c_per_g {
            let global_ic = g * in_c_per_g + ic;
            let channel_data = &input[global_ic * input_len..(global_ic + 1) * input_len];
            let sub = im2col_1d(
                channel_data,
                input_len,
                cfg.kernel_size,
                cfg.stride,
                cfg.dilation,
                pad_left,
                pad_right,
            );
            let row_off = ic * cfg.kernel_size;
            for k in 0..cfg.kernel_size {
                let src_start = k * out_len;
                let dst_start = (row_off + k) * out_len;
                col[dst_start..dst_start + out_len]
                    .copy_from_slice(&sub[src_start..src_start + out_len]);
            }
        }

        // Kernel sub-matrix: (out_c_per_g, in_c_per_g * kernel_size).
        let k_offset = g * out_c_per_g * in_c_per_g * cfg.kernel_size;
        let k_slice = &kernel[k_offset..k_offset + out_c_per_g * in_c_per_g * cfg.kernel_size];

        // GEMM: output_group = k_slice * col → (out_c_per_g, out_len).
        let o_offset = g * out_c_per_g * out_len;
        gemm_add(
            &mut output[o_offset..o_offset + out_c_per_g * out_len],
            k_slice,
            &col,
            out_c_per_g,
            col_rows,
            out_len,
        );
    }

    output
}

// ── Convenience wrappers ────────────────────────────────────────────────

/// Depthwise 1-D convolution (`groups == in_channels == out_channels`).
#[must_use]
pub fn depthwise_conv1d(
    input: &[f32],
    kernel: &[f32],
    channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: PaddingMode,
) -> Vec<f32> {
    let cfg = Conv1dConfig::new(channels, channels, kernel_size)
        .with_stride(stride)
        .with_groups(channels)
        .with_padding(padding);
    conv1d(input, kernel, &cfg)
}

/// Grouped 1-D convolution.
#[must_use]
pub fn grouped_conv1d(
    input: &[f32],
    kernel: &[f32],
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    groups: usize,
    padding: PaddingMode,
) -> Vec<f32> {
    let cfg = Conv1dConfig::new(in_channels, out_channels, kernel_size)
        .with_groups(groups)
        .with_padding(padding);
    conv1d(input, kernel, &cfg)
}

/// Dilated 1-D convolution.
#[must_use]
pub fn dilated_conv1d(
    input: &[f32],
    kernel: &[f32],
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    dilation: usize,
    padding: PaddingMode,
) -> Vec<f32> {
    let cfg = Conv1dConfig::new(in_channels, out_channels, kernel_size)
        .with_dilation(dilation)
        .with_padding(padding);
    conv1d(input, kernel, &cfg)
}

// ── GPU stub ────────────────────────────────────────────────────────────

/// GPU-accelerated 1-D convolution (requires `gpu` or `cuda` feature).
#[cfg(any(feature = "gpu", feature = "cuda"))]
#[must_use]
pub fn conv1d_gpu(input: &[f32], kernel: &[f32], cfg: &Conv1dConfig) -> Vec<f32> {
    // Placeholder: fall back to CPU implementation.
    // A real CUDA kernel would launch here.
    conv1d(input, kernel, cfg)
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::*;

    // ── helpers ─────────────────────────────────────────────────────

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() <= tol, "mismatch at index {i}: {x} vs {y} (tol={tol})");
        }
    }

    /// Naive single-channel conv1d reference (no groups, no multi-channel).
    fn naive_conv1d(
        input: &[f32],
        kernel: &[f32],
        stride: usize,
        dilation: usize,
        pad_left: usize,
        pad_right: usize,
    ) -> Vec<f32> {
        let ek = (kernel.len() - 1) * dilation + 1;
        let padded_len = input.len() + pad_left + pad_right;
        if padded_len < ek {
            return Vec::new();
        }
        let out_len = (padded_len - ek) / stride + 1;
        let mut out = vec![0.0f32; out_len];
        for (o, out_val) in out.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for (k, &kw) in kernel.iter().enumerate() {
                let idx = o * stride + k * dilation;
                let signed = idx.cast_signed() - pad_left.cast_signed();
                if signed >= 0 && signed.cast_unsigned() < input.len() {
                    acc += kw * input[signed.cast_unsigned()];
                }
            }
            *out_val = acc;
        }
        out
    }

    // ── PaddingMode ────────────────────────────────────────────────

    #[test]
    fn test_padding_mode_eq() {
        assert_eq!(PaddingMode::Valid, PaddingMode::Valid);
        assert_ne!(PaddingMode::Valid, PaddingMode::Same);
        assert_ne!(PaddingMode::Same, PaddingMode::Full);
    }

    // ── Conv1dConfig ───────────────────────────────────────────────

    #[test]
    fn test_config_defaults() {
        let c = Conv1dConfig::new(3, 6, 5);
        assert_eq!(c.stride, 1);
        assert_eq!(c.dilation, 1);
        assert_eq!(c.groups, 1);
        assert_eq!(c.padding, PaddingMode::Valid);
    }

    #[test]
    fn test_config_builder_chain() {
        let c = Conv1dConfig::new(4, 8, 3)
            .with_stride(2)
            .with_dilation(3)
            .with_groups(4)
            .with_padding(PaddingMode::Same);
        assert_eq!(c.stride, 2);
        assert_eq!(c.dilation, 3);
        assert_eq!(c.groups, 4);
        assert_eq!(c.padding, PaddingMode::Same);
    }

    #[test]
    fn test_effective_kernel_size_no_dilation() {
        let c = Conv1dConfig::new(1, 1, 5);
        assert_eq!(c.effective_kernel_size(), 5);
    }

    #[test]
    fn test_effective_kernel_size_with_dilation() {
        let c = Conv1dConfig::new(1, 1, 3).with_dilation(2);
        assert_eq!(c.effective_kernel_size(), 5);
    }

    #[test]
    fn test_in_out_channels_per_group() {
        let c = Conv1dConfig::new(6, 12, 3).with_groups(3);
        assert_eq!(c.in_channels_per_group(), 2);
        assert_eq!(c.out_channels_per_group(), 4);
    }

    #[test]
    fn test_output_len_valid() {
        let c = Conv1dConfig::new(1, 1, 3);
        assert_eq!(c.output_len(5), 3);
        assert_eq!(c.output_len(3), 1);
        assert_eq!(c.output_len(10), 8);
    }

    #[test]
    fn test_output_len_valid_stride2() {
        let c = Conv1dConfig::new(1, 1, 3).with_stride(2);
        assert_eq!(c.output_len(5), 2);
        assert_eq!(c.output_len(7), 3);
    }

    #[test]
    fn test_output_len_same() {
        let c = Conv1dConfig::new(1, 1, 3).with_padding(PaddingMode::Same);
        assert_eq!(c.output_len(5), 5);
        assert_eq!(c.output_len(1), 1);
    }

    #[test]
    fn test_output_len_same_stride2() {
        let c = Conv1dConfig::new(1, 1, 3).with_stride(2).with_padding(PaddingMode::Same);
        assert_eq!(c.output_len(5), 3);
        assert_eq!(c.output_len(6), 3);
    }

    #[test]
    fn test_output_len_full() {
        let c = Conv1dConfig::new(1, 1, 3).with_padding(PaddingMode::Full);
        assert_eq!(c.output_len(5), 7);
    }

    #[test]
    fn test_output_len_full_dilation() {
        let c = Conv1dConfig::new(1, 1, 3).with_dilation(2).with_padding(PaddingMode::Full);
        assert_eq!(c.output_len(5), 9);
    }

    #[test]
    fn test_total_padding_valid() {
        let c = Conv1dConfig::new(1, 1, 3);
        assert_eq!(c.total_padding(10), 0);
    }

    #[test]
    fn test_total_padding_same() {
        let c = Conv1dConfig::new(1, 1, 3).with_padding(PaddingMode::Same);
        assert_eq!(c.total_padding(5), 2);
    }

    #[test]
    fn test_total_padding_full() {
        let c = Conv1dConfig::new(1, 1, 3).with_padding(PaddingMode::Full);
        assert_eq!(c.total_padding(5), 4);
    }

    // ── im2col_1d ──────────────────────────────────────────────────

    #[test]
    fn test_im2col_basic() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let col = im2col_1d(&input, 5, 3, 1, 1, 0, 0);
        assert_eq!(col.len(), 9);
        assert_eq!(&col[0..3], &[1.0, 2.0, 3.0]);
        assert_eq!(&col[3..6], &[2.0, 3.0, 4.0]);
        assert_eq!(&col[6..9], &[3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_im2col_with_stride() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let col = im2col_1d(&input, 5, 3, 2, 1, 0, 0);
        assert_eq!(col.len(), 6);
    }

    #[test]
    fn test_im2col_with_dilation() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let col = im2col_1d(&input, 5, 3, 1, 2, 0, 0);
        assert_eq!(col.len(), 3);
        assert_eq!(&col, &[1.0, 3.0, 5.0]);
    }

    #[test]
    fn test_im2col_with_padding() {
        let input = [1.0, 2.0, 3.0];
        let col = im2col_1d(&input, 3, 3, 1, 1, 1, 1);
        assert_eq!(col.len(), 9);
        assert_eq!(&col[0..3], &[0.0, 1.0, 2.0]);
        assert_eq!(&col[3..6], &[1.0, 2.0, 3.0]);
        assert_eq!(&col[6..9], &[2.0, 3.0, 0.0]);
    }

    #[test]
    fn test_im2col_empty_output() {
        let input = [1.0];
        let col = im2col_1d(&input, 1, 5, 1, 1, 0, 0);
        assert!(col.is_empty());
    }

    // ── conv1d basic (single channel, valid) ───────────────────────

    #[test]
    fn test_conv1d_identity_kernel() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kernel = vec![1.0];
        let cfg = Conv1dConfig::new(1, 1, 1);
        let out = conv1d(&input, &kernel, &cfg);
        approx_eq(&out, &input, 1e-6);
    }

    #[test]
    fn test_conv1d_simple_kernel3() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kernel = vec![1.0, 0.0, -1.0];
        let cfg = Conv1dConfig::new(1, 1, 3);
        let out = conv1d(&input, &kernel, &cfg);
        let expected = naive_conv1d(&input, &kernel, 1, 1, 0, 0);
        approx_eq(&out, &expected, 1e-6);
    }

    #[test]
    fn test_conv1d_all_ones() {
        let input = vec![1.0; 8];
        let kernel = vec![1.0; 3];
        let cfg = Conv1dConfig::new(1, 1, 3);
        let out = conv1d(&input, &kernel, &cfg);
        assert_eq!(out.len(), 6);
        for &v in &out {
            approx_eq(&[v], &[3.0], 1e-6);
        }
    }

    #[test]
    fn test_conv1d_single_element() {
        let input = vec![5.0];
        let kernel = vec![2.0];
        let cfg = Conv1dConfig::new(1, 1, 1);
        let out = conv1d(&input, &kernel, &cfg);
        approx_eq(&out, &[10.0], 1e-6);
    }

    #[test]
    fn test_conv1d_kernel_equals_input() {
        let input = vec![1.0, 2.0, 3.0];
        let kernel = vec![1.0, 1.0, 1.0];
        let cfg = Conv1dConfig::new(1, 1, 3);
        let out = conv1d(&input, &kernel, &cfg);
        approx_eq(&out, &[6.0], 1e-6);
    }

    // ── conv1d with stride ─────────────────────────────────────────

    #[test]
    fn test_conv1d_stride2() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let kernel = vec![1.0, 1.0, 1.0];
        let cfg = Conv1dConfig::new(1, 1, 3).with_stride(2);
        let out = conv1d(&input, &kernel, &cfg);
        let expected = naive_conv1d(&input, &kernel, 2, 1, 0, 0);
        approx_eq(&out, &expected, 1e-6);
    }

    #[test]
    fn test_conv1d_stride3() {
        let input: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let kernel = vec![1.0, -1.0];
        let cfg = Conv1dConfig::new(1, 1, 2).with_stride(3);
        let out = conv1d(&input, &kernel, &cfg);
        let expected = naive_conv1d(&input, &kernel, 3, 1, 0, 0);
        approx_eq(&out, &expected, 1e-6);
    }

    // ── conv1d padding same ────────────────────────────────────────

    #[test]
    fn test_conv1d_same_padding_k3() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kernel = vec![1.0, 1.0, 1.0];
        let cfg = Conv1dConfig::new(1, 1, 3).with_padding(PaddingMode::Same);
        let out = conv1d(&input, &kernel, &cfg);
        assert_eq!(out.len(), 5);
        approx_eq(&out, &[3.0, 6.0, 9.0, 12.0, 9.0], 1e-6);
    }

    #[test]
    fn test_conv1d_same_padding_k1() {
        let input = vec![1.0, 2.0, 3.0];
        let kernel = vec![2.0];
        let cfg = Conv1dConfig::new(1, 1, 1).with_padding(PaddingMode::Same);
        let out = conv1d(&input, &kernel, &cfg);
        approx_eq(&out, &[2.0, 4.0, 6.0], 1e-6);
    }

    #[test]
    fn test_conv1d_same_preserves_length() {
        for len in [1, 2, 5, 10, 16] {
            let input = vec![1.0; len];
            let kernel = vec![1.0; 5];
            let cfg = Conv1dConfig::new(1, 1, 5).with_padding(PaddingMode::Same);
            let out = conv1d(&input, &kernel, &cfg);
            assert_eq!(out.len(), len, "Same padding should preserve length {len}");
        }
    }

    // ── conv1d padding full ────────────────────────────────────────

    #[test]
    fn test_conv1d_full_padding_k3() {
        let input = vec![1.0, 2.0, 3.0];
        let kernel = vec![1.0, 1.0, 1.0];
        let cfg = Conv1dConfig::new(1, 1, 3).with_padding(PaddingMode::Full);
        let out = conv1d(&input, &kernel, &cfg);
        assert_eq!(out.len(), 5);
        approx_eq(&out, &[1.0, 3.0, 6.0, 5.0, 3.0], 1e-6);
    }

    #[test]
    fn test_conv1d_full_output_length() {
        let cfg = Conv1dConfig::new(1, 1, 4).with_padding(PaddingMode::Full);
        assert_eq!(cfg.output_len(10), 13);
    }

    // ── conv1d with dilation ───────────────────────────────────────

    #[test]
    fn test_conv1d_dilation2() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kernel = vec![1.0, 0.0, -1.0];
        let cfg = Conv1dConfig::new(1, 1, 3).with_dilation(2);
        let out = conv1d(&input, &kernel, &cfg);
        approx_eq(&out, &[-4.0], 1e-6);
    }

    #[test]
    fn test_conv1d_dilation_same() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kernel = vec![1.0, 1.0];
        let cfg = Conv1dConfig::new(1, 1, 2).with_dilation(2).with_padding(PaddingMode::Same);
        let out = conv1d(&input, &kernel, &cfg);
        assert_eq!(out.len(), 5);
    }

    #[test]
    fn test_dilated_conv1d_wrapper() {
        let input = vec![1.0, 0.0, 2.0, 0.0, 3.0];
        let kernel = vec![1.0, 1.0];
        let out = dilated_conv1d(&input, &kernel, 1, 1, 2, 2, PaddingMode::Valid);
        approx_eq(&out, &[3.0, 0.0, 5.0], 1e-6);
    }

    // ── multi-channel ──────────────────────────────────────────────

    #[test]
    fn test_conv1d_multi_in_channel() {
        let input = vec![
            1.0, 2.0, 3.0, // channel 0
            4.0, 5.0, 6.0, // channel 1
        ];
        let kernel = vec![
            1.0, 1.0, // out0, in0
            1.0, 1.0, // out0, in1
        ];
        let cfg = Conv1dConfig::new(2, 1, 2);
        let out = conv1d(&input, &kernel, &cfg);
        approx_eq(&out, &[12.0, 16.0], 1e-6);
    }

    #[test]
    fn test_conv1d_multi_out_channel() {
        let input = vec![1.0, 2.0, 3.0];
        let kernel = vec![
            1.0, 0.0, // out0
            0.0, 1.0, // out1
        ];
        let cfg = Conv1dConfig::new(1, 2, 2);
        let out = conv1d(&input, &kernel, &cfg);
        approx_eq(&out, &[1.0, 2.0, 2.0, 3.0], 1e-6);
    }

    #[test]
    fn test_conv1d_multi_in_out() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let kernel = vec![
            1.0, 0.0, // out0: in0*1 + in1*0
            0.0, 1.0, // out1: in0*0 + in1*1
        ];
        let cfg = Conv1dConfig::new(2, 2, 1);
        let out = conv1d(&input, &kernel, &cfg);
        approx_eq(&out, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1e-6);
    }

    // ── depthwise ──────────────────────────────────────────────────

    #[test]
    fn test_depthwise_conv1d_basic() {
        let input = vec![
            1.0, 2.0, 3.0, // ch0
            4.0, 5.0, 6.0, // ch1
            7.0, 8.0, 9.0, // ch2
        ];
        let kernel = vec![
            1.0, 1.0, // ch0
            1.0, -1.0, // ch1
            0.0, 1.0, // ch2
        ];
        let out = depthwise_conv1d(&input, &kernel, 3, 2, 1, PaddingMode::Valid);
        approx_eq(&out, &[3.0, 5.0, -1.0, -1.0, 8.0, 9.0], 1e-6);
    }

    #[test]
    fn test_depthwise_conv1d_same() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let kernel = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let out = depthwise_conv1d(&input, &kernel, 2, 3, 1, PaddingMode::Same);
        assert_eq!(out.len(), 6);
    }

    #[test]
    fn test_depthwise_identity() {
        let input = vec![1.0, 2.0, 3.0];
        let kernel = vec![1.0];
        let out = depthwise_conv1d(&input, &kernel, 1, 1, 1, PaddingMode::Valid);
        approx_eq(&out, &input, 1e-6);
    }

    // ── grouped ────────────────────────────────────────────────────

    #[test]
    fn test_grouped_conv1d_2groups() {
        let input = vec![
            1.0, 2.0, 3.0, // ch0
            4.0, 5.0, 6.0, // ch1
            7.0, 8.0, 9.0, // ch2
            10.0, 11.0, 12.0, // ch3
        ];
        let kernel = vec![
            1.0, 0.0, // out0
            0.0, 1.0, // out1
            1.0, 0.0, // out2
            0.0, 1.0, // out3
        ];
        let out = grouped_conv1d(&input, &kernel, 4, 4, 1, 2, PaddingMode::Valid);
        approx_eq(&out, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 1e-6);
    }

    #[test]
    fn test_grouped_conv1d_groups_equals_1() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let kernel = vec![1.0, 1.0, 1.0];
        let cfg = Conv1dConfig::new(1, 1, 3);
        let standard = conv1d(&input, &kernel, &cfg);
        let grouped = grouped_conv1d(&input, &kernel, 1, 1, 3, 1, PaddingMode::Valid);
        approx_eq(&standard, &grouped, 1e-6);
    }

    // ── edge cases ─────────────────────────────────────────────────

    #[test]
    fn test_conv1d_zero_input() {
        let input = vec![0.0; 10];
        let kernel = vec![1.0, 2.0, 3.0];
        let cfg = Conv1dConfig::new(1, 1, 3);
        let out = conv1d(&input, &kernel, &cfg);
        for &v in &out {
            approx_eq(&[v], &[0.0], 1e-6);
        }
    }

    #[test]
    fn test_conv1d_zero_kernel() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kernel = vec![0.0, 0.0, 0.0];
        let cfg = Conv1dConfig::new(1, 1, 3);
        let out = conv1d(&input, &kernel, &cfg);
        for &v in &out {
            approx_eq(&[v], &[0.0], 1e-6);
        }
    }

    #[test]
    fn test_conv1d_negative_values() {
        let input = vec![-1.0, -2.0, -3.0, -4.0];
        let kernel = vec![-1.0, -1.0];
        let cfg = Conv1dConfig::new(1, 1, 2);
        let out = conv1d(&input, &kernel, &cfg);
        approx_eq(&out, &[3.0, 5.0, 7.0], 1e-6);
    }

    #[test]
    fn test_conv1d_large_kernel() {
        let input = vec![1.0; 100];
        let kernel = vec![1.0; 50];
        let cfg = Conv1dConfig::new(1, 1, 50);
        let out = conv1d(&input, &kernel, &cfg);
        assert_eq!(out.len(), 51);
        for &v in &out {
            approx_eq(&[v], &[50.0], 1e-6);
        }
    }

    #[test]
    fn test_conv1d_large_stride() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let kernel = vec![1.0];
        let cfg = Conv1dConfig::new(1, 1, 1).with_stride(5);
        let out = conv1d(&input, &kernel, &cfg);
        approx_eq(&out, &[1.0, 6.0], 1e-6);
    }

    #[test]
    fn test_conv1d_empty_output_kernel_too_large() {
        let input = vec![1.0, 2.0];
        let kernel = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let cfg = Conv1dConfig::new(1, 1, 5);
        let out = conv1d(&input, &kernel, &cfg);
        assert!(out.is_empty());
    }

    #[test]
    fn test_conv1d_fractional_values() {
        let input = vec![0.5, 1.5, 2.5];
        let kernel = vec![0.5, 0.5];
        let cfg = Conv1dConfig::new(1, 1, 2);
        let out = conv1d(&input, &kernel, &cfg);
        approx_eq(&out, &[1.0, 2.0], 1e-6);
    }

    // ── combined features ──────────────────────────────────────────

    #[test]
    fn test_conv1d_stride_and_dilation() {
        let input: Vec<f32> = (1..=10).map(|x| x as f32).collect();
        let kernel = vec![1.0, 1.0];
        let cfg = Conv1dConfig::new(1, 1, 2).with_stride(2).with_dilation(3);
        let out = conv1d(&input, &kernel, &cfg);
        let expected = naive_conv1d(&input, &kernel, 2, 3, 0, 0);
        approx_eq(&out, &expected, 1e-6);
    }

    #[test]
    fn test_conv1d_same_padding_stride_dilation() {
        let input: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let kernel = vec![1.0, 1.0, 1.0];
        let cfg = Conv1dConfig::new(1, 1, 3)
            .with_stride(2)
            .with_dilation(2)
            .with_padding(PaddingMode::Same);
        let out = conv1d(&input, &kernel, &cfg);
        assert_eq!(out.len(), cfg.output_len(8));
    }

    #[test]
    fn test_conv1d_full_padding_stride2() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let kernel = vec![1.0, 1.0, 1.0];
        let cfg = Conv1dConfig::new(1, 1, 3).with_stride(2).with_padding(PaddingMode::Full);
        let out = conv1d(&input, &kernel, &cfg);
        assert_eq!(out.len(), cfg.output_len(4));
    }

    // ── multi-channel with padding ─────────────────────────────────

    #[test]
    fn test_conv1d_multi_channel_same_padding() {
        let input = vec![
            1.0, 2.0, 3.0, // ch0
            4.0, 5.0, 6.0, // ch1
        ];
        let kernel = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let cfg = Conv1dConfig::new(2, 1, 3).with_padding(PaddingMode::Same);
        let out = conv1d(&input, &kernel, &cfg);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn test_conv1d_depthwise_with_dilation() {
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, // ch0
            6.0, 7.0, 8.0, 9.0, 10.0, // ch1
        ];
        let kernel = vec![
            1.0, -1.0, // ch0
            1.0, 1.0, // ch1
        ];
        let cfg = Conv1dConfig::new(2, 2, 2).with_groups(2).with_dilation(2);
        let out = conv1d(&input, &kernel, &cfg);
        approx_eq(&out, &[-2.0, -2.0, -2.0, 14.0, 16.0, 18.0], 1e-6);
    }

    // ── symmetry / invariants ──────────────────────────────────────

    #[test]
    fn test_conv1d_commutative_length() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, -1.0, 1.0];
        let out1 = conv1d(&a, &b, &Conv1dConfig::new(1, 1, 3));
        let out2 = naive_conv1d(&a, &b, 1, 1, 0, 0);
        assert_eq!(out1.len(), out2.len());
    }

    #[test]
    fn test_conv1d_linearity() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kernel = vec![1.0, 0.5, -0.5];
        let alpha = 3.0f32;
        let scaled_input: Vec<f32> = input.iter().map(|&x| x * alpha).collect();

        let cfg = Conv1dConfig::new(1, 1, 3);
        let out_scaled = conv1d(&scaled_input, &kernel, &cfg);
        let out_then_scale: Vec<f32> =
            conv1d(&input, &kernel, &cfg).iter().map(|&x| x * alpha).collect();
        approx_eq(&out_scaled, &out_then_scale, 1e-5);
    }

    // ── assert panics ──────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "in_channels must be > 0")]
    fn test_conv1d_panics_zero_in_channels() {
        let cfg = Conv1dConfig::new(0, 1, 3);
        let _ = conv1d(&[], &[], &cfg);
    }

    #[test]
    #[should_panic(expected = "out_channels must be > 0")]
    fn test_conv1d_panics_zero_out_channels() {
        let cfg = Conv1dConfig::new(1, 0, 3);
        let _ = conv1d(&[], &[], &cfg);
    }

    #[test]
    #[should_panic(expected = "kernel_size must be > 0")]
    fn test_conv1d_panics_zero_kernel_size() {
        let cfg = Conv1dConfig::new(1, 1, 0);
        let _ = conv1d(&[], &[], &cfg);
    }

    #[test]
    #[should_panic(expected = "stride must be > 0")]
    fn test_conv1d_panics_zero_stride() {
        let cfg = Conv1dConfig::new(1, 1, 3).with_stride(0);
        let _ = conv1d(&[0.0; 5], &[0.0; 3], &cfg);
    }

    #[test]
    #[should_panic(expected = "dilation must be > 0")]
    fn test_conv1d_panics_zero_dilation() {
        let cfg = Conv1dConfig::new(1, 1, 3).with_dilation(0);
        let _ = conv1d(&[0.0; 5], &[0.0; 3], &cfg);
    }

    #[test]
    #[should_panic(expected = "groups must be > 0")]
    fn test_conv1d_panics_zero_groups() {
        let cfg = Conv1dConfig::new(1, 1, 3).with_groups(0);
        let _ = conv1d(&[0.0; 5], &[0.0; 3], &cfg);
    }

    #[test]
    #[should_panic(expected = "in_channels must be divisible by groups")]
    fn test_conv1d_panics_bad_groups() {
        let cfg = Conv1dConfig::new(3, 6, 3).with_groups(2);
        let _ = conv1d(&[0.0; 15], &[0.0; 18], &cfg);
    }

    // ── GPU feature gate ───────────────────────────────────────────

    #[test]
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn test_conv1d_gpu_matches_cpu() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kernel = vec![1.0, 0.0, -1.0];
        let cfg = Conv1dConfig::new(1, 1, 3);
        let cpu_out = conv1d(&input, &kernel, &cfg);
        let gpu_out = conv1d_gpu(&input, &kernel, &cfg);
        approx_eq(&cpu_out, &gpu_out, 1e-6);
    }

    // ── various kernel sizes ───────────────────────────────────────

    #[test]
    fn test_conv1d_kernel_size_1() {
        let input = vec![3.0, 6.0, 9.0];
        let kernel = vec![0.5];
        let cfg = Conv1dConfig::new(1, 1, 1);
        let out = conv1d(&input, &kernel, &cfg);
        approx_eq(&out, &[1.5, 3.0, 4.5], 1e-6);
    }

    #[test]
    fn test_conv1d_kernel_size_5() {
        let input: Vec<f32> = (0..10).map(|x| x as f32).collect();
        let kernel = vec![1.0; 5];
        let cfg = Conv1dConfig::new(1, 1, 5);
        let out = conv1d(&input, &kernel, &cfg);
        let expected = naive_conv1d(&input, &kernel, 1, 1, 0, 0);
        approx_eq(&out, &expected, 1e-6);
    }

    #[test]
    fn test_conv1d_kernel_size_7() {
        let input: Vec<f32> = (0..14).map(|x| x as f32).collect();
        let kernel = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0];
        let cfg = Conv1dConfig::new(1, 1, 7);
        let out = conv1d(&input, &kernel, &cfg);
        let expected = naive_conv1d(&input, &kernel, 1, 1, 0, 0);
        approx_eq(&out, &expected, 1e-6);
    }

    // ── additional correctness ─────────────────────────────────────

    #[test]
    fn test_conv1d_matches_naive_random_like() {
        let input: Vec<f32> = (0..20).map(|i| ((i * 7 + 3) % 13) as f32 / 6.0).collect();
        let kernel: Vec<f32> = (0..5).map(|i| ((i * 3 + 1) % 7) as f32 / 3.0 - 1.0).collect();
        let cfg = Conv1dConfig::new(1, 1, 5);
        let out = conv1d(&input, &kernel, &cfg);
        let expected = naive_conv1d(&input, &kernel, 1, 1, 0, 0);
        approx_eq(&out, &expected, 1e-5);
    }

    #[test]
    fn test_conv1d_output_channels_ordering() {
        let input = vec![2.0, 4.0, 6.0];
        let kernel = vec![1.0, 2.0, 3.0];
        let cfg = Conv1dConfig::new(1, 3, 1);
        let out = conv1d(&input, &kernel, &cfg);
        approx_eq(&out, &[2.0, 4.0, 6.0, 4.0, 8.0, 12.0, 6.0, 12.0, 18.0], 1e-6);
    }

    #[test]
    fn test_conv1d_depthwise_stride2_same() {
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // ch0
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // ch1
        ];
        let kernel = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let out = depthwise_conv1d(&input, &kernel, 2, 3, 2, PaddingMode::Same);
        assert_eq!(out.len(), 6);
    }

    #[test]
    fn test_conv1d_grouped_same_padding() {
        let input = vec![1.0; 12];
        let kernel = vec![1.0; 8];
        let out = grouped_conv1d(&input, &kernel, 4, 4, 1, 2, PaddingMode::Same);
        assert_eq!(out.len(), 12);
    }

    #[test]
    fn test_conv1d_config_clone() {
        let c = Conv1dConfig::new(2, 4, 3).with_stride(2);
        let c2 = c.clone();
        assert_eq!(c, c2);
    }

    #[test]
    fn test_conv1d_config_debug() {
        let c = Conv1dConfig::new(1, 1, 3);
        let dbg = format!("{c:?}");
        assert!(dbg.contains("Conv1dConfig"));
    }

    #[test]
    fn test_padding_mode_clone() {
        let p = PaddingMode::Same;
        let p2 = p;
        assert_eq!(p, p2);
    }
}

// ── Property tests ──────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_output_len_valid(
            input_len in 1usize..200,
            kernel_size in 1usize..50,
            stride in 1usize..10,
            dilation in 1usize..5,
        ) {
            let cfg = Conv1dConfig::new(1, 1, kernel_size)
                .with_stride(stride)
                .with_dilation(dilation);
            let ek = cfg.effective_kernel_size();
            let out_len = cfg.output_len(input_len);
            if input_len >= ek {
                prop_assert_eq!(out_len, (input_len - ek) / stride + 1);
            } else {
                prop_assert_eq!(out_len, 0);
            }
        }

        #[test]
        fn prop_output_len_same_preserves(
            input_len in 1usize..200,
            kernel_size in 1usize..50,
        ) {
            let cfg = Conv1dConfig::new(1, 1, kernel_size)
                .with_padding(PaddingMode::Same);
            prop_assert_eq!(cfg.output_len(input_len), input_len);
        }

        #[test]
        fn prop_output_len_full(
            input_len in 1usize..100,
            kernel_size in 1usize..30,
        ) {
            let cfg = Conv1dConfig::new(1, 1, kernel_size)
                .with_padding(PaddingMode::Full);
            prop_assert_eq!(cfg.output_len(input_len), input_len + kernel_size - 1);
        }

        #[test]
        fn prop_zero_input_gives_zero_output(
            input_len in 3usize..50,
            kernel_size in 1usize..=3,
        ) {
            prop_assume!(input_len >= kernel_size);
            let input = vec![0.0f32; input_len];
            let kernel: Vec<f32> = (0..kernel_size).map(|i| (i as f32) + 1.0).collect();
            let cfg = Conv1dConfig::new(1, 1, kernel_size);
            let out = conv1d(&input, &kernel, &cfg);
            for &v in &out {
                prop_assert!((v).abs() < 1e-6, "expected zero, got {v}");
            }
        }

        #[test]
        fn prop_zero_kernel_gives_zero_output(
            input_len in 3usize..50,
            kernel_size in 1usize..=3,
        ) {
            prop_assume!(input_len >= kernel_size);
            let input: Vec<f32> = (0..input_len).map(|i| (i as f32) + 1.0).collect();
            let kernel = vec![0.0f32; kernel_size];
            let cfg = Conv1dConfig::new(1, 1, kernel_size);
            let out = conv1d(&input, &kernel, &cfg);
            for &v in &out {
                prop_assert!((v).abs() < 1e-6, "expected zero, got {v}");
            }
        }

        #[test]
        fn prop_linearity(
            alpha in -10.0f32..10.0,
            input_len in 3usize..30,
        ) {
            let kernel_size = 3usize;
            prop_assume!(input_len >= kernel_size);
            let input: Vec<f32> = (0..input_len).map(|i| (i as f32) * 0.1).collect();
            let kernel = vec![1.0f32, -0.5, 0.25];
            let scaled: Vec<f32> = input.iter().map(|&x| x * alpha).collect();
            let cfg = Conv1dConfig::new(1, 1, kernel_size);
            let out_scaled = conv1d(&scaled, &kernel, &cfg);
            let out_then: Vec<f32> = conv1d(&input, &kernel, &cfg)
                .iter()
                .map(|&x| x * alpha)
                .collect();
            for (a, b) in out_scaled.iter().zip(out_then.iter()) {
                prop_assert!((a - b).abs() < 1e-3, "linearity violated: {a} vs {b}");
            }
        }

        #[test]
        fn prop_valid_output_length(
            input_len in 1usize..100,
            kernel_size in 1usize..20,
            stride in 1usize..5,
        ) {
            let cfg = Conv1dConfig::new(1, 1, kernel_size).with_stride(stride);
            let out_len = cfg.output_len(input_len);
            if input_len >= kernel_size {
                let input = vec![1.0f32; input_len];
                let kernel = vec![1.0f32; kernel_size];
                let out = conv1d(&input, &kernel, &cfg);
                prop_assert_eq!(out.len(), out_len);
            }
        }

        #[test]
        fn prop_same_padding_output_len(
            input_len in 1usize..100,
            kernel_size in 1usize..20,
            stride in 1usize..5,
        ) {
            let cfg = Conv1dConfig::new(1, 1, kernel_size)
                .with_stride(stride)
                .with_padding(PaddingMode::Same);
            let expected = input_len.div_ceil(stride);
            prop_assert_eq!(cfg.output_len(input_len), expected);
        }

        #[test]
        fn prop_effective_kernel_size(
            kernel_size in 1usize..50,
            dilation in 1usize..10,
        ) {
            let cfg = Conv1dConfig::new(1, 1, kernel_size).with_dilation(dilation);
            prop_assert_eq!(cfg.effective_kernel_size(), (kernel_size - 1) * dilation + 1);
        }

        #[test]
        fn prop_depthwise_channels_preserved(
            channels in 1usize..8,
            input_len in 3usize..20,
        ) {
            let kernel_size = 3usize;
            prop_assume!(input_len >= kernel_size);
            let input = vec![1.0f32; channels * input_len];
            let kernel = vec![1.0f32; channels * kernel_size];
            let out = depthwise_conv1d(&input, &kernel, channels, kernel_size, 1, PaddingMode::Valid);
            let expected_len = input_len - kernel_size + 1;
            prop_assert_eq!(out.len(), channels * expected_len);
        }
    }
}

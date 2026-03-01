//! CPU 1-D convolution kernel.
//!
//! Provides grouped 1-D convolution on contiguous `f32` slices with
//! configurable stride, padding, dilation, and groups.  Supports
//! depthwise separable convolution (`groups == in_channels`).  Scalar
//! implementation for correctness; SIMD acceleration can be added later.
//!
//! # Layout
//!
//! * `input`:  `[in_channels, input_width]`  (channels-first, contiguous)
//! * `weight`: `[out_channels, in_channels / groups, kernel_size]`
//! * `bias`:   `[out_channels]`  (optional)
//! * output:   `[out_channels, output_width]`

use bitnet_common::{BitNetError, KernelError, Result};

// ── Configuration ──────────────────────────────────────────────────

/// Padding mode for 1-D convolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PaddingMode {
    /// Explicit zero-padding added to each side.
    Zero(usize),
    /// Automatically compute padding so that
    /// `output_width == ceil(input_width / stride)`.
    Same,
}

/// Parameters for a 1-D convolution operation.
#[derive(Debug, Clone)]
pub struct Conv1dConfig {
    /// Number of input channels.
    pub in_channels: usize,
    /// Number of output channels (number of filters).
    pub out_channels: usize,
    /// Spatial extent of each filter.
    pub kernel_size: usize,
    /// Step between successive convolution windows.
    pub stride: usize,
    /// Padding applied to the input.
    pub padding: PaddingMode,
    /// Spacing between kernel elements.
    pub dilation: usize,
    /// Number of blocked connections from input to output channels.
    /// Use `in_channels` for depthwise convolution.
    pub groups: usize,
    /// Whether a bias vector is expected.
    pub bias: bool,
}

impl Conv1dConfig {
    /// Effective kernel size accounting for dilation.
    #[inline]
    fn effective_kernel_size(&self) -> usize {
        self.dilation * (self.kernel_size - 1) + 1
    }

    /// Resolve padding as `(pad_left, pad_right)`.
    fn resolve_padding(&self, input_width: usize) -> (usize, usize) {
        match self.padding {
            PaddingMode::Zero(p) => (p, p),
            PaddingMode::Same => {
                let ek = self.effective_kernel_size();
                let out_w = input_width.div_ceil(self.stride);
                let needed = out_w.saturating_sub(1) * self.stride + ek;
                let total = needed.saturating_sub(input_width);
                let pad_left = total / 2;
                (pad_left, total - pad_left)
            }
        }
    }

    /// Compute output spatial width for a given input width.
    fn output_width(&self, input_width: usize) -> usize {
        let (pl, pr) = self.resolve_padding(input_width);
        let ek = self.effective_kernel_size();
        let padded = input_width + pl + pr;
        if padded < ek { 0 } else { (padded - ek) / self.stride + 1 }
    }
}

// ── Error helper ───────────────────────────────────────────────────

fn invalid_args(reason: &str) -> BitNetError {
    BitNetError::Kernel(KernelError::InvalidArguments { reason: reason.to_string() })
}

// ── Validation ─────────────────────────────────────────────────────

fn validate(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    config: &Conv1dConfig,
) -> Result<usize> {
    if config.in_channels == 0 {
        return Err(invalid_args("in_channels must be > 0"));
    }
    if config.out_channels == 0 {
        return Err(invalid_args("out_channels must be > 0"));
    }
    if config.kernel_size == 0 {
        return Err(invalid_args("kernel_size must be > 0"));
    }
    if config.stride == 0 {
        return Err(invalid_args("stride must be > 0"));
    }
    if config.dilation == 0 {
        return Err(invalid_args("dilation must be > 0"));
    }
    if config.groups == 0 {
        return Err(invalid_args("groups must be > 0"));
    }
    if !config.in_channels.is_multiple_of(config.groups) {
        return Err(invalid_args("in_channels must be divisible by groups"));
    }
    if !config.out_channels.is_multiple_of(config.groups) {
        return Err(invalid_args("out_channels must be divisible by groups"));
    }

    if !input.len().is_multiple_of(config.in_channels) {
        return Err(invalid_args("input length must be divisible by in_channels"));
    }
    let input_width = input.len() / config.in_channels;
    if input_width == 0 {
        return Err(invalid_args("input spatial width must be > 0"));
    }

    let ic_per_group = config.in_channels / config.groups;
    let expected_weight_len = config.out_channels * ic_per_group * config.kernel_size;
    if weight.len() != expected_weight_len {
        return Err(invalid_args(&format!(
            "weight length mismatch: expected {expected_weight_len}, got {}",
            weight.len()
        )));
    }

    if config.bias {
        match bias {
            Some(b) if b.len() != config.out_channels => {
                return Err(invalid_args(&format!(
                    "bias length mismatch: expected {}, got {}",
                    config.out_channels,
                    b.len()
                )));
            }
            None => {
                return Err(invalid_args("config.bias is true but no bias provided"));
            }
            _ => {}
        }
    }

    let out_w = config.output_width(input_width);
    if out_w == 0 {
        return Err(invalid_args(
            "convolution produces zero-width output (kernel larger than padded input)",
        ));
    }

    Ok(input_width)
}

// ── Public API ─────────────────────────────────────────────────────

/// Compute 1-D convolution on a single sample.
///
/// # Layout
///
/// * `input`:  `[in_channels, input_width]`  (channels-first, contiguous)
/// * `weight`: `[out_channels, in_channels / groups, kernel_size]`
/// * `bias`:   `[out_channels]`  (optional, must match `config.bias`)
/// * returns:  `[out_channels, output_width]`
///
/// # Errors
///
/// Returns `InvalidArguments` on dimension mismatches or invalid config.
pub fn conv1d_forward(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    config: &Conv1dConfig,
) -> Result<Vec<f32>> {
    let input_width = validate(input, weight, bias, config)?;
    let out_w = config.output_width(input_width);
    let (pad_left, _pad_right) = config.resolve_padding(input_width);
    let ic_per_group = config.in_channels / config.groups;
    let oc_per_group = config.out_channels / config.groups;

    let mut output = vec![0.0f32; config.out_channels * out_w];

    for g in 0..config.groups {
        for oc_local in 0..oc_per_group {
            let oc = g * oc_per_group + oc_local;
            for ow in 0..out_w {
                let mut sum = 0.0f32;
                for ic_local in 0..ic_per_group {
                    let ic = g * ic_per_group + ic_local;
                    let w_base = (oc * ic_per_group + ic_local) * config.kernel_size;
                    for k in 0..config.kernel_size {
                        let iw = ow * config.stride + k * config.dilation;
                        if iw >= pad_left && iw - pad_left < input_width {
                            sum += input[ic * input_width + (iw - pad_left)] * weight[w_base + k];
                        }
                    }
                }
                output[oc * out_w + ow] = sum;
            }
        }
    }

    // Add bias.
    if let Some(b) = bias
        && config.bias
    {
        for oc in 0..config.out_channels {
            let bias_val = b[oc];
            for v in &mut output[oc * out_w..(oc + 1) * out_w] {
                *v += bias_val;
            }
        }
    }

    Ok(output)
}

/// Compute the output spatial width for a 1-D convolution.
///
/// Useful for pre-allocating buffers or verifying network dimensions.
pub fn conv1d_output_width(config: &Conv1dConfig, input_width: usize) -> usize {
    config.output_width(input_width)
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-6;

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
    }

    /// Helper to build a minimal config.
    fn cfg(
        in_c: usize,
        out_c: usize,
        ks: usize,
        stride: usize,
        padding: PaddingMode,
        dilation: usize,
        groups: usize,
        bias: bool,
    ) -> Conv1dConfig {
        Conv1dConfig {
            in_channels: in_c,
            out_channels: out_c,
            kernel_size: ks,
            stride,
            padding,
            dilation,
            groups,
            bias,
        }
    }

    // ── Basic correctness ──────────────────────────────────

    #[test]
    fn identity_conv() {
        // kernel = [1.0], single channel → output == input
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let weight = vec![1.0];
        let c = cfg(1, 1, 1, 1, PaddingMode::Zero(0), 1, 1, false);
        let out = conv1d_forward(&input, &weight, None, &c).unwrap();
        assert!(approx_eq(&out, &input, TOL));
    }

    #[test]
    fn known_conv_edge_detect() {
        // kernel = [1, 0, -1] on [1,2,3,4,5] → [-2, -2, -2]
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let weight = vec![1.0, 0.0, -1.0];
        let c = cfg(1, 1, 3, 1, PaddingMode::Zero(0), 1, 1, false);
        let out = conv1d_forward(&input, &weight, None, &c).unwrap();
        assert!(approx_eq(&out, &[-2.0, -2.0, -2.0], TOL));
    }

    #[test]
    fn known_conv_multichannel() {
        // 2 in_channels, 1 out_channel, kernel_size=2
        // ch0=[1,2,3] ch1=[4,5,6]  weight=[1,-1, 2,0]
        // oc0,pos0: 1*1+2*(-1)+4*2+5*0 = 7
        // oc0,pos1: 2*1+3*(-1)+5*2+6*0 = 9
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weight = vec![1.0, -1.0, 2.0, 0.0];
        let c = cfg(2, 1, 2, 1, PaddingMode::Zero(0), 1, 1, false);
        let out = conv1d_forward(&input, &weight, None, &c).unwrap();
        assert!(approx_eq(&out, &[7.0, 9.0], TOL));
    }

    #[test]
    fn stride_2() {
        // [1,2,3,4,5] kernel=[1,1] stride=2 → [3, 7]
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let weight = vec![1.0, 1.0];
        let c = cfg(1, 1, 2, 2, PaddingMode::Zero(0), 1, 1, false);
        let out = conv1d_forward(&input, &weight, None, &c).unwrap();
        assert!(approx_eq(&out, &[3.0, 7.0], TOL));
    }

    // ── Padding modes ─────────────────────────────────────

    #[test]
    fn zero_padding() {
        // [1,2,3] kernel=[1,1,1] pad=1 → [0+1+2, 1+2+3, 2+3+0] = [3,6,5]
        let input = vec![1.0, 2.0, 3.0];
        let weight = vec![1.0, 1.0, 1.0];
        let c = cfg(1, 1, 3, 1, PaddingMode::Zero(1), 1, 1, false);
        let out = conv1d_forward(&input, &weight, None, &c).unwrap();
        assert!(approx_eq(&out, &[3.0, 6.0, 5.0], TOL));
    }

    #[test]
    fn same_padding_stride_1() {
        // [1,2,3,4,5] kernel=[1,2,1] Same → output_width=5
        // pad_left=1, pad_right=1
        // [0+2+2, 1+4+3, 2+6+4, 3+8+5, 4+10+0] = [4,8,12,16,14]
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let weight = vec![1.0, 2.0, 1.0];
        let c = cfg(1, 1, 3, 1, PaddingMode::Same, 1, 1, false);
        let out = conv1d_forward(&input, &weight, None, &c).unwrap();
        assert_eq!(out.len(), 5);
        assert!(approx_eq(&out, &[4.0, 8.0, 12.0, 16.0, 14.0], TOL));
    }

    #[test]
    fn same_padding_stride_2() {
        // [1,2,3,4,5] kernel=[1,1,1] stride=2, Same → out_w=ceil(5/2)=3
        // pad_left=1, pad_right=1, padded=[0,1,2,3,4,5,0]
        // ow=0: 0+1+2=3  ow=1: 2+3+4=9  ow=2: 4+5+0=9
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let weight = vec![1.0, 1.0, 1.0];
        let c = cfg(1, 1, 3, 2, PaddingMode::Same, 1, 1, false);
        let out = conv1d_forward(&input, &weight, None, &c).unwrap();
        assert_eq!(out.len(), 3);
        assert!(approx_eq(&out, &[3.0, 9.0, 9.0], TOL));
    }

    // ── Dilation ──────────────────────────────────────────

    #[test]
    fn dilation_2() {
        // [1,2,3,4,5] kernel=[1,1] dilation=2 → ek=3 out_w=3
        // pos0: in[0]+in[2]=4  pos1: in[1]+in[3]=6  pos2: in[2]+in[4]=8
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let weight = vec![1.0, 1.0];
        let c = cfg(1, 1, 2, 1, PaddingMode::Zero(0), 2, 1, false);
        let out = conv1d_forward(&input, &weight, None, &c).unwrap();
        assert!(approx_eq(&out, &[4.0, 6.0, 8.0], TOL));
    }

    #[test]
    fn same_padding_with_dilation() {
        // [1,2,3,4,5] kernel=[1,1,1] dilation=2, Same → out_w=5
        // ek=5, total_pad=4, pad_left=2, pad_right=2
        // ow=0: in[-2]+in[0]+in[2] → 0+1+3=4
        // ow=1: in[-1]+in[1]+in[3] → 0+2+4=6
        // ow=2: in[0]+in[2]+in[4] → 1+3+5=9
        // ow=3: in[1]+in[3]+in[5] → 2+4+0=6
        // ow=4: in[2]+in[4]+in[6] → 3+5+0=8
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let weight = vec![1.0, 1.0, 1.0];
        let c = cfg(1, 1, 3, 1, PaddingMode::Same, 2, 1, false);
        let out = conv1d_forward(&input, &weight, None, &c).unwrap();
        assert_eq!(out.len(), 5);
        assert!(approx_eq(&out, &[4.0, 6.0, 9.0, 6.0, 8.0], TOL));
    }

    // ── Groups / depthwise ────────────────────────────────

    #[test]
    fn groups_2() {
        // 4 in, 4 out, groups=2 → group0: ic0,ic1→oc0,oc1; group1: ic2,ic3→oc2,oc3
        // kernel_size=1, weight=[1,0, 0,1, 1,0, 0,1]
        let input = vec![
            1.0, 2.0, 3.0, // ch0
            4.0, 5.0, 6.0, // ch1
            7.0, 8.0, 9.0, // ch2
            10.0, 11.0, 12.0, // ch3
        ];
        let weight = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let c = cfg(4, 4, 1, 1, PaddingMode::Zero(0), 1, 2, false);
        let out = conv1d_forward(&input, &weight, None, &c).unwrap();
        #[rustfmt::skip]
        let expected = vec![
            1.0, 2.0, 3.0,     // oc0 = ch0
            4.0, 5.0, 6.0,     // oc1 = ch1
            7.0, 8.0, 9.0,     // oc2 = ch2
            10.0, 11.0, 12.0,  // oc3 = ch3
        ];
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn depthwise_conv() {
        // 3 in, 3 out, groups=3 (depthwise), kernel=[1,-1] per channel
        let input = vec![
            1.0, 2.0, 3.0, 4.0, // ch0
            5.0, 6.0, 7.0, 8.0, // ch1
            9.0, 10.0, 11.0, 12.0, // ch2
        ];
        let weight = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let c = cfg(3, 3, 2, 1, PaddingMode::Zero(0), 1, 3, false);
        let out = conv1d_forward(&input, &weight, None, &c).unwrap();
        let expected = vec![
            -1.0, -1.0, -1.0, // oc0: diff of ch0
            -1.0, -1.0, -1.0, // oc1: diff of ch1
            -1.0, -1.0, -1.0, // oc2: diff of ch2
        ];
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn depthwise_separable_known() {
        // Depthwise with distinct per-channel kernels
        // 2 in, 2 out, groups=2, kernel_size=3
        // ch0=[1,2,3,4] kernel0=[1,0,0] → [1,2]
        // ch1=[5,6,7,8] kernel1=[0,0,1] → [7,8]
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let weight = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let c = cfg(2, 2, 3, 1, PaddingMode::Zero(0), 1, 2, false);
        let out = conv1d_forward(&input, &weight, None, &c).unwrap();
        assert!(approx_eq(&out, &[1.0, 2.0, 7.0, 8.0], TOL));
    }

    // ── Bias ──────────────────────────────────────────────

    #[test]
    fn bias_addition() {
        // identity kernel + bias
        let input = vec![1.0, 2.0, 3.0];
        let weight = vec![1.0];
        let bias = vec![10.0];
        let c = cfg(1, 1, 1, 1, PaddingMode::Zero(0), 1, 1, true);
        let out = conv1d_forward(&input, &weight, Some(&bias), &c).unwrap();
        assert!(approx_eq(&out, &[11.0, 12.0, 13.0], TOL));
    }

    #[test]
    fn bias_multichannel() {
        // 1 in, 2 out, kernel_size=1, bias=[10, 20]
        let input = vec![1.0, 2.0, 3.0];
        let weight = vec![1.0, 2.0]; // oc0: [1], oc1: [2]
        let bias = vec![10.0, 20.0];
        let c = cfg(1, 2, 1, 1, PaddingMode::Zero(0), 1, 1, true);
        let out = conv1d_forward(&input, &weight, Some(&bias), &c).unwrap();
        // oc0: [1+10, 2+10, 3+10]  oc1: [2+20, 4+20, 6+20]
        assert!(approx_eq(&out, &[11.0, 12.0, 13.0, 22.0, 24.0, 26.0], TOL));
    }

    #[test]
    fn no_bias() {
        let input = vec![1.0, 2.0, 3.0];
        let weight = vec![1.0];
        let c = cfg(1, 1, 1, 1, PaddingMode::Zero(0), 1, 1, false);
        let out = conv1d_forward(&input, &weight, None, &c).unwrap();
        assert!(approx_eq(&out, &[1.0, 2.0, 3.0], TOL));
    }

    // ── Edge cases ────────────────────────────────────────

    #[test]
    fn single_element_input() {
        let input = vec![5.0];
        let weight = vec![2.0];
        let c = cfg(1, 1, 1, 1, PaddingMode::Zero(0), 1, 1, false);
        let out = conv1d_forward(&input, &weight, None, &c).unwrap();
        assert!(approx_eq(&out, &[10.0], TOL));
    }

    #[test]
    fn kernel_equals_input_width() {
        let input = vec![1.0, 2.0, 3.0];
        let weight = vec![1.0, 1.0, 1.0];
        let c = cfg(1, 1, 3, 1, PaddingMode::Zero(0), 1, 1, false);
        let out = conv1d_forward(&input, &weight, None, &c).unwrap();
        assert!(approx_eq(&out, &[6.0], TOL));
    }

    #[test]
    fn multiple_output_channels() {
        // 1 in, 3 out, kernel_size=1 → each oc is a scalar multiplication
        let input = vec![1.0, 2.0];
        let weight = vec![1.0, 2.0, 3.0]; // oc0:[1], oc1:[2], oc2:[3]
        let c = cfg(1, 3, 1, 1, PaddingMode::Zero(0), 1, 1, false);
        let out = conv1d_forward(&input, &weight, None, &c).unwrap();
        assert!(approx_eq(&out, &[1.0, 2.0, 2.0, 4.0, 3.0, 6.0], TOL));
    }

    // ── Dimension validation errors ───────────────────────

    #[test]
    fn error_zero_kernel_size() {
        let c = cfg(1, 1, 0, 1, PaddingMode::Zero(0), 1, 1, false);
        assert!(conv1d_forward(&[1.0], &[], None, &c).is_err());
    }

    #[test]
    fn error_zero_stride() {
        let c = cfg(1, 1, 1, 0, PaddingMode::Zero(0), 1, 1, false);
        assert!(conv1d_forward(&[1.0], &[1.0], None, &c).is_err());
    }

    #[test]
    fn error_zero_dilation() {
        let c = cfg(1, 1, 1, 1, PaddingMode::Zero(0), 0, 1, false);
        assert!(conv1d_forward(&[1.0], &[1.0], None, &c).is_err());
    }

    #[test]
    fn error_zero_groups() {
        let c = cfg(1, 1, 1, 1, PaddingMode::Zero(0), 1, 0, false);
        assert!(conv1d_forward(&[1.0], &[1.0], None, &c).is_err());
    }

    #[test]
    fn error_in_channels_not_divisible_by_groups() {
        let c = cfg(3, 2, 1, 1, PaddingMode::Zero(0), 1, 2, false);
        assert!(conv1d_forward(&[1.0, 2.0, 3.0], &[1.0], None, &c).is_err());
    }

    #[test]
    fn error_out_channels_not_divisible_by_groups() {
        let c = cfg(2, 3, 1, 1, PaddingMode::Zero(0), 1, 2, false);
        assert!(conv1d_forward(&[1.0, 2.0], &[1.0], None, &c).is_err());
    }

    #[test]
    fn error_weight_length_mismatch() {
        let c = cfg(1, 1, 2, 1, PaddingMode::Zero(0), 1, 1, false);
        // expects 2 weights, provide 3
        assert!(conv1d_forward(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], None, &c).is_err());
    }

    #[test]
    fn error_bias_length_mismatch() {
        let c = cfg(1, 2, 1, 1, PaddingMode::Zero(0), 1, 1, true);
        let bad_bias = vec![1.0]; // expects 2
        assert!(conv1d_forward(&[1.0], &[1.0, 2.0], Some(&bad_bias), &c).is_err());
    }

    #[test]
    fn error_bias_expected_but_none() {
        let c = cfg(1, 1, 1, 1, PaddingMode::Zero(0), 1, 1, true);
        assert!(conv1d_forward(&[1.0], &[1.0], None, &c).is_err());
    }

    #[test]
    fn error_input_not_divisible_by_channels() {
        let c = cfg(2, 1, 1, 1, PaddingMode::Zero(0), 1, 1, false);
        // 3 elements not divisible by 2 channels
        assert!(conv1d_forward(&[1.0, 2.0, 3.0], &[1.0], None, &c).is_err());
    }

    #[test]
    fn error_kernel_larger_than_input() {
        let c = cfg(1, 1, 5, 1, PaddingMode::Zero(0), 1, 1, false);
        assert!(conv1d_forward(&[1.0, 2.0], &[1.0; 5], None, &c).is_err());
    }

    // ── Output width helper ───────────────────────────────

    #[test]
    fn output_width_formula_cases() {
        // No padding: (5 - 3) / 1 + 1 = 3
        let c = cfg(1, 1, 3, 1, PaddingMode::Zero(0), 1, 1, false);
        assert_eq!(conv1d_output_width(&c, 5), 3);

        // With padding: (5 + 2*1 - 3) / 1 + 1 = 5
        let c = cfg(1, 1, 3, 1, PaddingMode::Zero(1), 1, 1, false);
        assert_eq!(conv1d_output_width(&c, 5), 5);

        // Stride 2: (6 - 2) / 2 + 1 = 3
        let c = cfg(1, 1, 2, 2, PaddingMode::Zero(0), 1, 1, false);
        assert_eq!(conv1d_output_width(&c, 6), 3);

        // Dilation 2: ek=3, (5 - 3) / 1 + 1 = 3
        let c = cfg(1, 1, 2, 1, PaddingMode::Zero(0), 2, 1, false);
        assert_eq!(conv1d_output_width(&c, 5), 3);

        // Same: ceil(5/1) = 5
        let c = cfg(1, 1, 3, 1, PaddingMode::Same, 1, 1, false);
        assert_eq!(conv1d_output_width(&c, 5), 5);

        // Same stride 2: ceil(5/2) = 3
        let c = cfg(1, 1, 3, 2, PaddingMode::Same, 1, 1, false);
        assert_eq!(conv1d_output_width(&c, 5), 3);
    }

    // ── Property tests ────────────────────────────────────

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_output_length(
                input_width in 1usize..=64,
                kernel_size in 1usize..=8,
                stride in 1usize..=4,
                pad in 0usize..=4,
                dilation in 1usize..=3,
            ) {
                let ek = dilation * (kernel_size - 1) + 1;
                let padded = input_width + 2 * pad;
                if padded >= ek {
                    let expected = (padded - ek) / stride + 1;
                    let c = cfg(1, 1, kernel_size, stride, PaddingMode::Zero(pad), dilation, 1, false);
                    prop_assert_eq!(conv1d_output_width(&c, input_width), expected);
                }
            }

            #[test]
            fn prop_same_padding_preserves_ceil(
                input_width in 1usize..=64,
                kernel_size in 1usize..=8,
                stride in 1usize..=4,
                dilation in 1usize..=3,
            ) {
                let expected = (input_width + stride - 1) / stride;
                let c = cfg(1, 1, kernel_size, stride, PaddingMode::Same, dilation, 1, false);
                prop_assert_eq!(conv1d_output_width(&c, input_width), expected);
            }

            #[test]
            fn prop_output_element_count(
                input_width in 1usize..=32,
                kernel_size in 1usize..=4,
                stride in 1usize..=3,
            ) {
                prop_assume!(input_width >= kernel_size);
                let c = cfg(1, 1, kernel_size, stride, PaddingMode::Zero(0), 1, 1, false);
                let out_w = conv1d_output_width(&c, input_width);
                let input: Vec<f32> = (0..input_width).map(|i| i as f32).collect();
                let weight: Vec<f32> = vec![1.0; kernel_size];
                let out = conv1d_forward(&input, &weight, None, &c).unwrap();
                prop_assert_eq!(out.len(), out_w);
            }

            #[test]
            fn prop_identity_kernel(
                input_width in 1usize..=64,
            ) {
                let input: Vec<f32> = (0..input_width).map(|i| i as f32 * 0.1).collect();
                let weight = vec![1.0f32];
                let c = cfg(1, 1, 1, 1, PaddingMode::Zero(0), 1, 1, false);
                let out = conv1d_forward(&input, &weight, None, &c).unwrap();
                prop_assert!(
                    approx_eq(&out, &input, TOL),
                    "identity kernel should preserve input"
                );
            }
        }
    }
}

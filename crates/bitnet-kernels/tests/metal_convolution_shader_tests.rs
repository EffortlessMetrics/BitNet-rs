//! Metal convolution shader validation tests for Apple Silicon.
//!
//! Tests validate convolution operation parameters, output shapes,
//! and threadgroup configurations for Metal compute shaders.
//! All tests run on CPU without requiring Metal/GPU hardware.

// ---------------------------------------------------------------------------
// Convolution parameter types
// ---------------------------------------------------------------------------

/// 1D convolution parameters.
#[derive(Debug, Clone)]
struct Conv1DParams {
    batch: usize,
    in_channels: usize,
    in_length: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
}

impl Conv1DParams {
    fn new(
        batch: usize,
        in_channels: usize,
        in_length: usize,
        out_channels: usize,
        kernel_size: usize,
    ) -> Self {
        Self {
            batch,
            in_channels,
            in_length,
            out_channels,
            kernel_size,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
        }
    }

    fn output_length(&self) -> usize {
        let effective_kernel = self.dilation * (self.kernel_size - 1) + 1;
        (self.in_length + 2 * self.padding - effective_kernel) / self.stride + 1
    }

    fn output_shape(&self) -> (usize, usize, usize) {
        (self.batch, self.out_channels, self.output_length())
    }

    fn output_numel(&self) -> usize {
        let (b, c, l) = self.output_shape();
        b * c * l
    }

    fn weight_numel(&self) -> usize {
        self.out_channels * (self.in_channels / self.groups) * self.kernel_size
    }

    fn is_valid(&self) -> bool {
        if self.groups == 0 || self.stride == 0 || self.dilation == 0 || self.kernel_size == 0 {
            return false;
        }
        if self.in_channels % self.groups != 0 || self.out_channels % self.groups != 0 {
            return false;
        }
        let effective_kernel = self.dilation * (self.kernel_size - 1) + 1;
        self.in_length + 2 * self.padding >= effective_kernel
    }
}

/// 2D convolution parameters.
#[derive(Debug, Clone)]
struct ConvParams {
    input_shape: Vec<usize>,  // [batch, channels, height, width]
    kernel_shape: Vec<usize>, // [out_channels, in_channels/groups, kH, kW]
    stride: Vec<usize>,       // [sH, sW]
    padding: Vec<usize>,      // [pH, pW]
    dilation: Vec<usize>,     // [dH, dW]
    groups: usize,
}

impl ConvParams {
    fn new(input_shape: Vec<usize>, kernel_shape: Vec<usize>) -> Self {
        Self {
            input_shape,
            kernel_shape,
            stride: vec![1, 1],
            padding: vec![0, 0],
            dilation: vec![1, 1],
            groups: 1,
        }
    }

    fn with_stride(mut self, sh: usize, sw: usize) -> Self {
        self.stride = vec![sh, sw];
        self
    }

    fn with_padding(mut self, ph: usize, pw: usize) -> Self {
        self.padding = vec![ph, pw];
        self
    }

    fn with_dilation(mut self, dh: usize, dw: usize) -> Self {
        self.dilation = vec![dh, dw];
        self
    }

    fn with_groups(mut self, g: usize) -> Self {
        self.groups = g;
        self
    }

    fn batch(&self) -> usize {
        self.input_shape[0]
    }
    fn in_channels(&self) -> usize {
        self.input_shape[1]
    }
    fn in_h(&self) -> usize {
        self.input_shape[2]
    }
    fn in_w(&self) -> usize {
        self.input_shape[3]
    }
    fn out_channels(&self) -> usize {
        self.kernel_shape[0]
    }
    fn kh(&self) -> usize {
        self.kernel_shape[2]
    }
    fn kw(&self) -> usize {
        self.kernel_shape[3]
    }

    fn output_h(&self) -> usize {
        let eff = self.dilation[0] * (self.kh() - 1) + 1;
        (self.in_h() + 2 * self.padding[0] - eff) / self.stride[0] + 1
    }

    fn output_w(&self) -> usize {
        let eff = self.dilation[1] * (self.kw() - 1) + 1;
        (self.in_w() + 2 * self.padding[1] - eff) / self.stride[1] + 1
    }

    fn output_shape(&self) -> Vec<usize> {
        vec![self.batch(), self.out_channels(), self.output_h(), self.output_w()]
    }

    fn output_numel(&self) -> usize {
        self.output_shape().iter().product()
    }

    fn im2col_size(&self) -> usize {
        let col_h = self.in_channels() / self.groups * self.kh() * self.kw();
        let col_w = self.output_h() * self.output_w();
        self.batch() * col_h * col_w
    }

    /// Metal-optimal threadgroup size for 2D convolution dispatch.
    fn threadgroup_size(&self) -> (usize, usize, usize) {
        let oh = self.output_h();
        let ow = self.output_w();
        let oc = self.out_channels();

        // Apple Silicon max threads-per-threadgroup = 1024
        const MAX_THREADS: usize = 1024;
        // Prefer 32-wide for SIMD lane utilization on Apple GPU
        let tw = ow.min(32).max(1);
        let th = oh.min(MAX_THREADS / tw).max(1);
        let td = oc.min(MAX_THREADS / (tw * th)).max(1);
        assert!(tw * th * td <= MAX_THREADS);
        (tw, th, td)
    }

    fn is_valid(&self) -> bool {
        if self.groups == 0 {
            return false;
        }
        if self.stride.iter().any(|&s| s == 0) {
            return false;
        }
        if self.dilation.iter().any(|&d| d == 0) {
            return false;
        }
        if self.kh() == 0 || self.kw() == 0 {
            return false;
        }
        if self.in_channels() % self.groups != 0 {
            return false;
        }
        if self.out_channels() % self.groups != 0 {
            return false;
        }
        // Check that padded input is large enough
        let eff_h = self.dilation[0] * (self.kh() - 1) + 1;
        let eff_w = self.dilation[1] * (self.kw() - 1) + 1;
        self.in_h() + 2 * self.padding[0] >= eff_h && self.in_w() + 2 * self.padding[1] >= eff_w
    }

    fn is_depthwise(&self) -> bool {
        self.groups == self.in_channels() && self.groups == self.out_channels()
    }

    fn is_pointwise(&self) -> bool {
        self.kh() == 1 && self.kw() == 1
    }
}

/// Transposed (deconvolution) parameters.
#[derive(Debug, Clone)]
struct TransposedConvParams {
    input_shape: Vec<usize>,  // [batch, channels, height, width]
    kernel_shape: Vec<usize>, // [in_channels, out_channels/groups, kH, kW]
    stride: Vec<usize>,
    padding: Vec<usize>,
    output_padding: Vec<usize>,
    dilation: Vec<usize>,
    groups: usize,
}

impl TransposedConvParams {
    fn new(input_shape: Vec<usize>, kernel_shape: Vec<usize>) -> Self {
        Self {
            input_shape,
            kernel_shape,
            stride: vec![1, 1],
            padding: vec![0, 0],
            output_padding: vec![0, 0],
            dilation: vec![1, 1],
            groups: 1,
        }
    }

    fn with_stride(mut self, sh: usize, sw: usize) -> Self {
        self.stride = vec![sh, sw];
        self
    }

    fn with_padding(mut self, ph: usize, pw: usize) -> Self {
        self.padding = vec![ph, pw];
        self
    }

    fn with_output_padding(mut self, oh: usize, ow: usize) -> Self {
        self.output_padding = vec![oh, ow];
        self
    }

    fn with_dilation(mut self, dh: usize, dw: usize) -> Self {
        self.dilation = vec![dh, dw];
        self
    }

    fn with_groups(mut self, g: usize) -> Self {
        self.groups = g;
        self
    }

    fn batch(&self) -> usize {
        self.input_shape[0]
    }
    fn in_channels(&self) -> usize {
        self.input_shape[1]
    }
    fn in_h(&self) -> usize {
        self.input_shape[2]
    }
    fn in_w(&self) -> usize {
        self.input_shape[3]
    }
    fn out_channels_per_group(&self) -> usize {
        self.kernel_shape[1]
    }
    fn out_channels(&self) -> usize {
        self.out_channels_per_group() * self.groups
    }
    fn kh(&self) -> usize {
        self.kernel_shape[2]
    }
    fn kw(&self) -> usize {
        self.kernel_shape[3]
    }

    fn output_h(&self) -> usize {
        (self.in_h() - 1) * self.stride[0] - 2 * self.padding[0]
            + self.dilation[0] * (self.kh() - 1)
            + self.output_padding[0]
            + 1
    }

    fn output_w(&self) -> usize {
        (self.in_w() - 1) * self.stride[1] - 2 * self.padding[1]
            + self.dilation[1] * (self.kw() - 1)
            + self.output_padding[1]
            + 1
    }

    fn output_shape(&self) -> Vec<usize> {
        vec![self.batch(), self.out_channels(), self.output_h(), self.output_w()]
    }

    fn is_valid(&self) -> bool {
        if self.groups == 0 {
            return false;
        }
        if self.stride.iter().any(|&s| s == 0) {
            return false;
        }
        if self.in_channels() % self.groups != 0 {
            return false;
        }
        // output_padding must be < stride
        self.output_padding[0] < self.stride[0] && self.output_padding[1] < self.stride[1]
    }
}

/// Winograd transform tile parameters.
#[derive(Debug, Clone, Copy)]
struct WinogradParams {
    tile_size: usize,   // output tile (e.g. 2 for F(2,3) or 4 for F(4,3))
    kernel_size: usize, // 3 for standard 3×3 winograd
}

impl WinogradParams {
    fn transform_size(&self) -> usize {
        self.tile_size + self.kernel_size - 1
    }

    fn transform_matrix_numel(&self) -> usize {
        let t = self.transform_size();
        t * t
    }

    fn tiles_per_dim(&self, spatial: usize, padding: usize) -> usize {
        let padded = spatial + 2 * padding;
        (padded + self.tile_size - 1) / self.tile_size
    }

    fn is_supported(&self) -> bool {
        matches!((self.tile_size, self.kernel_size), (2, 3) | (4, 3) | (6, 3) | (2, 5))
    }
}

/// Metal buffer alignment helper.
fn metal_aligned_size(size: usize) -> usize {
    // Metal requires 16-byte alignment for buffer binding offsets
    const ALIGNMENT: usize = 16;
    (size + ALIGNMENT - 1) & !(ALIGNMENT - 1)
}

/// Metal buffer alignment for page-aligned allocations.
fn metal_page_aligned_size(size: usize) -> usize {
    const PAGE: usize = 4096;
    (size + PAGE - 1) & !(PAGE - 1)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── 1D convolution parameter validation ──────────────────────────────

    #[test]
    fn conv1d_basic_output_length() {
        let p = Conv1DParams::new(1, 1, 10, 1, 3);
        assert_eq!(p.output_length(), 8);
    }

    #[test]
    fn conv1d_with_stride() {
        let mut p = Conv1DParams::new(1, 1, 10, 1, 3);
        p.stride = 2;
        assert_eq!(p.output_length(), 4);
    }

    #[test]
    fn conv1d_with_padding() {
        let mut p = Conv1DParams::new(1, 1, 10, 1, 3);
        p.padding = 1;
        assert_eq!(p.output_length(), 10); // same-padding
    }

    #[test]
    fn conv1d_with_dilation() {
        let mut p = Conv1DParams::new(1, 1, 10, 1, 3);
        p.dilation = 2;
        // effective kernel = 2*(3-1)+1 = 5
        assert_eq!(p.output_length(), 6);
    }

    #[test]
    fn conv1d_kernel_size_1() {
        let p = Conv1DParams::new(1, 1, 10, 1, 1);
        assert_eq!(p.output_length(), 10);
    }

    #[test]
    fn conv1d_stride_larger_than_kernel() {
        let mut p = Conv1DParams::new(1, 1, 10, 1, 2);
        p.stride = 3;
        // (10 - 2)/3 + 1 = 3
        assert_eq!(p.output_length(), 3);
    }

    #[test]
    fn conv1d_output_shape_batch() {
        let p = Conv1DParams::new(4, 3, 16, 8, 3);
        assert_eq!(p.output_shape(), (4, 8, 14));
    }

    #[test]
    fn conv1d_weight_numel() {
        let p = Conv1DParams::new(1, 3, 10, 8, 5);
        // 8 * 3 * 5 = 120
        assert_eq!(p.weight_numel(), 120);
    }

    #[test]
    fn conv1d_grouped_weight_numel() {
        let mut p = Conv1DParams::new(1, 6, 10, 12, 3);
        p.groups = 3;
        // 12 * (6/3) * 3 = 72
        assert_eq!(p.weight_numel(), 72);
    }

    #[test]
    fn conv1d_invalid_zero_stride() {
        let mut p = Conv1DParams::new(1, 1, 10, 1, 3);
        p.stride = 0;
        assert!(!p.is_valid());
    }

    #[test]
    fn conv1d_invalid_zero_groups() {
        let mut p = Conv1DParams::new(1, 1, 10, 1, 3);
        p.groups = 0;
        assert!(!p.is_valid());
    }

    #[test]
    fn conv1d_invalid_channel_groups() {
        let mut p = Conv1DParams::new(1, 3, 10, 8, 3);
        p.groups = 2; // 3 % 2 != 0
        assert!(!p.is_valid());
    }

    #[test]
    fn conv1d_invalid_kernel_too_large() {
        let p = Conv1DParams::new(1, 1, 3, 1, 5);
        assert!(!p.is_valid());
    }

    #[test]
    fn conv1d_valid_same_size_kernel() {
        let p = Conv1DParams::new(1, 1, 5, 1, 5);
        assert!(p.is_valid());
        assert_eq!(p.output_length(), 1);
    }

    // ── 2D convolution parameter validation ──────────────────────────────

    #[test]
    fn conv2d_basic_output_shape() {
        let p = ConvParams::new(vec![1, 1, 8, 8], vec![1, 1, 3, 3]);
        assert_eq!(p.output_shape(), vec![1, 1, 6, 6]);
    }

    #[test]
    fn conv2d_stride_2() {
        let p = ConvParams::new(vec![1, 1, 8, 8], vec![1, 1, 3, 3]).with_stride(2, 2);
        assert_eq!(p.output_shape(), vec![1, 1, 3, 3]);
    }

    #[test]
    fn conv2d_same_padding() {
        let p = ConvParams::new(vec![1, 1, 8, 8], vec![1, 1, 3, 3]).with_padding(1, 1);
        assert_eq!(p.output_shape(), vec![1, 1, 8, 8]);
    }

    #[test]
    fn conv2d_dilation_2() {
        let p = ConvParams::new(vec![1, 1, 8, 8], vec![1, 1, 3, 3]).with_dilation(2, 2);
        // effective kernel = 2*(3-1)+1 = 5 → (8-5)/1+1 = 4
        assert_eq!(p.output_shape(), vec![1, 1, 4, 4]);
    }

    #[test]
    fn conv2d_1x1_kernel() {
        let p = ConvParams::new(vec![1, 64, 16, 16], vec![128, 64, 1, 1]);
        assert_eq!(p.output_shape(), vec![1, 128, 16, 16]);
        assert!(p.is_pointwise());
    }

    #[test]
    fn conv2d_asymmetric_kernel() {
        let p = ConvParams::new(vec![1, 1, 10, 10], vec![1, 1, 1, 5]);
        assert_eq!(p.output_shape(), vec![1, 1, 10, 6]);
    }

    #[test]
    fn conv2d_asymmetric_stride() {
        let p = ConvParams::new(vec![1, 1, 10, 10], vec![1, 1, 3, 3]).with_stride(1, 2);
        assert_eq!(p.output_shape(), vec![1, 1, 8, 4]);
    }

    #[test]
    fn conv2d_asymmetric_padding() {
        let p = ConvParams::new(vec![1, 1, 8, 8], vec![1, 1, 3, 3]).with_padding(1, 2);
        assert_eq!(p.output_shape(), vec![1, 1, 8, 10]);
    }

    #[test]
    fn conv2d_large_padding() {
        let p = ConvParams::new(vec![1, 1, 4, 4], vec![1, 1, 3, 3]).with_padding(4, 4);
        // (4 + 8 - 3)/1 + 1 = 10
        assert_eq!(p.output_shape(), vec![1, 1, 10, 10]);
    }

    #[test]
    fn conv2d_stride_gt_kernel() {
        let p = ConvParams::new(vec![1, 1, 10, 10], vec![1, 1, 2, 2]).with_stride(3, 3);
        // (10 - 2)/3 + 1 = 3
        assert_eq!(p.output_shape(), vec![1, 1, 3, 3]);
    }

    #[test]
    fn conv2d_output_numel() {
        let p = ConvParams::new(vec![2, 3, 8, 8], vec![16, 3, 3, 3]);
        // 2*16*6*6 = 1152
        assert_eq!(p.output_numel(), 1152);
    }

    #[test]
    fn conv2d_batch_propagation() {
        let p = ConvParams::new(vec![8, 3, 32, 32], vec![64, 3, 3, 3]).with_padding(1, 1);
        let out = p.output_shape();
        assert_eq!(out[0], 8);
        assert_eq!(out[1], 64);
        assert_eq!(out[2], 32);
        assert_eq!(out[3], 32);
    }

    #[test]
    fn conv2d_multi_channel() {
        let p = ConvParams::new(vec![1, 3, 224, 224], vec![64, 3, 7, 7])
            .with_stride(2, 2)
            .with_padding(3, 3);
        // ResNet-style first conv: (224+6-7)/2+1 = 112
        assert_eq!(p.output_shape(), vec![1, 64, 112, 112]);
    }

    #[test]
    fn conv2d_is_valid_basic() {
        let p = ConvParams::new(vec![1, 1, 8, 8], vec![1, 1, 3, 3]);
        assert!(p.is_valid());
    }

    #[test]
    fn conv2d_invalid_zero_stride() {
        let p = ConvParams::new(vec![1, 1, 8, 8], vec![1, 1, 3, 3]).with_stride(0, 1);
        assert!(!p.is_valid());
    }

    #[test]
    fn conv2d_invalid_zero_dilation() {
        let p = ConvParams::new(vec![1, 1, 8, 8], vec![1, 1, 3, 3]).with_dilation(0, 1);
        assert!(!p.is_valid());
    }

    #[test]
    fn conv2d_invalid_kernel_too_large() {
        let p = ConvParams::new(vec![1, 1, 3, 3], vec![1, 1, 5, 5]);
        assert!(!p.is_valid());
    }

    #[test]
    fn conv2d_invalid_channel_group_mismatch() {
        let p = ConvParams::new(vec![1, 3, 8, 8], vec![4, 3, 3, 3]).with_groups(2); // 3 % 2 != 0
        assert!(!p.is_valid());
    }

    #[test]
    fn conv2d_not_pointwise() {
        let p = ConvParams::new(vec![1, 1, 8, 8], vec![1, 1, 3, 3]);
        assert!(!p.is_pointwise());
    }

    // ── Depthwise convolution ────────────────────────────────────────────

    #[test]
    fn depthwise_conv_detection() {
        let p = ConvParams::new(vec![1, 32, 8, 8], vec![32, 1, 3, 3]).with_groups(32);
        assert!(p.is_depthwise());
    }

    #[test]
    fn depthwise_conv_output_shape() {
        let p = ConvParams::new(vec![1, 64, 16, 16], vec![64, 1, 3, 3])
            .with_groups(64)
            .with_padding(1, 1);
        assert_eq!(p.output_shape(), vec![1, 64, 16, 16]);
    }

    #[test]
    fn depthwise_conv_is_valid() {
        let p = ConvParams::new(vec![1, 32, 8, 8], vec![32, 1, 3, 3]).with_groups(32);
        assert!(p.is_valid());
    }

    #[test]
    fn depthwise_conv_stride2() {
        let p = ConvParams::new(vec![1, 128, 32, 32], vec![128, 1, 3, 3])
            .with_groups(128)
            .with_stride(2, 2)
            .with_padding(1, 1);
        assert_eq!(p.output_shape(), vec![1, 128, 16, 16]);
    }

    #[test]
    fn depthwise_conv_5x5() {
        let p = ConvParams::new(vec![1, 16, 16, 16], vec![16, 1, 5, 5])
            .with_groups(16)
            .with_padding(2, 2);
        assert_eq!(p.output_shape(), vec![1, 16, 16, 16]);
    }

    #[test]
    fn non_depthwise_grouped() {
        let p = ConvParams::new(vec![1, 32, 8, 8], vec![64, 8, 3, 3]).with_groups(4);
        assert!(!p.is_depthwise());
        assert!(p.is_valid());
    }

    // ── Grouped convolution ──────────────────────────────────────────────

    #[test]
    fn grouped_conv_output_shape() {
        let p = ConvParams::new(vec![1, 32, 8, 8], vec![64, 8, 3, 3]).with_groups(4);
        assert_eq!(p.output_shape(), vec![1, 64, 6, 6]);
    }

    #[test]
    fn grouped_conv_2_groups() {
        let p = ConvParams::new(vec![1, 8, 16, 16], vec![16, 4, 3, 3])
            .with_groups(2)
            .with_padding(1, 1);
        assert_eq!(p.output_shape(), vec![1, 16, 16, 16]);
    }

    #[test]
    fn grouped_conv_weight_channels() {
        let p = ConvParams::new(vec![1, 12, 8, 8], vec![24, 4, 3, 3]).with_groups(3);
        // kernel_shape[1] should be in_channels/groups = 12/3 = 4
        assert_eq!(p.kernel_shape[1], p.in_channels() / p.groups);
    }

    #[test]
    fn grouped_conv_invalid_uneven_channels() {
        let p = ConvParams::new(vec![1, 7, 8, 8], vec![14, 7, 3, 3]).with_groups(3); // 7 % 3 != 0
        assert!(!p.is_valid());
    }

    // ── Output shape computation ─────────────────────────────────────────

    #[test]
    fn output_shape_resnet_block() {
        // Typical ResNet bottleneck: 1x1 → 3x3 → 1x1
        let p1 = ConvParams::new(vec![1, 256, 56, 56], vec![64, 256, 1, 1]);
        assert_eq!(p1.output_shape(), vec![1, 64, 56, 56]);

        let p2 = ConvParams::new(vec![1, 64, 56, 56], vec![64, 64, 3, 3]).with_padding(1, 1);
        assert_eq!(p2.output_shape(), vec![1, 64, 56, 56]);

        let p3 = ConvParams::new(vec![1, 64, 56, 56], vec![256, 64, 1, 1]);
        assert_eq!(p3.output_shape(), vec![1, 256, 56, 56]);
    }

    #[test]
    fn output_shape_mobilenet_dw() {
        // MobileNet depthwise separable
        let dw = ConvParams::new(vec![1, 32, 112, 112], vec![32, 1, 3, 3])
            .with_groups(32)
            .with_padding(1, 1);
        assert_eq!(dw.output_shape(), vec![1, 32, 112, 112]);

        let pw = ConvParams::new(vec![1, 32, 112, 112], vec![64, 32, 1, 1]);
        assert_eq!(pw.output_shape(), vec![1, 64, 112, 112]);
    }

    #[test]
    fn output_shape_dilated_cascade() {
        // Dilated convolutions with increasing rates
        let d1 = ConvParams::new(vec![1, 1, 64, 64], vec![1, 1, 3, 3])
            .with_dilation(1, 1)
            .with_padding(1, 1);
        assert_eq!(d1.output_shape(), vec![1, 1, 64, 64]);

        let d2 = ConvParams::new(vec![1, 1, 64, 64], vec![1, 1, 3, 3])
            .with_dilation(2, 2)
            .with_padding(2, 2);
        assert_eq!(d2.output_shape(), vec![1, 1, 64, 64]);

        let d4 = ConvParams::new(vec![1, 1, 64, 64], vec![1, 1, 3, 3])
            .with_dilation(4, 4)
            .with_padding(4, 4);
        assert_eq!(d4.output_shape(), vec![1, 1, 64, 64]);
    }

    #[test]
    fn output_shape_minimal_input() {
        let p = ConvParams::new(vec![1, 1, 1, 1], vec![1, 1, 1, 1]);
        assert_eq!(p.output_shape(), vec![1, 1, 1, 1]);
    }

    #[test]
    fn output_shape_large_batch() {
        let p = ConvParams::new(vec![128, 3, 32, 32], vec![64, 3, 3, 3]).with_padding(1, 1);
        assert_eq!(p.output_shape()[0], 128);
    }

    // ── Im2col buffer layout ─────────────────────────────────────────────

    #[test]
    fn im2col_size_basic() {
        let p = ConvParams::new(vec![1, 1, 8, 8], vec![1, 1, 3, 3]);
        // col_h = 1*3*3 = 9, col_w = 6*6 = 36 → 9*36 = 324
        assert_eq!(p.im2col_size(), 324);
    }

    #[test]
    fn im2col_size_multi_channel() {
        let p = ConvParams::new(vec![1, 3, 8, 8], vec![16, 3, 3, 3]);
        // col_h = 3*3*3 = 27, col_w = 6*6 = 36 → 27*36 = 972
        assert_eq!(p.im2col_size(), 972);
    }

    #[test]
    fn im2col_size_batch() {
        let p = ConvParams::new(vec![4, 3, 8, 8], vec![16, 3, 3, 3]);
        let single = ConvParams::new(vec![1, 3, 8, 8], vec![16, 3, 3, 3]).im2col_size();
        assert_eq!(p.im2col_size(), 4 * single);
    }

    #[test]
    fn im2col_size_1x1_kernel() {
        let p = ConvParams::new(vec![1, 64, 16, 16], vec![128, 64, 1, 1]);
        // col_h = 64*1*1 = 64, col_w = 16*16 = 256 → 64*256 = 16384
        assert_eq!(p.im2col_size(), 16384);
    }

    #[test]
    fn im2col_size_with_padding() {
        let p = ConvParams::new(vec![1, 1, 4, 4], vec![1, 1, 3, 3]).with_padding(1, 1);
        // output = 4x4 → col_h = 1*3*3=9, col_w = 4*4=16 → 144
        assert_eq!(p.im2col_size(), 144);
    }

    #[test]
    fn im2col_size_grouped() {
        let p = ConvParams::new(vec![1, 8, 8, 8], vec![16, 4, 3, 3]).with_groups(2);
        // col_h = (8/2)*3*3 = 36, col_w = 6*6 = 36 → 36*36 = 1296
        assert_eq!(p.im2col_size(), 1296);
    }

    // ── Winograd transform ───────────────────────────────────────────────

    #[test]
    fn winograd_f2x3_transform_size() {
        let w = WinogradParams { tile_size: 2, kernel_size: 3 };
        assert_eq!(w.transform_size(), 4);
    }

    #[test]
    fn winograd_f4x3_transform_size() {
        let w = WinogradParams { tile_size: 4, kernel_size: 3 };
        assert_eq!(w.transform_size(), 6);
    }

    #[test]
    fn winograd_f6x3_transform_size() {
        let w = WinogradParams { tile_size: 6, kernel_size: 3 };
        assert_eq!(w.transform_size(), 8);
    }

    #[test]
    fn winograd_f2x5_transform_size() {
        let w = WinogradParams { tile_size: 2, kernel_size: 5 };
        assert_eq!(w.transform_size(), 6);
    }

    #[test]
    fn winograd_transform_matrix_numel() {
        let w = WinogradParams { tile_size: 2, kernel_size: 3 };
        assert_eq!(w.transform_matrix_numel(), 16); // 4*4
    }

    #[test]
    fn winograd_tiles_per_dim_exact() {
        let w = WinogradParams { tile_size: 2, kernel_size: 3 };
        // 8 / 2 = 4 tiles exactly
        assert_eq!(w.tiles_per_dim(8, 1), 5); // (8+2)/2 = 5
    }

    #[test]
    fn winograd_tiles_per_dim_remainder() {
        let w = WinogradParams { tile_size: 4, kernel_size: 3 };
        // (14 + 2*1 + 4-1) / 4 = 19/4 = 4.75 → 5
        assert_eq!(w.tiles_per_dim(14, 1), 4); // (14+2)/4 = 4
    }

    #[test]
    fn winograd_supported_variants() {
        assert!(WinogradParams { tile_size: 2, kernel_size: 3 }.is_supported());
        assert!(WinogradParams { tile_size: 4, kernel_size: 3 }.is_supported());
        assert!(WinogradParams { tile_size: 6, kernel_size: 3 }.is_supported());
        assert!(WinogradParams { tile_size: 2, kernel_size: 5 }.is_supported());
    }

    #[test]
    fn winograd_unsupported_variants() {
        assert!(!WinogradParams { tile_size: 3, kernel_size: 3 }.is_supported());
        assert!(!WinogradParams { tile_size: 4, kernel_size: 5 }.is_supported());
        assert!(!WinogradParams { tile_size: 2, kernel_size: 7 }.is_supported());
    }

    // ── Metal threadgroup size computation ───────────────────────────────

    #[test]
    fn threadgroup_small_output() {
        let p = ConvParams::new(vec![1, 1, 4, 4], vec![1, 1, 3, 3]);
        let (tw, th, td) = p.threadgroup_size();
        assert!(tw * th * td <= 1024);
        assert!(tw >= 1 && th >= 1 && td >= 1);
    }

    #[test]
    fn threadgroup_large_output() {
        let p = ConvParams::new(vec![1, 3, 224, 224], vec![64, 3, 3, 3]).with_padding(1, 1);
        let (tw, th, td) = p.threadgroup_size();
        assert!(tw * th * td <= 1024);
        // Should use 32-wide for SIMD
        assert_eq!(tw, 32);
    }

    #[test]
    fn threadgroup_1x1_conv() {
        let p = ConvParams::new(vec![1, 64, 56, 56], vec![256, 64, 1, 1]);
        let (tw, th, td) = p.threadgroup_size();
        assert!(tw * th * td <= 1024);
        assert_eq!(tw, 32);
    }

    #[test]
    fn threadgroup_tiny_spatial() {
        let p = ConvParams::new(vec![1, 512, 2, 2], vec![512, 512, 1, 1]);
        let (tw, th, td) = p.threadgroup_size();
        assert!(tw * th * td <= 1024);
        assert_eq!(tw, 2);
    }

    #[test]
    fn threadgroup_many_channels() {
        let p = ConvParams::new(vec![1, 3, 8, 8], vec![1024, 3, 3, 3]).with_padding(1, 1);
        let (tw, th, td) = p.threadgroup_size();
        assert!(tw * th * td <= 1024);
    }

    #[test]
    fn threadgroup_total_never_exceeds_limit() {
        let configs = vec![
            (vec![1, 1, 1, 1], vec![1, 1, 1, 1]),
            (vec![1, 3, 224, 224], vec![64, 3, 7, 7]),
            (vec![1, 512, 7, 7], vec![512, 512, 3, 3]),
            (vec![1, 2048, 1, 1], vec![1000, 2048, 1, 1]),
        ];
        for (inp, ker) in configs {
            let p = ConvParams::new(inp.clone(), ker.clone()).with_padding(1, 1);
            let (tw, th, td) = p.threadgroup_size();
            assert!(
                tw * th * td <= 1024,
                "threadgroup exceeded 1024 for input={inp:?} kernel={ker:?}: {tw}*{th}*{td}={}",
                tw * th * td
            );
        }
    }

    // ── Metal buffer alignment ───────────────────────────────────────────

    #[test]
    fn alignment_zero() {
        assert_eq!(metal_aligned_size(0), 0);
    }

    #[test]
    fn alignment_exact_16() {
        assert_eq!(metal_aligned_size(16), 16);
    }

    #[test]
    fn alignment_round_up() {
        assert_eq!(metal_aligned_size(1), 16);
        assert_eq!(metal_aligned_size(15), 16);
        assert_eq!(metal_aligned_size(17), 32);
    }

    #[test]
    fn alignment_f32_buffer() {
        // 100 f32 values = 400 bytes → aligned to 400 (already 16-aligned: 400/16=25)
        assert_eq!(metal_aligned_size(100 * 4), 400);
    }

    #[test]
    fn alignment_f16_buffer() {
        // 100 f16 values = 200 bytes → aligned to 208 (200/16=12.5 → 13*16=208)
        assert_eq!(metal_aligned_size(100 * 2), 208);
    }

    #[test]
    fn page_alignment_small() {
        assert_eq!(metal_page_aligned_size(1), 4096);
    }

    #[test]
    fn page_alignment_exact() {
        assert_eq!(metal_page_aligned_size(4096), 4096);
    }

    #[test]
    fn page_alignment_large() {
        assert_eq!(metal_page_aligned_size(5000), 8192);
    }

    // ── Data type simulation ─────────────────────────────────────────────

    #[test]
    fn f32_conv_buffer_size() {
        let p = ConvParams::new(vec![1, 3, 32, 32], vec![16, 3, 3, 3]).with_padding(1, 1);
        let output_bytes = p.output_numel() * std::mem::size_of::<f32>();
        assert_eq!(p.output_numel(), 1 * 16 * 32 * 32);
        assert_eq!(output_bytes, 16384 * 4);
    }

    #[test]
    fn f16_simulated_buffer_size() {
        let p = ConvParams::new(vec![1, 3, 32, 32], vec![16, 3, 3, 3]).with_padding(1, 1);
        // f16 = 2 bytes per element
        let output_bytes = p.output_numel() * 2;
        assert_eq!(output_bytes, 16384 * 2);
    }

    #[test]
    fn im2col_f32_memory() {
        let p = ConvParams::new(vec![1, 3, 224, 224], vec![64, 3, 7, 7])
            .with_stride(2, 2)
            .with_padding(3, 3);
        let mem = p.im2col_size() * std::mem::size_of::<f32>();
        // Should be reasonable (< 1GB)
        assert!(mem < 1_000_000_000);
        assert!(mem > 0);
    }

    // ── Batch convolution ────────────────────────────────────────────────

    #[test]
    fn batch_output_scales_linearly() {
        let p1 = ConvParams::new(vec![1, 3, 16, 16], vec![8, 3, 3, 3]);
        let p4 = ConvParams::new(vec![4, 3, 16, 16], vec![8, 3, 3, 3]);
        assert_eq!(p4.output_numel(), 4 * p1.output_numel());
    }

    #[test]
    fn batch_shape_preserves_spatial() {
        let p = ConvParams::new(vec![16, 3, 32, 32], vec![64, 3, 3, 3]).with_padding(1, 1);
        let out = p.output_shape();
        assert_eq!(out[2], 32);
        assert_eq!(out[3], 32);
    }

    #[test]
    fn batch_im2col_scales() {
        let p1 = ConvParams::new(vec![1, 3, 8, 8], vec![16, 3, 3, 3]);
        let p8 = ConvParams::new(vec![8, 3, 8, 8], vec![16, 3, 3, 3]);
        assert_eq!(p8.im2col_size(), 8 * p1.im2col_size());
    }

    // ── Transposed convolution (deconvolution) ───────────────────────────

    #[test]
    fn transposed_basic_output_shape() {
        let p = TransposedConvParams::new(vec![1, 1, 4, 4], vec![1, 1, 3, 3]);
        // (4-1)*1 - 0 + 2 + 0 + 1 = 6
        assert_eq!(p.output_shape(), vec![1, 1, 6, 6]);
    }

    #[test]
    fn transposed_stride2() {
        let p = TransposedConvParams::new(vec![1, 1, 4, 4], vec![1, 1, 3, 3]).with_stride(2, 2);
        // (4-1)*2 - 0 + 2 + 0 + 1 = 9
        assert_eq!(p.output_shape(), vec![1, 1, 9, 9]);
    }

    #[test]
    fn transposed_with_padding() {
        let p = TransposedConvParams::new(vec![1, 1, 4, 4], vec![1, 1, 3, 3])
            .with_stride(2, 2)
            .with_padding(1, 1);
        // (4-1)*2 - 2 + 2 + 0 + 1 = 7
        assert_eq!(p.output_shape(), vec![1, 1, 7, 7]);
    }

    #[test]
    fn transposed_with_output_padding() {
        let p = TransposedConvParams::new(vec![1, 1, 4, 4], vec![1, 1, 3, 3])
            .with_stride(2, 2)
            .with_padding(1, 1)
            .with_output_padding(1, 1);
        // (4-1)*2 - 2 + 2 + 1 + 1 = 8
        assert_eq!(p.output_shape(), vec![1, 1, 8, 8]);
    }

    #[test]
    fn transposed_upsample_2x() {
        // Common upsampling: stride=2, kernel=4, padding=1
        let p = TransposedConvParams::new(vec![1, 64, 16, 16], vec![64, 32, 4, 4])
            .with_stride(2, 2)
            .with_padding(1, 1);
        // (16-1)*2 - 2 + 3 + 0 + 1 = 32
        assert_eq!(p.output_shape(), vec![1, 32, 32, 32]);
    }

    #[test]
    fn transposed_grouped() {
        let p = TransposedConvParams::new(vec![1, 8, 4, 4], vec![8, 2, 3, 3]).with_groups(4);
        assert_eq!(p.out_channels(), 8); // 2 * 4
    }

    #[test]
    fn transposed_is_valid() {
        let p = TransposedConvParams::new(vec![1, 1, 4, 4], vec![1, 1, 3, 3]).with_stride(2, 2);
        assert!(p.is_valid());
    }

    #[test]
    fn transposed_invalid_output_padding() {
        // output_padding must be < stride
        let p = TransposedConvParams::new(vec![1, 1, 4, 4], vec![1, 1, 3, 3])
            .with_stride(2, 2)
            .with_output_padding(2, 2); // 2 >= 2 → invalid
        assert!(!p.is_valid());
    }

    #[test]
    fn transposed_invalid_groups() {
        let p = TransposedConvParams::new(vec![1, 3, 4, 4], vec![3, 1, 3, 3]).with_groups(0);
        assert!(!p.is_valid());
    }

    #[test]
    fn transposed_dilation() {
        let p = TransposedConvParams::new(vec![1, 1, 4, 4], vec![1, 1, 3, 3]).with_dilation(2, 2);
        // (4-1)*1 - 0 + 2*(3-1) + 0 + 1 = 3 + 4 + 1 = 8
        assert_eq!(p.output_shape(), vec![1, 1, 8, 8]);
    }

    // ── Edge cases ───────────────────────────────────────────────────────

    #[test]
    fn conv2d_single_element_output() {
        let p = ConvParams::new(vec![1, 1, 3, 3], vec![1, 1, 3, 3]);
        assert_eq!(p.output_shape(), vec![1, 1, 1, 1]);
    }

    #[test]
    fn conv2d_wide_kernel() {
        let p = ConvParams::new(vec![1, 1, 1, 32], vec![1, 1, 1, 7]).with_padding(0, 3);
        assert_eq!(p.output_shape(), vec![1, 1, 1, 32]);
    }

    #[test]
    fn conv2d_tall_kernel() {
        let p = ConvParams::new(vec![1, 1, 32, 1], vec![1, 1, 7, 1]).with_padding(3, 0);
        assert_eq!(p.output_shape(), vec![1, 1, 32, 1]);
    }

    #[test]
    fn conv2d_dilation_and_stride() {
        let p = ConvParams::new(vec![1, 1, 16, 16], vec![1, 1, 3, 3])
            .with_stride(2, 2)
            .with_dilation(2, 2);
        // eff_kernel = 2*(3-1)+1 = 5, (16-5)/2+1 = 6
        assert_eq!(p.output_shape(), vec![1, 1, 6, 6]);
    }

    #[test]
    fn conv2d_large_dilation() {
        let p = ConvParams::new(vec![1, 1, 32, 32], vec![1, 1, 3, 3])
            .with_dilation(8, 8)
            .with_padding(8, 8);
        // eff = 8*(3-1)+1 = 17, (32+16-17)/1+1 = 32
        assert_eq!(p.output_shape(), vec![1, 1, 32, 32]);
    }

    #[test]
    fn conv2d_stride_equals_kernel() {
        // Non-overlapping patches
        let p = ConvParams::new(vec![1, 1, 16, 16], vec![1, 1, 4, 4]).with_stride(4, 4);
        assert_eq!(p.output_shape(), vec![1, 1, 4, 4]);
    }
}

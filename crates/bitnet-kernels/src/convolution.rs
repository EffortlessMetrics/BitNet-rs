use bitnet_common::{QuantizationType, Result};

/// Configuration parameters for 2D convolution operations.
#[derive(Clone, Copy, Debug)]
pub struct Conv2DParams {
    /// Stride along (height, width)
    pub stride: (usize, usize),
    /// Padding along (height, width)
    pub padding: (usize, usize),
    /// Dilation along (height, width)
    pub dilation: (usize, usize),
}

impl Default for Conv2DParams {
    fn default() -> Self {
        Self { stride: (1, 1), padding: (0, 0), dilation: (1, 1) }
    }
}

/// Perform a naive 2D convolution.
///
/// This implementation follows BitNet-rs patterns with proper error handling,
/// input validation, and efficient memory access patterns. It supports stride,
/// padding, and dilation operations commonly used in neural networks.
///
/// # Arguments
/// * `input` - Input tensor data in NCHW format (batch, channels, height, width)
/// * `weight` - Convolution kernel weights in OIHW format (out_channels, in_channels, height, width)
/// * `bias` - Optional bias vector with length equal to output channels
/// * `output` - Output buffer to store convolution results
/// * `params` - Convolution parameters (stride, padding, dilation)
/// * `input_dims` - Input tensor dimensions (N, C, H, W)
/// * `weight_dims` - Weight tensor dimensions (O, I, H, W)
pub fn conv2d(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    params: Conv2DParams,
    input_dims: (usize, usize, usize, usize),
    weight_dims: (usize, usize, usize, usize),
) -> Result<()> {
    // Extract dimensions
    let (n, ic, ih, iw) = input_dims;
    let (oc, kic, kh, kw) = weight_dims;

    // Validate input dimensions
    if ic != kic {
        return Err(bitnet_common::BitNetError::Kernel(
            bitnet_common::KernelError::InvalidArguments {
                reason: format!("input channels ({}) != weight input channels ({})", ic, kic),
            },
        ));
    }

    // Validate tensor sizes
    let expected_input_size = n * ic * ih * iw;
    let expected_weight_size = oc * kic * kh * kw;

    if input.len() != expected_input_size {
        return Err(bitnet_common::BitNetError::Kernel(
            bitnet_common::KernelError::InvalidArguments {
                reason: format!(
                    "input size mismatch: expected {}, got {}",
                    expected_input_size,
                    input.len()
                ),
            },
        ));
    }

    if weight.len() != expected_weight_size {
        return Err(bitnet_common::BitNetError::Kernel(
            bitnet_common::KernelError::InvalidArguments {
                reason: format!(
                    "weight size mismatch: expected {}, got {}",
                    expected_weight_size,
                    weight.len()
                ),
            },
        ));
    }

    // Validate bias dimensions if provided
    if let Some(bias_data) = bias
        && bias_data.len() != oc
    {
        return Err(bitnet_common::BitNetError::Kernel(
            bitnet_common::KernelError::InvalidArguments {
                reason: format!("bias size mismatch: expected {}, got {}", oc, bias_data.len()),
            },
        ));
    }

    // Calculate output dimensions
    let oh = (ih + 2 * params.padding.0 - params.dilation.0 * (kh - 1) - 1) / params.stride.0 + 1;
    let ow = (iw + 2 * params.padding.1 - params.dilation.1 * (kw - 1) - 1) / params.stride.1 + 1;

    let expected_output_size = n * oc * oh * ow;
    if output.len() != expected_output_size {
        return Err(bitnet_common::BitNetError::Kernel(
            bitnet_common::KernelError::InvalidArguments {
                reason: format!(
                    "output size mismatch: expected {}, got {}",
                    expected_output_size,
                    output.len()
                ),
            },
        ));
    }

    // Initialize output with zeros or bias
    output.fill(0.0);

    // Apply bias if provided
    if let Some(bias_data) = bias {
        for batch in 0..n {
            #[allow(clippy::needless_range_loop)]
            for out_ch in 0..oc {
                let bias_val = bias_data[out_ch];
                for y in 0..oh {
                    for x in 0..ow {
                        let output_idx = batch * (oc * oh * ow) + out_ch * (oh * ow) + y * ow + x;
                        output[output_idx] = bias_val;
                    }
                }
            }
        }
    }

    // Perform convolution
    for batch in 0..n {
        for out_ch in 0..oc {
            for in_ch in 0..ic {
                for y in 0..oh {
                    for x in 0..ow {
                        let output_idx = batch * (oc * oh * ow) + out_ch * (oh * ow) + y * ow + x;

                        // Convolve with kernel
                        for ky in 0..kh {
                            for kx in 0..kw {
                                // Calculate input coordinates with padding and dilation
                                let iy = y * params.stride.0 + ky * params.dilation.0;
                                let ix = x * params.stride.1 + kx * params.dilation.1;

                                // Check bounds with padding
                                if iy >= params.padding.0 && ix >= params.padding.1 {
                                    let iy_actual = iy - params.padding.0;
                                    let ix_actual = ix - params.padding.1;

                                    if iy_actual < ih && ix_actual < iw {
                                        let input_idx = batch * (ic * ih * iw)
                                            + in_ch * (ih * iw)
                                            + iy_actual * iw
                                            + ix_actual;
                                        let weight_idx = out_ch * (kic * kh * kw)
                                            + in_ch * (kh * kw)
                                            + ky * kw
                                            + kx;

                                        output[output_idx] += input[input_idx] * weight[weight_idx];
                                    }
                                    // Padding regions contribute 0, so we skip them
                                }
                                // Padding regions contribute 0, so we skip them
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// Perform quantized 2D convolution following BitNet patterns.
///
/// This function provides the same interface as conv2d but operates on quantized
/// weights, integrating with BitNet's quantization framework. It supports 1-bit
/// and other quantization schemes commonly used in BitNet models.
///
/// # Arguments
/// * `input` - Input tensor data in NCHW format (batch, channels, height, width)
/// * `weight_quantized` - Quantized convolution kernel weights
/// * `weight_scales` - Scale factors for dequantizing weights per output channel
/// * `bias` - Optional bias vector with length equal to output channels
/// * `output` - Output buffer to store convolution results
/// * `params` - Convolution parameters (stride, padding, dilation)
/// * `input_dims` - Input tensor dimensions (N, C, H, W)
/// * `weight_dims` - Weight tensor dimensions (O, I, H, W)
/// * `qtype` - Quantization type for the weights
#[allow(clippy::too_many_arguments)]
pub fn conv2d_quantized(
    input: &[f32],
    weight_quantized: &[u8],
    weight_scales: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    params: Conv2DParams,
    input_dims: (usize, usize, usize, usize),
    weight_dims: (usize, usize, usize, usize),
    qtype: QuantizationType,
) -> Result<()> {
    let (n, ic, ih, iw) = input_dims;
    let (oc, kic, kh, kw) = weight_dims;

    // Validate input dimensions
    if ic != kic {
        return Err(bitnet_common::BitNetError::Kernel(
            bitnet_common::KernelError::InvalidArguments {
                reason: format!("input channels ({}) != weight input channels ({})", ic, kic),
            },
        ));
    }

    // Validate scale dimensions
    if weight_scales.len() != oc {
        return Err(bitnet_common::BitNetError::Kernel(
            bitnet_common::KernelError::InvalidArguments {
                reason: format!(
                    "weight_scales size mismatch: expected {}, got {}",
                    oc,
                    weight_scales.len()
                ),
            },
        ));
    }

    // Calculate expected quantized weight size based on quantization type
    let elements_per_weight = kic * kh * kw;
    let expected_weight_size = match qtype {
        QuantizationType::I2S => (oc * elements_per_weight).div_ceil(4), // 2 bits per element, packed
        QuantizationType::TL1 | QuantizationType::TL2 => oc * elements_per_weight,
    };

    if weight_quantized.len() != expected_weight_size {
        return Err(bitnet_common::BitNetError::Kernel(
            bitnet_common::KernelError::InvalidArguments {
                reason: format!(
                    "quantized weight size mismatch: expected {}, got {}",
                    expected_weight_size,
                    weight_quantized.len()
                ),
            },
        ));
    }

    // Validate other dimensions similar to fp32 conv2d
    let expected_input_size = n * ic * ih * iw;
    if input.len() != expected_input_size {
        return Err(bitnet_common::BitNetError::Kernel(
            bitnet_common::KernelError::InvalidArguments {
                reason: format!(
                    "input size mismatch: expected {}, got {}",
                    expected_input_size,
                    input.len()
                ),
            },
        ));
    }

    if let Some(bias_data) = bias
        && bias_data.len() != oc
    {
        return Err(bitnet_common::BitNetError::Kernel(
            bitnet_common::KernelError::InvalidArguments {
                reason: format!("bias size mismatch: expected {}, got {}", oc, bias_data.len()),
            },
        ));
    }

    // Calculate output dimensions
    let oh = (ih + 2 * params.padding.0 - params.dilation.0 * (kh - 1) - 1) / params.stride.0 + 1;
    let ow = (iw + 2 * params.padding.1 - params.dilation.1 * (kw - 1) - 1) / params.stride.1 + 1;

    let expected_output_size = n * oc * oh * ow;
    if output.len() != expected_output_size {
        return Err(bitnet_common::BitNetError::Kernel(
            bitnet_common::KernelError::InvalidArguments {
                reason: format!(
                    "output size mismatch: expected {}, got {}",
                    expected_output_size,
                    output.len()
                ),
            },
        ));
    }

    // Initialize output with zeros or bias
    output.fill(0.0);

    // Apply bias if provided
    if let Some(bias_data) = bias {
        for batch in 0..n {
            #[allow(clippy::needless_range_loop)]
            for out_ch in 0..oc {
                let bias_val = bias_data[out_ch];
                for y in 0..oh {
                    for x in 0..ow {
                        let output_idx = batch * (oc * oh * ow) + out_ch * (oh * ow) + y * ow + x;
                        output[output_idx] = bias_val;
                    }
                }
            }
        }
    }

    // Perform quantized convolution with dequantization on-the-fly
    for batch in 0..n {
        #[allow(clippy::needless_range_loop)]
        for out_ch in 0..oc {
            let scale = weight_scales[out_ch];

            for in_ch in 0..ic {
                for y in 0..oh {
                    for x in 0..ow {
                        let output_idx = batch * (oc * oh * ow) + out_ch * (oh * ow) + y * ow + x;

                        // Convolve with dequantized kernel
                        for ky in 0..kh {
                            for kx in 0..kw {
                                // Calculate input coordinates with padding and dilation
                                let iy = y * params.stride.0 + ky * params.dilation.0;
                                let ix = x * params.stride.1 + kx * params.dilation.1;

                                // Check bounds with padding
                                if iy >= params.padding.0 && ix >= params.padding.1 {
                                    let iy_actual = iy - params.padding.0;
                                    let ix_actual = ix - params.padding.1;

                                    if iy_actual < ih && ix_actual < iw {
                                        let input_idx = batch * (ic * ih * iw)
                                            + in_ch * (ih * iw)
                                            + iy_actual * iw
                                            + ix_actual;

                                        // Get quantized weight and dequantize
                                        let weight_linear_idx = out_ch * elements_per_weight
                                            + in_ch * (kh * kw)
                                            + ky * kw
                                            + kx;
                                        let weight_val = dequantize_weight(
                                            weight_quantized,
                                            weight_linear_idx,
                                            qtype,
                                            scale,
                                        )?;

                                        output[output_idx] += input[input_idx] * weight_val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// Dequantize a single weight value based on the quantization type.
///
/// This function follows BitNet's quantization patterns and supports the
/// major quantization schemes used in the framework.
fn dequantize_weight(
    weight_quantized: &[u8],
    linear_idx: usize,
    qtype: QuantizationType,
    scale: f32,
) -> Result<f32> {
    match qtype {
        QuantizationType::I2S => {
            // 2-bit signed quantization: 4 values packed per byte
            let byte_idx = linear_idx / 4;
            let bit_offset = (linear_idx % 4) * 2;

            if byte_idx >= weight_quantized.len() {
                return Err(bitnet_common::BitNetError::Kernel(
                    bitnet_common::KernelError::InvalidArguments {
                        reason: format!("weight index {} out of bounds", byte_idx),
                    },
                ));
            }

            let quantized_byte = weight_quantized[byte_idx];
            let quantized_val = (quantized_byte >> bit_offset) & 0x03;

            // I2S uses signed 2-bit values: 00->-2, 01->-1, 10->1, 11->2
            let dequantized = match quantized_val {
                0x00 => -2.0,
                0x01 => -1.0,
                0x02 => 1.0,         // 0x10 in binary is 0x02 in hex
                0x03 => 2.0,         // 0x11 in binary is 0x03 in hex
                _ => unreachable!(), // Only 2 bits possible
            };

            Ok(dequantized * scale)
        }
        QuantizationType::TL1 => {
            // Table lookup 1: Direct byte-to-float mapping
            if linear_idx >= weight_quantized.len() {
                return Err(bitnet_common::BitNetError::Kernel(
                    bitnet_common::KernelError::InvalidArguments {
                        reason: format!("weight index {} out of bounds", linear_idx),
                    },
                ));
            }

            let quantized_val = weight_quantized[linear_idx];
            // Simple linear dequantization for TL1
            let dequantized = (quantized_val as f32 - 128.0) / 127.0; // Map [0,255] to [-1,1]
            Ok(dequantized * scale)
        }
        QuantizationType::TL2 => {
            // Table lookup 2: More sophisticated mapping
            if linear_idx >= weight_quantized.len() {
                return Err(bitnet_common::BitNetError::Kernel(
                    bitnet_common::KernelError::InvalidArguments {
                        reason: format!("weight index {} out of bounds", linear_idx),
                    },
                ));
            }

            let quantized_val = weight_quantized[linear_idx];
            // Non-linear dequantization for TL2 (simplified)
            let normalized = quantized_val as f32 / 255.0; // Map [0,255] to [0,1]
            let dequantized = 2.0 * normalized - 1.0; // Map to [-1,1]
            Ok(dequantized * scale)
        }
    }
}

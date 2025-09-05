use std::process::Command;

/// Integration tests comparing the conv2d implementation against PyTorch.
///
/// These tests validate the correctness of our convolution implementation by
/// comparing results with PyTorch's reference implementation. The tests are
/// marked as ignored by default to avoid requiring Python/PyTorch in CI,
/// but can be enabled with `cargo test -- --ignored` for validation.
#[test]
#[ignore = "requires python3 and PyTorch for reference implementation"]
fn conv2d_reference_cases() {
    let script = r#"
import json, torch, torch.nn.functional as F
cases = [
    ((1,1,4,4),(1,1,3,3),(1,1),(0,0),(1,1)),
    ((1,1,5,5),(1,1,3,3),(2,2),(1,1),(1,1)),
    ((1,1,6,6),(1,1,2,2),(1,1),(0,0),(2,2)),
]
results = []
for shape_in, shape_w, stride, pad, dil in cases:
    x = torch.randn(*shape_in)
    w = torch.randn(*shape_w)
    y = F.conv2d(x, w, None, stride=stride, padding=pad, dilation=dil)
    results.append({
        'input': x.flatten().tolist(),
        'weight': w.flatten().tolist(),
        'output': y.flatten().tolist(),
        'cfg': {
            'input_shape': shape_in,
            'weight_shape': shape_w,
            'stride': stride,
            'padding': pad,
            'dilation': dil,
        }
    })
print(json.dumps(results))
"#;

    let output = Command::new("python3")
        .arg("-c")
        .arg(script)
        .output()
        .expect("failed to run python script");
    assert!(
        output.status.success(),
        "python script failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let cases: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("invalid json from python");
    for case in cases.as_array().expect("cases should be array") {
        let cfg = &case["cfg"];
        let stride = (
            cfg["stride"][0].as_u64().unwrap() as usize,
            cfg["stride"][1].as_u64().unwrap() as usize,
        );
        let padding = (
            cfg["padding"][0].as_u64().unwrap() as usize,
            cfg["padding"][1].as_u64().unwrap() as usize,
        );
        let dilation = (
            cfg["dilation"][0].as_u64().unwrap() as usize,
            cfg["dilation"][1].as_u64().unwrap() as usize,
        );

        // Parse test data from PyTorch reference
        let input: Vec<f32> =
            case["input"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
        let weight: Vec<f32> =
            case["weight"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
        let output_ref: Vec<f32> =
            case["output"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();

        let mut output_buf = vec![0f32; output_ref.len()];

        // Call our conv2d implementation
        let result = bitnet_kernels::convolution::conv2d(
            &input,
            &weight,
            None,
            &mut output_buf,
            bitnet_kernels::convolution::Conv2DParams { stride, padding, dilation },
            (
                cfg["input_shape"][0].as_u64().unwrap() as usize,
                cfg["input_shape"][1].as_u64().unwrap() as usize,
                cfg["input_shape"][2].as_u64().unwrap() as usize,
                cfg["input_shape"][3].as_u64().unwrap() as usize,
            ),
            (
                cfg["weight_shape"][0].as_u64().unwrap() as usize,
                cfg["weight_shape"][1].as_u64().unwrap() as usize,
                cfg["weight_shape"][2].as_u64().unwrap() as usize,
                cfg["weight_shape"][3].as_u64().unwrap() as usize,
            ),
        );

        // Verify the convolution was successful
        assert!(result.is_ok(), "conv2d failed: {:?}", result.err());
        assert_eq!(output_buf.len(), output_ref.len(), "output size mismatch");

        // Compare results with tolerance for floating point precision
        const TOLERANCE: f32 = 1e-5;
        for (i, (actual, expected)) in output_buf.iter().zip(output_ref.iter()).enumerate() {
            let diff = (actual - expected).abs();
            assert!(
                diff < TOLERANCE,
                "Output mismatch at index {}: actual={}, expected={}, diff={}",
                i,
                actual,
                expected,
                diff
            );
        }
    }
}

/// Unit tests for conv2d functionality that don't require external dependencies.
/// These tests validate basic functionality, error handling, and edge cases.
mod unit_tests {
    use bitnet_common::QuantizationType;
    use bitnet_kernels::convolution::{Conv2DParams, conv2d, conv2d_quantized};

    #[test]
    fn test_conv2d_basic_functionality() {
        // Simple 1x1x3x3 input with 1x1x2x2 kernel
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let weight = vec![1.0, 0.0, 0.0, 1.0]; // Identity-like kernel
        let mut output = vec![0.0; 4]; // 2x2 output

        let result = conv2d(
            &input,
            &weight,
            None,
            &mut output,
            Conv2DParams::default(),
            (1, 1, 3, 3), // NCHW
            (1, 1, 2, 2), // OIHW
        );

        assert!(result.is_ok(), "Conv2d should succeed");
        // Expected output: [1+5=6, 2+6=8, 4+8=12, 5+9=14]
        let expected = vec![6.0, 8.0, 12.0, 14.0];
        assert_eq!(output, expected);
    }

    #[test]
    fn test_conv2d_with_bias() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0];
        let bias = vec![10.0];
        let mut output = vec![0.0; 4];

        let result = conv2d(
            &input,
            &weight,
            Some(&bias),
            &mut output,
            Conv2DParams::default(),
            (1, 1, 2, 2),
            (1, 1, 1, 1),
        );

        assert!(result.is_ok());
        // Each output should be input value + bias
        let expected = vec![11.0, 12.0, 13.0, 14.0];
        assert_eq!(output, expected);
    }

    #[test]
    fn test_conv2d_stride() {
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let weight = vec![1.0, 0.0, 0.0, 1.0];
        // Output size calculation: (input_size - kernel_size) / stride + 1
        // For 4x4 input with 2x2 kernel and stride 2: (4-2)/2 + 1 = 2
        let mut output = vec![0.0; 4]; // Should be 2x2 output with stride 2

        let result = conv2d(
            &input,
            &weight,
            None,
            &mut output,
            Conv2DParams { stride: (2, 2), padding: (0, 0), dilation: (1, 1) },
            (1, 1, 4, 4),
            (1, 1, 2, 2),
        );

        assert!(result.is_ok(), "Conv2d failed: {:?}", result.err());
        // With stride 2, we sample positions (0,0), (0,2), (2,0), (2,2) in the 4x4 input
        // Top-left sample: positions (0,0) and (1,1) -> values 1 + 6 = 7
        // Top-right sample: positions (0,2) and (1,3) -> values 3 + 8 = 11
        // Bottom-left sample: positions (2,0) and (3,1) -> values 9 + 14 = 23
        // Bottom-right sample: positions (2,2) and (3,3) -> values 11 + 16 = 27
        let expected = vec![7.0, 11.0, 23.0, 27.0];
        assert_eq!(output, expected);
    }

    #[test]
    fn test_conv2d_padding() {
        let input = vec![5.0]; // 1x1 input
        let weight = vec![1.0, 1.0, 1.0, 1.0]; // 2x2 kernel
        let mut output = vec![0.0; 4]; // 2x2 output with padding

        let result = conv2d(
            &input,
            &weight,
            None,
            &mut output,
            Conv2DParams { stride: (1, 1), padding: (1, 1), dilation: (1, 1) },
            (1, 1, 1, 1),
            (1, 1, 2, 2),
        );

        assert!(result.is_ok());
        // Only center position has non-zero input, others are padded with 0
        // Bottom-right position: 0+0+0+5 = 5
        assert_eq!(output[3], 5.0);
    }

    #[test]
    fn test_conv2d_dimension_mismatch() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let mut output = vec![0.0; 1];

        // Input channels (2) != weight input channels (1)
        let result = conv2d(
            &input,
            &weight,
            None,
            &mut output,
            Conv2DParams::default(),
            (1, 2, 1, 2), // 2 input channels
            (1, 1, 2, 2), // 1 input channel in weight
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_conv2d_invalid_input_size() {
        let input = vec![1.0, 2.0]; // Size 2, but expecting 4
        let weight = vec![1.0];
        let mut output = vec![0.0; 4];

        let result = conv2d(
            &input,
            &weight,
            None,
            &mut output,
            Conv2DParams::default(),
            (1, 1, 2, 2), // Expects 4 elements
            (1, 1, 1, 1),
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_conv2d_invalid_bias_size() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0];
        let bias = vec![1.0, 2.0]; // Wrong size: 2, but expecting 1
        let mut output = vec![0.0; 4];

        let result = conv2d(
            &input,
            &weight,
            Some(&bias),
            &mut output,
            Conv2DParams::default(),
            (1, 1, 2, 2),
            (1, 1, 1, 1), // 1 output channel
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_conv2d_dilation() {
        // Test dilation with a 3x3 input and 2x2 kernel with dilation 2
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let weight = vec![1.0, 0.0, 0.0, 1.0]; // corners only
        let mut output = vec![0.0; 1]; // Should be 1x1 output

        let result = conv2d(
            &input,
            &weight,
            None,
            &mut output,
            Conv2DParams { stride: (1, 1), padding: (0, 0), dilation: (2, 2) },
            (1, 1, 3, 3),
            (1, 1, 2, 2),
        );

        assert!(result.is_ok());
        // With dilation 2: sample positions (0,0), (0,2), (2,0), (2,2)
        // Values: 1, 3, 7, 9 with weights 1, 0, 0, 1
        // Result: 1*1 + 3*0 + 7*0 + 9*1 = 10
        assert_eq!(output[0], 10.0);
    }

    #[test]
    fn test_conv2d_quantized_i2s() {
        // Test quantized convolution with I2S quantization
        let input = vec![1.0, 2.0, 3.0, 4.0]; // 1x1x2x2 input

        // Pack I2S weights: 4 weights per byte
        // Weights: [-2, -1, 1, 2] -> [00, 01, 10, 11] -> 0b11100100 = 0xE4
        // But packed as bits 0-1, 2-3, 4-5, 6-7: 11100100 = 0xE4 (correct)
        let weight_quantized = vec![0xE4];
        let weight_scales = vec![1.0]; // Scale factor for output channel 0
        let mut output = vec![0.0; 1]; // 1x1 output

        let result = conv2d_quantized(
            &input,
            &weight_quantized,
            &weight_scales,
            None,
            &mut output,
            Conv2DParams::default(),
            (1, 1, 2, 2), // NCHW
            (1, 1, 2, 2), // OIHW
            QuantizationType::I2S,
        );

        assert!(result.is_ok(), "Quantized conv2d should succeed: {:?}", result.err());
        // Expected: 1*(-2) + 2*(-1) + 3*1 + 4*2 = -2 - 2 + 3 + 8 = 7
        assert_eq!(output[0], 7.0);
    }

    #[test]
    fn test_conv2d_quantized_tl1() {
        // Test quantized convolution with TL1 quantization
        let input = vec![1.0, 2.0, 3.0, 4.0]; // 1x1x2x2 input

        // TL1 weights: [0, 64, 192, 255] -> [-1, -0.5, 0.5, 1] approximately
        let weight_quantized = vec![0, 64, 192, 255];
        let weight_scales = vec![1.0];
        let mut output = vec![0.0; 1]; // 1x1 output

        let result = conv2d_quantized(
            &input,
            &weight_quantized,
            &weight_scales,
            None,
            &mut output,
            Conv2DParams::default(),
            (1, 1, 2, 2),
            (1, 1, 2, 2),
            QuantizationType::TL1,
        );

        assert!(result.is_ok(), "TL1 quantized conv2d should succeed: {:?}", result.err());
        // TL1 mapping: (val - 128) / 127
        // [0, 64, 192, 255] -> [(-128)/127, (-64)/127, (64)/127, (127)/127]
        // ≈ [-1.008, -0.504, 0.504, 1.0]
        // Result: 1*(-1.008) + 2*(-0.504) + 3*(0.504) + 4*(1.0) ≈ -1.008 - 1.008 + 1.512 + 4 ≈ 3.496
        assert!((output[0] - 3.496).abs() < 0.01, "Expected ~3.496, got {}", output[0]);
    }

    #[test]
    fn test_conv2d_quantized_with_bias() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight_quantized = vec![0xE4]; // I2S: [-2, -1, 1, 2]
        let weight_scales = vec![1.0];
        let bias = vec![10.0];
        let mut output = vec![0.0; 1];

        let result = conv2d_quantized(
            &input,
            &weight_quantized,
            &weight_scales,
            Some(&bias),
            &mut output,
            Conv2DParams::default(),
            (1, 1, 2, 2),
            (1, 1, 2, 2),
            QuantizationType::I2S,
        );

        assert!(result.is_ok());
        // Expected: 7 (from computation) + 10 (bias) = 17
        assert_eq!(output[0], 17.0);
    }

    #[test]
    fn test_conv2d_quantized_scale_factor() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight_quantized = vec![0xE4]; // I2S: [-2, -1, 1, 2]
        let weight_scales = vec![2.0]; // Scale by 2
        let mut output = vec![0.0; 1];

        let result = conv2d_quantized(
            &input,
            &weight_quantized,
            &weight_scales,
            None,
            &mut output,
            Conv2DParams::default(),
            (1, 1, 2, 2),
            (1, 1, 2, 2),
            QuantizationType::I2S,
        );

        assert!(result.is_ok());
        // Expected: 7 (base result) * 2 (scale) = 14
        assert_eq!(output[0], 14.0);
    }

    #[test]
    fn test_conv2d_quantized_invalid_scale_size() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight_quantized = vec![0xE4];
        let weight_scales = vec![1.0, 2.0]; // Wrong size: 2, should be 1
        let mut output = vec![0.0; 1];

        let result = conv2d_quantized(
            &input,
            &weight_quantized,
            &weight_scales,
            None,
            &mut output,
            Conv2DParams::default(),
            (1, 1, 2, 2),
            (1, 1, 2, 2), // 1 output channel
            QuantizationType::I2S,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_conv2d_quantized_invalid_weight_size() {
        // Test with wrong weight data size for I2S (should be 1 byte for 4 elements)
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight_quantized = vec![0, 1]; // Wrong size: 2 bytes, should be 1 byte
        let weight_scales = vec![1.0];
        let mut output = vec![0.0; 1];

        let result = conv2d_quantized(
            &input,
            &weight_quantized,
            &weight_scales,
            None,
            &mut output,
            Conv2DParams::default(),
            (1, 1, 2, 2),
            (1, 1, 2, 2),
            QuantizationType::I2S, // I2S expects packed weights
        );

        assert!(result.is_err());
    }
}

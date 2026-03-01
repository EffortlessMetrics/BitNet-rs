#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug, Clone, Copy)]
enum PaddingMode {
    Zero(u8),
    Same,
}

#[derive(Arbitrary, Debug)]
struct Conv1dInput {
    input_width: u8,
    kernel_size: u8,
    stride: u8,
    padding: PaddingMode,
    dilation: u8,
    in_channels: u8,
    out_channels: u8,
    /// Raw weight data.
    weight_data: Vec<u8>,
    /// Raw input data.
    input_data: Vec<u8>,
}

/// Compute expected conv1d output width using the standard formula.
fn conv1d_output_width(
    input_width: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Option<usize> {
    if stride == 0 || dilation == 0 || kernel_size == 0 {
        return None;
    }
    let effective_kernel = dilation * (kernel_size - 1) + 1;
    let numerator = input_width + 2 * padding;
    if numerator < effective_kernel {
        return None;
    }
    Some((numerator - effective_kernel) / stride + 1)
}

/// Minimal 1D convolution for shape validation (single-channel, no bias).
fn conv1d_forward(input: &[f32], kernel: &[f32], stride: usize, padding: usize) -> Vec<f32> {
    let input_len = input.len();
    let kernel_len = kernel.len();
    let out_len = match conv1d_output_width(input_len, kernel_len, stride, padding, 1) {
        Some(n) => n,
        None => return vec![],
    };

    // Build zero-padded input.
    let padded_len = input_len + 2 * padding;
    let mut padded = vec![0.0f32; padded_len];
    padded[padding..padding + input_len].copy_from_slice(input);

    let mut output = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let start = i * stride;
        let mut acc = 0.0f32;
        for k in 0..kernel_len {
            if start + k < padded_len {
                acc += padded[start + k] * kernel[k];
            }
        }
        output.push(acc);
    }
    output
}

fuzz_target!(|input: Conv1dInput| {
    // Clamp all dimensions to small but meaningful ranges.
    let input_width = (input.input_width as usize % 64) + 1;
    let kernel_size = (input.kernel_size as usize % 16) + 1;
    let stride = (input.stride as usize % 8) + 1;
    let dilation = (input.dilation as usize % 4) + 1;
    let _in_channels = (input.in_channels as usize % 8) + 1;
    let _out_channels = (input.out_channels as usize % 8) + 1;

    let padding = match input.padding {
        PaddingMode::Zero(p) => (p as usize) % 16,
        PaddingMode::Same => {
            let effective_kernel = dilation * (kernel_size - 1) + 1;
            (effective_kernel - 1) / 2
        }
    };

    // Invariant 1: Output width formula should not panic.
    let expected_width = conv1d_output_width(input_width, kernel_size, stride, padding, dilation);

    if let Some(width) = expected_width {
        // Invariant 2: Output width must be positive for valid inputs.
        assert!(width > 0, "output width must be >0 for valid config");

        // Invariant 3: Same-padding preserves ceil(input_width / stride).
        if matches!(input.padding, PaddingMode::Same) {
            let expected_same = (input_width + stride - 1) / stride;
            assert_eq!(width, expected_same, "Same-padding: expected {expected_same}, got {width}");
        }
    }

    // Invariant 4: Actual convolution must not panic and output size must match.
    let aligned_len = (input.input_data.len() / 4) * 4;
    let inp: Vec<f32> = input.input_data[..aligned_len]
        .chunks_exact(4)
        .take(256)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let aligned_wt = (input.weight_data.len() / 4) * 4;
    let wt: Vec<f32> = input.weight_data[..aligned_wt]
        .chunks_exact(4)
        .take(256)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    if inp.len() < input_width || wt.len() < kernel_size {
        return;
    }

    let inp_slice = &inp[..input_width];
    let wt_slice = &wt[..kernel_size];

    // Only use dilation=1 for actual forward pass (minimal implementation).
    if dilation == 1 {
        let output = conv1d_forward(inp_slice, wt_slice, stride, padding);
        if let Some(expected) = conv1d_output_width(input_width, kernel_size, stride, padding, 1) {
            assert_eq!(
                output.len(),
                expected,
                "conv1d output len mismatch: expected {expected}, got {}",
                output.len()
            );
        }

        // Invariant 5: Identity kernel (size=1, weight=1.0) with stride=1, pad=0 is a no-op.
        if kernel_size == 1 && stride == 1 && padding == 0 {
            let identity_kernel = [1.0f32];
            let identity_out = conv1d_forward(inp_slice, &identity_kernel, 1, 0);
            assert_eq!(identity_out.len(), input_width);
            for (a, b) in identity_out.iter().zip(inp_slice.iter()) {
                if a.is_finite() && b.is_finite() {
                    assert!((a - b).abs() < 1e-6, "identity kernel not no-op: {a} != {b}");
                }
            }
        }
    }
});

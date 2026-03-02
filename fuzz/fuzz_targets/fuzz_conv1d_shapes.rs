#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Conv1dInput {
    in_channels: u8,
    out_channels: u8,
    kernel_size: u8,
    stride: u8,
    dilation: u8,
    padding: u8,
    input_len: u8,
    data: Vec<u8>,
}

fn conv1d_output_len(
    input_len: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Option<usize> {
    if stride == 0 || dilation == 0 || kernel_size == 0 {
        return None;
    }
    let effective_k = dilation * (kernel_size - 1) + 1;
    let numerator = input_len + 2 * padding;
    if numerator < effective_k {
        return None;
    }
    Some((numerator - effective_k) / stride + 1)
}

fn conv1d_naive(
    input: &[f32],
    weight: &[f32],
    in_channels: usize,
    out_channels: usize,
    input_len: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Option<Vec<f32>> {
    let out_len = conv1d_output_len(input_len, kernel_size, stride, padding, dilation)?;

    if input.len() < in_channels * input_len {
        return None;
    }
    if weight.len() < out_channels * in_channels * kernel_size {
        return None;
    }

    let mut output = vec![0.0f32; out_channels * out_len];

    for oc in 0..out_channels {
        for o in 0..out_len {
            let mut acc = 0.0f32;
            for ic in 0..in_channels {
                for k in 0..kernel_size {
                    let pos = o * stride + k * dilation;
                    if pos >= padding && pos < padding + input_len {
                        let inp_idx = ic * input_len + (pos - padding);
                        let w_idx = oc * in_channels * kernel_size + ic * kernel_size + k;
                        acc += input[inp_idx] * weight[w_idx];
                    }
                }
            }
            output[oc * out_len + o] = acc;
        }
    }

    Some(output)
}

fn bytes_to_f32(data: &[u8], count: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    let mut out: Vec<f32> = data[..aligned]
        .chunks_exact(4)
        .take(count)
        .map(|b| {
            let v = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            if v.is_finite() { v } else { 0.0 }
        })
        .collect();
    out.resize(count, 0.0);
    out
}

fuzz_target!(|input: Conv1dInput| {
    let in_ch = (input.in_channels as usize % 4) + 1;
    let out_ch = (input.out_channels as usize % 4) + 1;
    let ks = (input.kernel_size as usize % 5) + 1;
    let stride = (input.stride as usize % 3) + 1;
    let dilation = (input.dilation as usize % 3) + 1;
    let pad = input.padding as usize % 4;
    let in_len = (input.input_len as usize % 16) + ks * dilation;

    // Invariant 1: Output length formula must not panic for valid params
    let out_len = conv1d_output_len(in_len, ks, stride, pad, dilation);
    if out_len.is_none() {
        return;
    }
    let out_len = out_len.unwrap();

    // Invariant 2: Output length must be positive for valid configs
    assert!(out_len > 0, "output length must be >0 for valid config");

    // Invariant 3: Stride=1, pad=0, dilation=1 yields input_len - kernel_size + 1
    if stride == 1 && pad == 0 && dilation == 1 && in_len >= ks {
        let expected = in_len - ks + 1;
        assert_eq!(
            conv1d_output_len(in_len, ks, 1, 0, 1).unwrap(),
            expected,
            "basic output len formula mismatch"
        );
    }

    let inp_count = in_ch * in_len;
    let wt_count = out_ch * in_ch * ks;
    let inp = bytes_to_f32(&input.data, inp_count);
    let wt = bytes_to_f32(&input.data, wt_count);

    // Invariant 4: Naive conv1d must not panic and must produce correct output shape
    if let Some(output) = conv1d_naive(&inp, &wt, in_ch, out_ch, in_len, ks, stride, pad, dilation)
    {
        assert_eq!(
            output.len(),
            out_ch * out_len,
            "output size mismatch: expected {}, got {}",
            out_ch * out_len,
            output.len()
        );

        // Invariant 5: All output values must be finite
        for (i, &val) in output.iter().enumerate() {
            assert!(val.is_finite(), "conv1d output non-finite at index {i}: {val}");
        }
    }

    // Invariant 6: Zero-weight convolution produces all-zero output
    let zero_wt = vec![0.0f32; wt_count];
    if let Some(output) =
        conv1d_naive(&inp, &zero_wt, in_ch, out_ch, in_len, ks, stride, pad, dilation)
    {
        for (i, &val) in output.iter().enumerate() {
            assert!(val.abs() < 1e-6, "zero-weight conv1d should produce ~0, got {val} at {i}");
        }
    }

    // Invariant 7: Increasing dilation increases effective receptive field
    if dilation < 3 {
        let d2 = dilation + 1;
        let out2 = conv1d_output_len(in_len, ks, stride, pad, d2);
        if let (Some(o1), Some(o2)) = (Some(out_len), out2) {
            assert!(o2 <= o1, "larger dilation should not increase output length");
        }
    }
});

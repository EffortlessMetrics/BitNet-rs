#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::conv1d::{Conv1dConfig, PaddingMode, conv1d_forward, conv1d_output_width};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Conv1dInput {
    in_channels: u8,
    out_channels: u8,
    kernel_size: u8,
    stride: u8,
    padding: u8,
    dilation: u8,
    groups: u8,
    use_same_padding: bool,
    use_bias: bool,
    input_width: u8,
    input_data: Vec<f32>,
    weight_data: Vec<f32>,
    bias_data: Vec<f32>,
}

fuzz_target!(|input: Conv1dInput| {
    let in_c = (input.in_channels as usize).clamp(1, 16);
    let out_c = (input.out_channels as usize).clamp(1, 16);
    let ks = (input.kernel_size as usize).clamp(1, 8);
    let stride = (input.stride as usize).clamp(1, 4);
    let dilation = (input.dilation as usize).clamp(1, 4);

    // Groups must divide both in_channels and out_channels
    let groups = {
        let g = (input.groups as usize).clamp(1, in_c);
        let mut best = 1;
        for candidate in 1..=g {
            if in_c % candidate == 0 && out_c % candidate == 0 {
                best = candidate;
            }
        }
        best
    };

    let padding = if input.use_same_padding {
        PaddingMode::Same
    } else {
        PaddingMode::Zero((input.padding as usize) % 8)
    };

    let input_width = (input.input_width as usize).clamp(1, 32);
    let ic_per_group = in_c / groups;

    let config = Conv1dConfig {
        in_channels: in_c,
        out_channels: out_c,
        kernel_size: ks,
        stride,
        padding,
        dilation,
        groups,
        bias: input.use_bias,
    };

    // Check output width computation doesn't panic
    let _ = conv1d_output_width(&config, input_width);

    let in_size = in_c * input_width;
    let w_size = out_c * ic_per_group * ks;

    let inp: Vec<f32> = input
        .input_data
        .iter()
        .copied()
        .take(in_size)
        .chain(std::iter::repeat_n(0.0f32, in_size.saturating_sub(input.input_data.len())))
        .take(in_size)
        .collect();

    let wt: Vec<f32> = input
        .weight_data
        .iter()
        .copied()
        .take(w_size)
        .chain(std::iter::repeat_n(0.0f32, w_size.saturating_sub(input.weight_data.len())))
        .take(w_size)
        .collect();

    let bias_vec: Vec<f32> = if input.use_bias {
        input
            .bias_data
            .iter()
            .copied()
            .take(out_c)
            .chain(std::iter::repeat_n(0.0f32, out_c.saturating_sub(input.bias_data.len())))
            .take(out_c)
            .collect()
    } else {
        Vec::new()
    };

    let bias_slice = if input.use_bias { Some(bias_vec.as_slice()) } else { None };

    match conv1d_forward(&inp, &wt, bias_slice, &config) {
        Ok(out) => {
            for v in &out {
                assert!(v.is_finite(), "conv1d produced non-finite: {v}");
            }
        }
        Err(_) => {}
    }
});

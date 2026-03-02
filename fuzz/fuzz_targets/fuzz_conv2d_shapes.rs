#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::{Conv2dConfig, conv2d, depthwise_conv2d, im2col};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Conv2dInput {
    in_channels: u8,
    out_channels: u8,
    kernel_h: u8,
    kernel_w: u8,
    stride_h: u8,
    stride_w: u8,
    padding_h: u8,
    padding_w: u8,
    in_h: u8,
    in_w: u8,
    batch_size: u8,
    mode: u8,
    data: Vec<u8>,
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

fuzz_target!(|input: Conv2dInput| {
    let ic = (input.in_channels as usize % 4) + 1;
    let oc = (input.out_channels as usize % 4) + 1;
    let kh = (input.kernel_h as usize % 3) + 1;
    let kw = (input.kernel_w as usize % 3) + 1;
    let sh = (input.stride_h as usize % 2) + 1;
    let sw = (input.stride_w as usize % 2) + 1;
    let ph = input.padding_h as usize % 2;
    let pw = input.padding_w as usize % 2;
    let in_h = (input.in_h as usize % 8) + kh;
    let in_w = (input.in_w as usize % 8) + kw;
    let batch = (input.batch_size as usize % 2) + 1;

    match input.mode % 3 {
        0 => {
            // Standard conv2d
            let config = Conv2dConfig {
                in_channels: ic,
                out_channels: oc,
                kernel_h: kh,
                kernel_w: kw,
                stride_h: sh,
                stride_w: sw,
                padding_h: ph,
                padding_w: pw,
                dilation_h: 1,
                dilation_w: 1,
                groups: 1,
            };
            let input_size = batch * ic * in_h * in_w;
            let weight_size = oc * ic * kh * kw;
            let inp = bytes_to_f32(&input.data, input_size);
            let wt = bytes_to_f32(&input.data, weight_size);
            let bias: Vec<f32> = vec![0.0; oc];
            let _ = conv2d(&inp, &wt, Some(&bias), &config, batch, in_h, in_w);
        }
        1 => {
            // Depthwise conv2d: groups == in_channels == out_channels
            let ch = ic;
            let config = Conv2dConfig {
                in_channels: ch,
                out_channels: ch,
                kernel_h: kh,
                kernel_w: kw,
                stride_h: sh,
                stride_w: sw,
                padding_h: ph,
                padding_w: pw,
                dilation_h: 1,
                dilation_w: 1,
                groups: ch,
            };
            let input_size = batch * ch * in_h * in_w;
            let weight_size = ch * kh * kw;
            let inp = bytes_to_f32(&input.data, input_size);
            let wt = bytes_to_f32(&input.data, weight_size);
            let _ = depthwise_conv2d(&inp, &wt, None, &config, batch, in_h, in_w);
        }
        2 => {
            // im2col transform
            let config = Conv2dConfig {
                in_channels: ic,
                out_channels: oc,
                kernel_h: kh,
                kernel_w: kw,
                stride_h: sh,
                stride_w: sw,
                padding_h: ph,
                padding_w: pw,
                dilation_h: 1,
                dilation_w: 1,
                groups: 1,
            };
            let input_size = ic * in_h * in_w;
            let inp = bytes_to_f32(&input.data, input_size);
            let _ = im2col(&inp, &config, in_h, in_w, 0);
        }
        _ => unreachable!(),
    }
});

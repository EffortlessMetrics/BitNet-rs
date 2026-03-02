#![no_main]

//! Fuzz quantization block decoding edge cases: exercises `I2SQuantizer`
//! dequantization with arbitrary block data, scales, shapes, and block sizes
//! to verify no panics or memory safety issues on malformed inputs.

use arbitrary::Arbitrary;
use bitnet_common::QuantizationType;
use bitnet_quantization::QuantizedTensor;
use bitnet_quantization::i2s::I2SQuantizer;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct BlockDecodeInput {
    raw_data: Vec<u8>,
    scales: Vec<u8>,
    shape_dims: Vec<u16>,
    block_size_selector: u8,
    has_zero_points: bool,
    zero_points: Vec<u8>,
}

fn bytes_to_f32_vec(data: &[u8]) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned].chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

fn bytes_to_i32_vec(data: &[u8]) -> Vec<i32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned].chunks_exact(4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

fuzz_target!(|input: BlockDecodeInput| {
    if input.raw_data.is_empty() || input.scales.is_empty() || input.shape_dims.is_empty() {
        return;
    }

    let scales = bytes_to_f32_vec(&input.scales);
    if scales.is_empty() {
        return;
    }

    let shape: Vec<usize> =
        input.shape_dims.iter().take(4).map(|&d| (d as usize).max(1).min(256)).collect();

    let block_size = match input.block_size_selector % 4 {
        0 => 32,
        1 => 64,
        2 => 128,
        _ => 256,
    };

    let zero_points = if input.has_zero_points {
        let zp = bytes_to_i32_vec(&input.zero_points);
        if zp.is_empty() { None } else { Some(zp) }
    } else {
        None
    };

    let qt = QuantizedTensor::new_with_params(
        input.raw_data.clone(),
        scales,
        zero_points,
        shape,
        QuantizationType::I2S,
        block_size,
    );

    // Attempt dequantization — may fail on inconsistent sizes, must not panic.
    let quantizer = I2SQuantizer::with_block_size(block_size);
    let _ = quantizer.dequantize_tensor(&qt);

    // Also try the default block-size quantizer.
    let default_quantizer = I2SQuantizer::new();
    let _ = default_quantizer.dequantize_tensor(&qt);
});

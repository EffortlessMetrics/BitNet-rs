#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cuda::{
    QuantMethod, QuantizeConfig, calibrate_scales, dequantize_i2s_cpu, dequantize_ternary_cpu,
    quantize_i2s_cpu, quantize_ternary_cpu,
};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct CudaQuantizeInput {
    block_size: u8,
    method_byte: u8,
    data: Vec<u8>,
    use_i2s: bool,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned]
        .chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fuzz_target!(|input: CudaQuantizeInput| {
    let block_size = (input.block_size as usize % 64) + 1;

    let floats: Vec<f32> = bytes_to_f32(&input.data, 512)
        .into_iter()
        .map(|v| if v.is_finite() { v } else { 0.0 })
        .collect();

    if floats.is_empty() {
        return;
    }

    let method = match input.method_byte % 4 {
        0 => QuantMethod::AbsMax,
        1 => QuantMethod::MinMax,
        2 => QuantMethod::Symmetric,
        _ => QuantMethod::Percentile(input.method_byte),
    };

    let config = QuantizeConfig { block_size, method };

    // Test calibrate_scales — must not panic.
    if let Ok(scales) = calibrate_scales(&floats, &config) {
        let expected_blocks = floats.len().div_ceil(block_size);
        assert_eq!(scales.len(), expected_blocks, "scales length mismatch");
        for (i, &s) in scales.iter().enumerate() {
            assert!(s.is_finite(), "scale non-finite at block {i}: {s}");
            assert!(s >= 0.0, "scale negative at block {i}: {s}");
        }
    }

    if input.use_i2s {
        // I2_S quantize → dequantize round-trip.
        if let Ok((packed, scales)) = quantize_i2s_cpu(&floats, block_size) {
            let expected_bytes = floats.len().div_ceil(4);
            assert_eq!(packed.len(), expected_bytes, "packed length mismatch");

            if let Ok(deq) = dequantize_i2s_cpu(&packed, &scales, block_size, floats.len()) {
                assert_eq!(deq.len(), floats.len(), "dequantized length mismatch");

                // Invariant: all dequantized values are finite.
                for (i, &val) in deq.iter().enumerate() {
                    assert!(val.is_finite(), "dequantized value non-finite at {i}: {val}");
                }

                // Invariant: dequantized values are ternary * scale.
                for (i, &val) in deq.iter().enumerate() {
                    let blk = i / block_size;
                    let s = scales[blk];
                    let valid =
                        (val - s).abs() < 1e-6 || (val + s).abs() < 1e-6 || val.abs() < 1e-6;
                    assert!(
                        valid,
                        "dequantized value {val} at index {i} is not ternary*scale ({s})"
                    );
                }
            }
        }
    } else {
        // Ternary quantize → dequantize round-trip.
        if let Ok((quantized, scale)) = quantize_ternary_cpu(&floats, &config) {
            assert_eq!(quantized.len(), floats.len(), "quantized length mismatch");

            // All values must be ternary.
            for (i, &v) in quantized.iter().enumerate() {
                assert!(v == -1 || v == 0 || v == 1, "non-ternary value {v} at index {i}");
            }

            assert!(scale.is_finite(), "global scale non-finite: {scale}");
            assert!(scale >= 0.0, "global scale negative: {scale}");

            let deq = dequantize_ternary_cpu(&quantized, scale);
            assert_eq!(deq.len(), floats.len(), "dequantized length mismatch");

            for (i, &val) in deq.iter().enumerate() {
                assert!(val.is_finite(), "ternary dequantized non-finite at {i}: {val}");
            }
        }
    }
});

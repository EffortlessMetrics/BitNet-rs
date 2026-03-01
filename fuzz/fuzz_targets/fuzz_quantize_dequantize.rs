#![no_main]

use arbitrary::Arbitrary;
use bitnet_common::{BitNetTensor, Device, QuantizationType};
use bitnet_quantization::{Quantize, convert_quantization, validate_round_trip};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct QuantRoundtripInput {
    data: Vec<u8>,
    dim: u8,
    qtype_selector: u8,
    tolerance: u8,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned]
        .chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn select_qtype(sel: u8) -> QuantizationType {
    match sel % 3 {
        0 => QuantizationType::I2S,
        1 => QuantizationType::TL1,
        2 => QuantizationType::TL2,
        _ => unreachable!(),
    }
}

fuzz_target!(|input: QuantRoundtripInput| {
    let dim = (input.dim as usize % 64) + 4;
    // Ensure dim is a multiple of 4 for I2S packing.
    let dim = (dim / 4) * 4;
    if dim == 0 {
        return;
    }

    let mut values = bytes_to_f32(&input.data, dim);
    if values.len() < dim {
        return;
    }
    values.truncate(dim);

    // Clamp to finite, reasonable range.
    for v in &mut values {
        if !v.is_finite() {
            *v = 0.0;
        } else if v.abs() > 1000.0 {
            *v = v.signum() * 1000.0;
        }
    }

    let qtype = select_qtype(input.qtype_selector);

    // Create tensor and quantize.
    let tensor = match BitNetTensor::from_slice(&values, &[dim], &Device::Cpu) {
        Ok(t) => t,
        Err(_) => return,
    };

    let quantized = match tensor.quantize(qtype) {
        Ok(q) => q,
        Err(_) => return,
    };

    // Invariant 1: quantized data is non-empty.
    assert!(!quantized.data.is_empty(), "quantized data is empty");
    assert!(!quantized.scales.is_empty(), "quantized scales are empty");
    assert_eq!(quantized.qtype, qtype, "qtype mismatch");

    // Invariant 2: dequantize does not panic and produces correct length.
    if let Ok(deq) = quantized.dequantize() {
        let deq_data = match deq.to_vec() {
            Ok(d) => d,
            Err(_) => return,
        };
        assert_eq!(
            deq_data.len(),
            dim,
            "dequantized length mismatch: expected {dim}, got {}",
            deq_data.len()
        );

        // Invariant 3: dequantized values are finite.
        for (i, &val) in deq_data.iter().enumerate() {
            assert!(val.is_finite(), "dequantized non-finite at {i}: {val}");
        }
    }

    // Invariant 4: validate_round_trip does not panic.
    let tolerance = (input.tolerance as f32 / 255.0) * 10.0 + 0.01;
    let _ = validate_round_trip(&tensor, qtype, tolerance);

    // Invariant 5: convert_quantization to same type is identity-ish.
    if let Ok(converted) = convert_quantization(&quantized, qtype) {
        assert_eq!(converted.qtype, qtype, "convert to same qtype changed qtype");
        assert!(!converted.data.is_empty(), "converted data is empty");
    }

    // Invariant 6: convert between different qtypes does not panic.
    for target_sel in 0..3u8 {
        let target = select_qtype(target_sel);
        let _ = convert_quantization(&quantized, target);
    }
});

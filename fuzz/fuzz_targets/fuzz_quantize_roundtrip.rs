#![no_main]

use arbitrary::Arbitrary;
use bitnet_common::{BitNetTensor, Device, QuantizationType};
use bitnet_quantization::Quantize;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct RoundtripInput {
    data: Vec<u8>,
    dim: u8,
    qtype_selector: u8,
}

fn select_qtype(sel: u8) -> QuantizationType {
    match sel % 3 {
        0 => QuantizationType::I2S,
        1 => QuantizationType::TL1,
        2 => QuantizationType::TL2,
        _ => unreachable!(),
    }
}

fuzz_target!(|input: RoundtripInput| {
    // Need at least 16 bytes for 4 f32 values.
    if input.data.len() < 16 {
        return;
    }

    let dim = ((input.dim as usize % 60) + 4) & !3; // multiple of 4, >= 4
    let float_count = input.data.len() / 4;
    if float_count < dim {
        return;
    }

    let mut values: Vec<f32> = input.data[..dim * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .map(|v| if v.is_finite() { v.clamp(-500.0, 500.0) } else { 0.0 })
        .collect();
    values.truncate(dim);

    let qtype = select_qtype(input.qtype_selector);

    let tensor = match BitNetTensor::from_slice(&values, &[dim], &Device::Cpu) {
        Ok(t) => t,
        Err(_) => return,
    };

    let quantized = match tensor.quantize(qtype) {
        Ok(q) => q,
        Err(_) => return,
    };

    // Invariant: quantized metadata is consistent.
    assert_eq!(quantized.qtype, qtype);
    assert!(!quantized.data.is_empty());
    assert!(!quantized.scales.is_empty());

    // Dequantize and check round-trip tolerance.
    let deq = match quantized.dequantize() {
        Ok(d) => d,
        Err(_) => return,
    };

    let deq_data = match deq.to_vec() {
        Ok(d) => d,
        Err(_) => return,
    };

    assert_eq!(deq_data.len(), dim, "dequantized length mismatch");

    for (i, (&orig, &recon)) in values.iter().zip(deq_data.iter()).enumerate() {
        assert!(recon.is_finite(), "non-finite dequantized value at {i}: {recon}");
        // Ternary quantization maps to {-1, 0, 1} × scale, so large absolute
        // error is expected. We check a generous tolerance that catches
        // catastrophic failures (e.g. NaN or wildly wrong reconstruction).
        let tol = orig.abs() + 10.0;
        assert!(
            (orig - recon).abs() <= tol,
            "roundtrip tolerance exceeded at {i}: orig={orig}, recon={recon}, tol={tol}",
        );
    }
});

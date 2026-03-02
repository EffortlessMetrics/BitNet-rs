#![no_main]

use arbitrary::Arbitrary;
use bitnet_common::{BitNetTensor, Device, QuantizationType, Tensor};
use bitnet_quantization::Quantize;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct QuantRoundTripInput {
    qtype_selector: u8,
    data: Vec<u8>,
    extra_len: u8,
}

fn bytes_to_f32_finite(data: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned]
        .chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .map(|v| if v.is_finite() { v } else { 0.0 })
        .collect()
}

fuzz_target!(|input: QuantRoundTripInput| {
    // Select quantization type from fuzzer byte.
    let qtype = match input.qtype_selector % 3 {
        0 => QuantizationType::I2S,
        1 => QuantizationType::TL1,
        _ => QuantizationType::TL2,
    };

    // Build a float vector with length in [4, 512].
    let target_len = ((input.extra_len as usize % 128) + 1) * 4;
    let floats = bytes_to_f32_finite(&input.data, target_len);

    if floats.len() < 4 || floats.len() > 64 * 1024 {
        return;
    }

    let shape = [floats.len()];

    let tensor = match BitNetTensor::from_slice(&floats, &shape, &Device::Cpu) {
        Ok(t) => t,
        Err(_) => return,
    };

    // Quantize — must not panic on any finite input.
    let quantized = match tensor.quantize(qtype) {
        Ok(q) => q,
        Err(_) => return,
    };

    // Invariant 1: Quantized type matches request.
    assert_eq!(
        quantized.qtype, qtype,
        "quantized type mismatch: expected {qtype:?}, got {:?}",
        quantized.qtype
    );

    // Dequantize — must not panic.
    let deq = match quantized.dequantize() {
        Ok(d) => d,
        Err(_) => return,
    };

    // Invariant 2: All dequantized values are finite.
    if let Ok(slice) = deq.as_slice::<f32>() {
        for (i, &v) in slice.iter().enumerate() {
            assert!(
                v.is_finite(),
                "dequantized value non-finite at index {i}: {v} (qtype={qtype:?})"
            );
        }

        // Invariant 3: Output length matches input length.
        assert_eq!(
            slice.len(),
            floats.len(),
            "round-trip length mismatch: input {} vs output {} (qtype={qtype:?})",
            floats.len(),
            slice.len()
        );
    }

    // Invariant 4: Double round-trip produces identical quantized output.
    if let Ok(re_tensor) = BitNetTensor::from_slice(
        &{
            match deq.as_slice::<f32>() {
                Ok(s) => s.to_vec(),
                Err(_) => return,
            }
        },
        &shape,
        &Device::Cpu,
    ) {
        if let Ok(re_quantized) = re_tensor.quantize(qtype) {
            if let Ok(re_deq) = re_quantized.dequantize() {
                if let (Ok(first), Ok(second)) = (deq.as_slice::<f32>(), re_deq.as_slice::<f32>()) {
                    assert_eq!(first.len(), second.len(), "double round-trip length mismatch");
                    for (i, (&a, &b)) in first.iter().zip(second.iter()).enumerate() {
                        assert_eq!(
                            a, b,
                            "double round-trip diverged at index {i}: {a} vs {b} (qtype={qtype:?})"
                        );
                    }
                }
            }
        }
    }
});

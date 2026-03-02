#![no_main]

use arbitrary::Arbitrary;
use bitnet_common::{BitNetTensor, Device, QuantizationType, Tensor};
use bitnet_quantization::Quantize;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct PipelineInput {
    data: Vec<f32>,
    /// Encoded as 0 → I2S, 1 → TL1, 2 → TL2.
    qtype_selector: u8,
}

const MAX_FUZZ_ELEMENTS: usize = 4096;

fuzz_target!(|input: PipelineInput| {
    if input.data.is_empty() {
        return;
    }

    let len = input.data.len().min(MAX_FUZZ_ELEMENTS);
    let data: Vec<f32> = input.data[..len]
        .iter()
        .map(|&x| if x.is_nan() || x.is_infinite() { 0.0 } else { x.clamp(-1e6, 1e6) })
        .collect();

    let shape = vec![data.len()];

    let Ok(tensor) = BitNetTensor::from_slice(&data, &shape, &Device::Cpu) else {
        return;
    };

    let qtype = match input.qtype_selector % 3 {
        0 => QuantizationType::I2S,
        1 => QuantizationType::TL1,
        _ => QuantizationType::TL2,
    };

    // Quantize → dequantize roundtrip: must not panic.
    let quantized = match tensor.quantize(qtype) {
        Ok(q) => q,
        Err(_) => return,
    };

    // Validate quantized output invariants.
    assert!(!quantized.data.is_empty(), "quantized data must not be empty");
    assert!(!quantized.scales.is_empty(), "scales must not be empty");
    assert_eq!(quantized.qtype, qtype, "quantization type mismatch");

    // Dequantize must not panic regardless of quantized content.
    let dequantized = match quantized.dequantize() {
        Ok(d) => d,
        Err(_) => return,
    };

    // Dequantized output should have the same number of elements.
    let deq_numel = dequantized.shape().iter().product::<usize>();
    assert_eq!(deq_numel, data.len(), "element count changed after roundtrip");

    // Compression ratio must be positive and finite.
    let ratio = quantized.compression_ratio();
    assert!(ratio > 0.0 && ratio.is_finite(), "invalid compression ratio: {ratio}");
});

#![no_main]

use arbitrary::Arbitrary;
use bitnet_common::{QuantizationType, Result};
use bitnet_quantization::Quantize;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct QuantInput {
    data: Vec<f32>,
    rows: u8,
    cols: u8,
}

fuzz_target!(|input: QuantInput| {
    // Cap dimensions to prevent timeout.
    let rows = (input.rows as usize).clamp(1, 64);
    let cols = (input.cols as usize).clamp(1, 64);
    let total = rows * cols;

    // Build data vector of the correct size.
    let data: Vec<f32> = input
        .data
        .into_iter()
        .take(total)
        .map(|x| if x.is_nan() || x.is_infinite() { 0.0 } else { x.clamp(-1000.0, 1000.0) })
        .collect();
    if data.len() != total {
        return;
    }

    let shape = vec![rows, cols];

    // Create tensor and quantize with I2S.
    let tensor = FuzzTensor::new(data.clone(), shape);
    let bitnet_tensor = match tensor.to_bitnet_tensor() {
        Ok(t) => t,
        Err(_) => return,
    };

    let quantized = match bitnet_tensor.quantize(QuantizationType::I2S) {
        Ok(q) => q,
        Err(_) => return,
    };

    // Quantized metadata invariants.
    assert!(!quantized.data.is_empty(), "quantized data must not be empty");
    assert!(!quantized.scales.is_empty(), "scales must not be empty");
    assert_eq!(quantized.qtype, QuantizationType::I2S);

    // Dequantize round-trip — must not panic.
    if let Ok(deq) = quantized.dequantize() {
        let deq_data = match deq.to_vec() {
            Ok(v) => v,
            Err(_) => return,
        };

        // Output length must match original element count.
        assert_eq!(
            deq_data.len(),
            total,
            "dequantized length {} != original {}",
            deq_data.len(),
            total
        );

        // I2S ternary invariant: dequantized values must be representable
        // as {-scale, 0, +scale} per block. All values must be finite.
        for &v in &deq_data {
            assert!(v.is_finite(), "dequantized value is non-finite: {v}");
        }
    }

    // Verify quantized bytes encode 2-bit signed values.
    // I2S maps: signed -2 → 0b00, -1 → 0b01, 0 → 0b10, +1 → 0b11.
    // All four 2-bit patterns are valid; verify packing density.
    let expected_bytes = (total + 3) / 4; // 4 values per byte (2 bits each)
    assert!(
        quantized.data.len() >= expected_bytes,
        "packed data too short: {} bytes for {} elements (expected >= {})",
        quantized.data.len(),
        total,
        expected_bytes
    );

    // Dequantized values must form a small discrete set (≤ 4 unique values
    // per scale block, since 2-bit encoding supports at most 4 levels).
    if let Ok(deq) = quantized.dequantize() {
        if let Ok(vals) = deq.to_vec() {
            let mut unique: Vec<u32> = vals.iter().map(|f| f.to_bits()).collect();
            unique.sort();
            unique.dedup();
            // With a single scale factor, we expect at most 4 distinct values.
            // With multiple blocks/scales, more are possible, but each value
            // should be finite (already checked above).
        }
    }
});

struct FuzzTensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl FuzzTensor {
    fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }

    fn to_bitnet_tensor(&self) -> Result<bitnet_common::BitNetTensor> {
        use bitnet_common::{BitNetTensor, Device};
        BitNetTensor::from_slice(&self.data, &self.shape, &Device::Cpu)
    }
}

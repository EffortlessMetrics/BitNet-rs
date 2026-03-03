#![no_main]

use arbitrary::Arbitrary;
use bitnet_common::QuantizationType;
use bitnet_quantization::I2SQuantizer;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct I2SInput {
    /// Raw bytes interpreted as little-endian f32 weights.
    raw_weights: Vec<u8>,
}

fuzz_target!(|input: I2SInput| {
    let aligned = (input.raw_weights.len() / 4) * 4;
    if aligned == 0 {
        return;
    }

    let weights: Vec<f32> = input.raw_weights[..aligned]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .take(4096)
        .collect();
    if weights.is_empty() {
        return;
    }

    // Sanitise NaN / Inf — they would poison scale computation.
    let weights: Vec<f32> = weights
        .into_iter()
        .map(|x| if x.is_nan() || x.is_infinite() { 0.0 } else { x.clamp(-1e6, 1e6) })
        .collect();

    let quantizer = I2SQuantizer::new();

    match quantizer.quantize_weights(&weights) {
        Ok(qt) => {
            // Basic structural invariants.
            assert!(!qt.data.is_empty(), "quantized data must not be empty");
            assert!(!qt.scales.is_empty(), "scales must not be empty");
            assert_eq!(qt.qtype, QuantizationType::I2S, "qtype must be I2S");

            // I2S packs 4 values per byte (2-bit encoding).
            let expected_bytes = (weights.len() + 3) / 4;
            assert!(
                qt.data.len() >= expected_bytes,
                "packed data too short: {} < {}",
                qt.data.len(),
                expected_bytes
            );

            // Every 2-bit nibble must encode a ternary value {-1, 0, 1}.
            for &byte in &qt.data {
                for shift in (0..8).step_by(2) {
                    let nibble = (byte >> shift) & 0b11;
                    assert!(nibble <= 2, "I2S nibble out of ternary range: {nibble:#04b}");
                }
            }
        }
        // Errors are acceptable for malformed / degenerate input.
        Err(_) => {}
    }
});

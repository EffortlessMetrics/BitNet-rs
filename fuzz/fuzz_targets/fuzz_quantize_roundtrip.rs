#![no_main]

use arbitrary::Arbitrary;
use bitnet_quantization::device_aware_quantizer::{
    AccuracyValidator, CPUQuantizer, ToleranceConfig,
};
use libfuzzer_sys::fuzz_target;

/// Fuzz quantize→dequantize round-trip accuracy using the
/// CPUQuantizer and AccuracyValidator, exercising I2S, TL1, TL2
/// with random data and tolerance configurations.
#[derive(Arbitrary, Debug)]
struct RoundtripInput {
    data: Vec<u8>,
    qtype: u8,
    block_size_sel: u8,
    strict: bool,
    i2s_tol: u8,
    tl_tol: u8,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned]
        .chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fuzz_target!(|input: RoundtripInput| {
    let block_size = match input.block_size_sel % 4 {
        0 => 4,
        1 => 8,
        2 => 16,
        _ => 32,
    };

    let mut values = bytes_to_f32(&input.data, 128);
    if values.len() < block_size {
        if values.is_empty() {
            return;
        }
        values.resize(block_size, 0.0);
    }
    let n = (values.len() / block_size) * block_size;
    if n == 0 {
        return;
    }
    values.truncate(n);

    for v in &mut values {
        if !v.is_finite() {
            *v = 0.0;
        } else if v.abs() > 1000.0 {
            *v = v.signum() * 1000.0;
        }
    }

    let tol_config = ToleranceConfig {
        i2s_tolerance: (input.i2s_tol as f64 / 255.0) * 0.1 + 1e-4,
        tl_tolerance: (input.tl_tol as f64 / 255.0) * 0.1 + 1e-3,
        strict_validation: input.strict,
        ..ToleranceConfig::default()
    };

    let quantizer = CPUQuantizer::new(tol_config.clone());
    let validator = AccuracyValidator::new(tol_config);

    match input.qtype % 3 {
        0 => {
            // I2S round-trip via CPUQuantizer
            if let Ok(quantized) = quantizer.quantize_i2s(&values) {
                if let Ok(dequantized) = quantizer.dequantize_i2s(&quantized) {
                    assert_eq!(dequantized.len(), n, "I2S dequant length mismatch");
                    for (i, &v) in dequantized.iter().enumerate() {
                        assert!(v.is_finite(), "I2S dequant non-finite at {i}: {v}");
                    }

                    if let Ok(report) = validator.validate_i2s_accuracy(&values, &quantized) {
                        assert!(
                            report.max_absolute_error.is_finite(),
                            "I2S accuracy: max_err non-finite"
                        );
                    }
                }

                // Double roundtrip: idempotency check
                if let Ok(deq1) = quantizer.dequantize_i2s(&quantized) {
                    if let Ok(q2) = quantizer.quantize_i2s(&deq1) {
                        if let Ok(deq2) = quantizer.dequantize_i2s(&q2) {
                            assert_eq!(deq1.len(), deq2.len());
                            for (i, (&a, &b)) in deq1.iter().zip(deq2.iter()).enumerate() {
                                assert!(
                                    (a - b).abs() < 1e-5,
                                    "I2S double roundtrip diverged at {i}: {a} vs {b}"
                                );
                            }
                        }
                    }
                }
            }
        }
        1 => {
            // TL1 round-trip
            if let Ok(quantized) = quantizer.quantize_tl1(&values) {
                if let Ok(dequantized) = quantizer.dequantize_tl1(&quantized) {
                    assert_eq!(dequantized.len(), n, "TL1 dequant length mismatch");
                    for (i, &v) in dequantized.iter().enumerate() {
                        assert!(v.is_finite(), "TL1 dequant non-finite at {i}: {v}");
                    }
                    if let Ok(report) = validator.validate_tl_accuracy(&values, &quantized) {
                        assert!(report.max_absolute_error.is_finite());
                    }
                }
            }
        }
        _ => {
            // TL2 round-trip
            if let Ok(quantized) = quantizer.quantize_tl2(&values) {
                if let Ok(dequantized) = quantizer.dequantize_tl2(&quantized) {
                    assert_eq!(dequantized.len(), n, "TL2 dequant length mismatch");
                    for (i, &v) in dequantized.iter().enumerate() {
                        assert!(v.is_finite(), "TL2 dequant non-finite at {i}: {v}");
                    }
                    if let Ok(report) = validator.validate_tl_accuracy(&values, &quantized) {
                        assert!(report.max_absolute_error.is_finite());
                    }
                }
            }
        }
    }
});

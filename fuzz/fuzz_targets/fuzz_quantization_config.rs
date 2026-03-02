#![no_main]

use arbitrary::Arbitrary;
use bitnet_quantization::int8_quant::{CalibrationMethod, Int8QuantConfig, quantize_tensor_int8};
use bitnet_quantization::pipeline::{PipelineConfig, Precision};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct QuantConfigInput {
    source: u8,
    target: u8,
    calibration_samples: u16,
    error_threshold_bytes: [u8; 8],
    per_channel: bool,
    symmetric: bool,
    calibration_method: u8,
    percentile_byte: u8,
    tensor_data: Vec<u8>,
}

fn pick_precision(v: u8) -> Precision {
    match v % 4 {
        0 => Precision::F32,
        1 => Precision::I2S,
        2 => Precision::TL1,
        _ => Precision::TL2,
    }
}

fuzz_target!(|input: QuantConfigInput| {
    // Fuzz PipelineConfig with arbitrary precision combinations.
    let config = PipelineConfig {
        source_precision: pick_precision(input.source),
        target_precision: pick_precision(input.target),
        calibration_samples: input.calibration_samples as usize,
        error_threshold: f64::from_le_bytes(input.error_threshold_bytes),
    };
    let _ = config.validate();

    // Exhaustive precision pair validation.
    for s in 0..4u8 {
        for t in 0..4u8 {
            let c = PipelineConfig {
                source_precision: pick_precision(s),
                target_precision: pick_precision(t),
                calibration_samples: 1,
                error_threshold: 0.01,
            };
            let _ = c.validate();
        }
    }

    // Fuzz Int8QuantConfig with arbitrary calibration methods.
    let cal_method = match input.calibration_method % 3 {
        0 => CalibrationMethod::MinMax,
        1 => CalibrationMethod::Percentile(input.percentile_byte as f32 / 255.0),
        _ => CalibrationMethod::MSE,
    };
    let quant_config = Int8QuantConfig {
        per_channel: input.per_channel,
        symmetric: input.symmetric,
        calibration_method: cal_method,
    };

    // Decode bytes → f32 for tensor quantization.
    let aligned = (input.tensor_data.len() / 4) * 4;
    if aligned > 0 {
        let data: Vec<f32> = input.tensor_data[..aligned]
            .chunks_exact(4)
            .take(1024)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .filter(|v| v.is_finite())
            .collect();
        if !data.is_empty() {
            let _ = quantize_tensor_int8(&data, &quant_config);
        }
    }

    // Empty and single-element tensors must not panic.
    let _ = quantize_tensor_int8(&[], &quant_config);
    let _ = quantize_tensor_int8(&[0.0], &quant_config);
    let _ = quantize_tensor_int8(&[f32::MAX, f32::MIN], &quant_config);
});

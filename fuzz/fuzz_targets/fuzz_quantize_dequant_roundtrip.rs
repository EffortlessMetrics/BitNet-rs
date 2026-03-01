#![no_main]

use bitnet_common::{BitNetTensor, Device, QuantizationType, Tensor};
use bitnet_quantization::Quantize;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Need at least 4 bytes for one f32.
    if data.len() < 4 || data.len() > 64 * 1024 {
        return;
    }

    // Interpret raw bytes as a f32 slice (truncate trailing bytes).
    let float_count = data.len() / 4;
    if float_count == 0 {
        return;
    }
    let floats: Vec<f32> = data[..float_count * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .map(|v| if v.is_finite() { v } else { 0.0 })
        .collect();

    let shape = [floats.len()];

    // Construct a BitNetTensor from the f32 slice.
    let tensor = match BitNetTensor::from_slice(&floats, &shape, &Device::Cpu) {
        Ok(t) => t,
        Err(_) => return,
    };

    // Quantize with I2_S, then dequantize — must not panic.
    if let Ok(quantized) = tensor.quantize(QuantizationType::I2S) {
        assert_eq!(quantized.qtype, QuantizationType::I2S);
        if let Ok(deq) = quantized.dequantize() {
            // Every output value must be finite.
            if let Ok(slice) = deq.as_slice::<f32>() {
                for &v in slice {
                    assert!(v.is_finite(), "dequantized value is not finite: {v}");
                }
            }
        }
    }
});

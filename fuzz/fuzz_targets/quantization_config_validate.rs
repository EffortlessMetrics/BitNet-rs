#![no_main]

use arbitrary::Arbitrary;
use bitnet_common::{BitNetConfig, ModelConfig, ModelFormat, QuantizationConfig, QuantizationType};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct QuantConfigInput {
    block_size: u32,
    precision: f32,
    quant_type_idx: u8,
    vocab_size: u32,
    hidden_size: u32,
    num_heads: u16,
    num_layers: u16,
    scale_factors: Vec<f32>,
}

fuzz_target!(|input: QuantConfigInput| {
    let quant_types = [QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2];
    let qt = quant_types[input.quant_type_idx as usize % quant_types.len()];

    // QuantizationConfig construction and serde must never panic.
    let qc = QuantizationConfig {
        quantization_type: qt,
        block_size: input.block_size as usize,
        precision: input.precision,
    };
    let _ = format!("{qc:?}");

    // Serde round-trip on QuantizationConfig must not panic.
    if let Ok(json) = serde_json::to_string(&qc) {
        let _ = serde_json::from_str::<QuantizationConfig>(&json);
    }

    // BitNetConfig builder with quantization must not panic.
    let result = BitNetConfig::builder()
        .vocab_size(input.vocab_size as usize)
        .hidden_size(input.hidden_size as usize)
        .num_heads(input.num_heads as usize)
        .num_layers(input.num_layers as usize)
        .build();

    if let Ok(config) = result {
        let _ = config.validate();
    }

    // ModelConfig with extreme values must not panic on serde.
    let mc = ModelConfig {
        vocab_size: input.vocab_size as usize,
        hidden_size: input.hidden_size as usize,
        num_layers: input.num_layers as usize,
        num_heads: input.num_heads as usize,
        num_key_value_heads: input.num_heads as usize,
        intermediate_size: (input.hidden_size as usize).saturating_mul(4),
        max_position_embeddings: 2048,
        format: ModelFormat::Gguf,
        ..Default::default()
    };
    if let Ok(json) = serde_json::to_string(&mc) {
        let _ = serde_json::from_str::<ModelConfig>(&json);
    }

    // Validate scale factors don't cause panics (NaN, inf, zero, negative).
    for &s in input.scale_factors.iter().take(256) {
        let clamped = if s.is_finite() { s } else { 0.0 };
        let _ = clamped.abs();
        // Division by scale: must not panic even with zero.
        if clamped != 0.0 {
            let _ = 1.0f32 / clamped;
        }
    }

    // Block size edge cases: zero, 1, powers of 2, odd values.
    for bs in [0usize, 1, 2, 31, 32, 33, 63, 64, 128, 255, 256, 512, usize::MAX] {
        let edge_qc = QuantizationConfig { quantization_type: qt, block_size: bs, precision: 1.0 };
        let _ = format!("{edge_qc:?}");
    }
});

#![no_main]

use arbitrary::Arbitrary;
use bitnet_common::BitNetConfig;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct ConfigInput {
    vocab_size: u32,
    hidden_size: u32,
    num_layers: u16,
    num_heads: u16,
    num_kv_heads: u16,
    max_length: u32,
    temperature: f32,
    top_k: Option<u16>,
    top_p: Option<f32>,
    batch_size: u16,
    num_threads: Option<u8>,
    /// Raw JSON for serde round-trip.
    json_bytes: Vec<u8>,
}

fuzz_target!(|input: ConfigInput| {
    // Builder + validate must never panic — only return Err.
    let builder = BitNetConfig::builder()
        .vocab_size(input.vocab_size as usize)
        .hidden_size(input.hidden_size as usize)
        .num_layers(input.num_layers as usize)
        .num_heads(input.num_heads as usize)
        .num_key_value_heads(input.num_kv_heads as usize)
        .max_length(input.max_length as usize)
        .temperature(input.temperature)
        .top_k(input.top_k.map(|k| k as usize))
        .top_p(input.top_p)
        .batch_size(input.batch_size as usize)
        .num_threads(input.num_threads.map(|t| t as usize));

    match builder.build() {
        Ok(config) => {
            // A successfully built config must also pass validate().
            let _ = config.validate();
        }
        Err(_) => {
            // Validation rejection is fine.
        }
    }

    // Edge cases: zero-value fields.
    let _ = BitNetConfig::builder()
        .vocab_size(0)
        .hidden_size(0)
        .num_layers(0)
        .num_heads(0)
        .build();

    // Edge: huge values must not cause overflow or panic.
    let _ = BitNetConfig::builder()
        .vocab_size(usize::MAX)
        .hidden_size(usize::MAX)
        .num_layers(usize::MAX)
        .num_heads(usize::MAX)
        .build();

    // Edge: kv_heads > num_heads.
    let _ = BitNetConfig::builder()
        .num_heads(input.num_heads as usize)
        .num_key_value_heads((input.num_heads as usize).saturating_add(1))
        .build();

    // Serde round-trip: deserialize arbitrary JSON as ModelConfig.
    if let Ok(s) = std::str::from_utf8(&input.json_bytes) {
        let _ = serde_json::from_str::<bitnet_common::ModelConfig>(s);
    }
});

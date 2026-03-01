#![no_main]

use arbitrary::Arbitrary;
use bitnet_inference::config_builder::InferenceConfigBuilder;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct ConfigInput {
    temperature: f32,
    top_k: u32,
    top_p: f32,
    repetition_penalty: f32,
    seed: Option<u64>,
    max_tokens: u32,
    num_threads: u16,
    memory_limit_mb: u16,
    stream: bool,
    stop_sequences: Vec<u8>,
    stop_token_ids: Vec<u32>,
    /// Preset selector (0..4 maps to the five presets).
    preset: u8,
}

fuzz_target!(|input: ConfigInput| {
    // Test preset-based builder first
    let preset_builder = match input.preset % 5 {
        0 => InferenceConfigBuilder::new()
            .preset(bitnet_inference::config_builder::InferencePreset::Fast),
        1 => InferenceConfigBuilder::new()
            .preset(bitnet_inference::config_builder::InferencePreset::Balanced),
        2 => InferenceConfigBuilder::new()
            .preset(bitnet_inference::config_builder::InferencePreset::Quality),
        3 => InferenceConfigBuilder::new()
            .preset(bitnet_inference::config_builder::InferencePreset::Deterministic),
        _ => InferenceConfigBuilder::new()
            .preset(bitnet_inference::config_builder::InferencePreset::Debug),
    };
    // Preset build must always succeed
    let _ = preset_builder.build();

    // Parse stop sequences from raw bytes
    let stop_seqs: Vec<String> = input
        .stop_sequences
        .chunks(8)
        .filter_map(|b| std::str::from_utf8(b).ok())
        .map(|s| s.to_owned())
        .take(4)
        .collect();

    let stop_ids: Vec<u32> = input.stop_token_ids.iter().copied().take(8).collect();

    // Build config with arbitrary parameters — must never panic
    let result = InferenceConfigBuilder::new()
        .temperature(input.temperature)
        .top_k(input.top_k)
        .top_p(input.top_p)
        .repetition_penalty(input.repetition_penalty)
        .max_tokens(input.max_tokens)
        .num_threads(input.num_threads as usize)
        .memory_limit_mb(input.memory_limit_mb as usize)
        .stream(input.stream)
        .stop_sequences(stop_seqs)
        .stop_token_ids(stop_ids)
        .build();

    match result {
        Ok(config) => {
            // Valid config: validate() must also pass
            assert!(config.validate().is_ok(), "build() succeeded but validate() failed");
        }
        Err(_) => {
            // Validation error is expected for bad inputs
        }
    }

    // Also test with seed if provided
    if let Some(seed) = input.seed {
        let _ = InferenceConfigBuilder::new().seed(seed).build();
    }
});

#![no_main]

use arbitrary::Arbitrary;
use bitnet_inference::config::{GenerationConfig, InferenceConfig};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct ConfigInput {
    gen_configs: Vec<GenConfigParams>,
    inf_configs: Vec<InfConfigParams>,
}

#[derive(Arbitrary, Debug)]
struct GenConfigParams {
    max_new_tokens: u32,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    repetition_penalty: f32,
    stop_string_window: u16,
    seed: Option<u64>,
    skip_special_tokens: bool,
    eos_token_id: Option<u32>,
    add_bos: bool,
    stop_token_ids: Vec<u32>,
    stop_sequences: Vec<String>,
}

#[derive(Arbitrary, Debug)]
struct InfConfigParams {
    max_context_length: u32,
    num_threads: u8,
    batch_size: u8,
    mixed_precision: bool,
    memory_pool_size: u32,
}

fuzz_target!(|input: ConfigInput| {
    for params in input.gen_configs.into_iter().take(256) {
        let config = GenerationConfig::default()
            .with_max_tokens(params.max_new_tokens)
            .with_temperature(params.temperature)
            .with_top_k(params.top_k)
            .with_top_p(params.top_p)
            .with_repetition_penalty(params.repetition_penalty)
            .with_stop_string_window(params.stop_string_window as usize)
            .with_skip_special_tokens(params.skip_special_tokens)
            .with_eos_token_id(params.eos_token_id)
            .with_add_bos(params.add_bos)
            .with_stop_token_ids(params.stop_token_ids.into_iter().take(64).collect::<Vec<_>>())
            .with_stop_sequences(params.stop_sequences.into_iter().take(16).collect::<Vec<_>>());

        if let Some(seed) = params.seed {
            let config = config.with_seed(seed);
            let _ = config.validate();
        } else {
            let _ = config.validate();
        }

        // Verify builder presets never panic
        let _ = GenerationConfig::greedy().validate();
        let _ = GenerationConfig::creative().validate();
        let _ = GenerationConfig::balanced().validate();
    }

    for params in input.inf_configs.into_iter().take(256) {
        let mut config = InferenceConfig::default()
            .with_threads(params.num_threads as usize)
            .with_batch_size(params.batch_size as usize)
            .with_mixed_precision(params.mixed_precision)
            .with_memory_pool_size(params.memory_pool_size as usize);
        config.max_context_length = params.max_context_length as usize;

        let _ = config.validate();

        // Verify presets never panic
        let _ = InferenceConfig::cpu_optimized().validate();
        let _ = InferenceConfig::gpu_optimized().validate();
        let _ = InferenceConfig::memory_efficient().validate();
    }
});

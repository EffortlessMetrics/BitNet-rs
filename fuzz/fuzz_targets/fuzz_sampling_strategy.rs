#![no_main]

use arbitrary::Arbitrary;
use bitnet_sampling::{SamplingConfig, SamplingStrategy};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct StrategyInput {
    temperature: f32,
    top_k: u32,
    top_p: f32,
    repetition_penalty: f32,
    seed: Option<u64>,
    /// Sequence of config updates to apply after construction.
    updates: Vec<ConfigPatch>,
    /// Raw logit bytes for a quick sample after each update (up to 1024 f32s).
    logit_bytes: Vec<u8>,
    /// Context token IDs for repetition penalty.
    context: Vec<u8>,
}

#[derive(Arbitrary, Debug)]
struct ConfigPatch {
    temperature: f32,
    top_k: u32,
    top_p: f32,
    repetition_penalty: f32,
    seed: Option<u64>,
}

fuzz_target!(|input: StrategyInput| {
    // Construction with arbitrary parameters must never panic.
    let config = SamplingConfig {
        temperature: input.temperature,
        top_k: input.top_k,
        top_p: input.top_p,
        repetition_penalty: input.repetition_penalty,
        seed: input.seed,
    };
    let mut strategy = SamplingStrategy::new(config);

    // Decode a small logit buffer once for reuse.
    let aligned = (input.logit_bytes.len() / 4) * 4;
    let logits: Vec<f32> = input.logit_bytes[..aligned]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .take(1024)
        .collect();
    let context: Vec<u32> = input.context.iter().map(|&b| b as u32).take(64).collect();

    // Apply up to 16 config updates; sample after each if we have logits.
    for patch in input.updates.into_iter().take(16) {
        let new_cfg = SamplingConfig {
            temperature: patch.temperature,
            top_k: patch.top_k,
            top_p: patch.top_p,
            repetition_penalty: patch.repetition_penalty,
            seed: patch.seed,
        };
        strategy.update_config(new_cfg);

        if !logits.is_empty() {
            // Sampling may error (NaN logits, empty after filter) — that is fine.
            let _ = strategy.sample(&logits, &context);
        }
    }

    // Reset must never panic.
    strategy.reset();

    // One more sample after reset should not panic.
    if !logits.is_empty() {
        let _ = strategy.sample(&logits, &context);
    }
});

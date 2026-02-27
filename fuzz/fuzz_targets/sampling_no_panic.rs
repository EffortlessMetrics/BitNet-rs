#![no_main]

use arbitrary::Arbitrary;
use bitnet_sampling::{SamplingConfig, SamplingStrategy};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct SamplingInput {
    /// Raw bytes interpreted as little-endian f32 logits (up to 32 768 values).
    raw_logits: Vec<u8>,
    /// Context token IDs used for repetition-penalty tracking.
    context_tokens: Vec<u8>,
    /// Sampling parameters.
    temperature: f32,
    top_k: u16,
    top_p: f32,
    repetition_penalty: f32,
    seed: Option<u64>,
}

fuzz_target!(|input: SamplingInput| {
    // Decode bytes → f32 (little-endian, truncate to nearest multiple of 4).
    let aligned_len = (input.raw_logits.len() / 4) * 4;
    if aligned_len == 0 {
        return;
    }
    let logits: Vec<f32> = input.raw_logits[..aligned_len]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .take(32_768)
        .collect();
    if logits.is_empty() {
        return;
    }

    let vocab_size = logits.len();
    let context_tokens: Vec<u32> =
        input.context_tokens.iter().map(|&b| b as u32).take(256).collect();

    let config = SamplingConfig {
        temperature: input.temperature,
        // top_k must not exceed vocab size to avoid over-filtering.
        top_k: (input.top_k as u32).min(vocab_size as u32),
        top_p: input.top_p,
        repetition_penalty: input.repetition_penalty,
        seed: input.seed,
    };

    let mut strategy = SamplingStrategy::new(config);

    match strategy.sample(&logits, &context_tokens) {
        Ok(token_id) => {
            assert!(
                (token_id as usize) < vocab_size,
                "sample returned out-of-bounds token {token_id} for vocab size {vocab_size}",
            );
        }
        // Errors are acceptable (all-NaN input, empty after filtering, etc.).
        Err(_) => {}
    }
});

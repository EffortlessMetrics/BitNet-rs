#![no_main]

use arbitrary::Arbitrary;
use bitnet_sampling::SamplerChain;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct ChainInput {
    /// Raw bytes interpreted as f32 logits.
    raw_logits: Vec<u8>,
    /// Temperature (0 = greedy).
    temperature: f32,
    /// Top-k limit.
    top_k: u16,
    /// Top-p threshold.
    top_p: f32,
    /// Min-p threshold.
    min_p: f32,
    /// Typical-p threshold.
    typical_p: f32,
    /// Seed for reproducibility.
    seed: Option<u64>,
}

fuzz_target!(|input: ChainInput| {
    let aligned_len = (input.raw_logits.len() / 4) * 4;
    if aligned_len == 0 {
        return;
    }
    let logits: Vec<f32> = input.raw_logits[..aligned_len]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .take(2048)
        .collect();
    if logits.is_empty() {
        return;
    }

    let vocab_size = logits.len();

    let chain = SamplerChain::builder()
        .temperature(input.temperature)
        .top_k(input.top_k as usize)
        .top_p(input.top_p)
        .min_p(input.min_p)
        .typical(input.typical_p)
        .build(input.seed);

    if let Ok(token_id) = chain.sample(&logits) {
        assert!(
            (token_id as usize) < vocab_size,
            "chain returned OOB token {token_id} for vocab size {vocab_size}",
        );
    }

    // Verify stages are queryable without panic
    let _ = chain.stages();
});

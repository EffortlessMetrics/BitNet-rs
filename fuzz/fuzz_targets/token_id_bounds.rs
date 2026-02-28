#![no_main]

use arbitrary::Arbitrary;
use bitnet_generation::{StopCriteria, check_stop};
use bitnet_logits::{apply_repetition_penalty, apply_top_k, argmax};
use bitnet_sampling::{SamplingConfig, SamplingStrategy, greedy_sample};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct TokenBoundsInput {
    /// Raw logits bytes (interpreted as f32 LE).
    raw_logits: Vec<u8>,
    /// Vocabulary size (may differ from logits length).
    vocab_size: u16,
    /// Token IDs from prior context.
    context_ids: Vec<u32>,
    /// A single token ID to probe boundary behavior.
    probe_id: u32,
    /// Sampling seed.
    seed: u64,
    /// Temperature for sampling.
    temperature: f32,
    /// EOS token id.
    eos_id: Option<u32>,
}

fuzz_target!(|input: TokenBoundsInput| {
    // Decode logits from raw bytes.
    let aligned = (input.raw_logits.len() / 4) * 4;
    if aligned == 0 {
        return;
    }
    let logits: Vec<f32> = input.raw_logits[..aligned]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .take(1024)
        .collect();
    if logits.is_empty() {
        return;
    }

    let vocab = logits.len();
    let context: Vec<u32> = input.context_ids.iter().copied().take(128).collect();

    // --- greedy_sample: returned ID must be < vocab ---
    if let Ok(id) = greedy_sample(&logits) {
        assert!((id as usize) < vocab, "greedy_sample returned {id} for vocab size {vocab}",);
    }

    // --- SamplingStrategy: returned ID must be < vocab ---
    let config = SamplingConfig {
        temperature: input.temperature.abs().clamp(0.01, 10.0),
        top_k: (input.vocab_size as u32).min(vocab as u32),
        top_p: 0.95,
        repetition_penalty: 1.0,
        seed: Some(input.seed),
    };
    let mut strategy = SamplingStrategy::new(config);
    if let Ok(id) = strategy.sample(&logits, &context) {
        assert!((id as usize) < vocab, "SamplingStrategy returned {id} for vocab size {vocab}",);
    }

    // --- argmax: must be < vocab ---
    let idx = argmax(&logits);
    assert!(idx < vocab, "argmax returned {idx} for vocab size {vocab}");

    // --- apply_repetition_penalty with out-of-bounds IDs: must not panic ---
    {
        let mut l = logits.clone();
        let oob_ids: Vec<u32> = vec![0, input.probe_id, u32::MAX, vocab as u32, (vocab + 1) as u32];
        apply_repetition_penalty(&mut l, &oob_ids, 1.2);
    }

    // --- apply_top_k with k > vocab: must not panic ---
    {
        let mut l = logits.clone();
        let kept = apply_top_k(&mut l, vocab + 100);
        assert!(kept <= vocab);
    }

    // --- check_stop with boundary token IDs ---
    let criteria = StopCriteria {
        stop_token_ids: vec![0, u32::MAX, input.probe_id],
        stop_strings: vec![],
        max_tokens: 0,
        eos_token_id: input.eos_id,
    };
    // Must not panic on any token ID value.
    let _ = check_stop(&criteria, input.probe_id, &context, "");
    let _ = check_stop(&criteria, u32::MAX, &context, "");
    let _ = check_stop(&criteria, 0, &context, "");
});

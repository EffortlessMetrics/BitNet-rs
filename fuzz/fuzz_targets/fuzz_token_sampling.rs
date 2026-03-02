#![no_main]

use arbitrary::Arbitrary;
use bitnet_sampling::{SamplingConfig, SamplingStrategy, greedy_sample};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct TokenSamplingInput {
    raw_logits: Vec<u8>,
    context_tokens: Vec<u8>,
    temperature: f32,
    top_k: u16,
    top_p: f32,
    repetition_penalty: f32,
    seed: Option<u64>,
    try_greedy: bool,
}

fuzz_target!(|input: TokenSamplingInput| {
    // Decode raw bytes into f32 logits.
    let aligned_len = (input.raw_logits.len() / 4) * 4;
    if aligned_len == 0 {
        return;
    }
    let logits: Vec<f32> = input.raw_logits[..aligned_len]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .take(4096)
        .collect();
    if logits.is_empty() {
        return;
    }

    let vocab_size = logits.len();
    let context_tokens: Vec<u32> =
        input.context_tokens.iter().map(|&b| b as u32).take(128).collect();

    // Test greedy sampling path.
    if input.try_greedy {
        match greedy_sample(&logits) {
            Ok(token_id) => {
                // Invariant 1: Greedy token is in-bounds.
                assert!(
                    (token_id as usize) < vocab_size,
                    "greedy_sample OOB: {token_id} >= {vocab_size}"
                );
                // Invariant 2: Greedy selects the argmax when finite.
                if logits.iter().all(|x| x.is_finite()) {
                    let expected = logits
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(i, _)| i as u32)
                        .unwrap();
                    assert_eq!(token_id, expected, "greedy_sample did not select argmax");
                }
            }
            Err(_) => {}
        }
    }

    // Test configurable sampling.
    let config = SamplingConfig {
        temperature: input.temperature,
        top_k: (input.top_k as u32).min(vocab_size as u32),
        top_p: input.top_p,
        repetition_penalty: input.repetition_penalty,
        seed: input.seed,
    };

    let mut strategy = SamplingStrategy::new(config);

    match strategy.sample(&logits, &context_tokens) {
        Ok(token_id) => {
            // Invariant 3: Sampled token is in-bounds.
            assert!((token_id as usize) < vocab_size, "sample OOB: {token_id} >= {vocab_size}");
        }
        // Errors are acceptable (all-NaN input, empty after filtering, etc.).
        Err(_) => {}
    }

    // Invariant 4: Deterministic seed produces same result twice.
    if let Some(seed) = input.seed {
        let det_config = SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: Some(seed),
        };
        // Only test with all-finite logits for determinism check.
        if logits.iter().all(|x| x.is_finite()) {
            let mut s1 = SamplingStrategy::new(det_config.clone());
            let mut s2 = SamplingStrategy::new(det_config);
            let r1 = s1.sample(&logits, &[]);
            let r2 = s2.sample(&logits, &[]);
            if let (Ok(t1), Ok(t2)) = (r1, r2) {
                assert_eq!(t1, t2, "same seed should produce same token");
            }
        }
    }
});

#![no_main]

use arbitrary::Arbitrary;
use bitnet_sampling::{SamplingConfig, SamplingStrategy};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct TokenSamplingInput {
    /// Raw bytes interpreted as little-endian f32 logits.
    raw_logits: Vec<u8>,
    /// Context token IDs for repetition penalty.
    context_tokens: Vec<u8>,
    /// Strategy selector: 0=greedy, 1=top-k, 2=nucleus, 3=combined.
    strategy: u8,
    /// Top-k value.
    top_k: u16,
    /// Top-p value.
    top_p_raw: u8,
    /// Repetition penalty (raw byte scaled to [0.5, 2.0]).
    rep_penalty_raw: u8,
    /// Random seed.
    seed: u64,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned]
        .chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fuzz_target!(|input: TokenSamplingInput| {
    let logits = bytes_to_f32(&input.raw_logits, 32_768);
    if logits.is_empty() {
        return;
    }
    let vocab_size = logits.len();

    let context_tokens: Vec<u32> =
        input.context_tokens.iter().take(256).map(|&b| (b as u32) % vocab_size as u32).collect();

    // Map top_p_raw (0..255) to [0.0, 1.0].
    let top_p = input.top_p_raw as f32 / 255.0;
    // Map rep_penalty_raw to [0.5, 2.0].
    let rep_penalty = 0.5 + (input.rep_penalty_raw as f32 / 255.0) * 1.5;

    let configs = match input.strategy % 4 {
        // Greedy: temperature=0, no top-k/top-p.
        0 => vec![SamplingConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: rep_penalty,
            seed: Some(input.seed),
        }],
        // Top-k only.
        1 => vec![SamplingConfig {
            temperature: 1.0,
            top_k: (input.top_k as u32).min(vocab_size as u32),
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: Some(input.seed),
        }],
        // Nucleus (top-p) only.
        2 => vec![SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p,
            repetition_penalty: 1.0,
            seed: Some(input.seed),
        }],
        // Combined: top-k + top-p + repetition penalty.
        _ => vec![SamplingConfig {
            temperature: 0.7,
            top_k: (input.top_k as u32).min(vocab_size as u32),
            top_p,
            repetition_penalty: rep_penalty,
            seed: Some(input.seed),
        }],
    };

    for config in configs {
        let mut strategy = SamplingStrategy::new(config);

        match strategy.sample(&logits, &context_tokens) {
            Ok(token_id) => {
                // Invariant 1: Token ID is within vocabulary bounds.
                assert!(
                    (token_id as usize) < vocab_size,
                    "sampled token {token_id} >= vocab_size {vocab_size}"
                );
            }
            Err(_) => {
                // Errors are expected for degenerate inputs (all-NaN, empty after filtering).
            }
        }
    }

    // Invariant 2: Greedy sampling is deterministic — same seed, same input, same result.
    let greedy_cfg = SamplingConfig {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(42),
    };
    let result1 = SamplingStrategy::new(greedy_cfg.clone()).sample(&logits, &context_tokens);
    let result2 = SamplingStrategy::new(greedy_cfg).sample(&logits, &context_tokens);
    match (result1, result2) {
        (Ok(a), Ok(b)) => {
            assert_eq!(a, b, "greedy sampling must be deterministic");
        }
        (Err(_), Err(_)) => {}
        _ => panic!("greedy sampling inconsistency: one succeeded, one failed"),
    }

    // Invariant 3: Same seed produces same result for stochastic sampling.
    let seeded_cfg = SamplingConfig {
        temperature: 0.7,
        top_k: 50,
        top_p: 0.9,
        repetition_penalty: 1.0,
        seed: Some(input.seed),
    };
    let r1 = SamplingStrategy::new(seeded_cfg.clone()).sample(&logits, &context_tokens);
    let r2 = SamplingStrategy::new(seeded_cfg).sample(&logits, &context_tokens);
    match (r1, r2) {
        (Ok(a), Ok(b)) => {
            assert_eq!(a, b, "seeded sampling must be reproducible");
        }
        (Err(_), Err(_)) => {}
        _ => panic!("seeded sampling inconsistency"),
    }
});

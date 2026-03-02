#![no_main]

use arbitrary::Arbitrary;
use bitnet_sampling::{SamplingConfig, SamplingStrategy, greedy_sample};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct SamplingStrategyInput {
    raw_logits: Vec<u8>,
    context_tokens: Vec<u8>,
    temperature: f32,
    top_k: u16,
    top_p: f32,
    repetition_penalty: f32,
    seed: Option<u64>,
    extreme_temps: Vec<f32>,
}

fuzz_target!(|input: SamplingStrategyInput| {
    // Decode bytes → f32 logits.
    let aligned = (input.raw_logits.len() / 4) * 4;
    if aligned == 0 {
        return;
    }
    let logits: Vec<f32> = input.raw_logits[..aligned]
        .chunks_exact(4)
        .take(32_768)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    if logits.is_empty() {
        return;
    }

    let vocab_size = logits.len();
    let context: Vec<u32> = input.context_tokens.iter().take(256).map(|&b| b as u32).collect();

    // Fuzz SamplingStrategy with arbitrary parameters.
    let config = SamplingConfig {
        temperature: input.temperature,
        top_k: (input.top_k as u32).min(vocab_size as u32),
        top_p: input.top_p,
        repetition_penalty: input.repetition_penalty,
        seed: input.seed,
    };

    let mut strategy = SamplingStrategy::new(config);
    match strategy.sample(&logits, &context) {
        Ok(token_id) => {
            assert!(
                (token_id as usize) < vocab_size,
                "out-of-bounds token {token_id} for vocab {vocab_size}",
            );
        }
        Err(_) => {}
    }

    // Reset and re-sample must not panic.
    strategy.reset();
    let _ = strategy.sample(&logits, &context);

    // Greedy sample must not panic.
    let _ = greedy_sample(&logits);

    // Extreme temperature configs must not panic.
    for &temp in input.extreme_temps.iter().take(8) {
        let extreme_config = SamplingConfig {
            temperature: temp,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: Some(42),
        };
        let mut s = SamplingStrategy::new(extreme_config);
        let _ = s.sample(&logits, &[]);
    }

    // Edge cases: single-element logits, all-same logits.
    let single = vec![0.5f32];
    let _ = greedy_sample(&single);
    let mut s2 = SamplingStrategy::new(SamplingConfig {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(0),
    });
    let _ = s2.sample(&single, &[]);
});

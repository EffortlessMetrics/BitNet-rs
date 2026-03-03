#![no_main]

use arbitrary::Arbitrary;
use bitnet_logits::{argmax, softmax_in_place};
use bitnet_sampling::{SamplingConfig, SamplingStrategy, greedy_sample};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct SamplingInput {
    vocab_size: u8,
    logit_bytes: Vec<u8>,
    temperature_byte: u8,
    top_k_byte: u8,
    top_p_byte: u8,
    rep_penalty_byte: u8,
    seed: u64,
    context_tokens: Vec<u8>,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned]
        .chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fuzz_target!(|input: SamplingInput| {
    let vocab_size = (input.vocab_size as usize % 128) + 2;

    let mut logits = bytes_to_f32(&input.logit_bytes, vocab_size);
    if logits.len() < vocab_size {
        return;
    }
    // Sanitize to finite values.
    for v in logits.iter_mut() {
        if !v.is_finite() {
            *v = 0.0;
        }
    }
    let logits = &mut logits[..vocab_size];

    // Map fuzz bytes to sampling parameters.
    let temperature = (input.temperature_byte as f32 / 255.0) * 2.0; // [0, 2]
    let top_k = (input.top_k_byte as u32) % (vocab_size as u32 + 1);
    let top_p = input.top_p_byte as f32 / 255.0; // [0, 1]
    let rep_penalty = 1.0 + (input.rep_penalty_byte as f32 / 255.0); // [1.0, 2.0]
    let context: Vec<u32> =
        input.context_tokens.iter().take(32).map(|&t| (t as u32) % (vocab_size as u32)).collect();

    // greedy_sample must not panic on valid logits.
    if let Ok(token) = greedy_sample(logits) {
        assert!((token as usize) < vocab_size, "greedy token {token} out of range");
        // Greedy must pick the argmax.
        let expected = argmax(logits);
        assert_eq!(token as usize, expected, "greedy_sample disagrees with argmax");
    }

    // softmax_in_place must produce valid probability distribution.
    let mut softmax_logits = logits.to_vec();
    softmax_in_place(&mut softmax_logits);
    for (i, &p) in softmax_logits.iter().enumerate() {
        assert!(!p.is_nan(), "softmax NaN at {i}");
        assert!(p >= 0.0, "softmax negative at {i}: {p}");
    }
    let sum: f32 = softmax_logits.iter().sum();
    if sum.is_finite() {
        assert!((sum - 1.0).abs() < 1e-3, "softmax sum {sum} not close to 1.0");
    }

    // SamplingStrategy: fuzz the full pipeline.
    let config = SamplingConfig {
        temperature,
        top_k,
        top_p,
        repetition_penalty: rep_penalty,
        seed: Some(input.seed),
    };
    let mut strategy = SamplingStrategy::new(config);
    // Sample must not panic (errors are OK for extreme parameters).
    if let Ok(token) = strategy.sample(logits, &context) {
        assert!((token as usize) < vocab_size, "sampled token {token} out of range");
    }

    // Reset must not panic.
    strategy.reset();

    // update_config must not panic.
    strategy.update_config(SamplingConfig {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(42),
    });
    if let Ok(token) = strategy.sample(logits, &[]) {
        assert!(
            (token as usize) < vocab_size,
            "sampled token {token} out of range after config update"
        );
    }
});

#![no_main]

use arbitrary::Arbitrary;
use bitnet_sampling::{
    SamplingConfig, SamplingStrategy, apply_temperature, apply_top_k, apply_top_p, greedy_sample,
    softmax_in_place,
};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct SamplingInput {
    /// Raw logit values.
    logits: Vec<f32>,
    /// Temperature (will be clamped).
    temperature: f32,
    /// Top-k value.
    top_k: u16,
    /// Top-p value.
    top_p: f32,
    /// Repetition penalty.
    rep_penalty: f32,
    /// Seed for reproducibility.
    seed: u64,
    /// Context tokens for repetition penalty.
    context_tokens: Vec<u32>,
    /// Inject adversarial logit patterns.
    inject_uniform: bool,
    inject_spike: bool,
    inject_all_negative: bool,
}

const MAX_VOCAB: usize = 512;

fuzz_target!(|input: SamplingInput| {
    if input.logits.is_empty() {
        return;
    }

    let logits: Vec<f32> = input
        .logits
        .iter()
        .take(MAX_VOCAB)
        .map(|&x| if x.is_nan() || x.is_infinite() { 0.0 } else { x.clamp(-1e6, 1e6) })
        .collect();

    let vocab_size = logits.len();

    // ── Test individual transforms ────────────────────────────
    {
        let temp = input.temperature.abs().clamp(0.01, 100.0);
        let mut buf = logits.clone();
        apply_temperature(&mut buf, temp);
        for (i, &v) in buf.iter().enumerate() {
            assert!(v.is_finite(), "temperature produced non-finite at {i}");
        }
    }

    {
        let k = (input.top_k as usize).clamp(1, vocab_size);
        let mut buf = logits.clone();
        apply_top_k(&mut buf, k);
        // After top-k, at most k values should remain non-NEG_INFINITY.
        let active = buf.iter().filter(|&&v| v > f32::NEG_INFINITY).count();
        assert!(active <= k, "top_k({k}) left {active} active logits");
    }

    {
        let mut probs = logits.clone();
        softmax_in_place(&mut probs);
        let p = input.top_p.abs().clamp(0.01, 1.0);
        apply_top_p(&mut probs, p);
        for (i, &v) in probs.iter().enumerate() {
            assert!(v >= 0.0, "top_p produced negative at {i}");
        }
    }

    // ── Test greedy sampling ──────────────────────────────────
    if let Ok(token) = greedy_sample(&logits) {
        assert!((token as usize) < vocab_size, "greedy token OOB");
    }

    // ── Test full sampling pipeline ───────────────────────────
    let config = SamplingConfig {
        temperature: input.temperature.abs().clamp(0.0, 10.0),
        top_k: (input.top_k as u32).min(vocab_size as u32),
        top_p: input.top_p.abs().clamp(0.0, 1.0),
        repetition_penalty: input.rep_penalty.abs().clamp(0.5, 5.0),
        seed: Some(input.seed),
    };

    let mut strategy = SamplingStrategy::new(config);
    let ctx: Vec<u32> = input.context_tokens.iter().take(64).copied().collect();

    // Must not panic.
    if let Ok(token) = strategy.sample(&logits, &ctx) {
        assert!((token as usize) < vocab_size, "sampled token {token} >= vocab {vocab_size}");
    }

    // ── Test adversarial distributions ────────────────────────
    if input.inject_uniform {
        let uniform = vec![1.0f32 / vocab_size as f32; vocab_size];
        let mut s = SamplingStrategy::new(SamplingConfig {
            temperature: 1.0,
            seed: Some(input.seed),
            ..Default::default()
        });
        let _ = s.sample(&uniform, &[]);
    }

    if input.inject_spike {
        let mut spike = vec![-1e6f32; vocab_size];
        let spike_idx = (input.seed as usize) % vocab_size;
        spike[spike_idx] = 1e6;
        if let Ok(token) = greedy_sample(&spike) {
            assert_eq!(token as usize, spike_idx, "greedy missed spike");
        }
    }

    if input.inject_all_negative {
        let neg = vec![-100.0f32; vocab_size];
        // All-negative logits must still produce a valid token.
        if let Ok(token) = greedy_sample(&neg) {
            assert!((token as usize) < vocab_size);
        }
    }
});

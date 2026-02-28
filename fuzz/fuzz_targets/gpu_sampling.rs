#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct GpuSamplingInput {
    logits: Vec<f32>,
    temperature: f32,
    top_k: u8,
    top_p: f32,
    context_tokens: Vec<u8>,
}

fuzz_target!(|input: GpuSamplingInput| {
    // Cap logits to prevent timeout.
    let logits: Vec<f32> = input.logits.into_iter().take(256).collect();
    if logits.is_empty() {
        return;
    }

    // Only test with valid temperatures.
    let temperature = input.temperature.abs().clamp(0.01, 100.0);
    let top_k = (input.top_k as usize).clamp(1, logits.len());
    let top_p = input.top_p.clamp(0.0, 1.0);

    // Temperature scaling — must not panic.
    {
        let mut l = logits.clone();
        bitnet_logits::apply_temperature(&mut l, temperature);
        for &v in &l {
            if v.is_finite() {
                // Scaled values should remain finite for finite inputs with finite temp.
            }
        }
    }

    // Top-k — returned count must be ≤ logits.len().
    {
        let mut l = logits.clone();
        let kept = bitnet_logits::apply_top_k(&mut l, top_k);
        assert!(kept <= l.len(), "apply_top_k returned count > len");
    }

    // Softmax invariant on finite logits: probs sum to ~1.0.
    {
        let finite: Vec<f32> = logits.iter().copied().filter(|x| x.is_finite()).collect();
        if !finite.is_empty() {
            let mut l = finite;
            bitnet_logits::softmax_in_place(&mut l);
            for &p in &l {
                assert!(p >= 0.0, "softmax produced negative probability: {p}");
                assert!(p.is_finite(), "softmax produced non-finite: {p}");
            }
            let sum: f32 = l.iter().sum();
            assert!(sum > 0.99 && sum < 1.01, "softmax sum out of range: {sum}");
        }
    }

    // Top-p after softmax — must not panic.
    {
        let finite: Vec<f32> = logits.iter().copied().filter(|x| x.is_finite()).collect();
        if !finite.is_empty() {
            let mut l = finite;
            bitnet_logits::softmax_in_place(&mut l);
            bitnet_logits::apply_top_p(&mut l, top_p);
        }
    }

    // Full SamplingStrategy pipeline — must not panic.
    {
        let finite: Vec<f32> = logits.iter().copied().filter(|x| x.is_finite()).collect();
        if !finite.is_empty() {
            let config = bitnet_sampling::SamplingConfig {
                temperature,
                top_k: top_k as u32,
                top_p,
                repetition_penalty: 1.0,
                seed: Some(42),
            };
            let context: Vec<u32> =
                input.context_tokens.iter().take(256).map(|&b| b as u32).collect();
            let mut strategy = bitnet_sampling::SamplingStrategy::new(config);
            if let Ok(token) = strategy.sample(&finite, &context) {
                assert!(
                    (token as usize) < finite.len(),
                    "sampled token index {token} >= logits len {}",
                    finite.len()
                );
            }
        }
    }
});

#![no_main]

use arbitrary::Arbitrary;
use bitnet_sampling::MirostatSampler;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct MirostatInput {
    /// Raw bytes interpreted as f32 logits.
    raw_logits: Vec<u8>,
    /// Target surprise (τ).
    tau: f32,
    /// Learning rate (η).
    eta: f32,
    /// Optional seed for RNG.
    seed: Option<u64>,
    /// Number of sequential samples to draw (tests mu drift).
    num_samples: u8,
}

fuzz_target!(|input: MirostatInput| {
    let aligned_len = (input.raw_logits.len() / 4) * 4;
    if aligned_len == 0 {
        return;
    }
    let logits: Vec<f32> = input.raw_logits[..aligned_len]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .take(1024)
        .collect();
    if logits.is_empty() {
        return;
    }

    // Clamp tau/eta to avoid degenerate but allow wide range
    let tau = input.tau.clamp(0.01, 100.0);
    let eta = input.eta.clamp(0.001, 10.0);
    let num_samples = (input.num_samples % 16) + 1;

    let mut sampler = MirostatSampler::new(tau, eta, input.seed);

    for _ in 0..num_samples {
        if let Ok(token_id) = sampler.sample(&logits) {
            assert!(
                (token_id as usize) < logits.len(),
                "mirostat returned OOB token {token_id} for vocab size {}",
                logits.len(),
            );
        }
    }

    // Reset must not panic
    sampler.reset();

    // Sample again after reset — must still work
    let _ = sampler.sample(&logits);
});

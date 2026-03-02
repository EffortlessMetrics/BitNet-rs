#![no_main]

//! Fuzz inference warmup configuration parsing: exercises `WarmupConfig`
//! builder methods, preset factories, and `run_warmup` with arbitrary
//! iteration counts, sequence lengths, and timeouts to verify no panics.

use std::time::Duration;

use arbitrary::Arbitrary;
use bitnet_inference::warmup::{
    WarmupConfig, WarmupStatus, run_warmup, skip_warmup, synthetic_tokens,
};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct WarmupInput {
    iterations: u16,
    seq_len: u16,
    timeout_ms: u32,
    synthetic_vocab_size: u16,
    synthetic_seq_len: u16,
}

fuzz_target!(|input: WarmupInput| {
    // Build config with arbitrary parameters.
    let timeout = Duration::from_millis(input.timeout_ms.min(100) as u64);
    let config = WarmupConfig::default()
        .with_iterations(input.iterations as usize)
        .with_seq_len(input.seq_len as usize)
        .with_timeout(timeout);

    let _ = format!("{:?}", config);

    // Preset factories must not panic.
    let _ = WarmupConfig::fast();
    let _ = WarmupConfig::thorough();

    // Run warmup with a trivial iteration function (bounded timeout).
    let capped_iters = (input.iterations as usize).min(8);
    let capped_config = WarmupConfig::default()
        .with_iterations(capped_iters)
        .with_seq_len((input.seq_len as usize).min(64))
        .with_timeout(Duration::from_millis(50));

    let result = run_warmup(&capped_config, |_i| Duration::from_micros(1));

    // Result accessors must not panic.
    let _ = result.avg_iteration_time();
    let _ = result.is_success();
    let _ = result.speedup_ratio();
    let _ = format!("{:?}", result.status);
    assert!(result.iterations_completed <= capped_iters);

    // skip_warmup must return a valid result.
    let skipped = skip_warmup();
    assert!(matches!(skipped.status, WarmupStatus::Skipped));
    assert!(skipped.is_success());

    // synthetic_tokens must not panic for any vocab/len combo.
    let vocab = (input.synthetic_vocab_size as u32).max(1);
    let slen = (input.synthetic_seq_len as usize).min(512);
    let tokens = synthetic_tokens(slen, vocab);
    assert_eq!(tokens.len(), slen);
    for &t in &tokens {
        assert!(t < vocab);
    }
});

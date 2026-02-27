#![no_main]

use arbitrary::Arbitrary;
use bitnet_engine_core::{ConcurrencyConfig, EngineStateTracker, SessionConfig};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct EngineInput {
    model_path: Vec<u8>,
    tokenizer_path: Vec<u8>,
    backend: Vec<u8>,
    max_context: usize,
    seed: Option<u64>,
    max_concurrent: usize,
    active_sessions: usize,
    /// Drive state-machine transitions (start=true, finish=false).
    transitions: Vec<bool>,
}

fuzz_target!(|input: EngineInput| {
    let model_path = String::from_utf8_lossy(&input.model_path).into_owned();
    let tokenizer_path = String::from_utf8_lossy(&input.tokenizer_path).into_owned();
    let backend = String::from_utf8_lossy(&input.backend).into_owned();

    // SessionConfig::validate must never panic — it may return Err.
    let config = SessionConfig {
        model_path,
        tokenizer_path,
        backend,
        max_context: input.max_context,
        seed: input.seed,
    };
    let _ = config.validate();

    // ConcurrencyConfig::allows must not panic for any combination.
    let concurrency = ConcurrencyConfig { max_concurrent: input.max_concurrent };
    let _ = concurrency.allows(input.active_sessions);

    // EngineStateTracker state machine must not panic regardless of call order.
    let mut tracker = EngineStateTracker::new();
    for &do_start in input.transitions.iter().take(16) {
        if do_start {
            let _ = tracker.start();
        } else {
            let _ = tracker.finish();
        }
    }
});

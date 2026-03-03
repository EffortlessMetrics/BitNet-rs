#![no_main]

use arbitrary::Arbitrary;
use bitnet_inference::{BatchConfig, BatchRequest, BatchScheduler, GenerationConfig};
use libfuzzer_sys::fuzz_target;
use std::time::Duration;

#[derive(Arbitrary, Debug)]
struct BatchInput {
    /// Max batch size (clamped to reasonable range).
    max_batch_size: u8,
    /// Timeout in milliseconds.
    timeout_ms: u16,
    /// Max total tokens.
    max_total_tokens: u16,
    /// Requests to add.
    requests: Vec<FuzzRequest>,
}

#[derive(Arbitrary, Debug)]
struct FuzzRequest {
    /// Raw bytes for prompt text.
    prompt_bytes: Vec<u8>,
    /// Max new tokens for this request.
    max_new_tokens: u16,
    /// Temperature (raw bits).
    temperature_bits: u16,
}

fuzz_target!(|input: BatchInput| {
    // Build config with clamped values to avoid panics from zero
    let max_batch_size = (input.max_batch_size as usize % 64) + 1;
    let timeout = Duration::from_millis(input.timeout_ms as u64);
    let max_total_tokens = (input.max_total_tokens as usize % 65536) + 1;

    let config = BatchConfig::new(max_batch_size, timeout).with_max_total_tokens(max_total_tokens);

    // validate must not panic
    let _ = config.validate();

    let scheduler = BatchScheduler::new(config);

    // Build batch request from fuzzed data
    let mut batch = BatchRequest::new();

    for req in input.requests.iter().take(128) {
        let prompt = String::from_utf8_lossy(&req.prompt_bytes).into_owned();
        let temperature = (req.temperature_bits as f32) / 1000.0;

        let mut gen_config = GenerationConfig::default();
        gen_config.max_new_tokens = (req.max_new_tokens as u32) % 4096;
        gen_config.temperature = temperature.clamp(0.0, 10.0);

        let id = batch.add(prompt, gen_config);

        // Invariant: returned ID matches insertion order
        assert_eq!(id, batch.len() - 1, "add() returned unexpected ID");
    }

    // schedule must not panic
    let scheduled = scheduler.schedule(&batch);

    // Invariant: scheduled count <= max_batch_size
    assert!(
        scheduled.len() <= max_batch_size,
        "scheduled {} > max_batch_size {}",
        scheduled.len(),
        max_batch_size
    );

    // Invariant: scheduled count <= total requests
    assert!(
        scheduled.len() <= batch.len(),
        "scheduled {} > batch size {}",
        scheduled.len(),
        batch.len()
    );

    // Invariant: all scheduled IDs are valid
    for &id in &scheduled {
        assert!(batch.get(id).is_some(), "scheduled ID {id} not found in batch");
    }

    // Invariant: no duplicate IDs
    let mut seen = std::collections::HashSet::new();
    for &id in &scheduled {
        assert!(seen.insert(id), "duplicate scheduled ID {id}");
    }

    // Empty batch must yield empty schedule
    let empty_batch = BatchRequest::new();
    let empty_schedule = scheduler.schedule(&empty_batch);
    assert!(empty_schedule.is_empty(), "empty batch should produce empty schedule");
});

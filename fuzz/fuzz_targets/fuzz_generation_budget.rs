//! Fuzz generation budget tracking with random token counts and stop criteria.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct BudgetInput {
    max_new_tokens: u16,
    seed: Option<u64>,
    stop_token_ids: Vec<u32>,
    stop_strings: Vec<String>,
    max_tokens: u16,
    eos_token_id: Option<u32>,
    token_sequence: Vec<u32>,
    decoded_tail: String,
}

fuzz_target!(|input: BudgetInput| {
    use bitnet_generation::{GenerationConfig, StopCriteria, check_stop};

    // Build config — must not panic
    let _cfg = GenerationConfig {
        max_new_tokens: input.max_new_tokens as usize,
        seed: input.seed,
        stop_criteria: StopCriteria {
            stop_token_ids: input.stop_token_ids.iter().copied().take(64).collect(),
            stop_strings: input
                .stop_strings
                .iter()
                .take(16)
                .map(|s| s.chars().take(128).collect())
                .collect(),
            max_tokens: input.max_tokens as usize,
            eos_token_id: input.eos_token_id,
        },
    };

    // Simulate generation loop with stop checking
    let criteria = &_cfg.stop_criteria;
    let tokens: Vec<u32> = input.token_sequence.iter().copied().take(256).collect();
    let tail: String = input.decoded_tail.chars().take(512).collect();

    for (i, &tok) in tokens.iter().enumerate() {
        let generated = &tokens[..=i];
        let reason = check_stop(criteria, tok, generated, &tail);
        // If we got a stop reason, generation loop would break — that's fine
        if reason.is_some() {
            break;
        }
    }

    // Verify default config doesn't panic
    let _default = GenerationConfig::default();

    // Verify serde roundtrip doesn't panic
    if let Ok(json) = serde_json::to_string(&_cfg) {
        let _: Result<GenerationConfig, _> = serde_json::from_str(&json);
    }
});

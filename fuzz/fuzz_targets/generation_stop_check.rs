#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct StopCheckInput {
    /// Token IDs that trigger a stop.
    stop_token_ids: Vec<u32>,
    /// Short stop strings (truncated to keep corpus small).
    stop_strings: Vec<u8>,
    max_tokens: u16,
    eos_token: Option<u32>,
    /// Token being evaluated.
    token_id: u32,
    /// Tokens already generated.
    generated: Vec<u32>,
    /// Decoded tail text bytes (arbitrary UTF-8 attempt; invalid sequences skipped).
    decoded_tail: Vec<u8>,
}

fuzz_target!(|input: StopCheckInput| {
    use bitnet_generation::{StopCriteria, check_stop};

    // Build stop strings: try to decode as UTF-8 chunks of ≤ 32 bytes.
    let stop_strings: Vec<String> = input
        .stop_strings
        .chunks(8)
        .filter_map(|b| std::str::from_utf8(b).ok())
        .map(|s| s.to_owned())
        .take(4)
        .collect();

    let criteria = StopCriteria {
        stop_token_ids: input.stop_token_ids.iter().copied().take(16).collect(),
        stop_strings,
        max_tokens: input.max_tokens as usize,
        eos_token_id: input.eos_token,
    };

    let generated: Vec<u32> = input.generated.iter().copied().take(256).collect();

    // Best-effort UTF-8 from raw bytes.
    let decoded_tail = std::str::from_utf8(&input.decoded_tail).unwrap_or("");

    // Must never panic, regardless of input.
    let _result = check_stop(&criteria, input.token_id, &generated, decoded_tail);
});

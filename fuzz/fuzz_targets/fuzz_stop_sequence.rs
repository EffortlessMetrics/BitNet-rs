#![no_main]

use arbitrary::Arbitrary;
use bitnet_generation::{StopCriteria, check_stop};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct StopSeqInput {
    /// Token IDs that should trigger a stop.
    stop_token_ids: Vec<u32>,
    /// Raw bytes split into candidate stop strings (UTF-8 validated).
    stop_string_bytes: Vec<u8>,
    /// Additional stop strings as separate byte vectors.
    extra_stops: Vec<Vec<u8>>,
    max_tokens: u16,
    eos_token: Option<u32>,
    /// The token being evaluated right now.
    current_token: u32,
    /// Tokens already generated.
    generated: Vec<u32>,
    /// Decoded tail text (raw bytes, best-effort UTF-8).
    tail_bytes: Vec<u8>,
}

fuzz_target!(|input: StopSeqInput| {
    // Build stop strings from both sources.
    let mut stop_strings: Vec<String> = input
        .stop_string_bytes
        .chunks(16)
        .filter_map(|b| std::str::from_utf8(b).ok())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_owned())
        .take(8)
        .collect();

    for extra in input.extra_stops.iter().take(4) {
        if let Ok(s) = std::str::from_utf8(extra) {
            if !s.is_empty() && s.len() <= 64 {
                stop_strings.push(s.to_owned());
            }
        }
    }

    let criteria = StopCriteria {
        stop_token_ids: input.stop_token_ids.iter().copied().take(32).collect(),
        stop_strings,
        max_tokens: input.max_tokens as usize,
        eos_token_id: input.eos_token,
    };

    let generated: Vec<u32> = input.generated.iter().copied().take(512).collect();
    let decoded_tail = std::str::from_utf8(&input.tail_bytes).unwrap_or("");

    // Must never panic regardless of input.
    let _result = check_stop(&criteria, input.current_token, &generated, decoded_tail);

    // Also exercise with empty generated history.
    let _result2 = check_stop(&criteria, input.current_token, &[], decoded_tail);

    // And with empty decoded tail.
    let _result3 = check_stop(&criteria, input.current_token, &generated, "");
});

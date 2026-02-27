#![no_main]

use arbitrary::Arbitrary;
use bitnet_generation::{StopCriteria, check_stop};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Input {
    /// Token IDs that immediately trigger a stop.
    stop_token_ids: Vec<u32>,
    /// Raw bytes used to build stop strings; chunked into valid UTF-8 slices.
    stop_strings_raw: Vec<u8>,
    /// Hard cap on generated tokens (0 = unlimited).
    max_tokens: u16,
    /// Model EOS token, if any.
    eos_token_id: Option<u32>,
    /// The token being evaluated this step.
    token_id: u32,
    /// Tokens produced so far (capped to keep corpus manageable).
    generated: Vec<u32>,
    /// Decoded tail bytes – valid UTF-8 sub-slices used; remainder skipped.
    decoded_tail_raw: Vec<u8>,
}

fuzz_target!(|input: Input| {
    let stop_strings: Vec<String> = input
        .stop_strings_raw
        .chunks(8)
        .filter_map(|b| std::str::from_utf8(b).ok())
        .map(str::to_owned)
        .take(4)
        .collect();

    let criteria = StopCriteria {
        stop_token_ids: input.stop_token_ids.into_iter().take(16).collect(),
        stop_strings,
        max_tokens: input.max_tokens as usize,
        eos_token_id: input.eos_token_id,
    };

    let generated: Vec<u32> = input.generated.into_iter().take(256).collect();
    let decoded_tail = std::str::from_utf8(&input.decoded_tail_raw).unwrap_or("");

    // Must never panic for any input.
    let result = check_stop(&criteria, input.token_id, &generated, decoded_tail);

    // Invariant: empty stop conditions → generation always continues.
    if criteria.stop_token_ids.is_empty()
        && criteria.stop_strings.is_empty()
        && criteria.eos_token_id.is_none()
        && criteria.max_tokens == 0
    {
        assert!(
            result.is_none(),
            "check_stop must return None when all stop conditions are empty, got {result:?}"
        );
    }
});

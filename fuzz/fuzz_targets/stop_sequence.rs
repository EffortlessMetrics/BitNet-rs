#![no_main]

use arbitrary::Arbitrary;
use bitnet_generation::{StopCriteria, StopReason, check_stop};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct StopSeqInput {
    /// Stop strings built from raw bytes (tests multi-byte UTF-8 boundaries).
    stop_fragments: Vec<Vec<u8>>,
    /// Token IDs that immediately stop.
    stop_ids: Vec<u32>,
    /// EOS token.
    eos: Option<u32>,
    /// Rolling tail buffer content (simulates decoded text accumulation).
    tail_chunks: Vec<Vec<u8>>,
    /// Token IDs generated so far.
    generated: Vec<u32>,
    /// Current token being evaluated.
    current_token: u32,
    /// Max tokens budget.
    max_tokens: u16,
}

fuzz_target!(|input: StopSeqInput| {
    // Build stop strings from fragments, filtering to valid UTF-8.
    let stop_strings: Vec<String> = input
        .stop_fragments
        .iter()
        .filter_map(|b| std::str::from_utf8(b).ok())
        .filter(|s| !s.is_empty())
        .take(8)
        .map(str::to_owned)
        .collect();

    let stop_ids: Vec<u32> = input.stop_ids.into_iter().take(16).collect();
    let generated: Vec<u32> = input.generated.into_iter().take(256).collect();

    let criteria = StopCriteria {
        stop_token_ids: stop_ids.clone(),
        stop_strings: stop_strings.clone(),
        max_tokens: input.max_tokens as usize,
        eos_token_id: input.eos,
    };

    // Simulate rolling tail buffer: accumulate chunks and check after each.
    let mut tail = String::new();
    for chunk in input.tail_chunks.iter().take(32) {
        if let Ok(s) = std::str::from_utf8(chunk) {
            tail.push_str(s);
            // Bound tail to prevent OOM.
            if tail.len() > 4096 {
                let drain = tail.len() - 2048;
                tail.drain(..drain);
            }
        }

        let result = check_stop(&criteria, input.current_token, &generated, &tail);

        // Verify stop reason consistency.
        if let Some(ref reason) = result {
            match reason {
                StopReason::StopTokenId(id) => {
                    assert!(
                        stop_ids.contains(id),
                        "StopTokenId {id} not in stop_ids list",
                    );
                }
                StopReason::StopString(s) => {
                    assert!(
                        tail.contains(s.as_str()),
                        "StopString '{s}' not found in tail",
                    );
                }
                StopReason::EosToken => {
                    assert_eq!(
                        input.eos,
                        Some(input.current_token),
                        "EosToken but current_token doesn't match eos",
                    );
                }
                StopReason::MaxTokens => {
                    assert!(
                        criteria.max_tokens > 0 && generated.len() >= criteria.max_tokens,
                        "MaxTokens but budget not exhausted",
                    );
                }
            }
        }
    }

    // Edge: empty tail, empty stops — must return None.
    let empty_criteria = StopCriteria {
        stop_token_ids: vec![],
        stop_strings: vec![],
        max_tokens: 0,
        eos_token_id: None,
    };
    let result = check_stop(&empty_criteria, input.current_token, &generated, "");
    assert!(
        result.is_none(),
        "check_stop must return None with all-empty criteria, got {result:?}",
    );
});

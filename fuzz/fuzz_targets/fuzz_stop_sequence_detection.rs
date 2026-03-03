#![no_main]

use arbitrary::Arbitrary;
use bitnet_generation_stop_core::{StopCriteria, StopReason, check_stop};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct StopSequenceInput {
    stop_token_ids: Vec<u8>,
    stop_strings: Vec<Vec<u8>>,
    max_tokens: u8,
    eos_token_id: Option<u8>,
    generated_tokens: Vec<u8>,
    tail_bytes: Vec<u8>,
}

fuzz_target!(|input: StopSequenceInput| {
    // Build stop criteria from fuzz data.
    let stop_token_ids: Vec<u32> =
        input.stop_token_ids.iter().take(8).map(|&id| id as u32).collect();
    let stop_strings: Vec<String> = input
        .stop_strings
        .iter()
        .take(8)
        .filter_map(|b| {
            let s = String::from_utf8_lossy(&b[..b.len().min(64)]).into_owned();
            if s.is_empty() { None } else { Some(s) }
        })
        .collect();
    let max_tokens = input.max_tokens as usize;
    let eos_token_id = input.eos_token_id.map(|id| id as u32);

    let criteria = StopCriteria {
        stop_token_ids: stop_token_ids.clone(),
        stop_strings: stop_strings.clone(),
        max_tokens,
        eos_token_id,
    };

    let generated: Vec<u32> = input.generated_tokens.iter().take(256).map(|&t| t as u32).collect();
    let decoded_tail =
        String::from_utf8_lossy(&input.tail_bytes[..input.tail_bytes.len().min(512)]);

    // Simulate a generation loop: check_stop at each position.
    for (step, &token_id) in generated.iter().enumerate() {
        let result = check_stop(&criteria, token_id, &generated[..=step], &decoded_tail);

        if let Some(ref reason) = result {
            match reason {
                StopReason::StopTokenId(id) => {
                    assert!(stop_token_ids.contains(id), "StopTokenId({id}) not in stop_token_ids");
                }
                StopReason::EosToken => {
                    assert_eq!(
                        eos_token_id,
                        Some(token_id),
                        "EosToken fired but token doesn't match"
                    );
                }
                StopReason::MaxTokens => {
                    assert!(
                        max_tokens > 0 && generated[..=step].len() >= max_tokens,
                        "MaxTokens fired prematurely"
                    );
                }
                StopReason::StopString(s) => {
                    assert!(
                        decoded_tail.contains(s.as_str()),
                        "StopString({s:?}) not found in tail"
                    );
                }
            }
        }
    }

    // Priority invariant: stop token IDs checked before EOS.
    if !stop_token_ids.is_empty() {
        let probe_id = stop_token_ids[0];
        let criteria_with_eos = StopCriteria {
            stop_token_ids: stop_token_ids.clone(),
            stop_strings: vec![],
            max_tokens: 0,
            eos_token_id: Some(probe_id), // Same ID as a stop token
        };
        let result = check_stop(&criteria_with_eos, probe_id, &[probe_id], "");
        // Should fire as StopTokenId, not EosToken.
        assert_eq!(
            result,
            Some(StopReason::StopTokenId(probe_id)),
            "StopTokenId must take priority over EosToken"
        );
    }

    // Empty criteria must never trigger stop.
    let empty = StopCriteria::default();
    for &token_id in generated.iter().take(16) {
        let result = check_stop(&empty, token_id, &generated, &decoded_tail);
        assert!(result.is_none(), "empty criteria should never stop");
    }
});

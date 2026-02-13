use bitnet_inference::sampling::{SamplingConfig, SamplingStrategy};

#[test]
fn test_robust_incremental_sampling() {
    let config = SamplingConfig {
        repetition_penalty: 2.0, // Strong penalty
        temperature: 0.0, // Greedy
        ..Default::default()
    };
    let mut strategy = SamplingStrategy::new(config);

    // Initial State: Context [1, 2]
    // Logits: 1=10.0, 2=10.0, 3=8.0
    // 1->5.0, 2->5.0, 3->8.0. Winner: 3.
    let mut logits = vec![0.0; 10];
    logits[1] = 10.0;
    logits[2] = 10.0;
    logits[3] = 8.0;

    let token = strategy.sample(&logits, &[1, 2]).unwrap();
    assert_eq!(token, 3, "Standard penalty should apply");

    // Scenario 1: Context Switch (same length, different content)
    // Old Context was effectively [1, 2, 3] (after update).
    // New Context: [1, 4, 5] (Length 3, same as old)
    // Logits: 2=10.0, 4=10.0.
    // If robust: 2 is NOT penalized (was in old context, not new). 4 IS penalized.
    // 2->10.0, 4->5.0. Winner: 2.
    // If buggy (length check only): 2 IS penalized (leftover), 4 is NOT (missed). Winner: 4.

    let mut logits2 = vec![0.0; 10];
    logits2[2] = 10.0;
    logits2[4] = 10.0;

    let token = strategy.sample(&logits2, &[1, 4, 5]).unwrap();
    assert_eq!(token, 2, "Context switch should clear old penalties (2) and apply new ones (4)");

    // Scenario 2: Backtracking (Context shrinks)
    // Current Context (after update): [1, 4, 5, 2]
    // New Context: [1, 4] (Backtracked)
    // Logits: 5=10.0, 6=8.0
    // If robust: 5 is NOT penalized (removed). 5->10.0. Winner: 5.
    // If buggy: 5 IS penalized (leftover). 5->5.0. Winner: 6.

    let mut logits3 = vec![0.0; 10];
    logits3[5] = 10.0;
    logits3[6] = 8.0;

    let token = strategy.sample(&logits3, &[1, 4]).unwrap();
    assert_eq!(token, 5, "Backtracking should remove penalties for dropped tokens");
}

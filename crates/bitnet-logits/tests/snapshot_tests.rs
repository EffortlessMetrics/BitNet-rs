//! Snapshot tests for `bitnet-logits` public API surface.
//!
//! Pins the output of logit transform functions for well-known inputs
//! so that numerical regressions are caught at review time.

use bitnet_logits::{apply_temperature, argmax, softmax_in_place};

#[test]
fn argmax_simple_snapshot() {
    let logits = vec![0.1f32, 0.5, 0.9, 0.3];
    let result = argmax(&logits);
    insta::assert_snapshot!("argmax_simple", result.to_string());
}

#[test]
fn argmax_last_wins_on_tie() {
    // bitnet-logits argmax uses max_by which returns the last maximum on ties.
    let logits = vec![1.0f32, 1.0, 0.5];
    let result = argmax(&logits);
    // Last occurrence of max value (1.0) is at index 1
    insta::assert_snapshot!("argmax_last_on_tie", result.to_string());
}

#[test]
fn softmax_uniform_snapshot() {
    let mut logits = vec![1.0f32, 1.0, 1.0, 1.0];
    softmax_in_place(&mut logits);
    // All equal logits → each probability should be 0.25
    let rounded: Vec<f32> = logits.iter().map(|x| (x * 1000.0).round() / 1000.0).collect();
    insta::assert_debug_snapshot!("softmax_uniform_4", rounded);
}

#[test]
fn temperature_identity_at_one() {
    let original = vec![1.0f32, 2.0, 3.0];
    let mut logits = original.clone();
    apply_temperature(&mut logits, 1.0);
    // Temperature=1.0 is a no-op
    assert_eq!(logits, original);
    insta::assert_snapshot!("temperature_1_is_noop", "unchanged");
}

#[test]
fn temperature_doubles_logits_at_half() {
    let mut logits = vec![2.0f32, 4.0, 6.0];
    apply_temperature(&mut logits, 0.5);
    let rounded: Vec<f32> = logits.iter().map(|x| (x * 10.0).round() / 10.0).collect();
    insta::assert_debug_snapshot!("temperature_0_5_doubles", rounded);
}

//! Wave 33 snapshot tests for bitnet-logits.
//!
//! Covers: logit processor chain display, filter descriptions,
//! sampler config output, numerical transform results.

use bitnet_logits::{
    apply_min_p, apply_repetition_penalty, apply_temperature, apply_top_k, apply_top_p,
    apply_typical, argmax, softmax_in_place,
};

// ── Temperature scaling ─────────────────────────────────────────────────────

#[test]
fn w33_temperature_zero_is_noop() {
    let original = vec![1.0f32, 2.0, 3.0];
    let mut logits = original.clone();
    apply_temperature(&mut logits, 0.0);
    let rounded: Vec<f32> = logits.iter().map(|x| (x * 1000.0).round() / 1000.0).collect();
    insta::assert_debug_snapshot!(rounded);
}

#[test]
fn w33_temperature_low_sharpens() {
    let mut logits = vec![1.0f32, 2.0, 3.0, 4.0];
    apply_temperature(&mut logits, 0.1);
    let rounded: Vec<f32> = logits.iter().map(|x| (x * 100.0).round() / 100.0).collect();
    insta::assert_debug_snapshot!(rounded);
}

#[test]
fn w33_temperature_high_flattens() {
    let mut logits = vec![1.0f32, 2.0, 3.0, 4.0];
    apply_temperature(&mut logits, 10.0);
    let rounded: Vec<f32> = logits.iter().map(|x| (x * 1000.0).round() / 1000.0).collect();
    insta::assert_debug_snapshot!(rounded);
}

// ── Softmax ─────────────────────────────────────────────────────────────────

#[test]
fn w33_softmax_simple_3() {
    let mut logits = vec![1.0f32, 2.0, 3.0];
    softmax_in_place(&mut logits);
    let rounded: Vec<f32> = logits.iter().map(|x| (x * 10000.0).round() / 10000.0).collect();
    insta::assert_debug_snapshot!(rounded);
}

#[test]
fn w33_softmax_with_negative() {
    let mut logits = vec![-1.0f32, 0.0, 1.0, 2.0];
    softmax_in_place(&mut logits);
    let rounded: Vec<f32> = logits.iter().map(|x| (x * 10000.0).round() / 10000.0).collect();
    insta::assert_debug_snapshot!(rounded);
}

#[test]
fn w33_softmax_single_element() {
    let mut logits = vec![5.0f32];
    softmax_in_place(&mut logits);
    let rounded: Vec<f32> = logits.iter().map(|x| (x * 1000.0).round() / 1000.0).collect();
    insta::assert_debug_snapshot!(rounded);
}

#[test]
fn w33_softmax_after_neg_inf() {
    let mut logits = vec![f32::NEG_INFINITY, 1.0, f32::NEG_INFINITY, 2.0];
    softmax_in_place(&mut logits);
    let rounded: Vec<f32> = logits.iter().map(|x| (x * 10000.0).round() / 10000.0).collect();
    insta::assert_debug_snapshot!(rounded);
}

// ── Top-k ───────────────────────────────────────────────────────────────────

#[test]
fn w33_top_k_3_of_5() {
    let mut logits = vec![1.0f32, 5.0, 3.0, 2.0, 4.0];
    let kept = apply_top_k(&mut logits, 3);
    let finite_mask: Vec<bool> = logits.iter().map(|x| x.is_finite()).collect();
    insta::assert_snapshot!(format!("kept={kept} finite={finite_mask:?}"));
}

#[test]
fn w33_top_k_1_of_4() {
    let mut logits = vec![0.1f32, 0.5, 0.9, 0.3];
    let kept = apply_top_k(&mut logits, 1);
    let finite_mask: Vec<bool> = logits.iter().map(|x| x.is_finite()).collect();
    insta::assert_snapshot!(format!("kept={kept} finite={finite_mask:?}"));
}

// ── Top-p ───────────────────────────────────────────────────────────────────

#[test]
fn w33_top_p_low_threshold() {
    let mut probs = vec![0.5f32, 0.3, 0.15, 0.05];
    apply_top_p(&mut probs, 0.5);
    let rounded: Vec<f32> = probs.iter().map(|x| (x * 1000.0).round() / 1000.0).collect();
    insta::assert_debug_snapshot!(rounded);
}

#[test]
fn w33_top_p_high_threshold() {
    let mut probs = vec![0.5f32, 0.3, 0.15, 0.05];
    apply_top_p(&mut probs, 0.95);
    let rounded: Vec<f32> = probs.iter().map(|x| (x * 1000.0).round() / 1000.0).collect();
    insta::assert_debug_snapshot!(rounded);
}

// ── Min-p ───────────────────────────────────────────────────────────────────

#[test]
fn w33_min_p_moderate() {
    let mut probs = vec![0.5f32, 0.3, 0.1, 0.05, 0.05];
    apply_min_p(&mut probs, 0.2);
    let rounded: Vec<f32> = probs.iter().map(|x| (x * 1000.0).round() / 1000.0).collect();
    insta::assert_debug_snapshot!(rounded);
}

#[test]
fn w33_min_p_aggressive() {
    let mut probs = vec![0.6f32, 0.2, 0.1, 0.05, 0.05];
    apply_min_p(&mut probs, 0.5);
    let rounded: Vec<f32> = probs.iter().map(|x| (x * 1000.0).round() / 1000.0).collect();
    insta::assert_debug_snapshot!(rounded);
}

// ── Typical ─────────────────────────────────────────────────────────────────

#[test]
fn w33_typical_moderate() {
    let mut probs = vec![0.4f32, 0.3, 0.15, 0.1, 0.05];
    apply_typical(&mut probs, 0.6);
    let non_zero: usize = probs.iter().filter(|&&p| p > 0.0).count();
    let sum: f32 = probs.iter().sum();
    insta::assert_snapshot!(format!("non_zero={non_zero} sum={:.4}", sum));
}

// ── Repetition penalty ──────────────────────────────────────────────────────

#[test]
fn w33_repetition_penalty_mixed_logits() {
    let mut logits = vec![3.0f32, -2.0, 1.0, 0.0, -0.5];
    apply_repetition_penalty(&mut logits, &[0, 1, 4], 1.5);
    let rounded: Vec<f32> = logits.iter().map(|x| (x * 1000.0).round() / 1000.0).collect();
    insta::assert_debug_snapshot!(rounded);
}

#[test]
fn w33_repetition_penalty_no_history() {
    let original = vec![1.0f32, 2.0, 3.0];
    let mut logits = original.clone();
    apply_repetition_penalty(&mut logits, &[], 2.0);
    let rounded: Vec<f32> = logits.iter().map(|x| (x * 1000.0).round() / 1000.0).collect();
    insta::assert_debug_snapshot!(rounded);
}

// ── Argmax ──────────────────────────────────────────────────────────────────

#[test]
fn w33_argmax_varied_logits() {
    let logits = vec![0.1f32, 0.3, 0.8, 0.2, 0.7];
    let result = argmax(&logits);
    insta::assert_snapshot!(result.to_string());
}

#[test]
fn w33_argmax_all_same() {
    let logits = vec![1.0f32, 1.0, 1.0, 1.0];
    let result = argmax(&logits);
    insta::assert_snapshot!(result.to_string());
}

#[test]
fn w33_argmax_single() {
    let logits = vec![42.0f32];
    let result = argmax(&logits);
    insta::assert_snapshot!(result.to_string());
}

// ── Full pipeline snapshots ─────────────────────────────────────────────────

#[test]
fn w33_full_pipeline_greedy() {
    let mut logits = vec![1.0f32, 3.0, 2.0, 0.5];
    apply_temperature(&mut logits, 1.0);
    let best = argmax(&logits);
    insta::assert_snapshot!(format!("greedy_pick={best}"));
}

#[test]
fn w33_full_pipeline_with_penalty_and_softmax() {
    let mut logits = vec![2.0f32, 4.0, 1.0, 3.0];
    apply_repetition_penalty(&mut logits, &[1], 2.0);
    apply_temperature(&mut logits, 0.8);
    softmax_in_place(&mut logits);
    let rounded: Vec<f32> = logits.iter().map(|x| (x * 10000.0).round() / 10000.0).collect();
    insta::assert_debug_snapshot!(rounded);
}

#[test]
fn w33_full_pipeline_top_k_then_softmax() {
    let mut logits = vec![1.0f32, 5.0, 3.0, 2.0, 4.0];
    apply_top_k(&mut logits, 2);
    softmax_in_place(&mut logits);
    let rounded: Vec<f32> = logits.iter().map(|x| (x * 10000.0).round() / 10000.0).collect();
    insta::assert_debug_snapshot!(rounded);
}

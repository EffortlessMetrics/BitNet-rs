//! Snapshot tests for `bitnet-validation` public API surface.
//!
//! Pins the ruleset detection output so that changes to default envelopes
//! are visible in code review.

use bitnet_validation::rules::{detect_rules, rules_bitnet_b158_f16, rules_generic};

#[test]
fn detect_rules_bitnet_f16_name() {
    let ruleset = detect_rules("bitnet", 1);
    insta::assert_debug_snapshot!("detect_rules_bitnet_f16_name", ruleset.name);
}

#[test]
fn detect_rules_generic_name() {
    let ruleset = detect_rules("unknown_arch", 0);
    insta::assert_debug_snapshot!("detect_rules_generic_name", ruleset.name);
}

#[test]
fn bitnet_f16_ruleset_has_thresholds() {
    let ruleset = rules_bitnet_b158_f16();
    // Snapshot the count and name — structural change to envelopes is flagged.
    let summary = format!("name={} ln_rules={}", ruleset.name, ruleset.ln.len());
    insta::assert_snapshot!("bitnet_f16_ruleset_summary", summary);
}

#[test]
fn generic_ruleset_name_snapshot() {
    let ruleset = rules_generic();
    insta::assert_snapshot!("generic_ruleset_name", ruleset.name);
}

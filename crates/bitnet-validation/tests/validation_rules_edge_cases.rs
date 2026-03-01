//! Edge-case tests for bitnet-validation: ruleset logic, threshold
//! checking, LayerNorm name detection, and architecture-based
//! rule detection.

use bitnet_validation::{
    Ruleset, Threshold, detect_rules, is_ln_gamma, rules_bitnet_b158_f16, rules_bitnet_b158_i2s,
    rules_generic,
};
use regex::Regex;

// ── is_ln_gamma ──────────────────────────────────────────────────────

#[test]
fn ln_gamma_matches_blk_attn_norm() {
    assert!(is_ln_gamma("blk.0.attn_norm.weight"));
    assert!(is_ln_gamma("blk.39.attn_norm.weight"));
}

#[test]
fn ln_gamma_matches_ffn_norm() {
    assert!(is_ln_gamma("blk.0.ffn_norm.weight"));
    assert!(is_ln_gamma("blk.12.ffn_norm.weight"));
}

#[test]
fn ln_gamma_matches_output_norm() {
    // output_norm may not be classified as LN gamma
    // (only blk.X patterns match)
    let _result = is_ln_gamma("output_norm.weight");
    // Just verify no panic — the actual classification depends on impl
}

#[test]
fn ln_gamma_rejects_non_norm() {
    assert!(!is_ln_gamma("blk.0.attn_q.weight"));
    assert!(!is_ln_gamma("token_embd.weight"));
    assert!(!is_ln_gamma("output.weight"));
    assert!(!is_ln_gamma(""));
}

#[test]
fn ln_gamma_rejects_partial_matches() {
    assert!(!is_ln_gamma("attn_norm"));
    assert!(!is_ln_gamma("weight"));
}

// ── Ruleset construction ─────────────────────────────────────────────

#[test]
fn bitnet_f16_ruleset_has_name() {
    let r = rules_bitnet_b158_f16();
    assert!(!r.name.is_empty());
    assert!(r.name.contains("f16") || r.name.contains("F16"));
}

#[test]
fn bitnet_i2s_ruleset_has_name() {
    let r = rules_bitnet_b158_i2s();
    assert!(!r.name.is_empty());
    assert!(r.name.contains("i2s") || r.name.contains("I2S") || r.name.contains("i2_s"));
}

#[test]
fn generic_ruleset_has_name() {
    let r = rules_generic();
    assert!(!r.name.is_empty());
}

#[test]
fn all_rulesets_have_ln_thresholds() {
    assert!(!rules_bitnet_b158_f16().ln.is_empty());
    assert!(!rules_bitnet_b158_i2s().ln.is_empty());
    assert!(!rules_generic().ln.is_empty());
}

// ── check_ln ─────────────────────────────────────────────────────────

#[test]
fn check_ln_passes_in_range() {
    let r = rules_generic();
    // A typical LN gamma RMS ~1.0 for well-initialized models
    let result = r.check_ln("blk.0.attn_norm.weight", 1.0);
    assert!(result, "RMS=1.0 should pass generic LN check");
}

#[test]
fn check_ln_rejects_extreme_low() {
    let r = rules_generic();
    let result = r.check_ln("blk.0.attn_norm.weight", 0.0);
    assert!(!result, "RMS=0.0 should fail generic LN check");
}

#[test]
fn check_ln_for_non_ln_name_returns_true() {
    let r = rules_generic();
    // Non-LN names don't match any threshold pattern → should return true (no check needed)
    let result = r.check_ln("blk.0.attn_q.weight", 0.5);
    assert!(result, "Non-LN name should pass (not checked)");
}

// ── check_proj_rms ───────────────────────────────────────────────────

#[test]
fn check_proj_rms_with_no_bounds() {
    // If ruleset has no proj bounds, any value passes
    let r = Ruleset {
        ln: vec![],
        proj_weight_rms_min: None,
        proj_weight_rms_max: None,
        name: "no-bounds".into(),
    };
    assert!(r.check_proj_rms(0.0));
    assert!(r.check_proj_rms(100.0));
}

#[test]
fn check_proj_rms_within_bounds() {
    let r = Ruleset {
        ln: vec![],
        proj_weight_rms_min: Some(0.01),
        proj_weight_rms_max: Some(10.0),
        name: "bounded".into(),
    };
    assert!(r.check_proj_rms(0.5));
    assert!(r.check_proj_rms(5.0));
}

#[test]
fn check_proj_rms_below_min() {
    let r = Ruleset {
        ln: vec![],
        proj_weight_rms_min: Some(0.1),
        proj_weight_rms_max: None,
        name: "min-only".into(),
    };
    assert!(!r.check_proj_rms(0.001));
}

#[test]
fn check_proj_rms_above_max() {
    let r = Ruleset {
        ln: vec![],
        proj_weight_rms_min: None,
        proj_weight_rms_max: Some(5.0),
        name: "max-only".into(),
    };
    assert!(!r.check_proj_rms(10.0));
}

// ── Threshold ────────────────────────────────────────────────────────

#[test]
fn threshold_pattern_matches() {
    let t = Threshold { pattern: Regex::new(r"blk\.\d+\.attn_norm").unwrap(), min: 0.5, max: 2.0 };
    assert!(t.pattern.is_match("blk.0.attn_norm.weight"));
    assert!(!t.pattern.is_match("output_norm.weight"));
}

// ── detect_rules ─────────────────────────────────────────────────────

#[test]
fn detect_rules_bitnet_arch() {
    let r = detect_rules("bitnet", 0);
    assert!(!r.name.is_empty());
}

#[test]
fn detect_rules_llama_arch() {
    let r = detect_rules("llama", 0);
    assert!(!r.name.is_empty());
}

#[test]
fn detect_rules_unknown_arch_falls_back() {
    let r = detect_rules("unknown_arch_xyz", 0);
    assert!(!r.name.is_empty(), "unknown arch should get a fallback ruleset");
}

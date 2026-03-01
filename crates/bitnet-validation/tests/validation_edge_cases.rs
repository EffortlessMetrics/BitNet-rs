//! Edge-case tests for bitnet-validation: LayerNorm name detection,
//! ruleset auto-detection, threshold checking, projection RMS validation.

use bitnet_validation::{
    detect_rules, is_ln_gamma, rules_bitnet_b158_f16, rules_bitnet_b158_i2s, rules_generic,
};

// ---------------------------------------------------------------------------
// is_ln_gamma — positive matches
// ---------------------------------------------------------------------------

#[test]
fn ln_gamma_attn_norm() {
    assert!(is_ln_gamma("blk.0.attn_norm.weight"));
    assert!(is_ln_gamma("blk.29.attn_norm.weight"));
}

#[test]
fn ln_gamma_ffn_norm() {
    assert!(is_ln_gamma("blk.0.ffn_norm.weight"));
}

#[test]
fn ln_gamma_ffn_layernorm() {
    assert!(is_ln_gamma("blk.3.ffn_layernorm.weight"));
}

#[test]
fn ln_gamma_input_layernorm() {
    assert!(is_ln_gamma("model.layers.0.input_layernorm.weight"));
}

#[test]
fn ln_gamma_post_attention_layernorm() {
    assert!(is_ln_gamma("model.layers.0.post_attention_layernorm.weight"));
}

#[test]
fn ln_gamma_final_norm() {
    assert!(is_ln_gamma("final_norm.weight"));
}

#[test]
fn ln_gamma_final_layernorm() {
    assert!(is_ln_gamma("final_layernorm.weight"));
}

#[test]
fn ln_gamma_rms_norm() {
    assert!(is_ln_gamma("blk.0.rms_norm.weight"));
}

#[test]
fn ln_gamma_bare_norm() {
    assert!(is_ln_gamma("norm.weight"));
}

// ---------------------------------------------------------------------------
// is_ln_gamma — negative matches
// ---------------------------------------------------------------------------

#[test]
fn ln_gamma_not_attn_q() {
    assert!(!is_ln_gamma("blk.0.attn_q.weight"));
}

#[test]
fn ln_gamma_not_output() {
    assert!(!is_ln_gamma("output.weight"));
}

#[test]
fn ln_gamma_not_embedding() {
    assert!(!is_ln_gamma("token_embd.weight"));
}

#[test]
fn ln_gamma_not_ffn_up() {
    assert!(!is_ln_gamma("blk.0.ffn_up.weight"));
}

#[test]
fn ln_gamma_not_bias() {
    // Must end with .weight, not .bias
    assert!(!is_ln_gamma("blk.0.attn_norm.bias"));
}

#[test]
fn ln_gamma_empty_string() {
    assert!(!is_ln_gamma(""));
}

#[test]
fn ln_gamma_just_weight() {
    assert!(!is_ln_gamma(".weight"));
}

// ---------------------------------------------------------------------------
// detect_rules
// ---------------------------------------------------------------------------

#[test]
fn detect_bitnet_f16() {
    let r = detect_rules("bitnet", 1);
    assert_eq!(r.name, "bitnet-b1.58:f16");
}

#[test]
fn detect_bitnet_i2s() {
    let r = detect_rules("bitnet", 2);
    assert_eq!(r.name, "bitnet-b1.58:i2_s");
}

#[test]
fn detect_bitnet_case_insensitive() {
    let r = detect_rules("BitNet", 1);
    assert_eq!(r.name, "bitnet-b1.58:f16");
}

#[test]
fn detect_b158_variant() {
    let r = detect_rules("b1.58-custom", 1);
    assert_eq!(r.name, "bitnet-b1.58:f16");
}

#[test]
fn detect_generic_llama() {
    let r = detect_rules("llama", 0);
    assert_eq!(r.name, "generic");
}

#[test]
fn detect_generic_gpt2() {
    let r = detect_rules("gpt2", 0);
    assert_eq!(r.name, "generic");
}

#[test]
fn detect_generic_unknown() {
    let r = detect_rules("unknown_arch", 99);
    assert_eq!(r.name, "generic");
}

// ---------------------------------------------------------------------------
// Ruleset — check_ln
// ---------------------------------------------------------------------------

#[test]
fn f16_check_ln_attn_norm_valid() {
    let r = rules_bitnet_b158_f16();
    assert!(r.check_ln("blk.0.attn_norm.weight", 0.8));
    assert!(r.check_ln("blk.0.attn_norm.weight", 1.5));
}

#[test]
fn f16_check_ln_ffn_layernorm_low_valid() {
    let r = rules_bitnet_b158_f16();
    // ffn_layernorm can have legitimately low RMS (~0.05-0.10)
    assert!(r.check_ln("blk.0.ffn_layernorm.weight", 0.06));
}

#[test]
fn f16_check_ln_ffn_layernorm_too_low() {
    let r = rules_bitnet_b158_f16();
    assert!(!r.check_ln("blk.0.ffn_layernorm.weight", 0.01));
}

#[test]
fn f16_check_ln_final_norm_valid() {
    let r = rules_bitnet_b158_f16();
    assert!(r.check_ln("final_norm.weight", 1.0));
}

#[test]
fn f16_check_ln_too_high() {
    let r = rules_bitnet_b158_f16();
    assert!(!r.check_ln("blk.0.attn_norm.weight", 5.0));
}

#[test]
fn i2s_check_ln_attn_norm_very_low_valid() {
    let r = rules_bitnet_b158_i2s();
    // I2S attn_norm can be very low (0.01-0.02)
    assert!(r.check_ln("blk.0.attn_norm.weight", 0.015));
}

#[test]
fn i2s_check_ln_attn_norm_too_low() {
    let r = rules_bitnet_b158_i2s();
    assert!(!r.check_ln("blk.0.attn_norm.weight", 0.005));
}

#[test]
fn generic_check_ln_near_one() {
    let r = rules_generic();
    assert!(r.check_ln("blk.0.norm.weight", 1.0));
}

#[test]
fn generic_check_ln_outside_range() {
    let r = rules_generic();
    assert!(!r.check_ln("blk.0.norm.weight", 0.5));
}

#[test]
fn check_ln_unknown_name_falls_to_generic_envelope() {
    let r = rules_bitnet_b158_f16();
    // Name doesn't match any specific pattern → generic 0.50..=2.0
    assert!(r.check_ln("some.unknown.tensor", 1.0));
    assert!(!r.check_ln("some.unknown.tensor", 0.3));
}

// ---------------------------------------------------------------------------
// Ruleset — check_proj_rms
// ---------------------------------------------------------------------------

#[test]
fn f16_proj_rms_valid() {
    let r = rules_bitnet_b158_f16();
    assert!(r.check_proj_rms(0.1));
    assert!(r.check_proj_rms(0.3));
}

#[test]
fn f16_proj_rms_too_low() {
    let r = rules_bitnet_b158_f16();
    assert!(!r.check_proj_rms(0.005));
}

#[test]
fn f16_proj_rms_too_high() {
    let r = rules_bitnet_b158_f16();
    assert!(!r.check_proj_rms(0.5));
}

#[test]
fn generic_proj_rms_no_limits() {
    let r = rules_generic();
    // Generic has no proj_weight_rms limits
    assert!(r.check_proj_rms(0.001));
    assert!(r.check_proj_rms(100.0));
}

// ---------------------------------------------------------------------------
// Ruleset properties
// ---------------------------------------------------------------------------

#[test]
fn rulesets_have_non_empty_names() {
    assert!(!rules_bitnet_b158_f16().name.is_empty());
    assert!(!rules_bitnet_b158_i2s().name.is_empty());
    assert!(!rules_generic().name.is_empty());
}

#[test]
fn rulesets_have_ln_rules() {
    assert!(!rules_bitnet_b158_f16().ln.is_empty());
    assert!(!rules_bitnet_b158_i2s().ln.is_empty());
    assert!(!rules_generic().ln.is_empty());
}

#[test]
fn ruleset_clone() {
    let r = rules_bitnet_b158_f16();
    let c = r.clone();
    assert_eq!(c.name, r.name);
    assert_eq!(c.ln.len(), r.ln.len());
}

#[test]
fn ruleset_debug() {
    let r = rules_bitnet_b158_f16();
    let dbg = format!("{r:?}");
    assert!(dbg.contains("bitnet-b1.58:f16"));
}

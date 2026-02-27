// SPDX-License-Identifier: MIT OR Apache-2.0
//! Comprehensive integration tests for `bitnet-validation`.
//!
//! Covers:
//! - LayerNorm name detection (`is_ln_gamma`)
//! - Ruleset construction and threshold checks
//! - Architecture detection via `detect_rules`
//! - Policy gate selection and YAML loading
//! - Edge cases for `Ruleset::default()` (no-rules fallback)

use bitnet_validation::{
    Ruleset, detect_rules, is_ln_gamma, load_policy, rules_bitnet_b158_f16, rules_bitnet_b158_i2s,
    rules_generic,
};
use tempfile::NamedTempFile;

// ── is_ln_gamma name detection ───────────────────────────────────────────────

#[test]
fn is_ln_gamma_attn_norm() {
    assert!(is_ln_gamma("blk.0.attn_norm.weight"));
}

#[test]
fn is_ln_gamma_ffn_norm() {
    assert!(is_ln_gamma("blk.7.ffn_norm.weight"));
}

#[test]
fn is_ln_gamma_ffn_layernorm() {
    assert!(is_ln_gamma("blk.3.ffn_layernorm.weight"));
}

#[test]
fn is_ln_gamma_rms_norm() {
    assert!(is_ln_gamma("blk.0.rms_norm.weight"));
}

#[test]
fn is_ln_gamma_input_layernorm() {
    assert!(is_ln_gamma("model.layers.5.input_layernorm.weight"));
}

#[test]
fn is_ln_gamma_post_attention_layernorm() {
    assert!(is_ln_gamma("model.layers.2.post_attention_layernorm.weight"));
}

#[test]
fn is_ln_gamma_final_norm() {
    assert!(is_ln_gamma("final_norm.weight"));
}

#[test]
fn is_ln_gamma_final_layernorm() {
    assert!(is_ln_gamma("model.final_layernorm.weight"));
}

#[test]
fn is_ln_gamma_bare_norm_weight() {
    assert!(is_ln_gamma("model.norm.weight"));
}

#[test]
fn is_ln_gamma_deep_block_path() {
    assert!(is_ln_gamma("transformer.layers.31.attn_norm.weight"));
}

#[test]
fn is_ln_gamma_rejects_projection_weight() {
    assert!(!is_ln_gamma("blk.0.attn_q.weight"));
    assert!(!is_ln_gamma("blk.0.ffn_up.weight"));
    assert!(!is_ln_gamma("blk.0.ffn_down.weight"));
    assert!(!is_ln_gamma("blk.0.attn_v.weight"));
}

#[test]
fn is_ln_gamma_rejects_output_weight() {
    assert!(!is_ln_gamma("output.weight"));
}

#[test]
fn is_ln_gamma_rejects_token_embedding() {
    assert!(!is_ln_gamma("token_embd.weight"));
}

#[test]
fn is_ln_gamma_rejects_missing_weight_suffix() {
    assert!(!is_ln_gamma("blk.0.attn_norm"));
    assert!(!is_ln_gamma("blk.0.attn_norm.bias"));
}

#[test]
fn is_ln_gamma_rejects_empty_string() {
    assert!(!is_ln_gamma(""));
}

#[test]
fn is_ln_gamma_is_deterministic() {
    let name = "blk.10.ffn_norm.weight";
    assert_eq!(is_ln_gamma(name), is_ln_gamma(name));
}

// ── Architecture detection ───────────────────────────────────────────────────

#[test]
fn detect_rules_bitnet_f16_gives_f16_ruleset() {
    let r = detect_rules("bitnet", 1);
    assert_eq!(r.name, "bitnet-b1.58:f16");
}

#[test]
fn detect_rules_bitnet_quantized_gives_i2s_ruleset() {
    let r = detect_rules("bitnet", 2);
    assert_eq!(r.name, "bitnet-b1.58:i2_s");
}

#[test]
fn detect_rules_b158_in_arch_gives_bitnet_ruleset() {
    let r = detect_rules("b1.58-llama", 1);
    assert_eq!(r.name, "bitnet-b1.58:f16");
}

#[test]
fn detect_rules_case_insensitive_bitnet_upper() {
    let r = detect_rules("BITNET", 1);
    assert_eq!(r.name, "bitnet-b1.58:f16");
}

#[test]
fn detect_rules_case_insensitive_mixed() {
    let r = detect_rules("BitNet-b1.58", 0);
    assert_eq!(r.name, "bitnet-b1.58:i2_s");
}

#[test]
fn detect_rules_llama_gives_generic() {
    let r = detect_rules("llama", 0);
    assert_eq!(r.name, "generic");
}

#[test]
fn detect_rules_unknown_arch_gives_generic() {
    let r = detect_rules("mistral", 1);
    assert_eq!(r.name, "generic");
}

#[test]
fn detect_rules_empty_arch_gives_generic() {
    let r = detect_rules("", 0);
    assert_eq!(r.name, "generic");
}

#[test]
fn detect_rules_never_returns_empty_name() {
    for arch in ["bitnet", "llama", "gpt", "mistral", "", "unknown"] {
        for ft in [0u32, 1, 2, 10] {
            let r = detect_rules(arch, ft);
            assert!(!r.name.is_empty(), "empty name for arch={arch:?} ft={ft}");
        }
    }
}

// ── F16 ruleset thresholds ────────────────────────────────────────────────────

#[test]
fn f16_ruleset_name_is_correct() {
    assert_eq!(rules_bitnet_b158_f16().name, "bitnet-b1.58:f16");
}

#[test]
fn f16_check_ln_attn_norm_accepts_valid_rms() {
    let r = rules_bitnet_b158_f16();
    // attn_norm falls through to ".*norm\.weight$" catch-all: min=0.50, max=2.0
    assert!(r.check_ln("blk.0.attn_norm.weight", 0.8));
    assert!(r.check_ln("blk.0.attn_norm.weight", 0.5));
    assert!(r.check_ln("blk.0.attn_norm.weight", 2.0));
}

#[test]
fn f16_check_ln_attn_norm_rejects_below_min() {
    let r = rules_bitnet_b158_f16();
    assert!(!r.check_ln("blk.0.attn_norm.weight", 0.49));
}

#[test]
fn f16_check_ln_ffn_layernorm_accepts_low_rms() {
    // ffn_layernorm has min=0.05 in F16 ruleset
    let r = rules_bitnet_b158_f16();
    assert!(r.check_ln("blk.0.ffn_layernorm.weight", 0.07));
}

#[test]
fn f16_check_ln_ffn_layernorm_rejects_below_min() {
    let r = rules_bitnet_b158_f16();
    assert!(!r.check_ln("blk.0.ffn_layernorm.weight", 0.01));
}

#[test]
fn f16_check_ln_post_attention_layernorm_accepts_valid() {
    let r = rules_bitnet_b158_f16();
    assert!(r.check_ln("model.layers.0.post_attention_layernorm.weight", 0.5));
}

#[test]
fn f16_check_ln_post_attention_layernorm_rejects_too_low() {
    let r = rules_bitnet_b158_f16();
    assert!(!r.check_ln("model.layers.0.post_attention_layernorm.weight", 0.1));
}

#[test]
fn f16_check_ln_input_layernorm_accepts_valid() {
    let r = rules_bitnet_b158_f16();
    assert!(r.check_ln("model.layers.3.input_layernorm.weight", 0.6));
}

#[test]
fn f16_check_ln_final_norm_accepts_valid() {
    let r = rules_bitnet_b158_f16();
    assert!(r.check_ln("final_norm.weight", 1.0));
}

#[test]
fn f16_check_ln_unrecognized_name_falls_back_to_generic_envelope() {
    let r = rules_bitnet_b158_f16();
    // Name does not end in "norm.weight" → fallback generic: 0.50..=2.0
    assert!(r.check_ln("blk.0.attn_q.weight", 0.9));
    assert!(!r.check_ln("blk.0.attn_q.weight", 2.5));
    assert!(!r.check_ln("blk.0.attn_q.weight", 0.1));
}

#[test]
fn f16_check_proj_rms_accepts_in_range() {
    let r = rules_bitnet_b158_f16();
    assert!(r.check_proj_rms(0.01)); // exactly at min
    assert!(r.check_proj_rms(0.40)); // exactly at max
    assert!(r.check_proj_rms(0.15)); // mid-range
}

#[test]
fn f16_check_proj_rms_rejects_below_min() {
    let r = rules_bitnet_b158_f16();
    assert!(!r.check_proj_rms(0.009));
    assert!(!r.check_proj_rms(0.0));
}

#[test]
fn f16_check_proj_rms_rejects_above_max() {
    let r = rules_bitnet_b158_f16();
    assert!(!r.check_proj_rms(0.401));
    assert!(!r.check_proj_rms(1.0));
}

// ── I2_S ruleset thresholds ───────────────────────────────────────────────────

#[test]
fn i2s_ruleset_name_is_correct() {
    assert_eq!(rules_bitnet_b158_i2s().name, "bitnet-b1.58:i2_s");
}

#[test]
fn i2s_check_ln_attn_norm_accepts_very_low_rms() {
    // I2_S attn_norm has min=0.01 (looser after quantization)
    let r = rules_bitnet_b158_i2s();
    assert!(r.check_ln("blk.0.attn_norm.weight", 0.01));
    assert!(r.check_ln("blk.0.attn_norm.weight", 0.015));
}

#[test]
fn i2s_check_ln_attn_norm_rejects_below_min() {
    let r = rules_bitnet_b158_i2s();
    assert!(!r.check_ln("blk.0.attn_norm.weight", 0.005));
}

#[test]
fn i2s_check_ln_ffn_norm_accepts_valid() {
    let r = rules_bitnet_b158_i2s();
    assert!(r.check_ln("blk.2.ffn_norm.weight", 0.6));
}

#[test]
fn i2s_check_ln_ffn_norm_rejects_below_min() {
    // I2_S ffn_norm has min=0.50
    let r = rules_bitnet_b158_i2s();
    assert!(!r.check_ln("blk.2.ffn_norm.weight", 0.1));
}

#[test]
fn i2s_check_proj_rms_accepts_very_low_rms() {
    // I2_S proj_weight_rms_min=0.002 (much lower than F16's 0.01)
    let r = rules_bitnet_b158_i2s();
    assert!(r.check_proj_rms(0.002));
    assert!(r.check_proj_rms(0.01));
    assert!(r.check_proj_rms(0.15));
}

#[test]
fn i2s_check_proj_rms_rejects_too_low() {
    let r = rules_bitnet_b158_i2s();
    assert!(!r.check_proj_rms(0.001));
}

#[test]
fn i2s_check_proj_rms_rejects_too_high() {
    // I2_S proj_weight_rms_max=0.20
    let r = rules_bitnet_b158_i2s();
    assert!(!r.check_proj_rms(0.21));
}

// ── Generic ruleset ───────────────────────────────────────────────────────────

#[test]
fn generic_ruleset_name_is_correct() {
    assert_eq!(rules_generic().name, "generic");
}

#[test]
fn generic_check_ln_accepts_near_one() {
    let r = rules_generic();
    assert!(r.check_ln("blk.0.attn_norm.weight", 0.90));
    assert!(r.check_ln("blk.0.attn_norm.weight", 1.00));
    assert!(r.check_ln("blk.0.attn_norm.weight", 1.10));
}

#[test]
fn generic_check_ln_rejects_far_from_one() {
    let r = rules_generic();
    assert!(!r.check_ln("blk.0.attn_norm.weight", 0.5));
    assert!(!r.check_ln("blk.0.attn_norm.weight", 1.5));
}

#[test]
fn generic_check_proj_rms_no_bounds_accepts_any_value() {
    let r = rules_generic();
    // Generic ruleset has no proj thresholds
    assert!(r.proj_weight_rms_min.is_none());
    assert!(r.proj_weight_rms_max.is_none());
    assert!(r.check_proj_rms(0.0));
    assert!(r.check_proj_rms(1.0));
    assert!(r.check_proj_rms(100.0));
    assert!(r.check_proj_rms(f32::MAX));
}

// ── Ruleset::default() (empty ruleset) ───────────────────────────────────────

#[test]
fn default_ruleset_has_no_ln_rules() {
    let r = Ruleset::default();
    assert!(r.ln.is_empty());
}

#[test]
fn default_ruleset_check_ln_uses_generic_fallback_envelope() {
    let r = Ruleset::default();
    // No patterns → fallback generic envelope: 0.50..=2.0
    assert!(r.check_ln("any_name.weight", 0.5));
    assert!(r.check_ln("any_name.weight", 1.5));
    assert!(!r.check_ln("any_name.weight", 0.4));
    assert!(!r.check_ln("any_name.weight", 2.1));
}

#[test]
fn default_ruleset_check_proj_rms_no_bounds_accepts_any() {
    let r = Ruleset::default();
    assert!(r.proj_weight_rms_min.is_none());
    assert!(r.proj_weight_rms_max.is_none());
    assert!(r.check_proj_rms(0.0));
    assert!(r.check_proj_rms(999.0));
}

// ── Policy loading ────────────────────────────────────────────────────────────

#[test]
fn load_policy_valid_yaml_with_ln_and_proj_bounds() {
    let yaml = r#"
version: 1
rules:
  my-arch:f16:
    name: "my-arch-f16-ruleset"
    ln:
      - pattern: ".*norm\\.weight$"
        min: 0.4
        max: 1.8
    proj_weight_rms_min: 0.02
    proj_weight_rms_max: 0.35
"#;
    let tmp = NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), yaml).unwrap();
    let r = load_policy(tmp.path(), "my-arch:f16").unwrap();
    assert_eq!(r.name, "my-arch-f16-ruleset");
    assert_eq!(r.ln.len(), 1);
    assert!(r.check_ln("blk.0.attn_norm.weight", 1.0));
    assert!(!r.check_ln("blk.0.attn_norm.weight", 0.1));
    assert!(r.check_proj_rms(0.20));
    assert!(!r.check_proj_rms(0.01));
    assert!(!r.check_proj_rms(0.40));
}

#[test]
fn load_policy_name_defaults_to_policy_colon_key_when_absent() {
    let yaml = r#"
version: 1
rules:
  unnamed-key:
    ln: []
"#;
    let tmp = NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), yaml).unwrap();
    let r = load_policy(tmp.path(), "unnamed-key").unwrap();
    assert_eq!(r.name, "policy:unnamed-key");
}

#[test]
fn load_policy_missing_key_returns_error() {
    let yaml = "version: 1\nrules:\n  present:\n    ln: []\n";
    let tmp = NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), yaml).unwrap();
    assert!(load_policy(tmp.path(), "absent").is_err());
}

#[test]
fn load_policy_missing_key_error_mentions_key() {
    let yaml = "version: 1\nrules:\n  x:\n    ln: []\n";
    let tmp = NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), yaml).unwrap();
    let err = load_policy(tmp.path(), "missing-key").unwrap_err();
    assert!(err.to_string().contains("missing-key"), "error: {err}");
}

#[test]
fn load_policy_invalid_yaml_returns_error() {
    let tmp = NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), "not: valid: yaml: [[[").unwrap();
    assert!(load_policy(tmp.path(), "any-key").is_err());
}

#[test]
fn load_policy_invalid_regex_pattern_returns_error() {
    let yaml = r#"
version: 1
rules:
  bad-regex:
    ln:
      - pattern: "["
        min: 0.5
        max: 2.0
"#;
    let tmp = NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), yaml).unwrap();
    assert!(load_policy(tmp.path(), "bad-regex").is_err());
}

#[test]
fn load_policy_no_proj_bounds_check_proj_accepts_any() {
    let yaml = r#"
version: 1
rules:
  no-proj:
    ln:
      - pattern: ".*norm\\.weight$"
        min: 0.5
        max: 2.0
"#;
    let tmp = NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), yaml).unwrap();
    let r = load_policy(tmp.path(), "no-proj").unwrap();
    assert!(r.proj_weight_rms_min.is_none());
    assert!(r.proj_weight_rms_max.is_none());
    assert!(r.check_proj_rms(0.0));
    assert!(r.check_proj_rms(1000.0));
}

#[test]
fn load_policy_file_not_found_returns_error() {
    let result = load_policy(std::path::Path::new("/nonexistent/path/policy.yml"), "key");
    assert!(result.is_err());
}

#[test]
fn load_policy_multiple_ln_rules_first_matching_pattern_wins() {
    let yaml = r#"
version: 1
rules:
  multi:
    ln:
      - pattern: "attn_norm\\.weight$"
        min: 0.5
        max: 1.5
      - pattern: ".*norm\\.weight$"
        min: 0.1
        max: 3.0
"#;
    let tmp = NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), yaml).unwrap();
    let r = load_policy(tmp.path(), "multi").unwrap();
    // attn_norm matches first rule (min=0.5, max=1.5)
    assert!(r.check_ln("blk.0.attn_norm.weight", 0.8));
    assert!(!r.check_ln("blk.0.attn_norm.weight", 0.2)); // below first rule's min
    // ffn_norm falls through to second rule (min=0.1, max=3.0)
    assert!(r.check_ln("blk.0.ffn_norm.weight", 0.2));
}

// ── Ruleset cloneability and immutability ─────────────────────────────────────

#[test]
fn rulesets_can_be_cloned_independently() {
    let a = rules_bitnet_b158_f16();
    let b = a.clone();
    assert_eq!(a.name, b.name);
    assert_eq!(a.ln.len(), b.ln.len());
    assert_eq!(a.proj_weight_rms_min, b.proj_weight_rms_min);
    assert_eq!(a.proj_weight_rms_max, b.proj_weight_rms_max);
}

#[test]
fn builtin_rulesets_have_ln_rules() {
    assert!(!rules_bitnet_b158_f16().ln.is_empty());
    assert!(!rules_bitnet_b158_i2s().ln.is_empty());
    assert!(!rules_generic().ln.is_empty());
}

#[test]
fn detect_rules_returns_same_ruleset_on_repeated_calls() {
    let r1 = detect_rules("bitnet", 1);
    let r2 = detect_rules("bitnet", 1);
    assert_eq!(r1.name, r2.name);
    assert_eq!(r1.ln.len(), r2.ln.len());
}

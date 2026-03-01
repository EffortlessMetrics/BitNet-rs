//! Snapshot wave 12 — bitnet-validation
//!
//! Covers: Ruleset fields and methods, detect_rules for multiple architectures,
//! is_ln_gamma patterns, check_ln envelope, check_proj_rms, load_policy errors.

use bitnet_validation::{
    Ruleset, detect_rules, is_ln_gamma, load_policy, rules_bitnet_b158_f16, rules_bitnet_b158_i2s,
    rules_generic,
};

// ── Rulesets ────────────────────────────────────────────────────────────────

#[test]
fn ruleset_f16_name() {
    let r = rules_bitnet_b158_f16();
    insta::assert_snapshot!(r.name);
}

#[test]
fn ruleset_f16_ln_count() {
    let r = rules_bitnet_b158_f16();
    insta::assert_snapshot!(format!("ln_thresholds={}", r.ln.len()));
}

#[test]
fn ruleset_f16_proj_rms_bounds() {
    let r = rules_bitnet_b158_f16();
    insta::assert_snapshot!(format!(
        "proj_min={:?} proj_max={:?}",
        r.proj_weight_rms_min, r.proj_weight_rms_max
    ));
}

#[test]
fn ruleset_i2s_name() {
    let r = rules_bitnet_b158_i2s();
    insta::assert_snapshot!(r.name);
}

#[test]
fn ruleset_i2s_ln_count() {
    let r = rules_bitnet_b158_i2s();
    insta::assert_snapshot!(format!("ln_thresholds={}", r.ln.len()));
}

#[test]
fn ruleset_i2s_proj_rms_bounds() {
    let r = rules_bitnet_b158_i2s();
    insta::assert_snapshot!(format!(
        "proj_min={:?} proj_max={:?}",
        r.proj_weight_rms_min, r.proj_weight_rms_max
    ));
}

#[test]
fn ruleset_generic_name() {
    let r = rules_generic();
    insta::assert_snapshot!(r.name);
}

#[test]
fn ruleset_generic_ln_count() {
    let r = rules_generic();
    insta::assert_snapshot!(format!("ln_thresholds={}", r.ln.len()));
}

#[test]
fn ruleset_generic_proj_rms_bounds() {
    let r = rules_generic();
    insta::assert_snapshot!(format!(
        "proj_min={:?} proj_max={:?}",
        r.proj_weight_rms_min, r.proj_weight_rms_max
    ));
}

// ── detect_rules ────────────────────────────────────────────────────────────

#[test]
fn detect_rules_bitnet_f16() {
    let r = detect_rules("bitnet", 1);
    insta::assert_snapshot!(r.name);
}

#[test]
fn detect_rules_bitnet_i2s() {
    let r = detect_rules("bitnet", 26);
    insta::assert_snapshot!(r.name);
}

#[test]
fn detect_rules_llama_f16() {
    let r = detect_rules("llama", 1);
    insta::assert_snapshot!(r.name);
}

#[test]
fn detect_rules_llama_q4() {
    let r = detect_rules("llama", 7);
    insta::assert_snapshot!(r.name);
}

#[test]
fn detect_rules_unknown_arch() {
    let r = detect_rules("transformer-xl", 1);
    insta::assert_snapshot!(r.name);
}

#[test]
fn detect_rules_empty_arch() {
    let r = detect_rules("", 0);
    insta::assert_snapshot!(r.name);
}

// ── check_ln ────────────────────────────────────────────────────────────────

#[test]
fn check_ln_f16_attn_norm_valid() {
    let r = rules_bitnet_b158_f16();
    let ok = r.check_ln("blk.0.attn_norm.weight", 0.8);
    insta::assert_snapshot!(format!("valid={ok}"));
}

#[test]
fn check_ln_f16_attn_norm_too_low() {
    let r = rules_bitnet_b158_f16();
    let ok = r.check_ln("blk.0.attn_norm.weight", 0.001);
    insta::assert_snapshot!(format!("valid={ok}"));
}

#[test]
fn check_ln_f16_ffn_norm_valid() {
    let r = rules_bitnet_b158_f16();
    let ok = r.check_ln("blk.5.ffn_norm.weight", 1.0);
    insta::assert_snapshot!(format!("valid={ok}"));
}

#[test]
fn check_ln_unmatched_name() {
    let r = rules_bitnet_b158_f16();
    let ok = r.check_ln("blk.0.some_random_tensor", 0.5);
    insta::assert_snapshot!(format!("valid={ok}"));
}

#[test]
fn check_ln_generic_input_layernorm() {
    let r = rules_generic();
    let ok = r.check_ln("input_layernorm.weight", 1.0);
    insta::assert_snapshot!(format!("valid={ok}"));
}

// ── check_proj_rms ──────────────────────────────────────────────────────────

#[test]
fn check_proj_rms_f16_in_range() {
    let r = rules_bitnet_b158_f16();
    let ok = r.check_proj_rms(0.5);
    insta::assert_snapshot!(format!("valid={ok}"));
}

#[test]
fn check_proj_rms_f16_zero() {
    let r = rules_bitnet_b158_f16();
    let ok = r.check_proj_rms(0.0);
    insta::assert_snapshot!(format!("valid={ok}"));
}

#[test]
fn check_proj_rms_i2s_in_range() {
    let r = rules_bitnet_b158_i2s();
    let ok = r.check_proj_rms(0.3);
    insta::assert_snapshot!(format!("valid={ok}"));
}

#[test]
fn check_proj_rms_default_ruleset() {
    let r = Ruleset::default();
    let ok = r.check_proj_rms(999.0);
    insta::assert_snapshot!(format!("valid={ok}"));
}

// ── is_ln_gamma ─────────────────────────────────────────────────────────────

#[test]
fn is_ln_gamma_attn_norm() {
    insta::assert_snapshot!(format!("{}", is_ln_gamma("blk.0.attn_norm.weight")));
}

#[test]
fn is_ln_gamma_ffn_norm() {
    insta::assert_snapshot!(format!("{}", is_ln_gamma("blk.3.ffn_norm.weight")));
}

#[test]
fn is_ln_gamma_final_norm() {
    insta::assert_snapshot!(format!("{}", is_ln_gamma("output_norm.weight")));
}

#[test]
fn is_ln_gamma_input_layernorm() {
    insta::assert_snapshot!(format!("{}", is_ln_gamma("model.layers.0.input_layernorm.weight")));
}

#[test]
fn is_ln_gamma_post_attention() {
    insta::assert_snapshot!(format!(
        "{}",
        is_ln_gamma("model.layers.0.post_attention_layernorm.weight")
    ));
}

#[test]
fn is_ln_gamma_not_ln() {
    insta::assert_snapshot!(format!("{}", is_ln_gamma("blk.0.attn_q.weight")));
}

#[test]
fn is_ln_gamma_ffn_layernorm() {
    insta::assert_snapshot!(format!("{}", is_ln_gamma("blk.0.ffn_layernorm.weight")));
}

#[test]
fn is_ln_gamma_empty() {
    insta::assert_snapshot!(format!("{}", is_ln_gamma("")));
}

// ── load_policy ─────────────────────────────────────────────────────────────

#[test]
fn load_policy_missing_file() {
    let result = load_policy(std::path::Path::new("/nonexistent/path.yml"), "key");
    insta::assert_snapshot!(format!("{}", result.unwrap_err()));
}

#[test]
fn load_policy_missing_key() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("policy.yml");
    std::fs::write(&path, "version: 1\nrules:\n  other_key:\n    name: test\n    ln: []\n")
        .unwrap();
    let result = load_policy(&path, "missing_key");
    let err = format!("{}", result.unwrap_err());
    // Strip the temp path (after "not found in ") for stable snapshots
    let stable = if let Some(idx) = err.find(" in /") { &err[..idx] } else { &err };
    insta::assert_snapshot!(stable);
}

#[test]
fn load_policy_valid_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("policy.yml");
    std::fs::write(
        &path,
        r#"version: 1
rules:
  my_model:
    name: "custom-model"
    ln:
      - pattern: "attn_norm"
        min: 0.1
        max: 2.0
    proj_weight_rms_min: 0.01
    proj_weight_rms_max: 5.0
"#,
    )
    .unwrap();
    let ruleset = load_policy(&path, "my_model").unwrap();
    insta::assert_snapshot!(format!("name={} ln_count={}", ruleset.name, ruleset.ln.len()));
}

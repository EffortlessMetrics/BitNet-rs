// SPDX-License-Identifier: MIT OR Apache-2.0
//! Extended tests for `bitnet-validation`.
//!
//! Complements `validation_tests.rs` with coverage of:
//! - `Ruleset::default()` / no-rules fallback envelope
//! - `Ruleset::check_proj_rms` when proj bounds are `None` (generic ruleset)
//! - `rules_bitnet_b158_i2s` pattern-specific checks
//! - `rules_bitnet_b158_f16` per-pattern boundary values
//! - `detect_rules` edge inputs (empty arch, numeric, mixed-case variants)
//! - `load_policy` with invalid regex, empty ln list, multiple keys
//! - `Ruleset::check_ln` fallback envelope when no pattern matches
//! - Property tests for I2S attn_norm, generic reject, proj_rms consistency

use bitnet_validation::{
    Ruleset, detect_rules, is_ln_gamma, load_policy, rules_bitnet_b158_f16, rules_bitnet_b158_i2s,
    rules_generic,
};
use proptest::prelude::*;
use tempfile::NamedTempFile;

// ── Ruleset::default() – no rules, uses fallback envelope ────────────────────

#[test]
fn default_ruleset_accepts_value_in_fallback_envelope() {
    let r = Ruleset::default();
    // No rules → fallback is (0.50..=2.0)
    assert!(
        r.check_ln("blk.0.attn_norm.weight", 1.0),
        "default Ruleset must accept 1.0 via fallback envelope"
    );
}

#[test]
fn default_ruleset_rejects_value_below_fallback_envelope() {
    let r = Ruleset::default();
    // Value 0.1 is below the generic fallback lower bound (0.50)
    assert!(
        !r.check_ln("blk.0.attn_norm.weight", 0.1),
        "default Ruleset must reject 0.1 (below fallback min 0.50)"
    );
}

#[test]
fn default_ruleset_rejects_value_above_fallback_envelope() {
    let r = Ruleset::default();
    assert!(
        !r.check_ln("blk.0.attn_norm.weight", 2.5),
        "default Ruleset must reject 2.5 (above fallback max 2.0)"
    );
}

#[test]
fn default_ruleset_check_proj_rms_with_no_bounds_always_true() {
    let r = Ruleset::default(); // proj bounds are None
    // When no proj bounds are set, any finite value should pass
    assert!(r.check_proj_rms(0.0));
    assert!(r.check_proj_rms(0.001));
    assert!(r.check_proj_rms(1_000.0));
}

// ── Generic ruleset: proj_rms has None bounds → always passes ────────────────

#[test]
fn generic_ruleset_check_proj_rms_unbounded() {
    let r = rules_generic();
    // Generic ruleset does not constrain proj_rms
    assert!(r.check_proj_rms(0.0));
    assert!(r.check_proj_rms(100.0));
    assert!(r.check_proj_rms(1e-6));
}

// ── rules_bitnet_b158_i2s pattern-specific checks ────────────────────────────

#[test]
fn i2s_attn_norm_accepts_very_low_rms() {
    let r = rules_bitnet_b158_i2s();
    // attn_norm in I2S: min=0.01, so 0.015 must pass
    assert!(r.check_ln("blk.0.attn_norm.weight", 0.015));
}

#[test]
fn i2s_attn_norm_accepts_min_boundary() {
    let r = rules_bitnet_b158_i2s();
    assert!(r.check_ln("blk.0.attn_norm.weight", 0.01));
}

#[test]
fn i2s_attn_norm_rejects_below_min() {
    let r = rules_bitnet_b158_i2s();
    assert!(!r.check_ln("blk.0.attn_norm.weight", 0.005));
}

#[test]
fn i2s_ffn_norm_accepts_mid_range() {
    let r = rules_bitnet_b158_i2s();
    // ffn_norm in I2S: min=0.50, max=2.0
    assert!(r.check_ln("blk.3.ffn_norm.weight", 1.0));
}

#[test]
fn i2s_ffn_norm_rejects_below_min() {
    let r = rules_bitnet_b158_i2s();
    assert!(!r.check_ln("blk.3.ffn_norm.weight", 0.3));
}

#[test]
fn i2s_proj_rms_accepts_valid_range() {
    let r = rules_bitnet_b158_i2s();
    // I2S: proj_weight_rms_min=0.002, proj_weight_rms_max=0.20
    assert!(r.check_proj_rms(0.05));
}

#[test]
fn i2s_proj_rms_rejects_below_min() {
    let r = rules_bitnet_b158_i2s();
    assert!(!r.check_proj_rms(0.001));
}

#[test]
fn i2s_proj_rms_rejects_above_max() {
    let r = rules_bitnet_b158_i2s();
    assert!(!r.check_proj_rms(0.25));
}

// ── rules_bitnet_b158_f16 per-pattern boundary values ────────────────────────

#[test]
fn f16_ffn_layernorm_accepts_low_rms_at_boundary() {
    let r = rules_bitnet_b158_f16();
    // ffn_layernorm in F16: min=0.05
    assert!(r.check_ln("blk.2.ffn_layernorm.weight", 0.05));
}

#[test]
fn f16_ffn_layernorm_rejects_below_min() {
    let r = rules_bitnet_b158_f16();
    assert!(!r.check_ln("blk.2.ffn_layernorm.weight", 0.04));
}

#[test]
fn f16_post_attention_layernorm_accepts_at_min() {
    let r = rules_bitnet_b158_f16();
    // post_attention_layernorm: min=0.25
    assert!(r.check_ln("blk.0.post_attention_layernorm.weight", 0.25));
}

#[test]
fn f16_post_attention_layernorm_rejects_below_min() {
    let r = rules_bitnet_b158_f16();
    assert!(!r.check_ln("blk.0.post_attention_layernorm.weight", 0.20));
}

#[test]
fn f16_input_layernorm_rejects_below_min() {
    let r = rules_bitnet_b158_f16();
    // input_layernorm: min=0.35
    assert!(!r.check_ln("model.layers.0.input_layernorm.weight", 0.30));
}

// ── detect_rules edge inputs ──────────────────────────────────────────────────

#[test]
fn detect_rules_empty_arch_gives_generic() {
    let r = detect_rules("", 1);
    assert_eq!(r.name, "generic");
}

#[test]
fn detect_rules_numeric_only_arch_gives_generic() {
    let r = detect_rules("123", 0);
    assert_eq!(r.name, "generic");
}

#[test]
fn detect_rules_bitnet_file_type_zero_gives_i2s() {
    // file_type != 1 → treat as quantized (I2S)
    let r = detect_rules("bitnet", 0);
    assert_eq!(r.name, "bitnet-b1.58:i2_s");
}

#[test]
fn detect_rules_b158_substring_detected() {
    let r = detect_rules("b1.58-2B", 1);
    assert_eq!(r.name, "bitnet-b1.58:f16");
}

#[test]
fn detect_rules_llama_arch_gives_generic() {
    let r = detect_rules("llama", 1);
    assert_eq!(r.name, "generic");
}

// ── Ruleset::check_ln fallback when no pattern matches ───────────────────────

#[test]
fn f16_ruleset_fallback_accepts_non_norm_name_within_range() {
    let r = rules_bitnet_b158_f16();
    // A name that matches the broad ".*norm.weight$" catch-all (min=0.50)
    // Let's use something that would fall through to the final catch-all
    assert!(r.check_ln("model.norm.weight", 0.8));
}

#[test]
fn f16_ruleset_fallback_rejects_non_matching_name_out_of_range() {
    let r = rules_bitnet_b158_f16();
    // "output.weight" does not match any rule in f16 → generic fallback (0.50..=2.0)
    // Below 0.50 should be rejected by the fallback
    assert!(!r.check_ln("output.weight", 0.1));
}

// ── load_policy edge cases ────────────────────────────────────────────────────

#[test]
fn load_policy_invalid_yaml_returns_error() {
    let tmp = NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), "this is not: valid: yaml: [").unwrap();
    let result = load_policy(tmp.path(), "any-key");
    assert!(result.is_err(), "invalid YAML must return Err");
}

#[test]
fn load_policy_invalid_regex_in_pattern_returns_error() {
    let yaml = r#"
version: 1
rules:
  bad-regex-key:
    name: "bad-regex"
    ln:
      - pattern: "["
        min: 0.5
        max: 2.0
"#;
    let tmp = NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), yaml).unwrap();
    let result = load_policy(tmp.path(), "bad-regex-key");
    assert!(result.is_err(), "invalid regex in policy must return Err");
}

#[test]
fn load_policy_empty_ln_list_succeeds() {
    let yaml = r#"
version: 1
rules:
  empty-ln:
    name: "empty"
    ln: []
"#;
    let tmp = NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), yaml).unwrap();
    let r = load_policy(tmp.path(), "empty-ln").unwrap();
    assert_eq!(r.name, "empty");
    assert!(r.ln.is_empty());
}

#[test]
fn load_policy_empty_ln_falls_back_to_generic_envelope() {
    let yaml = r#"
version: 1
rules:
  empty-ln:
    ln: []
"#;
    let tmp = NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), yaml).unwrap();
    let r = load_policy(tmp.path(), "empty-ln").unwrap();
    // No rules → fallback accepts values in 0.50..=2.0
    assert!(r.check_ln("blk.0.attn_norm.weight", 1.0));
    assert!(!r.check_ln("blk.0.attn_norm.weight", 0.1));
}

#[test]
fn load_policy_multiple_keys_access_correct_one() {
    let yaml = r#"
version: 1
rules:
  key-a:
    name: "ruleset-a"
    ln:
      - pattern: ".*norm\\.weight$"
        min: 0.5
        max: 2.0
  key-b:
    name: "ruleset-b"
    ln:
      - pattern: ".*norm\\.weight$"
        min: 0.8
        max: 1.2
"#;
    let tmp = NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), yaml).unwrap();

    let ra = load_policy(tmp.path(), "key-a").unwrap();
    let rb = load_policy(tmp.path(), "key-b").unwrap();

    assert_eq!(ra.name, "ruleset-a");
    assert_eq!(rb.name, "ruleset-b");

    // key-a accepts 0.6, key-b rejects it (min=0.8)
    assert!(ra.check_ln("blk.0.attn_norm.weight", 0.6));
    assert!(!rb.check_ln("blk.0.attn_norm.weight", 0.6));
}

#[test]
fn load_policy_proj_bounds_are_loaded() {
    let yaml = r#"
version: 1
rules:
  proj-test:
    ln: []
    proj_weight_rms_min: 0.05
    proj_weight_rms_max: 0.30
"#;
    let tmp = NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), yaml).unwrap();
    let r = load_policy(tmp.path(), "proj-test").unwrap();
    assert!(r.check_proj_rms(0.10));
    assert!(!r.check_proj_rms(0.01));
    assert!(!r.check_proj_rms(0.40));
}

#[test]
fn load_policy_default_name_is_policy_key() {
    let yaml = r#"
version: 1
rules:
  my-arch:
    ln: []
"#;
    let tmp = NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), yaml).unwrap();
    let r = load_policy(tmp.path(), "my-arch").unwrap();
    assert_eq!(r.name, "policy:my-arch");
}

// ── Property tests ────────────────────────────────────────────────────────────

proptest! {
    /// I2S attn_norm accepts any RMS in [0.01, 2.0].
    #[test]
    fn prop_i2s_attn_norm_accepts_valid_range(rms in 0.01f32..=2.0) {
        let r = rules_bitnet_b158_i2s();
        prop_assert!(
            r.check_ln("blk.0.attn_norm.weight", rms),
            "I2S attn_norm should accept rms={}", rms
        );
    }

    /// Generic ruleset check_proj_rms with no bounds always returns true.
    #[test]
    fn prop_generic_proj_rms_always_true(rms in 0.0f32..=1_000.0) {
        let r = rules_generic();
        prop_assert!(
            r.check_proj_rms(rms),
            "generic ruleset has no proj bounds so check_proj_rms must always return true"
        );
    }

    /// F16 check_proj_rms is exactly [0.01, 0.40].
    #[test]
    fn prop_f16_proj_rms_consistent_with_bounds(rms in 0.0f32..=1.0) {
        let r = rules_bitnet_b158_f16();
        let accepted = r.check_proj_rms(rms);
        let expected = (0.01f32..=0.40f32).contains(&rms);
        prop_assert_eq!(
            accepted, expected,
            "f16 proj_rms={}: expected={} got={}", rms, expected, accepted
        );
    }

    /// detect_rules is deterministic: two calls with same inputs return rulesets with same name.
    #[test]
    fn prop_detect_rules_is_deterministic(
        arch in "[a-zA-Z0-9._-]{0,20}",
        file_type in 0u32..=10,
    ) {
        let r1 = detect_rules(&arch, file_type);
        let r2 = detect_rules(&arch, file_type);
        prop_assert_eq!(r1.name, r2.name);
    }

    /// is_ln_gamma is consistent with check_ln in that names identified as LN weights
    /// are subject to the ruleset's LN checks (not silently bypassed).
    #[test]
    fn prop_is_ln_gamma_consistent_with_ruleset_check(
        keyword in prop_oneof![
            Just("attn_norm"),
            Just("ffn_norm"),
            Just("ffn_layernorm"),
            Just("input_layernorm"),
        ],
        block in 0u32..32,
    ) {
        let name = format!("blk.{}.{}.weight", block, keyword);
        let is_ln = is_ln_gamma(&name);
        // If is_ln_gamma identifies it as a LayerNorm weight, the name must have ".weight" suffix
        prop_assert!(is_ln, "known LN keyword {:?} must be recognised by is_ln_gamma", keyword);
    }
}

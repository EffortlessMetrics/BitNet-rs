//! Property-based tests for `bitnet-validation`.
//!
//! Tests validation name detection, ruleset thresholds, and built-in rule
//! factories against adversarial and fuzz-like string inputs.

use bitnet_validation::{
    detect_rules, is_ln_gamma, load_policy, rules_bitnet_b158_f16, rules_bitnet_b158_i2s,
    rules_generic,
};
use proptest::prelude::*;

// ── Strategies ──────────────────────────────────────────────────────────────

/// A valid LayerNorm weight name that `is_ln_gamma` should accept.
fn arb_ln_name() -> impl Strategy<Value = String> {
    let keywords = prop_oneof![
        Just("attn_norm"),
        Just("ffn_norm"),
        Just("ffn_layernorm"),
        Just("rms_norm"),
        Just("input_layernorm"),
        Just("post_attention_layernorm"),
        Just("final_layernorm"),
        Just("final_norm"),
        Just("norm"),
    ];
    let prefix = prop_oneof![
        Just("".to_string()),
        (0usize..=30usize).prop_map(|n| format!("blk.{n}.")),
        Just("model.layers.0.".to_string()),
    ];
    (prefix, keywords).prop_map(|(pre, kw)| format!("{pre}{kw}.weight"))
}

/// Something that definitely looks NOT like a LayerNorm weight name.
fn arb_non_ln_name() -> impl Strategy<Value = String> {
    // Non-weight names or names with clearly non-LN stems.
    prop_oneof![
        Just("blk.0.attn_q.weight".to_string()),
        Just("output.weight".to_string()),
        Just("token_embd.weight".to_string()),
        Just("blk.0.ffn_up.weight".to_string()),
        Just("blk.0.ffn_down.weight".to_string()),
        Just("blk.0.attn_v.weight".to_string()),
    ]
}

// ── Property tests ───────────────────────────────────────────────────────────

proptest! {
    /// All valid LayerNorm names constructed from known patterns return true.
    #[test]
    fn known_ln_patterns_detected(name in arb_ln_name()) {
        prop_assert!(
            is_ln_gamma(&name),
            "expected is_ln_gamma=true for: {name}"
        );
    }

    /// Projection/embedding weight names are never misclassified as LN.
    #[test]
    fn non_ln_names_not_detected(name in arb_non_ln_name()) {
        prop_assert!(
            !is_ln_gamma(&name),
            "expected is_ln_gamma=false for: {name}"
        );
    }

    /// Names not ending in ".weight" are never detected as LN, regardless of content.
    #[test]
    fn names_without_weight_suffix_not_detected(
        stem in "[a-z_]{1,30}",
    ) {
        // Append anything that isn't ".weight"
        let name = format!("{stem}.bias");
        prop_assert!(!is_ln_gamma(&name));
        let name2 = stem.to_string();
        prop_assert!(!is_ln_gamma(&name2));
    }

    /// `check_proj_rms` always returns true when no thresholds are set.
    #[test]
    fn proj_rms_check_without_bounds_always_passes(rms in -1e6f32..=1e6f32) {
        let ruleset = rules_generic();
        // Generic ruleset has no proj_weight_rms thresholds.
        if ruleset.proj_weight_rms_min.is_none() && ruleset.proj_weight_rms_max.is_none() {
            prop_assert!(ruleset.check_proj_rms(rms));
        }
    }

    /// `check_ln` for any name returns a bool (never panics) for any RMS.
    #[test]
    fn check_ln_never_panics(
        name in "[a-zA-Z0-9_.]{1,50}",
        rms in 0.0f32..=10.0f32,
    ) {
        let ruleset = rules_bitnet_b158_f16();
        let _ = ruleset.check_ln(&name, rms);
        let ruleset2 = rules_bitnet_b158_i2s();
        let _ = ruleset2.check_ln(&name, rms);
    }

    /// `detect_rules` never panics for any arch/file_type combination.
    #[test]
    fn detect_rules_never_panics(
        arch in "[a-zA-Z0-9_-]{1,20}",
        file_type in 0u32..=10u32,
    ) {
        let ruleset = detect_rules(&arch, file_type);
        prop_assert!(!ruleset.name.is_empty());
    }
}

// ── Additional property tests ────────────────────────────────────────────────

proptest! {
    /// LayerNorm gamma invariant: any RMS within 0.01..=2.0 is accepted for
    /// `attn_norm` weights in the I2_S ruleset (min=0.01, max=2.0).
    #[test]
    fn check_ln_accepts_in_range_rms_for_i2s_attn_norm(rms in 0.01f32..=2.0f32) {
        let r = rules_bitnet_b158_i2s();
        prop_assert!(
            r.check_ln("blk.0.attn_norm.weight", rms),
            "i2s attn_norm should accept rms={rms}"
        );
    }

    /// LayerNorm gamma invariant: any RMS strictly below 0.01 is rejected for
    /// `attn_norm` weights in the I2_S ruleset (min=0.01).
    #[test]
    fn check_ln_rejects_below_min_for_i2s_attn_norm(rms in 0.0001f32..0.0099f32) {
        let r = rules_bitnet_b158_i2s();
        prop_assert!(
            !r.check_ln("blk.0.attn_norm.weight", rms),
            "i2s attn_norm should reject rms={rms} (below min 0.01)"
        );
    }

    /// LayerNorm gamma invariant: projection RMS above 0.40 is always rejected
    /// in the F16 ruleset (proj_weight_rms_max = 0.40).
    #[test]
    fn check_proj_rms_rejects_above_max_for_f16(rms in 0.401f32..=100.0f32) {
        let r = rules_bitnet_b158_f16();
        prop_assert!(
            !r.check_proj_rms(rms),
            "f16 ruleset should reject proj rms={rms} (above max 0.40)"
        );
    }

    /// Error message formatting: when `load_policy` fails because a key is
    /// absent, the error message is always non-empty.
    #[test]
    fn load_policy_missing_key_has_nonempty_error_message(
        key in "[a-z]{2,8}:[a-z]{2,8}",
    ) {
        // Sentinel key can never match the generated `key` (contains uppercase).
        let yaml = "version: 1\nrules:\n  \"SENTINEL__\":\n    ln: []\n";
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), yaml).unwrap();
        let err = load_policy(tmp.path(), &key).unwrap_err();
        prop_assert!(
            !err.to_string().is_empty(),
            "error message should be non-empty for missing key {key:?}"
        );
    }

    /// Policy key format: a key of the form "arch:variant" stored in the YAML
    /// is successfully retrieved by `load_policy`.
    #[test]
    fn load_policy_arch_colon_variant_key_round_trips(
        arch in "[a-z]{3,8}",
        variant in "[a-z0-9]{2,8}",
    ) {
        let key = format!("{arch}:{variant}");
        let yaml = format!("version: 1\nrules:\n  \"{key}\":\n    ln: []\n");
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), yaml.as_bytes()).unwrap();
        let result = load_policy(tmp.path(), &key);
        prop_assert!(
            result.is_ok(),
            "arch:variant key {key:?} should succeed; got: {:?}",
            result.err()
        );
    }

    /// Policy key format: the empty string key is never found in a policy that
    /// stores only non-empty keys.
    #[test]
    fn load_policy_empty_key_is_not_found(
        stored_key in "[a-z]{2,10}",
    ) {
        let yaml = format!("version: 1\nrules:\n  \"{stored_key}\":\n    ln: []\n");
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), yaml.as_bytes()).unwrap();
        let result = load_policy(tmp.path(), "");
        prop_assert!(result.is_err(), "empty key should not match stored key {stored_key:?}");
    }

    /// Validation gate modes: `detect_rules` with any arch containing "bitnet"
    /// and file_type=1 always dispatches to the F16 ruleset.
    #[test]
    fn detect_rules_bitnet_arch_file_type_1_gives_f16(
        prefix in "[a-z]{0,4}",
        suffix in "[a-z]{0,4}",
    ) {
        let arch = format!("{prefix}bitnet{suffix}");
        let r = detect_rules(&arch, 1);
        prop_assert_eq!(
            r.name.as_str(),
            "bitnet-b1.58:f16",
            "arch={:?} with file_type=1 should give f16 ruleset",
            arch
        );
    }

    /// Validation gate modes: `detect_rules` with a non-bitnet arch string
    /// always falls back to the generic ruleset.
    #[test]
    fn detect_rules_non_bitnet_arch_gives_generic(
        arch in "[a-z]{4,12}".prop_filter("not bitnet", |s| !s.contains("bitnet")),
        file_type in 0u32..=10u32,
    ) {
        let r = detect_rules(&arch, file_type);
        prop_assert_eq!(
            r.name.as_str(),
            "generic",
            "arch={:?} should give generic ruleset",
            arch
        );
    }
}

// ── Unit tests ───────────────────────────────────────────────────────────────

#[test]
fn known_ln_names_are_detected() {
    let positive = [
        "blk.0.attn_norm.weight",
        "blk.3.ffn_layernorm.weight",
        "final_norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.1.post_attention_layernorm.weight",
    ];
    for name in positive {
        assert!(is_ln_gamma(name), "expected true for: {name}");
    }
}

#[test]
fn known_non_ln_names_are_not_detected() {
    let negative = [
        "blk.0.attn_q.weight",
        "output.weight",
        "token_embd.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_norm", // missing .weight suffix
        "not_a_norm.bias",
    ];
    for name in negative {
        assert!(!is_ln_gamma(name), "expected false for: {name}");
    }
}

#[test]
fn builtin_rulesets_have_names() {
    assert!(!rules_bitnet_b158_f16().name.is_empty());
    assert!(!rules_bitnet_b158_i2s().name.is_empty());
    assert!(!rules_generic().name.is_empty());
}

#[test]
fn detect_bitnet_arch_returns_named_ruleset() {
    let ruleset = detect_rules("bitnet", 0);
    assert!(!ruleset.name.is_empty());
}

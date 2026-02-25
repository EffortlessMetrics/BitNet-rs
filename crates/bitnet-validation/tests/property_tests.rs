//! Property-based tests for `bitnet-validation`.
//!
//! Tests validation name detection, ruleset thresholds, and built-in rule
//! factories against adversarial and fuzz-like string inputs.

use bitnet_validation::{
    detect_rules, is_ln_gamma, rules_bitnet_b158_f16, rules_bitnet_b158_i2s, rules_generic,
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
        let name2 = format!("{stem}");
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

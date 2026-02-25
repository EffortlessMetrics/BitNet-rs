//! Property-based tests for `bitnet-st-tools` — `is_ln_gamma` invariants.
//!
//! Tests the public function `is_ln_gamma(name: &str) -> bool` which identifies
//! LayerNorm gamma weight tensor names from SafeTensors / GGUF files.
//!
//! # Key invariants
//! - Any string not ending with `.weight` always returns `false` (fast-path)
//! - `is_ln_gamma` is deterministic and pure
//! - Known positive patterns always return `true`
//! - Known negative patterns always return `false`

use bitnet_st_tools::common::is_ln_gamma;
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

/// Generate a random identifier component (letters, digits, underscores).
fn ident_strategy() -> impl Strategy<Value = String> {
    "[a-z][a-z0-9_]{0,15}"
        .prop_map(|s| s)
}

/// Generate a tensor name that definitely does NOT end with ".weight".
fn non_weight_name_strategy() -> impl Strategy<Value = String> {
    (ident_strategy(), prop_oneof![
        Just(".bias"),
        Just(".scale"),
        Just(""),
        Just(".gamma"),
        Just(".beta"),
    ])
    .prop_map(|(base, suffix)| format!("model.{base}{suffix}"))
}

/// Generate a valid layernorm-like tensor name (ends with ".weight", contains norm keyword).
fn ln_weight_name_strategy() -> impl Strategy<Value = String> {
    (
        prop::num::usize::ANY.prop_map(|n| n % 32),
        prop_oneof![
            Just("input_layernorm"),
            Just("post_attention_layernorm"),
            Just("attn_norm"),
            Just("ffn_norm"),
            Just("final_layernorm"),
            Just("rms_norm"),
        ],
    )
    .prop_map(|(layer, norm)| format!("model.layers.{layer}.{norm}.weight"))
}

/// Generate a projection weight name (not a layernorm).
fn proj_weight_name_strategy() -> impl Strategy<Value = String> {
    (
        prop::num::usize::ANY.prop_map(|n| n % 32),
        prop_oneof![
            Just("self_attn.q_proj"),
            Just("self_attn.k_proj"),
            Just("self_attn.v_proj"),
            Just("self_attn.o_proj"),
            Just("mlp.gate_proj"),
            Just("mlp.up_proj"),
            Just("mlp.down_proj"),
        ],
    )
    .prop_map(|(layer, proj)| format!("model.layers.{layer}.{proj}.weight"))
}

// ---------------------------------------------------------------------------
// Properties
// ---------------------------------------------------------------------------

proptest! {
    /// Anything not ending in ".weight" is always rejected (fast-path invariant).
    #[test]
    fn prop_non_weight_suffix_always_false(name in non_weight_name_strategy()) {
        // Only test names that clearly don't end with ".weight"
        if !name.ends_with(".weight") {
            prop_assert!(
                !is_ln_gamma(&name),
                "Expected false for non-.weight name: {name}"
            );
        }
    }

    /// Known layernorm names with .weight suffix always return true.
    #[test]
    fn prop_layernorm_weight_names_always_true(name in ln_weight_name_strategy()) {
        prop_assert!(
            is_ln_gamma(&name),
            "Expected true for LN weight name: {name}"
        );
    }

    /// Known projection weight names always return false.
    #[test]
    fn prop_projection_weight_names_always_false(name in proj_weight_name_strategy()) {
        prop_assert!(
            !is_ln_gamma(&name),
            "Expected false for projection weight name: {name}"
        );
    }

    /// `is_ln_gamma` is deterministic — same input always gives same output.
    #[test]
    fn prop_is_ln_gamma_is_deterministic(name in any::<String>()) {
        prop_assert_eq!(is_ln_gamma(&name), is_ln_gamma(&name));
    }

    /// `is_ln_gamma` never panics on arbitrary string input.
    #[test]
    fn prop_is_ln_gamma_never_panics(name in any::<String>()) {
        let _ = is_ln_gamma(&name);
    }
}

// ---------------------------------------------------------------------------
// Unit regression tests
// ---------------------------------------------------------------------------

#[test]
fn empty_string_returns_false() {
    assert!(!is_ln_gamma(""));
}

#[test]
fn known_positive_names() {
    let cases = [
        "model.layers.0.input_layernorm.weight",
        "model.layers.1.post_attention_layernorm.weight",
        "transformer.blocks.0.attn_norm.weight",
        "transformer.blocks.0.ffn_norm.weight",
        "model.norm.weight",
        "decoder.final_layernorm.weight",
        "encoder.final_norm.weight",
        "norm.weight",
        "rms_norm.weight",
    ];
    for name in cases {
        assert!(is_ln_gamma(name), "expected true for: {name}");
    }
}

#[test]
fn known_negative_names() {
    let cases = [
        "model.layers.0.input_layernorm.bias",
        "model.embed_tokens.weight",
        "model.lm_head.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
    ];
    for name in cases {
        assert!(!is_ln_gamma(name), "expected false for: {name}");
    }
}

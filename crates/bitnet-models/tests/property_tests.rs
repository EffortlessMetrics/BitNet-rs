//! Property-based tests for bitnet-models public API.
//!
//! Tests structural invariants:
//! - `names::is_layernorm_weight` / `is_projection_weight`: mutual exclusivity,
//!   suffix stability, no panics
//! - `qk256_tolerance_bytes` re-export: matches bitnet-quantization directly
//! - `QK256_SIZE_TOLERANCE_PERCENT` re-export: value identity

#![cfg(all(test, feature = "cpu"))]

use bitnet_models::names::{is_layernorm_weight, is_projection_weight};
use bitnet_models::{QK256_SIZE_TOLERANCE_PERCENT, qk256_tolerance_bytes};
use proptest::prelude::*;

// ── Naming predicate invariants ─────────────────────────────────────────────

/// LN suffixes known to be matched by `is_layernorm_weight`.
const LN_SUFFIXES: &[&str] = &[
    ".attention_norm.weight",
    ".ffn_norm.weight",
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".attn_norm.weight",
    ".final_norm.weight",
    ".rms_norm.weight",
    ".norm.weight",
];

/// Projection suffixes known to be matched by `is_projection_weight`.
const PROJ_SUFFIXES: &[&str] = &[
    ".q_proj.weight",
    ".k_proj.weight",
    ".v_proj.weight",
    ".o_proj.weight",
    ".gate_proj.weight",
    ".up_proj.weight",
    ".down_proj.weight",
    ".attn_q.weight",
    ".attn_k.weight",
    ".attn_v.weight",
    ".attn_output.weight",
    ".ffn_gate.weight",
    ".ffn_up.weight",
    ".ffn_down.weight",
];

proptest! {
    /// Any name ending with a known LN suffix is classified as LayerNorm.
    #[test]
    fn ln_suffix_always_detected(
        prefix in "[a-z0-9_\\.]{0,20}",
        suffix_idx in 0usize..LN_SUFFIXES.len(),
    ) {
        let name = format!("{}{}", prefix, LN_SUFFIXES[suffix_idx]);
        prop_assert!(is_layernorm_weight(&name),
            "Expected is_layernorm_weight('{}') == true", name);
    }

    /// Any name ending with a known projection suffix is classified as projection.
    #[test]
    fn proj_suffix_always_detected(
        prefix in "[a-z0-9_\\.]{0,20}",
        suffix_idx in 0usize..PROJ_SUFFIXES.len(),
    ) {
        let name = format!("{}{}", prefix, PROJ_SUFFIXES[suffix_idx]);
        prop_assert!(is_projection_weight(&name),
            "Expected is_projection_weight('{}') == true", name);
    }

    /// LN and projection predicates are mutually exclusive for LN names.
    #[test]
    fn ln_names_not_proj(
        prefix in "[a-z0-9_\\.]{0,20}",
        suffix_idx in 0usize..LN_SUFFIXES.len(),
    ) {
        let name = format!("{}{}", prefix, LN_SUFFIXES[suffix_idx]);
        prop_assert!(!is_projection_weight(&name),
            "'{}' classified as both LN and projection", name);
    }

    /// Projection names are not classified as LayerNorm.
    #[test]
    fn proj_names_not_ln(
        prefix in "[a-z0-9_\\.]{0,20}",
        suffix_idx in 0usize..PROJ_SUFFIXES.len(),
    ) {
        let name = format!("{}{}", prefix, PROJ_SUFFIXES[suffix_idx]);
        prop_assert!(!is_layernorm_weight(&name),
            "'{}' classified as both LN and projection", name);
    }

    /// Neither predicate panics on arbitrary UTF-8 strings.
    #[test]
    fn predicates_never_panic(name in ".*") {
        let _ = is_layernorm_weight(&name);
        let _ = is_projection_weight(&name);
    }

    /// Predicates are pure (same input → same output, no hidden state).
    #[test]
    fn predicates_are_pure(name in "[a-zA-Z0-9_\\.]{1,60}") {
        let ln1 = is_layernorm_weight(&name);
        let ln2 = is_layernorm_weight(&name);
        let p1 = is_projection_weight(&name);
        let p2 = is_projection_weight(&name);
        prop_assert_eq!(ln1, ln2, "is_layernorm_weight not pure for '{}'", name);
        prop_assert_eq!(p1, p2, "is_projection_weight not pure for '{}'", name);
    }
}

// ── Re-export identity ───────────────────────────────────────────────────────

#[test]
fn reexport_tolerance_constant_matches_source() {
    use bitnet_quantization::QK256_SIZE_TOLERANCE_PERCENT as SRC;
    assert_eq!(QK256_SIZE_TOLERANCE_PERCENT, SRC, "re-exported constant must equal source");
}

proptest! {
    /// Re-exported `qk256_tolerance_bytes` produces identical results to the source.
    #[test]
    fn reexport_tolerance_fn_matches_source(n in 0usize..1_000_000_000usize) {
        use bitnet_quantization::qk256_tolerance_bytes as src;
        prop_assert_eq!(qk256_tolerance_bytes(n), src(n));
    }
}

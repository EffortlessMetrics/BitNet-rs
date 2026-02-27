//! Property-based tests for bitnet-models public API.
//!
//! Tests structural invariants:
//! - `names::is_layernorm_weight` / `is_projection_weight`: mutual exclusivity,
//!   suffix stability, no panics
//! - `qk256_tolerance_bytes` re-export: matches bitnet-quantization directly
//! - `QK256_SIZE_TOLERANCE_PERCENT` re-export: value identity

#![cfg(all(test, feature = "cpu"))]

use bitnet_common::BitNetConfig;
use bitnet_gguf::{GGUF_MAGIC, check_magic, read_version};
use bitnet_models::formats::gguf::GgufTensorType;
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

// ── ModelConfig / BitNetConfig validation ────────────────────────────────────

proptest! {
    /// A config whose key architecture fields are all valid positive values passes
    /// `validate()`.  `hidden_size` is always an exact multiple of `num_heads`
    /// so the divisibility constraint is satisfied.
    #[test]
    fn valid_model_config_passes_validation(
        num_heads   in 1usize..=8usize,
        head_dim    in 1usize..=64usize,
        vocab_size  in 1usize..=32000usize,
        num_layers  in 1usize..=16usize,
    ) {
        let hidden_size = num_heads * head_dim;
        let mut cfg = BitNetConfig::default();
        cfg.model.hidden_size = hidden_size;
        cfg.model.num_heads   = num_heads;
        cfg.model.num_key_value_heads = 0; // 0 ⇒ treated as equal to num_heads
        cfg.model.vocab_size  = vocab_size;
        cfg.model.num_layers  = num_layers;
        prop_assert!(
            cfg.validate().is_ok(),
            "hidden_size={hidden_size}, num_heads={num_heads}, vocab_size={vocab_size}, \
             num_layers={num_layers} should all be valid"
        );
    }

    /// Setting `hidden_size` to zero must always fail validation regardless of
    /// the other architecture hyper-parameters.
    #[test]
    fn zero_hidden_size_always_fails_validation(
        vocab_size in 1usize..=32000usize,
        num_layers in 1usize..=16usize,
    ) {
        let mut cfg = BitNetConfig::default();
        cfg.model.hidden_size = 0;
        cfg.model.vocab_size  = vocab_size;
        cfg.model.num_layers  = num_layers;
        prop_assert!(cfg.validate().is_err(), "hidden_size=0 must be rejected");
    }

    /// Setting `vocab_size` to zero must always fail validation.
    #[test]
    fn zero_vocab_size_always_fails_validation(
        num_heads  in 1usize..=8usize,
        head_dim   in 1usize..=64usize,
        num_layers in 1usize..=16usize,
    ) {
        let mut cfg = BitNetConfig::default();
        cfg.model.hidden_size = num_heads * head_dim;
        cfg.model.num_heads   = num_heads;
        cfg.model.vocab_size  = 0;
        cfg.model.num_layers  = num_layers;
        prop_assert!(cfg.validate().is_err(), "vocab_size=0 must be rejected");
    }

    /// Setting `num_layers` to zero must always fail validation.
    #[test]
    fn zero_num_layers_always_fails_validation(
        num_heads  in 1usize..=8usize,
        head_dim   in 1usize..=64usize,
        vocab_size in 1usize..=32000usize,
    ) {
        let mut cfg = BitNetConfig::default();
        cfg.model.hidden_size = num_heads * head_dim;
        cfg.model.num_heads   = num_heads;
        cfg.model.vocab_size  = vocab_size;
        cfg.model.num_layers  = 0;
        prop_assert!(cfg.validate().is_err(), "num_layers=0 must be rejected");
    }

    /// A `hidden_size` that is NOT a multiple of `num_heads` must fail
    /// validation.  We construct `hidden_size = base * num_heads + 1`, which
    /// always leaves remainder 1.
    #[test]
    fn non_divisible_hidden_by_heads_fails_validation(
        num_heads in 2usize..=16usize,
        base      in 1usize..=64usize,
    ) {
        let hidden_size = base * num_heads + 1; // remainder 1 ⇒ never divisible
        let mut cfg = BitNetConfig::default();
        cfg.model.hidden_size = hidden_size;
        cfg.model.num_heads   = num_heads;
        cfg.model.num_key_value_heads = 0;
        prop_assert!(
            cfg.validate().is_err(),
            "hidden_size={hidden_size} (not divisible by num_heads={num_heads}) must be rejected"
        );
    }
}

// ── GgufTensorType element-size / classification invariants ──────────────────

proptest! {
    /// For any number of elements, the byte count of an F32 tensor equals
    /// `n_elems * 4` — the element size must always be 4.
    #[test]
    fn f32_tensor_byte_count_is_four_per_element(n_elems in 0usize..=10_000usize) {
        let bytes = n_elems.saturating_mul(GgufTensorType::F32.element_size());
        prop_assert_eq!(
            bytes, n_elems * 4,
            "{} F32 elements should occupy {} bytes", n_elems, n_elems * 4
        );
    }

    /// For any number of elements, the byte count of an F16 tensor equals
    /// `n_elems * 2` — the element size must always be 2.
    #[test]
    fn f16_tensor_byte_count_is_two_per_element(n_elems in 0usize..=10_000usize) {
        let bytes = n_elems.saturating_mul(GgufTensorType::F16.element_size());
        prop_assert_eq!(
            bytes, n_elems * 2,
            "{} F16 elements should occupy {} bytes", n_elems, n_elems * 2
        );
    }

    /// F32, F16 and F64 must never be flagged as quantized types.
    #[test]
    fn unquantized_types_not_classified_as_quantized(
        dtype in prop_oneof![
            Just(GgufTensorType::F32),
            Just(GgufTensorType::F16),
            Just(GgufTensorType::F64),
        ],
    ) {
        prop_assert!(
            !dtype.is_quantized(),
            "{dtype:?} must not be classified as a quantized type"
        );
    }
}

// ── GGUF header invariants ───────────────────────────────────────────────────

proptest! {
    /// Any byte slice whose first four bytes are the GGUF magic must be
    /// accepted by `check_magic`.
    #[test]
    fn gguf_check_magic_accepts_valid_prefix(
        suffix in proptest::collection::vec(any::<u8>(), 0..=64usize),
    ) {
        let mut data = Vec::from(GGUF_MAGIC);
        data.extend_from_slice(&suffix);
        prop_assert!(check_magic(&data), "check_magic must accept data starting with GGUF magic");
    }

    /// Any four-byte header that differs from the GGUF magic must be rejected
    /// by `check_magic`.  The probability of a random 4-tuple equalling
    /// [G,G,U,F] is ~2.3 × 10⁻¹⁰, so `prop_assume!` almost never discards.
    #[test]
    fn gguf_check_magic_rejects_wrong_bytes(
        b0 in any::<u8>(),
        b1 in any::<u8>(),
        b2 in any::<u8>(),
        b3 in any::<u8>(),
    ) {
        prop_assume!([b0, b1, b2, b3] != [b'G', b'G', b'U', b'F']);
        let data = [b0, b1, b2, b3];
        prop_assert!(!check_magic(&data), "check_magic must reject non-GGUF magic bytes");
    }

    /// `read_version` must return the version embedded in the header for both
    /// supported GGUF versions (2 and 3).
    #[test]
    fn gguf_read_version_roundtrips_supported_versions(
        version in prop_oneof![Just(2u32), Just(3u32)],
        padding in proptest::collection::vec(any::<u8>(), 0..=16usize),
    ) {
        // Build a minimal valid-looking header: magic + version LE + padding
        let mut data = Vec::from(b"GGUF");
        data.extend_from_slice(&version.to_le_bytes());
        data.extend_from_slice(&padding);
        prop_assert_eq!(
            read_version(&data),
            Some(version),
            "read_version must return Some({}) for a header with that version", version
        );
    }
}

// ── Model loading path safety ────────────────────────────────────────────────

proptest! {
    /// Constructing a `std::path::Path` from any valid UTF-8 string must never
    /// panic or cause undefined behaviour.
    #[test]
    fn path_construction_from_valid_string_never_panics(
        s in "[a-zA-Z0-9/_.-]{0,128}",
    ) {
        let path = std::path::Path::new(&s);
        // These accessors must also not panic.
        let _ = path.to_str();
        let _ = path.extension();
        let _ = path.file_name();
        let _ = path.parent();
    }
}

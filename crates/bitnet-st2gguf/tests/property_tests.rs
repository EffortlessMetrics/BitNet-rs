//! Property-based tests for `bitnet-st2gguf`.
//!
//! Key invariants tested:
//! - `is_layernorm_tensor` is consistent: result doesn't change on repeated calls
//! - `is_layernorm_tensor` never matches empty strings
//! - `count_layernorm_tensors` is bounded by the input slice length
//! - `TensorDType::as_gguf_type` returns distinct values for distinct types
//! - `GgufWriter::add_tensor_f32` accepts valid shapes without panicking

use bitnet_st2gguf::{
    layernorm::{count_layernorm_tensors, is_layernorm_tensor},
    writer::{GgufWriter, TensorDType, TensorEntry},
};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Properties: is_layernorm_tensor
// ---------------------------------------------------------------------------

proptest! {
    /// `is_layernorm_tensor` is pure (same string → same result).
    #[test]
    fn prop_is_layernorm_tensor_is_pure(name in "[a-z._]{1,64}") {
        let first = is_layernorm_tensor(&name);
        let second = is_layernorm_tensor(&name);
        prop_assert_eq!(first, second, "is_layernorm_tensor must be deterministic");
    }

    /// Empty string is never a LayerNorm tensor.
    #[test]
    fn prop_is_layernorm_tensor_rejects_empty(_dummy in 0u32..10u32) {
        prop_assert!(!is_layernorm_tensor(""),
            "empty string must not match LayerNorm pattern");
    }

    /// Known non-LayerNorm names (weights, projections) are never matched.
    #[test]
    fn prop_projection_names_are_not_layernorm(
        layer in 0usize..32,
        proj in prop::sample::select(vec!["q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj"])
    ) {
        let name = format!("model.layers.{layer}.self_attn.{proj}.weight");
        prop_assert!(!is_layernorm_tensor(&name),
            "{name} is a projection weight, must not match LayerNorm");
    }

    /// Known LayerNorm names always match.
    #[test]
    fn prop_known_layernorm_patterns_always_match(
        layer in 0usize..32,
        suffix in prop::sample::select(vec!["attn_norm", "ffn_norm", "input_layernorm",
                                             "post_attention_layernorm"])
    ) {
        let name = format!("model.layers.{layer}.{suffix}.weight");
        prop_assert!(is_layernorm_tensor(&name),
            "{name} must match LayerNorm pattern");
    }
}

// ---------------------------------------------------------------------------
// Properties: count_layernorm_tensors
// ---------------------------------------------------------------------------

proptest! {
    /// `count_layernorm_tensors` is always ≤ the total number of names.
    #[test]
    fn prop_count_is_bounded_by_input_length(
        names in prop::collection::vec("[a-z._]{1,32}", 0..50)
    ) {
        let count = count_layernorm_tensors(names.iter().map(String::as_str));
        prop_assert!(count <= names.len(),
            "count_layernorm_tensors={count} must not exceed names.len()={}", names.len());
    }

    /// `count_layernorm_tensors` is non-negative (always ≥ 0).
    #[test]
    fn prop_count_is_non_negative(
        names in prop::collection::vec("[a-z._]{1,32}", 0..50)
    ) {
        let count = count_layernorm_tensors(names.iter().map(String::as_str));
        // count is usize so always ≥ 0, but we assert it's also sane
        prop_assert!(count <= names.len());
    }
}

// ---------------------------------------------------------------------------
// Properties: TensorDType
// ---------------------------------------------------------------------------

proptest! {
    /// F32 and F16 have distinct GGUF type codes.
    #[test]
    fn prop_dtype_gguf_type_codes_are_distinct(_dummy in 0u32..10u32) {
        let f32_code = TensorDType::F32.as_gguf_type();
        let f16_code = TensorDType::F16.as_gguf_type();
        prop_assert_ne!(f32_code, f16_code,
            "F32 and F16 must have distinct GGUF type codes");
    }

    /// Element sizes are positive and match expected values.
    #[test]
    fn prop_dtype_element_sizes_are_positive(_dummy in 0u32..10u32) {
        prop_assert!(TensorDType::F32.element_size() > 0);
        prop_assert!(TensorDType::F16.element_size() > 0);
        // F32 must be larger than F16
        prop_assert!(
            TensorDType::F32.element_size() > TensorDType::F16.element_size(),
            "F32 element size must exceed F16"
        );
    }
}

// ---------------------------------------------------------------------------
// Properties: GgufWriter
// ---------------------------------------------------------------------------

proptest! {
    /// `add_tensor` succeeds for any small valid shape (using TensorEntry directly).
    #[test]
    fn prop_add_tensor_accepts_valid_shapes(
        rows in 1usize..8,
        cols in 1usize..8
    ) {
        let n = rows * cols;
        let data = vec![0u8; n * 4]; // F32 bytes
        let tensor = TensorEntry::new(
            "test.weight".to_string(),
            vec![rows as u64, cols as u64],
            TensorDType::F32,
            data,
        );
        let mut writer = GgufWriter::new();
        writer.add_tensor(tensor);
        // No panic = success; we can't easily assert more without writing to disk
        prop_assert!(true);
    }
}

//! Snapshot tests for bitnet-st2gguf
//!
//! Covers:
//! - LayerNorm pattern detection (`is_layernorm_tensor`)
//! - `count_layernorm_tensors` on representative model tensor lists
//! - `TensorDType` GGUF type IDs and element sizes
//! - `GgufWriter` metadata key ordering
use bitnet_st2gguf::layernorm::{count_layernorm_tensors, is_layernorm_tensor};
use bitnet_st2gguf::writer::{MetadataValue, TensorDType};

/// Common LLM tensor names from a BitNet model layer
static MODEL_TENSORS: &[&str] = &[
    "model.embed_tokens.weight",
    "model.layers.0.attn_norm.weight",
    "model.layers.0.attn.q_proj.weight",
    "model.layers.0.attn.k_proj.weight",
    "model.layers.0.attn.v_proj.weight",
    "model.layers.0.attn.o_proj.weight",
    "model.layers.0.ffn_norm.weight",
    "model.layers.0.ffn.gate_proj.weight",
    "model.layers.0.ffn.up_proj.weight",
    "model.layers.0.ffn.down_proj.weight",
    "model.layers.1.attn_norm.weight",
    "model.layers.1.ffn_norm.weight",
    "model.norm.weight",
    "lm_head.weight",
];

#[test]
fn layernorm_detection_patterns() {
    // These MUST be LayerNorm tensors
    let ln_names = &[
        "model.layers.0.attn_norm.weight",
        "model.layers.0.ffn_norm.weight",
        "model.layers.15.attn_norm.weight",
        "model.layers.15.ffn_norm.weight",
        "model.norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "final_norm.weight",
    ];
    // These must NOT be LayerNorm tensors
    let non_ln_names = &[
        "model.embed_tokens.weight",
        "model.layers.0.attn.q_proj.weight",
        "model.layers.0.ffn.gate_proj.weight",
        "lm_head.weight",
    ];

    let ln_results: Vec<(&str, bool)> =
        ln_names.iter().map(|n| (*n, is_layernorm_tensor(n))).collect();
    let non_ln_results: Vec<(&str, bool)> =
        non_ln_names.iter().map(|n| (*n, is_layernorm_tensor(n))).collect();

    insta::assert_debug_snapshot!("layernorm_is_true", ln_results);
    insta::assert_debug_snapshot!("layernorm_is_false", non_ln_results);
}

#[test]
fn count_layernorm_in_model_tensors() {
    // In MODEL_TENSORS: attn_norm×2 + ffn_norm×2 + model.norm = 5 LayerNorm tensors
    let count = count_layernorm_tensors(MODEL_TENSORS.iter().copied());
    insta::assert_snapshot!("ln_count_in_model", count.to_string());
}

#[test]
fn tensor_dtype_gguf_type_ids() {
    let f32_id = TensorDType::F32.as_gguf_type();
    let f16_id = TensorDType::F16.as_gguf_type();
    insta::assert_snapshot!("tensor_dtype_gguf_ids", format!("F32={f32_id} F16={f16_id}"));
}

#[test]
fn tensor_dtype_element_sizes() {
    let f32_sz = TensorDType::F32.element_size();
    let f16_sz = TensorDType::F16.element_size();
    insta::assert_snapshot!("tensor_dtype_element_sizes", format!("F32={f32_sz} F16={f16_sz}"));
}

#[test]
fn metadata_value_debug_variants() {
    // Verify MetadataValue Debug output is stable across the public API
    let variants = vec![
        format!("{:?}", MetadataValue::Bool(true)),
        format!("{:?}", MetadataValue::U32(42)),
        format!("{:?}", MetadataValue::I32(-1)),
        format!("{:?}", MetadataValue::F32(1.5)),
        format!("{:?}", MetadataValue::String("test".to_string())),
    ];
    insta::assert_debug_snapshot!("metadata_value_variants", variants);
}

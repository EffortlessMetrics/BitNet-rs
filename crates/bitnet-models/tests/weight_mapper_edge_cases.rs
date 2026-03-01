//! Edge-case tests for weight_mapper module.
//!
//! Tests: normalize_vendor_key, dry_run_remap_names, and related public APIs.

use bitnet_models::weight_mapper::{dry_run_remap_names, normalize_vendor_key};

// ---------------------------------------------------------------------------
// normalize_vendor_key — GGUF blk.* attention patterns
// ---------------------------------------------------------------------------

#[test]
fn vendor_key_blk_attn_q() {
    let r = normalize_vendor_key("blk.0.attn_q.weight");
    assert_eq!(r, Some("layers.0.attention.q_proj.weight".to_string()));
}

#[test]
fn vendor_key_blk_attn_k() {
    let r = normalize_vendor_key("blk.5.attn_k.weight");
    assert_eq!(r, Some("layers.5.attention.k_proj.weight".to_string()));
}

#[test]
fn vendor_key_blk_attn_v() {
    let r = normalize_vendor_key("blk.12.attn_v.weight");
    assert_eq!(r, Some("layers.12.attention.v_proj.weight".to_string()));
}

#[test]
fn vendor_key_blk_attn_output() {
    let r = normalize_vendor_key("blk.3.attn_output.weight");
    assert_eq!(r, Some("layers.3.attention.o_proj.weight".to_string()));
}

#[test]
fn vendor_key_blk_attn_o_short() {
    let r = normalize_vendor_key("blk.7.attn_o.weight");
    assert_eq!(r, Some("layers.7.attention.o_proj.weight".to_string()));
}

// ---------------------------------------------------------------------------
// normalize_vendor_key — LLaMA-style attention patterns
// ---------------------------------------------------------------------------

#[test]
fn vendor_key_llama_wq() {
    let r = normalize_vendor_key("layers.0.self_attn.wq.weight");
    assert_eq!(r, Some("layers.0.attention.q_proj.weight".to_string()));
}

#[test]
fn vendor_key_llama_wk() {
    let r = normalize_vendor_key("layers.2.self_attn.wk.weight");
    assert_eq!(r, Some("layers.2.attention.k_proj.weight".to_string()));
}

#[test]
fn vendor_key_llama_wv() {
    let r = normalize_vendor_key("layers.10.self_attn.wv.weight");
    assert_eq!(r, Some("layers.10.attention.v_proj.weight".to_string()));
}

#[test]
fn vendor_key_llama_wo() {
    let r = normalize_vendor_key("layers.1.self_attn.wo.weight");
    assert_eq!(r, Some("layers.1.attention.o_proj.weight".to_string()));
}

#[test]
fn vendor_key_llama_model_prefix_wq() {
    let r = normalize_vendor_key("model.layers.3.self_attn.wq.weight");
    assert_eq!(r, Some("layers.3.attention.q_proj.weight".to_string()));
}

#[test]
fn vendor_key_llama_model_prefix_wk() {
    let r = normalize_vendor_key("model.layers.3.self_attn.wk.weight");
    assert_eq!(r, Some("layers.3.attention.k_proj.weight".to_string()));
}

#[test]
fn vendor_key_llama_model_prefix_wv() {
    let r = normalize_vendor_key("model.layers.3.self_attn.wv.weight");
    assert_eq!(r, Some("layers.3.attention.v_proj.weight".to_string()));
}

#[test]
fn vendor_key_llama_model_prefix_wo() {
    let r = normalize_vendor_key("model.layers.3.self_attn.wo.weight");
    assert_eq!(r, Some("layers.3.attention.o_proj.weight".to_string()));
}

// Without self_ prefix
#[test]
fn vendor_key_llama_attn_wq() {
    let r = normalize_vendor_key("layers.0.attn.wq.weight");
    assert_eq!(r, Some("layers.0.attention.q_proj.weight".to_string()));
}

// ---------------------------------------------------------------------------
// normalize_vendor_key — FFN / MLP patterns
// ---------------------------------------------------------------------------

#[test]
fn vendor_key_blk_ffn_gate() {
    let r = normalize_vendor_key("blk.0.ffn_gate.weight");
    assert_eq!(r, Some("layers.0.feed_forward.gate_proj.weight".to_string()));
}

#[test]
fn vendor_key_blk_ffn_gate_inp() {
    let r = normalize_vendor_key("blk.0.ffn_gate_inp.weight");
    assert_eq!(r, Some("layers.0.feed_forward.gate_proj.weight".to_string()));
}

#[test]
fn vendor_key_blk_ffn_up() {
    let r = normalize_vendor_key("blk.0.ffn_up.weight");
    assert_eq!(r, Some("layers.0.feed_forward.up_proj.weight".to_string()));
}

#[test]
fn vendor_key_blk_ffn_up_proj() {
    let r = normalize_vendor_key("blk.0.ffn_up_proj.weight");
    assert_eq!(r, Some("layers.0.feed_forward.up_proj.weight".to_string()));
}

#[test]
fn vendor_key_blk_ffn_down() {
    let r = normalize_vendor_key("blk.1.ffn_down.weight");
    assert_eq!(r, Some("layers.1.feed_forward.down_proj.weight".to_string()));
}

#[test]
fn vendor_key_blk_ffn_down_proj() {
    let r = normalize_vendor_key("blk.1.ffn_down_proj.weight");
    assert_eq!(r, Some("layers.1.feed_forward.down_proj.weight".to_string()));
}

#[test]
fn vendor_key_llama_ffn_w1() {
    let r = normalize_vendor_key("layers.0.feed_forward.w1.weight");
    assert_eq!(r, Some("layers.0.feed_forward.gate_proj.weight".to_string()));
}

#[test]
fn vendor_key_llama_ffn_w3() {
    let r = normalize_vendor_key("layers.0.feed_forward.w3.weight");
    assert_eq!(r, Some("layers.0.feed_forward.up_proj.weight".to_string()));
}

#[test]
fn vendor_key_llama_ffn_w2() {
    let r = normalize_vendor_key("layers.0.feed_forward.w2.weight");
    assert_eq!(r, Some("layers.0.feed_forward.down_proj.weight".to_string()));
}

#[test]
fn vendor_key_llama_mlp_gate_proj() {
    let r = normalize_vendor_key("layers.0.mlp.gate_proj.weight");
    assert_eq!(r, Some("layers.0.feed_forward.gate_proj.weight".to_string()));
}

#[test]
fn vendor_key_llama_mlp_up_proj() {
    let r = normalize_vendor_key("layers.0.mlp.up_proj.weight");
    assert_eq!(r, Some("layers.0.feed_forward.up_proj.weight".to_string()));
}

#[test]
fn vendor_key_llama_mlp_down_proj() {
    let r = normalize_vendor_key("layers.0.mlp.down_proj.weight");
    assert_eq!(r, Some("layers.0.feed_forward.down_proj.weight".to_string()));
}

#[test]
fn vendor_key_model_mlp_gate_proj() {
    let r = normalize_vendor_key("model.layers.4.mlp.gate_proj.weight");
    assert_eq!(r, Some("layers.4.feed_forward.gate_proj.weight".to_string()));
}

// ---------------------------------------------------------------------------
// normalize_vendor_key — Normalization patterns
// ---------------------------------------------------------------------------

#[test]
fn vendor_key_attn_norm() {
    let r = normalize_vendor_key("layers.0.attention_norm.weight");
    assert_eq!(r, Some("layers.0.attention_norm.weight".to_string()));
}

#[test]
fn vendor_key_input_layernorm() {
    let r = normalize_vendor_key("layers.0.input_layernorm.weight");
    assert_eq!(r, Some("layers.0.attention_norm.weight".to_string()));
}

#[test]
fn vendor_key_model_input_layernorm() {
    let r = normalize_vendor_key("model.layers.2.input_layernorm.weight");
    assert_eq!(r, Some("layers.2.attention_norm.weight".to_string()));
}

#[test]
fn vendor_key_ffn_norm() {
    let r = normalize_vendor_key("layers.0.ffn_norm.weight");
    assert_eq!(r, Some("layers.0.post_attention_layernorm.weight".to_string()));
}

#[test]
fn vendor_key_post_attention_layernorm() {
    let r = normalize_vendor_key("layers.0.post_attention_layernorm.weight");
    assert_eq!(r, Some("layers.0.post_attention_layernorm.weight".to_string()));
}

#[test]
fn vendor_key_model_post_attention_layernorm() {
    let r = normalize_vendor_key("model.layers.9.post_attention_layernorm.weight");
    assert_eq!(r, Some("layers.9.post_attention_layernorm.weight".to_string()));
}

// ---------------------------------------------------------------------------
// normalize_vendor_key — Unknown keys return None
// ---------------------------------------------------------------------------

#[test]
fn vendor_key_unknown_returns_none() {
    assert!(normalize_vendor_key("some.random.key").is_none());
}

#[test]
fn vendor_key_empty_returns_none() {
    assert!(normalize_vendor_key("").is_none());
}

#[test]
fn vendor_key_partial_match_returns_none() {
    // "blk.0" alone shouldn't match any pattern
    assert!(normalize_vendor_key("blk.0").is_none());
}

#[test]
fn vendor_key_wrong_suffix_returns_none() {
    assert!(normalize_vendor_key("blk.0.attn_q.bias").is_none());
}

// ---------------------------------------------------------------------------
// normalize_vendor_key — Multi-digit layer indices
// ---------------------------------------------------------------------------

#[test]
fn vendor_key_large_layer_index() {
    let r = normalize_vendor_key("blk.99.attn_q.weight");
    assert_eq!(r, Some("layers.99.attention.q_proj.weight".to_string()));
}

#[test]
fn vendor_key_three_digit_layer_index() {
    let r = normalize_vendor_key("blk.100.ffn_gate.weight");
    assert_eq!(r, Some("layers.100.feed_forward.gate_proj.weight".to_string()));
}

// ---------------------------------------------------------------------------
// dry_run_remap_names — basic behavior
// ---------------------------------------------------------------------------

#[test]
fn dry_run_empty_input() {
    let result = dry_run_remap_names(Vec::<String>::new());
    assert!(result.is_empty());
}

#[test]
fn dry_run_all_known_names_returns_empty() {
    let names = vec![
        "token_embd.weight".to_string(),
        "output.weight".to_string(),
        "output_norm.weight".to_string(),
        "blk.0.attn_q.weight".to_string(),
    ];
    let unmapped = dry_run_remap_names(names);
    assert!(unmapped.is_empty(), "Expected empty unmapped, got: {:?}", unmapped);
}

#[test]
fn dry_run_unknown_names_returns_them() {
    let names = vec!["unknown_tensor_1".to_string(), "another.unknown".to_string()];
    let unmapped = dry_run_remap_names(names);
    assert_eq!(unmapped.len(), 2);
    assert!(unmapped.contains(&"unknown_tensor_1".to_string()));
    assert!(unmapped.contains(&"another.unknown".to_string()));
}

#[test]
fn dry_run_mixed_known_and_unknown() {
    let names = vec![
        "token_embd.weight".to_string(),
        "output.weight".to_string(),
        "custom_layer.weight".to_string(),
    ];
    let unmapped = dry_run_remap_names(names);
    assert_eq!(unmapped.len(), 1);
    assert_eq!(unmapped[0], "custom_layer.weight");
}

#[test]
fn dry_run_blk_layer_tensors_all_mapped() {
    let names = vec![
        "blk.0.attn_q.weight".to_string(),
        "blk.0.attn_k.weight".to_string(),
        "blk.0.attn_v.weight".to_string(),
        "blk.0.attn_output.weight".to_string(),
        "blk.0.ffn_gate.weight".to_string(),
        "blk.0.ffn_up.weight".to_string(),
        "blk.0.ffn_down.weight".to_string(),
        "blk.0.attn_norm.weight".to_string(),
        "blk.0.ffn_norm.weight".to_string(),
        "blk.0.attn_sub_norm.weight".to_string(),
        "blk.0.ffn_sub_norm.weight".to_string(),
    ];
    let unmapped = dry_run_remap_names(names);
    assert!(unmapped.is_empty(), "Expected all blk tensors mapped, unmapped: {:?}", unmapped);
}

#[test]
fn dry_run_llama_style_all_mapped() {
    let names = vec![
        "layers.0.attention.wq.weight".to_string(),
        "layers.0.attention.wk.weight".to_string(),
        "layers.0.attention.wv.weight".to_string(),
        "layers.0.attention.wo.weight".to_string(),
        "layers.0.feed_forward.w1.weight".to_string(),
        "layers.0.feed_forward.w3.weight".to_string(),
        "layers.0.feed_forward.w2.weight".to_string(),
        "layers.0.attention_norm.weight".to_string(),
        "layers.0.post_attention_layernorm.weight".to_string(),
    ];
    let unmapped = dry_run_remap_names(names);
    assert!(unmapped.is_empty(), "Expected all llama tensors mapped, unmapped: {:?}", unmapped);
}

// ---------------------------------------------------------------------------
// Token embeddings and output layer variations
// ---------------------------------------------------------------------------

#[test]
fn dry_run_token_embedding_variations() {
    let names = vec![
        "token_embd.weight".to_string(),
        "tok_embeddings.weight".to_string(),
        "model.embed_tokens.weight".to_string(),
        "transformer.wte.weight".to_string(),
        "transformer.word_embeddings.weight".to_string(),
        "embeddings.word_embeddings.weight".to_string(),
        "embed.weight".to_string(),
        "embedding.weight".to_string(),
        "word_embeddings.weight".to_string(),
    ];
    let unmapped = dry_run_remap_names(names);
    assert!(
        unmapped.is_empty(),
        "Expected all embedding variations mapped, unmapped: {:?}",
        unmapped
    );
}

#[test]
fn dry_run_output_head_variations() {
    let names = vec![
        "output.weight".to_string(),
        "lm_head.weight".to_string(),
        "model.lm_head.weight".to_string(),
        "generator.weight".to_string(),
        "transformer.lm_head.weight".to_string(),
        "language_model_head.weight".to_string(),
        "head.weight".to_string(),
        "cls.weight".to_string(),
    ];
    let unmapped = dry_run_remap_names(names);
    assert!(
        unmapped.is_empty(),
        "Expected all output head variations mapped, unmapped: {:?}",
        unmapped
    );
}

#[test]
fn dry_run_final_norm_variations() {
    let names = vec![
        "output_norm.weight".to_string(),
        "norm.weight".to_string(),
        "model.norm.weight".to_string(),
        "transformer.ln_f.weight".to_string(),
        "ln_f.weight".to_string(),
        "final_norm.weight".to_string(),
        "final_layernorm.weight".to_string(),
        "final_rmsnorm.weight".to_string(),
    ];
    let unmapped = dry_run_remap_names(names);
    assert!(
        unmapped.is_empty(),
        "Expected all final norm variations mapped, unmapped: {:?}",
        unmapped
    );
}

// ---------------------------------------------------------------------------
// Self-attention style (HuggingFace self_attn.* patterns)
// ---------------------------------------------------------------------------

#[test]
fn dry_run_hf_self_attn_mapped() {
    let names = vec![
        "layers.0.self_attn.q_proj.weight".to_string(),
        "layers.0.self_attn.k_proj.weight".to_string(),
        "layers.0.self_attn.v_proj.weight".to_string(),
        "layers.0.self_attn.o_proj.weight".to_string(),
    ];
    let unmapped = dry_run_remap_names(names);
    assert!(unmapped.is_empty(), "Expected HF self_attn mapped, unmapped: {:?}", unmapped);
}

#[test]
fn dry_run_model_prefix_self_attn() {
    let names = vec![
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        "model.layers.0.self_attn.k_proj.weight".to_string(),
        "model.layers.0.self_attn.v_proj.weight".to_string(),
        "model.layers.0.self_attn.o_proj.weight".to_string(),
    ];
    let unmapped = dry_run_remap_names(names);
    assert!(
        unmapped.is_empty(),
        "Expected model.layers self_attn mapped, unmapped: {:?}",
        unmapped
    );
}

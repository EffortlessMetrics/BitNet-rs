// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Copyright 2025 BitNet Developers

//! Shared predicates for tensor naming conventions across GGUF, SafeTensors, and other formats.
//!
//! Centralizes the logic for identifying LayerNorm, RMSNorm, and projection weights
//! to ensure consistency between loaders, exporters, and validation tools.

#![allow(dead_code)]

/// Shared predicate for LayerNorm / RMSNorm gamma weights.
///
/// Keep this in sync with export & loader code across:
/// - `bitnet-models/formats/gguf/loader.rs`
/// - `bitnet-cli/commands/inspect.rs`
/// - `bitnet-st2gguf` (optional dependency)
///
/// # Examples
///
/// ```
/// use bitnet_models::names::is_layernorm_weight;
///
/// assert!(is_layernorm_weight("blk.0.attn_norm.weight"));
/// assert!(is_layernorm_weight("blk.5.ffn_norm.weight"));
/// assert!(is_layernorm_weight("final_norm.weight"));
/// assert!(!is_layernorm_weight("blk.0.attn_q.weight"));
/// ```
pub fn is_layernorm_weight(name: &str) -> bool {
    // LLaMA/HF-style
    name.ends_with(".attention_norm.weight")
        || name.ends_with(".ffn_norm.weight")
        || name.ends_with(".input_layernorm.weight")
        || name.ends_with(".post_attention_layernorm.weight")
        // BitNet-style
        || name.ends_with(".attn_norm.weight")
        || name.ends_with(".ffn_norm.weight")
        // Root-level
        || name.ends_with(".final_norm.weight")
        || name == "final_norm.weight"
        // Generic catch-all
        || name.ends_with(".rms_norm.weight")
        || name.ends_with(".norm.weight")
}

/// Shared predicate for linear projection weights (attention + feedforward).
///
/// Keep this in sync with export & loader code across:
/// - `bitnet-models/formats/gguf/loader.rs`
/// - `bitnet-cli/commands/inspect.rs`
/// - `bitnet-st2gguf` (optional dependency)
///
/// # Examples
///
/// ```
/// use bitnet_models::names::is_projection_weight;
///
/// assert!(is_projection_weight("blk.0.attn_q.weight"));
/// assert!(is_projection_weight("blk.0.attn_k.weight"));
/// assert!(is_projection_weight("blk.0.attn_output.weight"));
/// assert!(is_projection_weight("blk.0.ffn_gate.weight"));
/// assert!(!is_projection_weight("blk.0.attn_norm.weight"));
/// ```
pub fn is_projection_weight(name: &str) -> bool {
    // LLaMA/HF
    name.ends_with(".q_proj.weight")
        || name.ends_with(".k_proj.weight")
        || name.ends_with(".v_proj.weight")
        || name.ends_with(".o_proj.weight")
        || name.ends_with(".gate_proj.weight")
        || name.ends_with(".up_proj.weight")
        || name.ends_with(".down_proj.weight")
        // BitNet-style
        || name.ends_with(".attn_q.weight")
        || name.ends_with(".attn_k.weight")
        || name.ends_with(".attn_v.weight")
        || name.ends_with(".attn_output.weight")
        || name.ends_with(".ffn_gate.weight")
        || name.ends_with(".ffn_up.weight")
        || name.ends_with(".ffn_down.weight")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layernorm_predicates() {
        // BitNet-style
        assert!(is_layernorm_weight("blk.0.attn_norm.weight"));
        assert!(is_layernorm_weight("blk.5.ffn_norm.weight"));
        assert!(is_layernorm_weight("final_norm.weight"));

        // LLaMA/HF-style
        assert!(is_layernorm_weight("blk.0.input_layernorm.weight"));
        assert!(is_layernorm_weight("blk.0.post_attention_layernorm.weight"));

        // Generic
        assert!(is_layernorm_weight("blk.0.rms_norm.weight"));
        assert!(is_layernorm_weight("blk.0.norm.weight"));

        // Negatives
        assert!(!is_layernorm_weight("blk.0.attn_q.weight"));
        assert!(!is_layernorm_weight("token_embd.weight"));
    }

    #[test]
    fn test_projection_predicates() {
        // BitNet-style
        assert!(is_projection_weight("blk.0.attn_q.weight"));
        assert!(is_projection_weight("blk.0.attn_k.weight"));
        assert!(is_projection_weight("blk.0.attn_v.weight"));
        assert!(is_projection_weight("blk.0.attn_output.weight"));
        assert!(is_projection_weight("blk.0.ffn_gate.weight"));
        assert!(is_projection_weight("blk.0.ffn_up.weight"));
        assert!(is_projection_weight("blk.0.ffn_down.weight"));

        // LLaMA/HF-style
        assert!(is_projection_weight("blk.0.q_proj.weight"));
        assert!(is_projection_weight("blk.0.k_proj.weight"));
        assert!(is_projection_weight("blk.0.v_proj.weight"));
        assert!(is_projection_weight("blk.0.o_proj.weight"));
        assert!(is_projection_weight("blk.0.gate_proj.weight"));
        assert!(is_projection_weight("blk.0.up_proj.weight"));
        assert!(is_projection_weight("blk.0.down_proj.weight"));

        // Negatives
        assert!(!is_projection_weight("blk.0.attn_norm.weight"));
        assert!(!is_projection_weight("token_embd.weight"));
    }
}

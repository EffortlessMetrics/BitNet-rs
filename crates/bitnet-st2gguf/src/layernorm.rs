//! LayerNorm tensor pattern detection
//!
//! This module detects LayerNorm (and RMSNorm) tensors in model checkpoints
//! to ensure they are preserved as float (F16/F32) and never quantized.
//!
//! ## Pattern Detection
//!
//! LayerNorm tensors typically follow these naming patterns:
//! - `*.attn_norm.weight` / `*.attn_norm.bias`
//! - `*.ffn_norm.weight` / `*.ffn_norm.bias`
//! - `*.rms_norm.weight`
//! - `*.input_layernorm.weight`
//! - `*.post_attention_layernorm.weight`
//! - `ln_*.weight` / `ln_*.bias`
//! - `*norm*.weight` (with caution)
//!
//! ## Why This Matters
//!
//! LayerNorm statistics (gamma/scale, beta/bias) must remain in float format
//! because they are used for stabilizing activations. Quantizing them can cause
//! numerical instability and inference quality degradation.

use regex::Regex;
use std::sync::OnceLock;

/// Compile the LayerNorm pattern regex once
fn layernorm_pattern() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| {
        // Pattern breakdown:
        // - Match end of string or path separator before norm component
        // - Match various norm naming conventions:
        //   * attn_norm, ffn_norm, rms_norm
        //   * input_layernorm, post_attention_layernorm
        //   * pre_norm, post_norm
        //   * ln_<digit>, ln.<anything>
        //   * Any norm/layernorm at end of path
        // - Match .weight or .bias suffix
        Regex::new(
            r"(?:^|[\.\/])(?:(?:[\w_]*(?:attn|ffn|rms|input|post|pre)[\w_]*)?(?:layer)?norm(?:_|\.)?\d*|ln(?:_|\.)?\d*)\.(?:weight|bias)$"
        )
        .expect("layernorm pattern regex should compile")
    })
}

/// Check if a tensor name matches LayerNorm patterns
///
/// # Examples
///
/// ```
/// use bitnet_st2gguf::layernorm::is_layernorm_tensor;
///
/// assert!(is_layernorm_tensor("model.layers.0.attn_norm.weight"));
/// assert!(is_layernorm_tensor("model.layers.15.ffn_norm.weight"));
/// assert!(is_layernorm_tensor("model.norm.weight"));
/// assert!(is_layernorm_tensor("transformer.ln_1.weight"));
/// assert!(is_layernorm_tensor("model.layers.0.input_layernorm.weight"));
///
/// assert!(!is_layernorm_tensor("model.layers.0.attn.q_proj.weight"));
/// assert!(!is_layernorm_tensor("model.embed_tokens.weight"));
/// ```
pub fn is_layernorm_tensor(name: &str) -> bool {
    layernorm_pattern().is_match(name)
}

/// Count LayerNorm tensors in a list of names
pub fn count_layernorm_tensors<'a, I>(names: I) -> usize
where
    I: IntoIterator<Item = &'a str>,
{
    names.into_iter().filter(|name| is_layernorm_tensor(name)).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitnet_patterns() {
        // BitNet-style normalization
        assert!(is_layernorm_tensor("model.layers.0.attn_norm.weight"));
        assert!(is_layernorm_tensor("model.layers.0.ffn_norm.weight"));
        assert!(is_layernorm_tensor("model.norm.weight"));
    }

    #[test]
    fn test_llama_patterns() {
        // LLaMA/Mistral-style normalization
        assert!(is_layernorm_tensor("model.layers.0.input_layernorm.weight"));
        assert!(is_layernorm_tensor("model.layers.0.post_attention_layernorm.weight"));
        assert!(is_layernorm_tensor("model.norm.weight"));
    }

    #[test]
    fn test_gpt_patterns() {
        // GPT-style normalization
        assert!(is_layernorm_tensor("transformer.ln_1.weight"));
        assert!(is_layernorm_tensor("transformer.ln_1.bias"));
        assert!(is_layernorm_tensor("transformer.ln_2.weight"));
        assert!(is_layernorm_tensor("transformer.h.0.ln_1.weight"));
    }

    #[test]
    fn test_rms_norm_patterns() {
        // RMSNorm patterns
        assert!(is_layernorm_tensor("model.layers.0.rms_norm.weight"));
        assert!(is_layernorm_tensor("decoder.rms_norm.weight"));
    }

    #[test]
    fn test_non_layernorm_tensors() {
        // These should NOT match
        assert!(!is_layernorm_tensor("model.layers.0.attn.q_proj.weight"));
        assert!(!is_layernorm_tensor("model.layers.0.attn.k_proj.weight"));
        assert!(!is_layernorm_tensor("model.layers.0.attn.v_proj.weight"));
        assert!(!is_layernorm_tensor("model.layers.0.attn.o_proj.weight"));
        assert!(!is_layernorm_tensor("model.layers.0.mlp.gate_proj.weight"));
        assert!(!is_layernorm_tensor("model.layers.0.mlp.up_proj.weight"));
        assert!(!is_layernorm_tensor("model.layers.0.mlp.down_proj.weight"));
        assert!(!is_layernorm_tensor("model.embed_tokens.weight"));
        assert!(!is_layernorm_tensor("lm_head.weight"));
    }

    #[test]
    fn test_edge_cases() {
        // Potential edge cases
        assert!(is_layernorm_tensor("pre_norm.weight"));
        assert!(is_layernorm_tensor("post_norm.weight"));
        assert!(is_layernorm_tensor("ln.weight"));
        assert!(is_layernorm_tensor("norm.weight"));
    }

    #[test]
    fn test_count() {
        let names = [
            "model.layers.0.attn_norm.weight",
            "model.layers.0.attn.q_proj.weight",
            "model.layers.0.ffn_norm.weight",
            "model.norm.weight",
        ];
        assert_eq!(count_layernorm_tensors(names.iter().copied()), 3);
    }
}

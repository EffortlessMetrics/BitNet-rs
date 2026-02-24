// SPDX-License-Identifier: MIT OR Apache-2.0
//! LayerNorm tensor name detection patterns.
//!
//! Provides a regex-based predicate that identifies LayerNorm / RMSNorm gamma
//! weight tensors by name. Used consistently across:
//! - GGUF model inspection (bitnet-cli)
//! - SafeTensors export tools (bitnet-st-tools)

use std::sync::OnceLock;

use regex::Regex;

/// Returns `true` if `name` looks like a LayerNorm / RMSNorm gamma weight.
///
/// Matches variants including `attn_norm`, `ffn_norm`, `ffn_layernorm`,
/// `input_layernorm`, `post_attention_layernorm`, `final_layernorm`, and
/// the generic `norm` suffix.
///
/// # Examples
///
/// ```
/// use bitnet_validation::is_ln_gamma;
///
/// assert!(is_ln_gamma("blk.0.attn_norm.weight"));
/// assert!(is_ln_gamma("blk.3.ffn_layernorm.weight"));
/// assert!(is_ln_gamma("final_norm.weight"));
/// assert!(!is_ln_gamma("blk.0.attn_q.weight"));
/// assert!(!is_ln_gamma("output.weight"));
/// ```
pub fn is_ln_gamma(name: &str) -> bool {
    if !name.ends_with(".weight") {
        return false;
    }

    static RE: OnceLock<Regex> = OnceLock::new();
    let regex = RE.get_or_init(|| {
        Regex::new(
            r"(?x)
            (?:^|[./])
            (?:attn_norm|ffn_norm|ffn_layernorm|rms_norm|
               input_layernorm|post_attention_layernorm|
               final_layernorm|final_norm|norm)
            \.weight$
            ",
        )
        .expect("Failed to compile LayerNorm regex pattern")
    });

    regex.is_match(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attn_norm() {
        assert!(is_ln_gamma("blk.0.attn_norm.weight"));
    }

    #[test]
    fn test_ffn_norm() {
        assert!(is_ln_gamma("blk.5.ffn_norm.weight"));
    }

    #[test]
    fn test_ffn_layernorm() {
        assert!(is_ln_gamma("blk.3.ffn_layernorm.weight"));
    }

    #[test]
    fn test_input_layernorm() {
        assert!(is_ln_gamma("model.layers.0.input_layernorm.weight"));
    }

    #[test]
    fn test_post_attention_layernorm() {
        assert!(is_ln_gamma("model.layers.0.post_attention_layernorm.weight"));
    }

    #[test]
    fn test_final_norm() {
        assert!(is_ln_gamma("final_norm.weight"));
    }

    #[test]
    fn test_final_layernorm() {
        assert!(is_ln_gamma("model.final_layernorm.weight"));
    }

    #[test]
    fn test_generic_norm() {
        assert!(is_ln_gamma("model.norm.weight"));
    }

    #[test]
    fn test_not_projection_weight() {
        assert!(!is_ln_gamma("blk.0.attn_q.weight"));
    }

    #[test]
    fn test_not_output_weight() {
        assert!(!is_ln_gamma("output.weight"));
    }

    #[test]
    fn test_not_no_weight_suffix() {
        assert!(!is_ln_gamma("blk.0.attn_norm.bias"));
    }

    #[test]
    fn test_not_embedding() {
        assert!(!is_ln_gamma("token_embd.weight"));
    }
}

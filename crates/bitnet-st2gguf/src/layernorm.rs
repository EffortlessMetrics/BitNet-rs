//! LayerNorm tensor pattern detection
//!
//! This module re-exports the shared LayerNorm name predicates from `bitnet-models`
//! to ensure consistency between GGUF export (st2gguf) and loading (bitnet-models).
//!
//! ## Why This Matters
//!
//! LayerNorm statistics (gamma/scale, beta/bias) must remain in float format
//! because they are used for stabilizing activations. Quantizing them can cause
//! numerical instability and inference quality degradation.
//!
//! By using the exact same predicates as the loader, we guarantee that:
//! - What we export as float will be recognized as LayerNorm during loading
//! - Validation checks will be consistent across export and load paths

// Re-export the shared predicate from bitnet-models for consistency
pub use bitnet_models::names::is_layernorm_weight;

/// Check if a tensor name matches LayerNorm patterns
///
/// This is an alias for `is_layernorm_weight` to maintain API compatibility.
///
/// # Examples
///
/// ```
/// use bitnet_st2gguf::layernorm::is_layernorm_tensor;
///
/// assert!(is_layernorm_tensor("model.layers.0.attn_norm.weight"));
/// assert!(is_layernorm_tensor("model.layers.15.ffn_norm.weight"));
/// assert!(is_layernorm_tensor("model.norm.weight"));
/// assert!(is_layernorm_tensor("model.layers.0.input_layernorm.weight"));
///
/// assert!(!is_layernorm_tensor("model.layers.0.attn.q_proj.weight"));
/// assert!(!is_layernorm_tensor("model.embed_tokens.weight"));
/// ```
#[inline]
pub fn is_layernorm_tensor(name: &str) -> bool {
    is_layernorm_weight(name)
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
    fn test_alias_delegates_to_shared_predicate() {
        // Verify the alias correctly delegates to the shared implementation
        // (Full pattern tests are in bitnet-models::names)
        assert!(is_layernorm_tensor("model.layers.0.attn_norm.weight"));
        assert!(is_layernorm_tensor("final_norm.weight"));
        assert!(!is_layernorm_tensor("model.layers.0.attn.q_proj.weight"));
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

//! Exporter weight-name alias normalization helpers.
//!
//! This crate owns low-level string drift normalization for weight keys,
//! independent from tensor parsing/mapping logic.

use std::borrow::Cow;

/// Normalize known exporter naming drifts to canonical BitNet naming.
pub fn normalize_weight_alias(name: &str) -> Cow<'_, str> {
    if name.contains("attention_sub_norm") {
        return Cow::Owned(name.replace("attention_sub_norm", "attn_sub_norm"));
    }
    if name.contains("mlp_sub_layernorm") {
        return Cow::Owned(name.replace("mlp_sub_layernorm", "ffn_sub_norm"));
    }
    Cow::Borrowed(name)
}

#[cfg(test)]
mod tests {
    use super::normalize_weight_alias;

    #[test]
    fn normalizes_attention_sub_norm_alias() {
        let normalized = normalize_weight_alias("layers.0.attention_sub_norm.weight");
        assert_eq!(normalized.as_ref(), "layers.0.attn_sub_norm.weight");
    }

    #[test]
    fn normalizes_mlp_sub_layernorm_alias() {
        let normalized = normalize_weight_alias("layers.3.mlp_sub_layernorm.weight");
        assert_eq!(normalized.as_ref(), "layers.3.ffn_sub_norm.weight");
    }

    #[test]
    fn leaves_canonical_name_unchanged() {
        let input = "layers.1.attn_sub_norm.weight";
        let normalized = normalize_weight_alias(input);
        assert_eq!(normalized.as_ref(), input);
    }
}

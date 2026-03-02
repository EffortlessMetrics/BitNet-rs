//! Reusable GGUF metadata extraction helpers for model-shape parsing.

use anyhow::{Context, Result, anyhow};
use bitnet_common::BitNetConfig;
use std::collections::HashMap;

/// Minimal model-shape fields extracted from GGUF key/value metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufModelShape {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
}

/// Extract core model shape fields from GGUF metadata.
///
/// Supports both `bitnet-b1.58.*` and `llama.*` names for compatibility.
pub fn extract_model_shape(metadata: &HashMap<String, String>) -> Result<GgufModelShape> {
    Ok(GgufModelShape {
        vocab_size: parse_required_usize(
            metadata,
            "vocab_size",
            &["bitnet-b1.58.vocab_size", "llama.vocab_size"],
        )?,
        hidden_size: parse_required_usize(
            metadata,
            "hidden_size",
            &["bitnet-b1.58.embedding_length", "llama.embedding_length", "llama.hidden_size"],
        )?,
        num_layers: parse_required_usize(
            metadata,
            "num_layers",
            &["bitnet-b1.58.block_count", "llama.block_count", "llama.layer_count"],
        )?,
        num_heads: parse_required_usize(
            metadata,
            "num_heads",
            &[
                "bitnet-b1.58.attention.head_count",
                "llama.attention.head_count",
                "llama.head_count",
            ],
        )?,
        intermediate_size: parse_optional_usize(
            metadata,
            "intermediate_size",
            &["bitnet-b1.58.feed_forward_length", "llama.feed_forward_length"],
        )?
        .unwrap_or(0),
    })
}

/// Build `BitNetConfig` from GGUF metadata, preserving defaults for absent optional fields.
pub fn bitnet_config_from_metadata(metadata: &HashMap<String, String>) -> Result<BitNetConfig> {
    let shape = extract_model_shape(metadata)?;
    let mut config = BitNetConfig::default();
    config.model.vocab_size = shape.vocab_size;
    config.model.hidden_size = shape.hidden_size;
    config.model.num_layers = shape.num_layers;
    config.model.num_heads = shape.num_heads;

    if shape.intermediate_size > 0 {
        config.model.intermediate_size = shape.intermediate_size;
    }

    // BitNet default runtime assumptions for GGUF-hosted quantized checkpoints.
    config.quantization.block_size = 128;

    Ok(config)
}

fn parse_required_usize(
    metadata: &HashMap<String, String>,
    field: &str,
    keys: &[&str],
) -> Result<usize> {
    parse_optional_usize(metadata, field, keys)?.ok_or_else(|| {
        anyhow!("missing required GGUF field for {field} (tried: {})", keys.join(", "))
    })
}

fn parse_optional_usize(
    metadata: &HashMap<String, String>,
    field: &str,
    keys: &[&str],
) -> Result<Option<usize>> {
    for key in keys {
        if let Some(value) = metadata.get(*key) {
            let parsed = value
                .parse::<usize>()
                .with_context(|| format!("Failed to parse {field} from key '{key}'"))?;
            return Ok(Some(parsed));
        }
    }

    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_model_shape_prefers_bitnet_keys_when_present() {
        let metadata = HashMap::from([
            ("bitnet-b1.58.vocab_size".to_string(), "32000".to_string()),
            ("bitnet-b1.58.embedding_length".to_string(), "2048".to_string()),
            ("bitnet-b1.58.block_count".to_string(), "24".to_string()),
            ("bitnet-b1.58.attention.head_count".to_string(), "16".to_string()),
            ("bitnet-b1.58.feed_forward_length".to_string(), "8192".to_string()),
            ("llama.vocab_size".to_string(), "99999".to_string()),
        ]);

        let shape = extract_model_shape(&metadata).expect("shape should parse");

        assert_eq!(shape.vocab_size, 32000);
        assert_eq!(shape.hidden_size, 2048);
        assert_eq!(shape.num_layers, 24);
        assert_eq!(shape.num_heads, 16);
        assert_eq!(shape.intermediate_size, 8192);
    }

    #[test]
    fn extract_model_shape_uses_llama_fallbacks() {
        let metadata = HashMap::from([
            ("llama.vocab_size".to_string(), "128256".to_string()),
            ("llama.embedding_length".to_string(), "4096".to_string()),
            ("llama.block_count".to_string(), "32".to_string()),
            ("llama.attention.head_count".to_string(), "32".to_string()),
        ]);

        let shape = extract_model_shape(&metadata).expect("shape should parse");
        assert_eq!(shape.intermediate_size, 0);
        assert_eq!(shape.num_layers, 32);
    }

    #[test]
    fn bitnet_config_from_metadata_sets_quantization_defaults() {
        let metadata = HashMap::from([
            ("llama.vocab_size".to_string(), "32000".to_string()),
            ("llama.embedding_length".to_string(), "4096".to_string()),
            ("llama.block_count".to_string(), "32".to_string()),
            ("llama.attention.head_count".to_string(), "32".to_string()),
            ("llama.feed_forward_length".to_string(), "11008".to_string()),
        ]);

        let config = bitnet_config_from_metadata(&metadata).expect("config should parse");

        assert_eq!(config.model.vocab_size, 32000);
        assert_eq!(config.model.intermediate_size, 11008);
        assert_eq!(config.quantization.block_size, 128);
    }
}

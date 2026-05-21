//! Model metadata extraction from various sources.
//!
//! Extracts architecture info from GGUF headers, HF config.json, etc.

use std::collections::HashMap;

/// Extracted model metadata.
#[derive(Debug, Clone, Default)]
pub struct ModelMetadata {
    pub model_type: Option<String>,
    pub architecture: Option<String>,
    pub hidden_size: Option<usize>,
    pub num_layers: Option<usize>,
    pub num_heads: Option<usize>,
    pub num_kv_heads: Option<usize>,
    pub vocab_size: Option<usize>,
    pub max_position: Option<usize>,
    pub intermediate_size: Option<usize>,
    pub activation: Option<String>,
    pub norm_type: Option<String>,
    pub rope_base: Option<f32>,
    pub tie_word_embeddings: Option<bool>,
    pub extra: HashMap<String, String>,
}

impl ModelMetadata {
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if minimum required fields are present.
    pub fn is_complete(&self) -> bool {
        self.hidden_size.is_some()
            && self.num_layers.is_some()
            && self.num_heads.is_some()
            && self.vocab_size.is_some()
    }

    /// Missing required fields.
    pub fn missing_fields(&self) -> Vec<&'static str> {
        let mut missing = Vec::new();
        if self.hidden_size.is_none() {
            missing.push("hidden_size");
        }
        if self.num_layers.is_none() {
            missing.push("num_layers");
        }
        if self.num_heads.is_none() {
            missing.push("num_heads");
        }
        if self.vocab_size.is_none() {
            missing.push("vocab_size");
        }
        missing
    }

    /// Merge with another metadata, preferring non-None values from other.
    pub fn merge(&mut self, other: &ModelMetadata) {
        if other.model_type.is_some() {
            self.model_type.clone_from(&other.model_type);
        }
        if other.architecture.is_some() {
            self.architecture.clone_from(&other.architecture);
        }
        if other.hidden_size.is_some() {
            self.hidden_size = other.hidden_size;
        }
        if other.num_layers.is_some() {
            self.num_layers = other.num_layers;
        }
        if other.num_heads.is_some() {
            self.num_heads = other.num_heads;
        }
        if other.num_kv_heads.is_some() {
            self.num_kv_heads = other.num_kv_heads;
        }
        if other.vocab_size.is_some() {
            self.vocab_size = other.vocab_size;
        }
        if other.max_position.is_some() {
            self.max_position = other.max_position;
        }
        if other.intermediate_size.is_some() {
            self.intermediate_size = other.intermediate_size;
        }
        if other.activation.is_some() {
            self.activation.clone_from(&other.activation);
        }
        if other.norm_type.is_some() {
            self.norm_type.clone_from(&other.norm_type);
        }
        if other.rope_base.is_some() {
            self.rope_base = other.rope_base;
        }
        if other.tie_word_embeddings.is_some() {
            self.tie_word_embeddings = other.tie_word_embeddings;
        }
        for (k, v) in &other.extra {
            self.extra.insert(k.clone(), v.clone());
        }
    }

    /// Compute head dimension.
    pub fn head_dim(&self) -> Option<usize> {
        exact_division(self.hidden_size, self.num_heads)
    }

    /// GQA group count (num_heads / num_kv_heads).
    pub fn gqa_groups(&self) -> Option<usize> {
        exact_division(self.num_heads, self.num_kv_heads)
    }
}

fn exact_division(dividend: Option<usize>, divisor: Option<usize>) -> Option<usize> {
    let (Some(dividend), Some(divisor)) = (dividend, divisor) else {
        return None;
    };

    if divisor == 0 || dividend % divisor != 0 {
        return None;
    }

    Some(dividend / divisor)
}

/// Extract metadata from a key-value map (GGUF-style).
pub fn from_kv_pairs(pairs: &HashMap<String, String>) -> ModelMetadata {
    let mut m = ModelMetadata::new();

    for (k, v) in pairs {
        let key = k.to_lowercase();
        match key.as_str() {
            k if k.contains("hidden_size") || k.contains("embedding_length") => {
                m.hidden_size = v.parse().ok();
            }
            k if k.contains("num_hidden_layers") || k.contains("block_count") => {
                m.num_layers = v.parse().ok();
            }
            k if k.contains("num_key_value_heads") || k.contains("head_count_kv") => {
                m.num_kv_heads = v.parse().ok();
            }
            k if k.contains("num_attention_heads") || k.contains("head_count") => {
                m.num_heads = v.parse().ok();
            }
            k if k.contains("vocab_size") => {
                m.vocab_size = v.parse().ok();
            }
            k if k.contains("max_position") || k.contains("context_length") => {
                m.max_position = v.parse().ok();
            }
            k if k.contains("intermediate_size") || k.contains("feed_forward_length") => {
                m.intermediate_size = v.parse().ok();
            }
            k if k.contains("model_type") || k.contains("architecture") => {
                m.model_type = Some(v.clone());
            }
            k if k.contains("hidden_act") || k.contains("activation") => {
                m.activation = Some(v.clone());
            }
            _ => {
                m.extra.insert(k.to_string(), v.clone());
            }
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_metadata() {
        let m = ModelMetadata::new();
        assert!(!m.is_complete());
    }

    #[test]
    fn test_complete() {
        let m = ModelMetadata {
            hidden_size: Some(4096),
            num_layers: Some(32),
            num_heads: Some(32),
            vocab_size: Some(32000),
            ..Default::default()
        };
        assert!(m.is_complete());
    }

    #[test]
    fn test_missing_fields() {
        let m = ModelMetadata { hidden_size: Some(4096), ..Default::default() };
        let missing = m.missing_fields();
        assert!(missing.contains(&"num_layers"));
        assert!(!missing.contains(&"hidden_size"));
    }

    #[test]
    fn test_head_dim() {
        let m =
            ModelMetadata { hidden_size: Some(4096), num_heads: Some(32), ..Default::default() };
        assert_eq!(m.head_dim(), Some(128));
    }

    #[test]
    fn test_gqa_groups() {
        let m = ModelMetadata { num_heads: Some(40), num_kv_heads: Some(10), ..Default::default() };
        assert_eq!(m.gqa_groups(), Some(4));
    }

    #[test]
    fn test_head_dim_non_divisible_returns_none() {
        let m =
            ModelMetadata { hidden_size: Some(4097), num_heads: Some(32), ..Default::default() };
        assert_eq!(m.head_dim(), None);
    }

    #[test]
    fn test_gqa_groups_non_divisible_returns_none() {
        let m = ModelMetadata { num_heads: Some(41), num_kv_heads: Some(10), ..Default::default() };
        assert_eq!(m.gqa_groups(), None);
    }

    #[test]
    fn test_exact_division_zero_divisor_returns_none() {
        assert_eq!(exact_division(Some(10), Some(0)), None);
    }

    #[test]
    fn test_merge() {
        let mut a = ModelMetadata { hidden_size: Some(4096), ..Default::default() };
        let b = ModelMetadata { num_layers: Some(32), ..Default::default() };
        a.merge(&b);
        assert_eq!(a.hidden_size, Some(4096));
        assert_eq!(a.num_layers, Some(32));
    }

    #[test]
    fn test_merge_overwrite() {
        let mut a = ModelMetadata { hidden_size: Some(4096), ..Default::default() };
        let b = ModelMetadata { hidden_size: Some(5120), ..Default::default() };
        a.merge(&b);
        assert_eq!(a.hidden_size, Some(5120));
    }

    #[test]
    fn test_from_kv_pairs() {
        let mut pairs = HashMap::new();
        pairs.insert("hidden_size".into(), "5120".into());
        pairs.insert("num_hidden_layers".into(), "40".into());
        pairs.insert("num_attention_heads".into(), "40".into());
        pairs.insert("vocab_size".into(), "100352".into());
        let m = from_kv_pairs(&pairs);
        assert_eq!(m.hidden_size, Some(5120));
        assert_eq!(m.num_layers, Some(40));
        assert_eq!(m.vocab_size, Some(100352));
    }

    #[test]
    fn test_from_kv_gguf_style() {
        let mut pairs = HashMap::new();
        pairs.insert("llama.embedding_length".into(), "2560".into());
        pairs.insert("llama.block_count".into(), "30".into());
        pairs.insert("llama.head_count".into(), "20".into());
        pairs.insert("llama.head_count_kv".into(), "5".into());
        let m = from_kv_pairs(&pairs);
        assert_eq!(m.hidden_size, Some(2560));
        assert_eq!(m.num_layers, Some(30));
        assert_eq!(m.num_kv_heads, Some(5));
    }

    #[test]
    fn test_from_kv_activation() {
        let mut pairs = HashMap::new();
        pairs.insert("hidden_act".into(), "silu".into());
        let m = from_kv_pairs(&pairs);
        assert_eq!(m.activation, Some("silu".into()));
    }

    #[test]
    fn test_from_kv_extra() {
        let mut pairs = HashMap::new();
        pairs.insert("custom_field".into(), "value".into());
        let m = from_kv_pairs(&pairs);
        assert_eq!(m.extra.get("custom_field").unwrap(), "value");
    }

    #[test]
    fn test_head_dim_zero_heads() {
        let m = ModelMetadata { hidden_size: Some(4096), num_heads: Some(0), ..Default::default() };
        assert_eq!(m.head_dim(), None);
    }

    #[test]
    fn test_gqa_no_kv_heads() {
        let m = ModelMetadata { num_heads: Some(32), ..Default::default() };
        assert_eq!(m.gqa_groups(), None);
    }

    #[test]
    fn test_merge_extra() {
        let mut a = ModelMetadata::new();
        let mut b = ModelMetadata::new();
        b.extra.insert("key".into(), "val".into());
        a.merge(&b);
        assert_eq!(a.extra.get("key").unwrap(), "val");
    }
}

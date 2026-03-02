//! Model layer inspection for architecture analysis.
//!
//! Inspects tensor names to reconstruct model architecture
//! (layer count, component types, naming patterns).

use std::collections::{BTreeMap, BTreeSet};

/// A discovered layer in the model.
#[derive(Debug, Clone)]
pub struct LayerInfo {
    pub index: usize,
    pub components: BTreeSet<String>,
    pub tensor_count: usize,
}

/// Overall model structure from tensor names.
#[derive(Debug, Clone)]
pub struct ModelStructure {
    pub num_layers: usize,
    pub layers: Vec<LayerInfo>,
    pub non_layer_tensors: Vec<String>,
    pub tensor_name_pattern: NamingPattern,
}

/// Naming convention used in the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NamingPattern {
    /// `model.layers.{n}.self_attn.q_proj.weight` (HuggingFace style)
    HuggingFace,
    /// `blk.{n}.attn_q.weight` (GGUF style)
    Gguf,
    /// Unknown/mixed naming.
    Unknown,
}

impl std::fmt::Display for NamingPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HuggingFace => write!(f, "HuggingFace"),
            Self::Gguf => write!(f, "GGUF"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Inspect tensor names and reconstruct model structure.
pub fn inspect_layers(tensor_names: &[String]) -> ModelStructure {
    let mut layer_map: BTreeMap<usize, BTreeSet<String>> = BTreeMap::new();
    let mut non_layer = Vec::new();
    let mut hf_count = 0u32;
    let mut gguf_count = 0u32;

    for name in tensor_names {
        if let Some(idx) = extract_layer_index(name) {
            let component = extract_component(name);
            layer_map.entry(idx).or_default().insert(component);
            if name.contains("model.layers.") {
                hf_count += 1;
            }
            if name.starts_with("blk.") {
                gguf_count += 1;
            }
        } else {
            non_layer.push(name.clone());
        }
    }

    let layers: Vec<LayerInfo> = layer_map
        .into_iter()
        .map(|(index, components)| {
            let tensor_count = components.len();
            LayerInfo {
                index,
                components,
                tensor_count,
            }
        })
        .collect();

    let num_layers = layers.len();
    let pattern = if hf_count > gguf_count && hf_count > 0 {
        NamingPattern::HuggingFace
    } else if gguf_count > 0 {
        NamingPattern::Gguf
    } else {
        NamingPattern::Unknown
    };

    ModelStructure {
        num_layers,
        layers,
        non_layer_tensors: non_layer,
        tensor_name_pattern: pattern,
    }
}

/// Extract layer index from a tensor name.
fn extract_layer_index(name: &str) -> Option<usize> {
    for prefix in &[
        "model.layers.",
        "blk.",
        "layers.",
        "encoder.layer.",
        "decoder.layer.",
    ] {
        if let Some(rest) = name.strip_prefix(prefix) {
            if let Some(idx_str) = rest.split('.').next() {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    return Some(idx);
                }
            }
        }
    }
    None
}

/// Extract component name from a tensor name.
fn extract_component(name: &str) -> String {
    let parts: Vec<&str> = name.split('.').collect();
    for (i, &part) in parts.iter().enumerate() {
        if part.parse::<usize>().is_ok() && i + 1 < parts.len() {
            return parts[i + 1..].join(".");
        }
    }
    name.to_string()
}

/// Common components expected in transformer layers.
pub fn expected_components() -> Vec<&'static str> {
    vec![
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    ]
}

/// Check what expected components are missing from a layer.
pub fn missing_components(layer: &LayerInfo) -> Vec<String> {
    expected_components()
        .into_iter()
        .filter(|c| !layer.components.contains(*c))
        .map(|c| c.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hf_tensor_names() -> Vec<String> {
        vec![
            "model.embed_tokens.weight".into(),
            "model.layers.0.self_attn.q_proj.weight".into(),
            "model.layers.0.self_attn.k_proj.weight".into(),
            "model.layers.0.mlp.gate_proj.weight".into(),
            "model.layers.1.self_attn.q_proj.weight".into(),
            "model.layers.1.mlp.gate_proj.weight".into(),
            "lm_head.weight".into(),
        ]
    }

    fn gguf_tensor_names() -> Vec<String> {
        vec![
            "token_embd.weight".into(),
            "blk.0.attn_q.weight".into(),
            "blk.0.attn_k.weight".into(),
            "blk.1.attn_q.weight".into(),
            "output.weight".into(),
        ]
    }

    #[test]
    fn test_inspect_hf() {
        let s = inspect_layers(&hf_tensor_names());
        assert_eq!(s.num_layers, 2);
        assert_eq!(s.tensor_name_pattern, NamingPattern::HuggingFace);
    }

    #[test]
    fn test_inspect_gguf() {
        let s = inspect_layers(&gguf_tensor_names());
        assert_eq!(s.num_layers, 2);
        assert_eq!(s.tensor_name_pattern, NamingPattern::Gguf);
    }

    #[test]
    fn test_non_layer_tensors() {
        let s = inspect_layers(&hf_tensor_names());
        assert!(s.non_layer_tensors.contains(&"model.embed_tokens.weight".into()));
        assert!(s.non_layer_tensors.contains(&"lm_head.weight".into()));
    }

    #[test]
    fn test_layer_components() {
        let s = inspect_layers(&hf_tensor_names());
        let l0 = &s.layers[0];
        assert_eq!(l0.index, 0);
        assert!(l0.components.contains("self_attn.q_proj.weight"));
    }

    #[test]
    fn test_empty() {
        let s = inspect_layers(&[]);
        assert_eq!(s.num_layers, 0);
        assert_eq!(s.tensor_name_pattern, NamingPattern::Unknown);
    }

    #[test]
    fn test_extract_index_hf() {
        assert_eq!(
            extract_layer_index("model.layers.5.self_attn.q_proj.weight"),
            Some(5)
        );
    }

    #[test]
    fn test_extract_index_gguf() {
        assert_eq!(extract_layer_index("blk.3.attn_q.weight"), Some(3));
    }

    #[test]
    fn test_extract_index_none() {
        assert_eq!(extract_layer_index("lm_head.weight"), None);
    }

    #[test]
    fn test_extract_component() {
        assert_eq!(
            extract_component("model.layers.0.self_attn.q_proj.weight"),
            "self_attn.q_proj.weight"
        );
    }

    #[test]
    fn test_naming_pattern_display() {
        assert_eq!(format!("{}", NamingPattern::HuggingFace), "HuggingFace");
        assert_eq!(format!("{}", NamingPattern::Gguf), "GGUF");
    }

    #[test]
    fn test_expected_components() {
        let comps = expected_components();
        assert!(comps.len() >= 9);
    }

    #[test]
    fn test_missing_components() {
        let layer = LayerInfo {
            index: 0,
            components: BTreeSet::from(["self_attn.q_proj.weight".into()]),
            tensor_count: 1,
        };
        let missing = missing_components(&layer);
        assert!(missing.len() >= 8);
        assert!(!missing.contains(&"self_attn.q_proj.weight".into()));
    }

    #[test]
    fn test_layer_count() {
        let mut names = vec![];
        for i in 0..32 {
            names.push(format!("model.layers.{i}.self_attn.q_proj.weight"));
        }
        let s = inspect_layers(&names);
        assert_eq!(s.num_layers, 32);
    }

    #[test]
    fn test_tensor_count_per_layer() {
        let s = inspect_layers(&hf_tensor_names());
        assert_eq!(s.layers[0].tensor_count, 3); // q, k, gate
    }
}

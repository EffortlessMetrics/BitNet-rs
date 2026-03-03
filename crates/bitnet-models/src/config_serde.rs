//! Model configuration serialization/deserialization.

use std::collections::BTreeMap;
use std::collections::HashMap;

/// A portable model configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct PortableConfig {
    pub architecture: String,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: Option<usize>,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub activation: String,
    pub rope_theta: Option<f64>,
    pub tie_word_embeddings: bool,
    pub extra: HashMap<String, String>,
}

impl Default for PortableConfig {
    fn default() -> Self {
        Self {
            architecture: String::new(),
            hidden_size: 0,
            num_layers: 0,
            num_heads: 0,
            num_kv_heads: None,
            intermediate_size: 0,
            vocab_size: 0,
            max_position_embeddings: 2048,
            activation: "gelu".to_string(),
            rope_theta: None,
            tie_word_embeddings: false,
            extra: HashMap::new(),
        }
    }
}

impl PortableConfig {
    pub fn new(arch: &str) -> Self {
        Self { architecture: arch.to_string(), ..Default::default() }
    }

    pub fn head_dim(&self) -> usize {
        if self.num_heads == 0 { 0 } else { self.hidden_size / self.num_heads }
    }

    pub fn effective_kv_heads(&self) -> usize {
        self.num_kv_heads.unwrap_or(self.num_heads)
    }

    pub fn gqa_group_size(&self) -> usize {
        let kv = self.effective_kv_heads();
        if kv == 0 { 0 } else { self.num_heads / kv }
    }

    pub fn to_pairs(&self) -> Vec<(String, String)> {
        let mut pairs = vec![
            ("architecture".into(), self.architecture.clone()),
            ("hidden_size".into(), self.hidden_size.to_string()),
            ("num_hidden_layers".into(), self.num_layers.to_string()),
            ("num_attention_heads".into(), self.num_heads.to_string()),
            ("intermediate_size".into(), self.intermediate_size.to_string()),
            ("vocab_size".into(), self.vocab_size.to_string()),
            ("max_position_embeddings".into(), self.max_position_embeddings.to_string()),
            ("hidden_act".into(), self.activation.clone()),
            ("tie_word_embeddings".into(), self.tie_word_embeddings.to_string()),
        ];
        if let Some(kv) = self.num_kv_heads {
            pairs.push(("num_key_value_heads".into(), kv.to_string()));
        }
        if let Some(theta) = self.rope_theta {
            pairs.push(("rope_theta".into(), theta.to_string()));
        }
        pairs
    }

    pub fn from_pairs(pairs: &[(String, String)]) -> Self {
        let mut config = Self::default();
        let mut extra = HashMap::new();
        for (k, v) in pairs {
            match k.as_str() {
                "architecture" | "model_type" => config.architecture = v.clone(),
                "hidden_size" => {
                    config.hidden_size = v.parse().unwrap_or(0);
                }
                "num_hidden_layers" => {
                    config.num_layers = v.parse().unwrap_or(0);
                }
                "num_attention_heads" => {
                    config.num_heads = v.parse().unwrap_or(0);
                }
                "num_key_value_heads" => {
                    config.num_kv_heads = v.parse().ok();
                }
                "intermediate_size" => {
                    config.intermediate_size = v.parse().unwrap_or(0);
                }
                "vocab_size" => {
                    config.vocab_size = v.parse().unwrap_or(0);
                }
                "max_position_embeddings" => {
                    config.max_position_embeddings = v.parse().unwrap_or(2048);
                }
                "hidden_act" => config.activation = v.clone(),
                "rope_theta" => {
                    config.rope_theta = v.parse().ok();
                }
                "tie_word_embeddings" => {
                    config.tie_word_embeddings = v == "true";
                }
                _ => {
                    extra.insert(k.clone(), v.clone());
                }
            }
        }
        config.extra = extra;
        config
    }
}

pub fn phi4_config() -> PortableConfig {
    PortableConfig {
        architecture: "phi".into(),
        hidden_size: 5120,
        num_layers: 40,
        num_heads: 40,
        num_kv_heads: Some(10),
        intermediate_size: 17920,
        vocab_size: 100352,
        max_position_embeddings: 16384,
        activation: "silu".into(),
        rope_theta: Some(250000.0),
        tie_word_embeddings: false,
        extra: HashMap::new(),
    }
}

pub fn llama3_8b_config() -> PortableConfig {
    PortableConfig {
        architecture: "llama".into(),
        hidden_size: 4096,
        num_layers: 32,
        num_heads: 32,
        num_kv_heads: Some(8),
        intermediate_size: 14336,
        vocab_size: 128256,
        max_position_embeddings: 8192,
        activation: "silu".into(),
        rope_theta: Some(500000.0),
        tie_word_embeddings: false,
        extra: HashMap::new(),
    }
}

pub fn smollm2_config() -> PortableConfig {
    PortableConfig {
        architecture: "llama".into(),
        hidden_size: 2048,
        num_layers: 24,
        num_heads: 32,
        num_kv_heads: Some(32),
        intermediate_size: 8192,
        vocab_size: 49152,
        max_position_embeddings: 8192,
        activation: "silu".into(),
        rope_theta: Some(100000.0),
        tie_word_embeddings: true,
        extra: HashMap::new(),
    }
}

/// Serializable model configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct SerializableConfig {
    pub entries: BTreeMap<String, String>,
}

impl Default for SerializableConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl SerializableConfig {
    pub fn new() -> Self {
        Self { entries: BTreeMap::new() }
    }

    pub fn set(&mut self, key: &str, value: &str) -> &mut Self {
        self.entries.insert(key.to_string(), value.to_string());
        self
    }

    pub fn set_usize(&mut self, key: &str, value: usize) -> &mut Self {
        self.set(key, &value.to_string())
    }

    pub fn set_f64(&mut self, key: &str, value: f64) -> &mut Self {
        self.set(key, &value.to_string())
    }

    pub fn set_bool(&mut self, key: &str, value: bool) -> &mut Self {
        self.set(key, if value { "true" } else { "false" })
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.entries.get(key).map(|s| s.as_str())
    }

    pub fn get_usize(&self, key: &str) -> Option<usize> {
        self.get(key)?.parse().ok()
    }

    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.get(key)?.parse().ok()
    }

    pub fn get_bool(&self, key: &str) -> Option<bool> {
        match self.get(key)? {
            "true" | "1" | "yes" => Some(true),
            "false" | "0" | "no" => Some(false),
            _ => None,
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Serialize to key=value text.
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        for (k, v) in &self.entries {
            out.push_str(k);
            out.push('=');
            out.push_str(v);
            out.push('\n');
        }
        out
    }

    /// Parse from key=value text.
    pub fn from_text(text: &str) -> Self {
        let mut cfg = Self::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some((k, v)) = line.split_once('=') {
                cfg.entries.insert(k.trim().to_string(), v.trim().to_string());
            }
        }
        cfg
    }

    /// Merge another config (other wins on conflict).
    pub fn merge(&mut self, other: &SerializableConfig) {
        for (k, v) in &other.entries {
            self.entries.insert(k.clone(), v.clone());
        }
    }

    /// Keys that differ between two configs.
    pub fn diff(&self, other: &SerializableConfig) -> Vec<String> {
        let mut diffs = Vec::new();
        for (k, v) in &self.entries {
            match other.get(k) {
                Some(ov) if ov != v => diffs.push(k.clone()),
                None => diffs.push(k.clone()),
                _ => {}
            }
        }
        for k in other.entries.keys() {
            if !self.entries.contains_key(k) {
                diffs.push(k.clone());
            }
        }
        diffs.sort();
        diffs.dedup();
        diffs
    }

    /// Build a standard transformer config.
    pub fn transformer(
        arch: &str,
        layers: usize,
        hidden: usize,
        heads: usize,
        kv_heads: usize,
        vocab: usize,
        context: usize,
    ) -> Self {
        let mut cfg = Self::new();
        cfg.set("architecture", arch);
        cfg.set_usize("num_layers", layers);
        cfg.set_usize("hidden_size", hidden);
        cfg.set_usize("num_heads", heads);
        cfg.set_usize("num_kv_heads", kv_heads);
        cfg.set_usize("vocab_size", vocab);
        cfg.set_usize("max_context", context);
        cfg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default() {
        let c = PortableConfig::default();
        assert!(c.architecture.is_empty());
        assert_eq!(c.max_position_embeddings, 2048);
    }

    #[test]
    fn test_new() {
        assert_eq!(PortableConfig::new("llama").architecture, "llama");
    }

    #[test]
    fn test_head_dim() {
        assert_eq!(phi4_config().head_dim(), 128);
    }

    #[test]
    fn test_head_dim_zero() {
        assert_eq!(PortableConfig::default().head_dim(), 0);
    }

    #[test]
    fn test_effective_kv() {
        assert_eq!(phi4_config().effective_kv_heads(), 10);
    }

    #[test]
    fn test_effective_kv_none() {
        let c = PortableConfig { num_heads: 32, num_kv_heads: None, ..Default::default() };
        assert_eq!(c.effective_kv_heads(), 32);
    }

    #[test]
    fn test_gqa() {
        assert_eq!(phi4_config().gqa_group_size(), 4);
    }

    #[test]
    fn test_round_trip() {
        let orig = phi4_config();
        let pairs = orig.to_pairs();
        let restored = PortableConfig::from_pairs(&pairs);
        assert_eq!(restored.hidden_size, orig.hidden_size);
        assert_eq!(restored.num_layers, orig.num_layers);
        assert_eq!(restored.num_heads, orig.num_heads);
    }

    #[test]
    fn test_extra_keys() {
        let pairs = vec![("hidden_size".into(), "4096".into()), ("custom".into(), "val".into())];
        let c = PortableConfig::from_pairs(&pairs);
        assert_eq!(c.extra["custom"], "val");
    }

    #[test]
    fn test_phi4() {
        let c = phi4_config();
        assert_eq!(c.hidden_size, 5120);
        assert_eq!(c.num_layers, 40);
    }

    #[test]
    fn test_llama3() {
        assert_eq!(llama3_8b_config().hidden_size, 4096);
    }

    #[test]
    fn test_smollm2() {
        let c = smollm2_config();
        assert!(c.tie_word_embeddings);
        assert_eq!(c.vocab_size, 49152);
    }

    #[test]
    fn test_clone_eq() {
        let a = phi4_config();
        assert_eq!(a, a.clone());
    }

    #[test]
    fn test_gqa_zero() {
        let c = PortableConfig { num_heads: 32, num_kv_heads: Some(0), ..Default::default() };
        assert_eq!(c.gqa_group_size(), 0);
    }

    #[test]
    fn test_model_type() {
        let pairs = vec![("model_type".into(), "phi".into())];
        let c = PortableConfig::from_pairs(&pairs);
        assert_eq!(c.architecture, "phi");
    }

    // SerializableConfig tests

    #[test]
    fn test_sc_set_get() {
        let mut cfg = SerializableConfig::new();
        cfg.set("key", "value");
        assert_eq!(cfg.get("key"), Some("value"));
    }

    #[test]
    fn test_sc_set_usize() {
        let mut cfg = SerializableConfig::new();
        cfg.set_usize("layers", 40);
        assert_eq!(cfg.get_usize("layers"), Some(40));
    }

    #[test]
    fn test_sc_set_f64() {
        let mut cfg = SerializableConfig::new();
        cfg.set_f64("eps", 1e-5);
        assert!((cfg.get_f64("eps").unwrap() - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn test_sc_set_bool() {
        let mut cfg = SerializableConfig::new();
        cfg.set_bool("bias", false);
        assert_eq!(cfg.get_bool("bias"), Some(false));
    }

    #[test]
    fn test_sc_to_text() {
        let mut cfg = SerializableConfig::new();
        cfg.set("a", "1");
        cfg.set("b", "2");
        let text = cfg.to_text();
        assert!(text.contains("a=1"));
        assert!(text.contains("b=2"));
    }

    #[test]
    fn test_sc_from_text() {
        let text = "layers=40\nhidden=5120\n# comment\n\n";
        let cfg = SerializableConfig::from_text(text);
        assert_eq!(cfg.get("layers"), Some("40"));
        assert_eq!(cfg.get("hidden"), Some("5120"));
        assert_eq!(cfg.len(), 2);
    }

    #[test]
    fn test_sc_roundtrip() {
        let mut orig = SerializableConfig::new();
        orig.set("arch", "phi4");
        orig.set_usize("layers", 40);
        let text = orig.to_text();
        let parsed = SerializableConfig::from_text(&text);
        assert_eq!(orig, parsed);
    }

    #[test]
    fn test_sc_merge() {
        let mut a = SerializableConfig::new();
        a.set("x", "1");
        let mut b = SerializableConfig::new();
        b.set("x", "2");
        b.set("y", "3");
        a.merge(&b);
        assert_eq!(a.get("x"), Some("2")); // b wins
        assert_eq!(a.get("y"), Some("3"));
    }

    #[test]
    fn test_sc_diff() {
        let mut a = SerializableConfig::new();
        a.set("x", "1");
        a.set("y", "2");
        let mut b = SerializableConfig::new();
        b.set("x", "1");
        b.set("y", "3");
        b.set("z", "4");
        let d = a.diff(&b);
        assert!(d.contains(&"y".to_string()));
        assert!(d.contains(&"z".to_string()));
        assert!(!d.contains(&"x".to_string()));
    }

    #[test]
    fn test_sc_transformer() {
        let cfg = SerializableConfig::transformer("phi4", 40, 5120, 40, 10, 100352, 16384);
        assert_eq!(cfg.get("architecture"), Some("phi4"));
        assert_eq!(cfg.get_usize("num_layers"), Some(40));
    }

    #[test]
    fn test_sc_empty() {
        let cfg = SerializableConfig::new();
        assert!(cfg.is_empty());
        assert_eq!(cfg.len(), 0);
    }

    #[test]
    fn test_sc_get_missing() {
        let cfg = SerializableConfig::new();
        assert_eq!(cfg.get("nope"), None);
        assert_eq!(cfg.get_usize("nope"), None);
    }
}

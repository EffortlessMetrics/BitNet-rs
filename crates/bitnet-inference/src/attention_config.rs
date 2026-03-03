//! Attention mechanism configuration.
//!
//! Configure multi-head, grouped-query, and multi-query attention
//! with head dimensions, KV cache sizing, and masking.

/// Attention variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionType {
    /// Multi-Head Attention: each head has its own K/V.
    MultiHead,
    /// Grouped-Query Attention: multiple Q heads share K/V groups.
    GroupedQuery,
    /// Multi-Query Attention: all Q heads share one K/V.
    MultiQuery,
}

impl AttentionType {
    pub fn name(&self) -> &'static str {
        match self {
            Self::MultiHead => "MHA",
            Self::GroupedQuery => "GQA",
            Self::MultiQuery => "MQA",
        }
    }
}

/// Full attention configuration.
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub max_seq_len: usize,
    pub use_bias: bool,
    pub rope_enabled: bool,
    pub rope_base: f32,
}

impl AttentionConfig {
    /// Create a standard MHA config.
    pub fn mha(num_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        Self {
            num_heads,
            num_kv_heads: num_heads,
            head_dim,
            hidden_size: num_heads * head_dim,
            max_seq_len,
            use_bias: false,
            rope_enabled: true,
            rope_base: 10000.0,
        }
    }

    /// Create a GQA config (e.g., Phi-4: 40 heads, 10 KV heads).
    pub fn gqa(num_heads: usize, num_kv_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_size: num_heads * head_dim,
            max_seq_len,
            use_bias: false,
            rope_enabled: true,
            rope_base: 10000.0,
        }
    }

    /// Create an MQA config.
    pub fn mqa(num_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        Self {
            num_heads,
            num_kv_heads: 1,
            head_dim,
            hidden_size: num_heads * head_dim,
            max_seq_len,
            use_bias: false,
            rope_enabled: true,
            rope_base: 10000.0,
        }
    }

    /// Detect attention type.
    pub fn attention_type(&self) -> AttentionType {
        if self.num_kv_heads == self.num_heads {
            AttentionType::MultiHead
        } else if self.num_kv_heads == 1 {
            AttentionType::MultiQuery
        } else {
            AttentionType::GroupedQuery
        }
    }

    /// Number of Q heads per KV group.
    pub fn group_size(&self) -> usize {
        if self.num_kv_heads == 0 {
            return 0;
        }
        self.num_heads / self.num_kv_heads
    }

    /// Q projection size: num_heads * head_dim.
    pub fn q_proj_size(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// K projection size: num_kv_heads * head_dim.
    pub fn k_proj_size(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    /// V projection size: num_kv_heads * head_dim.
    pub fn v_proj_size(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    /// KV cache size per layer in elements (K + V).
    pub fn kv_cache_elements_per_layer(&self) -> usize {
        2 * self.num_kv_heads * self.head_dim * self.max_seq_len
    }

    /// KV cache size per layer in bytes (f16 = 2 bytes).
    pub fn kv_cache_bytes_per_layer(&self) -> usize {
        self.kv_cache_elements_per_layer() * 2
    }

    /// Validate configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.num_heads == 0 {
            return Err("num_heads must be > 0".into());
        }
        if self.num_kv_heads == 0 {
            return Err("num_kv_heads must be > 0".into());
        }
        if self.num_heads % self.num_kv_heads != 0 {
            return Err(format!(
                "num_heads ({}) must be divisible by num_kv_heads ({})",
                self.num_heads, self.num_kv_heads,
            ));
        }
        if self.head_dim == 0 {
            return Err("head_dim must be > 0".into());
        }
        Ok(())
    }

    pub fn summary(&self) -> String {
        format!(
            "{} h={} kv={} d={} seq={}",
            self.attention_type().name(),
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.max_seq_len,
        )
    }
}

/// Presets for common models.
pub mod presets {
    use super::AttentionConfig;

    /// BitNet b1.58-2B: 20 heads, 5 KV heads, dim=128.
    pub fn bitnet_2b() -> AttentionConfig {
        AttentionConfig::gqa(20, 5, 128, 4096)
    }

    /// Phi-4: 40 heads, 10 KV heads, dim=128, 16K context.
    pub fn phi4() -> AttentionConfig {
        AttentionConfig::gqa(40, 10, 128, 16384)
    }

    /// LLaMA-3 8B: 32 heads, 8 KV heads, dim=128.
    pub fn llama3_8b() -> AttentionConfig {
        AttentionConfig::gqa(32, 8, 128, 8192)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mha_type() {
        let cfg = AttentionConfig::mha(32, 64, 2048);
        assert_eq!(cfg.attention_type(), AttentionType::MultiHead);
        assert_eq!(cfg.group_size(), 1);
    }

    #[test]
    fn test_gqa_type() {
        let cfg = AttentionConfig::gqa(40, 10, 128, 16384);
        assert_eq!(cfg.attention_type(), AttentionType::GroupedQuery);
        assert_eq!(cfg.group_size(), 4);
    }

    #[test]
    fn test_mqa_type() {
        let cfg = AttentionConfig::mqa(32, 64, 2048);
        assert_eq!(cfg.attention_type(), AttentionType::MultiQuery);
        assert_eq!(cfg.group_size(), 32);
    }

    #[test]
    fn test_projection_sizes() {
        let cfg = AttentionConfig::gqa(40, 10, 128, 16384);
        assert_eq!(cfg.q_proj_size(), 5120);
        assert_eq!(cfg.k_proj_size(), 1280);
        assert_eq!(cfg.v_proj_size(), 1280);
    }

    #[test]
    fn test_kv_cache_size() {
        let cfg = AttentionConfig::gqa(40, 10, 128, 16384);
        let elems = cfg.kv_cache_elements_per_layer();
        assert_eq!(elems, 2 * 10 * 128 * 16384);
    }

    #[test]
    fn test_validate_ok() {
        let cfg = presets::phi4();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_validate_bad_divisibility() {
        let cfg = AttentionConfig::gqa(40, 3, 128, 2048);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_heads() {
        let mut cfg = presets::phi4();
        cfg.num_heads = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_summary() {
        let cfg = presets::phi4();
        let s = cfg.summary();
        assert!(s.contains("GQA"));
        assert!(s.contains("h=40"));
    }

    #[test]
    fn test_attention_type_name() {
        assert_eq!(AttentionType::MultiHead.name(), "MHA");
        assert_eq!(AttentionType::GroupedQuery.name(), "GQA");
        assert_eq!(AttentionType::MultiQuery.name(), "MQA");
    }

    #[test]
    fn test_preset_bitnet() {
        let cfg = presets::bitnet_2b();
        assert_eq!(cfg.num_heads, 20);
        assert_eq!(cfg.num_kv_heads, 5);
        assert_eq!(cfg.hidden_size, 2560);
    }

    #[test]
    fn test_preset_llama3() {
        let cfg = presets::llama3_8b();
        assert_eq!(cfg.num_heads, 32);
        assert_eq!(cfg.num_kv_heads, 8);
        assert_eq!(cfg.max_seq_len, 8192);
    }
}

//! Attention pattern analysis.
//!
//! Analyze and validate attention patterns for different model architectures.

/// Attention pattern type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttentionPattern {
    Full,          // Standard full attention
    Causal,        // Lower triangular
    SlidingWindow, // Fixed window size
    Sparse,        // Sparse attention
    Linear,        // Linear attention approximation
}

impl AttentionPattern {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Causal => "causal",
            Self::SlidingWindow => "sliding_window",
            Self::Sparse => "sparse",
            Self::Linear => "linear",
        }
    }

    /// Memory complexity class.
    pub fn memory_complexity(&self, seq_len: usize) -> u64 {
        match self {
            Self::Full | Self::Causal => (seq_len as u64) * (seq_len as u64),
            Self::SlidingWindow => seq_len as u64 * 256, // O(n * w)
            Self::Sparse => (seq_len as u64) * ((seq_len as f64).sqrt() as u64).max(1),
            Self::Linear => seq_len as u64 * 64, // O(n * d)
        }
    }
}

/// GQA (Grouped Query Attention) configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GqaConfig {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl GqaConfig {
    pub fn new(num_heads: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        Self { num_heads, num_kv_heads, head_dim }
    }

    /// Multi-head attention (MHA): every head has its own KV.
    pub fn mha(num_heads: usize, head_dim: usize) -> Self {
        Self { num_heads, num_kv_heads: num_heads, head_dim }
    }

    /// Multi-query attention (MQA): single KV head shared by all.
    pub fn mqa(num_heads: usize, head_dim: usize) -> Self {
        Self { num_heads, num_kv_heads: 1, head_dim }
    }

    pub fn group_size(&self) -> usize {
        if self.num_kv_heads == 0 {
            return 0;
        }
        self.num_heads / self.num_kv_heads
    }

    pub fn is_mha(&self) -> bool {
        self.num_heads == self.num_kv_heads
    }
    pub fn is_mqa(&self) -> bool {
        self.num_kv_heads == 1
    }
    pub fn is_gqa(&self) -> bool {
        !self.is_mha() && !self.is_mqa()
    }
    pub fn is_valid(&self) -> bool {
        self.num_kv_heads > 0 && self.num_heads % self.num_kv_heads == 0
    }

    pub fn q_size(&self) -> usize {
        self.num_heads * self.head_dim
    }
    pub fn kv_size(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    /// KV cache bytes per token per layer (fp16).
    pub fn kv_bytes_per_token(&self) -> usize {
        self.num_kv_heads * self.head_dim * 2 * 2 // K + V, 2 bytes each
    }
}

/// Standard model GQA configs.
pub fn bitnet_gqa() -> GqaConfig {
    GqaConfig::new(20, 5, 128)
}
pub fn phi4_gqa() -> GqaConfig {
    GqaConfig::new(40, 10, 128)
}
pub fn llama3_8b_gqa() -> GqaConfig {
    GqaConfig::new(32, 8, 128)
}
pub fn llama3_70b_gqa() -> GqaConfig {
    GqaConfig::new(64, 8, 128)
}
pub fn qwen2_7b_gqa() -> GqaConfig {
    GqaConfig::new(28, 4, 128)
}

/// Validate attention shape.
pub fn validate_attention_shape(
    batch: usize,
    seq_len: usize,
    config: &GqaConfig,
) -> Result<(), String> {
    if batch == 0 {
        return Err("batch size must be > 0".into());
    }
    if seq_len == 0 {
        return Err("sequence length must be > 0".into());
    }
    if !config.is_valid() {
        return Err("invalid GQA config".into());
    }
    Ok(())
}

/// Estimate attention memory (bytes) for a given config.
pub fn estimate_attention_memory(
    batch: usize,
    seq_len: usize,
    num_layers: usize,
    config: &GqaConfig,
) -> u64 {
    let kv_per_token = config.kv_bytes_per_token() as u64;
    batch as u64 * seq_len as u64 * num_layers as u64 * kv_per_token
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_str() {
        assert_eq!(AttentionPattern::Causal.as_str(), "causal");
        assert_eq!(AttentionPattern::SlidingWindow.as_str(), "sliding_window");
    }

    #[test]
    fn test_memory_complexity() {
        let full = AttentionPattern::Full.memory_complexity(1024);
        let window = AttentionPattern::SlidingWindow.memory_complexity(1024);
        assert!(full > window);
    }

    #[test]
    fn test_gqa_mha() {
        let config = GqaConfig::mha(32, 128);
        assert!(config.is_mha());
        assert!(!config.is_gqa());
        assert_eq!(config.group_size(), 1);
    }

    #[test]
    fn test_gqa_mqa() {
        let config = GqaConfig::mqa(32, 128);
        assert!(config.is_mqa());
        assert_eq!(config.group_size(), 32);
    }

    #[test]
    fn test_phi4_gqa() {
        let config = phi4_gqa();
        assert!(config.is_gqa());
        assert_eq!(config.group_size(), 4);
        assert!(config.is_valid());
    }

    #[test]
    fn test_bitnet_gqa() {
        let config = bitnet_gqa();
        assert_eq!(config.group_size(), 4);
    }

    #[test]
    fn test_q_kv_size() {
        let config = phi4_gqa();
        assert_eq!(config.q_size(), 5120);
        assert_eq!(config.kv_size(), 1280);
    }

    #[test]
    fn test_kv_bytes() {
        let config = phi4_gqa();
        let bytes = config.kv_bytes_per_token();
        assert!(bytes > 0);
    }

    #[test]
    fn test_validate_ok() {
        let config = phi4_gqa();
        assert!(validate_attention_shape(1, 1024, &config).is_ok());
    }

    #[test]
    fn test_validate_zero_batch() {
        let config = phi4_gqa();
        assert!(validate_attention_shape(0, 1024, &config).is_err());
    }

    #[test]
    fn test_estimate_memory() {
        let config = phi4_gqa();
        let mem = estimate_attention_memory(1, 16384, 40, &config);
        assert!(mem > 0);
        // Phi-4: 10 kv heads * 128 dim * 2 (K+V) * 2 bytes = 5120 bytes/token/layer
        // 16384 * 40 * 5120 = ~3.2 GB
        assert!(mem > 3_000_000_000);
    }

    #[test]
    fn test_invalid_gqa() {
        let config = GqaConfig::new(10, 3, 128); // 10 not divisible by 3
        assert!(!config.is_valid());
    }

    #[test]
    fn test_llama3_configs() {
        assert!(llama3_8b_gqa().is_valid());
        assert!(llama3_70b_gqa().is_valid());
        assert!(qwen2_7b_gqa().is_valid());
    }
}

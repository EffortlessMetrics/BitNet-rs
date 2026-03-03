//! Builder for model configurations across architectures.
//!
//! Fluent API to construct model configs for BitNet, Phi-4,
//! LLaMA, Qwen, and other SLM architectures.

use std::collections::HashMap;

/// Normalization type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    LayerNorm,
    RmsNorm,
    BitNorm,
}

impl NormType {
    pub fn name(&self) -> &'static str {
        match self {
            Self::LayerNorm => "layer_norm",
            Self::RmsNorm => "rms_norm",
            Self::BitNorm => "bit_norm",
        }
    }
}

/// Activation function type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationType {
    Silu,
    Gelu,
    ReluSquared,
    Relu,
    Mish,
}

impl ActivationType {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Silu => "silu",
            Self::Gelu => "gelu",
            Self::ReluSquared => "relu_squared",
            Self::Relu => "relu",
            Self::Mish => "mish",
        }
    }
}

/// Complete model configuration.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub name: String,
    pub arch: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub norm_type: NormType,
    pub activation: ActivationType,
    pub norm_eps: f32,
    pub rope_base: f32,
    pub use_bias: bool,
    pub tie_word_embeddings: bool,
    pub extra: HashMap<String, String>,
}

impl ModelConfig {
    pub fn total_params_estimate(&self) -> usize {
        // Rough estimate: embedding + layers * (attn + ffn + norms)
        let embed = self.vocab_size * self.hidden_size * 2; // input + output
        let attn_per_layer = self.hidden_size * self.hidden_size * 4; // Q,K,V,O
        let ffn_per_layer = self.hidden_size * self.intermediate_size * 3;
        let norm_per_layer = self.hidden_size * 4;
        embed + self.num_layers * (attn_per_layer + ffn_per_layer + norm_per_layer)
    }

    pub fn kv_cache_size_bytes(&self, seq_len: usize) -> usize {
        // 2 (K+V) * layers * kv_heads * head_dim * seq_len * 2 (f16)
        2 * self.num_layers * self.num_kv_heads * self.head_dim * seq_len * 2
    }

    pub fn summary(&self) -> String {
        format!(
            "{} ({}): {}L/{}H/{}KV d={} v={} ctx={}",
            self.name,
            self.arch,
            self.num_layers,
            self.num_heads,
            self.num_kv_heads,
            self.hidden_size,
            self.vocab_size,
            self.max_seq_len,
        )
    }
}

/// Fluent builder for ModelConfig.
#[derive(Debug)]
pub struct ModelConfigBuilder {
    config: ModelConfig,
}

impl ModelConfigBuilder {
    pub fn new(name: impl Into<String>, arch: impl Into<String>) -> Self {
        Self {
            config: ModelConfig {
                name: name.into(),
                arch: arch.into(),
                hidden_size: 768,
                intermediate_size: 3072,
                num_layers: 12,
                num_heads: 12,
                num_kv_heads: 12,
                head_dim: 64,
                vocab_size: 32000,
                max_seq_len: 2048,
                norm_type: NormType::RmsNorm,
                activation: ActivationType::Silu,
                norm_eps: 1e-5,
                rope_base: 10000.0,
                use_bias: false,
                tie_word_embeddings: false,
                extra: HashMap::new(),
            },
        }
    }

    pub fn hidden_size(mut self, v: usize) -> Self {
        self.config.hidden_size = v;
        self
    }
    pub fn intermediate_size(mut self, v: usize) -> Self {
        self.config.intermediate_size = v;
        self
    }
    pub fn num_layers(mut self, v: usize) -> Self {
        self.config.num_layers = v;
        self
    }
    pub fn num_heads(mut self, v: usize) -> Self {
        self.config.num_heads = v;
        self
    }
    pub fn num_kv_heads(mut self, v: usize) -> Self {
        self.config.num_kv_heads = v;
        self
    }
    pub fn head_dim(mut self, v: usize) -> Self {
        self.config.head_dim = v;
        self
    }
    pub fn vocab_size(mut self, v: usize) -> Self {
        self.config.vocab_size = v;
        self
    }
    pub fn max_seq_len(mut self, v: usize) -> Self {
        self.config.max_seq_len = v;
        self
    }
    pub fn norm_type(mut self, v: NormType) -> Self {
        self.config.norm_type = v;
        self
    }
    pub fn activation(mut self, v: ActivationType) -> Self {
        self.config.activation = v;
        self
    }
    pub fn norm_eps(mut self, v: f32) -> Self {
        self.config.norm_eps = v;
        self
    }
    pub fn rope_base(mut self, v: f32) -> Self {
        self.config.rope_base = v;
        self
    }
    pub fn use_bias(mut self, v: bool) -> Self {
        self.config.use_bias = v;
        self
    }
    pub fn tie_word_embeddings(mut self, v: bool) -> Self {
        self.config.tie_word_embeddings = v;
        self
    }

    pub fn extra(mut self, key: impl Into<String>, val: impl Into<String>) -> Self {
        self.config.extra.insert(key.into(), val.into());
        self
    }

    pub fn build(self) -> ModelConfig {
        self.config
    }
}

/// Presets for common models.
pub mod presets {
    use super::*;

    pub fn bitnet_2b() -> ModelConfig {
        ModelConfigBuilder::new("BitNet-b1.58-2B", "bitnet")
            .hidden_size(2560)
            .intermediate_size(6912)
            .num_layers(30)
            .num_heads(20)
            .num_kv_heads(5)
            .head_dim(128)
            .vocab_size(32000)
            .max_seq_len(4096)
            .norm_type(NormType::BitNorm)
            .activation(ActivationType::ReluSquared)
            .build()
    }

    pub fn phi4() -> ModelConfig {
        ModelConfigBuilder::new("Phi-4-14B", "phi4")
            .hidden_size(5120)
            .intermediate_size(17920)
            .num_layers(40)
            .num_heads(40)
            .num_kv_heads(10)
            .head_dim(128)
            .vocab_size(100352)
            .max_seq_len(16384)
            .norm_type(NormType::RmsNorm)
            .activation(ActivationType::Silu)
            .build()
    }

    pub fn llama3_8b() -> ModelConfig {
        ModelConfigBuilder::new("LLaMA-3-8B", "llama")
            .hidden_size(4096)
            .intermediate_size(14336)
            .num_layers(32)
            .num_heads(32)
            .num_kv_heads(8)
            .head_dim(128)
            .vocab_size(128256)
            .max_seq_len(8192)
            .norm_type(NormType::RmsNorm)
            .activation(ActivationType::Silu)
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_basic() {
        let cfg = ModelConfigBuilder::new("test", "arch").hidden_size(512).num_layers(6).build();
        assert_eq!(cfg.name, "test");
        assert_eq!(cfg.hidden_size, 512);
        assert_eq!(cfg.num_layers, 6);
    }

    #[test]
    fn test_builder_all_fields() {
        let cfg = ModelConfigBuilder::new("m", "a")
            .hidden_size(1024)
            .intermediate_size(4096)
            .num_heads(16)
            .num_kv_heads(4)
            .head_dim(64)
            .vocab_size(50000)
            .max_seq_len(4096)
            .norm_type(NormType::LayerNorm)
            .activation(ActivationType::Gelu)
            .use_bias(true)
            .build();
        assert_eq!(cfg.norm_type, NormType::LayerNorm);
        assert_eq!(cfg.activation, ActivationType::Gelu);
        assert!(cfg.use_bias);
    }

    #[test]
    fn test_preset_bitnet() {
        let cfg = presets::bitnet_2b();
        assert_eq!(cfg.num_layers, 30);
        assert_eq!(cfg.hidden_size, 2560);
        assert_eq!(cfg.norm_type, NormType::BitNorm);
    }

    #[test]
    fn test_preset_phi4() {
        let cfg = presets::phi4();
        assert_eq!(cfg.num_layers, 40);
        assert_eq!(cfg.num_heads, 40);
        assert_eq!(cfg.num_kv_heads, 10);
        assert_eq!(cfg.vocab_size, 100352);
    }

    #[test]
    fn test_preset_llama3() {
        let cfg = presets::llama3_8b();
        assert_eq!(cfg.num_layers, 32);
        assert_eq!(cfg.vocab_size, 128256);
    }

    #[test]
    fn test_params_estimate() {
        let cfg = presets::bitnet_2b();
        let params = cfg.total_params_estimate();
        assert!(params > 1_000_000_000); // > 1B
    }

    #[test]
    fn test_kv_cache_size() {
        let cfg = presets::phi4();
        let bytes = cfg.kv_cache_size_bytes(16384);
        assert!(bytes > 0);
        // 2 * 40 * 10 * 128 * 16384 * 2 = 3.36 GB
        assert!(bytes > 3_000_000_000);
    }

    #[test]
    fn test_summary() {
        let cfg = presets::phi4();
        let s = cfg.summary();
        assert!(s.contains("Phi-4"));
        assert!(s.contains("40L"));
    }

    #[test]
    fn test_norm_type_name() {
        assert_eq!(NormType::RmsNorm.name(), "rms_norm");
        assert_eq!(NormType::LayerNorm.name(), "layer_norm");
    }

    #[test]
    fn test_activation_type_name() {
        assert_eq!(ActivationType::Silu.name(), "silu");
        assert_eq!(ActivationType::ReluSquared.name(), "relu_squared");
    }

    #[test]
    fn test_extra_params() {
        let cfg = ModelConfigBuilder::new("test", "arch")
            .extra("rope_scaling", "linear")
            .extra("quantization", "int4")
            .build();
        assert_eq!(cfg.extra["rope_scaling"], "linear");
        assert_eq!(cfg.extra.len(), 2);
    }

    #[test]
    fn test_tie_embeddings() {
        let cfg = ModelConfigBuilder::new("test", "arch").tie_word_embeddings(true).build();
        assert!(cfg.tie_word_embeddings);
    }
}

//! Integration tests for dense model inference configurations.

/// Configuration for a dense (non-quantized) model.
#[derive(Debug, Clone)]
pub struct DenseModelConfig {
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
}

impl DenseModelConfig {
    /// Number of query heads per KV head group.
    pub fn gqa_groups(&self) -> usize {
        if self.num_kv_heads == 0 {
            return 0;
        }
        self.num_heads / self.num_kv_heads
    }

    /// Dimension per attention head.
    pub fn head_dim(&self) -> usize {
        if self.num_heads == 0 {
            return 0;
        }
        self.hidden_size / self.num_heads
    }

    /// Estimate KV cache size in bytes for a given sequence length and dtype.
    pub fn kv_cache_bytes(&self, seq_len: usize, bytes_per_element: usize) -> u64 {
        // KV cache: 2 * num_layers * num_kv_heads * head_dim * seq_len * bytes
        let head_dim = self.head_dim();
        2u64 * self.num_layers as u64
            * self.num_kv_heads as u64
            * head_dim as u64
            * seq_len as u64
            * bytes_per_element as u64
    }

    /// Estimate model weights size in bytes.
    pub fn weights_bytes(&self, bytes_per_param: usize) -> u64 {
        let params = self.estimate_params();
        params * bytes_per_param as u64
    }

    /// Rough parameter count estimate.
    pub fn estimate_params(&self) -> u64 {
        let h = self.hidden_size as u64;
        let ff = self.intermediate_size as u64;
        let v = self.vocab_size as u64;
        let l = self.num_layers as u64;
        // Embedding + per-layer (attention + FFN + norms) + final norm + lm_head
        let embedding = v * h;
        let per_layer = 4 * h * h + 3 * h * ff + 4 * h; // rough estimate
        let lm_head = v * h;
        embedding + l * per_layer + lm_head + h
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.num_heads == 0 {
            return Err("num_heads must be > 0".to_string());
        }
        if self.num_kv_heads == 0 {
            return Err("num_kv_heads must be > 0".to_string());
        }
        if self.hidden_size == 0 {
            return Err("hidden_size must be > 0".to_string());
        }
        if !self.hidden_size.is_multiple_of(self.num_heads) {
            return Err("hidden_size must be divisible by num_heads".to_string());
        }
        if !self.num_heads.is_multiple_of(self.num_kv_heads) {
            return Err("num_heads must be divisible by num_kv_heads".to_string());
        }
        if self.vocab_size == 0 {
            return Err("vocab_size must be > 0".to_string());
        }
        Ok(())
    }
}

/// Known model configurations for testing and validation.
pub fn phi4_config() -> DenseModelConfig {
    DenseModelConfig {
        num_layers: 40,
        num_heads: 40,
        num_kv_heads: 10,
        hidden_size: 5120,
        intermediate_size: 13824,
        vocab_size: 100352,
        max_position_embeddings: 16384,
        rms_norm_eps: 1e-5,
    }
}

pub fn llama_3_2_1b_config() -> DenseModelConfig {
    DenseModelConfig {
        num_layers: 16,
        num_heads: 32,
        num_kv_heads: 8,
        hidden_size: 2048,
        intermediate_size: 8192,
        vocab_size: 128256,
        max_position_embeddings: 131072,
        rms_norm_eps: 1e-5,
    }
}

pub fn qwen2_5_7b_config() -> DenseModelConfig {
    DenseModelConfig {
        num_layers: 28,
        num_heads: 28,
        num_kv_heads: 4,
        hidden_size: 3584,
        intermediate_size: 18944,
        vocab_size: 152064,
        max_position_embeddings: 131072,
        rms_norm_eps: 1e-6,
    }
}

pub fn gemma2_2b_config() -> DenseModelConfig {
    DenseModelConfig {
        num_layers: 26,
        num_heads: 8,
        num_kv_heads: 4,
        hidden_size: 2304,
        intermediate_size: 9216,
        vocab_size: 256000,
        max_position_embeddings: 8192,
        rms_norm_eps: 1e-6,
    }
}

pub fn mistral_7b_config() -> DenseModelConfig {
    DenseModelConfig {
        num_layers: 32,
        num_heads: 32,
        num_kv_heads: 8,
        hidden_size: 4096,
        intermediate_size: 14336,
        vocab_size: 32768,
        max_position_embeddings: 32768,
        rms_norm_eps: 1e-5,
    }
}

pub fn smollm2_1_7b_config() -> DenseModelConfig {
    DenseModelConfig {
        num_layers: 24,
        num_heads: 32,
        num_kv_heads: 32,
        hidden_size: 2048,
        intermediate_size: 8192,
        vocab_size: 49152,
        max_position_embeddings: 8192,
        rms_norm_eps: 1e-5,
    }
}

#[cfg(test)]
#[allow(clippy::all, clippy::pedantic, clippy::nursery)]
mod tests {
    use super::*;

    #[test]
    fn test_phi4_config() {
        let c = phi4_config();
        assert_eq!(c.gqa_groups(), 4);
        assert_eq!(c.head_dim(), 128);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn test_llama_config() {
        let c = llama_3_2_1b_config();
        assert_eq!(c.gqa_groups(), 4);
        assert_eq!(c.head_dim(), 64);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn test_qwen_config() {
        let c = qwen2_5_7b_config();
        assert_eq!(c.gqa_groups(), 7);
        assert_eq!(c.head_dim(), 128);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn test_gemma_config() {
        let c = gemma2_2b_config();
        assert_eq!(c.gqa_groups(), 2);
        assert_eq!(c.head_dim(), 288);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn test_mistral_config() {
        let c = mistral_7b_config();
        assert_eq!(c.gqa_groups(), 4);
        assert_eq!(c.head_dim(), 128);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn test_smollm2_config() {
        let c = smollm2_1_7b_config();
        assert_eq!(c.gqa_groups(), 1); // MHA (not GQA)
        assert_eq!(c.head_dim(), 64);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn test_phi4_kv_cache_16k() {
        let c = phi4_config();
        let bytes = c.kv_cache_bytes(16384, 2); // FP16
        // 2 * 40 * 10 * 128 * 16384 * 2 = ~3.36 GB
        assert!(bytes > 3_000_000_000);
        assert!(bytes < 4_000_000_000);
    }

    #[test]
    fn test_llama_kv_cache_4k() {
        let c = llama_3_2_1b_config();
        let bytes = c.kv_cache_bytes(4096, 2);
        assert!(bytes > 0);
        assert!(bytes < 1_000_000_000);
    }

    #[test]
    fn test_phi4_weights_bf16() {
        let c = phi4_config();
        let bytes = c.weights_bytes(2); // BF16
        assert!(bytes > 20_000_000_000); // > 20 GB
    }

    #[test]
    fn test_phi4_param_count() {
        let c = phi4_config();
        let params = c.estimate_params();
        // Phi-4 is ~14B params
        assert!(params > 10_000_000_000);
        assert!(params < 20_000_000_000);
    }

    #[test]
    fn test_validate_zero_heads() {
        let mut c = phi4_config();
        c.num_heads = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_validate_zero_kv_heads() {
        let mut c = phi4_config();
        c.num_kv_heads = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_validate_hidden_not_divisible() {
        let mut c = phi4_config();
        c.hidden_size = 5121; // not divisible by 40
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_validate_heads_not_divisible_by_kv() {
        let mut c = phi4_config();
        c.num_kv_heads = 7; // 40 not divisible by 7
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_validate_zero_vocab() {
        let mut c = phi4_config();
        c.vocab_size = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_gqa_groups_mha() {
        // Multi-head attention (no grouping)
        let c = smollm2_1_7b_config();
        assert_eq!(c.gqa_groups(), 1);
    }

    #[test]
    fn test_head_dim_consistency() {
        for config_fn in [
            phi4_config,
            llama_3_2_1b_config,
            qwen2_5_7b_config,
            gemma2_2b_config,
            mistral_7b_config,
            smollm2_1_7b_config,
        ] {
            let c = config_fn();
            assert_eq!(c.head_dim() * c.num_heads, c.hidden_size);
        }
    }

    #[test]
    fn test_all_configs_valid() {
        for config_fn in [
            phi4_config,
            llama_3_2_1b_config,
            qwen2_5_7b_config,
            gemma2_2b_config,
            mistral_7b_config,
            smollm2_1_7b_config,
        ] {
            assert!(config_fn().validate().is_ok());
        }
    }

    #[test]
    fn test_kv_cache_scales_with_seqlen() {
        let c = phi4_config();
        let short = c.kv_cache_bytes(1024, 2);
        let long = c.kv_cache_bytes(16384, 2);
        assert_eq!(long, short * 16);
    }

    #[test]
    fn test_kv_cache_scales_with_dtype() {
        let c = phi4_config();
        let fp16 = c.kv_cache_bytes(4096, 2);
        let fp32 = c.kv_cache_bytes(4096, 4);
        assert_eq!(fp32, fp16 * 2);
    }

    #[test]
    fn test_clone_config() {
        let c1 = phi4_config();
        let c2 = c1.clone();
        assert_eq!(c1.num_layers, c2.num_layers);
        assert_eq!(c1.hidden_size, c2.hidden_size);
    }

    #[test]
    fn test_debug_format() {
        let c = phi4_config();
        let debug = format!("{:?}", c);
        assert!(debug.contains("5120"));
        assert!(debug.contains("40"));
    }

    #[test]
    fn test_edge_case_single_head() {
        let c = DenseModelConfig {
            num_layers: 1,
            num_heads: 1,
            num_kv_heads: 1,
            hidden_size: 64,
            intermediate_size: 256,
            vocab_size: 100,
            max_position_embeddings: 512,
            rms_norm_eps: 1e-5,
        };
        assert_eq!(c.gqa_groups(), 1);
        assert_eq!(c.head_dim(), 64);
        assert!(c.validate().is_ok());
    }
}

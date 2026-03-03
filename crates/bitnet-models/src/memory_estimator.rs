//! Model memory estimation.
//!
//! Estimate memory requirements for model loading and inference.

/// Data type for memory calculations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    I8,
    I4,
    I2,
}

impl DType {
    pub fn bits(&self) -> usize {
        match self {
            Self::F32 => 32,
            Self::F16 | Self::BF16 => 16,
            Self::I8 => 8,
            Self::I4 => 4,
            Self::I2 => 2,
        }
    }

    pub fn bytes_per_element(&self) -> f64 {
        self.bits() as f64 / 8.0
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::I8 => "int8",
            Self::I4 => "int4",
            Self::I2 => "int2",
        }
    }
}

/// Model configuration for memory estimation.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub vocab_size: usize,
    pub max_context: usize,
    pub weight_dtype: DType,
}

impl ModelSpec {
    pub fn head_dim(&self) -> usize {
        if self.num_heads == 0 {
            return 0;
        }
        self.hidden_size / self.num_heads
    }
}

/// Memory breakdown.
#[derive(Debug, Clone)]
pub struct MemoryEstimate {
    pub weights_bytes: u64,
    pub kv_cache_bytes: u64,
    pub activations_bytes: u64,
    pub total_bytes: u64,
}

impl MemoryEstimate {
    pub fn weights_gb(&self) -> f64 {
        self.weights_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }
    pub fn kv_cache_gb(&self) -> f64 {
        self.kv_cache_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }
    pub fn total_gb(&self) -> f64 {
        self.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }
    pub fn total_mb(&self) -> f64 {
        self.total_bytes as f64 / (1024.0 * 1024.0)
    }
}

/// Estimate memory for a model.
pub fn estimate_memory(spec: &ModelSpec) -> MemoryEstimate {
    let bpe = spec.weight_dtype.bytes_per_element();
    let h = spec.hidden_size as u64;
    let ff = spec.intermediate_size as u64;
    let v = spec.vocab_size as u64;
    let l = spec.num_layers as u64;

    // Weights: embedding + per-layer (QKV + O + gate + up + down + norms) + output
    let embed = v * h;
    let qkv = h * (h + 2 * (spec.num_kv_heads as u64 * spec.head_dim() as u64));
    let output_proj = h * h;
    let ffn = h * ff * 3; // gate + up + down
    let norms = h * 2; // 2 norms per layer
    let per_layer = qkv + output_proj + ffn + norms;
    let total_params = embed + l * per_layer + v * h;
    let weights_bytes = (total_params as f64 * bpe) as u64;

    // KV cache: 2 * num_layers * num_kv_heads * head_dim * max_context * sizeof(f16)
    let kv_per_layer =
        2 * spec.num_kv_heads as u64 * spec.head_dim() as u64 * spec.max_context as u64 * 2; // f16
    let kv_cache_bytes = l * kv_per_layer;

    // Activations: ~2 * batch * seq * hidden (rough estimate)
    let activations_bytes = 2 * spec.max_context as u64 * h * 4; // f32 activations

    let total = weights_bytes + kv_cache_bytes + activations_bytes;

    MemoryEstimate { weights_bytes, kv_cache_bytes, activations_bytes, total_bytes: total }
}

/// Presets for known models.
pub fn phi4_spec() -> ModelSpec {
    ModelSpec {
        num_layers: 40,
        hidden_size: 5120,
        intermediate_size: 14336,
        num_heads: 40,
        num_kv_heads: 10,
        vocab_size: 100352,
        max_context: 16384,
        weight_dtype: DType::BF16,
    }
}

pub fn bitnet_2b_spec() -> ModelSpec {
    ModelSpec {
        num_layers: 30,
        hidden_size: 2560,
        intermediate_size: 6912,
        num_heads: 20,
        num_kv_heads: 5,
        vocab_size: 32000,
        max_context: 4096,
        weight_dtype: DType::I2,
    }
}

pub fn llama3_8b_spec() -> ModelSpec {
    ModelSpec {
        num_layers: 32,
        hidden_size: 4096,
        intermediate_size: 14336,
        num_heads: 32,
        num_kv_heads: 8,
        vocab_size: 128256,
        max_context: 8192,
        weight_dtype: DType::F16,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_bits() {
        assert_eq!(DType::F32.bits(), 32);
        assert_eq!(DType::F16.bits(), 16);
        assert_eq!(DType::I4.bits(), 4);
        assert_eq!(DType::I2.bits(), 2);
    }

    #[test]
    fn test_dtype_bytes() {
        assert!((DType::F32.bytes_per_element() - 4.0).abs() < 0.01);
        assert!((DType::I4.bytes_per_element() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_phi4_estimate() {
        let est = estimate_memory(&phi4_spec());
        assert!(est.weights_gb() > 20.0); // ~28GB BF16
        assert!(est.kv_cache_gb() > 1.0);
        assert!(est.total_gb() > 20.0);
    }

    #[test]
    fn test_bitnet_small() {
        let est = estimate_memory(&bitnet_2b_spec());
        assert!(est.weights_gb() < 5.0); // 2-bit = very small
    }

    #[test]
    fn test_llama_estimate() {
        let est = estimate_memory(&llama3_8b_spec());
        assert!(est.weights_gb() > 10.0);
    }

    #[test]
    fn test_head_dim() {
        assert_eq!(phi4_spec().head_dim(), 128);
        assert_eq!(bitnet_2b_spec().head_dim(), 128);
    }

    #[test]
    fn test_total_mb() {
        let est = estimate_memory(&bitnet_2b_spec());
        assert!(est.total_mb() > 0.0);
    }

    #[test]
    fn test_kv_cache_scales_with_context() {
        let mut spec = bitnet_2b_spec();
        spec.max_context = 4096;
        let est1 = estimate_memory(&spec);
        spec.max_context = 16384;
        let est2 = estimate_memory(&spec);
        assert!(est2.kv_cache_bytes > est1.kv_cache_bytes);
    }

    #[test]
    fn test_dtype_str() {
        assert_eq!(DType::BF16.as_str(), "bf16");
        assert_eq!(DType::I8.as_str(), "int8");
    }

    #[test]
    fn test_zero_heads() {
        let spec = ModelSpec {
            num_layers: 1,
            hidden_size: 128,
            intermediate_size: 256,
            num_heads: 0,
            num_kv_heads: 0,
            vocab_size: 100,
            max_context: 32,
            weight_dtype: DType::F32,
        };
        assert_eq!(spec.head_dim(), 0);
    }

    #[test]
    fn test_quantized_smaller() {
        let mut spec = llama3_8b_spec();
        let fp16_est = estimate_memory(&spec);
        spec.weight_dtype = DType::I4;
        let i4_est = estimate_memory(&spec);
        assert!(i4_est.weights_bytes < fp16_est.weights_bytes);
    }

    #[test]
    fn test_activations_included() {
        let est = estimate_memory(&bitnet_2b_spec());
        assert!(est.activations_bytes > 0);
        assert!(est.total_bytes > est.weights_bytes + est.kv_cache_bytes);
    }
}

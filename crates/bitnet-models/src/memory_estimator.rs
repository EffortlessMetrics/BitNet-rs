//! Model memory estimation.
//!
//! Estimate RAM/VRAM requirements for model loading and inference
//! based on model configuration parameters.

/// Precision of model weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightPrecision {
    /// 1.58-bit (BitNet I2_S)
    Bit158,
    /// 2-bit quantized
    Int2,
    /// 4-bit quantized
    Int4,
    /// 8-bit quantized
    Int8,
    /// 16-bit float (FP16/BF16)
    Float16,
    /// 32-bit float
    Float32,
}

impl WeightPrecision {
    /// Bits per weight element.
    pub fn bits_per_element(&self) -> f64 {
        match self {
            Self::Bit158 => 1.58,
            Self::Int2 => 2.0,
            Self::Int4 => 4.0,
            Self::Int8 => 8.0,
            Self::Float16 => 16.0,
            Self::Float32 => 32.0,
        }
    }

    /// Bytes per element (rounded up for sub-byte types).
    pub fn bytes_per_element(&self) -> f64 {
        self.bits_per_element() / 8.0
    }
}

/// Model configuration for estimation.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub precision: WeightPrecision,
}

impl ModelSpec {
    /// Total model parameters (approximate).
    pub fn total_params(&self) -> u64 {
        let h = self.hidden_size as u64;
        let l = self.num_layers as u64;
        let v = self.vocab_size as u64;
        let ff = self.intermediate_size as u64;

        // Embedding + output projection
        let embedding = v * h * 2;

        // Per-layer: QKV projections + output proj + FFN (gate + up + down) + norms
        let head_dim = h / self.num_heads as u64;
        let qkv = h * (h + 2 * self.num_kv_heads as u64 * head_dim);
        let out_proj = h * h;
        let ffn = h * ff * 3; // gate + up + down
        let norms = h * 4; // 2 norms per layer * 2 (weight + possible bias)
        let per_layer = qkv + out_proj + ffn + norms;

        embedding + l * per_layer
    }

    /// Estimated weight memory in bytes.
    pub fn weight_memory_bytes(&self) -> u64 {
        let params = self.total_params();
        (params as f64 * self.precision.bytes_per_element()) as u64
    }

    /// Estimated KV cache memory for a given batch size and sequence length.
    pub fn kv_cache_bytes(&self, batch_size: usize, seq_len: usize) -> u64 {
        // 2 (K+V) * layers * kv_heads * head_dim * seq_len * batch * 2 bytes (fp16)
        let head_dim = self.hidden_size / self.num_heads;
        2 * self.num_layers as u64
            * self.num_kv_heads as u64
            * head_dim as u64
            * seq_len as u64
            * batch_size as u64
            * 2 // fp16 for KV cache
    }

    /// Total inference memory estimate (weights + KV cache + overhead).
    pub fn inference_memory_bytes(&self, batch_size: usize, seq_len: usize) -> u64 {
        let weights = self.weight_memory_bytes();
        let kv = self.kv_cache_bytes(batch_size, seq_len);
        // ~20% overhead for activations, workspace buffers, etc.
        let overhead = (weights + kv) / 5;
        weights + kv + overhead
    }
}

/// Format bytes as a human-readable string.
pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// Preset: BitNet-b1.58-2B configuration.
pub fn bitnet_2b() -> ModelSpec {
    ModelSpec {
        hidden_size: 2560,
        num_layers: 30,
        num_heads: 20,
        num_kv_heads: 20,
        intermediate_size: 6912,
        vocab_size: 32000,
        max_seq_len: 4096,
        precision: WeightPrecision::Bit158,
    }
}

/// Preset: Phi-4 (14B) configuration.
pub fn phi4_14b() -> ModelSpec {
    ModelSpec {
        hidden_size: 5120,
        num_layers: 40,
        num_heads: 40,
        num_kv_heads: 10,
        intermediate_size: 17920,
        vocab_size: 100352,
        max_seq_len: 16384,
        precision: WeightPrecision::Float16,
    }
}

/// Preset: LLaMA-3 8B configuration.
pub fn llama3_8b() -> ModelSpec {
    ModelSpec {
        hidden_size: 4096,
        num_layers: 32,
        num_heads: 32,
        num_kv_heads: 8,
        intermediate_size: 14336,
        vocab_size: 128256,
        max_seq_len: 8192,
        precision: WeightPrecision::Float16,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_precision_bits() {
        assert!((WeightPrecision::Bit158.bits_per_element() - 1.58).abs() < 1e-6);
        assert!((WeightPrecision::Float16.bits_per_element() - 16.0).abs() < 1e-6);
        assert!((WeightPrecision::Int4.bits_per_element() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_bitnet_2b_params() {
        let spec = bitnet_2b();
        let params = spec.total_params();
        // Should be in the ~2B range
        assert!(params > 1_000_000_000);
        assert!(params < 5_000_000_000);
    }

    #[test]
    fn test_bitnet_2b_weight_memory() {
        let spec = bitnet_2b();
        let mem = spec.weight_memory_bytes();
        // 1.58 bits * ~2B params ≈ ~400 MB
        assert!(mem > 100_000_000);
        assert!(mem < 2_000_000_000);
    }

    #[test]
    fn test_phi4_params() {
        let spec = phi4_14b();
        let params = spec.total_params();
        // Should be in the ~14B range
        assert!(params > 5_000_000_000);
        assert!(params < 30_000_000_000);
    }

    #[test]
    fn test_phi4_weight_memory() {
        let spec = phi4_14b();
        let mem = spec.weight_memory_bytes();
        // FP16 * ~14B params ≈ ~28 GB
        assert!(mem > 10_000_000_000);
        assert!(mem < 60_000_000_000);
    }

    #[test]
    fn test_kv_cache_scaling() {
        let spec = phi4_14b();
        let kv1 = spec.kv_cache_bytes(1, 1024);
        let kv2 = spec.kv_cache_bytes(1, 2048);
        // Double seq_len should double KV cache
        assert_eq!(kv2, kv1 * 2);
    }

    #[test]
    fn test_kv_cache_batch_scaling() {
        let spec = llama3_8b();
        let kv1 = spec.kv_cache_bytes(1, 1024);
        let kv4 = spec.kv_cache_bytes(4, 1024);
        assert_eq!(kv4, kv1 * 4);
    }

    #[test]
    fn test_inference_memory_includes_overhead() {
        let spec = bitnet_2b();
        let weights = spec.weight_memory_bytes();
        let kv = spec.kv_cache_bytes(1, 512);
        let total = spec.inference_memory_bytes(1, 512);
        assert!(total > weights + kv);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1_048_576), "1.00 MB");
        assert_eq!(format_bytes(1_073_741_824), "1.00 GB");
    }

    #[test]
    fn test_format_bytes_fractional() {
        assert_eq!(format_bytes(1_500_000_000), "1.40 GB");
    }

    #[test]
    fn test_int4_precision() {
        let mut spec = bitnet_2b();
        spec.precision = WeightPrecision::Int4;
        let fp16_mem = {
            let mut s = spec.clone();
            s.precision = WeightPrecision::Float16;
            s.weight_memory_bytes()
        };
        let int4_mem = spec.weight_memory_bytes();
        // int4 should be ~4x smaller than fp16
        assert!((fp16_mem as f64 / int4_mem as f64 - 4.0).abs() < 0.1);
    }

    #[test]
    fn test_llama3_kv_cache() {
        let spec = llama3_8b();
        let kv = spec.kv_cache_bytes(1, 8192);
        // 8 KV heads * 128 dim * 32 layers * 8192 seq * 2 (K+V) * 2 bytes
        let expected: u64 = 2 * 32 * 8 * 128 * 8192 * 2;
        assert_eq!(kv, expected);
    }
}

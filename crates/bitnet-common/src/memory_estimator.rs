//! Memory estimation for model tensors and inference buffers.
//!
//! Pre-flight memory planning before loading models.

/// Data types with known byte sizes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DType {
    F32,
    F16,
    BF16,
    I8,
    I4,
    I2,
    Bool,
}

impl DType {
    /// Bits per element.
    pub fn bits(&self) -> usize {
        match self {
            DType::F32 => 32,
            DType::F16 | DType::BF16 => 16,
            DType::I8 => 8,
            DType::I4 => 4,
            DType::I2 => 2,
            DType::Bool => 1,
        }
    }

    /// Bytes needed for `n` elements (rounded up).
    pub fn bytes_for(&self, n: usize) -> usize {
        (n * self.bits()).div_ceil(8)
    }
}

/// A single tensor's memory estimate.
#[derive(Debug, Clone)]
pub struct TensorEstimate {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub bytes: usize,
}

impl TensorEstimate {
    pub fn new(name: &str, shape: &[usize], dtype: DType) -> Self {
        let elements: usize = shape.iter().product();
        Self {
            name: name.to_string(),
            shape: shape.to_vec(),
            dtype,
            bytes: dtype.bytes_for(elements),
        }
    }

    pub fn elements(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Estimate model memory footprint.
#[derive(Debug, Clone)]
pub struct ModelMemoryEstimate {
    pub weight_bytes: u64,
    pub kv_cache_bytes: u64,
    pub activation_bytes: u64,
    pub total_bytes: u64,
    pub tensors: Vec<TensorEstimate>,
}

/// Estimate total memory for a dense transformer model.
pub fn estimate_dense_model(
    vocab_size: usize,
    hidden_size: usize,
    num_layers: usize,
    num_heads: usize,
    num_kv_heads: usize,
    intermediate_size: usize,
    max_seq_len: usize,
    weight_dtype: DType,
) -> ModelMemoryEstimate {
    let mut tensors = Vec::new();

    // Embedding
    let embed = TensorEstimate::new("embed_tokens", &[vocab_size, hidden_size], weight_dtype);
    tensors.push(embed);

    // Per-layer weights
    for l in 0..num_layers {
        let prefix = format!("layer.{l}");
        let head_dim = hidden_size / num_heads;
        let q_size = num_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;

        tensors.push(TensorEstimate::new(
            &format!("{prefix}.q_proj"),
            &[q_size, hidden_size],
            weight_dtype,
        ));
        tensors.push(TensorEstimate::new(
            &format!("{prefix}.k_proj"),
            &[kv_size, hidden_size],
            weight_dtype,
        ));
        tensors.push(TensorEstimate::new(
            &format!("{prefix}.v_proj"),
            &[kv_size, hidden_size],
            weight_dtype,
        ));
        tensors.push(TensorEstimate::new(
            &format!("{prefix}.o_proj"),
            &[hidden_size, q_size],
            weight_dtype,
        ));

        // MLP
        tensors.push(TensorEstimate::new(
            &format!("{prefix}.gate_proj"),
            &[intermediate_size, hidden_size],
            weight_dtype,
        ));
        tensors.push(TensorEstimate::new(
            &format!("{prefix}.up_proj"),
            &[intermediate_size, hidden_size],
            weight_dtype,
        ));
        tensors.push(TensorEstimate::new(
            &format!("{prefix}.down_proj"),
            &[hidden_size, intermediate_size],
            weight_dtype,
        ));

        // Norms (always f32)
        tensors.push(TensorEstimate::new(
            &format!("{prefix}.input_norm"),
            &[hidden_size],
            DType::F32,
        ));
        tensors.push(TensorEstimate::new(
            &format!("{prefix}.post_norm"),
            &[hidden_size],
            DType::F32,
        ));
    }

    // LM head
    tensors.push(TensorEstimate::new("lm_head", &[vocab_size, hidden_size], weight_dtype));

    let weight_bytes: u64 = tensors.iter().map(|t| t.bytes as u64).sum();

    // KV cache: 2 (K+V) * num_layers * max_seq * num_kv_heads * head_dim * sizeof(f16)
    let head_dim = hidden_size / num_heads;
    let kv_per_layer = 2 * max_seq_len * num_kv_heads * head_dim;
    let kv_cache_bytes = (num_layers as u64) * (kv_per_layer as u64) * 2; // f16

    // Activation buffer: ~batch * seq * hidden * 4 (f32)
    let activation_bytes = (max_seq_len * hidden_size * 4) as u64;

    let total_bytes = weight_bytes + kv_cache_bytes + activation_bytes;

    ModelMemoryEstimate { weight_bytes, kv_cache_bytes, activation_bytes, total_bytes, tensors }
}

/// Format bytes into human-readable string.
pub fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.2} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.2} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_bits() {
        assert_eq!(DType::F32.bits(), 32);
        assert_eq!(DType::F16.bits(), 16);
        assert_eq!(DType::I2.bits(), 2);
    }

    #[test]
    fn test_dtype_bytes_for() {
        assert_eq!(DType::F32.bytes_for(10), 40);
        assert_eq!(DType::F16.bytes_for(10), 20);
        assert_eq!(DType::I4.bytes_for(2), 1);
        assert_eq!(DType::I2.bytes_for(4), 1);
    }

    #[test]
    fn test_dtype_bytes_rounding() {
        assert_eq!(DType::Bool.bytes_for(7), 1);
        assert_eq!(DType::Bool.bytes_for(9), 2);
    }

    #[test]
    fn test_tensor_estimate() {
        let t = TensorEstimate::new("w", &[100, 200], DType::F32);
        assert_eq!(t.elements(), 20_000);
        assert_eq!(t.bytes, 80_000);
    }

    #[test]
    fn test_tensor_empty() {
        let t = TensorEstimate::new("w", &[0, 100], DType::F16);
        assert_eq!(t.elements(), 0);
        assert_eq!(t.bytes, 0);
    }

    #[test]
    fn test_small_model() {
        let est = estimate_dense_model(1000, 64, 2, 4, 2, 128, 128, DType::F16);
        assert!(est.weight_bytes > 0);
        assert!(est.kv_cache_bytes > 0);
        assert!(est.total_bytes > est.weight_bytes);
    }

    #[test]
    fn test_phi4_estimate() {
        let est = estimate_dense_model(100_352, 5120, 40, 40, 10, 13824, 16384, DType::F16);
        assert!(est.weight_bytes > 20_000_000_000);
        assert!(est.weight_bytes < 40_000_000_000);
        assert!(est.kv_cache_bytes > 2_000_000_000);
    }

    #[test]
    fn test_format_bytes_gb() {
        assert!(format_bytes(2_000_000_000).contains("GB"));
    }

    #[test]
    fn test_format_bytes_mb() {
        assert!(format_bytes(5_000_000).contains("MB"));
    }

    #[test]
    fn test_format_bytes_kb() {
        assert!(format_bytes(2048).contains("KB"));
    }

    #[test]
    fn test_format_bytes_b() {
        assert_eq!(format_bytes(42), "42 B");
    }

    #[test]
    fn test_tensor_count() {
        let est = estimate_dense_model(100, 64, 2, 4, 2, 128, 64, DType::F32);
        // 1 embed + 2*(7 weights + 2 norms) + 1 lm_head = 20
        assert_eq!(est.tensors.len(), 20);
    }

    #[test]
    fn test_activation_buffer() {
        let est = estimate_dense_model(100, 64, 1, 4, 2, 128, 256, DType::F32);
        // activation = 256 * 64 * 4 = 65536
        assert_eq!(est.activation_bytes, 65536);
    }
}

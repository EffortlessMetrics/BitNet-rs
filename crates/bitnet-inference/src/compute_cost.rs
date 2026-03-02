//! Compute cost estimation for inference operations.

/// Model dimensions for cost estimation.
#[derive(Debug, Clone)]
pub struct ModelDims {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub head_dim: usize,
}

impl ModelDims {
    pub fn new(
        hidden: usize,
        layers: usize,
        heads: usize,
        kv_heads: usize,
        intermediate: usize,
        vocab: usize,
    ) -> Self {
        let head_dim = if heads > 0 { hidden / heads } else { 0 };
        Self {
            hidden_size: hidden,
            num_layers: layers,
            num_heads: heads,
            num_kv_heads: kv_heads,
            intermediate_size: intermediate,
            vocab_size: vocab,
            head_dim,
        }
    }
}

/// FLOPs breakdown for one token.
#[derive(Debug, Clone)]
pub struct FlopsEstimate {
    pub attention_flops: u64,
    pub ffn_flops: u64,
    pub lm_head_flops: u64,
    pub total_flops: u64,
}

pub fn estimate_flops_per_token(dims: &ModelDims) -> FlopsEstimate {
    let h = dims.hidden_size as u64;
    let l = dims.num_layers as u64;
    let hd = dims.head_dim as u64;
    let nkv = dims.num_kv_heads as u64;
    let i = dims.intermediate_size as u64;
    let v = dims.vocab_size as u64;

    let attn_per_layer = 2 * h * (h + 2 * nkv * hd) + 2 * h * h;
    let attention_flops = attn_per_layer * l;
    let ffn_flops = 6 * h * i * l;
    let lm_head_flops = 2 * h * v;
    let total = attention_flops + ffn_flops + lm_head_flops;

    FlopsEstimate { attention_flops, ffn_flops, lm_head_flops, total_flops: total }
}

/// Memory bandwidth estimate (bytes/token).
#[derive(Debug, Clone)]
pub struct BandwidthEstimate {
    pub weight_bytes: u64,
    pub kv_cache_bytes: u64,
    pub total_bytes: u64,
}

pub fn estimate_bandwidth(
    dims: &ModelDims,
    seq_len: usize,
    bytes_per_weight: u64,
) -> BandwidthEstimate {
    let h = dims.hidden_size as u64;
    let l = dims.num_layers as u64;
    let hd = dims.head_dim as u64;
    let nkv = dims.num_kv_heads as u64;
    let i = dims.intermediate_size as u64;
    let v = dims.vocab_size as u64;
    let s = seq_len as u64;

    let attn_w = h * h + 2 * h * nkv * hd + h * h;
    let ffn_w = 3 * h * i;
    let weight_bytes = (attn_w + ffn_w) * l * bytes_per_weight + h * v * bytes_per_weight;
    let kv_cache_bytes = 2 * nkv * hd * s * 2 * l;
    BandwidthEstimate { weight_bytes, kv_cache_bytes, total_bytes: weight_bytes + kv_cache_bytes }
}

pub fn estimate_tps(dims: &ModelDims, seq_len: usize, bytes_per_weight: u64, bw_gb_s: f64) -> f64 {
    let bw = estimate_bandwidth(dims, seq_len, bytes_per_weight);
    if bw.total_bytes == 0 {
        return 0.0;
    }
    bw_gb_s * 1e9 / bw.total_bytes as f64
}

pub fn phi4_dims() -> ModelDims {
    ModelDims::new(5120, 40, 40, 10, 17920, 100352)
}
pub fn llama3_8b_dims() -> ModelDims {
    ModelDims::new(4096, 32, 32, 8, 14336, 128256)
}
pub fn smollm2_dims() -> ModelDims {
    ModelDims::new(2048, 24, 32, 32, 8192, 49152)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_dims() {
        let d = ModelDims::new(4096, 32, 32, 8, 14336, 128256);
        assert_eq!(d.head_dim, 128);
    }

    #[test]
    fn test_zero_heads() {
        let d = ModelDims::new(4096, 32, 0, 0, 14336, 128256);
        assert_eq!(d.head_dim, 0);
    }

    #[test]
    fn test_flops_nonzero() {
        let f = estimate_flops_per_token(&phi4_dims());
        assert!(f.total_flops > 0);
        assert!(f.attention_flops > 0);
        assert!(f.ffn_flops > 0);
    }

    #[test]
    fn test_flops_billions() {
        let f = estimate_flops_per_token(&phi4_dims());
        assert!(f.total_flops > 1_000_000_000);
    }

    #[test]
    fn test_smaller_fewer_flops() {
        let small = estimate_flops_per_token(&smollm2_dims());
        let big = estimate_flops_per_token(&phi4_dims());
        assert!(small.total_flops < big.total_flops);
    }

    #[test]
    fn test_bandwidth_nonzero() {
        let bw = estimate_bandwidth(&llama3_8b_dims(), 512, 2);
        assert!(bw.total_bytes > 0);
    }

    #[test]
    fn test_bandwidth_seq_len() {
        let short = estimate_bandwidth(&llama3_8b_dims(), 100, 2);
        let long = estimate_bandwidth(&llama3_8b_dims(), 1000, 2);
        assert!(long.kv_cache_bytes > short.kv_cache_bytes);
    }

    #[test]
    fn test_bandwidth_precision() {
        let f16 = estimate_bandwidth(&llama3_8b_dims(), 512, 2);
        let f32 = estimate_bandwidth(&llama3_8b_dims(), 512, 4);
        assert!(f32.weight_bytes > f16.weight_bytes);
    }

    #[test]
    fn test_tps() {
        let tps = estimate_tps(&smollm2_dims(), 100, 2, 50.0);
        assert!(tps > 0.0);
        assert!(tps < 1000.0);
    }

    #[test]
    fn test_tps_zero() {
        let d = ModelDims::new(0, 0, 0, 0, 0, 0);
        assert_eq!(estimate_tps(&d, 0, 0, 50.0), 0.0);
    }

    #[test]
    fn test_phi4() {
        let d = phi4_dims();
        assert_eq!(d.hidden_size, 5120);
        assert_eq!(d.num_layers, 40);
    }

    #[test]
    fn test_llama3() {
        let d = llama3_8b_dims();
        assert_eq!(d.num_kv_heads, 8);
    }

    #[test]
    fn test_smollm2() {
        let d = smollm2_dims();
        assert_eq!(d.num_layers, 24);
    }

    #[test]
    fn test_ffn_dominates() {
        let f = estimate_flops_per_token(&phi4_dims());
        assert!(f.ffn_flops > f.attention_flops);
    }

    #[test]
    fn test_lm_head() {
        let f = estimate_flops_per_token(&phi4_dims());
        assert!(f.lm_head_flops > 1_000_000);
    }
}

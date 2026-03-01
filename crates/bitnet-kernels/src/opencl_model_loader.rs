//! GGUF model weight loader for the OpenCL inference pipeline.
//!
//! Loads model weights from GGUF format and prepares them for OpenCL
//! inference, handling I2_S quantized tensors and weight mapping.

use std::fmt;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Model architecture configuration extracted from GGUF metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub intermediate_dim: usize,
    pub max_position: usize,
    pub rope_base: f32,
}

/// Per-layer weight tensors (dequantized to f32).
#[derive(Debug, Clone)]
pub struct LayerWeights {
    pub attention_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub o_proj: Vec<f32>,
    pub gate_proj: Vec<f32>,
    pub up_proj: Vec<f32>,
    pub down_proj: Vec<f32>,
}

/// Complete model weights ready for OpenCL upload.
#[derive(Debug, Clone)]
pub struct ModelWeights {
    pub config: ModelConfig,
    pub token_embedding: Vec<f32>,
    pub layers: Vec<LayerWeights>,
    pub output_norm: Vec<f32>,
    pub lm_head: Vec<f32>,
}

/// Quantization format of a stored tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    Float32,
    Float16,
    I2S,
    QK256,
}

/// Metadata describing a single tensor inside a GGUF file.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorMeta {
    pub name: String,
    pub shape: Vec<usize>,
    pub format: WeightFormat,
    pub offset: usize,
    pub size_bytes: usize,
}

/// Errors that can occur during model loading.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelLoadError {
    MissingTensor(String),
    ShapeMismatch,
    UnsupportedFormat,
    InvalidConfig,
    IOError(String),
}

impl fmt::Display for ModelLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingTensor(name) => {
                write!(f, "missing tensor: {name}")
            }
            Self::ShapeMismatch => write!(f, "shape mismatch"),
            Self::UnsupportedFormat => write!(f, "unsupported format"),
            Self::InvalidConfig => write!(f, "invalid config"),
            Self::IOError(msg) => write!(f, "I/O error: {msg}"),
        }
    }
}

impl std::error::Error for ModelLoadError {}

// ---------------------------------------------------------------------------
// CPU reference implementations
// ---------------------------------------------------------------------------

/// Create a [`ModelConfig`] for a known model size variant.
///
/// Supported variants: `"tiny"` (4 layers/64 dim), `"small"` (8/128),
/// `"medium"` (16/256).
pub fn create_mock_model_config(variant: &str) -> ModelConfig {
    match variant {
        "tiny" => ModelConfig {
            vocab_size: 256,
            hidden_dim: 64,
            num_layers: 4,
            num_heads: 4,
            head_dim: 16,
            intermediate_dim: 128,
            max_position: 512,
            rope_base: 10000.0,
        },
        "small" => ModelConfig {
            vocab_size: 1024,
            hidden_dim: 128,
            num_layers: 8,
            num_heads: 8,
            head_dim: 16,
            intermediate_dim: 256,
            max_position: 1024,
            rope_base: 10000.0,
        },
        "medium" => ModelConfig {
            vocab_size: 4096,
            hidden_dim: 256,
            num_layers: 16,
            num_heads: 8,
            head_dim: 32,
            intermediate_dim: 512,
            max_position: 2048,
            rope_base: 10000.0,
        },
        _ => ModelConfig {
            vocab_size: 256,
            hidden_dim: 64,
            num_layers: 4,
            num_heads: 4,
            head_dim: 16,
            intermediate_dim: 128,
            max_position: 512,
            rope_base: 10000.0,
        },
    }
}

/// Generate deterministic mock weights from a seed (simple LCG PRNG).
pub fn create_mock_weights(config: &ModelConfig, seed: u64) -> ModelWeights {
    let mut rng_state = seed;
    let mut next_f32 = move || -> f32 {
        // LCG parameters (Numerical Recipes)
        rng_state = rng_state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        // Map upper bits to [-1, 1]
        let bits = ((rng_state >> 33) as u32) as f32;
        bits / (u32::MAX as f32 / 2.0) - 1.0
    };
    let mut fill = |n: usize| -> Vec<f32> { (0..n).map(|_| next_f32()).collect() };

    let h = config.hidden_dim;
    let inter = config.intermediate_dim;

    let token_embedding = fill(config.vocab_size * h);

    let layers = (0..config.num_layers)
        .map(|_| LayerWeights {
            attention_norm: fill(h),
            ffn_norm: fill(h),
            q_proj: fill(h * h),
            k_proj: fill(h * h),
            v_proj: fill(h * h),
            o_proj: fill(h * h),
            gate_proj: fill(inter * h),
            up_proj: fill(inter * h),
            down_proj: fill(h * inter),
        })
        .collect();

    let output_norm = fill(h);
    let lm_head = fill(config.vocab_size * h);

    ModelWeights { config: config.clone(), token_embedding, layers, output_norm, lm_head }
}

/// Validate internal consistency of a [`ModelConfig`].
pub fn validate_model_config(config: &ModelConfig) -> Result<(), ModelLoadError> {
    if config.head_dim * config.num_heads != config.hidden_dim {
        return Err(ModelLoadError::InvalidConfig);
    }
    if config.hidden_dim == 0
        || config.num_layers == 0
        || config.vocab_size == 0
        || config.num_heads == 0
        || config.intermediate_dim == 0
        || config.max_position == 0
    {
        return Err(ModelLoadError::InvalidConfig);
    }
    if config.rope_base <= 0.0 {
        return Err(ModelLoadError::InvalidConfig);
    }
    Ok(())
}

/// Generate the expected tensor name patterns for a BitNet model.
pub fn map_tensor_names(config: &ModelConfig) -> Vec<TensorMeta> {
    let h = config.hidden_dim;
    let inter = config.intermediate_dim;
    let vocab = config.vocab_size;
    let mut metas = Vec::new();
    let mut offset: usize = 0;

    let mut push = |name: String, shape: Vec<usize>, format: WeightFormat| {
        let elems: usize = shape.iter().product();
        let size_bytes = match format {
            WeightFormat::Float32 => elems * 4,
            WeightFormat::Float16 => elems * 2,
            // I2_S: 2 bits per element, packed into bytes
            WeightFormat::I2S => elems.div_ceil(4),
            // QK256: 256-element blocks, each block = 64 bytes data
            // + 2 bytes scale
            WeightFormat::QK256 => {
                let nblocks = elems.div_ceil(256);
                nblocks * 66
            }
        };
        metas.push(TensorMeta { name, shape, format, offset, size_bytes });
        offset += size_bytes;
    };

    // Token embedding
    push("model.embed_tokens.weight".into(), vec![vocab, h], WeightFormat::Float32);

    for i in 0..config.num_layers {
        let pfx = format!("model.layers.{i}");

        // Attention norms (always f32)
        push(format!("{pfx}.input_layernorm.weight"), vec![h], WeightFormat::Float32);

        // Attention projections (I2_S quantized)
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            push(format!("{pfx}.self_attn.{proj}.weight"), vec![h, h], WeightFormat::I2S);
        }

        // FFN norm
        push(format!("{pfx}.post_attention_layernorm.weight"), vec![h], WeightFormat::Float32);

        // FFN projections (I2_S quantized)
        push(format!("{pfx}.mlp.gate_proj.weight"), vec![inter, h], WeightFormat::I2S);
        push(format!("{pfx}.mlp.up_proj.weight"), vec![inter, h], WeightFormat::I2S);
        push(format!("{pfx}.mlp.down_proj.weight"), vec![h, inter], WeightFormat::I2S);
    }

    // Output norm + LM head
    push("model.norm.weight".into(), vec![h], WeightFormat::Float32);
    push("lm_head.weight".into(), vec![vocab, h], WeightFormat::Float32);

    metas
}

/// Estimate total weight memory in bytes for a given config and format.
pub fn compute_weight_memory(config: &ModelConfig, format: WeightFormat) -> usize {
    let h = config.hidden_dim;
    let inter = config.intermediate_dim;
    let vocab = config.vocab_size;

    // Embedding + LM head (always f32 in practice)
    let embed_elems = vocab * h;
    let lm_head_elems = vocab * h;
    let norms_elems = h * (config.num_layers * 2 + 1); // per-layer × 2 + output

    let f32_bytes = (embed_elems + lm_head_elems + norms_elems) * 4;

    // Per-layer weight elements
    let attn_elems = 4 * h * h; // q, k, v, o
    let ffn_elems = 2 * inter * h + h * inter; // gate, up, down
    let layer_elems = attn_elems + ffn_elems;
    let total_weight_elems = layer_elems * config.num_layers;

    let weight_bytes = match format {
        WeightFormat::Float32 => total_weight_elems * 4,
        WeightFormat::Float16 => total_weight_elems * 2,
        WeightFormat::I2S => total_weight_elems.div_ceil(4),
        WeightFormat::QK256 => {
            let nblocks = total_weight_elems.div_ceil(256);
            nblocks * 66
        }
    };

    f32_bytes + weight_bytes
}

/// Dequantize a single I2_S block: each byte packs 4 × 2-bit values
/// mapped as `{0b00 → -1.0, 0b01 → 0.0, 0b10 → 1.0}`.
pub fn cpu_dequantize_i2s_block(packed: &[u8], block_size: usize) -> Vec<f32> {
    let i2s_to_f32 = |bits: u8| -> f32 {
        match bits & 0b11 {
            0b00 => -1.0,
            0b01 => 0.0,
            0b10 => 1.0,
            _ => 0.0, // 0b11 treated as zero (unused sentinel)
        }
    };

    let mut out = Vec::with_capacity(block_size);
    for &byte in packed {
        if out.len() >= block_size {
            break;
        }
        // LSB-first packing order
        out.push(i2s_to_f32(byte));
        if out.len() < block_size {
            out.push(i2s_to_f32(byte >> 2));
        }
        if out.len() < block_size {
            out.push(i2s_to_f32(byte >> 4));
        }
        if out.len() < block_size {
            out.push(i2s_to_f32(byte >> 6));
        }
    }
    out
}

/// Group tensor metadata entries by layer index.
///
/// Non-layer tensors (embedding, output norm, lm_head) are placed in a
/// final group at the end.
pub fn partition_weights_by_layer(metas: &[TensorMeta]) -> Vec<Vec<&TensorMeta>> {
    // Determine max layer index present
    let mut max_layer: Option<usize> = None;
    for m in metas {
        if let Some(idx) = extract_layer_index(&m.name) {
            max_layer = Some(match max_layer {
                Some(cur) => cur.max(idx),
                None => idx,
            });
        }
    }

    let num_layers = max_layer.map_or(0, |m| m + 1);
    // +1 bucket for non-layer tensors
    let mut buckets: Vec<Vec<&TensorMeta>> = vec![Vec::new(); num_layers + 1];

    for m in metas {
        if let Some(idx) = extract_layer_index(&m.name) {
            buckets[idx].push(m);
        } else {
            buckets[num_layers].push(m);
        }
    }

    buckets
}

/// Validate that every tensor in `weights` has shapes consistent with the
/// embedded config. Returns a list of human-readable mismatch descriptions
/// (empty if everything is valid).
pub fn validate_weight_shapes(weights: &ModelWeights) -> Vec<String> {
    let c = &weights.config;
    let h = c.hidden_dim;
    let inter = c.intermediate_dim;
    let vocab = c.vocab_size;
    let mut errors = Vec::new();

    let check = |errors: &mut Vec<String>, name: &str, actual: usize, expected: usize| {
        if actual != expected {
            errors.push(format!("{name}: expected {expected} elements, got {actual}"));
        }
    };

    check(&mut errors, "token_embedding", weights.token_embedding.len(), vocab * h);
    check(&mut errors, "output_norm", weights.output_norm.len(), h);
    check(&mut errors, "lm_head", weights.lm_head.len(), vocab * h);

    if weights.layers.len() != c.num_layers {
        errors.push(format!("num_layers: expected {}, got {}", c.num_layers, weights.layers.len()));
    }

    for (i, layer) in weights.layers.iter().enumerate() {
        let lp = format!("layer[{i}]");
        check(&mut errors, &format!("{lp}.attention_norm"), layer.attention_norm.len(), h);
        check(&mut errors, &format!("{lp}.ffn_norm"), layer.ffn_norm.len(), h);
        check(&mut errors, &format!("{lp}.q_proj"), layer.q_proj.len(), h * h);
        check(&mut errors, &format!("{lp}.k_proj"), layer.k_proj.len(), h * h);
        check(&mut errors, &format!("{lp}.v_proj"), layer.v_proj.len(), h * h);
        check(&mut errors, &format!("{lp}.o_proj"), layer.o_proj.len(), h * h);
        check(&mut errors, &format!("{lp}.gate_proj"), layer.gate_proj.len(), inter * h);
        check(&mut errors, &format!("{lp}.up_proj"), layer.up_proj.len(), inter * h);
        check(&mut errors, &format!("{lp}.down_proj"), layer.down_proj.len(), h * inter);
    }

    errors
}

/// Estimate OpenCL memory requirements for the A770 (or similar) GPU.
///
/// Returns `(weight_bytes, activation_bytes)`.
pub fn estimate_opencl_memory(config: &ModelConfig, format: WeightFormat) -> (usize, usize) {
    let weight_bytes = compute_weight_memory(config, format);

    // Activation memory: we need buffers for hidden states, attention
    // scores, and intermediate FFN activations.
    let h = config.hidden_dim;
    let inter = config.intermediate_dim;
    let seq = config.max_position;

    // Hidden state buffer (seq × hidden)
    let hidden_buf = seq * h * 4;
    // Attention scores (num_heads × seq × seq)
    let attn_scores = config.num_heads * seq * seq * 4;
    // FFN intermediate (seq × intermediate_dim)
    let ffn_buf = seq * inter * 4;
    // KV cache (2 × num_layers × seq × hidden)
    let kv_cache = 2 * config.num_layers * seq * h * 4;

    let activation_bytes = hidden_buf + attn_scores + ffn_buf + kv_cache;

    (weight_bytes, activation_bytes)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract the layer index from a tensor name like
/// `"model.layers.5.self_attn.q_proj.weight"`.
fn extract_layer_index(name: &str) -> Option<usize> {
    let parts: Vec<&str> = name.split('.').collect();
    for (i, part) in parts.iter().enumerate() {
        if *part == "layers" {
            return parts.get(i + 1).and_then(|s| s.parse().ok());
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Config validation ---------------------------------------------------

    #[test]
    fn test_valid_tiny_config() {
        let cfg = create_mock_model_config("tiny");
        assert!(validate_model_config(&cfg).is_ok());
    }

    #[test]
    fn test_valid_small_config() {
        let cfg = create_mock_model_config("small");
        assert!(validate_model_config(&cfg).is_ok());
    }

    #[test]
    fn test_valid_medium_config() {
        let cfg = create_mock_model_config("medium");
        assert!(validate_model_config(&cfg).is_ok());
    }

    #[test]
    fn test_head_dim_mismatch_caught() {
        let mut cfg = create_mock_model_config("tiny");
        cfg.head_dim = 999;
        assert_eq!(validate_model_config(&cfg), Err(ModelLoadError::InvalidConfig));
    }

    #[test]
    fn test_zero_hidden_dim_rejected() {
        let mut cfg = create_mock_model_config("tiny");
        cfg.hidden_dim = 0;
        cfg.head_dim = 0;
        assert_eq!(validate_model_config(&cfg), Err(ModelLoadError::InvalidConfig));
    }

    #[test]
    fn test_zero_layers_rejected() {
        let mut cfg = create_mock_model_config("tiny");
        cfg.num_layers = 0;
        assert_eq!(validate_model_config(&cfg), Err(ModelLoadError::InvalidConfig));
    }

    #[test]
    fn test_zero_vocab_rejected() {
        let mut cfg = create_mock_model_config("tiny");
        cfg.vocab_size = 0;
        assert_eq!(validate_model_config(&cfg), Err(ModelLoadError::InvalidConfig));
    }

    #[test]
    fn test_negative_rope_base_rejected() {
        let mut cfg = create_mock_model_config("tiny");
        cfg.rope_base = -1.0;
        assert_eq!(validate_model_config(&cfg), Err(ModelLoadError::InvalidConfig));
    }

    #[test]
    fn test_zero_rope_base_rejected() {
        let mut cfg = create_mock_model_config("tiny");
        cfg.rope_base = 0.0;
        assert_eq!(validate_model_config(&cfg), Err(ModelLoadError::InvalidConfig));
    }

    // -- Tensor name mapping -------------------------------------------------

    #[test]
    fn test_tensor_names_contain_all_layers() {
        let cfg = create_mock_model_config("tiny");
        let metas = map_tensor_names(&cfg);
        for i in 0..cfg.num_layers {
            let prefix = format!("model.layers.{i}");
            assert!(
                metas.iter().any(|m| m.name.starts_with(&prefix)),
                "missing tensors for layer {i}"
            );
        }
    }

    #[test]
    fn test_tensor_names_include_embedding() {
        let cfg = create_mock_model_config("tiny");
        let metas = map_tensor_names(&cfg);
        assert!(metas.iter().any(|m| m.name == "model.embed_tokens.weight"));
    }

    #[test]
    fn test_tensor_names_include_lm_head() {
        let cfg = create_mock_model_config("tiny");
        let metas = map_tensor_names(&cfg);
        assert!(metas.iter().any(|m| m.name == "lm_head.weight"));
    }

    #[test]
    fn test_tensor_names_include_output_norm() {
        let cfg = create_mock_model_config("tiny");
        let metas = map_tensor_names(&cfg);
        assert!(metas.iter().any(|m| m.name == "model.norm.weight"));
    }

    #[test]
    fn test_tensor_names_correct_attn_proj_names() {
        let cfg = create_mock_model_config("tiny");
        let metas = map_tensor_names(&cfg);
        let layer0: Vec<_> =
            metas.iter().filter(|m| m.name.starts_with("model.layers.0.self_attn")).collect();
        assert_eq!(layer0.len(), 4, "expect q/k/v/o projections");
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            let expected = format!("model.layers.0.self_attn.{proj}.weight");
            assert!(layer0.iter().any(|m| m.name == expected), "missing {expected}");
        }
    }

    #[test]
    fn test_tensor_names_correct_ffn_names() {
        let cfg = create_mock_model_config("small");
        let metas = map_tensor_names(&cfg);
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            let expected = format!("model.layers.0.mlp.{proj}.weight");
            assert!(metas.iter().any(|m| m.name == expected), "missing {expected}");
        }
    }

    // -- Memory estimation ---------------------------------------------------

    #[test]
    fn test_memory_estimation_tiny_fits_16gb() {
        let cfg = create_mock_model_config("tiny");
        let mem = compute_weight_memory(&cfg, WeightFormat::I2S);
        let sixteen_gb = 16 * 1024 * 1024 * 1024;
        assert!(mem < sixteen_gb, "tiny model should fit in 16 GB, got {mem}");
    }

    #[test]
    fn test_memory_f32_larger_than_i2s() {
        let cfg = create_mock_model_config("small");
        let f32_mem = compute_weight_memory(&cfg, WeightFormat::Float32);
        let i2s_mem = compute_weight_memory(&cfg, WeightFormat::I2S);
        assert!(f32_mem > i2s_mem);
    }

    #[test]
    fn test_memory_f16_larger_than_i2s() {
        let cfg = create_mock_model_config("small");
        let f16_mem = compute_weight_memory(&cfg, WeightFormat::Float16);
        let i2s_mem = compute_weight_memory(&cfg, WeightFormat::I2S);
        assert!(f16_mem > i2s_mem);
    }

    #[test]
    fn test_memory_scales_linearly_with_layers() {
        // Use a config where per-layer weights dominate the constant
        // embedding/norm overhead so the scaling ratio is close to 2×.
        let base = ModelConfig {
            vocab_size: 32,
            hidden_dim: 256,
            num_layers: 16,
            num_heads: 8,
            head_dim: 32,
            intermediate_dim: 512,
            max_position: 512,
            rope_base: 10000.0,
        };
        let mem16 = compute_weight_memory(&base, WeightFormat::I2S);

        let doubled = ModelConfig { num_layers: 32, ..base };
        let mem32 = compute_weight_memory(&doubled, WeightFormat::I2S);

        let ratio = mem32 as f64 / mem16 as f64;
        assert!((1.5..2.5).contains(&ratio), "expected roughly 2× scaling, got {ratio:.2}×");
    }

    // -- Mock weights --------------------------------------------------------

    #[test]
    fn test_mock_weights_correct_embedding_shape() {
        let cfg = create_mock_model_config("tiny");
        let w = create_mock_weights(&cfg, 42);
        assert_eq!(w.token_embedding.len(), cfg.vocab_size * cfg.hidden_dim);
    }

    #[test]
    fn test_mock_weights_correct_layer_count() {
        let cfg = create_mock_model_config("small");
        let w = create_mock_weights(&cfg, 42);
        assert_eq!(w.layers.len(), cfg.num_layers);
    }

    #[test]
    fn test_mock_weights_correct_layer_shapes() {
        let cfg = create_mock_model_config("tiny");
        let w = create_mock_weights(&cfg, 42);
        let h = cfg.hidden_dim;
        let inter = cfg.intermediate_dim;
        for layer in &w.layers {
            assert_eq!(layer.attention_norm.len(), h);
            assert_eq!(layer.ffn_norm.len(), h);
            assert_eq!(layer.q_proj.len(), h * h);
            assert_eq!(layer.k_proj.len(), h * h);
            assert_eq!(layer.v_proj.len(), h * h);
            assert_eq!(layer.o_proj.len(), h * h);
            assert_eq!(layer.gate_proj.len(), inter * h);
            assert_eq!(layer.up_proj.len(), inter * h);
            assert_eq!(layer.down_proj.len(), h * inter);
        }
    }

    #[test]
    fn test_mock_weights_deterministic_same_seed() {
        let cfg = create_mock_model_config("tiny");
        let w1 = create_mock_weights(&cfg, 42);
        let w2 = create_mock_weights(&cfg, 42);
        assert_eq!(w1.token_embedding, w2.token_embedding);
        assert_eq!(w1.lm_head, w2.lm_head);
    }

    #[test]
    fn test_mock_weights_different_seeds_differ() {
        let cfg = create_mock_model_config("tiny");
        let w1 = create_mock_weights(&cfg, 42);
        let w2 = create_mock_weights(&cfg, 99);
        assert_ne!(w1.token_embedding, w2.token_embedding);
    }

    #[test]
    fn test_mock_weights_output_norm_shape() {
        let cfg = create_mock_model_config("medium");
        let w = create_mock_weights(&cfg, 1);
        assert_eq!(w.output_norm.len(), cfg.hidden_dim);
    }

    #[test]
    fn test_mock_weights_lm_head_shape() {
        let cfg = create_mock_model_config("medium");
        let w = create_mock_weights(&cfg, 1);
        assert_eq!(w.lm_head.len(), cfg.vocab_size * cfg.hidden_dim);
    }

    // -- I2_S dequantization -------------------------------------------------

    #[test]
    fn test_dequantize_i2s_known_values() {
        // 0b10_01_00_01 = byte 0x99 → values: 01→0, 00→-1, 01→0, 10→1
        let packed = [0b10_01_00_01u8];
        let result = cpu_dequantize_i2s_block(&packed, 4);
        assert_eq!(result, vec![0.0, -1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_dequantize_i2s_all_negative_one() {
        // 0b00_00_00_00 = 0x00 → all -1
        let packed = [0x00u8];
        let result = cpu_dequantize_i2s_block(&packed, 4);
        assert_eq!(result, vec![-1.0, -1.0, -1.0, -1.0]);
    }

    #[test]
    fn test_dequantize_i2s_all_zero() {
        // 0b01_01_01_01 = 0x55 → all 0
        let packed = [0x55u8];
        let result = cpu_dequantize_i2s_block(&packed, 4);
        assert_eq!(result, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_dequantize_i2s_all_positive_one() {
        // 0b10_10_10_10 = 0xAA → all +1
        let packed = [0xAAu8];
        let result = cpu_dequantize_i2s_block(&packed, 4);
        assert_eq!(result, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_dequantize_i2s_partial_block() {
        let packed = [0xAAu8];
        let result = cpu_dequantize_i2s_block(&packed, 2);
        assert_eq!(result, vec![1.0, 1.0]);
    }

    #[test]
    fn test_dequantize_i2s_multi_byte() {
        let packed = [0x00u8, 0xAAu8]; // -1,-1,-1,-1, +1,+1,+1,+1
        let result = cpu_dequantize_i2s_block(&packed, 8);
        assert_eq!(result, vec![-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    // -- Weight partitioning -------------------------------------------------

    #[test]
    fn test_partition_correct_layer_count() {
        let cfg = create_mock_model_config("tiny");
        let metas = map_tensor_names(&cfg);
        let parts = partition_weights_by_layer(&metas);
        // num_layers buckets + 1 for non-layer tensors
        assert_eq!(parts.len(), cfg.num_layers + 1);
    }

    #[test]
    fn test_partition_non_layer_tensors_in_last_bucket() {
        let cfg = create_mock_model_config("tiny");
        let metas = map_tensor_names(&cfg);
        let parts = partition_weights_by_layer(&metas);
        let last = parts.last().unwrap();
        assert!(last.iter().any(|m| m.name == "model.embed_tokens.weight"));
        assert!(last.iter().any(|m| m.name == "lm_head.weight"));
    }

    #[test]
    fn test_partition_each_layer_has_tensors() {
        let cfg = create_mock_model_config("small");
        let metas = map_tensor_names(&cfg);
        let parts = partition_weights_by_layer(&metas);
        for (i, bucket) in parts.iter().enumerate().take(cfg.num_layers) {
            assert!(!bucket.is_empty(), "layer {i} should have tensors");
        }
    }

    // -- Shape validation ----------------------------------------------------

    #[test]
    fn test_validate_shapes_correct_weights_pass() {
        let cfg = create_mock_model_config("tiny");
        let w = create_mock_weights(&cfg, 42);
        let errs = validate_weight_shapes(&w);
        assert!(errs.is_empty(), "unexpected errors: {errs:?}");
    }

    #[test]
    fn test_validate_shapes_wrong_embedding_caught() {
        let cfg = create_mock_model_config("tiny");
        let mut w = create_mock_weights(&cfg, 42);
        w.token_embedding = vec![0.0; 7]; // wrong size
        let errs = validate_weight_shapes(&w);
        assert!(
            errs.iter().any(|e| e.contains("token_embedding")),
            "should flag embedding: {errs:?}"
        );
    }

    #[test]
    fn test_validate_shapes_wrong_layer_proj_caught() {
        let cfg = create_mock_model_config("tiny");
        let mut w = create_mock_weights(&cfg, 42);
        w.layers[0].q_proj = vec![0.0; 3]; // wrong size
        let errs = validate_weight_shapes(&w);
        assert!(errs.iter().any(|e| e.contains("q_proj")), "should flag q_proj: {errs:?}");
    }

    // -- OpenCL memory estimation --------------------------------------------

    #[test]
    fn test_opencl_memory_has_weight_and_activation() {
        let cfg = create_mock_model_config("tiny");
        let (w, a) = estimate_opencl_memory(&cfg, WeightFormat::I2S);
        assert!(w > 0);
        assert!(a > 0);
    }

    #[test]
    fn test_opencl_memory_weight_matches_compute() {
        let cfg = create_mock_model_config("small");
        let (w, _) = estimate_opencl_memory(&cfg, WeightFormat::I2S);
        let direct = compute_weight_memory(&cfg, WeightFormat::I2S);
        assert_eq!(w, direct);
    }

    // -- Edge cases ----------------------------------------------------------

    #[test]
    fn test_single_layer_model() {
        let cfg = ModelConfig {
            vocab_size: 32,
            hidden_dim: 16,
            num_layers: 1,
            num_heads: 4,
            head_dim: 4,
            intermediate_dim: 32,
            max_position: 64,
            rope_base: 10000.0,
        };
        assert!(validate_model_config(&cfg).is_ok());
        let w = create_mock_weights(&cfg, 1);
        assert_eq!(w.layers.len(), 1);
        assert!(validate_weight_shapes(&w).is_empty());
    }

    #[test]
    fn test_single_head_model() {
        let cfg = ModelConfig {
            vocab_size: 32,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 1,
            head_dim: 16,
            intermediate_dim: 32,
            max_position: 64,
            rope_base: 10000.0,
        };
        assert!(validate_model_config(&cfg).is_ok());
    }

    #[test]
    fn test_vocab_size_one() {
        let cfg = ModelConfig {
            vocab_size: 1,
            hidden_dim: 8,
            num_layers: 1,
            num_heads: 2,
            head_dim: 4,
            intermediate_dim: 16,
            max_position: 32,
            rope_base: 10000.0,
        };
        assert!(validate_model_config(&cfg).is_ok());
        let w = create_mock_weights(&cfg, 0);
        assert!(validate_weight_shapes(&w).is_empty());
    }

    #[test]
    fn test_unknown_variant_defaults_to_tiny() {
        let cfg = create_mock_model_config("nonexistent");
        let tiny = create_mock_model_config("tiny");
        assert_eq!(cfg, tiny);
    }

    // -- Property: all layers have same shapes within a model ----------------

    #[test]
    fn test_all_layers_same_shapes() {
        let cfg = create_mock_model_config("medium");
        let w = create_mock_weights(&cfg, 7);
        let first = &w.layers[0];
        for (i, layer) in w.layers.iter().enumerate().skip(1) {
            assert_eq!(layer.q_proj.len(), first.q_proj.len(), "layer {i} q_proj size mismatch");
            assert_eq!(
                layer.gate_proj.len(),
                first.gate_proj.len(),
                "layer {i} gate_proj size mismatch"
            );
            assert_eq!(
                layer.attention_norm.len(),
                first.attention_norm.len(),
                "layer {i} attention_norm size mismatch"
            );
        }
    }

    // -- ModelLoadError display -----------------------------------------------

    #[test]
    fn test_error_display_missing_tensor() {
        let e = ModelLoadError::MissingTensor("q_proj".into());
        assert!(e.to_string().contains("q_proj"));
    }

    #[test]
    fn test_error_display_io_error() {
        let e = ModelLoadError::IOError("disk full".into());
        assert!(e.to_string().contains("disk full"));
    }

    // -- QK256 memory estimation ---------------------------------------------

    #[test]
    fn test_qk256_memory_between_f16_and_i2s() {
        let cfg = create_mock_model_config("small");
        let qk = compute_weight_memory(&cfg, WeightFormat::QK256);
        let i2s = compute_weight_memory(&cfg, WeightFormat::I2S);
        let f16 = compute_weight_memory(&cfg, WeightFormat::Float16);
        assert!(qk > i2s, "QK256 should be larger than I2S");
        assert!(qk < f16, "QK256 should be smaller than F16");
    }
}

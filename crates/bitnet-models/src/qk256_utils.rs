//! QK256 quantization utilities for orientation detection and validation
//!
//! Provides shared logic for detecting QK256 tensor orientations across different
//! loader implementations (GGUF and simple loaders).

use bitnet_common::BitNetConfig;

/// Expected [rows, cols] = [output_dim, input_dim] for kernel layout
///
/// Returns the expected shape for QK256 weight matrices based on tensor name
/// and model configuration. This is used to detect if a tensor needs transposition.
///
/// # Arguments
/// * `name` - Tensor name (e.g., "q_proj", "attn_k", "ffn_gate")
/// * `cfg` - Model configuration containing architecture dimensions
///
/// # Returns
/// `Some((rows, cols))` if the tensor name matches a known pattern, `None` otherwise
pub fn expected_qk256_shape(name: &str, cfg: &BitNetConfig) -> Option<(usize, usize)> {
    let hidden = cfg.model.hidden_size;
    let head_dim = hidden / cfg.model.num_heads;
    let kv_dim = head_dim * cfg.model.num_key_value_heads;
    let inter = cfg.model.intermediate_size;

    // Attention Q projection: [hidden, hidden]
    if name.contains("attn_q") || name.contains("q_proj") {
        Some((hidden, hidden))
    }
    // Attention K/V projections: [kv_dim, hidden]
    else if name.contains("attn_k")
        || name.contains("k_proj")
        || name.contains("attn_v")
        || name.contains("v_proj")
    {
        Some((kv_dim, hidden))
    }
    // Attention output projection: [hidden, hidden]
    else if name.contains("attn_output") || name.contains("o_proj") {
        Some((hidden, hidden))
    }
    // FFN gate/up projections: [intermediate, hidden]
    else if name.contains("ffn_gate")
        || name.contains("gate_proj")
        || name.contains("ffn_up")
        || name.contains("up_proj")
    {
        Some((inter, hidden))
    }
    // FFN down projection: [hidden, intermediate]
    else if name.contains("ffn_down") || name.contains("down_proj") {
        Some((hidden, inter))
    } else {
        None
    }
}

/// Pick orientation by payload size if config-based match is inconclusive
///
/// When we can't determine orientation from tensor name and config alone,
/// this function uses the actual tensor data size to choose between the
/// as-is and transposed orientations.
///
/// # Arguments
/// * `shape_as_is` - Shape if tensor is used as-is (rows, cols)
/// * `shape_transposed` - Shape if tensor is transposed (cols, rows)
/// * `available_bytes` - Actual size of tensor data in bytes
///
/// # Returns
/// The shape that best matches the available bytes
pub fn detect_qk256_orientation_by_bytes(
    shape_as_is: (usize, usize),
    shape_transposed: (usize, usize),
    available_bytes: usize,
) -> (usize, usize) {
    // QK256 format: each row has (cols+255)/256 blocks, each block is 64 bytes
    let blocks_as_is = shape_as_is.1.div_ceil(256);
    let expected_as_is = shape_as_is.0 * blocks_as_is * 64;

    let blocks_transposed = shape_transposed.1.div_ceil(256);
    let expected_transposed = shape_transposed.0 * blocks_transposed * 64;

    // Choose the orientation that better matches the available bytes
    if available_bytes.abs_diff(expected_transposed) < available_bytes.abs_diff(expected_as_is) {
        shape_transposed
    } else {
        shape_as_is
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitnet_common::{
        BitNetConfig, InferenceConfig, ModelConfig, PerformanceConfig, QuantizationConfig,
    };

    fn test_config() -> BitNetConfig {
        let mut cfg = BitNetConfig::default();
        cfg.model.hidden_size = 2560;
        cfg.model.num_heads = 20;
        cfg.model.num_key_value_heads = 5;
        cfg.model.intermediate_size = 6912;
        cfg.model.num_layers = 30;
        cfg.model.vocab_size = 128256;
        cfg.model.max_position_embeddings = 4096;
        cfg
    }

    #[test]
    fn test_expected_qk256_shape() {
        let cfg = test_config();

        // Q projection: [2560, 2560]
        assert_eq!(
            expected_qk256_shape("model.layers.0.self_attn.q_proj.weight", &cfg),
            Some((2560, 2560))
        );

        // K/V projections: [640, 2560] (kv_dim = 128 * 5 = 640)
        assert_eq!(
            expected_qk256_shape("model.layers.0.self_attn.k_proj.weight", &cfg),
            Some((640, 2560))
        );
        assert_eq!(
            expected_qk256_shape("model.layers.0.self_attn.v_proj.weight", &cfg),
            Some((640, 2560))
        );

        // O projection: [2560, 2560]
        assert_eq!(
            expected_qk256_shape("model.layers.0.self_attn.o_proj.weight", &cfg),
            Some((2560, 2560))
        );

        // FFN gate/up: [6912, 2560]
        assert_eq!(
            expected_qk256_shape("model.layers.0.mlp.gate_proj.weight", &cfg),
            Some((6912, 2560))
        );
        assert_eq!(
            expected_qk256_shape("model.layers.0.mlp.up_proj.weight", &cfg),
            Some((6912, 2560))
        );

        // FFN down: [2560, 6912]
        assert_eq!(
            expected_qk256_shape("model.layers.0.mlp.down_proj.weight", &cfg),
            Some((2560, 6912))
        );

        // Unknown tensor
        assert_eq!(expected_qk256_shape("model.embed_tokens.weight", &cfg), None);
    }

    #[test]
    fn test_detect_qk256_orientation_by_bytes() {
        // Example: Q projection [2560, 2560] vs transposed [2560, 2560]
        // As-is: rows=2560, cols=2560 -> blocks=(2560+255)/256=10 -> 2560*10*64=1,638,400
        let shape_as_is = (2560, 2560);
        let shape_transposed = (2560, 2560);
        let available = 1_638_400;

        let result = detect_qk256_orientation_by_bytes(shape_as_is, shape_transposed, available);
        assert_eq!(result, shape_as_is); // Both match, picks as-is

        // Example: K projection [640, 2560] vs transposed [2560, 640]
        // As-is: rows=640, cols=2560 -> blocks=10 -> 640*10*64=409,600
        // Transposed: rows=2560, cols=640 -> blocks=3 -> 2560*3*64=491,520
        let shape_as_is = (640, 2560);
        let shape_transposed = (2560, 640);

        // If available matches as-is
        let available = 409_600;
        let result = detect_qk256_orientation_by_bytes(shape_as_is, shape_transposed, available);
        assert_eq!(result, shape_as_is);

        // If available matches transposed
        let available = 491_520;
        let result = detect_qk256_orientation_by_bytes(shape_as_is, shape_transposed, available);
        assert_eq!(result, shape_transposed);
    }
}

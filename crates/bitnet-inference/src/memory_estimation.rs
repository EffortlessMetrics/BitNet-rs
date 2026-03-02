//! # Memory Estimation for SLM Models
//!
//! Utilities for estimating KV cache, model parameter, and activation memory
//! requirements. Includes precomputed profiles for known models (Phi-4,
//! LLaMA-3.2-1B, Qwen2.5-7B).

/// Combined memory estimation for loading and running a model.
#[derive(Debug, Clone)]
pub struct MemoryEstimation {
    /// Total weight memory in bytes.
    pub model_params_bytes: u64,
    /// KV cache memory for max context in bytes.
    pub kv_cache_bytes: u64,
    /// Peak activation memory during forward pass in bytes.
    pub activation_bytes: u64,
    /// Sum of all components.
    pub total_bytes: u64,
    /// Recommended GPU VRAM in GiB (total + 20% headroom).
    pub recommended_gpu_vram_gb: f32,
    /// Recommended system RAM in GiB (total + 50% headroom).
    pub recommended_system_ram_gb: f32,
}

/// KV cache memory breakdown.
#[derive(Debug, Clone)]
pub struct KvCacheEstimation {
    /// Memory for a single layer (K+V) in bytes.
    pub per_layer_bytes: u64,
    /// Total KV cache memory across all layers.
    pub total_bytes: u64,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    /// Bytes per element (2 = f16, 4 = f32, 1 = int8).
    pub dtype_bytes: usize,
}

/// Precomputed memory profiles for a known model at various context lengths.
#[derive(Debug, Clone)]
pub struct ModelMemoryProfile {
    pub model_name: String,
    pub architecture: String,
    /// (context_length, estimation) pairs.
    pub known_profiles: Vec<(usize, MemoryEstimation)>,
}

// ---------------------------------------------------------------------------
// Core estimation functions
// ---------------------------------------------------------------------------

/// Estimate KV cache memory.
///
/// Formula: `2 * num_layers * num_kv_heads * head_dim * max_seq_len * dtype_bytes`
/// (factor of 2 accounts for both Key and Value tensors).
pub fn estimate_kv_cache(
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    dtype_bytes: usize,
) -> KvCacheEstimation {
    let per_layer_bytes = 2u64
        * (num_kv_heads as u64)
        * (head_dim as u64)
        * (max_seq_len as u64)
        * (dtype_bytes as u64);
    let total_bytes = per_layer_bytes * (num_layers as u64);
    KvCacheEstimation {
        per_layer_bytes,
        total_bytes,
        num_layers,
        num_kv_heads,
        head_dim,
        max_seq_len,
        dtype_bytes,
    }
}

/// Estimate model weight memory: `num_params * dtype_bytes`.
pub fn estimate_model_memory(num_params: u64, dtype_bytes: usize) -> u64 {
    num_params * (dtype_bytes as u64)
}

/// Rough activation memory estimate.
///
/// `batch_size * seq_len * hidden_size * 4 * dtype_bytes`
/// (factor of 4 covers QKV projections + output).
pub fn estimate_activation_memory(
    batch_size: usize,
    seq_len: usize,
    hidden_size: usize,
    num_layers: usize,
    dtype_bytes: usize,
) -> u64 {
    (batch_size as u64)
        * (seq_len as u64)
        * (hidden_size as u64)
        * 4
        * (dtype_bytes as u64)
        * (num_layers as u64)
}

/// Model configuration required for a total memory estimate.
pub struct ModelConfig {
    pub num_params: u64,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub max_seq_len: usize,
    pub dtype_bytes: usize,
    pub batch_size: usize,
}

const GIB: f64 = (1u64 << 30) as f64;

/// Produce a combined [`MemoryEstimation`] from a model configuration.
pub fn estimate_total(cfg: &ModelConfig) -> MemoryEstimation {
    let model_params_bytes = estimate_model_memory(cfg.num_params, cfg.dtype_bytes);
    let kv = estimate_kv_cache(
        cfg.num_layers,
        cfg.num_kv_heads,
        cfg.head_dim,
        cfg.max_seq_len,
        cfg.dtype_bytes,
    );
    let activation_bytes = estimate_activation_memory(
        cfg.batch_size,
        cfg.max_seq_len,
        cfg.hidden_size,
        cfg.num_layers,
        cfg.dtype_bytes,
    );
    let total_bytes = model_params_bytes + kv.total_bytes + activation_bytes;
    let total_gib = total_bytes as f64 / GIB;
    MemoryEstimation {
        model_params_bytes,
        kv_cache_bytes: kv.total_bytes,
        activation_bytes,
        total_bytes,
        recommended_gpu_vram_gb: (total_gib * 1.2) as f32,
        recommended_system_ram_gb: (total_gib * 1.5) as f32,
    }
}

// ---------------------------------------------------------------------------
// Human-readable formatting
// ---------------------------------------------------------------------------

/// Format a byte count into a human-readable string (e.g. `"3.12 GB"`).
pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;
    const TB: u64 = 1024 * GB;

    if bytes >= TB {
        format!("{:.2} TB", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

// ---------------------------------------------------------------------------
// Known model profiles
// ---------------------------------------------------------------------------

fn phi4_config(context: usize) -> ModelConfig {
    ModelConfig {
        num_params: 14_000_000_000,
        num_layers: 40,
        num_kv_heads: 10,
        head_dim: 128,
        hidden_size: 5120,
        max_seq_len: context,
        dtype_bytes: 2, // f16
        batch_size: 1,
    }
}

fn llama32_1b_config(context: usize) -> ModelConfig {
    ModelConfig {
        num_params: 1_200_000_000,
        num_layers: 16,
        num_kv_heads: 8,
        head_dim: 64,
        hidden_size: 2048,
        max_seq_len: context,
        dtype_bytes: 2,
        batch_size: 1,
    }
}

fn qwen25_7b_config(context: usize) -> ModelConfig {
    ModelConfig {
        num_params: 7_600_000_000,
        num_layers: 28,
        num_kv_heads: 4,
        head_dim: 128,
        hidden_size: 3584,
        max_seq_len: context,
        dtype_bytes: 2,
        batch_size: 1,
    }
}

/// Return a precomputed [`ModelMemoryProfile`] for a known model, if available.
///
/// Recognised names (case-insensitive): `phi-4`, `llama-3.2-1b`, `qwen2.5-7b`.
pub fn get_known_profile(model_name: &str) -> Option<ModelMemoryProfile> {
    let name = model_name.to_ascii_lowercase();
    match name.as_str() {
        "phi-4" => {
            let contexts = [4096, 8192, 16384];
            let profiles: Vec<(usize, MemoryEstimation)> =
                contexts.iter().map(|&c| (c, estimate_total(&phi4_config(c)))).collect();
            Some(ModelMemoryProfile {
                model_name: "Phi-4".into(),
                architecture: "Phi".into(),
                known_profiles: profiles,
            })
        }
        "llama-3.2-1b" => {
            let contexts = [2048, 4096, 8192];
            let profiles: Vec<(usize, MemoryEstimation)> =
                contexts.iter().map(|&c| (c, estimate_total(&llama32_1b_config(c)))).collect();
            Some(ModelMemoryProfile {
                model_name: "LLaMA-3.2-1B".into(),
                architecture: "LLaMA".into(),
                known_profiles: profiles,
            })
        }
        "qwen2.5-7b" => {
            let contexts = [4096, 16384, 32768];
            let profiles: Vec<(usize, MemoryEstimation)> =
                contexts.iter().map(|&c| (c, estimate_total(&qwen25_7b_config(c)))).collect();
            Some(ModelMemoryProfile {
                model_name: "Qwen2.5-7B".into(),
                architecture: "Qwen".into(),
                known_profiles: profiles,
            })
        }
        _ => None,
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- KV cache estimation ------------------------------------------------

    #[test]
    fn test_kv_cache_phi4() {
        let est = estimate_kv_cache(40, 10, 128, 16384, 2);
        // 2 * 40 * 10 * 128 * 16384 * 2 = 3,355,443,200
        assert_eq!(est.total_bytes, 3_355_443_200);
        assert_eq!(est.per_layer_bytes, 3_355_443_200 / 40);
        assert_eq!(est.num_layers, 40);
    }

    #[test]
    fn test_kv_cache_llama32_1b() {
        let est = estimate_kv_cache(16, 8, 64, 8192, 2);
        // 2 * 16 * 8 * 64 * 8192 * 2 = 268,435,456
        assert_eq!(est.total_bytes, 268_435_456);
    }

    #[test]
    fn test_kv_cache_qwen25_7b() {
        let est = estimate_kv_cache(28, 4, 128, 32768, 2);
        // 2 * 28 * 4 * 128 * 32768 * 2 = 1,879,048,192
        assert_eq!(est.total_bytes, 1_879_048_192);
    }

    // -- Model memory estimation -------------------------------------------

    #[test]
    fn test_model_memory_1b_f16() {
        assert_eq!(estimate_model_memory(1_000_000_000, 2), 2_000_000_000);
    }

    #[test]
    fn test_model_memory_7b_f16() {
        assert_eq!(estimate_model_memory(7_000_000_000, 2), 14_000_000_000);
    }

    #[test]
    fn test_model_memory_14b_f16() {
        assert_eq!(estimate_model_memory(14_000_000_000, 2), 28_000_000_000);
    }

    #[test]
    fn test_model_memory_1b_f32() {
        assert_eq!(estimate_model_memory(1_000_000_000, 4), 4_000_000_000);
    }

    #[test]
    fn test_model_memory_7b_f32() {
        assert_eq!(estimate_model_memory(7_000_000_000, 4), 28_000_000_000);
    }

    #[test]
    fn test_model_memory_14b_f32() {
        assert_eq!(estimate_model_memory(14_000_000_000, 4), 56_000_000_000);
    }

    // -- Activation estimation ---------------------------------------------

    #[test]
    fn test_activation_memory() {
        let act = estimate_activation_memory(1, 2048, 5120, 40, 2);
        // 1 * 2048 * 5120 * 4 * 2 * 40 = 3,355,443,200
        assert_eq!(act, 3_355_443_200);
    }

    // -- Total estimation --------------------------------------------------

    #[test]
    fn test_total_phi4() {
        let est = estimate_total(&phi4_config(16384));
        // params ≈28 GB, kv ≈2.5 GB → total ≈ ~30 GB range
        let total_gb = est.total_bytes as f64 / GIB;
        assert!(total_gb > 25.0 && total_gb < 80.0, "Phi-4 total {total_gb:.1} GB");
        assert!(est.recommended_gpu_vram_gb > est.total_bytes as f32 / GIB as f32);
    }

    #[test]
    fn test_total_llama32_1b() {
        let est = estimate_total(&llama32_1b_config(8192));
        let total_gb = est.total_bytes as f64 / GIB;
        // Small model – should be well under 16 GB
        assert!(total_gb < 16.0, "LLaMA-3.2-1B total {total_gb:.1} GB");
    }

    #[test]
    fn test_total_components_sum() {
        let est = estimate_total(&phi4_config(4096));
        assert_eq!(
            est.total_bytes,
            est.model_params_bytes + est.kv_cache_bytes + est.activation_bytes
        );
    }

    // -- format_bytes ------------------------------------------------------

    #[test]
    fn test_format_bytes_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1023), "1023 B");
    }

    #[test]
    fn test_format_bytes_kb() {
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1536), "1.50 KB");
    }

    #[test]
    fn test_format_bytes_mb() {
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(256 * 1024 * 1024), "256.00 MB");
    }

    #[test]
    fn test_format_bytes_gb() {
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
        let three_gb = 3 * 1024 * 1024 * 1024u64 + 128 * 1024 * 1024;
        assert!(format_bytes(three_gb).contains("GB"));
    }

    #[test]
    fn test_format_bytes_tb() {
        assert_eq!(format_bytes(1024u64 * 1024 * 1024 * 1024), "1.00 TB");
    }

    // -- Known profiles ----------------------------------------------------

    #[test]
    fn test_known_profile_hit() {
        assert!(get_known_profile("phi-4").is_some());
        assert!(get_known_profile("llama-3.2-1b").is_some());
        assert!(get_known_profile("qwen2.5-7b").is_some());
    }

    #[test]
    fn test_known_profile_miss() {
        assert!(get_known_profile("nonexistent-model").is_none());
    }

    #[test]
    fn test_known_profile_case_insensitive() {
        assert!(get_known_profile("Phi-4").is_some());
        assert!(get_known_profile("PHI-4").is_some());
    }

    // -- Edge cases --------------------------------------------------------

    #[test]
    fn test_kv_cache_zero_layers() {
        let est = estimate_kv_cache(0, 10, 128, 16384, 2);
        assert_eq!(est.total_bytes, 0);
    }

    #[test]
    fn test_kv_cache_zero_seq_len() {
        let est = estimate_kv_cache(40, 10, 128, 0, 2);
        assert_eq!(est.total_bytes, 0);
    }

    #[test]
    fn test_kv_cache_one_head() {
        let est = estimate_kv_cache(1, 1, 128, 1024, 2);
        // 2 * 1 * 1 * 128 * 1024 * 2 = 524_288
        assert_eq!(est.total_bytes, 524_288);
    }

    // -- dtype variations --------------------------------------------------

    #[test]
    fn test_kv_cache_int8() {
        let est = estimate_kv_cache(40, 10, 128, 16384, 1);
        // Half of f16 result (3,355,443,200 / 2)
        assert_eq!(est.total_bytes, 3_355_443_200 / 2);
    }

    #[test]
    fn test_kv_cache_f32() {
        let est = estimate_kv_cache(40, 10, 128, 16384, 4);
        // Double the f16 result (3,355,443,200 * 2)
        assert_eq!(est.total_bytes, 3_355_443_200 * 2);
    }

    // -- Scaling properties ------------------------------------------------

    #[test]
    fn test_doubling_context_doubles_kv_cache() {
        let a = estimate_kv_cache(40, 10, 128, 8192, 2);
        let b = estimate_kv_cache(40, 10, 128, 16384, 2);
        assert_eq!(b.total_bytes, a.total_bytes * 2);
    }
}

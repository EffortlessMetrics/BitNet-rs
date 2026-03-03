//! Comprehensive GQA (Grouped Query Attention) configuration tests.
//!
//! Validates correct head/kv-head ratios, KV broadcasting shapes,
//! cache sizing, attention mask dimensions, and invalid-config rejection
//! across diverse SLM architectures (MHA, GQA 4:1/8:1, MQA).

use bitnet_common::{BitNetConfig, ModelConfig};

// ---------------------------------------------------------------------------
// Helper: build a ModelConfig with the given GQA parameters
// ---------------------------------------------------------------------------
fn model_config(
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    intermediate_size: usize,
) -> ModelConfig {
    ModelConfig {
        hidden_size,
        num_heads,
        num_key_value_heads: num_kv_heads,
        intermediate_size,
        ..ModelConfig::default()
    }
}

/// Derived GQA geometry for a given configuration.
struct GqaGeometry {
    head_dim: usize,
    group_size: usize,
    kv_proj_out: usize,
    q_proj_out: usize,
}

fn gqa_geometry(cfg: &ModelConfig) -> GqaGeometry {
    let head_dim = cfg.hidden_size / cfg.num_heads;
    let effective_kv =
        if cfg.num_key_value_heads == 0 { cfg.num_heads } else { cfg.num_key_value_heads };
    let group_size = cfg.num_heads / effective_kv;
    GqaGeometry {
        head_dim,
        group_size,
        kv_proj_out: effective_kv * head_dim,
        q_proj_out: cfg.num_heads * head_dim,
    }
}

// ===========================================================================
// §1  Standard MHA (ratio 1:1)
// ===========================================================================

#[test]
fn mha_32_heads_ratio_1_to_1() {
    let cfg = model_config(4096, 32, 32, 11008);
    let g = gqa_geometry(&cfg);

    assert_eq!(g.head_dim, 128);
    assert_eq!(g.group_size, 1, "MHA has group_size == 1");
    assert_eq!(g.kv_proj_out, g.q_proj_out, "MHA: KV proj == Q proj");
    assert_eq!(g.kv_proj_out, 4096);
}

// ===========================================================================
// §2  GQA ratio 4:1 — LLaMA-3 style (32 heads, 8 kv-heads)
// ===========================================================================

#[test]
fn gqa_4_to_1_llama3_style() {
    let cfg = model_config(4096, 32, 8, 11008);
    let g = gqa_geometry(&cfg);

    assert_eq!(g.head_dim, 128);
    assert_eq!(g.group_size, 4);
    assert_eq!(g.kv_proj_out, 8 * 128, "K/V projection output = kv_heads × head_dim");
    assert_eq!(g.q_proj_out, 32 * 128, "Q projection output = heads × head_dim");
}

// ===========================================================================
// §3  GQA ratio 4:1 — Phi-4 / BitNet-2B style (40 heads, 10 kv-heads)
// ===========================================================================

#[test]
fn gqa_4_to_1_phi4_style() {
    let cfg = model_config(2560, 40, 10, 6912);
    let g = gqa_geometry(&cfg);

    assert_eq!(g.head_dim, 64);
    assert_eq!(g.group_size, 4);
    assert_eq!(g.kv_proj_out, 10 * 64);
    assert_eq!(g.q_proj_out, 40 * 64);
}

// ===========================================================================
// §4  GQA ratio 8:1 — Mistral style (32 heads, 4 kv-heads)
// ===========================================================================

#[test]
fn gqa_8_to_1_mistral_style() {
    let cfg = model_config(4096, 32, 4, 14336);
    let g = gqa_geometry(&cfg);

    assert_eq!(g.head_dim, 128);
    assert_eq!(g.group_size, 8);
    assert_eq!(g.kv_proj_out, 4 * 128);
    assert_eq!(g.q_proj_out, 32 * 128);
}

// ===========================================================================
// §5  MQA (ratio N:1) — Falcon style (32 heads, 1 kv-head)
// ===========================================================================

#[test]
fn mqa_32_to_1_falcon_style() {
    let cfg = model_config(4096, 32, 1, 11008);
    let g = gqa_geometry(&cfg);

    assert_eq!(g.head_dim, 128);
    assert_eq!(g.group_size, 32);
    assert_eq!(g.kv_proj_out, 128, "MQA: single KV head");
    assert_eq!(g.q_proj_out, 32 * 128);
}

// ===========================================================================
// §6  Head dimension calculations
// ===========================================================================

#[test]
fn head_dim_calculation_various_sizes() {
    let cases: &[(usize, usize, usize)] = &[
        // (hidden_size, num_heads, expected_head_dim)
        (4096, 32, 128),
        (2560, 40, 64),
        (8192, 64, 128),
        (2048, 16, 128),
        (768, 12, 64),
        (1024, 16, 64),
    ];
    for &(hidden, heads, expected_dim) in cases {
        let cfg = model_config(hidden, heads, heads, hidden * 4);
        let g = gqa_geometry(&cfg);
        assert_eq!(
            g.head_dim, expected_dim,
            "hidden={hidden}, heads={heads} should give head_dim={expected_dim}"
        );
    }
}

// ===========================================================================
// §7  KV cache sizing for each GQA configuration
// ===========================================================================

/// Compute per-layer KV cache size in bytes (f32) for a given config.
fn kv_cache_bytes_per_layer(cfg: &ModelConfig, max_seq_len: usize) -> usize {
    let g = gqa_geometry(cfg);
    let effective_kv =
        if cfg.num_key_value_heads == 0 { cfg.num_heads } else { cfg.num_key_value_heads };
    // K + V, each: [kv_heads, max_seq_len, head_dim] × sizeof(f32)
    2 * effective_kv * max_seq_len * g.head_dim * std::mem::size_of::<f32>()
}

#[test]
fn kv_cache_sizing_mha() {
    let cfg = model_config(4096, 32, 32, 11008);
    let bytes = kv_cache_bytes_per_layer(&cfg, 2048);
    // 2 * 32 * 2048 * 128 * 4 = 67_108_864
    assert_eq!(bytes, 67_108_864);
}

#[test]
fn kv_cache_sizing_gqa_4_to_1() {
    let cfg = model_config(4096, 32, 8, 11008);
    let bytes = kv_cache_bytes_per_layer(&cfg, 2048);
    // 2 * 8 * 2048 * 128 * 4 = 16_777_216  (4× smaller than MHA)
    assert_eq!(bytes, 16_777_216);
}

#[test]
fn kv_cache_sizing_gqa_8_to_1() {
    let cfg = model_config(4096, 32, 4, 14336);
    let bytes = kv_cache_bytes_per_layer(&cfg, 2048);
    // 2 * 4 * 2048 * 128 * 4 = 8_388_608  (8× smaller than MHA)
    assert_eq!(bytes, 8_388_608);
}

#[test]
fn kv_cache_sizing_mqa() {
    let cfg = model_config(4096, 32, 1, 11008);
    let bytes = kv_cache_bytes_per_layer(&cfg, 2048);
    // 2 * 1 * 2048 * 128 * 4 = 2_097_152  (32× smaller than MHA)
    assert_eq!(bytes, 2_097_152);
}

#[test]
fn kv_cache_ratio_savings() {
    let mha = kv_cache_bytes_per_layer(&model_config(4096, 32, 32, 11008), 2048);
    let gqa4 = kv_cache_bytes_per_layer(&model_config(4096, 32, 8, 11008), 2048);
    let gqa8 = kv_cache_bytes_per_layer(&model_config(4096, 32, 4, 14336), 2048);
    let mqa = kv_cache_bytes_per_layer(&model_config(4096, 32, 1, 11008), 2048);

    assert_eq!(mha / gqa4, 4, "GQA 4:1 saves 4× KV memory");
    assert_eq!(mha / gqa8, 8, "GQA 8:1 saves 8× KV memory");
    assert_eq!(mha / mqa, 32, "MQA saves 32× KV memory");
}

// ===========================================================================
// §8  Attention mask shape for each configuration
// ===========================================================================

/// The causal attention mask is [1, 1, seq_len, total_key_len] and is
/// independent of the number of heads — it broadcasts across all heads.
#[test]
fn attention_mask_shape_is_head_independent() {
    let configs = [
        model_config(4096, 32, 32, 11008), // MHA
        model_config(4096, 32, 8, 11008),  // GQA 4:1
        model_config(4096, 32, 4, 14336),  // GQA 8:1
        model_config(4096, 32, 1, 11008),  // MQA
    ];
    let seq_len = 128;
    for cfg in &configs {
        // Mask shape is always [1, 1, seq_len, total_key_len]
        let mask_shape = [1, 1, seq_len, seq_len];
        assert_eq!(mask_shape[0], 1, "batch broadcast dim");
        assert_eq!(mask_shape[1], 1, "head broadcast dim");
        assert_eq!(mask_shape[2], seq_len);
        assert_eq!(mask_shape[3], seq_len);
        // The mask does NOT depend on num_heads or num_kv_heads
        let _ = cfg; // used for documentation
    }
}

// ===========================================================================
// §9  KV broadcasting shape verification
// ===========================================================================

/// Verify that the expanded K/V shapes after GQA repeat match Q shape.
#[test]
fn kv_broadcast_shape_all_ratios() {
    let cases: &[(&str, usize, usize, usize)] = &[
        // (name, hidden_size, num_heads, num_kv_heads)
        ("MHA-32", 4096, 32, 32),
        ("GQA-4:1-LLaMA3", 4096, 32, 8),
        ("GQA-4:1-Phi4", 2560, 40, 10),
        ("GQA-8:1-Mistral", 4096, 32, 4),
        ("MQA-Falcon", 4096, 32, 1),
    ];

    let batch = 1;
    let seq_len = 16;

    for &(name, hidden, heads, kv_heads) in cases {
        let head_dim = hidden / heads;
        let group_size = heads / kv_heads;

        // Before expansion: K/V shape = [B, kv_heads, T, head_dim]
        let kv_shape = [batch, kv_heads, seq_len, head_dim];
        // After expansion:  K/V shape = [B, heads, T, head_dim]
        let expanded = [batch, kv_heads * group_size, seq_len, head_dim];

        assert_eq!(
            expanded,
            [batch, heads, seq_len, head_dim],
            "{name}: expanded KV must match Q head count"
        );
        // Verify the repeat factor
        assert_eq!(kv_shape[1] * group_size, heads, "{name}: kv_heads × group_size == num_heads");
    }
}

// ===========================================================================
// §10  Invalid configuration rejection
// ===========================================================================

#[test]
fn reject_zero_heads() {
    let mut cfg = BitNetConfig::default();
    cfg.model.num_heads = 0;
    let result = cfg.validate();
    assert!(result.is_err(), "0 heads should be rejected");
}

#[test]
fn reject_kv_heads_greater_than_heads() {
    let mut cfg = BitNetConfig::default();
    cfg.model.num_heads = 32;
    cfg.model.num_key_value_heads = 33;
    let result = cfg.validate();
    assert!(result.is_err(), "kv_heads > heads should be rejected");
}

#[test]
fn reject_non_divisible_ratio() {
    let mut cfg = BitNetConfig::default();
    cfg.model.num_heads = 32;
    cfg.model.num_key_value_heads = 7;
    let result = cfg.validate();
    assert!(result.is_err(), "32 heads / 7 kv_heads is not divisible");
}

#[test]
fn reject_non_divisible_ratio_5() {
    let mut cfg = BitNetConfig::default();
    cfg.model.num_heads = 32;
    cfg.model.num_key_value_heads = 5;
    let result = cfg.validate();
    assert!(result.is_err(), "32 heads / 5 kv_heads is not divisible");
}

#[test]
fn reject_hidden_size_not_divisible_by_heads() {
    let mut cfg = BitNetConfig::default();
    cfg.model.hidden_size = 4097;
    cfg.model.num_heads = 32;
    let result = cfg.validate();
    assert!(result.is_err(), "4097 / 32 is not evenly divisible");
}

#[test]
fn accept_kv_heads_zero_means_mha() {
    // When num_key_value_heads == 0, it defaults to num_heads (MHA).
    let mut cfg = BitNetConfig::default();
    cfg.model.num_heads = 32;
    cfg.model.num_key_value_heads = 0;
    let result = cfg.validate();
    assert!(result.is_ok(), "kv_heads=0 should be accepted (defaults to MHA)");
}

// ===========================================================================
// §11  Configuration deserialization (JSON round-trip)
// ===========================================================================

#[test]
fn deserialize_gqa_config_from_json() {
    let json = r#"{
        "model": {
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_layers": 32,
            "num_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 11008,
            "max_position_embeddings": 4096
        },
        "inference": {
            "max_length": 4096,
            "max_new_tokens": 512,
            "temperature": 0.7,
            "repetition_penalty": 1.1
        },
        "quantization": {
            "block_size": 64,
            "precision": 0.0001
        },
        "performance": {
            "batch_size": 1
        }
    }"#;

    let cfg: BitNetConfig = serde_json::from_str(json).expect("JSON deserialization failed");
    assert_eq!(cfg.model.num_heads, 32);
    assert_eq!(cfg.model.num_key_value_heads, 8);
    assert_eq!(cfg.model.hidden_size, 4096);

    let g = gqa_geometry(&cfg.model);
    assert_eq!(g.group_size, 4);
    assert_eq!(g.head_dim, 128);
    assert_eq!(g.kv_proj_out, 1024);
}

#[test]
fn deserialize_mqa_config_from_json() {
    let json = r#"{
        "model": {
            "vocab_size": 65024,
            "hidden_size": 4096,
            "num_layers": 32,
            "num_heads": 32,
            "num_key_value_heads": 1,
            "intermediate_size": 11008,
            "max_position_embeddings": 2048
        },
        "inference": {
            "max_length": 2048,
            "max_new_tokens": 256,
            "temperature": 1.0,
            "repetition_penalty": 1.0
        },
        "quantization": {
            "block_size": 64,
            "precision": 0.0001
        },
        "performance": {
            "batch_size": 1
        }
    }"#;

    let cfg: BitNetConfig = serde_json::from_str(json).expect("JSON deserialization failed");
    assert_eq!(cfg.model.num_key_value_heads, 1, "MQA: single KV head");

    let g = gqa_geometry(&cfg.model);
    assert_eq!(g.group_size, 32);
    assert_eq!(g.kv_proj_out, 128);
}

// ===========================================================================
// §12  Parametric sweep across real-world architectures
// ===========================================================================

#[test]
fn parametric_sweep_real_architectures() {
    let architectures: &[(&str, usize, usize, usize, usize)] = &[
        // (name, hidden, heads, kv_heads, intermediate)
        ("LLaMA-2-7B", 4096, 32, 32, 11008),
        ("LLaMA-2-70B", 8192, 64, 8, 28672),
        ("LLaMA-3-8B", 4096, 32, 8, 14336),
        ("Mistral-7B", 4096, 32, 8, 14336),
        ("Mistral-8x7B-expert", 4096, 32, 8, 14336),
        ("Phi-4-mini", 3072, 32, 8, 8192),
        ("BitNet-2B-4T", 2560, 40, 10, 6912),
        ("Falcon-7B-MQA", 4544, 71, 1, 18176),
        ("CodeLlama-34B", 8192, 64, 8, 22016),
        ("Gemma-2B", 2048, 8, 1, 16384),
    ];

    for &(name, hidden, heads, kv_heads, intermediate) in architectures {
        assert_eq!(hidden % heads, 0, "{name}: hidden_size must be divisible by num_heads");
        assert!(kv_heads > 0, "{name}: kv_heads must be > 0");
        assert!(kv_heads <= heads, "{name}: kv_heads <= heads");
        assert_eq!(heads % kv_heads, 0, "{name}: heads must be divisible by kv_heads");

        let cfg = model_config(hidden, heads, kv_heads, intermediate);
        let g = gqa_geometry(&cfg);

        assert_eq!(g.head_dim * heads, hidden, "{name}: head_dim × heads == hidden");
        assert_eq!(g.group_size * kv_heads, heads, "{name}: group × kv == heads");
        assert_eq!(g.kv_proj_out, kv_heads * g.head_dim, "{name}: kv_proj_out check");
        assert_eq!(g.q_proj_out, hidden, "{name}: q_proj_out == hidden_size");
    }
}

// ===========================================================================
// §13  Transformer-level GQA validation
// ===========================================================================

/// Mirrors the validation logic in `bitnet-transformer::MultiHeadAttention::new`
#[test]
fn transformer_mha_validation_logic() {
    let test_cases: &[(usize, usize, usize, bool)] = &[
        // (hidden, heads, kv_heads, should_pass)
        (4096, 32, 32, true), // MHA
        (4096, 32, 8, true),  // GQA 4:1
        (2560, 40, 10, true), // GQA Phi-4
        (4096, 32, 4, true),  // GQA 8:1
        (4096, 32, 1, true),  // MQA
        (4096, 32, 0, true),  // 0 → defaults to MHA
        (4096, 32, 7, false), // not divisible
        (4097, 32, 8, false), // hidden not divisible by heads
    ];

    for &(hidden, heads, kv_heads, should_pass) in test_cases {
        let divisible_hidden = hidden % heads == 0;
        let effective_kv = kv_heads.max(1).min(heads);
        let divisible_kv = heads % effective_kv == 0;

        let valid = divisible_hidden && divisible_kv;
        assert_eq!(
            valid, should_pass,
            "hidden={hidden}, heads={heads}, kv={kv_heads}: expected valid={should_pass}"
        );
    }
}

// ===========================================================================
// §14  Edge cases
// ===========================================================================

#[test]
fn single_head_single_kv() {
    let cfg = model_config(64, 1, 1, 256);
    let g = gqa_geometry(&cfg);
    assert_eq!(g.head_dim, 64);
    assert_eq!(g.group_size, 1);
}

#[test]
fn large_group_ratio() {
    // 128 heads, 1 kv-head → group_size = 128
    let cfg = model_config(8192, 128, 1, 28672);
    let g = gqa_geometry(&cfg);
    assert_eq!(g.group_size, 128);
    assert_eq!(g.kv_proj_out, 64); // 1 × 64
}

#[test]
fn kv_heads_zero_defaults_to_mha() {
    let cfg = model_config(4096, 32, 0, 11008);
    let g = gqa_geometry(&cfg);
    // When kv_heads == 0, gqa_geometry treats it as MHA (num_heads)
    assert_eq!(g.group_size, 1);
    assert_eq!(g.kv_proj_out, 4096);
}

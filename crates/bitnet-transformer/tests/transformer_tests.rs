//! Comprehensive tests for `bitnet-transformer` public API.
//!
//! Covers:
//!   - `BitNetConfig` / `ModelConfig` construction and field values
//!   - `RmsNorm` computation: zero input, unit input, known values, shape invariants
//!   - `TransformerModel` construction, embed, and forward_full under small-dim configs
//!   - `KVCache` / `LayerKVCache` construction and edge cases
//!   - Error cases: non-divisible hidden/heads dimensions
//!   - Property tests: RmsNorm magnitude relationship, valid config acceptance
#![cfg(feature = "cpu")]

use bitnet_common::config::{BitNetConfig, ModelConfig};
use bitnet_transformer::{KVCache, LayerKVCache, TransformerModel};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{RmsNorm, VarBuilder};
use proptest::prelude::*;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Minimal valid config for fast model construction.
fn tiny_config(hidden: usize, vocab: usize, heads: usize) -> BitNetConfig {
    BitNetConfig {
        model: ModelConfig {
            hidden_size: hidden,
            vocab_size: vocab,
            num_heads: heads,
            num_key_value_heads: heads,
            num_layers: 1,
            intermediate_size: hidden * 2,
            max_position_embeddings: 16,
            rms_norm_eps: Some(1e-5),
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Build a model with all-zero weights.
fn make_model(hidden: usize, vocab: usize, heads: usize) -> anyhow::Result<TransformerModel> {
    let device = Device::Cpu;
    let cfg = tiny_config(hidden, vocab, heads);
    let vb = VarBuilder::zeros(DType::F32, &device);
    Ok(TransformerModel::new(cfg, vb)?)
}

// ── Config construction tests ─────────────────────────────────────────────────

#[test]
fn bitnet_config_default_has_nonzero_fields() {
    let cfg = BitNetConfig::default();
    assert!(cfg.model.vocab_size > 0, "default vocab_size must be nonzero");
    assert!(cfg.model.hidden_size > 0, "default hidden_size must be nonzero");
    assert!(cfg.model.num_heads > 0, "default num_heads must be nonzero");
    assert!(cfg.model.num_layers > 0, "default num_layers must be nonzero");
    assert!(cfg.model.intermediate_size > 0, "default intermediate_size must be nonzero");
    assert!(
        cfg.model.max_position_embeddings > 0,
        "default max_position_embeddings must be nonzero"
    );
}

#[test]
fn model_config_default_known_values() {
    let cfg = ModelConfig::default();
    assert_eq!(cfg.vocab_size, 32000);
    assert_eq!(cfg.hidden_size, 4096);
    assert_eq!(cfg.num_heads, 32);
    assert_eq!(cfg.num_layers, 32);
    assert_eq!(cfg.intermediate_size, 11008);
    assert_eq!(cfg.max_position_embeddings, 2048);
    // num_key_value_heads defaults to 0 (means "use num_heads")
    assert_eq!(cfg.num_key_value_heads, 0);
}

#[test]
fn model_config_explicit_construction_stores_values() {
    let cfg = ModelConfig {
        hidden_size: 256,
        vocab_size: 512,
        num_heads: 4,
        num_key_value_heads: 4,
        num_layers: 2,
        intermediate_size: 1024,
        max_position_embeddings: 64,
        rms_norm_eps: Some(1e-6),
        rope_theta: Some(10000.0),
        ..Default::default()
    };
    assert_eq!(cfg.hidden_size, 256);
    assert_eq!(cfg.vocab_size, 512);
    assert_eq!(cfg.num_heads, 4);
    assert_eq!(cfg.num_key_value_heads, 4);
    assert_eq!(cfg.num_layers, 2);
    assert_eq!(cfg.intermediate_size, 1024);
    assert_eq!(cfg.max_position_embeddings, 64);
    assert_eq!(cfg.rms_norm_eps, Some(1e-6));
    assert_eq!(cfg.rope_theta, Some(10000.0));
}

#[test]
fn model_config_clone_equals_original() {
    let cfg = tiny_config(8, 16, 2);
    let cfg2 = cfg.clone();
    assert_eq!(cfg.model.hidden_size, cfg2.model.hidden_size);
    assert_eq!(cfg.model.vocab_size, cfg2.model.vocab_size);
    assert_eq!(cfg.model.num_heads, cfg2.model.num_heads);
    assert_eq!(cfg.model.num_layers, cfg2.model.num_layers);
}

#[test]
fn minimal_valid_config_constructs_model() {
    // hidden=8, vocab=16, heads=2, head_dim=4 — all divisors line up
    let result = make_model(8, 16, 2);
    assert!(result.is_ok(), "minimal valid config should succeed: {:?}", result.err());
}

// ── RmsNorm computation tests ──────────────────────────────────────────────────

#[test]
fn rmsnorm_zero_input_near_zero_output() {
    // RMSNorm(0) = 0 / sqrt(eps) * gamma ≈ 0 for any reasonable eps
    let device = Device::Cpu;
    let hidden = 8;
    let eps = 1e-5f64;
    let gamma = Tensor::ones(hidden, DType::F32, &device).unwrap();
    let norm = RmsNorm::new(gamma, eps);

    let input = Tensor::zeros(&[1, hidden], DType::F32, &device).unwrap();
    let output = norm.forward(&input).unwrap();
    let vals: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();

    for v in &vals {
        assert!(v.abs() < 1e-2, "zero input → near-zero output, got {v}");
    }
}

#[test]
fn rmsnorm_unit_input_all_ones_gamma_known_value() {
    // input = [1, 1, ..., 1]; gamma = [1, ..., 1]
    // rms = sqrt(mean(1²) + eps) = sqrt(1 + eps)
    // output[i] = 1 / sqrt(1 + eps) * 1
    let device = Device::Cpu;
    let hidden = 16;
    let eps = 1e-5f64;
    let gamma = Tensor::ones(hidden, DType::F32, &device).unwrap();
    let norm = RmsNorm::new(gamma, eps);

    let input = Tensor::ones(&[1, hidden], DType::F32, &device).unwrap();
    let output = norm.forward(&input).unwrap();
    let vals: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
    let expected = 1.0f32 / (1.0f32 + eps as f32).sqrt();

    for v in &vals {
        assert!((v - expected).abs() < 1e-4, "expected {expected:.6} for unit input, got {v:.6}");
    }
}

#[test]
fn rmsnorm_known_values_3_and_4() {
    // input = [3, 4]; gamma = [1, 1]; eps = 0
    // rms = sqrt((9 + 16) / 2) = sqrt(12.5) ≈ 3.535534
    // output = [3/rms, 4/rms]
    let device = Device::Cpu;
    let eps = 1e-10f64; // near-zero eps for cleaner arithmetic
    let gamma = Tensor::ones(2, DType::F32, &device).unwrap();
    let norm = RmsNorm::new(gamma, eps);

    let input = Tensor::from_vec(vec![3.0f32, 4.0f32], &[1, 2], &device).unwrap();
    let output = norm.forward(&input).unwrap();
    let vals: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();

    let rms = ((9.0f32 + 16.0f32) / 2.0).sqrt();
    assert!((vals[0] - 3.0 / rms).abs() < 1e-4, "vals[0]={} expected {}", vals[0], 3.0 / rms);
    assert!((vals[1] - 4.0 / rms).abs() < 1e-4, "vals[1]={} expected {}", vals[1], 4.0 / rms);
}

#[test]
fn rmsnorm_output_is_finite_for_sinusoidal_input() {
    let device = Device::Cpu;
    let hidden = 64;
    let eps = 1e-5f64;
    let gamma = Tensor::ones(hidden, DType::F32, &device).unwrap();
    let norm = RmsNorm::new(gamma, eps);

    let vals: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.3).sin()).collect();
    let input = Tensor::from_vec(vals, &[1, hidden], &device).unwrap();
    let output = norm.forward(&input).unwrap();
    let out_vals: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();

    assert!(out_vals.iter().all(|v| v.is_finite()), "RmsNorm output must be finite");
}

#[test]
fn rmsnorm_doubling_gamma_doubles_output() {
    let device = Device::Cpu;
    let hidden = 16;
    let eps = 1e-5f64;

    let input_vals: Vec<f32> =
        (0..hidden).map(|i| (i as f32 / hidden as f32) * 0.5 + 0.1).collect();
    let input1 = Tensor::from_vec(input_vals.clone(), &[1, hidden], &device).unwrap();
    let input2 = Tensor::from_vec(input_vals, &[1, hidden], &device).unwrap();

    let gamma1 = Tensor::ones(hidden, DType::F32, &device).unwrap();
    let gamma2 = Tensor::from_vec(vec![2.0f32; hidden], hidden, &device).unwrap();

    let out1: Vec<f32> = RmsNorm::new(gamma1, eps)
        .forward(&input1)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let out2: Vec<f32> = RmsNorm::new(gamma2, eps)
        .forward(&input2)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    for (a, b) in out1.iter().zip(out2.iter()) {
        assert!((b - 2.0 * a).abs() < 1e-4, "2× gamma should double output: {a}×2 ≠ {b}");
    }
}

#[test]
fn rmsnorm_single_element_positive_input_normalizes_to_one() {
    // single elem x=[v], gamma=[1], eps≈0: output = v / sqrt(v²) * 1 = sign(v)
    let device = Device::Cpu;
    let eps = 1e-10f64;
    let gamma = Tensor::from_vec(vec![1.0f32], 1, &device).unwrap();
    let norm = RmsNorm::new(gamma, eps);

    let input = Tensor::from_vec(vec![5.0f32], &[1, 1], &device).unwrap();
    let output = norm.forward(&input).unwrap();
    let vals: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
    assert!((vals[0] - 1.0f32).abs() < 1e-4, "single positive: expected 1.0, got {}", vals[0]);
}

#[test]
fn rmsnorm_output_shape_preserved_2d() {
    let device = Device::Cpu;
    let (batch, hidden) = (3, 32);
    let eps = 1e-5f64;
    let gamma = Tensor::ones(hidden, DType::F32, &device).unwrap();
    let norm = RmsNorm::new(gamma, eps);

    let input = Tensor::ones(&[batch, hidden], DType::F32, &device).unwrap();
    let output = norm.forward(&input).unwrap();
    assert_eq!(output.dims(), &[batch, hidden], "2D shape must be preserved by RmsNorm");
}

#[test]
fn rmsnorm_output_shape_preserved_3d() {
    let device = Device::Cpu;
    let (batch, seq, hidden) = (2, 5, 16);
    let eps = 1e-5f64;
    let gamma = Tensor::ones(hidden, DType::F32, &device).unwrap();
    let norm = RmsNorm::new(gamma, eps);

    let input = Tensor::ones(&[batch, seq, hidden], DType::F32, &device).unwrap();
    let output = norm.forward(&input).unwrap();
    assert_eq!(output.dims(), &[batch, seq, hidden], "3D shape must be preserved by RmsNorm");
}

// ── TransformerModel forward tests ────────────────────────────────────────────

#[test]
fn model_rejects_hidden_not_divisible_by_heads() {
    // hidden=6, heads=4 → 6 % 4 ≠ 0 → construction error
    let device = Device::Cpu;
    let cfg = BitNetConfig {
        model: ModelConfig {
            hidden_size: 6,
            vocab_size: 10,
            num_heads: 4,
            num_key_value_heads: 4,
            num_layers: 1,
            intermediate_size: 24,
            max_position_embeddings: 4,
            rms_norm_eps: Some(1e-5),
            ..Default::default()
        },
        ..Default::default()
    };
    let vb = VarBuilder::zeros(DType::F32, &device);
    assert!(
        TransformerModel::new(cfg, vb).is_err(),
        "hidden not divisible by heads must return Err"
    );
}

#[test]
fn model_with_gqa_config_constructs_successfully() {
    // GQA: 4 query heads, 2 KV heads — both divide evenly
    let device = Device::Cpu;
    let cfg = BitNetConfig {
        model: ModelConfig {
            hidden_size: 8,
            vocab_size: 16,
            num_heads: 4,
            num_key_value_heads: 2,
            num_layers: 1,
            intermediate_size: 16,
            max_position_embeddings: 16,
            rms_norm_eps: Some(1e-5),
            ..Default::default()
        },
        ..Default::default()
    };
    let vb = VarBuilder::zeros(DType::F32, &device);
    assert!(TransformerModel::new(cfg, vb).is_ok(), "GQA config (4Q/2KV) must succeed");
}

#[test]
fn model_with_multi_layer_config_constructs_successfully() {
    let device = Device::Cpu;
    let cfg = BitNetConfig {
        model: ModelConfig {
            hidden_size: 8,
            vocab_size: 16,
            num_heads: 2,
            num_key_value_heads: 2,
            num_layers: 3,
            intermediate_size: 16,
            max_position_embeddings: 16,
            rms_norm_eps: Some(1e-5),
            ..Default::default()
        },
        ..Default::default()
    };
    let vb = VarBuilder::zeros(DType::F32, &device);
    assert!(TransformerModel::new(cfg, vb).is_ok(), "multi-layer config must construct");
}

#[test]
fn embed_output_shape_is_batch_seq_hidden() {
    let (hidden, vocab, heads) = (8, 16, 2);
    let model = make_model(hidden, vocab, heads).unwrap();
    let out = model.embed(&[0u32, 1u32, 2u32]).unwrap();
    assert_eq!(out.dims(), &[1, 3, hidden], "embed output must be [1, seq_len, hidden]");
}

#[test]
fn embed_single_token_shape_is_1_1_hidden() {
    let (hidden, vocab, heads) = (8, 16, 2);
    let model = make_model(hidden, vocab, heads).unwrap();
    let out = model.embed(&[7u32]).unwrap();
    assert_eq!(out.dims(), &[1, 1, hidden], "single token embed must be [1, 1, hidden]");
}

#[test]
fn forward_full_output_shape_is_batch_seq_vocab() {
    let (hidden, vocab, heads) = (8, 16, 2);
    let model = make_model(hidden, vocab, heads).unwrap();
    let device = Device::Cpu;
    let ids = Tensor::from_vec(vec![0u32, 1u32, 2u32], &[1, 3], &device).unwrap();
    let logits = model.forward_full(&ids).unwrap();
    assert_eq!(logits.dims(), &[1, 3, vocab], "forward_full shape must be [B, T, V]");
}

#[test]
fn forward_full_single_token_produces_1_1_vocab_logits() {
    let (hidden, vocab, heads) = (8, 16, 2);
    let model = make_model(hidden, vocab, heads).unwrap();
    let device = Device::Cpu;
    let ids = Tensor::from_vec(vec![0u32], &[1, 1], &device).unwrap();
    let logits = model.forward_full(&ids).unwrap();
    assert_eq!(logits.dims(), &[1, 1, vocab], "single-token logits must be [1, 1, V]");
}

#[test]
fn forward_full_output_is_finite() {
    // Zero-weight model should still produce finite (uniform) logits
    let (hidden, vocab, heads) = (8, 16, 2);
    let model = make_model(hidden, vocab, heads).unwrap();
    let device = Device::Cpu;
    let ids = Tensor::from_vec(vec![0u32, 1u32], &[1, 2], &device).unwrap();
    let logits = model.forward_full(&ids).unwrap();
    let vals: Vec<f32> = logits.flatten_all().unwrap().to_vec1().unwrap();
    assert!(vals.iter().all(|v| v.is_finite()), "forward_full must produce only finite values");
}

#[test]
fn forward_full_is_deterministic_for_same_input() {
    let (hidden, vocab, heads) = (8, 16, 2);
    let model = make_model(hidden, vocab, heads).unwrap();
    let device = Device::Cpu;

    let ids1 = Tensor::from_vec(vec![2u32, 5u32], &[1, 2], &device).unwrap();
    let ids2 = Tensor::from_vec(vec![2u32, 5u32], &[1, 2], &device).unwrap();

    let logits1: Vec<f32> =
        model.forward_full(&ids1).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    let logits2: Vec<f32> =
        model.forward_full(&ids2).unwrap().flatten_all().unwrap().to_vec1().unwrap();

    assert_eq!(logits1, logits2, "forward_full must be deterministic");
}

// ── KV Cache edge-case tests ───────────────────────────────────────────────────

#[test]
fn kv_cache_rejects_heads_not_divisible_by_kv_heads() {
    // num_heads=6, num_kv_heads=4: 6 % 4 ≠ 0 → error
    let device = Device::Cpu;
    let cfg = BitNetConfig {
        model: ModelConfig {
            hidden_size: 12,
            vocab_size: 10,
            num_heads: 6,
            num_key_value_heads: 4,
            num_layers: 1,
            intermediate_size: 48,
            max_position_embeddings: 8,
            rms_norm_eps: Some(1e-5),
            ..Default::default()
        },
        ..Default::default()
    };
    assert!(
        KVCache::new(&cfg, 1, &device).is_err(),
        "num_heads not divisible by num_kv_heads must fail"
    );
}

#[test]
fn kv_cache_layer_count_matches_config() {
    let device = Device::Cpu;
    let n_layers = 3;
    let cfg = BitNetConfig {
        model: ModelConfig {
            hidden_size: 8,
            vocab_size: 16,
            num_heads: 2,
            num_key_value_heads: 2,
            num_layers: n_layers,
            intermediate_size: 16,
            max_position_embeddings: 16,
            rms_norm_eps: Some(1e-5),
            ..Default::default()
        },
        ..Default::default()
    };
    let cache = KVCache::new(&cfg, 1, &device).unwrap();
    assert_eq!(cache.layers.len(), n_layers, "KVCache must have one layer per config layer");
}

#[test]
fn layer_kv_cache_multi_append_cumulates_seq_len() {
    let device = Device::Cpu;
    let mut cache = LayerKVCache::new(1, 2, 32, 4, &device).unwrap();

    for step in 1..=3 {
        let k = Tensor::zeros(&[1, 2, 1, 4], DType::F32, &device).unwrap();
        let v = Tensor::zeros(&[1, 2, 1, 4], DType::F32, &device).unwrap();
        cache.append(&k, &v).unwrap();
        assert_eq!(cache.seq_len, step, "seq_len after {step} appends");
    }
}

#[test]
fn layer_kv_cache_head_mismatch_on_second_append_returns_error() {
    let device = Device::Cpu;
    let mut cache = LayerKVCache::new(1, 4, 16, 8, &device).unwrap();

    // First append: correct
    let k1 = Tensor::zeros(&[1, 4, 1, 8], DType::F32, &device).unwrap();
    let v1 = Tensor::zeros(&[1, 4, 1, 8], DType::F32, &device).unwrap();
    cache.append(&k1, &v1).unwrap();

    // Second append: wrong head count
    let k2 = Tensor::zeros(&[1, 2, 1, 8], DType::F32, &device).unwrap();
    let v2 = Tensor::zeros(&[1, 2, 1, 8], DType::F32, &device).unwrap();
    assert!(cache.append(&k2, &v2).is_err(), "head count mismatch must be an error");
}

// ── Property tests ─────────────────────────────────────────────────────────────

proptest! {
    /// RmsNorm with constant gamma `g` and all-ones input:
    /// output[i] = 1 / sqrt(1 + eps) * g ≈ g for small eps.
    #[test]
    fn rmsnorm_output_magnitude_scales_with_constant_gamma(
        hidden in 4usize..=32,
        gamma_scale in 0.01f32..=10.0f32,
    ) {
        let device = Device::Cpu;
        let eps = 1e-5f64;
        let gamma_vals: Vec<f32> = vec![gamma_scale; hidden];
        let gamma = Tensor::from_vec(gamma_vals, hidden, &device).unwrap();
        let norm = RmsNorm::new(gamma, eps);

        let input = Tensor::ones(&[1, hidden], DType::F32, &device).unwrap();
        let output = norm.forward(&input).unwrap();
        let vals: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
        let expected = gamma_scale / (1.0f32 + eps as f32).sqrt();

        for v in &vals {
            prop_assert!(
                (v - expected).abs() < 1e-3,
                "expected ~{expected}, got {v} (gamma={gamma_scale}, hidden={hidden})"
            );
        }
    }

    /// Any hidden=n_heads×head_dim config must construct a model without error.
    #[test]
    fn valid_head_divisor_configs_construct_model(
        n_heads in 1usize..=4,
        head_dim in prop_oneof![Just(2usize), Just(4usize)],
    ) {
        let hidden = n_heads * head_dim;
        let vocab = 16usize;
        let device = Device::Cpu;
        let cfg = BitNetConfig {
            model: ModelConfig {
                hidden_size: hidden,
                vocab_size: vocab,
                num_heads: n_heads,
                num_key_value_heads: n_heads,
                num_layers: 1,
                intermediate_size: hidden * 2,
                max_position_embeddings: 8,
                rms_norm_eps: Some(1e-5),
                ..Default::default()
            },
            ..Default::default()
        };
        let vb = VarBuilder::zeros(DType::F32, &device);
        let result = TransformerModel::new(cfg, vb);
        prop_assert!(result.is_ok(), "valid config must succeed: {:?}", result.err());
    }

    /// RmsNorm output is always finite for finite input and positive eps.
    #[test]
    fn rmsnorm_output_always_finite_for_finite_input(
        hidden in 2usize..=16,
        input_scale in -10.0f32..=10.0f32,
    ) {
        let device = Device::Cpu;
        let eps = 1e-5f64;
        let gamma = Tensor::ones(hidden, DType::F32, &device).unwrap();
        let norm = RmsNorm::new(gamma, eps);

        let input_vals: Vec<f32> = (0..hidden)
            .map(|i| (i as f32 * 0.7).sin() * input_scale)
            .collect();
        let input = Tensor::from_vec(input_vals, &[1, hidden], &device).unwrap();
        let output = norm.forward(&input).unwrap();
        let vals: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();

        prop_assert!(vals.iter().all(|v| v.is_finite()), "RmsNorm must output finite values");
    }
}

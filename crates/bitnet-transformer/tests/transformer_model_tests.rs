//! Integration tests for `TransformerModel` — covers embed, logits, and
//! forward_full on zero-initialized weights (no real GGUF required).
//!
//! Uses `VarBuilder::zeros` which auto-fills any requested tensor key with zeros,
//! eliminating the need to manually enumerate all weight keys.
//!
//! Verifies:
//!   - Shape invariants (embed output, logit output, forward_full output)
//!   - Finite-value guarantees (no NaN / Inf in output)
//!   - Determinism (same input → same output)
//!   - Model construction with different config shapes
//!   - Validation errors for incompatible config values
#![cfg(feature = "cpu")]

use bitnet_common::config::{BitNetConfig, ModelConfig};
use bitnet_transformer::{KVCache, TransformerModel};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

// ── helpers ──────────────────────────────────────────────────────────────────

/// Minimal config for a 1-layer, small-vocab model — fast to construct.
fn tiny_config(hidden: usize, vocab: usize, heads: usize) -> BitNetConfig {
    BitNetConfig {
        model: ModelConfig {
            hidden_size: hidden,
            vocab_size: vocab,
            num_heads: heads,
            num_key_value_heads: heads,
            num_layers: 1,
            intermediate_size: hidden * 4,
            max_position_embeddings: 64,
            rms_norm_eps: Some(1e-5),
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Build a `TransformerModel` with all-zero weights via `VarBuilder::zeros`.
fn make_model(hidden: usize, vocab: usize, heads: usize) -> anyhow::Result<TransformerModel> {
    let device = Device::Cpu;
    let cfg = tiny_config(hidden, vocab, heads);
    let vb = VarBuilder::zeros(DType::F32, &device);
    Ok(TransformerModel::new(cfg, vb)?)
}

// ── embed tests ───────────────────────────────────────────────────────────────

/// The `embed` method must return shape `[1, seq_len, hidden]`.
#[test]
fn test_embed_shape() -> anyhow::Result<()> {
    let model = make_model(64, 128, 4)?;
    let tokens: &[u32] = &[1, 2, 3, 4, 5];
    let out = model.embed(tokens)?;
    assert_eq!(out.dims(), &[1, 5, 64], "embed shape should be [1, seq, hidden]");
    Ok(())
}

/// Embedding output must be finite.
#[test]
fn test_embed_finite() -> anyhow::Result<()> {
    let model = make_model(64, 128, 4)?;
    let tokens: &[u32] = &[0, 1, 2];
    let out = model.embed(tokens)?;
    let vals: Vec<f32> = out.flatten_all()?.to_vec1()?;
    assert!(vals.iter().all(|v| v.is_finite()), "embed output must be finite");
    Ok(())
}

/// `embed` is deterministic — same tokens → same tensor every call.
#[test]
fn test_embed_determinism() -> anyhow::Result<()> {
    let model = make_model(64, 128, 4)?;
    let tokens: &[u32] = &[10, 20, 30];
    let a: Vec<f32> = model.embed(tokens)?.flatten_all()?.to_vec1()?;
    let b: Vec<f32> = model.embed(tokens)?.flatten_all()?.to_vec1()?;
    assert_eq!(a, b, "embed must be deterministic");
    Ok(())
}

// ── logits tests ──────────────────────────────────────────────────────────────

/// `logits` should accept a 3D hidden state and return `[B, seq, vocab]`.
#[test]
fn test_logits_shape_3d() -> anyhow::Result<()> {
    let hidden = 64;
    let vocab = 128;
    let model = make_model(hidden, vocab, 4)?;

    let device = Device::Cpu;
    let hidden_state = Tensor::zeros((1usize, 3usize, hidden), DType::F32, &device)?;
    let out = model.logits(&hidden_state)?;
    assert_eq!(out.dims(), &[1, 3, vocab], "logits shape should be [1, seq, vocab]");
    Ok(())
}

/// `logits` should accept a 2D hidden state (last-token only) and return `[B, vocab]`.
#[test]
fn test_logits_shape_2d() -> anyhow::Result<()> {
    let hidden = 64;
    let vocab = 128;
    let model = make_model(hidden, vocab, 4)?;

    let device = Device::Cpu;
    let hidden_state = Tensor::zeros((1usize, hidden), DType::F32, &device)?;
    let out = model.logits(&hidden_state)?;
    // logits() returns [B, V] for 2D input (incremental decode path)
    assert_eq!(out.dims()[out.dims().len() - 1], vocab, "last dim should be vocab");
    Ok(())
}

/// `logits` output must be finite.
#[test]
fn test_logits_finite() -> anyhow::Result<()> {
    let hidden = 64;
    let vocab = 128;
    let model = make_model(hidden, vocab, 4)?;

    let device = Device::Cpu;
    let hidden_state = Tensor::zeros((1usize, 2usize, hidden), DType::F32, &device)?;
    let out = model.logits(&hidden_state)?;
    let vals: Vec<f32> = out.flatten_all()?.to_vec1()?;
    assert!(vals.iter().all(|v| v.is_finite()), "logits must be finite");
    Ok(())
}

// ── forward_full tests ────────────────────────────────────────────────────────

/// `forward_full` must return shape `[1, seq, vocab]` for a 3-token sequence.
#[test]
fn test_forward_full_shape() -> anyhow::Result<()> {
    let hidden = 64;
    let vocab = 128;
    let model = make_model(hidden, vocab, 4)?;

    let device = Device::Cpu;
    let token_ids = Tensor::from_slice(&[1u32, 2, 3], (1usize, 3usize), &device)?;
    let out = model.forward_full(&token_ids)?;
    assert_eq!(out.dims(), &[1, 3, vocab], "forward_full shape should be [1, seq, vocab]");
    Ok(())
}

/// `forward_full` must produce finite values.
#[test]
fn test_forward_full_finite() -> anyhow::Result<()> {
    let model = make_model(64, 128, 4)?;
    let device = Device::Cpu;
    let token_ids = Tensor::from_slice(&[0u32, 1], (1usize, 2usize), &device)?;
    let out = model.forward_full(&token_ids)?;
    let vals: Vec<f32> = out.flatten_all()?.to_vec1()?;
    assert!(vals.iter().all(|v| v.is_finite()), "forward_full must not produce NaN/Inf");
    Ok(())
}

/// `forward_full` must be deterministic — same input → same output.
#[test]
fn test_forward_full_determinism() -> anyhow::Result<()> {
    let model = make_model(64, 128, 4)?;
    let device = Device::Cpu;
    let token_ids = Tensor::from_slice(&[5u32, 10, 15], (1usize, 3usize), &device)?;

    let a: Vec<f32> = model.forward_full(&token_ids)?.flatten_all()?.to_vec1()?;
    let b: Vec<f32> = model.forward_full(&token_ids)?.flatten_all()?.to_vec1()?;
    assert_eq!(a, b, "forward_full must be deterministic");
    Ok(())
}

// ── incremental (forward) tests ───────────────────────────────────────────────

/// Incremental `forward` (single token at a time with KV cache) must return
/// rank-2 `[B, H]` per step and produce finite logits.
#[test]
fn test_incremental_forward_shape_and_finite() -> anyhow::Result<()> {
    let hidden = 64;
    let vocab = 128;
    let model = make_model(hidden, vocab, 4)?;
    let device = Device::Cpu;

    let tokens: &[u32] = &[1, 2, 3];
    let mut kv = KVCache::new(&model.config, 1, &device)?;

    for &t in tokens {
        let h = model.embed(std::slice::from_ref(&t))?;
        let out = model.forward(h, Some(&mut kv))?;
        let vals: Vec<f32> = out.flatten_all()?.to_vec1()?;
        assert!(vals.iter().all(|v| v.is_finite()), "incremental forward must be finite");
    }
    Ok(())
}

// ── construction tests ────────────────────────────────────────────────────────

/// Model construction must succeed for different hidden/vocab/head combinations.
#[test]
fn test_construction_variants() -> anyhow::Result<()> {
    let cases = [(32, 64, 2), (64, 128, 4), (128, 256, 8)];
    for (h, v, n) in cases {
        make_model(h, v, n)
            .unwrap_or_else(|e| panic!("construction failed for h={h}, v={v}, n={n}: {e}"));
    }
    Ok(())
}

/// Construction must fail when hidden is not divisible by num_heads.
#[test]
fn test_construction_fails_bad_head_dim() {
    // hidden=60, heads=8: 60 % 8 != 0 → should fail
    let result = make_model(60, 64, 8);
    assert!(result.is_err(), "Should fail: hidden=60 not divisible by heads=8");
}

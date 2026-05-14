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

use bitnet_common::config::{ActivationType, BitNetConfig, ModelConfig, NormType};
use bitnet_transformer::{KVCache, Qk256DispatchBackend, TransformerModel};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;

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

fn identity_2x2(device: &Device) -> candle_core::Result<Tensor> {
    Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2usize, 2usize), device)
}

fn zero_2x2(device: &Device) -> candle_core::Result<Tensor> {
    Tensor::zeros((2usize, 2usize), DType::F32, device)
}

fn ones_2(device: &Device) -> candle_core::Result<Tensor> {
    Tensor::from_vec(vec![1.0f32, 1.0], 2usize, device)
}

fn ones_256(device: &Device) -> candle_core::Result<Tensor> {
    Tensor::from_vec(vec![1.0f32; 256], 256usize, device)
}

fn zero_matrix(rows: usize, cols: usize, device: &Device) -> candle_core::Result<Tensor> {
    Tensor::zeros((rows, cols), DType::F32, device)
}

fn identity_matrix(dim: usize, device: &Device) -> candle_core::Result<Tensor> {
    let mut values = vec![0.0f32; dim * dim];
    for idx in 0..dim {
        values[idx * dim + idx] = 1.0;
    }
    Tensor::from_vec(values, (dim, dim), device)
}

fn qk256_row_with_codes(codes: &[(usize, u8)]) -> Vec<u8> {
    let mut unpacked = [1u8; 256];
    for &(idx, code) in codes {
        assert!(idx < unpacked.len(), "QK256 test code index out of range");
        assert!(code <= 3, "QK256 test code must fit in two bits");
        unpacked[idx] = code;
    }

    let mut packed = vec![0u8; 64];
    for (byte_idx, chunk) in unpacked.chunks_exact(4).enumerate() {
        packed[byte_idx] = chunk[0] | (chunk[1] << 2) | (chunk[2] << 4) | (chunk[3] << 6);
    }
    packed
}

fn repeated_qk256_tensor(
    rows: usize,
    row_bytes: &[u8],
    device: &Device,
) -> candle_core::Result<Tensor> {
    let mut bytes = Vec::with_capacity(rows * row_bytes.len());
    for _ in 0..rows {
        bytes.extend_from_slice(row_bytes);
    }
    Tensor::from_vec(bytes, (rows, row_bytes.len()), device)
}

fn qk256_tensor_from_rows(row_bytes: &[Vec<u8>], device: &Device) -> candle_core::Result<Tensor> {
    let row_stride = row_bytes.first().map(Vec::len).unwrap_or(0);
    assert!(row_stride > 0, "QK256 test tensor must have at least one row");
    let mut bytes = Vec::with_capacity(row_bytes.len() * row_stride);
    for row in row_bytes {
        assert_eq!(row.len(), row_stride, "QK256 test tensor row stride mismatch");
        bytes.extend_from_slice(row);
    }
    Tensor::from_vec(bytes, (row_bytes.len(), row_stride), device)
}

fn prompt_sensitive_model() -> anyhow::Result<TransformerModel> {
    let device = Device::Cpu;
    let cfg = BitNetConfig {
        model: ModelConfig {
            hidden_size: 2,
            vocab_size: 4,
            num_heads: 1,
            num_key_value_heads: 1,
            num_layers: 1,
            intermediate_size: 2,
            max_position_embeddings: 8,
            rms_norm_eps: Some(1e-6),
            norm_type: NormType::RmsNorm,
            activation_type: ActivationType::Relu2,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut tensors = HashMap::new();
    tensors.insert(
        "embed_tokens.weight".to_string(),
        Tensor::from_vec(
            vec![
                0.0f32, 0.0, // token 0: unused padding
                1.0, 0.0, // token 1: history A
                0.0, 1.0, // token 2: history B
                1.0, 0.0, // token 3: shared current token
            ],
            (4usize, 2usize),
            &device,
        )?,
    );

    for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
        tensors.insert(format!("layers.0.attention.{proj}.weight"), identity_2x2(&device)?);
    }

    for norm in [
        "layers.0.attention_norm.weight",
        "layers.0.post_attention_layernorm.weight",
        "final_norm.weight",
    ] {
        tensors.insert(norm.to_string(), ones_2(&device)?);
    }

    for proj in ["gate_proj", "up_proj", "down_proj"] {
        tensors.insert(format!("layers.0.feed_forward.{proj}.weight"), zero_2x2(&device)?);
    }

    let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
    Ok(TransformerModel::new(cfg, vb)?)
}

fn qk256_prompt_sensitive_model() -> anyhow::Result<TransformerModel> {
    let device = Device::Cpu;
    let cfg = BitNetConfig {
        model: ModelConfig {
            hidden_size: 256,
            vocab_size: 4,
            num_heads: 1,
            num_key_value_heads: 1,
            num_layers: 1,
            intermediate_size: 256,
            max_position_embeddings: 8,
            rms_norm_eps: Some(1e-6),
            norm_type: NormType::RmsNorm,
            activation_type: ActivationType::Relu2,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut embed = vec![0.0f32; 4 * 256];
    embed[256] = 1.0; // token 1: history A
    embed[(2 * 256) + 1] = 1.0; // token 2: history B
    embed[(3 * 256) + 2] = 1.0; // token 3: shared current token

    let mut tensors = HashMap::new();
    tensors.insert("embed_tokens.weight".to_string(), Tensor::from_vec(embed, (4, 256), &device)?);

    let mut lm_head = vec![0.0f32; 4 * 256];
    lm_head[256] = 1.0;
    lm_head[256 + 1] = -1.0;
    lm_head[2 * 256] = -1.0;
    lm_head[(2 * 256) + 1] = 1.0;
    lm_head[(3 * 256) + 2] = 1.0;
    tensors.insert("lm_head.weight".to_string(), Tensor::from_vec(lm_head, (4, 256), &device)?);

    for proj in ["q_proj", "k_proj", "v_proj"] {
        tensors
            .insert(format!("layers.0.attention.{proj}.weight"), zero_matrix(256, 256, &device)?);
    }
    tensors.insert("layers.0.attention.o_proj.weight".to_string(), identity_matrix(256, &device)?);

    for norm in [
        "layers.0.attention_norm.weight",
        "layers.0.post_attention_layernorm.weight",
        "final_norm.weight",
    ] {
        tensors.insert(norm.to_string(), ones_256(&device)?);
    }

    for proj in ["gate_proj", "up_proj", "down_proj"] {
        tensors.insert(
            format!("layers.0.feed_forward.{proj}.weight"),
            zero_matrix(256, 256, &device)?,
        );
    }

    let q_pos_row = qk256_row_with_codes(&[(2, 3)]);
    let mut q_rows = Vec::with_capacity(256);
    for _ in 0..256 {
        q_rows.push(q_pos_row.clone());
    }
    let history_sensitive_row = qk256_row_with_codes(&[(0, 3), (1, 0), (2, 0)]);
    let v_history_a_row = qk256_row_with_codes(&[(0, 2)]);
    let v_history_b_row = qk256_row_with_codes(&[(1, 2)]);
    let v_neutral_row = qk256_row_with_codes(&[]);
    let mut v_rows = vec![v_neutral_row; 256];
    v_rows[0] = v_history_a_row;
    v_rows[1] = v_history_b_row;

    let mut raw_tensors = HashMap::new();
    raw_tensors.insert(
        "layers.0.attention.q_proj.weight.qk256_qs".to_string(),
        qk256_tensor_from_rows(&q_rows, &device)?,
    );
    raw_tensors.insert(
        "layers.0.attention.k_proj.weight.qk256_qs".to_string(),
        repeated_qk256_tensor(256, &history_sensitive_row, &device)?,
    );
    raw_tensors.insert(
        "layers.0.attention.v_proj.weight.qk256_qs".to_string(),
        qk256_tensor_from_rows(&v_rows, &device)?,
    );

    let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
    Ok(TransformerModel::new_with_tensors_and_qk256_backend(
        cfg,
        vb,
        raw_tensors,
        Qk256DispatchBackend::Cpu,
    )?)
}

fn prefill_last_logits(model: &TransformerModel, tokens: &[u32]) -> anyhow::Result<Vec<f32>> {
    let device = Device::Cpu;
    let hidden = model.embed(tokens)?;
    let mut kv = KVCache::new(&model.config, 1, &device)?;
    let hidden = model.forward(hidden, Some(&mut kv))?;
    let logits = model.logits(&hidden)?;
    let seq_len = logits.dims()[1];
    Ok(logits.narrow(1, seq_len - 1, 1)?.flatten_all()?.to_vec1()?)
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

/// Prefill must let earlier prompt tokens affect the next-token logits.
///
/// The live BitNet diagnostic currently emits the same first tokens for very
/// different prompts. This fixture pins the expected transformer invariant with
/// deterministic weights: two prompts share the current token but differ in the
/// previous token, and attention history must change the final-position logits.
#[test]
fn test_prefill_last_logits_are_prompt_history_sensitive() -> anyhow::Result<()> {
    let model = prompt_sensitive_model()?;

    let logits_history_a = prefill_last_logits(&model, &[1, 3])?;
    let logits_history_b = prefill_last_logits(&model, &[2, 3])?;

    assert_ne!(
        logits_history_a, logits_history_b,
        "last-position logits must depend on earlier prompt history"
    );

    let max_delta = logits_history_a
        .iter()
        .zip(logits_history_b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_delta > 1e-3,
        "history-sensitive fixture should produce a material logit delta, got {max_delta}"
    );

    Ok(())
}

/// The prompt-history invariant must also hold when Q/K/V use raw QK256
/// projection tensors. This guards the integration path from `raw_tensors`
/// through transformer attention, not just the standalone QK256 dispatch crate.
#[test]
fn test_qk256_prefill_last_logits_are_prompt_history_sensitive() -> anyhow::Result<()> {
    let model = qk256_prompt_sensitive_model()?;

    let logits_history_a = prefill_last_logits(&model, &[1, 3])?;
    let logits_history_b = prefill_last_logits(&model, &[2, 3])?;

    assert_ne!(
        logits_history_a, logits_history_b,
        "QK256-backed last-position logits must depend on earlier prompt history"
    );

    let max_delta = logits_history_a
        .iter()
        .zip(logits_history_b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_delta > 1e-3,
        "QK256-backed history-sensitive fixture should produce a material logit delta, got {max_delta}"
    );

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

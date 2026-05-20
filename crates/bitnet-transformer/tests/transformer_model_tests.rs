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
use bitnet_transformer::{
    DenseLinearOutputStorageApiBoundary, DenseLinearPackedQ8Payload,
    DenseLinearRuntimeHookBoundary, DenseLinearRuntimeHookDescriptor,
    DenseLinearRuntimeHookRegistry, KVCache, TransformerForwardWorkspace, TransformerModel,
};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, VarBuilder};

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

#[test]
fn test_incremental_forward_workspace_matches_existing_path() -> anyhow::Result<()> {
    let hidden = 64;
    let model = make_model(hidden, 128, 4)?;
    let device = Device::Cpu;
    let token = 7u32;
    let h = model.embed(std::slice::from_ref(&token))?;

    let mut existing_kv = KVCache::new(&model.config, 1, &device)?;
    let mut workspace_kv = KVCache::new(&model.config, 1, &device)?;
    let mut workspace = TransformerForwardWorkspace::new();

    let existing: Vec<f32> =
        model.forward(h.clone(), Some(&mut existing_kv))?.flatten_all()?.to_vec1()?;
    let with_workspace =
        model.forward_with_workspace(h, Some(&mut workspace_kv), &mut workspace)?;
    let with_workspace_values: Vec<f32> = with_workspace.flatten_all()?.to_vec1()?;

    assert_eq!(existing, with_workspace_values, "workspace API must preserve forward output");
    assert_eq!(workspace.model_forward_calls(), 1);
    assert_eq!(workspace.block_forward_calls(), model.config.model.num_layers);
    assert_eq!(workspace.feed_forward_calls(), model.config.model.num_layers);
    assert_eq!(workspace.last_output_shape(), with_workspace.dims());
    assert_eq!(workspace.last_output_shape(), &[1, 1, hidden]);
    assert_eq!(
        workspace.reuse_status(),
        "dense_linear_output_storage_blocked_by_candle_tensor_ops"
    );
    assert_eq!(workspace.workspace_owned_output_count(), model.config.model.num_layers);
    assert_eq!(workspace.down_proj_output_storage_attempts(), model.config.model.num_layers);
    let Some(surface) = workspace.first_output_surface() else {
        anyhow::bail!("workspace should classify one output surface");
    };
    assert_eq!(surface.name, "feed_forward.down_proj.output");
    assert_eq!(surface.storage_owner, "TransformerForwardWorkspace");
    assert_eq!(surface.status, "dense_linear_output_storage_blocked_by_candle_tensor_ops");
    assert_eq!(surface.last_shape, vec![1, 1, hidden]);
    assert_eq!(surface.linear_weight_shape, vec![hidden, hidden * 4]);
    assert_eq!(surface.linear_bias_shape, Some(vec![hidden]));
    assert!(surface.weight_accessible);
    assert!(surface.bias_accessible);
    assert!(!surface.can_fill_caller_output_storage);
    assert!(
        !workspace.tensor_reuse_enabled(),
        "SLM-CPU-041 proves the dense linear output hook still lacks reusable Candle storage"
    );
    Ok(())
}

#[test]
fn dense_linear_output_storage_boundary_records_candle_tensor_blocker() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let weight = Tensor::zeros((3, 2), DType::F32, &device)?;
    let bias = Tensor::zeros(3, DType::F32, &device)?;
    let linear = Linear::new(weight, Some(bias));

    let boundary = DenseLinearOutputStorageApiBoundary::from_candle_linear(
        "feed_forward.down_proj.output",
        &linear,
    );

    assert_eq!(boundary.role, "feed_forward.down_proj.output");
    assert_eq!(boundary.weight_shape, vec![3, 2]);
    assert_eq!(boundary.bias_shape, Some(vec![3]));
    assert!(boundary.weight_accessible);
    assert!(boundary.bias_accessible);
    assert!(!boundary.can_fill_caller_output_storage);
    assert_eq!(boundary.status, "dense_linear_output_storage_blocked_by_candle_tensor_ops");
    assert!(
        boundary.reason.contains("Tensor::matmul")
            && boundary.reason.contains("caller-provided output-storage")
    );
    Ok(())
}

#[test]
fn dense_linear_runtime_hook_boundary_defaults_to_eager_f32() -> anyhow::Result<()> {
    let model = make_model(8, 16, 2)?;

    let boundary = model.dense_linear_runtime_hook_boundary("layers.0.attention.q_proj.weight");

    assert_eq!(boundary.selected_path, "eager_f32_candle");
    assert_eq!(boundary.selected_kernel, "dense-f32-candle-linear");
    assert!(!boundary.sidecar_descriptor_present);
    assert!(!boundary.runtime_compute_enabled);
    assert!(boundary.preserves_eager_f32());
    assert!(!boundary.speedup_claim);
    Ok(())
}

#[test]
fn dense_linear_runtime_hook_boundary_accepts_inert_q8_sidecar_descriptor() -> anyhow::Result<()> {
    let config = tiny_config(8, 16, 2);
    let device = Device::Cpu;
    let vb = VarBuilder::zeros(DType::F32, &device);
    let mut hooks = DenseLinearRuntimeHookRegistry::default();
    hooks.insert(
        "layers.0.attention.q_proj.weight".to_string(),
        DenseLinearRuntimeHookDescriptor {
            tensor_name: "blk.0.attn_q.weight".to_string(),
            role: "AttentionQ".to_string(),
            sidecar_payload_sha256: Some("abc123".to_string()),
            packed_q8_payload: None,
            runtime_compute_enabled: false,
        },
    );
    let model = TransformerModel::new_with_tensors_and_dense_linear_hooks(
        config,
        vb,
        Default::default(),
        hooks,
    )?;

    let boundary = model.dense_linear_runtime_hook_boundary("layers.0.attention.q_proj.weight");

    assert_eq!(boundary.selected_path, "eager_f32_candle");
    assert_eq!(boundary.selected_kernel, "dense-f32-candle-linear");
    assert!(boundary.sidecar_descriptor_present);
    assert_eq!(boundary.sidecar_role.as_deref(), Some("AttentionQ"));
    assert_eq!(boundary.sidecar_payload_sha256.as_deref(), Some("abc123"));
    assert!(!boundary.sidecar_payload_bytes_available);
    assert_eq!(boundary.sidecar_payload_bytes, None);
    assert!(!boundary.sidecar_payload_contract_valid);
    assert!(!boundary.runtime_compute_enabled);
    assert!(boundary.preserves_eager_f32());
    assert!(boundary.generated_id_preservation_required_before_runtime_use);
    Ok(())
}

#[test]
fn dense_linear_runtime_hook_boundary_can_carry_payload_without_enabling_compute()
-> anyhow::Result<()> {
    let config = tiny_config(8, 16, 2);
    let device = Device::Cpu;
    let vb = VarBuilder::zeros(DType::F32, &device);
    let payload_bytes = vec![0_u8; 68];
    let mut hooks = DenseLinearRuntimeHookRegistry::default();
    hooks.insert(
        "layers.0.attention.q_proj.weight".to_string(),
        DenseLinearRuntimeHookDescriptor {
            tensor_name: "blk.0.attn_q.weight".to_string(),
            role: "AttentionQ".to_string(),
            sidecar_payload_sha256: Some("sha256:payload".to_string()),
            packed_q8_payload: Some(DenseLinearPackedQ8Payload {
                tensor_name: "blk.0.attn_q.weight".to_string(),
                packed_q8_bytes: std::sync::Arc::from(payload_bytes.into_boxed_slice()),
                q8_block_size: 32,
                q8_block_count: 2,
                matrix_rows: 8,
                matrix_cols: 8,
            }),
            runtime_compute_enabled: true,
        },
    );
    let model = TransformerModel::new_with_tensors_and_dense_linear_hooks(
        config,
        vb,
        Default::default(),
        hooks,
    )?;

    let boundary = model.dense_linear_runtime_hook_boundary("layers.0.attention.q_proj.weight");

    assert_eq!(boundary.selected_path, "eager_f32_candle");
    assert_eq!(boundary.selected_kernel, "dense-f32-candle-linear");
    assert!(boundary.sidecar_descriptor_present);
    assert!(boundary.sidecar_payload_bytes_available);
    assert_eq!(boundary.sidecar_payload_bytes, Some(68));
    assert_eq!(boundary.sidecar_q8_block_count, Some(2));
    assert_eq!(boundary.sidecar_matrix_rows, Some(8));
    assert_eq!(boundary.sidecar_matrix_cols, Some(8));
    assert!(boundary.sidecar_payload_contract_valid);
    assert!(!boundary.runtime_compute_enabled);
    assert!(boundary.preserves_eager_f32());
    assert!(boundary.generated_id_preservation_required_before_runtime_use);
    Ok(())
}

#[test]
fn dense_linear_runtime_hook_boundary_rejects_payload_tensor_mismatch() -> anyhow::Result<()> {
    let config = tiny_config(8, 16, 2);
    let device = Device::Cpu;
    let vb = VarBuilder::zeros(DType::F32, &device);
    let payload_bytes = vec![0_u8; 68];
    let mut hooks = DenseLinearRuntimeHookRegistry::default();
    hooks.insert(
        "layers.0.attention.q_proj.weight".to_string(),
        DenseLinearRuntimeHookDescriptor {
            tensor_name: "blk.0.attn_q.weight".to_string(),
            role: "AttentionQ".to_string(),
            sidecar_payload_sha256: Some("sha256:payload".to_string()),
            packed_q8_payload: Some(DenseLinearPackedQ8Payload {
                tensor_name: "blk.0.attn_k.weight".to_string(),
                packed_q8_bytes: std::sync::Arc::from(payload_bytes.into_boxed_slice()),
                q8_block_size: 32,
                q8_block_count: 2,
                matrix_rows: 8,
                matrix_cols: 8,
            }),
            runtime_compute_enabled: true,
        },
    );
    let model = TransformerModel::new_with_tensors_and_dense_linear_hooks(
        config,
        vb,
        Default::default(),
        hooks,
    )?;

    let boundary = model.dense_linear_runtime_hook_boundary("layers.0.attention.q_proj.weight");

    assert!(boundary.sidecar_payload_bytes_available);
    assert_eq!(boundary.sidecar_payload_bytes, Some(68));
    assert!(!boundary.sidecar_payload_contract_valid);
    assert!(!boundary.runtime_compute_enabled);
    assert!(boundary.preserves_eager_f32());
    Ok(())
}

#[test]
fn dense_linear_runtime_hook_boundaries_report_sorted_receipt_identity() -> anyhow::Result<()> {
    let config = tiny_config(8, 16, 2);
    let device = Device::Cpu;
    let vb = VarBuilder::zeros(DType::F32, &device);
    let mut hooks = DenseLinearRuntimeHookRegistry::default();
    hooks.insert(
        "layers.0.feed_forward.down_proj.weight".to_string(),
        DenseLinearRuntimeHookDescriptor {
            tensor_name: "blk.0.ffn_down.weight".to_string(),
            role: "MlpDown".to_string(),
            sidecar_payload_sha256: Some("sha256:down".to_string()),
            packed_q8_payload: None,
            runtime_compute_enabled: false,
        },
    );
    hooks.insert(
        "layers.0.attention.q_proj.weight".to_string(),
        DenseLinearRuntimeHookDescriptor {
            tensor_name: "blk.0.attn_q.weight".to_string(),
            role: "AttentionQ".to_string(),
            sidecar_payload_sha256: Some("sha256:q".to_string()),
            packed_q8_payload: None,
            runtime_compute_enabled: false,
        },
    );
    let model = TransformerModel::new_with_tensors_and_dense_linear_hooks(
        config,
        vb,
        Default::default(),
        hooks,
    )?;

    let boundaries = model.dense_linear_runtime_hook_boundaries();

    assert_eq!(boundaries.len(), 2);
    assert_eq!(boundaries[0].tensor_name, "layers.0.attention.q_proj.weight");
    assert_eq!(boundaries[1].tensor_name, "layers.0.feed_forward.down_proj.weight");
    assert!(boundaries.iter().all(DenseLinearRuntimeHookBoundary::preserves_eager_f32));
    assert!(boundaries.iter().all(|boundary| boundary.sidecar_descriptor_present));
    assert!(boundaries.iter().all(|boundary| !boundary.runtime_compute_enabled));
    Ok(())
}

#[test]
fn dense_linear_runtime_hook_boundary_does_not_enable_packed_compute() -> anyhow::Result<()> {
    let config = tiny_config(8, 16, 2);
    let device = Device::Cpu;
    let vb = VarBuilder::zeros(DType::F32, &device);
    let mut hooks = DenseLinearRuntimeHookRegistry::default();
    hooks.insert(
        "layers.0.attention.q_proj.weight".to_string(),
        DenseLinearRuntimeHookDescriptor {
            tensor_name: "blk.0.attn_q.weight".to_string(),
            role: "AttentionQ".to_string(),
            sidecar_payload_sha256: Some("sha256:future".to_string()),
            packed_q8_payload: None,
            runtime_compute_enabled: true,
        },
    );
    let model = TransformerModel::new_with_tensors_and_dense_linear_hooks(
        config,
        vb,
        Default::default(),
        hooks,
    )?;

    let boundary = model.dense_linear_runtime_hook_boundary("layers.0.attention.q_proj.weight");

    assert_eq!(boundary.selected_path, "eager_f32_candle");
    assert_eq!(boundary.selected_kernel, "dense-f32-candle-linear");
    assert!(boundary.sidecar_descriptor_present);
    assert!(!boundary.runtime_compute_enabled);
    assert!(!boundary.dense_runtime_replaced);
    assert!(boundary.preserves_eager_f32());
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

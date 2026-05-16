use bitnet_common::{ActivationType, BitNetConfig, BitNetError, NormType, Result};
pub use bitnet_qk256_dispatch::Qk256DispatchBackend;
use bitnet_qk256_dispatch::forward_qk256_scaled_with_backend;
use bitnet_rope::{build_tables as build_rope_tables, resolve_base as resolve_rope_base};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{LayerNorm, Linear, VarBuilder};

const DIAG_RMSNORM_F64_ACCUM_ENV: &str = "BITNET_DIAG_RMSNORM_F64_ACCUM";

#[cfg(feature = "trace")]
fn trace_tensor_record(
    name: &str,
    tensor: &Tensor,
    seq: usize,
    layer: Option<isize>,
    stage: &str,
) -> Result<()> {
    if trace_target_seq().is_some_and(|target| target != seq) {
        return Ok(());
    }
    bitnet_trace::dump_trace(name, tensor, Some(seq), layer, Some(stage)).map_err(BitNetError::from)
}

#[cfg(feature = "trace")]
fn trace_tensor_token_axis_record(
    suffix: &str,
    tensor: &Tensor,
    base_seq: usize,
    token_axis: usize,
    layer: Option<isize>,
    stage: &str,
) -> Result<()> {
    let target = trace_target_seq();
    if let Some(target_seq) = target {
        let Some(&seq_len) = tensor.dims().get(token_axis) else {
            return Ok(());
        };
        let Some(local_seq) = target_seq.checked_sub(base_seq) else {
            return Ok(());
        };
        if local_seq >= seq_len {
            return Ok(());
        }
        let sliced = tensor.narrow(token_axis, local_seq, 1)?;
        let name = format!("t{target_seq}/{suffix}");
        bitnet_trace::dump_trace(&name, &sliced, Some(target_seq), layer, Some(stage))
            .map_err(BitNetError::from)
    } else {
        let name = format!("t{base_seq}/{suffix}");
        bitnet_trace::dump_trace(&name, tensor, Some(base_seq), layer, Some(stage))
            .map_err(BitNetError::from)
    }
}

#[cfg(feature = "trace")]
fn trace_layer0_tensor(
    layer_idx: usize,
    base_seq: usize,
    token_axis: usize,
    stage: &str,
    tensor: &Tensor,
) -> Result<()> {
    if layer_idx == 0 {
        let suffix = format!("blk0/{stage}");
        trace_tensor_token_axis_record(&suffix, tensor, base_seq, token_axis, Some(0), stage)?;
    }
    Ok(())
}

#[cfg(feature = "trace")]
fn trace_target_seq() -> Option<usize> {
    std::env::var("BITNET_TRACE_TARGET_SEQ").ok()?.parse().ok()
}

/// Debug helper for tensor statistics (only runs if DEBUG_ATTN env var is set)
fn dbg_stats(tag: &str, t: &Tensor) -> candle_core::Result<()> {
    if std::env::var("DEBUG_ATTN").is_ok() {
        let mean = t.mean_all()?.to_scalar::<f32>()?;
        // Compute std manually: sqrt(E[(x - mean)^2])
        let diff = t.broadcast_sub(&t.mean_all()?)?;
        let variance = diff.sqr()?.mean_all()?;
        let std = variance.sqrt()?.to_scalar::<f32>()?;
        eprintln!("[dbg] {tag}: mean={mean:.6} std={std:.6}");
    }
    Ok(())
}

/// Debug helper for checking finite values
fn dbg_finite(tag: &str, t: &Tensor) -> candle_core::Result<()> {
    if std::env::var("DEBUG_ATTN").is_ok() {
        let v: Vec<f32> = t.flatten_all()?.to_vec1()?;
        let n = v.len().min(4096);
        let mut n_nan = 0;
        let mut n_inf = 0;
        for &x in &v[..n] {
            if !x.is_finite() {
                if x.is_nan() {
                    n_nan += 1;
                } else {
                    n_inf += 1;
                }
            }
        }
        if n_nan + n_inf > 0 {
            eprintln!(
                "⚠️  [dbg] {tag}: non-finite values: NaN={n_nan} Inf={n_inf} (in first {n} elems)"
            );
        }
    }
    Ok(())
}

fn attention_f16_dot_input(tensor: &Tensor) -> Result<Tensor> {
    Ok(tensor.to_dtype(DType::F16)?.to_dtype(DType::F32)?)
}

fn attention_score_key_input(tensor: &Tensor) -> Result<Tensor> {
    Ok(tensor.to_dtype(DType::F32)?)
}

fn qk256_scale_for(
    raw_tensors: &std::collections::HashMap<String, Tensor>,
    qk256_key: &str,
) -> Result<f32> {
    let scale_key = qk256_key
        .strip_suffix(".qk256_qs")
        .map(|base| format!("{base}.qk256_scale"))
        .unwrap_or_else(|| format!("{qk256_key}.qk256_scale"));

    let Some(scale_tensor) = raw_tensors.get(&scale_key) else {
        return Ok(1.0);
    };

    let values = scale_tensor.flatten_all()?.to_vec1::<f32>()?;
    values.first().copied().ok_or_else(|| {
        BitNetError::Validation(format!("QK256 scale tensor '{scale_key}' is empty"))
    })
}

/// Helper to create linear layers with optional bias tensors (zero-injection)
fn linear_with_optional_bias(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilder,
) -> candle_core::Result<Linear> {
    let weight = vb.get((out_dim, in_dim), "weight")?;

    // Try to get bias, create zeros if missing
    let bias = match vb.get(out_dim, "bias") {
        Ok(b) => Some(b),
        Err(_) => {
            tracing::debug!("Bias tensor missing for linear layer; injecting zeros [{}]", out_dim);
            Some(Tensor::zeros(out_dim, DType::F32, vb.device())?)
        }
    };

    Ok(Linear::new(weight, bias))
}

/// Helper to create layer norm with optional bias.
/// If `bias` is missing we use no-bias LayerNorm by default, or error when
/// `BITNET_REQUIRE_LAYER_NORM_BIAS=1`.
fn layer_norm_with_optional_bias(
    normalized_shape: usize,
    eps: f64,
    norm_type: NormType,
    vb: VarBuilder,
) -> candle_core::Result<LayerNorm> {
    let weight = vb.get((normalized_shape,), "weight")?;
    if norm_type == NormType::RmsNorm {
        return Ok(LayerNorm::rms_norm(weight, eps));
    }

    match vb.get((normalized_shape,), "bias") {
        Ok(bias) => {
            // Bias exists → standard LayerNorm (with mean subtraction and bias)
            tracing::debug!("Using LayerNorm with bias [{}]", normalized_shape);
            Ok(LayerNorm::new(weight, bias, eps))
        }
        Err(err) => {
            if std::env::var("BITNET_REQUIRE_LAYER_NORM_BIAS")
                .ok()
                .is_some_and(|value| value == "1")
            {
                return Err(candle_core::Error::Msg(format!(
                    "LayerNorm bias tensor is required but missing (set BITNET_REQUIRE_LAYER_NORM_BIAS=0 or unset to allow no-bias LayerNorm): {err}"
                )));
            }

            // No bias → LayerNorm without bias (but WITH mean subtraction)
            // IMPORTANT: Use LayerNorm::new_no_bias (remove_mean=true) NOT rms_norm (remove_mean=false)
            // because the gamma weights in GGUF are calibrated for LayerNorm semantics (mean subtraction).
            // bitnet.cpp uses full LayerNorm even when bias is absent.
            tracing::debug!(
                "Bias tensor missing for norm layer; using LayerNorm without bias (mean subtraction enabled) [{}]",
                normalized_shape
            );
            Ok(LayerNorm::new_no_bias(weight, eps))
        }
    }
}

fn optional_layer_norm_with_optional_bias(
    normalized_shape: usize,
    eps: f64,
    norm_type: NormType,
    vb: VarBuilder,
) -> candle_core::Result<Option<LayerNorm>> {
    let weight = match vb.get((normalized_shape,), "weight") {
        Ok(weight) => weight,
        Err(err) => {
            tracing::debug!("Optional norm weight missing; skipping [{}]: {err}", normalized_shape);
            return Ok(None);
        }
    };

    if norm_type == NormType::RmsNorm {
        return Ok(Some(LayerNorm::rms_norm(weight, eps)));
    }

    match vb.get((normalized_shape,), "bias") {
        Ok(bias) => Ok(Some(LayerNorm::new(weight, bias, eps))),
        Err(_) => Ok(Some(LayerNorm::new_no_bias(weight, eps))),
    }
}

fn norm_forward(norm: &LayerNorm, x: &Tensor, eps: f64, norm_type: NormType) -> Result<Tensor> {
    if norm_type == NormType::RmsNorm && diag_rmsnorm_f64_accum_enabled() {
        return rmsnorm_f64_accum_forward(x, norm.weight(), eps);
    }
    norm.forward(x).map_err(BitNetError::from)
}

fn diag_rmsnorm_f64_accum_enabled() -> bool {
    std::env::var(DIAG_RMSNORM_F64_ACCUM_ENV).as_deref() == Ok("1")
}

fn rmsnorm_f64_accum_forward(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let hidden = dims.last().copied().ok_or_else(|| {
        BitNetError::Validation(
            "RMSNorm f64 diagnostic input must have at least one dimension".into(),
        )
    })?;
    if hidden == 0 {
        return Err(BitNetError::Validation(
            "RMSNorm f64 diagnostic input has zero hidden dimension".into(),
        ));
    }
    let values = x.flatten_all()?.to_vec1::<f32>()?;
    if values.len() % hidden != 0 {
        return Err(BitNetError::Validation(format!(
            "RMSNorm f64 diagnostic input has {} values not divisible by hidden {hidden}",
            values.len()
        )));
    }
    let gamma = weight.flatten_all()?.to_vec1::<f32>()?;
    if gamma.len() != hidden {
        return Err(BitNetError::Validation(format!(
            "RMSNorm f64 diagnostic weight has {} values, expected hidden {hidden}",
            gamma.len()
        )));
    }

    let mut output = Vec::with_capacity(values.len());
    for row in values.chunks(hidden) {
        let sum_sq = row.iter().map(|&value| (value as f64) * (value as f64)).sum::<f64>();
        let denom = ((sum_sq / hidden as f64) + eps).sqrt() as f32;
        output.extend(row.iter().zip(&gamma).map(|(&x, &g)| (x / denom) * g));
    }
    Tensor::from_vec(output, dims.as_slice(), x.device()).map_err(BitNetError::from)
}

fn feed_forward_activation(
    activation_type: ActivationType,
    gate: &Tensor,
) -> candle_core::Result<Tensor> {
    match activation_type {
        ActivationType::Silu => candle_nn::ops::silu(gate),
        ActivationType::Relu2 => gate.relu()?.sqr(),
        ActivationType::Gelu => gate.gelu_erf(),
    }
}

/// Rotary Position Embedding
pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    pub fn new(
        dim: usize,
        max_seq_len: usize,
        rope_theta: Option<f32>,
        device: &Device,
    ) -> Result<Self> {
        let theta = resolve_rope_base(rope_theta);
        let tables = build_rope_tables(dim, max_seq_len, theta)
            .map_err(|err| BitNetError::Validation(format!("invalid RoPE configuration: {err}")))?;
        let bitnet_rope::RopeTables { half_dim, sin, cos } = tables;

        let sin = Tensor::from_vec(sin, &[max_seq_len, half_dim], device)?;
        let cos = Tensor::from_vec(cos, &[max_seq_len, half_dim], device)?;

        // Log ROPE initialization parameters
        tracing::info!(
            "ROPE initialized: base={}, rope_dims={}, max_seq_len={}",
            theta,
            dim,
            max_seq_len
        );

        Ok(Self { sin, cos })
    }

    pub fn apply(&self, x: &Tensor, position: usize) -> Result<Tensor> {
        // x shape: [B, H, T, D] for multi-head attention
        if x.dims().len() == 4 {
            let (batch, n_heads, seq_len, head_dim) = x.dims4()?;
            let half_dim = head_dim / 2;

            // LLaMA RoPE uses SPLIT layout: [r0,r1,...,r_{d/2-1}, i0,i1,...,i_{d/2-1}]
            // NOT interleaved [r0,i0,r1,i1,...]
            let x0 = x.narrow(3, 0, half_dim)?; // First half (real)
            let x1 = x.narrow(3, half_dim, half_dim)?; // Second half (imaginary)

            // Get cos/sin for the position
            let cos = self.cos.narrow(0, position, seq_len)?
                .unsqueeze(0)?  // Add batch dim
                .unsqueeze(1)?  // Add heads dim
                .broadcast_as(&[batch, n_heads, seq_len, half_dim])?;
            let sin = self
                .sin
                .narrow(0, position, seq_len)?
                .unsqueeze(0)?
                .unsqueeze(1)?
                .broadcast_as(&[batch, n_heads, seq_len, half_dim])?;

            let x0_rot = (x0.mul(&cos)? - x1.mul(&sin)?)?;
            let x1_rot = (x0.mul(&sin)? + x1.mul(&cos)?)?;

            // Concatenate back in split layout [real, imag]
            let rotated = Tensor::cat(&[x0_rot, x1_rot], 3)?;

            Ok(rotated)
        } else {
            // Original 3D implementation for other uses
            let (_batch, _seq, dim) = x.dims3()?;
            let half_dim = dim / 2;

            // LLaMA RoPE uses SPLIT layout: [r0,r1,...,i0,i1,...]
            let x0 = x.narrow(2, 0, half_dim)?; // First half (real)
            let x1 = x.narrow(2, half_dim, half_dim)?; // Second half (imaginary)

            let cos = self.cos.narrow(0, position, 1)?;
            let sin = self.sin.narrow(0, position, 1)?;

            let x0_rot = (x0.mul(&cos)? - x1.mul(&sin)?)?;
            let x1_rot = (x0.mul(&sin)? + x1.mul(&cos)?)?;

            // Concatenate back in split layout [real, imag]
            let rotated = Tensor::cat(&[x0_rot, x1_rot], 2)?;

            Ok(rotated)
        }
    }
}

/// Multi-Head Attention Layer
pub struct MultiHeadAttention {
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    group_size: usize, // n_heads / n_kv_heads
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    sub_layernorm: Option<LayerNorm>,
    norm_type: NormType,
    norm_eps: f64,
    rope: Option<RotaryEmbedding>,
    layer_idx: usize, // Layer index for QK256 weight name generation
    qk256_backend: Qk256DispatchBackend,
}

impl MultiHeadAttention {
    pub fn new(
        config: &BitNetConfig,
        vb: VarBuilder,
        layer_idx: usize,
        qk256_backend: Qk256DispatchBackend,
    ) -> Result<Self> {
        let hidden_size = config.model.hidden_size;
        let n_heads = config.model.num_heads;
        let head_dim = hidden_size / n_heads;
        let eps = config.model.rms_norm_eps.map(|e| e as f64).unwrap_or(1e-5);

        if !hidden_size.is_multiple_of(n_heads) {
            return Err(BitNetError::Validation(format!(
                "hidden_size {} not divisible by num_heads {}",
                hidden_size, n_heads
            )));
        }

        let n_kv_heads = config.model.num_key_value_heads.max(1).min(n_heads);
        if !n_heads.is_multiple_of(n_kv_heads) {
            return Err(BitNetError::Validation(format!(
                "num_heads {} must be divisible by num_key_value_heads {}",
                n_heads, n_kv_heads
            )));
        }
        let group_size = n_heads / n_kv_heads;
        let kv_out = n_kv_heads * head_dim;

        tracing::info!(
            "layer{}: MultiHeadAttention dims: hidden={}, n_heads={}, n_kv_heads={}, head_dim={}, kv_out={}, group_size={}",
            layer_idx,
            hidden_size,
            n_heads,
            n_kv_heads,
            head_dim,
            kv_out,
            group_size
        );

        tracing::info!(
            "layer{}: About to create linear layers with: q_proj([{}, {}]), k_proj([{}, {}]), v_proj([{}, {}]), o_proj([{}, {}])",
            layer_idx,
            hidden_size,
            hidden_size,
            kv_out,
            hidden_size,
            kv_out,
            hidden_size,
            hidden_size,
            hidden_size
        );

        let q_proj = linear_with_optional_bias(hidden_size, hidden_size, vb.pp("q_proj"))?;
        let k_proj = linear_with_optional_bias(hidden_size, kv_out, vb.pp("k_proj"))?;
        let v_proj = linear_with_optional_bias(hidden_size, kv_out, vb.pp("v_proj"))?;
        let o_proj = linear_with_optional_bias(hidden_size, hidden_size, vb.pp("o_proj"))?;
        let sub_layernorm = optional_layer_norm_with_optional_bias(
            hidden_size,
            eps,
            config.model.norm_type,
            vb.pp("sub_layernorm"),
        )?;

        let rope = RotaryEmbedding::new(
            head_dim,
            config.model.max_position_embeddings,
            config.model.rope_theta,
            vb.device(),
        )
        .ok();

        Ok(Self {
            n_heads,
            n_kv_heads,
            head_dim,
            group_size,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            sub_layernorm,
            norm_type: config.model.norm_type,
            norm_eps: eps,
            rope,
            layer_idx,
            qk256_backend,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        kv_cache: Option<&mut LayerKVCache>,
        raw_tensors: &std::collections::HashMap<String, Tensor>,
        _trace_base_seq: usize,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;

        // PATCH 3: Project to Q, K, V separately (NOT fused QKV)
        // This is the correct implementation - separate projections ensure proper shape handling
        // Q: [B, T, hidden] -> [B, T, n_heads * head_dim] -> [B, n_heads, T, head_dim]
        // K: [B, T, hidden] -> [B, T, n_kv_heads * head_dim] -> [B, n_kv_heads, T, head_dim]
        // V: [B, T, hidden] -> [B, T, n_kv_heads * head_dim] -> [B, n_kv_heads, T, head_dim]
        let q_proj_out = self.apply_linear(x, &self.q_proj, "q_proj", raw_tensors)?;
        let k_proj_out = self.apply_linear(x, &self.k_proj, "k_proj", raw_tensors)?;
        let v_proj_out = self.apply_linear(x, &self.v_proj, "v_proj", raw_tensors)?;

        #[cfg(feature = "trace")]
        {
            trace_layer0_tensor(self.layer_idx, _trace_base_seq, 1, "attention_q", &q_proj_out)?;
            trace_layer0_tensor(self.layer_idx, _trace_base_seq, 1, "attention_k", &k_proj_out)?;
            trace_layer0_tensor(self.layer_idx, _trace_base_seq, 1, "attention_v", &v_proj_out)?;
        }

        // Probe A3: Q/K/V projection RMS (layer 0, step 0 only)
        if std::env::var("BITNET_TRACE_RMS").as_deref() == Ok("1") && self.layer_idx == 0 {
            static PROJ_LOGGED: std::sync::Once = std::sync::Once::new();
            PROJ_LOGGED.call_once(|| {
                let _ = (|| -> candle_core::Result<()> {
                    let q_vec = q_proj_out.flatten_all()?.to_vec1::<f32>()?;
                    let q_rms = (q_vec.iter().map(|x| x * x).sum::<f32>()
                        / q_vec.len().max(1) as f32)
                        .sqrt();
                    let k_vec = k_proj_out.flatten_all()?.to_vec1::<f32>()?;
                    let k_rms = (k_vec.iter().map(|x| x * x).sum::<f32>()
                        / k_vec.len().max(1) as f32)
                        .sqrt();
                    let v_vec = v_proj_out.flatten_all()?.to_vec1::<f32>()?;
                    let v_rms = (v_vec.iter().map(|x| x * x).sum::<f32>()
                        / v_vec.len().max(1) as f32)
                        .sqrt();
                    eprintln!(
                        "trace: q_proj_rms={:.6} k_proj_rms={:.6} v_proj_rms={:.6}",
                        q_rms, k_rms, v_rms
                    );
                    Ok(())
                })();
            });
        }

        // Tracepoint 3: Q projection output (layer-specific)
        #[cfg(feature = "trace")]
        {
            let trace_suffix = format!("blk{}/q_proj", self.layer_idx);
            trace_tensor_token_axis_record(
                &trace_suffix,
                &q_proj_out,
                _trace_base_seq,
                1,
                Some(self.layer_idx as isize),
                "q_proj",
            )?;
        }

        let q = q_proj_out
            .reshape(&[batch_size, seq_len, self.n_heads, self.head_dim])?
            .transpose(1, 2)?; // [B, Hq, T, D]

        let k = k_proj_out
            .reshape(&[batch_size, seq_len, self.n_kv_heads, self.head_dim])?
            .transpose(1, 2)?; // [B, HKV, T, D]

        let v = v_proj_out
            .reshape(&[batch_size, seq_len, self.n_kv_heads, self.head_dim])?
            .transpose(1, 2)?; // [B, HKV, T, D]

        // Debug Q, K, V projections
        dbg_stats("Q", &q)?;
        dbg_stats("K", &k)?;
        dbg_stats("V", &v)?;

        // GQA diagnostic: log Q/K/V dimensions and norms (once per run)
        if std::env::var("BITNET_DEBUG_GQA").is_ok() {
            static GQA_LOGGED: std::sync::Once = std::sync::Once::new();
            GQA_LOGGED.call_once(|| {
                let q_dims = q.dims();
                let k_dims = k.dims();
                let v_dims = v.dims();
                if let (Ok(q_mean), Ok(k_mean), Ok(v_mean)) = (
                    q.mean_all().and_then(|m| m.to_scalar::<f32>()),
                    k.mean_all().and_then(|m| m.to_scalar::<f32>()),
                    v.mean_all().and_then(|m| m.to_scalar::<f32>()),
                ) {
                    tracing::info!(
                        "GQA shapes - Q: {:?} (mean {:.3}), K: {:?} (mean {:.3}), V: {:?} (mean {:.3})",
                        q_dims, q_mean, k_dims, k_mean, v_dims, v_mean
                    );
                    tracing::info!(
                        "GQA config - n_heads={}, n_kv_heads={}, head_dim={}, group_size={}",
                        self.n_heads, self.n_kv_heads, self.head_dim, self.group_size
                    );
                }
            });
        }

        // Apply rotary embeddings if available (need to handle different K/V head counts)
        let (q, k) = if let Some(rope) = &self.rope {
            let position = kv_cache.as_ref().map(|c| c.seq_len).unwrap_or(0);

            // Log ROPE application details (once)
            if std::env::var("BITNET_DEBUG_ROPE").is_ok() {
                static ROPE_LOGGED: std::sync::Once = std::sync::Once::new();
                ROPE_LOGGED.call_once(|| {
                    tracing::info!(
                        "ROPE applied: position={}, q_shape={:?}, k_shape={:?}, head_dim={}",
                        position,
                        q.dims(),
                        k.dims(),
                        self.head_dim
                    );
                });
            }

            let q_rot = rope.apply(&q, position)?;
            let k_rot = rope.apply(&k, position)?;
            (q_rot, k_rot)
        } else {
            (q, k)
        };

        #[cfg(feature = "trace")]
        if self.layer_idx == 0 {
            trace_layer0_tensor(self.layer_idx, _trace_base_seq, 2, "attention_q_rope", &q)?;
        }

        // Update KV cache if provided (store HKV heads, not Hq)
        // **Performance note**: Borrow references instead of cloning after append.
        // Candle operations accept both owned and borrowed tensors.
        #[cfg(feature = "trace")]
        if self.layer_idx == 0 {
            for kv_head_idx in 0..self.n_kv_heads {
                let before_store = k
                    .narrow(1, kv_head_idx, 1)?
                    .reshape(&[seq_len, self.head_dim])?
                    .transpose(0, 1)?
                    .to_dtype(DType::F32)?;
                let trace_seq = trace_target_seq().unwrap_or(_trace_base_seq);
                let trace_name = format!(
                    "t{trace_seq}/blk0/attention_k_before_cache_store_kv_head{kv_head_idx}_ref_layout"
                );
                let stage =
                    format!("attention_k_before_cache_store_kv_head{kv_head_idx}_ref_layout");
                trace_tensor_record(&trace_name, &before_store, trace_seq, Some(0), &stage)?;
            }
            for kv_head_idx in 0..self.n_kv_heads {
                let before_store = v
                    .narrow(1, kv_head_idx, 1)?
                    .reshape(&[seq_len, self.head_dim])?
                    .transpose(0, 1)?
                    .to_dtype(DType::F32)?;
                let trace_seq = trace_target_seq().unwrap_or(_trace_base_seq);
                let trace_name = format!(
                    "t{trace_seq}/blk0/attention_v_before_cache_store_kv_head{kv_head_idx}_ref_layout"
                );
                let stage =
                    format!("attention_v_before_cache_store_kv_head{kv_head_idx}_ref_layout");
                trace_tensor_record(&trace_name, &before_store, trace_seq, Some(0), &stage)?;
            }
        }
        let (k_ctx, v_ctx) = if let Some(cache) = kv_cache {
            cache.append(&k, &v)?;
            // Borrow from cache - avoids cloning full KV history
            (&cache.k, &cache.v)
        } else {
            // No cache: use freshly computed K/V from this step
            (&k, &v)
        };

        // GQA core: expand K/V to Hq heads (repeat along head axis)
        // We want K,V of shape [B,Hq,Tk,D]. Repeat every KV head group_size times.
        let t_k = k_ctx.dims()[2];

        // Expand K: [B, HKV, Tk, D] -> [B, Hq, Tk, D]
        let k_expanded = k_ctx
            .unsqueeze(2)?                               // [B, HKV, 1, Tk, D]
            .repeat(&[1, 1, self.group_size, 1, 1])?    // [B, HKV, group, Tk, D]
            .reshape(&[batch_size, self.n_heads, t_k, self.head_dim])?; // [B, Hq, Tk, D]

        // Expand V: [B, HKV, Tk, D] -> [B, Hq, Tk, D]
        let v_expanded = v_ctx
            .unsqueeze(2)?                               // [B, HKV, 1, Tk, D]
            .repeat(&[1, 1, self.group_size, 1, 1])?    // [B, HKV, group, Tk, D]
            .reshape(&[batch_size, self.n_heads, t_k, self.head_dim])?; // [B, Hq, Tk, D]
        #[cfg(feature = "trace")]
        if self.layer_idx == 0 {
            for kv_head_idx in 0..self.n_kv_heads {
                let kv_live = k_ctx
                    .narrow(1, kv_head_idx, 1)?
                    .reshape(&[t_k, self.head_dim])?
                    .transpose(0, 1)?
                    .to_dtype(DType::F32)?;
                let trace_seq = trace_target_seq().unwrap_or(_trace_base_seq);
                let trace_name = format!(
                    "t{trace_seq}/blk0/attention_k_cache_kv_head{kv_head_idx}_live_ref_layout"
                );
                let stage = format!("attention_k_cache_kv_head{kv_head_idx}_live_ref_layout");
                trace_tensor_record(&trace_name, &kv_live, trace_seq, Some(0), &stage)?;

                let kv_live_f16_roundtrip = kv_live.to_dtype(DType::F16)?.to_dtype(DType::F32)?;
                let trace_name = format!(
                    "t{trace_seq}/blk0/attention_k_cache_f16_roundtrip_kv_head{kv_head_idx}_live_ref_layout"
                );
                let stage =
                    format!("attention_k_cache_f16_roundtrip_kv_head{kv_head_idx}_live_ref_layout");
                trace_tensor_record(
                    &trace_name,
                    &kv_live_f16_roundtrip,
                    trace_seq,
                    Some(0),
                    &stage,
                )?;
            }
            let head0_ref_layout =
                k_expanded.narrow(1, 0, 1)?.reshape(&[t_k, self.head_dim])?.transpose(0, 1)?;
            let reference_padded_tk = t_k.next_power_of_two().max(t_k);
            let head0_ref_layout = if reference_padded_tk > t_k {
                let pad = Tensor::zeros(
                    &[self.head_dim, reference_padded_tk - t_k],
                    DType::F32,
                    k_expanded.device(),
                )?;
                Tensor::cat(&[&head0_ref_layout.to_dtype(DType::F32)?, &pad], 1)?
            } else {
                head0_ref_layout.to_dtype(DType::F32)?
            };
            let trace_seq = trace_target_seq().unwrap_or(_trace_base_seq);
            let trace_name = format!("t{trace_seq}/blk0/attention_k_cache_head0_ref_layout_padded");
            trace_tensor_record(
                &trace_name,
                &head0_ref_layout,
                trace_seq,
                Some(0),
                "attention_k_cache_head0_ref_layout_padded",
            )?;

            for kv_head_idx in 0..self.n_kv_heads {
                let kv_live = v_ctx
                    .narrow(1, kv_head_idx, 1)?
                    .reshape(&[t_k, self.head_dim])?
                    .transpose(0, 1)?
                    .to_dtype(DType::F32)?;
                let trace_seq = trace_target_seq().unwrap_or(_trace_base_seq);
                let trace_name = format!(
                    "t{trace_seq}/blk0/attention_v_cache_readback_kv_head{kv_head_idx}_ref_layout"
                );
                let stage = format!("attention_v_cache_readback_kv_head{kv_head_idx}_ref_layout");
                trace_tensor_record(&trace_name, &kv_live, trace_seq, Some(0), &stage)?;

                let kv_live_f16_roundtrip = kv_live.to_dtype(DType::F16)?.to_dtype(DType::F32)?;
                let trace_name = format!(
                    "t{trace_seq}/blk0/attention_v_cache_stored_f16_kv_head{kv_head_idx}_ref_layout"
                );
                let stage = format!("attention_v_cache_stored_f16_kv_head{kv_head_idx}_ref_layout");
                trace_tensor_record(
                    &trace_name,
                    &kv_live_f16_roundtrip,
                    trace_seq,
                    Some(0),
                    &stage,
                )?;

                let trace_name = format!(
                    "t{trace_seq}/blk0/attention_v_cache_kv_head{kv_head_idx}_live_ref_layout"
                );
                let stage = format!("attention_v_cache_kv_head{kv_head_idx}_live_ref_layout");
                trace_tensor_record(&trace_name, &kv_live, trace_seq, Some(0), &stage)?;

                let trace_name = format!(
                    "t{trace_seq}/blk0/attention_v_cache_f16_roundtrip_kv_head{kv_head_idx}_live_ref_layout"
                );
                let stage =
                    format!("attention_v_cache_f16_roundtrip_kv_head{kv_head_idx}_live_ref_layout");
                trace_tensor_record(
                    &trace_name,
                    &kv_live_f16_roundtrip,
                    trace_seq,
                    Some(0),
                    &stage,
                )?;
            }
            let head0_ref_layout =
                v_expanded.narrow(1, 0, 1)?.reshape(&[t_k, self.head_dim])?.transpose(0, 1)?;
            let reference_padded_tk = t_k.next_power_of_two().max(t_k);
            let head0_ref_layout = if reference_padded_tk > t_k {
                let pad = Tensor::zeros(
                    &[self.head_dim, reference_padded_tk - t_k],
                    DType::F32,
                    v_expanded.device(),
                )?;
                Tensor::cat(&[&head0_ref_layout.to_dtype(DType::F32)?, &pad], 1)?
            } else {
                head0_ref_layout.to_dtype(DType::F32)?
            };
            let trace_seq = trace_target_seq().unwrap_or(_trace_base_seq);
            let trace_name = format!("t{trace_seq}/blk0/attention_v_cache_head0_ref_layout_padded");
            trace_tensor_record(
                &trace_name,
                &head0_ref_layout,
                trace_seq,
                Some(0),
                "attention_v_cache_head0_ref_layout_padded",
            )?;
        }

        // Scaled dot-product attention with explicit fp32 handling
        // For head_dim=128, scale = 1/sqrt(128) ≈ 0.0883883
        let scale_factor = (self.head_dim as f32).sqrt().recip();

        // Log scale computation once
        if std::env::var("BITNET_DEBUG_ATTN_SCALE").is_ok() {
            static SCALE_LOGGED: std::sync::Once = std::sync::Once::new();
            SCALE_LOGGED.call_once(|| {
                tracing::info!(
                    "Attention scale: head_dim={}, scale_factor=1/sqrt({})={:.7}",
                    self.head_dim,
                    self.head_dim,
                    scale_factor
                );
            });
        }

        let q_for_scores = attention_f16_dot_input(&q)?;
        let k_for_scores = attention_score_key_input(&k_expanded)?;
        #[cfg(feature = "trace")]
        if self.layer_idx == 0 {
            trace_layer0_tensor(
                self.layer_idx,
                _trace_base_seq,
                2,
                "attention_q_rope_f16_roundtrip",
                &q_for_scores,
            )?;
            for head_idx in 0..self.n_heads {
                let key_head = k_for_scores
                    .narrow(1, head_idx, 1)?
                    .reshape(&[t_k, self.head_dim])?
                    .transpose(0, 1)?
                    .to_dtype(DType::F32)?;
                let trace_seq = trace_target_seq().unwrap_or(_trace_base_seq);
                let trace_name = format!(
                    "t{trace_seq}/blk0/attention_k_score_input_head{head_idx}_live_ref_layout"
                );
                let stage = format!("attention_k_score_input_head{head_idx}_live_ref_layout");
                trace_tensor_record(&trace_name, &key_head, trace_seq, Some(0), &stage)?;
            }
        }
        let scores = q_for_scores.matmul(&k_for_scores.transpose(2, 3)?)?;

        // Convert to fp32 for numerically stable computation
        let scores_f32 = scores.to_dtype(DType::F32)?;
        #[cfg(feature = "trace")]
        if self.layer_idx == 0 {
            for head_idx in 0..self.n_heads {
                let head = scores_f32.narrow(1, head_idx, 1)?;
                let suffix = format!("blk0/attention_scores_raw_head{head_idx}");
                let stage = format!("attention_scores_raw_head{head_idx}");
                trace_tensor_token_axis_record(
                    &suffix,
                    &head,
                    _trace_base_seq,
                    2,
                    Some(0),
                    &stage,
                )?;
            }
        }

        // Scale in fp32
        let scores_f32 = scores_f32.affine(scale_factor as f64, 0.0)?;

        // Debug scores before mask
        dbg_stats("scores pre-mask", &scores_f32)?;
        dbg_finite("scores pre-mask", &scores_f32)?;

        // Apply causal mask so queries cannot attend to future positions.
        // When using a KV cache, k includes past tokens, so the mask must
        // account for the total key length.
        let total_len = k_expanded.dims()[2];
        // PATCH 5: create_causal_mask now returns [1, 1, Tq, Tk] directly - no need for unsqueeze
        let mask = self.create_causal_mask(seq_len, total_len, scores_f32.device())?;
        let scores_f32 = scores_f32.broadcast_add(&mask)?;

        // Debug scores after mask and before softmax (critical diagnostics)
        dbg_stats("scores post-mask", &scores_f32)?;
        dbg_finite("scores post-mask", &scores_f32)?;

        // Log scores range after mask for layer 0 (user's diagnostic request)
        if std::env::var("BITNET_DEBUG_ATTN_SCALE").is_ok() {
            static LAYER_LOGGED: std::sync::Once = std::sync::Once::new();
            LAYER_LOGGED.call_once(|| {
                if let Ok(flat) = scores_f32.flatten_all()
                    && let Ok(vals) = flat.to_vec1::<f32>()
                    && let (Some(&min_val), Some(&max_val)) = (
                        vals.iter()
                            .filter(|v| v.is_finite())
                            .min_by(|a, b| a.partial_cmp(b).unwrap()),
                        vals.iter()
                            .filter(|v| v.is_finite())
                            .max_by(|a, b| a.partial_cmp(b).unwrap()),
                    )
                {
                    tracing::info!(
                        "Layer 0 scores post-mask range: min={:.6}, max={:.6}",
                        min_val,
                        max_val
                    );
                }
            });
        }

        // PATCH 4: Softmax path verification
        // Apply max-subtraction for numerical stability before softmax
        // Compute row-wise max and subtract for stability (explicit max-subtraction)
        // VERIFIED: axis=3 is correct for [B, H, Tq, Tk] layout - normalizes across keys (Tk)
        let row_max = scores_f32.max_keepdim(3)?;
        let scores_stabilized = scores_f32.broadcast_sub(&row_max)?;

        // Log that max-subtraction ran (user's diagnostic request)
        if std::env::var("BITNET_DEBUG_ATTN_SCALE").is_ok() {
            static MAX_SUB_LOGGED: std::sync::Once = std::sync::Once::new();
            MAX_SUB_LOGGED.call_once(|| {
                tracing::info!("Attention: max-subtraction applied for numerical stability");
            });
        }

        // Apply softmax (exp then normalize)
        // VERIFIED: axis=3 is correct - softmax over keys (Tk dimension) in [B, H, Tq, Tk]
        let attn_weights = candle_nn::ops::softmax(&scores_stabilized, 3)?;
        #[cfg(feature = "trace")]
        if self.layer_idx == 0 {
            for head_idx in 0..self.n_heads {
                let head = attn_weights.narrow(1, head_idx, 1)?;
                let suffix = format!("blk0/attn_scores_softmax_head{head_idx}");
                let stage = format!("attn_scores_softmax_head{head_idx}");
                trace_tensor_token_axis_record(
                    &suffix,
                    &head,
                    _trace_base_seq,
                    2,
                    Some(0),
                    &stage,
                )?;
            }
        }

        // Tracepoint 4: Attention scores post-softmax (layer-specific)
        #[cfg(feature = "trace")]
        {
            let trace_suffix = format!("blk{}/attn_scores_softmax", self.layer_idx);
            trace_tensor_token_axis_record(
                &trace_suffix,
                &attn_weights,
                _trace_base_seq,
                2,
                Some(self.layer_idx as isize),
                "attn_scores_softmax",
            )?;
        }

        // Debug attention weights and row sums
        dbg_stats("attn softmax", &attn_weights)?;
        if std::env::var("DEBUG_ATTN").is_ok() {
            let sums = attn_weights.sum(3)?;
            let sums_host: Vec<f32> = sums.flatten_all()?.to_vec1()?;
            let take = sums_host.iter().take(4).cloned().collect::<Vec<_>>();
            eprintln!("[dbg] attn row-sums (first 4): {:?}", take);
        }

        let v_for_value_mix = attention_f16_dot_input(&v_expanded)?;
        #[cfg(feature = "trace")]
        if self.layer_idx == 0 {
            for head_idx in 0..self.n_heads {
                let head = v_for_value_mix
                    .narrow(1, head_idx, 1)?
                    .reshape(&[t_k, self.head_dim])?
                    .transpose(0, 1)?
                    .to_dtype(DType::F32)?;
                let trace_seq = trace_target_seq().unwrap_or(_trace_base_seq);
                let trace_name = format!(
                    "t{trace_seq}/blk0/attention_v_cache_expanded_for_value_mix_head{head_idx}_ref_layout"
                );
                let stage =
                    format!("attention_v_cache_expanded_for_value_mix_head{head_idx}_ref_layout");
                trace_tensor_record(&trace_name, &head, trace_seq, Some(0), &stage)?;
            }
        }
        let attn_value_mix = attn_weights.matmul(&v_for_value_mix)?;
        #[cfg(feature = "trace")]
        if self.layer_idx == 0 {
            for head_idx in 0..self.n_heads {
                let head = attn_value_mix.narrow(1, head_idx, 1)?;
                let suffix = format!("blk0/attention_value_mix_f16_cache_head{head_idx}");
                let stage = format!("attention_value_mix_f16_cache_head{head_idx}");
                trace_tensor_token_axis_record(
                    &suffix,
                    &head,
                    _trace_base_seq,
                    2,
                    Some(0),
                    &stage,
                )?;
            }
            trace_layer0_tensor(
                self.layer_idx,
                _trace_base_seq,
                2,
                "attention_value_mix_f16_cache",
                &attn_value_mix,
            )?;
            let attn_output_f16_cache = attn_value_mix.transpose(1, 2)?.reshape(&[
                batch_size,
                seq_len,
                self.n_heads * self.head_dim,
            ])?;
            trace_layer0_tensor(
                self.layer_idx,
                _trace_base_seq,
                1,
                "attention_value_mix_f16_cache_merged",
                &attn_output_f16_cache,
            )?;
        }
        #[cfg(feature = "trace")]
        if self.layer_idx == 0 {
            for head_idx in 0..self.n_heads {
                let head = attn_value_mix.narrow(1, head_idx, 1)?;
                let suffix = format!("blk0/attention_value_mix_head{head_idx}");
                let stage = format!("attention_value_mix_head{head_idx}");
                trace_tensor_token_axis_record(
                    &suffix,
                    &head,
                    _trace_base_seq,
                    2,
                    Some(0),
                    &stage,
                )?;
            }
        }
        #[cfg(feature = "trace")]
        trace_layer0_tensor(
            self.layer_idx,
            _trace_base_seq,
            2,
            "attention_value_mix",
            &attn_value_mix,
        )?;

        // Reshape and project output
        let attn_output = attn_value_mix.transpose(1, 2)?.reshape(&[
            batch_size,
            seq_len,
            self.n_heads * self.head_dim,
        ])?;
        #[cfg(feature = "trace")]
        trace_layer0_tensor(
            self.layer_idx,
            _trace_base_seq,
            1,
            "attention_value_mix_merged",
            &attn_output,
        )?;
        let attn_output = match &self.sub_layernorm {
            Some(norm) => norm_forward(norm, &attn_output, self.norm_eps, self.norm_type)?,
            None => attn_output,
        };
        #[cfg(feature = "trace")]
        trace_layer0_tensor(
            self.layer_idx,
            _trace_base_seq,
            1,
            "post_attention_subnorm",
            &attn_output,
        )?;

        let output = self.apply_linear(&attn_output, &self.o_proj, "o_proj", raw_tensors)?;
        #[cfg(feature = "trace")]
        trace_layer0_tensor(self.layer_idx, _trace_base_seq, 1, "post_o_proj", &output)?;
        Ok(output)
    }

    /// Apply linear transformation with QK256 dispatch
    fn apply_linear(
        &self,
        input: &Tensor,
        linear: &Linear,
        proj_name: &str,
        raw_tensors: &std::collections::HashMap<String, Tensor>,
    ) -> Result<Tensor> {
        // Generate weight name based on layer index and projection name
        // Format: "layers.{idx}.attention.{proj_name}.weight.qk256_qs"
        let qk256_key =
            format!("layers.{}.attention.{}.weight.qk256_qs", self.layer_idx, proj_name);

        // Check for QK256 data
        if let Some(qk256_tensor) = raw_tensors.get(&qk256_key) {
            let qk256_scale = qk256_scale_for(raw_tensors, &qk256_key)?;
            tracing::debug!("Using QK256 kernel for {} with scale {}", qk256_key, qk256_scale);
            return forward_qk256_scaled_with_backend(
                input,
                qk256_tensor,
                &qk256_key,
                self.qk256_backend,
                qk256_scale,
            );
        }

        // Probe: Why is QK256 not found? (layer 0 only, once)
        if std::env::var("BITNET_TRACE_RMS").as_deref() == Ok("1") && self.layer_idx == 0 {
            static FALLBACK_LOGGED: std::sync::Once = std::sync::Once::new();
            FALLBACK_LOGGED.call_once(|| {
                eprintln!(
                    "trace_fallback: QK256 key '{}' not found in raw_tensors ({}keys total)",
                    qk256_key,
                    raw_tensors.len()
                );
                // Show first few keys for debugging
                let sample_keys: Vec<_> = raw_tensors.keys().take(5).collect();
                eprintln!("trace_fallback: Sample keys: {:?}", sample_keys);
            });
        }

        // Fall back to standard linear
        tracing::trace!(
            "Using standard linear for layers.{}.attention.{}",
            self.layer_idx,
            proj_name
        );
        linear.forward(input).map_err(BitNetError::from)
    }

    /// PATCH 5: Create causal mask with [1, 1, Tq, Tk] shape
    fn create_causal_mask(&self, q_len: usize, k_len: usize, device: &Device) -> Result<Tensor> {
        // Past tokens are stored in the KV cache and increase k_len.
        // For each query position i, disallow attention to key positions
        // greater than past_len + i.
        let past_len = k_len.saturating_sub(q_len);
        let mut mask_vec = vec![0.0f32; q_len * k_len];
        for i in 0..q_len {
            let start = past_len + i + 1;
            for j in start..k_len {
                mask_vec[i * k_len + j] = f32::NEG_INFINITY;
            }
        }
        // Create [1, 1, q_len, k_len] shape directly for broadcast compatibility
        Tensor::from_vec(mask_vec, &[1, 1, q_len, k_len], device).map_err(BitNetError::from)
    }
}

/// Feed-Forward Network
pub struct FeedForward {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    sub_layernorm: Option<LayerNorm>,
    norm_type: NormType,
    norm_eps: f64,
    activation_type: ActivationType,
    layer_idx: usize, // Layer index for QK256 weight name generation
    qk256_backend: Qk256DispatchBackend,
}

impl FeedForward {
    pub fn new(
        config: &BitNetConfig,
        vb: VarBuilder,
        layer_idx: usize,
        qk256_backend: Qk256DispatchBackend,
    ) -> Result<Self> {
        let hidden_size = config.model.hidden_size;
        let intermediate_size = config.model.intermediate_size;
        let eps = config.model.rms_norm_eps.map(|e| e as f64).unwrap_or(1e-5);

        Ok(Self {
            gate_proj: linear_with_optional_bias(
                hidden_size,
                intermediate_size,
                vb.pp("gate_proj"),
            )?,
            up_proj: linear_with_optional_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_with_optional_bias(
                intermediate_size,
                hidden_size,
                vb.pp("down_proj"),
            )?,
            sub_layernorm: optional_layer_norm_with_optional_bias(
                intermediate_size,
                eps,
                config.model.norm_type,
                vb.pp("sub_layernorm"),
            )?,
            norm_type: config.model.norm_type,
            norm_eps: eps,
            activation_type: config.model.activation_type,
            layer_idx,
            qk256_backend,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        raw_tensors: &std::collections::HashMap<String, Tensor>,
        _trace_base_seq: usize,
    ) -> Result<Tensor> {
        let gate = self.apply_linear(x, &self.gate_proj, "gate_proj", raw_tensors)?;
        #[cfg(feature = "trace")]
        trace_layer0_tensor(self.layer_idx, _trace_base_seq, 1, "post_ffn_gate_proj", &gate)?;

        // MLP gating diagnostics (point 3 of user's plan)
        if std::env::var("BITNET_DEBUG_MLP").is_ok()
            && let Ok(u_norm) = gate.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()
        {
            tracing::debug!("MLP ||u|| (gate_proj): {:.6e}", u_norm);
        }

        let gate = feed_forward_activation(self.activation_type, &gate)?;
        #[cfg(feature = "trace")]
        trace_layer0_tensor(self.layer_idx, _trace_base_seq, 1, "post_ffn_gate_activation", &gate)?;

        if std::env::var("BITNET_DEBUG_MLP").is_ok()
            && let Ok(activation_norm) = gate.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()
        {
            tracing::debug!(
                "MLP ||activation({:?})||: {:.6e}",
                self.activation_type,
                activation_norm
            );
        }

        let up = self.apply_linear(x, &self.up_proj, "up_proj", raw_tensors)?;
        #[cfg(feature = "trace")]
        trace_layer0_tensor(self.layer_idx, _trace_base_seq, 1, "post_ffn_up_proj", &up)?;

        if std::env::var("BITNET_DEBUG_MLP").is_ok()
            && let Ok(v_norm) = up.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()
        {
            tracing::debug!("MLP ||v|| (up_proj): {:.6e}", v_norm);
        }

        let hidden = gate.mul(&up)?;
        #[cfg(feature = "trace")]
        trace_layer0_tensor(self.layer_idx, _trace_base_seq, 1, "post_swiglu", &hidden)?;

        let hidden = match &self.sub_layernorm {
            Some(norm) => norm_forward(norm, &hidden, self.norm_eps, self.norm_type)?,
            None => hidden,
        };
        #[cfg(feature = "trace")]
        trace_layer0_tensor(self.layer_idx, _trace_base_seq, 1, "post_ffn_subnorm", &hidden)?;

        if std::env::var("BITNET_DEBUG_MLP").is_ok()
            && let Ok(prod_norm) = hidden.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()
        {
            tracing::debug!(
                "MLP ||activation({:?}) * v||: {:.6e}",
                self.activation_type,
                prod_norm
            );
        }

        let output = self.apply_linear(&hidden, &self.down_proj, "down_proj", raw_tensors)?;
        #[cfg(feature = "trace")]
        trace_layer0_tensor(self.layer_idx, _trace_base_seq, 1, "post_down_proj", &output)?;

        if std::env::var("BITNET_DEBUG_MLP").is_ok()
            && let Ok(out_norm) = output.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()
        {
            tracing::debug!("MLP ||W2 * (...)||: {:.6e}", out_norm);
        }

        Ok(output)
    }

    /// Apply linear transformation with QK256 dispatch
    fn apply_linear(
        &self,
        input: &Tensor,
        linear: &Linear,
        proj_name: &str,
        raw_tensors: &std::collections::HashMap<String, Tensor>,
    ) -> Result<Tensor> {
        // Generate weight name based on layer index and projection name
        // Format: "layers.{idx}.feed_forward.{proj_name}.weight.qk256_qs"
        let qk256_key =
            format!("layers.{}.feed_forward.{}.weight.qk256_qs", self.layer_idx, proj_name);

        // Check for QK256 data
        if let Some(qk256_tensor) = raw_tensors.get(&qk256_key) {
            let qk256_scale = qk256_scale_for(raw_tensors, &qk256_key)?;
            tracing::debug!("Using QK256 kernel for {} with scale {}", qk256_key, qk256_scale);
            return forward_qk256_scaled_with_backend(
                input,
                qk256_tensor,
                &qk256_key,
                self.qk256_backend,
                qk256_scale,
            );
        }

        // Fall back to standard linear
        tracing::trace!(
            "Using standard linear for layers.{}.feed_forward.{}",
            self.layer_idx,
            proj_name
        );
        linear.forward(input).map_err(BitNetError::from)
    }
}

/// Transformer Block
pub struct TransformerBlock {
    attention: MultiHeadAttention,
    feed_forward: FeedForward,
    attention_norm: LayerNorm,
    ffn_norm: LayerNorm,
    norm_type: NormType,
    norm_eps: f64,
}

impl TransformerBlock {
    pub fn new(
        config: &BitNetConfig,
        vb: VarBuilder,
        layer_idx: usize,
        qk256_backend: Qk256DispatchBackend,
    ) -> Result<Self> {
        let hidden_size = config.model.hidden_size;
        // PATCH 1: Use RMSNorm epsilon from config header for ALL norms (per-layer + final)
        let eps = config.model.rms_norm_eps.map(|e| e as f64).unwrap_or(1e-5);

        tracing::debug!("TransformerBlock using RMSNorm eps={} (from header)", eps);

        Ok(Self {
            attention: MultiHeadAttention::new(
                config,
                vb.pp("attention"),
                layer_idx,
                qk256_backend,
            )?,
            feed_forward: FeedForward::new(
                config,
                vb.pp("feed_forward"),
                layer_idx,
                qk256_backend,
            )?,
            attention_norm: layer_norm_with_optional_bias(
                hidden_size,
                eps,
                config.model.norm_type,
                vb.pp("attention_norm"),
            )?,
            ffn_norm: layer_norm_with_optional_bias(
                hidden_size,
                eps,
                config.model.norm_type,
                vb.pp("post_attention_layernorm"),
            )?,
            norm_type: config.model.norm_type,
            norm_eps: eps,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        kv_cache: Option<&mut LayerKVCache>,
        raw_tensors: &std::collections::HashMap<String, Tensor>,
    ) -> Result<Tensor> {
        let _trace_base_seq = kv_cache.as_ref().map(|cache| cache.seq_len).unwrap_or(0);

        // Debug input activation norms
        if std::env::var("DEBUG_ATTN").is_ok() {
            let norm = x.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
            eprintln!("[norm] input: {norm:.6e}");
        }

        // Pre-norm attention
        let residual = x;

        // RMSNorm diagnostics (Layer 0 only) - attention norm
        // User's diagnostic: log mean(x^2) and rms = sqrt(mean(x^2) + eps) before/after norm
        if std::env::var("BITNET_DEBUG_RMSNORM").is_ok() {
            static ATTN_NORM_LOGGED: std::sync::Once = std::sync::Once::new();
            ATTN_NORM_LOGGED.call_once(|| {
                if let Ok(mean_sq) =
                    x.sqr().and_then(|s| s.mean_all()).and_then(|m| m.to_scalar::<f32>())
                {
                    // Note: RMSNorm formula is: rms = sqrt(mean(x^2) + eps), y = (x / rms) * weight
                    // The actual eps value is in the LayerNorm (handled by candle)
                    let rms_approx = mean_sq.sqrt(); // Approximate (actual includes eps inside sqrt)
                    tracing::info!(
                        "RMSNorm (attn, layer 0) - input mean(x^2): {:.6e}, approx_rms: {:.6e}",
                        mean_sq,
                        rms_approx
                    );
                    if !rms_approx.is_finite() {
                        tracing::warn!("⚠️  RMSNorm (attn) - input has non-finite values!");
                    }
                }
            });
        }

        let x = norm_forward(&self.attention_norm, x, self.norm_eps, self.norm_type)?;

        // Probe A2: LayerNorm gamma RMS + LN output RMS (layer 0, step 0 only)
        if std::env::var("BITNET_TRACE_RMS").as_deref() == Ok("1") && self.attention.layer_idx == 0
        {
            static LN0_LOGGED: std::sync::Once = std::sync::Once::new();
            LN0_LOGGED.call_once(|| {
                let _ = (|| -> candle_core::Result<()> {
                    // Get gamma (weight) from LayerNorm
                    let gamma_vec = self.attention_norm.weight().to_vec1::<f32>()?;
                    let g_rms = (gamma_vec.iter().map(|x| x * x).sum::<f32>()
                        / gamma_vec.len().max(1) as f32)
                        .sqrt();

                    // Get LN output RMS
                    let ln_vec = x.flatten_all()?.to_vec1::<f32>()?;
                    let ln_rms = (ln_vec.iter().map(|x| x * x).sum::<f32>()
                        / ln_vec.len().max(1) as f32)
                        .sqrt();
                    eprintln!("trace: ln0_gamma_rms={:.6} ln0_out_rms={:.6}", g_rms, ln_rms);
                    Ok(())
                })();
            });
        }

        // Tracepoint 2: Attention norm output (layer-specific)
        #[cfg(feature = "trace")]
        {
            let trace_suffix = format!("blk{}/attn_norm", self.attention.layer_idx);
            trace_tensor_token_axis_record(
                &trace_suffix,
                &x,
                _trace_base_seq,
                1,
                Some(self.attention.layer_idx as isize),
                "attn_norm",
            )?;
        }
        #[cfg(feature = "trace")]
        if self.attention.layer_idx == 0 {
            let dims = x.dims();
            if dims.len() == 3 && dims[0] == 1 {
                let seq_len = dims[1];
                let hidden = dims[2];
                let history =
                    x.reshape(&[seq_len, hidden])?.transpose(0, 1)?.to_dtype(DType::F32)?;
                let trace_seq = trace_target_seq().unwrap_or(_trace_base_seq);
                let trace_name = format!("t{trace_seq}/blk0/attention_v_input_history_ref_layout");
                trace_tensor_record(
                    &trace_name,
                    &history,
                    trace_seq,
                    Some(0),
                    "attention_v_input_history_ref_layout",
                )?;
            }
        }

        // Check norm output
        if std::env::var("BITNET_DEBUG_RMSNORM").is_ok() {
            static ATTN_NORM_OUT_LOGGED: std::sync::Once = std::sync::Once::new();
            ATTN_NORM_OUT_LOGGED.call_once(|| {
                if let Ok(norm_out) = x
                    .sqr()
                    .and_then(|s| s.mean_all())
                    .and_then(|m| m.sqrt())
                    .and_then(|r| r.to_scalar::<f32>())
                {
                    tracing::info!("RMSNorm (attn, layer 0) - output L2 norm: {:.6e}", norm_out);
                    if !norm_out.is_finite() {
                        tracing::warn!("⚠️  RMSNorm (attn) - output is non-finite!");
                    }
                }
            });
        }

        let x = self.attention.forward(&x, kv_cache, raw_tensors, _trace_base_seq)?;
        let x = (x + residual)?;
        #[cfg(feature = "trace")]
        trace_layer0_tensor(
            self.attention.layer_idx,
            _trace_base_seq,
            1,
            "post_attention_residual",
            &x,
        )?;

        // Debug post-attention activation norms
        if std::env::var("DEBUG_ATTN").is_ok() {
            let norm = x.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
            eprintln!("[norm] post-attn: {norm:.6e}");
        }

        // Pre-norm FFN
        let residual = &x;
        #[cfg(feature = "trace")]
        trace_layer0_tensor(self.attention.layer_idx, _trace_base_seq, 1, "pre_ffn_norm", &x)?;

        // RMSNorm diagnostics (Layer 0 only) - FFN norm
        if std::env::var("BITNET_DEBUG_RMSNORM").is_ok() {
            static FFN_NORM_LOGGED: std::sync::Once = std::sync::Once::new();
            FFN_NORM_LOGGED.call_once(|| {
                if let Ok(mean_sq) =
                    x.sqr().and_then(|s| s.mean_all()).and_then(|m| m.to_scalar::<f32>())
                {
                    let rms_approx = mean_sq.sqrt();
                    tracing::info!(
                        "RMSNorm (ffn, layer 0) - input mean(x^2): {:.6e}, approx_rms: {:.6e}",
                        mean_sq,
                        rms_approx
                    );
                    if !rms_approx.is_finite() {
                        tracing::warn!("⚠️  RMSNorm (ffn) - input has non-finite values!");
                    }
                }
            });
        }

        let x = norm_forward(&self.ffn_norm, &x, self.norm_eps, self.norm_type)?;
        #[cfg(feature = "trace")]
        trace_layer0_tensor(self.attention.layer_idx, _trace_base_seq, 1, "post_ffn_norm", &x)?;

        // Check norm output
        if std::env::var("BITNET_DEBUG_RMSNORM").is_ok() {
            static FFN_NORM_OUT_LOGGED: std::sync::Once = std::sync::Once::new();
            FFN_NORM_OUT_LOGGED.call_once(|| {
                if let Ok(norm_out) = x
                    .sqr()
                    .and_then(|s| s.mean_all())
                    .and_then(|m| m.sqrt())
                    .and_then(|r| r.to_scalar::<f32>())
                {
                    tracing::info!("RMSNorm (ffn, layer 0) - output L2 norm: {:.6e}", norm_out);
                    if !norm_out.is_finite() {
                        tracing::warn!("⚠️  RMSNorm (ffn) - output is non-finite!");
                    }
                }
            });
        }

        let x = self.feed_forward.forward(&x, raw_tensors, _trace_base_seq)?;
        let x = (x + residual)?;
        #[cfg(feature = "trace")]
        trace_layer0_tensor(self.attention.layer_idx, _trace_base_seq, 1, "post_layer", &x)?;

        // Debug post-FFN activation norms
        if std::env::var("DEBUG_ATTN").is_ok() {
            let norm = x.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
            eprintln!("[norm] post-ffn: {norm:.6e}");
        }

        Ok(x)
    }
}

/// KV Cache for a single layer
pub struct LayerKVCache {
    pub k: Tensor,
    pub v: Tensor,
    pub seq_len: usize,
    pub max_seq_len: usize,
    pub n_kv_heads: usize, // Store the number of KV heads for validation
}

impl LayerKVCache {
    pub fn new(
        batch_size: usize,
        n_kv_heads: usize, // Changed from n_heads to n_kv_heads
        max_seq_len: usize,
        head_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        let k =
            Tensor::zeros(&[batch_size, n_kv_heads, max_seq_len, head_dim], DType::F32, device)?;
        let v =
            Tensor::zeros(&[batch_size, n_kv_heads, max_seq_len, head_dim], DType::F32, device)?;

        Ok(Self { k, v, seq_len: 0, max_seq_len, n_kv_heads })
    }

    /// Append new K/V tensors to the cache
    ///
    /// **Performance note**: The clones on first append (lines 1130-1131) are necessary
    /// because we accept `&Tensor` but need to store owned tensors. Candle's `Tensor::clone()`
    /// is cheap - it only increments the Arc reference count, not a deep data copy.
    /// Subsequent appends use `Tensor::cat` which allocates new tensors regardless.
    ///
    /// To eliminate these clones would require API changes to accept owned tensors,
    /// which would complicate calling code.
    pub fn append(&mut self, k_new: &Tensor, v_new: &Tensor) -> Result<()> {
        // Expect shapes: k: [B,HKV,T_new,Hd], v: [B,HKV,T_new,Hd] where HKV = n_kv_heads
        let new_seq_len = k_new.dims()[2];

        // Validate that the incoming tensors have the expected number of KV heads
        let k_heads = k_new.dims()[1];
        if k_heads != self.n_kv_heads {
            return Err(BitNetError::Validation(format!(
                "KV cache expects {} heads, but received K tensor with {} heads",
                self.n_kv_heads, k_heads
            )));
        }

        if self.seq_len == 0 {
            // First append: clone is necessary (Arc increment only, not deep copy)
            self.k = k_new.clone();
            self.v = v_new.clone();
        } else {
            // Concatenate along time dimension (dim=2)
            if self.seq_len + new_seq_len > self.max_seq_len {
                return Err(BitNetError::from(candle_core::Error::Msg(
                    "KV cache overflow".to_string(),
                )));
            }
            // Tensor::cat allocates new tensor - no optimization possible here
            self.k = Tensor::cat(&[&self.k, k_new], 2)?;
            self.v = Tensor::cat(&[&self.v, v_new], 2)?;
        }

        self.seq_len += new_seq_len;
        Ok(())
    }

    pub fn clear(&mut self) {
        self.seq_len = 0;
    }
}

/// Full KV Cache for all layers
pub struct KVCache {
    pub layers: Vec<LayerKVCache>,
}

impl KVCache {
    pub fn new(config: &BitNetConfig, batch_size: usize, device: &Device) -> Result<Self> {
        let n_layers = config.model.num_layers;
        let n_heads = config.model.num_heads;
        let hidden_size = config.model.hidden_size;

        // Validate shape assumptions before calculating dimensions
        if !hidden_size.is_multiple_of(n_heads) {
            return Err(BitNetError::Validation(format!(
                "KVCache: hidden_size {} not divisible by num_heads {}",
                hidden_size, n_heads
            )));
        }

        let n_kv_heads = config.model.num_key_value_heads.max(1).min(n_heads);
        if !n_heads.is_multiple_of(n_kv_heads) {
            return Err(BitNetError::Validation(format!(
                "KVCache: num_heads {} not divisible by num_key_value_heads {}",
                n_heads, n_kv_heads
            )));
        }

        let head_dim = hidden_size / n_heads;
        let max_seq_len = config.model.max_position_embeddings;

        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(LayerKVCache::new(batch_size, n_kv_heads, max_seq_len, head_dim, device)?);
        }

        Ok(Self { layers })
    }

    pub fn layer_mut(&mut self, idx: usize) -> Option<&mut LayerKVCache> {
        self.layers.get_mut(idx)
    }

    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }
}

/// Complete Transformer Model
pub struct TransformerModel {
    pub config: BitNetConfig,
    pub embed_tokens: candle_nn::Embedding,
    pub embed_transposed: bool, // True if embeddings are stored as [hidden, vocab]
    pub embed_tied_weight: Option<Tensor>, // Cached transposed embedding weight for tied models [H, V]
    pub layers: Vec<TransformerBlock>,
    pub norm: LayerNorm,
    pub lm_head: Option<Linear>,        // Optional for tied weights
    pub lm_head_weight: Option<Tensor>, // Direct access to lm_head weight for transposed handling
    pub lm_head_transposed: bool,       // True if lm_head is stored as [hidden, vocab]
    device: Device,
    raw_tensors: std::collections::HashMap<String, Tensor>, // Store raw tensors for QK256 dispatch
}

impl TransformerModel {
    pub fn new(config: BitNetConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tensors(config, vb, std::collections::HashMap::new())
    }

    pub fn new_with_tensors(
        config: BitNetConfig,
        vb: VarBuilder,
        raw_tensors: std::collections::HashMap<String, Tensor>,
    ) -> Result<Self> {
        Self::new_with_tensors_and_qk256_backend(config, vb, raw_tensors, Qk256DispatchBackend::Cpu)
    }

    pub fn new_with_tensors_and_qk256_backend(
        config: BitNetConfig,
        vb: VarBuilder,
        raw_tensors: std::collections::HashMap<String, Tensor>,
        qk256_backend: Qk256DispatchBackend,
    ) -> Result<Self> {
        let device = vb.device().clone();
        let vocab_size = config.model.vocab_size;
        let hidden_size = config.model.hidden_size;
        let n_layers = config.model.num_layers;

        let embed_tokens = candle_nn::embedding(vocab_size, hidden_size, vb.pp("embed_tokens"))?;

        // Read transpose flag for embeddings (1-element tensor)
        let embed_transposed = match vb.get((1,), "embed_tokens.transposed") {
            Ok(t) => {
                let vals = t.to_vec1::<f32>()?;
                vals.first().copied().unwrap_or(0.0) > 0.5
            }
            Err(_) => false, // If flag doesn't exist, assume not transposed
        };

        if embed_transposed {
            tracing::info!(
                "Embeddings are transposed [hidden, vocab] - will handle efficiently at runtime"
            );
        }

        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            layers.push(TransformerBlock::new(
                &config,
                vb.pp(format!("layers.{}", i)),
                i,
                qk256_backend,
            )?);
        }

        // Use RMSNorm epsilon from config header (CRITICAL: must match per-layer norms)
        let eps = config.model.rms_norm_eps.map(|e| e as f64).unwrap_or(1e-5);
        tracing::info!("Final norm using RMSNorm eps={} (from header)", eps);

        let norm = layer_norm_with_optional_bias(
            hidden_size,
            eps,
            config.model.norm_type,
            vb.pp("final_norm"),
        )?;

        // Try to load lm_head, but it's optional (can be tied to embeddings).
        // A transposed lm_head flag means the stored weight is [hidden, vocab].
        // That shape is not loadable as a standard Candle Linear [vocab, hidden],
        // so handle it as a direct matmul weight instead of falling through to
        // tied embeddings.
        let lm_head_transposed_flag = match vb.get((1,), "lm_head.transposed") {
            Ok(t) => {
                let vals = t.to_vec1::<f32>()?;
                vals.first().copied().unwrap_or(0.0) > 0.5
            }
            Err(_) => false,
        };

        let (lm_head, lm_head_weight, lm_head_transposed) = if lm_head_transposed_flag {
            tracing::info!("LM head is transposed [hidden, vocab] - using direct matmul weight");
            let weight = vb.get((hidden_size, vocab_size), "lm_head.weight")?;
            (None, Some(weight), true)
        } else {
            match linear_with_optional_bias(hidden_size, vocab_size, vb.pp("lm_head")) {
                Ok(layer) => {
                    let weight = vb.get((vocab_size, hidden_size), "lm_head.weight").ok();
                    (Some(layer), weight, false)
                }
                Err(_) => {
                    tracing::info!("lm_head.weight not found, will use tied weights");
                    (None, None, false)
                }
            }
        };

        // PATCH 2: Optimize tied weights by pre-transposing embeddings once at load
        // NOTE: embed_tokens.embeddings() ALWAYS returns [V,H] (Candle's internal format)
        // regardless of how they were stored in GGUF. We need [H,V] for tied weights.
        let (embed_transposed, embed_tied_weight) = if lm_head.is_none() && lm_head_weight.is_none()
        {
            // No dedicated lm_head, we'll use tied weights - pre-transpose for efficiency
            let embed_weight = embed_tokens.embeddings();
            tracing::info!(
                "Embedding matrix from Candle: {:?} (always [V,H] internally)",
                embed_weight.dims()
            );

            // Always transpose [V,H] -> [H,V] for tied weights, regardless of embed_transposed flag
            // The embed_transposed flag tells us how GGUF stored it, but Candle normalizes to [V,H]
            tracing::info!("Pre-transposing tied embeddings [V,H] -> [H,V] for logits computation");
            let transposed_weight = embed_weight.transpose(0, 1)?; // [H, V]
            tracing::info!("Transposed weight shape: {:?}", transposed_weight.dims());
            (embed_transposed, Some(transposed_weight)) // Cache transposed weight
        } else {
            // Dedicated lm_head exists, no need to optimize embeddings
            (embed_transposed, None)
        };

        Ok(Self {
            config,
            embed_tokens,
            embed_transposed,
            embed_tied_weight,
            layers,
            norm,
            lm_head,
            lm_head_weight,
            lm_head_transposed,
            device,
            raw_tensors,
        })
    }

    pub fn embed(&self, tokens: &[u32]) -> Result<Tensor> {
        let token_ids = Tensor::from_vec(tokens.to_vec(), &[1, tokens.len()], &self.device)?;

        // Get dimensions
        let batch_size = token_ids.dims()[0];
        let seq_len = token_ids.dims()[1];
        let hidden_size = self.config.model.hidden_size;

        // Flatten to [B*S] for index_select
        let flat_ids = token_ids.flatten_all()?;

        if self.embed_transposed {
            // Column-gather path for [hidden, vocab] storage
            // This avoids materializing the full transpose
            let weight = self.embed_tokens.embeddings();

            // index_select on dim=1 gathers columns from [H, V]
            // Result: [H, B*S]
            let cols = weight.index_select(&flat_ids, 1)?;

            // Transpose to [B*S, H] (small transpose, only B*S elements)
            let embeddings = cols.t()?;

            // Reshape to [B, S, H]
            Ok(embeddings.reshape(&[batch_size, seq_len, hidden_size])?)
        } else {
            // Row-gather path for standard [vocab, hidden] storage
            let weight = self.embed_tokens.embeddings();

            // index_select on dim=0 gathers rows from [V, H]
            // Result: [B*S, H]
            let rows = weight.index_select(&flat_ids, 0)?;

            // Reshape to [B, S, H]
            Ok(rows.reshape(&[batch_size, seq_len, hidden_size])?)
        }
    }

    /// Teacher-forcing forward: full sequence `[B,T] -> [B,T,V]` logits
    ///
    /// This implementation mirrors the incremental decoding path by
    /// processing tokens step-by-step with a KV cache. This ensures that
    /// rotary (or absolute) positional encodings are applied per layer with
    /// the correct positions and that a causal mask prevents attending to
    /// future tokens.
    pub fn forward_full(&self, token_ids: &Tensor) -> Result<Tensor> {
        // Token ids expected shape: [B,T]
        let (batch_size, seq_len) = token_ids.dims2()?;

        // Embed the entire sequence once.
        let flat_ids = token_ids.flatten_all()?;
        let ids_vec: Vec<u32> = flat_ids.to_vec1()?;
        let hidden = self.embed(&ids_vec)?;
        let hidden_size = self.config.model.hidden_size;
        let hidden = hidden.reshape(&[batch_size, seq_len, hidden_size])?;

        // Probe A1: Embedding RMS (step 0 only)
        if std::env::var("BITNET_TRACE_RMS").as_deref() == Ok("1") {
            static EMB_LOGGED: std::sync::Once = std::sync::Once::new();
            EMB_LOGGED.call_once(|| {
                let _ = (|| -> candle_core::Result<()> {
                    let emb_vec = hidden.narrow(1, 0, 1)?.flatten_all()?.to_vec1::<f32>()?;
                    let rms = (emb_vec.iter().map(|x| x * x).sum::<f32>()
                        / emb_vec.len().max(1) as f32)
                        .sqrt();
                    eprintln!("trace: emb_rms={:.6}", rms);
                    Ok(())
                })();
            });
        }

        // Tracepoint 1: Embeddings output (after embed, before layers)
        #[cfg(feature = "trace")]
        {
            trace_tensor_token_axis_record("embeddings", &hidden, 0, 1, Some(-1), "embeddings")?;
        }

        // Create per-layer KV cache so that rotary/absolute positional
        // encodings use the proper positions during iterative decoding.
        let mut kv_cache = KVCache::new(&self.config, batch_size, &self.device)?;

        // Collect logits for each position.
        let mut logits_steps = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            // Select the current token's embedding as [B, 1, H] (keep seq dim for attention)
            let step_hidden = hidden.narrow(1, t, 1)?;

            // Run through all layers using the incremental path which applies
            // positional encoding per layer and causal masking internally.
            let step_hidden = self.forward(step_hidden, Some(&mut kv_cache))?;

            // Tracepoint: All layers output for this position
            #[cfg(feature = "trace")]
            {
                let trace_name = format!("t{t}/all_layers_out");
                trace_tensor_record(
                    &trace_name,
                    &step_hidden,
                    t,
                    Some(-2),         // layer=-2 (post-all-layers)
                    "all_layers_out", // stage name
                )?;
            }

            // Project to vocabulary logits for this step.
            let step_logits = self.logits(&step_hidden)?;

            // Trace logits for this position
            #[cfg(feature = "trace")]
            {
                let trace_name = format!("t{t}/logits");
                trace_tensor_record(
                    &trace_name,
                    &step_logits,
                    t,
                    Some(-1), // layer=-1 (post-layers stage)
                    "logits", // stage name
                )?;
            }

            logits_steps.push(step_logits);
        }

        // Stack logits: handle both [B,V] and [B,1,V] shapes
        let logits = if logits_steps[0].dims().len() == 2 {
            // logits are [B, V], stack them to [B, T, V]
            let logits_2d: Vec<_> = logits_steps
                .iter()
                .map(|t| t.unsqueeze(1))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            Tensor::cat(&logits_2d, 1)?
        } else {
            // logits are [B, 1, V], concatenate along time dimension
            Tensor::cat(&logits_steps, 1)?
        };

        // Tracepoint 5: Final logits (first token only)
        #[cfg(feature = "trace")]
        {
            trace_tensor_token_axis_record("logits", &logits, 0, 1, Some(-1), "logits")?;
        }

        Ok(logits)
    }

    /// Forward pass through transformer layers
    ///
    /// **Performance note**: Accepts ownership of `hidden` to avoid cloning on hot path.
    /// Caller should pass owned tensor or use `.clone()` explicitly if needed.
    pub fn forward(&self, hidden: Tensor, mut kv_cache: Option<&mut KVCache>) -> Result<Tensor> {
        let mut x = hidden; // Take ownership - no clone needed!
        let _trace_base_seq = kv_cache
            .as_ref()
            .and_then(|cache| cache.layers.first().map(|layer| layer.seq_len))
            .unwrap_or(0);

        // Tracepoint 1: Embeddings (incremental path - single token)
        // This captures the embedding for the current token being processed
        #[cfg(feature = "trace")]
        {
            trace_tensor_token_axis_record(
                "embeddings",
                &x,
                _trace_base_seq,
                1,
                Some(-1),
                "embeddings",
            )?;
        }

        // Debug input activation norm
        if std::env::var("DEBUG_ATTN").is_ok()
            && let Ok(norm) = x.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()
        {
            eprintln!("[norm] input: {:.6e}", norm);
        }

        for (i, layer) in self.layers.iter().enumerate() {
            let layer_cache = kv_cache.as_mut().and_then(|c| c.layer_mut(i));
            x = layer.forward(&x, layer_cache, &self.raw_tensors)?;

            // Debug layer activation norms (show all layers when debugging)
            if std::env::var("DEBUG_ATTN").is_ok()
                && let Ok(norm) = x.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()
            {
                eprintln!("[norm] layer {i}: {:.6e}", norm);
            }
        }

        let eps = self.config.model.rms_norm_eps.map(|e| e as f64).unwrap_or(1e-5);
        let normalized = norm_forward(&self.norm, &x, eps, self.config.model.norm_type)?;
        #[cfg(feature = "trace")]
        trace_tensor_token_axis_record(
            "final_norm",
            &normalized,
            _trace_base_seq,
            1,
            Some(-1),
            "final_norm",
        )?;
        if std::env::var("DEBUG_ATTN").is_ok()
            && let Ok(norm) = normalized.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()
        {
            eprintln!("[norm] final: {:.6e}", norm);
        }

        Ok(normalized)
    }

    pub fn logits(&self, hidden: &Tensor) -> Result<Tensor> {
        let vocab_size = self.config.model.vocab_size;

        match hidden.rank() {
            2 => {
                // [B, H] - last token only
                let (b, _h) = (hidden.dims()[0], hidden.dims()[1]);

                let logits = if self.lm_head_transposed {
                    if let Some(ref weight) = self.lm_head_weight {
                        hidden.matmul(weight)?
                    } else {
                        return Err(BitNetError::Validation(
                            "lm_head.transposed is set but lm_head.weight is missing".into(),
                        ));
                    }
                } else if let Some(ref lm_head) = self.lm_head {
                    // Use dedicated LM head if available
                    let logits = lm_head.forward(hidden)?; // [B, V]
                    logits.reshape(&[b, vocab_size])?
                } else {
                    // Tied weights: use embedding matrix
                    static LOGGED: std::sync::Once = std::sync::Once::new();
                    LOGGED.call_once(|| {
                        tracing::info!("LM head tied to input embeddings");
                    });

                    let result = if self.embed_transposed {
                        // Embeddings are [hidden, vocab]
                        let embeddings = self.embed_tokens.embeddings();
                        hidden.matmul(embeddings)? // [B, V]
                    } else if let Some(ref cached_weight) = self.embed_tied_weight {
                        // Use pre-transposed cached weight [H, V] - avoids per-step transpose!
                        hidden.matmul(cached_weight)? // [B, V]
                    } else {
                        // Fallback: transpose on-demand (should be rare after optimization)
                        let embeddings = self.embed_tokens.embeddings();
                        let w = embeddings.transpose(0, 1)?; // [H, V]
                        hidden.matmul(&w)? // [B, V]
                    };

                    // Debug: sanity check tied embeddings orientation (runs once)
                    if std::env::var("BITNET_DEBUG_LOGITS").is_ok() {
                        static SANITY_LOGGED: std::sync::Once = std::sync::Once::new();
                        SANITY_LOGGED.call_once(|| {
                            if let Ok(mean_val) = result.mean_all().and_then(|m| m.to_scalar::<f32>())
                                && let Ok(std_val) = result.broadcast_sub(&result.mean_all().unwrap())
                                    .and_then(|d| d.sqr())
                                    .and_then(|s| s.mean_all())
                                    .and_then(|v| v.sqrt())
                                    .and_then(|s| s.to_scalar::<f32>())
                            {
                                tracing::info!("tied logits sanity check - mean/std: {:.4}/{:.4}", mean_val, std_val);

                                // Float sanity check: compare with non-quantized path
                                if let Ok(emb) = self.embed_tokens.embeddings().transpose(0, 1)
                                    && let Ok(ref_logits) = hidden.matmul(&emb)
                                    && let Ok(ref_mean) = ref_logits.mean_all().and_then(|m| m.to_scalar::<f32>())
                                    && let Ok(ref_std) = ref_logits.broadcast_sub(&ref_logits.mean_all().unwrap())
                                        .and_then(|d| d.sqr())
                                        .and_then(|s| s.mean_all())
                                        .and_then(|v| v.sqrt())
                                        .and_then(|s| s.to_scalar::<f32>())
                                {
                                    tracing::info!("float ref logits - mean/std: {:.4}/{:.4}", ref_mean, ref_std);
                                    tracing::info!("correlation check: quantized vs float stats should be similar");
                                }
                            }
                        });
                    }

                    result
                };

                // Debug logits std
                if std::env::var("DEBUG_ATTN").is_ok()
                    && let Ok(mean) = logits.mean_all()
                    && let Ok(diff) = logits.broadcast_sub(&mean)
                    && let Ok(variance) = diff.sqr()?.mean_all()
                    && let Ok(std_val) = variance.sqrt()?.to_scalar::<f32>()
                {
                    eprintln!("[norm] logits std: {:.6e}", std_val);
                }

                // Tracepoint 5: Logits (incremental path - single token)
                // This captures the final logits for the current token [B, V]
                #[cfg(feature = "trace")]
                {
                    let trace_seq = trace_target_seq().unwrap_or(0);
                    let trace_name = format!("t{trace_seq}/logits");
                    trace_tensor_record(&trace_name, &logits, trace_seq, Some(-1), "logits")?;
                }

                Ok(logits)
            }
            3 => {
                // [B, T, H] - all timesteps
                let (b, t, h) = (hidden.dims()[0], hidden.dims()[1], hidden.dims()[2]);

                if self.lm_head_transposed {
                    if let Some(ref weight) = self.lm_head_weight {
                        let hidden_2d = hidden.reshape(&[b * t, h])?;
                        let logits_2d = hidden_2d.matmul(weight)?;
                        Ok(logits_2d.reshape(&[b, t, vocab_size])?)
                    } else {
                        Err(BitNetError::Validation(
                            "lm_head.transposed is set but lm_head.weight is missing".into(),
                        ))
                    }
                } else if let Some(ref lm_head) = self.lm_head {
                    // Use dedicated LM head if available
                    // Standard path: LM head weight is [vocab, hidden]
                    // Flatten to 2D for proper matmul
                    let hidden_2d = hidden.reshape(&[b * t, h])?;
                    let logits_2d = lm_head.forward(&hidden_2d)?;
                    Ok(logits_2d.reshape(&[b, t, vocab_size])?)
                } else {
                    // Tied weights: use embedding matrix
                    static LOGGED: std::sync::Once = std::sync::Once::new();
                    LOGGED.call_once(|| {
                        tracing::info!("LM head tied to input embeddings");
                    });

                    if self.embed_transposed {
                        // Embeddings are [hidden, vocab], flatten hidden for matmul
                        let embeddings = self.embed_tokens.embeddings();
                        let hidden_2d = hidden.reshape(&[b * t, h])?;
                        let logits_2d = hidden_2d.matmul(embeddings)?;
                        Ok(logits_2d.reshape(&[b, t, vocab_size])?)
                    } else if let Some(ref cached_weight) = self.embed_tied_weight {
                        // Use pre-transposed cached weight [H, V] - avoids per-step transpose!
                        let hidden_2d = hidden.reshape(&[b * t, h])?;
                        let logits_2d = hidden_2d.matmul(cached_weight)?;
                        Ok(logits_2d.reshape(&[b, t, vocab_size])?)
                    } else {
                        // Fallback: transpose on-demand (should be rare after optimization)
                        let embeddings = self.embed_tokens.embeddings();
                        let w = embeddings.transpose(0, 1)?; // [H, V]
                        let hidden_2d = hidden.reshape(&[b * t, h])?;
                        let logits_2d = hidden_2d.matmul(&w)?;
                        Ok(logits_2d.reshape(&[b, t, vocab_size])?)
                    }
                }
            }
            _ => Err(BitNetError::Validation("unexpected hidden rank".into())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::RmsNorm;

    /// Helper to compute RMS (root mean square) of a tensor
    fn compute_rms(tensor: &Tensor) -> candle_core::Result<f64> {
        let squared = tensor.sqr()?;
        let mean = squared.mean_all()?;
        let rms = mean.sqrt()?.to_scalar::<f32>()? as f64;
        Ok(rms)
    }

    fn rmsnorm_manual_f32(input: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
        let mut sum_sq = 0.0f32;
        for &value in input {
            sum_sq += value * value;
        }
        let denom = ((sum_sq / input.len() as f32) + eps).sqrt();
        input.iter().zip(gamma).map(|(&x, &g)| (x / denom) * g).collect()
    }

    fn rmsnorm_manual_f64(input: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
        let mut sum_sq = 0.0f64;
        for &value in input {
            sum_sq += (value as f64) * (value as f64);
        }
        let denom = ((sum_sq / input.len() as f64) + eps as f64).sqrt() as f32;
        input.iter().zip(gamma).map(|(&x, &g)| (x / denom) * g).collect()
    }

    fn max_abs_delta(left: &[f32], right: &[f32]) -> f32 {
        left.iter().zip(right).map(|(&l, &r)| (l - r).abs()).fold(0.0f32, f32::max)
    }

    #[test]
    fn test_layer_norm_with_standard_gamma() -> candle_core::Result<()> {
        // Test that RMSNorm behaves correctly with standard gamma (RMS ≈ 1.0)
        let device = Device::Cpu;
        let hidden_size = 2560;
        let eps = 1e-5;

        // Create input tensor [1, 1, 2560]
        let input_data: Vec<f32> = (0..hidden_size)
            .map(|i| {
                let x = i as f32 / hidden_size as f32;
                ((x * 10.0).sin() + (x * 20.0).cos()) * 0.5
            })
            .collect();

        let input = Tensor::from_slice(&input_data, (1, 1, hidden_size), &device)?;

        // Create standard gamma (all ones)
        let gamma = Tensor::ones(hidden_size, DType::F32, &device)?;

        // Apply RMSNorm
        let rms_norm = RmsNorm::new(gamma, eps);
        let output = rms_norm.forward(&input)?;

        // Verify output RMS is reasonable (should be close to 1.0)
        let output_rms = compute_rms(&output)?;

        assert!(
            output_rms > 0.5 && output_rms < 2.0,
            "Output RMS should be reasonable with standard gamma, got {:.6e}",
            output_rms
        );

        // Verify no NaN/Inf
        let vec_data: Vec<f32> = output.flatten_all()?.to_vec1()?;
        let has_nan = vec_data.iter().any(|x| x.is_nan());
        let has_inf = vec_data.iter().any(|x| x.is_infinite());
        assert!(!has_nan, "Output should not contain NaN");
        assert!(!has_inf, "Output should not contain Inf");

        Ok(())
    }

    #[test]
    fn test_rmsnorm_cpu_runtime_matches_f32_accumulation() -> candle_core::Result<()> {
        use std::collections::HashMap;

        let device = Device::Cpu;
        let hidden_size = 4096;
        let eps = 1e-5;
        let input_data = std::iter::once(4096.0f32)
            .chain(std::iter::repeat_n(1.0f32, hidden_size - 1))
            .collect::<Vec<_>>();
        let gamma_data = vec![1.0f32; hidden_size];
        let input = Tensor::from_slice(&input_data, (1, 1, hidden_size), &device)?;
        let mut tensors = HashMap::new();
        tensors
            .insert("weight".to_string(), Tensor::from_slice(&gamma_data, hidden_size, &device)?);
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);

        let norm = layer_norm_with_optional_bias(hidden_size, eps, NormType::RmsNorm, vb)?;
        let output = norm.forward(&input)?;
        let output = output.flatten_all()?.to_vec1::<f32>()?;
        let f32_replay = rmsnorm_manual_f32(&input_data, &gamma_data, eps as f32);
        let f64_replay = rmsnorm_manual_f64(&input_data, &gamma_data, eps as f32);
        let candle_vs_f32 = max_abs_delta(&output, &f32_replay);
        let candle_vs_f64 = max_abs_delta(&output, &f64_replay);
        let f32_vs_f64 = max_abs_delta(&f32_replay, &f64_replay);

        assert!(
            candle_vs_f32 <= 1.0e-8,
            "Candle RMSNorm CPU path should match f32 accumulation replay, got {candle_vs_f32:.6e}"
        );
        assert!(f32_vs_f64 > 0.0, "diagnostic vector should distinguish f32 and f64 accumulation");
        assert!(
            candle_vs_f32 <= candle_vs_f64,
            "Candle RMSNorm CPU path should be no farther from f32 replay than f64 replay: f32={candle_vs_f32:.6e}, f64={candle_vs_f64:.6e}"
        );

        Ok(())
    }

    #[test]
    #[serial_test::serial(bitnet_env)]
    fn test_diag_rmsnorm_f64_accum_env_matches_f64_replay() -> Result<()> {
        use std::collections::HashMap;

        let device = Device::Cpu;
        let hidden_size = 4096;
        let eps = 1e-5;
        let input_data = std::iter::once(4096.0f32)
            .chain(std::iter::repeat_n(1.0f32, hidden_size - 1))
            .collect::<Vec<_>>();
        let gamma_data = vec![1.0f32; hidden_size];
        let input = Tensor::from_slice(&input_data, (1, 1, hidden_size), &device)?;
        let mut tensors = HashMap::new();
        tensors
            .insert("weight".to_string(), Tensor::from_slice(&gamma_data, hidden_size, &device)?);
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let norm = layer_norm_with_optional_bias(hidden_size, eps, NormType::RmsNorm, vb)?;

        let mut scope = bitnet_test_support::EnvScope::new();
        scope.set(DIAG_RMSNORM_F64_ACCUM_ENV, "1");

        let output = norm_forward(&norm, &input, eps, NormType::RmsNorm)?;
        let output = output.flatten_all()?.to_vec1::<f32>()?;
        let f32_replay = rmsnorm_manual_f32(&input_data, &gamma_data, eps as f32);
        let f64_replay = rmsnorm_manual_f64(&input_data, &gamma_data, eps as f32);
        let diag_vs_f64 = max_abs_delta(&output, &f64_replay);
        let diag_vs_f32 = max_abs_delta(&output, &f32_replay);

        assert!(
            diag_vs_f64 <= 1.0e-8,
            "diagnostic f64 RMSNorm mode should match f64 accumulation replay, got {diag_vs_f64:.6e}"
        );
        assert!(
            diag_vs_f64 <= diag_vs_f32,
            "diagnostic f64 RMSNorm mode should be no farther from f64 replay than f32 replay: f64={diag_vs_f64:.6e}, f32={diag_vs_f32:.6e}"
        );

        Ok(())
    }

    #[test]
    #[serial_test::serial(bitnet_env)]
    fn test_diag_rmsnorm_f64_accum_env_does_not_change_layernorm() -> Result<()> {
        use std::collections::HashMap;

        let device = Device::Cpu;
        let hidden_size = 4;
        let eps = 1e-5;
        let input = Tensor::from_slice(&[1.0f32, 2.0, 4.0, 8.0], (1, hidden_size), &device)?;
        let mut tensors = HashMap::new();
        tensors.insert("weight".to_string(), Tensor::ones(hidden_size, DType::F32, &device)?);
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let norm = layer_norm_with_optional_bias(hidden_size, eps, NormType::LayerNorm, vb)?;
        let expected = norm.forward(&input)?.flatten_all()?.to_vec1::<f32>()?;

        let mut scope = bitnet_test_support::EnvScope::new();
        scope.set(DIAG_RMSNORM_F64_ACCUM_ENV, "1");

        let actual = norm_forward(&norm, &input, eps, NormType::LayerNorm)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        assert_eq!(max_abs_delta(&actual, &expected), 0.0);
        Ok(())
    }

    #[test]
    fn attention_f16_dot_input_uses_f16_roundtrip_values() -> Result<()> {
        let device = Device::Cpu;
        let input =
            Tensor::from_slice(&[1.0003f32, -2.0007, 3.1259, -4.2509], (1, 1, 1, 4), &device)?;

        let output = attention_f16_dot_input(&input)?;
        let values = output.flatten_all()?.to_vec1::<f32>()?;

        assert_eq!(values, vec![1.0, -2.0, 3.125, -4.25]);
        Ok(())
    }

    #[test]
    fn attention_score_key_input_preserves_f32_values() -> Result<()> {
        let device = Device::Cpu;
        let input =
            Tensor::from_slice(&[1.0003f32, -2.0007, 3.1259, -4.2509], (1, 1, 1, 4), &device)?;

        let output = attention_score_key_input(&input)?;
        let values = output.flatten_all()?.to_vec1::<f32>()?;

        assert_eq!(values, vec![1.0003, -2.0007, 3.1259, -4.2509]);
        Ok(())
    }

    #[test]
    fn attention_score_qk_inputs_match_reference_precision_contract() -> Result<()> {
        let device = Device::Cpu;
        let query =
            Tensor::from_slice(&[1.0003f32, -2.0007, 3.1259, -4.2509], (1, 1, 1, 4), &device)?;
        let key = Tensor::from_slice(&[5.0003f32, 6.0007, -7.1259, 8.2509], (1, 1, 1, 4), &device)?;

        let q_values = attention_f16_dot_input(&query)?.flatten_all()?.to_vec1::<f32>()?;
        let k_values = attention_score_key_input(&key)?.flatten_all()?.to_vec1::<f32>()?;
        let score = q_values.iter().zip(k_values.iter()).fold(0.0f32, |sum, (q, k)| sum + q * k);

        assert_eq!(q_values, vec![1.0, -2.0, 3.125, -4.25]);
        assert_eq!(k_values, vec![5.0003, 6.0007, -7.1259, 8.2509]);
        assert_eq!(score, -64.33586);
        Ok(())
    }

    #[test]
    fn attention_value_mix_uses_f16_roundtrip_values() -> Result<()> {
        let device = Device::Cpu;
        let weights = Tensor::from_slice(&[0.25f32, 0.25, 0.25, 0.25], (1, 1, 1, 4), &device)?;
        let values =
            Tensor::from_slice(&[1.0003f32, -2.0007, 3.1259, -4.2509], (1, 1, 4, 1), &device)?;

        let rounded_values = attention_f16_dot_input(&values)?;
        let mixed = weights.matmul(&rounded_values)?;
        let mixed = mixed.flatten_all()?.to_vec1::<f32>()?;

        assert_eq!(mixed, vec![-0.53125]);
        Ok(())
    }

    #[test]
    fn test_layer_norm_with_small_gamma() -> candle_core::Result<()> {
        // Test RMSNorm with gamma RMS ≈ 0.018 (our model's case)
        let device = Device::Cpu;
        let hidden_size = 2560;
        let eps = 1e-5;

        // Create input tensor [1, 1, 2560]
        let input_data: Vec<f32> = (0..hidden_size)
            .map(|i| {
                let x = i as f32 / hidden_size as f32;
                ((x * 10.0).sin() + (x * 20.0).cos()) * 0.5
            })
            .collect();

        let input = Tensor::from_slice(&input_data, (1, 1, hidden_size), &device)?;

        // Create gamma with RMS ≈ 1/√2560 ≈ 0.01976
        let target_rms = 1.0 / (hidden_size as f64).sqrt();
        let gamma_data: Vec<f32> = vec![target_rms as f32; hidden_size];
        let gamma = Tensor::from_slice(&gamma_data, hidden_size, &device)?;

        // Verify gamma RMS
        let gamma_rms = compute_rms(&gamma)?;
        assert!(
            (gamma_rms - target_rms).abs() < 0.001,
            "Gamma RMS should be close to {:.6e}, got {:.6e}",
            target_rms,
            gamma_rms
        );

        // Apply RMSNorm
        let rms_norm = RmsNorm::new(gamma, eps);
        let output = rms_norm.forward(&input)?;

        // Verify output RMS is smaller but reasonable
        let output_rms = compute_rms(&output)?;

        assert!(
            output_rms > 0.001 && output_rms < 0.1,
            "Output RMS should be reasonable with small gamma, got {:.6e}",
            output_rms
        );

        // Verify no NaN/Inf
        let vec_data: Vec<f32> = output.flatten_all()?.to_vec1()?;
        let has_nan = vec_data.iter().any(|x| x.is_nan());
        let has_inf = vec_data.iter().any(|x| x.is_infinite());
        assert!(!has_nan, "Output should not contain NaN");
        assert!(!has_inf, "Output should not contain Inf");

        Ok(())
    }

    #[test]
    #[serial_test::serial(bitnet_env)]
    fn test_layer_norm_with_optional_bias() -> candle_core::Result<()> {
        // Test layer_norm_with_optional_bias helper with no-bias LayerNorm path
        let device = Device::Cpu;
        let hidden_size = 128;
        let eps = 1e-5;

        // Create VarBuilder with only weight (no bias)
        use std::collections::HashMap;

        let mut tensors = HashMap::new();
        let weight = Tensor::ones(hidden_size, DType::F32, &device)?;
        tensors.insert("weight".to_string(), weight);

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);

        // Create LayerNorm (should use no-bias LayerNorm path due to missing bias)
        let layer_norm = layer_norm_with_optional_bias(hidden_size, eps, NormType::LayerNorm, vb)?;

        // Test forward pass
        let input_data: Vec<f32> =
            (0..hidden_size).map(|i| (i as f32 / hidden_size as f32).sin()).collect();
        let input = Tensor::from_slice(&input_data, (1, hidden_size), &device)?;

        let output = layer_norm.forward(&input)?;

        // Verify output shape
        assert_eq!(output.shape(), input.shape());

        // Verify no NaN/Inf
        let vec_data: Vec<f32> = output.flatten_all()?.to_vec1()?;
        let has_nan = vec_data.iter().any(|x| x.is_nan());
        let has_inf = vec_data.iter().any(|x| x.is_infinite());
        assert!(!has_nan, "Output should not contain NaN");
        assert!(!has_inf, "Output should not contain Inf");

        Ok(())
    }

    #[test]
    fn test_optional_layer_norm_with_optional_bias() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let hidden_size = 64;
        let eps = 1e-5;

        use std::collections::HashMap;

        let empty_vb = VarBuilder::from_tensors(HashMap::new(), DType::F32, &device);
        assert!(
            optional_layer_norm_with_optional_bias(
                hidden_size,
                eps,
                NormType::LayerNorm,
                empty_vb
            )?
            .is_none(),
            "missing optional norm weight should skip the layer"
        );

        let mut tensors = HashMap::new();
        tensors.insert("weight".to_string(), Tensor::ones(hidden_size, DType::F32, &device)?);

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let layer_norm =
            optional_layer_norm_with_optional_bias(hidden_size, eps, NormType::LayerNorm, vb)?
                .expect("weight-only optional norm should build a no-bias LayerNorm");

        let input_data: Vec<f32> =
            (0..hidden_size).map(|i| (i as f32 / hidden_size as f32).cos()).collect();
        let input = Tensor::from_slice(&input_data, (1, hidden_size), &device)?;

        let output = layer_norm.forward(&input)?;
        assert_eq!(output.shape(), input.shape());

        Ok(())
    }

    #[test]
    fn test_feed_forward_relu2_activation() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let input = Tensor::from_slice(&[-2.0f32, -0.5, 0.0, 1.5, 3.0], (5,), &device)?;

        let output = feed_forward_activation(ActivationType::Relu2, &input)?;
        let values = output.to_vec1::<f32>()?;

        assert_eq!(values, vec![0.0, 0.0, 0.0, 2.25, 9.0]);

        Ok(())
    }

    #[test]
    fn test_rms_norm_type_does_not_remove_mean() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let hidden_size = 3;
        let eps = 1e-5;

        use std::collections::HashMap;

        let mut tensors = HashMap::new();
        tensors.insert("weight".to_string(), Tensor::ones(hidden_size, DType::F32, &device)?);

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let norm = layer_norm_with_optional_bias(hidden_size, eps, NormType::RmsNorm, vb)?;

        let input = Tensor::from_slice(&[1.0f32, 2.0, 3.0], (1, hidden_size), &device)?;
        let output = norm.forward(&input)?;
        let values = output.to_vec2::<f32>()?;

        assert!(
            values[0].iter().all(|value| *value > 0.0),
            "RMSNorm should preserve positive sign instead of mean-centering: {values:?}"
        );

        Ok(())
    }

    #[test]
    fn transposed_lm_head_uses_dedicated_output_weight() -> Result<()> {
        use std::collections::HashMap;

        let device = Device::Cpu;
        let vocab_size = 4;
        let hidden_size = 2;
        let mut config = BitNetConfig::default();
        config.model.vocab_size = vocab_size;
        config.model.hidden_size = hidden_size;
        config.model.num_layers = 0;

        let mut tensors = HashMap::new();
        tensors.insert(
            "embed_tokens.weight".to_string(),
            Tensor::zeros((vocab_size, hidden_size), DType::F32, &device)?,
        );
        tensors.insert(
            "final_norm.weight".to_string(),
            Tensor::ones(hidden_size, DType::F32, &device)?,
        );
        tensors.insert(
            "final_norm.bias".to_string(),
            Tensor::zeros(hidden_size, DType::F32, &device)?,
        );
        tensors.insert(
            "lm_head.weight".to_string(),
            Tensor::from_slice(
                &[
                    1.0f32, 0.0, 0.0, 0.0, // hidden dim 0 -> token 0
                    0.0, 1.0, 0.0, 0.0, // hidden dim 1 -> token 1
                ],
                (hidden_size, vocab_size),
                &device,
            )?,
        );
        tensors.insert(
            "lm_head.transposed".to_string(),
            Tensor::from_slice(&[1.0f32], (1,), &device)?,
        );

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let model = TransformerModel::new_with_tensors_and_qk256_backend(
            config,
            vb,
            HashMap::new(),
            Qk256DispatchBackend::Cpu,
        )?;

        assert!(
            model.lm_head_transposed,
            "transposed lm_head flag must survive model construction"
        );
        assert!(
            model.lm_head_weight.is_some(),
            "transposed lm_head must keep its dedicated output weight"
        );
        assert!(
            model.embed_tied_weight.is_none(),
            "dedicated transposed lm_head must not fall back to tied embeddings"
        );

        let hidden = Tensor::from_slice(&[2.0f32, 3.0], (1, hidden_size), &device)?;
        let logits = model.logits(&hidden)?;
        let values = logits.to_vec2::<f32>()?;
        assert_eq!(values, vec![vec![2.0, 3.0, 0.0, 0.0]]);

        Ok(())
    }

    #[test]
    fn tied_embedding_logits_use_cached_embedding_transpose() -> Result<()> {
        use std::collections::HashMap;

        let device = Device::Cpu;
        let vocab_size = 4;
        let hidden_size = 3;
        let mut config = BitNetConfig::default();
        config.model.vocab_size = vocab_size;
        config.model.hidden_size = hidden_size;
        config.model.num_layers = 0;

        let mut tensors = HashMap::new();
        tensors.insert(
            "embed_tokens.weight".to_string(),
            Tensor::from_slice(
                &[
                    1.0f32, 0.0, 0.0, // token 0
                    0.0, 1.0, 0.0, // token 1
                    0.0, 0.0, 1.0, // token 2
                    1.0, 1.0, 1.0, // token 3
                ],
                (vocab_size, hidden_size),
                &device,
            )?,
        );
        tensors.insert(
            "final_norm.weight".to_string(),
            Tensor::ones(hidden_size, DType::F32, &device)?,
        );
        tensors.insert(
            "final_norm.bias".to_string(),
            Tensor::zeros(hidden_size, DType::F32, &device)?,
        );

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let model = TransformerModel::new_with_tensors_and_qk256_backend(
            config,
            vb,
            HashMap::new(),
            Qk256DispatchBackend::Cpu,
        )?;

        assert!(
            model.lm_head.is_none() && model.lm_head_weight.is_none(),
            "missing lm_head must use tied embeddings"
        );
        assert!(
            model.embed_tied_weight.is_some(),
            "tied embedding logits should cache [hidden, vocab] weight"
        );

        let hidden_2d = Tensor::from_slice(&[2.0f32, 3.0, 5.0], (1, hidden_size), &device)?;
        let logits_2d = model.logits(&hidden_2d)?;
        assert_eq!(logits_2d.to_vec2::<f32>()?, vec![vec![2.0, 3.0, 5.0, 10.0]]);

        let hidden_3d = Tensor::from_slice(
            &[
                2.0f32, 3.0, 5.0, // step 0
                7.0, 11.0, 13.0, // step 1
            ],
            (1, 2, hidden_size),
            &device,
        )?;
        let logits_3d = model.logits(&hidden_3d)?;
        assert_eq!(
            logits_3d.to_vec3::<f32>()?,
            vec![vec![vec![2.0, 3.0, 5.0, 10.0], vec![7.0, 11.0, 13.0, 31.0]]]
        );

        Ok(())
    }

    #[test]
    #[serial_test::serial(bitnet_env)]
    fn test_layer_norm_requires_bias_when_guard_enabled() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let hidden_size = 64;
        let eps = 1e-5;

        use std::collections::HashMap;

        let mut tensors = HashMap::new();
        let weight = Tensor::ones(hidden_size, DType::F32, &device)?;
        tensors.insert("weight".to_string(), weight);

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);

        let mut scope = bitnet_test_support::EnvScope::new();
        scope.set("BITNET_REQUIRE_LAYER_NORM_BIAS", "1");

        let err = layer_norm_with_optional_bias(hidden_size, eps, NormType::LayerNorm, vb)
            .expect_err("missing bias should error when BITNET_REQUIRE_LAYER_NORM_BIAS=1");
        assert!(
            err.to_string().contains("LayerNorm bias tensor is required"),
            "unexpected error: {err}"
        );

        Ok(())
    }

    #[test]
    fn test_rmsnorm_formula_consistency() -> candle_core::Result<()> {
        // Verify RMSNorm formula: output = (x / sqrt(mean(x²) + eps)) * gamma
        let device = Device::Cpu;
        let hidden_size = 256;
        let eps = 1e-5;

        // Create input
        let input_data: Vec<f32> = (0..hidden_size).map(|i| (i as f32 / 100.0).sin()).collect();
        let input = Tensor::from_slice(&input_data, (1, hidden_size), &device)?;

        // Create gamma
        let gamma = Tensor::ones(hidden_size, DType::F32, &device)?;

        // Apply RMSNorm via Candle
        let rms_norm = RmsNorm::new(gamma.clone(), eps);
        let output_candle = rms_norm.forward(&input)?;

        // Manually compute RMSNorm
        let squared = input.sqr()?;
        let mean_squared = squared.mean_keepdim(1)?; // Mean over last dimension
        let rms_denominator = (mean_squared + eps)?.sqrt()?;
        let normalized = input.broadcast_div(&rms_denominator)?;
        let output_manual = normalized.broadcast_mul(&gamma)?;

        // Compare outputs
        let diff = (output_candle.sub(&output_manual))?.abs()?;
        let diff_vec: Vec<f32> = diff.flatten_all()?.to_vec1()?;
        let max_diff = diff_vec.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;

        assert!(
            max_diff < 1e-5,
            "Candle's RMSNorm should match manual computation: max_diff={:.6e}",
            max_diff
        );

        Ok(())
    }

    #[test]
    fn test_rmsnorm_output_scale_relationship() -> candle_core::Result<()> {
        // Test that output RMS scales proportionally with gamma RMS
        let device = Device::Cpu;
        let hidden_size = 256;
        let eps = 1e-5;

        // Create same input for both tests
        let input_data: Vec<f32> = (0..hidden_size)
            .map(|i| {
                let x = i as f32 / hidden_size as f32;
                ((x * 10.0).sin() + (x * 20.0).cos()) * 2.0
            })
            .collect();
        let input = Tensor::from_slice(&input_data, (1, hidden_size), &device)?;

        // Test 1: Standard gamma (RMS ≈ 1.0)
        let gamma_std = Tensor::ones(hidden_size, DType::F32, &device)?;
        let rms_norm_std = RmsNorm::new(gamma_std.clone(), eps);
        let output_std = rms_norm_std.forward(&input)?;
        let output_std_rms = compute_rms(&output_std)?;

        // Test 2: Small gamma (RMS ≈ 0.02)
        let target_rms = 0.02;
        let gamma_small =
            Tensor::from_slice(&vec![target_rms as f32; hidden_size], hidden_size, &device)?;
        let rms_norm_small = RmsNorm::new(gamma_small.clone(), eps);
        let output_small = rms_norm_small.forward(&input)?;
        let output_small_rms = compute_rms(&output_small)?;

        // Verify scaling relationship
        let gamma_std_rms = compute_rms(&gamma_std)?;
        let gamma_small_rms = compute_rms(&gamma_small)?;
        let expected_ratio = gamma_small_rms / gamma_std_rms;
        let actual_ratio = output_small_rms / output_std_rms;

        assert!(
            (actual_ratio - expected_ratio).abs() < 0.01,
            "Output RMS should scale with gamma RMS: expected ratio {:.6}, got {:.6}",
            expected_ratio,
            actual_ratio
        );

        Ok(())
    }
}

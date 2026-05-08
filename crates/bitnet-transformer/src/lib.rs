use bitnet_common::{
    BitNetConfig, BitNetError, Result,
    config::{ActivationType, NormType},
};
use bitnet_qk256_dispatch::{
    forward_qk256_with_scale, record_bitnet_linear_cpu_fallback, record_bitnet_linear_unsupported,
    strict_cuda_bitnet_backend_requested,
};
use bitnet_rope::{build_tables as build_rope_tables, resolve_base as resolve_rope_base};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{LayerNorm, Linear, VarBuilder};

fn qwen_trace_path() -> Option<std::path::PathBuf> {
    std::env::var("BITNET_QWEN_TRACE_JSONL")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(std::path::PathBuf::from)
}

fn qwen_trace_enabled() -> bool {
    qwen_trace_path().is_some() || std::env::var("BITNET_QWEN_TRACE").as_deref() == Ok("1")
}

fn qwen_trace_active() -> bool {
    qwen_trace_enabled() && std::env::var("BITNET_QWEN_TRACE_ACTIVE").as_deref() == Ok("1")
}

fn qwen_trace_layer_enabled(layer_idx: usize) -> bool {
    if !qwen_trace_active() {
        return false;
    }
    let requested_layer = std::env::var("BITNET_QWEN_TRACE_LAYER")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    requested_layer == layer_idx
}

fn qwen_trace_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n").replace('\r', "\\r")
}

fn qwen_trace_number(value: f64) -> String {
    if value.is_finite() { format!("{value:.9}") } else { "null".to_string() }
}

fn qwen_trace_write_line(line: &str) {
    if let Some(path) = qwen_trace_path() {
        if let Some(parent) = path.parent()
            && let Err(err) = std::fs::create_dir_all(parent)
        {
            eprintln!("qwen_trace_write_failed: create_dir_all {}: {err}", parent.display());
            return;
        }
        match std::fs::OpenOptions::new().create(true).append(true).open(&path) {
            Ok(mut file) => {
                if let Err(err) = std::io::Write::write_all(&mut file, line.as_bytes())
                    .and_then(|_| std::io::Write::write_all(&mut file, b"\n"))
                {
                    eprintln!("qwen_trace_write_failed: {}: {err}", path.display());
                }
            }
            Err(err) => eprintln!("qwen_trace_write_failed: {}: {err}", path.display()),
        }
    } else if std::env::var("BITNET_QWEN_TRACE").as_deref() == Ok("1") {
        eprintln!("{line}");
    }
}

fn qwen_trace_event(stage: &str, fields_json: &str) {
    if !qwen_trace_enabled() {
        return;
    }
    let step = std::env::var("BITNET_QWEN_TRACE_STEP").unwrap_or_else(|_| "null".to_string());
    qwen_trace_write_line(&format!(
        "{{\"kind\":\"qwen_trace_event\",\"stage\":\"{}\",\"step\":{},{} }}",
        qwen_trace_escape(stage),
        step,
        fields_json
    ));
}

fn qwen_trace_tensor(
    stage: &str,
    layer_idx: Option<usize>,
    tensor: &Tensor,
) -> candle_core::Result<()> {
    if !qwen_trace_active() {
        return Ok(());
    }
    if let Some(layer_idx) = layer_idx
        && !qwen_trace_layer_enabled(layer_idx)
    {
        return Ok(());
    }

    let tensor_f32 =
        if tensor.dtype() == DType::F32 { tensor.clone() } else { tensor.to_dtype(DType::F32)? };
    let values = tensor_f32.flatten_all()?.to_vec1::<f32>()?;
    let mut finite_count = 0usize;
    let mut nonfinite_count = 0usize;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut checksum = 0.0f64;
    for (idx, value) in values.iter().enumerate() {
        let value = *value as f64;
        if value.is_finite() {
            finite_count += 1;
            sum += value;
            sum_sq += value * value;
            min = min.min(value);
            max = max.max(value);
            if idx < 4096 {
                checksum += value * ((idx % 257) + 1) as f64;
            }
        } else {
            nonfinite_count += 1;
        }
    }

    let denom = finite_count.max(1) as f64;
    let mean = sum / denom;
    let rms = (sum_sq / denom).sqrt();
    let sample = values
        .iter()
        .take(8)
        .map(|value| qwen_trace_number(*value as f64))
        .collect::<Vec<_>>()
        .join(",");
    let dims = tensor.dims().iter().map(|dim| dim.to_string()).collect::<Vec<_>>().join(",");
    let layer_json = layer_idx.map(|idx| idx.to_string()).unwrap_or_else(|| "null".to_string());
    let step = std::env::var("BITNET_QWEN_TRACE_STEP").unwrap_or_else(|_| "null".to_string());

    qwen_trace_write_line(&format!(
        "{{\"kind\":\"qwen_trace_tensor\",\"stage\":\"{}\",\"step\":{},\"layer\":{},\"dtype\":\"{:?}\",\"dims\":[{}],\"len\":{},\"finite\":{},\"nonfinite\":{},\"mean\":{},\"rms\":{},\"min\":{},\"max\":{},\"checksum\":{},\"sample\":[{}]}}",
        qwen_trace_escape(stage),
        step,
        layer_json,
        tensor.dtype(),
        dims,
        values.len(),
        finite_count,
        nonfinite_count,
        qwen_trace_number(mean),
        qwen_trace_number(rms),
        qwen_trace_number(min),
        qwen_trace_number(max),
        qwen_trace_number(checksum),
        sample
    ));
    Ok(())
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
#[cfg(test)]
fn layer_norm_with_optional_bias(
    normalized_shape: usize,
    eps: f64,
    vb: VarBuilder,
) -> candle_core::Result<LayerNorm> {
    norm_with_optional_bias(NormType::LayerNorm, normalized_shape, eps, vb)
}

fn norm_with_optional_bias(
    norm_type: NormType,
    normalized_shape: usize,
    eps: f64,
    vb: VarBuilder,
) -> candle_core::Result<LayerNorm> {
    let weight = vb.get((normalized_shape,), "weight")?;
    if matches!(norm_type, NormType::RmsNorm) {
        if vb.get((normalized_shape,), "bias").is_ok() {
            tracing::debug!(
                "Bias tensor present for RMSNorm layer; ignoring bias [{}]",
                normalized_shape
            );
        }
        tracing::debug!("Using RMSNorm without mean subtraction [{}]", normalized_shape);
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
            // because these gamma weights are calibrated for LayerNorm semantics
            // (mean subtraction). RMSNorm callers return earlier in this helper.
            tracing::debug!(
                "Bias tensor missing for norm layer; using LayerNorm without bias (mean subtraction enabled) [{}]",
                normalized_shape
            );
            Ok(LayerNorm::new_no_bias(weight, eps))
        }
    }
}

fn optional_layer_norm_with_optional_bias(
    norm_type: NormType,
    normalized_shape: usize,
    eps: f64,
    vb: VarBuilder,
) -> candle_core::Result<Option<LayerNorm>> {
    if !vb.contains_tensor("weight") {
        return Ok(None);
    }

    Ok(Some(norm_with_optional_bias(norm_type, normalized_shape, eps, vb)?))
}

fn qk256_scale_key(qk256_key: &str) -> String {
    if let Some(base) = qk256_key.strip_suffix(".qk256_qs") {
        format!("{base}.qk256_scale")
    } else {
        format!("{qk256_key}.qk256_scale")
    }
}

fn qk256_inline_scale(
    raw_tensors: &std::collections::HashMap<String, Tensor>,
    qk256_key: &str,
) -> Result<Option<f32>> {
    let scale_key = qk256_scale_key(qk256_key);
    let Some(scale_tensor) = raw_tensors.get(&scale_key) else {
        return Ok(None);
    };

    let scale_values = scale_tensor.flatten_all()?.to_vec1::<f32>().map_err(|e| {
        BitNetError::Validation(format!("failed to read QK256 inline scale {scale_key}: {e}"))
    })?;
    let [scale] = scale_values.as_slice() else {
        return Err(BitNetError::Validation(format!(
            "QK256 inline scale {scale_key} must contain exactly one value, got {}",
            scale_values.len()
        )));
    };
    let scale = *scale;
    if !scale.is_finite() {
        return Err(BitNetError::Validation(format!(
            "QK256 inline scale {scale_key} is not finite: {scale}"
        )));
    }

    Ok(Some(scale))
}

const TIED_EMBED_QK256_KEY: &str = "embed_tokens.weight.qk256_qs";

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
    rope: Option<RotaryEmbedding>,
    layer_idx: usize, // Layer index for QK256 weight name generation
}

impl MultiHeadAttention {
    pub fn new(config: &BitNetConfig, vb: VarBuilder, layer_idx: usize) -> Result<Self> {
        let hidden_size = config.model.hidden_size;
        let n_heads = config.model.num_heads;
        let head_dim = config.model.attention_head_dim.unwrap_or_else(|| hidden_size / n_heads);

        if config.model.attention_head_dim.is_none() && !hidden_size.is_multiple_of(n_heads) {
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
        let q_out = n_heads * head_dim;
        let kv_out = n_kv_heads * head_dim;

        tracing::info!(
            "layer{}: MultiHeadAttention dims: hidden={}, n_heads={}, n_kv_heads={}, head_dim={}, q_out={}, kv_out={}, group_size={}",
            layer_idx,
            hidden_size,
            n_heads,
            n_kv_heads,
            head_dim,
            q_out,
            kv_out,
            group_size
        );

        tracing::info!(
            "layer{}: About to create linear layers with: q_proj([{}, {}]), k_proj([{}, {}]), v_proj([{}, {}]), o_proj([{}, {}])",
            layer_idx,
            q_out,
            hidden_size,
            kv_out,
            hidden_size,
            kv_out,
            hidden_size,
            hidden_size,
            q_out
        );

        let q_proj = linear_with_optional_bias(hidden_size, q_out, vb.pp("q_proj"))?;
        let k_proj = linear_with_optional_bias(hidden_size, kv_out, vb.pp("k_proj"))?;
        let v_proj = linear_with_optional_bias(hidden_size, kv_out, vb.pp("v_proj"))?;
        let o_proj = linear_with_optional_bias(q_out, hidden_size, vb.pp("o_proj"))?;
        let sub_layernorm = optional_layer_norm_with_optional_bias(
            config.model.norm_type,
            q_out,
            eps_from_config(config),
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
            rope,
            layer_idx,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        kv_cache: Option<&mut LayerKVCache>,
        raw_tensors: &std::collections::HashMap<String, Tensor>,
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
        if qwen_trace_layer_enabled(self.layer_idx) {
            qwen_trace_tensor("attention.q_proj", Some(self.layer_idx), &q_proj_out)?;
            qwen_trace_tensor("attention.k_proj", Some(self.layer_idx), &k_proj_out)?;
            qwen_trace_tensor("attention.v_proj", Some(self.layer_idx), &v_proj_out)?;
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
            let trace_name = format!("t0/blk{}/q_proj", self.layer_idx);
            bitnet_trace::dump_trace(
                &trace_name,
                &q_proj_out,
                Some(0),
                Some(self.layer_idx as isize),
                Some("q_proj"),
            )
            .map_err(BitNetError::from)?;
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
            if qwen_trace_layer_enabled(self.layer_idx) {
                qwen_trace_event(
                    "attention.rope_metadata",
                    &format!(
                        "\"layer\":{},\"position\":{},\"head_dim\":{},\"n_heads\":{},\"n_kv_heads\":{}",
                        self.layer_idx, position, self.head_dim, self.n_heads, self.n_kv_heads
                    ),
                );
                qwen_trace_tensor("attention.q_rope", Some(self.layer_idx), &q_rot)?;
                qwen_trace_tensor("attention.k_rope", Some(self.layer_idx), &k_rot)?;
            }
            (q_rot, k_rot)
        } else {
            (q, k)
        };

        // Update KV cache if provided (store HKV heads, not Hq)
        // **Performance note**: Borrow references instead of cloning after append.
        // Candle operations accept both owned and borrowed tensors.
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

        let scores = q.matmul(&k_expanded.transpose(2, 3)?)?;

        // Convert to fp32 for numerically stable computation
        let scores_f32 = scores.to_dtype(DType::F32)?;

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
        if qwen_trace_layer_enabled(self.layer_idx) {
            qwen_trace_tensor("attention.scores_post_mask", Some(self.layer_idx), &scores_f32)?;
        }

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
        if qwen_trace_layer_enabled(self.layer_idx) {
            qwen_trace_tensor("attention.weights", Some(self.layer_idx), &attn_weights)?;
        }

        // Tracepoint 4: Attention scores post-softmax (layer-specific)
        #[cfg(feature = "trace")]
        {
            let trace_name = format!("t0/blk{}/attn_scores_softmax", self.layer_idx);
            bitnet_trace::dump_trace(
                &trace_name,
                &attn_weights,
                Some(0),
                Some(self.layer_idx as isize),
                Some("attn_scores_softmax"),
            )
            .map_err(BitNetError::from)?;
        }

        // Debug attention weights and row sums
        dbg_stats("attn softmax", &attn_weights)?;
        if std::env::var("DEBUG_ATTN").is_ok() {
            let sums = attn_weights.sum(3)?;
            let sums_host: Vec<f32> = sums.flatten_all()?.to_vec1()?;
            let take = sums_host.iter().take(4).cloned().collect::<Vec<_>>();
            eprintln!("[dbg] attn row-sums (first 4): {:?}", take);
        }

        let attn_output = attn_weights.matmul(&v_expanded)?;
        if qwen_trace_layer_enabled(self.layer_idx) {
            qwen_trace_tensor("attention.output_heads", Some(self.layer_idx), &attn_output)?;
        }

        // Reshape and project output
        let attn_output = attn_output.transpose(1, 2)?.reshape(&[
            batch_size,
            seq_len,
            self.n_heads * self.head_dim,
        ])?;
        let attn_output = if let Some(sub_layernorm) = &self.sub_layernorm {
            let normalized = sub_layernorm.forward(&attn_output)?;
            if qwen_trace_layer_enabled(self.layer_idx) {
                qwen_trace_tensor("attention.sub_layernorm", Some(self.layer_idx), &normalized)?;
            }
            normalized
        } else {
            attn_output
        };

        let projected = self.apply_linear(&attn_output, &self.o_proj, "o_proj", raw_tensors)?;
        if qwen_trace_layer_enabled(self.layer_idx) {
            qwen_trace_tensor("attention.o_proj", Some(self.layer_idx), &projected)?;
        }
        Ok(projected)
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
            tracing::debug!("Using QK256 kernel for {}", qk256_key);
            let inline_scale = qk256_inline_scale(raw_tensors, &qk256_key)?;
            return forward_qk256_with_scale(input, qk256_tensor, &qk256_key, inline_scale);
        }

        if strict_cuda_bitnet_backend_requested() {
            record_bitnet_linear_unsupported();
            return Err(BitNetError::Validation(format!(
                "strict CUDA BitNet linear dispatch requires QK256 raw tensor {}; refusing CPU fallback",
                qk256_key
            )));
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
        record_bitnet_linear_cpu_fallback();
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
    activation_type: ActivationType,
    layer_idx: usize, // Layer index for QK256 weight name generation
}

impl FeedForward {
    pub fn new(config: &BitNetConfig, vb: VarBuilder, layer_idx: usize) -> Result<Self> {
        let hidden_size = config.model.hidden_size;
        let intermediate_size = config.model.intermediate_size;

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
                config.model.norm_type,
                intermediate_size,
                eps_from_config(config),
                vb.pp("sub_layernorm"),
            )?,
            activation_type: config.model.activation_type,
            layer_idx,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        raw_tensors: &std::collections::HashMap<String, Tensor>,
    ) -> Result<Tensor> {
        let gate = self.apply_linear(x, &self.gate_proj, "gate_proj", raw_tensors)?;
        if qwen_trace_layer_enabled(self.layer_idx) {
            qwen_trace_tensor("mlp.gate_proj", Some(self.layer_idx), &gate)?;
        }

        // MLP gating diagnostics (point 3 of user's plan)
        if std::env::var("BITNET_DEBUG_MLP").is_ok()
            && let Ok(u_norm) = gate.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()
        {
            tracing::debug!("MLP ||u|| (gate_proj): {:.6e}", u_norm);
        }

        let gate = self.apply_activation(&gate)?;
        if qwen_trace_layer_enabled(self.layer_idx) {
            qwen_trace_tensor("mlp.gate_activation", Some(self.layer_idx), &gate)?;
        }

        if std::env::var("BITNET_DEBUG_MLP").is_ok()
            && let Ok(activation_norm) = gate.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()
        {
            tracing::debug!("MLP ||activation(u)||: {:.6e}", activation_norm);
        }

        let up = self.apply_linear(x, &self.up_proj, "up_proj", raw_tensors)?;
        if qwen_trace_layer_enabled(self.layer_idx) {
            qwen_trace_tensor("mlp.up_proj", Some(self.layer_idx), &up)?;
        }

        if std::env::var("BITNET_DEBUG_MLP").is_ok()
            && let Ok(v_norm) = up.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()
        {
            tracing::debug!("MLP ||v|| (up_proj): {:.6e}", v_norm);
        }

        let hidden = gate.mul(&up)?;
        if qwen_trace_layer_enabled(self.layer_idx) {
            qwen_trace_tensor("mlp.gated_product", Some(self.layer_idx), &hidden)?;
        }
        let hidden = if let Some(sub_layernorm) = &self.sub_layernorm {
            let normalized = sub_layernorm.forward(&hidden)?;
            if qwen_trace_layer_enabled(self.layer_idx) {
                qwen_trace_tensor("mlp.sub_layernorm", Some(self.layer_idx), &normalized)?;
            }
            normalized
        } else {
            hidden
        };

        if std::env::var("BITNET_DEBUG_MLP").is_ok()
            && let Ok(prod_norm) = hidden.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()
        {
            tracing::debug!("MLP ||silu(u) * v||: {:.6e}", prod_norm);
        }

        let output = self.apply_linear(&hidden, &self.down_proj, "down_proj", raw_tensors)?;
        if qwen_trace_layer_enabled(self.layer_idx) {
            qwen_trace_tensor("mlp.down_proj", Some(self.layer_idx), &output)?;
        }

        if std::env::var("BITNET_DEBUG_MLP").is_ok()
            && let Ok(out_norm) = output.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()
        {
            tracing::debug!("MLP ||W2 * (...)||: {:.6e}", out_norm);
        }

        Ok(output)
    }

    fn apply_activation(&self, input: &Tensor) -> Result<Tensor> {
        apply_ffn_activation(input, self.activation_type)
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
            tracing::debug!("Using QK256 kernel for {}", qk256_key);
            let inline_scale = qk256_inline_scale(raw_tensors, &qk256_key)?;
            return forward_qk256_with_scale(input, qk256_tensor, &qk256_key, inline_scale);
        }

        if strict_cuda_bitnet_backend_requested() {
            record_bitnet_linear_unsupported();
            return Err(BitNetError::Validation(format!(
                "strict CUDA BitNet linear dispatch requires QK256 raw tensor {}; refusing CPU fallback",
                qk256_key
            )));
        }

        // Fall back to standard linear
        tracing::trace!(
            "Using standard linear for layers.{}.feed_forward.{}",
            self.layer_idx,
            proj_name
        );
        record_bitnet_linear_cpu_fallback();
        linear.forward(input).map_err(BitNetError::from)
    }
}

fn apply_ffn_activation(input: &Tensor, activation_type: ActivationType) -> Result<Tensor> {
    match activation_type {
        ActivationType::Silu => input.silu().map_err(BitNetError::from),
        ActivationType::Relu2 => input.relu()?.sqr().map_err(BitNetError::from),
        ActivationType::Gelu => input.gelu_erf().map_err(BitNetError::from),
    }
}

/// Transformer Block
pub struct TransformerBlock {
    attention: MultiHeadAttention,
    feed_forward: FeedForward,
    attention_norm: LayerNorm,
    ffn_norm: LayerNorm,
}

impl TransformerBlock {
    pub fn new(config: &BitNetConfig, vb: VarBuilder, layer_idx: usize) -> Result<Self> {
        let hidden_size = config.model.hidden_size;
        // PATCH 1: Use RMSNorm epsilon from config header for ALL norms (per-layer + final)
        let eps = eps_from_config(config);

        tracing::debug!("TransformerBlock using RMSNorm eps={} (from header)", eps);

        Ok(Self {
            attention: MultiHeadAttention::new(config, vb.pp("attention"), layer_idx)?,
            feed_forward: FeedForward::new(config, vb.pp("feed_forward"), layer_idx)?,
            attention_norm: norm_with_optional_bias(
                config.model.norm_type,
                hidden_size,
                eps,
                vb.pp("attention_norm"),
            )?,
            ffn_norm: norm_with_optional_bias(
                config.model.norm_type,
                hidden_size,
                eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        kv_cache: Option<&mut LayerKVCache>,
        raw_tensors: &std::collections::HashMap<String, Tensor>,
    ) -> Result<Tensor> {
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

        let x = self.attention_norm.forward(x)?;
        if qwen_trace_layer_enabled(self.attention.layer_idx) {
            qwen_trace_tensor("block.attention_norm", Some(self.attention.layer_idx), &x)?;
        }

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
            let trace_name = format!("t0/blk{}/attn_norm", self.attention.layer_idx);
            bitnet_trace::dump_trace(
                &trace_name,
                &x,
                Some(0),
                Some(self.attention.layer_idx as isize),
                Some("attn_norm"),
            )
            .map_err(BitNetError::from)?;
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

        let x = self.attention.forward(&x, kv_cache, raw_tensors)?;
        let x = (x + residual)?;
        if qwen_trace_layer_enabled(self.attention.layer_idx) {
            qwen_trace_tensor("block.post_attention_residual", Some(self.attention.layer_idx), &x)?;
        }

        // Debug post-attention activation norms
        if std::env::var("DEBUG_ATTN").is_ok() {
            let norm = x.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
            eprintln!("[norm] post-attn: {norm:.6e}");
        }

        // Pre-norm FFN
        let residual = &x;

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

        let x = self.ffn_norm.forward(&x)?;
        if qwen_trace_layer_enabled(self.attention.layer_idx) {
            qwen_trace_tensor("block.ffn_norm", Some(self.attention.layer_idx), &x)?;
        }

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

        let x = self.feed_forward.forward(&x, raw_tensors)?;
        let x = (x + residual)?;
        if qwen_trace_layer_enabled(self.attention.layer_idx) {
            qwen_trace_tensor("block.output", Some(self.attention.layer_idx), &x)?;
        }

        // Debug post-FFN activation norms
        if std::env::var("DEBUG_ATTN").is_ok() {
            let norm = x.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
            eprintln!("[norm] post-ffn: {norm:.6e}");
        }

        Ok(x)
    }
}

fn eps_from_config(config: &BitNetConfig) -> f64 {
    config.model.rms_norm_eps.map(|e| e as f64).unwrap_or(1e-5)
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
        if config.model.attention_head_dim.is_none() && !hidden_size.is_multiple_of(n_heads) {
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

        let head_dim = config.model.attention_head_dim.unwrap_or_else(|| hidden_size / n_heads);
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
            layers.push(TransformerBlock::new(&config, vb.pp(format!("layers.{}", i)), i)?);
        }

        // Use RMSNorm epsilon from config header (CRITICAL: must match per-layer norms)
        let eps = config.model.rms_norm_eps.map(|e| e as f64).unwrap_or(1e-5);
        tracing::info!("Final norm using RMSNorm eps={} (from header)", eps);

        let norm =
            norm_with_optional_bias(config.model.norm_type, hidden_size, eps, vb.pp("final_norm"))?;

        // Try to load lm_head, but it's optional (can be tied to embeddings)
        // Try to create the linear layer, catching errors if weights don't exist
        let (lm_head, lm_head_weight, lm_head_transposed) = match linear_with_optional_bias(
            hidden_size,
            vocab_size,
            vb.pp("lm_head"),
        ) {
            Ok(layer) => {
                // Also get the weight tensor directly for transposed handling
                // Note: weight dimensions might be transposed
                let weight = vb
                    .get((vocab_size, hidden_size), "lm_head.weight")
                    .or_else(|_| vb.get((hidden_size, vocab_size), "lm_head.weight"))
                    .ok();

                // Read transpose flag for lm_head
                let transposed = match vb.get((1,), "lm_head.transposed") {
                    Ok(t) => {
                        let vals = t.to_vec1::<f32>()?;
                        vals.first().copied().unwrap_or(0.0) > 0.5
                    }
                    Err(_) => false, // If flag doesn't exist, assume not transposed
                };

                if transposed {
                    tracing::info!(
                        "LM head is transposed [hidden, vocab] - will handle efficiently at runtime"
                    );
                }
                (Some(layer), weight, transposed)
            }
            Err(_) => match vb.get((hidden_size, vocab_size), "lm_head.weight") {
                Ok(weight) => {
                    tracing::info!(
                        "LM head is stored transposed [hidden, vocab] - using direct matmul path"
                    );
                    (None, Some(weight), true)
                }
                Err(_) => {
                    tracing::info!("lm_head.weight not found, will use tied weights");
                    (None, None, false)
                }
            },
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
        qwen_trace_event(
            "model_config",
            &format!(
                "\"vocab_size\":{},\"hidden_size\":{},\"layers\":{},\"heads\":{},\"kv_heads\":{},\"norm_type\":\"{:?}\",\"rms_norm_eps\":{},\"rope_theta\":{},\"embed_transposed\":{},\"lm_head_present\":{},\"lm_head_weight_present\":{},\"lm_head_transposed\":{}",
                vocab_size,
                hidden_size,
                n_layers,
                config.model.num_heads,
                config.model.num_key_value_heads,
                config.model.norm_type,
                config
                    .model
                    .rms_norm_eps
                    .map(|value| qwen_trace_number(value as f64))
                    .unwrap_or_else(|| "null".to_string()),
                config
                    .model
                    .rope_theta
                    .map(|value| qwen_trace_number(value as f64))
                    .unwrap_or_else(|| "null".to_string()),
                embed_transposed,
                lm_head.is_some(),
                lm_head_weight.is_some(),
                lm_head_transposed
            ),
        );

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
            use bitnet_trace::dump_trace;
            // Extract first token's embedding for tracing [B, 1, H]
            let first_token_emb = hidden.narrow(1, 0, 1)?;
            let _ = dump_trace(
                "embeddings",
                &first_token_emb,
                Some(0),            // seq=0 (prefill step)
                Some(-1),           // layer=-1 (pre-layer operation)
                Some("embeddings"), // stage name
            );
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
                use bitnet_trace::dump_trace;
                let _ = dump_trace(
                    &format!("t{}_all_layers_out", t),
                    &step_hidden,
                    Some(t),                // seq=t (current position)
                    Some(-2),               // layer=-2 (post-all-layers)
                    Some("all_layers_out"), // stage name
                );
            }

            // Project to vocabulary logits for this step.
            let step_logits = self.logits(&step_hidden)?;

            // Trace logits for this position
            #[cfg(feature = "trace")]
            {
                use bitnet_trace::dump_trace;
                let _ = dump_trace(
                    &format!("t{}_logits", t),
                    &step_logits,
                    Some(t),        // seq=t (current position)
                    Some(-1),       // layer=-1 (post-layers stage)
                    Some("logits"), // stage name
                );
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
            // Extract first token's logits for tracing [B, 1, V]
            let first_token_logits = logits.narrow(1, 0, 1)?;
            bitnet_trace::dump_trace(
                "t0/logits",
                &first_token_logits,
                Some(0),
                Some(-1),
                Some("logits"),
            )
            .map_err(BitNetError::from)?;
        }

        Ok(logits)
    }

    /// Forward pass through transformer layers
    ///
    /// **Performance note**: Accepts ownership of `hidden` to avoid cloning on hot path.
    /// Caller should pass owned tensor or use `.clone()` explicitly if needed.
    pub fn forward(&self, hidden: Tensor, mut kv_cache: Option<&mut KVCache>) -> Result<Tensor> {
        let mut x = hidden; // Take ownership - no clone needed!

        // Tracepoint 1: Embeddings (incremental path - single token)
        // This captures the embedding for the current token being processed
        #[cfg(feature = "trace")]
        {
            // For incremental path, hidden is already [B, H] (single token)
            // Trace it directly without narrowing (unlike forward_full which has [B, T, H])
            bitnet_trace::dump_trace("t0/embeddings", &x, Some(0), Some(-1), Some("embeddings"))
                .map_err(BitNetError::from)?;
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

        let normalized = self.norm.forward(&x)?;
        qwen_trace_tensor("model.final_norm", None, &normalized)?;
        if std::env::var("DEBUG_ATTN").is_ok()
            && let Ok(norm) = normalized.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()
        {
            eprintln!("[norm] final: {:.6e}", norm);
        }

        Ok(normalized)
    }

    pub fn logits(&self, hidden: &Tensor) -> Result<Tensor> {
        let vocab_size = self.config.model.vocab_size;
        let has_tied_qk256_output = self.raw_tensors.contains_key(TIED_EMBED_QK256_KEY);
        qwen_trace_tensor("lm_head.input_hidden", None, hidden)?;
        qwen_trace_event(
            "lm_head.metadata",
            &format!(
                "\"lm_head_present\":{},\"lm_head_transposed\":{},\"embed_transposed\":{},\"has_cached_tied_weight\":{},\"has_tied_qk256_output\":{}",
                self.lm_head.is_some(),
                self.lm_head_transposed,
                self.embed_transposed,
                self.embed_tied_weight.is_some(),
                has_tied_qk256_output
            ),
        );

        match hidden.rank() {
            2 => {
                // [B, H] - last token only
                let (b, _h) = (hidden.dims()[0], hidden.dims()[1]);

                let logits = if self.lm_head_transposed {
                    if let Some(ref weight) = self.lm_head_weight {
                        hidden.matmul(weight)?.reshape(&[b, vocab_size])?
                    } else if let Some(ref lm_head) = self.lm_head {
                        let logits = lm_head.forward(hidden)?; // [B, V]
                        logits.reshape(&[b, vocab_size])?
                    } else {
                        return Err(BitNetError::Validation(
                            "lm_head is marked transposed but lm_head.weight is unavailable".into(),
                        ));
                    }
                } else if let Some(ref lm_head) = self.lm_head {
                    // Use dedicated LM head if available
                    let logits = lm_head.forward(hidden)?; // [B, V]
                    logits.reshape(&[b, vocab_size])?
                } else if let Some(qk256_tensor) = self.raw_tensors.get(TIED_EMBED_QK256_KEY) {
                    static LOGGED_QK256_TIED: std::sync::Once = std::sync::Once::new();
                    LOGGED_QK256_TIED.call_once(|| {
                        tracing::info!(
                            "LM head tied to raw QK256 token embeddings for BitNet.cpp parity"
                        );
                    });
                    let inline_scale = qk256_inline_scale(&self.raw_tensors, TIED_EMBED_QK256_KEY)?;
                    forward_qk256_with_scale(
                        hidden,
                        qk256_tensor,
                        TIED_EMBED_QK256_KEY,
                        inline_scale,
                    )?
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
                qwen_trace_tensor("lm_head.logits", None, &logits)?;

                // Tracepoint 5: Logits (incremental path - single token)
                // This captures the final logits for the current token [B, V]
                #[cfg(feature = "trace")]
                {
                    // For incremental path, logits are [B, V] (single token)
                    // Trace directly without narrowing (unlike forward_full which has [B, T, V])
                    bitnet_trace::dump_trace(
                        "t0/logits",
                        &logits,
                        Some(0),
                        Some(-1),
                        Some("logits"),
                    )
                    .map_err(BitNetError::from)?;
                }

                Ok(logits)
            }
            3 => {
                // [B, T, H] - all timesteps
                let (b, t, h) = (hidden.dims()[0], hidden.dims()[1], hidden.dims()[2]);

                if self.lm_head_transposed {
                    if let Some(ref weight) = self.lm_head_weight {
                        // LM head weight is stored as [hidden, vocab].
                        let hidden_2d = hidden.reshape(&[b * t, h])?;
                        let logits_2d = hidden_2d.matmul(weight)?;
                        Ok(logits_2d.reshape(&[b, t, vocab_size])?)
                    } else if let Some(ref lm_head) = self.lm_head {
                        let hidden_2d = hidden.reshape(&[b * t, h])?;
                        let logits_2d = lm_head.forward(&hidden_2d)?;
                        Ok(logits_2d.reshape(&[b, t, vocab_size])?)
                    } else {
                        Err(BitNetError::Validation(
                            "lm_head is marked transposed but lm_head.weight is unavailable".into(),
                        ))
                    }
                } else if let Some(ref lm_head) = self.lm_head {
                    // Use dedicated LM head if available
                    // Standard path: LM head weight is [vocab, hidden]
                    // Flatten to 2D for proper matmul
                    let hidden_2d = hidden.reshape(&[b * t, h])?;
                    let logits_2d = lm_head.forward(&hidden_2d)?;
                    Ok(logits_2d.reshape(&[b, t, vocab_size])?)
                } else if let Some(qk256_tensor) = self.raw_tensors.get(TIED_EMBED_QK256_KEY) {
                    static LOGGED_QK256_TIED: std::sync::Once = std::sync::Once::new();
                    LOGGED_QK256_TIED.call_once(|| {
                        tracing::info!(
                            "LM head tied to raw QK256 token embeddings for BitNet.cpp parity"
                        );
                    });
                    let inline_scale = qk256_inline_scale(&self.raw_tensors, TIED_EMBED_QK256_KEY)?;
                    let logits = forward_qk256_with_scale(
                        hidden,
                        qk256_tensor,
                        TIED_EMBED_QK256_KEY,
                        inline_scale,
                    )?;
                    Ok(logits.reshape(&[b, t, vocab_size])?)
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
    use bitnet_common::config::ModelConfig;
    use candle_nn::RmsNorm;
    use serial_test::serial;
    use std::collections::HashMap;

    /// Helper to compute RMS (root mean square) of a tensor
    fn compute_rms(tensor: &Tensor) -> candle_core::Result<f64> {
        let squared = tensor.sqr()?;
        let mean = squared.mean_all()?;
        let rms = mean.sqrt()?.to_scalar::<f32>()? as f64;
        Ok(rms)
    }

    #[test]
    fn test_relu2_activation_squares_positive_values() -> Result<()> {
        let device = Device::Cpu;
        let input = Tensor::from_slice(&[-2.0f32, -0.5, 0.0, 2.0, 3.0], (5,), &device)?;
        let output = apply_ffn_activation(&input, ActivationType::Relu2)?;
        let values = output.to_vec1::<f32>()?;

        assert_eq!(values, vec![0.0, 0.0, 0.0, 4.0, 9.0]);
        Ok(())
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
    #[serial]
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
        let layer_norm = layer_norm_with_optional_bias(hidden_size, eps, vb)?;

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
    #[serial]
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

        let err = layer_norm_with_optional_bias(hidden_size, eps, vb)
            .expect_err("missing bias should error when BITNET_REQUIRE_LAYER_NORM_BIAS=1");
        assert!(
            err.to_string().contains("LayerNorm bias tensor is required"),
            "unexpected error: {err}"
        );

        Ok(())
    }

    fn tiny_bitnet_config() -> BitNetConfig {
        BitNetConfig {
            model: ModelConfig {
                hidden_size: 2,
                vocab_size: 8,
                num_heads: 1,
                num_key_value_heads: 1,
                num_layers: 1,
                intermediate_size: 2,
                max_position_embeddings: 8,
                rms_norm_eps: Some(1e-5),
                norm_type: NormType::LayerNorm,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn identity_2(device: &Device) -> candle_core::Result<Tensor> {
        Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], &[2, 2], device)
    }

    #[test]
    #[serial]
    fn attention_applies_bitnet_sub_layernorm_before_output_projection() -> Result<()> {
        let device = Device::Cpu;
        let config = tiny_bitnet_config();
        let mut tensors = HashMap::new();
        for name in ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"] {
            tensors.insert(name.to_string(), identity_2(&device)?);
        }
        tensors.insert("sub_layernorm.weight".to_string(), Tensor::ones(2, DType::F32, &device)?);
        tensors.insert("sub_layernorm.bias".to_string(), Tensor::zeros(2, DType::F32, &device)?);

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let attention = MultiHeadAttention::new(&config, vb, 0)?;
        let x = Tensor::from_vec(vec![1.0f32, 3.0], &[1, 1, 2], &device)?;
        let output = attention.forward(&x, None, &HashMap::new())?;
        let values: Vec<f32> = output.flatten_all()?.to_vec1()?;

        assert!(
            values[0] < -0.99 && values[0] > -1.01,
            "attention sub-layernorm should center first value near -1, got {}",
            values[0]
        );
        assert!(
            values[1] > 0.99 && values[1] < 1.01,
            "attention sub-layernorm should center second value near 1, got {}",
            values[1]
        );

        Ok(())
    }

    #[test]
    #[serial]
    fn feed_forward_applies_bitnet_sub_layernorm_before_down_projection() -> Result<()> {
        let device = Device::Cpu;
        let config = tiny_bitnet_config();
        let mut tensors = HashMap::new();
        for name in ["gate_proj.weight", "up_proj.weight", "down_proj.weight"] {
            tensors.insert(name.to_string(), identity_2(&device)?);
        }
        tensors.insert("sub_layernorm.weight".to_string(), Tensor::ones(2, DType::F32, &device)?);
        tensors.insert("sub_layernorm.bias".to_string(), Tensor::zeros(2, DType::F32, &device)?);

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let feed_forward = FeedForward::new(&config, vb, 0)?;
        let x = Tensor::from_vec(vec![1.0f32, 2.0], &[1, 1, 2], &device)?;
        let output = feed_forward.forward(&x, &HashMap::new())?;
        let values: Vec<f32> = output.flatten_all()?.to_vec1()?;

        assert!(
            values[0] < -0.99 && values[0] > -1.01,
            "feed-forward sub-layernorm should center first value near -1, got {}",
            values[0]
        );
        assert!(
            values[1] > 0.99 && values[1] < 1.01,
            "feed-forward sub-layernorm should center second value near 1, got {}",
            values[1]
        );

        Ok(())
    }

    #[test]
    fn feed_forward_uses_relu2_activation_when_configured() -> Result<()> {
        let device = Device::Cpu;
        let mut config = tiny_bitnet_config();
        config.model.activation_type = ActivationType::Relu2;
        let mut tensors = HashMap::new();
        for name in ["gate_proj.weight", "up_proj.weight", "down_proj.weight"] {
            tensors.insert(name.to_string(), identity_2(&device)?);
        }

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let feed_forward = FeedForward::new(&config, vb, 0)?;
        let x = Tensor::from_vec(vec![1.0f32, 2.0], &[1, 1, 2], &device)?;
        let output = feed_forward.forward(&x, &HashMap::new())?;
        let values: Vec<f32> = output.flatten_all()?.to_vec1()?;

        assert!(
            (values[0] - 1.0).abs() < 1e-5,
            "relu2 gate should produce first output 1, got {}",
            values[0]
        );
        assert!(
            (values[1] - 8.0).abs() < 1e-5,
            "relu2 gate should square ReLU before multiplying by up projection, got {}",
            values[1]
        );

        Ok(())
    }

    #[test]
    fn qk256_inline_scale_reads_sibling_raw_tensor() -> Result<()> {
        let device = Device::Cpu;
        let mut raw_tensors = HashMap::new();
        raw_tensors.insert(
            "layers.0.attention.q_proj.weight.qk256_scale".to_string(),
            Tensor::from_vec(vec![0.25f32], &[1], &device)?,
        );

        let scale = qk256_inline_scale(&raw_tensors, "layers.0.attention.q_proj.weight.qk256_qs")?;

        assert_eq!(scale, Some(0.25));

        Ok(())
    }

    #[test]
    fn logits_prefer_raw_qk256_tied_embeddings_when_present() -> Result<()> {
        let device = Device::Cpu;
        let mut config = BitNetConfig::default();
        config.model.hidden_size = 256;
        config.model.vocab_size = 2;
        config.model.num_layers = 0;
        config.model.num_heads = 1;
        config.model.num_key_value_heads = 1;
        config.model.intermediate_size = 256;
        config.model.rms_norm_eps = Some(1e-5);
        config.model.norm_type = NormType::RmsNorm;

        let mut tensors = HashMap::new();
        tensors.insert(
            "embed_tokens.weight".to_string(),
            Tensor::zeros((2, 256), DType::F32, &device)?,
        );
        tensors.insert("final_norm.weight".to_string(), Tensor::ones(256, DType::F32, &device)?);

        let mut raw_tensors = HashMap::new();
        let mut packed = vec![0x00u8; 64];
        packed.extend(std::iter::repeat_n(0xAAu8, 64));
        raw_tensors.insert(
            TIED_EMBED_QK256_KEY.to_string(),
            Tensor::from_raw_buffer(&packed, DType::U8, &[2, 64], &device)?,
        );
        raw_tensors.insert(
            "embed_tokens.weight.qk256_scale".to_string(),
            Tensor::from_vec(vec![1.0f32], &[1], &device)?,
        );

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let model = TransformerModel::new_with_tensors(config, vb, raw_tensors)?;
        let hidden = Tensor::ones((1, 256), DType::F32, &device)?;
        let logits = model.logits(&hidden)?.to_vec2::<f32>()?;

        assert!(
            logits[0][0] < -200.0,
            "first raw QK256 tied logit should come from packed code-0 row, got {}",
            logits[0][0]
        );
        assert!(
            logits[0][1] > 200.0,
            "second raw QK256 tied logit should come from packed code-2 row, got {}",
            logits[0][1]
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

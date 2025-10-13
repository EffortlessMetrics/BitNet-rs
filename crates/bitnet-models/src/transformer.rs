use bitnet_common::{BitNetConfig, BitNetError, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{LayerNorm, Linear, VarBuilder};

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
/// If `bias` is missing we fall back to RMSNorm (no bias).
fn layer_norm_with_optional_bias(
    normalized_shape: usize,
    eps: f64,
    vb: VarBuilder,
) -> candle_core::Result<LayerNorm> {
    let weight = vb.get((normalized_shape,), "weight")?;
    match vb.get((normalized_shape,), "bias") {
        Ok(bias) => {
            // Bias exists → standard LayerNorm
            tracing::debug!("Using LayerNorm with bias [{}]", normalized_shape);
            Ok(LayerNorm::new(weight, bias, eps))
        }
        Err(_) => {
            // No bias → RMSNorm
            tracing::debug!(
                "Bias tensor missing for norm layer; using RMSNorm (no bias) [{}]",
                normalized_shape
            );
            Ok(LayerNorm::rms_norm(weight, eps))
        }
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
        let theta = rope_theta.unwrap_or(10000.0);
        let freqs = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / theta.powf(i as f32 / dim as f32))
            .collect::<Vec<_>>();

        let positions = (0..max_seq_len).map(|i| i as f32).collect::<Vec<_>>();
        let mut sin_vals = Vec::with_capacity(max_seq_len * dim / 2);
        let mut cos_vals = Vec::with_capacity(max_seq_len * dim / 2);

        for pos in &positions {
            for &freq in &freqs {
                let angle = pos * freq;
                sin_vals.push(angle.sin());
                cos_vals.push(angle.cos());
            }
        }

        let sin = Tensor::from_vec(sin_vals, &[max_seq_len, dim / 2], device)?;
        let cos = Tensor::from_vec(cos_vals, &[max_seq_len, dim / 2], device)?;

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

            // Reshape to separate real and imaginary parts
            let x_reshaped = x.reshape(&[batch, n_heads, seq_len, half_dim, 2])?;
            let x0 = x_reshaped.narrow(4, 0, 1)?.squeeze(4)?;
            let x1 = x_reshaped.narrow(4, 1, 1)?.squeeze(4)?;

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

            let rotated = Tensor::stack(&[x0_rot, x1_rot], 4)?
                .reshape(&[batch, n_heads, seq_len, head_dim])?;

            Ok(rotated)
        } else {
            // Original 3D implementation for other uses
            let (_batch, _seq, dim) = x.dims3()?;
            let half_dim = dim / 2;

            let x_reshaped = x.reshape(&[x.dims()[0], x.dims()[1], half_dim, 2])?;
            let x0 = x_reshaped.narrow(3, 0, 1)?.squeeze(3)?;
            let x1 = x_reshaped.narrow(3, 1, 1)?.squeeze(3)?;

            let cos = self.cos.narrow(0, position, 1)?;
            let sin = self.sin.narrow(0, position, 1)?;

            let x0_rot = (x0.mul(&cos)? - x1.mul(&sin)?)?;
            let x1_rot = (x0.mul(&sin)? + x1.mul(&cos)?)?;

            let rotated =
                Tensor::stack(&[x0_rot, x1_rot], 3)?.reshape(&[x.dims()[0], x.dims()[1], dim])?;

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
    rope: Option<RotaryEmbedding>,
}

impl MultiHeadAttention {
    pub fn new(config: &BitNetConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = config.model.hidden_size;
        let n_heads = config.model.num_heads;
        let head_dim = hidden_size / n_heads;

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

        let q_proj = linear_with_optional_bias(hidden_size, hidden_size, vb.pp("q_proj"))?;
        let k_proj = linear_with_optional_bias(hidden_size, kv_out, vb.pp("k_proj"))?;
        let v_proj = linear_with_optional_bias(hidden_size, kv_out, vb.pp("v_proj"))?;
        let o_proj = linear_with_optional_bias(hidden_size, hidden_size, vb.pp("o_proj"))?;

        let rope = RotaryEmbedding::new(
            head_dim,
            config.model.max_position_embeddings,
            config.model.rope_theta,
            vb.device(),
        )
        .ok();

        Ok(Self { n_heads, n_kv_heads, head_dim, group_size, q_proj, k_proj, v_proj, o_proj, rope })
    }

    pub fn forward(&self, x: &Tensor, kv_cache: Option<&mut LayerKVCache>) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;

        // PATCH 3: Project to Q, K, V separately (NOT fused QKV)
        // This is the correct implementation - separate projections ensure proper shape handling
        // Q: [B, T, hidden] -> [B, T, n_heads * head_dim] -> [B, n_heads, T, head_dim]
        // K: [B, T, hidden] -> [B, T, n_kv_heads * head_dim] -> [B, n_kv_heads, T, head_dim]
        // V: [B, T, hidden] -> [B, T, n_kv_heads * head_dim] -> [B, n_kv_heads, T, head_dim]
        let q = self
            .q_proj
            .forward(x)?
            .reshape(&[batch_size, seq_len, self.n_heads, self.head_dim])?
            .transpose(1, 2)?; // [B, Hq, T, D]

        let k = self
            .k_proj
            .forward(x)?
            .reshape(&[batch_size, seq_len, self.n_kv_heads, self.head_dim])?
            .transpose(1, 2)?; // [B, HKV, T, D]

        let v = self
            .v_proj
            .forward(x)?
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

        // Update KV cache if provided (store HKV heads, not Hq)
        let (k_ctx, v_ctx) = if let Some(cache) = kv_cache {
            cache.append(&k, &v)?;
            (cache.k.clone(), cache.v.clone())
        } else {
            (k, v)
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

        // Debug attention weights and row sums
        dbg_stats("attn softmax", &attn_weights)?;
        if std::env::var("DEBUG_ATTN").is_ok() {
            let sums = attn_weights.sum(3)?;
            let sums_host: Vec<f32> = sums.flatten_all()?.to_vec1()?;
            let take = sums_host.iter().take(4).cloned().collect::<Vec<_>>();
            eprintln!("[dbg] attn row-sums (first 4): {:?}", take);
        }

        let attn_output = attn_weights.matmul(&v_expanded)?;

        // Reshape and project output
        let attn_output = attn_output.transpose(1, 2)?.reshape(&[
            batch_size,
            seq_len,
            self.n_heads * self.head_dim,
        ])?;

        Ok(self.o_proj.forward(&attn_output)?)
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
}

impl FeedForward {
    pub fn new(config: &BitNetConfig, vb: VarBuilder) -> Result<Self> {
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
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;

        // MLP gating diagnostics (point 3 of user's plan)
        if std::env::var("BITNET_DEBUG_MLP").is_ok()
            && let Ok(u_norm) = gate.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()
        {
            tracing::debug!("MLP ||u|| (gate_proj): {:.6e}", u_norm);
        }

        let gate = candle_nn::ops::silu(&gate)?;

        if std::env::var("BITNET_DEBUG_MLP").is_ok()
            && let Ok(silu_norm) = gate.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()
        {
            tracing::debug!("MLP ||silu(u)||: {:.6e}", silu_norm);
        }

        let up = self.up_proj.forward(x)?;

        if std::env::var("BITNET_DEBUG_MLP").is_ok()
            && let Ok(v_norm) = up.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()
        {
            tracing::debug!("MLP ||v|| (up_proj): {:.6e}", v_norm);
        }

        let hidden = gate.mul(&up)?;

        if std::env::var("BITNET_DEBUG_MLP").is_ok()
            && let Ok(prod_norm) = hidden.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()
        {
            tracing::debug!("MLP ||silu(u) * v||: {:.6e}", prod_norm);
        }

        let output = self.down_proj.forward(&hidden)?;

        if std::env::var("BITNET_DEBUG_MLP").is_ok()
            && let Ok(out_norm) = output.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()
        {
            tracing::debug!("MLP ||W2 * (...)||: {:.6e}", out_norm);
        }

        Ok(output)
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
    pub fn new(config: &BitNetConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = config.model.hidden_size;
        // PATCH 1: Use RMSNorm epsilon from config header for ALL norms (per-layer + final)
        let eps = config.model.rms_norm_eps.map(|e| e as f64).unwrap_or(1e-5);

        tracing::debug!("TransformerBlock using RMSNorm eps={} (from header)", eps);

        Ok(Self {
            attention: MultiHeadAttention::new(config, vb.pp("attention"))?,
            feed_forward: FeedForward::new(config, vb.pp("feed_forward"))?,
            attention_norm: layer_norm_with_optional_bias(
                hidden_size,
                eps,
                vb.pp("attention_norm"),
            )?,
            ffn_norm: layer_norm_with_optional_bias(
                hidden_size,
                eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    pub fn forward(&self, x: &Tensor, kv_cache: Option<&mut LayerKVCache>) -> Result<Tensor> {
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

        let x = self.attention.forward(&x, kv_cache)?;
        let x = (x + residual)?;

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

        let x = self.feed_forward.forward(&x)?;
        let x = (x + residual)?;

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
            // First append - just store the tensors
            self.k = k_new.clone();
            self.v = v_new.clone();
        } else {
            // Concatenate along time dimension (dim=2)
            if self.seq_len + new_seq_len > self.max_seq_len {
                return Err(BitNetError::from(candle_core::Error::Msg(
                    "KV cache overflow".to_string(),
                )));
            }
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
        let n_kv_heads = config.model.num_key_value_heads.max(1).min(n_heads);
        let head_dim = config.model.hidden_size / n_heads;
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
}

impl TransformerModel {
    pub fn new(config: BitNetConfig, vb: VarBuilder) -> Result<Self> {
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
            layers.push(TransformerBlock::new(&config, vb.pp(format!("layers.{}", i)))?);
        }

        // Use RMSNorm epsilon from config header (CRITICAL: must match per-layer norms)
        let eps = config.model.rms_norm_eps.map(|e| e as f64).unwrap_or(1e-5);
        tracing::info!("Final norm using RMSNorm eps={} (from header)", eps);

        let norm = layer_norm_with_optional_bias(hidden_size, eps, vb.pp("final_norm"))?;

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
            Err(_) => {
                tracing::info!("lm_head.weight not found, will use tied weights");
                (None, None, false)
            }
        };

        // PATCH 2: Optimize tied weights by pre-transposing embeddings once at load
        // NOTE: embed_tokens.embeddings() ALWAYS returns [V,H] (Candle's internal format)
        // regardless of how they were stored in GGUF. We need [H,V] for tied weights.
        let (embed_transposed, embed_tied_weight) = if lm_head.is_none() {
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

        // Create per-layer KV cache so that rotary/absolute positional
        // encodings use the proper positions during iterative decoding.
        let mut kv_cache = KVCache::new(&self.config, batch_size, &self.device)?;

        // Collect logits for each position.
        let mut logits_steps = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            // Select the current token's embedding: [B,1,H]
            let step_hidden = hidden.narrow(1, t, 1)?;

            // Run through all layers using the incremental path which applies
            // positional encoding per layer and causal masking internally.
            let step_hidden = self.forward(&step_hidden, Some(&mut kv_cache))?;

            // Project to vocabulary logits for this step.
            let step_logits = self.logits(&step_hidden)?;
            logits_steps.push(step_logits);
        }

        // Concatenate logits from all steps: [B,T,V]
        Ok(Tensor::cat(&logits_steps, 1)?)
    }

    pub fn forward(&self, hidden: &Tensor, mut kv_cache: Option<&mut KVCache>) -> Result<Tensor> {
        let mut x = hidden.clone();

        // Debug input activation norm
        if std::env::var("DEBUG_ATTN").is_ok()
            && let Ok(norm) = x.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()
        {
            eprintln!("[norm] input: {:.6e}", norm);
        }

        for (i, layer) in self.layers.iter().enumerate() {
            let layer_cache = kv_cache.as_mut().and_then(|c| c.layer_mut(i));
            x = layer.forward(&x, layer_cache)?;

            // Debug layer activation norms (show all layers when debugging)
            if std::env::var("DEBUG_ATTN").is_ok()
                && let Ok(norm) = x.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()
            {
                eprintln!("[norm] layer {i}: {:.6e}", norm);
            }
        }

        let normalized = self.norm.forward(&x)?;
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

                let logits = if let Some(ref lm_head) = self.lm_head {
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

                Ok(logits)
            }
            3 => {
                // [B, T, H] - all timesteps
                let (b, t, h) = (hidden.dims()[0], hidden.dims()[1], hidden.dims()[2]);

                if let Some(ref lm_head) = self.lm_head {
                    // Use dedicated LM head if available
                    if self.lm_head_transposed {
                        if let Some(ref weight) = self.lm_head_weight {
                            // LM head weight is stored as [hidden, vocab]
                            // Flatten to 2D so Candle matmul is happy
                            let hidden_2d = hidden.reshape(&[b * t, h])?;
                            let logits_2d = hidden_2d.matmul(weight)?;
                            Ok(logits_2d.reshape(&[b, t, vocab_size])?)
                        } else {
                            // Fallback to standard forward if we couldn't get weight directly
                            let hidden_2d = hidden.reshape(&[b * t, h])?;
                            let logits_2d = lm_head.forward(&hidden_2d)?;
                            Ok(logits_2d.reshape(&[b, t, vocab_size])?)
                        }
                    } else {
                        // Standard path: LM head weight is [vocab, hidden]
                        // Flatten to 2D for proper matmul
                        let hidden_2d = hidden.reshape(&[b * t, h])?;
                        let logits_2d = lm_head.forward(&hidden_2d)?;
                        Ok(logits_2d.reshape(&[b, t, vocab_size])?)
                    }
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

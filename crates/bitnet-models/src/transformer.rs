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
            // Bias exists → standard LayerNorm (with mean subtraction and bias)
            tracing::debug!("Using LayerNorm with bias [{}]", normalized_shape);
            Ok(LayerNorm::new(weight, bias, eps))
        }
        Err(_) => {
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
    layer_idx: usize, // Layer index for QK256 weight name generation
}

impl MultiHeadAttention {
    pub fn new(config: &BitNetConfig, vb: VarBuilder, layer_idx: usize) -> Result<Self> {
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
            bitnet_trace::dump_trace(&trace_name, &q_proj_out).map_err(BitNetError::from)?;
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

        // Tracepoint 4: Attention scores post-softmax (layer-specific)
        #[cfg(feature = "trace")]
        {
            let trace_name = format!("t0/blk{}/attn_scores_softmax", self.layer_idx);
            bitnet_trace::dump_trace(&trace_name, &attn_weights).map_err(BitNetError::from)?;
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

        // Reshape and project output
        let attn_output = attn_output.transpose(1, 2)?.reshape(&[
            batch_size,
            seq_len,
            self.n_heads * self.head_dim,
        ])?;

        self.apply_linear(&attn_output, &self.o_proj, "o_proj", raw_tensors)
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
            return Self::forward_qk256(input, qk256_tensor, &qk256_key);
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

    /// Forward pass using QK256 kernel (static method)
    fn forward_qk256(input: &Tensor, qk256_tensor: &Tensor, weight_name: &str) -> Result<Tensor> {
        use crate::quant::i2s_qk256::gemv_qk256;

        // Extract dimensions
        let dims = qk256_tensor.dims();
        if dims.len() != 2 {
            return Err(BitNetError::Validation(format!(
                "QK256 tensor {} has invalid shape: {:?}",
                weight_name, dims
            )));
        }

        let rows = dims[0];
        let row_stride_bytes = dims[1];

        // Calculate cols from row_stride_bytes: each 256-element block uses 64 bytes
        // So: row_stride_bytes / 64 = number of 256-element blocks
        // And: cols = (row_stride_bytes / 64) * 256
        // Safe calculation: (bytes/64)*256 == bytes*4, with overflow check
        debug_assert!(
            row_stride_bytes.is_multiple_of(64),
            "QK256 row_stride_bytes must be multiple of 64"
        );
        let cols = row_stride_bytes
            .checked_mul(4) // (bytes/64)*256 == bytes*4
            .ok_or_else(|| {
                BitNetError::Validation(format!(
                    "QK256: row_stride_bytes overflow computing cols (row_stride={})",
                    row_stride_bytes
                ))
            })?;

        // Extract bytes
        let bytes_2d = qk256_tensor.to_vec2::<u8>().map_err(|e| {
            BitNetError::Validation(format!(
                "Failed to extract QK256 bytes for {}: {}",
                weight_name, e
            ))
        })?;
        let mut flat_bytes = Vec::with_capacity(rows * row_stride_bytes);
        for row in bytes_2d {
            flat_bytes.extend_from_slice(&row);
        }

        // Get input dimensions
        let input_dims = input.dims();
        let rank = input_dims.len();

        // Handle different input shapes: [B, T, H] or [B, H]
        let (batch_size, seq_len, input_cols) = match rank {
            3 => (input_dims[0], input_dims[1], input_dims[2]),
            2 => (input_dims[0], 1, input_dims[1]),
            _ => {
                return Err(BitNetError::Validation(format!(
                    "Unsupported input shape for QK256: {:?}",
                    input_dims
                )));
            }
        };

        // Validate input dimensions match QK256 tensor
        if input_cols != cols {
            return Err(BitNetError::Validation(format!(
                "QK256 dimension mismatch for {}: input has {} cols but QK256 tensor expects {} cols",
                weight_name, input_cols, cols
            )));
        }

        // Flatten to 2D [batch * seq_len, in_features]
        let input_flat = input.reshape(&[batch_size * seq_len, cols])?;
        let input_vec = input_flat.to_vec2::<f32>().map_err(|e| {
            BitNetError::Validation(format!(
                "Failed to convert input to f32 for {}: {}",
                weight_name, e
            ))
        })?;

        // Allocate output
        let mut output_vec = vec![vec![0.0f32; rows]; batch_size * seq_len];

        // Probe: Debug QK256 dimensions (layer 0 only, once)
        if std::env::var("BITNET_TRACE_RMS").as_deref() == Ok("1")
            && weight_name.contains("layers.0.")
        {
            static DIM_LOGGED: std::sync::Once = std::sync::Once::new();
            DIM_LOGGED.call_once(|| {
                eprintln!(
                    "trace_qk256: weight={} rows={} cols={} row_stride_bytes={} qk256_shape={:?}",
                    weight_name, rows, cols, row_stride_bytes, dims
                );
            });
        }

        // Call QK256 kernel for each input row
        for (i, input_row) in input_vec.iter().enumerate() {
            gemv_qk256(&flat_bytes, input_row, &mut output_vec[i], rows, cols, row_stride_bytes)
                .map_err(|e| {
                    BitNetError::Validation(format!(
                        "QK256 GEMV failed for {} at row {}: {}",
                        weight_name, i, e
                    ))
                })?;
        }

        // Flatten output and reshape
        let output_flat: Vec<f32> = output_vec.into_iter().flatten().collect();
        let output_tensor = if rank == 3 {
            Tensor::from_vec(output_flat, (batch_size, seq_len, rows), input.device())?
        } else {
            Tensor::from_vec(output_flat, (batch_size, rows), input.device())?
        };

        Ok(output_tensor)
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
            layer_idx,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        raw_tensors: &std::collections::HashMap<String, Tensor>,
    ) -> Result<Tensor> {
        let gate = self.apply_linear(x, &self.gate_proj, "gate_proj", raw_tensors)?;

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

        let up = self.apply_linear(x, &self.up_proj, "up_proj", raw_tensors)?;

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

        let output = self.apply_linear(&hidden, &self.down_proj, "down_proj", raw_tensors)?;

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
            tracing::debug!("Using QK256 kernel for {}", qk256_key);
            return Self::forward_qk256(input, qk256_tensor, &qk256_key);
        }

        // Fall back to standard linear
        tracing::trace!(
            "Using standard linear for layers.{}.feed_forward.{}",
            self.layer_idx,
            proj_name
        );
        linear.forward(input).map_err(BitNetError::from)
    }

    /// Forward pass using QK256 kernel (static method - shared with MultiHeadAttention)
    fn forward_qk256(input: &Tensor, qk256_tensor: &Tensor, weight_name: &str) -> Result<Tensor> {
        use crate::quant::i2s_qk256::gemv_qk256;

        // Extract dimensions
        let dims = qk256_tensor.dims();
        if dims.len() != 2 {
            return Err(BitNetError::Validation(format!(
                "QK256 tensor {} has invalid shape: {:?}",
                weight_name, dims
            )));
        }

        let rows = dims[0];
        let row_stride_bytes = dims[1];

        // Calculate cols from row_stride_bytes: each 256-element block uses 64 bytes
        // So: row_stride_bytes / 64 = number of 256-element blocks
        // And: cols = (row_stride_bytes / 64) * 256
        // Safe calculation: (bytes/64)*256 == bytes*4, with overflow check
        debug_assert!(
            row_stride_bytes.is_multiple_of(64),
            "QK256 row_stride_bytes must be multiple of 64"
        );
        let cols = row_stride_bytes
            .checked_mul(4) // (bytes/64)*256 == bytes*4
            .ok_or_else(|| {
                BitNetError::Validation(format!(
                    "QK256: row_stride_bytes overflow computing cols (row_stride={})",
                    row_stride_bytes
                ))
            })?;

        // Extract bytes
        let bytes_2d = qk256_tensor.to_vec2::<u8>().map_err(|e| {
            BitNetError::Validation(format!(
                "Failed to extract QK256 bytes for {}: {}",
                weight_name, e
            ))
        })?;
        let mut flat_bytes = Vec::with_capacity(rows * row_stride_bytes);
        for row in bytes_2d {
            flat_bytes.extend_from_slice(&row);
        }

        // Get input dimensions
        let input_dims = input.dims();
        let rank = input_dims.len();

        // Handle different input shapes: [B, T, H] or [B, H]
        let (batch_size, seq_len, input_cols) = match rank {
            3 => (input_dims[0], input_dims[1], input_dims[2]),
            2 => (input_dims[0], 1, input_dims[1]),
            _ => {
                return Err(BitNetError::Validation(format!(
                    "Unsupported input shape for QK256: {:?}",
                    input_dims
                )));
            }
        };

        // Validate input dimensions match QK256 tensor
        if input_cols != cols {
            return Err(BitNetError::Validation(format!(
                "QK256 dimension mismatch for {}: input has {} cols but QK256 tensor expects {} cols",
                weight_name, input_cols, cols
            )));
        }

        // Flatten to 2D [batch * seq_len, in_features]
        let input_flat = input.reshape(&[batch_size * seq_len, cols])?;
        let input_vec = input_flat.to_vec2::<f32>().map_err(|e| {
            BitNetError::Validation(format!(
                "Failed to convert input to f32 for {}: {}",
                weight_name, e
            ))
        })?;

        // Allocate output
        let mut output_vec = vec![vec![0.0f32; rows]; batch_size * seq_len];

        // Probe: Debug QK256 dimensions (layer 0 only, once)
        if std::env::var("BITNET_TRACE_RMS").as_deref() == Ok("1")
            && weight_name.contains("layers.0.")
        {
            static DIM_LOGGED: std::sync::Once = std::sync::Once::new();
            DIM_LOGGED.call_once(|| {
                eprintln!(
                    "trace_qk256: weight={} rows={} cols={} row_stride_bytes={} qk256_shape={:?}",
                    weight_name, rows, cols, row_stride_bytes, dims
                );
            });
        }

        // Call QK256 kernel for each input row
        for (i, input_row) in input_vec.iter().enumerate() {
            gemv_qk256(&flat_bytes, input_row, &mut output_vec[i], rows, cols, row_stride_bytes)
                .map_err(|e| {
                    BitNetError::Validation(format!(
                        "QK256 GEMV failed for {} at row {}: {}",
                        weight_name, i, e
                    ))
                })?;
        }

        // Flatten output and reshape
        let output_flat: Vec<f32> = output_vec.into_iter().flatten().collect();
        let output_tensor = if rank == 3 {
            Tensor::from_vec(output_flat, (batch_size, seq_len, rows), input.device())?
        } else {
            Tensor::from_vec(output_flat, (batch_size, rows), input.device())?
        };

        Ok(output_tensor)
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
        let eps = config.model.rms_norm_eps.map(|e| e as f64).unwrap_or(1e-5);

        tracing::debug!("TransformerBlock using RMSNorm eps={} (from header)", eps);

        Ok(Self {
            attention: MultiHeadAttention::new(config, vb.pp("attention"), layer_idx)?,
            feed_forward: FeedForward::new(config, vb.pp("feed_forward"), layer_idx)?,
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
            bitnet_trace::dump_trace(&trace_name, &x).map_err(BitNetError::from)?;
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

        let x = self.feed_forward.forward(&x, raw_tensors)?;
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
            // Extract first token's embedding for tracing [B, 1, H]
            let first_token_emb = hidden.narrow(1, 0, 1)?;
            bitnet_trace::dump_trace("t0/embeddings", &first_token_emb)
                .map_err(BitNetError::from)?;
        }

        // Create per-layer KV cache so that rotary/absolute positional
        // encodings use the proper positions during iterative decoding.
        let mut kv_cache = KVCache::new(&self.config, batch_size, &self.device)?;

        // Collect logits for each position.
        let mut logits_steps = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            // Select the current token's embedding and squeeze to [B,H]
            let step_hidden = hidden.narrow(1, t, 1)?.squeeze(1)?;

            // Run through all layers using the incremental path which applies
            // positional encoding per layer and causal masking internally.
            let step_hidden = self.forward(step_hidden, Some(&mut kv_cache))?;

            // Ensure forward preserves expected shape [B, H]
            if step_hidden.dims().len() != 2 {
                return Err(BitNetError::Validation(format!(
                    "forward() should return [B, H] shape, got {:?}",
                    step_hidden.dims()
                )));
            }

            // Project to vocabulary logits for this step.
            let step_logits = self.logits(&step_hidden)?;
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
            bitnet_trace::dump_trace("t0/logits", &first_token_logits)
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
            bitnet_trace::dump_trace("t0/embeddings", &x)
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

                // Tracepoint 5: Logits (incremental path - single token)
                // This captures the final logits for the current token [B, V]
                #[cfg(feature = "trace")]
                {
                    // For incremental path, logits are [B, V] (single token)
                    // Trace directly without narrowing (unlike forward_full which has [B, T, V])
                    bitnet_trace::dump_trace("t0/logits", &logits)
                        .map_err(BitNetError::from)?;
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
    fn test_layer_norm_with_optional_bias() -> candle_core::Result<()> {
        // Test layer_norm_with_optional_bias helper with RMSNorm (no bias)
        let device = Device::Cpu;
        let hidden_size = 128;
        let eps = 1e-5;

        // Create VarBuilder with only weight (no bias)
        use std::collections::HashMap;

        let mut tensors = HashMap::new();
        let weight = Tensor::ones(hidden_size, DType::F32, &device)?;
        tensors.insert("weight".to_string(), weight);

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);

        // Create LayerNorm (should use RMSNorm path due to missing bias)
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

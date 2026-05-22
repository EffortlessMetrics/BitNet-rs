//! Single-responsibility stages for [`MultiHeadAttention::forward`].
//!
//! The public forward path is intentionally a thin pipeline: projection,
//! rotary/cache preparation, grouped-query expansion, score preparation,
//! softmax, and output projection are each isolated below.

#[cfg(feature = "trace")]
use super::BitNetError;
use super::{
    DenseLinearRuntimeHookRegistry, LayerKVCache, MultiHeadAttention, attention_f16_dot_input,
    attention_score_key_input, dbg_finite, dbg_stats, debug_attn_enabled, debug_attn_scale_enabled,
    debug_gqa_enabled, debug_rope_enabled, qwen_trace_event, qwen_trace_events_enabled,
    qwen_trace_layer_enabled, qwen_trace_number, qwen_trace_tensor, trace_rms_enabled,
};
use bitnet_common::Result;
use candle_core::{DType, Module, Tensor};
use std::time::Instant;

struct QkvProjections {
    q: Tensor,
    k: Tensor,
    v: Tensor,
}

struct AttentionHeads {
    q: Tensor,
    k: Tensor,
    v: Tensor,
}

struct ExpandedKv {
    k: Tensor,
    v: Tensor,
}

impl MultiHeadAttention {
    pub fn forward(
        &self,
        x: &Tensor,
        kv_cache: Option<&mut LayerKVCache>,
        raw_tensors: &std::collections::HashMap<String, Tensor>,
        dense_linear_hooks: &DenseLinearRuntimeHookRegistry,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;
        let trace_attention = qwen_trace_events_enabled();
        let attention_start = Instant::now();
        qwen_attention_trace_event(trace_attention, "attention.forward_start", || {
            format!(
                "\"layer\":{},\"batch\":{},\"seq_len\":{},\"n_heads\":{},\"n_kv_heads\":{},\"head_dim\":{}",
                self.layer_idx, batch_size, seq_len, self.n_heads, self.n_kv_heads, self.head_dim
            )
        });

        let projection_start = Instant::now();
        qwen_attention_trace_event(trace_attention, "attention.qkv_projection_start", || {
            format!("\"layer\":{}", self.layer_idx)
        });
        let projections = self.project_qkv(x, raw_tensors, dense_linear_hooks)?;
        qwen_attention_trace_event(trace_attention, "attention.qkv_projection_finish", || {
            format!(
                "\"layer\":{},\"projection_ms\":{}",
                self.layer_idx,
                qwen_attention_elapsed_ms(projection_start)
            )
        });
        self.trace_projection_rms_once(&projections)?;
        self.trace_q_projection(&projections.q)?;

        let reshape_start = Instant::now();
        qwen_attention_trace_event(trace_attention, "attention.reshape_heads_start", || {
            format!("\"layer\":{}", self.layer_idx)
        });
        let heads = self.reshape_qkv_heads(projections, batch_size, seq_len)?;
        qwen_attention_trace_event(trace_attention, "attention.reshape_heads_finish", || {
            format!(
                "\"layer\":{},\"reshape_ms\":{}",
                self.layer_idx,
                qwen_attention_elapsed_ms(reshape_start)
            )
        });
        let qk_norm_start = Instant::now();
        qwen_attention_trace_event(trace_attention, "attention.qk_norm_start", || {
            format!("\"layer\":{}", self.layer_idx)
        });
        let heads = self.apply_qk_norms(heads)?;
        qwen_attention_trace_event(trace_attention, "attention.qk_norm_finish", || {
            format!(
                "\"layer\":{},\"qk_norm_ms\":{}",
                self.layer_idx,
                qwen_attention_elapsed_ms(qk_norm_start)
            )
        });
        self.log_gqa_shapes_once(&heads)?;

        let rope_start = Instant::now();
        qwen_attention_trace_event(trace_attention, "attention.rope_start", || {
            format!(
                "\"layer\":{},\"cache_seq_len\":{}",
                self.layer_idx,
                kv_cache.as_ref().map(|cache| cache.seq_len).unwrap_or(0)
            )
        });
        let heads = self.apply_rotary_embeddings(heads, kv_cache.as_ref().map(|c| c.seq_len))?;
        qwen_attention_trace_event(trace_attention, "attention.rope_finish", || {
            format!(
                "\"layer\":{},\"rope_ms\":{}",
                self.layer_idx,
                qwen_attention_elapsed_ms(rope_start)
            )
        });
        let kv_cache_start = Instant::now();
        qwen_attention_trace_event(trace_attention, "attention.kv_cache_start", || {
            format!("\"layer\":{}", self.layer_idx)
        });
        let (k_ctx, v_ctx) = Self::kv_context(&heads.k, &heads.v, kv_cache)?;
        qwen_attention_trace_event(trace_attention, "attention.kv_cache_finish", || {
            format!(
                "\"layer\":{},\"kv_cache_ms\":{}",
                self.layer_idx,
                qwen_attention_elapsed_ms(kv_cache_start)
            )
        });
        let gqa_start = Instant::now();
        qwen_attention_trace_event(trace_attention, "attention.gqa_expand_start", || {
            format!("\"layer\":{}", self.layer_idx)
        });
        let expanded = self.expand_grouped_query_kv(k_ctx, v_ctx, batch_size)?;
        qwen_attention_trace_event(trace_attention, "attention.gqa_expand_finish", || {
            format!(
                "\"layer\":{},\"gqa_expand_ms\":{}",
                self.layer_idx,
                qwen_attention_elapsed_ms(gqa_start)
            )
        });

        let scores_start = Instant::now();
        qwen_attention_trace_event(trace_attention, "attention.scores_start", || {
            format!("\"layer\":{}", self.layer_idx)
        });
        let scores = self.prepare_attention_scores(&heads.q, &expanded.k, seq_len)?;
        qwen_attention_trace_event(trace_attention, "attention.scores_finish", || {
            format!(
                "\"layer\":{},\"scores_ms\":{}",
                self.layer_idx,
                qwen_attention_elapsed_ms(scores_start)
            )
        });
        let softmax_start = Instant::now();
        qwen_attention_trace_event(trace_attention, "attention.softmax_start", || {
            format!("\"layer\":{}", self.layer_idx)
        });
        let attn_weights = self.softmax_attention_scores(&scores)?;
        qwen_attention_trace_event(trace_attention, "attention.softmax_finish", || {
            format!(
                "\"layer\":{},\"softmax_ms\":{}",
                self.layer_idx,
                qwen_attention_elapsed_ms(softmax_start)
            )
        });
        let value_mix_start = Instant::now();
        qwen_attention_trace_event(trace_attention, "attention.value_mix_start", || {
            format!("\"layer\":{}", self.layer_idx)
        });
        let output_heads = self.apply_attention_weights(&attn_weights, &expanded.v)?;
        qwen_attention_trace_event(trace_attention, "attention.value_mix_finish", || {
            format!(
                "\"layer\":{},\"value_mix_ms\":{}",
                self.layer_idx,
                qwen_attention_elapsed_ms(value_mix_start)
            )
        });

        let output_projection_start = Instant::now();
        qwen_attention_trace_event(trace_attention, "attention.output_projection_start", || {
            format!("\"layer\":{}", self.layer_idx)
        });
        let output = self.project_attention_output(
            output_heads,
            batch_size,
            seq_len,
            raw_tensors,
            dense_linear_hooks,
        )?;
        qwen_attention_trace_event(trace_attention, "attention.output_projection_finish", || {
            format!(
                "\"layer\":{},\"output_projection_ms\":{}",
                self.layer_idx,
                qwen_attention_elapsed_ms(output_projection_start)
            )
        });
        qwen_attention_trace_event(trace_attention, "attention.forward_finish", || {
            format!(
                "\"layer\":{},\"attention_ms\":{}",
                self.layer_idx,
                qwen_attention_elapsed_ms(attention_start)
            )
        });
        Ok(output)
    }

    fn project_qkv(
        &self,
        x: &Tensor,
        raw_tensors: &std::collections::HashMap<String, Tensor>,
        dense_linear_hooks: &DenseLinearRuntimeHookRegistry,
    ) -> Result<QkvProjections> {
        // PATCH 3: Project to Q, K, V separately (NOT fused QKV)
        // This is the correct implementation - separate projections ensure proper shape handling
        // Q: [B, T, hidden] -> [B, T, n_heads * head_dim] -> [B, n_heads, T, head_dim]
        // K: [B, T, hidden] -> [B, T, n_kv_heads * head_dim] -> [B, n_kv_heads, T, head_dim]
        // V: [B, T, hidden] -> [B, T, n_kv_heads * head_dim] -> [B, n_kv_heads, T, head_dim]
        let q = self.apply_linear(x, &self.q_proj, "q_proj", raw_tensors, dense_linear_hooks)?;
        let k = self.apply_linear(x, &self.k_proj, "k_proj", raw_tensors, dense_linear_hooks)?;
        let v = self.apply_linear(x, &self.v_proj, "v_proj", raw_tensors, dense_linear_hooks)?;
        if qwen_trace_layer_enabled(self.layer_idx) {
            qwen_trace_tensor("attention.q_proj", Some(self.layer_idx), &q)?;
            qwen_trace_tensor("attention.k_proj", Some(self.layer_idx), &k)?;
            qwen_trace_tensor("attention.v_proj", Some(self.layer_idx), &v)?;
        }
        Ok(QkvProjections { q, k, v })
    }

    fn trace_projection_rms_once(&self, projections: &QkvProjections) -> Result<()> {
        // Probe A3: Q/K/V projection RMS (layer 0, step 0 only)
        if trace_rms_enabled() && self.layer_idx == 0 {
            static PROJ_LOGGED: std::sync::Once = std::sync::Once::new();
            PROJ_LOGGED.call_once(|| {
                let _ = (|| -> candle_core::Result<()> {
                    let q_vec = projections.q.flatten_all()?.to_vec1::<f32>()?;
                    let q_rms = (q_vec.iter().map(|x| x * x).sum::<f32>()
                        / q_vec.len().max(1) as f32)
                        .sqrt();
                    let k_vec = projections.k.flatten_all()?.to_vec1::<f32>()?;
                    let k_rms = (k_vec.iter().map(|x| x * x).sum::<f32>()
                        / k_vec.len().max(1) as f32)
                        .sqrt();
                    let v_vec = projections.v.flatten_all()?.to_vec1::<f32>()?;
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
        Ok(())
    }

    fn trace_q_projection(&self, _q_proj_out: &Tensor) -> Result<()> {
        // Tracepoint 3: Q projection output (layer-specific)
        #[cfg(feature = "trace")]
        {
            let trace_name = format!("t0/blk{}/q_proj", self.layer_idx);
            bitnet_trace::dump_trace(
                &trace_name,
                _q_proj_out,
                Some(0),
                Some(self.layer_idx as isize),
                Some("q_proj"),
            )
            .map_err(BitNetError::from)?;
        }
        Ok(())
    }

    fn reshape_qkv_heads(
        &self,
        projections: QkvProjections,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<AttentionHeads> {
        let q = projections
            .q
            .reshape(&[batch_size, seq_len, self.n_heads, self.head_dim])?
            .transpose(1, 2)?; // [B, Hq, T, D]

        let k = projections
            .k
            .reshape(&[batch_size, seq_len, self.n_kv_heads, self.head_dim])?
            .transpose(1, 2)?; // [B, HKV, T, D]

        let v = projections
            .v
            .reshape(&[batch_size, seq_len, self.n_kv_heads, self.head_dim])?
            .transpose(1, 2)?; // [B, HKV, T, D]

        // Debug Q, K, V projections
        dbg_stats("Q", &q)?;
        dbg_stats("K", &k)?;
        dbg_stats("V", &v)?;

        Ok(AttentionHeads { q, k, v })
    }

    fn apply_qk_norms(&self, heads: AttentionHeads) -> Result<AttentionHeads> {
        let q = if let Some(norm) = &self.q_norm {
            let normalized = norm.forward(&heads.q)?;
            if qwen_trace_layer_enabled(self.layer_idx) {
                qwen_trace_tensor("attention.q_norm", Some(self.layer_idx), &normalized)?;
            }
            normalized
        } else {
            heads.q
        };

        let k = if let Some(norm) = &self.k_norm {
            let normalized = norm.forward(&heads.k)?;
            if qwen_trace_layer_enabled(self.layer_idx) {
                qwen_trace_tensor("attention.k_norm", Some(self.layer_idx), &normalized)?;
            }
            normalized
        } else {
            heads.k
        };

        Ok(AttentionHeads { q, k, v: heads.v })
    }

    fn log_gqa_shapes_once(&self, heads: &AttentionHeads) -> Result<()> {
        // GQA diagnostic: log Q/K/V dimensions and norms (once per run)
        if debug_gqa_enabled() {
            static GQA_LOGGED: std::sync::Once = std::sync::Once::new();
            GQA_LOGGED.call_once(|| {
                let q_dims = heads.q.dims();
                let k_dims = heads.k.dims();
                let v_dims = heads.v.dims();
                if let (Ok(q_mean), Ok(k_mean), Ok(v_mean)) = (
                    heads.q.mean_all().and_then(|m| m.to_scalar::<f32>()),
                    heads.k.mean_all().and_then(|m| m.to_scalar::<f32>()),
                    heads.v.mean_all().and_then(|m| m.to_scalar::<f32>()),
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
        Ok(())
    }

    fn apply_rotary_embeddings(
        &self,
        heads: AttentionHeads,
        cache_seq_len: Option<usize>,
    ) -> Result<AttentionHeads> {
        // Apply rotary embeddings if available (need to handle different K/V head counts)
        let Some(rope) = &self.rope else {
            return Ok(heads);
        };

        let position = cache_seq_len.unwrap_or(0);

        // Log ROPE application details (once)
        if debug_rope_enabled() {
            static ROPE_LOGGED: std::sync::Once = std::sync::Once::new();
            ROPE_LOGGED.call_once(|| {
                tracing::info!(
                    "ROPE applied: position={}, q_shape={:?}, k_shape={:?}, head_dim={}",
                    position,
                    heads.q.dims(),
                    heads.k.dims(),
                    self.head_dim
                );
            });
        }

        let q = rope.apply(&heads.q, position)?;
        let k = rope.apply(&heads.k, position)?;
        if qwen_trace_layer_enabled(self.layer_idx) {
            qwen_trace_event(
                "attention.rope_metadata",
                &format!(
                    "\"layer\":{},\"position\":{},\"head_dim\":{},\"n_heads\":{},\"n_kv_heads\":{}",
                    self.layer_idx, position, self.head_dim, self.n_heads, self.n_kv_heads
                ),
            );
            qwen_trace_tensor("attention.q_rope", Some(self.layer_idx), &q)?;
            qwen_trace_tensor("attention.k_rope", Some(self.layer_idx), &k)?;
        }

        Ok(AttentionHeads { q, k, v: heads.v })
    }

    fn kv_context<'a>(
        k: &'a Tensor,
        v: &'a Tensor,
        kv_cache: Option<&'a mut LayerKVCache>,
    ) -> Result<(&'a Tensor, &'a Tensor)> {
        // Update KV cache if provided (store HKV heads, not Hq)
        // **Performance note**: Borrow references instead of cloning after append.
        // Candle operations accept both owned and borrowed tensors.
        if let Some(cache) = kv_cache {
            cache.append(k, v)?;
            // Borrow from cache - avoids cloning full KV history
            Ok((&cache.k, &cache.v))
        } else {
            // No cache: use freshly computed K/V from this step
            Ok((k, v))
        }
    }

    fn expand_grouped_query_kv(
        &self,
        k_ctx: &Tensor,
        v_ctx: &Tensor,
        batch_size: usize,
    ) -> Result<ExpandedKv> {
        // GQA core: expand K/V to Hq heads (repeat along head axis)
        // We want K,V of shape [B,Hq,Tk,D]. Repeat every KV head group_size times.
        let t_k = k_ctx.dims()[2];

        // Expand K: [B, HKV, Tk, D] -> [B, Hq, Tk, D]
        let k = k_ctx
            .unsqueeze(2)? // [B, HKV, 1, Tk, D]
            .repeat(&[1, 1, self.group_size, 1, 1])? // [B, HKV, group, Tk, D]
            .reshape(&[batch_size, self.n_heads, t_k, self.head_dim])?; // [B, Hq, Tk, D]

        // Expand V: [B, HKV, Tk, D] -> [B, Hq, Tk, D]
        let v = v_ctx
            .unsqueeze(2)? // [B, HKV, 1, Tk, D]
            .repeat(&[1, 1, self.group_size, 1, 1])? // [B, HKV, group, Tk, D]
            .reshape(&[batch_size, self.n_heads, t_k, self.head_dim])?; // [B, Hq, Tk, D]

        Ok(ExpandedKv { k, v })
    }

    fn prepare_attention_scores(
        &self,
        q: &Tensor,
        k_expanded: &Tensor,
        seq_len: usize,
    ) -> Result<Tensor> {
        // Scaled dot-product attention with explicit fp32 handling
        // For head_dim=128, scale = 1/sqrt(128) ≈ 0.0883883
        let scale_factor = (self.head_dim as f32).sqrt().recip();

        // Log scale computation once
        if debug_attn_scale_enabled() {
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

        let q_for_scores = attention_f16_dot_input(q)?;
        let k_for_scores = attention_score_key_input(k_expanded)?;
        let scores = q_for_scores.matmul(&k_for_scores.transpose(2, 3)?)?;

        // Convert to fp32 for numerically stable computation
        let scores_f32 = scores.to_dtype(DType::F32)?;

        // Scale in fp32
        let scores_f32 = scores_f32.affine(scale_factor as f64, 0.0)?;

        // Debug scores before mask
        dbg_stats("scores pre-mask", &scores_f32)?;
        dbg_finite("scores pre-mask", &scores_f32)?;

        // Apply causal mask so queries cannot attend to future positions.
        // When using a KV cache, k includes past tokens, so the mask must
        // account for the total key length. Single-token decode has no future
        // key positions, so the mask is all zeros and can be skipped.
        let total_len = k_expanded.dims()[2];
        let scores_f32 = if seq_len == 1 {
            scores_f32
        } else {
            // PATCH 5: create_causal_mask now returns [1, 1, Tq, Tk] directly - no need for unsqueeze
            let mask = self.create_causal_mask(seq_len, total_len, scores_f32.device())?;
            scores_f32.broadcast_add(&mask)?
        };
        if qwen_trace_layer_enabled(self.layer_idx) {
            qwen_trace_tensor("attention.scores_post_mask", Some(self.layer_idx), &scores_f32)?;
        }

        // Debug scores after mask and before softmax (critical diagnostics)
        dbg_stats("scores post-mask", &scores_f32)?;
        dbg_finite("scores post-mask", &scores_f32)?;

        // Log scores range after mask for layer 0 (user's diagnostic request)
        if debug_attn_scale_enabled() {
            static LAYER_LOGGED: std::sync::Once = std::sync::Once::new();
            LAYER_LOGGED.call_once(|| {
                if let Ok(flat) = scores_f32.flatten_all()
                    && let Ok(vals) = flat.to_vec1::<f32>()
                    && let (Some(&min_val), Some(&max_val)) = (
                        vals.iter().filter(|v| v.is_finite()).min_by(|a, b| a.total_cmp(b)),
                        vals.iter().filter(|v| v.is_finite()).max_by(|a, b| a.total_cmp(b)),
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

        Ok(scores_f32)
    }

    fn softmax_attention_scores(&self, scores_f32: &Tensor) -> Result<Tensor> {
        // PATCH 4: Softmax path verification
        // Apply max-subtraction for numerical stability before softmax
        // Compute row-wise max and subtract for stability (explicit max-subtraction)
        // VERIFIED: axis=3 is correct for [B, H, Tq, Tk] layout - normalizes across keys (Tk)
        let row_max = scores_f32.max_keepdim(3)?;
        let scores_stabilized = scores_f32.broadcast_sub(&row_max)?;

        // Log that max-subtraction ran (user's diagnostic request)
        if debug_attn_scale_enabled() {
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
        if debug_attn_enabled() {
            let sums = attn_weights.sum(3)?;
            let sums_host: Vec<f32> = sums.flatten_all()?.to_vec1()?;
            let take = sums_host.iter().take(4).cloned().collect::<Vec<_>>();
            eprintln!("[dbg] attn row-sums (first 4): {:?}", take);
        }

        Ok(attn_weights)
    }

    fn apply_attention_weights(
        &self,
        attn_weights: &Tensor,
        v_expanded: &Tensor,
    ) -> Result<Tensor> {
        let v_for_value_mix = attention_f16_dot_input(v_expanded)?;
        let attn_output = attn_weights.matmul(&v_for_value_mix)?;
        if qwen_trace_layer_enabled(self.layer_idx) {
            qwen_trace_tensor("attention.output_heads", Some(self.layer_idx), &attn_output)?;
        }
        Ok(attn_output)
    }

    fn project_attention_output(
        &self,
        attn_output: Tensor,
        batch_size: usize,
        seq_len: usize,
        raw_tensors: &std::collections::HashMap<String, Tensor>,
        dense_linear_hooks: &DenseLinearRuntimeHookRegistry,
    ) -> Result<Tensor> {
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

        let projected = self.apply_linear(
            &attn_output,
            &self.o_proj,
            "o_proj",
            raw_tensors,
            dense_linear_hooks,
        )?;
        if qwen_trace_layer_enabled(self.layer_idx) {
            qwen_trace_tensor("attention.o_proj", Some(self.layer_idx), &projected)?;
        }
        Ok(projected)
    }
}

fn qwen_attention_trace_event(enabled: bool, stage: &str, fields_json: impl FnOnce() -> String) {
    if enabled {
        qwen_trace_event(stage, &fields_json());
    }
}

fn qwen_attention_elapsed_ms(start: Instant) -> String {
    qwen_trace_number(start.elapsed().as_secs_f64() * 1000.0)
}

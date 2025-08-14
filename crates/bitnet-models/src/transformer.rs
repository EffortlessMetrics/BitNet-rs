use bitnet_common::{BitNetConfig, BitNetError, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{LayerNorm, Linear, VarBuilder};

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
    head_dim: usize,
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

        let q_proj = candle_nn::linear(hidden_size, hidden_size, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear(hidden_size, hidden_size, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(hidden_size, hidden_size, vb.pp("v_proj"))?;
        let o_proj = candle_nn::linear(hidden_size, hidden_size, vb.pp("o_proj"))?;

        let rope = RotaryEmbedding::new(
            head_dim,
            config.model.max_position_embeddings,
            config.model.rope_theta,
            vb.device(),
        )
        .ok();

        Ok(Self {
            n_heads,
            head_dim,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rope,
        })
    }

    pub fn forward(&self, x: &Tensor, kv_cache: Option<&mut LayerKVCache>) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;

        // Project to Q, K, V
        let q = self
            .q_proj
            .forward(x)?
            .reshape(&[batch_size, seq_len, self.n_heads, self.head_dim])?
            .transpose(1, 2)?; // [B, H, T, D]

        let k = self
            .k_proj
            .forward(x)?
            .reshape(&[batch_size, seq_len, self.n_heads, self.head_dim])?
            .transpose(1, 2)?;

        let v = self
            .v_proj
            .forward(x)?
            .reshape(&[batch_size, seq_len, self.n_heads, self.head_dim])?
            .transpose(1, 2)?;

        // Apply rotary embeddings if available
        let (q, k) = if let Some(rope) = &self.rope {
            let position = kv_cache.as_ref().map(|c| c.seq_len).unwrap_or(0);
            let q_rot = rope.apply(&q, position)?;
            let k_rot = rope.apply(&k, position)?;
            (q_rot, k_rot)
        } else {
            (q, k)
        };

        // Update KV cache if provided
        let (k, v) = if let Some(cache) = kv_cache {
            cache.append(&k, &v)?;
            (cache.k.clone(), cache.v.clone())
        } else {
            (k, v)
        };

        // Scaled dot-product attention
        let scale = (self.head_dim as f32).sqrt();
        let scores = q.matmul(&k.transpose(2, 3)?)?;
        let scores = scores.affine((1.0 / scale) as f64, 0.0)?;

        // Apply causal mask
        let mask = self.create_causal_mask(seq_len, scores.device())?
            .unsqueeze(0)?  // Add batch dim
            .unsqueeze(0)?; // Add heads dim
        let scores = scores.broadcast_add(&mask)?;

        let attn_weights = candle_nn::ops::softmax(&scores, 3)?;
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape and project output
        let attn_output = attn_output.transpose(1, 2)?.reshape(&[
            batch_size,
            seq_len,
            self.n_heads * self.head_dim,
        ])?;

        Ok(self.o_proj.forward(&attn_output)?)
    }

    fn create_causal_mask(&self, seq_len: usize, device: &Device) -> Result<Tensor> {
        // Create a simple causal mask
        let mut mask_vec = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                mask_vec[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
        Ok(Tensor::from_vec(mask_vec, &[seq_len, seq_len], device)?)
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
            gate_proj: candle_nn::linear(hidden_size, intermediate_size, vb.pp("gate_proj"))?,
            up_proj: candle_nn::linear(hidden_size, intermediate_size, vb.pp("up_proj"))?,
            down_proj: candle_nn::linear(intermediate_size, hidden_size, vb.pp("down_proj"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let gate = candle_nn::ops::silu(&gate)?;
        let up = self.up_proj.forward(x)?;
        let hidden = gate.mul(&up)?;
        Ok(self.down_proj.forward(&hidden)?)
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
        let eps = 1e-5;

        Ok(Self {
            attention: MultiHeadAttention::new(config, vb.pp("attention"))?,
            feed_forward: FeedForward::new(config, vb.pp("feed_forward"))?,
            attention_norm: candle_nn::layer_norm(hidden_size, eps, vb.pp("attention_norm"))?,
            ffn_norm: candle_nn::layer_norm(hidden_size, eps, vb.pp("ffn_norm"))?,
        })
    }

    pub fn forward(&self, x: &Tensor, kv_cache: Option<&mut LayerKVCache>) -> Result<Tensor> {
        // Pre-norm attention
        let residual = x;
        let x = self.attention_norm.forward(x)?;
        let x = self.attention.forward(&x, kv_cache)?;
        let x = (x + residual)?;

        // Pre-norm FFN
        let residual = &x;
        let x = self.ffn_norm.forward(&x)?;
        let x = self.feed_forward.forward(&x)?;
        let x = (x + residual)?;

        Ok(x)
    }
}

/// KV Cache for a single layer
pub struct LayerKVCache {
    pub k: Tensor,
    pub v: Tensor,
    pub seq_len: usize,
    pub max_seq_len: usize,
}

impl LayerKVCache {
    pub fn new(
        batch_size: usize,
        n_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        let k = Tensor::zeros(
            &[batch_size, n_heads, max_seq_len, head_dim],
            DType::F32,
            device,
        )?;
        let v = Tensor::zeros(
            &[batch_size, n_heads, max_seq_len, head_dim],
            DType::F32,
            device,
        )?;

        Ok(Self {
            k,
            v,
            seq_len: 0,
            max_seq_len,
        })
    }

    pub fn append(&mut self, k_new: &Tensor, v_new: &Tensor) -> Result<()> {
        // Expect shapes: k: [B,H,T_new,Hd], v: [B,H,T_new,Hd]
        let new_seq_len = k_new.dims()[2];

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
        let head_dim = config.model.hidden_size / n_heads;
        let max_seq_len = config.model.max_position_embeddings;

        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(LayerKVCache::new(
                batch_size,
                n_heads,
                max_seq_len,
                head_dim,
                device,
            )?);
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
    pub layers: Vec<TransformerBlock>,
    pub norm: LayerNorm,
    pub lm_head: Linear,
    device: Device,
}

impl TransformerModel {
    pub fn new(config: BitNetConfig, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let vocab_size = config.model.vocab_size;
        let hidden_size = config.model.hidden_size;
        let n_layers = config.model.num_layers;

        let embed_tokens = candle_nn::embedding(vocab_size, hidden_size, vb.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            layers.push(TransformerBlock::new(
                &config,
                vb.pp(&format!("layers.{}", i)),
            )?);
        }

        let norm = candle_nn::layer_norm(hidden_size, 1e-5, vb.pp("norm"))?;
        let lm_head = candle_nn::linear(hidden_size, vocab_size, vb.pp("lm_head"))?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
            lm_head,
            device,
        })
    }

    pub fn embed(&self, tokens: &[u32]) -> Result<Tensor> {
        let token_ids = Tensor::from_vec(tokens.to_vec(), &[1, tokens.len()], &self.device)?;
        Ok(self.embed_tokens.forward(&token_ids)?)
    }

    pub fn forward(&self, hidden: &Tensor, mut kv_cache: Option<&mut KVCache>) -> Result<Tensor> {
        let mut x = hidden.clone();

        for (i, layer) in self.layers.iter().enumerate() {
            let layer_cache = kv_cache.as_mut().and_then(|c| c.layer_mut(i));
            x = layer.forward(&x, layer_cache)?;
        }

        Ok(self.norm.forward(&x)?)
    }

    pub fn logits(&self, hidden: &Tensor) -> Result<Tensor> {
        Ok(self.lm_head.forward(hidden)?)
    }
}

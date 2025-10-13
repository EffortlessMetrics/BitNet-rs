# Multi-Head Attention Specification

**Component**: Quantized multi-head attention with KV-cache and rotary embeddings
**Location**: `bitnet-inference/src/attention/multi_head_attention.rs`
**Dependencies**: bitnet-quantization, bitnet-kernels, quantized linear layers

## Overview

Multi-head attention forms the core of transformer computation, enabling the model to attend to different parts of the input sequence simultaneously. This specification defines a production-ready quantized multi-head attention implementation that integrates with BitNet.rs quantization infrastructure, supports efficient KV-cache for autoregressive generation, and includes rotary positional embeddings for improved sequence modeling.

## Architecture Design

### Core Components

```rust
/// Quantized Multi-Head Attention with Grouped Query Attention support
pub struct BitNetMultiHeadAttention {
    // Attention configuration
    n_heads: usize,                    // Number of query heads
    n_kv_heads: usize,                // Number of key/value heads (GQA)
    head_dim: usize,                  // Dimension per attention head
    hidden_size: usize,               // Total hidden dimension
    scale: f32,                       // 1.0 / sqrt(head_dim) for scaling

    // Quantized projection layers
    q_proj: QuantizedLinear,          // Query projection [H, H]
    k_proj: QuantizedLinear,          // Key projection [H, KV_H * head_dim]
    v_proj: QuantizedLinear,          // Value projection [H, KV_H * head_dim]
    o_proj: QuantizedLinear,          // Output projection [H, H]

    // Positional encodings
    rope: Option<RotaryEmbedding>,    // Rotary positional embeddings

    // Performance optimizations
    device: Device,                   // CPU/GPU device context
    fused_qkv: bool,                  // Whether to use fused QKV projection
    flash_attention: bool,            // Whether to use flash attention (GPU)

    // Memory optimization
    attention_workspace: Option<Tensor>, // Pre-allocated workspace
    kv_cache_format: KVCacheFormat,     // Cache tensor layout optimization
}

impl BitNetMultiHeadAttention {
    /// Create multi-head attention from configuration
    pub fn new(config: &AttentionConfig, device: &Device) -> Result<Self>;

    /// Load from GGUF tensors with quantized weights
    pub fn from_gguf(
        q_weight: &GgufTensor,
        k_weight: &GgufTensor,
        v_weight: &GgufTensor,
        o_weight: &GgufTensor,
        config: &AttentionConfig,
        device: &Device
    ) -> Result<Self>;

    /// Forward pass with optional KV cache
    /// Input: [batch_size, seq_len, hidden_size]
    /// Output: [batch_size, seq_len, hidden_size]
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        kv_cache: Option<&mut LayerKVCache>
    ) -> Result<AttentionOutput>;

    /// Streaming forward pass for autoregressive generation (single token)
    /// Input: [batch_size, 1, hidden_size]
    /// Output: [batch_size, 1, hidden_size]
    pub fn forward_streaming(
        &self,
        hidden_states: &Tensor,
        kv_cache: &mut LayerKVCache,
        position: usize
    ) -> Result<Tensor>;
}
```

### Attention Configuration

```rust
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    pub hidden_size: usize,           // Model hidden dimension
    pub num_attention_heads: usize,   // Number of query heads
    pub num_key_value_heads: usize,   // Number of KV heads (GQA)
    pub max_position_embeddings: usize, // Maximum sequence length
    pub rope_theta: f32,              // RoPE theta parameter
    pub attention_dropout: f32,       // Attention dropout (training only)
    pub scale_attn_weights: bool,     // Whether to scale attention weights
    pub quantization_type: QuantizationType, // I2S, TL1, or TL2
}

impl AttentionConfig {
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(AttentionError::InvalidConfig {
                reason: format!("hidden_size {} must be divisible by num_attention_heads {}",
                               self.hidden_size, self.num_attention_heads)
            });
        }

        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err(AttentionError::InvalidConfig {
                reason: format!("num_attention_heads {} must be divisible by num_key_value_heads {}",
                               self.num_attention_heads, self.num_key_value_heads)
            });
        }

        Ok(())
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn group_size(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}
```

## Core Attention Implementation

### Forward Pass with Quantized Projections

```rust
impl BitNetMultiHeadAttention {
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        kv_cache: Option<&mut LayerKVCache>
    ) -> Result<AttentionOutput> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;

        // Step 1: Quantized Q, K, V projections
        let (query_states, key_states, value_states) = if self.fused_qkv {
            self.forward_fused_qkv(hidden_states)?
        } else {
            self.forward_separate_qkv(hidden_states)?
        };

        // Step 2: Reshape for multi-head attention
        // Q: [B, T, H] -> [B, T, n_heads, head_dim] -> [B, n_heads, T, head_dim]
        let query_states = query_states
            .reshape(&[batch_size, seq_len, self.n_heads, self.head_dim])?
            .transpose(1, 2)?;

        // K, V: [B, T, KV_H * head_dim] -> [B, T, n_kv_heads, head_dim] -> [B, n_kv_heads, T, head_dim]
        let key_states = key_states
            .reshape(&[batch_size, seq_len, self.n_kv_heads, self.head_dim])?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape(&[batch_size, seq_len, self.n_kv_heads, self.head_dim])?
            .transpose(1, 2)?;

        // Step 3: Apply rotary positional embeddings
        let (query_states, key_states) = if let Some(ref rope) = self.rope {
            let position_offset = kv_cache.as_ref().map(|c| c.seq_len).unwrap_or(0);
            let query_rot = rope.apply_rotary_embedding(&query_states, position_offset)?;
            let key_rot = rope.apply_rotary_embedding(&key_states, position_offset)?;
            (query_rot, key_rot)
        } else {
            (query_states, key_states)
        };

        // Step 4: Update KV cache for autoregressive generation
        let (key_states, value_states, total_seq_len) = if let Some(cache) = kv_cache {
            cache.append(&key_states, &value_states)?;
            let total_len = cache.seq_len;
            (cache.k.clone(), cache.v.clone(), total_len)
        } else {
            (key_states, value_states, seq_len)
        };

        // Step 5: Grouped Query Attention - expand K/V to match Q heads
        let key_states = self.expand_kv_for_gqa(&key_states)?;    // [B, n_heads, T_total, head_dim]
        let value_states = self.expand_kv_for_gqa(&value_states)?; // [B, n_heads, T_total, head_dim]

        // Step 6: Compute attention (device-aware implementation)
        let attention_output = if self.flash_attention && matches!(self.device, Device::Cuda(_)) {
            self.flash_attention_forward(&query_states, &key_states, &value_states, attention_mask)?
        } else {
            self.standard_attention_forward(&query_states, &key_states, &value_states, attention_mask, total_seq_len)?
        };

        // Step 7: Reshape and apply output projection
        let attention_output = attention_output
            .transpose(1, 2)?  // [B, T, n_heads, head_dim]
            .reshape(&[batch_size, seq_len, self.hidden_size])?;

        let output = self.o_proj.forward(&attention_output)?;

        Ok(AttentionOutput {
            hidden_states: output,
            attention_weights: None, // Optional for analysis
        })
    }

    /// Separate Q, K, V projections using quantized linear layers
    fn forward_separate_qkv(&self, hidden_states: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let query_states = self.q_proj.forward(hidden_states)?;
        let key_states = self.k_proj.forward(hidden_states)?;
        let value_states = self.v_proj.forward(hidden_states)?;

        Ok((query_states, key_states, value_states))
    }

    /// Fused QKV projection for better performance (when available)
    fn forward_fused_qkv(&self, hidden_states: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        // This would be implemented if the model has fused QKV weights
        // For now, delegate to separate projections
        self.forward_separate_qkv(hidden_states)
    }
}
```

### Grouped Query Attention (GQA) Support

```rust
impl BitNetMultiHeadAttention {
    /// Expand K/V heads to match Q heads for Grouped Query Attention
    fn expand_kv_for_gqa(&self, kv_tensor: &Tensor) -> Result<Tensor> {
        if self.n_heads == self.n_kv_heads {
            // Standard multi-head attention - no expansion needed
            return Ok(kv_tensor.clone());
        }

        let (batch_size, n_kv_heads, seq_len, head_dim) = kv_tensor.dims4()?;
        let group_size = self.n_heads / self.n_kv_heads;

        // Expand each KV head to group_size query heads
        // [B, KV_H, T, D] -> [B, KV_H, group_size, T, D] -> [B, Q_H, T, D]
        let expanded = kv_tensor
            .unsqueeze(2)?  // [B, KV_H, 1, T, D]
            .repeat(&[1, 1, group_size, 1, 1])?  // [B, KV_H, group_size, T, D]
            .reshape(&[batch_size, self.n_heads, seq_len, head_dim])?; // [B, Q_H, T, D]

        Ok(expanded)
    }

    /// Validate GQA configuration
    fn validate_gqa_config(&self) -> Result<()> {
        if self.n_heads % self.n_kv_heads != 0 {
            return Err(AttentionError::InvalidConfig {
                reason: format!(
                    "Number of query heads ({}) must be divisible by KV heads ({})",
                    self.n_heads, self.n_kv_heads
                ),
            });
        }

        let group_size = self.n_heads / self.n_kv_heads;
        if group_size > 8 {
            log::warn!(
                "Large GQA group size ({}): may impact attention quality",
                group_size
            );
        }

        Ok(())
    }
}
```

### Standard Attention Computation

```rust
impl BitNetMultiHeadAttention {
    /// Standard scaled dot-product attention
    fn standard_attention_forward(
        &self,
        query_states: &Tensor,   // [B, H, T_q, D]
        key_states: &Tensor,     // [B, H, T_k, D]
        value_states: &Tensor,   // [B, H, T_k, D]
        attention_mask: Option<&Tensor>,
        key_seq_len: usize,
    ) -> Result<Tensor> {
        // Compute attention scores: Q @ K^T
        let attention_scores = query_states.matmul(&key_states.transpose(-2, -1)?)?;

        // Scale by 1/sqrt(head_dim)
        let scaled_scores = attention_scores.affine(self.scale as f64, 0.0)?;

        // Apply causal mask for autoregressive attention
        let masked_scores = if let Some(mask) = attention_mask {
            scaled_scores.broadcast_add(mask)?
        } else {
            // Create causal mask: queries can only attend to previous positions
            let causal_mask = self.create_causal_mask(
                query_states.dims()[2],  // query sequence length
                key_seq_len,            // key sequence length (includes cached tokens)
                &scaled_scores.device()
            )?;
            scaled_scores.broadcast_add(&causal_mask)?
        };

        // Apply softmax to get attention weights
        let attention_weights = masked_scores.softmax(-1)?;

        // Apply attention dropout (only during training)
        let attention_weights = if self.training && self.attention_dropout > 0.0 {
            self.apply_attention_dropout(&attention_weights)?
        } else {
            attention_weights
        };

        // Compute attention output: weights @ V
        let attention_output = attention_weights.matmul(value_states)?;

        Ok(attention_output)
    }

    /// Create causal attention mask for autoregressive generation
    fn create_causal_mask(&self, query_len: usize, key_len: usize, device: &Device) -> Result<Tensor> {
        // Past tokens are stored in KV cache, so key_len >= query_len
        let past_len = key_len.saturating_sub(query_len);

        let mut mask_values = vec![0.0f32; query_len * key_len];

        for i in 0..query_len {
            let current_pos = past_len + i;
            // Mask future positions (set to -inf)
            for j in (current_pos + 1)..key_len {
                mask_values[i * key_len + j] = f32::NEG_INFINITY;
            }
        }

        let mask = Tensor::from_vec(mask_values, &[query_len, key_len], device)?;

        // Add dimensions for broadcasting: [1, 1, query_len, key_len]
        Ok(mask.unsqueeze(0)?.unsqueeze(0)?)
    }

    /// Apply attention dropout (training only)
    fn apply_attention_dropout(&self, attention_weights: &Tensor) -> Result<Tensor> {
        // Dropout implementation would be here for training
        // For inference, this is a no-op
        Ok(attention_weights.clone())
    }
}
```

### Flash Attention for GPU Acceleration

```rust
#[cfg(feature = "gpu")]
impl BitNetMultiHeadAttention {
    /// Flash Attention implementation for memory-efficient GPU computation
    fn flash_attention_forward(
        &self,
        query_states: &Tensor,
        key_states: &Tensor,
        value_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Check if flash attention is available
        let cuda_kernel = self.get_cuda_kernel()?;

        if !cuda_kernel.supports_flash_attention() {
            log::warn!("Flash attention not available, falling back to standard attention");
            return self.standard_attention_forward(
                query_states, key_states, value_states, attention_mask, key_states.dims()[2]
            );
        }

        // Use pre-allocated workspace to avoid memory fragmentation
        let workspace = self.attention_workspace.as_ref()
            .ok_or(AttentionError::MemoryError {
                reason: "Flash attention workspace not allocated".to_string(),
            })?;

        // Call CUDA flash attention kernel
        let attention_output = cuda_kernel.flash_attention(
            query_states,
            key_states,
            value_states,
            attention_mask,
            workspace,
            self.scale,
            true, // is_causal
        )?;

        Ok(attention_output)
    }

    /// Get CUDA kernel for attention operations
    fn get_cuda_kernel(&self) -> Result<&dyn CudaAttentionKernel> {
        // Implementation would get the appropriate CUDA kernel
        todo!("CUDA kernel integration")
    }

    /// Allocate workspace for flash attention
    fn allocate_flash_attention_workspace(&mut self, max_batch_size: usize, max_seq_len: usize) -> Result<()> {
        let workspace_size = self.calculate_flash_attention_workspace_size(max_batch_size, max_seq_len);

        self.attention_workspace = Some(Tensor::zeros(
            &[workspace_size],
            DType::F32,
            &self.device,
        )?);

        log::debug!(
            "Allocated flash attention workspace: {} MB",
            workspace_size * 4 / 1024 / 1024
        );

        Ok(())
    }

    fn calculate_flash_attention_workspace_size(&self, max_batch_size: usize, max_seq_len: usize) -> usize {
        // Flash attention needs temporary storage for block-wise computation
        let block_size = 128; // Typical flash attention block size
        let blocks_per_seq = (max_seq_len + block_size - 1) / block_size;

        // Workspace needs: attention scores + intermediate results
        max_batch_size * self.n_heads * blocks_per_seq * block_size * block_size
    }
}
```

## Rotary Positional Embeddings (RoPE)

### RoPE Implementation

```rust
/// Rotary Positional Embedding for improved sequence modeling
pub struct RotaryEmbedding {
    sin_cached: Tensor,               // Precomputed sin values
    cos_cached: Tensor,               // Precomputed cos values
    dim: usize,                       // Embedding dimension
    max_seq_len: usize,              // Maximum supported sequence length
    base: f32,                       // RoPE base (theta)
    device: Device,                  // Device for tensor operations
}

impl RotaryEmbedding {
    pub fn new(dim: usize, max_seq_len: usize, base: f32, device: &Device) -> Result<Self> {
        // Precompute sin and cos values for all positions and dimensions
        let inv_freq = Self::compute_inv_freq(dim, base);
        let (sin_cached, cos_cached) = Self::precompute_sin_cos(&inv_freq, max_seq_len, device)?;

        Ok(Self {
            sin_cached,
            cos_cached,
            dim,
            max_seq_len,
            base,
            device: device.clone(),
        })
    }

    /// Compute inverse frequencies for RoPE
    fn compute_inv_freq(dim: usize, base: f32) -> Vec<f32> {
        (0..dim)
            .step_by(2)
            .map(|i| 1.0 / base.powf(i as f32 / dim as f32))
            .collect()
    }

    /// Precompute sin and cos values for all positions
    fn precompute_sin_cos(
        inv_freq: &[f32],
        max_seq_len: usize,
        device: &Device
    ) -> Result<(Tensor, Tensor)> {
        let positions: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
        let mut sin_values = Vec::with_capacity(max_seq_len * inv_freq.len());
        let mut cos_values = Vec::with_capacity(max_seq_len * inv_freq.len());

        for pos in &positions {
            for &freq in inv_freq {
                let angle = pos * freq;
                sin_values.push(angle.sin());
                cos_values.push(angle.cos());
            }
        }

        let sin_tensor = Tensor::from_vec(
            sin_values,
            &[max_seq_len, inv_freq.len()],
            device
        )?;
        let cos_tensor = Tensor::from_vec(
            cos_values,
            &[max_seq_len, inv_freq.len()],
            device
        )?;

        Ok((sin_tensor, cos_tensor))
    }

    /// Apply rotary embedding to query or key tensor
    /// Input: [batch_size, num_heads, seq_len, head_dim]
    /// Output: [batch_size, num_heads, seq_len, head_dim]
    pub fn apply_rotary_embedding(&self, tensor: &Tensor, position_offset: usize) -> Result<Tensor> {
        let (batch_size, num_heads, seq_len, head_dim) = tensor.dims4()?;

        if head_dim != self.dim {
            return Err(AttentionError::InvalidShape {
                expected: vec![batch_size, num_heads, seq_len, self.dim],
                actual: tensor.dims().to_vec(),
            });
        }

        if position_offset + seq_len > self.max_seq_len {
            return Err(AttentionError::SequenceTooLong {
                requested: position_offset + seq_len,
                max_supported: self.max_seq_len,
            });
        }

        // Split tensor into real and imaginary parts for rotation
        let half_dim = head_dim / 2;
        let x_real = tensor.narrow(-1, 0, half_dim)?;                    // [..., :half_dim]
        let x_imag = tensor.narrow(-1, half_dim, half_dim)?;             // [..., half_dim:]

        // Get sin/cos values for current positions
        let positions = (position_offset..position_offset + seq_len).collect::<Vec<_>>();
        let sin_vals = self.sin_cached.index_select(&Tensor::new(&positions, &self.device)?, 0)?;
        let cos_vals = self.cos_cached.index_select(&Tensor::new(&positions, &self.device)?, 0)?;

        // Broadcast sin/cos to match tensor dimensions
        let sin_vals = sin_vals
            .unsqueeze(0)?                                               // [1, seq_len, half_dim]
            .unsqueeze(0)?                                               // [1, 1, seq_len, half_dim]
            .broadcast_as(&[batch_size, num_heads, seq_len, half_dim])?;

        let cos_vals = cos_vals
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as(&[batch_size, num_heads, seq_len, half_dim])?;

        // Apply rotation: [cos * real - sin * imag, sin * real + cos * imag]
        let rotated_real = (x_real.mul(&cos_vals)? - x_imag.mul(&sin_vals)?)?;
        let rotated_imag = (x_real.mul(&sin_vals)? + x_imag.mul(&cos_vals)?)?;

        // Concatenate rotated parts
        let rotated = Tensor::cat(&[rotated_real, rotated_imag], -1)?;

        Ok(rotated)
    }

    /// Update cached sin/cos values for longer sequences
    pub fn extend_cache(&mut self, new_max_seq_len: usize) -> Result<()> {
        if new_max_seq_len <= self.max_seq_len {
            return Ok(());
        }

        log::info!("Extending RoPE cache from {} to {} positions", self.max_seq_len, new_max_seq_len);

        let inv_freq = Self::compute_inv_freq(self.dim, self.base);
        let (sin_cached, cos_cached) = Self::precompute_sin_cos(&inv_freq, new_max_seq_len, &self.device)?;

        self.sin_cached = sin_cached;
        self.cos_cached = cos_cached;
        self.max_seq_len = new_max_seq_len;

        Ok(())
    }
}
```

## KV-Cache Optimization

### Layer-Level KV Cache

```rust
/// KV cache for a single attention layer
pub struct LayerKVCache {
    // Cache tensors
    k: Tensor,                        // Key cache [B, KV_H, max_seq_len, head_dim]
    v: Tensor,                        // Value cache [B, KV_H, max_seq_len, head_dim]

    // Cache metadata
    seq_len: usize,                   // Current sequence length
    max_seq_len: usize,              // Maximum supported length
    batch_size: usize,               // Batch size
    n_kv_heads: usize,               // Number of KV heads
    head_dim: usize,                 // Dimension per head

    // Memory optimization
    format: KVCacheFormat,           // Memory layout optimization
    device: Device,                  // Device for cache tensors
}

impl LayerKVCache {
    pub fn new(
        batch_size: usize,
        n_kv_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
        device: &Device,
        format: KVCacheFormat,
    ) -> Result<Self> {
        let cache_shape = match format {
            KVCacheFormat::Standard => [batch_size, n_kv_heads, max_seq_len, head_dim],
            KVCacheFormat::Transposed => [batch_size, max_seq_len, n_kv_heads, head_dim],
        };

        let k = Tensor::zeros(&cache_shape, DType::F32, device)?;
        let v = Tensor::zeros(&cache_shape, DType::F32, device)?;

        Ok(Self {
            k,
            v,
            seq_len: 0,
            max_seq_len,
            batch_size,
            n_kv_heads,
            head_dim,
            format,
            device: device.clone(),
        })
    }

    /// Append new key/value tensors to cache
    pub fn append(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<()> {
        let new_seq_len = match self.format {
            KVCacheFormat::Standard => new_k.dims()[2],   // [B, H, T, D]
            KVCacheFormat::Transposed => new_k.dims()[1], // [B, T, H, D]
        };

        // Validate cache capacity
        if self.seq_len + new_seq_len > self.max_seq_len {
            return Err(AttentionError::CacheOverflow {
                current_len: self.seq_len,
                new_len: new_seq_len,
                max_len: self.max_seq_len,
            });
        }

        // Validate tensor shapes
        self.validate_append_shapes(new_k, new_v, new_seq_len)?;

        // Append to cache based on format
        match self.format {
            KVCacheFormat::Standard => self.append_standard_format(new_k, new_v, new_seq_len)?,
            KVCacheFormat::Transposed => self.append_transposed_format(new_k, new_v, new_seq_len)?,
        }

        self.seq_len += new_seq_len;
        Ok(())
    }

    /// Append with standard [B, H, T, D] format
    fn append_standard_format(&mut self, new_k: &Tensor, new_v: &Tensor, new_seq_len: usize) -> Result<()> {
        if self.seq_len == 0 {
            // First append - copy directly
            self.k = new_k.clone();
            self.v = new_v.clone();
        } else {
            // Concatenate along sequence dimension (dim=2)
            self.k = Tensor::cat(&[&self.k, new_k], 2)?;
            self.v = Tensor::cat(&[&self.v, new_v], 2)?;
        }
        Ok(())
    }

    /// Append with transposed [B, T, H, D] format (may be more cache-friendly)
    fn append_transposed_format(&mut self, new_k: &Tensor, new_v: &Tensor, new_seq_len: usize) -> Result<()> {
        // Convert from standard to transposed format if needed
        let new_k_transposed = if new_k.dims().len() == 4 && new_k.dims()[1] == self.n_kv_heads {
            new_k.transpose(1, 2)? // [B, H, T, D] -> [B, T, H, D]
        } else {
            new_k.clone()
        };

        let new_v_transposed = if new_v.dims().len() == 4 && new_v.dims()[1] == self.n_kv_heads {
            new_v.transpose(1, 2)? // [B, H, T, D] -> [B, T, H, D]
        } else {
            new_v.clone()
        };

        if self.seq_len == 0 {
            self.k = new_k_transposed;
            self.v = new_v_transposed;
        } else {
            // Concatenate along sequence dimension (dim=1 in transposed format)
            self.k = Tensor::cat(&[&self.k, &new_k_transposed], 1)?;
            self.v = Tensor::cat(&[&self.v, &new_v_transposed], 1)?;
        }
        Ok(())
    }

    /// Validate shapes before appending
    fn validate_append_shapes(&self, new_k: &Tensor, new_v: &Tensor, new_seq_len: usize) -> Result<()> {
        let k_dims = new_k.dims();
        let v_dims = new_v.dims();

        // Check basic shape compatibility
        if k_dims.len() != 4 || v_dims.len() != 4 {
            return Err(AttentionError::InvalidShape {
                expected: vec![self.batch_size, self.n_kv_heads, new_seq_len, self.head_dim],
                actual: k_dims.to_vec(),
            });
        }

        // Check dimensions match expectations
        let expected_k_shape = match self.format {
            KVCacheFormat::Standard => [self.batch_size, self.n_kv_heads, new_seq_len, self.head_dim],
            KVCacheFormat::Transposed => [self.batch_size, new_seq_len, self.n_kv_heads, self.head_dim],
        };

        if k_dims != expected_k_shape || v_dims != expected_k_shape {
            return Err(AttentionError::InvalidShape {
                expected: expected_k_shape.to_vec(),
                actual: k_dims.to_vec(),
            });
        }

        Ok(())
    }

    /// Clear cache for new sequence
    pub fn clear(&mut self) {
        self.seq_len = 0;
        // Note: We don't zero the tensors, just reset the length
        // This avoids expensive memory operations
    }

    /// Get current memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let elements_per_tensor = self.batch_size * self.n_kv_heads * self.max_seq_len * self.head_dim;
        let bytes_per_element = match self.k.dtype() {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::BF16 => 2,
            _ => 4, // Default
        };

        2 * elements_per_tensor * bytes_per_element // K + V tensors
    }

    /// Optimize cache layout for better memory access patterns
    pub fn optimize_layout(&mut self) -> Result<()> {
        // This could implement cache-aware optimizations like:
        // - Reordering for better cache locality
        // - Using packed formats for smaller memory footprint
        // - GPU-specific optimizations

        if matches!(self.device, Device::Cuda(_)) {
            self.optimize_gpu_layout()?;
        }

        Ok(())
    }

    #[cfg(feature = "gpu")]
    fn optimize_gpu_layout(&mut self) -> Result<()> {
        // GPU-specific optimizations:
        // - Ensure coalesced memory access
        // - Use texture memory if beneficial
        // - Consider using half precision for larger caches
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum KVCacheFormat {
    Standard,    // [B, H, T, D] - standard attention format
    Transposed,  // [B, T, H, D] - may be more cache-friendly
}
```

## Error Handling and Validation

### Comprehensive Error Types

```rust
#[derive(Debug, thiserror::Error)]
pub enum AttentionError {
    #[error("Invalid attention configuration: {reason}")]
    InvalidConfig { reason: String },

    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    InvalidShape { expected: Vec<usize>, actual: Vec<usize> },

    #[error("Sequence too long: requested {requested}, max supported {max_supported}")]
    SequenceTooLong { requested: usize, max_supported: usize },

    #[error("KV cache overflow: current={current_len}, new={new_len}, max={max_len}")]
    CacheOverflow { current_len: usize, new_len: usize, max_len: usize },

    #[error("Memory allocation failed: {reason}")]
    MemoryError { reason: String },

    #[error("Attention computation failed: {operation} - {reason}")]
    ComputationError { operation: String, reason: String },

    #[error("Device mismatch: {details}")]
    DeviceError { details: String },

    #[error("Quantization error in attention: {context}")]
    QuantizationError { context: String },
}

impl From<AttentionError> for crate::InferenceError {
    fn from(err: AttentionError) -> Self {
        crate::InferenceError::AttentionError(err)
    }
}
```

### Input Validation

```rust
impl BitNetMultiHeadAttention {
    /// Validate inputs before forward pass
    fn validate_forward_inputs(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        kv_cache: Option<&LayerKVCache>,
    ) -> Result<()> {
        let dims = hidden_states.dims();

        // Check tensor rank
        if dims.len() != 3 {
            return Err(AttentionError::InvalidShape {
                expected: vec![1, 1, self.hidden_size], // placeholder batch/seq
                actual: dims.to_vec(),
            });
        }

        // Check hidden dimension
        if dims[2] != self.hidden_size {
            return Err(AttentionError::InvalidShape {
                expected: vec![dims[0], dims[1], self.hidden_size],
                actual: dims.to_vec(),
            });
        }

        // Check attention mask shape if provided
        if let Some(mask) = attention_mask {
            let mask_dims = mask.dims();
            let expected_mask_shape = [dims[0], 1, dims[1], dims[1]]; // [B, 1, T, T]

            if mask_dims.len() != 4 || mask_dims != expected_mask_shape {
                return Err(AttentionError::InvalidShape {
                    expected: expected_mask_shape.to_vec(),
                    actual: mask_dims.to_vec(),
                });
            }
        }

        // Validate KV cache compatibility
        if let Some(cache) = kv_cache {
            if cache.n_kv_heads != self.n_kv_heads {
                return Err(AttentionError::InvalidConfig {
                    reason: format!(
                        "KV cache has {} heads, attention layer expects {}",
                        cache.n_kv_heads, self.n_kv_heads
                    ),
                });
            }

            if cache.head_dim != self.head_dim {
                return Err(AttentionError::InvalidConfig {
                    reason: format!(
                        "KV cache has head_dim {}, attention layer expects {}",
                        cache.head_dim, self.head_dim
                    ),
                });
            }

            if cache.batch_size != dims[0] {
                return Err(AttentionError::InvalidShape {
                    expected: vec![cache.batch_size, dims[1], dims[2]],
                    actual: dims.to_vec(),
                });
            }
        }

        // Check device compatibility
        if hidden_states.device() != &self.device {
            return Err(AttentionError::DeviceError {
                details: format!(
                    "Input tensor on {:?}, attention layer on {:?}",
                    hidden_states.device(),
                    self.device
                ),
            });
        }

        Ok(())
    }

    /// Validate numerical stability of attention computation
    fn validate_attention_stability(&self, attention_weights: &Tensor) -> Result<()> {
        #[cfg(debug_assertions)]
        {
            let weights_data: Vec<f32> = attention_weights.flatten_all()?.to_vec1()?;

            // Check for NaN/Inf values
            if weights_data.iter().any(|&x| !x.is_finite()) {
                return Err(AttentionError::ComputationError {
                    operation: "attention_softmax".to_string(),
                    reason: "Attention weights contain NaN or Inf values".to_string(),
                });
            }

            // Check attention weight sums (should sum to ~1.0 per head)
            let seq_len = attention_weights.dims()[3];
            for chunk in weights_data.chunks(seq_len) {
                let sum: f32 = chunk.iter().sum();
                if (sum - 1.0).abs() > 0.01 {
                    log::warn!("Attention weights sum to {:.4}, expected ~1.0", sum);
                }
            }
        }

        Ok(())
    }
}
```

## Testing Strategy

### Unit Tests with AC Coverage

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_head_attention_forward() { // AC:2
        let config = create_test_attention_config();
        let attention = BitNetMultiHeadAttention::new(&config, &Device::Cpu).unwrap();

        let batch_size = 2;
        let seq_len = 10;
        let hidden_size = config.hidden_size;
        let input = Tensor::randn(0.0, 1.0, &[batch_size, seq_len, hidden_size], &Device::Cpu).unwrap();

        let output = attention.forward(&input, None, None).unwrap();

        assert_eq!(output.hidden_states.shape(), [batch_size, seq_len, hidden_size]);

        // Verify no NaN/Inf values
        let output_data: Vec<f32> = output.hidden_states.flatten_all().unwrap().to_vec1().unwrap();
        assert!(output_data.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_grouped_query_attention() { // AC:2
        let mut config = create_test_attention_config();
        config.num_attention_heads = 8;
        config.num_key_value_heads = 2; // 4x compression

        let attention = BitNetMultiHeadAttention::new(&config, &Device::Cpu).unwrap();
        let input = create_test_input([1, 5, config.hidden_size]).unwrap();

        let output = attention.forward(&input, None, None).unwrap();

        assert_eq!(output.hidden_states.shape(), [1, 5, config.hidden_size]);

        // Test KV head expansion
        let dummy_kv = Tensor::randn(0.0, 1.0, &[1, 2, 5, 64], &Device::Cpu).unwrap();
        let expanded = attention.expand_kv_for_gqa(&dummy_kv).unwrap();
        assert_eq!(expanded.shape(), [1, 8, 5, 64]); // Expanded to match Q heads
    }

    #[test]
    fn test_kv_cache_append_and_retrieval() { // AC:5
        let mut cache = LayerKVCache::new(1, 4, 100, 64, &Device::Cpu, KVCacheFormat::Standard).unwrap();

        // First append
        let k1 = Tensor::ones(&[1, 4, 5, 64], DType::F32, &Device::Cpu).unwrap();
        let v1 = Tensor::ones(&[1, 4, 5, 64], DType::F32, &Device::Cpu).unwrap();
        cache.append(&k1, &v1).unwrap();
        assert_eq!(cache.seq_len, 5);

        // Second append
        let k2 = Tensor::ones(&[1, 4, 3, 64], DType::F32, &Device::Cpu).unwrap();
        let v2 = Tensor::ones(&[1, 4, 3, 64], DType::F32, &Device::Cpu).unwrap();
        cache.append(&k2, &v2).unwrap();
        assert_eq!(cache.seq_len, 8);

        // Verify cache shapes
        assert_eq!(cache.k.shape(), [1, 4, 8, 64]);
        assert_eq!(cache.v.shape(), [1, 4, 8, 64]);
    }

    #[test]
    fn test_rotary_embedding_application() { // AC:2
        let rope = RotaryEmbedding::new(64, 100, 10000.0, &Device::Cpu).unwrap();
        let input = Tensor::randn(0.0, 1.0, &[1, 4, 10, 64], &Device::Cpu).unwrap();

        let rotated = rope.apply_rotary_embedding(&input, 0).unwrap();
        assert_eq!(rotated.shape(), input.shape());

        // Test with position offset (simulating KV cache)
        let rotated_offset = rope.apply_rotary_embedding(&input, 5).unwrap();
        assert_eq!(rotated_offset.shape(), input.shape());

        // Rotated output should be different from input (unless input is zero)
        let input_data: Vec<f32> = input.flatten_all().unwrap().to_vec1().unwrap();
        let rotated_data: Vec<f32> = rotated.flatten_all().unwrap().to_vec1().unwrap();

        if input_data.iter().any(|&x| x.abs() > 1e-6) {
            assert_ne!(input_data, rotated_data, "RoPE should change the input");
        }
    }

    #[test]
    fn test_causal_mask_generation() { // AC:2
        let config = create_test_attention_config();
        let attention = BitNetMultiHeadAttention::new(&config, &Device::Cpu).unwrap();

        // Test causal mask for different sequence lengths
        let mask = attention.create_causal_mask(3, 5, &Device::Cpu).unwrap(); // 3 queries, 5 keys
        assert_eq!(mask.shape(), [1, 1, 3, 5]);

        let mask_data: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();

        // Check that future positions are masked (set to -inf)
        // For query 0 (position 2 in total context): positions 3,4 should be masked
        assert_eq!(mask_data[3], f32::NEG_INFINITY); // position 3
        assert_eq!(mask_data[4], f32::NEG_INFINITY); // position 4

        // For query 2 (position 4 in total context): no masking needed
        assert_ne!(mask_data[10], f32::NEG_INFINITY); // last position should be visible
    }

    #[test]
    fn test_quantized_attention_accuracy() { // AC:4
        let config = create_test_attention_config();
        let quantized_attention = BitNetMultiHeadAttention::new(&config, &Device::Cpu).unwrap();

        // Create FP32 reference for comparison
        let fp32_attention = create_fp32_reference_attention(&config).unwrap();

        let input = create_test_input([1, 10, config.hidden_size]).unwrap();

        let quantized_output = quantized_attention.forward(&input, None, None).unwrap();
        let fp32_output = fp32_attention.forward(&input, None, None).unwrap();

        let correlation = compute_tensor_correlation(&quantized_output.hidden_states, &fp32_output).unwrap();
        assert!(correlation > 0.99, "Quantized attention accuracy too low: {:.4}", correlation);
    }

    #[test]
    fn test_attention_memory_usage() { // AC:5
        let config = create_test_attention_config();
        let attention = BitNetMultiHeadAttention::new(&config, &Device::Cpu).unwrap();

        let mut cache = LayerKVCache::new(1, config.num_key_value_heads, 1024, config.head_dim(), &Device::Cpu, KVCacheFormat::Standard).unwrap();

        let memory_usage = cache.memory_usage();
        let expected_usage = 1 * config.num_key_value_heads * 1024 * config.head_dim() * 4 * 2; // K + V tensors, FP32

        assert_eq!(memory_usage, expected_usage);
        assert!(memory_usage < 10 * 1024 * 1024, "Memory usage too high: {} bytes", memory_usage);
    }

    #[test]
    fn test_error_handling_shape_mismatch() { // AC:10
        let config = create_test_attention_config();
        let attention = BitNetMultiHeadAttention::new(&config, &Device::Cpu).unwrap();

        let wrong_input = create_test_input([1, 10, config.hidden_size + 1]).unwrap(); // Wrong dimension
        let result = attention.forward(&wrong_input, None, None);

        assert!(result.is_err());
        match result.unwrap_err().downcast::<AttentionError>() {
            Ok(AttentionError::InvalidShape { expected, actual }) => {
                assert_eq!(actual[2], config.hidden_size + 1);
                assert_eq!(expected[2], config.hidden_size);
            }
            _ => panic!("Expected InvalidShape error"),
        }
    }

    // Helper functions for testing
    fn create_test_attention_config() -> AttentionConfig {
        AttentionConfig {
            hidden_size: 512,
            num_attention_heads: 8,
            num_key_value_heads: 8,
            max_position_embeddings: 2048,
            rope_theta: 10000.0,
            attention_dropout: 0.0,
            scale_attn_weights: true,
            quantization_type: QuantizationType::I2S,
        }
    }

    fn create_test_input(shape: [usize; 3]) -> Result<Tensor> {
        Tensor::randn(0.0, 1.0, &shape, &Device::Cpu)
    }
}
```

This comprehensive multi-head attention specification provides a production-ready implementation that integrates seamlessly with BitNet.rs quantization infrastructure while delivering high performance and accuracy for transformer-based neural network inference.

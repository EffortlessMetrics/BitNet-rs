# CPU Inference Architecture

**Issue:** #462 - CPU Forward Pass with Real Inference
**Status:** Specification
**Date:** 2025-10-14

## Context

BitNet-rs currently returns placeholder logits (zeros `[1, 32000]`) from the CPU forward path in `CpuInferenceEngine::forward_parallel()`. This blocks actual token generation and question-answering workflows, preventing the CPU MVP from performing real neural network inference.

**Current State:**
- `CpuInferenceEngine::forward_parallel()` at line 263 returns zero-filled tensors
- QuantizedLinear layer selection functional (I2S/TL1/TL2 paths ready)
- CLI can load GGUF models and create inference engines
- Strict mode enforces quantization-only hot paths
- KV cache structures exist but not populated during forward pass

**Required Transformation:**
- Placeholder → Real transformer forward pass
- Zero logits → Computed non-zero finite logits
- Empty KV cache → Populated K,V tensors per layer
- No token generation → Autoregressive decode loop

## Design

### Neural Network Architecture

BitNet-rs implements a standard transformer decoder architecture with 1-bit quantized weights:

```
Input Token (u32)
    ↓
[Embedding Layer] → x: [1, d_model]
    ↓
┌───────────────────────────────────────┐
│ Transformer Layer (0..num_layers)     │
│                                        │
│  Pre-LayerNorm                         │
│    ↓                                   │
│  [Attention Block]                     │
│    • Q/K/V Projection (QuantizedLinear)│
│    • KV Cache Update (append)          │
│    • Attention: softmax(Q·K^T/√d_h)·V  │
│    • Causal Masking                    │
│    • Output Projection (QuantizedLinear)│
│    ↓                                   │
│  Residual: x = x + attn_out            │
│    ↓                                   │
│  Pre-LayerNorm                         │
│    ↓                                   │
│  [FFN Block]                           │
│    • Gate Projection (QuantizedLinear) │
│    • Up Projection (QuantizedLinear)   │
│    • SwiGLU Activation                 │
│    • Down Projection (QuantizedLinear) │
│    ↓                                   │
│  Residual: x = x + ffn_out             │
└───────────────────────────────────────┘
    ↓
Final LayerNorm
    ↓
[LM Head] → logits: [1, vocab_size]
    ↓
Sampling → next_token
```

### Forward Pass Implementation

#### Single-Step Autoregressive Decode

The `forward_parallel()` function processes one token at a time in autoregressive mode:

**Input:**
- `input`: BitNetTensor containing token ID(s) or embedding
- `step`: Current sequence position (0-indexed)

**Output:**
- `logits`: BitNetTensor `[1, vocab_size]` with probability distributions

**Algorithm:**

```rust
fn forward_parallel(&self, input: &BitNetTensor, step: usize) -> Result<BitNetTensor> {
    // 1. Embedding lookup (if input is token IDs)
    let mut x = self.embed_token(input)?; // [1, d_model]

    // 2. Process each transformer layer
    for layer_idx in 0..self.config.num_layers {
        x = self.apply_layer(&x, layer_idx, step)?;
    }

    // 3. Final LayerNorm
    x = self.apply_layer_norm(&x, "model.norm")?;

    // 4. LM head projection
    let logits = self.compute_logits(&x)?; // [1, vocab_size]

    Ok(logits)
}
```

#### Layer Processing

Each transformer layer applies attention and FFN with residual connections:

```rust
fn apply_layer(&self, x: &BitNetTensor, layer_idx: usize, step: usize) -> Result<BitNetTensor> {
    // Pre-attention LayerNorm
    let x_norm = self.apply_layer_norm(x, &format!("model.layers.{}.input_layernorm", layer_idx))?;

    // Attention block
    let attn_out = self.apply_attention(&x_norm, layer_idx, step)?;

    // First residual connection
    let x = x.add(&attn_out)?;

    // Pre-FFN LayerNorm
    let x_norm = self.apply_layer_norm(&x, &format!("model.layers.{}.post_attention_layernorm", layer_idx))?;

    // FFN block
    let ffn_out = self.apply_ffn(&x_norm, layer_idx)?;

    // Second residual connection
    let x = x.add(&ffn_out)?;

    Ok(x)
}
```

#### Attention Mechanism

Multi-head attention with KV cache and causal masking:

```rust
fn apply_attention(&self, x: &BitNetTensor, layer_idx: usize, step: usize) -> Result<BitNetTensor> {
    let H = self.config.num_heads;
    let Dh = self.config.head_dim;

    // Q/K/V projections via QuantizedLinear (I2S/TL1/TL2)
    let q = self.quantized_linear(x, &format!("model.layers.{}.self_attn.q_proj", layer_idx))?; // [1, H*Dh]
    let k = self.quantized_linear(x, &format!("model.layers.{}.self_attn.k_proj", layer_idx))?; // [1, H*Dh]
    let v = self.quantized_linear(x, &format!("model.layers.{}.self_attn.v_proj", layer_idx))?; // [1, H*Dh]

    // Reshape to multi-head format
    let q = q.reshape(&[1, H, Dh])?; // [1, H, Dh]
    let k = k.reshape(&[1, H, Dh])?; // [1, H, Dh]
    let v = v.reshape(&[1, H, Dh])?; // [1, H, Dh]

    // Apply RoPE (if configured)
    let (q, k) = self.apply_rope(q, k, step)?;

    // Update KV cache: append current K,V at position `step`
    self.kv_cache.update(layer_idx, k.clone(), v.clone(), step)?;

    // Retrieve full KV cache for this layer (0..=step)
    let (k_cache, v_cache) = self.kv_cache.get(layer_idx)?; // [step+1, H, Dh]

    // Compute attention scores: Q @ K^T / sqrt(Dh)
    let scores = q.matmul(&k_cache.transpose(0, 2)?)? // [1, H, step+1]
        .div_scalar((Dh as f32).sqrt())?;

    // Apply causal mask (only attend to positions 0..=step)
    let scores = self.apply_causal_mask(&scores, step)?;

    // Softmax over sequence dimension
    let attn_weights = scores.softmax(-1)?; // [1, H, step+1]

    // Apply attention to values: attn_weights @ V
    let attn_out = attn_weights.matmul(&v_cache)?; // [1, H, Dh]

    // Reshape back to [1, H*Dh]
    let attn_out = attn_out.reshape(&[1, H * Dh])?;

    // Output projection via QuantizedLinear
    let out = self.quantized_linear(&attn_out, &format!("model.layers.{}.self_attn.o_proj", layer_idx))?;

    Ok(out)
}
```

#### FFN Block (SwiGLU)

Feed-forward network with gating:

```rust
fn apply_ffn(&self, x: &BitNetTensor, layer_idx: usize) -> Result<BitNetTensor> {
    // Gate and Up projections (parallel)
    let gate = self.quantized_linear(x, &format!("model.layers.{}.mlp.gate_proj", layer_idx))?;
    let up = self.quantized_linear(x, &format!("model.layers.{}.mlp.up_proj", layer_idx))?;

    // SwiGLU activation: gate.silu() * up
    let hidden = gate.silu()?.mul(&up)?;

    // Down projection
    let out = self.quantized_linear(&hidden, &format!("model.layers.{}.mlp.down_proj", layer_idx))?;

    Ok(out)
}
```

### KV Cache Management

#### Cache Structure

Per-layer key-value cache for autoregressive generation:

```rust
pub struct KVCache {
    k_cache: Vec<BitNetTensor>, // [num_layers] of [max_seq_len, num_heads, head_dim]
    v_cache: Vec<BitNetTensor>, // [num_layers] of [max_seq_len, num_heads, head_dim]
    max_seq_len: usize,
    current_len: usize,
}
```

**Memory Layout:**
- Each layer stores K,V tensors with shape `[max_seq_len, H, Dh]`
- At step `t`, write to `cache[layer].k[t, :, :]` and `cache[layer].v[t, :, :]`
- Read range `0..=t` for attention computation

#### Cache Update Semantics

**Append Operation:**
```rust
pub fn update(&mut self, layer_idx: usize, k: BitNetTensor, v: BitNetTensor, step: usize) -> Result<()> {
    // Bounds check
    if step >= self.max_seq_len {
        anyhow::bail!("Sequence position {} exceeds max_seq_len {}", step, self.max_seq_len);
    }

    // Write K,V to cache at position `step`
    // In-place slice assignment: cache[layer].k[step, :, :] = k
    let k_cache_slice = self.k_cache[layer_idx].slice_mut(step)?;
    k_cache_slice.copy_from(&k)?;

    let v_cache_slice = self.v_cache[layer_idx].slice_mut(step)?;
    v_cache_slice.copy_from(&v)?;

    // Update current length
    self.current_len = step + 1;

    Ok(())
}
```

**Retrieval Operation:**
```rust
pub fn get(&self, layer_idx: usize) -> Result<(BitNetTensor, BitNetTensor)> {
    // Return sliced view [0..current_len, :, :] to avoid processing padding
    let k_slice = self.k_cache[layer_idx].slice(0..self.current_len)?;
    let v_slice = self.v_cache[layer_idx].slice(0..self.current_len)?;

    Ok((k_slice, v_slice))
}
```

#### Memory Budget

**Per-Layer Memory:**
```
k_cache: max_seq_len × num_heads × head_dim × sizeof(f32)
v_cache: max_seq_len × num_heads × head_dim × sizeof(f32)
```

**Example (BitNet-2B):**
- `max_seq_len = 2048`
- `num_heads = 32`
- `head_dim = 64`
- `num_layers = 28`

Total KV cache: `2 × 2048 × 32 × 64 × 4 bytes × 28 layers ≈ 896 MB`

### Quantization Integration

#### QuantizedLinear Dispatch

All weight projections (Q/K/V, FFN, LM head) use `QuantizedLinear::forward()`:

```rust
async fn quantized_linear(&self, input: &BitNetTensor, weight_name: &str) -> Result<BitNetTensor> {
    let layer = self.model.get_layer(weight_name)?;
    let qlinear: &QuantizedLinear = layer.as_quantized_linear()?;

    // Dispatch to quantization-specific kernel
    let output = qlinear.forward(input).await?;

    Ok(output)
}
```

**QuantizedLinear Internal Dispatch:**
```rust
pub async fn forward(&self, input: &Tensor) -> Result<Tensor> {
    match self.quantization_type {
        QuantizationType::I2S => self.quantized_matmul_i2s(input, &self.provider).await?,
        QuantizationType::TL1 => self.quantized_matmul_tl1(input, &self.provider).await?,
        QuantizationType::TL2 => self.quantized_matmul_tl2(input, &self.provider).await?,
        _ => anyhow::bail!("Unsupported quantization type: {:?}", self.quantization_type),
    }
}
```

#### Strict Mode Enforcement

No FP32 staging in hot path (AC1 requirement):

```rust
// In QuantizedLinear::forward()
if StrictModeConfig::get().enforce_quantized_inference {
    // Block fallback to FP32 dequantization
    if self.quantization_type == QuantizationType::Unknown {
        anyhow::bail!("Strict mode: unknown quantization type not allowed");
    }

    // Ensure native quantized kernel path
    if !self.has_native_kernel() {
        anyhow::bail!("Strict mode: no native quantized kernel for {:?}", self.quantization_type);
    }
}
```

**Environment Variable:**
```bash
BITNET_STRICT_MODE=1 cargo run -p bitnet-cli --features cpu -- run --model model.gguf
```

### Priming and Decode Loops

#### Priming Phase (CLI)

Tokenize prompt and populate KV cache:

```rust
pub fn prime_cache(engine: &InferenceEngine, tokens: &[u32]) -> Result<()> {
    for (step, &token) in tokens.iter().enumerate() {
        // Convert token to tensor
        let input = BitNetTensor::from_slice(&[token], &[1], DType::U32, &Device::Cpu)?;

        // Forward pass to update KV cache (discard logits)
        let _logits = engine.forward(&input, step)?;
    }

    Ok(())
}
```

**Timing:**
- Measure: `prefill_tps = prompt_tokens / prefill_duration`
- Cache state: KV populated for positions `0..prompt_len`

#### Decode Loop (CLI)

Generate new tokens autoregressively:

```rust
pub fn decode_loop(
    engine: &InferenceEngine,
    sampler: &Sampler,
    max_tokens: usize,
    start_step: usize,
) -> Result<Vec<u32>> {
    let mut generated_tokens = Vec::new();
    let mut step = start_step;

    loop {
        // Forward pass with current KV cache
        let logits = engine.forward_last_token(step)?; // [1, vocab_size]

        // Sample next token
        let next_token = sampler.sample(&logits)?;

        // Check for EOS or max length
        if next_token == EOS_TOKEN || generated_tokens.len() >= max_tokens {
            break;
        }

        // Append to output
        generated_tokens.push(next_token);

        // Print streaming output
        print_token(next_token);

        // Update step
        step += 1;
    }

    Ok(generated_tokens)
}
```

**Timing:**
- Measure: `decode_tps = generated_tokens / decode_duration`
- Per-token latency: `1 / decode_tps`

### Performance Characteristics

#### CPU Baseline Targets

**Tiny Model (500M parameters):**
- Throughput: ≥10 tok/s
- First token latency: ≤1s
- Memory: KV cache ≤512 MB

**2B Model:**
- Throughput: ≥5 tok/s (target from issue spec)
- First token latency: ≤2s
- Memory: KV cache ≤1 GB

**Platform Assumptions:**
- Modern x86_64 CPU with AVX2/AVX-512
- Or ARM64 with NEON SIMD
- 16+ GB system RAM

#### Optimization Strategies

**SIMD Kernels:**
- I2S matmul: AVX2 vectorized 2-bit unpacking
- TL1/TL2: SIMD table lookups
- Attention: Vectorized softmax and matmul

**Memory Efficiency:**
- Zero-copy model loading (mmap GGUF)
- In-place KV cache updates
- Lazy tensor allocation

**Parallelism:**
- Rayon thread pool for batch processing
- Single-threaded decode for determinism
- Optional multi-head attention parallelism

### Deterministic Inference

#### Configuration

```bash
# Full deterministic setup
BITNET_DETERMINISTIC=1 \
BITNET_SEED=42 \
RAYON_NUM_THREADS=1 \
cargo run -p bitnet-cli --features cpu -- \
  run --model model.gguf --prompt "Test" --temperature 0.0
```

**Environment Variables:**
- `BITNET_DETERMINISTIC=1`: Enable deterministic mode
- `BITNET_SEED=42`: Set RNG seed for sampling
- `RAYON_NUM_THREADS=1`: Single-threaded execution
- `--temperature 0.0`: Greedy decoding (argmax)

#### Reproducibility Guarantees

**Deterministic Components:**
- Embedding lookup: Same input → same output
- QuantizedLinear: Quantized matmul deterministic
- Attention: Floating-point order preserved
- Sampling: Fixed seed → reproducible token sequence

**Non-Deterministic Risks:**
- Multi-threaded Rayon execution (mitigate with `RAYON_NUM_THREADS=1`)
- GPU async execution (CPU-only for determinism)
- FP32 accumulation order (avoid via strict quantization)

## Validation

### Acceptance Criteria Coverage

**AC1: CPU Forward Pass Real Inference**
- ✅ Implements full transformer forward pass
- ✅ Uses QuantizedLinear I2S/TL1/TL2 paths
- ✅ KV cache management with append semantics
- ✅ Returns non-zero finite logits `[1, vocab_size]`
- ✅ BOS token test: `forward(BOS, step=0)` → valid logits

**AC2: CLI Priming and Decode Loop**
- ✅ Priming loop: tokenize → forward each token → update KV cache
- ✅ Decode loop: logits → sample → forward → print → repeat
- ✅ Question-answering workflow: "Q: What is 2+2? A:" → "4"

**Validation Tests:**
```bash
# Unit test: BOS token forward pass
cargo test -p bitnet-inference test_cpu_forward_bos_nonzero --features cpu

# Integration test: 16-token greedy decode
cargo test -p bitnet-inference test_ac1_greedy_decode_16_tokens --features cpu

# E2E test: CLI question answering
cargo test -p bitnet-cli test_ac2_cli_inference_question_answering --features cpu
```

### Cross-Validation

**C++ Reference Comparison:**
```bash
# Run cross-validation against Microsoft BitNet C++
cargo run -p xtask -- crossval --model model.gguf --prompt "Test input"

# Tolerance: cosine_similarity ≥ 0.99 (1% error budget)
```

**Validation Metrics:**
- Logits output: Cosine similarity ≥0.99 vs C++ reference
- Token sequence: Exact match for greedy decoding (temperature=0.0)
- KV cache: Shape and value validation per layer

## References

### Related Documentation

- `docs/architecture-overview.md` - System design overview
- `docs/reference/quantization-support.md` - I2S/TL1/TL2 algorithms
- `docs/development/validation-framework.md` - Testing infrastructure
- `docs/performance-benchmarking.md` - Throughput measurement

### Existing Code Patterns

- `crates/bitnet-inference/src/cpu.rs:263` - Placeholder forward_parallel()
- `crates/bitnet-inference/src/layers/quantized_linear.rs:488` - I2S/TL1/TL2 dispatch
- `crates/bitnet-inference/src/layers/attention.rs` - KV cache structure
- `crates/bitnet-inference/src/sampling.rs` - Token sampling algorithms
- `crates/bitnet-common/src/strict_mode.rs` - Strict mode enforcement

### Issue References

- **Issue #462:** CPU Forward Pass with Real Inference (this issue)
- **Issue #439:** GPU determinism and fake device testing
- **Issue #254:** Real inference specification (predecessor)
- **PR #461:** Strict quantized hot-path enforcement
- **PR #452:** Receipt verification gate

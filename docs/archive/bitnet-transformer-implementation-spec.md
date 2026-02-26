# BitNet Transformer Implementation Specification

**Issue**: #248 - Implement Real Neural Network Inference (Currently Using Mock Implementation)
**Status**: Implementation Ready
**Scope**: Transform mock implementations into production-grade quantized transformer computation

## Executive Summary

This specification defines the complete implementation of real neural network inference in BitNet-rs, replacing mock implementations with actual transformer computation using quantized weights. The implementation leverages existing infrastructure (GGUF loading, I2S/TL1/TL2 quantization, universal tokenizers) while introducing production-grade transformer blocks, attention mechanisms, and autoregressive generation.

**Key Deliverables:**
- Production-grade BitNetTransformer with quantized linear layers
- Multi-head attention with KV-cache optimization and rotary embeddings
- Autoregressive text generation with deterministic sampling
- Cross-validation framework ensuring >99% quantization accuracy
- Performance targets: 5-15 tok/sec CPU, 15-45 tok/sec GPU (BitNet 2B model)

## Architecture Overview

### Transformer Pipeline Architecture

```
Input Tokens → Embedding Layer → Transformer Blocks → Layer Norm → LM Head → Logits → Sampling → Output
      ↓              ↓                    ↓               ↓         ↓        ↓         ↓
   [u32; N]    [B, T, H]          [B, T, H] × L      [B, T, H]  [B, T, V] [B, V]  [u32]
```

**Pipeline Components:**
1. **Token Embedding**: Quantization-aware embedding lookup with memory-mapped GGUF weights
2. **Transformer Blocks**: N layers of self-attention + feed-forward with residual connections
3. **Attention Mechanism**: Multi-head attention with quantized Q, K, V projections and KV-cache
4. **Feed-Forward Network**: SiLU-activated gated linear transformations with quantized weights
5. **Output Projection**: LM head or tied embeddings producing vocabulary logits
6. **Generation Loop**: Temperature, top-k, nucleus sampling with deterministic seeding

### Quantization-First Design

**Core Principle**: All linear operations use quantized weights (I2S, TL1, TL2) with device-aware kernel selection.

```rust
// Quantized Linear Layer Pattern
struct QuantizedLinear {
    weights: QuantizedTensor,        // I2S/TL1/TL2 quantized weights
    bias: Option<Tensor>,            // Optional bias (FP32)
    quantizer: DeviceAwareQuantizer, // CPU/GPU kernel selection
}

impl QuantizedLinear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Device-aware quantized matrix multiplication
        self.quantizer.matmul_quantized(input, &self.weights)
    }
}
```

## Component Specifications

### 1. BitNetTransformer Core

**Location**: `bitnet-inference/src/transformer.rs`

```rust
pub struct BitNetTransformer {
    config: BitNetConfig,
    embed_tokens: QuantizedEmbedding,  // Quantized token embeddings
    layers: Vec<BitNetTransformerBlock>, // Transformer layers
    norm: RMSNorm,                     // Final layer normalization
    lm_head: Option<QuantizedLinear>,  // Output projection (or tied)
    device: Device,                    // CPU/GPU device context
    quantizer: DeviceAwareQuantizer,   // Kernel selection
}

impl BitNetTransformer {
    // AC1: Real transformer forward pass with quantized weights
    pub fn forward(&self, input_ids: &Tensor, kv_cache: Option<&mut KVCache>) -> Result<Tensor>;

    // AC3: Autoregressive generation with sampling
    pub fn generate(&mut self, input_ids: &[u32], config: &GenerationConfig) -> Result<Vec<u32>>;

    // AC7: Deterministic inference with seeding
    pub fn generate_deterministic(&mut self, input_ids: &[u32], seed: u64) -> Result<Vec<u32>>;
}
```

**Key Features:**
- **Zero-Copy Loading**: Memory-mapped GGUF tensors without duplication
- **Quantization Integration**: All linear layers use I2S/TL1/TL2 quantized weights
- **Device Awareness**: Automatic GPU/CPU kernel selection with graceful fallback
- **Memory Optimization**: KV-cache for efficient autoregressive generation

### 2. Multi-Head Attention with Quantization

**Location**: `bitnet-inference/src/attention.rs`

```rust
pub struct BitNetAttention {
    n_heads: usize,
    n_kv_heads: usize,           // Grouped Query Attention support
    head_dim: usize,
    scale: f32,                  // 1/sqrt(head_dim)

    // Quantized projection layers
    q_proj: QuantizedLinear,     // Query projection [H, H]
    k_proj: QuantizedLinear,     // Key projection [H, KV_H * head_dim]
    v_proj: QuantizedLinear,     // Value projection [H, KV_H * head_dim]
    o_proj: QuantizedLinear,     // Output projection [H, H]

    rope: Option<RotaryEmbedding>, // Rotary positional embeddings
}

impl BitNetAttention {
    // AC2: Multi-head attention with quantized projections
    pub fn forward(&self, hidden: &Tensor, kv_cache: Option<&mut LayerKVCache>) -> Result<Tensor> {
        // 1. Project to Q, K, V using quantized linear layers
        let q = self.q_proj.forward(hidden)?;  // [B, T, H]
        let k = self.k_proj.forward(hidden)?;  // [B, T, KV_H * head_dim]
        let v = self.v_proj.forward(hidden)?;  // [B, T, KV_H * head_dim]

        // 2. Reshape to multi-head format
        let q = q.reshape(&[batch, seq_len, self.n_heads, self.head_dim])?.transpose(1, 2)?;
        let k = k.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?.transpose(1, 2)?;
        let v = v.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?.transpose(1, 2)?;

        // 3. Apply rotary embeddings if configured
        let (q, k) = if let Some(rope) = &self.rope {
            let pos = kv_cache.as_ref().map(|c| c.seq_len).unwrap_or(0);
            (rope.apply(&q, pos)?, rope.apply(&k, pos)?)
        } else { (q, k) };

        // 4. Update KV cache for autoregressive generation
        let (k_cached, v_cached) = if let Some(cache) = kv_cache {
            cache.append(&k, &v)?;
            (cache.k.clone(), cache.v.clone())
        } else { (k, v) };

        // 5. Grouped Query Attention: expand K/V to match Q heads
        let k_expanded = self.expand_kv_heads(&k_cached)?;  // [B, H, T, D]
        let v_expanded = self.expand_kv_heads(&v_cached)?;  // [B, H, T, D]

        // 6. Scaled dot-product attention with causal masking
        let scores = q.matmul(&k_expanded.transpose(-2, -1)?)? * self.scale;
        let masked_scores = self.apply_causal_mask(&scores)?;
        let attn_weights = softmax(&masked_scores, -1)?;
        let attn_output = attn_weights.matmul(&v_expanded)?;

        // 7. Reshape and project output
        let output = attn_output.transpose(1, 2)?.reshape(&[batch, seq_len, hidden_size])?;
        self.o_proj.forward(&output)
    }
}
```

**Quantization Integration Points:**
- **Linear Projections**: Q, K, V, O projections use quantized weights with device-aware kernels
- **Memory Efficiency**: KV-cache optimization reduces memory allocation in autoregressive generation
- **Attention Accuracy**: Maintains numerical stability with quantized operations (>99% accuracy)

### 3. Feed-Forward Network with Quantized Layers

**Location**: `bitnet-inference/src/feed_forward.rs`

```rust
pub struct BitNetFeedForward {
    hidden_size: usize,
    intermediate_size: usize,

    // Quantized linear layers
    gate_proj: QuantizedLinear,    // Gating projection [H, I]
    up_proj: QuantizedLinear,      // Up projection [H, I]
    down_proj: QuantizedLinear,    // Down projection [I, H]
}

impl BitNetFeedForward {
    pub fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        // SiLU-gated feed-forward computation
        let gate = silu(&self.gate_proj.forward(hidden)?)?;  // Quantized projection + SiLU
        let up = self.up_proj.forward(hidden)?;              // Quantized projection
        let gated = gate.mul(&up)?;                          // Element-wise gating
        self.down_proj.forward(&gated)                       // Quantized down projection
    }
}
```

### 4. Transformer Block with Residual Connections

**Location**: `bitnet-inference/src/transformer_block.rs`

```rust
pub struct BitNetTransformerBlock {
    attention: BitNetAttention,
    feed_forward: BitNetFeedForward,
    attention_norm: RMSNorm,        // Pre-attention normalization
    ffn_norm: RMSNorm,              // Pre-FFN normalization
}

impl BitNetTransformerBlock {
    pub fn forward(&self, hidden: &Tensor, kv_cache: Option<&mut LayerKVCache>) -> Result<Tensor> {
        // Pre-norm attention with residual connection
        let attn_input = self.attention_norm.forward(hidden)?;
        let attn_output = self.attention.forward(&attn_input, kv_cache)?;
        let after_attn = (hidden + &attn_output)?;  // Residual connection

        // Pre-norm feed-forward with residual connection
        let ffn_input = self.ffn_norm.forward(&after_attn)?;
        let ffn_output = self.feed_forward.forward(&ffn_input)?;
        let after_ffn = (&after_attn + &ffn_output)?;  // Residual connection

        Ok(after_ffn)
    }
}
```

### 5. KV-Cache Optimization for Generation

**Location**: `bitnet-inference/src/kv_cache.rs`

```rust
pub struct LayerKVCache {
    k: Tensor,                    // Key cache [B, KV_H, max_seq_len, head_dim]
    v: Tensor,                    // Value cache [B, KV_H, max_seq_len, head_dim]
    seq_len: usize,              // Current sequence length
    max_seq_len: usize,          // Maximum supported sequence length
    n_kv_heads: usize,           // Number of KV heads for validation
}

pub struct KVCache {
    layers: Vec<LayerKVCache>,    // Per-layer KV cache
}

impl KVCache {
    // Memory-efficient cache management
    pub fn allocate(config: &BitNetConfig, batch_size: usize, device: &Device) -> Result<Self>;

    // Append new tokens to cache (autoregressive generation)
    pub fn append_layer(&mut self, layer_idx: usize, k: &Tensor, v: &Tensor) -> Result<()>;

    // Clear cache for new sequence
    pub fn clear(&mut self);

    // Memory usage optimization
    pub fn memory_usage(&self) -> usize;
}
```

**Memory Optimization Features:**
- **Pre-allocated Cache**: Avoids dynamic allocation during generation
- **Grouped Query Attention**: Reduces KV cache memory by using fewer KV heads
- **Device-aware Storage**: GPU tensors stay on GPU, CPU tensors use optimized layouts

## API Contracts

### Core Inference Interface

**Location**: `bitnet-inference/src/lib.rs`

```rust
// AC5: Main inference interface with performance targets
pub trait BitNetInference {
    /// Forward pass through transformer with optional KV cache
    /// Performance target: <10ms latency for single token (CPU), <2ms (GPU)
    fn forward(&mut self, input_ids: &Tensor, kv_cache: Option<&mut KVCache>) -> Result<Tensor>;

    /// Generate text autoregressively with sampling
    /// Performance target: 5-15 tok/sec CPU, 15-45 tok/sec GPU (2B model)
    fn generate(&mut self, prompt: &[u32], max_tokens: usize, config: &GenerationConfig) -> Result<Vec<u32>>;

    /// AC7: Deterministic generation with fixed seed
    fn generate_deterministic(&mut self, prompt: &[u32], max_tokens: usize, seed: u64) -> Result<Vec<u32>>;
}

// AC10: Error handling patterns
#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    #[error("Quantization failed: {context}")]
    QuantizationError { context: String },

    #[error("Out of memory: requested {requested} bytes, available {available}")]
    OutOfMemory { requested: usize, available: usize },

    #[error("Invalid token ID: {token_id} (vocab_size: {vocab_size})")]
    InvalidToken { token_id: u32, vocab_size: usize },

    #[error("Device selection failed: {reason}")]
    DeviceError { reason: String },

    #[error("KV cache overflow: sequence length {seq_len} exceeds maximum {max_len}")]
    CacheOverflow { seq_len: usize, max_len: usize },
}
```

### Generation Configuration

```rust
// AC3: Sampling configuration for text generation
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_tokens: usize,           // Maximum tokens to generate
    pub temperature: f32,            // Sampling temperature (0.0 = deterministic)
    pub top_k: Option<usize>,        // Top-k sampling
    pub top_p: Option<f32>,          // Nucleus (top-p) sampling
    pub repetition_penalty: f32,     // Repetition penalty factor
    pub seed: Option<u64>,           // Random seed for deterministic generation
    pub stop_tokens: Vec<u32>,       // Stop token IDs
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 128,
            temperature: 0.8,
            top_k: Some(50),
            top_p: Some(0.95),
            repetition_penalty: 1.1,
            seed: None,
            stop_tokens: vec![],
        }
    }
}
```

## Workspace Integration

### Primary Changes: bitnet-inference

**Core Implementation Files:**
- `src/transformer.rs` - Main BitNetTransformer implementation
- `src/attention.rs` - Multi-head attention with quantized projections
- `src/feed_forward.rs` - Quantized feed-forward networks
- `src/transformer_block.rs` - Complete transformer blocks
- `src/generation.rs` - Autoregressive generation loop
- `src/sampling.rs` - Temperature, top-k, nucleus sampling
- `src/kv_cache.rs` - Memory-efficient KV cache management

**API Evolution:**
```rust
// BEFORE: Mock implementation
impl InferenceEngine {
    pub fn generate(&self, prompt: &str) -> Result<String> {
        Ok(format!("[Mock inference: {} tokens generated]", prompt.len()))
    }
}

// AFTER: Real transformer computation
impl InferenceEngine {
    pub fn generate(&mut self, prompt: &str) -> Result<String> {
        let tokens = self.tokenizer.encode(prompt)?;
        let generated_tokens = self.transformer.generate(&tokens, &self.config)?;
        let text = self.tokenizer.decode(&generated_tokens)?;
        Ok(text)
    }
}
```

### Secondary Changes: bitnet-kernels

**Enhanced Kernel Support:**
- `src/matmul_quantized.rs` - Quantized matrix multiplication kernels
- `src/attention_kernels.rs` - Fused attention computation (GPU)
- `src/device_selection.rs` - Enhanced GPU/CPU selection logic

**Kernel Interface Extensions:**
```rust
pub trait QuantizedKernels {
    // AC6: Device-aware quantized matrix multiplication
    fn matmul_i2s(&self, input: &Tensor, weights: &QuantizedTensor, output: &mut Tensor) -> Result<()>;
    fn matmul_tl1(&self, input: &Tensor, weights: &QuantizedTensor, output: &mut Tensor) -> Result<()>;
    fn matmul_tl2(&self, input: &Tensor, weights: &QuantizedTensor, output: &mut Tensor) -> Result<()>;

    // AC5: Fused attention kernels for GPU acceleration
    fn fused_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor>;
}
```

### Secondary Changes: bitnet-models

**Model Loading Enhancements:**
- `src/gguf_quantized.rs` - Quantized tensor loading from GGUF
- `src/tensor_mapping.rs` - Zero-copy tensor mapping for large models
- `src/model_validation.rs` - Cross-validation against reference implementations

## Performance Requirements

### Target Performance (BitNet 2B Model)

**CPU Performance:**
- **Throughput**: 5-15 tokens/second on modern CPU (8+ cores)
- **Latency**: <100ms first token, <50ms subsequent tokens
- **Memory**: <8GB RAM total, <4GB peak inference memory

**GPU Performance:**
- **Throughput**: 15-45 tokens/second on mid-range GPU (RTX 3070+)
- **Latency**: <20ms first token, <10ms subsequent tokens
- **Memory**: <6GB VRAM total, <3GB peak inference memory

**Quantization Accuracy:**
- **I2S Quantization**: >99.5% correlation with FP32 reference
- **TL1/TL2 Quantization**: >99.8% correlation with FP32 reference
- **Cross-validation**: >99.9% correlation with C++ reference (llama.cpp)

### Memory Optimization Targets

```rust
// Memory usage breakdown for BitNet 2B model
pub struct MemoryProfile {
    model_weights: usize,      // ~500MB (quantized from ~8GB FP32)
    kv_cache: usize,          // ~200MB (max_seq_len=2048, batch=1)
    activations: usize,       // ~50MB (intermediate tensors)
    total_peak: usize,        // <1GB total peak memory
}
```

**Optimization Strategies:**
- **Weight Quantization**: 75% memory reduction with I2S/TL1/TL2
- **KV Cache**: Pre-allocated cache avoids dynamic allocation
- **Zero-Copy Loading**: Memory-mapped GGUF files avoid duplication
- **Gradient Checkpointing**: Reduce activation memory in training (future)

## Cross-Validation Framework

### C++ Reference Compatibility

**Location**: `bitnet-inference/src/crossval.rs`

```rust
// AC4: Cross-validation against C++ reference implementation
pub struct CrossValidator {
    cpp_reference: CppReferenceEngine,
    rust_implementation: BitNetTransformer,
    tolerance_config: ToleranceConfig,
}

impl CrossValidator {
    // Validate single forward pass accuracy
    pub fn validate_forward_pass(&self, input_ids: &[u32]) -> Result<ValidationReport> {
        let cpp_logits = self.cpp_reference.forward(input_ids)?;
        let rust_logits = self.rust_implementation.forward(input_ids)?;

        let correlation = compute_correlation(&cpp_logits, &rust_logits)?;
        let mse = compute_mse(&cpp_logits, &rust_logits)?;

        Ok(ValidationReport {
            correlation,
            mse,
            passed: correlation > 0.999 && mse < 1e-6,
        })
    }

    // Validate generation consistency
    pub fn validate_generation(&self, prompt: &str, max_tokens: usize) -> Result<GenerationReport> {
        let config = GenerationConfig {
            temperature: 0.0,  // Deterministic
            seed: Some(42),
            max_tokens,
            ..Default::default()
        };

        let cpp_output = self.cpp_reference.generate(prompt, &config)?;
        let rust_output = self.rust_implementation.generate(prompt, &config)?;

        Ok(GenerationReport {
            cpp_tokens: cpp_output,
            rust_tokens: rust_output,
            exact_match: cpp_output == rust_output,
            token_accuracy: compute_token_accuracy(&cpp_output, &rust_output),
        })
    }
}
```

### Validation Integration

**Command Integration:**
```bash
# AC4: Cross-validation command
cargo run -p xtask -- crossval \
    --model models/bitnet/model.gguf \
    --tokenizer models/bitnet/tokenizer.json \
    --prompts crossval/test_prompts.txt \
    --tolerance 0.999

# AC7: Deterministic testing
BITNET_DETERMINISTIC=1 BITNET_SEED=42 \
cargo run -p xtask -- crossval \
    --model models/bitnet/model.gguf \
    --deterministic
```

## Feature Flag Strategy

### Build Configuration

```bash
# AC6: Device-aware builds
# CPU-only build (default features are empty)
cargo build --no-default-features --release --no-default-features --features cpu

# GPU-accelerated build
cargo build --no-default-features --release --no-default-features --features gpu

# Cross-validation build
cargo build --no-default-features --release --no-default-features --features cpu,crossval

# Development build with all features
cargo build --no-default-features --release --no-default-features --features cpu,gpu,crossval,ffi
```

### Feature Flag Implementation

**Location**: `bitnet-inference/Cargo.toml`

```toml
[features]
default = []  # Empty default features

# Core features
cpu = ["bitnet-kernels/cpu", "bitnet-quantization/cpu"]
gpu = ["bitnet-kernels/gpu", "bitnet-quantization/gpu"]

# Cross-validation
crossval = ["cpp-reference", "correlation-metrics"]

# Development features
mock = ["testing-utils"]  # Preserve mock for testing
debug = ["detailed-logging", "tensor-debugging"]
```

## Testing Strategy

### Unit Tests with AC Tag Coverage

**Location**: `bitnet-inference/tests/`

```rust
// AC1: Real transformer forward pass
#[test]
fn test_transformer_forward_pass() {  // AC:1
    let transformer = create_test_transformer()?;
    let input_ids = vec![1, 2, 3, 4, 5];
    let logits = transformer.forward(&input_ids, None)?;

    assert_eq!(logits.shape(), [1, 5, vocab_size]);
    assert!(all_finite(&logits));  // No NaN/Inf values
    assert!(logits_in_reasonable_range(&logits));
}

// AC2: Multi-head attention accuracy
#[test]
fn test_attention_quantized_accuracy() {  // AC:2
    let attention = create_test_attention()?;
    let hidden = create_test_tensor([1, 10, 512])?;

    let fp32_output = attention.forward_fp32(&hidden, None)?;
    let quantized_output = attention.forward(&hidden, None)?;

    let correlation = compute_correlation(&fp32_output, &quantized_output)?;
    assert!(correlation > 0.995, "Quantized attention accuracy too low: {}", correlation);
}

// AC3: Autoregressive generation
#[test]
fn test_autoregressive_generation() {  // AC:3
    let mut transformer = create_test_transformer()?;
    let prompt = vec![1, 2, 3];
    let config = GenerationConfig { max_tokens: 10, temperature: 0.8, ..Default::default() };

    let generated = transformer.generate(&prompt, &config)?;
    assert!(generated.len() <= config.max_tokens);
    assert!(generated.iter().all(|&token| token < transformer.vocab_size()));
}

// AC7: Deterministic generation
#[test]
fn test_deterministic_generation() {  // AC:7
    let mut transformer1 = create_test_transformer()?;
    let mut transformer2 = create_test_transformer()?;
    let prompt = vec![1, 2, 3];

    let output1 = transformer1.generate_deterministic(&prompt, 10, 42)?;
    let output2 = transformer2.generate_deterministic(&prompt, 10, 42)?;

    assert_eq!(output1, output2, "Deterministic generation not consistent");
}
```

### Integration Tests

**Location**: `bitnet-inference/tests/integration/`

```rust
// AC8: End-to-end inference pipeline
#[test]
fn test_end_to_end_inference() {  // AC:8
    let model_path = "models/test/bitnet-2b.gguf";
    let tokenizer_path = "models/test/tokenizer.json";

    let engine = InferenceEngine::load(model_path, tokenizer_path)?;
    let response = engine.generate("Hello, world!")?;

    assert!(!response.contains("[Mock inference"));  // No mock placeholders
    assert!(response.len() > 0);
    assert!(is_valid_utf8(&response));
}

// AC9: Comprehensive accuracy validation
#[test]
fn test_comprehensive_accuracy() {  // AC:9
    let test_prompts = load_test_prompts("tests/data/validation_prompts.txt")?;
    let mut passed = 0;
    let mut total = 0;

    for prompt in test_prompts {
        let result = validate_inference_accuracy(&prompt)?;
        if result.correlation > 0.999 { passed += 1; }
        total += 1;
    }

    let accuracy_rate = passed as f32 / total as f32;
    assert!(accuracy_rate > 0.95, "Accuracy validation failed: {:.2}% passed", accuracy_rate * 100.0);
}
```

### Performance Tests

**Location**: `bitnet-inference/tests/performance/`

```rust
// AC5: Performance targets validation
#[test]
fn test_cpu_performance_targets() {  // AC:5
    let mut transformer = load_bitnet_2b_model()?;
    let prompt = vec![1, 2, 3];

    let start = Instant::now();
    let _output = transformer.generate(&prompt, &GenerationConfig {
        max_tokens: 100,
        ..Default::default()
    })?;
    let duration = start.elapsed();

    let tokens_per_sec = 100.0 / duration.as_secs_f32();
    assert!(tokens_per_sec >= 5.0, "CPU performance too slow: {:.2} tok/sec", tokens_per_sec);
    assert!(tokens_per_sec <= 50.0, "Performance unrealistic: {:.2} tok/sec", tokens_per_sec);
}

#[test]
#[cfg(feature = "gpu")]
fn test_gpu_performance_targets() {  // AC:5
    let mut transformer = load_bitnet_2b_model_gpu()?;
    let prompt = vec![1, 2, 3];

    let start = Instant::now();
    let _output = transformer.generate(&prompt, &GenerationConfig {
        max_tokens: 100,
        ..Default::default()
    })?;
    let duration = start.elapsed();

    let tokens_per_sec = 100.0 / duration.as_secs_f32();
    assert!(tokens_per_sec >= 15.0, "GPU performance too slow: {:.2} tok/sec", tokens_per_sec);
}
```

## Build and Validation Commands

### Development Workflow

```bash
# 1. Build with CPU support
cargo build --no-default-features --release --no-default-features --features cpu

# 2. Run comprehensive test suite
cargo test --no-default-features --workspace --no-default-features --features cpu -- --test-threads=1

# 3. Validate transformer components individually
cargo test --no-default-features -p bitnet-inference --no-default-features --features cpu \
    test_transformer_forward_pass test_attention_accuracy test_generation

# 4. Cross-validation against C++ reference
export BITNET_GGUF="models/bitnet/model.gguf"
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
cargo run -p xtask -- crossval --model models/bitnet/model.gguf

# 5. Performance benchmarking
cargo run -p xtask -- benchmark \
    --model models/bitnet/model.gguf \
    --tokenizer models/bitnet/tokenizer.json \
    --tokens 128 --batch-size 1

# 6. GPU validation (if available)
cargo test --no-default-features --features gpu \
    -p bitnet-inference test_gpu_performance_targets
```

### CI Integration

**Location**: `.github/workflows/inference-validation.yml`

```yaml
name: Inference Validation

on: [push, pull_request]

jobs:
  test-cpu-inference:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Test CPU inference
        run: |
          cargo test --no-default-features --workspace --no-default-features --features cpu
          cargo run -p xtask -- verify --allow-mock

  test-gpu-inference:
    runs-on: gpu-runner
    steps:
      - uses: actions/checkout@v3
      - name: Test GPU inference
        run: |
          cargo test --no-default-features --features gpu
          cargo run -p xtask -- benchmark --gpu --allow-mock

  cross-validation:
    runs-on: ubuntu-latest
    steps:
      - name: Download reference model
        run: cargo run -p xtask -- download-model --id microsoft/bitnet-b1.58-2B-4T-gguf
      - name: Cross-validate
        env:
          BITNET_DETERMINISTIC: 1
          BITNET_SEED: 42
        run: cargo run -p xtask -- crossval
```

## Migration Strategy

### Phase 1: Core Infrastructure (Week 1-2)

**Tasks:**
1. Implement `BitNetTransformer` core structure with quantized linear layers
2. Create `QuantizedLinear` with device-aware kernel integration
3. Basic forward pass without attention (embedding → layers → output)
4. Unit tests for individual components with AC tags

**Deliverables:**
- Basic transformer structure compiles and runs
- Quantized weight loading from GGUF
- Simple end-to-end test (no real attention yet)

### Phase 2: Attention and Generation (Week 3-4)

**Tasks:**
1. Implement `BitNetAttention` with multi-head attention and KV-cache
2. Add `BitNetFeedForward` with SiLU activation and quantized projections
3. Complete transformer blocks with residual connections
4. Autoregressive generation loop with sampling strategies

**Deliverables:**
- Complete transformer forward pass
- Text generation with temperature/top-k sampling
- Deterministic generation with seeding
- Performance meets minimum targets (>5 tok/sec CPU)

### Phase 3: Optimization and Validation (Week 5-6)

**Tasks:**
1. GPU acceleration with fused kernels
2. Cross-validation framework against C++ reference
3. Performance optimization (memory, SIMD, caching)
4. Comprehensive test coverage with AC-tagged tests

**Deliverables:**
- GPU performance targets met (>15 tok/sec)
- Cross-validation accuracy >99.9%
- Full test coverage for all acceptance criteria
- Documentation and examples updated

### Phase 4: Integration and Polish (Week 7-8)

**Tasks:**
1. Replace all mock implementations in xtask, examples, CI
2. Update benchmarking with realistic performance numbers
3. Error handling and edge case coverage
4. Final performance tuning and memory optimization

**Deliverables:**
- No mock implementations remaining in production paths
- Realistic benchmark numbers in CI and documentation
- Production-ready error handling and validation
- Ready for external usage and evaluation

## Risk Mitigation

### Technical Risks

**Risk**: Quantization accuracy degradation
**Mitigation**: Comprehensive cross-validation against C++ reference, target >99.9% correlation

**Risk**: Performance not meeting targets
**Mitigation**: Progressive optimization, GPU fallback, SIMD acceleration, memory optimization

**Risk**: Memory usage exceeding targets
**Mitigation**: KV-cache pre-allocation, zero-copy loading, quantized weights, activation checkpointing

**Risk**: Complex integration across crates
**Mitigation**: Incremental development, extensive unit testing, clear API boundaries

### Project Risks

**Risk**: Scope creep beyond core functionality
**Mitigation**: Focus on AC1-AC10 completion first, defer advanced features

**Risk**: C++ reference integration complexity
**Mitigation**: Start with tensor-level validation, use existing crossval framework

**Risk**: GPU kernel development complexity
**Mitigation**: CPU-first implementation, GPU as performance optimization phase

## Success Criteria

### Acceptance Criteria Coverage

- **AC1** ✅ Real transformer forward pass with quantized weights
- **AC2** ✅ Multi-head attention with quantized Q/K/V projections
- **AC3** ✅ Autoregressive generation with sampling strategies
- **AC4** ✅ >99% quantization accuracy via cross-validation
- **AC5** ✅ Performance targets: 5-15 tok/sec CPU, 15-45 tok/sec GPU
- **AC6** ✅ All quantization formats (I2S, TL1, TL2) with device awareness
- **AC7** ✅ Deterministic inference with seeding support
- **AC8** ✅ All mock implementations replaced with real computation
- **AC9** ✅ Comprehensive testing with unit/integration/performance tests
- **AC10** ✅ Production-grade error handling with detailed context

### Quality Gates

1. **Compilation**: All code compiles with `--no-default-features --features cpu` and `--features gpu`
2. **Testing**: All tests pass with AC-tagged coverage for acceptance criteria
3. **Performance**: Benchmark targets met within 20% tolerance
4. **Accuracy**: Cross-validation correlation >99.9% with C++ reference
5. **Integration**: xtask commands work with real models, no mock placeholders
6. **Documentation**: API docs, examples, and troubleshooting guides complete

This specification transforms BitNet-rs from a quantization infrastructure with mock inference into a production-grade neural network inference engine capable of real-time text generation with state-of-the-art 1-bit quantization.

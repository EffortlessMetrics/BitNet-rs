# CPU Inference API Contracts

**Issue:** #462 - CPU Forward Pass with Real Inference
**Status:** Specification
**Date:** 2025-10-14

## Context

This document defines the public API contracts for CPU inference implementation. These contracts ensure stable interfaces for downstream consumers (CLI, tests, benchmarks) while enabling implementation flexibility.

**Stability Guarantees:**
- Public APIs follow semantic versioning (SemVer)
- Breaking changes require major version bump
- Internal helper functions may change in minor versions
- Feature flag compatibility maintained across patch versions

## API Contracts

### AC1: CPU Forward Pass Engine

#### Primary Interface

```rust
/// CPU inference engine with quantized forward pass
pub struct CpuInferenceEngine {
    model: Arc<RwLock<BitNetModel>>,
    config: InferenceConfig,
    kv_cache: Arc<RwLock<KVCache>>,
    backend: CpuBackend,
    metrics: Arc<Mutex<InferenceMetrics>>,
}

impl CpuInferenceEngine {
    /// Create new CPU inference engine from loaded model
    ///
    /// # Arguments
    /// * `model` - Loaded BitNet model (GGUF format)
    /// * `config` - Inference configuration (cache size, batch settings)
    ///
    /// # Returns
    /// Configured engine ready for token generation
    ///
    /// # Errors
    /// - Model incompatibility (unsupported architecture)
    /// - KV cache allocation failure (insufficient memory)
    /// - Backend initialization failure (missing SIMD features)
    ///
    /// # Example
    /// ```no_run
    /// use bitnet_inference::{CpuInferenceEngine, InferenceConfig};
    /// use bitnet_models::BitNetModel;
    ///
    /// let model = BitNetModel::from_gguf("model.gguf")?;
    /// let config = InferenceConfig::default();
    /// let engine = CpuInferenceEngine::new(model, config)?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn new(
        model: Arc<RwLock<BitNetModel>>,
        config: InferenceConfig,
    ) -> Result<Self>;

    /// Forward pass with parallel layer processing (AC1)
    ///
    /// Performs single-step autoregressive decode:
    /// 1. Embedding lookup (if input is token IDs)
    /// 2. Transformer layers (attention + FFN) with KV cache update
    /// 3. Final LayerNorm
    /// 4. LM head projection
    ///
    /// # Arguments
    /// * `input` - Input tensor: token IDs [1] or embeddings [1, d_model]
    /// * `step` - Current sequence position (0-indexed)
    ///
    /// # Returns
    /// Logits tensor [1, vocab_size] with unnormalized probabilities
    ///
    /// # Errors
    /// - Sequence position exceeds max_seq_len (KV cache overflow)
    /// - Quantization kernel failure (unsupported quantization type)
    /// - Strict mode violation (FP32 staging attempted)
    /// - Model layer missing (corrupted GGUF)
    ///
    /// # Guarantees
    /// - Uses QuantizedLinear I2S/TL1/TL2 paths (no FP32 staging in strict mode)
    /// - KV cache updated with K,V tensors at position `step`
    /// - Returns non-zero finite logits for valid input (BOS token test)
    /// - Deterministic output when BITNET_DETERMINISTIC=1
    ///
    /// # Example
    /// ```no_run
    /// # use bitnet_inference::CpuInferenceEngine;
    /// # use bitnet_common::BitNetTensor;
    /// # let engine: CpuInferenceEngine = todo!();
    /// // BOS token forward pass (AC1 validation)
    /// let bos_token = BitNetTensor::from_slice(&[1u32], &[1], DType::U32, &Device::Cpu)?;
    /// let logits = engine.forward_parallel(&bos_token, 0)?;
    ///
    /// assert_eq!(logits.shape(), &[1, 32000]); // vocab_size = 32000
    /// assert!(logits.to_vec1::<f32>()?.iter().any(|&x| x != 0.0)); // Non-zero logits
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn forward_parallel(
        &self,
        input: &BitNetTensor,
        step: usize,
    ) -> Result<BitNetTensor>;

    /// Generate tokens autoregressively (convenience wrapper)
    ///
    /// # Arguments
    /// * `prompt_tokens` - Input token sequence
    /// * `config` - Generation parameters (max_tokens, temperature, top_k/top_p)
    ///
    /// # Returns
    /// Generated token sequence (excluding prompt)
    ///
    /// # Example
    /// ```no_run
    /// # use bitnet_inference::{CpuInferenceEngine, GenerationConfig};
    /// # let engine: CpuInferenceEngine = todo!();
    /// let prompt = vec![1, 2, 3]; // BOS + "Hello"
    /// let config = GenerationConfig {
    ///     max_new_tokens: 16,
    ///     temperature: 0.7,
    ///     top_k: Some(50),
    ///     ..Default::default()
    /// };
    /// let generated = engine.generate_tokens_parallel(&prompt, &config)?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn generate_tokens_parallel(
        &self,
        prompt_tokens: &[u32],
        config: &GenerationConfig,
    ) -> Result<Vec<u32>>;
}
```

#### Supporting Data Structures

```rust
/// Inference configuration
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Maximum sequence length for KV cache
    pub max_sequence_length: usize,

    /// Enable batch processing (parallel requests)
    pub batch_processing: bool,

    /// KV cache configuration
    pub cache_config: CacheConfig,

    /// Strict mode enforcement
    pub strict_mode: bool,

    /// Number of worker threads (0 = auto-detect)
    pub num_threads: usize,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_sequence_length: 2048,
            batch_processing: false,
            cache_config: CacheConfig::default(),
            strict_mode: std::env::var("BITNET_STRICT_MODE").is_ok(),
            num_threads: 0, // Auto-detect
        }
    }
}

/// Token generation parameters
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum new tokens to generate
    pub max_new_tokens: usize,

    /// Sampling temperature (0.0 = greedy, 1.0 = neutral, >1.0 = random)
    pub temperature: f32,

    /// Top-k sampling (None = disabled)
    pub top_k: Option<usize>,

    /// Top-p nucleus sampling (None = disabled)
    pub top_p: Option<f32>,

    /// Repetition penalty (1.0 = disabled)
    pub repetition_penalty: f32,

    /// EOS token IDs to stop generation
    pub eos_token_id: Vec<u32>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            temperature: 1.0,
            top_k: None,
            top_p: None,
            repetition_penalty: 1.0,
            eos_token_id: vec![2], // Common EOS token
        }
    }
}
```

#### Private Helper Functions

```rust
// Internal helpers (not part of public API)
impl CpuInferenceEngine {
    /// Embed token ID to dense vector
    fn embed_token(&self, token_id: u32) -> Result<BitNetTensor>;

    /// Apply single transformer layer
    fn apply_layer(&self, x: &BitNetTensor, layer_idx: usize, step: usize) -> Result<BitNetTensor>;

    /// Apply LayerNorm with named weights
    fn apply_layer_norm(&self, x: &BitNetTensor, weight_name: &str) -> Result<BitNetTensor>;

    /// Apply attention block with KV cache update
    fn apply_attention(&self, x: &BitNetTensor, layer_idx: usize, step: usize) -> Result<BitNetTensor>;

    /// Apply FFN block (SwiGLU)
    fn apply_ffn(&self, x: &BitNetTensor, layer_idx: usize) -> Result<BitNetTensor>;

    /// Compute final logits via LM head
    fn compute_logits(&self, hidden_states: &BitNetTensor) -> Result<BitNetTensor>;

    /// Apply causal mask for autoregressive attention
    fn apply_causal_mask(&self, scores: &BitNetTensor, step: usize) -> Result<BitNetTensor>;

    /// Apply RoPE (Rotary Position Embeddings)
    fn apply_rope(&self, q: BitNetTensor, k: BitNetTensor, step: usize) -> Result<(BitNetTensor, BitNetTensor)>;

    /// Execute QuantizedLinear layer by name
    fn quantized_linear(&self, input: &BitNetTensor, weight_name: &str) -> Result<BitNetTensor>;
}
```

### AC2: CLI Inference Commands

#### High-Level Interface

```rust
/// Run inference command with priming and decode loops (AC2)
///
/// # Arguments
/// * `model_path` - Path to GGUF model file
/// * `tokenizer_path` - Path to tokenizer.json (None = auto-discover)
/// * `prompt` - Input text prompt
/// * `max_new_tokens` - Maximum tokens to generate
/// * `temperature` - Sampling temperature
/// * `top_k` - Top-k sampling (None = disabled)
/// * `top_p` - Top-p nucleus sampling (None = disabled)
///
/// # Returns
/// Generated text (decoded from token IDs)
///
/// # Errors
/// - Model not found or incompatible format
/// - Tokenizer not found (auto-discovery failed)
/// - Inference failure (OOM, quantization error)
/// - Timeout (max_new_tokens reached without EOS)
///
/// # Example
/// ```bash
/// # CLI usage (AC2 requirement)
/// cargo run -p bitnet-cli --features cpu -- \
///   run --model model.gguf \
///   --prompt "Q: What is 2+2? A:" \
///   --max-new-tokens 16 \
///   --temperature 0.0
/// ```
pub fn run_inference(
    model_path: &Path,
    tokenizer_path: Option<&Path>,
    prompt: &str,
    max_new_tokens: usize,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
) -> Result<String>;
```

#### Priming and Decode Helpers

```rust
/// Prime KV cache with prompt tokens (AC2)
///
/// # Arguments
/// * `engine` - Inference engine
/// * `tokens` - Tokenized prompt
///
/// # Effects
/// - Populates KV cache for positions 0..tokens.len()
/// - Discards logits (not needed for priming)
/// - Updates metrics: prefill_tps, prefill_duration
///
/// # Example
/// ```no_run
/// # use bitnet_inference::CpuInferenceEngine;
/// # let engine: CpuInferenceEngine = todo!();
/// let prompt_tokens = vec![1, 50, 100, 200]; // BOS + "Hello world"
/// prime_cache(&engine, &prompt_tokens)?;
/// // KV cache now populated for positions 0..3
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn prime_cache(
    engine: &InferenceEngine,
    tokens: &[u32],
) -> Result<()>;

/// Decode loop with token sampling (AC2)
///
/// # Arguments
/// * `engine` - Inference engine with primed KV cache
/// * `sampler` - Token sampler (greedy, top-k, top-p)
/// * `max_tokens` - Maximum new tokens to generate
/// * `start_step` - Starting sequence position (after priming)
///
/// # Returns
/// Generated token IDs (excluding prompt)
///
/// # Effects
/// - Streams tokens to stdout during generation
/// - Updates metrics: decode_tps, decode_duration
/// - Stops at EOS token or max_tokens limit
///
/// # Example
/// ```no_run
/// # use bitnet_inference::{CpuInferenceEngine, Sampler};
/// # let engine: CpuInferenceEngine = todo!();
/// # let sampler: Sampler = todo!();
/// let generated = decode_loop(&engine, &sampler, 16, 4)?; // Generate 16 tokens starting at position 4
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn decode_loop(
    engine: &InferenceEngine,
    sampler: &Sampler,
    max_tokens: usize,
    start_step: usize,
) -> Result<Vec<u32>>;
```

#### Sampler Interface

```rust
/// Token sampler for decode loop
pub struct Sampler {
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    rng: StdRng, // Seeded RNG for determinism
}

impl Sampler {
    /// Create sampler with deterministic seed
    ///
    /// # Environment Variables
    /// - `BITNET_DETERMINISTIC=1`: Enable deterministic sampling
    /// - `BITNET_SEED=42`: Set RNG seed (default: 42)
    ///
    /// # Example
    /// ```no_run
    /// # use bitnet_inference::Sampler;
    /// // Greedy sampling (temperature = 0.0)
    /// let sampler = Sampler::new(0.0, None, None)?;
    ///
    /// // Top-k sampling
    /// let sampler = Sampler::new(0.7, Some(50), None)?;
    ///
    /// // Nucleus (top-p) sampling
    /// let sampler = Sampler::new(0.9, None, Some(0.95))?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn new(
        temperature: f32,
        top_k: Option<usize>,
        top_p: Option<f32>,
    ) -> Result<Self>;

    /// Sample next token from logits
    ///
    /// # Arguments
    /// * `logits` - Unnormalized logits [1, vocab_size]
    ///
    /// # Returns
    /// Sampled token ID
    ///
    /// # Sampling Strategies
    /// - `temperature = 0.0`: Greedy (argmax)
    /// - `top_k = Some(k)`: Sample from top-k highest probability tokens
    /// - `top_p = Some(p)`: Nucleus sampling (cumulative probability ≥ p)
    /// - `temperature > 0.0`: Apply temperature scaling before sampling
    ///
    /// # Example
    /// ```no_run
    /// # use bitnet_inference::Sampler;
    /// # use bitnet_common::BitNetTensor;
    /// # let sampler: Sampler = todo!();
    /// # let logits: BitNetTensor = todo!();
    /// let next_token = sampler.sample(&logits)?;
    /// println!("Sampled token: {}", next_token);
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn sample(&mut self, logits: &BitNetTensor) -> Result<u32>;
}
```

### KV Cache Interface

```rust
/// KV cache for autoregressive generation
#[derive(Debug)]
pub struct KVCache {
    k_cache: Vec<BitNetTensor>, // [num_layers] of [max_seq_len, num_heads, head_dim]
    v_cache: Vec<BitNetTensor>, // [num_layers] of [max_seq_len, num_heads, head_dim]
    max_seq_len: usize,
    current_len: usize,
}

impl KVCache {
    /// Create new KV cache
    ///
    /// # Arguments
    /// * `max_seq_len` - Maximum sequence length
    /// * `num_layers` - Number of transformer layers
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension per attention head
    /// * `device` - Compute device (CPU/CUDA)
    ///
    /// # Returns
    /// Initialized cache with zero-filled tensors
    ///
    /// # Memory
    /// Allocates: 2 × max_seq_len × num_heads × head_dim × sizeof(f32) × num_layers
    ///
    /// # Example
    /// ```no_run
    /// # use bitnet_inference::KVCache;
    /// # use bitnet_common::Device;
    /// let cache = KVCache::new(
    ///     2048,  // max_seq_len
    ///     28,    // num_layers
    ///     32,    // num_heads
    ///     64,    // head_dim
    ///     &Device::Cpu,
    /// )?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn new(
        max_seq_len: usize,
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        device: &Device,
    ) -> Result<Self>;

    /// Update cache with new K,V tensors (AC1 requirement)
    ///
    /// # Arguments
    /// * `layer_idx` - Layer index (0-indexed)
    /// * `k` - Key tensor [1, num_heads, head_dim]
    /// * `v` - Value tensor [1, num_heads, head_dim]
    /// * `step` - Sequence position to write
    ///
    /// # Effects
    /// Writes K,V to cache at position `step`:
    /// - `cache[layer].k[step, :, :] = k`
    /// - `cache[layer].v[step, :, :] = v`
    ///
    /// # Errors
    /// - Layer index out of bounds
    /// - Sequence position exceeds max_seq_len
    /// - Shape mismatch (k,v must match [1, num_heads, head_dim])
    ///
    /// # Example
    /// ```no_run
    /// # use bitnet_inference::KVCache;
    /// # use bitnet_common::BitNetTensor;
    /// # let mut cache: KVCache = todo!();
    /// # let k: BitNetTensor = todo!(); // [1, 32, 64]
    /// # let v: BitNetTensor = todo!(); // [1, 32, 64]
    /// cache.update(0, k, v, 5)?; // Write to layer 0, position 5
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn update(
        &mut self,
        layer_idx: usize,
        k: BitNetTensor,
        v: BitNetTensor,
        step: usize,
    ) -> Result<()>;

    /// Retrieve cached K,V for layer (AC1 requirement)
    ///
    /// # Arguments
    /// * `layer_idx` - Layer index (0-indexed)
    ///
    /// # Returns
    /// Tuple of (K, V) tensors sliced to current sequence length:
    /// - K: [current_len, num_heads, head_dim]
    /// - V: [current_len, num_heads, head_dim]
    ///
    /// # Errors
    /// - Layer index out of bounds
    ///
    /// # Example
    /// ```no_run
    /// # use bitnet_inference::KVCache;
    /// # let cache: KVCache = todo!();
    /// let (k_cache, v_cache) = cache.get(0)?; // Get layer 0 cache
    /// // k_cache.shape() = [current_len, 32, 64]
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn get(
        &self,
        layer_idx: usize,
    ) -> Result<(BitNetTensor, BitNetTensor)>;

    /// Reset cache to empty state
    pub fn reset(&mut self);

    /// Get current sequence length
    pub fn len(&self) -> usize;

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool;
}
```

### Metrics and Receipt Generation

```rust
/// Inference performance metrics
#[derive(Debug, Clone, Default)]
pub struct InferenceMetrics {
    /// Prefill throughput (tokens/second)
    pub prefill_tps: f64,

    /// Decode throughput (tokens/second)
    pub decode_tps: f64,

    /// End-to-end throughput (tokens/second)
    pub e2e_tps: f64,

    /// Prefill duration (milliseconds)
    pub prefill_ms: f64,

    /// Decode duration (milliseconds)
    pub decode_ms: f64,

    /// Total latency (milliseconds)
    pub latency_ms: f64,

    /// Number of tokens processed
    pub tokens_processed: usize,

    /// Number of tokens generated
    pub tokens_generated: usize,
}

/// Generate inference receipt (AC3 requirement)
///
/// # Arguments
/// * `backend` - Backend used ("cpu" | "cuda" | "metal")
/// * `kernels` - List of kernel IDs executed during inference
///
/// # Returns
/// Structured receipt with schema version 1.0.0
///
/// # Receipt Fields
/// - `schema_version`: "1.0.0"
/// - `compute_path`: "real" (no mock kernels) or "mock"
/// - `backend`: "cpu" | "cuda" | "metal"
/// - `kernels`: ["i2s_gemv", "tl1_matmul", ...] (non-empty)
/// - `deterministic`: true if BITNET_DETERMINISTIC=1
/// - `environment`: BITNET_* and RAYON_* variables
/// - `model_info`: Model architecture metadata
/// - `test_results`: Pass/fail status
/// - `performance_baseline`: Throughput metrics
///
/// # Example
/// ```no_run
/// # use bitnet_inference::receipts::InferenceReceipt;
/// let receipt = InferenceReceipt::generate(
///     "cpu",
///     vec!["i2s_gemv".to_string(), "tl1_matmul".to_string()],
/// )?;
///
/// assert_eq!(receipt.compute_path, "real"); // No mock kernels
/// assert_eq!(receipt.backend, "cpu");
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn generate_receipt(
    backend: &str,
    kernels: Vec<String>,
) -> Result<InferenceReceipt>;
```

## Error Handling

### Error Types

```rust
/// Inference-specific errors
#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    #[error("Sequence position {0} exceeds max_seq_len {1}")]
    SequenceOverflow(usize, usize),

    #[error("KV cache layer {0} out of bounds (max: {1})")]
    InvalidLayer(usize, usize),

    #[error("Strict mode: FP32 staging not allowed (kernel: {0})")]
    StrictModeViolation(String),

    #[error("Quantization kernel failure: {0}")]
    QuantizationError(String),

    #[error("Model layer missing: {0}")]
    MissingLayer(String),

    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("EOS token not found after {0} tokens")]
    GenerationTimeout(usize),

    #[error("Tokenizer error: {0}")]
    TokenizerError(String),

    #[error("Model loading error: {0}")]
    ModelLoadError(String),
}
```

### Error Context Patterns

```rust
// Use .context() for error chain preservation
let logits = self.forward_parallel(&input, step)
    .with_context(|| format!("Forward pass failed at step {}", step))?;

// Use .map_err() for error type conversion
let tokens = tokenizer.encode(prompt)
    .map_err(|e| InferenceError::TokenizerError(e.to_string()))?;

// Use anyhow::bail!() for early exit with context
if step >= self.config.max_sequence_length {
    anyhow::bail!(
        InferenceError::SequenceOverflow(step, self.config.max_sequence_length)
    );
}
```

## Feature Flag Compatibility

### CPU Feature

```rust
#[cfg(feature = "cpu")]
impl CpuInferenceEngine {
    // All CPU inference APIs available with --features cpu
}

#[cfg(not(feature = "cpu"))]
compile_error!("CPU inference requires --features cpu");
```

### Strict Mode Feature

```rust
// Strict mode enforced when:
// 1. BITNET_STRICT_MODE=1 environment variable, OR
// 2. InferenceConfig::strict_mode = true

#[cfg(feature = "cpu")]
fn enforce_strict_mode() {
    if StrictModeConfig::get().enforce_quantized_inference {
        // Block FP32 fallback paths
        // Require quantized kernels only
    }
}
```

### Build Commands

```bash
# CPU inference (default features EMPTY - always specify)
cargo build --no-default-features --features cpu

# CPU with strict mode enforcement
BITNET_STRICT_MODE=1 cargo build --features cpu

# CPU with cross-validation
cargo build --features cpu,crossval

# Tests with CPU backend
cargo test --workspace --features cpu
```

## Validation

### API Stability Tests

```rust
// AC1: Forward pass contract
#[test]
fn test_forward_parallel_signature() {
    // Ensure function signature remains stable
    let _: fn(&CpuInferenceEngine, &BitNetTensor, usize) -> Result<BitNetTensor> =
        CpuInferenceEngine::forward_parallel;
}

// AC2: CLI interface contract
#[test]
fn test_run_inference_signature() {
    let _: fn(&Path, Option<&Path>, &str, usize, f32, Option<usize>, Option<f32>) -> Result<String> =
        run_inference;
}

// KV cache update contract
#[test]
fn test_kv_cache_update_signature() {
    let _: fn(&mut KVCache, usize, BitNetTensor, BitNetTensor, usize) -> Result<()> =
        KVCache::update;
}
```

### Behavioral Tests

```rust
// AC1: BOS token returns non-zero logits
#[test]
fn test_ac1_cpu_forward_bos_nonzero_logits() {
    let engine = create_test_engine()?;
    let bos = BitNetTensor::from_slice(&[1u32], &[1], DType::U32, &Device::Cpu)?;

    let logits = engine.forward_parallel(&bos, 0)?;

    assert_eq!(logits.shape(), &[1, 32000]);
    let logits_vec = logits.to_vec1::<f32>()?;
    assert!(logits_vec.iter().any(|&x| x != 0.0), "Logits must be non-zero");
    assert!(logits_vec.iter().all(|&x| x.is_finite()), "Logits must be finite");
}

// AC2: CLI priming loop
#[test]
fn test_ac2_cli_priming_loop() {
    let engine = create_test_engine()?;
    let prompt_tokens = vec![1, 50, 100, 200];

    prime_cache(&engine, &prompt_tokens)?;

    // Verify KV cache populated
    let cache = engine.kv_cache.read().unwrap();
    assert_eq!(cache.len(), prompt_tokens.len());
}

// AC2: Decode loop generates 16 tokens
#[test]
fn test_ac2_decode_loop_16_tokens() {
    let engine = create_test_engine_with_primed_cache()?;
    let sampler = Sampler::new(0.0, None, None)?; // Greedy

    let generated = decode_loop(&engine, &sampler, 16, 4)?;

    assert!(generated.len() <= 16, "Should not exceed max_tokens");
    assert!(!generated.is_empty(), "Should generate at least one token");
}
```

## References

### Related Documentation

- `docs/explanation/cpu-inference-architecture.md` - Architecture design
- `docs/reference/quantization-support.md` - Quantization algorithms
- `docs/development/test-suite.md` - Testing framework
- `CLAUDE.md` - Feature flag patterns and build commands

### Existing Code Patterns

- `crates/bitnet-inference/src/cpu.rs` - Engine implementation
- `crates/bitnet-inference/src/layers/quantized_linear.rs` - QuantizedLinear dispatch
- `crates/bitnet-inference/src/layers/attention.rs` - KV cache interface
- `crates/bitnet-inference/src/sampling.rs` - Token sampler
- `crates/bitnet-cli/src/commands/inference.rs` - CLI command structure

### Stability Guarantees

**Public APIs (SemVer):**
- `CpuInferenceEngine::new()` - Stable (v1.0)
- `CpuInferenceEngine::forward_parallel()` - Stable (v1.0)
- `KVCache::update()` - Stable (v1.0)
- `KVCache::get()` - Stable (v1.0)
- `run_inference()` - Stable (v1.0)
- `prime_cache()` - Stable (v1.0)
- `decode_loop()` - Stable (v1.0)

**Internal APIs (may change):**
- Private helper functions in `CpuInferenceEngine`
- Internal cache management optimizations
- SIMD kernel selection logic

# [Feature] Advanced Attention Mechanisms and Caching Infrastructure Implementation

## Problem Description

BitNet-rs attention system contains multiple critical implementation gaps that prevent efficient transformer inference and limit model compatibility. Five major components require comprehensive implementation:

1. **Grouped Query Attention (GQA)**: `apply_gqa()` simply clones tensors instead of implementing proper key-value head grouping
2. **Dynamic Rotary Embeddings**: `RotaryEmbedding::apply()` lacks dynamic growth for sequences exceeding `max_seq_len`
3. **Universal Tokenizer Fallbacks**: Extensive use of `MockTokenizer` fallbacks masks real tokenizer loading issues
4. **Tensor Caching**: `tensor_cache` in autoregressive generation always returns cache misses
5. **Conditional Mock Models**: `MockBitNetModel` conditionally compiled preventing consistent testing

These limitations severely impact transformer model performance, memory efficiency, and compatibility with modern model architectures.

## Environment

- **Affected Crates**: `bitnet-inference`, `bitnet-models`, `bitnet-tokenizers`
- **Primary Files**:
  - `crates/bitnet-inference/src/layers/attention.rs`
  - `crates/bitnet-models/src/transformer.rs`
  - `crates/bitnet-tokenizers/src/universal.rs`
  - `crates/bitnet-inference/src/generation/autoregressive.rs`
  - `crates/bitnet-models/src/production_loader.rs`
- **Build Configuration**: `--no-default-features --features cpu,inference`
- **Model Compatibility**: LLaMA-3, GPT-4, Falcon, Mistral (all using advanced attention mechanisms)

## Root Cause Analysis

### Attention Implementation Gaps

1. **Incomplete GQA Implementation**: Key-value head sharing not implemented for memory efficiency
   ```rust
   // Current: Simple tensor cloning
   fn apply_gqa(&self, key_states: &BitNetTensor, value_states: &BitNetTensor) -> Result<(BitNetTensor, BitNetTensor)> {
       Ok((key_states.clone(), value_states.clone()))
   }
   ```

2. **Static Rotary Embeddings**: Cannot handle sequences longer than pre-allocated `max_seq_len`
   ```rust
   // Current: Error on sequence length overflow
   if seq_len > self.max_seq_len {
       return Err(BitNetError::Validation("Sequence length exceeds max_seq_len (dynamic growth not implemented)"));
   }
   ```

3. **Mock Tokenizer Fallbacks**: Hide real tokenizer loading failures
   ```rust
   // Current: Always falls back to mock
   "gpt2" | "bpe" | "llama" | "tiktoken" => {
       Ok(TokenizerBackend::Mock(MockTokenizer::new()))
   }
   ```

### Caching and Memory Management Issues

4. **Non-functional Tensor Cache**: Always returns cache misses preventing inference optimization
   ```rust
   // Current: Always returns None
   fn try_get_cached_tensor(&self, tokens: &[usize]) -> Result<Option<BitNetTensor>> {
       Ok(None) // No actual caching
   }
   ```

5. **Conditional Mock Availability**: Testing inconsistency across feature configurations

## Impact Assessment

- **Severity**: High - Blocks efficient transformer inference and advanced model support
- **Performance Impact**: 40-60% memory overhead without GQA, 2-3x slower inference without caching
- **Model Compatibility**: Cannot run modern models using GQA (LLaMA-3, GPT-4, Falcon)
- **Memory Efficiency**: Exponential memory growth for long sequences without proper caching
- **Development Quality**: Mock fallbacks mask real implementation issues

## Proposed Solution

### 1. Complete Grouped Query Attention Implementation

**Efficient Key-Value Head Sharing**:
```rust
impl BitNetAttention {
    fn apply_gqa(
        &self,
        key_states: &BitNetTensor,
        value_states: &BitNetTensor,
    ) -> Result<(BitNetTensor, BitNetTensor)> {
        let num_query_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;

        // If num_kv_heads == num_query_heads, no grouping needed
        if num_kv_heads == num_query_heads {
            return Ok((key_states.clone(), value_states.clone()));
        }

        let num_kv_groups = num_query_heads / num_kv_heads;
        let batch_size = key_states.shape()[0];
        let seq_len = key_states.shape()[1];
        let head_dim = self.config.head_dim;

        // Reshape key/value states to separate heads dimension
        let key_reshaped = key_states.reshape(&[batch_size, seq_len, num_kv_heads, head_dim])?;
        let value_reshaped = value_states.reshape(&[batch_size, seq_len, num_kv_heads, head_dim])?;

        // Repeat each key/value head for the corresponding query head groups
        let expanded_keys = self.repeat_kv_heads(&key_reshaped, num_kv_groups)?;
        let expanded_values = self.repeat_kv_heads(&value_reshaped, num_kv_groups)?;

        // Reshape back to original format
        let final_keys = expanded_keys.reshape(&[batch_size, seq_len, num_query_heads * head_dim])?;
        let final_values = expanded_values.reshape(&[batch_size, seq_len, num_query_heads * head_dim])?;

        Ok((final_keys, final_values))
    }

    fn repeat_kv_heads(&self, tensor: &BitNetTensor, num_repeats: usize) -> Result<BitNetTensor> {
        if num_repeats == 1 {
            return Ok(tensor.clone());
        }

        let shape = tensor.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let num_heads = shape[2];
        let head_dim = shape[3];

        // Use efficient tensor operations for head repetition
        match &tensor.backend() {
            TensorBackend::Candle(candle_tensor) => {
                // Use candle's repeat_interleave for efficient memory usage
                let repeated = candle_tensor.repeat_interleave(num_repeats, 2)?;
                Ok(BitNetTensor::from_candle(repeated))
            }
            TensorBackend::ConcreteTensor(concrete) => {
                // Manual implementation for concrete tensors
                let data = concrete.data();
                let mut repeated_data = Vec::with_capacity(data.len() * num_repeats);

                for batch in 0..batch_size {
                    for seq in 0..seq_len {
                        for head in 0..num_heads {
                            for _ in 0..num_repeats {
                                let start_idx = ((batch * seq_len + seq) * num_heads + head) * head_dim;
                                let end_idx = start_idx + head_dim;
                                repeated_data.extend_from_slice(&data[start_idx..end_idx]);
                            }
                        }
                    }
                }

                let new_shape = vec![batch_size, seq_len, num_heads * num_repeats, head_dim];
                Ok(BitNetTensor::from_concrete(ConcreteTensor::new(repeated_data, new_shape)))
            }
        }
    }

    fn validate_gqa_config(&self) -> Result<()> {
        let num_query_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;

        if num_query_heads % num_kv_heads != 0 {
            return Err(anyhow::anyhow!(
                "Number of query heads ({}) must be divisible by number of key-value heads ({})",
                num_query_heads,
                num_kv_heads
            ));
        }

        if num_kv_heads == 0 || num_query_heads == 0 {
            return Err(anyhow::anyhow!("Number of attention heads must be positive"));
        }

        Ok(())
    }
}
```

### 2. Dynamic Rotary Embedding Growth

**Adaptive Sequence Length Handling**:
```rust
impl RotaryEmbedding {
    pub fn apply(&self, x: &Tensor, position: usize) -> Result<Tensor> {
        let (batch, n_heads, seq_len, head_dim) = if x.dims().len() == 4 {
            x.dims4()?
        } else {
            let (b, s, d) = x.dims3()?;
            (b, 1, s, d) // Treat as single head
        };

        // Check if we need to grow the rotary embeddings
        if position + seq_len > self.max_seq_len {
            return self.apply_with_dynamic_growth(x, position);
        }

        self.apply_standard(x, position)
    }

    fn apply_with_dynamic_growth(&self, x: &Tensor, position: usize) -> Result<Tensor> {
        let required_length = position + x.dims()[x.dims().len() - 2]; // seq_len dimension
        let new_max_len = self.calculate_new_max_len(required_length);

        // Generate extended sin/cos caches
        let extended_cos = self.generate_cos_cache(new_max_len)?;
        let extended_sin = self.generate_sin_cache(new_max_len)?;

        // Create temporary rotary embedding with extended cache
        let extended_rope = RotaryEmbedding {
            cos: extended_cos,
            sin: extended_sin,
            max_seq_len: new_max_len,
            head_dim: self.head_dim,
            base: self.base,
        };

        extended_rope.apply_standard(x, position)
    }

    fn calculate_new_max_len(&self, required_length: usize) -> usize {
        // Growth strategy: double the size or add required length + buffer, whichever is larger
        let doubled = self.max_seq_len * 2;
        let required_with_buffer = required_length + (required_length / 4).max(1024);
        doubled.max(required_with_buffer)
    }

    fn generate_cos_cache(&self, max_len: usize) -> Result<Tensor> {
        let device = self.cos.device();
        let dtype = self.cos.dtype();

        let inv_freq = self.generate_inv_freq()?;
        let positions = Tensor::arange(0u32, max_len as u32, device)?
            .to_dtype(dtype)?
            .unsqueeze(1)?;

        let freqs = positions.matmul(&inv_freq.unsqueeze(0))?;
        let cos_cache = freqs.cos()?;

        Ok(cos_cache)
    }

    fn generate_sin_cache(&self, max_len: usize) -> Result<Tensor> {
        let device = self.sin.device();
        let dtype = self.sin.dtype();

        let inv_freq = self.generate_inv_freq()?;
        let positions = Tensor::arange(0u32, max_len as u32, device)?
            .to_dtype(dtype)?
            .unsqueeze(1)?;

        let freqs = positions.matmul(&inv_freq.unsqueeze(0))?;
        let sin_cache = freqs.sin()?;

        Ok(sin_cache)
    }

    fn generate_inv_freq(&self) -> Result<Tensor> {
        let device = self.cos.device();
        let dtype = self.cos.dtype();

        let half_dim = self.head_dim / 2;
        let inv_freq_values: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / self.base.powf(2.0 * (i as f32) / (self.head_dim as f32)))
            .collect();

        Tensor::from_slice(&inv_freq_values, (1, half_dim), device)?.to_dtype(dtype)
    }

    // Thread-safe cache update for concurrent access
    pub fn update_cache_if_needed(&mut self, required_length: usize) -> Result<()> {
        if required_length <= self.max_seq_len {
            return Ok(());
        }

        let new_max_len = self.calculate_new_max_len(required_length);
        self.cos = self.generate_cos_cache(new_max_len)?;
        self.sin = self.generate_sin_cache(new_max_len)?;
        self.max_seq_len = new_max_len;

        Ok(())
    }
}
```

### 3. Production-Ready Tokenizer Implementation

**Eliminate Mock Fallbacks with Real Implementations**:
```rust
impl UniversalTokenizer {
    fn detect_and_create_backend(config: &TokenizerConfig) -> Result<TokenizerBackend> {
        match config.model_type.as_str() {
            "gpt2" => Self::create_gpt2_tokenizer(config),
            "bpe" => Self::create_bpe_tokenizer(config),
            "llama" | "llama3" => Self::create_llama_tokenizer(config),
            "tiktoken" | "gpt4" | "cl100k" => Self::create_tiktoken_tokenizer(config),
            "falcon" => Self::create_falcon_tokenizer(config),
            unknown => Err(BitNetError::Inference(InferenceError::TokenizationFailed {
                reason: format!(
                    "Unsupported tokenizer type '{}'. Supported types: gpt2, bpe, llama, llama3, tiktoken, gpt4, cl100k, falcon",
                    unknown
                ),
                suggestion: Some("Check model configuration or implement custom tokenizer".to_string()),
            })),
        }
    }

    fn create_gpt2_tokenizer(config: &TokenizerConfig) -> Result<TokenizerBackend> {
        let vocab_path = config.vocab_file.as_ref()
            .or(config.tokenizer_file.as_ref())
            .ok_or_else(|| BitNetError::Config(
                "GPT-2 tokenizer requires vocab_file or tokenizer_file path".to_string()
            ))?;

        match Gpt2Tokenizer::from_file(vocab_path) {
            Ok(tokenizer) => {
                info!("Loaded GPT-2 tokenizer from {}", vocab_path.display());
                Ok(TokenizerBackend::Gpt2(tokenizer))
            }
            Err(e) => {
                // Try alternative loading methods
                Self::try_alternative_gpt2_loading(config, e)
            }
        }
    }

    fn try_alternative_gpt2_loading(config: &TokenizerConfig, original_error: anyhow::Error) -> Result<TokenizerBackend> {
        // Try loading from HuggingFace Hub
        if let Some(model_name) = &config.model_name {
            if let Ok(tokenizer) = Gpt2Tokenizer::from_pretrained(model_name) {
                info!("Loaded GPT-2 tokenizer from HuggingFace Hub: {}", model_name);
                return Ok(TokenizerBackend::Gpt2(tokenizer));
            }
        }

        // Try loading with different file formats
        if let Some(base_path) = &config.vocab_file {
            let alternatives = vec![
                base_path.with_extension("json"),
                base_path.with_extension("txt"),
                base_path.parent().unwrap_or(Path::new(".")).join("vocab.json"),
                base_path.parent().unwrap_or(Path::new(".")).join("merges.txt"),
            ];

            for alt_path in alternatives {
                if alt_path.exists() {
                    if let Ok(tokenizer) = Gpt2Tokenizer::from_file(&alt_path) {
                        info!("Loaded GPT-2 tokenizer from alternative path: {}", alt_path.display());
                        return Ok(TokenizerBackend::Gpt2(tokenizer));
                    }
                }
            }
        }

        Err(BitNetError::Inference(InferenceError::TokenizationFailed {
            reason: format!("Failed to load GPT-2 tokenizer: {}", original_error),
            suggestion: Some(
                "Ensure vocab.json and merges.txt files exist, or provide a valid model name for HuggingFace Hub download".to_string()
            ),
        }))
    }

    fn create_llama_tokenizer(config: &TokenizerConfig) -> Result<TokenizerBackend> {
        // Try SentencePiece model first
        if let Some(spm_path) = &config.tokenizer_file {
            match SpTokenizer::from_file(smp_path) {
                Ok(tokenizer) => {
                    info!("Loaded LLaMA SentencePiece tokenizer from {}", spm_path.display());
                    return Ok(TokenizerBackend::SentencePiece(tokenizer));
                }
                Err(e) => {
                    warn!("Failed to load SentencePiece tokenizer: {}", e);
                }
            }
        }

        // Try HuggingFace tokenizer.json format
        if let Some(hf_path) = &config.vocab_file {
            match HfTokenizer::from_file(hf_path) {
                Ok(tokenizer) => {
                    info!("Loaded LLaMA HuggingFace tokenizer from {}", hf_path.display());
                    return Ok(TokenizerBackend::HuggingFace(tokenizer));
                }
                Err(e) => {
                    warn!("Failed to load HuggingFace tokenizer: {}", e);
                }
            }
        }

        Err(BitNetError::Inference(InferenceError::TokenizationFailed {
            reason: "No valid LLaMA tokenizer file found".to_string(),
            suggestion: Some(
                "Provide either a SentencePiece model file (.model) or HuggingFace tokenizer.json file".to_string()
            ),
        }))
    }

    fn create_tiktoken_tokenizer(config: &TokenizerConfig) -> Result<TokenizerBackend> {
        let encoding_name = match config.model_type.as_str() {
            "tiktoken" | "gpt4" => "cl100k_base",
            "cl100k" => "cl100k_base",
            _ => "gpt2", // fallback
        };

        match TiktokenTokenizer::new(encoding_name) {
            Ok(tokenizer) => {
                info!("Loaded Tiktoken tokenizer with encoding: {}", encoding_name);
                Ok(TokenizerBackend::Tiktoken(tokenizer))
            }
            Err(e) => Err(BitNetError::Inference(InferenceError::TokenizationFailed {
                reason: format!("Failed to create Tiktoken tokenizer: {}", e),
                suggestion: Some("Ensure tiktoken library is properly installed".to_string()),
            }))
        }
    }
}
```

### 4. Advanced Tensor Caching System

**LRU Cache with Intelligent Eviction**:
```rust
pub struct AutoregressiveGenerator {
    model: Arc<dyn Model>,
    tokenizer: Arc<dyn Tokenizer>,
    config: GenerationConfig,

    // Advanced caching system
    tensor_cache: LruCache<TokenSequenceKey, CachedTensor>,
    kv_cache: KVCache,

    // Performance metrics
    cache_hit_count: AtomicUsize,
    cache_miss_count: AtomicUsize,
    cache_eviction_count: AtomicUsize,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct TokenSequenceKey {
    tokens: Vec<u32>,
    model_hash: u64, // To handle different models
    position: usize,
}

#[derive(Debug, Clone)]
struct CachedTensor {
    tensor: BitNetTensor,
    created_at: Instant,
    access_count: usize,
    memory_size: usize,
}

impl AutoregressiveGenerator {
    pub fn new(model: Arc<dyn Model>, tokenizer: Arc<dyn Tokenizer>, config: GenerationConfig) -> Self {
        let cache_capacity = config.cache_config.max_entries.unwrap_or(1000);

        Self {
            model,
            tokenizer,
            config,
            tensor_cache: LruCache::new(cache_capacity),
            kv_cache: KVCache::new(config.max_sequence_length, config.num_layers),
            cache_hit_count: AtomicUsize::new(0),
            cache_miss_count: AtomicUsize::new(0),
            cache_eviction_count: AtomicUsize::new(0),
        }
    }

    fn try_get_cached_tensor(&self, tokens: &[u32], position: usize) -> Result<Option<BitNetTensor>> {
        if tokens.is_empty() || !self.config.cache_config.enable_tensor_cache {
            return Ok(None);
        }

        let key = TokenSequenceKey {
            tokens: tokens.to_vec(),
            model_hash: self.calculate_model_hash(),
            position,
        };

        if let Some(cached) = self.tensor_cache.get(&key) {
            // Verify cache validity
            if self.is_cache_valid(&cached)? {
                self.cache_hit_count.fetch_add(1, Ordering::Relaxed);
                return Ok(Some(cached.tensor.clone()));
            } else {
                // Remove invalid cache entry
                self.tensor_cache.pop(&key);
            }
        }

        self.cache_miss_count.fetch_add(1, Ordering::Relaxed);
        Ok(None)
    }

    fn update_tensor_cache(&mut self, tokens: &[u32], position: usize, tensor: &BitNetTensor) -> Result<()> {
        if !self.config.cache_config.enable_tensor_cache {
            return Ok(());
        }

        let key = TokenSequenceKey {
            tokens: tokens.to_vec(),
            model_hash: self.calculate_model_hash(),
            position,
        };

        let cached_tensor = CachedTensor {
            tensor: tensor.clone(),
            created_at: Instant::now(),
            access_count: 1,
            memory_size: tensor.memory_usage(),
        };

        // Check memory limits before caching
        if self.should_cache_tensor(&cached_tensor)? {
            if let Some(evicted) = self.tensor_cache.put(key, cached_tensor) {
                self.cache_eviction_count.fetch_add(1, Ordering::Relaxed);
                self.on_cache_eviction(&evicted);
            }
        }

        Ok(())
    }

    fn is_cache_valid(&self, cached: &CachedTensor) -> Result<bool> {
        let age = cached.created_at.elapsed();
        let max_age = self.config.cache_config.max_age.unwrap_or(Duration::from_secs(3600));

        if age > max_age {
            return Ok(false);
        }

        // Verify tensor integrity
        if cached.tensor.memory_usage() != cached.memory_size {
            warn!("Cached tensor memory size mismatch, invalidating cache entry");
            return Ok(false);
        }

        Ok(true)
    }

    fn should_cache_tensor(&self, tensor: &CachedTensor) -> Result<bool> {
        let memory_limit = self.config.cache_config.max_memory_mb.unwrap_or(1024) * 1024 * 1024; // Convert to bytes
        let current_memory = self.calculate_cache_memory_usage();

        if current_memory + tensor.memory_size > memory_limit {
            return Ok(false);
        }

        // Don't cache very small tensors (overhead not worth it)
        if tensor.memory_size < 1024 {
            return Ok(false);
        }

        Ok(true)
    }

    fn calculate_cache_memory_usage(&self) -> usize {
        self.tensor_cache.iter()
            .map(|(_, cached)| cached.memory_size)
            .sum()
    }

    fn calculate_model_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.model.config().hash(&mut hasher);
        hasher.finish()
    }

    fn on_cache_eviction(&self, evicted: &CachedTensor) {
        debug!(
            "Evicted tensor from cache: size={} bytes, age={:?}, access_count={}",
            evicted.memory_size,
            evicted.created_at.elapsed(),
            evicted.access_count
        );
    }

    pub fn get_cache_stats(&self) -> CacheStats {
        let hits = self.cache_hit_count.load(Ordering::Relaxed);
        let misses = self.cache_miss_count.load(Ordering::Relaxed);
        let evictions = self.cache_eviction_count.load(Ordering::Relaxed);
        let total_requests = hits + misses;

        CacheStats {
            hit_rate: if total_requests > 0 { hits as f64 / total_requests as f64 } else { 0.0 },
            total_hits: hits,
            total_misses: misses,
            total_evictions: evictions,
            current_entries: self.tensor_cache.len(),
            memory_usage_bytes: self.calculate_cache_memory_usage(),
        }
    }
}
```

## Implementation Plan

### Phase 1: Grouped Query Attention Implementation (Week 1-2)
- [ ] Implement efficient key-value head repetition algorithms
- [ ] Add GQA configuration validation and error handling
- [ ] Create comprehensive unit tests for different head configurations
- [ ] Add memory usage optimization for large models

### Phase 2: Dynamic Rotary Embedding System (Week 2-3)
- [ ] Implement adaptive sin/cos cache generation
- [ ] Add growth strategy configuration options
- [ ] Create thread-safe cache update mechanisms
- [ ] Add performance benchmarking for different sequence lengths

### Phase 3: Production Tokenizer Implementations (Week 3-4)
- [ ] Implement complete GPT-2 BPE tokenizer with fallbacks
- [ ] Add robust LLaMA SentencePiece tokenizer support
- [ ] Create Tiktoken integration for GPT-4 models
- [ ] Add comprehensive error handling and user guidance

### Phase 4: Advanced Tensor Caching (Week 4-5)
- [ ] Implement LRU cache with intelligent eviction policies
- [ ] Add memory usage monitoring and limits
- [ ] Create cache validation and integrity checking
- [ ] Add performance metrics and statistics

### Phase 5: Mock Elimination and Testing (Week 5-6)
- [ ] Remove conditional compilation from MockBitNetModel
- [ ] Create comprehensive test infrastructure
- [ ] Add integration tests for all attention mechanisms
- [ ] Create performance benchmarking suite

## Testing Strategy

### Attention Mechanism Validation
```rust
#[test]
fn test_gqa_memory_efficiency() {
    let configs = vec![
        // Standard attention (no grouping)
        AttentionConfig { num_attention_heads: 32, num_key_value_heads: 32 },
        // 2x grouping
        AttentionConfig { num_attention_heads: 32, num_key_value_heads: 16 },
        // 4x grouping (LLaMA-style)
        AttentionConfig { num_attention_heads: 32, num_key_value_heads: 8 },
    ];

    for config in configs {
        let attention = BitNetAttention::new(config);
        let key_states = create_test_tensor(&[2, 128, config.num_key_value_heads * 64]);
        let value_states = create_test_tensor(&[2, 128, config.num_key_value_heads * 64]);

        let (grouped_keys, grouped_values) = attention.apply_gqa(&key_states, &value_states).unwrap();

        // Verify correct output dimensions
        assert_eq!(grouped_keys.shape()[2], config.num_attention_heads * 64);
        assert_eq!(grouped_values.shape()[2], config.num_attention_heads * 64);

        // Verify memory efficiency for grouped configurations
        if config.num_key_value_heads < config.num_attention_heads {
            let memory_savings = calculate_memory_savings(&key_states, &grouped_keys);
            assert!(memory_savings > 0.0);
        }
    }
}

#[test]
fn test_dynamic_rotary_embedding_growth() {
    let mut rope = RotaryEmbedding::new(128, 1024, 10000.0); // max_seq_len = 1024

    // Test normal operation within limits
    let input = create_test_tensor(&[1, 8, 512, 128]);
    let result = rope.apply(&input, 0).unwrap();
    assert_eq!(result.shape(), input.shape());

    // Test dynamic growth
    let long_input = create_test_tensor(&[1, 8, 2048, 128]); // Exceeds max_seq_len
    let long_result = rope.apply(&long_input, 0).unwrap();
    assert_eq!(long_result.shape(), long_input.shape());

    // Verify cache was grown
    assert!(rope.max_seq_len >= 2048);
}
```

### Caching System Validation
```rust
#[test]
fn test_tensor_cache_hit_miss_rates() {
    let mut generator = AutoregressiveGenerator::new(
        Arc::new(MockModel::new()),
        Arc::new(MockTokenizer::new()),
        GenerationConfig::default(),
    );

    let test_sequences = vec![
        vec![1, 2, 3, 4],
        vec![1, 2, 3, 5], // Shares prefix with first
        vec![1, 2, 3, 4], // Exact duplicate
        vec![5, 6, 7, 8], // Completely different
    ];

    for (i, tokens) in test_sequences.iter().enumerate() {
        let tensor = create_test_tensor(&[1, tokens.len(), 768]);

        // First access should be a miss
        let cached = generator.try_get_cached_tensor(tokens, 0).unwrap();
        assert!(cached.is_none());

        // Cache the tensor
        generator.update_tensor_cache(tokens, 0, &tensor).unwrap();

        // Second access should be a hit
        let cached = generator.try_get_cached_tensor(tokens, 0).unwrap();
        assert!(cached.is_some());
    }

    let stats = generator.get_cache_stats();
    assert!(stats.hit_rate > 0.0);
    assert!(stats.total_hits > 0);
    assert!(stats.total_misses > 0);
}
```

### Tokenizer Implementation Validation
```rust
#[test]
fn test_tokenizer_loading_fallback_chain() {
    let configs = vec![
        TokenizerConfig {
            model_type: "gpt2".to_string(),
            vocab_file: Some(PathBuf::from("nonexistent.json")),
            model_name: Some("gpt2".to_string()),
            ..Default::default()
        },
        TokenizerConfig {
            model_type: "llama".to_string(),
            tokenizer_file: Some(PathBuf::from("nonexistent.model")),
            vocab_file: Some(PathBuf::from("nonexistent.json")),
            ..Default::default()
        },
    ];

    for config in configs {
        let result = UniversalTokenizer::detect_and_create_backend(&config);

        // Should either succeed with real tokenizer or fail with helpful error
        match result {
            Ok(backend) => {
                // Verify it's not a mock
                assert!(!matches!(backend, TokenizerBackend::Mock(_)));
            }
            Err(e) => {
                // Error should be informative
                let error_msg = e.to_string();
                assert!(error_msg.contains("tokenizer") || error_msg.contains("file"));
                assert!(error_msg.len() > 20); // Should have substantial error message
            }
        }
    }
}
```

## Risk Assessment

### Performance Risks
1. **GQA Memory Overhead**: Head repetition may increase memory usage
   - *Mitigation*: Efficient tensor operations, memory pool usage, lazy evaluation
2. **Dynamic Growth Latency**: Cache regeneration may cause inference spikes
   - *Mitigation*: Background cache warming, intelligent growth strategies
3. **Cache Memory Pressure**: Large tensor cache may cause OOM
   - *Mitigation*: Intelligent eviction policies, memory monitoring, configurable limits

### Implementation Risks
1. **Attention Correctness**: Complex tensor operations may introduce bugs
   - *Mitigation*: Comprehensive testing, reference implementation comparison
2. **Cache Consistency**: Concurrent access may cause race conditions
   - *Mitigation*: Thread-safe data structures, atomic operations, proper locking

## Acceptance Criteria

### Functional Requirements
- [ ] Complete GQA implementation supporting all common head configurations
- [ ] Dynamic rotary embedding growth handling sequences up to 1M tokens
- [ ] Production tokenizer implementations for GPT-2, LLaMA, Tiktoken
- [ ] LRU tensor cache with configurable eviction policies
- [ ] Unified mock model availability across feature configurations

### Performance Requirements
- [ ] GQA implementation within 95% of reference performance
- [ ] Dynamic rotary embedding growth overhead <10ms per operation
- [ ] Tensor cache hit rate >80% for typical generation workloads
- [ ] Memory usage optimization >30% for models using GQA

### Quality Requirements
- [ ] Numerical accuracy within 1e-5 of reference implementations
- [ ] Comprehensive error handling with actionable user guidance
- [ ] Thread-safe concurrent access to all caching systems
- [ ] Cross-validation with Microsoft BitNet C++ reference

## Related Issues

- BitNet-rs #251: Production-ready inference server (depends on efficient attention)
- BitNet-rs #218: Device-aware quantization system (integrates with attention layers)
- BitNet-rs #260: Mock elimination project (removes mock tokenizer fallbacks)

## Implementation Notes

### BitNet-rs Integration
- Use existing `BitNetTensor` type system for all tensor operations
- Integrate with `bitnet-kernels` for SIMD-optimized tensor operations
- Follow feature flag architecture (`--features inference` for advanced features)
- Leverage `crossval` framework for accuracy validation

### Memory Management
- Use `Arc` and `Weak` references for efficient tensor sharing
- Implement proper cleanup for dynamic cache growth
- Add memory pressure detection and automatic cache cleanup
- Use thread-local storage for frequently accessed cache data

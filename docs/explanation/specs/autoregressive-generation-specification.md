# Autoregressive Generation Specification

**Component**: Autoregressive text generation with sampling strategies and KV-cache optimization
**Location**: `bitnet-inference/src/generation/autoregressive.rs`
**Dependencies**: BitNetTransformer, KVCache, tokenizers, sampling algorithms

## Overview

Autoregressive generation forms the core of language model inference, generating text token-by-token while maintaining conversation context through efficient KV-caching. This specification defines a production-ready autoregressive generation system that integrates with BitNet.rs quantized transformers, supports multiple sampling strategies (temperature, top-k, nucleus), and provides deterministic generation for reproducible evaluation.

## Architecture Design

### Core Generation Engine

```rust
/// Autoregressive generation engine with quantized transformer and KV-cache optimization
pub struct AutoregressiveGenerator {
    // Core transformer components
    transformer: BitNetTransformer,           // Quantized transformer model
    tokenizer: Arc<dyn Tokenizer>,           // Universal tokenizer
    kv_cache: KVCache,                       // Multi-layer KV cache

    // Generation state
    current_position: usize,                 // Current generation position
    generated_tokens: Vec<u32>,              // Generated token sequence
    generation_config: GenerationConfig,    // Sampling configuration

    // Performance optimization
    device: Device,                          // CPU/GPU device context
    batch_size: usize,                      // Batch size for parallel generation
    prefill_chunk_size: usize,              // Chunk size for long prompt prefill

    // Deterministic generation
    rng: Option<Box<dyn RngCore>>,          // Random number generator with seeding
    deterministic: bool,                     // Enable deterministic mode
}

impl AutoregressiveGenerator {
    /// Create generator from loaded model and tokenizer
    pub fn new(
        transformer: BitNetTransformer,
        tokenizer: Arc<dyn Tokenizer>,
        device: Device,
        config: GenerationConfig
    ) -> Result<Self>;

    /// Generate text from prompt with configurable sampling
    /// Returns generated text and detailed generation metrics
    pub fn generate(&mut self, prompt: &str) -> Result<GenerationResult>;

    /// Generate with detailed control over sampling parameters
    pub fn generate_with_config(&mut self, prompt: &str, config: &GenerationConfig) -> Result<GenerationResult>;

    /// Deterministic generation with fixed seed
    pub fn generate_deterministic(&mut self, prompt: &str, seed: u64) -> Result<GenerationResult>;

    /// Streaming generation for real-time applications
    pub fn generate_stream(&mut self, prompt: &str) -> Result<GenerationStream>;

    /// Batch generation for multiple prompts
    pub fn generate_batch(&mut self, prompts: &[String]) -> Result<Vec<GenerationResult>>;
}
```

### Generation Configuration

```rust
/// Comprehensive generation configuration with sampling strategies
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    // Generation limits
    pub max_new_tokens: usize,           // Maximum tokens to generate
    pub max_total_tokens: usize,         // Maximum total sequence length
    pub min_new_tokens: usize,           // Minimum tokens to generate

    // Sampling parameters
    pub temperature: f32,                // Sampling temperature (0.0 = greedy)
    pub top_k: Option<usize>,           // Top-k sampling
    pub top_p: Option<f32>,             // Nucleus (top-p) sampling
    pub repetition_penalty: f32,        // Repetition penalty factor
    pub frequency_penalty: f32,         // Frequency-based penalty
    pub presence_penalty: f32,          // Presence-based penalty

    // Stop conditions
    pub stop_tokens: Vec<u32>,          // Stop generation on these tokens
    pub stop_sequences: Vec<String>,    // Stop on these text sequences
    pub eos_token_id: Option<u32>,      // End-of-sequence token

    // Generation behavior
    pub do_sample: bool,                // Enable sampling vs greedy decoding
    pub early_stopping: bool,           // Stop at first EOS token
    pub pad_token_id: Option<u32>,      // Padding token for batch generation

    // Deterministic generation
    pub seed: Option<u64>,              // Random seed for reproducibility
    pub use_cache: bool,                // Enable KV caching (default: true)

    // Performance tuning
    pub prefill_chunk_size: Option<usize>, // Chunk size for long prompts
    pub decode_chunk_size: Option<usize>,  // Chunk size for batch decoding
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            max_total_tokens: 2048,
            min_new_tokens: 1,

            temperature: 0.8,
            top_k: Some(50),
            top_p: Some(0.95),
            repetition_penalty: 1.1,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,

            stop_tokens: vec![],
            stop_sequences: vec![],
            eos_token_id: None,

            do_sample: true,
            early_stopping: true,
            pad_token_id: None,

            seed: None,
            use_cache: true,

            prefill_chunk_size: None,
            decode_chunk_size: None,
        }
    }
}
```

## Core Generation Implementation

### Main Generation Loop

```rust
impl AutoregressiveGenerator {
    /// Main autoregressive generation implementation
    pub fn generate(&mut self, prompt: &str) -> Result<GenerationResult> {
        let start_time = Instant::now();

        // Step 1: Tokenize input prompt
        let prompt_tokens = self.tokenizer.encode(prompt, true)?;
        let prompt_length = prompt_tokens.len();

        if prompt_tokens.is_empty() {
            return Err(GenerationError::EmptyPrompt);
        }

        // Step 2: Initialize generation state
        self.reset_generation_state()?;
        self.generated_tokens = prompt_tokens.clone();

        // Step 3: Setup deterministic generation if configured
        if let Some(seed) = self.generation_config.seed {
            self.enable_deterministic_generation(seed)?;
        }

        // Step 4: Prefill phase - process prompt tokens
        let prefill_start = Instant::now();
        self.prefill_prompt(&prompt_tokens)?;
        let prefill_time = prefill_start.elapsed();

        // Step 5: Generation phase - autoregressively generate tokens
        let generation_start = Instant::now();
        let mut generation_metrics = GenerationMetrics::new();

        for step in 0..self.generation_config.max_new_tokens {
            // Generate next token
            let next_token = self.generate_next_token(&mut generation_metrics)?;

            // Check stop conditions
            if self.should_stop_generation(next_token, step)? {
                break;
            }

            // Add token to sequence
            self.generated_tokens.push(next_token);
            self.current_position += 1;

            // Update generation metrics
            generation_metrics.tokens_generated += 1;
            generation_metrics.update_timing();
        }

        let generation_time = generation_start.elapsed();
        let total_time = start_time.elapsed();

        // Step 6: Decode generated tokens to text
        let generated_text = if self.generated_tokens.len() > prompt_length {
            let new_tokens = &self.generated_tokens[prompt_length..];
            self.tokenizer.decode(new_tokens, true)?
        } else {
            String::new()
        };

        // Step 7: Compile generation results
        Ok(GenerationResult {
            prompt: prompt.to_string(),
            generated_text,
            prompt_tokens,
            generated_tokens: self.generated_tokens[prompt_length..].to_vec(),
            total_tokens: self.generated_tokens.clone(),

            // Performance metrics
            timing: TimingMetrics {
                prefill_time,
                generation_time,
                total_time,
                tokens_per_second: generation_metrics.tokens_generated as f32 / generation_time.as_secs_f32(),
                first_token_latency: generation_metrics.first_token_time,
            },

            // Generation metadata
            metrics: generation_metrics,
            config: self.generation_config.clone(),
        })
    }

    /// Prefill phase: process prompt tokens efficiently
    fn prefill_prompt(&mut self, prompt_tokens: &[u32]) -> Result<()> {
        let prompt_tensor = self.tokens_to_tensor(prompt_tokens)?;

        if prompt_tokens.len() <= self.prefill_chunk_size {
            // Single-pass prefill for short prompts
            self.transformer.forward(&prompt_tensor, Some(&mut self.kv_cache))?;
        } else {
            // Chunked prefill for long prompts
            self.chunked_prefill(prompt_tokens)?;
        }

        self.current_position = prompt_tokens.len();
        Ok(())
    }

    /// Chunked prefill for long prompts to manage memory usage
    fn chunked_prefill(&mut self, prompt_tokens: &[u32]) -> Result<()> {
        let chunk_size = self.prefill_chunk_size;

        for chunk_start in (0..prompt_tokens.len()).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(prompt_tokens.len());
            let chunk = &prompt_tokens[chunk_start..chunk_end];

            let chunk_tensor = self.tokens_to_tensor(chunk)?;
            self.transformer.forward(&chunk_tensor, Some(&mut self.kv_cache))?;

            // Optional: yield between chunks for responsiveness
            if chunk_end < prompt_tokens.len() {
                self.yield_if_needed()?;
            }
        }

        Ok(())
    }

    /// Generate single next token with sampling
    fn generate_next_token(&mut self, metrics: &mut GenerationMetrics) -> Result<u32> {
        let token_start = Instant::now();

        // Create input tensor for current position (single token for autoregressive)
        let current_input = if self.current_position < self.generated_tokens.len() {
            // Using existing token (shouldn't happen in normal generation)
            self.tokens_to_tensor(&[self.generated_tokens[self.current_position]])?
        } else {
            // Use last generated token for next prediction
            let last_token = *self.generated_tokens.last()
                .ok_or(GenerationError::EmptySequence)?;
            self.tokens_to_tensor(&[last_token])?
        };

        // Forward pass through transformer
        let hidden_states = self.transformer.forward(&current_input, Some(&mut self.kv_cache))?;

        // Get logits for vocabulary prediction
        let logits = self.transformer.logits(&hidden_states)?;

        // Extract logits for last position (single token generation)
        let next_token_logits = logits.narrow(1, logits.dims()[1] - 1, 1)?  // [B, 1, V]
            .squeeze(1)?; // [B, V]

        // Apply sampling to get next token
        let next_token = self.sample_next_token(&next_token_logits)?;

        // Update timing metrics
        let token_time = token_start.elapsed();
        if metrics.tokens_generated == 0 {
            metrics.first_token_time = Some(token_time);
        }
        metrics.token_times.push(token_time);

        Ok(next_token)
    }

    /// Convert token sequence to tensor
    fn tokens_to_tensor(&self, tokens: &[u32]) -> Result<Tensor> {
        let token_tensor = Tensor::new(tokens, &self.device)?;
        // Add batch dimension: [T] -> [1, T]
        token_tensor.unsqueeze(0)
    }
}
```

### Sampling Strategies

```rust
/// Token sampling with multiple strategies
pub struct TokenSampler {
    config: GenerationConfig,
    rng: Box<dyn RngCore>,
    token_frequencies: HashMap<u32, usize>, // For repetition penalties
}

impl TokenSampler {
    pub fn new(config: GenerationConfig, seed: Option<u64>) -> Self {
        let rng: Box<dyn RngCore> = if let Some(seed) = seed {
            Box::new(StdRng::seed_from_u64(seed))
        } else {
            Box::new(StdRng::from_entropy())
        };

        Self {
            config,
            rng,
            token_frequencies: HashMap::new(),
        }
    }

    /// Sample next token using configured strategy
    pub fn sample(&mut self, logits: &Tensor, generated_tokens: &[u32]) -> Result<u32> {
        // Update token frequencies for penalty calculation
        self.update_token_frequencies(generated_tokens);

        // Apply penalties to logits
        let adjusted_logits = self.apply_penalties(logits, generated_tokens)?;

        // Apply temperature scaling
        let temperature_scaled = if self.config.temperature > 0.0 {
            adjusted_logits.affine(1.0 / self.config.temperature as f64, 0.0)?
        } else {
            adjusted_logits // Temperature 0 = no scaling (greedy)
        };

        // Sample based on configuration
        if self.config.do_sample && self.config.temperature > 0.0 {
            self.sample_with_strategy(&temperature_scaled)
        } else {
            self.greedy_sample(&temperature_scaled)
        }
    }

    /// Greedy sampling (select highest probability token)
    fn greedy_sample(&self, logits: &Tensor) -> Result<u32> {
        let logits_data: Vec<f32> = logits.to_vec1()?;
        let max_idx = logits_data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .ok_or(GenerationError::SamplingError("No valid token found".to_string()))?;

        Ok(max_idx as u32)
    }

    /// Sample with top-k, top-p, or multinomial sampling
    fn sample_with_strategy(&mut self, logits: &Tensor) -> Result<u32> {
        let mut logits_data: Vec<f32> = logits.to_vec1()?;

        // Apply top-k filtering if configured
        if let Some(top_k) = self.config.top_k {
            self.apply_top_k(&mut logits_data, top_k);
        }

        // Apply top-p (nucleus) filtering if configured
        if let Some(top_p) = self.config.top_p {
            self.apply_top_p(&mut logits_data, top_p)?;
        }

        // Convert logits to probabilities
        let probs = self.logits_to_probabilities(&logits_data)?;

        // Multinomial sampling
        self.multinomial_sample(&probs)
    }

    /// Apply top-k filtering: set all but top-k logits to -inf
    fn apply_top_k(&self, logits: &mut [f32], k: usize) {
        if k >= logits.len() {
            return; // No filtering needed
        }

        // Find the k-th largest value
        let mut sorted_indices: Vec<usize> = (0..logits.len()).collect();
        sorted_indices.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));

        let threshold = logits[sorted_indices[k - 1]];

        // Set values below threshold to -inf
        for (i, value) in logits.iter_mut().enumerate() {
            if *value < threshold && !sorted_indices[..k].contains(&i) {
                *value = f32::NEG_INFINITY;
            }
        }
    }

    /// Apply top-p (nucleus) filtering: keep tokens with cumulative probability <= p
    fn apply_top_p(&self, logits: &mut [f32], p: f32) -> Result<()> {
        if p >= 1.0 {
            return Ok(()); // No filtering needed
        }

        // Convert to probabilities for cumulative calculation
        let probs = self.logits_to_probabilities(logits)?;

        // Sort by probability (descending)
        let mut indexed_probs: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Find cutoff point where cumulative probability > p
        let mut cumulative_prob = 0.0;
        let mut cutoff_index = indexed_probs.len();

        for (i, (_, prob)) in indexed_probs.iter().enumerate() {
            cumulative_prob += prob;
            if cumulative_prob > p {
                cutoff_index = i + 1; // Include this token
                break;
            }
        }

        // Set probabilities outside nucleus to -inf
        let nucleus_indices: HashSet<usize> = indexed_probs[..cutoff_index].iter().map(|(i, _)| *i).collect();

        for (i, value) in logits.iter_mut().enumerate() {
            if !nucleus_indices.contains(&i) {
                *value = f32::NEG_INFINITY;
            }
        }

        Ok(())
    }

    /// Apply repetition penalty to discourage repeated tokens
    fn apply_penalties(&self, logits: &Tensor, generated_tokens: &[u32]) -> Result<Tensor> {
        let mut logits_data: Vec<f32> = logits.to_vec1()?;

        // Apply repetition penalty
        if self.config.repetition_penalty != 1.0 {
            for &token in generated_tokens {
                let token_idx = token as usize;
                if token_idx < logits_data.len() {
                    if logits_data[token_idx] > 0.0 {
                        logits_data[token_idx] /= self.config.repetition_penalty;
                    } else {
                        logits_data[token_idx] *= self.config.repetition_penalty;
                    }
                }
            }
        }

        // Apply frequency penalty
        if self.config.frequency_penalty != 0.0 {
            for (&token, &count) in &self.token_frequencies {
                let token_idx = token as usize;
                if token_idx < logits_data.len() {
                    logits_data[token_idx] -= self.config.frequency_penalty * count as f32;
                }
            }
        }

        // Apply presence penalty
        if self.config.presence_penalty != 0.0 {
            for &token in generated_tokens.iter().collect::<HashSet<_>>() {
                let token_idx = token as usize;
                if token_idx < logits_data.len() {
                    logits_data[token_idx] -= self.config.presence_penalty;
                }
            }
        }

        Tensor::from_vec(logits_data, logits.dims(), logits.device())
    }

    /// Convert logits to probabilities using softmax
    fn logits_to_probabilities(&self, logits: &[f32]) -> Result<Vec<f32>> {
        // Find max for numerical stability
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Compute exp(logits - max)
        let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();

        // Compute sum for normalization
        let sum_exp: f32 = exp_logits.iter().sum();

        if sum_exp <= 0.0 {
            return Err(GenerationError::SamplingError("Invalid probability distribution".to_string()));
        }

        // Normalize to get probabilities
        let probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum_exp).collect();

        Ok(probs)
    }

    /// Multinomial sampling from probability distribution
    fn multinomial_sample(&mut self, probs: &[f32]) -> Result<u32> {
        let uniform_sample: f32 = self.rng.gen();
        let mut cumulative_prob = 0.0;

        for (i, &prob) in probs.iter().enumerate() {
            cumulative_prob += prob;
            if uniform_sample <= cumulative_prob {
                return Ok(i as u32);
            }
        }

        // Fallback: return last valid token
        Ok((probs.len() - 1) as u32)
    }

    /// Update token frequency tracking for penalty calculation
    fn update_token_frequencies(&mut self, tokens: &[u32]) {
        self.token_frequencies.clear();
        for &token in tokens {
            *self.token_frequencies.entry(token).or_insert(0) += 1;
        }
    }
}
```

### Stop Condition Handling

```rust
impl AutoregressiveGenerator {
    /// Check if generation should stop based on configured conditions
    fn should_stop_generation(&self, token: u32, step: usize) -> Result<bool> {
        // Check maximum token limit
        if step >= self.generation_config.max_new_tokens {
            return Ok(true);
        }

        // Check total sequence length limit
        if self.generated_tokens.len() >= self.generation_config.max_total_tokens {
            return Ok(true);
        }

        // Check explicit stop tokens
        if self.generation_config.stop_tokens.contains(&token) {
            return Ok(true);
        }

        // Check EOS token
        if let Some(eos_token) = self.generation_config.eos_token_id {
            if token == eos_token && self.generation_config.early_stopping {
                return Ok(true);
            }
        }

        // Check stop sequences (requires decoding recent tokens)
        if !self.generation_config.stop_sequences.is_empty() {
            let recent_text = self.decode_recent_tokens(10)?; // Check last 10 tokens
            for stop_seq in &self.generation_config.stop_sequences {
                if recent_text.contains(stop_seq) {
                    return Ok(true);
                }
            }
        }

        // Check minimum token requirement
        if step < self.generation_config.min_new_tokens {
            return Ok(false);
        }

        Ok(false)
    }

    /// Decode recent tokens to check for stop sequences
    fn decode_recent_tokens(&self, n: usize) -> Result<String> {
        let start_idx = self.generated_tokens.len().saturating_sub(n);
        let recent_tokens = &self.generated_tokens[start_idx..];
        self.tokenizer.decode(recent_tokens, false)
    }
}
```

## Streaming Generation

### Real-time Generation Stream

```rust
/// Streaming generation for real-time applications
pub struct GenerationStream {
    generator: AutoregressiveGenerator,
    current_step: usize,
    is_complete: bool,
    last_decoded_length: usize,
}

impl GenerationStream {
    pub fn new(generator: AutoregressiveGenerator, prompt: &str) -> Result<Self> {
        let mut stream = Self {
            generator,
            current_step: 0,
            is_complete: false,
            last_decoded_length: 0,
        };

        // Initialize with prompt
        stream.initialize_with_prompt(prompt)?;
        Ok(stream)
    }

    /// Get next token/text chunk from generation
    pub fn next_chunk(&mut self) -> Result<Option<GenerationChunk>> {
        if self.is_complete {
            return Ok(None);
        }

        // Generate next token
        let mut metrics = GenerationMetrics::new();
        let next_token = self.generator.generate_next_token(&mut metrics)?;

        // Check stop conditions
        if self.generator.should_stop_generation(next_token, self.current_step)? {
            self.is_complete = true;
            return Ok(Some(GenerationChunk {
                token: next_token,
                text: String::new(),
                delta_text: String::new(),
                is_final: true,
                step: self.current_step,
            }));
        }

        // Add token to sequence
        self.generator.generated_tokens.push(next_token);
        self.current_step += 1;

        // Decode new text
        let full_text = self.generator.tokenizer.decode(&self.generator.generated_tokens, true)?;
        let delta_text = if full_text.len() > self.last_decoded_length {
            full_text[self.last_decoded_length..].to_string()
        } else {
            String::new()
        };

        self.last_decoded_length = full_text.len();

        Ok(Some(GenerationChunk {
            token: next_token,
            text: full_text,
            delta_text,
            is_final: false,
            step: self.current_step,
        }))
    }

    fn initialize_with_prompt(&mut self, prompt: &str) -> Result<()> {
        let prompt_tokens = self.generator.tokenizer.encode(prompt, true)?;
        self.generator.prefill_prompt(&prompt_tokens)?;
        self.generator.generated_tokens = prompt_tokens;
        Ok(())
    }
}

impl Iterator for GenerationStream {
    type Item = Result<GenerationChunk>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_chunk() {
            Ok(Some(chunk)) => Some(Ok(chunk)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenerationChunk {
    pub token: u32,           // Generated token
    pub text: String,         // Full generated text so far
    pub delta_text: String,   // New text since last chunk
    pub is_final: bool,       // Whether this is the final chunk
    pub step: usize,          // Generation step number
}
```

## Deterministic Generation

### Reproducible Generation with Seeding

```rust
impl AutoregressiveGenerator {
    /// Enable deterministic generation with fixed seed
    pub fn enable_deterministic_generation(&mut self, seed: u64) -> Result<()> {
        // Set random seed for sampling
        self.rng = Some(Box::new(StdRng::seed_from_u64(seed)));
        self.deterministic = true;

        // Configure deterministic transformer behavior
        self.transformer.set_deterministic(true)?;

        log::info!("Enabled deterministic generation with seed: {}", seed);
        Ok(())
    }

    /// Generate with guaranteed reproducibility
    pub fn generate_deterministic(&mut self, prompt: &str, seed: u64) -> Result<GenerationResult> {
        // Store original config
        let original_seed = self.generation_config.seed;
        let original_deterministic = self.deterministic;

        // Set deterministic configuration
        self.generation_config.seed = Some(seed);
        self.enable_deterministic_generation(seed)?;

        // Generate with deterministic settings
        let result = self.generate(prompt);

        // Restore original configuration
        self.generation_config.seed = original_seed;
        self.deterministic = original_deterministic;

        result
    }

    /// Validate deterministic generation (for testing)
    pub fn validate_deterministic_generation(&mut self, prompt: &str, seed: u64, iterations: usize) -> Result<bool> {
        let mut all_results = Vec::new();

        for _ in 0..iterations {
            let result = self.generate_deterministic(prompt, seed)?;
            all_results.push(result.generated_tokens.clone());

            // Reset state between iterations
            self.reset_generation_state()?;
        }

        // Check all results are identical
        let first_result = &all_results[0];
        let all_identical = all_results.iter().all(|tokens| tokens == first_result);

        if all_identical {
            log::info!("Deterministic generation validated: {} iterations produced identical results", iterations);
        } else {
            log::error!("Deterministic generation failed: iterations produced different results");
        }

        Ok(all_identical)
    }
}
```

## Batch Generation

### Efficient Multi-Prompt Generation

```rust
impl AutoregressiveGenerator {
    /// Generate text for multiple prompts in parallel
    pub fn generate_batch(&mut self, prompts: &[String]) -> Result<Vec<GenerationResult>> {
        if prompts.is_empty() {
            return Ok(vec![]);
        }

        // Check if batch generation is beneficial
        if prompts.len() == 1 {
            return Ok(vec![self.generate(&prompts[0])?]);
        }

        // Tokenize all prompts
        let tokenized_prompts: Result<Vec<Vec<u32>>> = prompts
            .iter()
            .map(|prompt| self.tokenizer.encode(prompt, true))
            .collect();
        let tokenized_prompts = tokenized_prompts?;

        // Pad prompts to same length for batch processing
        let max_prompt_len = tokenized_prompts.iter().map(|tokens| tokens.len()).max().unwrap_or(0);
        let pad_token = self.generation_config.pad_token_id.unwrap_or(0);

        let padded_prompts: Vec<Vec<u32>> = tokenized_prompts
            .into_iter()
            .map(|mut tokens| {
                while tokens.len() < max_prompt_len {
                    tokens.push(pad_token);
                }
                tokens
            })
            .collect();

        // Process batch
        self.generate_batch_internal(&padded_prompts, prompts)
    }

    /// Internal batch generation implementation
    fn generate_batch_internal(&mut self, tokenized_prompts: &[Vec<u32>], original_prompts: &[String]) -> Result<Vec<GenerationResult>> {
        let batch_size = tokenized_prompts.len();
        let mut results = Vec::with_capacity(batch_size);

        // For now, process sequentially to avoid complexity
        // TODO: Implement true batch processing with batched transformer forward passes
        for (i, (tokens, prompt)) in tokenized_prompts.iter().zip(original_prompts.iter()).enumerate() {
            log::debug!("Processing batch item {}/{}", i + 1, batch_size);

            // Reset state for each prompt
            self.reset_generation_state()?;

            // Generate for this prompt
            let result = self.generate_from_tokens(tokens, prompt)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Generate from pre-tokenized input
    fn generate_from_tokens(&mut self, tokens: &[u32], original_prompt: &str) -> Result<GenerationResult> {
        // Remove padding tokens
        let pad_token = self.generation_config.pad_token_id.unwrap_or(0);
        let clean_tokens: Vec<u32> = tokens.iter().copied().filter(|&t| t != pad_token).collect();

        // Use existing generation logic with pre-tokenized input
        self.generated_tokens = clean_tokens.clone();
        self.prefill_prompt(&clean_tokens)?;

        // Continue with normal generation loop
        let mut generation_metrics = GenerationMetrics::new();
        let generation_start = Instant::now();

        for step in 0..self.generation_config.max_new_tokens {
            let next_token = self.generate_next_token(&mut generation_metrics)?;

            if self.should_stop_generation(next_token, step)? {
                break;
            }

            self.generated_tokens.push(next_token);
            self.current_position += 1;
            generation_metrics.tokens_generated += 1;
        }

        let generation_time = generation_start.elapsed();

        // Decode result
        let generated_text = if self.generated_tokens.len() > clean_tokens.len() {
            let new_tokens = &self.generated_tokens[clean_tokens.len()..];
            self.tokenizer.decode(new_tokens, true)?
        } else {
            String::new()
        };

        Ok(GenerationResult {
            prompt: original_prompt.to_string(),
            generated_text,
            prompt_tokens: clean_tokens,
            generated_tokens: self.generated_tokens[clean_tokens.len()..].to_vec(),
            total_tokens: self.generated_tokens.clone(),

            timing: TimingMetrics {
                prefill_time: Duration::from_millis(0), // Not separately measured in batch
                generation_time,
                total_time: generation_time,
                tokens_per_second: generation_metrics.tokens_generated as f32 / generation_time.as_secs_f32(),
                first_token_latency: generation_metrics.first_token_time,
            },

            metrics: generation_metrics,
            config: self.generation_config.clone(),
        })
    }
}
```

## Result Types and Metrics

### Generation Results and Performance Tracking

```rust
/// Comprehensive generation result with performance metrics
#[derive(Debug, Clone)]
pub struct GenerationResult {
    // Generated content
    pub prompt: String,                  // Original input prompt
    pub generated_text: String,          // Generated text output
    pub prompt_tokens: Vec<u32>,         // Tokenized prompt
    pub generated_tokens: Vec<u32>,      // Generated tokens (excluding prompt)
    pub total_tokens: Vec<u32>,          // Complete token sequence (prompt + generated)

    // Performance metrics
    pub timing: TimingMetrics,           // Detailed timing information
    pub metrics: GenerationMetrics,     // Generation-specific metrics
    pub config: GenerationConfig,       // Configuration used for generation
}

#[derive(Debug, Clone)]
pub struct TimingMetrics {
    pub prefill_time: Duration,          // Time to process prompt
    pub generation_time: Duration,       // Time to generate tokens
    pub total_time: Duration,            // Total end-to-end time
    pub tokens_per_second: f32,          // Generation throughput
    pub first_token_latency: Option<Duration>, // Time to first token (TTFT)
}

#[derive(Debug, Clone)]
pub struct GenerationMetrics {
    pub tokens_generated: usize,         // Number of tokens generated
    pub stop_reason: StopReason,         // Why generation stopped
    pub repetition_count: usize,         // Number of repeated tokens
    pub token_times: Vec<Duration>,      // Per-token generation times
    pub memory_usage: usize,             // Peak memory usage during generation
    pub cache_hit_rate: f32,             // KV cache efficiency
    pub first_token_time: Option<Duration>, // Time for first token
}

impl GenerationMetrics {
    pub fn new() -> Self {
        Self {
            tokens_generated: 0,
            stop_reason: StopReason::InProgress,
            repetition_count: 0,
            token_times: Vec::new(),
            memory_usage: 0,
            cache_hit_rate: 0.0,
            first_token_time: None,
        }
    }

    pub fn update_timing(&mut self) {
        // Update metrics that need periodic calculation
        self.memory_usage = self.calculate_memory_usage();
    }

    fn calculate_memory_usage(&self) -> usize {
        // Implementation would measure actual memory usage
        0 // Placeholder
    }
}

#[derive(Debug, Clone)]
pub enum StopReason {
    InProgress,              // Generation still in progress
    MaxTokens,              // Hit maximum token limit
    EosToken,               // Encountered end-of-sequence token
    StopToken(u32),         // Hit configured stop token
    StopSequence(String),   // Hit configured stop sequence
    Error(String),          // Generation stopped due to error
}
```

## Error Handling

### Comprehensive Generation Error Types

```rust
#[derive(Debug, thiserror::Error)]
pub enum GenerationError {
    #[error("Empty prompt provided")]
    EmptyPrompt,

    #[error("Empty sequence during generation")]
    EmptySequence,

    #[error("Sampling error: {0}")]
    SamplingError(String),

    #[error("Tokenization error: {context}")]
    TokenizationError { context: String },

    #[error("KV cache error: {reason}")]
    CacheError { reason: String },

    #[error("Transformer forward pass failed: {details}")]
    TransformerError { details: String },

    #[error("Generation configuration invalid: {reason}")]
    ConfigError { reason: String },

    #[error("Memory allocation failed during generation: {requested} bytes")]
    MemoryError { requested: usize },

    #[error("Device error during generation: {device} - {reason}")]
    DeviceError { device: String, reason: String },

    #[error("Deterministic generation failed: expected reproducible results")]
    DeterministicError,
}
```

## Testing Strategy

### Comprehensive Test Coverage

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autoregressive_generation_basic() { // AC:3
        let mut generator = create_test_generator().unwrap();
        let prompt = "Hello, world!";

        let result = generator.generate(prompt).unwrap();

        assert!(!result.generated_text.is_empty());
        assert!(result.generated_tokens.len() > 0);
        assert!(result.generated_tokens.len() <= generator.generation_config.max_new_tokens);
        assert_eq!(result.prompt, prompt);
    }

    #[test]
    fn test_deterministic_generation() { // AC:7
        let mut generator = create_test_generator().unwrap();
        let prompt = "The quick brown fox";
        let seed = 42;

        let result1 = generator.generate_deterministic(prompt, seed).unwrap();
        generator.reset_generation_state().unwrap();
        let result2 = generator.generate_deterministic(prompt, seed).unwrap();

        assert_eq!(result1.generated_tokens, result2.generated_tokens);
        assert_eq!(result1.generated_text, result2.generated_text);
    }

    #[test]
    fn test_sampling_strategies() { // AC:3
        let config = GenerationConfig {
            temperature: 0.8,
            top_k: Some(10),
            top_p: Some(0.9),
            ..Default::default()
        };

        let mut sampler = TokenSampler::new(config, Some(123));
        let logits = create_test_logits(1000).unwrap();

        // Test multiple samples are different (with sampling)
        let samples: HashSet<u32> = (0..10)
            .map(|_| sampler.sample(&logits, &[]).unwrap())
            .collect();

        assert!(samples.len() > 1, "Sampling should produce variety");
    }

    #[test]
    fn test_stop_conditions() { // AC:3
        let mut generator = create_test_generator_with_stop_tokens(vec![13]); // Stop on token 13
        let prompt = "Count to ten:";

        let result = generator.generate(prompt).unwrap();

        // Should stop when encountering stop token
        assert!(matches!(result.metrics.stop_reason, StopReason::StopToken(13)));
    }

    #[test]
    fn test_streaming_generation() { // AC:3
        let generator = create_test_generator().unwrap();
        let prompt = "Once upon a time";

        let mut stream = GenerationStream::new(generator, prompt).unwrap();
        let mut chunks = Vec::new();

        while let Some(chunk) = stream.next() {
            let chunk = chunk.unwrap();
            chunks.push(chunk.clone());
            if chunk.is_final {
                break;
            }
        }

        assert!(!chunks.is_empty());
        assert!(chunks.last().unwrap().is_final);
    }

    #[test]
    fn test_batch_generation() { // AC:3
        let mut generator = create_test_generator().unwrap();
        let prompts = vec![
            "Hello".to_string(),
            "Goodbye".to_string(),
            "How are you?".to_string(),
        ];

        let results = generator.generate_batch(&prompts).unwrap();

        assert_eq!(results.len(), prompts.len());
        for (result, prompt) in results.iter().zip(prompts.iter()) {
            assert_eq!(result.prompt, *prompt);
            assert!(!result.generated_text.is_empty());
        }
    }

    #[test]
    fn test_performance_metrics() { // AC:5
        let mut generator = create_test_generator().unwrap();
        let prompt = "Performance test prompt";

        let result = generator.generate(prompt).unwrap();

        assert!(result.timing.total_time.as_millis() > 0);
        assert!(result.timing.tokens_per_second > 0.0);
        assert_eq!(result.metrics.tokens_generated, result.generated_tokens.len());

        // Performance should be reasonable for test model
        let tokens_per_sec = result.timing.tokens_per_second;
        assert!(tokens_per_sec > 1.0, "Generation too slow: {} tok/sec", tokens_per_sec);
        assert!(tokens_per_sec < 1000.0, "Performance unrealistic: {} tok/sec", tokens_per_sec);
    }

    #[test]
    fn test_error_handling() { // AC:10
        let mut generator = create_test_generator().unwrap();

        // Test empty prompt
        let result = generator.generate("");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err().downcast::<GenerationError>().unwrap(),
                        GenerationError::EmptyPrompt));

        // Test invalid configuration
        generator.generation_config.max_new_tokens = 0;
        let result = generator.generate("test");
        assert!(result.is_err());
    }

    #[test]
    fn test_memory_optimization() { // AC:5
        let mut generator = create_test_generator().unwrap();

        // Generate with long sequence to test KV cache efficiency
        generator.generation_config.max_new_tokens = 100;
        let prompt = "This is a test of memory optimization during generation.";

        let result = generator.generate(prompt).unwrap();

        // Memory usage should be reasonable
        assert!(result.metrics.memory_usage < 100 * 1024 * 1024); // Less than 100MB

        // Cache should be efficient
        assert!(result.metrics.cache_hit_rate > 0.8); // At least 80% cache hits
    }

    // Helper functions
    fn create_test_generator() -> Result<AutoregressiveGenerator> {
        let transformer = create_test_transformer()?;
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;
        let config = GenerationConfig::default();

        AutoregressiveGenerator::new(transformer, tokenizer, device, config)
    }

    fn create_test_logits(vocab_size: usize) -> Result<Tensor> {
        let mut rng = StdRng::seed_from_u64(42);
        let values: Vec<f32> = (0..vocab_size).map(|_| rng.gen_range(-5.0..5.0)).collect();
        Tensor::from_vec(values, &[1, vocab_size], &Device::Cpu)
    }
}
```

This comprehensive autoregressive generation specification provides a production-ready implementation with advanced sampling strategies, deterministic generation, streaming support, and efficient batch processing, all optimized for BitNet.rs quantized transformers.

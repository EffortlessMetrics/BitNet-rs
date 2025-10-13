# [STUB] Incremental Logits Evaluation Function is Placeholder - Missing KV Cache State Management

## Problem Description

The `eval_logits_incremental` function in `crates/bitnet-inference/src/parity.rs` is a placeholder implementation that simply calls `eval_logits_once`, completely ignoring the incremental evaluation requirements. This stub prevents efficient autoregressive generation and breaks the intended performance optimizations of maintaining KV cache state across token generation steps.

## Environment

- **File**: `crates/bitnet-inference/src/parity.rs`
- **Function**: `eval_logits_incremental` (lines 11-19)
- **Component**: Inference parity layer and autoregressive generation
- **Build Configuration**: All feature configurations
- **Context**: Token-by-token generation with state preservation

## Root Cause Analysis

### Technical Issues

1. **Complete Lack of Incremental Logic**:
   ```rust
   pub fn eval_logits_incremental(
       model_path: &str,
       tokens: &[i32],
       _n_past: usize,  // Ignored parameter
   ) -> Result<Vec<f32>> {
       // For now, just call the single-shot version
       eval_logits_once(model_path, tokens)  // No state management
   }
   ```

2. **Missing State Management**:
   - No KV cache initialization or maintenance
   - No model state persistence between calls
   - No tracking of past tokens or position embeddings

3. **Performance Implications**:
   - Recomputes entire sequence from scratch each time
   - No benefit from incremental computation
   - O(n²) complexity instead of O(n) for autoregressive generation

4. **API Contract Violation**:
   - Function signature promises incremental behavior
   - `n_past` parameter is completely ignored
   - Breaks expectations for efficient token generation

### Impact Assessment

- **Performance**: 10-100x slower than proper incremental evaluation
- **Memory**: Excessive memory usage due to full recomputation
- **Scalability**: Poor performance with longer sequences
- **API Reliability**: Misleading function behavior

## Reproduction Steps

1. Attempt to use incremental evaluation for token generation:
   ```rust
   let mut past_tokens = vec![1, 2, 3];
   let logits1 = eval_logits_incremental("model.gguf", &past_tokens, 0)?;

   past_tokens.push(4);
   let logits2 = eval_logits_incremental("model.gguf", &past_tokens, 3)?;
   // Should be much faster than first call, but isn't
   ```

2. **Expected**: Second call should only process new token with cached state
3. **Actual**: Both calls process entire sequence from scratch

## Proposed Solution

### Primary Approach: Proper Incremental Evaluation with State Management

Implement genuine incremental evaluation with persistent state management:

```rust
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// Global state management for incremental evaluation
lazy_static::lazy_static! {
    static ref INCREMENTAL_STATES: Arc<Mutex<HashMap<String, IncrementalState>>> =
        Arc::new(Mutex::new(HashMap::new()));
}

#[derive(Debug)]
struct IncrementalState {
    model: Arc<BitNetModel>,
    config: ModelConfig,
    kv_cache: Box<dyn KVCache>,
    past_tokens: Vec<i32>,
    last_position: usize,
    device: Device,
}

impl IncrementalState {
    fn new(model_path: &str, device: Device) -> Result<Self> {
        let (config, model_tensors) = load_gguf(Path::new(model_path), device.clone())?;
        let model = Arc::new(BitNetModel::from_gguf(config.clone(), model_tensors, device.clone())?);

        let kv_cache = create_kv_cache(&config, 1, &device)?;

        Ok(IncrementalState {
            model,
            config,
            kv_cache,
            past_tokens: Vec::new(),
            last_position: 0,
            device,
        })
    }

    fn process_tokens(&mut self, tokens: &[i32], n_past: usize) -> Result<Vec<f32>> {
        // Validate input consistency
        if n_past > self.past_tokens.len() {
            return Err(Error::InvalidInput(format!(
                "n_past ({}) exceeds stored token count ({})",
                n_past, self.past_tokens.len()
            )));
        }

        // Check if we need to reset state (e.g., tokens don't match history)
        if n_past < self.past_tokens.len() {
            if tokens[..n_past] != self.past_tokens[..n_past] {
                tracing::warn!("Token history mismatch, resetting KV cache");
                self.reset_to_position(n_past)?;
            } else {
                // Truncate to requested past length
                self.truncate_to_position(n_past)?;
            }
        }

        // Process only new tokens incrementally
        let new_tokens = &tokens[self.past_tokens.len()..];
        if new_tokens.is_empty() {
            // No new tokens, return cached logits if available
            return self.get_last_logits();
        }

        tracing::debug!("Processing {} new tokens incrementally", new_tokens.len());

        let mut final_logits = None;

        for &token in new_tokens {
            // Create input tensor for single token
            let input_ids = Tensor::new(&[token], &self.device)?
                .unsqueeze(0)?; // Add batch dimension

            // Get embeddings for the current token
            let embeddings = self.model.embed(&input_ids)?;

            // Run forward pass with KV cache
            let output = self.model.forward_with_cache(
                &embeddings,
                &mut self.kv_cache,
                self.past_tokens.len(),
            )?;

            // Extract logits
            final_logits = Some(self.model.lm_head(&output)?);

            // Update state
            self.past_tokens.push(token);
            self.last_position += 1;
        }

        // Extract final logits for the last token position
        if let Some(logits_tensor) = final_logits {
            let logits_shape = logits_tensor.dims();
            let vocab_size = logits_shape[logits_shape.len() - 1];

            // Get logits for the last position
            let last_logits = if logits_shape.len() == 3 {
                // [batch, seq_len, vocab_size]
                logits_tensor.i((0, logits_shape[1] - 1))?
            } else {
                // [batch, vocab_size] - already at last position
                logits_tensor.i(0)?
            };

            let logits_vec = last_logits.to_vec1::<f32>()?;
            Ok(logits_vec)
        } else {
            Err(Error::InternalError("No logits generated".to_string()))
        }
    }

    fn reset_to_position(&mut self, position: usize) -> Result<()> {
        // Reset KV cache and recompute from scratch
        self.kv_cache = create_kv_cache(&self.config, 1, &self.device)?;
        self.past_tokens.truncate(position);
        self.last_position = position;

        if position > 0 {
            // Recompute state up to position
            let tokens_to_recompute = self.past_tokens.clone();
            self.past_tokens.clear();
            self.last_position = 0;

            // Process tokens in batches for efficiency
            const BATCH_SIZE: usize = 32;
            for chunk in tokens_to_recompute.chunks(BATCH_SIZE) {
                let input_ids = Tensor::new(chunk, &self.device)?
                    .unsqueeze(0)?;
                let embeddings = self.model.embed(&input_ids)?;
                let _output = self.model.forward_with_cache(
                    &embeddings,
                    &mut self.kv_cache,
                    self.past_tokens.len(),
                )?;

                self.past_tokens.extend_from_slice(chunk);
                self.last_position += chunk.len();
            }
        }

        Ok(())
    }

    fn truncate_to_position(&mut self, position: usize) -> Result<()> {
        if position < self.past_tokens.len() {
            self.past_tokens.truncate(position);
            self.last_position = position;
            // Note: KV cache truncation would require cache-specific implementation
            self.kv_cache.truncate_to_length(position)?;
        }
        Ok(())
    }

    fn get_last_logits(&self) -> Result<Vec<f32>> {
        // Return cached logits if available, otherwise error
        Err(Error::InvalidInput("No cached logits available".to_string()))
    }
}

pub fn eval_logits_incremental(
    model_path: &str,
    tokens: &[i32],
    n_past: usize,
) -> Result<Vec<f32>> {
    let device = Device::Cpu; // TODO: Support GPU device selection

    // Get or create incremental state for this model
    let state_key = format!("{}:{}", model_path, device);
    let mut states = INCREMENTAL_STATES.lock()
        .map_err(|_| Error::InternalError("Failed to acquire state lock".to_string()))?;

    let state = states.entry(state_key.clone()).or_insert_with(|| {
        IncrementalState::new(model_path, device.clone())
            .unwrap_or_else(|e| {
                tracing::error!("Failed to create incremental state: {}", e);
                panic!("Failed to create incremental state");
            })
    });

    // Process tokens incrementally
    state.process_tokens(tokens, n_past)
}

// Utility functions for state management
pub fn clear_incremental_state(model_path: &str) -> Result<()> {
    let device = Device::Cpu;
    let state_key = format!("{}:{}", model_path, device);

    let mut states = INCREMENTAL_STATES.lock()
        .map_err(|_| Error::InternalError("Failed to acquire state lock".to_string()))?;

    states.remove(&state_key);
    Ok(())
}

pub fn get_incremental_stats(model_path: &str) -> Option<IncrementalStats> {
    let device = Device::Cpu;
    let state_key = format!("{}:{}", model_path, device);

    let states = INCREMENTAL_STATES.lock().ok()?;
    let state = states.get(&state_key)?;

    Some(IncrementalStats {
        past_tokens_count: state.past_tokens.len(),
        current_position: state.last_position,
        cache_size_mb: state.kv_cache.memory_usage_mb(),
    })
}

#[derive(Debug)]
pub struct IncrementalStats {
    pub past_tokens_count: usize,
    pub current_position: usize,
    pub cache_size_mb: f64,
}

// Enhanced KV cache trait for incremental evaluation
pub trait KVCache: Send + Sync {
    fn update(&mut self, key: &Tensor, value: &Tensor, position: usize) -> Result<()>;
    fn get_cached_kv(&self, layer: usize) -> Result<Option<(Tensor, Tensor)>>;
    fn truncate_to_length(&mut self, length: usize) -> Result<()>;
    fn memory_usage_mb(&self) -> f64;
    fn clear(&mut self);
}

// Optimized model interface for incremental evaluation
impl BitNetModel {
    pub fn forward_with_cache(
        &self,
        input: &Tensor,
        kv_cache: &mut dyn KVCache,
        start_position: usize,
    ) -> Result<Tensor> {
        let mut hidden_states = input.clone();

        // Process through transformer layers with cache
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Get cached K,V if available
            let cached_kv = kv_cache.get_cached_kv(layer_idx)?;

            // Run attention with cache
            hidden_states = layer.forward_with_cache(
                &hidden_states,
                cached_kv,
                start_position,
            )?;

            // Update cache with new K,V
            let (new_k, new_v) = layer.get_last_kv()?;
            kv_cache.update(&new_k, &new_v, start_position + input.dim(1)? - 1)?;
        }

        Ok(hidden_states)
    }
}
```

### Alternative Approaches

1. **Session-Based State Management**: Use explicit session objects instead of global state
2. **Streaming Interface**: Implement streaming evaluation with iterator-based API
3. **Memory-Mapped Cache**: Use memory-mapped files for persistent KV cache storage

## Implementation Plan

### Phase 1: Core Incremental Logic (Priority: Critical)
- [ ] Implement proper KV cache state management
- [ ] Add incremental token processing logic
- [ ] Create state persistence and validation
- [ ] Add comprehensive unit tests

### Phase 2: Performance Optimization (Priority: High)
- [ ] Optimize KV cache memory layout and access patterns
- [ ] Add batch processing for initial token sequences
- [ ] Implement cache truncation and rollback mechanisms
- [ ] Add performance benchmarking

### Phase 3: Advanced Features (Priority: Medium)
- [ ] Support for multiple concurrent sessions
- [ ] GPU device support and memory management
- [ ] Persistent cache storage for long sessions
- [ ] Integration with beam search and sampling strategies

### Phase 4: Integration & Testing (Priority: High)
- [ ] Integration with autoregressive generation pipeline
- [ ] Cross-validation with reference implementations
- [ ] Memory usage optimization and leak detection
- [ ] Production-ready error handling and logging

## Testing Strategy

### Unit Tests
```rust
#[test]
fn test_incremental_vs_full_evaluation() {
    let model_path = "test_model.gguf";
    let tokens = vec![1, 2, 3, 4, 5];

    // Full evaluation
    let full_logits = eval_logits_once(model_path, &tokens).unwrap();

    // Incremental evaluation
    clear_incremental_state(model_path).unwrap();
    let incremental_logits = eval_logits_incremental(model_path, &tokens, 0).unwrap();

    // Results should be identical
    assert_eq!(full_logits.len(), incremental_logits.len());
    for (full, inc) in full_logits.iter().zip(incremental_logits.iter()) {
        assert!((full - inc).abs() < 1e-6, "Logits mismatch: {} vs {}", full, inc);
    }
}

#[test]
fn test_incremental_performance() {
    let model_path = "test_model.gguf";
    let base_tokens = vec![1, 2, 3, 4, 5];

    // First call
    let start = Instant::now();
    let _logits1 = eval_logits_incremental(model_path, &base_tokens, 0).unwrap();
    let first_duration = start.elapsed();

    // Incremental call
    let mut extended_tokens = base_tokens.clone();
    extended_tokens.push(6);

    let start = Instant::now();
    let _logits2 = eval_logits_incremental(model_path, &extended_tokens, base_tokens.len()).unwrap();
    let incremental_duration = start.elapsed();

    // Incremental should be much faster
    assert!(incremental_duration < first_duration / 3);
}

#[test]
fn test_state_consistency() {
    let model_path = "test_model.gguf";

    // Build sequence incrementally
    let mut tokens = vec![1];
    let mut all_logits = Vec::new();

    for i in 2..=5 {
        tokens.push(i);
        let logits = eval_logits_incremental(model_path, &tokens, tokens.len() - 1).unwrap();
        all_logits.push(logits);
    }

    // Compare with full evaluation
    let full_logits = eval_logits_once(model_path, &tokens).unwrap();
    let last_incremental = &all_logits[all_logits.len() - 1];

    assert_eq!(full_logits.len(), last_incremental.len());
    for (full, inc) in full_logits.iter().zip(last_incremental.iter()) {
        assert!((full - inc).abs() < 1e-6);
    }
}
```

### Integration Tests
```bash
# Test incremental evaluation performance
cargo test --no-default-features --features cpu test_incremental_evaluation

# Benchmark against full evaluation
cargo run -p xtask -- benchmark --component incremental_eval

# Memory usage validation
cargo test test_incremental_memory_usage --features memory-profiling
```

## Acceptance Criteria

### Functional Requirements
- [ ] Correct incremental evaluation with KV cache state preservation
- [ ] Identical results to full evaluation for equivalent token sequences
- [ ] Proper handling of sequence truncation and rollback
- [ ] Thread-safe state management for concurrent access

### Performance Requirements
- [ ] >5x speedup for single token incremental evaluation
- [ ] >10x speedup for sequences with >50% overlap
- [ ] Memory usage proportional to sequence length (not quadratic)
- [ ] Cache hit rate >95% for typical autoregressive patterns

### Quality Requirements
- [ ] 100% unit test coverage for state management logic
- [ ] Memory leak detection and validation
- [ ] Comprehensive error handling and recovery
- [ ] Clear logging and debugging capabilities

## Related Issues

- Issue #251: Production-Ready Inference Server (incremental evaluation critical)
- Autoregressive generation performance optimization
- KV cache implementation and memory management
- Parity layer API consistency and correctness

## Dependencies

- KV cache implementation with state management
- BitNet model forward pass with cache support
- Tensor operations for incremental processing
- State persistence and serialization utilities

## Migration Impact

- **API Enhancement**: Maintains existing signature while adding real functionality
- **Performance**: Significant improvement for autoregressive generation
- **Memory**: Additional memory for state management, but better overall efficiency
- **Thread Safety**: New concurrent access considerations

---

**Labels**: `critical`, `stub`, `incremental-evaluation`, `kv-cache`, `performance`, `autoregressive-generation`
**Assignee**: Core team member with transformer inference and state management experience
**Milestone**: Efficient Autoregressive Generation (v0.3.0)
**Estimated Effort**: 2-3 weeks for full implementation and comprehensive testing

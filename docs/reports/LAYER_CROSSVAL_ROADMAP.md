# Layer-by-Layer Cross-Validation Implementation Roadmap

**Status**: Planning & Architecture  
**Based On**: Comprehensive analysis of existing crossval infrastructure  
**Target**: Layer-level comparison against C++ reference  
**Scope**: Intermediate tensor capture, comparison, and receipt integration

---

## Current State Analysis

### What We Have
- Mature parity harness (logit tolerance-based, cosine similarity)
- Receipt system v1.0.0 with 8 validation gates
- Kernel recorder (thread-safe, O(1) operation)
- FFI bridge to C++ reference implementation
- Property-based testing framework
- Baseline management + historical tracking

### What We Need
- Layer boundary instrumentation in inference engine
- Intermediate tensor capture mechanism
- Layer-level comparison utilities
- Extended receipt schema for layer metadata
- Layer parity tests with property-based validation

---

## Phase 1: Design & Infrastructure (Week 1)

### 1.1 Create `LayerRecorder` Module

**File**: `crates/bitnet-inference/src/layer_recorder.rs`

```rust
use std::sync::{Arc, Mutex};
use serde::{Serialize, Deserialize};

/// Layer execution record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerExecution {
    /// Unique layer identifier (0-indexed)
    pub layer_id: usize,
    
    /// Layer type classification
    pub layer_type: LayerType,
    
    /// Shape of output tensor (batch, hidden_size)
    pub output_shape: (usize, usize),
    
    /// Output tensor checksum (for quick comparison)
    pub output_checksum: u64,
    
    /// Activation statistics for diagnostics
    pub activation_stats: ActivationStats,
    
    /// Kernel IDs executed in this layer
    pub kernel_ids: Vec<String>,
    
    /// Compute time in microseconds
    pub compute_time_us: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum LayerType {
    Embedding,
    Attention,
    MLP,
    LayerNorm,
    Projection,
    Output,
    Unknown,
}

/// Activation statistics for numerical validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationStats {
    /// Min value in activation
    pub min: f32,
    
    /// Max value in activation
    pub max: f32,
    
    /// Mean value in activation
    pub mean: f32,
    
    /// Standard deviation
    pub std_dev: f32,
    
    /// Count of NaN/Inf values (should be 0)
    pub anomalies: usize,
    
    /// L2 norm of activation
    pub l2_norm: f32,
}

/// Thread-safe layer execution recorder
pub struct LayerRecorder {
    inner: Arc<Mutex<Vec<LayerExecution>>>,
}

impl LayerRecorder {
    pub fn new() -> Self {
        Self { inner: Arc::new(Mutex::new(Vec::new())) }
    }
    
    /// Record a layer execution
    pub fn record(&self, execution: LayerExecution) {
        if let Ok(mut layers) = self.inner.lock() {
            layers.push(execution);
        }
    }
    
    /// Get execution record for specific layer
    pub fn get_layer(&self, layer_id: usize) -> Option<LayerExecution> {
        if let Ok(layers) = self.inner.lock() {
            layers.iter().find(|l| l.layer_id == layer_id).cloned()
        } else {
            None
        }
    }
    
    /// Get all recorded layers
    pub fn snapshot(&self) -> Vec<LayerExecution> {
        self.inner.lock().map(|l| l.clone()).unwrap_or_default()
    }
    
    /// Get layer count
    pub fn count(&self) -> usize {
        self.inner.lock().map(|l| l.len()).unwrap_or(0)
    }
    
    /// Clear all recordings
    pub fn clear(&self) {
        if let Ok(mut layers) = self.inner.lock() {
            layers.clear();
        }
    }
    
    /// Get total compute time across all layers
    pub fn total_compute_time_us(&self) -> u64 {
        self.snapshot().iter().map(|l| l.compute_time_us).sum()
    }
}

impl Default for LayerRecorder {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for LayerRecorder {
    fn clone(&self) -> Self {
        Self { inner: Arc::clone(&self.inner) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_layer_recorder_basic() {
        let recorder = LayerRecorder::new();
        
        let exec = LayerExecution {
            layer_id: 0,
            layer_type: LayerType::Embedding,
            output_shape: (1, 2048),
            output_checksum: 12345,
            activation_stats: ActivationStats {
                min: -1.5, max: 1.5, mean: 0.0, std_dev: 0.5,
                anomalies: 0, l2_norm: 100.0,
            },
            kernel_ids: vec!["embedding_lookup".to_string()],
            compute_time_us: 1000,
        };
        
        recorder.record(exec.clone());
        assert_eq!(recorder.count(), 1);
        
        let retrieved = recorder.get_layer(0);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().layer_id, 0);
    }
    
    #[test]
    fn test_layer_recorder_snapshot() {
        let recorder = LayerRecorder::new();
        
        for i in 0..3 {
            recorder.record(LayerExecution {
                layer_id: i,
                layer_type: LayerType::Attention,
                output_shape: (1, 2048),
                output_checksum: i as u64,
                activation_stats: ActivationStats {
                    min: 0.0, max: 1.0, mean: 0.5, std_dev: 0.1,
                    anomalies: 0, l2_norm: 50.0,
                },
                kernel_ids: vec![],
                compute_time_us: 100,
            });
        }
        
        let snapshot = recorder.snapshot();
        assert_eq!(snapshot.len(), 3);
        assert_eq!(snapshot[0].layer_id, 0);
        assert_eq!(snapshot[2].layer_id, 2);
    }
}
```

### 1.2 Extend Receipt Schema

**File**: Modify `crates/bitnet-inference/src/receipts.rs`

Add to `InferenceReceipt`:

```rust
/// Layer-level execution data (optional, v1.1.0-beta)
pub layers: Option<Vec<LayerMetadata>> = None,

pub struct LayerMetadata {
    pub layer_id: usize,
    pub layer_type: String,  // "embedding", "attention", "mlp", etc.
    pub output_shape: [usize; 2],
    pub output_checksum: u64,
    pub activation_stats: ActivationStats,
    pub kernel_ids: Vec<String>,
}

pub struct ActivationStats {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub anomalies: usize,  // NaN/Inf count
}
```

**Migration**:
- Receipt v1.0.0: backward compatible, `layers` optional
- Schema version remains "1.0.0" (layers are additive field)
- Will upgrade to v1.1.0 once layer parity tests pass

### 1.3 Extend Crossval Validation Suite

**File**: Modify `crossval/src/validation.rs`

Add new validation method:

```rust
impl ValidationSuite {
    /// Validate layer-wise activations for parity
    pub fn validate_layer_activations(
        &self,
        rust_layers: &[LayerExecution],
        cpp_layers: &[LayerExecution],
        tolerance: f32,
    ) -> Result<LayerParity> {
        // Compare layer-by-layer
        let mut results = Vec::new();
        
        for (r_layer, c_layer) in rust_layers.iter().zip(cpp_layers.iter()) {
            let similarity = cosine_similarity_f32(&r_layer, &c_layer)?;
            let stats_match = validate_activation_stats(&r_layer, &c_layer)?;
            
            results.push(LayerParityResult {
                layer_id: r_layer.layer_id,
                cosine_similarity: similarity,
                activation_stats_match: stats_match,
                passed: similarity >= 0.99 && stats_match,
            });
        }
        
        Ok(LayerParity { results })
    }
}

pub struct LayerParity {
    pub results: Vec<LayerParityResult>,
}

pub struct LayerParityResult {
    pub layer_id: usize,
    pub cosine_similarity: f32,
    pub activation_stats_match: bool,
    pub passed: bool,
}
```

---

## Phase 2: Instrumentation (Week 2-3)

### 2.1 Add Layer Hooks in Inference Engine

**File**: `crates/bitnet-inference/src/engine.rs`

Add fields to `InferenceEngine`:

```rust
pub struct InferenceEngine {
    // ... existing fields ...
    
    /// Layer execution recorder (optional)
    #[cfg(feature = "crossval")]
    pub layer_recorder: Option<LayerRecorder>,
    
    /// Current layer being executed (for diagnostics)
    current_layer_id: Option<usize>,
}

impl InferenceEngine {
    /// Begin layer execution
    fn begin_layer(&mut self, layer_id: usize, layer_type: LayerType) {
        self.current_layer_id = Some(layer_id);
        #[cfg(feature = "crossval")]
        if let Some(ref recorder) = self.layer_recorder {
            // Could optionally log layer start
        }
    }
    
    /// End layer execution and record results
    fn end_layer(&mut self, output: &[f32], kernel_ids: Vec<String>) {
        if let Some(layer_id) = self.current_layer_id {
            #[cfg(feature = "crossval")]
            if let Some(ref recorder) = self.layer_recorder {
                let stats = compute_activation_stats(output);
                let checksum = compute_checksum(output);
                
                recorder.record(LayerExecution {
                    layer_id,
                    output_shape: (1, output.len()),  // Placeholder
                    output_checksum: checksum,
                    activation_stats: stats,
                    kernel_ids,
                    compute_time_us: elapsed_us,
                    layer_type: LayerType::Unknown,  // Set by caller
                });
            }
        }
        self.current_layer_id = None;
    }
}
```

### 2.2 Instrument Forward Pass

Example for embedding layer:

```rust
// In forward_embedding()
engine.begin_layer(0, LayerType::Embedding);

// ... embedding computation ...

engine.end_layer(&embedding_output, vec!["embedding_lookup".to_string()]);
```

Example for attention layer:

```rust
// In forward_attention()
engine.begin_layer(layer_idx, LayerType::Attention);

// ... attention computation ...

engine.end_layer(&attention_output, vec![
    "q_projection".to_string(),
    "k_projection".to_string(),
    "v_projection".to_string(),
    "rope_apply".to_string(),
    "attention_gemv".to_string(),
    "attention_softmax".to_string(),
]);
```

### 2.3 Implement Helper Functions

**File**: `crates/bitnet-inference/src/layer_utils.rs` (new)

```rust
use bitnet_common::Result;

/// Compute activation statistics for a tensor
pub fn compute_activation_stats(data: &[f32]) -> ActivationStats {
    let len = data.len() as f32;
    
    let min = data.iter().copied().fold(f32::INFINITY, f32::min);
    let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean = data.iter().sum::<f32>() / len;
    
    let variance = data.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f32>() / len;
    let std_dev = variance.sqrt();
    
    let anomalies = data.iter()
        .filter(|&&x| !x.is_finite())
        .count();
    
    let l2_norm = data.iter()
        .map(|&x| x * x)
        .sum::<f32>()
        .sqrt();
    
    ActivationStats {
        min, max, mean, std_dev, anomalies, l2_norm,
    }
}

/// Compute fast checksum for tensor
pub fn compute_checksum(data: &[f32]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    for &val in data {
        val.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

/// Cosine similarity between two float tensors
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32> {
    anyhow::ensure!(a.len() == b.len(), "Tensor shape mismatch");
    
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 && norm_b == 0.0 {
        Ok(1.0)
    } else if norm_a == 0.0 || norm_b == 0.0 {
        Ok(0.0)
    } else {
        Ok(dot_product / (norm_a * norm_b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_activation_stats() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = compute_activation_stats(&data);
        
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.anomalies, 0);
    }
    
    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0];
        assert!((cosine_similarity(&a, &b).unwrap() - 1.0).abs() < 1e-6);
        
        let c = vec![0.0, 1.0];
        assert!((cosine_similarity(&a, &c).unwrap() - 0.0).abs() < 1e-6);
    }
}
```

---

## Phase 3: FFI and C++ Comparison (Week 3-4)

### 3.1 Extend FFI Bridge for Layer Outputs

**File**: Modify `crates/bitnet-sys/` and `crossval/src/cpp_bindings.rs`

Add C++ binding for layer output:

```rust
use bitnet_sys::{BitnetContext, bitnet_get_layer_output};

/// Get intermediate layer output from C++ model
pub fn get_cpp_layer_output(
    ctx: &BitnetContext,
    layer_id: usize,
) -> Result<Vec<f32>> {
    let mut output = vec![0.0f32; 4096];  // Max size
    let size = bitnet_get_layer_output(ctx, layer_id as i32, &mut output)?;
    output.truncate(size);
    Ok(output)
}
```

### 3.2 Layer Parity Test

**File**: `crossval/tests/layer_parity.rs` (new)

```rust
#[cfg(all(test, feature = "crossval"))]
mod tests {
    use bitnet_inference::layer_recorder::LayerRecorder;
    use bitnet_crossval::comparison::CrossValidator;
    
    #[test]
    fn test_layer_0_embedding_parity() -> Result<()> {
        let model_path = match test_model_path() {
            Some(p) => p,
            None => return Ok(()),  // Skip if no model
        };
        
        // Initialize
        wrapper::init_backend();
        let _guard = scopeguard::guard((), |_| wrapper::free_backend());
        
        // Load models
        let mut cpp_session = CppSession::load_deterministic(&model_path)?;
        let mut rust_engine = InferenceEngine::load(&model_path)?;
        
        // Enable layer recording
        rust_engine.layer_recorder = Some(LayerRecorder::new());
        
        // Tokenize
        let tokens = cpp_session.tokenize("Hello world")?;
        
        // Run inference
        cpp_session.eval_and_get_logits(&tokens, 0)?;
        rust_engine.forward(&tokens)?;
        
        // Get layer recordings
        let rust_layers = rust_engine.layer_recorder
            .as_ref()
            .unwrap()
            .snapshot();
        
        // Compare layer 0
        let rust_layer = rust_layers.iter().find(|l| l.layer_id == 0).ok_or(...)?;
        let cpp_layer_output = get_cpp_layer_output(&cpp_ctx, 0)?;
        
        // Validation
        let similarity = cosine_similarity(&rust_layer.output, &cpp_layer_output)?;
        assert!(similarity >= 0.99, "Layer 0 embedding parity failed");
        
        Ok(())
    }
}
```

---

## Phase 4: Testing & Validation (Week 4-5)

### 4.1 Property-Based Layer Tests

**File**: `crossval/props/test_layer_parity.py` (new)

```python
from hypothesis import given, settings, HealthCheck
import strategies

@given(strategies.layer_ids(), strategies.prompts())
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
def test_layer_output_shape_invariant(layer_id, prompt):
    """All layers must produce consistent output shapes"""
    rust_output = rust_model.get_layer_output(layer_id, prompt)
    cpp_output = cpp_model.get_layer_output(layer_id, prompt)
    
    assert rust_output.shape == cpp_output.shape, \
        f"Layer {layer_id}: shape mismatch {rust_output.shape} vs {cpp_output.shape}"


@given(strategies.layer_ids())
@settings(max_examples=10)
def test_layer_activation_sanity(layer_id):
    """Layer outputs must be finite and within reasonable range"""
    output = rust_model.get_layer_output(layer_id, "test")
    
    assert not np.any(np.isnan(output)), f"Layer {layer_id}: NaN values"
    assert not np.any(np.isinf(output)), f"Layer {layer_id}: Inf values"
    assert np.abs(output).max() < 1000, f"Layer {layer_id}: values out of range"
```

### 4.2 Receipt Integration

Add layer metadata to receipt generation:

```rust
// In xtask or inference engine
let layer_recordings = engine.layer_recorder.snapshot();
let layer_metadata: Vec<LayerMetadata> = layer_recordings.iter()
    .map(|exec| LayerMetadata {
        layer_id: exec.layer_id,
        layer_type: format!("{:?}", exec.layer_type),
        output_shape: [exec.output_shape.0, exec.output_shape.1],
        output_checksum: exec.output_checksum,
        activation_stats: exec.activation_stats.clone(),
        kernel_ids: exec.kernel_ids.clone(),
    })
    .collect();

receipt.layers = Some(layer_metadata);
```

### 4.3 Extended Validation Gates

Add to xtask verification:

```rust
// In xtask/src/gates.rs
pub fn layer_parity_gate(receipt: &InferenceReceipt) -> Result<GateResult> {
    let layers = receipt.layers.as_ref()
        .ok_or("No layer metadata in receipt")?;
    
    for layer in layers {
        // Check for anomalies
        if layer.activation_stats.anomalies > 0 {
            return Err(format!(
                "Layer {} has {} NaN/Inf values",
                layer.layer_id, layer.activation_stats.anomalies
            ));
        }
        
        // Check checksum against baseline (if available)
        // Check activation ranges are reasonable
    }
    
    Ok(GateResult { passed: true, ... })
}
```

---

## Implementation Checklist

### Phase 1 (Week 1)
- [ ] Create `layer_recorder.rs` module with `LayerRecorder` struct
- [ ] Add tests for layer recorder (basic, snapshot, threading)
- [ ] Define `LayerType` enum
- [ ] Create `ActivationStats` struct
- [ ] Extend receipt schema (v1.0.0 with optional layers field)
- [ ] Add layer validation to `ValidationSuite`

### Phase 2 (Week 2-3)
- [ ] Create `layer_utils.rs` with helper functions
- [ ] Add `compute_activation_stats()` with tests
- [ ] Add `compute_checksum()` with tests
- [ ] Add `cosine_similarity()` with tests
- [ ] Instrument embedding layer
- [ ] Instrument attention layers
- [ ] Instrument MLP layers
- [ ] Instrument output projection
- [ ] Integration tests for instrumented paths

### Phase 3 (Week 3-4)
- [ ] Extend FFI for layer output capture
- [ ] Implement `get_cpp_layer_output()`
- [ ] Create `layer_parity.rs` test suite
- [ ] Add layer-by-layer comparison tests
- [ ] FFI lifecycle tests for layer capture

### Phase 4 (Week 4-5)
- [ ] Create property-based layer tests
- [ ] Layer shape invariant tests
- [ ] Layer activation sanity tests
- [ ] Receipt generation with layer metadata
- [ ] Layer parity validation gate
- [ ] Documentation and examples

---

## Feature Flags and Configuration

```toml
# Cargo.toml entries
[features]
crossval = ["layer-recording"]
layer-recording = ["layer_recorder"]

[dev-dependencies]
hypothesis = "..."  # Python
proptest = "0.10"   # Rust property testing
```

### Environment Variables
```bash
BITNET_RECORD_LAYERS=1          # Enable layer recording
BITNET_LAYER_RECORD_PATH=...    # Path to save layer dumps
BITNET_LAYER_PARITY_TOLERANCE=0.99   # Cosine similarity threshold
```

---

## Success Criteria

1. **Infrastructure**: LayerRecorder thread-safe, O(1) record, deduplicated snapshot
2. **Instrumentation**: All layers instrumented (embedding, attention, mlp, output)
3. **Comparison**: Layer-by-layer cosine similarity ≥ 0.99 vs C++
4. **Testing**: Property-based tests pass for all layer types
5. **Integration**: Layer metadata in receipts, validation gates enforce parity
6. **CI/CD**: Receipt verification includes layer parity gate

---

## Risk Mitigation

### Memory Overhead
- Layer recorder uses Arc<Mutex> - minimal footprint
- Activation stats computed incrementally, not stored
- Feature-gated to zero overhead when disabled

### Performance Impact
- Layer instrumentation adds ~5-10% latency (acceptable for testing)
- Checksum computation is fast (SHA256 alternative if needed)
- Gated behind compile-time feature

### Compatibility
- Receipt schema backward compatible (layers field optional)
- FFI extensions don't break existing code
- Graceful fallback if C++ layer outputs unavailable

---

## Next Steps

1. **Start Phase 1** this week
2. **Leverage existing patterns** (KernelRecorder as reference)
3. **Reuse comparison utilities** (cosine similarity already implemented)
4. **Build incrementally** - test after each phase
5. **Integration early** - wire into actual inference by Phase 2

All patterns (threading, feature gates, error handling) already proven in existing crossval infrastructure.


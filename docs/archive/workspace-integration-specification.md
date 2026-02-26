# Workspace Integration Specification

**Component**: Cross-crate integration across bitnet-inference, bitnet-kernels, and bitnet-models
**Location**: Workspace-wide integration patterns
**Dependencies**: All BitNet-rs workspace crates

## Overview

This specification defines the comprehensive integration requirements across the BitNet-rs workspace to implement real neural network inference. The integration spans three primary crates (bitnet-inference, bitnet-kernels, bitnet-models) and several supporting crates, ensuring seamless quantization-aware transformer computation while maintaining clear architectural boundaries and optimal performance.

## Workspace Architecture

### Crate Dependency Graph

```
bitnet-inference (primary implementation)
    ├── bitnet-kernels (quantized operations)
    ├── bitnet-models (model loading)
    ├── bitnet-quantization (quantization algorithms)
    ├── bitnet-tokenizers (text processing)
    └── bitnet-common (shared types)

bitnet-kernels (performance layer)
    ├── bitnet-quantization (quantization integration)
    └── bitnet-common (shared types)

bitnet-models (data layer)
    ├── bitnet-quantization (weight quantization)
    └── bitnet-common (shared types)

Supporting crates:
    ├── bitnet-py (Python bindings)
    ├── bitnet-wasm (WebAssembly)
    ├── bitnet-cli (command-line tools)
    └── crossval (validation framework)
```

### Integration Patterns

```rust
// Primary integration flow: Model Loading → Quantization → Kernels → Inference
use bitnet_models::{GgufLoader, ModelLoader};
use bitnet_quantization::{DeviceAwareQuantizer, QuantizationType};
use bitnet_kernels::{KernelManager, QuantizedKernels};
use bitnet_inference::{BitNetTransformer, InferenceEngine};

// Unified workflow pattern
pub struct IntegratedInferenceWorkflow {
    loader: Box<dyn ModelLoader>,           // bitnet-models
    quantizer: DeviceAwareQuantizer,        // bitnet-quantization
    kernels: KernelManager,                 // bitnet-kernels
    transformer: BitNetTransformer,         // bitnet-inference
}
```

## Primary Changes: bitnet-inference

### Core Implementation Structure

**New Files Required:**
- `src/transformer/` - Main transformer implementation
  - `mod.rs` - Module definitions and re-exports
  - `bitnet_transformer.rs` - Core BitNetTransformer struct
  - `transformer_block.rs` - Individual transformer layers
  - `embedding.rs` - Token embedding with quantization support
  - `output_projection.rs` - LM head and tied embeddings

- `src/attention/` - Attention mechanisms
  - `mod.rs` - Attention module exports
  - `multi_head_attention.rs` - Multi-head attention implementation
  - `grouped_query_attention.rs` - GQA optimization
  - `rotary_embedding.rs` - RoPE positional encodings
  - `flash_attention.rs` - GPU-optimized attention (when available)

- `src/layers/` - Neural network layers
  - `mod.rs` - Layer module exports
  - `quantized_linear.rs` - Quantized linear layers
  - `feed_forward.rs` - Feed-forward networks
  - `layer_norm.rs` - RMS normalization
  - `activation.rs` - Activation functions (SiLU, etc.)

- `src/generation/` - Text generation
  - `mod.rs` - Generation module exports
  - `autoregressive.rs` - Autoregressive generation engine
  - `sampling.rs` - Sampling strategies (temperature, top-k, nucleus)
  - `streaming.rs` - Real-time streaming generation
  - `batch.rs` - Batch generation optimization

- `src/cache/` - KV cache management
  - `mod.rs` - Cache module exports
  - `kv_cache.rs` - Multi-layer KV cache
  - `layer_cache.rs` - Per-layer cache implementation
  - `memory_pool.rs` - Memory pool optimization

### Integration Points with Other Crates

```rust
// bitnet-inference/src/transformer/bitnet_transformer.rs
use bitnet_models::{ModelWeights, WeightLoader};
use bitnet_quantization::{QuantizedTensor, DeviceAwareQuantizer};
use bitnet_kernels::{KernelManager, QuantizedMatMul};
use bitnet_common::{BitNetConfig, Result, Device};

pub struct BitNetTransformer {
    // Model structure loaded from bitnet-models
    weights: ModelWeights,                    // From bitnet-models

    // Quantization integration from bitnet-quantization
    quantizer: DeviceAwareQuantizer,         // From bitnet-quantization

    // High-performance kernels from bitnet-kernels
    kernel_manager: KernelManager,           // From bitnet-kernels

    // Core transformer components (implemented in this crate)
    layers: Vec<TransformerBlock>,
    embeddings: QuantizedEmbedding,
    output_projection: OutputProjection,

    // Configuration and device management
    config: BitNetConfig,                    // From bitnet-common
    device: Device,                          // From bitnet-common
}

impl BitNetTransformer {
    /// Load transformer from GGUF model with integrated quantization
    pub fn from_gguf_with_quantization(
        model_path: &str,
        qtype: QuantizationType,
        device: Device
    ) -> Result<Self> {
        // Step 1: Load model weights using bitnet-models
        let weight_loader = ModelWeights::from_gguf(model_path, device)?;
        let config = weight_loader.config();

        // Step 2: Initialize quantization system from bitnet-quantization
        let quantizer = DeviceAwareQuantizer::new(qtype, device)?;

        // Step 3: Initialize high-performance kernels from bitnet-kernels
        let kernel_manager = KernelManager::new(device, qtype)?;

        // Step 4: Build transformer layers with quantized weights
        let layers = Self::build_transformer_layers(&weight_loader, &quantizer, &config)?;
        let embeddings = QuantizedEmbedding::from_weights(&weight_loader, &quantizer)?;
        let output_projection = OutputProjection::from_weights(&weight_loader, &quantizer)?;

        Ok(Self {
            weights: weight_loader,
            quantizer,
            kernel_manager,
            layers,
            embeddings,
            output_projection,
            config,
            device,
        })
    }

    /// Core forward pass integrating all components
    pub fn forward(&mut self, input_ids: &Tensor, kv_cache: Option<&mut KVCache>) -> Result<Tensor> {
        // Step 1: Token embedding (uses quantized embeddings)
        let mut hidden_states = self.embeddings.forward(input_ids)?;

        // Step 2: Transformer layers (each uses quantized operations)
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let layer_cache = kv_cache.as_mut().and_then(|cache| cache.layer_mut(layer_idx));
            hidden_states = layer.forward(&hidden_states, layer_cache)?;
        }

        // Step 3: Output projection to vocabulary logits
        let logits = self.output_projection.forward(&hidden_states)?;

        Ok(logits)
    }
}
```

### bitnet-inference Cargo.toml Updates

```toml
[package]
name = "bitnet-inference"
version = "0.1.0"
edition = "2021"

[dependencies]
# Primary workspace dependencies
bitnet-common = { path = "../bitnet-common" }
bitnet-models = { path = "../bitnet-models" }
bitnet-quantization = { path = "../bitnet-quantization" }
bitnet-kernels = { path = "../bitnet-kernels" }
bitnet-tokenizers = { path = "../bitnet-tokenizers" }

# Core dependencies
anyhow = "1.0"
thiserror = "1.0"
tracing = "0.1"
candle-core = "0.8"
candle-nn = "0.8"
tokio = { version = "1.0", features = ["rt", "rt-multi-thread"], optional = true }
rayon = "1.7"
rand = "0.8"

# Serialization
serde = { version = "1.0", features = ["derive"] }

[features]
default = []

# Core features
cpu = ["bitnet-kernels/cpu", "bitnet-quantization/cpu"]
gpu = ["bitnet-kernels/gpu", "bitnet-quantization/gpu"]

# Generation features
streaming = ["tokio"]
batch = ["rayon"]

# Development features
crossval = ["crossval-framework"]
testing = ["bitnet-common/testing"]

[dev-dependencies]
crossval-framework = { path = "../crossval", optional = true }
criterion = "0.5"
tokio-test = "0.4"
```

## Secondary Changes: bitnet-kernels

### Enhanced Kernel Support for Transformer Operations

**New Files Required:**
- `src/transformer/` - Transformer-specific kernels
  - `mod.rs` - Transformer kernel exports
  - `attention_kernels.rs` - Optimized attention computation
  - `feedforward_kernels.rs` - Feed-forward network kernels
  - `layer_norm_kernels.rs` - Layer normalization kernels

- `src/quantized/` - Quantized operation kernels
  - `mod.rs` - Quantized kernel exports
  - `matmul_i2s.rs` - I2S matrix multiplication kernels
  - `matmul_tl1.rs` - TL1 table lookup kernels (ARM NEON)
  - `matmul_tl2.rs` - TL2 table lookup kernels (x86 AVX)
  - `quantization_kernels.rs` - Runtime quantization/dequantization

### Integration with bitnet-quantization

```rust
// bitnet-kernels/src/quantized/integrated_kernels.rs
use bitnet_quantization::{QuantizedTensor, QuantizationType, DeviceAwareQuantizer};
use bitnet_common::{Result, Device};

pub struct IntegratedQuantizedKernels {
    device: Device,
    quantizer: DeviceAwareQuantizer,
    cpu_kernels: CpuKernelProvider,
    #[cfg(feature = "gpu")]
    gpu_kernels: Option<GpuKernelProvider>,
}

impl IntegratedQuantizedKernels {
    pub fn new(device: Device, quantizer: DeviceAwareQuantizer) -> Result<Self> {
        let cpu_kernels = CpuKernelProvider::new()?;

        #[cfg(feature = "gpu")]
        let gpu_kernels = if matches!(device, Device::Cuda(_)) {
            Some(GpuKernelProvider::new(device)?)
        } else {
            None
        };

        Ok(Self {
            device,
            quantizer,
            cpu_kernels,
            #[cfg(feature = "gpu")]
            gpu_kernels,
        })
    }

    /// Quantized matrix multiplication with device-aware execution
    pub fn quantized_matmul(
        &self,
        input: &Tensor,
        quantized_weights: &QuantizedTensor,
        output: &mut Tensor
    ) -> Result<()> {
        match (&self.device, &quantized_weights.qtype) {
            (Device::Cpu, QuantizationType::I2S) => {
                self.cpu_kernels.matmul_i2s(input, quantized_weights, output)
            },
            (Device::Cpu, QuantizationType::TL1) => {
                self.cpu_kernels.matmul_tl1(input, quantized_weights, output)
            },
            (Device::Cpu, QuantizationType::TL2) => {
                self.cpu_kernels.matmul_tl2(input, quantized_weights, output)
            },
            #[cfg(feature = "gpu")]
            (Device::Cuda(_), qtype) => {
                if let Some(ref gpu_kernels) = self.gpu_kernels {
                    gpu_kernels.quantized_matmul(input, quantized_weights, output, *qtype)
                } else {
                    // Fallback to CPU
                    self.cpu_kernels.quantized_matmul(input, quantized_weights, output, quantized_weights.qtype)
                }
            },
        }
    }

    /// Fused attention kernel for GPU acceleration
    #[cfg(feature = "gpu")]
    pub fn fused_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
        scale: f32
    ) -> Result<Tensor> {
        if let Some(ref gpu_kernels) = self.gpu_kernels {
            gpu_kernels.fused_attention(query, key, value, mask, scale)
        } else {
            // Fallback to standard attention computation
            self.cpu_kernels.standard_attention(query, key, value, mask, scale)
        }
    }
}
```

### bitnet-kernels Integration API

```rust
// bitnet-kernels/src/integration.rs
pub trait TransformerKernels {
    /// Multi-head attention with quantized projections
    fn multi_head_attention(
        &self,
        hidden_states: &Tensor,
        q_weights: &QuantizedTensor,
        k_weights: &QuantizedTensor,
        v_weights: &QuantizedTensor,
        o_weights: &QuantizedTensor,
        config: &AttentionConfig
    ) -> Result<Tensor>;

    /// Feed-forward network with quantized layers
    fn feed_forward_network(
        &self,
        input: &Tensor,
        gate_weights: &QuantizedTensor,
        up_weights: &QuantizedTensor,
        down_weights: &QuantizedTensor
    ) -> Result<Tensor>;

    /// Layer normalization (RMS norm)
    fn rms_layer_norm(
        &self,
        input: &Tensor,
        weight: &Tensor,
        epsilon: f32
    ) -> Result<Tensor>;
}
```

## Secondary Changes: bitnet-models

### Enhanced Model Loading for Transformer Integration

**New Files Required:**
- `src/integration/` - Integration with other crates
  - `mod.rs` - Integration module exports
  - `quantization_bridge.rs` - Bridge to bitnet-quantization
  - `kernel_bridge.rs` - Bridge to bitnet-kernels
  - `inference_bridge.rs` - Bridge to bitnet-inference

- `src/transformer/` - Transformer-specific model loading
  - `transformer_weights.rs` - Weight loading and organization
  - `attention_weights.rs` - Attention layer weight management
  - `feedforward_weights.rs` - Feed-forward layer weights
  - `embedding_weights.rs` - Embedding and output projection weights

### Quantization-Aware Model Loading

```rust
// bitnet-models/src/integration/quantization_bridge.rs
use bitnet_quantization::{QuantizedTensor, QuantizationType, DeviceAwareQuantizer};
use crate::gguf::{GgufTensor, GgufReader};
use bitnet_common::{Result, Device};

pub struct QuantizationAwareModelLoader {
    gguf_reader: GgufReader,
    quantizer: DeviceAwareQuantizer,
    device: Device,
}

impl QuantizationAwareModelLoader {
    pub fn new(model_path: &str, qtype: QuantizationType, device: Device) -> Result<Self> {
        let gguf_reader = GgufReader::from_file(model_path)?;
        let quantizer = DeviceAwareQuantizer::new(qtype, device)?;

        Ok(Self {
            gguf_reader,
            quantizer,
            device,
        })
    }

    /// Load transformer weights with automatic quantization
    pub fn load_transformer_weights(&mut self) -> Result<TransformerWeights> {
        let mut weights = TransformerWeights::new();

        // Load and quantize embedding weights
        weights.embeddings = self.load_quantized_embeddings()?;

        // Load and quantize transformer layers
        let num_layers = self.gguf_reader.get_num_layers()?;
        for layer_idx in 0..num_layers {
            let layer_weights = self.load_quantized_layer(layer_idx)?;
            weights.layers.push(layer_weights);
        }

        // Load output projection (may be tied to embeddings)
        weights.output_projection = self.load_quantized_output_projection()?;

        Ok(weights)
    }

    /// Load and quantize individual layer weights
    fn load_quantized_layer(&mut self, layer_idx: usize) -> Result<LayerWeights> {
        // Load attention weights
        let q_tensor = self.gguf_reader.get_tensor(&format!("blk.{}.attn_q.weight", layer_idx))?;
        let k_tensor = self.gguf_reader.get_tensor(&format!("blk.{}.attn_k.weight", layer_idx))?;
        let v_tensor = self.gguf_reader.get_tensor(&format!("blk.{}.attn_v.weight", layer_idx))?;
        let o_tensor = self.gguf_reader.get_tensor(&format!("blk.{}.attn_output.weight", layer_idx))?;

        // Quantize attention weights
        let q_quantized = self.quantizer.quantize_tensor(&q_tensor)?;
        let k_quantized = self.quantizer.quantize_tensor(&k_tensor)?;
        let v_quantized = self.quantizer.quantize_tensor(&v_tensor)?;
        let o_quantized = self.quantizer.quantize_tensor(&o_tensor)?;

        // Load feed-forward weights
        let gate_tensor = self.gguf_reader.get_tensor(&format!("blk.{}.ffn_gate.weight", layer_idx))?;
        let up_tensor = self.gguf_reader.get_tensor(&format!("blk.{}.ffn_up.weight", layer_idx))?;
        let down_tensor = self.gguf_reader.get_tensor(&format!("blk.{}.ffn_down.weight", layer_idx))?;

        // Quantize feed-forward weights
        let gate_quantized = self.quantizer.quantize_tensor(&gate_tensor)?;
        let up_quantized = self.quantizer.quantize_tensor(&up_tensor)?;
        let down_quantized = self.quantizer.quantize_tensor(&down_tensor)?;

        // Load normalization weights (kept in FP32)
        let attn_norm = self.gguf_reader.get_tensor(&format!("blk.{}.attn_norm.weight", layer_idx))?;
        let ffn_norm = self.gguf_reader.get_tensor(&format!("blk.{}.ffn_norm.weight", layer_idx))?;

        Ok(LayerWeights {
            attention: AttentionWeights {
                q_proj: q_quantized,
                k_proj: k_quantized,
                v_proj: v_quantized,
                o_proj: o_quantized,
            },
            feed_forward: FeedForwardWeights {
                gate_proj: gate_quantized,
                up_proj: up_quantized,
                down_proj: down_quantized,
            },
            attention_norm: attn_norm,
            ffn_norm: ffn_norm,
        })
    }
}

pub struct TransformerWeights {
    pub embeddings: QuantizedTensor,
    pub layers: Vec<LayerWeights>,
    pub output_projection: Option<QuantizedTensor>, // None if tied to embeddings
    pub final_norm: Tensor,
}

pub struct LayerWeights {
    pub attention: AttentionWeights,
    pub feed_forward: FeedForwardWeights,
    pub attention_norm: Tensor,
    pub ffn_norm: Tensor,
}

pub struct AttentionWeights {
    pub q_proj: QuantizedTensor,
    pub k_proj: QuantizedTensor,
    pub v_proj: QuantizedTensor,
    pub o_proj: QuantizedTensor,
}

pub struct FeedForwardWeights {
    pub gate_proj: QuantizedTensor,
    pub up_proj: QuantizedTensor,
    pub down_proj: QuantizedTensor,
}
```

### Memory-Mapped Loading Integration

```rust
// bitnet-models/src/integration/kernel_bridge.rs
use bitnet_kernels::{KernelManager, QuantizedKernels};
use bitnet_common::{Result, Device};

pub struct KernelIntegratedLoader {
    kernel_manager: KernelManager,
    device: Device,
}

impl KernelIntegratedLoader {
    /// Create loader with optimized kernel selection
    pub fn new(device: Device) -> Result<Self> {
        let kernel_manager = KernelManager::new(device)?;

        Ok(Self {
            kernel_manager,
            device,
        })
    }

    /// Load weights with kernel-optimized memory layout
    pub fn load_with_kernel_optimization(&mut self, weights: &mut TransformerWeights) -> Result<()> {
        // Optimize memory layout for selected kernels
        let selected_kernel = self.kernel_manager.select_best()?;

        for layer in &mut weights.layers {
            // Optimize attention weight layout
            self.optimize_attention_weights(&mut layer.attention, selected_kernel)?;

            // Optimize feed-forward weight layout
            self.optimize_feedforward_weights(&mut layer.feed_forward, selected_kernel)?;
        }

        // Optimize embedding weights
        self.optimize_embedding_weights(&mut weights.embeddings, selected_kernel)?;

        Ok(())
    }

    fn optimize_attention_weights(&self, weights: &mut AttentionWeights, kernel: &dyn QuantizedKernels) -> Result<()> {
        // Kernel-specific weight layout optimizations
        match kernel.name() {
            "avx2" => self.optimize_for_avx2(weights),
            "neon" => self.optimize_for_neon(weights),
            "cuda" => self.optimize_for_cuda(weights),
            _ => Ok(()), // No specific optimization
        }
    }
}
```

## Cross-Crate Type Definitions

### Shared Types in bitnet-common

```rust
// bitnet-common/src/integration.rs
/// Cross-crate integration types for transformer implementation

use serde::{Deserialize, Serialize};

/// Unified configuration for transformer components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerIntegrationConfig {
    /// Model architecture configuration
    pub model: ModelArchitectureConfig,

    /// Quantization settings
    pub quantization: QuantizationIntegrationConfig,

    /// Kernel optimization settings
    pub kernels: KernelIntegrationConfig,

    /// Performance tuning parameters
    pub performance: PerformanceIntegrationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArchitectureConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationIntegrationConfig {
    pub quantization_type: QuantizationType,
    pub accuracy_threshold: f32,
    pub device_aware: bool,
    pub fallback_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelIntegrationConfig {
    pub prefer_fused_kernels: bool,
    pub enable_flash_attention: bool,
    pub memory_optimization_level: OptimizationLevel,
    pub simd_level: SimdLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceIntegrationConfig {
    pub batch_size_optimization: bool,
    pub kv_cache_optimization: bool,
    pub memory_pooling: bool,
    pub prefetch_weights: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OptimizationLevel {
    Conservative,
    Balanced,
    Aggressive,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SimdLevel {
    None,
    SSE,
    AVX2,
    AVX512,
    NEON,
}
```

### Integration Traits

```rust
// bitnet-common/src/integration_traits.rs
/// Traits for cross-crate integration

pub trait WeightProvider {
    fn get_layer_weights(&self, layer_index: usize) -> Result<LayerWeights>;
    fn get_embedding_weights(&self) -> Result<&QuantizedTensor>;
    fn get_output_weights(&self) -> Result<Option<&QuantizedTensor>>;
    fn get_normalization_weights(&self, layer_index: usize) -> Result<(&Tensor, &Tensor)>;
}

pub trait QuantizationProvider {
    fn quantize_weights(&self, weights: &Tensor, qtype: QuantizationType) -> Result<QuantizedTensor>;
    fn dequantize_weights(&self, quantized: &QuantizedTensor) -> Result<Tensor>;
    fn supports_quantization_type(&self, qtype: QuantizationType) -> bool;
    fn get_optimal_quantization_type(&self, device: &Device) -> QuantizationType;
}

pub trait KernelProvider {
    fn supports_operation(&self, operation: OperationType) -> bool;
    fn get_optimal_kernel(&self, operation: OperationType, device: &Device) -> Result<Box<dyn Kernel>>;
    fn benchmark_kernels(&self) -> Result<KernelBenchmarkResults>;
}

pub trait InferenceProvider {
    fn forward_pass(&mut self, input: &Tensor, cache: Option<&mut KVCache>) -> Result<InferenceOutput>;
    fn generate_text(&mut self, prompt: &[u32], config: &GenerationConfig) -> Result<GenerationResult>;
    fn reset_state(&mut self) -> Result<()>;
    fn get_memory_usage(&self) -> MemoryUsage;
}
```

## Build System Integration

### Workspace Cargo.toml Updates

```toml
[workspace]
members = [
    "crates/bitnet-common",
    "crates/bitnet-models",
    "crates/bitnet-quantization",
    "crates/bitnet-kernels",
    "crates/bitnet-inference",   # Primary implementation
    "crates/bitnet-tokenizers",
    "crates/bitnet-py",
    "crates/bitnet-wasm",
    "crates/bitnet-cli",
    "crates/crossval",
    "xtask"
]

resolver = "2"

[workspace.dependencies]
# Ensure consistent versions across all crates
anyhow = "1.0"
thiserror = "1.0"
tracing = "0.1"
candle-core = "0.8"
candle-nn = "0.8"
serde = { version = "1.0", features = ["derive"] }
rayon = "1.7"
tokio = { version = "1.0", features = ["rt", "rt-multi-thread"] }

[workspace.metadata.docs.rs]
# Documentation generation for all crates
all-features = true
rustdoc-args = ["--cfg", "docsrs"]

[profile.release]
# Optimized release builds for inference performance
opt-level = 3
lto = "thin"
codegen-units = 1
panic = "abort"
```

### Feature Flag Coordination

```rust
// Feature flag strategy across crates
// bitnet-inference/Cargo.toml
[features]
default = []
cpu = ["bitnet-kernels/cpu", "bitnet-quantization/cpu", "bitnet-models/cpu"]
gpu = ["bitnet-kernels/gpu", "bitnet-quantization/gpu", "bitnet-models/gpu", "candle-core/cuda"]
crossval = ["crossval-framework", "bitnet-kernels/crossval", "bitnet-quantization/crossval"]

# bitnet-kernels/Cargo.toml
[features]
default = []
cpu = ["simd", "rayon"]
gpu = ["cuda-runtime", "curand", "cublas"]
crossval = ["cpp-reference-bindings"]

# bitnet-models/Cargo.toml
[features]
default = []
cpu = ["mmap-support"]
gpu = ["gpu-memory-mapping"]
```

## Integration Workflows

### End-to-End Integration Pattern

```rust
// Integration workflow example in bitnet-inference
use bitnet_models::QuantizationAwareModelLoader;
use bitnet_quantization::DeviceAwareQuantizer;
use bitnet_kernels::IntegratedQuantizedKernels;

pub struct IntegratedBitNetEngine {
    // Components from different crates working together
    model_loader: QuantizationAwareModelLoader,      // bitnet-models
    quantizer: DeviceAwareQuantizer,                 // bitnet-quantization
    kernels: IntegratedQuantizedKernels,             // bitnet-kernels
    transformer: BitNetTransformer,                  // bitnet-inference (this crate)
}

impl IntegratedBitNetEngine {
    /// Complete end-to-end initialization
    pub fn from_gguf(
        model_path: &str,
        qtype: QuantizationType,
        device: Device
    ) -> Result<Self> {
        // Step 1: Initialize quantization system
        let quantizer = DeviceAwareQuantizer::new(qtype, device)?;

        // Step 2: Initialize model loader with quantization awareness
        let mut model_loader = QuantizationAwareModelLoader::new(model_path, qtype, device)?;

        // Step 3: Initialize kernel system
        let kernels = IntegratedQuantizedKernels::new(device, quantizer.clone())?;

        // Step 4: Load and quantize model weights
        let weights = model_loader.load_transformer_weights()?;

        // Step 5: Build transformer with all components integrated
        let transformer = BitNetTransformer::new(weights, quantizer.clone(), kernels.clone(), device)?;

        Ok(Self {
            model_loader,
            quantizer,
            kernels,
            transformer,
        })
    }

    /// Complete forward pass using integrated components
    pub fn forward(&mut self, input_ids: &[u32]) -> Result<GenerationResult> {
        // Convert tokens to tensor
        let input_tensor = Tensor::new(input_ids, &self.transformer.device())?
            .unsqueeze(0)?; // Add batch dimension

        // Integrated forward pass
        let logits = self.transformer.forward(&input_tensor, None)?;

        // Sample next token using integrated sampling
        let next_token_logits = logits.narrow(1, logits.dims()[1] - 1, 1)?.squeeze(1)?;
        let next_token = self.sample_next_token(&next_token_logits)?;

        Ok(GenerationResult {
            logits,
            next_token: Some(next_token),
            performance_metrics: self.collect_metrics()?,
        })
    }

    fn collect_metrics(&self) -> Result<IntegratedPerformanceMetrics> {
        Ok(IntegratedPerformanceMetrics {
            model_loading: self.model_loader.get_metrics(),
            quantization: self.quantizer.get_metrics(),
            kernel_execution: self.kernels.get_metrics(),
            transformer_inference: self.transformer.get_metrics(),
        })
    }
}
```

### Cross-Crate Testing Integration

```rust
// Integration testing across crates
#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_end_to_end_integration() {  // AC:1, AC:8
        // Test complete pipeline: model loading → quantization → kernels → inference
        let engine = IntegratedBitNetEngine::from_gguf(
            "tests/data/test_model.gguf",
            QuantizationType::I2S,
            Device::Cpu
        ).unwrap();

        let input_tokens = vec![1, 2, 3, 4, 5];
        let result = engine.forward(&input_tokens).unwrap();

        // Verify integration worked correctly
        assert!(result.logits.dims()[2] > 0); // Has vocabulary dimension
        assert!(result.next_token.is_some());

        // Verify all components contributed
        let metrics = result.performance_metrics;
        assert!(metrics.model_loading.loading_time > Duration::from_millis(0));
        assert!(metrics.quantization.quantization_time > Duration::from_millis(0));
        assert!(metrics.kernel_execution.execution_time > Duration::from_millis(0));
    }

    #[test]
    fn test_cross_crate_error_propagation() {  // AC:10
        // Test that errors propagate correctly across crate boundaries
        let result = IntegratedBitNetEngine::from_gguf(
            "nonexistent_model.gguf",
            QuantizationType::I2S,
            Device::Cpu
        );

        assert!(result.is_err());

        // Should be a model loading error from bitnet-models crate
        match result.unwrap_err() {
            IntegrationError::ModelLoading(_) => {},
            _ => panic!("Expected ModelLoading error"),
        }
    }

    #[test]
    fn test_quantization_kernel_compatibility() {  // AC:6
        // Test that quantization and kernels work together correctly
        let quantizer = DeviceAwareQuantizer::new(QuantizationType::I2S, Device::Cpu).unwrap();
        let kernels = IntegratedQuantizedKernels::new(Device::Cpu, quantizer.clone()).unwrap();

        // Create test tensor and quantize it
        let test_tensor = Tensor::randn(0.0, 1.0, &[128, 256], &Device::Cpu).unwrap();
        let quantized = quantizer.quantize_tensor(&test_tensor).unwrap();

        // Use quantized tensor in kernel operation
        let input = Tensor::randn(0.0, 1.0, &[1, 128], &Device::Cpu).unwrap();
        let mut output = Tensor::zeros(&[1, 256], DType::F32, &Device::Cpu).unwrap();

        // This should work seamlessly across crates
        kernels.quantized_matmul(&input, &quantized, &mut output).unwrap();

        // Verify computation produced valid results
        let output_data: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
        assert!(output_data.iter().all(|&x| x.is_finite()));
    }
}
```

## Performance Optimization Across Crates

### Memory Management Integration

```rust
// Shared memory management across crates
pub struct IntegratedMemoryManager {
    model_memory: ModelMemoryManager,      // bitnet-models
    quantization_memory: QuantMemoryManager, // bitnet-quantization
    kernel_memory: KernelMemoryManager,    // bitnet-kernels
    inference_memory: InferenceMemoryManager, // bitnet-inference
}

impl IntegratedMemoryManager {
    /// Coordinate memory usage across all crates
    pub fn optimize_memory_layout(&mut self, config: &MemoryOptimizationConfig) -> Result<()> {
        // Step 1: Optimize model weight storage
        self.model_memory.optimize_weight_layout(config.weight_optimization)?;

        // Step 2: Configure quantization memory pools
        self.quantization_memory.setup_pools(config.quantization_pools)?;

        // Step 3: Allocate kernel workspaces
        self.kernel_memory.allocate_workspaces(config.kernel_workspaces)?;

        // Step 4: Setup inference caches
        self.inference_memory.setup_caches(config.cache_configuration)?;

        Ok(())
    }

    /// Get total memory usage across all components
    pub fn total_memory_usage(&self) -> MemoryUsage {
        MemoryUsage {
            model_weights: self.model_memory.usage(),
            quantization_buffers: self.quantization_memory.usage(),
            kernel_workspaces: self.kernel_memory.usage(),
            inference_caches: self.inference_memory.usage(),
        }
    }
}
```

### Performance Coordination

```rust
// Performance optimization coordination
pub struct CrossCratePerformanceOptimizer {
    target_latency: Duration,
    target_throughput: f32,
    memory_budget: usize,
}

impl CrossCratePerformanceOptimizer {
    pub fn optimize_for_target(&self, config: &mut TransformerIntegrationConfig) -> Result<()> {
        // Coordinate optimizations across crates

        // 1. Model loading optimizations (bitnet-models)
        if self.target_latency < Duration::from_millis(50) {
            config.model.prefetch_weights = true;
            config.model.memory_mapping = true;
        }

        // 2. Quantization optimizations (bitnet-quantization)
        if self.memory_budget < 4 * 1024 * 1024 * 1024 { // < 4GB
            config.quantization.quantization_type = QuantizationType::I2S; // Most memory efficient
        }

        // 3. Kernel optimizations (bitnet-kernels)
        if self.target_throughput > 20.0 {
            config.kernels.enable_flash_attention = true;
            config.kernels.prefer_fused_kernels = true;
        }

        // 4. Inference optimizations (bitnet-inference)
        if self.target_latency < Duration::from_millis(100) {
            config.performance.kv_cache_optimization = true;
            config.performance.batch_size_optimization = true;
        }

        Ok(())
    }
}
```

## Error Handling Across Crates

### Unified Error Propagation

```rust
// Cross-crate error handling
#[derive(Debug, thiserror::Error)]
pub enum IntegrationError {
    #[error("Model loading failed: {0}")]
    ModelLoading(#[from] bitnet_models::ModelError),

    #[error("Quantization failed: {0}")]
    Quantization(#[from] bitnet_quantization::QuantizationError),

    #[error("Kernel execution failed: {0}")]
    KernelExecution(#[from] bitnet_kernels::KernelError),

    #[error("Inference failed: {0}")]
    Inference(#[from] bitnet_inference::InferenceError),

    #[error("Integration configuration error: {reason}")]
    ConfigurationError { reason: String },

    #[error("Cross-crate compatibility error: {details}")]
    CompatibilityError { details: String },
}

impl IntegrationError {
    /// Determine which crate the error originated from
    pub fn source_crate(&self) -> &'static str {
        match self {
            IntegrationError::ModelLoading(_) => "bitnet-models",
            IntegrationError::Quantization(_) => "bitnet-quantization",
            IntegrationError::KernelExecution(_) => "bitnet-kernels",
            IntegrationError::Inference(_) => "bitnet-inference",
            IntegrationError::ConfigurationError { .. } => "bitnet-common",
            IntegrationError::CompatibilityError { .. } => "integration-layer",
        }
    }

    /// Get suggested fix for cross-crate errors
    pub fn suggested_fix(&self) -> &'static str {
        match self {
            IntegrationError::ModelLoading(_) => "Check model file path and format",
            IntegrationError::Quantization(_) => "Verify quantization type compatibility with device",
            IntegrationError::KernelExecution(_) => "Check device capabilities and kernel availability",
            IntegrationError::Inference(_) => "Verify input shapes and model configuration",
            IntegrationError::ConfigurationError { .. } => "Review integration configuration",
            IntegrationError::CompatibilityError { .. } => "Check crate version compatibility",
        }
    }
}
```

This comprehensive workspace integration specification ensures seamless collaboration between all BitNet-rs crates while implementing real neural network inference with quantized transformers, maintaining clear architectural boundaries, and enabling optimal performance across CPU and GPU devices.

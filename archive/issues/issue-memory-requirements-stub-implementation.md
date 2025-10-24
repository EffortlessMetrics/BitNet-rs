# [Memory] Replace hardcoded memory requirements with dynamic calculation

## Problem Description

The `get_memory_requirements` function in `crates/bitnet-models/src/production_loader.rs` uses hardcoded values instead of analyzing the actual model file to calculate precise memory needs. This prevents accurate memory planning for different model sizes and architectures.

## Environment
- **Affected File**: `crates/bitnet-models/src/production_loader.rs`
- **Function**: `get_memory_requirements`
- **Impact**: Memory allocation accuracy, deployment planning, OOM prevention
- **Severity**: High (production readiness)

## Current Implementation

```rust
pub fn get_memory_requirements(&self, device: &str) -> MemoryRequirements {
    // This is a simplified implementation
    // In reality, this would analyze the model file and calculate precise memory needs

    let base_memory = 1000; // Base memory in MB

    match device {
        "cpu" => MemoryRequirements {
            total_mb: base_memory,
            gpu_memory_mb: None,
            cpu_memory_mb: base_memory - 200,
            kv_cache_mb: 100,
            activation_mb: 50,
            headroom_mb: 50,
        },
        "gpu" => MemoryRequirements {
            total_mb: base_memory,
            gpu_memory_mb: Some(800),
            cpu_memory_mb: 200,
            kv_cache_mb: 100,
            activation_mb: 50,
            headroom_mb: 50,
        },
        _ => MemoryRequirements {
            total_mb: base_memory,
            gpu_memory_mb: None,
            cpu_memory_mb: base_memory,
            kv_cache_mb: 0,
            activation_mb: 0,
            headroom_mb: 0,
        },
    }
}
```

## Root Cause Analysis

1. **Development Placeholder**: Function was created as a stub during initial development
2. **Hardcoded Values**: Uses fixed 1000MB base memory regardless of actual model size
3. **Missing Model Analysis**: Doesn't inspect model file structure or configuration
4. **Device-Agnostic**: Doesn't account for device-specific memory constraints
5. **No Quantization Awareness**: Ignores quantization impact on memory usage

## Impact Assessment

- **Production Risk**: Inaccurate memory estimates can lead to OOM errors or resource waste
- **Deployment Issues**: Cannot plan hardware requirements for different models
- **Performance Impact**: May allocate insufficient memory for KV cache and activations
- **Model Support**: Cannot properly support models of varying sizes

## Proposed Solution

Replace hardcoded values with dynamic analysis of model files to calculate precise memory requirements based on:
- Model architecture parameters
- Quantization method and precision
- Device capabilities and constraints
- Runtime configuration

### Implementation Plan

#### 1. Model Analysis Infrastructure

```rust
#[derive(Debug, Clone)]
pub struct ModelMemoryAnalyzer {
    config: ModelConfig,
    quantization_info: Option<QuantizationInfo>,
    device_constraints: DeviceConstraints,
}

impl ModelMemoryAnalyzer {
    pub fn new(model_path: &Path, device: &str) -> Result<Self> {
        let config = Self::load_model_config(model_path)?;
        let quantization_info = Self::detect_quantization(model_path)?;
        let device_constraints = DeviceConstraints::for_device(device)?;

        Ok(Self {
            config,
            quantization_info,
            device_constraints,
        })
    }

    fn load_model_config(model_path: &Path) -> Result<ModelConfig> {
        // Parse GGUF metadata or SafeTensors header
        if model_path.extension() == Some(OsStr::new("gguf")) {
            Self::parse_gguf_config(model_path)
        } else if model_path.extension() == Some(OsStr::new("safetensors")) {
            Self::parse_safetensors_config(model_path)
        } else {
            Err(anyhow::anyhow!("Unsupported model format"))
        }
    }

    fn parse_gguf_config(model_path: &Path) -> Result<ModelConfig> {
        let file = File::open(model_path)?;
        let mut reader = GgufReader::new(file)?;

        let metadata = reader.metadata()?;

        Ok(ModelConfig {
            vocab_size: metadata.get_u32("tokenizer.ggml.tokens")?,
            hidden_size: metadata.get_u32("llama.embedding_length")?,
            num_layers: metadata.get_u32("llama.block_count")?,
            num_heads: metadata.get_u32("llama.attention.head_count")?,
            num_key_value_heads: metadata.get_u32("llama.attention.head_count_kv").unwrap_or(0),
            max_position_embeddings: metadata.get_u32("llama.context_length")?,
            rope_theta: metadata.get_f32("llama.rope.freq_base").unwrap_or(10000.0),
        })
    }

    fn detect_quantization(model_path: &Path) -> Result<Option<QuantizationInfo>> {
        // Analyze model tensors to determine quantization method
        let file = File::open(model_path)?;
        let mut reader = GgufReader::new(file)?;

        let tensor_infos = reader.tensor_infos()?;

        // Check for quantized tensor types
        let mut quantization_methods = HashSet::new();
        for tensor_info in tensor_infos {
            match tensor_info.ggml_dtype {
                GgmlDType::Q4_0 | GgmlDType::Q4_1 => {
                    quantization_methods.insert(QuantizationMethod::Q4);
                },
                GgmlDType::Q8_0 => {
                    quantization_methods.insert(QuantizationMethod::Q8);
                },
                GgmlDType::F16 => {
                    quantization_methods.insert(QuantizationMethod::FP16);
                },
                // Add I2S detection logic
                _ if Self::is_i2s_tensor(&tensor_info) => {
                    quantization_methods.insert(QuantizationMethod::I2S);
                },
                _ => {}
            }
        }

        if quantization_methods.is_empty() {
            Ok(None)
        } else {
            Ok(Some(QuantizationInfo {
                primary_method: quantization_methods.iter().next().unwrap().clone(),
                mixed_precision: quantization_methods.len() > 1,
                compression_ratio: Self::estimate_compression_ratio(&quantization_methods),
            }))
        }
    }
}
```

#### 2. Dynamic Memory Calculation

```rust
impl ModelMemoryAnalyzer {
    pub fn calculate_memory_requirements(&self, device: &str, config: &InferenceConfig) -> Result<MemoryRequirements> {
        // Calculate base model memory
        let model_memory = self.calculate_model_memory()?;

        // Calculate KV cache memory
        let kv_cache_memory = self.calculate_kv_cache_memory(config)?;

        // Calculate activation memory
        let activation_memory = self.calculate_activation_memory(config)?;

        // Add safety headroom
        let headroom = self.calculate_headroom(&model_memory, &kv_cache_memory, &activation_memory);

        match device {
            "cpu" => Ok(MemoryRequirements {
                total_mb: model_memory.cpu + kv_cache_memory + activation_memory + headroom,
                gpu_memory_mb: None,
                cpu_memory_mb: model_memory.cpu + kv_cache_memory + activation_memory,
                kv_cache_mb: kv_cache_memory,
                activation_mb: activation_memory,
                headroom_mb: headroom,
            }),
            "gpu" => Ok(MemoryRequirements {
                total_mb: model_memory.gpu + kv_cache_memory + activation_memory + headroom,
                gpu_memory_mb: Some(model_memory.gpu + kv_cache_memory + activation_memory),
                cpu_memory_mb: model_memory.cpu_overhead,
                kv_cache_mb: kv_cache_memory,
                activation_mb: activation_memory,
                headroom_mb: headroom,
            }),
            "auto" => {
                // Calculate both and return the more suitable option
                let cpu_reqs = self.calculate_memory_requirements("cpu", config)?;
                let gpu_reqs = self.calculate_memory_requirements("gpu", config)?;

                // Choose based on available memory and performance requirements
                if self.device_constraints.gpu_memory >= gpu_reqs.gpu_memory_mb.unwrap_or(0) as u64 * 1024 * 1024 {
                    Ok(gpu_reqs)
                } else {
                    Ok(cpu_reqs)
                }
            },
            _ => Err(anyhow::anyhow!("Unsupported device type: {}", device))
        }
    }

    fn calculate_model_memory(&self) -> Result<ModelMemory> {
        let param_count = self.estimate_parameter_count();
        let precision_bytes = self.get_precision_bytes();

        let base_model_size = param_count * precision_bytes;

        // Apply quantization compression if present
        let compressed_size = if let Some(quant_info) = &self.quantization_info {
            (base_model_size as f32 * quant_info.compression_ratio) as usize
        } else {
            base_model_size
        };

        Ok(ModelMemory {
            cpu: compressed_size / (1024 * 1024), // Convert to MB
            gpu: compressed_size / (1024 * 1024),
            cpu_overhead: 200, // CPU overhead for GPU inference
        })
    }

    fn calculate_kv_cache_memory(&self, config: &InferenceConfig) -> Result<usize> {
        let hidden_size = self.config.hidden_size;
        let num_layers = self.config.num_layers;
        let max_sequence_length = config.max_length.unwrap_or(self.config.max_position_embeddings);
        let batch_size = config.batch_size.unwrap_or(1);

        // Key and value caches for each layer
        let kv_size_per_token = 2 * hidden_size * num_layers; // 2 for key+value
        let precision_bytes = if config.use_fp16 { 2 } else { 4 };

        let total_kv_size = kv_size_per_token * max_sequence_length * batch_size * precision_bytes;

        Ok(total_kv_size / (1024 * 1024)) // Convert to MB
    }

    fn calculate_activation_memory(&self, config: &InferenceConfig) -> Result<usize> {
        let hidden_size = self.config.hidden_size;
        let intermediate_size = hidden_size * 4; // Typical feed-forward expansion
        let batch_size = config.batch_size.unwrap_or(1);
        let sequence_length = config.max_length.unwrap_or(512).min(1024); // Reasonable activation window

        // Estimate activations for attention + feed-forward
        let attention_activations = hidden_size * sequence_length * batch_size;
        let ff_activations = intermediate_size * sequence_length * batch_size;
        let total_activations = (attention_activations + ff_activations) * 4; // 4 bytes for F32

        Ok(total_activations / (1024 * 1024)) // Convert to MB
    }

    fn estimate_parameter_count(&self) -> usize {
        let hidden_size = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;
        let num_layers = self.config.num_layers;
        let intermediate_size = hidden_size * 4; // Typical expansion ratio

        // Embedding parameters
        let embedding_params = vocab_size * hidden_size;

        // Transformer layer parameters (per layer)
        let attention_params = 4 * hidden_size * hidden_size; // Q, K, V, O projections
        let ff_params = 2 * hidden_size * intermediate_size; // Up and down projections
        let layer_norm_params = 2 * hidden_size; // Pre and post layer norms
        let params_per_layer = attention_params + ff_params + layer_norm_params;

        // Total parameters
        embedding_params + (params_per_layer * num_layers)
    }

    fn get_precision_bytes(&self) -> usize {
        if let Some(quant_info) = &self.quantization_info {
            match quant_info.primary_method {
                QuantizationMethod::I2S => 1, // Approximately 1 byte per parameter
                QuantizationMethod::Q4 => 1, // 4-bit quantization
                QuantizationMethod::Q8 => 1, // 8-bit quantization
                QuantizationMethod::FP16 => 2, // 16-bit float
                _ => 4, // Default to F32
            }
        } else {
            4 // F32 default
        }
    }

    fn calculate_headroom(&self, model_memory: &ModelMemory, kv_cache_mb: &usize, activation_mb: &usize) -> usize {
        let total_memory = model_memory.cpu + kv_cache_mb + activation_mb;

        // 10% headroom minimum, 20% for large models
        if total_memory > 8000 { // > 8GB
            total_memory / 5 // 20%
        } else {
            total_memory / 10 // 10%
        }
    }
}
```

## Testing Strategy

- **Unit Tests**: Test memory calculation accuracy for different model sizes
- **Integration Tests**: Validate against actual model loading and memory usage
- **Performance Tests**: Ensure calculation speed doesn't impact startup time
- **Cross-Platform Tests**: Verify calculations across CPU and GPU devices
- **Accuracy Tests**: Compare calculated vs actual memory usage during inference

## Implementation Tasks

### Phase 1: Infrastructure
- [ ] Implement ModelMemoryAnalyzer with GGUF parsing
- [ ] Add SafeTensors support for memory analysis
- [ ] Implement device constraint detection
- [ ] Add quantization method detection

### Phase 2: Calculation Logic
- [ ] Implement parameter count estimation
- [ ] Add KV cache memory calculation
- [ ] Implement activation memory estimation
- [ ] Add device-specific memory allocation logic

### Phase 3: Integration
- [ ] Update ProductionModelLoader to use analyzer
- [ ] Add configuration integration
- [ ] Implement error handling and validation
- [ ] Add comprehensive logging

### Phase 4: Testing & Validation
- [ ] Create test cases for different model architectures
- [ ] Validate against actual memory usage
- [ ] Add performance benchmarks
- [ ] Test cross-platform compatibility

## Acceptance Criteria

- [ ] Memory requirements calculated dynamically from model files
- [ ] Supports GGUF and SafeTensors model formats
- [ ] Accounts for quantization impact on memory usage
- [ ] Provides device-specific memory breakdowns
- [ ] Includes appropriate safety headroom
- [ ] Calculation completes in <500ms for typical models
- [ ] Accuracy within 10% of actual memory usage
- [ ] Comprehensive error handling for unsupported formats

## Performance Targets

- **Calculation Speed**: <500ms for model analysis
- **Memory Accuracy**: Within 10% of actual usage
- **Startup Impact**: <100ms additional startup time
- **Memory Overhead**: <50MB for analysis metadata

## Dependencies

- GGUF file format parsing libraries
- SafeTensors metadata reading
- System memory detection utilities
- GPU information libraries (for GPU device analysis)

## Labels

- `memory-management`
- `model-loading`
- `production-readiness`
- `performance`
- `priority-high`

## Related Issues

- GPU memory management implementation
- Model loading optimization
- Production deployment requirements
- System requirements validation

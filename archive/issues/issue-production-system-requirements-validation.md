# [Production Engine] Implement Comprehensive System Requirements Validation

## Problem Description

The `ProductionInferenceEngine::validate_system_requirements` function in `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/production_engine.rs:516` contains only placeholder comments and minimal validation logic. This critical production-ready feature should perform comprehensive system validation including memory checks, device capabilities assessment, operation testing, and model compatibility verification to ensure reliable inference operation.

## Environment

- **File**: `crates/bitnet-inference/src/production_engine.rs`
- **Function**: `ProductionInferenceEngine::validate_system_requirements` (line 516)
- **MSRV**: Rust 1.90.0
- **Feature flags**: Both `cpu` and `gpu` features require validation
- **Dependencies**: `DeviceManager`, `BitNetConfig`, potential new system info crates

## Current Implementation Analysis

### Existing Code
```rust
pub fn validate_system_requirements(&self) -> Result<()> {
    // In a real implementation, this would:
    // 1. Check available memory
    // 2. Validate device capabilities
    // 3. Test basic operations
    // 4. Verify model compatibility

    self.device_manager.validate_device_compatibility(1024 * 1024 * 1024)?; // 1GB requirement
    Ok(())
}
```

### Current DeviceManager Validation
```rust
pub fn validate_device_compatibility(&self, _required_memory: u64) -> Result<()> {
    // Device validation logic would go here
    Ok(())
}
```

### Gap Analysis

**Missing Validations:**
1. **System Memory Assessment**: No actual memory checking beyond passing a hardcoded value
2. **Device-Specific Capabilities**: CUDA compute capability, AVX/SIMD support verification
3. **Operational Testing**: No validation that basic tensor operations work
4. **Model-Specific Requirements**: No verification against actual model needs
5. **Environment Configuration**: No checking of environment variables or system limits

## Root Cause Analysis

1. **Placeholder Development**: Function was stubbed during initial development
2. **Production Readiness Gap**: Critical infrastructure missing for production deployment
3. **Missing Dependencies**: No system information libraries integrated
4. **Incomplete DeviceManager**: Underlying device validation also stubbed

## Impact Assessment

### Severity: High
### Affected Components: Production inference deployment, system reliability

**Production Impact:**
- Inference engines may start but fail during operation due to insufficient resources
- Silent failures in CUDA/compute capability mismatches
- Memory exhaustion crashes without early detection
- Inconsistent behavior across different deployment environments

**Operational Impact:**
- Difficult to diagnose deployment issues
- No early warning for resource constraints
- Manual validation required for each deployment
- Increased support burden for runtime failures

## Proposed Solution

### Primary Approach: Comprehensive System Validation Framework

Implement a robust, multi-layered validation system that covers all critical requirements for reliable inference operation.

#### Implementation Plan

**1. Enhanced System Memory Validation**

```rust
use sysinfo::{System, SystemExt, MemoryRefreshKind, RefreshKind};

impl ProductionInferenceEngine {
    pub fn validate_system_requirements(&self) -> Result<()> {
        info!("Validating system requirements for production inference");

        // 1. Comprehensive memory validation
        self.validate_memory_requirements()?;

        // 2. Device-specific capability validation
        self.validate_device_capabilities()?;

        // 3. Basic operation testing
        self.validate_basic_operations().await?;

        // 4. Model-specific compatibility
        self.validate_model_compatibility()?;

        // 5. Environment configuration validation
        self.validate_environment_configuration()?;

        info!("System requirements validation completed successfully");
        Ok(())
    }

    fn validate_memory_requirements(&self) -> Result<()> {
        let mut system = System::new_with_specifics(
            RefreshKind::new().with_memory(MemoryRefreshKind::everything())
        );
        system.refresh_memory();

        let available_memory = system.available_memory();
        let total_memory = system.total_memory();

        // Calculate model-specific memory requirements
        let model_memory = self.calculate_model_memory_requirements()?;
        let cache_memory = self.calculate_cache_memory_requirements()?;
        let overhead_memory = self.calculate_system_overhead();

        let total_required = model_memory + cache_memory + overhead_memory;

        // Validate available memory with safety margin
        let safety_margin = 0.1; // 10% safety margin
        let required_with_margin = (total_required as f64 * (1.0 + safety_margin)) as u64;

        if available_memory < required_with_margin {
            return Err(anyhow::anyhow!(
                "Insufficient system memory: {:.2}GB available, {:.2}GB required (including {:.0}% safety margin)",
                available_memory as f64 / (1024.0 * 1024.0 * 1024.0),
                required_with_margin as f64 / (1024.0 * 1024.0 * 1024.0),
                safety_margin * 100.0
            ));
        }

        info!(
            "Memory validation passed: {:.2}GB available, {:.2}GB required",
            available_memory as f64 / (1024.0 * 1024.0 * 1024.0),
            total_required as f64 / (1024.0 * 1024.0 * 1024.0)
        );

        Ok(())
    }

    fn calculate_model_memory_requirements(&self) -> Result<u64> {
        let config = &self.model.config();

        // Calculate based on model architecture
        let vocab_size = config.model.vocab_size as u64;
        let hidden_size = config.model.hidden_size as u64;
        let num_layers = config.model.num_layers as u64;
        let num_heads = config.model.num_heads as u64;
        let intermediate_size = config.model.intermediate_size as u64;

        // Estimate memory for different quantization schemes
        let bytes_per_param = match config.quantization.method {
            QuantizationMethod::I2S => 0.25, // ~2 bits per parameter with overhead
            QuantizationMethod::TL1 => 0.125, // ~1 bit per parameter
            QuantizationMethod::TL2 => 0.125, // ~1 bit per parameter
            QuantizationMethod::None => 4.0, // FP32
        };

        // Rough parameter count estimation
        let embedding_params = vocab_size * hidden_size;
        let attention_params = num_layers * num_heads * hidden_size * hidden_size * 4; // Q, K, V, O
        let ffn_params = num_layers * hidden_size * intermediate_size * 2; // up and down projections
        let layer_norm_params = num_layers * hidden_size * 2; // attention and FFN layer norms

        let total_params = embedding_params + attention_params + ffn_params + layer_norm_params;
        let model_memory = (total_params as f64 * bytes_per_param) as u64;

        // Add activation memory (context-dependent)
        let max_context = config.model.max_position_embeddings as u64;
        let activation_memory = max_context * hidden_size * 4; // FP32 activations

        Ok(model_memory + activation_memory)
    }

    fn calculate_cache_memory_requirements(&self) -> Result<u64> {
        let config = &self.model.config();
        let max_context = config.model.max_position_embeddings as u64;
        let hidden_size = config.model.hidden_size as u64;
        let num_layers = config.model.num_layers as u64;
        let num_kv_heads = config.model.num_key_value_heads as u64;

        // KV cache memory: [batch_size, num_heads, seq_len, head_dim]
        let head_dim = hidden_size / config.model.num_heads as u64;
        let batch_size = 1; // Conservative estimate

        // Assume FP16 for KV cache
        let kv_cache_memory = batch_size * num_kv_heads * max_context * head_dim * 2 * 2; // K and V, FP16
        let layer_cache_memory = kv_cache_memory * num_layers;

        Ok(layer_cache_memory)
    }

    fn calculate_system_overhead(&self) -> u64 {
        // Conservative estimate for system overhead
        512 * 1024 * 1024 // 512MB for runtime overhead
    }
}
```

**2. Device Capability Validation**

```rust
impl ProductionInferenceEngine {
    fn validate_device_capabilities(&self) -> Result<()> {
        match self.device {
            Device::Cpu => self.validate_cpu_capabilities(),
            Device::Gpu(_) => self.validate_gpu_capabilities(),
            Device::Auto => {
                // Validate both, preferring GPU if available
                match self.validate_gpu_capabilities() {
                    Ok(_) => Ok(()),
                    Err(_) => self.validate_cpu_capabilities(),
                }
            }
        }
    }

    fn validate_cpu_capabilities(&self) -> Result<()> {
        info!("Validating CPU capabilities");

        // Check SIMD support
        #[cfg(target_arch = "x86_64")]
        {
            if !is_x86_feature_detected!("avx2") {
                warn!("AVX2 not detected, falling back to slower CPU operations");
            }
            if is_x86_feature_detected!("avx512f") {
                info!("AVX-512 detected, enabling optimized kernels");
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // Check for NEON support (standard on ARM64)
            info!("ARM64 NEON support available");
        }

        // Validate thread count
        let num_threads = rayon::current_num_threads();
        if num_threads < 2 {
            warn!("Only {} thread available, performance may be limited", num_threads);
        }

        Ok(())
    }

    #[cfg(feature = "gpu")]
    fn validate_gpu_capabilities(&self) -> Result<()> {
        info!("Validating GPU capabilities");

        // Check CUDA availability
        let device_count = cudarc::driver::result::device::get_device_count()
            .map_err(|e| anyhow::anyhow!("Failed to get CUDA device count: {}", e))?;

        if device_count == 0 {
            return Err(anyhow::anyhow!("No CUDA devices found"));
        }

        // Check compute capability
        let device = cudarc::driver::CudaDevice::new(0)
            .map_err(|e| anyhow::anyhow!("Failed to initialize CUDA device: {}", e))?;

        let (major, minor) = device.compute_capability()
            .map_err(|e| anyhow::anyhow!("Failed to get compute capability: {}", e))?;

        // Require at least compute capability 6.0 for efficient FP16
        if major < 6 {
            return Err(anyhow::anyhow!(
                "Insufficient CUDA compute capability: {}.{} found, 6.0+ required",
                major, minor
            ));
        }

        // Check memory
        let memory_info = device.memory_info()
            .map_err(|e| anyhow::anyhow!("Failed to get GPU memory info: {}", e))?;

        info!(
            "GPU validation passed: Compute {}.{}, {:.2}GB memory available",
            major, minor,
            memory_info.free as f64 / (1024.0 * 1024.0 * 1024.0)
        );

        Ok(())
    }

    #[cfg(not(feature = "gpu"))]
    fn validate_gpu_capabilities(&self) -> Result<()> {
        Err(anyhow::anyhow!("GPU feature not enabled in build"))
    }
}
```

**3. Basic Operations Testing**

```rust
impl ProductionInferenceEngine {
    async fn validate_basic_operations(&self) -> Result<()> {
        info!("Running basic operation validation tests");

        // Test tensor creation and basic operations
        self.test_tensor_operations()?;

        // Test quantization/dequantization
        self.test_quantization_operations()?;

        // Test simple forward pass
        self.test_simple_forward_pass().await?;

        info!("Basic operation validation completed successfully");
        Ok(())
    }

    fn test_tensor_operations(&self) -> Result<()> {
        // Create test tensors
        let test_shape = vec![32, 64];
        let tensor1 = ConcreteTensor::zeros(&test_shape)?;
        let tensor2 = ConcreteTensor::ones(&test_shape)?;

        // Test basic arithmetic
        let _result = tensor1.add(&tensor2)?;

        Ok(())
    }

    fn test_quantization_operations(&self) -> Result<()> {
        let config = &self.model.config();

        match config.quantization.method {
            QuantizationMethod::I2S => {
                // Test I2S quantization round-trip
                let test_data = vec![0.5, -0.3, 1.2, -0.8];
                let quantized = i2s_quantize(&test_data)?;
                let _dequantized = i2s_dequantize(&quantized)?;
            }
            QuantizationMethod::TL1 | QuantizationMethod::TL2 => {
                // Test table lookup quantization
                let test_tensor = ConcreteTensor::from_vec(vec![0.1, 0.5, -0.3, 0.9], &[2, 2])?;
                let _quantized = tl_quantize(&test_tensor, config.quantization.method)?;
            }
            QuantizationMethod::None => {
                // No quantization testing needed
            }
        }

        Ok(())
    }

    async fn test_simple_forward_pass(&self) -> Result<()> {
        // Create minimal test input
        let test_tokens = vec![1u32]; // BOS token
        let input_tensor = self.model.embed(&test_tokens)?;

        // Test forward pass with minimal cache
        let mut test_cache = self.create_empty_cache()?;
        let _output = self.model.forward(&input_tensor, &mut test_cache)?;

        Ok(())
    }
}
```

**4. Model Compatibility Validation**

```rust
impl ProductionInferenceEngine {
    fn validate_model_compatibility(&self) -> Result<()> {
        info!("Validating model compatibility");

        let config = &self.model.config();

        // Validate context length
        let max_context = config.model.max_position_embeddings;
        if max_context > 32768 {
            warn!("Large context length ({}), ensure sufficient memory", max_context);
        }

        // Validate vocabulary size
        let vocab_size = config.model.vocab_size;
        if vocab_size > 100000 {
            warn!("Large vocabulary ({}), performance may be impacted", vocab_size);
        }

        // Validate quantization compatibility
        match config.quantization.method {
            QuantizationMethod::I2S => {
                if !self.supports_i2s_quantization()? {
                    return Err(anyhow::anyhow!("I2S quantization not supported on this device"));
                }
            }
            QuantizationMethod::TL1 | QuantizationMethod::TL2 => {
                if !self.supports_table_lookup_quantization()? {
                    return Err(anyhow::anyhow!("Table lookup quantization not supported"));
                }
            }
            QuantizationMethod::None => {}
        }

        // Validate device-specific constraints
        match self.device {
            Device::Gpu(_) => {
                if config.model.hidden_size % 64 != 0 {
                    warn!("Hidden size not aligned to 64, GPU performance may be suboptimal");
                }
            }
            Device::Cpu => {
                if config.model.hidden_size > 8192 {
                    warn!("Large hidden size on CPU, consider GPU acceleration");
                }
            }
            Device::Auto => {}
        }

        Ok(())
    }

    fn supports_i2s_quantization(&self) -> Result<bool> {
        // Check for I2S-specific requirements
        Ok(true) // Placeholder - implement based on device capabilities
    }

    fn supports_table_lookup_quantization(&self) -> Result<bool> {
        // Check for TL quantization requirements
        Ok(true) // Placeholder - implement based on device capabilities
    }
}
```

**5. Environment Configuration Validation**

```rust
impl ProductionInferenceEngine {
    fn validate_environment_configuration(&self) -> Result<()> {
        info!("Validating environment configuration");

        // Check critical environment variables
        self.validate_environment_variables()?;

        // Validate file system permissions
        self.validate_file_system_access()?;

        // Check system limits
        self.validate_system_limits()?;

        Ok(())
    }

    fn validate_environment_variables(&self) -> Result<()> {
        // Check for deterministic execution settings
        if env::var("BITNET_DETERMINISTIC").is_ok() {
            info!("Deterministic execution mode enabled");
            if env::var("BITNET_SEED").is_err() {
                warn!("BITNET_DETERMINISTIC set but BITNET_SEED not provided");
            }
        }

        // Validate thread settings
        if let Ok(threads) = env::var("RAYON_NUM_THREADS") {
            match threads.parse::<usize>() {
                Ok(n) if n > 0 => info!("Thread count limited to {} by environment", n),
                _ => warn!("Invalid RAYON_NUM_THREADS value: {}", threads),
            }
        }

        // Check memory limits
        if let Ok(limit) = env::var("BITNET_MEMORY_LIMIT") {
            match parse_memory_size(&limit) {
                Ok(bytes) => info!("Memory limit set to {:.2}GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0)),
                Err(_) => warn!("Invalid BITNET_MEMORY_LIMIT format: {}", limit),
            }
        }

        Ok(())
    }

    fn validate_file_system_access(&self) -> Result<()> {
        // Validate model file access
        if let Some(model_path) = &self.model.config().model.path {
            if !model_path.exists() {
                return Err(anyhow::anyhow!("Model file not found: {}", model_path.display()));
            }

            // Check read permissions
            std::fs::File::open(model_path)
                .with_context(|| format!("Cannot read model file: {}", model_path.display()))?;
        }

        // Check cache directory (if configured)
        // This would check where KV cache or temporary files are stored

        Ok(())
    }

    fn validate_system_limits(&self) -> Result<()> {
        // Check file descriptor limits
        // Check memory limits (ulimit -m)
        // Check process limits
        // Platform-specific validation

        Ok(())
    }
}

fn parse_memory_size(size_str: &str) -> Result<u64> {
    // Parse sizes like "4GB", "512MB", "1024"
    // Implementation details...
    Ok(0) // Placeholder
}
```

### Alternative Solutions Considered

1. **Gradual validation**: Implement basic checks first, enhance later
2. **External validation tool**: Separate binary for system validation
3. **Runtime validation**: Check requirements during inference rather than startup

## Implementation Breakdown

### Phase 1: Core Infrastructure (Week 1)
- [ ] Add `sysinfo` dependency for system information
- [ ] Implement basic memory validation with model-specific calculations
- [ ] Update `DeviceManager::validate_device_compatibility` with real logic
- [ ] Add comprehensive error types for validation failures

### Phase 2: Device-Specific Validation (Week 1)
- [ ] Implement CPU capability checking (SIMD support)
- [ ] Add GPU capability validation (CUDA, compute capability)
- [ ] Create device-specific memory requirement calculations
- [ ] Add performance warnings for suboptimal configurations

### Phase 3: Operation Testing (Week 2)
- [ ] Implement basic tensor operation testing
- [ ] Add quantization method validation and testing
- [ ] Create minimal forward pass validation
- [ ] Add error recovery and fallback mechanisms

### Phase 4: Model Compatibility (Week 2)
- [ ] Implement model-specific requirement validation
- [ ] Add quantization method compatibility checks
- [ ] Create device-specific optimization warnings
- [ ] Add context length and vocabulary size validation

### Phase 5: Environment Validation (Week 3)
- [ ] Add environment variable validation
- [ ] Implement file system access checking
- [ ] Add system limits validation (ulimits, permissions)
- [ ] Create configuration recommendation system

### Phase 6: Integration and Testing (Week 3)
- [ ] Integrate all validation components
- [ ] Add comprehensive test suite
- [ ] Create validation performance benchmarks
- [ ] Add documentation and troubleshooting guides

## Testing Strategy

### Validation Test Suite
```bash
# Test system validation in various scenarios
cargo test --no-default-features --features cpu validate_system_requirements
cargo test --no-default-features --features gpu validate_system_requirements

# Test with different memory configurations
BITNET_MEMORY_LIMIT=1GB cargo test system_validation
BITNET_MEMORY_LIMIT=insufficient cargo test system_validation_failure

# Test with different thread configurations
RAYON_NUM_THREADS=1 cargo test cpu_validation
RAYON_NUM_THREADS=16 cargo test cpu_validation
```

### Performance Testing
- Validation overhead should be < 100ms for typical configurations
- Memory calculation accuracy within 10% of actual usage
- GPU validation should not affect inference performance

### Error Scenarios
- Insufficient memory conditions
- Missing CUDA drivers
- Incompatible quantization methods
- Invalid environment configurations

## Acceptance Criteria

- [ ] Comprehensive memory validation with accurate model-specific calculations
- [ ] Device capability validation for both CPU (SIMD) and GPU (CUDA) backends
- [ ] Basic operation testing validates quantization and forward pass functionality
- [ ] Model compatibility checking for all supported quantization methods
- [ ] Environment configuration validation with helpful error messages
- [ ] Validation completes in < 100ms for typical configurations
- [ ] Comprehensive error messages with suggested remediation steps
- [ ] Full test coverage for validation success and failure paths
- [ ] Documentation includes troubleshooting guide for common validation failures

## Dependencies

### New Dependencies
```toml
[dependencies]
sysinfo = "0.30"  # System information
```

### Potential GPU Dependencies
```toml
[dependencies]
cudarc = { version = "0.10", optional = true }  # CUDA information (if not already present)
```

## Related Issues

- DeviceManager stub implementations need completion
- GPU memory management validation
- Model memory requirement calculations
- Performance benchmarking integration

## BitNet-Specific Considerations

- **Quantization Method Support**: Each quantization method (I2S, TL1, TL2) has specific requirements
- **Device Optimization**: Different validation criteria for CPU vs GPU deployment
- **Memory Patterns**: BitNet models have unique memory usage patterns due to 1-bit quantization
- **Cross-Validation**: Validation should support comparison with C++ reference implementation

This comprehensive system validation will significantly improve the production readiness and reliability of BitNet-rs inference deployments.

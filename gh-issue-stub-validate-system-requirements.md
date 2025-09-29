# [Production Engine] Implement comprehensive system requirements validation

## Problem Description

The `ProductionInferenceEngine::validate_system_requirements` function in `crates/bitnet-inference/src/production_engine.rs` currently contains only a placeholder implementation with a hardcoded 1GB memory requirement check. The function includes a detailed comment outlining what a real implementation should do but lacks the actual implementation, representing a critical gap in production-readiness validation.

## Environment

- **File**: `crates/bitnet-inference/src/production_engine.rs`
- **Function**: `ProductionInferenceEngine::validate_system_requirements`
- **Component**: Production inference engine with device manager integration
- **Architecture**: Device-aware inference with fallback capabilities
- **Dependencies**: `DeviceManager`, BitNet model configuration, system introspection

## Root Cause Analysis

### Current Implementation

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

### Identified Gaps

1. **Memory Validation**: No actual system memory checking against model requirements
2. **Device Capability Validation**: Limited to a single hardcoded compatibility check
3. **Operational Testing**: No verification that basic operations can be performed
4. **Model Compatibility**: No validation of model-specific requirements against device capabilities
5. **Comprehensive Error Reporting**: Missing detailed failure diagnostics

### Architecture Context

The `ProductionInferenceEngine` uses a `DeviceManager` with the following structure:
- Primary device configuration with fallback support
- Device capabilities tracking (memory, compute capability, mixed precision support)
- Optimal batch size determination
- Cross-device compatibility validation

## Impact Assessment

- **Severity**: High - Critical for production deployment safety
- **Production Impact**: High - May lead to runtime failures in production without proper validation
- **User Experience**: High - Poor error messages when system requirements aren't met
- **Debugging Difficulty**: High - Failures occur during inference rather than initialization
- **Business Risk**: Medium - Potential for customer-facing inference failures

## Proposed Solution

### Comprehensive System Requirements Validation

Implement a multi-layered validation approach that checks all critical system requirements:

```rust
pub fn validate_system_requirements(&self) -> Result<()> {
    info!("Validating system requirements for production inference");

    // 1. System Memory Validation
    self.validate_memory_requirements()?;

    // 2. Device Capability Validation
    self.validate_device_capabilities()?;

    // 3. Basic Operations Testing
    self.test_basic_operations().await?;

    // 4. Model Compatibility Verification
    self.verify_model_compatibility()?;

    // 5. Performance Baseline Validation
    self.validate_performance_baselines().await?;

    info!("System requirements validation completed successfully");
    Ok(())
}

fn validate_memory_requirements(&self) -> Result<()> {
    let model_config = self.engine.model.config();
    let estimated_memory_mb = self.estimate_model_memory_requirements(model_config)?;

    // Check system memory
    let system_memory_info = self.get_system_memory_info()?;
    let required_memory_mb = estimated_memory_mb + MEMORY_BUFFER_MB;

    if system_memory_info.available_mb < required_memory_mb {
        return Err(BitNetError::Validation(format!(
            "Insufficient system memory: {}MB available, {}MB required (model: {}MB + {}MB buffer)",
            system_memory_info.available_mb,
            required_memory_mb,
            estimated_memory_mb,
            MEMORY_BUFFER_MB
        )));
    }

    // Check device-specific memory if applicable
    if let Device::Cuda(_) = self.device_manager.primary_device {
        self.validate_gpu_memory_requirements(required_memory_mb)?;
    }

    debug!(
        "Memory validation passed: {}MB available, {}MB required",
        system_memory_info.available_mb, required_memory_mb
    );

    Ok(())
}

fn validate_device_capabilities(&self) -> Result<()> {
    let model_config = self.engine.model.config();

    // Validate primary device
    self.device_manager.validate_device_compatibility(
        self.estimate_memory_bytes(model_config)?
    )?;

    // Check specific capability requirements
    match &self.device_manager.primary_device {
        Device::Cuda(device_id) => {
            self.validate_cuda_capabilities(*device_id, model_config)?;
        },
        Device::Cpu => {
            self.validate_cpu_capabilities(model_config)?;
        },
        Device::Metal => {
            self.validate_metal_capabilities(model_config)?;
        }
    }

    // Verify fallback device if different from primary
    if self.device_manager.fallback_device != self.device_manager.primary_device {
        self.validate_fallback_device_capabilities(model_config)?;
    }

    Ok(())
}

async fn test_basic_operations(&self) -> Result<()> {
    debug!("Testing basic inference operations");

    // Create minimal test input
    let test_tokens = vec![1u32]; // Single test token
    let mut test_engine = self.engine.clone();

    // Test prefill operation
    let prefill_start = Instant::now();
    test_engine.prefill(&test_tokens).await.map_err(|e| {
        BitNetError::Validation(format!("Basic prefill operation failed: {}", e))
    })?;
    let prefill_duration = prefill_start.elapsed();

    // Test single decode step
    let decode_start = Instant::now();
    let _next_token = test_engine.decode_next().await.map_err(|e| {
        BitNetError::Validation(format!("Basic decode operation failed: {}", e))
    })?;
    let decode_duration = decode_start.elapsed();

    // Validate operation timing is reasonable
    if prefill_duration > Duration::from_secs(30) {
        return Err(BitNetError::Validation(format!(
            "Prefill operation too slow: {:?} (limit: 30s)", prefill_duration
        )));
    }

    if decode_duration > Duration::from_secs(10) {
        return Err(BitNetError::Validation(format!(
            "Decode operation too slow: {:?} (limit: 10s)", decode_duration
        )));
    }

    debug!(
        "Basic operations test passed: prefill={:?}, decode={:?}",
        prefill_duration, decode_duration
    );

    Ok(())
}

fn verify_model_compatibility(&self) -> Result<()> {
    let model_config = self.engine.model.config();
    let device_caps = &self.device_manager.capabilities;

    // Check quantization compatibility
    match model_config.quantization_type {
        QuantizationType::I2_S => {
            // I2_S requires specific device support
            if !self.device_supports_i2s_quantization()? {
                return Err(BitNetError::Validation(
                    "Device does not support I2_S quantization".to_string()
                ));
            }
        },
        QuantizationType::TL1 | QuantizationType::TL2 => {
            // Table lookup quantization requirements
            if !self.device_supports_table_lookup()? {
                return Err(BitNetError::Validation(
                    "Device does not support table lookup quantization".to_string()
                ));
            }
        },
        _ => {}
    }

    // Check context length support
    if model_config.max_context_length > device_caps.max_supported_context_length {
        return Err(BitNetError::Validation(format!(
            "Model context length ({}) exceeds device limit ({})",
            model_config.max_context_length,
            device_caps.max_supported_context_length
        )));
    }

    // Check batch size compatibility
    if self.device_manager.optimal_batch_size > device_caps.max_batch_size {
        return Err(BitNetError::Validation(format!(
            "Optimal batch size ({}) exceeds device limit ({})",
            self.device_manager.optimal_batch_size,
            device_caps.max_batch_size
        )));
    }

    debug!("Model compatibility verification passed");
    Ok(())
}

async fn validate_performance_baselines(&self) -> Result<()> {
    debug!("Validating performance baselines");

    // Run brief performance test
    let performance_test_tokens = vec![1u32, 2u32, 3u32, 4u32, 5u32];
    let mut test_engine = self.engine.clone();

    let start_time = Instant::now();
    test_engine.prefill(&performance_test_tokens).await?;

    // Generate a few tokens to establish baseline
    for _ in 0..3 {
        let _token = test_engine.decode_next().await?;
    }

    let total_duration = start_time.elapsed();
    let tokens_per_second = (performance_test_tokens.len() + 3) as f64 / total_duration.as_secs_f64();

    // Check against minimum performance threshold
    let min_tokens_per_second = 1.0; // Very conservative baseline
    if tokens_per_second < min_tokens_per_second {
        return Err(BitNetError::Validation(format!(
            "Performance below baseline: {:.2} tokens/sec (minimum: {:.2})",
            tokens_per_second, min_tokens_per_second
        )));
    }

    debug!(
        "Performance baseline validation passed: {:.2} tokens/sec",
        tokens_per_second
    );

    Ok(())
}
```

### Supporting Infrastructure

```rust
#[derive(Debug)]
struct SystemMemoryInfo {
    total_mb: usize,
    available_mb: usize,
    used_mb: usize,
}

impl ProductionInferenceEngine {
    fn get_system_memory_info(&self) -> Result<SystemMemoryInfo> {
        // Platform-specific memory detection
        #[cfg(target_os = "linux")]
        {
            self.get_linux_memory_info()
        }
        #[cfg(target_os = "macos")]
        {
            self.get_macos_memory_info()
        }
        #[cfg(target_os = "windows")]
        {
            self.get_windows_memory_info()
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        {
            // Conservative fallback
            Ok(SystemMemoryInfo {
                total_mb: 8192,
                available_mb: 4096,
                used_mb: 4096,
            })
        }
    }

    fn estimate_model_memory_requirements(&self, config: &BitNetConfig) -> Result<usize> {
        // Calculate memory requirements based on model configuration
        let param_count = config.vocab_size * config.hidden_size +
                         config.num_layers * config.hidden_size * config.hidden_size;

        // Estimate memory usage (parameters + activations + overhead)
        let base_memory_mb = match config.quantization_type {
            QuantizationType::I2_S => param_count / 4 / 1024 / 1024, // ~2 bits per parameter
            QuantizationType::TL1 | QuantizationType::TL2 => param_count / 2 / 1024 / 1024, // ~4 bits per parameter
            _ => param_count * 4 / 1024 / 1024, // 32 bits per parameter fallback
        };

        // Add activation memory (context_length * hidden_size * batch_size * 4 bytes)
        let activation_memory_mb = config.max_context_length * config.hidden_size *
                                  self.device_manager.optimal_batch_size * 4 / 1024 / 1024;

        // Add overhead for KV cache, intermediate computations
        let overhead_memory_mb = (base_memory_mb + activation_memory_mb) / 2;

        Ok(base_memory_mb + activation_memory_mb + overhead_memory_mb)
    }
}

const MEMORY_BUFFER_MB: usize = 512; // 512MB safety buffer
```

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)
- [ ] Implement system memory detection for target platforms (Linux, macOS, Windows)
- [ ] Create model memory estimation algorithms based on quantization type
- [ ] Add comprehensive device capability validation
- [ ] Implement basic error reporting and diagnostics

### Phase 2: Operational Testing (Week 2)
- [ ] Implement basic operations testing (prefill, decode)
- [ ] Add performance baseline validation
- [ ] Create model compatibility verification
- [ ] Add fallback device validation

### Phase 3: Enhanced Validation (Week 3)
- [ ] Implement device-specific capability checks (CUDA compute capability, CPU features)
- [ ] Add quantization-specific validation (I2_S, TL1, TL2)
- [ ] Create comprehensive error messaging and diagnostics
- [ ] Add performance profiling and optimization recommendations

### Phase 4: Integration and Testing (Week 4)
- [ ] Integrate with existing `DeviceManager` infrastructure
- [ ] Add comprehensive test coverage
- [ ] Update documentation and examples
- [ ] Validate across different hardware configurations

## Testing Strategy

### Unit Testing
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_validation_sufficient() {
        let engine = create_test_production_engine().await;
        let result = engine.validate_memory_requirements();
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_memory_validation_insufficient() {
        let engine = create_memory_constrained_test_engine().await;
        let result = engine.validate_memory_requirements();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Insufficient system memory"));
    }

    #[tokio::test]
    async fn test_basic_operations_validation() {
        let engine = create_test_production_engine().await;
        let result = engine.test_basic_operations().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_model_compatibility_validation() {
        let engine = create_test_production_engine().await;
        let result = engine.verify_model_compatibility();
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_full_system_requirements_validation() {
        let engine = create_test_production_engine().await;
        let result = engine.validate_system_requirements().await;
        assert!(result.is_ok());
    }
}
```

### Integration Testing
```bash
# Test across different device configurations
cargo test --package bitnet-inference production_engine::validate_system_requirements --no-default-features --features cpu
cargo test --package bitnet-inference production_engine::validate_system_requirements --no-default-features --features gpu

# Test with different model configurations
BITNET_TEST_MODEL_SIZE=small cargo test validate_system_requirements
BITNET_TEST_MODEL_SIZE=large cargo test validate_system_requirements
```

### Performance Testing
```rust
#[tokio::test]
async fn test_validation_performance() {
    let engine = create_test_production_engine().await;

    let start = Instant::now();
    let result = engine.validate_system_requirements().await;
    let validation_time = start.elapsed();

    assert!(result.is_ok());
    assert!(validation_time < Duration::from_secs(5), "Validation should complete quickly");
}
```

## Related Issues/PRs

- Related to `DeviceManager` capability detection improvements
- Connects to model loading and compatibility validation
- May inform inference server startup validation procedures
- Links to performance benchmarking and optimization efforts

## Acceptance Criteria

- [ ] Comprehensive memory validation against model requirements
- [ ] Device capability validation for all supported device types
- [ ] Basic operations testing with reasonable performance thresholds
- [ ] Model compatibility verification for all quantization types
- [ ] Performance baseline validation
- [ ] Clear, actionable error messages for all validation failures
- [ ] Cross-platform system memory detection
- [ ] Integration with existing `DeviceManager` infrastructure
- [ ] Comprehensive test coverage including edge cases
- [ ] Documentation with usage examples and troubleshooting guide

## Priority: High

This is a critical production-readiness feature that prevents runtime failures and improves user experience by providing clear feedback about system compatibility issues during initialization rather than during inference.
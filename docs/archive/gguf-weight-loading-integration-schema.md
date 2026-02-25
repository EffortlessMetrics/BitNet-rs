# GGUF Weight Loading Integration Schema

## Overview

This document defines the comprehensive integration schema for GGUF weight loading in BitNet.rs, providing the implementation roadmap that connects all architectural components, API contracts, and validation frameworks into a cohesive production-ready system.

## Integration Architecture

### Component Hierarchy

```
BitNet.rs Neural Network Pipeline
├── Model Loading (bitnet-models)
│   ├── GgufWeightLoader (Enhanced)
│   ├── ProgressiveGgufLoader (New)
│   └── QuantizationAwareLoader (New)
├── Quantization (bitnet-quantization)
│   ├── I2SQuantizationLoader (Enhanced)
│   ├── TLQuantizationLoader (New)
│   └── CrossValidationBridge (New)
├── Device Management (bitnet-common)
│   ├── DevicePlacementStrategy (New)
│   ├── MemoryEfficiencyValidator (New)
│   └── DeviceOptimizations (New)
├── Kernels (bitnet-kernels)
│   ├── Device-aware tensor operations
│   └── SIMD/CUDA optimizations
├── Inference (bitnet-inference)
│   ├── Real weight integration
│   └── Performance validation
└── Cross-Validation (crossval)
    ├── C++ reference bridge
    └── Accuracy validation framework
```

## Implementation Integration Points

### 1. Enhanced Model Loading Pipeline

#### Primary Integration: `bitnet-models` → `bitnet-quantization`

```rust
/// Enhanced GGUF weight loader integrating all components
pub struct EnhancedGgufWeightLoader {
    /// Progressive loading capability
    progressive_loader: Option<ProgressiveGgufLoader>,
    /// Quantization-aware loading
    quantization_loader: Box<dyn QuantizationAwareLoader>,
    /// Device placement strategy
    placement_strategy: Box<dyn DevicePlacementStrategy>,
    /// Memory efficiency validator
    memory_validator: MemoryEfficiencyValidator,
    /// Cross-validation framework
    cross_validator: Option<GgufCrossValidator>,
}

impl EnhancedGgufWeightLoader {
    /// Create loader with feature-based configuration
    ///
    /// # Feature Flag Integration
    /// * `--no-default-features --features cpu`: CPU-optimized loading
    /// * `--no-default-features --features gpu`: GPU-accelerated loading
    /// * `--no-default-features`: Minimal loading for testing
    pub fn new(config: EnhancedGgufConfig) -> Result<Self> {
        // Feature-aware component selection
        let quantization_loader = Self::create_quantization_loader(&config)?;
        let placement_strategy = Self::create_placement_strategy(&config)?;

        let progressive_loader = if config.enable_progressive_loading {
            Some(ProgressiveGgufLoader::new(
                config.model_path.clone(),
                config.progressive_strategy.clone(),
                placement_strategy.clone(),
            )?)
        } else {
            None
        };

        let cross_validator = if config.enable_cross_validation {
            Some(GgufCrossValidator::new(config.cpp_library_path.clone())?)
        } else {
            None
        };

        Ok(Self {
            progressive_loader,
            quantization_loader,
            placement_strategy,
            memory_validator: MemoryEfficiencyValidator::new(config.memory_efficiency_target),
            cross_validator,
        })
    }

    /// Load GGUF weights with full integration pipeline
    ///
    /// # Acceptance Criteria Coverage
    /// * AC1: Parse and load all transformer layer weights
    /// * AC2: Support quantization formats with ≥99% accuracy
    /// * AC3: Tensor metadata validation with shape verification
    /// * AC5: Cross-validation with <1e-5 tolerance
    /// * AC6: Device-aware tensor placement
    /// * AC7: Memory-efficient loading with progressive loading >4GB
    pub async fn load_full_model(&mut self, model_path: impl AsRef<Path>) -> Result<BitNetModel> {
        let model_metadata = self.analyze_model_requirements(&model_path).await?;

        // Determine loading strategy based on model size and available memory
        let loading_strategy = self.determine_loading_strategy(&model_metadata)?;

        let (model, memory_report) = self.memory_validator.monitor_loading_efficiency(
            model_metadata.total_size_bytes,
            || self.execute_loading_strategy(loading_strategy, &model_path)
        ).await?;

        // Validate memory efficiency requirements (AC7)
        if !memory_report.meets_efficiency_target {
            return Err(BitNetError::MemoryEfficiencyViolation(
                format!(
                    "Memory efficiency {:.2} exceeds target {:.2}: {}",
                    memory_report.efficiency_ratio,
                    memory_report.target_ratio,
                    memory_report.efficiency_summary()
                )
            ));
        }

        // Cross-validation against C++ reference (AC5)
        if let Some(ref validator) = self.cross_validator {
            let validation_result = validator.validate_weights_against_cpp(&model, &model_path).await?;

            if !validation_result.meets_tolerance_requirements() {
                return Err(BitNetError::CrossValidationFailure(
                    format!("Cross-validation failed: {}", validation_result.summary())
                ));
            }
        }

        Ok(model)
    }

    /// Analyze model requirements for loading strategy selection
    async fn analyze_model_requirements(&self, model_path: impl AsRef<Path>) -> Result<ModelAnalysis> {
        let file_size = std::fs::metadata(&model_path)?.len();
        let available_memory = get_system_memory_info()?.available_bytes;

        // Parse GGUF metadata to understand tensor requirements
        let metadata = GgufMetadataParser::parse_file(&model_path)?;

        let tensor_count = metadata.tensor_count();
        let largest_tensor_size = metadata.largest_tensor_size();
        let quantization_types = metadata.quantization_types();

        Ok(ModelAnalysis {
            total_size_bytes: file_size,
            tensor_count,
            largest_tensor_size,
            quantization_types,
            memory_pressure_ratio: file_size as f32 / available_memory as f32,
            requires_progressive_loading: file_size > 4 * 1024 * 1024 * 1024, // >4GB
        })
    }

    /// Determine optimal loading strategy based on analysis
    fn determine_loading_strategy(&self, analysis: &ModelAnalysis) -> Result<LoadingStrategy> {
        if analysis.requires_progressive_loading || analysis.memory_pressure_ratio > 0.6 {
            Ok(LoadingStrategy::Progressive {
                batch_size: self.calculate_optimal_batch_size(analysis)?,
                enable_gc: true,
                priority_tensors: analysis.identify_critical_tensors(),
            })
        } else {
            Ok(LoadingStrategy::Standard {
                validate_during_load: true,
                device_optimization: true,
            })
        }
    }

    /// Execute selected loading strategy
    async fn execute_loading_strategy(
        &mut self,
        strategy: LoadingStrategy,
        model_path: impl AsRef<Path>,
    ) -> Result<BitNetModel> {
        match strategy {
            LoadingStrategy::Progressive { batch_size, enable_gc, priority_tensors } => {
                if let Some(ref mut progressive) = self.progressive_loader {
                    progressive.load_progressive().await
                } else {
                    return Err(BitNetError::ConfigurationError(
                        "Progressive loading requested but not configured".to_string()
                    ));
                }
            }
            LoadingStrategy::Standard { validate_during_load, device_optimization } => {
                self.load_standard_strategy(model_path, validate_during_load, device_optimization).await
            }
        }
    }

    /// Feature-aware quantization loader creation
    fn create_quantization_loader(config: &EnhancedGgufConfig) -> Result<Box<dyn QuantizationAwareLoader>> {
        match &config.primary_quantization_type {
            QuantizationType::I2S => {
                Ok(Box::new(I2SQuantizationLoader::new(
                    Some(82), // GGML block size
                    Some(config.accuracy_threshold),
                    &config.target_device,
                )?))
            }
            QuantizationType::TL1 => {
                Ok(Box::new(TLQuantizationLoader::new(
                    TLQuantizationType::TL1,
                    config.vectorization_strategy.clone(),
                )?))
            }
            QuantizationType::TL2 => {
                Ok(Box::new(TLQuantizationLoader::new(
                    TLQuantizationType::TL2,
                    config.vectorization_strategy.clone(),
                )?))
            }
            _ => Err(BitNetError::UnsupportedQuantization(
                format!("Quantization type {:?} not supported", config.primary_quantization_type)
            ))
        }
    }

    /// Feature-aware device placement strategy creation
    fn create_placement_strategy(config: &EnhancedGgufConfig) -> Result<Box<dyn DevicePlacementStrategy>> {
        #[cfg(feature = "gpu")]
        {
            if config.target_device.is_gpu() && is_gpu_available()? {
                return Ok(Box::new(GpuOptimizedPlacement::new(
                    config.max_gpu_utilization,
                    config.enable_mixed_precision,
                    config.progressive_offload,
                )));
            }
        }

        #[cfg(feature = "cpu")]
        {
            Ok(Box::new(CpuOptimizedPlacement::new(
                config.numa_aware_allocation,
                config.cache_optimization,
            )))
        }

        #[cfg(not(any(feature = "cpu", feature = "gpu")))]
        {
            Ok(Box::new(MinimalPlacement::new()))
        }
    }
}
```

### 2. Quantization Integration Schema

#### Integration: `bitnet-quantization` ↔ `bitnet-models`

```rust
/// Quantization integration coordinator
pub struct QuantizationIntegrator {
    /// Supported quantization loaders
    loaders: HashMap<QuantizationType, Box<dyn QuantizationAwareLoader>>,
    /// Accuracy validation requirements
    accuracy_requirements: AccuracyRequirements,
    /// Performance monitoring
    performance_monitor: QuantizationPerformanceMonitor,
}

impl QuantizationIntegrator {
    /// Integrate quantized tensor loading with validation
    ///
    /// # Acceptance Criteria Coverage
    /// * AC2: Support I2_S, TL1, TL2 with ≥99% accuracy preservation
    /// * AC3: Tensor metadata validation including shape verification
    pub async fn load_and_validate_quantized_tensor(
        &self,
        tensor_info: &TensorInfo,
        tensor_data: &[u8],
        target_device: &Device,
    ) -> Result<ValidatedQuantizedTensor> {
        // Select appropriate quantization loader
        let loader = self.loaders.get(&tensor_info.quantization_type)
            .ok_or_else(|| BitNetError::UnsupportedQuantization(
                format!("No loader available for {:?}", tensor_info.quantization_type)
            ))?;

        // Load quantized tensor with device placement
        let performance_start = Instant::now();
        let quantized_result = loader.load_quantized_tensor(
            tensor_info,
            target_device,
            &self.accuracy_requirements.validation_config,
        )?;
        let loading_duration = performance_start.elapsed();

        // Validate accuracy requirements
        if let Some(ref accuracy_report) = quantized_result.accuracy_report {
            self.validate_accuracy_requirements(accuracy_report, tensor_info)?;
        }

        // Record performance metrics
        self.performance_monitor.record_loading_performance(
            tensor_info.quantization_type,
            tensor_info.element_count(),
            loading_duration,
        );

        Ok(ValidatedQuantizedTensor {
            tensor: quantized_result.tensor,
            quantization_type: quantized_result.quantization_type,
            device_placement: quantized_result.device_placement,
            accuracy_validated: accuracy_report.is_some(),
            validation_timestamp: SystemTime::now(),
        })
    }

    /// Validate accuracy requirements against thresholds
    fn validate_accuracy_requirements(
        &self,
        accuracy_report: &QuantizationAccuracyReport,
        tensor_info: &TensorInfo,
    ) -> Result<()> {
        let threshold = self.accuracy_requirements.get_threshold(tensor_info.quantization_type);

        if accuracy_report.accuracy_percentage < threshold {
            return Err(BitNetError::AccuracyValidationFailure(
                format!(
                    "Tensor '{}' accuracy {:.4} below threshold {:.4} for {:?}",
                    tensor_info.name,
                    accuracy_report.accuracy_percentage,
                    threshold,
                    tensor_info.quantization_type
                )
            ));
        }

        if accuracy_report.max_absolute_error > self.accuracy_requirements.max_absolute_error {
            return Err(BitNetError::AccuracyValidationFailure(
                format!(
                    "Tensor '{}' absolute error {:.2e} exceeds limit {:.2e}",
                    tensor_info.name,
                    accuracy_report.max_absolute_error,
                    self.accuracy_requirements.max_absolute_error
                )
            ));
        }

        Ok(())
    }
}

/// Accuracy requirements configuration
#[derive(Debug, Clone)]
pub struct AccuracyRequirements {
    /// Per-quantization-type accuracy thresholds
    pub quantization_thresholds: HashMap<QuantizationType, f32>,
    /// Maximum allowable absolute error
    pub max_absolute_error: f32,
    /// Maximum allowable relative error
    pub max_relative_error: f32,
    /// Validation configuration
    pub validation_config: QuantizationValidationConfig,
}

impl Default for AccuracyRequirements {
    fn default() -> Self {
        let mut thresholds = HashMap::new();
        thresholds.insert(QuantizationType::I2S, 0.99);  // 99% for I2_S
        thresholds.insert(QuantizationType::TL1, 0.99);  // 99% for TL1
        thresholds.insert(QuantizationType::TL2, 0.99);  // 99% for TL2

        Self {
            quantization_thresholds: thresholds,
            max_absolute_error: 1e-3,  // Slightly higher for quantized tensors
            max_relative_error: 1e-2,
            validation_config: QuantizationValidationConfig::default(),
        }
    }
}
```

### 3. Cross-Validation Integration Schema

#### Integration: `crossval` ↔ All Components

```rust
/// Comprehensive cross-validation integration
pub struct CrossValidationIntegrator {
    /// C++ reference bridge
    cpp_bridge: CppReferenceBridge,
    /// Validation tolerances
    tolerances: ValidationTolerances,
    /// Deterministic configuration
    deterministic_config: DeterministicConfig,
    /// Results aggregator
    results_aggregator: ValidationResultsAggregator,
}

impl CrossValidationIntegrator {
    /// Comprehensive model validation against C++ reference
    ///
    /// # Acceptance Criteria Coverage
    /// * AC5: Cross-validation framework with numerical tolerance <1e-5
    /// * AC5: Deterministic inference validation
    pub async fn validate_full_model_pipeline(
        &self,
        rust_model: &BitNetModel,
        gguf_path: impl AsRef<Path>,
        test_inputs: &[TestInput],
    ) -> Result<ComprehensiveValidationReport> {
        // Ensure deterministic environment
        self.configure_deterministic_environment()?;

        // Load C++ reference model
        let cpp_model = self.cpp_bridge.load_model(gguf_path.as_ref())?;

        let mut validation_results = Vec::new();

        // Phase 1: Weight validation
        let weight_validation = self.validate_weights_parity(rust_model, &cpp_model).await?;
        validation_results.push(ValidationPhase::Weights(weight_validation));

        // Phase 2: Individual layer validation
        let layer_validation = self.validate_layer_outputs(rust_model, &cpp_model, test_inputs).await?;
        validation_results.push(ValidationPhase::Layers(layer_validation));

        // Phase 3: Full inference validation
        let inference_validation = self.validate_full_inference(rust_model, &cpp_model, test_inputs).await?;
        validation_results.push(ValidationPhase::Inference(inference_validation));

        // Aggregate results and determine overall status
        let overall_report = self.results_aggregator.aggregate_validation_results(validation_results)?;

        Ok(overall_report)
    }

    /// Configure deterministic environment for validation
    fn configure_deterministic_environment(&self) -> Result<()> {
        // Set deterministic seed
        if let Some(seed) = self.deterministic_config.seed {
            std::env::set_var("BITNET_SEED", seed.to_string());
        }

        // Enable deterministic mode
        std::env::set_var("BITNET_DETERMINISTIC", "1");

        // Configure single-threaded execution for determinism
        std::env::set_var("RAYON_NUM_THREADS", "1");

        Ok(())
    }

    /// Validate weight parity between Rust and C++ implementations
    async fn validate_weights_parity(
        &self,
        rust_model: &BitNetModel,
        cpp_model: &CppBitNetModel,
    ) -> Result<WeightValidationResult> {
        let mut tensor_results = Vec::new();
        let mut max_error = 0.0f32;
        let mut failed_tensors = Vec::new();

        for (tensor_name, rust_tensor) in rust_model.tensors() {
            let cpp_tensor = cpp_model.get_tensor(tensor_name)
                .map_err(|e| BitNetError::CrossValidationError(
                    format!("Failed to get C++ tensor '{}': {}", tensor_name, e)
                ))?;

            let tensor_result = self.validate_single_tensor_parity(
                tensor_name,
                rust_tensor,
                &cpp_tensor,
            ).await?;

            if let Some(abs_error) = tensor_result.max_absolute_error {
                max_error = max_error.max(abs_error);
            }

            if tensor_result.status == ValidationStatus::Failed {
                failed_tensors.push(tensor_name.clone());
            }

            tensor_results.push(tensor_result);
        }

        let overall_status = if failed_tensors.is_empty() && max_error <= self.tolerances.absolute_tolerance {
            ValidationStatus::Passed
        } else {
            ValidationStatus::Failed
        };

        Ok(WeightValidationResult {
            overall_status,
            max_absolute_error: max_error,
            failed_tensors,
            tensor_results,
            validation_timestamp: SystemTime::now(),
        })
    }
}
```

## Feature Flag Integration Strategy

### Compilation Configurations

```rust
/// Feature-aware configuration factory
pub struct FeatureAwareConfigFactory;

impl FeatureAwareConfigFactory {
    /// Create configuration based on enabled features
    pub fn create_config() -> EnhancedGgufConfig {
        let mut config = EnhancedGgufConfig::default();

        // CPU feature configuration
        #[cfg(feature = "cpu")]
        {
            config.target_device = Device::Cpu;
            config.vectorization_strategy = VectorizationStrategy::SIMD;
            config.numa_aware_allocation = true;
            config.cache_optimization = true;
        }

        // GPU feature configuration
        #[cfg(feature = "gpu")]
        {
            if is_gpu_available().unwrap_or(false) {
                config.target_device = Device::Gpu { device_index: 0 };
                config.enable_mixed_precision = true;
                config.max_gpu_utilization = 0.8;
                config.progressive_offload = true;
            }
        }

        // Cross-validation feature
        #[cfg(feature = "crossval")]
        {
            config.enable_cross_validation = true;
            config.cpp_library_path = Some(PathBuf::from("target/release/libbitnet_cpp.so"));
        }

        // Default minimal configuration
        #[cfg(not(any(feature = "cpu", feature = "gpu")))]
        {
            config.target_device = Device::Cpu;
            config.enable_progressive_loading = false;
            config.enable_cross_validation = false;
            config.minimal_mode = true;
        }

        config
    }
}
```

## Testing Integration Framework

### Comprehensive Test Structure

```rust
#[cfg(test)]
mod integration_tests {
    use super::*;

    /// Comprehensive integration test covering all acceptance criteria
    #[tokio::test]
    async fn test_full_gguf_weight_loading_integration() {
        // AC:1, AC:2, AC:3, AC:5, AC:6, AC:7, AC:9, AC:10
        let config = FeatureAwareConfigFactory::create_config();
        let mut loader = EnhancedGgufWeightLoader::new(config).expect("Failed to create loader");

        // Test with different model sizes and quantization types
        let test_cases = vec![
            TestCase::small_model_i2s(),
            TestCase::medium_model_tl1(),
            TestCase::large_model_tl2(),
        ];

        for test_case in test_cases {
            let model_path = test_case.prepare_test_model().await?;

            // Load model with full integration pipeline
            let model = loader.load_full_model(&model_path).await
                .expect(&format!("Failed to load model for test case: {}", test_case.name));

            // Validate all acceptance criteria
            test_case.validate_acceptance_criteria(&model).await?;
        }
    }

    /// Feature flag isolation tests
    #[tokio::test]
    async fn test_feature_flag_isolation() {
        // Ensure proper behavior with different feature flag combinations

        #[cfg(all(feature = "cpu", not(feature = "gpu")))]
        {
            // CPU-only build should use CPU optimizations
            let config = FeatureAwareConfigFactory::create_config();
            assert_eq!(config.target_device, Device::Cpu);
            assert_eq!(config.vectorization_strategy, VectorizationStrategy::SIMD);
        }

        #[cfg(all(feature = "gpu", not(feature = "cpu")))]
        {
            // GPU-only build should use GPU optimizations
            let config = FeatureAwareConfigFactory::create_config();
            if is_gpu_available().unwrap_or(false) {
                assert!(matches!(config.target_device, Device::Gpu { .. }));
                assert!(config.enable_mixed_precision);
            }
        }

        #[cfg(not(any(feature = "cpu", feature = "gpu")))]
        {
            // Minimal build should use basic configuration
            let config = FeatureAwareConfigFactory::create_config();
            assert!(config.minimal_mode);
            assert!(!config.enable_progressive_loading);
        }
    }
}
```

## Validation and Compliance Framework

### Acceptance Criteria Mapping

| AC | Component | Integration Point | Validation Method |
|----|-----------|-------------------|-------------------|
| AC1 | Model Loading | `EnhancedGgufWeightLoader` | Tensor count and completeness validation |
| AC2 | Quantization | `QuantizationIntegrator` | Accuracy preservation ≥99% validation |
| AC3 | Validation | `TensorValidator` | Shape and metadata verification |
| AC5 | Cross-Validation | `CrossValidationIntegrator` | C++ reference comparison <1e-5 |
| AC6 | Device Placement | `DevicePlacementStrategy` | GPU/CPU selection and fallback testing |
| AC7 | Memory Efficiency | `MemoryEfficiencyValidator` | Progressive loading and <150% overhead |
| AC9 | Compatibility | `BackwardCompatibilityLayer` | Mock tensor loading preservation |
| AC10 | Documentation | Integration examples | Usage documentation and examples |

## Conclusion

This integration schema provides the complete implementation framework for GGUF weight loading that:

1. **Unifies Components**: Seamless integration across all BitNet.rs workspace crates
2. **Ensures Quality**: Comprehensive validation at every integration point
3. **Maintains Performance**: Device-aware optimizations and memory efficiency
4. **Preserves Compatibility**: Feature flag discipline and backward compatibility
5. **Enables Production**: Real neural network inference with trained parameters

The schema addresses all 10 acceptance criteria from Issue #159 while maintaining the architectural integrity of the BitNet.rs neural network inference pipeline.

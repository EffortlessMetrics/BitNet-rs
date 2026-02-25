# GGUF Weight Loading Integration Architecture and Test Strategy

## Overview

This document defines the comprehensive integration architecture and systematic test strategy for implementing real GGUF model weight loading in BitNet.rs. The architecture ensures seamless integration with existing workspace crates while the test strategy provides comprehensive coverage of all acceptance criteria through Test-Driven Development (TDD) practices.

## Integration Architecture

### Workspace Integration Strategy

#### Crate Dependency Architecture

```rust
// Primary integration flow across BitNet.rs workspace
pub mod integration_architecture {
    use bitnet_models::GgufWeightLoader;
    use bitnet_quantization::{I2SQuantizer, TL1Quantizer, TL2Quantizer};
    use bitnet_inference::InferenceEngine;
    use bitnet_kernels::KernelProvider;
    use bitnet_common::{BitNetConfig, Device, Result};

    /// Central integration coordinator for GGUF weight loading
    pub struct WeightLoadingOrchestrator {
        /// Core weight loader
        loader: GgufWeightLoader,
        /// Quantization manager
        quantization_manager: QuantizationManager,
        /// Device manager for GPU/CPU placement
        device_manager: DeviceManager,
        /// Validation coordinator
        validation_coordinator: ValidationCoordinator,
    }

    impl WeightLoadingOrchestrator {
        /// Create orchestrator with optimal configuration
        pub fn new_optimized() -> Self {
            Self {
                loader: GgufWeightLoader::new(),
                quantization_manager: QuantizationManager::new_with_all_formats(),
                device_manager: DeviceManager::with_auto_detection(),
                validation_coordinator: ValidationCoordinator::with_cpp_crossval(),
            }
        }

        /// Orchestrate complete model loading with all integrations
        pub fn load_complete_model_integrated(
            &self,
            path: &Path,
            target_device: Device,
        ) -> Result<IntegratedModelBundle> {
            // Phase 1: Load raw weights with device optimization
            let (config, raw_weights) = self.loader.load_complete_model(path, target_device)?;

            // Phase 2: Apply quantization optimizations if needed
            let optimized_weights = self.quantization_manager
                .optimize_weights_for_device(&raw_weights, target_device)?;

            // Phase 3: Device-specific kernel preparations
            let kernel_context = self.device_manager
                .prepare_kernels_for_weights(&optimized_weights, target_device)?;

            // Phase 4: Cross-validation with C++ reference
            let validation_report = self.validation_coordinator
                .validate_complete_integration(&optimized_weights, &config)?;

            Ok(IntegratedModelBundle {
                config,
                weights: optimized_weights,
                kernel_context,
                validation_report,
                device: target_device,
            })
        }
    }
}
```

#### Inter-Crate Communication Contracts

```rust
/// Communication protocols between workspace crates
pub mod inter_crate_protocols {
    /// Protocol for bitnet-models -> bitnet-quantization communication
    pub trait QuantizationIntegration {
        /// Request quantization for loaded tensor
        fn quantize_loaded_tensor(
            &self,
            tensor: &CandleTensor,
            qtype: QuantizationType,
            device: &Device,
        ) -> Result<QuantizedTensor>;

        /// Validate quantization accuracy during loading
        fn validate_quantization_accuracy(
            &self,
            original: &CandleTensor,
            quantized: &QuantizedTensor,
        ) -> Result<AccuracyReport>;
    }

    /// Protocol for bitnet-models -> bitnet-kernels communication
    pub trait KernelIntegration {
        /// Prepare GPU kernels for loaded weights
        fn prepare_kernels_for_loaded_weights(
            &self,
            weights: &HashMap<String, CandleTensor>,
            device: &Device,
        ) -> Result<KernelContext>;

        /// Optimize tensor layout for kernel operations
        fn optimize_tensor_layout(
            &self,
            tensor: &CandleTensor,
            operation_type: KernelOperationType,
        ) -> Result<CandleTensor>;
    }

    /// Protocol for bitnet-models -> bitnet-inference communication
    pub trait InferenceIntegration {
        /// Validate loaded weights are suitable for inference
        fn validate_weights_for_inference(
            &self,
            weights: &HashMap<String, CandleTensor>,
            config: &BitNetConfig,
        ) -> Result<InferenceCompatibilityReport>;

        /// Initialize inference engine with loaded weights
        fn initialize_with_loaded_weights(
            &mut self,
            weights: HashMap<String, CandleTensor>,
            config: BitNetConfig,
        ) -> Result<()>;
    }
}
```

### Backward Compatibility Strategy (AC9)

#### Mock Loading Preservation

```rust
/// Backward compatibility layer maintaining mock loading functionality
pub mod backward_compatibility {
    /// Compatibility wrapper preserving existing mock loading behavior
    pub struct BackwardCompatibleLoader {
        real_loader: GgufWeightLoader,
        mock_mode: bool,
        fallback_strategy: FallbackStrategy,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum FallbackStrategy {
        /// Use mock loading when real loading fails
        FailSafeToMock,
        /// Use mock loading for development/testing
        DevelopmentMock,
        /// Hybrid: real for some tensors, mock for others
        HybridMode,
        /// No fallback, fail if real loading fails
        NoFallback,
    }

    impl BackwardCompatibleLoader {
        /// Create loader with automatic fallback detection
        pub fn with_auto_fallback() -> Self {
            let fallback = if cfg!(debug_assertions) {
                FallbackStrategy::DevelopmentMock
            } else {
                FallbackStrategy::FailSafeToMock
            };

            Self {
                real_loader: GgufWeightLoader::new(),
                mock_mode: false,
                fallback_strategy: fallback,
            }
        }

        /// Load model with compatibility fallback (AC9)
        pub fn load_gguf_compatible(
            &self,
            path: &Path,
            device: Device,
        ) -> Result<(BitNetConfig, HashMap<String, CandleTensor>)> {
            match self.fallback_strategy {
                FallbackStrategy::DevelopmentMock if cfg!(debug_assertions) => {
                    tracing::info!("Using mock loading for development");
                    self.load_with_mock_fallback(path, device)
                }
                FallbackStrategy::NoFallback => {
                    self.real_loader.load_complete_model(path, device)
                }
                _ => {
                    // Try real loading first, fallback to mock if needed
                    match self.real_loader.load_complete_model(path, device) {
                        Ok(result) => Ok(result),
                        Err(error) => {
                            tracing::warn!("Real loading failed: {}, falling back to mock", error);
                            self.load_with_mock_fallback(path, device)
                        }
                    }
                }
            }
        }

        /// Enhanced gguf_simple.rs integration maintaining API compatibility
        fn load_with_mock_fallback(
            &self,
            path: &Path,
            device: Device,
        ) -> Result<(BitNetConfig, HashMap<String, CandleTensor>)> {
            // Load embeddings and output with real parsing (existing functionality)
            let two = crate::gguf_min::load_two(path)?;
            let mut config = bitnet_common::BitNetConfig::default();
            config.model.vocab_size = two.vocab;
            config.model.hidden_size = two.dim;

            // Create device-appropriate tensors
            let cdevice = device.to_candle_device()?;
            let mut tensor_map = HashMap::new();

            // Real embeddings and output (preserved)
            tensor_map.insert(
                "token_embd.weight".to_string(),
                CandleTensor::from_vec(two.tok_embeddings, (config.model.vocab_size, config.model.hidden_size), &cdevice)?,
            );
            tensor_map.insert(
                "output.weight".to_string(),
                CandleTensor::from_vec(two.lm_head, (config.model.hidden_size, config.model.vocab_size), &cdevice)?,
            );

            // Mock transformer layers (preserved behavior)
            for layer in 0..config.model.num_layers {
                self.add_mock_layer_weights(&mut tensor_map, layer, &config, &cdevice)?;
            }

            Ok((config, tensor_map))
        }
    }
}
```

### Device-Aware Architecture (AC6)

#### Multi-Device Support Strategy

```rust
/// Device-aware architecture supporting CPU/GPU with graceful fallback
pub mod device_aware_architecture {
    /// Device capability detection and optimization
    pub struct DeviceCapabilityManager {
        cpu_capabilities: CpuCapabilities,
        gpu_capabilities: Option<GpuCapabilities>,
        memory_limits: DeviceMemoryLimits,
    }

    #[derive(Debug, Clone)]
    pub struct CpuCapabilities {
        pub simd_support: SimdSupport,
        pub numa_topology: NumaTopology,
        pub memory_bandwidth_gbps: f32,
        pub cache_sizes: CacheSizes,
    }

    #[derive(Debug, Clone)]
    pub struct GpuCapabilities {
        pub device_name: String,
        pub compute_capability: (u32, u32),
        pub memory_gb: f32,
        pub memory_bandwidth_gbps: f32,
        pub tensor_core_support: bool,
        pub mixed_precision_support: MixedPrecisionSupport,
    }

    impl DeviceCapabilityManager {
        /// Auto-detect optimal device configuration
        pub fn detect_optimal_configuration(&self) -> DeviceConfiguration {
            let cpu_score = self.calculate_cpu_performance_score();
            let gpu_score = self.gpu_capabilities.as_ref()
                .map(|gpu| self.calculate_gpu_performance_score(gpu))
                .unwrap_or(0.0);

            if gpu_score > cpu_score * 1.2 { // 20% advantage threshold
                DeviceConfiguration {
                    primary_device: Device::Cuda(0),
                    fallback_device: Device::Cpu,
                    memory_strategy: MemoryStrategy::GpuOptimized,
                    mixed_precision: self.gpu_capabilities.as_ref()
                        .map(|gpu| gpu.mixed_precision_support.optimal_precision())
                        .unwrap_or(Precision::F32),
                }
            } else {
                DeviceConfiguration {
                    primary_device: Device::Cpu,
                    fallback_device: Device::Cpu,
                    memory_strategy: MemoryStrategy::CpuOptimized,
                    mixed_precision: Precision::F32,
                }
            }
        }

        /// Implement graceful GPU -> CPU fallback
        pub fn create_fallback_strategy(&self) -> FallbackStrategy {
            FallbackStrategy::new()
                .with_primary_device(self.detect_optimal_configuration().primary_device)
                .with_fallback_device(Device::Cpu)
                .with_memory_pressure_threshold(0.85) // Fallback at 85% memory usage
                .with_error_recovery(ErrorRecoveryStrategy::GradualDegradation)
        }
    }

    /// Device-aware tensor loading with automatic optimization
    pub struct DeviceAwareTensorLoader {
        device_manager: DeviceCapabilityManager,
        memory_optimizer: MemoryOptimizer,
        placement_strategy: TensorPlacementStrategy,
    }

    impl DeviceAwareTensorLoader {
        /// Load tensor with optimal device placement (AC6)
        pub fn load_tensor_optimized(
            &self,
            tensor_info: &TensorInfo,
            mmap: &Mmap,
            target_device: Device,
        ) -> Result<CandleTensor> {
            // Determine optimal placement based on tensor size and device capabilities
            let optimal_device = self.determine_optimal_placement(tensor_info, target_device)?;

            // Load tensor with device-specific optimizations
            let tensor_data = self.extract_tensor_data_optimized(tensor_info, mmap, &optimal_device)?;

            // Apply device-specific optimizations
            match optimal_device {
                Device::Cuda(_) => self.optimize_for_gpu(tensor_data, tensor_info),
                Device::Cpu => self.optimize_for_cpu(tensor_data, tensor_info),
                Device::Metal => self.optimize_for_metal(tensor_data, tensor_info),
            }
        }

        /// Implement progressive GPU memory management
        fn manage_gpu_memory_progressive(
            &self,
            tensors: &[TensorInfo],
            gpu_device: &Device,
        ) -> Result<Vec<TensorPlacement>> {
            let available_memory = self.device_manager.get_available_gpu_memory()?;
            let mut placements = Vec::new();
            let mut allocated_memory = 0usize;

            // Sort tensors by importance and frequency of access
            let mut sorted_tensors = tensors.to_vec();
            sorted_tensors.sort_by_key(|t| (t.access_frequency, t.size_bytes));

            for tensor in sorted_tensors {
                if allocated_memory + tensor.size_bytes <= (available_memory as f32 * 0.85) as usize {
                    placements.push(TensorPlacement::Gpu {
                        tensor_name: tensor.name.clone(),
                        device: gpu_device.clone(),
                    });
                    allocated_memory += tensor.size_bytes;
                } else {
                    placements.push(TensorPlacement::Cpu {
                        tensor_name: tensor.name.clone(),
                    });
                }
            }

            Ok(placements)
        }
    }
}
```

## Test Strategy Architecture

### Test-Driven Development Framework

#### Acceptance Criteria Test Structure

```rust
/// TDD framework with AC tag integration for systematic testing
pub mod tdd_framework {
    /// Test suite structure aligned with Issue #159 acceptance criteria
    pub struct GgufWeightLoadingTestSuite {
        ac1_tests: AC1TestSuite, // Parse and load all transformer layer weights
        ac2_tests: AC2TestSuite, // Support quantization formats with ≥99% accuracy
        ac3_tests: AC3TestSuite, // Robust tensor metadata validation
        ac4_tests: AC4TestSuite, // Graceful error handling with descriptive messages
        ac5_tests: AC5TestSuite, // Cross-validation against C++ reference
        ac6_tests: AC6TestSuite, // CPU/GPU feature flag support
        ac7_tests: AC7TestSuite, // Memory-efficient loading
        ac8_tests: AC8TestSuite, // Comprehensive test coverage
        ac9_tests: AC9TestSuite, // Backward compatibility with mock loading
        ac10_tests: AC10TestSuite, // Document tensor naming conventions
    }

    /// AC1: Complete weight parsing test suite
    pub struct AC1TestSuite {
        complete_model_tests: Vec<CompleteModelTest>,
        layer_specific_tests: Vec<LayerSpecificTest>,
        architecture_compatibility_tests: Vec<ArchitectureCompatibilityTest>,
    }

    impl AC1TestSuite {
        /// Test complete transformer layer weight parsing
        #[test]
        // AC1: Parse and load all transformer layer weights
        fn test_complete_weight_parsing_ac1() {
            let test_cases = vec![
                TestCase::new("bitnet-1.58b", "tests/fixtures/bitnet-1.58b.gguf"),
                TestCase::new("llama-7b", "tests/fixtures/llama-7b.gguf"),
                TestCase::new("custom-architecture", "tests/fixtures/custom.gguf"),
            ];

            for test_case in test_cases {
                let loader = GgufWeightLoader::new();
                let (config, weights) = loader.load_complete_model(
                    &test_case.model_path,
                    Device::Cpu
                ).expect(&format!("Failed to load {}", test_case.name));

                // Verify all expected transformer layers are loaded
                self.verify_complete_transformer_layers(&weights, &config)?;

                // Verify no mock tensors remain
                self.verify_no_zero_initialized_weights(&weights)?;

                // Verify proper tensor shapes and data types
                self.verify_tensor_metadata(&weights, &config)?;
            }
        }

        /// Verify all transformer layer types are loaded with real data
        fn verify_complete_transformer_layers(
            &self,
            weights: &HashMap<String, CandleTensor>,
            config: &BitNetConfig,
        ) -> Result<()> {
            for layer in 0..config.model.num_layers {
                let layer_prefix = format!("blk.{}", layer);

                // Attention layer weights
                let attention_weights = vec![
                    format!("{}.attn_q.weight", layer_prefix),
                    format!("{}.attn_k.weight", layer_prefix),
                    format!("{}.attn_v.weight", layer_prefix),
                    format!("{}.attn_output.weight", layer_prefix),
                ];

                for weight_name in attention_weights {
                    let weight = weights.get(&weight_name)
                        .ok_or_else(|| anyhow::anyhow!("Missing weight: {}", weight_name))?;

                    // Verify not zero-initialized (real data)
                    assert!(!self.is_zero_initialized(weight),
                           "Weight {} appears to be zero-initialized", weight_name);

                    // Verify proper shape
                    let expected_shape = vec![config.model.hidden_size, config.model.hidden_size];
                    assert_eq!(weight.shape(), expected_shape,
                              "Weight {} has incorrect shape", weight_name);
                }

                // Feed-forward network weights
                let ffn_weights = vec![
                    format!("{}.ffn_gate.weight", layer_prefix),
                    format!("{}.ffn_up.weight", layer_prefix),
                    format!("{}.ffn_down.weight", layer_prefix),
                ];

                for weight_name in ffn_weights {
                    let weight = weights.get(&weight_name)
                        .ok_or_else(|| anyhow::anyhow!("Missing weight: {}", weight_name))?;

                    assert!(!self.is_zero_initialized(weight),
                           "Weight {} appears to be zero-initialized", weight_name);
                }

                // Normalization weights
                let norm_weights = vec![
                    format!("{}.attn_norm.weight", layer_prefix),
                    format!("{}.ffn_norm.weight", layer_prefix),
                ];

                for weight_name in norm_weights {
                    let weight = weights.get(&weight_name)
                        .ok_or_else(|| anyhow::anyhow!("Missing weight: {}", weight_name))?;

                    // Norm weights may legitimately be ones, but should not be zero
                    assert!(!self.is_all_zeros(weight),
                           "Norm weight {} is all zeros", weight_name);
                }
            }

            Ok(())
        }

        /// Check if tensor is zero-initialized (mock data)
        fn is_zero_initialized(&self, tensor: &CandleTensor) -> bool {
            match tensor.to_vec1::<f32>() {
                Ok(data) => data.iter().all(|&x| x == 0.0),
                Err(_) => false, // If we can't extract data, assume it's real
            }
        }
    }

    /// AC2: Quantization accuracy test suite
    pub struct AC2TestSuite {
        i2s_accuracy_tests: Vec<I2SAccuracyTest>,
        tl1_accuracy_tests: Vec<TL1AccuracyTest>,
        tl2_accuracy_tests: Vec<TL2AccuracyTest>,
        cross_format_tests: Vec<CrossFormatTest>,
    }

    impl AC2TestSuite {
        #[test]
        // AC2: Support quantization formats with ≥99% accuracy
        fn test_quantization_accuracy_preservation_ac2() {
            let quantization_formats = vec![
                QuantizationType::I2S,
                QuantizationType::TL1,
                QuantizationType::TL2,
            ];

            for qtype in quantization_formats {
                let test_tensors = self.generate_test_tensors_for_quantization();

                for test_tensor in test_tensors {
                    let quantizer = self.create_quantizer_for_type(qtype);

                    // Perform quantization round-trip
                    let quantized = quantizer.quantize_tensor(&test_tensor)?;
                    let dequantized = quantizer.dequantize_tensor(&quantized)?;

                    // Calculate accuracy metrics
                    let accuracy = self.calculate_accuracy_metrics(&test_tensor, &dequantized);

                    // Verify ≥99% accuracy requirement
                    assert!(accuracy.cosine_similarity >= 0.99,
                           "Quantization {} accuracy {:.4} below required 99%",
                           qtype, accuracy.cosine_similarity);

                    assert!(accuracy.mse < 1e-4,
                           "Quantization {} MSE {:.6} exceeds threshold",
                           qtype, accuracy.mse);
                }
            }
        }

        /// Generate realistic test tensors for quantization validation
        fn generate_test_tensors_for_quantization(&self) -> Vec<BitNetTensor> {
            vec![
                // Small attention weight matrix
                self.create_test_tensor(vec![512, 512], TensorDistribution::Xavier),
                // Large feed-forward weight matrix
                self.create_test_tensor(vec![2048, 512], TensorDistribution::HeNormal),
                // Sparse attention pattern
                self.create_test_tensor(vec![1024, 1024], TensorDistribution::Sparse(0.8)),
                // Edge case: very small values
                self.create_test_tensor(vec![256, 256], TensorDistribution::SmallValues),
                // Edge case: mixed range values
                self.create_test_tensor(vec![512, 1024], TensorDistribution::MixedRange),
            ]
        }
    }
}
```

### Integration Test Architecture

#### Multi-Crate Integration Tests

```rust
/// Integration tests spanning multiple workspace crates
pub mod integration_tests {
    use bitnet_models::GgufWeightLoader;
    use bitnet_quantization::{I2SQuantizer, QuantizerTrait};
    use bitnet_inference::InferenceEngine;
    use bitnet_kernels::KernelProvider;

    /// End-to-end integration test suite
    pub struct EndToEndIntegrationTests;

    impl EndToEndIntegrationTests {
        #[test]
        // Integration: bitnet-models -> bitnet-quantization -> bitnet-inference
        fn test_complete_pipeline_integration() {
            // Phase 1: Load weights with bitnet-models
            let loader = GgufWeightLoader::new();
            let (config, weights) = loader.load_complete_model(
                Path::new("tests/fixtures/integration-test-model.gguf"),
                Device::Cpu,
            )?;

            // Phase 2: Apply quantization with bitnet-quantization
            let quantizer = I2SQuantizer::new();
            let mut quantized_weights = HashMap::new();

            for (name, weight) in &weights {
                if self.should_quantize_weight(name) {
                    let quantized = quantizer.quantize_tensor(
                        &BitNetTensor::from_candle_tensor(weight.clone())?
                    )?;
                    let dequantized = quantizer.dequantize_tensor(&quantized)?;
                    quantized_weights.insert(name.clone(), dequantized.to_candle_tensor()?);
                } else {
                    quantized_weights.insert(name.clone(), weight.clone());
                }
            }

            // Phase 3: Initialize inference with bitnet-inference
            let mut inference_engine = InferenceEngine::new(config.clone())?;
            inference_engine.load_weights(quantized_weights)?;

            // Phase 4: Perform test inference
            let test_input = vec![1, 2, 3, 4, 5]; // Simple token sequence
            let output = inference_engine.generate(
                &test_input,
                1.0, // temperature
                10,  // max_tokens
            )?;

            // Verify output is reasonable (not zeros, proper length, valid tokens)
            assert!(!output.is_empty(), "Inference produced no output");
            assert!(output.len() <= 15, "Inference produced too many tokens"); // 5 input + 10 max
            assert!(output.iter().all(|&token| token < config.model.vocab_size as u32),
                   "Invalid token in output");

            // Verify output is different from zero-initialized baseline
            let zero_baseline = vec![0u32; output.len()];
            assert_ne!(output, zero_baseline, "Inference output appears to be zero-initialized");
        }

        #[test]
        // AC5: Cross-validation with C++ reference implementation
        fn test_cpp_reference_cross_validation_ac5() {
            // Set deterministic environment
            std::env::set_var("BITNET_DETERMINISTIC", "1");
            std::env::set_var("BITNET_SEED", "42");
            std::env::set_var("RAYON_NUM_THREADS", "1");

            // Load model with our implementation
            let loader = GgufWeightLoader::new();
            let (config, weights) = loader.load_complete_model(
                Path::new("tests/fixtures/crossval-model.gguf"),
                Device::Cpu,
            )?;

            // Load C++ reference results
            let cpp_reference = self.load_cpp_reference_data("tests/fixtures/cpp_reference_outputs.json")?;

            // Initialize inference engine
            let mut inference_engine = InferenceEngine::new(config)?;
            inference_engine.load_weights(weights)?;

            // Run inference with same inputs as C++ reference
            for test_case in cpp_reference.test_cases {
                let output = inference_engine.generate(
                    &test_case.input_tokens,
                    test_case.temperature,
                    test_case.max_tokens,
                )?;

                // Compare with C++ reference output
                let similarity = self.calculate_output_similarity(&output, &test_case.expected_output);

                assert!(similarity >= 0.95,
                       "Cross-validation failed for test case '{}': similarity {:.4} < 0.95",
                       test_case.name, similarity);
            }
        }

        #[test]
        // AC6: Device-aware operation with CPU/GPU fallback
        fn test_device_aware_loading_ac6() {
            let test_cases = vec![
                (Device::Cpu, "CPU loading"),
                #[cfg(feature = "cuda")]
                (Device::Cuda(0), "CUDA loading"),
                #[cfg(feature = "metal")]
                (Device::Metal, "Metal loading"),
            ];

            for (device, description) in test_cases {
                let loader = GgufWeightLoader::new();

                // Test successful loading or graceful fallback
                match loader.load_complete_model(
                    Path::new("tests/fixtures/device-test-model.gguf"),
                    device.clone(),
                ) {
                    Ok((config, weights)) => {
                        // Verify weights are on correct device
                        self.verify_weights_on_device(&weights, &device)?;

                        // Verify all expected weights are present
                        self.verify_complete_weight_set(&weights, &config)?;

                        println!("✓ {} successful", description);
                    }
                    Err(error) => {
                        // For GPU devices, verify fallback to CPU works
                        if device != Device::Cpu {
                            println!("⚠ {} failed: {}, testing CPU fallback", description, error);

                            let fallback_result = loader.load_complete_model(
                                Path::new("tests/fixtures/device-test-model.gguf"),
                                Device::Cpu,
                            );

                            assert!(fallback_result.is_ok(),
                                   "CPU fallback failed after {} failure", description);
                        } else {
                            panic!("CPU loading should always succeed: {}", error);
                        }
                    }
                }
            }
        }

        #[test]
        // AC7: Memory-efficient loading with large models
        fn test_memory_efficient_loading_ac7() {
            // Test with different model sizes
            let test_models = vec![
                ("small", "tests/fixtures/small-model-100M.gguf", 1.0f32),
                ("medium", "tests/fixtures/medium-model-1B.gguf", 4.0f32),
                ("large", "tests/fixtures/large-model-7B.gguf", 28.0f32),
            ];

            for (size_name, model_path, expected_size_gb) in test_models {
                let memory_tracker = MemoryUsageTracker::new();
                let initial_memory = memory_tracker.current_memory_usage();

                let loader = GgufWeightLoader::new();
                let (config, weights) = loader.load_complete_model(
                    Path::new(model_path),
                    Device::Cpu,
                )?;

                let peak_memory = memory_tracker.peak_memory_usage();
                let final_memory = memory_tracker.current_memory_usage();

                // Calculate memory efficiency metrics
                let memory_overhead = (peak_memory - initial_memory) as f32 / 1_000_000_000.0; // GB
                let memory_multiplier = memory_overhead / expected_size_gb;

                // Verify memory efficiency targets
                assert!(memory_multiplier < 1.5,
                       "{} model memory multiplier {:.2} exceeds 1.5x limit",
                       size_name, memory_multiplier);

                // Verify memory is released after loading
                let final_overhead = (final_memory - initial_memory) as f32 / 1_000_000_000.0;
                let retention_ratio = final_overhead / memory_overhead;

                assert!(retention_ratio > 0.8,
                       "{} model retains too little memory ({:.2}), possible over-release",
                       size_name, retention_ratio);

                assert!(retention_ratio < 1.2,
                       "{} model retains too much memory ({:.2}), possible memory leak",
                       size_name, retention_ratio);

                println!("✓ {} model: {:.2}GB peak, {:.2}x multiplier, {:.1}% retained",
                        size_name, memory_overhead, memory_multiplier, retention_ratio * 100.0);
            }
        }
    }
}
```

### Error Handling Test Strategy

#### Comprehensive Error Scenario Testing

```rust
/// Error handling and edge case test suite
pub mod error_handling_tests {
    /// AC4: Graceful error handling with descriptive messages
    pub struct ErrorHandlingTestSuite;

    impl ErrorHandlingTestSuite {
        #[test]
        // AC4: Handle GGUF parsing errors gracefully with descriptive messages
        fn test_graceful_error_handling_ac4() {
            let error_test_cases = vec![
                ErrorTestCase {
                    name: "corrupted_header",
                    file_path: "tests/fixtures/corrupted-header.gguf",
                    expected_error: WeightLoadingError::GgufParsingError { .. },
                    expected_message_contains: "corrupted GGUF header",
                    recovery_possible: false,
                },
                ErrorTestCase {
                    name: "missing_tensor",
                    file_path: "tests/fixtures/incomplete-tensors.gguf",
                    expected_error: WeightLoadingError::TensorNotFound { .. },
                    expected_message_contains: "required tensor not found",
                    recovery_possible: true, // Can fallback to mock
                },
                ErrorTestCase {
                    name: "invalid_quantization",
                    file_path: "tests/fixtures/invalid-quantization.gguf",
                    expected_error: WeightLoadingError::UnsupportedQuantization { .. },
                    expected_message_contains: "unsupported quantization format",
                    recovery_possible: true, // Can disable quantization
                },
                ErrorTestCase {
                    name: "memory_exhaustion",
                    file_path: "tests/fixtures/huge-model-simulation.gguf",
                    expected_error: WeightLoadingError::OutOfMemory { .. },
                    expected_message_contains: "insufficient memory",
                    recovery_possible: true, // Can enable progressive loading
                },
                ErrorTestCase {
                    name: "device_unavailable",
                    file_path: "tests/fixtures/normal-model.gguf",
                    expected_error: WeightLoadingError::DeviceError { .. },
                    expected_message_contains: "GPU device unavailable",
                    recovery_possible: true, // Can fallback to CPU
                },
            ];

            for test_case in error_test_cases {
                let loader = GgufWeightLoader::new();

                let result = if test_case.name == "device_unavailable" {
                    // Simulate GPU unavailable
                    loader.load_complete_model(Path::new(test_case.file_path), Device::Cuda(99))
                } else {
                    loader.load_complete_model(Path::new(test_case.file_path), Device::Cpu)
                };

                // Verify error occurs as expected
                assert!(result.is_err(), "Expected error for test case: {}", test_case.name);

                let error = result.unwrap_err();

                // Verify error type matches expectation
                assert!(std::mem::discriminant(&error) == std::mem::discriminant(&test_case.expected_error),
                       "Wrong error type for test case {}: got {:?}",
                       test_case.name, error);

                // Verify error message is descriptive
                let error_message = error.to_string();
                assert!(error_message.contains(test_case.expected_message_contains),
                       "Error message for {} doesn't contain expected text '{}': {}",
                       test_case.name, test_case.expected_message_contains, error_message);

                // Test recovery if applicable
                if test_case.recovery_possible {
                    let recovery_result = self.attempt_error_recovery(&error, Path::new(test_case.file_path));
                    assert!(recovery_result.is_ok(),
                           "Recovery failed for recoverable error in test case: {}",
                           test_case.name);
                }

                // Verify error provides helpful guidance
                if let Some(suggestion) = error.recovery_suggestion() {
                    assert!(!suggestion.is_empty(),
                           "Recovery suggestion is empty for test case: {}", test_case.name);
                    assert!(suggestion.len() > 10,
                           "Recovery suggestion too short for test case: {}", test_case.name);
                }

                println!("✓ {} error handling validated", test_case.name);
            }
        }

        /// Test error recovery mechanisms
        fn attempt_error_recovery(
            &self,
            error: &WeightLoadingError,
            file_path: &Path,
        ) -> Result<(BitNetConfig, HashMap<String, CandleTensor>)> {
            match error.category() {
                ErrorCategory::Device => {
                    // Try CPU fallback
                    let loader = GgufWeightLoader::new();
                    loader.load_complete_model(file_path, Device::Cpu)
                }
                ErrorCategory::Memory => {
                    // Try with progressive loading enabled
                    let loader = GgufWeightLoader::builder()
                        .memory_config(MemoryConfig {
                            enable_progressive_loading: true,
                            memory_limit: 4_000_000_000, // 4GB limit
                            ..Default::default()
                        })
                        .build();
                    loader.load_complete_model(file_path, Device::Cpu)
                }
                ErrorCategory::Quantization => {
                    // Try without quantization
                    let loader = GgufWeightLoader::builder()
                        .quantization_config(QuantizationConfig {
                            enable_i2s: false,
                            enable_tl1: false,
                            enable_tl2: false,
                            ..Default::default()
                        })
                        .build();
                    loader.load_complete_model(file_path, Device::Cpu)
                }
                _ => {
                    // Try with mock fallback
                    let loader = BackwardCompatibleLoader::with_auto_fallback();
                    loader.load_gguf_compatible(file_path, Device::Cpu)
                }
            }
        }
    }

    struct ErrorTestCase {
        name: &'static str,
        file_path: &'static str,
        expected_error: WeightLoadingError,
        expected_message_contains: &'static str,
        recovery_possible: bool,
    }
}
```

### Performance and Regression Testing

#### Automated Performance Validation

```rust
/// Performance regression testing framework
pub mod performance_regression_tests {
    /// Performance benchmarks integrated with test suite
    pub struct PerformanceRegressionTestSuite {
        baseline_metrics: PerformanceBaseline,
        regression_thresholds: RegressionThresholds,
    }

    impl PerformanceRegressionTestSuite {
        #[test]
        #[ignore] // Run with --ignored for performance testing
        fn test_loading_performance_regression() {
            let test_models = vec![
                ("bitnet-160m", "tests/fixtures/bitnet-160m.gguf"),
                ("bitnet-1b", "tests/fixtures/bitnet-1b.gguf"),
                ("bitnet-7b", "tests/fixtures/bitnet-7b.gguf"),
            ];

            for (model_name, model_path) in test_models {
                let baseline = self.baseline_metrics.get_model_baseline(model_name)
                    .expect(&format!("No baseline for model: {}", model_name));

                // Measure current performance
                let performance = self.measure_loading_performance(Path::new(model_path))?;

                // Check for regressions
                let loading_regression = (performance.loading_time_seconds - baseline.loading_time_seconds)
                    / baseline.loading_time_seconds;

                assert!(loading_regression < self.regression_thresholds.max_loading_time_regression,
                       "Loading time regression for {}: {:.2}% increase (threshold: {:.2}%)",
                       model_name, loading_regression * 100.0,
                       self.regression_thresholds.max_loading_time_regression * 100.0);

                let memory_regression = (performance.peak_memory_multiplier - baseline.peak_memory_multiplier)
                    / baseline.peak_memory_multiplier;

                assert!(memory_regression < self.regression_thresholds.max_memory_usage_regression,
                       "Memory usage regression for {}: {:.2}% increase (threshold: {:.2}%)",
                       model_name, memory_regression * 100.0,
                       self.regression_thresholds.max_memory_usage_regression * 100.0);

                // Verify performance targets are still met
                assert!(performance.loading_time_seconds < baseline.max_acceptable_time,
                       "Loading time for {} exceeds maximum acceptable time: {:.2}s > {:.2}s",
                       model_name, performance.loading_time_seconds, baseline.max_acceptable_time);

                println!("✓ {} performance validated: {:.2}s loading, {:.2}x memory",
                        model_name, performance.loading_time_seconds, performance.peak_memory_multiplier);
            }
        }

        #[test]
        #[ignore] // GPU performance testing
        #[cfg(feature = "cuda")]
        fn test_gpu_performance_targets() {
            if !bitnet_kernels::gpu::cuda::is_cuda_available() {
                eprintln!("⚠ CUDA not available, skipping GPU performance tests");
                return;
            }

            let loader = GgufWeightLoader::new();
            let model_path = Path::new("tests/fixtures/gpu-test-model.gguf");

            // Measure CPU baseline
            let cpu_start = std::time::Instant::now();
            let (config, cpu_weights) = loader.load_complete_model(model_path, Device::Cpu)?;
            let cpu_time = cpu_start.elapsed();

            // Measure GPU performance
            let gpu_start = std::time::Instant::now();
            let (_, gpu_weights) = loader.load_complete_model(model_path, Device::Cuda(0))?;
            let gpu_time = gpu_start.elapsed();

            // Verify GPU transfer overhead is acceptable
            let gpu_overhead = gpu_time.as_secs_f64() - cpu_time.as_secs_f64();
            let overhead_percentage = gpu_overhead / cpu_time.as_secs_f64();

            assert!(overhead_percentage < 0.20, // 20% overhead limit
                   "GPU transfer overhead too high: {:.1}% (limit: 20%)",
                   overhead_percentage * 100.0);

            // Verify weights are actually on GPU
            for (name, weight) in &gpu_weights {
                match weight.device() {
                    candle_core::Device::Cuda(_) => {},
                    _ => panic!("Weight {} not placed on GPU device", name),
                }
            }

            println!("✓ GPU performance validated: {:.2}s total, {:.2}s overhead ({:.1}%)",
                    gpu_time.as_secs_f64(), gpu_overhead, overhead_percentage * 100.0);
        }
    }
}
```

## Test Execution Strategy

### Test Organization and Execution

#### Test Suite Execution Workflow

```bash
#!/bin/bash
# Test execution workflow for GGUF weight loading

echo "🧪 GGUF Weight Loading Test Suite"
echo "================================="

# Phase 1: Unit tests with AC tags
echo "Phase 1: Unit Tests (AC1-AC10)"
cargo test --no-default-features --lib --no-default-features --features cpu \
    test_.*_ac[0-9]+ \
    --test-threads=1 \
    -- --nocapture

# Phase 2: Integration tests
echo "Phase 2: Integration Tests"
cargo test --no-default-features --test integration_tests --no-default-features --features cpu \
    --test-threads=1 \
    -- --nocapture

# Phase 3: Cross-validation tests (if C++ reference available)
if [[ -n "$BITNET_CPP_REFERENCE" ]]; then
    echo "Phase 3: Cross-Validation Tests"
    export BITNET_DETERMINISTIC=1
    export BITNET_SEED=42
    export RAYON_NUM_THREADS=1

    cargo test --no-default-features --test crossval_tests --no-default-features --features cpu \
        test_cpp_.*_ac5 \
        --test-threads=1 \
        -- --nocapture
fi

# Phase 4: GPU tests (if CUDA available)
if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
    echo "Phase 4: GPU Tests"
    cargo test --no-default-features --test gpu_tests --no-default-features --features gpu,cuda \
        test_.*_ac6 \
        --test-threads=1 \
        -- --nocapture
fi

# Phase 5: Performance tests (optional)
if [[ "$RUN_PERF_TESTS" == "1" ]]; then
    echo "Phase 5: Performance Tests"
    cargo test --no-default-features --test performance_tests --release --no-default-features --features cpu \
        --test-threads=1 \
        -- --ignored --nocapture
fi

# Phase 6: Error handling tests
echo "Phase 6: Error Handling Tests"
cargo test --no-default-features --test error_tests --no-default-features --features cpu \
    test_.*_ac4 \
    --test-threads=1 \
    -- --nocapture

echo "✅ All test phases completed!"
```

#### Continuous Integration Test Matrix

```yaml
# CI test matrix for comprehensive validation
name: GGUF Weight Loading Tests

on: [push, pull_request]

jobs:
  test-matrix:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        rust: [stable, beta]
        features:
          - "cpu"
          - "gpu,cuda"
          - "ffi"
        include:
          - os: ubuntu-latest
            rust: stable
            features: "cpu"
            run_crossval: true
          - os: ubuntu-latest
            rust: stable
            features: "gpu,cuda"
            run_gpu_tests: true

    runs-on: ${{ matrix.os }}

    steps:
    - uses: actions/checkout@v3

    - name: Install Rust ${{ matrix.rust }}
      uses: dtolnay/rust-toolchain@master
      with:
        toolchain: ${{ matrix.rust }}

    - name: Download test fixtures
      run: |
        cargo run -p xtask -- download-test-fixtures

    - name: Run AC-tagged unit tests
      run: |
        cargo test --no-default-features --workspace --no-default-features --features ${{ matrix.features }} \
          test_.*_ac[0-9]+ \
          --test-threads=1

    - name: Run integration tests
      run: |
        cargo test --no-default-features --workspace --no-default-features --features ${{ matrix.features }} \
          --test integration_tests

    - name: Run cross-validation tests
      if: matrix.run_crossval
      env:
        BITNET_CPP_REFERENCE: "tests/fixtures/cpp_reference/"
        BITNET_DETERMINISTIC: "1"
        BITNET_SEED: "42"
      run: |
        cargo test --no-default-features --workspace --no-default-features --features ${{ matrix.features }} \
          test_cpp_.*_ac5

    - name: Run GPU-specific tests
      if: matrix.run_gpu_tests
      run: |
        cargo test --no-default-features --workspace --no-default-features --features ${{ matrix.features }} \
          test_.*_ac6 \
          --test-threads=1
```

## Test Data Management

### Test Fixture Strategy

#### Comprehensive Test Model Collection

```rust
/// Test fixture management for comprehensive testing
pub mod test_fixtures {
    /// Test model collection covering various architectures and sizes
    pub struct TestModelCollection {
        models: Vec<TestModel>,
        fixture_path: PathBuf,
    }

    pub struct TestModel {
        pub name: String,
        pub architecture: ModelArchitecture,
        pub size_category: ModelSizeCategory,
        pub quantization_format: Option<QuantizationType>,
        pub file_path: PathBuf,
        pub expected_properties: ModelProperties,
        pub test_scenarios: Vec<TestScenario>,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum ModelSizeCategory {
        Tiny,    // <100M parameters
        Small,   // 100M-1B parameters
        Medium,  // 1B-10B parameters
        Large,   // >10B parameters
    }

    pub struct ModelProperties {
        pub num_layers: usize,
        pub hidden_size: usize,
        pub vocab_size: usize,
        pub intermediate_size: usize,
        pub num_attention_heads: usize,
        pub has_bias: bool,
        pub activation: String,
    }

    impl TestModelCollection {
        pub fn standard_collection() -> Self {
            let models = vec![
                // BitNet models
                TestModel {
                    name: "bitnet-tiny-test".to_string(),
                    architecture: ModelArchitecture::BitNet,
                    size_category: ModelSizeCategory::Tiny,
                    quantization_format: Some(QuantizationType::I2S),
                    file_path: PathBuf::from("tests/fixtures/bitnet-tiny-test.gguf"),
                    expected_properties: ModelProperties {
                        num_layers: 12,
                        hidden_size: 768,
                        vocab_size: 32000,
                        intermediate_size: 3072,
                        num_attention_heads: 12,
                        has_bias: false,
                        activation: "silu".to_string(),
                    },
                    test_scenarios: vec![
                        TestScenario::CompleteLoading,
                        TestScenario::QuantizationAccuracy,
                        TestScenario::CrossValidation,
                        TestScenario::DeviceTransfer,
                    ],
                },

                // LLaMA compatibility model
                TestModel {
                    name: "llama-compat-test".to_string(),
                    architecture: ModelArchitecture::LLaMA,
                    size_category: ModelSizeCategory::Small,
                    quantization_format: Some(QuantizationType::TL1),
                    file_path: PathBuf::from("tests/fixtures/llama-compat-test.gguf"),
                    expected_properties: ModelProperties {
                        num_layers: 32,
                        hidden_size: 4096,
                        vocab_size: 32000,
                        intermediate_size: 11008,
                        num_attention_heads: 32,
                        has_bias: false,
                        activation: "silu".to_string(),
                    },
                    test_scenarios: vec![
                        TestScenario::CompleteLoading,
                        TestScenario::BackwardCompatibility,
                        TestScenario::MemoryEfficiency,
                    ],
                },

                // Error testing models
                TestModel {
                    name: "corrupted-header-test".to_string(),
                    architecture: ModelArchitecture::BitNet,
                    size_category: ModelSizeCategory::Tiny,
                    quantization_format: None,
                    file_path: PathBuf::from("tests/fixtures/corrupted-header.gguf"),
                    expected_properties: ModelProperties::default(),
                    test_scenarios: vec![
                        TestScenario::ErrorHandling,
                        TestScenario::GracefulFailure,
                    ],
                },

                // Performance testing models
                TestModel {
                    name: "performance-large-test".to_string(),
                    architecture: ModelArchitecture::BitNet,
                    size_category: ModelSizeCategory::Large,
                    quantization_format: Some(QuantizationType::I2S),
                    file_path: PathBuf::from("tests/fixtures/performance-large.gguf"),
                    expected_properties: ModelProperties {
                        num_layers: 80,
                        hidden_size: 8192,
                        vocab_size: 32000,
                        intermediate_size: 28672,
                        num_attention_heads: 64,
                        has_bias: false,
                        activation: "silu".to_string(),
                    },
                    test_scenarios: vec![
                        TestScenario::MemoryEfficiency,
                        TestScenario::LoadingPerformance,
                        TestScenario::ProgressiveLoading,
                    ],
                },
            ];

            Self {
                models,
                fixture_path: PathBuf::from("tests/fixtures"),
            }
        }

        /// Download or generate test fixtures as needed
        pub fn ensure_test_fixtures(&self) -> Result<()> {
            for model in &self.models {
                if !model.file_path.exists() {
                    self.create_or_download_fixture(model)?;
                }
            }
            Ok(())
        }

        /// Create synthetic test fixture for testing purposes
        fn create_or_download_fixture(&self, model: &TestModel) -> Result<()> {
            match model.name.as_str() {
                "corrupted-header-test" => self.create_corrupted_header_fixture(&model.file_path),
                "bitnet-tiny-test" => self.create_synthetic_bitnet_model(&model.file_path, &model.expected_properties),
                _ => {
                    tracing::warn!("Test fixture {} not available, skipping", model.name);
                    Ok(())
                }
            }
        }

        /// Create synthetic BitNet model for testing
        fn create_synthetic_bitnet_model(
            &self,
            path: &Path,
            properties: &ModelProperties,
        ) -> Result<()> {
            use bitnet_models::gguf_writer::GgufModelWriter;

            let writer = GgufModelWriter::new();

            // Create synthetic weights with realistic distributions
            let mut weights = HashMap::new();

            // Token embeddings
            weights.insert(
                "token_embd.weight".to_string(),
                self.create_synthetic_tensor(vec![properties.vocab_size, properties.hidden_size], TensorDistribution::Xavier),
            );

            // Output projection
            weights.insert(
                "output.weight".to_string(),
                self.create_synthetic_tensor(vec![properties.hidden_size, properties.vocab_size], TensorDistribution::Xavier),
            );

            // Transformer layers
            for layer in 0..properties.num_layers {
                let prefix = format!("blk.{}", layer);

                // Attention weights
                weights.insert(
                    format!("{}.attn_q.weight", prefix),
                    self.create_synthetic_tensor(vec![properties.hidden_size, properties.hidden_size], TensorDistribution::HeNormal),
                );
                weights.insert(
                    format!("{}.attn_k.weight", prefix),
                    self.create_synthetic_tensor(vec![properties.hidden_size, properties.hidden_size], TensorDistribution::HeNormal),
                );
                weights.insert(
                    format!("{}.attn_v.weight", prefix),
                    self.create_synthetic_tensor(vec![properties.hidden_size, properties.hidden_size], TensorDistribution::HeNormal),
                );
                weights.insert(
                    format!("{}.attn_output.weight", prefix),
                    self.create_synthetic_tensor(vec![properties.hidden_size, properties.hidden_size], TensorDistribution::HeNormal),
                );

                // Feed-forward weights
                weights.insert(
                    format!("{}.ffn_gate.weight", prefix),
                    self.create_synthetic_tensor(vec![properties.intermediate_size, properties.hidden_size], TensorDistribution::HeNormal),
                );
                weights.insert(
                    format!("{}.ffn_up.weight", prefix),
                    self.create_synthetic_tensor(vec![properties.intermediate_size, properties.hidden_size], TensorDistribution::HeNormal),
                );
                weights.insert(
                    format!("{}.ffn_down.weight", prefix),
                    self.create_synthetic_tensor(vec![properties.hidden_size, properties.intermediate_size], TensorDistribution::HeNormal),
                );

                // Normalization weights
                weights.insert(
                    format!("{}.attn_norm.weight", prefix),
                    self.create_synthetic_tensor(vec![properties.hidden_size], TensorDistribution::Ones),
                );
                weights.insert(
                    format!("{}.ffn_norm.weight", prefix),
                    self.create_synthetic_tensor(vec![properties.hidden_size], TensorDistribution::Ones),
                );
            }

            // Output norm
            weights.insert(
                "output_norm.weight".to_string(),
                self.create_synthetic_tensor(vec![properties.hidden_size], TensorDistribution::Ones),
            );

            writer.write_model_to_file(path, &weights, properties)?;
            Ok(())
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum TestScenario {
        CompleteLoading,
        QuantizationAccuracy,
        CrossValidation,
        DeviceTransfer,
        BackwardCompatibility,
        MemoryEfficiency,
        ErrorHandling,
        GracefulFailure,
        LoadingPerformance,
        ProgressiveLoading,
    }
}
```

## Conclusion

This comprehensive integration architecture and test strategy provides:

**Integration Architecture:**
- **Workspace Integration**: Seamless coordination across BitNet.rs crates
- **Device-Aware Operations**: CPU/GPU support with automatic fallback
- **Backward Compatibility**: Preservation of mock loading functionality
- **Cross-Crate Communication**: Well-defined protocols between components

**Test Strategy:**
- **TDD Framework**: Systematic AC-tagged tests for all acceptance criteria
- **Integration Testing**: Multi-crate end-to-end validation
- **Error Handling**: Comprehensive error scenario coverage
- **Performance Testing**: Automated regression detection and benchmarking
- **Cross-Validation**: C++ reference compatibility verification

**Test Execution:**
- **Automated CI/CD**: Multi-platform, multi-configuration testing
- **Test Fixtures**: Comprehensive model collection for various scenarios
- **Performance Monitoring**: Continuous regression detection
- **Documentation Integration**: Test coverage aligned with specifications

The architecture ensures that Issue #159 (GGUF model weight loading) is implemented with:
- **Quality Assurance**: >95% test coverage with AC-tagged tests
- **Performance Validation**: Automated benchmarking and regression detection
- **Cross-Platform Support**: Linux, macOS, Windows compatibility
- **Production Readiness**: Error handling, fallback mechanisms, and monitoring

This systematic approach guarantees successful implementation of real GGUF weight loading while maintaining BitNet.rs's high standards for performance, reliability, and maintainability.

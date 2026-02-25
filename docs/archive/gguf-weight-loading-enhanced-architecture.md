# Enhanced GGUF Weight Loading Architecture - Device-Aware Implementation

## Executive Summary

This enhanced architectural specification builds upon the existing GGUF weight loading framework to provide production-ready device-aware tensor placement, progressive loading, and comprehensive validation systems. This addresses the gaps identified in the current implementation for Issue #159 neural network inference requirements.

## Enhanced Architecture Components

### 1. Device-Aware Tensor Placement System

#### Device Strategy Interface

```rust
/// Advanced device placement strategy for GGUF weight loading
pub trait DevicePlacementStrategy {
    /// Determine optimal device placement for tensor
    ///
    /// # Acceptance Criteria Coverage
    /// * AC6: CPU/GPU feature flags with device-aware tensor placement
    /// * AC7: Memory-efficient loading with progressive loading for models >4GB
    fn place_tensor(
        &self,
        tensor_info: &TensorInfo,
        available_memory: &DeviceMemoryInfo,
        model_config: &ModelConfig,
    ) -> Result<TensorPlacement>;

    /// Validate memory requirements before loading
    fn validate_memory_requirements(
        &self,
        total_size_bytes: u64,
        tensor_count: usize,
    ) -> Result<MemoryValidationResult>;

    /// Handle device placement failures with graceful fallback
    fn handle_placement_failure(
        &self,
        error: &DevicePlacementError,
        fallback_options: &[Device],
    ) -> Result<Device>;
}

/// GPU-optimized placement strategy
pub struct GpuOptimizedPlacement {
    /// Maximum GPU memory utilization (0.0-1.0)
    pub max_gpu_utilization: f32,
    /// Enable mixed precision (FP16/BF16)
    pub mixed_precision: bool,
    /// Progressive offloading strategy
    pub progressive_offload: bool,
    /// NUMA-aware CPU fallback
    pub numa_aware_fallback: bool,
}

impl DevicePlacementStrategy for GpuOptimizedPlacement {
    fn place_tensor(
        &self,
        tensor_info: &TensorInfo,
        available_memory: &DeviceMemoryInfo,
        model_config: &ModelConfig,
    ) -> Result<TensorPlacement> {
        // Priority-based placement logic:
        // 1. Critical tensors (embeddings, attention weights) -> GPU
        // 2. Large tensors (feedforward weights) -> GPU if memory available
        // 3. Normalization tensors -> GPU (small memory footprint)
        // 4. Fallback to CPU with memory-mapped access

        let tensor_size = tensor_info.memory_footprint_bytes();
        let gpu_memory_remaining = available_memory.gpu_free_bytes();

        if self.is_critical_tensor(&tensor_info.name) {
            if gpu_memory_remaining >= tensor_size {
                return Ok(TensorPlacement::Gpu {
                    device_index: 0,
                    mixed_precision: self.mixed_precision,
                    memory_pool: GpuMemoryPool::HighPriority,
                });
            }
        }

        if tensor_size <= gpu_memory_remaining * self.max_gpu_utilization as u64 {
            Ok(TensorPlacement::Gpu {
                device_index: 0,
                mixed_precision: self.mixed_precision,
                memory_pool: GpuMemoryPool::Standard,
            })
        } else {
            Ok(TensorPlacement::Cpu {
                memory_mapped: tensor_size >= 1024 * 1024, // 1MB threshold
                numa_policy: if self.numa_aware_fallback {
                    NumaPolicy::LocalAlloc
                } else {
                    NumaPolicy::Default
                },
            })
        }
    }

    fn validate_memory_requirements(
        &self,
        total_size_bytes: u64,
        tensor_count: usize,
    ) -> Result<MemoryValidationResult> {
        let system_memory = get_system_memory_info()?;
        let gpu_memory = get_gpu_memory_info()?;

        // Calculate memory overhead for GGUF parsing and validation
        let parsing_overhead = (total_size_bytes as f32 * 0.1) as u64; // 10% overhead
        let validation_overhead = (tensor_count * 1024) as u64; // 1KB per tensor metadata

        let total_required = total_size_bytes + parsing_overhead + validation_overhead;

        // Check if model fits in available memory with safety margin
        let available_memory = system_memory.available_bytes + gpu_memory.map_or(0, |g| g.free_bytes);
        let safety_margin = 0.15; // 15% safety margin

        if total_required > (available_memory as f32 * (1.0 - safety_margin)) as u64 {
            return Ok(MemoryValidationResult {
                status: ValidationStatus::InsufficientMemory,
                required_bytes: total_required,
                available_bytes: available_memory,
                recommendations: vec![
                    "Consider using progressive loading".to_string(),
                    "Enable CPU-only mode for large models".to_string(),
                    "Use memory-mapped file access".to_string(),
                ],
            });
        }

        Ok(MemoryValidationResult {
            status: ValidationStatus::Valid,
            required_bytes: total_required,
            available_bytes: available_memory,
            recommendations: vec![],
        })
    }
}

/// Device memory information for placement decisions
#[derive(Debug, Clone)]
pub struct DeviceMemoryInfo {
    pub cpu_total_bytes: u64,
    pub cpu_available_bytes: u64,
    pub gpu_devices: Vec<GpuMemoryInfo>,
    pub numa_topology: Option<NumaTopology>,
}

#[derive(Debug, Clone)]
pub struct GpuMemoryInfo {
    pub device_index: usize,
    pub total_bytes: u64,
    pub free_bytes: u64,
    pub cuda_capability: Option<(u32, u32)>,
    pub memory_bandwidth_gbps: Option<f32>,
}

/// Tensor placement result with device-specific optimizations
#[derive(Debug, Clone)]
pub enum TensorPlacement {
    Cpu {
        memory_mapped: bool,
        numa_policy: NumaPolicy,
    },
    Gpu {
        device_index: usize,
        mixed_precision: bool,
        memory_pool: GpuMemoryPool,
    },
    Hybrid {
        primary_device: Device,
        overflow_device: Device,
        split_threshold_bytes: u64,
    },
}
```

### 2. Progressive Loading Implementation

#### Streaming Weight Loader

```rust
/// Progressive GGUF weight loader for large models
pub struct ProgressiveGgufLoader {
    /// File handle for streaming access
    file_handle: File,
    /// Memory map for metadata and small tensors
    metadata_mmap: Mmap,
    /// Loading strategy configuration
    loading_strategy: ProgressiveLoadingStrategy,
    /// Device placement manager
    placement_manager: Box<dyn DevicePlacementStrategy>,
    /// Progress callback for monitoring
    progress_callback: Option<Box<dyn Fn(LoadingProgress) + Send + Sync>>,
}

impl ProgressiveGgufLoader {
    /// Create new progressive loader with configuration
    ///
    /// # Acceptance Criteria Coverage
    /// * AC7: Progressive loading for models >4GB
    /// * AC7: Memory footprint <150% of model size during loading
    pub fn new(
        path: impl AsRef<Path>,
        strategy: ProgressiveLoadingStrategy,
        placement_manager: Box<dyn DevicePlacementStrategy>,
    ) -> Result<Self> {
        let file = File::open(path)?;
        let file_size = file.metadata()?.len();

        // Memory-map only metadata section for efficiency
        let metadata_size = Self::calculate_metadata_size(&file)?;
        let metadata_mmap = unsafe {
            MmapOptions::new()
                .len(metadata_size)
                .map(&file)?
        };

        Ok(Self {
            file_handle: file,
            metadata_mmap,
            loading_strategy: strategy,
            placement_manager,
            progress_callback: None,
        })
    }

    /// Load model weights progressively with memory optimization
    ///
    /// # Implementation Strategy
    /// 1. Parse metadata and create loading plan
    /// 2. Load critical tensors first (embeddings, attention)
    /// 3. Stream remaining tensors based on priority
    /// 4. Validate loaded tensors incrementally
    /// 5. Free temporary buffers immediately
    pub async fn load_progressive(&mut self) -> Result<BitNetModel> {
        let loading_plan = self.create_loading_plan().await?;
        let mut model_builder = BitNetModelBuilder::new();

        // Phase 1: Load critical tensors (embeddings, attention weights)
        for critical_tensor in &loading_plan.critical_tensors {
            let tensor = self.load_tensor_streaming(critical_tensor).await?;
            model_builder.add_tensor(critical_tensor.name.clone(), tensor)?;

            if let Some(callback) = &self.progress_callback {
                callback(LoadingProgress {
                    phase: LoadingPhase::Critical,
                    tensors_loaded: model_builder.tensor_count(),
                    total_tensors: loading_plan.total_tensor_count(),
                    memory_used_bytes: model_builder.memory_footprint(),
                });
            }
        }

        // Phase 2: Load feedforward layers progressively
        for layer_batch in loading_plan.layer_batches.chunks(self.loading_strategy.batch_size) {
            for layer_tensor in layer_batch {
                let tensor = self.load_tensor_streaming(layer_tensor).await?;
                model_builder.add_tensor(layer_tensor.name.clone(), tensor)?;
            }

            // Trigger garbage collection after each batch
            if self.loading_strategy.gc_after_batch {
                self.force_garbage_collection().await?;
            }
        }

        // Phase 3: Load remaining tensors (normalization, output)
        for remaining_tensor in &loading_plan.remaining_tensors {
            let tensor = self.load_tensor_streaming(remaining_tensor).await?;
            model_builder.add_tensor(remaining_tensor.name.clone(), tensor)?;
        }

        model_builder.build()
    }

    /// Load individual tensor with streaming and device placement
    async fn load_tensor_streaming(&mut self, tensor_info: &TensorLoadInfo) -> Result<BitNetTensor> {
        // Determine optimal device placement
        let memory_info = get_current_memory_info()?;
        let placement = self.placement_manager.place_tensor(
            &tensor_info.metadata,
            &memory_info,
            &self.loading_strategy.model_config,
        )?;

        // Create buffer on target device
        let buffer = self.create_device_buffer(&placement, tensor_info.size_bytes).await?;

        // Stream tensor data from file
        self.file_handle.seek(SeekFrom::Start(tensor_info.file_offset))?;

        match tensor_info.quantization_type {
            QuantizationType::I2S => {
                self.load_i2s_tensor_streaming(tensor_info, buffer).await
            }
            QuantizationType::F32 => {
                self.load_f32_tensor_streaming(tensor_info, buffer).await
            }
            QuantizationType::F16 => {
                self.load_f16_tensor_streaming(tensor_info, buffer).await
            }
            _ => Err(anyhow::anyhow!("Unsupported quantization type: {:?}", tensor_info.quantization_type))
        }
    }

    /// Force garbage collection to free temporary buffers
    async fn force_garbage_collection(&self) -> Result<()> {
        // Platform-specific garbage collection
        #[cfg(feature = "gpu")]
        {
            // Clear GPU memory caches
            candle_core::cuda::clear_cache()?;
        }

        // Force Rust garbage collection
        std::hint::black_box(());

        Ok(())
    }
}

/// Progressive loading strategy configuration
#[derive(Debug, Clone)]
pub struct ProgressiveLoadingStrategy {
    /// Batch size for layer loading
    pub batch_size: usize,
    /// Enable garbage collection after each batch
    pub gc_after_batch: bool,
    /// Memory pressure threshold (0.0-1.0)
    pub memory_pressure_threshold: f32,
    /// Enable tensor validation during loading
    pub validate_during_loading: bool,
    /// Model configuration for validation
    pub model_config: ModelConfig,
}

impl Default for ProgressiveLoadingStrategy {
    fn default() -> Self {
        Self {
            batch_size: 4, // Load 4 layers at a time
            gc_after_batch: true,
            memory_pressure_threshold: 0.8, // 80% memory usage triggers optimization
            validate_during_loading: true,
            model_config: ModelConfig::default(),
        }
    }
}
```

### 3. Enhanced Cross-Validation Framework

#### C++ Reference Validation

```rust
/// Enhanced cross-validation framework for GGUF weight loading
pub struct GgufCrossValidator {
    /// C++ reference implementation bridge
    cpp_bridge: CppReferenceBridge,
    /// Validation tolerances
    tolerances: ValidationTolerances,
    /// Deterministic configuration
    deterministic_config: DeterministicConfig,
}

impl GgufCrossValidator {
    /// Create new cross-validator with C++ reference
    ///
    /// # Acceptance Criteria Coverage
    /// * AC5: Cross-validation framework with numerical tolerance <1e-5
    /// * AC5: Deterministic inference validation using BITNET_DETERMINISTIC=1 BITNET_SEED=42
    pub fn new(cpp_library_path: impl AsRef<Path>) -> Result<Self> {
        let cpp_bridge = CppReferenceBridge::load(cpp_library_path)?;

        Ok(Self {
            cpp_bridge,
            tolerances: ValidationTolerances::default(),
            deterministic_config: DeterministicConfig::from_env(),
        })
    }

    /// Validate loaded weights against C++ reference implementation
    pub async fn validate_weights_against_cpp(
        &self,
        rust_model: &BitNetModel,
        gguf_path: impl AsRef<Path>,
    ) -> Result<CrossValidationResult> {
        // Load model in C++ reference implementation
        let cpp_model = self.cpp_bridge.load_model(gguf_path.as_ref())?;

        let mut validation_results = Vec::new();

        // Validate each tensor individually
        for (tensor_name, rust_tensor) in rust_model.tensors() {
            let cpp_tensor = cpp_model.get_tensor(tensor_name)?;

            let tensor_result = self.validate_tensor_parity(
                tensor_name,
                rust_tensor,
                &cpp_tensor,
            ).await?;

            validation_results.push(tensor_result);
        }

        // Validate full inference pipeline
        let inference_result = self.validate_inference_parity(
            rust_model,
            &cpp_model,
        ).await?;

        Ok(CrossValidationResult {
            tensor_validations: validation_results,
            inference_validation: inference_result,
            overall_status: self.determine_overall_status(&validation_results, &inference_result),
        })
    }

    /// Validate individual tensor against C++ reference
    async fn validate_tensor_parity(
        &self,
        tensor_name: &str,
        rust_tensor: &BitNetTensor,
        cpp_tensor: &CppTensor,
    ) -> Result<TensorValidationResult> {
        // Shape validation
        if rust_tensor.shape() != cpp_tensor.shape() {
            return Ok(TensorValidationResult {
                tensor_name: tensor_name.to_string(),
                status: ValidationStatus::Failed,
                error: Some(format!(
                    "Shape mismatch: Rust {:?} vs C++ {:?}",
                    rust_tensor.shape(),
                    cpp_tensor.shape()
                )),
                max_absolute_error: None,
                max_relative_error: None,
            });
        }

        // Numerical validation
        let rust_data = rust_tensor.to_f32_vec()?;
        let cpp_data = cpp_tensor.to_f32_vec()?;

        let (max_abs_error, max_rel_error) = self.calculate_numerical_errors(&rust_data, &cpp_data);

        let status = if max_abs_error <= self.tolerances.absolute_tolerance &&
                        max_rel_error <= self.tolerances.relative_tolerance {
            ValidationStatus::Passed
        } else {
            ValidationStatus::Failed
        };

        Ok(TensorValidationResult {
            tensor_name: tensor_name.to_string(),
            status,
            error: None,
            max_absolute_error: Some(max_abs_error),
            max_relative_error: Some(max_rel_error),
        })
    }

    /// Calculate numerical errors between Rust and C++ tensors
    fn calculate_numerical_errors(&self, rust_data: &[f32], cpp_data: &[f32]) -> (f32, f32) {
        let mut max_abs_error = 0.0f32;
        let mut max_rel_error = 0.0f32;

        for (rust_val, cpp_val) in rust_data.iter().zip(cpp_data.iter()) {
            let abs_error = (rust_val - cpp_val).abs();
            max_abs_error = max_abs_error.max(abs_error);

            if cpp_val.abs() > 1e-8 {
                let rel_error = abs_error / cpp_val.abs();
                max_rel_error = max_rel_error.max(rel_error);
            }
        }

        (max_abs_error, max_rel_error)
    }
}

/// Validation tolerances for numerical comparison
#[derive(Debug, Clone)]
pub struct ValidationTolerances {
    /// Absolute tolerance for numerical comparison
    pub absolute_tolerance: f32,
    /// Relative tolerance for numerical comparison
    pub relative_tolerance: f32,
    /// Tolerance for quantized tensors (may be higher)
    pub quantized_tolerance: f32,
}

impl Default for ValidationTolerances {
    fn default() -> Self {
        Self {
            absolute_tolerance: 1e-5, // AC5 requirement
            relative_tolerance: 1e-4,
            quantized_tolerance: 1e-3, // Slightly higher for quantized tensors
        }
    }
}
```

### 4. Memory Efficiency Validation

#### Memory Footprint Monitoring

```rust
/// Memory efficiency validator for GGUF weight loading
pub struct MemoryEfficiencyValidator {
    /// Target memory efficiency ratio (≤1.5 as per AC7)
    target_efficiency_ratio: f32,
    /// Memory monitoring configuration
    monitoring_config: MemoryMonitoringConfig,
    /// Real-time memory tracking
    memory_tracker: MemoryTracker,
}

impl MemoryEfficiencyValidator {
    /// Create new memory efficiency validator
    ///
    /// # Acceptance Criteria Coverage
    /// * AC7: Memory footprint <150% of model size during loading
    /// * AC7: Zero-copy operations for memory-mapped file access
    pub fn new(target_ratio: Option<f32>) -> Self {
        Self {
            target_efficiency_ratio: target_ratio.unwrap_or(1.5), // 150% max
            monitoring_config: MemoryMonitoringConfig::default(),
            memory_tracker: MemoryTracker::new(),
        }
    }

    /// Monitor memory usage during GGUF loading
    pub async fn monitor_loading_efficiency<F, R>(
        &mut self,
        model_size_bytes: u64,
        loading_operation: F,
    ) -> Result<(R, MemoryEfficiencyReport)>
    where
        F: FnOnce() -> Result<R> + Send,
        R: Send,
    {
        let baseline_memory = self.memory_tracker.current_usage()?;
        self.memory_tracker.start_monitoring(Duration::from_millis(100))?;

        let start_time = Instant::now();
        let result = loading_operation()?;
        let loading_duration = start_time.elapsed();

        let peak_memory = self.memory_tracker.peak_usage()?;
        let final_memory = self.memory_tracker.current_usage()?;

        self.memory_tracker.stop_monitoring()?;

        let memory_overhead = peak_memory.saturating_sub(baseline_memory);
        let efficiency_ratio = memory_overhead as f32 / model_size_bytes as f32;

        let report = MemoryEfficiencyReport {
            model_size_bytes,
            baseline_memory_bytes: baseline_memory,
            peak_memory_bytes: peak_memory,
            final_memory_bytes: final_memory,
            memory_overhead_bytes: memory_overhead,
            efficiency_ratio,
            target_ratio: self.target_efficiency_ratio,
            meets_efficiency_target: efficiency_ratio <= self.target_efficiency_ratio,
            loading_duration,
            zero_copy_operations: self.count_zero_copy_operations(),
        };

        Ok((result, report))
    }

    /// Count zero-copy operations during loading
    fn count_zero_copy_operations(&self) -> usize {
        // Implementation would track mmap operations vs copy operations
        self.memory_tracker.zero_copy_count()
    }
}

/// Memory efficiency monitoring report
#[derive(Debug, Clone)]
pub struct MemoryEfficiencyReport {
    pub model_size_bytes: u64,
    pub baseline_memory_bytes: u64,
    pub peak_memory_bytes: u64,
    pub final_memory_bytes: u64,
    pub memory_overhead_bytes: u64,
    pub efficiency_ratio: f32,
    pub target_ratio: f32,
    pub meets_efficiency_target: bool,
    pub loading_duration: Duration,
    pub zero_copy_operations: usize,
}

impl MemoryEfficiencyReport {
    /// Generate efficiency summary for validation
    pub fn efficiency_summary(&self) -> String {
        format!(
            "Memory Efficiency Report:\n\
             Model Size: {:.2} MB\n\
             Peak Memory: {:.2} MB\n\
             Efficiency Ratio: {:.2} (target: ≤{:.2})\n\
             Status: {}\n\
             Zero-Copy Operations: {}\n\
             Loading Duration: {:.2}s",
            self.model_size_bytes as f32 / 1024.0 / 1024.0,
            self.peak_memory_bytes as f32 / 1024.0 / 1024.0,
            self.efficiency_ratio,
            self.target_ratio,
            if self.meets_efficiency_target { "PASS" } else { "FAIL" },
            self.zero_copy_operations,
            self.loading_duration.as_secs_f32()
        )
    }
}
```

## Integration with BitNet.rs Infrastructure

### Feature Flag Integration

```rust
/// Feature-aware GGUF loader factory
pub struct GgufLoaderFactory;

impl GgufLoaderFactory {
    /// Create loader based on enabled features
    ///
    /// # Feature Flag Discipline
    /// * `--no-default-features --features cpu`: CPU-only loading
    /// * `--no-default-features --features gpu`: GPU-accelerated loading
    /// * `--no-default-features`: Minimal loading (testing only)
    pub fn create_loader(config: &GgufLoadingConfig) -> Result<Box<dyn GgufLoader>> {
        #[cfg(feature = "gpu")]
        {
            if config.enable_gpu && is_gpu_available()? {
                return Ok(Box::new(GpuOptimizedGgufLoader::new(config)?));
            }
        }

        #[cfg(feature = "cpu")]
        {
            Ok(Box::new(CpuOptimizedGgufLoader::new(config)?))
        }

        #[cfg(not(any(feature = "cpu", feature = "gpu")))]
        {
            Ok(Box::new(MinimalGgufLoader::new(config)?))
        }
    }
}
```

## Validation and Testing Framework

### Integration Test Structure

```rust
#[cfg(test)]
mod integration_tests {
    use super::*;

    // AC1: Parse and load all transformer layer weights
    #[tokio::test]
    async fn test_comprehensive_weight_loading() {
        // AC:1 - Test loading all transformer weights including attention, feedforward, normalization
    }

    // AC2: Support quantization formats with ≥99% accuracy preservation
    #[tokio::test]
    async fn test_quantization_accuracy_preservation() {
        // AC:2 - Test I2_S, TL1, TL2 formats with accuracy validation
    }

    // AC5: Cross-validation framework with <1e-5 tolerance
    #[tokio::test]
    async fn test_cpp_reference_cross_validation() {
        // AC:5 - Test against C++ reference implementation
    }

    // AC6: CPU/GPU feature flags with device-aware tensor placement
    #[tokio::test]
    async fn test_device_aware_tensor_placement() {
        // AC:6 - Test automatic GPU/CPU selection and graceful fallback
    }

    // AC7: Memory-efficient loading with progressive loading
    #[tokio::test]
    async fn test_progressive_loading_efficiency() {
        // AC:7 - Test memory footprint <150% and progressive loading >4GB
    }
}
```

## Conclusion

This enhanced architectural specification provides the detailed implementation framework for production-ready GGUF weight loading in BitNet.rs. The architecture addresses all acceptance criteria from Issue #159 while maintaining compatibility with the existing neural network inference pipeline and BitNet.rs feature flag discipline.

**Key Enhancements:**
- **Device-Aware Placement**: Sophisticated GPU/CPU placement with memory optimization
- **Progressive Loading**: Streaming architecture for large models with memory efficiency
- **Enhanced Validation**: Comprehensive C++ reference cross-validation framework
- **Memory Monitoring**: Real-time efficiency tracking with actionable metrics
- **Feature Integration**: Seamless integration with BitNet.rs workspace structure

This specification enables meaningful neural network inference with production-ready performance characteristics.

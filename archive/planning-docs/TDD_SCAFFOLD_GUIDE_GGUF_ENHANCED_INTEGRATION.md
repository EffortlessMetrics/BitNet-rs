# TDD Scaffold Implementation Guide: GGUF Enhanced & Integration Tests

**Issue**: #159  
**Files**:
- `crates/bitnet-models/tests/gguf_weight_loading_property_tests_enhanced.rs`
- `crates/bitnet-models/tests/gguf_weight_loading_integration_tests.rs`
- `crates/bitnet-models/tests/gguf_weight_loading_device_aware_tests.rs`

**Total Scaffolds**: 3 #[ignore] tests + 27 TODO markers  
**Priority**: MEDIUM (Post-MVP optimization and integration phase)

---

## Overview

This guide covers three complementary test categories for GGUF weight loading:

1. **Enhanced Property Tests** (`property_tests_enhanced.rs`): Advanced property-based testing with higher-order statistics (skewness, kurtosis), block-level consistency validation, and cross-platform quantization validation
2. **Integration Tests** (`integration_tests.rs`): Cross-crate integration testing covering bitnet-models + quantization/kernels/inference/FFI/WASM pipelines
3. **Device-Aware Tests** (`device_aware_tests.rs`): CPU/GPU tensor placement, mixed precision support, automatic fallback, and cross-device consistency

**Test Relationship**:
- Enhanced property tests validate **quantization correctness** at fine-grained level
- Integration tests validate **cross-crate interoperability**
- Device-aware tests validate **device placement and memory management**

---

## Enhanced Property Tests

### Scaffold 1: property_cross_platform_quantization_consistency

**File**: `gguf_weight_loading_property_tests_enhanced.rs`  
**Lines**: 516-551  
**Status**: Stub (blocked by proptest compilation + C++ reference integration)  
**Priority**: MEDIUM (cross-validation infrastructure)

**Current Implementation**:
```rust
#[cfg(all(feature = "cpu", feature = "crossval"))]
proptest! {
    #[ignore = "TODO: Fix proptest compilation errors"]
    #[ignore] // Issue #159: TDD placeholder - cross-platform consistency implementation needed
    fn property_cross_platform_quantization_consistency(
        tensor_data in prop::collection::vec(-2.0f32..2.0f32, 128..512),
    ) {
        let quantizer = I2SQuantizer::new();
        let original_tensor = to_test_error(create_test_tensor_from_data(tensor_data.clone(), vec![tensor_data.len()]))?;

        // Perform Rust quantization
        let rust_quantized = to_test_error(quantizer.quantize(&original_tensor, &candle_core::Device::Cpu))?;
        let rust_dequantized = to_test_error(quantizer.dequantize(&rust_quantized, &candle_core::Device::Cpu))?;

        // TODO: Integrate with actual C++ reference implementation
        let cpp_reference_result = to_test_error(simulate_cpp_quantization(&original_tensor))?;

        // Property: Rust and C++ implementations should produce consistent results
        let consistency = to_test_error(calculate_cosine_similarity(&rust_dequantized, &cpp_reference_result))?;
        prop_assert!(
            consistency >= 0.999, // Very high consistency requirement for cross-validation
            "Cross-platform consistency {} below threshold 0.999",
            consistency
        );

        // Property: Numerical tolerance should be within specified bounds
        let numerical_difference = to_test_error(calculate_max_absolute_difference(&rust_dequantized, &cpp_reference_result))?;
        prop_assert!(
            numerical_difference < 1e-5,
            "Cross-platform numerical difference {} exceeds tolerance 1e-5",
            numerical_difference
        );
    }
}
```

**Acceptance Criteria**:
1. ✅ Rust I2S quantization produces deterministic results
2. ❌ C++ reference implementation integrated via FFI/crossval framework
3. ❌ Cosine similarity ≥ 0.999 between Rust and C++ implementations
4. ❌ Maximum absolute difference < 1e-5 for cross-platform consistency
5. ❌ Property test runs on 20 random tensor inputs (128-512 elements)

**Required Dependencies**:
- **Crates**: `bitnet-quantization` (I2SQuantizer), `crossval` framework, `proptest`
- **Feature Gates**: `cpu`, `crossval`
- **Helper Functions**:
  - `simulate_cpp_quantization()` (line 838-847): Replace mock with actual FFI call to C++ reference
  - `calculate_cosine_similarity()` (line 719-736): ✅ Already implemented
  - `calculate_max_absolute_difference()` (line 761-779): ✅ Already implemented
  - FFI bridge to C++ bitnet quantization (`bitnet_cpp_quantize_i2s()`)

**Blockers**:
- Issue #469: FFI build hygiene and C++ reference integration
- Issue #439: Feature gate consistency for crossval predicates
- Proptest compilation errors (likely generic lifetime issues)

**Implementation Steps**:
1. **Fix proptest compilation errors**:
   - Investigate generic lifetime issues in proptest macro expansion
   - Ensure `to_test_error()` conversion works correctly with proptest error types
   - Add explicit type annotations if needed

2. **Integrate C++ reference implementation**:
   ```rust
   // Replace simulate_cpp_quantization() with actual FFI call
   #[cfg(feature = "crossval")]
   fn cpp_reference_quantize(tensor: &BitNetTensor) -> Result<BitNetTensor> {
       use bitnet_ffi::cpp_bridge;
       let data = tensor.to_vec()?;
       let cpp_result = unsafe {
           cpp_bridge::bitnet_cpp_quantize_i2s(
               data.as_ptr(),
               data.len(),
               /* block_size */ 32,
           )
       };
       create_test_tensor_from_data(cpp_result, tensor.shape().to_vec())
   }
   ```

3. **Validate consistency thresholds**:
   - Run test with known-good GGUF weights
   - Adjust consistency threshold (0.999) based on empirical C++ reference behavior
   - Validate numerical tolerance (1e-5) is achievable for 2-bit quantization

4. **Add comprehensive logging**:
   ```rust
   println!("Cross-platform validation:");
   println!("  - Rust cosine similarity: {:.6}", consistency);
   println!("  - Max absolute difference: {:.6}", numerical_difference);
   println!("  - Tensor size: {}", tensor_data.len());
   ```

5. **Document cross-validation requirements** in test header

**Success Metrics**:
- Test passes with C++ reference integration
- Consistency ≥ 0.999 for all property test inputs
- Max absolute diff < 1e-5 for cross-platform parity
- No proptest compilation errors

---

### TODO Marker 1: TL2 Lookup Table Size Validation

**File**: `gguf_weight_loading_property_tests_enhanced.rs`  
**Line**: 454  
**Context**: `property_tl2_quantization_precision_improvement` test

**Current Code**:
```rust
// Property: TL2 lookup table should be larger than TL1 (8-bit vs 4-bit)
// TODO: Validate lookup table sizes when API is available
// prop_assert!(tl2_lookup_table.size > tl1_lookup_table.size);
```

**Required Implementation**:
```rust
// Validate lookup table sizes when QuantizedTensor exposes metadata
let tl1_lookup_size = tl1_quantized.lookup_table_size()
    .context("TL1 quantization should expose lookup table size")?;
let tl2_lookup_size = tl2_quantized.lookup_table_size()
    .context("TL2 quantization should expose lookup table size")?;

prop_assert!(
    tl2_lookup_size > tl1_lookup_size,
    "TL2 lookup table size {} should exceed TL1 size {} (8-bit vs 4-bit)",
    tl2_lookup_size,
    tl1_lookup_size
);

// Expected sizes: TL1 = 16 entries (4-bit), TL2 = 256 entries (8-bit)
prop_assert!(
    tl1_lookup_size <= 16,
    "TL1 lookup table size {} exceeds expected 16 entries",
    tl1_lookup_size
);
prop_assert!(
    tl2_lookup_size <= 256,
    "TL2 lookup table size {} exceeds expected 256 entries",
    tl2_lookup_size
);
```

**API Extension Required**:
```rust
// In bitnet-quantization/src/quantized_tensor.rs
impl QuantizedTensor {
    /// Returns the size of the lookup table (if applicable)
    pub fn lookup_table_size(&self) -> Option<usize> {
        match self.quantization_type {
            QuantizationType::TL1 => Some(16),  // 4-bit = 2^4 entries
            QuantizationType::TL2 => Some(256), // 8-bit = 2^8 entries
            _ => None, // I2S doesn't use lookup tables
        }
    }
}
```

**Priority**: LOW (validation enhancement, not blocking core functionality)

---

## Integration Tests

### TODO Marker 2: Create Actual GGUF File with Proper Structure

**File**: `gguf_weight_loading_integration_tests.rs`  
**Line**: 72  
**Context**: `IntegrationTestFixture::create_test_model()`

**Current Code**:
```rust
pub fn create_test_model(&self) -> Result<PathBuf> {
    let model_path = self.temp_dir.path().join("integration_test_model.gguf");

    // TODO: Create actual GGUF file with proper structure
    // For now, create placeholder to enable test compilation
    std::fs::write(&model_path, b"integration_test_gguf_content")
        .context("Failed to create test GGUF file")?;

    Ok(model_path)
}
```

**Required Implementation**:
```rust
pub fn create_test_model(&self) -> Result<PathBuf> {
    use bitnet_models::gguf_builder::GgufBuilder;
    
    let model_path = self.temp_dir.path().join("integration_test_model.gguf");

    // Create proper GGUF file with complete model structure
    let mut builder = GgufBuilder::new()
        .with_version(3)
        .with_metadata("model.name", "bitnet-integration-test")
        .with_metadata("model.architecture", "bitnet")
        .with_metadata("bitnet.hidden_size", self.config.hidden_size as u64)
        .with_metadata("bitnet.intermediate_size", self.config.intermediate_size as u64)
        .with_metadata("bitnet.num_layers", self.config.test_model_layers as u64)
        .with_metadata("bitnet.vocab_size", self.config.vocab_size as u64);

    // Add model weights for each layer
    for layer_idx in 0..self.config.test_model_layers {
        let layer_prefix = format!("blk.{}", layer_idx);
        
        // Attention weights (Q, K, V, O projections)
        builder = builder
            .add_tensor(&format!("{}.attn_q.weight", layer_prefix), 
                       create_random_tensor(self.config.hidden_size, self.config.hidden_size)?)
            .add_tensor(&format!("{}.attn_k.weight", layer_prefix),
                       create_random_tensor(self.config.hidden_size, self.config.hidden_size)?)
            .add_tensor(&format!("{}.attn_v.weight", layer_prefix),
                       create_random_tensor(self.config.hidden_size, self.config.hidden_size)?)
            .add_tensor(&format!("{}.attn_output.weight", layer_prefix),
                       create_random_tensor(self.config.hidden_size, self.config.hidden_size)?);
        
        // Feed-forward weights (gate, up, down projections)
        builder = builder
            .add_tensor(&format!("{}.ffn_gate.weight", layer_prefix),
                       create_random_tensor(self.config.intermediate_size, self.config.hidden_size)?)
            .add_tensor(&format!("{}.ffn_up.weight", layer_prefix),
                       create_random_tensor(self.config.intermediate_size, self.config.hidden_size)?)
            .add_tensor(&format!("{}.ffn_down.weight", layer_prefix),
                       create_random_tensor(self.config.hidden_size, self.config.intermediate_size)?);
        
        // LayerNorm weights
        builder = builder
            .add_tensor(&format!("{}.attn_norm.weight", layer_prefix),
                       create_random_tensor(1, self.config.hidden_size)?)
            .add_tensor(&format!("{}.ffn_norm.weight", layer_prefix),
                       create_random_tensor(1, self.config.hidden_size)?);
    }

    // Add embedding and output projection
    builder = builder
        .add_tensor("token_embd.weight", 
                   create_random_tensor(self.config.vocab_size, self.config.hidden_size)?)
        .add_tensor("output.weight",
                   create_random_tensor(self.config.vocab_size, self.config.hidden_size)?);

    // Write GGUF file
    builder.write(&model_path)
        .context("Failed to write integration test GGUF file")?;

    Ok(model_path)
}

/// Helper: Create random tensor with normal distribution
fn create_random_tensor(rows: usize, cols: usize) -> Result<Vec<f32>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let data: Vec<f32> = (0..rows * cols)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    Ok(data)
}
```

**API Extension Required**:
```rust
// In bitnet-models/src/gguf_builder.rs (new module)
pub struct GgufBuilder {
    version: u32,
    metadata: HashMap<String, MetadataValue>,
    tensors: Vec<(String, Vec<f32>, Vec<usize>)>,
}

impl GgufBuilder {
    pub fn new() -> Self { /* ... */ }
    pub fn with_version(mut self, version: u32) -> Self { /* ... */ }
    pub fn with_metadata(mut self, key: &str, value: impl Into<MetadataValue>) -> Self { /* ... */ }
    pub fn add_tensor(mut self, name: &str, data: Vec<f32>, shape: Vec<usize>) -> Self { /* ... */ }
    pub fn write(&self, path: &Path) -> Result<()> { /* ... */ }
}
```

**Priority**: HIGH (blocks all integration tests from running with real GGUF files)

---

### Scaffold 2: test_integration_performance_pipeline_cpu

**File**: `gguf_weight_loading_integration_tests.rs`  
**Lines**: 345-388  
**Status**: Stub (ignored, waiting for optimized weight loading)  
**Priority**: LOW (post-MVP optimization)

**Current Implementation**:
```rust
#[ignore] // Issue #159: TDD placeholder - optimized weight loading implementation needed
#[cfg(feature = "cpu")]
#[tokio::test]
async fn test_integration_performance_pipeline_cpu() -> Result<()> {
    let config = IntegrationTestConfig::default();
    let fixture = IntegrationTestFixture::new()?.with_config(config.clone());
    let model_path = fixture.create_test_model()?;

    // Time the complete loading pipeline
    let start_time = std::time::Instant::now();
    let load_result = bitnet_models::gguf_simple::load_gguf(&model_path, Device::Cpu);
    let loading_time = start_time.elapsed();

    match load_result {
        Ok((_, tensor_map)) => {
            // Validate loading performance meets requirements
            assert!(
                loading_time.as_secs() <= 30,
                "Loading time {}s exceeds 30s requirement",
                loading_time.as_secs()
            );

            // Test memory efficiency
            let memory_usage = estimate_tensor_memory_usage(&tensor_map);
            let model_size = estimate_model_size(&config);
            let memory_overhead = memory_usage as f32 / model_size as f32;

            assert!(
                memory_overhead <= 1.5,
                "Memory overhead {:.2}x exceeds 1.5x limit",
                memory_overhead
            );

            // Test quantization performance impact
            test_quantization_performance_impact(&tensor_map)?;
        }
        Err(err) => {
            eprintln!("Performance integration test correctly failing (TDD Red): {}", err);
            panic!("Performance integration will pass once optimized weight loading is complete");
        }
    }

    Ok(())
}
```

**Acceptance Criteria**:
1. ✅ Complete GGUF loading pipeline completes in ≤30 seconds
2. ✅ Memory overhead ≤ 1.5× theoretical model size
3. ❌ Quantization performance impact measured (throughput degradation)
4. ❌ Zero-copy loading validated for large tensors
5. ❌ Memory-mapped file access efficiency measured

**Required Dependencies**:
- **Helper Functions**:
  - `test_quantization_performance_impact()` (line 548-552): Currently a stub
  - `estimate_tensor_memory_usage()` (line 555-557): ✅ Already implemented
  - `estimate_model_size()` (line 560-565): ✅ Already implemented

**Implementation for `test_quantization_performance_impact()`**:
```rust
fn test_quantization_performance_impact(tensor_map: &HashMap<String, CandleTensor>) -> Result<()> {
    use bitnet_quantization::{I2SQuantizer, TL1Quantizer};
    
    let quantizer_i2s = I2SQuantizer::new();
    let quantizer_tl1 = TL1Quantizer::new();
    
    let mut total_quantization_time = std::time::Duration::ZERO;
    let mut total_dequantization_time = std::time::Duration::ZERO;
    let mut total_elements = 0usize;
    
    for (tensor_name, tensor) in tensor_map {
        if !tensor_name.contains("weight") {
            continue; // Only test weights
        }
        
        // Benchmark I2S quantization round-trip
        let start = std::time::Instant::now();
        let quantized = quantizer_i2s.quantize(&BitNetTensor::new(tensor.clone()), &candle_core::Device::Cpu)?;
        total_quantization_time += start.elapsed();
        
        let start = std::time::Instant::now();
        let _dequantized = quantizer_i2s.dequantize(&quantized, &candle_core::Device::Cpu)?;
        total_dequantization_time += start.elapsed();
        
        total_elements += tensor.dims().iter().product::<usize>();
    }
    
    // Calculate throughput metrics
    let quantization_throughput = total_elements as f64 / total_quantization_time.as_secs_f64() / 1e6; // M elements/sec
    let dequantization_throughput = total_elements as f64 / total_dequantization_time.as_secs_f64() / 1e6;
    
    println!("Quantization performance:");
    println!("  - Quantization throughput: {:.2} M elements/sec", quantization_throughput);
    println!("  - Dequantization throughput: {:.2} M elements/sec", dequantization_throughput);
    
    // Validate throughput meets baseline requirements
    assert!(
        quantization_throughput >= 0.5, // At least 0.5M elements/sec
        "Quantization throughput {:.2} M elements/sec below baseline 0.5",
        quantization_throughput
    );
    
    assert!(
        dequantization_throughput >= 1.0, // At least 1M elements/sec
        "Dequantization throughput {:.2} M elements/sec below baseline 1.0",
        dequantization_throughput
    );
    
    Ok(())
}
```

**Blockers**:
- Optimized weight loading not yet implemented (MVP phase uses basic loading)
- Memory-mapped file access API not exposed in public interface
- Zero-copy validation requires internal loader metrics

**Priority**: LOW (post-MVP optimization and benchmarking)

---

### Integration Test Helper TODOs

The integration test file contains 16 helper function TODOs (lines 400-550). These are **intentional TDD stubs** that will be implemented as cross-crate integration progresses:

**Quantization Integration Helpers** (lines 400-425):
- `test_i2s_quantization_with_loaded_weight()`: Integrate bitnet-quantization I2S quantizer
- `test_tl1_quantization_with_loaded_weight()`: Integrate TL1 quantizer
- `test_tl2_quantization_with_loaded_weight()`: Integrate TL2 quantizer

**Kernel Integration Helpers** (lines 428-446):
- `test_attention_kernel_with_loaded_weights()`: Integrate bitnet-kernels attention ops
- `test_feedforward_kernel_with_loaded_weights()`: Integrate feed-forward ops

**Inference Integration Helpers** (lines 449-497):
- `test_inference_engine_with_real_weights()`: Integrate bitnet-inference engine
- `test_gpu_quantization_integration()`: GPU quantization ops
- `test_gpu_kernel_integration()`: GPU kernel ops
- `test_gpu_inference_integration()`: GPU inference pipeline

**Cross-Validation Helpers** (lines 500-513):
- `test_cpp_reference_validation()`: Integrate crossval framework
- `test_deterministic_weight_validation()`: Test deterministic loading

**FFI Integration Helpers** (lines 516-528):
- `test_ffi_bridge_compatibility()`: Test FFI bridge operations
- `test_cpp_rust_weight_comparison()`: Compare C++ and Rust weight loading

**WASM Integration Helpers** (lines 533-545):
- `test_wasm_tensor_compatibility()`: WASM-specific tensor operations
- `test_wasm_memory_management()`: WASM memory constraints

**Implementation Priority**:
1. **HIGH**: Quantization integration helpers (enables AC2 validation)
2. **HIGH**: Kernel integration helpers (enables AC3 validation)
3. **MEDIUM**: Inference integration helpers (enables end-to-end validation)
4. **MEDIUM**: Cross-validation helpers (enables AC5 validation)
5. **LOW**: FFI/WASM helpers (post-MVP platform support)

**Implementation Pattern** (example for I2S quantization):
```rust
fn test_i2s_quantization_with_loaded_weight(
    weight_tensor: &CandleTensor,
    weight_name: &str,
) -> Result<()> {
    use bitnet_quantization::I2SQuantizer;
    
    let quantizer = I2SQuantizer::new();
    let bitnet_tensor = BitNetTensor::new(weight_tensor.clone());
    
    // Quantize loaded weight
    let quantized = quantizer
        .quantize(&bitnet_tensor, weight_tensor.device())
        .context("Failed to quantize loaded weight")?;
    
    // Validate quantization properties
    assert!(quantized.data.len() < weight_tensor.elem_count() * 4 / 8, 
           "I2S quantization should reduce memory (2-bit)");
    
    // Dequantize and validate accuracy
    let dequantized = quantizer
        .dequantize(&quantized, weight_tensor.device())
        .context("Failed to dequantize weight")?;
    
    let accuracy = calculate_cosine_similarity(&bitnet_tensor, &dequantized)?;
    assert!(
        accuracy >= 0.70, // Baseline for 2-bit quantization
        "I2S quantization accuracy {} below baseline 0.70 for weight '{}'",
        accuracy,
        weight_name
    );
    
    println!("I2S quantization test passed for '{}': accuracy {:.4}", weight_name, accuracy);
    Ok(())
}
```

---

## Device-Aware Tests

### Scaffold 3: test_ac6_memory_efficiency_validation

**File**: `gguf_weight_loading_device_aware_tests.rs`  
**Lines**: 390-442  
**Status**: Stub (ignored, temp file lifetime management issue)  
**Priority**: MEDIUM (memory efficiency validation)

**Current Implementation**:
```rust
#[ignore] // Issue #159: TDD placeholder - temp file lifetime management needed
#[cfg(feature = "cpu")]
#[test]
fn test_ac6_memory_efficiency_validation() -> Result<()> {
    let config = DeviceAwareTestConfig::default();

    // Test different model sizes to validate memory scaling
    for &(rows, cols) in &config.test_tensor_sizes {
        let tensor_size_mb = (rows * cols * 4) / (1024 * 1024); // FP32 size in MB

        // Skip very large tensors if they exceed memory limits
        if tensor_size_mb > config.memory_limit_mb / 4 {
            continue;
        }

        let (_temp_dir, test_model_path) =
            create_sized_test_model(rows, cols).context("Failed to create sized test model")?;

        // Test CPU memory efficiency
        let memory_before = get_process_memory_usage_mb();
        #[allow(deprecated)]
        let (_, weights) = bitnet_models::gguf_simple::load_gguf(&test_model_path, Device::Cpu)?;
        let memory_after = get_process_memory_usage_mb();

        let memory_used = memory_after.saturating_sub(memory_before);
        let memory_efficiency = tensor_size_mb as f32 / memory_used.max(1) as f32;

        // Validate memory efficiency meets threshold
        assert!(
            memory_efficiency >= config.memory_efficiency_threshold,
            "Memory efficiency {:.2} below threshold {:.2} for tensor {}x{} (used: {} MB, expected: {} MB)",
            memory_efficiency,
            config.memory_efficiency_threshold,
            rows,
            cols,
            memory_used,
            tensor_size_mb
        );

        // Validate zero-copy operations for large tensors
        if tensor_size_mb > 100 {
            validate_zero_copy_operations(&weights, rows * cols)
                .context("Zero-copy validation failed")?;
        }

        println!(
            "Memory efficiency test passed for {}x{}: {:.2} efficiency",
            rows, cols, memory_efficiency
        );
    }

    Ok(())
}
```

**Issue**: Temp directory (`_temp_dir`) goes out of scope when `create_sized_test_model()` returns, causing the test file to be deleted before `load_gguf()` can read it.

**Solution**: Keep temp directory alive for the duration of the test:
```rust
#[cfg(feature = "cpu")]
#[test]
fn test_ac6_memory_efficiency_validation() -> Result<()> {
    let config = DeviceAwareTestConfig::default();

    // Test different model sizes to validate memory scaling
    for &(rows, cols) in &config.test_tensor_sizes {
        let tensor_size_mb = (rows * cols * 4) / (1024 * 1024); // FP32 size in MB

        // Skip very large tensors if they exceed memory limits
        if tensor_size_mb > config.memory_limit_mb / 4 {
            continue;
        }

        // Keep temp directory alive for the duration of this iteration
        let (temp_dir, test_model_path) =
            create_sized_test_model(rows, cols).context("Failed to create sized test model")?;

        // Test CPU memory efficiency
        let memory_before = get_process_memory_usage_mb();
        #[allow(deprecated)]
        let (_, weights) = bitnet_models::gguf_simple::load_gguf(&test_model_path, Device::Cpu)?;
        let memory_after = get_process_memory_usage_mb();

        let memory_used = memory_after.saturating_sub(memory_before);
        let memory_efficiency = tensor_size_mb as f32 / memory_used.max(1) as f32;

        // Validate memory efficiency meets threshold
        assert!(
            memory_efficiency >= config.memory_efficiency_threshold,
            "Memory efficiency {:.2} below threshold {:.2} for tensor {}x{} (used: {} MB, expected: {} MB)",
            memory_efficiency,
            config.memory_efficiency_threshold,
            rows,
            cols,
            memory_used,
            tensor_size_mb
        );

        // Validate zero-copy operations for large tensors
        if tensor_size_mb > 100 {
            validate_zero_copy_operations(&weights, rows * cols)
                .context("Zero-copy validation failed")?;
        }

        println!(
            "Memory efficiency test passed for {}x{}: {:.2} efficiency",
            rows, cols, memory_efficiency
        );

        // Temp directory will be cleaned up when `temp_dir` goes out of scope here
    }

    Ok(())
}
```

**Acceptance Criteria**:
1. ✅ Memory efficiency ≥ 80% (memory_efficiency_threshold)
2. ❌ Zero-copy operations validated for large tensors (>100 MB)
3. ❌ Memory scaling validated across different tensor sizes (512×512 to 8192×32000)
4. ❌ Process memory usage tracking works cross-platform

**Blockers**:
- `get_process_memory_usage_mb()` is a stub (line 629-633) returning mock value
- `validate_zero_copy_operations()` is a stub (line 752-768) with placeholder validation

**Implementation for `get_process_memory_usage_mb()`**:
```rust
#[cfg(target_os = "linux")]
fn get_process_memory_usage_mb() -> usize {
    use std::fs;
    
    // Read /proc/self/statm (memory usage in pages)
    let statm = fs::read_to_string("/proc/self/statm")
        .unwrap_or_else(|_| "0".to_string());
    
    let pages: usize = statm.split_whitespace()
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    
    // Convert pages to MB (typically 4KB per page)
    (pages * 4096) / (1024 * 1024)
}

#[cfg(target_os = "macos")]
fn get_process_memory_usage_mb() -> usize {
    use std::process::Command;
    
    // Use `ps` to get RSS memory
    let output = Command::new("ps")
        .args(&["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok();
    
    output
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<usize>().ok())
        .map(|kb| kb / 1024) // Convert KB to MB
        .unwrap_or(0)
}

#[cfg(target_os = "windows")]
fn get_process_memory_usage_mb() -> usize {
    use windows::Win32::System::ProcessStatus::GetProcessMemoryInfo;
    use windows::Win32::Foundation::GetCurrentProcess;
    
    unsafe {
        let mut mem_info = std::mem::zeroed();
        if GetProcessMemoryInfo(
            GetCurrentProcess(),
            &mut mem_info,
            std::mem::size_of_val(&mem_info) as u32,
        ).as_bool() {
            (mem_info.WorkingSetSize / (1024 * 1024)) as usize
        } else {
            0
        }
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn get_process_memory_usage_mb() -> usize {
    // Fallback for unsupported platforms
    1024 // 1GB mock usage
}
```

**Priority**: MEDIUM (memory efficiency is important but not blocking MVP)

---

### Device-Aware Test Helper TODOs

The device-aware test file contains 10 helper function TODOs (lines 605-818):

**Device Detection Helpers**:
- `is_cuda_available()` (line 605): Currently checks env vars, should use candle_core CUDA detection
- `get_process_memory_usage_mb()` (line 630): Mock implementation, needs cross-platform memory tracking

**Memory Management Helpers**:
- `test_cpu_memory_mapped_access()` (line 646): Validate memory-mapped file access when API available
- `test_mixed_precision_support()` (line 660): Implement mixed precision testing when API available
- `validate_zero_copy_operations()` (line 756): Implement zero-copy validation when API available

**Device Selection Helpers**:
- `test_automatic_device_selection()` (line 776): Implement automatic device selection when API available
- `test_gpu_memory_fallback()` (line 795): Simulate GPU memory pressure and test fallback
- `test_mixed_precision_fallback()` (line 808): Test mixed precision fallback when API available

**Implementation Priority**:
1. **HIGH**: `is_cuda_available()` - enables GPU test gating
2. **HIGH**: `get_process_memory_usage_mb()` - enables memory efficiency validation
3. **MEDIUM**: Memory management helpers - enables zero-copy validation
4. **MEDIUM**: Device selection helpers - enables automatic fallback testing
5. **LOW**: Mixed precision helpers - post-MVP feature

**Implementation Pattern** (example for CUDA detection):
```rust
fn is_cuda_available() -> bool {
    #[cfg(feature = "gpu")]
    {
        // Use candle_core's CUDA detection
        candle_core::Device::cuda_if_available(0).is_ok()
    }
    
    #[cfg(not(feature = "gpu"))]
    {
        false // GPU feature not compiled
    }
}
```

---

## Implementation Order Recommendation

Based on dependencies and priority:

### Phase 1: Foundation (HIGH Priority)
1. **TODO Marker 2**: Implement `IntegrationTestFixture::create_test_model()` with proper GGUF builder
   - **Rationale**: Blocks all integration tests from running
   - **Estimated Effort**: 4-6 hours (requires GgufBuilder implementation)
   - **Dependencies**: None

2. **Device Helper**: Implement `is_cuda_available()` with candle_core detection
   - **Rationale**: Enables GPU test gating across all device-aware tests
   - **Estimated Effort**: 30 minutes
   - **Dependencies**: None

3. **Device Helper**: Implement cross-platform `get_process_memory_usage_mb()`
   - **Rationale**: Enables memory efficiency validation (AC6.4)
   - **Estimated Effort**: 2-3 hours
   - **Dependencies**: None

### Phase 2: Integration Helpers (HIGH Priority)
4. **Integration Helpers**: Implement quantization integration helpers
   - `test_i2s_quantization_with_loaded_weight()`
   - `test_tl1_quantization_with_loaded_weight()`
   - `test_tl2_quantization_with_loaded_weight()`
   - **Rationale**: Enables AC2 quantization accuracy validation
   - **Estimated Effort**: 4-6 hours
   - **Dependencies**: Phase 1.1 (proper GGUF files)

5. **Integration Helpers**: Implement kernel integration helpers
   - `test_attention_kernel_with_loaded_weights()`
   - `test_feedforward_kernel_with_loaded_weights()`
   - **Rationale**: Enables AC3 kernel integration validation
   - **Estimated Effort**: 6-8 hours
   - **Dependencies**: Phase 1.1, bitnet-kernels API

### Phase 3: Device Awareness (MEDIUM Priority)
6. **Scaffold 3**: Fix `test_ac6_memory_efficiency_validation()` temp file lifetime
   - **Rationale**: Enables memory efficiency validation
   - **Estimated Effort**: 30 minutes (simple fix)
   - **Dependencies**: Phase 1.3 (memory tracking)

7. **Device Helpers**: Implement memory management helpers
   - `test_cpu_memory_mapped_access()`
   - `validate_zero_copy_operations()`
   - **Rationale**: Enables zero-copy validation
   - **Estimated Effort**: 4-6 hours
   - **Dependencies**: Phase 1.1, loader API exposure

### Phase 4: Cross-Validation (MEDIUM Priority)
8. **Scaffold 1**: Fix `property_cross_platform_quantization_consistency()` proptest errors
   - **Rationale**: Enables cross-platform consistency validation
   - **Estimated Effort**: 2-3 hours
   - **Dependencies**: None (isolated to proptest macro)

9. **Scaffold 1**: Integrate C++ reference implementation
   - **Rationale**: Enables AC5 deterministic validation
   - **Estimated Effort**: 8-12 hours (requires FFI bridge)
   - **Dependencies**: Issue #469 (FFI hygiene), crossval framework

10. **Integration Helpers**: Implement cross-validation helpers
    - `test_cpp_reference_validation()`
    - `test_deterministic_weight_validation()`
    - **Rationale**: Enables end-to-end cross-validation
    - **Estimated Effort**: 6-8 hours
    - **Dependencies**: Phase 4.9 (C++ integration)

### Phase 5: Performance & Optimization (LOW Priority)
11. **Scaffold 2**: Implement `test_integration_performance_pipeline_cpu()`
    - **Rationale**: Post-MVP performance benchmarking
    - **Estimated Effort**: 4-6 hours
    - **Dependencies**: Optimized weight loading, Phase 1.1

12. **Integration Helper**: Implement `test_quantization_performance_impact()`
    - **Rationale**: Quantization throughput benchmarking
    - **Estimated Effort**: 2-3 hours
    - **Dependencies**: Phase 2.4 (quantization integration)

13. **TODO Marker 1**: Implement TL2 lookup table size validation
    - **Rationale**: Enhanced validation, not blocking
    - **Estimated Effort**: 1-2 hours
    - **Dependencies**: QuantizedTensor API extension

### Phase 6: Platform Support (LOW Priority)
14. **Integration Helpers**: Implement FFI/WASM helpers
    - `test_ffi_bridge_compatibility()`
    - `test_cpp_rust_weight_comparison()`
    - `test_wasm_tensor_compatibility()`
    - `test_wasm_memory_management()`
    - **Rationale**: Post-MVP platform support
    - **Estimated Effort**: 12-16 hours
    - **Dependencies**: FFI bridge, WASM build system

---

## Common Patterns

### Pattern 1: Integration Test Fixture with Proper GGUF Files

**Current Problem**: Mock GGUF files (`b"mock_gguf_content"`) don't have proper structure

**Solution Template**:
```rust
use bitnet_models::gguf_builder::GgufBuilder;

fn create_integration_test_model(config: &IntegrationTestConfig, temp_dir: &TempDir) -> Result<PathBuf> {
    let model_path = temp_dir.path().join("test_model.gguf");
    
    let mut builder = GgufBuilder::new()
        .with_version(3)
        .with_metadata("model.architecture", "bitnet")
        .with_metadata("bitnet.hidden_size", config.hidden_size as u64);
    
    // Add model weights
    for layer_idx in 0..config.test_model_layers {
        builder = builder
            .add_tensor(&format!("blk.{}.attn_q.weight", layer_idx), 
                       generate_random_weight(config.hidden_size, config.hidden_size)?)
            .add_tensor(&format!("blk.{}.attn_k.weight", layer_idx),
                       generate_random_weight(config.hidden_size, config.hidden_size)?);
    }
    
    builder.write(&model_path)?;
    Ok(model_path)
}
```

**Applies to**: All integration tests, device-aware tests

---

### Pattern 2: Cross-Platform Memory Tracking

**Current Problem**: Memory tracking is a stub returning mock values

**Solution Template**:
```rust
#[cfg(target_os = "linux")]
fn get_process_memory_usage_mb() -> usize {
    std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|s| s.split_whitespace().next().and_then(|p| p.parse().ok()))
        .map(|pages: usize| (pages * 4096) / (1024 * 1024))
        .unwrap_or(0)
}

#[cfg(target_os = "macos")]
fn get_process_memory_usage_mb() -> usize {
    // Use `ps` command to get RSS
    std::process::Command::new("ps")
        .args(&["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<usize>().ok())
        .map(|kb| kb / 1024)
        .unwrap_or(0)
}

#[cfg(target_os = "windows")]
fn get_process_memory_usage_mb() -> usize {
    // Use Windows API (GetProcessMemoryInfo)
    // ... (see detailed implementation above)
}
```

**Applies to**: Device-aware tests, performance tests

---

### Pattern 3: Quantization Integration Testing

**Current Problem**: Quantization integration helpers are stubs

**Solution Template**:
```rust
fn test_quantization_integration(
    weight_tensor: &CandleTensor,
    weight_name: &str,
    quantizer: impl Quantizer,
    min_accuracy: f32,
) -> Result<()> {
    let bitnet_tensor = BitNetTensor::new(weight_tensor.clone());
    
    // Quantize
    let quantized = quantizer
        .quantize(&bitnet_tensor, weight_tensor.device())
        .context("Quantization failed")?;
    
    // Validate memory reduction
    let original_size = weight_tensor.elem_count() * 4; // FP32
    let quantized_size = quantized.data.len();
    assert!(
        quantized_size < original_size / 2,
        "Quantization should reduce memory by at least 50%"
    );
    
    // Dequantize and validate accuracy
    let dequantized = quantizer
        .dequantize(&quantized, weight_tensor.device())
        .context("Dequantization failed")?;
    
    let accuracy = calculate_cosine_similarity(&bitnet_tensor, &dequantized)?;
    assert!(
        accuracy >= min_accuracy,
        "Quantization accuracy {} below threshold {} for '{}'",
        accuracy,
        min_accuracy,
        weight_name
    );
    
    Ok(())
}
```

**Applies to**: Integration tests (quantization helpers)

---

### Pattern 4: Device-Aware Tensor Placement Validation

**Current Problem**: GPU availability detection is env-var based

**Solution Template**:
```rust
fn validate_device_placement(
    tensor_map: &HashMap<String, CandleTensor>,
    expected_device: Device,
) -> Result<()> {
    for (tensor_name, tensor) in tensor_map {
        let actual_device = tensor.device();
        
        match expected_device {
            Device::Cpu => {
                assert!(
                    actual_device.is_cpu(),
                    "Tensor '{}' not on CPU: {:?}",
                    tensor_name,
                    actual_device
                );
            }
            Device::Cuda(_) => {
                // Allow CPU fallback for GPU tensors
                assert!(
                    actual_device.is_cuda() || actual_device.is_cpu(),
                    "Tensor '{}' on unexpected device: {:?}",
                    tensor_name,
                    actual_device
                );
            }
        }
    }
    Ok(())
}
```

**Applies to**: Device-aware tests, GPU integration tests

---

### Pattern 5: Temp Directory Lifetime Management

**Current Problem**: Temp directories deleted before tests can use them

**Solution Template**:
```rust
#[test]
fn test_with_temp_file() -> Result<()> {
    // Keep temp_dir in scope for entire test
    let (temp_dir, model_path) = create_test_model()?;
    
    // Use model_path here
    let result = load_gguf(&model_path, Device::Cpu)?;
    
    // Validate result
    assert!(!result.1.is_empty());
    
    // temp_dir dropped here, cleaning up files
    Ok(())
}

fn create_test_model() -> Result<(TempDir, PathBuf)> {
    let temp_dir = TempDir::new()?;
    let model_path = temp_dir.path().join("model.gguf");
    std::fs::write(&model_path, create_gguf_content())?;
    Ok((temp_dir, model_path)) // Return both to keep temp_dir alive
}
```

**Applies to**: All integration tests, device-aware tests

---

## Success Metrics

### Enhanced Property Tests
- ✅ All property tests pass with 100+ random inputs
- ❌ Cross-platform consistency ≥ 0.999 (blocked by C++ integration)
- ✅ Higher-order statistics (skewness, kurtosis) validated within tolerances
- ✅ Block-level consistency validated (coefficient of variation < 0.15)

### Integration Tests
- ❌ All cross-crate integration tests pass (blocked by proper GGUF builder)
- ❌ Quantization integration validated for I2S, TL1, TL2
- ❌ Kernel integration validated for attention and feed-forward ops
- ❌ Inference engine integration validated end-to-end
- ❌ Performance baseline established (≤30s loading, ≤1.5× memory overhead)

### Device-Aware Tests
- ✅ CPU tensor placement validated with SIMD detection
- ❌ GPU tensor placement validated with fallback (requires GPU test environment)
- ❌ Cross-device consistency validated (cosine similarity ≥ 0.9999)
- ❌ Memory efficiency ≥ 80% validated (blocked by temp file lifetime fix)
- ❌ Automatic device selection and fallback validated

### Overall Progress
- **Passing Tests**: 0/3 ignored tests (all blocked)
- **Implemented TODOs**: 0/27 (all placeholders)
- **Estimated Completion**: 60-80 hours across 6 implementation phases
- **Critical Path**: Phase 1 (foundation) → Phase 2 (integration helpers) → Phase 3 (device awareness)

---

## Notes

1. **Test Philosophy**: These tests follow TDD principles with intentional scaffolding. Many tests are designed to fail until specific features are implemented (e.g., optimized weight loading, C++ reference integration).

2. **Feature Gate Coordination**: Integration tests require careful feature gate management:
   - `cpu` for CPU-specific tests
   - `gpu` for GPU-specific tests
   - `crossval` for C++ reference validation
   - `ffi` for FFI bridge tests
   - `wasm32` + `browser` for WASM tests

3. **Test Environment Requirements**:
   - CPU tests: Any platform with Rust compiler
   - GPU tests: CUDA toolkit, NVIDIA GPU
   - Cross-validation tests: C++ reference implementation (`BITNET_CPP_DIR`)
   - Performance tests: Release builds recommended

4. **Blocking Issues**:
   - **Issue #469**: FFI build hygiene (blocks cross-validation)
   - **Issue #439**: Feature gate consistency (blocks GPU tests)
   - **Issue #254**: Shape mismatch (may affect inference integration)

5. **Post-MVP Optimizations**: Many scaffolds (especially performance tests) are intentionally deferred to post-MVP phase when optimized loading and SIMD kernels are implemented.

---

## Related Documentation

- Feature Spec: `docs/specs/gguf-weight-loading.md`
- API Contracts: `docs/specs/gguf-weight-loading-api-contracts.md`
- Integration Testing: `docs/specs/gguf-weight-loading-integration-testing.md`
- CLAUDE.md: Project guidance and test status section

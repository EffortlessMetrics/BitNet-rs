# GPU/CUDA Development Guide

This document covers GPU/CUDA development practices, testing strategies, and troubleshooting for BitNet.rs.

## GPU Backend Detection and Hardware Querying

BitNet.rs provides comprehensive GPU detection utilities supporting multiple backends (CUDA, Metal, ROCm, WebGPU) alongside production-ready CUDA device querying using the cudarc API to enable intelligent GPU acceleration and automatic fallback mechanisms.

### GPU Detection API

The new GPU detection utilities provide backend-agnostic GPU availability checking:

```rust
use bitnet_kernels::gpu_utils::{gpu_available, get_gpu_info, preflight_check};

// Quick availability check
if gpu_available() {
    println!("GPU acceleration available");
}

// Detailed backend information
let gpu_info = get_gpu_info();
println!("{}", gpu_info.summary());

// Available backends:
println!("CUDA: {}", gpu_info.cuda);
println!("Metal: {}", gpu_info.metal); 
println!("ROCm: {}", gpu_info.rocm);
println!("WebGPU: {}", gpu_info.wgpu);

// Version information (when available)
if let Some(version) = gpu_info.cuda_version {
    println!("CUDA Version: {}", version);
}

// Preflight check with helpful error messages
match preflight_check() {
    Ok(()) => println!("GPU ready for acceleration"),
    Err(msg) => eprintln!("GPU setup issue: {}", msg),
}
```

### GPU Detection Commands

```bash
# Test GPU detection functionality
cargo test -p bitnet-kernels --no-default-features test_gpu_info_summary

# Run xtask commands with GPU detection
cargo run -p xtask -- download-model  # Uses GPU detection for optimizations

# Mock GPU scenarios for testing (see Testing section)
BITNET_GPU_FAKE="cuda,rocm" cargo test -p bitnet-kernels test_gpu_info_mocked_scenarios
```

### Backend-Specific Detection

1. **CUDA Detection**:
   - Uses `nvidia-smi` to query available GPUs
   - Extracts CUDA version from `nvcc --version`
   - Provides compute capability and memory information

2. **Metal Detection**:
   - Automatic detection on macOS systems
   - Uses system information to identify Apple Silicon

3. **ROCm Detection**:
   - Uses `rocm-smi` to query AMD GPUs
   - Extracts ROCm version information
   - Supports multiple AMD GPU configurations

4. **WebGPU Detection**:
   - Available when any other backend is present
   - Provides fallback compatibility for unsupported hardware

### Mock Testing Support

The GPU detection system includes comprehensive mock testing capabilities:

```bash
# Test scenarios without actual GPU hardware
export BITNET_GPU_FAKE="cuda"        # Mock CUDA-only
export BITNET_GPU_FAKE="metal"       # Mock Metal-only  
export BITNET_GPU_FAKE="cuda,rocm"   # Mock multiple backends
export BITNET_GPU_FAKE=""            # Mock no GPU available

# Run tests with mocked GPU environments
cargo test -p bitnet-kernels test_gpu_info_mocked_scenarios
```

## CUDA Device Querying and Hardware Detection

BitNet.rs implements production-ready CUDA device querying using the cudarc API to enable intelligent GPU acceleration and automatic fallback mechanisms.

### Device Information Available

The `CudaDeviceInfo` structure provides comprehensive hardware details:

```rust
pub struct CudaDeviceInfo {
    pub device_id: usize,                    // CUDA device index
    pub name: String,                        // Device name (e.g., "GeForce RTX 4090")
    pub compute_capability: (i32, i32),      // Major.minor compute capability
    pub total_memory: usize,                 // Total device memory in bytes
    pub multiprocessor_count: i32,           // Number of streaming multiprocessors
    pub max_threads_per_block: i32,          // Maximum threads per block
    pub max_shared_memory_per_block: usize,  // Maximum shared memory per block
    pub supports_fp16: bool,                 // Half-precision floating point support
    pub supports_bf16: bool,                 // Brain floating point support
}
```

### Device Querying Commands

```bash
# Test CUDA device detection and querying
cargo test -p bitnet-kernels --no-default-features --features gpu test_cuda_device_info_query

# List all available CUDA devices with detailed information
cargo run --example gpu_validation --no-default-features --features gpu

# Test CUDA availability in your application
cargo test -p bitnet-kernels --no-default-features --features gpu test_cuda_availability

# Validate device capabilities for BitNet quantization
cargo test -p bitnet-kernels --no-default-features --features gpu test_device_capability_validation
```

### Hardware-Aware Optimization

The CUDA implementation automatically optimizes based on detected hardware:

1. **Compute Capability Detection**:
   - **CC 6.0+**: Basic CUDA operations with FP32
   - **CC 6.1+**: FP16 tensor core operations enabled
   - **CC 8.0+**: BF16 tensor core operations enabled
   - **CC 9.0+**: FP8 operations (future enhancement)

2. **Memory-Based Optimization**:
   - Large memory devices (>16GB): Larger batch processing
   - Limited memory devices (<8GB): Conservative memory allocation
   - Automatic shared memory configuration based on device limits

3. **Multiprocessor Scaling**:
   - Grid dimensions automatically scaled to multiprocessor count
   - Work distribution optimized for available execution units

### Device Selection and Fallback

```bash
# Test multi-GPU device selection
cargo test -p bitnet-kernels --no-default-features --features gpu test_multi_gpu_selection

# Test automatic CPU fallback when GPU operations fail
cargo test -p bitnet-kernels --no-default-features --features gpu test_gpu_quantization_fallback --ignored

# Test concurrent GPU operations across devices
cargo test -p bitnet-kernels --no-default-features --features gpu test_concurrent_gpu_operations --ignored
```

### API Usage Examples

```rust
use bitnet_kernels::gpu::cuda::{list_cuda_devices, CudaKernel};

// Query all available CUDA devices
let devices = list_cuda_devices()?;
for device in devices {
    println!("Device {}: {} (CC {}.{})", 
        device.device_id, device.name, 
        device.compute_capability.0, device.compute_capability.1);
    println!("  Memory: {:.1} GB", device.total_memory as f64 / 1e9);
    println!("  FP16: {}, BF16: {}", device.supports_fp16, device.supports_bf16);
}

// Create kernel with specific device
let kernel = CudaKernel::new_with_device(0)?;
let info = kernel.device_info();
println!("Using device: {} with {} SMs", info.name, info.multiprocessor_count);

// Automatic optimization based on device capabilities
let optimal_params = kernel.calculate_optimal_launch_params(1024, 1024);
```

### Integration with Quantization

The CUDA device querying integrates with BitNet's quantization system:

- **Device-Aware Quantization**: Selects optimal quantization kernels based on compute capability
- **Automatic GPU Acceleration**: Falls back to CPU when GPU is unavailable or insufficient
- **Memory-Constrained Operation**: Adjusts quantization batch sizes based on available memory
- **Performance Monitoring**: Tracks GPU utilization and performance across operations
- **Host Memory Tracking**: Real-time monitoring of system memory usage with detailed statistics

## GPU Testing Strategy

GPU testing requires special consideration due to hardware dependencies and resource management:

### Test Classification by Hardware Requirements

```bash
# Always available (no GPU required)
cargo test --workspace --no-default-features --features cpu

# GPU smoke tests (basic availability, run on CI with GPU)
cargo test -p bitnet-kernels --no-default-features --features gpu --test gpu_smoke

# GPU integration tests (comprehensive, manual execution)
cargo test -p bitnet-kernels --no-default-features --features gpu --test gpu_quantization --ignored

# GPU performance tests (benchmarking, development only)
cargo test -p bitnet-kernels --no-default-features --features gpu test_gpu_performance --ignored
```

### GPU Test Categories

- **Availability Tests**: Check CUDA installation and device access
- **Functionality Tests**: Verify GPU operations produce correct results
- **Accuracy Tests**: Compare GPU vs CPU results for numerical consistency
- **Performance Tests**: Benchmark GPU acceleration vs CPU baseline
- **Resource Tests**: Validate memory management and concurrent operations
- **Fallback Tests**: Ensure graceful degradation when GPU unavailable

### Hardware-Specific Test Configuration

```bash
# Test matrix for different hardware scenarios
SCENARIOS=(
  "no-gpu:cpu-only"
  "gpu-low-mem:4gb-gpu" 
  "gpu-mid-mem:8gb-16gb-gpu"
  "gpu-high-mem:16gb-plus-gpu"
  "multi-gpu:multiple-devices"
)

# Compute capability matrix
CC_TARGETS=(
  "6.0:maxwell-pascal"
  "7.0:volta" 
  "8.0:ampere"
  "9.0:hopper"
)
```

### GPU Test Best Practices

- Use `#[ignore]` for hardware-dependent tests
- Implement comprehensive error handling and fallback testing
- Test both successful GPU operations and failure scenarios
- Validate memory cleanup and resource management
- Include cross-device testing for multi-GPU scenarios
- Test performance regression detection

### CI/CD GPU Testing

- **Tier 1**: CPU-only tests (always run)
- **Tier 2**: GPU availability and smoke tests (run on GPU CI)
- **Tier 3**: Integration tests with `--ignored` (manual/scheduled)
- **Tier 4**: Performance and multi-GPU tests (development/release)

For comprehensive test execution strategies and test suite configuration, see the [Test Suite Guide](test-suite.md).

## GPU/CUDA Development Best Practices

### PR Scope Management for GPU Features

Based on lessons learned from PR #102, follow these guidelines for GPU/CUDA development:

1. **Break Large Features into Focused PRs**:
   - **Device Querying PR**: Focus only on CUDA device property querying
   - **Quantization Enhancement PR**: Focus only on device-aware quantization
   - **Memory Management PR**: Focus only on GPU memory optimization
   - **Integration PR**: Combine smaller, well-tested components

2. **GPU Feature Development Workflow**:
   ```bash
   # Step 1: Implement core CUDA functionality (small PR)
   cargo test -p bitnet-kernels --no-default-features --features gpu --test gpu_smoke
   
   # Step 2: Add device querying (focused PR) 
   cargo test -p bitnet-kernels --no-default-features --features gpu test_cuda_device_info_query
   
   # Step 3: Enhance quantization with device awareness (focused PR)
   cargo test -p bitnet-kernels --no-default-features --features gpu test_gpu_vs_cpu_quantization_accuracy
   
   # Step 4: Integration testing (final PR)
   cargo test -p bitnet-kernels --no-default-features --features gpu --test gpu_quantization --ignored
   ```

3. **CUDA Implementation Guidelines**:
   - Use cudarc API consistently for all CUDA operations
   - Implement comprehensive error handling with meaningful messages
   - Add automatic CPU fallback for GPU operation failures
   - Include device capability detection for optimization
   - Test on multiple CUDA compute capabilities when possible

### Hardware-Dependent Testing Strategy

GPU/CUDA tests require special handling due to hardware dependencies:

1. **Test Categories**:
   - **Smoke Tests**: Basic functionality, run on CI
   - **Integration Tests**: Marked with `#[ignore]`, run manually with `--ignored`
   - **Performance Tests**: Benchmark comparisons, run locally
   - **Cross-Device Tests**: Multiple GPU testing, manual verification

2. **CI/CD Considerations**:
   - Default tests should pass without GPU hardware
   - Use feature gates to conditionally compile GPU code
   - Provide clear error messages when GPU is unavailable
   - Include CPU fallback path testing in all scenarios

## Memory Tracking and Performance Monitoring

### Host Memory Statistics

BitNet.rs now includes comprehensive host memory tracking using the `sysinfo` crate, providing real-time monitoring of system memory usage alongside GPU operations.

#### DeviceStats with Memory Tracking

The `DeviceStats` structure now includes actual memory usage statistics:

```rust
use bitnet_kernels::device_aware::DeviceAwareQuantizer;

let quantizer = DeviceAwareQuantizer::new(Device::Cpu)?;

// Perform some operations
let input = vec![1.0f32; 1024];
let mut output = vec![0u8; 256];
let mut scales = vec![0.0f32; 8];
quantizer.quantize(&input, &mut output, &mut scales, QuantizationType::I2S)?;

// Get comprehensive statistics including memory usage
if let Some(stats) = quantizer.get_stats() {
    println!("Device stats: {}", stats.summary());
    println!("Memory used: {:.2} MB", stats.memory_used_bytes as f64 / (1024.0 * 1024.0));
    println!("Memory total: {:.2} MB", stats.memory_total_bytes as f64 / (1024.0 * 1024.0));
    println!("Memory usage: {:.1}%", 
        (stats.memory_used_bytes as f64 / stats.memory_total_bytes as f64) * 100.0);
}
```

#### Memory Tracking Features

- **Real-time Monitoring**: Memory statistics are updated on each request using `sysinfo::System`
- **Byte-accurate Reporting**: Both used and total memory reported in bytes for precise tracking
- **Human-readable Display**: The `summary()` method includes memory usage with percentage
- **Performance Integration**: Memory tracking integrated with existing performance statistics

#### Platform-Specific CPU Kernel Selection

The device-aware quantizer now automatically selects the best CPU kernel based on platform architecture:

```rust
// Automatic platform detection and optimization
let quantizer = DeviceAwareQuantizer::new(Device::Cpu)?;
println!("Active kernel: {}", quantizer.active_provider());

// Expected outputs:
// - x86_64 with AVX2: "AVX2Kernel"  
// - aarch64 with NEON: "NeonKernel"
// - Fallback systems: "FallbackKernel"
```

#### Memory Tracking Commands

```bash
# Test comprehensive memory tracking implementation
cargo test -p bitnet-kernels --no-default-features --features cpu test_memory_tracking

# Test platform-specific kernel selection
cargo test -p bitnet-kernels --no-default-features --features cpu test_platform_kernel_selection

# Test CPU provider creation across architectures
cargo test -p bitnet-kernels --no-default-features --features cpu test_cpu_provider_creation

# Architecture-specific feature detection tests
cargo test -p bitnet-kernels --no-default-features --features cpu test_x86_64_feature_detection  # x86_64 only
cargo test -p bitnet-kernels --no-default-features --features cpu test_aarch64_feature_detection  # aarch64 only
```

#### Memory and Performance Analysis

The enhanced statistics provide comprehensive monitoring capabilities:

```rust
#[derive(Debug, Clone)]
pub struct DeviceStats {
    pub memory_used_bytes: u64,      // Host memory currently used in bytes
    pub memory_total_bytes: u64,     // Total host memory available in bytes
    pub gpu_efficiency: f64,         // Ratio of GPU operations to total operations
    pub fallback_count: u64,         // Number of times fallback to CPU occurred
    // ... existing fields
}
```

Key statistics methods:
- `summary()`: Human-readable summary with memory usage percentage
- `is_gpu_effective()`: Checks if GPU is being used effectively (>80% efficiency)
- `avg_quantization_time_ms()`: Average time per quantization operation
- `avg_matmul_time_ms()`: Average time per matrix multiplication operation

## Advanced GPU/CUDA Troubleshooting

### GPU Backend Detection Issues

1. **GPU Detection Fails**:
   ```bash
   # Test GPU detection manually
   cargo test -p bitnet-kernels --no-default-features test_gpu_info_summary
   
   # Check system tools availability
   which nvidia-smi rocm-smi
   
   # Test with mock environment
   BITNET_GPU_FAKE="cuda" cargo run -p xtask -- download-model --dry-run
   ```

2. **Incorrect Backend Detection**:
   ```bash
   # Verify system detection (using existing GPU validation example)
   cargo run --example gpu_validation --no-default-features --features gpu
   
   # Override detection for testing
   export BITNET_GPU_FAKE="cuda,metal"
   cargo test -p bitnet-kernels test_gpu_info_mocked_scenarios
   ```

3. **Version Detection Issues**:
   ```bash
   # Check CUDA toolkit installation
   nvcc --version
   which nvcc
   
   # Check ROCm installation  
   rocm-smi --version
   which rocm-smi
   
   # Test GPU detection functionality
   cargo test -p bitnet-kernels --no-default-features test_gpu_info_summary
   ```

4. **Missing System Commands**:
   ```bash
   # Install missing NVIDIA tools
   sudo apt-get install nvidia-utils-* nvidia-cuda-toolkit
   
   # Install missing AMD tools
   sudo apt-get install rocm-smi-lib rocm-dev
   
   # Verify installation
   nvidia-smi --query-gpu=gpu_name --format=csv,noheader
   rocm-smi --showid
   ```

### GPU Detection and Initialization Issues

1. **CUDA Driver/Runtime Mismatch**:
   ```bash
   # Check NVIDIA driver version
   nvidia-smi
   
   # Check CUDA runtime version
   nvcc --version
   
   # Verify cudarc compatibility
   cargo test -p bitnet-kernels --no-default-features --features gpu test_cuda_kernel_creation
   ```

2. **GPU Memory Issues**:
   ```bash
   # Monitor GPU memory usage
   nvidia-smi -l 1  # Update every second
   
   # Test memory allocation patterns
   cargo test -p bitnet-kernels --no-default-features --features gpu test_cuda_memory_management --ignored
   
   # Check for memory leaks
   cargo test -p bitnet-kernels --no-default-features --features gpu test_memory_cleanup --ignored
   
   # Test host memory tracking and statistics
   cargo test -p bitnet-kernels --no-default-features --features cpu test_memory_tracking
   ```

3. **Compute Capability Issues**:
   ```bash
   # Query device compute capability
   cargo run --example gpu_validation --no-default-features --features gpu | grep "compute capability"
   
   # Test operations on different compute capabilities
   cargo test -p bitnet-kernels --no-default-features --features gpu test_device_capability_validation
   ```

### Performance Debugging

1. **GPU vs CPU Performance Analysis**:
   ```bash
   # Run comprehensive performance comparison
   cargo test -p bitnet-kernels --no-default-features --features gpu test_gpu_vs_cpu_quantization_accuracy --ignored
   
   # Profile GPU kernel execution
   cargo test -p bitnet-kernels --no-default-features --features gpu test_cuda_numerical_accuracy --ignored
   ```

2. **Memory Transfer Optimization**:
   ```bash
   # Test memory access patterns
   cargo test -p bitnet-kernels --no-default-features --features gpu test_memory_access_patterns --ignored
   
   # Validate optimized memory layouts
   cargo test -p bitnet-kernels --no-default-features --features gpu test_memory_optimization --ignored
   ```

### Fallback and Error Handling

1. **GPU Unavailable Scenarios**:
   ```bash
   # Test CPU fallback when GPU unavailable
   CUDA_VISIBLE_DEVICES="" cargo test --workspace --no-default-features --features gpu
   
   # Test partial GPU failure handling
   cargo test -p bitnet-kernels --no-default-features --features gpu test_gpu_quantization_fallback --ignored
   ```

2. **Multi-GPU Configuration**:
   ```bash
   # Test device selection
   cargo test -p bitnet-kernels --no-default-features --features gpu test_multi_gpu_selection --ignored
   
   # Test concurrent operations
   cargo test -p bitnet-kernels --no-default-features --features gpu test_concurrent_gpu_operations --ignored
   ```

### Common Error Messages and Solutions

1. **"CUDA driver version is insufficient"**:
   - Update NVIDIA drivers to support installed CUDA toolkit
   - Check compatibility matrix at nvidia.com/drivers

2. **"out of memory" during GPU operations**:
   - Reduce batch sizes or model parameters
   - Enable GPU memory management optimizations
   - Check for memory leaks in previous operations

3. **"device kernel execution timed out"**:
   - Reduce operation complexity or batch size
   - Check for infinite loops in CUDA kernels
   - Monitor GPU temperature and power limits

4. **"no CUDA-capable device is detected"**:
   - Verify GPU is CUDA-compatible (not AMD/Intel)
   - Check GPU is not being used by other processes
   - Ensure proper driver installation

### Debug Logging and Monitoring

1. **Enable GPU Debug Logging**:
   ```bash
   # Enable CUDA-specific logging
   RUST_LOG=bitnet_kernels::gpu=debug cargo test -p bitnet-kernels --no-default-features --features gpu
   
   # Enable cudarc internal logging
   CUDA_LOG_LEVEL=debug cargo test -p bitnet-kernels --no-default-features --features gpu
   ```

2. **Performance Monitoring**:
   ```bash
   # Profile GPU operations
   nsys profile cargo test -p bitnet-kernels --no-default-features --features gpu test_gpu_performance --ignored
   
   # Monitor GPU utilization
   nvidia-smi dmon -s puc -d 1
   ```

## GPU Development Recipes

```bash
# GPU backend detection and availability
cargo test -p bitnet-kernels --no-default-features test_gpu_info_summary

# Mock GPU testing scenarios
BITNET_GPU_FAKE="cuda,rocm" cargo test -p bitnet-kernels test_gpu_info_mocked_scenarios

# GPU smoke test (basic availability)
cargo test -p bitnet-kernels --no-default-features --features gpu --test gpu_smoke

# CUDA device information and capabilities
cargo run --example gpu_validation --no-default-features --features gpu

# GPU quantization accuracy validation
cargo test -p bitnet-kernels --no-default-features --features gpu test_gpu_vs_cpu_quantization_accuracy --ignored

# GPU memory management and cleanup testing
cargo test -p bitnet-kernels --no-default-features --features gpu test_cuda_memory_management --ignored

# GPU fallback mechanism testing
cargo test -p bitnet-kernels --no-default-features --features gpu test_gpu_quantization_fallback --ignored

# Multi-GPU and concurrent operations
cargo test -p bitnet-kernels --no-default-features --features gpu test_concurrent_gpu_operations --ignored

# GPU numerical accuracy verification
cargo test -p bitnet-kernels --no-default-features --features gpu test_cuda_numerical_accuracy --ignored

# GPU vs CPU parity testing across quantization schemes
cargo test --workspace --no-default-features --features cuda gpu_parity
```
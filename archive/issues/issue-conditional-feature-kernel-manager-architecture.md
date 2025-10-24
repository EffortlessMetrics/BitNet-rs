# [Architecture] Conditional Feature Compilation in Kernel Manager

## Problem Description

The `KernelManager::new` function relies heavily on conditional compilation (`#[cfg(...)]`) to include or exclude kernel providers based on feature flags, creating a fragmented and build-dependent architecture that complicates deployment and runtime behavior.

## Environment

- **File**: `crates/bitnet-kernels/src/lib.rs`
- **Function**: `KernelManager::new`
- **Component**: Kernel Management System
- **Rust Version**: 1.90.0+ (2024 edition)
- **Features**: `gpu`, `avx2`, `avx512`, `neon`, `ffi`

## Root Cause Analysis

The current kernel manager implementation uses extensive conditional compilation:

### **Current Implementation:**
```rust
impl KernelManager {
    pub fn new() -> Self {
        #[allow(unused_mut)]
        let mut providers: Vec<Box<dyn KernelProvider>> = vec![Box::new(cpu::FallbackKernel)];

        // Add GPU kernels first (highest priority)
        #[cfg(feature = "gpu")]
        {
            if let Ok(cuda_kernel) = gpu::CudaKernel::new() {
                if cuda_kernel.is_available() {
                    log::info!("CUDA kernel available, adding to providers");
                    providers.insert(0, Box::new(cuda_kernel));
                }
            }
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        {
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
                providers.insert(insert_pos, Box::new(cpu::Avx512Kernel));
            }
        }

        // ... more conditional compilation
    }
}
```

### **Problems Identified:**
1. **Build-Time Dependencies**: Available kernels determined at compile time, not runtime
2. **Complex Feature Matrix**: Multiple combinations of features create testing complexity
3. **Deployment Rigidity**: Single binary cannot adapt to different hardware environments
4. **Inconsistent Behavior**: Same code produces different functionality based on build flags
5. **Development Overhead**: Multiple build configurations needed for comprehensive testing

## Impact Assessment

### **Severity**: Medium
### **Affected Operations**: Kernel selection and compute optimization
### **Business Impact**: Deployment complexity and runtime inflexibility

**Current Limitations:**
- Cannot deploy single binary across different hardware configurations
- Build matrix complexity for CI/CD systems
- Runtime kernel discovery limited by compile-time decisions
- Inconsistent performance characteristics across builds

## Proposed Solution

### **Primary Approach**: Runtime Kernel Discovery Architecture

Replace compile-time conditional compilation with runtime kernel discovery and dynamic loading, maintaining performance while improving deployment flexibility.

### **Implementation Strategy:**

#### **1. Unified Kernel Registry**
```rust
use std::sync::OnceLock;
use std::collections::HashMap;

pub struct KernelRegistry {
    available_kernels: HashMap<KernelType, Box<dyn KernelProvider>>,
    capability_cache: OnceLock<SystemCapabilities>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum KernelType {
    CudaGpu,
    Avx512Cpu,
    Avx2Cpu,
    NeonArm,
    FfiCpp,
    FallbackCpu,
}

#[derive(Debug, Clone)]
pub struct SystemCapabilities {
    pub has_cuda: bool,
    pub cuda_compute_capability: Option<(u32, u32)>,
    pub has_avx512: bool,
    pub has_avx2: bool,
    pub has_neon: bool,
    pub cpu_cores: usize,
    pub memory_gb: f64,
}

impl KernelRegistry {
    fn new() -> Self {
        Self {
            available_kernels: HashMap::new(),
            capability_cache: OnceLock::new(),
        }
    }

    fn register_kernel(&mut self, kernel_type: KernelType, provider: Box<dyn KernelProvider>) {
        self.available_kernels.insert(kernel_type, provider);
    }

    fn detect_system_capabilities() -> SystemCapabilities {
        SystemCapabilities {
            has_cuda: Self::detect_cuda(),
            cuda_compute_capability: Self::get_cuda_compute_capability(),
            has_avx512: Self::detect_avx512(),
            has_avx2: Self::detect_avx2(),
            has_neon: Self::detect_neon(),
            cpu_cores: num_cpus::get(),
            memory_gb: Self::get_system_memory_gb(),
        }
    }

    fn detect_cuda() -> bool {
        // Attempt to initialize CUDA runtime
        #[cfg(feature = "gpu")]
        {
            match gpu::CudaKernel::new() {
                Ok(kernel) => kernel.is_available(),
                Err(_) => false,
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }

    fn detect_avx512() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    fn detect_avx2() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            is_x86_feature_detected!("avx2")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    fn detect_neon() -> bool {
        #[cfg(target_arch = "aarch64")]
        {
            std::arch::is_aarch64_feature_detected!("neon")
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            false
        }
    }
}
```

#### **2. Dynamic Kernel Manager**
```rust
impl KernelManager {
    pub fn new() -> Self {
        let mut registry = KernelRegistry::new();
        let capabilities = registry.detect_system_capabilities();

        // Always register fallback kernel
        registry.register_kernel(
            KernelType::FallbackCpu,
            Box::new(cpu::FallbackKernel),
        );

        // Register kernels based on runtime capabilities
        Self::register_available_kernels(&mut registry, &capabilities);

        // Select optimal kernel provider chain
        let providers = Self::create_provider_chain(&registry, &capabilities);

        Self {
            providers,
            selected: OnceLock::new(),
            capabilities: Some(capabilities),
        }
    }

    fn register_available_kernels(
        registry: &mut KernelRegistry,
        capabilities: &SystemCapabilities,
    ) {
        // GPU kernels
        if capabilities.has_cuda {
            #[cfg(feature = "gpu")]
            {
                if let Ok(cuda_kernel) = gpu::CudaKernel::new() {
                    if cuda_kernel.is_available() {
                        log::info!("Registering CUDA kernel (compute capability: {:?})",
                                 capabilities.cuda_compute_capability);
                        registry.register_kernel(KernelType::CudaGpu, Box::new(cuda_kernel));
                    }
                } else {
                    log::debug!("CUDA kernel creation failed despite detection");
                }
            }
            #[cfg(not(feature = "gpu"))]
            {
                log::warn!("CUDA detected but GPU feature not compiled in");
            }
        }

        // CPU SIMD kernels
        if capabilities.has_avx512 {
            #[cfg(feature = "avx512")]
            {
                log::info!("Registering AVX-512 kernel");
                registry.register_kernel(KernelType::Avx512Cpu, Box::new(cpu::Avx512Kernel));
            }
            #[cfg(not(feature = "avx512"))]
            {
                log::warn!("AVX-512 detected but feature not compiled in");
            }
        }

        if capabilities.has_avx2 {
            #[cfg(feature = "avx2")]
            {
                log::info!("Registering AVX2 kernel");
                registry.register_kernel(KernelType::Avx2Cpu, Box::new(cpu::Avx2Kernel));
            }
            #[cfg(not(feature = "avx2"))]
            {
                log::warn!("AVX2 detected but feature not compiled in");
            }
        }

        if capabilities.has_neon {
            #[cfg(feature = "neon")]
            {
                log::info!("Registering NEON kernel");
                registry.register_kernel(KernelType::NeonArm, Box::new(cpu::NeonKernel));
            }
            #[cfg(not(feature = "neon"))]
            {
                log::warn!("NEON detected but feature not compiled in");
            }
        }

        // FFI kernel (as fallback)
        #[cfg(feature = "ffi")]
        {
            if let Ok(ffi_kernel) = ffi::FfiKernel::new() {
                if ffi_kernel.is_available() {
                    log::info!("Registering FFI kernel as fallback");
                    registry.register_kernel(KernelType::FfiCpp, Box::new(ffi_kernel));
                }
            }
        }
    }

    fn create_provider_chain(
        registry: &KernelRegistry,
        capabilities: &SystemCapabilities,
    ) -> Vec<Box<dyn KernelProvider>> {
        let mut providers = Vec::new();

        // Priority order: GPU > AVX512 > AVX2 > NEON > FFI > Fallback
        let priority_order = [
            KernelType::CudaGpu,
            KernelType::Avx512Cpu,
            KernelType::Avx2Cpu,
            KernelType::NeonArm,
            KernelType::FfiCpp,
            KernelType::FallbackCpu,
        ];

        for kernel_type in &priority_order {
            if let Some(provider) = registry.available_kernels.get(kernel_type) {
                // Clone the provider (this requires implementing Clone for providers)
                if let Ok(cloned_provider) = provider.try_clone() {
                    providers.push(cloned_provider);
                }
            }
        }

        // Ensure we always have at least the fallback
        if providers.is_empty() {
            providers.push(Box::new(cpu::FallbackKernel));
        }

        log::info!("Kernel provider chain: {:?}",
                   providers.iter().map(|p| p.name()).collect::<Vec<_>>());

        providers
    }

    pub fn get_system_capabilities(&self) -> Option<&SystemCapabilities> {
        self.capabilities.as_ref()
    }

    pub fn get_available_kernel_types(&self) -> Vec<&str> {
        self.providers.iter().map(|p| p.name()).collect()
    }

    pub fn force_kernel_type(&mut self, kernel_type: &str) -> Result<()> {
        if let Some(provider) = self.providers.iter()
            .find(|p| p.name() == kernel_type) {

            // Create new manager with only the requested provider
            let forced_provider = provider.try_clone()?;
            self.providers = vec![forced_provider];
            self.selected = OnceLock::new(); // Reset selection

            log::info!("Forced kernel type to: {}", kernel_type);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Kernel type '{}' not available", kernel_type))
        }
    }
}
```

#### **3. Enhanced Provider Trait**
```rust
pub trait KernelProvider: Send + Sync {
    fn name(&self) -> &str;
    fn is_available(&self) -> bool;
    fn try_clone(&self) -> Result<Box<dyn KernelProvider>>;

    // Capability reporting
    fn supported_operations(&self) -> Vec<OperationType>;
    fn performance_characteristics(&self) -> PerformanceProfile;
    fn memory_requirements(&self) -> MemoryRequirements;

    // Core operations
    fn matmul_i2s(&self, a: &[i8], b: &[u8], c: &mut [f32], m: usize, n: usize, k: usize) -> Result<()>;
    fn quantize(&self, input: &[f32], output: &mut [u8], scales: &mut [f32], qtype: QuantizationType) -> Result<()>;
}

#[derive(Debug, Clone)]
pub struct PerformanceProfile {
    pub relative_speed: f32,  // Relative to fallback kernel
    pub memory_efficiency: f32,
    pub parallel_efficiency: f32,
    pub precision_level: PrecisionLevel,
}

#[derive(Debug, Clone)]
pub enum PrecisionLevel {
    Full,      // Full precision
    Mixed,     // Mixed precision
    Reduced,   // Reduced precision but acceptable
}

#[derive(Debug, Clone)]
pub struct MemoryRequirements {
    pub min_memory_mb: u64,
    pub optimal_memory_mb: u64,
    pub supports_streaming: bool,
}
```

#### **4. Configuration-Driven Selection**
```rust
#[derive(Debug, Clone, Deserialize)]
pub struct KernelConfig {
    pub preferred_kernel: Option<String>,
    pub fallback_strategy: FallbackStrategy,
    pub performance_priority: PerformancePriority,
    pub memory_limit_mb: Option<u64>,
    pub force_cpu_only: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub enum FallbackStrategy {
    BestAvailable,
    FastestFirst,
    MemoryEfficient,
    HighestPrecision,
}

#[derive(Debug, Clone, Deserialize)]
pub enum PerformancePriority {
    Speed,
    Memory,
    Precision,
    Balanced,
}

impl KernelManager {
    pub fn new_with_config(config: &KernelConfig) -> Self {
        let mut manager = Self::new();

        // Apply configuration preferences
        if config.force_cpu_only {
            manager.providers.retain(|p| !p.name().contains("GPU"));
        }

        if let Some(memory_limit) = config.memory_limit_mb {
            manager.providers.retain(|p| {
                p.memory_requirements().min_memory_mb <= memory_limit
            });
        }

        // Reorder providers based on priority
        match config.performance_priority {
            PerformancePriority::Speed => {
                manager.providers.sort_by(|a, b| {
                    b.performance_characteristics().relative_speed
                        .partial_cmp(&a.performance_characteristics().relative_speed)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            PerformancePriority::Memory => {
                manager.providers.sort_by(|a, b| {
                    a.memory_requirements().min_memory_mb
                        .cmp(&b.memory_requirements().min_memory_mb)
                });
            }
            // ... other priority implementations
        }

        manager
    }
}
```

## Implementation Plan

### **Phase 1: System Capability Detection (Week 1)**
- Implement comprehensive hardware detection
- Create capability caching system
- Add runtime CUDA detection
- Implement CPU feature detection

### **Phase 2: Dynamic Registry (Week 2)**
- Build kernel registry infrastructure
- Implement provider chain creation
- Add kernel type management
- Create provider cloning mechanism

### **Phase 3: Configuration System (Week 3)**
- Add configuration-driven kernel selection
- Implement fallback strategies
- Create performance-based prioritization
- Add memory-aware filtering

### **Phase 4: Enhanced Provider Interface (Week 4)**
- Extend provider trait with capability reporting
- Add performance characteristics
- Implement memory requirement specification
- Create provider benchmarking

## Testing Strategy

### **Unit Tests:**
```rust
#[cfg(test)]
mod kernel_manager_tests {
    use super::*;

    #[test]
    fn test_capability_detection() {
        let capabilities = KernelRegistry::detect_system_capabilities();

        // Basic sanity checks
        assert!(capabilities.cpu_cores > 0);
        assert!(capabilities.memory_gb > 0.0);

        // Architecture-specific tests
        #[cfg(target_arch = "x86_64")]
        {
            // Should detect at least SSE2 on modern x86_64
            assert!(is_x86_feature_detected!("sse2"));
        }
    }

    #[test]
    fn test_dynamic_kernel_registration() {
        let mut registry = KernelRegistry::new();
        let capabilities = SystemCapabilities {
            has_cuda: false,
            has_avx2: true,
            has_avx512: false,
            has_neon: false,
            cpu_cores: 8,
            memory_gb: 16.0,
            cuda_compute_capability: None,
        };

        KernelManager::register_available_kernels(&mut registry, &capabilities);

        // Should register fallback and possibly AVX2
        assert!(!registry.available_kernels.is_empty());
        assert!(registry.available_kernels.contains_key(&KernelType::FallbackCpu));
    }

    #[test]
    fn test_provider_chain_creation() {
        let manager = KernelManager::new();

        // Should always have at least one provider
        assert!(!manager.providers.is_empty());

        // Should always have fallback as last resort
        assert!(manager.providers.iter().any(|p| p.name().contains("Fallback")));
    }
}
```

### **Integration Tests:**
```rust
#[test]
fn test_kernel_selection_consistency() {
    let manager = KernelManager::new();

    // Test that kernel selection is deterministic
    let first_selection = manager.select_best_kernel(&[]).unwrap();
    let second_selection = manager.select_best_kernel(&[]).unwrap();

    assert_eq!(first_selection.name(), second_selection.name());
}

#[test]
fn test_configuration_driven_selection() {
    let config = KernelConfig {
        preferred_kernel: Some("AVX2".to_string()),
        fallback_strategy: FallbackStrategy::BestAvailable,
        performance_priority: PerformancePriority::Speed,
        memory_limit_mb: Some(1024),
        force_cpu_only: true,
    };

    let manager = KernelManager::new_with_config(&config);

    // Should not contain GPU providers
    assert!(!manager.providers.iter().any(|p| p.name().contains("GPU")));
}
```

## Success Metrics

### **Functionality:**
- [ ] Runtime kernel detection replaces compile-time conditionals
- [ ] Single binary supports multiple hardware configurations
- [ ] Configuration-driven kernel selection works correctly
- [ ] Fallback mechanisms ensure robustness

### **Performance:**
- [ ] Kernel selection overhead <1ms
- [ ] Optimal kernel chosen based on hardware capabilities
- [ ] No performance regression compared to compile-time selection
- [ ] Memory usage overhead <1MB

### **Deployment:**
- [ ] Single binary works across different hardware
- [ ] Clear logging of available and selected kernels
- [ ] Configuration file support for deployment tuning
- [ ] Graceful degradation when optimal kernels unavailable

## Acceptance Criteria

- [ ] `KernelManager::new` uses runtime detection instead of conditional compilation
- [ ] System capabilities are detected accurately at runtime
- [ ] Kernel providers are registered dynamically based on hardware
- [ ] Configuration system allows deployment-time kernel selection
- [ ] Logging provides clear visibility into kernel selection process
- [ ] Performance characteristics are preserved or improved
- [ ] Single binary supports multiple deployment environments
- [ ] Documentation explains new architecture and configuration options

## Labels

- `architecture`
- `kernel-management`
- `runtime-detection`
- `deployment-flexibility`
- `configuration`

## Related Issues

- **Dependencies**: Issue #XXX (Kernel Provider Interface)
- **Related**: Issue #XXX (Build System Simplification), Issue #XXX (Performance Optimization)
- **Enables**: Flexible deployment, improved testing, better hardware utilization

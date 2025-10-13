# [PERF] TL2 Configuration Runtime SIMD Detection Architecture

## Problem Description

The `TL2Config::default` implementation uses compile-time conditional compilation for SIMD feature detection, preventing optimal runtime adaptation to actual hardware capabilities. This approach limits performance on heterogeneous deployment environments where the optimal SIMD configuration cannot be determined at compile time.

## Environment

- **Component**: `bitnet-quantization` crate
- **File**: `crates/bitnet-quantization/src/tl2.rs`
- **Rust Version**: 1.90.0+ (2024 edition)
- **Target Architectures**: x86_64, ARM64, with focus on SIMD optimization
- **Quantization Method**: TL2 (Table Lookup 2-bit quantization)

## Current Implementation Analysis

### Compile-Time SIMD Detection Limitation
```rust
impl Default for TL2Config {
    fn default() -> Self {
        // PROBLEM: Compile-time feature detection prevents runtime optimization
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let (use_avx512, use_avx2) =
            (is_x86_feature_detected!("avx512f"), is_x86_feature_detected!("avx2"));

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        let (use_avx512, use_avx2) = (false, false); // Misses ARM NEON opportunities

        Self {
            block_size: 128, // Fixed size doesn't adapt to SIMD width
            lookup_table_size: 256,
            use_avx512,
            use_avx2,
            precision_bits: 2,
            vectorized_tables: true,
        }
    }
}
```

### Missing Optimization Opportunities
1. **ARM64 NEON**: No detection for ARM NEON SIMD capabilities
2. **Dynamic Sizing**: Block sizes not optimized per SIMD architecture
3. **Deployment Flexibility**: Cannot adapt to different hardware in same binary
4. **Performance Suboptimality**: Missing AVX-512 vs AVX2 vs NEON selection

## Root Cause Analysis

1. **Compile-Time Constraints**: SIMD detection at compilation prevents runtime adaptation
2. **Architecture Gaps**: Missing ARM64 NEON and other architectures
3. **Fixed Configuration**: Block sizes and parameters don't adapt to SIMD capabilities
4. **Binary Limitations**: One configuration per compilation target

## Impact Assessment

**Severity**: Medium-High - Suboptimal TL2 quantization performance

**Performance Impact**:
- Missed AVX-512 acceleration opportunities (2x performance)
- Suboptimal block sizes for different SIMD widths
- No ARM64 NEON optimization path
- Fixed configuration regardless of actual hardware

**Deployment Impact**:
- Need separate binaries for different CPU targets
- Suboptimal performance in heterogeneous environments
- Complex deployment for cloud environments

## Proposed Solution

### Runtime SIMD Detection with Adaptive Configuration

```rust
use std::sync::OnceLock;

/// SIMD capabilities detected at runtime
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SimdCapability {
    Scalar,
    Avx2,
    Avx512,
    Neon,
    NeonDotprod,
    SveBf16,  // ARM SVE with BF16
}

impl SimdCapability {
    /// Detect best available SIMD capability at runtime
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
                return SimdCapability::Avx512;
            }
            if is_x86_feature_detected!("avx2") {
                return SimdCapability::Avx2;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if is_aarch64_feature_detected!("bf16") && is_aarch64_feature_detected!("sve") {
                return SimdCapability::SveBf16;
            }
            if is_aarch64_feature_detected!("dotprod") {
                return SimdCapability::NeonDotprod;
            }
            if is_aarch64_feature_detected!("neon") {
                return SimdCapability::Neon;
            }
        }

        SimdCapability::Scalar
    }

    /// Get optimal vector width for this SIMD capability
    pub fn vector_width(self) -> usize {
        match self {
            SimdCapability::Avx512 => 64,    // 512 bits / 8 bits per element
            SimdCapability::Avx2 => 32,      // 256 bits / 8 bits per element
            SimdCapability::NeonDotprod => 16, // 128 bits / 8 bits per element
            SimdCapability::Neon => 16,      // 128 bits / 8 bits per element
            SimdCapability::SveBf16 => 64,   // Variable, assume 512-bit
            SimdCapability::Scalar => 1,
        }
    }

    /// Get optimal block size for TL2 quantization
    pub fn optimal_block_size(self) -> usize {
        match self {
            SimdCapability::Avx512 => 256,   // Large blocks for 512-bit vectors
            SimdCapability::Avx2 => 128,     // Medium blocks for 256-bit vectors
            SimdCapability::NeonDotprod => 64, // Smaller blocks with dot product
            SimdCapability::Neon => 64,      // Standard NEON blocks
            SimdCapability::SveBf16 => 256,  // Large blocks for SVE
            SimdCapability::Scalar => 32,    // Small blocks for scalar code
        }
    }

    /// Check if this capability supports vectorized table lookups
    pub fn supports_vectorized_tables(self) -> bool {
        match self {
            SimdCapability::Avx512 | SimdCapability::Avx2 => true,
            SimdCapability::NeonDotprod | SimdCapability::Neon => true,
            SimdCapability::SveBf16 => true,
            SimdCapability::Scalar => false,
        }
    }
}

/// Global SIMD capability - detected once at program start
static DETECTED_SIMD: OnceLock<SimdCapability> = OnceLock::new();

pub fn get_simd_capability() -> SimdCapability {
    *DETECTED_SIMD.get_or_init(SimdCapability::detect)
}

/// Enhanced TL2 configuration with runtime SIMD optimization
#[derive(Debug, Clone)]
pub struct TL2Config {
    /// Block size optimized for SIMD width
    pub block_size: usize,
    /// Lookup table size
    pub lookup_table_size: usize,
    /// SIMD capability being used
    pub simd_capability: SimdCapability,
    /// Precision bits for quantization
    pub precision_bits: u8,
    /// Enable vectorized table operations
    pub vectorized_tables: bool,
    /// Cache alignment for optimal SIMD access
    pub cache_alignment: usize,
}

impl TL2Config {
    /// Create configuration optimized for detected hardware
    pub fn optimized() -> Self {
        let simd_capability = get_simd_capability();

        Self {
            block_size: simd_capability.optimal_block_size(),
            lookup_table_size: Self::optimal_table_size(simd_capability),
            simd_capability,
            precision_bits: 2,
            vectorized_tables: simd_capability.supports_vectorized_tables(),
            cache_alignment: Self::optimal_cache_alignment(simd_capability),
        }
    }

    /// Create configuration for specific SIMD capability
    pub fn for_simd(capability: SimdCapability) -> Self {
        Self {
            block_size: capability.optimal_block_size(),
            lookup_table_size: Self::optimal_table_size(capability),
            simd_capability: capability,
            precision_bits: 2,
            vectorized_tables: capability.supports_vectorized_tables(),
            cache_alignment: Self::optimal_cache_alignment(capability),
        }
    }

    /// Create high-performance configuration
    pub fn high_performance() -> Self {
        let mut config = Self::optimized();

        // Increase block size for better vectorization
        config.block_size = std::cmp::max(config.block_size, 128);

        // Larger lookup tables for better cache locality
        config.lookup_table_size = 512;

        // Always enable vectorized tables if supported
        config.vectorized_tables = config.simd_capability.supports_vectorized_tables();

        config
    }

    /// Create memory-optimized configuration
    pub fn memory_optimized() -> Self {
        let mut config = Self::optimized();

        // Smaller block sizes to reduce memory usage
        config.block_size = std::cmp::min(config.block_size, 64);

        // Smaller lookup tables
        config.lookup_table_size = 128;

        config
    }

    /// Get optimal lookup table size for SIMD capability
    fn optimal_table_size(capability: SimdCapability) -> usize {
        match capability {
            SimdCapability::Avx512 => 512,   // Larger tables for wide vectors
            SimdCapability::Avx2 => 256,     // Standard table size
            SimdCapability::NeonDotprod => 256, // Good cache locality
            SimdCapability::Neon => 128,     // Smaller cache on ARM
            SimdCapability::SveBf16 => 512,  // Large tables for SVE
            SimdCapability::Scalar => 64,    // Minimize cache pressure
        }
    }

    /// Get optimal cache alignment for SIMD capability
    fn optimal_cache_alignment(capability: SimdCapability) -> usize {
        match capability {
            SimdCapability::Avx512 => 64,    // 512-bit alignment
            SimdCapability::Avx2 => 32,      // 256-bit alignment
            SimdCapability::NeonDotprod |
            SimdCapability::Neon => 16,      // 128-bit alignment
            SimdCapability::SveBf16 => 64,   // SVE alignment
            SimdCapability::Scalar => 8,     // Basic alignment
        }
    }

    /// Validate configuration consistency
    pub fn validate(&self) -> Result<(), TL2ConfigError> {
        if self.block_size == 0 {
            return Err(TL2ConfigError::InvalidBlockSize(self.block_size));
        }

        if self.lookup_table_size == 0 || !self.lookup_table_size.is_power_of_two() {
            return Err(TL2ConfigError::InvalidTableSize(self.lookup_table_size));
        }

        if self.precision_bits == 0 || self.precision_bits > 8 {
            return Err(TL2ConfigError::InvalidPrecision(self.precision_bits));
        }

        // Ensure block size is compatible with SIMD width
        let vector_width = self.simd_capability.vector_width();
        if self.block_size % vector_width != 0 {
            return Err(TL2ConfigError::MisalignedBlockSize {
                block_size: self.block_size,
                vector_width,
            });
        }

        Ok(())
    }

    /// Get performance characteristics of this configuration
    pub fn performance_profile(&self) -> TL2PerformanceProfile {
        TL2PerformanceProfile {
            simd_capability: self.simd_capability,
            vector_width: self.simd_capability.vector_width(),
            expected_speedup: self.estimate_speedup(),
            memory_usage: self.estimate_memory_usage(),
            cache_efficiency: self.estimate_cache_efficiency(),
        }
    }

    fn estimate_speedup(&self) -> f32 {
        match self.simd_capability {
            SimdCapability::Avx512 => 8.0,
            SimdCapability::Avx2 => 4.0,
            SimdCapability::NeonDotprod => 4.0,
            SimdCapability::Neon => 2.0,
            SimdCapability::SveBf16 => 8.0,
            SimdCapability::Scalar => 1.0,
        }
    }

    fn estimate_memory_usage(&self) -> usize {
        // Estimate based on lookup table and block processing memory
        self.lookup_table_size * 4 + // Table storage (4 bytes per entry)
        self.block_size * 2 +         // Block processing buffers
        self.cache_alignment * 2      // Alignment padding
    }

    fn estimate_cache_efficiency(&self) -> f32 {
        // Simple heuristic based on table size and access patterns
        let table_cache_lines = (self.lookup_table_size * 4 + 63) / 64;
        if table_cache_lines <= 32 {
            1.0 // Fits in L1 cache
        } else if table_cache_lines <= 512 {
            0.8 // Fits in L2 cache
        } else {
            0.6 // L3 cache or memory access
        }
    }
}

impl Default for TL2Config {
    fn default() -> Self {
        Self::optimized()
    }
}

#[derive(Debug, Clone)]
pub struct TL2PerformanceProfile {
    pub simd_capability: SimdCapability,
    pub vector_width: usize,
    pub expected_speedup: f32,
    pub memory_usage: usize,
    pub cache_efficiency: f32,
}

#[derive(Debug, thiserror::Error)]
pub enum TL2ConfigError {
    #[error("Invalid block size: {0} (must be > 0)")]
    InvalidBlockSize(usize),

    #[error("Invalid table size: {0} (must be power of 2)")]
    InvalidTableSize(usize),

    #[error("Invalid precision: {0} (must be 1-8 bits)")]
    InvalidPrecision(u8),

    #[error("Block size {block_size} not aligned to SIMD vector width {vector_width}")]
    MisalignedBlockSize {
        block_size: usize,
        vector_width: usize,
    },
}

/// Architecture-specific configuration presets
impl TL2Config {
    /// Configuration optimized for Intel/AMD x86_64 with AVX-512
    pub fn intel_avx512() -> Self {
        Self::for_simd(SimdCapability::Avx512)
    }

    /// Configuration optimized for Intel/AMD x86_64 with AVX2
    pub fn intel_avx2() -> Self {
        Self::for_simd(SimdCapability::Avx2)
    }

    /// Configuration optimized for ARM64 with NEON and dot product
    pub fn arm_neon_dotprod() -> Self {
        Self::for_simd(SimdCapability::NeonDotprod)
    }

    /// Configuration optimized for ARM64 with standard NEON
    pub fn arm_neon() -> Self {
        Self::for_simd(SimdCapability::Neon)
    }

    /// Configuration for ARM SVE with BF16 support
    pub fn arm_sve_bf16() -> Self {
        Self::for_simd(SimdCapability::SveBf16)
    }

    /// Fallback scalar configuration
    pub fn scalar() -> Self {
        Self::for_simd(SimdCapability::Scalar)
    }
}
```

## Implementation Plan

### Phase 1: Runtime Detection Infrastructure (Week 1)
- [ ] Implement comprehensive SIMD capability detection
- [ ] Create adaptive configuration system based on detected capabilities
- [ ] Add validation framework for configuration consistency
- [ ] Establish performance profiling and estimation

### Phase 2: Architecture-Specific Optimization (Week 2)
- [ ] Add ARM64 NEON optimization paths
- [ ] Implement AVX-512 specific optimizations
- [ ] Create architecture-specific presets
- [ ] Add cache alignment and memory optimization

### Phase 3: Integration & Testing (Week 3)
- [ ] Replace compile-time detection with runtime system
- [ ] Add comprehensive testing across architectures
- [ ] Benchmark performance improvements
- [ ] Validate numerical accuracy across SIMD implementations

### Phase 4: Production Features (Week 4)
- [ ] Add monitoring and telemetry for SIMD usage
- [ ] Implement configuration caching and optimization
- [ ] Add debugging and diagnostic tools
- [ ] Documentation and migration guide

## Success Criteria

- [ ] **Runtime Adaptation**: Optimal configuration selected based on actual hardware
- [ ] **Architecture Support**: x86_64 AVX2/AVX-512 and ARM64 NEON support
- [ ] **Performance Gains**: >= 2x speedup on SIMD-capable hardware
- [ ] **Memory Efficiency**: Optimal block sizes and cache alignment
- [ ] **Deployment Flexibility**: Single binary adapts to different hardware
- [ ] **Configuration Validation**: Robust error handling and validation

## Related Issues

- #XXX: SIMD quantization kernel optimization
- #XXX: Performance benchmarking automation
- #XXX: Cross-platform SIMD testing
- #XXX: Memory alignment optimization

## Implementation Notes

This runtime SIMD detection approach enables BitNet.rs to automatically adapt TL2 quantization performance to the actual deployment hardware while maintaining single-binary deployment simplicity. The architecture provides clear performance profiling and validation while supporting diverse CPU architectures.

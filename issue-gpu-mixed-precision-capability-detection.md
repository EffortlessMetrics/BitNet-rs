# [GPU] Mixed Precision Capability Detection Implementation

## Problem Description

The `GpuBackend::supports_mixed_precision` function returns a configuration value instead of detecting actual GPU hardware capabilities for mixed precision (FP16/BF16) support, preventing optimal performance tuning and accurate capability reporting.

## Environment

- **File**: `crates/bitnet-inference/src/gpu.rs`
- **Function**: `GpuBackend::supports_mixed_precision`
- **Component**: GPU Backend Capability Detection
- **Features**: `gpu`, CUDA support

## Root Cause Analysis

### **Current Implementation:**
```rust
pub fn supports_mixed_precision(&self) -> bool {
    // Would check actual GPU capabilities
    self.performance_config.enable_mixed_precision
}
```

### **Problems:**
1. **Configuration vs Capability**: Returns config setting, not hardware capability
2. **No Hardware Detection**: Missing actual CUDA device capability queries
3. **Suboptimal Performance**: Cannot automatically enable optimal precision modes

## Proposed Solution

Implement proper GPU capability detection using CUDA device queries:

```rust
pub fn supports_mixed_precision(&self) -> bool {
    #[cfg(feature = "gpu")]
    {
        match self.device.compute_capability() {
            Ok((major, minor)) => {
                let supports_fp16 = major > 5 || (major == 5 && minor >= 3);
                let supports_bf16 = major >= 8;
                (supports_fp16 || supports_bf16) &&
                self.performance_config.enable_mixed_precision
            },
            Err(_) => false,
        }
    }
    #[cfg(not(feature = "gpu"))]
    { false }
}
```

## Success Metrics

- [ ] Accurate hardware capability detection
- [ ] Automatic mixed precision mode selection
- [ ] Performance optimization through proper precision selection

## Labels

- `gpu-capabilities`
- `mixed-precision`
- `hardware-detection`

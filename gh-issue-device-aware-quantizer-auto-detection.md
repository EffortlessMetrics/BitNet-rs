# [Quantization] Implement Hardware Auto-Detection in DeviceAwareQuantizer

## Problem Description

The `DeviceAwareQuantizer::auto_detect` function in `crates/bitnet-quantization/src/device_aware_quantizer.rs` returns a default configuration instead of detecting actual hardware capabilities and selecting optimal quantization settings.

## Environment

- **Component**: `crates/bitnet-quantization/src/device_aware_quantizer.rs`
- **Function**: `DeviceAwareQuantizer::auto_detect`
- **Impact**: Quantization performance optimization

## Current Implementation

```rust
pub fn auto_detect() -> bitnet_common::Result<Self> {
    // In a full implementation, this would detect GPU availability, CPU features, etc.
    // For now, return a default configuration
    Ok(Self::new())
}
```

## Proposed Solution

Implement comprehensive hardware detection:

```rust
pub fn auto_detect() -> bitnet_common::Result<Self> {
    let tolerance_config = ToleranceConfig::default();
    let cpu_backend = CPUQuantizer::new(tolerance_config.clone());

    #[cfg(feature = "gpu")]
    if let Ok(gpu_device) = bitnet_common::Device::new_cuda(0) {
        info!("GPU detected, configuring GPU quantizer");
        return Ok(Self::with_gpu(cpu_backend, gpu_device, tolerance_config));
    }

    info!("CPU-only configuration");
    Ok(Self::cpu_only(cpu_backend, tolerance_config))
}
```

## Implementation Tasks

- [ ] Add GPU availability detection
- [ ] Implement CPU feature detection (AVX2, NEON, etc.)
- [ ] Create optimal configuration selection logic
- [ ] Add hardware capability caching

## Acceptance Criteria

- [ ] Automatically detects GPU availability
- [ ] Selects optimal quantizer configuration for hardware
- [ ] Falls back gracefully to CPU-only mode
- [ ] Caches detection results for performance

## Related Issues

- **Depends on**: Hardware detection infrastructure
- **Blocks**: Optimal quantization performance
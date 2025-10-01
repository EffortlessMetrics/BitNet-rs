# [Performance] KV Cache Prefetch Implementation

## Problem Description

The `KVCache::prefetch` function is a no-op placeholder that doesn't implement platform-specific prefetch instructions to improve memory access patterns during attention computation.

## Environment

- **File**: `crates/bitnet-inference/src/layers/attention.rs`
- **Function**: `KVCache::prefetch`
- **Component**: Attention Layer KV Cache

## Root Cause Analysis

### **Current Implementation:**
```rust
pub fn prefetch(&self, layer_idx: usize, _seq_len: usize) -> Result<()> {
    if layer_idx >= self.k_cache.len() {
        return Err(anyhow::anyhow!("Layer index {} out of bounds", layer_idx));
    }

    // In a full implementation, this would use platform-specific prefetch instructions
    // For now, it's a no-op placeholder
    Ok(())
}
```

### **Issues:**
1. **No Prefetching**: Missing memory prefetch implementation
2. **Cache Miss Penalties**: Higher latency for cache misses during attention
3. **Suboptimal Memory Access**: Not preparing cache lines for upcoming operations

## Proposed Solution

Implement platform-specific prefetch instructions:

```rust
pub fn prefetch(&self, layer_idx: usize, seq_len: usize) -> Result<()> {
    if layer_idx >= self.k_cache.len() {
        return Err(anyhow::anyhow!("Layer index {} out of bounds", layer_idx));
    }

    // Prefetch key and value cache data
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
        _mm_prefetch(self.k_cache[layer_idx].as_ptr() as *const i8, _MM_HINT_T0);
        _mm_prefetch(self.v_cache[layer_idx].as_ptr() as *const i8, _MM_HINT_T0);
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::_prefetch;
        _prefetch(self.k_cache[layer_idx].as_ptr(), _PLDL1KEEP);
        _prefetch(self.v_cache[layer_idx].as_ptr(), _PLDL1KEEP);
    }

    Ok(())
}
```

## Implementation Plan

- Add x86_64 prefetch instructions using `_mm_prefetch`
- Implement ARM64 prefetch using `_prefetch`
- Add intelligent prefetch distance calculation
- Benchmark memory access improvements

## Success Metrics

- [ ] Reduced cache miss rates during attention computation
- [ ] Measurable performance improvement in attention layers
- [ ] Cross-platform prefetch support (x86_64, ARM64)

## Labels

- `memory-optimization`
- `prefetch`
- `attention-performance`
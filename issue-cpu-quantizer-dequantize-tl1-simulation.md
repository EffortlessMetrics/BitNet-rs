# [Quantization] Replace TL1 dequantization simulation with production implementation

## Problem Description

The CPU quantizer's TL1 dequantization uses simulation/placeholder logic instead of production-ready implementation based on BitNet research specifications.

## Environment

- **Component:** CPU quantization, TL1 dequantization
- **Issue:** Simulation code in production paths

## Proposed Solution

1. Implement proper TL1 lookup table dequantization
2. Add BitNet-compliant table generation
3. Optimize for SIMD execution
4. Add accuracy validation against reference

## Implementation Plan

- [ ] Research BitNet TL1 specification requirements
- [ ] Implement proper lookup table generation
- [ ] Add SIMD-optimized dequantization kernels
- [ ] Create accuracy validation tests
- [ ] Benchmark performance vs simulation

---

**Labels:** `quantization`, `cpu-optimization`, `tl1`, `production-ready`
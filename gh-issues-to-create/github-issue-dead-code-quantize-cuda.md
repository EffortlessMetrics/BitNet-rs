# [Quantization] Remove or integrate unused `quantize_cuda` method in I2SQuantizer

## Problem Description

The `quantize_cuda` method in `I2SQuantizer` at `crates/bitnet-quantization/src/i2s.rs:240` is never used, as identified by cargo clippy. This represents either dead code that should be removed or an incomplete feature that needs integration.

## Environment
- **File**: `crates/bitnet-quantization/src/i2s.rs`
- **Line**: 240
- **Method**: `quantize_cuda`
- **Detection**: cargo clippy dead code analysis

## Root Cause Analysis

The `quantize_cuda` method was likely implemented to provide CUDA-accelerated I2S quantization but was never integrated into the quantization dispatch system. The current quantization flow doesn't have a mechanism to select between CPU and GPU implementations.

## Impact Assessment
- **Severity**: Medium
- **Impact**: Code maintenance overhead, potential confusion for developers
- **Affected Components**: `bitnet-quantization` crate
- **Performance**: No current impact, but missing GPU acceleration opportunity

## Proposed Solution

### Option 1: Remove Dead Code (Immediate)
If CUDA quantization is not planned for the near future, remove the method to reduce codebase complexity.

### Option 2: Integrate CUDA Quantization (Recommended)
Implement a device-aware quantization system that allows selecting between CPU and CUDA implementations.

**Implementation Plan:**
1. **Add QuantizationDevice enum** in `bitnet-common/src/types.rs`:
```rust
pub enum QuantizationDevice {
    Cpu,
    Cuda,
}
```

2. **Update I2SQuantizer interface**:
```rust
impl I2SQuantizer {
    pub fn quantize(&self, tensor: &BitNetTensor, device: QuantizationDevice) -> Result<QuantizedTensor> {
        match device {
            QuantizationDevice::Cpu => self.quantize_cpu(tensor),
            QuantizationDevice::Cuda => self.quantize_cuda(tensor),
        }
    }

    fn quantize_cpu(&self, tensor: &BitNetTensor) -> Result<QuantizedTensor> {
        // Existing CPU implementation
    }

    fn quantize_cuda(&self, tensor: &BitNetTensor) -> Result<QuantizedTensor> {
        // Existing CUDA implementation
    }
}
```

3. **Add configuration support** for device selection
4. **Update call sites** to pass device parameter
5. **Add feature gating** with `--features gpu` flag

## Testing Strategy
- **Unit Tests**: Test both CPU and CUDA quantization paths
- **Integration Tests**: Verify device selection works correctly
- **Cross-validation**: Compare CPU vs CUDA quantization results (MSE < 1e-6)
- **Performance Tests**: Benchmark CUDA vs CPU quantization speed

## Acceptance Criteria
- [ ] Either dead code is removed OR CUDA quantization is properly integrated
- [ ] No clippy warnings about unused code
- [ ] If integrated: Device selection works via configuration
- [ ] If integrated: Feature flag `gpu` properly gates CUDA code
- [ ] Tests pass with both CPU and GPU features
- [ ] Documentation updated for new quantization API

## Labels
- `enhancement`
- `quantization`
- `gpu`
- `tech-debt`
- `priority-medium`

## Related Issues
- Related to GPU acceleration efforts
- Part of quantization optimization work
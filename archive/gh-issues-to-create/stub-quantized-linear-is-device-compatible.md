# Stub code: `QuantizedLinear::is_device_compatible` in `quantized_linear.rs` always returns `true`

The `QuantizedLinear::is_device_compatible` function in `crates/bitnet-inference/src/layers/quantized_linear.rs` always returns `true`, indicating that it doesn't actually enforce strict device matching. This is a form of stubbing.

**File:** `crates/bitnet-inference/src/layers/quantized_linear.rs`

**Function:** `QuantizedLinear::is_device_compatible`

**Code:**
```rust
    fn is_device_compatible(&self, _tensor: &BitNetTensor) -> bool {
        // For now, allow any device combination
        // In a full implementation, this would enforce strict device matching
        true
    }
```

## Proposed Fix

The `QuantizedLinear::is_device_compatible` function should be implemented to enforce strict device matching. This would involve comparing the device of the input tensor with the device of the layer.

### Example Implementation

```rust
    fn is_device_compatible(&self, tensor: &BitNetTensor) -> bool {
        tensor.device() == &self.device
    }
```

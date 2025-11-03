# Simulation: `MockTensor` in `property_tests.rs` is a simplified implementation

The `MockTensor` struct in `crates/bitnet-quantization/src/property_tests.rs` is a simplified implementation that uses unsafe code for `as_slice`. This is a form of simulation and should be replaced with a proper tensor implementation.

**File:** `crates/bitnet-quantization/src/property_tests.rs`

**Struct:** `MockTensor`

**Code:**
```rust
#[cfg(test)]
pub struct MockTensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

#[cfg(test)]
impl MockTensor {
    pub fn from_vec(data: Vec<f32>) -> Self {
        let shape = vec![data.len()];
        Self { data, shape }
    }

    pub fn from_vec_with_shape(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }
}

#[cfg(test)]
impl crate::Tensor for MockTensor {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> crate::DType {
        crate::DType::F32
    }

    fn device(&self) -> &crate::Device {
        &crate::Device::Cpu
    }

    fn as_slice<T>(&self) -> Result<&[T], crate::BitNetError> {
        // Unsafe cast for testing - in real implementation this would be properly typed
        unsafe {
            let ptr = self.data.as_ptr() as *const T;
            let slice = std::slice::from_raw_parts(ptr, self.data.len());
            Ok(slice)
        }
    }
}
```

## Proposed Fix

The `MockTensor` struct should be replaced with a proper tensor implementation that does not use unsafe code. This would involve using the `BitNetTensor` struct from `bitnet_common`.

### Example Implementation

```rust
#[cfg(test)]
pub struct MockTensor {
    inner: BitNetTensor,
}

#[cfg(test)]
impl MockTensor {
    pub fn from_vec(data: Vec<f32>) -> Self {
        let shape = vec![data.len()];
        Self { inner: BitNetTensor::from_slice(&data, &shape, &Device::Cpu).unwrap() }
    }

    pub fn from_vec_with_shape(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { inner: BitNetTensor::from_slice(&data, &shape, &Device::Cpu).unwrap() }
    }
}

#[cfg(test)]
impl crate::Tensor for MockTensor {
    fn shape(&self) -> &[usize] {
        self.inner.shape()
    }

    fn dtype(&self) -> crate::DType {
        self.inner.dtype()
    }

    fn device(&self) -> &crate::Device {
        self.inner.device()
    }

    fn as_slice<T>(&self) -> Result<&[T], crate::BitNetError> {
        self.inner.as_slice()
    }
}
```

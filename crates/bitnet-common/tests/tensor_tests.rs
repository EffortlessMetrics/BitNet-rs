#![cfg(feature = "integration-tests")]
//! Comprehensive tensor tests for bitnet-common

use bitnet_common::*;
use candle_core::DType;
use proptest::prelude::*;

#[test]
fn test_mock_tensor_creation() {
    let tensor = MockTensor::new(vec![2, 3]);

    assert_eq!(tensor.shape(), &[2, 3]);
    assert_eq!(tensor.dtype(), DType::F32);
    assert_eq!(tensor.device(), &Device::Cpu);
}

#[test]
fn test_mock_tensor_with_device() {
    let tensor = MockTensor::new(vec![4, 4]).with_device(Device::Cuda(0));

    assert_eq!(tensor.shape(), &[4, 4]);
    assert_eq!(tensor.device(), &Device::Cuda(0));
}

#[test]
fn test_mock_tensor_data_access() {
    let tensor = MockTensor::new(vec![2, 2]);

    // Test as_slice
    let slice: &[f32] = tensor.as_slice().unwrap();
    assert_eq!(slice.len(), 4);

    // All values should be 0.1 (default value in MockTensor::new)
    for &value in slice {
        assert_eq!(value, 0.1);
    }
}

#[test]
fn test_mock_tensor_empty() {
    let tensor = MockTensor::new(vec![0]);

    assert_eq!(tensor.shape(), &[0]);
    let slice: &[f32] = tensor.as_slice().unwrap();
    assert!(slice.is_empty());
}

#[test]
fn test_mock_tensor_large_dimensions() {
    let tensor = MockTensor::new(vec![10, 20, 30]);

    assert_eq!(tensor.shape(), &[10, 20, 30]);
    let slice: &[f32] = tensor.as_slice().unwrap();
    assert_eq!(slice.len(), 10 * 20 * 30);
}

#[test]
fn test_concrete_tensor_mock_variant() {
    let tensor = ConcreteTensor::mock(vec![3, 3]);

    assert_eq!(tensor.shape(), &[3, 3]);
    assert_eq!(tensor.dtype(), DType::F32);
    assert_eq!(tensor.device(), &Device::Cpu);

    let slice: &[f32] = tensor.as_slice().unwrap();
    assert_eq!(slice.len(), 9);
}

#[test]
fn test_concrete_tensor_trait_implementation() {
    let tensor = ConcreteTensor::mock(vec![2, 2]);

    // Test Tensor trait methods
    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(tensor.dtype(), DType::F32);
    assert_eq!(tensor.device(), &Device::Cpu);

    // Test as_slice through trait
    let slice: &[f32] = tensor.as_slice().unwrap();
    assert_eq!(slice.len(), 4);
}

#[test]
fn test_tensor_to_candle_conversion() {
    let tensor = ConcreteTensor::mock(vec![2, 2]);

    // Test conversion to Candle tensor
    let candle_tensor = tensor.to_candle().unwrap();
    assert_eq!(candle_tensor.shape().dims(), &[2, 2]);
    assert_eq!(candle_tensor.dtype(), DType::F32);
}

#[test]
fn test_bitnet_tensor_creation() {
    // Create a simple Candle tensor for testing
    let candle_tensor =
        candle_core::Tensor::zeros(&[2, 2], DType::F32, &candle_core::Device::Cpu).unwrap();
    let bitnet_tensor = BitNetTensor::new(candle_tensor);

    assert_eq!(bitnet_tensor.shape(), &[2, 2]);
    assert_eq!(bitnet_tensor.dtype(), DType::F32);
}

#[test]
fn test_bitnet_tensor_from_slice() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let shape = vec![2, 2];
    let device = Device::Cpu;

    let tensor = BitNetTensor::from_slice(&data, &shape, &device).unwrap();
    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(tensor.dtype(), DType::F32);
}

#[test]
fn test_bitnet_tensor_zeros() {
    let shape = vec![3, 3];
    let device = Device::Cpu;

    let tensor = BitNetTensor::zeros(&shape, DType::F32, &device).unwrap();
    assert_eq!(tensor.shape(), &[3, 3]);
    assert_eq!(tensor.dtype(), DType::F32);
}

#[test]
fn test_bitnet_tensor_inner_access() {
    let candle_tensor =
        candle_core::Tensor::zeros(&[2, 2], DType::F32, &candle_core::Device::Cpu).unwrap();
    let bitnet_tensor = BitNetTensor::new(candle_tensor.clone());

    // Test inner access
    let inner = bitnet_tensor.inner();
    assert_eq!(inner.shape().dims(), candle_tensor.shape().dims());

    // Test into_inner
    let recovered = bitnet_tensor.into_inner();
    assert_eq!(recovered.shape().dims(), candle_tensor.shape().dims());
}

#[test]
fn test_tensor_error_handling() {
    // Test invalid device conversion (when GPU features are not available)
    let device = Device::Cuda(0);
    let _result = BitNetTensor::zeros(&[2, 2], DType::F32, &device);

    // This should fail when GPU features are not enabled
    #[cfg(not(feature = "gpu"))]
    assert!(_result.is_err());
    #[cfg(feature = "gpu")]
    let _ = _result; // Suppress unused variable warning when GPU features are enabled

    // Test Metal device
    let device = Device::Metal;
    let _result = BitNetTensor::zeros(&[2, 2], DType::F32, &device);

    #[cfg(not(feature = "gpu"))]
    assert!(_result.is_err());
    #[cfg(feature = "gpu")]
    let _ = _result; // Suppress unused variable warning when GPU features are enabled
}

#[test]
fn test_tensor_different_dtypes() {
    let device = Device::Cpu;

    // Test different data types
    let f32_tensor = BitNetTensor::zeros(&[2, 2], DType::F32, &device).unwrap();
    assert_eq!(f32_tensor.dtype(), DType::F32);

    let f64_tensor = BitNetTensor::zeros(&[2, 2], DType::F64, &device).unwrap();
    assert_eq!(f64_tensor.dtype(), DType::F64);

    let u32_tensor = BitNetTensor::zeros(&[2, 2], DType::U32, &device).unwrap();
    assert_eq!(u32_tensor.dtype(), DType::U32);
}

#[test]
fn test_concrete_tensor_variants() {
    // Test Mock variant
    let mock_tensor = ConcreteTensor::mock(vec![2, 2]);
    match mock_tensor {
        ConcreteTensor::Mock(_) => {} // Expected
        _ => panic!("Expected Mock variant"),
    }

    // Test BitNet variant
    let candle_tensor =
        candle_core::Tensor::zeros(&[2, 2], DType::F32, &candle_core::Device::Cpu).unwrap();
    let bitnet_tensor = ConcreteTensor::bitnet(candle_tensor);
    match bitnet_tensor {
        ConcreteTensor::BitNet(_) => {} // Expected
        _ => panic!("Expected BitNet variant"),
    }
}

// Property-based tests
proptest! {
    #[test]
    fn test_mock_tensor_arbitrary_shapes(shape in prop::collection::vec(1usize..100, 1..5)) {
        let tensor = MockTensor::new(shape.clone());
        assert_eq!(tensor.shape(), shape.as_slice());
        assert_eq!(tensor.dtype(), DType::F32);

        let expected_size: usize = shape.iter().product();
        let slice: &[f32] = tensor.as_slice().unwrap();
        assert_eq!(slice.len(), expected_size);
    }

    #[test]
    fn test_concrete_tensor_arbitrary_shapes(shape in prop::collection::vec(1usize..50, 1..4)) {
        let tensor = ConcreteTensor::mock(shape.clone());
        assert_eq!(tensor.shape(), shape.as_slice());

        let expected_size: usize = shape.iter().product();
        let slice: &[f32] = tensor.as_slice().unwrap();
        assert_eq!(slice.len(), expected_size);
    }

    #[test]
    fn test_bitnet_tensor_arbitrary_shapes(shape in prop::collection::vec(1usize..20, 1..3)) {
        let device = Device::Cpu;
        let tensor = BitNetTensor::zeros(&shape, DType::F32, &device).unwrap();
        assert_eq!(tensor.shape(), shape.as_slice());
        assert_eq!(tensor.dtype(), DType::F32);
    }

    #[test]
    fn test_mock_tensor_with_arbitrary_devices(device_id in any::<usize>()) {
        let tensor = MockTensor::new(vec![2, 2]).with_device(Device::Cuda(device_id));
        assert_eq!(tensor.device(), &Device::Cuda(device_id));
    }
}

#[test]
fn test_tensor_clone() {
    let tensor = MockTensor::new(vec![2, 2]);
    let cloned = tensor.clone();

    assert_eq!(tensor.shape(), cloned.shape());
    assert_eq!(tensor.dtype(), cloned.dtype());
    assert_eq!(tensor.device(), cloned.device());
}

#[test]
fn test_tensor_debug() {
    let tensor = MockTensor::new(vec![2, 2]);
    let debug_str = format!("{:?}", tensor);
    assert!(debug_str.contains("MockTensor"));

    let concrete_tensor = ConcreteTensor::mock(vec![2, 2]);
    let debug_str = format!("{:?}", concrete_tensor);
    assert!(debug_str.contains("Mock"));
}

#[test]
fn test_tensor_send_sync() {
    // Ensure tensors are Send + Sync for use in async contexts
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<MockTensor>();
    assert_send_sync::<BitNetTensor>();
    assert_send_sync::<ConcreteTensor>();
}

#[test]
fn test_tensor_edge_cases() {
    // Test single element tensor
    let tensor = MockTensor::new(vec![1]);
    assert_eq!(tensor.shape(), &[1]);
    let slice: &[f32] = tensor.as_slice().unwrap();
    assert_eq!(slice.len(), 1);

    // Test high-dimensional tensor
    let tensor = MockTensor::new(vec![2, 2, 2, 2]);
    assert_eq!(tensor.shape(), &[2, 2, 2, 2]);
    let slice: &[f32] = tensor.as_slice().unwrap();
    assert_eq!(slice.len(), 16);
}

#[test]
fn test_tensor_consistency() {
    // Test that tensor operations are consistent
    let shape = vec![3, 4];
    let tensor = ConcreteTensor::mock(shape.clone());

    // Shape should be consistent
    assert_eq!(tensor.shape(), shape.as_slice());

    // Data size should match shape
    let expected_size: usize = shape.iter().product();
    let slice: &[f32] = tensor.as_slice().unwrap();
    assert_eq!(slice.len(), expected_size);

    // Candle conversion should preserve shape
    let candle_tensor = tensor.to_candle().unwrap();
    assert_eq!(candle_tensor.shape().dims(), shape.as_slice());
}

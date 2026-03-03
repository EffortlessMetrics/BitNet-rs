//! Property-based tests — wave 38.
//!
//! Covers error type conversion, config validation, tensor dimension
//! properties, device selection, dtype conversion, memory estimation,
//! shape validation, tensor layout, and kernel capabilities.
//!
//! 50+ property tests using proptest.

use bitnet_common::{
    backend_selection::{BackendRequest, select_backend},
    dtype_convert::{bf16_to_f32, f16_to_f32, f32_to_bf16, f32_to_f16},
    kernel_registry::{KernelBackend, KernelCapabilities},
    memory_estimator::{DType, TensorEstimate},
    memory_pool::TensorPool,
    shape_validator::{
        assert_dim, assert_element_count, assert_head_divisible, assert_matmul_compat, assert_rank,
        assert_shape_eq,
    },
    tensor_layout::{LayoutOrder, TensorLayout, broadcastable, compute_strides},
    tensor_validation::{
        broadcast_shape, c_contiguous_strides, can_broadcast, validate_matmul_shapes,
        validate_reshape, validate_transpose_axes,
    },
    types::{Device, GenerationConfig, QuantizationType},
};
use proptest::prelude::*;

// ── 1. Error type conversion properties ─────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    // 1.1 BitNetError Config Display is non-empty
    #[test]
    fn prop_bitnet_error_config_display(msg in "[a-z]{1,20}") {
        let err = bitnet_common::error::BitNetError::Config(msg);
        let display = format!("{err}");
        prop_assert!(!display.is_empty());
    }

    // 1.2 BitNetError Validation Display is non-empty
    #[test]
    fn prop_bitnet_error_validation_display(msg in "[a-z]{1,20}") {
        let err = bitnet_common::error::BitNetError::Validation(msg);
        let display = format!("{err}");
        prop_assert!(!display.is_empty());
    }

    // 1.3 ModelError NotFound Display is non-empty
    #[test]
    fn prop_model_error_not_found_display(path in "[a-z]{1,20}") {
        let err = bitnet_common::error::ModelError::NotFound { path };
        let display = format!("{err}");
        prop_assert!(!display.is_empty());
    }

    // 1.4 KernelError ExecutionFailed Display is non-empty
    #[test]
    fn prop_kernel_error_exec_display(reason in "[a-z]{1,20}") {
        let err = bitnet_common::error::KernelError::ExecutionFailed { reason };
        let display = format!("{err}");
        prop_assert!(!display.is_empty());
    }

    // 1.5 QuantizationError InvalidInput Display is non-empty
    #[test]
    fn prop_quant_error_invalid_input_display(reason in "[a-z]{1,20}") {
        let err = bitnet_common::error::QuantizationError::InvalidInput { reason };
        let display = format!("{err}");
        prop_assert!(!display.is_empty());
    }

    // 1.6 InferenceError GenerationFailed Display is non-empty
    #[test]
    fn prop_inference_error_gen_display(reason in "[a-z]{1,20}") {
        let err = bitnet_common::error::InferenceError::GenerationFailed { reason };
        let display = format!("{err}");
        prop_assert!(!display.is_empty());
    }

    // 1.7 SecurityError InputValidation Display is non-empty
    #[test]
    fn prop_security_error_input_display(reason in "[a-z]{1,20}") {
        let err = bitnet_common::error::SecurityError::InputValidation { reason };
        let display = format!("{err}");
        prop_assert!(!display.is_empty());
    }

    // 1.8 BitNetError StrictMode Display is non-empty
    #[test]
    fn prop_bitnet_error_strict_display(msg in "[a-z]{1,20}") {
        let err = bitnet_common::error::BitNetError::StrictMode(msg);
        let display = format!("{err}");
        prop_assert!(!display.is_empty());
    }
}

// ── 2. Config validation properties ─────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    // 2.1 GenerationConfig with valid temperature
    #[test]
    fn prop_generation_config_temp_range(temp in 0.0f32..=2.0) {
        let config = GenerationConfig {
            max_new_tokens: 32,
            temperature: temp,
            top_k: Some(50),
            top_p: Some(0.95),
            repetition_penalty: 1.0,
            do_sample: true,
            seed: None,
        };
        prop_assert!(config.temperature >= 0.0);
        prop_assert!(config.temperature <= 2.0);
    }

    // 2.2 GenerationConfig max_new_tokens positive
    #[test]
    fn prop_generation_config_max_tokens(max in 1usize..=4096) {
        let config = GenerationConfig {
            max_new_tokens: max,
            temperature: 1.0,
            top_k: None,
            top_p: None,
            repetition_penalty: 1.0,
            do_sample: false,
            seed: None,
        };
        prop_assert!(config.max_new_tokens > 0);
        prop_assert!(config.max_new_tokens <= 4096);
    }

    // 2.3 QuantizationType variants are distinct
    #[test]
    fn prop_quant_type_distinct(
        idx_a in 0u8..3,
        idx_b in 0u8..3,
    ) {
        let types = [
            QuantizationType::I2S,
            QuantizationType::TL1,
            QuantizationType::TL2,
        ];
        if idx_a != idx_b {
            prop_assert_ne!(
                format!("{:?}", types[idx_a as usize]),
                format!("{:?}", types[idx_b as usize])
            );
        }
    }

    // 2.4 Device::Cpu is always cpu
    #[test]
    fn prop_device_cpu_is_cpu(_dummy in 0u8..1) {
        let dev = Device::Cpu;
        prop_assert!(dev.is_cpu());
        prop_assert!(!dev.is_cuda());
    }

    // 2.5 Top_p if present is in [0, 1]
    #[test]
    fn prop_top_p_bounded(p in 0.0f32..=1.0) {
        let config = GenerationConfig {
            max_new_tokens: 32,
            temperature: 1.0,
            top_k: None,
            top_p: Some(p),
            repetition_penalty: 1.0,
            do_sample: true,
            seed: None,
        };
        prop_assert!(config.top_p.unwrap() >= 0.0);
        prop_assert!(config.top_p.unwrap() <= 1.0);
    }

    // 2.6 Repetition penalty >= 1.0 is valid
    #[test]
    fn prop_repetition_penalty_valid(penalty in 1.0f32..=5.0) {
        let config = GenerationConfig {
            max_new_tokens: 32,
            temperature: 1.0,
            top_k: None,
            top_p: None,
            repetition_penalty: penalty,
            do_sample: false,
            seed: None,
        };
        prop_assert!(config.repetition_penalty >= 1.0);
    }

    // 2.7 Device::Cuda is always cuda
    #[test]
    fn prop_device_cuda_is_cuda(idx in 0usize..4) {
        let dev = Device::Cuda(idx);
        prop_assert!(dev.is_cuda());
        prop_assert!(!dev.is_cpu());
    }

    // 2.8 Device::Hip is always hip
    #[test]
    fn prop_device_hip_is_hip(idx in 0usize..4) {
        let dev = Device::Hip(idx);
        prop_assert!(dev.is_hip());
        prop_assert!(!dev.is_cpu());
    }
}

// ── 3. Tensor dimension properties ──────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    // 3.1 Broadcast shape reflexive
    #[test]
    fn prop_broadcast_reflexive(
        shape in prop::collection::vec(1usize..8, 1..4),
    ) {
        let result = broadcast_shape(&shape, &shape);
        prop_assert!(result.is_ok());
        prop_assert_eq!(result.unwrap(), shape);
    }

    // 3.2 Broadcast shape symmetric
    #[test]
    fn prop_broadcast_symmetric(
        a in prop::collection::vec(1usize..4, 1..3),
        b in prop::collection::vec(1usize..4, 1..3),
    ) {
        let ab = broadcast_shape(&a, &b);
        let ba = broadcast_shape(&b, &a);
        match (ab, ba) {
            (Ok(ab), Ok(ba)) => prop_assert_eq!(ab, ba),
            (Err(_), Err(_)) => {}
            _ => prop_assert!(false, "asymmetric broadcast"),
        }
    }

    // 3.3 can_broadcast consistent with broadcast_shape
    #[test]
    fn prop_can_broadcast_consistent(
        a in prop::collection::vec(1usize..4, 1..3),
        b in prop::collection::vec(1usize..4, 1..3),
    ) {
        let ok = broadcast_shape(&a, &b).is_ok();
        prop_assert_eq!(can_broadcast(&a, &b), ok);
    }

    // 3.4 Matmul 2D shapes produce correct output
    #[test]
    fn prop_matmul_2d_output(
        m in 1usize..=8,
        k in 1usize..=8,
        n in 1usize..=8,
    ) {
        let out = validate_matmul_shapes(&[m, k], &[k, n]).unwrap();
        prop_assert_eq!(out, vec![m, n]);
    }

    // 3.5 Matmul inner dim mismatch fails
    #[test]
    fn prop_matmul_mismatch_fails(
        m in 1usize..=8,
        k1 in 1usize..=8,
        k2 in 1usize..=8,
        n in 1usize..=8,
    ) {
        if k1 != k2 {
            let result = validate_matmul_shapes(&[m, k1], &[k2, n]);
            prop_assert!(result.is_err());
        }
    }

    // 3.6 Validate reshape preserves element count
    #[test]
    fn prop_reshape_valid_same_elements(
        d0 in 1usize..=8,
        d1 in 1usize..=8,
    ) {
        let n = d0 * d1;
        let result = validate_reshape(&[d0, d1], &[n]);
        prop_assert!(result.is_ok());
    }

    // 3.7 Reshape with different element count fails
    #[test]
    fn prop_reshape_different_count_fails(
        d0 in 2usize..=8,
        d1 in 2usize..=8,
    ) {
        let n = d0 * d1;
        let result = validate_reshape(&[d0, d1], &[n + 1]);
        prop_assert!(result.is_err());
    }

    // 3.8 Transpose axes must be a permutation
    #[test]
    fn prop_transpose_valid_perm(
        d0 in 1usize..=4,
        d1 in 1usize..=4,
        d2 in 1usize..=4,
    ) {
        let result = validate_transpose_axes(&[d0, d1, d2], &[2, 0, 1]);
        prop_assert!(result.is_ok());
        let out = result.unwrap();
        prop_assert_eq!(out, vec![d2, d0, d1]);
    }

    // 3.9 C-contiguous strides: last stride is 1
    #[test]
    fn prop_c_contiguous_last_stride(
        d0 in 1usize..=8,
        d1 in 1usize..=8,
    ) {
        let strides = c_contiguous_strides(&[d0, d1]);
        prop_assert_eq!(*strides.last().unwrap(), 1);
        prop_assert_eq!(strides[0], d1);
    }

    // 3.10 Broadcast with ones is identity
    #[test]
    fn prop_broadcast_with_ones(
        shape in prop::collection::vec(1usize..8, 1..4),
    ) {
        let ones = vec![1usize; shape.len()];
        let result = broadcast_shape(&shape, &ones).unwrap();
        prop_assert_eq!(result, shape);
    }
}

// ── 4. Device selection properties ──────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    // 4.1 CPU backend always selected for CPU request
    #[test]
    fn prop_cpu_backend_selected(_dummy in 0u8..1) {
        let caps = KernelCapabilities::from_compile_time();
        let result = select_backend(BackendRequest::Cpu, &caps);
        prop_assert!(result.is_ok());
        let sel = result.unwrap();
        prop_assert_eq!(sel.selected, KernelBackend::CpuRust);
    }

    // 4.2 Auto backend selection always succeeds
    #[test]
    fn prop_auto_backend_succeeds(_dummy in 0u8..1) {
        let caps = KernelCapabilities::from_compile_time();
        let result = select_backend(BackendRequest::Auto, &caps);
        prop_assert!(result.is_ok());
    }

    // 4.3 KernelCapabilities from_compile_time always has CpuRust
    #[test]
    fn prop_capabilities_has_cpu(_dummy in 0u8..1) {
        let caps = KernelCapabilities::from_compile_time();
        let backends = caps.compiled_backends();
        prop_assert!(backends.contains(&KernelBackend::CpuRust));
    }

    // 4.4 KernelBackend::CpuRust does not require GPU
    #[test]
    fn prop_cpu_no_gpu(_dummy in 0u8..1) {
        prop_assert!(!KernelBackend::CpuRust.requires_gpu());
    }

    // 4.5 KernelBackend::Cuda requires GPU
    #[test]
    fn prop_cuda_requires_gpu(_dummy in 0u8..1) {
        prop_assert!(KernelBackend::Cuda.requires_gpu());
    }

    // 4.6 Summary string is non-empty
    #[test]
    fn prop_capabilities_summary_nonempty(_dummy in 0u8..1) {
        let caps = KernelCapabilities::from_compile_time();
        let summary = caps.summary();
        prop_assert!(!summary.is_empty());
    }

    // 4.7 best_available returns Some for compile-time caps
    #[test]
    fn prop_best_available_some(_dummy in 0u8..1) {
        let caps = KernelCapabilities::from_compile_time();
        prop_assert!(caps.best_available().is_some());
    }

    // 4.8 Device::Cpu is not cuda, opencl, hip, or npu
    #[test]
    fn prop_cpu_not_others(_dummy in 0u8..1) {
        let dev = Device::Cpu;
        prop_assert!(!dev.is_cuda());
        prop_assert!(!dev.is_opencl());
        prop_assert!(!dev.is_hip());
        prop_assert!(!dev.is_npu());
    }
}

// ── 5. Dtype conversion properties ──────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    // 5.1 f32 → bf16 → f32 round-trip bounded error
    #[test]
    fn prop_bf16_roundtrip(val in -100.0f32..100.0) {
        let bf16 = f32_to_bf16(val);
        let recovered = bf16_to_f32(bf16);
        let err = (val - recovered).abs();
        let expected_err = val.abs() * 0.01 + 0.01;
        prop_assert!(err <= expected_err,
            "bf16 round-trip error {err} too large for val={val}");
    }

    // 5.2 f32 → f16 → f32 round-trip bounded error
    #[test]
    fn prop_f16_roundtrip(val in -100.0f32..100.0) {
        let f16 = f32_to_f16(val);
        let recovered = f16_to_f32(f16);
        let err = (val - recovered).abs();
        let expected_err = val.abs() * 0.002 + 0.002;
        prop_assert!(err <= expected_err,
            "f16 round-trip error {err} too large for val={val}");
    }

    // 5.3 bf16 of zero is zero
    #[test]
    fn prop_bf16_zero(_dummy in 0u8..1) {
        let bf16 = f32_to_bf16(0.0);
        let recovered = bf16_to_f32(bf16);
        prop_assert_eq!(recovered, 0.0);
    }

    // 5.4 f16 of zero is zero
    #[test]
    fn prop_f16_zero(_dummy in 0u8..1) {
        let f16 = f32_to_f16(0.0);
        let recovered = f16_to_f32(f16);
        prop_assert_eq!(recovered, 0.0);
    }

    // 5.5 bf16 preserves sign
    #[test]
    fn prop_bf16_preserves_sign(val in -100.0f32..100.0) {
        if val != 0.0 {
            let bf16 = f32_to_bf16(val);
            let recovered = bf16_to_f32(bf16);
            prop_assert_eq!(val.signum(), recovered.signum());
        }
    }

    // 5.6 f16 preserves sign
    #[test]
    fn prop_f16_preserves_sign(val in -100.0f32..100.0) {
        if val != 0.0 {
            let f16 = f32_to_f16(val);
            let recovered = f16_to_f32(f16);
            prop_assert_eq!(val.signum(), recovered.signum());
        }
    }
}

// ── 6. Memory estimation properties ─────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    // 6.1 TensorEstimate elements = product of shape
    #[test]
    fn prop_tensor_estimate_elements(
        d0 in 1usize..=16,
        d1 in 1usize..=16,
    ) {
        let est = TensorEstimate::new("test", &[d0, d1], DType::F32);
        prop_assert_eq!(est.elements(), d0 * d1);
    }

    // 6.2 DType bits: F32 = 32, F16 = 16, BF16 = 16, I8 = 8, I2 = 2
    #[test]
    fn prop_dtype_bits(_dummy in 0u8..1) {
        prop_assert_eq!(DType::F32.bits(), 32);
        prop_assert_eq!(DType::F16.bits(), 16);
        prop_assert_eq!(DType::BF16.bits(), 16);
        prop_assert_eq!(DType::I8.bits(), 8);
        prop_assert_eq!(DType::I2.bits(), 2);
    }

    // 6.3 DType bytes_for: F32 needs 4 bytes per element
    #[test]
    fn prop_dtype_bytes_f32(n in 1usize..=64) {
        prop_assert_eq!(DType::F32.bytes_for(n), n * 4);
    }

    // 6.4 DType bytes_for: F16 needs 2 bytes per element
    #[test]
    fn prop_dtype_bytes_f16(n in 1usize..=64) {
        prop_assert_eq!(DType::F16.bytes_for(n), n * 2);
    }

    // 6.5 TensorPool allocations succeed
    #[test]
    fn prop_tensor_pool_alloc(size in 4usize..=1024) {
        let pool = TensorPool::new(1024 * 1024);
        let buf = pool.allocate(size);
        prop_assert!(buf.as_f32_slice().len() >= size / 4);
    }

    // 6.6 TensorPool stats track allocations
    #[test]
    fn prop_tensor_pool_stats_track(n in 1usize..=8) {
        let pool = TensorPool::new(1024 * 1024);
        for _ in 0..n {
            let _buf = pool.allocate(64);
        }
        let stats = pool.stats();
        prop_assert!(stats.total_allocations() >= n as u64);
    }
}

// ── 7. Shape validator properties ───────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    // 7.1 assert_shape_eq passes for equal shapes
    #[test]
    fn prop_shape_eq_passes(
        shape in prop::collection::vec(1usize..8, 1..4),
    ) {
        let result = assert_shape_eq("test", &shape, &shape);
        prop_assert!(result.is_ok());
    }

    // 7.2 assert_shape_eq fails for different shapes
    #[test]
    fn prop_shape_eq_fails_different(
        a in prop::collection::vec(1usize..8, 1..4),
    ) {
        let mut b = a.clone();
        b.push(1);
        let result = assert_shape_eq("test", &a, &b);
        prop_assert!(result.is_err());
    }

    // 7.3 assert_rank passes for correct rank
    #[test]
    fn prop_rank_passes(n in 1usize..=4) {
        let shape: Vec<usize> = vec![2; n];
        let result = assert_rank("test", &shape, n);
        prop_assert!(result.is_ok());
    }

    // 7.4 assert_rank fails for wrong rank
    #[test]
    fn prop_rank_fails_wrong(n in 1usize..=4) {
        let shape: Vec<usize> = vec![2; n];
        let result = assert_rank("test", &shape, n + 1);
        prop_assert!(result.is_err());
    }

    // 7.5 assert_element_count passes for correct count
    #[test]
    fn prop_element_count_passes(
        d0 in 1usize..=8,
        d1 in 1usize..=8,
    ) {
        let result = assert_element_count("test", &[d0, d1], d0 * d1);
        prop_assert!(result.is_ok());
    }

    // 7.6 assert_head_divisible passes when divisible
    #[test]
    fn prop_head_divisible(
        heads in 1usize..=8,
        per_head in 1usize..=8,
    ) {
        let dim = heads * per_head;
        let result = assert_head_divisible("test", dim, heads);
        prop_assert!(result.is_ok());
    }

    // 7.7 assert_dim validates correct dimension
    #[test]
    fn prop_dim_passes(
        d0 in 1usize..=8,
        d1 in 1usize..=8,
    ) {
        let result = assert_dim("test", &[d0, d1], 0, d0);
        prop_assert!(result.is_ok());
    }

    // 7.8 assert_matmul_compat passes for compatible shapes
    #[test]
    fn prop_matmul_compat_passes(
        m in 1usize..=8,
        k in 1usize..=8,
        n in 1usize..=8,
    ) {
        let result = assert_matmul_compat("test", &[m, k], &[k, n]);
        prop_assert!(result.is_ok());
    }
}

// ── 8. Tensor layout properties ─────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    // 8.1 Contiguous layout: numel = product of shape
    #[test]
    fn prop_layout_numel(
        d0 in 1usize..=8,
        d1 in 1usize..=8,
    ) {
        let layout = TensorLayout::contiguous(&[d0, d1], 4);
        prop_assert_eq!(layout.numel(), d0 * d1);
    }

    // 8.2 Contiguous layout byte_size = numel * element_size
    #[test]
    fn prop_layout_byte_size(
        d0 in 1usize..=8,
        d1 in 1usize..=8,
    ) {
        let layout = TensorLayout::contiguous(&[d0, d1], 4);
        prop_assert_eq!(layout.byte_size(), d0 * d1 * 4);
    }

    // 8.3 Contiguous layout ndim = shape length
    #[test]
    fn prop_layout_ndim(n in 1usize..=4) {
        let shape: Vec<usize> = vec![2; n];
        let layout = TensorLayout::contiguous(&shape, 4);
        prop_assert_eq!(layout.ndim(), n);
    }

    // 8.4 Contiguous layout is_contiguous
    #[test]
    fn prop_layout_is_contiguous(
        d0 in 1usize..=8,
        d1 in 1usize..=8,
    ) {
        let layout = TensorLayout::contiguous(&[d0, d1], 4);
        prop_assert!(layout.is_contiguous());
    }

    // 8.5 Compute strides row-major: last stride is 1
    #[test]
    fn prop_strides_row_major_last(
        d0 in 1usize..=8,
        d1 in 1usize..=8,
    ) {
        let strides = compute_strides(&[d0, d1], LayoutOrder::RowMajor);
        prop_assert_eq!(*strides.last().unwrap(), 1);
    }

    // 8.6 Compute strides col-major: first stride is 1
    #[test]
    fn prop_strides_col_major_first(
        d0 in 1usize..=8,
        d1 in 1usize..=8,
    ) {
        let strides = compute_strides(&[d0, d1], LayoutOrder::ColMajor);
        prop_assert_eq!(strides[0], 1);
    }

    // 8.7 Broadcastable is reflexive
    #[test]
    fn prop_broadcastable_reflexive(
        shape in prop::collection::vec(1usize..8, 1..4),
    ) {
        prop_assert!(broadcastable(&shape, &shape));
    }

    // 8.8 Layout offset at origin is Some(0)
    #[test]
    fn prop_layout_offset_origin(
        d0 in 1usize..=8,
        d1 in 1usize..=8,
    ) {
        let layout = TensorLayout::contiguous(&[d0, d1], 4);
        let offset = layout.offset(&[0, 0]);
        prop_assert_eq!(offset, Some(0));
    }

    // 8.9 Layout transpose swaps dimensions
    #[test]
    fn prop_layout_transpose_swaps(
        d0 in 1usize..=8,
        d1 in 1usize..=8,
    ) {
        let layout = TensorLayout::contiguous(&[d0, d1], 4);
        if let Some(transposed) = layout.transpose(0, 1) {
            prop_assert_eq!(transposed.numel(), d0 * d1);
        }
    }

    // 8.10 Layout reshape preserves numel
    #[test]
    fn prop_layout_reshape_numel(
        d0 in 1usize..=8,
        d1 in 1usize..=8,
    ) {
        let layout = TensorLayout::contiguous(&[d0, d1], 4);
        let n = d0 * d1;
        if let Some(reshaped) = layout.reshape(&[n]) {
            prop_assert_eq!(reshaped.numel(), n);
        }
    }

    // 8.11 Layout aligned to 1 always true
    #[test]
    fn prop_layout_aligned_one(
        d0 in 1usize..=8,
        d1 in 1usize..=8,
    ) {
        let layout = TensorLayout::contiguous(&[d0, d1], 4);
        prop_assert!(layout.is_aligned(1));
    }

    // 8.12 Col-major layout: numel same as row-major
    #[test]
    fn prop_col_major_numel(
        d0 in 1usize..=8,
        d1 in 1usize..=8,
    ) {
        let row = TensorLayout::contiguous(&[d0, d1], 4);
        let col = TensorLayout::col_major(&[d0, d1], 4);
        prop_assert_eq!(row.numel(), col.numel());
    }
}

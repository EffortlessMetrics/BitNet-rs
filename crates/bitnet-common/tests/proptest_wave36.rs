//! Property-based tests — wave 36.
//!
//! Covers bitnet-common: tensor shape invariants, device type round-trips,
//! shape validator properties, kernel registry consistency, and memory
//! estimator correctness.

use bitnet_common::kernel_registry::{KernelBackend, KernelCapabilities, SimdLevel};
use bitnet_common::memory_estimator::{DType, ModelMemoryEstimate, TensorEstimate};
use bitnet_common::shape_validator::{
    assert_broadcastable, assert_dim, assert_element_count, assert_head_divisible,
    assert_matmul_compat, assert_rank, assert_shape_eq,
};
use bitnet_common::types::{Device, QuantizationType};
use proptest::prelude::*;

// ── Strategies ──────────────────────────────────────────────────────────────

fn arb_device() -> impl Strategy<Value = Device> {
    prop_oneof![
        Just(Device::Cpu),
        (0usize..8).prop_map(Device::Cuda),
        (0usize..4).prop_map(Device::Hip),
        Just(Device::Npu),
        Just(Device::Metal),
        (0usize..4).prop_map(Device::OpenCL),
    ]
}

fn arb_quant_type() -> impl Strategy<Value = QuantizationType> {
    prop_oneof![
        Just(QuantizationType::I2S),
        Just(QuantizationType::TL1),
        Just(QuantizationType::TL2),
    ]
}

fn arb_shape(max_rank: usize, max_dim: usize) -> impl Strategy<Value = Vec<usize>> {
    proptest::collection::vec(1usize..=max_dim, 1..=max_rank)
}

fn arb_dtype() -> impl Strategy<Value = DType> {
    prop_oneof![
        Just(DType::F32),
        Just(DType::F16),
        Just(DType::BF16),
        Just(DType::I8),
        Just(DType::I4),
        Just(DType::I2),
        Just(DType::Bool),
    ]
}

fn arb_simd_level() -> impl Strategy<Value = SimdLevel> {
    prop_oneof![
        Just(SimdLevel::Scalar),
        Just(SimdLevel::Neon),
        Just(SimdLevel::Sse42),
        Just(SimdLevel::Avx2),
        Just(SimdLevel::Avx512),
    ]
}

// ── Property tests ──────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    // ════════════════════════════════════════════════════════════════
    // 1. Device type properties
    // ════════════════════════════════════════════════════════════════

    /// Device serde round-trip preserves identity.
    #[test]
    fn prop_device_serde_roundtrip(dev in arb_device()) {
        let json = serde_json::to_string(&dev).unwrap();
        let recovered: Device = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(dev, recovered);
    }

    /// Device Debug representation is non-empty.
    #[test]
    fn prop_device_debug_nonempty(dev in arb_device()) {
        let dbg = format!("{:?}", dev);
        prop_assert!(!dbg.is_empty());
    }

    /// Device Ord is total — transitivity holds.
    #[test]
    fn prop_device_ord_transitivity(
        a in arb_device(),
        b in arb_device(),
        c in arb_device()
    ) {
        if a <= b && b <= c {
            prop_assert!(a <= c);
        }
    }

    /// Device equality is consistent with Hash.
    #[test]
    fn prop_device_eq_hash_consistency(a in arb_device(), b in arb_device()) {
        if a == b {
            use std::hash::{Hash, Hasher};
            let mut ha = std::collections::hash_map::DefaultHasher::new();
            let mut hb = std::collections::hash_map::DefaultHasher::new();
            a.hash(&mut ha);
            b.hash(&mut hb);
            prop_assert_eq!(ha.finish(), hb.finish());
        }
    }

    /// Device is_cpu/is_cuda/is_opencl predicates are mutually consistent.
    #[test]
    fn prop_device_predicate_consistency(dev in arb_device()) {
        let cpu = dev.is_cpu();
        let cuda = dev.is_cuda();
        let opencl = dev.is_opencl();
        // At most one category (Npu/Metal/Hip don't have predicates but
        // none of cpu/cuda/opencl should be true for them).
        if cpu {
            prop_assert!(!cuda && !opencl);
        }
        if cuda {
            prop_assert!(!cpu && !opencl);
        }
        if opencl {
            prop_assert!(!cpu && !cuda);
        }
    }

    // ════════════════════════════════════════════════════════════════
    // 2. QuantizationType properties
    // ════════════════════════════════════════════════════════════════

    /// QuantizationType Display output is non-empty.
    #[test]
    fn prop_quant_display_nonempty(qt in arb_quant_type()) {
        let s = qt.to_string();
        prop_assert!(!s.is_empty());
    }

    /// QuantizationType serde round-trip preserves identity.
    #[test]
    fn prop_quant_serde_roundtrip(qt in arb_quant_type()) {
        let json = serde_json::to_string(&qt).unwrap();
        let recovered: QuantizationType = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(qt, recovered);
    }

    // ════════════════════════════════════════════════════════════════
    // 3. Shape validator properties
    // ════════════════════════════════════════════════════════════════

    /// assert_shape_eq succeeds when shapes are identical.
    #[test]
    fn prop_shape_eq_reflexive(shape in arb_shape(4, 64)) {
        prop_assert!(assert_shape_eq("test", &shape, &shape).is_ok());
    }

    /// assert_rank succeeds when rank matches.
    #[test]
    fn prop_rank_correct(shape in arb_shape(4, 64)) {
        prop_assert!(assert_rank("test", &shape, shape.len()).is_ok());
    }

    /// assert_rank fails when rank does not match.
    #[test]
    fn prop_rank_wrong_fails(shape in arb_shape(4, 64), extra in 1usize..4) {
        let wrong_rank = shape.len() + extra;
        prop_assert!(assert_rank("test", &shape, wrong_rank).is_err());
    }

    /// assert_dim succeeds for valid dimension index and matching size.
    #[test]
    fn prop_dim_correct(shape in arb_shape(4, 64)) {
        for (i, &dim_size) in shape.iter().enumerate() {
            prop_assert!(assert_dim("test", &shape, i, dim_size).is_ok());
        }
    }

    /// Product of shape dimensions equals element count.
    #[test]
    fn prop_element_count_product(shape in arb_shape(4, 32)) {
        let expected: usize = shape.iter().product();
        prop_assert!(assert_element_count("test", &shape, expected).is_ok());
    }

    /// assert_element_count fails for wrong count.
    #[test]
    fn prop_element_count_wrong_fails(shape in arb_shape(4, 32)) {
        let correct: usize = shape.iter().product();
        if correct > 0 {
            prop_assert!(assert_element_count("test", &shape, correct + 1).is_err());
        }
    }

    /// assert_matmul_compat succeeds when inner dims match.
    #[test]
    fn prop_matmul_compat_valid(
        m in 1usize..32,
        k in 1usize..32,
        n in 1usize..32
    ) {
        let a_shape = vec![m, k];
        let b_shape = vec![k, n];
        prop_assert!(assert_matmul_compat("test", &a_shape, &b_shape).is_ok());
    }

    /// assert_matmul_compat fails when inner dims mismatch.
    #[test]
    fn prop_matmul_incompat_fails(
        m in 1usize..32,
        k1 in 1usize..32,
        k2 in 1usize..32,
        n in 1usize..32
    ) {
        prop_assume!(k1 != k2);
        let a_shape = vec![m, k1];
        let b_shape = vec![k2, n];
        prop_assert!(assert_matmul_compat("test", &a_shape, &b_shape).is_err());
    }

    /// A shape is always broadcastable with itself.
    #[test]
    fn prop_broadcast_self(shape in arb_shape(4, 32)) {
        prop_assert!(assert_broadcastable("test", &shape, &shape).is_ok());
    }

    /// A shape is broadcastable with a scalar [1].
    #[test]
    fn prop_broadcast_scalar(shape in arb_shape(4, 32)) {
        prop_assert!(assert_broadcastable("test", &shape, &[1]).is_ok());
    }

    /// assert_head_divisible succeeds when hidden_size % num_heads == 0.
    #[test]
    fn prop_head_divisible_valid(heads in 1usize..32, factor in 1usize..64) {
        let hidden = heads * factor;
        prop_assert!(assert_head_divisible("test", hidden, heads).is_ok());
    }

    /// assert_head_divisible fails for non-divisible.
    #[test]
    fn prop_head_divisible_invalid(heads in 2usize..32, hidden in 1usize..512) {
        prop_assume!(hidden % heads != 0);
        prop_assert!(assert_head_divisible("test", hidden, heads).is_err());
    }

    // ════════════════════════════════════════════════════════════════
    // 4. Memory estimator properties
    // ════════════════════════════════════════════════════════════════

    /// TensorEstimate elements == product of shape.
    #[test]
    fn prop_tensor_estimate_elements(
        shape in arb_shape(3, 64),
        dtype in arb_dtype()
    ) {
        let est = TensorEstimate::new("test", &shape, dtype);
        let expected: usize = shape.iter().product();
        prop_assert_eq!(est.elements(), expected);
    }

    /// TensorEstimate bytes >= elements * bits / 8 (rounded up).
    #[test]
    fn prop_tensor_estimate_bytes_bound(
        shape in arb_shape(3, 32),
        dtype in arb_dtype()
    ) {
        let est = TensorEstimate::new("test", &shape, dtype);
        let expected_bytes = dtype.bytes_for(est.elements());
        prop_assert_eq!(est.bytes, expected_bytes);
    }

    /// DType bytes_for is monotonically increasing with element count.
    #[test]
    fn prop_dtype_bytes_monotone(dtype in arb_dtype(), n in 0usize..10000) {
        let a = dtype.bytes_for(n);
        let b = dtype.bytes_for(n + 1);
        prop_assert!(b >= a, "bytes_for({}) = {} but bytes_for({}) = {}", n, a, n + 1, b);
    }

    // ════════════════════════════════════════════════════════════════
    // 5. Kernel registry properties
    // ════════════════════════════════════════════════════════════════

    /// SimdLevel Ord is total.
    #[test]
    fn prop_simd_level_ord_total(a in arb_simd_level(), b in arb_simd_level()) {
        // Either a <= b or b <= a (totality).
        prop_assert!(a <= b || b <= a);
    }

    /// SimdLevel Display is non-empty.
    #[test]
    fn prop_simd_display_nonempty(level in arb_simd_level()) {
        prop_assert!(!level.to_string().is_empty());
    }

    /// KernelBackend Display is non-empty.
    #[test]
    fn prop_kernel_backend_display_nonempty(
        backend in prop_oneof![
            Just(KernelBackend::CpuRust),
            Just(KernelBackend::Cuda),
            Just(KernelBackend::Hip),
            Just(KernelBackend::OneApi),
            Just(KernelBackend::OpenCL),
            Just(KernelBackend::CppFfi),
        ]
    ) {
        prop_assert!(!backend.to_string().is_empty());
    }

    /// KernelCapabilities from_compile_time always includes cpu_rust when
    /// compiled with cpu feature.
    #[test]
    fn prop_capabilities_cpu_always_present(_dummy in 0u8..1) {
        let caps = KernelCapabilities::from_compile_time();
        // We're running with --features cpu, so cpu_rust must be true.
        prop_assert!(caps.cpu_rust);
    }

    /// KernelCapabilities best_available never returns None when cpu_rust is true.
    #[test]
    fn prop_capabilities_best_available_with_cpu(simd in arb_simd_level()) {
        let caps = KernelCapabilities {
            cpu_rust: true,
            cuda_compiled: false,
            cuda_runtime: false,
            hip_compiled: false,
            hip_runtime: false,
            oneapi_compiled: false,
            oneapi_runtime: false,
            opencl_compiled: false,
            opencl_runtime: false,
            cpp_ffi: false,
            simd_level: simd,
        };
        prop_assert!(caps.best_available().is_some());
        prop_assert_eq!(caps.best_available(), Some(KernelBackend::CpuRust));
    }

    /// compiled_backends always contains CpuRust when cpu_rust is true.
    #[test]
    fn prop_compiled_backends_include_cpu(
        cuda in any::<bool>(),
        hip in any::<bool>(),
        oneapi in any::<bool>(),
        opencl in any::<bool>(),
        ffi in any::<bool>()
    ) {
        let caps = KernelCapabilities {
            cpu_rust: true,
            cuda_compiled: cuda,
            cuda_runtime: false,
            hip_compiled: hip,
            hip_runtime: false,
            oneapi_compiled: oneapi,
            oneapi_runtime: false,
            opencl_compiled: opencl,
            opencl_runtime: false,
            cpp_ffi: ffi,
            simd_level: SimdLevel::Scalar,
        };
        let backends = caps.compiled_backends();
        prop_assert!(backends.contains(&KernelBackend::CpuRust));
    }

    /// summary() is non-empty for any valid capabilities.
    #[test]
    fn prop_capabilities_summary_nonempty(simd in arb_simd_level()) {
        let caps = KernelCapabilities {
            cpu_rust: true,
            cuda_compiled: false,
            cuda_runtime: false,
            hip_compiled: false,
            hip_runtime: false,
            oneapi_compiled: false,
            oneapi_runtime: false,
            opencl_compiled: false,
            opencl_runtime: false,
            cpp_ffi: false,
            simd_level: simd,
        };
        prop_assert!(!caps.summary().is_empty());
    }
}

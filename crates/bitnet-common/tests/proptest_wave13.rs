//! Wave 13 property tests: common type safety, shape arithmetic, tensor
//! validation, backend selection, and kernel registry invariants.
//!
//! Key invariants tested (12 properties):
//! - Broadcast: reflexive (a⊕a = a), commutative, can_broadcast consistent
//! - Matmul shapes: compatible shapes produce correct output dimensions
//! - Reshape: matching element counts always succeed
//! - Transpose: identity permutation is no-op, output shape permuted correctly
//! - C-contiguous strides: last stride is 1, product matches element count
//! - KernelCapabilities: from_compile_time is idempotent, cpu_rust always present
//! - ArchitectureRegistry: known architectures are non-empty strings

use bitnet_common::arch_registry::ArchitectureRegistry;
use bitnet_common::kernel_registry::{KernelBackend, KernelCapabilities};
use bitnet_common::tensor_validation::{
    broadcast_shape, c_contiguous_strides, can_broadcast, validate_matmul_shapes, validate_reshape,
    validate_transpose_axes,
};
use bitnet_common::types::Device;
use proptest::prelude::*;

// -------------------------------------------------------------------
// Strategy helpers
// -------------------------------------------------------------------

/// Small shape vector (1-4 dims, each dim 1..=8).
fn small_shape(max_dims: usize) -> impl Strategy<Value = Vec<usize>> {
    prop::collection::vec(1usize..=8, 1..=max_dims)
}

// ===================================================================
// 1. Broadcast shape arithmetic
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Broadcasting a shape with itself always succeeds and returns the same shape.
    #[test]
    fn prop_broadcast_reflexive(shape in small_shape(4)) {
        let result = broadcast_shape(&shape, &shape).unwrap();
        prop_assert_eq!(&result, &shape, "broadcast(a,a) should equal a");
    }

    /// Broadcasting is commutative: broadcast(a,b) == broadcast(b,a) when both succeed.
    #[test]
    fn prop_broadcast_commutative(
        a in small_shape(3),
        b in small_shape(3),
    ) {
        let ab = broadcast_shape(&a, &b);
        let ba = broadcast_shape(&b, &a);
        match (ab, ba) {
            (Ok(r1), Ok(r2)) => prop_assert_eq!(r1, r2, "broadcast must be commutative"),
            (Err(_), Err(_)) => {} // both fail — fine
            (Ok(_), Err(_)) | (Err(_), Ok(_)) => {
                prop_assert!(false, "broadcast symmetry broken: one succeeded, other failed");
            }
        }
    }

    /// can_broadcast is consistent with broadcast_shape.
    #[test]
    fn prop_can_broadcast_consistent(
        a in small_shape(3),
        b in small_shape(3),
    ) {
        let result = broadcast_shape(&a, &b);
        let can = can_broadcast(&a, &b);
        prop_assert_eq!(result.is_ok(), can, "can_broadcast inconsistent with broadcast_shape");
    }

    /// Broadcasting with a scalar (empty shape) always succeeds.
    #[test]
    fn prop_broadcast_with_scalar(shape in small_shape(4)) {
        let result = broadcast_shape(&shape, &[]).unwrap();
        prop_assert_eq!(&result, &shape, "broadcast(a, []) should equal a");
    }
}

// ===================================================================
// 2. Matmul shape validation
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// 2D matmul with compatible inner dimensions produces correct output shape.
    #[test]
    fn prop_matmul_2d_compatible(
        m in 1usize..=16,
        k in 1usize..=16,
        n in 1usize..=16,
    ) {
        let result = validate_matmul_shapes(&[m, k], &[k, n]).unwrap();
        prop_assert_eq!(result, vec![m, n], "expected [m,n] output shape");
    }

    /// 2D matmul with incompatible inner dimensions fails.
    #[test]
    fn prop_matmul_2d_incompatible(
        m in 1usize..=16,
        k1 in 1usize..=16,
        k2 in 1usize..=16,
        n in 1usize..=16,
    ) {
        prop_assume!(k1 != k2);
        let result = validate_matmul_shapes(&[m, k1], &[k2, n]);
        prop_assert!(result.is_err(), "mismatched inner dims should fail");
    }

    /// 1D dot product of same-length vectors succeeds with scalar output.
    #[test]
    fn prop_matmul_1d_dot_product(k in 1usize..=32) {
        let result = validate_matmul_shapes(&[k], &[k]).unwrap();
        prop_assert!(result.is_empty(), "1D dot product should produce scalar (empty shape)");
    }
}

// ===================================================================
// 3. Reshape validation
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Reshape with matching element counts always succeeds.
    #[test]
    fn prop_reshape_matching_counts(
        a in 1usize..=8,
        b in 1usize..=8,
    ) {
        let total = a * b;
        let result = validate_reshape(&[a, b], &[total]);
        prop_assert!(result.is_ok(), "matching element counts should succeed");
    }

    /// Reshape with mismatched element counts always fails.
    #[test]
    fn prop_reshape_mismatched_counts(
        a in 2usize..=8,
        b in 2usize..=8,
    ) {
        let total = a * b;
        let result = validate_reshape(&[a, b], &[total + 1]);
        prop_assert!(result.is_err(), "mismatched element counts should fail");
    }
}

// ===================================================================
// 4. Transpose validation
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Identity permutation [0,1,...,n-1] returns the original shape.
    #[test]
    fn prop_transpose_identity_permutation(shape in small_shape(4)) {
        let axes: Vec<usize> = (0..shape.len()).collect();
        let result = validate_transpose_axes(&shape, &axes).unwrap();
        prop_assert_eq!(&result, &shape, "identity permutation should return original shape");
    }

    /// Reversing axes on a 2D shape swaps dimensions.
    #[test]
    fn prop_transpose_2d_swap(
        a in 1usize..=16,
        b in 1usize..=16,
    ) {
        let result = validate_transpose_axes(&[a, b], &[1, 0]).unwrap();
        prop_assert_eq!(result, vec![b, a], "transposing [a,b] with [1,0] should give [b,a]");
    }

    /// Transpose with wrong number of axes fails.
    #[test]
    fn prop_transpose_wrong_axes_count(shape in small_shape(3)) {
        let wrong_axes: Vec<usize> = (0..shape.len() + 1).collect();
        let result = validate_transpose_axes(&shape, &wrong_axes);
        prop_assert!(result.is_err(), "wrong axes count should fail");
    }
}

// ===================================================================
// 5. C-contiguous strides
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Last stride is always 1 for non-empty shapes.
    #[test]
    fn prop_c_strides_last_is_one(shape in small_shape(4)) {
        let strides = c_contiguous_strides(&shape);
        prop_assert_eq!(strides.len(), shape.len());
        prop_assert_eq!(*strides.last().unwrap(), 1, "last stride must be 1");
    }

    /// First stride equals the product of all dimensions except the first.
    #[test]
    fn prop_c_strides_first_matches_trailing_product(shape in small_shape(4)) {
        let strides = c_contiguous_strides(&shape);
        let trailing_product: usize = shape[1..].iter().product();
        prop_assert_eq!(
            strides[0], trailing_product,
            "first stride should be product of trailing dims"
        );
    }

    /// stride[i] = stride[i+1] * shape[i+1] for all i.
    #[test]
    fn prop_c_strides_recurrence(
        shape in prop::collection::vec(1usize..=8, 2..=5),
    ) {
        let strides = c_contiguous_strides(&shape);
        for i in 0..shape.len() - 1 {
            prop_assert_eq!(
                strides[i],
                strides[i + 1] * shape[i + 1],
                "stride[{}] should be stride[{}] * shape[{}]", i, i + 1, i + 1
            );
        }
    }
}

// ===================================================================
// 6. KernelCapabilities and KernelBackend
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// from_compile_time is idempotent: calling twice gives same result.
    #[test]
    fn prop_kernel_caps_compile_time_idempotent(_dummy in 0..1i32) {
        let c1 = KernelCapabilities::from_compile_time();
        let c2 = KernelCapabilities::from_compile_time();
        let b1 = c1.compiled_backends();
        let b2 = c2.compiled_backends();
        prop_assert_eq!(b1, b2, "from_compile_time should be deterministic");
    }

    /// CpuRust backend is always in compiled backends.
    #[test]
    fn prop_kernel_caps_always_has_cpu_rust(_dummy in 0..1i32) {
        let caps = KernelCapabilities::from_compile_time();
        let backends = caps.compiled_backends();
        prop_assert!(
            backends.contains(&KernelBackend::CpuRust),
            "CpuRust should always be compiled"
        );
    }

    /// CpuRust does not require GPU.
    #[test]
    fn prop_cpu_rust_does_not_require_gpu(_dummy in 0..1i32) {
        prop_assert!(!KernelBackend::CpuRust.requires_gpu());
    }

    /// KernelBackend::is_compiled returns true for CpuRust.
    #[test]
    fn prop_cpu_rust_is_compiled(_dummy in 0..1i32) {
        prop_assert!(KernelBackend::CpuRust.is_compiled());
    }
}

// ===================================================================
// 7. ArchitectureRegistry
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    /// All known architecture strings are non-empty.
    #[test]
    fn prop_known_architectures_non_empty(_dummy in 0..1i32) {
        let archs = ArchitectureRegistry::known_architectures();
        prop_assert!(!archs.is_empty(), "should have at least one known architecture");
        for arch in archs {
            prop_assert!(!arch.is_empty(), "architecture name should be non-empty");
        }
    }

    /// Looking up a known architecture always returns Some.
    #[test]
    fn prop_known_architecture_lookup_succeeds(_dummy in 0..1i32) {
        let archs = ArchitectureRegistry::known_architectures();
        for arch in archs {
            let result = ArchitectureRegistry::lookup(arch);
            prop_assert!(
                result.is_some(),
                "lookup({arch}) should succeed for known architecture"
            );
        }
    }

    /// is_known is consistent with lookup for known architectures.
    #[test]
    fn prop_is_known_consistent_with_lookup(_dummy in 0..1i32) {
        let archs = ArchitectureRegistry::known_architectures();
        for arch in archs {
            prop_assert_eq!(
                ArchitectureRegistry::is_known(arch),
                ArchitectureRegistry::lookup(arch).is_some()
            );
        }
    }

    /// Device::Cpu is the default device.
    #[test]
    fn prop_default_device_is_cpu(_dummy in 0..1i32) {
        let d = Device::default();
        prop_assert!(d.is_cpu(), "default device should be CPU");
    }
}

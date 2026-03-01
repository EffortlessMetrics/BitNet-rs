//! Property-based tests — wave 16.
//!
//! Tensor shape validation: broadcasting, matmul, reshape, transpose,
//! attention shapes, and KernelCapabilities invariants.

use bitnet_common::{
    kernel_registry::{KernelBackend, KernelCapabilities, SimdLevel},
    tensor_validation::{
        broadcast_shape, can_broadcast, validate_attention_shapes, validate_matmul_shapes,
        validate_reshape, validate_transpose_axes,
    },
    types::{Device, GenerationConfig, PerformanceMetrics, QuantizationType},
};
use proptest::prelude::*;

// ── Broadcasting properties ─────────────────────────────────────────────────

proptest! {
    /// Broadcasting is reflexive: any shape broadcasts with itself.
    #[test]
    fn broadcast_reflexive(
        shape in prop::collection::vec(1usize..8, 1..4),
    ) {
        let result = broadcast_shape(&shape, &shape);
        prop_assert!(result.is_ok(), "shape {:?} should broadcast with itself", shape);
        prop_assert_eq!(result.unwrap(), shape);
    }

    /// Broadcasting is symmetric: broadcast(a, b) == broadcast(b, a).
    #[test]
    fn broadcast_symmetric(
        a in prop::collection::vec(1usize..4, 1..3),
        b in prop::collection::vec(1usize..4, 1..3),
    ) {
        let ab = broadcast_shape(&a, &b);
        let ba = broadcast_shape(&b, &a);
        match (ab, ba) {
            (Ok(ab), Ok(ba)) => prop_assert_eq!(ab, ba),
            (Err(_), Err(_)) => {} // both fail — OK
            (ab, ba) => prop_assert!(false,
                "asymmetric broadcast: a={:?} b={:?} → {:?} vs {:?}", a, b, ab, ba),
        }
    }

    /// can_broadcast agrees with broadcast_shape.
    #[test]
    fn can_broadcast_agrees(
        a in prop::collection::vec(1usize..4, 1..3),
        b in prop::collection::vec(1usize..4, 1..3),
    ) {
        let ok = broadcast_shape(&a, &b).is_ok();
        prop_assert_eq!(can_broadcast(&a, &b), ok);
    }

    /// Broadcasting with [1] never changes the other shape.
    #[test]
    fn broadcast_with_ones(
        shape in prop::collection::vec(1usize..8, 1..4),
    ) {
        let ones = vec![1usize; shape.len()];
        let result = broadcast_shape(&shape, &ones).unwrap();
        prop_assert_eq!(result, shape);
    }

    /// Broadcast output ndim = max(a.ndim, b.ndim).
    #[test]
    fn broadcast_output_ndim(
        a in prop::collection::vec(1usize..4, 1..4),
        b in prop::collection::vec(1usize..4, 1..4),
    ) {
        if let Ok(out) = broadcast_shape(&a, &b) {
            prop_assert_eq!(out.len(), a.len().max(b.len()));
        }
    }
}

// ── Matmul shape validation ─────────────────────────────────────────────────

proptest! {
    /// 2D matmul [M,K] × [K,N] → [M,N].
    #[test]
    fn matmul_2d_output_shape(
        m in 1usize..16,
        k in 1usize..16,
        n in 1usize..16,
    ) {
        let out = validate_matmul_shapes(&[m, k], &[k, n]).unwrap();
        prop_assert_eq!(out, vec![m, n]);
    }

    /// 1D dot product [K] × [K] → [].
    #[test]
    fn matmul_1d_dot_product(k in 1usize..32) {
        let out = validate_matmul_shapes(&[k], &[k]).unwrap();
        prop_assert!(out.is_empty(), "dot product should produce scalar shape");
    }

    /// Mismatched inner dims always fail.
    #[test]
    fn matmul_inner_mismatch_fails(
        m in 1usize..8,
        k1 in 1usize..8,
        k2 in 1usize..8,
        n in 1usize..8,
    ) {
        prop_assume!(k1 != k2);
        prop_assert!(validate_matmul_shapes(&[m, k1], &[k2, n]).is_err());
    }

    /// Empty shapes always fail.
    #[test]
    fn matmul_empty_shapes_fail(_dummy in 0u8..1) {
        prop_assert!(validate_matmul_shapes(&[], &[4]).is_err());
        prop_assert!(validate_matmul_shapes(&[4], &[]).is_err());
        prop_assert!(validate_matmul_shapes(&[], &[]).is_err());
    }
}

// ── Reshape validation ──────────────────────────────────────────────────────

proptest! {
    /// Reshape to same element count always succeeds.
    #[test]
    fn reshape_same_elements_ok(
        d0 in 1usize..16,
        d1 in 1usize..16,
    ) {
        let n = d0 * d1;
        prop_assert!(validate_reshape(&[d0, d1], &[n]).is_ok());
        prop_assert!(validate_reshape(&[n], &[d0, d1]).is_ok());
    }

    /// Reshape to different element count always fails.
    #[test]
    fn reshape_different_elements_fail(
        a in 2usize..64,
        b in 2usize..64,
    ) {
        prop_assume!(a != b);
        prop_assert!(validate_reshape(&[a], &[b]).is_err());
    }

    /// Reshape is reflexive: any shape reshapes to itself.
    #[test]
    fn reshape_reflexive(
        shape in prop::collection::vec(1usize..8, 1..4),
    ) {
        prop_assert!(validate_reshape(&shape, &shape).is_ok());
    }
}

// ── Transpose validation ────────────────────────────────────────────────────

proptest! {
    /// Identity permutation always succeeds and preserves shape.
    #[test]
    fn transpose_identity(
        shape in prop::collection::vec(1usize..8, 1..5),
    ) {
        let axes: Vec<usize> = (0..shape.len()).collect();
        let result = validate_transpose_axes(&shape, &axes).unwrap();
        prop_assert_eq!(result, shape);
    }

    /// Double transpose is identity.
    #[test]
    fn transpose_double_is_identity(
        d0 in 1usize..8,
        d1 in 1usize..8,
    ) {
        let shape = vec![d0, d1];
        let perm = vec![1, 0];
        let first = validate_transpose_axes(&shape, &perm).unwrap();
        let second = validate_transpose_axes(&first, &perm).unwrap();
        prop_assert_eq!(second, shape);
    }

    /// Transpose preserves total element count.
    #[test]
    fn transpose_preserves_elements(
        d0 in 1usize..8,
        d1 in 1usize..8,
        d2 in 1usize..8,
    ) {
        let shape = vec![d0, d1, d2];
        let perm = vec![2, 0, 1];
        let result = validate_transpose_axes(&shape, &perm).unwrap();
        let orig_elems: usize = shape.iter().product();
        let new_elems: usize = result.iter().product();
        prop_assert_eq!(orig_elems, new_elems);
    }

    /// Out-of-range axis always fails.
    #[test]
    fn transpose_out_of_range_fails(
        ndim in 1usize..5,
    ) {
        let shape = vec![2; ndim];
        let mut axes: Vec<usize> = (0..ndim).collect();
        axes[0] = ndim; // out of range
        prop_assert!(validate_transpose_axes(&shape, &axes).is_err());
    }

    /// Wrong number of axes always fails.
    #[test]
    fn transpose_wrong_axes_count_fails(
        ndim in 2usize..5,
    ) {
        let shape = vec![2; ndim];
        let axes: Vec<usize> = (0..ndim - 1).collect();
        prop_assert!(validate_transpose_axes(&shape, &axes).is_err());
    }
}

// ── Attention shape validation ──────────────────────────────────────────────

proptest! {
    /// Valid attention shapes pass validation.
    #[test]
    fn attention_valid_shapes(
        batch in 1usize..4,
        heads in 1usize..8,
        seq_len in 1usize..16,
        kv_len in 1usize..16,
        head_dim in 1usize..16,
        v_dim in 1usize..16,
    ) {
        let q = vec![batch, heads, seq_len, head_dim];
        let k = vec![batch, heads, kv_len, head_dim];
        let v = vec![batch, heads, kv_len, v_dim];
        prop_assert!(validate_attention_shapes(&q, &k, &v).is_ok());
    }

    /// GQA: Q heads must be a multiple of KV heads.
    #[test]
    fn attention_gqa_heads(
        batch in 1usize..4,
        kv_heads in 1usize..4,
        multiplier in 1usize..4,
        seq_len in 1usize..8,
        head_dim in 1usize..8,
    ) {
        let q_heads = kv_heads * multiplier;
        let q = vec![batch, q_heads, seq_len, head_dim];
        let k = vec![batch, kv_heads, seq_len, head_dim];
        let v = vec![batch, kv_heads, seq_len, head_dim];
        prop_assert!(validate_attention_shapes(&q, &k, &v).is_ok());
    }

    /// Non-4D tensors always fail.
    #[test]
    fn attention_non_4d_fails(
        ndim in 1usize..4,
    ) {
        prop_assume!(ndim != 4);
        let shape = vec![2; ndim];
        let valid = vec![2; 4];
        prop_assert!(validate_attention_shapes(&shape, &valid, &valid).is_err());
        prop_assert!(validate_attention_shapes(&valid, &shape, &valid).is_err());
        prop_assert!(validate_attention_shapes(&valid, &valid, &shape).is_err());
    }

    /// Mismatched batch dims always fail.
    #[test]
    fn attention_batch_mismatch_fails(
        b1 in 1usize..4,
        b2 in 1usize..4,
    ) {
        prop_assume!(b1 != b2);
        let q = vec![b1, 2, 4, 8];
        let k = vec![b2, 2, 4, 8];
        let v = vec![b2, 2, 4, 8];
        prop_assert!(validate_attention_shapes(&q, &k, &v).is_err());
    }
}

// ── KernelCapabilities properties ───────────────────────────────────────────

proptest! {
    /// compiled_backends always includes CpuRust when cpu_rust is true.
    #[test]
    fn capabilities_cpu_rust_in_backends(_dummy in 0u8..1) {
        let caps = KernelCapabilities::from_compile_time();
        let backends = caps.compiled_backends();
        // cpu feature is always on when testing with --features cpu
        prop_assert!(backends.contains(&KernelBackend::CpuRust));
    }

    /// best_available never returns a backend that requires GPU without runtime.
    #[test]
    fn capabilities_best_no_gpu_without_runtime(_dummy in 0u8..1) {
        let caps = KernelCapabilities::from_compile_time();
        if let Some(best) = caps.best_available()
            && best.requires_gpu()
        {
            let backends = caps.compiled_backends();
            prop_assert!(backends.contains(&best));
        }
    }

    /// summary is non-empty.
    #[test]
    fn capabilities_summary_non_empty(_dummy in 0u8..1) {
        let caps = KernelCapabilities::from_compile_time();
        prop_assert!(!caps.summary().is_empty());
    }
}

// ── SimdLevel ordering ──────────────────────────────────────────────────────

proptest! {
    /// SimdLevel ordering: Scalar < Sse42 < Avx2 < Avx512.
    #[test]
    fn simd_level_ordering(_dummy in 0u8..1) {
        prop_assert!(SimdLevel::Scalar < SimdLevel::Sse42);
        prop_assert!(SimdLevel::Sse42 < SimdLevel::Avx2);
        prop_assert!(SimdLevel::Avx2 < SimdLevel::Avx512);
    }

    /// SimdLevel ordering is reflexive.
    #[test]
    fn simd_level_reflexive(
        level in prop_oneof![
            Just(SimdLevel::Scalar),
            Just(SimdLevel::Sse42),
            Just(SimdLevel::Avx2),
            Just(SimdLevel::Avx512),
            Just(SimdLevel::Neon),
        ]
    ) {
        prop_assert!(level <= level);
        prop_assert!(level >= level);
    }
}

// ── GenerationConfig defaults ───────────────────────────────────────────────

proptest! {
    /// Default GenerationConfig has sane values.
    #[test]
    fn gen_config_defaults_sane(_dummy in 0u8..1) {
        let cfg = GenerationConfig::default();
        prop_assert!(cfg.max_new_tokens > 0);
        prop_assert!(cfg.temperature > 0.0);
        prop_assert!(cfg.temperature.is_finite());
        prop_assert!(cfg.repetition_penalty >= 1.0);
    }

    /// PerformanceMetrics default is all zeros.
    #[test]
    fn perf_metrics_default_zero(_dummy in 0u8..1) {
        let m = PerformanceMetrics::default();
        prop_assert_eq!(m.tokens_per_second, 0.0);
        prop_assert_eq!(m.latency_ms, 0.0);
        prop_assert_eq!(m.memory_usage_mb, 0.0);
        prop_assert_eq!(m.gpu_utilization, None);
    }
}

// ── Device type properties ──────────────────────────────────────────────────

proptest! {
    /// All device variants have mutually exclusive predicates.
    #[test]
    fn device_predicates_exclusive(idx in 0usize..4) {
        let devices = vec![
            Device::Cpu,
            Device::Cuda(idx),
            Device::OpenCL(idx),
        ];
        for d in &devices {
            let count = [d.is_cpu(), d.is_cuda(), d.is_opencl()].iter()
                .filter(|&&x| x).count();
            prop_assert_eq!(count, 1,
                "device {:?} has {} true predicates, expected exactly 1", d, count);
        }
    }

    /// Device Clone produces equal copy.
    #[test]
    fn device_clone_eq(idx in 0usize..4) {
        let d = Device::Cuda(idx);
        let d2 = d;
        prop_assert_eq!(d, d2);
    }
}

// ── QuantizationType properties ─────────────────────────────────────────────

proptest! {
    /// All QuantizationType variants have distinct Display output.
    #[test]
    fn quantization_type_display_distinct(_dummy in 0u8..1) {
        let variants = [
            QuantizationType::I2S,
            QuantizationType::TL1,
            QuantizationType::TL2,
        ];
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                prop_assert_ne!(
                    variants[i].to_string(),
                    variants[j].to_string(),
                    "{:?} and {:?} have same Display", variants[i], variants[j]
                );
            }
        }
    }
}

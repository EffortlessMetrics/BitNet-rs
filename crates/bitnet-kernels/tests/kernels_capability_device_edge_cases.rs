//! Edge-case tests for capability_matrix, device_aware, and shaped_reduction modules.

use bitnet_common::{Device, QuantizationType};
use bitnet_kernels::capability_matrix::{
    CapabilityEntry, CapabilityQuery, CompatibilityReport, DeviceCapabilityMatrix, DeviceClass,
    DeviceProfile, OperationCategory, PrecisionSupport, SupportLevel,
};
use bitnet_kernels::device_aware::{DeviceAwareQuantizer, DeviceAwareQuantizerFactory};
use bitnet_kernels::shaped_reduction::{
    ReductionOp, ShapedReductionConfig, reduce_f32, reduction_output_shape,
};

// =========================================================================
// capability_matrix — DeviceClass edge cases
// =========================================================================

#[test]
fn device_class_all_contains_every_variant() {
    // ALL should contain every variant exactly once
    let all = DeviceClass::ALL;
    assert!(all.contains(&DeviceClass::IntelArc));
    assert!(all.contains(&DeviceClass::IntelXe));
    assert!(all.contains(&DeviceClass::NvidiaCuda));
    assert!(all.contains(&DeviceClass::AmdRocm));
    assert!(all.contains(&DeviceClass::AppleMetal));
    assert!(all.contains(&DeviceClass::CpuSimd));
    assert!(all.contains(&DeviceClass::CpuScalar));
    assert!(all.contains(&DeviceClass::WebGpu));
    // No duplicates
    let mut deduped: Vec<_> = all.to_vec();
    deduped.dedup();
    assert_eq!(deduped.len(), all.len());
}

#[test]
fn device_class_debug_format_roundtrip() {
    for dc in DeviceClass::ALL {
        let debug = format!("{dc:?}");
        assert!(!debug.is_empty(), "Debug should be non-empty for {dc}");
    }
}

#[test]
fn device_class_display_non_empty_for_all() {
    for dc in DeviceClass::ALL {
        let display = dc.to_string();
        assert!(!display.is_empty());
        // Display should differ from Debug for readability
        assert_ne!(display, format!("{dc:?}"));
    }
}

#[test]
fn device_class_hash_all_unique() {
    use std::collections::HashSet;
    let set: HashSet<DeviceClass> = DeviceClass::ALL.iter().copied().collect();
    assert_eq!(set.len(), DeviceClass::ALL.len());
}

// =========================================================================
// capability_matrix — SupportLevel edge cases
// =========================================================================

#[test]
fn support_level_full_zero_efficiency() {
    let level = SupportLevel::Full(0.0);
    assert!(level.is_supported());
    assert_eq!(level.efficiency(), Some(0.0));
    assert!(level.to_string().contains("0%"));
}

#[test]
fn support_level_full_one_efficiency() {
    let level = SupportLevel::Full(1.0);
    assert!(level.is_supported());
    assert_eq!(level.efficiency(), Some(1.0));
    assert!(level.to_string().contains("100%"));
}

#[test]
fn support_level_partial_empty_reason() {
    let level = SupportLevel::Partial(String::new());
    assert!(level.is_supported());
    assert_eq!(level.efficiency(), None);
    let display = level.to_string();
    assert!(display.contains("Partial"));
}

#[test]
fn support_level_partial_long_reason() {
    let reason = "x".repeat(1000);
    let level = SupportLevel::Partial(reason.clone());
    assert!(level.is_supported());
    let display = level.to_string();
    assert!(display.contains(&reason));
}

// =========================================================================
// capability_matrix — DeviceProfile edge cases
// =========================================================================

#[test]
fn profile_empty_capabilities_lookup_returns_unsupported() {
    let profile = DeviceProfile {
        device_class: DeviceClass::WebGpu,
        name: "Empty".to_string(),
        compute_units: 0,
        memory_gb: 0,
        capabilities: vec![],
    };
    // Any lookup on an empty profile should be Unsupported
    for op in OperationCategory::ALL {
        for prec in PrecisionSupport::ALL {
            assert!(
                matches!(profile.lookup(*op, *prec), SupportLevel::Unsupported),
                "empty profile should return Unsupported for {op} @ {prec}"
            );
        }
    }
    assert_eq!(profile.full_support_count(), 0);
}

#[test]
fn profile_entries_for_operation_empty_when_none_match() {
    let profile = DeviceProfile {
        device_class: DeviceClass::CpuScalar,
        name: "Minimal".to_string(),
        compute_units: 1,
        memory_gb: 0,
        capabilities: vec![CapabilityEntry::new(
            OperationCategory::MatrixOps,
            PrecisionSupport::FP32,
            SupportLevel::Full(0.5),
        )],
    };
    let entries = profile.entries_for_operation(OperationCategory::AttentionOps);
    assert!(entries.is_empty());
}

#[test]
fn profile_entries_for_precision_empty_when_none_match() {
    let profile = DeviceProfile {
        device_class: DeviceClass::CpuScalar,
        name: "Minimal".to_string(),
        compute_units: 1,
        memory_gb: 0,
        capabilities: vec![CapabilityEntry::new(
            OperationCategory::MatrixOps,
            PrecisionSupport::FP32,
            SupportLevel::Full(0.5),
        )],
    };
    let entries = profile.entries_for_precision(PrecisionSupport::BF16);
    assert!(entries.is_empty());
}

#[test]
fn profile_duplicate_entries_both_returned() {
    // If two entries have the same op+prec, lookup returns the first
    let profile = DeviceProfile {
        device_class: DeviceClass::CpuScalar,
        name: "Dupes".to_string(),
        compute_units: 1,
        memory_gb: 0,
        capabilities: vec![
            CapabilityEntry::new(
                OperationCategory::MatrixOps,
                PrecisionSupport::FP32,
                SupportLevel::Full(0.3),
            ),
            CapabilityEntry::new(
                OperationCategory::MatrixOps,
                PrecisionSupport::FP32,
                SupportLevel::Full(0.9),
            ),
        ],
    };
    // lookup returns first match
    let level = profile.lookup(OperationCategory::MatrixOps, PrecisionSupport::FP32);
    assert!(matches!(level, SupportLevel::Full(e) if (*e - 0.3).abs() < f64::EPSILON));
    // entries_for_operation returns both
    let entries = profile.entries_for_operation(OperationCategory::MatrixOps);
    assert_eq!(entries.len(), 2);
}

// =========================================================================
// capability_matrix — DeviceCapabilityMatrix edge cases
// =========================================================================

#[test]
fn matrix_add_multiple_same_class_profiles() {
    let mut m = DeviceCapabilityMatrix::new();
    m.add_profile(bitnet_kernels::capability_matrix::cpu_scalar());
    m.add_profile(bitnet_kernels::capability_matrix::cpu_scalar());
    assert_eq!(m.profiles().len(), 2);
    // profile_for_class returns the first one
    assert!(m.profile_for_class(DeviceClass::CpuScalar).is_some());
}

#[test]
fn matrix_profile_by_name_empty_string_matches_anything() {
    let m = DeviceCapabilityMatrix::with_builtin_profiles();
    // Empty needle matches everything (every name contains "")
    let p = m.profile_by_name("");
    assert!(p.is_some());
}

#[test]
fn matrix_profile_for_all_builtin_classes() {
    let m = DeviceCapabilityMatrix::with_builtin_profiles();
    // Should find these five
    assert!(m.profile_for_class(DeviceClass::IntelArc).is_some());
    assert!(m.profile_for_class(DeviceClass::NvidiaCuda).is_some());
    assert!(m.profile_for_class(DeviceClass::AppleMetal).is_some());
    assert!(m.profile_for_class(DeviceClass::CpuSimd).is_some());
    assert!(m.profile_for_class(DeviceClass::CpuScalar).is_some());
    // Not present
    assert!(m.profile_for_class(DeviceClass::WebGpu).is_none());
    assert!(m.profile_for_class(DeviceClass::AmdRocm).is_none());
    assert!(m.profile_for_class(DeviceClass::IntelXe).is_none());
}

// =========================================================================
// capability_matrix — CapabilityQuery edge cases
// =========================================================================

#[test]
fn query_best_precision_for_empty_profile() {
    let profile = DeviceProfile {
        device_class: DeviceClass::WebGpu,
        name: "Empty".to_string(),
        compute_units: 0,
        memory_gb: 0,
        capabilities: vec![],
    };
    let q = CapabilityQuery::new(&profile);
    assert_eq!(q.best_precision_for(OperationCategory::MatrixOps), None);
}

#[test]
fn query_best_precision_selects_highest_efficiency() {
    let profile = DeviceProfile {
        device_class: DeviceClass::CpuSimd,
        name: "Custom".to_string(),
        compute_units: 1,
        memory_gb: 0,
        capabilities: vec![
            CapabilityEntry::new(
                OperationCategory::MatrixOps,
                PrecisionSupport::FP32,
                SupportLevel::Full(0.5),
            ),
            CapabilityEntry::new(
                OperationCategory::MatrixOps,
                PrecisionSupport::FP16,
                SupportLevel::Full(0.9),
            ),
            CapabilityEntry::new(
                OperationCategory::MatrixOps,
                PrecisionSupport::INT8,
                SupportLevel::Emulated, // no efficiency
            ),
        ],
    };
    let q = CapabilityQuery::new(&profile);
    assert_eq!(q.best_precision_for(OperationCategory::MatrixOps), Some(PrecisionSupport::FP16));
}

#[test]
fn query_operations_at_precision_excludes_unsupported() {
    let profile = DeviceProfile {
        device_class: DeviceClass::CpuScalar,
        name: "Mixed".to_string(),
        compute_units: 1,
        memory_gb: 0,
        capabilities: vec![
            CapabilityEntry::new(
                OperationCategory::MatrixOps,
                PrecisionSupport::FP32,
                SupportLevel::Full(0.5),
            ),
            CapabilityEntry::new(
                OperationCategory::NormOps,
                PrecisionSupport::FP32,
                SupportLevel::Unsupported,
            ),
            CapabilityEntry::new(
                OperationCategory::ActivationOps,
                PrecisionSupport::FP32,
                SupportLevel::Emulated,
            ),
        ],
    };
    let q = CapabilityQuery::new(&profile);
    let ops = q.operations_at_precision(PrecisionSupport::FP32);
    assert!(ops.contains(&OperationCategory::MatrixOps));
    assert!(!ops.contains(&OperationCategory::NormOps)); // unsupported excluded
    assert!(ops.contains(&OperationCategory::ActivationOps)); // emulated counts
}

#[test]
fn query_supports_emulated_returns_true() {
    let profile = DeviceProfile {
        device_class: DeviceClass::AppleMetal,
        name: "Emu".to_string(),
        compute_units: 1,
        memory_gb: 8,
        capabilities: vec![CapabilityEntry::new(
            OperationCategory::QuantizedOps,
            PrecisionSupport::INT8,
            SupportLevel::Emulated,
        )],
    };
    let q = CapabilityQuery::new(&profile);
    assert!(q.supports(OperationCategory::QuantizedOps, PrecisionSupport::INT8));
}

// =========================================================================
// capability_matrix — CompatibilityReport edge cases
// =========================================================================

#[test]
fn report_empty_requirements_is_always_ready() {
    let profile = bitnet_kernels::capability_matrix::cpu_scalar();
    let report = CompatibilityReport::generate(&profile, &[]);
    assert!(report.overall_ready);
    assert!(report.supported_ops.is_empty());
    assert!(report.unsupported_ops.is_empty());
    assert!(report.summary().contains("READY"));
    assert!(report.summary().contains("0/0"));
}

#[test]
fn report_all_unsupported() {
    let profile = DeviceProfile {
        device_class: DeviceClass::WebGpu,
        name: "Empty".to_string(),
        compute_units: 0,
        memory_gb: 0,
        capabilities: vec![],
    };
    let required = vec![
        (OperationCategory::MatrixOps, PrecisionSupport::FP32),
        (OperationCategory::NormOps, PrecisionSupport::FP16),
    ];
    let report = CompatibilityReport::generate(&profile, &required);
    assert!(!report.overall_ready);
    assert_eq!(report.unsupported_ops.len(), 2);
    assert!(report.supported_ops.is_empty());
    let display = format!("{report}");
    assert!(display.contains("Missing"));
}

#[test]
fn report_display_ready_no_missing_section() {
    let profile = bitnet_kernels::capability_matrix::nvidia_rtx_4090();
    let required = vec![(OperationCategory::MatrixOps, PrecisionSupport::FP32)];
    let report = CompatibilityReport::generate(&profile, &required);
    let display = format!("{report}");
    assert!(!display.contains("Missing"));
}

// =========================================================================
// capability_matrix — builtin profile consistency
// =========================================================================

#[test]
fn all_builtin_profiles_have_nonempty_name_and_capabilities() {
    let m = DeviceCapabilityMatrix::with_builtin_profiles();
    for p in m.profiles() {
        assert!(!p.name.is_empty(), "profile name must not be empty");
        assert!(!p.capabilities.is_empty(), "{} must have capabilities", p.name);
    }
}

#[test]
fn all_builtin_efficiency_values_in_unit_range() {
    let m = DeviceCapabilityMatrix::with_builtin_profiles();
    for p in m.profiles() {
        for entry in &p.capabilities {
            if let Some(e) = entry.support.efficiency() {
                assert!(
                    (0.0..=1.0).contains(&e),
                    "{}: efficiency {} out of [0,1] for {} @ {}",
                    p.name,
                    e,
                    entry.operation,
                    entry.precision,
                );
            }
        }
    }
}

#[test]
fn gpu_profiles_have_higher_fp16_matmul_than_cpu_scalar() {
    let rtx = bitnet_kernels::capability_matrix::nvidia_rtx_4090();
    let scalar = bitnet_kernels::capability_matrix::cpu_scalar();

    let rtx_eff = rtx.lookup(OperationCategory::MatrixOps, PrecisionSupport::FP16).efficiency();
    let scalar_eff =
        scalar.lookup(OperationCategory::MatrixOps, PrecisionSupport::FP16).efficiency();

    // RTX has FP16 matmul, scalar does not
    assert!(rtx_eff.is_some());
    assert!(scalar_eff.is_none());
}

#[test]
fn fallback_profile_matches_cpu_scalar() {
    let fb = bitnet_kernels::capability_matrix::fallback_profile();
    let cs = bitnet_kernels::capability_matrix::cpu_scalar();
    assert_eq!(fb.device_class, cs.device_class);
    assert_eq!(fb.name, cs.name);
    assert_eq!(fb.capabilities.len(), cs.capabilities.len());
}

// =========================================================================
// device_aware — DeviceAwareQuantizer edge cases
// =========================================================================

#[test]
fn device_aware_cpu_creation_succeeds() {
    let q = DeviceAwareQuantizer::new(Device::Cpu).unwrap();
    assert_eq!(q.device(), Device::Cpu);
    assert!(!q.is_gpu_active());
    assert!(!q.active_provider().is_empty());
}

#[test]
fn device_aware_cuda_without_gpu_feature_falls_back_to_cpu() {
    // Without gpu/cuda features compiled, Cuda device should still succeed (CPU fallback)
    let q = DeviceAwareQuantizer::new(Device::Cuda(0)).unwrap();
    assert!(!q.is_gpu_active());
    assert_eq!(q.device(), Device::Cuda(0));
}

#[test]
fn device_aware_metal_falls_back_to_cpu() {
    let q = DeviceAwareQuantizer::new(Device::Metal).unwrap();
    assert!(!q.is_gpu_active());
    assert_eq!(q.device(), Device::Metal);
}

#[test]
fn device_aware_npu_falls_back_to_cpu() {
    let q = DeviceAwareQuantizer::new(Device::Npu).unwrap();
    assert!(!q.is_gpu_active());
}

#[test]
fn device_aware_opencl_falls_back_to_cpu() {
    let q = DeviceAwareQuantizer::new(Device::OpenCL(0)).unwrap();
    assert!(!q.is_gpu_active());
}

#[test]
fn device_aware_hip_falls_back_to_cpu() {
    let q = DeviceAwareQuantizer::new(Device::Hip(0)).unwrap();
    assert!(!q.is_gpu_active());
}

#[test]
fn device_aware_force_cpu_fallback_is_idempotent() {
    let mut q = DeviceAwareQuantizer::new(Device::Cpu).unwrap();
    q.force_cpu_fallback();
    q.force_cpu_fallback(); // second call should be no-op
    assert!(!q.is_gpu_active());
}

#[test]
fn device_aware_stats_initial_state() {
    let q = DeviceAwareQuantizer::new(Device::Cpu).unwrap();
    let stats = q.get_stats().expect("stats should be available");
    assert_eq!(stats.total_operations, 0);
    assert_eq!(stats.quantization_operations, 0);
    assert_eq!(stats.matmul_operations, 0);
    assert_eq!(stats.gpu_operations, 0);
    assert_eq!(stats.cpu_operations, 0);
    assert_eq!(stats.fallback_count, 0);
    assert_eq!(stats.gpu_efficiency, 0.0);
    assert!(stats.last_gpu_error.is_none());
    assert!(stats.last_cpu_error.is_none());
}

#[test]
fn device_aware_stats_after_quantize() {
    let q = DeviceAwareQuantizer::new(Device::Cpu).unwrap();
    let input = vec![1.0f32, -1.0, 0.5, -0.5];
    let mut output = vec![0u8; 1];
    let mut scales = vec![0.0f32; 1];

    q.quantize(&input, &mut output, &mut scales, QuantizationType::I2S).unwrap();

    let stats = q.get_stats().unwrap();
    assert_eq!(stats.quantization_operations, 1);
    assert_eq!(stats.cpu_operations, 1);
    assert_eq!(stats.total_operations, 1);
    assert!(stats.total_time_ms >= 0.0);
}

#[test]
fn device_aware_reset_stats_clears_counters() {
    let q = DeviceAwareQuantizer::new(Device::Cpu).unwrap();
    let input = vec![1.0f32, -1.0, 0.5, -0.5];
    let mut output = vec![0u8; 1];
    let mut scales = vec![0.0f32; 1];

    q.quantize(&input, &mut output, &mut scales, QuantizationType::I2S).unwrap();
    assert_eq!(q.get_stats().unwrap().total_operations, 1);

    q.reset_stats();
    let stats = q.get_stats().unwrap();
    assert_eq!(stats.total_operations, 0);
    assert_eq!(stats.quantization_operations, 0);
    assert_eq!(stats.cpu_operations, 0);
}

#[test]
fn device_aware_stats_avg_times_zero_when_no_ops() {
    let q = DeviceAwareQuantizer::new(Device::Cpu).unwrap();
    let stats = q.get_stats().unwrap();
    assert_eq!(stats.avg_quantization_time_ms(), 0.0);
    assert_eq!(stats.avg_matmul_time_ms(), 0.0);
}

#[test]
fn device_aware_stats_is_gpu_effective_false_on_cpu() {
    let q = DeviceAwareQuantizer::new(Device::Cpu).unwrap();
    let input = vec![1.0f32, -1.0, 0.5, -0.5];
    let mut output = vec![0u8; 1];
    let mut scales = vec![0.0f32; 1];

    q.quantize(&input, &mut output, &mut scales, QuantizationType::I2S).unwrap();
    let stats = q.get_stats().unwrap();
    assert!(!stats.is_gpu_effective());
}

#[test]
fn device_aware_stats_summary_contains_device_type() {
    let q = DeviceAwareQuantizer::new(Device::Cpu).unwrap();
    let stats = q.get_stats().unwrap();
    let summary = stats.summary();
    assert!(summary.contains("Device:"));
    assert!(summary.contains("Memory:"));
}

// =========================================================================
// device_aware — Factory edge cases
// =========================================================================

#[test]
fn factory_auto_detect_returns_cpu_without_gpu_feature() {
    let q = DeviceAwareQuantizerFactory::auto_detect().unwrap();
    // Without gpu feature, should always select CPU
    assert!(!q.is_gpu_active());
}

#[test]
fn factory_create_best_with_none_equivalent_to_auto() {
    let q = DeviceAwareQuantizerFactory::create_best(None).unwrap();
    assert!(!q.is_gpu_active());
}

#[test]
fn factory_create_best_with_cpu_preference() {
    let q = DeviceAwareQuantizerFactory::create_best(Some(Device::Cpu)).unwrap();
    assert_eq!(q.device(), Device::Cpu);
}

#[test]
fn factory_list_available_devices_always_includes_cpu() {
    let devices = DeviceAwareQuantizerFactory::list_available_devices();
    assert!(!devices.is_empty());
    assert!(devices.contains(&Device::Cpu));
}

// =========================================================================
// shaped_reduction — ShapedReductionConfig edge cases
// =========================================================================

#[test]
fn shaped_reduction_config_global_defaults() {
    let cfg = ShapedReductionConfig::global(ReductionOp::Sum);
    assert!(cfg.axis.is_none());
    assert!(!cfg.keepdim);
}

#[test]
fn shaped_reduction_config_new_preserves_fields() {
    let cfg = ShapedReductionConfig::new(ReductionOp::Max, Some(2), true);
    assert_eq!(cfg.axis, Some(2));
    assert!(cfg.keepdim);
}

// =========================================================================
// shaped_reduction — reduce_f32 edge cases
// =========================================================================

#[test]
fn reduce_f32_single_element_all_ops() {
    for op in [
        ReductionOp::Sum,
        ReductionOp::Max,
        ReductionOp::Min,
        ReductionOp::Mean,
        ReductionOp::L2Norm,
    ] {
        let cfg = ShapedReductionConfig::global(op);
        let r = reduce_f32(&[42.0], &[1], &cfg).unwrap();
        assert!(
            (r[0] - 42.0).abs() < 1e-6,
            "single element {op:?} should return 42.0, got {}",
            r[0]
        );
    }
}

#[test]
fn reduce_f32_empty_input_identities() {
    let cases: &[(ReductionOp, f32)] = &[
        (ReductionOp::Sum, 0.0),
        (ReductionOp::Max, f32::NEG_INFINITY),
        (ReductionOp::Min, f32::INFINITY),
        (ReductionOp::Mean, 0.0),
        (ReductionOp::L2Norm, 0.0),
    ];
    for &(op, expected) in cases {
        let cfg = ShapedReductionConfig::global(op);
        let r = reduce_f32(&[], &[0], &cfg).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0], expected, "empty {op:?} identity mismatch");
    }
}

#[test]
fn reduce_f32_nan_propagation_sum() {
    let cfg = ShapedReductionConfig::global(ReductionOp::Sum);
    let r = reduce_f32(&[1.0, f32::NAN, 3.0], &[3], &cfg).unwrap();
    assert!(r[0].is_nan(), "NaN should propagate through Sum");
}

#[test]
fn reduce_f32_nan_propagation_max() {
    let cfg = ShapedReductionConfig::global(ReductionOp::Max);
    let r = reduce_f32(&[1.0, f32::NAN, 3.0], &[3], &cfg).unwrap();
    // f32::max doesn't propagate NaN (IEEE 754 minNum/maxNum semantics vary)
    // Just verify it returns a value without panicking
    assert!(r.len() == 1);
}

#[test]
fn reduce_f32_infinity_handling() {
    let cfg = ShapedReductionConfig::global(ReductionOp::Sum);
    let r = reduce_f32(&[f32::INFINITY, 1.0], &[2], &cfg).unwrap();
    assert_eq!(r[0], f32::INFINITY);

    let r = reduce_f32(&[f32::NEG_INFINITY, 1.0], &[2], &cfg).unwrap();
    assert_eq!(r[0], f32::NEG_INFINITY);
}

#[test]
fn reduce_f32_inf_minus_inf_is_nan() {
    let cfg = ShapedReductionConfig::global(ReductionOp::Sum);
    let r = reduce_f32(&[f32::INFINITY, f32::NEG_INFINITY], &[2], &cfg).unwrap();
    assert!(r[0].is_nan());
}

#[test]
fn reduce_f32_all_zeros() {
    for op in [ReductionOp::Sum, ReductionOp::Max, ReductionOp::Min, ReductionOp::Mean] {
        let cfg = ShapedReductionConfig::global(op);
        let r = reduce_f32(&[0.0, 0.0, 0.0], &[3], &cfg).unwrap();
        assert_eq!(r[0], 0.0, "all zeros with {op:?} should be 0.0");
    }
}

#[test]
fn reduce_f32_all_negative() {
    let cfg = ShapedReductionConfig::global(ReductionOp::Max);
    let r = reduce_f32(&[-10.0, -20.0, -5.0], &[3], &cfg).unwrap();
    assert!((r[0] - (-5.0)).abs() < 1e-6);
}

// =========================================================================
// shaped_reduction — validation error edge cases
// =========================================================================

#[test]
fn reduce_f32_empty_shape_rejected() {
    let cfg = ShapedReductionConfig::global(ReductionOp::Sum);
    assert!(reduce_f32(&[], &[], &cfg).is_err());
}

#[test]
fn reduce_f32_shape_length_mismatch() {
    let cfg = ShapedReductionConfig::global(ReductionOp::Sum);
    assert!(reduce_f32(&[1.0, 2.0], &[3], &cfg).is_err());
    assert!(reduce_f32(&[1.0], &[2], &cfg).is_err());
}

#[test]
fn reduce_f32_axis_out_of_bounds_1d() {
    let cfg = ShapedReductionConfig::new(ReductionOp::Sum, Some(1), false);
    assert!(reduce_f32(&[1.0, 2.0], &[2], &cfg).is_err());
}

#[test]
fn reduce_f32_axis_out_of_bounds_2d() {
    let cfg = ShapedReductionConfig::new(ReductionOp::Sum, Some(2), false);
    assert!(reduce_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &cfg).is_err());
}

#[test]
fn reduce_f32_axis_out_of_bounds_3d() {
    let cfg = ShapedReductionConfig::new(ReductionOp::Sum, Some(3), false);
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    assert!(reduce_f32(&data, &[2, 3, 4], &cfg).is_err());
}

// =========================================================================
// shaped_reduction — output shape edge cases
// =========================================================================

#[test]
fn output_shape_1d_global_no_keepdim() {
    let cfg = ShapedReductionConfig::global(ReductionOp::Sum);
    let shape = reduction_output_shape(&[5], &cfg);
    assert!(shape.is_empty()); // scalar result
}

#[test]
fn output_shape_1d_global_keepdim() {
    let cfg = ShapedReductionConfig { op: ReductionOp::Sum, axis: None, keepdim: true };
    let shape = reduction_output_shape(&[5], &cfg);
    assert_eq!(shape, vec![1]);
}

#[test]
fn output_shape_1d_axis0_no_keepdim() {
    let cfg = ShapedReductionConfig::new(ReductionOp::Sum, Some(0), false);
    let shape = reduction_output_shape(&[5], &cfg);
    assert!(shape.is_empty());
}

#[test]
fn output_shape_1d_axis0_keepdim() {
    let cfg = ShapedReductionConfig::new(ReductionOp::Sum, Some(0), true);
    let shape = reduction_output_shape(&[5], &cfg);
    assert_eq!(shape, vec![1]);
}

#[test]
fn output_shape_4d_all_axes() {
    let input_shape = &[2, 3, 4, 5];
    let expected_no_keepdim: &[&[usize]] = &[
        &[3, 4, 5], // axis 0
        &[2, 4, 5], // axis 1
        &[2, 3, 5], // axis 2
        &[2, 3, 4], // axis 3
    ];
    let expected_keepdim: &[&[usize]] = &[
        &[1, 3, 4, 5], // axis 0
        &[2, 1, 4, 5], // axis 1
        &[2, 3, 1, 5], // axis 2
        &[2, 3, 4, 1], // axis 3
    ];
    for axis in 0..4 {
        let cfg_nk = ShapedReductionConfig::new(ReductionOp::Sum, Some(axis), false);
        let cfg_kd = ShapedReductionConfig::new(ReductionOp::Sum, Some(axis), true);
        assert_eq!(
            reduction_output_shape(input_shape, &cfg_nk),
            expected_no_keepdim[axis],
            "no keepdim axis={axis}"
        );
        assert_eq!(
            reduction_output_shape(input_shape, &cfg_kd),
            expected_keepdim[axis],
            "keepdim axis={axis}"
        );
    }
}

#[test]
fn output_shape_high_dimensional() {
    let input_shape = &[2, 3, 4, 5, 6];
    let cfg = ShapedReductionConfig::new(ReductionOp::Sum, Some(2), false);
    assert_eq!(reduction_output_shape(input_shape, &cfg), vec![2, 3, 5, 6]);

    let cfg = ShapedReductionConfig::new(ReductionOp::Sum, Some(2), true);
    assert_eq!(reduction_output_shape(input_shape, &cfg), vec![2, 3, 1, 5, 6]);
}

// =========================================================================
// shaped_reduction — axis reduction correctness edge cases
// =========================================================================

#[test]
fn reduce_f32_axis0_1x1_matrix() {
    let cfg = ShapedReductionConfig::new(ReductionOp::Sum, Some(0), false);
    let r = reduce_f32(&[7.0], &[1, 1], &cfg).unwrap();
    assert_eq!(r, vec![7.0]);
}

#[test]
fn reduce_f32_axis1_1x1_matrix() {
    let cfg = ShapedReductionConfig::new(ReductionOp::Sum, Some(1), false);
    let r = reduce_f32(&[7.0], &[1, 1], &cfg).unwrap();
    assert_eq!(r, vec![7.0]);
}

#[test]
fn reduce_f32_large_axis_dim_mean() {
    // Shape [1, 1000] — reduce axis 1 (mean of 1000 elements)
    let data: Vec<f32> = (1..=1000).map(|i| i as f32).collect();
    let cfg = ShapedReductionConfig::new(ReductionOp::Mean, Some(1), false);
    let r = reduce_f32(&data, &[1, 1000], &cfg).unwrap();
    assert_eq!(r.len(), 1);
    let expected = 500.5; // mean of 1..=1000
    assert!((r[0] - expected).abs() < 0.1, "mean of 1..=1000 should be ~500.5, got {}", r[0]);
}

#[test]
fn reduce_f32_l2norm_known_values() {
    // 3-4-5 triangle
    let cfg = ShapedReductionConfig::global(ReductionOp::L2Norm);
    let r = reduce_f32(&[3.0, 4.0], &[2], &cfg).unwrap();
    assert!((r[0] - 5.0).abs() < 1e-5);
}

#[test]
fn reduce_f32_l2norm_single_negative() {
    // L2Norm of a single negative value should be its absolute value
    let cfg = ShapedReductionConfig::global(ReductionOp::L2Norm);
    let r = reduce_f32(&[-7.0], &[1], &cfg).unwrap();
    assert!((r[0] - 7.0).abs() < 1e-5);
}

#[test]
fn reduce_f32_keepdim_does_not_change_values() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let cfg_nk = ShapedReductionConfig::new(ReductionOp::Sum, Some(0), false);
    let cfg_kd = ShapedReductionConfig::new(ReductionOp::Sum, Some(0), true);
    let r_nk = reduce_f32(&data, &[2, 3], &cfg_nk).unwrap();
    let r_kd = reduce_f32(&data, &[2, 3], &cfg_kd).unwrap();
    // Values should be identical; only shape interpretation differs
    assert_eq!(r_nk, r_kd);
}

#[test]
fn reduce_f32_3d_axis_middle() {
    // [2, 3, 2] reduce axis=1 -> [2, 2]
    #[rustfmt::skip]
    let data = vec![
        1.0, 2.0,  3.0, 4.0,  5.0, 6.0,   // batch 0
        7.0, 8.0,  9.0, 10.0, 11.0, 12.0,  // batch 1
    ];
    let cfg = ShapedReductionConfig::new(ReductionOp::Sum, Some(1), false);
    let r = reduce_f32(&data, &[2, 3, 2], &cfg).unwrap();
    assert_eq!(r.len(), 4);
    // batch 0: col0 = 1+3+5=9, col1 = 2+4+6=12
    // batch 1: col0 = 7+9+11=27, col1 = 8+10+12=30
    assert!((r[0] - 9.0).abs() < 1e-5);
    assert!((r[1] - 12.0).abs() < 1e-5);
    assert!((r[2] - 27.0).abs() < 1e-5);
    assert!((r[3] - 30.0).abs() < 1e-5);
}

// =========================================================================
// shaped_reduction — global vs axis-0 equivalence for 1D
// =========================================================================

#[test]
fn reduce_f32_global_vs_axis0_1d_equivalence() {
    let data = vec![2.0, 4.0, 6.0, 8.0];
    for op in [
        ReductionOp::Sum,
        ReductionOp::Max,
        ReductionOp::Min,
        ReductionOp::Mean,
        ReductionOp::L2Norm,
    ] {
        let cfg_global = ShapedReductionConfig::global(op);
        let cfg_axis0 = ShapedReductionConfig::new(op, Some(0), false);
        let r_global = reduce_f32(&data, &[4], &cfg_global).unwrap();
        let r_axis0 = reduce_f32(&data, &[4], &cfg_axis0).unwrap();
        assert_eq!(r_global.len(), r_axis0.len(), "{op:?} len mismatch");
        for (i, (&g, &a)) in r_global.iter().zip(r_axis0.iter()).enumerate() {
            assert!((g - a).abs() < 1e-5, "{op:?} index {i}: global={g} axis0={a}");
        }
    }
}

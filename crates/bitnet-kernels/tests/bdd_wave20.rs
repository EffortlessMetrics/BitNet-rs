//! BDD-style integration tests — Wave 20
//!
//! Each test follows the `given_*_when_*_then_*` naming convention and exercises
//! kernel registry operations, dispatch tables, capability matrices, SIMD
//! diagnostics, activation/normalization registries, and fallback chains.

use bitnet_kernels::activation_registry::{
    ActivationType, activate, activate_inplace, activate_vec,
};
use bitnet_kernels::capability_matrix::{
    CapabilityQuery, CompatibilityReport, DeviceCapabilityMatrix, DeviceClass, DeviceProfile,
    OperationCategory, PrecisionSupport, SupportLevel, cpu_scalar, fallback_profile,
    intel_arc_a770, nvidia_rtx_4090,
};
use bitnet_kernels::dispatch_table::{DispatchBackend, DispatchTable, KernelOp};
use bitnet_kernels::norm_registry::{NormConfig, NormType, layer_norm, rms_norm};
use bitnet_kernels::simd_diagnostics::{
    SimdCapabilities, SimdLevel, format_diagnostics, recommend_dispatch,
};
use bitnet_kernels::{FallbackKernel, KernelManager, KernelProvider};

const TOL: f32 = 1e-5;

fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() < TOL
}

// ═══════════════════════════════════════════════════════════════════
// Section 1 — KernelManager / Registry
// ═══════════════════════════════════════════════════════════════════

#[test]
fn given_default_build_when_creating_kernel_manager_then_at_least_one_provider_available() {
    // Given: a default (CPU) build
    // When: creating a KernelManager
    let mgr = KernelManager::new();
    // Then: at least one provider is listed
    assert!(!mgr.list_available_providers().is_empty());
}

#[test]
fn given_kernel_manager_when_listing_providers_then_fallback_is_always_present() {
    // Given: a KernelManager
    let mgr = KernelManager::new();
    // When: listing available providers
    let providers = mgr.list_available_providers();
    // Then: "fallback" is in the list
    assert!(providers.contains(&"fallback"));
}

#[test]
fn given_kernel_manager_when_selecting_best_then_returns_ok() {
    // Given: a KernelManager
    let mgr = KernelManager::new();
    // When: selecting the best provider
    let result = mgr.select_best();
    // Then: selection succeeds
    assert!(result.is_ok());
}

#[test]
fn given_kernel_manager_when_selecting_best_then_provider_is_available() {
    // Given: a KernelManager
    let mgr = KernelManager::new();
    // When: selecting the best provider
    let provider = mgr.select_best().unwrap();
    // Then: it reports itself as available
    assert!(provider.is_available());
}

#[test]
fn given_kernel_manager_when_selecting_best_then_provider_name_is_non_empty() {
    // Given: a KernelManager
    let mgr = KernelManager::new();
    // When: selecting the best provider
    let provider = mgr.select_best().unwrap();
    // Then: its name is not empty
    assert!(!provider.name().is_empty());
}

#[test]
fn given_kernel_manager_when_selected_provider_name_called_then_returns_some() {
    // Given: a KernelManager with a selection
    let mgr = KernelManager::new();
    let _ = mgr.select_best();
    // When: querying the selected provider name
    let name = mgr.selected_provider_name();
    // Then: it returns Some
    assert!(name.is_some());
}

#[test]
fn given_fallback_kernel_when_querying_name_then_returns_fallback() {
    // Given: a FallbackKernel
    let kernel = FallbackKernel;
    // When: querying its name
    // Then: it returns "fallback"
    assert_eq!(kernel.name(), "fallback");
}

#[test]
fn given_fallback_kernel_when_checking_availability_then_always_true() {
    // Given: a FallbackKernel
    let kernel = FallbackKernel;
    // When: checking availability
    // Then: always available
    assert!(kernel.is_available());
}

// ═══════════════════════════════════════════════════════════════════
// Section 2 — DispatchTable
// ═══════════════════════════════════════════════════════════════════

#[test]
fn given_empty_dispatch_table_when_resolving_any_op_then_returns_none() {
    // Given: an empty table
    let table = DispatchTable::new();
    // When: resolving MatMul
    // Then: None
    assert!(table.resolve(KernelOp::MatMul).is_none());
}

#[test]
fn given_single_entry_when_resolving_then_returns_that_backend() {
    // Given: a table with one scalar MatMul entry
    let mut table = DispatchTable::new();
    table.register(KernelOp::MatMul, DispatchBackend::Scalar, 1, true);
    // When: resolving MatMul
    let result = table.resolve(KernelOp::MatMul);
    // Then: returns Scalar
    assert_eq!(result, Some(DispatchBackend::Scalar));
}

#[test]
fn given_two_entries_when_resolving_then_highest_priority_wins() {
    // Given: scalar (pri=1) and avx2 (pri=10) for MatMul
    let mut table = DispatchTable::new();
    table.register(KernelOp::MatMul, DispatchBackend::Scalar, 1, true);
    table.register(KernelOp::MatMul, DispatchBackend::Avx2, 10, true);
    // When: resolving
    let result = table.resolve(KernelOp::MatMul);
    // Then: AVX2 wins
    assert_eq!(result, Some(DispatchBackend::Avx2));
}

#[test]
fn given_unavailable_high_priority_when_resolving_then_falls_back() {
    // Given: CUDA (pri=100, unavailable) and Scalar (pri=1, available)
    let mut table = DispatchTable::new();
    table.register(KernelOp::MatMul, DispatchBackend::Scalar, 1, true);
    table.register(KernelOp::MatMul, DispatchBackend::Cuda, 100, false);
    // When: resolving
    let result = table.resolve(KernelOp::MatMul);
    // Then: falls back to Scalar
    assert_eq!(result, Some(DispatchBackend::Scalar));
}

#[test]
fn given_override_set_when_resolving_then_override_takes_precedence() {
    // Given: scalar (pri=1) and avx2 (pri=10), override to Scalar
    let mut table = DispatchTable::new();
    table.register(KernelOp::MatMul, DispatchBackend::Scalar, 1, true);
    table.register(KernelOp::MatMul, DispatchBackend::Avx2, 10, true);
    table.override_backend(KernelOp::MatMul, DispatchBackend::Scalar);
    // When: resolving
    let result = table.resolve(KernelOp::MatMul);
    // Then: override wins
    assert_eq!(result, Some(DispatchBackend::Scalar));
}

#[test]
fn given_override_for_unavailable_backend_when_resolving_then_ignores_override() {
    // Given: override points to CUDA which is registered but unavailable
    let mut table = DispatchTable::new();
    table.register(KernelOp::MatMul, DispatchBackend::Scalar, 1, true);
    table.register(KernelOp::MatMul, DispatchBackend::Cuda, 100, false);
    table.override_backend(KernelOp::MatMul, DispatchBackend::Cuda);
    // When: resolving
    let result = table.resolve(KernelOp::MatMul);
    // Then: falls back to priority-based resolution
    assert_eq!(result, Some(DispatchBackend::Scalar));
}

#[test]
fn given_cpu_defaults_when_resolving_matmul_then_returns_some_backend() {
    // Given: CPU default table
    let table = DispatchTable::cpu_defaults();
    // When: resolving MatMul
    let result = table.resolve(KernelOp::MatMul);
    // Then: some backend is returned
    assert!(result.is_some());
}

#[test]
fn given_cpu_defaults_when_resolving_all_ops_then_all_have_backends() {
    // Given: CPU default table
    let table = DispatchTable::cpu_defaults();
    // When/Then: every known op resolves
    for &op in KernelOp::all() {
        assert!(table.resolve(op).is_some(), "op {:?} should resolve", op);
    }
}

#[test]
fn given_cpu_defaults_when_listing_backends_for_matmul_then_scalar_present() {
    // Given: CPU default table
    let table = DispatchTable::cpu_defaults();
    // When: listing available backends for MatMul
    let backends = table.available_backends(KernelOp::MatMul);
    // Then: Scalar is present
    assert!(backends.contains(&DispatchBackend::Scalar));
}

#[test]
fn given_empty_table_when_checking_unsupported_ops_then_all_ops_unsupported() {
    // Given: empty table
    let table = DispatchTable::new();
    // When: checking unsupported ops
    let unsupported = table.unsupported_ops();
    // Then: all ops are unsupported
    assert_eq!(unsupported.len(), KernelOp::all().len());
}

#[test]
fn given_cpu_defaults_when_checking_unsupported_ops_then_empty() {
    // Given: CPU default table
    let table = DispatchTable::cpu_defaults();
    // When: checking unsupported ops
    let unsupported = table.unsupported_ops();
    // Then: all ops covered
    assert!(unsupported.is_empty());
}

#[test]
fn given_dispatch_table_when_counting_entries_then_returns_correct_count() {
    // Given: a table with 3 entries
    let mut table = DispatchTable::new();
    table.register(KernelOp::MatMul, DispatchBackend::Scalar, 1, true);
    table.register(KernelOp::MatMul, DispatchBackend::Avx2, 10, true);
    table.register(KernelOp::Softmax, DispatchBackend::Scalar, 1, true);
    // When: counting
    // Then: 3
    assert_eq!(table.entry_count(), 3);
}

// ═══════════════════════════════════════════════════════════════════
// Section 3 — KernelOp / DispatchBackend properties
// ═══════════════════════════════════════════════════════════════════

#[test]
fn given_kernel_op_all_when_counting_then_returns_13() {
    // Given/When: KernelOp::all()
    // Then: 13 operations
    assert_eq!(KernelOp::all().len(), 13);
}

#[test]
fn given_kernel_op_when_querying_name_then_non_empty_string() {
    // Given: every kernel op
    for &op in KernelOp::all() {
        // When: querying name
        let name = op.name();
        // Then: non-empty
        assert!(!name.is_empty(), "op {:?} has empty name", op);
    }
}

#[test]
fn given_dispatch_backend_scalar_when_checking_simd_then_false() {
    // Given: Scalar backend
    // When/Then: is_simd is false
    assert!(!DispatchBackend::Scalar.is_simd());
}

#[test]
fn given_dispatch_backend_avx2_when_checking_simd_then_true() {
    // Given: Avx2 backend
    // When/Then: is_simd is true
    assert!(DispatchBackend::Avx2.is_simd());
}

#[test]
fn given_dispatch_backend_cuda_when_checking_gpu_then_true() {
    // Given: CUDA backend
    // When/Then: is_gpu is true
    assert!(DispatchBackend::Cuda.is_gpu());
}

#[test]
fn given_dispatch_backend_metal_when_checking_gpu_then_true() {
    assert!(DispatchBackend::Metal.is_gpu());
}

#[test]
fn given_dispatch_backend_opencl_when_checking_gpu_then_true() {
    assert!(DispatchBackend::OpenCL.is_gpu());
}

#[test]
fn given_dispatch_backend_scalar_when_checking_gpu_then_false() {
    assert!(!DispatchBackend::Scalar.is_gpu());
}

#[test]
fn given_dispatch_backend_neon_when_checking_simd_then_true() {
    assert!(DispatchBackend::Neon.is_simd());
}

#[test]
fn given_dispatch_backend_avx512_when_checking_simd_then_true() {
    assert!(DispatchBackend::Avx512.is_simd());
}

// ═══════════════════════════════════════════════════════════════════
// Section 4 — Capability Matrix & Device Profiles
// ═══════════════════════════════════════════════════════════════════

#[test]
fn given_empty_matrix_when_querying_profiles_then_empty() {
    // Given: empty matrix
    let matrix = DeviceCapabilityMatrix::new();
    // When/Then: no profiles
    assert!(matrix.profiles().is_empty());
}

#[test]
fn given_builtin_matrix_when_querying_profiles_then_has_five_profiles() {
    // Given: matrix with builtin profiles
    let matrix = DeviceCapabilityMatrix::with_builtin_profiles();
    // When/Then: exactly 5 profiles
    assert_eq!(matrix.profiles().len(), 5);
}

#[test]
fn given_builtin_matrix_when_querying_intel_arc_then_found() {
    // Given: builtin matrix
    let matrix = DeviceCapabilityMatrix::with_builtin_profiles();
    // When: querying IntelArc
    let profile = matrix.profile_for_class(DeviceClass::IntelArc);
    // Then: found
    assert!(profile.is_some());
}

#[test]
fn given_builtin_matrix_when_querying_nvidia_cuda_then_found() {
    let matrix = DeviceCapabilityMatrix::with_builtin_profiles();
    assert!(matrix.profile_for_class(DeviceClass::NvidiaCuda).is_some());
}

#[test]
fn given_builtin_matrix_when_querying_apple_metal_then_found() {
    let matrix = DeviceCapabilityMatrix::with_builtin_profiles();
    assert!(matrix.profile_for_class(DeviceClass::AppleMetal).is_some());
}

#[test]
fn given_builtin_matrix_when_querying_by_name_then_case_insensitive() {
    // Given: builtin matrix
    let matrix = DeviceCapabilityMatrix::with_builtin_profiles();
    // When: querying by lowercase name
    let profile = matrix.profile_by_name("arc a770");
    // Then: found
    assert!(profile.is_some());
}

#[test]
fn given_cpu_scalar_profile_when_checking_matmul_fp32_then_supported() {
    // Given: CPU scalar profile
    let profile = cpu_scalar();
    let query = CapabilityQuery::new(&profile);
    // When/Then: MatrixOps at FP32 is supported
    assert!(query.supports(OperationCategory::MatrixOps, PrecisionSupport::FP32));
}

#[test]
fn given_intel_arc_profile_when_finding_best_precision_for_matmul_then_returns_some() {
    // Given: Intel Arc profile
    let profile = intel_arc_a770();
    let query = CapabilityQuery::new(&profile);
    // When: finding best precision for MatrixOps
    let best = query.best_precision_for(OperationCategory::MatrixOps);
    // Then: returns some precision
    assert!(best.is_some());
}

#[test]
fn given_nvidia_profile_when_listing_ops_at_fp16_then_non_empty() {
    // Given: NVIDIA RTX 4090 profile
    let profile = nvidia_rtx_4090();
    let query = CapabilityQuery::new(&profile);
    // When: listing operations at FP16
    let ops = query.operations_at_precision(PrecisionSupport::FP16);
    // Then: at least one op
    assert!(!ops.is_empty());
}

#[test]
fn given_device_profile_when_counting_full_support_then_positive() {
    // Given: Intel Arc profile
    let profile = intel_arc_a770();
    // When: counting full support
    // Then: at least one
    assert!(profile.full_support_count() > 0);
}

#[test]
fn given_fallback_profile_when_checking_capabilities_then_all_emulated_or_supported() {
    // Given: fallback (worst-case) profile
    let profile = fallback_profile();
    // When: checking each capability
    for entry in &profile.capabilities {
        // Then: all are at least emulated
        assert!(entry.support.is_supported());
    }
}

// ═══════════════════════════════════════════════════════════════════
// Section 5 — CompatibilityReport
// ═══════════════════════════════════════════════════════════════════

#[test]
fn given_nvidia_profile_when_all_required_ops_supported_then_overall_ready() {
    // Given: NVIDIA RTX 4090 + required ops
    let profile = nvidia_rtx_4090();
    let required = vec![(OperationCategory::MatrixOps, PrecisionSupport::FP32)];
    // When: generating report
    let report = CompatibilityReport::generate(&profile, &required);
    // Then: overall ready
    assert!(report.overall_ready);
}

#[test]
fn given_profile_when_missing_op_then_not_ready() {
    // Given: a profile with no capabilities
    let profile = DeviceProfile {
        device_class: DeviceClass::CpuScalar,
        name: "empty".to_string(),
        compute_units: 1,
        memory_gb: 1,
        capabilities: vec![],
    };
    let required = vec![(OperationCategory::MatrixOps, PrecisionSupport::FP32)];
    // When: generating report
    let report = CompatibilityReport::generate(&profile, &required);
    // Then: not ready
    assert!(!report.overall_ready);
    assert_eq!(report.unsupported_ops.len(), 1);
}

#[test]
fn given_compatibility_report_when_formatting_summary_then_contains_device_name() {
    // Given: a report for Intel Arc
    let profile = intel_arc_a770();
    let report = CompatibilityReport::generate(&profile, &[]);
    // When: formatting summary
    let summary = report.summary();
    // Then: contains "Intel Arc A770"
    assert!(summary.contains("Intel Arc A770"));
}

// ═══════════════════════════════════════════════════════════════════
// Section 6 — SIMD Diagnostics
// ═══════════════════════════════════════════════════════════════════

#[test]
fn given_runtime_when_detecting_simd_then_arch_non_empty() {
    // Given/When: detecting SIMD capabilities
    let caps = SimdCapabilities::detect();
    // Then: arch is non-empty
    assert!(!caps.arch.is_empty());
}

#[test]
fn given_simd_caps_when_querying_best_level_then_at_least_scalar() {
    // Given: detected capabilities
    let caps = SimdCapabilities::detect();
    // When: querying best level
    let level = caps.best_level();
    // Then: at least Scalar
    assert!(level >= SimdLevel::Scalar);
}

#[test]
fn given_simd_caps_when_querying_vector_width_then_at_least_64() {
    // Given: detected capabilities
    let caps = SimdCapabilities::detect();
    // When: querying vector width
    let width = caps.vector_width_bits();
    // Then: at least 64 (scalar)
    assert!(width >= 64);
}

#[test]
fn given_simd_caps_when_querying_f32_lanes_then_at_least_2() {
    // Given: detected capabilities
    let caps = SimdCapabilities::detect();
    // When: querying f32 lanes
    let lanes = caps.f32_lanes();
    // Then: at least 2 (64 / 32)
    assert!(lanes >= 2);
}

#[test]
fn given_simd_caps_when_formatting_summary_then_contains_arch() {
    // Given: detected capabilities
    let caps = SimdCapabilities::detect();
    // When: formatting summary
    let summary = caps.summary();
    // Then: contains the arch
    assert!(summary.contains(&caps.arch));
}

#[test]
fn given_simd_caps_when_getting_recommendations_then_non_empty() {
    // Given: detected capabilities
    let caps = SimdCapabilities::detect();
    // When: getting dispatch recommendations
    let recs = recommend_dispatch(&caps);
    // Then: at least the 4 default recommendations
    assert!(recs.len() >= 4);
}

#[test]
fn given_simd_caps_when_formatting_diagnostics_then_contains_dispatch_plan() {
    // Given: detected capabilities
    let caps = SimdCapabilities::detect();
    // When: formatting diagnostics
    let diag = format_diagnostics(&caps);
    // Then: contains "Dispatch plan"
    assert!(diag.contains("Dispatch plan"));
}

#[test]
fn given_simd_level_scalar_when_displaying_then_shows_scalar() {
    assert_eq!(SimdLevel::Scalar.display_name(), "Scalar");
}

#[test]
fn given_simd_level_avx2_when_displaying_then_shows_avx2() {
    assert_eq!(SimdLevel::Avx2.display_name(), "AVX2");
}

#[test]
fn given_simd_level_ordering_when_comparing_then_scalar_less_than_avx2() {
    assert!(SimdLevel::Scalar < SimdLevel::Avx2);
    assert!(SimdLevel::Avx2 < SimdLevel::Avx512);
}

// ═══════════════════════════════════════════════════════════════════
// Section 7 — Activation Registry
// ═══════════════════════════════════════════════════════════════════

#[test]
fn given_relu_when_activate_positive_then_unchanged() {
    // Given: ReLU activation
    // When: activating positive value
    let result = activate(2.0, ActivationType::ReLU);
    // Then: unchanged
    assert!(approx_eq(result, 2.0));
}

#[test]
fn given_relu_when_activate_negative_then_zero() {
    let result = activate(-3.0, ActivationType::ReLU);
    assert!(approx_eq(result, 0.0));
}

#[test]
fn given_sigmoid_when_activate_zero_then_half() {
    let result = activate(0.0, ActivationType::Sigmoid);
    assert!(approx_eq(result, 0.5));
}

#[test]
fn given_tanh_when_activate_zero_then_zero() {
    let result = activate(0.0, ActivationType::Tanh);
    assert!(approx_eq(result, 0.0));
}

#[test]
fn given_silu_when_activate_zero_then_zero() {
    // SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    let result = activate(0.0, ActivationType::SiLU);
    assert!(approx_eq(result, 0.0));
}

#[test]
fn given_activation_type_when_parsing_known_name_then_returns_some() {
    assert_eq!(ActivationType::from_name("relu"), Some(ActivationType::ReLU));
    assert_eq!(ActivationType::from_name("silu"), Some(ActivationType::SiLU));
    assert_eq!(ActivationType::from_name("swish"), Some(ActivationType::SiLU));
    assert_eq!(ActivationType::from_name("gelu"), Some(ActivationType::GeLU));
}

#[test]
fn given_activation_type_when_parsing_unknown_name_then_returns_none() {
    assert!(ActivationType::from_name("unknown_activation").is_none());
}

#[test]
fn given_activation_type_when_querying_name_then_round_trips() {
    // Given: each activation type
    for act in [
        ActivationType::ReLU,
        ActivationType::SiLU,
        ActivationType::GeLU,
        ActivationType::Tanh,
        ActivationType::Sigmoid,
    ] {
        // When: name → from_name
        let name = act.name();
        let parsed = ActivationType::from_name(name);
        // Then: round-trips
        assert_eq!(parsed, Some(act));
    }
}

#[test]
fn given_bitnet_family_when_querying_activation_then_relu2() {
    assert_eq!(ActivationType::for_family("bitnet"), ActivationType::ReLU2);
}

#[test]
fn given_llama_family_when_querying_activation_then_silu() {
    assert_eq!(ActivationType::for_family("llama"), ActivationType::SiLU);
    assert_eq!(ActivationType::for_family("llama3"), ActivationType::SiLU);
}

#[test]
fn given_gpt2_family_when_querying_activation_then_gelu() {
    assert_eq!(ActivationType::for_family("gpt2"), ActivationType::GeLU);
}

#[test]
fn given_vector_when_activate_inplace_then_modifies_in_place() {
    // Given: a vector of values
    let mut data = vec![1.0, -1.0, 0.0, 2.0];
    // When: activating in-place with ReLU
    activate_inplace(&mut data, ActivationType::ReLU);
    // Then: negatives become 0
    assert!(approx_eq(data[0], 1.0));
    assert!(approx_eq(data[1], 0.0));
    assert!(approx_eq(data[2], 0.0));
    assert!(approx_eq(data[3], 2.0));
}

#[test]
fn given_vector_when_activate_vec_then_returns_new_vec() {
    // Given: a vector of values
    let data = vec![1.0, -1.0, 0.0];
    // When: activate_vec with ReLU
    let result = activate_vec(&data, ActivationType::ReLU);
    // Then: new vector with activations applied
    assert_eq!(result.len(), 3);
    assert!(approx_eq(result[0], 1.0));
    assert!(approx_eq(result[1], 0.0));
}

// ═══════════════════════════════════════════════════════════════════
// Section 8 — Normalization Registry
// ═══════════════════════════════════════════════════════════════════

#[test]
fn given_norm_type_when_parsing_layernorm_then_returns_layer_norm() {
    assert_eq!(NormType::from_name("layer_norm"), Some(NormType::LayerNorm));
    assert_eq!(NormType::from_name("layernorm"), Some(NormType::LayerNorm));
    assert_eq!(NormType::from_name("ln"), Some(NormType::LayerNorm));
}

#[test]
fn given_norm_type_when_parsing_rmsnorm_then_returns_rms_norm() {
    assert_eq!(NormType::from_name("rms_norm"), Some(NormType::RmsNorm));
    assert_eq!(NormType::from_name("rmsnorm"), Some(NormType::RmsNorm));
}

#[test]
fn given_norm_type_when_parsing_unknown_then_returns_none() {
    assert!(NormType::from_name("weird_norm").is_none());
}

#[test]
fn given_bitnet_family_when_querying_norm_then_sub_norm() {
    assert_eq!(NormType::for_family("bitnet"), NormType::SubNorm);
}

#[test]
fn given_llama_family_when_querying_norm_then_rms_norm() {
    assert_eq!(NormType::for_family("llama"), NormType::RmsNorm);
    assert_eq!(NormType::for_family("mistral"), NormType::RmsNorm);
}

#[test]
fn given_gpt2_family_when_querying_norm_then_layer_norm() {
    assert_eq!(NormType::for_family("gpt2"), NormType::LayerNorm);
}

#[test]
fn given_uniform_data_when_layer_norm_then_all_bias() {
    // Given: uniform data [1.0, 1.0, 1.0, 1.0]
    let mut data = vec![1.0, 1.0, 1.0, 1.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0];
    // When: applying layer_norm
    layer_norm(&mut data, &weight, None, 1e-5);
    // Then: normalized to approximately 0
    for &v in &data {
        assert!(v.abs() < 1e-3);
    }
}

#[test]
fn given_varied_data_when_rms_norm_then_output_scaled() {
    // Given: data [2.0, 2.0, 2.0, 2.0]
    let mut data = vec![2.0, 2.0, 2.0, 2.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0];
    // When: applying rms_norm
    rms_norm(&mut data, &weight, 1e-5);
    // Then: all values are approximately 1.0 (since rms=2, data/rms=1)
    for &v in &data {
        assert!((v - 1.0).abs() < 0.01);
    }
}

#[test]
fn given_empty_data_when_layer_norm_then_no_panic() {
    let mut data: Vec<f32> = vec![];
    let weight: Vec<f32> = vec![];
    layer_norm(&mut data, &weight, None, 1e-5);
}

#[test]
fn given_empty_data_when_rms_norm_then_no_panic() {
    let mut data: Vec<f32> = vec![];
    let weight: Vec<f32> = vec![];
    rms_norm(&mut data, &weight, 1e-5);
}

#[test]
fn given_norm_config_default_when_checking_then_reasonable_defaults() {
    let config = NormConfig::default();
    assert_eq!(config.norm_type, NormType::LayerNorm);
    assert_eq!(config.hidden_size, 4096);
    assert!(config.affine);
}

// ═══════════════════════════════════════════════════════════════════
// Section 9 — Dispatch Fallback Chains
// ═══════════════════════════════════════════════════════════════════

#[test]
fn given_three_tier_chain_when_top_two_unavailable_then_falls_to_scalar() {
    // Given: CUDA (pri=100, unavail), AVX2 (pri=10, unavail), Scalar (pri=1, avail)
    let mut table = DispatchTable::new();
    table.register(KernelOp::Softmax, DispatchBackend::Cuda, 100, false);
    table.register(KernelOp::Softmax, DispatchBackend::Avx2, 10, false);
    table.register(KernelOp::Softmax, DispatchBackend::Scalar, 1, true);
    // When: resolving
    let result = table.resolve(KernelOp::Softmax);
    // Then: Scalar
    assert_eq!(result, Some(DispatchBackend::Scalar));
}

#[test]
fn given_gpu_chain_when_cuda_unavailable_metal_available_then_selects_metal() {
    // Given: CUDA (pri=100, unavail), Metal (pri=90, avail), Scalar (pri=1, avail)
    let mut table = DispatchTable::new();
    table.register(KernelOp::MatMul, DispatchBackend::Cuda, 100, false);
    table.register(KernelOp::MatMul, DispatchBackend::Metal, 90, true);
    table.register(KernelOp::MatMul, DispatchBackend::Scalar, 1, true);
    // When: resolving
    let result = table.resolve(KernelOp::MatMul);
    // Then: Metal
    assert_eq!(result, Some(DispatchBackend::Metal));
}

#[test]
fn given_all_unavailable_when_resolving_then_returns_none() {
    // Given: all backends unavailable
    let mut table = DispatchTable::new();
    table.register(KernelOp::MatMul, DispatchBackend::Cuda, 100, false);
    table.register(KernelOp::MatMul, DispatchBackend::Avx2, 10, false);
    // When: resolving
    let result = table.resolve(KernelOp::MatMul);
    // Then: None
    assert!(result.is_none());
}

#[test]
fn given_mixed_ops_when_resolving_independently_then_each_resolves_separately() {
    // Given: MatMul → AVX2, Softmax → Scalar
    let mut table = DispatchTable::new();
    table.register(KernelOp::MatMul, DispatchBackend::Avx2, 10, true);
    table.register(KernelOp::MatMul, DispatchBackend::Scalar, 1, true);
    table.register(KernelOp::Softmax, DispatchBackend::Scalar, 1, true);
    // When/Then
    assert_eq!(table.resolve(KernelOp::MatMul), Some(DispatchBackend::Avx2));
    assert_eq!(table.resolve(KernelOp::Softmax), Some(DispatchBackend::Scalar));
}

// ═══════════════════════════════════════════════════════════════════
// Section 10 — Device Features & SupportLevel
// ═══════════════════════════════════════════════════════════════════

#[test]
fn given_support_level_full_when_checking_supported_then_true() {
    let level = SupportLevel::Full(0.95);
    assert!(level.is_supported());
}

#[test]
fn given_support_level_partial_when_checking_supported_then_true() {
    let level = SupportLevel::Partial("slow path".to_string());
    assert!(level.is_supported());
}

#[test]
fn given_support_level_emulated_when_checking_supported_then_true() {
    let level = SupportLevel::Emulated;
    assert!(level.is_supported());
}

#[test]
fn given_support_level_unsupported_when_checking_supported_then_false() {
    let level = SupportLevel::Unsupported;
    assert!(!level.is_supported());
}

#[test]
fn given_support_level_full_when_querying_efficiency_then_returns_value() {
    let level = SupportLevel::Full(0.85);
    assert_eq!(level.efficiency(), Some(0.85));
}

#[test]
fn given_support_level_partial_when_querying_efficiency_then_returns_none() {
    let level = SupportLevel::Partial("reason".to_string());
    assert!(level.efficiency().is_none());
}

#[test]
fn given_device_class_all_when_counting_then_eight_classes() {
    assert_eq!(DeviceClass::ALL.len(), 8);
}

#[test]
fn given_device_class_when_displaying_then_non_empty_string() {
    for &class in DeviceClass::ALL {
        let s = format!("{}", class);
        assert!(!s.is_empty());
    }
}

#[test]
fn given_operation_category_all_when_counting_then_six_categories() {
    assert_eq!(OperationCategory::ALL.len(), 6);
}

#[test]
fn given_precision_support_all_when_counting_then_six_types() {
    assert_eq!(PrecisionSupport::ALL.len(), 6);
}

//! Edge-case tests for `bitnet-runtime-feature-flags-core`.
//!
//! Coverage:
//! - FeatureActivation default (all false)
//! - FeatureActivation individual fields → FeatureSet mapping
//! - CPU flag implies inference+kernels+tokenizers
//! - GPU flag implies inference+kernels
//! - CUDA flag implies gpu
//! - Multiple flags combined (additive)
//! - to_labels / feature_labels_from_activation
//! - feature_line_from_activation formatting
//! - Clone, Copy, Debug traits
//! - All 20 boolean fields individually

use bitnet_bdd_grid_core::BitnetFeature;
use bitnet_runtime_feature_flags_core::*;

// ---------------------------------------------------------------------------
// FeatureActivation — default
// ---------------------------------------------------------------------------

#[test]
fn default_activation_all_false() {
    let a = FeatureActivation::default();
    assert!(!a.cpu);
    assert!(!a.gpu);
    assert!(!a.cuda);
    assert!(!a.inference);
    assert!(!a.kernels);
    assert!(!a.tokenizers);
    assert!(!a.quantization);
    assert!(!a.cli);
    assert!(!a.server);
    assert!(!a.ffi);
    assert!(!a.python);
    assert!(!a.wasm);
    assert!(!a.crossval);
    assert!(!a.trace);
    assert!(!a.iq2s_ffi);
    assert!(!a.cpp_ffi);
    assert!(!a.fixtures);
    assert!(!a.reporting);
    assert!(!a.trend);
    assert!(!a.integration_tests);
}

#[test]
fn default_activation_produces_empty_feature_set() {
    let features = active_features_from_activation(FeatureActivation::default());
    assert!(features.is_empty());
}

// ---------------------------------------------------------------------------
// CPU flag — implies inference, kernels, tokenizers
// ---------------------------------------------------------------------------

#[test]
fn cpu_flag_implies_inference_kernels_tokenizers() {
    let a = FeatureActivation { cpu: true, ..Default::default() };
    let fs = active_features_from_activation(a);
    assert!(fs.contains(BitnetFeature::Cpu));
    assert!(fs.contains(BitnetFeature::Inference));
    assert!(fs.contains(BitnetFeature::Kernels));
    assert!(fs.contains(BitnetFeature::Tokenizers));
    // Should not imply GPU
    assert!(!fs.contains(BitnetFeature::Gpu));
    assert!(!fs.contains(BitnetFeature::Cuda));
}

// ---------------------------------------------------------------------------
// GPU flag — implies inference, kernels
// ---------------------------------------------------------------------------

#[test]
fn gpu_flag_implies_inference_kernels() {
    let a = FeatureActivation { gpu: true, ..Default::default() };
    let fs = active_features_from_activation(a);
    assert!(fs.contains(BitnetFeature::Gpu));
    assert!(fs.contains(BitnetFeature::Inference));
    assert!(fs.contains(BitnetFeature::Kernels));
    // Should not imply CPU or tokenizers
    assert!(!fs.contains(BitnetFeature::Cpu));
    assert!(!fs.contains(BitnetFeature::Tokenizers));
}

// ---------------------------------------------------------------------------
// CUDA flag — implies gpu
// ---------------------------------------------------------------------------

#[test]
fn cuda_flag_implies_gpu() {
    let a = FeatureActivation { cuda: true, ..Default::default() };
    let fs = active_features_from_activation(a);
    assert!(fs.contains(BitnetFeature::Cuda));
    assert!(fs.contains(BitnetFeature::Gpu));
    // CUDA alone does not add inference/kernels — only gpu does those
    // Actually, let's check: cuda adds Cuda+Gpu; gpu is not set to true in struct
    // So gpu implications (inference, kernels) are NOT triggered
    assert!(!fs.contains(BitnetFeature::Inference));
}

// ---------------------------------------------------------------------------
// Combined flags
// ---------------------------------------------------------------------------

#[test]
fn cpu_and_gpu_combined() {
    let a = FeatureActivation { cpu: true, gpu: true, ..Default::default() };
    let fs = active_features_from_activation(a);
    assert!(fs.contains(BitnetFeature::Cpu));
    assert!(fs.contains(BitnetFeature::Gpu));
    assert!(fs.contains(BitnetFeature::Inference));
    assert!(fs.contains(BitnetFeature::Kernels));
    assert!(fs.contains(BitnetFeature::Tokenizers));
}

#[test]
fn all_flags_true() {
    let a = FeatureActivation {
        cpu: true,
        gpu: true,
        cuda: true,
        inference: true,
        kernels: true,
        tokenizers: true,
        quantization: true,
        cli: true,
        server: true,
        ffi: true,
        python: true,
        wasm: true,
        crossval: true,
        trace: true,
        iq2s_ffi: true,
        cpp_ffi: true,
        fixtures: true,
        reporting: true,
        trend: true,
        integration_tests: true,
    };
    let fs = active_features_from_activation(a);
    // All features should be present
    assert!(fs.contains(BitnetFeature::Cpu));
    assert!(fs.contains(BitnetFeature::Gpu));
    assert!(fs.contains(BitnetFeature::Cuda));
    assert!(fs.contains(BitnetFeature::Inference));
    assert!(fs.contains(BitnetFeature::Kernels));
    assert!(fs.contains(BitnetFeature::Tokenizers));
    assert!(fs.contains(BitnetFeature::Quantization));
    assert!(fs.contains(BitnetFeature::Cli));
    assert!(fs.contains(BitnetFeature::Server));
    assert!(fs.contains(BitnetFeature::Ffi));
    assert!(fs.contains(BitnetFeature::Python));
    assert!(fs.contains(BitnetFeature::Wasm));
    assert!(fs.contains(BitnetFeature::CrossValidation));
    assert!(fs.contains(BitnetFeature::Trace));
    assert!(fs.contains(BitnetFeature::Iq2sFfi));
    assert!(fs.contains(BitnetFeature::CppFfi));
    assert!(fs.contains(BitnetFeature::Fixtures));
    assert!(fs.contains(BitnetFeature::Reporting));
    assert!(fs.contains(BitnetFeature::Trend));
    assert!(fs.contains(BitnetFeature::IntegrationTests));
}

// ---------------------------------------------------------------------------
// Individual flags (each produces the expected feature)
// ---------------------------------------------------------------------------

#[test]
fn inference_flag() {
    let a = FeatureActivation { inference: true, ..Default::default() };
    let fs = active_features_from_activation(a);
    assert!(fs.contains(BitnetFeature::Inference));
    assert_eq!(fs.labels().len(), 1);
}

#[test]
fn kernels_flag() {
    let a = FeatureActivation { kernels: true, ..Default::default() };
    let fs = active_features_from_activation(a);
    assert!(fs.contains(BitnetFeature::Kernels));
    assert_eq!(fs.labels().len(), 1);
}

#[test]
fn tokenizers_flag() {
    let a = FeatureActivation { tokenizers: true, ..Default::default() };
    let fs = active_features_from_activation(a);
    assert!(fs.contains(BitnetFeature::Tokenizers));
    assert_eq!(fs.labels().len(), 1);
}

#[test]
fn quantization_flag() {
    let a = FeatureActivation { quantization: true, ..Default::default() };
    let fs = active_features_from_activation(a);
    assert!(fs.contains(BitnetFeature::Quantization));
    assert_eq!(fs.labels().len(), 1);
}

#[test]
fn cli_flag() {
    let a = FeatureActivation { cli: true, ..Default::default() };
    let fs = active_features_from_activation(a);
    assert!(fs.contains(BitnetFeature::Cli));
    assert_eq!(fs.labels().len(), 1);
}

#[test]
fn server_flag() {
    let a = FeatureActivation { server: true, ..Default::default() };
    let fs = active_features_from_activation(a);
    assert!(fs.contains(BitnetFeature::Server));
    assert_eq!(fs.labels().len(), 1);
}

#[test]
fn ffi_flag() {
    let a = FeatureActivation { ffi: true, ..Default::default() };
    let fs = active_features_from_activation(a);
    assert!(fs.contains(BitnetFeature::Ffi));
    assert_eq!(fs.labels().len(), 1);
}

#[test]
fn python_flag() {
    let a = FeatureActivation { python: true, ..Default::default() };
    let fs = active_features_from_activation(a);
    assert!(fs.contains(BitnetFeature::Python));
    assert_eq!(fs.labels().len(), 1);
}

#[test]
fn wasm_flag() {
    let a = FeatureActivation { wasm: true, ..Default::default() };
    let fs = active_features_from_activation(a);
    assert!(fs.contains(BitnetFeature::Wasm));
    assert_eq!(fs.labels().len(), 1);
}

#[test]
fn crossval_flag() {
    let a = FeatureActivation { crossval: true, ..Default::default() };
    let fs = active_features_from_activation(a);
    assert!(fs.contains(BitnetFeature::CrossValidation));
    assert_eq!(fs.labels().len(), 1);
}

#[test]
fn trace_flag() {
    let a = FeatureActivation { trace: true, ..Default::default() };
    let fs = active_features_from_activation(a);
    assert!(fs.contains(BitnetFeature::Trace));
    assert_eq!(fs.labels().len(), 1);
}

#[test]
fn iq2s_ffi_flag() {
    let a = FeatureActivation { iq2s_ffi: true, ..Default::default() };
    let fs = active_features_from_activation(a);
    assert!(fs.contains(BitnetFeature::Iq2sFfi));
    assert_eq!(fs.labels().len(), 1);
}

#[test]
fn cpp_ffi_flag() {
    let a = FeatureActivation { cpp_ffi: true, ..Default::default() };
    let fs = active_features_from_activation(a);
    assert!(fs.contains(BitnetFeature::CppFfi));
    assert_eq!(fs.labels().len(), 1);
}

#[test]
fn fixtures_flag() {
    let a = FeatureActivation { fixtures: true, ..Default::default() };
    let fs = active_features_from_activation(a);
    assert!(fs.contains(BitnetFeature::Fixtures));
    assert_eq!(fs.labels().len(), 1);
}

#[test]
fn reporting_flag() {
    let a = FeatureActivation { reporting: true, ..Default::default() };
    let fs = active_features_from_activation(a);
    assert!(fs.contains(BitnetFeature::Reporting));
    assert_eq!(fs.labels().len(), 1);
}

#[test]
fn trend_flag() {
    let a = FeatureActivation { trend: true, ..Default::default() };
    let fs = active_features_from_activation(a);
    assert!(fs.contains(BitnetFeature::Trend));
    assert_eq!(fs.labels().len(), 1);
}

#[test]
fn integration_tests_flag() {
    let a = FeatureActivation { integration_tests: true, ..Default::default() };
    let fs = active_features_from_activation(a);
    assert!(fs.contains(BitnetFeature::IntegrationTests));
    assert_eq!(fs.labels().len(), 1);
}

// ---------------------------------------------------------------------------
// to_labels
// ---------------------------------------------------------------------------

#[test]
fn to_labels_empty() {
    let a = FeatureActivation::default();
    assert!(a.to_labels().is_empty());
}

#[test]
fn to_labels_cpu_includes_implied() {
    let a = FeatureActivation { cpu: true, ..Default::default() };
    let labels = a.to_labels();
    assert!(labels.contains(&"cpu".to_string()));
    assert!(labels.contains(&"inference".to_string()));
    assert!(labels.contains(&"kernels".to_string()));
    assert!(labels.contains(&"tokenizers".to_string()));
}

// ---------------------------------------------------------------------------
// feature_labels_from_activation
// ---------------------------------------------------------------------------

#[test]
fn feature_labels_from_activation_matches_to_labels() {
    let a = FeatureActivation { cpu: true, trace: true, ..Default::default() };
    let labels1 = a.to_labels();
    let labels2 = feature_labels_from_activation(a);
    assert_eq!(labels1, labels2);
}

// ---------------------------------------------------------------------------
// feature_line_from_activation
// ---------------------------------------------------------------------------

#[test]
fn feature_line_none_when_empty() {
    let a = FeatureActivation::default();
    assert_eq!(feature_line_from_activation(a), "features: none");
}

#[test]
fn feature_line_shows_features() {
    let a = FeatureActivation { cpu: true, ..Default::default() };
    let line = feature_line_from_activation(a);
    assert!(line.starts_with("features: "));
    assert!(line.contains("cpu"));
    assert!(line.contains("inference"));
}

#[test]
fn feature_line_single_feature() {
    let a = FeatureActivation { trace: true, ..Default::default() };
    let line = feature_line_from_activation(a);
    assert_eq!(line, "features: trace");
}

// ---------------------------------------------------------------------------
// Traits — Clone, Copy, Debug, Default
// ---------------------------------------------------------------------------

#[test]
fn feature_activation_is_copy() {
    let a = FeatureActivation { cpu: true, ..Default::default() };
    let b = a; // Copy
    assert!(b.cpu);
    assert!(a.cpu); // a still valid
}

#[test]
fn feature_activation_clone() {
    let a = FeatureActivation { gpu: true, ..Default::default() };
    let b = a.clone();
    assert!(b.gpu);
}

#[test]
fn feature_activation_debug() {
    let a = FeatureActivation { cpu: true, ..Default::default() };
    let dbg = format!("{:?}", a);
    assert!(dbg.contains("cpu: true"));
}

// ---------------------------------------------------------------------------
// Edge: implied features are not double-counted
// ---------------------------------------------------------------------------

#[test]
fn cpu_implies_no_duplicates_in_labels() {
    let a = FeatureActivation { cpu: true, inference: true, kernels: true, ..Default::default() };
    let labels = a.to_labels();
    // Even though cpu implies inference+kernels, and they're also explicitly set,
    // the FeatureSet deduplicates
    let unique: std::collections::HashSet<_> = labels.iter().collect();
    assert_eq!(labels.len(), unique.len(), "labels should have no duplicates");
}

#[test]
fn cuda_alone_adds_cuda_and_gpu_only() {
    let a = FeatureActivation { cuda: true, ..Default::default() };
    let fs = active_features_from_activation(a);
    let labels = fs.labels();
    assert_eq!(labels, vec!["gpu", "cuda"]);
}

// ---------------------------------------------------------------------------
// Edge: multiple GPU backends
// ---------------------------------------------------------------------------

#[test]
fn gpu_and_cuda_combined() {
    let a = FeatureActivation { gpu: true, cuda: true, ..Default::default() };
    let fs = active_features_from_activation(a);
    assert!(fs.contains(BitnetFeature::Gpu));
    assert!(fs.contains(BitnetFeature::Cuda));
    assert!(fs.contains(BitnetFeature::Inference));
    assert!(fs.contains(BitnetFeature::Kernels));
}

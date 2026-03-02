//! Wave-29 snapshot tests for bitnet-kernels API surface stability.
//!
//! Pins KernelProvider trait shape, SimdLevel variants, KernelManager defaults,
//! OpenCL pipeline stage ordering, and attention computation output.

use bitnet_kernels::KernelManager;
use bitnet_kernels::opencl_attention::{AttentionConfig, scaled_dot_product_attention_ref};
use bitnet_kernels::opencl_pipeline::{InferencePipeline, PipelineConfig};
use bitnet_kernels::simd_diagnostics::SimdLevel;

// ── KernelProvider trait method list via provider info ──────────────

#[test]
fn kernel_provider_trait_methods_via_manager() {
    let mgr = KernelManager::new();
    let providers = mgr.list_available_providers();
    let best = mgr.select_best().expect("at least one provider");
    let info = format!(
        "provider_count={}\navailable={:?}\nselected_name={}\nis_available={}",
        providers.len(),
        providers,
        best.name(),
        best.is_available(),
    );
    insta::assert_snapshot!(info);
}

// ── SimdLevel variants on x86_64 ──────────────────────────────────

#[test]
fn simd_level_variants_debug() {
    let variants = [
        SimdLevel::Scalar,
        SimdLevel::Sse2,
        SimdLevel::Sse42,
        SimdLevel::Neon,
        SimdLevel::Avx,
        SimdLevel::Avx2,
        SimdLevel::Avx512,
    ];
    let debug: Vec<String> = variants.iter().map(|v| format!("{v:?}")).collect();
    let display: Vec<String> = variants.iter().map(|v| format!("{v}")).collect();
    insta::assert_debug_snapshot!("simd_level_variants", (&debug, &display));
}

// ── Default KernelManager configuration ───────────────────────────

#[test]
fn kernel_manager_default_configuration() {
    let mgr = KernelManager::default();
    let providers = mgr.list_available_providers();
    let has_fallback = providers.iter().any(|p| p.contains("fallback") || p.contains("cpu"));
    let selected = mgr.select_best().map(|p| p.name()).unwrap_or("none");
    let info = format!(
        "provider_count={}\nhas_cpu_fallback={}\nselected={}",
        providers.len(),
        has_fallback,
        selected,
    );
    insta::assert_snapshot!(info);
}

// ── OpenCL pipeline stage ordering ────────────────────────────────

#[test]
fn opencl_pipeline_stage_ordering() {
    let config = PipelineConfig {
        num_layers: 2,
        hidden_dim: 32,
        num_heads: 4,
        head_dim: 8,
        intermediate_dim: 64,
        vocab_size: 64,
        max_seq_len: 128,
        use_gpu: false,
        fallback_to_cpu: true,
    };
    let pipeline = InferencePipeline::new(config).expect("valid config");
    let stages = pipeline.stage_order();
    let stage_names: Vec<String> = stages.iter().map(|s| format!("{s}")).collect();
    insta::assert_debug_snapshot!("pipeline_stage_order", stage_names);
}

// ── Attention computation output (4 heads, dim=8, seq=4) ──────────

#[test]
fn attention_computation_known_input() {
    let num_heads: usize = 4;
    let head_dim: usize = 8;
    let seq_len: usize = 4;

    let config =
        AttentionConfig::new(num_heads, head_dim, seq_len, true).expect("valid attention config");

    // Deterministic input: sequential values scaled down
    let total_q = seq_len * head_dim;
    let q: Vec<f32> = (0..total_q).map(|i| (i as f32) * 0.1).collect();
    let k: Vec<f32> = (0..total_q).map(|i| ((total_q - i) as f32) * 0.1).collect();
    let v: Vec<f32> = (0..total_q).map(|i| (i as f32) * 0.05).collect();
    let mut output = vec![0.0f32; total_q];

    // Run single-head attention for head 0
    scaled_dot_product_attention_ref(
        &q[..seq_len * head_dim],
        &k[..seq_len * head_dim],
        &v[..seq_len * head_dim],
        &mut output[..seq_len * head_dim],
        seq_len,
        seq_len,
        head_dim,
        config.scale,
        config.causal,
    );

    // Round to 4 decimal places for stability
    let rounded: Vec<String> = output.iter().map(|v| format!("{v:.4}")).collect();
    insta::assert_debug_snapshot!("attention_output_4h_d8_s4", rounded);
}

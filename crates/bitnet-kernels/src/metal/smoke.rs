use std::fmt;

use crate::cpu::linear::{LinearConfig, linear_cpu};
use crate::cpu::quantized_matmul::{i2s_matmul_f32, pack_i2s};

pub const MACHINE_ID: &str = "apple-m4-mac-mini";
pub const ARTIFACT_KIND: &str = "smoke";
pub const PARITY_ARTIFACT_KIND: &str = "parity";
pub const PHASE_CONTRIBUTION_ARTIFACT_KIND: &str = "phase_contribution";
pub const SLM_APPLE_M4_METAL_PHASE_ARTIFACT_KIND: &str = "slm_apple_m4_metal_phase";
pub const SUBGRAPH_ARTIFACT_KIND: &str = "subgraph";
pub const REQUESTED_BACKEND: &str = "apple-m4-metal";
pub const SELECTED_BACKEND: &str = "apple-m4-metal";
pub const REFERENCE_BACKEND: &str = "apple-m4-cpu-neon";
pub const RUNTIME_API: &str = "metal";
pub const TINY_METAL_ADD_SMOKE_KERNEL_ID: &str = "tiny_metal_add_smoke";
pub const TINY_METAL_ADD_PARITY_KERNEL_ID: &str = "tiny_metal_add_parity";
pub const I2S_METAL_PARITY_KERNEL_ID: &str = "tiny_metal_i2s_parity";
pub const I2S_METAL_PREFILL_CONTRIBUTION_KERNEL_ID: &str = "tiny_metal_i2s_prefill_contribution";
pub const I2S_METAL_PROJECTION_RESIDUAL_KERNEL_ID: &str = "tiny_metal_i2s_projection_residual";
pub const I2S_PROJECTION_RESIDUAL_GRAPH_ID: &str = "tiny_i2s_projection_residual_subgraph";
pub const DENSE_METAL_PREFILL_LINEAR_KERNEL_ID: &str = "tiny_metal_dense_prefill_linear_projection";
pub const DENSE_METAL_PREFILL_QKV_KERNEL_ID: &str = "tiny_metal_dense_prefill_qkv_projection";
pub const DENSE_METAL_PREFILL_ATTENTION_SCORES_KERNEL_ID: &str =
    "tiny_metal_dense_prefill_attention_scores";
pub const I2S_KERNEL_FAMILY: &str = "i2_s";
pub const DENSE_KERNEL_FAMILY: &str = "dense_f32";
pub const DENSE_MODEL_FAMILY: &str = "qwen2.5";
pub const I2S_EXECUTION_PHASE: &str = "parity";
pub const I2S_PREFILL_EXECUTION_PHASE: &str = "prefill";
pub const I2S_PREFILL_PHASE_SCOPE: &str = "prefill_projection_fixture";
pub const I2S_PREFILL_KV_CACHE_BEHAVIOR: &str = "not_exercised";
pub const DENSE_PREFILL_LINEAR_EXECUTION_PHASE: &str = "prefill_linear_projection";
pub const DENSE_PREFILL_LINEAR_PHASE_SCOPE: &str =
    "qwen2_5_dense_prefill_linear_projection_fixture";
pub const DENSE_PREFILL_LINEAR_KV_CACHE_BEHAVIOR: &str = "not_exercised";
pub const DENSE_PREFILL_QKV_EXECUTION_PHASE: &str = "prefill_qkv_projection";
pub const DENSE_PREFILL_QKV_PHASE_SCOPE: &str = "qwen2_5_dense_prefill_qkv_projection_fixture";
pub const DENSE_PREFILL_QKV_KV_CACHE_BEHAVIOR: &str = "not_exercised";
pub const DENSE_PREFILL_ATTENTION_SCORES_EXECUTION_PHASE: &str = "prefill_attention_scores";
pub const DENSE_PREFILL_ATTENTION_SCORES_PHASE_SCOPE: &str =
    "qwen2_5_dense_prefill_attention_scores_fixture";
pub const DENSE_PREFILL_ATTENTION_SCORES_KV_CACHE_BEHAVIOR: &str =
    "read_only_fixture_keys_no_cache_write";
pub const DENSE_PREFILL_ATTENTION_SCORES_LAYOUT_SOURCE: &str = "fixture_dense_f32_qk_rope_applied";
pub const DENSE_PREFILL_LINEAR_REST_OF_PIPELINE_BACKEND: &str = "apple-m4-cpu-neon";
pub const DENSE_PREFILL_LINEAR_TIMING_SCOPE: &str =
    "single_live_phase_dispatch_readback_vs_cpu_reference_fixture";
pub const DENSE_PREFILL_QKV_TIMING_SCOPE: &str =
    "single_live_qkv_phase_dispatch_readback_vs_cpu_reference_fixture";
pub const DENSE_PREFILL_ATTENTION_SCORES_TIMING_SCOPE: &str =
    "single_live_attention_score_phase_dispatch_readback_vs_cpu_reference_fixture";
pub const I2S_PROJECTION_RESIDUAL_EXECUTION_PHASE: &str = "parity";
pub const I2S_PROJECTION_RESIDUAL_PHASE_SCOPE: &str = "projection_residual_subgraph";
pub const I2S_PROJECTION_RESIDUAL_OPS: [&str; 2] = ["packed_i2_s_matmul", "residual_add"];
pub const I2S_LAYOUT_SOURCE: &str = "fixture_packed_i2_s";
pub const I2S_TRANSPORT_LAYOUT: &str = "u32_le_words_from_i2s_bytes";
pub const DENSE_LAYOUT_SOURCE: &str = "fixture_dense_f32_row_major";
pub const DENSE_TRANSPORT_LAYOUT: &str = "row_major_f32";
pub const I2S_PARITY_M: usize = 1;
pub const I2S_PARITY_N: usize = 4;
pub const I2S_PARITY_K: usize = 32;
pub const I2S_PARITY_BLOCK_SIZE: usize = 32;
pub const I2S_PREFILL_TOKENS: usize = 2;
pub const DENSE_PREFILL_TOKENS: usize = 2;
pub const DENSE_PREFILL_ATTENTION_SCORE_TOKENS: usize = 4;
pub const DENSE_PREFILL_IN_FEATURES: usize = 8;
pub const DENSE_PREFILL_OUT_FEATURES: usize = 6;
pub const QWEN2_5_0_5B_HIDDEN_SIZE: usize = 896;
pub const QWEN2_5_0_5B_ATTENTION_HEADS: usize = 14;
pub const QWEN2_5_0_5B_KV_HEADS: usize = 2;
pub const QWEN2_5_0_5B_HEAD_DIM: usize = QWEN2_5_0_5B_HIDDEN_SIZE / QWEN2_5_0_5B_ATTENTION_HEADS;
pub const QWEN2_5_0_5B_Q_DIM: usize = QWEN2_5_0_5B_HIDDEN_SIZE;
pub const QWEN2_5_0_5B_KV_DIM: usize = QWEN2_5_0_5B_KV_HEADS * QWEN2_5_0_5B_HEAD_DIM;
pub const SMOKE_ELEMENT_COUNT: usize = 64;
pub const SMOKE_WORKGROUP_SIZE: u32 = 64;

#[derive(Debug, Clone, PartialEq)]
pub struct TinyMetalAddSmokeReceipt {
    pub machine_id: &'static str,
    pub artifact_kind: &'static str,
    pub requested_backend: &'static str,
    pub selected_backend: &'static str,
    pub runtime_api: &'static str,
    pub kernel_id: &'static str,
    pub fallback_used: bool,
    pub result: &'static str,
    pub artifact_path: String,
    pub element_count: usize,
    pub max_abs_error: f32,
    pub mean_abs_error: f32,
}

impl TinyMetalAddSmokeReceipt {
    pub fn passed(
        artifact_path: impl Into<String>,
        element_count: usize,
        comparison: SmokeComparison,
    ) -> Self {
        Self {
            machine_id: MACHINE_ID,
            artifact_kind: ARTIFACT_KIND,
            requested_backend: REQUESTED_BACKEND,
            selected_backend: SELECTED_BACKEND,
            runtime_api: RUNTIME_API,
            kernel_id: TINY_METAL_ADD_SMOKE_KERNEL_ID,
            fallback_used: false,
            result: "pass",
            artifact_path: artifact_path.into(),
            element_count,
            max_abs_error: comparison.max_abs_error,
            mean_abs_error: comparison.mean_abs_error,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TinyMetalAddParityReceipt {
    pub machine_id: &'static str,
    pub artifact_kind: &'static str,
    pub requested_backend: &'static str,
    pub selected_backend: &'static str,
    pub runtime_api: &'static str,
    pub reference_backend: &'static str,
    pub target_backend: &'static str,
    pub kernel_id: &'static str,
    pub fallback_used: bool,
    pub result: &'static str,
    pub artifact_path: String,
    pub element_count: usize,
    pub max_abs_error: f32,
    pub mean_abs_error: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct I2sMetalParityReceipt {
    pub machine_id: &'static str,
    pub artifact_kind: &'static str,
    pub requested_backend: &'static str,
    pub selected_backend: &'static str,
    pub runtime_api: &'static str,
    pub reference_backend: &'static str,
    pub target_backend: &'static str,
    pub kernel_id: &'static str,
    pub kernel_family: &'static str,
    pub execution_phase: &'static str,
    pub layout_source: &'static str,
    pub transport_layout: &'static str,
    pub fallback_used: bool,
    pub result: &'static str,
    pub artifact_path: String,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub block_size: usize,
    pub max_abs_error: f32,
    pub mean_abs_error: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct I2sMetalPrefillContributionReceipt {
    pub machine_id: &'static str,
    pub artifact_kind: &'static str,
    pub requested_backend: &'static str,
    pub selected_backend: &'static str,
    pub runtime_api: &'static str,
    pub reference_backend: &'static str,
    pub target_backend: &'static str,
    pub kernel_id: &'static str,
    pub kernel_family: &'static str,
    pub execution_phase: &'static str,
    pub phase_scope: &'static str,
    pub layout_source: &'static str,
    pub transport_layout: &'static str,
    pub kv_cache_behavior: &'static str,
    pub fallback_used: bool,
    pub result: &'static str,
    pub artifact_path: String,
    pub prefill_tokens: usize,
    pub n: usize,
    pub k: usize,
    pub block_size: usize,
    pub max_abs_error: f32,
    pub mean_abs_error: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct I2sMetalProjectionResidualReceipt {
    pub machine_id: &'static str,
    pub artifact_kind: &'static str,
    pub requested_backend: &'static str,
    pub selected_backend: &'static str,
    pub runtime_api: &'static str,
    pub reference_backend: &'static str,
    pub target_backend: &'static str,
    pub graph_id: &'static str,
    pub kernel_id: &'static str,
    pub kernel_family: &'static str,
    pub execution_phase: &'static str,
    pub phase_scope: &'static str,
    pub layout_source: &'static str,
    pub transport_layout: &'static str,
    pub fallback_used: bool,
    pub result: &'static str,
    pub artifact_path: String,
    pub tokens: usize,
    pub n: usize,
    pub k: usize,
    pub block_size: usize,
    pub max_abs_error: f32,
    pub mean_abs_error: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DenseMetalPrefillLinearReceipt {
    pub machine_id: &'static str,
    pub artifact_kind: &'static str,
    pub requested_backend: &'static str,
    pub selected_backend: &'static str,
    pub runtime_api: &'static str,
    pub reference_backend: &'static str,
    pub target_backend: &'static str,
    pub rest_of_pipeline_backend: &'static str,
    pub kernel_id: &'static str,
    pub model_family: &'static str,
    pub kernel_family: &'static str,
    pub execution_phase: &'static str,
    pub phase_scope: &'static str,
    pub layout_source: &'static str,
    pub transport_layout: &'static str,
    pub kv_cache_behavior: &'static str,
    pub fallback_used: bool,
    pub result: &'static str,
    pub artifact_path: String,
    pub prefill_tokens: usize,
    pub in_features: usize,
    pub out_features: usize,
    pub max_abs_error: f32,
    pub mean_abs_error: f32,
    pub cpu_reference_token_id: usize,
    pub metal_phase_token_id: usize,
    pub timing: DenseMetalPrefillLinearTiming,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DenseMetalPrefillQkvReceipt {
    pub machine_id: &'static str,
    pub artifact_kind: &'static str,
    pub requested_backend: &'static str,
    pub selected_backend: &'static str,
    pub runtime_api: &'static str,
    pub reference_backend: &'static str,
    pub target_backend: &'static str,
    pub rest_of_pipeline_backend: &'static str,
    pub kernel_id: &'static str,
    pub model_family: &'static str,
    pub kernel_family: &'static str,
    pub execution_phase: &'static str,
    pub phase_scope: &'static str,
    pub layout_source: &'static str,
    pub transport_layout: &'static str,
    pub kv_cache_behavior: &'static str,
    pub fallback_used: bool,
    pub result: &'static str,
    pub artifact_path: String,
    pub prefill_tokens: usize,
    pub hidden_size: usize,
    pub attention_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
    pub q_max_abs_error: f32,
    pub q_mean_abs_error: f32,
    pub k_max_abs_error: f32,
    pub k_mean_abs_error: f32,
    pub v_max_abs_error: f32,
    pub v_mean_abs_error: f32,
    pub max_abs_error: f32,
    pub mean_abs_error: f32,
    pub q_argmax_index: usize,
    pub k_argmax_index: usize,
    pub v_argmax_index: usize,
    pub timing: DenseMetalPrefillQkvTiming,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DenseMetalPrefillAttentionScoresReceipt {
    pub machine_id: &'static str,
    pub artifact_kind: &'static str,
    pub requested_backend: &'static str,
    pub selected_backend: &'static str,
    pub runtime_api: &'static str,
    pub reference_backend: &'static str,
    pub target_backend: &'static str,
    pub rest_of_pipeline_backend: &'static str,
    pub kernel_id: &'static str,
    pub model_family: &'static str,
    pub kernel_family: &'static str,
    pub execution_phase: &'static str,
    pub phase_scope: &'static str,
    pub layout_source: &'static str,
    pub transport_layout: &'static str,
    pub kv_cache_behavior: &'static str,
    pub fallback_used: bool,
    pub result: &'static str,
    pub artifact_path: String,
    pub prefill_tokens: usize,
    pub attention_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub score_count: usize,
    pub scale: f32,
    pub scores_match_cpu_reference: bool,
    pub score_shape_matches_cpu_reference: bool,
    pub head_mapping_matches_cpu_reference: bool,
    pub max_abs_error: f32,
    pub mean_abs_error: f32,
    pub score_argmax_index: usize,
    pub timing: DenseMetalPrefillAttentionScoresTiming,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DenseMetalPrefillLinearTiming {
    pub cpu_reference_ms: f64,
    pub metal_phase_ms: f64,
    pub timing_delta_ms: f64,
    pub timing_scope: &'static str,
    pub speedup_claim: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DenseMetalPrefillQkvTiming {
    pub cpu_reference_ms: f64,
    pub metal_phase_ms: f64,
    pub metal_q_ms: f64,
    pub metal_k_ms: f64,
    pub metal_v_ms: f64,
    pub dispatch_readback_ms: f64,
    pub timing_delta_ms: f64,
    pub timing_scope: &'static str,
    pub speedup_claim: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DenseMetalPrefillAttentionScoresTiming {
    pub cpu_reference_ms: f64,
    pub metal_phase_ms: f64,
    pub dispatch_readback_ms: f64,
    pub timing_delta_ms: f64,
    pub timing_scope: &'static str,
    pub speedup_claim: bool,
}

impl DenseMetalPrefillLinearTiming {
    pub fn measured(cpu_reference_ms: f64, metal_phase_ms: f64) -> Self {
        Self {
            cpu_reference_ms,
            metal_phase_ms,
            timing_delta_ms: metal_phase_ms - cpu_reference_ms,
            timing_scope: DENSE_PREFILL_LINEAR_TIMING_SCOPE,
            speedup_claim: false,
        }
    }
}

impl DenseMetalPrefillQkvTiming {
    pub fn measured(cpu_reference_ms: f64, metal_phase_ms: f64) -> Self {
        Self {
            cpu_reference_ms,
            metal_phase_ms,
            metal_q_ms: 0.0,
            metal_k_ms: 0.0,
            metal_v_ms: 0.0,
            dispatch_readback_ms: metal_phase_ms,
            timing_delta_ms: metal_phase_ms - cpu_reference_ms,
            timing_scope: DENSE_PREFILL_QKV_TIMING_SCOPE,
            speedup_claim: false,
        }
    }
}

impl DenseMetalPrefillAttentionScoresTiming {
    pub fn measured(cpu_reference_ms: f64, metal_phase_ms: f64) -> Self {
        Self {
            cpu_reference_ms,
            metal_phase_ms,
            dispatch_readback_ms: metal_phase_ms,
            timing_delta_ms: metal_phase_ms - cpu_reference_ms,
            timing_scope: DENSE_PREFILL_ATTENTION_SCORES_TIMING_SCOPE,
            speedup_claim: false,
        }
    }
}

impl TinyMetalAddParityReceipt {
    pub fn passed(
        artifact_path: impl Into<String>,
        element_count: usize,
        comparison: SmokeComparison,
    ) -> Self {
        Self {
            machine_id: MACHINE_ID,
            artifact_kind: PARITY_ARTIFACT_KIND,
            requested_backend: REQUESTED_BACKEND,
            selected_backend: SELECTED_BACKEND,
            runtime_api: RUNTIME_API,
            reference_backend: REFERENCE_BACKEND,
            target_backend: SELECTED_BACKEND,
            kernel_id: TINY_METAL_ADD_PARITY_KERNEL_ID,
            fallback_used: false,
            result: "pass",
            artifact_path: artifact_path.into(),
            element_count,
            max_abs_error: comparison.max_abs_error,
            mean_abs_error: comparison.mean_abs_error,
        }
    }
}

impl I2sMetalParityReceipt {
    pub fn passed(artifact_path: impl Into<String>, comparison: SmokeComparison) -> Self {
        Self {
            machine_id: MACHINE_ID,
            artifact_kind: PARITY_ARTIFACT_KIND,
            requested_backend: REQUESTED_BACKEND,
            selected_backend: SELECTED_BACKEND,
            runtime_api: RUNTIME_API,
            reference_backend: REFERENCE_BACKEND,
            target_backend: SELECTED_BACKEND,
            kernel_id: I2S_METAL_PARITY_KERNEL_ID,
            kernel_family: I2S_KERNEL_FAMILY,
            execution_phase: I2S_EXECUTION_PHASE,
            layout_source: I2S_LAYOUT_SOURCE,
            transport_layout: I2S_TRANSPORT_LAYOUT,
            fallback_used: false,
            result: "pass",
            artifact_path: artifact_path.into(),
            m: I2S_PARITY_M,
            n: I2S_PARITY_N,
            k: I2S_PARITY_K,
            block_size: I2S_PARITY_BLOCK_SIZE,
            max_abs_error: comparison.max_abs_error,
            mean_abs_error: comparison.mean_abs_error,
        }
    }
}

impl I2sMetalPrefillContributionReceipt {
    pub fn passed(artifact_path: impl Into<String>, comparison: SmokeComparison) -> Self {
        Self {
            machine_id: MACHINE_ID,
            artifact_kind: PHASE_CONTRIBUTION_ARTIFACT_KIND,
            requested_backend: REQUESTED_BACKEND,
            selected_backend: SELECTED_BACKEND,
            runtime_api: RUNTIME_API,
            reference_backend: REFERENCE_BACKEND,
            target_backend: SELECTED_BACKEND,
            kernel_id: I2S_METAL_PREFILL_CONTRIBUTION_KERNEL_ID,
            kernel_family: I2S_KERNEL_FAMILY,
            execution_phase: I2S_PREFILL_EXECUTION_PHASE,
            phase_scope: I2S_PREFILL_PHASE_SCOPE,
            layout_source: I2S_LAYOUT_SOURCE,
            transport_layout: I2S_TRANSPORT_LAYOUT,
            kv_cache_behavior: I2S_PREFILL_KV_CACHE_BEHAVIOR,
            fallback_used: false,
            result: "pass",
            artifact_path: artifact_path.into(),
            prefill_tokens: I2S_PREFILL_TOKENS,
            n: I2S_PARITY_N,
            k: I2S_PARITY_K,
            block_size: I2S_PARITY_BLOCK_SIZE,
            max_abs_error: comparison.max_abs_error,
            mean_abs_error: comparison.mean_abs_error,
        }
    }
}

impl I2sMetalProjectionResidualReceipt {
    pub fn passed(artifact_path: impl Into<String>, comparison: SmokeComparison) -> Self {
        Self {
            machine_id: MACHINE_ID,
            artifact_kind: SUBGRAPH_ARTIFACT_KIND,
            requested_backend: REQUESTED_BACKEND,
            selected_backend: SELECTED_BACKEND,
            runtime_api: RUNTIME_API,
            reference_backend: REFERENCE_BACKEND,
            target_backend: SELECTED_BACKEND,
            graph_id: I2S_PROJECTION_RESIDUAL_GRAPH_ID,
            kernel_id: I2S_METAL_PROJECTION_RESIDUAL_KERNEL_ID,
            kernel_family: I2S_KERNEL_FAMILY,
            execution_phase: I2S_PROJECTION_RESIDUAL_EXECUTION_PHASE,
            phase_scope: I2S_PROJECTION_RESIDUAL_PHASE_SCOPE,
            layout_source: I2S_LAYOUT_SOURCE,
            transport_layout: I2S_TRANSPORT_LAYOUT,
            fallback_used: false,
            result: "pass",
            artifact_path: artifact_path.into(),
            tokens: I2S_PREFILL_TOKENS,
            n: I2S_PARITY_N,
            k: I2S_PARITY_K,
            block_size: I2S_PARITY_BLOCK_SIZE,
            max_abs_error: comparison.max_abs_error,
            mean_abs_error: comparison.mean_abs_error,
        }
    }
}

impl DenseMetalPrefillLinearReceipt {
    pub fn passed(
        artifact_path: impl Into<String>,
        comparison: SmokeComparison,
        cpu_reference_token_id: usize,
        metal_phase_token_id: usize,
        timing: DenseMetalPrefillLinearTiming,
    ) -> Self {
        Self {
            machine_id: MACHINE_ID,
            artifact_kind: PHASE_CONTRIBUTION_ARTIFACT_KIND,
            requested_backend: REQUESTED_BACKEND,
            selected_backend: SELECTED_BACKEND,
            runtime_api: RUNTIME_API,
            reference_backend: REFERENCE_BACKEND,
            target_backend: SELECTED_BACKEND,
            rest_of_pipeline_backend: DENSE_PREFILL_LINEAR_REST_OF_PIPELINE_BACKEND,
            kernel_id: DENSE_METAL_PREFILL_LINEAR_KERNEL_ID,
            model_family: DENSE_MODEL_FAMILY,
            kernel_family: DENSE_KERNEL_FAMILY,
            execution_phase: DENSE_PREFILL_LINEAR_EXECUTION_PHASE,
            phase_scope: DENSE_PREFILL_LINEAR_PHASE_SCOPE,
            layout_source: DENSE_LAYOUT_SOURCE,
            transport_layout: DENSE_TRANSPORT_LAYOUT,
            kv_cache_behavior: DENSE_PREFILL_LINEAR_KV_CACHE_BEHAVIOR,
            fallback_used: false,
            result: "pass",
            artifact_path: artifact_path.into(),
            prefill_tokens: DENSE_PREFILL_TOKENS,
            in_features: DENSE_PREFILL_IN_FEATURES,
            out_features: DENSE_PREFILL_OUT_FEATURES,
            max_abs_error: comparison.max_abs_error,
            mean_abs_error: comparison.mean_abs_error,
            cpu_reference_token_id,
            metal_phase_token_id,
            timing,
        }
    }
}

impl DenseMetalPrefillQkvReceipt {
    pub fn passed(
        artifact_path: impl Into<String>,
        q_comparison: SmokeComparison,
        k_comparison: SmokeComparison,
        v_comparison: SmokeComparison,
        fixture: &DenseMetalPrefillQkvFixture,
        timing: DenseMetalPrefillQkvTiming,
    ) -> Self {
        let max_abs_error = q_comparison
            .max_abs_error
            .max(k_comparison.max_abs_error)
            .max(v_comparison.max_abs_error);
        let mean_abs_error = (q_comparison.mean_abs_error
            + k_comparison.mean_abs_error
            + v_comparison.mean_abs_error)
            / 3.0;
        Self {
            machine_id: MACHINE_ID,
            artifact_kind: PHASE_CONTRIBUTION_ARTIFACT_KIND,
            requested_backend: REQUESTED_BACKEND,
            selected_backend: SELECTED_BACKEND,
            runtime_api: RUNTIME_API,
            reference_backend: REFERENCE_BACKEND,
            target_backend: SELECTED_BACKEND,
            rest_of_pipeline_backend: DENSE_PREFILL_LINEAR_REST_OF_PIPELINE_BACKEND,
            kernel_id: DENSE_METAL_PREFILL_QKV_KERNEL_ID,
            model_family: DENSE_MODEL_FAMILY,
            kernel_family: DENSE_KERNEL_FAMILY,
            execution_phase: DENSE_PREFILL_QKV_EXECUTION_PHASE,
            phase_scope: DENSE_PREFILL_QKV_PHASE_SCOPE,
            layout_source: DENSE_LAYOUT_SOURCE,
            transport_layout: DENSE_TRANSPORT_LAYOUT,
            kv_cache_behavior: DENSE_PREFILL_QKV_KV_CACHE_BEHAVIOR,
            fallback_used: false,
            result: "pass",
            artifact_path: artifact_path.into(),
            prefill_tokens: fixture.prefill_tokens,
            hidden_size: fixture.hidden_size,
            attention_heads: fixture.attention_heads,
            kv_heads: fixture.kv_heads,
            head_dim: fixture.head_dim,
            q_dim: fixture.q_dim,
            kv_dim: fixture.kv_dim,
            q_max_abs_error: q_comparison.max_abs_error,
            q_mean_abs_error: q_comparison.mean_abs_error,
            k_max_abs_error: k_comparison.max_abs_error,
            k_mean_abs_error: k_comparison.mean_abs_error,
            v_max_abs_error: v_comparison.max_abs_error,
            v_mean_abs_error: v_comparison.mean_abs_error,
            max_abs_error,
            mean_abs_error,
            q_argmax_index: argmax_index(&fixture.expected_q),
            k_argmax_index: argmax_index(&fixture.expected_k),
            v_argmax_index: argmax_index(&fixture.expected_v),
            timing,
        }
    }
}

impl DenseMetalPrefillAttentionScoresReceipt {
    pub fn passed(
        artifact_path: impl Into<String>,
        comparison: SmokeComparison,
        fixture: &DenseMetalPrefillAttentionScoresFixture,
        timing: DenseMetalPrefillAttentionScoresTiming,
    ) -> Self {
        Self {
            machine_id: MACHINE_ID,
            artifact_kind: SLM_APPLE_M4_METAL_PHASE_ARTIFACT_KIND,
            requested_backend: REQUESTED_BACKEND,
            selected_backend: SELECTED_BACKEND,
            runtime_api: RUNTIME_API,
            reference_backend: REFERENCE_BACKEND,
            target_backend: SELECTED_BACKEND,
            rest_of_pipeline_backend: DENSE_PREFILL_LINEAR_REST_OF_PIPELINE_BACKEND,
            kernel_id: DENSE_METAL_PREFILL_ATTENTION_SCORES_KERNEL_ID,
            model_family: DENSE_MODEL_FAMILY,
            kernel_family: DENSE_KERNEL_FAMILY,
            execution_phase: DENSE_PREFILL_ATTENTION_SCORES_EXECUTION_PHASE,
            phase_scope: DENSE_PREFILL_ATTENTION_SCORES_PHASE_SCOPE,
            layout_source: DENSE_PREFILL_ATTENTION_SCORES_LAYOUT_SOURCE,
            transport_layout: DENSE_TRANSPORT_LAYOUT,
            kv_cache_behavior: DENSE_PREFILL_ATTENTION_SCORES_KV_CACHE_BEHAVIOR,
            fallback_used: false,
            result: "pass",
            artifact_path: artifact_path.into(),
            prefill_tokens: fixture.prefill_tokens,
            attention_heads: fixture.attention_heads,
            kv_heads: fixture.kv_heads,
            head_dim: fixture.head_dim,
            score_count: fixture.expected_scores.len(),
            scale: fixture.scale,
            scores_match_cpu_reference: comparison.max_abs_error <= 1e-5,
            score_shape_matches_cpu_reference: true,
            head_mapping_matches_cpu_reference: true,
            max_abs_error: comparison.max_abs_error,
            mean_abs_error: comparison.mean_abs_error,
            score_argmax_index: argmax_index(&fixture.expected_scores),
            timing,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct I2sMetalParityFixture {
    pub activations: Vec<f32>,
    pub weights_packed: Vec<u8>,
    pub weights_packed_words: Vec<u32>,
    pub scales: Vec<f32>,
    pub expected: Vec<f32>,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub block_size: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct I2sMetalProjectionResidualFixture {
    pub base: I2sMetalParityFixture,
    pub residual: Vec<f32>,
    pub expected: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DenseMetalPrefillLinearFixture {
    pub activations: Vec<f32>,
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
    pub expected: Vec<f32>,
    pub batch_size: usize,
    pub in_features: usize,
    pub out_features: usize,
    pub cpu_reference_token_id: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DenseMetalPrefillQkvFixture {
    pub activations: Vec<f32>,
    pub q_weights: Vec<f32>,
    pub k_weights: Vec<f32>,
    pub v_weights: Vec<f32>,
    pub q_bias: Vec<f32>,
    pub k_bias: Vec<f32>,
    pub v_bias: Vec<f32>,
    pub expected_q: Vec<f32>,
    pub expected_k: Vec<f32>,
    pub expected_v: Vec<f32>,
    pub prefill_tokens: usize,
    pub hidden_size: usize,
    pub attention_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DenseMetalPrefillAttentionScoresFixture {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub expected_scores: Vec<f32>,
    pub prefill_tokens: usize,
    pub attention_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub scale: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SmokeComparison {
    pub max_abs_error: f32,
    pub mean_abs_error: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TinyMetalSmokeError {
    EmptyInput,
    LengthMismatch { expected: usize, actual: usize },
    NonFiniteOutput { index: usize, value: f32 },
    OutputMismatch { index: usize, expected: f32, actual: f32, tolerance: f32 },
}

impl fmt::Display for TinyMetalSmokeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "tiny Metal smoke input must not be empty"),
            Self::LengthMismatch { expected, actual } => {
                write!(f, "tiny Metal smoke length mismatch: expected {expected}, actual {actual}")
            }
            Self::NonFiniteOutput { index, value } => {
                write!(f, "tiny Metal smoke output at index {index} is non-finite: {value}")
            }
            Self::OutputMismatch { index, expected, actual, tolerance } => write!(
                f,
                "tiny Metal smoke output mismatch at index {index}: expected {expected}, actual {actual}, tolerance {tolerance}"
            ),
        }
    }
}

impl std::error::Error for TinyMetalSmokeError {}

pub fn metal_smoke_artifact_path(date: &str) -> String {
    format!("ci/hardware/{MACHINE_ID}/{date}/metal-smoke.json")
}

pub fn metal_parity_artifact_path(date: &str) -> String {
    format!("ci/hardware/{MACHINE_ID}/{date}/metal-parity.json")
}

pub fn metal_i2s_parity_artifact_path(date: &str) -> String {
    format!("ci/hardware/{MACHINE_ID}/{date}/metal-i2s-parity.json")
}

pub fn metal_i2s_prefill_contribution_artifact_path(date: &str) -> String {
    format!("ci/hardware/{MACHINE_ID}/{date}/metal-i2s-prefill-contribution.json")
}

pub fn metal_i2s_projection_residual_artifact_path(date: &str) -> String {
    format!("ci/hardware/{MACHINE_ID}/{date}/metal-i2s-projection-residual.json")
}

pub fn metal_dense_prefill_linear_artifact_path(date: &str) -> String {
    format!("ci/hardware/{MACHINE_ID}/{date}/metal-dense-prefill-linear.json")
}

pub fn metal_dense_prefill_qkv_artifact_path(date: &str) -> String {
    format!("ci/hardware/{MACHINE_ID}/{date}/metal-dense-prefill-qkv.json")
}

pub fn metal_dense_prefill_attention_scores_artifact_path(date: &str) -> String {
    format!(
        "ci/hardware/{MACHINE_ID}/{date}/slm-metal-phases/metal-dense-prefill-attention-scores.json"
    )
}

pub fn tiny_add_inputs() -> (Vec<f32>, Vec<f32>) {
    let lhs = (0..SMOKE_ELEMENT_COUNT).map(|i| i as f32).collect();
    let rhs = (0..SMOKE_ELEMENT_COUNT).map(|i| (i as f32) * 2.0).collect();
    (lhs, rhs)
}

pub fn expected_tiny_add(lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>, TinyMetalSmokeError> {
    if lhs.is_empty() {
        return Err(TinyMetalSmokeError::EmptyInput);
    }
    if lhs.len() != rhs.len() {
        return Err(TinyMetalSmokeError::LengthMismatch { expected: lhs.len(), actual: rhs.len() });
    }

    Ok(lhs.iter().zip(rhs.iter()).map(|(left, right)| left + right).collect())
}

pub fn i2s_metal_parity_fixture() -> I2sMetalParityFixture {
    i2s_metal_fixture(I2S_PARITY_M)
}

pub fn i2s_metal_prefill_fixture() -> I2sMetalParityFixture {
    i2s_metal_fixture(I2S_PREFILL_TOKENS)
}

pub fn i2s_metal_projection_residual_fixture() -> I2sMetalProjectionResidualFixture {
    let base = i2s_metal_fixture(I2S_PREFILL_TOKENS);
    let residual = (0..base.expected.len())
        .map(|index| {
            let row = index / base.n;
            let col = index % base.n;
            ((row as i32 * 3 + col as i32 * 2) % 7 - 3) as f32 * 0.125
        })
        .collect::<Vec<_>>();
    let expected = base
        .expected
        .iter()
        .zip(residual.iter())
        .map(|(value, residual)| value + residual)
        .collect();

    I2sMetalProjectionResidualFixture { base, residual, expected }
}

pub fn dense_metal_prefill_linear_fixture() -> DenseMetalPrefillLinearFixture {
    let activations = (0..DENSE_PREFILL_TOKENS * DENSE_PREFILL_IN_FEATURES)
        .map(|index| {
            let row = index / DENSE_PREFILL_IN_FEATURES;
            ((index as i32 * 3 + row as i32 * 5) % 17 - 8) as f32 * 0.0625
        })
        .collect::<Vec<_>>();
    let weights = (0..DENSE_PREFILL_OUT_FEATURES * DENSE_PREFILL_IN_FEATURES)
        .map(|index| {
            let row = index / DENSE_PREFILL_IN_FEATURES;
            ((index as i32 * 7 + row as i32 * 3) % 19 - 9) as f32 * 0.03125
        })
        .collect::<Vec<_>>();
    let bias = (0..DENSE_PREFILL_OUT_FEATURES)
        .map(|index| ((index as i32 * 2) % 5 - 2) as f32 * 0.015625)
        .collect::<Vec<_>>();
    let mut expected = vec![0.0; DENSE_PREFILL_TOKENS * DENSE_PREFILL_OUT_FEATURES];
    let config = LinearConfig::new(
        DENSE_PREFILL_TOKENS,
        DENSE_PREFILL_IN_FEATURES,
        DENSE_PREFILL_OUT_FEATURES,
    )
    .expect("deterministic dense M4 prefill linear fixture shape is valid")
    .with_bias(true);
    linear_cpu(&activations, &weights, Some(&bias), &mut expected, &config)
        .expect("deterministic dense M4 prefill linear fixture is valid");
    let cpu_reference_token_id = argmax_index(&expected);

    DenseMetalPrefillLinearFixture {
        activations,
        weights,
        bias,
        expected,
        batch_size: DENSE_PREFILL_TOKENS,
        in_features: DENSE_PREFILL_IN_FEATURES,
        out_features: DENSE_PREFILL_OUT_FEATURES,
        cpu_reference_token_id,
    }
}

pub fn dense_metal_prefill_qkv_fixture() -> DenseMetalPrefillQkvFixture {
    let prefill_tokens = DENSE_PREFILL_TOKENS;
    let hidden_size = QWEN2_5_0_5B_HIDDEN_SIZE;
    let q_dim = QWEN2_5_0_5B_Q_DIM;
    let kv_dim = QWEN2_5_0_5B_KV_DIM;
    let activations = (0..prefill_tokens * hidden_size)
        .map(|index| {
            let row = index / hidden_size;
            ((index as i32 * 5 + row as i32 * 7) % 23 - 11) as f32 * 0.015625
        })
        .collect::<Vec<_>>();
    let q_weights = deterministic_dense_weights(q_dim, hidden_size, 7, 29, 0.0078125);
    let k_weights = deterministic_dense_weights(kv_dim, hidden_size, 11, 31, 0.0078125);
    let v_weights = deterministic_dense_weights(kv_dim, hidden_size, 13, 37, 0.0078125);
    let q_bias = deterministic_dense_bias(q_dim, 3, 17, 0.00390625);
    let k_bias = deterministic_dense_bias(kv_dim, 5, 19, 0.00390625);
    let v_bias = deterministic_dense_bias(kv_dim, 7, 23, 0.00390625);
    let expected_q = dense_linear_expected(
        &activations,
        &q_weights,
        &q_bias,
        prefill_tokens,
        hidden_size,
        q_dim,
    );
    let expected_k = dense_linear_expected(
        &activations,
        &k_weights,
        &k_bias,
        prefill_tokens,
        hidden_size,
        kv_dim,
    );
    let expected_v = dense_linear_expected(
        &activations,
        &v_weights,
        &v_bias,
        prefill_tokens,
        hidden_size,
        kv_dim,
    );

    DenseMetalPrefillQkvFixture {
        activations,
        q_weights,
        k_weights,
        v_weights,
        q_bias,
        k_bias,
        v_bias,
        expected_q,
        expected_k,
        expected_v,
        prefill_tokens,
        hidden_size,
        attention_heads: QWEN2_5_0_5B_ATTENTION_HEADS,
        kv_heads: QWEN2_5_0_5B_KV_HEADS,
        head_dim: QWEN2_5_0_5B_HEAD_DIM,
        q_dim,
        kv_dim,
    }
}

pub fn dense_metal_prefill_attention_scores_fixture() -> DenseMetalPrefillAttentionScoresFixture {
    let prefill_tokens = DENSE_PREFILL_ATTENTION_SCORE_TOKENS;
    let attention_heads = QWEN2_5_0_5B_ATTENTION_HEADS;
    let kv_heads = QWEN2_5_0_5B_KV_HEADS;
    let head_dim = QWEN2_5_0_5B_HEAD_DIM;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let q = (0..prefill_tokens * attention_heads * head_dim)
        .map(|index| {
            let token = index / (attention_heads * head_dim);
            let head = (index / head_dim) % attention_heads;
            let dim = index % head_dim;
            ((token as i32 * 17 + head as i32 * 5 + dim as i32 * 3) % 31 - 15) as f32 * 0.015625
        })
        .collect::<Vec<_>>();
    let k = (0..prefill_tokens * kv_heads * head_dim)
        .map(|index| {
            let token = index / (kv_heads * head_dim);
            let kv_head = (index / head_dim) % kv_heads;
            let dim = index % head_dim;
            ((token as i32 * 13 + kv_head as i32 * 7 + dim as i32 * 2) % 29 - 14) as f32 * 0.015625
        })
        .collect::<Vec<_>>();
    let expected_scores = dense_prefill_attention_scores_expected(
        &q,
        &k,
        prefill_tokens,
        attention_heads,
        kv_heads,
        head_dim,
    );

    DenseMetalPrefillAttentionScoresFixture {
        q,
        k,
        expected_scores,
        prefill_tokens,
        attention_heads,
        kv_heads,
        head_dim,
        scale,
    }
}

pub fn dense_prefill_attention_scores_expected(
    q: &[f32],
    k: &[f32],
    prefill_tokens: usize,
    attention_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    assert_eq!(q.len(), prefill_tokens * attention_heads * head_dim);
    assert_eq!(k.len(), prefill_tokens * kv_heads * head_dim);
    assert_eq!(attention_heads % kv_heads, 0);

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut scores = vec![0.0; attention_heads * prefill_tokens * prefill_tokens];
    for head in 0..attention_heads {
        let kv_head = dense_prefill_attention_scores_kv_head(head, attention_heads, kv_heads);
        for query_token in 0..prefill_tokens {
            for key_token in 0..prefill_tokens {
                let mut acc = 0.0_f32;
                for dim in 0..head_dim {
                    let q_index = ((query_token * attention_heads + head) * head_dim) + dim;
                    let k_index = ((key_token * kv_heads + kv_head) * head_dim) + dim;
                    acc += q[q_index] * k[k_index];
                }
                let score_index = (head * prefill_tokens * prefill_tokens)
                    + (query_token * prefill_tokens)
                    + key_token;
                scores[score_index] = acc * scale;
            }
        }
    }
    scores
}

pub fn dense_prefill_attention_scores_kv_head(
    attention_head: usize,
    attention_heads: usize,
    kv_heads: usize,
) -> usize {
    attention_head / (attention_heads / kv_heads)
}

fn deterministic_dense_weights(
    out_features: usize,
    in_features: usize,
    multiplier: i32,
    modulus: i32,
    scale: f32,
) -> Vec<f32> {
    (0..out_features * in_features)
        .map(|index| {
            let row = index / in_features;
            ((index as i32 * multiplier + row as i32 * 3) % modulus - modulus / 2) as f32 * scale
        })
        .collect()
}

fn deterministic_dense_bias(
    out_features: usize,
    multiplier: i32,
    modulus: i32,
    scale: f32,
) -> Vec<f32> {
    (0..out_features)
        .map(|index| ((index as i32 * multiplier) % modulus - modulus / 2) as f32 * scale)
        .collect()
}

fn dense_linear_expected(
    activations: &[f32],
    weights: &[f32],
    bias: &[f32],
    prefill_tokens: usize,
    in_features: usize,
    out_features: usize,
) -> Vec<f32> {
    let mut expected = vec![0.0; prefill_tokens * out_features];
    let config = LinearConfig::new(prefill_tokens, in_features, out_features)
        .expect("deterministic dense QKV fixture shape is valid")
        .with_bias(true);
    linear_cpu(activations, weights, Some(bias), &mut expected, &config)
        .expect("deterministic dense QKV fixture is valid");
    expected
}

fn i2s_metal_fixture(m: usize) -> I2sMetalParityFixture {
    let activations = (0..m * I2S_PARITY_K)
        .map(|index| {
            let row = index / I2S_PARITY_K;
            ((index as i32 + row as i32 * 2) % 9 - 4) as f32 * 0.25
        })
        .collect::<Vec<_>>();
    let scales = vec![0.25, 0.5, 0.75, 1.0];

    let mut weights_packed = Vec::with_capacity((I2S_PARITY_K / 4) * I2S_PARITY_N);
    for col in 0..I2S_PARITY_N {
        for chunk_start in (0..I2S_PARITY_K).step_by(4) {
            let vals = std::array::from_fn(|offset| {
                let selector = (chunk_start + offset + col * 3) % 5;
                match selector {
                    0 | 3 => -1,
                    1 => 0,
                    _ => 1,
                }
            });
            weights_packed.push(pack_i2s(vals));
        }
    }

    let weights_packed_words = pack_i2s_bytes_to_u32_words(&weights_packed);
    let mut expected = vec![0.0; m * I2S_PARITY_N];
    i2s_matmul_f32(
        &activations,
        &weights_packed,
        &scales,
        &mut expected,
        m,
        I2S_PARITY_N,
        I2S_PARITY_K,
        I2S_PARITY_BLOCK_SIZE,
    )
    .expect("deterministic M4 I2_S parity fixture is valid");

    I2sMetalParityFixture {
        activations,
        weights_packed,
        weights_packed_words,
        scales,
        expected,
        m,
        n: I2S_PARITY_N,
        k: I2S_PARITY_K,
        block_size: I2S_PARITY_BLOCK_SIZE,
    }
}

pub fn i2s_parity_shape_words(fixture: &I2sMetalParityFixture) -> [u32; 6] {
    [
        fixture.m as u32,
        fixture.n as u32,
        fixture.k as u32,
        fixture.k.div_ceil(4) as u32,
        fixture.block_size as u32,
        fixture.k.div_ceil(fixture.block_size) as u32,
    ]
}

pub fn dense_prefill_linear_shape_words(fixture: &DenseMetalPrefillLinearFixture) -> [u32; 3] {
    [fixture.batch_size as u32, fixture.out_features as u32, fixture.in_features as u32]
}

pub fn dense_prefill_qkv_shape_words(fixture: &DenseMetalPrefillQkvFixture) -> [u32; 7] {
    [
        fixture.prefill_tokens as u32,
        fixture.hidden_size as u32,
        fixture.q_dim as u32,
        fixture.kv_dim as u32,
        fixture.attention_heads as u32,
        fixture.kv_heads as u32,
        fixture.head_dim as u32,
    ]
}

pub fn dense_prefill_attention_scores_shape_words(
    fixture: &DenseMetalPrefillAttentionScoresFixture,
) -> [u32; 4] {
    [
        fixture.prefill_tokens as u32,
        fixture.attention_heads as u32,
        fixture.kv_heads as u32,
        fixture.head_dim as u32,
    ]
}

pub fn argmax_index(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| {
            left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(index, _)| index)
        .unwrap_or_default()
}

pub fn pack_i2s_bytes_to_u32_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks(4)
        .map(|chunk| {
            let mut padded = [0_u8; 4];
            padded[..chunk.len()].copy_from_slice(chunk);
            u32::from_le_bytes(padded)
        })
        .collect()
}

pub fn compare_tiny_add_outputs(
    expected: &[f32],
    actual: &[f32],
    tolerance: f32,
) -> Result<SmokeComparison, TinyMetalSmokeError> {
    if expected.is_empty() {
        return Err(TinyMetalSmokeError::EmptyInput);
    }
    if expected.len() != actual.len() {
        return Err(TinyMetalSmokeError::LengthMismatch {
            expected: expected.len(),
            actual: actual.len(),
        });
    }

    let mut max_abs_error = 0.0_f32;
    let mut total_abs_error = 0.0_f32;

    for (index, (&expected_value, &actual_value)) in expected.iter().zip(actual.iter()).enumerate()
    {
        if !actual_value.is_finite() {
            return Err(TinyMetalSmokeError::NonFiniteOutput { index, value: actual_value });
        }

        let abs_error = (actual_value - expected_value).abs();
        if abs_error > tolerance {
            return Err(TinyMetalSmokeError::OutputMismatch {
                index,
                expected: expected_value,
                actual: actual_value,
                tolerance,
            });
        }

        max_abs_error = max_abs_error.max(abs_error);
        total_abs_error += abs_error;
    }

    Ok(SmokeComparison { max_abs_error, mean_abs_error: total_abs_error / expected.len() as f32 })
}

pub fn is_apple_m4_adapter_name(adapter_name: &str) -> bool {
    adapter_name.to_ascii_lowercase().contains("apple m4")
}

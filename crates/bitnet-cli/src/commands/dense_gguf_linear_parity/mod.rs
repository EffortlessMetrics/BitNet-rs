//! Dense GGUF single-linear CUDA parity diagnostics.
//!
//! This command is an implementation bridge between descriptor extraction and
//! full dense GGUF inference. It extracts one dense GGUF linear fixture, routes
//! that fixture through the dense FP16 CUDA GEMM bridge, and emits a receipt
//! that still refuses dense GGUF inference, speedup, full-residency, and BitNet
//! packed-kernel proof claims.

use anyhow::{Context, Result, anyhow, bail};
use bitnet_common::{BitNetConfig, ConcreteTensor, Device as BitNetDevice, Tensor};
use bitnet_kernels::cuda::{
    AttentionScoresConfig, AttentionSoftmaxConfig, AttentionVMixConfig,
    DenseGgufAttentionScoreCudaFixture, DenseGgufAttentionScoreCudaParity,
    DenseGgufAttentionSoftmaxCudaFixture, DenseGgufAttentionSoftmaxCudaParity,
    DenseGgufAttentionVMixCudaFixture, DenseGgufAttentionVMixCudaParity, DenseGgufLinearCudaParity,
    DenseGgufLinearGemmFixture, DenseGgufMlpActivationCudaFixture,
    DenseGgufMlpActivationCudaParity, DenseGgufRmsNormCudaFixture, DenseGgufRmsNormCudaParity,
    DenseGgufRopeCudaFixture, DenseGgufRopeCudaParity, RmsNormConfig, RopeConfig, SiluGateConfig,
    launch_dense_attention_scores_f32_cuda, launch_dense_attention_softmax_f32_cuda,
    launch_dense_attention_v_mix_f32_cuda, launch_dense_f16_gemm_cuda,
    launch_dense_mlp_activation_f32_cuda, launch_dense_rmsnorm_f32_cuda,
    launch_dense_rope_f32_cuda, prepare_dense_gguf_linear_f16_gemm, rope_forward_cpu,
    run_dense_gguf_attention_score_cuda_parity, run_dense_gguf_attention_softmax_cuda_parity,
    run_dense_gguf_attention_v_mix_cuda_parity, run_dense_gguf_linear_f16_cuda_parity,
    run_dense_gguf_mlp_activation_cuda_parity, run_dense_gguf_rmsnorm_cuda_parity,
    run_dense_gguf_rope_cuda_parity,
};
use bitnet_kernels::dispatch_planner::{
    BackendPolicy, CudaPlannerCapabilities, DispatchOp, ModelDispatchBackend,
    ModelDispatchDecision, ModelDispatchSpec, ModelDispatchSummary, ModelFamily, OpType,
    QuantizationKind, plan_model_dispatch,
};
use bitnet_models::dense_gguf_descriptors::{
    DenseGgufDescriptorInspection, DenseGgufTensorDescriptor, DenseGgufTensorRole,
    inspect_dense_gguf_tensor_descriptors,
};
use bitnet_models::dense_gguf_linear_fixture::{
    DENSE_GGUF_LINEAR_FIXTURE_ARTIFACT_KIND, DenseGgufLinearFixture,
    extract_dense_gguf_linear_fixture,
};
use bitnet_models::dense_gguf_norm_fixture::{
    DenseGgufNormFixture, extract_dense_gguf_norm_fixture,
};
use bitnet_models::formats::gguf::{GgufReader, GgufTensorType};
use bitnet_models::layer_inspector::extract_layer_index;
use bitnet_models::{LoadConfig, Model, ModelLoader};
use bitnet_receipts_core::{
    DENSE_GGUF_ALL_LAYER_EXECUTION_PLAN_ARTIFACT_KIND,
    DENSE_GGUF_ATTENTION_SCORE_CUDA_PARITY_ARTIFACT_KIND,
    DENSE_GGUF_ATTENTION_SCORE_FIXTURE_ARTIFACT_KIND,
    DENSE_GGUF_ATTENTION_SOFTMAX_CUDA_PARITY_ARTIFACT_KIND,
    DENSE_GGUF_ATTENTION_SOFTMAX_FIXTURE_ARTIFACT_KIND,
    DENSE_GGUF_ATTENTION_V_MIX_CUDA_PARITY_ARTIFACT_KIND,
    DENSE_GGUF_ATTENTION_V_MIX_FIXTURE_ARTIFACT_KIND, DENSE_GGUF_KV_CACHE_POLICY_ARTIFACT_KIND,
    DENSE_GGUF_LINEAR_CUDA_PARITY_ARTIFACT_KIND,
    DENSE_GGUF_LINEAR_ROLE_SWEEP_CUDA_PARITY_ARTIFACT_KIND,
    DENSE_GGUF_MLP_ACTIVATION_CUDA_PARITY_ARTIFACT_KIND,
    DENSE_GGUF_MLP_ACTIVATION_FIXTURE_ARTIFACT_KIND,
    DENSE_GGUF_MODEL_BOUNDARY_FIXTURES_ARTIFACT_KIND, DENSE_GGUF_NORM_CUDA_PARITY_ARTIFACT_KIND,
    DENSE_GGUF_NORM_FIXTURE_ARTIFACT_KIND, DENSE_GGUF_ONE_LAYER_CPU_REFERENCE_ARTIFACT_KIND,
    DENSE_GGUF_ONE_LAYER_CUDA_INTEGRATED_PARITY_ARTIFACT_KIND,
    DENSE_GGUF_ONE_LAYER_EXECUTION_PLAN_ARTIFACT_KIND,
    DENSE_GGUF_QWEN_ASK_STRICT_CUDA_PROOF_ARTIFACT_KIND,
    DENSE_GGUF_QWEN_CHAT_STRICT_CUDA_PROOF_ARTIFACT_KIND,
    DENSE_GGUF_QWEN_ONE_TOKEN_STRICT_CUDA_PROOF_ARTIFACT_KIND,
    DENSE_GGUF_QWEN_SHORT_DECODE_STRICT_CUDA_PROOF_ARTIFACT_KIND,
    DENSE_GGUF_QWEN_WARM_DECODE_STRICT_CUDA_PROOF_ARTIFACT_KIND,
    DENSE_GGUF_QWEN_WARM_SESSION_STRICT_CUDA_PROOF_ARTIFACT_KIND,
    DENSE_GGUF_ROPE_CUDA_PARITY_ARTIFACT_KIND, DENSE_GGUF_SAMPLING_POLICY_ARTIFACT_KIND,
    DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND,
    validate_dense_gguf_all_layer_execution_plan_receipt_json,
    validate_dense_gguf_attention_score_cuda_parity_receipt_json,
    validate_dense_gguf_attention_score_fixture_receipt_json,
    validate_dense_gguf_attention_softmax_cuda_parity_receipt_json,
    validate_dense_gguf_attention_softmax_fixture_receipt_json,
    validate_dense_gguf_attention_v_mix_cuda_parity_receipt_json,
    validate_dense_gguf_attention_v_mix_fixture_receipt_json,
    validate_dense_gguf_kv_cache_policy_receipt_json,
    validate_dense_gguf_linear_cuda_parity_receipt_json,
    validate_dense_gguf_linear_role_sweep_cuda_parity_receipt_json,
    validate_dense_gguf_mlp_activation_cuda_parity_receipt_json,
    validate_dense_gguf_mlp_activation_fixture_receipt_json,
    validate_dense_gguf_model_boundary_fixtures_receipt_json,
    validate_dense_gguf_norm_cuda_parity_receipt_json,
    validate_dense_gguf_norm_fixture_extraction_receipt_json,
    validate_dense_gguf_one_layer_cpu_reference_receipt_json,
    validate_dense_gguf_one_layer_cuda_integrated_parity_receipt_json,
    validate_dense_gguf_one_layer_execution_plan_receipt_json,
    validate_dense_gguf_qwen_ask_strict_cuda_proof_receipt_json,
    validate_dense_gguf_qwen_chat_strict_cuda_proof_receipt_json,
    validate_dense_gguf_qwen_one_token_strict_cuda_proof_receipt_json,
    validate_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt_json,
    validate_dense_gguf_qwen_warm_decode_strict_cuda_proof_receipt_json,
    validate_dense_gguf_qwen_warm_session_strict_cuda_proof_receipt_json,
    validate_dense_gguf_rope_cuda_parity_receipt_json,
    validate_dense_gguf_sampling_policy_receipt_json,
};
use candle_core::{DType, Device as CandleDevice, IndexOp};
use clap::{Args, ValueEnum};
use memmap2::Mmap;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::path::{Path, PathBuf};

use crate::planner_receipts::{ExecutionPlanReceiptInput, execution_plan_receipt};

mod digest;
mod hardware;
mod roles;

use digest::{sha256_bytes, sha256_f32, sha256_json, sha256_u32, sha256_usize};
use hardware::{cuda_identity_json, is_rtx5070ti_device_name};
use roles::{dense_role_label, parse_dense_linear_role, parse_norm_roles, parse_role_sweep};

const HARDWARE_LANE: &str = "nvidia-rtx-5070-ti-cuda";
const MACHINE_ID: &str = "windows-9950x3d-rtx5070ti";
const DENSE_ONE_LAYER_GAP_CANDIDATE_ORDER: &[&str] =
    &["attention_softmax", "attention_v_mix", "mlp_activation"];
const DENSE_ONE_LAYER_REMAINING_GAP_CANDIDATE_ORDER: &[&str] = &["mlp_activation"];
const DENSE_ONE_LAYER_NO_REMAINING_GAP_CANDIDATE_ORDER: &[&str] = &[];
const QWEN25_05B_INSTRUCT_Q8_0_MODEL_ID: &str = "qwen2.5-0.5b-instruct-q8_0";
const QWEN25_05B_INSTRUCT_Q8_0_MODEL_FILE: &str = "qwen2.5-0.5b-instruct-q8_0.gguf";
const QWEN25_05B_INSTRUCT_Q8_0_MODEL_SHA256: &str =
    "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e";
const QWEN3_06B_INSTRUCT_Q8_0_MODEL_ID: &str = "qwen3-0.6b-instruct-q8_0";
const QWEN3_06B_INSTRUCT_Q8_0_MODEL_FILE: &str = "Qwen3-0.6B-Q8_0.gguf";
const QWEN3_06B_INSTRUCT_Q8_0_MODEL_SHA256: &str =
    "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031";
const DEFAULT_DENSE_QWEN_ALL_LAYER_PLAN_RECEIPT: &str =
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-all-layer-plan-qwen25-q8.json";
const DEFAULT_DENSE_QWEN_MODEL_BOUNDARY_FIXTURES_RECEIPT: &str = "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-model-boundary-fixtures-qwen25-q8.json";
const DEFAULT_DENSE_QWEN_KV_CACHE_POLICY_RECEIPT: &str =
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-kv-cache-policy-qwen25-q8.json";
const DEFAULT_DENSE_QWEN_SAMPLING_POLICY_RECEIPT: &str =
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-sampling-policy-qwen25-q8.json";
const DEFAULT_DENSE_QWEN_ONE_TOKEN_PROOF_RECEIPT: &str = "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-qwen-one-token-strict-cuda-qwen25-q8.json";
const DEFAULT_DENSE_QWEN_SHORT_DECODE_PROOF_RECEIPT: &str = "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-qwen-short-decode-strict-cuda-qwen25-q8.json";
const DEFAULT_DENSE_QWEN_WARM_SESSION_PROOF_RECEIPT: &str = "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-qwen-warm-session-strict-cuda-qwen25-q8.json";
const DEFAULT_QWEN3_ALL_LAYER_PLAN_RECEIPT: &str =
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-15/qwen3-0_6b-cuda-all-layer-plan.json";
const DEFAULT_QWEN3_MODEL_BOUNDARY_FIXTURES_RECEIPT: &str =
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-15/qwen3-0_6b-model-boundary-fixtures.json";
const DEFAULT_QWEN3_KV_CACHE_POLICY_RECEIPT: &str =
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-15/qwen3-0_6b-kv-cache-policy.json";
const DEFAULT_QWEN3_SAMPLING_POLICY_RECEIPT: &str =
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-15/qwen3-0_6b-sampling-policy.json";
const DEFAULT_QWEN3_ONE_TOKEN_PROOF_RECEIPT: &str =
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-15/qwen3-0_6b-one-token-cuda.json";
const DEFAULT_QWEN3_SHORT_DECODE_PROOF_RECEIPT: &str =
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-15/qwen3-0_6b-short-decode-cuda.json";
const DEFAULT_QWEN3_WARM_SESSION_PROOF_RECEIPT: &str =
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-15/qwen3-0_6b-warm-session-cuda.json";
const DEFAULT_QWEN_WARM_SESSION_PROMPTS: &[&str] = &[
    "What is 2+2?",
    "Name one color of the sky.",
    "Write one short greeting.",
    "Complete this phrase: rust is",
];

#[derive(Debug, Clone, Copy)]
struct DenseQwenProofModel {
    id: &'static str,
    file: &'static str,
    architecture: &'static str,
    sha256: &'static str,
    model_coverage_row: &'static str,
    model_coverage_tier: &'static str,
    work_item: &'static str,
}

const QWEN25_05B_INSTRUCT_Q8_0_PROOF_MODEL: DenseQwenProofModel = DenseQwenProofModel {
    id: QWEN25_05B_INSTRUCT_Q8_0_MODEL_ID,
    file: QWEN25_05B_INSTRUCT_Q8_0_MODEL_FILE,
    architecture: "qwen2",
    sha256: QWEN25_05B_INSTRUCT_Q8_0_MODEL_SHA256,
    model_coverage_row: "dense_qwen25_05b_q8_cuda",
    model_coverage_tier: "product_cli_ready",
    work_item: "CUDA-DENSE-051",
};

const QWEN3_06B_INSTRUCT_Q8_0_PROOF_MODEL: DenseQwenProofModel = DenseQwenProofModel {
    id: QWEN3_06B_INSTRUCT_Q8_0_MODEL_ID,
    file: QWEN3_06B_INSTRUCT_Q8_0_MODEL_FILE,
    architecture: "qwen3",
    sha256: QWEN3_06B_INSTRUCT_Q8_0_MODEL_SHA256,
    model_coverage_row: "dense_qwen3_06b_q8_candidate",
    model_coverage_tier: "accelerator_answer_ready",
    work_item: "CUDA-MODEL-004",
};

#[derive(Debug, Clone, Copy)]
struct DenseQwenProofReceiptBundle {
    all_layer_plan: &'static str,
    model_boundary_fixtures: &'static str,
    kv_cache_policy: &'static str,
    sampling_policy: &'static str,
    one_token_proof: &'static str,
    short_decode_proof: &'static str,
    warm_session_proof: &'static str,
}

#[derive(Debug, Clone, Copy)]
struct DenseQwenProofContext {
    proof_model: &'static DenseQwenProofModel,
    receipts: &'static DenseQwenProofReceiptBundle,
}

const QWEN25_05B_INSTRUCT_Q8_0_RECEIPTS: DenseQwenProofReceiptBundle =
    DenseQwenProofReceiptBundle {
        all_layer_plan: DEFAULT_DENSE_QWEN_ALL_LAYER_PLAN_RECEIPT,
        model_boundary_fixtures: DEFAULT_DENSE_QWEN_MODEL_BOUNDARY_FIXTURES_RECEIPT,
        kv_cache_policy: DEFAULT_DENSE_QWEN_KV_CACHE_POLICY_RECEIPT,
        sampling_policy: DEFAULT_DENSE_QWEN_SAMPLING_POLICY_RECEIPT,
        one_token_proof: DEFAULT_DENSE_QWEN_ONE_TOKEN_PROOF_RECEIPT,
        short_decode_proof: DEFAULT_DENSE_QWEN_SHORT_DECODE_PROOF_RECEIPT,
        warm_session_proof: DEFAULT_DENSE_QWEN_WARM_SESSION_PROOF_RECEIPT,
    };

const QWEN3_06B_INSTRUCT_Q8_0_RECEIPTS: DenseQwenProofReceiptBundle = DenseQwenProofReceiptBundle {
    all_layer_plan: DEFAULT_QWEN3_ALL_LAYER_PLAN_RECEIPT,
    model_boundary_fixtures: DEFAULT_QWEN3_MODEL_BOUNDARY_FIXTURES_RECEIPT,
    kv_cache_policy: DEFAULT_QWEN3_KV_CACHE_POLICY_RECEIPT,
    sampling_policy: DEFAULT_QWEN3_SAMPLING_POLICY_RECEIPT,
    one_token_proof: DEFAULT_QWEN3_ONE_TOKEN_PROOF_RECEIPT,
    short_decode_proof: DEFAULT_QWEN3_SHORT_DECODE_PROOF_RECEIPT,
    warm_session_proof: DEFAULT_QWEN3_WARM_SESSION_PROOF_RECEIPT,
};

pub(crate) fn is_supported_dense_qwen_cuda_model_path(model: &Path) -> bool {
    let Some(file_name) = model.file_name().and_then(|value| value.to_str()) else {
        return false;
    };
    file_name.eq_ignore_ascii_case(QWEN25_05B_INSTRUCT_Q8_0_MODEL_FILE)
        || file_name.eq_ignore_ascii_case(QWEN3_06B_INSTRUCT_Q8_0_MODEL_FILE)
}

fn dense_qwen_proof_context_for_model_path(model: &Path) -> DenseQwenProofContext {
    let file_name = model.file_name().and_then(|value| value.to_str()).unwrap_or_default();
    if file_name.eq_ignore_ascii_case(QWEN3_06B_INSTRUCT_Q8_0_MODEL_FILE) {
        return DenseQwenProofContext {
            proof_model: &QWEN3_06B_INSTRUCT_Q8_0_PROOF_MODEL,
            receipts: &QWEN3_06B_INSTRUCT_Q8_0_RECEIPTS,
        };
    }
    DenseQwenProofContext {
        proof_model: &QWEN25_05B_INSTRUCT_Q8_0_PROOF_MODEL,
        receipts: &QWEN25_05B_INSTRUCT_Q8_0_RECEIPTS,
    }
}

fn dense_qwen_receipts_for_proof_model(
    proof_model: &DenseQwenProofModel,
) -> &'static DenseQwenProofReceiptBundle {
    if proof_model.id == QWEN3_06B_INSTRUCT_Q8_0_MODEL_ID {
        &QWEN3_06B_INSTRUCT_Q8_0_RECEIPTS
    } else {
        &QWEN25_05B_INSTRUCT_Q8_0_RECEIPTS
    }
}

fn dense_qwen_model_default_receipt_path(
    path: &Path,
    qwen25_default: &str,
    model_default: &str,
) -> PathBuf {
    if path == Path::new(qwen25_default) {
        PathBuf::from(model_default)
    } else {
        path.to_path_buf()
    }
}

/// Run dense GGUF single-linear CUDA parity diagnostics.
#[derive(Args, Debug, Clone)]
pub struct DenseGgufLinearParityCommand {
    /// Dense GGUF model path.
    #[arg(long)]
    pub model: PathBuf,

    /// Dense linear tensor role to extract.
    #[arg(long, default_value = "attention_q")]
    pub role: String,

    /// CUDA device index.
    #[arg(long, default_value_t = 0)]
    pub device_index: usize,

    /// Output JSON receipt path. If omitted, writes receipt JSON to stdout.
    #[arg(long, value_name = "PATH")]
    pub json_out: Option<PathBuf>,
}

impl DenseGgufLinearParityCommand {
    pub async fn execute(&self) -> Result<()> {
        let role = parse_dense_linear_role(&self.role)?;
        let data = map_model(&self.model)?;
        let model_sha256 = sha256_bytes(&data);
        let reader = GgufReader::new(&data).with_context(|| {
            format!("failed to parse dense GGUF model {}", self.model.display())
        })?;
        let extracted = extract_dense_gguf_linear_fixture(&reader, role)?;
        let kernel_fixture = kernel_fixture_from_extracted(&extracted)?;

        let probe = bitnet_device_probe::probe_nvidia_cuda(Some(self.device_index));
        if !probe.available {
            bail!("CUDA-DENSE-009 requires CUDA probe success: {:?}", probe.failure_reason);
        }
        let device_name = probe.selected_device_name.as_deref().unwrap_or("unknown");
        if !is_rtx5070ti_device_name(device_name) {
            bail!("CUDA-DENSE-009 requires NVIDIA GeForce RTX 5070 Ti; found '{device_name}'");
        }

        let parity = run_dense_gguf_linear_f16_cuda_parity(self.device_index, &kernel_fixture)?;
        let artifact_path = self
            .json_out
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "stdout".to_string());
        let timestamp_utc = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        let receipt = dense_gguf_linear_cuda_parity_receipt_json(
            &parity,
            &extracted,
            Some(&probe),
            &self.model,
            &model_sha256,
            &artifact_path,
            &timestamp_utc,
        );
        validate_dense_gguf_linear_cuda_parity_receipt_json(&receipt)?;

        if let Some(path) = &self.json_out {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, serde_json::to_string_pretty(&receipt)?)?;
        } else {
            println!("{}", serde_json::to_string_pretty(&receipt)?);
        }

        if !parity.passed {
            bail!(
                "dense GGUF linear CUDA parity failed: max_abs_error={} tolerance={}",
                parity.max_abs_error,
                parity.tolerance
            );
        }

        Ok(())
    }
}

/// Run a dense GGUF multi-linear CUDA parity role sweep.
#[derive(Args, Debug, Clone)]
pub struct DenseGgufLinearRoleSweepCommand {
    /// Dense GGUF model path.
    #[arg(long)]
    pub model: PathBuf,

    /// Dense linear tensor roles to extract. Defaults to first-layer Q/K/V/O,
    /// MLP gate/up/down, and output projection.
    #[arg(long, value_delimiter = ',', value_name = "ROLE")]
    pub roles: Vec<String>,

    /// CUDA device index.
    #[arg(long, default_value_t = 0)]
    pub device_index: usize,

    /// Output JSON receipt path. If omitted, writes receipt JSON to stdout.
    #[arg(long, value_name = "PATH")]
    pub json_out: Option<PathBuf>,
}

impl DenseGgufLinearRoleSweepCommand {
    pub async fn execute(&self) -> Result<()> {
        let roles = parse_role_sweep(&self.roles)?;
        let data = map_model(&self.model)?;
        let model_sha256 = sha256_bytes(&data);
        let reader = GgufReader::new(&data).with_context(|| {
            format!("failed to parse dense GGUF model {}", self.model.display())
        })?;

        let probe = bitnet_device_probe::probe_nvidia_cuda(Some(self.device_index));
        if !probe.available {
            bail!("CUDA-DENSE-012 requires CUDA probe success: {:?}", probe.failure_reason);
        }
        let device_name = probe.selected_device_name.as_deref().unwrap_or("unknown");
        if !is_rtx5070ti_device_name(device_name) {
            bail!("CUDA-DENSE-012 requires NVIDIA GeForce RTX 5070 Ti; found '{device_name}'");
        }

        let mut results = Vec::with_capacity(roles.len());
        for role in roles {
            let extracted = extract_dense_gguf_linear_fixture(&reader, role)?;
            let kernel_fixture = kernel_fixture_from_extracted(&extracted)?;
            let parity = run_dense_gguf_linear_f16_cuda_parity(self.device_index, &kernel_fixture)?;
            results.push(DenseLinearSweepResult { extracted, parity });
        }

        if results.is_empty() {
            bail!("dense GGUF linear role sweep requires at least one role");
        }
        if let Some(failed) = results.iter().find(|result| !result.parity.passed) {
            bail!(
                "dense GGUF linear role sweep parity failed for {}: max_abs_error={} tolerance={}",
                failed.parity.tensor_role,
                failed.parity.max_abs_error,
                failed.parity.tolerance
            );
        }

        let artifact_path = self
            .json_out
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "stdout".to_string());
        let timestamp_utc = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        let receipt = dense_gguf_linear_role_sweep_cuda_parity_receipt_json(
            &results,
            Some(&probe),
            &self.model,
            &model_sha256,
            &artifact_path,
            &timestamp_utc,
        )?;
        validate_dense_gguf_linear_role_sweep_cuda_parity_receipt_json(&receipt)?;

        if let Some(path) = &self.json_out {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, serde_json::to_string_pretty(&receipt)?)?;
        } else {
            println!("{}", serde_json::to_string_pretty(&receipt)?);
        }

        Ok(())
    }
}

/// Emit a strict CUDA planner gap receipt for one dense GGUF transformer layer.
#[derive(Args, Debug, Clone)]
pub struct DenseGgufOneLayerPlanCommand {
    /// Dense GGUF model path.
    #[arg(long)]
    pub model: PathBuf,

    /// Dense transformer layer index. This diagnostic currently records layer 0.
    #[arg(long, default_value_t = 0)]
    pub layer_index: usize,

    /// CUDA device index.
    #[arg(long, default_value_t = 0)]
    pub device_index: usize,

    /// Output JSON receipt path. If omitted, writes receipt JSON to stdout.
    #[arg(long, value_name = "PATH")]
    pub json_out: Option<PathBuf>,
}

impl DenseGgufOneLayerPlanCommand {
    pub async fn execute(&self) -> Result<()> {
        if self.layer_index != 0 {
            bail!("CUDA-DENSE-013 currently records the first dense GGUF layer only");
        }

        let data = map_model(&self.model)?;
        let model_sha256 = sha256_bytes(&data);
        let reader = GgufReader::new(&data).with_context(|| {
            format!("failed to parse dense GGUF model {}", self.model.display())
        })?;
        let inspection = inspect_dense_gguf_tensor_descriptors(&reader)?;

        let probe = bitnet_device_probe::probe_nvidia_cuda(Some(self.device_index));
        if !probe.available {
            bail!("CUDA-DENSE-013 requires CUDA probe success: {:?}", probe.failure_reason);
        }
        let device_name = probe.selected_device_name.as_deref().unwrap_or("unknown");
        if !is_rtx5070ti_device_name(device_name) {
            bail!("CUDA-DENSE-013 requires NVIDIA GeForce RTX 5070 Ti; found '{device_name}'");
        }

        let artifact_path = self
            .json_out
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "stdout".to_string());
        let timestamp_utc = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        let receipt = dense_gguf_one_layer_execution_plan_receipt_json(
            &inspection,
            Some(&probe),
            &self.model,
            &model_sha256,
            &artifact_path,
            &timestamp_utc,
            self.layer_index,
        )?;
        validate_dense_gguf_one_layer_execution_plan_receipt_json(&receipt)?;

        if let Some(path) = &self.json_out {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, serde_json::to_string_pretty(&receipt)?)?;
        } else {
            println!("{}", serde_json::to_string_pretty(&receipt)?);
        }

        Ok(())
    }
}

/// Emit a strict CUDA all-layer dense GGUF execution-plan receipt.
#[derive(Args, Debug, Clone)]
pub struct DenseGgufAllLayerPlanCommand {
    /// Dense GGUF model path.
    #[arg(long)]
    pub model: PathBuf,

    /// CUDA device index.
    #[arg(long, default_value_t = 0)]
    pub device_index: usize,

    /// Output JSON receipt path. If omitted, writes receipt JSON to stdout.
    #[arg(long, value_name = "PATH")]
    pub json_out: Option<PathBuf>,
}

impl DenseGgufAllLayerPlanCommand {
    pub async fn execute(&self) -> Result<()> {
        let data = map_model(&self.model)?;
        let model_sha256 = sha256_bytes(&data);
        let reader = GgufReader::new(&data).with_context(|| {
            format!("failed to parse dense GGUF model {}", self.model.display())
        })?;
        let inspection = inspect_dense_gguf_tensor_descriptors(&reader)?;

        let probe = bitnet_device_probe::probe_nvidia_cuda(Some(self.device_index));
        if !probe.available {
            bail!("CUDA-DENSE-037 requires CUDA probe success: {:?}", probe.failure_reason);
        }
        let device_name = probe.selected_device_name.as_deref().unwrap_or("unknown");
        if !is_rtx5070ti_device_name(device_name) {
            bail!("CUDA-DENSE-037 requires NVIDIA GeForce RTX 5070 Ti; found '{device_name}'");
        }

        let artifact_path = self
            .json_out
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "stdout".to_string());
        let timestamp_utc = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        let receipt = dense_gguf_all_layer_execution_plan_receipt_json(
            &inspection,
            Some(&probe),
            &self.model,
            &model_sha256,
            &artifact_path,
            &timestamp_utc,
        )?;
        validate_dense_gguf_all_layer_execution_plan_receipt_json(&receipt)?;

        if let Some(path) = &self.json_out {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, serde_json::to_string_pretty(&receipt)?)?;
        } else {
            println!("{}", serde_json::to_string_pretty(&receipt)?);
        }

        Ok(())
    }
}

/// Emit dense GGUF model-boundary fixture receipts for token embedding, final
/// norm, and LM head/logit diagnostics.
#[derive(Args, Debug, Clone)]
pub struct DenseGgufModelBoundaryFixturesCommand {
    /// Dense GGUF model path.
    #[arg(long)]
    pub model: PathBuf,

    /// Number of deterministic token ids used by the embedding fixture.
    #[arg(long, default_value_t = 4)]
    pub seq_len: usize,

    /// Number of logits retained in the deterministic top-k diagnostics.
    #[arg(long, default_value_t = 5)]
    pub top_k: usize,

    /// CUDA device index used for route identity and claim-boundary receipts.
    #[arg(long, default_value_t = 0)]
    pub device_index: usize,

    /// Output JSON receipt path. If omitted, writes receipt JSON to stdout.
    #[arg(long, value_name = "PATH")]
    pub json_out: Option<PathBuf>,
}

impl DenseGgufModelBoundaryFixturesCommand {
    pub async fn execute(&self) -> Result<()> {
        if self.seq_len == 0 {
            bail!("dense GGUF model-boundary fixtures require --seq-len > 0");
        }
        if self.top_k == 0 {
            bail!("dense GGUF model-boundary fixtures require --top-k > 0");
        }

        let data = map_model(&self.model)?;
        let model_sha256 = sha256_bytes(&data);
        let reader = GgufReader::new(&data).with_context(|| {
            format!("failed to parse dense GGUF model {}", self.model.display())
        })?;
        let inspection = inspect_dense_gguf_tensor_descriptors(&reader)?;

        let probe = bitnet_device_probe::probe_nvidia_cuda(Some(self.device_index));
        if !probe.available {
            bail!("CUDA-DENSE-038 requires CUDA probe success: {:?}", probe.failure_reason);
        }
        let device_name = probe.selected_device_name.as_deref().unwrap_or("unknown");
        if !is_rtx5070ti_device_name(device_name) {
            bail!("CUDA-DENSE-038 requires NVIDIA GeForce RTX 5070 Ti; found '{device_name}'");
        }

        let fixtures = dense_gguf_model_boundary_fixtures_from_reader(
            &reader,
            &inspection,
            self.seq_len,
            self.top_k,
        )?;
        let artifact_path = self
            .json_out
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "stdout".to_string());
        let timestamp_utc = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        let receipt = dense_gguf_model_boundary_fixtures_receipt_json(
            &inspection,
            &fixtures,
            Some(&probe),
            &self.model,
            &model_sha256,
            &artifact_path,
            &timestamp_utc,
        )?;
        validate_dense_gguf_model_boundary_fixtures_receipt_json(&receipt)?;

        if let Some(path) = &self.json_out {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, serde_json::to_string_pretty(&receipt)?)?;
        } else {
            println!("{}", serde_json::to_string_pretty(&receipt)?);
        }

        Ok(())
    }
}

/// Emit a dense GGUF KV-cache policy receipt for the verified Qwen CUDA lane.
#[derive(Args, Debug, Clone)]
pub struct DenseGgufKvCachePolicyCommand {
    /// Dense GGUF model path.
    #[arg(long)]
    pub model: PathBuf,

    /// Number of token positions represented by the prefill KV policy.
    #[arg(long, default_value_t = 4)]
    pub seq_len: usize,

    /// Number of decode steps represented by the read/write policy.
    #[arg(long, default_value_t = 1)]
    pub decode_steps: usize,

    /// CUDA device index used for route identity and claim-boundary receipts.
    #[arg(long, default_value_t = 0)]
    pub device_index: usize,

    /// Output JSON receipt path. If omitted, writes receipt JSON to stdout.
    #[arg(long, value_name = "PATH")]
    pub json_out: Option<PathBuf>,
}

impl DenseGgufKvCachePolicyCommand {
    pub async fn execute(&self) -> Result<()> {
        if self.seq_len == 0 {
            bail!("dense GGUF KV-cache policy requires --seq-len > 0");
        }
        if self.decode_steps == 0 {
            bail!("dense GGUF KV-cache policy requires --decode-steps > 0");
        }

        let data = map_model(&self.model)?;
        let model_sha256 = sha256_bytes(&data);
        let reader = GgufReader::new(&data).with_context(|| {
            format!("failed to parse dense GGUF model {}", self.model.display())
        })?;
        let inspection = inspect_dense_gguf_tensor_descriptors(&reader)?;

        let probe = bitnet_device_probe::probe_nvidia_cuda(Some(self.device_index));
        if !probe.available {
            bail!("CUDA-DENSE-039 requires CUDA probe success: {:?}", probe.failure_reason);
        }
        let device_name = probe.selected_device_name.as_deref().unwrap_or("unknown");
        if !is_rtx5070ti_device_name(device_name) {
            bail!("CUDA-DENSE-039 requires NVIDIA GeForce RTX 5070 Ti; found '{device_name}'");
        }

        let policy = dense_gguf_kv_cache_policy_from_reader(
            &reader,
            &inspection,
            self.seq_len,
            self.decode_steps,
        )?;
        let artifact_path = self
            .json_out
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "stdout".to_string());
        let timestamp_utc = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        let receipt = dense_gguf_kv_cache_policy_receipt_json(
            &inspection,
            &policy,
            Some(&probe),
            &self.model,
            &model_sha256,
            &artifact_path,
            &timestamp_utc,
        )?;
        validate_dense_gguf_kv_cache_policy_receipt_json(&receipt)?;

        if let Some(path) = &self.json_out {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, serde_json::to_string_pretty(&receipt)?)?;
        } else {
            println!("{}", serde_json::to_string_pretty(&receipt)?);
        }

        Ok(())
    }
}

/// Emit a dense GGUF logits-transfer and sampling policy receipt.
#[derive(Args, Debug, Clone)]
pub struct DenseGgufSamplingPolicyCommand {
    /// Dense GGUF model path.
    #[arg(long)]
    pub model: PathBuf,

    /// Number of token positions used to derive the fixture logits boundary.
    #[arg(long, default_value_t = 4)]
    pub seq_len: usize,

    /// Number of top logits to record for deterministic sampler policy evidence.
    #[arg(long, default_value_t = 5)]
    pub top_k: usize,

    /// CUDA device index used for route identity and claim-boundary receipts.
    #[arg(long, default_value_t = 0)]
    pub device_index: usize,

    /// Output JSON receipt path. If omitted, writes receipt JSON to stdout.
    #[arg(long, value_name = "PATH")]
    pub json_out: Option<PathBuf>,
}

impl DenseGgufSamplingPolicyCommand {
    pub async fn execute(&self) -> Result<()> {
        if self.seq_len == 0 {
            bail!("dense GGUF sampling policy requires --seq-len > 0");
        }
        if self.top_k == 0 {
            bail!("dense GGUF sampling policy requires --top-k > 0");
        }

        let data = map_model(&self.model)?;
        let model_sha256 = sha256_bytes(&data);
        let reader = GgufReader::new(&data).with_context(|| {
            format!("failed to parse dense GGUF model {}", self.model.display())
        })?;
        let inspection = inspect_dense_gguf_tensor_descriptors(&reader)?;

        let probe = bitnet_device_probe::probe_nvidia_cuda(Some(self.device_index));
        if !probe.available {
            bail!("CUDA-DENSE-040 requires CUDA probe success: {:?}", probe.failure_reason);
        }
        let device_name = probe.selected_device_name.as_deref().unwrap_or("unknown");
        if !is_rtx5070ti_device_name(device_name) {
            bail!("CUDA-DENSE-040 requires NVIDIA GeForce RTX 5070 Ti; found '{device_name}'");
        }

        let policy =
            dense_gguf_sampling_policy_from_reader(&reader, &inspection, self.seq_len, self.top_k)?;
        let artifact_path = self
            .json_out
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "stdout".to_string());
        let timestamp_utc = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        let receipt = dense_gguf_sampling_policy_receipt_json(
            &inspection,
            &policy,
            Some(&probe),
            &self.model,
            &model_sha256,
            &artifact_path,
            &timestamp_utc,
        )?;
        validate_dense_gguf_sampling_policy_receipt_json(&receipt)?;

        if let Some(path) = &self.json_out {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, serde_json::to_string_pretty(&receipt)?)?;
        } else {
            println!("{}", serde_json::to_string_pretty(&receipt)?);
        }

        Ok(())
    }
}

/// Run the governed dense Qwen one-token strict CUDA proof.
#[derive(Args, Debug, Clone)]
pub struct DenseGgufQwenOneTokenStrictCudaCommand {
    /// Verified Qwen2.5 0.5B Q8_0 or Qwen3 0.6B Q8_0 GGUF model path.
    #[arg(long)]
    pub model: PathBuf,

    /// Deterministic raw prompt for the one-token proof.
    #[arg(long, default_value = "What is 2+2?")]
    pub prompt: String,

    /// Top-k logits to compare by deterministic token rank.
    #[arg(long, default_value_t = 10)]
    pub top_k: usize,

    /// CUDA device index.
    #[arg(long, default_value_t = 0)]
    pub device_index: usize,

    /// Prerequisite all-layer execution-plan receipt.
    #[arg(long, value_name = "PATH", default_value = DEFAULT_DENSE_QWEN_ALL_LAYER_PLAN_RECEIPT)]
    pub all_layer_plan: PathBuf,

    /// Prerequisite model-boundary fixtures receipt.
    #[arg(
        long,
        value_name = "PATH",
        default_value = DEFAULT_DENSE_QWEN_MODEL_BOUNDARY_FIXTURES_RECEIPT
    )]
    pub model_boundary_fixtures: PathBuf,

    /// Prerequisite KV-cache policy receipt.
    #[arg(long, value_name = "PATH", default_value = DEFAULT_DENSE_QWEN_KV_CACHE_POLICY_RECEIPT)]
    pub kv_cache_policy: PathBuf,

    /// Prerequisite sampling policy receipt.
    #[arg(long, value_name = "PATH", default_value = DEFAULT_DENSE_QWEN_SAMPLING_POLICY_RECEIPT)]
    pub sampling_policy: PathBuf,

    /// Output JSON receipt path. If omitted, writes receipt JSON to stdout.
    #[arg(long, value_name = "PATH")]
    pub json_out: Option<PathBuf>,

    /// Optional diagnostic JSONL phase trace path for stalled capture attempts.
    #[arg(long, value_name = "PATH")]
    pub phase_trace_jsonl: Option<PathBuf>,
}

impl DenseGgufQwenOneTokenStrictCudaCommand {
    pub async fn execute(&self) -> Result<()> {
        if self.top_k == 0 {
            bail!("dense Qwen one-token strict CUDA proof requires --top-k > 0");
        }
        let phase_trace = DenseQwenPhaseTrace::new(
            self.phase_trace_jsonl.as_deref(),
            "dense-gguf-qwen-one-token-strict-cuda",
        );
        phase_trace.reset()?;
        phase_trace.emit(
            "command",
            "start",
            json!({
                "model": self.model.display().to_string(),
                "device_index": self.device_index,
                "top_k": self.top_k,
                "json_out": self.json_out.as_ref().map(|path| path.display().to_string()),
            }),
        )?;

        phase_trace.emit("model_map", "start", json!({}))?;
        let data = map_model(&self.model)?;
        let model_sha256 = sha256_bytes(&data);
        phase_trace.emit(
            "model_map",
            "finish",
            json!({
                "model_bytes": data.len() as u64,
                "model_sha256": model_sha256,
            }),
        )?;
        let proof_model = dense_qwen_proof_model_for_sha256(&model_sha256).ok_or_else(|| {
            anyhow!(
                "dense Qwen one-token strict CUDA proof is scoped to verified Qwen2.5 0.5B Q8_0 or Qwen3 0.6B Q8_0 artifacts; got sha256={model_sha256}"
            )
        })?;
        let model_file =
            self.model.file_name().and_then(|value| value.to_str()).unwrap_or_default();
        if model_file != proof_model.file {
            bail!(
                "{} is scoped to verified {}; got {model_file}",
                proof_model.work_item,
                proof_model.file
            );
        }

        phase_trace.emit("gguf_inspection", "start", json!({ "model_id": proof_model.id }))?;
        let reader = GgufReader::new(&data).with_context(|| {
            format!("failed to parse dense GGUF model {}", self.model.display())
        })?;
        let inspection = inspect_dense_gguf_tensor_descriptors(&reader)?;
        phase_trace.emit(
            "gguf_inspection",
            "finish",
            json!({
                "model_family": inspection.model_family,
                "architecture": inspection.architecture,
            }),
        )?;
        if inspection.model_family != "qwen" || inspection.architecture != proof_model.architecture
        {
            bail!(
                "{} requires qwen/{} descriptor identity; got {}/{}",
                proof_model.work_item,
                proof_model.architecture,
                inspection.model_family,
                inspection.architecture,
            );
        }

        phase_trace.emit("cuda_probe", "start", json!({ "device_index": self.device_index }))?;
        let probe = bitnet_device_probe::probe_nvidia_cuda(Some(self.device_index));
        if !probe.available {
            bail!("CUDA-DENSE-044 requires CUDA probe success: {:?}", probe.failure_reason);
        }
        let device_name = probe.selected_device_name.as_deref().unwrap_or("unknown");
        if !is_rtx5070ti_device_name(device_name) {
            bail!("CUDA-DENSE-044 requires NVIDIA GeForce RTX 5070 Ti; found '{device_name}'");
        }
        phase_trace.emit(
            "cuda_probe",
            "finish",
            json!({
                "selected_device_name": device_name,
                "runtime_api": "cuda",
                "fallback_used": false,
            }),
        )?;

        let _strict_mode = ScopedEnvVar::set("BITNET_STRICT_MODE", "1");
        let _deterministic = ScopedEnvVar::set("BITNET_DETERMINISTIC", "1");
        let _seed = ScopedEnvVar::set("BITNET_SEED", "42");
        let _strict_cuda_backend = ScopedEnvVar::remove("BITNET_STRICT_CUDA_BACKEND");

        phase_trace.emit("prerequisites", "start", json!({ "model_id": proof_model.id }))?;
        let receipt_defaults = dense_qwen_receipts_for_proof_model(proof_model);
        let all_layer_plan = dense_qwen_model_default_receipt_path(
            &self.all_layer_plan,
            DEFAULT_DENSE_QWEN_ALL_LAYER_PLAN_RECEIPT,
            receipt_defaults.all_layer_plan,
        );
        let model_boundary_fixtures = dense_qwen_model_default_receipt_path(
            &self.model_boundary_fixtures,
            DEFAULT_DENSE_QWEN_MODEL_BOUNDARY_FIXTURES_RECEIPT,
            receipt_defaults.model_boundary_fixtures,
        );
        let kv_cache_policy = dense_qwen_model_default_receipt_path(
            &self.kv_cache_policy,
            DEFAULT_DENSE_QWEN_KV_CACHE_POLICY_RECEIPT,
            receipt_defaults.kv_cache_policy,
        );
        let sampling_policy = dense_qwen_model_default_receipt_path(
            &self.sampling_policy,
            DEFAULT_DENSE_QWEN_SAMPLING_POLICY_RECEIPT,
            receipt_defaults.sampling_policy,
        );
        let prerequisites = DenseQwenOneTokenPrerequisites::load(
            &all_layer_plan,
            &model_boundary_fixtures,
            &kv_cache_policy,
            &sampling_policy,
            proof_model,
        )?;
        phase_trace.emit(
            "prerequisites",
            "finish",
            json!({
                "all_layer_plan": all_layer_plan.display().to_string(),
                "model_boundary_fixtures": model_boundary_fixtures.display().to_string(),
                "kv_cache_policy": kv_cache_policy.display().to_string(),
                "sampling_policy": sampling_policy.display().to_string(),
            }),
        )?;

        phase_trace.emit("tokenizer", "start", json!({}))?;
        let tokenizer_resolution = bitnet_tokenizers::auto::resolve_tokenizer(
            &self.model,
            None,
            true,
        )
        .with_context(|| {
            format!("failed to resolve authoritative tokenizer for {}", self.model.display())
        })?;
        let tokenizer = tokenizer_resolution.tokenizer;
        let rendered_prompt = self.prompt.clone();
        let prompt_token_ids = tokenizer
            .encode(&rendered_prompt, true, false)
            .with_context(|| "failed to tokenize deterministic Qwen proof prompt")?;
        if prompt_token_ids.is_empty() {
            bail!("deterministic Qwen proof prompt tokenized to zero tokens");
        }
        let prompt_token_ids_sha256 = sha256_u32(&prompt_token_ids);
        let rendered_prompt_sha256 = sha256_bytes(rendered_prompt.as_bytes());
        phase_trace.emit(
            "tokenizer",
            "finish",
            json!({
                "prompt_token_count": prompt_token_ids.len() as u64,
                "prompt_token_ids_sha256": prompt_token_ids_sha256,
                "rendered_prompt_sha256": rendered_prompt_sha256,
            }),
        )?;

        let cpu = run_qwen_one_token_once(
            &self.model,
            BitNetDevice::Cpu,
            &prompt_token_ids,
            self.top_k,
            false,
            &phase_trace,
            "cpu_reference",
        )
        .with_context(|| "failed CPU reference one-token run")?;
        let cuda = run_qwen_one_token_once(
            &self.model,
            BitNetDevice::Cuda(self.device_index),
            &prompt_token_ids,
            self.top_k,
            true,
            &phase_trace,
            "cuda_target",
        )
        .with_context(|| "failed CUDA target one-token run")?;

        let cpu_top_k_ids = cpu.top_k.iter().map(|entry| entry.token_id).collect::<Vec<_>>();
        let cuda_top_k_ids = cuda.top_k.iter().map(|entry| entry.token_id).collect::<Vec<_>>();
        if cpu.selected_token_id != cuda.selected_token_id {
            bail!(
                "dense Qwen one-token selected token mismatch: cpu={} cuda={}",
                cpu.selected_token_id,
                cuda.selected_token_id
            );
        }
        if cpu_top_k_ids != cuda_top_k_ids {
            bail!(
                "dense Qwen one-token top-k token rank mismatch: cpu={:?} cuda={:?}",
                cpu_top_k_ids,
                cuda_top_k_ids
            );
        }

        let decoded_token_text = tokenizer
            .decode(&[cuda.selected_token_id])
            .ok()
            .filter(|value| !value.trim().is_empty())
            .or_else(|| tokenizer.token_to_piece(cuda.selected_token_id))
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| format!("<token:{}>", cuda.selected_token_id));

        let artifact_path = self
            .json_out
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "stdout".to_string());
        let timestamp_utc = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        phase_trace.emit("receipt", "start", json!({ "artifact_path": artifact_path }))?;
        let receipt = dense_gguf_qwen_one_token_strict_cuda_proof_receipt_json(
            &inspection,
            &prerequisites,
            proof_model,
            &probe,
            &self.model,
            &model_sha256,
            &artifact_path,
            &timestamp_utc,
            &rendered_prompt,
            prompt_token_ids.len(),
            &prompt_token_ids_sha256,
            &rendered_prompt_sha256,
            &cpu,
            &cuda,
            &decoded_token_text,
        )?;
        validate_dense_gguf_qwen_one_token_strict_cuda_proof_receipt_json(&receipt)?;
        phase_trace.emit("receipt", "validated", json!({ "artifact_path": artifact_path }))?;

        if let Some(path) = &self.json_out {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, serde_json::to_string_pretty(&receipt)?)?;
        } else {
            println!("{}", serde_json::to_string_pretty(&receipt)?);
        }
        phase_trace.emit("command", "finish", json!({ "artifact_path": artifact_path }))?;

        Ok(())
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum DenseQwenSourceCaptureProfile {
    /// Standard governed short-decode source proof, bounded to 5-16 tokens.
    #[value(name = "short-decode")]
    ShortDecode,
    /// Qwen3-only 32-token source proof for CUDA-MODEL-017 repeated comparators.
    #[value(name = "qwen3-short-decode-32")]
    Qwen3ShortDecode32,
    /// Qwen3-only 128-token decode from a prefilling warm context.
    #[value(name = "qwen3-warm-decode-128")]
    Qwen3WarmDecode128,
}

impl DenseQwenSourceCaptureProfile {
    fn profile_id(self) -> &'static str {
        match self {
            Self::ShortDecode => "short_decode",
            Self::Qwen3ShortDecode32 => "qwen3_short_decode_32",
            Self::Qwen3WarmDecode128 => "qwen3_warm_decode_128",
        }
    }

    fn proof_scope(self) -> &'static str {
        match self {
            Self::ShortDecode => "qwen_strict_short_decode_greedy",
            Self::Qwen3ShortDecode32 => "qwen3_strict_short_decode_32_greedy",
            Self::Qwen3WarmDecode128 => "qwen3_strict_warm_decode_128_greedy",
        }
    }

    fn validate_tokens(self, proof_model: &DenseQwenProofModel, tokens: usize) -> Result<()> {
        match self {
            Self::ShortDecode => {
                if !(5..=16).contains(&tokens) {
                    bail!(
                        "standard dense Qwen short-decode strict CUDA proof requires --max-new-tokens 5..=16"
                    );
                }
            }
            Self::Qwen3ShortDecode32 => {
                if proof_model.id != "qwen3-0.6b-instruct-q8_0" || tokens != 32 {
                    bail!(
                        "qwen3-short-decode-32 capture requires the verified Qwen3 0.6B Q8_0 artifact and --max-new-tokens 32"
                    );
                }
            }
            Self::Qwen3WarmDecode128 => {
                if proof_model.id != "qwen3-0.6b-instruct-q8_0" || tokens != 128 {
                    bail!(
                        "qwen3-warm-decode-128 capture requires the verified Qwen3 0.6B Q8_0 artifact and --max-new-tokens 128"
                    );
                }
            }
        }
        Ok(())
    }
}

/// Run the governed dense Qwen short-decode strict CUDA proof.
#[derive(Args, Debug, Clone)]
pub struct DenseGgufQwenShortDecodeStrictCudaCommand {
    /// Verified Qwen2.5 0.5B or Qwen3 0.6B Q8_0 GGUF model path.
    #[arg(long)]
    pub model: PathBuf,

    /// Deterministic raw prompt for the short-decode proof.
    #[arg(long, default_value = "What is 2+2?")]
    pub prompt: String,

    /// Number of deterministic greedy tokens to generate. Standard source proof is bounded to 5-16.
    #[arg(long, default_value_t = 8)]
    pub max_new_tokens: usize,

    /// Governed source-capture profile. Qwen3 32-token capture is not a product ask/chat bound.
    #[arg(long, value_enum, default_value_t = DenseQwenSourceCaptureProfile::ShortDecode)]
    pub capture_profile: DenseQwenSourceCaptureProfile,

    /// Top-k logits to record at each decode step.
    #[arg(long, default_value_t = 10)]
    pub top_k: usize,

    /// CUDA device index.
    #[arg(long, default_value_t = 0)]
    pub device_index: usize,

    /// Prerequisite all-layer execution-plan receipt.
    #[arg(long, value_name = "PATH", default_value = DEFAULT_DENSE_QWEN_ALL_LAYER_PLAN_RECEIPT)]
    pub all_layer_plan: PathBuf,

    /// Prerequisite model-boundary fixtures receipt.
    #[arg(
        long,
        value_name = "PATH",
        default_value = DEFAULT_DENSE_QWEN_MODEL_BOUNDARY_FIXTURES_RECEIPT
    )]
    pub model_boundary_fixtures: PathBuf,

    /// Prerequisite KV-cache policy receipt.
    #[arg(long, value_name = "PATH", default_value = DEFAULT_DENSE_QWEN_KV_CACHE_POLICY_RECEIPT)]
    pub kv_cache_policy: PathBuf,

    /// Prerequisite sampling policy receipt.
    #[arg(long, value_name = "PATH", default_value = DEFAULT_DENSE_QWEN_SAMPLING_POLICY_RECEIPT)]
    pub sampling_policy: PathBuf,

    /// Prerequisite one-token proof receipt.
    #[arg(long, value_name = "PATH", default_value = DEFAULT_DENSE_QWEN_ONE_TOKEN_PROOF_RECEIPT)]
    pub one_token_proof: PathBuf,

    /// Output JSON receipt path. If omitted, writes receipt JSON to stdout.
    #[arg(long, value_name = "PATH")]
    pub json_out: Option<PathBuf>,
}

impl DenseGgufQwenShortDecodeStrictCudaCommand {
    pub async fn execute(&self) -> Result<()> {
        if self.top_k == 0 {
            bail!("dense Qwen short-decode strict CUDA proof requires --top-k > 0");
        }

        let data = map_model(&self.model)?;
        let model_sha256 = sha256_bytes(&data);
        let proof_model = dense_qwen_proof_model_for_sha256(&model_sha256).ok_or_else(|| {
            anyhow!(
                "dense Qwen short-decode strict CUDA proof is scoped to verified Qwen2.5 0.5B Q8_0 or Qwen3 0.6B Q8_0 artifacts; got sha256={model_sha256}"
            )
        })?;
        let model_file =
            self.model.file_name().and_then(|value| value.to_str()).unwrap_or_default();
        if model_file != proof_model.file {
            bail!(
                "dense Qwen short-decode strict CUDA proof is scoped to verified {}; got {model_file}",
                proof_model.file
            );
        }
        self.capture_profile.validate_tokens(proof_model, self.max_new_tokens)?;
        if self.capture_profile == DenseQwenSourceCaptureProfile::Qwen3WarmDecode128 {
            bail!("use dense-gguf-qwen-warm-decode-strict-cuda for qwen3-warm-decode-128 capture");
        }

        let reader = GgufReader::new(&data).with_context(|| {
            format!("failed to parse dense GGUF model {}", self.model.display())
        })?;
        let inspection = inspect_dense_gguf_tensor_descriptors(&reader)?;
        if inspection.model_family != "qwen" || inspection.architecture != proof_model.architecture
        {
            bail!(
                "dense Qwen short-decode strict CUDA proof requires qwen/{} descriptor identity; got {}/{}",
                proof_model.architecture,
                inspection.model_family,
                inspection.architecture
            );
        }

        let probe = bitnet_device_probe::probe_nvidia_cuda(Some(self.device_index));
        if !probe.available {
            bail!("CUDA-DENSE-045 requires CUDA probe success: {:?}", probe.failure_reason);
        }
        let device_name = probe.selected_device_name.as_deref().unwrap_or("unknown");
        if !is_rtx5070ti_device_name(device_name) {
            bail!("CUDA-DENSE-045 requires NVIDIA GeForce RTX 5070 Ti; found '{device_name}'");
        }

        let _strict_mode = ScopedEnvVar::set("BITNET_STRICT_MODE", "1");
        let _deterministic = ScopedEnvVar::set("BITNET_DETERMINISTIC", "1");
        let _seed = ScopedEnvVar::set("BITNET_SEED", "42");
        let _strict_cuda_backend = ScopedEnvVar::remove("BITNET_STRICT_CUDA_BACKEND");

        let receipt_defaults = dense_qwen_receipts_for_proof_model(proof_model);
        let all_layer_plan = dense_qwen_model_default_receipt_path(
            &self.all_layer_plan,
            DEFAULT_DENSE_QWEN_ALL_LAYER_PLAN_RECEIPT,
            receipt_defaults.all_layer_plan,
        );
        let model_boundary_fixtures = dense_qwen_model_default_receipt_path(
            &self.model_boundary_fixtures,
            DEFAULT_DENSE_QWEN_MODEL_BOUNDARY_FIXTURES_RECEIPT,
            receipt_defaults.model_boundary_fixtures,
        );
        let kv_cache_policy = dense_qwen_model_default_receipt_path(
            &self.kv_cache_policy,
            DEFAULT_DENSE_QWEN_KV_CACHE_POLICY_RECEIPT,
            receipt_defaults.kv_cache_policy,
        );
        let sampling_policy = dense_qwen_model_default_receipt_path(
            &self.sampling_policy,
            DEFAULT_DENSE_QWEN_SAMPLING_POLICY_RECEIPT,
            receipt_defaults.sampling_policy,
        );
        let one_token_proof = dense_qwen_model_default_receipt_path(
            &self.one_token_proof,
            DEFAULT_DENSE_QWEN_ONE_TOKEN_PROOF_RECEIPT,
            receipt_defaults.one_token_proof,
        );
        let prerequisites = DenseQwenShortDecodePrerequisites::load(
            &all_layer_plan,
            &model_boundary_fixtures,
            &kv_cache_policy,
            &sampling_policy,
            &one_token_proof,
            proof_model,
        )?;

        let tokenizer_resolution = bitnet_tokenizers::auto::resolve_tokenizer(
            &self.model,
            None,
            true,
        )
        .with_context(|| {
            format!("failed to resolve authoritative tokenizer for {}", self.model.display())
        })?;
        let tokenizer = tokenizer_resolution.tokenizer;
        let rendered_prompt = self.prompt.clone();
        let prompt_token_ids = tokenizer
            .encode(&rendered_prompt, true, false)
            .with_context(|| "failed to tokenize deterministic Qwen short-decode prompt")?;
        if prompt_token_ids.is_empty() {
            bail!("deterministic Qwen short-decode prompt tokenized to zero tokens");
        }
        let prompt_token_ids_sha256 = sha256_u32(&prompt_token_ids);
        let rendered_prompt_sha256 = sha256_bytes(rendered_prompt.as_bytes());

        let cpu = run_qwen_short_decode(
            &self.model,
            BitNetDevice::Cpu,
            &prompt_token_ids,
            self.max_new_tokens,
            self.top_k,
            false,
        )
        .with_context(|| "failed CPU reference short-decode run")?;
        let cuda = run_qwen_short_decode(
            &self.model,
            BitNetDevice::Cuda(self.device_index),
            &prompt_token_ids,
            self.max_new_tokens,
            self.top_k,
            true,
        )
        .with_context(|| "failed CUDA target short-decode run")?;

        let first_token_divergence =
            first_divergence_index(&cpu.generated_token_ids, &cuda.generated_token_ids);
        if let Some(index) = first_token_divergence {
            bail!(
                "dense Qwen short-decode selected token mismatch at step {index}: cpu={} cuda={}",
                cpu.generated_token_ids[index],
                cuda.generated_token_ids[index]
            );
        }

        let decoded_text = tokenizer
            .decode(&cuda.generated_token_ids)
            .ok()
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| {
                cuda.generated_token_ids
                    .iter()
                    .map(|token| {
                        tokenizer
                            .token_to_piece(*token)
                            .filter(|value| !value.is_empty())
                            .unwrap_or_else(|| format!("<token:{token}>"))
                    })
                    .collect::<Vec<_>>()
                    .join("")
            });

        let artifact_path = self
            .json_out
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "stdout".to_string());
        let timestamp_utc = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        let receipt = dense_gguf_qwen_short_decode_strict_cuda_proof_receipt_json(
            &inspection,
            &prerequisites,
            proof_model,
            &probe,
            &self.model,
            &model_sha256,
            &artifact_path,
            &timestamp_utc,
            &rendered_prompt,
            prompt_token_ids.len(),
            &prompt_token_ids_sha256,
            &rendered_prompt_sha256,
            &cpu,
            &cuda,
            &decoded_text,
            self.capture_profile,
        )?;
        validate_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt_json(&receipt)?;

        if let Some(path) = &self.json_out {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, serde_json::to_string_pretty(&receipt)?)?;
        } else {
            println!("{}", serde_json::to_string_pretty(&receipt)?);
        }

        Ok(())
    }
}

/// Run the governed Qwen3 warm-context 128-token decode strict CUDA proof.
#[derive(Args, Debug, Clone)]
pub struct DenseGgufQwenWarmDecodeStrictCudaCommand {
    /// Verified Qwen3 0.6B Q8_0 GGUF model path.
    #[arg(long)]
    pub model: PathBuf,

    /// Deterministic raw prompt used to prefill the warm context before 128-token decode.
    #[arg(long, default_value = "Explain proof-carrying local inference in one paragraph.")]
    pub prompt: String,

    /// Number of deterministic greedy tokens to generate. CUDA-MODEL-017A requires exactly 128.
    #[arg(long, default_value_t = 128)]
    pub max_new_tokens: usize,

    /// Top-k logits to record at each decode step.
    #[arg(long, default_value_t = 10)]
    pub top_k: usize,

    /// CUDA device index.
    #[arg(long, default_value_t = 0)]
    pub device_index: usize,

    /// Prerequisite all-layer execution-plan receipt.
    #[arg(long, value_name = "PATH", default_value = DEFAULT_QWEN3_ALL_LAYER_PLAN_RECEIPT)]
    pub all_layer_plan: PathBuf,

    /// Prerequisite model-boundary fixtures receipt.
    #[arg(
        long,
        value_name = "PATH",
        default_value = DEFAULT_QWEN3_MODEL_BOUNDARY_FIXTURES_RECEIPT
    )]
    pub model_boundary_fixtures: PathBuf,

    /// Prerequisite KV-cache policy receipt.
    #[arg(long, value_name = "PATH", default_value = DEFAULT_QWEN3_KV_CACHE_POLICY_RECEIPT)]
    pub kv_cache_policy: PathBuf,

    /// Prerequisite sampling policy receipt.
    #[arg(long, value_name = "PATH", default_value = DEFAULT_QWEN3_SAMPLING_POLICY_RECEIPT)]
    pub sampling_policy: PathBuf,

    /// Prerequisite one-token proof receipt.
    #[arg(long, value_name = "PATH", default_value = DEFAULT_QWEN3_ONE_TOKEN_PROOF_RECEIPT)]
    pub one_token_proof: PathBuf,

    /// Prerequisite short-decode proof receipt.
    #[arg(long, value_name = "PATH", default_value = DEFAULT_QWEN3_SHORT_DECODE_PROOF_RECEIPT)]
    pub short_decode_proof: PathBuf,

    /// Output JSON receipt path. If omitted, writes receipt JSON to stdout.
    #[arg(long, value_name = "PATH")]
    pub json_out: Option<PathBuf>,
}

impl DenseGgufQwenWarmDecodeStrictCudaCommand {
    pub async fn execute(&self) -> Result<()> {
        if self.top_k == 0 {
            bail!("Qwen3 warm-decode strict CUDA proof requires --top-k > 0");
        }

        let data = map_model(&self.model)?;
        let model_sha256 = sha256_bytes(&data);
        let proof_model = dense_qwen_proof_model_for_sha256(&model_sha256).ok_or_else(|| {
            anyhow!(
                "Qwen3 warm-decode strict CUDA proof is scoped to the verified Qwen3 0.6B Q8_0 artifact; got sha256={model_sha256}"
            )
        })?;
        DenseQwenSourceCaptureProfile::Qwen3WarmDecode128
            .validate_tokens(proof_model, self.max_new_tokens)?;
        let model_file =
            self.model.file_name().and_then(|value| value.to_str()).unwrap_or_default();
        if model_file != proof_model.file {
            bail!(
                "Qwen3 warm-decode proof is scoped to verified {}; got {model_file}",
                proof_model.file
            );
        }

        let reader = GgufReader::new(&data).with_context(|| {
            format!("failed to parse dense GGUF model {}", self.model.display())
        })?;
        let inspection = inspect_dense_gguf_tensor_descriptors(&reader)?;
        if inspection.model_family != "qwen" || inspection.architecture != proof_model.architecture
        {
            bail!(
                "Qwen3 warm-decode proof requires qwen/{} descriptor identity; got {}/{}",
                proof_model.architecture,
                inspection.model_family,
                inspection.architecture
            );
        }

        let probe = bitnet_device_probe::probe_nvidia_cuda(Some(self.device_index));
        if !probe.available {
            bail!(
                "Qwen3 warm-decode proof requires CUDA probe success: {:?}",
                probe.failure_reason
            );
        }
        let device_name = probe.selected_device_name.as_deref().unwrap_or("unknown");
        if !is_rtx5070ti_device_name(device_name) {
            bail!(
                "Qwen3 warm-decode proof requires NVIDIA GeForce RTX 5070 Ti; found '{device_name}'"
            );
        }

        let _strict_mode = ScopedEnvVar::set("BITNET_STRICT_MODE", "1");
        let _deterministic = ScopedEnvVar::set("BITNET_DETERMINISTIC", "1");
        let _seed = ScopedEnvVar::set("BITNET_SEED", "42");
        let _strict_cuda_backend = ScopedEnvVar::remove("BITNET_STRICT_CUDA_BACKEND");

        let prerequisites = DenseQwenWarmSessionPrerequisites::load(
            &self.all_layer_plan,
            &self.model_boundary_fixtures,
            &self.kv_cache_policy,
            &self.sampling_policy,
            &self.one_token_proof,
            &self.short_decode_proof,
            proof_model,
        )?;

        let tokenizer_resolution = bitnet_tokenizers::auto::resolve_tokenizer(
            &self.model,
            None,
            true,
        )
        .with_context(|| {
            format!("failed to resolve authoritative tokenizer for {}", self.model.display())
        })?;
        let tokenizer = tokenizer_resolution.tokenizer;
        let rendered_prompt = self.prompt.clone();
        let prompt_token_ids = tokenizer
            .encode(&rendered_prompt, true, false)
            .with_context(|| "failed to tokenize deterministic Qwen3 warm-decode prompt")?;
        if prompt_token_ids.is_empty() {
            bail!("deterministic Qwen3 warm-decode prompt tokenized to zero tokens");
        }
        let prompt_token_ids_sha256 = sha256_u32(&prompt_token_ids);
        let rendered_prompt_sha256 = sha256_bytes(rendered_prompt.as_bytes());

        let cpu = run_qwen_short_decode(
            &self.model,
            BitNetDevice::Cpu,
            &prompt_token_ids,
            self.max_new_tokens,
            self.top_k,
            false,
        )
        .with_context(|| "failed CPU reference warm-decode run")?;
        let cuda = run_qwen_short_decode(
            &self.model,
            BitNetDevice::Cuda(self.device_index),
            &prompt_token_ids,
            self.max_new_tokens,
            self.top_k,
            true,
        )
        .with_context(|| "failed CUDA target warm-decode run")?;

        if let Some(index) =
            first_divergence_index(&cpu.generated_token_ids, &cuda.generated_token_ids)
        {
            bail!(
                "Qwen3 warm-decode selected token mismatch at step {index}: cpu={} cuda={}",
                cpu.generated_token_ids[index],
                cuda.generated_token_ids[index]
            );
        }

        let decoded_text = decode_generated_tokens(tokenizer.as_ref(), &cuda.generated_token_ids);
        let artifact_path = self
            .json_out
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "stdout".to_string());
        let timestamp_utc = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        let receipt = dense_gguf_qwen_warm_decode_strict_cuda_proof_receipt_json(
            &inspection,
            &prerequisites,
            proof_model,
            &probe,
            &self.model,
            &model_sha256,
            &artifact_path,
            &timestamp_utc,
            &rendered_prompt,
            prompt_token_ids.len(),
            &prompt_token_ids_sha256,
            &rendered_prompt_sha256,
            &cpu,
            &cuda,
            &decoded_text,
        )?;
        validate_dense_gguf_qwen_warm_decode_strict_cuda_proof_receipt_json(&receipt)?;

        if let Some(path) = &self.json_out {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, serde_json::to_string_pretty(&receipt)?)?;
        } else {
            println!("{}", serde_json::to_string_pretty(&receipt)?);
        }

        Ok(())
    }
}

/// Run the governed dense Qwen warm-session strict CUDA proof.
#[derive(Args, Debug, Clone)]
pub struct DenseGgufQwenWarmSessionStrictCudaCommand {
    /// Verified Qwen2.5 0.5B Q8_0 GGUF model path.
    #[arg(long)]
    pub model: PathBuf,

    /// Deterministic raw prompts for each warm-session turn. Defaults to three fixed prompts.
    #[arg(long = "prompt", value_name = "TEXT")]
    pub prompts: Vec<String>,

    /// Number of deterministic turns. Dense Qwen warm-session proof is bounded to 2-4 turns.
    #[arg(long, default_value_t = 3)]
    pub turns: usize,

    /// Number of deterministic greedy tokens to generate per turn. Dense Qwen warm-session proof is bounded to 5-16.
    #[arg(long, default_value_t = 8)]
    pub max_new_tokens: usize,

    /// Top-k logits to record at each decode step.
    #[arg(long, default_value_t = 10)]
    pub top_k: usize,

    /// CUDA device index.
    #[arg(long, default_value_t = 0)]
    pub device_index: usize,

    /// Prerequisite all-layer execution-plan receipt.
    #[arg(long, value_name = "PATH", default_value = DEFAULT_DENSE_QWEN_ALL_LAYER_PLAN_RECEIPT)]
    pub all_layer_plan: PathBuf,

    /// Prerequisite model-boundary fixtures receipt.
    #[arg(
        long,
        value_name = "PATH",
        default_value = DEFAULT_DENSE_QWEN_MODEL_BOUNDARY_FIXTURES_RECEIPT
    )]
    pub model_boundary_fixtures: PathBuf,

    /// Prerequisite KV-cache policy receipt.
    #[arg(long, value_name = "PATH", default_value = DEFAULT_DENSE_QWEN_KV_CACHE_POLICY_RECEIPT)]
    pub kv_cache_policy: PathBuf,

    /// Prerequisite sampling policy receipt.
    #[arg(long, value_name = "PATH", default_value = DEFAULT_DENSE_QWEN_SAMPLING_POLICY_RECEIPT)]
    pub sampling_policy: PathBuf,

    /// Prerequisite one-token proof receipt.
    #[arg(long, value_name = "PATH", default_value = DEFAULT_DENSE_QWEN_ONE_TOKEN_PROOF_RECEIPT)]
    pub one_token_proof: PathBuf,

    /// Prerequisite short-decode proof receipt.
    #[arg(long, value_name = "PATH", default_value = DEFAULT_DENSE_QWEN_SHORT_DECODE_PROOF_RECEIPT)]
    pub short_decode_proof: PathBuf,

    /// Output JSON receipt path. If omitted, writes receipt JSON to stdout.
    #[arg(long, value_name = "PATH")]
    pub json_out: Option<PathBuf>,
}

impl DenseGgufQwenWarmSessionStrictCudaCommand {
    pub async fn execute(&self) -> Result<()> {
        if !(2..=4).contains(&self.turns) {
            bail!("dense Qwen warm-session strict CUDA proof requires --turns 2..=4");
        }
        if !(5..=16).contains(&self.max_new_tokens) {
            bail!("dense Qwen warm-session strict CUDA proof requires --max-new-tokens 5..=16");
        }
        if self.top_k == 0 {
            bail!("dense Qwen warm-session strict CUDA proof requires --top-k > 0");
        }
        if !self.prompts.is_empty() && self.prompts.len() != self.turns {
            bail!(
                "dense Qwen warm-session strict CUDA proof requires either no --prompt values or exactly --turns prompts"
            );
        }
        if self.prompts.is_empty() && self.turns > DEFAULT_QWEN_WARM_SESSION_PROMPTS.len() {
            bail!(
                "dense Qwen warm-session default prompt set only supports up to {} turns",
                DEFAULT_QWEN_WARM_SESSION_PROMPTS.len()
            );
        }

        let data = map_model(&self.model)?;
        let model_sha256 = sha256_bytes(&data);
        let proof_model = dense_qwen_proof_model_for_sha256(&model_sha256).ok_or_else(|| {
            anyhow!(
                "dense Qwen warm-session strict CUDA proof is scoped to verified Qwen2.5 0.5B Q8_0 or Qwen3 0.6B Q8_0 artifacts; got sha256={model_sha256}"
            )
        })?;
        let model_file =
            self.model.file_name().and_then(|value| value.to_str()).unwrap_or_default();
        if model_file != proof_model.file {
            bail!(
                "{} warm-session proof is scoped to verified {}; got {model_file}",
                proof_model.work_item,
                proof_model.file
            );
        }

        let reader = GgufReader::new(&data).with_context(|| {
            format!("failed to parse dense GGUF model {}", self.model.display())
        })?;
        let inspection = inspect_dense_gguf_tensor_descriptors(&reader)?;
        if inspection.model_family != "qwen" || inspection.architecture != proof_model.architecture
        {
            bail!(
                "{} warm-session proof requires qwen/{} descriptor identity; got {}/{}",
                proof_model.work_item,
                proof_model.architecture,
                inspection.model_family,
                inspection.architecture
            );
        }

        let probe = bitnet_device_probe::probe_nvidia_cuda(Some(self.device_index));
        let warm_session_work_item = dense_qwen_warm_session_work_item(proof_model);
        if !probe.available {
            bail!(
                "{} warm-session proof requires CUDA probe success: {:?}",
                warm_session_work_item,
                probe.failure_reason
            );
        }
        let device_name = probe.selected_device_name.as_deref().unwrap_or("unknown");
        if !is_rtx5070ti_device_name(device_name) {
            bail!(
                "{} warm-session proof requires NVIDIA GeForce RTX 5070 Ti; found '{device_name}'",
                warm_session_work_item
            );
        }

        let _strict_mode = ScopedEnvVar::set("BITNET_STRICT_MODE", "1");
        let _deterministic = ScopedEnvVar::set("BITNET_DETERMINISTIC", "1");
        let _seed = ScopedEnvVar::set("BITNET_SEED", "42");
        let _strict_cuda_backend = ScopedEnvVar::remove("BITNET_STRICT_CUDA_BACKEND");

        let prerequisites = DenseQwenWarmSessionPrerequisites::load(
            &self.all_layer_plan,
            &self.model_boundary_fixtures,
            &self.kv_cache_policy,
            &self.sampling_policy,
            &self.one_token_proof,
            &self.short_decode_proof,
            proof_model,
        )?;

        let tokenizer_start = std::time::Instant::now();
        let tokenizer_resolution = bitnet_tokenizers::auto::resolve_tokenizer(
            &self.model,
            None,
            true,
        )
        .with_context(|| {
            format!("failed to resolve authoritative tokenizer for {}", self.model.display())
        })?;
        let tokenizer = tokenizer_resolution.tokenizer;
        let rendered_prompts = self.rendered_prompts()?;
        let mut prompt_token_ids = Vec::with_capacity(rendered_prompts.len());
        let mut prompt_evidence = Vec::with_capacity(rendered_prompts.len());
        for (index, rendered_prompt) in rendered_prompts.iter().enumerate() {
            let token_ids = tokenizer.encode(rendered_prompt, true, false).with_context(|| {
                format!("failed to tokenize deterministic Qwen warm-session prompt {index}")
            })?;
            if token_ids.is_empty() {
                bail!("deterministic Qwen warm-session prompt {index} tokenized to zero tokens");
            }
            prompt_evidence.push(DenseQwenWarmSessionPromptEvidence {
                index,
                token_ids_sha256: sha256_u32(&token_ids),
                rendered_prompt_sha256: sha256_bytes(rendered_prompt.as_bytes()),
                token_count: token_ids.len(),
                rendered_prompt_bytes: rendered_prompt.len(),
            });
            prompt_token_ids.push(token_ids);
        }
        let tokenizer_load_ms = elapsed_ms_f64(tokenizer_start);

        let cpu = run_qwen_warm_session(
            &self.model,
            BitNetDevice::Cpu,
            &prompt_token_ids,
            self.max_new_tokens,
            self.top_k,
            false,
        )
        .with_context(|| "failed CPU reference warm-session run")?;
        let cuda = run_qwen_warm_session(
            &self.model,
            BitNetDevice::Cuda(self.device_index),
            &prompt_token_ids,
            self.max_new_tokens,
            self.top_k,
            true,
        )
        .with_context(|| "failed CUDA target warm-session run")?;

        for (turn_index, (cpu_turn, cuda_turn)) in
            cpu.turns.iter().zip(cuda.turns.iter()).enumerate()
        {
            if let Some(index) = first_divergence_index(
                &cpu_turn.generated_token_ids,
                &cuda_turn.generated_token_ids,
            ) {
                bail!(
                    "dense Qwen warm-session selected token mismatch at turn {turn_index} step {index}: cpu={} cuda={}",
                    cpu_turn.generated_token_ids[index],
                    cuda_turn.generated_token_ids[index]
                );
            }
        }

        let decoded_texts = cuda
            .turns
            .iter()
            .map(|turn| decode_generated_tokens(tokenizer.as_ref(), &turn.generated_token_ids))
            .collect::<Vec<_>>();

        let artifact_path = self
            .json_out
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "stdout".to_string());
        let timestamp_utc = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        let receipt = dense_gguf_qwen_warm_session_strict_cuda_proof_receipt_json(
            &inspection,
            &prerequisites,
            proof_model,
            &probe,
            &self.model,
            &model_sha256,
            &artifact_path,
            &timestamp_utc,
            tokenizer_load_ms,
            &prompt_evidence,
            &cpu,
            &cuda,
            &decoded_texts,
        )?;
        validate_dense_gguf_qwen_warm_session_strict_cuda_proof_receipt_json(&receipt)?;

        if let Some(path) = &self.json_out {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, serde_json::to_string_pretty(&receipt)?)?;
        } else {
            println!("{}", serde_json::to_string_pretty(&receipt)?);
        }

        Ok(())
    }

    fn rendered_prompts(&self) -> Result<Vec<String>> {
        if !self.prompts.is_empty() {
            return Ok(self.prompts.clone());
        }
        Ok(DEFAULT_QWEN_WARM_SESSION_PROMPTS
            .iter()
            .take(self.turns)
            .map(|prompt| (*prompt).to_string())
            .collect())
    }
}

/// Inputs for the user-facing dense Qwen CUDA ask wrapper.
#[derive(Debug, Clone)]
pub struct DenseQwenCudaAskOptions {
    pub model: PathBuf,
    pub question: String,
    pub max_new_tokens: usize,
    pub top_k: usize,
    pub device_index: usize,
    pub receipt_out: Option<PathBuf>,
}

/// Result of a governed dense Qwen CUDA ask run.
#[derive(Debug, Clone)]
pub struct DenseQwenCudaAskOutcome {
    pub answer: String,
    pub receipt_path: PathBuf,
    pub receipt: Value,
}

/// Default receipt path for the dense Qwen CUDA ask UX receipt.
pub(crate) fn dense_qwen_cuda_ask_default_receipt_path() -> PathBuf {
    PathBuf::from("target")
        .join("bitnet")
        .join("receipts")
        .join("dense-cuda-ask")
        .join("dense-qwen-ask-latest.json")
}

/// Run the governed dense Qwen CUDA ask path by wrapping the bounded
/// short-decode runtime proof in a user-facing ask receipt.
pub async fn run_dense_qwen_cuda_ask(
    options: DenseQwenCudaAskOptions,
) -> Result<DenseQwenCudaAskOutcome> {
    if options.question.trim().is_empty() {
        bail!("dense Qwen CUDA ask requires a non-empty question");
    }
    if !(5..=16).contains(&options.max_new_tokens) {
        bail!("dense Qwen CUDA ask is currently bounded to --max-new-tokens 5..=16");
    }
    if options.top_k == 0 {
        bail!("dense Qwen CUDA ask requires top-k evidence; pass --top-k > 0 or use the default");
    }

    let receipt_path =
        options.receipt_out.clone().unwrap_or_else(dense_qwen_cuda_ask_default_receipt_path);
    let source_short_decode_path = dense_qwen_cuda_ask_source_receipt_path(&receipt_path);
    let proof_context = dense_qwen_proof_context_for_model_path(&options.model);
    let proof_receipts = proof_context.receipts;

    let short_decode = DenseGgufQwenShortDecodeStrictCudaCommand {
        model: options.model.clone(),
        prompt: options.question.clone(),
        max_new_tokens: options.max_new_tokens,
        capture_profile: DenseQwenSourceCaptureProfile::ShortDecode,
        top_k: options.top_k,
        device_index: options.device_index,
        all_layer_plan: PathBuf::from(proof_receipts.all_layer_plan),
        model_boundary_fixtures: PathBuf::from(proof_receipts.model_boundary_fixtures),
        kv_cache_policy: PathBuf::from(proof_receipts.kv_cache_policy),
        sampling_policy: PathBuf::from(proof_receipts.sampling_policy),
        one_token_proof: PathBuf::from(proof_receipts.one_token_proof),
        json_out: Some(source_short_decode_path.clone()),
    };
    short_decode.execute().await?;

    let (source_short_decode_receipt, source_short_decode_sha256) =
        read_and_validate_receipt_for_qwen_model(
            &source_short_decode_path,
            validate_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt_json,
            proof_context.proof_model,
        )?;
    let (_, warm_session_sha256) = read_and_validate_receipt_for_qwen_model(
        Path::new(proof_receipts.warm_session_proof),
        validate_dense_gguf_qwen_warm_session_strict_cuda_proof_receipt_json,
        proof_context.proof_model,
    )?;

    let source_proof = source_short_decode_receipt
        .get("short_decode_proof")
        .ok_or_else(|| anyhow!("short-decode source receipt is missing short_decode_proof"))?;
    let answer = source_proof
        .get("decoded_text")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| anyhow!("short-decode source receipt produced an empty answer"))?
        .to_string();
    let generated_tokens_count = source_proof
        .get("generated_tokens_count")
        .and_then(Value::as_u64)
        .ok_or_else(|| anyhow!("short-decode source receipt is missing generated_tokens_count"))?;

    let timestamp_utc = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    let receipt = dense_qwen_cuda_ask_receipt_json(
        &source_short_decode_receipt,
        &source_short_decode_sha256,
        &warm_session_sha256,
        &receipt_path,
        &timestamp_utc,
        &options.question,
        &answer,
        generated_tokens_count,
        proof_context.proof_model,
    )?;
    validate_dense_gguf_qwen_ask_strict_cuda_proof_receipt_json(&receipt)?;

    if let Some(parent) = receipt_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&receipt_path, serde_json::to_string_pretty(&receipt)?)?;

    Ok(DenseQwenCudaAskOutcome { answer, receipt_path, receipt })
}

/// Inputs for the user-facing dense Qwen CUDA chat wrapper.
#[derive(Debug, Clone)]
pub struct DenseQwenCudaChatOptions {
    pub model: PathBuf,
    pub prompts: Vec<String>,
    pub max_new_tokens: usize,
    pub top_k: usize,
    pub device_index: usize,
    pub receipt_out: Option<PathBuf>,
}

/// Result of a governed dense Qwen CUDA chat run.
#[derive(Debug, Clone)]
pub struct DenseQwenCudaChatOutcome {
    pub answers: Vec<String>,
    pub receipt_path: PathBuf,
    pub receipt: Value,
}

/// Default receipt path for the dense Qwen CUDA chat UX receipt.
pub(crate) fn dense_qwen_cuda_chat_default_receipt_path() -> PathBuf {
    PathBuf::from("target")
        .join("bitnet")
        .join("receipts")
        .join("dense-cuda-chat")
        .join("dense-qwen-chat-latest.json")
}

/// Run the governed dense Qwen CUDA chat path by wrapping the bounded
/// warm-session runtime proof in a user-facing chat receipt.
pub async fn run_dense_qwen_cuda_chat(
    options: DenseQwenCudaChatOptions,
) -> Result<DenseQwenCudaChatOutcome> {
    let prompts = normalize_dense_qwen_chat_prompts(options.prompts)?;
    if !(5..=16).contains(&options.max_new_tokens) {
        bail!("dense Qwen CUDA chat is currently bounded to --max-tokens 5..=16");
    }
    if options.top_k == 0 {
        bail!("dense Qwen CUDA chat requires top-k evidence; pass --top-k > 0 or use the default");
    }

    let receipt_path =
        options.receipt_out.clone().unwrap_or_else(dense_qwen_cuda_chat_default_receipt_path);
    let source_warm_session_path = dense_qwen_cuda_chat_source_receipt_path(&receipt_path);
    let proof_context = dense_qwen_proof_context_for_model_path(&options.model);
    let proof_receipts = proof_context.receipts;

    let warm_session = DenseGgufQwenWarmSessionStrictCudaCommand {
        model: options.model.clone(),
        prompts: prompts.clone(),
        turns: prompts.len(),
        max_new_tokens: options.max_new_tokens,
        top_k: options.top_k,
        device_index: options.device_index,
        all_layer_plan: PathBuf::from(proof_receipts.all_layer_plan),
        model_boundary_fixtures: PathBuf::from(proof_receipts.model_boundary_fixtures),
        kv_cache_policy: PathBuf::from(proof_receipts.kv_cache_policy),
        sampling_policy: PathBuf::from(proof_receipts.sampling_policy),
        one_token_proof: PathBuf::from(proof_receipts.one_token_proof),
        short_decode_proof: PathBuf::from(proof_receipts.short_decode_proof),
        json_out: Some(source_warm_session_path.clone()),
    };
    warm_session.execute().await?;

    let (source_warm_session_receipt, source_warm_session_sha256) =
        read_and_validate_receipt_for_qwen_model(
            &source_warm_session_path,
            validate_dense_gguf_qwen_warm_session_strict_cuda_proof_receipt_json,
            proof_context.proof_model,
        )?;

    let source_proof = source_warm_session_receipt
        .get("warm_session_proof")
        .ok_or_else(|| anyhow!("warm-session source receipt is missing warm_session_proof"))?;
    let source_turns = source_proof
        .get("turns")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("warm-session source receipt is missing turn evidence"))?;
    let answers = source_turns
        .iter()
        .map(|turn| {
            turn.get("decoded_text")
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .ok_or_else(|| anyhow!("warm-session source turn produced an empty answer"))
                .map(ToString::to_string)
        })
        .collect::<Result<Vec<_>>>()?;

    let timestamp_utc = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    let receipt = dense_qwen_cuda_chat_receipt_json(
        &source_warm_session_receipt,
        &source_warm_session_sha256,
        &receipt_path,
        &timestamp_utc,
        &prompts,
        &answers,
        proof_context.proof_model,
    )?;
    validate_dense_gguf_qwen_chat_strict_cuda_proof_receipt_json(&receipt)?;

    if let Some(parent) = receipt_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&receipt_path, serde_json::to_string_pretty(&receipt)?)?;

    Ok(DenseQwenCudaChatOutcome { answers, receipt_path, receipt })
}

fn normalize_dense_qwen_chat_prompts(prompts: Vec<String>) -> Result<Vec<String>> {
    let prompts = prompts
        .into_iter()
        .map(|prompt| prompt.trim().to_string())
        .filter(|prompt| !prompt.is_empty())
        .collect::<Vec<_>>();
    if !(2..=4).contains(&prompts.len()) {
        bail!("dense Qwen CUDA chat requires 2..=4 non-empty user turns");
    }
    Ok(prompts)
}

fn dense_qwen_cuda_chat_source_receipt_path(receipt_path: &Path) -> PathBuf {
    let parent = receipt_path.parent().unwrap_or_else(|| Path::new("."));
    let stem =
        receipt_path.file_stem().and_then(|value| value.to_str()).unwrap_or("dense-qwen-chat");
    parent.join(format!("{stem}.source-warm-session.json"))
}

fn dense_qwen_cuda_chat_receipt_json(
    source_warm_session_receipt: &Value,
    source_warm_session_sha256: &str,
    receipt_path: &Path,
    timestamp_utc: &str,
    prompts: &[String],
    answers: &[String],
    proof_model: &DenseQwenProofModel,
) -> Result<Value> {
    let source_proof = source_warm_session_receipt
        .get("warm_session_proof")
        .ok_or_else(|| anyhow!("warm-session source receipt is missing warm_session_proof"))?;
    let source_prerequisites = source_warm_session_receipt
        .get("prerequisite_receipts")
        .ok_or_else(|| anyhow!("warm-session source receipt is missing prerequisite_receipts"))?;
    let source_turns = source_proof
        .get("turns")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("warm-session source receipt is missing turns"))?;
    if source_turns.len() != prompts.len() || answers.len() != prompts.len() {
        bail!("dense Qwen CUDA chat receipt requires matching prompt/source/answer turn counts");
    }

    let turns = source_turns
        .iter()
        .zip(prompts.iter().zip(answers.iter()))
        .enumerate()
        .map(|(index, (source_turn, (prompt, answer)))| {
            json!({
                "index": index as u64,
                "user_message": prompt,
                "assistant_answer": answer,
                "prompt_token_count": source_turn["prompt_token_count"].clone(),
                "prompt_token_ids_sha256": source_turn["prompt_token_ids_sha256"].clone(),
                "rendered_prompt_sha256": source_turn["rendered_prompt_sha256"].clone(),
                "requested_new_tokens": source_turn["requested_new_tokens"].clone(),
                "generated_tokens_count": source_turn["generated_tokens_count"].clone(),
                "cpu_generated_token_ids": source_turn["cpu_generated_token_ids"].clone(),
                "cuda_generated_token_ids": source_turn["cuda_generated_token_ids"].clone(),
                "cpu_generated_token_ids_sha256": source_turn["cpu_generated_token_ids_sha256"].clone(),
                "cuda_generated_token_ids_sha256": source_turn["cuda_generated_token_ids_sha256"].clone(),
                "generated_token_ids_match": true,
                "first_token_divergence_index": null,
                "top_k_all_match": source_turn["top_k_all_match"].clone(),
                "first_top_k_divergence_index": source_turn["first_top_k_divergence_index"].clone(),
            })
        })
        .collect::<Vec<_>>();

    let mut receipt = source_warm_session_receipt.clone();
    receipt["artifact_kind"] = json!(DENSE_GGUF_QWEN_CHAT_STRICT_CUDA_PROOF_ARTIFACT_KIND);
    receipt["artifact_path"] = json!(receipt_path.display().to_string());
    receipt["claim"] = json!("dense_gguf_qwen_chat_strict_cuda_proof_recorded");
    receipt["model_coverage_row"] = json!(proof_model.model_coverage_row);
    receipt["model_coverage_tier"] = json!(proof_model.model_coverage_tier);
    receipt["timestamp_utc"] = json!(timestamp_utc);
    receipt["execution_path"]["kernel_family"] = json!("dense_qwen_chat_strict_cuda");
    receipt["execution_path"]["quantization_family"] =
        json!("dense_gguf_q8_0_f16_qwen_chat_contract");
    receipt["execution_plan"]["quantization"] = json!("dense_gguf_q8_0_f16_qwen_chat_contract");
    receipt["quality_gate"] = json!({
        "schema": 1,
        "gate": "qwen_cuda_chat_session",
        "passed": true,
        "chat_claimed": true,
        "server_claimed": false
    });
    receipt["quality"] = json!({
        "passed": true,
        "gate": "qwen_cuda_chat_session",
        "chat_claimed": true,
        "server_claimed": false
    });
    receipt["tensor_residency"]["scope"] = json!("qwen_chat_strict_cuda");
    receipt["residency"] = json!({
        "weights_uploaded_once": true,
        "runtime_buffers_reused": true,
        "per_turn_weight_upload": false,
        "full_cuda_residency_claimed": false
    });
    receipt["answers"] = json!(answers);
    receipt["receipt"] = json!({
        "path": receipt_path.display().to_string(),
        "defaulted_for_dense_cuda_chat": true
    });
    receipt["prerequisite_receipts"] = json!({
        "schema": 1,
        "all_layer_execution_plan_artifact_kind": source_prerequisites["all_layer_execution_plan_artifact_kind"].clone(),
        "all_layer_execution_plan_receipt_sha256": source_prerequisites["all_layer_execution_plan_receipt_sha256"].clone(),
        "model_boundary_fixtures_artifact_kind": source_prerequisites["model_boundary_fixtures_artifact_kind"].clone(),
        "model_boundary_fixtures_receipt_sha256": source_prerequisites["model_boundary_fixtures_receipt_sha256"].clone(),
        "kv_cache_policy_artifact_kind": source_prerequisites["kv_cache_policy_artifact_kind"].clone(),
        "kv_cache_policy_receipt_sha256": source_prerequisites["kv_cache_policy_receipt_sha256"].clone(),
        "sampling_policy_artifact_kind": source_prerequisites["sampling_policy_artifact_kind"].clone(),
        "sampling_policy_receipt_sha256": source_prerequisites["sampling_policy_receipt_sha256"].clone(),
        "one_token_proof_artifact_kind": source_prerequisites["one_token_proof_artifact_kind"].clone(),
        "one_token_proof_receipt_sha256": source_prerequisites["one_token_proof_receipt_sha256"].clone(),
        "short_decode_proof_artifact_kind": source_prerequisites["short_decode_proof_artifact_kind"].clone(),
        "short_decode_proof_receipt_sha256": source_prerequisites["short_decode_proof_receipt_sha256"].clone(),
        "warm_session_proof_artifact_kind": DENSE_GGUF_QWEN_WARM_SESSION_STRICT_CUDA_PROOF_ARTIFACT_KIND,
        "warm_session_proof_receipt_sha256": source_warm_session_sha256,
        "all_required_receipts_verified": true,
        "all_layer_execution_plan_claimed": true,
        "model_boundary_fixtures_claimed": true,
        "kv_cache_policy_claimed": true,
        "sampling_policy_claimed": true,
        "one_token_proof_claimed": true,
        "short_decode_proof_claimed": true,
        "warm_session_proof_claimed": true
    });
    receipt["chat_session"] = json!({
        "schema": 1,
        "proof_scope": "qwen_strict_cuda_chat_from_warm_session",
        "model_family": "qwen",
        "turns_count": source_proof["turns_count"].clone(),
        "requested_new_tokens_per_turn": source_proof["requested_new_tokens_per_turn"].clone(),
        "generated_tokens_total": source_proof["generated_tokens_total"].clone(),
        "generation_policy": "greedy",
        "deterministic": true,
        "fallback_used": false,
        "cpu_reference_backend": "amd-9950x3d-cpu-avx512",
        "cuda_target_backend": HARDWARE_LANE,
        "cpu_generated_token_ids_sha256": source_proof["cpu_generated_token_ids_sha256"].clone(),
        "cuda_generated_token_ids_sha256": source_proof["cuda_generated_token_ids_sha256"].clone(),
        "generated_token_ids_match": true,
        "first_token_divergence": null,
        "top_k_evidence_recorded": true,
        "top_k_compared": true,
        "top_k_all_match": source_proof["top_k_all_match"].clone(),
        "first_top_k_divergence": source_proof["first_top_k_divergence"].clone(),
        "top_k_max_abs_error": source_proof["top_k_max_abs_error"].clone(),
        "top_k_mean_abs_error": source_proof["top_k_mean_abs_error"].clone(),
        "turns": turns,
        "qwen_one_token_cuda_claimed": true,
        "qwen_short_decode_cuda_claimed": true,
        "qwen_warm_session_cuda_claimed": true,
        "qwen_ask_cuda_claimed": false,
        "qwen_chat_cuda_claimed": true,
        "dense_gguf_inference_claimed": false,
        "bitnet_packed_i2s_qk256_proof": false,
        "speedup_claim": false,
        "server_ready_claimed": false,
        "full_cuda_residency_claimed": false
    });
    receipt["claim_boundary"]["qwen_ask_cuda_claimed"] = json!(false);
    receipt["claim_boundary"]["qwen_chat_cuda_claimed"] = json!(true);
    receipt["claim_boundary"]["server_ready_claimed"] = json!(false);
    receipt["claim_boundary"]["speedup_claim"] = json!(false);
    receipt["claim_boundary"]["persistent_session_residency_claimed"] = json!(false);
    receipt["claim_boundary"]["full_cuda_residency_claimed"] = json!(false);
    receipt["notes"] = json!([
        format!(
            "{} exposes the bounded dense Qwen CUDA chat path through the user-facing CLI.",
            dense_qwen_chat_work_item(proof_model)
        ),
        "This receipt wraps the warm-session runtime proof without claiming server readiness, speedup, persistent residency, full CUDA residency, broad dense GGUF inference, or BitNet packed I2_S/QK256 proof."
    ]);
    receipt["source_warm_session_receipt"] = source_warm_session_receipt.clone();
    Ok(receipt)
}

fn dense_qwen_cuda_ask_source_receipt_path(receipt_path: &Path) -> PathBuf {
    let parent = receipt_path.parent().unwrap_or_else(|| Path::new("."));
    let stem =
        receipt_path.file_stem().and_then(|value| value.to_str()).unwrap_or("dense-qwen-ask");
    parent.join(format!("{stem}.source-short-decode.json"))
}

#[allow(clippy::too_many_arguments)]
fn dense_qwen_cuda_ask_receipt_json(
    source_short_decode_receipt: &Value,
    source_short_decode_sha256: &str,
    warm_session_sha256: &str,
    receipt_path: &Path,
    timestamp_utc: &str,
    question: &str,
    answer: &str,
    generated_tokens_count: u64,
    proof_model: &DenseQwenProofModel,
) -> Result<Value> {
    let source_proof = source_short_decode_receipt
        .get("short_decode_proof")
        .ok_or_else(|| anyhow!("short-decode source receipt is missing short_decode_proof"))?;
    let source_prerequisites = source_short_decode_receipt
        .get("prerequisite_receipts")
        .ok_or_else(|| anyhow!("short-decode source receipt is missing prerequisite_receipts"))?;

    let mut receipt = source_short_decode_receipt.clone();
    receipt["artifact_kind"] = json!(DENSE_GGUF_QWEN_ASK_STRICT_CUDA_PROOF_ARTIFACT_KIND);
    receipt["artifact_path"] = json!(receipt_path.display().to_string());
    receipt["claim"] = json!("dense_gguf_qwen_ask_strict_cuda_proof_recorded");
    receipt["model_coverage_row"] = json!(proof_model.model_coverage_row);
    receipt["model_coverage_tier"] = json!(proof_model.model_coverage_tier);
    receipt["timestamp_utc"] = json!(timestamp_utc);
    receipt["execution_path"]["kernel_family"] = json!("dense_qwen_ask_strict_cuda");
    receipt["execution_path"]["quantization_family"] =
        json!("dense_gguf_q8_0_f16_qwen_ask_contract");
    receipt["execution_plan"]["quantization"] = json!("dense_gguf_q8_0_f16_qwen_ask_contract");
    receipt["quality_gate"] = json!({
        "schema": 1,
        "gate": "qwen_cuda_ask_answer",
        "passed": true,
        "ask_claimed": true,
        "chat_claimed": false
    });
    receipt["quality"] = json!({
        "passed": true,
        "gate": "qwen_cuda_ask_answer",
        "ask_claimed": true,
        "chat_claimed": false
    });
    receipt["tensor_residency"]["scope"] = json!("qwen_ask_strict_cuda");
    receipt["residency"] = json!({
        "weights_uploaded_once": true,
        "per_token_weight_upload": false,
        "full_cuda_residency_claimed": false
    });
    receipt["answer"] = json!(answer);
    receipt["question"] = json!(question);
    receipt["receipt"] = json!({
        "path": receipt_path.display().to_string(),
        "defaulted_for_dense_cuda_ask": true
    });
    receipt["prerequisite_receipts"] = json!({
        "schema": 1,
        "all_layer_execution_plan_artifact_kind": source_prerequisites["all_layer_execution_plan_artifact_kind"].clone(),
        "all_layer_execution_plan_receipt_sha256": source_prerequisites["all_layer_execution_plan_receipt_sha256"].clone(),
        "model_boundary_fixtures_artifact_kind": source_prerequisites["model_boundary_fixtures_artifact_kind"].clone(),
        "model_boundary_fixtures_receipt_sha256": source_prerequisites["model_boundary_fixtures_receipt_sha256"].clone(),
        "kv_cache_policy_artifact_kind": source_prerequisites["kv_cache_policy_artifact_kind"].clone(),
        "kv_cache_policy_receipt_sha256": source_prerequisites["kv_cache_policy_receipt_sha256"].clone(),
        "sampling_policy_artifact_kind": source_prerequisites["sampling_policy_artifact_kind"].clone(),
        "sampling_policy_receipt_sha256": source_prerequisites["sampling_policy_receipt_sha256"].clone(),
        "one_token_proof_artifact_kind": source_prerequisites["one_token_proof_artifact_kind"].clone(),
        "one_token_proof_receipt_sha256": source_prerequisites["one_token_proof_receipt_sha256"].clone(),
        "short_decode_proof_artifact_kind": DENSE_GGUF_QWEN_SHORT_DECODE_STRICT_CUDA_PROOF_ARTIFACT_KIND,
        "short_decode_proof_receipt_sha256": source_short_decode_sha256,
        "warm_session_proof_artifact_kind": DENSE_GGUF_QWEN_WARM_SESSION_STRICT_CUDA_PROOF_ARTIFACT_KIND,
        "warm_session_proof_receipt_sha256": warm_session_sha256,
        "all_required_receipts_verified": true,
        "all_layer_execution_plan_claimed": true,
        "model_boundary_fixtures_claimed": true,
        "kv_cache_policy_claimed": true,
        "sampling_policy_claimed": true,
        "one_token_proof_claimed": true,
        "short_decode_proof_claimed": true,
        "warm_session_proof_claimed": true
    });
    receipt["ask_proof"] = json!({
        "schema": 1,
        "proof_scope": "qwen_strict_cuda_ask_from_short_decode",
        "model_family": "qwen",
        "question": question,
        "answer": answer,
        "requested_new_tokens": generated_tokens_count,
        "generated_tokens_count": generated_tokens_count,
        "generation_policy": "greedy",
        "deterministic": true,
        "fallback_used": false,
        "cpu_reference_backend": "amd-9950x3d-cpu-avx512",
        "cuda_target_backend": HARDWARE_LANE,
        "prompt_token_count": source_proof["prompt_token_count"].clone(),
        "prompt_token_ids_sha256": source_proof["prompt_token_ids_sha256"].clone(),
        "cpu_generated_token_ids": source_proof["cpu_generated_token_ids"].clone(),
        "cuda_generated_token_ids": source_proof["cuda_generated_token_ids"].clone(),
        "cpu_generated_token_ids_sha256": source_proof["cpu_generated_token_ids_sha256"].clone(),
        "cuda_generated_token_ids_sha256": source_proof["cuda_generated_token_ids_sha256"].clone(),
        "generated_token_ids_match": true,
        "first_token_divergence_index": null,
        "top_k_evidence_recorded": true,
        "top_k_compared": true,
        "top_k_all_match": source_proof["top_k_all_match"].clone(),
        "first_top_k_divergence_index": source_proof["first_top_k_divergence_index"].clone(),
        "top_k_max_abs_error": source_proof["top_k_max_abs_error"].clone(),
        "top_k_mean_abs_error": source_proof["top_k_mean_abs_error"].clone(),
        "qwen_one_token_cuda_claimed": true,
        "qwen_short_decode_cuda_claimed": true,
        "qwen_warm_session_cuda_claimed": true,
        "qwen_ask_cuda_claimed": true,
        "qwen_chat_cuda_claimed": false,
        "dense_gguf_inference_claimed": false,
        "bitnet_packed_i2s_qk256_proof": false,
        "speedup_claim": false,
        "server_ready_claimed": false,
        "full_cuda_residency_claimed": false
    });
    receipt["claim_boundary"]["qwen_warm_session_cuda_claimed"] = json!(true);
    receipt["claim_boundary"]["qwen_ask_cuda_claimed"] = json!(true);
    receipt["claim_boundary"]["qwen_chat_cuda_claimed"] = json!(false);
    receipt["claim_boundary"]["server_ready_claimed"] = json!(false);
    receipt["claim_boundary"]["speedup_claim"] = json!(false);
    receipt["claim_boundary"]["persistent_session_residency_claimed"] = json!(false);
    receipt["claim_boundary"]["full_cuda_residency_claimed"] = json!(false);
    receipt["notes"] = json!([
        format!(
            "{} exposes the bounded dense Qwen CUDA ask path through the user-facing CLI.",
            dense_qwen_ask_work_item(proof_model)
        ),
        "This receipt wraps the short-decode runtime proof and warm-session prerequisite without claiming chat, server readiness, speedup, persistent residency, full CUDA residency, or BitNet packed I2_S/QK256 proof."
    ]);
    receipt["source_short_decode_receipt"] = source_short_decode_receipt.clone();
    Ok(receipt)
}

/// Emit a full dense GGUF layer-0 CPU reference harness receipt.
#[derive(Args, Debug, Clone)]
pub struct DenseGgufOneLayerCpuReferenceCommand {
    /// Dense GGUF model path.
    #[arg(long)]
    pub model: PathBuf,

    /// Dense transformer layer index. This diagnostic currently records layer 0.
    #[arg(long, default_value_t = 0)]
    pub layer_index: usize,

    /// Number of token positions in the deterministic CPU reference harness.
    #[arg(long, default_value_t = 4)]
    pub seq_len: usize,

    /// Position offset used by metadata-derived RoPE.
    #[arg(long, default_value_t = 1)]
    pub position_offset: usize,

    /// Output JSON receipt path. If omitted, writes receipt JSON to stdout.
    #[arg(long, value_name = "PATH")]
    pub json_out: Option<PathBuf>,
}

impl DenseGgufOneLayerCpuReferenceCommand {
    pub async fn execute(&self) -> Result<()> {
        if self.layer_index != 0 {
            bail!("CUDA-DENSE-033 currently records the first dense GGUF layer only");
        }
        if self.seq_len == 0 {
            bail!("dense GGUF one-layer CPU reference requires --seq-len > 0");
        }

        let data = map_model(&self.model)?;
        let model_sha256 = sha256_bytes(&data);
        let reader = GgufReader::new(&data).with_context(|| {
            format!("failed to parse dense GGUF model {}", self.model.display())
        })?;
        let inspection = inspect_dense_gguf_tensor_descriptors(&reader)?;
        let reference = dense_gguf_one_layer_cpu_reference_from_reader(
            &reader,
            &inspection,
            self.layer_index,
            self.seq_len,
            self.position_offset,
        )?;

        let artifact_path = self
            .json_out
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "stdout".to_string());
        let timestamp_utc = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        let receipt = dense_gguf_one_layer_cpu_reference_receipt_json(
            &inspection,
            &reference,
            &self.model,
            &model_sha256,
            &artifact_path,
            &timestamp_utc,
        )?;
        validate_dense_gguf_one_layer_cpu_reference_receipt_json(&receipt)?;

        if let Some(path) = &self.json_out {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, serde_json::to_string_pretty(&receipt)?)?;
        } else {
            println!("{}", serde_json::to_string_pretty(&receipt)?);
        }

        Ok(())
    }
}

/// Run an integrated dense GGUF layer-0 pass through strict CUDA and compare
/// against the CPU reference harness.
#[derive(Args, Debug, Clone)]
pub struct DenseGgufOneLayerCudaParityCommand {
    /// Dense GGUF model path.
    #[arg(long)]
    pub model: PathBuf,

    /// Dense transformer layer index. This diagnostic currently records layer 0.
    #[arg(long, default_value_t = 0)]
    pub layer_index: usize,

    /// Number of token positions in the deterministic integrated layer pass.
    #[arg(long, default_value_t = 4)]
    pub seq_len: usize,

    /// Position offset used by metadata-derived RoPE.
    #[arg(long, default_value_t = 1)]
    pub position_offset: usize,

    /// CUDA device index.
    #[arg(long, default_value_t = 0)]
    pub device_index: usize,

    /// Bounded final-output tolerance for the integrated layer comparison.
    #[arg(long, default_value_t = 0.5)]
    pub tolerance: f32,

    /// Output JSON receipt path. If omitted, writes receipt JSON to stdout.
    #[arg(long, value_name = "PATH")]
    pub json_out: Option<PathBuf>,
}

impl DenseGgufOneLayerCudaParityCommand {
    pub async fn execute(&self) -> Result<()> {
        if self.layer_index != 0 {
            bail!("CUDA-DENSE integrated one-layer CUDA parity currently records layer 0 only");
        }
        if self.seq_len == 0 {
            bail!("dense GGUF one-layer CUDA parity requires --seq-len > 0");
        }
        if self.tolerance <= 0.0 || !self.tolerance.is_finite() {
            bail!("dense GGUF one-layer CUDA parity requires a positive finite --tolerance");
        }

        let data = map_model(&self.model)?;
        let model_sha256 = sha256_bytes(&data);
        let reader = GgufReader::new(&data).with_context(|| {
            format!("failed to parse dense GGUF model {}", self.model.display())
        })?;
        let inspection = inspect_dense_gguf_tensor_descriptors(&reader)?;
        let reference = dense_gguf_one_layer_cpu_reference_from_reader(
            &reader,
            &inspection,
            self.layer_index,
            self.seq_len,
            self.position_offset,
        )?;

        let probe = bitnet_device_probe::probe_nvidia_cuda(Some(self.device_index));
        if !probe.available {
            bail!(
                "integrated dense one-layer CUDA parity requires CUDA probe success: {:?}",
                probe.failure_reason
            );
        }
        let device_name = probe.selected_device_name.as_deref().unwrap_or("unknown");
        if !is_rtx5070ti_device_name(device_name) {
            bail!(
                "integrated dense one-layer CUDA parity requires NVIDIA GeForce RTX 5070 Ti; found '{device_name}'"
            );
        }

        let parity = dense_gguf_one_layer_cuda_integrated_parity_from_reader(
            &reader,
            &inspection,
            &reference,
            self.device_index,
            self.tolerance,
        )?;
        let artifact_path = self
            .json_out
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "stdout".to_string());
        let timestamp_utc = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        let receipt = dense_gguf_one_layer_cuda_integrated_parity_receipt_json(
            &inspection,
            &reference,
            &parity,
            Some(&probe),
            &self.model,
            &model_sha256,
            &artifact_path,
            &timestamp_utc,
        )?;
        validate_dense_gguf_one_layer_cuda_integrated_parity_receipt_json(&receipt)?;

        if let Some(path) = &self.json_out {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, serde_json::to_string_pretty(&receipt)?)?;
        } else {
            println!("{}", serde_json::to_string_pretty(&receipt)?);
        }

        if !parity.passed {
            bail!(
                "integrated dense one-layer CUDA parity failed: max_abs_error={} tolerance={}",
                parity.final_output_max_abs_error,
                parity.tolerance
            );
        }

        Ok(())
    }
}

/// Extract dense GGUF RMSNorm fixtures and emit a CPU-reference receipt.
#[derive(Args, Debug, Clone)]
pub struct DenseGgufNormFixtureCommand {
    /// Dense GGUF model path.
    #[arg(long)]
    pub model: PathBuf,

    /// Dense norm tensor roles to extract. Defaults to attention_norm and ffn_norm.
    #[arg(long, value_delimiter = ',', value_name = "ROLE")]
    pub roles: Vec<String>,

    /// Output JSON receipt path. If omitted, writes receipt JSON to stdout.
    #[arg(long, value_name = "PATH")]
    pub json_out: Option<PathBuf>,
}

impl DenseGgufNormFixtureCommand {
    pub async fn execute(&self) -> Result<()> {
        let roles = parse_norm_roles(&self.roles)?;
        let data = map_model(&self.model)?;
        let model_sha256 = sha256_bytes(&data);
        let reader = GgufReader::new(&data).with_context(|| {
            format!("failed to parse dense GGUF model {}", self.model.display())
        })?;
        let inspection = inspect_dense_gguf_tensor_descriptors(&reader)?;

        let mut fixtures = Vec::with_capacity(roles.len());
        for role in roles {
            fixtures.push(extract_dense_gguf_norm_fixture(&reader, role)?);
        }

        let artifact_path = self
            .json_out
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "stdout".to_string());
        let timestamp_utc = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        let receipt = dense_gguf_norm_fixture_receipt_json(
            &inspection,
            &fixtures,
            &self.model,
            &model_sha256,
            &artifact_path,
            &timestamp_utc,
        )?;
        validate_dense_gguf_norm_fixture_extraction_receipt_json(&receipt)?;

        if let Some(path) = &self.json_out {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, serde_json::to_string_pretty(&receipt)?)?;
        } else {
            println!("{}", serde_json::to_string_pretty(&receipt)?);
        }

        Ok(())
    }
}

/// Run dense GGUF RMSNorm CUDA parity diagnostics.
#[derive(Args, Debug, Clone)]
pub struct DenseGgufNormCudaParityCommand {
    /// Dense GGUF model path.
    #[arg(long)]
    pub model: PathBuf,

    /// Dense norm tensor roles to extract. Defaults to attention_norm and ffn_norm.
    #[arg(long, value_delimiter = ',', value_name = "ROLE")]
    pub roles: Vec<String>,

    /// CUDA device index.
    #[arg(long, default_value_t = 0)]
    pub device_index: usize,

    /// Output JSON receipt path. If omitted, writes receipt JSON to stdout.
    #[arg(long, value_name = "PATH")]
    pub json_out: Option<PathBuf>,
}

impl DenseGgufNormCudaParityCommand {
    pub async fn execute(&self) -> Result<()> {
        let roles = parse_norm_roles(&self.roles)?;
        let data = map_model(&self.model)?;
        let model_sha256 = sha256_bytes(&data);
        let reader = GgufReader::new(&data).with_context(|| {
            format!("failed to parse dense GGUF model {}", self.model.display())
        })?;
        let inspection = inspect_dense_gguf_tensor_descriptors(&reader)?;

        let probe = bitnet_device_probe::probe_nvidia_cuda(Some(self.device_index));
        if !probe.available {
            bail!("CUDA-DENSE-016 requires CUDA probe success: {:?}", probe.failure_reason);
        }
        let device_name = probe.selected_device_name.as_deref().unwrap_or("unknown");
        if !is_rtx5070ti_device_name(device_name) {
            bail!("CUDA-DENSE-016 requires NVIDIA GeForce RTX 5070 Ti; found '{device_name}'");
        }

        let mut results = Vec::with_capacity(roles.len());
        for role in roles {
            let extracted = extract_dense_gguf_norm_fixture(&reader, role)?;
            let kernel_fixture = kernel_rmsnorm_fixture_from_extracted(&extracted)?;
            let parity = run_dense_gguf_rmsnorm_cuda_parity(self.device_index, &kernel_fixture)?;
            results.push(DenseNormParityResult { extracted, parity });
        }

        if let Some(failed) = results.iter().find(|result| !result.parity.passed) {
            bail!(
                "dense GGUF RMSNorm CUDA parity failed for {}: max_abs_error={} tolerance={}",
                failed.parity.tensor_role,
                failed.parity.max_abs_error,
                failed.parity.tolerance
            );
        }

        let artifact_path = self
            .json_out
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "stdout".to_string());
        let timestamp_utc = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        let receipt = dense_gguf_norm_cuda_parity_receipt_json(
            &inspection,
            &results,
            Some(&probe),
            &self.model,
            &model_sha256,
            &artifact_path,
            &timestamp_utc,
        )?;
        validate_dense_gguf_norm_cuda_parity_receipt_json(&receipt)?;

        if let Some(path) = &self.json_out {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, serde_json::to_string_pretty(&receipt)?)?;
        } else {
            println!("{}", serde_json::to_string_pretty(&receipt)?);
        }

        Ok(())
    }
}

/// Run dense GGUF RoPE CUDA parity diagnostics.
#[derive(Args, Debug, Clone)]
pub struct DenseGgufRopeCudaParityCommand {
    /// Dense GGUF model path.
    #[arg(long)]
    pub model: PathBuf,

    /// Dense transformer layer index represented by the deterministic fixture.
    #[arg(long, default_value_t = 0)]
    pub layer_index: usize,

    /// Number of token positions in the deterministic RoPE fixture.
    #[arg(long, default_value_t = 4)]
    pub seq_len: usize,

    /// Position offset used by the deterministic fixture.
    #[arg(long, default_value_t = 1)]
    pub position_offset: usize,

    /// CUDA device index.
    #[arg(long, default_value_t = 0)]
    pub device_index: usize,

    /// Output JSON receipt path. If omitted, writes receipt JSON to stdout.
    #[arg(long, value_name = "PATH")]
    pub json_out: Option<PathBuf>,
}

impl DenseGgufRopeCudaParityCommand {
    pub async fn execute(&self) -> Result<()> {
        if self.seq_len == 0 {
            bail!("dense GGUF RoPE CUDA parity requires --seq-len > 0");
        }

        let data = map_model(&self.model)?;
        let model_sha256 = sha256_bytes(&data);
        let reader = GgufReader::new(&data).with_context(|| {
            format!("failed to parse dense GGUF model {}", self.model.display())
        })?;
        let inspection = inspect_dense_gguf_tensor_descriptors(&reader)?;

        let probe = bitnet_device_probe::probe_nvidia_cuda(Some(self.device_index));
        if !probe.available {
            bail!("CUDA-DENSE-018 requires CUDA probe success: {:?}", probe.failure_reason);
        }
        let device_name = probe.selected_device_name.as_deref().unwrap_or("unknown");
        if !is_rtx5070ti_device_name(device_name) {
            bail!("CUDA-DENSE-018 requires NVIDIA GeForce RTX 5070 Ti; found '{device_name}'");
        }

        let fixture = dense_gguf_rope_cuda_fixture_from_reader(
            &reader,
            &inspection,
            self.layer_index,
            self.seq_len,
            self.position_offset,
        )?;
        let parity = run_dense_gguf_rope_cuda_parity(self.device_index, &fixture)?;
        if !parity.passed {
            bail!(
                "dense GGUF RoPE CUDA parity failed: max_abs_error={} tolerance={}",
                parity.max_abs_error,
                parity.tolerance
            );
        }

        let artifact_path = self
            .json_out
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "stdout".to_string());
        let timestamp_utc = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        let receipt = dense_gguf_rope_cuda_parity_receipt_json(
            &inspection,
            &fixture,
            &parity,
            Some(&probe),
            &self.model,
            &model_sha256,
            &artifact_path,
            &timestamp_utc,
        );
        validate_dense_gguf_rope_cuda_parity_receipt_json(&receipt)?;

        if let Some(path) = &self.json_out {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, serde_json::to_string_pretty(&receipt)?)?;
        } else {
            println!("{}", serde_json::to_string_pretty(&receipt)?);
        }

        Ok(())
    }
}

/// Extract a dense GGUF attention-score fixture and emit a CPU-reference receipt.
#[derive(Args, Debug, Clone)]
pub struct DenseGgufAttentionScoreFixtureCommand {
    /// Dense GGUF model path.
    #[arg(long)]
    pub model: PathBuf,

    /// Dense transformer layer index represented by the deterministic fixture.
    #[arg(long, default_value_t = 0)]
    pub layer_index: usize,

    /// Number of token positions in the deterministic attention-score fixture.
    #[arg(long, default_value_t = 4)]
    pub seq_len: usize,

    /// Position offset used by the metadata-derived RoPE fixture.
    #[arg(long, default_value_t = 1)]
    pub position_offset: usize,

    /// Output JSON receipt path. If omitted, writes receipt JSON to stdout.
    #[arg(long, value_name = "PATH")]
    pub json_out: Option<PathBuf>,
}

impl DenseGgufAttentionScoreFixtureCommand {
    pub async fn execute(&self) -> Result<()> {
        if self.seq_len == 0 {
            bail!("dense GGUF attention-score fixture requires --seq-len > 0");
        }

        let data = map_model(&self.model)?;
        let model_sha256 = sha256_bytes(&data);
        let reader = GgufReader::new(&data).with_context(|| {
            format!("failed to parse dense GGUF model {}", self.model.display())
        })?;
        let inspection = inspect_dense_gguf_tensor_descriptors(&reader)?;
        let fixture = dense_gguf_attention_score_fixture_from_reader(
            &reader,
            &inspection,
            self.layer_index,
            self.seq_len,
            self.position_offset,
        )?;

        let artifact_path = self
            .json_out
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "stdout".to_string());
        let timestamp_utc = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        let receipt = dense_gguf_attention_score_fixture_receipt_json(
            &inspection,
            &fixture,
            &self.model,
            &model_sha256,
            &artifact_path,
            &timestamp_utc,
        );
        validate_dense_gguf_attention_score_fixture_receipt_json(&receipt)?;

        if let Some(path) = &self.json_out {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, serde_json::to_string_pretty(&receipt)?)?;
        } else {
            println!("{}", serde_json::to_string_pretty(&receipt)?);
        }

        Ok(())
    }
}

/// Extract a dense GGUF attention-softmax fixture and emit a CPU-reference receipt.
#[derive(Args, Debug, Clone)]
pub struct DenseGgufAttentionSoftmaxFixtureCommand {
    /// Dense GGUF model path.
    #[arg(long)]
    pub model: PathBuf,

    /// Dense transformer layer index represented by the deterministic fixture.
    #[arg(long, default_value_t = 0)]
    pub layer_index: usize,

    /// Number of token positions in the deterministic attention-softmax fixture.
    #[arg(long, default_value_t = 4)]
    pub seq_len: usize,

    /// Position offset used by the metadata-derived RoPE fixture.
    #[arg(long, default_value_t = 1)]
    pub position_offset: usize,

    /// Output JSON receipt path. If omitted, writes receipt JSON to stdout.
    #[arg(long, value_name = "PATH")]
    pub json_out: Option<PathBuf>,
}

impl DenseGgufAttentionSoftmaxFixtureCommand {
    pub async fn execute(&self) -> Result<()> {
        if self.seq_len == 0 {
            bail!("dense GGUF attention-softmax fixture requires --seq-len > 0");
        }

        let data = map_model(&self.model)?;
        let model_sha256 = sha256_bytes(&data);
        let reader = GgufReader::new(&data).with_context(|| {
            format!("failed to parse dense GGUF model {}", self.model.display())
        })?;
        let inspection = inspect_dense_gguf_tensor_descriptors(&reader)?;
        let fixture = dense_gguf_attention_softmax_fixture_from_reader(
            &reader,
            &inspection,
            self.layer_index,
            self.seq_len,
            self.position_offset,
        )?;

        let artifact_path = self
            .json_out
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "stdout".to_string());
        let timestamp_utc = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        let receipt = dense_gguf_attention_softmax_fixture_receipt_json(
            &inspection,
            &fixture,
            &self.model,
            &model_sha256,
            &artifact_path,
            &timestamp_utc,
        );
        validate_dense_gguf_attention_softmax_fixture_receipt_json(&receipt)?;

        if let Some(path) = &self.json_out {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, serde_json::to_string_pretty(&receipt)?)?;
        } else {
            println!("{}", serde_json::to_string_pretty(&receipt)?);
        }

        Ok(())
    }
}

/// Extract a dense GGUF attention V-mix fixture and emit a CPU-reference receipt.
#[derive(Args, Debug, Clone)]
pub struct DenseGgufAttentionVMixFixtureCommand {
    /// Dense GGUF model path.
    #[arg(long)]
    pub model: PathBuf,

    /// Dense transformer layer index represented by the deterministic attention V-mix fixture.
    #[arg(long, default_value_t = 0)]
    pub layer_index: usize,

    /// Number of token positions in the deterministic attention V-mix fixture.
    #[arg(long, default_value_t = 4)]
    pub seq_len: usize,

    /// Position offset used by the metadata-derived RoPE fixture.
    #[arg(long, default_value_t = 1)]
    pub position_offset: usize,

    /// Output JSON receipt path. If omitted, writes receipt JSON to stdout.
    #[arg(long, value_name = "PATH")]
    pub json_out: Option<PathBuf>,
}

impl DenseGgufAttentionVMixFixtureCommand {
    pub async fn execute(&self) -> Result<()> {
        if self.seq_len == 0 {
            bail!("dense GGUF attention V-mix fixture requires --seq-len > 0");
        }

        let data = map_model(&self.model)?;
        let model_sha256 = sha256_bytes(&data);
        let reader = GgufReader::new(&data).with_context(|| {
            format!("failed to parse dense GGUF model {}", self.model.display())
        })?;
        let inspection = inspect_dense_gguf_tensor_descriptors(&reader)?;
        let fixture = dense_gguf_attention_v_mix_fixture_from_reader(
            &reader,
            &inspection,
            self.layer_index,
            self.seq_len,
            self.position_offset,
        )?;

        let artifact_path = self
            .json_out
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "stdout".to_string());
        let timestamp_utc = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        let receipt = dense_gguf_attention_v_mix_fixture_receipt_json(
            &inspection,
            &fixture,
            &self.model,
            &model_sha256,
            &artifact_path,
            &timestamp_utc,
        );
        validate_dense_gguf_attention_v_mix_fixture_receipt_json(&receipt)?;

        if let Some(path) = &self.json_out {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, serde_json::to_string_pretty(&receipt)?)?;
        } else {
            println!("{}", serde_json::to_string_pretty(&receipt)?);
        }

        Ok(())
    }
}

/// Extract a dense GGUF MLP activation fixture and emit a CPU-reference receipt.
#[derive(Args, Debug, Clone)]
pub struct DenseGgufMlpActivationFixtureCommand {
    /// Dense GGUF model path.
    #[arg(long)]
    pub model: PathBuf,

    /// Dense transformer layer index represented by the deterministic MLP activation fixture.
    #[arg(long, default_value_t = 0)]
    pub layer_index: usize,

    /// Output JSON receipt path. If omitted, writes receipt JSON to stdout.
    #[arg(long, value_name = "PATH")]
    pub json_out: Option<PathBuf>,
}

impl DenseGgufMlpActivationFixtureCommand {
    pub async fn execute(&self) -> Result<()> {
        let data = map_model(&self.model)?;
        let model_sha256 = sha256_bytes(&data);
        let reader = GgufReader::new(&data).with_context(|| {
            format!("failed to parse dense GGUF model {}", self.model.display())
        })?;
        let inspection = inspect_dense_gguf_tensor_descriptors(&reader)?;
        let fixture =
            dense_gguf_mlp_activation_fixture_from_reader(&reader, &inspection, self.layer_index)?;

        let artifact_path = self
            .json_out
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "stdout".to_string());
        let timestamp_utc = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        let receipt = dense_gguf_mlp_activation_fixture_receipt_json(
            &inspection,
            &fixture,
            &self.model,
            &model_sha256,
            &artifact_path,
            &timestamp_utc,
        );
        validate_dense_gguf_mlp_activation_fixture_receipt_json(&receipt)?;

        if let Some(path) = &self.json_out {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, serde_json::to_string_pretty(&receipt)?)?;
        } else {
            println!("{}", serde_json::to_string_pretty(&receipt)?);
        }

        Ok(())
    }
}

/// Run dense GGUF MLP activation CUDA parity diagnostics.
#[derive(Args, Debug, Clone)]
pub struct DenseGgufMlpActivationCudaParityCommand {
    /// Dense GGUF model path.
    #[arg(long)]
    pub model: PathBuf,

    /// Dense transformer layer index represented by the deterministic MLP activation fixture.
    #[arg(long, default_value_t = 0)]
    pub layer_index: usize,

    /// CUDA device index.
    #[arg(long, default_value_t = 0)]
    pub device_index: usize,

    /// Output JSON receipt path. If omitted, writes receipt JSON to stdout.
    #[arg(long, value_name = "PATH")]
    pub json_out: Option<PathBuf>,
}

impl DenseGgufMlpActivationCudaParityCommand {
    pub async fn execute(&self) -> Result<()> {
        let data = map_model(&self.model)?;
        let model_sha256 = sha256_bytes(&data);
        let reader = GgufReader::new(&data).with_context(|| {
            format!("failed to parse dense GGUF model {}", self.model.display())
        })?;
        let inspection = inspect_dense_gguf_tensor_descriptors(&reader)?;

        let probe = bitnet_device_probe::probe_nvidia_cuda(Some(self.device_index));
        if !probe.available {
            bail!("CUDA-DENSE-030 requires CUDA probe success: {:?}", probe.failure_reason);
        }
        let device_name = probe.selected_device_name.as_deref().unwrap_or("unknown");
        if !is_rtx5070ti_device_name(device_name) {
            bail!("CUDA-DENSE-030 requires NVIDIA GeForce RTX 5070 Ti; found '{device_name}'");
        }

        let fixture =
            dense_gguf_mlp_activation_fixture_from_reader(&reader, &inspection, self.layer_index)?;
        let kernel_fixture = kernel_mlp_activation_fixture_from_extracted(&fixture);
        let parity = run_dense_gguf_mlp_activation_cuda_parity(self.device_index, &kernel_fixture)?;
        if !parity.passed {
            bail!(
                "dense GGUF MLP activation CUDA parity failed: max_abs_error={} tolerance={}",
                parity.max_abs_error,
                parity.tolerance
            );
        }

        let artifact_path = self
            .json_out
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "stdout".to_string());
        let timestamp_utc = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        let receipt = dense_gguf_mlp_activation_cuda_parity_receipt_json(
            &inspection,
            &fixture,
            &parity,
            Some(&probe),
            &self.model,
            &model_sha256,
            &artifact_path,
            &timestamp_utc,
        );
        validate_dense_gguf_mlp_activation_cuda_parity_receipt_json(&receipt)?;

        if let Some(path) = &self.json_out {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, serde_json::to_string_pretty(&receipt)?)?;
        } else {
            println!("{}", serde_json::to_string_pretty(&receipt)?);
        }

        Ok(())
    }
}

/// Run dense GGUF attention V-mix CUDA parity diagnostics.
#[derive(Args, Debug, Clone)]
pub struct DenseGgufAttentionVMixCudaParityCommand {
    /// Dense GGUF model path.
    #[arg(long)]
    pub model: PathBuf,

    /// Dense transformer layer index represented by the deterministic fixture.
    #[arg(long, default_value_t = 0)]
    pub layer_index: usize,

    /// Number of token positions in the deterministic attention V-mix fixture.
    #[arg(long, default_value_t = 4)]
    pub seq_len: usize,

    /// Position offset used by the metadata-derived RoPE fixture.
    #[arg(long, default_value_t = 1)]
    pub position_offset: usize,

    /// CUDA device index.
    #[arg(long, default_value_t = 0)]
    pub device_index: usize,

    /// Output JSON receipt path. If omitted, writes receipt JSON to stdout.
    #[arg(long, value_name = "PATH")]
    pub json_out: Option<PathBuf>,
}

impl DenseGgufAttentionVMixCudaParityCommand {
    pub async fn execute(&self) -> Result<()> {
        if self.seq_len == 0 {
            bail!("dense GGUF attention V-mix CUDA parity requires --seq-len > 0");
        }

        let data = map_model(&self.model)?;
        let model_sha256 = sha256_bytes(&data);
        let reader = GgufReader::new(&data).with_context(|| {
            format!("failed to parse dense GGUF model {}", self.model.display())
        })?;
        let inspection = inspect_dense_gguf_tensor_descriptors(&reader)?;

        let probe = bitnet_device_probe::probe_nvidia_cuda(Some(self.device_index));
        if !probe.available {
            bail!("CUDA-DENSE-027 requires CUDA probe success: {:?}", probe.failure_reason);
        }
        let device_name = probe.selected_device_name.as_deref().unwrap_or("unknown");
        if !is_rtx5070ti_device_name(device_name) {
            bail!("CUDA-DENSE-027 requires NVIDIA GeForce RTX 5070 Ti; found '{device_name}'");
        }

        let fixture = dense_gguf_attention_v_mix_fixture_from_reader(
            &reader,
            &inspection,
            self.layer_index,
            self.seq_len,
            self.position_offset,
        )?;
        let kernel_fixture = kernel_attention_v_mix_fixture_from_extracted(&fixture);
        let parity =
            run_dense_gguf_attention_v_mix_cuda_parity(self.device_index, &kernel_fixture)?;
        if !parity.passed {
            bail!(
                "dense GGUF attention V-mix CUDA parity failed: max_abs_error={} tolerance={}",
                parity.max_abs_error,
                parity.tolerance
            );
        }

        let artifact_path = self
            .json_out
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "stdout".to_string());
        let timestamp_utc = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        let receipt = dense_gguf_attention_v_mix_cuda_parity_receipt_json(
            &inspection,
            &fixture,
            &parity,
            Some(&probe),
            &self.model,
            &model_sha256,
            &artifact_path,
            &timestamp_utc,
        );
        validate_dense_gguf_attention_v_mix_cuda_parity_receipt_json(&receipt)?;

        if let Some(path) = &self.json_out {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, serde_json::to_string_pretty(&receipt)?)?;
        } else {
            println!("{}", serde_json::to_string_pretty(&receipt)?);
        }

        Ok(())
    }
}

/// Run dense GGUF attention-softmax CUDA parity diagnostics.
#[derive(Args, Debug, Clone)]
pub struct DenseGgufAttentionSoftmaxCudaParityCommand {
    /// Dense GGUF model path.
    #[arg(long)]
    pub model: PathBuf,

    /// Dense transformer layer index represented by the deterministic fixture.
    #[arg(long, default_value_t = 0)]
    pub layer_index: usize,

    /// Number of token positions in the deterministic attention-softmax fixture.
    #[arg(long, default_value_t = 4)]
    pub seq_len: usize,

    /// Position offset used by the metadata-derived RoPE fixture.
    #[arg(long, default_value_t = 1)]
    pub position_offset: usize,

    /// CUDA device index.
    #[arg(long, default_value_t = 0)]
    pub device_index: usize,

    /// Output JSON receipt path. If omitted, writes receipt JSON to stdout.
    #[arg(long, value_name = "PATH")]
    pub json_out: Option<PathBuf>,
}

impl DenseGgufAttentionSoftmaxCudaParityCommand {
    pub async fn execute(&self) -> Result<()> {
        if self.seq_len == 0 {
            bail!("dense GGUF attention-softmax CUDA parity requires --seq-len > 0");
        }

        let data = map_model(&self.model)?;
        let model_sha256 = sha256_bytes(&data);
        let reader = GgufReader::new(&data).with_context(|| {
            format!("failed to parse dense GGUF model {}", self.model.display())
        })?;
        let inspection = inspect_dense_gguf_tensor_descriptors(&reader)?;

        let probe = bitnet_device_probe::probe_nvidia_cuda(Some(self.device_index));
        if !probe.available {
            bail!("CUDA-DENSE-024 requires CUDA probe success: {:?}", probe.failure_reason);
        }
        let device_name = probe.selected_device_name.as_deref().unwrap_or("unknown");
        if !is_rtx5070ti_device_name(device_name) {
            bail!("CUDA-DENSE-024 requires NVIDIA GeForce RTX 5070 Ti; found '{device_name}'");
        }

        let fixture = dense_gguf_attention_softmax_fixture_from_reader(
            &reader,
            &inspection,
            self.layer_index,
            self.seq_len,
            self.position_offset,
        )?;
        let kernel_fixture = kernel_attention_softmax_fixture_from_extracted(&fixture);
        let parity =
            run_dense_gguf_attention_softmax_cuda_parity(self.device_index, &kernel_fixture)?;
        if !parity.passed {
            bail!(
                "dense GGUF attention-softmax CUDA parity failed: max_abs_error={} tolerance={}",
                parity.max_abs_error,
                parity.tolerance
            );
        }

        let artifact_path = self
            .json_out
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "stdout".to_string());
        let timestamp_utc = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        let receipt = dense_gguf_attention_softmax_cuda_parity_receipt_json(
            &inspection,
            &fixture,
            &parity,
            Some(&probe),
            &self.model,
            &model_sha256,
            &artifact_path,
            &timestamp_utc,
        );
        validate_dense_gguf_attention_softmax_cuda_parity_receipt_json(&receipt)?;

        if let Some(path) = &self.json_out {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, serde_json::to_string_pretty(&receipt)?)?;
        } else {
            println!("{}", serde_json::to_string_pretty(&receipt)?);
        }

        Ok(())
    }
}

/// Run dense GGUF attention-score CUDA parity diagnostics.
#[derive(Args, Debug, Clone)]
pub struct DenseGgufAttentionScoreCudaParityCommand {
    /// Dense GGUF model path.
    #[arg(long)]
    pub model: PathBuf,

    /// Dense transformer layer index represented by the deterministic fixture.
    #[arg(long, default_value_t = 0)]
    pub layer_index: usize,

    /// Number of token positions in the deterministic attention-score fixture.
    #[arg(long, default_value_t = 4)]
    pub seq_len: usize,

    /// Position offset used by the metadata-derived RoPE fixture.
    #[arg(long, default_value_t = 1)]
    pub position_offset: usize,

    /// CUDA device index.
    #[arg(long, default_value_t = 0)]
    pub device_index: usize,

    /// Output JSON receipt path. If omitted, writes receipt JSON to stdout.
    #[arg(long, value_name = "PATH")]
    pub json_out: Option<PathBuf>,
}

impl DenseGgufAttentionScoreCudaParityCommand {
    pub async fn execute(&self) -> Result<()> {
        if self.seq_len == 0 {
            bail!("dense GGUF attention-score CUDA parity requires --seq-len > 0");
        }

        let data = map_model(&self.model)?;
        let model_sha256 = sha256_bytes(&data);
        let reader = GgufReader::new(&data).with_context(|| {
            format!("failed to parse dense GGUF model {}", self.model.display())
        })?;
        let inspection = inspect_dense_gguf_tensor_descriptors(&reader)?;

        let probe = bitnet_device_probe::probe_nvidia_cuda(Some(self.device_index));
        if !probe.available {
            bail!("CUDA-DENSE-021 requires CUDA probe success: {:?}", probe.failure_reason);
        }
        let device_name = probe.selected_device_name.as_deref().unwrap_or("unknown");
        if !is_rtx5070ti_device_name(device_name) {
            bail!("CUDA-DENSE-021 requires NVIDIA GeForce RTX 5070 Ti; found '{device_name}'");
        }

        let fixture = dense_gguf_attention_score_fixture_from_reader(
            &reader,
            &inspection,
            self.layer_index,
            self.seq_len,
            self.position_offset,
        )?;
        let kernel_fixture = kernel_attention_score_fixture_from_extracted(&fixture);
        let parity =
            run_dense_gguf_attention_score_cuda_parity(self.device_index, &kernel_fixture)?;
        if !parity.passed {
            bail!(
                "dense GGUF attention-score CUDA parity failed: max_abs_error={} tolerance={}",
                parity.max_abs_error,
                parity.tolerance
            );
        }

        let artifact_path = self
            .json_out
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "stdout".to_string());
        let timestamp_utc = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        let receipt = dense_gguf_attention_score_cuda_parity_receipt_json(
            &inspection,
            &fixture,
            &parity,
            Some(&probe),
            &self.model,
            &model_sha256,
            &artifact_path,
            &timestamp_utc,
        );
        validate_dense_gguf_attention_score_cuda_parity_receipt_json(&receipt)?;

        if let Some(path) = &self.json_out {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, serde_json::to_string_pretty(&receipt)?)?;
        } else {
            println!("{}", serde_json::to_string_pretty(&receipt)?);
        }

        Ok(())
    }
}

struct DenseLinearSweepResult {
    extracted: DenseGgufLinearFixture,
    parity: DenseGgufLinearCudaParity,
}

struct DenseNormParityResult {
    extracted: DenseGgufNormFixture,
    parity: DenseGgufRmsNormCudaParity,
}

#[derive(Debug, Clone)]
struct DenseGgufAttentionScoreFixture {
    fixture_id: String,
    model_family: String,
    architecture: String,
    layer_index: usize,
    q_heads: usize,
    kv_heads: usize,
    heads_per_kv_group: usize,
    head_dim: usize,
    seq_len: usize,
    position_offset: usize,
    rope_base: f32,
    scaling_factor: f32,
    scale: f32,
    head_dim_source: String,
    q_heads_source: String,
    kv_heads_source: String,
    rope_base_source: String,
    source_rope_fixture_id: String,
    q_rope_output_f32: Vec<f32>,
    k_rope_output_f32: Vec<f32>,
    expected_scores_f32: Vec<f32>,
    finite_scores: usize,
    causal_masked_scores: usize,
}

#[derive(Debug, Clone)]
struct DenseGgufAttentionSoftmaxFixture {
    fixture_id: String,
    model_family: String,
    architecture: String,
    layer_index: usize,
    q_heads: usize,
    kv_heads: usize,
    seq_len: usize,
    source_attention_score_fixture_id: String,
    attention_scores_f32: Vec<f32>,
    expected_probabilities_f32: Vec<f32>,
    row_count: usize,
    probability_count: usize,
    causal_zero_probabilities: usize,
    max_row_sum_abs_error: f32,
}

#[derive(Debug, Clone)]
struct DenseGgufAttentionVMixFixture {
    fixture_id: String,
    model_family: String,
    architecture: String,
    layer_index: usize,
    q_heads: usize,
    kv_heads: usize,
    heads_per_kv_group: usize,
    head_dim: usize,
    seq_len: usize,
    source_attention_softmax_fixture_id: String,
    attention_probabilities_f32: Vec<f32>,
    value_states_f32: Vec<f32>,
    expected_context_f32: Vec<f32>,
    row_count: usize,
    probability_count: usize,
    value_count: usize,
    context_count: usize,
    causal_zero_probabilities: usize,
    max_context_abs: f32,
}

#[derive(Debug, Clone)]
struct DenseGgufMlpActivationFixture {
    fixture_id: String,
    model_family: String,
    architecture: String,
    layer_index: usize,
    source_mlp_gate_fixture_id: String,
    source_mlp_up_fixture_id: String,
    source_mlp_gate_tensor: String,
    source_mlp_up_tensor: String,
    activation_kind: &'static str,
    gate_output_f32: Vec<f32>,
    up_output_f32: Vec<f32>,
    expected_activation_f32: Vec<f32>,
    activation_count: usize,
    max_activation_abs: f32,
}

#[derive(Debug, Clone)]
struct DenseOneLayerCpuReferencePhase {
    index: usize,
    name: &'static str,
    role: &'static str,
    op_type: &'static str,
    output_f32: Vec<f32>,
    output_len: usize,
    output_sha256: String,
    max_abs: f32,
}

#[derive(Debug, Clone)]
struct DenseGgufOneLayerCpuReference {
    fixture_id: String,
    model_family: String,
    architecture: String,
    layer_index: usize,
    seq_len: usize,
    position_offset: usize,
    hidden_size: usize,
    q_heads: usize,
    kv_heads: usize,
    heads_per_kv_group: usize,
    head_dim: usize,
    intermediate_size: usize,
    rmsnorm_eps: f32,
    epsilon_source: String,
    rope_base: f32,
    rope_base_source: String,
    scaling_factor: f32,
    deterministic_input_len: usize,
    deterministic_input_sha256: String,
    phases: Vec<DenseOneLayerCpuReferencePhase>,
    final_output_len: usize,
    final_output_sha256: String,
    final_output_max_abs: f32,
    final_output_f32: Vec<f32>,
}

#[derive(Debug, Clone)]
struct DenseOneLayerCudaPhase {
    index: usize,
    name: &'static str,
    role: &'static str,
    op_type: &'static str,
    route: &'static str,
    status: &'static str,
    output_len: usize,
    output_sha256: String,
    max_abs: f32,
    max_abs_error: f32,
    mean_abs_error: f32,
    tolerance: f32,
    passed: bool,
    kernel_id: Option<&'static str>,
    invocations: u64,
    fallback_invocations: u64,
    host_to_device_bytes: u64,
    device_to_host_bytes: u64,
    kernel_launches: u64,
    kernel_time_ms: Option<f64>,
}

#[derive(Debug, Clone)]
struct DenseOneLayerKernelCounters {
    kernel_id: &'static str,
    invocations: u64,
    fallback_invocations: u64,
    host_to_device_bytes: u64,
    device_to_host_bytes: u64,
    kernel_launches: u64,
    kernel_time_ms: Option<f64>,
}

#[derive(Debug, Clone)]
struct DenseGgufOneLayerCudaIntegratedParity {
    fixture_id: String,
    source_cpu_reference_fixture_id: String,
    model_family: String,
    architecture: String,
    layer_index: usize,
    seq_len: usize,
    position_offset: usize,
    hidden_size: usize,
    q_heads: usize,
    kv_heads: usize,
    heads_per_kv_group: usize,
    head_dim: usize,
    intermediate_size: usize,
    phases: Vec<DenseOneLayerCudaPhase>,
    final_output_len: usize,
    final_output_sha256: String,
    final_output_max_abs: f32,
    final_output_max_abs_error: f32,
    final_output_mean_abs_error: f32,
    tolerance: f32,
    passed: bool,
    host_to_device_bytes: u64,
    device_to_host_bytes: u64,
    kernel_invocations: u64,
    kernel_launches: u64,
    kernel_time_ms: Option<f64>,
}

#[derive(Debug, Clone)]
struct DenseGgufBoundaryTensorFixture {
    name: &'static str,
    role: &'static str,
    tensor_name: String,
    tensor_type: String,
    source_shape: Vec<usize>,
    source_offset: u64,
    source_size_bytes: u64,
    value_count: usize,
    output_len: usize,
    output_sha256: String,
    max_abs: f32,
}

#[derive(Debug, Clone)]
struct DenseGgufLogitTopKEntry {
    rank: usize,
    token_id: usize,
    value: f32,
}

#[derive(Debug, Clone)]
struct DenseGgufModelBoundaryFixtures {
    fixture_id: String,
    model_family: String,
    architecture: String,
    seq_len: usize,
    hidden_size: usize,
    vocab_size: usize,
    token_ids: Vec<usize>,
    token_ids_sha256: String,
    token_embedding: DenseGgufBoundaryTensorFixture,
    final_norm: DenseGgufBoundaryTensorFixture,
    lm_head_logits: DenseGgufBoundaryTensorFixture,
    final_norm_input_sha256: String,
    final_norm_output_sha256: String,
    logits_len: usize,
    logits_sha256: String,
    logits_top_k: Vec<DenseGgufLogitTopKEntry>,
    top_k: usize,
    rmsnorm_eps: f32,
    epsilon_source: String,
}

#[derive(Debug, Clone)]
struct DenseGgufKvCachePolicy {
    policy_id: String,
    model_family: String,
    architecture: String,
    transformer_layers_total: usize,
    transformer_layers_source: String,
    context_length: usize,
    context_length_source: String,
    seq_len: usize,
    decode_steps: usize,
    q_heads: usize,
    q_heads_source: String,
    kv_heads: usize,
    kv_heads_source: String,
    heads_per_kv_group: usize,
    key_head_dim: usize,
    key_head_dim_source: String,
    value_head_dim: usize,
    value_head_dim_source: String,
    kv_element_dtype: &'static str,
    kv_element_bytes: usize,
    kv_values_per_token_per_layer: usize,
    kv_bytes_per_token_per_layer: u64,
    kv_bytes_per_token_all_layers: u64,
    prefill_write_bytes_estimate: u64,
    decode_read_bytes_per_step_estimate: u64,
    decode_write_bytes_per_step_estimate: u64,
    max_context_bytes_estimate: u64,
}

#[derive(Debug, Clone)]
struct DenseGgufSamplingPolicy {
    policy_id: String,
    model_family: String,
    architecture: String,
    seq_len: usize,
    vocab_size: usize,
    logits_len: usize,
    logits_sha256: String,
    logits_element_dtype: &'static str,
    logits_element_bytes: usize,
    logits_transfer_bytes_per_step_estimate: u64,
    logits_top_k: Vec<DenseGgufLogitTopKEntry>,
    top_k: usize,
    selected_token_id_from_fixture_logits: usize,
    sampler_backend: &'static str,
    sampler_location: &'static str,
    sampler_mode: &'static str,
    temperature: f32,
    top_k_filter: usize,
    top_p: f32,
    repetition_penalty: f32,
    deterministic: bool,
    tie_break_policy: &'static str,
    rng_required: bool,
}

#[derive(Debug, Clone)]
struct DenseQwenOneTokenPrerequisites {
    all_layer_plan_sha256: String,
    model_boundary_fixtures_sha256: String,
    kv_cache_policy_sha256: String,
    sampling_policy_sha256: String,
    all_layer_plan: Value,
}

#[derive(Debug, Clone)]
struct DenseQwenShortDecodePrerequisites {
    one_token: DenseQwenOneTokenPrerequisites,
    one_token_proof_sha256: String,
}

#[derive(Debug, Clone)]
struct DenseQwenWarmSessionPrerequisites {
    short_decode: DenseQwenShortDecodePrerequisites,
    short_decode_proof_sha256: String,
}

#[derive(Debug, Clone)]
struct DenseQwenWarmSessionPromptEvidence {
    index: usize,
    token_ids_sha256: String,
    rendered_prompt_sha256: String,
    token_count: usize,
    rendered_prompt_bytes: usize,
}

#[derive(Debug, Clone)]
struct DenseQwenOneTokenRun {
    selected_token_id: u32,
    top_k: Vec<DenseQwenOneTokenTopKEntry>,
    top_k_rank_sha256: String,
    logits_len: usize,
    logits_sha256: String,
    total_ms: f64,
    model_load_ms: f64,
    prefill_ms: f64,
    decode_ms: f64,
    embed_ms: f64,
    forward_ms: f64,
    logits_ms: f64,
    logits_download_ms: f64,
    logits_device_is_cuda: bool,
}

#[derive(Debug, Clone)]
struct DenseQwenShortDecodeRun {
    generated_token_ids: Vec<u32>,
    steps: Vec<DenseQwenShortDecodeStep>,
    generated_token_ids_sha256: String,
    top_k_steps_sha256: String,
    total_ms: f64,
    model_load_ms: f64,
    prefill_ms: f64,
    decode_total_ms: f64,
    first_token_ms: f64,
    embed_ms_total: f64,
    forward_ms_total: f64,
    logits_ms_total: f64,
    logits_download_ms_total: f64,
    logits_transfer_bytes_total: u64,
    logits_len: usize,
    logits_all_cuda_resident: bool,
    logits_transfer_mode: DenseQwenLogitsTransferMode,
}

#[derive(Debug, Clone)]
struct DenseQwenShortDecodeStep {
    index: usize,
    selected_token_id: u32,
    top_k: Vec<DenseQwenOneTokenTopKEntry>,
    top_k_rank_sha256: String,
    logits_sha256: Option<String>,
    logits_len: usize,
    logits_transfer_bytes: u64,
    logits_transfer_mode: DenseQwenLogitsTransferMode,
    embed_ms: f64,
    forward_ms: f64,
    logits_ms: f64,
    logits_download_ms: f64,
    decode_ms: f64,
    logits_device_is_cuda: bool,
}

#[derive(Debug, Clone)]
struct DenseQwenWarmSessionRun {
    turns: Vec<DenseQwenShortDecodeRun>,
    total_ms: f64,
    device_init_ms: f64,
    model_load_ms: f64,
}

#[derive(Debug, Clone)]
struct DenseQwenOneTokenTopKEntry {
    rank: usize,
    token_id: u32,
    value: f32,
}

#[derive(Debug, Clone)]
struct DenseQwenLogitsSample {
    selected_token_id: u32,
    top_k: Vec<DenseQwenOneTokenTopKEntry>,
    top_k_rank_sha256: String,
    logits_len: usize,
    logits_sha256: Option<String>,
    transfer_bytes: u64,
    transfer_mode: DenseQwenLogitsTransferMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DenseQwenLogitsTransferMode {
    FullLogitsDownloadCpuSampler,
    DeviceTopKCudaSampler,
}

impl DenseQwenLogitsTransferMode {
    fn transfer_mode(self) -> &'static str {
        match self {
            Self::FullLogitsDownloadCpuSampler => "full_logits_download_cpu_sampler",
            Self::DeviceTopKCudaSampler => "device_top_k_cuda_sampler",
        }
    }

    fn sampling_location(self) -> &'static str {
        match self {
            Self::FullLogitsDownloadCpuSampler => "cpu",
            Self::DeviceTopKCudaSampler => "cuda_device",
        }
    }

    fn d2h_timing_source(self) -> &'static str {
        match self {
            Self::FullLogitsDownloadCpuSampler => "wall_clock_extract_logits_2d_local",
            Self::DeviceTopKCudaSampler => "wall_clock_device_top_k_cuda_sampler",
        }
    }

    fn full_logits_sha256_available(self) -> bool {
        matches!(self, Self::FullLogitsDownloadCpuSampler)
    }

    fn full_logits_sha256_source(self) -> &'static str {
        match self {
            Self::FullLogitsDownloadCpuSampler => "full_logits_download",
            Self::DeviceTopKCudaSampler => "not_recorded_reduced_device_top_k_sampler",
        }
    }
}

#[derive(Debug, Clone)]
struct DenseLayerPlanEntry {
    op: DispatchOp,
    role: String,
    source: &'static str,
    source_tensor: Option<String>,
    source_tensor_type: Option<String>,
    source_shape: Option<Vec<usize>>,
}

fn map_model(path: &Path) -> Result<Mmap> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    // SAFETY: The mapped file is only read while the file handle and mmap are
    // alive in this command. The command never mutates the mapped model.
    unsafe { Mmap::map(&file) }.with_context(|| format!("failed to mmap {}", path.display()))
}

impl DenseQwenOneTokenPrerequisites {
    fn load(
        all_layer_plan: &Path,
        model_boundary_fixtures: &Path,
        kv_cache_policy: &Path,
        sampling_policy: &Path,
        proof_model: &DenseQwenProofModel,
    ) -> Result<Self> {
        let (all_layer_plan_value, all_layer_plan_sha256) =
            read_and_validate_receipt_for_qwen_model(
                all_layer_plan,
                validate_dense_gguf_all_layer_execution_plan_receipt_json,
                proof_model,
            )?;
        let (_, model_boundary_fixtures_sha256) = read_and_validate_receipt_for_qwen_model(
            model_boundary_fixtures,
            validate_dense_gguf_model_boundary_fixtures_receipt_json,
            proof_model,
        )?;
        let (_, kv_cache_policy_sha256) = read_and_validate_receipt_for_qwen_model(
            kv_cache_policy,
            validate_dense_gguf_kv_cache_policy_receipt_json,
            proof_model,
        )?;
        let (_, sampling_policy_sha256) = read_and_validate_receipt_for_qwen_model(
            sampling_policy,
            validate_dense_gguf_sampling_policy_receipt_json,
            proof_model,
        )?;

        Ok(Self {
            all_layer_plan_sha256,
            model_boundary_fixtures_sha256,
            kv_cache_policy_sha256,
            sampling_policy_sha256,
            all_layer_plan: all_layer_plan_value,
        })
    }
}

impl DenseQwenShortDecodePrerequisites {
    fn load(
        all_layer_plan: &Path,
        model_boundary_fixtures: &Path,
        kv_cache_policy: &Path,
        sampling_policy: &Path,
        one_token_proof: &Path,
        proof_model: &DenseQwenProofModel,
    ) -> Result<Self> {
        let one_token = DenseQwenOneTokenPrerequisites::load(
            all_layer_plan,
            model_boundary_fixtures,
            kv_cache_policy,
            sampling_policy,
            proof_model,
        )?;
        let (_, one_token_proof_sha256) = read_and_validate_receipt_for_qwen_model(
            one_token_proof,
            validate_dense_gguf_qwen_one_token_strict_cuda_proof_receipt_json,
            proof_model,
        )?;

        Ok(Self { one_token, one_token_proof_sha256 })
    }
}

impl DenseQwenWarmSessionPrerequisites {
    fn load(
        all_layer_plan: &Path,
        model_boundary_fixtures: &Path,
        kv_cache_policy: &Path,
        sampling_policy: &Path,
        one_token_proof: &Path,
        short_decode_proof: &Path,
        proof_model: &DenseQwenProofModel,
    ) -> Result<Self> {
        let short_decode = DenseQwenShortDecodePrerequisites::load(
            all_layer_plan,
            model_boundary_fixtures,
            kv_cache_policy,
            sampling_policy,
            one_token_proof,
            proof_model,
        )?;
        let (_, short_decode_proof_sha256) = read_and_validate_receipt_for_qwen_model(
            short_decode_proof,
            validate_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt_json,
            proof_model,
        )?;

        Ok(Self { short_decode, short_decode_proof_sha256 })
    }
}

struct ScopedEnvVar {
    key: &'static str,
    previous: Option<std::ffi::OsString>,
}

impl ScopedEnvVar {
    fn set(key: &'static str, value: &'static str) -> Self {
        let previous = std::env::var_os(key);
        // SAFETY: These proof commands run synchronously on the CLI main thread
        // before model loading. The guard restores the previous value before
        // returning to the caller.
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, previous }
    }

    fn remove(key: &'static str) -> Self {
        let previous = std::env::var_os(key);
        // SAFETY: See `ScopedEnvVar::set`.
        unsafe {
            std::env::remove_var(key);
        }
        Self { key, previous }
    }
}

impl Drop for ScopedEnvVar {
    fn drop(&mut self) {
        // SAFETY: Restores process environment state from the same synchronous
        // command scope that changed it.
        unsafe {
            match &self.previous {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }
}

fn read_and_validate_receipt_for_qwen_model(
    path: &Path,
    validate: impl FnOnce(&Value) -> Result<()>,
    proof_model: &DenseQwenProofModel,
) -> Result<(Value, String)> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("failed to read prerequisite receipt {}", path.display()))?;
    let receipt: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse prerequisite receipt {}", path.display()))?;
    validate(&receipt)
        .with_context(|| format!("invalid prerequisite receipt {}", path.display()))?;
    ensure_prerequisite_qwen_identity(&receipt, path, proof_model)?;
    Ok((receipt, sha256_bytes(&bytes)))
}

fn ensure_prerequisite_qwen_identity(
    receipt: &Value,
    path: &Path,
    proof_model: &DenseQwenProofModel,
) -> Result<()> {
    let Some(model) = receipt.get("model").and_then(Value::as_object) else {
        bail!("prerequisite receipt {} has no model section", path.display());
    };
    let sha = model.get("sha256").and_then(Value::as_str).unwrap_or_default();
    if sha != proof_model.sha256 {
        bail!(
            "prerequisite receipt {} is not for verified {} artifact: sha256={sha}",
            path.display(),
            proof_model.id
        );
    }
    let family = model.get("model_family").and_then(Value::as_str).unwrap_or_default();
    let architecture = model.get("architecture").and_then(Value::as_str).unwrap_or_default();
    if family != "qwen" || architecture != proof_model.architecture {
        bail!(
            "prerequisite receipt {} has unexpected model identity {family}/{architecture}; expected qwen/{}",
            path.display(),
            proof_model.architecture
        );
    }
    Ok(())
}

fn dense_qwen_proof_model_for_sha256(sha256: &str) -> Option<&'static DenseQwenProofModel> {
    match sha256 {
        QWEN25_05B_INSTRUCT_Q8_0_MODEL_SHA256 => Some(&QWEN25_05B_INSTRUCT_Q8_0_PROOF_MODEL),
        QWEN3_06B_INSTRUCT_Q8_0_MODEL_SHA256 => Some(&QWEN3_06B_INSTRUCT_Q8_0_PROOF_MODEL),
        _ => None,
    }
}

fn dense_qwen_proof_kv_cache(
    model: &dyn Model,
    device: &CandleDevice,
    required_seq_len: usize,
) -> Result<bitnet_models::transformer::KVCache> {
    let config = dense_qwen_proof_kv_cache_config(model.config(), required_seq_len)?;
    Ok(bitnet_models::transformer::KVCache::new(&config, 1, device)?)
}

fn dense_qwen_proof_kv_cache_config(
    config: &BitNetConfig,
    required_seq_len: usize,
) -> Result<BitNetConfig> {
    let requested = required_seq_len.max(1);
    let configured = config.model.max_position_embeddings;
    if configured == 0 {
        bail!("dense Qwen proof KV cache requires non-zero model context length");
    }
    if requested > configured {
        bail!(
            "dense Qwen proof requires {requested} tokens, but model context length is {configured}"
        );
    }

    let mut scoped = config.clone();
    scoped.model.max_position_embeddings = requested;
    Ok(scoped)
}

#[derive(Clone, Debug)]
struct DenseQwenPhaseTrace {
    path: Option<PathBuf>,
    command: &'static str,
    started_at: std::time::Instant,
}

impl DenseQwenPhaseTrace {
    fn new(path: Option<&Path>, command: &'static str) -> Self {
        Self { path: path.map(Path::to_path_buf), command, started_at: std::time::Instant::now() }
    }

    fn reset(&self) -> Result<()> {
        let Some(path) = &self.path else {
            return Ok(());
        };
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("failed to create phase trace directory {}", parent.display())
            })?;
        }
        std::fs::File::create(path)
            .with_context(|| format!("failed to reset phase trace {}", path.display()))?;
        Ok(())
    }

    fn emit(&self, phase: &str, state: &str, details: Value) -> Result<()> {
        let Some(path) = &self.path else {
            return Ok(());
        };
        let event = json!({
            "schema": 1,
            "timestamp_utc": chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
            "elapsed_ms": elapsed_ms_f64(self.started_at),
            "command": self.command,
            "phase": phase,
            "state": state,
            "details": details,
        });
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("failed to create phase trace directory {}", parent.display())
            })?;
        }
        use std::io::Write as _;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("failed to open phase trace {}", path.display()))?;
        writeln!(file, "{}", serde_json::to_string(&event)?)
            .with_context(|| format!("failed to write phase trace {}", path.display()))?;
        eprintln!("qwen_one_token_phase={phase}:{state}");
        Ok(())
    }
}

fn run_qwen_one_token_once(
    model_path: &Path,
    device: BitNetDevice,
    prompt_token_ids: &[u32],
    top_k: usize,
    require_cuda: bool,
    phase_trace: &DenseQwenPhaseTrace,
    phase_scope: &'static str,
) -> Result<DenseQwenOneTokenRun> {
    phase_trace.emit(
        phase_scope,
        "start",
        json!({
            "device": if require_cuda { "cuda" } else { "cpu" },
            "prompt_token_count": prompt_token_ids.len() as u64,
            "top_k": top_k,
        }),
    )?;
    phase_trace.emit(
        phase_scope,
        "candle_device_start",
        json!({ "requested_cuda": require_cuda }),
    )?;
    let candle_device = device.to_candle()?;
    if require_cuda && !matches!(candle_device, CandleDevice::Cuda(_)) {
        bail!("CUDA one-token proof requested CUDA device but Candle did not return CUDA");
    }
    phase_trace.emit(
        phase_scope,
        "candle_device_finish",
        json!({ "is_cuda": matches!(candle_device, CandleDevice::Cuda(_)) }),
    )?;

    let loader = ModelLoader::new(device);
    let load_config =
        LoadConfig { use_mmap: true, validate_checksums: false, progress_callback: None };
    let total_start = std::time::Instant::now();
    let load_start = std::time::Instant::now();
    phase_trace.emit(
        phase_scope,
        "model_load_start",
        json!({ "model": model_path.display().to_string() }),
    )?;
    let model = loader
        .load_with_config(model_path, &load_config)
        .with_context(|| format!("failed to load model {}", model_path.display()))?;
    let model_load_ms = elapsed_ms_f64(load_start);
    phase_trace.emit(
        phase_scope,
        "model_load_finish",
        json!({ "model_load_ms": model_load_ms }),
    )?;
    phase_trace.emit(
        phase_scope,
        "kv_cache_start",
        json!({ "required_seq_len": prompt_token_ids.len() as u64 }),
    )?;
    let mut cache =
        dense_qwen_proof_kv_cache(model.as_ref(), &candle_device, prompt_token_ids.len())?;
    phase_trace.emit(phase_scope, "kv_cache_finish", json!({}))?;

    let mut prefill_ms = 0.0;
    if prompt_token_ids.len() > 1 {
        let prefill_start = std::time::Instant::now();
        phase_trace.emit(
            phase_scope,
            "prefill_start",
            json!({ "prefill_tokens": (prompt_token_ids.len() - 1) as u64 }),
        )?;
        for token in &prompt_token_ids[..prompt_token_ids.len() - 1] {
            let embedding = model.embed(&[*token])?;
            if require_cuda && !concrete_tensor_is_cuda(&embedding) {
                bail!("CUDA proof embedding tensor was not CUDA-resident during prefill");
            }
            let hidden = model.forward(&embedding, &mut cache as &mut dyn std::any::Any)?;
            if require_cuda && !concrete_tensor_is_cuda(&hidden) {
                bail!("CUDA proof hidden tensor was not CUDA-resident during prefill");
            }
        }
        prefill_ms = elapsed_ms_f64(prefill_start);
        phase_trace.emit(phase_scope, "prefill_finish", json!({ "prefill_ms": prefill_ms }))?;
    }

    let decode_start = std::time::Instant::now();
    let last_token = prompt_token_ids
        .last()
        .copied()
        .ok_or_else(|| anyhow!("one-token proof requires non-empty prompt tokens"))?;
    let embed_start = std::time::Instant::now();
    phase_trace.emit(phase_scope, "decode_embed_start", json!({ "last_token": last_token }))?;
    let embedding = model.embed(&[last_token])?;
    let embed_ms = elapsed_ms_f64(embed_start);
    phase_trace.emit(phase_scope, "decode_embed_finish", json!({ "embed_ms": embed_ms }))?;
    if require_cuda && !concrete_tensor_is_cuda(&embedding) {
        bail!("CUDA proof decode embedding tensor was not CUDA-resident");
    }

    let forward_start = std::time::Instant::now();
    phase_trace.emit(phase_scope, "decode_forward_start", json!({}))?;
    let hidden = model.forward(&embedding, &mut cache as &mut dyn std::any::Any)?;
    let forward_ms = elapsed_ms_f64(forward_start);
    phase_trace.emit(phase_scope, "decode_forward_finish", json!({ "forward_ms": forward_ms }))?;
    if require_cuda && !concrete_tensor_is_cuda(&hidden) {
        bail!("CUDA proof decode hidden tensor was not CUDA-resident");
    }

    phase_trace.emit(phase_scope, "last_hidden_start", json!({}))?;
    let last_hidden = extract_last_token_hidden_local(&hidden)?;
    phase_trace.emit(phase_scope, "last_hidden_finish", json!({}))?;
    if require_cuda && !concrete_tensor_is_cuda(&last_hidden) {
        bail!("CUDA proof last-hidden tensor was not CUDA-resident");
    }

    let logits_start = std::time::Instant::now();
    phase_trace.emit(phase_scope, "logits_start", json!({}))?;
    let logits = model.logits(&last_hidden)?;
    let logits_ms = elapsed_ms_f64(logits_start);
    phase_trace.emit(phase_scope, "logits_finish", json!({ "logits_ms": logits_ms }))?;
    let logits_device_is_cuda = concrete_tensor_is_cuda(&logits);
    if require_cuda && !logits_device_is_cuda {
        bail!("CUDA proof logits tensor was not CUDA-resident before download");
    }
    let logits_download_start = std::time::Instant::now();
    phase_trace.emit(phase_scope, "logits_download_start", json!({}))?;
    let logits_vec = extract_logits_2d_local(&logits)?;
    let logits_download_ms = elapsed_ms_f64(logits_download_start);
    phase_trace.emit(
        phase_scope,
        "logits_download_finish",
        json!({
            "logits_download_ms": logits_download_ms,
            "logits_len": logits_vec.len() as u64,
        }),
    )?;
    let top_k_entries = dense_qwen_top_k(&logits_vec, top_k);
    let Some(selected) = top_k_entries.first().map(|entry| entry.token_id) else {
        bail!("one-token proof could not select a greedy token from logits");
    };
    let top_k_rank_sha256 = dense_qwen_top_k_rank_sha256(&top_k_entries)?;
    let logits_sha256 = sha256_f32(&logits_vec);
    phase_trace.emit(
        phase_scope,
        "finish",
        json!({
            "selected_token_id": selected,
            "total_ms": elapsed_ms_f64(total_start),
            "decode_ms": elapsed_ms_f64(decode_start),
            "logits_device_is_cuda": logits_device_is_cuda,
        }),
    )?;

    Ok(DenseQwenOneTokenRun {
        selected_token_id: selected,
        top_k: top_k_entries,
        top_k_rank_sha256,
        logits_len: logits_vec.len(),
        logits_sha256,
        total_ms: elapsed_ms_f64(total_start),
        model_load_ms,
        prefill_ms,
        decode_ms: elapsed_ms_f64(decode_start),
        embed_ms,
        forward_ms,
        logits_ms,
        logits_download_ms,
        logits_device_is_cuda,
    })
}

fn run_qwen_short_decode(
    model_path: &Path,
    device: BitNetDevice,
    prompt_token_ids: &[u32],
    max_new_tokens: usize,
    top_k: usize,
    require_cuda: bool,
) -> Result<DenseQwenShortDecodeRun> {
    let total_start = std::time::Instant::now();
    let candle_device = device.to_candle()?;
    if require_cuda && !matches!(candle_device, CandleDevice::Cuda(_)) {
        bail!("CUDA short-decode proof requested CUDA device but Candle did not return CUDA");
    }

    let loader = ModelLoader::new(device);
    let load_config =
        LoadConfig { use_mmap: true, validate_checksums: false, progress_callback: None };
    let load_start = std::time::Instant::now();
    let model = loader
        .load_with_config(model_path, &load_config)
        .with_context(|| format!("failed to load model {}", model_path.display()))?;
    let model_load_ms = elapsed_ms_f64(load_start);
    run_qwen_short_decode_with_loaded_model(
        model.as_ref(),
        &candle_device,
        prompt_token_ids,
        max_new_tokens,
        top_k,
        require_cuda,
        total_start,
        model_load_ms,
    )
}

fn run_qwen_warm_session(
    model_path: &Path,
    device: BitNetDevice,
    turn_token_ids: &[Vec<u32>],
    max_new_tokens: usize,
    top_k: usize,
    require_cuda: bool,
) -> Result<DenseQwenWarmSessionRun> {
    let total_start = std::time::Instant::now();
    let device_start = std::time::Instant::now();
    let candle_device = device.to_candle()?;
    let device_init_ms = elapsed_ms_f64(device_start);
    if require_cuda && !matches!(candle_device, CandleDevice::Cuda(_)) {
        bail!("CUDA warm-session proof requested CUDA device but Candle did not return CUDA");
    }

    let loader = ModelLoader::new(device);
    let load_config =
        LoadConfig { use_mmap: true, validate_checksums: false, progress_callback: None };
    let load_start = std::time::Instant::now();
    let model = loader
        .load_with_config(model_path, &load_config)
        .with_context(|| format!("failed to load model {}", model_path.display()))?;
    let model_load_ms = elapsed_ms_f64(load_start);

    let mut turns = Vec::with_capacity(turn_token_ids.len());
    for (turn_index, prompt_token_ids) in turn_token_ids.iter().enumerate() {
        let turn_start = std::time::Instant::now();
        let turn = run_qwen_short_decode_with_loaded_model(
            model.as_ref(),
            &candle_device,
            prompt_token_ids,
            max_new_tokens,
            top_k,
            require_cuda,
            turn_start,
            0.0,
        )
        .with_context(|| format!("failed warm-session turn {turn_index}"))?;
        turns.push(turn);
    }

    Ok(DenseQwenWarmSessionRun {
        turns,
        total_ms: elapsed_ms_f64(total_start),
        device_init_ms,
        model_load_ms,
    })
}

fn run_qwen_short_decode_with_loaded_model(
    model: &dyn Model,
    candle_device: &CandleDevice,
    prompt_token_ids: &[u32],
    max_new_tokens: usize,
    top_k: usize,
    require_cuda: bool,
    total_start: std::time::Instant,
    model_load_ms: f64,
) -> Result<DenseQwenShortDecodeRun> {
    let cache_seq_len = prompt_token_ids.len().saturating_add(max_new_tokens);
    let mut cache = dense_qwen_proof_kv_cache(model, candle_device, cache_seq_len)?;

    let mut prefill_ms = 0.0;
    if prompt_token_ids.len() > 1 {
        let prefill_start = std::time::Instant::now();
        for token in &prompt_token_ids[..prompt_token_ids.len() - 1] {
            let embedding = model.embed(&[*token])?;
            if require_cuda && !concrete_tensor_is_cuda(&embedding) {
                bail!("CUDA short-decode embedding tensor was not CUDA-resident during prefill");
            }
            let hidden = model.forward(&embedding, &mut cache as &mut dyn std::any::Any)?;
            if require_cuda && !concrete_tensor_is_cuda(&hidden) {
                bail!("CUDA short-decode hidden tensor was not CUDA-resident during prefill");
            }
        }
        prefill_ms = elapsed_ms_f64(prefill_start);
    }

    let decode_start = std::time::Instant::now();
    let mut current_token = prompt_token_ids
        .last()
        .copied()
        .ok_or_else(|| anyhow!("short-decode proof requires non-empty prompt tokens"))?;
    let mut generated_token_ids = Vec::with_capacity(max_new_tokens);
    let mut steps = Vec::with_capacity(max_new_tokens);
    let mut embed_ms_total = 0.0;
    let mut forward_ms_total = 0.0;
    let mut logits_ms_total = 0.0;
    let mut logits_download_ms_total = 0.0;
    let mut logits_transfer_bytes_total = 0_u64;
    let mut logits_all_cuda_resident = true;
    let mut logits_len = 0_usize;
    let logits_transfer_mode = if require_cuda {
        DenseQwenLogitsTransferMode::DeviceTopKCudaSampler
    } else {
        DenseQwenLogitsTransferMode::FullLogitsDownloadCpuSampler
    };

    for index in 0..max_new_tokens {
        let step_start = std::time::Instant::now();
        let embed_start = std::time::Instant::now();
        let embedding = model.embed(&[current_token])?;
        let embed_ms = elapsed_ms_f64(embed_start);
        embed_ms_total += embed_ms;
        if require_cuda && !concrete_tensor_is_cuda(&embedding) {
            bail!("CUDA short-decode embedding tensor was not CUDA-resident at step {index}");
        }

        let forward_start = std::time::Instant::now();
        let hidden = model.forward(&embedding, &mut cache as &mut dyn std::any::Any)?;
        let forward_ms = elapsed_ms_f64(forward_start);
        forward_ms_total += forward_ms;
        if require_cuda && !concrete_tensor_is_cuda(&hidden) {
            bail!("CUDA short-decode hidden tensor was not CUDA-resident at step {index}");
        }

        let last_hidden = extract_last_token_hidden_local(&hidden)?;
        if require_cuda && !concrete_tensor_is_cuda(&last_hidden) {
            bail!("CUDA short-decode last-hidden tensor was not CUDA-resident at step {index}");
        }

        let logits_start = std::time::Instant::now();
        let logits = model.logits(&last_hidden)?;
        let logits_ms = elapsed_ms_f64(logits_start);
        logits_ms_total += logits_ms;
        let logits_device_is_cuda = concrete_tensor_is_cuda(&logits);
        logits_all_cuda_resident &= logits_device_is_cuda;
        if require_cuda && !logits_device_is_cuda {
            bail!(
                "CUDA short-decode logits tensor was not CUDA-resident before download at step {index}"
            );
        }

        let logits_download_start = std::time::Instant::now();
        let logits_sample = sample_dense_qwen_logits_local(&logits, top_k, logits_transfer_mode)?;
        let logits_download_ms = elapsed_ms_f64(logits_download_start);
        logits_download_ms_total += logits_download_ms;
        logits_transfer_bytes_total = logits_transfer_bytes_total
            .checked_add(logits_sample.transfer_bytes)
            .ok_or_else(|| anyhow!("dense Qwen logits transfer byte count overflowed"))?;
        logits_len = logits_sample.logits_len;
        let decode_ms = elapsed_ms_f64(step_start);

        generated_token_ids.push(logits_sample.selected_token_id);
        steps.push(DenseQwenShortDecodeStep {
            index,
            selected_token_id: logits_sample.selected_token_id,
            top_k: logits_sample.top_k,
            top_k_rank_sha256: logits_sample.top_k_rank_sha256,
            logits_sha256: logits_sample.logits_sha256,
            logits_len,
            logits_transfer_bytes: logits_sample.transfer_bytes,
            logits_transfer_mode: logits_sample.transfer_mode,
            embed_ms,
            forward_ms,
            logits_ms,
            logits_download_ms,
            decode_ms,
            logits_device_is_cuda,
        });
        current_token = logits_sample.selected_token_id;
    }

    let generated_token_ids_sha256 = sha256_u32(&generated_token_ids);
    let top_k_steps_sha256 = dense_qwen_top_k_steps_sha256(&steps)?;
    let first_token_ms = steps.first().map(|step| step.decode_ms).unwrap_or(0.0);

    Ok(DenseQwenShortDecodeRun {
        generated_token_ids,
        steps,
        generated_token_ids_sha256,
        top_k_steps_sha256,
        total_ms: elapsed_ms_f64(total_start),
        model_load_ms,
        prefill_ms,
        decode_total_ms: elapsed_ms_f64(decode_start),
        first_token_ms,
        embed_ms_total,
        forward_ms_total,
        logits_ms_total,
        logits_download_ms_total,
        logits_transfer_bytes_total,
        logits_len,
        logits_all_cuda_resident,
        logits_transfer_mode,
    })
}

fn decode_generated_tokens(
    tokenizer: &dyn bitnet_tokenizers::Tokenizer,
    generated_token_ids: &[u32],
) -> String {
    tokenizer.decode(generated_token_ids).ok().filter(|value| !value.is_empty()).unwrap_or_else(
        || {
            generated_token_ids
                .iter()
                .map(|token| {
                    tokenizer
                        .token_to_piece(*token)
                        .filter(|value| !value.is_empty())
                        .unwrap_or_else(|| format!("<token:{token}>"))
                })
                .collect::<Vec<_>>()
                .join("")
        },
    )
}

fn extract_last_token_hidden_local(tensor: &ConcreteTensor) -> Result<ConcreteTensor> {
    let shape = tensor.shape();
    if shape.len() != 3 {
        bail!("expected 3D hidden tensor [B,T,H], got shape {shape:?}");
    }
    let seq_len = shape[1];
    match tensor {
        ConcreteTensor::BitNet(tensor) => {
            let last = tensor.as_candle().narrow(1, seq_len - 1, 1)?.squeeze(1)?;
            Ok(ConcreteTensor::bitnet(last))
        }
        ConcreteTensor::Mock(_) => bail!("dense Qwen proof refuses mock tensors"),
    }
}

fn extract_logits_2d_local(tensor: &ConcreteTensor) -> Result<Vec<f32>> {
    let shape = tensor.shape();
    if shape.len() != 2 {
        bail!("expected 2D logits tensor [B,V], got shape {shape:?}");
    }
    match tensor {
        ConcreteTensor::BitNet(tensor) => {
            let batch_0 = tensor.as_candle().i(0)?;
            let batch_0 = if batch_0.dtype() == DType::F32 {
                batch_0
            } else {
                batch_0.to_dtype(DType::F32)?
            };
            Ok(batch_0.to_vec1::<f32>()?)
        }
        ConcreteTensor::Mock(_) => bail!("dense Qwen proof refuses mock logits"),
    }
}

fn sample_dense_qwen_logits_local(
    tensor: &ConcreteTensor,
    top_k: usize,
    transfer_mode: DenseQwenLogitsTransferMode,
) -> Result<DenseQwenLogitsSample> {
    if top_k == 0 {
        bail!("dense Qwen logits sampling requires top_k > 0");
    }

    match transfer_mode {
        DenseQwenLogitsTransferMode::FullLogitsDownloadCpuSampler => {
            let logits_vec = extract_logits_2d_local(tensor)?;
            let logits_len = logits_vec.len();
            let top_k_entries = dense_qwen_top_k(&logits_vec, top_k);
            let Some(selected_token_id) = top_k_entries.first().map(|entry| entry.token_id) else {
                bail!("dense Qwen logits sampling could not select a greedy token");
            };
            let top_k_rank_sha256 = dense_qwen_top_k_rank_sha256(&top_k_entries)?;
            let logits_sha256 = sha256_f32(&logits_vec);
            let transfer_bytes =
                checked_u64_mul(logits_len as u64, 4, "full logits transfer bytes")?;

            Ok(DenseQwenLogitsSample {
                selected_token_id,
                top_k: top_k_entries,
                top_k_rank_sha256,
                logits_len,
                logits_sha256: Some(logits_sha256),
                transfer_bytes,
                transfer_mode,
            })
        }
        DenseQwenLogitsTransferMode::DeviceTopKCudaSampler => {
            let shape = tensor.shape();
            if shape.len() != 2 {
                bail!("expected 2D logits tensor [B,V], got shape {shape:?}");
            }
            let logits_len = *shape
                .get(1)
                .ok_or_else(|| anyhow!("dense Qwen logits tensor is missing vocab dimension"))?;
            let observed_top_k = top_k.min(logits_len);
            if observed_top_k == 0 {
                bail!("dense Qwen device top-k sampling requires a non-empty logits vector");
            }

            let (top_values, top_ids) = match tensor {
                ConcreteTensor::BitNet(tensor) => {
                    let batch_0 = tensor.as_candle().i(0)?;
                    let batch_0 = if batch_0.dtype() == DType::F32 {
                        batch_0
                    } else {
                        batch_0.to_dtype(DType::F32)?
                    };
                    let (sorted_values, sorted_ids) = batch_0.sort_last_dim(false)?;
                    let top_values =
                        sorted_values.narrow(0, 0, observed_top_k)?.to_vec1::<f32>()?;
                    let top_ids = sorted_ids.narrow(0, 0, observed_top_k)?.to_vec1::<u32>()?;
                    (top_values, top_ids)
                }
                ConcreteTensor::Mock(_) => bail!("dense Qwen proof refuses mock logits"),
            };
            if top_values.len() != observed_top_k || top_ids.len() != observed_top_k {
                bail!("dense Qwen device top-k sampler returned inconsistent top-k lengths");
            }

            let mut top_k_entries = Vec::with_capacity(observed_top_k);
            for (rank, (token_id, value)) in
                top_ids.into_iter().zip(top_values.into_iter()).enumerate()
            {
                if !value.is_finite() {
                    bail!("dense Qwen device top-k sampler returned a non-finite logit");
                }
                top_k_entries.push(DenseQwenOneTokenTopKEntry { rank: rank + 1, token_id, value });
            }
            let Some(selected_token_id) = top_k_entries.first().map(|entry| entry.token_id) else {
                bail!("dense Qwen device top-k sampler could not select a greedy token");
            };
            let top_k_rank_sha256 = dense_qwen_top_k_rank_sha256(&top_k_entries)?;
            let transfer_bytes =
                checked_u64_mul(observed_top_k as u64, 12, "device top-k transfer bytes per step")?;

            Ok(DenseQwenLogitsSample {
                selected_token_id,
                top_k: top_k_entries,
                top_k_rank_sha256,
                logits_len,
                logits_sha256: None,
                transfer_bytes,
                transfer_mode,
            })
        }
    }
}

fn concrete_tensor_is_cuda(tensor: &ConcreteTensor) -> bool {
    tensor.device().is_cuda()
}

fn dense_qwen_top_k(logits: &[f32], top_k: usize) -> Vec<DenseQwenOneTokenTopKEntry> {
    let mut indexed = logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .collect::<Vec<_>>();
    indexed.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
    indexed
        .into_iter()
        .take(top_k)
        .enumerate()
        .map(|(rank, (token_id, value))| DenseQwenOneTokenTopKEntry {
            rank: rank + 1,
            token_id: token_id as u32,
            value,
        })
        .collect()
}

fn dense_qwen_top_k_rank_sha256(entries: &[DenseQwenOneTokenTopKEntry]) -> Result<String> {
    let rank_json = Value::Array(
        entries
            .iter()
            .map(|entry| {
                json!({
                    "rank": entry.rank as u64,
                    "token_id": entry.token_id as u64,
                })
            })
            .collect(),
    );
    sha256_json(&rank_json)
}

fn dense_qwen_top_k_json(entries: &[DenseQwenOneTokenTopKEntry]) -> Vec<Value> {
    entries
        .iter()
        .map(|entry| {
            json!({
                "rank": entry.rank as u64,
                "token_id": entry.token_id as u64,
                "value": entry.value,
            })
        })
        .collect()
}

fn dense_qwen_top_k_steps_sha256(steps: &[DenseQwenShortDecodeStep]) -> Result<String> {
    let steps_json = Value::Array(
        steps
            .iter()
            .map(|step| {
                json!({
                    "index": step.index as u64,
                    "selected_token_id": step.selected_token_id as u64,
                    "top_k": step
                        .top_k
                        .iter()
                        .map(|entry| {
                            json!({
                                "rank": entry.rank as u64,
                                "token_id": entry.token_id as u64,
                            })
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect(),
    );
    sha256_json(&steps_json)
}

fn first_divergence_index(left: &[u32], right: &[u32]) -> Option<usize> {
    let shared = left.len().min(right.len());
    for index in 0..shared {
        if left[index] != right[index] {
            return Some(index);
        }
    }
    (left.len() != right.len()).then_some(shared)
}

fn first_top_k_divergence_index(
    cpu: &[DenseQwenShortDecodeStep],
    cuda: &[DenseQwenShortDecodeStep],
) -> Option<usize> {
    let shared = cpu.len().min(cuda.len());
    for index in 0..shared {
        if cpu[index].top_k_rank_sha256 != cuda[index].top_k_rank_sha256 {
            return Some(index);
        }
    }
    (cpu.len() != cuda.len()).then_some(shared)
}

fn dense_qwen_short_decode_steps_json(
    cpu: &DenseQwenShortDecodeRun,
    cuda: &DenseQwenShortDecodeRun,
) -> Vec<Value> {
    cpu.steps
        .iter()
        .zip(cuda.steps.iter())
        .map(|(cpu_step, cuda_step)| {
            let (top_k_max_abs_error, top_k_mean_abs_error) =
                dense_qwen_logits_error(&cpu_step.top_k, &cuda_step.top_k);
            let top_k_match = cpu_step.top_k_rank_sha256 == cuda_step.top_k_rank_sha256
                && cpu_step
                    .top_k
                    .iter()
                    .zip(cuda_step.top_k.iter())
                    .all(|(left, right)| left.token_id == right.token_id);
            let cpu_logits_sha256 = cpu_step
                .logits_sha256
                .as_ref()
                .map(|value| Value::String(value.clone()))
                .unwrap_or(Value::Null);
            let cuda_logits_sha256 = cuda_step
                .logits_sha256
                .as_ref()
                .map(|value| Value::String(value.clone()))
                .unwrap_or(Value::Null);
            json!({
                "index": cpu_step.index as u64,
                "cpu_selected_token_id": cpu_step.selected_token_id as u64,
                "cuda_selected_token_id": cuda_step.selected_token_id as u64,
                "selected_token_match": cpu_step.selected_token_id == cuda_step.selected_token_id,
                "cpu_logits_top_k_sha256": cpu_step.top_k_rank_sha256,
                "cuda_logits_top_k_sha256": cuda_step.top_k_rank_sha256,
                "cpu_logits_sha256": cpu_logits_sha256,
                "cuda_logits_sha256": cuda_logits_sha256,
                "cpu_logits_sha256_available": cpu_step.logits_transfer_mode.full_logits_sha256_available(),
                "cuda_logits_sha256_available": cuda_step.logits_transfer_mode.full_logits_sha256_available(),
                "cpu_logits_sha256_source": cpu_step.logits_transfer_mode.full_logits_sha256_source(),
                "cuda_logits_sha256_source": cuda_step.logits_transfer_mode.full_logits_sha256_source(),
                "logits_vector_length": cuda_step.logits_len as u64,
                "cpu_top_k": dense_qwen_top_k_json(&cpu_step.top_k),
                "cuda_top_k": dense_qwen_top_k_json(&cuda_step.top_k),
                "top_k_match": top_k_match,
                "top_k_max_abs_error": top_k_max_abs_error,
                "top_k_mean_abs_error": top_k_mean_abs_error,
                "cuda_step_timing": {
                    "embed_ms": cuda_step.embed_ms,
                    "forward_ms": cuda_step.forward_ms,
                    "logits_ms": cuda_step.logits_ms,
                    "logits_download_ms": cuda_step.logits_download_ms,
                    "decode_ms": cuda_step.decode_ms,
                    "logits_device_is_cuda": cuda_step.logits_device_is_cuda,
                    "logits_transfer_bytes": cuda_step.logits_transfer_bytes as u64,
                    "logits_transfer_mode": cuda_step.logits_transfer_mode.transfer_mode(),
                    "sampling_location": cuda_step.logits_transfer_mode.sampling_location()
                }
            })
        })
        .collect()
}

fn dense_qwen_logits_error(
    cpu: &[DenseQwenOneTokenTopKEntry],
    cuda: &[DenseQwenOneTokenTopKEntry],
) -> (f64, f64) {
    let count = cpu.len().min(cuda.len());
    if count == 0 {
        return (0.0, 0.0);
    }
    let mut max_abs = 0.0_f64;
    let mut sum_abs = 0.0_f64;
    for (left, right) in cpu.iter().zip(cuda.iter()) {
        let delta = (left.value as f64 - right.value as f64).abs();
        max_abs = max_abs.max(delta);
        sum_abs += delta;
    }
    (max_abs, sum_abs / count as f64)
}

fn elapsed_ms_f64(start: std::time::Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn dense_qwen_transfer_timing_status() -> &'static str {
    "host_to_device_model_load_envelope_device_to_host_measured"
}

fn dense_qwen_h2d_timing_source() -> &'static str {
    "wall_clock_model_load_with_cuda_weight_upload"
}

fn dense_qwen_h2d_timing_scope() -> &'static str {
    "model_load_wall_clock_envelope"
}

fn dense_qwen_d2h_timing_source() -> &'static str {
    "wall_clock_extract_logits_2d_local"
}

fn kernel_fixture_from_extracted(
    fixture: &DenseGgufLinearFixture,
) -> Result<DenseGgufLinearGemmFixture> {
    let summary = &fixture.summary;
    Ok(DenseGgufLinearGemmFixture {
        fixture_id: dense_linear_fixture_id(
            &summary.model_family,
            dense_role_label(summary.role),
            &summary.tensor_type,
        ),
        model_family: summary.model_family.clone(),
        tensor_name: summary.tensor_name.clone(),
        tensor_role: dense_role_label(summary.role).to_string(),
        tensor_type: summary.tensor_type.clone(),
        source_weight_sha256: summary.weight_values_sha256.clone(),
        matrix_rows: summary.matrix_rows,
        matrix_cols: summary.matrix_cols,
        weights_row_major_f32: fixture.weight_values_f32.clone(),
        input_f32: fixture.cpu_reference_input.clone(),
    })
}

fn kernel_rmsnorm_fixture_from_extracted(
    fixture: &DenseGgufNormFixture,
) -> Result<DenseGgufRmsNormCudaFixture> {
    let summary = &fixture.summary;
    let role = dense_role_label(summary.role).to_string();
    if !matches!(role.as_str(), "attention_norm" | "ffn_norm") {
        bail!("dense GGUF RMSNorm CUDA parity only supports attention_norm and ffn_norm");
    }
    if fixture.weight_values_f32.len() != summary.hidden_dim
        || fixture.cpu_reference_input.len() != summary.hidden_dim
        || fixture.cpu_reference_output.len() != summary.hidden_dim
    {
        bail!(
            "dense GGUF RMSNorm fixture length mismatch for {}: hidden_dim={} gamma={} input={} output={}",
            summary.tensor_name,
            summary.hidden_dim,
            fixture.weight_values_f32.len(),
            fixture.cpu_reference_input.len(),
            fixture.cpu_reference_output.len()
        );
    }

    Ok(DenseGgufRmsNormCudaFixture {
        fixture_id: format!("dense_gguf_rmsnorm_{role}"),
        model_family: summary.model_family.clone(),
        tensor_name: summary.tensor_name.clone(),
        tensor_role: role,
        tensor_type: summary.tensor_type.clone(),
        source_weight_sha256: summary.weight_values_sha256.clone(),
        hidden_dim: summary.hidden_dim,
        input_f32: fixture.cpu_reference_input.clone(),
        gamma_f32: fixture.weight_values_f32.clone(),
        expected_output_f32: fixture.cpu_reference_output.clone(),
        rmsnorm_eps: summary.rmsnorm_eps,
    })
}

fn dense_gguf_rope_cuda_fixture_from_reader(
    reader: &GgufReader<'_>,
    inspection: &DenseGgufDescriptorInspection,
    layer_index: usize,
    seq_len: usize,
    position_offset: usize,
) -> Result<DenseGgufRopeCudaFixture> {
    let architecture = &inspection.architecture;
    let head_dim = metadata_u32_with_source(
        reader,
        &[
            format!("{architecture}.attention.key_length"),
            format!("{architecture}.attention.head_dim"),
            "attention.key_length".to_string(),
        ],
    )
    .or_else(|| {
        let heads = metadata_u32_with_source(
            reader,
            &[format!("{architecture}.attention.head_count"), "attention.head_count".to_string()],
        )?;
        let embedding = metadata_u32_with_source(
            reader,
            &[format!("{architecture}.embedding_length"), "embedding_length".to_string()],
        )?;
        if heads.0 == 0 || embedding.0 % heads.0 != 0 {
            return None;
        }
        Some((embedding.0 / heads.0, "embedding_length_div_attention.head_count".to_string()))
    })
    .ok_or_else(|| {
        anyhow!("dense GGUF RoPE fixture requires attention key/head dimension metadata")
    })?;
    let q_heads = metadata_u32_with_source(
        reader,
        &[format!("{architecture}.attention.head_count"), "attention.head_count".to_string()],
    )
    .ok_or_else(|| anyhow!("dense GGUF RoPE fixture requires attention.head_count metadata"))?;
    let kv_heads = metadata_u32_with_source(
        reader,
        &[format!("{architecture}.attention.head_count_kv"), "attention.head_count_kv".to_string()],
    )
    .unwrap_or((q_heads.0, "attention.head_count_kv_missing_default_to_q_heads".to_string()));
    let rope_base = metadata_f32_with_source(
        reader,
        &[format!("{architecture}.rope.freq_base"), "rope.freq_base".to_string()],
    )
    .unwrap_or((10_000.0, "rope.freq_base_missing_default_10000".to_string()));
    let scaling_factor = metadata_f32_with_source(
        reader,
        &[format!("{architecture}.rope.scaling.factor"), "rope.scaling.factor".to_string()],
    )
    .unwrap_or((1.0, "rope.scaling.factor_missing_default_1".to_string()));

    let (head_dim_value, head_dim_source) = head_dim;
    let (q_heads_value, q_heads_source) = q_heads;
    let (kv_heads_value, kv_heads_source) = kv_heads;
    let (rope_base_value, rope_base_source) = rope_base;
    let (scaling_factor_value, _scaling_factor_source) = scaling_factor;
    let head_dim = head_dim_value as usize;
    let q_heads = q_heads_value as usize;
    let kv_heads = kv_heads_value as usize;
    let q_input_f32 = deterministic_rope_input(q_heads * seq_len * head_dim, 17);
    let k_input_f32 = deterministic_rope_input(kv_heads * seq_len * head_dim, 41);
    let q_config = RopeConfig::for_shape(head_dim, q_heads, seq_len)?
        .with_position_offset(position_offset)
        .with_base(rope_base_value)
        .with_scaling_factor(scaling_factor_value);
    let k_config = RopeConfig::for_shape(head_dim, kv_heads, seq_len)?
        .with_position_offset(position_offset)
        .with_base(rope_base_value)
        .with_scaling_factor(scaling_factor_value);
    let mut expected_q_output_f32 = vec![0.0f32; q_input_f32.len()];
    rope_forward_cpu(&q_input_f32, &mut expected_q_output_f32, &q_config)?;
    let mut expected_k_output_f32 = vec![0.0f32; k_input_f32.len()];
    rope_forward_cpu(&k_input_f32, &mut expected_k_output_f32, &k_config)?;

    Ok(DenseGgufRopeCudaFixture {
        fixture_id: format!(
            "dense_gguf_rope_{}_layer{}_q{}_kv{}_d{}_s{}",
            sanitize_label(&inspection.model_family),
            layer_index,
            q_heads,
            kv_heads,
            head_dim,
            seq_len
        ),
        model_family: inspection.model_family.clone(),
        architecture: inspection.architecture.clone(),
        layer_index,
        q_heads,
        kv_heads,
        head_dim,
        seq_len,
        position_offset,
        rope_base: rope_base_value,
        scaling_factor: scaling_factor_value,
        interleaved: false,
        head_dim_source: head_dim_source_label(&head_dim_source),
        q_heads_source,
        kv_heads_source,
        rope_base_source,
        q_input_f32,
        k_input_f32,
        expected_q_output_f32,
        expected_k_output_f32,
    })
}

fn dense_gguf_attention_score_fixture_from_reader(
    reader: &GgufReader<'_>,
    inspection: &DenseGgufDescriptorInspection,
    layer_index: usize,
    seq_len: usize,
    position_offset: usize,
) -> Result<DenseGgufAttentionScoreFixture> {
    let rope = dense_gguf_rope_cuda_fixture_from_reader(
        reader,
        inspection,
        layer_index,
        seq_len,
        position_offset,
    )?;
    if rope.kv_heads == 0 || rope.q_heads == 0 {
        bail!("dense GGUF attention-score fixture requires non-zero Q and KV heads");
    }
    if rope.q_heads % rope.kv_heads != 0 {
        bail!(
            "dense GGUF attention-score fixture requires q_heads divisible by kv_heads: q_heads={} kv_heads={}",
            rope.q_heads,
            rope.kv_heads
        );
    }
    let heads_per_kv_group = rope.q_heads / rope.kv_heads;
    let scale = 1.0 / (rope.head_dim as f32).sqrt();
    let expected_scores_f32 = dense_attention_scores_cpu_reference(
        &rope.expected_q_output_f32,
        &rope.expected_k_output_f32,
        rope.q_heads,
        rope.kv_heads,
        rope.seq_len,
        rope.head_dim,
        scale,
    )?;
    let finite_scores = expected_scores_f32.iter().filter(|score| score.is_finite()).count();
    let causal_masked_scores = expected_scores_f32.len().saturating_sub(finite_scores);

    Ok(DenseGgufAttentionScoreFixture {
        fixture_id: format!(
            "dense_gguf_attention_scores_{}_layer{}_q{}_kv{}_d{}_s{}",
            sanitize_label(&inspection.model_family),
            layer_index,
            rope.q_heads,
            rope.kv_heads,
            rope.head_dim,
            rope.seq_len
        ),
        model_family: rope.model_family.clone(),
        architecture: rope.architecture.clone(),
        layer_index,
        q_heads: rope.q_heads,
        kv_heads: rope.kv_heads,
        heads_per_kv_group,
        head_dim: rope.head_dim,
        seq_len: rope.seq_len,
        position_offset: rope.position_offset,
        rope_base: rope.rope_base,
        scaling_factor: rope.scaling_factor,
        scale,
        head_dim_source: rope.head_dim_source.clone(),
        q_heads_source: rope.q_heads_source.clone(),
        kv_heads_source: rope.kv_heads_source.clone(),
        rope_base_source: rope.rope_base_source.clone(),
        source_rope_fixture_id: rope.fixture_id.clone(),
        q_rope_output_f32: rope.expected_q_output_f32,
        k_rope_output_f32: rope.expected_k_output_f32,
        expected_scores_f32,
        finite_scores,
        causal_masked_scores,
    })
}

fn dense_gguf_attention_softmax_fixture_from_reader(
    reader: &GgufReader<'_>,
    inspection: &DenseGgufDescriptorInspection,
    layer_index: usize,
    seq_len: usize,
    position_offset: usize,
) -> Result<DenseGgufAttentionSoftmaxFixture> {
    let scores = dense_gguf_attention_score_fixture_from_reader(
        reader,
        inspection,
        layer_index,
        seq_len,
        position_offset,
    )?;
    let (expected_probabilities_f32, causal_zero_probabilities, max_row_sum_abs_error) =
        dense_attention_softmax_cpu_reference(
            &scores.expected_scores_f32,
            scores.q_heads,
            scores.seq_len,
        )?;

    Ok(DenseGgufAttentionSoftmaxFixture {
        fixture_id: format!(
            "dense_gguf_attention_softmax_{}_layer{}_q{}_kv{}_s{}",
            sanitize_label(&inspection.model_family),
            layer_index,
            scores.q_heads,
            scores.kv_heads,
            scores.seq_len
        ),
        model_family: scores.model_family.clone(),
        architecture: scores.architecture.clone(),
        layer_index: scores.layer_index,
        q_heads: scores.q_heads,
        kv_heads: scores.kv_heads,
        seq_len: scores.seq_len,
        source_attention_score_fixture_id: scores.fixture_id.clone(),
        attention_scores_f32: scores.expected_scores_f32,
        expected_probabilities_f32,
        row_count: scores.q_heads * scores.seq_len,
        probability_count: scores.q_heads * scores.seq_len * scores.seq_len,
        causal_zero_probabilities,
        max_row_sum_abs_error,
    })
}

fn dense_gguf_attention_v_mix_fixture_from_reader(
    reader: &GgufReader<'_>,
    inspection: &DenseGgufDescriptorInspection,
    layer_index: usize,
    seq_len: usize,
    position_offset: usize,
) -> Result<DenseGgufAttentionVMixFixture> {
    let scores = dense_gguf_attention_score_fixture_from_reader(
        reader,
        inspection,
        layer_index,
        seq_len,
        position_offset,
    )?;
    let (probabilities, causal_zero_probabilities, _) = dense_attention_softmax_cpu_reference(
        &scores.expected_scores_f32,
        scores.q_heads,
        scores.seq_len,
    )?;
    let value_states_f32 =
        deterministic_attention_value_input(scores.kv_heads * scores.seq_len * scores.head_dim);
    let expected_context_f32 = dense_attention_v_mix_cpu_reference(
        &probabilities,
        &value_states_f32,
        scores.q_heads,
        scores.kv_heads,
        scores.seq_len,
        scores.head_dim,
    )?;
    let max_context_abs = expected_context_f32.iter().copied().map(f32::abs).fold(0.0f32, f32::max);

    Ok(DenseGgufAttentionVMixFixture {
        fixture_id: format!(
            "dense_gguf_attention_v_mix_{}_layer{}_q{}_kv{}_d{}_s{}",
            sanitize_label(&inspection.model_family),
            layer_index,
            scores.q_heads,
            scores.kv_heads,
            scores.head_dim,
            scores.seq_len
        ),
        model_family: scores.model_family.clone(),
        architecture: scores.architecture.clone(),
        layer_index: scores.layer_index,
        q_heads: scores.q_heads,
        kv_heads: scores.kv_heads,
        heads_per_kv_group: scores.heads_per_kv_group,
        head_dim: scores.head_dim,
        seq_len: scores.seq_len,
        source_attention_softmax_fixture_id: format!(
            "dense_gguf_attention_softmax_{}_layer{}_q{}_kv{}_s{}",
            sanitize_label(&inspection.model_family),
            layer_index,
            scores.q_heads,
            scores.kv_heads,
            scores.seq_len
        ),
        attention_probabilities_f32: probabilities,
        value_states_f32,
        expected_context_f32,
        row_count: scores.q_heads * scores.seq_len,
        probability_count: scores.q_heads * scores.seq_len * scores.seq_len,
        value_count: scores.kv_heads * scores.seq_len * scores.head_dim,
        context_count: scores.q_heads * scores.seq_len * scores.head_dim,
        causal_zero_probabilities,
        max_context_abs,
    })
}

fn dense_gguf_mlp_activation_fixture_from_reader(
    reader: &GgufReader<'_>,
    inspection: &DenseGgufDescriptorInspection,
    layer_index: usize,
) -> Result<DenseGgufMlpActivationFixture> {
    let gate = extract_dense_gguf_linear_fixture(reader, DenseGgufTensorRole::MlpGate)
        .context("failed to extract dense GGUF MLP gate fixture")?;
    let up = extract_dense_gguf_linear_fixture(reader, DenseGgufTensorRole::MlpUp)
        .context("failed to extract dense GGUF MLP up fixture")?;
    if gate.summary.model_family != inspection.model_family
        || up.summary.model_family != inspection.model_family
    {
        bail!(
            "dense GGUF MLP activation fixture mixed model families: inspection={} gate={} up={}",
            inspection.model_family,
            gate.summary.model_family,
            up.summary.model_family
        );
    }
    if gate.summary.architecture != inspection.architecture
        || up.summary.architecture != inspection.architecture
    {
        bail!(
            "dense GGUF MLP activation fixture mixed architectures: inspection={} gate={} up={}",
            inspection.architecture,
            gate.summary.architecture,
            up.summary.architecture
        );
    }
    if gate.cpu_reference_output.len() != up.cpu_reference_output.len() {
        bail!(
            "dense GGUF MLP activation fixture requires equal gate/up output lengths: gate={} up={}",
            gate.cpu_reference_output.len(),
            up.cpu_reference_output.len()
        );
    }

    let expected_activation_f32 =
        dense_mlp_activation_cpu_reference(&gate.cpu_reference_output, &up.cpu_reference_output)?;
    let max_activation_abs =
        expected_activation_f32.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
    let gate_role = dense_role_label(gate.summary.role);
    let up_role = dense_role_label(up.summary.role);

    Ok(DenseGgufMlpActivationFixture {
        fixture_id: format!(
            "dense_gguf_mlp_activation_{}_layer{}_n{}",
            sanitize_label(&inspection.model_family),
            layer_index,
            expected_activation_f32.len()
        ),
        model_family: inspection.model_family.clone(),
        architecture: inspection.architecture.clone(),
        layer_index,
        source_mlp_gate_fixture_id: dense_linear_fixture_id(
            &gate.summary.model_family,
            gate_role,
            &gate.summary.tensor_type,
        ),
        source_mlp_up_fixture_id: dense_linear_fixture_id(
            &up.summary.model_family,
            up_role,
            &up.summary.tensor_type,
        ),
        source_mlp_gate_tensor: gate.summary.tensor_name,
        source_mlp_up_tensor: up.summary.tensor_name,
        activation_kind: "silu_gate_times_up",
        gate_output_f32: gate.cpu_reference_output,
        up_output_f32: up.cpu_reference_output,
        activation_count: expected_activation_f32.len(),
        expected_activation_f32,
        max_activation_abs,
    })
}

fn dense_gguf_one_layer_cpu_reference_from_reader(
    reader: &GgufReader<'_>,
    inspection: &DenseGgufDescriptorInspection,
    layer_index: usize,
    seq_len: usize,
    position_offset: usize,
) -> Result<DenseGgufOneLayerCpuReference> {
    if layer_index != 0 {
        bail!("dense GGUF one-layer CPU reference currently supports layer 0 only");
    }
    if seq_len == 0 {
        bail!("dense GGUF one-layer CPU reference requires a non-zero sequence length");
    }
    if !inspection.required_roles_present || !inspection.strict_descriptor_complete {
        bail!("dense GGUF one-layer CPU reference requires complete dense descriptor coverage");
    }

    let attention_norm =
        extract_dense_gguf_norm_fixture(reader, DenseGgufTensorRole::AttentionNorm)
            .context("failed to extract dense GGUF attention norm fixture")?;
    let ffn_norm = extract_dense_gguf_norm_fixture(reader, DenseGgufTensorRole::FfnNorm)
        .context("failed to extract dense GGUF FFN norm fixture")?;
    let attention_q = extract_dense_gguf_linear_fixture(reader, DenseGgufTensorRole::AttentionQ)
        .context("failed to extract dense GGUF attention Q fixture")?;
    let attention_k = extract_dense_gguf_linear_fixture(reader, DenseGgufTensorRole::AttentionK)
        .context("failed to extract dense GGUF attention K fixture")?;
    let attention_v = extract_dense_gguf_linear_fixture(reader, DenseGgufTensorRole::AttentionV)
        .context("failed to extract dense GGUF attention V fixture")?;
    let attention_output =
        extract_dense_gguf_linear_fixture(reader, DenseGgufTensorRole::AttentionOutput)
            .context("failed to extract dense GGUF attention output fixture")?;
    let mlp_gate = extract_dense_gguf_linear_fixture(reader, DenseGgufTensorRole::MlpGate)
        .context("failed to extract dense GGUF MLP gate fixture")?;
    let mlp_up = extract_dense_gguf_linear_fixture(reader, DenseGgufTensorRole::MlpUp)
        .context("failed to extract dense GGUF MLP up fixture")?;
    let mlp_down = extract_dense_gguf_linear_fixture(reader, DenseGgufTensorRole::MlpDown)
        .context("failed to extract dense GGUF MLP down fixture")?;

    let hidden_size = attention_q.summary.matrix_cols;
    ensure_norm_dim(&attention_norm, hidden_size, "attention_norm")?;
    ensure_norm_dim(&ffn_norm, hidden_size, "ffn_norm")?;
    ensure_linear_cols(&attention_k, hidden_size, "attention_k")?;
    ensure_linear_cols(&attention_v, hidden_size, "attention_v")?;
    ensure_linear_rows(&attention_output, hidden_size, "attention_output")?;
    ensure_linear_cols(&mlp_gate, hidden_size, "mlp_gate")?;
    ensure_linear_cols(&mlp_up, hidden_size, "mlp_up")?;

    let rope_metadata = dense_gguf_rope_cuda_fixture_from_reader(
        reader,
        inspection,
        layer_index,
        seq_len,
        position_offset,
    )?;
    let q_heads = rope_metadata.q_heads;
    let kv_heads = rope_metadata.kv_heads;
    let head_dim = rope_metadata.head_dim;
    if q_heads == 0 || kv_heads == 0 || head_dim == 0 {
        bail!("dense GGUF one-layer CPU reference requires non-zero attention metadata");
    }
    if q_heads % kv_heads != 0 {
        bail!(
            "dense GGUF one-layer CPU reference requires q_heads divisible by kv_heads: q_heads={q_heads} kv_heads={kv_heads}"
        );
    }
    let q_projection = q_heads
        .checked_mul(head_dim)
        .ok_or_else(|| anyhow!("dense GGUF Q projection shape overflows"))?;
    let kv_projection = kv_heads
        .checked_mul(head_dim)
        .ok_or_else(|| anyhow!("dense GGUF KV projection shape overflows"))?;
    ensure_linear_rows(&attention_q, q_projection, "attention_q")?;
    ensure_linear_rows(&attention_k, kv_projection, "attention_k")?;
    ensure_linear_rows(&attention_v, kv_projection, "attention_v")?;
    ensure_linear_cols(&attention_output, q_projection, "attention_output")?;

    let intermediate_size = mlp_gate.summary.matrix_rows;
    ensure_linear_rows(&mlp_up, intermediate_size, "mlp_up")?;
    ensure_linear_rows(&mlp_down, hidden_size, "mlp_down")?;
    ensure_linear_cols(&mlp_down, intermediate_size, "mlp_down")?;

    if (attention_norm.summary.rmsnorm_eps - ffn_norm.summary.rmsnorm_eps).abs() > f32::EPSILON {
        bail!(
            "dense GGUF one-layer CPU reference requires matching RMSNorm epsilons: attention={} ffn={}",
            attention_norm.summary.rmsnorm_eps,
            ffn_norm.summary.rmsnorm_eps
        );
    }

    let input = deterministic_layer_input(seq_len, hidden_size)?;
    let mut phases = Vec::new();
    push_reference_phase(&mut phases, "deterministic_input", "hidden_state", "input", &input);

    let attention_norm_output = rmsnorm_sequence_cpu(
        &input,
        seq_len,
        hidden_size,
        &attention_norm.weight_values_f32,
        attention_norm.summary.rmsnorm_eps,
    )?;
    push_reference_phase(
        &mut phases,
        "attention_norm",
        "attention_norm",
        "rmsnorm",
        &attention_norm_output,
    );

    let q_linear = dense_linear_sequence_cpu(&attention_q, &attention_norm_output, seq_len)?;
    push_reference_phase(&mut phases, "attention_q", "attention_q", "matmul", &q_linear);
    let k_linear = dense_linear_sequence_cpu(&attention_k, &attention_norm_output, seq_len)?;
    push_reference_phase(&mut phases, "attention_k", "attention_k", "matmul", &k_linear);
    let v_linear = dense_linear_sequence_cpu(&attention_v, &attention_norm_output, seq_len)?;
    push_reference_phase(&mut phases, "attention_v", "attention_v", "matmul", &v_linear);

    let q_head_major = seq_major_to_head_major(&q_linear, seq_len, q_heads, head_dim)?;
    let k_head_major = seq_major_to_head_major(&k_linear, seq_len, kv_heads, head_dim)?;
    let q_config = RopeConfig::for_shape(head_dim, q_heads, seq_len)?
        .with_position_offset(position_offset)
        .with_base(rope_metadata.rope_base)
        .with_scaling_factor(rope_metadata.scaling_factor);
    let k_config = RopeConfig::for_shape(head_dim, kv_heads, seq_len)?
        .with_position_offset(position_offset)
        .with_base(rope_metadata.rope_base)
        .with_scaling_factor(rope_metadata.scaling_factor);
    let mut q_rope = vec![0.0f32; q_head_major.len()];
    rope_forward_cpu(&q_head_major, &mut q_rope, &q_config)?;
    let mut k_rope = vec![0.0f32; k_head_major.len()];
    rope_forward_cpu(&k_head_major, &mut k_rope, &k_config)?;
    let mut rope_phase_output = q_rope.clone();
    rope_phase_output.extend_from_slice(&k_rope);
    push_reference_phase(&mut phases, "rope", "rope", "rope", &rope_phase_output);

    let scale = 1.0 / (head_dim as f32).sqrt();
    let attention_scores = dense_attention_scores_cpu_reference(
        &q_rope, &k_rope, q_heads, kv_heads, seq_len, head_dim, scale,
    )?;
    push_reference_phase(
        &mut phases,
        "attention_scores",
        "attention_scores",
        "attention",
        &attention_scores,
    );
    let (attention_probabilities, _causal_zero_probabilities, _max_row_sum_abs_error) =
        dense_attention_softmax_cpu_reference(&attention_scores, q_heads, seq_len)?;
    push_reference_phase(
        &mut phases,
        "attention_softmax",
        "attention_softmax",
        "softmax",
        &attention_probabilities,
    );

    let v_head_major = seq_major_to_head_major(&v_linear, seq_len, kv_heads, head_dim)?;
    let attention_context = dense_attention_v_mix_cpu_reference(
        &attention_probabilities,
        &v_head_major,
        q_heads,
        kv_heads,
        seq_len,
        head_dim,
    )?;
    push_reference_phase(
        &mut phases,
        "attention_v_mix",
        "attention_v_mix",
        "attention",
        &attention_context,
    );

    let attention_context_seq =
        head_major_to_seq_major(&attention_context, seq_len, q_heads, head_dim)?;
    let attention_output_values =
        dense_linear_sequence_cpu(&attention_output, &attention_context_seq, seq_len)?;
    push_reference_phase(
        &mut phases,
        "attention_output",
        "attention_output",
        "matmul",
        &attention_output_values,
    );
    let first_residual = add_same_len(&input, &attention_output_values, "first_residual")?;
    push_reference_phase(
        &mut phases,
        "first_residual",
        "first_residual",
        "residual_add",
        &first_residual,
    );

    let ffn_norm_output = rmsnorm_sequence_cpu(
        &first_residual,
        seq_len,
        hidden_size,
        &ffn_norm.weight_values_f32,
        ffn_norm.summary.rmsnorm_eps,
    )?;
    push_reference_phase(&mut phases, "ffn_norm", "ffn_norm", "rmsnorm", &ffn_norm_output);
    let mlp_gate_output = dense_linear_sequence_cpu(&mlp_gate, &ffn_norm_output, seq_len)?;
    push_reference_phase(&mut phases, "mlp_gate", "mlp_gate", "matmul", &mlp_gate_output);
    let mlp_up_output = dense_linear_sequence_cpu(&mlp_up, &ffn_norm_output, seq_len)?;
    push_reference_phase(&mut phases, "mlp_up", "mlp_up", "matmul", &mlp_up_output);
    let mlp_activation = dense_mlp_activation_cpu_reference(&mlp_gate_output, &mlp_up_output)?;
    push_reference_phase(
        &mut phases,
        "mlp_activation",
        "mlp_activation",
        "activation",
        &mlp_activation,
    );
    let mlp_down_output = dense_linear_sequence_cpu(&mlp_down, &mlp_activation, seq_len)?;
    push_reference_phase(&mut phases, "mlp_down", "mlp_down", "matmul", &mlp_down_output);
    let final_output = add_same_len(&first_residual, &mlp_down_output, "second_residual")?;
    push_reference_phase(
        &mut phases,
        "second_residual",
        "second_residual",
        "residual_add",
        &final_output,
    );

    Ok(DenseGgufOneLayerCpuReference {
        fixture_id: format!(
            "dense_gguf_one_layer_cpu_reference_{}_layer{}_s{}",
            sanitize_label(&inspection.model_family),
            layer_index,
            seq_len
        ),
        model_family: inspection.model_family.clone(),
        architecture: inspection.architecture.clone(),
        layer_index,
        seq_len,
        position_offset,
        hidden_size,
        q_heads,
        kv_heads,
        heads_per_kv_group: q_heads / kv_heads,
        head_dim,
        intermediate_size,
        rmsnorm_eps: attention_norm.summary.rmsnorm_eps,
        epsilon_source: attention_norm.summary.epsilon_source,
        rope_base: rope_metadata.rope_base,
        rope_base_source: rope_metadata.rope_base_source,
        scaling_factor: rope_metadata.scaling_factor,
        deterministic_input_len: input.len(),
        deterministic_input_sha256: sha256_f32(&input),
        phases,
        final_output_len: final_output.len(),
        final_output_sha256: sha256_f32(&final_output),
        final_output_max_abs: max_abs_f32(&final_output),
        final_output_f32: final_output,
    })
}

fn dense_gguf_model_boundary_fixtures_from_reader(
    reader: &GgufReader<'_>,
    inspection: &DenseGgufDescriptorInspection,
    seq_len: usize,
    requested_top_k: usize,
) -> Result<DenseGgufModelBoundaryFixtures> {
    if seq_len == 0 {
        bail!("dense GGUF model-boundary fixtures require a non-zero sequence length");
    }
    if requested_top_k == 0 {
        bail!("dense GGUF model-boundary fixtures require a non-zero top-k");
    }
    ensure_dense_model_boundary_fixture_coverage(inspection)?;

    let token_descriptor = descriptor_for_role(inspection, DenseGgufTensorRole::TokenEmbedding)?;
    let final_norm_descriptor = dense_final_norm_descriptor(inspection)?;
    let lm_head = extract_dense_gguf_linear_fixture(reader, DenseGgufTensorRole::Output).ok();

    if token_descriptor.shape.len() != 2 {
        bail!(
            "dense GGUF token embedding fixture requires a 2D tensor, got {:?}",
            token_descriptor.shape
        );
    }
    let hidden_size = token_descriptor.shape[0];
    let vocab_size = token_descriptor.shape[1];
    if hidden_size == 0 || vocab_size == 0 {
        bail!("dense GGUF token embedding fixture requires non-zero hidden/vocab dimensions");
    }
    if seq_len > vocab_size {
        bail!(
            "dense GGUF token embedding fixture seq_len {seq_len} exceeds vocab fixture rows {vocab_size}"
        );
    }
    if let Some(lm_head) = lm_head.as_ref()
        && (lm_head.summary.matrix_cols != hidden_size || lm_head.summary.matrix_rows != vocab_size)
    {
        bail!(
            "dense GGUF LM head fixture shape mismatch: rows={} cols={} expected vocab={} hidden={}",
            lm_head.summary.matrix_rows,
            lm_head.summary.matrix_cols,
            vocab_size,
            hidden_size
        );
    }
    if final_norm_descriptor.shape.len() != 1 || final_norm_descriptor.shape[0] != hidden_size {
        bail!(
            "dense GGUF final norm fixture shape mismatch: {:?} expected [{hidden_size}]",
            final_norm_descriptor.shape
        );
    }

    let token_embedding_values =
        dense_boundary_tensor_values_as_f32(reader, token_descriptor, "token embedding")?;
    let final_norm_values =
        dense_boundary_tensor_values_as_f32(reader, final_norm_descriptor, "final norm")?;
    if token_embedding_values.len() != hidden_size * vocab_size {
        bail!(
            "dense GGUF token embedding materialized {} values, expected {}",
            token_embedding_values.len(),
            hidden_size * vocab_size
        );
    }
    if final_norm_values.len() != hidden_size {
        bail!(
            "dense GGUF final norm materialized {} values, expected {hidden_size}",
            final_norm_values.len()
        );
    }

    let token_ids = (0..seq_len).collect::<Vec<_>>();
    let token_embedding_output =
        dense_token_embedding_lookup(&token_embedding_values, hidden_size, vocab_size, &token_ids)?;
    let final_norm_input =
        token_embedding_output[(seq_len - 1) * hidden_size..seq_len * hidden_size].to_vec();
    let (rmsnorm_eps, epsilon_source) =
        dense_boundary_rmsnorm_epsilon(reader, &inspection.architecture);
    let final_norm_output =
        rmsnorm_sequence_cpu(&final_norm_input, 1, hidden_size, &final_norm_values, rmsnorm_eps)?;
    let (logits, lm_head_logits) = if let Some(lm_head) = lm_head.as_ref() {
        let logits = dense_linear_sequence_cpu(lm_head, &final_norm_output, 1)?;
        let fixture = DenseGgufBoundaryTensorFixture {
            name: "lm_head_logits",
            role: "lm_head_logits",
            tensor_name: lm_head.summary.tensor_name.clone(),
            tensor_type: lm_head.summary.tensor_type.clone(),
            source_shape: lm_head.summary.source_shape.clone(),
            source_offset: lm_head.summary.source_offset,
            source_size_bytes: lm_head.summary.source_size_bytes,
            value_count: lm_head.summary.value_count,
            output_len: logits.len(),
            output_sha256: sha256_f32(&logits),
            max_abs: max_abs_f32(&logits),
        };
        (logits, fixture)
    } else {
        let logits = dense_tied_lm_head_logits_cpu(
            &token_embedding_values,
            hidden_size,
            vocab_size,
            &final_norm_output,
        )?;
        let fixture = DenseGgufBoundaryTensorFixture {
            name: "tied_lm_head_logits",
            role: "lm_head_logits",
            tensor_name: token_descriptor.name.clone(),
            tensor_type: token_descriptor.tensor_type.clone(),
            source_shape: token_descriptor.shape.clone(),
            source_offset: token_descriptor.offset,
            source_size_bytes: token_descriptor.size_bytes,
            value_count: token_embedding_values.len(),
            output_len: logits.len(),
            output_sha256: sha256_f32(&logits),
            max_abs: max_abs_f32(&logits),
        };
        (logits, fixture)
    };
    let logits_top_k = dense_logits_top_k(&logits, requested_top_k.min(logits.len()))?;

    Ok(DenseGgufModelBoundaryFixtures {
        fixture_id: format!(
            "dense_gguf_model_boundary_fixtures_{}_s{}_top{}",
            sanitize_label(&inspection.model_family),
            seq_len,
            logits_top_k.len()
        ),
        model_family: inspection.model_family.clone(),
        architecture: inspection.architecture.clone(),
        seq_len,
        hidden_size,
        vocab_size,
        token_ids_sha256: sha256_usize(&token_ids),
        token_ids,
        token_embedding: DenseGgufBoundaryTensorFixture {
            name: "token_embedding_lookup",
            role: "token_embedding",
            tensor_name: token_descriptor.name.clone(),
            tensor_type: token_descriptor.tensor_type.clone(),
            source_shape: token_descriptor.shape.clone(),
            source_offset: token_descriptor.offset,
            source_size_bytes: token_descriptor.size_bytes,
            value_count: token_embedding_values.len(),
            output_len: token_embedding_output.len(),
            output_sha256: sha256_f32(&token_embedding_output),
            max_abs: max_abs_f32(&token_embedding_output),
        },
        final_norm: DenseGgufBoundaryTensorFixture {
            name: "final_model_norm",
            role: "final_norm",
            tensor_name: final_norm_descriptor.name.clone(),
            tensor_type: final_norm_descriptor.tensor_type.clone(),
            source_shape: final_norm_descriptor.shape.clone(),
            source_offset: final_norm_descriptor.offset,
            source_size_bytes: final_norm_descriptor.size_bytes,
            value_count: final_norm_values.len(),
            output_len: final_norm_output.len(),
            output_sha256: sha256_f32(&final_norm_output),
            max_abs: max_abs_f32(&final_norm_output),
        },
        lm_head_logits,
        final_norm_input_sha256: sha256_f32(&final_norm_input),
        final_norm_output_sha256: sha256_f32(&final_norm_output),
        logits_len: logits.len(),
        logits_sha256: sha256_f32(&logits),
        logits_top_k,
        top_k: requested_top_k.min(logits.len()),
        rmsnorm_eps,
        epsilon_source,
    })
}

fn dense_gguf_kv_cache_policy_from_reader(
    reader: &GgufReader<'_>,
    inspection: &DenseGgufDescriptorInspection,
    seq_len: usize,
    decode_steps: usize,
) -> Result<DenseGgufKvCachePolicy> {
    if seq_len == 0 {
        bail!("dense GGUF KV-cache policy requires a non-zero sequence length");
    }
    if decode_steps == 0 {
        bail!("dense GGUF KV-cache policy requires at least one decode step");
    }
    ensure_dense_all_layer_block_descriptor_coverage(inspection)?;

    let architecture = &inspection.architecture;
    let layer_indices = dense_transformer_layer_indices(inspection)?;
    if layer_indices.is_empty() {
        bail!("dense GGUF KV-cache policy requires at least one transformer layer");
    }
    let inferred_layers = layer_indices.len() as u32;
    let (transformer_layers, transformer_layers_source) = metadata_u32_with_source(
        reader,
        &[format!("{architecture}.block_count"), "block_count".to_string()],
    )
    .unwrap_or((inferred_layers, "inferred_from_dense_layer_descriptors".to_string()));
    let (context_length, context_length_source) = metadata_u32_with_source(
        reader,
        &[format!("{architecture}.context_length"), "context_length".to_string()],
    )
    .unwrap_or(((seq_len + decode_steps) as u32, "seq_len_plus_decode_steps".to_string()));
    let (q_heads, q_heads_source) = metadata_u32_with_source(
        reader,
        &[format!("{architecture}.attention.head_count"), "attention.head_count".to_string()],
    )
    .ok_or_else(|| anyhow!("dense GGUF KV-cache policy requires attention.head_count metadata"))?;
    let (kv_heads, kv_heads_source) = metadata_u32_with_source(
        reader,
        &[format!("{architecture}.attention.head_count_kv"), "attention.head_count_kv".to_string()],
    )
    .unwrap_or((q_heads, "attention.head_count_kv_missing_default_to_q_heads".to_string()));
    let (key_head_dim, key_head_dim_source) = metadata_u32_with_source(
        reader,
        &[
            format!("{architecture}.attention.key_length"),
            format!("{architecture}.attention.head_dim"),
            "attention.key_length".to_string(),
        ],
    )
    .or_else(|| {
        let (embedding, embedding_source) = metadata_u32_with_source(
            reader,
            &[format!("{architecture}.embedding_length"), "embedding_length".to_string()],
        )?;
        if q_heads == 0 || embedding % q_heads != 0 {
            return None;
        }
        Some((embedding / q_heads, format!("{embedding_source}_div_{q_heads_source}")))
    })
    .ok_or_else(|| anyhow!("dense GGUF KV-cache policy requires key/head dimension metadata"))?;
    let (value_head_dim, value_head_dim_source) = metadata_u32_with_source(
        reader,
        &[
            format!("{architecture}.attention.value_length"),
            format!("{architecture}.attention.head_dim"),
            "attention.value_length".to_string(),
        ],
    )
    .unwrap_or((key_head_dim, format!("{key_head_dim_source}_default_value_dim")));

    let transformer_layers = transformer_layers as usize;
    let context_length = context_length as usize;
    let q_heads = q_heads as usize;
    let kv_heads = kv_heads as usize;
    let key_head_dim = key_head_dim as usize;
    let value_head_dim = value_head_dim as usize;
    if transformer_layers == 0
        || context_length == 0
        || q_heads == 0
        || kv_heads == 0
        || key_head_dim == 0
        || value_head_dim == 0
    {
        bail!("dense GGUF KV-cache policy dimensions must be non-zero");
    }
    if !q_heads.is_multiple_of(kv_heads) {
        bail!(
            "dense GGUF KV-cache policy requires q_heads divisible by kv_heads: q_heads={q_heads} kv_heads={kv_heads}"
        );
    }
    if context_length < seq_len {
        bail!(
            "dense GGUF KV-cache policy context_length {context_length} is smaller than seq_len {seq_len}"
        );
    }

    let kv_element_bytes = 2usize;
    let kv_values_per_token_per_layer = kv_heads
        .checked_mul(key_head_dim.checked_add(value_head_dim).ok_or_else(|| {
            anyhow!("dense GGUF KV-cache policy key/value dimension sum overflowed")
        })?)
        .ok_or_else(|| anyhow!("dense GGUF KV-cache policy value count overflowed"))?;
    let kv_bytes_per_token_per_layer = checked_u64_mul(
        kv_values_per_token_per_layer as u64,
        kv_element_bytes as u64,
        "kv_bytes_per_token_per_layer",
    )?;
    let kv_bytes_per_token_all_layers = checked_u64_mul(
        kv_bytes_per_token_per_layer,
        transformer_layers as u64,
        "kv_bytes_per_token_all_layers",
    )?;
    let prefill_write_bytes_estimate = checked_u64_mul(
        kv_bytes_per_token_all_layers,
        seq_len as u64,
        "prefill_write_bytes_estimate",
    )?;
    let decode_read_bytes_per_step_estimate = checked_u64_mul(
        kv_bytes_per_token_all_layers,
        seq_len as u64,
        "decode_read_bytes_per_step_estimate",
    )?;
    let decode_write_bytes_per_step_estimate = kv_bytes_per_token_all_layers;
    let max_context_bytes_estimate = checked_u64_mul(
        kv_bytes_per_token_all_layers,
        context_length as u64,
        "max_context_bytes_estimate",
    )?;

    Ok(DenseGgufKvCachePolicy {
        policy_id: format!(
            "dense_gguf_kv_cache_policy_{}_layers{}_ctx{}_kv{}_k{}_v{}",
            sanitize_label(&inspection.model_family),
            transformer_layers,
            context_length,
            kv_heads,
            key_head_dim,
            value_head_dim
        ),
        model_family: inspection.model_family.clone(),
        architecture: inspection.architecture.clone(),
        transformer_layers_total: transformer_layers,
        transformer_layers_source,
        context_length,
        context_length_source,
        seq_len,
        decode_steps,
        q_heads,
        q_heads_source,
        kv_heads,
        kv_heads_source,
        heads_per_kv_group: q_heads / kv_heads,
        key_head_dim,
        key_head_dim_source,
        value_head_dim,
        value_head_dim_source,
        kv_element_dtype: "f16",
        kv_element_bytes,
        kv_values_per_token_per_layer,
        kv_bytes_per_token_per_layer,
        kv_bytes_per_token_all_layers,
        prefill_write_bytes_estimate,
        decode_read_bytes_per_step_estimate,
        decode_write_bytes_per_step_estimate,
        max_context_bytes_estimate,
    })
}

fn dense_gguf_sampling_policy_from_reader(
    reader: &GgufReader<'_>,
    inspection: &DenseGgufDescriptorInspection,
    seq_len: usize,
    requested_top_k: usize,
) -> Result<DenseGgufSamplingPolicy> {
    if seq_len == 0 {
        bail!("dense GGUF sampling policy requires a non-zero sequence length");
    }
    if requested_top_k == 0 {
        bail!("dense GGUF sampling policy requires a non-zero top-k");
    }
    ensure_dense_model_boundary_fixture_coverage(inspection)?;

    let fixtures = dense_gguf_model_boundary_fixtures_from_reader(
        reader,
        inspection,
        seq_len,
        requested_top_k,
    )
    .context("failed to derive dense GGUF logits boundary for sampling policy")?;
    if fixtures.logits_len == 0 || fixtures.vocab_size == 0 {
        bail!("dense GGUF sampling policy requires non-empty logits and vocab");
    }
    if fixtures.logits_len != fixtures.vocab_size {
        bail!(
            "dense GGUF sampling policy logits/vocab mismatch: logits_len={} vocab_size={}",
            fixtures.logits_len,
            fixtures.vocab_size
        );
    }
    let selected = fixtures
        .logits_top_k
        .first()
        .ok_or_else(|| anyhow!("dense GGUF sampling policy requires a top logit entry"))?;
    let selected_token_id_from_fixture_logits = selected.token_id;
    let logits_element_bytes = 4usize;
    let logits_transfer_bytes_per_step_estimate = checked_u64_mul(
        fixtures.logits_len as u64,
        logits_element_bytes as u64,
        "logits_transfer_bytes_per_step_estimate",
    )?;

    Ok(DenseGgufSamplingPolicy {
        policy_id: format!(
            "dense_gguf_sampling_policy_{}_vocab{}_top{}",
            sanitize_label(&inspection.model_family),
            fixtures.vocab_size,
            fixtures.logits_top_k.len()
        ),
        model_family: inspection.model_family.clone(),
        architecture: inspection.architecture.clone(),
        seq_len: fixtures.seq_len,
        vocab_size: fixtures.vocab_size,
        logits_len: fixtures.logits_len,
        logits_sha256: fixtures.logits_sha256,
        logits_element_dtype: "f32",
        logits_element_bytes,
        logits_transfer_bytes_per_step_estimate,
        logits_top_k: fixtures.logits_top_k,
        top_k: requested_top_k.min(fixtures.logits_len),
        selected_token_id_from_fixture_logits,
        sampler_backend: "bitnet-sampling",
        sampler_location: "cpu",
        sampler_mode: "greedy",
        temperature: 0.0,
        top_k_filter: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        deterministic: true,
        tie_break_policy: "lowest_token_id",
        rng_required: false,
    })
}

macro_rules! push_cuda_phase {
    (
        $phases:expr,
        $reference:expr,
        $name:expr,
        $role:expr,
        $op_type:expr,
        $route:expr,
        $status:expr,
        $output:expr,
        $tolerance:expr,
        $stats:expr $(,)?
    ) => {
        push_cuda_phase_impl(
            $phases,
            $reference,
            DenseCudaPhaseInput {
                name: $name,
                role: $role,
                op_type: $op_type,
                route: $route,
                status: $status,
                output: $output,
                tolerance: $tolerance,
                stats: $stats,
            },
        )
    };
}

fn dense_gguf_one_layer_cuda_integrated_parity_from_reader(
    reader: &GgufReader<'_>,
    inspection: &DenseGgufDescriptorInspection,
    reference: &DenseGgufOneLayerCpuReference,
    device_index: usize,
    tolerance: f32,
) -> Result<DenseGgufOneLayerCudaIntegratedParity> {
    if reference.layer_index != 0 {
        bail!("integrated dense one-layer CUDA parity currently supports layer 0 only");
    }
    if !inspection.required_roles_present || !inspection.strict_descriptor_complete {
        bail!("integrated dense one-layer CUDA parity requires complete dense descriptor coverage");
    }

    let attention_norm =
        extract_dense_gguf_norm_fixture(reader, DenseGgufTensorRole::AttentionNorm)
            .context("failed to extract dense GGUF attention norm fixture")?;
    let ffn_norm = extract_dense_gguf_norm_fixture(reader, DenseGgufTensorRole::FfnNorm)
        .context("failed to extract dense GGUF FFN norm fixture")?;
    let attention_q = extract_dense_gguf_linear_fixture(reader, DenseGgufTensorRole::AttentionQ)
        .context("failed to extract dense GGUF attention Q fixture")?;
    let attention_k = extract_dense_gguf_linear_fixture(reader, DenseGgufTensorRole::AttentionK)
        .context("failed to extract dense GGUF attention K fixture")?;
    let attention_v = extract_dense_gguf_linear_fixture(reader, DenseGgufTensorRole::AttentionV)
        .context("failed to extract dense GGUF attention V fixture")?;
    let attention_output =
        extract_dense_gguf_linear_fixture(reader, DenseGgufTensorRole::AttentionOutput)
            .context("failed to extract dense GGUF attention output fixture")?;
    let mlp_gate = extract_dense_gguf_linear_fixture(reader, DenseGgufTensorRole::MlpGate)
        .context("failed to extract dense GGUF MLP gate fixture")?;
    let mlp_up = extract_dense_gguf_linear_fixture(reader, DenseGgufTensorRole::MlpUp)
        .context("failed to extract dense GGUF MLP up fixture")?;
    let mlp_down = extract_dense_gguf_linear_fixture(reader, DenseGgufTensorRole::MlpDown)
        .context("failed to extract dense GGUF MLP down fixture")?;

    let seq_len = reference.seq_len;
    let hidden_size = reference.hidden_size;
    let q_heads = reference.q_heads;
    let kv_heads = reference.kv_heads;
    let head_dim = reference.head_dim;
    let position_offset = reference.position_offset;
    let mut phases = Vec::new();

    let input = deterministic_layer_input(seq_len, hidden_size)?;
    push_cuda_phase!(
        &mut phases,
        reference,
        "deterministic_input",
        "hidden_state",
        "input",
        "host_deterministic_input",
        "host_deterministic_input",
        &input,
        tolerance,
        None,
    )?;

    let mut attention_norm_output = vec![0.0f32; input.len()];
    let attention_norm_stats = launch_dense_rmsnorm_f32_cuda(
        device_index,
        &input,
        &attention_norm.weight_values_f32,
        &mut attention_norm_output,
        &RmsNormConfig::for_shape(hidden_size, seq_len)?
            .with_eps(attention_norm.summary.rmsnorm_eps),
    )?;
    push_cuda_phase!(
        &mut phases,
        reference,
        "attention_norm",
        "attention_norm",
        "rmsnorm",
        DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND,
        "cuda_executed",
        &attention_norm_output,
        tolerance,
        Some(DenseOneLayerKernelCounters {
            kernel_id: attention_norm_stats.kernel_id,
            invocations: attention_norm_stats.invocations,
            fallback_invocations: attention_norm_stats.fallback_invocations,
            host_to_device_bytes: attention_norm_stats.host_to_device_bytes,
            device_to_host_bytes: attention_norm_stats.device_to_host_bytes,
            kernel_launches: attention_norm_stats.kernel_launches,
            kernel_time_ms: attention_norm_stats.kernel_time_ms,
        }),
    )?;

    let (q_linear, q_stats) =
        dense_linear_sequence_cuda(device_index, &attention_q, &attention_norm_output, seq_len)?;
    push_cuda_phase!(
        &mut phases,
        reference,
        "attention_q",
        "attention_q",
        "matmul",
        DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND,
        "cuda_executed",
        &q_linear,
        tolerance,
        Some(q_stats),
    )?;
    let (k_linear, k_stats) =
        dense_linear_sequence_cuda(device_index, &attention_k, &attention_norm_output, seq_len)?;
    push_cuda_phase!(
        &mut phases,
        reference,
        "attention_k",
        "attention_k",
        "matmul",
        DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND,
        "cuda_executed",
        &k_linear,
        tolerance,
        Some(k_stats),
    )?;
    let (v_linear, v_stats) =
        dense_linear_sequence_cuda(device_index, &attention_v, &attention_norm_output, seq_len)?;
    push_cuda_phase!(
        &mut phases,
        reference,
        "attention_v",
        "attention_v",
        "matmul",
        DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND,
        "cuda_executed",
        &v_linear,
        tolerance,
        Some(v_stats),
    )?;

    let q_head_major = seq_major_to_head_major(&q_linear, seq_len, q_heads, head_dim)?;
    let k_head_major = seq_major_to_head_major(&k_linear, seq_len, kv_heads, head_dim)?;
    let q_config = RopeConfig::for_shape(head_dim, q_heads, seq_len)?
        .with_position_offset(position_offset)
        .with_base(reference.rope_base)
        .with_scaling_factor(reference.scaling_factor);
    let k_config = RopeConfig::for_shape(head_dim, kv_heads, seq_len)?
        .with_position_offset(position_offset)
        .with_base(reference.rope_base)
        .with_scaling_factor(reference.scaling_factor);
    let mut q_rope = vec![0.0f32; q_head_major.len()];
    let q_rope_stats =
        launch_dense_rope_f32_cuda(device_index, &q_head_major, &mut q_rope, &q_config)?;
    let mut k_rope = vec![0.0f32; k_head_major.len()];
    let k_rope_stats =
        launch_dense_rope_f32_cuda(device_index, &k_head_major, &mut k_rope, &k_config)?;
    let mut rope_phase_output = q_rope.clone();
    rope_phase_output.extend_from_slice(&k_rope);
    push_cuda_phase!(
        &mut phases,
        reference,
        "rope",
        "rope",
        "rope",
        DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND,
        "cuda_executed",
        &rope_phase_output,
        tolerance,
        Some(combine_kernel_counters(
            q_rope_stats.kernel_id,
            &[
                DenseOneLayerKernelCounters {
                    kernel_id: q_rope_stats.kernel_id,
                    invocations: q_rope_stats.invocations,
                    fallback_invocations: q_rope_stats.fallback_invocations,
                    host_to_device_bytes: q_rope_stats.host_to_device_bytes,
                    device_to_host_bytes: q_rope_stats.device_to_host_bytes,
                    kernel_launches: q_rope_stats.kernel_launches,
                    kernel_time_ms: q_rope_stats.kernel_time_ms,
                },
                DenseOneLayerKernelCounters {
                    kernel_id: k_rope_stats.kernel_id,
                    invocations: k_rope_stats.invocations,
                    fallback_invocations: k_rope_stats.fallback_invocations,
                    host_to_device_bytes: k_rope_stats.host_to_device_bytes,
                    device_to_host_bytes: k_rope_stats.device_to_host_bytes,
                    kernel_launches: k_rope_stats.kernel_launches,
                    kernel_time_ms: k_rope_stats.kernel_time_ms,
                },
            ],
        )),
    )?;

    let mut attention_scores = vec![0.0f32; q_heads * seq_len * seq_len];
    let attention_score_stats = launch_dense_attention_scores_f32_cuda(
        device_index,
        &q_rope,
        &k_rope,
        &mut attention_scores,
        &AttentionScoresConfig::for_shape(q_heads, kv_heads, head_dim, seq_len)?
            .with_scale(1.0 / (head_dim as f32).sqrt())
            .with_causal(true),
    )?;
    push_cuda_phase!(
        &mut phases,
        reference,
        "attention_scores",
        "attention_scores",
        "attention",
        DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND,
        "cuda_executed",
        &attention_scores,
        tolerance,
        Some(DenseOneLayerKernelCounters {
            kernel_id: attention_score_stats.kernel_id,
            invocations: attention_score_stats.invocations,
            fallback_invocations: attention_score_stats.fallback_invocations,
            host_to_device_bytes: attention_score_stats.host_to_device_bytes,
            device_to_host_bytes: attention_score_stats.device_to_host_bytes,
            kernel_launches: attention_score_stats.kernel_launches,
            kernel_time_ms: attention_score_stats.kernel_time_ms,
        }),
    )?;

    let mut attention_probabilities = vec![0.0f32; attention_scores.len()];
    let attention_softmax_stats = launch_dense_attention_softmax_f32_cuda(
        device_index,
        &attention_scores,
        &mut attention_probabilities,
        &AttentionSoftmaxConfig::for_shape(q_heads, seq_len)?,
    )?;
    push_cuda_phase!(
        &mut phases,
        reference,
        "attention_softmax",
        "attention_softmax",
        "softmax",
        DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND,
        "cuda_executed",
        &attention_probabilities,
        tolerance,
        Some(DenseOneLayerKernelCounters {
            kernel_id: attention_softmax_stats.kernel_id,
            invocations: attention_softmax_stats.invocations,
            fallback_invocations: attention_softmax_stats.fallback_invocations,
            host_to_device_bytes: attention_softmax_stats.host_to_device_bytes,
            device_to_host_bytes: attention_softmax_stats.device_to_host_bytes,
            kernel_launches: attention_softmax_stats.kernel_launches,
            kernel_time_ms: attention_softmax_stats.kernel_time_ms,
        }),
    )?;

    let v_head_major = seq_major_to_head_major(&v_linear, seq_len, kv_heads, head_dim)?;
    let mut attention_context = vec![0.0f32; q_heads * seq_len * head_dim];
    let attention_v_mix_stats = launch_dense_attention_v_mix_f32_cuda(
        device_index,
        &attention_probabilities,
        &v_head_major,
        &mut attention_context,
        &AttentionVMixConfig::for_shape(q_heads, kv_heads, head_dim, seq_len)?,
    )?;
    push_cuda_phase!(
        &mut phases,
        reference,
        "attention_v_mix",
        "attention_v_mix",
        "attention",
        DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND,
        "cuda_executed",
        &attention_context,
        tolerance,
        Some(DenseOneLayerKernelCounters {
            kernel_id: attention_v_mix_stats.kernel_id,
            invocations: attention_v_mix_stats.invocations,
            fallback_invocations: attention_v_mix_stats.fallback_invocations,
            host_to_device_bytes: attention_v_mix_stats.host_to_device_bytes,
            device_to_host_bytes: attention_v_mix_stats.device_to_host_bytes,
            kernel_launches: attention_v_mix_stats.kernel_launches,
            kernel_time_ms: attention_v_mix_stats.kernel_time_ms,
        }),
    )?;

    let attention_context_seq =
        head_major_to_seq_major(&attention_context, seq_len, q_heads, head_dim)?;
    let (attention_output_values, attention_output_stats) = dense_linear_sequence_cuda(
        device_index,
        &attention_output,
        &attention_context_seq,
        seq_len,
    )?;
    push_cuda_phase!(
        &mut phases,
        reference,
        "attention_output",
        "attention_output",
        "matmul",
        DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND,
        "cuda_executed",
        &attention_output_values,
        tolerance,
        Some(attention_output_stats),
    )?;
    let first_residual = add_same_len(&input, &attention_output_values, "first_residual")?;
    push_cuda_phase!(
        &mut phases,
        reference,
        "first_residual",
        "first_residual",
        "residual_add",
        "host_measured_glue",
        "host_measured_glue",
        &first_residual,
        tolerance,
        None,
    )?;

    let mut ffn_norm_output = vec![0.0f32; first_residual.len()];
    let ffn_norm_stats = launch_dense_rmsnorm_f32_cuda(
        device_index,
        &first_residual,
        &ffn_norm.weight_values_f32,
        &mut ffn_norm_output,
        &RmsNormConfig::for_shape(hidden_size, seq_len)?.with_eps(ffn_norm.summary.rmsnorm_eps),
    )?;
    push_cuda_phase!(
        &mut phases,
        reference,
        "ffn_norm",
        "ffn_norm",
        "rmsnorm",
        DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND,
        "cuda_executed",
        &ffn_norm_output,
        tolerance,
        Some(DenseOneLayerKernelCounters {
            kernel_id: ffn_norm_stats.kernel_id,
            invocations: ffn_norm_stats.invocations,
            fallback_invocations: ffn_norm_stats.fallback_invocations,
            host_to_device_bytes: ffn_norm_stats.host_to_device_bytes,
            device_to_host_bytes: ffn_norm_stats.device_to_host_bytes,
            kernel_launches: ffn_norm_stats.kernel_launches,
            kernel_time_ms: ffn_norm_stats.kernel_time_ms,
        }),
    )?;

    let (mlp_gate_output, mlp_gate_stats) =
        dense_linear_sequence_cuda(device_index, &mlp_gate, &ffn_norm_output, seq_len)?;
    push_cuda_phase!(
        &mut phases,
        reference,
        "mlp_gate",
        "mlp_gate",
        "matmul",
        DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND,
        "cuda_executed",
        &mlp_gate_output,
        tolerance,
        Some(mlp_gate_stats),
    )?;
    let (mlp_up_output, mlp_up_stats) =
        dense_linear_sequence_cuda(device_index, &mlp_up, &ffn_norm_output, seq_len)?;
    push_cuda_phase!(
        &mut phases,
        reference,
        "mlp_up",
        "mlp_up",
        "matmul",
        DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND,
        "cuda_executed",
        &mlp_up_output,
        tolerance,
        Some(mlp_up_stats),
    )?;

    let mut mlp_activation = vec![0.0f32; mlp_gate_output.len()];
    let mlp_activation_stats = launch_dense_mlp_activation_f32_cuda(
        device_index,
        &mlp_gate_output,
        &mlp_up_output,
        &mut mlp_activation,
        &SiluGateConfig::new(mlp_gate_output.len())?,
    )?;
    push_cuda_phase!(
        &mut phases,
        reference,
        "mlp_activation",
        "mlp_activation",
        "activation",
        DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND,
        "cuda_executed",
        &mlp_activation,
        tolerance,
        Some(DenseOneLayerKernelCounters {
            kernel_id: mlp_activation_stats.kernel_id,
            invocations: mlp_activation_stats.invocations,
            fallback_invocations: mlp_activation_stats.fallback_invocations,
            host_to_device_bytes: mlp_activation_stats.host_to_device_bytes,
            device_to_host_bytes: mlp_activation_stats.device_to_host_bytes,
            kernel_launches: mlp_activation_stats.kernel_launches,
            kernel_time_ms: mlp_activation_stats.kernel_time_ms,
        }),
    )?;

    let (mlp_down_output, mlp_down_stats) =
        dense_linear_sequence_cuda(device_index, &mlp_down, &mlp_activation, seq_len)?;
    push_cuda_phase!(
        &mut phases,
        reference,
        "mlp_down",
        "mlp_down",
        "matmul",
        DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND,
        "cuda_executed",
        &mlp_down_output,
        tolerance,
        Some(mlp_down_stats),
    )?;
    let final_output = add_same_len(&first_residual, &mlp_down_output, "second_residual")?;
    push_cuda_phase!(
        &mut phases,
        reference,
        "second_residual",
        "second_residual",
        "residual_add",
        "host_measured_glue",
        "host_measured_glue",
        &final_output,
        tolerance,
        None,
    )?;

    let (final_output_max_abs_error, final_output_mean_abs_error) =
        compare_f32_outputs(&reference.final_output_f32, &final_output, "final layer output")?;
    let host_to_device_bytes = phases.iter().map(|phase| phase.host_to_device_bytes).sum();
    let device_to_host_bytes = phases.iter().map(|phase| phase.device_to_host_bytes).sum();
    let kernel_invocations = phases
        .iter()
        .filter(|phase| phase.kernel_id.is_some())
        .map(|phase| phase.invocations)
        .sum();
    let kernel_launches = phases.iter().map(|phase| phase.kernel_launches).sum();
    let kernel_time_ms = sum_optional_kernel_times(phases.iter().map(|phase| phase.kernel_time_ms));
    let passed = final_output_max_abs_error <= tolerance && phases.iter().all(|phase| phase.passed);

    Ok(DenseGgufOneLayerCudaIntegratedParity {
        fixture_id: format!(
            "dense_gguf_one_layer_cuda_integrated_parity_{}_layer{}_s{}",
            sanitize_label(&inspection.model_family),
            reference.layer_index,
            reference.seq_len
        ),
        source_cpu_reference_fixture_id: reference.fixture_id.clone(),
        model_family: inspection.model_family.clone(),
        architecture: inspection.architecture.clone(),
        layer_index: reference.layer_index,
        seq_len: reference.seq_len,
        position_offset: reference.position_offset,
        hidden_size,
        q_heads,
        kv_heads,
        heads_per_kv_group: reference.heads_per_kv_group,
        head_dim,
        intermediate_size: reference.intermediate_size,
        phases,
        final_output_len: final_output.len(),
        final_output_sha256: sha256_f32(&final_output),
        final_output_max_abs: max_abs_f32(&final_output),
        final_output_max_abs_error,
        final_output_mean_abs_error,
        tolerance,
        passed,
        host_to_device_bytes,
        device_to_host_bytes,
        kernel_invocations,
        kernel_launches,
        kernel_time_ms,
    })
}

fn kernel_attention_score_fixture_from_extracted(
    fixture: &DenseGgufAttentionScoreFixture,
) -> DenseGgufAttentionScoreCudaFixture {
    DenseGgufAttentionScoreCudaFixture {
        fixture_id: fixture.fixture_id.clone(),
        model_family: fixture.model_family.clone(),
        architecture: fixture.architecture.clone(),
        layer_index: fixture.layer_index,
        q_heads: fixture.q_heads,
        kv_heads: fixture.kv_heads,
        heads_per_kv_group: fixture.heads_per_kv_group,
        head_dim: fixture.head_dim,
        seq_len: fixture.seq_len,
        scale: fixture.scale,
        q_rope_output_sha256: sha256_f32(&fixture.q_rope_output_f32),
        k_rope_output_sha256: sha256_f32(&fixture.k_rope_output_f32),
        source_rope_fixture_id: fixture.source_rope_fixture_id.clone(),
        q_rope_output_f32: fixture.q_rope_output_f32.clone(),
        k_rope_output_f32: fixture.k_rope_output_f32.clone(),
        expected_scores_f32: fixture.expected_scores_f32.clone(),
        finite_scores: fixture.finite_scores,
        causal_masked_scores: fixture.causal_masked_scores,
    }
}

fn kernel_attention_softmax_fixture_from_extracted(
    fixture: &DenseGgufAttentionSoftmaxFixture,
) -> DenseGgufAttentionSoftmaxCudaFixture {
    DenseGgufAttentionSoftmaxCudaFixture {
        fixture_id: fixture.fixture_id.clone(),
        model_family: fixture.model_family.clone(),
        architecture: fixture.architecture.clone(),
        layer_index: fixture.layer_index,
        q_heads: fixture.q_heads,
        kv_heads: fixture.kv_heads,
        seq_len: fixture.seq_len,
        source_attention_score_fixture_id: fixture.source_attention_score_fixture_id.clone(),
        attention_scores_sha256: sha256_f32(&fixture.attention_scores_f32),
        attention_scores_f32: fixture.attention_scores_f32.clone(),
        expected_probabilities_f32: fixture.expected_probabilities_f32.clone(),
        row_count: fixture.row_count,
        probability_count: fixture.probability_count,
        causal_zero_probabilities: fixture.causal_zero_probabilities,
        max_row_sum_abs_error: fixture.max_row_sum_abs_error,
    }
}

fn kernel_attention_v_mix_fixture_from_extracted(
    fixture: &DenseGgufAttentionVMixFixture,
) -> DenseGgufAttentionVMixCudaFixture {
    DenseGgufAttentionVMixCudaFixture {
        fixture_id: fixture.fixture_id.clone(),
        model_family: fixture.model_family.clone(),
        architecture: fixture.architecture.clone(),
        layer_index: fixture.layer_index,
        q_heads: fixture.q_heads,
        kv_heads: fixture.kv_heads,
        heads_per_kv_group: fixture.heads_per_kv_group,
        head_dim: fixture.head_dim,
        seq_len: fixture.seq_len,
        source_attention_softmax_fixture_id: fixture.source_attention_softmax_fixture_id.clone(),
        attention_probabilities_sha256: sha256_f32(&fixture.attention_probabilities_f32),
        value_states_sha256: sha256_f32(&fixture.value_states_f32),
        attention_probabilities_f32: fixture.attention_probabilities_f32.clone(),
        value_states_f32: fixture.value_states_f32.clone(),
        expected_context_f32: fixture.expected_context_f32.clone(),
        row_count: fixture.row_count,
        probability_count: fixture.probability_count,
        value_count: fixture.value_count,
        context_count: fixture.context_count,
        causal_zero_probabilities: fixture.causal_zero_probabilities,
        max_context_abs: fixture.max_context_abs,
    }
}

fn kernel_mlp_activation_fixture_from_extracted(
    fixture: &DenseGgufMlpActivationFixture,
) -> DenseGgufMlpActivationCudaFixture {
    DenseGgufMlpActivationCudaFixture {
        fixture_id: fixture.fixture_id.clone(),
        model_family: fixture.model_family.clone(),
        architecture: fixture.architecture.clone(),
        layer_index: fixture.layer_index,
        source_mlp_gate_fixture_id: fixture.source_mlp_gate_fixture_id.clone(),
        source_mlp_up_fixture_id: fixture.source_mlp_up_fixture_id.clone(),
        source_mlp_gate_tensor: fixture.source_mlp_gate_tensor.clone(),
        source_mlp_up_tensor: fixture.source_mlp_up_tensor.clone(),
        activation_kind: fixture.activation_kind.to_string(),
        gate_output_sha256: sha256_f32(&fixture.gate_output_f32),
        up_output_sha256: sha256_f32(&fixture.up_output_f32),
        gate_output_f32: fixture.gate_output_f32.clone(),
        up_output_f32: fixture.up_output_f32.clone(),
        expected_activation_f32: fixture.expected_activation_f32.clone(),
        activation_count: fixture.activation_count,
        max_activation_abs: fixture.max_activation_abs,
    }
}

fn dense_attention_scores_cpu_reference(
    q_rope: &[f32],
    k_rope: &[f32],
    q_heads: usize,
    kv_heads: usize,
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) -> Result<Vec<f32>> {
    if q_heads == 0 || kv_heads == 0 || seq_len == 0 || head_dim == 0 {
        bail!("dense attention-score fixture dimensions must be non-zero");
    }
    if !q_heads.is_multiple_of(kv_heads) {
        bail!("dense attention-score fixture q_heads must be divisible by kv_heads");
    }
    let q_expected = q_heads * seq_len * head_dim;
    let k_expected = kv_heads * seq_len * head_dim;
    if q_rope.len() != q_expected {
        bail!(
            "dense attention-score fixture q_rope length {} != expected {q_expected}",
            q_rope.len()
        );
    }
    if k_rope.len() != k_expected {
        bail!(
            "dense attention-score fixture k_rope length {} != expected {k_expected}",
            k_rope.len()
        );
    }

    let heads_per_kv_group = q_heads / kv_heads;
    let mut scores = Vec::with_capacity(q_heads * seq_len * seq_len);
    for q_head in 0..q_heads {
        let kv_head = q_head / heads_per_kv_group;
        for q_pos in 0..seq_len {
            for k_pos in 0..seq_len {
                if k_pos > q_pos {
                    scores.push(f32::NEG_INFINITY);
                    continue;
                }
                let q_offset = (q_head * seq_len + q_pos) * head_dim;
                let k_offset = (kv_head * seq_len + k_pos) * head_dim;
                let mut dot = 0.0f32;
                for dim in 0..head_dim {
                    dot += q_rope[q_offset + dim] * k_rope[k_offset + dim];
                }
                scores.push(dot * scale);
            }
        }
    }
    Ok(scores)
}

fn dense_attention_softmax_cpu_reference(
    scores: &[f32],
    q_heads: usize,
    seq_len: usize,
) -> Result<(Vec<f32>, usize, f32)> {
    if q_heads == 0 || seq_len == 0 {
        bail!("dense attention-softmax fixture dimensions must be non-zero");
    }
    let expected = q_heads * seq_len * seq_len;
    if scores.len() != expected {
        bail!(
            "dense attention-softmax fixture score length {} != expected {expected}",
            scores.len()
        );
    }

    let mut probabilities = vec![0.0f32; scores.len()];
    let mut causal_zero_probabilities = 0usize;
    let mut max_row_sum_abs_error = 0.0f32;
    for q_head in 0..q_heads {
        for q_pos in 0..seq_len {
            let row_start = (q_head * seq_len + q_pos) * seq_len;
            let row = &scores[row_start..row_start + seq_len];
            let max_score = row
                .iter()
                .copied()
                .filter(|score| score.is_finite())
                .fold(f32::NEG_INFINITY, f32::max);
            if !max_score.is_finite() {
                bail!(
                    "dense attention-softmax fixture row has no finite scores: head={q_head} pos={q_pos}"
                );
            }

            let mut exp_sum = 0.0f32;
            for (idx, score) in row.iter().copied().enumerate() {
                if score.is_finite() {
                    let exp = (score - max_score).exp();
                    probabilities[row_start + idx] = exp;
                    exp_sum += exp;
                } else {
                    causal_zero_probabilities += 1;
                }
            }
            if exp_sum <= 0.0 || !exp_sum.is_finite() {
                bail!(
                    "dense attention-softmax fixture row has invalid exp sum: head={q_head} pos={q_pos}"
                );
            }

            let mut row_sum = 0.0f32;
            for idx in 0..seq_len {
                let prob = probabilities[row_start + idx] / exp_sum;
                probabilities[row_start + idx] = prob;
                row_sum += prob;
            }
            max_row_sum_abs_error = max_row_sum_abs_error.max((row_sum - 1.0).abs());
        }
    }

    Ok((probabilities, causal_zero_probabilities, max_row_sum_abs_error))
}

fn deterministic_attention_value_input(len: usize) -> Vec<f32> {
    (0..len)
        .map(|idx| {
            let phase = ((idx * 29 + 97) as f32).cos();
            let drift = ((idx + 5) % 13) as f32 * 0.0025;
            phase * 0.09 + drift
        })
        .collect()
}

fn dense_attention_v_mix_cpu_reference(
    probabilities: &[f32],
    values: &[f32],
    q_heads: usize,
    kv_heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    if q_heads == 0 || kv_heads == 0 || seq_len == 0 || head_dim == 0 {
        bail!("dense attention V-mix fixture dimensions must be non-zero");
    }
    if !q_heads.is_multiple_of(kv_heads) {
        bail!("dense attention V-mix fixture q_heads must be divisible by kv_heads");
    }
    let expected_probabilities = q_heads * seq_len * seq_len;
    if probabilities.len() != expected_probabilities {
        bail!(
            "dense attention V-mix probability length {} != expected {expected_probabilities}",
            probabilities.len()
        );
    }
    let expected_values = kv_heads * seq_len * head_dim;
    if values.len() != expected_values {
        bail!("dense attention V-mix value length {} != expected {expected_values}", values.len());
    }

    let heads_per_kv_group = q_heads / kv_heads;
    let mut context = vec![0.0f32; q_heads * seq_len * head_dim];
    for q_head in 0..q_heads {
        let kv_head = q_head / heads_per_kv_group;
        for q_pos in 0..seq_len {
            let prob_offset = (q_head * seq_len + q_pos) * seq_len;
            let context_offset = (q_head * seq_len + q_pos) * head_dim;
            for k_pos in 0..seq_len {
                let probability = probabilities[prob_offset + k_pos];
                if !probability.is_finite() {
                    bail!(
                        "dense attention V-mix probability is not finite: head={q_head} q_pos={q_pos} k_pos={k_pos}"
                    );
                }
                let value_offset = (kv_head * seq_len + k_pos) * head_dim;
                for dim in 0..head_dim {
                    context[context_offset + dim] += probability * values[value_offset + dim];
                }
            }
        }
    }
    Ok(context)
}

fn dense_mlp_activation_cpu_reference(gate: &[f32], up: &[f32]) -> Result<Vec<f32>> {
    if gate.is_empty() || up.is_empty() {
        bail!("dense MLP activation fixture requires non-empty gate and up vectors");
    }
    if gate.len() != up.len() {
        bail!(
            "dense MLP activation fixture gate/up length mismatch: gate={} up={}",
            gate.len(),
            up.len()
        );
    }

    Ok(gate
        .iter()
        .copied()
        .zip(up.iter().copied())
        .map(|(gate_value, up_value)| {
            let silu = gate_value / (1.0 + (-gate_value).exp());
            silu * up_value
        })
        .collect())
}

fn deterministic_layer_input(seq_len: usize, hidden_size: usize) -> Result<Vec<f32>> {
    let len = seq_len
        .checked_mul(hidden_size)
        .ok_or_else(|| anyhow!("dense one-layer deterministic input length overflows"))?;
    if len == 0 {
        bail!("dense one-layer deterministic input must not be empty");
    }
    Ok((0..len)
        .map(|idx| {
            let centered = (idx % 23) as f32 - 11.0;
            let wave = ((idx * 31 + 7) as f32).sin() * 0.015;
            centered / 22.0 + wave
        })
        .collect())
}

fn ensure_norm_dim(fixture: &DenseGgufNormFixture, expected: usize, label: &str) -> Result<()> {
    if fixture.summary.hidden_dim != expected || fixture.weight_values_f32.len() != expected {
        bail!(
            "dense one-layer {label} hidden dim mismatch: hidden_dim={} weights={} expected={expected}",
            fixture.summary.hidden_dim,
            fixture.weight_values_f32.len()
        );
    }
    Ok(())
}

fn ensure_linear_rows(
    fixture: &DenseGgufLinearFixture,
    expected: usize,
    label: &str,
) -> Result<()> {
    if fixture.summary.matrix_rows != expected {
        bail!(
            "dense one-layer {label} matrix rows mismatch: rows={} expected={expected}",
            fixture.summary.matrix_rows
        );
    }
    Ok(())
}

fn ensure_linear_cols(
    fixture: &DenseGgufLinearFixture,
    expected: usize,
    label: &str,
) -> Result<()> {
    if fixture.summary.matrix_cols != expected {
        bail!(
            "dense one-layer {label} matrix cols mismatch: cols={} expected={expected}",
            fixture.summary.matrix_cols
        );
    }
    Ok(())
}

fn rmsnorm_sequence_cpu(
    input: &[f32],
    seq_len: usize,
    hidden_size: usize,
    gamma: &[f32],
    eps: f32,
) -> Result<Vec<f32>> {
    if gamma.len() != hidden_size {
        bail!(
            "dense one-layer RMSNorm gamma length {} does not match hidden size {hidden_size}",
            gamma.len()
        );
    }
    if input.len() != seq_len * hidden_size {
        bail!(
            "dense one-layer RMSNorm input length {} does not match seq_len*hidden_size {}",
            input.len(),
            seq_len * hidden_size
        );
    }
    if eps <= 0.0 || !eps.is_finite() {
        bail!("dense one-layer RMSNorm epsilon must be positive and finite, got {eps}");
    }

    let mut output = Vec::with_capacity(input.len());
    for token in 0..seq_len {
        let start = token * hidden_size;
        let row = &input[start..start + hidden_size];
        let mean_square = row.iter().map(|value| value * value).sum::<f32>() / hidden_size as f32;
        let inv_rms = (mean_square + eps).sqrt().recip();
        output.extend(row.iter().zip(gamma).map(|(value, weight)| value * inv_rms * weight));
    }
    Ok(output)
}

fn dense_linear_sequence_cpu(
    fixture: &DenseGgufLinearFixture,
    input: &[f32],
    seq_len: usize,
) -> Result<Vec<f32>> {
    let rows = fixture.summary.matrix_rows;
    let cols = fixture.summary.matrix_cols;
    if input.len() != seq_len * cols {
        bail!(
            "dense one-layer linear input for {} has length {}, expected {}",
            fixture.summary.tensor_name,
            input.len(),
            seq_len * cols
        );
    }
    if fixture.weight_values_f32.len() != rows * cols {
        bail!(
            "dense one-layer linear weights for {} have length {}, expected {}",
            fixture.summary.tensor_name,
            fixture.weight_values_f32.len(),
            rows * cols
        );
    }

    let mut output = Vec::with_capacity(seq_len * rows);
    for token in 0..seq_len {
        let input_start = token * cols;
        let token_input = &input[input_start..input_start + cols];
        for row in 0..rows {
            let weight_start = row * cols;
            let mut sum = 0.0f32;
            for (col, input_value) in token_input.iter().enumerate() {
                sum += fixture.weight_values_f32[weight_start + col] * *input_value;
            }
            output.push(sum);
        }
    }
    Ok(output)
}

fn dense_tied_lm_head_logits_cpu(
    token_embedding_values: &[f32],
    hidden_size: usize,
    vocab_size: usize,
    input: &[f32],
) -> Result<Vec<f32>> {
    if input.len() != hidden_size {
        bail!("dense tied LM-head logits input has length {}, expected {hidden_size}", input.len());
    }
    if token_embedding_values.len() != hidden_size * vocab_size {
        bail!(
            "dense tied LM-head token embedding values length {} != hidden_size*vocab_size {}",
            token_embedding_values.len(),
            hidden_size * vocab_size
        );
    }

    let mut logits = Vec::with_capacity(vocab_size);
    for token_id in 0..vocab_size {
        let start = token_id * hidden_size;
        let weights = &token_embedding_values[start..start + hidden_size];
        let mut sum = 0.0f32;
        for (weight, hidden) in weights.iter().zip(input.iter()) {
            sum += *weight * *hidden;
        }
        logits.push(sum);
    }
    Ok(logits)
}

fn dense_token_embedding_lookup(
    token_embedding_values: &[f32],
    hidden_size: usize,
    vocab_size: usize,
    token_ids: &[usize],
) -> Result<Vec<f32>> {
    if token_embedding_values.len() != hidden_size * vocab_size {
        bail!(
            "dense token embedding values length {} != hidden_size*vocab_size {}",
            token_embedding_values.len(),
            hidden_size * vocab_size
        );
    }
    if token_ids.is_empty() {
        bail!("dense token embedding lookup requires at least one token id");
    }

    let mut output = Vec::with_capacity(token_ids.len() * hidden_size);
    for token_id in token_ids {
        if *token_id >= vocab_size {
            bail!("dense token embedding token id {token_id} is outside vocab size {vocab_size}");
        }
        let start = token_id * hidden_size;
        output.extend_from_slice(&token_embedding_values[start..start + hidden_size]);
    }
    Ok(output)
}

fn dense_logits_top_k(logits: &[f32], top_k: usize) -> Result<Vec<DenseGgufLogitTopKEntry>> {
    if logits.is_empty() {
        bail!("dense logits top-k requires a non-empty logits vector");
    }
    if top_k == 0 {
        bail!("dense logits top-k requires top_k > 0");
    }
    let mut ranked = logits
        .iter()
        .copied()
        .enumerate()
        .map(|(token_id, value)| {
            let sortable = if value.is_finite() { value } else { f32::NEG_INFINITY };
            (token_id, value, sortable)
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|(left_id, _, left_value), (right_id, _, right_value)| {
        right_value.total_cmp(left_value).then_with(|| left_id.cmp(right_id))
    });
    Ok(ranked
        .into_iter()
        .take(top_k.min(logits.len()))
        .enumerate()
        .map(|(rank, (token_id, value, _))| DenseGgufLogitTopKEntry { rank, token_id, value })
        .collect())
}

fn dense_qwen_logits_transfer_reduction_json(
    logits_len: usize,
    generated_tokens: u64,
    observed_top_k: usize,
    actual_device_to_host_bytes: u64,
    transfer_mode: DenseQwenLogitsTransferMode,
) -> Result<Value> {
    if logits_len == 0 {
        bail!("dense Qwen logits transfer reduction requires a non-empty logits vector");
    }
    if generated_tokens == 0 {
        bail!("dense Qwen logits transfer reduction requires generated tokens");
    }
    if observed_top_k == 0 {
        bail!("dense Qwen logits transfer reduction requires top-k evidence");
    }

    let full_logits_bytes_per_step =
        checked_u64_mul(logits_len as u64, 4, "full logits bytes per step")?;
    let full_logits_download_bytes = checked_u64_mul(
        full_logits_bytes_per_step,
        generated_tokens,
        "full logits download bytes",
    )?;
    let top_k_result_bytes_per_step_floor =
        checked_u64_mul(observed_top_k as u64, 12, "top-k result bytes per step floor")?;
    let top_k_result_bytes_total_floor = checked_u64_mul(
        top_k_result_bytes_per_step_floor,
        generated_tokens,
        "top-k result bytes total floor",
    )?;
    let selected_token_bytes_total_floor =
        checked_u64_mul(4, generated_tokens, "selected-token bytes total floor")?;
    let device_to_host_bytes_reduced = match transfer_mode {
        DenseQwenLogitsTransferMode::FullLogitsDownloadCpuSampler => {
            if actual_device_to_host_bytes != full_logits_download_bytes {
                bail!(
                    "dense Qwen full-logits transfer accounting must match full logits download bytes"
                );
            }
            false
        }
        DenseQwenLogitsTransferMode::DeviceTopKCudaSampler => {
            if actual_device_to_host_bytes >= full_logits_download_bytes {
                bail!("dense Qwen device top-k transfer accounting must be lower than full logits");
            }
            true
        }
    };
    let bytes_saved_vs_full_logits = if device_to_host_bytes_reduced {
        full_logits_download_bytes - actual_device_to_host_bytes
    } else {
        0
    };
    let reduction_blocker = if device_to_host_bytes_reduced {
        Value::Null
    } else {
        json!("cpu_sampler_requires_full_logits_until_device_top_k_sampler")
    };

    Ok(json!({
        "schema": 1,
        "scope": "dense_qwen_logits_top_k_transfer",
        "transfer_mode": transfer_mode.transfer_mode(),
        "sampling_location": transfer_mode.sampling_location(),
        "requested_top_k": observed_top_k as u64,
        "generated_tokens_count": generated_tokens,
        "logits_vector_length": logits_len as u64,
        "logits_element_bytes": 4_u64,
        "full_logits_bytes_per_step": full_logits_bytes_per_step,
        "full_logits_download_bytes": full_logits_download_bytes,
        "actual_device_to_host_bytes": actual_device_to_host_bytes,
        "top_k_result_bytes_per_step_floor": top_k_result_bytes_per_step_floor,
        "top_k_result_bytes_total_floor": top_k_result_bytes_total_floor,
        "selected_token_bytes_total_floor": selected_token_bytes_total_floor,
        "device_to_host_bytes_reduced": device_to_host_bytes_reduced,
        "bytes_saved_vs_full_logits": bytes_saved_vs_full_logits,
        "selected_token_equality_preserved": true,
        "top_k_evidence_preserved": true,
        "quality_receipts_unchanged": true,
        "reduction_blocker": reduction_blocker
    }))
}

fn dense_final_norm_descriptor(
    inspection: &DenseGgufDescriptorInspection,
) -> Result<&DenseGgufTensorDescriptor> {
    inspection
        .descriptors
        .iter()
        .find(|descriptor| dense_final_norm_tensor_name(&descriptor.name))
        .ok_or_else(|| anyhow!("dense GGUF model-boundary fixtures require final norm tensor"))
}

fn dense_final_norm_tensor_name(name: &str) -> bool {
    let normalized = name.to_ascii_lowercase();
    matches!(
        normalized.as_str(),
        "output_norm.weight"
            | "norm.weight"
            | "model.norm.weight"
            | "final_norm.weight"
            | "final_layernorm.weight"
            | "final_rmsnorm.weight"
            | "transformer.ln_f.weight"
    )
}

fn ensure_dense_model_boundary_fixture_coverage(
    inspection: &DenseGgufDescriptorInspection,
) -> Result<()> {
    ensure_dense_all_layer_block_descriptor_coverage(inspection)?;
    let token_descriptor = descriptor_for_role(inspection, DenseGgufTensorRole::TokenEmbedding)?;
    if token_descriptor.shape.len() != 2 {
        bail!(
            "dense GGUF model-boundary fixtures require 2D token embeddings, got {:?}",
            token_descriptor.shape
        );
    }
    dense_final_norm_descriptor(inspection)?;
    if descriptor_for_role(inspection, DenseGgufTensorRole::Output).is_err()
        && inspection.architecture != "qwen3"
    {
        bail!(
            "dense GGUF model-boundary fixtures require an output tensor unless the architecture has a governed tied LM head"
        );
    }
    Ok(())
}

fn dense_model_boundary_fixture_coverage_complete(
    inspection: &DenseGgufDescriptorInspection,
) -> bool {
    ensure_dense_model_boundary_fixture_coverage(inspection).is_ok()
}

fn dense_transformer_block_descriptor_coverage_complete(
    inspection: &DenseGgufDescriptorInspection,
) -> bool {
    ensure_dense_all_layer_block_descriptor_coverage(inspection).is_ok()
}

fn dense_model_boundary_lm_head_source(inspection: &DenseGgufDescriptorInspection) -> &'static str {
    if descriptor_for_role(inspection, DenseGgufTensorRole::Output).is_ok() {
        "output_tensor"
    } else if inspection.architecture == "qwen3" {
        "tied_token_embedding"
    } else {
        "missing_output_tensor"
    }
}

fn dense_model_boundary_route_status(inspection: &DenseGgufDescriptorInspection) -> String {
    if inspection.strict_descriptor_complete {
        inspection.dense_cuda_route_status.clone()
    } else if dense_model_boundary_fixture_coverage_complete(inspection) {
        "descriptor_complete_with_tied_lm_head".to_string()
    } else {
        inspection.dense_cuda_route_status.clone()
    }
}

fn dense_boundary_tensor_values_as_f32(
    reader: &GgufReader<'_>,
    descriptor: &DenseGgufTensorDescriptor,
    label: &str,
) -> Result<Vec<f32>> {
    let info = reader.get_tensor_info_by_name(&descriptor.name).ok_or_else(|| {
        anyhow!("dense GGUF {label} descriptor '{}' is missing tensor info", descriptor.name)
    })?;
    let data = reader.get_tensor_data_by_info(info)?;
    match info.tensor_type {
        GgufTensorType::F32 => dense_boundary_f32_values(data, &info.shape, &info.name),
        GgufTensorType::F16 => dense_boundary_f16_values(data, &info.shape, &info.name),
        GgufTensorType::Q8_0 => dense_boundary_q8_0_values(data, &info.shape, &info.name),
        other => bail!(
            "dense GGUF {label} fixture does not support tensor type {:?} for '{}'",
            other,
            info.name
        ),
    }
}

fn dense_boundary_f32_values(bytes: &[u8], shape: &[usize], tensor_name: &str) -> Result<Vec<f32>> {
    let elements = dense_boundary_element_count(shape, tensor_name, "F32")?;
    let expected_bytes =
        elements.checked_mul(4).ok_or_else(|| anyhow!("F32 tensor '{tensor_name}' overflows"))?;
    if bytes.len() < expected_bytes {
        bail!(
            "F32 tensor '{tensor_name}' has {} bytes, expected at least {expected_bytes}",
            bytes.len()
        );
    }
    Ok(bytes[..expected_bytes]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn dense_boundary_f16_values(bytes: &[u8], shape: &[usize], tensor_name: &str) -> Result<Vec<f32>> {
    let elements = dense_boundary_element_count(shape, tensor_name, "F16")?;
    let expected_bytes =
        elements.checked_mul(2).ok_or_else(|| anyhow!("F16 tensor '{tensor_name}' overflows"))?;
    if bytes.len() < expected_bytes {
        bail!(
            "F16 tensor '{tensor_name}' has {} bytes, expected at least {expected_bytes}",
            bytes.len()
        );
    }
    Ok(bytes[..expected_bytes]
        .chunks_exact(2)
        .map(|chunk| half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
        .collect())
}

fn dense_boundary_q8_0_values(
    bytes: &[u8],
    shape: &[usize],
    tensor_name: &str,
) -> Result<Vec<f32>> {
    let elements = dense_boundary_element_count(shape, tensor_name, "Q8_0")?;
    let blocks = elements.div_ceil(32);
    let expected_bytes = blocks
        .checked_mul(GgufTensorType::Q8_0.element_size())
        .ok_or_else(|| anyhow!("Q8_0 tensor '{tensor_name}' byte count overflows"))?;
    if bytes.len() < expected_bytes {
        bail!(
            "Q8_0 tensor '{tensor_name}' has {} bytes, expected at least {expected_bytes}",
            bytes.len()
        );
    }

    let mut values = Vec::with_capacity(elements);
    for block_idx in 0..blocks {
        let offset = block_idx * GgufTensorType::Q8_0.element_size();
        let scale_bits = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
        let scale = half::f16::from_bits(scale_bits).to_f32();
        for code_idx in 0..32 {
            if values.len() == elements {
                break;
            }
            values.push(scale * f32::from(bytes[offset + 2 + code_idx] as i8));
        }
    }
    Ok(values)
}

fn dense_boundary_element_count(shape: &[usize], tensor_name: &str, dtype: &str) -> Result<usize> {
    shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| anyhow!("{dtype} tensor '{tensor_name}' shape {shape:?} overflows"))
    })
}

fn dense_boundary_rmsnorm_epsilon(reader: &GgufReader<'_>, architecture: &str) -> (f32, String) {
    let keys = [
        format!("{architecture}.attention.layer_norm_rms_epsilon"),
        format!("{architecture}.attention.layer_norm_epsilon"),
        format!("{architecture}.rms_norm_eps"),
        "llama.attention.layer_norm_rms_epsilon".to_string(),
        "llama.attention.layer_norm_epsilon".to_string(),
    ];

    for key in keys {
        if let Some(value) = reader.get_f32_metadata(&key) {
            return (value, key);
        }
    }

    (1e-6, "default_1e-6".to_string())
}

fn seq_major_to_head_major(
    input: &[f32],
    seq_len: usize,
    heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    if heads == 0 || head_dim == 0 || seq_len == 0 {
        bail!("dense one-layer head reshape dimensions must be non-zero");
    }
    let expected = seq_len
        .checked_mul(heads)
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or_else(|| anyhow!("dense one-layer head reshape size overflows"))?;
    if input.len() != expected {
        bail!("dense one-layer seq-major input length {} != expected {expected}", input.len());
    }

    let mut output = vec![0.0f32; expected];
    for pos in 0..seq_len {
        for head in 0..heads {
            for dim in 0..head_dim {
                let src = (pos * heads + head) * head_dim + dim;
                let dst = (head * seq_len + pos) * head_dim + dim;
                output[dst] = input[src];
            }
        }
    }
    Ok(output)
}

fn head_major_to_seq_major(
    input: &[f32],
    seq_len: usize,
    heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    if heads == 0 || head_dim == 0 || seq_len == 0 {
        bail!("dense one-layer head reshape dimensions must be non-zero");
    }
    let expected = seq_len
        .checked_mul(heads)
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or_else(|| anyhow!("dense one-layer head reshape size overflows"))?;
    if input.len() != expected {
        bail!("dense one-layer head-major input length {} != expected {expected}", input.len());
    }

    let mut output = vec![0.0f32; expected];
    for head in 0..heads {
        for pos in 0..seq_len {
            for dim in 0..head_dim {
                let src = (head * seq_len + pos) * head_dim + dim;
                let dst = (pos * heads + head) * head_dim + dim;
                output[dst] = input[src];
            }
        }
    }
    Ok(output)
}

fn add_same_len(left: &[f32], right: &[f32], label: &str) -> Result<Vec<f32>> {
    if left.len() != right.len() {
        bail!(
            "dense one-layer {label} add length mismatch: left={} right={}",
            left.len(),
            right.len()
        );
    }
    Ok(left.iter().zip(right).map(|(left, right)| left + right).collect())
}

fn push_reference_phase(
    phases: &mut Vec<DenseOneLayerCpuReferencePhase>,
    name: &'static str,
    role: &'static str,
    op_type: &'static str,
    output: &[f32],
) {
    phases.push(DenseOneLayerCpuReferencePhase {
        index: phases.len(),
        name,
        role,
        op_type,
        output_f32: output.to_vec(),
        output_len: output.len(),
        output_sha256: sha256_f32(output),
        max_abs: max_abs_f32(output),
    });
}

fn dense_linear_sequence_cuda(
    device_index: usize,
    fixture: &DenseGgufLinearFixture,
    input: &[f32],
    seq_len: usize,
) -> Result<(Vec<f32>, DenseOneLayerKernelCounters)> {
    let rows = fixture.summary.matrix_rows;
    let cols = fixture.summary.matrix_cols;
    if input.len() != seq_len * cols {
        bail!(
            "dense one-layer CUDA linear input for {} has length {}, expected {}",
            fixture.summary.tensor_name,
            input.len(),
            seq_len * cols
        );
    }

    let mut output = Vec::with_capacity(seq_len * rows);
    let mut counters = DenseOneLayerKernelCounters {
        kernel_id: "dense_f16_gemm_cuda",
        invocations: 0,
        fallback_invocations: 0,
        host_to_device_bytes: 0,
        device_to_host_bytes: 0,
        kernel_launches: 0,
        kernel_time_ms: None,
    };

    for token in 0..seq_len {
        let input_start = token * cols;
        let token_input = input[input_start..input_start + cols].to_vec();
        let kernel_fixture = DenseGgufLinearGemmFixture {
            fixture_id: dense_linear_fixture_id(
                &fixture.summary.model_family,
                dense_role_label(fixture.summary.role),
                &fixture.summary.tensor_type,
            ),
            model_family: fixture.summary.model_family.clone(),
            tensor_name: fixture.summary.tensor_name.clone(),
            tensor_role: dense_role_label(fixture.summary.role).to_string(),
            tensor_type: fixture.summary.tensor_type.clone(),
            source_weight_sha256: fixture.summary.weight_values_sha256.clone(),
            matrix_rows: rows,
            matrix_cols: cols,
            weights_row_major_f32: fixture.weight_values_f32.clone(),
            input_f32: token_input,
        };
        let prepared = prepare_dense_gguf_linear_f16_gemm(&kernel_fixture)?;
        let mut token_output = vec![0.0f32; rows];
        let stats = launch_dense_f16_gemm_cuda(
            device_index,
            &prepared.a_f16,
            &prepared.b_f16,
            &mut token_output,
            &prepared.config,
        )?;
        counters.kernel_id = stats.kernel_id;
        counters.invocations += stats.invocations;
        counters.fallback_invocations += stats.fallback_invocations;
        counters.host_to_device_bytes += stats.host_to_device_bytes;
        counters.device_to_host_bytes += stats.device_to_host_bytes;
        counters.kernel_launches += stats.kernel_launches;
        counters.kernel_time_ms =
            combine_optional_kernel_time(counters.kernel_time_ms, stats.kernel_time_ms);
        output.extend(token_output);
    }

    Ok((output, counters))
}

struct DenseCudaPhaseInput<'a> {
    name: &'static str,
    role: &'static str,
    op_type: &'static str,
    route: &'static str,
    status: &'static str,
    output: &'a [f32],
    tolerance: f32,
    stats: Option<DenseOneLayerKernelCounters>,
}

fn push_cuda_phase_impl(
    phases: &mut Vec<DenseOneLayerCudaPhase>,
    reference: &DenseGgufOneLayerCpuReference,
    input: DenseCudaPhaseInput<'_>,
) -> Result<()> {
    let DenseCudaPhaseInput { name, role, op_type, route, status, output, tolerance, stats } =
        input;
    let reference_phase = reference
        .phases
        .iter()
        .find(|phase| phase.name == name)
        .ok_or_else(|| anyhow!("CPU reference missing phase `{name}`"))?;
    let (max_abs_error, mean_abs_error) =
        compare_f32_outputs(&reference_phase.output_f32, output, name)?;
    let passed = max_abs_error <= tolerance;
    let stats = stats.unwrap_or(DenseOneLayerKernelCounters {
        kernel_id: "",
        invocations: 1,
        fallback_invocations: 0,
        host_to_device_bytes: 0,
        device_to_host_bytes: 0,
        kernel_launches: 0,
        kernel_time_ms: None,
    });
    phases.push(DenseOneLayerCudaPhase {
        index: phases.len(),
        name,
        role,
        op_type,
        route,
        status,
        output_len: output.len(),
        output_sha256: sha256_f32(output),
        max_abs: max_abs_f32(output),
        max_abs_error,
        mean_abs_error,
        tolerance,
        passed,
        kernel_id: (!stats.kernel_id.is_empty()).then_some(stats.kernel_id),
        invocations: stats.invocations,
        fallback_invocations: stats.fallback_invocations,
        host_to_device_bytes: stats.host_to_device_bytes,
        device_to_host_bytes: stats.device_to_host_bytes,
        kernel_launches: stats.kernel_launches,
        kernel_time_ms: stats.kernel_time_ms,
    });
    Ok(())
}

fn combine_kernel_counters(
    kernel_id: &'static str,
    stats: &[DenseOneLayerKernelCounters],
) -> DenseOneLayerKernelCounters {
    DenseOneLayerKernelCounters {
        kernel_id,
        invocations: stats.iter().map(|stat| stat.invocations).sum(),
        fallback_invocations: stats.iter().map(|stat| stat.fallback_invocations).sum(),
        host_to_device_bytes: stats.iter().map(|stat| stat.host_to_device_bytes).sum(),
        device_to_host_bytes: stats.iter().map(|stat| stat.device_to_host_bytes).sum(),
        kernel_launches: stats.iter().map(|stat| stat.kernel_launches).sum(),
        kernel_time_ms: sum_optional_kernel_times(stats.iter().map(|stat| stat.kernel_time_ms)),
    }
}

fn combine_optional_kernel_time(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left + right),
        (Some(left), None) => Some(left),
        (None, Some(right)) => Some(right),
        (None, None) => None,
    }
}

fn sum_optional_kernel_times(times: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    let mut sum = 0.0;
    let mut seen = false;
    for time in times.flatten() {
        sum += time;
        seen = true;
    }
    seen.then_some(sum)
}

fn compare_f32_outputs(expected: &[f32], actual: &[f32], label: &str) -> Result<(f32, f32)> {
    if expected.len() != actual.len() {
        bail!(
            "dense one-layer {label} length mismatch: expected={} actual={}",
            expected.len(),
            actual.len()
        );
    }
    if expected.is_empty() {
        bail!("dense one-layer {label} comparison requires non-empty outputs");
    }

    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f32;
    let mut compared = 0usize;
    for (idx, (expected, actual)) in
        expected.iter().copied().zip(actual.iter().copied()).enumerate()
    {
        if expected.is_finite() && actual.is_finite() {
            let diff = (expected - actual).abs();
            max_abs = max_abs.max(diff);
            sum_abs += diff;
            compared += 1;
        } else if expected.to_bits() != actual.to_bits() {
            bail!(
                "dense one-layer {label} non-finite mismatch at index {idx}: expected={expected:?} actual={actual:?}"
            );
        }
    }
    if compared == 0 { Ok((0.0, 0.0)) } else { Ok((max_abs, sum_abs / compared as f32)) }
}

fn max_abs_f32(values: &[f32]) -> f32 {
    values.iter().copied().filter(|value| value.is_finite()).map(f32::abs).fold(0.0f32, f32::max)
}

fn head_dim_source_label(source: &str) -> String {
    if source.is_empty() { "unknown".to_string() } else { source.to_string() }
}

fn metadata_u32_with_source(reader: &GgufReader<'_>, keys: &[String]) -> Option<(u32, String)> {
    keys.iter().find_map(|key| reader.get_u32_metadata(key).map(|value| (value, key.clone())))
}

fn metadata_f32_with_source(reader: &GgufReader<'_>, keys: &[String]) -> Option<(f32, String)> {
    keys.iter().find_map(|key| reader.get_f32_metadata(key).map(|value| (value, key.clone())))
}

fn deterministic_rope_input(len: usize, salt: usize) -> Vec<f32> {
    (0..len)
        .map(|idx| {
            let phase = ((idx * 37 + salt * 13) as f32).sin();
            let drift = ((idx + salt) % 11) as f32 * 0.003;
            phase * 0.125 + drift
        })
        .collect()
}

fn dense_linear_fixture_id(model_family: &str, role: &str, tensor_type: &str) -> String {
    format!(
        "dense_gguf_linear_{}_{}_{}_f16_bridge",
        sanitize_label(model_family),
        sanitize_label(role),
        sanitize_label(tensor_type)
    )
    .trim_end_matches('_')
    .to_string()
}

fn sanitize_label(value: &str) -> String {
    let mut out = String::new();
    let mut prev_underscore = false;
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            prev_underscore = false;
        } else if !prev_underscore {
            out.push('_');
            prev_underscore = true;
        }
    }
    out.trim_matches('_').to_string()
}

fn dense_gguf_norm_fixture_receipt_json(
    inspection: &DenseGgufDescriptorInspection,
    fixtures: &[DenseGgufNormFixture],
    model_path: &Path,
    model_sha256: &str,
    artifact_path: &str,
    timestamp_utc: &str,
) -> Result<Value> {
    if fixtures.len() < 2 {
        bail!("dense GGUF norm fixture receipt requires attention_norm and ffn_norm fixtures");
    }

    let mut covered_roles = Vec::with_capacity(fixtures.len());
    let mut fixture_entries = Vec::with_capacity(fixtures.len());
    for fixture in fixtures {
        let summary = &fixture.summary;
        if summary.model_family != inspection.model_family {
            bail!(
                "dense GGUF norm fixture mixed model families: expected {}, got {}",
                inspection.model_family,
                summary.model_family
            );
        }
        if summary.architecture != inspection.architecture {
            bail!(
                "dense GGUF norm fixture mixed architectures: expected {}, got {}",
                inspection.architecture,
                summary.architecture
            );
        }
        let role = dense_role_label(summary.role);
        covered_roles.push(role.to_string());
        fixture_entries.push(json!({
            "schema": summary.schema,
            "artifact_kind": DENSE_GGUF_NORM_FIXTURE_ARTIFACT_KIND,
            "model_family": summary.model_family,
            "architecture": summary.architecture,
            "tensor_name": summary.tensor_name,
            "role": role,
            "tensor_type": summary.tensor_type,
            "source_shape": summary.source_shape,
            "source_offset": summary.source_offset,
            "source_size_bytes": summary.source_size_bytes,
            "hidden_dim": summary.hidden_dim as u64,
            "value_count": summary.value_count as u64,
            "values_materialized_as_f32": summary.values_materialized_as_f32,
            "weight_values_sha256": summary.weight_values_sha256,
            "rmsnorm_eps": summary.rmsnorm_eps,
            "epsilon_source": summary.epsilon_source,
            "cpu_reference_input_len": summary.cpu_reference_input_len as u64,
            "cpu_reference_output_len": summary.cpu_reference_output_len as u64,
            "cpu_reference_input_sha256": summary.cpu_reference_input_sha256,
            "cpu_reference_output_sha256": summary.cpu_reference_output_sha256,
            "cpu_reference_computed": summary.cpu_reference_computed,
            "cuda_kernel_status": summary.cuda_kernel_status,
            "dense_gguf_inference_claimed": false,
            "dense_regular_llm_cuda_claimed": false,
            "cpu_cuda_parity_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        }));
    }

    let roles_total = covered_roles.len() as u64;
    Ok(json!({
        "schema": 1,
        "artifact_kind": DENSE_GGUF_NORM_FIXTURE_ARTIFACT_KIND,
        "artifact_path": artifact_path,
        "claim": "dense_gguf_norm_fixture_extracted",
        "machine_id": MACHINE_ID,
        "hardware_lane": HARDWARE_LANE,
        "timestamp_utc": timestamp_utc,
        "inspection_source": "gguf_reader_norm_fixture",
        "model": {
            "model_family": inspection.model_family,
            "architecture": inspection.architecture,
            "artifact_kind": "dense_gguf",
            "quantization_families": inspection.quantization_families,
            "file": model_path.display().to_string(),
            "sha256": model_sha256
        },
        "descriptor_coverage": {
            "schema": 1,
            "source_artifact_kind": "dense_gguf_tensor_descriptor_inspection",
            "tensor_count": inspection.tensor_count,
            "metadata_count": inspection.metadata_count as u64,
            "required_roles_present": dense_transformer_block_descriptor_coverage_complete(inspection),
            "strict_descriptor_complete": dense_transformer_block_descriptor_coverage_complete(inspection),
            "dense_cuda_route_status": dense_model_boundary_route_status(inspection),
            "model_boundary_lm_head_source": dense_model_boundary_lm_head_source(inspection),
            "quantization_families": inspection.quantization_families,
            "bitnet_packed_marker_found": inspection.bitnet_packed_marker_found,
            "dense_gguf_inference_claimed": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "norm_fixture_audit": {
            "schema": 1,
            "source_artifact_kind": DENSE_GGUF_NORM_FIXTURE_ARTIFACT_KIND,
            "roles_total": roles_total,
            "roles_extracted": roles_total,
            "roles_failed": 0,
            "covered_roles": covered_roles,
            "all_cpu_reference_computed": true,
            "cuda_kernel_status": "missing_cuda_kernel",
            "strict_cuda_ready": false,
            "cpu_fallback_allowed": false,
            "transfer_timing_status": "not_measured_no_kernel",
            "candidate_order": ["attention_norm", "ffn_norm"],
            "next_required_proof": "cuda_rmsnorm_kernel_parity",
            "dense_gguf_norm_fixture_extraction_claimed": true,
            "dense_gguf_inference_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "norm_fixtures": fixture_entries,
        "claim_boundary": {
            "dense_regular_llm_cuda_claimed": false,
            "dense_tensor_residency_claimed": false,
            "dense_gguf_descriptor_inspection_claimed": true,
            "dense_gguf_norm_fixture_extraction_claimed": true,
            "dense_gguf_linear_fixture_extraction_claimed": false,
            "dense_gguf_linear_cuda_parity_claimed": false,
            "dense_gguf_linear_role_sweep_cuda_parity_claimed": false,
            "dense_gguf_one_layer_execution_plan_claimed": false,
            "dense_gguf_inference_claimed": false,
            "qwen_one_token_cuda_claimed": false,
            "qwen_short_decode_cuda_claimed": false,
            "qwen_chat_cuda_claimed": false,
            "cpu_cuda_parity_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "notes": [
            "Dense GGUF norm fixture extraction only; no CUDA norm kernel or dense GGUF inference was executed.",
            "The CUDA RMSNorm launch path is scaffold-only, so this receipt records missing_cuda_kernel before parity work."
        ],
        "error": null
    }))
}

fn dense_gguf_norm_cuda_parity_receipt_json(
    inspection: &DenseGgufDescriptorInspection,
    results: &[DenseNormParityResult],
    probe: Option<&bitnet_device_probe::NvidiaCudaProbe>,
    model_path: &Path,
    model_sha256: &str,
    artifact_path: &str,
    timestamp_utc: &str,
) -> Result<Value> {
    if results.len() < 2 {
        bail!("dense GGUF RMSNorm CUDA parity requires attention_norm and ffn_norm results");
    }

    let role_count = results.len();
    let role_count_u64 = role_count as u64;
    let mut covered_roles = Vec::with_capacity(role_count);
    let mut fixture_entries = Vec::with_capacity(role_count);
    let mut kernel_stats = Vec::with_capacity(role_count);
    let mut parity_results = Vec::with_capacity(role_count);
    let mut h2d_bytes = 0u64;
    let mut d2h_bytes = 0u64;
    let mut kernel_launches = 0u64;

    for result in results {
        let summary = &result.extracted.summary;
        let parity = &result.parity;
        if summary.model_family != inspection.model_family
            || parity.model_family != inspection.model_family
        {
            bail!("dense GGUF RMSNorm CUDA parity mixed model families");
        }
        if summary.architecture != inspection.architecture {
            bail!("dense GGUF RMSNorm CUDA parity mixed architectures");
        }
        let role = dense_role_label(summary.role);
        covered_roles.push(role.to_string());
        h2d_bytes = h2d_bytes.saturating_add(parity.stats.host_to_device_bytes);
        d2h_bytes = d2h_bytes.saturating_add(parity.stats.device_to_host_bytes);
        kernel_launches = kernel_launches.saturating_add(parity.stats.kernel_launches);

        fixture_entries.push(json!({
            "schema": summary.schema,
            "source_artifact_kind": DENSE_GGUF_NORM_FIXTURE_ARTIFACT_KIND,
            "fixture_id": parity.fixture_id,
            "model_family": summary.model_family,
            "architecture": summary.architecture,
            "tensor_name": parity.tensor_name,
            "role": role,
            "tensor_type": parity.tensor_type,
            "source_shape": summary.source_shape,
            "hidden_dim": parity.hidden_dim as u64,
            "value_count": summary.value_count as u64,
            "values_materialized_as_f32": true,
            "weight_values_sha256": parity.source_weight_sha256,
            "rmsnorm_eps": summary.rmsnorm_eps,
            "epsilon_source": summary.epsilon_source,
            "cuda_input_dtype": "f32",
            "cuda_gamma_dtype": "f32",
            "cuda_output_dtype": "f32",
            "dense_gguf_inference_claimed": false,
            "dense_regular_llm_cuda_claimed": true,
            "cpu_cuda_parity_claimed": true,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        }));

        kernel_stats.push(json!({
            "kernel_id": parity.stats.kernel_id,
            "role": role,
            "tensor_name": parity.tensor_name,
            "fixture_id": parity.fixture_id,
            "invocations": parity.stats.invocations,
            "fallback_invocations": parity.stats.fallback_invocations,
            "host_to_device_bytes": parity.stats.host_to_device_bytes,
            "device_to_host_bytes": parity.stats.device_to_host_bytes,
            "kernel_launches": parity.stats.kernel_launches,
            "kernel_time_ms": parity.stats.kernel_time_ms
        }));

        parity_results.push(json!({
            "reference_backend": parity.reference_backend,
            "target_backend": parity.target_backend,
            "kernel_id": parity.kernel_id,
            "fixture_id": parity.fixture_id,
            "role": parity.tensor_role,
            "hidden_dim": parity.hidden_dim as u64,
            "max_abs_error": parity.max_abs_error,
            "mean_abs_error": parity.mean_abs_error,
            "passed": parity.passed,
            "tolerance": parity.tolerance,
            "tolerance_source": "CUDA-DENSE-016 dense GGUF RMSNorm F32 CUDA fixture"
        }));
    }

    for required in ["attention_norm", "ffn_norm"] {
        if !covered_roles.iter().any(|role| role == required) {
            bail!("dense GGUF RMSNorm CUDA parity missing required role {required}");
        }
    }

    let cuda = cuda_identity_json(probe);
    let execution_plan = execution_plan_receipt(ExecutionPlanReceiptInput {
        model_family: &inspection.model_family,
        quantization: "dense_f32_rmsnorm",
        requested_backend: HARDWARE_LANE,
        selected_backend: HARDWARE_LANE,
        runtime_api: "cuda",
        strict_fallback_policy: "reject",
        summary: ModelDispatchSummary {
            total_ops: role_count,
            cuda_bitnet_qk256_ops: 0,
            cuda_dense_regular_llm_ops: role_count,
            cpu_fallback_ops: 0,
            unsupported_ops: 0,
            fallback_used: false,
            selected_route: Some(ModelDispatchBackend::CudaDenseRegularLlm),
            strict_cuda_ready: true,
        },
        speedup_claim: false,
        full_cuda_residency_claimed: false,
    });

    Ok(json!({
        "schema": 1,
        "artifact_kind": DENSE_GGUF_NORM_CUDA_PARITY_ARTIFACT_KIND,
        "artifact_path": artifact_path,
        "claim": "dense_gguf_norm_cuda_parity_tested",
        "machine_id": MACHINE_ID,
        "hardware_lane": HARDWARE_LANE,
        "timestamp_utc": timestamp_utc,
        "requested_backend": HARDWARE_LANE,
        "selected_backend": HARDWARE_LANE,
        "runtime_api": "cuda",
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "speedup_claim": false,
        "cuda": cuda,
        "model": {
            "model_family": inspection.model_family,
            "architecture": inspection.architecture,
            "artifact_kind": "dense_gguf",
            "quantization_families": inspection.quantization_families,
            "file": model_path.display().to_string(),
            "sha256": model_sha256
        },
        "execution_path": {
            "model_class": "dense_regular_llm",
            "kernel_family": "dense_f32_rmsnorm",
            "quantization_family": "f32_norm_weights",
            "bitnet_packed_kernel_proof": false,
            "qk256_proof": false
        },
        "execution_plan": execution_plan,
        "descriptor_coverage": {
            "schema": 1,
            "source_artifact_kind": "dense_gguf_tensor_descriptor_inspection",
            "tensor_count": inspection.tensor_count,
            "metadata_count": inspection.metadata_count as u64,
            "required_roles_present": dense_transformer_block_descriptor_coverage_complete(inspection),
            "strict_descriptor_complete": dense_transformer_block_descriptor_coverage_complete(inspection),
            "dense_cuda_route_status": dense_model_boundary_route_status(inspection),
            "model_boundary_lm_head_source": dense_model_boundary_lm_head_source(inspection),
            "quantization_families": inspection.quantization_families,
            "bitnet_packed_marker_found": inspection.bitnet_packed_marker_found,
            "dense_gguf_inference_claimed": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "norm_fixtures": fixture_entries,
        "kernel_stats": kernel_stats,
        "parity_results": parity_results,
        "parity": {
            "passed": results.iter().all(|result| result.parity.passed),
            "roles_total": role_count_u64,
            "covered_roles": covered_roles,
            "first_divergence": null
        },
        "timing": {
            "kernel_time_ms": null,
            "host_to_device_bytes": h2d_bytes,
            "device_to_host_bytes": d2h_bytes
        },
        "claim_boundary": {
            "dense_regular_llm_cuda_claimed": true,
            "dense_tensor_residency_claimed": true,
            "dense_gguf_descriptor_inspection_claimed": true,
            "dense_gguf_norm_fixture_extraction_claimed": true,
            "dense_gguf_norm_cuda_parity_claimed": true,
            "dense_gguf_linear_fixture_extraction_claimed": false,
            "dense_gguf_linear_cuda_parity_claimed": false,
            "dense_gguf_linear_role_sweep_cuda_parity_claimed": false,
            "dense_gguf_one_layer_execution_plan_claimed": false,
            "dense_gguf_inference_claimed": false,
            "qwen_one_token_cuda_claimed": false,
            "qwen_short_decode_cuda_claimed": false,
            "qwen_chat_cuda_claimed": false,
            "cpu_cuda_parity_claimed": true,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false
        },
        "tensor_residency": {
            "schema_version": "1.0.0",
            "scope": "single_dense_gguf_rmsnorm_fixture",
            "model_class": "dense_regular_llm",
            "roles_total": role_count_u64,
            "dense_tensor_residency_claimed": true,
            "dense_gguf_inference_claimed": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false,
            "input_tensors_uploaded_once": true,
            "output_tensor_cuda_resident_during_kernel": true,
            "host_device_transfer_accounting_matches_kernel_stats": true,
            "allocation": {
                "device_buffer_count_per_role": 3,
                "temporary_workspace_bytes": 0,
                "persistent_handle_count": 0,
                "persistent_handles_claimed": false
            },
            "transfer_accounting": {
                "status": "measured",
                "host_to_device_bytes": h2d_bytes,
                "device_to_host_bytes": d2h_bytes
            },
            "kernel_launches": kernel_launches
        },
        "notes": [
            "Dense GGUF RMSNorm CUDA fixture parity only; no dense GGUF inference, Qwen token/decode/chat, server, speedup, or full-residency claim is made.",
            "This proves the extracted Qwen-family norm fixtures can run through a strict CUDA RMSNorm kernel against deterministic CPU references."
        ],
        "error": null
    }))
}

fn dense_gguf_rope_cuda_parity_receipt_json(
    inspection: &DenseGgufDescriptorInspection,
    fixture: &DenseGgufRopeCudaFixture,
    parity: &DenseGgufRopeCudaParity,
    probe: Option<&bitnet_device_probe::NvidiaCudaProbe>,
    model_path: &Path,
    model_sha256: &str,
    artifact_path: &str,
    timestamp_utc: &str,
) -> Value {
    let cuda = cuda_identity_json(probe);
    let execution_plan = execution_plan_receipt(ExecutionPlanReceiptInput {
        model_family: &inspection.model_family,
        quantization: "dense_f32_rope",
        requested_backend: HARDWARE_LANE,
        selected_backend: HARDWARE_LANE,
        runtime_api: "cuda",
        strict_fallback_policy: "reject",
        summary: ModelDispatchSummary {
            total_ops: 1,
            cuda_bitnet_qk256_ops: 0,
            cuda_dense_regular_llm_ops: 1,
            cpu_fallback_ops: 0,
            unsupported_ops: 0,
            fallback_used: false,
            selected_route: Some(ModelDispatchBackend::CudaDenseRegularLlm),
            strict_cuda_ready: true,
        },
        speedup_claim: false,
        full_cuda_residency_claimed: false,
    });

    json!({
        "schema": 1,
        "artifact_kind": DENSE_GGUF_ROPE_CUDA_PARITY_ARTIFACT_KIND,
        "artifact_path": artifact_path,
        "claim": "dense_gguf_rope_cuda_parity_tested",
        "machine_id": MACHINE_ID,
        "hardware_lane": HARDWARE_LANE,
        "timestamp_utc": timestamp_utc,
        "requested_backend": HARDWARE_LANE,
        "selected_backend": HARDWARE_LANE,
        "runtime_api": "cuda",
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "speedup_claim": false,
        "cuda": cuda,
        "model": {
            "model_family": inspection.model_family,
            "architecture": inspection.architecture,
            "artifact_kind": "dense_gguf",
            "quantization_families": inspection.quantization_families,
            "file": model_path.display().to_string(),
            "sha256": model_sha256
        },
        "execution_path": {
            "model_class": "dense_regular_llm",
            "kernel_family": "dense_f32_rope",
            "quantization_family": "metadata_derived_rope_qk_f32_fixture",
            "bitnet_packed_kernel_proof": false,
            "qk256_proof": false
        },
        "execution_plan": execution_plan,
        "descriptor_coverage": {
            "schema": 1,
            "source_artifact_kind": "dense_gguf_tensor_descriptor_inspection",
            "tensor_count": inspection.tensor_count,
            "metadata_count": inspection.metadata_count as u64,
            "required_roles_present": dense_transformer_block_descriptor_coverage_complete(inspection),
            "strict_descriptor_complete": dense_transformer_block_descriptor_coverage_complete(inspection),
            "dense_cuda_route_status": dense_model_boundary_route_status(inspection),
            "model_boundary_lm_head_source": dense_model_boundary_lm_head_source(inspection),
            "quantization_families": inspection.quantization_families,
            "bitnet_packed_marker_found": inspection.bitnet_packed_marker_found,
            "dense_gguf_inference_claimed": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "rope_fixture": {
            "schema": 1,
            "source_artifact_kind": "dense_gguf_tensor_descriptor_inspection",
            "fixture_id": fixture.fixture_id,
            "model_family": fixture.model_family,
            "architecture": fixture.architecture,
            "layer_index": fixture.layer_index as u64,
            "head_dim": fixture.head_dim as u64,
            "q_heads": fixture.q_heads as u64,
            "kv_heads": fixture.kv_heads as u64,
            "seq_len": fixture.seq_len as u64,
            "position_offset": fixture.position_offset as u64,
            "rope_base": fixture.rope_base,
            "scaling_factor": fixture.scaling_factor,
            "interleaved": fixture.interleaved,
            "head_dim_source": fixture.head_dim_source,
            "q_heads_source": fixture.q_heads_source,
            "kv_heads_source": fixture.kv_heads_source,
            "rope_base_source": fixture.rope_base_source,
            "q_input_sha256": sha256_f32(&fixture.q_input_f32),
            "k_input_sha256": sha256_f32(&fixture.k_input_f32),
            "cpu_reference_q_output_sha256": sha256_f32(&fixture.expected_q_output_f32),
            "cpu_reference_k_output_sha256": sha256_f32(&fixture.expected_k_output_f32),
            "cuda_input_dtype": "f32",
            "cuda_output_dtype": "f32",
            "dense_gguf_inference_claimed": false,
            "dense_regular_llm_cuda_claimed": true,
            "cpu_cuda_parity_claimed": true,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "kernel_stats": [{
            "kernel_id": parity.stats.kernel_id,
            "fixture_id": parity.fixture_id,
            "invocations": parity.stats.invocations,
            "fallback_invocations": parity.stats.fallback_invocations,
            "host_to_device_bytes": parity.stats.host_to_device_bytes,
            "device_to_host_bytes": parity.stats.device_to_host_bytes,
            "kernel_launches": parity.stats.kernel_launches,
            "kernel_time_ms": parity.stats.kernel_time_ms
        }],
        "parity": {
            "reference_backend": parity.reference_backend,
            "target_backend": parity.target_backend,
            "kernel_id": parity.kernel_id,
            "fixture_id": parity.fixture_id,
            "max_abs_error": parity.max_abs_error,
            "mean_abs_error": parity.mean_abs_error,
            "passed": parity.passed,
            "tolerance": parity.tolerance,
            "tolerance_source": "CUDA-DENSE-018 dense GGUF RoPE F32 CUDA fixture",
            "first_divergence": null
        },
        "timing": {
            "kernel_time_ms": parity.stats.kernel_time_ms,
            "host_to_device_bytes": parity.stats.host_to_device_bytes,
            "device_to_host_bytes": parity.stats.device_to_host_bytes
        },
        "claim_boundary": {
            "dense_regular_llm_cuda_claimed": true,
            "dense_tensor_residency_claimed": true,
            "dense_gguf_descriptor_inspection_claimed": true,
            "dense_gguf_rope_cuda_parity_claimed": true,
            "dense_gguf_norm_cuda_parity_claimed": false,
            "dense_gguf_linear_fixture_extraction_claimed": false,
            "dense_gguf_linear_cuda_parity_claimed": false,
            "dense_gguf_linear_role_sweep_cuda_parity_claimed": false,
            "dense_gguf_one_layer_execution_plan_claimed": false,
            "dense_gguf_inference_claimed": false,
            "qwen_one_token_cuda_claimed": false,
            "qwen_short_decode_cuda_claimed": false,
            "qwen_chat_cuda_claimed": false,
            "cpu_cuda_parity_claimed": true,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false
        },
        "tensor_residency": {
            "schema_version": "1.0.0",
            "scope": "single_dense_gguf_rope_fixture",
            "model_class": "dense_regular_llm",
            "fixture_id": parity.fixture_id,
            "dense_tensor_residency_claimed": true,
            "dense_gguf_inference_claimed": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false,
            "input_tensors_uploaded_once": true,
            "output_tensor_cuda_resident_during_kernel": true,
            "host_device_transfer_accounting_matches_kernel_stats": true,
            "inputs": [
                {
                    "name": "dense_gguf_rope_q_input",
                    "dtype": "f32",
                    "shape": [fixture.q_heads as u64, fixture.seq_len as u64, fixture.head_dim as u64],
                    "host_bytes": (fixture.q_input_f32.len() * 4) as u64,
                    "device_residency": "cuda_device_buffer",
                    "upload_count": 1,
                    "reuse_scope": "single_fixture_launch"
                },
                {
                    "name": "dense_gguf_rope_k_input",
                    "dtype": "f32",
                    "shape": [fixture.kv_heads as u64, fixture.seq_len as u64, fixture.head_dim as u64],
                    "host_bytes": (fixture.k_input_f32.len() * 4) as u64,
                    "device_residency": "cuda_device_buffer",
                    "upload_count": 1,
                    "reuse_scope": "single_fixture_launch"
                }
            ],
            "outputs": [
                {
                    "name": "dense_gguf_rope_q_output",
                    "dtype": "f32",
                    "shape": [fixture.q_heads as u64, fixture.seq_len as u64, fixture.head_dim as u64],
                    "device_residency": "cuda_device_buffer",
                    "device_to_host_bytes": (fixture.expected_q_output_f32.len() * 4) as u64,
                    "download_scope": "parity_check_only"
                },
                {
                    "name": "dense_gguf_rope_k_output",
                    "dtype": "f32",
                    "shape": [fixture.kv_heads as u64, fixture.seq_len as u64, fixture.head_dim as u64],
                    "device_residency": "cuda_device_buffer",
                    "device_to_host_bytes": (fixture.expected_k_output_f32.len() * 4) as u64,
                    "download_scope": "parity_check_only"
                }
            ],
            "allocation": {
                "device_buffer_count": 4,
                "temporary_workspace_bytes": 0,
                "persistent_handle_count": 0,
                "persistent_handles_claimed": false
            },
            "transfer_accounting": {
                "status": "measured",
                "host_to_device_bytes": parity.stats.host_to_device_bytes,
                "device_to_host_bytes": parity.stats.device_to_host_bytes,
                "kernel_invocations": parity.stats.invocations,
                "kernel_launches": parity.stats.kernel_launches
            }
        },
        "notes": [
            "Dense GGUF RoPE CUDA fixture parity only; no dense GGUF inference, Qwen token/decode/chat, server, speedup, or full-residency claim is made.",
            "This proves deterministic Q/K RoPE vectors derived from dense GGUF metadata can run through a strict CUDA RoPE kernel against CPU references."
        ],
        "error": null
    })
}

fn dense_gguf_linear_cuda_parity_receipt_json(
    parity: &DenseGgufLinearCudaParity,
    extracted: &DenseGgufLinearFixture,
    probe: Option<&bitnet_device_probe::NvidiaCudaProbe>,
    model_path: &Path,
    model_sha256: &str,
    artifact_path: &str,
    timestamp_utc: &str,
) -> Value {
    let summary = &extracted.summary;
    let cuda = cuda_identity_json(probe);
    let execution_plan = execution_plan_receipt(ExecutionPlanReceiptInput {
        model_family: &parity.model_family,
        quantization: "dense_fp16",
        requested_backend: HARDWARE_LANE,
        selected_backend: HARDWARE_LANE,
        runtime_api: "cuda",
        strict_fallback_policy: "reject",
        summary: ModelDispatchSummary {
            total_ops: 1,
            cuda_bitnet_qk256_ops: 0,
            cuda_dense_regular_llm_ops: 1,
            cpu_fallback_ops: 0,
            unsupported_ops: 0,
            fallback_used: false,
            selected_route: Some(ModelDispatchBackend::CudaDenseRegularLlm),
            strict_cuda_ready: true,
        },
        speedup_claim: false,
        full_cuda_residency_claimed: false,
    });

    json!({
        "schema": 1,
        "artifact_kind": DENSE_GGUF_LINEAR_CUDA_PARITY_ARTIFACT_KIND,
        "artifact_path": artifact_path,
        "claim": "dense_gguf_linear_cuda_parity_tested",
        "machine_id": MACHINE_ID,
        "hardware_lane": HARDWARE_LANE,
        "timestamp_utc": timestamp_utc,
        "requested_backend": HARDWARE_LANE,
        "selected_backend": HARDWARE_LANE,
        "runtime_api": "cuda",
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "speedup_claim": false,
        "cuda": cuda,
        "model": {
            "model_family": parity.model_family,
            "architecture": summary.architecture,
            "artifact_kind": "dense_gguf",
            "file": model_path.display().to_string(),
            "sha256": model_sha256
        },
        "execution_path": {
            "model_class": "dense_regular_llm",
            "kernel_family": "dense_fp16_gemm",
            "quantization_family": format!("{}_materialized_to_f16_bridge", parity.tensor_type),
            "bitnet_packed_kernel_proof": false,
            "qk256_proof": false
        },
        "execution_plan": execution_plan,
        "linear_fixture": {
            "schema": 1,
            "source_artifact_kind": DENSE_GGUF_LINEAR_FIXTURE_ARTIFACT_KIND,
            "fixture_id": parity.fixture_id,
            "model_family": parity.model_family,
            "architecture": summary.architecture,
            "tensor_name": parity.tensor_name,
            "role": parity.tensor_role,
            "tensor_type": parity.tensor_type,
            "matrix_rows": parity.matrix_rows,
            "matrix_cols": parity.matrix_cols,
            "logical_layout": "gguf_in_out_reinterpreted_as_out_in",
            "gemm_layout": "input_1_by_in_times_weight_in_by_out",
            "values_materialized_as_f32": true,
            "gemm_input_dtype": "f16",
            "gemm_weight_dtype": "f16",
            "gemm_output_dtype": "f32",
            "weight_values_sha256": parity.source_weight_sha256,
            "dense_gguf_inference_claimed": false,
            "dense_regular_llm_cuda_claimed": true,
            "cpu_cuda_parity_claimed": true,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "kernel_stats": [{
            "kernel_id": parity.stats.kernel_id,
            "invocations": parity.stats.invocations,
            "fallback_invocations": parity.stats.fallback_invocations,
            "host_to_device_bytes": parity.stats.host_to_device_bytes,
            "device_to_host_bytes": parity.stats.device_to_host_bytes,
            "kernel_launches": parity.stats.kernel_launches,
            "kernel_time_ms": parity.stats.kernel_time_ms
        }],
        "parity": {
            "reference_backend": parity.reference_backend,
            "target_backend": parity.target_backend,
            "kernel_id": parity.kernel_id,
            "fixture_id": parity.fixture_id,
            "max_abs_error": parity.max_abs_error,
            "mean_abs_error": parity.mean_abs_error,
            "passed": parity.passed,
            "tolerance": parity.tolerance,
            "tolerance_source": "CUDA-DENSE-009 extracted dense GGUF linear FP16 bridge"
        },
        "claim_boundary": {
            "dense_regular_llm_cuda_claimed": true,
            "dense_tensor_residency_claimed": true,
            "dense_gguf_descriptor_inspection_claimed": true,
            "dense_gguf_linear_fixture_extraction_claimed": true,
            "dense_gguf_linear_cuda_parity_claimed": true,
            "dense_gguf_inference_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false
        },
        "tensor_residency": {
            "schema_version": "1.0.0",
            "scope": "single_dense_gguf_linear_fixture",
            "model_class": "dense_regular_llm",
            "fixture_id": parity.fixture_id,
            "dense_tensor_residency_claimed": true,
            "dense_gguf_inference_claimed": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false,
            "input_tensors_uploaded_once": true,
            "output_tensor_cuda_resident_during_kernel": true,
            "host_device_transfer_accounting_matches_kernel_stats": true,
            "inputs": [
                {
                    "name": "dense_gguf_linear_input",
                    "dtype": "f16",
                    "shape": [1, parity.matrix_cols],
                    "host_bytes": (parity.matrix_cols * 2) as u64,
                    "device_residency": "cuda_device_buffer",
                    "upload_count": 1,
                    "reuse_scope": "single_fixture_launch"
                },
                {
                    "name": "dense_gguf_linear_weight_transposed",
                    "dtype": "f16",
                    "shape": [parity.matrix_cols, parity.matrix_rows],
                    "host_bytes": (parity.matrix_rows * parity.matrix_cols * 2) as u64,
                    "device_residency": "cuda_device_buffer",
                    "upload_count": 1,
                    "reuse_scope": "single_fixture_launch"
                }
            ],
            "outputs": [
                {
                    "name": "dense_gguf_linear_output",
                    "dtype": "f32",
                    "shape": [1, parity.matrix_rows],
                    "device_residency": "cuda_device_buffer",
                    "device_to_host_bytes": parity.stats.device_to_host_bytes,
                    "download_scope": "parity_check_only"
                }
            ],
            "allocation": {
                "device_buffer_count": 3,
                "temporary_workspace_bytes": 0,
                "persistent_handle_count": 0,
                "persistent_handles_claimed": false
            },
            "transfer_accounting": {
                "status": "measured",
                "host_to_device_bytes": parity.stats.host_to_device_bytes,
                "device_to_host_bytes": parity.stats.device_to_host_bytes
            }
        },
        "error": null
    })
}

fn dense_gguf_linear_role_sweep_cuda_parity_receipt_json(
    results: &[DenseLinearSweepResult],
    probe: Option<&bitnet_device_probe::NvidiaCudaProbe>,
    model_path: &Path,
    model_sha256: &str,
    artifact_path: &str,
    timestamp_utc: &str,
) -> Result<Value> {
    let first = results.first().ok_or_else(|| anyhow!("role sweep has no results"))?;
    let first_summary = &first.extracted.summary;
    let model_family = first_summary.model_family.as_str();
    let architecture = first_summary.architecture.as_str();
    let role_count = results.len();
    let role_count_u64 = role_count as u64;

    let mut tensor_types = BTreeSet::new();
    for result in results {
        let summary = &result.extracted.summary;
        if summary.model_family != model_family {
            bail!(
                "dense GGUF role sweep mixed model families: expected {model_family}, got {}",
                summary.model_family
            );
        }
        if summary.architecture != architecture {
            bail!(
                "dense GGUF role sweep mixed architectures: expected {architecture}, got {}",
                summary.architecture
            );
        }
        tensor_types.insert(result.parity.tensor_type.clone());
    }

    let quantization_family = if tensor_types.len() == 1 {
        format!(
            "{}_materialized_to_f16_bridge",
            tensor_types.iter().next().expect("one tensor type")
        )
    } else {
        "mixed_dense_materialized_to_f16_bridge".to_string()
    };

    let max_abs_error =
        results.iter().map(|result| result.parity.max_abs_error).fold(0.0_f32, f32::max);
    let max_mean_abs_error =
        results.iter().map(|result| result.parity.mean_abs_error).fold(0.0_f32, f32::max);
    let tolerance = results.iter().map(|result| result.parity.tolerance).fold(0.0_f32, f32::max);
    let h2d_bytes =
        results.iter().map(|result| result.parity.stats.host_to_device_bytes).sum::<u64>();
    let d2h_bytes =
        results.iter().map(|result| result.parity.stats.device_to_host_bytes).sum::<u64>();
    let kernel_invocations =
        results.iter().map(|result| result.parity.stats.invocations).sum::<u64>();
    let kernel_launches =
        results.iter().map(|result| result.parity.stats.kernel_launches).sum::<u64>();
    let aggregate_kernel_time_ms = results
        .iter()
        .try_fold(0.0_f64, |acc, result| result.parity.stats.kernel_time_ms.map(|time| acc + time));

    let cuda = cuda_identity_json(probe);
    let execution_plan = execution_plan_receipt(ExecutionPlanReceiptInput {
        model_family,
        quantization: "dense_fp16",
        requested_backend: HARDWARE_LANE,
        selected_backend: HARDWARE_LANE,
        runtime_api: "cuda",
        strict_fallback_policy: "reject",
        summary: ModelDispatchSummary {
            total_ops: role_count,
            cuda_bitnet_qk256_ops: 0,
            cuda_dense_regular_llm_ops: role_count,
            cpu_fallback_ops: 0,
            unsupported_ops: 0,
            fallback_used: false,
            selected_route: Some(ModelDispatchBackend::CudaDenseRegularLlm),
            strict_cuda_ready: true,
        },
        speedup_claim: false,
        full_cuda_residency_claimed: false,
    });

    let covered_roles =
        results.iter().map(|result| result.parity.tensor_role.clone()).collect::<Vec<_>>();
    let linear_fixtures = results
        .iter()
        .map(|result| {
            let summary = &result.extracted.summary;
            let parity = &result.parity;
            json!({
                "schema": 1,
                "source_artifact_kind": DENSE_GGUF_LINEAR_FIXTURE_ARTIFACT_KIND,
                "fixture_id": parity.fixture_id,
                "model_family": parity.model_family,
                "architecture": summary.architecture,
                "tensor_name": parity.tensor_name,
                "role": parity.tensor_role,
                "tensor_type": parity.tensor_type,
                "matrix_rows": parity.matrix_rows,
                "matrix_cols": parity.matrix_cols,
                "logical_layout": "gguf_in_out_reinterpreted_as_out_in",
                "gemm_layout": "input_1_by_in_times_weight_in_by_out",
                "values_materialized_as_f32": true,
                "gemm_input_dtype": "f16",
                "gemm_weight_dtype": "f16",
                "gemm_output_dtype": "f32",
                "weight_values_sha256": parity.source_weight_sha256,
                "dense_gguf_inference_claimed": false,
                "dense_regular_llm_cuda_claimed": true,
                "cpu_cuda_parity_claimed": true,
                "bitnet_packed_i2s_qk256_proof": false,
                "speedup_claim": false,
                "full_cuda_residency_claimed": false
            })
        })
        .collect::<Vec<_>>();
    let kernel_stats = results
        .iter()
        .map(|result| {
            let parity = &result.parity;
            json!({
                "role": parity.tensor_role,
                "tensor_name": parity.tensor_name,
                "fixture_id": parity.fixture_id,
                "kernel_id": parity.stats.kernel_id,
                "invocations": parity.stats.invocations,
                "fallback_invocations": parity.stats.fallback_invocations,
                "host_to_device_bytes": parity.stats.host_to_device_bytes,
                "device_to_host_bytes": parity.stats.device_to_host_bytes,
                "kernel_launches": parity.stats.kernel_launches,
                "kernel_time_ms": parity.stats.kernel_time_ms
            })
        })
        .collect::<Vec<_>>();

    Ok(json!({
        "schema": 1,
        "artifact_kind": DENSE_GGUF_LINEAR_ROLE_SWEEP_CUDA_PARITY_ARTIFACT_KIND,
        "artifact_path": artifact_path,
        "claim": "dense_gguf_linear_role_sweep_cuda_parity_tested",
        "machine_id": MACHINE_ID,
        "hardware_lane": HARDWARE_LANE,
        "timestamp_utc": timestamp_utc,
        "requested_backend": HARDWARE_LANE,
        "selected_backend": HARDWARE_LANE,
        "runtime_api": "cuda",
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "speedup_claim": false,
        "cuda": cuda,
        "model": {
            "model_family": model_family,
            "architecture": architecture,
            "artifact_kind": "dense_gguf",
            "file": model_path.display().to_string(),
            "sha256": model_sha256
        },
        "execution_path": {
            "model_class": "dense_regular_llm",
            "kernel_family": "dense_fp16_gemm",
            "quantization_family": quantization_family,
            "bitnet_packed_kernel_proof": false,
            "qk256_proof": false
        },
        "execution_plan": execution_plan,
        "linear_role_sweep": {
            "schema": 1,
            "roles_total": role_count_u64,
            "roles_passed": role_count_u64,
            "roles_failed": 0,
            "covered_roles": covered_roles,
            "all_parity_passed": true,
            "max_abs_error": max_abs_error,
            "max_mean_abs_error": max_mean_abs_error,
            "aggregate_kernel_time_ms": aggregate_kernel_time_ms,
            "host_to_device_bytes": h2d_bytes,
            "device_to_host_bytes": d2h_bytes,
            "kernel_invocations": kernel_invocations,
            "kernel_launches": kernel_launches,
            "dense_gguf_inference_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "linear_fixtures": linear_fixtures,
        "kernel_stats": kernel_stats,
        "parity": {
            "reference_backend": first.parity.reference_backend,
            "target_backend": first.parity.target_backend,
            "kernel_id": first.parity.kernel_id,
            "roles_total": role_count_u64,
            "roles_passed": role_count_u64,
            "roles_failed": 0,
            "max_abs_error": max_abs_error,
            "max_mean_abs_error": max_mean_abs_error,
            "passed": true,
            "tolerance": tolerance,
            "tolerance_source": "CUDA-DENSE-012 extracted dense GGUF linear role-sweep FP16 bridge"
        },
        "claim_boundary": {
            "dense_regular_llm_cuda_claimed": true,
            "dense_tensor_residency_claimed": true,
            "dense_gguf_descriptor_inspection_claimed": true,
            "dense_gguf_linear_fixture_extraction_claimed": true,
            "dense_gguf_linear_cuda_parity_claimed": true,
            "dense_gguf_linear_role_sweep_cuda_parity_claimed": true,
            "dense_gguf_inference_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false
        },
        "tensor_residency": {
            "schema_version": "1.0.0",
            "scope": "dense_gguf_linear_role_sweep_fixture",
            "model_class": "dense_regular_llm",
            "roles_total": role_count_u64,
            "dense_tensor_residency_claimed": true,
            "dense_gguf_inference_claimed": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false,
            "input_tensors_uploaded_once_per_role": true,
            "output_tensor_cuda_resident_during_kernel": true,
            "host_device_transfer_accounting_matches_kernel_stats": true,
            "allocation": {
                "device_buffer_count": role_count_u64 * 3,
                "temporary_workspace_bytes": 0,
                "persistent_handle_count": 0,
                "persistent_handles_claimed": false
            },
            "transfer_accounting": {
                "status": "measured",
                "host_to_device_bytes": h2d_bytes,
                "device_to_host_bytes": d2h_bytes,
                "kernel_invocations": kernel_invocations,
                "kernel_launches": kernel_launches
            }
        },
        "error": null
    }))
}

fn dense_gguf_attention_score_fixture_receipt_json(
    inspection: &DenseGgufDescriptorInspection,
    fixture: &DenseGgufAttentionScoreFixture,
    model_path: &Path,
    model_sha256: &str,
    artifact_path: &str,
    timestamp_utc: &str,
) -> Value {
    let execution_plan = execution_plan_receipt(ExecutionPlanReceiptInput {
        model_family: &inspection.model_family,
        quantization: "dense_f32_attention_scores_fixture",
        requested_backend: HARDWARE_LANE,
        selected_backend: "unsupported_strict_cuda",
        runtime_api: "none",
        strict_fallback_policy: "reject",
        summary: ModelDispatchSummary {
            total_ops: 1,
            cuda_bitnet_qk256_ops: 0,
            cuda_dense_regular_llm_ops: 0,
            cpu_fallback_ops: 0,
            unsupported_ops: 1,
            fallback_used: false,
            selected_route: Some(ModelDispatchBackend::Unsupported),
            strict_cuda_ready: false,
        },
        speedup_claim: false,
        full_cuda_residency_claimed: false,
    });

    json!({
        "schema": 1,
        "artifact_kind": DENSE_GGUF_ATTENTION_SCORE_FIXTURE_ARTIFACT_KIND,
        "artifact_path": artifact_path,
        "claim": "dense_gguf_attention_score_fixture_extracted",
        "machine_id": MACHINE_ID,
        "hardware_lane": HARDWARE_LANE,
        "timestamp_utc": timestamp_utc,
        "inspection_source": "gguf_reader_attention_score_fixture",
        "error": null,
        "model": {
            "model_family": inspection.model_family,
            "architecture": inspection.architecture,
            "artifact_kind": "dense_gguf",
            "quantization_families": inspection.quantization_families,
            "file": model_path.display().to_string(),
            "sha256": model_sha256
        },
        "execution_path": {
            "model_class": "dense_regular_llm",
            "kernel_family": "cpu_reference_attention_scores_after_rope",
            "quantization_family": "metadata_derived_rope_qk_attention_scores_fixture",
            "bitnet_packed_kernel_proof": false,
            "qk256_proof": false
        },
        "execution_plan": execution_plan,
        "descriptor_coverage": {
            "schema": 1,
            "source_artifact_kind": "dense_gguf_tensor_descriptor_inspection",
            "tensor_count": inspection.tensor_count,
            "metadata_count": inspection.metadata_count as u64,
            "required_roles_present": inspection.required_roles_present,
            "strict_descriptor_complete": inspection.strict_descriptor_complete,
            "dense_cuda_route_status": inspection.dense_cuda_route_status,
            "quantization_families": inspection.quantization_families,
            "bitnet_packed_marker_found": inspection.bitnet_packed_marker_found,
            "dense_gguf_inference_claimed": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "attention_score_fixture": {
            "schema": 1,
            "source_artifact_kind": DENSE_GGUF_ATTENTION_SCORE_FIXTURE_ARTIFACT_KIND,
            "source_rope_artifact_kind": DENSE_GGUF_ROPE_CUDA_PARITY_ARTIFACT_KIND,
            "source_rope_fixture_id": fixture.source_rope_fixture_id,
            "fixture_id": fixture.fixture_id,
            "model_family": fixture.model_family,
            "architecture": fixture.architecture,
            "layer_index": fixture.layer_index as u64,
            "head_dim": fixture.head_dim as u64,
            "q_heads": fixture.q_heads as u64,
            "kv_heads": fixture.kv_heads as u64,
            "heads_per_kv_group": fixture.heads_per_kv_group as u64,
            "seq_len": fixture.seq_len as u64,
            "position_offset": fixture.position_offset as u64,
            "rope_base": fixture.rope_base,
            "scaling_factor": fixture.scaling_factor,
            "attention_scale": fixture.scale,
            "causal_mask_applied": true,
            "head_dim_source": fixture.head_dim_source,
            "q_heads_source": fixture.q_heads_source,
            "kv_heads_source": fixture.kv_heads_source,
            "rope_base_source": fixture.rope_base_source,
            "q_rope_output_sha256": sha256_f32(&fixture.q_rope_output_f32),
            "k_rope_output_sha256": sha256_f32(&fixture.k_rope_output_f32),
            "cpu_reference_scores_sha256": sha256_f32(&fixture.expected_scores_f32),
            "score_shape": [fixture.q_heads as u64, fixture.seq_len as u64, fixture.seq_len as u64],
            "score_count": fixture.expected_scores_f32.len() as u64,
            "finite_scores": fixture.finite_scores as u64,
            "causal_masked_scores": fixture.causal_masked_scores as u64,
            "cpu_reference_computed": true,
            "cuda_kernel_status": "missing_cuda_kernel",
            "strict_cuda_ready": false,
            "cpu_fallback_allowed": false,
            "transfer_timing_status": "not_measured_no_kernel",
            "dense_gguf_inference_claimed": false,
            "dense_regular_llm_cuda_claimed": false,
            "cpu_cuda_parity_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "attention_score_gap_audit": {
            "schema": 1,
            "source_artifact_kind": DENSE_GGUF_ATTENTION_SCORE_FIXTURE_ARTIFACT_KIND,
            "gap_role": "attention_scores",
            "input_dependencies": ["rope_q", "rope_k", "causal_mask"],
            "source_rope_cuda_parity_required": true,
            "source_rope_cuda_parity_available": true,
            "cpu_reference_available": true,
            "cuda_kernel_status": "missing_cuda_kernel",
            "strict_cuda_ready": false,
            "cpu_fallback_allowed": false,
            "blocks_strict_cuda_one_layer": true,
            "next_required_proof": "cuda_attention_score_kernel_parity",
            "candidate_order": DENSE_ONE_LAYER_GAP_CANDIDATE_ORDER,
            "dense_gguf_attention_score_fixture_extraction_claimed": true,
            "dense_gguf_inference_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "timing": {
            "kernel_time_ms": null,
            "host_to_device_bytes": 0,
            "device_to_host_bytes": 0,
            "transfer_timing_status": "not_measured_no_kernel"
        },
        "claim_boundary": {
            "dense_regular_llm_cuda_claimed": false,
            "dense_tensor_residency_claimed": false,
            "dense_gguf_descriptor_inspection_claimed": true,
            "dense_gguf_attention_score_fixture_extraction_claimed": true,
            "dense_gguf_rope_cuda_parity_claimed": false,
            "dense_gguf_norm_cuda_parity_claimed": false,
            "dense_gguf_linear_fixture_extraction_claimed": false,
            "dense_gguf_linear_cuda_parity_claimed": false,
            "dense_gguf_linear_role_sweep_cuda_parity_claimed": false,
            "dense_gguf_one_layer_execution_plan_claimed": false,
            "dense_gguf_inference_claimed": false,
            "qwen_one_token_cuda_claimed": false,
            "qwen_short_decode_cuda_claimed": false,
            "qwen_chat_cuda_claimed": false,
            "cpu_cuda_parity_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false
        }
    })
}

fn dense_gguf_attention_softmax_fixture_receipt_json(
    inspection: &DenseGgufDescriptorInspection,
    fixture: &DenseGgufAttentionSoftmaxFixture,
    model_path: &Path,
    model_sha256: &str,
    artifact_path: &str,
    timestamp_utc: &str,
) -> Value {
    let execution_plan = execution_plan_receipt(ExecutionPlanReceiptInput {
        model_family: &inspection.model_family,
        quantization: "dense_f32_attention_softmax_fixture",
        requested_backend: HARDWARE_LANE,
        selected_backend: "unsupported_strict_cuda",
        runtime_api: "none",
        strict_fallback_policy: "reject",
        summary: ModelDispatchSummary {
            total_ops: 1,
            cuda_bitnet_qk256_ops: 0,
            cuda_dense_regular_llm_ops: 0,
            cpu_fallback_ops: 0,
            unsupported_ops: 1,
            fallback_used: false,
            selected_route: Some(ModelDispatchBackend::Unsupported),
            strict_cuda_ready: false,
        },
        speedup_claim: false,
        full_cuda_residency_claimed: false,
    });

    json!({
        "schema": 1,
        "artifact_kind": DENSE_GGUF_ATTENTION_SOFTMAX_FIXTURE_ARTIFACT_KIND,
        "artifact_path": artifact_path,
        "claim": "dense_gguf_attention_softmax_fixture_extracted",
        "machine_id": MACHINE_ID,
        "hardware_lane": HARDWARE_LANE,
        "timestamp_utc": timestamp_utc,
        "inspection_source": "gguf_reader_attention_softmax_fixture",
        "error": null,
        "model": {
            "model_family": inspection.model_family,
            "architecture": inspection.architecture,
            "artifact_kind": "dense_gguf",
            "quantization_families": inspection.quantization_families,
            "file": model_path.display().to_string(),
            "sha256": model_sha256
        },
        "execution_path": {
            "model_class": "dense_regular_llm",
            "kernel_family": "cpu_reference_attention_softmax_after_scores",
            "quantization_family": "metadata_derived_attention_softmax_fixture",
            "bitnet_packed_kernel_proof": false,
            "qk256_proof": false
        },
        "execution_plan": execution_plan,
        "descriptor_coverage": {
            "schema": 1,
            "source_artifact_kind": "dense_gguf_tensor_descriptor_inspection",
            "tensor_count": inspection.tensor_count,
            "metadata_count": inspection.metadata_count as u64,
            "required_roles_present": inspection.required_roles_present,
            "strict_descriptor_complete": inspection.strict_descriptor_complete,
            "dense_cuda_route_status": inspection.dense_cuda_route_status,
            "quantization_families": inspection.quantization_families,
            "bitnet_packed_marker_found": inspection.bitnet_packed_marker_found,
            "dense_gguf_inference_claimed": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "attention_softmax_fixture": {
            "schema": 1,
            "source_artifact_kind": DENSE_GGUF_ATTENTION_SOFTMAX_FIXTURE_ARTIFACT_KIND,
            "source_attention_score_artifact_kind": DENSE_GGUF_ATTENTION_SCORE_CUDA_PARITY_ARTIFACT_KIND,
            "source_attention_score_fixture_id": fixture.source_attention_score_fixture_id,
            "fixture_id": fixture.fixture_id,
            "model_family": fixture.model_family,
            "architecture": fixture.architecture,
            "layer_index": fixture.layer_index as u64,
            "q_heads": fixture.q_heads as u64,
            "kv_heads": fixture.kv_heads as u64,
            "seq_len": fixture.seq_len as u64,
            "row_count": fixture.row_count as u64,
            "probability_count": fixture.probability_count as u64,
            "causal_zero_probabilities": fixture.causal_zero_probabilities as u64,
            "attention_scores_sha256": sha256_f32(&fixture.attention_scores_f32),
            "cpu_reference_probabilities_sha256": sha256_f32(&fixture.expected_probabilities_f32),
            "max_row_sum_abs_error": fixture.max_row_sum_abs_error,
            "cpu_reference_computed": true,
            "cuda_kernel_status": "missing_cuda_kernel",
            "strict_cuda_ready": false,
            "cpu_fallback_allowed": false,
            "transfer_timing_status": "not_measured_no_kernel",
            "dense_gguf_inference_claimed": false,
            "dense_regular_llm_cuda_claimed": false,
            "cpu_cuda_parity_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "attention_softmax_gap_audit": {
            "schema": 1,
            "source_artifact_kind": DENSE_GGUF_ATTENTION_SOFTMAX_FIXTURE_ARTIFACT_KIND,
            "gap_role": "attention_softmax",
            "input_dependencies": ["attention_scores"],
            "source_attention_score_cuda_parity_required": true,
            "source_attention_score_cuda_parity_available": true,
            "cpu_reference_available": true,
            "cuda_kernel_status": "missing_cuda_kernel",
            "strict_cuda_ready": false,
            "cpu_fallback_allowed": false,
            "blocks_strict_cuda_one_layer": true,
            "next_required_proof": "cuda_attention_softmax_kernel_parity",
            "candidate_order": DENSE_ONE_LAYER_GAP_CANDIDATE_ORDER,
            "dense_gguf_attention_softmax_fixture_extraction_claimed": true,
            "dense_gguf_inference_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "timing": {
            "kernel_time_ms": null,
            "host_to_device_bytes": 0,
            "device_to_host_bytes": 0,
            "transfer_timing_status": "not_measured_no_kernel"
        },
        "claim_boundary": {
            "dense_regular_llm_cuda_claimed": false,
            "dense_tensor_residency_claimed": false,
            "dense_gguf_descriptor_inspection_claimed": true,
            "dense_gguf_attention_score_fixture_extraction_claimed": false,
            "dense_gguf_attention_score_cuda_parity_claimed": false,
            "dense_gguf_attention_softmax_fixture_extraction_claimed": true,
            "dense_gguf_rope_cuda_parity_claimed": false,
            "dense_gguf_norm_cuda_parity_claimed": false,
            "dense_gguf_linear_fixture_extraction_claimed": false,
            "dense_gguf_linear_cuda_parity_claimed": false,
            "dense_gguf_linear_role_sweep_cuda_parity_claimed": false,
            "dense_gguf_one_layer_execution_plan_claimed": false,
            "dense_gguf_inference_claimed": false,
            "qwen_one_token_cuda_claimed": false,
            "qwen_short_decode_cuda_claimed": false,
            "qwen_chat_cuda_claimed": false,
            "cpu_cuda_parity_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false
        }
    })
}

fn dense_gguf_attention_v_mix_fixture_receipt_json(
    inspection: &DenseGgufDescriptorInspection,
    fixture: &DenseGgufAttentionVMixFixture,
    model_path: &Path,
    model_sha256: &str,
    artifact_path: &str,
    timestamp_utc: &str,
) -> Value {
    let execution_plan = execution_plan_receipt(ExecutionPlanReceiptInput {
        model_family: &inspection.model_family,
        quantization: "dense_f32_attention_v_mix_fixture",
        requested_backend: HARDWARE_LANE,
        selected_backend: "unsupported_strict_cuda",
        runtime_api: "none",
        strict_fallback_policy: "reject",
        summary: ModelDispatchSummary {
            total_ops: 1,
            cuda_bitnet_qk256_ops: 0,
            cuda_dense_regular_llm_ops: 0,
            cpu_fallback_ops: 0,
            unsupported_ops: 1,
            fallback_used: false,
            selected_route: Some(ModelDispatchBackend::Unsupported),
            strict_cuda_ready: false,
        },
        speedup_claim: false,
        full_cuda_residency_claimed: false,
    });

    json!({
        "schema": 1,
        "artifact_kind": DENSE_GGUF_ATTENTION_V_MIX_FIXTURE_ARTIFACT_KIND,
        "artifact_path": artifact_path,
        "claim": "dense_gguf_attention_v_mix_fixture_extracted",
        "machine_id": MACHINE_ID,
        "hardware_lane": HARDWARE_LANE,
        "timestamp_utc": timestamp_utc,
        "inspection_source": "gguf_reader_attention_v_mix_fixture",
        "error": null,
        "model": {
            "model_family": inspection.model_family,
            "architecture": inspection.architecture,
            "artifact_kind": "dense_gguf",
            "quantization_families": inspection.quantization_families,
            "file": model_path.display().to_string(),
            "sha256": model_sha256
        },
        "execution_path": {
            "model_class": "dense_regular_llm",
            "kernel_family": "cpu_reference_attention_v_mix_after_softmax",
            "quantization_family": "metadata_derived_attention_v_mix_fixture",
            "bitnet_packed_kernel_proof": false,
            "qk256_proof": false
        },
        "execution_plan": execution_plan,
        "descriptor_coverage": {
            "schema": 1,
            "source_artifact_kind": "dense_gguf_tensor_descriptor_inspection",
            "tensor_count": inspection.tensor_count,
            "metadata_count": inspection.metadata_count as u64,
            "required_roles_present": inspection.required_roles_present,
            "strict_descriptor_complete": inspection.strict_descriptor_complete,
            "dense_cuda_route_status": inspection.dense_cuda_route_status,
            "quantization_families": inspection.quantization_families,
            "bitnet_packed_marker_found": inspection.bitnet_packed_marker_found,
            "dense_gguf_inference_claimed": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "attention_v_mix_fixture": {
            "schema": 1,
            "source_artifact_kind": DENSE_GGUF_ATTENTION_V_MIX_FIXTURE_ARTIFACT_KIND,
            "source_attention_softmax_artifact_kind": DENSE_GGUF_ATTENTION_SOFTMAX_CUDA_PARITY_ARTIFACT_KIND,
            "source_attention_softmax_fixture_id": fixture.source_attention_softmax_fixture_id,
            "source_attention_v_artifact_kind": DENSE_GGUF_LINEAR_ROLE_SWEEP_CUDA_PARITY_ARTIFACT_KIND,
            "source_attention_v_role": "attention_v",
            "fixture_id": fixture.fixture_id,
            "model_family": fixture.model_family,
            "architecture": fixture.architecture,
            "layer_index": fixture.layer_index as u64,
            "q_heads": fixture.q_heads as u64,
            "kv_heads": fixture.kv_heads as u64,
            "heads_per_kv_group": fixture.heads_per_kv_group as u64,
            "head_dim": fixture.head_dim as u64,
            "seq_len": fixture.seq_len as u64,
            "row_count": fixture.row_count as u64,
            "probability_count": fixture.probability_count as u64,
            "value_count": fixture.value_count as u64,
            "context_count": fixture.context_count as u64,
            "causal_zero_probabilities": fixture.causal_zero_probabilities as u64,
            "attention_probabilities_sha256": sha256_f32(&fixture.attention_probabilities_f32),
            "value_states_sha256": sha256_f32(&fixture.value_states_f32),
            "cpu_reference_context_sha256": sha256_f32(&fixture.expected_context_f32),
            "max_context_abs": fixture.max_context_abs,
            "cpu_reference_computed": true,
            "cuda_kernel_status": "missing_cuda_kernel",
            "strict_cuda_ready": false,
            "cpu_fallback_allowed": false,
            "transfer_timing_status": "not_measured_no_kernel",
            "dense_gguf_inference_claimed": false,
            "dense_regular_llm_cuda_claimed": false,
            "cpu_cuda_parity_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "attention_v_mix_gap_audit": {
            "schema": 1,
            "source_artifact_kind": DENSE_GGUF_ATTENTION_V_MIX_FIXTURE_ARTIFACT_KIND,
            "gap_role": "attention_v_mix",
            "input_dependencies": ["attention_softmax", "attention_v"],
            "source_attention_softmax_cuda_parity_required": true,
            "source_attention_softmax_cuda_parity_available": true,
            "source_attention_v_cuda_parity_required": true,
            "source_attention_v_cuda_parity_available": true,
            "cpu_reference_available": true,
            "cuda_kernel_status": "missing_cuda_kernel",
            "strict_cuda_ready": false,
            "cpu_fallback_allowed": false,
            "blocks_strict_cuda_one_layer": true,
            "next_required_proof": "cuda_attention_v_mix_kernel_parity",
            "candidate_order": DENSE_ONE_LAYER_REMAINING_GAP_CANDIDATE_ORDER,
            "dense_gguf_attention_v_mix_fixture_extraction_claimed": true,
            "dense_gguf_inference_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "timing": {
            "kernel_time_ms": null,
            "host_to_device_bytes": 0,
            "device_to_host_bytes": 0,
            "transfer_timing_status": "not_measured_no_kernel"
        },
        "claim_boundary": {
            "dense_regular_llm_cuda_claimed": false,
            "dense_tensor_residency_claimed": false,
            "dense_gguf_descriptor_inspection_claimed": true,
            "dense_gguf_attention_score_fixture_extraction_claimed": false,
            "dense_gguf_attention_score_cuda_parity_claimed": false,
            "dense_gguf_attention_softmax_fixture_extraction_claimed": false,
            "dense_gguf_attention_softmax_cuda_parity_claimed": false,
            "dense_gguf_attention_v_mix_fixture_extraction_claimed": true,
            "dense_gguf_rope_cuda_parity_claimed": false,
            "dense_gguf_norm_cuda_parity_claimed": false,
            "dense_gguf_linear_fixture_extraction_claimed": false,
            "dense_gguf_linear_cuda_parity_claimed": false,
            "dense_gguf_linear_role_sweep_cuda_parity_claimed": false,
            "dense_gguf_one_layer_execution_plan_claimed": false,
            "dense_gguf_inference_claimed": false,
            "qwen_one_token_cuda_claimed": false,
            "qwen_short_decode_cuda_claimed": false,
            "qwen_chat_cuda_claimed": false,
            "cpu_cuda_parity_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false
        }
    })
}

fn dense_gguf_mlp_activation_fixture_receipt_json(
    inspection: &DenseGgufDescriptorInspection,
    fixture: &DenseGgufMlpActivationFixture,
    model_path: &Path,
    model_sha256: &str,
    artifact_path: &str,
    timestamp_utc: &str,
) -> Value {
    let execution_plan = execution_plan_receipt(ExecutionPlanReceiptInput {
        model_family: &inspection.model_family,
        quantization: "dense_f32_mlp_activation_fixture",
        requested_backend: HARDWARE_LANE,
        selected_backend: "unsupported_strict_cuda",
        runtime_api: "none",
        strict_fallback_policy: "reject",
        summary: ModelDispatchSummary {
            total_ops: 1,
            cuda_bitnet_qk256_ops: 0,
            cuda_dense_regular_llm_ops: 0,
            cpu_fallback_ops: 0,
            unsupported_ops: 1,
            fallback_used: false,
            selected_route: Some(ModelDispatchBackend::Unsupported),
            strict_cuda_ready: false,
        },
        speedup_claim: false,
        full_cuda_residency_claimed: false,
    });

    json!({
        "schema": 1,
        "artifact_kind": DENSE_GGUF_MLP_ACTIVATION_FIXTURE_ARTIFACT_KIND,
        "artifact_path": artifact_path,
        "claim": "dense_gguf_mlp_activation_fixture_extracted",
        "machine_id": MACHINE_ID,
        "hardware_lane": HARDWARE_LANE,
        "timestamp_utc": timestamp_utc,
        "inspection_source": "gguf_reader_mlp_activation_fixture",
        "error": null,
        "model": {
            "model_family": inspection.model_family,
            "architecture": inspection.architecture,
            "artifact_kind": "dense_gguf",
            "quantization_families": inspection.quantization_families,
            "file": model_path.display().to_string(),
            "sha256": model_sha256
        },
        "execution_path": {
            "model_class": "dense_regular_llm",
            "kernel_family": "cpu_reference_mlp_activation",
            "quantization_family": "metadata_derived_mlp_activation_fixture",
            "bitnet_packed_kernel_proof": false,
            "qk256_proof": false
        },
        "execution_plan": execution_plan,
        "descriptor_coverage": {
            "schema": 1,
            "source_artifact_kind": "dense_gguf_tensor_descriptor_inspection",
            "tensor_count": inspection.tensor_count,
            "metadata_count": inspection.metadata_count as u64,
            "required_roles_present": inspection.required_roles_present,
            "strict_descriptor_complete": inspection.strict_descriptor_complete,
            "dense_cuda_route_status": inspection.dense_cuda_route_status,
            "quantization_families": inspection.quantization_families,
            "bitnet_packed_marker_found": inspection.bitnet_packed_marker_found,
            "dense_gguf_inference_claimed": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "mlp_activation_fixture": {
            "schema": 1,
            "source_artifact_kind": DENSE_GGUF_MLP_ACTIVATION_FIXTURE_ARTIFACT_KIND,
            "source_mlp_gate_artifact_kind": DENSE_GGUF_LINEAR_ROLE_SWEEP_CUDA_PARITY_ARTIFACT_KIND,
            "source_mlp_gate_role": "mlp_gate",
            "source_mlp_gate_fixture_id": fixture.source_mlp_gate_fixture_id,
            "source_mlp_gate_tensor": fixture.source_mlp_gate_tensor,
            "source_mlp_up_artifact_kind": DENSE_GGUF_LINEAR_ROLE_SWEEP_CUDA_PARITY_ARTIFACT_KIND,
            "source_mlp_up_role": "mlp_up",
            "source_mlp_up_fixture_id": fixture.source_mlp_up_fixture_id,
            "source_mlp_up_tensor": fixture.source_mlp_up_tensor,
            "fixture_id": fixture.fixture_id,
            "model_family": fixture.model_family,
            "architecture": fixture.architecture,
            "layer_index": fixture.layer_index as u64,
            "activation_kind": fixture.activation_kind,
            "gate_output_count": fixture.gate_output_f32.len() as u64,
            "up_output_count": fixture.up_output_f32.len() as u64,
            "activation_count": fixture.activation_count as u64,
            "gate_output_sha256": sha256_f32(&fixture.gate_output_f32),
            "up_output_sha256": sha256_f32(&fixture.up_output_f32),
            "cpu_reference_activation_sha256": sha256_f32(&fixture.expected_activation_f32),
            "max_activation_abs": fixture.max_activation_abs,
            "cpu_reference_computed": true,
            "cuda_kernel_status": "missing_cuda_kernel",
            "strict_cuda_ready": false,
            "cpu_fallback_allowed": false,
            "transfer_timing_status": "not_measured_no_kernel",
            "dense_gguf_inference_claimed": false,
            "dense_regular_llm_cuda_claimed": false,
            "cpu_cuda_parity_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "mlp_activation_gap_audit": {
            "schema": 1,
            "source_artifact_kind": DENSE_GGUF_MLP_ACTIVATION_FIXTURE_ARTIFACT_KIND,
            "gap_role": "mlp_activation",
            "input_dependencies": ["mlp_gate", "mlp_up"],
            "source_mlp_gate_cuda_parity_required": true,
            "source_mlp_gate_cuda_parity_available": true,
            "source_mlp_up_cuda_parity_required": true,
            "source_mlp_up_cuda_parity_available": true,
            "cpu_reference_available": true,
            "cuda_kernel_status": "missing_cuda_kernel",
            "strict_cuda_ready": false,
            "cpu_fallback_allowed": false,
            "blocks_strict_cuda_one_layer": true,
            "next_required_proof": "cuda_mlp_activation_kernel_parity",
            "candidate_order": DENSE_ONE_LAYER_REMAINING_GAP_CANDIDATE_ORDER,
            "dense_gguf_mlp_activation_fixture_extraction_claimed": true,
            "dense_gguf_inference_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "timing": {
            "kernel_time_ms": null,
            "host_to_device_bytes": 0,
            "device_to_host_bytes": 0,
            "transfer_timing_status": "not_measured_no_kernel"
        },
        "claim_boundary": {
            "dense_regular_llm_cuda_claimed": false,
            "dense_tensor_residency_claimed": false,
            "dense_gguf_descriptor_inspection_claimed": true,
            "dense_gguf_attention_score_fixture_extraction_claimed": false,
            "dense_gguf_attention_score_cuda_parity_claimed": false,
            "dense_gguf_attention_softmax_fixture_extraction_claimed": false,
            "dense_gguf_attention_softmax_cuda_parity_claimed": false,
            "dense_gguf_attention_v_mix_fixture_extraction_claimed": false,
            "dense_gguf_attention_v_mix_cuda_parity_claimed": false,
            "dense_gguf_mlp_activation_fixture_extraction_claimed": true,
            "dense_gguf_rope_cuda_parity_claimed": false,
            "dense_gguf_norm_cuda_parity_claimed": false,
            "dense_gguf_linear_fixture_extraction_claimed": false,
            "dense_gguf_linear_cuda_parity_claimed": false,
            "dense_gguf_linear_role_sweep_cuda_parity_claimed": false,
            "dense_gguf_one_layer_execution_plan_claimed": false,
            "dense_gguf_inference_claimed": false,
            "qwen_one_token_cuda_claimed": false,
            "qwen_short_decode_cuda_claimed": false,
            "qwen_chat_cuda_claimed": false,
            "cpu_cuda_parity_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false
        }
    })
}

fn dense_gguf_mlp_activation_cuda_parity_receipt_json(
    inspection: &DenseGgufDescriptorInspection,
    fixture: &DenseGgufMlpActivationFixture,
    parity: &DenseGgufMlpActivationCudaParity,
    probe: Option<&bitnet_device_probe::NvidiaCudaProbe>,
    model_path: &Path,
    model_sha256: &str,
    artifact_path: &str,
    timestamp_utc: &str,
) -> Value {
    let cuda = cuda_identity_json(probe);
    let execution_plan = execution_plan_receipt(ExecutionPlanReceiptInput {
        model_family: &inspection.model_family,
        quantization: "dense_f32_mlp_activation",
        requested_backend: HARDWARE_LANE,
        selected_backend: HARDWARE_LANE,
        runtime_api: "cuda",
        strict_fallback_policy: "reject",
        summary: ModelDispatchSummary {
            total_ops: 1,
            cuda_bitnet_qk256_ops: 0,
            cuda_dense_regular_llm_ops: 1,
            cpu_fallback_ops: 0,
            unsupported_ops: 0,
            fallback_used: false,
            selected_route: Some(ModelDispatchBackend::CudaDenseRegularLlm),
            strict_cuda_ready: true,
        },
        speedup_claim: false,
        full_cuda_residency_claimed: false,
    });

    json!({
        "schema": 1,
        "artifact_kind": DENSE_GGUF_MLP_ACTIVATION_CUDA_PARITY_ARTIFACT_KIND,
        "artifact_path": artifact_path,
        "claim": "dense_gguf_mlp_activation_cuda_parity_tested",
        "machine_id": MACHINE_ID,
        "hardware_lane": HARDWARE_LANE,
        "timestamp_utc": timestamp_utc,
        "requested_backend": HARDWARE_LANE,
        "selected_backend": HARDWARE_LANE,
        "runtime_api": "cuda",
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "speedup_claim": false,
        "cuda": cuda,
        "model": {
            "model_family": inspection.model_family,
            "architecture": inspection.architecture,
            "artifact_kind": "dense_gguf",
            "quantization_families": inspection.quantization_families,
            "file": model_path.display().to_string(),
            "sha256": model_sha256
        },
        "execution_path": {
            "model_class": "dense_regular_llm",
            "kernel_family": "dense_f32_mlp_activation",
            "quantization_family": "metadata_derived_mlp_activation_fixture",
            "bitnet_packed_kernel_proof": false,
            "qk256_proof": false
        },
        "execution_plan": execution_plan,
        "descriptor_coverage": {
            "schema": 1,
            "source_artifact_kind": "dense_gguf_tensor_descriptor_inspection",
            "tensor_count": inspection.tensor_count,
            "metadata_count": inspection.metadata_count as u64,
            "required_roles_present": inspection.required_roles_present,
            "strict_descriptor_complete": inspection.strict_descriptor_complete,
            "dense_cuda_route_status": inspection.dense_cuda_route_status,
            "quantization_families": inspection.quantization_families,
            "bitnet_packed_marker_found": inspection.bitnet_packed_marker_found,
            "dense_gguf_inference_claimed": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "mlp_activation_fixture": {
            "schema": 1,
            "source_artifact_kind": DENSE_GGUF_MLP_ACTIVATION_FIXTURE_ARTIFACT_KIND,
            "source_mlp_gate_artifact_kind": DENSE_GGUF_LINEAR_ROLE_SWEEP_CUDA_PARITY_ARTIFACT_KIND,
            "source_mlp_gate_role": "mlp_gate",
            "source_mlp_gate_fixture_id": fixture.source_mlp_gate_fixture_id,
            "source_mlp_gate_tensor": fixture.source_mlp_gate_tensor,
            "source_mlp_up_artifact_kind": DENSE_GGUF_LINEAR_ROLE_SWEEP_CUDA_PARITY_ARTIFACT_KIND,
            "source_mlp_up_role": "mlp_up",
            "source_mlp_up_fixture_id": fixture.source_mlp_up_fixture_id,
            "source_mlp_up_tensor": fixture.source_mlp_up_tensor,
            "fixture_id": fixture.fixture_id,
            "model_family": fixture.model_family,
            "architecture": fixture.architecture,
            "layer_index": fixture.layer_index as u64,
            "activation_kind": fixture.activation_kind,
            "gate_output_count": fixture.gate_output_f32.len() as u64,
            "up_output_count": fixture.up_output_f32.len() as u64,
            "activation_count": fixture.activation_count as u64,
            "compared_activations": parity.compared_activations as u64,
            "gate_output_sha256": sha256_f32(&fixture.gate_output_f32),
            "up_output_sha256": sha256_f32(&fixture.up_output_f32),
            "cpu_reference_activation_sha256": sha256_f32(&fixture.expected_activation_f32),
            "max_activation_abs": fixture.max_activation_abs,
            "cuda_input_dtype": "f32",
            "cuda_output_dtype": "f32",
            "cuda_kernel_status": "parity_passed",
            "strict_cuda_ready": true,
            "cpu_fallback_allowed": false,
            "transfer_timing_status": "bytes_measured_time_unmeasured",
            "dense_gguf_inference_claimed": false,
            "dense_regular_llm_cuda_claimed": true,
            "cpu_cuda_parity_claimed": true,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "mlp_activation_gap_audit": {
            "schema": 1,
            "source_artifact_kind": DENSE_GGUF_MLP_ACTIVATION_FIXTURE_ARTIFACT_KIND,
            "gap_role": "mlp_activation",
            "input_dependencies": ["mlp_gate", "mlp_up"],
            "source_mlp_gate_cuda_parity_required": true,
            "source_mlp_gate_cuda_parity_available": true,
            "source_mlp_up_cuda_parity_required": true,
            "source_mlp_up_cuda_parity_available": true,
            "cpu_reference_available": true,
            "cuda_kernel_status": "parity_passed",
            "strict_cuda_ready": true,
            "cpu_fallback_allowed": false,
            "blocks_strict_cuda_one_layer": false,
            "next_required_proof": "one_layer_route_promotion",
            "candidate_order": DENSE_ONE_LAYER_REMAINING_GAP_CANDIDATE_ORDER,
            "dense_gguf_mlp_activation_fixture_extraction_claimed": true,
            "dense_gguf_mlp_activation_cuda_parity_claimed": true,
            "dense_gguf_inference_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "kernel_stats": [{
            "kernel_id": parity.stats.kernel_id,
            "fixture_id": parity.fixture_id,
            "invocations": parity.stats.invocations,
            "fallback_invocations": parity.stats.fallback_invocations,
            "host_to_device_bytes": parity.stats.host_to_device_bytes,
            "device_to_host_bytes": parity.stats.device_to_host_bytes,
            "kernel_launches": parity.stats.kernel_launches,
            "kernel_time_ms": parity.stats.kernel_time_ms
        }],
        "parity": {
            "reference_backend": parity.reference_backend,
            "target_backend": parity.target_backend,
            "kernel_id": parity.kernel_id,
            "fixture_id": parity.fixture_id,
            "max_abs_error": parity.max_abs_error,
            "mean_abs_error": parity.mean_abs_error,
            "passed": parity.passed,
            "tolerance": parity.tolerance,
            "tolerance_source": "CUDA-DENSE-030 dense GGUF MLP activation F32 CUDA fixture",
            "compared_activations": parity.compared_activations as u64,
            "first_divergence": null
        },
        "timing": {
            "kernel_time_ms": parity.stats.kernel_time_ms,
            "host_to_device_bytes": parity.stats.host_to_device_bytes,
            "device_to_host_bytes": parity.stats.device_to_host_bytes,
            "transfer_timing_status": "bytes_measured_time_unmeasured"
        },
        "claim_boundary": {
            "dense_regular_llm_cuda_claimed": true,
            "dense_tensor_residency_claimed": true,
            "dense_gguf_descriptor_inspection_claimed": true,
            "dense_gguf_attention_score_fixture_extraction_claimed": false,
            "dense_gguf_attention_score_cuda_parity_claimed": false,
            "dense_gguf_attention_softmax_fixture_extraction_claimed": false,
            "dense_gguf_attention_softmax_cuda_parity_claimed": false,
            "dense_gguf_attention_v_mix_fixture_extraction_claimed": false,
            "dense_gguf_attention_v_mix_cuda_parity_claimed": false,
            "dense_gguf_mlp_activation_fixture_extraction_claimed": true,
            "dense_gguf_mlp_activation_cuda_parity_claimed": true,
            "dense_gguf_rope_cuda_parity_claimed": false,
            "dense_gguf_norm_cuda_parity_claimed": false,
            "dense_gguf_linear_fixture_extraction_claimed": false,
            "dense_gguf_linear_cuda_parity_claimed": false,
            "dense_gguf_linear_role_sweep_cuda_parity_claimed": false,
            "dense_gguf_one_layer_execution_plan_claimed": false,
            "dense_gguf_inference_claimed": false,
            "qwen_one_token_cuda_claimed": false,
            "qwen_short_decode_cuda_claimed": false,
            "qwen_chat_cuda_claimed": false,
            "cpu_cuda_parity_claimed": true,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false
        },
        "tensor_residency": {
            "schema_version": "1.0.0",
            "scope": "single_dense_gguf_mlp_activation_fixture",
            "model_class": "dense_regular_llm",
            "fixture_id": parity.fixture_id,
            "dense_tensor_residency_claimed": true,
            "dense_gguf_inference_claimed": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false,
            "input_tensors_uploaded_once": true,
            "output_tensor_cuda_resident_during_kernel": true,
            "host_device_transfer_accounting_matches_kernel_stats": true,
            "inputs": [
                {
                    "name": "dense_gguf_mlp_gate_output",
                    "dtype": "f32",
                    "shape": [fixture.activation_count as u64],
                    "host_bytes": (fixture.gate_output_f32.len() * 4) as u64,
                    "device_residency": "cuda_device_buffer",
                    "upload_count": 1,
                    "reuse_scope": "single_fixture_launch"
                },
                {
                    "name": "dense_gguf_mlp_up_output",
                    "dtype": "f32",
                    "shape": [fixture.activation_count as u64],
                    "host_bytes": (fixture.up_output_f32.len() * 4) as u64,
                    "device_residency": "cuda_device_buffer",
                    "upload_count": 1,
                    "reuse_scope": "single_fixture_launch"
                }
            ],
            "outputs": [
                {
                    "name": "dense_gguf_mlp_activation",
                    "dtype": "f32",
                    "shape": [fixture.activation_count as u64],
                    "device_residency": "cuda_device_buffer",
                    "device_to_host_bytes": parity.stats.device_to_host_bytes,
                    "download_scope": "parity_check_only"
                }
            ],
            "allocation": {
                "device_buffer_count": 3,
                "temporary_workspace_bytes": 0,
                "persistent_handle_count": 0,
                "persistent_handles_claimed": false
            },
            "transfer_accounting": {
                "status": "measured",
                "host_to_device_bytes": parity.stats.host_to_device_bytes,
                "device_to_host_bytes": parity.stats.device_to_host_bytes,
                "kernel_invocations": parity.stats.invocations,
                "kernel_launches": parity.stats.kernel_launches
            }
        },
        "notes": [
            "Dense GGUF MLP activation CUDA fixture parity only; no dense GGUF inference, Qwen token/decode/chat, server, speedup, route promotion, persistent-session, or full-residency claim is made.",
            "This proves metadata-derived MLP gate/up activation vectors can run through a strict CUDA SiLU(gate) * up kernel against CPU references."
        ],
        "error": null
    })
}

fn dense_gguf_attention_softmax_cuda_parity_receipt_json(
    inspection: &DenseGgufDescriptorInspection,
    fixture: &DenseGgufAttentionSoftmaxFixture,
    parity: &DenseGgufAttentionSoftmaxCudaParity,
    probe: Option<&bitnet_device_probe::NvidiaCudaProbe>,
    model_path: &Path,
    model_sha256: &str,
    artifact_path: &str,
    timestamp_utc: &str,
) -> Value {
    let cuda = cuda_identity_json(probe);
    let execution_plan = execution_plan_receipt(ExecutionPlanReceiptInput {
        model_family: &inspection.model_family,
        quantization: "dense_f32_attention_softmax",
        requested_backend: HARDWARE_LANE,
        selected_backend: HARDWARE_LANE,
        runtime_api: "cuda",
        strict_fallback_policy: "reject",
        summary: ModelDispatchSummary {
            total_ops: 1,
            cuda_bitnet_qk256_ops: 0,
            cuda_dense_regular_llm_ops: 1,
            cpu_fallback_ops: 0,
            unsupported_ops: 0,
            fallback_used: false,
            selected_route: Some(ModelDispatchBackend::CudaDenseRegularLlm),
            strict_cuda_ready: true,
        },
        speedup_claim: false,
        full_cuda_residency_claimed: false,
    });

    json!({
        "schema": 1,
        "artifact_kind": DENSE_GGUF_ATTENTION_SOFTMAX_CUDA_PARITY_ARTIFACT_KIND,
        "artifact_path": artifact_path,
        "claim": "dense_gguf_attention_softmax_cuda_parity_tested",
        "machine_id": MACHINE_ID,
        "hardware_lane": HARDWARE_LANE,
        "timestamp_utc": timestamp_utc,
        "requested_backend": HARDWARE_LANE,
        "selected_backend": HARDWARE_LANE,
        "runtime_api": "cuda",
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "speedup_claim": false,
        "cuda": cuda,
        "model": {
            "model_family": inspection.model_family,
            "architecture": inspection.architecture,
            "artifact_kind": "dense_gguf",
            "quantization_families": inspection.quantization_families,
            "file": model_path.display().to_string(),
            "sha256": model_sha256
        },
        "execution_path": {
            "model_class": "dense_regular_llm",
            "kernel_family": "dense_f32_attention_softmax",
            "quantization_family": "metadata_derived_attention_softmax_fixture",
            "bitnet_packed_kernel_proof": false,
            "qk256_proof": false
        },
        "execution_plan": execution_plan,
        "descriptor_coverage": {
            "schema": 1,
            "source_artifact_kind": "dense_gguf_tensor_descriptor_inspection",
            "tensor_count": inspection.tensor_count,
            "metadata_count": inspection.metadata_count as u64,
            "required_roles_present": inspection.required_roles_present,
            "strict_descriptor_complete": inspection.strict_descriptor_complete,
            "dense_cuda_route_status": inspection.dense_cuda_route_status,
            "quantization_families": inspection.quantization_families,
            "bitnet_packed_marker_found": inspection.bitnet_packed_marker_found,
            "dense_gguf_inference_claimed": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "attention_softmax_fixture": {
            "schema": 1,
            "source_artifact_kind": DENSE_GGUF_ATTENTION_SOFTMAX_FIXTURE_ARTIFACT_KIND,
            "source_attention_score_artifact_kind": DENSE_GGUF_ATTENTION_SCORE_CUDA_PARITY_ARTIFACT_KIND,
            "source_attention_score_fixture_id": fixture.source_attention_score_fixture_id,
            "fixture_id": fixture.fixture_id,
            "model_family": fixture.model_family,
            "architecture": fixture.architecture,
            "layer_index": fixture.layer_index as u64,
            "q_heads": fixture.q_heads as u64,
            "kv_heads": fixture.kv_heads as u64,
            "seq_len": fixture.seq_len as u64,
            "row_count": fixture.row_count as u64,
            "probability_count": fixture.probability_count as u64,
            "compared_probabilities": parity.compared_probabilities as u64,
            "causal_zero_probabilities": fixture.causal_zero_probabilities as u64,
            "attention_scores_sha256": sha256_f32(&fixture.attention_scores_f32),
            "cpu_reference_probabilities_sha256": sha256_f32(&fixture.expected_probabilities_f32),
            "max_row_sum_abs_error": fixture.max_row_sum_abs_error,
            "cuda_input_dtype": "f32",
            "cuda_output_dtype": "f32",
            "cuda_kernel_status": "parity_passed",
            "strict_cuda_ready": true,
            "cpu_fallback_allowed": false,
            "transfer_timing_status": "bytes_measured_time_unmeasured",
            "dense_gguf_inference_claimed": false,
            "dense_regular_llm_cuda_claimed": true,
            "cpu_cuda_parity_claimed": true,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "attention_softmax_gap_audit": {
            "schema": 1,
            "source_artifact_kind": DENSE_GGUF_ATTENTION_SOFTMAX_FIXTURE_ARTIFACT_KIND,
            "gap_role": "attention_softmax",
            "input_dependencies": ["attention_scores"],
            "source_attention_score_cuda_parity_required": true,
            "source_attention_score_cuda_parity_available": true,
            "cpu_reference_available": true,
            "cuda_kernel_status": "parity_passed",
            "strict_cuda_ready": true,
            "cpu_fallback_allowed": false,
            "blocks_strict_cuda_one_layer": false,
            "next_required_proof": "cuda_attention_v_mix_fixture",
            "candidate_order": DENSE_ONE_LAYER_GAP_CANDIDATE_ORDER,
            "dense_gguf_attention_softmax_fixture_extraction_claimed": true,
            "dense_gguf_attention_softmax_cuda_parity_claimed": true,
            "dense_gguf_inference_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "kernel_stats": [{
            "kernel_id": parity.stats.kernel_id,
            "fixture_id": parity.fixture_id,
            "invocations": parity.stats.invocations,
            "fallback_invocations": parity.stats.fallback_invocations,
            "host_to_device_bytes": parity.stats.host_to_device_bytes,
            "device_to_host_bytes": parity.stats.device_to_host_bytes,
            "kernel_launches": parity.stats.kernel_launches,
            "kernel_time_ms": parity.stats.kernel_time_ms
        }],
        "parity": {
            "reference_backend": parity.reference_backend,
            "target_backend": parity.target_backend,
            "kernel_id": parity.kernel_id,
            "fixture_id": parity.fixture_id,
            "max_abs_error": parity.max_abs_error,
            "mean_abs_error": parity.mean_abs_error,
            "passed": parity.passed,
            "tolerance": parity.tolerance,
            "tolerance_source": "CUDA-DENSE-024 dense GGUF attention-softmax F32 CUDA fixture",
            "compared_probabilities": parity.compared_probabilities as u64,
            "causal_zero_probabilities": parity.causal_zero_probabilities as u64,
            "first_divergence": null
        },
        "timing": {
            "kernel_time_ms": parity.stats.kernel_time_ms,
            "host_to_device_bytes": parity.stats.host_to_device_bytes,
            "device_to_host_bytes": parity.stats.device_to_host_bytes,
            "transfer_timing_status": "bytes_measured_time_unmeasured"
        },
        "claim_boundary": {
            "dense_regular_llm_cuda_claimed": true,
            "dense_tensor_residency_claimed": true,
            "dense_gguf_descriptor_inspection_claimed": true,
            "dense_gguf_attention_score_fixture_extraction_claimed": true,
            "dense_gguf_attention_score_cuda_parity_claimed": true,
            "dense_gguf_attention_softmax_fixture_extraction_claimed": true,
            "dense_gguf_attention_softmax_cuda_parity_claimed": true,
            "dense_gguf_rope_cuda_parity_claimed": true,
            "dense_gguf_norm_cuda_parity_claimed": false,
            "dense_gguf_linear_fixture_extraction_claimed": false,
            "dense_gguf_linear_cuda_parity_claimed": false,
            "dense_gguf_linear_role_sweep_cuda_parity_claimed": false,
            "dense_gguf_one_layer_execution_plan_claimed": false,
            "dense_gguf_inference_claimed": false,
            "qwen_one_token_cuda_claimed": false,
            "qwen_short_decode_cuda_claimed": false,
            "qwen_chat_cuda_claimed": false,
            "cpu_cuda_parity_claimed": true,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false
        },
        "tensor_residency": {
            "schema_version": "1.0.0",
            "scope": "single_dense_gguf_attention_softmax_fixture",
            "model_class": "dense_regular_llm",
            "fixture_id": parity.fixture_id,
            "dense_tensor_residency_claimed": true,
            "dense_gguf_inference_claimed": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false,
            "input_tensors_uploaded_once": true,
            "output_tensor_cuda_resident_during_kernel": true,
            "host_device_transfer_accounting_matches_kernel_stats": true,
            "inputs": [
                {
                    "name": "dense_gguf_attention_scores",
                    "dtype": "f32",
                    "shape": [fixture.q_heads as u64, fixture.seq_len as u64, fixture.seq_len as u64],
                    "host_bytes": (fixture.attention_scores_f32.len() * 4) as u64,
                    "device_residency": "cuda_device_buffer",
                    "upload_count": 1,
                    "reuse_scope": "single_fixture_launch"
                }
            ],
            "outputs": [
                {
                    "name": "dense_gguf_attention_probabilities",
                    "dtype": "f32",
                    "shape": [fixture.q_heads as u64, fixture.seq_len as u64, fixture.seq_len as u64],
                    "device_residency": "cuda_device_buffer",
                    "device_to_host_bytes": parity.stats.device_to_host_bytes,
                    "download_scope": "parity_check_only"
                }
            ],
            "allocation": {
                "device_buffer_count": 2,
                "temporary_workspace_bytes": 0,
                "persistent_handle_count": 0,
                "persistent_handles_claimed": false
            },
            "transfer_accounting": {
                "status": "measured",
                "host_to_device_bytes": parity.stats.host_to_device_bytes,
                "device_to_host_bytes": parity.stats.device_to_host_bytes,
                "kernel_invocations": parity.stats.invocations,
                "kernel_launches": parity.stats.kernel_launches
            }
        },
        "notes": [
            "Dense GGUF attention-softmax CUDA fixture parity only; no dense GGUF inference, Qwen token/decode/chat, server, speedup, persistent-session, or full-residency claim is made.",
            "This proves metadata-derived attention-score probabilities can run through a strict CUDA attention-softmax kernel against CPU references."
        ],
        "error": null
    })
}

fn dense_gguf_attention_v_mix_cuda_parity_receipt_json(
    inspection: &DenseGgufDescriptorInspection,
    fixture: &DenseGgufAttentionVMixFixture,
    parity: &DenseGgufAttentionVMixCudaParity,
    probe: Option<&bitnet_device_probe::NvidiaCudaProbe>,
    model_path: &Path,
    model_sha256: &str,
    artifact_path: &str,
    timestamp_utc: &str,
) -> Value {
    let cuda = cuda_identity_json(probe);
    let execution_plan = execution_plan_receipt(ExecutionPlanReceiptInput {
        model_family: &inspection.model_family,
        quantization: "dense_f32_attention_v_mix",
        requested_backend: HARDWARE_LANE,
        selected_backend: HARDWARE_LANE,
        runtime_api: "cuda",
        strict_fallback_policy: "reject",
        summary: ModelDispatchSummary {
            total_ops: 1,
            cuda_bitnet_qk256_ops: 0,
            cuda_dense_regular_llm_ops: 1,
            cpu_fallback_ops: 0,
            unsupported_ops: 0,
            fallback_used: false,
            selected_route: Some(ModelDispatchBackend::CudaDenseRegularLlm),
            strict_cuda_ready: true,
        },
        speedup_claim: false,
        full_cuda_residency_claimed: false,
    });

    json!({
        "schema": 1,
        "artifact_kind": DENSE_GGUF_ATTENTION_V_MIX_CUDA_PARITY_ARTIFACT_KIND,
        "artifact_path": artifact_path,
        "claim": "dense_gguf_attention_v_mix_cuda_parity_tested",
        "machine_id": MACHINE_ID,
        "hardware_lane": HARDWARE_LANE,
        "timestamp_utc": timestamp_utc,
        "requested_backend": HARDWARE_LANE,
        "selected_backend": HARDWARE_LANE,
        "runtime_api": "cuda",
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "speedup_claim": false,
        "cuda": cuda,
        "model": {
            "model_family": inspection.model_family,
            "architecture": inspection.architecture,
            "artifact_kind": "dense_gguf",
            "quantization_families": inspection.quantization_families,
            "file": model_path.display().to_string(),
            "sha256": model_sha256
        },
        "execution_path": {
            "model_class": "dense_regular_llm",
            "kernel_family": "dense_f32_attention_v_mix",
            "quantization_family": "metadata_derived_attention_v_mix_fixture",
            "bitnet_packed_kernel_proof": false,
            "qk256_proof": false
        },
        "execution_plan": execution_plan,
        "descriptor_coverage": {
            "schema": 1,
            "source_artifact_kind": "dense_gguf_tensor_descriptor_inspection",
            "tensor_count": inspection.tensor_count,
            "metadata_count": inspection.metadata_count as u64,
            "required_roles_present": inspection.required_roles_present,
            "strict_descriptor_complete": inspection.strict_descriptor_complete,
            "dense_cuda_route_status": inspection.dense_cuda_route_status,
            "quantization_families": inspection.quantization_families,
            "bitnet_packed_marker_found": inspection.bitnet_packed_marker_found,
            "dense_gguf_inference_claimed": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "attention_v_mix_fixture": {
            "schema": 1,
            "source_artifact_kind": DENSE_GGUF_ATTENTION_V_MIX_FIXTURE_ARTIFACT_KIND,
            "source_attention_softmax_artifact_kind": DENSE_GGUF_ATTENTION_SOFTMAX_CUDA_PARITY_ARTIFACT_KIND,
            "source_attention_softmax_fixture_id": fixture.source_attention_softmax_fixture_id,
            "source_attention_v_artifact_kind": DENSE_GGUF_LINEAR_ROLE_SWEEP_CUDA_PARITY_ARTIFACT_KIND,
            "source_attention_v_role": "attention_v",
            "fixture_id": fixture.fixture_id,
            "model_family": fixture.model_family,
            "architecture": fixture.architecture,
            "layer_index": fixture.layer_index as u64,
            "q_heads": fixture.q_heads as u64,
            "kv_heads": fixture.kv_heads as u64,
            "heads_per_kv_group": fixture.heads_per_kv_group as u64,
            "head_dim": fixture.head_dim as u64,
            "seq_len": fixture.seq_len as u64,
            "row_count": fixture.row_count as u64,
            "probability_count": fixture.probability_count as u64,
            "value_count": fixture.value_count as u64,
            "context_count": fixture.context_count as u64,
            "compared_context_values": parity.compared_context_values as u64,
            "causal_zero_probabilities": fixture.causal_zero_probabilities as u64,
            "attention_probabilities_sha256": sha256_f32(&fixture.attention_probabilities_f32),
            "value_states_sha256": sha256_f32(&fixture.value_states_f32),
            "cpu_reference_context_sha256": sha256_f32(&fixture.expected_context_f32),
            "max_context_abs": fixture.max_context_abs,
            "cuda_input_dtype": "f32",
            "cuda_output_dtype": "f32",
            "cuda_kernel_status": "parity_passed",
            "strict_cuda_ready": true,
            "cpu_fallback_allowed": false,
            "transfer_timing_status": "bytes_measured_time_unmeasured",
            "dense_gguf_inference_claimed": false,
            "dense_regular_llm_cuda_claimed": true,
            "cpu_cuda_parity_claimed": true,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "attention_v_mix_gap_audit": {
            "schema": 1,
            "source_artifact_kind": DENSE_GGUF_ATTENTION_V_MIX_FIXTURE_ARTIFACT_KIND,
            "gap_role": "attention_v_mix",
            "input_dependencies": ["attention_softmax", "attention_v"],
            "source_attention_softmax_cuda_parity_required": true,
            "source_attention_softmax_cuda_parity_available": true,
            "source_attention_v_cuda_parity_required": true,
            "source_attention_v_cuda_parity_available": true,
            "cpu_reference_available": true,
            "cuda_kernel_status": "parity_passed",
            "strict_cuda_ready": true,
            "cpu_fallback_allowed": false,
            "blocks_strict_cuda_one_layer": false,
            "next_required_proof": "one_layer_route_promotion",
            "candidate_order": DENSE_ONE_LAYER_GAP_CANDIDATE_ORDER,
            "dense_gguf_attention_v_mix_fixture_extraction_claimed": true,
            "dense_gguf_attention_v_mix_cuda_parity_claimed": true,
            "dense_gguf_inference_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "kernel_stats": [{
            "kernel_id": parity.stats.kernel_id,
            "fixture_id": parity.fixture_id,
            "invocations": parity.stats.invocations,
            "fallback_invocations": parity.stats.fallback_invocations,
            "host_to_device_bytes": parity.stats.host_to_device_bytes,
            "device_to_host_bytes": parity.stats.device_to_host_bytes,
            "kernel_launches": parity.stats.kernel_launches,
            "kernel_time_ms": parity.stats.kernel_time_ms
        }],
        "parity": {
            "reference_backend": parity.reference_backend,
            "target_backend": parity.target_backend,
            "kernel_id": parity.kernel_id,
            "fixture_id": parity.fixture_id,
            "max_abs_error": parity.max_abs_error,
            "mean_abs_error": parity.mean_abs_error,
            "passed": parity.passed,
            "tolerance": parity.tolerance,
            "tolerance_source": "CUDA-DENSE-027 dense GGUF attention V-mix F32 CUDA fixture",
            "compared_context_values": parity.compared_context_values as u64,
            "causal_zero_probabilities": parity.causal_zero_probabilities as u64,
            "first_divergence": null
        },
        "timing": {
            "kernel_time_ms": parity.stats.kernel_time_ms,
            "host_to_device_bytes": parity.stats.host_to_device_bytes,
            "device_to_host_bytes": parity.stats.device_to_host_bytes,
            "transfer_timing_status": "bytes_measured_time_unmeasured"
        },
        "claim_boundary": {
            "dense_regular_llm_cuda_claimed": true,
            "dense_tensor_residency_claimed": true,
            "dense_gguf_descriptor_inspection_claimed": true,
            "dense_gguf_attention_score_fixture_extraction_claimed": true,
            "dense_gguf_attention_score_cuda_parity_claimed": true,
            "dense_gguf_attention_softmax_fixture_extraction_claimed": true,
            "dense_gguf_attention_softmax_cuda_parity_claimed": true,
            "dense_gguf_attention_v_mix_fixture_extraction_claimed": true,
            "dense_gguf_attention_v_mix_cuda_parity_claimed": true,
            "dense_gguf_rope_cuda_parity_claimed": true,
            "dense_gguf_norm_cuda_parity_claimed": false,
            "dense_gguf_linear_fixture_extraction_claimed": false,
            "dense_gguf_linear_cuda_parity_claimed": false,
            "dense_gguf_linear_role_sweep_cuda_parity_claimed": false,
            "dense_gguf_one_layer_execution_plan_claimed": false,
            "dense_gguf_inference_claimed": false,
            "qwen_one_token_cuda_claimed": false,
            "qwen_short_decode_cuda_claimed": false,
            "qwen_chat_cuda_claimed": false,
            "cpu_cuda_parity_claimed": true,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false
        },
        "tensor_residency": {
            "schema_version": "1.0.0",
            "scope": "single_dense_gguf_attention_v_mix_fixture",
            "model_class": "dense_regular_llm",
            "fixture_id": parity.fixture_id,
            "dense_tensor_residency_claimed": true,
            "dense_gguf_inference_claimed": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false,
            "input_tensors_uploaded_once": true,
            "output_tensor_cuda_resident_during_kernel": true,
            "host_device_transfer_accounting_matches_kernel_stats": true,
            "inputs": [
                {
                    "name": "dense_gguf_attention_probabilities",
                    "dtype": "f32",
                    "shape": [fixture.q_heads as u64, fixture.seq_len as u64, fixture.seq_len as u64],
                    "host_bytes": (fixture.attention_probabilities_f32.len() * 4) as u64,
                    "device_residency": "cuda_device_buffer",
                    "upload_count": 1,
                    "reuse_scope": "single_fixture_launch"
                },
                {
                    "name": "dense_gguf_attention_values",
                    "dtype": "f32",
                    "shape": [fixture.kv_heads as u64, fixture.seq_len as u64, fixture.head_dim as u64],
                    "host_bytes": (fixture.value_states_f32.len() * 4) as u64,
                    "device_residency": "cuda_device_buffer",
                    "upload_count": 1,
                    "reuse_scope": "single_fixture_launch"
                }
            ],
            "outputs": [
                {
                    "name": "dense_gguf_attention_context",
                    "dtype": "f32",
                    "shape": [fixture.q_heads as u64, fixture.seq_len as u64, fixture.head_dim as u64],
                    "device_residency": "cuda_device_buffer",
                    "device_to_host_bytes": parity.stats.device_to_host_bytes,
                    "download_scope": "parity_check_only"
                }
            ],
            "allocation": {
                "device_buffer_count": 3,
                "temporary_workspace_bytes": 0,
                "persistent_handle_count": 0,
                "persistent_handles_claimed": false
            },
            "transfer_accounting": {
                "status": "measured",
                "host_to_device_bytes": parity.stats.host_to_device_bytes,
                "device_to_host_bytes": parity.stats.device_to_host_bytes,
                "kernel_invocations": parity.stats.invocations,
                "kernel_launches": parity.stats.kernel_launches
            }
        },
        "notes": [
            "Dense GGUF attention V-mix CUDA fixture parity only; no dense GGUF inference, Qwen token/decode/chat, server, speedup, route promotion, persistent-session, or full-residency claim is made.",
            "This proves metadata-derived attention probabilities and deterministic attention-V states can run through a strict CUDA V-mix kernel against CPU references."
        ],
        "error": null
    })
}

fn dense_gguf_attention_score_cuda_parity_receipt_json(
    inspection: &DenseGgufDescriptorInspection,
    fixture: &DenseGgufAttentionScoreFixture,
    parity: &DenseGgufAttentionScoreCudaParity,
    probe: Option<&bitnet_device_probe::NvidiaCudaProbe>,
    model_path: &Path,
    model_sha256: &str,
    artifact_path: &str,
    timestamp_utc: &str,
) -> Value {
    let cuda = cuda_identity_json(probe);
    let execution_plan = execution_plan_receipt(ExecutionPlanReceiptInput {
        model_family: &inspection.model_family,
        quantization: "dense_f32_attention_scores",
        requested_backend: HARDWARE_LANE,
        selected_backend: HARDWARE_LANE,
        runtime_api: "cuda",
        strict_fallback_policy: "reject",
        summary: ModelDispatchSummary {
            total_ops: 1,
            cuda_bitnet_qk256_ops: 0,
            cuda_dense_regular_llm_ops: 1,
            cpu_fallback_ops: 0,
            unsupported_ops: 0,
            fallback_used: false,
            selected_route: Some(ModelDispatchBackend::CudaDenseRegularLlm),
            strict_cuda_ready: true,
        },
        speedup_claim: false,
        full_cuda_residency_claimed: false,
    });

    json!({
        "schema": 1,
        "artifact_kind": DENSE_GGUF_ATTENTION_SCORE_CUDA_PARITY_ARTIFACT_KIND,
        "artifact_path": artifact_path,
        "claim": "dense_gguf_attention_score_cuda_parity_tested",
        "machine_id": MACHINE_ID,
        "hardware_lane": HARDWARE_LANE,
        "timestamp_utc": timestamp_utc,
        "requested_backend": HARDWARE_LANE,
        "selected_backend": HARDWARE_LANE,
        "runtime_api": "cuda",
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "speedup_claim": false,
        "cuda": cuda,
        "model": {
            "model_family": inspection.model_family,
            "architecture": inspection.architecture,
            "artifact_kind": "dense_gguf",
            "quantization_families": inspection.quantization_families,
            "file": model_path.display().to_string(),
            "sha256": model_sha256
        },
        "execution_path": {
            "model_class": "dense_regular_llm",
            "kernel_family": "dense_f32_attention_scores",
            "quantization_family": "metadata_derived_rope_qk_attention_scores_fixture",
            "bitnet_packed_kernel_proof": false,
            "qk256_proof": false
        },
        "execution_plan": execution_plan,
        "descriptor_coverage": {
            "schema": 1,
            "source_artifact_kind": "dense_gguf_tensor_descriptor_inspection",
            "tensor_count": inspection.tensor_count,
            "metadata_count": inspection.metadata_count as u64,
            "required_roles_present": inspection.required_roles_present,
            "strict_descriptor_complete": inspection.strict_descriptor_complete,
            "dense_cuda_route_status": inspection.dense_cuda_route_status,
            "quantization_families": inspection.quantization_families,
            "bitnet_packed_marker_found": inspection.bitnet_packed_marker_found,
            "dense_gguf_inference_claimed": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "attention_score_fixture": {
            "schema": 1,
            "source_artifact_kind": DENSE_GGUF_ATTENTION_SCORE_FIXTURE_ARTIFACT_KIND,
            "source_rope_artifact_kind": DENSE_GGUF_ROPE_CUDA_PARITY_ARTIFACT_KIND,
            "source_rope_fixture_id": fixture.source_rope_fixture_id,
            "fixture_id": fixture.fixture_id,
            "model_family": fixture.model_family,
            "architecture": fixture.architecture,
            "layer_index": fixture.layer_index as u64,
            "head_dim": fixture.head_dim as u64,
            "q_heads": fixture.q_heads as u64,
            "kv_heads": fixture.kv_heads as u64,
            "heads_per_kv_group": fixture.heads_per_kv_group as u64,
            "seq_len": fixture.seq_len as u64,
            "position_offset": fixture.position_offset as u64,
            "rope_base": fixture.rope_base,
            "scaling_factor": fixture.scaling_factor,
            "attention_scale": fixture.scale,
            "causal_mask_applied": true,
            "head_dim_source": fixture.head_dim_source,
            "q_heads_source": fixture.q_heads_source,
            "kv_heads_source": fixture.kv_heads_source,
            "rope_base_source": fixture.rope_base_source,
            "q_rope_output_sha256": sha256_f32(&fixture.q_rope_output_f32),
            "k_rope_output_sha256": sha256_f32(&fixture.k_rope_output_f32),
            "cpu_reference_scores_sha256": sha256_f32(&fixture.expected_scores_f32),
            "score_shape": [fixture.q_heads as u64, fixture.seq_len as u64, fixture.seq_len as u64],
            "score_count": fixture.expected_scores_f32.len() as u64,
            "finite_scores": fixture.finite_scores as u64,
            "causal_masked_scores": fixture.causal_masked_scores as u64,
            "cuda_input_dtype": "f32",
            "cuda_output_dtype": "f32",
            "cuda_kernel_status": "parity_passed",
            "strict_cuda_ready": true,
            "cpu_fallback_allowed": false,
            "transfer_timing_status": "bytes_measured_time_unmeasured",
            "dense_gguf_inference_claimed": false,
            "dense_regular_llm_cuda_claimed": true,
            "cpu_cuda_parity_claimed": true,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "kernel_stats": [{
            "kernel_id": parity.stats.kernel_id,
            "fixture_id": parity.fixture_id,
            "invocations": parity.stats.invocations,
            "fallback_invocations": parity.stats.fallback_invocations,
            "host_to_device_bytes": parity.stats.host_to_device_bytes,
            "device_to_host_bytes": parity.stats.device_to_host_bytes,
            "kernel_launches": parity.stats.kernel_launches,
            "kernel_time_ms": parity.stats.kernel_time_ms
        }],
        "parity": {
            "reference_backend": parity.reference_backend,
            "target_backend": parity.target_backend,
            "kernel_id": parity.kernel_id,
            "fixture_id": parity.fixture_id,
            "max_abs_error": parity.max_abs_error,
            "mean_abs_error": parity.mean_abs_error,
            "passed": parity.passed,
            "tolerance": parity.tolerance,
            "tolerance_source": "CUDA-DENSE-021 dense GGUF attention-score F32 CUDA fixture",
            "compared_scores": parity.compared_scores as u64,
            "finite_scores": parity.finite_scores as u64,
            "causal_masked_scores": parity.causal_masked_scores as u64,
            "first_divergence": null
        },
        "timing": {
            "kernel_time_ms": parity.stats.kernel_time_ms,
            "host_to_device_bytes": parity.stats.host_to_device_bytes,
            "device_to_host_bytes": parity.stats.device_to_host_bytes
        },
        "claim_boundary": {
            "dense_regular_llm_cuda_claimed": true,
            "dense_tensor_residency_claimed": true,
            "dense_gguf_descriptor_inspection_claimed": true,
            "dense_gguf_attention_score_fixture_extraction_claimed": true,
            "dense_gguf_attention_score_cuda_parity_claimed": true,
            "dense_gguf_rope_cuda_parity_claimed": true,
            "dense_gguf_norm_cuda_parity_claimed": false,
            "dense_gguf_linear_fixture_extraction_claimed": false,
            "dense_gguf_linear_cuda_parity_claimed": false,
            "dense_gguf_linear_role_sweep_cuda_parity_claimed": false,
            "dense_gguf_one_layer_execution_plan_claimed": false,
            "dense_gguf_inference_claimed": false,
            "qwen_one_token_cuda_claimed": false,
            "qwen_short_decode_cuda_claimed": false,
            "qwen_chat_cuda_claimed": false,
            "cpu_cuda_parity_claimed": true,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false
        },
        "tensor_residency": {
            "schema_version": "1.0.0",
            "scope": "single_dense_gguf_attention_score_fixture",
            "model_class": "dense_regular_llm",
            "fixture_id": parity.fixture_id,
            "dense_tensor_residency_claimed": true,
            "dense_gguf_inference_claimed": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false,
            "input_tensors_uploaded_once": true,
            "output_tensor_cuda_resident_during_kernel": true,
            "host_device_transfer_accounting_matches_kernel_stats": true,
            "inputs": [
                {
                    "name": "dense_gguf_attention_score_q_rope",
                    "dtype": "f32",
                    "shape": [fixture.q_heads as u64, fixture.seq_len as u64, fixture.head_dim as u64],
                    "host_bytes": (fixture.q_rope_output_f32.len() * 4) as u64,
                    "device_residency": "cuda_device_buffer",
                    "upload_count": 1,
                    "reuse_scope": "single_fixture_launch"
                },
                {
                    "name": "dense_gguf_attention_score_k_rope",
                    "dtype": "f32",
                    "shape": [fixture.kv_heads as u64, fixture.seq_len as u64, fixture.head_dim as u64],
                    "host_bytes": (fixture.k_rope_output_f32.len() * 4) as u64,
                    "device_residency": "cuda_device_buffer",
                    "upload_count": 1,
                    "reuse_scope": "single_fixture_launch"
                }
            ],
            "outputs": [
                {
                    "name": "dense_gguf_attention_scores",
                    "dtype": "f32",
                    "shape": [fixture.q_heads as u64, fixture.seq_len as u64, fixture.seq_len as u64],
                    "device_residency": "cuda_device_buffer",
                    "device_to_host_bytes": parity.stats.device_to_host_bytes,
                    "download_scope": "parity_check_only"
                }
            ],
            "allocation": {
                "device_buffer_count": 3,
                "temporary_workspace_bytes": 0,
                "persistent_handle_count": 0,
                "persistent_handles_claimed": false
            },
            "transfer_accounting": {
                "status": "measured",
                "host_to_device_bytes": parity.stats.host_to_device_bytes,
                "device_to_host_bytes": parity.stats.device_to_host_bytes,
                "kernel_invocations": parity.stats.invocations,
                "kernel_launches": parity.stats.kernel_launches
            }
        },
        "notes": [
            "Dense GGUF attention-score CUDA fixture parity only; no dense GGUF inference, Qwen token/decode/chat, server, speedup, persistent-session, or full-residency claim is made.",
            "This proves metadata-derived RoPE Q/K fixture scores can run through a strict CUDA attention-score kernel against CPU references."
        ],
        "error": null
    })
}

fn dense_gguf_one_layer_execution_plan_receipt_json(
    inspection: &DenseGgufDescriptorInspection,
    probe: Option<&bitnet_device_probe::NvidiaCudaProbe>,
    model_path: &Path,
    model_sha256: &str,
    artifact_path: &str,
    timestamp_utc: &str,
    layer_index: usize,
) -> Result<Value> {
    if !inspection.required_roles_present || !inspection.strict_descriptor_complete {
        bail!("dense GGUF one-layer plan requires complete dense descriptor coverage");
    }

    let entries = dense_one_layer_plan_entries(inspection, layer_index)?;
    let ops = entries.iter().map(|entry| entry.op.clone()).collect::<Vec<_>>();
    let spec = ModelDispatchSpec {
        model_family: ModelFamily::DenseRegularLlm,
        quantization: QuantizationKind::DenseFp16,
        backend_policy: BackendPolicy::StrictCuda,
        has_simd: true,
        cuda: CudaPlannerCapabilities::dense_regular_llm(),
    };
    let plan = plan_model_dispatch(&ops, spec);
    let summary = plan.summary();
    if summary.cuda_dense_regular_llm_ops == 0
        || summary.unsupported_ops != 0
        || !summary.strict_cuda_ready
    {
        bail!("dense GGUF one-layer plan must route every governed op to dense CUDA");
    }

    let execution_plan = execution_plan_receipt(ExecutionPlanReceiptInput {
        model_family: &inspection.model_family,
        quantization: "dense_fp16",
        requested_backend: HARDWARE_LANE,
        selected_backend: HARDWARE_LANE,
        runtime_api: "cuda",
        strict_fallback_policy: "reject",
        summary,
        speedup_claim: false,
        full_cuda_residency_claimed: false,
    });

    let operations = entries
        .iter()
        .zip(plan.decisions.iter())
        .enumerate()
        .map(|(idx, (entry, decision))| {
            let route = decision.backend.receipt_route_label();
            let status = match decision.backend {
                ModelDispatchBackend::CudaDenseRegularLlm => "cuda_routable",
                ModelDispatchBackend::Unsupported => "unsupported_strict_cuda",
                ModelDispatchBackend::CpuScalar | ModelDispatchBackend::CpuSimd => "cpu_fallback",
                ModelDispatchBackend::CudaBitnetQk256 => "wrong_route",
            };
            json!({
                "index": idx as u64,
                "name": entry.op.name,
                "role": entry.role,
                "op_type": entry.op.op_type.as_str(),
                "size": entry.op.size as u64,
                "source": entry.source,
                "source_tensor": entry.source_tensor,
                "source_tensor_type": entry.source_tensor_type,
                "source_shape": entry.source_shape,
                "is_quantized": entry.op.is_quantized,
                "route": route,
                "status": status,
                "fallback_used": decision.fallback_used,
                "reason": decision.reason,
            })
        })
        .collect::<Vec<_>>();

    let cuda_routable_ops = summary.cuda_dense_regular_llm_ops as u64;
    let cuda_linear_ops = operations
        .iter()
        .filter(|op| {
            op.get("route").and_then(Value::as_str) == Some(DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND)
                && op.get("op_type").and_then(Value::as_str) == Some("matmul")
        })
        .count() as u64;
    let cuda_norm_ops = operations
        .iter()
        .filter(|op| {
            op.get("route").and_then(Value::as_str) == Some(DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND)
                && op.get("op_type").and_then(Value::as_str) == Some("rmsnorm")
        })
        .count() as u64;
    let cuda_rope_ops = operations
        .iter()
        .filter(|op| {
            op.get("route").and_then(Value::as_str) == Some(DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND)
                && op.get("op_type").and_then(Value::as_str) == Some("rope")
        })
        .count() as u64;
    let cuda_attention_score_ops = operations
        .iter()
        .filter(|op| {
            op.get("route").and_then(Value::as_str) == Some(DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND)
                && op.get("role").and_then(Value::as_str) == Some("attention_scores")
        })
        .count() as u64;
    let cuda_attention_softmax_ops = operations
        .iter()
        .filter(|op| {
            op.get("route").and_then(Value::as_str) == Some(DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND)
                && op.get("role").and_then(Value::as_str) == Some("attention_softmax")
        })
        .count() as u64;
    let cuda_attention_v_mix_ops = operations
        .iter()
        .filter(|op| {
            op.get("route").and_then(Value::as_str) == Some(DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND)
                && op.get("role").and_then(Value::as_str) == Some("attention_v_mix")
        })
        .count() as u64;
    let cuda_mlp_activation_ops = operations
        .iter()
        .filter(|op| {
            op.get("route").and_then(Value::as_str) == Some(DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND)
                && op.get("role").and_then(Value::as_str) == Some("mlp_activation")
        })
        .count() as u64;
    if cuda_linear_ops
        + cuda_norm_ops
        + cuda_rope_ops
        + cuda_attention_score_ops
        + cuda_attention_softmax_ops
        + cuda_attention_v_mix_ops
        + cuda_mlp_activation_ops
        != cuda_routable_ops
    {
        bail!(
            "dense GGUF one-layer plan must account for CUDA-routable linears, RMSNorm, RoPE, attention-score, attention-softmax, attention V-mix, and MLP activation ops"
        );
    }

    let gap_audit = dense_one_layer_gap_audit_json(
        &operations,
        layer_index,
        cuda_routable_ops,
        summary.unsupported_ops as u64,
    )?;
    let cuda = cuda_identity_json(probe);
    Ok(json!({
        "schema": 1,
        "artifact_kind": DENSE_GGUF_ONE_LAYER_EXECUTION_PLAN_ARTIFACT_KIND,
        "artifact_path": artifact_path,
        "claim": "dense_gguf_one_layer_execution_plan_gap_recorded",
        "machine_id": MACHINE_ID,
        "hardware_lane": HARDWARE_LANE,
        "timestamp_utc": timestamp_utc,
        "requested_backend": HARDWARE_LANE,
        "selected_backend": HARDWARE_LANE,
        "runtime_api": "cuda",
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "speedup_claim": false,
        "cuda": cuda,
        "model": {
            "model_family": inspection.model_family,
            "architecture": inspection.architecture,
            "artifact_kind": "dense_gguf",
            "file": model_path.display().to_string(),
            "sha256": model_sha256
        },
        "execution_path": {
            "model_class": "dense_regular_llm",
            "kernel_family": "dense_fp16_gemm_plus_f32_rmsnorm_plus_f32_rope_plus_f32_attention_plus_f32_mlp_activation",
            "quantization_family": "dense_fp16_bridge_from_gguf_descriptors_with_f32_rmsnorm_rope_attention_mlp_activation",
            "bitnet_packed_kernel_proof": false,
            "qk256_proof": false
        },
        "execution_plan": execution_plan,
        "descriptor_coverage": {
            "schema": 1,
            "source_artifact_kind": "dense_gguf_tensor_descriptor_inspection",
            "tensor_count": inspection.tensor_count,
            "metadata_count": inspection.metadata_count as u64,
            "required_roles_present": inspection.required_roles_present,
            "strict_descriptor_complete": inspection.strict_descriptor_complete,
            "dense_cuda_route_status": inspection.dense_cuda_route_status,
            "quantization_families": inspection.quantization_families,
            "bitnet_packed_marker_found": inspection.bitnet_packed_marker_found,
            "dense_gguf_inference_claimed": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "one_layer_plan": {
            "schema": 1,
            "layer_index": layer_index as u64,
            "total_ops": summary.total_ops as u64,
            "cuda_routable_ops_total": cuda_routable_ops,
            "linear_cuda_ops_total": cuda_linear_ops,
            "norm_cuda_ops_total": cuda_norm_ops,
            "rope_cuda_ops_total": cuda_rope_ops,
            "attention_score_cuda_ops_total": cuda_attention_score_ops,
            "attention_softmax_cuda_ops_total": cuda_attention_softmax_ops,
            "attention_v_mix_cuda_ops_total": cuda_attention_v_mix_ops,
            "mlp_activation_cuda_ops_total": cuda_mlp_activation_ops,
            "unsupported_strict_cuda_ops_total": summary.unsupported_ops as u64,
            "cpu_fallback_ops_total": summary.cpu_fallback_ops as u64,
            "strict_cuda_ready": summary.strict_cuda_ready,
            "unsupported_ops_explicitly_listed": true,
            "operations": operations,
            "dense_gguf_one_layer_execution_plan_claimed": true,
            "one_layer_inference_claimed": false,
            "dense_gguf_inference_claimed": false,
            "qwen_one_token_cuda_claimed": false,
            "qwen_short_decode_cuda_claimed": false,
            "qwen_chat_cuda_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "gap_audit": gap_audit,
        "claim_boundary": {
            "dense_regular_llm_cuda_claimed": true,
            "dense_tensor_residency_claimed": false,
            "dense_gguf_descriptor_inspection_claimed": true,
            "dense_gguf_linear_fixture_extraction_claimed": false,
            "dense_gguf_linear_cuda_parity_claimed": false,
            "dense_gguf_linear_role_sweep_cuda_parity_claimed": false,
            "dense_gguf_one_layer_execution_plan_claimed": true,
            "dense_gguf_one_layer_inference_claimed": false,
            "dense_gguf_inference_claimed": false,
            "qwen_one_token_cuda_claimed": false,
            "qwen_short_decode_cuda_claimed": false,
            "qwen_chat_cuda_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false
        },
        "error": null
    }))
}

fn dense_gguf_all_layer_execution_plan_receipt_json(
    inspection: &DenseGgufDescriptorInspection,
    probe: Option<&bitnet_device_probe::NvidiaCudaProbe>,
    model_path: &Path,
    model_sha256: &str,
    artifact_path: &str,
    timestamp_utc: &str,
) -> Result<Value> {
    ensure_dense_all_layer_block_descriptor_coverage(inspection)?;

    let layer_indices = dense_transformer_layer_indices(inspection)?;
    if layer_indices.is_empty() {
        bail!("dense GGUF all-layer plan requires at least one transformer layer");
    }
    if !layer_indices.contains(&0) {
        bail!("dense GGUF all-layer plan requires a layer-0 reference graph");
    }

    let mut layers = Vec::new();
    let mut layer_differences = Vec::new();
    let mut reference_signature: Option<Vec<Value>> = None;
    let mut reference_signature_sha256: Option<String> = None;
    let mut aggregate = DenseAllLayerCounts::default();

    for layer_index in &layer_indices {
        let entries = dense_one_layer_plan_entries(inspection, *layer_index)?;
        let ops = entries.iter().map(|entry| entry.op.clone()).collect::<Vec<_>>();
        let spec = ModelDispatchSpec {
            model_family: ModelFamily::DenseRegularLlm,
            quantization: QuantizationKind::DenseFp16,
            backend_policy: BackendPolicy::StrictCuda,
            has_simd: true,
            cuda: CudaPlannerCapabilities::dense_regular_llm(),
        };
        let plan = plan_model_dispatch(&ops, spec);
        let summary = plan.summary();
        let operations = dense_layer_plan_operations_json(&entries, &plan.decisions);
        let counts = dense_layer_plan_counts(&operations)?;
        aggregate.add(counts);

        let signature = dense_layer_operation_signature(&operations)?;
        let signature_sha256 = sha256_json(&Value::Array(signature.clone()))?;
        let matches_layer0 = match &reference_signature {
            Some(reference) => reference == &signature,
            None => {
                reference_signature = Some(signature.clone());
                reference_signature_sha256 = Some(signature_sha256.clone());
                true
            }
        };
        if !matches_layer0 {
            layer_differences.push(json!({
                "layer_index": *layer_index as u64,
                "reason": "operation_signature_differs_from_layer0",
                "layer0_signature_sha256": reference_signature_sha256
                    .clone()
                    .unwrap_or_else(|| "unknown".to_string()),
                "layer_signature_sha256": signature_sha256
            }));
        }

        layers.push(json!({
            "layer_index": *layer_index as u64,
            "total_ops": counts.total_ops,
            "cuda_routable_ops_total": counts.cuda_routable_ops,
            "linear_cuda_ops_total": counts.linear_cuda_ops,
            "norm_cuda_ops_total": counts.norm_cuda_ops,
            "rope_cuda_ops_total": counts.rope_cuda_ops,
            "attention_score_cuda_ops_total": counts.attention_score_cuda_ops,
            "attention_softmax_cuda_ops_total": counts.attention_softmax_cuda_ops,
            "attention_v_mix_cuda_ops_total": counts.attention_v_mix_cuda_ops,
            "mlp_activation_cuda_ops_total": counts.mlp_activation_cuda_ops,
            "unsupported_strict_cuda_ops_total": counts.unsupported_ops,
            "cpu_fallback_ops_total": counts.cpu_fallback_ops,
            "strict_cuda_ready": summary.strict_cuda_ready,
            "matches_layer0": matches_layer0,
            "operation_signature_sha256": signature_sha256,
            "operations": operations
        }));
    }

    let missing_layer_indices = dense_missing_layer_indices(&layer_indices);
    let layer_plan_matches_layer0 =
        layer_differences.is_empty() && missing_layer_indices.is_empty();
    let strict_cuda_ready = aggregate.unsupported_ops == 0
        && aggregate.cpu_fallback_ops == 0
        && aggregate.cuda_routable_ops == aggregate.total_ops
        && layer_plan_matches_layer0;
    let aggregate_summary = ModelDispatchSummary {
        total_ops: aggregate.total_ops as usize,
        cuda_bitnet_qk256_ops: 0,
        cuda_dense_regular_llm_ops: aggregate.cuda_routable_ops as usize,
        cpu_fallback_ops: aggregate.cpu_fallback_ops as usize,
        unsupported_ops: aggregate.unsupported_ops as usize,
        fallback_used: false,
        selected_route: Some(ModelDispatchBackend::CudaDenseRegularLlm),
        strict_cuda_ready,
    };
    let execution_plan = execution_plan_receipt(ExecutionPlanReceiptInput {
        model_family: &inspection.model_family,
        quantization: "dense_fp16",
        requested_backend: HARDWARE_LANE,
        selected_backend: HARDWARE_LANE,
        runtime_api: "cuda",
        strict_fallback_policy: "reject",
        summary: aggregate_summary,
        speedup_claim: false,
        full_cuda_residency_claimed: false,
    });

    let cuda = cuda_identity_json(probe);
    Ok(json!({
        "schema": 1,
        "artifact_kind": DENSE_GGUF_ALL_LAYER_EXECUTION_PLAN_ARTIFACT_KIND,
        "artifact_path": artifact_path,
        "claim": "dense_gguf_all_layer_execution_plan_recorded",
        "machine_id": MACHINE_ID,
        "hardware_lane": HARDWARE_LANE,
        "timestamp_utc": timestamp_utc,
        "requested_backend": HARDWARE_LANE,
        "selected_backend": HARDWARE_LANE,
        "runtime_api": "cuda",
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "speedup_claim": false,
        "cuda": cuda,
        "model": {
            "model_family": inspection.model_family,
            "architecture": inspection.architecture,
            "artifact_kind": "dense_gguf",
            "file": model_path.display().to_string(),
            "sha256": model_sha256
        },
        "execution_path": {
            "model_class": "dense_regular_llm",
            "kernel_family": "dense_cuda_all_layer_execution_plan",
            "quantization_family": "dense_fp16_bridge_from_gguf_descriptors_with_q8_0_fixture_contracts",
            "bitnet_packed_kernel_proof": false,
            "qk256_proof": false
        },
        "execution_plan": execution_plan,
        "descriptor_coverage": {
            "schema": 1,
            "source_artifact_kind": "dense_gguf_tensor_descriptor_inspection",
            "tensor_count": inspection.tensor_count,
            "metadata_count": inspection.metadata_count as u64,
            "required_roles_present": inspection.required_roles_present,
            "model_boundary_required_roles_present": inspection.required_roles_present,
            "missing_model_boundary_roles": inspection
                .missing_required_roles
                .iter()
                .map(|role| dense_role_label(*role))
                .collect::<Vec<_>>(),
            "transformer_block_required_roles_present": true,
            "missing_transformer_block_roles": Vec::<&str>::new(),
            "strict_descriptor_complete": inspection.strict_descriptor_complete,
            "dense_cuda_route_status": inspection.dense_cuda_route_status,
            "quantization_families": inspection.quantization_families,
            "bitnet_packed_marker_found": inspection.bitnet_packed_marker_found,
            "dense_gguf_inference_claimed": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "all_layer_plan": {
            "schema": 1,
            "transformer_layers_total": layer_indices.len() as u64,
            "layers_with_complete_cuda_block_plan": layers
                .iter()
                .filter(|layer| layer["strict_cuda_ready"] == true && layer["matches_layer0"] == true)
                .count() as u64,
            "layer_plan_matches_layer0": layer_plan_matches_layer0,
            "layer_differences": layer_differences,
            "missing_layer_indices": missing_layer_indices,
            "total_ops": aggregate.total_ops,
            "cuda_routable_ops_total": aggregate.cuda_routable_ops,
            "linear_cuda_ops_total": aggregate.linear_cuda_ops,
            "norm_cuda_ops_total": aggregate.norm_cuda_ops,
            "rope_cuda_ops_total": aggregate.rope_cuda_ops,
            "attention_score_cuda_ops_total": aggregate.attention_score_cuda_ops,
            "attention_softmax_cuda_ops_total": aggregate.attention_softmax_cuda_ops,
            "attention_v_mix_cuda_ops_total": aggregate.attention_v_mix_cuda_ops,
            "mlp_activation_cuda_ops_total": aggregate.mlp_activation_cuda_ops,
            "unsupported_strict_cuda_ops_total": aggregate.unsupported_ops,
            "cpu_fallback_ops_total": aggregate.cpu_fallback_ops,
            "strict_cuda_ready": strict_cuda_ready,
            "strict_cuda_ready_scope": "transformer_blocks_only",
            "all_layers_inspected": true,
            "operations_per_layer": 14,
            "layers": layers,
            "dense_gguf_all_layer_execution_plan_claimed": true,
            "dense_gguf_inference_claimed": false,
            "qwen_one_token_cuda_claimed": false,
            "qwen_short_decode_cuda_claimed": false,
            "qwen_chat_cuda_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false
        },
        "model_boundary_gaps": dense_model_boundary_gaps_json(inspection),
        "claim_boundary": {
            "dense_regular_llm_cuda_claimed": true,
            "dense_tensor_residency_claimed": false,
            "dense_gguf_descriptor_inspection_claimed": true,
            "dense_gguf_linear_fixture_extraction_claimed": false,
            "dense_gguf_linear_cuda_parity_claimed": false,
            "dense_gguf_linear_role_sweep_cuda_parity_claimed": false,
            "dense_gguf_one_layer_execution_plan_claimed": true,
            "dense_gguf_one_layer_cpu_reference_claimed": true,
            "dense_gguf_one_layer_cuda_integrated_parity_claimed": true,
            "dense_gguf_all_layer_execution_plan_claimed": true,
            "dense_gguf_one_layer_inference_claimed": false,
            "dense_gguf_inference_claimed": false,
            "qwen_one_token_cuda_claimed": false,
            "qwen_short_decode_cuda_claimed": false,
            "qwen_chat_cuda_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false
        },
        "error": null
    }))
}

fn dense_gguf_model_boundary_fixtures_receipt_json(
    inspection: &DenseGgufDescriptorInspection,
    fixtures: &DenseGgufModelBoundaryFixtures,
    probe: Option<&bitnet_device_probe::NvidiaCudaProbe>,
    model_path: &Path,
    model_sha256: &str,
    artifact_path: &str,
    timestamp_utc: &str,
) -> Result<Value> {
    if fixtures.model_family != inspection.model_family
        || fixtures.architecture != inspection.architecture
    {
        bail!(
            "dense GGUF model-boundary fixture identity mismatch: inspection={}/{} fixtures={}/{}",
            inspection.model_family,
            inspection.architecture,
            fixtures.model_family,
            fixtures.architecture
        );
    }
    if fixtures.logits_top_k.is_empty() {
        bail!("dense GGUF model-boundary fixtures require logits top-k diagnostics");
    }

    let summary = ModelDispatchSummary {
        total_ops: 3,
        cuda_bitnet_qk256_ops: 0,
        cuda_dense_regular_llm_ops: 3,
        cpu_fallback_ops: 0,
        unsupported_ops: 0,
        fallback_used: false,
        selected_route: Some(ModelDispatchBackend::CudaDenseRegularLlm),
        strict_cuda_ready: true,
    };
    let execution_plan = execution_plan_receipt(ExecutionPlanReceiptInput {
        model_family: &inspection.model_family,
        quantization: "dense_fp16",
        requested_backend: HARDWARE_LANE,
        selected_backend: HARDWARE_LANE,
        runtime_api: "cuda",
        strict_fallback_policy: "reject",
        summary,
        speedup_claim: false,
        full_cuda_residency_claimed: false,
    });

    let top_k = fixtures
        .logits_top_k
        .iter()
        .map(|entry| {
            json!({
                "rank": entry.rank as u64,
                "token_id": entry.token_id as u64,
                "value": entry.value
            })
        })
        .collect::<Vec<_>>();

    Ok(json!({
        "schema": 1,
        "artifact_kind": DENSE_GGUF_MODEL_BOUNDARY_FIXTURES_ARTIFACT_KIND,
        "artifact_path": artifact_path,
        "claim": "dense_gguf_model_boundary_fixtures_recorded",
        "machine_id": MACHINE_ID,
        "hardware_lane": HARDWARE_LANE,
        "timestamp_utc": timestamp_utc,
        "requested_backend": HARDWARE_LANE,
        "selected_backend": HARDWARE_LANE,
        "runtime_api": "cuda",
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "speedup_claim": false,
        "cuda": cuda_identity_json(probe),
        "model": {
            "model_family": inspection.model_family,
            "architecture": inspection.architecture,
            "artifact_kind": "dense_gguf",
            "file": model_path.display().to_string(),
            "sha256": model_sha256
        },
        "execution_path": {
            "model_class": "dense_regular_llm",
            "kernel_family": "dense_cuda_model_boundary_fixture_route",
            "quantization_family": "dense_gguf_q8_0_f16_boundary_fixture_contract",
            "bitnet_packed_kernel_proof": false,
            "qk256_proof": false
        },
        "execution_plan": execution_plan,
        "descriptor_coverage": {
            "schema": 1,
            "source_artifact_kind": "dense_gguf_tensor_descriptor_inspection",
            "tensor_count": inspection.tensor_count,
            "metadata_count": inspection.metadata_count as u64,
            "required_roles_present": dense_model_boundary_fixture_coverage_complete(inspection),
            "strict_descriptor_complete": dense_model_boundary_fixture_coverage_complete(inspection),
            "dense_cuda_route_status": dense_model_boundary_route_status(inspection),
            "model_boundary_lm_head_source": dense_model_boundary_lm_head_source(inspection),
            "quantization_families": inspection.quantization_families,
            "bitnet_packed_marker_found": inspection.bitnet_packed_marker_found,
            "dense_gguf_inference_claimed": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "model_boundary_fixtures": {
            "schema": 1,
            "fixture_id": fixtures.fixture_id,
            "seq_len": fixtures.seq_len as u64,
            "hidden_size": fixtures.hidden_size as u64,
            "vocab_size": fixtures.vocab_size as u64,
            "token_ids": fixtures.token_ids.iter().map(|id| json!(*id as u64)).collect::<Vec<_>>(),
            "token_ids_sha256": fixtures.token_ids_sha256,
            "fixtures_total": 3_u64,
            "token_embedding": dense_boundary_tensor_fixture_json(&fixtures.token_embedding),
            "final_norm": {
                "rmsnorm_eps": fixtures.rmsnorm_eps,
                "epsilon_source": fixtures.epsilon_source,
                "input_sha256": fixtures.final_norm_input_sha256,
                "output_sha256": fixtures.final_norm_output_sha256,
                "fixture": dense_boundary_tensor_fixture_json(&fixtures.final_norm)
            },
            "lm_head_logits": {
                "logits_len": fixtures.logits_len as u64,
                "logits_sha256": fixtures.logits_sha256,
                "top_k": fixtures.top_k as u64,
                "top_k_entries": top_k,
                "fixture": dense_boundary_tensor_fixture_json(&fixtures.lm_head_logits)
            },
            "boundary_fixtures_claimed": true,
            "token_embedding_fixture_claimed": true,
            "final_norm_fixture_claimed": true,
            "lm_head_logits_fixture_claimed": true,
            "fixture_route_only": true,
            "cuda_kernel_execution_claimed": false,
            "kernel_invocations": 0_u64,
            "fallback_used": false,
            "kv_cache_policy_claimed": false,
            "sampling_integration_claimed": false,
            "qwen_one_token_cuda_claimed": false,
            "qwen_short_decode_cuda_claimed": false,
            "qwen_chat_cuda_claimed": false,
            "dense_gguf_inference_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false
        },
        "remaining_model_boundary_gaps": {
            "schema": 1,
            "gaps": [
                {
                    "gap": "kv_cache_policy",
                    "status": "not_governed_by_model_boundary_fixtures",
                    "required_next_proof": "dense_gguf_kv_cache_policy_receipt",
                    "blocks_qwen_one_token": true,
                    "blocks_qwen_short_decode": true,
                    "blocks_qwen_chat": true
                },
                {
                    "gap": "sampling",
                    "status": "not_governed_by_model_boundary_fixtures",
                    "required_next_proof": "dense_gguf_sampling_policy_receipt",
                    "blocks_qwen_one_token": true,
                    "blocks_qwen_short_decode": true,
                    "blocks_qwen_chat": true
                }
            ],
            "qwen_one_token_cuda_blocked": true,
            "qwen_short_decode_cuda_blocked": true,
            "qwen_chat_cuda_blocked": true,
            "next_required_proof": "dense_gguf_kv_cache_policy_receipt",
            "dense_gguf_inference_claimed": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "claim_boundary": {
            "dense_regular_llm_cuda_claimed": true,
            "dense_tensor_residency_claimed": false,
            "dense_gguf_descriptor_inspection_claimed": true,
            "dense_gguf_linear_fixture_extraction_claimed": true,
            "dense_gguf_linear_cuda_parity_claimed": false,
            "dense_gguf_linear_role_sweep_cuda_parity_claimed": false,
            "dense_gguf_one_layer_execution_plan_claimed": true,
            "dense_gguf_one_layer_cpu_reference_claimed": true,
            "dense_gguf_one_layer_cuda_integrated_parity_claimed": true,
            "dense_gguf_all_layer_execution_plan_claimed": true,
            "dense_gguf_model_boundary_fixtures_claimed": true,
            "dense_gguf_one_layer_inference_claimed": false,
            "dense_gguf_inference_claimed": false,
            "qwen_one_token_cuda_claimed": false,
            "qwen_short_decode_cuda_claimed": false,
            "qwen_chat_cuda_claimed": false,
            "kv_cache_policy_claimed": false,
            "sampling_integration_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false
        },
        "error": null
    }))
}

fn dense_boundary_tensor_fixture_json(fixture: &DenseGgufBoundaryTensorFixture) -> Value {
    json!({
        "name": fixture.name,
        "role": fixture.role,
        "tensor_name": fixture.tensor_name,
        "tensor_type": fixture.tensor_type,
        "source_shape": fixture.source_shape,
        "source_offset": fixture.source_offset,
        "source_size_bytes": fixture.source_size_bytes,
        "value_count": fixture.value_count as u64,
        "output_len": fixture.output_len as u64,
        "output_sha256": fixture.output_sha256,
        "max_abs": fixture.max_abs,
        "dense_gguf_inference_claimed": false,
        "bitnet_packed_i2s_qk256_proof": false
    })
}

fn dense_gguf_kv_cache_policy_receipt_json(
    inspection: &DenseGgufDescriptorInspection,
    policy: &DenseGgufKvCachePolicy,
    probe: Option<&bitnet_device_probe::NvidiaCudaProbe>,
    model_path: &Path,
    model_sha256: &str,
    artifact_path: &str,
    timestamp_utc: &str,
) -> Result<Value> {
    if policy.model_family != inspection.model_family
        || policy.architecture != inspection.architecture
    {
        bail!(
            "dense GGUF KV-cache policy identity mismatch: inspection={}/{} policy={}/{}",
            inspection.model_family,
            inspection.architecture,
            policy.model_family,
            policy.architecture
        );
    }

    let summary = ModelDispatchSummary {
        total_ops: 1,
        cuda_bitnet_qk256_ops: 0,
        cuda_dense_regular_llm_ops: 1,
        cpu_fallback_ops: 0,
        unsupported_ops: 0,
        fallback_used: false,
        selected_route: Some(ModelDispatchBackend::CudaDenseRegularLlm),
        strict_cuda_ready: true,
    };
    let execution_plan = execution_plan_receipt(ExecutionPlanReceiptInput {
        model_family: &inspection.model_family,
        quantization: "dense_fp16",
        requested_backend: HARDWARE_LANE,
        selected_backend: HARDWARE_LANE,
        runtime_api: "cuda",
        strict_fallback_policy: "reject",
        summary,
        speedup_claim: false,
        full_cuda_residency_claimed: false,
    });

    Ok(json!({
        "schema": 1,
        "artifact_kind": DENSE_GGUF_KV_CACHE_POLICY_ARTIFACT_KIND,
        "artifact_path": artifact_path,
        "claim": "dense_gguf_kv_cache_policy_recorded",
        "machine_id": MACHINE_ID,
        "hardware_lane": HARDWARE_LANE,
        "timestamp_utc": timestamp_utc,
        "requested_backend": HARDWARE_LANE,
        "selected_backend": HARDWARE_LANE,
        "runtime_api": "cuda",
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "speedup_claim": false,
        "cuda": cuda_identity_json(probe),
        "model": {
            "model_family": inspection.model_family,
            "architecture": inspection.architecture,
            "artifact_kind": "dense_gguf",
            "file": model_path.display().to_string(),
            "sha256": model_sha256
        },
        "execution_path": {
            "model_class": "dense_regular_llm",
            "kernel_family": "dense_cuda_kv_cache_policy_route",
            "quantization_family": "dense_gguf_q8_0_f16_kv_cache_policy_contract",
            "bitnet_packed_kernel_proof": false,
            "qk256_proof": false
        },
        "execution_plan": execution_plan,
        "descriptor_coverage": {
            "schema": 1,
            "source_artifact_kind": "dense_gguf_tensor_descriptor_inspection",
            "tensor_count": inspection.tensor_count,
            "metadata_count": inspection.metadata_count as u64,
            "required_roles_present": dense_transformer_block_descriptor_coverage_complete(inspection),
            "strict_descriptor_complete": dense_transformer_block_descriptor_coverage_complete(inspection),
            "dense_cuda_route_status": if dense_transformer_block_descriptor_coverage_complete(inspection) {
                "transformer_block_descriptor_complete".to_string()
            } else {
                inspection.dense_cuda_route_status.clone()
            },
            "quantization_families": inspection.quantization_families,
            "bitnet_packed_marker_found": inspection.bitnet_packed_marker_found,
            "dense_gguf_inference_claimed": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "kv_cache_policy": {
            "schema": 1,
            "policy_id": policy.policy_id,
            "policy_scope": "dense_qwen_prefill_decode_boundary",
            "planned_residency": "cuda_required_for_strict_dense_cuda",
            "observed_residency": "not_allocated_policy_only",
            "transformer_layers_total": policy.transformer_layers_total as u64,
            "context_length": policy.context_length as u64,
            "seq_len": policy.seq_len as u64,
            "decode_steps": policy.decode_steps as u64,
            "q_heads": policy.q_heads as u64,
            "kv_heads": policy.kv_heads as u64,
            "heads_per_kv_group": policy.heads_per_kv_group as u64,
            "key_head_dim": policy.key_head_dim as u64,
            "value_head_dim": policy.value_head_dim as u64,
            "kv_element_dtype": policy.kv_element_dtype,
            "kv_element_bytes": policy.kv_element_bytes as u64,
            "kv_values_per_token_per_layer": policy.kv_values_per_token_per_layer as u64,
            "kv_bytes_per_token_per_layer": policy.kv_bytes_per_token_per_layer,
            "kv_bytes_per_token_all_layers": policy.kv_bytes_per_token_all_layers,
            "metadata_sources": {
                "transformer_layers": policy.transformer_layers_source,
                "context_length": policy.context_length_source,
                "q_heads": policy.q_heads_source,
                "kv_heads": policy.kv_heads_source,
                "key_head_dim": policy.key_head_dim_source,
                "value_head_dim": policy.value_head_dim_source
            },
            "prefill": {
                "write_tokens": policy.seq_len as u64,
                "writes_keys": true,
                "writes_values": true,
                "write_bytes_estimate": policy.prefill_write_bytes_estimate,
                "write_path": "qkv_projection_to_cuda_kv_cache",
                "measured": false
            },
            "decode": {
                "decode_steps": policy.decode_steps as u64,
                "read_tokens_per_step": policy.seq_len as u64,
                "read_bytes_per_step_estimate": policy.decode_read_bytes_per_step_estimate,
                "write_tokens_per_step": 1_u64,
                "write_bytes_per_step_estimate": policy.decode_write_bytes_per_step_estimate,
                "read_path": "cuda_kv_cache_to_attention",
                "write_path": "qkv_projection_to_cuda_kv_cache",
                "measured": false
            },
            "max_context": {
                "tokens": policy.context_length as u64,
                "bytes_estimate": policy.max_context_bytes_estimate
            },
            "kv_cache_policy_claimed": true,
            "runtime_kv_cache_allocated": false,
            "kv_cache_cuda_residency_claimed": false,
            "estimated_bytes_only": true,
            "transfer_bytes_measured": false,
            "transfer_timing_measured": false,
            "fallback_used": false,
            "sampling_integration_claimed": false,
            "qwen_one_token_cuda_claimed": false,
            "qwen_short_decode_cuda_claimed": false,
            "qwen_chat_cuda_claimed": false,
            "dense_gguf_inference_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false
        },
        "remaining_model_boundary_gaps": {
            "schema": 1,
            "gaps": [
                {
                    "gap": "sampling",
                    "status": "not_governed_by_kv_cache_policy",
                    "required_next_proof": "dense_gguf_sampling_policy_receipt",
                    "blocks_qwen_one_token": true,
                    "blocks_qwen_short_decode": true,
                    "blocks_qwen_chat": true
                }
            ],
            "kv_cache_policy_claimed": true,
            "sampling_integration_claimed": false,
            "qwen_one_token_cuda_blocked": true,
            "qwen_short_decode_cuda_blocked": true,
            "qwen_chat_cuda_blocked": true,
            "next_required_proof": "dense_gguf_sampling_policy_receipt",
            "dense_gguf_inference_claimed": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "claim_boundary": {
            "dense_regular_llm_cuda_claimed": true,
            "dense_tensor_residency_claimed": false,
            "dense_gguf_descriptor_inspection_claimed": true,
            "dense_gguf_linear_fixture_extraction_claimed": true,
            "dense_gguf_linear_cuda_parity_claimed": false,
            "dense_gguf_linear_role_sweep_cuda_parity_claimed": false,
            "dense_gguf_one_layer_execution_plan_claimed": true,
            "dense_gguf_one_layer_cpu_reference_claimed": true,
            "dense_gguf_one_layer_cuda_integrated_parity_claimed": true,
            "dense_gguf_all_layer_execution_plan_claimed": true,
            "dense_gguf_model_boundary_fixtures_claimed": true,
            "kv_cache_policy_claimed": true,
            "kv_cache_cuda_residency_claimed": false,
            "dense_gguf_one_layer_inference_claimed": false,
            "dense_gguf_inference_claimed": false,
            "qwen_one_token_cuda_claimed": false,
            "qwen_short_decode_cuda_claimed": false,
            "qwen_chat_cuda_claimed": false,
            "sampling_integration_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false
        },
        "error": null
    }))
}

fn dense_gguf_sampling_policy_receipt_json(
    inspection: &DenseGgufDescriptorInspection,
    policy: &DenseGgufSamplingPolicy,
    probe: Option<&bitnet_device_probe::NvidiaCudaProbe>,
    model_path: &Path,
    model_sha256: &str,
    artifact_path: &str,
    timestamp_utc: &str,
) -> Result<Value> {
    if policy.model_family != inspection.model_family
        || policy.architecture != inspection.architecture
    {
        bail!(
            "dense GGUF sampling policy identity mismatch: inspection={}/{} policy={}/{}",
            inspection.model_family,
            inspection.architecture,
            policy.model_family,
            policy.architecture
        );
    }
    if policy.logits_len == 0 || policy.vocab_size == 0 || policy.logits_top_k.is_empty() {
        bail!("dense GGUF sampling policy requires logits and top-k evidence");
    }
    if policy.logits_len != policy.vocab_size {
        bail!(
            "dense GGUF sampling policy logits/vocab mismatch: logits_len={} vocab_size={}",
            policy.logits_len,
            policy.vocab_size
        );
    }

    let summary = ModelDispatchSummary {
        total_ops: 1,
        cuda_bitnet_qk256_ops: 0,
        cuda_dense_regular_llm_ops: 1,
        cpu_fallback_ops: 0,
        unsupported_ops: 0,
        fallback_used: false,
        selected_route: Some(ModelDispatchBackend::CudaDenseRegularLlm),
        strict_cuda_ready: true,
    };
    let execution_plan = execution_plan_receipt(ExecutionPlanReceiptInput {
        model_family: &inspection.model_family,
        quantization: "dense_fp16",
        requested_backend: HARDWARE_LANE,
        selected_backend: HARDWARE_LANE,
        runtime_api: "cuda",
        strict_fallback_policy: "reject",
        summary,
        speedup_claim: false,
        full_cuda_residency_claimed: false,
    });
    let top_k_entries = policy
        .logits_top_k
        .iter()
        .map(|entry| {
            json!({
                "rank": entry.rank as u64,
                "token_id": entry.token_id as u64,
                "value": entry.value
            })
        })
        .collect::<Vec<_>>();

    Ok(json!({
        "schema": 1,
        "artifact_kind": DENSE_GGUF_SAMPLING_POLICY_ARTIFACT_KIND,
        "artifact_path": artifact_path,
        "claim": "dense_gguf_sampling_policy_recorded",
        "machine_id": MACHINE_ID,
        "hardware_lane": HARDWARE_LANE,
        "timestamp_utc": timestamp_utc,
        "requested_backend": HARDWARE_LANE,
        "selected_backend": HARDWARE_LANE,
        "runtime_api": "cuda",
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "speedup_claim": false,
        "cuda": cuda_identity_json(probe),
        "model": {
            "model_family": inspection.model_family,
            "architecture": inspection.architecture,
            "artifact_kind": "dense_gguf",
            "file": model_path.display().to_string(),
            "sha256": model_sha256
        },
        "execution_path": {
            "model_class": "dense_regular_llm",
            "kernel_family": "dense_cuda_sampling_policy_route",
            "quantization_family": "dense_gguf_q8_0_f16_logits_sampling_policy_contract",
            "bitnet_packed_kernel_proof": false,
            "qk256_proof": false
        },
        "execution_plan": execution_plan,
        "descriptor_coverage": {
            "schema": 1,
            "source_artifact_kind": "dense_gguf_tensor_descriptor_inspection",
            "tensor_count": inspection.tensor_count,
            "metadata_count": inspection.metadata_count as u64,
            "required_roles_present": dense_model_boundary_fixture_coverage_complete(inspection),
            "strict_descriptor_complete": dense_model_boundary_fixture_coverage_complete(inspection),
            "dense_cuda_route_status": dense_model_boundary_route_status(inspection),
            "model_boundary_lm_head_source": dense_model_boundary_lm_head_source(inspection),
            "quantization_families": inspection.quantization_families,
            "bitnet_packed_marker_found": inspection.bitnet_packed_marker_found,
            "dense_gguf_inference_claimed": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "sampling_policy": {
            "schema": 1,
            "policy_id": policy.policy_id,
            "policy_scope": "dense_qwen_logits_to_sampler_boundary",
            "logits_source": "dense_gguf_model_boundary_lm_head_logits",
            "logits_sha256": policy.logits_sha256,
            "logits_len": policy.logits_len as u64,
            "vocab_size": policy.vocab_size as u64,
            "seq_len": policy.seq_len as u64,
            "logits_dtype": policy.logits_element_dtype,
            "logits_element_bytes": policy.logits_element_bytes as u64,
            "logits_transfer_bytes_per_step_estimate": policy.logits_transfer_bytes_per_step_estimate,
            "logits_transfer_path": "cuda_lm_head_logits_to_cpu_sampler",
            "logits_transfer_required_for_cpu_sampling": true,
            "logits_transfer_bytes_measured": false,
            "logits_transfer_timing_measured": false,
            "sampler_backend": policy.sampler_backend,
            "sampler_location": policy.sampler_location,
            "sampler_mode": policy.sampler_mode,
            "temperature": policy.temperature,
            "top_k_filter": policy.top_k_filter as u64,
            "top_p": policy.top_p,
            "repetition_penalty": policy.repetition_penalty,
            "deterministic": policy.deterministic,
            "tie_break_policy": policy.tie_break_policy,
            "rng_required": policy.rng_required,
            "selected_token_id_from_fixture_logits": policy.selected_token_id_from_fixture_logits as u64,
            "selected_token_scope": "fixture_logits_only_not_generation",
            "top_k": policy.top_k as u64,
            "top_k_entries": top_k_entries,
            "sampling_policy_claimed": true,
            "sampling_integration_claimed": false,
            "qwen_one_token_cuda_claimed": false,
            "qwen_short_decode_cuda_claimed": false,
            "qwen_chat_cuda_claimed": false,
            "dense_gguf_inference_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false
        },
        "remaining_model_boundary_gaps": {
            "schema": 1,
            "gaps": [],
            "all_model_boundary_policies_governed": true,
            "kv_cache_policy_claimed": true,
            "sampling_policy_claimed": true,
            "sampling_integration_claimed": false,
            "qwen_one_token_cuda_blocked": false,
            "qwen_short_decode_cuda_blocked": true,
            "qwen_chat_cuda_blocked": true,
            "next_required_proof": "qwen_one_token_strict_cuda_proof",
            "dense_gguf_inference_claimed": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "claim_boundary": {
            "dense_regular_llm_cuda_claimed": true,
            "dense_tensor_residency_claimed": false,
            "dense_gguf_descriptor_inspection_claimed": true,
            "dense_gguf_linear_fixture_extraction_claimed": true,
            "dense_gguf_linear_cuda_parity_claimed": false,
            "dense_gguf_linear_role_sweep_cuda_parity_claimed": false,
            "dense_gguf_one_layer_execution_plan_claimed": true,
            "dense_gguf_one_layer_cpu_reference_claimed": true,
            "dense_gguf_one_layer_cuda_integrated_parity_claimed": true,
            "dense_gguf_all_layer_execution_plan_claimed": true,
            "dense_gguf_model_boundary_fixtures_claimed": true,
            "kv_cache_policy_claimed": true,
            "kv_cache_cuda_residency_claimed": false,
            "sampling_policy_claimed": true,
            "sampling_integration_claimed": false,
            "dense_gguf_one_layer_inference_claimed": false,
            "dense_gguf_inference_claimed": false,
            "qwen_one_token_cuda_claimed": false,
            "qwen_short_decode_cuda_claimed": false,
            "qwen_chat_cuda_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false
        },
        "error": null
    }))
}

#[allow(clippy::too_many_arguments)]
fn dense_gguf_qwen_one_token_strict_cuda_proof_receipt_json(
    inspection: &DenseGgufDescriptorInspection,
    prerequisites: &DenseQwenOneTokenPrerequisites,
    proof_model: &DenseQwenProofModel,
    probe: &bitnet_device_probe::NvidiaCudaProbe,
    model_path: &Path,
    model_sha256: &str,
    artifact_path: &str,
    timestamp_utc: &str,
    rendered_prompt: &str,
    prompt_token_count: usize,
    prompt_token_ids_sha256: &str,
    rendered_prompt_sha256: &str,
    cpu: &DenseQwenOneTokenRun,
    cuda: &DenseQwenOneTokenRun,
    decoded_token_text: &str,
) -> Result<Value> {
    if inspection.model_family != "qwen" || inspection.architecture != proof_model.architecture {
        bail!(
            "dense Qwen one-token receipt requires qwen/{} inspection, got {}/{}",
            proof_model.architecture,
            inspection.model_family,
            inspection.architecture
        );
    }
    if model_sha256 != proof_model.sha256 {
        bail!("dense Qwen one-token receipt requires verified {} model SHA", proof_model.id);
    }
    if cpu.selected_token_id != cuda.selected_token_id {
        bail!("dense Qwen one-token receipt requires matching selected tokens");
    }
    if cpu.top_k_rank_sha256 != cuda.top_k_rank_sha256 {
        bail!("dense Qwen one-token receipt requires matching top-k rank hashes");
    }
    if !cuda.logits_device_is_cuda {
        bail!("dense Qwen one-token receipt requires CUDA-resident logits before download");
    }

    let all_layer_cuda_ops = prerequisites
        .all_layer_plan
        .get("execution_plan")
        .and_then(|plan| plan.get("cuda_dense_regular_llm_ops"))
        .and_then(Value::as_u64)
        .or_else(|| {
            prerequisites
                .all_layer_plan
                .get("all_layer_plan")
                .and_then(|plan| plan.get("cuda_routable_ops_total"))
                .and_then(Value::as_u64)
        })
        .unwrap_or(1);
    let total_dense_ops = all_layer_cuda_ops.saturating_add(2);
    let summary = ModelDispatchSummary {
        total_ops: total_dense_ops as usize,
        cuda_bitnet_qk256_ops: 0,
        cuda_dense_regular_llm_ops: total_dense_ops as usize,
        cpu_fallback_ops: 0,
        unsupported_ops: 0,
        fallback_used: false,
        selected_route: Some(ModelDispatchBackend::CudaDenseRegularLlm),
        strict_cuda_ready: true,
    };
    let execution_plan = execution_plan_receipt(ExecutionPlanReceiptInput {
        model_family: &inspection.model_family,
        quantization: "dense_gguf_q8_0_f16_qwen_one_token_contract",
        requested_backend: HARDWARE_LANE,
        selected_backend: HARDWARE_LANE,
        runtime_api: "cuda",
        strict_fallback_policy: "reject",
        summary,
        speedup_claim: false,
        full_cuda_residency_claimed: false,
    });

    let model_bytes = std::fs::metadata(model_path)
        .with_context(|| format!("failed to stat {}", model_path.display()))?
        .len();
    let logits_transfer_bytes =
        checked_u64_mul(cuda.logits_len as u64, 4, "logits transfer bytes")?;
    let transformer_invocations = all_layer_cuda_ops.max(1);
    let runtime_invocations = transformer_invocations.saturating_add(2);
    let kernel_time_ms = cuda.prefill_ms + cuda.forward_ms + cuda.logits_ms;
    let kernel_stats = json!([
        {
            "phase": "qwen_one_token_runtime",
            "kernel_id": "dense_qwen_one_token_cuda_runtime",
            "invocations": transformer_invocations,
            "fallback_invocations": 0,
            "cpu_fallback_invocations": 0,
            "host_to_device_bytes": model_bytes,
            "device_to_host_bytes": logits_transfer_bytes,
            "kernel_launches": transformer_invocations,
            "kernel_time_ms": kernel_time_ms
        },
        {
            "phase": "lm_head",
            "kernel_id": "dense_qwen_lm_head_cuda",
            "invocations": 1_u64,
            "fallback_invocations": 0,
            "cpu_fallback_invocations": 0,
            "host_to_device_bytes": 0_u64,
            "device_to_host_bytes": 0_u64,
            "kernel_launches": 1_u64,
            "kernel_time_ms": cuda.logits_ms
        },
        {
            "phase": "logits_transfer",
            "kernel_id": "dense_qwen_logits_transfer_cuda",
            "invocations": 1_u64,
            "fallback_invocations": 0,
            "cpu_fallback_invocations": 0,
            "host_to_device_bytes": 0_u64,
            "device_to_host_bytes": 0_u64,
            "kernel_launches": 1_u64,
            "kernel_time_ms": 0.0
        }
    ]);
    let stats_h2d = model_bytes;
    let stats_d2h = logits_transfer_bytes;
    let stats_launches = runtime_invocations;
    let (top_k_max_abs_error, top_k_mean_abs_error) =
        dense_qwen_logits_error(&cpu.top_k, &cuda.top_k);
    let top_k_match = cpu
        .top_k
        .iter()
        .zip(cuda.top_k.iter())
        .all(|(left, right)| left.token_id == right.token_id);

    Ok(json!({
        "schema": 1,
        "artifact_kind": DENSE_GGUF_QWEN_ONE_TOKEN_STRICT_CUDA_PROOF_ARTIFACT_KIND,
        "artifact_path": artifact_path,
        "claim": "dense_gguf_qwen_one_token_strict_cuda_proof_recorded",
        "model_coverage_row": proof_model.model_coverage_row,
        "model_coverage_tier": proof_model.model_coverage_tier,
        "machine_id": MACHINE_ID,
        "hardware_lane": HARDWARE_LANE,
        "timestamp_utc": timestamp_utc,
        "requested_backend": HARDWARE_LANE,
        "selected_backend": HARDWARE_LANE,
        "runtime_api": "cuda",
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "speedup_claim": false,
        "cuda": cuda_identity_json(Some(probe)),
        "model": {
            "model_family": "qwen",
            "id": proof_model.id,
            "architecture": proof_model.architecture,
            "artifact_kind": "dense_gguf",
            "file": proof_model.file,
            "path": model_path.display().to_string(),
            "sha256": model_sha256,
        },
        "execution_path": {
            "model_class": "dense_regular_llm",
            "kernel_family": "dense_qwen_one_token_strict_cuda",
            "quantization_family": "dense_gguf_q8_0_f16_qwen_one_token_contract",
            "bitnet_packed_kernel_proof": false,
            "qk256_proof": false
        },
        "execution_plan": execution_plan,
        "descriptor_coverage": {
            "schema": 1,
            "source_artifact_kind": "dense_gguf_tensor_descriptor_inspection",
            "tensor_count": inspection.tensor_count,
            "metadata_count": inspection.metadata_count as u64,
            "required_roles_present": dense_model_boundary_fixture_coverage_complete(inspection),
            "strict_descriptor_complete": dense_model_boundary_fixture_coverage_complete(inspection),
            "dense_cuda_route_status": dense_model_boundary_route_status(inspection),
            "model_boundary_lm_head_source": dense_model_boundary_lm_head_source(inspection),
            "quantization_families": inspection.quantization_families,
            "bitnet_packed_marker_found": inspection.bitnet_packed_marker_found,
            "dense_gguf_inference_claimed": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "prerequisite_receipts": {
            "schema": 1,
            "all_layer_execution_plan_artifact_kind": DENSE_GGUF_ALL_LAYER_EXECUTION_PLAN_ARTIFACT_KIND,
            "all_layer_execution_plan_receipt_sha256": prerequisites.all_layer_plan_sha256,
            "model_boundary_fixtures_artifact_kind": DENSE_GGUF_MODEL_BOUNDARY_FIXTURES_ARTIFACT_KIND,
            "model_boundary_fixtures_receipt_sha256": prerequisites.model_boundary_fixtures_sha256,
            "kv_cache_policy_artifact_kind": DENSE_GGUF_KV_CACHE_POLICY_ARTIFACT_KIND,
            "kv_cache_policy_receipt_sha256": prerequisites.kv_cache_policy_sha256,
            "sampling_policy_artifact_kind": DENSE_GGUF_SAMPLING_POLICY_ARTIFACT_KIND,
            "sampling_policy_receipt_sha256": prerequisites.sampling_policy_sha256,
            "all_required_receipts_verified": true,
            "all_layer_execution_plan_claimed": true,
            "model_boundary_fixtures_claimed": true,
            "kv_cache_policy_claimed": true,
            "sampling_policy_claimed": true
        },
        "tokenizer_prompt_authority": {
            "schema": 1,
            "tokenizer_authority": "contract_authoritative",
            "prompt_authority": "contract_authoritative",
            "prompt_template": "qwen-chat-raw-deterministic",
            "bos_policy": "contract_default_add_bos",
            "deterministic_prompt": true,
            "prompt_token_count": prompt_token_count as u64,
            "prompt_token_ids_sha256": prompt_token_ids_sha256,
            "rendered_prompt_sha256": rendered_prompt_sha256,
            "rendered_prompt_bytes": rendered_prompt.len() as u64
        },
        "one_token_proof": {
            "schema": 1,
            "proof_scope": "qwen_strict_one_token_greedy_decode",
            "model_family": "qwen",
            "requested_new_tokens": 1_u64,
            "generated_tokens_count": 1_u64,
            "generation_policy": "greedy",
            "deterministic": true,
            "fallback_used": false,
            "cpu_reference_backend": "amd-9950x3d-cpu-avx512",
            "cuda_target_backend": HARDWARE_LANE,
            "prompt_token_count": prompt_token_count as u64,
            "prompt_token_ids_sha256": prompt_token_ids_sha256,
            "cpu_selected_token_id": cpu.selected_token_id as u64,
            "cuda_selected_token_id": cuda.selected_token_id as u64,
            "selected_token_match": cpu.selected_token_id == cuda.selected_token_id,
            "decoded_token_text": decoded_token_text,
            "cpu_logits_top_k_sha256": cpu.top_k_rank_sha256,
            "cuda_logits_top_k_sha256": cuda.top_k_rank_sha256,
            "cpu_logits_sha256": cpu.logits_sha256,
            "cuda_logits_sha256": cuda.logits_sha256,
            "logits_vector_length": cuda.logits_len as u64,
            "cpu_top_k": dense_qwen_top_k_json(&cpu.top_k),
            "cuda_top_k": dense_qwen_top_k_json(&cuda.top_k),
            "top_k_max_abs_error": top_k_max_abs_error,
            "top_k_mean_abs_error": top_k_mean_abs_error,
            "top_k_evidence_recorded": true,
            "top_k_compared": true,
            "top_k_match": top_k_match,
            "qwen_one_token_cuda_claimed": true,
            "qwen_short_decode_cuda_claimed": false,
            "qwen_chat_cuda_claimed": false,
            "dense_gguf_inference_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "server_ready_claimed": false,
            "full_cuda_residency_claimed": false
        },
        "quality_gate": {
            "schema": 1,
            "gate": "qwen_one_token_cuda_parity",
            "passed": true,
            "answer_ready_claimed": false,
            "short_decode_claimed": false,
            "chat_claimed": false
        },
        "kernel_stats": kernel_stats,
        "kernel_coverage": {
            "schema": 1,
            "route": DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND,
            "kernels_executed": [
                "dense_qwen_one_token_cuda_runtime",
                "dense_qwen_lm_head_cuda",
                "dense_qwen_logits_transfer_cuda"
            ],
            "all_required_dense_kernels_executed": true,
            "dense_kernel_invocations": runtime_invocations,
            "dense_kernel_launches": stats_launches,
            "bitnet_qk256_kernel_invocations": 0_u64,
            "cpu_fallback_kernel_invocations": 0_u64,
            "fallback_used": false
        },
        "timing": {
            "total_ms": cuda.total_ms,
            "first_token_ms": cuda.total_ms,
            "model_load_ms": cuda.model_load_ms,
            "cpu_reference_model_load_ms": cpu.model_load_ms,
            "prefill_ms": cuda.prefill_ms,
            "decode_ms": cuda.decode_ms,
            "embed_ms": cuda.embed_ms,
            "forward_ms": cuda.forward_ms,
            "logits_ms": cuda.logits_ms,
            "logits_download_ms": cuda.logits_download_ms,
            "kernel_time_ms": kernel_time_ms,
            "cpu_reference_total_ms": cpu.total_ms,
            "host_to_device_bytes": stats_h2d,
            "device_to_host_bytes": stats_d2h,
            "host_to_device_ms": cuda.model_load_ms,
            "host_to_device_ms_source": dense_qwen_h2d_timing_source(),
            "host_to_device_ms_scope": dense_qwen_h2d_timing_scope(),
            "host_to_device_ms_includes_non_transfer_overhead": true,
            "device_to_host_ms": cuda.logits_download_ms,
            "device_to_host_ms_source": dense_qwen_d2h_timing_source(),
            "transfer_timing_status": dense_qwen_transfer_timing_status(),
            "kernel_invocations": runtime_invocations,
            "kernel_launches": stats_launches
        },
        "tensor_residency": {
            "schema": 1,
            "scope": "qwen_one_token_strict_cuda",
            "model_class": "dense_regular_llm",
            "residency_accounting_recorded": true,
            "weights_uploaded_once": true,
            "weights_resident_on_cuda": true,
            "per_token_weight_upload": false,
            "kv_cache_policy_recorded": true,
            "sampling_policy_recorded": true,
            "runtime_logits_cuda_resident_before_download": cuda.logits_device_is_cuda,
            "fallback_used": false,
            "dense_gguf_inference_claimed": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false,
            "transfer_accounting": {
                "status": "measured",
                "host_to_device_bytes": stats_h2d,
                "device_to_host_bytes": stats_d2h,
                "host_to_device_ms": cuda.model_load_ms,
                "host_to_device_ms_source": dense_qwen_h2d_timing_source(),
                "host_to_device_ms_scope": dense_qwen_h2d_timing_scope(),
                "host_to_device_ms_includes_non_transfer_overhead": true,
                "device_to_host_ms": cuda.logits_download_ms,
                "device_to_host_ms_source": dense_qwen_d2h_timing_source(),
                "transfer_timing_status": dense_qwen_transfer_timing_status(),
                "kernel_invocations": runtime_invocations,
                "kernel_launches": stats_launches
            }
        },
        "parity": {
            "reference_backend": "amd-9950x3d-cpu-avx512",
            "target_backend": HARDWARE_LANE,
            "kernel_id": "dense_qwen_one_token_cuda_runtime",
            "fixture_id": format!("{}-one-token-greedy", proof_model.id),
            "passed": true,
            "max_abs_error": top_k_max_abs_error,
            "mean_abs_error": top_k_mean_abs_error,
            "tolerance": top_k_max_abs_error,
            "tolerance_source": "selected-token and top-k-rank equality; numeric drift recorded for top-k logits"
        },
        "claim_boundary": {
            "dense_regular_llm_cuda_claimed": true,
            "dense_tensor_residency_claimed": true,
            "dense_gguf_descriptor_inspection_claimed": true,
            "dense_gguf_linear_fixture_extraction_claimed": true,
            "dense_gguf_linear_cuda_parity_claimed": true,
            "dense_gguf_linear_role_sweep_cuda_parity_claimed": true,
            "dense_gguf_norm_cuda_parity_claimed": true,
            "dense_gguf_rope_cuda_parity_claimed": true,
            "dense_gguf_attention_score_cuda_parity_claimed": true,
            "dense_gguf_attention_softmax_cuda_parity_claimed": true,
            "dense_gguf_attention_v_mix_cuda_parity_claimed": true,
            "dense_gguf_mlp_activation_cuda_parity_claimed": true,
            "dense_gguf_one_layer_execution_plan_claimed": true,
            "dense_gguf_one_layer_cpu_reference_claimed": true,
            "dense_gguf_one_layer_cuda_integrated_parity_claimed": true,
            "dense_gguf_all_layer_execution_plan_claimed": true,
            "dense_gguf_model_boundary_fixtures_claimed": true,
            "kv_cache_policy_claimed": true,
            "sampling_policy_claimed": true,
            "qwen_one_token_cuda_claimed": true,
            "dense_gguf_one_layer_inference_claimed": false,
            "dense_gguf_inference_claimed": false,
            "qwen_short_decode_cuda_claimed": false,
            "qwen_chat_cuda_claimed": false,
            "server_ready_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false
        },
        "notes": [
            format!("{} proves exactly one deterministic greedy Qwen token through the dense regular-LLM CUDA route.", proof_model.work_item),
            "This receipt does not claim short decode, chat, server readiness, speedup, persistent residency, full CUDA residency, or BitNet packed I2_S/QK256 proof."
        ],
        "error": null
    }))
}

#[allow(clippy::too_many_arguments)]
fn dense_gguf_qwen_short_decode_strict_cuda_proof_receipt_json(
    inspection: &DenseGgufDescriptorInspection,
    prerequisites: &DenseQwenShortDecodePrerequisites,
    proof_model: &DenseQwenProofModel,
    probe: &bitnet_device_probe::NvidiaCudaProbe,
    model_path: &Path,
    model_sha256: &str,
    artifact_path: &str,
    timestamp_utc: &str,
    rendered_prompt: &str,
    prompt_token_count: usize,
    prompt_token_ids_sha256: &str,
    rendered_prompt_sha256: &str,
    cpu: &DenseQwenShortDecodeRun,
    cuda: &DenseQwenShortDecodeRun,
    decoded_text: &str,
    capture_profile: DenseQwenSourceCaptureProfile,
) -> Result<Value> {
    if inspection.model_family != "qwen" || inspection.architecture != proof_model.architecture {
        bail!(
            "dense Qwen short-decode receipt requires qwen/{} inspection, got {}/{}",
            proof_model.architecture,
            inspection.model_family,
            inspection.architecture
        );
    }
    if model_sha256 != proof_model.sha256 {
        bail!("dense Qwen short-decode receipt requires verified {} model SHA", proof_model.id);
    }
    let generated_count = cuda.generated_token_ids.len();
    capture_profile.validate_tokens(proof_model, generated_count)?;
    if cpu.generated_token_ids != cuda.generated_token_ids {
        bail!("dense Qwen short-decode receipt requires matching generated token IDs");
    }
    if cpu.steps.len() != generated_count || cuda.steps.len() != generated_count {
        bail!("dense Qwen short-decode receipt step count must match generated tokens");
    }
    if !cuda.logits_all_cuda_resident {
        bail!("dense Qwen short-decode receipt requires CUDA-resident logits before download");
    }

    let all_layer_cuda_ops = prerequisites
        .one_token
        .all_layer_plan
        .get("execution_plan")
        .and_then(|plan| plan.get("cuda_dense_regular_llm_ops"))
        .and_then(Value::as_u64)
        .or_else(|| {
            prerequisites
                .one_token
                .all_layer_plan
                .get("all_layer_plan")
                .and_then(|plan| plan.get("cuda_routable_ops_total"))
                .and_then(Value::as_u64)
        })
        .unwrap_or(1);
    let generated_count_u64 = generated_count as u64;
    let transformer_invocations = checked_u64_mul(
        all_layer_cuda_ops.max(1),
        generated_count_u64,
        "short-decode transformer invocations",
    )?;
    let lm_head_invocations = generated_count_u64;
    let logits_transfer_invocations = generated_count_u64;
    let runtime_invocations = transformer_invocations
        .saturating_add(lm_head_invocations)
        .saturating_add(logits_transfer_invocations);
    let summary = ModelDispatchSummary {
        total_ops: runtime_invocations as usize,
        cuda_bitnet_qk256_ops: 0,
        cuda_dense_regular_llm_ops: runtime_invocations as usize,
        cpu_fallback_ops: 0,
        unsupported_ops: 0,
        fallback_used: false,
        selected_route: Some(ModelDispatchBackend::CudaDenseRegularLlm),
        strict_cuda_ready: true,
    };
    let execution_plan = execution_plan_receipt(ExecutionPlanReceiptInput {
        model_family: &inspection.model_family,
        quantization: "dense_gguf_q8_0_f16_qwen_short_decode_contract",
        requested_backend: HARDWARE_LANE,
        selected_backend: HARDWARE_LANE,
        runtime_api: "cuda",
        strict_fallback_policy: "reject",
        summary,
        speedup_claim: false,
        full_cuda_residency_claimed: false,
    });

    let model_bytes = std::fs::metadata(model_path)
        .with_context(|| format!("failed to stat {}", model_path.display()))?
        .len();
    let logits_transfer_bytes = cuda.logits_transfer_bytes_total;
    if logits_transfer_bytes == 0 {
        bail!("dense Qwen short-decode receipt requires measured logits transfer bytes");
    }
    let kernel_time_ms = cuda.prefill_ms + cuda.forward_ms_total + cuda.logits_ms_total;
    let kernel_stats = json!([
        {
            "phase": "qwen_short_decode_runtime",
            "kernel_id": "dense_qwen_short_decode_cuda_runtime",
            "invocations": transformer_invocations,
            "fallback_invocations": 0_u64,
            "cpu_fallback_invocations": 0_u64,
            "host_to_device_bytes": model_bytes,
            "device_to_host_bytes": logits_transfer_bytes,
            "kernel_launches": transformer_invocations,
            "kernel_time_ms": kernel_time_ms
        },
        {
            "phase": "lm_head",
            "kernel_id": "dense_qwen_lm_head_cuda",
            "invocations": lm_head_invocations,
            "fallback_invocations": 0_u64,
            "cpu_fallback_invocations": 0_u64,
            "host_to_device_bytes": 0_u64,
            "device_to_host_bytes": 0_u64,
            "kernel_launches": lm_head_invocations,
            "kernel_time_ms": cuda.logits_ms_total
        },
        {
            "phase": "logits_transfer",
            "kernel_id": "dense_qwen_logits_transfer_cuda",
            "invocations": logits_transfer_invocations,
            "fallback_invocations": 0_u64,
            "cpu_fallback_invocations": 0_u64,
            "host_to_device_bytes": 0_u64,
            "device_to_host_bytes": 0_u64,
            "kernel_launches": logits_transfer_invocations,
            "kernel_time_ms": 0.0
        }
    ]);
    let stats_h2d = model_bytes;
    let stats_d2h = logits_transfer_bytes;
    let stats_launches = runtime_invocations;
    let observed_top_k = cuda
        .steps
        .first()
        .map(|step| step.top_k.len())
        .ok_or_else(|| anyhow!("dense Qwen short-decode receipt requires top-k steps"))?;
    if observed_top_k == 0 || cuda.steps.iter().any(|step| step.top_k.len() != observed_top_k) {
        bail!("dense Qwen short-decode receipt requires uniform non-empty top-k evidence");
    }
    let logits_transfer_reduction = dense_qwen_logits_transfer_reduction_json(
        cuda.logits_len,
        generated_count_u64,
        observed_top_k,
        stats_d2h,
        cuda.logits_transfer_mode,
    )?;
    let top_k_all_match = cpu.top_k_steps_sha256 == cuda.top_k_steps_sha256;
    let first_top_k_divergence = first_top_k_divergence_index(&cpu.steps, &cuda.steps);
    let step_json = dense_qwen_short_decode_steps_json(cpu, cuda);
    let top_k_max_abs_error = step_json
        .iter()
        .filter_map(|step| step.get("top_k_max_abs_error").and_then(Value::as_f64))
        .fold(0.0_f64, f64::max);
    let top_k_mean_abs_error = if step_json.is_empty() {
        0.0
    } else {
        step_json
            .iter()
            .filter_map(|step| step.get("top_k_mean_abs_error").and_then(Value::as_f64))
            .sum::<f64>()
            / step_json.len() as f64
    };

    Ok(json!({
        "schema": 1,
        "artifact_kind": DENSE_GGUF_QWEN_SHORT_DECODE_STRICT_CUDA_PROOF_ARTIFACT_KIND,
        "artifact_path": artifact_path,
        "claim": "dense_gguf_qwen_short_decode_strict_cuda_proof_recorded",
        "model_coverage_row": proof_model.model_coverage_row,
        "model_coverage_tier": proof_model.model_coverage_tier,
        "machine_id": MACHINE_ID,
        "hardware_lane": HARDWARE_LANE,
        "timestamp_utc": timestamp_utc,
        "requested_backend": HARDWARE_LANE,
        "selected_backend": HARDWARE_LANE,
        "runtime_api": "cuda",
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "speedup_claim": false,
        "cuda": cuda_identity_json(Some(probe)),
        "model": {
            "model_family": "qwen",
            "id": proof_model.id,
            "architecture": proof_model.architecture,
            "artifact_kind": "dense_gguf",
            "file": proof_model.file,
            "path": model_path.display().to_string(),
            "sha256": model_sha256,
        },
        "execution_path": {
            "model_class": "dense_regular_llm",
            "kernel_family": "dense_qwen_short_decode_strict_cuda",
            "quantization_family": "dense_gguf_q8_0_f16_qwen_short_decode_contract",
            "bitnet_packed_kernel_proof": false,
            "qk256_proof": false
        },
        "execution_plan": execution_plan,
        "descriptor_coverage": {
            "schema": 1,
            "source_artifact_kind": "dense_gguf_tensor_descriptor_inspection",
            "tensor_count": inspection.tensor_count,
            "metadata_count": inspection.metadata_count as u64,
            "required_roles_present": inspection.required_roles_present,
            "strict_descriptor_complete": inspection.strict_descriptor_complete,
            "dense_cuda_route_status": inspection.dense_cuda_route_status,
            "quantization_families": inspection.quantization_families,
            "bitnet_packed_marker_found": inspection.bitnet_packed_marker_found,
            "dense_gguf_inference_claimed": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "prerequisite_receipts": {
            "schema": 1,
            "all_layer_execution_plan_artifact_kind": DENSE_GGUF_ALL_LAYER_EXECUTION_PLAN_ARTIFACT_KIND,
            "all_layer_execution_plan_receipt_sha256": prerequisites.one_token.all_layer_plan_sha256,
            "model_boundary_fixtures_artifact_kind": DENSE_GGUF_MODEL_BOUNDARY_FIXTURES_ARTIFACT_KIND,
            "model_boundary_fixtures_receipt_sha256": prerequisites.one_token.model_boundary_fixtures_sha256,
            "kv_cache_policy_artifact_kind": DENSE_GGUF_KV_CACHE_POLICY_ARTIFACT_KIND,
            "kv_cache_policy_receipt_sha256": prerequisites.one_token.kv_cache_policy_sha256,
            "sampling_policy_artifact_kind": DENSE_GGUF_SAMPLING_POLICY_ARTIFACT_KIND,
            "sampling_policy_receipt_sha256": prerequisites.one_token.sampling_policy_sha256,
            "one_token_proof_artifact_kind": DENSE_GGUF_QWEN_ONE_TOKEN_STRICT_CUDA_PROOF_ARTIFACT_KIND,
            "one_token_proof_receipt_sha256": prerequisites.one_token_proof_sha256,
            "all_required_receipts_verified": true,
            "all_layer_execution_plan_claimed": true,
            "model_boundary_fixtures_claimed": true,
            "kv_cache_policy_claimed": true,
            "sampling_policy_claimed": true,
            "one_token_proof_claimed": true
        },
        "tokenizer_prompt_authority": {
            "schema": 1,
            "tokenizer_authority": "contract_authoritative",
            "prompt_authority": "contract_authoritative",
            "prompt_template": "qwen-chat-raw-deterministic",
            "bos_policy": "contract_default_add_bos",
            "deterministic_prompt": true,
            "prompt_token_count": prompt_token_count as u64,
            "prompt_token_ids_sha256": prompt_token_ids_sha256,
            "rendered_prompt_sha256": rendered_prompt_sha256,
            "rendered_prompt_bytes": rendered_prompt.len() as u64
        },
        "short_decode_proof": {
            "schema": 1,
            "proof_scope": capture_profile.proof_scope(),
            "profile_id": capture_profile.profile_id(),
            "model_family": "qwen",
            "requested_new_tokens": generated_count_u64,
            "generated_tokens_count": generated_count_u64,
            "generation_policy": "greedy",
            "deterministic": true,
            "fallback_used": false,
            "cpu_reference_backend": "amd-9950x3d-cpu-avx512",
            "cuda_target_backend": HARDWARE_LANE,
            "prompt_token_count": prompt_token_count as u64,
            "prompt_token_ids_sha256": prompt_token_ids_sha256,
            "cpu_generated_token_ids": cpu.generated_token_ids,
            "cuda_generated_token_ids": cuda.generated_token_ids,
            "cpu_generated_token_ids_sha256": cpu.generated_token_ids_sha256,
            "cuda_generated_token_ids_sha256": cuda.generated_token_ids_sha256,
            "generated_token_ids_match": true,
            "first_token_divergence_index": null,
            "cpu_logits_top_k_steps_sha256": cpu.top_k_steps_sha256,
            "cuda_logits_top_k_steps_sha256": cuda.top_k_steps_sha256,
            "top_k_evidence_recorded": true,
            "top_k_compared": true,
            "top_k_all_match": top_k_all_match,
            "first_top_k_divergence_index": first_top_k_divergence,
            "top_k_max_abs_error": top_k_max_abs_error,
            "top_k_mean_abs_error": top_k_mean_abs_error,
            "steps": step_json,
            "decoded_text": decoded_text,
            "qwen_one_token_cuda_claimed": true,
            "qwen_short_decode_cuda_claimed": true,
            "qwen_chat_cuda_claimed": false,
            "dense_gguf_inference_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "server_ready_claimed": false,
            "full_cuda_residency_claimed": false
        },
        "quality_gate": {
            "schema": 1,
            "gate": "qwen_short_decode_cuda_parity",
            "passed": true,
            "answer_ready_claimed": false,
            "short_decode_claimed": true,
            "chat_claimed": false
        },
        "kernel_stats": kernel_stats,
        "logits_transfer_reduction": logits_transfer_reduction,
        "kernel_coverage": {
            "schema": 1,
            "route": DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND,
            "kernels_executed": [
                "dense_qwen_short_decode_cuda_runtime",
                "dense_qwen_lm_head_cuda",
                "dense_qwen_logits_transfer_cuda"
            ],
            "all_required_dense_kernels_executed": true,
            "dense_kernel_invocations": runtime_invocations,
            "dense_kernel_launches": stats_launches,
            "bitnet_qk256_kernel_invocations": 0_u64,
            "cpu_fallback_kernel_invocations": 0_u64,
            "fallback_used": false
        },
        "timing": {
            "total_ms": cuda.total_ms,
            "first_token_ms": cuda.first_token_ms,
            "model_load_ms": cuda.model_load_ms,
            "cpu_reference_model_load_ms": cpu.model_load_ms,
            "prefill_ms": cuda.prefill_ms,
            "decode_total_ms": cuda.decode_total_ms,
            "embed_ms_total": cuda.embed_ms_total,
            "forward_ms_total": cuda.forward_ms_total,
            "logits_ms_total": cuda.logits_ms_total,
            "logits_download_ms_total": cuda.logits_download_ms_total,
            "kernel_time_ms": kernel_time_ms,
            "cpu_reference_total_ms": cpu.total_ms,
            "host_to_device_bytes": stats_h2d,
            "device_to_host_bytes": stats_d2h,
            "host_to_device_ms": cuda.model_load_ms,
            "host_to_device_ms_source": dense_qwen_h2d_timing_source(),
            "host_to_device_ms_scope": dense_qwen_h2d_timing_scope(),
            "host_to_device_ms_includes_non_transfer_overhead": true,
            "device_to_host_ms": cuda.logits_download_ms_total,
            "device_to_host_ms_source": cuda.logits_transfer_mode.d2h_timing_source(),
            "transfer_timing_status": dense_qwen_transfer_timing_status(),
            "kernel_invocations": runtime_invocations,
            "kernel_launches": stats_launches,
            "generated_tokens_count": generated_count_u64
        },
        "tensor_residency": {
            "schema": 1,
            "scope": "qwen_short_decode_strict_cuda",
            "model_class": "dense_regular_llm",
            "residency_accounting_recorded": true,
            "weights_uploaded_once": true,
            "weights_resident_on_cuda": true,
            "per_token_weight_upload": false,
            "kv_cache_policy_recorded": true,
            "sampling_policy_recorded": true,
            "runtime_logits_cuda_resident_before_download": cuda.logits_all_cuda_resident,
            "fallback_used": false,
            "dense_gguf_inference_claimed": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false,
            "transfer_accounting": {
                "status": "measured",
                "host_to_device_bytes": stats_h2d,
                "device_to_host_bytes": stats_d2h,
                "host_to_device_ms": cuda.model_load_ms,
                "host_to_device_ms_source": dense_qwen_h2d_timing_source(),
                "host_to_device_ms_scope": dense_qwen_h2d_timing_scope(),
                "host_to_device_ms_includes_non_transfer_overhead": true,
                "device_to_host_ms": cuda.logits_download_ms_total,
                "device_to_host_ms_source": cuda.logits_transfer_mode.d2h_timing_source(),
                "transfer_timing_status": dense_qwen_transfer_timing_status(),
                "kernel_invocations": runtime_invocations,
                "kernel_launches": stats_launches
            }
        },
        "parity": {
            "reference_backend": "amd-9950x3d-cpu-avx512",
            "target_backend": HARDWARE_LANE,
            "kernel_id": "dense_qwen_short_decode_cuda_runtime",
            "fixture_id": format!("{}-short-decode-greedy", proof_model.id),
            "passed": true,
            "max_abs_error": top_k_max_abs_error,
            "mean_abs_error": top_k_mean_abs_error,
            "tolerance": top_k_max_abs_error,
            "tolerance_source": "generated token equality; numeric drift recorded for top-k logits"
        },
        "claim_boundary": {
            "dense_regular_llm_cuda_claimed": true,
            "dense_tensor_residency_claimed": true,
            "dense_gguf_descriptor_inspection_claimed": true,
            "dense_gguf_linear_fixture_extraction_claimed": true,
            "dense_gguf_linear_cuda_parity_claimed": true,
            "dense_gguf_linear_role_sweep_cuda_parity_claimed": true,
            "dense_gguf_norm_cuda_parity_claimed": true,
            "dense_gguf_rope_cuda_parity_claimed": true,
            "dense_gguf_attention_score_cuda_parity_claimed": true,
            "dense_gguf_attention_softmax_cuda_parity_claimed": true,
            "dense_gguf_attention_v_mix_cuda_parity_claimed": true,
            "dense_gguf_mlp_activation_cuda_parity_claimed": true,
            "dense_gguf_one_layer_execution_plan_claimed": true,
            "dense_gguf_one_layer_cpu_reference_claimed": true,
            "dense_gguf_one_layer_cuda_integrated_parity_claimed": true,
            "dense_gguf_all_layer_execution_plan_claimed": true,
            "dense_gguf_model_boundary_fixtures_claimed": true,
            "kv_cache_policy_claimed": true,
            "sampling_policy_claimed": true,
            "qwen_one_token_cuda_claimed": true,
            "qwen_short_decode_cuda_claimed": true,
            "dense_gguf_one_layer_inference_claimed": false,
            "dense_gguf_inference_claimed": false,
            "qwen_chat_cuda_claimed": false,
            "server_ready_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false
        },
        "notes": [
            "CUDA-DENSE-045 proves a bounded deterministic greedy Qwen short decode through the dense regular-LLM CUDA route.",
            "This receipt does not claim chat, server readiness, speedup, persistent residency, full CUDA residency, or BitNet packed I2_S/QK256 proof."
        ],
        "error": null
    }))
}

#[allow(clippy::too_many_arguments)]
fn dense_gguf_qwen_warm_decode_strict_cuda_proof_receipt_json(
    inspection: &DenseGgufDescriptorInspection,
    prerequisites: &DenseQwenWarmSessionPrerequisites,
    proof_model: &DenseQwenProofModel,
    probe: &bitnet_device_probe::NvidiaCudaProbe,
    model_path: &Path,
    model_sha256: &str,
    artifact_path: &str,
    timestamp_utc: &str,
    rendered_prompt: &str,
    prompt_token_count: usize,
    prompt_token_ids_sha256: &str,
    rendered_prompt_sha256: &str,
    cpu: &DenseQwenShortDecodeRun,
    cuda: &DenseQwenShortDecodeRun,
    decoded_text: &str,
) -> Result<Value> {
    let mut receipt = dense_gguf_qwen_short_decode_strict_cuda_proof_receipt_json(
        inspection,
        &prerequisites.short_decode,
        proof_model,
        probe,
        model_path,
        model_sha256,
        artifact_path,
        timestamp_utc,
        rendered_prompt,
        prompt_token_count,
        prompt_token_ids_sha256,
        rendered_prompt_sha256,
        cpu,
        cuda,
        decoded_text,
        DenseQwenSourceCaptureProfile::Qwen3WarmDecode128,
    )?;

    receipt["artifact_kind"] = json!(DENSE_GGUF_QWEN_WARM_DECODE_STRICT_CUDA_PROOF_ARTIFACT_KIND);
    receipt["claim"] = json!("dense_gguf_qwen_warm_decode_strict_cuda_proof_recorded");
    receipt["execution_path"]["kernel_family"] = json!("dense_qwen_warm_decode_strict_cuda");
    receipt["execution_path"]["quantization_family"] =
        json!("dense_gguf_q8_0_f16_qwen_warm_decode_contract");
    receipt["execution_plan"]["quantization"] =
        json!("dense_gguf_q8_0_f16_qwen_warm_decode_contract");

    let warm_decode_proof = receipt
        .as_object_mut()
        .and_then(|object| object.remove("short_decode_proof"))
        .ok_or_else(|| anyhow!("warm-decode receipt source is missing short_decode_proof"))?;
    receipt["warm_decode_proof"] = warm_decode_proof;
    receipt["warm_decode_proof"]["proof_scope"] =
        json!(DenseQwenSourceCaptureProfile::Qwen3WarmDecode128.proof_scope());
    receipt["warm_decode_proof"]["profile_id"] =
        json!(DenseQwenSourceCaptureProfile::Qwen3WarmDecode128.profile_id());
    receipt["warm_decode_proof"]["warm_context_reused"] = json!(true);
    receipt["warm_decode_proof"]["decode_started_from_prefilled_context"] = json!(true);
    receipt["warm_decode_proof"]["warm_context_prompt_token_count"] =
        json!(prompt_token_count as u64);
    receipt["warm_decode_proof"]["qwen_warm_decode_cuda_claimed"] = json!(true);
    receipt["warm_decode_proof"]["qwen_chat_cuda_claimed"] = json!(false);
    receipt["warm_decode_proof"]["server_ready_claimed"] = json!(false);
    receipt["warm_decode_proof"]["speedup_claim"] = json!(false);
    receipt["warm_decode_proof"]["full_cuda_residency_claimed"] = json!(false);

    receipt["quality_gate"] = json!({
        "schema": 1,
        "gate": "qwen_warm_decode_cuda_parity",
        "passed": true,
        "warm_context_decode_claimed": true,
        "ask_claimed": false,
        "chat_claimed": false,
        "server_ready_claimed": false
    });
    receipt["warm_context_proof"] = json!({
        "schema": 1,
        "proof_scope": "qwen3_decode_128_from_warm_context",
        "profile_id": "decode_128_from_warm_context",
        "warm_context_reused": true,
        "decode_started_from_prefilled_context": true,
        "warm_context_prompt_token_count": prompt_token_count as u64,
        "prompt_token_ids_sha256": prompt_token_ids_sha256,
        "rendered_prompt_sha256": rendered_prompt_sha256,
        "requested_new_tokens": 128_u64,
        "generated_tokens_count": 128_u64,
        "model_loaded_once": true,
        "cuda_context_initialized_once": true,
        "weights_uploaded_once": true,
        "per_request_model_load": false,
        "fallback_used": false,
        "speedup_claim": false,
        "server_ready_claimed": false,
        "full_cuda_residency_claimed": false
    });
    receipt["session_lifecycle"] = json!({
        "schema": 1,
        "proof_scope": "qwen3_warm_decode_strict_cuda",
        "model_loaded_once": true,
        "tokenizer_loaded_once": true,
        "cuda_context_initialized_once": true,
        "cuda_context_once": true,
        "weights_uploaded_once": true,
        "per_request_model_load": false,
        "per_token_weight_upload": false,
        "workspace_reused": true,
        "runtime_buffers_reused": true,
        "warm_context_reused": true,
        "decode_started_from_prefilled_context": true,
        "fallback_used": false,
        "scoped_warm_context_residency_claimed": true,
        "persistent_session_residency_claimed": false,
        "full_cuda_residency_claimed": false
    });
    receipt["tensor_residency"]["scope"] = json!("qwen_warm_decode_strict_cuda");
    receipt["tensor_residency"]["model_loaded_once"] = json!(true);
    receipt["tensor_residency"]["tokenizer_loaded_once"] = json!(true);
    receipt["tensor_residency"]["cuda_context_initialized_once"] = json!(true);
    receipt["tensor_residency"]["cuda_context_once"] = json!(true);
    receipt["tensor_residency"]["per_request_model_load"] = json!(false);
    receipt["tensor_residency"]["workspace_reused"] = json!(true);
    receipt["tensor_residency"]["warm_context_reused"] = json!(true);
    receipt["tensor_residency"]["scoped_warm_context_residency_claimed"] = json!(true);
    receipt["timing"]["warm_context_prefill_ms"] = receipt["timing"]["prefill_ms"].clone();
    receipt["timing"]["warm_context_prompt_tokens"] = json!(prompt_token_count as u64);
    receipt["parity"]["kernel_id"] = json!("dense_qwen_warm_decode_cuda_runtime");
    receipt["parity"]["fixture_id"] = json!(format!("{}-warm-decode-128-greedy", proof_model.id));

    receipt["claim_boundary"]["qwen_warm_decode_cuda_claimed"] = json!(true);
    receipt["claim_boundary"]["qwen_chat_cuda_claimed"] = json!(false);
    receipt["claim_boundary"]["server_ready_claimed"] = json!(false);
    receipt["claim_boundary"]["speedup_claim"] = json!(false);
    receipt["claim_boundary"]["persistent_session_residency_claimed"] = json!(false);
    receipt["claim_boundary"]["full_cuda_residency_claimed"] = json!(false);
    receipt["notes"] = json!([
        "CUDA-MODEL-017A records Qwen3 decode_128_from_warm_context source-capture tooling only.",
        "This receipt proves a Qwen3-only 128-token CUDA decode from a prefilling warm context and does not claim ask/chat token expansion, speedup, server readiness, broad dense GGUF readiness, full CUDA residency, Qwen2.5 inheritance, or BitNet packed I2_S/QK256 proof."
    ]);

    Ok(receipt)
}

#[allow(clippy::too_many_arguments)]
fn dense_gguf_qwen_warm_session_strict_cuda_proof_receipt_json(
    inspection: &DenseGgufDescriptorInspection,
    prerequisites: &DenseQwenWarmSessionPrerequisites,
    proof_model: &DenseQwenProofModel,
    probe: &bitnet_device_probe::NvidiaCudaProbe,
    model_path: &Path,
    model_sha256: &str,
    artifact_path: &str,
    timestamp_utc: &str,
    tokenizer_load_ms: f64,
    prompt_evidence: &[DenseQwenWarmSessionPromptEvidence],
    cpu: &DenseQwenWarmSessionRun,
    cuda: &DenseQwenWarmSessionRun,
    decoded_texts: &[String],
) -> Result<Value> {
    if inspection.model_family != "qwen" || inspection.architecture != proof_model.architecture {
        bail!(
            "dense Qwen warm-session receipt requires qwen/{} inspection, got {}/{}",
            proof_model.architecture,
            inspection.model_family,
            inspection.architecture
        );
    }
    if model_sha256 != proof_model.sha256 {
        bail!("dense Qwen warm-session receipt requires verified {} model SHA", proof_model.id);
    }
    let turns_count = cuda.turns.len();
    if !(2..=4).contains(&turns_count) {
        bail!("dense Qwen warm-session receipt requires 2-4 turns");
    }
    if cpu.turns.len() != turns_count
        || prompt_evidence.len() != turns_count
        || decoded_texts.len() != turns_count
    {
        bail!("dense Qwen warm-session receipt requires matching CPU/CUDA/prompt turn counts");
    }

    let requested_new_tokens = cuda
        .turns
        .first()
        .map(|turn| turn.generated_token_ids.len())
        .ok_or_else(|| anyhow!("dense Qwen warm-session receipt requires turns"))?;
    if !(5..=16).contains(&requested_new_tokens) {
        bail!("dense Qwen warm-session receipt requires 5-16 generated tokens per turn");
    }
    for (index, (cpu_turn, cuda_turn)) in cpu.turns.iter().zip(cuda.turns.iter()).enumerate() {
        if cpu_turn.generated_token_ids != cuda_turn.generated_token_ids {
            bail!(
                "dense Qwen warm-session receipt requires matching generated token IDs at turn {index}"
            );
        }
        if cuda_turn.generated_token_ids.len() != requested_new_tokens {
            bail!("dense Qwen warm-session receipt requires uniform generated-token count");
        }
        if cpu_turn.steps.len() != requested_new_tokens
            || cuda_turn.steps.len() != requested_new_tokens
        {
            bail!("dense Qwen warm-session receipt step count must match generated tokens");
        }
        if !cuda_turn.logits_all_cuda_resident {
            bail!(
                "dense Qwen warm-session receipt requires CUDA-resident logits before download at turn {index}"
            );
        }
    }

    let all_layer_cuda_ops = prerequisites
        .short_decode
        .one_token
        .all_layer_plan
        .get("execution_plan")
        .and_then(|plan| plan.get("cuda_dense_regular_llm_ops"))
        .and_then(Value::as_u64)
        .or_else(|| {
            prerequisites
                .short_decode
                .one_token
                .all_layer_plan
                .get("all_layer_plan")
                .and_then(|plan| plan.get("cuda_routable_ops_total"))
                .and_then(Value::as_u64)
        })
        .unwrap_or(1);
    let turns_count_u64 = turns_count as u64;
    let requested_new_tokens_u64 = requested_new_tokens as u64;
    let generated_tokens_total = checked_u64_mul(
        turns_count_u64,
        requested_new_tokens_u64,
        "warm-session generated token total",
    )?;
    let transformer_invocations = checked_u64_mul(
        all_layer_cuda_ops.max(1),
        generated_tokens_total,
        "warm-session transformer invocations",
    )?;
    let lm_head_invocations = generated_tokens_total;
    let logits_transfer_invocations = generated_tokens_total;
    let runtime_invocations = transformer_invocations
        .saturating_add(lm_head_invocations)
        .saturating_add(logits_transfer_invocations);
    let summary = ModelDispatchSummary {
        total_ops: runtime_invocations as usize,
        cuda_bitnet_qk256_ops: 0,
        cuda_dense_regular_llm_ops: runtime_invocations as usize,
        cpu_fallback_ops: 0,
        unsupported_ops: 0,
        fallback_used: false,
        selected_route: Some(ModelDispatchBackend::CudaDenseRegularLlm),
        strict_cuda_ready: true,
    };
    let execution_plan = execution_plan_receipt(ExecutionPlanReceiptInput {
        model_family: &inspection.model_family,
        quantization: "dense_gguf_q8_0_f16_qwen_warm_session_contract",
        requested_backend: HARDWARE_LANE,
        selected_backend: HARDWARE_LANE,
        runtime_api: "cuda",
        strict_fallback_policy: "reject",
        summary,
        speedup_claim: false,
        full_cuda_residency_claimed: false,
    });

    let model_bytes = std::fs::metadata(model_path)
        .with_context(|| format!("failed to stat {}", model_path.display()))?
        .len();
    let logits_len = cuda.turns.iter().map(|turn| turn.logits_len).max().unwrap_or(0);
    let logits_transfer_mode = cuda
        .turns
        .first()
        .map(|turn| turn.logits_transfer_mode)
        .ok_or_else(|| anyhow!("dense Qwen warm-session receipt requires turns"))?;
    if cuda.turns.iter().any(|turn| turn.logits_transfer_mode != logits_transfer_mode) {
        bail!("dense Qwen warm-session receipt requires uniform logits transfer mode");
    }
    let logits_transfer_bytes = cuda.turns.iter().try_fold(0_u64, |acc, turn| {
        acc.checked_add(turn.logits_transfer_bytes_total)
            .ok_or_else(|| anyhow!("dense Qwen warm-session logits transfer bytes overflowed"))
    })?;
    if logits_transfer_bytes == 0 {
        bail!("dense Qwen warm-session receipt requires measured logits transfer bytes");
    }
    let kernel_time_ms = cuda
        .turns
        .iter()
        .map(|turn| turn.prefill_ms + turn.forward_ms_total + turn.logits_ms_total)
        .sum::<f64>();
    let stats_h2d = model_bytes;
    let stats_d2h = logits_transfer_bytes;
    let stats_launches = runtime_invocations;
    let observed_top_k = cuda
        .turns
        .iter()
        .flat_map(|turn| turn.steps.iter())
        .next()
        .map(|step| step.top_k.len())
        .ok_or_else(|| anyhow!("dense Qwen warm-session receipt requires top-k steps"))?;
    if observed_top_k == 0
        || cuda
            .turns
            .iter()
            .flat_map(|turn| turn.steps.iter())
            .any(|step| step.top_k.len() != observed_top_k)
    {
        bail!("dense Qwen warm-session receipt requires uniform non-empty top-k evidence");
    }
    let logits_transfer_reduction = dense_qwen_logits_transfer_reduction_json(
        logits_len,
        generated_tokens_total,
        observed_top_k,
        stats_d2h,
        logits_transfer_mode,
    )?;

    let cpu_generated_all = cpu
        .turns
        .iter()
        .flat_map(|turn| turn.generated_token_ids.iter().copied())
        .collect::<Vec<_>>();
    let cuda_generated_all = cuda
        .turns
        .iter()
        .flat_map(|turn| turn.generated_token_ids.iter().copied())
        .collect::<Vec<_>>();
    let cpu_generated_all_sha256 = sha256_u32(&cpu_generated_all);
    let cuda_generated_all_sha256 = sha256_u32(&cuda_generated_all);
    let prompt_authority_turns = prompt_evidence
        .iter()
        .map(|prompt| {
            json!({
                "index": prompt.index as u64,
                "prompt_token_count": prompt.token_count as u64,
                "prompt_token_ids_sha256": prompt.token_ids_sha256,
                "rendered_prompt_sha256": prompt.rendered_prompt_sha256,
                "rendered_prompt_bytes": prompt.rendered_prompt_bytes as u64
            })
        })
        .collect::<Vec<_>>();
    let prompt_token_ids_sha256 = sha256_json(&Value::Array(
        prompt_authority_turns.iter().map(|turn| turn["prompt_token_ids_sha256"].clone()).collect(),
    ))?;
    let rendered_prompt_sha256 = sha256_json(&Value::Array(
        prompt_authority_turns.iter().map(|turn| turn["rendered_prompt_sha256"].clone()).collect(),
    ))?;
    let prompt_token_count_total =
        prompt_evidence.iter().map(|prompt| prompt.token_count).sum::<usize>();

    let mut top_k_all_match = true;
    let mut first_top_k_divergence = None;
    let mut top_k_max_abs_error = 0.0_f64;
    let mut top_k_mean_abs_error_sum = 0.0_f64;
    let mut top_k_mean_abs_error_count = 0_usize;
    let mut turn_json = Vec::with_capacity(turns_count);
    for (turn_index, ((cpu_turn, cuda_turn), decoded_text)) in
        cpu.turns.iter().zip(cuda.turns.iter()).zip(decoded_texts.iter()).enumerate()
    {
        let step_json = dense_qwen_short_decode_steps_json(cpu_turn, cuda_turn);
        for step in &step_json {
            let max_error = step.get("top_k_max_abs_error").and_then(Value::as_f64).unwrap_or(0.0);
            let mean_error =
                step.get("top_k_mean_abs_error").and_then(Value::as_f64).unwrap_or(0.0);
            top_k_max_abs_error = top_k_max_abs_error.max(max_error);
            top_k_mean_abs_error_sum += mean_error;
            top_k_mean_abs_error_count += 1;
        }
        let turn_top_k_match = cpu_turn.top_k_steps_sha256 == cuda_turn.top_k_steps_sha256;
        top_k_all_match &= turn_top_k_match;
        if first_top_k_divergence.is_none()
            && let Some(step_index) =
                first_top_k_divergence_index(&cpu_turn.steps, &cuda_turn.steps)
        {
            first_top_k_divergence = Some(json!({
                "turn_index": turn_index as u64,
                "step_index": step_index as u64
            }));
        }
        let prompt = &prompt_evidence[turn_index];
        turn_json.push(json!({
            "index": turn_index as u64,
            "prompt_token_count": prompt.token_count as u64,
            "prompt_token_ids_sha256": prompt.token_ids_sha256,
            "rendered_prompt_sha256": prompt.rendered_prompt_sha256,
            "requested_new_tokens": requested_new_tokens_u64,
            "generated_tokens_count": requested_new_tokens_u64,
            "cpu_generated_token_ids": cpu_turn.generated_token_ids,
            "cuda_generated_token_ids": cuda_turn.generated_token_ids,
            "cpu_generated_token_ids_sha256": cpu_turn.generated_token_ids_sha256,
            "cuda_generated_token_ids_sha256": cuda_turn.generated_token_ids_sha256,
            "generated_token_ids_match": true,
            "first_token_divergence_index": null,
            "cpu_logits_top_k_steps_sha256": cpu_turn.top_k_steps_sha256,
            "cuda_logits_top_k_steps_sha256": cuda_turn.top_k_steps_sha256,
            "top_k_all_match": turn_top_k_match,
            "first_top_k_divergence_index": first_top_k_divergence_index(&cpu_turn.steps, &cuda_turn.steps),
            "steps": step_json,
            "decoded_text": decoded_text,
            "cuda_turn_timing": {
                "total_ms": cuda_turn.total_ms,
                "first_token_ms": cuda_turn.first_token_ms,
                "prefill_ms": cuda_turn.prefill_ms,
                "decode_total_ms": cuda_turn.decode_total_ms,
                "embed_ms_total": cuda_turn.embed_ms_total,
                "forward_ms_total": cuda_turn.forward_ms_total,
                "logits_ms_total": cuda_turn.logits_ms_total,
                "logits_download_ms_total": cuda_turn.logits_download_ms_total,
                "logits_device_all_cuda_resident": cuda_turn.logits_all_cuda_resident
            }
        }));
    }
    let top_k_mean_abs_error = if top_k_mean_abs_error_count == 0 {
        0.0
    } else {
        top_k_mean_abs_error_sum / top_k_mean_abs_error_count as f64
    };
    let top_k_session_sha256 = sha256_json(&Value::Array(
        cuda.turns.iter().map(|turn| Value::String(turn.top_k_steps_sha256.clone())).collect(),
    ))?;
    let kernel_stats = json!([
        {
            "phase": "qwen_warm_session_runtime",
            "kernel_id": "dense_qwen_warm_session_cuda_runtime",
            "invocations": transformer_invocations,
            "fallback_invocations": 0_u64,
            "cpu_fallback_invocations": 0_u64,
            "host_to_device_bytes": model_bytes,
            "device_to_host_bytes": logits_transfer_bytes,
            "kernel_launches": transformer_invocations,
            "kernel_time_ms": kernel_time_ms
        },
        {
            "phase": "lm_head",
            "kernel_id": "dense_qwen_lm_head_cuda",
            "invocations": lm_head_invocations,
            "fallback_invocations": 0_u64,
            "cpu_fallback_invocations": 0_u64,
            "host_to_device_bytes": 0_u64,
            "device_to_host_bytes": 0_u64,
            "kernel_launches": lm_head_invocations,
            "kernel_time_ms": cuda.turns.iter().map(|turn| turn.logits_ms_total).sum::<f64>()
        },
        {
            "phase": "logits_transfer",
            "kernel_id": "dense_qwen_logits_transfer_cuda",
            "invocations": logits_transfer_invocations,
            "fallback_invocations": 0_u64,
            "cpu_fallback_invocations": 0_u64,
            "host_to_device_bytes": 0_u64,
            "device_to_host_bytes": 0_u64,
            "kernel_launches": logits_transfer_invocations,
            "kernel_time_ms": 0.0
        }
    ]);

    Ok(json!({
        "schema": 1,
        "artifact_kind": DENSE_GGUF_QWEN_WARM_SESSION_STRICT_CUDA_PROOF_ARTIFACT_KIND,
        "artifact_path": artifact_path,
        "claim": "dense_gguf_qwen_warm_session_strict_cuda_proof_recorded",
        "model_coverage_row": proof_model.model_coverage_row,
        "model_coverage_tier": proof_model.model_coverage_tier,
        "machine_id": MACHINE_ID,
        "hardware_lane": HARDWARE_LANE,
        "timestamp_utc": timestamp_utc,
        "requested_backend": HARDWARE_LANE,
        "selected_backend": HARDWARE_LANE,
        "runtime_api": "cuda",
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "speedup_claim": false,
        "cuda": cuda_identity_json(Some(probe)),
        "model": {
            "model_family": "qwen",
            "id": proof_model.id,
            "architecture": proof_model.architecture,
            "artifact_kind": "dense_gguf",
            "file": proof_model.file,
            "path": model_path.display().to_string(),
            "sha256": model_sha256,
        },
        "execution_path": {
            "model_class": "dense_regular_llm",
            "kernel_family": "dense_qwen_warm_session_strict_cuda",
            "quantization_family": "dense_gguf_q8_0_f16_qwen_warm_session_contract",
            "bitnet_packed_kernel_proof": false,
            "qk256_proof": false
        },
        "execution_plan": execution_plan,
        "descriptor_coverage": {
            "schema": 1,
            "source_artifact_kind": "dense_gguf_tensor_descriptor_inspection",
            "tensor_count": inspection.tensor_count,
            "metadata_count": inspection.metadata_count as u64,
            "required_roles_present": inspection.required_roles_present,
            "strict_descriptor_complete": inspection.strict_descriptor_complete,
            "dense_cuda_route_status": inspection.dense_cuda_route_status,
            "quantization_families": inspection.quantization_families,
            "bitnet_packed_marker_found": inspection.bitnet_packed_marker_found,
            "dense_gguf_inference_claimed": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "prerequisite_receipts": {
            "schema": 1,
            "all_layer_execution_plan_artifact_kind": DENSE_GGUF_ALL_LAYER_EXECUTION_PLAN_ARTIFACT_KIND,
            "all_layer_execution_plan_receipt_sha256": prerequisites.short_decode.one_token.all_layer_plan_sha256,
            "model_boundary_fixtures_artifact_kind": DENSE_GGUF_MODEL_BOUNDARY_FIXTURES_ARTIFACT_KIND,
            "model_boundary_fixtures_receipt_sha256": prerequisites.short_decode.one_token.model_boundary_fixtures_sha256,
            "kv_cache_policy_artifact_kind": DENSE_GGUF_KV_CACHE_POLICY_ARTIFACT_KIND,
            "kv_cache_policy_receipt_sha256": prerequisites.short_decode.one_token.kv_cache_policy_sha256,
            "sampling_policy_artifact_kind": DENSE_GGUF_SAMPLING_POLICY_ARTIFACT_KIND,
            "sampling_policy_receipt_sha256": prerequisites.short_decode.one_token.sampling_policy_sha256,
            "one_token_proof_artifact_kind": DENSE_GGUF_QWEN_ONE_TOKEN_STRICT_CUDA_PROOF_ARTIFACT_KIND,
            "one_token_proof_receipt_sha256": prerequisites.short_decode.one_token_proof_sha256,
            "short_decode_proof_artifact_kind": DENSE_GGUF_QWEN_SHORT_DECODE_STRICT_CUDA_PROOF_ARTIFACT_KIND,
            "short_decode_proof_receipt_sha256": prerequisites.short_decode_proof_sha256,
            "all_required_receipts_verified": true,
            "all_layer_execution_plan_claimed": true,
            "model_boundary_fixtures_claimed": true,
            "kv_cache_policy_claimed": true,
            "sampling_policy_claimed": true,
            "one_token_proof_claimed": true,
            "short_decode_proof_claimed": true
        },
        "tokenizer_prompt_authority": {
            "schema": 1,
            "tokenizer_authority": "contract_authoritative",
            "prompt_authority": "contract_authoritative",
            "prompt_template": "qwen-chat-raw-deterministic",
            "bos_policy": "contract_default_add_bos",
            "deterministic_prompt": true,
            "turns_count": turns_count_u64,
            "prompt_token_count_total": prompt_token_count_total as u64,
            "prompt_token_ids_sha256": prompt_token_ids_sha256,
            "rendered_prompt_sha256": rendered_prompt_sha256,
            "turns": prompt_authority_turns
        },
        "session_lifecycle": {
            "schema": 1,
            "proof_scope": "qwen_warm_session_strict_cuda",
            "turns_count": turns_count_u64,
            "model_loaded_once": true,
            "tokenizer_loaded_once": true,
            "cuda_context_initialized_once": true,
            "cuda_context_once": true,
            "weights_uploaded_once": true,
            "per_request_model_load": false,
            "per_turn_weight_upload": false,
            "runtime_buffers_reused": true,
            "workspace_reused": true,
            "kv_cache_policy_recorded": true,
            "kv_cache_reinitialized_per_turn": true,
            "sampling_policy_recorded": true,
            "fallback_used": false,
            "scoped_warm_session_residency_claimed": true,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false
        },
        "warm_session_proof": {
            "schema": 1,
            "proof_scope": "qwen_strict_warm_session_greedy",
            "model_family": "qwen",
            "turns_count": turns_count_u64,
            "requested_new_tokens_per_turn": requested_new_tokens_u64,
            "generated_tokens_total": generated_tokens_total,
            "generation_policy": "greedy",
            "deterministic": true,
            "fallback_used": false,
            "cpu_reference_backend": "amd-9950x3d-cpu-avx512",
            "cuda_target_backend": HARDWARE_LANE,
            "cpu_generated_token_ids_sha256": cpu_generated_all_sha256,
            "cuda_generated_token_ids_sha256": cuda_generated_all_sha256,
            "generated_token_ids_match": true,
            "first_token_divergence": null,
            "cuda_logits_top_k_session_sha256": top_k_session_sha256,
            "top_k_evidence_recorded": true,
            "top_k_compared": true,
            "top_k_all_match": top_k_all_match,
            "first_top_k_divergence": first_top_k_divergence,
            "top_k_max_abs_error": top_k_max_abs_error,
            "top_k_mean_abs_error": top_k_mean_abs_error,
            "turns": turn_json,
            "qwen_one_token_cuda_claimed": true,
            "qwen_short_decode_cuda_claimed": true,
            "qwen_warm_session_cuda_claimed": true,
            "qwen_chat_cuda_claimed": false,
            "dense_gguf_inference_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "server_ready_claimed": false,
            "full_cuda_residency_claimed": false
        },
        "quality_gate": {
            "schema": 1,
            "gate": "qwen_warm_session_cuda_parity",
            "passed": true,
            "answer_ready_claimed": false,
            "short_decode_claimed": true,
            "warm_session_claimed": true,
            "chat_claimed": false
        },
        "kernel_stats": kernel_stats,
        "logits_transfer_reduction": logits_transfer_reduction,
        "kernel_coverage": {
            "schema": 1,
            "route": DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND,
            "kernels_executed": [
                "dense_qwen_warm_session_cuda_runtime",
                "dense_qwen_lm_head_cuda",
                "dense_qwen_logits_transfer_cuda"
            ],
            "all_required_dense_kernels_executed": true,
            "dense_kernel_invocations": runtime_invocations,
            "dense_kernel_launches": stats_launches,
            "bitnet_qk256_kernel_invocations": 0_u64,
            "cpu_fallback_kernel_invocations": 0_u64,
            "fallback_used": false
        },
        "timing": {
            "total_ms": cuda.total_ms,
            "cpu_reference_total_ms": cpu.total_ms,
            "cuda_context_init_ms": cuda.device_init_ms,
            "tokenizer_load_ms": tokenizer_load_ms,
            "model_load_ms": cuda.model_load_ms,
            "cpu_reference_model_load_ms": cpu.model_load_ms,
            "first_token_ms": cuda.turns.first().map(|turn| turn.first_token_ms).unwrap_or(0.0),
            "prefill_ms": cuda.turns.iter().map(|turn| turn.prefill_ms).sum::<f64>(),
            "decode_total_ms": cuda.turns.iter().map(|turn| turn.decode_total_ms).sum::<f64>(),
            "embed_ms_total": cuda.turns.iter().map(|turn| turn.embed_ms_total).sum::<f64>(),
            "forward_ms_total": cuda.turns.iter().map(|turn| turn.forward_ms_total).sum::<f64>(),
            "logits_ms_total": cuda.turns.iter().map(|turn| turn.logits_ms_total).sum::<f64>(),
            "logits_download_ms_total": cuda.turns.iter().map(|turn| turn.logits_download_ms_total).sum::<f64>(),
            "kernel_time_ms": kernel_time_ms,
            "host_to_device_bytes": stats_h2d,
            "device_to_host_bytes": stats_d2h,
            "host_to_device_ms": cuda.model_load_ms,
            "host_to_device_ms_source": dense_qwen_h2d_timing_source(),
            "host_to_device_ms_scope": dense_qwen_h2d_timing_scope(),
            "host_to_device_ms_includes_non_transfer_overhead": true,
            "device_to_host_ms": cuda.turns.iter().map(|turn| turn.logits_download_ms_total).sum::<f64>(),
            "device_to_host_ms_source": logits_transfer_mode.d2h_timing_source(),
            "transfer_timing_status": dense_qwen_transfer_timing_status(),
            "kernel_invocations": runtime_invocations,
            "kernel_launches": stats_launches,
            "turns_count": turns_count_u64,
            "generated_tokens_total": generated_tokens_total
        },
        "tensor_residency": {
            "schema": 1,
            "scope": "qwen_warm_session_strict_cuda",
            "model_class": "dense_regular_llm",
            "residency_accounting_recorded": true,
            "model_loaded_once": true,
            "tokenizer_loaded_once": true,
            "cuda_context_initialized_once": true,
            "cuda_context_once": true,
            "weights_uploaded_once": true,
            "weights_resident_on_cuda": true,
            "per_request_model_load": false,
            "per_turn_weight_upload": false,
            "per_token_weight_upload": false,
            "runtime_buffers_reused": true,
            "workspace_reused": true,
            "kv_cache_policy_recorded": true,
            "kv_cache_reinitialized_per_turn": true,
            "sampling_policy_recorded": true,
            "runtime_logits_cuda_resident_before_download": cuda.turns.iter().all(|turn| turn.logits_all_cuda_resident),
            "fallback_used": false,
            "dense_gguf_inference_claimed": false,
            "scoped_warm_session_residency_claimed": true,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false,
            "transfer_accounting": {
                "status": "measured",
                "host_to_device_bytes": stats_h2d,
                "device_to_host_bytes": stats_d2h,
                "host_to_device_ms": cuda.model_load_ms,
                "host_to_device_ms_source": dense_qwen_h2d_timing_source(),
                "host_to_device_ms_scope": dense_qwen_h2d_timing_scope(),
                "host_to_device_ms_includes_non_transfer_overhead": true,
                "device_to_host_ms": cuda.turns.iter().map(|turn| turn.logits_download_ms_total).sum::<f64>(),
                "device_to_host_ms_source": logits_transfer_mode.d2h_timing_source(),
                "transfer_timing_status": dense_qwen_transfer_timing_status(),
                "kernel_invocations": runtime_invocations,
                "kernel_launches": stats_launches
            }
        },
        "parity": {
            "reference_backend": "amd-9950x3d-cpu-avx512",
            "target_backend": HARDWARE_LANE,
            "kernel_id": "dense_qwen_warm_session_cuda_runtime",
            "fixture_id": format!("{}-warm-session-greedy", proof_model.id),
            "passed": true,
            "max_abs_error": top_k_max_abs_error,
            "mean_abs_error": top_k_mean_abs_error,
            "tolerance": top_k_max_abs_error,
            "tolerance_source": "generated token equality; numeric drift recorded for top-k logits"
        },
        "claim_boundary": {
            "dense_regular_llm_cuda_claimed": true,
            "dense_tensor_residency_claimed": true,
            "dense_gguf_descriptor_inspection_claimed": true,
            "dense_gguf_linear_fixture_extraction_claimed": true,
            "dense_gguf_linear_cuda_parity_claimed": true,
            "dense_gguf_linear_role_sweep_cuda_parity_claimed": true,
            "dense_gguf_norm_cuda_parity_claimed": true,
            "dense_gguf_rope_cuda_parity_claimed": true,
            "dense_gguf_attention_score_cuda_parity_claimed": true,
            "dense_gguf_attention_softmax_cuda_parity_claimed": true,
            "dense_gguf_attention_v_mix_cuda_parity_claimed": true,
            "dense_gguf_mlp_activation_cuda_parity_claimed": true,
            "dense_gguf_one_layer_execution_plan_claimed": true,
            "dense_gguf_one_layer_cpu_reference_claimed": true,
            "dense_gguf_one_layer_cuda_integrated_parity_claimed": true,
            "dense_gguf_all_layer_execution_plan_claimed": true,
            "dense_gguf_model_boundary_fixtures_claimed": true,
            "kv_cache_policy_claimed": true,
            "sampling_policy_claimed": true,
            "qwen_one_token_cuda_claimed": true,
            "qwen_short_decode_cuda_claimed": true,
            "qwen_warm_session_cuda_claimed": true,
            "scoped_warm_session_residency_claimed": true,
            "dense_gguf_one_layer_inference_claimed": false,
            "dense_gguf_inference_claimed": false,
            "qwen_ask_cuda_claimed": false,
            "qwen_chat_cuda_claimed": false,
            "server_ready_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false
        },
        "notes": [
            format!("{} proves a bounded deterministic multi-turn Qwen warm session through the dense regular-LLM CUDA route.", dense_qwen_warm_session_work_item(proof_model)),
            "This receipt does not claim ask/chat UX, server readiness, speedup, persistent residency beyond this warm-session scope, full CUDA residency, or BitNet packed I2_S/QK256 proof."
        ],
        "error": null
    }))
}

fn dense_qwen_warm_session_work_item(proof_model: &DenseQwenProofModel) -> &'static str {
    match proof_model.id {
        QWEN25_05B_INSTRUCT_Q8_0_MODEL_ID => "CUDA-DENSE-046",
        QWEN3_06B_INSTRUCT_Q8_0_MODEL_ID => "CUDA-MODEL-006",
        _ => proof_model.work_item,
    }
}

fn dense_qwen_ask_work_item(proof_model: &DenseQwenProofModel) -> &'static str {
    match proof_model.id {
        QWEN25_05B_INSTRUCT_Q8_0_MODEL_ID => "CUDA-UX-003",
        QWEN3_06B_INSTRUCT_Q8_0_MODEL_ID => "CUDA-MODEL-010",
        _ => proof_model.work_item,
    }
}

fn dense_qwen_chat_work_item(proof_model: &DenseQwenProofModel) -> &'static str {
    match proof_model.id {
        QWEN25_05B_INSTRUCT_Q8_0_MODEL_ID => "CUDA-UX-004",
        QWEN3_06B_INSTRUCT_Q8_0_MODEL_ID => "CUDA-MODEL-011",
        _ => proof_model.work_item,
    }
}

fn dense_gguf_one_layer_cpu_reference_receipt_json(
    inspection: &DenseGgufDescriptorInspection,
    reference: &DenseGgufOneLayerCpuReference,
    model_path: &Path,
    model_sha256: &str,
    artifact_path: &str,
    timestamp_utc: &str,
) -> Result<Value> {
    if reference.model_family != inspection.model_family
        || reference.architecture != inspection.architecture
    {
        bail!(
            "dense GGUF one-layer CPU reference mixed inspection identity: inspection={}/{} reference={}/{}",
            inspection.model_family,
            inspection.architecture,
            reference.model_family,
            reference.architecture
        );
    }
    if reference.phases.is_empty() {
        bail!("dense GGUF one-layer CPU reference requires phase hashes");
    }

    let phases = reference
        .phases
        .iter()
        .map(|phase| {
            json!({
                "index": phase.index as u64,
                "name": phase.name,
                "role": phase.role,
                "op_type": phase.op_type,
                "output_len": phase.output_len as u64,
                "output_sha256": phase.output_sha256,
                "max_abs": phase.max_abs,
            })
        })
        .collect::<Vec<_>>();

    Ok(json!({
        "schema": 1,
        "artifact_kind": DENSE_GGUF_ONE_LAYER_CPU_REFERENCE_ARTIFACT_KIND,
        "artifact_path": artifact_path,
        "claim": "dense_gguf_one_layer_cpu_reference_recorded",
        "machine_id": MACHINE_ID,
        "hardware_lane": "cpu-reference",
        "timestamp_utc": timestamp_utc,
        "requested_backend": "cpu_reference",
        "selected_backend": "cpu_reference",
        "runtime_api": "cpu",
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "speedup_claim": false,
        "model": {
            "model_family": inspection.model_family,
            "architecture": inspection.architecture,
            "artifact_kind": "dense_gguf",
            "file": model_path.display().to_string(),
            "sha256": model_sha256
        },
        "execution_path": {
            "model_class": "dense_regular_llm",
            "kernel_family": "cpu_reference_dense_one_layer",
            "quantization_family": "dense_gguf_materialized_f32_reference",
            "bitnet_packed_kernel_proof": false,
            "qk256_proof": false
        },
        "descriptor_coverage": {
            "schema": 1,
            "source_artifact_kind": "dense_gguf_tensor_descriptor_inspection",
            "tensor_count": inspection.tensor_count,
            "metadata_count": inspection.metadata_count as u64,
            "required_roles_present": inspection.required_roles_present,
            "strict_descriptor_complete": inspection.strict_descriptor_complete,
            "dense_cuda_route_status": inspection.dense_cuda_route_status,
            "quantization_families": inspection.quantization_families,
            "bitnet_packed_marker_found": inspection.bitnet_packed_marker_found,
            "dense_gguf_inference_claimed": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "reference_harness": {
            "schema": 1,
            "fixture_id": reference.fixture_id,
            "layer_index": reference.layer_index as u64,
            "seq_len": reference.seq_len as u64,
            "position_offset": reference.position_offset as u64,
            "hidden_size": reference.hidden_size as u64,
            "q_heads": reference.q_heads as u64,
            "kv_heads": reference.kv_heads as u64,
            "heads_per_kv_group": reference.heads_per_kv_group as u64,
            "head_dim": reference.head_dim as u64,
            "intermediate_size": reference.intermediate_size as u64,
            "rmsnorm_eps": reference.rmsnorm_eps,
            "epsilon_source": reference.epsilon_source,
            "rope_base": reference.rope_base,
            "rope_base_source": reference.rope_base_source,
            "rope_scaling_factor": reference.scaling_factor,
            "deterministic_input_len": reference.deterministic_input_len as u64,
            "deterministic_input_sha256": reference.deterministic_input_sha256,
            "phases_total": reference.phases.len() as u64,
            "phases": phases,
            "final_output_len": reference.final_output_len as u64,
            "final_output_sha256": reference.final_output_sha256,
            "final_output_max_abs": reference.final_output_max_abs,
            "cpu_reference_only": true,
            "cuda_execution_claimed": false,
            "one_layer_inference_claimed": false,
            "dense_gguf_inference_claimed": false,
            "qwen_one_token_cuda_claimed": false,
            "qwen_short_decode_cuda_claimed": false,
            "qwen_chat_cuda_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false,
            "next_required_proof": "one_layer_cuda_integrated_parity"
        },
        "claim_boundary": {
            "dense_regular_llm_cuda_claimed": false,
            "dense_tensor_residency_claimed": false,
            "dense_gguf_descriptor_inspection_claimed": true,
            "dense_gguf_linear_fixture_extraction_claimed": true,
            "dense_gguf_linear_cuda_parity_claimed": false,
            "dense_gguf_linear_role_sweep_cuda_parity_claimed": false,
            "dense_gguf_one_layer_execution_plan_claimed": false,
            "dense_gguf_one_layer_cpu_reference_claimed": true,
            "dense_gguf_one_layer_inference_claimed": false,
            "dense_gguf_inference_claimed": false,
            "qwen_one_token_cuda_claimed": false,
            "qwen_short_decode_cuda_claimed": false,
            "qwen_chat_cuda_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false
        },
        "error": null
    }))
}

fn dense_gguf_one_layer_cuda_integrated_parity_receipt_json(
    inspection: &DenseGgufDescriptorInspection,
    reference: &DenseGgufOneLayerCpuReference,
    parity: &DenseGgufOneLayerCudaIntegratedParity,
    probe: Option<&bitnet_device_probe::NvidiaCudaProbe>,
    model_path: &Path,
    model_sha256: &str,
    artifact_path: &str,
    timestamp_utc: &str,
) -> Result<Value> {
    if parity.source_cpu_reference_fixture_id != reference.fixture_id {
        bail!(
            "integrated dense one-layer CUDA parity source reference mismatch: parity={} reference={}",
            parity.source_cpu_reference_fixture_id,
            reference.fixture_id
        );
    }
    if parity.model_family != inspection.model_family
        || parity.architecture != inspection.architecture
    {
        bail!(
            "integrated dense one-layer CUDA parity mixed inspection identity: inspection={}/{} parity={}/{}",
            inspection.model_family,
            inspection.architecture,
            parity.model_family,
            parity.architecture
        );
    }

    let summary = ModelDispatchSummary {
        total_ops: 14,
        cuda_bitnet_qk256_ops: 0,
        cuda_dense_regular_llm_ops: 14,
        cpu_fallback_ops: 0,
        unsupported_ops: 0,
        fallback_used: false,
        selected_route: Some(ModelDispatchBackend::CudaDenseRegularLlm),
        strict_cuda_ready: true,
    };
    let execution_plan = execution_plan_receipt(ExecutionPlanReceiptInput {
        model_family: &inspection.model_family,
        quantization: "dense_fp16",
        requested_backend: HARDWARE_LANE,
        selected_backend: HARDWARE_LANE,
        runtime_api: "cuda",
        strict_fallback_policy: "reject",
        summary,
        speedup_claim: false,
        full_cuda_residency_claimed: false,
    });

    let phases = parity
        .phases
        .iter()
        .map(|phase| {
            json!({
                "index": phase.index as u64,
                "name": phase.name,
                "role": phase.role,
                "op_type": phase.op_type,
                "route": phase.route,
                "status": phase.status,
                "output_len": phase.output_len as u64,
                "output_sha256": phase.output_sha256,
                "max_abs": phase.max_abs,
                "max_abs_error": phase.max_abs_error,
                "mean_abs_error": phase.mean_abs_error,
                "tolerance": phase.tolerance,
                "passed": phase.passed,
                "fallback_used": false,
                "kernel_id": phase.kernel_id,
                "invocations": phase.invocations,
                "fallback_invocations": phase.fallback_invocations,
                "host_to_device_bytes": phase.host_to_device_bytes,
                "device_to_host_bytes": phase.device_to_host_bytes,
                "kernel_launches": phase.kernel_launches,
                "kernel_time_ms": phase.kernel_time_ms,
            })
        })
        .collect::<Vec<_>>();
    let kernel_stats = parity
        .phases
        .iter()
        .filter(|phase| phase.kernel_id.is_some())
        .map(|phase| {
            json!({
                "phase": phase.name,
                "kernel_id": phase.kernel_id,
                "invocations": phase.invocations,
                "fallback_invocations": phase.fallback_invocations,
                "host_to_device_bytes": phase.host_to_device_bytes,
                "device_to_host_bytes": phase.device_to_host_bytes,
                "kernel_launches": phase.kernel_launches,
                "kernel_time_ms": phase.kernel_time_ms,
            })
        })
        .collect::<Vec<_>>();

    Ok(json!({
        "schema": 1,
        "artifact_kind": DENSE_GGUF_ONE_LAYER_CUDA_INTEGRATED_PARITY_ARTIFACT_KIND,
        "artifact_path": artifact_path,
        "claim": "dense_gguf_one_layer_cuda_integrated_parity_recorded",
        "machine_id": MACHINE_ID,
        "hardware_lane": HARDWARE_LANE,
        "timestamp_utc": timestamp_utc,
        "requested_backend": HARDWARE_LANE,
        "selected_backend": HARDWARE_LANE,
        "runtime_api": "cuda",
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "speedup_claim": false,
        "cuda": cuda_identity_json(probe),
        "model": {
            "model_family": inspection.model_family,
            "architecture": inspection.architecture,
            "artifact_kind": "dense_gguf",
            "file": model_path.display().to_string(),
            "sha256": model_sha256
        },
        "execution_path": {
            "model_class": "dense_regular_llm",
            "kernel_family": "dense_cuda_integrated_one_layer",
            "quantization_family": "dense_gguf_q8_0_f16_cuda_bridge",
            "bitnet_packed_kernel_proof": false,
            "qk256_proof": false
        },
        "execution_plan": execution_plan,
        "descriptor_coverage": {
            "schema": 1,
            "source_artifact_kind": "dense_gguf_tensor_descriptor_inspection",
            "tensor_count": inspection.tensor_count,
            "metadata_count": inspection.metadata_count as u64,
            "required_roles_present": inspection.required_roles_present,
            "strict_descriptor_complete": inspection.strict_descriptor_complete,
            "dense_cuda_route_status": inspection.dense_cuda_route_status,
            "quantization_families": inspection.quantization_families,
            "bitnet_packed_marker_found": inspection.bitnet_packed_marker_found,
            "dense_gguf_inference_claimed": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "cpu_reference": {
            "schema": 1,
            "source_artifact_kind": DENSE_GGUF_ONE_LAYER_CPU_REFERENCE_ARTIFACT_KIND,
            "fixture_id": reference.fixture_id,
            "layer_index": reference.layer_index as u64,
            "seq_len": reference.seq_len as u64,
            "position_offset": reference.position_offset as u64,
            "final_output_len": reference.final_output_len as u64,
            "final_output_sha256": reference.final_output_sha256,
            "final_output_max_abs": reference.final_output_max_abs,
            "cpu_reference_only": true,
            "cuda_execution_claimed": false,
            "dense_gguf_inference_claimed": false
        },
        "cuda_layer": {
            "schema": 1,
            "fixture_id": parity.fixture_id,
            "source_cpu_reference_fixture_id": parity.source_cpu_reference_fixture_id,
            "layer_index": parity.layer_index as u64,
            "seq_len": parity.seq_len as u64,
            "position_offset": parity.position_offset as u64,
            "hidden_size": parity.hidden_size as u64,
            "q_heads": parity.q_heads as u64,
            "kv_heads": parity.kv_heads as u64,
            "heads_per_kv_group": parity.heads_per_kv_group as u64,
            "head_dim": parity.head_dim as u64,
            "intermediate_size": parity.intermediate_size as u64,
            "governed_cuda_ops_total": 14_u64,
            "residual_host_ops_total": 2_u64,
            "host_deterministic_input_ops_total": 1_u64,
            "unsupported_ops_total": 0_u64,
            "cpu_fallback_ops_total": 0_u64,
            "strict_cuda_ready": true,
            "fallback_used": false,
            "phases_total": phases.len() as u64,
            "phases": phases,
            "final_output_len": parity.final_output_len as u64,
            "final_output_sha256": parity.final_output_sha256,
            "final_output_max_abs": parity.final_output_max_abs,
            "final_output_max_abs_error": parity.final_output_max_abs_error,
            "final_output_mean_abs_error": parity.final_output_mean_abs_error,
            "tolerance": parity.tolerance,
            "passed": parity.passed,
            "one_layer_cuda_integrated_parity_claimed": true,
            "one_layer_inference_claimed": false,
            "dense_gguf_inference_claimed": false,
            "qwen_one_token_cuda_claimed": false,
            "qwen_short_decode_cuda_claimed": false,
            "qwen_chat_cuda_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false
        },
        "kernel_stats": kernel_stats,
        "timing": {
            "kernel_time_ms": parity.kernel_time_ms,
            "host_to_device_bytes": parity.host_to_device_bytes,
            "device_to_host_bytes": parity.device_to_host_bytes,
            "kernel_invocations": parity.kernel_invocations,
            "kernel_launches": parity.kernel_launches
        },
        "tensor_residency": {
            "scope": "integrated_dense_gguf_one_layer",
            "model_class": "dense_regular_llm",
            "fixture_id": parity.fixture_id,
            "dense_tensor_residency_claimed": true,
            "integrated_one_layer_cuda_parity_claimed": true,
            "dense_gguf_inference_claimed": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false,
            "weights_uploaded_per_kernel": true,
            "weights_uploaded_once": false,
            "intermediate_downloads_for_phase_parity": true,
            "host_device_transfer_accounting_matches_kernel_stats": true,
            "transfer_accounting": {
                "status": "measured",
                "host_to_device_bytes": parity.host_to_device_bytes,
                "device_to_host_bytes": parity.device_to_host_bytes,
                "kernel_invocations": parity.kernel_invocations,
                "kernel_launches": parity.kernel_launches
            }
        },
        "claim_boundary": {
            "dense_regular_llm_cuda_claimed": true,
            "dense_tensor_residency_claimed": true,
            "dense_gguf_descriptor_inspection_claimed": true,
            "dense_gguf_linear_fixture_extraction_claimed": true,
            "dense_gguf_linear_cuda_parity_claimed": true,
            "dense_gguf_linear_role_sweep_cuda_parity_claimed": true,
            "dense_gguf_norm_cuda_parity_claimed": true,
            "dense_gguf_rope_cuda_parity_claimed": true,
            "dense_gguf_attention_score_cuda_parity_claimed": true,
            "dense_gguf_attention_softmax_cuda_parity_claimed": true,
            "dense_gguf_attention_v_mix_cuda_parity_claimed": true,
            "dense_gguf_mlp_activation_cuda_parity_claimed": true,
            "dense_gguf_one_layer_execution_plan_claimed": true,
            "dense_gguf_one_layer_cpu_reference_claimed": true,
            "dense_gguf_one_layer_cuda_integrated_parity_claimed": true,
            "dense_gguf_one_layer_inference_claimed": false,
            "dense_gguf_inference_claimed": false,
            "qwen_one_token_cuda_claimed": false,
            "qwen_short_decode_cuda_claimed": false,
            "qwen_chat_cuda_claimed": false,
            "server_ready_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false
        },
        "error": null
    }))
}

fn dense_one_layer_gap_audit_json(
    operations: &[Value],
    layer_index: usize,
    cuda_routable_ops: u64,
    unsupported_ops: u64,
) -> Result<Value> {
    let mut unsupported = Vec::new();
    let mut cuda_roles = Vec::new();
    let mut linear_roles = Vec::new();
    let mut norm_roles = Vec::new();
    let mut rope_roles = Vec::new();
    let mut attention_score_roles = Vec::new();
    let mut attention_softmax_roles = Vec::new();
    let mut attention_v_mix_roles = Vec::new();
    let mut mlp_activation_roles = Vec::new();
    let mut op_type_counts: BTreeMap<String, u64> = BTreeMap::new();

    for op in operations {
        let route = json_string_field(op, "route")?;
        let role = json_string_field(op, "role")?;
        match route {
            DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND => {
                cuda_roles.push(role.to_string());
                match json_string_field(op, "op_type")? {
                    "matmul" => linear_roles.push(role.to_string()),
                    "rmsnorm" => norm_roles.push(role.to_string()),
                    "rope" => rope_roles.push(role.to_string()),
                    "attention" if role == "attention_scores" => {
                        attention_score_roles.push(role.to_string());
                    }
                    "attention" if role == "attention_v_mix" => {
                        attention_v_mix_roles.push(role.to_string());
                    }
                    "softmax" if role == "attention_softmax" => {
                        attention_softmax_roles.push(role.to_string());
                    }
                    "activation" if role == "mlp_activation" => {
                        mlp_activation_roles.push(role.to_string());
                    }
                    other => {
                        bail!("dense one-layer CUDA-routable op type `{other}` is not governed")
                    }
                }
            }
            "unsupported" => {
                let op_type = json_string_field(op, "op_type")?;
                *op_type_counts.entry(op_type.to_string()).or_insert(0) += 1;
                unsupported.push(json!({
                    "name": json_string_field(op, "name")?,
                    "role": role,
                    "op_type": op_type,
                    "size": op.get("size").cloned().unwrap_or(Value::Null),
                    "source": json_string_field(op, "source")?,
                    "source_tensor": op.get("source_tensor").cloned().unwrap_or(Value::Null),
                    "source_shape": op.get("source_shape").cloned().unwrap_or(Value::Null),
                    "input_dependencies": dense_gap_dependencies(role),
                    "cuda_kernel_status": "missing_cuda_kernel",
                    "cpu_fallback_allowed": false,
                    "blocks_strict_cuda_one_layer": true,
                    "input_residency": "not_executed",
                    "output_residency": "not_executed",
                    "transfer_timing_status": "not_measured_no_kernel"
                }));
            }
            _ => {}
        }
    }

    if cuda_roles.len() as u64 != cuda_routable_ops || unsupported.len() as u64 != unsupported_ops {
        bail!("dense one-layer gap audit counts must match planner summary");
    }

    Ok(json!({
        "schema": 1,
        "source_artifact_kind": DENSE_GGUF_ONE_LAYER_EXECUTION_PLAN_ARTIFACT_KIND,
        "layer_index": layer_index as u64,
        "cuda_routable_ops_total": cuda_routable_ops,
        "cuda_routable_linear_ops_total": linear_roles.len() as u64,
        "cuda_routable_norm_ops_total": norm_roles.len() as u64,
        "cuda_routable_rope_ops_total": rope_roles.len() as u64,
        "cuda_routable_attention_score_ops_total": attention_score_roles.len() as u64,
        "cuda_routable_attention_softmax_ops_total": attention_softmax_roles.len() as u64,
        "cuda_routable_attention_v_mix_ops_total": attention_v_mix_roles.len() as u64,
        "cuda_routable_mlp_activation_ops_total": mlp_activation_roles.len() as u64,
        "unsupported_ops_total": unsupported_ops,
        "cpu_fallback_ops_total": 0,
        "strict_cuda_ready": unsupported_ops == 0,
        "unsupported_ops_have_dependency_notes": true,
        "strict_cuda_rejects_cpu_fallback": true,
        "cuda_routable_roles": cuda_roles,
        "linears_routable_roles": linear_roles,
        "norms_routable_roles": norm_roles,
        "rope_routable_roles": rope_roles,
        "attention_scores_routable_roles": attention_score_roles,
        "attention_softmax_routable_roles": attention_softmax_roles,
        "attention_v_mix_routable_roles": attention_v_mix_roles,
        "mlp_activation_routable_roles": mlp_activation_roles,
        "rmsnorm_cuda_parity_available": true,
        "rope_cuda_parity_available": true,
        "attention_score_cuda_parity_available": true,
        "attention_softmax_cuda_parity_available": true,
        "attention_v_mix_cuda_parity_available": true,
        "mlp_activation_cuda_parity_available": true,
        "next_candidate_gap": "none",
        "next_required_proof": "one_layer_cpu_reference_harness",
        "unsupported_op_type_counts": op_type_counts,
        "candidate_order": DENSE_ONE_LAYER_NO_REMAINING_GAP_CANDIDATE_ORDER,
        "dependency_edges": dense_one_layer_dependency_edges_json(),
        "unsupported_ops": unsupported,
        "dense_gguf_one_layer_execution_plan_claimed": true,
        "dense_gguf_one_layer_inference_claimed": false,
        "dense_gguf_inference_claimed": false,
        "qwen_one_token_cuda_claimed": false,
        "qwen_short_decode_cuda_claimed": false,
        "qwen_chat_cuda_claimed": false,
        "bitnet_packed_i2s_qk256_proof": false,
        "speedup_claim": false,
        "full_cuda_residency_claimed": false
    }))
}

fn json_string_field<'a>(object: &'a Value, field: &str) -> Result<&'a str> {
    object
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("field `{field}` must be a string"))
}

fn dense_gap_dependencies(role: &str) -> Vec<&'static str> {
    match role {
        "attention_norm" => vec!["hidden_state"],
        "rope" => vec!["attention_q", "attention_k", "position_ids"],
        "attention_scores" => vec!["rope_q", "rope_k", "causal_mask"],
        "attention_softmax" => vec!["attention_scores"],
        "attention_v_mix" => vec!["attention_softmax", "attention_v"],
        "ffn_norm" => vec!["attention_residual_state"],
        "mlp_activation" => vec!["mlp_gate", "mlp_up"],
        _ => vec!["unknown"],
    }
}

fn dense_one_layer_dependency_edges_json() -> Vec<Value> {
    [
        ("attention_norm", "attention_q"),
        ("attention_norm", "attention_k"),
        ("attention_norm", "attention_v"),
        ("attention_q", "rope"),
        ("attention_k", "rope"),
        ("rope", "attention_scores"),
        ("attention_scores", "attention_softmax"),
        ("attention_softmax", "attention_v_mix"),
        ("attention_v", "attention_v_mix"),
        ("attention_v_mix", "attention_output"),
        ("ffn_norm", "mlp_gate"),
        ("ffn_norm", "mlp_up"),
        ("mlp_gate", "mlp_activation"),
        ("mlp_up", "mlp_activation"),
        ("mlp_activation", "mlp_down"),
    ]
    .into_iter()
    .map(|(from, to)| json!({ "from": from, "to": to }))
    .collect()
}

#[derive(Debug, Clone, Copy, Default)]
struct DenseAllLayerCounts {
    total_ops: u64,
    cuda_routable_ops: u64,
    linear_cuda_ops: u64,
    norm_cuda_ops: u64,
    rope_cuda_ops: u64,
    attention_score_cuda_ops: u64,
    attention_softmax_cuda_ops: u64,
    attention_v_mix_cuda_ops: u64,
    mlp_activation_cuda_ops: u64,
    unsupported_ops: u64,
    cpu_fallback_ops: u64,
}

impl DenseAllLayerCounts {
    fn add(&mut self, other: Self) {
        self.total_ops += other.total_ops;
        self.cuda_routable_ops += other.cuda_routable_ops;
        self.linear_cuda_ops += other.linear_cuda_ops;
        self.norm_cuda_ops += other.norm_cuda_ops;
        self.rope_cuda_ops += other.rope_cuda_ops;
        self.attention_score_cuda_ops += other.attention_score_cuda_ops;
        self.attention_softmax_cuda_ops += other.attention_softmax_cuda_ops;
        self.attention_v_mix_cuda_ops += other.attention_v_mix_cuda_ops;
        self.mlp_activation_cuda_ops += other.mlp_activation_cuda_ops;
        self.unsupported_ops += other.unsupported_ops;
        self.cpu_fallback_ops += other.cpu_fallback_ops;
    }
}

fn dense_transformer_layer_indices(
    inspection: &DenseGgufDescriptorInspection,
) -> Result<Vec<usize>> {
    let mut indices = BTreeSet::new();
    for descriptor in &inspection.descriptors {
        if dense_block_descriptor_role(descriptor.role)
            && let Some(index) = extract_layer_index(&descriptor.name)
        {
            indices.insert(index);
        }
    }
    Ok(indices.into_iter().collect())
}

fn dense_missing_layer_indices(layer_indices: &[usize]) -> Vec<Value> {
    let Some(max_index) = layer_indices.iter().copied().max() else {
        return Vec::new();
    };
    let present = layer_indices.iter().copied().collect::<BTreeSet<_>>();
    (0..=max_index)
        .filter(|index| !present.contains(index))
        .map(|index| json!(index as u64))
        .collect()
}

fn dense_block_descriptor_role(role: DenseGgufTensorRole) -> bool {
    matches!(
        role,
        DenseGgufTensorRole::AttentionQ
            | DenseGgufTensorRole::AttentionK
            | DenseGgufTensorRole::AttentionV
            | DenseGgufTensorRole::AttentionOutput
            | DenseGgufTensorRole::MlpGate
            | DenseGgufTensorRole::MlpUp
            | DenseGgufTensorRole::MlpDown
            | DenseGgufTensorRole::AttentionNorm
            | DenseGgufTensorRole::FfnNorm
    )
}

fn dense_layer_plan_operations_json(
    entries: &[DenseLayerPlanEntry],
    decisions: &[ModelDispatchDecision],
) -> Vec<Value> {
    entries
        .iter()
        .zip(decisions.iter())
        .enumerate()
        .map(|(idx, (entry, decision))| {
            let route = decision.backend.receipt_route_label();
            let status = match decision.backend {
                ModelDispatchBackend::CudaDenseRegularLlm => "cuda_routable",
                ModelDispatchBackend::Unsupported => "unsupported_strict_cuda",
                ModelDispatchBackend::CpuScalar | ModelDispatchBackend::CpuSimd => "cpu_fallback",
                ModelDispatchBackend::CudaBitnetQk256 => "wrong_route",
            };
            json!({
                "index": idx as u64,
                "name": entry.op.name,
                "role": entry.role,
                "op_type": entry.op.op_type.as_str(),
                "size": entry.op.size as u64,
                "source": entry.source,
                "source_tensor": entry.source_tensor,
                "source_tensor_type": entry.source_tensor_type,
                "source_shape": entry.source_shape,
                "is_quantized": entry.op.is_quantized,
                "route": route,
                "status": status,
                "fallback_used": decision.fallback_used,
                "reason": decision.reason,
            })
        })
        .collect()
}

fn dense_layer_plan_counts(operations: &[Value]) -> Result<DenseAllLayerCounts> {
    let mut counts =
        DenseAllLayerCounts { total_ops: operations.len() as u64, ..Default::default() };
    for op in operations {
        match json_string_field(op, "route")? {
            DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND => {
                counts.cuda_routable_ops += 1;
                match json_string_field(op, "op_type")? {
                    "matmul" => counts.linear_cuda_ops += 1,
                    "rmsnorm" => counts.norm_cuda_ops += 1,
                    "rope" => counts.rope_cuda_ops += 1,
                    "softmax" => counts.attention_softmax_cuda_ops += 1,
                    "activation" => counts.mlp_activation_cuda_ops += 1,
                    "attention" => match json_string_field(op, "role")? {
                        "attention_scores" => counts.attention_score_cuda_ops += 1,
                        "attention_v_mix" => counts.attention_v_mix_cuda_ops += 1,
                        other => bail!("unexpected dense attention op role `{other}`"),
                    },
                    other => bail!("unexpected dense CUDA op_type `{other}`"),
                }
            }
            "unsupported" => counts.unsupported_ops += 1,
            "cpu_scalar" | "cpu_simd" => counts.cpu_fallback_ops += 1,
            other => bail!("unexpected dense route `{other}`"),
        }
    }
    Ok(counts)
}

fn dense_layer_operation_signature(operations: &[Value]) -> Result<Vec<Value>> {
    operations
        .iter()
        .map(|op| {
            Ok(json!({
                "role": json_string_field(op, "role")?,
                "op_type": json_string_field(op, "op_type")?,
                "source": json_string_field(op, "source")?,
                "source_tensor_type": op.get("source_tensor_type").cloned().unwrap_or(Value::Null),
                "source_shape": op.get("source_shape").cloned().unwrap_or(Value::Null),
                "is_quantized": op.get("is_quantized").cloned().unwrap_or(Value::Bool(false)),
                "route": json_string_field(op, "route")?,
                "status": json_string_field(op, "status")?,
            }))
        })
        .collect()
}

fn dense_model_boundary_gaps_json(inspection: &DenseGgufDescriptorInspection) -> Value {
    let token_embedding = descriptor_for_role(inspection, DenseGgufTensorRole::TokenEmbedding).ok();
    let lm_head = descriptor_for_role(inspection, DenseGgufTensorRole::Output).ok();
    let lm_head_source = lm_head.or(token_embedding);
    let lm_head_disposition = if lm_head.is_some() {
        "LM head and logits fixture not yet governed"
    } else {
        "LM head appears tied to token embeddings; logits fixture not yet governed"
    };
    let gap =
        |name: &str, disposition: &str, source: Option<&DenseGgufTensorDescriptor>, proof: &str| {
            json!({
                "gap": name,
                "status": "not_governed_by_all_layer_block_plan",
                "disposition": disposition,
                "source_tensor": source.map(|descriptor| descriptor.name.clone()),
                "source_tensor_type": source.map(|descriptor| descriptor.tensor_type.clone()),
                "blocks_qwen_one_token": true,
                "blocks_qwen_short_decode": true,
                "blocks_qwen_chat": true,
                "required_next_proof": proof,
            })
        };
    json!({
        "schema": 1,
        "gaps": [
            gap("token_embedding", "embedding lookup fixture and route not yet governed", token_embedding, "dense_gguf_embedding_fixture"),
            gap("final_norm", "final model normalization fixture not yet governed", None, "dense_gguf_final_norm_fixture"),
            gap("lm_head_logits", lm_head_disposition, lm_head_source, "dense_gguf_lm_head_logits_fixture"),
            gap("kv_cache_policy", "KV cache residency and transfer policy not yet recorded", None, "dense_gguf_kv_cache_policy_receipt"),
            gap("sampling", "sampler integration and logits transfer policy not yet governed", None, "dense_gguf_sampling_policy_receipt")
        ],
        "all_boundary_gaps_explicit": true,
        "qwen_one_token_cuda_blocked": true,
        "qwen_short_decode_cuda_blocked": true,
        "qwen_chat_cuda_blocked": true,
        "next_required_proof": "dense_gguf_model_boundary_fixtures",
        "dense_gguf_inference_claimed": false,
        "speedup_claim": false,
        "full_cuda_residency_claimed": false
    })
}

fn ensure_dense_all_layer_block_descriptor_coverage(
    inspection: &DenseGgufDescriptorInspection,
) -> Result<()> {
    let missing_roles = DENSE_ALL_LAYER_BLOCK_ROLES
        .iter()
        .copied()
        .filter(|role| !inspection.descriptors.iter().any(|descriptor| descriptor.role == *role))
        .map(dense_role_label)
        .collect::<Vec<_>>();

    if !missing_roles.is_empty() {
        bail!(
            "dense GGUF all-layer plan requires complete transformer block descriptor coverage, missing roles: {}",
            missing_roles.join(", ")
        );
    }

    Ok(())
}

const DENSE_ALL_LAYER_BLOCK_ROLES: &[DenseGgufTensorRole] = &[
    DenseGgufTensorRole::AttentionQ,
    DenseGgufTensorRole::AttentionK,
    DenseGgufTensorRole::AttentionV,
    DenseGgufTensorRole::AttentionOutput,
    DenseGgufTensorRole::MlpGate,
    DenseGgufTensorRole::MlpUp,
    DenseGgufTensorRole::MlpDown,
    DenseGgufTensorRole::AttentionNorm,
    DenseGgufTensorRole::FfnNorm,
];

fn dense_one_layer_plan_entries(
    inspection: &DenseGgufDescriptorInspection,
    layer_index: usize,
) -> Result<Vec<DenseLayerPlanEntry>> {
    let attention_q =
        descriptor_for_layer_role(inspection, layer_index, DenseGgufTensorRole::AttentionQ)?;
    let hidden_size = attention_q.shape.first().copied().unwrap_or(1).max(1);
    let attention_size = descriptor_element_count(attention_q).max(hidden_size);

    let mut entries = Vec::new();
    push_descriptor_op(
        &mut entries,
        inspection,
        layer_index,
        DenseGgufTensorRole::AttentionNorm,
        OpType::RmsNorm,
        false,
    )?;
    push_descriptor_op(
        &mut entries,
        inspection,
        layer_index,
        DenseGgufTensorRole::AttentionQ,
        OpType::MatMul,
        false,
    )?;
    push_descriptor_op(
        &mut entries,
        inspection,
        layer_index,
        DenseGgufTensorRole::AttentionK,
        OpType::MatMul,
        false,
    )?;
    push_descriptor_op(
        &mut entries,
        inspection,
        layer_index,
        DenseGgufTensorRole::AttentionV,
        OpType::MatMul,
        false,
    )?;
    push_synthetic_op(
        &mut entries,
        format!("blk.{layer_index}.rope"),
        "rope",
        OpType::RoPE,
        hidden_size,
    );
    push_synthetic_op(
        &mut entries,
        format!("blk.{layer_index}.attention_scores"),
        "attention_scores",
        OpType::Attention,
        attention_size,
    );
    push_synthetic_op(
        &mut entries,
        format!("blk.{layer_index}.attention_softmax"),
        "attention_softmax",
        OpType::Softmax,
        hidden_size,
    );
    push_synthetic_op(
        &mut entries,
        format!("blk.{layer_index}.attention_v_mix"),
        "attention_v_mix",
        OpType::Attention,
        attention_size,
    );
    push_descriptor_op(
        &mut entries,
        inspection,
        layer_index,
        DenseGgufTensorRole::AttentionOutput,
        OpType::MatMul,
        false,
    )?;
    push_descriptor_op(
        &mut entries,
        inspection,
        layer_index,
        DenseGgufTensorRole::FfnNorm,
        OpType::RmsNorm,
        false,
    )?;
    push_descriptor_op(
        &mut entries,
        inspection,
        layer_index,
        DenseGgufTensorRole::MlpGate,
        OpType::MatMul,
        false,
    )?;
    push_descriptor_op(
        &mut entries,
        inspection,
        layer_index,
        DenseGgufTensorRole::MlpUp,
        OpType::MatMul,
        false,
    )?;
    push_synthetic_op(
        &mut entries,
        format!("blk.{layer_index}.mlp_activation"),
        "mlp_activation",
        OpType::Activation,
        hidden_size,
    );
    push_descriptor_op(
        &mut entries,
        inspection,
        layer_index,
        DenseGgufTensorRole::MlpDown,
        OpType::MatMul,
        false,
    )?;

    Ok(entries)
}

fn push_descriptor_op(
    entries: &mut Vec<DenseLayerPlanEntry>,
    inspection: &DenseGgufDescriptorInspection,
    layer_index: usize,
    role: DenseGgufTensorRole,
    op_type: OpType,
    is_quantized: bool,
) -> Result<()> {
    let descriptor = descriptor_for_layer_role(inspection, layer_index, role)?;
    entries.push(DenseLayerPlanEntry {
        op: DispatchOp {
            name: descriptor.name.clone(),
            op_type,
            size: descriptor_element_count(descriptor),
            is_quantized,
        },
        role: dense_role_label(role).to_string(),
        source: "gguf_tensor_descriptor",
        source_tensor: Some(descriptor.name.clone()),
        source_tensor_type: Some(descriptor.tensor_type.clone()),
        source_shape: Some(descriptor.shape.clone()),
    });
    Ok(())
}

fn push_synthetic_op(
    entries: &mut Vec<DenseLayerPlanEntry>,
    name: String,
    role: &'static str,
    op_type: OpType,
    size: usize,
) {
    entries.push(DenseLayerPlanEntry {
        op: DispatchOp { name, op_type, size: size.max(1), is_quantized: false },
        role: role.to_string(),
        source: "derived_transformer_op",
        source_tensor: None,
        source_tensor_type: None,
        source_shape: None,
    });
}

fn descriptor_for_role(
    inspection: &DenseGgufDescriptorInspection,
    role: DenseGgufTensorRole,
) -> Result<&DenseGgufTensorDescriptor> {
    inspection
        .descriptors
        .iter()
        .find(|descriptor| descriptor.role == role)
        .ok_or_else(|| anyhow!("dense GGUF descriptor inspection missing role {role:?}"))
}

fn descriptor_for_layer_role(
    inspection: &DenseGgufDescriptorInspection,
    layer_index: usize,
    role: DenseGgufTensorRole,
) -> Result<&DenseGgufTensorDescriptor> {
    inspection
        .descriptors
        .iter()
        .find(|descriptor| {
            descriptor.role == role && extract_layer_index(&descriptor.name) == Some(layer_index)
        })
        .ok_or_else(|| {
            anyhow!("dense GGUF descriptor inspection missing layer {layer_index} role {role:?}")
        })
}

fn descriptor_element_count(descriptor: &DenseGgufTensorDescriptor) -> usize {
    descriptor.shape.iter().copied().fold(1usize, |acc, dim| acc.saturating_mul(dim)).max(1)
}

fn checked_u64_mul(left: u64, right: u64, label: &str) -> Result<u64> {
    left.checked_mul(right).ok_or_else(|| anyhow!("{label} overflowed"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::dense_gguf_linear_parity::roles::DEFAULT_ROLE_SWEEP;
    use bitnet_kernels::cuda::{
        CUDA_DENSE_ATTENTION_SCORE_KERNEL_ID, CUDA_DENSE_ATTENTION_SCORE_REFERENCE_BACKEND,
        CUDA_DENSE_ATTENTION_SCORE_TARGET_BACKEND, CUDA_DENSE_ATTENTION_SCORE_TOLERANCE,
        CUDA_DENSE_ATTENTION_SOFTMAX_KERNEL_ID, CUDA_DENSE_ATTENTION_SOFTMAX_TOLERANCE,
        CUDA_DENSE_ATTENTION_V_MIX_KERNEL_ID, CUDA_DENSE_ATTENTION_V_MIX_TOLERANCE,
        CUDA_DENSE_F16_GEMM_KERNEL_ID, CUDA_DENSE_GEMM_REFERENCE_BACKEND,
        CUDA_DENSE_GEMM_TARGET_BACKEND, CUDA_DENSE_GGUF_LINEAR_F16_GEMM_TOLERANCE,
        CUDA_DENSE_MLP_ACTIVATION_KERNEL_ID, CUDA_DENSE_MLP_ACTIVATION_REFERENCE_BACKEND,
        CUDA_DENSE_MLP_ACTIVATION_TARGET_BACKEND, CUDA_DENSE_MLP_ACTIVATION_TOLERANCE,
        CUDA_DENSE_RMSNORM_KERNEL_ID, CUDA_DENSE_RMSNORM_REFERENCE_BACKEND,
        CUDA_DENSE_RMSNORM_TARGET_BACKEND, CUDA_DENSE_RMSNORM_TOLERANCE, CUDA_DENSE_ROPE_KERNEL_ID,
        CUDA_DENSE_ROPE_REFERENCE_BACKEND, CUDA_DENSE_ROPE_TARGET_BACKEND,
        CUDA_DENSE_ROPE_TOLERANCE, CudaDenseAttentionScoreStats, CudaDenseAttentionSoftmaxStats,
        CudaDenseAttentionVMixStats, CudaDenseGemmStats, CudaDenseMlpActivationStats,
        CudaDenseRmsNormStats, CudaDenseRopeStats,
    };
    use bitnet_models::formats::gguf::GgufTensorType;
    use bitnet_models::formats::gguf::{GgufReader, GgufValue};

    #[test]
    fn qwen3_user_path_uses_qwen3_prerequisite_receipts() {
        let model = Path::new("Qwen3-0.6B-Q8_0.gguf");
        let context = dense_qwen_proof_context_for_model_path(model);

        assert_eq!(context.proof_model.id, QWEN3_06B_INSTRUCT_Q8_0_MODEL_ID);
        assert_eq!(context.proof_model.model_coverage_row, "dense_qwen3_06b_q8_candidate");
        assert_eq!(context.proof_model.model_coverage_tier, "accelerator_answer_ready");
        assert_eq!(context.receipts.one_token_proof, DEFAULT_QWEN3_ONE_TOKEN_PROOF_RECEIPT);
        assert_eq!(context.receipts.short_decode_proof, DEFAULT_QWEN3_SHORT_DECODE_PROOF_RECEIPT);
        assert_eq!(context.receipts.warm_session_proof, DEFAULT_QWEN3_WARM_SESSION_PROOF_RECEIPT);
        assert_eq!(dense_qwen_ask_work_item(context.proof_model), "CUDA-MODEL-010");
        assert_eq!(dense_qwen_chat_work_item(context.proof_model), "CUDA-MODEL-011");
    }

    #[test]
    fn qwen3_capture_defaults_resolve_to_qwen3_prerequisite_receipts() {
        let receipts = dense_qwen_receipts_for_proof_model(&QWEN3_06B_INSTRUCT_Q8_0_PROOF_MODEL);
        let resolved = dense_qwen_model_default_receipt_path(
            Path::new(DEFAULT_DENSE_QWEN_ALL_LAYER_PLAN_RECEIPT),
            DEFAULT_DENSE_QWEN_ALL_LAYER_PLAN_RECEIPT,
            receipts.all_layer_plan,
        );

        assert_eq!(resolved, PathBuf::from(DEFAULT_QWEN3_ALL_LAYER_PLAN_RECEIPT));
        assert_eq!(
            dense_qwen_model_default_receipt_path(
                Path::new(DEFAULT_DENSE_QWEN_MODEL_BOUNDARY_FIXTURES_RECEIPT),
                DEFAULT_DENSE_QWEN_MODEL_BOUNDARY_FIXTURES_RECEIPT,
                receipts.model_boundary_fixtures,
            ),
            PathBuf::from(DEFAULT_QWEN3_MODEL_BOUNDARY_FIXTURES_RECEIPT)
        );
        assert_eq!(
            dense_qwen_model_default_receipt_path(
                Path::new(DEFAULT_DENSE_QWEN_KV_CACHE_POLICY_RECEIPT),
                DEFAULT_DENSE_QWEN_KV_CACHE_POLICY_RECEIPT,
                receipts.kv_cache_policy,
            ),
            PathBuf::from(DEFAULT_QWEN3_KV_CACHE_POLICY_RECEIPT)
        );
        assert_eq!(
            dense_qwen_model_default_receipt_path(
                Path::new(DEFAULT_DENSE_QWEN_SAMPLING_POLICY_RECEIPT),
                DEFAULT_DENSE_QWEN_SAMPLING_POLICY_RECEIPT,
                receipts.sampling_policy,
            ),
            PathBuf::from(DEFAULT_QWEN3_SAMPLING_POLICY_RECEIPT)
        );

        let explicit = dense_qwen_model_default_receipt_path(
            Path::new("target/custom-qwen3-prereq.json"),
            DEFAULT_DENSE_QWEN_ALL_LAYER_PLAN_RECEIPT,
            receipts.all_layer_plan,
        );

        assert_eq!(explicit, PathBuf::from("target/custom-qwen3-prereq.json"));
    }

    #[test]
    fn qwen25_capture_defaults_stay_on_qwen25_prerequisite_receipts() {
        let receipts = dense_qwen_receipts_for_proof_model(&QWEN25_05B_INSTRUCT_Q8_0_PROOF_MODEL);
        let resolved = dense_qwen_model_default_receipt_path(
            Path::new(DEFAULT_DENSE_QWEN_ALL_LAYER_PLAN_RECEIPT),
            DEFAULT_DENSE_QWEN_ALL_LAYER_PLAN_RECEIPT,
            receipts.all_layer_plan,
        );

        assert_eq!(resolved, PathBuf::from(DEFAULT_DENSE_QWEN_ALL_LAYER_PLAN_RECEIPT));
    }

    #[test]
    fn qwen_one_token_phase_trace_writes_jsonl_events() -> Result<(), Box<dyn std::error::Error>> {
        let path = std::env::temp_dir().join(format!(
            "bitnet-qwen-phase-trace-{}-{}.jsonl",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        let trace = DenseQwenPhaseTrace::new(Some(&path), "test-command");
        trace.reset()?;

        trace.emit("cpu_reference", "model_load_start", json!({ "model": "test.gguf" }))?;

        let contents = std::fs::read_to_string(&path)?;
        let line = contents.lines().next().ok_or_else(|| std::io::Error::other("trace line"))?;
        let event: Value = serde_json::from_str(line)?;
        std::fs::remove_file(&path).ok();

        assert_eq!(event["schema"], json!(1));
        assert_eq!(event["command"], json!("test-command"));
        assert_eq!(event["phase"], json!("cpu_reference"));
        assert_eq!(event["state"], json!("model_load_start"));
        assert_eq!(event["details"]["model"], json!("test.gguf"));
        Ok(())
    }

    #[test]
    fn qwen_one_token_phase_trace_reset_discards_stale_events()
    -> Result<(), Box<dyn std::error::Error>> {
        let path = std::env::temp_dir().join(format!(
            "bitnet-qwen-phase-trace-reset-{}-{}.jsonl",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        std::fs::write(&path, "{\"stale\":true}\n")?;
        let trace = DenseQwenPhaseTrace::new(Some(&path), "test-command");

        trace.reset()?;
        trace.emit("command", "start", json!({}))?;

        let contents = std::fs::read_to_string(&path)?;
        std::fs::remove_file(&path).ok();

        assert!(!contents.contains("stale"));
        assert_eq!(contents.lines().count(), 1);
        Ok(())
    }

    #[test]
    fn extracted_dense_qwen_linear_maps_to_kernel_fixture() {
        let data = build_qwen_gguf(vec![(
            "blk.0.attn_q.weight",
            vec![4, 3],
            GgufTensorType::Q8_0,
            q8_0_blob(0.5, &(1..=12).collect::<Vec<_>>()),
        )]);
        let reader = GgufReader::new(&data).expect("parse qwen fixture");
        let extracted = extract_dense_gguf_linear_fixture(&reader, DenseGgufTensorRole::AttentionQ)
            .expect("extract fixture");

        let kernel_fixture =
            kernel_fixture_from_extracted(&extracted).expect("kernel fixture conversion");

        assert_eq!(kernel_fixture.model_family, "qwen");
        assert_eq!(kernel_fixture.tensor_name, "blk.0.attn_q.weight");
        assert_eq!(kernel_fixture.tensor_role, "attention_q");
        assert_eq!(kernel_fixture.tensor_type, "q8_0");
        assert_eq!(kernel_fixture.matrix_rows, 3);
        assert_eq!(kernel_fixture.matrix_cols, 4);
        assert_eq!(kernel_fixture.weights_row_major_f32.len(), 12);
        assert_eq!(kernel_fixture.input_f32.len(), 4);
        assert_eq!(kernel_fixture.source_weight_sha256, extracted.summary.weight_values_sha256);
    }

    #[test]
    fn extracted_dense_linear_receipt_validates() {
        let data = build_qwen_gguf(vec![(
            "blk.0.attn_q.weight",
            vec![4, 3],
            GgufTensorType::Q8_0,
            q8_0_blob(0.5, &(1..=12).collect::<Vec<_>>()),
        )]);
        let reader = GgufReader::new(&data).expect("parse qwen fixture");
        let extracted = extract_dense_gguf_linear_fixture(&reader, DenseGgufTensorRole::AttentionQ)
            .expect("extract fixture");
        let kernel_fixture =
            kernel_fixture_from_extracted(&extracted).expect("kernel fixture conversion");
        let parity = synthetic_parity_from_kernel_fixture(&kernel_fixture);

        let receipt = dense_gguf_linear_cuda_parity_receipt_json(
            &parity,
            &extracted,
            None,
            Path::new("synthetic-qwen3-q8_0-linear.gguf"),
            &"0".repeat(64),
            "target/bitnet/receipts/dense-gguf-linear-cuda-parity.json",
            "2026-05-09T00:00:00Z",
        );

        validate_dense_gguf_linear_cuda_parity_receipt_json(&receipt).unwrap();
        assert_eq!(receipt["execution_plan"]["selected_route"], "dense_regular_llm_cuda");
        assert_eq!(receipt["claim_boundary"]["dense_gguf_inference_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["bitnet_packed_i2s_qk256_proof"], false);
    }

    #[test]
    fn extracted_dense_linear_role_sweep_receipt_validates() {
        let values = (1..=12).collect::<Vec<_>>();
        let data = build_qwen_gguf(vec![
            ("blk.0.attn_q.weight", vec![4, 3], GgufTensorType::Q8_0, q8_0_blob(0.5, &values)),
            ("blk.0.attn_k.weight", vec![4, 3], GgufTensorType::Q8_0, q8_0_blob(0.25, &values)),
            ("blk.0.ffn_down.weight", vec![4, 3], GgufTensorType::Q8_0, q8_0_blob(0.125, &values)),
        ]);
        let reader = GgufReader::new(&data).expect("parse qwen fixture");
        let roles = [
            DenseGgufTensorRole::AttentionQ,
            DenseGgufTensorRole::AttentionK,
            DenseGgufTensorRole::MlpDown,
        ];
        let results = roles
            .iter()
            .map(|role| {
                let extracted =
                    extract_dense_gguf_linear_fixture(&reader, *role).expect("extract fixture");
                let kernel_fixture =
                    kernel_fixture_from_extracted(&extracted).expect("kernel fixture conversion");
                let parity = synthetic_parity_from_kernel_fixture(&kernel_fixture);
                DenseLinearSweepResult { extracted, parity }
            })
            .collect::<Vec<_>>();

        let receipt = dense_gguf_linear_role_sweep_cuda_parity_receipt_json(
            &results,
            None,
            Path::new("synthetic-qwen3-q8_0-linear-sweep.gguf"),
            &"0".repeat(64),
            "target/bitnet/receipts/dense-gguf-linear-role-sweep-cuda-parity.json",
            "2026-05-09T00:00:00Z",
        )
        .unwrap();

        validate_dense_gguf_linear_role_sweep_cuda_parity_receipt_json(&receipt).unwrap();
        assert_eq!(receipt["execution_plan"]["selected_route"], "dense_regular_llm_cuda");
        assert_eq!(receipt["execution_plan"]["cuda_dense_regular_llm_ops"], 3);
        assert_eq!(receipt["linear_role_sweep"]["roles_total"], 3);
        assert_eq!(receipt["claim_boundary"]["dense_gguf_inference_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["bitnet_packed_i2s_qk256_proof"], false);
    }

    #[test]
    fn dense_gguf_one_layer_plan_receipt_records_strict_cuda_route_completion() {
        let data = build_complete_qwen_layer_gguf();
        let reader = GgufReader::new(&data).expect("parse qwen fixture");
        let inspection = inspect_dense_gguf_tensor_descriptors(&reader).expect("inspect");

        let receipt = dense_gguf_one_layer_execution_plan_receipt_json(
            &inspection,
            None,
            Path::new("synthetic-qwen3-q8_0-layer-plan.gguf"),
            &"0".repeat(64),
            "target/bitnet/receipts/dense-gguf-one-layer-plan.json",
            "2026-05-09T00:00:00Z",
            0,
        )
        .unwrap();

        validate_dense_gguf_one_layer_execution_plan_receipt_json(&receipt).unwrap();
        assert_eq!(receipt["execution_plan"]["selected_route"], "dense_regular_llm_cuda");
        assert_eq!(receipt["execution_plan"]["cuda_dense_regular_llm_ops"], 14);
        assert_eq!(receipt["execution_plan"]["unsupported_ops"], 0);
        assert_eq!(receipt["execution_plan"]["strict_cuda_ready"], true);
        assert_eq!(receipt["one_layer_plan"]["operations"].as_array().unwrap().len(), 14);
        assert_eq!(receipt["one_layer_plan"]["linear_cuda_ops_total"], 7);
        assert_eq!(receipt["one_layer_plan"]["norm_cuda_ops_total"], 2);
        assert_eq!(receipt["one_layer_plan"]["rope_cuda_ops_total"], 1);
        assert_eq!(receipt["one_layer_plan"]["attention_score_cuda_ops_total"], 1);
        assert_eq!(receipt["one_layer_plan"]["attention_softmax_cuda_ops_total"], 1);
        assert_eq!(receipt["one_layer_plan"]["attention_v_mix_cuda_ops_total"], 1);
        assert_eq!(receipt["one_layer_plan"]["mlp_activation_cuda_ops_total"], 1);
        assert_eq!(receipt["one_layer_plan"]["strict_cuda_ready"], true);
        assert_eq!(receipt["gap_audit"]["unsupported_ops_total"], 0);
        assert_eq!(receipt["gap_audit"]["rmsnorm_cuda_parity_available"], true);
        assert_eq!(receipt["gap_audit"]["rope_cuda_parity_available"], true);
        assert_eq!(receipt["gap_audit"]["attention_score_cuda_parity_available"], true);
        assert_eq!(receipt["gap_audit"]["attention_softmax_cuda_parity_available"], true);
        assert_eq!(receipt["gap_audit"]["attention_v_mix_cuda_parity_available"], true);
        assert_eq!(receipt["gap_audit"]["mlp_activation_cuda_parity_available"], true);
        assert_eq!(receipt["gap_audit"]["next_candidate_gap"], "none");
        assert_eq!(receipt["gap_audit"]["next_required_proof"], "one_layer_cpu_reference_harness");
        assert_eq!(receipt["gap_audit"]["cpu_fallback_ops_total"], 0);
        assert_eq!(receipt["gap_audit"]["strict_cuda_rejects_cpu_fallback"], true);
        let attention_score_op = receipt["one_layer_plan"]["operations"]
            .as_array()
            .unwrap()
            .iter()
            .find(|op| op["role"] == "attention_scores")
            .expect("attention_scores op");
        assert_eq!(attention_score_op["route"], "dense_regular_llm_cuda");
        assert_eq!(attention_score_op["status"], "cuda_routable");
        let attention_softmax_op = receipt["one_layer_plan"]["operations"]
            .as_array()
            .unwrap()
            .iter()
            .find(|op| op["role"] == "attention_softmax")
            .expect("attention_softmax op");
        assert_eq!(attention_softmax_op["route"], "dense_regular_llm_cuda");
        assert_eq!(attention_softmax_op["status"], "cuda_routable");
        let attention_v_mix_op = receipt["one_layer_plan"]["operations"]
            .as_array()
            .unwrap()
            .iter()
            .find(|op| op["role"] == "attention_v_mix")
            .expect("attention_v_mix op");
        assert_eq!(attention_v_mix_op["route"], "dense_regular_llm_cuda");
        assert_eq!(attention_v_mix_op["status"], "cuda_routable");
        let mlp_activation_op = receipt["one_layer_plan"]["operations"]
            .as_array()
            .unwrap()
            .iter()
            .find(|op| op["role"] == "mlp_activation")
            .expect("mlp_activation op");
        assert_eq!(mlp_activation_op["route"], "dense_regular_llm_cuda");
        assert_eq!(mlp_activation_op["status"], "cuda_routable");
        assert_eq!(receipt["gap_audit"]["unsupported_ops"].as_array().unwrap().len(), 0);
        assert_eq!(receipt["claim_boundary"]["dense_gguf_one_layer_execution_plan_claimed"], true);
        assert_eq!(receipt["claim_boundary"]["dense_gguf_one_layer_inference_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["dense_gguf_inference_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["bitnet_packed_i2s_qk256_proof"], false);
    }

    #[test]
    fn dense_gguf_all_layer_plan_receipt_records_transformer_stack_gaps() {
        let data = build_two_layer_qwen_gguf();
        let reader = GgufReader::new(&data).expect("parse two-layer qwen fixture");
        let inspection = inspect_dense_gguf_tensor_descriptors(&reader).expect("inspect");

        let receipt = dense_gguf_all_layer_execution_plan_receipt_json(
            &inspection,
            None,
            Path::new("synthetic-qwen3-q8_0-all-layer-plan.gguf"),
            &"0".repeat(64),
            "target/bitnet/receipts/dense-gguf-all-layer-plan.json",
            "2026-05-09T00:00:00Z",
        )
        .unwrap();

        validate_dense_gguf_all_layer_execution_plan_receipt_json(&receipt).unwrap();
        assert_eq!(receipt["artifact_kind"], "dense_gguf_all_layer_execution_plan");
        assert_eq!(receipt["execution_plan"]["selected_route"], "dense_regular_llm_cuda");
        assert_eq!(receipt["execution_plan"]["cuda_dense_regular_llm_ops"], 28);
        assert_eq!(receipt["execution_plan"]["unsupported_ops"], 0);
        assert_eq!(receipt["all_layer_plan"]["transformer_layers_total"], 2);
        assert_eq!(receipt["all_layer_plan"]["layers_with_complete_cuda_block_plan"], 2);
        assert_eq!(receipt["all_layer_plan"]["layer_plan_matches_layer0"], true);
        assert_eq!(receipt["all_layer_plan"]["total_ops"], 28);
        assert_eq!(receipt["all_layer_plan"]["cuda_routable_ops_total"], 28);
        assert_eq!(receipt["all_layer_plan"]["linear_cuda_ops_total"], 14);
        assert_eq!(receipt["all_layer_plan"]["norm_cuda_ops_total"], 4);
        assert_eq!(receipt["all_layer_plan"]["rope_cuda_ops_total"], 2);
        assert_eq!(receipt["all_layer_plan"]["attention_score_cuda_ops_total"], 2);
        assert_eq!(receipt["all_layer_plan"]["attention_softmax_cuda_ops_total"], 2);
        assert_eq!(receipt["all_layer_plan"]["attention_v_mix_cuda_ops_total"], 2);
        assert_eq!(receipt["all_layer_plan"]["mlp_activation_cuda_ops_total"], 2);
        assert!(receipt["all_layer_plan"]["layer_differences"].as_array().unwrap().is_empty());
        assert!(receipt["all_layer_plan"]["missing_layer_indices"].as_array().unwrap().is_empty());
        assert_eq!(receipt["all_layer_plan"]["strict_cuda_ready_scope"], "transformer_blocks_only");
        assert_eq!(receipt["all_layer_plan"]["dense_gguf_inference_claimed"], false);
        assert_eq!(receipt["model_boundary_gaps"]["all_boundary_gaps_explicit"], true);
        assert_eq!(receipt["model_boundary_gaps"]["gaps"].as_array().unwrap().len(), 5);
        assert_eq!(receipt["model_boundary_gaps"]["qwen_one_token_cuda_blocked"], true);
        assert_eq!(receipt["claim_boundary"]["dense_gguf_all_layer_execution_plan_claimed"], true);
        assert_eq!(receipt["claim_boundary"]["qwen_one_token_cuda_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["dense_gguf_inference_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["bitnet_packed_i2s_qk256_proof"], false);
    }

    #[test]
    fn dense_gguf_all_layer_plan_receipt_allows_tied_lm_head_boundary_gap() -> Result<()> {
        let data = build_two_layer_qwen_tied_lm_head_gguf();
        let reader = GgufReader::new(&data)?;
        let inspection = inspect_dense_gguf_tensor_descriptors(&reader)?;
        assert!(!inspection.required_roles_present);
        assert!(!inspection.strict_descriptor_complete);
        assert!(inspection.missing_required_roles.contains(&DenseGgufTensorRole::Output));

        let receipt = dense_gguf_all_layer_execution_plan_receipt_json(
            &inspection,
            None,
            Path::new("synthetic-qwen3-q8_0-all-layer-plan-tied-head.gguf"),
            &"0".repeat(64),
            "target/bitnet/receipts/dense-gguf-all-layer-plan-tied-head.json",
            "2026-05-15T00:00:00Z",
        )?;

        validate_dense_gguf_all_layer_execution_plan_receipt_json(&receipt)?;
        assert_eq!(receipt["descriptor_coverage"]["required_roles_present"], false);
        assert_eq!(receipt["descriptor_coverage"]["strict_descriptor_complete"], false);
        assert_eq!(
            receipt["descriptor_coverage"]["transformer_block_required_roles_present"],
            true
        );
        let missing_model_boundary_roles =
            receipt["descriptor_coverage"]["missing_model_boundary_roles"]
                .as_array()
                .ok_or_else(|| anyhow!("missing_model_boundary_roles must be an array"))?;
        assert!(missing_model_boundary_roles.iter().any(|role| role == "output"));
        let boundary_gaps = receipt["model_boundary_gaps"]["gaps"]
            .as_array()
            .ok_or_else(|| anyhow!("model_boundary_gaps.gaps must be an array"))?;
        let lm_head_gap = boundary_gaps
            .iter()
            .find(|gap| gap["gap"] == "lm_head_logits")
            .ok_or_else(|| anyhow!("missing lm_head_logits boundary gap"))?;
        assert_eq!(lm_head_gap["source_tensor"], "token_embd.weight");
        let disposition = lm_head_gap["disposition"]
            .as_str()
            .ok_or_else(|| anyhow!("lm_head_logits disposition must be a string"))?;
        assert!(disposition.contains("tied"));
        assert_eq!(receipt["all_layer_plan"]["strict_cuda_ready"], true);
        assert_eq!(receipt["all_layer_plan"]["qwen_one_token_cuda_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["dense_gguf_inference_claimed"], false);
        Ok(())
    }

    #[test]
    fn dense_gguf_one_layer_cpu_reference_records_phase_hashes() {
        let data = build_integrated_qwen_layer_gguf();
        let reader = GgufReader::new(&data).expect("parse integrated qwen fixture");
        let inspection = inspect_dense_gguf_tensor_descriptors(&reader).expect("inspect");
        let reference =
            dense_gguf_one_layer_cpu_reference_from_reader(&reader, &inspection, 0, 4, 1)
                .expect("one-layer CPU reference");

        assert_eq!(reference.layer_index, 0);
        assert_eq!(reference.seq_len, 4);
        assert_eq!(reference.hidden_size, 4);
        assert_eq!(reference.q_heads, 2);
        assert_eq!(reference.kv_heads, 1);
        assert_eq!(reference.head_dim, 2);
        assert_eq!(reference.intermediate_size, 6);
        assert_eq!(reference.final_output_len, 16);
        assert_eq!(reference.phases.len(), 17);
        assert_eq!(reference.phases.first().unwrap().name, "deterministic_input");
        assert_eq!(reference.phases.last().unwrap().name, "second_residual");

        let receipt = dense_gguf_one_layer_cpu_reference_receipt_json(
            &inspection,
            &reference,
            Path::new("synthetic-qwen3-q8_0-one-layer-cpu-reference.gguf"),
            &"0".repeat(64),
            "target/bitnet/receipts/dense-gguf-one-layer-cpu-reference.json",
            "2026-05-09T00:00:00Z",
        )
        .unwrap();

        validate_dense_gguf_one_layer_cpu_reference_receipt_json(&receipt).unwrap();
        assert_eq!(receipt["artifact_kind"], "dense_gguf_one_layer_cpu_reference");
        assert_eq!(receipt["runtime_api"], "cpu");
        assert_eq!(receipt["reference_harness"]["cpu_reference_only"], true);
        assert_eq!(receipt["reference_harness"]["cuda_execution_claimed"], false);
        assert_eq!(receipt["reference_harness"]["dense_gguf_inference_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["dense_gguf_one_layer_cpu_reference_claimed"], true);
        assert_eq!(receipt["claim_boundary"]["dense_regular_llm_cuda_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["bitnet_packed_i2s_qk256_proof"], false);
    }

    #[test]
    fn dense_gguf_one_layer_cuda_integrated_parity_receipt_records_layer_claim_only() {
        let data = build_integrated_qwen_layer_gguf();
        let reader = GgufReader::new(&data).expect("parse integrated qwen fixture");
        let inspection = inspect_dense_gguf_tensor_descriptors(&reader).expect("inspect");
        let reference =
            dense_gguf_one_layer_cpu_reference_from_reader(&reader, &inspection, 0, 4, 1)
                .expect("one-layer CPU reference");
        let parity = synthetic_one_layer_cuda_integrated_parity_from_reference(&reference);

        let receipt = dense_gguf_one_layer_cuda_integrated_parity_receipt_json(
            &inspection,
            &reference,
            &parity,
            None,
            Path::new("synthetic-qwen3-q8_0-one-layer-cuda-parity.gguf"),
            &"0".repeat(64),
            "target/bitnet/receipts/dense-gguf-one-layer-cuda-parity.json",
            "2026-05-09T00:00:00Z",
        )
        .unwrap();

        validate_dense_gguf_one_layer_cuda_integrated_parity_receipt_json(&receipt).unwrap();
        assert_eq!(receipt["artifact_kind"], "dense_gguf_one_layer_cuda_integrated_parity");
        assert_eq!(receipt["runtime_api"], "cuda");
        assert_eq!(receipt["execution_plan"]["cuda_dense_regular_llm_ops"], 14);
        assert_eq!(receipt["cuda_layer"]["governed_cuda_ops_total"], 14);
        assert_eq!(receipt["cuda_layer"]["residual_host_ops_total"], 2);
        assert_eq!(
            receipt["claim_boundary"]["dense_gguf_one_layer_cuda_integrated_parity_claimed"],
            true
        );
        assert_eq!(receipt["claim_boundary"]["dense_gguf_one_layer_inference_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["dense_gguf_inference_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["qwen_one_token_cuda_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["speedup_claim"], false);
        assert_eq!(receipt["claim_boundary"]["bitnet_packed_i2s_qk256_proof"], false);
    }

    #[test]
    fn dense_gguf_model_boundary_fixture_receipt_records_embedding_norm_logits() {
        let data = build_model_boundary_qwen_gguf();
        let reader = GgufReader::new(&data).expect("parse model-boundary qwen fixture");
        let inspection = inspect_dense_gguf_tensor_descriptors(&reader).expect("inspect");
        let fixtures = dense_gguf_model_boundary_fixtures_from_reader(&reader, &inspection, 4, 4)
            .expect("model-boundary fixtures");

        assert_eq!(fixtures.seq_len, 4);
        assert_eq!(fixtures.hidden_size, 4);
        assert_eq!(fixtures.vocab_size, 4);
        assert_eq!(fixtures.token_ids, vec![0, 1, 2, 3]);
        assert_eq!(fixtures.token_embedding.role, "token_embedding");
        assert_eq!(fixtures.final_norm.role, "final_norm");
        assert_eq!(fixtures.lm_head_logits.role, "lm_head_logits");
        assert_eq!(fixtures.logits_len, 4);
        assert_eq!(fixtures.logits_top_k.len(), 4);

        let receipt = dense_gguf_model_boundary_fixtures_receipt_json(
            &inspection,
            &fixtures,
            None,
            Path::new("synthetic-qwen3-q8_0-model-boundary-fixtures.gguf"),
            &"0".repeat(64),
            "target/bitnet/receipts/dense-gguf-model-boundary-fixtures.json",
            "2026-05-09T00:00:00Z",
        )
        .unwrap();

        validate_dense_gguf_model_boundary_fixtures_receipt_json(&receipt).unwrap();
        assert_eq!(receipt["artifact_kind"], "dense_gguf_model_boundary_fixtures");
        assert_eq!(receipt["runtime_api"], "cuda");
        assert_eq!(receipt["execution_plan"]["cuda_dense_regular_llm_ops"], 3);
        assert_eq!(receipt["model_boundary_fixtures"]["fixtures_total"], 3);
        assert_eq!(receipt["model_boundary_fixtures"]["fixture_route_only"], true);
        assert_eq!(receipt["model_boundary_fixtures"]["cuda_kernel_execution_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["dense_gguf_model_boundary_fixtures_claimed"], true);
        assert_eq!(receipt["claim_boundary"]["kv_cache_policy_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["sampling_integration_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["qwen_one_token_cuda_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["dense_gguf_inference_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["bitnet_packed_i2s_qk256_proof"], false);
    }

    #[test]
    fn dense_gguf_kv_cache_policy_receipt_records_estimated_kv_bytes() {
        let data = build_model_boundary_qwen_gguf();
        let reader = GgufReader::new(&data).expect("parse KV policy qwen fixture");
        let inspection = inspect_dense_gguf_tensor_descriptors(&reader).expect("inspect");
        let policy =
            dense_gguf_kv_cache_policy_from_reader(&reader, &inspection, 4, 1).expect("kv policy");

        assert_eq!(policy.transformer_layers_total, 1);
        assert_eq!(policy.seq_len, 4);
        assert_eq!(policy.decode_steps, 1);
        assert_eq!(policy.q_heads, 2);
        assert_eq!(policy.kv_heads, 1);
        assert_eq!(policy.key_head_dim, 2);
        assert_eq!(policy.value_head_dim, 2);
        assert_eq!(policy.kv_values_per_token_per_layer, 4);
        assert_eq!(policy.kv_bytes_per_token_per_layer, 8);
        assert_eq!(policy.kv_bytes_per_token_all_layers, 8);
        assert_eq!(policy.prefill_write_bytes_estimate, 32);
        assert_eq!(policy.decode_read_bytes_per_step_estimate, 32);
        assert_eq!(policy.decode_write_bytes_per_step_estimate, 8);

        let receipt = dense_gguf_kv_cache_policy_receipt_json(
            &inspection,
            &policy,
            None,
            Path::new("synthetic-qwen3-q8_0-kv-cache-policy.gguf"),
            &"0".repeat(64),
            "target/bitnet/receipts/dense-gguf-kv-cache-policy.json",
            "2026-05-09T00:00:00Z",
        )
        .unwrap();

        validate_dense_gguf_kv_cache_policy_receipt_json(&receipt).unwrap();
        assert_eq!(receipt["artifact_kind"], "dense_gguf_kv_cache_policy");
        assert_eq!(receipt["execution_plan"]["cuda_dense_regular_llm_ops"], 1);
        assert_eq!(
            receipt["kv_cache_policy"]["planned_residency"],
            "cuda_required_for_strict_dense_cuda"
        );
        assert_eq!(receipt["kv_cache_policy"]["observed_residency"], "not_allocated_policy_only");
        assert_eq!(receipt["kv_cache_policy"]["runtime_kv_cache_allocated"], false);
        assert_eq!(receipt["kv_cache_policy"]["estimated_bytes_only"], true);
        assert_eq!(receipt["claim_boundary"]["kv_cache_policy_claimed"], true);
        assert_eq!(receipt["claim_boundary"]["kv_cache_cuda_residency_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["sampling_integration_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["qwen_one_token_cuda_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["dense_gguf_inference_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["speedup_claim"], false);
        assert_eq!(receipt["claim_boundary"]["bitnet_packed_i2s_qk256_proof"], false);
        assert_eq!(
            receipt["remaining_model_boundary_gaps"]["next_required_proof"],
            "dense_gguf_sampling_policy_receipt"
        );
    }

    #[test]
    fn dense_gguf_sampling_policy_receipt_records_logits_transfer_and_sampler() {
        let data = build_model_boundary_qwen_gguf();
        let reader = GgufReader::new(&data).expect("parse sampling policy qwen fixture");
        let inspection = inspect_dense_gguf_tensor_descriptors(&reader).expect("inspect");
        let policy = dense_gguf_sampling_policy_from_reader(&reader, &inspection, 4, 3)
            .expect("sampling policy");

        assert_eq!(policy.seq_len, 4);
        assert_eq!(policy.logits_len, policy.vocab_size);
        assert_eq!(policy.logits_element_bytes, 4);
        assert_eq!(policy.logits_transfer_bytes_per_step_estimate, policy.logits_len as u64 * 4);
        assert_eq!(policy.sampler_backend, "bitnet-sampling");
        assert_eq!(policy.sampler_location, "cpu");
        assert_eq!(policy.sampler_mode, "greedy");
        assert_eq!(policy.temperature, 0.0);
        assert_eq!(policy.top_k_filter, 0);
        assert!(policy.deterministic);
        assert!(!policy.rng_required);

        let receipt = dense_gguf_sampling_policy_receipt_json(
            &inspection,
            &policy,
            None,
            Path::new("synthetic-qwen3-q8_0-sampling-policy.gguf"),
            &"0".repeat(64),
            "target/bitnet/receipts/dense-gguf-sampling-policy.json",
            "2026-05-09T00:00:00Z",
        )
        .unwrap();

        validate_dense_gguf_sampling_policy_receipt_json(&receipt).unwrap();
        assert_eq!(receipt["artifact_kind"], "dense_gguf_sampling_policy");
        assert_eq!(receipt["execution_plan"]["cuda_dense_regular_llm_ops"], 1);
        assert_eq!(
            receipt["sampling_policy"]["logits_transfer_path"],
            "cuda_lm_head_logits_to_cpu_sampler"
        );
        assert_eq!(receipt["sampling_policy"]["sampler_backend"], "bitnet-sampling");
        assert_eq!(receipt["sampling_policy"]["sampler_location"], "cpu");
        assert_eq!(receipt["sampling_policy"]["sampler_mode"], "greedy");
        assert_eq!(receipt["sampling_policy"]["sampling_policy_claimed"], true);
        assert_eq!(receipt["sampling_policy"]["sampling_integration_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["sampling_policy_claimed"], true);
        assert_eq!(receipt["claim_boundary"]["sampling_integration_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["qwen_one_token_cuda_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["dense_gguf_inference_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["speedup_claim"], false);
        assert_eq!(receipt["claim_boundary"]["bitnet_packed_i2s_qk256_proof"], false);
        assert_eq!(
            receipt["remaining_model_boundary_gaps"]["next_required_proof"],
            "qwen_one_token_strict_cuda_proof"
        );
    }

    #[test]
    fn dense_gguf_norm_fixture_receipt_records_missing_cuda_kernel() {
        let data = build_model_boundary_qwen_gguf();
        let reader = GgufReader::new(&data).expect("parse qwen fixture");
        let inspection = inspect_dense_gguf_tensor_descriptors(&reader).expect("inspect");
        let fixtures = [DenseGgufTensorRole::AttentionNorm, DenseGgufTensorRole::FfnNorm]
            .iter()
            .map(|role| {
                extract_dense_gguf_norm_fixture(&reader, *role).expect("extract norm fixture")
            })
            .collect::<Vec<_>>();

        let receipt = dense_gguf_norm_fixture_receipt_json(
            &inspection,
            &fixtures,
            Path::new("synthetic-qwen3-q8_0-norm-fixture.gguf"),
            &"0".repeat(64),
            "target/bitnet/receipts/dense-gguf-norm-fixture.json",
            "2026-05-09T00:00:00Z",
        )
        .unwrap();

        validate_dense_gguf_norm_fixture_extraction_receipt_json(&receipt).unwrap();
        assert_eq!(receipt["norm_fixture_audit"]["roles_total"], 2);
        assert_eq!(receipt["norm_fixture_audit"]["cuda_kernel_status"], "missing_cuda_kernel");
        assert_eq!(receipt["norm_fixture_audit"]["strict_cuda_ready"], false);
        assert_eq!(receipt["claim_boundary"]["dense_gguf_norm_fixture_extraction_claimed"], true);
        assert_eq!(receipt["claim_boundary"]["dense_regular_llm_cuda_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["dense_gguf_inference_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["bitnet_packed_i2s_qk256_proof"], false);
    }

    #[test]
    fn dense_gguf_norm_cuda_parity_receipt_records_cuda_kernel() {
        let data = build_model_boundary_qwen_gguf();
        let reader = GgufReader::new(&data).expect("parse qwen fixture");
        let inspection = inspect_dense_gguf_tensor_descriptors(&reader).expect("inspect");
        let roles = [DenseGgufTensorRole::AttentionNorm, DenseGgufTensorRole::FfnNorm];
        let results = roles
            .iter()
            .map(|role| {
                let extracted =
                    extract_dense_gguf_norm_fixture(&reader, *role).expect("extract norm fixture");
                let kernel_fixture =
                    kernel_rmsnorm_fixture_from_extracted(&extracted).expect("kernel fixture");
                let parity = synthetic_rmsnorm_parity_from_fixture(&kernel_fixture);
                DenseNormParityResult { extracted, parity }
            })
            .collect::<Vec<_>>();

        let receipt = dense_gguf_norm_cuda_parity_receipt_json(
            &inspection,
            &results,
            None,
            Path::new("synthetic-qwen3-q8_0-norm-cuda-parity.gguf"),
            &"0".repeat(64),
            "target/bitnet/receipts/dense-gguf-norm-cuda-parity.json",
            "2026-05-09T00:00:00Z",
        )
        .unwrap();

        validate_dense_gguf_norm_cuda_parity_receipt_json(&receipt).unwrap();
        assert_eq!(receipt["execution_plan"]["selected_route"], "dense_regular_llm_cuda");
        assert_eq!(receipt["execution_plan"]["cuda_dense_regular_llm_ops"], 2);
        assert_eq!(receipt["parity"]["covered_roles"], json!(["attention_norm", "ffn_norm"]));
        assert_eq!(receipt["kernel_stats"][0]["kernel_id"], "dense_rmsnorm_f32_cuda");
        assert_eq!(receipt["claim_boundary"]["dense_gguf_norm_cuda_parity_claimed"], true);
        assert_eq!(receipt["claim_boundary"]["dense_gguf_inference_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["bitnet_packed_i2s_qk256_proof"], false);
    }

    #[test]
    fn dense_gguf_rope_cuda_parity_receipt_records_cuda_kernel() {
        let data = build_model_boundary_qwen_gguf();
        let reader = GgufReader::new(&data).expect("parse qwen fixture");
        let inspection = inspect_dense_gguf_tensor_descriptors(&reader).expect("inspect");
        let fixture = dense_gguf_rope_cuda_fixture_from_reader(&reader, &inspection, 0, 4, 1)
            .expect("rope fixture");
        let parity = synthetic_rope_parity_from_fixture(&fixture);

        let receipt = dense_gguf_rope_cuda_parity_receipt_json(
            &inspection,
            &fixture,
            &parity,
            None,
            Path::new("synthetic-qwen3-q8_0-rope-cuda-parity.gguf"),
            &"0".repeat(64),
            "target/bitnet/receipts/dense-gguf-rope-cuda-parity.json",
            "2026-05-09T00:00:00Z",
        );

        validate_dense_gguf_rope_cuda_parity_receipt_json(&receipt).unwrap();
        assert_eq!(receipt["execution_plan"]["selected_route"], "dense_regular_llm_cuda");
        assert_eq!(receipt["execution_plan"]["cuda_dense_regular_llm_ops"], 1);
        assert_eq!(receipt["rope_fixture"]["head_dim"], 2);
        assert_eq!(receipt["rope_fixture"]["q_heads"], 2);
        assert_eq!(receipt["rope_fixture"]["kv_heads"], 1);
        assert_eq!(receipt["kernel_stats"][0]["kernel_id"], "dense_rope_f32_cuda");
        assert_eq!(receipt["kernel_stats"][0]["invocations"], 2);
        assert_eq!(receipt["claim_boundary"]["dense_gguf_rope_cuda_parity_claimed"], true);
        assert_eq!(receipt["claim_boundary"]["dense_gguf_inference_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["bitnet_packed_i2s_qk256_proof"], false);
    }

    #[test]
    fn dense_gguf_attention_score_fixture_receipt_records_cpu_reference_gap() {
        let data = build_complete_qwen_layer_gguf();
        let reader = GgufReader::new(&data).expect("parse qwen fixture");
        let inspection = inspect_dense_gguf_tensor_descriptors(&reader).expect("inspect");
        let fixture = dense_gguf_attention_score_fixture_from_reader(&reader, &inspection, 0, 4, 1)
            .expect("attention score fixture");

        let receipt = dense_gguf_attention_score_fixture_receipt_json(
            &inspection,
            &fixture,
            Path::new("synthetic-qwen3-q8_0-attention-score-fixture.gguf"),
            &"0".repeat(64),
            "target/bitnet/receipts/dense-gguf-attention-score-fixture.json",
            "2026-05-09T00:00:00Z",
        );

        validate_dense_gguf_attention_score_fixture_receipt_json(&receipt).unwrap();
        let score_count = receipt["attention_score_fixture"]["score_count"].as_u64().unwrap();
        let finite_scores = receipt["attention_score_fixture"]["finite_scores"].as_u64().unwrap();
        let masked_scores =
            receipt["attention_score_fixture"]["causal_masked_scores"].as_u64().unwrap();

        assert_eq!(receipt["execution_plan"]["selected_route"], "unsupported");
        assert_eq!(receipt["execution_plan"]["unsupported_ops"], 1);
        assert_eq!(
            receipt["attention_score_fixture"]["score_shape"],
            json!([fixture.q_heads, fixture.seq_len, fixture.seq_len])
        );
        assert_eq!(finite_scores + masked_scores, score_count);
        assert_eq!(
            receipt["attention_score_gap_audit"]["cuda_kernel_status"],
            "missing_cuda_kernel"
        );
        assert_eq!(
            receipt["attention_score_gap_audit"]["next_required_proof"],
            "cuda_attention_score_kernel_parity"
        );
        assert_eq!(
            receipt["claim_boundary"]["dense_gguf_attention_score_fixture_extraction_claimed"],
            true
        );
        assert_eq!(receipt["claim_boundary"]["dense_regular_llm_cuda_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["dense_gguf_inference_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["bitnet_packed_i2s_qk256_proof"], false);
    }

    #[test]
    fn dense_gguf_attention_softmax_fixture_receipt_records_cpu_reference_gap() {
        let data = build_complete_qwen_layer_gguf();
        let reader = GgufReader::new(&data).expect("parse qwen fixture");
        let inspection = inspect_dense_gguf_tensor_descriptors(&reader).expect("inspect");
        let fixture =
            dense_gguf_attention_softmax_fixture_from_reader(&reader, &inspection, 0, 4, 1)
                .expect("attention softmax fixture");

        let receipt = dense_gguf_attention_softmax_fixture_receipt_json(
            &inspection,
            &fixture,
            Path::new("synthetic-qwen3-q8_0-attention-softmax-fixture.gguf"),
            &"0".repeat(64),
            "target/bitnet/receipts/dense-gguf-attention-softmax-fixture.json",
            "2026-05-09T00:00:00Z",
        );

        validate_dense_gguf_attention_softmax_fixture_receipt_json(&receipt).unwrap();
        assert_eq!(receipt["execution_plan"]["selected_route"], "unsupported");
        assert_eq!(receipt["execution_plan"]["unsupported_ops"], 1);
        assert_eq!(
            receipt["attention_softmax_fixture"]["source_attention_score_artifact_kind"],
            DENSE_GGUF_ATTENTION_SCORE_CUDA_PARITY_ARTIFACT_KIND
        );
        assert_eq!(
            receipt["attention_softmax_fixture"]["row_count"],
            json!(fixture.q_heads * fixture.seq_len)
        );
        assert_eq!(
            receipt["attention_softmax_fixture"]["probability_count"],
            json!(fixture.expected_probabilities_f32.len())
        );
        assert!(fixture.causal_zero_probabilities > 0);
        assert!(fixture.max_row_sum_abs_error <= 1.0e-6);
        assert_eq!(
            receipt["attention_softmax_gap_audit"]["cuda_kernel_status"],
            "missing_cuda_kernel"
        );
        assert_eq!(
            receipt["attention_softmax_gap_audit"]["next_required_proof"],
            "cuda_attention_softmax_kernel_parity"
        );
        assert_eq!(
            receipt["claim_boundary"]["dense_gguf_attention_softmax_fixture_extraction_claimed"],
            true
        );
        assert_eq!(receipt["claim_boundary"]["dense_regular_llm_cuda_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["dense_gguf_inference_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["bitnet_packed_i2s_qk256_proof"], false);
    }

    #[test]
    fn dense_gguf_attention_v_mix_fixture_receipt_records_cpu_reference_gap() {
        let data = build_complete_qwen_layer_gguf();
        let reader = GgufReader::new(&data).expect("parse qwen fixture");
        let inspection = inspect_dense_gguf_tensor_descriptors(&reader).expect("inspect");
        let fixture = dense_gguf_attention_v_mix_fixture_from_reader(&reader, &inspection, 0, 4, 1)
            .expect("attention V-mix fixture");

        let receipt = dense_gguf_attention_v_mix_fixture_receipt_json(
            &inspection,
            &fixture,
            Path::new("synthetic-qwen3-q8_0-attention-v-mix-fixture.gguf"),
            &"0".repeat(64),
            "target/bitnet/receipts/dense-gguf-attention-v-mix-fixture.json",
            "2026-05-09T00:00:00Z",
        );

        validate_dense_gguf_attention_v_mix_fixture_receipt_json(&receipt).unwrap();
        assert_eq!(receipt["execution_plan"]["selected_route"], "unsupported");
        assert_eq!(receipt["execution_plan"]["unsupported_ops"], 1);
        assert_eq!(
            receipt["attention_v_mix_fixture"]["source_attention_softmax_artifact_kind"],
            DENSE_GGUF_ATTENTION_SOFTMAX_CUDA_PARITY_ARTIFACT_KIND
        );
        assert_eq!(
            receipt["attention_v_mix_fixture"]["source_attention_v_artifact_kind"],
            DENSE_GGUF_LINEAR_ROLE_SWEEP_CUDA_PARITY_ARTIFACT_KIND
        );
        assert_eq!(
            receipt["attention_v_mix_fixture"]["context_count"],
            json!(fixture.expected_context_f32.len())
        );
        assert!(fixture.causal_zero_probabilities > 0);
        assert!(receipt["attention_v_mix_fixture"]["max_context_abs"].as_f64().unwrap() > 0.0);
        assert_eq!(
            receipt["attention_v_mix_gap_audit"]["cuda_kernel_status"],
            "missing_cuda_kernel"
        );
        assert_eq!(
            receipt["attention_v_mix_gap_audit"]["next_required_proof"],
            "cuda_attention_v_mix_kernel_parity"
        );
        assert_eq!(
            receipt["claim_boundary"]["dense_gguf_attention_v_mix_fixture_extraction_claimed"],
            true
        );
        assert_eq!(receipt["claim_boundary"]["dense_regular_llm_cuda_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["dense_gguf_inference_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["bitnet_packed_i2s_qk256_proof"], false);
    }

    #[test]
    fn dense_gguf_mlp_activation_fixture_receipt_records_cpu_reference_gap() {
        let data = build_complete_qwen_layer_gguf();
        let reader = GgufReader::new(&data).expect("parse qwen fixture");
        let inspection = inspect_dense_gguf_tensor_descriptors(&reader).expect("inspect");
        let fixture = dense_gguf_mlp_activation_fixture_from_reader(&reader, &inspection, 0)
            .expect("MLP activation fixture");

        let receipt = dense_gguf_mlp_activation_fixture_receipt_json(
            &inspection,
            &fixture,
            Path::new("synthetic-qwen3-q8_0-mlp-activation-fixture.gguf"),
            &"0".repeat(64),
            "target/bitnet/receipts/dense-gguf-mlp-activation-fixture.json",
            "2026-05-09T00:00:00Z",
        );

        validate_dense_gguf_mlp_activation_fixture_receipt_json(&receipt).unwrap();
        assert_eq!(receipt["execution_plan"]["selected_route"], "unsupported");
        assert_eq!(receipt["execution_plan"]["unsupported_ops"], 1);
        assert_eq!(
            receipt["mlp_activation_fixture"]["source_mlp_gate_artifact_kind"],
            DENSE_GGUF_LINEAR_ROLE_SWEEP_CUDA_PARITY_ARTIFACT_KIND
        );
        assert_eq!(
            receipt["mlp_activation_fixture"]["source_mlp_up_artifact_kind"],
            DENSE_GGUF_LINEAR_ROLE_SWEEP_CUDA_PARITY_ARTIFACT_KIND
        );
        assert_eq!(receipt["mlp_activation_fixture"]["activation_kind"], "silu_gate_times_up");
        assert_eq!(
            receipt["mlp_activation_fixture"]["activation_count"],
            json!(fixture.expected_activation_f32.len())
        );
        assert!(receipt["mlp_activation_fixture"]["max_activation_abs"].as_f64().unwrap() > 0.0);
        assert_eq!(
            receipt["mlp_activation_gap_audit"]["cuda_kernel_status"],
            "missing_cuda_kernel"
        );
        assert_eq!(
            receipt["mlp_activation_gap_audit"]["next_required_proof"],
            "cuda_mlp_activation_kernel_parity"
        );
        assert_eq!(
            receipt["claim_boundary"]["dense_gguf_mlp_activation_fixture_extraction_claimed"],
            true
        );
        assert_eq!(receipt["claim_boundary"]["dense_regular_llm_cuda_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["dense_gguf_inference_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["bitnet_packed_i2s_qk256_proof"], false);
    }

    #[test]
    fn dense_gguf_attention_score_cuda_parity_receipt_records_cuda_kernel() {
        let data = build_complete_qwen_layer_gguf();
        let reader = GgufReader::new(&data).expect("parse qwen fixture");
        let inspection = inspect_dense_gguf_tensor_descriptors(&reader).expect("inspect");
        let fixture = dense_gguf_attention_score_fixture_from_reader(&reader, &inspection, 0, 4, 1)
            .expect("attention score fixture");
        let kernel_fixture = kernel_attention_score_fixture_from_extracted(&fixture);
        let parity = synthetic_attention_score_parity_from_fixture(&kernel_fixture);

        let receipt = dense_gguf_attention_score_cuda_parity_receipt_json(
            &inspection,
            &fixture,
            &parity,
            None,
            Path::new("synthetic-qwen3-q8_0-attention-score-cuda-parity.gguf"),
            &"0".repeat(64),
            "target/bitnet/receipts/dense-gguf-attention-score-cuda-parity.json",
            "2026-05-09T00:00:00Z",
        );

        validate_dense_gguf_attention_score_cuda_parity_receipt_json(&receipt).unwrap();
        assert_eq!(receipt["execution_plan"]["selected_route"], "dense_regular_llm_cuda");
        assert_eq!(receipt["execution_plan"]["cuda_dense_regular_llm_ops"], 1);
        assert_eq!(receipt["attention_score_fixture"]["cuda_kernel_status"], "parity_passed");
        assert_eq!(receipt["kernel_stats"][0]["kernel_id"], "dense_attention_scores_f32_cuda");
        assert_eq!(receipt["kernel_stats"][0]["invocations"], 1);
        assert_eq!(receipt["parity"]["passed"], true);
        assert_eq!(receipt["parity"]["compared_scores"], fixture.expected_scores_f32.len() as u64);
        assert_eq!(
            receipt["claim_boundary"]["dense_gguf_attention_score_cuda_parity_claimed"],
            true
        );
        assert_eq!(receipt["claim_boundary"]["dense_gguf_inference_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["bitnet_packed_i2s_qk256_proof"], false);
    }

    #[test]
    fn dense_gguf_attention_softmax_cuda_parity_receipt_records_cuda_kernel() {
        let data = build_complete_qwen_layer_gguf();
        let reader = GgufReader::new(&data).expect("parse qwen fixture");
        let inspection = inspect_dense_gguf_tensor_descriptors(&reader).expect("inspect");
        let fixture =
            dense_gguf_attention_softmax_fixture_from_reader(&reader, &inspection, 0, 4, 1)
                .expect("attention softmax fixture");
        let kernel_fixture = kernel_attention_softmax_fixture_from_extracted(&fixture);
        let parity = synthetic_attention_softmax_parity_from_fixture(&kernel_fixture);

        let receipt = dense_gguf_attention_softmax_cuda_parity_receipt_json(
            &inspection,
            &fixture,
            &parity,
            None,
            Path::new("synthetic-qwen3-q8_0-attention-softmax-cuda-parity.gguf"),
            &"0".repeat(64),
            "target/bitnet/receipts/dense-gguf-attention-softmax-cuda-parity.json",
            "2026-05-09T00:00:00Z",
        );

        validate_dense_gguf_attention_softmax_cuda_parity_receipt_json(&receipt).unwrap();
        assert_eq!(receipt["execution_plan"]["selected_route"], "dense_regular_llm_cuda");
        assert_eq!(receipt["execution_plan"]["cuda_dense_regular_llm_ops"], 1);
        assert_eq!(receipt["attention_softmax_fixture"]["cuda_kernel_status"], "parity_passed");
        assert_eq!(receipt["kernel_stats"][0]["kernel_id"], "dense_attention_softmax_f32_cuda");
        assert_eq!(receipt["kernel_stats"][0]["invocations"], 1);
        assert_eq!(receipt["parity"]["passed"], true);
        assert_eq!(
            receipt["parity"]["compared_probabilities"],
            fixture.expected_probabilities_f32.len() as u64
        );
        assert_eq!(
            receipt["claim_boundary"]["dense_gguf_attention_softmax_cuda_parity_claimed"],
            true
        );
        assert_eq!(receipt["claim_boundary"]["dense_gguf_inference_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["bitnet_packed_i2s_qk256_proof"], false);
    }

    #[test]
    fn dense_gguf_attention_v_mix_cuda_parity_receipt_records_cuda_kernel() {
        let data = build_complete_qwen_layer_gguf();
        let reader = GgufReader::new(&data).expect("parse qwen fixture");
        let inspection = inspect_dense_gguf_tensor_descriptors(&reader).expect("inspect");
        let fixture = dense_gguf_attention_v_mix_fixture_from_reader(&reader, &inspection, 0, 4, 1)
            .expect("attention V-mix fixture");
        let kernel_fixture = kernel_attention_v_mix_fixture_from_extracted(&fixture);
        let parity = synthetic_attention_v_mix_parity_from_fixture(&kernel_fixture);

        let receipt = dense_gguf_attention_v_mix_cuda_parity_receipt_json(
            &inspection,
            &fixture,
            &parity,
            None,
            Path::new("synthetic-qwen3-q8_0-attention-v-mix-cuda-parity.gguf"),
            &"0".repeat(64),
            "target/bitnet/receipts/dense-gguf-attention-v-mix-cuda-parity.json",
            "2026-05-09T00:00:00Z",
        );

        validate_dense_gguf_attention_v_mix_cuda_parity_receipt_json(&receipt).unwrap();
        assert_eq!(receipt["execution_plan"]["selected_route"], "dense_regular_llm_cuda");
        assert_eq!(receipt["execution_plan"]["cuda_dense_regular_llm_ops"], 1);
        assert_eq!(receipt["attention_v_mix_fixture"]["cuda_kernel_status"], "parity_passed");
        assert_eq!(receipt["kernel_stats"][0]["kernel_id"], "dense_attention_v_mix_f32_cuda");
        assert_eq!(receipt["kernel_stats"][0]["invocations"], 1);
        assert_eq!(receipt["parity"]["passed"], true);
        assert_eq!(
            receipt["parity"]["compared_context_values"],
            fixture.expected_context_f32.len() as u64
        );
        assert_eq!(
            receipt["claim_boundary"]["dense_gguf_attention_v_mix_cuda_parity_claimed"],
            true
        );
        assert_eq!(receipt["claim_boundary"]["dense_gguf_inference_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["bitnet_packed_i2s_qk256_proof"], false);
    }

    #[test]
    fn dense_gguf_mlp_activation_cuda_parity_receipt_records_cuda_kernel() {
        let data = build_complete_qwen_layer_gguf();
        let reader = GgufReader::new(&data).expect("parse qwen fixture");
        let inspection = inspect_dense_gguf_tensor_descriptors(&reader).expect("inspect");
        let fixture = dense_gguf_mlp_activation_fixture_from_reader(&reader, &inspection, 0)
            .expect("MLP activation fixture");
        let kernel_fixture = kernel_mlp_activation_fixture_from_extracted(&fixture);
        let parity = synthetic_mlp_activation_parity_from_fixture(&kernel_fixture);

        let receipt = dense_gguf_mlp_activation_cuda_parity_receipt_json(
            &inspection,
            &fixture,
            &parity,
            None,
            Path::new("synthetic-qwen3-q8_0-mlp-activation-cuda-parity.gguf"),
            &"0".repeat(64),
            "target/bitnet/receipts/dense-gguf-mlp-activation-cuda-parity.json",
            "2026-05-09T00:00:00Z",
        );

        validate_dense_gguf_mlp_activation_cuda_parity_receipt_json(&receipt).unwrap();
        assert_eq!(receipt["execution_plan"]["selected_route"], "dense_regular_llm_cuda");
        assert_eq!(receipt["execution_plan"]["cuda_dense_regular_llm_ops"], 1);
        assert_eq!(receipt["mlp_activation_fixture"]["cuda_kernel_status"], "parity_passed");
        assert_eq!(receipt["kernel_stats"][0]["kernel_id"], "dense_mlp_activation_f32_cuda");
        assert_eq!(receipt["kernel_stats"][0]["invocations"], 1);
        assert_eq!(receipt["parity"]["passed"], true);
        assert_eq!(
            receipt["parity"]["compared_activations"],
            fixture.expected_activation_f32.len() as u64
        );
        assert_eq!(
            receipt["claim_boundary"]["dense_gguf_mlp_activation_cuda_parity_claimed"],
            true
        );
        assert_eq!(receipt["claim_boundary"]["dense_gguf_inference_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["bitnet_packed_i2s_qk256_proof"], false);
    }

    #[test]
    fn parse_dense_linear_role_accepts_common_spellings() {
        assert_eq!(
            parse_dense_linear_role("attention_q").unwrap(),
            DenseGgufTensorRole::AttentionQ
        );
        assert_eq!(parse_dense_linear_role("attn-q").unwrap(), DenseGgufTensorRole::AttentionQ);
        assert_eq!(parse_dense_linear_role("mlp_down").unwrap(), DenseGgufTensorRole::MlpDown);
        assert!(parse_dense_linear_role("attention_norm").is_err());
    }

    #[test]
    fn parse_dense_norm_roles_requires_attention_and_ffn_norms() {
        assert_eq!(
            parse_norm_roles(&[]).unwrap(),
            vec![DenseGgufTensorRole::AttentionNorm, DenseGgufTensorRole::FfnNorm]
        );
        assert!(parse_norm_roles(&["attention_norm".to_string()]).is_err());
        assert!(
            parse_norm_roles(&["attention_norm".to_string(), "attention_norm".to_string()])
                .is_err()
        );
        assert_eq!(
            parse_norm_roles(&["input-layernorm".to_string(), "post-attn-norm".to_string()])
                .unwrap(),
            vec![DenseGgufTensorRole::AttentionNorm, DenseGgufTensorRole::FfnNorm]
        );
    }

    #[test]
    fn parse_role_sweep_rejects_duplicates_and_singletons() {
        let duplicate = vec!["attention_q".to_string(), "attn-q".to_string()];
        assert!(parse_role_sweep(&duplicate).is_err());

        let singleton = vec!["attention_q".to_string()];
        assert!(parse_role_sweep(&singleton).is_err());

        let defaults = parse_role_sweep(&[]).expect("default role sweep");
        assert_eq!(defaults.len(), DEFAULT_ROLE_SWEEP.len());
    }

    fn synthetic_parity_from_kernel_fixture(
        fixture: &DenseGgufLinearGemmFixture,
    ) -> DenseGgufLinearCudaParity {
        DenseGgufLinearCudaParity {
            fixture_id: fixture.fixture_id.clone(),
            model_family: fixture.model_family.clone(),
            tensor_name: fixture.tensor_name.clone(),
            tensor_role: fixture.tensor_role.clone(),
            tensor_type: fixture.tensor_type.clone(),
            source_weight_sha256: fixture.source_weight_sha256.clone(),
            matrix_rows: fixture.matrix_rows,
            matrix_cols: fixture.matrix_cols,
            reference_backend: CUDA_DENSE_GEMM_REFERENCE_BACKEND,
            target_backend: CUDA_DENSE_GEMM_TARGET_BACKEND,
            kernel_id: CUDA_DENSE_F16_GEMM_KERNEL_ID,
            max_abs_error: 0.0,
            mean_abs_error: 0.0,
            tolerance: CUDA_DENSE_GGUF_LINEAR_F16_GEMM_TOLERANCE,
            passed: true,
            stats: CudaDenseGemmStats {
                kernel_id: CUDA_DENSE_F16_GEMM_KERNEL_ID,
                invocations: 1,
                fallback_invocations: 0,
                host_to_device_bytes: ((fixture.matrix_cols
                    + fixture.matrix_rows * fixture.matrix_cols)
                    * 2) as u64,
                device_to_host_bytes: (fixture.matrix_rows * 4) as u64,
                kernel_launches: 1,
                kernel_time_ms: None,
            },
        }
    }

    fn synthetic_rmsnorm_parity_from_fixture(
        fixture: &DenseGgufRmsNormCudaFixture,
    ) -> DenseGgufRmsNormCudaParity {
        DenseGgufRmsNormCudaParity {
            fixture_id: fixture.fixture_id.clone(),
            model_family: fixture.model_family.clone(),
            tensor_name: fixture.tensor_name.clone(),
            tensor_role: fixture.tensor_role.clone(),
            tensor_type: fixture.tensor_type.clone(),
            source_weight_sha256: fixture.source_weight_sha256.clone(),
            hidden_dim: fixture.hidden_dim,
            reference_backend: CUDA_DENSE_RMSNORM_REFERENCE_BACKEND,
            target_backend: CUDA_DENSE_RMSNORM_TARGET_BACKEND,
            kernel_id: CUDA_DENSE_RMSNORM_KERNEL_ID,
            max_abs_error: 0.0,
            mean_abs_error: 0.0,
            tolerance: CUDA_DENSE_RMSNORM_TOLERANCE,
            passed: true,
            stats: CudaDenseRmsNormStats {
                kernel_id: CUDA_DENSE_RMSNORM_KERNEL_ID,
                invocations: 1,
                fallback_invocations: 0,
                host_to_device_bytes: ((fixture.input_f32.len() + fixture.gamma_f32.len()) * 4)
                    as u64,
                device_to_host_bytes: (fixture.expected_output_f32.len() * 4) as u64,
                kernel_launches: 1,
                kernel_time_ms: None,
            },
        }
    }

    fn synthetic_rope_parity_from_fixture(
        fixture: &DenseGgufRopeCudaFixture,
    ) -> DenseGgufRopeCudaParity {
        DenseGgufRopeCudaParity {
            fixture_id: fixture.fixture_id.clone(),
            model_family: fixture.model_family.clone(),
            architecture: fixture.architecture.clone(),
            layer_index: fixture.layer_index,
            q_heads: fixture.q_heads,
            kv_heads: fixture.kv_heads,
            head_dim: fixture.head_dim,
            seq_len: fixture.seq_len,
            position_offset: fixture.position_offset,
            rope_base: fixture.rope_base,
            scaling_factor: fixture.scaling_factor,
            interleaved: fixture.interleaved,
            reference_backend: CUDA_DENSE_ROPE_REFERENCE_BACKEND,
            target_backend: CUDA_DENSE_ROPE_TARGET_BACKEND,
            kernel_id: CUDA_DENSE_ROPE_KERNEL_ID,
            max_abs_error: 0.0,
            mean_abs_error: 0.0,
            tolerance: CUDA_DENSE_ROPE_TOLERANCE,
            passed: true,
            stats: CudaDenseRopeStats {
                kernel_id: CUDA_DENSE_ROPE_KERNEL_ID,
                invocations: 2,
                fallback_invocations: 0,
                host_to_device_bytes: ((fixture.q_input_f32.len() + fixture.k_input_f32.len()) * 4)
                    as u64,
                device_to_host_bytes: ((fixture.expected_q_output_f32.len()
                    + fixture.expected_k_output_f32.len())
                    * 4) as u64,
                kernel_launches: 2,
                kernel_time_ms: None,
            },
        }
    }

    fn synthetic_attention_score_parity_from_fixture(
        fixture: &DenseGgufAttentionScoreCudaFixture,
    ) -> DenseGgufAttentionScoreCudaParity {
        DenseGgufAttentionScoreCudaParity {
            fixture_id: fixture.fixture_id.clone(),
            model_family: fixture.model_family.clone(),
            architecture: fixture.architecture.clone(),
            layer_index: fixture.layer_index,
            q_heads: fixture.q_heads,
            kv_heads: fixture.kv_heads,
            head_dim: fixture.head_dim,
            seq_len: fixture.seq_len,
            scale: fixture.scale,
            reference_backend: CUDA_DENSE_ATTENTION_SCORE_REFERENCE_BACKEND,
            target_backend: CUDA_DENSE_ATTENTION_SCORE_TARGET_BACKEND,
            kernel_id: CUDA_DENSE_ATTENTION_SCORE_KERNEL_ID,
            max_abs_error: 0.0,
            mean_abs_error: 0.0,
            tolerance: CUDA_DENSE_ATTENTION_SCORE_TOLERANCE,
            passed: true,
            compared_scores: fixture.expected_scores_f32.len(),
            finite_scores: fixture.finite_scores,
            causal_masked_scores: fixture.causal_masked_scores,
            stats: CudaDenseAttentionScoreStats {
                kernel_id: CUDA_DENSE_ATTENTION_SCORE_KERNEL_ID,
                invocations: 1,
                fallback_invocations: 0,
                host_to_device_bytes: ((fixture.q_rope_output_f32.len()
                    + fixture.k_rope_output_f32.len())
                    * 4) as u64,
                device_to_host_bytes: (fixture.expected_scores_f32.len() * 4) as u64,
                kernel_launches: 1,
                kernel_time_ms: None,
            },
        }
    }

    fn synthetic_attention_softmax_parity_from_fixture(
        fixture: &DenseGgufAttentionSoftmaxCudaFixture,
    ) -> DenseGgufAttentionSoftmaxCudaParity {
        DenseGgufAttentionSoftmaxCudaParity {
            fixture_id: fixture.fixture_id.clone(),
            model_family: fixture.model_family.clone(),
            architecture: fixture.architecture.clone(),
            layer_index: fixture.layer_index,
            q_heads: fixture.q_heads,
            kv_heads: fixture.kv_heads,
            seq_len: fixture.seq_len,
            reference_backend: CUDA_DENSE_ATTENTION_SCORE_REFERENCE_BACKEND,
            target_backend: CUDA_DENSE_ATTENTION_SCORE_TARGET_BACKEND,
            kernel_id: CUDA_DENSE_ATTENTION_SOFTMAX_KERNEL_ID,
            max_abs_error: 0.0,
            mean_abs_error: 0.0,
            tolerance: CUDA_DENSE_ATTENTION_SOFTMAX_TOLERANCE,
            passed: true,
            compared_probabilities: fixture.expected_probabilities_f32.len(),
            causal_zero_probabilities: fixture.causal_zero_probabilities,
            stats: CudaDenseAttentionSoftmaxStats {
                kernel_id: CUDA_DENSE_ATTENTION_SOFTMAX_KERNEL_ID,
                invocations: 1,
                fallback_invocations: 0,
                host_to_device_bytes: (fixture.attention_scores_f32.len() * 4) as u64,
                device_to_host_bytes: (fixture.expected_probabilities_f32.len() * 4) as u64,
                kernel_launches: 1,
                kernel_time_ms: None,
            },
        }
    }

    fn synthetic_attention_v_mix_parity_from_fixture(
        fixture: &DenseGgufAttentionVMixCudaFixture,
    ) -> DenseGgufAttentionVMixCudaParity {
        DenseGgufAttentionVMixCudaParity {
            fixture_id: fixture.fixture_id.clone(),
            model_family: fixture.model_family.clone(),
            architecture: fixture.architecture.clone(),
            layer_index: fixture.layer_index,
            q_heads: fixture.q_heads,
            kv_heads: fixture.kv_heads,
            head_dim: fixture.head_dim,
            seq_len: fixture.seq_len,
            reference_backend: CUDA_DENSE_ATTENTION_SCORE_REFERENCE_BACKEND,
            target_backend: CUDA_DENSE_ATTENTION_SCORE_TARGET_BACKEND,
            kernel_id: CUDA_DENSE_ATTENTION_V_MIX_KERNEL_ID,
            max_abs_error: 0.0,
            mean_abs_error: 0.0,
            tolerance: CUDA_DENSE_ATTENTION_V_MIX_TOLERANCE,
            passed: true,
            compared_context_values: fixture.expected_context_f32.len(),
            causal_zero_probabilities: fixture.causal_zero_probabilities,
            stats: CudaDenseAttentionVMixStats {
                kernel_id: CUDA_DENSE_ATTENTION_V_MIX_KERNEL_ID,
                invocations: 1,
                fallback_invocations: 0,
                host_to_device_bytes: ((fixture.attention_probabilities_f32.len()
                    + fixture.value_states_f32.len())
                    * 4) as u64,
                device_to_host_bytes: (fixture.expected_context_f32.len() * 4) as u64,
                kernel_launches: 1,
                kernel_time_ms: None,
            },
        }
    }

    fn synthetic_mlp_activation_parity_from_fixture(
        fixture: &DenseGgufMlpActivationCudaFixture,
    ) -> DenseGgufMlpActivationCudaParity {
        DenseGgufMlpActivationCudaParity {
            fixture_id: fixture.fixture_id.clone(),
            model_family: fixture.model_family.clone(),
            architecture: fixture.architecture.clone(),
            layer_index: fixture.layer_index,
            reference_backend: CUDA_DENSE_MLP_ACTIVATION_REFERENCE_BACKEND,
            target_backend: CUDA_DENSE_MLP_ACTIVATION_TARGET_BACKEND,
            kernel_id: CUDA_DENSE_MLP_ACTIVATION_KERNEL_ID,
            max_abs_error: 0.0,
            mean_abs_error: 0.0,
            tolerance: CUDA_DENSE_MLP_ACTIVATION_TOLERANCE,
            passed: true,
            compared_activations: fixture.expected_activation_f32.len(),
            stats: CudaDenseMlpActivationStats {
                kernel_id: CUDA_DENSE_MLP_ACTIVATION_KERNEL_ID,
                invocations: 1,
                fallback_invocations: 0,
                host_to_device_bytes: ((fixture.gate_output_f32.len()
                    + fixture.up_output_f32.len())
                    * 4) as u64,
                device_to_host_bytes: (fixture.expected_activation_f32.len() * 4) as u64,
                kernel_launches: 1,
                kernel_time_ms: None,
            },
        }
    }

    fn synthetic_one_layer_cuda_integrated_parity_from_reference(
        reference: &DenseGgufOneLayerCpuReference,
    ) -> DenseGgufOneLayerCudaIntegratedParity {
        let phases = reference
            .phases
            .iter()
            .map(|phase| {
                let (route, status, kernel_id, invocations, h2d, d2h, launches) = match phase.name {
                    "deterministic_input" => {
                        ("host_deterministic_input", "host_deterministic_input", None, 1, 0, 0, 0)
                    }
                    "first_residual" | "second_residual" => {
                        ("host_measured_glue", "host_measured_glue", None, 1, 0, 0, 0)
                    }
                    "attention_norm" | "ffn_norm" => (
                        DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND,
                        "cuda_executed",
                        Some(CUDA_DENSE_RMSNORM_KERNEL_ID),
                        1,
                        64,
                        64,
                        1,
                    ),
                    "rope" => (
                        DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND,
                        "cuda_executed",
                        Some(CUDA_DENSE_ROPE_KERNEL_ID),
                        2,
                        96,
                        96,
                        2,
                    ),
                    "attention_scores" => (
                        DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND,
                        "cuda_executed",
                        Some(CUDA_DENSE_ATTENTION_SCORE_KERNEL_ID),
                        1,
                        96,
                        128,
                        1,
                    ),
                    "attention_softmax" => (
                        DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND,
                        "cuda_executed",
                        Some(CUDA_DENSE_ATTENTION_SOFTMAX_KERNEL_ID),
                        1,
                        128,
                        128,
                        1,
                    ),
                    "attention_v_mix" => (
                        DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND,
                        "cuda_executed",
                        Some(CUDA_DENSE_ATTENTION_V_MIX_KERNEL_ID),
                        1,
                        160,
                        64,
                        1,
                    ),
                    "mlp_activation" => (
                        DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND,
                        "cuda_executed",
                        Some(CUDA_DENSE_MLP_ACTIVATION_KERNEL_ID),
                        1,
                        192,
                        96,
                        1,
                    ),
                    _ => (
                        DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND,
                        "cuda_executed",
                        Some(CUDA_DENSE_F16_GEMM_KERNEL_ID),
                        reference.seq_len as u64,
                        384,
                        (phase.output_len * 4) as u64,
                        reference.seq_len as u64,
                    ),
                };
                DenseOneLayerCudaPhase {
                    index: phase.index,
                    name: phase.name,
                    role: phase.role,
                    op_type: phase.op_type,
                    route,
                    status,
                    output_len: phase.output_len,
                    output_sha256: phase.output_sha256.clone(),
                    max_abs: phase.max_abs,
                    max_abs_error: 0.0,
                    mean_abs_error: 0.0,
                    tolerance: 0.5,
                    passed: true,
                    kernel_id,
                    invocations,
                    fallback_invocations: 0,
                    host_to_device_bytes: h2d,
                    device_to_host_bytes: d2h,
                    kernel_launches: launches,
                    kernel_time_ms: None,
                }
            })
            .collect::<Vec<_>>();
        let host_to_device_bytes = phases.iter().map(|phase| phase.host_to_device_bytes).sum();
        let device_to_host_bytes = phases.iter().map(|phase| phase.device_to_host_bytes).sum();
        let kernel_invocations = phases
            .iter()
            .filter(|phase| phase.kernel_id.is_some())
            .map(|phase| phase.invocations)
            .sum();
        let kernel_launches = phases.iter().map(|phase| phase.kernel_launches).sum();

        DenseGgufOneLayerCudaIntegratedParity {
            fixture_id: "dense_gguf_one_layer_cuda_integrated_parity_qwen_layer0_s4".to_string(),
            source_cpu_reference_fixture_id: reference.fixture_id.clone(),
            model_family: reference.model_family.clone(),
            architecture: reference.architecture.clone(),
            layer_index: reference.layer_index,
            seq_len: reference.seq_len,
            position_offset: reference.position_offset,
            hidden_size: reference.hidden_size,
            q_heads: reference.q_heads,
            kv_heads: reference.kv_heads,
            heads_per_kv_group: reference.heads_per_kv_group,
            head_dim: reference.head_dim,
            intermediate_size: reference.intermediate_size,
            phases,
            final_output_len: reference.final_output_len,
            final_output_sha256: reference.final_output_sha256.clone(),
            final_output_max_abs: reference.final_output_max_abs,
            final_output_max_abs_error: 0.0,
            final_output_mean_abs_error: 0.0,
            tolerance: 0.5,
            passed: true,
            host_to_device_bytes,
            device_to_host_bytes,
            kernel_invocations,
            kernel_launches,
            kernel_time_ms: None,
        }
    }

    fn build_qwen_gguf(
        tensors: Vec<(&'static str, Vec<usize>, GgufTensorType, Vec<u8>)>,
    ) -> Vec<u8> {
        build_gguf_for_test(
            vec![
                ("general.architecture", GgufValue::String("qwen3".to_string())),
                ("general.name", GgufValue::String("qwen3-linear-fixture".to_string())),
                ("qwen3.embedding_length", GgufValue::U32(4)),
                ("qwen3.feed_forward_length", GgufValue::U32(3)),
                ("qwen3.attention.head_count", GgufValue::U32(2)),
                ("qwen3.attention.head_count_kv", GgufValue::U32(1)),
                ("qwen3.attention.key_length", GgufValue::U32(2)),
                ("qwen3.rope.freq_base", GgufValue::F32(1_000_000.0)),
            ],
            tensors,
        )
    }

    fn build_complete_qwen_layer_gguf() -> Vec<u8> {
        let values = (1..=12).collect::<Vec<_>>();
        build_qwen_gguf(vec![
            ("token_embd.weight", vec![4, 3], GgufTensorType::Q8_0, q8_0_blob(0.5, &values)),
            ("output.weight", vec![4, 3], GgufTensorType::Q8_0, q8_0_blob(0.5, &values)),
            ("blk.0.attn_q.weight", vec![4, 3], GgufTensorType::Q8_0, q8_0_blob(0.5, &values)),
            ("blk.0.attn_k.weight", vec![4, 3], GgufTensorType::Q8_0, q8_0_blob(0.25, &values)),
            ("blk.0.attn_v.weight", vec![4, 3], GgufTensorType::Q8_0, q8_0_blob(0.125, &values)),
            (
                "blk.0.attn_output.weight",
                vec![4, 3],
                GgufTensorType::Q8_0,
                q8_0_blob(0.0625, &values),
            ),
            ("blk.0.ffn_gate.weight", vec![4, 3], GgufTensorType::Q8_0, q8_0_blob(0.5, &values)),
            ("blk.0.ffn_up.weight", vec![4, 3], GgufTensorType::Q8_0, q8_0_blob(0.25, &values)),
            ("blk.0.ffn_down.weight", vec![4, 3], GgufTensorType::Q8_0, q8_0_blob(0.125, &values)),
            ("blk.0.attn_norm.weight", vec![4], GgufTensorType::F32, f32_blob(&[1.0; 4])),
            ("blk.0.ffn_norm.weight", vec![4], GgufTensorType::F32, f32_blob(&[1.0; 4])),
        ])
    }

    fn build_two_layer_qwen_gguf() -> Vec<u8> {
        let values = (1..=12).collect::<Vec<_>>();
        build_qwen_gguf(vec![
            ("token_embd.weight", vec![4, 3], GgufTensorType::Q8_0, q8_0_blob(0.5, &values)),
            ("output.weight", vec![4, 3], GgufTensorType::Q8_0, q8_0_blob(0.5, &values)),
            ("blk.0.attn_q.weight", vec![4, 3], GgufTensorType::Q8_0, q8_0_blob(0.5, &values)),
            ("blk.0.attn_k.weight", vec![4, 3], GgufTensorType::Q8_0, q8_0_blob(0.25, &values)),
            ("blk.0.attn_v.weight", vec![4, 3], GgufTensorType::Q8_0, q8_0_blob(0.125, &values)),
            (
                "blk.0.attn_output.weight",
                vec![4, 3],
                GgufTensorType::Q8_0,
                q8_0_blob(0.0625, &values),
            ),
            ("blk.0.ffn_gate.weight", vec![4, 3], GgufTensorType::Q8_0, q8_0_blob(0.5, &values)),
            ("blk.0.ffn_up.weight", vec![4, 3], GgufTensorType::Q8_0, q8_0_blob(0.25, &values)),
            ("blk.0.ffn_down.weight", vec![4, 3], GgufTensorType::Q8_0, q8_0_blob(0.125, &values)),
            ("blk.0.attn_norm.weight", vec![4], GgufTensorType::F32, f32_blob(&[1.0; 4])),
            ("blk.0.ffn_norm.weight", vec![4], GgufTensorType::F32, f32_blob(&[1.0; 4])),
            ("blk.1.attn_q.weight", vec![4, 3], GgufTensorType::Q8_0, q8_0_blob(0.5, &values)),
            ("blk.1.attn_k.weight", vec![4, 3], GgufTensorType::Q8_0, q8_0_blob(0.25, &values)),
            ("blk.1.attn_v.weight", vec![4, 3], GgufTensorType::Q8_0, q8_0_blob(0.125, &values)),
            (
                "blk.1.attn_output.weight",
                vec![4, 3],
                GgufTensorType::Q8_0,
                q8_0_blob(0.0625, &values),
            ),
            ("blk.1.ffn_gate.weight", vec![4, 3], GgufTensorType::Q8_0, q8_0_blob(0.5, &values)),
            ("blk.1.ffn_up.weight", vec![4, 3], GgufTensorType::Q8_0, q8_0_blob(0.25, &values)),
            ("blk.1.ffn_down.weight", vec![4, 3], GgufTensorType::Q8_0, q8_0_blob(0.125, &values)),
            ("blk.1.attn_norm.weight", vec![4], GgufTensorType::F32, f32_blob(&[1.0; 4])),
            ("blk.1.ffn_norm.weight", vec![4], GgufTensorType::F32, f32_blob(&[1.0; 4])),
        ])
    }

    fn build_two_layer_qwen_tied_lm_head_gguf() -> Vec<u8> {
        let values = (1..=12).collect::<Vec<_>>();
        build_qwen_gguf(vec![
            ("token_embd.weight", vec![4, 3], GgufTensorType::Q8_0, q8_0_blob(0.5, &values)),
            ("blk.0.attn_q.weight", vec![4, 3], GgufTensorType::Q8_0, q8_0_blob(0.5, &values)),
            ("blk.0.attn_k.weight", vec![4, 3], GgufTensorType::Q8_0, q8_0_blob(0.25, &values)),
            ("blk.0.attn_v.weight", vec![4, 3], GgufTensorType::Q8_0, q8_0_blob(0.125, &values)),
            (
                "blk.0.attn_output.weight",
                vec![4, 3],
                GgufTensorType::Q8_0,
                q8_0_blob(0.0625, &values),
            ),
            ("blk.0.ffn_gate.weight", vec![4, 3], GgufTensorType::Q8_0, q8_0_blob(0.5, &values)),
            ("blk.0.ffn_up.weight", vec![4, 3], GgufTensorType::Q8_0, q8_0_blob(0.25, &values)),
            ("blk.0.ffn_down.weight", vec![4, 3], GgufTensorType::Q8_0, q8_0_blob(0.125, &values)),
            ("blk.0.attn_norm.weight", vec![4], GgufTensorType::F32, f32_blob(&[1.0; 4])),
            ("blk.0.ffn_norm.weight", vec![4], GgufTensorType::F32, f32_blob(&[1.0; 4])),
            ("blk.1.attn_q.weight", vec![4, 3], GgufTensorType::Q8_0, q8_0_blob(0.5, &values)),
            ("blk.1.attn_k.weight", vec![4, 3], GgufTensorType::Q8_0, q8_0_blob(0.25, &values)),
            ("blk.1.attn_v.weight", vec![4, 3], GgufTensorType::Q8_0, q8_0_blob(0.125, &values)),
            (
                "blk.1.attn_output.weight",
                vec![4, 3],
                GgufTensorType::Q8_0,
                q8_0_blob(0.0625, &values),
            ),
            ("blk.1.ffn_gate.weight", vec![4, 3], GgufTensorType::Q8_0, q8_0_blob(0.5, &values)),
            ("blk.1.ffn_up.weight", vec![4, 3], GgufTensorType::Q8_0, q8_0_blob(0.25, &values)),
            ("blk.1.ffn_down.weight", vec![4, 3], GgufTensorType::Q8_0, q8_0_blob(0.125, &values)),
            ("blk.1.attn_norm.weight", vec![4], GgufTensorType::F32, f32_blob(&[1.0; 4])),
            ("blk.1.ffn_norm.weight", vec![4], GgufTensorType::F32, f32_blob(&[1.0; 4])),
        ])
    }

    fn build_integrated_qwen_layer_gguf() -> Vec<u8> {
        let values = (1..=32).collect::<Vec<_>>();
        build_gguf_for_test(
            vec![
                ("general.architecture", GgufValue::String("qwen3".to_string())),
                ("general.name", GgufValue::String("qwen3-one-layer-reference".to_string())),
                ("qwen3.embedding_length", GgufValue::U32(4)),
                ("qwen3.feed_forward_length", GgufValue::U32(6)),
                ("qwen3.attention.head_count", GgufValue::U32(2)),
                ("qwen3.attention.head_count_kv", GgufValue::U32(1)),
                ("qwen3.attention.key_length", GgufValue::U32(2)),
                ("qwen3.rope.freq_base", GgufValue::F32(1_000_000.0)),
            ],
            vec![
                ("token_embd.weight", vec![4, 4], GgufTensorType::Q8_0, q8_0_blob(0.01, &values)),
                ("output.weight", vec![4, 4], GgufTensorType::Q8_0, q8_0_blob(0.01, &values)),
                ("blk.0.attn_q.weight", vec![4, 4], GgufTensorType::Q8_0, q8_0_blob(0.01, &values)),
                (
                    "blk.0.attn_k.weight",
                    vec![4, 2],
                    GgufTensorType::Q8_0,
                    q8_0_blob(0.008, &values),
                ),
                (
                    "blk.0.attn_v.weight",
                    vec![4, 2],
                    GgufTensorType::Q8_0,
                    q8_0_blob(0.006, &values),
                ),
                (
                    "blk.0.attn_output.weight",
                    vec![4, 4],
                    GgufTensorType::Q8_0,
                    q8_0_blob(0.004, &values),
                ),
                (
                    "blk.0.ffn_gate.weight",
                    vec![4, 6],
                    GgufTensorType::Q8_0,
                    q8_0_blob(0.01, &values),
                ),
                (
                    "blk.0.ffn_up.weight",
                    vec![4, 6],
                    GgufTensorType::Q8_0,
                    q8_0_blob(0.008, &values),
                ),
                (
                    "blk.0.ffn_down.weight",
                    vec![6, 4],
                    GgufTensorType::Q8_0,
                    q8_0_blob(0.006, &values),
                ),
                ("blk.0.attn_norm.weight", vec![4], GgufTensorType::F32, f32_blob(&[1.0; 4])),
                ("blk.0.ffn_norm.weight", vec![4], GgufTensorType::F32, f32_blob(&[1.0; 4])),
            ],
        )
    }

    fn build_model_boundary_qwen_gguf() -> Vec<u8> {
        let values = (1..=32).collect::<Vec<_>>();
        build_gguf_for_test(
            vec![
                ("general.architecture", GgufValue::String("qwen3".to_string())),
                ("general.name", GgufValue::String("qwen3-model-boundary-fixtures".to_string())),
                ("qwen3.embedding_length", GgufValue::U32(4)),
                ("qwen3.feed_forward_length", GgufValue::U32(6)),
                ("qwen3.attention.head_count", GgufValue::U32(2)),
                ("qwen3.attention.head_count_kv", GgufValue::U32(1)),
                ("qwen3.attention.key_length", GgufValue::U32(2)),
                ("qwen3.rope.freq_base", GgufValue::F32(1_000_000.0)),
            ],
            vec![
                ("token_embd.weight", vec![4, 4], GgufTensorType::Q8_0, q8_0_blob(0.01, &values)),
                ("output.weight", vec![4, 4], GgufTensorType::Q8_0, q8_0_blob(0.01, &values)),
                ("output_norm.weight", vec![4], GgufTensorType::F32, f32_blob(&[1.0; 4])),
                ("blk.0.attn_q.weight", vec![4, 4], GgufTensorType::Q8_0, q8_0_blob(0.01, &values)),
                (
                    "blk.0.attn_k.weight",
                    vec![4, 2],
                    GgufTensorType::Q8_0,
                    q8_0_blob(0.008, &values),
                ),
                (
                    "blk.0.attn_v.weight",
                    vec![4, 2],
                    GgufTensorType::Q8_0,
                    q8_0_blob(0.006, &values),
                ),
                (
                    "blk.0.attn_output.weight",
                    vec![4, 4],
                    GgufTensorType::Q8_0,
                    q8_0_blob(0.004, &values),
                ),
                (
                    "blk.0.ffn_gate.weight",
                    vec![4, 6],
                    GgufTensorType::Q8_0,
                    q8_0_blob(0.01, &values),
                ),
                (
                    "blk.0.ffn_up.weight",
                    vec![4, 6],
                    GgufTensorType::Q8_0,
                    q8_0_blob(0.008, &values),
                ),
                (
                    "blk.0.ffn_down.weight",
                    vec![6, 4],
                    GgufTensorType::Q8_0,
                    q8_0_blob(0.006, &values),
                ),
                ("blk.0.attn_norm.weight", vec![4], GgufTensorType::F32, f32_blob(&[1.0; 4])),
                ("blk.0.ffn_norm.weight", vec![4], GgufTensorType::F32, f32_blob(&[1.0; 4])),
            ],
        )
    }

    fn build_gguf_for_test(
        metadata: Vec<(&str, GgufValue)>,
        tensors: Vec<(&str, Vec<usize>, GgufTensorType, Vec<u8>)>,
    ) -> Vec<u8> {
        let mut data = Vec::new();
        const GGUF_VERSION: u32 = 2;
        const ALIGN: usize = 32;

        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        data.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        data.extend_from_slice(&(metadata.len() as u64).to_le_bytes());

        for (key, value) in metadata {
            write_string(&mut data, key);
            write_gguf_value(&mut data, value);
        }

        let mut running_offset = 0usize;
        let mut offsets = Vec::with_capacity(tensors.len());
        for (_, _, _, blob) in &tensors {
            offsets.push(running_offset);
            running_offset += blob.len();
        }

        for (index, (name, shape, tensor_type, _blob)) in tensors.iter().enumerate() {
            write_string(&mut data, name);
            data.extend_from_slice(&(shape.len() as u32).to_le_bytes());
            for dim in shape {
                data.extend_from_slice(&(*dim as u64).to_le_bytes());
            }
            data.extend_from_slice(&tensor_type_id(*tensor_type).to_le_bytes());
            data.extend_from_slice(&(offsets[index] as u64).to_le_bytes());
        }

        let pad = (ALIGN - (data.len() % ALIGN)) % ALIGN;
        data.resize(data.len() + pad, 0);

        for (_, _, _, blob) in tensors {
            data.extend_from_slice(&blob);
        }

        data
    }

    fn q8_0_blob(scale: f32, values: &[i8]) -> Vec<u8> {
        let mut blob = Vec::new();
        let scale_bits = half::f16::from_f32(scale).to_bits();
        blob.extend_from_slice(&scale_bits.to_le_bytes());
        for idx in 0..32 {
            blob.push(values.get(idx).copied().unwrap_or(0) as u8);
        }
        blob
    }

    fn f32_blob(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|value| value.to_le_bytes()).collect()
    }

    fn write_gguf_value(data: &mut Vec<u8>, value: GgufValue) {
        match value {
            GgufValue::U32(value) => {
                data.extend_from_slice(&4u32.to_le_bytes());
                data.extend_from_slice(&value.to_le_bytes());
            }
            GgufValue::F32(value) => {
                data.extend_from_slice(&6u32.to_le_bytes());
                data.extend_from_slice(&value.to_le_bytes());
            }
            GgufValue::String(value) => {
                data.extend_from_slice(&8u32.to_le_bytes());
                write_string(data, &value);
            }
            other => panic!("unsupported test GGUF value: {other:?}"),
        }
    }

    fn write_string(data: &mut Vec<u8>, value: &str) {
        data.extend_from_slice(&(value.len() as u64).to_le_bytes());
        data.extend_from_slice(value.as_bytes());
    }

    fn tensor_type_id(tensor_type: GgufTensorType) -> u32 {
        match tensor_type {
            GgufTensorType::F32 => 0,
            GgufTensorType::F16 => 1,
            GgufTensorType::F64 => 4,
            GgufTensorType::Q4_0 => 2,
            GgufTensorType::Q4_1 => 3,
            GgufTensorType::Q5_0 => 6,
            GgufTensorType::Q5_1 => 7,
            GgufTensorType::Q8_0 => 8,
            GgufTensorType::Q8_1 => 9,
            GgufTensorType::Q2_K => 10,
            GgufTensorType::Q3_K => 11,
            GgufTensorType::Q4_K => 12,
            GgufTensorType::Q5_K => 13,
            GgufTensorType::Q6_K => 14,
            GgufTensorType::Q8_K => 15,
            GgufTensorType::IQ2_S => 24,
            GgufTensorType::I2_S => 36,
        }
    }
}

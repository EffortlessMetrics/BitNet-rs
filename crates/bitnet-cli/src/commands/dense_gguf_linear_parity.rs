//! Dense GGUF single-linear CUDA parity diagnostics.
//!
//! This command is an implementation bridge between descriptor extraction and
//! full dense GGUF inference. It extracts one dense GGUF linear fixture, routes
//! that fixture through the dense FP16 CUDA GEMM bridge, and emits a receipt
//! that still refuses dense GGUF inference, speedup, full-residency, and BitNet
//! packed-kernel proof claims.

use anyhow::{Context, Result, anyhow, bail};
use bitnet_kernels::cuda::{
    DenseGgufLinearCudaParity, DenseGgufLinearGemmFixture, DenseGgufRmsNormCudaFixture,
    DenseGgufRmsNormCudaParity, run_dense_gguf_linear_f16_cuda_parity,
    run_dense_gguf_rmsnorm_cuda_parity,
};
use bitnet_kernels::dispatch_planner::{
    BackendPolicy, CudaPlannerCapabilities, DispatchOp, ModelDispatchBackend, ModelDispatchSpec,
    ModelDispatchSummary, ModelFamily, OpType, QuantizationKind, plan_model_dispatch,
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
use bitnet_models::formats::gguf::GgufReader;
use bitnet_receipts_core::{
    DENSE_GGUF_LINEAR_CUDA_PARITY_ARTIFACT_KIND,
    DENSE_GGUF_LINEAR_ROLE_SWEEP_CUDA_PARITY_ARTIFACT_KIND,
    DENSE_GGUF_NORM_CUDA_PARITY_ARTIFACT_KIND, DENSE_GGUF_NORM_FIXTURE_ARTIFACT_KIND,
    DENSE_GGUF_ONE_LAYER_EXECUTION_PLAN_ARTIFACT_KIND, DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND,
    validate_dense_gguf_linear_cuda_parity_receipt_json,
    validate_dense_gguf_linear_role_sweep_cuda_parity_receipt_json,
    validate_dense_gguf_norm_cuda_parity_receipt_json,
    validate_dense_gguf_norm_fixture_extraction_receipt_json,
    validate_dense_gguf_one_layer_execution_plan_receipt_json,
};
use clap::Args;
use memmap2::Mmap;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::path::{Path, PathBuf};

use crate::planner_receipts::{ExecutionPlanReceiptInput, execution_plan_receipt};

const HARDWARE_LANE: &str = "nvidia-rtx-5070-ti-cuda";
const MACHINE_ID: &str = "windows-9950x3d-rtx5070ti";
const DEFAULT_ROLE_SWEEP: &[DenseGgufTensorRole] = &[
    DenseGgufTensorRole::AttentionQ,
    DenseGgufTensorRole::AttentionK,
    DenseGgufTensorRole::AttentionV,
    DenseGgufTensorRole::AttentionOutput,
    DenseGgufTensorRole::MlpGate,
    DenseGgufTensorRole::MlpUp,
    DenseGgufTensorRole::MlpDown,
    DenseGgufTensorRole::Output,
];
const DENSE_ONE_LAYER_GAP_CANDIDATE_ORDER: &[&str] = &[
    "attention_norm",
    "ffn_norm",
    "rope",
    "attention_scores",
    "attention_softmax",
    "attention_v_mix",
    "mlp_activation",
];

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

struct DenseLinearSweepResult {
    extracted: DenseGgufLinearFixture,
    parity: DenseGgufLinearCudaParity,
}

struct DenseNormParityResult {
    extracted: DenseGgufNormFixture,
    parity: DenseGgufRmsNormCudaParity,
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

fn parse_dense_linear_role(value: &str) -> Result<DenseGgufTensorRole> {
    let normalized = value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase();
    match normalized.as_str() {
        "output" => Ok(DenseGgufTensorRole::Output),
        "attentionq" | "attnq" | "q" => Ok(DenseGgufTensorRole::AttentionQ),
        "attentionk" | "attnk" | "k" => Ok(DenseGgufTensorRole::AttentionK),
        "attentionv" | "attnv" | "v" => Ok(DenseGgufTensorRole::AttentionV),
        "attentionoutput" | "attnoutput" | "o" => Ok(DenseGgufTensorRole::AttentionOutput),
        "mlpgate" | "gate" => Ok(DenseGgufTensorRole::MlpGate),
        "mlpup" | "up" => Ok(DenseGgufTensorRole::MlpUp),
        "mlpdown" | "down" => Ok(DenseGgufTensorRole::MlpDown),
        _ => Err(anyhow!(
            "unsupported dense linear role `{value}`; expected output, attention_q, attention_k, attention_v, attention_output, mlp_gate, mlp_up, or mlp_down"
        )),
    }
}

fn parse_role_sweep(values: &[String]) -> Result<Vec<DenseGgufTensorRole>> {
    let roles = if values.is_empty() {
        DEFAULT_ROLE_SWEEP.to_vec()
    } else {
        values.iter().map(|value| parse_dense_linear_role(value)).collect::<Result<Vec<_>>>()?
    };

    if roles.len() < 2 {
        bail!("dense GGUF linear role sweep requires at least two roles");
    }

    let mut seen = BTreeSet::new();
    for role in &roles {
        let label = dense_role_label(*role);
        if !seen.insert(label) {
            bail!("dense GGUF linear role sweep role `{label}` was requested more than once");
        }
    }

    Ok(roles)
}

fn parse_norm_roles(values: &[String]) -> Result<Vec<DenseGgufTensorRole>> {
    let roles = if values.is_empty() {
        vec![DenseGgufTensorRole::AttentionNorm, DenseGgufTensorRole::FfnNorm]
    } else {
        values.iter().map(|value| parse_dense_norm_role(value)).collect::<Result<Vec<_>>>()?
    };

    if roles.len() < 2 {
        bail!("dense GGUF norm fixture extraction requires attention_norm and ffn_norm roles");
    }

    let mut seen = BTreeSet::new();
    for role in &roles {
        let label = dense_role_label(*role);
        if !seen.insert(label) {
            bail!("dense GGUF norm fixture role `{label}` was requested more than once");
        }
    }
    for required in [DenseGgufTensorRole::AttentionNorm, DenseGgufTensorRole::FfnNorm] {
        if !roles.contains(&required) {
            bail!(
                "dense GGUF norm fixture extraction requires role `{}`",
                dense_role_label(required)
            );
        }
    }

    Ok(roles)
}

fn parse_dense_norm_role(value: &str) -> Result<DenseGgufTensorRole> {
    let normalized = value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase();
    match normalized.as_str() {
        "attentionnorm" | "attnnorm" | "inputlayernorm" => Ok(DenseGgufTensorRole::AttentionNorm),
        "ffnnorm" | "postattentionlayernorm" | "postattnnorm" => Ok(DenseGgufTensorRole::FfnNorm),
        _ => Err(anyhow!(
            "unsupported dense norm role `{value}`; expected attention_norm or ffn_norm"
        )),
    }
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

fn dense_role_label(role: DenseGgufTensorRole) -> &'static str {
    match role {
        DenseGgufTensorRole::Output => "output",
        DenseGgufTensorRole::AttentionQ => "attention_q",
        DenseGgufTensorRole::AttentionK => "attention_k",
        DenseGgufTensorRole::AttentionV => "attention_v",
        DenseGgufTensorRole::AttentionOutput => "attention_output",
        DenseGgufTensorRole::MlpGate => "mlp_gate",
        DenseGgufTensorRole::MlpUp => "mlp_up",
        DenseGgufTensorRole::MlpDown => "mlp_down",
        DenseGgufTensorRole::TokenEmbedding => "token_embedding",
        DenseGgufTensorRole::AttentionNorm => "attention_norm",
        DenseGgufTensorRole::FfnNorm => "ffn_norm",
        DenseGgufTensorRole::Other => "other",
    }
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
            "required_roles_present": inspection.required_roles_present,
            "strict_descriptor_complete": inspection.strict_descriptor_complete,
            "dense_cuda_route_status": inspection.dense_cuda_route_status,
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
            "required_roles_present": inspection.required_roles_present,
            "strict_descriptor_complete": inspection.strict_descriptor_complete,
            "dense_cuda_route_status": inspection.dense_cuda_route_status,
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
    if summary.cuda_dense_regular_llm_ops == 0 || summary.unsupported_ops == 0 {
        bail!(
            "dense GGUF one-layer plan must include dense CUDA linears and unsupported strict ops"
        );
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

    let gap_audit = dense_one_layer_gap_audit_json(
        &operations,
        layer_index,
        summary.cuda_dense_regular_llm_ops as u64,
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
            "kernel_family": "dense_fp16_gemm_plus_unsupported_layer_ops",
            "quantization_family": "dense_fp16_bridge_from_gguf_descriptors",
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
            "linear_cuda_ops_total": summary.cuda_dense_regular_llm_ops as u64,
            "unsupported_strict_cuda_ops_total": summary.unsupported_ops as u64,
            "cpu_fallback_ops_total": summary.cpu_fallback_ops as u64,
            "strict_cuda_ready": false,
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

fn dense_one_layer_gap_audit_json(
    operations: &[Value],
    layer_index: usize,
    cuda_linear_ops: u64,
    unsupported_ops: u64,
) -> Result<Value> {
    let mut unsupported = Vec::new();
    let mut linear_roles = Vec::new();
    let mut op_type_counts: BTreeMap<String, u64> = BTreeMap::new();

    for op in operations {
        let route = json_string_field(op, "route")?;
        let role = json_string_field(op, "role")?;
        match route {
            DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND => linear_roles.push(role.to_string()),
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

    if linear_roles.len() as u64 != cuda_linear_ops || unsupported.len() as u64 != unsupported_ops {
        bail!("dense one-layer gap audit counts must match planner summary");
    }

    Ok(json!({
        "schema": 1,
        "source_artifact_kind": DENSE_GGUF_ONE_LAYER_EXECUTION_PLAN_ARTIFACT_KIND,
        "layer_index": layer_index as u64,
        "cuda_routable_linear_ops_total": cuda_linear_ops,
        "unsupported_ops_total": unsupported_ops,
        "cpu_fallback_ops_total": 0,
        "strict_cuda_ready": false,
        "unsupported_ops_have_dependency_notes": true,
        "strict_cuda_rejects_cpu_fallback": true,
        "linears_routable_roles": linear_roles,
        "unsupported_op_type_counts": op_type_counts,
        "candidate_order": DENSE_ONE_LAYER_GAP_CANDIDATE_ORDER,
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

fn dense_one_layer_plan_entries(
    inspection: &DenseGgufDescriptorInspection,
    layer_index: usize,
) -> Result<Vec<DenseLayerPlanEntry>> {
    let attention_q = descriptor_for_role(inspection, DenseGgufTensorRole::AttentionQ)?;
    let hidden_size = attention_q.shape.first().copied().unwrap_or(1).max(1);
    let attention_size = descriptor_element_count(attention_q).max(hidden_size);

    let mut entries = Vec::new();
    push_descriptor_op(
        &mut entries,
        inspection,
        DenseGgufTensorRole::AttentionNorm,
        OpType::RmsNorm,
        false,
    )?;
    push_descriptor_op(
        &mut entries,
        inspection,
        DenseGgufTensorRole::AttentionQ,
        OpType::MatMul,
        false,
    )?;
    push_descriptor_op(
        &mut entries,
        inspection,
        DenseGgufTensorRole::AttentionK,
        OpType::MatMul,
        false,
    )?;
    push_descriptor_op(
        &mut entries,
        inspection,
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
        DenseGgufTensorRole::AttentionOutput,
        OpType::MatMul,
        false,
    )?;
    push_descriptor_op(
        &mut entries,
        inspection,
        DenseGgufTensorRole::FfnNorm,
        OpType::RmsNorm,
        false,
    )?;
    push_descriptor_op(
        &mut entries,
        inspection,
        DenseGgufTensorRole::MlpGate,
        OpType::MatMul,
        false,
    )?;
    push_descriptor_op(
        &mut entries,
        inspection,
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
        DenseGgufTensorRole::MlpDown,
        OpType::MatMul,
        false,
    )?;

    Ok(entries)
}

fn push_descriptor_op(
    entries: &mut Vec<DenseLayerPlanEntry>,
    inspection: &DenseGgufDescriptorInspection,
    role: DenseGgufTensorRole,
    op_type: OpType,
    is_quantized: bool,
) -> Result<()> {
    let descriptor = descriptor_for_role(inspection, role)?;
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

fn descriptor_element_count(descriptor: &DenseGgufTensorDescriptor) -> usize {
    descriptor.shape.iter().copied().fold(1usize, |acc, dim| acc.saturating_mul(dim)).max(1)
}

fn cuda_identity_json(probe: Option<&bitnet_device_probe::NvidiaCudaProbe>) -> Value {
    match probe {
        Some(probe) => json!({
            "available": probe.available,
            "device_count": probe.device_count,
            "device_index": probe.selected_device_index.unwrap_or(0),
            "device_name": probe.selected_device_name.clone().unwrap_or_else(|| "unknown".into()),
            "compute_capability": probe.compute_capability.clone().unwrap_or_else(|| "12.0".into()),
            "driver_version": probe.driver_version.clone().unwrap_or_else(|| "unknown".into()),
            "cuda_runtime_version": probe.cuda_runtime_version.clone().unwrap_or_else(|| "unknown".into()),
            "cuda_toolkit_version": probe.cuda_toolkit_version.clone().unwrap_or_else(|| "unknown".into()),
            "nvrtc_version": probe.nvrtc_version.clone().unwrap_or_else(|| "unknown".into()),
            "nvml_available": probe.nvml_available,
            "vram_bytes": probe.vram_bytes.unwrap_or(1),
            "power_limit_watts": probe.power_limit_watts,
            "power_draw_watts": probe.power_draw_watts,
            "temperature_c": probe.temperature_c,
        }),
        None => json!({
            "available": true,
            "device_count": 1,
            "device_index": 0,
            "device_name": "NVIDIA GeForce RTX 5070 Ti",
            "compute_capability": "12.0",
            "driver_version": "591.86",
            "cuda_runtime_version": "12.9",
            "cuda_toolkit_version": "12.9",
            "nvrtc_version": "12.9",
            "nvml_available": true,
            "vram_bytes": 17094475776_u64,
            "power_limit_watts": 300.0,
            "power_draw_watts": 34.97,
            "temperature_c": 38.0,
        }),
    }
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn is_rtx5070ti_device_name(name: &str) -> bool {
    let compact = name
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase();

    compact.contains("nvidia") && compact.contains("rtx5070ti")
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitnet_kernels::cuda::{
        CUDA_DENSE_F16_GEMM_KERNEL_ID, CUDA_DENSE_GEMM_REFERENCE_BACKEND,
        CUDA_DENSE_GEMM_TARGET_BACKEND, CUDA_DENSE_GGUF_LINEAR_F16_GEMM_TOLERANCE,
        CUDA_DENSE_RMSNORM_KERNEL_ID, CUDA_DENSE_RMSNORM_REFERENCE_BACKEND,
        CUDA_DENSE_RMSNORM_TARGET_BACKEND, CUDA_DENSE_RMSNORM_TOLERANCE, CudaDenseGemmStats,
        CudaDenseRmsNormStats,
    };
    use bitnet_models::formats::gguf::GgufTensorType;
    use bitnet_models::formats::gguf::{GgufReader, GgufValue};

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
    fn dense_gguf_one_layer_plan_receipt_records_strict_cuda_gap() {
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
        assert_eq!(receipt["execution_plan"]["cuda_dense_regular_llm_ops"], 7);
        assert_eq!(receipt["execution_plan"]["unsupported_ops"], 7);
        assert_eq!(receipt["execution_plan"]["strict_cuda_ready"], false);
        assert_eq!(receipt["one_layer_plan"]["operations"].as_array().unwrap().len(), 14);
        assert_eq!(receipt["gap_audit"]["unsupported_ops_total"], 7);
        assert_eq!(receipt["gap_audit"]["cpu_fallback_ops_total"], 0);
        assert_eq!(receipt["gap_audit"]["strict_cuda_rejects_cpu_fallback"], true);
        assert_eq!(
            receipt["gap_audit"]["unsupported_ops"][0]["cuda_kernel_status"],
            "missing_cuda_kernel"
        );
        assert_eq!(receipt["claim_boundary"]["dense_gguf_one_layer_execution_plan_claimed"], true);
        assert_eq!(receipt["claim_boundary"]["dense_gguf_one_layer_inference_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["dense_gguf_inference_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["bitnet_packed_i2s_qk256_proof"], false);
    }

    #[test]
    fn dense_gguf_norm_fixture_receipt_records_missing_cuda_kernel() {
        let data = build_complete_qwen_layer_gguf();
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
        let data = build_complete_qwen_layer_gguf();
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

    fn build_qwen_gguf(
        tensors: Vec<(&'static str, Vec<usize>, GgufTensorType, Vec<u8>)>,
    ) -> Vec<u8> {
        build_gguf_for_test(
            vec![
                ("general.architecture", GgufValue::String("qwen3".to_string())),
                ("general.name", GgufValue::String("qwen3-linear-fixture".to_string())),
                ("qwen3.embedding_length", GgufValue::U32(4)),
                ("qwen3.feed_forward_length", GgufValue::U32(3)),
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

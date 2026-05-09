//! Dense GGUF single-linear CUDA parity diagnostics.
//!
//! This command is an implementation bridge between descriptor extraction and
//! full dense GGUF inference. It extracts one dense GGUF linear fixture, routes
//! that fixture through the dense FP16 CUDA GEMM bridge, and emits a receipt
//! that still refuses dense GGUF inference, speedup, full-residency, and BitNet
//! packed-kernel proof claims.

use anyhow::{Context, Result, anyhow, bail};
use bitnet_kernels::cuda::{
    DenseGgufLinearCudaParity, DenseGgufLinearGemmFixture, run_dense_gguf_linear_f16_cuda_parity,
};
use bitnet_kernels::dispatch_planner::{ModelDispatchBackend, ModelDispatchSummary};
use bitnet_models::dense_gguf_descriptors::DenseGgufTensorRole;
use bitnet_models::dense_gguf_linear_fixture::{
    DENSE_GGUF_LINEAR_FIXTURE_ARTIFACT_KIND, DenseGgufLinearFixture,
    extract_dense_gguf_linear_fixture,
};
use bitnet_models::formats::gguf::GgufReader;
use bitnet_receipts_core::{
    DENSE_GGUF_LINEAR_CUDA_PARITY_ARTIFACT_KIND,
    validate_dense_gguf_linear_cuda_parity_receipt_json,
};
use clap::Args;
use memmap2::Mmap;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::path::{Path, PathBuf};

use crate::planner_receipts::{ExecutionPlanReceiptInput, execution_plan_receipt};

const HARDWARE_LANE: &str = "nvidia-rtx-5070-ti-cuda";
const MACHINE_ID: &str = "windows-9950x3d-rtx5070ti";

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
        DenseGgufTensorRole::TokenEmbedding
        | DenseGgufTensorRole::AttentionNorm
        | DenseGgufTensorRole::FfnNorm
        | DenseGgufTensorRole::Other => "other",
    }
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
        CudaDenseGemmStats,
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
    fn parse_dense_linear_role_accepts_common_spellings() {
        assert_eq!(
            parse_dense_linear_role("attention_q").unwrap(),
            DenseGgufTensorRole::AttentionQ
        );
        assert_eq!(parse_dense_linear_role("attn-q").unwrap(), DenseGgufTensorRole::AttentionQ);
        assert_eq!(parse_dense_linear_role("mlp_down").unwrap(), DenseGgufTensorRole::MlpDown);
        assert!(parse_dense_linear_role("attention_norm").is_err());
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

//! Dense GGUF norm fixture extraction.
//!
//! This module extracts recognized dense GGUF RMSNorm weight tensors and
//! computes a deterministic CPU reference output for a synthetic hidden state.
//! It does not execute CUDA, load a full dense model, or claim dense GGUF
//! inference support.

use crate::dense_gguf_descriptors::{DenseGgufTensorRole, inspect_dense_gguf_tensor_descriptors};
use crate::formats::gguf::{GgufReader, GgufTensorType};
use bitnet_common::{BitNetError, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Receipt artifact kind for descriptor-driven dense GGUF norm fixture extraction.
pub const DENSE_GGUF_NORM_FIXTURE_ARTIFACT_KIND: &str = "dense_gguf_norm_fixture_extraction";

/// Hash-only receipt summary for an extracted dense GGUF norm fixture.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DenseGgufNormFixtureSummary {
    pub schema: u64,
    pub artifact_kind: String,
    pub architecture: String,
    pub model_family: String,
    pub tensor_name: String,
    pub role: DenseGgufTensorRole,
    pub tensor_type: String,
    pub source_shape: Vec<usize>,
    pub source_offset: u64,
    pub source_size_bytes: u64,
    pub hidden_dim: usize,
    pub value_count: usize,
    pub values_materialized_as_f32: bool,
    pub weight_values_sha256: String,
    pub rmsnorm_eps: f32,
    pub epsilon_source: String,
    pub cpu_reference_input_len: usize,
    pub cpu_reference_output_len: usize,
    pub cpu_reference_input_sha256: String,
    pub cpu_reference_output_sha256: String,
    pub cpu_reference_computed: bool,
    pub cuda_kernel_status: String,
    pub dense_gguf_inference_claimed: bool,
    pub dense_regular_llm_cuda_claimed: bool,
    pub cpu_cuda_parity_claimed: bool,
    pub bitnet_packed_i2s_qk256_proof: bool,
    pub speedup_claim: bool,
    pub full_cuda_residency_claimed: bool,
}

/// Extracted fixture data plus its receipt-ready summary.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseGgufNormFixture {
    pub summary: DenseGgufNormFixtureSummary,
    pub weight_values_f32: Vec<f32>,
    pub cpu_reference_input: Vec<f32>,
    pub cpu_reference_output: Vec<f32>,
}

/// Extract the first tensor for `role` as a dense RMSNorm CPU-reference fixture.
///
/// The extractor fails closed for non-norm roles, BitNet packed markers,
/// unsupported dense tensor types, and non-vector tensors.
pub fn extract_dense_gguf_norm_fixture(
    reader: &GgufReader<'_>,
    role: DenseGgufTensorRole,
) -> Result<DenseGgufNormFixture> {
    if !is_extractable_norm_role(role) {
        return Err(BitNetError::Validation(format!(
            "dense GGUF norm fixture extraction requires an extractable norm role, got {role:?}"
        )));
    }

    let inspection = inspect_dense_gguf_tensor_descriptors(reader)?;
    let descriptor =
        inspection.descriptors.iter().find(|descriptor| descriptor.role == role).ok_or_else(
            || {
                BitNetError::Validation(format!(
                    "dense GGUF norm fixture extraction could not find role {role:?}"
                ))
            },
        )?;
    let info = reader.get_tensor_info_by_name(&descriptor.name).ok_or_else(|| {
        BitNetError::Validation(format!(
            "dense GGUF norm fixture extraction descriptor '{}' is missing tensor info",
            descriptor.name
        ))
    })?;
    if info.shape.len() != 1 {
        return Err(BitNetError::Validation(format!(
            "dense GGUF norm fixture '{}' requires a 1D tensor, got {:?}",
            info.name, info.shape
        )));
    }

    let data = reader.get_tensor_data_by_info(info)?;
    let weight_values_f32 = tensor_values_as_f32(data, info.tensor_type, &info.shape, &info.name)?;
    let hidden_dim = info.shape[0];
    if hidden_dim == 0 || weight_values_f32.len() != hidden_dim {
        return Err(BitNetError::Validation(format!(
            "dense GGUF norm fixture '{}' materialized {} values, expected hidden_dim {}",
            info.name,
            weight_values_f32.len(),
            hidden_dim
        )));
    }

    let (rmsnorm_eps, epsilon_source) = rmsnorm_epsilon(reader, &inspection.architecture);
    let cpu_reference_input = deterministic_reference_input(hidden_dim);
    let cpu_reference_output =
        rmsnorm_reference(&cpu_reference_input, &weight_values_f32, rmsnorm_eps)?;

    let summary = DenseGgufNormFixtureSummary {
        schema: 1,
        artifact_kind: DENSE_GGUF_NORM_FIXTURE_ARTIFACT_KIND.to_string(),
        architecture: inspection.architecture,
        model_family: inspection.model_family,
        tensor_name: info.name.clone(),
        role,
        tensor_type: tensor_type_label(info.tensor_type).to_string(),
        source_shape: info.shape.clone(),
        source_offset: info.offset,
        source_size_bytes: info.size,
        hidden_dim,
        value_count: weight_values_f32.len(),
        values_materialized_as_f32: true,
        weight_values_sha256: f32_values_sha256(&weight_values_f32),
        rmsnorm_eps,
        epsilon_source,
        cpu_reference_input_len: cpu_reference_input.len(),
        cpu_reference_output_len: cpu_reference_output.len(),
        cpu_reference_input_sha256: f32_values_sha256(&cpu_reference_input),
        cpu_reference_output_sha256: f32_values_sha256(&cpu_reference_output),
        cpu_reference_computed: true,
        cuda_kernel_status: "missing_cuda_kernel".to_string(),
        dense_gguf_inference_claimed: false,
        dense_regular_llm_cuda_claimed: false,
        cpu_cuda_parity_claimed: false,
        bitnet_packed_i2s_qk256_proof: false,
        speedup_claim: false,
        full_cuda_residency_claimed: false,
    };

    Ok(DenseGgufNormFixture {
        summary,
        weight_values_f32,
        cpu_reference_input,
        cpu_reference_output,
    })
}

fn is_extractable_norm_role(role: DenseGgufTensorRole) -> bool {
    matches!(role, DenseGgufTensorRole::AttentionNorm | DenseGgufTensorRole::FfnNorm)
}

fn tensor_values_as_f32(
    bytes: &[u8],
    tensor_type: GgufTensorType,
    shape: &[usize],
    tensor_name: &str,
) -> Result<Vec<f32>> {
    match tensor_type {
        GgufTensorType::F32 => f32_tensor_values(bytes, shape, tensor_name),
        GgufTensorType::F16 => f16_tensor_values(bytes, shape, tensor_name),
        other => Err(BitNetError::Validation(format!(
            "dense GGUF norm fixture extraction for '{}' does not support tensor type {} yet",
            tensor_name,
            tensor_type_label(other)
        ))),
    }
}

fn f32_tensor_values(bytes: &[u8], shape: &[usize], tensor_name: &str) -> Result<Vec<f32>> {
    let elements = checked_element_count(shape, tensor_name, "F32")?;
    let expected_bytes = elements.checked_mul(4).ok_or_else(|| {
        BitNetError::Validation(format!(
            "F32 tensor '{tensor_name}' byte count overflows for shape {shape:?}"
        ))
    })?;
    if bytes.len() < expected_bytes {
        return Err(BitNetError::Validation(format!(
            "F32 tensor '{tensor_name}' has {} bytes, expected at least {}",
            bytes.len(),
            expected_bytes
        )));
    }

    Ok(bytes[..expected_bytes]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn f16_tensor_values(bytes: &[u8], shape: &[usize], tensor_name: &str) -> Result<Vec<f32>> {
    let elements = checked_element_count(shape, tensor_name, "F16")?;
    let expected_bytes = elements.checked_mul(2).ok_or_else(|| {
        BitNetError::Validation(format!(
            "F16 tensor '{tensor_name}' byte count overflows for shape {shape:?}"
        ))
    })?;
    if bytes.len() < expected_bytes {
        return Err(BitNetError::Validation(format!(
            "F16 tensor '{tensor_name}' has {} bytes, expected at least {}",
            bytes.len(),
            expected_bytes
        )));
    }

    Ok(bytes[..expected_bytes]
        .chunks_exact(2)
        .map(|chunk| half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
        .collect())
}

fn checked_element_count(shape: &[usize], tensor_name: &str, dtype: &str) -> Result<usize> {
    shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim).ok_or_else(|| {
            BitNetError::Validation(format!(
                "{dtype} tensor '{tensor_name}' shape {shape:?} overflows element count"
            ))
        })
    })
}

fn rmsnorm_epsilon(reader: &GgufReader<'_>, architecture: &str) -> (f32, String) {
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

fn deterministic_reference_input(len: usize) -> Vec<f32> {
    (0..len)
        .map(|idx| {
            let centered = (idx % 19) as f32 - 9.0;
            centered / 18.0
        })
        .collect()
}

fn rmsnorm_reference(input: &[f32], gamma: &[f32], eps: f32) -> Result<Vec<f32>> {
    if input.is_empty() {
        return Err(BitNetError::Validation("RMSNorm reference input must not be empty".into()));
    }
    if input.len() != gamma.len() {
        return Err(BitNetError::Validation(format!(
            "RMSNorm reference input length {} does not match gamma length {}",
            input.len(),
            gamma.len()
        )));
    }
    if eps <= 0.0 || !eps.is_finite() {
        return Err(BitNetError::Validation(format!(
            "RMSNorm reference epsilon must be positive and finite, got {eps}"
        )));
    }

    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let inv_rms = (mean_square + eps).sqrt().recip();
    Ok(input.iter().zip(gamma).map(|(value, weight)| value * inv_rms * weight).collect())
}

fn f32_values_sha256(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn tensor_type_label(tensor_type: GgufTensorType) -> &'static str {
    match tensor_type {
        GgufTensorType::F32 => "f32",
        GgufTensorType::F16 => "f16",
        GgufTensorType::F64 => "f64",
        GgufTensorType::Q4_0 => "q4_0",
        GgufTensorType::Q4_1 => "q4_1",
        GgufTensorType::Q5_0 => "q5_0",
        GgufTensorType::Q5_1 => "q5_1",
        GgufTensorType::Q8_0 => "q8_0",
        GgufTensorType::Q8_1 => "q8_1",
        GgufTensorType::Q2_K => "q2_k",
        GgufTensorType::Q3_K => "q3_k",
        GgufTensorType::Q4_K => "q4_k",
        GgufTensorType::Q5_K => "q5_k",
        GgufTensorType::Q6_K => "q6_k",
        GgufTensorType::Q8_K => "q8_k",
        GgufTensorType::IQ2_S => "iq2_s",
        GgufTensorType::I2_S => "i2_s",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::gguf::GgufValue;

    #[test]
    fn qwen_attention_norm_fixture_materializes_cpu_reference() {
        let data = build_qwen_gguf(
            vec![("qwen3.attention.layer_norm_rms_epsilon", GgufValue::F32(1e-5))],
            vec![(
                "blk.0.attn_norm.weight",
                vec![4],
                GgufTensorType::F32,
                f32_blob(&[1.0, 1.25, 1.5, 1.75]),
            )],
        );
        let reader = GgufReader::new(&data).expect("parse qwen norm fixture");

        let fixture = extract_dense_gguf_norm_fixture(&reader, DenseGgufTensorRole::AttentionNorm)
            .expect("extract norm fixture");

        assert_eq!(fixture.summary.artifact_kind, DENSE_GGUF_NORM_FIXTURE_ARTIFACT_KIND);
        assert_eq!(fixture.summary.model_family, "qwen");
        assert_eq!(fixture.summary.tensor_name, "blk.0.attn_norm.weight");
        assert_eq!(fixture.summary.tensor_type, "f32");
        assert_eq!(fixture.summary.source_shape, vec![4]);
        assert_eq!(fixture.summary.hidden_dim, 4);
        assert_eq!(fixture.summary.value_count, 4);
        assert_eq!(fixture.weight_values_f32, [1.0, 1.25, 1.5, 1.75]);
        assert_eq!(fixture.cpu_reference_input.len(), 4);
        assert_eq!(fixture.cpu_reference_output.len(), 4);
        assert_eq!(fixture.summary.epsilon_source, "qwen3.attention.layer_norm_rms_epsilon");
        assert!((fixture.summary.rmsnorm_eps - 1e-5).abs() < 1e-10);
        assert!(fixture.summary.cpu_reference_computed);
        assert_eq!(fixture.summary.cuda_kernel_status, "missing_cuda_kernel");
        assert!(!fixture.summary.cpu_cuda_parity_claimed);
        assert!(!fixture.summary.dense_gguf_inference_claimed);
        assert!(!fixture.summary.bitnet_packed_i2s_qk256_proof);
    }

    #[test]
    fn qwen_ffn_norm_fixture_supports_f16_weights() {
        let data = build_qwen_gguf(
            Vec::new(),
            vec![(
                "blk.0.ffn_norm.weight",
                vec![4],
                GgufTensorType::F16,
                f16_blob(&[1.0, 0.5, 0.25, 0.125]),
            )],
        );
        let reader = GgufReader::new(&data).expect("parse qwen f16 norm fixture");

        let fixture = extract_dense_gguf_norm_fixture(&reader, DenseGgufTensorRole::FfnNorm)
            .expect("extract ffn norm fixture");

        assert_eq!(fixture.summary.tensor_type, "f16");
        assert_eq!(fixture.summary.hidden_dim, 4);
        assert_eq!(fixture.summary.epsilon_source, "default_1e-6");
        assert_eq!(fixture.cpu_reference_output.len(), 4);
    }

    #[test]
    fn non_norm_roles_are_rejected() {
        let data = build_qwen_gguf(
            Vec::new(),
            vec![("blk.0.attn_q.weight", vec![4, 4], GgufTensorType::F32, f32_blob(&[1.0; 16]))],
        );
        let reader = GgufReader::new(&data).expect("parse qwen linear fixture");

        let err = extract_dense_gguf_norm_fixture(&reader, DenseGgufTensorRole::AttentionQ)
            .unwrap_err()
            .to_string();

        assert!(err.contains("extractable norm role"), "unexpected error: {err}");
    }

    #[test]
    fn quantized_norm_weights_are_rejected() {
        let data = build_qwen_gguf(
            Vec::new(),
            vec![(
                "blk.0.attn_norm.weight",
                vec![4],
                GgufTensorType::Q8_0,
                vec![0; GgufTensorType::Q8_0.element_size()],
            )],
        );
        let reader = GgufReader::new(&data).expect("parse qwen quantized norm fixture");

        let err = extract_dense_gguf_norm_fixture(&reader, DenseGgufTensorRole::AttentionNorm)
            .unwrap_err()
            .to_string();

        assert!(err.contains("does not support tensor type q8_0"), "unexpected error: {err}");
    }

    fn build_qwen_gguf(
        extra_metadata: Vec<(&str, GgufValue)>,
        tensors: Vec<(&'static str, Vec<usize>, GgufTensorType, Vec<u8>)>,
    ) -> Vec<u8> {
        let mut metadata = vec![
            ("general.architecture", GgufValue::String("qwen3".to_string())),
            ("general.name", GgufValue::String("qwen3-norm-fixture".to_string())),
            ("qwen3.embedding_length", GgufValue::U32(4)),
            ("qwen3.feed_forward_length", GgufValue::U32(8)),
        ];
        metadata.extend(extra_metadata);
        build_gguf_for_test(metadata, tensors)
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

    fn f32_blob(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|value| value.to_le_bytes()).collect()
    }

    fn f16_blob(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| half::f16::from_f32(*value).to_bits().to_le_bytes())
            .collect()
    }

    fn write_string(data: &mut Vec<u8>, value: &str) {
        data.extend_from_slice(&(value.len() as u64).to_le_bytes());
        data.extend_from_slice(value.as_bytes());
    }

    fn write_gguf_value(data: &mut Vec<u8>, value: GgufValue) {
        match value {
            GgufValue::U32(v) => {
                data.extend_from_slice(&4u32.to_le_bytes());
                data.extend_from_slice(&v.to_le_bytes());
            }
            GgufValue::F32(v) => {
                data.extend_from_slice(&6u32.to_le_bytes());
                data.extend_from_slice(&v.to_le_bytes());
            }
            GgufValue::String(s) => {
                data.extend_from_slice(&8u32.to_le_bytes());
                write_string(data, &s);
            }
            other => panic!("unsupported test metadata value: {other:?}"),
        }
    }

    fn tensor_type_id(tensor_type: GgufTensorType) -> u32 {
        match tensor_type {
            GgufTensorType::F32 => 0,
            GgufTensorType::F16 => 1,
            GgufTensorType::Q8_0 => 8,
            other => panic!("unsupported test tensor type: {other:?}"),
        }
    }
}

//! Dense GGUF tensor descriptor inspection.
//!
//! This module is intentionally metadata-only. It inspects tensor names,
//! shapes, offsets, and GGUF tensor types so dense model families can be
//! routed conservatively later. It does not load a dense model, execute CUDA,
//! or claim dense GGUF inference support.

use crate::formats::gguf::{GgufReader, GgufTensorType};
use bitnet_common::{BitNetError, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Receipt artifact kind for descriptor-only dense GGUF inspection.
pub const DENSE_GGUF_DESCRIPTOR_INSPECTION_ARTIFACT_KIND: &str =
    "dense_gguf_tensor_descriptor_inspection";

/// Tensor role inferred from common GGUF dense model naming conventions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseGgufTensorRole {
    TokenEmbedding,
    Output,
    AttentionQ,
    AttentionK,
    AttentionV,
    AttentionOutput,
    MlpGate,
    MlpUp,
    MlpDown,
    AttentionNorm,
    FfnNorm,
    Other,
}

/// Descriptor-level support status for a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseGgufDescriptorStatus {
    DenseFloatDescriptorCandidate,
    DenseQuantDescriptorOnly,
    NormOrMetadataDescriptorOnly,
}

/// One inspected dense GGUF tensor descriptor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DenseGgufTensorDescriptor {
    pub name: String,
    pub role: DenseGgufTensorRole,
    pub shape: Vec<usize>,
    pub tensor_type: String,
    pub offset: u64,
    pub size_bytes: u64,
    pub quantized: bool,
    pub descriptor_status: DenseGgufDescriptorStatus,
}

/// Summary returned by descriptor-only inspection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DenseGgufDescriptorInspection {
    pub schema: u64,
    pub artifact_kind: String,
    pub architecture: String,
    pub model_family: String,
    pub tensor_count: u64,
    pub metadata_count: usize,
    pub quantization_families: Vec<String>,
    pub descriptors: Vec<DenseGgufTensorDescriptor>,
    pub required_roles_present: bool,
    pub missing_required_roles: Vec<DenseGgufTensorRole>,
    pub strict_descriptor_complete: bool,
    pub dense_cuda_route_status: String,
    pub bitnet_packed_marker_found: bool,
    pub dense_gguf_inference_claimed: bool,
    pub dense_regular_llm_cuda_claimed: bool,
    pub speedup_claim: bool,
    pub full_cuda_residency_claimed: bool,
}

const REQUIRED_DENSE_ROLES: &[DenseGgufTensorRole] = &[
    DenseGgufTensorRole::TokenEmbedding,
    DenseGgufTensorRole::Output,
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

/// Inspect dense GGUF tensor descriptors from a parsed reader.
///
/// The function fails closed for BitNet packed artifacts or I2_S/IQ2_S tensors
/// because dense GGUF descriptor evidence must not leak into BitNet packed
/// QK256 proof gates.
pub fn inspect_dense_gguf_tensor_descriptors(
    reader: &GgufReader<'_>,
) -> Result<DenseGgufDescriptorInspection> {
    let architecture =
        reader.get_string_metadata("general.architecture").unwrap_or_else(|| "unknown".into());
    if contains_bitnet_marker(&architecture) {
        return Err(BitNetError::Validation(format!(
            "dense GGUF descriptor inspection rejects BitNet packed architecture `{architecture}`"
        )));
    }
    let model_family = dense_model_family(&architecture).ok_or_else(|| {
        BitNetError::Validation(format!(
            "dense GGUF descriptor inspection requires a recognized dense architecture, got `{architecture}`"
        ))
    })?;

    let mut descriptors = Vec::new();
    let mut quantization_families = BTreeSet::new();
    let mut seen_roles = BTreeSet::new();
    let mut bitnet_packed_marker_found = false;
    let mut quantized_descriptor_seen = false;

    for index in 0..reader.tensor_count() as usize {
        let info = reader.get_tensor_info(index)?;
        if matches!(info.tensor_type, GgufTensorType::I2_S | GgufTensorType::IQ2_S)
            || contains_bitnet_marker(&info.name)
        {
            bitnet_packed_marker_found = true;
        }

        let role = classify_dense_tensor_role(&info.name);
        if role != DenseGgufTensorRole::Other {
            seen_roles.insert(role);
        }

        let tensor_type = tensor_type_label(info.tensor_type).to_string();
        quantization_families.insert(tensor_type.clone());
        let quantized = info.tensor_type.is_quantized();
        quantized_descriptor_seen |= quantized;

        descriptors.push(DenseGgufTensorDescriptor {
            name: info.name.clone(),
            role,
            shape: info.shape.clone(),
            tensor_type,
            offset: info.offset,
            size_bytes: info.size,
            quantized,
            descriptor_status: descriptor_status(role, info.tensor_type),
        });
    }

    if bitnet_packed_marker_found {
        return Err(BitNetError::Validation(
            "dense GGUF descriptor inspection rejects BitNet packed tensor markers".into(),
        ));
    }

    let missing_required_roles: Vec<_> =
        REQUIRED_DENSE_ROLES.iter().copied().filter(|role| !seen_roles.contains(role)).collect();
    let required_roles_present = missing_required_roles.is_empty();
    let strict_descriptor_complete = required_roles_present && !descriptors.is_empty();
    let dense_cuda_route_status = if !strict_descriptor_complete {
        "descriptor_incomplete"
    } else if quantized_descriptor_seen {
        "descriptor_only_quant_bridge_required"
    } else {
        "dense_float_descriptor_candidate"
    }
    .to_string();

    Ok(DenseGgufDescriptorInspection {
        schema: 1,
        artifact_kind: DENSE_GGUF_DESCRIPTOR_INSPECTION_ARTIFACT_KIND.to_string(),
        architecture,
        model_family,
        tensor_count: reader.tensor_count(),
        metadata_count: reader.metadata_count(),
        quantization_families: quantization_families.into_iter().collect(),
        descriptors,
        required_roles_present,
        missing_required_roles,
        strict_descriptor_complete,
        dense_cuda_route_status,
        bitnet_packed_marker_found: false,
        dense_gguf_inference_claimed: false,
        dense_regular_llm_cuda_claimed: false,
        speedup_claim: false,
        full_cuda_residency_claimed: false,
    })
}

fn dense_model_family(architecture: &str) -> Option<String> {
    let arch = normalize_label(architecture);
    if arch.starts_with("qwen") {
        Some("qwen".to_string())
    } else if arch.starts_with("llama") {
        Some("llama".to_string())
    } else if arch.starts_with("mistral") {
        Some("mistral".to_string())
    } else if arch.starts_with("mixtral") {
        Some("mixtral".to_string())
    } else if arch.starts_with("phi") {
        Some("phi".to_string())
    } else if arch.starts_with("gemma") {
        Some("gemma".to_string())
    } else if arch.starts_with("deepseek") {
        Some("deepseek".to_string())
    } else if arch.starts_with("falcon") {
        Some("falcon".to_string())
    } else if arch.starts_with("yi") {
        Some("yi".to_string())
    } else if arch.starts_with("internlm") {
        Some("internlm".to_string())
    } else if arch.starts_with("baichuan") {
        Some("baichuan".to_string())
    } else {
        None
    }
}

pub fn classify_dense_tensor_role(name: &str) -> DenseGgufTensorRole {
    let n = name.to_ascii_lowercase();
    if n == "token_embd.weight"
        || n.ends_with(".embed_tokens.weight")
        || n.ends_with(".tok_embeddings.weight")
        || n.ends_with(".wte.weight")
        || n.contains("embed_tokens.weight")
        || n.contains("tok_embeddings.weight")
    {
        DenseGgufTensorRole::TokenEmbedding
    } else if n == "output.weight"
        || n.ends_with(".lm_head.weight")
        || n.ends_with(".output.weight")
    {
        DenseGgufTensorRole::Output
    } else if n.contains("attn_q.weight")
        || n.contains("self_attn.q_proj.weight")
        || n.contains("attention.q_proj.weight")
    {
        DenseGgufTensorRole::AttentionQ
    } else if n.contains("attn_k.weight")
        || n.contains("self_attn.k_proj.weight")
        || n.contains("attention.k_proj.weight")
    {
        DenseGgufTensorRole::AttentionK
    } else if n.contains("attn_v.weight")
        || n.contains("self_attn.v_proj.weight")
        || n.contains("attention.v_proj.weight")
    {
        DenseGgufTensorRole::AttentionV
    } else if n.contains("attn_output.weight")
        || n.contains("self_attn.o_proj.weight")
        || n.contains("attention.o_proj.weight")
    {
        DenseGgufTensorRole::AttentionOutput
    } else if n.contains("ffn_gate.weight")
        || n.contains("mlp.gate_proj.weight")
        || n.contains("feed_forward.gate_proj.weight")
    {
        DenseGgufTensorRole::MlpGate
    } else if n.contains("ffn_up.weight")
        || n.contains("mlp.up_proj.weight")
        || n.contains("feed_forward.up_proj.weight")
    {
        DenseGgufTensorRole::MlpUp
    } else if n.contains("ffn_down.weight")
        || n.contains("mlp.down_proj.weight")
        || n.contains("feed_forward.down_proj.weight")
    {
        DenseGgufTensorRole::MlpDown
    } else if n.contains("attn_norm.weight") || n.contains("input_layernorm.weight") {
        DenseGgufTensorRole::AttentionNorm
    } else if n.contains("ffn_norm.weight")
        || n.contains("post_attention_layernorm.weight")
        || n.contains("post_attn_norm.weight")
    {
        DenseGgufTensorRole::FfnNorm
    } else {
        DenseGgufTensorRole::Other
    }
}

fn descriptor_status(
    role: DenseGgufTensorRole,
    tensor_type: GgufTensorType,
) -> DenseGgufDescriptorStatus {
    if matches!(role, DenseGgufTensorRole::AttentionNorm | DenseGgufTensorRole::FfnNorm) {
        DenseGgufDescriptorStatus::NormOrMetadataDescriptorOnly
    } else if tensor_type.is_quantized() {
        DenseGgufDescriptorStatus::DenseQuantDescriptorOnly
    } else {
        DenseGgufDescriptorStatus::DenseFloatDescriptorCandidate
    }
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

fn contains_bitnet_marker(value: &str) -> bool {
    let normalized = normalize_label(value);
    ["bitnet", "i2s", "iq2s", "qk256", "w158a8"].iter().any(|marker| normalized.contains(marker))
}

fn normalize_label(value: &str) -> String {
    value.chars().filter(|ch| ch.is_ascii_alphanumeric()).collect::<String>().to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::gguf::GgufValue;

    #[test]
    fn qwen_q8_descriptors_are_complete_but_quant_bridge_required() {
        let data =
            build_qwen_gguf(GgufTensorType::Q8_0, required_dense_tensors(GgufTensorType::Q8_0));
        let reader = GgufReader::new(&data).expect("parse qwen gguf fixture");

        let inspection =
            inspect_dense_gguf_tensor_descriptors(&reader).expect("descriptor inspection");

        assert_eq!(inspection.architecture, "qwen3");
        assert_eq!(inspection.model_family, "qwen");
        assert!(inspection.required_roles_present);
        assert!(inspection.strict_descriptor_complete);
        assert_eq!(inspection.dense_cuda_route_status, "descriptor_only_quant_bridge_required");
        assert!(inspection.quantization_families.contains(&"q8_0".to_string()));
        assert!(!inspection.dense_gguf_inference_claimed);
        assert!(!inspection.dense_regular_llm_cuda_claimed);
        assert!(!inspection.speedup_claim);
        assert!(!inspection.full_cuda_residency_claimed);
    }

    #[test]
    fn qwen_f16_descriptors_are_dense_float_candidates() {
        let data =
            build_qwen_gguf(GgufTensorType::F16, required_dense_tensors(GgufTensorType::F16));
        let reader = GgufReader::new(&data).expect("parse qwen gguf fixture");

        let inspection =
            inspect_dense_gguf_tensor_descriptors(&reader).expect("descriptor inspection");

        assert!(inspection.required_roles_present);
        assert_eq!(inspection.dense_cuda_route_status, "dense_float_descriptor_candidate");
        assert!(inspection.quantization_families.contains(&"f16".to_string()));
    }

    #[test]
    fn missing_required_role_is_descriptor_incomplete() {
        let mut tensors = required_dense_tensors(GgufTensorType::F16);
        tensors.retain(|(name, _, _, _)| *name != "blk.0.ffn_down.weight");
        let data = build_qwen_gguf(GgufTensorType::F16, tensors);
        let reader = GgufReader::new(&data).expect("parse qwen gguf fixture");

        let inspection =
            inspect_dense_gguf_tensor_descriptors(&reader).expect("descriptor inspection");

        assert!(!inspection.required_roles_present);
        assert!(!inspection.strict_descriptor_complete);
        assert_eq!(inspection.dense_cuda_route_status, "descriptor_incomplete");
        assert!(inspection.missing_required_roles.contains(&DenseGgufTensorRole::MlpDown));
    }

    #[test]
    fn bitnet_architecture_is_rejected() {
        let data = build_gguf_for_descriptor_test(
            vec![
                ("general.architecture", GgufValue::String("bitnet".to_string())),
                ("general.name", GgufValue::String("bitnet-fixture".to_string())),
            ],
            vec![(
                "token_embd.weight",
                vec![16, 16],
                GgufTensorType::F16,
                blob_for(GgufTensorType::F16, 256),
            )],
        );
        let reader = GgufReader::new(&data).expect("parse bitnet gguf fixture");

        let err = inspect_dense_gguf_tensor_descriptors(&reader).unwrap_err().to_string();

        assert!(err.contains("BitNet packed architecture"), "unexpected error: {err}");
    }

    #[test]
    fn bitnet_packed_tensor_marker_is_rejected() {
        let data = build_qwen_gguf(
            GgufTensorType::I2_S,
            vec![(
                "blk.0.attn_q.weight",
                vec![16, 16],
                GgufTensorType::I2_S,
                blob_for(GgufTensorType::I2_S, 256),
            )],
        );
        let reader = GgufReader::new(&data).expect("parse qwen gguf fixture");

        let err = inspect_dense_gguf_tensor_descriptors(&reader).unwrap_err().to_string();

        assert!(err.contains("BitNet packed tensor markers"), "unexpected error: {err}");
    }

    fn build_qwen_gguf(
        _default_type: GgufTensorType,
        tensors: Vec<(&'static str, Vec<usize>, GgufTensorType, Vec<u8>)>,
    ) -> Vec<u8> {
        build_gguf_for_descriptor_test(
            vec![
                ("general.architecture", GgufValue::String("qwen3".to_string())),
                ("general.name", GgufValue::String("qwen3-descriptor-fixture".to_string())),
                ("qwen3.embedding_length", GgufValue::U32(16)),
                ("qwen3.feed_forward_length", GgufValue::U32(32)),
            ],
            tensors,
        )
    }

    fn required_dense_tensors(
        tensor_type: GgufTensorType,
    ) -> Vec<(&'static str, Vec<usize>, GgufTensorType, Vec<u8>)> {
        let matrix_shape = vec![16, 16];
        let norm_shape = vec![16];
        vec![
            ("token_embd.weight", matrix_shape.clone(), tensor_type, blob_for(tensor_type, 256)),
            ("output.weight", matrix_shape.clone(), tensor_type, blob_for(tensor_type, 256)),
            ("blk.0.attn_q.weight", matrix_shape.clone(), tensor_type, blob_for(tensor_type, 256)),
            ("blk.0.attn_k.weight", matrix_shape.clone(), tensor_type, blob_for(tensor_type, 256)),
            ("blk.0.attn_v.weight", matrix_shape.clone(), tensor_type, blob_for(tensor_type, 256)),
            (
                "blk.0.attn_output.weight",
                matrix_shape.clone(),
                tensor_type,
                blob_for(tensor_type, 256),
            ),
            (
                "blk.0.ffn_gate.weight",
                matrix_shape.clone(),
                tensor_type,
                blob_for(tensor_type, 256),
            ),
            ("blk.0.ffn_up.weight", matrix_shape.clone(), tensor_type, blob_for(tensor_type, 256)),
            ("blk.0.ffn_down.weight", matrix_shape, tensor_type, blob_for(tensor_type, 256)),
            (
                "blk.0.attn_norm.weight",
                norm_shape.clone(),
                GgufTensorType::F32,
                blob_for(GgufTensorType::F32, 16),
            ),
            (
                "blk.0.ffn_norm.weight",
                norm_shape,
                GgufTensorType::F32,
                blob_for(GgufTensorType::F32, 16),
            ),
        ]
    }

    fn blob_for(tensor_type: GgufTensorType, elements: usize) -> Vec<u8> {
        if tensor_type.is_quantized() {
            let blocks = elements.div_ceil(tensor_type.block_size());
            vec![0u8; blocks * tensor_type.element_size()]
        } else {
            vec![0u8; elements * tensor_type.element_size()]
        }
    }

    fn build_gguf_for_descriptor_test(
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
            GgufValue::Bool(value) => {
                data.extend_from_slice(&7u32.to_le_bytes());
                data.push(u8::from(value));
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

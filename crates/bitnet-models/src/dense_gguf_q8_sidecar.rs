//! Inert Q8_0 GGUF sidecar descriptors.
//!
//! This module carries packed Q8_0 tensor metadata from strict GGUF loading
//! toward future dense-linear runtime work. It deliberately does not replace
//! the eager F32 Candle tensors used for model execution.

use crate::dense_gguf_descriptors::{DenseGgufTensorRole, classify_dense_tensor_role};
use crate::formats::gguf::{GgufTensorType, TensorInfo};
use crate::names::is_projection_weight;
use bitnet_common::{BitNetError, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const DENSE_GGUF_Q8_SIDECAR_REGISTRY_ARTIFACT_KIND: &str = "dense_gguf_q8_sidecar_registry";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DenseGgufQ8SidecarDescriptor {
    pub tensor_name: String,
    pub role: DenseGgufTensorRole,
    pub tensor_type: String,
    pub source_shape: Vec<usize>,
    pub runtime_candle_shape: Vec<usize>,
    pub source_offset: u64,
    pub source_size_bytes: u64,
    pub value_count: usize,
    pub q8_block_size: usize,
    pub q8_block_count: usize,
    pub q8_payload_bytes: usize,
    pub packed_q8_bytes_sha256: String,
    pub shape_reshaped_without_transpose: bool,
    pub eager_f32_runtime_preserved: bool,
    pub runtime_compute_enabled: bool,
    pub dense_runtime_replaced: bool,
    pub speedup_claim: bool,
    pub generated_id_preservation_required_before_runtime_use: bool,
    pub next_runtime_api_hook: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DenseGgufQ8SidecarRegistry {
    pub schema: u64,
    pub artifact_kind: String,
    pub descriptors: Vec<DenseGgufQ8SidecarDescriptor>,
    pub eager_f32_runtime_preserved: bool,
    pub runtime_compute_enabled: bool,
    pub dense_runtime_replaced: bool,
    pub speedup_claim: bool,
    pub generated_id_preservation_required_before_runtime_use: bool,
    pub next_runtime_api_hook: String,
}

impl Default for DenseGgufQ8SidecarRegistry {
    fn default() -> Self {
        Self {
            schema: 1,
            artifact_kind: DENSE_GGUF_Q8_SIDECAR_REGISTRY_ARTIFACT_KIND.to_string(),
            descriptors: Vec::new(),
            eager_f32_runtime_preserved: true,
            runtime_compute_enabled: false,
            dense_runtime_replaced: false,
            speedup_claim: false,
            generated_id_preservation_required_before_runtime_use: true,
            next_runtime_api_hook: "add a behavior-preserving dense linear dispatch API that can select a packed Q8_0 sidecar only after generated-ID and strict-receipt equivalence gates pass".to_string(),
        }
    }
}

impl DenseGgufQ8SidecarRegistry {
    pub fn is_empty(&self) -> bool {
        self.descriptors.is_empty()
    }

    pub fn descriptor_count(&self) -> usize {
        self.descriptors.len()
    }

    pub fn descriptor_for_tensor(
        &self,
        tensor_name: &str,
    ) -> Option<&DenseGgufQ8SidecarDescriptor> {
        self.descriptors.iter().find(|descriptor| descriptor.tensor_name == tensor_name)
    }

    pub fn try_push_tensor(&mut self, info: &TensorInfo, data: &[u8]) -> Result<()> {
        let Some(descriptor) = DenseGgufQ8SidecarDescriptor::from_tensor(info, data)? else {
            return Ok(());
        };
        self.descriptors.push(descriptor);
        Ok(())
    }
}

impl DenseGgufQ8SidecarDescriptor {
    pub fn from_tensor(info: &TensorInfo, data: &[u8]) -> Result<Option<Self>> {
        if info.tensor_type != GgufTensorType::Q8_0 {
            return Ok(None);
        }

        let role = classify_dense_tensor_role(&info.name);
        if !is_dense_linear_sidecar_role(role) {
            return Ok(None);
        }

        let value_count = checked_element_count(&info.shape, &info.name)?;
        let q8_block_size = GgufTensorType::Q8_0.block_size();
        let q8_block_count = value_count.div_ceil(q8_block_size);
        let q8_payload_bytes =
            q8_block_count.checked_mul(GgufTensorType::Q8_0.element_size()).ok_or_else(|| {
                BitNetError::Validation(format!(
                    "Q8_0 sidecar descriptor '{}' byte count overflows for {q8_block_count} blocks",
                    info.name
                ))
            })?;
        if data.len() < q8_payload_bytes {
            return Err(BitNetError::Validation(format!(
                "Q8_0 sidecar descriptor '{}' has {} bytes, expected at least {}",
                info.name,
                data.len(),
                q8_payload_bytes
            )));
        }

        let runtime_candle_shape = dense_q8_runtime_shape(info, role);
        let shape_reshaped_without_transpose = runtime_candle_shape != info.shape;

        Ok(Some(Self {
            tensor_name: info.name.clone(),
            role,
            tensor_type: "q8_0".to_string(),
            source_shape: info.shape.clone(),
            runtime_candle_shape,
            source_offset: info.offset,
            source_size_bytes: info.size,
            value_count,
            q8_block_size,
            q8_block_count,
            q8_payload_bytes,
            packed_q8_bytes_sha256: bytes_sha256(&data[..q8_payload_bytes]),
            shape_reshaped_without_transpose,
            eager_f32_runtime_preserved: true,
            runtime_compute_enabled: false,
            dense_runtime_replaced: false,
            speedup_claim: false,
            generated_id_preservation_required_before_runtime_use: true,
            next_runtime_api_hook: "dense_linear_dispatch_q8_sidecar_candidate".to_string(),
        }))
    }
}

fn is_dense_linear_sidecar_role(role: DenseGgufTensorRole) -> bool {
    matches!(
        role,
        DenseGgufTensorRole::TokenEmbedding
            | DenseGgufTensorRole::Output
            | DenseGgufTensorRole::AttentionQ
            | DenseGgufTensorRole::AttentionK
            | DenseGgufTensorRole::AttentionV
            | DenseGgufTensorRole::AttentionOutput
            | DenseGgufTensorRole::MlpGate
            | DenseGgufTensorRole::MlpUp
            | DenseGgufTensorRole::MlpDown
    )
}

fn dense_q8_runtime_shape(info: &TensorInfo, role: DenseGgufTensorRole) -> Vec<usize> {
    if info.shape.len() == 2
        && ((matches!(role, DenseGgufTensorRole::TokenEmbedding | DenseGgufTensorRole::Output)
            && embedding_is_transposed(&info.shape))
            || is_projection_weight(&info.name))
    {
        vec![info.shape[1], info.shape[0]]
    } else {
        info.shape.clone()
    }
}

fn embedding_is_transposed(shape: &[usize]) -> bool {
    shape.len() == 2 && shape[0] < shape[1] && shape[1] >= 32768
}

fn checked_element_count(shape: &[usize], tensor_name: &str) -> Result<usize> {
    shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim).ok_or_else(|| {
            BitNetError::Validation(format!(
                "Q8_0 sidecar descriptor '{tensor_name}' element count overflows for shape {shape:?}"
            ))
        })
    })
}

fn bytes_sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q8_info(name: &str, shape: Vec<usize>, size: u64) -> TensorInfo {
        TensorInfo {
            name: name.to_string(),
            shape,
            tensor_type: GgufTensorType::Q8_0,
            offset: 128,
            size,
        }
    }

    #[test]
    fn dense_gguf_q8_sidecar_descriptor_is_metadata_only_and_inert() {
        let info = q8_info("blk.0.attn_q.weight", vec![2, 64], 136);
        let mut data = vec![0u8; 136];
        data.extend_from_slice(&[1, 2, 3, 4]);

        let descriptor = match DenseGgufQ8SidecarDescriptor::from_tensor(&info, &data) {
            Ok(Some(descriptor)) => descriptor,
            result => {
                assert!(false, "expected sidecar descriptor, got {result:?}");
                return;
            }
        };

        assert_eq!(descriptor.role, DenseGgufTensorRole::AttentionQ);
        assert_eq!(descriptor.value_count, 128);
        assert_eq!(descriptor.q8_block_size, 32);
        assert_eq!(descriptor.q8_block_count, 4);
        assert_eq!(descriptor.q8_payload_bytes, 136);
        assert_eq!(descriptor.packed_q8_bytes_sha256, bytes_sha256(&data[..136]));
        assert_ne!(descriptor.packed_q8_bytes_sha256, bytes_sha256(&data));
        assert_eq!(descriptor.runtime_candle_shape, vec![64, 2]);
        assert!(descriptor.shape_reshaped_without_transpose);
        assert!(descriptor.eager_f32_runtime_preserved);
        assert!(!descriptor.runtime_compute_enabled);
        assert!(!descriptor.dense_runtime_replaced);
        assert!(!descriptor.speedup_claim);
        assert!(descriptor.generated_id_preservation_required_before_runtime_use);
    }

    #[test]
    fn dense_gguf_q8_sidecar_registry_ignores_non_linear_q8_tensors() {
        let mut registry = DenseGgufQ8SidecarRegistry::default();
        let info = q8_info("blk.0.attn_norm.weight", vec![64], 68);
        let data = vec![0u8; 68];

        assert!(registry.try_push_tensor(&info, &data).is_ok());

        assert!(registry.is_empty());
        assert!(registry.eager_f32_runtime_preserved);
        assert!(!registry.runtime_compute_enabled);
        assert!(!registry.dense_runtime_replaced);
        assert!(!registry.speedup_claim);
    }

    #[test]
    fn dense_gguf_q8_sidecar_embedding_shape_matches_loader_heuristic() {
        let transposed = q8_info("token_embd.weight", vec![64, 32768], 2_228_224);
        let token_major = q8_info("token_embd.weight", vec![32768, 64], 2_228_224);
        let data = vec![0u8; 2_228_224];

        let transposed_descriptor =
            match DenseGgufQ8SidecarDescriptor::from_tensor(&transposed, &data) {
                Ok(Some(descriptor)) => descriptor,
                result => {
                    assert!(false, "expected transposed descriptor, got {result:?}");
                    return;
                }
            };
        let token_major_descriptor =
            match DenseGgufQ8SidecarDescriptor::from_tensor(&token_major, &data) {
                Ok(Some(descriptor)) => descriptor,
                result => {
                    assert!(false, "expected token-major descriptor, got {result:?}");
                    return;
                }
            };

        assert_eq!(transposed_descriptor.runtime_candle_shape, vec![32768, 64]);
        assert!(transposed_descriptor.shape_reshaped_without_transpose);
        assert_eq!(token_major_descriptor.runtime_candle_shape, vec![32768, 64]);
        assert!(!token_major_descriptor.shape_reshaped_without_transpose);
    }

    #[test]
    fn dense_gguf_q8_sidecar_rejects_truncated_payload() {
        let info = q8_info("blk.0.ffn_down.weight", vec![4, 32], 135);
        let data = vec![0u8; 135];

        let err = match DenseGgufQ8SidecarDescriptor::from_tensor(&info, &data) {
            Err(err) => err,
            result => {
                assert!(false, "expected truncated payload error, got {result:?}");
                return;
            }
        };

        assert!(err.to_string().contains("expected at least 136"));
    }
}

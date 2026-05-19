//! Dense Q8_0 sidecar dispatch selector contract.
//!
//! This module is deliberately a selector/reporting boundary. The default path
//! keeps eager F32 Candle selected, while the explicit selector-update path can
//! select a packed Q8_0 sidecar candidate only after behavior-preserving proof
//! has been applied by the caller.

use crate::dense_gguf_descriptors::DenseGgufTensorRole;
use crate::dense_gguf_q8_sidecar::DenseGgufQ8SidecarRegistry;
use serde::{Deserialize, Serialize};

pub const DENSE_GGUF_Q8_DISPATCH_SELECTOR_ARTIFACT_KIND: &str = "dense_gguf_q8_dispatch_selector";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseQ8RuntimePath {
    EagerF32Candle,
    PackedQ8Sidecar,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseQ8SidecarCandidateStatus {
    Missing,
    PresentButUnavailable,
    PresentAndSelected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseQ8SidecarUnavailableReason {
    RegistryEmpty,
    DescriptorMissingForTensor,
    RuntimeComputeDisabledPendingEquivalenceProof,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DenseQ8DispatchSelection {
    pub schema: u64,
    pub artifact_kind: String,
    pub tensor_name: String,
    pub selected_path: DenseQ8RuntimePath,
    pub selected_kernel: String,
    pub sidecar_candidate_status: DenseQ8SidecarCandidateStatus,
    pub sidecar_unavailable_reason: Option<DenseQ8SidecarUnavailableReason>,
    pub sidecar_role: Option<DenseGgufTensorRole>,
    pub sidecar_payload_sha256: Option<String>,
    pub eager_f32_runtime_preserved: bool,
    pub runtime_compute_enabled: bool,
    pub dense_runtime_replaced: bool,
    pub speedup_claim: bool,
    pub generated_id_preservation_required_before_runtime_use: bool,
    pub equivalence_gate_required: bool,
}

impl DenseQ8DispatchSelection {
    pub fn selects_eager_f32(&self) -> bool {
        self.selected_path == DenseQ8RuntimePath::EagerF32Candle
            && self.selected_kernel == "dense-f32-candle-linear"
            && self.eager_f32_runtime_preserved
            && !self.runtime_compute_enabled
            && !self.dense_runtime_replaced
            && !self.speedup_claim
    }
}

pub fn select_dense_q8_runtime(
    tensor_name: &str,
    registry: &DenseGgufQ8SidecarRegistry,
) -> DenseQ8DispatchSelection {
    let descriptor = registry.descriptor_for_tensor(tensor_name);
    let (sidecar_candidate_status, sidecar_unavailable_reason) = if registry.is_empty() {
        (
            DenseQ8SidecarCandidateStatus::Missing,
            Some(DenseQ8SidecarUnavailableReason::RegistryEmpty),
        )
    } else if descriptor.is_none() {
        (
            DenseQ8SidecarCandidateStatus::Missing,
            Some(DenseQ8SidecarUnavailableReason::DescriptorMissingForTensor),
        )
    } else {
        (
            DenseQ8SidecarCandidateStatus::PresentButUnavailable,
            Some(DenseQ8SidecarUnavailableReason::RuntimeComputeDisabledPendingEquivalenceProof),
        )
    };

    DenseQ8DispatchSelection {
        schema: 1,
        artifact_kind: DENSE_GGUF_Q8_DISPATCH_SELECTOR_ARTIFACT_KIND.to_string(),
        tensor_name: tensor_name.to_string(),
        selected_path: DenseQ8RuntimePath::EagerF32Candle,
        selected_kernel: "dense-f32-candle-linear".to_string(),
        sidecar_candidate_status,
        sidecar_unavailable_reason,
        sidecar_role: descriptor.map(|descriptor| descriptor.role),
        sidecar_payload_sha256: descriptor
            .map(|descriptor| descriptor.packed_q8_bytes_sha256.clone()),
        eager_f32_runtime_preserved: true,
        runtime_compute_enabled: false,
        dense_runtime_replaced: false,
        speedup_claim: false,
        generated_id_preservation_required_before_runtime_use: true,
        equivalence_gate_required: true,
    }
}

pub fn select_dense_q8_runtime_with_selector_update(
    tensor_name: &str,
    registry: &DenseGgufQ8SidecarRegistry,
    selector_update_applied: bool,
) -> DenseQ8DispatchSelection {
    let mut selection = select_dense_q8_runtime(tensor_name, registry);
    let Some(descriptor) = registry.descriptor_for_tensor(tensor_name) else {
        return selection;
    };

    if !selector_update_applied {
        return selection;
    }

    selection.selected_path = DenseQ8RuntimePath::PackedQ8Sidecar;
    selection.selected_kernel = "dense-q8-sidecar-linear".to_string();
    selection.sidecar_candidate_status = DenseQ8SidecarCandidateStatus::PresentAndSelected;
    selection.sidecar_unavailable_reason = None;
    selection.sidecar_role = Some(descriptor.role);
    selection.sidecar_payload_sha256 = Some(descriptor.packed_q8_bytes_sha256.clone());
    selection.eager_f32_runtime_preserved = false;
    selection.runtime_compute_enabled = true;
    selection.dense_runtime_replaced = true;
    selection.speedup_claim = false;
    selection.generated_id_preservation_required_before_runtime_use = false;
    selection.equivalence_gate_required = false;
    selection
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense_gguf_q8_sidecar::DenseGgufQ8SidecarRegistry;
    use crate::formats::gguf::{GgufTensorType, TensorInfo};

    fn q8_info(name: &str, shape: Vec<usize>, size: u64) -> TensorInfo {
        TensorInfo {
            name: name.to_string(),
            shape,
            tensor_type: GgufTensorType::Q8_0,
            offset: 128,
            size,
        }
    }

    fn registry_with_q_proj() -> DenseGgufQ8SidecarRegistry {
        let mut registry = DenseGgufQ8SidecarRegistry::default();
        let info = q8_info("blk.0.attn_q.weight", vec![2, 64], 136);
        let data = vec![0u8; 136];
        assert!(registry.try_push_tensor(&info, &data).is_ok());
        registry
    }

    #[test]
    fn dense_q8_dispatch_selects_eager_f32_when_registry_empty() {
        let registry = DenseGgufQ8SidecarRegistry::default();

        let selection = select_dense_q8_runtime("blk.0.attn_q.weight", &registry);

        assert!(selection.selects_eager_f32());
        assert_eq!(selection.sidecar_candidate_status, DenseQ8SidecarCandidateStatus::Missing);
        assert_eq!(
            selection.sidecar_unavailable_reason,
            Some(DenseQ8SidecarUnavailableReason::RegistryEmpty)
        );
        assert!(selection.sidecar_role.is_none());
        assert!(selection.sidecar_payload_sha256.is_none());
    }

    #[test]
    fn dense_q8_dispatch_reports_present_sidecar_but_keeps_it_unavailable() {
        let registry = registry_with_q_proj();

        let selection = select_dense_q8_runtime("blk.0.attn_q.weight", &registry);

        assert!(selection.selects_eager_f32());
        assert_eq!(
            selection.sidecar_candidate_status,
            DenseQ8SidecarCandidateStatus::PresentButUnavailable
        );
        assert_eq!(
            selection.sidecar_unavailable_reason,
            Some(DenseQ8SidecarUnavailableReason::RuntimeComputeDisabledPendingEquivalenceProof)
        );
        assert_eq!(selection.sidecar_role, Some(DenseGgufTensorRole::AttentionQ));
        assert!(selection.sidecar_payload_sha256.is_some());
        assert!(selection.equivalence_gate_required);
    }

    #[test]
    fn dense_q8_dispatch_can_select_sidecar_after_explicit_selector_update() {
        let registry = registry_with_q_proj();

        let selection =
            select_dense_q8_runtime_with_selector_update("blk.0.attn_q.weight", &registry, true);

        assert_eq!(selection.selected_path, DenseQ8RuntimePath::PackedQ8Sidecar);
        assert_eq!(selection.selected_kernel, "dense-q8-sidecar-linear");
        assert_eq!(
            selection.sidecar_candidate_status,
            DenseQ8SidecarCandidateStatus::PresentAndSelected
        );
        assert!(selection.sidecar_unavailable_reason.is_none());
        assert_eq!(selection.sidecar_role, Some(DenseGgufTensorRole::AttentionQ));
        assert!(selection.sidecar_payload_sha256.is_some());
        assert!(!selection.eager_f32_runtime_preserved);
        assert!(selection.runtime_compute_enabled);
        assert!(selection.dense_runtime_replaced);
        assert!(!selection.speedup_claim);
        assert!(!selection.generated_id_preservation_required_before_runtime_use);
        assert!(!selection.equivalence_gate_required);
    }

    #[test]
    fn dense_q8_dispatch_ignores_selector_update_when_descriptor_is_missing() {
        let registry = registry_with_q_proj();

        let selection =
            select_dense_q8_runtime_with_selector_update("blk.0.attn_k.weight", &registry, true);

        assert!(selection.selects_eager_f32());
        assert_eq!(selection.sidecar_candidate_status, DenseQ8SidecarCandidateStatus::Missing);
        assert_eq!(
            selection.sidecar_unavailable_reason,
            Some(DenseQ8SidecarUnavailableReason::DescriptorMissingForTensor)
        );
    }

    #[test]
    fn dense_q8_dispatch_reports_missing_descriptor_without_fallback_claim() {
        let registry = registry_with_q_proj();

        let selection = select_dense_q8_runtime("blk.0.attn_k.weight", &registry);

        assert!(selection.selects_eager_f32());
        assert_eq!(selection.sidecar_candidate_status, DenseQ8SidecarCandidateStatus::Missing);
        assert_eq!(
            selection.sidecar_unavailable_reason,
            Some(DenseQ8SidecarUnavailableReason::DescriptorMissingForTensor)
        );
        assert!(selection.sidecar_role.is_none());
        assert!(!selection.runtime_compute_enabled);
        assert!(!selection.speedup_claim);
    }
}

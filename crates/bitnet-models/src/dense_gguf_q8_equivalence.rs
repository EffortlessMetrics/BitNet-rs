//! Dense Q8_0 sidecar equivalence gate.
//!
//! This module connects the fixture-level packed Q8_0 sidecar prototype to the
//! dense-linear dispatch selector. It deliberately does not enable packed Q8_0
//! sidecar runtime compute; it records whether fixture math matches the eager
//! F32 reference and keeps the production selector blocked until full
//! generated-ID/text receipt equivalence exists.

use crate::dense_gguf_descriptors::DenseGgufTensorRole;
use crate::dense_gguf_linear_fixture::DenseGgufQ8LinearSidecarSummary;
use crate::dense_gguf_q8_dispatch::{
    DenseQ8DispatchSelection, DenseQ8RuntimePath, DenseQ8SidecarCandidateStatus,
};
use bitnet_common::{BitNetError, Result};
use serde::{Deserialize, Serialize};

pub const DENSE_GGUF_Q8_SIDECAR_EQUIVALENCE_GATE_ARTIFACT_KIND: &str =
    "dense_gguf_q8_sidecar_equivalence_gate";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseQ8RuntimeBlocker {
    MissingSidecarCandidate,
    FixtureOutputMismatch,
    GeneratedIdReceiptEquivalenceMissing,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DenseQ8SidecarEquivalenceGate {
    pub schema: u64,
    pub artifact_kind: String,
    pub tensor_name: String,
    pub role: DenseGgufTensorRole,
    pub selected_path: DenseQ8RuntimePath,
    pub selected_kernel: String,
    pub sidecar_candidate_status: DenseQ8SidecarCandidateStatus,
    pub sidecar_payload_sha256: Option<String>,
    pub fused_output_sha256: String,
    pub eager_output_sha256: String,
    pub max_abs_diff_vs_eager_f32: f32,
    pub fixture_abs_tolerance: f32,
    pub fixture_equivalence_passed: bool,
    pub generated_id_receipt_equivalence_passed: bool,
    pub sidecar_runtime_compute_allowed: bool,
    pub runtime_blockers: Vec<DenseQ8RuntimeBlocker>,
    pub eager_f32_runtime_preserved: bool,
    pub dense_runtime_replaced: bool,
    pub speedup_claim: bool,
}

impl DenseQ8SidecarEquivalenceGate {
    pub fn runtime_still_blocked(&self) -> bool {
        !self.sidecar_runtime_compute_allowed
            && self.eager_f32_runtime_preserved
            && !self.dense_runtime_replaced
            && !self.speedup_claim
    }
}

pub fn build_dense_q8_sidecar_equivalence_gate(
    sidecar: &DenseGgufQ8LinearSidecarSummary,
    selection: &DenseQ8DispatchSelection,
    fixture_abs_tolerance: f32,
) -> Result<DenseQ8SidecarEquivalenceGate> {
    if sidecar.tensor_name != selection.tensor_name {
        return Err(BitNetError::Validation(format!(
            "Q8_0 sidecar equivalence gate tensor mismatch: sidecar '{}' != selector '{}'",
            sidecar.tensor_name, selection.tensor_name
        )));
    }
    if !fixture_abs_tolerance.is_finite() || fixture_abs_tolerance < 0.0 {
        return Err(BitNetError::Validation(format!(
            "Q8_0 sidecar equivalence gate tolerance must be finite and non-negative, got {fixture_abs_tolerance}"
        )));
    }

    let fixture_equivalence_passed = sidecar.compares_against_eager_f32_reference
        && sidecar.dequantizes_inside_matvec
        && !sidecar.materializes_full_f32_weights
        && !sidecar.speedup_claim
        && !sidecar.dense_runtime_replaced
        && sidecar.max_abs_diff_vs_eager_f32 <= fixture_abs_tolerance;

    let mut runtime_blockers = Vec::new();
    if selection.sidecar_candidate_status != DenseQ8SidecarCandidateStatus::PresentButUnavailable {
        runtime_blockers.push(DenseQ8RuntimeBlocker::MissingSidecarCandidate);
    }
    if !fixture_equivalence_passed {
        runtime_blockers.push(DenseQ8RuntimeBlocker::FixtureOutputMismatch);
    }
    runtime_blockers.push(DenseQ8RuntimeBlocker::GeneratedIdReceiptEquivalenceMissing);

    Ok(DenseQ8SidecarEquivalenceGate {
        schema: 1,
        artifact_kind: DENSE_GGUF_Q8_SIDECAR_EQUIVALENCE_GATE_ARTIFACT_KIND.to_string(),
        tensor_name: sidecar.tensor_name.clone(),
        role: sidecar.role,
        selected_path: selection.selected_path,
        selected_kernel: selection.selected_kernel.clone(),
        sidecar_candidate_status: selection.sidecar_candidate_status,
        sidecar_payload_sha256: selection.sidecar_payload_sha256.clone(),
        fused_output_sha256: sidecar.fused_output_sha256.clone(),
        eager_output_sha256: sidecar.eager_output_sha256.clone(),
        max_abs_diff_vs_eager_f32: sidecar.max_abs_diff_vs_eager_f32,
        fixture_abs_tolerance,
        fixture_equivalence_passed,
        generated_id_receipt_equivalence_passed: false,
        sidecar_runtime_compute_allowed: false,
        runtime_blockers,
        eager_f32_runtime_preserved: true,
        dense_runtime_replaced: false,
        speedup_claim: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense_gguf_linear_fixture::DENSE_GGUF_Q8_LINEAR_SIDECAR_ARTIFACT_KIND;
    use crate::dense_gguf_q8_dispatch::select_dense_q8_runtime;
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

    fn sidecar_summary(max_abs_diff_vs_eager_f32: f32) -> DenseGgufQ8LinearSidecarSummary {
        DenseGgufQ8LinearSidecarSummary {
            schema: 1,
            artifact_kind: DENSE_GGUF_Q8_LINEAR_SIDECAR_ARTIFACT_KIND.to_string(),
            tensor_name: "blk.0.attn_q.weight".to_string(),
            role: DenseGgufTensorRole::AttentionQ,
            tensor_type: "q8_0".to_string(),
            source_shape: vec![2, 64],
            matrix_rows: 64,
            matrix_cols: 2,
            value_count: 128,
            q8_block_size: 32,
            q8_block_count: 4,
            packed_q8_bytes_sha256: "packed".to_string(),
            cpu_reference_input_sha256: "input".to_string(),
            fused_output_sha256: "fused".to_string(),
            eager_output_sha256: "eager".to_string(),
            max_abs_diff_vs_eager_f32,
            dequantizes_inside_matvec: true,
            materializes_full_f32_weights: false,
            compares_against_eager_f32_reference: true,
            generated_id_preservation_required_before_runtime_use: true,
            speedup_claim: false,
            dense_runtime_replaced: false,
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
    fn q8_sidecar_equivalence_gate_keeps_runtime_blocked_after_fixture_match() -> Result<()> {
        let registry = registry_with_q_proj();
        let selection = select_dense_q8_runtime("blk.0.attn_q.weight", &registry);
        let gate =
            build_dense_q8_sidecar_equivalence_gate(&sidecar_summary(0.0), &selection, 1e-6)?;

        assert!(gate.fixture_equivalence_passed);
        assert!(!gate.generated_id_receipt_equivalence_passed);
        assert!(gate.runtime_still_blocked());
        assert_eq!(
            gate.runtime_blockers,
            vec![DenseQ8RuntimeBlocker::GeneratedIdReceiptEquivalenceMissing]
        );
        assert_eq!(gate.selected_kernel, "dense-f32-candle-linear");
        Ok(())
    }

    #[test]
    fn q8_sidecar_equivalence_gate_records_fixture_mismatch_blocker() -> Result<()> {
        let registry = registry_with_q_proj();
        let selection = select_dense_q8_runtime("blk.0.attn_q.weight", &registry);
        let gate =
            build_dense_q8_sidecar_equivalence_gate(&sidecar_summary(0.25), &selection, 1e-6)?;

        assert!(!gate.fixture_equivalence_passed);
        assert!(gate.runtime_still_blocked());
        assert!(gate.runtime_blockers.contains(&DenseQ8RuntimeBlocker::FixtureOutputMismatch));
        assert!(
            gate.runtime_blockers
                .contains(&DenseQ8RuntimeBlocker::GeneratedIdReceiptEquivalenceMissing)
        );
        Ok(())
    }

    #[test]
    fn q8_sidecar_equivalence_gate_requires_matching_selector_tensor() {
        let registry = registry_with_q_proj();
        let selection = select_dense_q8_runtime("blk.0.attn_k.weight", &registry);
        let err = build_dense_q8_sidecar_equivalence_gate(&sidecar_summary(0.0), &selection, 1e-6)
            .expect_err("tensor mismatch should fail closed");

        assert!(err.to_string().contains("tensor mismatch"), "unexpected error: {err}");
    }
}

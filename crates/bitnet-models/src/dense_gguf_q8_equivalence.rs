//! Dense Q8_0 sidecar equivalence gate.
//!
//! This module connects the fixture-level packed Q8_0 sidecar prototype to the
//! dense-linear dispatch selector. It records whether fixture math and
//! generated-ID/text behavior match the eager F32 reference, then emits an
//! explicit selector-update artifact that can unlock packed sidecar selection
//! only after the proof chain is ready.

use crate::dense_gguf_descriptors::DenseGgufTensorRole;
use crate::dense_gguf_linear_fixture::DenseGgufQ8LinearSidecarSummary;
use crate::dense_gguf_q8_dispatch::{
    DenseQ8DispatchSelection, DenseQ8RuntimePath, DenseQ8SidecarCandidateStatus,
    select_dense_q8_runtime_with_selector_update,
};
use crate::dense_gguf_q8_sidecar::DenseGgufQ8SidecarRegistry;
use bitnet_common::{BitNetError, Result};
use serde::{Deserialize, Serialize};

pub const DENSE_GGUF_Q8_SIDECAR_EQUIVALENCE_GATE_ARTIFACT_KIND: &str =
    "dense_gguf_q8_sidecar_equivalence_gate";
pub const DENSE_GGUF_Q8_GENERATED_ID_TEXT_EQUIVALENCE_ARTIFACT_KIND: &str =
    "dense_gguf_q8_generated_id_text_equivalence";
pub const DENSE_GGUF_Q8_PRODUCTION_COMPUTE_HOOK_ARTIFACT_KIND: &str =
    "dense_gguf_q8_production_compute_hook";
pub const DENSE_GGUF_Q8_SELECTOR_READINESS_GATE_ARTIFACT_KIND: &str =
    "dense_gguf_q8_selector_readiness_gate";
pub const DENSE_GGUF_Q8_SELECTOR_UPDATE_ARTIFACT_KIND: &str = "dense_gguf_q8_selector_update";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseQ8RuntimeBlocker {
    MissingSidecarCandidate,
    FixtureOutputMismatch,
    GeneratedIdReceiptEquivalenceMissing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseQ8RuntimePreflightBlocker {
    FixtureEquivalenceMissing,
    GeneratedIdReceiptEquivalenceMissing,
    ProductionComputeHookMissing,
    ProductionSelectorStillEagerF32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseQ8GeneratedIdTextMismatch {
    ModelSha256,
    TokenizerSource,
    TokenizerStrict,
    CorpusId,
    PromptId,
    PromptIds,
    GeneratedIds,
    DecodedText,
    SelectedBackend,
    SelectedKernel,
    BaselineFallbackUsed,
    CandidateFallbackUsed,
    BaselineSpeedupClaim,
    CandidateSpeedupClaim,
    RuntimePreflightNotEagerF32,
    RuntimePreflightAllowsSidecarCompute,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseQ8ProductionComputeHookStatus {
    Missing,
    AvailableButSelectorStillEagerF32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseQ8SelectorReadinessStatus {
    Blocked,
    ReadyForSeparateSelectorUpdate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseQ8SelectorUpdateStatus {
    Blocked,
    AppliedToPackedSidecarCandidate,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DenseQ8SidecarRuntimePreflight {
    pub schema: u64,
    pub artifact_kind: String,
    pub tensor_name: String,
    pub role: DenseGgufTensorRole,
    pub selected_path: DenseQ8RuntimePath,
    pub selected_kernel: String,
    pub fixture_equivalence_passed: bool,
    pub generated_id_receipt_equivalence_passed: bool,
    pub production_compute_hook_available: bool,
    pub sidecar_runtime_compute_allowed: bool,
    pub runtime_blockers: Vec<DenseQ8RuntimePreflightBlocker>,
    pub eager_f32_runtime_preserved: bool,
    pub dense_runtime_replaced: bool,
    pub speedup_claim: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DenseQ8BehaviorReceiptSummary {
    pub receipt_id: String,
    pub model_sha256: String,
    pub tokenizer_source: String,
    pub tokenizer_strict: bool,
    pub corpus_id: Option<String>,
    pub prompt_id: Option<String>,
    pub prompt_ids: Vec<i64>,
    pub generated_ids: Vec<i64>,
    pub decoded_text: String,
    pub selected_backend: String,
    pub selected_kernel: String,
    pub fallback_used: bool,
    pub speedup_claim: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DenseQ8GeneratedIdTextEquivalenceGate {
    pub schema: u64,
    pub artifact_kind: String,
    pub selected_path: DenseQ8RuntimePath,
    pub selected_kernel: String,
    pub baseline_receipt: DenseQ8BehaviorReceiptSummary,
    pub candidate_receipt: DenseQ8BehaviorReceiptSummary,
    pub generated_id_receipt_equivalence_passed: bool,
    pub sidecar_runtime_compute_allowed: bool,
    pub mismatches: Vec<DenseQ8GeneratedIdTextMismatch>,
    pub eager_f32_runtime_preserved: bool,
    pub dense_runtime_replaced: bool,
    pub speedup_claim: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DenseQ8ProductionComputeHookAvailability {
    pub schema: u64,
    pub artifact_kind: String,
    pub selected_path: DenseQ8RuntimePath,
    pub selected_kernel: String,
    pub hook_status: DenseQ8ProductionComputeHookStatus,
    pub hook_name: Option<String>,
    pub generated_id_receipt_equivalence_passed: bool,
    pub production_compute_hook_available: bool,
    pub selector_update_required_before_runtime_use: bool,
    pub sidecar_runtime_compute_allowed: bool,
    pub runtime_blockers: Vec<DenseQ8RuntimePreflightBlocker>,
    pub eager_f32_runtime_preserved: bool,
    pub dense_runtime_replaced: bool,
    pub speedup_claim: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DenseQ8SelectorReadinessGate {
    pub schema: u64,
    pub artifact_kind: String,
    pub selected_path: DenseQ8RuntimePath,
    pub selected_kernel: String,
    pub readiness_status: DenseQ8SelectorReadinessStatus,
    pub generated_id_receipt_equivalence_passed: bool,
    pub production_compute_hook_available: bool,
    pub selector_update_ready: bool,
    pub selector_update_required_before_runtime_use: bool,
    pub sidecar_runtime_compute_allowed: bool,
    pub runtime_blockers: Vec<DenseQ8RuntimePreflightBlocker>,
    pub eager_f32_runtime_preserved: bool,
    pub dense_runtime_replaced: bool,
    pub speedup_claim: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DenseQ8SelectorUpdate {
    pub schema: u64,
    pub artifact_kind: String,
    pub previous_selected_path: DenseQ8RuntimePath,
    pub selected_path: DenseQ8RuntimePath,
    pub selected_kernel: String,
    pub update_status: DenseQ8SelectorUpdateStatus,
    pub generated_id_receipt_equivalence_passed: bool,
    pub production_compute_hook_available: bool,
    pub selector_update_ready: bool,
    pub selector_update_applied: bool,
    pub sidecar_runtime_compute_allowed: bool,
    pub runtime_blockers: Vec<DenseQ8RuntimePreflightBlocker>,
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

impl DenseQ8SidecarRuntimePreflight {
    pub fn selects_eager_f32(&self) -> bool {
        self.selected_path == DenseQ8RuntimePath::EagerF32Candle
            && self.selected_kernel == "dense-f32-candle-linear"
            && self.eager_f32_runtime_preserved
            && !self.dense_runtime_replaced
            && !self.speedup_claim
    }
}

impl DenseQ8GeneratedIdTextEquivalenceGate {
    pub fn runtime_still_blocked(&self) -> bool {
        !self.sidecar_runtime_compute_allowed
            && self.eager_f32_runtime_preserved
            && !self.dense_runtime_replaced
            && !self.speedup_claim
    }
}

impl DenseQ8ProductionComputeHookAvailability {
    pub fn runtime_still_blocked(&self) -> bool {
        !self.sidecar_runtime_compute_allowed
            && self.eager_f32_runtime_preserved
            && !self.dense_runtime_replaced
            && !self.speedup_claim
            && self.selector_update_required_before_runtime_use
    }
}

impl DenseQ8SelectorReadinessGate {
    pub fn runtime_still_blocked(&self) -> bool {
        !self.sidecar_runtime_compute_allowed
            && self.eager_f32_runtime_preserved
            && !self.dense_runtime_replaced
            && !self.speedup_claim
            && self.selector_update_required_before_runtime_use
    }
}

impl DenseQ8SelectorUpdate {
    pub fn selects_packed_sidecar_without_speed_claim(&self) -> bool {
        self.update_status == DenseQ8SelectorUpdateStatus::AppliedToPackedSidecarCandidate
            && self.previous_selected_path == DenseQ8RuntimePath::EagerF32Candle
            && self.selected_path == DenseQ8RuntimePath::PackedQ8Sidecar
            && self.selected_kernel == "dense-q8-sidecar-linear"
            && self.generated_id_receipt_equivalence_passed
            && self.production_compute_hook_available
            && self.selector_update_ready
            && self.selector_update_applied
            && self.sidecar_runtime_compute_allowed
            && !self.eager_f32_runtime_preserved
            && self.dense_runtime_replaced
            && !self.speedup_claim
            && self.runtime_blockers.is_empty()
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

pub fn build_dense_q8_generated_id_text_equivalence_gate(
    preflight: &DenseQ8SidecarRuntimePreflight,
    baseline_receipt: DenseQ8BehaviorReceiptSummary,
    candidate_receipt: DenseQ8BehaviorReceiptSummary,
) -> DenseQ8GeneratedIdTextEquivalenceGate {
    let mut mismatches = Vec::new();

    if baseline_receipt.model_sha256 != candidate_receipt.model_sha256 {
        mismatches.push(DenseQ8GeneratedIdTextMismatch::ModelSha256);
    }
    if baseline_receipt.tokenizer_source != candidate_receipt.tokenizer_source {
        mismatches.push(DenseQ8GeneratedIdTextMismatch::TokenizerSource);
    }
    if baseline_receipt.tokenizer_strict != candidate_receipt.tokenizer_strict {
        mismatches.push(DenseQ8GeneratedIdTextMismatch::TokenizerStrict);
    }
    if baseline_receipt.corpus_id != candidate_receipt.corpus_id {
        mismatches.push(DenseQ8GeneratedIdTextMismatch::CorpusId);
    }
    if baseline_receipt.prompt_id != candidate_receipt.prompt_id {
        mismatches.push(DenseQ8GeneratedIdTextMismatch::PromptId);
    }
    if baseline_receipt.prompt_ids != candidate_receipt.prompt_ids {
        mismatches.push(DenseQ8GeneratedIdTextMismatch::PromptIds);
    }
    if baseline_receipt.generated_ids != candidate_receipt.generated_ids {
        mismatches.push(DenseQ8GeneratedIdTextMismatch::GeneratedIds);
    }
    if baseline_receipt.decoded_text != candidate_receipt.decoded_text {
        mismatches.push(DenseQ8GeneratedIdTextMismatch::DecodedText);
    }
    if baseline_receipt.selected_backend != candidate_receipt.selected_backend {
        mismatches.push(DenseQ8GeneratedIdTextMismatch::SelectedBackend);
    }
    if baseline_receipt.selected_kernel != candidate_receipt.selected_kernel {
        mismatches.push(DenseQ8GeneratedIdTextMismatch::SelectedKernel);
    }
    if baseline_receipt.fallback_used {
        mismatches.push(DenseQ8GeneratedIdTextMismatch::BaselineFallbackUsed);
    }
    if candidate_receipt.fallback_used {
        mismatches.push(DenseQ8GeneratedIdTextMismatch::CandidateFallbackUsed);
    }
    if baseline_receipt.speedup_claim {
        mismatches.push(DenseQ8GeneratedIdTextMismatch::BaselineSpeedupClaim);
    }
    if candidate_receipt.speedup_claim {
        mismatches.push(DenseQ8GeneratedIdTextMismatch::CandidateSpeedupClaim);
    }
    if !preflight.selects_eager_f32() {
        mismatches.push(DenseQ8GeneratedIdTextMismatch::RuntimePreflightNotEagerF32);
    }
    if preflight.sidecar_runtime_compute_allowed {
        mismatches.push(DenseQ8GeneratedIdTextMismatch::RuntimePreflightAllowsSidecarCompute);
    }

    DenseQ8GeneratedIdTextEquivalenceGate {
        schema: 1,
        artifact_kind: DENSE_GGUF_Q8_GENERATED_ID_TEXT_EQUIVALENCE_ARTIFACT_KIND.to_string(),
        selected_path: preflight.selected_path,
        selected_kernel: preflight.selected_kernel.clone(),
        baseline_receipt,
        candidate_receipt,
        generated_id_receipt_equivalence_passed: mismatches.is_empty(),
        sidecar_runtime_compute_allowed: false,
        mismatches,
        eager_f32_runtime_preserved: true,
        dense_runtime_replaced: false,
        speedup_claim: false,
    }
}

pub fn build_dense_q8_production_compute_hook_availability(
    gate: &DenseQ8GeneratedIdTextEquivalenceGate,
    hook_name: Option<&str>,
) -> DenseQ8ProductionComputeHookAvailability {
    let production_compute_hook_available = hook_name.is_some();
    let hook_status = if production_compute_hook_available {
        DenseQ8ProductionComputeHookStatus::AvailableButSelectorStillEagerF32
    } else {
        DenseQ8ProductionComputeHookStatus::Missing
    };

    let mut runtime_blockers = Vec::new();
    if !gate.generated_id_receipt_equivalence_passed {
        runtime_blockers.push(DenseQ8RuntimePreflightBlocker::GeneratedIdReceiptEquivalenceMissing);
    }
    if !production_compute_hook_available {
        runtime_blockers.push(DenseQ8RuntimePreflightBlocker::ProductionComputeHookMissing);
    }
    runtime_blockers.push(DenseQ8RuntimePreflightBlocker::ProductionSelectorStillEagerF32);

    DenseQ8ProductionComputeHookAvailability {
        schema: 1,
        artifact_kind: DENSE_GGUF_Q8_PRODUCTION_COMPUTE_HOOK_ARTIFACT_KIND.to_string(),
        selected_path: gate.selected_path,
        selected_kernel: gate.selected_kernel.clone(),
        hook_status,
        hook_name: hook_name.map(ToOwned::to_owned),
        generated_id_receipt_equivalence_passed: gate.generated_id_receipt_equivalence_passed,
        production_compute_hook_available,
        selector_update_required_before_runtime_use: true,
        sidecar_runtime_compute_allowed: false,
        runtime_blockers,
        eager_f32_runtime_preserved: true,
        dense_runtime_replaced: false,
        speedup_claim: false,
    }
}

pub fn build_dense_q8_selector_readiness_gate(
    availability: &DenseQ8ProductionComputeHookAvailability,
) -> DenseQ8SelectorReadinessGate {
    let evidence_ready = availability.generated_id_receipt_equivalence_passed
        && availability.production_compute_hook_available
        && availability.eager_f32_runtime_preserved
        && !availability.dense_runtime_replaced
        && !availability.speedup_claim
        && !availability.sidecar_runtime_compute_allowed;
    let selector_update_ready = evidence_ready
        && availability.runtime_blockers
            == vec![DenseQ8RuntimePreflightBlocker::ProductionSelectorStillEagerF32];
    let readiness_status = if selector_update_ready {
        DenseQ8SelectorReadinessStatus::ReadyForSeparateSelectorUpdate
    } else {
        DenseQ8SelectorReadinessStatus::Blocked
    };

    DenseQ8SelectorReadinessGate {
        schema: 1,
        artifact_kind: DENSE_GGUF_Q8_SELECTOR_READINESS_GATE_ARTIFACT_KIND.to_string(),
        selected_path: availability.selected_path,
        selected_kernel: availability.selected_kernel.clone(),
        readiness_status,
        generated_id_receipt_equivalence_passed: availability
            .generated_id_receipt_equivalence_passed,
        production_compute_hook_available: availability.production_compute_hook_available,
        selector_update_ready,
        selector_update_required_before_runtime_use: true,
        sidecar_runtime_compute_allowed: false,
        runtime_blockers: availability.runtime_blockers.clone(),
        eager_f32_runtime_preserved: true,
        dense_runtime_replaced: false,
        speedup_claim: false,
    }
}

pub fn build_dense_q8_selector_update(
    readiness: &DenseQ8SelectorReadinessGate,
) -> DenseQ8SelectorUpdate {
    let selector_update_applied = readiness.selector_update_ready
        && readiness.generated_id_receipt_equivalence_passed
        && readiness.production_compute_hook_available
        && readiness.readiness_status
            == DenseQ8SelectorReadinessStatus::ReadyForSeparateSelectorUpdate
        && readiness.runtime_blockers
            == vec![DenseQ8RuntimePreflightBlocker::ProductionSelectorStillEagerF32]
        && readiness.eager_f32_runtime_preserved
        && !readiness.dense_runtime_replaced
        && !readiness.sidecar_runtime_compute_allowed
        && !readiness.speedup_claim;

    if selector_update_applied {
        DenseQ8SelectorUpdate {
            schema: 1,
            artifact_kind: DENSE_GGUF_Q8_SELECTOR_UPDATE_ARTIFACT_KIND.to_string(),
            previous_selected_path: DenseQ8RuntimePath::EagerF32Candle,
            selected_path: DenseQ8RuntimePath::PackedQ8Sidecar,
            selected_kernel: "dense-q8-sidecar-linear".to_string(),
            update_status: DenseQ8SelectorUpdateStatus::AppliedToPackedSidecarCandidate,
            generated_id_receipt_equivalence_passed: true,
            production_compute_hook_available: true,
            selector_update_ready: true,
            selector_update_applied: true,
            sidecar_runtime_compute_allowed: true,
            runtime_blockers: Vec::new(),
            eager_f32_runtime_preserved: false,
            dense_runtime_replaced: true,
            speedup_claim: false,
        }
    } else {
        DenseQ8SelectorUpdate {
            schema: 1,
            artifact_kind: DENSE_GGUF_Q8_SELECTOR_UPDATE_ARTIFACT_KIND.to_string(),
            previous_selected_path: readiness.selected_path,
            selected_path: readiness.selected_path,
            selected_kernel: readiness.selected_kernel.clone(),
            update_status: DenseQ8SelectorUpdateStatus::Blocked,
            generated_id_receipt_equivalence_passed: readiness
                .generated_id_receipt_equivalence_passed,
            production_compute_hook_available: readiness.production_compute_hook_available,
            selector_update_ready: readiness.selector_update_ready,
            selector_update_applied: false,
            sidecar_runtime_compute_allowed: false,
            runtime_blockers: readiness.runtime_blockers.clone(),
            eager_f32_runtime_preserved: true,
            dense_runtime_replaced: false,
            speedup_claim: false,
        }
    }
}

pub fn select_dense_q8_runtime_after_selector_update(
    tensor_name: &str,
    registry: &DenseGgufQ8SidecarRegistry,
    update: &DenseQ8SelectorUpdate,
) -> DenseQ8DispatchSelection {
    select_dense_q8_runtime_with_selector_update(
        tensor_name,
        registry,
        update.selects_packed_sidecar_without_speed_claim(),
    )
}

pub fn build_dense_q8_sidecar_runtime_preflight(
    gate: &DenseQ8SidecarEquivalenceGate,
    production_compute_hook_available: bool,
) -> DenseQ8SidecarRuntimePreflight {
    let mut runtime_blockers = Vec::new();
    if !gate.fixture_equivalence_passed {
        runtime_blockers.push(DenseQ8RuntimePreflightBlocker::FixtureEquivalenceMissing);
    }
    if !gate.generated_id_receipt_equivalence_passed {
        runtime_blockers.push(DenseQ8RuntimePreflightBlocker::GeneratedIdReceiptEquivalenceMissing);
    }
    if !production_compute_hook_available {
        runtime_blockers.push(DenseQ8RuntimePreflightBlocker::ProductionComputeHookMissing);
    }

    runtime_blockers.push(DenseQ8RuntimePreflightBlocker::ProductionSelectorStillEagerF32);
    let sidecar_runtime_compute_allowed = false;

    DenseQ8SidecarRuntimePreflight {
        schema: 1,
        artifact_kind: "dense_gguf_q8_sidecar_runtime_preflight".to_string(),
        tensor_name: gate.tensor_name.clone(),
        role: gate.role,
        selected_path: gate.selected_path,
        selected_kernel: gate.selected_kernel.clone(),
        fixture_equivalence_passed: gate.fixture_equivalence_passed,
        generated_id_receipt_equivalence_passed: gate.generated_id_receipt_equivalence_passed,
        production_compute_hook_available,
        sidecar_runtime_compute_allowed,
        runtime_blockers,
        eager_f32_runtime_preserved: true,
        dense_runtime_replaced: false,
        speedup_claim: false,
    }
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

    fn behavior_receipt(receipt_id: &str) -> DenseQ8BehaviorReceiptSummary {
        DenseQ8BehaviorReceiptSummary {
            receipt_id: receipt_id.to_string(),
            model_sha256: "model-sha".to_string(),
            tokenizer_source: "gguf_metadata".to_string(),
            tokenizer_strict: true,
            corpus_id: Some("qwen3-kaby-corpus".to_string()),
            prompt_id: Some("math_2_plus_2".to_string()),
            prompt_ids: vec![151644, 3838, 374, 220, 17, 10, 17],
            generated_ids: vec![19],
            decoded_text: "4".to_string(),
            selected_backend: "cpu-rust".to_string(),
            selected_kernel: "dense-f32-candle-linear".to_string(),
            fallback_used: false,
            speedup_claim: false,
        }
    }

    fn runtime_preflight() -> Result<DenseQ8SidecarRuntimePreflight> {
        let registry = registry_with_q_proj();
        let selection = select_dense_q8_runtime("blk.0.attn_q.weight", &registry);
        let gate =
            build_dense_q8_sidecar_equivalence_gate(&sidecar_summary(0.0), &selection, 1e-6)?;
        Ok(build_dense_q8_sidecar_runtime_preflight(&gate, false))
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

    #[test]
    fn q8_sidecar_runtime_preflight_names_generated_id_and_compute_hook_blockers() -> Result<()> {
        let registry = registry_with_q_proj();
        let selection = select_dense_q8_runtime("blk.0.attn_q.weight", &registry);
        let gate =
            build_dense_q8_sidecar_equivalence_gate(&sidecar_summary(0.0), &selection, 1e-6)?;

        let preflight = build_dense_q8_sidecar_runtime_preflight(&gate, false);

        assert!(preflight.fixture_equivalence_passed);
        assert!(!preflight.generated_id_receipt_equivalence_passed);
        assert!(!preflight.production_compute_hook_available);
        assert!(!preflight.sidecar_runtime_compute_allowed);
        assert!(preflight.selects_eager_f32());
        assert_eq!(
            preflight.runtime_blockers,
            vec![
                DenseQ8RuntimePreflightBlocker::GeneratedIdReceiptEquivalenceMissing,
                DenseQ8RuntimePreflightBlocker::ProductionComputeHookMissing,
                DenseQ8RuntimePreflightBlocker::ProductionSelectorStillEagerF32
            ]
        );
        Ok(())
    }

    #[test]
    fn q8_sidecar_runtime_preflight_blocks_fixture_mismatch() -> Result<()> {
        let registry = registry_with_q_proj();
        let selection = select_dense_q8_runtime("blk.0.attn_q.weight", &registry);
        let gate =
            build_dense_q8_sidecar_equivalence_gate(&sidecar_summary(0.25), &selection, 1e-6)?;

        let preflight = build_dense_q8_sidecar_runtime_preflight(&gate, false);

        assert!(!preflight.fixture_equivalence_passed);
        assert!(!preflight.sidecar_runtime_compute_allowed);
        assert!(
            preflight
                .runtime_blockers
                .contains(&DenseQ8RuntimePreflightBlocker::FixtureEquivalenceMissing)
        );
        Ok(())
    }

    #[test]
    fn q8_generated_id_text_equivalence_passes_for_matching_receipts_but_keeps_runtime_blocked()
    -> Result<()> {
        let preflight = runtime_preflight()?;
        let baseline = behavior_receipt("eager-f32-baseline");
        let candidate = behavior_receipt("q8-sidecar-candidate");

        let gate =
            build_dense_q8_generated_id_text_equivalence_gate(&preflight, baseline, candidate);

        assert_eq!(gate.artifact_kind, DENSE_GGUF_Q8_GENERATED_ID_TEXT_EQUIVALENCE_ARTIFACT_KIND);
        assert!(gate.generated_id_receipt_equivalence_passed);
        assert!(gate.mismatches.is_empty());
        assert!(gate.runtime_still_blocked());
        assert_eq!(gate.selected_path, DenseQ8RuntimePath::EagerF32Candle);
        assert_eq!(gate.selected_kernel, "dense-f32-candle-linear");
        Ok(())
    }

    #[test]
    fn q8_generated_id_text_equivalence_records_behavior_and_claim_mismatches() -> Result<()> {
        let preflight = runtime_preflight()?;
        let baseline = behavior_receipt("eager-f32-baseline");
        let mut candidate = behavior_receipt("q8-sidecar-candidate");
        candidate.generated_ids = vec![84644];
        candidate.decoded_text = "htar".to_string();
        candidate.fallback_used = true;
        candidate.speedup_claim = true;

        let gate =
            build_dense_q8_generated_id_text_equivalence_gate(&preflight, baseline, candidate);

        assert!(!gate.generated_id_receipt_equivalence_passed);
        assert!(gate.runtime_still_blocked());
        assert!(gate.mismatches.contains(&DenseQ8GeneratedIdTextMismatch::GeneratedIds));
        assert!(gate.mismatches.contains(&DenseQ8GeneratedIdTextMismatch::DecodedText));
        assert!(gate.mismatches.contains(&DenseQ8GeneratedIdTextMismatch::CandidateFallbackUsed));
        assert!(gate.mismatches.contains(&DenseQ8GeneratedIdTextMismatch::CandidateSpeedupClaim));
        Ok(())
    }

    #[test]
    fn q8_generated_id_text_equivalence_requires_strict_same_provenance() -> Result<()> {
        let preflight = runtime_preflight()?;
        let baseline = behavior_receipt("eager-f32-baseline");
        let mut candidate = behavior_receipt("q8-sidecar-candidate");
        candidate.model_sha256 = "other-model-sha".to_string();
        candidate.tokenizer_source = "sibling_file".to_string();
        candidate.tokenizer_strict = false;
        candidate.selected_kernel = "dense-q8-sidecar-linear".to_string();

        let gate =
            build_dense_q8_generated_id_text_equivalence_gate(&preflight, baseline, candidate);

        assert!(!gate.generated_id_receipt_equivalence_passed);
        assert!(gate.runtime_still_blocked());
        assert!(gate.mismatches.contains(&DenseQ8GeneratedIdTextMismatch::ModelSha256));
        assert!(gate.mismatches.contains(&DenseQ8GeneratedIdTextMismatch::TokenizerSource));
        assert!(gate.mismatches.contains(&DenseQ8GeneratedIdTextMismatch::TokenizerStrict));
        assert!(gate.mismatches.contains(&DenseQ8GeneratedIdTextMismatch::SelectedKernel));
        Ok(())
    }

    #[test]
    fn q8_production_compute_hook_availability_names_missing_hook_and_generated_gate_blockers()
    -> Result<()> {
        let preflight = runtime_preflight()?;
        let baseline = behavior_receipt("eager-f32-baseline");
        let mut candidate = behavior_receipt("q8-sidecar-candidate");
        candidate.generated_ids = vec![84644];

        let gate =
            build_dense_q8_generated_id_text_equivalence_gate(&preflight, baseline, candidate);
        let availability = build_dense_q8_production_compute_hook_availability(&gate, None);

        assert_eq!(availability.artifact_kind, DENSE_GGUF_Q8_PRODUCTION_COMPUTE_HOOK_ARTIFACT_KIND);
        assert_eq!(availability.hook_status, DenseQ8ProductionComputeHookStatus::Missing);
        assert!(availability.hook_name.is_none());
        assert!(!availability.generated_id_receipt_equivalence_passed);
        assert!(!availability.production_compute_hook_available);
        assert!(availability.runtime_still_blocked());
        assert_eq!(availability.selected_path, DenseQ8RuntimePath::EagerF32Candle);
        assert_eq!(availability.selected_kernel, "dense-f32-candle-linear");
        assert_eq!(
            availability.runtime_blockers,
            vec![
                DenseQ8RuntimePreflightBlocker::GeneratedIdReceiptEquivalenceMissing,
                DenseQ8RuntimePreflightBlocker::ProductionComputeHookMissing,
                DenseQ8RuntimePreflightBlocker::ProductionSelectorStillEagerF32
            ]
        );
        assert!(availability.eager_f32_runtime_preserved);
        assert!(!availability.dense_runtime_replaced);
        assert!(!availability.speedup_claim);
        Ok(())
    }

    #[test]
    fn q8_production_compute_hook_availability_keeps_selector_blocked_after_behavior_equivalence()
    -> Result<()> {
        let preflight = runtime_preflight()?;
        let baseline = behavior_receipt("eager-f32-baseline");
        let candidate = behavior_receipt("q8-sidecar-candidate");

        let gate =
            build_dense_q8_generated_id_text_equivalence_gate(&preflight, baseline, candidate);
        let availability = build_dense_q8_production_compute_hook_availability(
            &gate,
            Some("dense-q8-sidecar-linear-hook"),
        );

        assert!(availability.generated_id_receipt_equivalence_passed);
        assert!(availability.production_compute_hook_available);
        assert_eq!(
            availability.hook_status,
            DenseQ8ProductionComputeHookStatus::AvailableButSelectorStillEagerF32
        );
        assert_eq!(availability.hook_name.as_deref(), Some("dense-q8-sidecar-linear-hook"));
        assert!(availability.runtime_still_blocked());
        assert!(!availability.sidecar_runtime_compute_allowed);
        assert_eq!(
            availability.runtime_blockers,
            vec![DenseQ8RuntimePreflightBlocker::ProductionSelectorStillEagerF32]
        );
        assert_eq!(availability.selected_path, DenseQ8RuntimePath::EagerF32Candle);
        assert_eq!(availability.selected_kernel, "dense-f32-candle-linear");
        assert!(availability.selector_update_required_before_runtime_use);
        assert!(availability.eager_f32_runtime_preserved);
        assert!(!availability.dense_runtime_replaced);
        assert!(!availability.speedup_claim);
        Ok(())
    }

    #[test]
    fn q8_production_compute_hook_availability_keeps_generated_equivalence_blocker_when_hook_exists()
    -> Result<()> {
        let preflight = runtime_preflight()?;
        let baseline = behavior_receipt("eager-f32-baseline");
        let mut candidate = behavior_receipt("q8-sidecar-candidate");
        candidate.decoded_text = "wrong".to_string();

        let gate =
            build_dense_q8_generated_id_text_equivalence_gate(&preflight, baseline, candidate);
        let availability = build_dense_q8_production_compute_hook_availability(
            &gate,
            Some("dense-q8-sidecar-linear-hook"),
        );

        assert!(!availability.generated_id_receipt_equivalence_passed);
        assert!(availability.production_compute_hook_available);
        assert!(availability.runtime_still_blocked());
        assert_eq!(
            availability.runtime_blockers,
            vec![
                DenseQ8RuntimePreflightBlocker::GeneratedIdReceiptEquivalenceMissing,
                DenseQ8RuntimePreflightBlocker::ProductionSelectorStillEagerF32
            ]
        );
        assert!(!availability.sidecar_runtime_compute_allowed);
        assert!(!availability.dense_runtime_replaced);
        assert!(!availability.speedup_claim);
        Ok(())
    }

    #[test]
    fn q8_selector_readiness_gate_blocks_when_hook_or_behavior_evidence_is_missing() -> Result<()> {
        let preflight = runtime_preflight()?;
        let baseline = behavior_receipt("eager-f32-baseline");
        let mut candidate = behavior_receipt("q8-sidecar-candidate");
        candidate.generated_ids = vec![84644];

        let gate =
            build_dense_q8_generated_id_text_equivalence_gate(&preflight, baseline, candidate);
        let availability = build_dense_q8_production_compute_hook_availability(&gate, None);
        let readiness = build_dense_q8_selector_readiness_gate(&availability);

        assert_eq!(readiness.artifact_kind, DENSE_GGUF_Q8_SELECTOR_READINESS_GATE_ARTIFACT_KIND);
        assert_eq!(readiness.readiness_status, DenseQ8SelectorReadinessStatus::Blocked);
        assert!(!readiness.generated_id_receipt_equivalence_passed);
        assert!(!readiness.production_compute_hook_available);
        assert!(!readiness.selector_update_ready);
        assert!(readiness.runtime_still_blocked());
        assert_eq!(readiness.selected_path, DenseQ8RuntimePath::EagerF32Candle);
        assert_eq!(
            readiness.runtime_blockers,
            vec![
                DenseQ8RuntimePreflightBlocker::GeneratedIdReceiptEquivalenceMissing,
                DenseQ8RuntimePreflightBlocker::ProductionComputeHookMissing,
                DenseQ8RuntimePreflightBlocker::ProductionSelectorStillEagerF32
            ]
        );
        assert!(!readiness.sidecar_runtime_compute_allowed);
        assert!(readiness.eager_f32_runtime_preserved);
        assert!(!readiness.dense_runtime_replaced);
        assert!(!readiness.speedup_claim);
        Ok(())
    }

    #[test]
    fn q8_selector_readiness_gate_is_ready_only_for_a_separate_selector_update() -> Result<()> {
        let preflight = runtime_preflight()?;
        let baseline = behavior_receipt("eager-f32-baseline");
        let candidate = behavior_receipt("q8-sidecar-candidate");

        let gate =
            build_dense_q8_generated_id_text_equivalence_gate(&preflight, baseline, candidate);
        let availability = build_dense_q8_production_compute_hook_availability(
            &gate,
            Some("dense-q8-sidecar-linear-hook"),
        );
        let readiness = build_dense_q8_selector_readiness_gate(&availability);

        assert_eq!(
            readiness.readiness_status,
            DenseQ8SelectorReadinessStatus::ReadyForSeparateSelectorUpdate
        );
        assert!(readiness.generated_id_receipt_equivalence_passed);
        assert!(readiness.production_compute_hook_available);
        assert!(readiness.selector_update_ready);
        assert!(readiness.runtime_still_blocked());
        assert!(!readiness.sidecar_runtime_compute_allowed);
        assert_eq!(
            readiness.runtime_blockers,
            vec![DenseQ8RuntimePreflightBlocker::ProductionSelectorStillEagerF32]
        );
        assert_eq!(readiness.selected_path, DenseQ8RuntimePath::EagerF32Candle);
        assert_eq!(readiness.selected_kernel, "dense-f32-candle-linear");
        assert!(readiness.selector_update_required_before_runtime_use);
        assert!(readiness.eager_f32_runtime_preserved);
        assert!(!readiness.dense_runtime_replaced);
        assert!(!readiness.speedup_claim);
        Ok(())
    }

    #[test]
    fn q8_selector_update_applies_only_after_readiness_gate_is_ready() -> Result<()> {
        let preflight = runtime_preflight()?;
        let baseline = behavior_receipt("eager-f32-baseline");
        let candidate = behavior_receipt("q8-sidecar-candidate");

        let gate =
            build_dense_q8_generated_id_text_equivalence_gate(&preflight, baseline, candidate);
        let availability = build_dense_q8_production_compute_hook_availability(
            &gate,
            Some("dense-q8-sidecar-linear-hook"),
        );
        let readiness = build_dense_q8_selector_readiness_gate(&availability);
        let update = build_dense_q8_selector_update(&readiness);

        assert_eq!(update.artifact_kind, DENSE_GGUF_Q8_SELECTOR_UPDATE_ARTIFACT_KIND);
        assert_eq!(
            update.update_status,
            DenseQ8SelectorUpdateStatus::AppliedToPackedSidecarCandidate
        );
        assert!(update.selects_packed_sidecar_without_speed_claim());
        assert_eq!(update.previous_selected_path, DenseQ8RuntimePath::EagerF32Candle);
        assert_eq!(update.selected_path, DenseQ8RuntimePath::PackedQ8Sidecar);
        assert_eq!(update.selected_kernel, "dense-q8-sidecar-linear");
        assert!(update.generated_id_receipt_equivalence_passed);
        assert!(update.production_compute_hook_available);
        assert!(update.selector_update_ready);
        assert!(update.selector_update_applied);
        assert!(update.sidecar_runtime_compute_allowed);
        assert!(update.runtime_blockers.is_empty());
        assert!(!update.eager_f32_runtime_preserved);
        assert!(update.dense_runtime_replaced);
        assert!(!update.speedup_claim);

        let registry = registry_with_q_proj();
        let selection = select_dense_q8_runtime_after_selector_update(
            "blk.0.attn_q.weight",
            &registry,
            &update,
        );
        assert_eq!(selection.selected_path, DenseQ8RuntimePath::PackedQ8Sidecar);
        assert_eq!(selection.selected_kernel, "dense-q8-sidecar-linear");
        assert!(selection.runtime_compute_enabled);
        assert!(selection.dense_runtime_replaced);
        assert!(!selection.speedup_claim);
        Ok(())
    }

    #[test]
    fn q8_selector_update_stays_blocked_when_behavior_evidence_differs() -> Result<()> {
        let preflight = runtime_preflight()?;
        let baseline = behavior_receipt("eager-f32-baseline");
        let mut candidate = behavior_receipt("q8-sidecar-candidate");
        candidate.generated_ids = vec![84644];

        let gate =
            build_dense_q8_generated_id_text_equivalence_gate(&preflight, baseline, candidate);
        let availability = build_dense_q8_production_compute_hook_availability(
            &gate,
            Some("dense-q8-sidecar-linear-hook"),
        );
        let readiness = build_dense_q8_selector_readiness_gate(&availability);
        let update = build_dense_q8_selector_update(&readiness);

        assert_eq!(update.update_status, DenseQ8SelectorUpdateStatus::Blocked);
        assert_eq!(update.selected_path, DenseQ8RuntimePath::EagerF32Candle);
        assert_eq!(update.selected_kernel, "dense-f32-candle-linear");
        assert!(!update.generated_id_receipt_equivalence_passed);
        assert!(update.production_compute_hook_available);
        assert!(!update.selector_update_ready);
        assert!(!update.selector_update_applied);
        assert!(!update.sidecar_runtime_compute_allowed);
        assert!(update.eager_f32_runtime_preserved);
        assert!(!update.dense_runtime_replaced);
        assert!(!update.speedup_claim);
        assert_eq!(
            update.runtime_blockers,
            vec![
                DenseQ8RuntimePreflightBlocker::GeneratedIdReceiptEquivalenceMissing,
                DenseQ8RuntimePreflightBlocker::ProductionSelectorStillEagerF32
            ]
        );

        let registry = registry_with_q_proj();
        let selection = select_dense_q8_runtime_after_selector_update(
            "blk.0.attn_q.weight",
            &registry,
            &update,
        );
        assert!(selection.selects_eager_f32());
        Ok(())
    }

    #[test]
    fn q8_selector_readiness_gate_blocks_when_hook_exists_but_behavior_differs() -> Result<()> {
        let preflight = runtime_preflight()?;
        let baseline = behavior_receipt("eager-f32-baseline");
        let mut candidate = behavior_receipt("q8-sidecar-candidate");
        candidate.decoded_text = "wrong".to_string();

        let gate =
            build_dense_q8_generated_id_text_equivalence_gate(&preflight, baseline, candidate);
        let availability = build_dense_q8_production_compute_hook_availability(
            &gate,
            Some("dense-q8-sidecar-linear-hook"),
        );
        let readiness = build_dense_q8_selector_readiness_gate(&availability);

        assert_eq!(readiness.readiness_status, DenseQ8SelectorReadinessStatus::Blocked);
        assert!(!readiness.generated_id_receipt_equivalence_passed);
        assert!(readiness.production_compute_hook_available);
        assert!(!readiness.selector_update_ready);
        assert!(readiness.runtime_still_blocked());
        assert_eq!(
            readiness.runtime_blockers,
            vec![
                DenseQ8RuntimePreflightBlocker::GeneratedIdReceiptEquivalenceMissing,
                DenseQ8RuntimePreflightBlocker::ProductionSelectorStillEagerF32
            ]
        );
        assert!(!readiness.sidecar_runtime_compute_allowed);
        assert!(!readiness.dense_runtime_replaced);
        assert!(!readiness.speedup_claim);
        Ok(())
    }
}

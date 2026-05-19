use bitnet_device_probe::{
    APPLE_M3_AIR_MACHINE_ID, APPLE_M3_AIR_METAL_BACKEND, APPLE_M3_AIR_MPSGRAPH_BACKEND,
    APPLE_VISIBILITY_PREFLIGHT_KIND, AppleBackendReceipt, AppleBackendVisibilityPreflight,
    AppleM3AirHostProfileContract, AppleM3AirUnsupportedClaim, AppleReceiptError,
    AppleResolvedDevice,
};
use std::error::Error;
use std::io;

fn m4_device() -> AppleResolvedDevice {
    AppleResolvedDevice::new("Apple M4").with_gpu_cores(10).with_unified_memory(true)
}

fn m3_air_device() -> AppleResolvedDevice {
    AppleResolvedDevice::new("Apple M3").with_gpu_cores(10).with_unified_memory(true)
}

#[test]
fn apple_m3_air_host_profile_contract_records_device_and_claim_boundaries()
-> Result<(), Box<dyn Error>> {
    let contract = AppleM3AirHostProfileContract::current();

    contract.validate()?;

    let value = serde_json::to_value(&contract)?;
    ensure(value["machine_id"] == APPLE_M3_AIR_MACHINE_ID, "wrong machine id")?;
    ensure(value["soc_family"] == "Apple M3", "wrong SoC family")?;
    ensure(value["thermal_policy"] == "fanless_mobile", "wrong thermal policy")?;
    ensure(value["core_split_required"] == true, "core split is not required")?;
    ensure(value["memory_tier_required"] == true, "memory tier is not required")?;
    ensure(value["storage"]["cache_root_required"] == true, "cache root is not required")?;
    ensure(
        value["storage"]["large_artifact_sweep_allowed"] == true,
        "large artifact sweep is not allowed",
    )?;
    ensure(
        value["storage"]["model_binaries_committed"] == false,
        "model binaries are allowed in git",
    )?;

    let labels = value["proof_lane_labels"]
        .as_array()
        .ok_or_else(|| io::Error::other("proof_lane_labels is not an array"))?;
    ensure(
        labels.iter().any(|label| {
            label["backend_label"] == "apple-m3-air-cpu-neon"
                && label["runtime_api"] == "cpu-neon"
                && label["execution_available"] == true
        }),
        "CPU/NEON proof label is missing",
    )?;
    ensure(
        labels.iter().any(|label| {
            label["backend_label"] == APPLE_M3_AIR_METAL_BACKEND
                && label["runtime_api"] == "metal"
                && label["execution_available"] == false
        }),
        "Metal visibility-only label is missing",
    )?;
    ensure(
        labels.iter().any(|label| {
            label["backend_label"] == APPLE_M3_AIR_MPSGRAPH_BACKEND
                && label["runtime_api"] == "mpsgraph"
                && label["execution_available"] == false
        }),
        "MPSGraph visibility-only label is missing",
    )?;
    ensure(
        contract.unsupported_claims.contains(&AppleM3AirUnsupportedClaim::MetalModelInference),
        "Metal model inference is not explicitly unsupported",
    )?;
    ensure(
        contract.unsupported_claims.contains(&AppleM3AirUnsupportedClaim::Qk256AppleSilicon),
        "QK256 Apple Silicon support is not explicitly unsupported",
    )?;
    Ok(())
}

#[test]
fn apple_m3_air_host_profile_contract_rejects_accelerator_execution_claims()
-> Result<(), Box<dyn Error>> {
    let mut contract = AppleM3AirHostProfileContract::current();
    let metal = contract
        .proof_lane_labels
        .iter_mut()
        .find(|label| label.backend_label == APPLE_M3_AIR_METAL_BACKEND)
        .ok_or_else(|| io::Error::other("missing Metal label"))?;
    metal.execution_available = true;

    ensure(
        contract.validate()
            == Err(AppleReceiptError::ClaimBoundaryViolation("accelerator_execution")),
        "Metal execution claim was not rejected",
    )?;
    Ok(())
}

#[test]
fn apple_m3_air_host_profile_contract_rejects_mislabelled_runtime_api() -> Result<(), Box<dyn Error>>
{
    let mut contract = AppleM3AirHostProfileContract::current();
    let metal = contract
        .proof_lane_labels
        .iter_mut()
        .find(|label| label.backend_label == APPLE_M3_AIR_METAL_BACKEND)
        .ok_or_else(|| io::Error::other("missing Metal label"))?;
    metal.runtime_api = "cpu-neon".to_owned();

    ensure(
        contract.validate()
            == Err(AppleReceiptError::InvalidProfileField("proof_lane_labels.runtime_api")),
        "Metal label accepted a CPU runtime API",
    )?;
    Ok(())
}

#[test]
fn apple_metal_smoke_receipt_preserves_backend_identity() {
    let receipt = AppleBackendReceipt::new(
        "apple-m4-mac-mini",
        "smoke",
        "apple-m4-metal",
        Some("apple-m4-metal"),
        "metal",
        m4_device(),
        false,
        "ci/hardware/apple-m4-mac-mini/2026-05-06/metal-smoke.json",
    )
    .with_kernel_id("tiny_metal_add_smoke")
    .with_result("pass");

    receipt.validate().expect("valid Apple Metal smoke receipt");

    let value = serde_json::to_value(&receipt).expect("receipt serializes");
    assert_eq!(value["requested_backend"], "apple-m4-metal");
    assert_eq!(value["selected_backend"], "apple-m4-metal");
    assert_eq!(value["runtime_api"], "metal");
    assert_eq!(value["resolved_device"]["chip"], "Apple M4");
    assert_eq!(value["resolved_device"]["gpu_cores"], 10);
    assert_eq!(value["resolved_device"]["unified_memory"], true);
    assert_eq!(value["fallback_used"], false);
    assert_eq!(value["kernel_id"], "tiny_metal_add_smoke");
    assert!(value.get("fallback_reason").is_none());
}

#[test]
fn apple_mpsgraph_receipt_records_graph_not_native_kernel() {
    let receipt = AppleBackendReceipt::new(
        "apple-m4-mac-mini",
        "smoke",
        "apple-m4-mpsgraph",
        Some("apple-m4-mpsgraph"),
        "mpsgraph",
        m4_device(),
        false,
        "ci/hardware/apple-m4-mac-mini/2026-05-06/mpsgraph-smoke.json",
    )
    .with_graph_id("tiny_mpsgraph_matmul")
    .with_resolved_target("unknown")
    .with_result("pass");

    receipt.validate().expect("valid Apple MPSGraph smoke receipt");

    let value = serde_json::to_value(&receipt).expect("receipt serializes");
    assert_eq!(value["requested_backend"], "apple-m4-mpsgraph");
    assert_eq!(value["selected_backend"], "apple-m4-mpsgraph");
    assert_eq!(value["runtime_api"], "mpsgraph");
    assert_eq!(value["graph_id"], "tiny_mpsgraph_matmul");
    assert_eq!(value["resolved_target"], "unknown");
    assert!(value.get("kernel_id").is_none());
}

#[test]
fn fallback_receipts_require_a_reason() {
    let receipt = AppleBackendReceipt::new(
        "apple-m4-mac-mini",
        "smoke",
        "apple-m4-metal",
        Some("apple-m4-cpu-neon"),
        "cpu",
        m4_device(),
        true,
        "ci/hardware/apple-m4-mac-mini/2026-05-06/fallback.json",
    );

    assert_eq!(receipt.validate(), Err(AppleReceiptError::MissingFallbackReason));
}

#[test]
fn nonfallback_receipts_reject_fallback_reason() {
    let receipt = AppleBackendReceipt::new(
        "apple-m4-mac-mini",
        "smoke",
        "apple-m4-metal",
        Some("apple-m4-metal"),
        "metal",
        m4_device(),
        false,
        "ci/hardware/apple-m4-mac-mini/2026-05-06/metal-smoke.json",
    )
    .with_fallback_reason("Metal unavailable");

    assert_eq!(receipt.validate(), Err(AppleReceiptError::UnexpectedFallbackReason));
}

#[test]
fn receipts_reject_ambiguous_kernel_and_graph_identity() {
    let receipt = AppleBackendReceipt::new(
        "apple-m4-mac-mini",
        "smoke",
        "apple-m4-mpsgraph",
        Some("apple-m4-mpsgraph"),
        "mpsgraph",
        m4_device(),
        false,
        "ci/hardware/apple-m4-mac-mini/2026-05-06/mpsgraph-smoke.json",
    )
    .with_kernel_id("native_metal_kernel")
    .with_graph_id("tiny_mpsgraph_matmul");

    assert_eq!(receipt.validate(), Err(AppleReceiptError::AmbiguousWorkId));
}

#[test]
fn m3_air_metal_visibility_preflight_preserves_bounded_claims() -> Result<(), Box<dyn Error>> {
    let receipt = AppleBackendVisibilityPreflight::m3_air_metal(
        Some(APPLE_M3_AIR_METAL_BACKEND),
        m3_air_device(),
        true,
        false,
        "target/apple-m3-air/preflight/metal-visibility.json",
    );

    receipt.validate()?;

    let value = serde_json::to_value(&receipt)?;
    ensure(value["machine_id"] == APPLE_M3_AIR_MACHINE_ID, "wrong machine id")?;
    ensure(value["artifact_kind"] == APPLE_VISIBILITY_PREFLIGHT_KIND, "wrong artifact kind")?;
    ensure(value["requested_backend"] == APPLE_M3_AIR_METAL_BACKEND, "wrong requested backend")?;
    ensure(value["selected_backend"] == APPLE_M3_AIR_METAL_BACKEND, "wrong selected backend")?;
    ensure(value["runtime_api"] == "metal", "wrong runtime API")?;
    ensure(value["metal_visible"] == true, "Metal visibility not recorded")?;
    ensure(value["fallback_used"] == false, "unexpected fallback")?;
    ensure(value["claim_boundary"]["model_loaded"] == false, "model load was claimed")?;
    ensure(value["claim_boundary"]["model_inference"] == false, "model inference was claimed")?;
    ensure(value["claim_boundary"]["performance_claimed"] == false, "performance was claimed")?;
    Ok(())
}

#[test]
fn m3_air_mpsgraph_visibility_preflight_is_graph_visibility_not_model_inference()
-> Result<(), Box<dyn Error>> {
    let receipt = AppleBackendVisibilityPreflight::m3_air_mpsgraph(
        Some(APPLE_M3_AIR_MPSGRAPH_BACKEND),
        m3_air_device(),
        true,
        true,
        false,
        "target/apple-m3-air/preflight/mpsgraph-visibility.json",
    );

    receipt.validate()?;

    let value = serde_json::to_value(&receipt)?;
    ensure(value["machine_id"] == APPLE_M3_AIR_MACHINE_ID, "wrong machine id")?;
    ensure(value["requested_backend"] == APPLE_M3_AIR_MPSGRAPH_BACKEND, "wrong requested backend")?;
    ensure(value["selected_backend"] == APPLE_M3_AIR_MPSGRAPH_BACKEND, "wrong selected backend")?;
    ensure(value["runtime_api"] == "mpsgraph", "wrong runtime API")?;
    ensure(value["metal_visible"] == true, "Metal visibility not recorded")?;
    ensure(value["mpsgraph_visible"] == true, "MPSGraph visibility not recorded")?;
    ensure(
        value["claim_boundary"]["mpsgraph_model_inference_claimed"] == false,
        "MPSGraph model inference was claimed",
    )?;
    ensure(
        value["claim_boundary"]["neural_engine_claimed"] == false,
        "Neural Engine execution was claimed",
    )?;
    Ok(())
}

#[test]
fn m3_air_visibility_preflight_rejects_generic_backend_alias() -> Result<(), Box<dyn Error>> {
    let receipt = AppleBackendVisibilityPreflight::new(
        APPLE_M3_AIR_MACHINE_ID,
        "metal",
        Some("metal"),
        "metal",
        m3_air_device(),
        bitnet_device_probe::AppleRuntimeVisibility { metal_visible: true, mpsgraph_visible: None },
        false,
        "target/apple-m3-air/preflight/generic-metal.json",
    );

    let expected = Err(AppleReceiptError::UnsupportedAppleBackend {
        machine_id: APPLE_M3_AIR_MACHINE_ID,
        requested_backend: "metal".to_string(),
    });
    ensure(
        receipt.validate() == expected,
        "generic backend alias was not rejected for M3 Air preflight",
    )?;
    Ok(())
}

#[test]
fn m3_air_visibility_preflight_rejects_generic_selected_backend() -> Result<(), Box<dyn Error>> {
    let receipt = AppleBackendVisibilityPreflight::m3_air_metal(
        Some("metal"),
        m3_air_device(),
        true,
        false,
        "target/apple-m3-air/preflight/generic-selected-metal.json",
    );

    ensure(
        receipt.validate()
            == Err(AppleReceiptError::UnsupportedAppleSelectedBackend {
                machine_id: APPLE_M3_AIR_MACHINE_ID,
                selected_backend: "metal".to_string(),
            }),
        "generic selected backend was not rejected for M3 Air preflight",
    )?;
    Ok(())
}

#[test]
fn visibility_preflight_rejects_model_claims() -> Result<(), Box<dyn Error>> {
    let mut receipt = AppleBackendVisibilityPreflight::m3_air_metal(
        Some(APPLE_M3_AIR_METAL_BACKEND),
        m3_air_device(),
        true,
        false,
        "target/apple-m3-air/preflight/metal-visibility.json",
    );
    receipt.claim_boundary.model_loaded = true;

    ensure(
        receipt.validate() == Err(AppleReceiptError::ClaimBoundaryViolation("model_loaded")),
        "model-loaded claim was not rejected",
    )?;
    Ok(())
}

fn ensure(condition: bool, message: &'static str) -> Result<(), Box<dyn Error>> {
    if condition { Ok(()) } else { Err(Box::new(io::Error::other(message))) }
}

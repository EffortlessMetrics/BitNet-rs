use bitnet_device_probe::{AppleBackendReceipt, AppleReceiptError, AppleResolvedDevice};

fn m4_device() -> AppleResolvedDevice {
    AppleResolvedDevice::new("Apple M4").with_gpu_cores(10).with_unified_memory(true)
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

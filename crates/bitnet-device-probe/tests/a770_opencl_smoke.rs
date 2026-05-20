#![cfg(feature = "opencl")]

use std::{error::Error, io, path::Path};

use bitnet_device_probe::runtimes::run_a770_opencl_tiny_kernel_smoke;
use serde_json::json;

const RUN_ENV: &str = "BITNET_RUN_A770_OPENCL_SMOKE";
const RECEIPT_ENV: &str = "BITNET_A770_OPENCL_SMOKE_RECEIPT";

#[test]
fn a770_selected_opencl_tiny_vector_add_smoke_runs_when_enabled() -> Result<(), Box<dyn Error>> {
    if std::env::var(RUN_ENV).as_deref() != Ok("1") {
        eprintln!("skipping live A770 OpenCL smoke; set {RUN_ENV}=1 to run it");
        return Ok(());
    }

    let smoke = run_a770_opencl_tiny_kernel_smoke();
    if !smoke.passed {
        return Err(io_error(format!(
            "A770-005 tiny OpenCL smoke did not pass: {:?}",
            smoke.error
        )));
    }

    assert_eq!(smoke.requested_backend, "intel-arc-a770");
    assert_eq!(smoke.selected_backend.as_deref(), Some("intel-arc-a770-opencl"));
    assert_eq!(smoke.runtime_api.as_deref(), Some("opencl"));
    assert!(smoke.runtime_device.as_deref().is_some_and(is_a770_device_name));
    assert!(smoke.platform_index.is_some());
    assert!(smoke.device_index.is_some());
    assert!(smoke.kernel_execution);
    assert!(!smoke.fallback_used);
    assert!(!smoke.cpu_fallback_allowed);
    assert!(!smoke.bitnet_inference);
    assert!(!smoke.qk256_decode);
    assert!(smoke.max_abs_error.is_some_and(|value| value <= smoke.tolerance));

    let mut receipt = serde_json::to_value(&smoke)?;
    let object =
        receipt.as_object_mut().ok_or_else(|| io_error("OpenCL smoke receipt is not an object"))?;
    object.insert("campaign".to_owned(), json!("intel-a770"));
    object.insert("work_item".to_owned(), json!("A770-005"));
    object.insert("proof_family".to_owned(), json!("a770_opencl_tiny_vector_add_smoke"));
    object.insert("model_family".to_owned(), serde_json::Value::Null);
    object.insert("claim_allowed".to_owned(), json!(false));
    object.insert("diagnostic_only".to_owned(), json!(true));
    object.insert("performance_claim".to_owned(), json!(false));
    object.insert("bitnet_qk256_claim".to_owned(), json!(false));
    object.insert("full_residency_claim".to_owned(), json!(false));
    object.insert(
        "must_not_claim".to_owned(),
        json!([
            "BitNet QK256 runs on A770",
            "BitNet inference works on A770",
            "A770 trusted partial acceleration is claim-grade",
            "Arc 140V DEV_64A0 can satisfy A770 DEV_56A0 selected-device smoke"
        ]),
    );

    if let Ok(path) = std::env::var(RECEIPT_ENV) {
        if let Some(parent) = Path::new(&path).parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, serde_json::to_string_pretty(&receipt)?)?;
    }

    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}

fn is_a770_device_name(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    (lower.contains("arc") && lower.contains("a770")) || lower.contains("56a0")
}

fn io_error(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::other(message.into()))
}

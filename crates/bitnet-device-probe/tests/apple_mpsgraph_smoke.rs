#![cfg(feature = "metal")]

use bitnet_device_probe::{
    APPLE_M4_MPSGRAPH_BACKEND, APPLE_M4_MPSGRAPH_RESOLVED_TARGET_UNKNOWN,
    APPLE_M4_MPSGRAPH_RUNTIME_API, TINY_MPSGRAPH_MATMUL_GRAPH_ID, TinyMpsGraphSmokeComparison,
    TinyMpsGraphSmokeReceipt, apple_mpsgraph_smoke_artifact_path,
    compare_tiny_mpsgraph_matmul_outputs, expected_tiny_mpsgraph_matmul,
    tiny_mpsgraph_matmul_inputs, tiny_mpsgraph_smoke_swift_source,
};

#[test]
fn tiny_mpsgraph_matmul_expected_output_matches_cpu_reference() {
    let (lhs, rhs) = tiny_mpsgraph_matmul_inputs();
    let expected = expected_tiny_mpsgraph_matmul(&lhs, &rhs).expect("valid tiny matmul inputs");

    assert_eq!(expected, vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn mpsgraph_receipt_contract_keeps_reference_lane_separate() {
    let receipt = TinyMpsGraphSmokeReceipt::passed(
        apple_mpsgraph_smoke_artifact_path("2026-05-06"),
        4,
        TinyMpsGraphSmokeComparison { max_abs_error: 0.0, mean_abs_error: 0.0 },
    );

    assert_eq!(receipt.machine_id, "apple-m4-mac-mini");
    assert_eq!(receipt.artifact_kind, "smoke");
    assert_eq!(receipt.requested_backend, APPLE_M4_MPSGRAPH_BACKEND);
    assert_eq!(receipt.selected_backend, APPLE_M4_MPSGRAPH_BACKEND);
    assert_eq!(receipt.runtime_api, APPLE_M4_MPSGRAPH_RUNTIME_API);
    assert_eq!(receipt.graph_id, TINY_MPSGRAPH_MATMUL_GRAPH_ID);
    assert_eq!(receipt.resolved_target, APPLE_M4_MPSGRAPH_RESOLVED_TARGET_UNKNOWN);
    assert!(!receipt.fallback_used);
    assert_eq!(receipt.result, "pass");
    assert_eq!(
        receipt.artifact_path,
        "ci/hardware/apple-m4-mac-mini/2026-05-06/mpsgraph-smoke.json"
    );
}

#[test]
fn comparison_fails_instead_of_claiming_graph_success() {
    let expected = [19.0_f32, 22.0, 43.0, 50.0];
    let actual = [19.0_f32, 220.0, 43.0, 50.0];

    let error = compare_tiny_mpsgraph_matmul_outputs(&expected, &actual, 1e-6)
        .expect_err("mismatch should fail the MPSGraph smoke contract");

    assert!(error.to_string().contains("output mismatch"), "unexpected error: {error}");
}

#[test]
fn swift_source_names_mpsgraph_not_native_metal_or_neural_engine() {
    let source = tiny_mpsgraph_smoke_swift_source();

    assert!(source.contains("MetalPerformanceShadersGraph"));
    assert!(source.contains("tiny_mpsgraph_matmul"));
    assert!(source.contains("\"resolved_target\": \"unknown\""));
    assert!(!source.contains("NeuralEngine"));
    assert!(!source.contains("MTLComputePipelineState"));
}

#[cfg(target_os = "macos")]
mod live_mpsgraph {
    use super::*;
    use serde_json::json;
    use std::error::Error;
    use std::io;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    const RUN_ENV: &str = "BITNET_RUN_M4_MPSGRAPH_SMOKE";
    const RECEIPT_ENV: &str = "BITNET_M4_MPSGRAPH_SMOKE_RECEIPT";
    const ARTIFACT_PATH_ENV: &str = "BITNET_M4_MPSGRAPH_SMOKE_ARTIFACT_PATH";

    #[test]
    fn tiny_m4_mpsgraph_matmul_smoke_runs_when_enabled() -> Result<(), Box<dyn Error>> {
        if std::env::var(RUN_ENV).as_deref() != Ok("1") {
            eprintln!("skipping live M4 MPSGraph smoke; set {RUN_ENV}=1 to run it");
            return Ok(());
        }

        let swift_output = run_swift_mpsgraph_smoke()?;
        let device_name = swift_output
            .get("device_name")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| io_error("MPSGraph smoke output missing device_name"))?;
        if !is_apple_m4_device_name(device_name) {
            return Err(io_error(format!(
                "M4-007 MPSGraph smoke requires an Apple M4-family device; found '{device_name}'"
            )));
        }

        let actual = parse_output_values(&swift_output)?;
        let (lhs, rhs) = tiny_mpsgraph_matmul_inputs();
        let expected = expected_tiny_mpsgraph_matmul(&lhs, &rhs)?;
        let comparison = compare_tiny_mpsgraph_matmul_outputs(&expected, &actual, 1e-6)?;

        let artifact_path = std::env::var(ARTIFACT_PATH_ENV)
            .or_else(|_| std::env::var(RECEIPT_ENV))
            .unwrap_or_else(|_| {
                "ci/hardware/apple-m4-mac-mini/<date>/mpsgraph-smoke.json".to_string()
            });
        let receipt =
            TinyMpsGraphSmokeReceipt::passed(artifact_path.clone(), expected.len(), comparison);

        let receipt_json = json!({
            "machine_id": receipt.machine_id,
            "artifact_kind": receipt.artifact_kind,
            "requested_backend": receipt.requested_backend,
            "selected_backend": receipt.selected_backend,
            "runtime_api": receipt.runtime_api,
            "resolved_device": {
                "chip": device_name,
                "unified_memory": true
            },
            "graph_id": receipt.graph_id,
            "resolved_target": receipt.resolved_target,
            "fallback_used": receipt.fallback_used,
            "result": receipt.result,
            "artifact_path": receipt.artifact_path,
            "element_count": receipt.element_count,
            "max_abs_error": receipt.max_abs_error,
            "mean_abs_error": receipt.mean_abs_error
        });

        if let Ok(path) = std::env::var(RECEIPT_ENV) {
            if let Some(parent) = Path::new(&path).parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&path, serde_json::to_string_pretty(&receipt_json)?)?;
        }

        println!("{}", serde_json::to_string_pretty(&receipt_json)?);
        Ok(())
    }

    fn run_swift_mpsgraph_smoke() -> Result<serde_json::Value, Box<dyn Error>> {
        let script_path = write_temp_swift_script()?;
        let output = Command::new("xcrun").args(["swift"]).arg(&script_path).output()?;
        let _ = std::fs::remove_file(&script_path);

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(io_error(format!("xcrun swift MPSGraph smoke failed: {stderr}")));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        serde_json::from_str(stdout.trim())
            .map_err(|error| io_error(format!("failed to parse MPSGraph smoke JSON: {error}")))
    }

    fn write_temp_swift_script() -> Result<PathBuf, Box<dyn Error>> {
        let path = std::env::temp_dir().join(format!(
            "bitnet-m4-mpsgraph-smoke-{}-{}.swift",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::write(&path, tiny_mpsgraph_smoke_swift_source())?;
        Ok(path)
    }

    fn parse_output_values(value: &serde_json::Value) -> Result<Vec<f32>, Box<dyn Error>> {
        let values = value
            .get("output")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| io_error("MPSGraph smoke output missing output array"))?;

        values
            .iter()
            .map(|value| {
                value
                    .as_f64()
                    .map(|number| number as f32)
                    .ok_or_else(|| io_error("MPSGraph smoke output array contains non-number"))
            })
            .collect()
    }

    fn is_apple_m4_device_name(device_name: &str) -> bool {
        device_name.to_ascii_lowercase().contains("apple m4")
    }

    fn io_error(message: impl Into<String>) -> Box<dyn Error> {
        Box::new(io::Error::other(message.into()))
    }
}

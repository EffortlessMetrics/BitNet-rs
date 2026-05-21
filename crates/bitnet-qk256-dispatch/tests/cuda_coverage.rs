#[cfg(feature = "opencl")]
use bitnet_qk256_dispatch::{
    Qk256A770OpenClRuntimeStats, Qk256DispatchCoverageCounters, forward_qk256_with_scale,
    qk256_a770_opencl_runtime_stats,
};
use bitnet_qk256_dispatch::{
    forward_qk256, qk256_cuda_runtime_stats, qk256_dispatch_coverage,
    record_bitnet_linear_cpu_fallback, record_bitnet_linear_unsupported,
    reset_qk256_dispatch_coverage,
};
use candle_core::{DType, Device, Tensor};
use std::error::Error;
#[cfg(feature = "opencl")]
use std::path::PathBuf;
use std::sync::Mutex;

static ENV_LOCK: Mutex<()> = Mutex::new(());
type TestResult = Result<(), Box<dyn Error>>;
#[cfg(feature = "opencl")]
const A770_DISPATCH_RECEIPT_ENV: &str = "BITNET_A770_OPENCL_DISPATCH_RECEIPT";
#[cfg(feature = "opencl")]
const A770_STRICT_DISPATCH_NOT_CLAIMS: &[&str] = &[
    "BitNet inference works on A770",
    "A770 answer quality is proven",
    "Activation quantization is GPU-resident",
    "Selected attention is resident",
    "Resident KV is proven",
    "Attention score residency is proven",
    "Softmax residency is proven",
    "Value-mix residency is proven",
    "Full A770 residency is proven",
    "A770 performance speedup is proven",
    "A770 trusted partial acceleration is claim-grade",
];

fn clear_backend_env() {
    unsafe {
        std::env::remove_var("BITNET_SELECTED_BACKEND");
        std::env::remove_var("BITNET_REQUESTED_BACKEND");
        std::env::remove_var("BITNET_BACKEND");
        std::env::remove_var("BITNET_STRICT_MODE");
        std::env::remove_var("BITNET_STRICT_CUDA_BACKEND");
    }
}

fn qk256_tensor(rows: usize, cols: usize, device: &Device) -> Tensor {
    let row_stride = cols.div_ceil(256) * 64;
    Tensor::from_vec(vec![0xaa_u8; rows * row_stride], &[rows, row_stride], device).unwrap()
}

#[test]
fn cpu_selected_qk256_forward_records_total_without_fallback() {
    let _guard = ENV_LOCK.lock().unwrap();
    clear_backend_env();
    reset_qk256_dispatch_coverage();
    let device = Device::Cpu;
    let input = Tensor::ones(&[1, 1, 256], DType::F32, &device).unwrap();
    let qk256 = qk256_tensor(2, 256, &device);

    let output =
        forward_qk256(&input, &qk256, "layers.0.attention.q_proj.weight.qk256_qs").unwrap();
    let coverage = qk256_dispatch_coverage();

    assert_eq!(output.dims(), &[1, 1, 2]);
    assert_eq!(coverage.bitnet_linear_layers_total, 1);
    assert_eq!(coverage.bitnet_linear_layers_on_cuda, 0);
    assert_eq!(coverage.bitnet_linear_layers_cpu_fallback, 0);
    assert!(coverage.unsupported_ops.is_empty());
    assert_eq!(coverage.execution_claim, "cpu_reference");
}

#[test]
fn missing_qk256_tensor_fallback_counter_is_backend_aware() {
    let _guard = ENV_LOCK.lock().unwrap();
    clear_backend_env();
    reset_qk256_dispatch_coverage();
    record_bitnet_linear_cpu_fallback();
    assert_eq!(qk256_dispatch_coverage().bitnet_linear_layers_cpu_fallback, 0);

    unsafe {
        std::env::set_var("BITNET_SELECTED_BACKEND", "nvidia-rtx-5070-ti-cuda");
    }
    record_bitnet_linear_cpu_fallback();
    let coverage = qk256_dispatch_coverage();
    assert_eq!(coverage.bitnet_linear_layers_total, 2);
    assert_eq!(coverage.bitnet_linear_layers_cpu_fallback, 1);
    assert_eq!(coverage.unsupported_ops, vec!["qk256_cpu_fallback"]);
    assert_eq!(coverage.execution_claim, "cuda_bitnet_not_routed");
    clear_backend_env();
}

#[test]
fn strict_unsupported_counter_records_receipt_boundary() {
    let _guard = ENV_LOCK.lock().unwrap();
    clear_backend_env();
    reset_qk256_dispatch_coverage();
    unsafe {
        std::env::set_var("BITNET_SELECTED_BACKEND", "nvidia-rtx-5070-ti-cuda");
    }

    record_bitnet_linear_unsupported();
    let coverage = qk256_dispatch_coverage();

    assert_eq!(coverage.bitnet_linear_layers_total, 1);
    assert_eq!(coverage.bitnet_linear_layers_on_cuda, 0);
    assert_eq!(coverage.bitnet_linear_layers_cpu_fallback, 0);
    assert_eq!(coverage.unsupported_ops, vec!["qk256_strict_cuda_unsupported"]);
    assert_eq!(coverage.execution_claim, "cuda_bitnet_not_routed");
    clear_backend_env();
}

#[test]
fn a770_opencl_request_records_cpu_fallback_as_not_routed() -> TestResult {
    let _guard = ENV_LOCK.lock().map_err(|_| std::io::Error::other("env lock poisoned"))?;
    clear_backend_env();
    reset_qk256_dispatch_coverage();
    unsafe {
        std::env::set_var("BITNET_SELECTED_BACKEND", "intel-a770-opencl");
    }
    let device = Device::Cpu;
    let input = Tensor::ones(&[1, 1, 256], DType::F32, &device)?;
    let qk256 = qk256_tensor(2, 256, &device);

    let output = forward_qk256(&input, &qk256, "layers.0.attention.q_proj.weight.qk256_qs")?;
    let coverage = qk256_dispatch_coverage();

    assert_eq!(output.dims(), &[1, 1, 2]);
    assert_eq!(coverage.bitnet_linear_layers_total, 1);
    assert_eq!(coverage.bitnet_linear_layers_on_cuda, 0);
    assert_eq!(coverage.bitnet_linear_layers_cpu_fallback, 1);
    assert_eq!(
        coverage.unsupported_ops,
        vec!["qk256_cpu_fallback".to_string(), "qk256_a770_opencl_not_routed".to_string(),]
    );
    assert_eq!(coverage.execution_claim, "a770_opencl_not_routed");
    clear_backend_env();
    Ok(())
}

#[test]
fn strict_a770_opencl_request_rejects_cpu_qk256_fallback() -> TestResult {
    let _guard = ENV_LOCK.lock().map_err(|_| std::io::Error::other("env lock poisoned"))?;
    clear_backend_env();
    reset_qk256_dispatch_coverage();
    unsafe {
        std::env::set_var("BITNET_SELECTED_BACKEND", "intel-arc-a770-opencl");
        std::env::set_var("BITNET_STRICT_MODE", "1");
    }
    let device = Device::Cpu;
    let input = Tensor::ones(&[1, 1, 256], DType::F32, &device)?;
    let qk256 = qk256_tensor(2, 256, &device);

    let result = forward_qk256(&input, &qk256, "layers.0.attention.q_proj.weight.qk256_qs");

    let err = match result {
        Ok(_) => {
            return Err(
                std::io::Error::other("strict A770 OpenCL QK256 request must fail closed").into()
            );
        }
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("requires an inline BitNet scale"),
        "unexpected strict A770 error: {err}"
    );
    let coverage = qk256_dispatch_coverage();
    assert_eq!(coverage.bitnet_linear_layers_total, 1);
    assert_eq!(coverage.bitnet_linear_layers_on_cuda, 0);
    assert_eq!(coverage.bitnet_linear_layers_cpu_fallback, 0);
    assert_eq!(coverage.unsupported_ops, vec!["qk256_a770_opencl_not_routed".to_string()]);
    assert_eq!(coverage.execution_claim, "a770_opencl_not_routed");
    clear_backend_env();
    Ok(())
}

#[cfg(feature = "opencl")]
#[test]
fn opt_in_strict_a770_opencl_inline_scale_routes_qk256_without_cpu_fallback() -> TestResult {
    if std::env::var("BITNET_RUN_A770_OPENCL_DISPATCH").as_deref() != Ok("1") {
        return Ok(());
    }

    let _guard = ENV_LOCK.lock().map_err(|_| std::io::Error::other("env lock poisoned"))?;
    clear_backend_env();
    reset_qk256_dispatch_coverage();

    let device = Device::Cpu;
    let input = Tensor::ones(&[1, 1, 256], DType::F32, &device)?;
    let qk256 = qk256_tensor(2, 256, &device);
    let cpu_output = forward_qk256_with_scale(
        &input,
        &qk256,
        "layers.0.attention.q_proj.weight.qk256_qs",
        Some(0.5),
    )?;
    reset_qk256_dispatch_coverage();

    unsafe {
        std::env::set_var("BITNET_SELECTED_BACKEND", "intel-arc-a770-opencl");
        std::env::set_var("BITNET_STRICT_MODE", "1");
    }
    let a770_output = forward_qk256_with_scale(
        &input,
        &qk256,
        "layers.0.attention.q_proj.weight.qk256_qs",
        Some(0.5),
    )?;

    let a770_values = a770_output.to_vec3::<f32>()?;
    let cpu_values = cpu_output.to_vec3::<f32>()?;
    let max_abs_error = max_abs_error_3d(&a770_values, &cpu_values);
    assert_eq!(a770_values, cpu_values);
    let coverage = qk256_dispatch_coverage();
    assert_eq!(coverage.bitnet_linear_layers_total, 1);
    assert_eq!(coverage.bitnet_linear_layers_on_a770_opencl, 1);
    assert_eq!(coverage.bitnet_linear_layers_cpu_fallback, 0);
    assert!(coverage.unsupported_ops.is_empty());
    assert_eq!(coverage.execution_claim, "a770_opencl_qk256_contribution");

    let stats = qk256_a770_opencl_runtime_stats();
    assert!(stats.host_to_device_bytes > 0);
    assert!(stats.device_to_host_bytes > 0);
    assert_eq!(stats.kernel_invocations, 1);
    let device = stats.last_device.as_ref().ok_or_else(|| {
        std::io::Error::other("strict A770 dispatch did not record selected OpenCL device")
    })?;
    assert!(
        device.runtime_device.to_ascii_lowercase().contains("a770"),
        "strict dispatch selected non-A770 OpenCL device: {}",
        device.runtime_device
    );
    write_a770_strict_dispatch_receipt_if_requested(&coverage, &stats, max_abs_error)?;
    clear_backend_env();
    Ok(())
}

#[cfg(feature = "opencl")]
fn max_abs_error_3d(actual: &[Vec<Vec<f32>>], expected: &[Vec<Vec<f32>>]) -> f32 {
    actual
        .iter()
        .flatten()
        .flatten()
        .zip(expected.iter().flatten().flatten())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max)
}

#[cfg(feature = "opencl")]
fn write_a770_strict_dispatch_receipt_if_requested(
    coverage: &Qk256DispatchCoverageCounters,
    stats: &Qk256A770OpenClRuntimeStats,
    max_abs_error: f32,
) -> TestResult {
    let Some(path) = std::env::var_os(A770_DISPATCH_RECEIPT_ENV).map(PathBuf::from) else {
        return Ok(());
    };
    let path = workspace_relative_path(path);
    let device = stats.last_device.as_ref().ok_or_else(|| {
        std::io::Error::other("strict A770 dispatch receipt missing selected OpenCL device")
    })?;
    let unsupported_ops = json_string_array(coverage.unsupported_ops.iter().map(String::as_str));
    let not_claims = json_string_array(A770_STRICT_DISPATCH_NOT_CLAIMS.iter().copied());
    let receipt = format!(
        concat!(
            "{{\n",
            "  \"schema_version\": 1,\n",
            "  \"work_item\": \"A770-012\",\n",
            "  \"artifact_kind\": \"a770_strict_qk256_dispatch_receipt\",\n",
            "  \"proof_family\": \"a770_selected_device_qk256_dispatch\",\n",
            "  \"claim_level\": \"diagnostic\",\n",
            "  \"requested_backend\": \"intel-arc-a770-opencl\",\n",
            "  \"selected_backend\": \"intel-arc-a770-opencl\",\n",
            "  \"runtime_api\": \"opencl\",\n",
            "  \"runtime_device\": {{\n",
            "    \"platform_index\": {platform_index},\n",
            "    \"device_index\": {device_index},\n",
            "    \"platform_name\": {platform_name},\n",
            "    \"name\": {runtime_device},\n",
            "    \"vendor\": {vendor},\n",
            "    \"driver_version\": {driver_version}\n",
            "  }},\n",
            "  \"strict_mode\": true,\n",
            "  \"fallback_used\": false,\n",
            "  \"cpu_fallback_allowed\": false,\n",
            "  \"kernel_execution\": true,\n",
            "  \"qk256_dispatch\": true,\n",
            "  \"qk256_decode\": false,\n",
            "  \"bitnet_inference\": false,\n",
            "  \"full_model_decode\": false,\n",
            "  \"activation_quantization_resident\": false,\n",
            "  \"quality_claim\": false,\n",
            "  \"speedup_claim\": false,\n",
            "  \"residency_claim\": false,\n",
            "  \"trusted_partial_claim\": false,\n",
            "  \"coverage\": {{\n",
            "    \"bitnet_linear_layers_total\": {layers_total},\n",
            "    \"bitnet_linear_layers_on_a770_opencl\": {layers_on_a770},\n",
            "    \"bitnet_linear_layers_cpu_fallback\": {layers_cpu_fallback},\n",
            "    \"unsupported_ops\": {unsupported_ops},\n",
            "    \"execution_claim\": {execution_claim}\n",
            "  }},\n",
            "  \"runtime_stats\": {{\n",
            "    \"host_to_device_bytes\": {host_to_device_bytes},\n",
            "    \"device_to_host_bytes\": {device_to_host_bytes},\n",
            "    \"kernel_invocations\": {kernel_invocations}\n",
            "  }},\n",
            "  \"parity\": {{\n",
            "    \"reference_backend\": \"cpu_qk256_reference\",\n",
            "    \"target_backend\": \"intel-arc-a770-opencl\",\n",
            "    \"exact\": true,\n",
            "    \"max_abs_error\": {max_abs_error}\n",
            "  }},\n",
            "  \"not_claims\": {not_claims}\n",
            "}}\n",
        ),
        platform_index = device.platform_index,
        device_index = device.device_index,
        platform_name = json_string(&device.platform_name),
        runtime_device = json_string(&device.runtime_device),
        vendor = json_string(&device.vendor),
        driver_version = json_string(&device.driver_version),
        layers_total = coverage.bitnet_linear_layers_total,
        layers_on_a770 = coverage.bitnet_linear_layers_on_a770_opencl,
        layers_cpu_fallback = coverage.bitnet_linear_layers_cpu_fallback,
        unsupported_ops = unsupported_ops,
        execution_claim = json_string(coverage.execution_claim),
        host_to_device_bytes = stats.host_to_device_bytes,
        device_to_host_bytes = stats.device_to_host_bytes,
        kernel_invocations = stats.kernel_invocations,
        max_abs_error = max_abs_error,
        not_claims = not_claims,
    );
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, receipt)?;
    Ok(())
}

#[cfg(feature = "opencl")]
fn workspace_relative_path(path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        return path;
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join(path)
}

#[cfg(feature = "opencl")]
fn json_string_array<'a>(values: impl IntoIterator<Item = &'a str>) -> String {
    let values = values.into_iter().map(json_string).collect::<Vec<_>>();
    format!("[{}]", values.join(", "))
}

#[cfg(feature = "opencl")]
fn json_string(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len() + 2);
    escaped.push('"');
    for ch in value.chars() {
        match ch {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            ch if ch.is_control() => escaped.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => escaped.push(ch),
        }
    }
    escaped.push('"');
    escaped
}

#[test]
fn reset_clears_cuda_runtime_accounting_counters() {
    let _guard = ENV_LOCK.lock().unwrap();
    clear_backend_env();
    reset_qk256_dispatch_coverage();

    let stats = qk256_cuda_runtime_stats();

    assert_eq!(stats.host_to_device_bytes, 0);
    assert_eq!(stats.device_to_host_bytes, 0);
    assert_eq!(stats.kernel_time_ms, None);
    assert_eq!(stats.kernel_time_samples, 0);
}

#[cfg(not(feature = "cuda"))]
#[test]
fn strict_cuda_request_rejects_cpu_qk256_fallback_without_cuda_feature() {
    let _guard = ENV_LOCK.lock().unwrap();
    clear_backend_env();
    reset_qk256_dispatch_coverage();
    unsafe {
        std::env::set_var("BITNET_SELECTED_BACKEND", "nvidia-rtx-5070-ti-cuda");
        std::env::set_var("BITNET_STRICT_MODE", "1");
    }
    let device = Device::Cpu;
    let input = Tensor::ones(&[1, 1, 256], DType::F32, &device).unwrap();
    let qk256 = qk256_tensor(2, 256, &device);

    let result = forward_qk256(&input, &qk256, "layers.0.attention.q_proj.weight.qk256_qs");

    assert!(result.is_err());
    let coverage = qk256_dispatch_coverage();
    assert_eq!(coverage.bitnet_linear_layers_total, 1);
    assert_eq!(coverage.bitnet_linear_layers_on_cuda, 0);
    assert_eq!(coverage.bitnet_linear_layers_cpu_fallback, 0);
    clear_backend_env();
}

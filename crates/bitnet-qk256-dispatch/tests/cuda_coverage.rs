use bitnet_qk256_dispatch::{
    forward_qk256, forward_qk256_with_scale, qk256_a770_opencl_runtime_stats,
    qk256_cuda_runtime_stats, qk256_dispatch_coverage, record_bitnet_linear_cpu_fallback,
    record_bitnet_linear_unsupported, reset_qk256_dispatch_coverage,
};
use candle_core::{DType, Device, Tensor};
use std::error::Error;
use std::sync::Mutex;

static ENV_LOCK: Mutex<()> = Mutex::new(());
type TestResult = Result<(), Box<dyn Error>>;

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

    assert_eq!(a770_output.to_vec3::<f32>()?, cpu_output.to_vec3::<f32>()?);
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
    clear_backend_env();
    Ok(())
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

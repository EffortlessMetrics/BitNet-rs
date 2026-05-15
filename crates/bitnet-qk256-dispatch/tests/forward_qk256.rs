use bitnet_qk256_dispatch::{
    Qk256DispatchBackend, forward_qk256, forward_qk256_scaled_with_backend,
    forward_qk256_with_backend, qk256_dispatch_status,
};
use bitnet_quantization::i2s_qk256::{
    QK256_BLOCK, gemv_qk256_row_activation_quantized_reference, pack_qk256_codes_for_cols,
};
use candle_core::{Device, Tensor};

#[test]
fn forward_qk256_supports_rank2_input() {
    let device = Device::Cpu;
    let input = Tensor::from_vec(vec![1.0f32; 256], (1, 256), &device).unwrap();
    let qk = Tensor::from_vec(vec![0xAAu8; 64], (1, 64), &device).unwrap();

    let out = forward_qk256(&input, &qk, "layers.0.attention.q_proj.weight.qk256_qs").unwrap();
    assert_eq!(out.dims(), &[1, 1]);

    let out_vals = out.to_vec2::<f32>().unwrap();
    assert!((out_vals[0][0] - 256.0).abs() < 1e-4);
}

#[test]
fn forward_qk256_explicit_cpu_backend_matches_default() {
    let device = Device::Cpu;
    let input = Tensor::from_vec(vec![1.0f32; 256], (1, 256), &device).unwrap();
    let qk = Tensor::from_vec(vec![0xAAu8; 64], (1, 64), &device).unwrap();

    let default_out =
        forward_qk256(&input, &qk, "layers.0.attention.q_proj.weight.qk256_qs").unwrap();
    let explicit_out = forward_qk256_with_backend(
        &input,
        &qk,
        "layers.0.attention.q_proj.weight.qk256_qs",
        Qk256DispatchBackend::Cpu,
    )
    .unwrap();

    assert_eq!(default_out.to_vec2::<f32>().unwrap(), explicit_out.to_vec2::<f32>().unwrap());
}

#[test]
fn forward_qk256_supports_rank3_input() {
    let device = Device::Cpu;
    let input = Tensor::from_vec(vec![1.0f32; 2 * 2 * 256], (2, 2, 256), &device).unwrap();
    let qk = Tensor::from_vec(vec![0xAAu8; 64], (1, 64), &device).unwrap();

    let out = forward_qk256(&input, &qk, "layers.0.feed_forward.up_proj.weight.qk256_qs").unwrap();
    assert_eq!(out.dims(), &[2, 2, 1]);

    let out_vals = out.to_vec3::<f32>().unwrap();
    for batch in out_vals {
        for token in batch {
            assert!((token[0] - 256.0).abs() < 1e-4);
        }
    }
}

#[test]
fn forward_qk256_rank3_preserves_varied_token_rows() {
    let device = Device::Cpu;
    let mut input_rows = Vec::with_capacity(2 * 2 * 256);
    for value in [1.0f32, 2.0, -1.0, 0.5] {
        input_rows.extend(std::iter::repeat_n(value, 256));
    }
    let input = Tensor::from_vec(input_rows, (2, 2, 256), &device).unwrap();
    let qk = Tensor::from_vec(vec![0xAAu8; 64], (1, 64), &device).unwrap();

    let out = forward_qk256(&input, &qk, "layers.0.feed_forward.up_proj.weight.qk256_qs").unwrap();
    assert_eq!(out.dims(), &[2, 2, 1]);

    let out_vals = out.to_vec3::<f32>().unwrap();
    let expected = [[[256.0f32], [512.0]], [[-256.0], [128.0]]];
    for (batch_idx, batch) in out_vals.iter().enumerate() {
        for (token_idx, token) in batch.iter().enumerate() {
            assert!(
                (token[0] - expected[batch_idx][token_idx][0]).abs() < 1e-4,
                "rank3 QK256 row mismatch at batch {batch_idx}, token {token_idx}: expected {}, actual {}",
                expected[batch_idx][token_idx][0],
                token[0]
            );
        }
    }
}

#[test]
fn forward_qk256_cpu_uses_reference_activation_quantization() {
    let device = Device::Cpu;
    let cols = QK256_BLOCK;
    let scale = 0.375f32;
    let codes: Vec<u8> = (0..cols).map(|i| (i % 4) as u8).collect();
    let qk_bytes = pack_qk256_codes_for_cols(&codes, cols);
    let input_values: Vec<f32> = (0..cols).map(|i| ((i % 13) as f32 - 6.0) / 7.0).collect();

    let input = Tensor::from_vec(input_values.clone(), (1, cols), &device).unwrap();
    let qk = Tensor::from_vec(qk_bytes.clone(), (1, qk_bytes.len()), &device).unwrap();
    let out = forward_qk256_scaled_with_backend(
        &input,
        &qk,
        "layers.0.attention.q_proj.weight.qk256_qs",
        Qk256DispatchBackend::Cpu,
        scale,
    )
    .unwrap();

    let expected =
        gemv_qk256_row_activation_quantized_reference(&qk_bytes, &input_values, cols, scale);
    let out_vals = out.to_vec2::<f32>().unwrap();
    assert!((out_vals[0][0] - expected).abs() < 1e-5);
}

#[test]
fn forward_qk256_rejects_dimension_mismatch() {
    let device = Device::Cpu;
    let input = Tensor::from_vec(vec![1.0f32; 128], (1, 128), &device).unwrap();
    let qk = Tensor::from_vec(vec![0xAAu8; 64], (1, 64), &device).unwrap();

    let err = forward_qk256(&input, &qk, "layers.1.attention.k_proj.weight.qk256_qs").unwrap_err();
    assert!(err.to_string().contains("dimension mismatch"));
}

#[cfg(not(feature = "opencl"))]
#[test]
fn forward_qk256_opencl_backend_requires_opencl_feature() {
    let device = Device::Cpu;
    let input = Tensor::from_vec(vec![1.0f32; 256], (1, 256), &device).unwrap();
    let qk = Tensor::from_vec(vec![0xAAu8; 64], (1, 64), &device).unwrap();

    let err = forward_qk256_with_backend(
        &input,
        &qk,
        "layers.0.attention.q_proj.weight.qk256_qs",
        Qk256DispatchBackend::OpenCl,
    )
    .unwrap_err();

    assert!(err.to_string().contains("opencl feature is disabled"));
}

#[test]
fn qk256_dispatch_status_keeps_opencl_non_claiming() {
    let status = qk256_dispatch_status();

    assert_eq!(status.compiled_opencl, cfg!(feature = "opencl"));
    assert_eq!(status.compiled_oneapi, cfg!(feature = "oneapi"));
    assert_eq!(status.opencl_launcher_available, cfg!(feature = "opencl"));
    assert!(!status.accelerator_claimable);
    assert!(status.not_claims.contains(&"a770_qk256_opencl_execution"));

    if cfg!(feature = "oneapi") {
        assert_eq!(status.runtime_backend, "oneapi_qk256_activation_quantized_diagnostic");
        assert_eq!(status.blocker, Some("oneapi_qk256_semantic_quality_unproven"));
    } else if cfg!(feature = "opencl") {
        assert_eq!(status.runtime_backend, "opencl_qk256_activation_quantized_diagnostic");
        assert_eq!(status.blocker, Some("opencl_qk256_semantic_quality_unproven"));
    } else {
        assert_eq!(status.runtime_backend, "cpu_qk256_activation_quantized_reference");
        assert_eq!(status.blocker, Some("cpu_qk256_semantic_quality_unproven"));
    }
}

#[cfg(feature = "opencl")]
#[test]
fn qk256_opencl_source_matches_microsoft_bitnet_mapping() {
    use bitnet_qk256_dispatch::{QK256_OPENCL_KERNEL_NAME, QK256_OPENCL_KERNEL_SRC};

    assert_eq!(QK256_OPENCL_KERNEL_NAME, "qk256_gemm_no_scale");
    assert!(QK256_OPENCL_KERNEL_SRC.contains("__kernel void qk256_gemm_no_scale"));
    assert!(QK256_OPENCL_KERNEL_SRC.contains("const uint group128 = col / 128u"));
    assert!(
        QK256_OPENCL_KERNEL_SRC.contains("const uchar packed = row_bytes[(group128 * 32u) + pos]")
    );
    assert!(
        QK256_OPENCL_KERNEL_SRC.contains("const uchar code = (packed >> (6u - (lane * 2u))) & 3u")
    );
    assert!(QK256_OPENCL_KERNEL_SRC.contains("qk256_nearest_int_reference"));
    assert!(QK256_OPENCL_KERNEL_SRC.contains("const float act_scale = 127.0f / max_abs"));
    assert!(QK256_OPENCL_KERNEL_SRC.contains("integer_dot += ((int)code) * q"));
    assert!(QK256_OPENCL_KERNEL_SRC.contains("(integer_dot - act_sum"));
    assert!(QK256_OPENCL_KERNEL_SRC.contains("const float scale"));
}

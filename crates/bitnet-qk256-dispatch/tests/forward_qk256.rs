use bitnet_qk256_dispatch::{
    Qk256DispatchBackend, forward_qk256, forward_qk256_with_backend, qk256_dispatch_status,
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
    assert_eq!(status.runtime_backend, "cpu_qk256_reference");
    assert!(!status.accelerator_claimable);
    assert!(status.not_claims.contains(&"a770_qk256_opencl_execution"));

    if cfg!(feature = "oneapi") {
        assert_eq!(status.blocker, Some("oneapi_qk256_transformer_dispatch_not_wired"));
    } else if cfg!(feature = "opencl") {
        assert_eq!(status.blocker, Some("opencl_qk256_transformer_dispatch_not_wired"));
    } else {
        assert_eq!(status.blocker, Some("cpu_qk256_dispatch_only"));
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
    assert!(QK256_OPENCL_KERNEL_SRC.contains("const float w = ((float)code) - 1.0f"));
    assert!(QK256_OPENCL_KERNEL_SRC.contains("acc * scale"));
    assert!(QK256_OPENCL_KERNEL_SRC.contains("const float scale"));
}

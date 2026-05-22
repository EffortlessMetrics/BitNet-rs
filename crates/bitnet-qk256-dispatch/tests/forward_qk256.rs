use bitnet_common::{BitNetError, Result};
use bitnet_qk256_dispatch::{forward_qk256, forward_qk256_with_scale, qk256_dispatch_status};
use candle_core::{Device, Tensor};

#[test]
fn forward_qk256_supports_rank2_input() -> Result<()> {
    let device = Device::Cpu;
    let input = Tensor::from_vec(vec![1.0f32; 256], (1, 256), &device)?;
    let qk = Tensor::from_vec(vec![0xAAu8; 64], (1, 64), &device)?;

    let out = forward_qk256(&input, &qk, "layers.0.attention.q_proj.weight.qk256_qs")?;
    assert_eq!(out.dims(), &[1, 1]);

    let out_vals = out.to_vec2::<f32>()?;
    assert!((out_vals[0][0] - 256.0).abs() < 1e-4);
    Ok(())
}

#[test]
fn forward_qk256_supports_rank3_input() -> Result<()> {
    let device = Device::Cpu;
    let input = Tensor::from_vec(vec![1.0f32; 2 * 2 * 256], (2, 2, 256), &device)?;
    let qk = Tensor::from_vec(vec![0xAAu8; 64], (1, 64), &device)?;

    let out = forward_qk256(&input, &qk, "layers.0.feed_forward.up_proj.weight.qk256_qs")?;
    assert_eq!(out.dims(), &[2, 2, 1]);

    let out_vals = out.to_vec3::<f32>()?;
    for batch in out_vals {
        for token in batch {
            assert!((token[0] - 256.0).abs() < 1e-4);
        }
    }
    Ok(())
}

#[test]
fn forward_qk256_rank3_preserves_varied_token_rows() -> Result<()> {
    let device = Device::Cpu;
    let mut input_rows = Vec::with_capacity(2 * 2 * 256);
    for value in [1.0f32, 2.0, -1.0, 0.5] {
        input_rows.extend(std::iter::repeat(value).take(256));
    }
    let input = Tensor::from_vec(input_rows, (2, 2, 256), &device)?;
    let qk = Tensor::from_vec(vec![0xAAu8; 64], (1, 64), &device)?;

    let out = forward_qk256(&input, &qk, "layers.0.feed_forward.up_proj.weight.qk256_qs")?;
    assert_eq!(out.dims(), &[2, 2, 1]);

    let out_vals = out.to_vec3::<f32>()?;
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
    Ok(())
}

#[test]
fn forward_qk256_with_scale_uses_bitnet_i8s_activation_path() -> Result<()> {
    let device = Device::Cpu;
    let input = Tensor::from_vec(vec![1.0f32; 256], (1, 256), &device)?;
    let qk = Tensor::from_vec(vec![0xAAu8; 64], (1, 64), &device)?;

    let out = forward_qk256_with_scale(
        &input,
        &qk,
        "layers.0.attention.q_proj.weight.qk256_qs",
        Some(0.5),
    )?;
    assert_eq!(out.dims(), &[1, 1]);

    let out_vals = out.to_vec2::<f32>()?;
    assert!((out_vals[0][0] - 128.0).abs() < 1e-4);
    Ok(())
}

#[test]
fn forward_qk256_rejects_dimension_mismatch() -> Result<()> {
    let device = Device::Cpu;
    let input = Tensor::from_vec(vec![1.0f32; 128], (1, 128), &device)?;
    let qk = Tensor::from_vec(vec![0xAAu8; 64], (1, 64), &device)?;

    let err = match forward_qk256(&input, &qk, "layers.1.attention.k_proj.weight.qk256_qs") {
        Ok(_) => {
            return Err(BitNetError::Validation("dimension mismatch unexpectedly passed".into()));
        }
        Err(err) => err,
    };
    assert!(err.to_string().contains("dimension mismatch"));
    Ok(())
}

#[test]
fn cpu_hot_path_audit_distinguishes_no_scale_and_scaled_paths() -> Result<()> {
    bitnet_qk256_dispatch::reset_qk256_dispatch_coverage();
    let device = Device::Cpu;
    let input = Tensor::from_vec(vec![1.0f32; 256], (1, 256), &device)?;
    let qk = Tensor::from_vec(vec![0xAAu8; 64], (1, 64), &device)?;

    forward_qk256(&input, &qk, "layers.0.attention.q_proj.weight.qk256_qs")?;
    forward_qk256_with_scale(&input, &qk, "layers.0.attention.q_proj.weight.qk256_qs", Some(0.5))?;

    let mixed = bitnet_qk256_dispatch::qk256_cpu_hot_path_counters();
    assert!(mixed.qk256_f32_scalar_gemv_invocations + mixed.qk256_f32_avx2_gemv_invocations >= 1);
    assert!(mixed.qk256_i8s_scaled_scalar_invocations >= 1);
    assert_eq!(mixed.qk256_i8s_scaled_avx2_invocations, 0);
    assert_eq!(mixed.qk256_execution_path, "mixed_scaled_and_no_scale");
    assert_eq!(mixed.selected_kernel.as_deref(), Some("mixed-qk256-cpu-hot-paths"));
    assert!(mixed.qk256_flat_bytes_extracted_count >= 2);
    assert!(mixed.input_rows_materialized_count >= 2);
    assert!(mixed.output_rows_allocated_count >= 2);
    Ok(())
}

#[test]
fn qk256_dispatch_status_keeps_opencl_non_claiming() {
    let status = qk256_dispatch_status();

    assert_eq!(status.compiled_opencl, cfg!(feature = "opencl"));
    assert_eq!(status.compiled_oneapi, cfg!(feature = "oneapi"));
    if cfg!(all(feature = "opencl", not(feature = "oneapi"))) {
        assert_eq!(status.runtime_backend, "a770_opencl_qk256_i8s_scaled_candidate");
    } else {
        assert_eq!(status.runtime_backend, "cpu_qk256_reference");
    }
    assert!(!status.accelerator_claimable);
    assert!(status.not_claims.contains(&"a770_qk256_opencl_claim_grade_execution"));
    assert!(status.not_claims.contains(&"a770_qk256_opencl_performance"));
    assert!(status.not_claims.contains(&"activation_quantization_residency"));

    for not_claim in [
        "selected_attention_residency",
        "resident_kv_decode",
        "attention_scores_residency",
        "softmax_residency",
        "attention_value_mix_residency",
        "full_support_op_residency",
        "full_device_residency",
        "completion",
    ] {
        assert!(
            status.not_claims.contains(&not_claim),
            "qk256 dispatch status must preserve A770 not-claim `{not_claim}`"
        );
    }

    if cfg!(feature = "oneapi") {
        assert_eq!(status.blocker, Some("oneapi_qk256_runtime_not_wired"));
    } else if cfg!(feature = "opencl") {
        assert_eq!(
            status.blocker,
            Some("activation_quantization_cpu_resident_and_partial_qk256_only")
        );
    } else {
        assert_eq!(status.blocker, Some("cpu_qk256_dispatch_only"));
    }
}

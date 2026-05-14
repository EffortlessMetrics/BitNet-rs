#![cfg(feature = "opencl")]

use bitnet_qk256_dispatch::{Qk256DispatchBackend, forward_qk256, forward_qk256_with_backend};
use candle_core::{Device, Tensor};

#[test]
#[ignore = "requires an Intel OpenCL GPU runtime; run manually on the A770 proof station"]
fn opencl_qk256_matches_cpu_reference_for_known_rows() {
    let device = Device::Cpu;
    let input = Tensor::from_vec(vec![1.0f32; 256], (1, 256), &device).unwrap();

    let mut qk_bytes = Vec::with_capacity(2 * 64);
    qk_bytes.extend_from_slice(&[0xAAu8; 64]);
    qk_bytes.extend_from_slice(&[0x55u8; 64]);
    let qk = Tensor::from_vec(qk_bytes, (2, 64), &device).unwrap();

    let weight_name = "layers.0.attention.q_proj.weight.qk256_qs";
    let cpu = forward_qk256(&input, &qk, weight_name).unwrap();
    let opencl =
        forward_qk256_with_backend(&input, &qk, weight_name, Qk256DispatchBackend::OpenCl).unwrap();

    let cpu_vals = cpu.to_vec2::<f32>().unwrap();
    let opencl_vals = opencl.to_vec2::<f32>().unwrap();

    assert_eq!(cpu_vals.len(), opencl_vals.len());
    assert_eq!(cpu_vals[0].len(), opencl_vals[0].len());
    for (expected_row, actual_row) in cpu_vals.iter().zip(opencl_vals.iter()) {
        for (&expected, &actual) in expected_row.iter().zip(actual_row.iter()) {
            assert!(
                (expected - actual).abs() <= 1e-4,
                "OpenCL QK256 mismatch: expected {expected}, actual {actual}"
            );
        }
    }
}

#[test]
#[ignore = "requires an Intel OpenCL GPU runtime; run manually on the A770 proof station"]
fn opencl_qk256_rank3_matches_cpu_reference_for_varied_rows() {
    let device = Device::Cpu;
    let mut input_rows = Vec::with_capacity(2 * 2 * 256);
    for value in [1.0f32, 2.0, -1.0, 0.5] {
        input_rows.extend(std::iter::repeat_n(value, 256));
    }
    let input = Tensor::from_vec(input_rows, (2, 2, 256), &device).unwrap();

    let mut qk_bytes = Vec::with_capacity(2 * 64);
    qk_bytes.extend_from_slice(&[0xAAu8; 64]);
    qk_bytes.extend_from_slice(&[0x55u8; 64]);
    let qk = Tensor::from_vec(qk_bytes, (2, 64), &device).unwrap();

    let weight_name = "layers.0.attention.q_proj.weight.qk256_qs";
    let cpu = forward_qk256(&input, &qk, weight_name).unwrap();
    let opencl =
        forward_qk256_with_backend(&input, &qk, weight_name, Qk256DispatchBackend::OpenCl).unwrap();

    let cpu_vals = cpu.to_vec3::<f32>().unwrap();
    let opencl_vals = opencl.to_vec3::<f32>().unwrap();

    assert_eq!(cpu_vals.len(), opencl_vals.len());
    for (batch_idx, (expected_batch, actual_batch)) in
        cpu_vals.iter().zip(opencl_vals.iter()).enumerate()
    {
        assert_eq!(expected_batch.len(), actual_batch.len());
        for (token_idx, (expected_row, actual_row)) in
            expected_batch.iter().zip(actual_batch.iter()).enumerate()
        {
            assert_eq!(expected_row.len(), actual_row.len());
            for (col_idx, (&expected, &actual)) in
                expected_row.iter().zip(actual_row.iter()).enumerate()
            {
                assert!(
                    (expected - actual).abs() <= 1e-4,
                    "OpenCL rank3 QK256 mismatch at batch {batch_idx}, token {token_idx}, col {col_idx}: expected {expected}, actual {actual}"
                );
            }
        }
    }
}

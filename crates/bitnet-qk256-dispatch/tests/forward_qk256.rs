use bitnet_qk256_dispatch::forward_qk256;
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
fn forward_qk256_rejects_dimension_mismatch() {
    let device = Device::Cpu;
    let input = Tensor::from_vec(vec![1.0f32; 128], (1, 128), &device).unwrap();
    let qk = Tensor::from_vec(vec![0xAAu8; 64], (1, 64), &device).unwrap();

    let err = forward_qk256(&input, &qk, "layers.1.attention.k_proj.weight.qk256_qs").unwrap_err();
    assert!(err.to_string().contains("dimension mismatch"));
}

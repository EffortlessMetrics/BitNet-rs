use std::collections::HashMap;

use bitnet_models::weight_mapper::remap_gguf_weights_with_options;
use candle_core::{Device, Tensor};

#[test]
fn remap_normalizes_exporter_drift_for_attention_sub_norm() {
    let device = Device::Cpu;
    let tensor = Tensor::zeros((2, 2), candle_core::DType::F32, &device).expect("tensor");
    let mut tensors = HashMap::new();
    tensors.insert("blk.0.attention_sub_norm.weight".to_string(), tensor);

    let mapped = remap_gguf_weights_with_options(&tensors, false).expect("remap succeeds");

    assert!(mapped.contains_key("layers.0.attention.sub_layernorm.weight"));
}

#[test]
fn remap_normalizes_exporter_drift_for_mlp_sub_layernorm() {
    let device = Device::Cpu;
    let tensor = Tensor::zeros((2, 2), candle_core::DType::F32, &device).expect("tensor");
    let mut tensors = HashMap::new();
    tensors.insert("blk.0.mlp_sub_layernorm.weight".to_string(), tensor);

    let mapped = remap_gguf_weights_with_options(&tensors, false).expect("remap succeeds");

    assert!(mapped.contains_key("layers.0.feed_forward.sub_layernorm.weight"));
}

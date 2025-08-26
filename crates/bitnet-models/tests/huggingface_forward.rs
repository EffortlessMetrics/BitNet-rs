//! Integration test for loading a minimal HuggingFace model and running a forward pass.

use std::collections::BTreeMap;
use std::fs;
use tempfile::TempDir;

use bitnet_common::Device as CommonDevice;
use bitnet_common::Tensor;
use bitnet_models::{
    formats::huggingface::HuggingFaceLoader,
    loader::{FormatLoader, LoadConfig},
    transformer::KVCache,
};
use candle_core::Device as CandleDevice;
use safetensors::tensor::{Dtype, TensorView};

#[test]
fn test_load_huggingface_and_forward() {
    // Build a minimal HuggingFace model directory.
    let dir = TempDir::new().unwrap();
    let model_dir = dir.path();

    // Write config.json
    let config = serde_json::json!({
        "_name_or_path": "test",
        "model_type": "bitnet",
        "vocab_size": 16,
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "intermediate_size": 8,
        "max_position_embeddings": 8,
    });
    fs::write(model_dir.join("config.json"), serde_json::to_vec(&config).unwrap()).unwrap();

    // Create minimal weights with embeddings and lm_head
    let vocab_size = 16;
    let hidden = 4;
    let embed_data = vec![0f32; vocab_size * hidden];
    let lm_head_data = vec![0f32; vocab_size * hidden];

    let mut tensors = BTreeMap::new();
    let embed_view =
        TensorView::new(Dtype::F32, vec![vocab_size, hidden], bytemuck::cast_slice(&embed_data))
            .unwrap();
    tensors.insert("model.embed_tokens.weight", embed_view);
    let lm_head_view =
        TensorView::new(Dtype::F32, vec![vocab_size, hidden], bytemuck::cast_slice(&lm_head_data))
            .unwrap();
    tensors.insert("lm_head.weight", lm_head_view);

    let data = safetensors::serialize(tensors.iter().map(|(k, v)| (*k, v)), None).unwrap();
    let shard_name = "model-00001-of-00001.safetensors";
    fs::write(model_dir.join(shard_name), &data).unwrap();

    // Write index.json mapping tensors to shard
    let mut weight_map = serde_json::Map::new();
    weight_map.insert(
        "model.embed_tokens.weight".to_string(),
        serde_json::Value::String(shard_name.to_string()),
    );
    weight_map
        .insert("lm_head.weight".to_string(), serde_json::Value::String(shard_name.to_string()));
    let index = serde_json::json!({
        "metadata": { "total_size": data.len() },
        "weight_map": weight_map,
    });
    fs::write(model_dir.join("model.safetensors.index.json"), serde_json::to_vec(&index).unwrap())
        .unwrap();

    // Load model
    let loader = HuggingFaceLoader;
    let device = CommonDevice::Cpu;
    let model = loader.load(model_dir, &device, &LoadConfig::default()).expect("load model");

    // Embed and run forward
    let tokens = vec![0u32, 1, 2, 3];
    let embedded = model.embed(&tokens).unwrap();
    let candle_device = CandleDevice::Cpu;
    let mut cache = KVCache::new(model.config(), 1, &candle_device).unwrap();
    let hidden = model.forward(&embedded, &mut cache).unwrap();

    assert_eq!(hidden.shape()[0], 1); // batch
    assert_eq!(hidden.shape()[1], tokens.len());
}

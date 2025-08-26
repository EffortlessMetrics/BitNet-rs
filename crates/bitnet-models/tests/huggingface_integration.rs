use bitnet_common::Device;
use bitnet_models::ModelLoader;
use std::fs::{self, File};
use std::io::Write;
use tempfile::TempDir;

fn write_fake_weights(path: &std::path::Path) {
    let mut file = File::create(path).unwrap();
    // Add padding spaces so that header length + 8 is divisible by 4 for alignment
    let header = r#"{"token_embd.weight":{"dtype":"F32","shape":[2,4],"data_offsets":[0,32]}}   "#;
    let header_len = header.len() as u64;
    file.write_all(&header_len.to_le_bytes()).unwrap();
    file.write_all(header.as_bytes()).unwrap();
    let data = [0f32; 8];
    for f in &data {
        file.write_all(&f.to_le_bytes()).unwrap();
    }
    file.flush().unwrap();
}

#[test]
fn test_huggingface_directory_loading() {
    let tmp = TempDir::new().unwrap();
    let dir = tmp.path();
    let config = r#"{
        "model_type": "bitnet",
        "vocab_size": 2,
        "max_position_embeddings": 16,
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "intermediate_size": 8
    }"#;
    fs::write(dir.join("config.json"), config).unwrap();
    write_fake_weights(&dir.join("pytorch_model.bin"));

    let loader = ModelLoader::new(Device::Cpu);
    let metadata = loader.extract_metadata(dir).unwrap();
    assert_eq!(metadata.vocab_size, 2);
    assert_eq!(metadata.context_length, 16);

    let model = loader.load(dir).unwrap();
    assert_eq!(model.config().model.vocab_size, 2);
}

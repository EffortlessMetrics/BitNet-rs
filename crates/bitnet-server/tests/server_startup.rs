use bitnet_server::{BitNetServer, ServerConfig};
use tempfile::TempDir;
use tokio::fs;

fn write_string(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

fn create_minimal_gguf() -> Vec<u8> {
    let mut data = Vec::new();
    // header
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&2u64.to_le_bytes()); // two tensors
    data.extend_from_slice(&2u64.to_le_bytes()); // two metadata entries
    data.extend_from_slice(&32u32.to_le_bytes()); // alignment
    data.extend_from_slice(&0u64.to_le_bytes()); // data_offset placeholder

    // metadata: general.architecture = "bitnet", general.name = "test"
    write_string(&mut data, "general.architecture");
    data.extend_from_slice(&8u32.to_le_bytes()); // string type
    write_string(&mut data, "bitnet");
    write_string(&mut data, "general.name");
    data.extend_from_slice(&8u32.to_le_bytes());
    write_string(&mut data, "test");

    // tensor infos
    write_string(&mut data, "token_embd.weight");
    data.extend_from_slice(&2u32.to_le_bytes()); // n_dims
    data.extend_from_slice(&1u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes()); // F32
    data.extend_from_slice(&0u64.to_le_bytes()); // offset 0

    write_string(&mut data, "output.weight");
    data.extend_from_slice(&2u32.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());
    data.extend_from_slice(&4u64.to_le_bytes()); // offset after first tensor

    // compute data_start alignment
    let header_len = data.len();
    let alignment = 32usize;
    let data_start = ((header_len + alignment - 1) / alignment) * alignment;
    // update data_offset field at position 28
    let data_offset_pos = 4 + 4 + 8 + 8 + 4;
    data[data_offset_pos..data_offset_pos + 8]
        .copy_from_slice(&(data_start as u64).to_le_bytes());
    // pad to data_start
    data.resize(data_start, 0);
    // tensor data: two f32 values
    data.extend_from_slice(&0f32.to_le_bytes());
    data.extend_from_slice(&0f32.to_le_bytes());
    data
}

#[tokio::test]
async fn server_starts_with_real_tokenizer_file() {
    let temp = TempDir::new().unwrap();
    let model_path = temp.path().join("model.gguf");
    let tokenizer_path = temp.path().join("tokenizer.gguf");
    fs::write(&model_path, create_minimal_gguf()).await.unwrap();
    fs::write(&tokenizer_path, create_minimal_gguf()).await.unwrap();

    let config = ServerConfig {
        model_path: Some(model_path.to_string_lossy().into_owned()),
        tokenizer_path: Some(tokenizer_path.to_string_lossy().into_owned()),
        ..Default::default()
    };

    let server = BitNetServer::new(config).await.expect("server should start");
    use axum::{body::Body, http::{Request, StatusCode}};
    use tower::ServiceExt;

    let app = server.create_app();
    let req = Request::builder()
        .method("POST")
        .uri("/inference")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"prompt":"hi"}"#))
        .unwrap();

    let res = app.oneshot(req).await.unwrap();
    assert_ne!(res.status(), StatusCode::OK);
}

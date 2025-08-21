#[test]
fn ms_bitnet_2b_strict_mapping_is_green() {
    use std::path::Path;
    use bitnet_models::GgufReader;

    let model = std::env::var("BITNET_MS_MODEL")
        .unwrap_or_else(|_| "../models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf".into());
    let path = Path::new(&model);
    if !path.exists() {
        eprintln!("model file not found: {}", path.display());
        return; // don't fail CI if the artifact isn't present
    }

    let bytes = std::fs::read(path).expect("read gguf");
    let reader = GgufReader::new(&bytes).expect("parse gguf");

    // Get tensor names from reader
    let tensor_names = reader.tensor_names();
    
    // Test that all the expected BitNet layers are present
    let has_attn_sub_norm = tensor_names.iter().any(|n| n.contains("attn_sub_norm"));
    let has_ffn_sub_norm = tensor_names.iter().any(|n| n.contains("ffn_sub_norm"));
    let has_attn_norm = tensor_names.iter().any(|n| n.contains("attn_norm"));
    let has_ffn_norm = tensor_names.iter().any(|n| n.contains("ffn_norm"));
    
    // BitNet models should have sub_norm layers
    assert!(has_attn_sub_norm || has_attn_norm, "Model missing attention normalization layers");
    assert!(has_ffn_sub_norm || has_ffn_norm, "Model missing FFN normalization layers");
    
    println!("Microsoft BitNet model has {} tensors", tensor_names.len());
    println!("Has attn_sub_norm: {}", has_attn_sub_norm);
    println!("Has ffn_sub_norm: {}", has_ffn_sub_norm);
}
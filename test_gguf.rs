fn main() {
    let path = std::path::Path::new("models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf");
    println!("Testing GGUF at: {}", path.display());
    println!("File exists: {}", path.exists());
    
    if path.exists() {
        match bitnet_models::gguf_min::load_two(path) {
            Ok(tensors) => {
                println!("Success! vocab={}, dim={}", tensors.vocab, tensors.dim);
                println!("tok_embeddings len: {}", tensors.tok_embeddings.len());
                println!("lm_head len: {}", tensors.lm_head.len());
            }
            Err(e) => {
                println!("Error loading GGUF: {:#}", e);
            }
        }
    }
}
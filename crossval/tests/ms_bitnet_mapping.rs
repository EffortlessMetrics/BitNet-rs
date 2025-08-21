#[test]
fn ms_bitnet_names_map_clean() {
    use bitnet_models::{GgufReader, weight_mapper::dry_run_remap_names};
    use std::path::Path;
    
    let model = std::env::var("BITNET_MS_MODEL")
        .unwrap_or_else(|_| "models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf".into());
    let path = Path::new(&model);
    if !path.exists() {
        eprintln!("missing: {}", path.display());
        return;
    }

    let bytes = std::fs::read(path).expect("read gguf");
    let r = GgufReader::new(&bytes).expect("parse");
    let names = r.tensor_names().into_iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let unmapped = dry_run_remap_names(names.clone());

    println!("Total tensors: {}", names.len());
    println!("Unmapped tensors: {}", unmapped.len());
    if !unmapped.is_empty() {
        println!("First unmapped: {:?}", &unmapped[..unmapped.len().min(10)]);
    }
    
    assert!(unmapped.is_empty(), "unmapped: {:?}", &unmapped[..unmapped.len().min(10)]);
}
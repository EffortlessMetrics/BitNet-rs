#[test]
fn ms_bitnet_names_map_clean() {
    use bitnet_crossval::validation::ValidationSuite;
    use std::path::Path;

    let model = std::env::var("BITNET_MS_MODEL")
        .unwrap_or_else(|_| "models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf".into());
    let path = Path::new(&model);
    if !path.exists() {
        eprintln!("missing: {}", path.display());
        return;
    }

    let suite = ValidationSuite::new(model);
    let result = suite.validate_tensor_mapping().expect("tensor mapping");

    println!(
        "Total tensors: {}",
        result.metrics.get("total_tensors").and_then(|v| v.as_u64()).unwrap_or(0)
    );
    println!(
        "Unmapped tensors: {}",
        result.metrics.get("unmapped_count").and_then(|v| v.as_u64()).unwrap_or(0)
    );
    if let Some(unmapped) = result.metrics.get("unmapped_tensors") {
        let list: Vec<String> = serde_json::from_value(unmapped.clone()).unwrap();
        if !list.is_empty() {
            println!("First unmapped: {:?}", &list[..list.len().min(10)]);
        }
        assert!(list.is_empty(), "unmapped: {:?}", &list[..list.len().min(10)]);
    } else {
        assert!(result.passed);
    }
}

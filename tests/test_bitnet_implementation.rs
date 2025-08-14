use bitnet_tests::cross_validation::test_implementation::MockImplementation;
use bitnet_tests::*;
use std::path::PathBuf;

#[tokio::test]
async fn test_bitnet_implementation_abstraction() {
    let mut impl_ = MockImplementation::new("test".to_string(), "1.0".to_string());

    // Test basic properties
    assert_eq!(impl_.implementation_name(), "test");
    assert_eq!(impl_.implementation_version(), "1.0");
    assert!(impl_.is_available().await);
    assert!(!impl_.is_model_loaded());

    // Test model loading
    let model_path = PathBuf::from("/tmp/test_model.gguf");
    impl_.load_model(&model_path).await.unwrap();
    assert!(impl_.is_model_loaded());

    let model_info = impl_.get_model_info().unwrap();
    assert_eq!(model_info.name, "test_model");

    // Test tokenization
    let text = "hello world";
    let tokens = impl_.tokenize(text).await.unwrap();
    let detokenized = impl_.detokenize(&tokens).await.unwrap();
    assert_eq!(detokenized, text);

    // Test inference
    let config = InferenceConfig::default();
    let result = impl_.inference(&tokens, &config).await.unwrap();
    assert!(!result.text.is_empty());
    assert!(result.tokens.len() > tokens.len());

    // Test cleanup
    impl_.cleanup().await.unwrap();
}

#[tokio::test]
async fn test_implementation_registry() {
    use bitnet_tests::cross_validation::test_implementation::MockImplementationFactory;

    let mut registry = ImplementationRegistry::new();

    // Register implementations
    registry.register(
        "rust".to_string(),
        MockImplementationFactory::new("BitNet.rs".to_string(), "1.0".to_string(), true),
    );
    registry.register(
        "cpp".to_string(),
        MockImplementationFactory::new("BitNet.cpp".to_string(), "1.0".to_string(), true),
    );

    // Test discovery
    let available = registry.discover_implementations().await.unwrap();
    assert_eq!(available.len(), 2);
    assert!(available.contains(&"rust".to_string()));
    assert!(available.contains(&"cpp".to_string()));

    // Test creation
    let rust_impl = registry.create_implementation("rust").await.unwrap();
    assert_eq!(rust_impl.implementation_name(), "BitNet.rs");

    let cpp_impl = registry.create_implementation("cpp").await.unwrap();
    assert_eq!(cpp_impl.implementation_name(), "BitNet.cpp");
}

#[tokio::test]
async fn test_resource_manager() {
    let limits = ResourceLimits {
        max_memory: Some(10 * 1024 * 1024), // 10MB
        max_implementations: Some(2),
        max_models_per_implementation: Some(1),
    };
    let mut manager = ResourceManager::new(limits);

    // Add implementations
    let impl1 = Box::new(MockImplementation::new("test1".to_string(), "1.0".to_string()));
    let impl2 = Box::new(MockImplementation::new("test2".to_string(), "1.0".to_string()));

    manager.add_implementation("impl1".to_string(), impl1).await.unwrap();
    manager.add_implementation("impl2".to_string(), impl2).await.unwrap();

    // Test resource tracking
    let summary = manager.get_resource_summary();
    assert_eq!(summary.total_implementations, 2);
    assert_eq!(summary.total_memory, 2 * 1024 * 1024); // 2MB total

    // Test cleanup
    manager.cleanup_all().await.unwrap();
    assert_eq!(manager.get_total_memory_usage(), 0);
}

#[test]
fn test_performance_metrics() {
    let mut metrics = PerformanceMetrics::new();

    // Test custom metrics
    metrics.add_custom_metric("accuracy".to_string(), 0.95);
    assert_eq!(metrics.custom_metrics.get("accuracy"), Some(&0.95));

    // Test calculations
    metrics.inference_time = std::time::Duration::from_secs(2);
    metrics.calculate_tokens_per_second(200);
    assert_eq!(metrics.tokens_per_second, 100.0);

    metrics.peak_memory = 1024;
    metrics.calculate_memory_efficiency(2048);
    assert_eq!(metrics.memory_efficiency, 2.0);
}

#[test]
fn test_inference_config() {
    let config = InferenceConfig::default();
    assert_eq!(config.max_tokens, 100);
    assert_eq!(config.temperature, 0.7);
    assert_eq!(config.top_p, 0.9);
    assert!(config.top_k.is_none());
    assert_eq!(config.repetition_penalty, 1.0);
    assert!(config.stop_tokens.contains(&"</s>".to_string()));
}

#[test]
fn test_model_format() {
    use serde_json;

    let format = ModelFormat::GGUF;
    let serialized = serde_json::to_string(&format).unwrap();
    let deserialized: ModelFormat = serde_json::from_str(&serialized).unwrap();

    match deserialized {
        ModelFormat::GGUF => assert!(true),
        _ => panic!("Deserialization failed"),
    }
}

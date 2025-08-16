use super::implementation::*;
use crate::errors::ImplementationResult;
use async_trait::async_trait;
use std::path::Path;
use std::time::Duration;

/// Mock implementation for testing
pub struct MockImplementation {
    name: String,
    version: String,
    is_available: bool,
    model_loaded: bool,
    metrics: PerformanceMetrics,
}

impl MockImplementation {
    pub fn new(name: String, version: String) -> Self {
        Self {
            name,
            version,
            is_available: true,
            model_loaded: false,
            metrics: PerformanceMetrics::new(),
        }
    }

    pub fn set_available(&mut self, available: bool) {
        self.is_available = available;
    }
}

#[async_trait]
impl BitNetImplementation for MockImplementation {
    fn implementation_name(&self) -> &str {
        &self.name
    }

    fn implementation_version(&self) -> &str {
        &self.version
    }

    async fn is_available(&self) -> bool {
        self.is_available
    }

    async fn initialize(&mut self, _config: Option<&str>) -> ImplementationResult<()> {
        Ok(())
    }

    async fn load_model(&mut self, _model_path: &Path) -> ImplementationResult<()> {
        self.model_loaded = true;
        self.metrics.model_load_time = Duration::from_millis(100);
        Ok(())
    }

    async fn unload_model(&mut self) -> ImplementationResult<()> {
        self.model_loaded = false;
        Ok(())
    }

    fn is_model_loaded(&self) -> bool {
        self.model_loaded
    }

    fn get_model_info(&self) -> Option<ModelInfo> {
        if self.model_loaded {
            Some(ModelInfo {
                name: "test_model".to_string(),
                path: "/tmp/test_model.gguf".into(),
                format: ModelFormat::GGUF,
                size_bytes: BYTES_PER_MB,
                parameter_count: Some(1_000_000),
                context_length: Some(2048),
                vocabulary_size: Some(32000),
                architecture: Some("llama".to_string()),
                metadata: std::collections::HashMap::new(),
            })
        } else {
            None
        }
    }

    async fn tokenize(&self, text: &str) -> ImplementationResult<Vec<u32>> {
        // Simple mock tokenization - just convert chars to u32
        Ok(text.chars().map(|c| c as u32).collect())
    }

    async fn detokenize(&self, tokens: &[u32]) -> ImplementationResult<String> {
        // Simple mock detokenization - convert u32 back to chars
        let chars: Vec<char> = tokens.iter().filter_map(|&t| char::from_u32(t)).collect();
        Ok(chars.into_iter().collect())
    }

    async fn inference(
        &self,
        tokens: &[u32],
        _config: &InferenceConfig,
    ) -> ImplementationResult<InferenceResult> {
        let start_time = std::time::Instant::now();

        // Mock inference - just echo the input tokens with some additional ones
        let mut output_tokens = tokens.to_vec();
        output_tokens.extend_from_slice(&[32, 116, 101, 115, 116]); // " test"

        let text = self.detokenize(&output_tokens).await?;
        let duration = start_time.elapsed();

        Ok(InferenceResult {
            tokens: output_tokens.clone(),
            text,
            probabilities: Some(vec![0.9; output_tokens.len()]),
            logits: None,
            duration,
            memory_usage: BYTES_PER_MB, // 1MB
            token_count: output_tokens.len(),
        })
    }

    fn get_metrics(&self) -> PerformanceMetrics {
        self.metrics.clone()
    }

    fn reset_metrics(&mut self) {
        self.metrics = PerformanceMetrics::new();
    }

    fn get_resource_info(&self) -> ResourceInfo {
        ResourceInfo {
            memory_usage: BYTES_PER_MB, // 1MB
            file_handles: 1,
            thread_count: 1,
            gpu_memory: None,
        }
    }

    async fn cleanup(&mut self) -> ImplementationResult<()> {
        self.model_loaded = false;
        Ok(())
    }

    fn get_capabilities(&self) -> ImplementationCapabilities {
        ImplementationCapabilities {
            supports_streaming: false,
            supports_batching: false,
            supports_gpu: false,
            supports_quantization: true,
            max_context_length: Some(2048),
            supported_formats: vec![ModelFormat::GGUF, ModelFormat::SafeTensors],
            custom_capabilities: std::collections::HashMap::new(),
        }
    }
}

/// Mock implementation factory
pub struct MockImplementationFactory {
    name: String,
    version: String,
    available: bool,
}

impl MockImplementationFactory {
    pub fn new(name: String, version: String, available: bool) -> Self {
        Self { name, version, available }
    }
}

#[async_trait]
impl ImplementationFactory for MockImplementationFactory {
    async fn create(&self) -> ImplementationResult<Box<dyn BitNetImplementation>> {
        let mut mock = MockImplementation::new(self.name.clone(), self.version.clone());
        mock.set_available(self.available);
        Ok(Box::new(mock))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[tokio::test]
    async fn test_mock_implementation_basic() {
        let mut impl_ = MockImplementation::new("test".to_string(), "1.0".to_string());

        assert_eq!(impl_.implementation_name(), "test");
        assert_eq!(impl_.implementation_version(), "1.0");
        assert!(impl_.is_available().await);
        assert!(!impl_.is_model_loaded());
        assert!(impl_.get_model_info().is_none());
    }

    #[tokio::test]
    async fn test_mock_implementation_model_loading() {
        let mut impl_ = MockImplementation::new("test".to_string(), "1.0".to_string());
        let model_path = PathBuf::from("/tmp/test_model.gguf");

        // Load model
        impl_.load_model(&model_path).await.unwrap();
        assert!(impl_.is_model_loaded());

        let model_info = impl_.get_model_info().unwrap();
        assert_eq!(model_info.name, "test_model");
        assert_eq!(model_info.format, ModelFormat::GGUF);

        // Unload model
        impl_.unload_model().await.unwrap();
        assert!(!impl_.is_model_loaded());
        assert!(impl_.get_model_info().is_none());
    }

    #[tokio::test]
    async fn test_mock_implementation_tokenization() {
        let impl_ = MockImplementation::new("test".to_string(), "1.0".to_string());

        let text = "hello";
        let tokens = impl_.tokenize(text).await.unwrap();
        let detokenized = impl_.detokenize(&tokens).await.unwrap();

        assert_eq!(detokenized, text);
    }

    #[tokio::test]
    async fn test_mock_implementation_inference() {
        let mut impl_ = MockImplementation::new("test".to_string(), "1.0".to_string());
        let model_path = PathBuf::from("/tmp/test_model.gguf");

        impl_.load_model(&model_path).await.unwrap();

        let tokens = vec![104, 101, 108, 108, 111]; // "hello"
        let config = InferenceConfig::default();

        let result = impl_.inference(&tokens, &config).await.unwrap();

        assert!(result.tokens.len() > tokens.len()); // Should have additional tokens
        assert!(!result.text.is_empty());
        assert!(result.probabilities.is_some());
        assert!(result.duration.as_millis() >= 0);
        assert_eq!(result.memory_usage, BYTES_PER_MB);
    }

    #[tokio::test]
    async fn test_implementation_registry() {
        let mut registry = ImplementationRegistry::new();

        // Register mock implementations
        registry.register(
            "mock1".to_string(),
            MockImplementationFactory::new("Mock1".to_string(), "1.0".to_string(), true),
        );
        registry.register(
            "mock2".to_string(),
            MockImplementationFactory::new("Mock2".to_string(), "2.0".to_string(), false),
        );

        let available = registry.list_available();
        assert_eq!(available.len(), 2);
        assert!(available.contains(&"mock1".to_string()));
        assert!(available.contains(&"mock2".to_string()));

        // Create implementation
        let impl_ = registry.create_implementation("mock1").await.unwrap();
        assert_eq!(impl_.implementation_name(), "Mock1");
        assert_eq!(impl_.implementation_version(), "1.0");

        // Discover available implementations
        let discovered = registry.discover_implementations().await.unwrap();
        assert_eq!(discovered.len(), 1); // Only mock1 is available
        assert!(discovered.contains(&"mock1".to_string()));
    }

    #[tokio::test]
    async fn test_resource_manager() {
        let limits = ResourceLimits {
            max_memory: Some(10 * crate::BYTES_PER_MB), // 10MB
            max_implementations: Some(2),
            max_models_per_implementation: Some(1),
        };
        let mut manager = ResourceManager::new(limits);

        // Add implementation
        let impl1 = Box::new(MockImplementation::new("test1".to_string(), "1.0".to_string()));
        manager.add_implementation("impl1".to_string(), impl1).await.unwrap();

        assert_eq!(manager.get_total_memory_usage(), BYTES_PER_MB); // 1MB

        let summary = manager.get_resource_summary();
        assert_eq!(summary.total_implementations, 1);
        assert_eq!(summary.total_memory, BYTES_PER_MB);
        assert_eq!(summary.active_implementations.len(), 1);

        // Cleanup
        manager.cleanup_all().await.unwrap();
        assert_eq!(manager.get_total_memory_usage(), 0);
    }

    #[tokio::test]
    async fn test_performance_metrics() {
        let mut metrics = PerformanceMetrics::new();

        metrics.add_custom_metric("test_metric".to_string(), 42.0);
        assert_eq!(metrics.custom_metrics.get("test_metric"), Some(&42.0));

        metrics.calculate_tokens_per_second(100);
        assert_eq!(metrics.tokens_per_second, 0.0); // inference_time is 0

        metrics.inference_time = Duration::from_secs(1);
        metrics.calculate_tokens_per_second(100);
        assert_eq!(metrics.tokens_per_second, 100.0);

        metrics.calculate_memory_efficiency(1024);
        assert_eq!(metrics.memory_efficiency, 0.0); // peak_memory is 0

        metrics.peak_memory = 512;
        metrics.calculate_memory_efficiency(1024);
        assert_eq!(metrics.memory_efficiency, 2.0);
    }

    #[test]
    fn test_memory_utils() {
        let memory = utils::get_memory_usage();
        assert!(memory > 0);

        let peak_memory = utils::get_peak_memory_usage();
        assert!(peak_memory >= memory);
    }

    #[tokio::test]
    async fn test_measure_performance() {
        let (result, duration, memory_delta, peak_memory) = utils::measure_performance(async {
            tokio::time::sleep(Duration::from_millis(10)).await;
            42
        })
        .await;

        assert_eq!(result, 42);
        assert!(duration >= Duration::from_millis(10));
        assert!(peak_memory > 0);
    }
}

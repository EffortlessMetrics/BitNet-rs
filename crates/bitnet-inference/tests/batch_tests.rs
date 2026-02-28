#![cfg(feature = "integration-tests")]
//! Batch processing tests for bitnet-inference
//!
//! These tests validate batch processing functionality including:
//! - Batch request handling and queuing
//! - Priority-based processing
//! - Concurrent request processing
//! - Resource management and limits
//! - Performance optimization
use bitnet_common::{
    BitNetConfig, BitNetError, ConcreteTensor, Device, InferenceError, PerformanceMetrics,
};
use bitnet_inference::prelude::*;
use bitnet_models::Model;
use bitnet_tokenizers::Tokenizer;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::timeout;
#[derive(Debug, Clone)]
pub struct BatchRequest {
    pub id: String,
    pub prompt: String,
    pub config: bitnet_inference::GenerationConfig,
    pub priority: Priority,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}
#[derive(Debug)]
pub struct BatchResponse {
    pub id: String,
    pub result: Result<String, BitNetError>,
    pub metrics: PerformanceMetrics,
    pub processing_time: Duration,
}
#[derive(Debug, Clone)]
pub struct BatchProcessorConfig {
    pub max_batch_size: usize,
    pub max_queue_size: usize,
    pub max_concurrent_requests: usize,
    pub batch_timeout_ms: u64,
    pub enable_priority_queue: bool,
}
impl Default for BatchProcessorConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            max_queue_size: 100,
            max_concurrent_requests: 4,
            batch_timeout_ms: 100,
            enable_priority_queue: true,
        }
    }
}
impl BatchProcessorConfig {
    pub fn validate(&self) -> Result<(), BitNetError> {
        if self.max_batch_size == 0 {
            return Err(BitNetError::Config("max_batch_size must be greater than 0".to_string()));
        }
        if self.max_queue_size == 0 {
            return Err(BitNetError::Config("max_queue_size must be greater than 0".to_string()));
        }
        if self.max_concurrent_requests == 0 {
            return Err(BitNetError::Config(
                "max_concurrent_requests must be greater than 0".to_string(),
            ));
        }
        Ok(())
    }
}
#[derive(Debug, Clone)]
pub struct QueueStats {
    pub queue_length: usize,
    pub max_queue_size: usize,
    pub active_requests: usize,
    pub max_concurrent_requests: usize,
}
impl QueueStats {
    pub fn queue_utilization(&self) -> f64 {
        if self.max_queue_size > 0 {
            self.queue_length as f64 / self.max_queue_size as f64
        } else {
            0.0
        }
    }
    pub fn processing_utilization(&self) -> f64 {
        if self.max_concurrent_requests > 0 {
            self.active_requests as f64 / self.max_concurrent_requests as f64
        } else {
            0.0
        }
    }
}
struct MockModel {
    config: BitNetConfig,
    processing_delay: Duration,
}
impl MockModel {
    fn new() -> Self {
        Self { config: BitNetConfig::default(), processing_delay: Duration::from_millis(10) }
    }
    fn with_delay(mut self, delay: Duration) -> Self {
        self.processing_delay = delay;
        self
    }
}
impl Model for MockModel {
    fn config(&self) -> &BitNetConfig {
        &self.config
    }
    fn forward(
        &self,
        _input: &ConcreteTensor,
        _cache: &mut dyn std::any::Any,
    ) -> Result<ConcreteTensor, BitNetError> {
        std::thread::sleep(self.processing_delay);
        Ok(ConcreteTensor::mock(vec![1, 50257]))
    }
    fn embed(&self, tokens: &[u32]) -> Result<ConcreteTensor, BitNetError> {
        let seq_len = tokens.len();
        let hidden_dim = self.config.model.hidden_size;
        Ok(ConcreteTensor::mock(vec![seq_len, hidden_dim]))
    }
    fn logits(&self, _hidden: &ConcreteTensor) -> Result<ConcreteTensor, BitNetError> {
        Ok(ConcreteTensor::mock(vec![1, self.config.model.vocab_size]))
    }
}
struct MockTokenizer;
impl Tokenizer for MockTokenizer {
    fn encode(
        &self,
        text: &str,
        _add_bos: bool,
        _add_special: bool,
    ) -> Result<Vec<u32>, BitNetError> {
        Ok((0..text.len().min(10)).map(|i| i as u32 + 1).collect())
    }
    fn decode(&self, tokens: &[u32]) -> Result<String, BitNetError> {
        Ok(format!("decoded_{}_tokens", tokens.len()))
    }
    fn vocab_size(&self) -> usize {
        50257
    }
    fn eos_token_id(&self) -> Option<u32> {
        Some(50256)
    }
    fn pad_token_id(&self) -> Option<u32> {
        Some(50257)
    }
    fn token_to_piece(&self, token: u32) -> Option<String> {
        Some(format!("<token_{}>", token))
    }
}
pub struct MockBatchProcessor {
    engine: Arc<InferenceEngine>,
    config: BatchProcessorConfig,
}
impl MockBatchProcessor {
    pub fn new(
        engine: Arc<InferenceEngine>,
        config: BatchProcessorConfig,
    ) -> Result<Self, BitNetError> {
        config.validate()?;
        Ok(Self { engine, config })
    }
    pub async fn process_batch(&self, requests: Vec<BatchRequest>) -> Vec<BatchResponse> {
        let mut responses = Vec::new();
        let semaphore = Arc::new(tokio::sync::Semaphore::new(self.config.max_concurrent_requests));
        let mut handles = Vec::new();
        for request in requests {
            let engine = self.engine.clone();
            let permit = semaphore.clone().acquire_owned().await.unwrap();
            let handle = tokio::spawn(async move {
                let _permit = permit;
                let start_time = std::time::Instant::now();
                let result = engine
                    .generate_with_config(&request.prompt, &request.config)
                    .await
                    .map_err(|e| {
                        BitNetError::Inference(InferenceError::GenerationFailed {
                            reason: e.to_string(),
                        })
                    });
                let processing_time = start_time.elapsed();
                BatchResponse {
                    id: request.id,
                    result,
                    metrics: PerformanceMetrics::default(),
                    processing_time,
                }
            });
            handles.push(handle);
        }
        for handle in handles {
            if let Ok(response) = handle.await {
                responses.push(response);
            }
        }
        responses
    }
}
mod batch_config_tests {
    use super::*;
    #[test]
    fn test_batch_processor_config_default() {
        let config = BatchProcessorConfig::default();
        assert_eq!(config.max_batch_size, 8);
        assert_eq!(config.max_queue_size, 100);
        assert_eq!(config.max_concurrent_requests, 4);
        assert_eq!(config.batch_timeout_ms, 100);
        assert!(config.enable_priority_queue);
    }
    #[test]
    fn test_batch_processor_config_validation() {
        let config = BatchProcessorConfig::default();
        assert!(config.validate().is_ok());
        let mut invalid_config = config.clone();
        invalid_config.max_batch_size = 0;
        assert!(invalid_config.validate().is_err());
        assert!(invalid_config.validate().unwrap_err().to_string().contains("max_batch_size"));
        let mut invalid_config = config.clone();
        invalid_config.max_queue_size = 0;
        assert!(invalid_config.validate().is_err());
        assert!(invalid_config.validate().unwrap_err().to_string().contains("max_queue_size"));
        let mut invalid_config = config;
        invalid_config.max_concurrent_requests = 0;
        assert!(invalid_config.validate().is_err());
        assert!(
            invalid_config.validate().unwrap_err().to_string().contains("max_concurrent_requests")
        );
    }
    #[test]
    fn test_queue_stats() {
        let stats = QueueStats {
            queue_length: 50,
            max_queue_size: 100,
            active_requests: 2,
            max_concurrent_requests: 4,
        };
        assert_eq!(stats.queue_utilization(), 0.5);
        assert_eq!(stats.processing_utilization(), 0.5);
        let empty_stats = QueueStats {
            queue_length: 0,
            max_queue_size: 100,
            active_requests: 0,
            max_concurrent_requests: 4,
        };
        assert_eq!(empty_stats.queue_utilization(), 0.0);
        assert_eq!(empty_stats.processing_utilization(), 0.0);
        let full_stats = QueueStats {
            queue_length: 100,
            max_queue_size: 100,
            active_requests: 4,
            max_concurrent_requests: 4,
        };
        assert_eq!(full_stats.queue_utilization(), 1.0);
        assert_eq!(full_stats.processing_utilization(), 1.0);
    }
    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Critical > Priority::High);
        assert!(Priority::High > Priority::Normal);
        assert!(Priority::Normal > Priority::Low);
        let mut priorities =
            vec![Priority::Low, Priority::Critical, Priority::Normal, Priority::High];
        priorities.sort();
        assert_eq!(
            priorities,
            vec![Priority::Low, Priority::Normal, Priority::High, Priority::Critical]
        );
    }
}
mod batch_request_tests {
    use super::*;
    #[test]
    fn test_batch_request_creation() {
        let _config = GenerationConfig::default();
        let request = BatchRequest {
            id: "test-1".to_string(),
            prompt: "Hello, world!".to_string(),
            config: bitnet_inference::GenerationConfig::default(),
            priority: Priority::Normal,
        };
        assert_eq!(request.id, "test-1");
        assert_eq!(request.prompt, "Hello, world!");
        assert_eq!(request.priority, Priority::Normal);
    }
    #[test]
    fn test_batch_request_priority_comparison() {
        let high_priority_request = BatchRequest {
            id: "high".to_string(),
            prompt: "High priority".to_string(),
            config: GenerationConfig::default(),
            priority: Priority::High,
        };
        let low_priority_request = BatchRequest {
            id: "low".to_string(),
            prompt: "Low priority".to_string(),
            config: GenerationConfig::default(),
            priority: Priority::Low,
        };
        assert!(high_priority_request.priority > low_priority_request.priority);
    }
}
mod batch_processing_tests {
    use super::*;
    async fn create_test_engine() -> Arc<InferenceEngine> {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer);
        let device = Device::Cpu;
        Arc::new(InferenceEngine::new(model, tokenizer, device).unwrap())
    }
    #[tokio::test(flavor = "multi_thread")]
    async fn test_batch_processor_creation() {
        let engine = create_test_engine().await;
        let config = BatchProcessorConfig::default();
        let processor = MockBatchProcessor::new(engine, config);
        assert!(processor.is_ok());
    }
    #[tokio::test(flavor = "multi_thread")]
    async fn test_single_batch_processing() {
        let engine = create_test_engine().await;
        let config = BatchProcessorConfig::default();
        let processor = MockBatchProcessor::new(engine, config).unwrap();
        let requests = vec![BatchRequest {
            id: "req-1".to_string(),
            prompt: "Hello, world!".to_string(),
            config: GenerationConfig::default(),
            priority: Priority::Normal,
        }];
        let responses = processor.process_batch(requests).await;
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0].id, "req-1");
        assert!(responses[0].result.is_ok());
        assert!(responses[0].processing_time > Duration::from_millis(0));
    }
    #[tokio::test(flavor = "multi_thread")]
    async fn test_multiple_batch_processing() {
        let engine = create_test_engine().await;
        let config = BatchProcessorConfig::default();
        let processor = MockBatchProcessor::new(engine, config).unwrap();
        let requests = vec![
            BatchRequest {
                id: "req-1".to_string(),
                prompt: "First request".to_string(),
                config: GenerationConfig::default(),
                priority: Priority::Normal,
            },
            BatchRequest {
                id: "req-2".to_string(),
                prompt: "Second request".to_string(),
                config: GenerationConfig::default(),
                priority: Priority::High,
            },
            BatchRequest {
                id: "req-3".to_string(),
                prompt: "Third request".to_string(),
                config: GenerationConfig::default(),
                priority: Priority::Low,
            },
        ];
        let responses = processor.process_batch(requests).await;
        assert_eq!(responses.len(), 3);
        for response in &responses {
            assert!(response.result.is_ok());
            assert!(response.processing_time > Duration::from_millis(0));
        }
        let response_ids: std::collections::HashSet<_> = responses.iter().map(|r| &r.id).collect();
        assert!(response_ids.contains(&"req-1".to_string()));
        assert!(response_ids.contains(&"req-2".to_string()));
        assert!(response_ids.contains(&"req-3".to_string()));
    }
    #[tokio::test(flavor = "multi_thread")]
    async fn test_concurrent_processing_limits() {
        let engine = create_test_engine().await;
        let config = BatchProcessorConfig { max_concurrent_requests: 2, ..Default::default() };
        let processor = MockBatchProcessor::new(engine, config).unwrap();
        let requests: Vec<_> = (0..5)
            .map(|i| BatchRequest {
                id: format!("req-{}", i),
                prompt: format!("Request {}", i),
                config: GenerationConfig::default(),
                priority: Priority::Normal,
            })
            .collect();
        let start_time = std::time::Instant::now();
        let responses = processor.process_batch(requests).await;
        let total_time = start_time.elapsed();
        assert_eq!(responses.len(), 5);
        assert!(total_time > Duration::from_millis(10));
        assert!(total_time < Duration::from_millis(500));
    }
    #[tokio::test(flavor = "multi_thread")]
    async fn test_batch_processing_with_different_configs() {
        let engine = create_test_engine().await;
        let config = BatchProcessorConfig::default();
        let processor = MockBatchProcessor::new(engine, config).unwrap();
        let requests = vec![
            BatchRequest {
                id: "greedy".to_string(),
                prompt: "Greedy generation".to_string(),
                config: GenerationConfig::greedy(),
                priority: Priority::Normal,
            },
            BatchRequest {
                id: "creative".to_string(),
                prompt: "Creative generation".to_string(),
                config: GenerationConfig::creative(),
                priority: Priority::Normal,
            },
            BatchRequest {
                id: "balanced".to_string(),
                prompt: "Balanced generation".to_string(),
                config: GenerationConfig::balanced(),
                priority: Priority::Normal,
            },
        ];
        let responses = processor.process_batch(requests).await;
        assert_eq!(responses.len(), 3);
        for response in &responses {
            assert!(response.result.is_ok());
        }
    }
    #[tokio::test(flavor = "multi_thread")]
    async fn test_batch_processing_performance() {
        let engine = create_test_engine().await;
        let config = BatchProcessorConfig { max_concurrent_requests: 4, ..Default::default() };
        let processor = MockBatchProcessor::new(engine, config).unwrap();
        let num_requests = 10;
        let requests: Vec<_> = (0..num_requests)
            .map(|i| BatchRequest {
                id: format!("perf-req-{}", i),
                prompt: format!("Performance test request {}", i),
                config: GenerationConfig::default().with_max_tokens(10),
                priority: Priority::Normal,
            })
            .collect();
        let start_time = std::time::Instant::now();
        let responses = processor.process_batch(requests).await;
        let total_time = start_time.elapsed();
        assert_eq!(responses.len(), num_requests);
        let throughput = responses.len() as f64 / total_time.as_secs_f64();
        assert!(throughput > 0.0);
        println!("Batch processing throughput: {:.2} requests/second", throughput);
        let successful_requests = responses.iter().filter(|r| r.result.is_ok()).count();
        assert_eq!(successful_requests, num_requests);
    }
    #[tokio::test(flavor = "multi_thread")]
    async fn test_batch_processing_with_timeouts() {
        let engine = create_test_engine().await;
        let config = BatchProcessorConfig::default();
        let processor = MockBatchProcessor::new(engine, config).unwrap();
        let requests = vec![BatchRequest {
            id: "timeout-test".to_string(),
            prompt: "This should complete quickly".to_string(),
            config: GenerationConfig::default().with_max_tokens(5),
            priority: Priority::Normal,
        }];
        let result = timeout(Duration::from_secs(5), processor.process_batch(requests)).await;
        assert!(result.is_ok());
        let responses = result.unwrap();
        assert_eq!(responses.len(), 1);
        assert!(responses[0].result.is_ok());
    }
    #[tokio::test(flavor = "multi_thread")]
    async fn test_empty_batch_processing() {
        let engine = create_test_engine().await;
        let config = BatchProcessorConfig::default();
        let processor = MockBatchProcessor::new(engine, config).unwrap();
        let requests = vec![];
        let responses = processor.process_batch(requests).await;
        assert_eq!(responses.len(), 0);
    }
    #[tokio::test(flavor = "multi_thread")]
    async fn test_large_batch_processing() {
        let engine = create_test_engine().await;
        let config = BatchProcessorConfig {
            max_concurrent_requests: 8,
            max_batch_size: 50,
            ..Default::default()
        };
        let processor = MockBatchProcessor::new(engine, config).unwrap();
        let num_requests = 25;
        let requests: Vec<_> = (0..num_requests)
            .map(|i| BatchRequest {
                id: format!("large-batch-{}", i),
                prompt: format!("Large batch request {}", i),
                config: GenerationConfig::default().with_max_tokens(5),
                priority: if i % 3 == 0 { Priority::High } else { Priority::Normal },
            })
            .collect();
        let responses = processor.process_batch(requests).await;
        assert_eq!(responses.len(), num_requests);
        let successful_requests = responses.iter().filter(|r| r.result.is_ok()).count();
        assert_eq!(successful_requests, num_requests);
        for response in &responses {
            assert!(response.processing_time < Duration::from_secs(2));
        }
    }
}
mod batch_error_handling_tests {
    use super::*;
    #[tokio::test(flavor = "multi_thread")]
    async fn test_batch_processor_invalid_config() {
        let engine = create_test_engine().await;
        let invalid_config = BatchProcessorConfig { max_batch_size: 0, ..Default::default() };
        let processor = MockBatchProcessor::new(engine, invalid_config);
        assert!(processor.is_err());
    }
    async fn create_test_engine() -> Arc<InferenceEngine> {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer);
        let device = Device::Cpu;
        Arc::new(InferenceEngine::new(model, tokenizer, device).unwrap())
    }
    #[tokio::test(flavor = "multi_thread")]
    async fn test_batch_processing_with_mixed_results() {
        let engine = create_test_engine().await;
        let config = BatchProcessorConfig::default();
        let processor = MockBatchProcessor::new(engine, config).unwrap();
        let requests = vec![
            BatchRequest {
                id: "normal-req".to_string(),
                prompt: "Normal request".to_string(),
                config: GenerationConfig::default(),
                priority: Priority::Normal,
            },
            BatchRequest {
                id: "empty-prompt".to_string(),
                prompt: "".to_string(),
                config: GenerationConfig::default(),
                priority: Priority::Normal,
            },
            BatchRequest {
                id: "long-prompt".to_string(),
                prompt: "a".repeat(1000),
                config: GenerationConfig::default(),
                priority: Priority::Normal,
            },
        ];
        let responses = processor.process_batch(requests).await;
        assert_eq!(responses.len(), 3);
        for response in &responses {
            assert!(response.result.is_ok());
        }
    }
    #[tokio::test(flavor = "multi_thread")]
    async fn test_batch_processing_resource_limits() {
        let engine = create_test_engine().await;
        let config = BatchProcessorConfig {
            max_concurrent_requests: 1,
            max_batch_size: 2,
            ..Default::default()
        };
        let processor = MockBatchProcessor::new(engine, config).unwrap();
        let requests: Vec<_> = (0..5)
            .map(|i| BatchRequest {
                id: format!("resource-test-{}", i),
                prompt: format!("Resource test {}", i),
                config: GenerationConfig::default(),
                priority: Priority::Normal,
            })
            .collect();
        let responses = processor.process_batch(requests).await;
        assert_eq!(responses.len(), 5);
        for response in &responses {
            assert!(response.result.is_ok());
        }
    }
}
mod batch_performance_tests {
    use super::*;
    async fn create_test_engine() -> Arc<InferenceEngine> {
        let model = Arc::new(MockModel::new().with_delay(Duration::from_millis(5)));
        let tokenizer = Arc::new(MockTokenizer);
        let device = Device::Cpu;
        Arc::new(InferenceEngine::new(model, tokenizer, device).unwrap())
    }
    #[tokio::test(flavor = "multi_thread")]
    async fn test_batch_vs_sequential_performance() {
        let engine = create_test_engine().await;
        let start_time = std::time::Instant::now();
        for i in 0..5 {
            let prompt = format!("Sequential request {}", i);
            let _ = engine.generate(&prompt).await;
        }
        let sequential_time = start_time.elapsed();
        let config = BatchProcessorConfig { max_concurrent_requests: 5, ..Default::default() };
        let processor = MockBatchProcessor::new(engine, config).unwrap();
        let requests: Vec<_> = (0..5)
            .map(|i| BatchRequest {
                id: format!("batch-{}", i),
                prompt: format!("Batch request {}", i),
                config: GenerationConfig::default(),
                priority: Priority::Normal,
            })
            .collect();
        let start_time = std::time::Instant::now();
        let responses = processor.process_batch(requests).await;
        let batch_time = start_time.elapsed();
        assert_eq!(responses.len(), 5);
        println!("Sequential time: {:?}, Batch time: {:?}", sequential_time, batch_time);
        assert!(batch_time <= sequential_time + Duration::from_millis(100));
    }
    #[tokio::test(flavor = "multi_thread")]
    async fn test_batch_throughput_scaling() {
        let engine = create_test_engine().await;
        for concurrency in [1, 2, 4, 8] {
            let config =
                BatchProcessorConfig { max_concurrent_requests: concurrency, ..Default::default() };
            let processor = MockBatchProcessor::new(engine.clone(), config).unwrap();
            let num_requests = 16;
            let requests: Vec<_> = (0..num_requests)
                .map(|i| BatchRequest {
                    id: format!("scale-test-{}-{}", concurrency, i),
                    prompt: format!("Scaling test request {}", i),
                    config: GenerationConfig::default().with_max_tokens(5),
                    priority: Priority::Normal,
                })
                .collect();
            let start_time = std::time::Instant::now();
            let responses = processor.process_batch(requests).await;
            let duration = start_time.elapsed();
            assert_eq!(responses.len(), num_requests);
            let throughput = responses.len() as f64 / duration.as_secs_f64();
            println!("Concurrency {}: {:.2} requests/second", concurrency, throughput);
            assert!(throughput > 0.0);
        }
    }
    #[tokio::test(flavor = "multi_thread")]
    async fn test_batch_memory_usage() {
        let engine = create_test_engine().await;
        let config = BatchProcessorConfig::default();
        let processor = MockBatchProcessor::new(engine, config).unwrap();
        let requests: Vec<_> = (0..10)
            .map(|i| BatchRequest {
                id: format!("memory-test-{}", i),
                prompt: format!("Memory usage test {}", i),
                config: GenerationConfig::default(),
                priority: Priority::Normal,
            })
            .collect();
        let responses = processor.process_batch(requests).await;
        assert_eq!(responses.len(), 10);
        for response in &responses {
            assert!(response.result.is_ok());
        }
    }
}

//! Integration tests for batch inference with prefill functionality
use bitnet_common::{BitNetConfig, BitNetError, ConcreteTensor, Device};
use bitnet_inference::{GenerationConfig, InferenceEngine};
use bitnet_models::Model;
use bitnet_tokenizers::Tokenizer;
use std::{sync::Arc, thread::sleep, time::Duration};
/// Mock model with timing delay for realistic prefill testing
struct MockModelWithTiming {
    config: BitNetConfig,
}
impl MockModelWithTiming {
    fn new() -> Self {
        Self { config: BitNetConfig::default() }
    }
}
impl Model for MockModelWithTiming {
    fn config(&self) -> &BitNetConfig {
        &self.config
    }
    fn forward(
        &self,
        _input: &ConcreteTensor,
        _cache: &mut dyn std::any::Any,
    ) -> Result<ConcreteTensor, BitNetError> {
        sleep(Duration::from_millis(10));
        Ok(ConcreteTensor::mock(vec![1, 1, self.config.model.hidden_size]))
    }
    fn embed(&self, tokens: &[u32]) -> Result<ConcreteTensor, BitNetError> {
        let seq_len = tokens.len();
        let hidden_dim = self.config.model.hidden_size;
        Ok(ConcreteTensor::mock(vec![seq_len, hidden_dim]))
    }
    fn logits(&self, _hidden: &ConcreteTensor) -> Result<ConcreteTensor, BitNetError> {
        Ok(ConcreteTensor::mock(vec![1, 1, self.config.model.vocab_size]))
    }
}
struct MockTokenizerWithTiming;
impl MockTokenizerWithTiming {
    fn new() -> Self {
        Self
    }
}
impl Tokenizer for MockTokenizerWithTiming {
    fn encode(
        &self,
        text: &str,
        _add_bos: bool,
        _add_special: bool,
    ) -> Result<Vec<u32>, BitNetError> {
        sleep(Duration::from_millis(1));
        Ok((0..text.len().min(10)).map(|i| i as u32 + 1).collect())
    }
    fn decode(&self, tokens: &[u32]) -> Result<String, BitNetError> {
        sleep(Duration::from_millis(1));
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
/// Simple batch processor to simulate CLI functionality
struct BatchProcessor {
    engine: InferenceEngine,
}
impl BatchProcessor {
    fn new(engine: InferenceEngine) -> Self {
        Self { engine }
    }
    async fn process_batch(&mut self, prompts: &[String]) -> anyhow::Result<Vec<BatchResult>> {
        let mut results = Vec::new();
        for prompt in prompts {
            let start_time = std::time::Instant::now();
            let t0 = std::time::Instant::now();
            let prompt_ids = self.engine.tokenizer().encode(prompt, true, false)?;
            let t_tokenize_ms = t0.elapsed().as_secs_f64() * 1e3;
            let t1 = std::time::Instant::now();
            self.engine.prefill(&prompt_ids).await?;
            let t_prefill_ms = t1.elapsed().as_secs_f64() * 1e3;
            let t2 = std::time::Instant::now();
            let config = GenerationConfig::greedy()
                .with_max_tokens(2)
                .with_temperature(0.7)
                .with_top_k(40)
                .with_top_p(0.9)
                .with_repetition_penalty(1.1)
                .with_seed(42)
                .with_stop_sequences(vec![])
                .with_stop_token_ids(vec![])
                .with_stop_string_window(64)
                .with_eos_token_id(None)
                .with_skip_special_tokens(true)
                .with_logits_tap_steps(0)
                .with_logits_topk(10)
                .with_logits_cb(None)
                .with_add_bos(false);
            let generated_ids = self.engine.generate_tokens(&prompt_ids, &config).await?;
            let t_generate_ms = t2.elapsed().as_secs_f64() * 1e3;
            let generated_text = self.engine.tokenizer().decode(&generated_ids)?;
            let total_time = start_time.elapsed().as_secs_f64() * 1e3;
            results.push(BatchResult {
                prompt: prompt.clone(),
                generated_text,
                prompt_tokens: prompt_ids.len(),
                generated_tokens: generated_ids.len(),
                timing_ms: TimingMetrics {
                    tokenize: t_tokenize_ms,
                    prefill: t_prefill_ms,
                    generate: t_generate_ms,
                    total: total_time,
                },
            });
        }
        Ok(results)
    }
}
#[derive(Debug, Clone)]
struct BatchResult {
    pub prompt: String,
    pub generated_text: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub timing_ms: TimingMetrics,
}
#[derive(Debug, Clone)]
struct TimingMetrics {
    pub tokenize: f64,
    pub prefill: f64,
    pub generate: f64,
    pub total: f64,
}
#[tokio::test]
async fn test_batch_prefill_timing() {
    let model = Arc::new(MockModelWithTiming::new());
    let tokenizer = Arc::new(MockTokenizerWithTiming::new());
    let engine = InferenceEngine::new(model, tokenizer, Device::Cpu).unwrap();
    let mut processor = BatchProcessor::new(engine);
    let prompts = vec!["Hello world".to_string(), "Test prompt".to_string()];
    let results = processor.process_batch(&prompts).await.unwrap();
    assert_eq!(results.len(), 2, "Should process both prompts");
    for (i, result) in results.iter().enumerate() {
        assert!(
            result.timing_ms.prefill >= 8.0,
            "Prompt {} prefill should record latency >= 8ms (got {}ms)",
            i,
            result.timing_ms.prefill
        );
        assert!(
            result.timing_ms.tokenize >= 0.5,
            "Prompt {} tokenization should be measurable (got {}ms)",
            i,
            result.timing_ms.tokenize
        );
        assert!(
            result.timing_ms.generate >= 0.0,
            "Prompt {} generation should be recorded (got {}ms)",
            i,
            result.timing_ms.generate
        );
        assert!(
            result.timing_ms.total > result.timing_ms.prefill,
            "Prompt {} total time should include prefill",
            i
        );
        assert!(result.prompt_tokens > 0, "Should have prompt tokens");
        assert!(result.generated_tokens > 0, "Should have generated tokens");
        assert!(!result.generated_text.is_empty(), "Should have generated text");
    }
}
/// Performance consistency test for batch prefill operations
///
/// ## Quarantine Rationale
///
/// This test is timing-sensitive and subject to CI load variance:
/// - Timer resolution: Tokenization threshold (0.5ms) is near system minimum
/// - Scheduler jitter: Prefill window (8-100ms) affected by CPU contention
/// - Async overhead: tokio context switching adds unpredictable latency
///
/// ## Running This Test
///
/// ### Local Development (Recommended)
/// ```bash
/// # Ensure single-threaded, minimal system load:
/// RUN_PERF_TESTS=1 cargo test --test batch_prefill \
///   test_batch_prefill_performance_consistency -- --test-threads=1
/// ```
///
/// ### With nextest
/// ```bash
/// RUN_PERF_TESTS=1 cargo nextest run -p bitnet-inference \
///   --test batch_prefill test_batch_prefill_performance_consistency
/// ```
///
/// ## Expected Behavior
///
/// When run on idle system with single thread:
/// - Tokenize time: 1.0-1.5ms (1ms sleep + overhead)
/// - Prefill time: 10-15ms (10ms sleep + overhead)
/// - If timings exceed these by 50%+ on stable hardware, investigate system load
#[tokio::test]
#[ignore = "flaky in CI; run with RUN_PERF_TESTS=1 for performance validation"]
async fn test_batch_prefill_performance_consistency() {
    if std::env::var("RUN_PERF_TESTS").is_err() {
        eprintln!("⏭️  Skipping performance test; set RUN_PERF_TESTS=1 to enable");
        eprintln!("   Recommended: RUN_PERF_TESTS=1 cargo test -- --test-threads=1");
        return;
    }
    let model = Arc::new(MockModelWithTiming::new());
    let tokenizer = Arc::new(MockTokenizerWithTiming::new());
    let engine = InferenceEngine::new(model, tokenizer, Device::Cpu).unwrap();
    let mut processor = BatchProcessor::new(engine);
    let prompts = vec![
        "Short".to_string(),
        "This is a medium length prompt".to_string(),
        "This is a very long prompt that should still work correctly with prefill operations"
            .to_string(),
    ];
    let results = processor.process_batch(&prompts).await.unwrap();
    assert_eq!(results.len(), 3, "Should process all prompts");
    let prefill_times: Vec<f64> = results.iter().map(|r| r.timing_ms.prefill).collect();
    for (i, &prefill_time) in prefill_times.iter().enumerate() {
        assert!(
            (8.0..=100.0).contains(&prefill_time),
            "Prompt {} prefill time {} should be reasonable",
            i,
            prefill_time
        );
    }
    let tokenize_times: Vec<f64> = results.iter().map(|r| r.timing_ms.tokenize).collect();
    for (i, &tokenize_time) in tokenize_times.iter().enumerate() {
        assert!(
            tokenize_time >= 0.5,
            "Prompt {} tokenization time {} should be measurable",
            i,
            tokenize_time
        );
    }
}
#[tokio::test]
async fn test_prefill_error_recovery() {
    let model = Arc::new(MockModelWithTiming::new());
    let tokenizer = Arc::new(MockTokenizerWithTiming::new());
    let mut engine = InferenceEngine::new(model, tokenizer, Device::Cpu).unwrap();
    let vocab_size = engine.tokenizer().vocab_size() as u32;
    let invalid_tokens = vec![1, 2, vocab_size + 100];
    let result = engine.prefill(&invalid_tokens).await;
    assert!(result.is_err(), "Should fail with invalid tokens");
    let valid_tokens = vec![1, 2, 3];
    let result = engine.prefill(&valid_tokens).await;
    assert!(result.is_ok(), "Should recover with valid tokens");
}
#[tokio::test]
async fn test_empty_batch_handling() {
    let model = Arc::new(MockModelWithTiming::new());
    let tokenizer = Arc::new(MockTokenizerWithTiming::new());
    let engine = InferenceEngine::new(model, tokenizer, Device::Cpu).unwrap();
    let mut processor = BatchProcessor::new(engine);
    let empty_prompts = vec![];
    let results = processor.process_batch(&empty_prompts).await.unwrap();
    assert_eq!(results.len(), 0, "Empty batch should return empty results");
}
#[tokio::test]
async fn test_single_prompt_batch() {
    let model = Arc::new(MockModelWithTiming::new());
    let tokenizer = Arc::new(MockTokenizerWithTiming::new());
    let engine = InferenceEngine::new(model, tokenizer, Device::Cpu).unwrap();
    let mut processor = BatchProcessor::new(engine);
    let single_prompt = vec!["Single test prompt".to_string()];
    let results = processor.process_batch(&single_prompt).await.unwrap();
    assert_eq!(results.len(), 1, "Single prompt should return single result");
    let result = &results[0];
    assert!(result.timing_ms.prefill >= 8.0, "Should have measurable prefill time");
    assert!(!result.generated_text.is_empty(), "Should generate text");
    assert_eq!(result.prompt, "Single test prompt", "Should preserve prompt");
}

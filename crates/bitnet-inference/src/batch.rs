//! # Batch Inference Engine
//!
//! Process multiple inference requests simultaneously with configurable
//! scheduling, priority ordering, and per-request sampling parameters.
//!
//! ## Usage
//!
//! ```rust
//! use bitnet_inference::batch::{BatchConfig, BatchRequest, BatchResult, BatchScheduler};
//! use bitnet_inference::config::GenerationConfig;
//! use std::time::Duration;
//!
//! let mut batch = BatchRequest::new();
//! let id0 = batch.add("What is 2+2?".into(), GenerationConfig::greedy());
//! let id1 = batch.add("Hello world".into(), GenerationConfig::creative());
//! assert_eq!(batch.len(), 2);
//!
//! let config = BatchConfig::new(8, Duration::from_secs(30))
//!     .with_max_total_tokens(4096);
//! let scheduler = BatchScheduler::new(config);
//! let ordered = scheduler.schedule(&batch);
//! assert_eq!(ordered.len(), 2);
//! ```

use crate::config::GenerationConfig;
use serde::{Deserialize, Serialize};
use std::time::Duration;

// ---------------------------------------------------------------------------
// BatchRequest
// ---------------------------------------------------------------------------

/// A single queued inference request.
#[derive(Debug, Clone)]
pub struct RequestEntry {
    /// Unique request ID (position index within the batch).
    pub id: usize,
    /// Input prompt text.
    pub prompt: String,
    /// Per-request generation parameters.
    pub params: GenerationConfig,
}

/// Collection of inference requests to be processed as a batch.
#[derive(Debug, Clone, Default)]
pub struct BatchRequest {
    entries: Vec<RequestEntry>,
}

impl BatchRequest {
    /// Create an empty batch.
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Add a request and return its unique ID.
    pub fn add(&mut self, prompt: String, params: GenerationConfig) -> usize {
        let id = self.entries.len();
        self.entries.push(RequestEntry { id, prompt, params });
        id
    }

    /// Number of requests in the batch.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the batch contains no requests.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over the request entries.
    pub fn iter(&self) -> impl Iterator<Item = &RequestEntry> {
        self.entries.iter()
    }

    /// Get a request by its ID.
    pub fn get(&self, id: usize) -> Option<&RequestEntry> {
        self.entries.get(id)
    }
}

impl<'a> IntoIterator for &'a BatchRequest {
    type Item = &'a RequestEntry;
    type IntoIter = std::slice::Iter<'a, RequestEntry>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.iter()
    }
}

// ---------------------------------------------------------------------------
// BatchResult
// ---------------------------------------------------------------------------

/// Result of a single completed inference request.
#[derive(Debug, Clone)]
pub struct SingleResult {
    /// The request ID this result corresponds to.
    pub id: usize,
    /// Generated text output.
    pub text: String,
    /// Number of tokens generated.
    pub tokens_generated: usize,
}

/// Aggregated results for a completed batch.
#[derive(Debug, Clone, Default)]
pub struct BatchResult {
    results: Vec<Option<SingleResult>>,
}

impl BatchResult {
    /// Pre-allocate result slots for `capacity` requests.
    pub fn with_capacity(capacity: usize) -> Self {
        Self { results: vec![None; capacity] }
    }

    /// Store a result for the given request ID.
    pub fn insert(&mut self, result: SingleResult) {
        let id = result.id;
        if id >= self.results.len() {
            self.results.resize_with(id + 1, || None);
        }
        self.results[id] = Some(result);
    }

    /// Get a result by request ID.
    pub fn get(&self, id: usize) -> Option<&SingleResult> {
        self.results.get(id).and_then(|r| r.as_ref())
    }

    /// Number of completed results.
    pub fn completed_count(&self) -> usize {
        self.results.iter().filter(|r| r.is_some()).count()
    }

    /// Total capacity (number of request slots).
    pub fn capacity(&self) -> usize {
        self.results.len()
    }

    /// Iterate over all completed results.
    pub fn iter(&self) -> impl Iterator<Item = &SingleResult> {
        self.results.iter().filter_map(|r| r.as_ref())
    }
}

impl<'a> IntoIterator for &'a BatchResult {
    type Item = &'a SingleResult;
    type IntoIter = BatchResultIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        BatchResultIter { inner: self.results.iter() }
    }
}

/// Iterator over completed results in a [`BatchResult`].
pub struct BatchResultIter<'a> {
    inner: std::slice::Iter<'a, Option<SingleResult>>,
}

impl<'a> Iterator for BatchResultIter<'a> {
    type Item = &'a SingleResult;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.inner.next() {
                Some(Some(r)) => return Some(r),
                Some(None) => continue,
                None => return None,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// BatchConfig
// ---------------------------------------------------------------------------

/// Configuration for batch processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Maximum number of requests in a single scheduling batch.
    pub max_batch_size: usize,
    /// Maximum time to wait for a batch to fill before processing.
    #[serde(with = "duration_millis")]
    pub timeout: Duration,
    /// Maximum total tokens (prompt + generation) across all requests in a batch.
    pub max_total_tokens: usize,
}

mod duration_millis {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S: Serializer>(d: &Duration, s: S) -> Result<S::Ok, S::Error> {
        d.as_millis().serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Duration, D::Error> {
        let ms = u64::deserialize(d)?;
        Ok(Duration::from_millis(ms))
    }
}

impl BatchConfig {
    /// Create a new batch configuration.
    ///
    /// # Panics
    ///
    /// Panics if `max_batch_size` is zero.
    pub fn new(max_batch_size: usize, timeout: Duration) -> Self {
        assert!(max_batch_size > 0, "max_batch_size must be > 0");
        Self { max_batch_size, timeout, max_total_tokens: 8192 }
    }

    /// Set `max_total_tokens`.
    #[must_use]
    pub fn with_max_total_tokens(mut self, max_total_tokens: usize) -> Self {
        self.max_total_tokens = max_total_tokens;
        self
    }

    /// Validate configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.max_batch_size == 0 {
            return Err("max_batch_size must be > 0".into());
        }
        if self.max_total_tokens == 0 {
            return Err("max_total_tokens must be > 0".into());
        }
        Ok(())
    }
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self { max_batch_size: 8, timeout: Duration::from_secs(30), max_total_tokens: 8192 }
    }
}

// ---------------------------------------------------------------------------
// BatchScheduler
// ---------------------------------------------------------------------------

/// Schedules batch requests for efficient execution.
///
/// The scheduler reorders requests (shorter prompts first) and enforces the
/// configured `max_batch_size` and `max_total_tokens` limits.
#[derive(Debug, Clone)]
pub struct BatchScheduler {
    config: BatchConfig,
}

impl BatchScheduler {
    /// Create a scheduler with the given configuration.
    pub fn new(config: BatchConfig) -> Self {
        Self { config }
    }

    /// Return the active configuration.
    pub fn config(&self) -> &BatchConfig {
        &self.config
    }

    /// Schedule a batch of requests.
    ///
    /// Returns an ordered list of request IDs that should be processed in the
    /// next batch.  Shorter prompts are prioritised; the list is truncated to
    /// `max_batch_size` and the cumulative estimated token budget is capped at
    /// `max_total_tokens`.
    pub fn schedule(&self, batch: &BatchRequest) -> Vec<usize> {
        if batch.is_empty() {
            return Vec::new();
        }

        // Sort by prompt length (ascending) – shorter prompts first.
        let mut indices: Vec<usize> = (0..batch.len()).collect();
        indices.sort_by_key(|&i| batch.entries[i].prompt.len());

        let mut selected = Vec::new();
        let mut total_tokens: usize = 0;

        for &idx in &indices {
            if selected.len() >= self.config.max_batch_size {
                break;
            }
            let entry = &batch.entries[idx];
            // Rough estimate: 1 token ≈ 4 chars for prompt + max_new_tokens.
            let est = estimate_tokens(&entry.prompt) + entry.params.max_new_tokens as usize;
            if total_tokens.saturating_add(est) > self.config.max_total_tokens
                && !selected.is_empty()
            {
                break;
            }
            total_tokens = total_tokens.saturating_add(est);
            selected.push(entry.id);
        }

        selected
    }
}

/// Rough token estimate: ~4 characters per token.
fn estimate_tokens(text: &str) -> usize {
    text.len().div_ceil(4)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GenerationConfig;

    // -- BatchRequest -------------------------------------------------------

    #[test]
    fn test_empty_batch() {
        let batch = BatchRequest::new();
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
    }

    #[test]
    fn test_single_request() {
        let mut batch = BatchRequest::new();
        let id = batch.add("Hello".into(), GenerationConfig::greedy());
        assert_eq!(id, 0);
        assert_eq!(batch.len(), 1);
        assert!(!batch.is_empty());
        assert_eq!(batch.get(id).unwrap().prompt, "Hello");
    }

    #[test]
    fn test_multiple_requests_different_params() {
        let mut batch = BatchRequest::new();
        let id0 = batch.add("Short".into(), GenerationConfig::greedy());
        let id1 = batch.add("A longer prompt".into(), GenerationConfig::creative());
        let id2 = batch.add("Medium text".into(), GenerationConfig::balanced());

        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(batch.len(), 3);
        assert_eq!(batch.get(0).unwrap().params.temperature, 0.0);
        assert_eq!(batch.get(1).unwrap().params.temperature, 0.9);
        assert_eq!(batch.get(2).unwrap().params.temperature, 0.7);
    }

    #[test]
    fn test_request_ids_are_sequential() {
        let mut batch = BatchRequest::new();
        for i in 0..5 {
            let id = batch.add(format!("prompt {i}"), GenerationConfig::default());
            assert_eq!(id, i);
        }
    }

    #[test]
    fn test_get_nonexistent_request() {
        let batch = BatchRequest::new();
        assert!(batch.get(0).is_none());
        assert!(batch.get(999).is_none());
    }

    #[test]
    fn test_batch_iteration() {
        let mut batch = BatchRequest::new();
        batch.add("a".into(), GenerationConfig::default());
        batch.add("b".into(), GenerationConfig::default());
        let prompts: Vec<&str> = batch.iter().map(|e| e.prompt.as_str()).collect();
        assert_eq!(prompts, vec!["a", "b"]);
    }

    // -- BatchResult --------------------------------------------------------

    #[test]
    fn test_empty_result() {
        let result = BatchResult::with_capacity(3);
        assert_eq!(result.completed_count(), 0);
        assert_eq!(result.capacity(), 3);
        assert!(result.get(0).is_none());
    }

    #[test]
    fn test_result_insert_and_retrieve() {
        let mut result = BatchResult::with_capacity(2);
        result.insert(SingleResult { id: 0, text: "four".into(), tokens_generated: 1 });
        result.insert(SingleResult { id: 1, text: "hi there".into(), tokens_generated: 2 });

        assert_eq!(result.completed_count(), 2);
        assert_eq!(result.get(0).unwrap().text, "four");
        assert_eq!(result.get(1).unwrap().tokens_generated, 2);
    }

    #[test]
    fn test_result_iteration() {
        let mut result = BatchResult::with_capacity(3);
        result.insert(SingleResult { id: 0, text: "a".into(), tokens_generated: 1 });
        // slot 1 intentionally left empty
        result.insert(SingleResult { id: 2, text: "c".into(), tokens_generated: 1 });

        let texts: Vec<&str> = result.iter().map(|r| r.text.as_str()).collect();
        assert_eq!(texts, vec!["a", "c"]);
    }

    #[test]
    fn test_result_into_iter() {
        let mut result = BatchResult::with_capacity(2);
        result.insert(SingleResult { id: 0, text: "x".into(), tokens_generated: 1 });
        result.insert(SingleResult { id: 1, text: "y".into(), tokens_generated: 1 });

        let texts: Vec<&str> = (&result).into_iter().map(|r| r.text.as_str()).collect();
        assert_eq!(texts, vec!["x", "y"]);
    }

    #[test]
    fn test_result_auto_grows_on_insert() {
        let mut result = BatchResult::with_capacity(1);
        result.insert(SingleResult { id: 5, text: "late".into(), tokens_generated: 1 });
        assert_eq!(result.capacity(), 6);
        assert_eq!(result.get(5).unwrap().text, "late");
    }

    // -- BatchConfig --------------------------------------------------------

    #[test]
    fn test_config_defaults() {
        let config = BatchConfig::default();
        assert_eq!(config.max_batch_size, 8);
        assert_eq!(config.timeout, Duration::from_secs(30));
        assert_eq!(config.max_total_tokens, 8192);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_custom() {
        let config = BatchConfig::new(16, Duration::from_millis(500)).with_max_total_tokens(4096);
        assert_eq!(config.max_batch_size, 16);
        assert_eq!(config.timeout, Duration::from_millis(500));
        assert_eq!(config.max_total_tokens, 4096);
        assert!(config.validate().is_ok());
    }

    #[test]
    #[should_panic(expected = "max_batch_size must be > 0")]
    fn test_config_zero_batch_size_panics() {
        BatchConfig::new(0, Duration::from_secs(1));
    }

    #[test]
    fn test_config_validate_zero_tokens() {
        let config = BatchConfig { max_total_tokens: 0, ..Default::default() };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_serialization_roundtrip() {
        let original = BatchConfig::new(4, Duration::from_millis(1500)).with_max_total_tokens(2048);
        let json = serde_json::to_string(&original).unwrap();
        let restored: BatchConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(original.max_batch_size, restored.max_batch_size);
        assert_eq!(original.timeout, restored.timeout);
        assert_eq!(original.max_total_tokens, restored.max_total_tokens);
    }

    // -- BatchScheduler -----------------------------------------------------

    #[test]
    fn test_schedule_empty_batch() {
        let scheduler = BatchScheduler::new(BatchConfig::default());
        let batch = BatchRequest::new();
        assert!(scheduler.schedule(&batch).is_empty());
    }

    #[test]
    fn test_schedule_single_request() {
        let scheduler = BatchScheduler::new(BatchConfig::default());
        let mut batch = BatchRequest::new();
        batch.add("Hello".into(), GenerationConfig::greedy());
        let ids = scheduler.schedule(&batch);
        assert_eq!(ids, vec![0]);
    }

    #[test]
    fn test_schedule_orders_short_prompts_first() {
        let scheduler = BatchScheduler::new(BatchConfig::default());
        let mut batch = BatchRequest::new();
        batch.add("This is a very long prompt with many words".into(), GenerationConfig::greedy());
        batch.add("Short".into(), GenerationConfig::greedy());
        batch.add("A medium length prompt".into(), GenerationConfig::greedy());

        let ids = scheduler.schedule(&batch);
        // Expect: short (id=1), medium (id=2), long (id=0)
        assert_eq!(ids[0], 1);
        assert_eq!(ids[1], 2);
        assert_eq!(ids[2], 0);
    }

    #[test]
    fn test_schedule_respects_max_batch_size() {
        let config = BatchConfig::new(2, Duration::from_secs(10));
        let scheduler = BatchScheduler::new(config);

        let mut batch = BatchRequest::new();
        for i in 0..5 {
            batch.add(format!("prompt {i}"), GenerationConfig::greedy());
        }

        let ids = scheduler.schedule(&batch);
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn test_schedule_respects_max_total_tokens() {
        // With max_total_tokens = 50 and max_new_tokens = 100 per request,
        // only the first request fits after estimation.
        let config = BatchConfig::new(10, Duration::from_secs(10)).with_max_total_tokens(50);
        let scheduler = BatchScheduler::new(config);

        let mut batch = BatchRequest::new();
        // First request: ~2 prompt tokens + 100 gen = 102 (exceeds 50, but it's
        // the first so it's always admitted).
        batch.add("Hi".into(), GenerationConfig::default());
        // Second request: would push total well over limit.
        batch.add("Hey".into(), GenerationConfig::default());

        let ids = scheduler.schedule(&batch);
        // First request always admitted; second would exceed budget.
        assert_eq!(ids.len(), 1);
    }

    #[test]
    fn test_scheduler_config_accessor() {
        let config = BatchConfig::new(4, Duration::from_millis(200));
        let scheduler = BatchScheduler::new(config.clone());
        assert_eq!(scheduler.config().max_batch_size, 4);
    }

    #[test]
    fn test_estimate_tokens_heuristic() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("Hi"), 1);
        assert_eq!(estimate_tokens("Hello world!"), 3);
        // 20 chars → 5 tokens
        assert_eq!(estimate_tokens("12345678901234567890"), 5);
    }

    // -- Integration-style ---------------------------------------------------

    #[test]
    fn test_full_batch_workflow() {
        let mut batch = BatchRequest::new();
        let id0 = batch.add("What is 2+2?".into(), GenerationConfig::greedy().with_max_tokens(8));
        let id1 =
            batch.add("Tell me a story".into(), GenerationConfig::creative().with_max_tokens(32));

        let config = BatchConfig::new(8, Duration::from_secs(5)).with_max_total_tokens(4096);
        let scheduler = BatchScheduler::new(config);
        let order = scheduler.schedule(&batch);
        assert!(!order.is_empty());

        // Simulate execution by populating results.
        let mut results = BatchResult::with_capacity(batch.len());
        for &req_id in &order {
            let entry = batch.get(req_id).unwrap();
            results.insert(SingleResult {
                id: entry.id,
                text: format!("output for '{}'", entry.prompt),
                tokens_generated: entry.params.max_new_tokens as usize,
            });
        }

        assert!(results.get(id0).is_some());
        assert!(results.get(id1).is_some());
        assert_eq!(results.completed_count(), 2);
    }
}

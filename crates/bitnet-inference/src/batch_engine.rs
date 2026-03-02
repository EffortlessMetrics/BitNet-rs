//! # Batch Inference Engine for Dense SLM Models
//!
//! Queue-based batch inference engine with configurable scheduling policies,
//! priority-based request management, and throughput/latency tracking.

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Priority
// ---------------------------------------------------------------------------

/// Request priority levels for scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

// ---------------------------------------------------------------------------
// SchedulingPolicy
// ---------------------------------------------------------------------------

/// Scheduling policy for batch request ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingPolicy {
    /// First-in, first-out ordering.
    FIFO,
    /// Higher-priority requests are processed first.
    PriorityBased,
    /// Requests with fewer max_tokens are processed first.
    ShortestJobFirst,
    /// Round-robin across priority levels.
    RoundRobin,
}

// ---------------------------------------------------------------------------
// BatchRequest
// ---------------------------------------------------------------------------

/// A single inference request submitted to the batch engine.
#[derive(Debug, Clone)]
pub struct BatchRequest {
    /// Unique request identifier.
    pub id: String,
    /// Input prompt text.
    pub prompt: String,
    /// Maximum number of tokens to generate.
    pub max_tokens: usize,
    /// Sampling temperature.
    pub temperature: f32,
    /// Request priority.
    pub priority: Priority,
}

// ---------------------------------------------------------------------------
// BatchResponse
// ---------------------------------------------------------------------------

/// Result of a completed inference request.
#[derive(Debug, Clone)]
pub struct BatchResponse {
    /// ID of the originating request.
    pub request_id: String,
    /// Generated text.
    pub text: String,
    /// Number of tokens produced.
    pub tokens_generated: usize,
    /// Reason generation stopped (e.g. "length", "stop", "cancelled").
    pub finish_reason: String,
    /// Wall-clock time in milliseconds.
    pub time_ms: u64,
}

// ---------------------------------------------------------------------------
// BatchConfig
// ---------------------------------------------------------------------------

/// Configuration for the batch inference engine.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum requests processed in a single batch.
    pub max_batch_size: usize,
    /// Maximum number of pending requests in the queue.
    pub max_queue_depth: usize,
    /// Scheduling policy.
    pub scheduling_policy: SchedulingPolicy,
    /// Per-request timeout in milliseconds.
    pub timeout_ms: u64,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            max_queue_depth: 1000,
            scheduling_policy: SchedulingPolicy::FIFO,
            timeout_ms: 30_000,
        }
    }
}

// ---------------------------------------------------------------------------
// BatchStats
// ---------------------------------------------------------------------------

/// Runtime statistics for the batch engine.
#[derive(Debug, Clone)]
pub struct BatchStats {
    pub total_requests: u64,
    pub completed_requests: u64,
    pub failed_requests: u64,
    pub avg_latency_ms: f64,
    pub avg_tokens_per_second: f64,
    pub queue_depth: usize,
    pub active_batch_size: usize,
    pub uptime_ms: u64,
}

impl Default for BatchStats {
    fn default() -> Self {
        Self {
            total_requests: 0,
            completed_requests: 0,
            failed_requests: 0,
            avg_latency_ms: 0.0,
            avg_tokens_per_second: 0.0,
            queue_depth: 0,
            active_batch_size: 0,
            uptime_ms: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal queued entry
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct QueuedRequest {
    request: BatchRequest,
    enqueued_at: Instant,
}

// ---------------------------------------------------------------------------
// BatchEngine
// ---------------------------------------------------------------------------

/// Queue-based batch inference engine.
///
/// Requests are submitted via [`submit`](Self::submit), immediately "processed"
/// (placeholder logic generates a mock response), and results can be retrieved
/// via [`get_result`](Self::get_result).
pub struct BatchEngine {
    config: BatchConfig,
    stats: BatchStats,
    queue: VecDeque<QueuedRequest>,
    results: HashMap<String, BatchResponse>,
    start_time: Instant,
    round_robin_index: usize,
}

impl BatchEngine {
    /// Create a new engine with the given configuration.
    pub fn new(config: BatchConfig) -> Self {
        Self {
            config,
            stats: BatchStats::default(),
            queue: VecDeque::new(),
            results: HashMap::new(),
            start_time: Instant::now(),
            round_robin_index: 0,
        }
    }

    /// Submit a request for processing.
    ///
    /// Returns the request ID on success, or an error string if the queue is
    /// full or the request is invalid.
    pub fn submit(&mut self, request: BatchRequest) -> Result<String, String> {
        if request.prompt.is_empty() {
            return Err("empty prompt".to_string());
        }
        if request.max_tokens == 0 {
            return Err("max_tokens must be > 0".to_string());
        }
        if self.queue.len() >= self.config.max_queue_depth {
            self.stats.failed_requests += 1;
            return Err("queue is full".to_string());
        }

        let id = request.id.clone();
        self.queue.push_back(QueuedRequest { request, enqueued_at: Instant::now() });
        self.stats.total_requests += 1;
        self.stats.queue_depth = self.queue.len();

        self.process_queue();

        Ok(id)
    }

    /// Retrieve the result for a completed request.
    pub fn get_result(&self, request_id: &str) -> Option<BatchResponse> {
        self.results.get(request_id).cloned()
    }

    /// Current queue depth (pending requests).
    pub fn queue_depth(&self) -> usize {
        self.queue.len()
    }

    /// Returns `true` when the queue has reached `max_queue_depth`.
    pub fn is_full(&self) -> bool {
        self.queue.len() >= self.config.max_queue_depth
    }

    /// Cancel a pending request. Returns `true` if the request was found and
    /// removed.
    pub fn cancel(&mut self, request_id: &str) -> bool {
        let before = self.queue.len();
        self.queue.retain(|q| q.request.id != request_id);
        let removed = self.queue.len() < before;
        if removed {
            self.stats.queue_depth = self.queue.len();
            self.results.insert(
                request_id.to_string(),
                BatchResponse {
                    request_id: request_id.to_string(),
                    text: String::new(),
                    tokens_generated: 0,
                    finish_reason: "cancelled".to_string(),
                    time_ms: 0,
                },
            );
        }
        removed
    }

    /// Snapshot of current engine statistics.
    pub fn get_stats(&self) -> &BatchStats {
        &self.stats
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn process_queue(&mut self) {
        self.sort_queue();

        let batch_size = self.config.max_batch_size.min(self.queue.len());
        let mut batch: Vec<QueuedRequest> = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            if let Some(entry) = self.queue.pop_front() {
                batch.push(entry);
            }
        }

        self.stats.active_batch_size = batch.len();

        for entry in &batch {
            let elapsed = entry.enqueued_at.elapsed();
            let timed_out = elapsed.as_millis() as u64 > self.config.timeout_ms;

            let response = if timed_out {
                self.stats.failed_requests += 1;
                BatchResponse {
                    request_id: entry.request.id.clone(),
                    text: String::new(),
                    tokens_generated: 0,
                    finish_reason: "timeout".to_string(),
                    time_ms: elapsed.as_millis() as u64,
                }
            } else {
                let tokens = entry.request.max_tokens;
                let text = format!("Generated response for: {}", entry.request.prompt);
                let time_ms = elapsed.as_millis() as u64;

                self.update_latency(time_ms, tokens);
                self.stats.completed_requests += 1;

                BatchResponse {
                    request_id: entry.request.id.clone(),
                    text,
                    tokens_generated: tokens,
                    finish_reason: "length".to_string(),
                    time_ms,
                }
            };

            self.results.insert(entry.request.id.clone(), response);
        }

        self.stats.queue_depth = self.queue.len();
        self.stats.uptime_ms = self.start_time.elapsed().as_millis() as u64;
    }

    fn sort_queue(&mut self) {
        match self.config.scheduling_policy {
            SchedulingPolicy::FIFO => { /* already in insertion order */ }
            SchedulingPolicy::PriorityBased => {
                let mut v: Vec<_> = self.queue.drain(..).collect();
                v.sort_by(|a, b| b.request.priority.cmp(&a.request.priority));
                self.queue = v.into_iter().collect();
            }
            SchedulingPolicy::ShortestJobFirst => {
                let mut v: Vec<_> = self.queue.drain(..).collect();
                v.sort_by_key(|q| q.request.max_tokens);
                self.queue = v.into_iter().collect();
            }
            SchedulingPolicy::RoundRobin => {
                // Rotate the queue by the round-robin index to distribute
                // processing across arrival order.
                let len = self.queue.len();
                if len > 1 {
                    let rotate = self.round_robin_index % len;
                    self.queue.rotate_left(rotate);
                    self.round_robin_index = self.round_robin_index.wrapping_add(1);
                }
            }
        }
    }

    fn update_latency(&mut self, time_ms: u64, tokens: usize) {
        let n = self.stats.completed_requests as f64;
        // Incremental average: completed_requests has NOT been bumped yet.
        self.stats.avg_latency_ms = (self.stats.avg_latency_ms * n + time_ms as f64) / (n + 1.0);

        if time_ms > 0 && tokens > 0 {
            let tps = (tokens as f64 / time_ms as f64) * 1000.0;
            self.stats.avg_tokens_per_second =
                (self.stats.avg_tokens_per_second * n + tps) / (n + 1.0);
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_request(id: &str, prompt: &str, max_tokens: usize) -> BatchRequest {
        BatchRequest {
            id: id.to_string(),
            prompt: prompt.to_string(),
            max_tokens,
            temperature: 0.7,
            priority: Priority::Normal,
        }
    }

    fn make_request_with_priority(
        id: &str,
        prompt: &str,
        max_tokens: usize,
        priority: Priority,
    ) -> BatchRequest {
        BatchRequest {
            id: id.to_string(),
            prompt: prompt.to_string(),
            max_tokens,
            temperature: 0.7,
            priority,
        }
    }

    // -- BatchConfig defaults ------------------------------------------------

    #[test]
    fn test_batch_config_defaults() {
        let cfg = BatchConfig::default();
        assert_eq!(cfg.max_batch_size, 8);
        assert_eq!(cfg.max_queue_depth, 1000);
        assert_eq!(cfg.scheduling_policy, SchedulingPolicy::FIFO);
        assert_eq!(cfg.timeout_ms, 30_000);
    }

    #[test]
    fn test_batch_config_custom() {
        let cfg = BatchConfig {
            max_batch_size: 16,
            max_queue_depth: 500,
            scheduling_policy: SchedulingPolicy::PriorityBased,
            timeout_ms: 60_000,
        };
        assert_eq!(cfg.max_batch_size, 16);
        assert_eq!(cfg.max_queue_depth, 500);
    }

    // -- BatchRequest / BatchResponse construction ---------------------------

    #[test]
    fn test_batch_request_construction() {
        let req = make_request("r1", "Hello", 32);
        assert_eq!(req.id, "r1");
        assert_eq!(req.prompt, "Hello");
        assert_eq!(req.max_tokens, 32);
        assert!((req.temperature - 0.7).abs() < f32::EPSILON);
        assert_eq!(req.priority, Priority::Normal);
    }

    #[test]
    fn test_batch_response_construction() {
        let resp = BatchResponse {
            request_id: "r1".to_string(),
            text: "world".to_string(),
            tokens_generated: 5,
            finish_reason: "length".to_string(),
            time_ms: 42,
        };
        assert_eq!(resp.request_id, "r1");
        assert_eq!(resp.tokens_generated, 5);
        assert_eq!(resp.finish_reason, "length");
        assert_eq!(resp.time_ms, 42);
    }

    // -- Priority ordering ---------------------------------------------------

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Critical > Priority::High);
        assert!(Priority::High > Priority::Normal);
        assert!(Priority::Normal > Priority::Low);
    }

    #[test]
    fn test_priority_equality() {
        assert_eq!(Priority::Normal, Priority::Normal);
        assert_ne!(Priority::Low, Priority::High);
    }

    // -- SchedulingPolicy variants -------------------------------------------

    #[test]
    fn test_scheduling_policy_variants() {
        let policies = [
            SchedulingPolicy::FIFO,
            SchedulingPolicy::PriorityBased,
            SchedulingPolicy::ShortestJobFirst,
            SchedulingPolicy::RoundRobin,
        ];
        assert_eq!(policies.len(), 4);
        assert_eq!(policies[0], SchedulingPolicy::FIFO);
    }

    #[test]
    fn test_scheduling_policy_equality() {
        assert_eq!(SchedulingPolicy::FIFO, SchedulingPolicy::FIFO);
        assert_ne!(SchedulingPolicy::FIFO, SchedulingPolicy::PriorityBased);
    }

    // -- BatchStats initialization -------------------------------------------

    #[test]
    fn test_batch_stats_default() {
        let stats = BatchStats::default();
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.completed_requests, 0);
        assert_eq!(stats.failed_requests, 0);
        assert!((stats.avg_latency_ms - 0.0).abs() < f64::EPSILON);
        assert!((stats.avg_tokens_per_second - 0.0).abs() < f64::EPSILON);
        assert_eq!(stats.queue_depth, 0);
        assert_eq!(stats.active_batch_size, 0);
        assert_eq!(stats.uptime_ms, 0);
    }

    // -- BatchEngine: submit → get_result ------------------------------------

    #[test]
    fn test_submit_and_get_result() {
        let mut engine = BatchEngine::new(BatchConfig::default());
        let req = make_request("r1", "Hello world", 10);
        let id = engine.submit(req).unwrap();
        assert_eq!(id, "r1");

        let resp = engine.get_result("r1").expect("result should exist");
        assert_eq!(resp.request_id, "r1");
        assert_eq!(resp.tokens_generated, 10);
        assert_eq!(resp.finish_reason, "length");
        assert!(!resp.text.is_empty());
    }

    #[test]
    fn test_get_result_missing() {
        let engine = BatchEngine::new(BatchConfig::default());
        assert!(engine.get_result("nonexistent").is_none());
    }

    // -- Queue depth tracking ------------------------------------------------

    #[test]
    fn test_queue_depth_tracking() {
        let config =
            BatchConfig { max_batch_size: 1, max_queue_depth: 100, ..BatchConfig::default() };
        let mut engine = BatchEngine::new(config);

        // After submit, the single item is immediately processed.
        engine.submit(make_request("r1", "a", 5)).unwrap();
        // Queue should be drained after processing.
        assert_eq!(engine.queue_depth(), 0);

        let stats = engine.get_stats();
        assert_eq!(stats.total_requests, 1);
        assert_eq!(stats.completed_requests, 1);
    }

    // -- is_full -------------------------------------------------------------

    #[test]
    fn test_is_full_when_queue_at_capacity() {
        // Use a tiny batch size so items are not immediately drained,
        // and a tiny queue depth to fill quickly.
        let config = BatchConfig {
            max_batch_size: 0, // process nothing on submit
            max_queue_depth: 2,
            ..BatchConfig::default()
        };
        let mut engine = BatchEngine::new(config);
        assert!(!engine.is_full());

        engine.queue.push_back(QueuedRequest {
            request: make_request("r1", "a", 5),
            enqueued_at: Instant::now(),
        });
        engine.queue.push_back(QueuedRequest {
            request: make_request("r2", "b", 5),
            enqueued_at: Instant::now(),
        });
        assert!(engine.is_full());
    }

    // -- Cancel request ------------------------------------------------------

    #[test]
    fn test_cancel_request() {
        let config =
            BatchConfig { max_batch_size: 0, max_queue_depth: 10, ..BatchConfig::default() };
        let mut engine = BatchEngine::new(config);
        engine.queue.push_back(QueuedRequest {
            request: make_request("r1", "test", 5),
            enqueued_at: Instant::now(),
        });

        assert!(engine.cancel("r1"));
        assert_eq!(engine.queue_depth(), 0);

        let resp = engine.get_result("r1").expect("cancelled result");
        assert_eq!(resp.finish_reason, "cancelled");
    }

    #[test]
    fn test_cancel_nonexistent() {
        let mut engine = BatchEngine::new(BatchConfig::default());
        assert!(!engine.cancel("missing"));
    }

    // -- FIFO scheduling order -----------------------------------------------

    #[test]
    fn test_fifo_scheduling_order() {
        let config = BatchConfig {
            max_batch_size: 10,
            scheduling_policy: SchedulingPolicy::FIFO,
            ..BatchConfig::default()
        };
        let mut engine = BatchEngine::new(config);
        engine.submit(make_request("r1", "first", 5)).unwrap();
        engine.submit(make_request("r2", "second", 5)).unwrap();
        engine.submit(make_request("r3", "third", 5)).unwrap();

        // All processed; verify all results exist (FIFO means insertion order).
        assert!(engine.get_result("r1").is_some());
        assert!(engine.get_result("r2").is_some());
        assert!(engine.get_result("r3").is_some());
    }

    // -- Priority scheduling order -------------------------------------------

    #[test]
    fn test_priority_scheduling_order() {
        let config = BatchConfig {
            max_batch_size: 0, // don't auto-process
            max_queue_depth: 100,
            scheduling_policy: SchedulingPolicy::PriorityBased,
            ..BatchConfig::default()
        };
        let mut engine = BatchEngine::new(config);

        // Enqueue in reverse priority order.
        engine.queue.push_back(QueuedRequest {
            request: make_request_with_priority("low", "lo", 5, Priority::Low),
            enqueued_at: Instant::now(),
        });
        engine.queue.push_back(QueuedRequest {
            request: make_request_with_priority("crit", "cr", 5, Priority::Critical),
            enqueued_at: Instant::now(),
        });
        engine.queue.push_back(QueuedRequest {
            request: make_request_with_priority("norm", "no", 5, Priority::Normal),
            enqueued_at: Instant::now(),
        });

        // Sort and verify ordering.
        engine.sort_queue();
        let ids: Vec<_> = engine.queue.iter().map(|q| q.request.id.clone()).collect();
        assert_eq!(ids, vec!["crit", "norm", "low"]);
    }

    // -- ShortestJobFirst scheduling -----------------------------------------

    #[test]
    fn test_shortest_job_first_scheduling() {
        let config = BatchConfig {
            max_batch_size: 0,
            max_queue_depth: 100,
            scheduling_policy: SchedulingPolicy::ShortestJobFirst,
            ..BatchConfig::default()
        };
        let mut engine = BatchEngine::new(config);

        engine.queue.push_back(QueuedRequest {
            request: make_request("long", "x", 100),
            enqueued_at: Instant::now(),
        });
        engine.queue.push_back(QueuedRequest {
            request: make_request("short", "y", 5),
            enqueued_at: Instant::now(),
        });
        engine.queue.push_back(QueuedRequest {
            request: make_request("mid", "z", 50),
            enqueued_at: Instant::now(),
        });

        engine.sort_queue();
        let ids: Vec<_> = engine.queue.iter().map(|q| q.request.id.clone()).collect();
        assert_eq!(ids, vec!["short", "mid", "long"]);
    }

    // -- Submit with queue full → error --------------------------------------

    #[test]
    fn test_submit_queue_full_error() {
        let config =
            BatchConfig { max_batch_size: 0, max_queue_depth: 1, ..BatchConfig::default() };
        let mut engine = BatchEngine::new(config);

        // Manually fill queue to capacity.
        engine.queue.push_back(QueuedRequest {
            request: make_request("r1", "a", 5),
            enqueued_at: Instant::now(),
        });

        let result = engine.submit(make_request("r2", "b", 5));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("queue is full"));
    }

    // -- Multiple concurrent requests ----------------------------------------

    #[test]
    fn test_multiple_concurrent_requests() {
        let config = BatchConfig { max_batch_size: 4, ..BatchConfig::default() };
        let mut engine = BatchEngine::new(config);

        for i in 0..4 {
            let id = format!("r{i}");
            engine.submit(make_request(&id, &format!("prompt {i}"), 8)).unwrap();
        }

        for i in 0..4 {
            let id = format!("r{i}");
            let resp = engine.get_result(&id).expect("result should exist");
            assert_eq!(resp.tokens_generated, 8);
        }

        let stats = engine.get_stats();
        assert_eq!(stats.completed_requests, 4);
    }

    // -- Stats updates -------------------------------------------------------

    #[test]
    fn test_stats_updates_after_submit() {
        let mut engine = BatchEngine::new(BatchConfig::default());
        engine.submit(make_request("r1", "test", 10)).unwrap();

        let stats = engine.get_stats();
        assert_eq!(stats.total_requests, 1);
        assert_eq!(stats.completed_requests, 1);
        assert_eq!(stats.failed_requests, 0);
    }

    #[test]
    fn test_stats_failed_increments_on_queue_full() {
        let config =
            BatchConfig { max_batch_size: 0, max_queue_depth: 0, ..BatchConfig::default() };
        let mut engine = BatchEngine::new(config);
        let _ = engine.submit(make_request("r1", "test", 5));

        assert_eq!(engine.get_stats().failed_requests, 1);
    }

    // -- Edge cases ----------------------------------------------------------

    #[test]
    fn test_empty_prompt_rejected() {
        let mut engine = BatchEngine::new(BatchConfig::default());
        let req = make_request("r1", "", 10);
        let result = engine.submit(req);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty prompt"));
    }

    #[test]
    fn test_zero_max_tokens_rejected() {
        let mut engine = BatchEngine::new(BatchConfig::default());
        let req = make_request("r1", "hello", 0);
        let result = engine.submit(req);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("max_tokens"));
    }

    #[test]
    fn test_round_robin_scheduling() {
        let config = BatchConfig {
            max_batch_size: 0,
            max_queue_depth: 100,
            scheduling_policy: SchedulingPolicy::RoundRobin,
            ..BatchConfig::default()
        };
        let mut engine = BatchEngine::new(config);

        for i in 0..3 {
            engine.queue.push_back(QueuedRequest {
                request: make_request(&format!("r{i}"), &format!("p{i}"), 5),
                enqueued_at: Instant::now(),
            });
        }

        // First sort rotates by index 0 → no change.
        engine.sort_queue();
        let ids: Vec<_> = engine.queue.iter().map(|q| q.request.id.clone()).collect();
        assert_eq!(ids.len(), 3);

        // Second sort rotates by index 1.
        engine.sort_queue();
        let ids2: Vec<_> = engine.queue.iter().map(|q| q.request.id.clone()).collect();
        assert_eq!(ids2.len(), 3);
    }

    #[test]
    fn test_large_batch_processing() {
        let config =
            BatchConfig { max_batch_size: 50, max_queue_depth: 200, ..BatchConfig::default() };
        let mut engine = BatchEngine::new(config);

        for i in 0..50 {
            engine.submit(make_request(&format!("r{i}"), &format!("prompt {i}"), 4)).unwrap();
        }

        assert_eq!(engine.get_stats().completed_requests, 50);
    }

    #[test]
    fn test_engine_new_with_default_config() {
        let engine = BatchEngine::new(BatchConfig::default());
        assert_eq!(engine.queue_depth(), 0);
        assert!(!engine.is_full());
        assert_eq!(engine.get_stats().total_requests, 0);
    }

    #[test]
    fn test_batch_response_text_content() {
        let mut engine = BatchEngine::new(BatchConfig::default());
        engine.submit(make_request("r1", "Explain rust", 16)).unwrap();
        let resp = engine.get_result("r1").unwrap();
        assert!(resp.text.contains("Explain rust"));
    }
}

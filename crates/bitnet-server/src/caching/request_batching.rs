//! Request batching and queuing for optimal throughput

use anyhow::Result;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock, Semaphore};
use uuid::Uuid;

use super::CachingConfig;

/// Batched inference request
#[derive(Debug, Clone)]
pub struct BatchedRequest {
    /// Request identifier
    pub id: String,
    /// Request prompt
    pub prompt: String,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Model to use
    pub model: String,
    /// Request timestamp
    pub timestamp: Instant,
    /// Response sender
    pub response_sender: tokio::sync::oneshot::Sender<BatchedResponse>,
}

/// Batched inference response
#[derive(Debug, Clone)]
pub struct BatchedResponse {
    /// Request identifier
    pub request_id: String,
    /// Generated text
    pub text: String,
    /// Number of tokens generated
    pub tokens_generated: u64,
    /// Processing time
    pub processing_time_ms: u64,
    /// Batch size when processed
    pub batch_size: usize,
}

/// Request batch for processing
#[derive(Debug)]
pub struct RequestBatch {
    /// Batch identifier
    pub id: String,
    /// Requests in the batch
    pub requests: Vec<BatchedRequest>,
    /// Batch creation time
    pub created_at: Instant,
    /// Target model for the batch
    pub model: String,
}

/// Request batcher with intelligent batching
pub struct RequestBatcher {
    config: CachingConfig,
    request_queue: Arc<RwLock<VecDeque<BatchedRequest>>>,
    batch_sender: mpsc::UnboundedSender<RequestBatch>,
    batch_receiver: Arc<RwLock<Option<mpsc::UnboundedReceiver<RequestBatch>>>>,
    processing_semaphore: Arc<Semaphore>,
    statistics: Arc<RwLock<BatchingStatistics>>,
}

/// Batching statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct BatchingStatistics {
    pub total_requests: u64,
    pub total_batches: u64,
    pub average_batch_size: f64,
    pub average_wait_time_ms: f64,
    pub average_processing_time_ms: f64,
    pub throughput_requests_per_second: f64,
    pub queue_depth: usize,
    pub batching_efficiency: f64,
    pub timeout_rate: f64,
}

impl Default for BatchingStatistics {
    fn default() -> Self {
        Self {
            total_requests: 0,
            total_batches: 0,
            average_batch_size: 0.0,
            average_wait_time_ms: 0.0,
            average_processing_time_ms: 0.0,
            throughput_requests_per_second: 0.0,
            queue_depth: 0,
            batching_efficiency: 0.0,
            timeout_rate: 0.0,
        }
    }
}

impl RequestBatcher {
    /// Create a new request batcher
    pub async fn new(config: &CachingConfig) -> Result<Self> {
        let (batch_sender, batch_receiver) = mpsc::unbounded_channel();
        let processing_semaphore = Arc::new(Semaphore::new(config.max_batch_size));

        Ok(Self {
            config: config.clone(),
            request_queue: Arc::new(RwLock::new(VecDeque::new())),
            batch_sender,
            batch_receiver: Arc::new(RwLock::new(Some(batch_receiver))),
            processing_semaphore,
            statistics: Arc::new(RwLock::new(BatchingStatistics::default())),
        })
    }

    /// Submit a request for batching
    pub async fn submit_request(
        &self,
        prompt: String,
        max_tokens: usize,
        model: String,
    ) -> Result<BatchedResponse> {
        let (response_sender, response_receiver) = tokio::sync::oneshot::channel();

        let request = BatchedRequest {
            id: Uuid::new_v4().to_string(),
            prompt,
            max_tokens,
            model,
            timestamp: Instant::now(),
            response_sender,
        };

        // Add to queue
        {
            let mut queue = self.request_queue.write().await;
            queue.push_back(request);
        }

        // Update statistics
        {
            let mut stats = self.statistics.write().await;
            stats.total_requests += 1;
            stats.queue_depth = self.request_queue.read().await.len();
        }

        // Wait for response
        match tokio::time::timeout(Duration::from_secs(30), response_receiver).await {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(_)) => Err(anyhow::anyhow!("Request processing failed")),
            Err(_) => Err(anyhow::anyhow!("Request timeout")),
        }
    }

    /// Start the batching task
    pub async fn start_batching_task(&self) {
        let request_queue = self.request_queue.clone();
        let batch_sender = self.batch_sender.clone();
        let statistics = self.statistics.clone();
        let config = self.config.clone();

        // Start batch formation task
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(config.batch_timeout_ms));

            loop {
                interval.tick().await;

                // Form batches from the queue
                let batch = Self::form_batch(&request_queue, &config).await;

                if let Some(batch) = batch {
                    // Update statistics
                    {
                        let mut stats = statistics.write().await;
                        stats.total_batches += 1;
                        let total_batch_size = stats.average_batch_size * (stats.total_batches - 1) as f64;
                        stats.average_batch_size = (total_batch_size + batch.requests.len() as f64) / stats.total_batches as f64;
                        stats.queue_depth = request_queue.read().await.len();
                    }

                    // Send batch for processing
                    if batch_sender.send(batch).is_err() {
                        break;
                    }
                }
            }
        });

        // Start batch processing task
        let batch_receiver = {
            let mut receiver_guard = self.batch_receiver.write().await;
            receiver_guard.take()
        };

        if let Some(mut batch_receiver) = batch_receiver {
            let processing_semaphore = self.processing_semaphore.clone();
            let statistics = self.statistics.clone();

            tokio::spawn(async move {
                while let Some(batch) = batch_receiver.recv().await {
                    let permit = processing_semaphore.acquire().await.unwrap();
                    let statistics = statistics.clone();

                    tokio::spawn(async move {
                        let _permit = permit; // Keep permit until task completes
                        Self::process_batch(batch, statistics).await;
                    });
                }
            });
        }
    }

    /// Form a batch from the request queue
    async fn form_batch(
        request_queue: &Arc<RwLock<VecDeque<BatchedRequest>>>,
        config: &CachingConfig,
    ) -> Option<RequestBatch> {
        let mut queue = request_queue.write().await;

        if queue.is_empty() {
            return None;
        }

        let mut batch_requests = Vec::new();
        let mut target_model = None;
        let now = Instant::now();

        // Group requests by model and respect timeout
        while let Some(request) = queue.front() {
            // Check if request has timed out
            if now.duration_since(request.timestamp) > Duration::from_millis(config.batch_timeout_ms * 2) {
                // Remove timed out request
                if let Some(timed_out) = queue.pop_front() {
                    let _ = timed_out.response_sender.send(BatchedResponse {
                        request_id: timed_out.id,
                        text: "Request timed out".to_string(),
                        tokens_generated: 0,
                        processing_time_ms: now.duration_since(timed_out.timestamp).as_millis() as u64,
                        batch_size: 0,
                    });
                }
                continue;
            }

            // Check if we can add this request to the batch
            if target_model.is_none() {
                target_model = Some(request.model.clone());
            }

            if target_model.as_ref() == Some(&request.model) && batch_requests.len() < config.max_batch_size {
                if let Some(request) = queue.pop_front() {
                    batch_requests.push(request);
                }
            } else {
                break;
            }
        }

        if batch_requests.is_empty() {
            return None;
        }

        Some(RequestBatch {
            id: Uuid::new_v4().to_string(),
            requests: batch_requests,
            created_at: Instant::now(),
            model: target_model.unwrap_or_else(|| "default".to_string()),
        })
    }

    /// Process a batch of requests
    async fn process_batch(batch: RequestBatch, statistics: Arc<RwLock<BatchingStatistics>>) {
        let start_time = Instant::now();
        let batch_size = batch.requests.len();

        // Simulate batch processing (in real implementation, this would call the inference engine)
        let processing_time = Duration::from_millis(50 + (batch_size as u64 * 10)); // Simulate processing time
        tokio::time::sleep(processing_time).await;

        let processing_time_ms = start_time.elapsed().as_millis() as u64;

        // Send responses to all requests in the batch
        for request in batch.requests {
            let wait_time_ms = start_time.duration_since(request.timestamp).as_millis() as u64;

            let response = BatchedResponse {
                request_id: request.id,
                text: format!("Generated response for: {}", request.prompt),
                tokens_generated: request.max_tokens as u64,
                processing_time_ms,
                batch_size,
            };

            let _ = request.response_sender.send(response);

            // Update wait time statistics
            {
                let mut stats = statistics.write().await;
                let total_wait_time = stats.average_wait_time_ms * (stats.total_requests - 1) as f64;
                stats.average_wait_time_ms = (total_wait_time + wait_time_ms as f64) / stats.total_requests as f64;
            }
        }

        // Update processing time statistics
        {
            let mut stats = statistics.write().await;
            let total_processing_time = stats.average_processing_time_ms * (stats.total_batches - 1) as f64;
            stats.average_processing_time_ms = (total_processing_time + processing_time_ms as f64) / stats.total_batches as f64;

            // Calculate batching efficiency (higher is better)
            stats.batching_efficiency = stats.average_batch_size / batch_size as f64;

            // Calculate throughput
            if stats.average_processing_time_ms > 0.0 {
                stats.throughput_requests_per_second = (stats.average_batch_size * 1000.0) / stats.average_processing_time_ms;
            }
        }
    }

    /// Get batching statistics
    pub async fn get_statistics(&self) -> BatchingStatistics {
        let mut stats = self.statistics.read().await.clone();
        stats.queue_depth = self.request_queue.read().await.len();
        stats
    }

    /// Shutdown the request batcher
    pub async fn shutdown(&self) -> Result<()> {
        println!("Shutting down request batcher");

        // Process remaining requests in queue
        let remaining_requests: Vec<BatchedRequest> = {
            let mut queue = self.request_queue.write().await;
            queue.drain(..).collect()
        };

        // Send timeout responses to remaining requests
        for request in remaining_requests {
            let _ = request.response_sender.send(BatchedResponse {
                request_id: request.id,
                text: "Server shutting down".to_string(),
                tokens_generated: 0,
                processing_time_ms: 0,
                batch_size: 0,
            });
        }

        Ok(())
    }
}

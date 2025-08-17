//! Batch processing for inference

use crate::InferenceEngine;
use bitnet_common::{BitNetError, GenerationConfig, PerformanceMetrics, Result};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock, Semaphore};


/// Batch inference request
#[derive(Debug, Clone)]
pub struct BatchRequest {
    pub id: String,
    pub prompt: String,
    pub config: GenerationConfig,
    pub priority: Priority,
}

/// Request priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Batch inference response
#[derive(Debug)]
pub struct BatchResponse {
    pub id: String,
    pub result: Result<String>,
    pub metrics: PerformanceMetrics,
    pub processing_time: Duration,
}

/// Batch processor for handling multiple inference requests
pub struct BatchProcessor {
    engine: Arc<RwLock<Box<dyn InferenceEngine>>>,
    request_queue: Arc<RwLock<VecDeque<BatchRequest>>>,
    response_sender: mpsc::UnboundedSender<BatchResponse>,
    config: BatchProcessorConfig,
    semaphore: Arc<Semaphore>,
    is_running: Arc<RwLock<bool>>,
}

impl BatchProcessor {
    /// Create a new batch processor
    pub fn new(
        engine: Box<dyn InferenceEngine>,
        config: BatchProcessorConfig,
    ) -> Result<(Self, mpsc::UnboundedReceiver<BatchResponse>)> {
        config.validate()?;
        
        let (response_sender, response_receiver) = mpsc::unbounded_channel();
        let semaphore = Arc::new(Semaphore::new(config.max_concurrent_requests));
        
        let processor = Self {
            engine: Arc::new(RwLock::new(engine)),
            request_queue: Arc::new(RwLock::new(VecDeque::new())),
            response_sender,
            config,
            semaphore,
            is_running: Arc::new(RwLock::new(false)),
        };
        
        Ok((processor, response_receiver))
    }
    
    /// Start the batch processor
    pub async fn start(&self) -> Result<()> {
        let mut is_running = self.is_running.write().await;
        if *is_running {
            return Err(BitNetError::Validation(
                "Batch processor is already running".to_string()
            ));
        }
        *is_running = true;
        drop(is_running);
        
        // Start processing loop
        let engine = self.engine.clone();
        let request_queue = self.request_queue.clone();
        let response_sender = self.response_sender.clone();
        let config = self.config.clone();
        let semaphore = self.semaphore.clone();
        let is_running = self.is_running.clone();
        
        crate::rt::task::spawn(async move {
            Self::processing_loop(
                engine,
                request_queue,
                response_sender,
                config,
                semaphore,
                is_running,
            ).await;
        });
        
        Ok(())
    }
    
    /// Stop the batch processor
    pub async fn stop(&self) -> Result<()> {
        let mut is_running = self.is_running.write().await;
        *is_running = false;
        Ok(())
    }
    
    /// Submit a request for batch processing
    pub async fn submit_request(&self, request: BatchRequest) -> Result<()> {
        let mut queue = self.request_queue.write().await;
        
        // Check queue capacity
        if queue.len() >= self.config.max_queue_size {
            return Err(BitNetError::Validation(
                "Request queue is full".to_string()
            ));
        }
        
        // Insert request based on priority
        let insert_pos = queue
            .iter()
            .position(|r| r.priority < request.priority)
            .unwrap_or(queue.len());
        
        queue.insert(insert_pos, request);
        Ok(())
    }
    
    /// Get queue statistics
    pub async fn queue_stats(&self) -> QueueStats {
        let queue = self.request_queue.read().await;
        let available_permits = self.semaphore.available_permits();
        
        QueueStats {
            queue_length: queue.len(),
            max_queue_size: self.config.max_queue_size,
            active_requests: self.config.max_concurrent_requests - available_permits,
            max_concurrent_requests: self.config.max_concurrent_requests,
        }
    }
    
    /// Main processing loop
    async fn processing_loop(
        engine: Arc<RwLock<Box<dyn InferenceEngine>>>,
        request_queue: Arc<RwLock<VecDeque<BatchRequest>>>,
        response_sender: mpsc::UnboundedSender<BatchResponse>,
        config: BatchProcessorConfig,
        semaphore: Arc<Semaphore>,
        is_running: Arc<RwLock<bool>>,
    ) {
        let mut batch_buffer = Vec::new();
        let mut last_batch_time = Instant::now();
        
        while *is_running.read().await {
            // Collect requests for batching
            {
                let mut queue = request_queue.write().await;
                while batch_buffer.len() < config.max_batch_size && !queue.is_empty() {
                    if let Some(request) = queue.pop_front() {
                        batch_buffer.push(request);
                    }
                }
            }
            
            // Process batch if we have requests or timeout reached
            let should_process = !batch_buffer.is_empty() && (
                batch_buffer.len() >= config.max_batch_size ||
                last_batch_time.elapsed() >= Duration::from_millis(config.batch_timeout_ms)
            );
            
            if should_process {
                Self::process_batch(
                    &engine,
                    &mut batch_buffer,
                    &response_sender,
                    &semaphore,
                ).await;
                last_batch_time = Instant::now();
            } else {
                // Small delay to prevent busy waiting
                crate::rt::time::sleep(Duration::from_millis(1)).await;
            }
        }
    }
    
    /// Process a batch of requests
    async fn process_batch(
        engine: &Arc<RwLock<Box<dyn InferenceEngine>>>,
        batch: &mut Vec<BatchRequest>,
        response_sender: &mpsc::UnboundedSender<BatchResponse>,
        semaphore: &Arc<Semaphore>,
    ) {
        let batch_requests = std::mem::take(batch);
        
        // Process requests concurrently
        let tasks: Vec<_> = batch_requests
            .into_iter()
            .map(|request| {
                let engine = engine.clone();
                let sender = response_sender.clone();
                let semaphore = semaphore.clone();
                
                crate::rt::task::spawn(async move {
                    let _permit = semaphore.acquire().await.unwrap();
                    Self::process_single_request(engine, request, sender).await;
                })
            })
            .collect();
        
        // Wait for all tasks to complete
        for task in tasks {
            let _ = task.await;
        }
    }
    
    /// Process a single request
    async fn process_single_request(
        engine: Arc<RwLock<Box<dyn InferenceEngine>>>,
        request: BatchRequest,
        response_sender: mpsc::UnboundedSender<BatchResponse>,
    ) {
        let start_time = Instant::now();
        
        let result = {
            let mut engine = engine.write().await;
            engine.generate(&request.prompt, &request.config)
        };
        
        let processing_time = start_time.elapsed();
        let metrics = {
            let engine = engine.read().await;
            engine.metrics().clone()
        };
        
        let response = BatchResponse {
            id: request.id,
            result,
            metrics,
            processing_time,
        };
        
        let _ = response_sender.send(response);
    }
}

/// Batch processor configuration
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
    pub fn validate(&self) -> Result<()> {
        if self.max_batch_size == 0 {
            return Err(BitNetError::Config(
                "max_batch_size must be greater than 0".to_string()
            ));
        }
        
        if self.max_queue_size == 0 {
            return Err(BitNetError::Config(
                "max_queue_size must be greater than 0".to_string()
            ));
        }
        
        if self.max_concurrent_requests == 0 {
            return Err(BitNetError::Config(
                "max_concurrent_requests must be greater than 0".to_string()
            ));
        }
        
        Ok(())
    }
}

/// Queue statistics
#[derive(Debug, Clone)]
pub struct QueueStats {
    pub queue_length: usize,
    pub max_queue_size: usize,
    pub active_requests: usize,
    pub max_concurrent_requests: usize,
}

impl QueueStats {
    /// Calculate queue utilization (0.0 to 1.0)
    pub fn queue_utilization(&self) -> f64 {
        if self.max_queue_size > 0 {
            self.queue_length as f64 / self.max_queue_size as f64
        } else {
            0.0
        }
    }
    
    /// Calculate processing utilization (0.0 to 1.0)
    pub fn processing_utilization(&self) -> f64 {
        if self.max_concurrent_requests > 0 {
            self.active_requests as f64 / self.max_concurrent_requests as f64
        } else {
            0.0
        }
    }
}

/// Dynamic batch sizing based on load
pub struct DynamicBatchSizer {
    current_batch_size: usize,
    min_batch_size: usize,
    max_batch_size: usize,
    target_latency_ms: u64,
    recent_latencies: VecDeque<u64>,
    adjustment_factor: f64,
}

impl DynamicBatchSizer {
    pub fn new(
        initial_batch_size: usize,
        min_batch_size: usize,
        max_batch_size: usize,
        target_latency_ms: u64,
    ) -> Self {
        Self {
            current_batch_size: initial_batch_size,
            min_batch_size,
            max_batch_size,
            target_latency_ms,
            recent_latencies: VecDeque::with_capacity(10),
            adjustment_factor: 0.1,
        }
    }
    
    /// Update batch size based on recent performance
    pub fn update(&mut self, latency_ms: u64) {
        self.recent_latencies.push_back(latency_ms);
        if self.recent_latencies.len() > 10 {
            self.recent_latencies.pop_front();
        }
        
        if self.recent_latencies.len() >= 3 {
            let avg_latency: f64 = self.recent_latencies.iter().sum::<u64>() as f64 
                / self.recent_latencies.len() as f64;
            
            if avg_latency > self.target_latency_ms as f64 * 1.2 {
                // Latency too high, reduce batch size
                let new_size = (self.current_batch_size as f64 * (1.0 - self.adjustment_factor)) as usize;
                self.current_batch_size = new_size.max(self.min_batch_size);
            } else if avg_latency < self.target_latency_ms as f64 * 0.8 {
                // Latency acceptable, try increasing batch size
                let new_size = (self.current_batch_size as f64 * (1.0 + self.adjustment_factor)) as usize;
                self.current_batch_size = new_size.min(self.max_batch_size);
            }
        }
    }
    
    /// Get current recommended batch size
    pub fn current_batch_size(&self) -> usize {
        self.current_batch_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_batch_processor_config_validation() {
        let config = BatchProcessorConfig::default();
        assert!(config.validate().is_ok());
        
        let mut invalid_config = config.clone();
        invalid_config.max_batch_size = 0;
        assert!(invalid_config.validate().is_err());
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
    }
    
    #[test]
    fn test_dynamic_batch_sizer() {
        let mut sizer = DynamicBatchSizer::new(4, 1, 16, 100);
        assert_eq!(sizer.current_batch_size(), 4);
        
        // Simulate high latency
        for _ in 0..5 {
            sizer.update(200);
        }
        assert!(sizer.current_batch_size() < 4);
        
        // Simulate low latency
        for _ in 0..5 {
            sizer.update(50);
        }
        assert!(sizer.current_batch_size() >= 4);
    }
    
    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Critical > Priority::High);
        assert!(Priority::High > Priority::Normal);
        assert!(Priority::Normal > Priority::Low);
    }
}
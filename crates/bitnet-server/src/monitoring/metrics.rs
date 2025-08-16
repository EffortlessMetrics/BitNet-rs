//! Metrics collection and reporting for BitNet inference

use anyhow::Result;
use metrics::{counter, gauge, histogram, Counter, Gauge, Histogram};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

use super::MonitoringConfig;

/// Standard ML inference metrics
#[derive(Debug, Clone)]
pub struct InferenceMetrics {
    /// Total number of inference requests
    pub requests_total: Counter,
    /// Number of active inference requests
    pub requests_active: Gauge,
    /// Request duration histogram
    pub request_duration: Histogram,
    /// Tokens generated per second
    pub tokens_per_second: Gauge,
    /// Total tokens generated
    pub tokens_generated_total: Counter,
    /// Model loading time
    pub model_load_duration: Histogram,
    /// Memory usage in bytes
    pub memory_usage_bytes: Gauge,
    /// GPU memory usage in bytes (if available)
    pub gpu_memory_usage_bytes: Gauge,
    /// Error count by type
    pub errors_total: Counter,
    /// Queue depth for batched requests
    pub queue_depth: Gauge,
    /// Cache hit rate
    pub cache_hit_rate: Gauge,
}

impl Default for InferenceMetrics {
    fn default() -> Self {
        Self {
            requests_total: counter!("bitnet_requests_total"),
            requests_active: gauge!("bitnet_requests_active"),
            request_duration: histogram!("bitnet_request_duration_seconds"),
            tokens_per_second: gauge!("bitnet_tokens_per_second"),
            tokens_generated_total: counter!("bitnet_tokens_generated_total"),
            model_load_duration: histogram!("bitnet_model_load_duration_seconds"),
            memory_usage_bytes: gauge!("bitnet_memory_usage_bytes"),
            gpu_memory_usage_bytes: gauge!("bitnet_gpu_memory_usage_bytes"),
            errors_total: counter!("bitnet_errors_total"),
            queue_depth: gauge!("bitnet_queue_depth"),
            cache_hit_rate: gauge!("bitnet_cache_hit_rate"),
        }
    }
}

impl InferenceMetrics {
    pub fn new() -> Self {
        Self::default()
    }
}

/// System-level metrics
#[derive(Debug, Clone)]
pub struct SystemMetrics {
    /// CPU usage percentage
    pub cpu_usage_percent: Gauge,
    /// Memory usage percentage
    pub memory_usage_percent: Gauge,
    /// Disk usage percentage
    pub disk_usage_percent: Gauge,
    /// Network bytes received
    pub network_bytes_received: Counter,
    /// Network bytes sent
    pub network_bytes_sent: Counter,
    /// Process uptime in seconds
    pub uptime_seconds: Gauge,
}

impl Default for SystemMetrics {
    fn default() -> Self {
        Self {
            cpu_usage_percent: gauge!("system_cpu_usage_percent"),
            memory_usage_percent: gauge!("system_memory_usage_percent"),
            disk_usage_percent: gauge!("system_disk_usage_percent"),
            network_bytes_received: counter!("system_network_bytes_received_total"),
            network_bytes_sent: counter!("system_network_bytes_sent_total"),
            uptime_seconds: gauge!("system_uptime_seconds"),
        }
    }
}

impl SystemMetrics {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Request tracking for performance monitoring
#[derive(Debug)]
pub struct RequestTracker {
    start_time: Instant,
    request_id: String,
    metrics: Arc<InferenceMetrics>,
}

impl RequestTracker {
    pub fn new(request_id: String, metrics: Arc<InferenceMetrics>) -> Self {
        metrics.requests_active.increment(1.0);
        metrics.requests_total.increment(1);

        Self { start_time: Instant::now(), request_id, metrics }
    }

    pub fn record_tokens(&self, token_count: u64) {
        self.metrics.tokens_generated_total.increment(token_count);

        let duration = self.start_time.elapsed();
        if duration.as_secs() > 0 {
            let tokens_per_sec = token_count as f64 / duration.as_secs_f64();
            self.metrics.tokens_per_second.set(tokens_per_sec);
        }
    }

    pub fn record_error(&self, error_type: &str) {
        self.metrics.errors_total.increment(1);
        tracing::error!(
            request_id = %self.request_id,
            error_type = error_type,
            "Request failed"
        );
    }
}

impl Drop for RequestTracker {
    fn drop(&mut self) {
        let duration = self.start_time.elapsed();
        self.metrics.request_duration.record(duration.as_secs_f64());
        self.metrics.requests_active.decrement(1.0);

        tracing::info!(
            request_id = %self.request_id,
            duration_ms = duration.as_millis(),
            "Request completed"
        );
    }
}

/// Central metrics collector
pub struct MetricsCollector {
    pub inference: Arc<InferenceMetrics>,
    pub system: Arc<SystemMetrics>,
    start_time: Instant,
    #[allow(dead_code)]
    config: MonitoringConfig,
    performance_history: Arc<RwLock<Vec<PerformanceSnapshot>>>,
}

#[derive(Debug, Clone)]
pub struct PerformanceSnapshot {
    pub timestamp: Instant,
    pub tokens_per_second: f64,
    pub memory_usage_mb: f64,
    pub active_requests: f64,
    pub error_rate: f64,
}

impl MetricsCollector {
    pub fn new(config: &MonitoringConfig) -> Result<Self> {
        Ok(Self {
            inference: Arc::new(InferenceMetrics::new()),
            system: Arc::new(SystemMetrics::new()),
            start_time: Instant::now(),
            config: config.clone(),
            performance_history: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Create a new request tracker
    pub fn track_request(&self, request_id: String) -> RequestTracker {
        RequestTracker::new(request_id, self.inference.clone())
    }

    /// Record model loading time
    pub fn record_model_load_time(&self, duration: Duration) {
        self.inference.model_load_duration.record(duration.as_secs_f64());
        tracing::info!(duration_ms = duration.as_millis(), "Model loaded");
    }

    /// Update queue depth
    pub fn update_queue_depth(&self, depth: usize) {
        self.inference.queue_depth.set(depth as f64);
    }

    /// Update cache hit rate
    pub fn update_cache_hit_rate(&self, hit_rate: f64) {
        self.inference.cache_hit_rate.set(hit_rate);
    }

    /// Collect system-level metrics
    pub async fn collect_system_metrics(&self) -> Result<()> {
        // Update uptime
        let uptime = self.start_time.elapsed().as_secs() as f64;
        self.system.uptime_seconds.set(uptime);

        // Collect memory usage
        if let Ok(memory_info) = self.get_memory_info().await {
            self.inference.memory_usage_bytes.set(memory_info.used_bytes as f64);
            self.system.memory_usage_percent.set(memory_info.usage_percent);
        }

        // Collect CPU usage (simplified - in production you'd use a proper system monitoring crate)
        if let Ok(cpu_usage) = self.get_cpu_usage().await {
            self.system.cpu_usage_percent.set(cpu_usage);
        }

        // Store performance snapshot
        let snapshot = PerformanceSnapshot {
            timestamp: Instant::now(),
            tokens_per_second: 0.0, // This would be calculated from recent history
            memory_usage_mb: 0.0,   // This would be from actual memory usage
            active_requests: 0.0,   // This would be from the gauge
            error_rate: 0.0,        // This would be calculated from error counts
        };

        let mut history = self.performance_history.write().await;
        history.push(snapshot);

        // Keep only last 1000 snapshots
        if history.len() > 1000 {
            history.remove(0);
        }

        Ok(())
    }

    /// Get performance regression alerts
    pub async fn check_performance_regression(&self) -> Result<Vec<String>> {
        let history = self.performance_history.read().await;
        let mut alerts = Vec::new();

        if history.len() < 10 {
            return Ok(alerts);
        }

        // Check for significant performance degradation
        let recent_avg =
            history.iter().rev().take(5).map(|s| s.tokens_per_second).sum::<f64>() / 5.0;

        let baseline_avg =
            history.iter().rev().skip(5).take(5).map(|s| s.tokens_per_second).sum::<f64>() / 5.0;

        if baseline_avg > 0.0 && recent_avg < baseline_avg * 0.95 {
            alerts.push(format!(
                "Performance regression detected: {:.2} -> {:.2} tokens/sec ({:.1}% decrease)",
                baseline_avg,
                recent_avg,
                (1.0 - recent_avg / baseline_avg) * 100.0
            ));
        }

        // Check for high error rate
        let recent_error_rate =
            history.iter().rev().take(5).map(|s| s.error_rate).sum::<f64>() / 5.0;

        if recent_error_rate > 0.05 {
            alerts.push(format!("High error rate detected: {:.2}%", recent_error_rate * 100.0));
        }

        Ok(alerts)
    }

    async fn get_memory_info(&self) -> Result<MemoryInfo> {
        // Simplified memory info - in production use a proper system monitoring crate
        Ok(MemoryInfo { used_bytes: 0, total_bytes: 0, usage_percent: 0.0 })
    }

    async fn get_cpu_usage(&self) -> Result<f64> {
        // Simplified CPU usage - in production use a proper system monitoring crate
        Ok(0.0)
    }
}

#[derive(Debug)]
struct MemoryInfo {
    used_bytes: u64,
    #[allow(dead_code)]
    total_bytes: u64,
    usage_percent: f64,
}

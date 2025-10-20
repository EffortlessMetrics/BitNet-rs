//! GPU health monitoring for AC05
//!
//! This module provides GPU metrics collection for health monitoring endpoints.
//! Monitors GPU utilization, memory usage, temperature, and CUDA errors.
//!
//! # Feature Gates
//!
//! GPU monitoring is only available when compiled with `feature = "gpu"` or `feature = "cuda"`.
//!
//! # Example
//!
//! ```rust,no_run
//! use bitnet_server::health::gpu_monitor::GpuMetrics;
//!
//! # #[cfg(any(feature = "gpu", feature = "cuda"))]
//! async fn example() {
//!     let metrics = GpuMetrics::collect().await;
//!     println!("GPU Utilization: {}%", metrics.utilization_percent);
//!     println!("GPU Memory: {} MB / {} MB", metrics.memory_used_mb, metrics.memory_total_mb);
//! }
//! ```

use serde::{Deserialize, Serialize};

#[cfg(any(feature = "gpu", feature = "cuda"))]
use std::process::Command;
#[cfg(any(feature = "gpu", feature = "cuda"))]
use std::sync::Arc;
#[cfg(any(feature = "gpu", feature = "cuda"))]
use std::time::Instant;
#[cfg(any(feature = "gpu", feature = "cuda"))]
use tokio::sync::RwLock;

/// GPU metrics collected from CUDA runtime
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMetrics {
    /// GPU utilization percentage (0.0 - 100.0)
    pub utilization_percent: f64,
    /// GPU memory used in MB
    pub memory_used_mb: f64,
    /// GPU memory total in MB
    pub memory_total_mb: f64,
    /// GPU temperature in Celsius
    pub temperature_celsius: f64,
    /// Whether CUDA errors were detected
    pub has_cuda_errors: bool,
    /// Optional error message if collection failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
}

impl GpuMetrics {
    /// Collect GPU metrics using nvidia-smi (when available)
    ///
    /// This is a minimal implementation that uses nvidia-smi CLI to query GPU stats.
    /// For production, this could be optimized with direct CUDA API calls.
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    pub async fn collect() -> Self {
        // Use nvidia-smi to query GPU metrics
        // Format: utilization.gpu,memory.used,memory.total,temperature.gpu
        let result = tokio::task::spawn_blocking(|| {
            Command::new("nvidia-smi")
                .args([
                    "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ])
                .output()
        })
        .await;

        match result {
            Ok(Ok(output)) if output.status.success() => {
                // Parse nvidia-smi output
                let stdout = String::from_utf8_lossy(&output.stdout);
                Self::parse_nvidia_smi_output(&stdout)
            }
            Ok(Ok(output)) => {
                // nvidia-smi failed
                let stderr = String::from_utf8_lossy(&output.stderr);
                Self::error_metrics(&format!("nvidia-smi failed: {}", stderr))
            }
            Ok(Err(e)) => {
                // Failed to execute nvidia-smi
                Self::error_metrics(&format!("Failed to execute nvidia-smi: {}", e))
            }
            Err(e) => {
                // Task join error
                Self::error_metrics(&format!("Task execution error: {}", e))
            }
        }
    }

    /// Parse nvidia-smi CSV output
    ///
    /// Expected format: "utilization, memory_used, memory_total, temperature"
    /// Example: "45, 2048, 8192, 65"
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn parse_nvidia_smi_output(output: &str) -> Self {
        let line = output.trim();
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();

        if parts.len() != 4 {
            return Self::error_metrics(&format!(
                "Unexpected nvidia-smi output format (expected 4 fields, got {}): {}",
                parts.len(),
                line
            ));
        }

        // Parse each field with error handling
        let utilization = parts[0].parse::<f64>().unwrap_or(0.0);
        let memory_used = parts[1].parse::<f64>().unwrap_or(0.0);
        let memory_total = parts[2].parse::<f64>().unwrap_or(0.0);
        let temperature = parts[3].parse::<f64>().unwrap_or(0.0);

        Self {
            utilization_percent: utilization,
            memory_used_mb: memory_used,
            memory_total_mb: memory_total,
            temperature_celsius: temperature,
            has_cuda_errors: false,
            error_message: None,
        }
    }

    /// Create error metrics with default values
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn error_metrics(error: &str) -> Self {
        Self {
            utilization_percent: 0.0,
            memory_used_mb: 0.0,
            memory_total_mb: 0.0,
            temperature_celsius: 0.0,
            has_cuda_errors: true,
            error_message: Some(error.to_string()),
        }
    }

    /// Stub implementation when GPU not compiled
    #[cfg(not(any(feature = "gpu", feature = "cuda")))]
    pub async fn collect() -> Self {
        Self {
            utilization_percent: 0.0,
            memory_used_mb: 0.0,
            memory_total_mb: 0.0,
            temperature_celsius: 0.0,
            has_cuda_errors: false,
            error_message: Some("GPU support not compiled".to_string()),
        }
    }

    /// Check if GPU metrics are healthy
    ///
    /// Returns true if:
    /// - No CUDA errors
    /// - Memory usage is under 95%
    /// - Temperature is under 90°C
    pub fn is_healthy(&self) -> bool {
        if self.has_cuda_errors {
            return false;
        }

        // Check memory usage (< 95%)
        if self.memory_total_mb > 0.0 {
            let memory_usage_percent = (self.memory_used_mb / self.memory_total_mb) * 100.0;
            if memory_usage_percent >= 95.0 {
                return false;
            }
        }

        // Check temperature (< 90°C)
        if self.temperature_celsius >= 90.0 {
            return false;
        }

        true
    }

    /// Get health status message
    pub fn health_message(&self) -> String {
        if let Some(error) = &self.error_message {
            return format!("GPU monitoring error: {}", error);
        }

        if !self.is_healthy() {
            if self.has_cuda_errors {
                return "CUDA errors detected".to_string();
            }

            let memory_usage_percent = if self.memory_total_mb > 0.0 {
                (self.memory_used_mb / self.memory_total_mb) * 100.0
            } else {
                0.0
            };

            if memory_usage_percent >= 95.0 {
                return format!("High GPU memory usage: {:.1}%", memory_usage_percent);
            }

            if self.temperature_celsius >= 90.0 {
                return format!("High GPU temperature: {:.1}°C", self.temperature_celsius);
            }
        }

        let memory_usage_percent = if self.memory_total_mb > 0.0 {
            (self.memory_used_mb / self.memory_total_mb) * 100.0
        } else {
            0.0
        };

        format!(
            "GPU healthy: {:.1}% utilization, {:.1}% memory, {:.1}°C",
            self.utilization_percent, memory_usage_percent, self.temperature_celsius
        )
    }
}

/// Memory sample for leak detection
#[cfg(any(feature = "gpu", feature = "cuda"))]
#[derive(Debug, Clone)]
pub struct MemorySample {
    /// Timestamp when sample was taken
    pub timestamp: Instant,
    /// Memory used in MB
    pub memory_used_mb: f64,
    /// Memory total in MB
    pub memory_total_mb: f64,
}

/// GPU memory leak detection status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMemoryLeakStatus {
    /// Whether a potential leak was detected
    pub leak_detected: bool,
    /// Memory growth rate in MB/minute
    pub growth_rate_mb_per_min: f64,
    /// Number of samples collected
    pub sample_count: usize,
    /// Current memory usage in MB
    pub current_memory_mb: f64,
    /// Baseline memory usage in MB (first sample)
    pub baseline_memory_mb: f64,
    /// Memory trend data (recent samples)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub memory_trend: Vec<f64>,
    /// Optional warning message
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warning: Option<String>,
}

/// GPU memory leak detector
///
/// Tracks GPU memory usage over time and detects abnormal growth patterns
/// indicating potential memory leaks.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub struct GpuMemoryLeakDetector {
    /// Memory samples collected over time (limited to last N samples)
    samples: Arc<RwLock<Vec<MemorySample>>>,
    /// Maximum samples to retain
    max_samples: usize,
    /// Leak detection threshold in MB/minute
    leak_threshold_mb_per_min: f64,
}

#[cfg(any(feature = "gpu", feature = "cuda"))]
impl Default for GpuMemoryLeakDetector {
    fn default() -> Self {
        Self::new(60, 10.0)
    }
}

#[cfg(any(feature = "gpu", feature = "cuda"))]
impl GpuMemoryLeakDetector {
    /// Create a new GPU memory leak detector
    ///
    /// # Arguments
    /// * `max_samples` - Maximum number of samples to retain (default: 60 for ~1 hour at 1min intervals)
    /// * `leak_threshold_mb_per_min` - Growth rate threshold in MB/minute to flag as leak (default: 10.0)
    pub fn new(max_samples: usize, leak_threshold_mb_per_min: f64) -> Self {
        Self {
            samples: Arc::new(RwLock::new(Vec::with_capacity(max_samples))),
            max_samples,
            leak_threshold_mb_per_min,
        }
    }

    /// Record a memory sample from current GPU metrics
    pub async fn record_sample(&self, metrics: &GpuMetrics) {
        let sample = MemorySample {
            timestamp: Instant::now(),
            memory_used_mb: metrics.memory_used_mb,
            memory_total_mb: metrics.memory_total_mb,
        };

        let mut samples = self.samples.write().await;
        samples.push(sample);

        // Trim old samples to maintain max_samples limit
        if samples.len() > self.max_samples {
            samples.drain(0..(samples.len() - self.max_samples));
        }
    }

    /// Analyze memory samples and detect potential leaks
    pub async fn analyze(&self) -> GpuMemoryLeakStatus {
        let samples = self.samples.read().await;

        if samples.is_empty() {
            return GpuMemoryLeakStatus {
                leak_detected: false,
                growth_rate_mb_per_min: 0.0,
                sample_count: 0,
                current_memory_mb: 0.0,
                baseline_memory_mb: 0.0,
                memory_trend: Vec::new(),
                warning: Some("No samples collected yet".to_string()),
            };
        }

        // Need at least 2 samples to detect growth
        if samples.len() < 2 {
            return GpuMemoryLeakStatus {
                leak_detected: false,
                growth_rate_mb_per_min: 0.0,
                sample_count: samples.len(),
                current_memory_mb: samples[0].memory_used_mb,
                baseline_memory_mb: samples[0].memory_used_mb,
                memory_trend: vec![samples[0].memory_used_mb],
                warning: Some("Insufficient samples for leak detection (need ≥2)".to_string()),
            };
        }

        // Calculate linear regression to estimate growth rate
        let first_sample = &samples[0];
        let last_sample = &samples[samples.len() - 1];
        let time_span_secs =
            last_sample.timestamp.duration_since(first_sample.timestamp).as_secs_f64();

        if time_span_secs < 1.0 {
            // Samples too close together
            return GpuMemoryLeakStatus {
                leak_detected: false,
                growth_rate_mb_per_min: 0.0,
                sample_count: samples.len(),
                current_memory_mb: last_sample.memory_used_mb,
                baseline_memory_mb: first_sample.memory_used_mb,
                memory_trend: samples.iter().map(|s| s.memory_used_mb).collect(),
                warning: Some("Samples too close in time for reliable analysis".to_string()),
            };
        }

        // Calculate growth rate in MB/minute
        let memory_delta_mb = last_sample.memory_used_mb - first_sample.memory_used_mb;
        let growth_rate_mb_per_min = (memory_delta_mb / time_span_secs) * 60.0;

        // Detect leak if growth rate exceeds threshold AND memory is increasing
        let leak_detected = growth_rate_mb_per_min > self.leak_threshold_mb_per_min;

        // Build memory trend (recent samples, max 10)
        let trend_start = samples.len().saturating_sub(10);
        let memory_trend: Vec<f64> =
            samples[trend_start..].iter().map(|s| s.memory_used_mb).collect();

        let warning = if leak_detected {
            Some(format!(
                "Potential memory leak: growing at {:.2} MB/min (threshold: {:.2} MB/min)",
                growth_rate_mb_per_min, self.leak_threshold_mb_per_min
            ))
        } else {
            None
        };

        GpuMemoryLeakStatus {
            leak_detected,
            growth_rate_mb_per_min,
            sample_count: samples.len(),
            current_memory_mb: last_sample.memory_used_mb,
            baseline_memory_mb: first_sample.memory_used_mb,
            memory_trend,
            warning,
        }
    }

    /// Clear all collected samples (useful for testing or reset)
    pub async fn reset(&self) {
        let mut samples = self.samples.write().await;
        samples.clear();
    }
}

/// Stub implementation when GPU not compiled
#[cfg(not(any(feature = "gpu", feature = "cuda")))]
#[derive(Default)]
pub struct GpuMemoryLeakDetector;

#[cfg(not(any(feature = "gpu", feature = "cuda")))]
impl GpuMemoryLeakDetector {
    pub async fn record_sample(&self, _metrics: &GpuMetrics) {
        // No-op for CPU-only builds
    }

    pub async fn analyze(&self) -> GpuMemoryLeakStatus {
        GpuMemoryLeakStatus {
            leak_detected: false,
            growth_rate_mb_per_min: 0.0,
            sample_count: 0,
            current_memory_mb: 0.0,
            baseline_memory_mb: 0.0,
            memory_trend: Vec::new(),
            warning: Some("GPU support not compiled".to_string()),
        }
    }

    pub async fn reset(&self) {
        // No-op for CPU-only builds
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn test_parse_nvidia_smi_output() {
        let output = "45, 2048, 8192, 65";
        let metrics = GpuMetrics::parse_nvidia_smi_output(output);

        assert_eq!(metrics.utilization_percent, 45.0);
        assert_eq!(metrics.memory_used_mb, 2048.0);
        assert_eq!(metrics.memory_total_mb, 8192.0);
        assert_eq!(metrics.temperature_celsius, 65.0);
        assert!(!metrics.has_cuda_errors);
        assert!(metrics.error_message.is_none());
    }

    #[test]
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn test_parse_invalid_output() {
        let output = "invalid output";
        let metrics = GpuMetrics::parse_nvidia_smi_output(output);

        assert!(metrics.has_cuda_errors);
        assert!(metrics.error_message.is_some());
    }

    #[test]
    fn test_is_healthy_normal() {
        let metrics = GpuMetrics {
            utilization_percent: 50.0,
            memory_used_mb: 4096.0,
            memory_total_mb: 8192.0,
            temperature_celsius: 65.0,
            has_cuda_errors: false,
            error_message: None,
        };

        assert!(metrics.is_healthy());
    }

    #[test]
    fn test_is_healthy_high_memory() {
        let metrics = GpuMetrics {
            utilization_percent: 50.0,
            memory_used_mb: 7800.0, // 95.2% of 8192
            memory_total_mb: 8192.0,
            temperature_celsius: 65.0,
            has_cuda_errors: false,
            error_message: None,
        };

        assert!(!metrics.is_healthy());
    }

    #[test]
    fn test_is_healthy_high_temperature() {
        let metrics = GpuMetrics {
            utilization_percent: 50.0,
            memory_used_mb: 4096.0,
            memory_total_mb: 8192.0,
            temperature_celsius: 91.0,
            has_cuda_errors: false,
            error_message: None,
        };

        assert!(!metrics.is_healthy());
    }

    #[test]
    fn test_is_healthy_cuda_errors() {
        let metrics = GpuMetrics {
            utilization_percent: 50.0,
            memory_used_mb: 4096.0,
            memory_total_mb: 8192.0,
            temperature_celsius: 65.0,
            has_cuda_errors: true,
            error_message: Some("CUDA error".to_string()),
        };

        assert!(!metrics.is_healthy());
    }

    #[test]
    fn test_health_message() {
        let metrics = GpuMetrics {
            utilization_percent: 78.5,
            memory_used_mb: 2700.0,
            memory_total_mb: 8192.0,
            temperature_celsius: 72.0,
            has_cuda_errors: false,
            error_message: None,
        };

        let message = metrics.health_message();
        assert!(message.contains("GPU healthy"));
        assert!(message.contains("78.5"));
        assert!(message.contains("72.0"));
    }

    #[tokio::test]
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    async fn test_leak_detector_no_samples() {
        let detector = GpuMemoryLeakDetector::default();
        let status = detector.analyze().await;

        assert!(!status.leak_detected);
        assert_eq!(status.sample_count, 0);
        assert!(status.warning.is_some());
    }

    #[tokio::test]
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    async fn test_leak_detector_single_sample() {
        let detector = GpuMemoryLeakDetector::default();
        let metrics = GpuMetrics {
            utilization_percent: 50.0,
            memory_used_mb: 4096.0,
            memory_total_mb: 8192.0,
            temperature_celsius: 65.0,
            has_cuda_errors: false,
            error_message: None,
        };

        detector.record_sample(&metrics).await;
        let status = detector.analyze().await;

        assert!(!status.leak_detected);
        assert_eq!(status.sample_count, 1);
        assert_eq!(status.current_memory_mb, 4096.0);
        assert!(status.warning.is_some());
    }

    #[tokio::test]
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    async fn test_leak_detector_stable_memory() {
        let detector = GpuMemoryLeakDetector::new(10, 10.0);

        // Record samples with stable memory
        for _ in 0..5 {
            let metrics = GpuMetrics {
                utilization_percent: 50.0,
                memory_used_mb: 4096.0,
                memory_total_mb: 8192.0,
                temperature_celsius: 65.0,
                has_cuda_errors: false,
                error_message: None,
            };
            detector.record_sample(&metrics).await;
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        let status = detector.analyze().await;
        assert!(!status.leak_detected);
        assert_eq!(status.sample_count, 5);
        assert!(status.growth_rate_mb_per_min.abs() < 1.0); // Should be near zero
    }

    #[tokio::test]
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    async fn test_leak_detector_memory_growth() {
        // Use aggressive threshold for test
        let detector = GpuMemoryLeakDetector::new(10, 5.0);

        // Simulate memory leak: 100 MB growth per sample
        for i in 0..5 {
            let metrics = GpuMetrics {
                utilization_percent: 50.0,
                memory_used_mb: 4000.0 + (i as f64 * 100.0), // Growing memory
                memory_total_mb: 8192.0,
                temperature_celsius: 65.0,
                has_cuda_errors: false,
                error_message: None,
            };
            detector.record_sample(&metrics).await;
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        let status = detector.analyze().await;
        assert!(status.leak_detected, "Should detect leak with growing memory");
        assert!(status.growth_rate_mb_per_min > 5.0);
        assert_eq!(status.sample_count, 5);
        assert_eq!(status.baseline_memory_mb, 4000.0);
        assert_eq!(status.current_memory_mb, 4400.0);
        assert!(status.warning.is_some());
    }

    #[tokio::test]
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    async fn test_leak_detector_reset() {
        let detector = GpuMemoryLeakDetector::default();
        let metrics = GpuMetrics {
            utilization_percent: 50.0,
            memory_used_mb: 4096.0,
            memory_total_mb: 8192.0,
            temperature_celsius: 65.0,
            has_cuda_errors: false,
            error_message: None,
        };

        detector.record_sample(&metrics).await;
        assert_eq!(detector.analyze().await.sample_count, 1);

        detector.reset().await;
        assert_eq!(detector.analyze().await.sample_count, 0);
    }

    #[tokio::test]
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    async fn test_leak_detector_max_samples() {
        let detector = GpuMemoryLeakDetector::new(3, 10.0); // Only keep 3 samples

        // Record 5 samples
        for i in 0..5 {
            let metrics = GpuMetrics {
                utilization_percent: 50.0,
                memory_used_mb: 4000.0 + (i as f64 * 10.0),
                memory_total_mb: 8192.0,
                temperature_celsius: 65.0,
                has_cuda_errors: false,
                error_message: None,
            };
            detector.record_sample(&metrics).await;
        }

        let status = detector.analyze().await;
        assert_eq!(status.sample_count, 3, "Should only retain 3 samples");
        // Baseline should be from the 3rd sample (index 2), not the 1st
        assert_eq!(status.baseline_memory_mb, 4020.0);
    }
}

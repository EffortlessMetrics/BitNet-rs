use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

use crate::common::{TestError, TestResult};

/// Performance measurement utilities for testing
pub struct PerformanceMeasurement {
    /// Start time of the measurement
    pub start_time: Instant,
    /// End time of the measurement
    pub end_time: Option<Instant>,
    /// Memory usage at start
    pub start_memory: u64,
    /// Peak memory usage during measurement
    pub peak_memory: u64,
    /// End memory usage
    pub end_memory: Option<u64>,
    /// Custom metrics collected during measurement
    pub custom_metrics: HashMap<String, f64>,
}

impl PerformanceMeasurement {
    /// Create a new performance measurement
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            end_time: None,
            start_memory: crate::common::get_memory_usage(),
            peak_memory: crate::common::get_memory_usage(),
            end_memory: None,
            custom_metrics: HashMap::new(),
        }
    }

    /// Finish the measurement
    pub fn finish(&mut self) {
        self.end_time = Some(Instant::now());
        self.end_memory = Some(crate::common::get_memory_usage());
        self.peak_memory = crate::common::get_peak_memory_usage();
    }

    /// Get the duration of the measurement
    pub fn duration(&self) -> Option<Duration> {
        self.end_time.map(|end| end - self.start_time)
    }

    /// Get memory usage delta
    pub fn memory_delta(&self) -> Option<i64> {
        self.end_memory
            .map(|end| end as i64 - self.start_memory as i64)
    }

    /// Add a custom metric
    pub fn add_metric<S: Into<String>>(&mut self, name: S, value: f64) {
        self.custom_metrics.insert(name.into(), value);
    }

    /// Get a custom metric
    pub fn get_metric(&self, name: &str) -> Option<f64> {
        self.custom_metrics.get(name).copied()
    }

    /// Update peak memory if current usage is higher
    pub fn update_peak_memory(&mut self) {
        let current = crate::common::get_memory_usage();
        if current > self.peak_memory {
            self.peak_memory = current;
        }
    }
}

impl Default for PerformanceMeasurement {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance tracker for continuous monitoring
pub struct PerformanceTracker {
    measurements: Arc<RwLock<Vec<PerformanceMeasurement>>>,
    current: Arc<RwLock<Option<PerformanceMeasurement>>>,
    monitoring: Arc<AtomicU64>, // 0 = stopped, 1 = running
}

impl PerformanceTracker {
    /// Create a new performance tracker
    pub fn new() -> Self {
        Self {
            measurements: Arc::new(RwLock::new(Vec::new())),
            current: Arc::new(RwLock::new(None)),
            monitoring: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Start a new measurement
    pub async fn start_measurement(&self) -> TestResult<()> {
        let mut current = self.current.write().await;
        if current.is_some() {
            return Err(TestError::execution("A measurement is already in progress"));
        }

        *current = Some(PerformanceMeasurement::new());
        self.monitoring.store(1, Ordering::Relaxed);

        // Start background monitoring
        let current_clone = Arc::clone(&self.current);
        let monitoring_clone = Arc::clone(&self.monitoring);

        tokio::spawn(async move {
            while monitoring_clone.load(Ordering::Relaxed) == 1 {
                if let Some(ref mut measurement) = *current_clone.write().await {
                    measurement.update_peak_memory();
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        });

        Ok(())
    }

    /// Finish the current measurement
    pub async fn finish_measurement(&self) -> TestResult<PerformanceMeasurement> {
        self.monitoring.store(0, Ordering::Relaxed);

        let mut current = self.current.write().await;
        let mut measurement = current
            .take()
            .ok_or_else(|| TestError::execution("No measurement in progress"))?;

        measurement.finish();

        // Store the completed measurement
        self.measurements.write().await.push(measurement.clone());

        Ok(measurement)
    }

    /// Add a custom metric to the current measurement
    pub async fn add_metric<S: Into<String>>(&self, name: S, value: f64) -> TestResult<()> {
        let mut current = self.current.write().await;
        if let Some(ref mut measurement) = *current {
            measurement.add_metric(name, value);
            Ok(())
        } else {
            Err(TestError::execution("No measurement in progress"))
        }
    }

    /// Get all completed measurements
    pub async fn get_measurements(&self) -> Vec<PerformanceMeasurement> {
        self.measurements.read().await.clone()
    }

    /// Get summary statistics
    pub async fn get_summary(&self) -> PerformanceSummary {
        let measurements = self.measurements.read().await;
        PerformanceSummary::from_measurements(&measurements)
    }

    /// Clear all measurements
    pub async fn clear(&self) {
        self.measurements.write().await.clear();
    }
}

impl Default for PerformanceTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary statistics for performance measurements
#[derive(Debug, Clone)]
pub struct PerformanceSummary {
    /// Number of measurements
    pub count: usize,
    /// Average duration
    pub avg_duration: Option<Duration>,
    /// Minimum duration
    pub min_duration: Option<Duration>,
    /// Maximum duration
    pub max_duration: Option<Duration>,
    /// Average memory usage
    pub avg_memory_usage: Option<u64>,
    /// Peak memory usage across all measurements
    pub peak_memory_usage: Option<u64>,
    /// Total memory allocated
    pub total_memory_allocated: Option<u64>,
    /// Custom metric summaries
    pub custom_metrics: HashMap<String, MetricSummary>,
}

impl PerformanceSummary {
    /// Create a summary from a collection of measurements
    pub fn from_measurements(measurements: &[PerformanceMeasurement]) -> Self {
        if measurements.is_empty() {
            return Self {
                count: 0,
                avg_duration: None,
                min_duration: None,
                max_duration: None,
                avg_memory_usage: None,
                peak_memory_usage: None,
                total_memory_allocated: None,
                custom_metrics: HashMap::new(),
            };
        }

        let durations: Vec<Duration> = measurements.iter().filter_map(|m| m.duration()).collect();

        let avg_duration = if !durations.is_empty() {
            let total_nanos: u64 = durations.iter().map(|d| d.as_nanos() as u64).sum();
            Some(Duration::from_nanos(total_nanos / durations.len() as u64))
        } else {
            None
        };

        let min_duration = durations.iter().min().copied();
        let max_duration = durations.iter().max().copied();

        let memory_usages: Vec<u64> = measurements.iter().filter_map(|m| m.end_memory).collect();

        let avg_memory_usage = if !memory_usages.is_empty() {
            Some(memory_usages.iter().sum::<u64>() / memory_usages.len() as u64)
        } else {
            None
        };

        let peak_memory_usage = measurements.iter().map(|m| m.peak_memory).max();

        let total_memory_allocated = measurements
            .iter()
            .filter_map(|m| m.memory_delta())
            .filter(|&delta| delta > 0)
            .map(|delta| delta as u64)
            .sum::<u64>()
            .into();

        // Aggregate custom metrics
        let mut custom_metrics = HashMap::new();
        let mut metric_values: HashMap<String, Vec<f64>> = HashMap::new();

        for measurement in measurements {
            for (name, value) in &measurement.custom_metrics {
                metric_values
                    .entry(name.clone())
                    .or_insert_with(Vec::new)
                    .push(*value);
            }
        }

        for (name, values) in metric_values {
            custom_metrics.insert(name, MetricSummary::from_values(&values));
        }

        Self {
            count: measurements.len(),
            avg_duration,
            min_duration,
            max_duration,
            avg_memory_usage,
            peak_memory_usage,
            total_memory_allocated,
            custom_metrics,
        }
    }
}

/// Summary statistics for a custom metric
#[derive(Debug, Clone)]
pub struct MetricSummary {
    /// Number of values
    pub count: usize,
    /// Average value
    pub average: f64,
    /// Minimum value
    pub minimum: f64,
    /// Maximum value
    pub maximum: f64,
    /// Standard deviation
    pub std_deviation: f64,
}

impl MetricSummary {
    /// Create a summary from a collection of values
    pub fn from_values(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self {
                count: 0,
                average: 0.0,
                minimum: 0.0,
                maximum: 0.0,
                std_deviation: 0.0,
            };
        }

        let count = values.len();
        let sum: f64 = values.iter().sum();
        let average = sum / count as f64;
        let minimum = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let maximum = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let variance = values.iter().map(|&x| (x - average).powi(2)).sum::<f64>() / count as f64;
        let std_deviation = variance.sqrt();

        Self {
            count,
            average,
            minimum,
            maximum,
            std_deviation,
        }
    }
}

/// Benchmark runner for performance testing
pub struct BenchmarkRunner {
    name: String,
    iterations: usize,
    warmup_iterations: usize,
    tracker: PerformanceTracker,
}

impl BenchmarkRunner {
    /// Create a new benchmark runner
    pub fn new<S: Into<String>>(name: S) -> Self {
        Self {
            name: name.into(),
            iterations: 10,
            warmup_iterations: 3,
            tracker: PerformanceTracker::new(),
        }
    }

    /// Set the number of iterations
    pub fn iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Set the number of warmup iterations
    pub fn warmup_iterations(mut self, warmup_iterations: usize) -> Self {
        self.warmup_iterations = warmup_iterations;
        self
    }

    /// Run a benchmark
    pub async fn run<F, Fut, T>(&self, mut benchmark_fn: F) -> TestResult<BenchmarkResult>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = TestResult<T>>,
    {
        tracing::info!("Running benchmark: {}", self.name);

        // Warmup iterations
        tracing::debug!("Running {} warmup iterations", self.warmup_iterations);
        for i in 0..self.warmup_iterations {
            tracing::trace!("Warmup iteration {}", i + 1);
            let _ = benchmark_fn().await?;
        }

        // Clear any measurements from warmup
        self.tracker.clear().await;

        // Actual benchmark iterations
        tracing::debug!("Running {} benchmark iterations", self.iterations);
        for i in 0..self.iterations {
            tracing::trace!("Benchmark iteration {}", i + 1);

            self.tracker.start_measurement().await?;
            let _result = benchmark_fn().await?;
            self.tracker.finish_measurement().await?;
        }

        let summary = self.tracker.get_summary().await;

        Ok(BenchmarkResult {
            name: self.name.clone(),
            iterations: self.iterations,
            warmup_iterations: self.warmup_iterations,
            summary,
        })
    }
}

/// Result of a benchmark run
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Name of the benchmark
    pub name: String,
    /// Number of iterations run
    pub iterations: usize,
    /// Number of warmup iterations run
    pub warmup_iterations: usize,
    /// Performance summary
    pub summary: PerformanceSummary,
}

impl BenchmarkResult {
    /// Get the average operations per second (if duration is available)
    pub fn ops_per_second(&self) -> Option<f64> {
        self.summary
            .avg_duration
            .map(|avg_duration| 1.0 / avg_duration.as_secs_f64())
    }

    /// Get the throughput in MB/s (requires a "bytes_processed" custom metric)
    pub fn throughput_mbps(&self) -> Option<f64> {
        if let (Some(avg_duration), Some(bytes_metric)) = (
            self.summary.avg_duration,
            self.summary.custom_metrics.get("bytes_processed"),
        ) {
            let bytes_per_second = bytes_metric.average / avg_duration.as_secs_f64();
            Some(bytes_per_second / (1024.0 * 1024.0)) // Convert to MB/s
        } else {
            None
        }
    }

    /// Format the result as a human-readable string
    pub fn format_summary(&self) -> String {
        let mut summary = format!("Benchmark: {}\n", self.name);
        summary.push_str(&format!(
            "Iterations: {} (+ {} warmup)\n",
            self.iterations, self.warmup_iterations
        ));

        if let Some(avg_duration) = self.summary.avg_duration {
            summary.push_str(&format!(
                "Average duration: {}\n",
                crate::common::format_duration(avg_duration)
            ));
        }

        if let Some(min_duration) = self.summary.min_duration {
            summary.push_str(&format!(
                "Minimum duration: {}\n",
                crate::common::format_duration(min_duration)
            ));
        }

        if let Some(max_duration) = self.summary.max_duration {
            summary.push_str(&format!(
                "Maximum duration: {}\n",
                crate::common::format_duration(max_duration)
            ));
        }

        if let Some(ops_per_sec) = self.ops_per_second() {
            summary.push_str(&format!("Operations per second: {:.2}\n", ops_per_sec));
        }

        if let Some(throughput) = self.throughput_mbps() {
            summary.push_str(&format!("Throughput: {:.2} MB/s\n", throughput));
        }

        if let Some(peak_memory) = self.summary.peak_memory_usage {
            summary.push_str(&format!(
                "Peak memory usage: {}\n",
                crate::common::format_bytes(peak_memory)
            ));
        }

        summary
    }
}

/// Utility function to measure the performance of an async operation
pub async fn measure_performance<F, Fut, T>(operation: F) -> TestResult<(T, PerformanceMeasurement)>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = TestResult<T>>,
{
    let mut measurement = PerformanceMeasurement::new();
    let result = operation().await?;
    measurement.finish();
    Ok((result, measurement))
}

/// Utility function to run a simple benchmark
pub async fn simple_benchmark<F, Fut, T>(
    name: &str,
    iterations: usize,
    mut operation: F,
) -> TestResult<BenchmarkResult>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = TestResult<T>>,
{
    let runner = BenchmarkRunner::new(name).iterations(iterations);
    runner.run(operation).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_performance_measurement() {
        let mut measurement = PerformanceMeasurement::new();

        // Simulate some work
        sleep(Duration::from_millis(10)).await;
        measurement.add_metric("test_metric", 42.0);

        measurement.finish();

        assert!(measurement.duration().is_some());
        assert!(measurement.duration().unwrap() >= Duration::from_millis(10));
        assert_eq!(measurement.get_metric("test_metric"), Some(42.0));
        assert!(measurement.memory_delta().is_some());
    }

    #[tokio::test]
    async fn test_performance_tracker() {
        let tracker = PerformanceTracker::new();

        // Start and finish a measurement
        tracker.start_measurement().await.unwrap();
        sleep(Duration::from_millis(10)).await;
        tracker.add_metric("test", 123.0).await.unwrap();
        let measurement = tracker.finish_measurement().await.unwrap();

        assert!(measurement.duration().is_some());
        assert_eq!(measurement.get_metric("test"), Some(123.0));

        let measurements = tracker.get_measurements().await;
        assert_eq!(measurements.len(), 1);

        let summary = tracker.get_summary().await;
        assert_eq!(summary.count, 1);
        assert!(summary.avg_duration.is_some());
    }

    #[tokio::test]
    async fn test_benchmark_runner() {
        let runner = BenchmarkRunner::new("test_benchmark")
            .iterations(3)
            .warmup_iterations(1);

        let result = runner
            .run(|| async {
                sleep(Duration::from_millis(5)).await;
                Ok(42)
            })
            .await
            .unwrap();

        assert_eq!(result.name, "test_benchmark");
        assert_eq!(result.iterations, 3);
        assert_eq!(result.warmup_iterations, 1);
        assert_eq!(result.summary.count, 3);
        assert!(result.summary.avg_duration.is_some());
        assert!(result.ops_per_second().is_some());
    }

    #[tokio::test]
    async fn test_measure_performance() {
        let (result, measurement) = measure_performance(|| async {
            sleep(Duration::from_millis(5)).await;
            Ok("test_result")
        })
        .await
        .unwrap();

        assert_eq!(result, "test_result");
        assert!(measurement.duration().is_some());
        assert!(measurement.duration().unwrap() >= Duration::from_millis(5));
    }

    #[tokio::test]
    async fn test_simple_benchmark() {
        let result = simple_benchmark("simple_test", 2, || async {
            sleep(Duration::from_millis(1)).await;
            Ok(())
        })
        .await
        .unwrap();

        assert_eq!(result.name, "simple_test");
        assert_eq!(result.iterations, 2);
        assert_eq!(result.summary.count, 2);
    }

    #[test]
    fn test_metric_summary() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let summary = MetricSummary::from_values(&values);

        assert_eq!(summary.count, 5);
        assert_eq!(summary.average, 3.0);
        assert_eq!(summary.minimum, 1.0);
        assert_eq!(summary.maximum, 5.0);
        assert!(summary.std_deviation > 0.0);
    }

    #[test]
    fn test_performance_summary() {
        let mut measurements = Vec::new();

        for i in 0..3 {
            let mut measurement = PerformanceMeasurement::new();
            measurement.end_time = Some(measurement.start_time + Duration::from_millis(10 + i));
            measurement.end_memory = Some(1000 + i * 100);
            measurement.add_metric("test_metric", (i + 1) as f64);
            measurements.push(measurement);
        }

        let summary = PerformanceSummary::from_measurements(&measurements);

        assert_eq!(summary.count, 3);
        assert!(summary.avg_duration.is_some());
        assert!(summary.min_duration.is_some());
        assert!(summary.max_duration.is_some());
        assert!(summary.custom_metrics.contains_key("test_metric"));

        let metric_summary = &summary.custom_metrics["test_metric"];
        assert_eq!(metric_summary.count, 3);
        assert_eq!(metric_summary.average, 2.0);
    }
}

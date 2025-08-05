//! Automatic performance tuning based on hardware capabilities

use anyhow::Result;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

use super::CachingConfig;

/// Performance metrics sample
#[derive(Debug, Clone)]
pub struct PerformanceSample {
    pub timestamp: Instant,
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub requests_per_second: f64,
    pub average_latency_ms: f64,
    pub error_rate: f64,
    pub batch_size: usize,
    pub cache_hit_rate: f64,
}

/// Performance tuning recommendations
#[derive(Debug, Clone)]
pub struct TuningRecommendation {
    pub parameter: String,
    pub current_value: f64,
    pub recommended_value: f64,
    pub reason: String,
    pub expected_improvement: f64,
    pub confidence: f64,
}

/// Automatic performance tuner
pub struct PerformanceTuner {
    config: CachingConfig,
    performance_history: VecDeque<PerformanceSample>,
    current_parameters: PerformanceParameters,
    statistics: PerformanceStatistics,
    last_optimization: Instant,
}

/// Tunable performance parameters
#[derive(Debug, Clone)]
pub struct PerformanceParameters {
    pub batch_size: usize,
    pub batch_timeout_ms: u64,
    pub cache_size_mb: usize,
    pub connection_pool_size: usize,
    pub worker_threads: usize,
    pub memory_threshold: f64,
    pub cpu_threshold: f64,
}

/// Performance tuning statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct PerformanceStatistics {
    pub total_optimizations: u64,
    pub successful_optimizations: u64,
    pub optimization_success_rate: f64,
    pub average_improvement_percent: f64,
    pub last_optimization_time: Option<String>,
    pub current_performance_score: f64,
    pub peak_performance_score: f64,
    pub recommendations_applied: u64,
}

impl Default for PerformanceStatistics {
    fn default() -> Self {
        Self {
            total_optimizations: 0,
            successful_optimizations: 0,
            optimization_success_rate: 0.0,
            average_improvement_percent: 0.0,
            last_optimization_time: None,
            current_performance_score: 0.0,
            peak_performance_score: 0.0,
            recommendations_applied: 0,
        }
    }
}

impl PerformanceTuner {
    /// Create a new performance tuner
    pub fn new(config: &CachingConfig) -> Result<Self> {
        let current_parameters = PerformanceParameters {
            batch_size: config.max_batch_size,
            batch_timeout_ms: config.batch_timeout_ms,
            cache_size_mb: config.kv_cache_size_mb,
            connection_pool_size: config.connection_pool_size,
            worker_threads: num_cpus::get(),
            memory_threshold: 80.0,
            cpu_threshold: 80.0,
        };

        Ok(Self {
            config: config.clone(),
            performance_history: VecDeque::with_capacity(1000),
            current_parameters,
            statistics: PerformanceStatistics::default(),
            last_optimization: Instant::now(),
        })
    }

    /// Record a performance sample
    pub fn record_sample(&mut self, sample: PerformanceSample) {
        self.performance_history.push_back(sample);
        
        // Keep only the last 1000 samples
        if self.performance_history.len() > 1000 {
            self.performance_history.pop_front();
        }

        // Update current performance score
        self.update_performance_score();
    }

    /// Run optimization based on recent performance data
    pub async fn run_optimization(&mut self) -> Result<Vec<TuningRecommendation>> {
        if self.performance_history.len() < 10 {
            return Ok(Vec::new());
        }

        let mut recommendations = Vec::new();

        // Analyze recent performance trends
        let recent_samples: Vec<_> = self.performance_history
            .iter()
            .rev()
            .take(20)
            .cloned()
            .collect();

        // Generate recommendations based on analysis
        recommendations.extend(self.analyze_batch_performance(&recent_samples));
        recommendations.extend(self.analyze_memory_usage(&recent_samples));
        recommendations.extend(self.analyze_cpu_usage(&recent_samples));
        recommendations.extend(self.analyze_latency(&recent_samples));
        recommendations.extend(self.analyze_error_rate(&recent_samples));

        // Apply high-confidence recommendations
        let applied_recommendations = self.apply_recommendations(&recommendations).await?;

        // Update statistics
        self.statistics.total_optimizations += 1;
        if !applied_recommendations.is_empty() {
            self.statistics.successful_optimizations += 1;
            self.statistics.recommendations_applied += applied_recommendations.len() as u64;
        }
        
        self.statistics.optimization_success_rate = 
            self.statistics.successful_optimizations as f64 / self.statistics.total_optimizations as f64;
        
        self.statistics.last_optimization_time = Some(
            chrono::Utc::now().to_rfc3339()
        );

        self.last_optimization = Instant::now();

        Ok(recommendations)
    }

    /// Analyze batch performance and generate recommendations
    fn analyze_batch_performance(&self, samples: &[PerformanceSample]) -> Vec<TuningRecommendation> {
        let mut recommendations = Vec::new();

        if samples.is_empty() {
            return recommendations;
        }

        let avg_rps = samples.iter().map(|s| s.requests_per_second).sum::<f64>() / samples.len() as f64;
        let avg_latency = samples.iter().map(|s| s.average_latency_ms).sum::<f64>() / samples.len() as f64;
        let avg_batch_size = samples.iter().map(|s| s.batch_size).sum::<usize>() / samples.len();

        // If latency is high but batch size is small, recommend increasing batch size
        if avg_latency > 100.0 && avg_batch_size < self.config.max_batch_size / 2 {
            recommendations.push(TuningRecommendation {
                parameter: "batch_size".to_string(),
                current_value: self.current_parameters.batch_size as f64,
                recommended_value: (self.current_parameters.batch_size * 2).min(self.config.max_batch_size) as f64,
                reason: "Low batch utilization with high latency detected".to_string(),
                expected_improvement: 15.0,
                confidence: 0.8,
            });
        }

        // If RPS is low and batch timeout is high, recommend reducing timeout
        if avg_rps < 10.0 && self.current_parameters.batch_timeout_ms > 20 {
            recommendations.push(TuningRecommendation {
                parameter: "batch_timeout_ms".to_string(),
                current_value: self.current_parameters.batch_timeout_ms as f64,
                recommended_value: (self.current_parameters.batch_timeout_ms / 2).max(5) as f64,
                reason: "Low throughput with high batch timeout detected".to_string(),
                expected_improvement: 10.0,
                confidence: 0.7,
            });
        }

        recommendations
    }

    /// Analyze memory usage patterns
    fn analyze_memory_usage(&self, samples: &[PerformanceSample]) -> Vec<TuningRecommendation> {
        let mut recommendations = Vec::new();

        if samples.is_empty() {
            return recommendations;
        }

        let avg_memory = samples.iter().map(|s| s.memory_usage).sum::<f64>() / samples.len() as f64;
        let max_memory = samples.iter().map(|s| s.memory_usage).fold(0.0, f64::max);

        // If memory usage is consistently high, recommend reducing cache size
        if avg_memory > 85.0 {
            let reduction_factor = if max_memory > 95.0 { 0.7 } else { 0.8 };
            recommendations.push(TuningRecommendation {
                parameter: "cache_size_mb".to_string(),
                current_value: self.current_parameters.cache_size_mb as f64,
                recommended_value: (self.current_parameters.cache_size_mb as f64 * reduction_factor),
                reason: "High memory usage detected".to_string(),
                expected_improvement: 5.0,
                confidence: 0.9,
            });
        }

        // If memory usage is very low, we could increase cache size for better performance
        if avg_memory < 50.0 && self.current_parameters.cache_size_mb < 1024 {
            recommendations.push(TuningRecommendation {
                parameter: "cache_size_mb".to_string(),
                current_value: self.current_parameters.cache_size_mb as f64,
                recommended_value: (self.current_parameters.cache_size_mb as f64 * 1.2),
                reason: "Low memory usage - can increase cache for better performance".to_string(),
                expected_improvement: 8.0,
                confidence: 0.6,
            });
        }

        recommendations
    }

    /// Analyze CPU usage patterns
    fn analyze_cpu_usage(&self, samples: &[PerformanceSample]) -> Vec<TuningRecommendation> {
        let mut recommendations = Vec::new();

        if samples.is_empty() {
            return recommendations;
        }

        let avg_cpu = samples.iter().map(|s| s.cpu_usage).sum::<f64>() / samples.len() as f64;
        let max_cpu = samples.iter().map(|s| s.cpu_usage).fold(0.0, f64::max);

        // If CPU usage is consistently high, recommend reducing concurrent connections
        if avg_cpu > 85.0 {
            let reduction_factor = if max_cpu > 95.0 { 0.7 } else { 0.8 };
            recommendations.push(TuningRecommendation {
                parameter: "connection_pool_size".to_string(),
                current_value: self.current_parameters.connection_pool_size as f64,
                recommended_value: (self.current_parameters.connection_pool_size as f64 * reduction_factor),
                reason: "High CPU usage detected".to_string(),
                expected_improvement: 10.0,
                confidence: 0.8,
            });
        }

        // If CPU usage is low, we could increase connection pool size
        if avg_cpu < 40.0 && self.current_parameters.connection_pool_size < 200 {
            recommendations.push(TuningRecommendation {
                parameter: "connection_pool_size".to_string(),
                current_value: self.current_parameters.connection_pool_size as f64,
                recommended_value: (self.current_parameters.connection_pool_size as f64 * 1.3),
                reason: "Low CPU usage - can handle more concurrent connections".to_string(),
                expected_improvement: 12.0,
                confidence: 0.7,
            });
        }

        recommendations
    }

    /// Analyze latency patterns
    fn analyze_latency(&self, samples: &[PerformanceSample]) -> Vec<TuningRecommendation> {
        let mut recommendations = Vec::new();

        if samples.len() < 5 {
            return recommendations;
        }

        let recent_latency: Vec<f64> = samples.iter().rev().take(5).map(|s| s.average_latency_ms).collect();
        let older_latency: Vec<f64> = samples.iter().rev().skip(5).take(5).map(|s| s.average_latency_ms).collect();

        if !older_latency.is_empty() {
            let recent_avg = recent_latency.iter().sum::<f64>() / recent_latency.len() as f64;
            let older_avg = older_latency.iter().sum::<f64>() / older_latency.len() as f64;

            // If latency is increasing, recommend more aggressive caching
            if recent_avg > older_avg * 1.2 {
                recommendations.push(TuningRecommendation {
                    parameter: "cache_size_mb".to_string(),
                    current_value: self.current_parameters.cache_size_mb as f64,
                    recommended_value: (self.current_parameters.cache_size_mb as f64 * 1.1),
                    reason: "Increasing latency trend detected".to_string(),
                    expected_improvement: 7.0,
                    confidence: 0.6,
                });
            }
        }

        recommendations
    }

    /// Analyze error rate patterns
    fn analyze_error_rate(&self, samples: &[PerformanceSample]) -> Vec<TuningRecommendation> {
        let mut recommendations = Vec::new();

        if samples.is_empty() {
            return recommendations;
        }

        let avg_error_rate = samples.iter().map(|s| s.error_rate).sum::<f64>() / samples.len() as f64;

        // If error rate is high, recommend reducing load
        if avg_error_rate > 5.0 {
            recommendations.push(TuningRecommendation {
                parameter: "connection_pool_size".to_string(),
                current_value: self.current_parameters.connection_pool_size as f64,
                recommended_value: (self.current_parameters.connection_pool_size as f64 * 0.8),
                reason: "High error rate detected - reducing load".to_string(),
                expected_improvement: 15.0,
                confidence: 0.9,
            });

            recommendations.push(TuningRecommendation {
                parameter: "batch_size".to_string(),
                current_value: self.current_parameters.batch_size as f64,
                recommended_value: (self.current_parameters.batch_size as f64 * 0.8),
                reason: "High error rate detected - reducing batch size".to_string(),
                expected_improvement: 10.0,
                confidence: 0.8,
            });
        }

        recommendations
    }

    /// Apply high-confidence recommendations
    async fn apply_recommendations(&mut self, recommendations: &[TuningRecommendation]) -> Result<Vec<TuningRecommendation>> {
        let mut applied = Vec::new();

        for recommendation in recommendations {
            // Only apply recommendations with high confidence
            if recommendation.confidence >= 0.8 {
                match recommendation.parameter.as_str() {
                    "batch_size" => {
                        self.current_parameters.batch_size = recommendation.recommended_value as usize;
                        applied.push(recommendation.clone());
                    }
                    "batch_timeout_ms" => {
                        self.current_parameters.batch_timeout_ms = recommendation.recommended_value as u64;
                        applied.push(recommendation.clone());
                    }
                    "cache_size_mb" => {
                        self.current_parameters.cache_size_mb = recommendation.recommended_value as usize;
                        applied.push(recommendation.clone());
                    }
                    "connection_pool_size" => {
                        self.current_parameters.connection_pool_size = recommendation.recommended_value as usize;
                        applied.push(recommendation.clone());
                    }
                    _ => {}
                }
            }
        }

        Ok(applied)
    }

    /// Update the current performance score
    fn update_performance_score(&mut self) {
        if let Some(latest_sample) = self.performance_history.back() {
            // Calculate a composite performance score (0-100)
            let rps_score = (latest_sample.requests_per_second / 100.0).min(1.0) * 30.0;
            let latency_score = (1.0 - (latest_sample.average_latency_ms / 1000.0).min(1.0)) * 25.0;
            let error_score = (1.0 - (latest_sample.error_rate / 10.0).min(1.0)) * 20.0;
            let cache_score = latest_sample.cache_hit_rate * 15.0;
            let resource_score = (1.0 - ((latest_sample.cpu_usage + latest_sample.memory_usage) / 200.0).min(1.0)) * 10.0;

            let current_score = rps_score + latency_score + error_score + cache_score + resource_score;
            
            self.statistics.current_performance_score = current_score;
            self.statistics.peak_performance_score = self.statistics.peak_performance_score.max(current_score);
        }
    }

    /// Get current performance parameters
    pub fn get_parameters(&self) -> &PerformanceParameters {
        &self.current_parameters
    }

    /// Get performance statistics
    pub async fn get_statistics(&self) -> PerformanceStatistics {
        self.statistics.clone()
    }

    /// Get recent performance trend
    pub fn get_performance_trend(&self, duration: Duration) -> Vec<PerformanceSample> {
        let cutoff = Instant::now() - duration;
        self.performance_history
            .iter()
            .filter(|sample| sample.timestamp >= cutoff)
            .cloned()
            .collect()
    }

    /// Generate performance report
    pub fn generate_report(&self) -> PerformanceReport {
        let recent_samples: Vec<_> = self.performance_history
            .iter()
            .rev()
            .take(100)
            .cloned()
            .collect();

        let avg_rps = if !recent_samples.is_empty() {
            recent_samples.iter().map(|s| s.requests_per_second).sum::<f64>() / recent
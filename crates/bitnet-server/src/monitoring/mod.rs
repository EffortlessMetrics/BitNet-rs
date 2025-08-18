//! Monitoring and observability infrastructure for BitNet server

pub mod config;
pub mod health;
pub mod metrics;
#[cfg(feature = "prometheus")]
pub mod prometheus;
pub mod tracing;

#[cfg(feature = "opentelemetry")]
pub mod opentelemetry;

use anyhow::Result;
use std::sync::Arc;

/// Monitoring configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MonitoringConfig {
    /// Enable Prometheus metrics
    pub prometheus_enabled: bool,
    /// Prometheus metrics endpoint path
    pub prometheus_path: String,
    /// Enable OpenTelemetry tracing
    pub opentelemetry_enabled: bool,
    /// OpenTelemetry endpoint URL
    pub opentelemetry_endpoint: Option<String>,
    /// Health check endpoint path
    pub health_path: String,
    /// Metrics collection interval in seconds
    pub metrics_interval: u64,
    /// Enable structured logging
    pub structured_logging: bool,
    /// Log level filter
    pub log_level: String,
    /// Log output format (json, pretty, compact)
    pub log_format: String,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            prometheus_enabled: true,
            prometheus_path: "/metrics".to_string(),
            opentelemetry_enabled: false,
            opentelemetry_endpoint: None,
            health_path: "/health".to_string(),
            metrics_interval: 10,
            structured_logging: true,
            log_level: "info".to_string(),
            log_format: "json".to_string(),
        }
    }
}

/// Monitoring system that coordinates all observability components
pub struct MonitoringSystem {
    config: MonitoringConfig,
    metrics: Arc<metrics::MetricsCollector>,
    _tracing_guard: Option<tracing::TracingGuard>,
}

impl MonitoringSystem {
    /// Initialize the monitoring system with the given configuration
    pub async fn new(config: MonitoringConfig) -> Result<Self> {
        // Initialize tracing first
        let tracing_guard = if config.structured_logging {
            Some(tracing::init_tracing(&config).await?)
        } else {
            None
        };

        // Initialize metrics collection
        let metrics = Arc::new(metrics::MetricsCollector::new(&config)?);

        // Initialize OpenTelemetry if enabled
        #[cfg(feature = "opentelemetry")]
        if config.opentelemetry_enabled {
            opentelemetry::init_opentelemetry(&config).await?;
        }

        Ok(Self { config, metrics, _tracing_guard: tracing_guard })
    }

    /// Get the metrics collector
    pub fn metrics(&self) -> Arc<metrics::MetricsCollector> {
        self.metrics.clone()
    }

    /// Get the monitoring configuration
    pub fn config(&self) -> &MonitoringConfig {
        &self.config
    }

    /// Start background monitoring tasks
    pub async fn start_background_tasks(&self) -> Result<()> {
        let metrics = self.metrics.clone();
        let interval = self.config.metrics_interval;

        // Start metrics collection task
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(interval));
            loop {
                interval.tick().await;
                if let Err(e) = metrics.collect_system_metrics().await {
                    eprintln!("Failed to collect system metrics: {}", e);
                }
            }
        });

        Ok(())
    }

    /// Shutdown the monitoring system gracefully
    pub async fn shutdown(&self) -> Result<()> {
        println!("Shutting down monitoring system");

        #[cfg(feature = "opentelemetry")]
        if self.config.opentelemetry_enabled {
            opentelemetry::shutdown().await?;
        }

        Ok(())
    }
}

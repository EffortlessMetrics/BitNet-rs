//! Structured logging and tracing configuration

use anyhow::Result;
use std::io;
use tracing_appender::{non_blocking, rolling};
use tracing_subscriber::{
    fmt::{self, format::FmtSpan},
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter,
};

use super::MonitoringConfig;

/// Guard that ensures proper cleanup of tracing resources
pub struct TracingGuard {
    _file_guard: Option<non_blocking::WorkerGuard>,
}

impl TracingGuard {
    fn new(file_guard: Option<non_blocking::WorkerGuard>) -> Self {
        Self {
            _file_guard: file_guard,
        }
    }
}

/// Initialize structured logging and tracing
pub async fn init_tracing(config: &MonitoringConfig) -> Result<TracingGuard> {
    // Create environment filter
    let env_filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new(&config.log_level))
        .unwrap_or_else(|_| EnvFilter::new("info"));

    // Simple console layer
    let console_layer = fmt::layer()
        .with_span_events(FmtSpan::CLOSE)
        .with_thread_ids(true)
        .with_thread_names(true)
        .with_target(true)
        .with_file(true)
        .with_line_number(true)
        .with_writer(io::stdout);

    // File output layer with rotation
    let file_appender = rolling::daily("logs", "bitnet-server.log");
    let (file_writer, file_guard) = non_blocking(file_appender);

    let file_layer = fmt::layer()
        .with_span_events(FmtSpan::CLOSE)
        .with_thread_ids(true)
        .with_thread_names(true)
        .with_target(true)
        .with_file(true)
        .with_line_number(true)
        .with_writer(file_writer);

    // Build the subscriber
    tracing_subscriber::registry()
        .with(env_filter)
        .with(console_layer)
        .with(file_layer)
        .init();

    tracing::info!(
        log_level = %config.log_level,
        log_format = %config.log_format,
        opentelemetry_enabled = config.opentelemetry_enabled,
        "Tracing initialized"
    );

    Ok(TracingGuard::new(Some(file_guard)))
}

/// Tracing utilities for request correlation
pub mod request_tracing {
    use tracing::{info_span, Span};
    use uuid::Uuid;

    /// Create a new request span with correlation ID
    pub fn create_request_span(method: &str, path: &str) -> Span {
        let request_id = Uuid::new_v4().to_string();
        info_span!(
            "request",
            request_id = %request_id,
            method = method,
            path = path,
            status_code = tracing::field::Empty,
            duration_ms = tracing::field::Empty,
            tokens_generated = tracing::field::Empty,
        )
    }

    /// Record request completion in the current span
    pub fn record_request_completion(
        span: &Span,
        status_code: u16,
        duration_ms: u64,
        tokens_generated: Option<u64>,
    ) {
        span.record("status_code", status_code);
        span.record("duration_ms", duration_ms);
        if let Some(tokens) = tokens_generated {
            span.record("tokens_generated", tokens);
        }
    }

    /// Create an inference span for detailed operation tracking
    pub fn create_inference_span(model_name: &str, prompt_length: usize) -> Span {
        info_span!(
            "inference",
            model_name = model_name,
            prompt_length = prompt_length,
            tokens_generated = tracing::field::Empty,
            inference_time_ms = tracing::field::Empty,
            tokens_per_second = tracing::field::Empty,
        )
    }

    /// Record inference completion
    pub fn record_inference_completion(span: &Span, tokens_generated: u64, inference_time_ms: u64) {
        span.record("tokens_generated", tokens_generated);
        span.record("inference_time_ms", inference_time_ms);

        if inference_time_ms > 0 {
            let tokens_per_second = (tokens_generated as f64 * 1000.0) / inference_time_ms as f64;
            span.record("tokens_per_second", tokens_per_second);
        }
    }
}

/// Structured logging macros for common events
#[macro_export]
macro_rules! log_inference_start {
    ($request_id:expr, $model:expr, $prompt_len:expr) => {
        tracing::info!(
            request_id = %$request_id,
            model = %$model,
            prompt_length = $prompt_len,
            "Starting inference"
        );
    };
}

#[macro_export]
macro_rules! log_inference_complete {
    ($request_id:expr, $tokens:expr, $duration_ms:expr, $tokens_per_sec:expr) => {
        tracing::info!(
            request_id = %$request_id,
            tokens_generated = $tokens,
            duration_ms = $duration_ms,
            tokens_per_second = $tokens_per_sec,
            "Inference completed"
        );
    };
}

#[macro_export]
macro_rules! log_error {
    ($request_id:expr, $error:expr, $context:expr) => {
        tracing::error!(
            request_id = %$request_id,
            error = %$error,
            context = $context,
            "Request failed"
        );
    };
}

#[macro_export]
macro_rules! log_model_load {
    ($model_name:expr, $duration_ms:expr, $size_mb:expr) => {
        tracing::info!(
            model_name = %$model_name,
            load_duration_ms = $duration_ms,
            model_size_mb = $size_mb,
            "Model loaded successfully"
        );
    };
}

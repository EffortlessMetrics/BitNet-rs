#![allow(unused)]
#![allow(dead_code)]

//! HTTP Response Fixtures for BitNet-rs Inference Server Testing
//!
//! This module provides expected response fixtures for validating the production
//! inference server REST API responses, streaming outputs, and error conditions.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::LazyLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub text: String,
    pub tokens_generated: u32,
    pub inference_time_ms: u64,
    pub tokens_per_second: f32,
    pub model_id: String,
    pub request_id: String,
    pub device_used: String,
    pub quantization_used: String,
    pub accuracy_metrics: AccuracyMetrics,
    pub memory_usage: MemoryUsage,
    pub stop_reason: String,
    pub input_tokens: u32,
    pub timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyMetrics {
    pub quantization_accuracy: f32,
    pub inference_quality_score: f32,
    pub reference_correlation: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsage {
    pub peak_memory_mb: u64,
    pub gpu_memory_mb: Option<u64>,
    pub model_memory_mb: u64,
    pub cache_memory_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingChunk {
    pub token: String,
    pub is_final: bool,
    pub chunk_index: u32,
    pub request_id: String,
    pub partial_text: String,
    pub timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResponse {
    pub responses: Vec<InferenceResponse>,
    pub batch_id: String,
    pub total_processing_time_ms: u64,
    pub successful_requests: u32,
    pub failed_requests: u32,
    pub batch_metrics: BatchMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchMetrics {
    pub average_latency_ms: f32,
    pub throughput_requests_per_second: f32,
    pub peak_memory_usage_mb: u64,
    pub device_utilization_percent: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
    pub error_code: String,
    pub message: String,
    pub request_id: Option<String>,
    pub timestamp: String,
    pub details: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResponse {
    pub status: String,
    pub version: String,
    pub uptime_seconds: u64,
    pub models_loaded: u32,
    pub system_metrics: SystemMetrics,
    pub device_status: DeviceStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub cpu_usage_percent: f32,
    pub memory_usage_mb: u64,
    pub memory_total_mb: u64,
    pub disk_usage_percent: f32,
    pub active_requests: u32,
    pub total_requests_processed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceStatus {
    pub cpu_available: bool,
    pub gpu_available: bool,
    pub gpu_memory_total_mb: Option<u64>,
    pub gpu_memory_free_mb: Option<u64>,
    pub cuda_version: Option<String>,
    pub compute_capability: Option<String>,
}

/// Basic inference response fixtures
pub static BASIC_INFERENCE_RESPONSES: LazyLock<HashMap<&'static str, InferenceResponse>> =
    LazyLock::new(|| {
        let mut responses = HashMap::new();

        responses.insert("simple_question_i2s_cpu", InferenceResponse {
        text: "Neural network quantization is a technique used to reduce the precision of weights and activations in neural networks, typically from 32-bit floating point to lower bit representations like 8-bit integers or even 1-bit values. This process significantly reduces model size and computational requirements while maintaining acceptable accuracy levels. The main benefits include faster inference, lower memory usage, and reduced energy consumption, making it essential for deploying models on edge devices and mobile platforms.".to_string(),
        tokens_generated: 78,
        inference_time_ms: 1234,
        tokens_per_second: 63.2,
        model_id: "small_i2s_model".to_string(),
        request_id: "test-001".to_string(),
        device_used: "cpu".to_string(),
        quantization_used: "i2s".to_string(),
        accuracy_metrics: AccuracyMetrics {
            quantization_accuracy: 0.991,
            inference_quality_score: 0.94,
            reference_correlation: Some(0.987),
        },
        memory_usage: MemoryUsage {
            peak_memory_mb: 156,
            gpu_memory_mb: None,
            model_memory_mb: 25,
            cache_memory_mb: 12,
        },
        stop_reason: "max_tokens".to_string(),
        input_tokens: 6,
        timestamp: "2024-09-29T12:00:00Z".to_string(),
    });

        responses.insert("gpu_tl2_detailed", InferenceResponse {
        text: "1-bit neural networks, particularly BitNet architectures, offer several significant advantages over traditional floating-point models:\n\n1. **Extreme Compression**: By representing weights with just 1 bit (-1 or +1), these networks achieve up to 32x compression compared to FP32 models, dramatically reducing storage and memory requirements.\n\n2. **Energy Efficiency**: 1-bit operations consume significantly less energy than floating-point arithmetic, making them ideal for battery-powered devices and edge computing scenarios.\n\n3. **Hardware Optimization**: Many modern processors include specialized instructions for bit operations, allowing 1-bit networks to leverage hardware acceleration more effectively.\n\n4. **Maintained Accuracy**: Advanced quantization techniques like I2S (2-bit signed) can maintain 99%+ accuracy compared to full-precision models while still providing substantial benefits.".to_string(),
        tokens_generated: 142,
        inference_time_ms: 892,
        tokens_per_second: 159.2,
        model_id: "large_tl2_model".to_string(),
        request_id: "test-gpu-001".to_string(),
        device_used: "gpu".to_string(),
        quantization_used: "tl2".to_string(),
        accuracy_metrics: AccuracyMetrics {
            quantization_accuracy: 0.983,
            inference_quality_score: 0.96,
            reference_correlation: Some(0.979),
        },
        memory_usage: MemoryUsage {
            peak_memory_mb: 2048,
            gpu_memory_mb: Some(1800),
            model_memory_mb: 500,
            cache_memory_mb: 256,
        },
        stop_reason: "stop_sequence".to_string(),
        input_tokens: 11,
        timestamp: "2024-09-29T12:05:00Z".to_string(),
    });

        responses.insert(
            "minimal_defaults",
            InferenceResponse {
                text: "Hello! How can I assist you today?".to_string(),
                tokens_generated: 8,
                inference_time_ms: 234,
                tokens_per_second: 34.2,
                model_id: "small_i2s_model".to_string(),
                request_id: "auto-generated-001".to_string(),
                device_used: "cpu".to_string(),
                quantization_used: "i2s".to_string(),
                accuracy_metrics: AccuracyMetrics {
                    quantization_accuracy: 0.995,
                    inference_quality_score: 0.98,
                    reference_correlation: None,
                },
                memory_usage: MemoryUsage {
                    peak_memory_mb: 89,
                    gpu_memory_mb: None,
                    model_memory_mb: 25,
                    cache_memory_mb: 8,
                },
                stop_reason: "eos_token".to_string(),
                input_tokens: 1,
                timestamp: "2024-09-29T12:10:00Z".to_string(),
            },
        );

        responses
    });

/// Streaming response fixtures (Server-Sent Events format)
pub static STREAMING_RESPONSES: LazyLock<HashMap<&'static str, Vec<StreamingChunk>>> =
    LazyLock::new(|| {
        let mut responses = HashMap::new();

        responses.insert("bitnet_guide_stream", vec![
        StreamingChunk {
            token: "BitNet".to_string(),
            is_final: false,
            chunk_index: 0,
            request_id: "test-stream-001".to_string(),
            partial_text: "BitNet".to_string(),
            timestamp: "2024-09-29T12:15:00.100Z".to_string(),
        },
        StreamingChunk {
            token: " architecture".to_string(),
            is_final: false,
            chunk_index: 1,
            request_id: "test-stream-001".to_string(),
            partial_text: "BitNet architecture".to_string(),
            timestamp: "2024-09-29T12:15:00.250Z".to_string(),
        },
        StreamingChunk {
            token: " represents".to_string(),
            is_final: false,
            chunk_index: 2,
            request_id: "test-stream-001".to_string(),
            partial_text: "BitNet architecture represents".to_string(),
            timestamp: "2024-09-29T12:15:00.400Z".to_string(),
        },
        StreamingChunk {
            token: " a".to_string(),
            is_final: false,
            chunk_index: 3,
            request_id: "test-stream-001".to_string(),
            partial_text: "BitNet architecture represents a".to_string(),
            timestamp: "2024-09-29T12:15:00.550Z".to_string(),
        },
        StreamingChunk {
            token: " revolutionary".to_string(),
            is_final: false,
            chunk_index: 4,
            request_id: "test-stream-001".to_string(),
            partial_text: "BitNet architecture represents a revolutionary".to_string(),
            timestamp: "2024-09-29T12:15:00.700Z".to_string(),
        },
        StreamingChunk {
            token: "".to_string(),
            is_final: true,
            chunk_index: 5,
            request_id: "test-stream-001".to_string(),
            partial_text: "BitNet architecture represents a revolutionary approach to neural network quantization...".to_string(),
            timestamp: "2024-09-29T12:15:05.000Z".to_string(),
        },
    ]);

        responses
    });

/// Batch response fixtures
pub static BATCH_RESPONSES: LazyLock<HashMap<&'static str, BatchResponse>> = LazyLock::new(|| {
    let mut responses = HashMap::new();

    responses.insert("small_batch_success", BatchResponse {
        responses: vec![
            InferenceResponse {
                text: "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.".to_string(),
                tokens_generated: 24,
                inference_time_ms: 567,
                tokens_per_second: 42.3,
                model_id: "small_i2s_model".to_string(),
                request_id: "batch-1-1".to_string(),
                device_used: "cpu".to_string(),
                quantization_used: "i2s".to_string(),
                accuracy_metrics: AccuracyMetrics {
                    quantization_accuracy: 0.992,
                    inference_quality_score: 0.93,
                    reference_correlation: None,
                },
                memory_usage: MemoryUsage {
                    peak_memory_mb: 134,
                    gpu_memory_mb: None,
                    model_memory_mb: 25,
                    cache_memory_mb: 10,
                },
                stop_reason: "max_tokens".to_string(),
                input_tokens: 4,
                timestamp: "2024-09-29T12:20:00Z".to_string(),
            },
            InferenceResponse {
                text: "Deep learning is a machine learning technique that uses neural networks with multiple layers to model and understand complex patterns in data.".to_string(),
                tokens_generated: 25,
                inference_time_ms: 612,
                tokens_per_second: 40.8,
                model_id: "small_i2s_model".to_string(),
                request_id: "batch-1-2".to_string(),
                device_used: "cpu".to_string(),
                quantization_used: "i2s".to_string(),
                accuracy_metrics: AccuracyMetrics {
                    quantization_accuracy: 0.994,
                    inference_quality_score: 0.95,
                    reference_correlation: None,
                },
                memory_usage: MemoryUsage {
                    peak_memory_mb: 142,
                    gpu_memory_mb: None,
                    model_memory_mb: 25,
                    cache_memory_mb: 12,
                },
                stop_reason: "max_tokens".to_string(),
                input_tokens: 4,
                timestamp: "2024-09-29T12:20:01Z".to_string(),
            },
        ],
        batch_id: "batch-001".to_string(),
        total_processing_time_ms: 1200,
        successful_requests: 2,
        failed_requests: 0,
        batch_metrics: BatchMetrics {
            average_latency_ms: 589.5,
            throughput_requests_per_second: 1.67,
            peak_memory_usage_mb: 142,
            device_utilization_percent: 78.5,
        },
    });

    responses
});

/// Error response fixtures
pub static ERROR_RESPONSES: LazyLock<HashMap<&'static str, ErrorResponse>> = LazyLock::new(|| {
    let mut responses = HashMap::new();

    responses.insert(
        "empty_prompt",
        ErrorResponse {
            error: "Validation Error".to_string(),
            error_code: "INVALID_PROMPT".to_string(),
            message: "Prompt cannot be empty".to_string(),
            request_id: None,
            timestamp: "2024-09-29T12:25:00Z".to_string(),
            details: Some(json!({
                "field": "prompt",
                "constraint": "non_empty",
                "provided_value": ""
            })),
        },
    );

    responses.insert(
        "missing_prompt",
        ErrorResponse {
            error: "Schema Validation Error".to_string(),
            error_code: "MISSING_FIELD".to_string(),
            message: "Required field 'prompt' is missing".to_string(),
            request_id: None,
            timestamp: "2024-09-29T12:26:00Z".to_string(),
            details: Some(json!({
                "missing_fields": ["prompt"],
                "schema_version": "v1"
            })),
        },
    );

    responses.insert(
        "invalid_temperature",
        ErrorResponse {
            error: "Parameter Validation Error".to_string(),
            error_code: "INVALID_RANGE".to_string(),
            message: "Temperature must be between 0.0 and 2.0".to_string(),
            request_id: None,
            timestamp: "2024-09-29T12:27:00Z".to_string(),
            details: Some(json!({
                "field": "temperature",
                "valid_range": [0.0, 2.0],
                "provided_value": 2.5
            })),
        },
    );

    responses.insert(
        "server_overload",
        ErrorResponse {
            error: "Server Overload".to_string(),
            error_code: "TOO_MANY_REQUESTS".to_string(),
            message: "Server is currently processing too many requests. Please try again later."
                .to_string(),
            request_id: Some("overload-001".to_string()),
            timestamp: "2024-09-29T12:30:00Z".to_string(),
            details: Some(json!({
                "retry_after_seconds": 30,
                "current_load": 95.7,
                "max_concurrent_requests": 100
            })),
        },
    );

    responses.insert(
        "model_load_error",
        ErrorResponse {
            error: "Model Loading Error".to_string(),
            error_code: "MODEL_UNAVAILABLE".to_string(),
            message: "Failed to load requested model. Model file may be corrupted or missing."
                .to_string(),
            request_id: Some("model-error-001".to_string()),
            timestamp: "2024-09-29T12:35:00Z".to_string(),
            details: Some(json!({
                "model_id": "invalid_model",
                "error_type": "corrupted_gguf",
                "suggested_action": "verify model file integrity"
            })),
        },
    );

    responses
});

/// Health check response fixtures
pub static HEALTH_RESPONSES: LazyLock<HashMap<&'static str, HealthCheckResponse>> =
    LazyLock::new(|| {
        let mut responses = HashMap::new();

        responses.insert(
            "healthy_cpu_only",
            HealthCheckResponse {
                status: "healthy".to_string(),
                version: "1.0.0".to_string(),
                uptime_seconds: 3600,
                models_loaded: 1,
                system_metrics: SystemMetrics {
                    cpu_usage_percent: 25.5,
                    memory_usage_mb: 512,
                    memory_total_mb: 8192,
                    disk_usage_percent: 45.2,
                    active_requests: 3,
                    total_requests_processed: 1247,
                },
                device_status: DeviceStatus {
                    cpu_available: true,
                    gpu_available: false,
                    gpu_memory_total_mb: None,
                    gpu_memory_free_mb: None,
                    cuda_version: None,
                    compute_capability: None,
                },
            },
        );

        responses.insert(
            "healthy_gpu_enabled",
            HealthCheckResponse {
                status: "healthy".to_string(),
                version: "1.0.0".to_string(),
                uptime_seconds: 7200,
                models_loaded: 3,
                system_metrics: SystemMetrics {
                    cpu_usage_percent: 15.2,
                    memory_usage_mb: 2048,
                    memory_total_mb: 16384,
                    disk_usage_percent: 38.7,
                    active_requests: 8,
                    total_requests_processed: 5432,
                },
                device_status: DeviceStatus {
                    cpu_available: true,
                    gpu_available: true,
                    gpu_memory_total_mb: Some(8192),
                    gpu_memory_free_mb: Some(6144),
                    cuda_version: Some("12.0".to_string()),
                    compute_capability: Some("8.6".to_string()),
                },
            },
        );

        responses.insert(
            "degraded_high_load",
            HealthCheckResponse {
                status: "degraded".to_string(),
                version: "1.0.0".to_string(),
                uptime_seconds: 86400,
                models_loaded: 2,
                system_metrics: SystemMetrics {
                    cpu_usage_percent: 89.3,
                    memory_usage_mb: 7680,
                    memory_total_mb: 8192,
                    disk_usage_percent: 78.5,
                    active_requests: 95,
                    total_requests_processed: 125000,
                },
                device_status: DeviceStatus {
                    cpu_available: true,
                    gpu_available: true,
                    gpu_memory_total_mb: Some(8192),
                    gpu_memory_free_mb: Some(512),
                    cuda_version: Some("12.0".to_string()),
                    compute_capability: Some("8.6".to_string()),
                },
            },
        );

        responses
    });

/// Metrics response fixture (Prometheus format)
pub static PROMETHEUS_METRICS: LazyLock<&'static str> = LazyLock::new(|| {
    r#"# HELP bitnet_requests_total Total number of inference requests processed
# TYPE bitnet_requests_total counter
bitnet_requests_total{device="cpu",quantization="i2s",status="success"} 1247
bitnet_requests_total{device="gpu",quantization="tl2",status="success"} 856
bitnet_requests_total{device="cpu",quantization="i2s",status="error"} 23
bitnet_requests_total{device="gpu",quantization="tl2",status="error"} 12

# HELP bitnet_inference_duration_seconds Inference duration in seconds
# TYPE bitnet_inference_duration_seconds histogram
bitnet_inference_duration_seconds_bucket{device="cpu",quantization="i2s",le="0.5"} 234
bitnet_inference_duration_seconds_bucket{device="cpu",quantization="i2s",le="1.0"} 567
bitnet_inference_duration_seconds_bucket{device="cpu",quantization="i2s",le="2.0"} 1124
bitnet_inference_duration_seconds_bucket{device="cpu",quantization="i2s",le="+Inf"} 1247
bitnet_inference_duration_seconds_sum{device="cpu",quantization="i2s"} 892.5
bitnet_inference_duration_seconds_count{device="cpu",quantization="i2s"} 1247

# HELP bitnet_model_memory_usage_bytes Memory usage per loaded model
# TYPE bitnet_model_memory_usage_bytes gauge
bitnet_model_memory_usage_bytes{model="small_i2s_model"} 26214400
bitnet_model_memory_usage_bytes{model="large_tl2_model"} 524288000

# HELP bitnet_gpu_memory_usage_bytes GPU memory usage
# TYPE bitnet_gpu_memory_usage_bytes gauge
bitnet_gpu_memory_usage_bytes{type="total"} 8589934592
bitnet_gpu_memory_usage_bytes{type="used"} 2147483648
bitnet_gpu_memory_usage_bytes{type="free"} 6442450944

# HELP bitnet_quantization_accuracy Quantization accuracy metrics
# TYPE bitnet_quantization_accuracy gauge
bitnet_quantization_accuracy{quantization="i2s"} 0.991
bitnet_quantization_accuracy{quantization="tl1"} 0.983
bitnet_quantization_accuracy{quantization="tl2"} 0.987

# HELP bitnet_active_requests Current number of active requests
# TYPE bitnet_active_requests gauge
bitnet_active_requests 8

# HELP bitnet_tokens_per_second Inference throughput in tokens per second
# TYPE bitnet_tokens_per_second gauge
bitnet_tokens_per_second{device="cpu",quantization="i2s"} 45.7
bitnet_tokens_per_second{device="gpu",quantization="tl2"} 159.3
"#
});

/// Server-Sent Events format for streaming responses
pub fn format_as_sse(chunk: &StreamingChunk) -> String {
    if chunk.is_final {
        format!(
            "data: {}\n\nevent: done\ndata: {}\n\n",
            serde_json::to_string(chunk).unwrap(),
            json!({"request_id": chunk.request_id, "final_text": chunk.partial_text})
        )
    } else {
        format!("data: {}\n\n", serde_json::to_string(chunk).unwrap())
    }
}

/// Get response fixture by name and type
pub fn get_inference_response(name: &str) -> Option<&'static InferenceResponse> {
    BASIC_INFERENCE_RESPONSES.get(name)
}

pub fn get_streaming_response(name: &str) -> Option<&'static Vec<StreamingChunk>> {
    STREAMING_RESPONSES.get(name)
}

pub fn get_batch_response(name: &str) -> Option<&'static BatchResponse> {
    BATCH_RESPONSES.get(name)
}

pub fn get_error_response(name: &str) -> Option<&'static ErrorResponse> {
    ERROR_RESPONSES.get(name)
}

pub fn get_health_response(name: &str) -> Option<&'static HealthCheckResponse> {
    HEALTH_RESPONSES.get(name)
}

pub fn get_prometheus_metrics() -> &'static str {
    &PROMETHEUS_METRICS
}

/// Convert response to JSON for testing
pub fn response_to_json<T: Serialize>(response: &T) -> Value {
    serde_json::to_value(response).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_response_structure() {
        let response = get_inference_response("simple_question_i2s_cpu").unwrap();
        assert!(!response.text.is_empty());
        assert!(response.tokens_generated > 0);
        assert!(response.tokens_per_second > 0.0);
        assert!(response.accuracy_metrics.quantization_accuracy >= 0.99);
    }

    #[test]
    fn test_streaming_response_structure() {
        let chunks = get_streaming_response("bitnet_guide_stream").unwrap();
        assert!(!chunks.is_empty());

        let final_chunk = chunks.last().unwrap();
        assert!(final_chunk.is_final);

        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.chunk_index, i as u32);
            assert_eq!(chunk.request_id, "test-stream-001");
        }
    }

    #[test]
    fn test_sse_formatting() {
        let chunk = StreamingChunk {
            token: "test".to_string(),
            is_final: false,
            chunk_index: 0,
            request_id: "test-001".to_string(),
            partial_text: "test".to_string(),
            timestamp: "2024-09-29T12:00:00Z".to_string(),
        };

        let sse = format_as_sse(&chunk);
        assert!(sse.starts_with("data: "));
        assert!(sse.contains("test"));
    }

    #[test]
    fn test_error_response_structure() {
        let error = get_error_response("empty_prompt").unwrap();
        assert!(!error.error.is_empty());
        assert!(!error.error_code.is_empty());
        assert!(!error.message.is_empty());
        assert!(error.details.is_some());
    }

    #[test]
    fn test_health_response_structure() {
        let health = get_health_response("healthy_cpu_only").unwrap();
        assert_eq!(health.status, "healthy");
        assert!(health.system_metrics.cpu_usage_percent >= 0.0);
        assert!(health.device_status.cpu_available);
        assert!(!health.device_status.gpu_available);
    }

    #[test]
    fn test_batch_response_structure() {
        let batch = get_batch_response("small_batch_success").unwrap();
        assert_eq!(batch.responses.len(), 2);
        assert_eq!(batch.successful_requests, 2);
        assert_eq!(batch.failed_requests, 0);
        assert!(batch.batch_metrics.average_latency_ms > 0.0);
    }

    #[test]
    fn test_prometheus_metrics_format() {
        let metrics = get_prometheus_metrics();
        assert!(metrics.contains("# HELP"));
        assert!(metrics.contains("# TYPE"));
        assert!(metrics.contains("bitnet_requests_total"));
        assert!(metrics.contains("bitnet_inference_duration_seconds"));
    }

    #[test]
    fn test_response_json_conversion() {
        let response = get_inference_response("minimal_defaults").unwrap();
        let json_value = response_to_json(response);

        assert!(json_value.get("text").is_some());
        assert!(json_value.get("tokens_generated").is_some());
        assert!(json_value.get("accuracy_metrics").is_some());
    }
}

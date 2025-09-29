#![allow(unused)]
#![allow(dead_code)]

//! HTTP Request Fixtures for BitNet.rs Inference Server Testing
//!
//! This module provides comprehensive request fixtures for testing the production
//! inference server REST API endpoints with various scenarios and edge cases.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::LazyLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub prompt: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub device_preference: Option<String>,
    pub quantization_preference: Option<String>,
    pub stream: Option<bool>,
    pub model_id: Option<String>,
    pub stop_sequences: Option<Vec<String>>,
    pub seed: Option<u64>,
    pub request_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchInferenceRequest {
    pub requests: Vec<InferenceRequest>,
    pub batch_size: Option<u32>,
    pub priority: Option<String>,
    pub timeout_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestRequestMetadata {
    pub description: &'static str,
    pub test_scenario: &'static str,
    pub expected_status_code: u16,
    pub expected_response_time_ms: u64,
    pub device_requirements: Vec<&'static str>,
    pub validation_notes: &'static str,
}

/// Basic inference request fixtures
pub static BASIC_INFERENCE_REQUESTS: LazyLock<
    HashMap<&'static str, (InferenceRequest, TestRequestMetadata)>,
> = LazyLock::new(|| {
    let mut requests = HashMap::new();

    requests.insert(
        "simple_question",
        (
            InferenceRequest {
                prompt: "What is neural network quantization?".to_string(),
                max_tokens: Some(150),
                temperature: Some(0.7),
                top_p: Some(0.9),
                device_preference: Some("cpu".to_string()),
                quantization_preference: Some("i2s".to_string()),
                stream: Some(false),
                model_id: None,
                stop_sequences: None,
                seed: Some(42),
                request_id: Some("test-001".to_string()),
                top_k: None,
            },
            TestRequestMetadata {
                description: "Simple question about neural network quantization",
                test_scenario: "Basic synchronous inference with CPU device",
                expected_status_code: 200,
                expected_response_time_ms: 2000,
                device_requirements: vec!["cpu"],
                validation_notes: "Should return educational content about quantization",
            },
        ),
    );

    requests.insert(
        "gpu_inference",
        (
            InferenceRequest {
                prompt: "Explain the advantages of 1-bit neural networks in detail.".to_string(),
                max_tokens: Some(300),
                temperature: Some(0.5),
                top_p: Some(0.95),
                device_preference: Some("gpu".to_string()),
                quantization_preference: Some("tl2".to_string()),
                stream: Some(false),
                model_id: Some("large_tl2_model".to_string()),
                stop_sequences: Some(vec!["###".to_string(), "END".to_string()]),
                seed: Some(123),
                request_id: Some("test-gpu-001".to_string()),
                top_k: Some(50),
            },
            TestRequestMetadata {
                description: "GPU inference with TL2 quantization for detailed explanation",
                test_scenario: "GPU-accelerated inference with advanced quantization",
                expected_status_code: 200,
                expected_response_time_ms: 1500,
                device_requirements: vec!["gpu"],
                validation_notes: "Should utilize GPU tensor cores and TL2 quantization",
            },
        ),
    );

    requests.insert(
        "streaming_request",
        (
            InferenceRequest {
                prompt: "Write a comprehensive guide to BitNet architecture.".to_string(),
                max_tokens: Some(500),
                temperature: Some(0.3),
                top_p: Some(0.8),
                device_preference: Some("auto".to_string()),
                quantization_preference: Some("auto".to_string()),
                stream: Some(true),
                model_id: None,
                stop_sequences: None,
                seed: Some(456),
                request_id: Some("test-stream-001".to_string()),
                top_k: None,
            },
            TestRequestMetadata {
                description: "Streaming inference for long-form content generation",
                test_scenario: "Server-Sent Events streaming with automatic device selection",
                expected_status_code: 200,
                expected_response_time_ms: 5000,
                device_requirements: vec!["cpu", "gpu"],
                validation_notes: "Should stream tokens incrementally via SSE",
            },
        ),
    );

    requests.insert(
        "minimal_request",
        (
            InferenceRequest {
                prompt: "Hello".to_string(),
                max_tokens: None,
                temperature: None,
                top_p: None,
                device_preference: None,
                quantization_preference: None,
                stream: None,
                model_id: None,
                stop_sequences: None,
                seed: None,
                request_id: None,
                top_k: None,
            },
            TestRequestMetadata {
                description: "Minimal request with only required prompt field",
                test_scenario: "Default parameter inference with server-side defaults",
                expected_status_code: 200,
                expected_response_time_ms: 1000,
                device_requirements: vec!["cpu"],
                validation_notes: "Should use server defaults for all optional parameters",
            },
        ),
    );

    requests
});

/// Long prompt test fixtures for stress testing
pub static LONG_PROMPT_REQUESTS: LazyLock<
    HashMap<&'static str, (InferenceRequest, TestRequestMetadata)>,
> = LazyLock::new(|| {
    let mut requests = HashMap::new();

    let long_prompt = format!(
        "{}{}{}",
        "Analyze the following research paper abstract and provide detailed insights: ",
        "Neural network quantization has emerged as a critical technique for deploying large language models in resource-constrained environments. ".repeat(20),
        "Please provide a comprehensive analysis covering methodology, advantages, limitations, and future research directions."
    );

    requests.insert(
        "long_context",
        (
            InferenceRequest {
                prompt: long_prompt,
                max_tokens: Some(800),
                temperature: Some(0.6),
                top_p: Some(0.9),
                device_preference: Some("gpu".to_string()),
                quantization_preference: Some("i2s".to_string()),
                stream: Some(true),
                model_id: Some("large_tl2_model".to_string()),
                stop_sequences: None,
                seed: Some(789),
                request_id: Some("test-long-001".to_string()),
                top_k: None,
            },
            TestRequestMetadata {
                description: "Long context prompt for memory and performance testing",
                test_scenario: "Large context window processing with streaming output",
                expected_status_code: 200,
                expected_response_time_ms: 8000,
                device_requirements: vec!["gpu"],
                validation_notes: "Should handle long context without memory issues",
            },
        ),
    );

    requests
});

/// Batch processing request fixtures
pub static BATCH_REQUESTS: LazyLock<
    HashMap<&'static str, (BatchInferenceRequest, TestRequestMetadata)>,
> = LazyLock::new(|| {
    let mut requests = HashMap::new();

    requests.insert(
        "small_batch",
        (
            BatchInferenceRequest {
                requests: vec![
                    InferenceRequest {
                        prompt: "What is machine learning?".to_string(),
                        max_tokens: Some(100),
                        temperature: Some(0.7),
                        top_p: Some(0.9),
                        device_preference: Some("cpu".to_string()),
                        quantization_preference: Some("i2s".to_string()),
                        stream: Some(false),
                        model_id: None,
                        stop_sequences: None,
                        seed: Some(1),
                        request_id: Some("batch-1-1".to_string()),
                        top_k: None,
                    },
                    InferenceRequest {
                        prompt: "Explain deep learning briefly.".to_string(),
                        max_tokens: Some(100),
                        temperature: Some(0.7),
                        top_p: Some(0.9),
                        device_preference: Some("cpu".to_string()),
                        quantization_preference: Some("i2s".to_string()),
                        stream: Some(false),
                        model_id: None,
                        stop_sequences: None,
                        seed: Some(2),
                        request_id: Some("batch-1-2".to_string()),
                        top_k: None,
                    },
                ],
                batch_size: Some(2),
                priority: Some("normal".to_string()),
                timeout_ms: Some(5000),
            },
            TestRequestMetadata {
                description: "Small batch of 2 requests for basic batch processing",
                test_scenario: "Concurrent batch processing with CPU device",
                expected_status_code: 200,
                expected_response_time_ms: 3000,
                device_requirements: vec!["cpu"],
                validation_notes: "Should process both requests efficiently in parallel",
            },
        ),
    );

    requests.insert("large_batch", (
        BatchInferenceRequest {
            requests: (0..10).map(|i| InferenceRequest {
                prompt: format!("Question {}: Explain neural networks from perspective {}.", i + 1, i + 1),
                max_tokens: Some(150),
                temperature: Some(0.5),
                top_p: Some(0.9),
                device_preference: Some("gpu".to_string()),
                quantization_preference: Some("tl1".to_string()),
                stream: Some(false),
                model_id: None,
                stop_sequences: None,
                seed: Some(i as u64 + 100),
                request_id: Some(format!("batch-large-{}", i + 1)),
                top_k: None,
            }).collect(),
            batch_size: Some(5),
            priority: Some("high".to_string()),
            timeout_ms: Some(15000),
        },
        TestRequestMetadata {
            description: "Large batch of 10 requests for stress testing",
            test_scenario: "High-priority batch processing with GPU acceleration",
            expected_status_code: 200,
            expected_response_time_ms: 12000,
            device_requirements: vec!["gpu"],
            validation_notes: "Should handle large batch efficiently with proper resource management",
        }
    ));

    requests
});

/// Error condition request fixtures
pub static ERROR_REQUESTS: LazyLock<HashMap<&'static str, (Value, TestRequestMetadata)>> =
    LazyLock::new(|| {
        let mut requests = HashMap::new();

        requests.insert(
            "empty_prompt",
            (
                json!({
                    "prompt": "",
                    "max_tokens": 100
                }),
                TestRequestMetadata {
                    description: "Empty prompt should trigger validation error",
                    test_scenario: "Input validation for empty prompt field",
                    expected_status_code: 400,
                    expected_response_time_ms: 100,
                    device_requirements: vec![],
                    validation_notes: "Should return 400 Bad Request with validation error message",
                },
            ),
        );

        requests.insert(
            "missing_prompt",
            (
                json!({
                    "max_tokens": 100,
                    "temperature": 0.7
                }),
                TestRequestMetadata {
                    description: "Missing required prompt field",
                    test_scenario: "Schema validation for missing required fields",
                    expected_status_code: 400,
                    expected_response_time_ms: 100,
                    device_requirements: vec![],
                    validation_notes: "Should return 400 Bad Request indicating missing prompt",
                },
            ),
        );

        requests.insert(
            "invalid_temperature",
            (
                json!({
                    "prompt": "Test prompt",
                    "temperature": 2.5,
                    "max_tokens": 100
                }),
                TestRequestMetadata {
                    description: "Temperature value outside valid range (0.0-2.0)",
                    test_scenario: "Parameter validation for temperature bounds",
                    expected_status_code: 400,
                    expected_response_time_ms: 100,
                    device_requirements: vec![],
                    validation_notes: "Should return 400 Bad Request with temperature range error",
                },
            ),
        );

        requests.insert(
            "negative_max_tokens",
            (
                json!({
                    "prompt": "Test prompt",
                    "max_tokens": -10
                }),
                TestRequestMetadata {
                    description: "Negative max_tokens value",
                    test_scenario: "Parameter validation for negative values",
                    expected_status_code: 400,
                    expected_response_time_ms: 100,
                    device_requirements: vec![],
                    validation_notes: "Should return 400 Bad Request for negative max_tokens",
                },
            ),
        );

        requests.insert(
            "excessive_max_tokens",
            (
                json!({
                    "prompt": "Test prompt",
                    "max_tokens": 100000
                }),
                TestRequestMetadata {
                    description: "Excessive max_tokens exceeding server limits",
                    test_scenario: "Server-side limit enforcement",
                    expected_status_code: 400,
                    expected_response_time_ms: 100,
                    device_requirements: vec![],
                    validation_notes: "Should return 400 Bad Request for token limit exceeded",
                },
            ),
        );

        requests.insert(
            "malformed_json",
            (
                json!("{ invalid json }"),
                TestRequestMetadata {
                    description: "Malformed JSON request body",
                    test_scenario: "JSON parsing error handling",
                    expected_status_code: 400,
                    expected_response_time_ms: 100,
                    device_requirements: vec![],
                    validation_notes: "Should return 400 Bad Request with JSON parsing error",
                },
            ),
        );

        requests.insert(
            "unsupported_device",
            (
                json!({
                    "prompt": "Test prompt",
                    "device_preference": "quantum"
                }),
                TestRequestMetadata {
                    description: "Unsupported device preference",
                    test_scenario: "Device validation and fallback handling",
                    expected_status_code: 400,
                    expected_response_time_ms: 100,
                    device_requirements: vec![],
                    validation_notes: "Should return 400 Bad Request for unsupported device",
                },
            ),
        );

        requests
    });

/// Concurrent load testing fixtures
pub fn generate_concurrent_requests(count: usize, base_prompt: &str) -> Vec<InferenceRequest> {
    (0..count)
        .map(|i| InferenceRequest {
            prompt: format!("{} (Request {})", base_prompt, i + 1),
            max_tokens: Some(100 + (i % 50) as u32),
            temperature: Some(0.5 + (i % 10) as f32 * 0.05),
            top_p: Some(0.8 + (i % 20) as f32 * 0.01),
            device_preference: if i % 2 == 0 {
                Some("cpu".to_string())
            } else {
                Some("gpu".to_string())
            },
            quantization_preference: match i % 3 {
                0 => Some("i2s".to_string()),
                1 => Some("tl1".to_string()),
                _ => Some("tl2".to_string()),
            },
            stream: Some(i % 3 == 0),
            model_id: None,
            stop_sequences: None,
            seed: Some(i as u64 + 1000),
            request_id: Some(format!("concurrent-{}", i + 1)),
            top_k: None,
        })
        .collect()
}

/// Security testing fixtures
pub static SECURITY_TEST_REQUESTS: LazyLock<HashMap<&'static str, (Value, TestRequestMetadata)>> =
    LazyLock::new(|| {
        let mut requests = HashMap::new();

        requests.insert(
            "sql_injection_attempt",
            (
                json!({
                    "prompt": "'; DROP TABLE users; --",
                    "max_tokens": 100
                }),
                TestRequestMetadata {
                    description: "SQL injection attempt in prompt",
                    test_scenario: "Security validation against injection attacks",
                    expected_status_code: 400,
                    expected_response_time_ms: 100,
                    device_requirements: vec![],
                    validation_notes: "Should sanitize or reject potentially malicious input",
                },
            ),
        );

        requests.insert(
            "xss_attempt",
            (
                json!({
                    "prompt": "<script>alert('xss')</script>",
                    "max_tokens": 100
                }),
                TestRequestMetadata {
                    description: "XSS script injection in prompt",
                    test_scenario: "Web security validation against script injection",
                    expected_status_code: 400,
                    expected_response_time_ms: 100,
                    device_requirements: vec![],
                    validation_notes: "Should sanitize HTML/script content in prompts",
                },
            ),
        );

        requests.insert(
            "excessive_length_prompt",
            (
                json!({
                    "prompt": "A".repeat(1000000),
                    "max_tokens": 100
                }),
                TestRequestMetadata {
                    description: "Extremely long prompt for DoS testing",
                    test_scenario: "Request size limit enforcement",
                    expected_status_code: 413,
                    expected_response_time_ms: 100,
                    device_requirements: vec![],
                    validation_notes: "Should return 413 Payload Too Large for oversized requests",
                },
            ),
        );

        requests
    });

/// Get request fixture by name and type
pub fn get_basic_request(name: &str) -> Option<&'static (InferenceRequest, TestRequestMetadata)> {
    BASIC_INFERENCE_REQUESTS.get(name)
}

pub fn get_long_prompt_request(
    name: &str,
) -> Option<&'static (InferenceRequest, TestRequestMetadata)> {
    LONG_PROMPT_REQUESTS.get(name)
}

pub fn get_batch_request(
    name: &str,
) -> Option<&'static (BatchInferenceRequest, TestRequestMetadata)> {
    BATCH_REQUESTS.get(name)
}

pub fn get_error_request(name: &str) -> Option<&'static (Value, TestRequestMetadata)> {
    ERROR_REQUESTS.get(name)
}

pub fn get_security_request(name: &str) -> Option<&'static (Value, TestRequestMetadata)> {
    SECURITY_TEST_REQUESTS.get(name)
}

/// Get all request names by category
pub fn get_all_basic_request_names() -> Vec<&'static str> {
    BASIC_INFERENCE_REQUESTS.keys().copied().collect()
}

pub fn get_all_error_request_names() -> Vec<&'static str> {
    ERROR_REQUESTS.keys().copied().collect()
}

pub fn get_all_security_request_names() -> Vec<&'static str> {
    SECURITY_TEST_REQUESTS.keys().copied().collect()
}

/// Convert InferenceRequest to JSON Value for HTTP testing
pub fn request_to_json(request: &InferenceRequest) -> Value {
    serde_json::to_value(request).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_request_fixtures() {
        let names = get_all_basic_request_names();
        assert!(!names.is_empty());

        for name in names {
            let (request, metadata) = get_basic_request(name).unwrap();
            assert!(!request.prompt.is_empty());
            assert!(!metadata.description.is_empty());
            assert!(metadata.expected_status_code >= 200 && metadata.expected_status_code < 300);
        }
    }

    #[test]
    fn test_error_request_fixtures() {
        let names = get_all_error_request_names();
        assert!(!names.is_empty());

        for name in names {
            let (_, metadata) = get_error_request(name).unwrap();
            assert!(metadata.expected_status_code >= 400);
            assert!(!metadata.validation_notes.is_empty());
        }
    }

    #[test]
    fn test_concurrent_request_generation() {
        let requests = generate_concurrent_requests(50, "Test concurrent processing");
        assert_eq!(requests.len(), 50);

        // Verify each request has unique identifier
        let mut request_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        for request in &requests {
            if let Some(id) = &request.request_id {
                assert!(request_ids.insert(id.clone()));
            }
        }
        assert_eq!(request_ids.len(), 50);
    }

    #[test]
    fn test_batch_request_structure() {
        let (batch, metadata) = get_batch_request("small_batch").unwrap();
        assert_eq!(batch.requests.len(), 2);
        assert!(batch.batch_size.is_some());
        assert!(metadata.expected_response_time_ms > 0);
    }

    #[test]
    fn test_request_json_conversion() {
        let (request, _) = get_basic_request("simple_question").unwrap();
        let json_value = request_to_json(request);

        assert!(json_value.get("prompt").is_some());
        assert!(json_value.get("max_tokens").is_some());
        assert!(json_value.get("temperature").is_some());
    }

    #[test]
    fn test_security_request_fixtures() {
        let names = get_all_security_request_names();
        assert!(!names.is_empty());

        for name in names {
            let (_, metadata) = get_security_request(name).unwrap();
            assert!(metadata.expected_status_code >= 400);
            assert!(
                metadata.test_scenario.contains("security")
                    || metadata.test_scenario.contains("Security")
                    || metadata.description.contains("injection")
                    || metadata.description.contains("DoS")
            );
        }
    }
}

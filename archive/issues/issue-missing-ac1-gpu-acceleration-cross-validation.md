# [Testing] Missing AC1 GPU Acceleration Cross-Validation Tests

## Problem Description

Critical testing gap identified: AC1 (HTTP REST API surface area) implementation lacks comprehensive GPU acceleration cross-validation tests. While the production inference server provides GPU-CPU device routing and the quantization infrastructure includes GPU-CPU parity tests, there are no specific tests validating that AC1's REST API endpoints correctly utilize GPU acceleration and maintain cross-validation compatibility with the existing cross-validation framework.

## Environment

- **Affected Components**:
  - `crates/bitnet-server/tests/ac01_rest_api_inference.rs` - AC1 REST API tests
  - `crossval/` - Cross-validation framework infrastructure
  - `crates/bitnet-kernels/` - GPU acceleration kernels
  - `crates/bitnet-inference/` - Device-aware inference engine
- **Missing Test Coverage**: GPU acceleration validation within AC1 HTTP endpoints
- **Impact**: Production deployments may not validate GPU acceleration correctness through REST API

## Root Cause Analysis

### Current Test Coverage Analysis

**Existing GPU Tests**:
- ✅ GPU kernels have smoke tests: `cargo gpu-smoke`
- ✅ GPU-CPU parity tests exist in quantization layer
- ✅ Device-aware inference has unit tests
- ✅ Cross-validation framework exists with C++ reference

**AC1 Test Coverage**:
- ✅ `ac01_rest_api_inference.rs` tests HTTP API surface area
- ✅ Request/response format validation
- ❌ **Missing**: GPU acceleration validation through REST endpoints
- ❌ **Missing**: Cross-validation integration with HTTP API
- ❌ **Missing**: Device routing validation in production server context

**Gap Identified**:
```rust
// File: crates/bitnet-server/tests/ac01_rest_api_inference.rs
// Current tests validate HTTP API but not GPU acceleration:

#[tokio::test]
async fn test_ac1_inference_endpoint_structure() {
    // ✅ Tests HTTP request/response format
    // ❌ Does NOT test GPU acceleration
    // ❌ Does NOT validate cross-device consistency
}
```

### Technical Root Causes

1. **Test Separation**: AC1 tests focus on HTTP layer, GPU tests focus on kernel layer
2. **Integration Gap**: No tests bridging REST API → Device Selection → GPU Acceleration
3. **Cross-Validation Isolation**: Existing crossval framework not integrated with HTTP server tests
4. **Production Validation Gap**: Server-level GPU acceleration not validated in production context

## Impact Assessment

- **Severity**: High (Production reliability)
- **Impact**:
  - GPU acceleration correctness not validated through production HTTP API
  - Potential silent failures when REST API requests route to GPU
  - Cross-validation framework not exercised at server layer
  - AC15 (device-aware routing) implementation cannot be fully validated
- **Affected Workflows**:
  - Production deployments with GPU acceleration
  - HTTP API clients expecting GPU-accelerated inference
  - DevOps validation of server deployment correctness

## Proposed Solution

Implement comprehensive AC1 GPU acceleration cross-validation tests that bridge the HTTP API layer with GPU device validation and cross-validation framework integration.

### Implementation Plan

#### 1. AC1 GPU Acceleration Integration Tests

**A. HTTP-Level GPU Validation**:
```rust
// File: crates/bitnet-server/tests/ac01_gpu_acceleration_tests.rs

#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_ac1_gpu_acceleration_inference_endpoint() -> Result<()> {
    // AC1: HTTP server provides complete REST API surface area
    // VALIDATION: GPU acceleration through REST API

    let server = test_server_with_gpu_enabled().await?;

    // 1. Verify GPU is available and selected
    let device_info = server.get_device_info().await?;
    assert_eq!(device_info.device_type, DeviceType::Gpu);

    // 2. HTTP inference request with GPU routing
    let request = InferenceRequest {
        prompt: "Explain quantum computing".to_string(),
        max_tokens: 50,
        temperature: 0.7,
        device_preference: Some(DeviceType::Gpu), // AC15 integration
    };

    let response = server
        .post("/v1/inference")
        .json(&request)
        .send()
        .await?;

    // 3. Validate HTTP response structure (AC1)
    assert_eq!(response.status(), 200);
    let inference_response: InferenceResponse = response.json().await?;

    // 4. Validate GPU was actually used
    assert_eq!(inference_response.device_used, Some(DeviceType::Gpu));
    assert!(inference_response.metrics.gpu_utilization.is_some());

    // 5. Validate inference quality (cross-validation requirement)
    validate_inference_quality(&inference_response.text).await?;

    Ok(())
}

#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_ac1_gpu_cpu_parity_through_http_api() -> Result<()> {
    // AC1 + Cross-validation: REST API maintains GPU-CPU parity

    let server = test_server_with_multi_device().await?;

    let test_prompt = "What is machine learning?";

    // GPU inference via HTTP
    let gpu_request = InferenceRequest {
        prompt: test_prompt.to_string(),
        max_tokens: 30,
        temperature: 0.0, // Deterministic
        device_preference: Some(DeviceType::Gpu),
        seed: Some(42), // Reproducible
    };

    let gpu_response = server
        .post("/v1/inference")
        .json(&gpu_request)
        .send()
        .await?;

    // CPU inference via HTTP
    let cpu_request = InferenceRequest {
        prompt: test_prompt.to_string(),
        max_tokens: 30,
        temperature: 0.0,
        device_preference: Some(DeviceType::Cpu),
        seed: Some(42),
    };

    let cpu_response = server
        .post("/v1/inference")
        .json(&cpu_request)
        .send()
        .await?;

    // Cross-validation: GPU and CPU should produce equivalent results
    let gpu_result: InferenceResponse = gpu_response.json().await?;
    let cpu_result: InferenceResponse = cpu_response.json().await?;

    // Validate output equivalence (within tolerance)
    let similarity = calculate_semantic_similarity(&gpu_result.text, &cpu_result.text)?;
    assert!(similarity > 0.95, "GPU-CPU inference parity violation: similarity = {}", similarity);

    // Validate quantization consistency
    validate_quantization_consistency(&gpu_result.metrics, &cpu_result.metrics)?;

    Ok(())
}
```

#### 2. Cross-Validation Framework Integration

**A. HTTP Server Cross-Validation**:
```rust
// File: crates/bitnet-server/tests/ac01_crossval_integration.rs

#[cfg(all(feature = "gpu", feature = "crossval"))]
#[tokio::test]
async fn test_ac1_crossval_integration() -> Result<()> {
    // AC1 + Cross-validation: HTTP API integrates with crossval framework

    use crossval::{CrossValidationFramework, ValidationTarget};

    let server = test_server_with_crossval().await?;
    let crossval = CrossValidationFramework::new()?;

    // HTTP inference for cross-validation
    let test_cases = crossval.get_standard_test_cases()?;

    for test_case in test_cases {
        let request = InferenceRequest {
            prompt: test_case.prompt.clone(),
            max_tokens: test_case.expected_tokens,
            temperature: 0.0,
            seed: Some(test_case.seed),
        };

        let response = server
            .post("/v1/inference")
            .json(&request)
            .send()
            .await?;

        let inference_result: InferenceResponse = response.json().await?;

        // Cross-validate against reference implementation
        let validation_result = crossval.validate_inference(
            &test_case,
            &inference_result.text,
            ValidationTarget::RestApiLayer,
        ).await?;

        assert!(validation_result.passed,
               "Cross-validation failed for test case {}: {}",
               test_case.id, validation_result.error_detail);
    }

    Ok(())
}
```

#### 3. Production Validation Tests

**A. AC1 Production GPU Deployment Validation**:
```rust
// File: crates/bitnet-server/tests/ac01_production_validation.rs

#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_ac1_production_gpu_deployment_validation() -> Result<()> {
    // AC1 Production: Complete validation of GPU acceleration in production context

    let server = production_test_server().await?;

    // 1. System validation
    let health_response = server.get("/health/ready").send().await?;
    assert_eq!(health_response.status(), 200);

    let health_data: HealthResponse = health_response.json().await?;
    assert!(health_data.gpu_available);
    assert!(health_data.gpu_memory_sufficient);

    // 2. Model management with GPU validation (AC3 integration)
    let model_status = server.get("/v1/models/status").send().await?;
    let models: ModelStatusResponse = model_status.json().await?;

    assert!(models.models.iter().any(|m| m.supports_gpu));

    // 3. Streaming inference with GPU (AC7 integration)
    let stream_request = StreamInferenceRequest {
        prompt: "Explain neural networks".to_string(),
        max_tokens: 100,
        device_preference: Some(DeviceType::Gpu),
    };

    let mut stream = server
        .post("/v1/stream")
        .json(&stream_request)
        .send()
        .await?
        .bytes_stream();

    let mut token_count = 0;
    let mut gpu_confirmed = false;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        let event: StreamEvent = serde_json::from_slice(&chunk)?;

        match event.event_type {
            StreamEventType::Token => token_count += 1,
            StreamEventType::Metrics => {
                if let Some(gpu_util) = event.data.gpu_utilization {
                    assert!(gpu_util > 0.0, "GPU utilization should be > 0");
                    gpu_confirmed = true;
                }
            },
            _ => {}
        }
    }

    assert!(token_count > 0, "Should generate tokens");
    assert!(gpu_confirmed, "GPU utilization should be confirmed in stream");

    Ok(())
}
```

#### 4. Performance Validation

**A. AC1 GPU Performance Validation**:
```rust
// File: crates/bitnet-server/tests/ac01_performance_validation.rs

#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_ac1_gpu_performance_requirements() -> Result<()> {
    // AC1 + AC10: Performance requirements with GPU acceleration

    let server = test_server_with_gpu().await?;

    // Performance test: 100-token inference < 2 seconds (AC10)
    let start = std::time::Instant::now();

    let request = InferenceRequest {
        prompt: "Write a technical explanation of".to_string(),
        max_tokens: 100,
        device_preference: Some(DeviceType::Gpu),
    };

    let response = server
        .post("/v1/inference")
        .json(&request)
        .send()
        .await?;

    let duration = start.elapsed();

    assert_eq!(response.status(), 200);
    let result: InferenceResponse = response.json().await?;

    // Validate performance requirements
    assert!(duration.as_secs() < 2, "Response time {} exceeds 2s requirement", duration.as_secs_f64());
    assert!(result.token_count >= 90, "Should generate close to requested tokens");
    assert_eq!(result.device_used, Some(DeviceType::Gpu));

    // Validate memory usage (AC10)
    let memory_usage = result.metrics.peak_memory_usage_bytes;
    assert!(memory_usage < 8_000_000_000, "Memory usage {} exceeds 8GB limit", memory_usage);

    Ok(())
}
```

## Testing Strategy

### Unit Tests
- HTTP API endpoint GPU routing validation
- Device selection logic through REST interface
- Error handling for GPU failures in production context

### Integration Tests
- End-to-end HTTP → GPU acceleration → Response validation
- Cross-validation framework integration with server layer
- Multi-device consistency through REST API

### Performance Tests
- GPU acceleration performance through HTTP endpoints
- Concurrent request handling with GPU resource management
- Memory usage validation under GPU workloads

### Production Tests
- Container deployment with GPU acceleration validation
- Kubernetes health check integration with GPU status
- Real-world workload simulation with GPU routing

## Implementation Tasks

### Phase 1: Core AC1 GPU Integration Tests
- [ ] Implement `ac01_gpu_acceleration_tests.rs` with HTTP-level GPU validation
- [ ] Add GPU-CPU parity tests through REST API endpoints
- [ ] Integrate device preference handling in HTTP request processing
- [ ] Validate GPU utilization metrics in HTTP responses

### Phase 2: Cross-Validation Integration
- [ ] Implement `ac01_crossval_integration.rs` with crossval framework
- [ ] Add HTTP server integration with existing cross-validation infrastructure
- [ ] Implement REST API cross-validation test cases
- [ ] Validate server-layer inference against reference implementations

### Phase 3: Production Validation
- [ ] Implement `ac01_production_validation.rs` with production scenarios
- [ ] Add streaming inference GPU validation tests
- [ ] Implement model management GPU compatibility tests
- [ ] Add health check GPU status validation

### Phase 4: Performance and Load Testing
- [ ] Implement `ac01_performance_validation.rs` with GPU performance tests
- [ ] Add concurrent request GPU resource management tests
- [ ] Implement memory usage validation under GPU workloads
- [ ] Add performance regression detection for GPU acceleration

## Acceptance Criteria

- [ ] AC1 HTTP endpoints correctly utilize GPU acceleration when requested
- [ ] GPU-CPU parity maintained through REST API interface
- [ ] Cross-validation framework integrates with HTTP server layer
- [ ] Production deployment scenarios validate GPU acceleration
- [ ] Performance requirements (AC10) met with GPU acceleration
- [ ] Error handling covers GPU failure scenarios in production context
- [ ] Device routing (AC15) validated through HTTP API
- [ ] Streaming inference (AC7) correctly utilizes GPU acceleration

## Performance Targets

- **HTTP GPU Inference**: Complete in <2 seconds for 100 tokens (AC10)
- **Cross-Validation**: GPU-CPU similarity >95% through REST API
- **Concurrent Requests**: Support 100+ concurrent GPU requests
- **Memory Usage**: <8GB for 2B parameter models via HTTP (AC10)
- **GPU Utilization**: >70% during active inference requests

## Dependencies

- GPU-enabled test environment with CUDA support
- Cross-validation framework integration
- Production test server infrastructure
- Performance benchmarking framework
- HTTP client testing utilities

## Related Issues

- Issue #251: Production-Ready Inference Server (AC1 implementation)
- Issue #260: Mock Elimination (GPU acceleration validation)
- Cross-validation framework enhancement (GPU-CPU parity)
- AC15: Device-aware inference routing implementation

## Labels

- `testing`
- `gpu-acceleration`
- `cross-validation`
- `ac1-rest-api`
- `production-readiness`
- `priority-high`
- `issue-251`

## Cross-References

- **AC1**: HTTP server REST API surface area (primary focus)
- **AC10**: Performance requirements validation
- **AC15**: Device-aware inference routing
- **AC7**: Streaming inference GPU integration
- **Issue #260**: Real computation validation (GPU acceleration)
- **Cross-validation framework**: GPU-CPU parity validation

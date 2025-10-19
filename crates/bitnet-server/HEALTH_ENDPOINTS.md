# BitNet.rs Health Endpoints

Production-ready health check endpoints for BitNet inference server with real system metrics.

## Overview

The BitNet server provides three health endpoints optimized for Kubernetes-style orchestration:

- **`GET /health`** - Comprehensive health check with full system status
- **`GET /health/live`** - Fast liveness probe (<200ms response time)
- **`GET /health/ready`** - Readiness probe for traffic routing decisions

## Endpoints

### `/health` - Comprehensive Health Check

Returns detailed system health information including component status, build info, and metrics.

**Response Schema:**
```json
{
  "status": "healthy" | "degraded" | "unhealthy",
  "timestamp": "2025-10-19T...",
  "uptime_seconds": 3600,
  "version": "0.1.0",
  "build": {
    "version": "0.1.0",
    "git_sha": "abc123...",
    "git_branch": "main",
    "build_timestamp": "2025-10-19T...",
    "rustc_version": "1.90.0",
    "cargo_target": "x86_64-unknown-linux-gnu",
    "cargo_profile": "release",
    "cuda_version": "12.3" // Optional, only with GPU features
  },
  "components": {
    "model": {
      "status": "healthy",
      "message": "Model loaded and ready",
      "last_check": "2025-10-19T...",
      "response_time_ms": 5
    },
    "memory": {
      "status": "healthy",
      "message": "Memory usage normal: 45.2%",
      "last_check": "2025-10-19T...",
      "response_time_ms": 3
    },
    "inference_engine": {
      "status": "healthy",
      "message": "Inference engine responsive",
      "last_check": "2025-10-19T...",
      "response_time_ms": 2
    },
    "gpu": { // Only with GPU features
      "status": "healthy",
      "message": "GPU available: CUDA device 0",
      "last_check": "2025-10-19T...",
      "response_time_ms": 8
    }
  },
  "metrics": {
    "active_requests": 0,
    "total_requests": 1234,
    "error_rate_percent": 0.5,
    "avg_response_time_ms": 125.3,
    "memory_usage_mb": 2048.5,
    "tokens_per_second": 15.7,
    "cpu_usage_percent": 45.2, // Optional
    "gpu_memory_mb": 1024.0    // Optional, only with GPU features
  }
}
```

**HTTP Status Codes:**
- `200 OK` - System is healthy
- `503 Service Unavailable` - System is degraded or unhealthy (default fail-fast mode)

**Note:** With `--features degraded-ok`, degraded status returns `200 OK` instead of `503`.

### `/health/live` - Liveness Probe

Fast endpoint for Kubernetes liveness checks. Returns minimal response optimized for speed.

**Behavior:**
- Returns `Degraded` (503) during first 5 seconds of startup
- Returns `Healthy` (200) once basic functionality is verified
- Response time: <200ms (including CPU measurement overhead)

**Response:** HTTP status code only (no JSON body required)

**HTTP Status Codes:**
- `200 OK` - Process is alive and functional
- `503 Service Unavailable` - Process is degraded or unhealthy

**Cache Headers:** All responses include `Cache-Control: no-store`

### `/health/ready` - Readiness Probe

Comprehensive readiness check for Kubernetes traffic routing.

**Checks:**
- Model loaded and accessible
- Memory usage within acceptable thresholds
- Inference engine responsive
- Device availability (CPU/GPU)

**Response:** HTTP status code only (no JSON body required)

**HTTP Status Codes:**
- `200 OK` - Ready to accept traffic
- `503 Service Unavailable` - Not ready (always fail-fast, even with `degraded-ok` feature)

**Cache Headers:** All responses include `Cache-Control: no-store`

## Component Health Status

Each component reports one of three statuses:

- **`healthy`** - Component is operating normally
- **`degraded`** - Component is functional but experiencing issues
- **`unhealthy`** - Component has failed

### Critical Components

The following components are considered critical for overall system health:

- `model` - Model loading and management
- `memory` - System memory availability
- `inference_engine` - Inference pipeline responsiveness

If any critical component is `unhealthy`, overall status becomes `unhealthy`.

## Metrics Collection

Real-time system metrics include:

- **Memory Usage** - Current system memory consumption in MB
- **CPU Usage** - Current CPU utilization percentage (0-100%)
- **GPU Memory** - GPU memory usage when GPU features enabled
- **Active Requests** - Number of in-flight inference requests
- **Total Requests** - Cumulative request count
- **Error Rate** - Percentage of failed requests
- **Tokens/Second** - Inference throughput
- **Response Time** - Average request latency in milliseconds

## Feature Flags

### `degraded-ok`

Changes HTTP status mapping for `/health` endpoint:

**Default (fail-fast):**
- `healthy` → 200 OK
- `degraded` → 503 Service Unavailable
- `unhealthy` → 503 Service Unavailable

**With `--features degraded-ok`:**
- `healthy` → 200 OK
- `degraded` → 200 OK (graceful degradation)
- `unhealthy` → 503 Service Unavailable

**Note:** Readiness probe (`/health/ready`) always uses fail-fast mapping regardless of feature.

### GPU Features

When built with `--features gpu` or `--features cuda`, health checks include:

- GPU component status
- GPU memory metrics
- CUDA version information
- Device-specific health monitoring

## Example Usage

### Query Health Status

```bash
# Comprehensive health check
curl http://localhost:8080/health | jq

# Liveness probe (fast)
curl -w "%{http_code}\n" http://localhost:8080/health/live

# Readiness probe
curl -w "%{http_code}\n" http://localhost:8080/health/ready
```

### Kubernetes Integration

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: bitnet-server
spec:
  containers:
  - name: bitnet
    image: bitnet-server:latest
    ports:
    - containerPort: 8080
    livenessProbe:
      httpGet:
        path: /health/live
        port: 8080
      initialDelaySeconds: 10
      periodSeconds: 10
      timeoutSeconds: 2
      failureThreshold: 3
    readinessProbe:
      httpGet:
        path: /health/ready
        port: 8080
      initialDelaySeconds: 5
      periodSeconds: 5
      timeoutSeconds: 2
      failureThreshold: 2
```

### Run Example Server

```bash
# Start example health endpoint server
cargo run -p bitnet-server --no-default-features --features cpu --example health_endpoints

# Query endpoints
curl http://localhost:8080/health | jq
curl http://localhost:8080/health/live
curl http://localhost:8080/health/ready
```

## Testing

Run integration tests:

```bash
# Run health endpoint integration tests
cargo test -p bitnet-server --no-default-features --features cpu --test health_endpoints_integration

# Run all server tests
cargo test -p bitnet-server --no-default-features --features cpu
```

## Performance

- **Liveness probe**: <200ms response time (including 100ms CPU measurement)
- **Readiness probe**: <500ms response time (includes component checks)
- **Comprehensive health**: <1s response time (full system metrics)

All endpoints include `Cache-Control: no-store` headers to prevent caching of stale health data.

## Implementation Details

- **Real Metrics**: Health checks collect actual system metrics using `sysinfo` crate
- **Non-Blocking**: Metrics collection uses async operations
- **Startup Grace**: Liveness probe returns `Degraded` for first 5 seconds to allow initialization
- **Component Isolation**: Individual component health tracked independently
- **Thread-Safe**: Health checker is `Arc`-wrapped for concurrent access
- **Production-Ready**: Used in production server with full integration

## See Also

- `/health/live` documentation: [Kubernetes Liveness Probes](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/)
- `/health/ready` documentation: [Kubernetes Readiness Probes](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/)
- Integration tests: `crates/bitnet-server/tests/health_endpoints_integration.rs`
- Example server: `crates/bitnet-server/examples/health_endpoints.rs`

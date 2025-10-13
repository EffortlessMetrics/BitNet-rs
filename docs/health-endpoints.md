# Health Endpoints

The BitNet server provides health check endpoints for monitoring and load balancer integration.

## Endpoints

### `/health`
Returns overall system health status.

### `/health/live`
Basic liveness check - returns 200 OK if the server is running.

### `/health/ready`
Readiness check - returns 200 OK only when the server is ready to handle inference requests.

## Health Status Mapping

By default the server uses a **fail-fast** policy:

| Status | HTTP Code | Description |
|--------|-----------|-------------|
| `Healthy` | 200 OK | All systems operational |
| `Degraded` | 503 Service Unavailable | Partial degradation detected (e.g., high memory usage) |
| `Unhealthy` | 503 Service Unavailable | Critical issues detected (e.g., model loading failed) |

### Optional: treat `Degraded` as 200 OK

If you prefer to keep the load balancer routing traffic during partial issues, build the server with:

```bash
cargo build --no-default-features -p bitnet-server --features degraded-ok
```

With `degraded-ok` enabled:

| Status | HTTP Code | Description |
|--------|-----------|-------------|
| `Healthy` | 200 OK | All systems operational |
| `Degraded` | 200 OK | Partial degradation but still serving |
| `Unhealthy` | 503 Service Unavailable | Critical issues detected |

### Endpoint Semantics

* **`/health`** – Returns the overall health JSON and **HTTP status according to the mapping** (default fail-fast; with `--features degraded-ok`, Degraded → 200).
* **`/health/live`** – **Uses the same mapping** as `/health`.
* **`/health/ready`** – **Always fail-fast**: `Healthy` → 200; `Degraded|Unhealthy` → 503, regardless of `degraded-ok`.

All health endpoints set `Cache-Control: no-store` to prevent caching.

## Examples

```bash
# Check overall health
curl -i http://localhost:3000/health

# Check liveness (always 200 if server is up)
curl -i http://localhost:3000/health/live

# Check readiness (200 only when ready for inference)
curl -i http://localhost:3000/health/ready
```

## Response Format

### Basic Health Response
```json
{
  "status": "healthy",
  "checks": {
    "memory": "healthy",
    "model": "healthy",
    "gpu": "degraded"
  }
}
```

### Detailed Health Response (from `/health`)
```json
{
  "status": "healthy",
  "timestamp": "2025-08-26T10:00:00Z",
  "uptime_seconds": 3600,
  "version": "0.1.0",
  "build": {
    "version": "0.1.0",
    "git_sha": "unavailable",
    "git_branch": "unavailable",
    "build_timestamp": "2025-08-26T09:00:00Z",
    "rustc_version": "1.89.0",
    "cargo_target": "x86_64-unknown-linux-gnu",
    "cargo_profile": "release",
    "cuda_version": "12.0"  // Only present when built with CUDA support
  },
  "checks": {
    "memory": {
      "status": "healthy",
      "details": {
        "used_bytes": 1073741824,
        "total_bytes": 8589934592,
        "percent_used": 12.5
      }
    },
    "model": {
      "status": "healthy",
      "details": {
        "loaded": true,
        "name": "bitnet_b1_58-3B"
      }
    },
    "gpu": {
      "status": "degraded",
      "details": {
        "available": true,
        "memory_used": 2147483648,
        "memory_total": 8589934592
      }
    }
  }
}
```

**Note**: Git SHA and branch information are captured at build time using vergen-gix. If `.git` is not available during build, these fields will show "unknown". You can override by setting `VERGEN_GIT_SHA` and `VERGEN_GIT_BRANCH` environment variables during build.

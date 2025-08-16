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
cargo build -p bitnet-server --features degraded-ok
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
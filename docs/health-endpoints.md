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

| Status | HTTP Code | Description |
|--------|-----------|-------------|
| `Healthy` | 200 OK | All systems operational |
| `Degraded` | 503 Service Unavailable | Partial degradation detected (e.g., high memory usage) |
| `Unhealthy` | 503 Service Unavailable | Critical issues detected (e.g., model loading failed) |

**Note**: Both `Degraded` and `Unhealthy` return 503 to trigger fail-fast behavior in load balancers. If you need load balancers to continue routing traffic during partial degradation, consider mapping `Degraded` to 200 OK.

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
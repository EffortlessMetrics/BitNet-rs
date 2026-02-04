## 2025-05-18 - Overly Permissive CORS Default
**Vulnerability:** The server defaulted to allowing all origins (`*`) for CORS requests, which is unsafe for production environments as it exposes the API to CSRF-like attacks and data leakage.
**Learning:** `tower-http`'s `CorsLayer` is powerful but requires explicit configuration to be secure. The project used `Any` for convenience but sacrificed security.
**Prevention:** Always default to a restrictive CORS policy (empty or specific domain) and require explicit configuration for allowed origins. Use `SecurityConfig` to drive middleware configuration.

# Sentinel's Journal

## 2025-10-22 - Configurable CORS Middleware
**Vulnerability:** The server was configured with an overly permissive CORS policy (`AllowAll`) which could allow malicious websites to access the internal API if exposed.
**Learning:** The configuration struct `SecurityConfig` already had an `allowed_origins` field, but it was completely ignored by the implementation which hardcoded `AllowAll` using `tower_http::cors::Any`.
**Prevention:** When implementing security features, always cross-reference the configuration struct to ensure all security-related settings are actually used in the implementation.

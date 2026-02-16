## 2025-10-21 - Overly Permissive CORS Configuration
**Vulnerability:** The server was configured to allow all origins (`*`) by default using `tower_http::cors::CorsLayer::allow_origin(Any)`, ignoring the `allowed_origins` setting in `SecurityConfig`.
**Learning:** Middleware configuration functions must explicitly check and use application configuration objects rather than relying on library defaults or hardcoded values. In `tower-http` 0.6+, `CorsLayer` is not generic, and `allow_origin` accepts `Any` or `Vec<HeaderValue>`, simplifying conditional logic but requiring careful type handling.
**Prevention:** Always verify middleware configuration against the intended security policy in the application config. Use integration tests that assert on response headers to catch misconfigurations.

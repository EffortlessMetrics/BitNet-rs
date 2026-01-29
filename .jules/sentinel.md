## 2025-10-21 - [Overly Permissive CORS Default Ignoring Config]
**Vulnerability:** The `configure_cors` function hardcoded `AllowOrigin::any()`, completely ignoring the `SecurityConfig::allowed_origins` field which was correctly populated from environment variables.
**Learning:** Configuration structs might be populated but ignored in implementation. Always verify that security configuration is actually consumed by the relevant middleware/logic.
**Prevention:** Add integration tests that assert on the behavior of configured security features, not just unit tests for the configuration loading.

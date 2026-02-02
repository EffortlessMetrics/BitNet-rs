## 2025-05-22 - Unapplied Security Configuration
**Vulnerability:** The `configure_cors` function in `bitnet-server` was ignoring the `SecurityConfig` and defaulting to `AllowOrigin::any()`, effectively disabling CORS protection despite configuration options existing.
**Learning:** Middleware configuration functions must strictly accept and apply the security configuration object. Existence of a configuration struct does not imply its usage.
**Prevention:** Verify that all security-related configuration fields are actually referenced in the code, especially in middleware setup. Use unit tests to assert that configuration changes affect behavior.

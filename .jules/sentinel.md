## 2025-05-18 - Overly Permissive CORS Configuration

**Vulnerability:** The server was configured with `Allow-Origin: *` by default, ignoring user configuration.
**Learning:** The `configure_cors` function hardcoded `Any` origin instead of using `SecurityConfig`.
**Prevention:** Middleware configuration functions must take and use the configuration object. Unit tests now verify that `allowed_origins` setting is respected.

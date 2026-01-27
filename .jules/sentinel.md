## 2025-10-21 - Overly Permissive CORS Configuration
**Vulnerability:** The default CORS configuration was hardcoded to allow all origins (`*`), methods, and headers, even though `SecurityConfig` had an `allowed_origins` field.
**Learning:** Middleware configurations in `bitnet-server` were not fully utilizing the `SecurityConfig` struct, leading to security features being defined but not enforced.
**Prevention:** Ensure that all security-related middleware (CORS, Rate Limiting, etc.) are explicitly configured using the `SecurityConfig` passed from `BitNetServer`.

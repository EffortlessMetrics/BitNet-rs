# Sentinel Journal - Security Learnings

## 2024-05-24 - Overly Permissive CORS Configuration
**Vulnerability:** The `configure_cors` function in `bitnet-server` was hardcoded to allow any origin (`Any`) regardless of the configuration in `SecurityConfig`. This exposed the API to potential CSRF and data leakage risks from malicious sites.
**Learning:** Hardcoding security configurations during development (likely for ease of testing) and failing to connect them to the configuration system is a common pitfall. Additionally, updating to `tower-http` 0.6 requires understanding that `CorsLayer` is no longer generic, necessitating the use of `AllowOrigin::predicate` for dynamic runtime configuration based on settings.
**Prevention:** Always verify that security-related configuration fields (like `allowed_origins`) are actually utilized in the code. Use integration tests that specifically assert behavior for both allowed and blocked scenarios to catch configuration disconnects.

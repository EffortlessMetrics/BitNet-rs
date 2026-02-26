# Sentinel Journal - Security Learnings

## 2024-05-24 - Overly Permissive CORS Configuration
**Vulnerability:** The `configure_cors` function in `bitnet-server` was hardcoded to allow any origin (`Any`) regardless of the configuration in `SecurityConfig`. This exposed the API to potential CSRF and data leakage risks from malicious sites.
**Learning:** Hardcoding security configurations during development (likely for ease of testing) and failing to connect them to the configuration system is a common pitfall. Additionally, updating to `tower-http` 0.6 requires understanding that `CorsLayer` is no longer generic, necessitating the use of `AllowOrigin::predicate` for dynamic runtime configuration based on settings.
**Prevention:** Always verify that security-related configuration fields (like `allowed_origins`) are actually utilized in the code. Use integration tests that specifically assert behavior for both allowed and blocked scenarios to catch configuration disconnects.

## 2025-06-03 - Unrestricted Model Loading Path
**Vulnerability:** The server allowed loading model files from any path on the filesystem (e.g., via absolute paths) provided the file extension matched `.gguf` or `.safetensors`. This could allow attackers to probe for the existence of files or load sensitive data if it happened to have the correct extension.
**Learning:** Checking for file extensions and blocking `..` is insufficient for path security when absolute paths are allowed. Always restrict file operations to a specific root directory or allowlist.
**Prevention:** Implement a configuration option (`allowed_model_directories`) to restrict file loading to specific directories. Use `std::path::Path::starts_with` for robust path prefix checking, rather than string manipulation which can be bypassed (e.g., `/var/log` matching `/var/login`). Ensure existing path traversal protections are maintained.

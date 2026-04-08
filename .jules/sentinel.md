# Sentinel Journal - Security Learnings

## 2024-05-24 - Overly Permissive CORS Configuration
**Vulnerability:** The `configure_cors` function in `bitnet-server` was hardcoded to allow any origin (`Any`) regardless of the configuration in `SecurityConfig`. This exposed the API to potential CSRF and data leakage risks from malicious sites.
**Learning:** Hardcoding security configurations during development (likely for ease of testing) and failing to connect them to the configuration system is a common pitfall. Additionally, updating to `tower-http` 0.6 requires understanding that `CorsLayer` is no longer generic, necessitating the use of `AllowOrigin::predicate` for dynamic runtime configuration based on settings.
**Prevention:** Always verify that security-related configuration fields (like `allowed_origins`) are actually utilized in the code. Use integration tests that specifically assert behavior for both allowed and blocked scenarios to catch configuration disconnects.

## 2025-06-03 - Unrestricted Model Loading Path
**Vulnerability:** The server allowed loading model files from any path on the filesystem (e.g., via absolute paths) provided the file extension matched `.gguf` or `.safetensors`. This could allow attackers to probe for the existence of files or load sensitive data if it happened to have the correct extension.
**Learning:** Checking for file extensions and blocking `..` is insufficient for path security when absolute paths are allowed. Always restrict file operations to a specific root directory or allowlist.
**Prevention:** Implement a configuration option (`allowed_model_directories`) to restrict file loading to specific directories. Use `std::path::Path::starts_with` for robust path prefix checking, rather than string manipulation which can be bypassed (e.g., `/var/log` matching `/var/login`). Ensure existing path traversal protections are maintained.
## 2024-03-01 - [Input Validation Blocks Valid CRLF]
**Vulnerability:** Input validation in `sanitize_input` blocks carriage return (`\r`) characters, categorizing them as invalid control characters.
**Learning:** This restricts valid payloads coming from environments (like Windows) or protocols (like HTTP standard format) that use CRLF for newlines, leading to unintentional denial of service for these valid requests.
**Prevention:** Explicitly allow `\r` alongside `\n` and `\t` when filtering out control characters in text payloads.

## 2025-06-03 - [TOCTOU in RateLimitBucket leads to bypass via integer underflow]
**Vulnerability:** The `try_consume` method in `RateLimitBucket` was vulnerable to a Time-of-Check to Time-of-Use (TOCTOU) bug because it used a separate `load` and `fetch_sub` when verifying and decrementing available tokens. Concurrently running tasks could observe a positive number of tokens, pass the conditional check, and subtract tokens simultaneously, leading to integer underflow and a bypass of the rate limiter. Additionally, the `refill` method was subject to a data race that could overwrite consumed tokens with a stale calculation.
**Learning:** Separate read-then-write operations on atomics are inherently susceptible to race conditions under heavy concurrency.
**Prevention:** Use atomic `fetch_update` operations to guarantee atomic Read-Modify-Write functionality when an atomic value change is conditional on its current value.
## 2024-04-08 - Prevent Path Truncation Vulnerability in Model Validation
**Vulnerability:** The model path validation relied on `.ends_with(".gguf")` and `.ends_with(".safetensors")` which is vulnerable to path truncation attacks if the input contains a null byte (e.g., `model.txt\0.gguf`). Underlying C APIs would terminate at the null byte, potentially loading an unverified file.
**Learning:** Rust's string methods like `.ends_with` do not treat null bytes specially, creating a discrepancy with C/OS APIs that use null-terminated strings. Relying solely on these methods for file extension validation is insufficient when the path is eventually passed to underlying C/OS functions.
**Prevention:** Always explicitly reject strings containing null bytes (`\0`) early in the validation logic, especially when dealing with file paths that will interact with C libraries or the OS file system.

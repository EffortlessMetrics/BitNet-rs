## 2025-11-03 - Config-Code Disconnect in Security Modules
**Vulnerability:** CORS configuration in `SecurityConfig` was ignored in favor of hardcoded permissive settings (`Any`).
**Learning:** Security configurations defined in `config.rs` structures are not automatically applied to middleware; manual plumbing is required.
**Prevention:** Verify that every field in a security configuration struct is actually referenced in the implementation code.

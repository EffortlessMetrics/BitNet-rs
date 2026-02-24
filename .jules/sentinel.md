## 2025-11-03 - [Ignored Security Configuration]
**Vulnerability:** The `configure_cors` function hardcoded permissive settings (`Any`) and ignored the `SecurityConfig` passed to the application, rendering the configuration file's security settings ineffective.
**Learning:** Middleware configuration in `axum` services must explicitly use the loaded configuration object; default constructors in libraries often default to permissive development settings.
**Prevention:** Always verify that configuration objects are threaded through to the functions that initialize middleware. Add integration tests that verify configuration changes actually affect runtime behavior.

**Update (2025-11-03):** This fix was superseded by refactoring work (PRs #608–#638). The learning about secure defaults remains valid.

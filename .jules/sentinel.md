## 2025-10-21 - Ignored Security Configuration
**Vulnerability:** The `configure_cors` function in `bitnet-server` was hardcoded to allow all origins (`Any`), completely ignoring the `SecurityConfig.allowed_origins` setting.
**Learning:** Having a configuration struct does not guarantee the configuration is applied. Default implementations of middleware setup functions often carry "permissive by default" risks if not explicitly connected to the config.
**Prevention:** Ensure that all security-related configuration parameters are traced to their usage sites. Add unit tests that verify the configuration actually alters the behavior.

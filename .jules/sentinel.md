## 2024-11-15 - Unconnected Security Configuration
**Vulnerability:** CORS configuration `allowed_origins` was parsed from env/config but completely ignored in the `configure_cors` function, which defaulted to allowing all origins (`Any`).
**Learning:** Having configuration fields and loading logic doesn't guarantee they are used. Security features often fail silently if not explicitly wired up.
**Prevention:** Always verify that security configuration parameters are actually used in the implementation logic. Integration tests should verify the *effect* of configuration changes.

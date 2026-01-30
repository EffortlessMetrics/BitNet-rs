## 2024-11-04 - [High] Overly Permissive CORS Configuration
**Vulnerability:** The server was configured with `Access-Control-Allow-Origin: *` by default, ignoring the `allowed_origins` configuration field.
**Learning:** Hardcoded security defaults can easily override intended configuration if not explicitly connected.
**Prevention:** Always use configuration objects to drive security middleware setup rather than static defaults.

## 2025-07-01 - Prevent Chunked Transfer Encoding Bypass
**Vulnerability:** HTTP requests using chunked transfer encoding bypassed `content-length` body size limits in the request sanitization middleware because their length isn't predetermined.
**Learning:** Checking the `content-length` header is insufficient for preventing resource exhaustion if chunked encoding is still permitted.
**Prevention:** If an API relies on `content-length` for prompt size validation and does not implement a streaming global body parser limit, explicitly reject requests containing `transfer-encoding: chunked` with `411 Length Required`.

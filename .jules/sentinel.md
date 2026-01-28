# Sentinel Journal 🛡️

## 2025-10-24 - Tower HTTP CORS Configuration
**Vulnerability:** Overly permissive CORS configuration (wildcard `*` hardcoded) in `bitnet-server`.
**Learning:** `tower-http` version 0.6.6 significantly changed the `CorsLayer` API. It is no longer generic, and `AllowOrigin` is a struct, not a trait to be implemented. Dynamic configuration requires conditional logic to pass either `Any` or `Vec<HeaderValue>` to `allow_origin`, rather than returning different generic types.
**Prevention:** When upgrading or using `tower-http`, check documentation for `CorsLayer` generics. If dynamic origin configuration is needed, use the non-generic `allow_origin` method which accepts `impl Into<AllowOrigin>`.

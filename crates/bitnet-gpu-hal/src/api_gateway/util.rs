use bitnet_http_auth_core::strip_bearer_prefix;

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Generate a request id from a v4 UUID. Uses cryptographic randomness so the
/// id cannot be predicted from request timing.
pub(super) fn generate_request_id() -> String {
    let uuid = uuid::Uuid::new_v4().simple().to_string();
    format!("req-{uuid}")
}

/// Generate an API key from a v4 UUID. Time-based generation was previously
/// vulnerable to brute-force prediction by an attacker who knew the
/// approximate creation time.
pub(super) fn generate_api_key() -> String {
    let uuid = uuid::Uuid::new_v4().simple().to_string();
    format!("sk-bitnet-{uuid}")
}

/// Strip `"Bearer "` prefix from a header value.
pub(super) fn strip_bearer(val: &str) -> String {
    strip_bearer_prefix(val).to_string()
}

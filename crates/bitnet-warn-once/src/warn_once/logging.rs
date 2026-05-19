use super::registry::record_warning_occurrence;

pub fn warn_once_fn(key: &str, message: &str) {
    if record_warning_occurrence(key) {
        tracing::warn!(key = %key, "{}", message);
    } else {
        tracing::debug!(key = %key, "(rate-limited) {}", message);
    }
}

//! SRP core primitives for API-key authentication flows.
//!
//! This crate owns key-store policy and token authentication semantics while
//! remaining independent from HTTP framework types.

mod api_key;
mod auth_result;
mod key_store;

pub use api_key::ApiKey;
pub use auth_result::AuthResult;
pub use key_store::{AuthMode, KeyStore};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_mode_enforces_credentials() {
        let mut store = KeyStore::new(AuthMode::Required);
        store.add_key("test", "secret123");
        assert_eq!(store.authenticate(None), AuthResult::NoCredentials);
        assert!(store.authenticate(Some("secret123")).is_allowed());
    }

    #[test]
    fn optional_mode_allows_missing_but_rejects_invalid() {
        let mut store = KeyStore::new(AuthMode::Optional);
        store.add_key("test", "secret123");
        assert!(store.authenticate(None).is_allowed());
        assert_eq!(store.authenticate(Some("wrong")), AuthResult::Denied("invalid API key".into()));
    }

    #[test]
    fn key_enable_disable_roundtrip() {
        let mut store = KeyStore::new(AuthMode::Required);
        store.add_key("test", "secret123");
        assert!(store.disable_key("test"));
        assert!(!store.authenticate(Some("secret123")).is_allowed());
        assert!(store.enable_key("test"));
        assert!(store.authenticate(Some("secret123")).is_allowed());
    }

    #[test]
    fn usage_tracking_counts_successes() {
        let mut store = KeyStore::new(AuthMode::Required);
        store.add_key("test", "key1");
        let _ = store.authenticate(Some("key1"));
        let _ = store.authenticate(Some("key1"));
        let stats = store.usage_stats();
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].1, 2);
    }

    #[test]
    fn bearer_extraction_uses_shared_http_auth_core() {
        assert_eq!(KeyStore::extract_bearer("Bearer abc123"), Some("abc123"));
        assert_eq!(KeyStore::extract_bearer("Token abc123"), None);
    }

    #[test]
    fn disabled_mode_allows_without_keys() {
        let mut store = KeyStore::disabled();
        assert_eq!(store.mode(), AuthMode::Disabled);
        assert_eq!(store.authenticate(None), AuthResult::Allowed);
        assert_eq!(store.authenticate(Some("anything")), AuthResult::Allowed);
    }

    #[test]
    fn add_key_replaces_existing_entry_with_same_name() {
        let mut store = KeyStore::new(AuthMode::Required);
        store.add_key("api", "first");
        store.add_key("api", "second");
        assert_eq!(store.key_count(), 1);
        assert!(!store.authenticate(Some("first")).is_allowed());
        assert!(store.authenticate(Some("second")).is_allowed());
    }

    #[test]
    fn remove_key_reports_presence() {
        let mut store = KeyStore::new(AuthMode::Required);
        store.add_key("x", "k");
        assert!(store.remove_key("x"));
        assert!(!store.remove_key("x"));
        assert_eq!(store.key_count(), 0);
    }

    #[test]
    fn disable_and_enable_missing_key_return_false() {
        let mut store = KeyStore::new(AuthMode::Required);
        assert!(!store.disable_key("ghost"));
        assert!(!store.enable_key("ghost"));
    }

    #[test]
    fn key_counts_track_enabled_state() {
        let mut store = KeyStore::new(AuthMode::Required);
        store.add_key("a", "ka");
        store.add_key("b", "kb");
        store.add_key("c", "kc");
        assert_eq!(store.key_count(), 3);
        assert_eq!(store.active_key_count(), 3);

        assert!(store.disable_key("b"));
        assert_eq!(store.key_count(), 3);
        assert_eq!(store.active_key_count(), 2);
    }

    #[test]
    fn usage_count_does_not_increase_on_denial() {
        let mut store = KeyStore::new(AuthMode::Required);
        store.add_key("k", "right");
        let _ = store.authenticate(Some("wrong"));
        let _ = store.authenticate(Some("wrong"));
        assert_eq!(store.usage_stats()[0].1, 0);
    }

    #[test]
    fn required_mode_denies_when_no_keys_registered() {
        let mut store = KeyStore::new(AuthMode::Required);
        assert_eq!(
            store.authenticate(Some("anything")),
            AuthResult::Denied("invalid API key".into())
        );
        assert_eq!(store.authenticate(None), AuthResult::NoCredentials);
    }

    #[test]
    fn auth_result_is_allowed_only_for_allowed_variant() {
        assert!(AuthResult::Allowed.is_allowed());
        assert!(!AuthResult::NoCredentials.is_allowed());
        assert!(!AuthResult::Denied("nope".into()).is_allowed());
    }

    #[test]
    fn api_key_matches_only_when_enabled() {
        let mut key = ApiKey::new("name", "secret");
        assert!(key.matches("secret"));
        assert!(!key.matches("other"));

        key.enabled = false;
        assert!(!key.matches("secret"), "disabled key must not match");
    }

    #[test]
    fn mode_accessor_returns_constructor_value() {
        assert_eq!(KeyStore::new(AuthMode::Disabled).mode(), AuthMode::Disabled);
        assert_eq!(KeyStore::new(AuthMode::Required).mode(), AuthMode::Required);
        assert_eq!(KeyStore::new(AuthMode::Optional).mode(), AuthMode::Optional);
    }
}

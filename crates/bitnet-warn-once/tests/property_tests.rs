use bitnet_warn_once::warn_once_fn;
use proptest::prelude::*;

// ── warn_once! macro — key-based deduplication ───────────────────────────────

proptest! {
    /// warn_once_fn never panics for arbitrary valid string keys and messages.
    /// Calling twice with the same key must also be safe (second call is rate-limited).
    #[test]
    fn prop_warn_once_fn_no_panic(
        key in "[a-z][a-z0-9_]{0,31}",
        msg in "[a-zA-Z0-9 ]{1,64}",
    ) {
        warn_once_fn(&key, &msg);
        // Second call with same key: rate-limited to DEBUG, must not panic.
        warn_once_fn(&key, &msg);
    }
}

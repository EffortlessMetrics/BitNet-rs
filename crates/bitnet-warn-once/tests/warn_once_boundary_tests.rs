//! Boundary and stress tests for bitnet-warn-once.
//!
//! These integration tests exercise the public API (`warn_once_fn`, `warn_once!`)
//! without access to `clear_registry_for_test`. All keys are unique per test to
//! avoid cross-test pollution in the global registry.

use bitnet_warn_once::{warn_once, warn_once_fn};
use std::sync::{Arc, Barrier};
use std::thread;

// ---------------------------------------------------------------------------
// Empty and whitespace keys
// ---------------------------------------------------------------------------

#[test]
fn empty_key_does_not_panic() {
    warn_once_fn("", "empty key message");
    warn_once_fn("", "empty key again (rate-limited)");
}

#[test]
fn whitespace_only_key_does_not_panic() {
    warn_once_fn("   ", "whitespace key");
    warn_once_fn("   ", "whitespace key again");
}

#[test]
fn newline_in_key() {
    warn_once_fn("key\nwith\nnewlines", "newline key");
}

#[test]
fn tab_in_key() {
    warn_once_fn("key\twith\ttabs", "tab key");
}

#[test]
fn null_byte_in_key() {
    warn_once_fn("key\0null", "null byte key");
}

// ---------------------------------------------------------------------------
// Unicode keys
// ---------------------------------------------------------------------------

#[test]
fn unicode_key_emoji() {
    warn_once_fn("🚀_rocket_key", "emoji key message");
    warn_once_fn("🚀_rocket_key", "emoji key again");
}

#[test]
fn unicode_key_cjk() {
    warn_once_fn("模型加载警告", "CJK key message");
}

#[test]
fn unicode_key_rtl() {
    warn_once_fn("مفتاح_عربي", "RTL key message");
}

#[test]
fn unicode_key_mixed_scripts() {
    warn_once_fn("αβγ_abc_日本語", "mixed script key");
}

// ---------------------------------------------------------------------------
// Very long keys and messages
// ---------------------------------------------------------------------------

#[test]
fn very_long_key_1000_chars() {
    let key: String = (0..1000).map(|_| 'x').collect();
    warn_once_fn(&key, "long key message");
    warn_once_fn(&key, "long key repeated");
}

#[test]
fn very_long_message_10000_chars() {
    let msg: String = (0..10000).map(|_| 'M').collect();
    warn_once_fn("boundary_long_msg", &msg);
}

#[test]
fn key_with_special_chars() {
    warn_once_fn("key/with/slashes", "slashes");
    warn_once_fn("key\\with\\backslashes", "backslashes");
    warn_once_fn("key:with:colons", "colons");
    warn_once_fn("key=with=equals", "equals");
    warn_once_fn("key&with&ampersands", "ampersands");
}

// ---------------------------------------------------------------------------
// Macro with various format patterns
// ---------------------------------------------------------------------------

#[test]
fn macro_no_format_args() {
    warn_once!("boundary_macro_plain", "plain message no args");
}

#[test]
fn macro_single_arg() {
    let val = 42;
    warn_once!("boundary_macro_single", "value: {}", val);
}

#[test]
fn macro_multiple_args() {
    warn_once!("boundary_macro_multi", "a={}, b={}, c={}", 1, "two", 3.0);
}

#[test]
fn macro_debug_format() {
    let v = vec![1, 2, 3];
    warn_once!("boundary_macro_debug", "vec: {:?}", v);
}

#[test]
fn macro_named_args() {
    let name = "test";
    let count = 5;
    warn_once!("boundary_macro_named", "name={name}, count={count}");
}

#[test]
fn macro_padding_and_precision() {
    warn_once!("boundary_macro_pad", "pi={:.5}, padded={:>10}", 3.14159, "hi");
}

#[test]
fn macro_empty_format() {
    warn_once!("boundary_macro_empty_fmt", "");
}

// ---------------------------------------------------------------------------
// High-volume single-key stress
// ---------------------------------------------------------------------------

#[test]
fn stress_same_key_10000_calls() {
    for _ in 0..10_000 {
        warn_once_fn("boundary_stress_same", "stress message");
    }
}

#[test]
fn stress_many_unique_keys_1000() {
    for i in 0..1_000 {
        warn_once_fn(&format!("boundary_stress_unique_{}", i), "unique key stress");
    }
}

// ---------------------------------------------------------------------------
// Concurrent access patterns
// ---------------------------------------------------------------------------

#[test]
fn concurrent_same_key_100_threads() {
    let barrier = Arc::new(Barrier::new(100));
    let handles: Vec<_> = (0..100)
        .map(|i| {
            let b = barrier.clone();
            thread::spawn(move || {
                b.wait();
                warn_once_fn("boundary_concurrent_100", &format!("thread {} message", i));
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn concurrent_distinct_keys_per_thread() {
    let barrier = Arc::new(Barrier::new(20));
    let handles: Vec<_> = (0..20)
        .map(|i| {
            let b = barrier.clone();
            thread::spawn(move || {
                b.wait();
                for j in 0..50 {
                    warn_once_fn(&format!("boundary_thread_{}_key_{}", i, j), "per-thread key");
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn concurrent_mixed_same_and_distinct_keys() {
    let barrier = Arc::new(Barrier::new(10));
    let handles: Vec<_> = (0..10)
        .map(|i| {
            let b = barrier.clone();
            thread::spawn(move || {
                b.wait();
                // All threads hit the same key
                warn_once_fn("boundary_mixed_shared", "shared key");
                // Each thread also has a unique key
                warn_once_fn(&format!("boundary_mixed_unique_{}", i), "unique key");
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}

// ---------------------------------------------------------------------------
// Interaction: macro vs function with same key
// ---------------------------------------------------------------------------

#[test]
fn macro_and_fn_share_same_key_space() {
    // If key is the same, the second call should be rate-limited regardless
    // of whether it uses warn_once! or warn_once_fn
    warn_once_fn("boundary_macro_fn_shared", "from fn");
    warn_once!("boundary_macro_fn_shared", "from macro");
}

// ---------------------------------------------------------------------------
// Keys that look like other formats
// ---------------------------------------------------------------------------

#[test]
fn json_like_key() {
    warn_once_fn(r#"{"type":"deprecated"}"#, "json key");
}

#[test]
fn url_like_key() {
    warn_once_fn("https://example.com/warn?id=123", "url key");
}

#[test]
fn path_like_key() {
    warn_once_fn("crates/bitnet-kernels/src/cpu.rs:42", "path key");
}

#[test]
fn numeric_string_key() {
    warn_once_fn("12345", "numeric key");
    warn_once_fn("0", "zero key");
    warn_once_fn("-1", "negative key");
}

// ---------------------------------------------------------------------------
// Sequential vs interleaved key patterns
// ---------------------------------------------------------------------------

#[test]
fn interleaved_keys() {
    for _ in 0..100 {
        warn_once_fn("boundary_interleave_a", "key A");
        warn_once_fn("boundary_interleave_b", "key B");
        warn_once_fn("boundary_interleave_c", "key C");
    }
}

#[test]
fn similar_prefix_keys_are_distinct() {
    warn_once_fn("boundary_prefix", "prefix");
    warn_once_fn("boundary_prefix_extended", "prefix extended");
    warn_once_fn("boundary_prefix_extended_more", "prefix extended more");
    // All three are distinct keys and should all warn
}

#[test]
fn case_sensitive_keys() {
    warn_once_fn("boundary_CASE", "upper");
    warn_once_fn("boundary_case", "lower");
    warn_once_fn("boundary_Case", "mixed");
    // All three are distinct
}

// ---------------------------------------------------------------------------
// Edge: empty message
// ---------------------------------------------------------------------------

#[test]
fn empty_message_does_not_panic() {
    warn_once_fn("boundary_empty_msg", "");
}

#[test]
fn message_with_format_specifiers_literal() {
    // Not using the macro - passing literal format-like strings
    warn_once_fn("boundary_fmt_literal", "value: {}, other: {:?}");
}

//! Edge-case tests for bitnet-warn-once public API.
//!
//! Note: `clear_registry_for_test` is only available in unit tests (#[cfg(test)]),
//! so integration tests here cannot reset the global registry. We test for
//! correctness of the public API (no panics, thread safety).

use bitnet_warn_once::{warn_once, warn_once_fn};

// ---------------------------------------------------------------------------
// warn_once_fn: basic calls
// ---------------------------------------------------------------------------

#[test]
fn warn_once_fn_does_not_panic() {
    warn_once_fn("edge_basic", "basic message");
}

#[test]
fn warn_once_fn_repeated_same_key() {
    for _ in 0..100 {
        warn_once_fn("edge_repeated", "same message");
    }
}

#[test]
fn warn_once_fn_many_unique_keys() {
    for i in 0..50 {
        warn_once_fn(&format!("edge_unique_{}", i), "unique key message");
    }
}

#[test]
fn warn_once_fn_empty_key() {
    warn_once_fn("", "empty key message");
}

#[test]
fn warn_once_fn_empty_message() {
    warn_once_fn("edge_empty_msg", "");
}

#[test]
fn warn_once_fn_unicode_key() {
    warn_once_fn("edge_日本語", "unicode key test");
}

#[test]
fn warn_once_fn_unicode_message() {
    warn_once_fn("edge_unicode_msg", "日本語メッセージ 🦀");
}

#[test]
fn warn_once_fn_long_key() {
    let long_key = "x".repeat(1000);
    warn_once_fn(&long_key, "long key message");
}

#[test]
fn warn_once_fn_long_message() {
    let long_msg = "y".repeat(10_000);
    warn_once_fn("edge_long_msg", &long_msg);
}

// ---------------------------------------------------------------------------
// warn_once! macro
// ---------------------------------------------------------------------------

#[test]
fn warn_once_macro_simple() {
    warn_once!("edge_macro_simple", "macro simple message");
}

#[test]
fn warn_once_macro_formatted() {
    let value = 42;
    warn_once!("edge_macro_fmt", "formatted: {}", value);
}

#[test]
fn warn_once_macro_multiple_args() {
    warn_once!("edge_macro_multi", "a={} b={} c={}", 1, "two", 3.0);
}

#[test]
fn warn_once_macro_repeated() {
    for _ in 0..10 {
        warn_once!("edge_macro_repeated", "repeated macro");
    }
}

// ---------------------------------------------------------------------------
// Thread safety
// ---------------------------------------------------------------------------

#[test]
fn warn_once_fn_concurrent_same_key() {
    use std::sync::Arc;
    use std::thread;

    let barrier = Arc::new(std::sync::Barrier::new(8));
    let mut handles = vec![];

    for i in 0..8 {
        let b = barrier.clone();
        handles.push(thread::spawn(move || {
            b.wait();
            warn_once_fn("edge_concurrent_same", &format!("thread {}", i));
        }));
    }

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn warn_once_fn_concurrent_different_keys() {
    use std::sync::Arc;
    use std::thread;

    let barrier = Arc::new(std::sync::Barrier::new(8));
    let mut handles = vec![];

    for i in 0..8 {
        let b = barrier.clone();
        handles.push(thread::spawn(move || {
            b.wait();
            warn_once_fn(&format!("edge_conc_diff_{}", i), "different keys");
        }));
    }

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn warn_once_macro_concurrent() {
    use std::sync::Arc;
    use std::thread;

    let barrier = Arc::new(std::sync::Barrier::new(4));
    let mut handles = vec![];

    for i in 0..4 {
        let b = barrier.clone();
        handles.push(thread::spawn(move || {
            b.wait();
            warn_once!("edge_macro_conc", "thread {} via macro", i);
        }));
    }

    for h in handles {
        h.join().unwrap();
    }
}

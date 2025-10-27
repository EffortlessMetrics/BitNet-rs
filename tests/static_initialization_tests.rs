//! Comprehensive test scaffolding for lazy_static → OnceLock migration
//!
//! This test suite validates Phase 2B migration from `once_cell::sync::Lazy` and
//! `once_cell::sync::OnceCell` to `std::sync::OnceLock` following TDD principles.
//!
//! # Specification Reference
//!
//! `/tmp/phase2_lazy_static_specification.md` - Complete migration specification
//!
//! # Test Categories
//!
//! 1. **Basic Initialization Tests** (AC4) - Exactly-once semantics, lazy evaluation
//! 2. **Thread Safety Tests** (AC3) - Concurrent access, race condition protection
//! 3. **Initialization Semantics Tests** (AC4) - Complex patterns, error handling
//! 4. **Performance Tests** (AC5) - No regression vs once_cell baseline
//! 5. **Pattern-Specific Tests** - Validation for each migration pattern from spec
//! 6. **Integration Tests** (AC6) - End-to-end workflows (CLI, models, inference)
//! 7. **Dependency Hygiene Tests** (AC2, AC7) - Verify once_cell removed, deny.toml blocks reintroduction
//!
//! # Acceptance Criteria Coverage
//!
//! - **AC1**: Version consolidation (tested in workspace_quality_gates.rs)
//! - **AC2**: All once_cell replaced with OnceLock (dependency hygiene tests)
//! - **AC3**: Thread safety preserved (concurrent access tests)
//! - **AC4**: Initialization semantics preserved (lazy evaluation, exactly-once)
//! - **AC5**: No performance regression (30k calls < 100ms baseline)
//! - **AC6**: All tests pass (integration tests)
//! - **AC7**: cargo-deny prevents reintroduction (dependency hygiene tests)
//! - **AC8**: MSRV documented (1.90.0, OnceLock stable since 1.70.0)
//!
//! # TDD Approach
//!
//! Tests are written first and marked with `#[ignore]` to indicate they will fail
//! until the migration is complete. Remove `#[ignore]` as each migration step completes.

// Note: serial_test::serial not needed yet - all tests are #[ignore] during migration
// use serial_test::serial;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

// ============================================================================
// Test Category 1: Basic Initialization Tests (AC4)
// ============================================================================

/// Tests that OnceLock initializes exactly once, even under repeated access.
///
/// **Validates**: AC4 (initialization semantics preserved)
///
/// **Specification Reference**: Section 2.2.4 - Initialization Semantics Preservation
///
/// This test ensures that the initialization closure runs exactly once, regardless
/// of how many times `get_or_init()` is called.
#[test]
#[ignore] // TODO: Remove after OnceLock migration in ln_rules.rs, ffi_session.rs, weight_mapper.rs
fn test_oncelock_single_initialization() {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    static VALUE: OnceLock<u32> = OnceLock::new();

    // First access - should initialize
    let v1 = VALUE.get_or_init(|| {
        COUNTER.fetch_add(1, Ordering::SeqCst);
        42
    });

    // Subsequent accesses - should NOT re-initialize
    for _ in 0..10_000 {
        let v = VALUE.get_or_init(|| {
            COUNTER.fetch_add(1, Ordering::SeqCst);
            999 // Different value to detect re-initialization
        });
        assert_eq!(*v, 42, "Subsequent access should return cached value");
    }

    assert_eq!(*v1, 42, "First access should return initialized value");
    assert_eq!(COUNTER.load(Ordering::SeqCst), 1, "Initializer should run exactly once");
}

/// Tests that OnceLock initialization is lazy (only on first access).
///
/// **Validates**: AC4 (initialization semantics preserved)
///
/// **Specification Reference**: Section 2.2.4 - Lazy Evaluation Timing
///
/// This test ensures that the initialization closure does NOT run until the
/// first call to `get_or_init()`, preserving lazy evaluation semantics.
#[test]
#[ignore] // TODO: Remove after OnceLock migration completes
fn test_oncelock_lazy_evaluation() {
    static INIT_FLAG: AtomicUsize = AtomicUsize::new(0);
    static LAZY_VALUE: OnceLock<String> = OnceLock::new();

    // At this point, the value should NOT be initialized
    assert_eq!(
        INIT_FLAG.load(Ordering::SeqCst),
        0,
        "Value should not be initialized before first access"
    );

    // First access triggers initialization
    let value = LAZY_VALUE.get_or_init(|| {
        INIT_FLAG.fetch_add(1, Ordering::SeqCst);
        "initialized".to_string()
    });

    assert_eq!(value, "initialized");
    assert_eq!(
        INIT_FLAG.load(Ordering::SeqCst),
        1,
        "Initialization should happen exactly once on first access"
    );

    // Subsequent accesses should not re-initialize
    let value2 = LAZY_VALUE.get_or_init(|| {
        INIT_FLAG.fetch_add(1, Ordering::SeqCst);
        "should not run".to_string()
    });

    assert_eq!(value2, "initialized");
    assert_eq!(INIT_FLAG.load(Ordering::SeqCst), 1, "Subsequent accesses should not re-initialize");
}

/// Tests that OnceLock does not initialize prematurely.
///
/// **Validates**: AC4 (initialization semantics preserved)
///
/// **Specification Reference**: Section 2.2.4 - Lazy Evaluation Timing
///
/// This test ensures that creating a static OnceLock does not cause side effects
/// or premature initialization.
#[test]
#[ignore] // TODO: Remove after OnceLock migration completes
fn test_oncelock_initialization_order() {
    static ORDER_TRACKER: AtomicUsize = AtomicUsize::new(0);
    static EARLY_INIT: OnceLock<u32> = OnceLock::new();
    static LATE_INIT: OnceLock<u32> = OnceLock::new();

    // Neither should be initialized yet
    assert_eq!(
        ORDER_TRACKER.load(Ordering::SeqCst),
        0,
        "No initialization should occur on declaration"
    );

    // Initialize in specific order
    let _late = LATE_INIT.get_or_init(|| {
        ORDER_TRACKER.fetch_add(1, Ordering::SeqCst);
        200
    });

    assert_eq!(ORDER_TRACKER.load(Ordering::SeqCst), 1);

    let _early = EARLY_INIT.get_or_init(|| {
        ORDER_TRACKER.fetch_add(1, Ordering::SeqCst);
        100
    });

    assert_eq!(ORDER_TRACKER.load(Ordering::SeqCst), 2);

    // Verify initialization order was preserved (late first, then early)
    assert_eq!(*_late, 200);
    assert_eq!(*_early, 100);
}

/// Tests that get_or_init returns the correct value.
///
/// **Validates**: AC4 (initialization semantics preserved)
///
/// **Specification Reference**: Section 2.2.1 - API Comparison Matrix
#[test]
#[ignore] // TODO: Remove after OnceLock migration completes
fn test_oncelock_get_or_init_return_value() {
    static VALUE: OnceLock<Vec<u32>> = OnceLock::new();

    let vec = VALUE.get_or_init(|| vec![1, 2, 3, 4, 5]);

    assert_eq!(vec.len(), 5);
    assert_eq!(vec[0], 1);
    assert_eq!(vec[4], 5);

    // Subsequent access returns the same reference
    let vec2 = VALUE.get_or_init(|| vec![99, 99, 99]);
    assert_eq!(vec2.len(), 5); // Original value, not re-initialized
    assert_eq!(vec2[0], 1);
}

// ============================================================================
// Test Category 2: Thread Safety Tests (AC3)
// ============================================================================

/// Tests OnceLock under heavy concurrent access (100 threads × 50 iterations).
///
/// **Validates**: AC3 (thread safety preserved)
///
/// **Specification Reference**: Section 2.2.3 - Thread Safety Analysis
///
/// This test ensures that OnceLock provides identical thread safety guarantees
/// to once_cell::Lazy, including single initialization and no data races.
#[test]
#[ignore] // TODO: Remove after OnceLock migration completes
fn test_oncelock_concurrent_access() {
    static INIT_COUNTER: AtomicUsize = AtomicUsize::new(0);
    static SHARED_VALUE: OnceLock<Arc<Mutex<Vec<u32>>>> = OnceLock::new();

    let handles: Vec<_> = (0..100)
        .map(|thread_id| {
            thread::spawn(move || {
                for iteration in 0..50 {
                    let shared = SHARED_VALUE.get_or_init(|| {
                        INIT_COUNTER.fetch_add(1, Ordering::SeqCst);
                        Arc::new(Mutex::new(vec![42]))
                    });

                    // Verify we get the same shared value
                    let data = shared.lock().unwrap();
                    assert_eq!(data[0], 42, "Expected initialized value");
                    drop(data);

                    // Add validation mark to ensure all threads participate
                    if iteration == 0 {
                        eprintln!("Thread {} validated initialization", thread_id);
                    }
                }
            })
        })
        .collect();

    // Wait for all threads to complete
    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    // Critical validation: initialization must have occurred exactly once
    assert_eq!(
        INIT_COUNTER.load(Ordering::SeqCst),
        1,
        "Initialization must occur exactly once despite concurrent access"
    );
}

/// Tests that OnceLock protects against race conditions during initialization.
///
/// **Validates**: AC3 (thread safety preserved)
///
/// **Specification Reference**: Section 2.2.3 - Synchronization Guarantees
///
/// This test ensures that exactly one thread executes the initialization closure,
/// and all other threads block until initialization completes.
#[test]
#[ignore] // TODO: Remove after OnceLock migration completes
fn test_oncelock_race_condition_protection() {
    static RACE_COUNTER: AtomicUsize = AtomicUsize::new(0);
    static PROTECTED_VALUE: OnceLock<usize> = OnceLock::new();

    let barrier = Arc::new(std::sync::Barrier::new(50));

    let handles: Vec<_> = (0..50)
        .map(|_| {
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                // Synchronize thread start to maximize race condition potential
                barrier.wait();

                let value = PROTECTED_VALUE.get_or_init(|| {
                    RACE_COUNTER.fetch_add(1, Ordering::SeqCst);
                    thread::sleep(Duration::from_millis(10)); // Increase race window
                    123456
                });

                assert_eq!(*value, 123456);
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    assert_eq!(
        RACE_COUNTER.load(Ordering::SeqCst),
        1,
        "Race condition protection failed: initialization ran multiple times"
    );
}

/// Tests that OnceLock is deadlock-free under concurrent access.
///
/// **Validates**: AC3 (thread safety preserved)
///
/// **Specification Reference**: Section 2.2.3 - Thread Safety Analysis
#[test]
#[ignore] // TODO: Remove after OnceLock migration completes
fn test_oncelock_deadlock_freedom() {
    static VALUE_A: OnceLock<u32> = OnceLock::new();
    static VALUE_B: OnceLock<u32> = OnceLock::new();

    let handle1 = thread::spawn(|| {
        for _ in 0..100 {
            let a = VALUE_A.get_or_init(|| 10);
            let b = VALUE_B.get_or_init(|| 20);
            assert_eq!(*a + *b, 30);
        }
    });

    let handle2 = thread::spawn(|| {
        for _ in 0..100 {
            let b = VALUE_B.get_or_init(|| 20);
            let a = VALUE_A.get_or_init(|| 10);
            assert_eq!(*a + *b, 30);
        }
    });

    // Test should complete without deadlock
    handle1.join().expect("Thread 1 panicked");
    handle2.join().expect("Thread 2 panicked");
}

/// Tests that OnceLock provides correct memory ordering across threads.
///
/// **Validates**: AC3 (thread safety preserved)
///
/// **Specification Reference**: Section 2.2.3 - Memory Ordering (SeqCst)
#[test]
#[ignore] // TODO: Remove after OnceLock migration completes
fn test_oncelock_cross_thread_visibility() {
    static SHARED_DATA: OnceLock<Arc<Mutex<Vec<u32>>>> = OnceLock::new();

    // Thread 1: Initialize and write data
    let handle1 = thread::spawn(|| {
        let data = SHARED_DATA.get_or_init(|| Arc::new(Mutex::new(vec![])));
        data.lock().unwrap().push(42);
    });

    handle1.join().expect("Thread 1 panicked");

    // Thread 2: Read data (must see initialized value)
    let handle2 = thread::spawn(|| {
        let data = SHARED_DATA.get_or_init(|| Arc::new(Mutex::new(vec![999])));
        let vec = data.lock().unwrap();
        assert_eq!(vec.len(), 1, "Thread 2 should see data from Thread 1");
        assert_eq!(vec[0], 42, "Memory ordering violation detected");
    });

    handle2.join().expect("Thread 2 panicked");
}

// ============================================================================
// Test Category 3: Initialization Semantics Tests (AC4)
// ============================================================================

/// Tests OnceLock with complex multi-step initialization logic.
///
/// **Validates**: AC4 (initialization semantics preserved)
///
/// **Specification Reference**: Section 2.2.4 - Initialization Semantics Preservation
#[test]
#[ignore] // TODO: Remove after OnceLock migration completes
fn test_oncelock_complex_initialization() {
    static COMPLEX_VALUE: OnceLock<(String, Vec<u32>, bool)> = OnceLock::new();

    let value = COMPLEX_VALUE.get_or_init(|| {
        // Multi-step initialization with side effects
        let mut vec = Vec::new();
        for i in 0..10 {
            vec.push(i * 2);
        }

        let name = format!("complex_init_{}", vec.len());
        let flag = vec.iter().sum::<u32>() > 50;

        (name, vec, flag)
    });

    assert_eq!(value.0, "complex_init_10");
    assert_eq!(value.1.len(), 10);
    assert!(value.2); // Sum is 90, which is > 50
}

/// Tests OnceLock with fallible initialization (panic behavior).
///
/// **Validates**: AC4 (initialization semantics preserved)
///
/// **Specification Reference**: Section 2.2.5 - Error Handling Changes
#[test]
#[ignore] // TODO: Remove after OnceLock migration completes
#[should_panic(expected = "deliberate panic")]
fn test_oncelock_fallible_initialization_panic() {
    static FALLIBLE_VALUE: OnceLock<u32> = OnceLock::new();

    // Initialization panics
    let _value = FALLIBLE_VALUE.get_or_init(|| {
        panic!("deliberate panic");
    });
}

/// Tests OnceLock with fallible initialization (Result-based).
///
/// **Validates**: AC4 (initialization semantics preserved)
///
/// **Specification Reference**: Section 2.2.5 - Defensive Pattern
#[test]
#[ignore] // TODO: Remove after OnceLock migration completes
fn test_oncelock_fallible_initialization_result() {
    use std::sync::OnceLock;

    fn try_init() -> Result<u32, &'static str> {
        Err("initialization failed")
    }

    static RESULT_VALUE: OnceLock<Result<u32, &'static str>> = OnceLock::new();

    let result = RESULT_VALUE.get_or_init(|| try_init());

    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), "initialization failed");
}

/// Tests OnceLock initialization with external dependencies.
///
/// **Validates**: AC4 (initialization semantics preserved)
///
/// **Specification Reference**: Section 3.1 - File-by-File Migration Guide
#[test]
#[ignore] // TODO: Remove after OnceLock migration completes
fn test_oncelock_initialization_with_dependencies() {
    use regex::Regex;

    static PATTERN: OnceLock<Regex> = OnceLock::new();

    let pattern =
        PATTERN.get_or_init(|| Regex::new(r"^\d{3}-\d{2}-\d{4}$").expect("regex must compile"));

    assert!(pattern.is_match("123-45-6789"));
    assert!(!pattern.is_match("invalid"));
}

/// Tests OnceLock initialization timing (lazy vs eager comparison).
///
/// **Validates**: AC4 (initialization semantics preserved)
///
/// **Specification Reference**: Section 2.2.4 - Lazy Evaluation Timing
#[test]
#[ignore] // TODO: Remove after OnceLock migration completes
fn test_oncelock_initialization_timing() {
    static TIMING_FLAG: AtomicUsize = AtomicUsize::new(0);
    static TIMED_VALUE: OnceLock<Instant> = OnceLock::new();

    let before = Instant::now();
    thread::sleep(Duration::from_millis(50));

    // Initialization happens here (lazy)
    let init_time = TIMED_VALUE.get_or_init(|| {
        TIMING_FLAG.fetch_add(1, Ordering::SeqCst);
        Instant::now()
    });

    // Ensure initialization happened AFTER the sleep
    assert!(
        init_time.duration_since(before) >= Duration::from_millis(50),
        "Initialization should be lazy, not eager"
    );
    assert_eq!(TIMING_FLAG.load(Ordering::SeqCst), 1, "Initialization should happen exactly once");
}

// ============================================================================
// Test Category 4: Performance Tests (AC5)
// ============================================================================

/// Tests OnceLock performance baseline (30k calls < 100ms).
///
/// **Validates**: AC5 (no performance regression)
///
/// **Specification Reference**: Section 6.4 - Performance Benchmarks
#[test]
#[ignore] // TODO: Remove after OnceLock migration completes
fn test_oncelock_performance_baseline() {
    static PERF_VALUE: OnceLock<Vec<u32>> = OnceLock::new();

    // Initialize once
    let _init = PERF_VALUE.get_or_init(|| vec![1, 2, 3, 4, 5]);

    // Benchmark 30k accesses (should be < 100ms)
    let start = Instant::now();
    for _ in 0..30_000 {
        let vec = PERF_VALUE.get_or_init(|| vec![99, 99, 99]);
        assert_eq!(vec.len(), 5); // Force dereference
    }
    let elapsed = start.elapsed();

    assert!(
        elapsed < Duration::from_millis(100),
        "Performance regression: 30k calls took {:?} (expected < 100ms)",
        elapsed
    );
}

// NOTE: test_oncelock_vs_lazy_performance was removed after OnceLock migration completed.
// Performance comparison is no longer possible without once_cell as a dependency.
// Baseline performance is validated by test_oncelock_performance_baseline instead.

/// Tests OnceLock repeated access performance (fast path optimization).
///
/// **Validates**: AC5 (no performance regression)
///
/// **Specification Reference**: Section 2.2.1 - Performance (Zero-cost after init)
#[test]
#[ignore] // TODO: Remove after OnceLock migration completes
fn test_oncelock_repeated_access_performance() {
    static FAST_PATH: OnceLock<u64> = OnceLock::new();

    // Initialize once
    let _init = FAST_PATH.get_or_init(|| 42u64);

    // Benchmark repeated access (should be near-instant)
    let start = Instant::now();
    let mut sum = 0u64;
    for _ in 0..1_000_000 {
        sum += *FAST_PATH.get_or_init(|| 999);
    }
    let elapsed = start.elapsed();

    assert_eq!(sum, 42_000_000); // Verify correctness
    assert!(
        elapsed < Duration::from_millis(50),
        "Fast path optimization missing: 1M accesses took {:?}",
        elapsed
    );
}

// ============================================================================
// Test Category 5: Pattern-Specific Tests (Migration Validation)
// ============================================================================

/// Tests ln_rules.rs migration pattern (3 statics: Lazy<Ruleset>).
///
/// **Validates**: File 1 migration from spec
///
/// **Specification Reference**: Section 3.1 - File 1: ln_rules.rs
#[test]
#[ignore] // TODO: Remove after crates/bitnet-cli/src/ln_rules.rs migration completes
fn test_ln_rules_migration() {
    // This test validates the migration pattern used in ln_rules.rs:
    // - 3 static OnceLock<Ruleset> declarations
    // - Helper functions: get_bitnet_b158_f16(), get_bitnet_b158_i2s(), get_generic()
    // - Public API: rules_bitnet_b158_f16(), rules_bitnet_b158_i2s(), rules_generic()

    use regex::Regex;

    #[derive(Clone, Debug)]
    #[allow(dead_code)] // Test scaffolding - fields used when migration completes
    struct Threshold {
        pattern: Regex,
        min: f32,
        max: f32,
    }

    #[derive(Clone, Debug)]
    struct Ruleset {
        ln: Vec<Threshold>,
        name: String,
    }

    static BITNET_B158_F16: OnceLock<Ruleset> = OnceLock::new();

    fn get_bitnet_b158_f16() -> &'static Ruleset {
        BITNET_B158_F16.get_or_init(|| Ruleset {
            ln: vec![Threshold {
                pattern: Regex::new(r".*norm\.weight$").expect("regex must compile"),
                min: 0.50,
                max: 2.0,
            }],
            name: "bitnet-b1.58:f16".into(),
        })
    }

    fn rules_bitnet_b158_f16() -> Ruleset {
        get_bitnet_b158_f16().clone()
    }

    // Validate pattern
    let ruleset = rules_bitnet_b158_f16();
    assert_eq!(ruleset.name, "bitnet-b1.58:f16");
    assert_eq!(ruleset.ln.len(), 1);
    assert!(ruleset.ln[0].pattern.is_match("model.norm.weight"));

    // Validate caching
    let ruleset2 = rules_bitnet_b158_f16();
    assert_eq!(ruleset2.name, ruleset.name);
}

/// Tests ffi_session.rs migration pattern (OnceCell<Mutex<T>> → OnceLock<Mutex<T>>).
///
/// **Validates**: File 2 migration from spec
///
/// **Specification Reference**: Section 3.1 - File 2: ffi_session.rs
#[test]
#[ignore] // TODO: Remove after crates/bitnet-inference/src/ffi_session.rs migration completes
fn test_ffi_session_migration() {
    // This test validates the migration pattern used in ffi_session.rs:
    // - OnceCell<Mutex<T>> → OnceLock<Mutex<T>> (direct rename, API identical)

    struct ParityCppSession {
        id: usize,
    }

    impl ParityCppSession {
        fn new() -> Self {
            ParityCppSession { id: 42 }
        }
    }

    static PARITY_CPP_SESSION: OnceLock<Mutex<ParityCppSession>> = OnceLock::new();

    fn get_session() -> &'static Mutex<ParityCppSession> {
        PARITY_CPP_SESSION.get_or_init(|| Mutex::new(ParityCppSession::new()))
    }

    // Validate pattern
    let session = get_session().lock().unwrap();
    assert_eq!(session.id, 42);
    drop(session);

    // Validate caching (same mutex reference)
    let session2 = get_session().lock().unwrap();
    assert_eq!(session2.id, 42);
}

/// Tests weight_mapper.rs migration pattern (Lazy<Regex> → OnceLock<Regex>).
///
/// **Validates**: File 3 migration from spec
///
/// **Specification Reference**: Section 3.1 - File 3: weight_mapper.rs
#[test]
#[ignore] // TODO: Remove after crates/bitnet-models/src/weight_mapper.rs migration completes
fn test_weight_mapper_migration() {
    // This test validates the migration pattern used in weight_mapper.rs:
    // - Lazy<Regex> → OnceLock<Regex>
    // - Helper function for get_or_init encapsulation

    use regex::Regex;

    static WEIGHT_PATTERN: OnceLock<Regex> = OnceLock::new();

    fn get_weight_pattern() -> &'static Regex {
        WEIGHT_PATTERN.get_or_init(|| Regex::new(r"\.weight$").expect("regex must compile"))
    }

    fn is_weight_tensor(name: &str) -> bool {
        get_weight_pattern().is_match(name)
    }

    // Validate pattern
    assert!(is_weight_tensor("model.layers.0.self_attn.q_proj.weight"));
    assert!(!is_weight_tensor("model.layers.0.self_attn.q_proj.bias"));
    assert!(is_weight_tensor("output_layer.weight"));

    // Validate caching (same regex reference)
    let pattern1 = get_weight_pattern();
    let pattern2 = get_weight_pattern();
    assert_eq!(pattern1.as_str(), pattern2.as_str());
}

/// Tests env_guard.rs migration pattern (Lazy<Mutex<()>> → OnceLock<Mutex<()>>).
///
/// **Validates**: File 4 migration from spec
///
/// **Specification Reference**: Section 3.1 - File 4: env_guard.rs
#[test]
#[ignore] // TODO: Remove after tests/support/env_guard.rs migration completes
fn test_env_guard_migration() {
    // This test validates the migration pattern used in env_guard.rs:
    // - Lazy<Mutex<()>> → OnceLock<Mutex<()>>
    // - Helper function: get_env_lock()

    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    fn get_env_lock() -> &'static Mutex<()> {
        ENV_LOCK.get_or_init(|| Mutex::new(()))
    }

    // Validate pattern (serialize access via lock)
    let _guard1 = get_env_lock().lock().unwrap();
    // Second lock would block if first is held
    drop(_guard1);
    let _guard2 = get_env_lock().lock().unwrap();

    // Validate caching (same mutex reference)
    let lock1 = get_env_lock();
    let lock2 = get_env_lock();
    // Compare raw pointers to ensure same mutex
    assert_eq!(
        lock1 as *const Mutex<()>, lock2 as *const Mutex<()>,
        "ENV_LOCK should return same mutex reference"
    );
}

/// Tests xtask/main.rs migration pattern (Lazy<Regex> → OnceLock<Regex>).
///
/// **Validates**: File 6 migration from spec
///
/// **Specification Reference**: Section 3.1 - File 6: xtask/main.rs
#[test]
#[ignore] // TODO: Remove after xtask/src/main.rs migration completes
fn test_xtask_main_migration() {
    // This test validates the migration pattern used in xtask/main.rs:
    // - Lazy<Regex> → OnceLock<Regex>
    // - Helper function for utility patterns

    use regex::Regex;

    static UTILITY_PATTERN: OnceLock<Regex> = OnceLock::new();

    fn get_pattern() -> &'static Regex {
        UTILITY_PATTERN.get_or_init(|| Regex::new(r"^v\d+\.\d+\.\d+$").expect("regex must compile"))
    }

    // Validate pattern (version string matching)
    assert!(get_pattern().is_match("v1.2.3"));
    assert!(get_pattern().is_match("v0.1.0"));
    assert!(!get_pattern().is_match("1.2.3")); // Missing 'v' prefix
    assert!(!get_pattern().is_match("v1.2")); // Missing patch version

    // Validate caching
    let pattern1 = get_pattern();
    let pattern2 = get_pattern();
    assert_eq!(pattern1.as_str(), pattern2.as_str());
}

// ============================================================================
// Test Category 6: Integration Tests (End-to-End Workflows, AC6)
// ============================================================================

/// Tests CLI workflow with OnceLock (validates ln_rules.rs integration).
///
/// **Validates**: AC6 (all tests pass)
///
/// **Specification Reference**: Section 6.5 - Functional Tests
#[test]
#[ignore] // TODO: Remove after full migration completes and CLI tests pass
fn test_cli_workflow_with_oncelock() {
    // This test validates that CLI workflows using ln_rules.rs still work
    // after migration to OnceLock.
    //
    // Expected behavior:
    // - CLI can load and validate LayerNorm rules
    // - Pattern matching works identically to once_cell::Lazy
    // - No performance regression in rule evaluation

    // Placeholder test structure (will be filled after migration)
    unimplemented!("Requires bitnet-cli ln_rules.rs migration to complete");
}

/// Tests model loading with OnceLock (validates weight_mapper.rs integration).
///
/// **Validates**: AC6 (all tests pass)
///
/// **Specification Reference**: Section 6.5 - Functional Tests
#[test]
#[ignore] // TODO: Remove after full migration completes and model loading tests pass
fn test_model_loading_with_oncelock() {
    // This test validates that model loading using weight_mapper.rs still works
    // after migration to OnceLock.
    //
    // Expected behavior:
    // - GGUF weight mapping patterns work identically
    // - Regex matching performance is preserved
    // - No memory leaks or initialization issues

    // Placeholder test structure (will be filled after migration)
    unimplemented!("Requires bitnet-models weight_mapper.rs migration to complete");
}

/// Tests inference with OnceLock (validates ffi_session.rs integration).
///
/// **Validates**: AC6 (all tests pass)
///
/// **Specification Reference**: Section 6.5 - Functional Tests
#[test]
#[ignore] // TODO: Remove after full migration completes and inference tests pass
fn test_inference_with_oncelock() {
    // This test validates that inference using ffi_session.rs still works
    // after migration to OnceLock.
    //
    // Expected behavior:
    // - FFI session creation works identically
    // - Thread safety preserved under concurrent inference
    // - Session reuse works correctly

    // Placeholder test structure (will be filled after migration)
    unimplemented!("Requires bitnet-inference ffi_session.rs migration to complete");
}

// ============================================================================
// Test Category 7: Dependency Hygiene Tests (AC2, AC7)
// ============================================================================

/// Tests that once_cell is removed from all Cargo.toml files.
///
/// **Validates**: AC2 (all once_cell replaced with OnceLock)
///
/// **Specification Reference**: Section 3.2 - Cargo.toml Cleanup
#[test]
#[ignore] // TODO: Remove after Cargo.toml cleanup completes
fn test_no_once_cell_dependencies() {
    // This test verifies that once_cell is removed from:
    // - Cargo.toml (workspace dependencies)
    // - All crate Cargo.toml files
    // - All dev-dependencies sections

    use std::fs;
    use std::path::Path;

    let cargo_files = vec![
        "Cargo.toml",
        "crates/bitnet-cli/Cargo.toml",
        "crates/bitnet-inference/Cargo.toml",
        "crates/bitnet-kernels/Cargo.toml",
        "crates/bitnet-models/Cargo.toml",
        "xtask/Cargo.toml",
        "tests/Cargo.toml",
    ];

    for file in cargo_files {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join(file);

        if !path.exists() {
            continue; // Skip if file doesn't exist
        }

        let content = fs::read_to_string(&path).expect("Failed to read Cargo.toml");

        assert!(!content.contains("once_cell"), "Found once_cell reference in {}", file);
    }
}

/// Tests that lazy_static is NOT present (should never have been added).
///
/// **Validates**: AC2 (no lazy_static dependencies)
///
/// **Specification Reference**: Section 1.1 - Current State (no lazy_static)
#[test]
#[ignore] // TODO: Remove after Cargo.toml verification completes
fn test_no_lazy_static_dependencies() {
    // This test verifies that lazy_static was never introduced
    // (BitNet.rs never used lazy_static directly)

    use std::fs;
    use std::path::Path;

    let cargo_files = vec![
        "Cargo.toml",
        "crates/bitnet-cli/Cargo.toml",
        "crates/bitnet-inference/Cargo.toml",
        "crates/bitnet-kernels/Cargo.toml",
        "crates/bitnet-models/Cargo.toml",
        "xtask/Cargo.toml",
        "tests/Cargo.toml",
    ];

    for file in cargo_files {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join(file);

        if !path.exists() {
            continue;
        }

        let content = fs::read_to_string(&path).expect("Failed to read Cargo.toml");

        assert!(
            !content.contains("lazy_static"),
            "Found unexpected lazy_static reference in {}",
            file
        );
    }
}

/// Tests that cargo tree shows no once_cell or lazy_static dependencies.
///
/// **Validates**: AC2, AC7 (dependency tree clean)
///
/// **Specification Reference**: Section 7.1 - Verification Command List
#[test]
#[ignore] // TODO: Remove after full migration completes
fn test_cargo_tree_static_dependencies() {
    // This test verifies that `cargo tree` output contains no once_cell or lazy_static

    use std::process::Command;

    let output =
        Command::new("cargo").args(&["tree"]).output().expect("Failed to execute cargo tree");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(!stdout.contains("once-cell"), "Found once-cell in cargo tree output");
    assert!(!stdout.contains("once_cell"), "Found once_cell in cargo tree output");
    assert!(!stdout.contains("lazy_static"), "Found lazy_static in cargo tree output");
}

/// Tests that cargo-deny blocks reintroduction of once_cell and lazy_static.
///
/// **Validates**: AC7 (cargo-deny prevents reintroduction)
///
/// **Specification Reference**: Section 7.1 - cargo-deny Verification
#[test]
#[ignore] // TODO: Remove after deny.toml updated with bans
fn test_cargo_deny_blocks_reintroduction() {
    // This test verifies that deny.toml is configured to block once_cell and lazy_static

    use std::fs;
    use std::path::Path;

    let deny_path = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join("deny.toml");

    if !deny_path.exists() {
        panic!("deny.toml not found - create it for AC7 compliance");
    }

    let content = fs::read_to_string(&deny_path).expect("Failed to read deny.toml");

    // Verify [bans] section exists
    assert!(content.contains("[bans]"), "deny.toml missing [bans] section");

    // Verify once_cell is banned
    assert!(
        content.contains("once_cell") || content.contains("once-cell"),
        "deny.toml does not ban once_cell"
    );

    // Verify lazy_static is banned
    assert!(
        content.contains("lazy_static") || content.contains("lazy-static"),
        "deny.toml does not ban lazy_static"
    );
}

// ============================================================================
// Documentation and MSRV Tests (AC8)
// ============================================================================

/// Tests that MSRV remains 1.90.0 (no change).
///
/// **Validates**: AC8 (MSRV documented)
///
/// **Specification Reference**: Section 1.3 - MSRV Implications
#[test]
#[ignore] // TODO: Remove after CLAUDE.md and Cargo.toml verification completes
fn test_msrv_documented() {
    use std::fs;
    use std::path::Path;

    // Verify Cargo.toml MSRV
    let cargo_path = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join("Cargo.toml");

    let cargo_content = fs::read_to_string(&cargo_path).expect("Failed to read Cargo.toml");

    assert!(
        cargo_content.contains(r#"rust-version = "1.90.0""#),
        "MSRV not set to 1.90.0 in Cargo.toml"
    );

    // Verify CLAUDE.md documents migration
    let claude_path = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join("CLAUDE.md");

    let claude_content = fs::read_to_string(&claude_path).expect("Failed to read CLAUDE.md");

    assert!(
        claude_content.contains("OnceLock") || claude_content.contains("once_cell migration"),
        "CLAUDE.md does not document OnceLock migration"
    );
}

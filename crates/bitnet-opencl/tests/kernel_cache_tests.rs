use serial_test::serial;
use tempfile::TempDir;

use bitnet_opencl::kernel_cache::{CacheEntry, CacheKey, KernelCache, hash_source};
use bitnet_opencl::kernel_compiler::{CompilationOptions, KernelCompiler, OptimizationLevel};

// ── helpers ────────────────────────────────────────────────────────

fn test_key(name: &str) -> CacheKey {
    CacheKey {
        kernel_name: name.to_string(),
        device_id: "test-device-0".to_string(),
        compiler_options: "-O2".to_string(),
    }
}

fn test_entry(hash: u64) -> CacheEntry {
    CacheEntry {
        binary_data: vec![0xCA, 0xFE],
        source_hash: hash,
        timestamp: 1_700_000_000,
        device_name: "TestGPU".to_string(),
    }
}

// ── cache hit / miss ───────────────────────────────────────────────

#[test]
fn cache_hit_returns_stored_binary() {
    let cache = KernelCache::with_config(None, true);
    let key = test_key("matmul");
    cache.insert(key.clone(), test_entry(1));

    let entry = cache.get(&key, 1).expect("should hit");
    assert_eq!(entry.binary_data, vec![0xCA, 0xFE]);
}

#[test]
fn cache_miss_on_empty_cache() {
    let cache = KernelCache::with_config(None, true);
    assert!(cache.get(&test_key("x"), 1).is_none());
}

#[test]
fn cache_miss_triggers_compilation_via_get_or_compile() {
    let cache = KernelCache::with_config(None, true);
    let key = test_key("relu");
    let compiled = std::sync::atomic::AtomicBool::new(false);

    let _entry: Result<CacheEntry, String> = cache.get_or_compile(&key, 7, || {
        compiled.store(true, std::sync::atomic::Ordering::SeqCst);
        Ok(test_entry(7))
    });
    assert!(compiled.load(std::sync::atomic::Ordering::SeqCst));
}

// ── source-hash invalidation ───────────────────────────────────────

#[test]
fn source_change_invalidates_cache() {
    let cache = KernelCache::with_config(None, true);
    let key = test_key("conv");
    cache.insert(key.clone(), test_entry(10));

    // Source hash changed → stale.
    assert!(cache.get(&key, 99).is_none());
    assert_eq!(cache.stats().evictions, 1);
}

// ── compiler options ───────────────────────────────────────────────

#[test]
fn different_compiler_options_produce_different_entries() {
    let cache = KernelCache::with_config(None, true);
    let key_a =
        CacheKey { kernel_name: "f".into(), device_id: "d".into(), compiler_options: "-O0".into() };
    let key_b =
        CacheKey { kernel_name: "f".into(), device_id: "d".into(), compiler_options: "-O3".into() };
    cache.insert(key_a.clone(), test_entry(1));
    cache.insert(key_b.clone(), test_entry(1));

    assert!(cache.get(&key_a, 1).is_some());
    assert!(cache.get(&key_b, 1).is_some());
    assert_eq!(cache.stats().entries, 2);
}

// ── thread safety ──────────────────────────────────────────────────

#[test]
fn concurrent_access_is_safe() {
    let cache = KernelCache::with_config(None, true);
    let handles: Vec<_> = (0..8)
        .map(|i| {
            let c = cache.clone();
            std::thread::spawn(move || {
                let key = test_key(&format!("k{i}"));
                c.insert(key.clone(), test_entry(i));
                c.get(&key, i)
            })
        })
        .collect();

    for h in handles {
        assert!(h.join().unwrap().is_some());
    }
    assert_eq!(cache.stats().entries, 8);
}

#[test]
fn concurrent_get_or_compile() {
    let cache = KernelCache::with_config(None, true);
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let c = cache.clone();
            std::thread::spawn(move || {
                let key = test_key("shared");
                c.get_or_compile(&key, 1, || Ok::<CacheEntry, String>(test_entry(1)))
            })
        })
        .collect();

    for h in handles {
        assert!(h.join().unwrap().is_ok());
    }
    // Exactly 1 entry regardless of concurrency.
    assert_eq!(cache.stats().entries, 1);
}

// ── stats ──────────────────────────────────────────────────────────

#[test]
fn stats_report_correctly() {
    let cache = KernelCache::with_config(None, true);
    let key = test_key("s");
    cache.insert(key.clone(), test_entry(5));

    let _ = cache.get(&key, 5); // hit
    let _ = cache.get(&test_key("nope"), 1); // miss

    let stats = cache.stats();
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.misses, 1);
    assert_eq!(stats.entries, 1);
}

// ── disabled cache ─────────────────────────────────────────────────

#[test]
fn disabled_cache_always_recompiles() {
    let cache = KernelCache::with_config(None, false);
    let key = test_key("x");
    cache.insert(key.clone(), test_entry(1));
    assert!(cache.get(&key, 1).is_none());
    assert!(!cache.is_enabled());
}

#[test]
#[serial(bitnet_env)]
fn env_var_disables_cache() {
    temp_env::with_var("BITNET_KERNEL_CACHE", Some("0"), || {
        let cache = KernelCache::new();
        assert!(!cache.is_enabled());
    });
}

// ── custom cache directory via env var ──────────────────────────────

#[test]
#[serial(bitnet_env)]
fn custom_cache_dir_via_env() {
    let tmp = TempDir::new().unwrap();
    let dir = tmp.path().to_str().unwrap().to_string();

    temp_env::with_vars(
        [("BITNET_KERNEL_CACHE", Some("1")), ("BITNET_KERNEL_CACHE_DIR", Some(&dir))],
        || {
            let cache = KernelCache::new();
            let key = test_key("env_test");
            cache.insert(key.clone(), test_entry(42));

            // Should persist to disk inside the temp dir.
            let files: Vec<_> =
                std::fs::read_dir(tmp.path()).unwrap().filter_map(Result::ok).collect();
            assert!(!files.is_empty(), "expected at least one cache file on disk");
        },
    );
}

// ── clear ──────────────────────────────────────────────────────────

#[test]
fn clear_removes_all_entries() {
    let cache = KernelCache::with_config(None, true);
    cache.insert(test_key("a"), test_entry(1));
    cache.insert(test_key("b"), test_entry(2));
    cache.clear();

    assert_eq!(cache.stats().entries, 0);
    assert_eq!(cache.stats().evictions, 2);
}

// ── invalidate ─────────────────────────────────────────────────────

#[test]
fn invalidate_removes_single_entry() {
    let cache = KernelCache::with_config(None, true);
    let key = test_key("norm");
    cache.insert(key.clone(), test_entry(3));
    cache.invalidate(&key);
    assert!(cache.get(&key, 3).is_none());
    assert_eq!(cache.stats().evictions, 1);
}

// ── disk persistence ───────────────────────────────────────────────

#[test]
fn disk_persistence_roundtrip() {
    let tmp = TempDir::new().unwrap();
    let dir = Some(tmp.path().to_path_buf());

    let key = test_key("disk");
    let entry = test_entry(55);

    // Write.
    {
        let cache = KernelCache::with_config(dir.clone(), true);
        cache.insert(key.clone(), entry.clone());
    }

    // Read from a fresh in-memory cache backed by the same dir.
    {
        let cache = KernelCache::with_config(dir, true);
        let loaded = cache.get(&key, 55).expect("should load from disk");
        assert_eq!(loaded.binary_data, entry.binary_data);
        assert_eq!(loaded.source_hash, 55);
    }
}

#[test]
fn disk_stale_entry_rejected() {
    let tmp = TempDir::new().unwrap();
    let dir = Some(tmp.path().to_path_buf());

    let key = test_key("stale");
    {
        let cache = KernelCache::with_config(dir.clone(), true);
        cache.insert(key.clone(), test_entry(10));
    }

    // New cache instance, different source hash.
    let cache = KernelCache::with_config(dir, true);
    assert!(cache.get(&key, 999).is_none());
}

#[test]
fn clear_removes_disk_files() {
    let tmp = TempDir::new().unwrap();
    let dir = Some(tmp.path().to_path_buf());

    let cache = KernelCache::with_config(dir, true);
    cache.insert(test_key("x"), test_entry(1));
    cache.clear();

    assert!(!tmp.path().exists() || tmp.path().read_dir().unwrap().count() == 0);
}

// ── hash_source ────────────────────────────────────────────────────

#[test]
fn hash_source_is_deterministic() {
    let a = hash_source("code");
    let b = hash_source("code");
    assert_eq!(a, b);
}

#[test]
fn hash_source_differs_for_different_inputs() {
    assert_ne!(hash_source("aaa"), hash_source("bbb"));
}

// ── KernelCompiler ─────────────────────────────────────────────────

#[test]
fn compiler_compile_miss_then_hit() {
    let cache = KernelCache::with_config(None, true);
    let mut compiler = KernelCompiler::new(cache);
    compiler.register_source("k", "__kernel void k() {}");

    let opts = CompilationOptions::default();
    let _ = compiler.compile("k", &opts).unwrap();
    assert_eq!(compiler.cache().stats().misses, 1);

    let _ = compiler.compile("k", &opts).unwrap();
    assert_eq!(compiler.cache().stats().hits, 1);
}

#[test]
fn compiler_unknown_kernel_errors() {
    let cache = KernelCache::with_config(None, true);
    let compiler = KernelCompiler::new(cache);
    let opts = CompilationOptions::default();
    assert!(compiler.compile("no_such_kernel", &opts).is_err());
}

#[test]
fn compiler_disabled_cache_always_recompiles() {
    let cache = KernelCache::with_config(None, false);
    let mut compiler = KernelCompiler::new(cache);
    compiler.register_source("k", "__kernel void k() {}");
    let opts = CompilationOptions::default();

    let _ = compiler.compile("k", &opts).unwrap();
    let _ = compiler.compile("k", &opts).unwrap();
    assert_eq!(compiler.cache().stats().misses, 2);
}

#[test]
fn compilation_options_flags_contain_defines() {
    let opts = CompilationOptions {
        optimization_level: OptimizationLevel::O3,
        target_device: "gpu0".into(),
        defines: vec![("BLOCK".into(), "256".into())],
    };
    let flags = opts.to_flags_string();
    assert!(flags.contains("-O3"));
    assert!(flags.contains("-DBLOCK=256"));
}

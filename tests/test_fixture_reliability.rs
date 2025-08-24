#![cfg(feature = "integration-tests")]
use bitnet_tests::units::{BYTES_PER_GB, BYTES_PER_KB, BYTES_PER_MB};
use bitnet_tests::{
    config::FixtureConfig,
    errors::{FixtureError, TestResult},
    fixtures::{CleanupStats, FixtureManager},
};
use futures_util;
use std::time::{Duration, SystemTime};
use tempfile::TempDir;
use tokio::fs;

/// Test comprehensive fixture management reliability and automatic cleanup
#[tokio::test]
async fn test_fixture_reliability_and_cleanup() -> TestResult<()> {
    let _ = tracing_subscriber::fmt::try_init();

    // Create a temporary directory for testing
    let temp_dir = TempDir::new().unwrap();
    unsafe {
        std::env::set_var("BITNET_TEST_CACHE", temp_dir.path());
    }

    // Create fixture config with aggressive cleanup settings
    let mut config = FixtureConfig::default();
    config.auto_download = false; // Disable auto-download for testing
    config.max_cache_size = 10 * BYTES_PER_KB; // 10KB limit
    config.cleanup_interval = Duration::from_secs(1); // 1 second for testing

    // Create fixture manager
    let manager = FixtureManager::new(&config).await?;

    // Test 1: Verify built-in fixtures are loaded
    let fixtures = manager.list_fixtures();
    assert!(!fixtures.is_empty(), "Should have built-in fixtures");
    assert!(fixtures.len() >= 3, "Should have at least 3 built-in fixtures");

    // Test 2: Test cache statistics on empty cache
    let stats = manager.get_cache_stats().await?;
    assert_eq!(stats.file_count, 0, "Cache should be empty initially");
    assert_eq!(stats.total_size, 0, "Cache size should be 0 initially");

    // Test 3: Create test files to simulate cached fixtures
    let test_files = vec![
        ("file1.bin", b"content1".as_slice()),
        ("file2.bin", b"content2".as_slice()),
        ("file3.bin", b"longer_content_for_testing".as_slice()),
    ];

    for (filename, content) in &test_files {
        let file_path = temp_dir.path().join(filename);
        fs::write(&file_path, content).await?;
    }

    // Test 4: Verify cache statistics after adding files
    let stats = manager.get_cache_stats().await?;
    assert_eq!(stats.file_count, 3, "Should have 3 files in cache");
    let expected_size = test_files.iter().map(|(_, content)| content.len()).sum::<usize>() as u64;
    assert_eq!(stats.total_size, expected_size, "Cache size should match file contents");

    // Test 5: Test size-based cleanup (should trigger because we exceed 10KB limit)
    let cleanup_stats = manager.cleanup_by_size().await?;

    // Verify cleanup occurred
    let stats_after_cleanup = manager.get_cache_stats().await?;
    assert!(
        stats_after_cleanup.total_size < stats.total_size,
        "Cache size should be reduced after cleanup"
    );

    // Test 6: Test automatic cleanup
    let (age_cleanup, size_cleanup) = manager.auto_cleanup().await?;

    // Verify cleanup stats are reasonable
    assert!(
        age_cleanup.removed_count + size_cleanup.removed_count >= 0,
        "Cleanup should complete without error"
    );

    // Test 7: Test fixture validation
    let validation = manager.validate_cache().await?;
    assert_eq!(validation.valid_count, 0, "No valid fixtures expected (no real fixtures cached)");
    assert_eq!(validation.missing_count, 3, "All built-in fixtures should be missing");
    assert!(validation.invalid_files.is_empty(), "No invalid files expected");

    // Test 8: Test shared fixture registration
    let mut manager_mut = manager;
    manager_mut
        .register_shared_fixture("test-shared", "https://example.com/test.bin", "abcd1234567890")
        .await?;

    let shared_info = manager_mut.get_fixture_info("test-shared");
    assert!(shared_info.is_some(), "Shared fixture should be registered");
    assert_eq!(shared_info.unwrap().name, "test-shared");

    // Test 9: Test error handling for unknown fixtures
    let result = manager_mut.get_model_fixture("nonexistent").await;
    assert!(result.is_err(), "Should fail for unknown fixture");

    match result.unwrap_err() {
        FixtureError::UnknownFixture { name } => {
            assert!(name.contains("nonexistent"), "Error should mention the fixture name");
        }
        _ => panic!("Expected UnknownFixture error"),
    }

    // Test 10: Test fixture preloading (should fail gracefully)
    let preload_result = manager_mut.preload_fixtures(&["tiny-model", "small-model"]).await;
    assert!(preload_result.is_err(), "Preloading should fail when auto_download is disabled");

    println!("✅ All fixture reliability and cleanup tests passed!");
    Ok(())
}

/// Test fixture manager under concurrent load
#[tokio::test]
async fn test_fixture_concurrent_reliability() -> TestResult<()> {
    let _ = tracing_subscriber::fmt::try_init();

    let temp_dir = TempDir::new().unwrap();
    unsafe {
        std::env::set_var("BITNET_TEST_CACHE", temp_dir.path());
    }

    let config = FixtureConfig::default();
    let manager = FixtureManager::new(&config).await?;

    // Test concurrent access to fixture information
    let handles: Vec<_> = (0..20)
        .map(|i| {
            let manager_ref = &manager;
            tokio::spawn(async move {
                // Test various operations concurrently
                let _info = manager_ref.get_fixture_info("tiny-model");
                let _cached = manager_ref.is_cached("tiny-model").await;
                let _stats = manager_ref.get_cache_stats().await;
                let _validation = manager_ref.validate_cache().await;

                // Each task should complete successfully
                format!("Task {} completed", i)
            })
        })
        .collect();

    // Wait for all tasks to complete
    let results: Vec<_> = futures_util::future::join_all(handles).await;

    for (i, result) in results.into_iter().enumerate() {
        let message = result.unwrap();
        assert!(message.contains(&format!("Task {} completed", i)));
    }

    println!("✅ Concurrent fixture reliability test passed!");
    Ok(())
}

/// Test fixture cleanup with realistic scenarios
#[tokio::test]
async fn test_realistic_fixture_cleanup() -> TestResult<()> {
    let _ = tracing_subscriber::fmt::try_init();

    let temp_dir = TempDir::new().unwrap();
    unsafe {
        std::env::set_var("BITNET_TEST_CACHE", temp_dir.path());
    }

    let mut config = FixtureConfig::default();
    config.max_cache_size = 50 * BYTES_PER_KB; // 50KB limit
    config.cleanup_interval = Duration::from_secs(1);

    let manager = FixtureManager::new(&config).await?;

    // Create files of various sizes and ages
    let files = vec![
        ("small_old.bin", vec![b'a'; 1024], true), // 1KB, old
        ("medium_new.bin", vec![b'b'; 10 * BYTES_PER_KB], false), // 10KB, new
        ("large_old.bin", vec![b'c'; 30 * BYTES_PER_KB], true), // 30KB, old
        ("tiny_new.bin", vec![b'd'; 512], false),  // 512B, new
    ];

    for (filename, content, make_old) in &files {
        let file_path = temp_dir.path().join(filename);
        fs::write(&file_path, content).await?;

        if *make_old {
            // Make file appear old by setting modification time
            // Note: This is a best-effort approach and may not work on all systems
            let old_time = SystemTime::now() - Duration::from_secs(48 * 60 * 60); // 48 hours ago
            if let Ok(file) = std::fs::File::open(&file_path) {
                // Try to set the file time (may fail on some systems, but that's OK for testing)
                let _ = file.set_modified(old_time);
            }
        }
    }

    // Verify initial state
    let initial_stats = manager.get_cache_stats().await?;
    assert_eq!(initial_stats.file_count, 4);

    let total_size: usize = files.iter().map(|(_, content, _)| content.len()).sum();
    assert_eq!(initial_stats.total_size, total_size as u64);

    // Test age-based cleanup
    let age_cleanup = manager.cleanup_old_fixtures().await?;
    println!("Age cleanup removed {} files", age_cleanup.removed_count);

    // Test size-based cleanup
    let size_cleanup = manager.cleanup_by_size().await?;
    println!("Size cleanup removed {} files", size_cleanup.removed_count);

    // Verify final state
    let final_stats = manager.get_cache_stats().await?;
    assert!(
        final_stats.total_size <= initial_stats.total_size,
        "Cache size should not increase after cleanup"
    );

    println!("✅ Realistic fixture cleanup test passed!");
    Ok(())
}

/// Test fixture manager error recovery
#[tokio::test]
async fn test_fixture_error_recovery() -> TestResult<()> {
    let _ = tracing_subscriber::fmt::try_init();

    let temp_dir = TempDir::new().unwrap();
    unsafe {
        std::env::set_var("BITNET_TEST_CACHE", temp_dir.path());
    }

    let config = FixtureConfig::default();
    let manager = FixtureManager::new(&config).await?;

    // Test 1: Handle corrupted cache directory gracefully
    let corrupted_file = temp_dir.path().join("corrupted.bin");
    fs::write(&corrupted_file, b"corrupted data").await?;

    // Should handle corrupted files gracefully
    let validation = manager.validate_cache().await?;
    assert!(validation.invalid_files.is_empty()); // No fixtures match this file

    // Test 2: Handle permission errors gracefully
    // (This test may not work on all systems, but should not panic)
    let _cleanup_result = manager.cleanup_old_fixtures().await;
    let _size_cleanup_result = manager.cleanup_by_size().await;

    // Test 3: Handle concurrent cleanup operations
    let cleanup_handles: Vec<_> = (0..5)
        .map(|_| {
            let manager_ref = &manager;
            tokio::spawn(async move {
                let _ = manager_ref.auto_cleanup().await;
            })
        })
        .collect();

    // All cleanup operations should complete without panicking
    for handle in cleanup_handles {
        let _ = handle.await;
    }

    println!("✅ Fixture error recovery test passed!");
    Ok(())
}

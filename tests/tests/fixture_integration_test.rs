#[cfg(feature = "fixtures")]
mod fixtures_tests {
    use bitnet_tests::common::{
        config::FixtureConfig,
        fixtures::{FixtureManager, ModelFormat, ModelType},
    };
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_fixture_manager_integration() {
        // Create a temporary directory for testing
        let temp_dir = TempDir::new().unwrap();
        unsafe {
            std::env::set_var("BITNET_TEST_CACHE", temp_dir.path());
        }

        // Create fixture config
        let mut config = FixtureConfig::default();
        config.auto_download = false; // Disable auto-download for testing

        // Create fixture manager
        let manager = FixtureManager::new(&config).await.unwrap();

        // Test that built-in fixtures are loaded
        let fixtures = manager.list_fixtures();
        assert!(!fixtures.is_empty());

        // Test fixture info retrieval
        let tiny_model = manager.get_fixture_info("tiny-model");
        assert!(tiny_model.is_some());

        let info = tiny_model.unwrap();
        assert_eq!(info.name, "tiny-model");
        assert_eq!(info.filename, "tiny-bitnet.gguf");
        assert!(matches!(info.model_type, ModelType::BitNet));
        assert!(matches!(info.format, ModelFormat::Gguf));

        // Test cache stats
        let stats = manager.get_cache_stats().await.unwrap();
        assert_eq!(stats.file_count, 0); // No files cached yet

        // Test fixture validation
        let validation = manager.validate_cache().await.unwrap();
        assert_eq!(validation.valid_count, 0);
        assert_eq!(validation.missing_count, 3); // 3 built-in fixtures not cached

        // Test shared fixture registration
        let mut manager_mut = manager;
        manager_mut
            .register_shared_fixture("test-shared", "https://example.com/test.gguf", "abcd1234")
            .await
            .unwrap();

        let shared_info = manager_mut.get_fixture_info("test-shared");
        assert!(shared_info.is_some());
        assert_eq!(shared_info.unwrap().name, "test-shared");
    }

    #[tokio::test]
    async fn test_fixture_lifecycle_management() {
        let temp_dir = TempDir::new().unwrap();
        unsafe {
            std::env::set_var("BITNET_TEST_CACHE", temp_dir.path());
        }

        let config = FixtureConfig::default();
        let manager = FixtureManager::new(&config).await.unwrap();

        // Test cleanup operations
        let cleanup_stats = manager.cleanup_old_fixtures().await.unwrap();
        assert_eq!(cleanup_stats.removed_count, 0); // No old files to clean

        let size_cleanup_stats = manager.cleanup_by_size().await.unwrap();
        assert_eq!(size_cleanup_stats.removed_count, 0); // No files over size limit

        // Test cache validation
        let validation = manager.validate_cache().await.unwrap();
        assert!(validation.invalid_files.is_empty());

        // Test invalid fixture removal
        let removal_stats = manager.remove_invalid_fixtures().await.unwrap();
        assert_eq!(removal_stats.removed_count, 0); // No invalid files to remove
    }
}

// Placeholder test when fixtures feature is not enabled
#[cfg(not(feature = "fixtures"))]
#[test]
fn test_fixtures_disabled() {
    // This test ensures the file compiles when fixtures feature is disabled
    assert!(true, "Fixtures feature is disabled");
}

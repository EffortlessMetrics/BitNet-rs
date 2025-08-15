/// Facade for fixture management that provides a consistent API regardless of feature flags
/// This reduces cfg scatter throughout the codebase

use crate::errors::TestOpResult;
use crate::config::TestConfig;

#[cfg(feature = "fixtures")]
use std::sync::Arc;

/// Facade for fixture manager that adapts based on feature flags
#[cfg(feature = "fixtures")]
#[derive(Clone)]
pub struct Fixtures(pub Arc<crate::fixtures::FixtureManager>);

#[cfg(not(feature = "fixtures"))]
#[derive(Clone, Default)]
pub struct Fixtures;

impl Fixtures {
    /// Create a new fixtures facade from config
    pub async fn new(cfg: &TestConfig) -> TestOpResult<Self> {
        #[cfg(feature = "fixtures")]
        {
            let inner = crate::fixtures::FixtureManager::new(&cfg.fixtures).await?;
            return Ok(Self(Arc::new(inner)));
        }
        #[cfg(not(feature = "fixtures"))]
        {
            let _ = cfg;
            return Ok(Self);
        }
    }

    /// Cleanup old fixtures from cache
    pub async fn cleanup_old_fixtures(&self) -> TestOpResult<()> {
        #[cfg(feature = "fixtures")]
        {
            self.0.cleanup_old_fixtures().await.map_err(|e| {
                use crate::errors::TestError;
                TestError::setup(format!("Failed to cleanup fixtures: {}", e))
            })?;
            Ok(())
        }
        #[cfg(not(feature = "fixtures"))]
        Ok(())
    }

    /// Get a model fixture by name
    pub async fn get_model_fixture(&self, _name: &str) -> TestOpResult<std::path::PathBuf> {
        #[cfg(feature = "fixtures")]
        return self.0.get_model_fixture(_name).await.map_err(|e| {
            use crate::errors::TestError;
            TestError::setup(format!("Failed to get model fixture: {}", e))
        });
        #[cfg(not(feature = "fixtures"))]
        {
            use crate::errors::TestError;
            Err(TestError::setup("Fixtures feature not enabled"))
        }
    }

    /// Get a dataset fixture by name
    pub async fn get_dataset_fixture(&self, _name: &str) -> TestOpResult<std::path::PathBuf> {
        #[cfg(feature = "fixtures")]
        return self.0.get_dataset_fixture(_name).await.map_err(|e| {
            use crate::errors::TestError;
            TestError::setup(format!("Failed to get dataset fixture: {}", e))
        });
        #[cfg(not(feature = "fixtures"))]
        {
            use crate::errors::TestError;
            Err(TestError::setup("Fixtures feature not enabled"))
        }
    }

    /// Get cache statistics
    pub async fn get_cache_stats(&self) -> TestOpResult<CacheStats> {
        #[cfg(feature = "fixtures")]
        {
            let stats = self.0.get_cache_stats().await.map_err(|e| {
                use crate::errors::TestError;
                TestError::setup(format!("Failed to get cache stats: {}", e))
            })?;
            Ok(CacheStats {
                total_size: stats.total_size,
                file_count: stats.file_count,
                cache_hits: 0,  // Not tracked in current implementation
                cache_misses: 0, // Not tracked in current implementation
            })
        }
        #[cfg(not(feature = "fixtures"))]
        Ok(CacheStats::default())
    }
}

/// Facade for cache statistics
#[derive(Debug, Default, Clone)]
pub struct CacheStats {
    pub total_size: u64,
    pub file_count: usize,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

#[cfg(feature = "fixtures")]
impl std::ops::Deref for Fixtures {
    type Target = crate::fixtures::FixtureManager;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
//! Caching and performance optimization for BitNet server
#![cfg_attr(doc, allow(dead_code, unused_imports, unused_variables))]

#[cfg(feature = "connection_pool")]
pub mod connection_pool;
pub mod kv_cache;
pub mod model_cache;
#[cfg(any(test, feature = "tuning"))]
pub mod performance_tuning;
#[cfg(feature = "request_batching")]
pub mod request_batching;

// ---- Stubs for disabled features (types only; keep callers compiling) ----
#[cfg(not(feature = "request_batching"))]
pub mod request_batching {
    #[doc(hidden)]
    #[derive(Default, Clone, Debug, serde::Serialize)]
    pub struct BatchingStatistics {}
}

#[cfg(not(feature = "connection_pool"))]
pub mod connection_pool {
    #[doc(hidden)]
    #[derive(Default, Clone, Debug, serde::Serialize)]
    pub struct ConnectionStatistics {}
}

#[cfg(not(any(test, feature = "tuning")))]
pub mod performance_tuning {
    #[doc(hidden)]
    #[derive(Default, Clone, Debug, serde::Serialize)]
    pub struct PerformanceStatistics {}
}

use anyhow::Result;
use std::sync::Arc;

/// Caching configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CachingConfig {
    /// Enable model caching
    pub model_cache_enabled: bool,
    /// Maximum number of models to cache
    pub model_cache_size: usize,
    /// Model cache TTL in seconds
    pub model_cache_ttl: u64,
    /// Enable KV cache optimization
    pub kv_cache_enabled: bool,
    /// KV cache size in MB
    pub kv_cache_size_mb: usize,
    /// Enable request batching
    pub batching_enabled: bool,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Batch timeout in milliseconds
    pub batch_timeout_ms: u64,
    /// Connection pool size
    pub connection_pool_size: usize,
    /// Enable automatic performance tuning
    pub auto_tuning_enabled: bool,
    /// Performance tuning interval in seconds
    pub tuning_interval_seconds: u64,
}

impl Default for CachingConfig {
    fn default() -> Self {
        Self {
            model_cache_enabled: true,
            model_cache_size: 3,
            model_cache_ttl: 3600, // 1 hour
            kv_cache_enabled: true,
            kv_cache_size_mb: 512,
            batching_enabled: true,
            max_batch_size: 8,
            batch_timeout_ms: 10,
            connection_pool_size: 100,
            auto_tuning_enabled: true,
            tuning_interval_seconds: 300, // 5 minutes
        }
    }
}

/// Central caching system that coordinates all caching components
pub struct CachingSystem {
    config: CachingConfig,
    model_cache: Arc<model_cache::ModelCache>,
    kv_cache_manager: Arc<kv_cache::KVCacheManager>,
    #[cfg(feature = "request_batching")]
    request_batcher: Arc<request_batching::RequestBatcher>,
    #[cfg(feature = "connection_pool")]
    connection_pool: Arc<connection_pool::ConnectionPool>,
    #[cfg(any(test, feature = "tuning"))]
    performance_tuner: Arc<tokio::sync::RwLock<performance_tuning::PerformanceTuner>>,
}

impl CachingSystem {
    /// Create a new caching system
    pub async fn new(config: CachingConfig) -> Result<Self> {
        let model_cache = Arc::new(model_cache::ModelCache::new(&config).await?);
        let kv_cache_manager = Arc::new(kv_cache::KVCacheManager::new(&config)?);
        #[cfg(feature = "request_batching")]
        let request_batcher = Arc::new(request_batching::RequestBatcher::new(&config).await?);
        #[cfg(feature = "connection_pool")]
        let connection_pool = Arc::new(connection_pool::ConnectionPool::new(&config)?);
        #[cfg(any(test, feature = "tuning"))]
        let performance_tuner =
            Arc::new(tokio::sync::RwLock::new(performance_tuning::PerformanceTuner::new(&config)?));

        Ok(Self {
            config,
            model_cache,
            kv_cache_manager,
            #[cfg(feature = "request_batching")]
            request_batcher,
            #[cfg(feature = "connection_pool")]
            connection_pool,
            #[cfg(any(test, feature = "tuning"))]
            performance_tuner,
        })
    }

    /// Get the model cache
    pub fn model_cache(&self) -> Arc<model_cache::ModelCache> {
        self.model_cache.clone()
    }

    /// Get the KV cache manager
    pub fn kv_cache_manager(&self) -> Arc<kv_cache::KVCacheManager> {
        self.kv_cache_manager.clone()
    }

    /// Get the request batcher
    #[cfg(feature = "request_batching")]
    pub fn request_batcher(&self) -> Arc<request_batching::RequestBatcher> {
        self.request_batcher.clone()
    }

    /// Get the connection pool
    #[cfg(feature = "connection_pool")]
    pub fn connection_pool(&self) -> Arc<connection_pool::ConnectionPool> {
        self.connection_pool.clone()
    }

    /// Start background optimization tasks
    pub async fn start_background_tasks(&self) -> Result<()> {
        // Start model cache cleanup task
        if self.config.model_cache_enabled {
            let model_cache = self.model_cache.clone();
            tokio::spawn(async move {
                model_cache.start_cleanup_task().await;
            });
        }

        // Start KV cache optimization task
        if self.config.kv_cache_enabled {
            let kv_cache_manager = self.kv_cache_manager.clone();
            tokio::spawn(async move {
                kv_cache_manager.start_optimization_task().await;
            });
        }

        // Start request batching task
        #[cfg(feature = "request_batching")]
        if self.config.batching_enabled {
            let request_batcher = self.request_batcher.clone();
            tokio::spawn(async move {
                request_batcher.start_batching_task().await;
            });
        }

        // Start performance tuning task
        #[cfg(any(test, feature = "tuning"))]
        if self.config.auto_tuning_enabled {
            let performance_tuner = self.performance_tuner.clone();
            let interval = self.config.tuning_interval_seconds;
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(std::time::Duration::from_secs(interval));
                loop {
                    interval.tick().await;
                    {
                        let mut tuner = performance_tuner.write().await;
                        if let Err(e) = tuner.run_optimization().await {
                            eprintln!("Performance tuning error: {}", e);
                        }
                    }
                }
            });
        }

        Ok(())
    }

    /// Get caching statistics
    pub async fn get_statistics(&self) -> CachingStatistics {
        let model_cache_stats = self.model_cache.get_statistics().await;
        let kv_cache_stats = self.kv_cache_manager.get_statistics().await;
        #[cfg(feature = "request_batching")]
        let batching_stats = self.request_batcher.get_statistics().await;
        #[cfg(not(feature = "request_batching"))]
        let batching_stats = request_batching::BatchingStatistics::default();
        #[cfg(feature = "connection_pool")]
        let connection_stats = self.connection_pool.get_statistics().await;
        #[cfg(not(feature = "connection_pool"))]
        let connection_stats = connection_pool::ConnectionStatistics::default();
        #[cfg(any(test, feature = "tuning"))]
        let performance_stats = self.performance_tuner.read().await.get_statistics().await;
        #[cfg(not(any(test, feature = "tuning")))]
        let performance_stats = performance_tuning::PerformanceStatistics::default();

        CachingStatistics {
            model_cache: model_cache_stats,
            kv_cache: kv_cache_stats,
            batching: batching_stats,
            connections: connection_stats,
            performance: performance_stats,
        }
    }

    /// Shutdown the caching system gracefully
    pub async fn shutdown(&self) -> Result<()> {
        println!("Shutting down caching system");

        // Shutdown components in reverse order
        #[cfg(feature = "request_batching")]
        self.request_batcher.shutdown().await?;
        self.kv_cache_manager.shutdown().await?;
        self.model_cache.shutdown().await?;
        #[cfg(feature = "connection_pool")]
        self.connection_pool.shutdown().await?;

        Ok(())
    }
}

/// Combined caching statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct CachingStatistics {
    pub model_cache: model_cache::ModelCacheStatistics,
    pub kv_cache: kv_cache::KVCacheStatistics,
    pub batching: request_batching::BatchingStatistics,
    pub connections: connection_pool::ConnectionStatistics,
    pub performance: performance_tuning::PerformanceStatistics,
}

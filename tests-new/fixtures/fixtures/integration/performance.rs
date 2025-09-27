//! Performance integration test fixtures
//!
//! Provides test fixtures for validating GGUF weight loading
//! performance characteristics and benchmarking

use anyhow::Result;
use std::path::PathBuf;

/// Get performance fixtures directory path
pub fn get_performance_fixtures_dir() -> PathBuf {
    super::get_integration_fixtures_dir().join("performance")
}

/// Performance benchmark categories
#[derive(Debug, Clone)]
pub enum PerformanceBenchmark {
    LoadTime,
    MemoryUsage,
    ThroughputTokensPerSecond,
    Accuracy,
}

impl PerformanceBenchmark {
    /// Get all available benchmark types
    pub fn all() -> Vec<Self> {
        vec![Self::LoadTime, Self::MemoryUsage, Self::ThroughputTokensPerSecond, Self::Accuracy]
    }

    /// Get benchmark name
    pub fn name(&self) -> &'static str {
        match self {
            Self::LoadTime => "load_time",
            Self::MemoryUsage => "memory_usage",
            Self::ThroughputTokensPerSecond => "throughput",
            Self::Accuracy => "accuracy",
        }
    }
}

/// Validate performance fixtures are available
pub fn validate_performance_fixtures() -> Result<()> {
    let performance_dir = get_performance_fixtures_dir();

    if !performance_dir.exists() {
        return Err(anyhow::anyhow!(
            "Performance fixtures directory not found: {}",
            performance_dir.display()
        ));
    }

    Ok(())
}

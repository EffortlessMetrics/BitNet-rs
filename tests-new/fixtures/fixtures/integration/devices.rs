//! Device-specific integration test fixtures
//!
//! Provides test fixtures for validating GGUF weight loading
//! across different device types (CPU, GPU, Metal)

use anyhow::Result;
use std::path::PathBuf;

/// Get device fixtures directory path
pub fn get_device_fixtures_dir() -> PathBuf {
    super::get_integration_fixtures_dir().join("devices")
}

/// Device fixture types available for testing
#[derive(Debug, Clone)]
pub enum DeviceFixtureType {
    Cpu,
    CudaGpu,
    MetalGpu,
}

impl DeviceFixtureType {
    /// Get all available device fixture types
    pub fn all() -> Vec<Self> {
        vec![Self::Cpu, Self::CudaGpu, Self::MetalGpu]
    }

    /// Get fixture type name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::CudaGpu => "cuda",
            Self::MetalGpu => "metal",
        }
    }
}

/// Validate device fixtures are available
pub fn validate_device_fixtures() -> Result<()> {
    let devices_dir = get_device_fixtures_dir();

    if !devices_dir.exists() {
        return Err(anyhow::anyhow!(
            "Device fixtures directory not found: {}",
            devices_dir.display()
        ));
    }

    Ok(())
}

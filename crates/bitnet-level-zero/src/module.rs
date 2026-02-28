//! SPIR-V module loading for Level-Zero.
//!
//! Wraps `ze_module_handle_t` creation from SPIR-V binary blobs.

use crate::context::LevelZeroContext;
use crate::error::{LevelZeroError, Result};
use crate::ffi::{ZeModuleFormat, ZeModuleHandle};

/// Configuration for loading a SPIR-V module.
#[derive(Debug, Clone)]
pub struct ModuleConfig {
    /// Module format (SPIR-V or native).
    pub format: ZeModuleFormat,
    /// Optional build flags (e.g. "-cl-fast-relaxed-math").
    pub build_flags: Option<String>,
}

impl Default for ModuleConfig {
    fn default() -> Self {
        Self {
            format: ZeModuleFormat::IlSpirv,
            build_flags: None,
        }
    }
}

/// Builder for creating a Level-Zero module from SPIR-V.
#[derive(Debug)]
pub struct ModuleBuilder {
    config: ModuleConfig,
    spirv_data: Vec<u8>,
}

impl ModuleBuilder {
    /// Create a builder from raw SPIR-V bytes.
    pub fn from_spirv(spirv: &[u8]) -> Self {
        Self {
            config: ModuleConfig::default(),
            spirv_data: spirv.to_vec(),
        }
    }

    /// Set the module format.
    pub fn format(mut self, fmt: ZeModuleFormat) -> Self {
        self.config.format = fmt;
        self
    }

    /// Set build flags.
    pub fn build_flags(mut self, flags: impl Into<String>) -> Self {
        self.config.build_flags = Some(flags.into());
        self
    }

    /// Size of the SPIR-V blob in bytes.
    pub fn spirv_size(&self) -> usize {
        self.spirv_data.len()
    }

    /// Build the module within the given context.
    ///
    /// Placeholder: real implementation calls `zeModuleCreate`.
    pub fn build(self, _ctx: &LevelZeroContext) -> Result<LevelZeroModule> {
        if self.spirv_data.is_empty() {
            return Err(LevelZeroError::InvalidArgument {
                message: "SPIR-V data is empty".into(),
            });
        }
        tracing::debug!(
            spirv_bytes = self.spirv_data.len(),
            "Loading SPIR-V module (placeholder)"
        );
        Ok(LevelZeroModule {
            config: self.config,
            spirv_size: self.spirv_data.len(),
            _handle: None,
        })
    }
}

/// An owned Level-Zero module.
pub struct LevelZeroModule {
    config: ModuleConfig,
    spirv_size: usize,
    _handle: Option<ZeModuleHandle>,
}

impl LevelZeroModule {
    /// Size of the original SPIR-V blob.
    pub fn spirv_size(&self) -> usize {
        self.spirv_size
    }

    /// Whether this module has a live L0 handle.
    pub fn is_initialized(&self) -> bool {
        self._handle.is_some()
    }

    /// The build flags used (if any).
    pub fn build_flags(&self) -> Option<&str> {
        self.config.build_flags.as_deref()
    }
}

impl std::fmt::Debug for LevelZeroModule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LevelZeroModule")
            .field("spirv_size", &self.spirv_size)
            .field("format", &self.config.format)
            .field("initialized", &self.is_initialized())
            .finish()
    }
}

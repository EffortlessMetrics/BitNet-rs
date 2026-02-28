//! MTLComputePipelineState management abstraction.
//!
//! Provides a `PipelineCache` that compiles MSL source into pipeline state
//! objects (on macOS) or validates source availability (cross-platform).

use crate::error::{MetalError, Result};
use crate::shader;
use std::collections::HashMap;

/// Descriptor for a compiled (or compilable) compute pipeline.
#[derive(Debug, Clone)]
pub struct PipelineDescriptor {
    /// Kernel function name (entry point in MSL source).
    pub function_name: &'static str,
    /// MSL source code.
    pub source: &'static str,
    /// Threadgroup size (threads per threadgroup).
    pub threadgroup_size: (u32, u32, u32),
}

/// Cache of compute pipeline descriptors, keyed by kernel name.
///
/// On macOS with the `metal` crate, this would hold `MTLComputePipelineState`
/// objects. In the cross-platform stub, it tracks descriptors for validation.
pub struct PipelineCache {
    pipelines: HashMap<String, PipelineDescriptor>,
}

impl PipelineCache {
    /// Build a new cache with all built-in kernel pipelines.
    pub fn new() -> Self {
        let mut pipelines = HashMap::new();
        pipelines.insert(
            "matmul".into(),
            PipelineDescriptor {
                function_name: "matmul",
                source: shader::MATMUL_MSL,
                threadgroup_size: (16, 16, 1),
            },
        );
        pipelines.insert(
            "softmax".into(),
            PipelineDescriptor {
                function_name: "softmax",
                source: shader::SOFTMAX_MSL,
                threadgroup_size: (256, 1, 1),
            },
        );
        pipelines.insert(
            "rmsnorm".into(),
            PipelineDescriptor {
                function_name: "rmsnorm",
                source: shader::RMSNORM_MSL,
                threadgroup_size: (256, 1, 1),
            },
        );
        pipelines.insert(
            "rope".into(),
            PipelineDescriptor {
                function_name: "rope",
                source: shader::ROPE_MSL,
                threadgroup_size: (256, 1, 1),
            },
        );
        pipelines.insert(
            "attention".into(),
            PipelineDescriptor {
                function_name: "attention",
                source: shader::ATTENTION_MSL,
                threadgroup_size: (256, 1, 1),
            },
        );
        Self { pipelines }
    }

    /// Look up a pipeline descriptor by kernel name.
    pub fn get(&self, name: &str) -> Option<&PipelineDescriptor> {
        self.pipelines.get(name)
    }

    /// Return all registered kernel names.
    pub fn kernel_names(&self) -> Vec<&str> {
        self.pipelines.keys().map(|s| s.as_str()).collect()
    }

    /// Number of cached pipelines.
    pub fn len(&self) -> usize {
        self.pipelines.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.pipelines.is_empty()
    }

    /// Validate that a kernel exists and its source is non-empty.
    pub fn validate(&self, name: &str) -> Result<()> {
        let desc = self
            .get(name)
            .ok_or_else(|| MetalError::ShaderCompilation(format!("unknown kernel: {name}")))?;
        if desc.source.is_empty() {
            return Err(MetalError::ShaderCompilation(format!(
                "empty source for kernel: {name}"
            )));
        }
        Ok(())
    }
}

impl Default for PipelineCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_has_all_kernels() {
        let cache = PipelineCache::new();
        assert_eq!(cache.len(), 5);
        assert!(cache.get("matmul").is_some());
        assert!(cache.get("softmax").is_some());
        assert!(cache.get("rmsnorm").is_some());
        assert!(cache.get("rope").is_some());
        assert!(cache.get("attention").is_some());
    }

    #[test]
    fn cache_missing_returns_none() {
        let cache = PipelineCache::new();
        assert!(cache.get("nonexistent").is_none());
    }

    #[test]
    fn validate_existing_kernel_succeeds() {
        let cache = PipelineCache::new();
        assert!(cache.validate("matmul").is_ok());
        assert!(cache.validate("attention").is_ok());
    }

    #[test]
    fn validate_missing_kernel_fails() {
        let cache = PipelineCache::new();
        assert!(cache.validate("nonexistent").is_err());
    }

    #[test]
    fn matmul_has_2d_threadgroup() {
        let cache = PipelineCache::new();
        let desc = cache.get("matmul").unwrap();
        assert_eq!(desc.threadgroup_size, (16, 16, 1));
    }

    #[test]
    fn reduction_kernels_have_256_threadgroup() {
        let cache = PipelineCache::new();
        for name in ["softmax", "rmsnorm", "attention"] {
            let desc = cache.get(name).unwrap();
            assert_eq!(desc.threadgroup_size.0, 256, "{name} should use 256 threads");
        }
    }
}

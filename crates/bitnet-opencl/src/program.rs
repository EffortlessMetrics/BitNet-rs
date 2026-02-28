//! Kernel compilation and caching.

use crate::context::OpenClContext;
use crate::error::{OpenClError, Result};
use opencl3::program::Program;
use std::collections::HashMap;
use tracing::{info, warn};

/// Compiled OpenCL program with optional caching.
pub struct OpenClProgram {
    /// The compiled opencl3 program handle.
    pub inner: Program,
    /// Human-readable name for diagnostics.
    pub name: String,
}

impl std::fmt::Debug for OpenClProgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenClProgram")
            .field("name", &self.name)
            .finish()
    }
}

// SAFETY: Compiled programs are immutable and safe to share.
unsafe impl Send for OpenClProgram {}
unsafe impl Sync for OpenClProgram {}

impl OpenClProgram {
    /// Compile an OpenCL program from source.
    pub fn from_source(
        ctx: &OpenClContext,
        source: &str,
        name: &str,
    ) -> Result<Self> {
        let inner =
            Program::create_and_build_from_source(&ctx.inner, source, "")
                .map_err(|e| OpenClError::ProgramCompilation {
                    name: name.to_string(),
                    reason: e.to_string(),
                })?;

        info!("Compiled OpenCL program: {}", name);
        Ok(Self {
            inner,
            name: name.to_string(),
        })
    }

    /// Try to compile; returns `None` instead of error on failure.
    pub fn try_from_source(
        ctx: &OpenClContext,
        source: &str,
        name: &str,
    ) -> Option<Self> {
        match Self::from_source(ctx, source, name) {
            Ok(p) => Some(p),
            Err(e) => {
                warn!(
                    "Failed to compile OpenCL program '{}': {}",
                    name, e
                );
                None
            }
        }
    }
}

/// A simple cache for compiled OpenCL programs keyed by name.
pub struct ProgramCache {
    programs: HashMap<String, OpenClProgram>,
}

impl ProgramCache {
    /// Create an empty program cache.
    pub fn new() -> Self {
        Self {
            programs: HashMap::new(),
        }
    }

    /// Compile and cache a program.
    pub fn compile_and_insert(
        &mut self,
        ctx: &OpenClContext,
        source: &str,
        name: &str,
    ) -> Result<()> {
        let program = OpenClProgram::from_source(ctx, source, name)?;
        self.programs.insert(name.to_string(), program);
        Ok(())
    }

    /// Get a cached program by name.
    pub fn get(&self, name: &str) -> Option<&OpenClProgram> {
        self.programs.get(name)
    }

    /// Number of cached programs.
    pub fn len(&self) -> usize {
        self.programs.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.programs.is_empty()
    }
}

impl Default for ProgramCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn program_cache_empty_by_default() {
        let cache = ProgramCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert!(cache.get("nonexistent").is_none());
    }

    #[test]
    fn program_cache_default_is_empty() {
        let cache = ProgramCache::default();
        assert!(cache.is_empty());
    }

    #[test]
    fn program_debug_formatting() {
        let dbg = format!(
            "{:?}",
            "OpenClProgram { name: \"matmul\" }"
        );
        assert!(dbg.contains("OpenClProgram"));
    }
}

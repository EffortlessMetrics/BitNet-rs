//! Vulkan compute kernel sources and registry.
//!
//! Embeds GLSL shader sources as compile-time constants. A pre-compiled
//! SPIR-V binary is used when available (built by `build.rs` with `glslc`),
//! otherwise the raw GLSL source is available for runtime compilation.
//!
//! The [`KernelRegistry`] maps kernel names to their GLSL source and default
//! workgroup configuration, enabling dynamic pipeline creation.

/// GLSL source for the tiled matrix multiply compute shader.
pub const MATMUL_GLSL: &str = include_str!("matmul.glsl");

/// GLSL source for the row-wise softmax compute shader.
pub const SOFTMAX_GLSL: &str = include_str!("softmax.comp");

/// GLSL source for the RMS normalization compute shader.
pub const RMSNORM_GLSL: &str = include_str!("rmsnorm.comp");

/// GLSL source for the rotary position embedding compute shader.
pub const ROPE_GLSL: &str = include_str!("rope.comp");

/// GLSL source for the scaled dot-product attention compute shader.
pub const ATTENTION_GLSL: &str = include_str!("attention.comp");

/// GLSL source for the token embedding lookup compute shader.
pub const EMBEDDING_GLSL: &str = include_str!("embedding.comp");

/// GLSL source for the SiLU activation compute shader.
pub const SILU_GLSL: &str = include_str!("silu.comp");

/// Pre-compiled SPIR-V binary for the matmul shader (if available).
///
/// Built by `build.rs` when `glslc` is found on PATH. When `None`, the
/// consumer must compile `MATMUL_GLSL` at runtime via a GLSL-to-SPIR-V
/// compiler (e.g. `shaderc`).
#[cfg(feature = "precompiled-spirv")]
pub const MATMUL_SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/matmul.spv"));

/// Tile size used by the matmul shader (workgroup dimensions).
pub const TILE_SIZE: u32 = 16;

/// Default workgroup size for 1-D compute shaders.
pub const DEFAULT_WORKGROUP_SIZE: u32 = 256;

/// Descriptor for a single compute kernel.
#[derive(Debug, Clone)]
pub struct KernelDescriptor {
    /// Human-readable kernel name.
    pub name: &'static str,
    /// GLSL 450 source code.
    pub source: &'static str,
    /// Workgroup X dimension.
    pub workgroup_x: u32,
    /// Workgroup Y dimension.
    pub workgroup_y: u32,
    /// Workgroup Z dimension.
    pub workgroup_z: u32,
}

/// Registry mapping kernel names to their shader source and workgroup config.
///
/// Used to enumerate available kernels and build compute pipelines dynamically.
pub struct KernelRegistry {
    entries: Vec<KernelDescriptor>,
}

impl KernelRegistry {
    /// Build the default registry with all built-in kernels.
    pub fn new() -> Self {
        let entries = vec![
            KernelDescriptor {
                name: "matmul",
                source: MATMUL_GLSL,
                workgroup_x: TILE_SIZE,
                workgroup_y: TILE_SIZE,
                workgroup_z: 1,
            },
            KernelDescriptor {
                name: "softmax",
                source: SOFTMAX_GLSL,
                workgroup_x: DEFAULT_WORKGROUP_SIZE,
                workgroup_y: 1,
                workgroup_z: 1,
            },
            KernelDescriptor {
                name: "rmsnorm",
                source: RMSNORM_GLSL,
                workgroup_x: DEFAULT_WORKGROUP_SIZE,
                workgroup_y: 1,
                workgroup_z: 1,
            },
            KernelDescriptor {
                name: "rope",
                source: ROPE_GLSL,
                workgroup_x: DEFAULT_WORKGROUP_SIZE,
                workgroup_y: 1,
                workgroup_z: 1,
            },
            KernelDescriptor {
                name: "attention",
                source: ATTENTION_GLSL,
                workgroup_x: DEFAULT_WORKGROUP_SIZE,
                workgroup_y: 1,
                workgroup_z: 1,
            },
            KernelDescriptor {
                name: "embedding",
                source: EMBEDDING_GLSL,
                workgroup_x: DEFAULT_WORKGROUP_SIZE,
                workgroup_y: 1,
                workgroup_z: 1,
            },
            KernelDescriptor {
                name: "silu",
                source: SILU_GLSL,
                workgroup_x: DEFAULT_WORKGROUP_SIZE,
                workgroup_y: 1,
                workgroup_z: 1,
            },
        ];
        Self { entries }
    }

    /// Look up a kernel by name.
    pub fn get(&self, name: &str) -> Option<&KernelDescriptor> {
        self.entries.iter().find(|e| e.name == name)
    }

    /// Return all registered kernel names.
    pub fn kernel_names(&self) -> Vec<&'static str> {
        self.entries.iter().map(|e| e.name).collect()
    }

    /// Number of registered kernels.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over all kernel descriptors.
    pub fn iter(&self) -> impl Iterator<Item = &KernelDescriptor> {
        self.entries.iter()
    }
}

impl Default for KernelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- existing matmul shader tests ---

    #[test]
    fn glsl_source_is_non_empty() {
        assert!(
            !MATMUL_GLSL.is_empty(),
            "matmul GLSL source should be embedded"
        );
    }

    #[test]
    fn glsl_source_is_valid_glsl450() {
        assert!(
            MATMUL_GLSL.contains("#version 450"),
            "shader must target GLSL 450"
        );
    }

    #[test]
    fn glsl_contains_workgroup_layout() {
        assert!(
            MATMUL_GLSL.contains("local_size_x = 16")
                && MATMUL_GLSL.contains("local_size_y = 16"),
            "shader must declare 16x16 workgroup size"
        );
    }

    #[test]
    fn glsl_contains_shared_memory() {
        assert!(
            MATMUL_GLSL.contains("shared float tileA")
                && MATMUL_GLSL.contains("shared float tileB"),
            "shader must use shared memory tiles"
        );
    }

    #[test]
    fn glsl_contains_push_constants() {
        assert!(
            MATMUL_GLSL.contains("push_constant") && MATMUL_GLSL.contains("uint M, N, K"),
            "shader must accept M, N, K as push constants"
        );
    }

    #[test]
    fn glsl_contains_barrier_sync() {
        assert!(
            MATMUL_GLSL.contains("barrier()"),
            "shader must synchronize with barrier()"
        );
    }

    #[test]
    fn tile_size_matches_shader() {
        let expected = format!("local_size_x = {TILE_SIZE}");
        assert!(
            MATMUL_GLSL.contains(&expected),
            "TILE_SIZE constant must match shader workgroup size"
        );
    }

    #[test]
    fn glsl_has_three_buffer_bindings() {
        assert!(MATMUL_GLSL.contains("binding = 0"));
        assert!(MATMUL_GLSL.contains("binding = 1"));
        assert!(MATMUL_GLSL.contains("binding = 2"));
    }

    // --- new shader source tests ---

    #[test]
    fn all_shaders_target_glsl450() {
        let shaders = [
            ("softmax", SOFTMAX_GLSL),
            ("rmsnorm", RMSNORM_GLSL),
            ("rope", ROPE_GLSL),
            ("attention", ATTENTION_GLSL),
            ("embedding", EMBEDDING_GLSL),
            ("silu", SILU_GLSL),
        ];
        for (name, src) in shaders {
            assert!(
                src.contains("#version 450"),
                "{name} shader must target GLSL 450"
            );
        }
    }

    #[test]
    fn all_shaders_have_workgroup_layout() {
        let shaders = [
            ("softmax", SOFTMAX_GLSL),
            ("rmsnorm", RMSNORM_GLSL),
            ("rope", ROPE_GLSL),
            ("attention", ATTENTION_GLSL),
            ("embedding", EMBEDDING_GLSL),
            ("silu", SILU_GLSL),
        ];
        for (name, src) in shaders {
            assert!(
                src.contains("local_size_x = 256"),
                "{name} shader must declare workgroup size 256"
            );
        }
    }

    #[test]
    fn softmax_uses_shared_memory_reduction() {
        assert!(SOFTMAX_GLSL.contains("shared float"));
        assert!(SOFTMAX_GLSL.contains("barrier()"));
        assert!(SOFTMAX_GLSL.contains("push_constant"));
    }

    #[test]
    fn rmsnorm_uses_shared_memory_reduction() {
        assert!(RMSNORM_GLSL.contains("shared float shared_sq_sum"));
        assert!(RMSNORM_GLSL.contains("barrier()"));
        assert!(RMSNORM_GLSL.contains("eps"));
    }

    #[test]
    fn rope_has_frequency_buffers() {
        assert!(ROPE_GLSL.contains("FreqCos"));
        assert!(ROPE_GLSL.contains("FreqSin"));
        assert!(ROPE_GLSL.contains("head_dim"));
    }

    #[test]
    fn attention_has_qkv_bindings() {
        assert!(ATTENTION_GLSL.contains("Query"));
        assert!(ATTENTION_GLSL.contains("Key"));
        assert!(ATTENTION_GLSL.contains("Value"));
        assert!(ATTENTION_GLSL.contains("sqrt"));
    }

    #[test]
    fn embedding_has_lookup_logic() {
        assert!(EMBEDDING_GLSL.contains("token_ids"));
        assert!(EMBEDDING_GLSL.contains("emb_table"));
        assert!(EMBEDDING_GLSL.contains("embed_dim"));
    }

    #[test]
    fn silu_has_activation_logic() {
        assert!(SILU_GLSL.contains("exp(-x)"));
        assert!(SILU_GLSL.contains("fused_gate"));
    }

    // --- kernel registry tests ---

    #[test]
    fn registry_has_all_kernels() {
        let reg = KernelRegistry::new();
        assert_eq!(reg.len(), 7);
        let names = reg.kernel_names();
        assert!(names.contains(&"matmul"));
        assert!(names.contains(&"softmax"));
        assert!(names.contains(&"rmsnorm"));
        assert!(names.contains(&"rope"));
        assert!(names.contains(&"attention"));
        assert!(names.contains(&"embedding"));
        assert!(names.contains(&"silu"));
    }

    #[test]
    fn registry_lookup_returns_correct_source() {
        let reg = KernelRegistry::new();
        let desc = reg.get("softmax").expect("softmax must exist");
        assert_eq!(desc.source, SOFTMAX_GLSL);
        assert_eq!(desc.workgroup_x, DEFAULT_WORKGROUP_SIZE);
    }

    #[test]
    fn registry_lookup_missing_returns_none() {
        let reg = KernelRegistry::new();
        assert!(reg.get("nonexistent").is_none());
    }

    #[test]
    fn registry_matmul_has_2d_workgroup() {
        let reg = KernelRegistry::new();
        let desc = reg.get("matmul").expect("matmul must exist");
        assert_eq!(desc.workgroup_x, TILE_SIZE);
        assert_eq!(desc.workgroup_y, TILE_SIZE);
    }

    #[test]
    fn registry_is_not_empty() {
        let reg = KernelRegistry::default();
        assert!(!reg.is_empty());
    }

    #[test]
    fn registry_iter_yields_all_entries() {
        let reg = KernelRegistry::new();
        let count = reg.iter().count();
        assert_eq!(count, 7);
    }

    #[test]
    fn all_registry_sources_are_non_empty() {
        let reg = KernelRegistry::new();
        for desc in reg.iter() {
            assert!(
                !desc.source.is_empty(),
                "kernel '{}' has empty source",
                desc.name
            );
        }
    }
}

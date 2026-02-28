//! Vulkan compute kernel sources.
//!
//! Embeds GLSL shader sources as compile-time constants. A pre-compiled
//! SPIR-V binary is used when available (built by `build.rs` with `glslc`),
//! otherwise the raw GLSL source is available for runtime compilation.

/// GLSL source for the tiled matrix multiply compute shader.
pub const MATMUL_GLSL: &str = include_str!("matmul.glsl");

/// Pre-compiled SPIR-V binary for the matmul shader (if available).
///
/// Built by `build.rs` when `glslc` is found on PATH. When `None`, the
/// consumer must compile `MATMUL_GLSL` at runtime via a GLSL-to-SPIR-V
/// compiler (e.g. `shaderc`).
#[cfg(feature = "precompiled-spirv")]
pub const MATMUL_SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/matmul.spv"));

/// Tile size used by the matmul shader (workgroup dimensions).
pub const TILE_SIZE: u32 = 16;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glsl_source_is_non_empty() {
        assert!(!MATMUL_GLSL.is_empty(), "matmul GLSL source should be embedded");
    }

    #[test]
    fn glsl_source_is_valid_glsl450() {
        assert!(MATMUL_GLSL.contains("#version 450"), "shader must target GLSL 450");
    }

    #[test]
    fn glsl_contains_workgroup_layout() {
        assert!(
            MATMUL_GLSL.contains("local_size_x = 16") && MATMUL_GLSL.contains("local_size_y = 16"),
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
        assert!(MATMUL_GLSL.contains("barrier()"), "shader must synchronize with barrier()");
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
}

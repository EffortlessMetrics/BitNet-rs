//! Structural validation tests for Vulkan GLSL compute shaders.

use bitnet_vulkan::kernels::VulkanShaderSource;

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn all_shaders() -> &'static [VulkanShaderSource] {
    VulkanShaderSource::ALL
}

// ---------------------------------------------------------------------------
// Basic source validity
// ---------------------------------------------------------------------------

#[test]
fn all_sources_non_empty() {
    for shader in all_shaders() {
        assert!(!shader.glsl_source().is_empty(), "{} shader source is empty", shader.name());
    }
}

#[test]
fn all_sources_valid_utf8() {
    // include_str! guarantees UTF-8, but verify we can round-trip.
    for shader in all_shaders() {
        let src = shader.glsl_source();
        assert!(
            std::str::from_utf8(src.as_bytes()).is_ok(),
            "{} source is not valid UTF-8",
            shader.name()
        );
    }
}

#[test]
fn all_sources_contain_version_450() {
    for shader in all_shaders() {
        assert!(
            shader.glsl_source().contains("#version 450"),
            "{} missing #version 450 directive",
            shader.name()
        );
    }
}

#[test]
fn all_entry_points_are_main() {
    for shader in all_shaders() {
        assert_eq!(shader.entry_point(), "main", "{} entry point is not \"main\"", shader.name());
    }
}

#[test]
fn all_sources_have_void_main() {
    for shader in all_shaders() {
        assert!(
            shader.glsl_source().contains("void main()"),
            "{} missing void main() definition",
            shader.name()
        );
    }
}

// ---------------------------------------------------------------------------
// Layout declarations
// ---------------------------------------------------------------------------

#[test]
fn all_sources_have_local_size() {
    for shader in all_shaders() {
        assert!(
            shader.glsl_source().contains("local_size_x"),
            "{} missing layout(local_size_x ...) declaration",
            shader.name()
        );
    }
}

#[test]
fn all_sources_have_buffer_bindings() {
    for shader in all_shaders() {
        assert!(
            shader.glsl_source().contains("binding"),
            "{} missing buffer binding declaration",
            shader.name()
        );
    }
}

#[test]
fn all_sources_have_push_constants_or_buffer() {
    for shader in all_shaders() {
        let src = shader.glsl_source();
        let has_push = src.contains("push_constant");
        let has_buffer = src.contains("buffer");
        assert!(
            has_push || has_buffer,
            "{} missing push_constant or buffer declaration",
            shader.name()
        );
    }
}

// ---------------------------------------------------------------------------
// Per-shader content checks
// ---------------------------------------------------------------------------

#[test]
fn matmul_has_shared_memory() {
    let src = VulkanShaderSource::Matmul.glsl_source();
    assert!(src.contains("shared float"), "matmul missing shared memory");
}

#[test]
fn matmul_has_barrier() {
    let src = VulkanShaderSource::Matmul.glsl_source();
    assert!(src.contains("barrier()"), "matmul missing barrier()");
}

#[test]
fn matmul_has_tile_size() {
    let src = VulkanShaderSource::Matmul.glsl_source();
    assert!(src.contains("TILE_SIZE"), "matmul missing TILE_SIZE constant");
}

#[test]
fn softmax_has_subgroup_extension() {
    let src = VulkanShaderSource::Softmax.glsl_source();
    assert!(
        src.contains("GL_KHR_shader_subgroup_arithmetic"),
        "softmax missing subgroup extension"
    );
}

#[test]
fn softmax_has_exp() {
    let src = VulkanShaderSource::Softmax.glsl_source();
    assert!(src.contains("exp("), "softmax missing exp() call");
}

#[test]
fn rmsnorm_has_sqrt() {
    let src = VulkanShaderSource::RmsNorm.glsl_source();
    assert!(src.contains("sqrt("), "rmsnorm missing sqrt()");
}

#[test]
fn rmsnorm_has_epsilon() {
    let src = VulkanShaderSource::RmsNorm.glsl_source();
    assert!(src.contains("eps"), "rmsnorm missing epsilon parameter");
}

#[test]
fn rope_has_cos_sin() {
    let src = VulkanShaderSource::Rope.glsl_source();
    assert!(src.contains("cos("), "rope missing cos()");
    assert!(src.contains("sin("), "rope missing sin()");
}

#[test]
fn rope_has_theta() {
    let src = VulkanShaderSource::Rope.glsl_source();
    assert!(src.contains("theta"), "rope missing theta parameter");
}

#[test]
fn attention_has_scale() {
    let src = VulkanShaderSource::Attention.glsl_source();
    assert!(src.contains("scale"), "attention missing scale parameter");
}

#[test]
fn attention_has_qkv_buffers() {
    let src = VulkanShaderSource::Attention.glsl_source();
    assert!(src.contains("Query"), "attention missing Query buffer");
    assert!(src.contains("Key"), "attention missing Key buffer");
    assert!(src.contains("Value"), "attention missing Value buffer");
}

#[test]
fn elementwise_has_silu() {
    let src = VulkanShaderSource::Elementwise.glsl_source();
    assert!(src.contains("silu"), "elementwise missing SiLU");
}

#[test]
fn elementwise_has_gelu() {
    let src = VulkanShaderSource::Elementwise.glsl_source();
    assert!(src.contains("gelu"), "elementwise missing GELU");
}

#[test]
fn elementwise_has_binary_ops() {
    let src = VulkanShaderSource::Elementwise.glsl_source();
    assert!(src.contains("OP_ADD"), "elementwise missing add op");
    assert!(src.contains("OP_MUL"), "elementwise missing mul op");
}

// ---------------------------------------------------------------------------
// Enum coverage
// ---------------------------------------------------------------------------

#[test]
fn all_variant_count_is_six() {
    assert_eq!(VulkanShaderSource::ALL.len(), 6);
}

#[test]
fn all_names_unique() {
    let names: Vec<&str> = all_shaders().iter().map(|s| s.name()).collect();
    let mut deduped = names.clone();
    deduped.sort();
    deduped.dedup();
    assert_eq!(names.len(), deduped.len(), "duplicate shader names");
}

#[test]
fn shader_debug_impl() {
    // Ensure Debug is derived and doesn't panic.
    let dbg = format!("{:?}", VulkanShaderSource::Matmul);
    assert!(!dbg.is_empty());
}

#[test]
fn shader_clone_eq() {
    let a = VulkanShaderSource::Softmax;
    let b = a;
    assert_eq!(a, b);
}

#[test]
fn push_constants_present_in_all() {
    for shader in all_shaders() {
        assert!(
            shader.glsl_source().contains("push_constant"),
            "{} missing push_constant uniform block",
            shader.name()
        );
    }
}

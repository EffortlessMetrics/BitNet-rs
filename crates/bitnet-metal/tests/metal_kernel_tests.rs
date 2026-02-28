use bitnet_metal::{MetalKernelSource, kernel_function_names, kernel_source};

// ── Source embedding tests ──────────────────────────────────────────

#[test]
fn all_kernel_sources_non_empty() {
    for kernel in MetalKernelSource::ALL {
        let src = kernel_source(*kernel);
        assert!(!src.is_empty(), "Kernel source for {kernel:?} should not be empty");
    }
}

#[test]
fn all_kernel_sources_valid_utf8() {
    for kernel in MetalKernelSource::ALL {
        let src = kernel_source(*kernel);
        // include_str! guarantees UTF-8, but verify no null bytes
        assert!(!src.contains('\0'), "Kernel source for {kernel:?} contains null bytes");
    }
}

#[test]
fn all_kernels_include_metal_stdlib() {
    for kernel in MetalKernelSource::ALL {
        let src = kernel_source(*kernel);
        assert!(
            src.contains("#include <metal_stdlib>"),
            "Kernel {kernel:?} should include metal_stdlib"
        );
    }
}

#[test]
fn all_kernels_use_metal_namespace() {
    for kernel in MetalKernelSource::ALL {
        let src = kernel_source(*kernel);
        assert!(
            src.contains("using namespace metal;"),
            "Kernel {kernel:?} should use metal namespace"
        );
    }
}

// ── Kernel function name tests ──────────────────────────────────────

#[test]
fn all_kernel_functions_declared_in_source() {
    for kernel in MetalKernelSource::ALL {
        let src = kernel_source(*kernel);
        for name in kernel_function_names(*kernel) {
            let pattern = format!("kernel void {name}(");
            assert!(src.contains(&pattern), "Kernel {kernel:?} should contain function '{name}'");
        }
    }
}

#[test]
fn matmul_has_expected_functions() {
    let names = kernel_function_names(MetalKernelSource::Matmul);
    assert!(names.contains(&"matmul"));
    assert!(names.contains(&"matmul_tiled"));
}

#[test]
fn softmax_has_expected_function() {
    let names = kernel_function_names(MetalKernelSource::Softmax);
    assert!(names.contains(&"softmax"));
}

#[test]
fn rmsnorm_has_expected_function() {
    let names = kernel_function_names(MetalKernelSource::RmsNorm);
    assert!(names.contains(&"rmsnorm"));
}

#[test]
fn rope_has_expected_functions() {
    let names = kernel_function_names(MetalKernelSource::Rope);
    assert!(names.contains(&"rope"));
    assert!(names.contains(&"rope_build_tables"));
}

#[test]
fn attention_has_expected_functions() {
    let names = kernel_function_names(MetalKernelSource::Attention);
    assert!(names.contains(&"attention_scores"));
    assert!(names.contains(&"attention_weighted_sum"));
}

#[test]
fn elementwise_has_expected_functions() {
    let names = kernel_function_names(MetalKernelSource::Elementwise);
    assert!(names.contains(&"add"));
    assert!(names.contains(&"mul"));
    assert!(names.contains(&"silu"));
    assert!(names.contains(&"gelu"));
    assert!(names.contains(&"silu_mul"));
    assert!(names.contains(&"scalar_mul"));
}

// ── Buffer binding tests ────────────────────────────────────────────

#[test]
fn all_kernels_have_sequential_buffer_bindings() {
    for kernel in MetalKernelSource::ALL {
        let src = kernel_source(*kernel);
        for name in kernel_function_names(*kernel) {
            // Extract the function signature for this kernel
            let fn_start = format!("kernel void {name}(");
            let start = src.find(&fn_start).unwrap_or_else(|| {
                panic!("Function {name} not found in {kernel:?}");
            });
            // Find the closing paren of the parameter list
            let rest = &src[start..];
            let body_start = rest.find('{').unwrap_or(rest.len());
            let signature = &rest[..body_start];

            // Collect buffer indices
            let mut indices: Vec<u32> = Vec::new();
            for part in signature.split("[[buffer(") {
                if let Some(end) = part.find(")]]") {
                    if let Ok(idx) = part[..end].parse::<u32>() {
                        indices.push(idx);
                    }
                }
            }

            assert!(
                !indices.is_empty(),
                "Function {name} in {kernel:?} should have buffer bindings"
            );

            // Verify sequential starting from 0
            for (i, idx) in indices.iter().enumerate() {
                assert_eq!(
                    *idx, i as u32,
                    "Buffer binding {i} in {name} ({kernel:?}) should be \
                     {i}, got {idx}"
                );
            }
        }
    }
}

// ── Threadgroup / barrier tests ─────────────────────────────────────

#[test]
fn reduction_kernels_use_threadgroup_barrier() {
    let reduction_kernels = [MetalKernelSource::Softmax, MetalKernelSource::RmsNorm];
    for kernel in &reduction_kernels {
        let src = kernel_source(*kernel);
        assert!(
            src.contains("threadgroup_barrier"),
            "Reduction kernel {kernel:?} should use threadgroup_barrier"
        );
    }
}

#[test]
fn tiled_matmul_uses_threadgroup_memory() {
    let src = kernel_source(MetalKernelSource::Matmul);
    assert!(src.contains("threadgroup float"), "Tiled matmul should use threadgroup shared memory");
    assert!(src.contains("threadgroup_barrier"), "Tiled matmul should use threadgroup barriers");
}

#[test]
fn softmax_uses_threadgroup_reduction() {
    let src = kernel_source(MetalKernelSource::Softmax);
    assert!(src.contains("threadgroup float"));
    assert!(src.contains("threadgroup_barrier"));
    // Verify numerical stability (subtracts max)
    assert!(src.contains("row_max"), "Softmax should subtract max for numerical stability");
}

// ── Kernel-specific structure tests ─────────────────────────────────

#[test]
fn attention_supports_causal_mask() {
    let src = kernel_source(MetalKernelSource::Attention);
    assert!(src.contains("causal"), "Attention kernel should support causal masking");
    assert!(src.contains("-INFINITY"), "Causal mask should use -INFINITY for masked positions");
}

#[test]
fn rope_applies_rotation() {
    let src = kernel_source(MetalKernelSource::Rope);
    assert!(
        src.contains("cos_val") && src.contains("sin_val"),
        "RoPE should apply rotation with cos/sin"
    );
}

#[test]
fn elementwise_silu_uses_sigmoid() {
    let src = kernel_source(MetalKernelSource::Elementwise);
    // SiLU = x * sigmoid(x) = x / (1 + exp(-x))
    assert!(
        src.contains("exp(-x)") || src.contains("exp(- x)"),
        "SiLU should use sigmoid (exp(-x))"
    );
}

#[test]
fn all_enum_variants_covered() {
    assert_eq!(MetalKernelSource::ALL.len(), 6, "Should have exactly 6 kernel sources");
}

// ── Metal-on-device test (requires macOS with Metal) ────────────────

#[test]
#[ignore = "requires macOS with Metal GPU - run manually on Apple Silicon"]
fn compile_all_kernels_on_metal_device() {
    // This test would compile each MSL source via the Metal framework.
    // Requires metal-rs or objc bindings on macOS.
    for kernel in MetalKernelSource::ALL {
        let _src = kernel_source(*kernel);
        // TODO: MTLDevice::newLibraryWithSource(src, ...)
    }
}

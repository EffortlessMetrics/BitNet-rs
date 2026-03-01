//! Tests for WGSL kernel source structure and embedding.

use bitnet_webgpu::{
    ATTENTION_WGSL, ELEMENTWISE_WGSL, MATMUL_WGSL, RMSNORM_WGSL, ROPE_WGSL, SOFTMAX_WGSL,
    WgslKernelSource, kernel_source, validate_wgsl_structure,
};

// ---------------------------------------------------------------------------
// UTF-8 and non-empty checks
// ---------------------------------------------------------------------------

#[test]
fn all_kernel_sources_are_valid_utf8_and_non_empty() {
    for kernel in WgslKernelSource::all() {
        let src = kernel_source(*kernel);
        assert!(!src.is_empty(), "kernel {:?} has empty source", kernel.name());
        // `include_str!` guarantees UTF-8, but verify the string is
        // meaningful (not just whitespace).
        assert!(
            src.trim().len() > 10,
            "kernel {:?} source is too short to be valid",
            kernel.name()
        );
    }
}

// ---------------------------------------------------------------------------
// @compute entry point checks
// ---------------------------------------------------------------------------

#[test]
fn matmul_contains_compute_entry_point() {
    assert!(MATMUL_WGSL.contains("@compute"), "matmul.wgsl missing @compute");
}

#[test]
fn softmax_contains_compute_entry_point() {
    assert!(SOFTMAX_WGSL.contains("@compute"), "softmax.wgsl missing @compute");
}

#[test]
fn rmsnorm_contains_compute_entry_point() {
    assert!(RMSNORM_WGSL.contains("@compute"), "rmsnorm.wgsl missing @compute");
}

#[test]
fn rope_contains_compute_entry_point() {
    assert!(ROPE_WGSL.contains("@compute"), "rope.wgsl missing @compute");
}

#[test]
fn attention_contains_compute_entry_point() {
    assert!(ATTENTION_WGSL.contains("@compute"), "attention.wgsl missing @compute");
}

#[test]
fn elementwise_contains_compute_entry_point() {
    assert!(ELEMENTWISE_WGSL.contains("@compute"), "elementwise.wgsl missing @compute");
}

// ---------------------------------------------------------------------------
// Workgroup size checks
// ---------------------------------------------------------------------------

#[test]
fn all_kernels_have_workgroup_size() {
    for kernel in WgslKernelSource::all() {
        let src = kernel_source(*kernel);
        assert!(
            src.contains("@workgroup_size"),
            "kernel {:?} missing @workgroup_size",
            kernel.name()
        );
    }
}

// ---------------------------------------------------------------------------
// Binding index checks (sequential starting from 0)
// ---------------------------------------------------------------------------

#[test]
fn matmul_has_sequential_bindings() {
    assert_sequential_bindings(MATMUL_WGSL, "matmul");
}

#[test]
fn softmax_has_sequential_bindings() {
    assert_sequential_bindings(SOFTMAX_WGSL, "softmax");
}

#[test]
fn rmsnorm_has_sequential_bindings() {
    assert_sequential_bindings(RMSNORM_WGSL, "rmsnorm");
}

#[test]
fn rope_has_sequential_bindings() {
    assert_sequential_bindings(ROPE_WGSL, "rope");
}

#[test]
fn attention_has_sequential_bindings() {
    assert_sequential_bindings(ATTENTION_WGSL, "attention");
}

#[test]
fn elementwise_has_sequential_bindings() {
    assert_sequential_bindings(ELEMENTWISE_WGSL, "elementwise");
}

/// Helper: assert that `@binding(N)` indices are 0..max without gaps.
fn assert_sequential_bindings(source: &str, name: &str) {
    let mut indices: Vec<u32> = Vec::new();
    for segment in source.split("@binding(") {
        if let Some(end) = segment.find(')') {
            if let Ok(n) = segment[..end].parse::<u32>() {
                if !indices.contains(&n) {
                    indices.push(n);
                }
            }
        }
    }
    indices.sort_unstable();
    assert!(!indices.is_empty(), "{name}: no @binding declarations found");
    let expected: Vec<u32> = (0..indices.len() as u32).collect();
    assert_eq!(indices, expected, "{name}: binding indices are not sequential from 0: {indices:?}");
}

// ---------------------------------------------------------------------------
// No unresolved includes/imports
// ---------------------------------------------------------------------------

#[test]
fn no_kernel_contains_unresolved_includes() {
    for kernel in WgslKernelSource::all() {
        let src = kernel_source(*kernel);
        assert!(!src.contains("#include"), "kernel {:?} has unresolved #include", kernel.name());
        assert!(!src.contains("#import"), "kernel {:?} has unresolved #import", kernel.name());
    }
}

// ---------------------------------------------------------------------------
// Structural validation function
// ---------------------------------------------------------------------------

#[test]
fn validate_wgsl_structure_passes_for_all_kernels() {
    for kernel in WgslKernelSource::all() {
        let src = kernel_source(*kernel);
        let issues = validate_wgsl_structure(src);
        assert!(issues.is_empty(), "kernel {:?} has structural issues: {issues:?}", kernel.name());
    }
}

#[test]
fn validate_wgsl_structure_rejects_empty_source() {
    let issues = validate_wgsl_structure("");
    assert!(issues.iter().any(|i| i.contains("empty")), "expected 'empty' issue for empty source");
}

#[test]
fn validate_wgsl_structure_rejects_missing_compute() {
    let issues = validate_wgsl_structure("fn main() {}");
    assert!(issues.iter().any(|i| i.contains("@compute")), "expected missing @compute issue");
}

// ---------------------------------------------------------------------------
// Kernel embedding correctness
// ---------------------------------------------------------------------------

#[test]
fn kernel_source_returns_same_as_constants() {
    assert_eq!(kernel_source(WgslKernelSource::Matmul), MATMUL_WGSL);
    assert_eq!(kernel_source(WgslKernelSource::Softmax), SOFTMAX_WGSL);
    assert_eq!(kernel_source(WgslKernelSource::RmsNorm), RMSNORM_WGSL);
    assert_eq!(kernel_source(WgslKernelSource::Rope), ROPE_WGSL);
    assert_eq!(kernel_source(WgslKernelSource::Attention), ATTENTION_WGSL);
    assert_eq!(kernel_source(WgslKernelSource::Elementwise), ELEMENTWISE_WGSL);
}

#[test]
fn all_enum_variants_covered_by_all() {
    assert_eq!(WgslKernelSource::all().len(), 6, "expected 6 kernel variants");
}

// ---------------------------------------------------------------------------
// Elementwise has multiple entry points
// ---------------------------------------------------------------------------

#[test]
fn elementwise_has_all_expected_entry_points() {
    for name in &["fn add(", "fn mul(", "fn silu(", "fn gelu("] {
        assert!(ELEMENTWISE_WGSL.contains(name), "elementwise.wgsl missing entry point: {name}");
    }
}

// ---------------------------------------------------------------------------
// Balanced braces in all kernels
// ---------------------------------------------------------------------------

#[test]
fn all_kernels_have_balanced_braces() {
    for kernel in WgslKernelSource::all() {
        let src = kernel_source(*kernel);
        let open = src.chars().filter(|&c| c == '{').count();
        let close = src.chars().filter(|&c| c == '}').count();
        assert_eq!(open, close, "kernel {:?}: {open} '{{' vs {close} '}}'", kernel.name());
    }
}

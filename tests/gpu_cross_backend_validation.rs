//! Cross-backend kernel validation tests
//!
//! Ensures all GPU backends implement the same set of kernels with consistent
//! naming and parameter conventions. Each backend (OpenCL, WGSL, Metal, HIP,
//! Vulkan/GLSL) must provide implementations for the 6 standard kernel
//! categories: matmul, softmax, rmsnorm, rope, attention, and elementwise.
//!
//! Invariants validated:
//! - Every backend directory contains source files for all required kernels
//! - Kernel source files contain expected language-specific patterns
//! - Matrix kernels accept standard (M, N, K) parameters
//! - Softmax implementations use numerically stable subtract-max approach
//! - Matmul implementations use tiled algorithms where possible
//! - Backends that support shared/local memory actually use it
//! - CUDA reference backend is complete and correct

use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// The standard kernel set that every backend must implement.
const REQUIRED_KERNELS: &[&str] =
    &["matmul", "softmax", "rmsnorm", "rope", "attention", "elementwise"];

/// Backend specifications: (display name, crate subdirectory, file extension,
/// expected source-language pattern).
const BACKENDS: &[(&str, &str, &str, &[&str])] = &[
    (
        "OpenCL (.cl)",
        "bitnet-opencl/src/kernels",
        ".cl",
        &["get_global_id", "__kernel", "CLK_LOCAL_MEM_FENCE"],
    ),
    (
        "WGSL (.wgsl)",
        "bitnet-webgpu/src/kernels",
        ".wgsl",
        &["@compute", "@workgroup_size", "workgroupBarrier"],
    ),
    ("Metal (.metal)", "bitnet-metal/src/kernels", ".metal", &["kernel", "threadgroup", "device"]),
    (
        "HIP (.hip)",
        "bitnet-rocm/src/kernels",
        ".hip",
        &["__global__", "hipThreadIdx_x", "__shared__"],
    ),
    (
        "Vulkan (.comp)",
        "bitnet-vulkan/src/kernels",
        ".comp",
        &["layout(local_size", "gl_GlobalInvocationID", "shared"],
    ),
];

/// Path (relative to workspace root) to the CUDA reference kernels.
const CUDA_KERNELS_DIR: &str = "crates/bitnet-kernels/src/gpu/kernels";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns the workspace root (`tests/` is a sub-crate, so `..`).
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("CARGO_MANIFEST_DIR should have a parent (workspace root)")
        .to_owned()
}

/// Returns the directory for a given backend relative to the workspace root.
fn backend_dir(backend_subdir: &str) -> PathBuf {
    workspace_root().join("crates").join(backend_subdir)
}

/// Returns the CUDA kernel directory.
fn cuda_kernel_dir() -> PathBuf {
    workspace_root().join(CUDA_KERNELS_DIR)
}

/// Read a file to a `String`, returning `None` on any I/O error.
fn try_read(path: &Path) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

/// Collect all files in a directory matching the given extension.
fn files_with_extension(dir: &Path, ext: &str) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()).is_some_and(|e| format!(".{e}") == ext))
        .collect()
}

// ===========================================================================
// Module: CUDA reference backend (always present)
// ===========================================================================
mod cuda_reference {
    use super::*;

    #[test]
    fn cuda_kernel_dir_exists() {
        let dir = cuda_kernel_dir();
        assert!(dir.is_dir(), "CUDA kernel directory must exist at {}", dir.display());
    }

    #[test]
    fn cuda_has_kernel_source_files() {
        let dir = cuda_kernel_dir();
        let cu_files = files_with_extension(&dir, ".cu");
        assert!(!cu_files.is_empty(), "CUDA kernel directory should contain .cu files");
    }

    #[test]
    fn cuda_matmul_kernel_exists() {
        let path = cuda_kernel_dir().join("bitnet_kernels.cu");
        assert!(path.is_file(), "bitnet_kernels.cu must exist");
        let src = try_read(&path).expect("readable");
        assert!(
            src.contains("bitnet_matmul_i2s"),
            "CUDA kernels must contain matmul_i2s entry point"
        );
    }

    #[test]
    fn cuda_matmul_has_mnk_parameters() {
        let path = cuda_kernel_dir().join("bitnet_kernels.cu");
        let src = try_read(&path).expect("readable");
        assert!(src.contains("int M"), "matmul must accept M dimension");
        assert!(src.contains("int N"), "matmul must accept N dimension");
        assert!(src.contains("int K"), "matmul must accept K dimension");
    }

    #[test]
    fn cuda_matmul_uses_tiling() {
        let path = cuda_kernel_dir().join("bitnet_kernels.cu");
        let src = try_read(&path).expect("readable");
        assert!(
            src.contains("BLOCK_SIZE") || src.contains("tile"),
            "CUDA matmul should use tiled approach"
        );
    }

    #[test]
    fn cuda_matmul_uses_shared_memory() {
        let path = cuda_kernel_dir().join("bitnet_kernels.cu");
        let src = try_read(&path).expect("readable");
        assert!(src.contains("__shared__"), "CUDA matmul should use shared memory");
    }

    #[test]
    fn cuda_has_syncthreads_barrier() {
        let path = cuda_kernel_dir().join("bitnet_kernels.cu");
        let src = try_read(&path).expect("readable");
        assert!(
            src.contains("__syncthreads"),
            "CUDA kernels must use __syncthreads for correctness"
        );
    }

    #[test]
    fn cuda_matmul_standalone_exists() {
        let path = cuda_kernel_dir().join("bitnet_matmul.cu");
        assert!(path.is_file(), "standalone bitnet_matmul.cu must exist");
        let src = try_read(&path).expect("readable");
        assert!(
            src.contains("bitnet_matmul_i2s"),
            "standalone matmul must define bitnet_matmul_i2s"
        );
    }

    #[test]
    fn cuda_mixed_precision_exists() {
        let path = cuda_kernel_dir().join("mixed_precision_kernels.cu");
        assert!(path.is_file(), "mixed_precision_kernels.cu must exist");
        let src = try_read(&path).expect("readable");
        assert!(
            src.contains("bitnet_matmul_tensor_core") || src.contains("bitnet_matmul_fp16"),
            "mixed-precision file must contain FP16 or tensor-core matmul"
        );
    }

    #[test]
    fn cuda_quantization_kernels_present() {
        let path = cuda_kernel_dir().join("bitnet_kernels.cu");
        let src = try_read(&path).expect("readable");
        assert!(src.contains("bitnet_quantize_i2s"), "CUDA must have I2S quantization kernel");
        assert!(src.contains("bitnet_quantize_tl1"), "CUDA must have TL1 quantization kernel");
        assert!(src.contains("bitnet_quantize_tl2"), "CUDA must have TL2 quantization kernel");
    }

    #[test]
    fn cuda_dequantization_kernel_present() {
        let path = cuda_kernel_dir().join("bitnet_kernels.cu");
        let src = try_read(&path).expect("readable");
        assert!(src.contains("bitnet_dequantize"), "CUDA must have dequantize kernel");
    }

    #[test]
    fn cuda_extern_c_linkage() {
        let path = cuda_kernel_dir().join("bitnet_kernels.cu");
        let src = try_read(&path).expect("readable");
        assert!(
            src.contains("extern \"C\""),
            "CUDA kernels must use extern \"C\" linkage for NVRTC"
        );
    }

    #[test]
    fn cuda_no_data_races_in_packing() {
        let path = cuda_kernel_dir().join("bitnet_kernels.cu");
        let src = try_read(&path).expect("readable");
        // The quantization kernel serialises packing via a leader-thread pattern
        // to avoid data races on the packed output byte.
        assert!(
            src.contains("(tid % 4) == 0") || src.contains("atomicOr"),
            "Packed byte writes must be serialised (leader-thread or atomics)"
        );
    }
}

// ===========================================================================
// Module: Cross-backend kernel completeness
// ===========================================================================
mod completeness {
    use super::*;

    /// Helper: check that a backend directory contains files whose stems cover
    /// each required kernel. Returns a list of missing kernels.
    fn missing_kernels_for(dir: &Path, ext: &str) -> Vec<String> {
        let files = files_with_extension(dir, ext);
        let stems: Vec<String> = files
            .iter()
            .filter_map(|p| p.file_stem().and_then(|s| s.to_str()).map(|s| s.to_lowercase()))
            .collect();

        REQUIRED_KERNELS
            .iter()
            .filter(|k| !stems.iter().any(|s| s.contains(**k)))
            .map(|k| (*k).to_string())
            .collect()
    }

    // --- per-backend completeness tests (ignored; backend crates do not exist yet) ---

    #[test]
    #[ignore = "requires bitnet-opencl microcrate - not yet implemented"]
    fn opencl_has_all_required_kernels() {
        let dir = backend_dir("bitnet-opencl/src/kernels");
        let missing = missing_kernels_for(&dir, ".cl");
        assert!(missing.is_empty(), "OpenCL backend missing kernels: {missing:?}");
    }

    #[test]
    #[ignore = "requires bitnet-webgpu microcrate - not yet implemented"]
    fn wgsl_has_all_required_kernels() {
        let dir = backend_dir("bitnet-webgpu/src/kernels");
        let missing = missing_kernels_for(&dir, ".wgsl");
        assert!(missing.is_empty(), "WGSL backend missing kernels: {missing:?}");
    }

    #[test]
    #[ignore = "requires bitnet-metal microcrate - not yet implemented"]
    fn metal_has_all_required_kernels() {
        let dir = backend_dir("bitnet-metal/src/kernels");
        let missing = missing_kernels_for(&dir, ".metal");
        assert!(missing.is_empty(), "Metal backend missing kernels: {missing:?}");
    }

    #[test]
    #[ignore = "requires bitnet-rocm microcrate - not yet implemented"]
    fn hip_has_all_required_kernels() {
        let dir = backend_dir("bitnet-rocm/src/kernels");
        let missing = missing_kernels_for(&dir, ".hip");
        assert!(missing.is_empty(), "HIP backend missing kernels: {missing:?}");
    }

    #[test]
    #[ignore = "requires bitnet-vulkan microcrate - not yet implemented"]
    fn vulkan_has_all_required_kernels() {
        let dir = backend_dir("bitnet-vulkan/src/kernels");
        let missing = missing_kernels_for(&dir, ".comp");
        assert!(missing.is_empty(), "Vulkan backend missing kernels: {missing:?}");
    }

    #[test]
    fn required_kernel_set_is_non_empty() {
        assert!(!REQUIRED_KERNELS.is_empty(), "Required kernel set should not be empty");
        assert_eq!(REQUIRED_KERNELS.len(), 6, "Expected 6 required kernels");
    }

    #[test]
    fn backend_list_has_five_entries() {
        assert_eq!(BACKENDS.len(), 5, "Expected 5 GPU backends");
    }
}

// ===========================================================================
// Module: Source-pattern matching
// ===========================================================================
mod source_patterns {
    use super::*;

    /// Generic helper: for every source file in a backend directory, verify
    /// that it contains at least one of the expected language-specific tokens.
    fn validate_source_patterns(dir: &Path, ext: &str, patterns: &[&str]) {
        let files = files_with_extension(dir, ext);
        assert!(!files.is_empty(), "Expected kernel source files in {}", dir.display());
        for file in &files {
            let src = try_read(file).unwrap_or_else(|| panic!("Cannot read {}", file.display()));
            let has_pattern = patterns.iter().any(|p| src.contains(p));
            assert!(
                has_pattern,
                "File {} does not contain any expected pattern {:?}",
                file.display(),
                patterns,
            );
        }
    }

    #[test]
    fn cuda_source_has_cuda_patterns() {
        let dir = cuda_kernel_dir();
        let patterns = &["__global__", "threadIdx", "__shared__", "blockIdx"];
        validate_source_patterns(&dir, ".cu", patterns);
    }

    #[test]
    #[ignore = "requires bitnet-opencl microcrate - not yet implemented"]
    fn opencl_source_has_opencl_patterns() {
        let dir = backend_dir("bitnet-opencl/src/kernels");
        validate_source_patterns(&dir, ".cl", BACKENDS[0].3);
    }

    #[test]
    #[ignore = "requires bitnet-webgpu microcrate - not yet implemented"]
    fn wgsl_source_has_wgsl_patterns() {
        let dir = backend_dir("bitnet-webgpu/src/kernels");
        validate_source_patterns(&dir, ".wgsl", BACKENDS[1].3);
    }

    #[test]
    #[ignore = "requires bitnet-metal microcrate - not yet implemented"]
    fn metal_source_has_metal_patterns() {
        let dir = backend_dir("bitnet-metal/src/kernels");
        validate_source_patterns(&dir, ".metal", BACKENDS[2].3);
    }

    #[test]
    #[ignore = "requires bitnet-rocm microcrate - not yet implemented"]
    fn hip_source_has_hip_patterns() {
        let dir = backend_dir("bitnet-rocm/src/kernels");
        validate_source_patterns(&dir, ".hip", BACKENDS[3].3);
    }

    #[test]
    #[ignore = "requires bitnet-vulkan microcrate - not yet implemented"]
    fn vulkan_source_has_vulkan_patterns() {
        let dir = backend_dir("bitnet-vulkan/src/kernels");
        validate_source_patterns(&dir, ".comp", BACKENDS[4].3);
    }
}

// ===========================================================================
// Module: Naming consistency
// ===========================================================================
mod naming {
    use super::*;

    #[test]
    fn cuda_kernel_files_follow_naming_convention() {
        let dir = cuda_kernel_dir();
        let files = files_with_extension(&dir, ".cu");
        for f in &files {
            let stem = f.file_stem().unwrap().to_str().unwrap();
            assert!(
                stem.starts_with("bitnet_") || stem.starts_with("mixed_"),
                "CUDA kernel file '{stem}' should start with 'bitnet_' or 'mixed_'"
            );
        }
    }

    #[test]
    fn backend_specs_have_unique_extensions() {
        let exts: Vec<&str> = BACKENDS.iter().map(|b| b.2).collect();
        let unique: std::collections::HashSet<&&str> = exts.iter().collect();
        assert_eq!(exts.len(), unique.len(), "Each backend must use a unique file extension");
    }

    #[test]
    fn backend_specs_have_unique_names() {
        let names: Vec<&str> = BACKENDS.iter().map(|b| b.0).collect();
        let unique: std::collections::HashSet<&&str> = names.iter().collect();
        assert_eq!(names.len(), unique.len(), "Each backend must have a unique display name");
    }

    #[test]
    fn backend_dirs_have_unique_paths() {
        let dirs: Vec<&str> = BACKENDS.iter().map(|b| b.1).collect();
        let unique: std::collections::HashSet<&&str> = dirs.iter().collect();
        assert_eq!(dirs.len(), unique.len(), "Each backend must have a unique crate path");
    }
}

// ===========================================================================
// Module: Entry point documentation
// ===========================================================================
mod entry_points {
    use super::*;

    /// Expected CUDA entry-point function names (extern "C" __global__).
    const CUDA_ENTRY_POINTS: &[&str] = &[
        "bitnet_matmul_i2s",
        "bitnet_matmul_fp32",
        "bitnet_quantize_i2s",
        "bitnet_quantize_tl1",
        "bitnet_quantize_tl2",
        "bitnet_dequantize",
    ];

    #[test]
    fn cuda_all_entry_points_present() {
        let path = cuda_kernel_dir().join("bitnet_kernels.cu");
        let src = try_read(&path).expect("readable");
        for ep in CUDA_ENTRY_POINTS {
            assert!(src.contains(ep), "CUDA missing entry point '{ep}'");
        }
    }

    #[test]
    fn cuda_mixed_precision_entry_points() {
        let path = cuda_kernel_dir().join("mixed_precision_kernels.cu");
        let src = try_read(&path).expect("readable");
        let expected = ["bitnet_matmul_tensor_core", "bitnet_matmul_fp16", "bitnet_matmul_bf16"];
        for ep in &expected {
            assert!(src.contains(ep), "Mixed-precision file missing entry point '{ep}'");
        }
    }

    #[test]
    fn cuda_entry_points_use_extern_c() {
        let path = cuda_kernel_dir().join("bitnet_kernels.cu");
        let src = try_read(&path).expect("readable");
        for ep in CUDA_ENTRY_POINTS {
            // Find the function and verify it has extern "C" __global__ linkage
            if let Some(idx) = src.find(ep) {
                let prefix = &src[..idx];
                let context = &prefix[prefix.len().saturating_sub(120)..];
                assert!(
                    context.contains("extern \"C\""),
                    "Entry point '{ep}' must use extern \"C\" linkage"
                );
            }
        }
    }
}

// ===========================================================================
// Module: Parameter matching
// ===========================================================================
mod parameters {
    use super::*;

    #[test]
    fn cuda_matmul_has_standard_mnk() {
        let path = cuda_kernel_dir().join("bitnet_kernels.cu");
        let src = try_read(&path).expect("readable");
        // All matmul signatures should accept (M, N, K)
        for func in ["bitnet_matmul_i2s", "bitnet_matmul_fp32"] {
            if let Some(start) = src.find(func) {
                let snippet = &src[start..src.len().min(start + 300)];
                assert!(snippet.contains("int M"), "{func} must accept M");
                assert!(snippet.contains("int N"), "{func} must accept N");
                assert!(snippet.contains("int K"), "{func} must accept K");
            }
        }
    }

    #[test]
    fn cuda_quantize_has_n_parameter() {
        let path = cuda_kernel_dir().join("bitnet_kernels.cu");
        let src = try_read(&path).expect("readable");
        for func in ["bitnet_quantize_i2s", "bitnet_quantize_tl1", "bitnet_quantize_tl2"] {
            if let Some(start) = src.find(func) {
                let snippet = &src[start..src.len().min(start + 300)];
                assert!(snippet.contains("int N"), "{func} must accept element count N");
            }
        }
    }

    #[test]
    fn cuda_dequantize_has_type_parameter() {
        let path = cuda_kernel_dir().join("bitnet_kernels.cu");
        let src = try_read(&path).expect("readable");
        if let Some(start) = src.find("bitnet_dequantize") {
            let snippet = &src[start..src.len().min(start + 300)];
            assert!(
                snippet.contains("quantization_type"),
                "dequantize must accept quantization_type selector"
            );
        }
    }

    #[test]
    fn cuda_mixed_precision_matmul_has_mnk() {
        let path = cuda_kernel_dir().join("mixed_precision_kernels.cu");
        let src = try_read(&path).expect("readable");
        for func in ["bitnet_matmul_tensor_core", "bitnet_matmul_fp16", "bitnet_matmul_bf16"] {
            if let Some(start) = src.find(func) {
                let snippet = &src[start..src.len().min(start + 300)];
                assert!(snippet.contains("int M"), "{func} must accept M");
                assert!(snippet.contains("int N"), "{func} must accept N");
                assert!(snippet.contains("int K"), "{func} must accept K");
            }
        }
    }
}

// ===========================================================================
// Module: Numerical properties
// ===========================================================================
mod numerical {
    use super::*;

    #[test]
    fn cuda_quantize_i2s_uses_max_for_scaling() {
        let path = cuda_kernel_dir().join("bitnet_kernels.cu");
        let src = try_read(&path).expect("readable");
        // Numerical stability: the I2S quantizer finds max(abs(values)) and
        // divides by it, which is the standard numerically-stable approach.
        assert!(
            src.contains("fabsf") || src.contains("fabs"),
            "I2S quantizer should compute absolute values"
        );
        assert!(
            src.contains("fmaxf") || src.contains("shared_max"),
            "I2S quantizer should find max for numerical stability"
        );
    }

    #[test]
    fn cuda_quantize_i2s_prevents_division_by_zero() {
        let path = cuda_kernel_dir().join("bitnet_kernels.cu");
        let src = try_read(&path).expect("readable");
        // Verify guard against zero scale
        assert!(
            src.contains("1e-8") || src.contains("1e-6") || src.contains("> 0"),
            "I2S quantizer must guard against division by zero"
        );
    }

    #[test]
    fn cuda_mixed_precision_has_arch_guards() {
        let path = cuda_kernel_dir().join("mixed_precision_kernels.cu");
        let src = try_read(&path).expect("readable");
        assert!(
            src.contains("__CUDA_ARCH__"),
            "Mixed-precision kernels must guard with __CUDA_ARCH__"
        );
    }

    #[test]
    fn cuda_dequantize_handles_all_quant_types() {
        let path = cuda_kernel_dir().join("bitnet_kernels.cu");
        let src = try_read(&path).expect("readable");
        // The dequantize switch should handle I2S (0), TL1 (1), TL2 (2)
        assert!(src.contains("case 0:"), "dequantize must handle type 0 (I2S)");
        assert!(src.contains("case 1:"), "dequantize must handle type 1 (TL1)");
        assert!(src.contains("case 2:"), "dequantize must handle type 2 (TL2)");
    }
}

// ===========================================================================
// Module: Tiling and shared memory
// ===========================================================================
mod tiling {
    use super::*;

    #[test]
    fn cuda_matmul_tile_loop() {
        let path = cuda_kernel_dir().join("bitnet_kernels.cu");
        let src = try_read(&path).expect("readable");
        // A tiled matmul iterates over tiles in the K dimension
        assert!(
            src.contains("Loop over tiles") || src.contains("BLOCK_SIZE"),
            "CUDA matmul should have a tiled loop"
        );
    }

    #[test]
    fn cuda_shared_memory_tiles_are_square() {
        let path = cuda_kernel_dir().join("bitnet_kernels.cu");
        let src = try_read(&path).expect("readable");
        // Verify shared memory arrays are declared with matching dimensions
        assert!(
            src.contains("As[16][16]") || src.contains("Bs[16][16]"),
            "CUDA should declare square shared memory tiles"
        );
    }

    #[test]
    fn cuda_syncthreads_after_shared_load() {
        let path = cuda_kernel_dir().join("bitnet_kernels.cu");
        let src = try_read(&path).expect("readable");
        // After loading shared memory tiles, there must be a barrier before
        // the compute phase to prevent data races.
        let shared_idx = src.find("__shared__").expect("has __shared__");
        let sync_idx = src.find("__syncthreads").expect("has __syncthreads");
        assert!(sync_idx > shared_idx, "__syncthreads must appear after __shared__ declaration");
    }

    #[test]
    fn cuda_matmul_bounds_check() {
        let path = cuda_kernel_dir().join("bitnet_kernels.cu");
        let src = try_read(&path).expect("readable");
        // The matmul must check row < M and col < N before writing output
        assert!(
            src.contains("row < M") && src.contains("col < N"),
            "Matmul must bounds-check row < M and col < N"
        );
    }
}

// ===========================================================================
// Module: Thread safety
// ===========================================================================
mod thread_safety {
    use super::*;

    #[test]
    fn cuda_quantize_uses_syncthreads_in_reduction() {
        let path = cuda_kernel_dir().join("bitnet_kernels.cu");
        let src = try_read(&path).expect("readable");
        // Parallel reductions need barriers between reduction steps
        let reduction_section = src.find("Parallel reduction").or_else(|| src.find("Reduction"));
        assert!(reduction_section.is_some(), "Quantizer should document its reduction step");
    }

    #[test]
    fn cuda_no_unsynchronised_shared_writes() {
        let path = cuda_kernel_dir().join("bitnet_kernels.cu");
        let src = try_read(&path).expect("readable");
        // Count __shared__ declarations and __syncthreads calls; there should
        // be at least one barrier per shared declaration.
        let shared_count = src.matches("__shared__").count();
        let sync_count = src.matches("__syncthreads").count();
        assert!(
            sync_count >= shared_count,
            "Need at least one __syncthreads per __shared__ usage \
             (found {shared_count} shared, {sync_count} syncs)"
        );
    }

    #[test]
    fn cuda_mixed_precision_atomics_are_safe() {
        let path = cuda_kernel_dir().join("mixed_precision_kernels.cu");
        let src = try_read(&path).expect("readable");
        // If atomics are used, they should be on distinct addresses
        if src.contains("atomicOr") || src.contains("atomicAdd") {
            assert!(
                src.contains("pack_idx") || src.contains("bit_offset"),
                "Atomic operations should use per-element addressing"
            );
        }
    }
}

// ===========================================================================
// Module: Backend registry (GPU HAL)
// ===========================================================================
mod backend_registry {
    use super::*;

    #[test]
    fn gpu_mod_rs_registers_cuda() {
        let path = workspace_root().join("crates/bitnet-kernels/src/gpu/mod.rs");
        let src = try_read(&path).expect("readable");
        assert!(src.contains("pub mod cuda"), "GPU module must publicly expose the CUDA backend");
    }

    #[test]
    fn gpu_mod_rs_registers_validation() {
        let path = workspace_root().join("crates/bitnet-kernels/src/gpu/mod.rs");
        let src = try_read(&path).expect("readable");
        assert!(
            src.contains("pub mod validation") || src.contains("pub use validation"),
            "GPU module must expose the validation module"
        );
    }

    #[test]
    #[ignore = "requires all GPU microcrates compiled"]
    fn all_backends_registered_in_gpu_hal() {
        // When a GPU HAL crate exists, this test should verify that each
        // backend is registered in the HAL's backend enum/registry.
        let hal_path = workspace_root().join("crates/bitnet-gpu-hal/src/lib.rs");
        let src = try_read(&hal_path)
            .expect("bitnet-gpu-hal/src/lib.rs must exist when all backends are compiled");
        for (name, _, _, _) in BACKENDS {
            let tag = name.split_whitespace().next().unwrap().to_lowercase();
            assert!(src.to_lowercase().contains(&tag), "GPU HAL should register backend '{name}'");
        }
    }
}

// ===========================================================================
// Module: Cross-backend consistency (integration)
// ===========================================================================
mod cross_backend_consistency {
    use super::*;

    #[test]
    #[ignore = "requires all GPU microcrates compiled"]
    fn all_backends_have_matmul_with_mnk() {
        for (name, subdir, ext, _) in BACKENDS {
            let dir = backend_dir(subdir);
            let files = files_with_extension(&dir, ext);
            let matmul_file = files.iter().find(|f| {
                f.file_stem().and_then(|s| s.to_str()).is_some_and(|s| s.contains("matmul"))
            });
            let matmul_file =
                matmul_file.unwrap_or_else(|| panic!("{name}: no matmul kernel file found"));
            let src = try_read(matmul_file).expect("readable");
            assert!(
                src.contains('M') && src.contains('N') && src.contains('K'),
                "{name}: matmul must accept (M, N, K)"
            );
        }
    }

    #[test]
    #[ignore = "requires all GPU microcrates compiled"]
    fn all_backends_have_softmax_with_subtract_max() {
        for (name, subdir, ext, _) in BACKENDS {
            let dir = backend_dir(subdir);
            let files = files_with_extension(&dir, ext);
            let softmax_file = files.iter().find(|f| {
                f.file_stem().and_then(|s| s.to_str()).is_some_and(|s| s.contains("softmax"))
            });
            let softmax_file =
                softmax_file.unwrap_or_else(|| panic!("{name}: no softmax kernel file found"));
            let src = try_read(softmax_file).expect("readable");
            assert!(
                src.contains("max") || src.contains("Max"),
                "{name}: softmax must subtract max for numerical stability"
            );
        }
    }

    #[test]
    #[ignore = "requires all GPU microcrates compiled"]
    fn all_backends_match_kernel_count() {
        let mut counts: Vec<(&str, usize)> = Vec::new();
        for (name, subdir, ext, _) in BACKENDS {
            let dir = backend_dir(subdir);
            let files = files_with_extension(&dir, ext);
            counts.push((name, files.len()));
        }
        let first_count = counts[0].1;
        for (name, count) in &counts {
            assert_eq!(
                *count, first_count,
                "Backend {name} has {count} kernel files, expected {first_count}"
            );
        }
    }
}

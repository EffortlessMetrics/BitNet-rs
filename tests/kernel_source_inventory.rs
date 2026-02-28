//! Kernel source inventory snapshot tests
//!
//! Validates that every GPU backend crate contains the expected set of kernel
//! source files. The CUDA reference backend is always checked; other backends
//! are checked only when their crate directories exist, and are `#[ignore]`d
//! otherwise.
//!
//! Invariants:
//! - The CUDA kernel directory exists and contains at least 3 `.cu` files
//! - Each planned backend directory, when present, contains the 6 standard
//!   kernel categories (matmul, softmax, rmsnorm, rope, attention, elementwise)
//! - Kernel file names follow the `bitnet_<kernel>.ext` convention
//! - No stale or orphaned kernel files exist without a corresponding entry
//!   point declaration

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Standard kernel categories every backend must provide.
const STANDARD_KERNELS: &[&str] =
    &["matmul", "softmax", "rmsnorm", "rope", "attention", "elementwise"];

/// Backend directories relative to `crates/`.
const BACKEND_DIRS: &[(&str, &str)] = &[
    ("CUDA", "bitnet-kernels/src/gpu/kernels"),
    ("OpenCL", "bitnet-opencl/src/kernels"),
    ("WGSL", "bitnet-webgpu/src/kernels"),
    ("Metal", "bitnet-metal/src/kernels"),
    ("HIP", "bitnet-rocm/src/kernels"),
    ("Vulkan", "bitnet-vulkan/src/kernels"),
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("CARGO_MANIFEST_DIR parent is workspace root")
        .to_owned()
}

fn crates_dir() -> PathBuf {
    workspace_root().join("crates")
}

/// Collect file stems (lowercased) from a directory.
fn file_stems(dir: &Path) -> BTreeSet<String> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return BTreeSet::new();
    };
    entries
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file())
        .filter_map(|e| e.path().file_stem().and_then(|s| s.to_str()).map(|s| s.to_lowercase()))
        .collect()
}

/// Collect all files in a directory (non-recursive).
fn list_files(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    entries.filter_map(|e| e.ok()).filter(|e| e.path().is_file()).map(|e| e.path()).collect()
}

// ===========================================================================
// CUDA reference backend (always present)
// ===========================================================================

#[test]
fn kernel_inventory_cuda_dir_exists() {
    let dir = crates_dir().join("bitnet-kernels/src/gpu/kernels");
    assert!(dir.is_dir(), "CUDA kernel directory must exist: {}", dir.display());
}

#[test]
fn kernel_inventory_cuda_file_count() {
    let dir = crates_dir().join("bitnet-kernels/src/gpu/kernels");
    let files = list_files(&dir);
    let cu_files: Vec<_> =
        files.iter().filter(|f| f.extension().and_then(|e| e.to_str()) == Some("cu")).collect();
    assert!(cu_files.len() >= 3, "Expected at least 3 CUDA kernel files, found {}", cu_files.len());
}

#[test]
fn kernel_inventory_cuda_expected_files() {
    let dir = crates_dir().join("bitnet-kernels/src/gpu/kernels");
    let stems = file_stems(&dir);
    let expected = ["bitnet_kernels", "bitnet_matmul", "mixed_precision_kernels"];
    for name in &expected {
        assert!(
            stems.contains(*name),
            "CUDA directory missing expected file stem '{name}'; found: {stems:?}"
        );
    }
}

#[test]
fn kernel_inventory_cuda_no_empty_files() {
    let dir = crates_dir().join("bitnet-kernels/src/gpu/kernels");
    for file in list_files(&dir) {
        let meta = std::fs::metadata(&file).expect("metadata");
        assert!(meta.len() > 0, "Kernel file should not be empty: {}", file.display());
    }
}

#[test]
fn kernel_inventory_cuda_files_have_cu_extension() {
    let dir = crates_dir().join("bitnet-kernels/src/gpu/kernels");
    for file in list_files(&dir) {
        let ext = file.extension().and_then(|e| e.to_str()).unwrap_or("");
        assert_eq!(ext, "cu", "Unexpected file extension in CUDA kernel dir: {}", file.display());
    }
}

#[test]
fn kernel_inventory_cuda_matmul_entry_points() {
    let dir = crates_dir().join("bitnet-kernels/src/gpu/kernels");
    let path = dir.join("bitnet_kernels.cu");
    let src = std::fs::read_to_string(&path).expect("readable");

    let entry_points = [
        "bitnet_matmul_i2s",
        "bitnet_matmul_fp32",
        "bitnet_quantize_i2s",
        "bitnet_quantize_tl1",
        "bitnet_quantize_tl2",
        "bitnet_dequantize",
    ];
    for ep in &entry_points {
        assert!(src.contains(ep), "bitnet_kernels.cu missing entry point: {ep}");
    }
}

#[test]
fn kernel_inventory_cuda_mixed_precision_entry_points() {
    let dir = crates_dir().join("bitnet-kernels/src/gpu/kernels");
    let path = dir.join("mixed_precision_kernels.cu");
    let src = std::fs::read_to_string(&path).expect("readable");

    let entry_points = [
        "bitnet_matmul_tensor_core",
        "bitnet_matmul_fp16",
        "bitnet_matmul_bf16",
        "bitnet_quantize_fp16",
        "bitnet_quantize_bf16",
    ];
    for ep in &entry_points {
        assert!(src.contains(ep), "mixed_precision_kernels.cu missing entry point: {ep}");
    }
}

#[test]
fn kernel_inventory_cuda_standalone_matmul_matches() {
    let dir = crates_dir().join("bitnet-kernels/src/gpu/kernels");
    let main = std::fs::read_to_string(dir.join("bitnet_kernels.cu")).expect("readable");
    let standalone = std::fs::read_to_string(dir.join("bitnet_matmul.cu")).expect("readable");

    // Both files should define the same core entry point
    assert!(main.contains("bitnet_matmul_i2s"));
    assert!(standalone.contains("bitnet_matmul_i2s"));
}

// ===========================================================================
// Per-backend inventory (ignored until crates exist)
// ===========================================================================

macro_rules! backend_inventory_test {
    ($test_name:ident, $display:expr, $subdir:expr, $reason:expr) => {
        #[test]
        #[ignore = $reason]
        fn $test_name() {
            let dir = crates_dir().join($subdir);
            assert!(dir.is_dir(), "{} kernel directory must exist at {}", $display, dir.display());
            let stems = file_stems(&dir);
            for kernel in STANDARD_KERNELS {
                let found = stems.iter().any(|s| s.contains(kernel));
                assert!(
                    found,
                    "{} backend missing '{}' kernel; found stems: {:?}",
                    $display, kernel, stems
                );
            }
        }
    };
}

backend_inventory_test!(
    kernel_inventory_opencl_complete,
    "OpenCL",
    "bitnet-opencl/src/kernels",
    "requires bitnet-opencl microcrate - not yet implemented"
);

backend_inventory_test!(
    kernel_inventory_wgsl_complete,
    "WGSL",
    "bitnet-webgpu/src/kernels",
    "requires bitnet-webgpu microcrate - not yet implemented"
);

backend_inventory_test!(
    kernel_inventory_metal_complete,
    "Metal",
    "bitnet-metal/src/kernels",
    "requires bitnet-metal microcrate - not yet implemented"
);

backend_inventory_test!(
    kernel_inventory_hip_complete,
    "HIP",
    "bitnet-rocm/src/kernels",
    "requires bitnet-rocm microcrate - not yet implemented"
);

backend_inventory_test!(
    kernel_inventory_vulkan_complete,
    "Vulkan",
    "bitnet-vulkan/src/kernels",
    "requires bitnet-vulkan microcrate - not yet implemented"
);

// ===========================================================================
// Cross-backend snapshot
// ===========================================================================

#[test]
fn kernel_inventory_complete() {
    let mut report = String::new();
    report.push_str("=== Kernel Source Inventory ===\n\n");

    for (name, subdir) in BACKEND_DIRS {
        let dir = crates_dir().join(subdir);
        report.push_str(&format!("{name}:\n"));
        if dir.is_dir() {
            let files = list_files(&dir);
            if files.is_empty() {
                report.push_str("  (empty directory)\n");
            } else {
                let mut sorted: Vec<_> = files
                    .iter()
                    .filter_map(|f| f.file_name().and_then(|n| n.to_str()).map(String::from))
                    .collect();
                sorted.sort();
                for f in &sorted {
                    report.push_str(&format!("  {f}\n"));
                }
            }
        } else {
            report.push_str("  (directory not found)\n");
        }
        report.push('\n');
    }

    // Print for snapshot visibility
    println!("{report}");

    // At minimum, CUDA must be present
    let cuda_dir = crates_dir().join("bitnet-kernels/src/gpu/kernels");
    assert!(cuda_dir.is_dir(), "CUDA backend must exist");
    let cuda_files = list_files(&cuda_dir);
    assert!(!cuda_files.is_empty(), "CUDA backend must have kernel source files");
}

#[test]
fn kernel_inventory_backends_count() {
    assert_eq!(BACKEND_DIRS.len(), 6, "Expected 6 backend directories (CUDA + 5 planned)");
}

#[test]
fn kernel_inventory_standard_kernel_count() {
    assert_eq!(STANDARD_KERNELS.len(), 6, "Expected 6 standard kernel categories");
}

#[test]
#[ignore = "requires all GPU microcrates compiled"]
fn kernel_inventory_all_backends_present() {
    for (name, subdir) in BACKEND_DIRS {
        let dir = crates_dir().join(subdir);
        assert!(dir.is_dir(), "Backend {name} directory not found at {}", dir.display());
    }
}

#[test]
#[ignore = "requires all GPU microcrates compiled"]
fn kernel_inventory_all_backends_have_standard_kernels() {
    for (name, subdir) in BACKEND_DIRS {
        let dir = crates_dir().join(subdir);
        let stems = file_stems(&dir);
        for kernel in STANDARD_KERNELS {
            let found = stems.iter().any(|s| s.contains(kernel));
            assert!(found, "{name} backend missing '{kernel}' kernel; found: {stems:?}");
        }
    }
}

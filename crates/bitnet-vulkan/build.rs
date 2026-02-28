//! Build script for bitnet-vulkan.
//!
//! Optionally compiles GLSL shaders to SPIR-V when `glslc` is available.

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/kernels/matmul.glsl");

    // Only attempt SPIR-V compilation when the feature is enabled
    if std::env::var("CARGO_FEATURE_PRECOMPILED_SPIRV").is_ok() {
        compile_spirv();
    }
}

fn compile_spirv() {
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let glsl_path = "src/kernels/matmul.glsl";
    let spv_path = format!("{out_dir}/matmul.spv");

    match Command::new("glslc")
        .args(["-fshader-stage=compute", glsl_path, "-o", &spv_path])
        .status()
    {
        Ok(status) if status.success() => {
            println!("cargo:warning=Compiled matmul.glsl → matmul.spv");
        }
        Ok(status) => {
            panic!("glslc failed with status {status} (precompiled-spirv feature requires glslc)");
        }
        Err(e) => {
            panic!("glslc not found: {e} (precompiled-spirv feature requires glslc on PATH)");
        }
    }
}

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

#[path = "build_helpers/opencl_compile.rs"]
mod opencl_compile;

fn main() {
    // Minimal, dependency-free metadata so the build never blocks on 'vergen'
    let ts = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    println!("cargo:rustc-env=BITNET_BUILD_TS={ts}");

    // --- Optional SPIR-V pre-compilation for OpenCL kernels ---
    maybe_compile_spirv();
}

/// Attempt to pre-compile every `.cl` kernel file to `.spv` (SPIR-V).
///
/// If `clang` or `ocloc` is on `$PATH`, each `.cl` under
/// `crates/bitnet-kernels/src/gpu/kernels/` is compiled to SPIR-V and
/// the resulting `.spv` bytes are emitted as `BITNET_SPV_<NAME>` env vars
/// pointing to the output path.  The kernel provider checks these at
/// runtime to decide whether to use `clCreateProgramWithIL` (SPIR-V)
/// or `clCreateProgramWithSource` (source fallback).
fn maybe_compile_spirv() {
    let kernel_dir = Path::new("crates/bitnet-kernels/src/gpu/kernels");
    if !kernel_dir.is_dir() {
        return; // Not building from workspace root or kernels dir absent
    }

    let out_dir = match std::env::var("OUT_DIR") {
        Ok(d) => std::path::PathBuf::from(d),
        Err(_) => return,
    };

    let cl_files = ["matmul_i2s", "quantize_i2s", "elementwise", "attention"];

    for name in &cl_files {
        let cl_path = kernel_dir.join(format!("{name}.cl"));
        if !cl_path.exists() {
            continue;
        }

        // Rerun build.rs if the kernel source changes
        println!("cargo:rerun-if-changed={}", cl_path.display());

        let spv_path = out_dir.join(format!("{name}.spv"));

        match opencl_compile::compile_cl_to_spv(&cl_path, &spv_path) {
            opencl_compile::SpvCompileResult::Compiled(path) => {
                let env_key = format!("BITNET_SPV_{}", name.to_uppercase());
                println!("cargo:rustc-env={env_key}={}", path.display());
                println!("cargo:warning=SPIR-V compiled: {name}.cl → {name}.spv");
            }
            opencl_compile::SpvCompileResult::CompilerNotFound => {
                // Silent — runtime source compilation will be used
            }
            opencl_compile::SpvCompileResult::Failed(err) => {
                println!("cargo:warning=SPIR-V compilation failed for {name}.cl: {err}");
            }
        }
    }
}

//! Build-time OpenCL → SPIR-V compilation helper.
//!
//! Attempts to compile `.cl` kernel sources into `.spv` (SPIR-V) binaries
//! using either `clang` (LLVM/SPIR-V) or Intel `ocloc` (oneAPI offline compiler).
//! If neither tool is available the build silently falls back to runtime
//! source compilation.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Result of a SPIR-V compilation attempt.
#[derive(Debug)]
pub enum SpvCompileResult {
    /// Compiled successfully — `.spv` path is valid.
    Compiled(PathBuf),
    /// Compiler not found — fall back to runtime source compilation.
    CompilerNotFound,
    /// Compilation failed with an error message.
    Failed(String),
}

/// Try to locate `clang` (with SPIR-V target support) on `$PATH`.
pub fn find_clang() -> Option<PathBuf> {
    which_in_path("clang")
}

/// Try to locate Intel `ocloc` (oneAPI offline compiler) on `$PATH`.
pub fn find_ocloc() -> Option<PathBuf> {
    which_in_path("ocloc")
}

/// Compile an OpenCL `.cl` source file to SPIR-V `.spv`.
///
/// Tries `clang` first, then `ocloc`. Returns [`SpvCompileResult`].
pub fn compile_cl_to_spv(cl_path: &Path, spv_path: &Path) -> SpvCompileResult {
    if let Some(clang) = find_clang() {
        return compile_with_clang(&clang, cl_path, spv_path);
    }
    if let Some(ocloc) = find_ocloc() {
        return compile_with_ocloc(&ocloc, cl_path, spv_path);
    }
    SpvCompileResult::CompilerNotFound
}

/// Compile using `clang -cl-std=CL3.0 -target spir64 -O2 -o output.spv input.cl`
fn compile_with_clang(clang: &Path, cl_path: &Path, spv_path: &Path) -> SpvCompileResult {
    let output = Command::new(clang)
        .args([
            "-cl-std=CL3.0",
            "-target",
            "spir64",
            "-O2",
            "-Xclang",
            "-finclude-default-header",
            "-o",
        ])
        .arg(spv_path)
        .arg(cl_path)
        .output();

    match output {
        Ok(o) if o.status.success() => SpvCompileResult::Compiled(spv_path.to_path_buf()),
        Ok(o) => SpvCompileResult::Failed(format!(
            "clang exited {}: {}",
            o.status,
            String::from_utf8_lossy(&o.stderr)
        )),
        Err(e) => SpvCompileResult::Failed(format!("clang exec error: {e}")),
    }
}

/// Compile using `ocloc compile -file input.cl -output output_dir -out_dir …`
fn compile_with_ocloc(ocloc: &Path, cl_path: &Path, spv_path: &Path) -> SpvCompileResult {
    let out_dir = spv_path.parent().unwrap_or(Path::new("."));
    let stem = spv_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("kernel");

    let output = Command::new(ocloc)
        .args(["compile", "-file"])
        .arg(cl_path)
        .args(["-output", stem, "-out_dir"])
        .arg(out_dir)
        .output();

    match output {
        Ok(o) if o.status.success() => SpvCompileResult::Compiled(spv_path.to_path_buf()),
        Ok(o) => SpvCompileResult::Failed(format!(
            "ocloc exited {}: {}",
            o.status,
            String::from_utf8_lossy(&o.stderr)
        )),
        Err(e) => SpvCompileResult::Failed(format!("ocloc exec error: {e}")),
    }
}

/// Simple `which`-style lookup on `$PATH`.
fn which_in_path(name: &str) -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths).find_map(|dir| {
            let candidate = dir.join(exe_name(name));
            candidate.is_file().then_some(candidate)
        })
    })
}

#[cfg(windows)]
fn exe_name(name: &str) -> String {
    format!("{name}.exe")
}

#[cfg(not(windows))]
fn exe_name(name: &str) -> String {
    name.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn compiler_not_found_returns_gracefully() {
        // With a fabricated path, neither tool exists
        let result = compile_cl_to_spv(
            Path::new("nonexistent.cl"),
            Path::new("nonexistent.spv"),
        );
        // Either CompilerNotFound or Failed is acceptable
        match result {
            SpvCompileResult::CompilerNotFound => {}
            SpvCompileResult::Failed(_) => {}
            SpvCompileResult::Compiled(_) => {
                panic!("should not compile from nonexistent file")
            }
        }
    }

    #[test]
    fn find_functions_return_none_or_some() {
        // Smoke: these should not panic
        let _clang = find_clang();
        let _ocloc = find_ocloc();
    }

    #[test]
    fn spv_compile_result_debug_display() {
        // Verify Debug impl works
        let r = SpvCompileResult::CompilerNotFound;
        let dbg = format!("{r:?}");
        assert!(dbg.contains("CompilerNotFound"));
    }

    #[test]
    fn compile_with_missing_source_file() {
        let tmp = std::env::temp_dir().join("bitnet_spirv_test");
        let _ = fs::create_dir_all(&tmp);
        let cl = tmp.join("missing.cl");
        let spv = tmp.join("missing.spv");

        let result = compile_cl_to_spv(&cl, &spv);
        match result {
            SpvCompileResult::Compiled(_) => {
                panic!("should not succeed with missing source")
            }
            _ => {} // CompilerNotFound or Failed are both valid
        }
    }
}

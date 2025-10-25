//! Automatic C++ reference setup with cross-platform dynamic loader exports.
//!
//! This module provides one-command bootstrapping for the Microsoft BitNet C++
//! reference, emitting shell-specific environment variable exports that the
//! parent shell can eval.
//!
//! # Usage
//!
//! ```bash
//! # Linux/macOS (sh/bash/zsh)
//! eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"
//!
//! # fish
//! cargo run -p xtask -- setup-cpp-auto --emit=fish | source
//!
//! # Windows PowerShell
//! cargo run -p xtask -- setup-cpp-auto --emit=pwsh | Invoke-Expression
//!
//! # Windows cmd
//! cargo run -p xtask -- setup-cpp-auto --emit=cmd
//! ```

use anyhow::{Context, Result, bail};
use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};
use walkdir::WalkDir;

/// Shell format for environment variable exports
#[derive(Clone, Copy, Debug)]
pub enum Emit {
    /// POSIX sh/bash/zsh (default)
    Sh,
    /// fish shell
    Fish,
    /// PowerShell
    Pwsh,
    /// Windows cmd
    Cmd,
}

impl Emit {
    /// Parse emit format from string
    pub fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "fish" => Emit::Fish,
            "pwsh" | "powershell" => Emit::Pwsh,
            "cmd" | "batch" => Emit::Cmd,
            _ => Emit::Sh,
        }
    }
}

/// Find the directory containing the shared library (libllama.so/dylib/dll)
///
/// Searches common CMake output locations first, then falls back to recursive search.
fn find_lib_dir(build: &Path) -> Result<PathBuf> {
    // Check common locations first (fast path)
    let candidates = [
        build.to_path_buf(),
        build.join("bin"),
        build.join("Release"),
        build.join("Debug"),
        build.join("lib"),
    ];

    for dir in &candidates {
        if let Ok(read_dir) = fs::read_dir(dir) {
            for entry in read_dir.flatten() {
                if let Some(name) = entry.file_name().to_str()
                    && (name.contains("llama") || name.contains("bitnet"))
                    && (name.ends_with(".so") || name.ends_with(".dylib") || name.ends_with(".dll"))
                {
                    return Ok(dir.clone());
                }
            }
        }
    }

    // Fallback: recursive search (slower but comprehensive)
    // Depth 5 needed for build/3rdparty/llama.cpp/src/libllama.so
    for entry in WalkDir::new(build).max_depth(5).into_iter().flatten() {
        let path = entry.path();
        if let Some(name) = path.file_name().and_then(|s| s.to_str())
            && (name.contains("llama") || name.contains("bitnet"))
            && (name.ends_with(".so") || name.ends_with(".dylib") || name.ends_with(".dll"))
            && let Some(parent) = path.parent()
        {
            return Ok(parent.to_path_buf());
        }
    }

    bail!(
        "Shared library not found under {}. Expected libbitnet.* or libllama.{{so,dylib,dll}}",
        build.display()
    )
}

/// Auto-bootstrap C++ reference and emit dynamic loader exports
///
/// # Workflow
///
/// 1. Resolve `BITNET_CPP_DIR` (default: `~/.cache/bitnet_cpp`)
/// 2. Fetch and build C++ reference if not present (calls `fetch-cpp`)
/// 3. Verify build directory exists
/// 4. Emit shell-specific exports for dynamic loader path
///
/// # Environment
///
/// - `BITNET_CPP_DIR`: Override default C++ reference location
/// - `BITNET_CPP_PATH`: Deprecated alias for `BITNET_CPP_DIR` (fallback)
///
/// # Returns
///
/// Prints shell exports to stdout on success. Returns error if:
/// - Home directory not found
/// - `fetch-cpp` command fails
/// - Build directory missing after fetch
pub fn run(emit: Emit) -> Result<()> {
    let home = dirs::home_dir().context("no home directory found")?;

    // Tier 1: Explicit BITNET_CPP_DIR (highest priority)
    // Tier 2: Deprecated BITNET_CPP_PATH (fallback for backward compatibility)
    // Tier 3: Runtime default ~/.cache/bitnet_cpp (via dirs::home_dir())
    let repo = env::var("BITNET_CPP_DIR")
        .or_else(|_| env::var("BITNET_CPP_PATH"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| home.join(".cache/bitnet_cpp"));

    // 1) Fetch/build C++ reference if not present
    if !repo.exists() {
        eprintln!("[bitnet] C++ reference not found at {}", repo.display());
        eprintln!("[bitnet] Fetching and building C++ reference...");

        let status = Command::new("cargo")
            .args(["run", "-p", "xtask", "--", "fetch-cpp"])
            .status()
            .context("failed to spawn cargo for fetch-cpp")?;

        if !status.success() {
            bail!("fetch-cpp command failed with exit code: {:?}", status.code());
        }
    }

    // 2) Verify build directory exists
    let build = repo.join("build");
    if !build.exists() {
        bail!(
            "C++ build directory not found at {}. Try re-running: cargo run -p xtask -- fetch-cpp",
            build.display()
        );
    }

    // 3) Find the actual lib directory
    let lib_dir = find_lib_dir(&build)?;

    // 4) Auto-discover BITNET_CROSSVAL_LIBDIR if not explicitly set
    // Check multiple candidate locations in priority order
    let crossval_libdir = env::var("BITNET_CROSSVAL_LIBDIR").ok().or_else(|| {
        let candidates = [
            // Priority 1: BitNet.cpp embedded llama.cpp (CMake standard)
            repo.join("build/3rdparty/llama.cpp/build/bin"),
            // Priority 2: llama.cpp source directory (in-tree build)
            repo.join("build/3rdparty/llama.cpp/src"),
            // Priority 3: Standard CMake output locations
            repo.join("build/bin"),
            repo.join("build/lib"),
        ];

        for candidate in &candidates {
            if candidate.is_dir() {
                // Verify it contains libraries
                if let Ok(entries) = fs::read_dir(candidate) {
                    for entry in entries.flatten() {
                        if let Some(name) = entry.file_name().to_str()
                            && (name.contains("llama") || name.contains("bitnet"))
                            && (name.ends_with(".so")
                                || name.ends_with(".dylib")
                                || name.ends_with(".dll"))
                        {
                            return Some(candidate.display().to_string());
                        }
                    }
                }
            }
        }
        None
    });

    // 5) Emit shell-specific dynamic loader exports
    emit_exports(emit, &repo, &lib_dir, crossval_libdir.as_deref());

    Ok(())
}

/// Emit environment variable exports for the specified shell
fn emit_exports(emit: Emit, repo: &Path, lib_dir: &Path, crossval_libdir: Option<&str>) {
    match emit {
        Emit::Sh => {
            // POSIX sh/bash/zsh
            println!(r#"export BITNET_CPP_DIR="{}""#, repo.display());

            // Optionally emit BITNET_CROSSVAL_LIBDIR if auto-discovered
            if let Some(libdir) = crossval_libdir {
                println!(r#"export BITNET_CROSSVAL_LIBDIR="{}""#, libdir);
            }

            #[cfg(target_os = "linux")]
            println!(r#"export LD_LIBRARY_PATH="{}:${{LD_LIBRARY_PATH:-}}""#, lib_dir.display());

            #[cfg(target_os = "macos")]
            println!(
                r#"export DYLD_LIBRARY_PATH="{}:${{DYLD_LIBRARY_PATH:-}}""#,
                lib_dir.display()
            );

            #[cfg(target_os = "windows")]
            println!(r#"export PATH="{}:${{PATH:-}}""#, lib_dir.display());

            println!(r#"echo "[bitnet] C++ ready at $BITNET_CPP_DIR""#);
        }

        Emit::Fish => {
            // fish shell
            println!(r#"set -gx BITNET_CPP_DIR "{}""#, repo.display());

            // Optionally emit BITNET_CROSSVAL_LIBDIR if auto-discovered
            if let Some(libdir) = crossval_libdir {
                println!(r#"set -gx BITNET_CROSSVAL_LIBDIR "{}""#, libdir);
            }

            #[cfg(target_os = "linux")]
            println!(r#"set -gx LD_LIBRARY_PATH "{}" $LD_LIBRARY_PATH"#, lib_dir.display());

            #[cfg(target_os = "macos")]
            println!(r#"set -gx DYLD_LIBRARY_PATH "{}" $DYLD_LIBRARY_PATH"#, lib_dir.display());

            #[cfg(target_os = "windows")]
            println!(r#"set -gx PATH "{}" $PATH"#, lib_dir.display());

            println!(r#"echo "[bitnet] C++ ready at $BITNET_CPP_DIR""#);
        }

        Emit::Pwsh => {
            // PowerShell
            println!(r#"$env:BITNET_CPP_DIR = "{}""#, repo.display());

            // Optionally emit BITNET_CROSSVAL_LIBDIR if auto-discovered
            if let Some(libdir) = crossval_libdir {
                println!(r#"$env:BITNET_CROSSVAL_LIBDIR = "{}""#, libdir);
            }

            println!(r#"$env:PATH = "{};" + $env:PATH"#, lib_dir.display());
            println!(r#"Write-Host "[bitnet] C++ ready at $env:BITNET_CPP_DIR""#);
        }

        Emit::Cmd => {
            // Windows cmd (batch)
            println!(r#"set BITNET_CPP_DIR={}"#, repo.display());

            // Optionally emit BITNET_CROSSVAL_LIBDIR if auto-discovered
            if let Some(libdir) = crossval_libdir {
                println!(r#"set BITNET_CROSSVAL_LIBDIR={}"#, libdir);
            }

            println!(r#"set PATH={};%PATH%"#, lib_dir.display());
            println!(r#"echo [bitnet] C++ ready at %BITNET_CPP_DIR%"#);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emit_from_string() {
        assert!(matches!(Emit::from("sh"), Emit::Sh));
        assert!(matches!(Emit::from("fish"), Emit::Fish));
        assert!(matches!(Emit::from("pwsh"), Emit::Pwsh));
        assert!(matches!(Emit::from("PowerShell"), Emit::Pwsh));
        assert!(matches!(Emit::from("cmd"), Emit::Cmd));
        assert!(matches!(Emit::from("batch"), Emit::Cmd));
        assert!(matches!(Emit::from("unknown"), Emit::Sh)); // default
    }
}

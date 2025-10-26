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

// TDD scaffolding - suppress warnings for planned implementation
#![allow(dead_code)]
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

/// Extract last N lines from a string
///
/// Helper for truncating build logs to show only relevant output.
fn last_n_lines(s: &str, n: usize) -> String {
    s.lines().rev().take(n).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join("\n")
}

/// GitHub URL for Microsoft BitNet repository
const BITNET_REPO_URL: &str = "https://github.com/microsoft/BitNet";

/// GitHub URL for llama.cpp repository
const LLAMA_REPO_URL: &str = "https://github.com/ggerganov/llama.cpp";

/// C++ backend selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CppBackend {
    BitNet,
    Llama,
}

impl CppBackend {
    /// Get the display name for the backend
    pub fn name(&self) -> &'static str {
        match self {
            CppBackend::BitNet => "bitnet.cpp",
            CppBackend::Llama => "llama.cpp",
        }
    }

    /// Get the GitHub repository URL for the backend
    pub fn repo_url(&self) -> &'static str {
        match self {
            CppBackend::BitNet => BITNET_REPO_URL,
            CppBackend::Llama => LLAMA_REPO_URL,
        }
    }

    /// Get the default installation subdirectory name
    pub fn install_subdir(&self) -> &'static str {
        match self {
            CppBackend::BitNet => "bitnet_cpp",
            CppBackend::Llama => "llama_cpp",
        }
    }
}

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

/// Check if a directory contains BitNet or llama libraries
///
/// Returns true if the directory contains at least one library file matching:
/// - libbitnet*.so/.dylib or bitnet*.dll (BitNet.cpp)
/// - libllama*.so/.dylib or llama*.dll (llama.cpp)
/// - libggml*.so/.dylib or ggml*.dll (GGML backend)
fn has_libraries(dir: &Path) -> bool {
    if !dir.is_dir() {
        return false;
    }

    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                let is_library =
                    (name.contains("bitnet") || name.contains("llama") || name.contains("ggml"))
                        && (name.ends_with(".so")
                            || name.ends_with(".dylib")
                            || name.ends_with(".dll"));
                if is_library {
                    return true;
                }
            }
        }
    }

    false
}

/// Check if a directory contains specific required libraries
///
/// Returns true only if ALL required libraries are present in the directory.
///
/// # Arguments
///
/// * `dir` - Directory to check
/// * `required_libs` - List of library base names (without lib prefix or extension)
///
/// # Example
///
/// ```ignore
/// has_required_libraries(&path, &["llama", "ggml"]) // Checks for libllama.so AND libggml.so
/// ```
fn has_required_libraries(dir: &Path, required_libs: &[&str]) -> bool {
    if !dir.is_dir() {
        return false;
    }

    // Check that ALL required libraries are present
    for required_lib in required_libs {
        let mut found = false;

        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    // Check if filename contains the library name AND has correct extension
                    let is_library = name.contains(required_lib)
                        && (name.ends_with(".so")
                            || name.ends_with(".dylib")
                            || name.ends_with(".dll"));

                    if is_library {
                        found = true;
                        break;
                    }
                }
            }
        }

        // If any required library is missing, return false
        if !found {
            return false;
        }
    }

    true
}

/// Find all library directories for BitNet.cpp installation
///
/// Discovers both BitNet.cpp libraries and vendored llama.cpp libraries
/// in priority order. Returns all directories containing libraries.
///
/// # Search Tiers
///
/// **Tier 1 (BitNet.cpp libraries)**:
/// - `build/bin` - Primary CMake output
/// - `build/lib` - Standard CMake library location
/// - `build` - Root build directory
///
/// **Tier 2 (Vendored llama.cpp libraries)**:
/// - `build/3rdparty/llama.cpp/build/bin` - Vendored llama CMake output
/// - `build/3rdparty/llama.cpp/build/lib` - Vendored llama lib directory
///
/// **Tier 3 (Fallback)**:
/// - `lib` - Root lib directory
///
/// # Environment Variable Override
///
/// - `BITNET_CROSSVAL_LIBDIR`: Overrides all auto-discovery (takes highest priority)
/// - `BITNET_CPP_DIR`: Affects base install_dir for discovery
///
/// # Returns
///
/// Vector of PathBufs for directories containing libraries, in priority order.
/// Empty vector if no libraries found (caller should handle error).
pub fn find_bitnet_lib_dirs(install_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut lib_dirs = vec![];

    // Priority 0: Explicit override via BITNET_CROSSVAL_LIBDIR
    if let Ok(explicit_libdir) = env::var("BITNET_CROSSVAL_LIBDIR") {
        let explicit_path = PathBuf::from(explicit_libdir);
        if has_libraries(&explicit_path) {
            return Ok(vec![explicit_path]);
        }
    }

    // Tier 1: BitNet.cpp libraries
    let bitnet_candidates =
        [install_dir.join("build/bin"), install_dir.join("build/lib"), install_dir.join("build")];

    // Tier 2: Vendored llama.cpp libraries
    let llama_candidates = [
        install_dir.join("build/3rdparty/llama.cpp/build/bin"),
        install_dir.join("build/3rdparty/llama.cpp/build/lib"),
    ];

    // Tier 3: Fallback
    let fallback = [install_dir.join("lib")];

    // Search in priority order
    for candidate in bitnet_candidates.iter().chain(llama_candidates.iter()).chain(fallback.iter())
    {
        if has_libraries(candidate) {
            lib_dirs.push(candidate.clone());
        }
    }

    Ok(lib_dirs)
}
/// Determine the installation directory for a C++ backend
///
/// Follows environment variable precedence:
/// 1. {BACKEND}_CPP_DIR (explicit override, highest priority)
/// 2. ~/.cache/{backend}_cpp (default)
///
/// Creates parent directories if they don't exist.
///
/// # Arguments
///
/// * `backend` - The C++ backend to determine installation directory for
///
/// # Returns
///
/// PathBuf to the installation directory
fn determine_install_dir(backend: CppBackend) -> Result<PathBuf> {
    let home = dirs::home_dir().context("no home directory found")?;

    let env_var = match backend {
        CppBackend::BitNet => "BITNET_CPP_DIR",
        CppBackend::Llama => "LLAMA_CPP_DIR",
    };

    let install_dir = env::var(env_var)
        .map(PathBuf::from)
        .unwrap_or_else(|_| home.join(".cache").join(backend.install_subdir()));

    // Create parent directories if they don't exist
    if let Some(parent) = install_dir.parent()
        && !parent.exists()
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create parent directory: {}", parent.display()))?;
    }

    Ok(install_dir)
}

/// Determine the installation directory for BitNet.cpp
///
/// Follows environment variable precedence:
/// 1. BITNET_CPP_DIR (explicit override, highest priority)
/// 2. ~/.cache/bitnet_cpp (default)
///
/// Creates parent directories if they don't exist.
///
/// # Returns
///
/// PathBuf to the BitNet.cpp installation directory
fn determine_bitnet_cpp_dir() -> Result<PathBuf> {
    determine_install_dir(CppBackend::BitNet)
}

/// Clone a C++ repository from GitHub with recursive submodules
///
/// Clones with --recurse-submodules to initialize all submodules recursively.
/// This is required for BitNet.cpp (which vendors llama.cpp as a submodule).
///
/// # Arguments
///
/// * `url` - GitHub repository URL
/// * `dest` - Target directory for clone
///
/// # Returns
///
/// Ok(()) if clone succeeds, Err on network or git failures
fn clone_repository(url: &str, dest: &Path) -> Result<()> {
    // Check if git is available
    let git_check = Command::new("git")
        .arg("--version")
        .output()
        .context("Failed to find git - is it installed?")?;

    if !git_check.status.success() {
        bail!("git command not available");
    }

    eprintln!("[bitnet] Cloning from {}...", url);

    // Clone with recursive submodule initialization
    // Note: --depth=1 removed to avoid issues with detached HEAD state
    let output = Command::new("git")
        .arg("clone")
        .arg("--recurse-submodules")
        .arg(url)
        .arg(dest)
        .output()
        .context("Failed to execute git clone")?;

    if !output.status.success() {
        bail!(
            "git clone failed with exit code {:?}\nStderr: {}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    eprintln!("[bitnet] Clone succeeded");

    Ok(())
}

/// Clone BitNet.cpp repository from GitHub
///
/// Clones with --recurse-submodules to initialize the vendored llama.cpp submodule.
///
/// # Arguments
///
/// * `install_dir` - Target directory for clone
///
/// # Returns
///
/// Ok(()) if clone succeeds, Err on network or git failures
fn clone_bitnet_cpp(install_dir: &Path) -> Result<()> {
    clone_repository(BITNET_REPO_URL, install_dir)
}

/// Update existing C++ repository
///
/// Runs `git pull --ff-only` to update the repository and
/// `git submodule update --init --recursive` to update submodules.
///
/// Handles detached HEAD state gracefully by skipping pull.
///
/// # Arguments
///
/// * `install_dir` - Root directory of the repository
///
/// # Returns
///
/// Ok(()) if update succeeds or repo is already up-to-date, Err on git failures
fn update_repository(install_dir: &Path) -> Result<()> {
    let git_dir = install_dir.join(".git");

    if !git_dir.exists() {
        eprintln!("[bitnet] Not a git repository, skipping update");
        return Ok(());
    }

    eprintln!("[bitnet] Updating repository...");

    // Try git pull (idempotent, fast-forward only)
    let pull_output = Command::new("git")
        .arg("pull")
        .arg("--ff-only")
        .current_dir(install_dir)
        .output()
        .context("Failed to execute git pull")?;

    if !pull_output.status.success() {
        // Check if we're in detached HEAD state
        let stderr = String::from_utf8_lossy(&pull_output.stderr);
        if stderr.contains("detached HEAD") || stderr.contains("not currently on a branch") {
            eprintln!("[bitnet] Detached HEAD detected, skipping pull");
        } else {
            eprintln!("[bitnet] Warning: git pull failed (non-fatal): {}", stderr);
        }
    } else {
        eprintln!("[bitnet] Repository updated");
    }

    // Update submodules (always safe, idempotent)
    eprintln!("[bitnet] Updating submodules...");
    let submodule_output = Command::new("git")
        .arg("submodule")
        .arg("update")
        .arg("--init")
        .arg("--recursive")
        .current_dir(install_dir)
        .output()
        .context("Failed to update git submodules")?;

    if !submodule_output.status.success() {
        eprintln!(
            "[bitnet] Warning: submodule update failed (non-fatal): {}",
            String::from_utf8_lossy(&submodule_output.stderr)
        );
    } else {
        eprintln!("[bitnet] Submodules updated");
    }

    Ok(())
}

/// Update existing BitNet.cpp repository
///
/// Wrapper for update_repository (for backward compatibility).
///
/// # Arguments
///
/// * `install_dir` - Root directory of BitNet.cpp repository
///
/// # Returns
///
/// Ok(()) if update succeeds or repo is already up-to-date, Err on git failures
fn update_bitnet_cpp(install_dir: &Path) -> Result<()> {
    update_repository(install_dir)
}

/// Install or update a C++ backend and build if necessary
///
/// # Workflow
///
/// 1. Determine installation directory (env var or default)
/// 2. If directory doesn't exist, clone from GitHub
/// 3. If directory exists, update (git pull + submodule update)
/// 4. Build if not already built (fast-path check)
///
/// # Arguments
///
/// * `backend` - The C++ backend to install or update
///
/// # Returns
///
/// PathBuf to the installation directory
fn install_or_update_backend(backend: CppBackend) -> Result<PathBuf> {
    let install_dir = determine_install_dir(backend)?;

    if !install_dir.exists() {
        // Fresh installation
        clone_repository(backend.repo_url(), &install_dir)?;
        build_backend(backend, &install_dir)?;
    } else {
        // Update existing installation
        update_repository(&install_dir)?;

        // Check if build exists (fast-path)
        let build_dir = install_dir.join("build");
        if !build_dir.exists() || !has_libraries(&build_dir) {
            eprintln!("[bitnet] Build artifacts not found, building...");
            build_backend(backend, &install_dir)?;
        } else {
            eprintln!("[bitnet] {} already built at {}", backend.name(), install_dir.display());
        }
    }

    Ok(install_dir)
}

/// Build a C++ backend
///
/// Routes to the appropriate build function based on backend type.
///
/// # Arguments
///
/// * `backend` - The C++ backend to build
/// * `install_dir` - Root directory of the repository
///
/// # Returns
///
/// Ok(()) on successful build, Err otherwise
fn build_backend(backend: CppBackend, install_dir: &Path) -> Result<()> {
    match backend {
        CppBackend::BitNet => build_bitnet_cpp(install_dir),
        CppBackend::Llama => build_llama_cpp(install_dir),
    }
}

/// Build llama.cpp using CMake
///
/// llama.cpp uses CMake-only build (no setup_env.py).
///
/// # Arguments
///
/// * `install_dir` - Root directory of llama.cpp repository
///
/// # Returns
///
/// Ok(()) on successful build, Err otherwise
fn build_llama_cpp(install_dir: &Path) -> Result<()> {
    eprintln!("[llama] Building llama.cpp with CMake...");

    let build_dir = install_dir.join("build");

    // Create build directory if it doesn't exist
    if !build_dir.exists() {
        fs::create_dir_all(&build_dir).context("Failed to create llama.cpp build directory")?;
    }

    // Check if cmake is available
    let cmake_available = Command::new("cmake")
        .arg("--version")
        .output()
        .map(|out| out.status.success())
        .unwrap_or(false);

    if !cmake_available {
        bail!(
            "CMake not found. Please install cmake:\n\
             \n\
             Ubuntu/Debian:  sudo apt install cmake\n\
             macOS:          brew install cmake\n\
             Windows:        choco install cmake"
        );
    }

    // Configure with CMake (llama.cpp-specific flags)
    eprintln!("[llama] Configuring with CMake...");
    let configure_output = Command::new("cmake")
        .arg("..")
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg("-DBUILD_SHARED_LIBS=ON")
        .current_dir(&build_dir)
        .output()
        .context("Failed to run cmake configuration")?;

    if !configure_output.status.success() {
        let stdout = String::from_utf8_lossy(&configure_output.stdout);
        let stderr = String::from_utf8_lossy(&configure_output.stderr);

        bail!(
            "llama.cpp CMake configuration failed\n\
             \n\
             Stdout (last 50 lines):\n{}\n\
             \n\
             Stderr (last 50 lines):\n{}\n\
             \n\
             Troubleshooting:\n\
             1. Check CMake version: cmake --version (requires ≥3.18)\n\
             2. For CPU-only build: -DGGML_CUDA=OFF\n\
             3. Full log: {}/CMakeFiles/CMakeOutput.log\n\
             \n\
             See: docs/howto/cpp-setup.md#troubleshooting-build-failures",
            last_n_lines(&stdout, 50),
            last_n_lines(&stderr, 50),
            build_dir.display()
        );
    }

    // Build with CMake
    eprintln!("[llama] Building with CMake (parallel)...");
    let build_output = Command::new("cmake")
        .arg("--build")
        .arg(".")
        .arg("--parallel")
        .current_dir(&build_dir)
        .output()
        .context("Failed to run cmake build")?;

    if !build_output.status.success() {
        let stdout = String::from_utf8_lossy(&build_output.stdout);
        let stderr = String::from_utf8_lossy(&build_output.stderr);

        bail!(
            "llama.cpp CMake build failed\n\
             \n\
             Stdout (last 50 lines):\n{}\n\
             \n\
             Stderr (last 50 lines):\n{}\n\
             \n\
             Troubleshooting:\n\
             1. Check compiler is installed: gcc --version or clang --version\n\
             2. Check disk space: df -h\n\
             3. Try sequential build: cmake --build . (without --parallel)\n\
             \n\
             See: docs/howto/cpp-setup.md#troubleshooting-build-failures",
            last_n_lines(&stdout, 50),
            last_n_lines(&stderr, 50)
        );
    }

    eprintln!("[llama] llama.cpp build succeeded");
    Ok(())
}

/// Install or update BitNet.cpp and build if necessary
///
/// # Workflow
///
/// 1. Determine installation directory (env var or default)
/// 2. If directory doesn't exist, clone from GitHub
/// 3. If directory exists, update (git pull + submodule update)
/// 4. Build if not already built (fast-path check)
///
/// # Returns
///
/// PathBuf to the BitNet.cpp installation directory
pub fn install_or_update_bitnet_cpp() -> Result<PathBuf> {
    install_or_update_backend(CppBackend::BitNet)
}

/// Install or update llama.cpp and build if necessary
///
/// # Workflow
///
/// 1. Determine installation directory (env var or default)
/// 2. If directory doesn't exist, clone from GitHub
/// 3. If directory exists, update (git pull + submodule update)
/// 4. Build if not already built (fast-path check)
///
/// # Returns
///
/// PathBuf to the llama.cpp installation directory
pub fn install_or_update_llama_cpp() -> Result<PathBuf> {
    install_or_update_backend(CppBackend::Llama)
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

        let output = Command::new("cargo")
            .args(["run", "-p", "xtask", "--", "fetch-cpp"])
            .output()
            .context("failed to spawn cargo for fetch-cpp")?;

        if !output.status.success() {
            // Capture stderr for detailed error diagnostics
            let stderr = String::from_utf8_lossy(&output.stderr);

            // Preserve full error context for upstream error classification
            bail!(
                "fetch-cpp command failed with exit code: {:?}\n\nError output:\n{}",
                output.status.code(),
                stderr
            );
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

/// Build BitNet.cpp using setup_env.py or CMake fallback
///
/// Tries setup_env.py first (if present), falls back to manual CMake build.
/// Also builds vendored llama.cpp in 3rdparty/llama.cpp/.
/// Verifies libraries exist after build.
///
/// # Arguments
///
/// * `install_dir` - Root directory of BitNet.cpp repository
///
/// # Returns
///
/// Ok(()) on successful build and verification, Err if build fails or libraries not found
fn build_bitnet_cpp(install_dir: &Path) -> Result<()> {
    // AC2: Build detection - Try setup_env.py first (preferred method)
    if install_dir.join("setup_env.py").exists() {
        eprintln!("[bitnet] Detected setup_env.py, using as primary build method");
        match run_setup_env_py(install_dir) {
            Ok(_) => {
                eprintln!("[bitnet] setup_env.py build succeeded");
            }
            Err(e) => {
                eprintln!("[bitnet] setup_env.py failed: {}, trying CMake fallback...", e);
                run_cmake_build(install_dir)?;
            }
        }
    } else {
        eprintln!("[bitnet] setup_env.py not found, using CMake build");
        run_cmake_build(install_dir)?;
    }

    // Build vendored llama.cpp
    build_vendored_llama_cpp(install_dir)?;

    // AC3: Library verification - Verify libraries were built successfully
    eprintln!("[bitnet] Verifying built libraries...");
    let lib_dirs = find_bitnet_lib_dirs(install_dir)?;

    if lib_dirs.is_empty() {
        bail!(
            "No BitNet.cpp libraries found after build.\n\
             \n\
             Expected libraries:\n\
             - libbitnet.so (or .dylib/.dll)\n\
             - libllama.so (vendored llama.cpp)\n\
             \n\
             Searched directories:\n\
             ✗ {}/build/bin (not found or empty)\n\
             ✗ {}/build/3rdparty/llama.cpp/build/bin (not found or empty)\n\
             \n\
             Debugging steps:\n\
             1. Check build logs for errors\n\
             2. Verify CMake completed successfully\n\
             3. Check disk space: df -h\n\
             \n\
             See: docs/howto/cpp-setup.md#troubleshooting-build-failures",
            install_dir.display(),
            install_dir.display()
        );
    }

    eprintln!(
        "[bitnet] Library verification passed: {} director{} with libraries",
        lib_dirs.len(),
        if lib_dirs.len() == 1 { "y" } else { "ies" }
    );

    for (idx, dir) in lib_dirs.iter().enumerate() {
        eprintln!("  {}. {}", idx + 1, dir.display());
    }

    Ok(())
}

/// Execute setup_env.py in the BitNet.cpp repository
///
/// Runs BitNet.cpp's setup_env.py build script with proper error diagnostics.
///
/// # Arguments
///
/// * `repo_path` - Root directory of BitNet.cpp repository
///
/// # Returns
///
/// Ok(()) if setup_env.py succeeds, Err with detailed diagnostics otherwise
///
/// # Error Handling
///
/// - Python not found → BuildFailure with platform-specific installation commands
/// - Build failure → BuildFailure with last 50 lines of stdout/stderr
pub fn run_setup_env_py(repo_path: &Path) -> Result<()> {
    // Find python3 executable
    let python = if cfg!(target_os = "windows") { "python" } else { "python3" };

    // Check if python is available
    let python_available = Command::new(python)
        .arg("--version")
        .output()
        .map(|out| out.status.success())
        .unwrap_or(false);

    if !python_available {
        bail!(
            "Python not found. Please install python3:\n\
             \n\
             Ubuntu/Debian:  sudo apt install python3\n\
             macOS:          brew install python3\n\
             Windows:        choco install python3"
        );
    }

    eprintln!("[bitnet] Running setup_env.py...");

    let output = Command::new(python)
        .arg("setup_env.py")
        .current_dir(repo_path)
        .output()
        .context("Failed to spawn setup_env.py process")?;

    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        bail!(
            "setup_env.py failed with exit code {:?}\n\
             \n\
             Stdout (last 50 lines):\n{}\n\
             \n\
             Stderr (last 50 lines):\n{}\n\
             \n\
             Troubleshooting:\n\
             1. Check CMake is installed: cmake --version\n\
             2. Check build dependencies are available\n\
             3. Try CMake fallback: --force with manual build\n\
             \n\
             See: docs/howto/cpp-setup.md#troubleshooting-build-failures",
            output.status.code(),
            last_n_lines(&stdout, 50),
            last_n_lines(&stderr, 50)
        );
    }

    eprintln!("[bitnet] setup_env.py succeeded");
    Ok(())
}

/// Run manual CMake build as fallback
///
/// Creates build directory and runs cmake configuration and build with
/// comprehensive error diagnostics.
///
/// # Arguments
///
/// * `repo_path` - Root directory of BitNet.cpp repository
///
/// # Returns
///
/// Ok(()) if CMake build succeeds, Err with detailed diagnostics otherwise
///
/// # Error Handling
///
/// - CMake not found → BuildFailure with platform-specific installation commands
/// - Configuration failure → BuildFailure with CMake log excerpt
/// - Build failure → BuildFailure with last 50 lines of stdout/stderr
pub fn run_cmake_build(repo_path: &Path) -> Result<()> {
    let build_dir = repo_path.join("build");

    // Create build directory if it doesn't exist
    if !build_dir.exists() {
        fs::create_dir_all(&build_dir).context("Failed to create build directory")?;
    }

    // Check if cmake is available
    let cmake_available = Command::new("cmake")
        .arg("--version")
        .output()
        .map(|out| out.status.success())
        .unwrap_or(false);

    if !cmake_available {
        bail!(
            "CMake not found. Please install cmake:\n\
             \n\
             Ubuntu/Debian:  sudo apt install cmake\n\
             macOS:          brew install cmake\n\
             Windows:        choco install cmake"
        );
    }

    // Configure with CMake
    eprintln!("[bitnet] Configuring with CMake...");
    let configure_output = Command::new("cmake")
        .arg("..")
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .current_dir(&build_dir)
        .output()
        .context("Failed to run cmake configuration")?;

    if !configure_output.status.success() {
        let stdout = String::from_utf8_lossy(&configure_output.stdout);
        let stderr = String::from_utf8_lossy(&configure_output.stderr);

        bail!(
            "CMake configuration failed\n\
             \n\
             Stdout (last 50 lines):\n{}\n\
             \n\
             Stderr (last 50 lines):\n{}\n\
             \n\
             Troubleshooting:\n\
             1. For CPU-only build: -DGGML_CUDA=OFF\n\
             2. Check CMake version: cmake --version (requires ≥3.18)\n\
             3. Full log: {}/CMakeFiles/CMakeOutput.log\n\
             \n\
             See: docs/howto/cpp-setup.md#troubleshooting-build-failures",
            last_n_lines(&stdout, 50),
            last_n_lines(&stderr, 50),
            build_dir.display()
        );
    }

    // Build with CMake
    eprintln!("[bitnet] Building with CMake (parallel)...");
    let build_output = Command::new("cmake")
        .arg("--build")
        .arg(".")
        .arg("--parallel")
        .current_dir(&build_dir)
        .output()
        .context("Failed to run cmake build")?;

    if !build_output.status.success() {
        let stdout = String::from_utf8_lossy(&build_output.stdout);
        let stderr = String::from_utf8_lossy(&build_output.stderr);

        bail!(
            "CMake build failed\n\
             \n\
             Stdout (last 50 lines):\n{}\n\
             \n\
             Stderr (last 50 lines):\n{}\n\
             \n\
             Troubleshooting:\n\
             1. Check compiler is installed: gcc --version or clang --version\n\
             2. Check disk space: df -h\n\
             3. Try sequential build: cmake --build . (without --parallel)\n\
             \n\
             See: docs/howto/cpp-setup.md#troubleshooting-build-failures",
            last_n_lines(&stdout, 50),
            last_n_lines(&stderr, 50)
        );
    }

    eprintln!("[bitnet] CMake build succeeded");
    Ok(())
}

/// Build vendored llama.cpp in 3rdparty/llama.cpp
///
/// BitNet.cpp includes llama.cpp as a submodule. This function builds
/// the vendored version to produce libllama.so and libggml.so.
///
/// # Arguments
///
/// * `install_dir` - Root directory of BitNet.cpp repository
///
/// # Returns
///
/// Ok(()) if vendored llama.cpp build succeeds, Err otherwise
fn build_vendored_llama_cpp(install_dir: &Path) -> Result<()> {
    let llama_dir = install_dir.join("3rdparty").join("llama.cpp");

    if !llama_dir.exists() {
        eprintln!("[bitnet] Vendored llama.cpp not found at {}, skipping", llama_dir.display());
        return Ok(());
    }

    eprintln!("[bitnet] Building vendored llama.cpp...");

    let build_dir = llama_dir.join("build");

    // Create build directory if it doesn't exist
    if !build_dir.exists() {
        fs::create_dir_all(&build_dir).context("Failed to create llama.cpp build directory")?;
    }

    // Configure with CMake
    let configure_output = Command::new("cmake")
        .arg("..")
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg("-DBUILD_SHARED_LIBS=ON")
        .current_dir(&build_dir)
        .output()
        .context("Failed to configure vendored llama.cpp")?;

    if !configure_output.status.success() {
        bail!(
            "Vendored llama.cpp CMake configuration failed\nStderr: {}",
            String::from_utf8_lossy(&configure_output.stderr)
        );
    }

    // Build with CMake
    let build_output = Command::new("cmake")
        .arg("--build")
        .arg(".")
        .arg("--parallel")
        .current_dir(&build_dir)
        .output()
        .context("Failed to build vendored llama.cpp")?;

    if !build_output.status.success() {
        bail!(
            "Vendored llama.cpp build failed\nStderr: {}",
            String::from_utf8_lossy(&build_output.stderr)
        );
    }

    eprintln!("[bitnet] Vendored llama.cpp build succeeded");

    Ok(())
}

/// Find all library directories for llama.cpp installation
///
/// Discovers llama.cpp libraries (libllama.so and libggml.so) in priority order.
/// Returns all directories containing BOTH required libraries.
///
/// # Search Tiers
///
/// **Tier 1 (Primary CMake outputs)**:
/// - `build/bin` - Primary CMake output
/// - `build/lib` - Standard CMake library location
/// - `build` - Root build directory
///
/// **Tier 2 (Fallback)**:
/// - `lib` - Root lib directory
///
/// # Environment Variable Override
///
/// - `LLAMA_CROSSVAL_LIBDIR`: Overrides all auto-discovery (takes highest priority)
/// - `LLAMA_CPP_DIR`: Affects base install_dir for discovery
///
/// # Returns
///
/// Vector of PathBufs for directories containing BOTH libllama and libggml.
/// Empty vector if libraries not found (caller should handle error).
///
/// # Required Libraries
///
/// Both libraries must be present:
/// - `libllama.so` (or .dylib/.dll)
/// - `libggml.so` (or .dylib/.dll)
pub fn find_llama_lib_dirs(install_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut lib_dirs = vec![];
    let required_libs = &["llama", "ggml"];

    // Priority 0: Explicit override via LLAMA_CROSSVAL_LIBDIR
    if let Ok(explicit_libdir) = env::var("LLAMA_CROSSVAL_LIBDIR") {
        let explicit_path = PathBuf::from(explicit_libdir);
        if has_required_libraries(&explicit_path, required_libs) {
            return Ok(vec![explicit_path]);
        }
    }

    // Tier 1: Primary CMake outputs
    let tier1_candidates =
        [install_dir.join("build/bin"), install_dir.join("build/lib"), install_dir.join("build")];

    // Tier 2: Fallback
    let fallback = [install_dir.join("lib")];

    // Search in priority order - only add directories with BOTH required libraries
    for candidate in tier1_candidates.iter().chain(fallback.iter()) {
        if has_required_libraries(candidate, required_libs) {
            lib_dirs.push(candidate.clone());
        }
    }

    Ok(lib_dirs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

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

    #[test]
    fn test_build_vendored_llama_cpp_missing_directory() {
        // Test graceful handling when vendored llama.cpp doesn't exist
        let temp_dir = TempDir::new().unwrap();
        let install_dir = temp_dir.path();

        // Should not fail when 3rdparty/llama.cpp doesn't exist
        let result = build_vendored_llama_cpp(install_dir);
        assert!(result.is_ok(), "Should gracefully handle missing vendored llama.cpp");
    }

    #[test]
    fn test_run_cmake_build_creates_build_directory() {
        // Test that run_cmake_build creates the build directory if it doesn't exist
        let temp_dir = TempDir::new().unwrap();
        let install_dir = temp_dir.path();

        // Create a minimal CMakeLists.txt
        fs::write(
            install_dir.join("CMakeLists.txt"),
            "cmake_minimum_required(VERSION 3.18)\nproject(test)\n",
        )
        .unwrap();

        // This will fail because cmake won't find valid source, but it should create the build dir
        let build_dir = install_dir.join("build");
        assert!(!build_dir.exists(), "Build directory should not exist initially");

        // Try to run cmake build (will fail on empty project, but that's ok for this test)
        let _ = run_cmake_build(install_dir);

        // Check that build directory was created
        assert!(build_dir.exists(), "Build directory should be created by run_cmake_build");
    }

    // ============================================================================
    // Library Discovery Tests (AC8-AC13)
    // ============================================================================

    #[test]
    fn test_has_libraries_empty_dir() {
        let temp_dir = TempDir::new().unwrap();
        assert!(!has_libraries(temp_dir.path()));
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_has_libraries_with_bitnet_so() {
        let temp_dir = TempDir::new().unwrap();
        fs::write(temp_dir.path().join("libbitnet.so"), b"mock").unwrap();
        assert!(has_libraries(temp_dir.path()));
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_has_libraries_with_llama_so() {
        let temp_dir = TempDir::new().unwrap();
        fs::write(temp_dir.path().join("libllama.so"), b"mock").unwrap();
        assert!(has_libraries(temp_dir.path()));
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_has_libraries_with_ggml_so() {
        let temp_dir = TempDir::new().unwrap();
        fs::write(temp_dir.path().join("libggml.so"), b"mock").unwrap();
        assert!(has_libraries(temp_dir.path()));
    }

    #[test]
    fn test_find_bitnet_lib_dirs_empty() {
        let temp_dir = TempDir::new().unwrap();
        let result = find_bitnet_lib_dirs(temp_dir.path()).unwrap();
        assert_eq!(result.len(), 0, "Should return empty vector when no libraries found");
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_find_bitnet_lib_dirs_tier1_bitnet() {
        let temp_dir = TempDir::new().unwrap();
        let build_bin = temp_dir.path().join("build/bin");
        fs::create_dir_all(&build_bin).unwrap();
        fs::write(build_bin.join("libbitnet.so"), b"mock").unwrap();

        let result = find_bitnet_lib_dirs(temp_dir.path()).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], build_bin);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_find_bitnet_lib_dirs_tier2_llama() {
        let temp_dir = TempDir::new().unwrap();
        let llama_bin = temp_dir.path().join("build/3rdparty/llama.cpp/build/bin");
        fs::create_dir_all(&llama_bin).unwrap();
        fs::write(llama_bin.join("libllama.so"), b"mock").unwrap();

        let result = find_bitnet_lib_dirs(temp_dir.path()).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], llama_bin);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_find_bitnet_lib_dirs_both_tiers() {
        let temp_dir = TempDir::new().unwrap();
        let build_bin = temp_dir.path().join("build/bin");
        let llama_bin = temp_dir.path().join("build/3rdparty/llama.cpp/build/bin");

        fs::create_dir_all(&build_bin).unwrap();
        fs::create_dir_all(&llama_bin).unwrap();
        fs::write(build_bin.join("libbitnet.so"), b"mock").unwrap();
        fs::write(llama_bin.join("libllama.so"), b"mock").unwrap();

        let result = find_bitnet_lib_dirs(temp_dir.path()).unwrap();
        assert_eq!(result.len(), 2);
        // Tier 1 should come before Tier 2
        assert_eq!(result[0], build_bin);
        assert_eq!(result[1], llama_bin);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_find_bitnet_lib_dirs_env_override() {
        use serial_test::serial;

        #[serial(bitnet_env)]
        fn run_test() {
            let temp_dir = TempDir::new().unwrap();
            let override_dir = temp_dir.path().join("override");
            fs::create_dir_all(&override_dir).unwrap();
            fs::write(override_dir.join("libbitnet.so"), b"mock").unwrap();

            // Set BITNET_CROSSVAL_LIBDIR to override auto-discovery
            unsafe {
                env::set_var("BITNET_CROSSVAL_LIBDIR", override_dir.to_str().unwrap());
            }

            let result = find_bitnet_lib_dirs(temp_dir.path()).unwrap();
            assert_eq!(result.len(), 1);
            assert_eq!(result[0], override_dir);

            // Clean up
            unsafe {
                env::remove_var("BITNET_CROSSVAL_LIBDIR");
            }
        }

        run_test();
    }
}

//! Build-time helper functions for xtask
//!
//! This module provides reusable build-time logic extracted from build.rs
//! to enable unit testing and maintainability.
//!
//! See also: docs/specs/rpath-merging-strategy.md

use std::collections::HashSet;
use std::env;
use std::path::{Path, PathBuf};

/// C++ backend types for library discovery
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CppBackend {
    /// BitNet.cpp backend
    BitNet,
    /// llama.cpp backend
    Llama,
}

impl CppBackend {
    /// Get backend name for diagnostics
    pub fn name(&self) -> &'static str {
        match self {
            CppBackend::BitNet => "bitnet.cpp",
            CppBackend::Llama => "llama.cpp",
        }
    }

    /// Get library pattern (prefix) for this backend
    fn lib_pattern(&self) -> &[&'static str] {
        match self {
            CppBackend::BitNet => &["libbitnet"],
            CppBackend::Llama => &["libllama", "libggml"],
        }
    }

    /// Get primary search paths for this backend (Tier 1)
    fn primary_search_paths(&self, base_dir: &Path) -> Vec<PathBuf> {
        match self {
            CppBackend::BitNet => vec![
                base_dir.join("build/bin"),
                base_dir.join("build/lib"),
                base_dir.join("build/3rdparty/llama.cpp/build/bin"),
            ],
            CppBackend::Llama => {
                vec![base_dir.join("build"), base_dir.join("build/bin"), base_dir.join("build/lib")]
            }
        }
    }

    /// Get fallback search paths (Tier 2)
    fn fallback_search_paths(&self, base_dir: &Path) -> Vec<PathBuf> {
        vec![base_dir.join("build"), base_dir.join("lib")]
    }
}

/// Get platform-specific library file extensions
fn library_extensions() -> &'static [&'static str] {
    #[cfg(target_os = "linux")]
    return &["so"];

    #[cfg(target_os = "macos")]
    return &["dylib"];

    #[cfg(target_os = "windows")]
    return &["dll"];
}

/// Find libraries matching a pattern in a directory
///
/// Searches for library files matching the given pattern (e.g., "libbitnet")
/// with platform-specific extensions (.so/.dylib/.dll).
///
/// # Arguments
///
/// * `dir` - Directory to search in
/// * `pattern` - Library name prefix (e.g., "libbitnet", "libllama")
///
/// # Returns
///
/// Vector of discovered library paths (may be empty if none found)
pub fn find_libraries_in_dir(dir: &Path, pattern: &str) -> Vec<PathBuf> {
    if !dir.exists() {
        return vec![];
    }

    let mut found = Vec::new();
    let extensions = library_extensions();

    // Read directory entries
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return vec![],
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        if let Some(file_name) = path.file_name() {
            let name = file_name.to_string_lossy();

            // Check if filename matches pattern and has correct extension
            for ext in extensions {
                // Match: libbitnet.so, libbitnet.so.1, libbitnet.dylib, bitnet.dll, etc.
                if name.starts_with(pattern) && (name.contains(&format!(".{}", ext))) {
                    found.push(path.clone());
                    break;
                }
                // Windows: Match bitnet.dll (no lib prefix)
                #[cfg(target_os = "windows")]
                if pattern.starts_with("lib")
                    && name.starts_with(&pattern[3..])
                    && name.contains(&format!(".{}", ext))
                {
                    found.push(path.clone());
                    break;
                }
            }
        }
    }

    found
}

/// Verify if a library path is loadable
///
/// Performs basic validation:
/// - File exists
/// - Is a regular file (not directory/symlink)
/// - Has appropriate permissions (readable)
///
/// Note: Does not perform full dlopen() test - just filesystem checks.
pub fn verify_library_loadable(lib_path: &Path) -> bool {
    if !lib_path.exists() {
        return false;
    }

    // Must be a file (or symlink to file)
    let metadata = match std::fs::metadata(lib_path) {
        Ok(m) => m,
        Err(_) => return false,
    };

    if !metadata.is_file() {
        return false;
    }

    // Check read permission (basic heuristic)
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = metadata.permissions().mode();
        // Check if any read bit is set (owner, group, or other)
        if mode & 0o444 == 0 {
            return false;
        }
    }

    true
}

/// Discover backend libraries using three-tier search hierarchy
///
/// Search priority (AC4):
/// 1. **Tier 1**: Environment variable override (BITNET_CPP_DIR/LLAMA_CPP_DIR)
/// 2. **Tier 2**: RPATH embedded paths (build-time detection)
/// 3. **Tier 3**: Auto-discovery in build artifacts (lib/, build/lib/, etc.)
///
/// # Arguments
///
/// * `backend` - Backend type (BitNet or Llama)
///
/// # Returns
///
/// Vector of directories containing discovered libraries (deduplicated)
///
/// # Examples
///
/// ```no_run
/// use xtask::build_helpers::{discover_backend_libraries, CppBackend};
///
/// let lib_dirs = discover_backend_libraries(CppBackend::BitNet).unwrap();
/// for dir in lib_dirs {
///     println!("Found libraries in: {}", dir.display());
/// }
/// ```
pub fn discover_backend_libraries(backend: CppBackend) -> anyhow::Result<Vec<PathBuf>> {
    let mut discovered_dirs = Vec::new();

    // Tier 1: Environment variable override (highest priority)
    let env_var = match backend {
        CppBackend::BitNet => "BITNET_CPP_DIR",
        CppBackend::Llama => "LLAMA_CPP_DIR",
    };

    let base_dir = if let Ok(dir) = env::var(env_var) {
        PathBuf::from(dir)
    } else {
        // Tier 2: Default installation directory
        let home = dirs::home_dir().ok_or_else(|| anyhow::anyhow!("No home directory found"))?;
        match backend {
            CppBackend::BitNet => home.join(".cache/bitnet_cpp"),
            CppBackend::Llama => home.join(".cache/llama_cpp"),
        }
    };

    if !base_dir.exists() {
        return Ok(vec![]);
    }

    // Search primary paths (Tier 1 backend-specific)
    let primary_paths = backend.primary_search_paths(&base_dir);
    for path in primary_paths {
        if path.exists() && has_any_library(&path, backend.lib_pattern()) {
            discovered_dirs.push(path);
        }
    }

    // Search fallback paths (Tier 2) if needed
    if discovered_dirs.is_empty() {
        let fallback_paths = backend.fallback_search_paths(&base_dir);
        for path in fallback_paths {
            if path.exists() && has_any_library(&path, backend.lib_pattern()) {
                discovered_dirs.push(path);
            }
        }
    }

    // Tier 3: Environment variable RPATH overrides (granular control)
    let rpath_var = match backend {
        CppBackend::BitNet => "CROSSVAL_RPATH_BITNET",
        CppBackend::Llama => "CROSSVAL_RPATH_LLAMA",
    };

    if let Ok(rpath) = env::var(rpath_var) {
        for path_str in rpath.split(':') {
            let path = PathBuf::from(path_str);
            if path.exists() {
                discovered_dirs.push(path);
            }
        }
    }

    // Deduplicate while preserving order
    let unique_dirs = merge_paths_preserve_order(discovered_dirs);

    Ok(unique_dirs)
}

/// Check if directory contains any library matching patterns
fn has_any_library(dir: &Path, patterns: &[&str]) -> bool {
    for pattern in patterns {
        if !find_libraries_in_dir(dir, pattern).is_empty() {
            return true;
        }
    }
    false
}

/// Merge paths while preserving insertion order and deduplicating
fn merge_paths_preserve_order(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut seen = HashSet::new();
    let mut merged = Vec::new();

    for path in paths {
        // Canonicalize if possible, otherwise use as-is
        let canonical = path.canonicalize().unwrap_or_else(|_| path.clone());
        if seen.insert(canonical.clone()) {
            merged.push(canonical);
        }
    }

    merged
}

/// Merge and deduplicate library paths for RPATH embedding
///
/// This function takes a vector of library paths and:
/// 1. Canonicalizes each path (resolves symlinks, normalizes case on macOS)
/// 2. Deduplicates using HashSet while preserving insertion order
/// 3. Returns a colon-separated string suitable for `-Wl,-rpath` linker argument
///
/// # Arguments
///
/// * `paths` - Vector of path strings to merge
///
/// # Returns
///
/// A colon-separated string of canonical, deduplicated paths.
/// Returns empty string if no valid paths are provided.
///
/// # Example
///
/// ```rust
/// # use xtask::build_helpers::merge_and_deduplicate;
/// let paths = vec!["/opt/bitnet/lib", "/usr/local/lib"];
/// let result = merge_and_deduplicate(&paths);
/// // Result: "/opt/bitnet/lib:/usr/local/lib" (if paths exist)
/// ```
///
/// # Edge Cases
///
/// - Invalid paths (non-existent): Skipped with warning emitted (during build.rs execution)
/// - Duplicate paths: Deduplicated to single entry
/// - Symlinks: Resolved to canonical path for deduplication
/// - Empty input: Returns empty string
///
/// # Platform Behavior
///
/// - **Linux**: Case-sensitive path comparison
/// - **macOS**: Case-insensitive comparison via canonicalize()
/// - **Windows**: N/A (RPATH not applicable, but function works for testing)
pub fn merge_and_deduplicate(paths: &[&str]) -> String {
    const MAX_RPATH_LENGTH: usize = 4096; // Conservative limit for linker

    let mut seen = HashSet::new();
    let mut merged = Vec::new();

    for path_str in paths {
        let path = PathBuf::from(path_str);

        // Canonicalize to resolve symlinks and normalize paths
        let canonical = match path.canonicalize() {
            Ok(p) => p,
            Err(_) => {
                // In build.rs context, emit cargo:warning
                // In test context, skip silently
                #[cfg(not(test))]
                println!(
                    "cargo:warning=xtask: Failed to canonicalize path {}. Skipping.",
                    path.display()
                );
                continue; // Skip invalid paths
            }
        };

        // Deduplicate using canonical path
        if seen.insert(canonical.clone()) {
            merged.push(canonical);
        }
    }

    // Join with colon separator (POSIX RPATH syntax)
    let result = merged.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(":");

    // Sanity check: RPATH length limit
    if result.len() > MAX_RPATH_LENGTH {
        panic!(
            "Merged RPATH exceeds maximum length ({} > {}). \
             Please use BITNET_CROSSVAL_LIBDIR to specify a single directory, \
             or reduce the number of library paths.",
            result.len(),
            MAX_RPATH_LENGTH
        );
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    // =========================================================================
    // Library Discovery Tests (AC4-AC5)
    // =========================================================================

    #[test]
    fn test_library_extensions() {
        let extensions = library_extensions();

        #[cfg(target_os = "linux")]
        assert_eq!(extensions, &["so"]);

        #[cfg(target_os = "macos")]
        assert_eq!(extensions, &["dylib"]);

        #[cfg(target_os = "windows")]
        assert_eq!(extensions, &["dll"]);
    }

    #[test]
    fn test_find_libraries_in_nonexistent_dir() {
        let result = find_libraries_in_dir(Path::new("/nonexistent/path"), "libbitnet");
        assert_eq!(result.len(), 0);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_find_libraries_linux() {
        let temp_dir = TempDir::new().unwrap();

        // Create mock library files
        fs::write(temp_dir.path().join("libbitnet.so"), b"mock").unwrap();
        fs::write(temp_dir.path().join("libbitnet.so.1"), b"mock").unwrap();
        fs::write(temp_dir.path().join("libllama.so"), b"mock").unwrap();
        fs::write(temp_dir.path().join("other.txt"), b"mock").unwrap();

        let bitnet_libs = find_libraries_in_dir(temp_dir.path(), "libbitnet");
        assert_eq!(bitnet_libs.len(), 2); // libbitnet.so and libbitnet.so.1

        let llama_libs = find_libraries_in_dir(temp_dir.path(), "libllama");
        assert_eq!(llama_libs.len(), 1);

        let nonexistent = find_libraries_in_dir(temp_dir.path(), "libnonexistent");
        assert_eq!(nonexistent.len(), 0);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_find_libraries_macos() {
        let temp_dir = TempDir::new().unwrap();

        fs::write(temp_dir.path().join("libbitnet.dylib"), b"mock").unwrap();
        fs::write(temp_dir.path().join("libllama.dylib"), b"mock").unwrap();

        let bitnet_libs = find_libraries_in_dir(temp_dir.path(), "libbitnet");
        assert_eq!(bitnet_libs.len(), 1);

        let llama_libs = find_libraries_in_dir(temp_dir.path(), "libllama");
        assert_eq!(llama_libs.len(), 1);
    }

    #[test]
    #[cfg(target_os = "windows")]
    fn test_find_libraries_windows() {
        let temp_dir = TempDir::new().unwrap();

        // Windows libraries (no lib prefix)
        fs::write(temp_dir.path().join("bitnet.dll"), b"mock").unwrap();
        fs::write(temp_dir.path().join("llama.dll"), b"mock").unwrap();

        let bitnet_libs = find_libraries_in_dir(temp_dir.path(), "libbitnet");
        assert_eq!(bitnet_libs.len(), 1); // Matches bitnet.dll (lib prefix stripped)

        let llama_libs = find_libraries_in_dir(temp_dir.path(), "libllama");
        assert_eq!(llama_libs.len(), 1);
    }

    #[test]
    fn test_verify_library_loadable_valid() {
        let temp_dir = TempDir::new().unwrap();
        let lib_path = temp_dir.path().join("libtest.so");
        fs::write(&lib_path, b"mock library").unwrap();

        assert!(verify_library_loadable(&lib_path));
    }

    #[test]
    fn test_verify_library_loadable_nonexistent() {
        assert!(!verify_library_loadable(Path::new("/nonexistent/lib.so")));
    }

    #[test]
    fn test_verify_library_loadable_directory() {
        let temp_dir = TempDir::new().unwrap();
        assert!(!verify_library_loadable(temp_dir.path()));
    }

    #[test]
    #[cfg(unix)]
    fn test_verify_library_loadable_no_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let temp_dir = TempDir::new().unwrap();
        let lib_path = temp_dir.path().join("libtest.so");
        fs::write(&lib_path, b"mock").unwrap();

        // Remove all read permissions
        let mut perms = fs::metadata(&lib_path).unwrap().permissions();
        perms.set_mode(0o000);
        fs::set_permissions(&lib_path, perms).unwrap();

        assert!(!verify_library_loadable(&lib_path));
    }

    #[test]
    fn test_has_any_library() {
        let temp_dir = TempDir::new().unwrap();

        #[cfg(target_os = "linux")]
        {
            fs::write(temp_dir.path().join("libbitnet.so"), b"mock").unwrap();
            assert!(has_any_library(temp_dir.path(), &["libbitnet"]));
            assert!(!has_any_library(temp_dir.path(), &["libllama"]));
        }

        #[cfg(target_os = "macos")]
        {
            fs::write(temp_dir.path().join("libbitnet.dylib"), b"mock").unwrap();
            assert!(has_any_library(temp_dir.path(), &["libbitnet"]));
            assert!(!has_any_library(temp_dir.path(), &["libllama"]));
        }
    }

    #[test]
    fn test_merge_paths_preserve_order_empty() {
        let result = merge_paths_preserve_order(vec![]);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_merge_paths_preserve_order_dedup() {
        let temp_dir = TempDir::new().unwrap();
        let path1 = temp_dir.path().join("lib1");
        let path2 = temp_dir.path().join("lib2");

        fs::create_dir(&path1).unwrap();
        fs::create_dir(&path2).unwrap();

        let paths = vec![path1.clone(), path2.clone(), path1.clone()];
        let result = merge_paths_preserve_order(paths);

        assert_eq!(result.len(), 2);
        // Order should be preserved
        assert_eq!(result[0].file_name().unwrap(), "lib1");
        assert_eq!(result[1].file_name().unwrap(), "lib2");
    }

    #[test]
    fn test_cpp_backend_name() {
        assert_eq!(CppBackend::BitNet.name(), "bitnet.cpp");
        assert_eq!(CppBackend::Llama.name(), "llama.cpp");
    }

    #[test]
    fn test_cpp_backend_lib_pattern() {
        assert_eq!(CppBackend::BitNet.lib_pattern(), &["libbitnet"]);
        assert_eq!(CppBackend::Llama.lib_pattern(), &["libllama", "libggml"]);
    }

    #[test]
    fn test_cpp_backend_primary_search_paths() {
        let base = PathBuf::from("/test/base");

        let bitnet_paths = CppBackend::BitNet.primary_search_paths(&base);
        assert_eq!(bitnet_paths.len(), 3);
        assert_eq!(bitnet_paths[0], base.join("build/bin"));
        assert_eq!(bitnet_paths[1], base.join("build/lib"));
        assert_eq!(bitnet_paths[2], base.join("build/3rdparty/llama.cpp/build/bin"));

        let llama_paths = CppBackend::Llama.primary_search_paths(&base);
        assert_eq!(llama_paths.len(), 3);
        assert_eq!(llama_paths[0], base.join("build"));
        assert_eq!(llama_paths[1], base.join("build/bin"));
        assert_eq!(llama_paths[2], base.join("build/lib"));
    }

    #[test]
    fn test_cpp_backend_fallback_search_paths() {
        let base = PathBuf::from("/test/base");

        let bitnet_fallback = CppBackend::BitNet.fallback_search_paths(&base);
        assert_eq!(bitnet_fallback.len(), 2);
        assert_eq!(bitnet_fallback[0], base.join("build"));
        assert_eq!(bitnet_fallback[1], base.join("lib"));

        let llama_fallback = CppBackend::Llama.fallback_search_paths(&base);
        assert_eq!(llama_fallback.len(), 2);
        assert_eq!(llama_fallback[0], base.join("build"));
        assert_eq!(llama_fallback[1], base.join("lib"));
    }

    // =========================================================================
    // RPATH Merge Tests (existing)
    // =========================================================================

    #[test]
    fn test_empty_input() {
        let paths: Vec<&str> = vec![];
        let result = merge_and_deduplicate(&paths);
        assert_eq!(result, "");
    }

    #[test]
    fn test_single_valid_path() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap();

        let result = merge_and_deduplicate(&[path]);
        let expected = temp_dir.path().canonicalize().unwrap();

        assert_eq!(result, expected.display().to_string());
        assert_eq!(result.matches(':').count(), 0);
    }

    #[test]
    fn test_two_distinct_paths() {
        let temp_dir = TempDir::new().unwrap();
        let path1 = temp_dir.path().join("lib1");
        let path2 = temp_dir.path().join("lib2");

        fs::create_dir(&path1).unwrap();
        fs::create_dir(&path2).unwrap();

        let result = merge_and_deduplicate(&[path1.to_str().unwrap(), path2.to_str().unwrap()]);

        let canonical1 = path1.canonicalize().unwrap();
        let canonical2 = path2.canonicalize().unwrap();
        let expected = format!("{}:{}", canonical1.display(), canonical2.display());

        assert_eq!(result, expected);
        assert_eq!(result.matches(':').count(), 1);
    }

    #[test]
    fn test_duplicate_paths_deduplicated() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap();

        let result = merge_and_deduplicate(&[path, path]);
        let expected = temp_dir.path().canonicalize().unwrap();

        assert_eq!(result, expected.display().to_string());
        assert_eq!(result.matches(':').count(), 0);
    }

    #[test]
    #[cfg(unix)]
    fn test_symlink_canonicalization() {
        let temp_dir = TempDir::new().unwrap();
        let real_path = temp_dir.path().join("real");
        let symlink_path = temp_dir.path().join("link");

        fs::create_dir(&real_path).unwrap();
        std::os::unix::fs::symlink(&real_path, &symlink_path).unwrap();

        let result =
            merge_and_deduplicate(&[real_path.to_str().unwrap(), symlink_path.to_str().unwrap()]);

        // Both should resolve to same canonical path, so only one entry
        assert_eq!(result.matches(':').count(), 0);
        let expected = real_path.canonicalize().unwrap();
        assert_eq!(result, expected.display().to_string());
    }

    #[test]
    fn test_invalid_path_skipped() {
        let temp_dir = TempDir::new().unwrap();
        let valid_path = temp_dir.path().to_str().unwrap();
        let invalid_path = "/this/path/does/not/exist/and/never/will";

        let result = merge_and_deduplicate(&[valid_path, invalid_path]);
        let expected = temp_dir.path().canonicalize().unwrap();

        // Should only contain valid path
        assert_eq!(result, expected.display().to_string());
        assert_eq!(result.matches(':').count(), 0);
    }

    #[test]
    fn test_ordering_preserved() {
        let temp_dir = TempDir::new().unwrap();
        let path_bitnet = temp_dir.path().join("bitnet");
        let path_llama = temp_dir.path().join("llama");

        fs::create_dir(&path_bitnet).unwrap();
        fs::create_dir(&path_llama).unwrap();

        let result =
            merge_and_deduplicate(&[path_bitnet.to_str().unwrap(), path_llama.to_str().unwrap()]);

        let canonical_bitnet = path_bitnet.canonicalize().unwrap();
        let canonical_llama = path_llama.canonicalize().unwrap();

        // Verify bitnet appears before llama in the result
        assert!(result.starts_with(&canonical_bitnet.display().to_string()));
        assert!(result.contains(&format!(":{}", canonical_llama.display())));
    }
}

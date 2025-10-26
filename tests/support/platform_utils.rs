//! Platform-specific utility functions for cross-platform testing
//!
//! This module provides platform-agnostic utilities for working with shared libraries,
//! dynamic loader paths, and environment variables across Linux, macOS, and Windows.
//!
//! # Coverage
//!
//! Tests spec: docs/specs/test-infra-auto-repair-ci.md#ac4
//!
//! # Platform Support
//!
//! - **Linux**: `.so` libraries, `LD_LIBRARY_PATH`
//! - **macOS**: `.dylib` libraries, `DYLD_LIBRARY_PATH`
//! - **Windows**: `.dll` libraries, `PATH`
//!
//! # Examples
//!
//! ```rust
//! use tests::support::platform_utils::{get_loader_path_var, format_lib_name};
//!
//! // Get platform-specific loader path variable
//! let loader_var = get_loader_path_var();
//! // Linux: "LD_LIBRARY_PATH"
//! // macOS: "DYLD_LIBRARY_PATH"
//! // Windows: "PATH"
//!
//! // Format library name with platform conventions
//! let lib_name = format_lib_name("bitnet");
//! // Linux: "libbitnet.so"
//! // macOS: "libbitnet.dylib"
//! // Windows: "bitnet.dll"
//! ```

/// Returns platform-specific dynamic loader path variable name
///
/// # Returns
/// - `"LD_LIBRARY_PATH"` on Linux
/// - `"DYLD_LIBRARY_PATH"` on macOS
/// - `"PATH"` on Windows
///
/// # Panics
/// Panics on unsupported platforms with clear error message
///
/// # Examples
///
/// ```rust
/// use tests::support::platform_utils::get_loader_path_var;
///
/// let loader_var = get_loader_path_var();
///
/// #[cfg(target_os = "linux")]
/// assert_eq!(loader_var, "LD_LIBRARY_PATH");
///
/// #[cfg(target_os = "macos")]
/// assert_eq!(loader_var, "DYLD_LIBRARY_PATH");
///
/// #[cfg(target_os = "windows")]
/// assert_eq!(loader_var, "PATH");
/// ```
pub fn get_loader_path_var() -> &'static str {
    if cfg!(target_os = "linux") {
        "LD_LIBRARY_PATH"
    } else if cfg!(target_os = "macos") {
        "DYLD_LIBRARY_PATH"
    } else if cfg!(target_os = "windows") {
        "PATH"
    } else {
        panic!("Unsupported platform: {}", std::env::consts::OS)
    }
}

/// Returns platform-specific shared library file extension
///
/// # Returns
/// - `"so"` on Linux
/// - `"dylib"` on macOS
/// - `"dll"` on Windows
///
/// # Panics
/// Panics on unsupported platforms
///
/// # Examples
///
/// ```rust
/// use tests::support::platform_utils::get_lib_extension;
///
/// let ext = get_lib_extension();
///
/// #[cfg(target_os = "linux")]
/// assert_eq!(ext, "so");
///
/// #[cfg(target_os = "macos")]
/// assert_eq!(ext, "dylib");
///
/// #[cfg(target_os = "windows")]
/// assert_eq!(ext, "dll");
/// ```
pub fn get_lib_extension() -> &'static str {
    if cfg!(target_os = "linux") {
        "so"
    } else if cfg!(target_os = "macos") {
        "dylib"
    } else if cfg!(target_os = "windows") {
        "dll"
    } else {
        panic!("Unsupported platform: {}", std::env::consts::OS)
    }
}

/// Formats library name with platform-specific prefix and extension
///
/// # Arguments
/// - `stem`: Base library name (e.g., "bitnet", "llama", "ggml")
///
/// # Returns
/// - Linux: `lib{stem}.so` (e.g., `libbitnet.so`)
/// - macOS: `lib{stem}.dylib` (e.g., `libbitnet.dylib`)
/// - Windows: `{stem}.dll` (e.g., `bitnet.dll`)
///
/// # Examples
///
/// ```rust
/// use tests::support::platform_utils::format_lib_name;
///
/// let name = format_lib_name("bitnet");
///
/// #[cfg(target_os = "linux")]
/// assert_eq!(name, "libbitnet.so");
///
/// #[cfg(target_os = "macos")]
/// assert_eq!(name, "libbitnet.dylib");
///
/// #[cfg(target_os = "windows")]
/// assert_eq!(name, "bitnet.dll");
/// ```
pub fn format_lib_name(stem: &str) -> String {
    if cfg!(target_os = "windows") {
        format!("{}.dll", stem)
    } else if cfg!(target_os = "macos") {
        format!("lib{}.dylib", stem)
    } else {
        // Default to Linux/Unix convention
        format!("lib{}.so", stem)
    }
}

/// Appends path to platform-specific loader path variable
///
/// # Arguments
/// - `new_path`: Path to prepend to loader variable
///
/// # Returns
/// Formatted string for environment variable assignment
///
/// # Platform Behavior
/// - Unix (Linux/macOS): Uses `:` as separator
/// - Windows: Uses `;` as separator
/// - If current path is empty, returns `new_path` only
/// - If current path exists, prepends `new_path` with separator
///
/// # Examples
///
/// ```rust
/// use tests::support::platform_utils::append_to_loader_path;
///
/// // With existing path
/// std::env::set_var("LD_LIBRARY_PATH", "/existing/path");
/// let updated = append_to_loader_path("/opt/libs");
/// // Linux: "/opt/libs:/existing/path"
/// // macOS: "/opt/libs:/existing/path"
/// // Windows: "/opt/libs;C:\\existing\\path"
///
/// // With empty path
/// std::env::remove_var("LD_LIBRARY_PATH");
/// let updated = append_to_loader_path("/opt/libs");
/// // All platforms: "/opt/libs"
/// ```
pub fn append_to_loader_path(new_path: &str) -> String {
    let loader_var = get_loader_path_var();
    let separator = if cfg!(target_os = "windows") { ";" } else { ":" };

    let current = std::env::var(loader_var).unwrap_or_default();
    if current.is_empty() {
        new_path.to_string()
    } else {
        format!("{}{}{}", new_path, separator, current)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_loader_path_var_platform_specific() {
        let var = get_loader_path_var();

        #[cfg(target_os = "linux")]
        assert_eq!(var, "LD_LIBRARY_PATH");

        #[cfg(target_os = "macos")]
        assert_eq!(var, "DYLD_LIBRARY_PATH");

        #[cfg(target_os = "windows")]
        assert_eq!(var, "PATH");

        // Ensure non-empty result on all platforms
        assert!(!var.is_empty());
    }

    #[test]
    fn test_get_lib_extension_platform_specific() {
        let ext = get_lib_extension();

        #[cfg(target_os = "linux")]
        assert_eq!(ext, "so");

        #[cfg(target_os = "macos")]
        assert_eq!(ext, "dylib");

        #[cfg(target_os = "windows")]
        assert_eq!(ext, "dll");

        // Ensure non-empty result on all platforms
        assert!(!ext.is_empty());
    }

    #[test]
    fn test_format_lib_name_platform_specific() {
        let name = format_lib_name("bitnet");

        #[cfg(target_os = "linux")]
        assert_eq!(name, "libbitnet.so");

        #[cfg(target_os = "macos")]
        assert_eq!(name, "libbitnet.dylib");

        #[cfg(target_os = "windows")]
        assert_eq!(name, "bitnet.dll");

        // Ensure stem is preserved
        assert!(name.contains("bitnet"));
    }

    #[test]
    fn test_format_lib_name_multiple_stems() {
        let names = vec!["llama", "ggml", "bitnet"];

        for stem in names {
            let formatted = format_lib_name(stem);

            // Verify stem is in formatted name
            assert!(formatted.contains(stem));

            // Verify correct extension
            #[cfg(target_os = "linux")]
            assert!(formatted.ends_with(".so"));

            #[cfg(target_os = "macos")]
            assert!(formatted.ends_with(".dylib"));

            #[cfg(target_os = "windows")]
            assert!(formatted.ends_with(".dll"));
        }
    }

    #[test]
    #[cfg(unix)]
    fn test_append_to_loader_path_unix() {
        use crate::support::env_guard::EnvGuard;

        // Save original value
        let loader_var = get_loader_path_var();
        let _guard = EnvGuard::new(loader_var);

        // Test with empty path
        unsafe {
            std::env::remove_var(loader_var);
        }
        let updated = append_to_loader_path("/new/path");
        assert_eq!(updated, "/new/path");

        // Test with existing path
        unsafe {
            std::env::set_var(loader_var, "/existing/path");
        }
        let updated = append_to_loader_path("/new/path");
        assert_eq!(updated, "/new/path:/existing/path");
    }

    #[test]
    #[cfg(target_os = "windows")]
    fn test_append_to_loader_path_windows() {
        use crate::support::env_guard::EnvGuard;

        // Save original value
        let loader_var = get_loader_path_var();
        let _guard = EnvGuard::new(loader_var);

        // Test with empty path
        unsafe {
            std::env::remove_var(loader_var);
        }
        let updated = append_to_loader_path("C:\\new\\path");
        assert_eq!(updated, "C:\\new\\path");

        // Test with existing path
        unsafe {
            std::env::set_var(loader_var, "C:\\existing\\path");
        }
        let updated = append_to_loader_path("C:\\new\\path");
        assert_eq!(updated, "C:\\new\\path;C:\\existing\\path");
    }
}

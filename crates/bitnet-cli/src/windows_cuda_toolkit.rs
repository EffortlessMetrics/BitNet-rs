//! Windows CUDA Toolkit DLL discovery for strict CUDA CLI paths.
//!
//! The rest of the CLI should not need to know how PATH is searched or mutated;
//! it asks this module to make CUDA runtime DLLs visible before CUDA/NVRTC load.

#[cfg(all(feature = "cuda", target_os = "windows"))]
use anyhow::{Context, Result};

#[cfg(all(feature = "cuda", target_os = "windows"))]
pub(crate) fn ensure_windows_cuda_toolkit_bin_on_path() -> Result<Option<std::path::PathBuf>> {
    if windows_cuda_runtime_libraries_visible_on_path() {
        return Ok(None);
    }

    let Some(cuda_bin) = discover_windows_cuda_toolkit_bin() else {
        return Ok(None);
    };
    prepend_process_path(&cuda_bin).with_context(|| {
        format!("failed to add CUDA Toolkit bin to PATH: {}", cuda_bin.display())
    })?;
    Ok(Some(cuda_bin))
}

#[cfg(all(feature = "cuda", target_os = "windows"))]
fn discover_windows_cuda_toolkit_bin() -> Option<std::path::PathBuf> {
    discover_cuda_toolkit_bin_from_roots(windows_cuda_toolkit_search_roots())
}

#[cfg(any(test, all(feature = "cuda", target_os = "windows")))]
pub(crate) fn discover_cuda_toolkit_bin_from_roots<I, P>(roots: I) -> Option<std::path::PathBuf>
where
    I: IntoIterator<Item = P>,
    P: AsRef<std::path::Path>,
{
    let mut candidates = Vec::new();
    for root in roots {
        collect_cuda_toolkit_bin_candidates(root.as_ref(), &mut candidates);
    }
    candidates.sort_by(|left, right| {
        cuda_bin_version_key(right).cmp(&cuda_bin_version_key(left)).then_with(|| left.cmp(right))
    });
    candidates.into_iter().find(|candidate| cuda_toolkit_bin_has_runtime_libraries(candidate))
}

#[cfg(any(test, all(feature = "cuda", target_os = "windows")))]
fn collect_cuda_toolkit_bin_candidates(
    root: &std::path::Path,
    candidates: &mut Vec<std::path::PathBuf>,
) {
    candidates.push(root.to_path_buf());
    candidates.push(root.join("bin"));

    let Ok(children) = std::fs::read_dir(root) else {
        return;
    };
    for child in children.flatten() {
        let path = child.path();
        if path.is_dir()
            && path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with('v'))
        {
            candidates.push(path.join("bin"));
        }
    }
}

#[cfg(any(test, all(feature = "cuda", target_os = "windows")))]
fn cuda_toolkit_bin_has_runtime_libraries(bin: &std::path::Path) -> bool {
    cuda_toolkit_bin_has_any(bin, WINDOWS_NVRTC_LIBRARY_NAMES)
        && cuda_toolkit_bin_has_any(bin, WINDOWS_CUDART_LIBRARY_NAMES)
}

#[cfg(any(test, all(feature = "cuda", target_os = "windows")))]
fn cuda_toolkit_bin_has_any(bin: &std::path::Path, names: &[&str]) -> bool {
    names.iter().any(|name| bin.join(name).is_file())
}

#[cfg(all(feature = "cuda", target_os = "windows"))]
fn windows_cuda_runtime_libraries_visible_on_path() -> bool {
    let Some(path) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&path).any(|entry| cuda_toolkit_bin_has_runtime_libraries(&entry))
}

#[cfg(all(feature = "cuda", target_os = "windows"))]
fn windows_cuda_toolkit_search_roots() -> Vec<std::path::PathBuf> {
    let mut roots = Vec::new();
    for (key, value) in std::env::vars_os() {
        if key.to_string_lossy().to_ascii_uppercase().starts_with("CUDA_PATH") && !value.is_empty()
        {
            roots.push(std::path::PathBuf::from(value));
        }
    }

    for key in ["ProgramW6432", "ProgramFiles"] {
        if let Some(program_files) = std::env::var_os(key) {
            roots.push(
                std::path::PathBuf::from(program_files)
                    .join("NVIDIA GPU Computing Toolkit")
                    .join("CUDA"),
            );
        }
    }
    roots.push(std::path::PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"));

    dedupe_paths(roots)
}

#[cfg(all(feature = "cuda", target_os = "windows"))]
fn dedupe_paths(paths: Vec<std::path::PathBuf>) -> Vec<std::path::PathBuf> {
    let mut deduped = Vec::<std::path::PathBuf>::new();
    for path in paths {
        if !deduped.iter().any(|existing| paths_equal_for_process_path(existing, &path)) {
            deduped.push(path);
        }
    }
    deduped
}

#[cfg(all(feature = "cuda", target_os = "windows"))]
fn prepend_process_path(path: &std::path::Path) -> Result<()> {
    let current = std::env::var_os("PATH").unwrap_or_default();
    let mut entries = Vec::from([path.to_path_buf()]);
    entries.extend(
        std::env::split_paths(&current).filter(|entry| !paths_equal_for_process_path(entry, path)),
    );
    let updated_path = std::env::join_paths(entries)?;
    // SAFETY: CLI CUDA entry points call this before CUDA/NVRTC loading
    // starts, so cudarc can discover Toolkit DLLs installed in the standard
    // Windows location. The CLI does not read PATH concurrently in this block.
    unsafe {
        std::env::set_var("PATH", updated_path);
    }
    Ok(())
}

#[cfg(all(feature = "cuda", target_os = "windows"))]
fn paths_equal_for_process_path(left: &std::path::Path, right: &std::path::Path) -> bool {
    left.to_string_lossy().eq_ignore_ascii_case(&right.to_string_lossy())
}

#[cfg(any(test, all(feature = "cuda", target_os = "windows")))]
fn cuda_bin_version_key(path: &std::path::Path) -> (u32, u32, u32) {
    let version_name =
        path.parent().and_then(|parent| parent.file_name()).and_then(|name| name.to_str());
    parse_cuda_version_name(version_name.unwrap_or_default())
}

#[cfg(any(test, all(feature = "cuda", target_os = "windows")))]
fn parse_cuda_version_name(name: &str) -> (u32, u32, u32) {
    let Some(rest) = name.strip_prefix('v') else {
        return (0, 0, 0);
    };
    let mut parts = rest.split('.');
    let major = parts.next().and_then(|value| value.parse().ok()).unwrap_or_default();
    let minor = parts.next().and_then(|value| value.parse().ok()).unwrap_or_default();
    let patch = parts.next().and_then(|value| value.parse().ok()).unwrap_or_default();
    (major, minor, patch)
}

#[cfg(any(test, all(feature = "cuda", target_os = "windows")))]
const WINDOWS_NVRTC_LIBRARY_NAMES: &[&str] =
    &["nvrtc64_120_0.dll", "nvrtc64_120.dll", "nvrtc64_12.dll", "nvrtc64.dll", "nvrtc.dll"];

#[cfg(any(test, all(feature = "cuda", target_os = "windows")))]
const WINDOWS_CUDART_LIBRARY_NAMES: &[&str] =
    &["cudart64_120.dll", "cudart64_12.dll", "cudart64.dll", "cudart.dll"];

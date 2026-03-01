use std::io;
use std::path::{Path, PathBuf};

/// Resolve a fixture path relative to a crate manifest directory.
pub fn fixture_path_from_manifest(
    manifest_dir: impl AsRef<Path>,
    relative_path: impl AsRef<Path>,
) -> PathBuf {
    manifest_dir.as_ref().join(relative_path)
}

/// Resolve a fixture path relative to the workspace root.
///
/// `levels_up` is how many parent traversals are needed to go from `manifest_dir`
/// to workspace root (e.g. `2` for `crates/<name>`).
pub fn fixture_path_from_workspace(
    manifest_dir: impl AsRef<Path>,
    levels_up: usize,
    relative_path: impl AsRef<Path>,
) -> PathBuf {
    let root = workspace_root(manifest_dir, levels_up);
    root.join(relative_path)
}

/// Load fixture bytes from a path.
pub fn load_fixture_bytes(path: impl AsRef<Path>) -> io::Result<Vec<u8>> {
    std::fs::read(path)
}

/// Load fixture text from a path.
pub fn load_fixture_string(path: impl AsRef<Path>) -> io::Result<String> {
    std::fs::read_to_string(path)
}

fn workspace_root(manifest_dir: impl AsRef<Path>, levels_up: usize) -> PathBuf {
    let mut current = manifest_dir.as_ref();
    for _ in 0..levels_up {
        current = current.parent().expect("Failed to determine workspace root from manifest dir");
    }
    current.to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_manifest_relative_paths() {
        let manifest = Path::new("/tmp/project/crates/demo");
        let path = fixture_path_from_manifest(manifest, "tests/fixtures/a.bin");
        assert_eq!(path, PathBuf::from("/tmp/project/crates/demo/tests/fixtures/a.bin"));
    }

    #[test]
    fn resolves_workspace_relative_paths() {
        let manifest = Path::new("/tmp/project/crates/demo");
        let path = fixture_path_from_workspace(manifest, 2, "ci/fixtures/a.bin");
        assert_eq!(path, PathBuf::from("/tmp/project/ci/fixtures/a.bin"));
    }
}

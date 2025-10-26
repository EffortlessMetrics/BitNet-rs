/// Directory Layout Builder for Integration Test Fixtures
///
/// Creates temporary directory structures matching real BitNet.cpp/llama.cpp layouts
/// for integration testing without requiring actual C++ installations.
///
/// # Test Specification
///
/// Tests feature spec: bitnet-integration-tests.md#fixture-requirements
use super::{FixtureError, LibraryType, MockLibrary, Platform};
use std::path::{Path, PathBuf};
use tempfile::TempDir;

/// Directory layout variants for different test scenarios
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutType {
    /// Standard BitNet.cpp CMake build layout
    /// - include/ggml-bitnet.h
    /// - 3rdparty/llama.cpp/include/llama.h
    /// - build/lib/{libbitnet,libllama,libggml}.{so,dylib,lib}
    BitNetStandard,

    /// Standalone llama.cpp layout (no BitNet)
    /// - include/llama.h
    /// - build/bin/{libllama,libggml}.{so,dylib,lib}
    LlamaStandalone,

    /// Custom library directory override (BITNET_CROSSVAL_LIBDIR)
    /// - Custom path with libraries only (no standard structure)
    CustomLibDir,

    /// Dual backend (BitNet embedding llama)
    /// - BitNet standard structure
    /// - llama.cpp embedded in 3rdparty/
    DualBackend,

    /// Missing libraries (headers only, graceful failure test)
    /// - include/ggml-bitnet.h (headers present)
    /// - No build/ directory (libraries missing)
    MissingLibs,
}

/// Builder for creating directory layout fixtures
pub struct DirectoryLayoutBuilder {
    layout_type: LayoutType,
    include_libs: bool,
    include_headers: bool,
    #[allow(dead_code)]
    custom_paths: Vec<PathBuf>,
    platform: Platform,
}

impl DirectoryLayoutBuilder {
    /// Create a new directory layout builder
    pub fn new(layout_type: LayoutType) -> Self {
        Self {
            layout_type,
            include_libs: true,
            include_headers: true,
            custom_paths: Vec::new(),
            platform: Self::detect_platform(),
        }
    }

    /// Configure whether to include library files
    pub fn with_libs(mut self, enabled: bool) -> Self {
        self.include_libs = enabled;
        self
    }

    /// Configure whether to include header files
    pub fn with_headers(mut self, enabled: bool) -> Self {
        self.include_headers = enabled;
        self
    }

    /// Add custom library path (for CustomLibDir layout)
    #[allow(dead_code)]
    pub fn add_custom_path(mut self, path: PathBuf) -> Self {
        self.custom_paths.push(path);
        self
    }

    /// Build the directory layout fixture
    ///
    /// # Returns
    ///
    /// - `Ok(DirectoryLayout)`: Successfully created fixture
    /// - `Err(FixtureError)`: Failed to create directories or generate libraries
    ///
    /// # Test Coverage
    ///
    /// Tests feature spec: bitnet-integration-tests.md#scenario-1-5
    pub fn build(self) -> Result<DirectoryLayout, FixtureError> {
        let temp_dir = TempDir::new()?;
        let root = temp_dir.path().to_path_buf();

        match self.layout_type {
            LayoutType::BitNetStandard => self.create_bitnet_standard(&root)?,
            LayoutType::LlamaStandalone => self.create_llama_standalone(&root)?,
            LayoutType::CustomLibDir => self.create_custom_libdir(&root)?,
            LayoutType::DualBackend => self.create_dual_backend(&root)?,
            LayoutType::MissingLibs => self.create_missing_libs(&root)?,
        }

        Ok(DirectoryLayout { root, layout_type: self.layout_type, _temp_dir: temp_dir })
    }

    /// Detect the current platform for library generation
    fn detect_platform() -> Platform {
        #[cfg(target_os = "linux")]
        return Platform::Linux;

        #[cfg(target_os = "macos")]
        return Platform::MacOS;

        #[cfg(target_os = "windows")]
        return Platform::Windows;

        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        Platform::Linux // Default fallback
    }

    /// Create BitNet.cpp standard CMake layout
    ///
    /// Tests feature spec: bitnet-integration-tests.md#scenario-1-bitnet-standard-layout
    fn create_bitnet_standard(&self, root: &Path) -> Result<(), FixtureError> {
        // Create directory structure
        let include_dir = root.join("include");
        let llama_include = root.join("3rdparty/llama.cpp/include");
        let build_lib = root.join("build/lib");
        let build_llama_src = root.join("build/3rdparty/llama.cpp/src");
        let build_ggml_src = root.join("build/3rdparty/llama.cpp/ggml/src");

        std::fs::create_dir_all(&include_dir)?;
        std::fs::create_dir_all(&llama_include)?;
        std::fs::create_dir_all(&build_lib)?;
        std::fs::create_dir_all(&build_llama_src)?;
        std::fs::create_dir_all(&build_ggml_src)?;

        // Create header files
        if self.include_headers {
            std::fs::write(include_dir.join("ggml-bitnet.h"), "// Mock BitNet header\n")?;
            std::fs::write(llama_include.join("llama.h"), "// Mock llama header\n")?;
        }

        // Create library files
        if self.include_libs {
            MockLibrary::new(LibraryType::BitNet, self.platform)
                .at_path(build_lib.clone())
                .generate()?;

            MockLibrary::new(LibraryType::Llama, self.platform)
                .at_path(build_lib.clone())
                .generate()?;

            MockLibrary::new(LibraryType::Ggml, self.platform)
                .at_path(build_lib.clone())
                .generate()?;

            // Also place llama/ggml in 3rdparty paths
            MockLibrary::new(LibraryType::Llama, self.platform)
                .at_path(build_llama_src.clone())
                .generate()?;

            MockLibrary::new(LibraryType::Ggml, self.platform)
                .at_path(build_ggml_src.clone())
                .generate()?;
        }

        Ok(())
    }

    /// Create standalone llama.cpp layout
    ///
    /// Tests feature spec: bitnet-integration-tests.md#scenario-3-standalone-llama
    fn create_llama_standalone(&self, root: &Path) -> Result<(), FixtureError> {
        let include_dir = root.join("include");
        let build_bin = root.join("build/bin");

        std::fs::create_dir_all(&include_dir)?;
        std::fs::create_dir_all(&build_bin)?;

        if self.include_headers {
            std::fs::write(include_dir.join("llama.h"), "// Mock llama header\n")?;
        }

        if self.include_libs {
            MockLibrary::new(LibraryType::Llama, self.platform)
                .at_path(build_bin.clone())
                .generate()?;

            MockLibrary::new(LibraryType::Ggml, self.platform)
                .at_path(build_bin.clone())
                .generate()?;
        }

        Ok(())
    }

    /// Create custom library directory layout
    ///
    /// Tests feature spec: bitnet-integration-tests.md#scenario-2-custom-libdir
    fn create_custom_libdir(&self, root: &Path) -> Result<(), FixtureError> {
        let lib_dir = root.join("custom_libs");
        std::fs::create_dir_all(&lib_dir)?;

        if self.include_libs {
            MockLibrary::new(LibraryType::BitNet, self.platform)
                .at_path(lib_dir.clone())
                .generate()?;

            MockLibrary::new(LibraryType::Llama, self.platform)
                .at_path(lib_dir.clone())
                .generate()?;

            MockLibrary::new(LibraryType::Ggml, self.platform)
                .at_path(lib_dir.clone())
                .generate()?;
        }

        // Minimal headers in separate location
        if self.include_headers {
            let include_dir = root.join("include");
            std::fs::create_dir_all(&include_dir)?;
            std::fs::write(include_dir.join("ggml-bitnet.h"), "// Mock BitNet header\n")?;
        }

        Ok(())
    }

    /// Create dual backend layout (BitNet embedding llama)
    ///
    /// Tests feature spec: bitnet-integration-tests.md#scenario-4-dual-backend
    fn create_dual_backend(&self, root: &Path) -> Result<(), FixtureError> {
        // Essentially same as BitNetStandard, but explicit for dual backend testing
        self.create_bitnet_standard(root)
    }

    /// Create missing libraries layout (graceful failure test)
    ///
    /// Tests feature spec: bitnet-integration-tests.md#scenario-5-missing-libs
    fn create_missing_libs(&self, root: &Path) -> Result<(), FixtureError> {
        // Headers only, no build directory
        let include_dir = root.join("include");
        std::fs::create_dir_all(&include_dir)?;

        if self.include_headers {
            std::fs::write(include_dir.join("ggml-bitnet.h"), "// Mock BitNet header\n")?;
        }

        // Explicitly do NOT create build/ directory or libraries

        Ok(())
    }
}

/// A generated directory layout fixture
pub struct DirectoryLayout {
    root: PathBuf,
    layout_type: LayoutType,
    _temp_dir: TempDir, // Kept to prevent premature cleanup
}

impl DirectoryLayout {
    /// Get the root path of this layout
    pub fn root_path(&self) -> &Path {
        &self.root
    }

    /// Get the layout type
    pub fn layout_type(&self) -> LayoutType {
        self.layout_type
    }

    /// Get all library paths in this layout
    ///
    /// Returns paths where libraries are located, based on layout type.
    pub fn lib_paths(&self) -> Vec<PathBuf> {
        match self.layout_type {
            LayoutType::BitNetStandard | LayoutType::DualBackend => {
                vec![
                    self.root.join("build/lib"),
                    self.root.join("build/3rdparty/llama.cpp/src"),
                    self.root.join("build/3rdparty/llama.cpp/ggml/src"),
                ]
            }
            LayoutType::LlamaStandalone => {
                vec![self.root.join("build/bin")]
            }
            LayoutType::CustomLibDir => {
                vec![self.root.join("custom_libs")]
            }
            LayoutType::MissingLibs => {
                vec![] // No library paths
            }
        }
    }

    /// Get all header paths in this layout
    pub fn header_paths(&self) -> Vec<PathBuf> {
        match self.layout_type {
            LayoutType::BitNetStandard | LayoutType::DualBackend => {
                vec![self.root.join("include"), self.root.join("3rdparty/llama.cpp/include")]
            }
            LayoutType::LlamaStandalone => {
                vec![self.root.join("include")]
            }
            LayoutType::CustomLibDir | LayoutType::MissingLibs => {
                vec![self.root.join("include")]
            }
        }
    }

    /// Check if a specific library exists in this layout
    pub fn has_library(&self, lib_type: LibraryType) -> bool {
        let platform = DirectoryLayoutBuilder::detect_platform();
        let lib_name = MockLibrary::library_filename(lib_type, platform);

        self.lib_paths().iter().any(|path| path.join(&lib_name).exists())
    }

    /// Check if a specific header exists in this layout
    pub fn has_header(&self, header_name: &str) -> bool {
        self.header_paths().iter().any(|path| path.join(header_name).exists())
    }
}
